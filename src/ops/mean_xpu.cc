#include "ctranslate2/ops/mean.h"

#ifdef CT2_WITH_XPU

#include <sycl/sycl.hpp>
#include <dnnl_sycl.hpp>

#include "ctranslate2/bfloat16.h"
#include "xpu/utils.h"

namespace ctranslate2 {
  namespace ops {

    namespace {

      // Type promotion: float16_t and bfloat16_t accumulate in float
      // to avoid precision loss during the cooperative reduction.
      template <typename T>
      struct MeanComputeType {
        using type = T;
      };
      template <>
      struct MeanComputeType<float16_t> {
        using type = float;
      };
      template <>
      struct MeanComputeType<bfloat16_t> {
        using type = float;
      };

      constexpr dim_t kMeanWorkGroupSize = 256;
      constexpr dim_t kMeanMaxGroups = std::numeric_limits<int32_t>::max();

      template <typename T>
      void mean_kernel_xpu(sycl::queue& queue,
                           const T* src,
                           T* dst,
                           dim_t outer,
                           dim_t axis,
                           dim_t inner,
                           bool divide,
                           float inv_axis) {
        using C = typename MeanComputeType<T>::type;

        const dim_t total_groups = std::min(outer * inner, kMeanMaxGroups);
        if (total_groups == 0)
          return;

        const size_t num_groups = static_cast<size_t>(total_groups);
        const size_t local_size = static_cast<size_t>(kMeanWorkGroupSize);

        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::nd_range<1>(
                  sycl::range<1>(num_groups * local_size),
                  sycl::range<1>(local_size)),
              [=](sycl::nd_item<1> item) {
                const dim_t flat_idx = static_cast<dim_t>(item.get_group_linear_id());
                const dim_t i = flat_idx / inner;
                const dim_t j = flat_idx % inner;

                C thread_sum = C(0);
                for (dim_t k = static_cast<dim_t>(item.get_local_linear_id());
                     k < axis;
                     k += static_cast<dim_t>(local_size)) {
                  thread_sum += static_cast<C>(src[i * axis * inner + k * inner + j]);
                }

                C group_sum = sycl::reduce_over_group(item.get_group(),
                                                      thread_sum,
                                                      sycl::plus<C>());

                if (item.get_local_linear_id() == 0) {
                  dst[i * inner + j] = static_cast<T>(
                      divide ? (group_sum * static_cast<C>(inv_axis)) : group_sum);
                }
              });
        }).wait();
      }

    }  // namespace

    template <Device D, typename T>
    void Mean::compute(const StorageView& input,
                       const dim_t outer_size,
                       const dim_t axis_size,
                       const dim_t inner_size,
                       const bool get_sum,
                       StorageView& output) const {

      if (output.size() == 0)
        return;

      auto queue = xpu::get_sycl_queue();

      mean_kernel_xpu<T>(
          queue,
          input.data<T>(),
          output.data<T>(),
          outer_size,
          axis_size,
          inner_size,
          !get_sum,
          1.0f / static_cast<float>(axis_size));
    }

#define DECLARE_IMPL_XPU(T)                                         \
    template void                                                          \
    Mean::compute<Device::XPU, T>(                                   \
        const StorageView& input,                                          \
        const dim_t outer_size,                                            \
        const dim_t axis_size,                                             \
        const dim_t inner_size,                                            \
        const bool get_sum,                                                \
        StorageView& output) const;

    DECLARE_IMPL_XPU(float)
    DECLARE_IMPL_XPU(float16_t)
    DECLARE_IMPL_XPU(bfloat16_t)

  }
}

#endif
