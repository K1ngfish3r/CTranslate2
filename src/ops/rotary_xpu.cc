#include "ctranslate2/ops/rotary.h"

#ifdef CT2_WITH_XPU

#include <sycl/sycl.hpp>

#include "ctranslate2/bfloat16.h"
#include "xpu/utils.h"

namespace ctranslate2 {
  namespace ops {

    namespace {

      // Type promotion matching CUDA's ComputeType pattern.
      // float16_t and bfloat16_t compute in float, then cast back.
      template <typename T>
      struct ComputeType {
        using type = T;
      };
      template <>
      struct ComputeType<float16_t> {
        using type = float;
      };
      template <>
      struct ComputeType<bfloat16_t> {
        using type = float;
      };

      constexpr dim_t kMaxThreads = 1024;

      template <typename T, bool interleave>
      void rotary_kernel_xpu(sycl::queue& queue,
                             const T* x,
                             const T* sin,
                             const T* cos,
                             T* y,
                             dim_t max_time,
                             dim_t head_size,
                             dim_t ndims,
                             dim_t depth,
                             bool transpose,
                             dim_t num_rows) {
        if (num_rows == 0) return;

        using C = typename ComputeType<T>::type;
        const dim_t middle = ndims / 2;
        const size_t local_size =
            static_cast<size_t>(std::min(depth, kMaxThreads));
        const size_t num_groups = static_cast<size_t>(num_rows);

        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::nd_range<1>(
                  sycl::range<1>(num_groups * local_size),
                  sycl::range<1>(local_size)),
              [=](sycl::nd_item<1> item) {
                const size_t row = item.get_group_linear_id();
                const size_t tid = item.get_local_linear_id();

                // Decode the time position from the flat row index.
                // Sin/cos depends only on the position within the sequence,
                // not on batch or head, so we always reduce modulo max_time:
                //   transpose (B, H, T, D):  time = row % max_time
                //   default   (B, T, H, D):  time = (row / head_size) % max_time
                const size_t mt = static_cast<size_t>(max_time);
                const size_t time = transpose
                    ? row % mt
                    : (row / static_cast<size_t>(head_size)) % mt;

                const T* row_x = x + row * static_cast<size_t>(depth);
                T* row_y = y + row * static_cast<size_t>(depth);
                const T* row_sin = sin + time * static_cast<size_t>(ndims);
                const T* row_cos = cos + time * static_cast<size_t>(ndims);

                for (dim_t j = static_cast<dim_t>(tid);
                     j < depth;
                     j += static_cast<dim_t>(local_size)) {
                  if (j >= ndims) {
                    row_y[j] = row_x[j];
                  } else if (interleave) {
                    if (j % 2 == 0)
                      row_y[j] = static_cast<T>(
                          static_cast<C>(row_x[j])
                              * static_cast<C>(row_cos[j])
                          + (-static_cast<C>(row_x[j + 1]))
                                * static_cast<C>(row_sin[j]));
                    else
                      row_y[j] = static_cast<T>(
                          static_cast<C>(row_x[j])
                              * static_cast<C>(row_cos[j])
                          + static_cast<C>(row_x[j - 1])
                                * static_cast<C>(row_sin[j]));
                  } else {
                    if (j < middle)
                      row_y[j] = static_cast<T>(
                          static_cast<C>(row_x[j])
                              * static_cast<C>(row_cos[j])
                          + (-static_cast<C>(row_x[j + middle]))
                                * static_cast<C>(row_sin[j]));
                    else
                      row_y[j] = static_cast<T>(
                          static_cast<C>(row_x[j])
                              * static_cast<C>(row_cos[j])
                          + static_cast<C>(row_x[j - middle])
                                * static_cast<C>(row_sin[j]));
                  }
                }
              });
        }).wait();
      }

    }  // namespace

    template <typename T>
    static void compute_impl(
        const StorageView& input,
        const StorageView& sin,
        const StorageView& cos,
        StorageView& output,
        bool is_transposed,
        bool interleave,
        dim_t ndims_param) {

      auto queue = xpu::get_sycl_queue();

      const dim_t max_time = is_transposed ? input.dim(-2) : input.dim(-3);
      const dim_t head_size = is_transposed ? input.dim(-3) : input.dim(-2);
      const dim_t depth = input.dim(-1);
      const dim_t ndims = ndims_param == 0 ? depth : ndims_param;
      const dim_t num_rows = input.size() / depth;

      const auto* x = input.data<T>();
      const auto* s = sin.data<T>();
      const auto* c = cos.data<T>();
      auto* y = output.data<T>();

      if (interleave)
        rotary_kernel_xpu<T, true>(queue, x, s, c, y,
                                       max_time, head_size, ndims, depth,
                                       is_transposed, num_rows);
      else
        rotary_kernel_xpu<T, false>(queue, x, s, c, y,
                                        max_time, head_size, ndims, depth,
                                        is_transposed, num_rows);
    }

    template <Device D, typename T>
    void Rotary::compute(
        const StorageView& input,
        const StorageView& sin,
        const StorageView& cos,
        StorageView& output,
        bool is_transposed) const {
      compute_impl<T>(input, sin, cos, output, is_transposed, _interleave, _ndims);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Rotary::compute<Device::XPU, T>(const StorageView&,                 \
                                    const StorageView&,                 \
                                    const StorageView&,                 \
                                    StorageView&,                       \
                                    bool) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}

#endif
