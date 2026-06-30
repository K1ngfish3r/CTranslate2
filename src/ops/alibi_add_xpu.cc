#include "ctranslate2/ops/alibi_add.h"

#ifdef CT2_WITH_XPU

#include <sycl/sycl.hpp>

#include "ctranslate2/bfloat16.h"
#include "xpu/utils.h"

namespace ctranslate2 {
  namespace ops {

    namespace {

      // Type promotion matching CUDA's plus<T> for half types.
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

      template <typename T>
      void alibi_add_kernel_xpu(sycl::queue& queue,
                                const T* input_data,
                                const T* alibi_data,
                                T* output_data,
                                dim_t alibi_offset,
                                dim_t num_heads,
                                dim_t query_length,
                                dim_t key_length,
                                dim_t cached_key_length,
                                dim_t total_queries) {
        if (total_queries == 0) return;

        using C = typename ComputeType<T>::type;
        const size_t local_size = static_cast<size_t>(std::min(key_length, kMaxThreads));
        const size_t num_groups = static_cast<size_t>(total_queries);

        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::nd_range<1>(
                  sycl::range<1>(num_groups * local_size),
                  sycl::range<1>(local_size)),
              [=](sycl::nd_item<1> item) {
                const size_t i = item.get_group_linear_id();
                const size_t h = (i / static_cast<size_t>(query_length))
                                   % static_cast<size_t>(num_heads);

                const T* row_input = input_data + i * static_cast<size_t>(key_length);
                T* row_output = output_data + i * static_cast<size_t>(key_length);
                const T* row_alibi = alibi_data
                                     + h * static_cast<size_t>(cached_key_length)
                                     + static_cast<size_t>(alibi_offset);

                for (size_t j = item.get_local_linear_id();
                     j < static_cast<size_t>(key_length);
                     j += local_size) {
                  row_output[j] = static_cast<T>(
                      static_cast<C>(row_input[j]) + static_cast<C>(row_alibi[j]));
                }
              });
        }).wait();
      }

    }  // namespace

    template <Device D, typename T>
    void AlibiAdd::compute(
        const StorageView& input,
        const StorageView& alibi,
        const dim_t alibi_offset,
        StorageView& output) const {

      const dim_t batch_size = input.dim(0);
      const dim_t num_heads = input.dim(1);
      const dim_t query_length = input.dim(2);
      const dim_t key_length = input.dim(3);
      const dim_t cached_key_length = alibi.dim(-1);
      const dim_t total_queries = batch_size * num_heads * query_length;

      auto queue = xpu::get_sycl_queue();

      alibi_add_kernel_xpu<T>(
          queue,
          input.data<T>(),
          alibi.data<T>(),
          output.data<T>(),
          alibi_offset,
          num_heads,
          query_length,
          key_length,
          cached_key_length,
          total_queries);
    }

#define DECLARE_IMPL_XPU(T)                                         \
    template void                                                   \
    AlibiAdd::compute<Device::XPU, T>(                              \
        const StorageView& input, const StorageView& alibi, const dim_t alibi_offset, \
        StorageView& output) const;

    DECLARE_IMPL_XPU(float)
    DECLARE_IMPL_XPU(float16_t)
    DECLARE_IMPL_XPU(bfloat16_t)

  }
}

#endif
