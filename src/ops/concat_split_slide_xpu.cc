#include "ctranslate2/ops/concat.h"
#include "ctranslate2/ops/split.h"
#include "ctranslate2/ops/slide.h"

#ifdef CT2_WITH_XPU

#include <cstdint>
#include <sycl/sycl.hpp>

#include "xpu/utils.h"

namespace ctranslate2 {
  namespace ops {

    namespace {

      // 128-bit (16-byte) packed type for vectorized memory access.
      // Matches CUDA's uint4 layout, enabling wider load/store transactions.
      struct alignas(16) uint4 {
        uint32_t x, y, z, w;
      };

      // Scatter input elements to output with a depth offset.
      //
      // This maps each linear index i to the output position:
      //   row = i / input_dim
      //   col = i % input_dim
      //   output[row * output_dim + col + offset] = input[i]
      //
      // Used when stride(axis) == 1 (contiguous along the concat dimension).
      template <typename T>
      void scatter_depth_offset(sycl::queue& queue,
                                const T* input_data,
                                T* output_data,
                                dim_t input_dim,
                                dim_t output_dim,
                                dim_t offset,
                                dim_t size) {
        if (size == 0) return;
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(sycl::range<1>(static_cast<size_t>(size)), [=](sycl::id<1> idx) {
            const size_t i = idx[0];
            const size_t row = i / static_cast<size_t>(input_dim);
            const size_t col = i % static_cast<size_t>(input_dim);
            output_data[row * static_cast<size_t>(output_dim)
                        + col + static_cast<size_t>(offset)] = input_data[i];
          });
        }).wait();
      }

      // Scatter input elements to output with an inner dimension offset.
      //
      // This maps each linear index i to the output position:
      //   i0 = i / (input_dim * inner_size)
      //   i1 = (i / inner_size) % input_dim
      //   i2 = i % inner_size
      //   output[i0 * (output_dim * inner_size) + (i1 + offset) * inner_size + i2] = input[i]
      //
      // Used when stride(axis) > 1.
      template <typename T>
      void scatter_inner_dim_offset(sycl::queue& queue,
                                    const T* input_data,
                                    T* output_data,
                                    dim_t input_dim,
                                    dim_t output_dim,
                                    dim_t inner_size,
                                    dim_t offset,
                                    dim_t size) {
        if (size == 0) return;
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(sycl::range<1>(static_cast<size_t>(size)), [=](sycl::id<1> idx) {
            const size_t i = idx[0];
            const size_t s_inner = static_cast<size_t>(inner_size);
            const size_t in_row = static_cast<size_t>(input_dim) * s_inner;
            const size_t out_row = static_cast<size_t>(output_dim) * s_inner;
            const size_t i0 = i / in_row;
            const size_t i1 = (i / s_inner) % static_cast<size_t>(input_dim);
            const size_t i2 = i % s_inner;
            output_data[i0 * out_row
                        + (i1 + static_cast<size_t>(offset)) * s_inner
                        + i2] = input_data[i];
          });
        }).wait();
      }

      // Gather output elements from input with a depth offset.
      //
      // This maps each linear index i to the input position:
      //   row = i / output_dim
      //   col = i % output_dim
      //   output[i] = input[row * input_dim + col + offset]
      //
      // Used when stride(axis) == 1 (contiguous along the split/slide dimension).
      template <typename T>
      void gather_depth_offset(sycl::queue& queue,
                               const T* input_data,
                               T* output_data,
                               dim_t output_dim,
                               dim_t input_dim,
                               dim_t offset,
                               dim_t size) {
        if (size == 0) return;
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(sycl::range<1>(static_cast<size_t>(size)), [=](sycl::id<1> idx) {
            const size_t i = idx[0];
            const size_t row = i / static_cast<size_t>(output_dim);
            const size_t col = i % static_cast<size_t>(output_dim);
            output_data[i] = input_data[row * static_cast<size_t>(input_dim)
                                        + col + static_cast<size_t>(offset)];
          });
        }).wait();
      }

      // Gather output elements from input with an inner dimension offset.
      //
      // This maps each linear index i to the input position:
      //   i0 = i / (output_dim * inner_size)
      //   i1 = (i / inner_size) % output_dim
      //   i2 = i % inner_size
      //   output[i] = input[i0 * (input_dim * inner_size) + (i1 + offset) * inner_size + i2]
      //
      // Used when stride(axis) > 1.
      template <typename T>
      void gather_inner_dim_offset(sycl::queue& queue,
                                   const T* input_data,
                                   T* output_data,
                                   dim_t output_dim,
                                   dim_t input_dim,
                                   dim_t inner_size,
                                   dim_t offset,
                                   dim_t size) {
        if (size == 0) return;
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(sycl::range<1>(static_cast<size_t>(size)), [=](sycl::id<1> idx) {
            const size_t i = idx[0];
            const size_t s_inner = static_cast<size_t>(inner_size);
            const size_t out_row = static_cast<size_t>(output_dim) * s_inner;
            const size_t in_row = static_cast<size_t>(input_dim) * s_inner;
            const size_t i0 = i / out_row;
            const size_t i1 = (i / s_inner) % static_cast<size_t>(output_dim);
            const size_t i2 = i % s_inner;
            output_data[i] = input_data[i0 * in_row
                                        + (i1 + static_cast<size_t>(offset)) * s_inner
                                        + i2];
          });
        }).wait();
      }

    }  // namespace

    template <typename T>
    static void concat_compute(
        const dim_t _axis,
        const std::vector<const StorageView*>& inputs,
        StorageView& output) {

      auto queue = xpu::get_sycl_queue();
      const dim_t axis = _axis < 0 ? output.rank() + _axis : _axis;

      T* output_data = output.data<T>();
      dim_t offset = 0;

      for (const StorageView* input : inputs) {
        const T* input_data = input->data<T>();
        const dim_t input_size = input->size();

        if (axis == 0) {
          // Contiguous concatenation along the outer dimension.
          queue.memcpy(output_data + offset, input_data,
                       static_cast<size_t>(input_size) * sizeof(T)).wait();
          offset += input_size;
        } else {
          // Non-contiguous concatenation requires scattering elements.
          const dim_t input_dim = input->dim(axis);
          const dim_t output_dim = output.dim(axis);
          const dim_t inner_size = output.stride(axis);
          const dim_t inner_bytes = inner_size * sizeof(T);
          const dim_t input_bytes = input_size * sizeof(T);

          if (inner_size == 1) {
            scatter_depth_offset(queue, input_data, output_data,
                                 input_dim, output_dim, offset, input_size);
          } else if (inner_bytes % sizeof(uint4) == 0 && input_bytes % sizeof(uint4) == 0) {
            // Vectorized path: pack elements into 128-bit transactions.
            scatter_inner_dim_offset(queue,
                                     reinterpret_cast<const uint4*>(input_data),
                                     reinterpret_cast<uint4*>(output_data),
                                     input_dim,
                                     output_dim,
                                     inner_bytes / sizeof(uint4),
                                     offset,
                                     input_bytes / sizeof(uint4));
          } else {
            scatter_inner_dim_offset(queue, input_data, output_data,
                                     input_dim, output_dim, inner_size, offset, input_size);
          }
          offset += input_dim;
        }
      }
    }

    template <typename T>
    static void split_compute(
        const dim_t _axis,
        const StorageView& input,
        std::vector<StorageView*>& outputs) {

      auto queue = xpu::get_sycl_queue();
      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      const T* input_data = input.data<T>();
      dim_t offset = 0;

      for (StorageView* output : outputs) {
        T* output_data = output->data<T>();
        const dim_t output_size = output->size();

        if (axis == 0) {
          // Contiguous split along the outer dimension.
          queue.memcpy(output_data, input_data + offset,
                       static_cast<size_t>(output_size) * sizeof(T)).wait();
          offset += output_size;
        } else {
          // Non-contiguous split requires gathering elements.
          const dim_t output_dim = output->dim(axis);
          const dim_t input_dim = input.dim(axis);
          const dim_t inner_size = input.stride(axis);
          const dim_t inner_bytes = inner_size * sizeof(T);
          const dim_t output_bytes = output_size * sizeof(T);

          if (inner_size == 1) {
            gather_depth_offset(queue, input_data, output_data,
                                output_dim, input_dim, offset, output_size);
          } else if (inner_bytes % sizeof(uint4) == 0 && output_bytes % sizeof(uint4) == 0) {
            // Vectorized path: pack elements into 128-bit transactions.
            gather_inner_dim_offset(queue,
                                    reinterpret_cast<const uint4*>(input_data),
                                    reinterpret_cast<uint4*>(output_data),
                                    output_dim,
                                    input_dim,
                                    inner_bytes / sizeof(uint4),
                                    offset,
                                    output_bytes / sizeof(uint4));
          } else {
            gather_inner_dim_offset(queue, input_data, output_data,
                                    output_dim, input_dim, inner_size, offset, output_size);
          }
          offset += output_dim;
        }
      }
    }

    template <typename T>
    static void slide_compute(
        const dim_t _axis,
        const StorageView& input,
        StorageView& output,
        const dim_t& index) {

      auto queue = xpu::get_sycl_queue();
      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      const T* input_data = input.data<T>();
      T* output_data = output.data<T>();
      const dim_t output_size = output.size();

      if (axis == 0) {
        // Contiguous slide along the outer dimension.
        const dim_t offset = index * output.stride(axis);
        queue.memcpy(output_data, input_data + offset,
                     static_cast<size_t>(output_size) * sizeof(T)).wait();
      } else {
        // Non-contiguous slide requires gathering elements.
        const dim_t inner_size = input.stride(axis) == 0 ? 1 : input.stride(axis);
        const dim_t inner_bytes = inner_size * sizeof(T);
        const dim_t output_bytes = output_size * sizeof(T);
        const dim_t input_dim = input.dim(axis);
        const dim_t output_dim = output.dim(axis);

        if (inner_size == 1) {
          gather_depth_offset(queue, input_data, output_data,
                              output_dim, input_dim, index, output_size);
        } else if (inner_bytes % sizeof(uint4) == 0 && output_bytes % sizeof(uint4) == 0) {
          // Vectorized path: pack elements into 128-bit transactions.
          gather_inner_dim_offset(queue,
                                  reinterpret_cast<const uint4*>(input_data),
                                  reinterpret_cast<uint4*>(output_data),
                                  output_dim,
                                  input_dim,
                                  inner_bytes / sizeof(uint4),
                                  index,
                                  output_bytes / sizeof(uint4));
        } else {
          gather_inner_dim_offset(queue, input_data, output_data,
                                  output_dim, input_dim, inner_size, index, output_size);
        }
      }
    }

    template <Device D, typename T>
    void Concat::compute(const std::vector<const StorageView*>& inputs, StorageView& output) const {
      concat_compute<T>(_axis, inputs, output);
    }

    template <Device D, typename T>
    void Split::compute(const StorageView& input, std::vector<StorageView*>& outputs) const {
      split_compute<T>(_axis, input, outputs);
    }

    template <Device D, typename T>
    void Slide::compute(const StorageView& input, StorageView& output, const dim_t& index) const {
      slide_compute<T>(_axis, input, output, index);
    }

#define DECLARE_IMPL_XPU(T)                                         \
    template void                                                   \
    Concat::compute<Device::XPU, T>(                                \
        const std::vector<const StorageView*>& inputs, StorageView& output) const; \
    template void                                                   \
    Split::compute<Device::XPU, T>(                                 \
        const StorageView& input, std::vector<StorageView*>& outputs) const; \
    template void                                                   \
    Slide::compute<Device::XPU, T>(                                 \
        const StorageView& input, StorageView& output, const dim_t& index) const;

    DECLARE_IMPL_XPU(float)
    DECLARE_IMPL_XPU(int8_t)
    DECLARE_IMPL_XPU(int16_t)
    DECLARE_IMPL_XPU(int32_t)
    DECLARE_IMPL_XPU(float16_t)
    DECLARE_IMPL_XPU(bfloat16_t)

  }
}

#endif
