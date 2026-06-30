#include "ctranslate2/ops/gather.h"

#ifdef CT2_WITH_XPU

#include <cstdint>
#include <sycl/sycl.hpp>

#include "xpu/utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    namespace {

      // 128-bit (16-byte) packed type for vectorized memory access.
      // Matches CUDA's uint4 layout, enabling wider load/store transactions.
      struct alignas(16) uint4 {
        uint32_t x, y, z, w;
      };

      // Dispatches element-wise batch gather to either vectorized (uint4) or scalar
      // kernel depending on alignment of the axis stride.
      //
      // Each output element at linear index i is gathered from the source position:
      //   inner_index  = i % axis_stride           (element within the gathered chunk)
      //   outer_index  = i / axis_stride           (which index slot)
      //   batch_index  = outer_index / num_indices_per_batch
      //   src_index    = batch_index * batch_stride
      //                 + indices[outer_index] * axis_stride
      //                 + inner_index
      //
      template <typename T>
      void gather_impl(sycl::queue& queue,
                       const T* src,
                       T* dst,
                       const int32_t* indices,
                       const dim_t batch_stride,
                       const dim_t axis_stride,
                       const dim_t num_indices_per_batch,
                       const dim_t dst_size) {
        if (dst_size == 0) return;

        constexpr dim_t elem_size = static_cast<dim_t>(sizeof(T));
        const dim_t axis_bytes = axis_stride * elem_size;
        constexpr dim_t vec_size = static_cast<dim_t>(sizeof(uint4));

        // Vectorized path: use 128-bit loads/stores when the axis stride is aligned,
        // matching the CUDA optimization that uses uint4.
        if (axis_bytes % vec_size == 0) {
          constexpr dim_t vec_count = vec_size / elem_size;  // T elements per uint4
          const dim_t vec_dst_size = dst_size / vec_count;
          const dim_t vec_batch_stride = batch_stride / vec_count;
          const dim_t vec_axis_stride = axis_stride / vec_count;

          const uint4* src4 = reinterpret_cast<const uint4*>(src);
          uint4* dst4 = reinterpret_cast<uint4*>(dst);

          queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::range<1>(static_cast<size_t>(vec_dst_size)),
                [=](sycl::id<1> idx) {
              const size_t i = idx[0];
              const size_t axis_str = static_cast<size_t>(vec_axis_stride);
              const size_t batch_str = static_cast<size_t>(vec_batch_stride);
              const size_t num_idx = static_cast<size_t>(num_indices_per_batch);
              const size_t inner_index = i % axis_str;
              const size_t outer_index = i / axis_str;
              const size_t batch_index = outer_index / num_idx;
              dst4[i] = src4[batch_index * batch_str
                             + static_cast<size_t>(indices[outer_index]) * axis_str
                             + inner_index];
            });
          }).wait();

        } else {
          // Scalar path: one element per work-item.
          queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::range<1>(static_cast<size_t>(dst_size)),
                [=](sycl::id<1> idx) {
              const size_t i = idx[0];
              const size_t axis_str = static_cast<size_t>(axis_stride);
              const size_t batch_str = static_cast<size_t>(batch_stride);
              const size_t num_idx = static_cast<size_t>(num_indices_per_batch);
              const size_t inner_index = i % axis_str;
              const size_t outer_index = i / axis_str;
              const size_t batch_index = outer_index / num_idx;
              dst[i] = src[batch_index * batch_str
                           + static_cast<size_t>(indices[outer_index]) * axis_str
                           + inner_index];
            });
          }).wait();
        }
      }

      // Shared host-side dispatch: extracts buffers, validates axis, and calls gather_impl.
      template <typename T>
      void gather_compute(const StorageView& data,
                          const StorageView& input,
                          const dim_t axis,
                          const dim_t batch_dims,
                          StorageView& output) {
        auto queue = xpu::get_sycl_queue();

        if (axis == batch_dims) {
          const dim_t batch_stride = axis > 0 ? data.stride(axis - 1) : data.size();
          const dim_t batch_size = data.size() / batch_stride;
          const dim_t num_indices = input.size();
          const dim_t num_indices_per_batch = num_indices / batch_size;
          const dim_t axis_stride = data.stride(axis);
          const dim_t dst_size = output.size();

          const T* src = data.data<T>();
          T* dst = output.data<T>();
          const int32_t* indices = input.data<int32_t>();

          gather_impl(queue, src, dst, indices,
                      batch_stride, axis_stride,
                      num_indices_per_batch, dst_size);
        } else {
          throw std::invalid_argument("Gather only supports indexing the first non batch dimension");
        }
      }

    }  // namespace

    // Full specializations for each type on XPU.
    // These mirror the CUDA DECLARE_ALL_TYPES instantiation pattern but use full
    // specializations (not a primary template) to avoid ODR violations with the
    // CPU backend's primary template definition.

    template <Device D, typename T>
    void Gather::compute(
        const StorageView& data,
        const StorageView& input,
        const dim_t axis,
        const dim_t batch_dims,
        StorageView& output) const {
      gather_compute<T>(data, input, axis, batch_dims, output);
    }

#define DECLARE_XPU_GATHER(T)                                           \
    template void                                                       \
    Gather::compute<Device::XPU, T>(                                    \
        const StorageView& data,                                        \
        const StorageView& input,                                       \
        const dim_t axis,                                               \
        const dim_t batch_dims,                                         \
        StorageView& output) const;

    DECLARE_ALL_TYPES(DECLARE_XPU_GATHER)

  }
}

#endif
