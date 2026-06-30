#include "ctranslate2/ops/median_filter.h"

#ifdef CT2_WITH_XPU

#include <sycl/sycl.hpp>
#include <dnnl_sycl.hpp>

#include "ctranslate2/bfloat16.h"
#include "xpu/utils.h"

namespace ctranslate2 {
  namespace ops {

    namespace {
      constexpr dim_t kMaxWindow = 129;  // supports window widths up to 129 (rank 64)

      template <typename T>
      void median_filter_kernel_xpu(sycl::queue& queue,
                                    const T* src,
                                    T* dst,
                                    dim_t batch_size,
                                    dim_t depth,
                                    dim_t width) {
        const dim_t rank = width / 2;
        const size_t total = static_cast<size_t>(batch_size * depth);

        queue.parallel_for(
            sycl::range<1>(total),
            [=](sycl::id<1> idx) {
              const dim_t tid = static_cast<dim_t>(idx[0]);
              const dim_t row = tid / depth;
              const dim_t col = tid % depth;
              const dim_t row_offset = row * depth;

              // Private memory for the window (fixed max size).
              // Always operate in float to share the insertion-sort path
              // regardless of the input type (float, float16_t, bfloat16_t).
              float window[kMaxWindow];

              // Reflection gather — cast to float for half-precision types.
              for (dim_t k = -rank; k <= rank; ++k) {
                dim_t read = col + k;
                if (read < 0) read = -read;
                if (read >= depth) read = 2 * depth - read - 2;
                window[k + rank] = static_cast<float>(src[row_offset + read]);
              }

              // Insertion sort (width is small: <= kMaxWindow, typically < 129).
              for (dim_t i = 1; i < width; ++i) {
                float key = window[i];
                dim_t j = i - 1;
                while (j >= 0 && window[j] > key) {
                  window[j + 1] = window[j];
                  --j;
                }
                window[j + 1] = key;
              }

              dst[tid] = static_cast<T>(window[rank]);
            }).wait();
      }
    }

    template <Device D, typename T>
    void MedianFilter::compute(const StorageView& input,
                               const dim_t axis_size,
                               StorageView& output) const {

      const dim_t depth = axis_size;
      const dim_t batch_size = input.size() / depth;
      const dim_t width = _width;
      const dim_t rank = width / 2;

      if (width <= 1) {
        if (&output != &input)
          output.copy_from(input);
        return;
      }
      if ((width & 1) == 0)
        throw std::invalid_argument("MedianFilter width must be odd");
      if (width > kMaxWindow)
        throw std::invalid_argument("MedianFilter width exceeds supported XPU max ("
                                    + std::to_string(kMaxWindow) + ")");
      if (depth <= rank) {
        if (&output != &input)
          output.copy_from(input);
        return;
      }

      auto queue = xpu::get_sycl_queue();

      median_filter_kernel_xpu<T>(
          queue,
          input.data<T>(),
          output.data<T>(),
          batch_size,
          depth,
          width);
    }

#define DECLARE_IMPL_XPU(T)                                         \
    template void                                                          \
    MedianFilter::compute<Device::XPU, T>(                           \
        const StorageView&, const dim_t, StorageView&) const;

    DECLARE_IMPL_XPU(float)
    DECLARE_IMPL_XPU(float16_t)
    DECLARE_IMPL_XPU(bfloat16_t)

  }
}

#endif
