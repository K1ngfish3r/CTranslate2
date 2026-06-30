#include "ctranslate2/ops/quantize.h"

#ifdef CT2_WITH_XPU

#include <sycl/sycl.hpp>

#include "xpu/utils.h"

namespace ctranslate2 {
  namespace ops {

    template <typename T>
    static void quantize_int8_impl(const StorageView& input,
                                   StorageView& output,
                                   StorageView& scale,
                                   bool shift_to_uint8,
                                   bool round_before_cast) {
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = scale.size();
      const dim_t total_size = input.size();

      const auto* input_data = input.data<T>();
      auto* output_data = output.data<int8_t>();
      auto* scale_data = scale.data<float>();

      auto queue = xpu::get_sycl_queue();

      // Phase 1: Compute per-row scale = 127.0f / max_abs(row).
      queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(batch_size)), [=](sycl::id<1> idx) {
          const size_t row = idx[0];
          const size_t offset = row * static_cast<size_t>(depth);

          float amax = 0.0f;
          for (size_t d = 0; d < static_cast<size_t>(depth); ++d) {
            const float fval = static_cast<float>(input_data[offset + d]);
            const float abs_val = sycl::fabs(fval);
            if (abs_val > amax)
              amax = abs_val;
          }

          scale_data[row] = (amax != 0.0f) ? 127.0f / amax : 1.0f;
        });
      }).wait();

      // Phase 2: Quantize each element.
      queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_size)), [=](sycl::id<1> idx) {
          const size_t i = idx[0];
          const size_t row = i / static_cast<size_t>(depth);

          float scaled = static_cast<float>(input_data[i]) * scale_data[row];

          if (shift_to_uint8) {
            scaled = scaled + 128.0f;
          }

          if (round_before_cast) {
            scaled = sycl::rint(scaled);
          }

          constexpr float int8_min = -128.0f;
          constexpr float int8_max = 127.0f;
          if (scaled < int8_min)
            scaled = int8_min;
          if (scaled > int8_max)
            scaled = int8_max;

          if (shift_to_uint8) {
            auto* output_u8 = reinterpret_cast<uint8_t*>(output_data);
            output_u8[i] = static_cast<uint8_t>(scaled);
          } else {
            output_data[i] = static_cast<int8_t>(scaled);
          }
        });
      }).wait();
    }

    template <Device D, typename InT, typename OutT>
    void Quantize::quantize(
        const StorageView& input,
        StorageView& output,
        StorageView& scale) const {
      quantize_int8_impl<InT>(input, output, scale, _shift_to_uint8, _round_before_cast);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Quantize::quantize<Device::XPU, T, int8_t>(const StorageView&,      \
                                               StorageView&,            \
                                               StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}

#endif
