#include "ctranslate2/ops/dequantize.h"

#ifdef CT2_WITH_XPU

#include <sycl/sycl.hpp>

#include "xpu/utils.h"

namespace ctranslate2 {
  namespace ops {

    // Shared helper to dequantize int8 to any output type (float, float16_t, bfloat16_t).
    template <typename T>
    static void dequantize_int8_impl(const StorageView& input,
                                     const StorageView& scale,
                                     StorageView& output) {
      const dim_t depth = input.dim(-1);
      const dim_t total_size = input.size();

      const auto* input_data = input.data<int8_t>();
      const auto* scale_data = scale.data<float>();
      auto* output_data = output.data<T>();

      auto queue = xpu::get_sycl_queue();

      queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_size)), [=](sycl::id<1> idx) {
          const size_t i = idx[0];
          const size_t row = i / static_cast<size_t>(depth);
          output_data[i] = static_cast<T>(static_cast<float>(input_data[i]) / scale_data[row]);
        });
      }).wait();
    }

    template <Device D, typename InT, typename OutT>
    void Dequantize::dequantize(
        const StorageView& input,
        const StorageView& scale,
        StorageView& output) const {
      if constexpr (std::is_same_v<InT, int8_t>) {
        dequantize_int8_impl<OutT>(input, scale, output);
      } else if constexpr (std::is_same_v<InT, int16_t>) {
        const dim_t total_size = input.size();
        const float scale_val = scale.as_scalar<float>();
        const float r_scale = 1.0f / scale_val;

        const auto* input_data = input.data<int16_t>();
        auto* output_data = output.data<float>();

        auto queue = xpu::get_sycl_queue();

        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_size)), [=](sycl::id<1> idx) {
            output_data[idx[0]] = static_cast<float>(input_data[idx[0]]) * r_scale;
          });
        }).wait();
      }
    }

    // -----------------------------------------------------------------------
    // dequantize_gemm_output — fused rescale + bias + activation for int32 GEMM output
    // -----------------------------------------------------------------------

    namespace {
      template <typename T>
      void dequantize_gemm_output_kernel(const int32_t* c,
                                         const float* a_scales,
                                         const float* b_scales,
                                         const bool transpose_a,
                                         const bool transpose_b,
                                         const T* bias,
                                         const ActivationType* activation_type,
                                         T* y,
                                         const dim_t batch_size,
                                         const dim_t depth) {
        if (batch_size == 0 || depth == 0)
          return;

        auto queue = xpu::get_sycl_queue();
        const dim_t total_size = batch_size * depth;

        // Resolve activation on the host side so the device lambda only carries
        // a plain integer (avoiding host-pointer dereference in device code).
        const bool has_activation = (activation_type != nullptr);
        const int act_type = has_activation ? static_cast<int>(*activation_type) : 0;

        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::range<1>(static_cast<size_t>(total_size)),
              [=](sycl::id<1> idx) {
                const dim_t i = static_cast<dim_t>(idx[0]) / depth;  // row
                const dim_t j = static_cast<dim_t>(idx[0]) % depth;  // col

                const float scale_a = transpose_a ? a_scales[j] : a_scales[i];
                const float scale_b = transpose_b ? b_scales[j] : b_scales[i];
                float v = static_cast<float>(c[idx[0]]) / (scale_a * scale_b);

                if (bias)
                  v += static_cast<float>(bias[j]);

                if (has_activation) {
                  switch (static_cast<ActivationType>(act_type)) {
                  case ActivationType::ReLU:
                    v = sycl::fmax(0.0f, v);
                    break;
                  case ActivationType::GELU:
                    v = 0.5f * v * (1.0f + sycl::erf(0.7071067811865475f * v));
                    break;
                  case ActivationType::GELUTanh: {
                    const float t = sycl::tanh(0.7978845608028654f
                                               * (v + 0.044715f * v * v * v));
                    v = 0.5f * v * (1.0f + t);
                    break;
                  }
                  case ActivationType::GELUSigmoid:
                    v = v / (1.0f + sycl::exp(-1.702f * v));
                    break;
                  case ActivationType::Sigmoid:
                    v = 1.0f / (1.0f + sycl::exp(-v));
                    break;
                  case ActivationType::Swish:
                    v = v / (1.0f + sycl::exp(-v));
                    break;
                  case ActivationType::Tanh:
                    v = sycl::tanh(v);
                    break;
                  }
                }

                y[idx[0]] = static_cast<T>(v);
              });
        }).wait();
      }
    }

    template <Device D, typename T>
    void Dequantize::dequantize_gemm_output(
        const StorageView& c,
        const StorageView& a_scale,
        const StorageView& b_scale,
        const bool transpose_a,
        const bool transpose_b,
        const StorageView* bias,
        StorageView& y) const {
      dequantize_gemm_output_kernel(
          c.data<int32_t>(),
          a_scale.data<float>(),
          b_scale.data<float>(),
          transpose_a,
          transpose_b,
          bias ? bias->data<T>() : nullptr,
          _activation_type,
          y.data<T>(),
          a_scale.size(),
          c.dim(-1));
    }

#define DECLARE_IMPL(InT, OutT)                                         \
    template void                                                       \
    Dequantize::dequantize<Device::XPU, InT, OutT>(                     \
        const StorageView&, const StorageView&, StorageView&) const;

#define DECLARE_IMPL_GEMM_OUTPUT(T)                                     \
    template void                                                       \
    Dequantize::dequantize_gemm_output<Device::XPU, T>(                 \
        const StorageView&, const StorageView&, const StorageView&,     \
        const bool, const bool, const StorageView*, StorageView&) const;

    DECLARE_IMPL(int8_t, float)
    DECLARE_IMPL(int8_t, float16_t)
    DECLARE_IMPL(int8_t, bfloat16_t)
    DECLARE_IMPL(int16_t, float)

    DECLARE_IMPL_GEMM_OUTPUT(float)
    DECLARE_IMPL_GEMM_OUTPUT(float16_t)
    DECLARE_IMPL_GEMM_OUTPUT(bfloat16_t)

  }
}

#endif
