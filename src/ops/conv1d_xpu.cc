#include "ctranslate2/ops/conv1d.h"
#include "ctranslate2/ops/gemm.h"
#include "ctranslate2/primitives.h"
#include "type_dispatch.h"

#ifdef CT2_WITH_XPU

#include <sycl/sycl.hpp>

#include <dnnl.hpp>
#include <dnnl_sycl.hpp>

#include "xpu/utils.h"

namespace ctranslate2 {
  namespace ops {

    // SYCL kernel for im2col transformation (transposed version).
    // Mirrors the CUDA kernel in conv1d_gpu.cu.
    template <Device D, typename T>
    void Conv1D::compute(
        const StorageView& input,
        const StorageView& weight,
        const StorageView* bias,
        StorageView& output,
        const StorageView* qscale) const {

      if (qscale)
        throw std::runtime_error("Quantization is not supported in this Conv1D implementation");

      const dim_t batch_size = input.dim(0);
      const dim_t in_channels = input.dim(1);
      const dim_t input_length = input.dim(2);
      const dim_t out_channels = weight.dim(0);
      const dim_t kernel_size = weight.dim(2);
      const dim_t output_length = output.dim(2);
      const dim_t in_channels_per_group = in_channels / _groups;
      const dim_t out_channels_per_group = out_channels / _groups;
      const dim_t k = in_channels_per_group * kernel_size;

      // Allocate the im2col buffer on the XPU device.
      StorageView buffer({batch_size, _groups, output_length, k},
                         DataTypeToEnum<T>::value, Device::XPU);

      const T* x = input.data<T>();
      const T* w = weight.data<T>();
      T* o = output.data<T>();
      T* p = buffer.data<T>();

      const dim_t in_batch_stride = in_channels * input_length;
      const dim_t in_group_stride = in_batch_stride / _groups;

      // Copy member variables to local for SYCL lambda capture.
      const dim_t groups = _groups;
      const dim_t stride = _stride;
      const dim_t padding = _padding;
      const dim_t dilation = _dilation;

      // Launch the im2col kernel.
      {
        auto queue = xpu::get_sycl_queue();
        queue.parallel_for(
            sycl::range<3>(
                static_cast<size_t>(batch_size * groups),   // batch_group
                static_cast<size_t>(output_length),          // ti_idx
                static_cast<size_t>(k)),                     // idx
            [=](sycl::id<3> idx) {
              const int batch_group = static_cast<int>(idx[0]);
              const int ti_idx = static_cast<int>(idx[1]);
              const int li = static_cast<int>(idx[2]);

              // Decompose the linear index (same logic as CUDA kernel).
              const int c_offset = li / kernel_size;
              const int k_offset = li - c_offset * kernel_size;
              const int batch_idx = batch_group / groups;
              const int group_idx = batch_group - batch_idx * groups;

              // Calculate input position.
              const int ti = ti_idx * stride - padding;
              const int window_i = dilation * k_offset + ti;

              // Calculate input offset.
              const int input_idx = (batch_idx * in_batch_stride
                                     + group_idx * in_group_stride
                                     + c_offset * input_length
                                     + window_i);
              const int output_idx = (batch_group * output_length + ti_idx) * k + li;

              // Fill output (zero-pad out-of-bounds accesses).
              p[output_idx] = (window_i >= 0 && window_i < input_length)
                              ? x[input_idx]
                              : T(0);
            }).wait();
      }

      // Run a batched GEMM per group (same pattern as CUDA version).
      const dim_t stridew = out_channels_per_group * in_channels_per_group * kernel_size;
      const dim_t stridep = k * output_length;
      const dim_t strideo = out_channels_per_group * output_length;

      for (dim_t g = 0; g < groups; ++g) {
        const T* w_g = w + g * stridew;
        const T* p_g = p + g * stridep;
        T* o_g = o + g * strideo;

        primitives<Device::XPU>::gemm_batch_strided(
            /*transpose_a=*/false, /*transpose_b=*/true,
            out_channels_per_group, output_length, k,
            1.0f,
            w_g, k, /*stridea=*/0,
            p_g, k, /*strideb=*/groups * stridep,
            0.0f,
            o_g, output_length, /*stridec=*/groups * strideo,
            batch_size);
      }

      apply_bias_and_activation(output, bias, _activation_type,
                                /*residual=*/nullptr, /*axis=*/-2);
    }

    #define DECLARE_IMPL_XPU(T)                                             \
    template void                                                           \
    Conv1D::compute<Device::XPU, T>(const StorageView& input,              \
                                     const StorageView& weight,             \
                                     const StorageView* bias,               \
                                     StorageView& output,                   \
                                     const StorageView* qscale) const;

    DECLARE_IMPL_XPU(float)
    DECLARE_IMPL_XPU(float16_t)
    DECLARE_IMPL_XPU(bfloat16_t)


  }
}

#endif
