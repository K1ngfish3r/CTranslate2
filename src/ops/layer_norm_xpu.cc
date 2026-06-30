#include "ctranslate2/ops/layer_norm.h"

#ifdef CT2_WITH_XPU

#include <sycl/sycl.hpp>

#include "xpu/utils.h"

namespace ctranslate2 {
  namespace ops {

    namespace {

      // -------------------------------------------------------------------------
      // SYCL kernel for last-axis layer normalization (inner_size == 1).
      //
      // Layout:  (outer_size, axis_size)
      //
      // Each work-group processes one row.  All work-items participate in the
      // reduction collectives (contributing 0 if they had no elements to load),
      // which satisfies the SYCL 2020 group-algorithm convergence rule.
      // -------------------------------------------------------------------------
      template <typename T>
      void layer_norm_last_axis(sycl::queue& queue,
                                const T* input,
                                const T* gamma,
                                const T* beta,
                                T* output,
                                dim_t outer_size,
                                dim_t axis_size,
                                float epsilon) {
        static constexpr size_t WG = 512;

        // Special-case: empty input
        if (outer_size == 0 || axis_size == 0) return;

        auto* g = gamma;
        auto* b = beta;
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(outer_size) * WG),
                                sycl::range<1>(WG)),
              [=](sycl::nd_item<1> item) {
                const size_t row = item.get_group_linear_id();
                if (row >= static_cast<size_t>(outer_size)) return;

                const size_t tid = item.get_local_linear_id();
                const size_t n   = static_cast<size_t>(axis_size);
                const size_t off = row * n;

                // --- partial sums (each work-item accumulates a strided slice) ---
                float sum1 = 0.0f;
                float sum2 = 0.0f;
                for (size_t j = tid; j < n; j += WG) {
                  float v = static_cast<float>(input[off + j]);
                  sum1 += v;
                  sum2 += v * v;
                }

                // --- work-group reduction ---
                float total1 = sycl::reduce_over_group(item.get_group(), sum1,
                                                       sycl::plus<float>());
                float total2 = sycl::reduce_over_group(item.get_group(), sum2,
                                                       sycl::plus<float>());

                // --- mean & variance ---
                float scale   = 1.0f / static_cast<float>(n);
                float mean    = total1 * scale;
                float var     = sycl::max(total2 * scale - mean * mean, 0.0f);
                float inv_std = sycl::rsqrt(var + epsilon);

                // --- normalize & apply gamma/beta ---
                for (size_t j = tid; j < n; j += WG) {
                  float gv = (g == nullptr) ? 1.0f : static_cast<float>(g[j]);
                  float bv = (b == nullptr) ? 0.0f : static_cast<float>(b[j]);
                  output[off + j] = static_cast<T>(
                      (static_cast<float>(input[off + j]) - mean) * inv_std * gv + bv);
                }
              });
        }).wait();
      }

      // -------------------------------------------------------------------------
      // SYCL kernel for non-last-axis layer normalization (inner_size > 1).
      //
      // Layout:  (outer_size, axis_size, inner_size)
      //
      // Each work-group processes one (outer_idx, inner_idx) pair and reduces
      // across the axis dimension.
      // -------------------------------------------------------------------------
      template <typename T>
      void layer_norm_axis(sycl::queue& queue,
                           const T* input,
                           const T* gamma,
                           const T* beta,
                           T* output,
                           dim_t outer_size,
                           dim_t axis_size,
                           dim_t inner_size,
                           float epsilon) {
        static constexpr size_t WG = 512;

        if (outer_size == 0 || axis_size == 0 || inner_size == 0) return;

        size_t n_groups = static_cast<size_t>(outer_size) *
                          static_cast<size_t>(inner_size);

        auto* g = gamma;
        auto* b = beta;
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::nd_range<1>(
                  sycl::range<1>(n_groups * WG),
                  sycl::range<1>(WG)),
              [=](sycl::nd_item<1> item) {
                size_t gid = item.get_group_linear_id();
                if (gid >= n_groups) return;

                size_t outer_idx = gid / static_cast<size_t>(inner_size);
                size_t inner_idx = gid % static_cast<size_t>(inner_size);

                size_t tid = item.get_local_linear_id();
                size_t n   = static_cast<size_t>(axis_size);

                // --- partial sums ---
                float sum1 = 0.0f;
                float sum2 = 0.0f;
                for (size_t k = tid; k < n; k += WG) {
                  size_t idx = (outer_idx * n + k) *
                                   static_cast<size_t>(inner_size) + inner_idx;
                  float v = static_cast<float>(input[idx]);
                  sum1 += v;
                  sum2 += v * v;
                }

                // --- work-group reduction ---
                float total1 = sycl::reduce_over_group(item.get_group(), sum1,
                                                       sycl::plus<float>());
                float total2 = sycl::reduce_over_group(item.get_group(), sum2,
                                                       sycl::plus<float>());

                // --- mean & variance ---
                float scale   = 1.0f / static_cast<float>(n);
                float mean    = total1 * scale;
                float var     = sycl::max(total2 * scale - mean * mean, 0.0f);
                float inv_std = sycl::rsqrt(var + epsilon);

                // --- normalize & apply gamma/beta ---
                for (size_t k = tid; k < n; k += WG) {
                  size_t idx = (outer_idx * n + k) *
                                   static_cast<size_t>(inner_size) + inner_idx;
                  float gv = (g == nullptr) ? 1.0f : static_cast<float>(g[inner_idx]);
                  float bv = (b == nullptr) ? 0.0f : static_cast<float>(b[inner_idx]);
                  output[idx] = static_cast<T>(
                      (static_cast<float>(input[idx]) - mean) * inv_std * gv + bv);
                }
              });
        }).wait();
      }

      // -------------------------------------------------------------------------
      // Helper to dispatch to the correct kernel function based on axis position.
      // -------------------------------------------------------------------------
      template <typename T>
      void dispatch_layer_norm(sycl::queue& queue,
                               const StorageView& input,
                               const StorageView* gamma,
                               const StorageView* beta,
                               StorageView& output,
                               dim_t axis,
                               dim_t outer_size,
                               dim_t axis_size,
                               dim_t inner_size,
                               float epsilon) {
        const T* input_data = input.data<T>();
        T* output_data = output.data<T>();
        const T* gamma_data = gamma ? gamma->data<T>() : nullptr;
        const T* beta_data  = beta  ? beta->data<T>()  : nullptr;

        if (axis == input.rank() - 1) {
          layer_norm_last_axis(queue, input_data, gamma_data, beta_data,
                               output_data, outer_size, axis_size, epsilon);
        } else {
          layer_norm_axis(queue, input_data, gamma_data, beta_data,
                          output_data, outer_size, axis_size, inner_size, epsilon);
        }
      }

    }  // anonymous namespace

    // ==========================================================================
    // LayerNorm::compute<Device::XPU, T>
    // ==========================================================================

    template <Device D, typename T>
    void LayerNorm::compute(
        const StorageView* beta,
        const StorageView* gamma,
        const StorageView& input,
        const dim_t axis,
        const dim_t outer_size,
        const dim_t axis_size,
        const dim_t inner_size,
        StorageView& output) const {
      auto queue = xpu::get_sycl_queue();
      dispatch_layer_norm<T>(queue, input, gamma, beta, output,
                             axis, outer_size, axis_size,
                             inner_size, _epsilon);
    }

#define DEFINE_IMPL_XPU(T)                                              \
    template void                                                       \
    LayerNorm::compute<Device::XPU, T>(                                 \
        const StorageView* beta,                                        \
        const StorageView* gamma,                                       \
        const StorageView& input,                                       \
        const dim_t axis,                                               \
        const dim_t outer_size,                                         \
        const dim_t axis_size,                                          \
        const dim_t inner_size,                                         \
        StorageView& output) const;

    DEFINE_IMPL_XPU(float)
    DEFINE_IMPL_XPU(float16_t)
    DEFINE_IMPL_XPU(bfloat16_t)

  }
}

#endif
