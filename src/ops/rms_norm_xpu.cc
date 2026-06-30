#include "ctranslate2/ops/rms_norm.h"

#ifdef CT2_WITH_XPU

#include <sycl/sycl.hpp>

#include "xpu/utils.h"

namespace ctranslate2 {
  namespace ops {

    namespace {

      // -------------------------------------------------------------------------
      // SYCL kernel for RMS normalization (last-axis).
      //
      // T  – host type (float / float16_t / bfloat16_t).  All computation is
      //      performed in float; values are widened on load and narrowed on store
      //      via `static_cast`.
      //
      // Layout:  (outer_size, depth)
      //
      // Each work-group processes one row.  All work-items participate in the
      // reduction collectives (contributing 0 if they had no elements to load),
      // which satisfies the SYCL 2020 group-algorithm convergence rule.
      //
      // Optimizations applied on top of the original CUDA-style implementation:
      //   - Larger work-group size: 512 (was 256)
      //   - .wait() on both kernel submissions for predictable synchronization
      //     (consistent with layer_norm_xpu.cc and safe regardless of queue
      //     properties)
      //   - 4-element loop unrolling for coalesced vector memory access
      //   - Gamma preloaded into local (shared) memory (saved per-row global
      //     traffic; falls back to global reads when depth is large).
      // -------------------------------------------------------------------------
      template <typename T>
      void rms_norm_kernel(sycl::queue& queue,
                           const T* input,
                           const T* gamma,
                           T* output,
                           dim_t outer_size,
                           dim_t depth,
                           float epsilon,
                           bool use_residual) {
        static constexpr size_t WG = 512;

        if (outer_size == 0 || depth == 0) return;

        const size_t depth_sz = static_cast<size_t>(depth);

        // When depth fits in local memory, cache gamma there.
        // Intel GPU local (SLM) memory: 64 KB typical, so 16384 * sizeof(T)
        // is a safe upper bound for all three types (float 64 KB, half/bf16 32 KB).
        constexpr size_t max_local_depth = 16384;

        if (depth_sz <= max_local_depth) {
          // --- Gamma-cached path (gamma in local memory) ---
          queue.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<T, 1> s_gamma(
                sycl::range<1>(depth_sz), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(
                    sycl::range<1>(static_cast<size_t>(outer_size) * WG),
                    sycl::range<1>(WG)),
                [=](sycl::nd_item<1> item) {
                  const size_t row = item.get_group_linear_id();
                  if (row >= static_cast<size_t>(outer_size)) return;

                  const size_t tid = item.get_local_linear_id();
                  const size_t n   = depth_sz;
                  const size_t off = row * n;

                  // Preload gamma (coalesced, work-items cooperate)
                  for (size_t j = tid; j < n; j += WG)
                    s_gamma[j] = gamma[j];
                  sycl::group_barrier(item.get_group());

                  // --- sum of squares with 4-wide unrolling ---
                  float sum_sq = 0.0f;
                  size_t j = tid;
                  for (; j + 4 <= n; j += WG * 4) {
                    const float v0 = static_cast<float>(input[off + j + 0]);
                    const float v1 = static_cast<float>(input[off + j + 1]);
                    const float v2 = static_cast<float>(input[off + j + 2]);
                    const float v3 = static_cast<float>(input[off + j + 3]);
                    sum_sq += v0*v0 + v1*v1 + v2*v2 + v3*v3;
                  }
                  for (; j < n; j += WG) {
                    const float v = static_cast<float>(input[off + j]);
                    sum_sq += v * v;
                  }

                  // --- work-group reduction ---
                  const float total = sycl::reduce_over_group(
                      item.get_group(), sum_sq, sycl::plus<float>());
                  const float inv_rms =
                      sycl::rsqrt(total / static_cast<float>(n) + epsilon);

                  // --- normalize & apply gamma (from local memory) ---
                  j = tid;
                  for (; j + 4 <= n; j += WG * 4) {
                    const float x0 = static_cast<float>(input[off + j + 0]);
                    const float x1 = static_cast<float>(input[off + j + 1]);
                    const float x2 = static_cast<float>(input[off + j + 2]);
                    const float x3 = static_cast<float>(input[off + j + 3]);
                    const float g0 = static_cast<float>(s_gamma[j + 0]);
                    const float g1 = static_cast<float>(s_gamma[j + 1]);
                    const float g2 = static_cast<float>(s_gamma[j + 2]);
                    const float g3 = static_cast<float>(s_gamma[j + 3]);
                    if (use_residual) {
                      output[off + j + 0] =
                          static_cast<T>(x0 * inv_rms * (1.0f + g0));
                      output[off + j + 1] =
                          static_cast<T>(x1 * inv_rms * (1.0f + g1));
                      output[off + j + 2] =
                          static_cast<T>(x2 * inv_rms * (1.0f + g2));
                      output[off + j + 3] =
                          static_cast<T>(x3 * inv_rms * (1.0f + g3));
                    } else {
                      output[off + j + 0] =
                          static_cast<T>(x0 * inv_rms * g0);
                      output[off + j + 1] =
                          static_cast<T>(x1 * inv_rms * g1);
                      output[off + j + 2] =
                          static_cast<T>(x2 * inv_rms * g2);
                      output[off + j + 3] =
                          static_cast<T>(x3 * inv_rms * g3);
                    }
                  }
                  for (; j < n; j += WG) {
                    const float x = static_cast<float>(input[off + j]);
                    const float gv = static_cast<float>(s_gamma[j]);
                    if (use_residual)
                      output[off + j] =
                          static_cast<T>(x * inv_rms * (1.0f + gv));
                    else
                      output[off + j] =
                          static_cast<T>(x * inv_rms * gv);
                  }
                });
          }).wait();
        } else {
          // --- Gamma from global memory (depth too large for local memory) ---
          queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(
                    sycl::range<1>(static_cast<size_t>(outer_size) * WG),
                    sycl::range<1>(WG)),
                [=](sycl::nd_item<1> item) {
                  const size_t row = item.get_group_linear_id();
                  if (row >= static_cast<size_t>(outer_size)) return;

                  const size_t tid = item.get_local_linear_id();
                  const size_t n   = depth_sz;
                  const size_t off = row * n;

                  // --- sum of squares ---
                  float sum_sq = 0.0f;
                  size_t j = tid;
                  for (; j + 4 <= n; j += WG * 4) {
                    const float v0 = static_cast<float>(input[off + j + 0]);
                    const float v1 = static_cast<float>(input[off + j + 1]);
                    const float v2 = static_cast<float>(input[off + j + 2]);
                    const float v3 = static_cast<float>(input[off + j + 3]);
                    sum_sq += v0*v0 + v1*v1 + v2*v2 + v3*v3;
                  }
                  for (; j < n; j += WG) {
                    const float v = static_cast<float>(input[off + j]);
                    sum_sq += v * v;
                  }

                  const float total = sycl::reduce_over_group(
                      item.get_group(), sum_sq, sycl::plus<float>());
                  const float inv_rms =
                      sycl::rsqrt(total / static_cast<float>(n) + epsilon);

                  // --- normalize & apply gamma (from global memory) ---
                  j = tid;
                  for (; j + 4 <= n; j += WG * 4) {
                    const float x0 = static_cast<float>(input[off + j + 0]);
                    const float x1 = static_cast<float>(input[off + j + 1]);
                    const float x2 = static_cast<float>(input[off + j + 2]);
                    const float x3 = static_cast<float>(input[off + j + 3]);
                    const float g0 = static_cast<float>(gamma[j + 0]);
                    const float g1 = static_cast<float>(gamma[j + 1]);
                    const float g2 = static_cast<float>(gamma[j + 2]);
                    const float g3 = static_cast<float>(gamma[j + 3]);
                    if (use_residual) {
                      output[off + j + 0] =
                          static_cast<T>(x0 * inv_rms * (1.0f + g0));
                      output[off + j + 1] =
                          static_cast<T>(x1 * inv_rms * (1.0f + g1));
                      output[off + j + 2] =
                          static_cast<T>(x2 * inv_rms * (1.0f + g2));
                      output[off + j + 3] =
                          static_cast<T>(x3 * inv_rms * (1.0f + g3));
                    } else {
                      output[off + j + 0] =
                          static_cast<T>(x0 * inv_rms * g0);
                      output[off + j + 1] =
                          static_cast<T>(x1 * inv_rms * g1);
                      output[off + j + 2] =
                          static_cast<T>(x2 * inv_rms * g2);
                      output[off + j + 3] =
                          static_cast<T>(x3 * inv_rms * g3);
                    }
                  }
                  for (; j < n; j += WG) {
                    const float x = static_cast<float>(input[off + j]);
                    const float gv = static_cast<float>(gamma[j]);
                    if (use_residual)
                      output[off + j] =
                          static_cast<T>(x * inv_rms * (1.0f + gv));
                    else
                      output[off + j] =
                          static_cast<T>(x * inv_rms * gv);
                  }
                });
          }).wait();
        }
      }

    }  // anonymous namespace

    // ==========================================================================
    // RMSNorm::compute<Device::XPU, T>
    // ==========================================================================

    template <Device D, typename T>
    void RMSNorm::compute(
        const StorageView& gamma,
        const StorageView& input,
        StorageView& output) const {
      auto queue = xpu::get_sycl_queue();
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      rms_norm_kernel(queue,
                      input.data<T>(),
                      gamma.data<T>(),
                      output.data<T>(),
                      batch_size, depth, _epsilon, _use_residual);
    }

#define DEFINE_IMPL_XPU(T)                                              \
    template void                                                       \
    RMSNorm::compute<Device::XPU, T>(                                   \
        const StorageView& gamma,                                       \
        const StorageView& input,                                       \
        StorageView& output) const;

    DEFINE_IMPL_XPU(float)
    DEFINE_IMPL_XPU(float16_t)
    DEFINE_IMPL_XPU(bfloat16_t)

  }
}

#endif
