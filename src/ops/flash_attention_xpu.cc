#include "ctranslate2/ops/flash_attention.h"

#ifdef CT2_WITH_XPU

#include <sycl/sycl.hpp>
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>

#include <cmath>
#include <limits>

#include "ctranslate2/bfloat16.h"
#include "ctranslate2/ops/transpose.h"
#include "dispatch.h"
#include "xpu/utils.h"

namespace ctranslate2 {
namespace ops {

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------
namespace {

/// Promote fp16/bf16 to float for computation; pass through for float.
template <typename T>
struct ComputeType { using type = T; };
template <>
struct ComputeType<float16_t> { using type = float; };
template <>
struct ComputeType<bfloat16_t> { using type = float; };

// =====================================================================
// Tiled flash attention kernel (forward pass)
//
// Implements the flash attention algorithm [Dao 2022] with online safe
// softmax and tiled QK^T / PV computation.  All arithmetic that involves
// accumulation (QK^T, softmax, PV) is done in `float` even when the
// storage types are fp16/bf16.
//
// Layout: Q/K/V/Out :: [B, H, S, D]  (row-major contiguous within each
//         batch-head-(seq x d) plane).
// =====================================================================
template <typename T>
void flash_attention_impl(
    sycl::queue& queue,
    const T* q_ptr,
    const T* k_ptr,
    const T* v_ptr,
    T* out_ptr,
    float* softmax_lse_ptr,
    dim_t B, dim_t H_q, dim_t H_kv,
    dim_t S_q, dim_t S_k, dim_t D,
    float scale,
    bool is_causal,
    dim_t window_left,
    dim_t window_right,
    const float* alibi_slopes_ptr) {

  if (B == 0 || H_q == 0 || S_q == 0 || S_k == 0 || D == 0) return;

  using C = float;  // Always compute in fp32.

  // Tile dimensions – tuned for Intel Data Center GPU.
  static constexpr dim_t BLOCK_M = 32;
  static constexpr dim_t BLOCK_N = 64;
  static constexpr size_t WG_X   = 128;  // work-group size

  const dim_t h_h_k_ratio = H_q / H_kv;
  const dim_t num_q_blocks = (S_q + BLOCK_M - 1) / BLOCK_M;
  const dim_t num_k_blocks = (S_k + BLOCK_N - 1) / BLOCK_N;

  const dim_t total_groups = B * H_q * num_q_blocks;
  if (total_groups == 0) return;

  // Upper bound for allocations (max head dimension we support: 256)
  static constexpr dim_t MAX_D = 256;
  static constexpr dim_t SLM_Q   = BLOCK_M * MAX_D;
  static constexpr dim_t SLM_KV  = BLOCK_N * MAX_D;
  static constexpr dim_t SLM_S   = BLOCK_M * BLOCK_N;
  static constexpr dim_t SLM_O   = BLOCK_M * MAX_D;

  queue.submit([&](sycl::handler& cgh) {
    auto slm_q  = sycl::local_accessor<C, 1>(SLM_Q, cgh);
    auto slm_k  = sycl::local_accessor<C, 1>(SLM_KV, cgh);
    auto slm_v  = sycl::local_accessor<C, 1>(SLM_KV, cgh);
    auto slm_s  = sycl::local_accessor<C, 1>(SLM_S, cgh);
    auto slm_o  = sycl::local_accessor<C, 1>(SLM_O, cgh);
    auto slm_m  = sycl::local_accessor<C, 1>(BLOCK_M, cgh);
    auto slm_l  = sycl::local_accessor<C, 1>(BLOCK_M, cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<size_t>(total_groups) * WG_X),
            sycl::range<1>(WG_X)),
        [=](sycl::nd_item<1> item) {
          const size_t group_id = item.get_group_linear_id();
          const size_t tid      = item.get_local_linear_id();

          // Decode (b, h_q, qblk) from the linear group id.
          const size_t batch_head_stride = static_cast<size_t>(H_q) * num_q_blocks;
          const size_t b   = group_id / batch_head_stride;
          const size_t rem = group_id % batch_head_stride;
          const size_t h_q = rem / num_q_blocks;
          const size_t qblk = rem % num_q_blocks;

          const size_t h_kv = h_q / h_h_k_ratio;
          const dim_t q_start = static_cast<dim_t>(qblk) * BLOCK_M;
          const dim_t q_cnt   = sycl::min(BLOCK_M, S_q - q_start);
          const dim_t d       = D;

          // Compute pointer offsets.
          const size_t batch_q_off  = b * static_cast<size_t>(H_q) * S_q * d;
          const size_t batch_k_off  = b * static_cast<size_t>(H_kv) * S_k * d;
          const size_t batch_v_off  = b * static_cast<size_t>(H_kv) * S_k * d;
          const size_t head_q_off   = h_q * S_q * d;
          const size_t head_kv_off  = h_kv * S_k * d;
          const size_t q_blk_off    = head_q_off + static_cast<size_t>(q_start) * d;

          const T* q_slice = q_ptr + batch_q_off + q_blk_off;
          const T* k_slice = k_ptr + batch_k_off + head_kv_off;
          const T* v_slice = v_ptr + batch_v_off + head_kv_off;
          T* out_slice     = out_ptr + batch_q_off + q_blk_off;

          const float alibi_slope = alibi_slopes_ptr ? alibi_slopes_ptr[h_q] : 0.0f;

          // ---------------------------------------------------------------
          // Load Q tile into local memory.
          // ---------------------------------------------------------------
          for (dim_t i = static_cast<dim_t>(tid); i < q_cnt * d; i += static_cast<dim_t>(WG_X))
            slm_q[i] = static_cast<C>(q_slice[i]);
          for (dim_t i = static_cast<dim_t>(tid) + q_cnt * d;
               i < BLOCK_M * d; i += static_cast<dim_t>(WG_X))
            slm_q[i] = C(0);
          item.barrier(sycl::access::fence_space::local_space);

          // ---------------------------------------------------------------
          // Initialize online-softmax state.
          // ---------------------------------------------------------------
          for (dim_t i = static_cast<dim_t>(tid); i < BLOCK_M; i += static_cast<dim_t>(WG_X)) {
            slm_m[i] = -std::numeric_limits<float>::infinity();
            slm_l[i] = C(0);
          }
          for (dim_t i = static_cast<dim_t>(tid); i < BLOCK_M * d; i += static_cast<dim_t>(WG_X))
            slm_o[i] = C(0);
          item.barrier(sycl::access::fence_space::local_space);

          // ---------------------------------------------------------------
          // Main loop over KV tiles.
          // ---------------------------------------------------------------
          for (dim_t kblk = 0; kblk < num_k_blocks; ++kblk) {
            const dim_t kv_start  = kblk * BLOCK_N;
            const dim_t kv_cnt    = sycl::min(BLOCK_N, S_k - kv_start);

            // Load K tile.
            for (dim_t i = static_cast<dim_t>(tid); i < kv_cnt * d; i += static_cast<dim_t>(WG_X)) {
              dim_t row = i / d, col = i % d;
              slm_k[row * d + col] = static_cast<C>(k_slice[(kv_start + row) * d + col]);
            }
            for (dim_t i = static_cast<dim_t>(tid) + kv_cnt * d;
                 i < BLOCK_N * d; i += static_cast<dim_t>(WG_X))
              slm_k[i] = C(0);

            // Load V tile.
            for (dim_t i = static_cast<dim_t>(tid); i < kv_cnt * d; i += static_cast<dim_t>(WG_X)) {
              dim_t row = i / d, col = i % d;
              slm_v[row * d + col] = static_cast<C>(v_slice[(kv_start + row) * d + col]);
            }
            for (dim_t i = static_cast<dim_t>(tid) + kv_cnt * d;
                 i < BLOCK_N * d; i += static_cast<dim_t>(WG_X))
              slm_v[i] = C(0);
            item.barrier(sycl::access::fence_space::local_space);

            // ---- Compute S = (Q * K^T) * scale ----
            for (dim_t idx = static_cast<dim_t>(tid); idx < BLOCK_M * BLOCK_N;
                 idx += static_cast<dim_t>(WG_X)) {
              dim_t qi = idx / BLOCK_N, kj = idx % BLOCK_N;
              C sum = C(0);
              #pragma unroll
              for (dim_t kk = 0; kk < d; ++kk)
                sum += slm_q[qi * d + kk] * slm_k[kj * d + kk];
              sum *= static_cast<C>(scale);

              // Causal mask.
              if (is_causal) {
                dim_t gq = q_start + qi, gk = kv_start + kj;
                if (gk > gq)
                  sum = -std::numeric_limits<float>::infinity();
              }

              // Sliding window mask.
              if (window_left >= 0 || window_right >= 0) {
                dim_t gq = q_start + qi, gk = kv_start + kj;
                if ((window_left >= 0 && gq > gk + static_cast<dim_t>(window_left)) ||
                    (window_right >= 0 && gk > gq + static_cast<dim_t>(window_right)))
                  sum = -std::numeric_limits<float>::infinity();
              }

              // ALiBi bias.
              if (alibi_slopes_ptr) {
                dim_t gq = q_start + qi, gk = kv_start + kj;
                sum += static_cast<C>(alibi_slope *
                    (static_cast<float>(gk) - static_cast<float>(gq)));
              }

              slm_s[idx] = sum;
            }
            item.barrier(sycl::access::fence_space::local_space);

            // ---- Online safe softmax + PV accumulation ----
            for (dim_t qi = static_cast<dim_t>(tid); qi < BLOCK_M;
                 qi += static_cast<dim_t>(WG_X)) {
              if (qi >= q_cnt) continue;

              // Row max of S (valid entries only, not the zero-padded tail).
              C row_max = slm_s[qi * BLOCK_N];
              for (dim_t kj = 1; kj < kv_cnt; ++kj)
                row_max = sycl::max(row_max, slm_s[qi * BLOCK_N + kj]);

              C old_m     = slm_m[qi];
              C new_m     = sycl::max(old_m, row_max);
              C scale_old = sycl::exp(old_m - new_m);

              C row_sum = C(0);
              for (dim_t kj = 0; kj < kv_cnt; ++kj) {
                C p = sycl::exp(slm_s[qi * BLOCK_N + kj] - new_m);
                slm_s[qi * BLOCK_N + kj] = p;
                row_sum += p;
              }
              for (dim_t kj = kv_cnt; kj < BLOCK_N; ++kj)
                slm_s[qi * BLOCK_N + kj] = C(0);

              C new_l = scale_old * slm_l[qi] + row_sum;

              // O = diag(scale_old) * O + P @ V
              for (dim_t kk = 0; kk < d; ++kk) {
                C acc = C(0);
                for (dim_t kj = 0; kj < kv_cnt; ++kj)
                  acc += slm_s[qi * BLOCK_N + kj] * slm_v[kj * d + kk];
                slm_o[qi * d + kk] = scale_old * slm_o[qi * d + kk] + acc;
              }

              slm_m[qi] = new_m;
              slm_l[qi] = new_l;
            }
            item.barrier(sycl::access::fence_space::local_space);
          } // kblk

          // ---- Finalize: O = diag(1/l) * O ----
          for (dim_t qi = static_cast<dim_t>(tid); qi < q_cnt; qi += static_cast<dim_t>(WG_X)) {
            C inv_l = C(1) / slm_l[qi];
            for (dim_t kk = 0; kk < d; ++kk)
              out_slice[qi * d + kk] = static_cast<T>(slm_o[qi * d + kk] * inv_l);
          }

          // Write softmax_lse = m + log(l).
          if (softmax_lse_ptr) {
            size_t lse_off = b * H_q * S_q + h_q * S_q + q_start;
            for (dim_t qi = static_cast<dim_t>(tid); qi < q_cnt; qi += static_cast<dim_t>(WG_X))
              softmax_lse_ptr[lse_off + qi] =
                  static_cast<float>(slm_m[qi] + sycl::log(slm_l[qi]));
          }
        });
  }).wait();
}

// =====================================================================
// Cache path: prefill / decode with KV cache.
//
// Reads KV from two segments: the cache (previously computed tokens)
// and the new tokens passed in the current call.
//
// NOTE: Rotary embeddings are expected to be already applied to `knew`
// by the caller (the `FlashMultiHeadAttention` layer applies rotary
// via `_rotary_embeddings->apply()` before invoking the op).  The
// `rotary_cos`/`rotary_sin` parameters from the CUDA interface are
// therefore ignored in this SYCL implementation.
// =====================================================================
template <typename T>
void flash_attention_cached_impl(
    sycl::queue& queue,
    const T* q_ptr,
    const T* k_cache_ptr,
    const T* v_cache_ptr,
    T* out_ptr,
    float* softmax_lse_ptr,
    dim_t B, dim_t H_q, dim_t H_kv,
    dim_t S_q, dim_t S_k_cache, dim_t D,
    float scale,
    bool is_causal,
    dim_t window_left,
    dim_t window_right,
    const float* alibi_slopes_ptr,
    const T* knew_ptr,
    const T* vnew_ptr,
    dim_t S_new,
    // Rotary embedding params (nullptr = no RoPE).
    // cos/sin each have shape [max_seqlen, D/2] (same element type as KV).
    // Interleave: pair (x[2i], x[2i+1]) rotated by cos/sin[i].
    // Contiguous: pair (x[i], x[i+D/2]) rotated by cos/sin[i].
    const T* rotary_cos = nullptr,
    const T* rotary_sin = nullptr,
    const bool rotary_interleave = false,
    dim_t rotary_offset = 0) {

  if (B == 0 || H_q == 0 || S_q == 0 || D == 0) return;

  using C = float;

  static constexpr dim_t BLOCK_M = 32;
  static constexpr dim_t BLOCK_N = 64;
  static constexpr size_t WG_X   = 128;

  const dim_t h_h_k_ratio = H_q / H_kv;
  const dim_t S_k = S_k_cache + S_new;
  const dim_t num_q_blocks = (S_q + BLOCK_M - 1) / BLOCK_M;
  const dim_t num_k_blocks = (S_k + BLOCK_N - 1) / BLOCK_N;
  const dim_t total_groups = B * H_q * num_q_blocks;
  if (total_groups == 0) return;

  static constexpr dim_t MAX_D = 256;
  static constexpr dim_t SLM_Q  = BLOCK_M * MAX_D;
  static constexpr dim_t SLM_KV = BLOCK_N * MAX_D;
  static constexpr dim_t SLM_S  = BLOCK_M * BLOCK_N;
  static constexpr dim_t SLM_O  = BLOCK_M * MAX_D;

  queue.submit([&](sycl::handler& cgh) {
    auto slm_q  = sycl::local_accessor<C, 1>(SLM_Q, cgh);
    auto slm_k  = sycl::local_accessor<C, 1>(SLM_KV, cgh);
    auto slm_v  = sycl::local_accessor<C, 1>(SLM_KV, cgh);
    auto slm_s  = sycl::local_accessor<C, 1>(SLM_S, cgh);
    auto slm_o  = sycl::local_accessor<C, 1>(SLM_O, cgh);
    auto slm_m  = sycl::local_accessor<C, 1>(BLOCK_M, cgh);
    auto slm_l  = sycl::local_accessor<C, 1>(BLOCK_M, cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<size_t>(total_groups) * WG_X),
            sycl::range<1>(WG_X)),
        [=](sycl::nd_item<1> item) {
          const size_t group_id = item.get_group_linear_id();
          const size_t tid      = item.get_local_linear_id();

          const size_t batch_head_stride = static_cast<size_t>(H_q) * num_q_blocks;
          const size_t b   = group_id / batch_head_stride;
          const size_t rem = group_id % batch_head_stride;
          const size_t h_q = rem / num_q_blocks;
          const size_t qblk = rem % num_q_blocks;

          const size_t h_kv    = h_q / h_h_k_ratio;
          const dim_t q_start  = static_cast<dim_t>(qblk) * BLOCK_M;
          const dim_t q_cnt    = sycl::min(BLOCK_M, S_q - q_start);
          const dim_t d        = D;

          const size_t batch_q_off = b * static_cast<size_t>(H_q) * S_q * d;
          const size_t batch_kc_off = b * static_cast<size_t>(H_kv) * S_k_cache * d;
          const size_t batch_vc_off = b * static_cast<size_t>(H_kv) * S_k_cache * d;
          const size_t batch_kn_off = b * static_cast<size_t>(H_kv) * S_new * d;
          const size_t batch_vn_off = b * static_cast<size_t>(H_kv) * S_new * d;

          const T* q_slice  = q_ptr + batch_q_off + h_q * S_q * d + q_start * d;
          T* out_slice      = out_ptr + batch_q_off + h_q * S_q * d + q_start * d;

          const float alibi_slope = alibi_slopes_ptr ? alibi_slopes_ptr[h_q] : 0.0f;

          // Load Q tile.
          for (dim_t i = static_cast<dim_t>(tid); i < q_cnt * d; i += static_cast<dim_t>(WG_X))
            slm_q[i] = static_cast<C>(q_slice[i]);
          for (dim_t i = static_cast<dim_t>(tid) + q_cnt * d;
               i < BLOCK_M * d; i += static_cast<dim_t>(WG_X))
            slm_q[i] = C(0);

          // Apply RoPE to Q in-place on SLM (cached decode: Q is unrotated).
          if (rotary_cos != nullptr) {
            if (rotary_interleave) {
              for (dim_t qi = static_cast<dim_t>(tid); qi < q_cnt;
                   qi += static_cast<dim_t>(WG_X)) {
                for (dim_t j = 0; j < d; j += 2) {
                  C cos_v = static_cast<C>(
                      rotary_cos[rotary_offset * (d / 2) + j / 2]);
                  C sin_v = static_cast<C>(
                      rotary_sin[rotary_offset * (d / 2) + j / 2]);
                  C a = slm_q[qi * d + j];
                  C b = slm_q[qi * d + j + 1];
                  slm_q[qi * d + j]     = a * cos_v - b * sin_v;
                  slm_q[qi * d + j + 1] = a * sin_v + b * cos_v;
                }
              }
            } else {
              for (dim_t qi = static_cast<dim_t>(tid); qi < q_cnt;
                   qi += static_cast<dim_t>(WG_X)) {
                for (dim_t j = 0; j < d / 2; ++j) {
                  C cos_v = static_cast<C>(
                      rotary_cos[rotary_offset * (d / 2) + j]);
                  C sin_v = static_cast<C>(
                      rotary_sin[rotary_offset * (d / 2) + j]);
                  C a = slm_q[qi * d + j];
                  C b = slm_q[qi * d + j + d / 2];
                  slm_q[qi * d + j]          = a * cos_v - b * sin_v;
                  slm_q[qi * d + j + d / 2]  = a * sin_v + b * cos_v;
                }
              }
            }
          }

          item.barrier(sycl::access::fence_space::local_space);

          // Initialize online-softmax state.
          for (dim_t i = static_cast<dim_t>(tid); i < BLOCK_M; i += static_cast<dim_t>(WG_X)) {
            slm_m[i] = -std::numeric_limits<float>::infinity();
            slm_l[i] = C(0);
          }
          for (dim_t i = static_cast<dim_t>(tid); i < BLOCK_M * d; i += static_cast<dim_t>(WG_X))
            slm_o[i] = C(0);
          item.barrier(sycl::access::fence_space::local_space);

          // Main loop over KV tiles (cache + new combined).
          for (dim_t kblk = 0; kblk < num_k_blocks; ++kblk) {
            const dim_t kv_start = kblk * BLOCK_N;
            const dim_t kv_cnt   = sycl::min(BLOCK_N, S_k - kv_start);

            // Load K/V tile from the appropriate source.
            for (dim_t i = static_cast<dim_t>(tid); i < kv_cnt * d;
                 i += static_cast<dim_t>(WG_X)) {
              dim_t row = i / d, col = i % d;
              dim_t gk = kv_start + row;  // global KV position

              if (gk < S_k_cache) {
                slm_k[row * d + col] = static_cast<C>(
                    k_cache_ptr[batch_kc_off + h_kv * S_k_cache * d + gk * d + col]);
                slm_v[row * d + col] = static_cast<C>(
                    v_cache_ptr[batch_vc_off + h_kv * S_k_cache * d + gk * d + col]);
              } else {
                dim_t new_pos = gk - S_k_cache;
                slm_k[row * d + col] = static_cast<C>(
                    knew_ptr[batch_kn_off + h_kv * S_new * d + new_pos * d + col]);
                slm_v[row * d + col] = static_cast<C>(
                    vnew_ptr[batch_vn_off + h_kv * S_new * d + new_pos * d + col]);
              }
            }
            for (dim_t i = static_cast<dim_t>(tid) + kv_cnt * d;
                 i < BLOCK_N * d; i += static_cast<dim_t>(WG_X)) {
              slm_k[i] = C(0);
              slm_v[i] = C(0);
            }

            // Apply RoPE to new K values in SLM (cached entries already rotated).
            if (rotary_cos != nullptr) {
              if (rotary_interleave) {
                for (dim_t r = static_cast<dim_t>(tid); r < kv_cnt;
                     r += static_cast<dim_t>(WG_X)) {
                  dim_t gk = kv_start + r;
                  if (gk < S_k_cache) continue;  // already rotated in cache
                  dim_t new_pos = gk - S_k_cache;
                  dim_t pos = rotary_offset + new_pos;
                  for (dim_t j = 0; j < d; j += 2) {
                    C cos_v = static_cast<C>(
                        rotary_cos[pos * (d / 2) + j / 2]);
                    C sin_v = static_cast<C>(
                        rotary_sin[pos * (d / 2) + j / 2]);
                    C a = slm_k[r * d + j];
                    C b = slm_k[r * d + j + 1];
                    slm_k[r * d + j]     = a * cos_v - b * sin_v;
                    slm_k[r * d + j + 1] = a * sin_v + b * cos_v;
                  }
                }
              } else {
                for (dim_t r = static_cast<dim_t>(tid); r < kv_cnt;
                     r += static_cast<dim_t>(WG_X)) {
                  dim_t gk = kv_start + r;
                  if (gk < S_k_cache) continue;
                  dim_t new_pos = gk - S_k_cache;
                  dim_t pos = rotary_offset + new_pos;
                  for (dim_t j = 0; j < d / 2; ++j) {
                    C cos_v = static_cast<C>(
                        rotary_cos[pos * (d / 2) + j]);
                    C sin_v = static_cast<C>(
                        rotary_sin[pos * (d / 2) + j]);
                    C a = slm_k[r * d + j];
                    C b = slm_k[r * d + j + d / 2];
                    slm_k[r * d + j]          = a * cos_v - b * sin_v;
                    slm_k[r * d + j + d / 2]  = a * sin_v + b * cos_v;
                  }
                }
              }
            }

            item.barrier(sycl::access::fence_space::local_space);

            // Compute S = Q * K^T.
            for (dim_t idx = static_cast<dim_t>(tid); idx < BLOCK_M * BLOCK_N;
                 idx += static_cast<dim_t>(WG_X)) {
              dim_t qi = idx / BLOCK_N, kj = idx % BLOCK_N;
              C sum = C(0);
              #pragma unroll
              for (dim_t kk = 0; kk < d; ++kk)
                sum += slm_q[qi * d + kk] * slm_k[kj * d + kk];
              sum *= static_cast<C>(scale);

              if (is_causal) {
                dim_t gq = q_start + qi, gk = kv_start + kj;
                if (gk > gq) sum = -std::numeric_limits<float>::infinity();
              }
              if (window_left >= 0 || window_right >= 0) {
                dim_t gq = q_start + qi, gk = kv_start + kj;
                if ((window_left >= 0 && gq > gk + static_cast<dim_t>(window_left)) ||
                    (window_right >= 0 && gk > gq + static_cast<dim_t>(window_right)))
                  sum = -std::numeric_limits<float>::infinity();
              }
              if (alibi_slopes_ptr) {
                dim_t gq = q_start + qi, gk = kv_start + kj;
                sum += static_cast<C>(alibi_slope *
                    (static_cast<float>(gk) - static_cast<float>(gq)));
              }

              slm_s[idx] = sum;
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Online safe softmax + PV.
            for (dim_t qi = static_cast<dim_t>(tid); qi < BLOCK_M;
                 qi += static_cast<dim_t>(WG_X)) {
              if (qi >= q_cnt) continue;

              // Row max of S (valid entries only, not the zero-padded tail).
              C row_max = slm_s[qi * BLOCK_N];
              for (dim_t kj = 1; kj < kv_cnt; ++kj)
                row_max = sycl::max(row_max, slm_s[qi * BLOCK_N + kj]);

              C old_m     = slm_m[qi];
              C new_m     = sycl::max(old_m, row_max);
              C scale_old = sycl::exp(old_m - new_m);

              C row_sum = C(0);
              for (dim_t kj = 0; kj < kv_cnt; ++kj) {
                C p = sycl::exp(slm_s[qi * BLOCK_N + kj] - new_m);
                slm_s[qi * BLOCK_N + kj] = p;
                row_sum += p;
              }
              for (dim_t kj = kv_cnt; kj < BLOCK_N; ++kj)
                slm_s[qi * BLOCK_N + kj] = C(0);

              C new_l = scale_old * slm_l[qi] + row_sum;

              for (dim_t kk = 0; kk < d; ++kk) {
                C acc = C(0);
                for (dim_t kj = 0; kj < kv_cnt; ++kj)
                  acc += slm_s[qi * BLOCK_N + kj] * slm_v[kj * d + kk];
                slm_o[qi * d + kk] = scale_old * slm_o[qi * d + kk] + acc;
              }

              slm_m[qi] = new_m;
              slm_l[qi] = new_l;
            }
            item.barrier(sycl::access::fence_space::local_space);
          }

          // Finalize.
          for (dim_t qi = static_cast<dim_t>(tid); qi < q_cnt; qi += static_cast<dim_t>(WG_X)) {
            C inv_l = C(1) / slm_l[qi];
            for (dim_t kk = 0; kk < d; ++kk)
              out_slice[qi * d + kk] = static_cast<T>(slm_o[qi * d + kk] * inv_l);
          }

          if (softmax_lse_ptr) {
            size_t off = b * H_q * S_q + h_q * S_q + q_start;
            for (dim_t qi = static_cast<dim_t>(tid); qi < q_cnt; qi += static_cast<dim_t>(WG_X))
              softmax_lse_ptr[off + qi] =
                  static_cast<float>(slm_m[qi] + sycl::log(slm_l[qi]));
          }
        });
  }).wait();
}

} // anonymous namespace

// =========================================================================
// Public compute entry point
// =========================================================================
template<>
void FlashAttention::compute<Device::XPU>(
    StorageView& queries,
    StorageView& keys,
    StorageView& values,
    StorageView& output,
    StorageView* cached_keys,
    StorageView* cached_values,
    StorageView* attention,
    bool return_normalized_attention,
    StorageView* rotary_cos,
    StorageView* rotary_sin,
    const bool rotary_interleave,
    StorageView* alibi,
    dim_t offset) const {

  if (return_normalized_attention && attention) {
    throw std::runtime_error(
        "FlashAttention XPU: returning normalized attention is not implemented");
  }

  const Device device = queries.device();
  const DataType dtype = queries.dtype();

  dim_t window_left  = _sliding_window > 0 ? _sliding_window : -1;
  dim_t window_right = _sliding_window > 0 ? 0 : -1;

  const auto& shape = queries.shape();
  const dim_t batch_size  = shape[0];
  dim_t seqlen_q          = shape[1];
  dim_t num_heads         = shape[2];
  const dim_t head_size_og = shape[3];

  dim_t seqlen_k, num_heads_k;
  bool is_cached = (offset > 0);

  if (!is_cached) {
    seqlen_k    = keys.dim(1);
    num_heads_k = keys.dim(2);
    if (window_left >= seqlen_k)  window_left  = -1;
    if (window_right >= seqlen_k) window_right = -1;
  } else {
    seqlen_k    = cached_keys->dim(1);
    num_heads_k = cached_keys->dim(2);
  }

  bool is_causal = _is_causal;
  if (seqlen_q == 1 && !alibi)
    is_causal = false;
  if (is_causal)
    window_right = 0;

  // GQA / MQA: transpose [B, 1, H_q * ngroups, D] -> [B, ngroups, H_kv, D]
  // when seqlen_q == 1 (decode step).  Mirrors the CUDA path exactly.
  static const ops::Transpose transpose_op({0, 2, 1, 3});
  const int seqlenq_ngroups_swapped =
      seqlen_q == 1 && num_heads > num_heads_k
      && window_left < 0 && window_right < 0
      && head_size_og % 8 == 0;

  if (seqlenq_ngroups_swapped) {
    const int ngroups = static_cast<int>(num_heads) / static_cast<int>(num_heads_k);
    StorageView tmp(dtype, device);
    transpose_op(queries.reshape({batch_size, num_heads_k, ngroups, head_size_og}), tmp);
    queries = std::move(tmp);
    seqlen_q  = ngroups;
    num_heads = num_heads_k;
  }

  if (is_cached) {
    if (window_left >= seqlen_k)  window_left  = -1;
    if (window_right >= seqlen_k) window_right = -1;
  }

  // Allocate softmax LSE buffer (matches CUDA interface).
  StorageView softmax_lse(
      {batch_size, num_heads, seqlen_q}, DataType::FLOAT32, device);
  output.resize(queries.shape());

  auto queue = xpu::get_sycl_queue();

  // Type-dispatch lambda.
  auto run = [&](auto type_tag) {
    using T = decltype(type_tag);

    if (!is_cached) {
      flash_attention_impl<T>(
          queue,
          queries.data<T>(),
          keys.data<T>(),
          values.data<T>(),
          output.data<T>(),
          softmax_lse.data<float>(),
          batch_size, num_heads, num_heads_k,
          seqlen_q, seqlen_k, head_size_og,
          _queries_scale,
          is_causal,
          window_left, window_right,
          alibi ? alibi->data<float>() : nullptr);
    } else {
      const dim_t seqlen_knew = keys.dim(1);

      // Fused rotary embeddings: The FlashMultiHeadAttention layer
      // calls _rotary_embeddings->apply() with fa2=true before this op,
      // which returns early when offset>0 (no rotation applied).
      // Therefore the new KV values are UNROTATED in the cached path.
      // We apply the rotary inline during the kernel's SLM load,
      // matching the CUDA flash-attention library's internal rotation.
      // (cos/sin have shape [max_seqlen, head_size/2], type matches T)
      const T* rc = (rotary_cos && rotary_sin)
          ? rotary_cos->data<T>() : nullptr;
      const T* rs = (rotary_cos && rotary_sin)
          ? rotary_sin->data<T>() : nullptr;

      flash_attention_cached_impl<T>(
          queue,
          queries.data<T>(),
          cached_keys->data<T>(),
          cached_values->data<T>(),
          output.data<T>(),
          softmax_lse.data<float>(),
          batch_size, num_heads, num_heads_k,
          seqlen_q, seqlen_k, head_size_og,
          _queries_scale,
          is_causal,
          window_left, window_right,
          alibi ? alibi->data<float>() : nullptr,
          keys.data<T>(),
          values.data<T>(),
          seqlen_knew,
          rc, rs, rotary_interleave, offset);
    }

    // Undo the GQA transpose if applied.
    if (seqlenq_ngroups_swapped) {
      StorageView tmp(dtype, device);
      transpose_op(output, tmp);
      output = std::move(tmp);
      output.reshape(
          {batch_size, 1,
           static_cast<dim_t>(num_heads_k) * static_cast<dim_t>(seqlen_q),
           head_size_og});
    }
  };

  switch (dtype) {
    case DataType::FLOAT32:
      run(float{});
      break;
    case DataType::FLOAT16:
      run(float16_t{});
      break;
    case DataType::BFLOAT16:
      run(bfloat16_t{});
      break;
    default:
      throw std::runtime_error("FlashAttention XPU: unsupported data type");
  }
}

} // namespace ops
} // namespace ctranslate2

#else // CT2_WITH_XPU

#include "dispatch.h"

namespace ctranslate2 {
namespace ops {
template<>
void FlashAttention::compute<Device::XPU>(
    StorageView&, StorageView&, StorageView&, StorageView&,
    StorageView*, StorageView*, StorageView*, bool,
    StorageView*, StorageView*, const bool, StorageView*, dim_t) const {
  throw std::runtime_error("FlashAttention is not supported on XPU");
}
}
}

#endif
