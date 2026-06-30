#include "ctranslate2/ops/topk.h"

#ifdef CT2_WITH_XPU

#include <sycl/sycl.hpp>
#include <dnnl_sycl.hpp>

#include <algorithm>

#include <limits>

#include "ctranslate2/types.h"
#include "xpu/utils.h"

namespace ctranslate2 {
  namespace ops {

    namespace {

      // -----------------------------------------------------------------------
      // SYCL kernel for TopK.
      //
      // Layout:  (batch_size, depth)
      //
      // Each work-group processes one row of the input.
      //
      // Phase 1:  Each work-item scans a strided subset of the row and maintains
      //           a descending sorted list of its local top-k candidates in
      //           work-group local memory.
      //
      // Phase 2:  Work-item 0 merges the per-work-item candidate lists into a
      //           single descending top-k list (insertion-merge into slot 0) and
      //           writes the final result for the row to the output buffers.
      // -----------------------------------------------------------------------
      template <typename T>
      void topk_kernel(sycl::queue& queue,
                       const T* x_data,
                       T* v_data,
                       int32_t* i_data,
                       const dim_t batch_size,
                       const dim_t depth,
                       const dim_t k) {
        if (batch_size <= 0 || depth <= 0 || k <= 0) return;

        // Work-group size – each WG processes one row.
        static constexpr size_t WG = 128;
        // Clamp k so that it never exceeds the number of available elements.
        const dim_t eff_k = std::min(k, depth);

        // Local memory: one top-k candidate list per work-item.
        //   local_vals[tid * eff_k  ..  (tid+1) * eff_k)   – values   (float)
        //   local_idxs[tid * eff_k  ..  (tid+1) * eff_k)   – indices  (int32)
        const size_t entries = WG * static_cast<size_t>(eff_k);

        queue.submit([&](sycl::handler& cgh) {
          auto local_vals = sycl::local_accessor<float>(entries, cgh);
          auto local_idxs = sycl::local_accessor<int32_t>(entries, cgh);

          cgh.parallel_for(
              sycl::nd_range<1>(
                  sycl::range<1>(static_cast<size_t>(batch_size) * WG),
                  sycl::range<1>(WG)),
              [=](sycl::nd_item<1> item) {
                const size_t row = item.get_group_linear_id();
                if (row >= static_cast<size_t>(batch_size)) return;

                const size_t tid = item.get_local_linear_id();
                const size_t row_off = row * static_cast<size_t>(depth);
                const size_t slot = tid * static_cast<size_t>(eff_k);

                // ---- Phase 1: per-work-item strided scan + local top-k ----

                // Initialise with -inf / -1 sentinel.
                for (size_t j = 0; j < static_cast<size_t>(eff_k); ++j) {
                  local_vals[slot + j] = -std::numeric_limits<float>::infinity();
                  local_idxs[slot + j] = -1;
                }

                // Each work-item processes every WG-th element.
                for (size_t j = tid; j < static_cast<size_t>(depth); j += WG) {
                  const float val = x_data[row_off + j];

                  // If the new value is larger than the current k-th best,
                  // insert it into the descending sorted list.
                  if (val > local_vals[slot + static_cast<size_t>(eff_k) - 1]) {
                    size_t pos = static_cast<size_t>(eff_k) - 1;
                    while (pos > 0 && local_vals[slot + pos - 1] < val) {
                      local_vals[slot + pos] = local_vals[slot + pos - 1];
                      local_idxs[slot + pos] = local_idxs[slot + pos - 1];
                      --pos;
                    }
                    local_vals[slot + pos] = val;
                    local_idxs[slot + pos] = static_cast<int32_t>(j);
                  }
                }

                // Ensure all work-items have finished writing before merging.
                item.barrier(sycl::access::fence_space::local_space);

                // ---- Phase 2: merge per-item lists into global result ----
                //
                // Work-item 0 iterates over the candidate lists of all other
                // work-items and insertion-merges them into its own slot (slot 0).
                // No extra scratch memory is needed because insertion shifts
                // elements right-to-left inside slot 0, always reading before
                // overwriting.

                if (tid == 0) {
                  // Slot 0 already holds work-item 0's descending list.
                  // Merge in candidates from work-items 1 .. WG-1.
                  for (size_t t = 1; t < WG; ++t) {
                    const size_t src = t * static_cast<size_t>(eff_k);

                    // The list at src is sorted descending, so once we hit a
                    // value <= the current k-th element, the rest are irrelevant.
                    for (size_t j = 0; j < static_cast<size_t>(eff_k); ++j) {
                      const float val = local_vals[src + j];
                      if (val <= local_vals[static_cast<size_t>(eff_k) - 1])
                        break;

                      // Insert val into the descending list at [0 .. eff_k).
                      size_t pos = static_cast<size_t>(eff_k) - 1;
                      while (pos > 0 && local_vals[pos - 1] < val) {
                        local_vals[pos] = local_vals[pos - 1];
                        local_idxs[pos] = local_idxs[pos - 1];
                        --pos;
                      }
                      local_vals[pos] = val;
                      local_idxs[pos] = local_idxs[src + j];
                    }
                  }

                  // Write final row result to global output.
                  const size_t out_off = row * static_cast<size_t>(eff_k);
                  for (size_t j = 0; j < static_cast<size_t>(eff_k); ++j) {
                    v_data[out_off + j] = local_vals[j];
                    i_data[out_off + j] = local_idxs[j];
                  }
                }
              });
        }).wait();
      }

      // Templated helper to avoid duplicating the compute body for each
      // supported value type.
      template <typename DataType>
      void topk_compute_impl(const StorageView& x,
                             StorageView& values,
                             StorageView& indices,
                             dim_t k) {
        const dim_t depth = x.dim(-1);
        const dim_t batch_size = x.size() / depth;

        const dim_t eff_k = std::min(k, depth);
        if (eff_k != k) {
          values.resize({batch_size, eff_k});
          indices.resize({batch_size, eff_k});
        }

        auto queue = xpu::get_sycl_queue();

        topk_kernel<DataType>(queue,
                              x.data<DataType>(),
                              values.data<DataType>(),
                              indices.data<int32_t>(),
                              batch_size,
                              depth,
                              eff_k);
      }

    }  // anonymous namespace

    template <Device D, typename DataType, typename IndexType>
    void TopK::compute(
        const StorageView& x,
        StorageView& values,
        StorageView& indices) const {
      topk_compute_impl<DataType>(x, values, indices, _k);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    TopK::compute<Device::XPU, T, int32_t>(const StorageView& x,        \
                                           StorageView& values,         \
                                           StorageView& indices) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}

#endif
