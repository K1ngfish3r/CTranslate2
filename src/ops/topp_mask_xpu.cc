#include "ctranslate2/ops/topp_mask.h"

#ifdef CT2_WITH_XPU

#include <sycl/sycl.hpp>
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>

#include "ctranslate2/types.h"
#include "xpu/utils.h"

namespace ctranslate2 {
  namespace ops {

    namespace {

      constexpr dim_t kWorkgroupSize = 256;
      // Each workgroup handles one batch item.
      // We store (prob, original_index) pairs in local memory and sort them.
      // Max items per workgroup: 8192, matching the CUDA kernel capacity.
      // Local memory usage: 8192 * (sizeof(float) + sizeof(int)) = 64 KB.
      constexpr dim_t kMaxItems = kWorkgroupSize * 32;

      static inline dim_t next_power_of_2(dim_t v) {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;
      }

      // SYCL kernel for TopP masking.
      //
      // For each batch item:
      //   1. Load probabilities and their original indices into local memory.
      //   2. Bitonic sort (prob, index) pairs in descending order of probability.
      //   3. Inclusive prefix scan on the sorted probabilities (in-place).
      //   4. Write output: for each item in sorted order, if the cumulative
      //      probability of all more-likely items < p, keep the original logit;
      //      otherwise set it to the mask value.
      template <typename T>
      void topp_mask_kernel(const T* input,
                            const T* probs,
                            T* output,
                            float p,
                            float mask_value,
                            dim_t batch_size,
                            dim_t class_size) {
        auto queue = xpu::get_sycl_queue();

        const dim_t sort_size = next_power_of_2(static_cast<dim_t>(class_size));
        const dim_t local_items = std::min(sort_size, kMaxItems);

        queue.submit([&](sycl::handler& cgh) {
          // Allocate local memory for probabilities and original indices.
          sycl::local_accessor<float, 1> probs_local(kMaxItems, cgh);
          sycl::local_accessor<int, 1> ids_local(kMaxItems, cgh);

          cgh.parallel_for(
              sycl::nd_range<1>(
                  sycl::range<1>(static_cast<size_t>(batch_size) * kWorkgroupSize),
                  sycl::range<1>(kWorkgroupSize)),
              [=](sycl::nd_item<1> item) {
                const dim_t batch_id = static_cast<dim_t>(item.get_group_linear_id());
                const dim_t local_id = static_cast<dim_t>(item.get_local_linear_id());
                const dim_t local_size = static_cast<dim_t>(item.get_local_range(0));

                // Pointers to this batch's data.
                const auto* batch_input = input + batch_id * class_size;
                const auto* batch_probs = probs + batch_id * class_size;
                auto* batch_output = output + batch_id * class_size;

                // Step 1: Load data into local memory.
                // Each thread loads elements: local_id, local_id + local_size, ...
                for (dim_t t = local_id; t < class_size; t += local_size) {
                  probs_local[t] = static_cast<float>(batch_probs[t]);
                  ids_local[t] = static_cast<int>(t);
                }
                // Pad remaining elements with sentinel values so the bitonic sort
                // (which requires a power-of-2 size) works correctly. Sentinels have
                // probability -1.0 so they sort to the end in descending order.
                for (dim_t t = class_size + local_id; t < local_items; t += local_size) {
                  probs_local[t] = -1.0f;
                  ids_local[t] = -1;
                }
                sycl::group_barrier(item.get_group());

                // Step 2: Bitonic sort pairs (probability, original_index) in
                // descending order by probability.
                //
                // Standard bitonic sort where (t & k) == 0 indicates the "ascending"
                // half (which for us means descending order), and (t & k) != 0
                // indicates the "descending" half (ascending order). This alternation
                // builds the bitonic sequence correctly.
                for (dim_t k = 2; k <= local_items; k <<= 1) {
                  for (dim_t j = k >> 1; j > 0; j >>= 1) {
                    for (dim_t t = local_id; t < local_items; t += local_size) {
                      const dim_t partner = t ^ j;
                      if (partner < local_items && t < partner) {
                        const bool direction = ((t & k) == 0);
                        if ((probs_local[t] < probs_local[partner]) == direction) {
                          std::swap(probs_local[t], probs_local[partner]);
                          std::swap(ids_local[t], ids_local[partner]);
                        }
                      }
                    }
                    sycl::group_barrier(item.get_group());
                  }
                }
                // After sorting, indices in [0, class_size) correspond to valid
                // items, sorted by probability descending. Sentinels are at the end.

                // Step 3: In-place inclusive prefix scan on sorted probabilities.
                // Hillis-Steele algorithm: at step `stride`, element t adds the
                // value at t - stride from the previous iteration. Barriers ensure
                // all reads complete before any writes, and all writes are visible
                // before the next step.
                for (dim_t stride = 1; stride < local_items; stride <<= 1) {
                  for (dim_t t = local_id; t < local_items; t += local_size) {
                    if (t >= stride) {
                      probs_local[t] += probs_local[t - stride];
                    }
                  }
                  sycl::group_barrier(item.get_group());
                }
                // probs_local[t] = sum_{i=0}^{t} sorted_prob[i]  (inclusive scan).

                // Step 4: Write output.
                // For sorted position t:
                //   exclusive_sum = sum_{i=0}^{t-1} sorted_prob[i]
                //                  = (t == 0) ? 0 : probs_local[t-1]
                //   If exclusive_sum < p, keep original input, else mask.
                for (dim_t t = local_id; t < class_size; t += local_size) {
                  const float cumulative = (t > 0) ? probs_local[t - 1] : 0.0f;
                  const int original_id = ids_local[t];
                  if (original_id >= 0) {
                    batch_output[original_id] = (cumulative < p)
                        ? batch_input[original_id]
                        : T(mask_value);
                  }
                }
              });
        }).wait();
      }

      // Templated helper to avoid duplicating the compute body for each
      // supported value type.
      template <typename T>
      void topp_mask_compute_impl(const StorageView& input,
                                  const StorageView& probs,
                                  StorageView& output,
                                  float p,
                                  float mask_value) {
        const dim_t depth = input.dim(-1);
        const dim_t batch_size = input.size() / depth;

        if (depth > kMaxItems) {
          throw std::runtime_error(
              "The TopP operator on XPU supports at most "
              + std::to_string(kMaxItems)
              + " classes, but the input has "
              + std::to_string(depth)
              + " classes.");
        }

        topp_mask_kernel<T>(
            input.data<T>(),
            probs.data<T>(),
            output.data<T>(),
            p,
            mask_value,
            batch_size,
            depth);
      }

    }  // anonymous namespace

    template <Device D, typename T>
    void TopPMask::compute(
        const StorageView& input,
        const StorageView& probs,
        StorageView& output) const {
      topp_mask_compute_impl<T>(input, probs, output, _p, _mask_value);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    TopPMask::compute<Device::XPU, T>(const StorageView& input,         \
                                      const StorageView& probs,         \
                                      StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

    template<>
    dim_t TopPMask::max_num_classes<Device::XPU>() {
      return kMaxItems;
    }

  }
}

#endif
