#include "ctranslate2/ops/multinomial.h"

#ifdef CT2_WITH_XPU

#include <sycl/sycl.hpp>

#include "ctranslate2/random.h"
#include "xpu/utils.h"

namespace ctranslate2 {
  namespace ops {

    namespace {
      // Helper: multiply two 32-bit integers and return both halves of the
      // 64-bit result.  Used by the Philox round function.
      inline uint32_t mulhilo32(uint32_t a, uint32_t b, uint32_t &lo) {
        uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
        lo = static_cast<uint32_t>(product);
        return static_cast<uint32_t>(product >> 32);
      }

      // Philox-4x32-10: counter-based PRNG that generates a uniform float
      // in (0, 1] for a given (seed, batch_id, call_count) tuple.  This is
      // the same algorithm used by cuRAND on CUDA, providing statistical
      // quality parity with the GPU implementation.
      //
      // The 128-bit counter is composed from batch_id (low 64 bits) and
      // call_count (high 64 bits), ensuring that each (batch, call) pair
      // produces an independent sequence.  The 64-bit key is derived from
      // seed alone.
      inline float philox_uniform(unsigned long long seed,
                                  unsigned long long batch_id,
                                  unsigned long long call_count) {
        // Split inputs into 32-bit halves for the 128-bit counter and 64-bit key.
        uint32_t ctr[4] = {
            static_cast<uint32_t>(batch_id),
            static_cast<uint32_t>(batch_id >> 32),
            static_cast<uint32_t>(call_count),
            static_cast<uint32_t>(call_count >> 32)
        };
        uint32_t key[2] = {
            static_cast<uint32_t>(seed),
            static_cast<uint32_t>(seed >> 32)
        };

        // 10 rounds of Philox.  Every 4th round the key is shedded by
        // adding the golden ratio constant 0x9E3779B9.
        for (int i = 0; i < 10; ++i) {
          uint32_t lo0, lo1;
          uint32_t hi0 = mulhilo32(0xD2511F53, ctr[0], lo0);
          uint32_t hi1 = mulhilo32(0xCD9E8D57, ctr[2], lo1);

          ctr[0] = hi0 ^ ctr[1] ^ key[0];
          ctr[1] = lo0;
          ctr[2] = hi1 ^ ctr[3] ^ key[1];
          ctr[3] = lo1;

          // Key schedule: update every 4th round (after rounds 4 and 8).
          if ((i & 3) == 3) {
            key[0] += 0x9E3779B9;
            key[1] += 0x9E3779B9;
          }
        }

        // Convert the lowest 32 bits of the output to a uniform float in
        // (0, 1] using the same approach as cuRAND.
        constexpr float inv_2pow32 = 2.3283064365386963e-10f;
        float u = static_cast<float>(ctr[0]) * inv_2pow32;
        return (u <= 0.0f) ? 1.17549435e-38f : u;
      }
    }

    constexpr dim_t kMultinomialWgSize = 256;

    // SYCL kernel for multinomial sampling (single sample per batch element).
    //
    // Each work-group processes one batch element.  Within a work-group we:
    //   1. Generate a uniform random sample using the Philox-4x32-10 PRNG
    //      (same algorithm as cuRAND on CUDA), with state advanced across
    //      kernel invocations via a monotonically increasing call_count.
    //   2. Scan the class probabilities in chunks (Kogge-Stone prefix sum in
    //      local memory), tracking the running total across chunks.
    //   3. Record the first index whose cumulative prefix sum crosses the
    //      random threshold.
    //   4. Min-reduce the candidate index across all work-items voting on
    //      the result.
    //
    // The algorithm is a direct SYCL port of the CUDA kernel in
    // multinomial_gpu.cu.
    template <typename In, typename Out>
    static void multinomial_sycl_kernel(
        sycl::queue& queue,
        const In* input,
        dim_t batch_size,
        dim_t class_size,
        Out* output,
        unsigned long long base_seed,
        unsigned long long call_count) {

      if (batch_size == 0 || class_size == 0)
        return;

      const size_t local_size = kMultinomialWgSize;
      const size_t num_groups = static_cast<size_t>(batch_size);

      queue.submit([&](sycl::handler& cgh) {
        // Local memory for the per-chunk prefix sum.
        sycl::local_accessor<float, 1> shared_sum(local_size, cgh);
        // Local memory for the work-group-wide min-reduction of candidate
        // indices.  Out is typically int32_t.
        sycl::local_accessor<Out, 1> shared_cand(local_size, cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(num_groups * local_size),
                sycl::range<1>(local_size)),
            [=](sycl::nd_item<1> item) {
              const size_t batch_id = item.get_group_linear_id();
              const size_t tid = item.get_local_linear_id();

              // 1. One random uniform per batch (so each work-group member
              //    sees the same value).  Use the batch index and call counter
              //    as the Philox counter so different batches and different
              //    invocations get independent draws — matching cuRAND's
              //    persistent per-block state advancement.
              const float random_sample = philox_uniform(
                  base_seed,
                  static_cast<unsigned long long>(batch_id),
                  call_count);

              // 2. Scan chunks and locate the winning index.
              float prefix_base = 0.0f;
              Out candidate = static_cast<Out>(class_size - 1);
              const In* batch_input = input + batch_id * class_size;

              for (dim_t offset = 0; offset < class_size;
                   offset += static_cast<dim_t>(local_size)) {
                const dim_t i = offset + static_cast<dim_t>(tid);

                // Load probability (zero-pad out-of-bounds positions).
                float prob = (i < class_size)
                    ? static_cast<float>(batch_input[i])
                    : 0.0f;
                shared_sum[tid] = prob;
                item.barrier(sycl::access::fence_space::local_space);

                // Kogge-Stone inclusive prefix sum in local memory.
                for (size_t d = 1; d < local_size; d <<= 1) {
                  if (tid >= d) {
                    float prev = shared_sum[tid - d];
                    // Barrier to ensure all reads complete before any write.
                    item.barrier(sycl::access::fence_space::local_space);
                    shared_sum[tid] += prev;
                  }
                  item.barrier(sycl::access::fence_space::local_space);
                }

                // shared_sum[tid] now holds the inclusive prefix sum of this
                // chunk up to position tid.  Add the carry-over from previous
                // chunks to get the true cumulative sum.
                const float cum = prefix_base + shared_sum[tid];

                if (i < class_size
                    && cum >= random_sample
                    && static_cast<dim_t>(i) < static_cast<dim_t>(candidate)) {
                  candidate = static_cast<Out>(i);
                }

                // Carry the chunk's total sum forward.
                prefix_base += shared_sum[local_size - 1];
                item.barrier(sycl::access::fence_space::local_space);
              }

              // 3. Work-group-wide tree min-reduction to obtain the first
              //    (smallest) index that satisfied the threshold condition.
              shared_cand[tid] = candidate;
              item.barrier(sycl::access::fence_space::local_space);

              for (size_t d = local_size / 2; d > 0; d >>= 1) {
                if (tid < d) {
                  shared_cand[tid] =
                      sycl::min(shared_cand[tid], shared_cand[tid + d]);
                }
                item.barrier(sycl::access::fence_space::local_space);
              }

              if (tid == 0) {
                // In the degenerate case where no prefix sum crossed the
                // threshold (e.g. all probabilities are zero), return the
                // last index as a safe default — this matches the behaviour
                // of the CUDA kernel.
                Out result = shared_cand[0];
                output[batch_id] = (result < static_cast<Out>(class_size))
                    ? result
                    : static_cast<Out>(class_size - 1);
              }
            });
      }).wait();
    }

    template <Device D, typename T>
    void Multinomial::compute(
        const StorageView& input,
        StorageView& output) const {

      // The current SYCL kernel only returns a single sample per batch, so
      // fall back on CPU for multi-sample requests (same approach as the
      // CUDA kernel in multinomial_gpu.cu).
      if (_sample_size != 1) {
        StorageView output_host(output.shape(), output.dtype(), Device::CPU);
        dispatch(input.to_float32().to(Device::CPU), output_host);
        output.copy_from(output_host);
        return;
      }

      const dim_t class_size = input.dim(-1);
      const dim_t batch_size = input.size() / class_size;

      if (batch_size == 0 || class_size == 0)
        return;

      auto queue = xpu::get_sycl_queue();

      // Derive a per-op seed from the host RNG so that set_random_seed() is
      // respected and results are reproducible.  The XOR with a constant
      // avoids correlation with other ops that use the same pattern.
      unsigned long long base_seed =
          static_cast<unsigned long long>(get_random_seed());
      base_seed ^= 0x9e3779b97f4a7c15ULL;

      // Thread-local call counter for PRNG state advancement across
      // invocations, replicating the effect of cuRAND's persistent
      // per-block Philox states on CUDA.
      static thread_local unsigned long long call_counter = 0;
      const unsigned long long call_count = call_counter++;

      multinomial_sycl_kernel(
          queue,
          input.data<T>(),
          batch_size,
          class_size,
          output.data<int32_t>(),
          base_seed,
          call_count);
    }

#define DECLARE_IMPL_XPU(T)                                         \
    template void                                                          \
    Multinomial::compute<Device::XPU, T>(                            \
        const StorageView&, StorageView&) const;

    DECLARE_IMPL_XPU(float)
    DECLARE_IMPL_XPU(float16_t)
    DECLARE_IMPL_XPU(bfloat16_t)

  }
}

#endif
