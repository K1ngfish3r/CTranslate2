#include "ctranslate2/ops/gumbel_max.h"

#ifdef CT2_WITH_XPU

#include <sycl/sycl.hpp>

#include "ctranslate2/random.h"
#include "xpu/utils.h"

namespace ctranslate2 {
  namespace ops {

    namespace {
      // Counter-based PRNG: generates a uniform float in (0, 1] for a given
      // (seed, idx) pair using a splitmix64-style hash.  Each element gets an
      // independent random draw without requiring shared PRNG state, which is
      // ideal for SYCL/SIMT architectures.
      inline float sycl_uniform(unsigned long long seed, unsigned long long idx) {
        unsigned long long x = seed + (idx * 0x9e3779b97f4a7c15ULL);
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        // Map to (0, 1] and clamp to stay above zero (log(0) is -inf).
        constexpr double inv_max = 1.0 / 18446744073709551616.0;  // 1.0 / 2^64
        float u = static_cast<float>(static_cast<double>(x) * inv_max);
        return (u <= 0.0f) ? 1.17549435e-38f : u;
      }
    }

    template <Device D, typename T>
    void GumbelMax::add_gumbel_noise(
        const StorageView& x,
        StorageView& y) const {

      const dim_t size = x.size();
      const T* src = x.data<T>();
      T* dst = y.data<T>();

      if (size == 0)
        return;

      auto queue = xpu::get_sycl_queue();

      // Derive a per-op seed from the host RNG so that set_random_seed() is
      // respected and results are reproducible.
      unsigned long long base_seed = static_cast<unsigned long long>(get_random_seed());
      base_seed ^= 0x9e3779b97f4a7c15ULL;

      queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(size)), [=](sycl::id<1> idx) {
          const size_t i = idx[0];
          const float u = sycl_uniform(base_seed, static_cast<unsigned long long>(i));
          dst[i] = static_cast<float>(src[i]) - sycl::log(u);
        });
      }).wait();
    }

#define DECLARE_IMPL_XPU(T)                                         \
    template void                                                          \
    GumbelMax::add_gumbel_noise<Device::XPU, T>(                     \
        const StorageView&, StorageView&) const;

    DECLARE_IMPL_XPU(float)
    DECLARE_IMPL_XPU(float16_t)
    DECLARE_IMPL_XPU(bfloat16_t)

  }
}

#endif
