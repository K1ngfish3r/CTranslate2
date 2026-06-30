#include <ctranslate2/ops/awq/gemv.h>

namespace ctranslate2 {
  namespace ops {
    template <Device D, typename In, typename Out>
    void GemvAwq::compute_gemv(const StorageView&,
                               const StorageView&,
                               const StorageView&,
                               const StorageView&,
                               StorageView&) const {
      throw std::runtime_error("AWQ gemv is not supported on XPU");
    }
    template <Device D, typename In, typename Out>
    void GemvAwq::compute_gemv2(const StorageView&,
                                const StorageView&,
                                const StorageView&,
                                const StorageView&,
                                StorageView&) const {
      throw std::runtime_error("AWQ gemv2 is not supported on XPU");
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    GemvAwq::compute_gemv2<Device::XPU, T, int>(                       \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      StorageView&) const;                                              \
    template void                                                       \
    GemvAwq::compute_gemv<Device::XPU, T, int>(                        \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      StorageView&) const;

    DECLARE_IMPL(float16_t)
  }
}
