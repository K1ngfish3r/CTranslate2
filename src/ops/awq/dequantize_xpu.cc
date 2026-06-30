#include <ctranslate2/ops/awq/dequantize_awq.h>

namespace ctranslate2 {
  namespace ops {
    template <Device D, typename InT, typename OutT>
    void DequantizeAwq::dequantize(const StorageView&,
                                   const StorageView&,
                                   const StorageView&,
                                   StorageView&) const {
      throw std::runtime_error("AWQ dequantize is not supported on XPU");
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    DequantizeAwq::dequantize<Device::XPU, int, T>(                     \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      StorageView&) const;

    DECLARE_IMPL(float16_t)
  }
}
