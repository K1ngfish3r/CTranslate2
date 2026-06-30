#include "xpu/utils.h"

#include <dnnl_sycl.hpp>

#include <memory>
#include <vector>

namespace ctranslate2 {
namespace xpu {

  static thread_local int current_device_index = 0;

  dnnl::engine& get_dnnl_engine() {
    static thread_local std::vector<std::unique_ptr<dnnl::engine>> engines;
    if (engines.size() <= static_cast<size_t>(current_device_index)) {
      engines.resize(current_device_index + 1);
    }
    if (!engines[current_device_index]) {
      engines[current_device_index] = std::make_unique<dnnl::engine>(dnnl::engine::kind::gpu, current_device_index);
    }
    return *engines[current_device_index];
  }

  dnnl::stream& get_dnnl_stream() {
    static thread_local std::vector<std::unique_ptr<dnnl::stream>> streams;
    if (streams.size() <= static_cast<size_t>(current_device_index)) {
      streams.resize(current_device_index + 1);
    }
    if (!streams[current_device_index]) {
      streams[current_device_index] = std::make_unique<dnnl::stream>(get_dnnl_engine());
    }
    return *streams[current_device_index];
  }

  int get_gpu_count() {
    try {
      return dnnl::engine::get_count(dnnl::engine::kind::gpu);
    } catch (...) {
      return 0;
    }
  }

  bool has_gpu() {
    return get_gpu_count() > 0;
  }

  int get_device_index() {
    return current_device_index;
  }

  void set_device_index(int index) {
    if (index < 0 || index >= get_gpu_count())
      throw std::invalid_argument("Invalid XPU device index: " + std::to_string(index));
    current_device_index = index;
  }

  void synchronize() {
    get_dnnl_stream().wait();
  }

  sycl::queue get_sycl_queue() {
    return dnnl::sycl_interop::get_queue(get_dnnl_stream());
  }

  dnnl::memory::data_type get_dnnl_dtype(DataType dtype) {
    switch (dtype) {
    case DataType::FLOAT32:
      return dnnl::memory::data_type::f32;
    case DataType::FLOAT16:
      return dnnl::memory::data_type::f16;
    case DataType::BFLOAT16:
      return dnnl::memory::data_type::bf16;
    case DataType::INT8:
      return dnnl::memory::data_type::s8;
    case DataType::INT16:
      throw std::invalid_argument("INT16 is not supported by XPU backend");
    case DataType::INT32:
      return dnnl::memory::data_type::s32;
    default:
      throw std::invalid_argument("unsupported data type");
    }
  }

}
}
