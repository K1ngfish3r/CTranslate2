#pragma once

#include <dnnl.hpp>
#include <dnnl_sycl.hpp>

#include "ctranslate2/types.h"

namespace ctranslate2 {
namespace xpu {

  dnnl::engine& get_dnnl_engine();
  dnnl::stream& get_dnnl_stream();
  int get_gpu_count();
  bool has_gpu();
  int get_device_index();
  void set_device_index(int index);
  void synchronize();
  sycl::queue get_sycl_queue();
  dnnl::memory::data_type get_dnnl_dtype(DataType dtype);

}
}
