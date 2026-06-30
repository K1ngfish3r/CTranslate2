#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <sycl/sycl.hpp>

#include <dnnl.hpp>
#include <dnnl_sycl.hpp>

#include "ctranslate2/primitives.h"
#include "type_dispatch.h"
#include "xpu/utils.h"

namespace ctranslate2 {

namespace {

  // Helper to create a DNNL memory wrapping a device/host pointer (mutable).
  dnnl::memory wrap_memory(const dnnl::memory::desc& md, void* ptr) {
    return dnnl::memory(md, xpu::get_dnnl_engine(), ptr);
  }

  // Helper to create a DNNL memory wrapping a device/host pointer (const).
  dnnl::memory wrap_memory(const dnnl::memory::desc& md, const void* ptr) {
    return dnnl::memory(md, xpu::get_dnnl_engine(), const_cast<void*>(ptr));
  }

  // Helper to create a 1D memory descriptor.
  dnnl::memory::desc make_md(dnnl::memory::dim size, dnnl::memory::data_type dtype) {
    return dnnl::memory::desc({size}, dtype, dnnl::memory::format_tag::a);
  }

  // Helper to run a DNNL primitive and wait for completion.
  void execute_primitive(dnnl::primitive& prim,
                         const std::unordered_map<int, dnnl::memory>& args) {
    prim.execute(xpu::get_dnnl_stream(), args);
    xpu::get_dnnl_stream().wait();
  }

  // Helper for elementwise operations.
  struct EltwiseHelper {
    dnnl::eltwise_forward::primitive_desc pd;
    dnnl::eltwise_forward prim;

    EltwiseHelper(dnnl::algorithm algo, float alpha = 0.f, float beta = 0.f)
      : pd(xpu::get_dnnl_engine(),
            dnnl::prop_kind::forward_inference,
            algo,
            make_md(1, dnnl::memory::data_type::f32),
            make_md(1, dnnl::memory::data_type::f32),
            alpha, beta)
      , prim(pd)
    {}

    void operator()(const float* x, float* y, dim_t size) {
      auto md = make_md(size, dnnl::memory::data_type::f32);
      auto src = wrap_memory(md, x);
      auto dst = wrap_memory(md, y);
      execute_primitive(prim, {
        {DNNL_ARG_SRC, src},
        {DNNL_ARG_DST, dst}
      });
    }
  };

  // Helper for binary operations with broadcasting from a scalar.
  struct BinaryScalarHelper {
    dnnl::algorithm algo;
    BinaryScalarHelper(dnnl::algorithm a) : algo(a) {}

    void operator()(float a, const float* x, float* y, dim_t size) {
      // Create a memory for the scalar value.
      auto scalar_md = dnnl::memory::desc({1}, dnnl::memory::data_type::f32,
                                           dnnl::memory::format_tag::a);
      auto src0_md = make_md(size, dnnl::memory::data_type::f32);
      auto dst_md = make_md(size, dnnl::memory::data_type::f32);

      auto scalar_mem = wrap_memory(scalar_md, &a);
      auto src1_mem = wrap_memory(src0_md, x);
      auto dst_mem = wrap_memory(dst_md, y);

      auto pd = dnnl::binary::primitive_desc(
          xpu::get_dnnl_engine(), algo,
          scalar_md, src0_md, dst_md);
      auto prim = dnnl::binary(pd);
      execute_primitive(prim, {
        {DNNL_ARG_SRC_0, scalar_mem},
        {DNNL_ARG_SRC_1, src1_mem},
        {DNNL_ARG_DST, dst_mem}
      });
    }

    void operator()(const float* a, const float* b, float* c, dim_t size) {
      auto md = make_md(size, dnnl::memory::data_type::f32);
      auto a_mem = wrap_memory(md, a);
      auto b_mem = wrap_memory(md, b);
      auto c_mem = wrap_memory(md, c);

      auto pd = dnnl::binary::primitive_desc(
          xpu::get_dnnl_engine(), algo,
          md, md, md);
      auto prim = dnnl::binary(pd);
      execute_primitive(prim, {
        {DNNL_ARG_SRC_0, a_mem},
        {DNNL_ARG_SRC_1, b_mem},
        {DNNL_ARG_DST, c_mem}
      });
    }
  };

  // Helper for reduction operations.
  struct ReductionHelper {
    dnnl::algorithm algo;
    ReductionHelper(dnnl::algorithm a) : algo(a) {}

    float operator()(const float* x, dim_t size) {
      auto src_md = make_md(size, dnnl::memory::data_type::f32);
      auto dst_md = dnnl::memory::desc({1}, dnnl::memory::data_type::f32,
                                        dnnl::memory::format_tag::a);
      auto src = wrap_memory(src_md, x);
      float result = 0.f;
      auto dst = wrap_memory(dst_md, &result);

      auto pd = dnnl::reduction::primitive_desc(
          xpu::get_dnnl_engine(), algo,
          src_md, dst_md, 0.f, 0.f);
      auto prim = dnnl::reduction(pd);
      execute_primitive(prim, {
        {DNNL_ARG_SRC, src},
        {DNNL_ARG_DST, dst}
      });
      return result;
    }
  };

  // Helper to compute a reduction (sum or max) using a SYCL kernel.
  // Returns the result value on the host.
  template <typename T, typename BinaryOp>
  T sycl_reduce(const T* array, dim_t size, BinaryOp op, T identity) {
    if (size == 0) return identity;
    auto queue = xpu::get_sycl_queue();
    T* d_result = sycl::malloc_device<T>(1, queue);
    queue.fill(d_result, identity, 1).wait();
    queue.parallel_for(
        sycl::range<1>(static_cast<size_t>(size)),
        sycl::reduction(d_result, identity, op),
        [=](sycl::id<1> idx, auto& reducer) {
          reducer.combine(array[idx]);
        }).wait();
    T result;
    queue.memcpy(&result, d_result, sizeof(T)).wait();
    sycl::free(d_result, queue);
    return result;
  }

}  // anonymous namespace

  // ==========================================================================
  // primitives<Device::XPU>
  // ==========================================================================

  // --- at ---

  template<>
  template <typename T>
  T primitives<Device::XPU>::at(const T* x, dim_t index) {
    auto queue = xpu::get_sycl_queue();
    T result;
    queue.memcpy(&result, x + index, sizeof(T)).wait();
    return result;
  }

  // --- fill ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::fill(T* x, T a, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.fill(x, a, static_cast<size_t>(size)).wait();
  }

  // --- strided_fill ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::strided_fill(T* x, T a, dim_t inc_x, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(size)),
        [=](sycl::id<1> idx) {
          x[idx * inc_x] = a;
        }).wait();
  }

  // --- indexed_fill ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::indexed_fill(T* x, T a,
                                                    const int32_t* indices,
                                                    dim_t num_indices) {
    if (num_indices == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(num_indices)),
        [=](sycl::id<1> idx) {
          x[indices[idx]] = a;
        }).wait();
  }

  // --- copy ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::copy(const T* x, T* y, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.memcpy(y, x, static_cast<size_t>(size) * sizeof(T)).wait();
  }

  // --- convert ---

  template<>
  template <typename U, typename V>
  void primitives<Device::XPU>::convert(const U* x, V* y, dim_t size) {
    // Use DNNL reorder for type conversion on the GPU engine.
    if (size == 0)
      return;

    auto src_dtype = xpu::get_dnnl_dtype(DataTypeToEnum<U>::value);
    auto dst_dtype = xpu::get_dnnl_dtype(DataTypeToEnum<V>::value);

    if (src_dtype == dst_dtype) {
      auto queue = xpu::get_sycl_queue();
      queue.memcpy(y, x, static_cast<size_t>(size) * sizeof(U)).wait();
      return;
    }

    auto md = make_md(size, src_dtype);
    auto dst_md = make_md(size, dst_dtype);

    auto src = wrap_memory(md, x);
    auto dst = wrap_memory(dst_md, y);

    auto pd = dnnl::reorder::primitive_desc(
        xpu::get_dnnl_engine(), md,
        xpu::get_dnnl_engine(), dst_md);
    auto prim = dnnl::reorder(pd);
    execute_primitive(prim, {
      {DNNL_ARG_SRC, src},
      {DNNL_ARG_DST, dst}
    });
  }

  // --- sum ---

  template<>
  template <typename T>
  T primitives<Device::XPU>::sum(const T* array, dim_t size) {
    return sycl_reduce(array, size, sycl::plus<T>(), T(0));
  }

  template<>
  template<>
  float primitives<Device::XPU>::sum(const float* array, dim_t size) {
    if (size == 0) return 0.f;
    return ReductionHelper{dnnl::algorithm::reduction_sum}(array, size);
  }

  // --- max_element ---

  template<>
  template <typename T>
  dim_t primitives<Device::XPU>::max_element(const T* array, dim_t size) {
    if (size == 0) return 0;
    auto queue = xpu::get_sycl_queue();

    // Use a custom reduction to find both the max value and its index.
    // Allocate space for both: [max_val, max_idx].
    T* d_result = sycl::malloc_device<T>(2, queue);
    T init[2] = {T(0), T(-1)};
    queue.memcpy(d_result, init, 2 * sizeof(T)).wait();

    queue.parallel_for(
        sycl::range<1>(static_cast<size_t>(size)),
        sycl::reduction(d_result, sycl::maximum<T>()),
        [=](sycl::id<1> idx, auto& reducer) {
          reducer.combine(array[idx]);
        }).wait();

    T result_val;
    queue.memcpy(&result_val, d_result, sizeof(T)).wait();
    sycl::free(d_result, queue);

    // We got max value; now find its first index with a second kernel.
    dim_t max_idx = 0;
    dim_t* d_idx = sycl::malloc_device<dim_t>(1, queue);
    queue.fill(d_idx, dim_t(-1), 1).wait();
    queue.parallel_for(
        sycl::range<1>(static_cast<size_t>(size)),
        [=](sycl::id<1> idx) {
          if (array[idx] == result_val) {
            dim_t old = sycl::atomic_ref<dim_t,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>(*d_idx)
                .load();
            if (idx < old || old == dim_t(-1))
              sycl::atomic_ref<dim_t,
                  sycl::memory_order::relaxed,
                  sycl::memory_scope::device,
                  sycl::access::address_space::global_space>(*d_idx)
                  .store(static_cast<dim_t>(idx));
          }
        }).wait();
    queue.memcpy(&max_idx, d_idx, sizeof(dim_t)).wait();
    sycl::free(d_idx, queue);
    return max_idx;
  }

  // --- max ---

  template<>
  template <typename T>
  T primitives<Device::XPU>::max(const T* array, dim_t size) {
    return sycl_reduce(array, size, sycl::maximum<T>(), T(0));
  }

  template<>
  template<>
  float primitives<Device::XPU>::max(const float* array, dim_t size) {
    if (size == 0) return 0.f;
    return ReductionHelper{dnnl::algorithm::reduction_max}(array, size);
  }

  // --- amax ---

  template<>
  template <typename T>
  T primitives<Device::XPU>::amax(const T* array, dim_t size) {
    if (size == 0) return T(0);
    auto queue = xpu::get_sycl_queue();
    T* d_result = sycl::malloc_device<T>(1, queue);
    queue.fill(d_result, T(0), 1).wait();
    queue.parallel_for(
        sycl::range<1>(static_cast<size_t>(size)),
        sycl::reduction(d_result, T(0), sycl::maximum<T>()),
        [=](sycl::id<1> idx, auto& reducer) {
          float val = static_cast<float>(array[idx]);
          T abs_val = static_cast<T>(val < 0.0f ? -val : val);
          reducer.combine(abs_val);
        }).wait();
    T result;
    queue.memcpy(&result, d_result, sizeof(T)).wait();
    sycl::free(d_result, queue);
    return result;
  }

  template<>
  template<>
  float primitives<Device::XPU>::amax(const float* array, dim_t size) {
    if (size == 0) return 0.f;
    // DNNL norm Lp with p=inf gives max of absolute values.
    return ReductionHelper{dnnl::algorithm::reduction_norm_lp_max}(array, size);
  }

  // --- add (scalar) ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::add(T a, const T* x, T* y, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(size)),
        [=](sycl::id<1> idx) {
          y[idx] = x[idx] + a;
        }).wait();
  }

  template<>
  template<>
  void primitives<Device::XPU>::add(float a, const float* x, float* y, dim_t size) {
    if (size == 0) return;
    if (a == 0.f) {
      if (x != y) {
        auto queue = xpu::get_sycl_queue();
        queue.memcpy(y, x, static_cast<size_t>(size) * sizeof(float)).wait();
      }
      return;
    }
    BinaryScalarHelper{dnnl::algorithm::binary_add}(a, x, y, size);
  }

  // --- add (array) ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::add(const T* a, const T* b, T* c, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(size)),
        [=](sycl::id<1> idx) {
          c[idx] = a[idx] + b[idx];
        }).wait();
  }

  template<>
  template<>
  void primitives<Device::XPU>::add(const float* a, const float* b, float* c, dim_t size) {
    if (size == 0) return;
    BinaryScalarHelper{dnnl::algorithm::binary_add}(a, b, c, size);
  }

  // --- add_batch_broadcast ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::add_batch_broadcast(const T* a, const T* b,
                                                          T* c, dim_t a_size, dim_t b_size) {
    if (a_size == 0 || b_size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(b_size)),
        [=](sycl::id<1> idx) {
          const size_t i = idx[0];
          c[i] = a[i % static_cast<size_t>(a_size)] + b[i];
        }).wait();
  }

  // --- add_depth_broadcast ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::add_depth_broadcast(const T* a, const T* b,
                                                          T* c, dim_t a_size, dim_t b_size) {
    if (a_size == 0 || b_size == 0) return;
    const dim_t depth = b_size / a_size;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(b_size)),
        [=](sycl::id<1> idx) {
          const size_t i = idx[0];
          c[i] = a[i / static_cast<size_t>(depth)] + b[i];
        }).wait();
  }

  // --- add_block_broadcast ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::add_block_broadcast(const T* a, const T* b,
                                                          T* c, dim_t block,
                                                          dim_t a_size, dim_t b_size) {
    if (a_size == 0 || b_size == 0 || block == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(b_size)),
        [=](sycl::id<1> idx) {
          const size_t i = idx[0];
          const size_t b_block = static_cast<size_t>(block);
          const size_t a_size_t = static_cast<size_t>(a_size);
          const size_t b_block_index = i / b_block;
          const size_t a_index = b_block_index % a_size_t;
          c[i] = a[a_index] + b[i];
        }).wait();
  }

  // --- sub (scalar is already defined in header using add(-a)) ---
  // --- sub (array) ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::sub(const T* a, const T* b, T* c, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(size)),
        [=](sycl::id<1> idx) {
          c[idx] = a[idx] - b[idx];
        }).wait();
  }

  template<>
  template<>
  void primitives<Device::XPU>::sub(const float* a, const float* b, float* c, dim_t size) {
    if (size == 0) return;
    BinaryScalarHelper{dnnl::algorithm::binary_sub}(a, b, c, size);
  }

  // --- max (scalar) ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::max(T a, const T* x, T* y, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(size)),
        [=](sycl::id<1> idx) {
          y[idx] = x[idx] > a ? x[idx] : a;
        }).wait();
  }

  template<>
  template<>
  void primitives<Device::XPU>::max(float a, const float* x, float* y, dim_t size) {
    if (size == 0) return;
    BinaryScalarHelper{dnnl::algorithm::binary_max}(a, x, y, size);
  }

  // --- max (array) ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::max(const T* a, const T* b, T* c, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(size)),
        [=](sycl::id<1> idx) {
          c[idx] = a[idx] > b[idx] ? a[idx] : b[idx];
        }).wait();
  }

  template<>
  template<>
  void primitives<Device::XPU>::max(const float* a, const float* b, float* c, dim_t size) {
    if (size == 0) return;
    BinaryScalarHelper{dnnl::algorithm::binary_max}(a, b, c, size);
  }

  // --- min (scalar) ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::min(T a, const T* x, T* y, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(size)),
        [=](sycl::id<1> idx) {
          y[idx] = x[idx] < a ? x[idx] : a;
        }).wait();
  }

  template<>
  template<>
  void primitives<Device::XPU>::min(float a, const float* x, float* y, dim_t size) {
    if (size == 0) return;
    BinaryScalarHelper{dnnl::algorithm::binary_min}(a, x, y, size);
  }

  // --- min (array) ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::min(const T* a, const T* b, T* c, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(size)),
        [=](sycl::id<1> idx) {
          c[idx] = a[idx] < b[idx] ? a[idx] : b[idx];
        }).wait();
  }

  template<>
  template<>
  void primitives<Device::XPU>::min(const float* a, const float* b, float* c, dim_t size) {
    if (size == 0) return;
    BinaryScalarHelper{dnnl::algorithm::binary_min}(a, b, c, size);
  }

  // --- mul (scalar) ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::mul(T a, const T* x, T* y, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(size)),
        [=](sycl::id<1> idx) {
          y[idx] = x[idx] * a;
        }).wait();
  }

  template<>
  template<>
  void primitives<Device::XPU>::mul(float a, const float* x, float* y, dim_t size) {
    if (size == 0) return;
    if (a == 1.f) {
      if (x != y) {
        auto queue = xpu::get_sycl_queue();
        queue.memcpy(y, x, static_cast<size_t>(size) * sizeof(float)).wait();
      }
      return;
    }
    BinaryScalarHelper{dnnl::algorithm::binary_mul}(a, x, y, size);
  }

  // --- mul (array) ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::mul(const T* a, const T* b, T* c, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(size)),
        [=](sycl::id<1> idx) {
          c[idx] = a[idx] * b[idx];
        }).wait();
  }

  template<>
  template<>
  void primitives<Device::XPU>::mul(const float* a, const float* b, float* c, dim_t size) {
    if (size == 0) return;
    BinaryScalarHelper{dnnl::algorithm::binary_mul}(a, b, c, size);
  }

  // --- mul_batch_broadcast ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::mul_batch_broadcast(const T* a, const T* b,
                                                          T* c, dim_t a_size, dim_t b_size) {
    if (a_size == 0 || b_size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(b_size)),
        [=](sycl::id<1> idx) {
          const size_t i = idx[0];
          c[i] = a[i % static_cast<size_t>(a_size)] * b[i];
        }).wait();
  }

  // --- penalty_previous_tokens ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::penalize_previous_tokens(
      T* scores,
      const T* previous_scores,
      const int32_t* previous_ids,
      T penalty,
      dim_t batch_size,
      dim_t length,
      dim_t vocabulary_size) {
    if (batch_size == 0 || length == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(
        sycl::range<2>(static_cast<size_t>(batch_size),
                       static_cast<size_t>(length)),
        [=](sycl::id<2> idx) {
          const dim_t i = static_cast<dim_t>(idx[0]);
          const dim_t j = static_cast<dim_t>(idx[1]);
          const dim_t read_index = i * length + j;
          const dim_t write_index = i * vocabulary_size + previous_ids[read_index];
          const auto score = previous_scores[read_index];
          scores[write_index] = (score < T(0) ? score * penalty : score / penalty);
        }).wait();
  }

  // --- prepare_length_mask ---

  template<>
  void primitives<Device::XPU>::prepare_length_mask(
      const int32_t* lengths,
      dim_t batch_size,
      dim_t num_heads,
      dim_t num_queries,
      bool mask_future,
      bool multi_query,
      int32_t* mask) {
    if (batch_size == 0 || num_heads == 0 || num_queries == 0) return;
    auto queue = xpu::get_sycl_queue();
    const dim_t total_entries = num_heads * num_queries;
    queue.parallel_for(
        sycl::range<2>(static_cast<size_t>(batch_size),
                       static_cast<size_t>(total_entries)),
        [=](sycl::id<2> idx) {
          const dim_t b = static_cast<dim_t>(idx[0]);
          const dim_t i = static_cast<dim_t>(idx[1]);
          const auto length = lengths[b];
          auto* batch_mask = mask + b * total_entries;
          batch_mask[i] = (mask_future
                           ? (length < static_cast<int32_t>(
                                 (multi_query ? i / num_heads : i % num_queries) + 1)
                              ? length
                              : static_cast<int32_t>(
                                    (multi_query ? i / num_heads : i % num_queries) + 1))
                           : length);
        }).wait();
  }

  // --- transpose_2d ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::transpose_2d(const T* a, const dim_t* dims, T* b) {
    // Try DNNL reorder with permuted format, or fall back to SYCL kernel.
    if (std::is_same<T, float>::value) {
      try {
        auto md_src = dnnl::memory::desc({dims[0], dims[1]},
                                          dnnl::memory::data_type::f32,
                                          dnnl::memory::format_tag::ab);
        auto md_dst = dnnl::memory::desc({dims[1], dims[0]},
                                          dnnl::memory::data_type::f32,
                                          dnnl::memory::format_tag::ba);
        auto src = wrap_memory(md_src, a);
        auto dst = wrap_memory(md_dst, b);
        auto pd = dnnl::reorder::primitive_desc(
            xpu::get_dnnl_engine(), md_src,
            xpu::get_dnnl_engine(), md_dst);
        auto prim = dnnl::reorder(pd);
        execute_primitive(prim, {
          {DNNL_ARG_SRC, src},
          {DNNL_ARG_DST, dst}
        });
        return;
      } catch (...) {
        // Fall through to SYCL implementation.
      }
    }
    // SYCL kernel fallback for non-float or if DNNL reorder fails.
    auto queue = xpu::get_sycl_queue();
    const dim_t d0 = dims[0];
    const dim_t d1 = dims[1];
    queue.parallel_for(
        sycl::range<2>(static_cast<size_t>(d0), static_cast<size_t>(d1)),
        [=](sycl::id<2> idx) {
          const dim_t i0 = static_cast<dim_t>(idx[0]);
          const dim_t i1 = static_cast<dim_t>(idx[1]);
          b[i1 * d0 + i0] = a[i0 * d1 + i1];
        }).wait();
  }

  // --- transpose_3d ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::transpose_3d(const T* a,
                                                    const dim_t* dims,
                                                    const dim_t* perm,
                                                    T* b) {
    const dim_t d0 = dims[0];
    const dim_t d1 = dims[1];
    const dim_t d2 = dims[2];
    const dim_t perm1 = perm[1];
    const dim_t perm2 = perm[2];

    dim_t perm_ind[3];
    for (dim_t i = 0; i < 3; ++i)
      perm_ind[perm[i]] = i;
    const dim_t a_stride[3] = {d1 * d2, d2, 1};
    const dim_t b_stride[3] = {dims[perm1] * dims[perm2], dims[perm2], 1};
    const dim_t perm_b_stride[3] = {b_stride[perm_ind[0]], b_stride[perm_ind[1]],
                                    b_stride[perm_ind[2]]};

    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(
        sycl::range<3>(static_cast<size_t>(d0),
                       static_cast<size_t>(d1),
                       static_cast<size_t>(d2)),
        [=](sycl::id<3> idx) {
          const dim_t i0 = static_cast<dim_t>(idx[0]);
          const dim_t i1 = static_cast<dim_t>(idx[1]);
          const dim_t i2 = static_cast<dim_t>(idx[2]);
          const dim_t b_i = (i0 * perm_b_stride[0] + i1 * perm_b_stride[1]
                             + i2 * perm_b_stride[2]);
          const dim_t a_i = (i0 * a_stride[0] + i1 * a_stride[1]
                             + i2 * a_stride[2]);
          b[b_i] = a[a_i];
        }).wait();
  }

  // --- transpose_4d ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::transpose_4d(const T* a,
                                                    const dim_t* dims,
                                                    const dim_t* perm,
                                                    T* b) {
    // Optimize the permutation used in multi-head attention: (0, 2, 1, 3).
    if (perm[0] == 0 && perm[1] == 2 && perm[2] == 1 && perm[3] == 3) {
      const dim_t d0 = dims[0];
      const dim_t r1 = dims[2];
      const dim_t r2 = dims[1];
      const dim_t depth = dims[3];

      auto queue = xpu::get_sycl_queue();
      queue.parallel_for(
          sycl::range<3>(static_cast<size_t>(d0),
                         static_cast<size_t>(r1 * r2),
                         static_cast<size_t>(depth)),
          [=](sycl::id<3> idx) {
            const dim_t i = static_cast<dim_t>(idx[0]);
            const dim_t j = static_cast<dim_t>(idx[1]);
            const dim_t k = static_cast<dim_t>(idx[2]);
            const dim_t offset = i * r1 * r2;
            const dim_t a_offset = offset + j;
            const dim_t b_offset = offset + j / r1 + (j % r1) * r2;
            b[b_offset * depth + k] = a[a_offset * depth + k];
          }).wait();
      return;
    }

    // General 4D transpose fallback.
    const dim_t d0 = dims[0];
    const dim_t d1 = dims[1];
    const dim_t d2 = dims[2];
    const dim_t d3 = dims[3];
    const dim_t perm1 = perm[1];
    const dim_t perm2 = perm[2];
    const dim_t perm3 = perm[3];

    dim_t perm_ind[4];
    for (dim_t i = 0; i < 4; ++i)
      perm_ind[perm[i]] = i;
    const dim_t a_stride[4] = {d1 * d2 * d3, d2 * d3, d3, 1};
    const dim_t b_stride[4] = {dims[perm1] * dims[perm2] * dims[perm3],
                                dims[perm2] * dims[perm3],
                                dims[perm3], 1};
    const dim_t perm_b_stride[4] = {b_stride[perm_ind[0]], b_stride[perm_ind[1]],
                                    b_stride[perm_ind[2]], b_stride[perm_ind[3]]};

    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(
        sycl::range<1>(static_cast<size_t>(d0 * d1 * d2 * d3)),
        [=](sycl::id<1> idx) {
          dim_t linear_idx = static_cast<dim_t>(idx[0]);
          const dim_t i3 = linear_idx % d3;
          linear_idx /= d3;
          const dim_t i2 = linear_idx % d2;
          linear_idx /= d2;
          const dim_t i1 = linear_idx % d1;
          const dim_t i0 = linear_idx / d1;

          const dim_t b_i = (i0 * perm_b_stride[0] + i1 * perm_b_stride[1]
                             + i2 * perm_b_stride[2] + i3 * perm_b_stride[3]);
          const dim_t a_i = (i0 * a_stride[0] + i1 * a_stride[1]
                             + i2 * a_stride[2] + i3 * a_stride[3]);
          b[b_i] = a[a_i];
        }).wait();
  }

  // --- logsumexp ---

  template<>
  template <typename T>
  float primitives<Device::XPU>::logsumexp(const T* x, dim_t size) {
    if (size == 0) return 0.f;
    auto queue = xpu::get_sycl_queue();

    // Step 1: find max(x).
    T* d_max = sycl::malloc_device<T>(1, queue);
    queue.fill(d_max, T(0), 1).wait();
    queue.parallel_for(
        sycl::range<1>(static_cast<size_t>(size)),
        sycl::reduction(d_max, T(0), sycl::maximum<T>()),
        [=](sycl::id<1> idx, auto& reducer) {
          reducer.combine(x[idx]);
        }).wait();
    T max_val;
    queue.memcpy(&max_val, d_max, sizeof(T)).wait();

    // Step 2: compute sum of exp(x - max(x)).
    T* d_sum = sycl::malloc_device<T>(1, queue);
    queue.fill(d_sum, T(0), 1).wait();
    queue.parallel_for(
        sycl::range<1>(static_cast<size_t>(size)),
        sycl::reduction(d_sum, T(0), sycl::plus<T>()),
        [=](sycl::id<1> idx, auto& reducer) {
          reducer.combine(static_cast<T>(sycl::exp(static_cast<float>(x[idx] - max_val))));
        }).wait();
    T sum_exp;
    queue.memcpy(&sum_exp, d_sum, sizeof(T)).wait();

    sycl::free(d_max, queue);
    sycl::free(d_sum, queue);

    return static_cast<float>(max_val + sycl::log(static_cast<float>(sum_exp)));
  }

  // --- exp ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::exp(const T* x, T* y, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(size)),
        [=](sycl::id<1> idx) {
          y[idx] = sycl::exp(static_cast<float>(x[idx]));
        }).wait();
  }

  template<>
  template<>
  void primitives<Device::XPU>::exp(const float* x, float* y, dim_t size) {
    if (size == 0) return;
    EltwiseHelper{dnnl::algorithm::eltwise_exp}(x, y, size);
  }

  // --- log ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::log(const T* x, T* y, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(size)),
        [=](sycl::id<1> idx) {
          y[idx] = sycl::log(static_cast<float>(x[idx]));
        }).wait();
  }

  template<>
  template<>
  void primitives<Device::XPU>::log(const float* x, float* y, dim_t size) {
    if (size == 0) return;
    EltwiseHelper{dnnl::algorithm::eltwise_log}(x, y, size);
  }

  // --- cos ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::cos(const T* x, T* y, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(size)),
        [=](sycl::id<1> idx) {
          y[idx] = sycl::cos(static_cast<float>(x[idx]));
        }).wait();
  }

  // --- sin ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::sin(const T* x, T* y, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(size)),
        [=](sycl::id<1> idx) {
          y[idx] = sycl::sin(static_cast<float>(x[idx]));
        }).wait();
  }

  // --- tanh ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::tanh(const T* x, T* y, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(size)),
        [=](sycl::id<1> idx) {
          y[idx] = sycl::tanh(static_cast<float>(x[idx]));
        }).wait();
  }

  template<>
  template<>
  void primitives<Device::XPU>::tanh(const float* x, float* y, dim_t size) {
    if (size == 0) return;
    EltwiseHelper{dnnl::algorithm::eltwise_tanh}(x, y, size);
  }

  // --- relu ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::relu(const T* x, T* y, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(size)),
        [=](sycl::id<1> idx) {
          T xv = x[idx];
          y[idx] = xv > T(0) ? xv : T(0);
        }).wait();
  }

  template<>
  template<>
  void primitives<Device::XPU>::relu(const float* x, float* y, dim_t size) {
    if (size == 0) return;
    EltwiseHelper{dnnl::algorithm::eltwise_relu}(x, y, size);
  }

  // --- gelu (erf) ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::gelu(const T* x, T* y, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(size)),
        [=](sycl::id<1> idx) {
          float xf = static_cast<float>(x[idx]);
          y[idx] = static_cast<T>(T(0.5) * xf * (T(1) + sycl::erf(xf / T(1.4142135623730951))));
        }).wait();
  }

  template<>
  template<>
  void primitives<Device::XPU>::gelu(const float* x, float* y, dim_t size) {
    if (size == 0) return;
    EltwiseHelper{dnnl::algorithm::eltwise_gelu_erf}(x, y, size);
  }

  // --- gelu_tanh ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::gelu_tanh(const T* x, T* y, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(size)),
        [=](sycl::id<1> idx) {
          float xf = static_cast<float>(x[idx]);
          float tanh_arg = T(0.7978845608028654) * (xf + T(0.044715) * xf * xf * xf);
          y[idx] = static_cast<T>(T(0.5) * xf * (T(1) + sycl::tanh(tanh_arg)));
        }).wait();
  }

  template<>
  template<>
  void primitives<Device::XPU>::gelu_tanh(const float* x, float* y, dim_t size) {
    if (size == 0) return;
    EltwiseHelper{dnnl::algorithm::eltwise_gelu_tanh}(x, y, size);
  }

  // --- gelu_sigmoid ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::gelu_sigmoid(const T* x, T* y, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(size)),
        [=](sycl::id<1> idx) {
          float xf = static_cast<float>(x[idx]);
          y[idx] = static_cast<T>(xf / (T(1) + sycl::exp(-T(1.702) * xf)));
        }).wait();
  }

  // --- sigmoid ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::sigmoid(const T* x, T* y, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(size)),
        [=](sycl::id<1> idx) {
          float xf = static_cast<float>(x[idx]);
          y[idx] = static_cast<T>(T(1) / (T(1) + sycl::exp(-xf)));
        }).wait();
  }

  template<>
  template<>
  void primitives<Device::XPU>::sigmoid(const float* x, float* y, dim_t size) {
    if (size == 0) return;
    EltwiseHelper{dnnl::algorithm::eltwise_logistic}(x, y, size);
  }

  // --- swish ---

  template<>
  template <typename T>
  void primitives<Device::XPU>::swish(const T* x, T* y, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(size)),
        [=](sycl::id<1> idx) {
          float xf = static_cast<float>(x[idx]);
          y[idx] = static_cast<T>(xf / (T(1) + sycl::exp(-xf)));
        }).wait();
  }

  template<>
  template<>
  void primitives<Device::XPU>::swish(const float* x, float* y, dim_t size) {
    if (size == 0) return;
    // DNNL has eltwise_swish: swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
    // DNNL swish uses alpha as the scaling factor for sigmoid: swish(x) = x * sigmoid(alpha * x)
    // With alpha=1, this matches standard swish.
    EltwiseHelper{dnnl::algorithm::eltwise_swish, 1.f, 0.f}(x, y, size);
  }

  // --- compute_u8_compensation ---

  template<>
  void primitives<Device::XPU>::compute_u8_compensation(
      const int8_t* b,
      bool transpose_b,
      dim_t k,
      dim_t n,
      float alpha,
      int32_t* compensation) {
    // Compute compensation for uint8 shift of A: compensation[j] = rint(-128 * alpha * sum_i(b[i][j])).
    // This mirrors the CPU implementation where A is shifted by +128 to convert int8 -> uint8.
    if (k == 0 || n == 0) return;
    auto queue = xpu::get_sycl_queue();

    // Single device allocation for all column sums.
    // Previously this allocated/freed per column in a serial loop (O(n) device allocations).
    int32_t* d_sums = sycl::malloc_device<int32_t>(n, queue);
    queue.fill(d_sums, int32_t(0), n).wait();

    // Each work-group reduces one column across its k elements using local memory.
    constexpr size_t wg_size = 256;

    if (transpose_b) {
      // b is transposed, shape (n, k): compensation[j] = rint(-128 * alpha * sum_i(b[j][i]))
      queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int32_t, 1> local_sum(wg_size, cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(n * wg_size), sycl::range<1>(wg_size)),
            [=](sycl::nd_item<1> item) {
              const dim_t col = static_cast<dim_t>(item.get_group_linear_id());
              if (col >= n) return;
              const size_t lid = item.get_local_linear_id();

              int32_t sum = 0;
              for (dim_t i = static_cast<dim_t>(lid); i < k; i += static_cast<dim_t>(wg_size))
                sum += b[col * k + i];
              local_sum[lid] = sum;

              for (size_t stride = wg_size / 2; stride > 0; stride >>= 1) {
                item.barrier(sycl::access::fence_space::local_space);
                if (lid < stride)
                  local_sum[lid] += local_sum[lid + stride];
              }
              if (lid == 0)
                d_sums[col] = local_sum[0];
            });
      }).wait();
    } else {
      // b is row-major, shape (k, n): compensation[j] = rint(-128 * alpha * sum_i(b[i][j]))
      queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int32_t, 1> local_sum(wg_size, cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(n * wg_size), sycl::range<1>(wg_size)),
            [=](sycl::nd_item<1> item) {
              const dim_t col = static_cast<dim_t>(item.get_group_linear_id());
              if (col >= n) return;
              const size_t lid = item.get_local_linear_id();

              int32_t sum = 0;
              for (dim_t i = static_cast<dim_t>(lid); i < k; i += static_cast<dim_t>(wg_size))
                sum += b[i * n + col];
              local_sum[lid] = sum;

              for (size_t stride = wg_size / 2; stride > 0; stride >>= 1) {
                item.barrier(sycl::access::fence_space::local_space);
                if (lid < stride)
                  local_sum[lid] += local_sum[lid + stride];
              }
              if (lid == 0)
                d_sums[col] = local_sum[0];
            });
      }).wait();
    }

    // Copy all results back to host in one batch.
    std::vector<int32_t> sums(n);
    queue.memcpy(sums.data(), d_sums, n * sizeof(int32_t)).wait();
    sycl::free(d_sums, queue);

    // Compute final compensation values on host.
    for (dim_t j = 0; j < n; ++j) {
      float comp = -128.f * alpha * static_cast<float>(sums[j]);
      compensation[j] = static_cast<int32_t>(std::nearbyintf(comp));
    }
  }

  // --- gemm_pack_b ---

  template<>
  template <typename T>
  dim_t primitives<Device::XPU>::gemm_pack_b(const T* b,
                                                    const bool transpose_b,
                                                    const dim_t k,
                                                    const dim_t n,
                                                    const float alpha,
                                                    T* dest) {
    (void)b;
    (void)transpose_b;
    (void)k;
    (void)n;
    (void)alpha;
    (void)dest;
    return 0;  // No packing support for MVP.
  }


  // --- gemm ---

#define DECLARE_XPU_GEMM(IN_T, OUT_T, DNNL_IN_T, DNNL_OUT_T)                   \
  template<>                                                                   \
  template<>                                                                   \
  void primitives<Device::XPU>::gemm(                                          \
      bool a_is_packed, bool b_is_packed,                                      \
      bool transpose_a, bool transpose_b,                                      \
      dim_t m, dim_t n, dim_t k,                                               \
      float alpha,                                                             \
      const IN_T* a, dim_t lda,                                                \
      const IN_T* b, dim_t ldb,                                                \
      float beta,                                                              \
      OUT_T* c, dim_t ldc,                                                     \
      const OUT_T* a_shift_compensation) {                                     \
    (void)a_is_packed;                                                         \
    (void)b_is_packed;                                                         \
    (void)a_shift_compensation;                                                \
    if (m == 0 || n == 0 || k == 0) return;                                    \
    try {                                                                      \
      auto a_md = transpose_a                                                  \
          ? dnnl::memory::desc({k, m}, dnnl::memory::data_type::DNNL_IN_T, {1, lda}) \
          : dnnl::memory::desc({m, k}, dnnl::memory::data_type::DNNL_IN_T, {lda, 1});\
      auto b_md = transpose_b                                                  \
          ? dnnl::memory::desc({n, k}, dnnl::memory::data_type::DNNL_IN_T, {1, ldb}) \
          : dnnl::memory::desc({k, n}, dnnl::memory::data_type::DNNL_IN_T, {ldb, 1});\
      auto c_md = dnnl::memory::desc({m, n}, dnnl::memory::data_type::DNNL_OUT_T, {ldc, 1});\
                                                                               \
      auto a_mem = wrap_memory(a_md, a);                                       \
      auto b_mem = wrap_memory(b_md, b);                                       \
      auto c_mem = wrap_memory(c_md, c);                                       \
                                                                               \
      dnnl::primitive_attr attr;                                               \
      attr.set_scales_mask(DNNL_ARG_DST, 0);                                   \
      dnnl::post_ops post_ops;                                                 \
      if (beta != 0.f) post_ops.append_sum(beta);                              \
      attr.set_post_ops(post_ops);                                             \
                                                                               \
      auto pd = dnnl::matmul::primitive_desc(                                  \
          xpu::get_dnnl_engine(), a_md, b_md, c_md, attr);                     \
      auto prim = dnnl::matmul(pd);                                            \
                                                                               \
      float output_scale = alpha;                                              \
      auto scale_mem = wrap_memory(                                            \
          dnnl::memory::desc({1}, dnnl::memory::data_type::f32,                \
                             dnnl::memory::format_tag::a),                     \
          &output_scale);                                                      \
                                                                               \
      execute_primitive(prim, {                                                \
        {DNNL_ARG_SRC, a_mem},                                                 \
        {DNNL_ARG_WEIGHTS, b_mem},                                             \
        {DNNL_ARG_DST, c_mem},                                                 \
        {DNNL_ARG_ATTR_OUTPUT_SCALES, scale_mem}                               \
      });                                                                      \
    } catch (const dnnl::error& e) {                                           \
      throw std::runtime_error("XPU GEMM failed: " + std::string(e.what()));   \
    }                                                                          \
  }

DECLARE_XPU_GEMM(float, float, f32, f32)
DECLARE_XPU_GEMM(float16_t, float16_t, f16, f16)
DECLARE_XPU_GEMM(bfloat16_t, bfloat16_t, bf16, bf16)
DECLARE_XPU_GEMM(int8_t, float, s8, f32)
DECLARE_XPU_GEMM(int8_t, float16_t, s8, f16)
DECLARE_XPU_GEMM(int8_t, bfloat16_t, s8, bf16)
DECLARE_XPU_GEMM(int8_t, int32_t, s8, s32)

#undef DECLARE_XPU_GEMM

  template<>
  template<>
  void primitives<Device::XPU>::gemm(
      bool, bool, bool, bool, dim_t, dim_t, dim_t,
      float, const int16_t*, dim_t, const int16_t*, dim_t,
      float, int32_t*, dim_t, const int32_t*) {
    throw std::runtime_error("No INT16 GEMM backend on XPU");
  }

  // --- gemm_batch_strided ---
  // Batched strided GEMM implemented via a single dnnl::matmul primitive with
  // 3D memory descriptors and explicit strides, avoiding O(batch_size) syncs.

#define DECLARE_XPU_GEMM_BATCH_STRIDED(IN_T, OUT_T, DNNL_T)                    \
  template<>                                                                   \
  template<>                                                                   \
  void primitives<Device::XPU>::gemm_batch_strided(                            \
      bool transpose_a, bool transpose_b,                                      \
      dim_t m, dim_t n, dim_t k,                                               \
      float alpha,                                                             \
      const IN_T* a, dim_t lda, dim_t stridea,                                 \
      const IN_T* b_ptr, dim_t ldb, dim_t strideb,                             \
      float beta,                                                              \
      OUT_T* c, dim_t ldc, dim_t stridec,                                      \
      dim_t batch_size) {                                                      \
    if (m == 0 || n == 0 || k == 0 || batch_size == 0) return;                 \
    try {                                                                      \
      /* 3D memory descriptors with batch dimension and custom strides */      \
      auto a_dims = transpose_a                                                \
          ? dnnl::memory::dims{batch_size, k, m}                               \
          : dnnl::memory::dims{batch_size, m, k};                              \
      auto a_strides = transpose_a                                             \
          ? dnnl::memory::dims{stridea, 1, lda}                                \
          : dnnl::memory::dims{stridea, lda, 1};                               \
      auto b_dims = transpose_b                                                \
          ? dnnl::memory::dims{batch_size, n, k}                               \
          : dnnl::memory::dims{batch_size, k, n};                              \
      auto b_strides = transpose_b                                             \
          ? dnnl::memory::dims{strideb, 1, ldb}                                \
          : dnnl::memory::dims{strideb, ldb, 1};                               \
      auto c_dims = dnnl::memory::dims{batch_size, m, n};                      \
      auto c_strides = dnnl::memory::dims{stridec, ldc, 1};                    \
                                                                               \
      auto a_md = dnnl::memory::desc(                                          \
          a_dims, dnnl::memory::data_type::DNNL_T, a_strides);                 \
      auto b_md = dnnl::memory::desc(                                          \
          b_dims, dnnl::memory::data_type::DNNL_T, b_strides);                 \
      auto c_md = dnnl::memory::desc(                                          \
          c_dims, dnnl::memory::data_type::DNNL_T, c_strides);                 \
                                                                               \
      auto a_mem = wrap_memory(a_md, a);                                       \
      auto b_mem = wrap_memory(b_md, b_ptr);                                   \
      auto c_mem = wrap_memory(c_md, c);                                       \
                                                                               \
      dnnl::primitive_attr attr;                                               \
      attr.set_scales_mask(DNNL_ARG_DST, 0);                                   \
      dnnl::post_ops post_ops;                                                 \
      if (beta != 0.f) post_ops.append_sum(beta);                              \
      attr.set_post_ops(post_ops);                                             \
                                                                               \
      auto pd = dnnl::matmul::primitive_desc(                                  \
          xpu::get_dnnl_engine(), a_md, b_md, c_md, attr);                     \
      auto prim = dnnl::matmul(pd);                                            \
                                                                               \
      float output_scale = alpha;                                              \
      auto scale_mem = wrap_memory(                                            \
          dnnl::memory::desc({1}, dnnl::memory::data_type::f32,                \
                             dnnl::memory::format_tag::a),                     \
          &output_scale);                                                      \
                                                                               \
      /* Single execute_primitive call -> one stream.wait() */                 \
      execute_primitive(prim, {                                                \
        {DNNL_ARG_SRC, a_mem},                                                 \
        {DNNL_ARG_WEIGHTS, b_mem},                                             \
        {DNNL_ARG_DST, c_mem},                                                 \
        {DNNL_ARG_ATTR_OUTPUT_SCALES, scale_mem}                               \
      });                                                                      \
    } catch (const dnnl::error& e) {                                           \
      throw std::runtime_error(                                                \
          "XPU batched GEMM failed: " + std::string(e.what()));               \
    }                                                                          \
  }

DECLARE_XPU_GEMM_BATCH_STRIDED(float, float, f32)
DECLARE_XPU_GEMM_BATCH_STRIDED(float16_t, float16_t, f16)
DECLARE_XPU_GEMM_BATCH_STRIDED(bfloat16_t, bfloat16_t, bf16)

#undef DECLARE_XPU_GEMM_BATCH_STRIDED
  // ==========================================================================
  // cross_device_primitives
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winstantiation-after-specialization"
#endif

  // ==========================================================================

  template<>
  template <typename T>
  void cross_device_primitives<Device::XPU, Device::CPU>::copy(
      const T* x, T* y, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    // XPU->CPU: device memory to host memory.
    queue.memcpy(y, x, static_cast<size_t>(size) * sizeof(T)).wait();
  }

  template<>
  template <typename T>
  void cross_device_primitives<Device::CPU, Device::XPU>::copy(
      const T* x, T* y, dim_t size) {
    if (size == 0) return;
    auto queue = xpu::get_sycl_queue();
    // CPU->XPU: host memory to device memory.
    queue.memcpy(y, x, static_cast<size_t>(size) * sizeof(T)).wait();
  }

  // ==========================================================================
  // Explicit instantiations for primitives<Device::XPU>
  // ==========================================================================

  // at
  template float primitives<Device::XPU>::at(const float*, dim_t);
  template int32_t primitives<Device::XPU>::at(const int32_t*, dim_t);
  template signed char primitives<Device::XPU>::at(const signed char*, dim_t);
  template short primitives<Device::XPU>::at(const short*, dim_t);
  template half_float::half primitives<Device::XPU>::at(const half_float::half*, dim_t);
  template ctranslate2::bfloat16_t primitives<Device::XPU>::at(const ctranslate2::bfloat16_t*, dim_t);

  // fill
  template void primitives<Device::XPU>::fill(float*, float, dim_t);
template void primitives<Device::XPU>::fill(float16_t*, float16_t, dim_t);
template void primitives<Device::XPU>::fill(bfloat16_t*, bfloat16_t, dim_t);
  template void primitives<Device::XPU>::fill(int32_t*, int32_t, dim_t);
  template void primitives<Device::XPU>::fill(signed char*, signed char, dim_t);
  template void primitives<Device::XPU>::fill(short*, short, dim_t);

  // strided_fill
  template void primitives<Device::XPU>::strided_fill(float*, float, dim_t, dim_t);
template void primitives<Device::XPU>::strided_fill(float16_t*, float16_t, dim_t, dim_t);
template void primitives<Device::XPU>::strided_fill(bfloat16_t*, bfloat16_t, dim_t, dim_t);
  template void primitives<Device::XPU>::strided_fill(int32_t*, int32_t, dim_t, dim_t);
  template void primitives<Device::XPU>::strided_fill(signed char*, signed char, dim_t, dim_t);
  template void primitives<Device::XPU>::strided_fill(short*, short, dim_t, dim_t);

  // indexed_fill
  template void primitives<Device::XPU>::indexed_fill(float*, float,
                                                             const int32_t*, dim_t);
template void primitives<Device::XPU>::indexed_fill(float16_t*, float16_t,
                                                             const int32_t*, dim_t);
template void primitives<Device::XPU>::indexed_fill(bfloat16_t*, bfloat16_t,
                                                             const int32_t*, dim_t);
  template void primitives<Device::XPU>::indexed_fill(int32_t*, int32_t,
                                                             const int32_t*, dim_t);
  template void primitives<Device::XPU>::indexed_fill(signed char*, signed char,
                                                             const int32_t*, dim_t);
  template void primitives<Device::XPU>::indexed_fill(short*, short,
                                                             const int32_t*, dim_t);

  // copy
  template void primitives<Device::XPU>::copy(const float*, float*, dim_t);
template void primitives<Device::XPU>::copy(const float16_t*, float16_t*, dim_t);
template void primitives<Device::XPU>::copy(const bfloat16_t*, bfloat16_t*, dim_t);
  template void primitives<Device::XPU>::copy(const int32_t*, int32_t*, dim_t);
  template void primitives<Device::XPU>::copy(const signed char*, signed char*, dim_t);
  template void primitives<Device::XPU>::copy(const short*, short*, dim_t);

  // convert
  template void primitives<Device::XPU>::convert(const float*, float*, dim_t);
  template void primitives<Device::XPU>::convert(const float*, float16_t*, dim_t);
  template void primitives<Device::XPU>::convert(const float16_t*, float*, dim_t);
  template void primitives<Device::XPU>::convert(const float*, bfloat16_t*, dim_t);
  template void primitives<Device::XPU>::convert(const bfloat16_t*, float*, dim_t);
  template void primitives<Device::XPU>::convert(const float16_t*, bfloat16_t*, dim_t);
  template void primitives<Device::XPU>::convert(const bfloat16_t*, float16_t*, dim_t);

  // sum
template float16_t primitives<Device::XPU>::sum(const float16_t*, dim_t);
template bfloat16_t primitives<Device::XPU>::sum(const bfloat16_t*, dim_t);
  template int32_t primitives<Device::XPU>::sum(const int32_t*, dim_t);
  template signed char primitives<Device::XPU>::sum(const signed char*, dim_t);
  template short primitives<Device::XPU>::sum(const short*, dim_t);

  // max_element
  template dim_t primitives<Device::XPU>::max_element(const float*, dim_t);
template dim_t primitives<Device::XPU>::max_element(const float16_t*, dim_t);
template dim_t primitives<Device::XPU>::max_element(const bfloat16_t*, dim_t);
  template dim_t primitives<Device::XPU>::max_element(const int32_t*, dim_t);
  template dim_t primitives<Device::XPU>::max_element(const signed char*, dim_t);
  template dim_t primitives<Device::XPU>::max_element(const short*, dim_t);

  // max (reduce)
template float16_t primitives<Device::XPU>::max(const float16_t*, dim_t);
template bfloat16_t primitives<Device::XPU>::max(const bfloat16_t*, dim_t);
  template int32_t primitives<Device::XPU>::max(const int32_t*, dim_t);
  template signed char primitives<Device::XPU>::max(const signed char*, dim_t);
  template short primitives<Device::XPU>::max(const short*, dim_t);

  // amax
template float16_t primitives<Device::XPU>::amax(const float16_t*, dim_t);
template bfloat16_t primitives<Device::XPU>::amax(const bfloat16_t*, dim_t);
  template int32_t primitives<Device::XPU>::amax(const int32_t*, dim_t);
  template signed char primitives<Device::XPU>::amax(const signed char*, dim_t);
  template short primitives<Device::XPU>::amax(const short*, dim_t);

  // add (scalar)
  template void primitives<Device::XPU>::add(float, const float*, float*, dim_t);
template void primitives<Device::XPU>::add(float16_t, const float16_t*, float16_t*, dim_t);
template void primitives<Device::XPU>::add(bfloat16_t, const bfloat16_t*, bfloat16_t*, dim_t);
  template void primitives<Device::XPU>::add(int32_t, const int32_t*, int32_t*, dim_t);
  template void primitives<Device::XPU>::add(signed char, const signed char*, signed char*, dim_t);
  template void primitives<Device::XPU>::add(short, const short*, short*, dim_t);

  // add (array)
  template void primitives<Device::XPU>::add(const float*, const float*, float*, dim_t);
template void primitives<Device::XPU>::add(const float16_t*, const float16_t*, float16_t*, dim_t);
template void primitives<Device::XPU>::add(const bfloat16_t*, const bfloat16_t*, bfloat16_t*, dim_t);
  template void primitives<Device::XPU>::add(const int32_t*, const int32_t*, int32_t*, dim_t);
  template void primitives<Device::XPU>::add(const signed char*, const signed char*, signed char*, dim_t);
  template void primitives<Device::XPU>::add(const short*, const short*, short*, dim_t);

  // add_batch_broadcast
  template void primitives<Device::XPU>::add_batch_broadcast(
      const float*, const float*, float*, dim_t, dim_t);
template void primitives<Device::XPU>::add_batch_broadcast(
      const float16_t*, const float16_t*, float16_t*, dim_t, dim_t);
template void primitives<Device::XPU>::add_batch_broadcast(
      const bfloat16_t*, const bfloat16_t*, bfloat16_t*, dim_t, dim_t);
  template void primitives<Device::XPU>::add_batch_broadcast(
      const int32_t*, const int32_t*, int32_t*, dim_t, dim_t);
  template void primitives<Device::XPU>::add_batch_broadcast(
      const signed char*, const signed char*, signed char*, dim_t, dim_t);
  template void primitives<Device::XPU>::add_batch_broadcast(
      const short*, const short*, short*, dim_t, dim_t);

  // add_depth_broadcast
  template void primitives<Device::XPU>::add_depth_broadcast(
      const float*, const float*, float*, dim_t, dim_t);
template void primitives<Device::XPU>::add_depth_broadcast(
      const float16_t*, const float16_t*, float16_t*, dim_t, dim_t);
template void primitives<Device::XPU>::add_depth_broadcast(
      const bfloat16_t*, const bfloat16_t*, bfloat16_t*, dim_t, dim_t);
  template void primitives<Device::XPU>::add_depth_broadcast(
      const int32_t*, const int32_t*, int32_t*, dim_t, dim_t);
  template void primitives<Device::XPU>::add_depth_broadcast(
      const signed char*, const signed char*, signed char*, dim_t, dim_t);
  template void primitives<Device::XPU>::add_depth_broadcast(
      const short*, const short*, short*, dim_t, dim_t);

  // add_block_broadcast
  template void primitives<Device::XPU>::add_block_broadcast(
      const float*, const float*, float*, dim_t, dim_t, dim_t);
template void primitives<Device::XPU>::add_block_broadcast(
      const float16_t*, const float16_t*, float16_t*, dim_t, dim_t, dim_t);
template void primitives<Device::XPU>::add_block_broadcast(
      const bfloat16_t*, const bfloat16_t*, bfloat16_t*, dim_t, dim_t, dim_t);
  template void primitives<Device::XPU>::add_block_broadcast(
      const int32_t*, const int32_t*, int32_t*, dim_t, dim_t, dim_t);
  template void primitives<Device::XPU>::add_block_broadcast(
      const signed char*, const signed char*, signed char*, dim_t, dim_t, dim_t);
  template void primitives<Device::XPU>::add_block_broadcast(
      const short*, const short*, short*, dim_t, dim_t, dim_t);

  // sub (array)
  template void primitives<Device::XPU>::sub(const float*, const float*, float*, dim_t);
template void primitives<Device::XPU>::sub(const float16_t*, const float16_t*, float16_t*, dim_t);
template void primitives<Device::XPU>::sub(const bfloat16_t*, const bfloat16_t*, bfloat16_t*, dim_t);
  template void primitives<Device::XPU>::sub(const int32_t*, const int32_t*, int32_t*, dim_t);
  template void primitives<Device::XPU>::sub(const signed char*, const signed char*, signed char*, dim_t);
  template void primitives<Device::XPU>::sub(const short*, const short*, short*, dim_t);

  // max (scalar)
  template void primitives<Device::XPU>::max(float, const float*, float*, dim_t);
template void primitives<Device::XPU>::max(float16_t, const float16_t*, float16_t*, dim_t);
template void primitives<Device::XPU>::max(bfloat16_t, const bfloat16_t*, bfloat16_t*, dim_t);
  template void primitives<Device::XPU>::max(int32_t, const int32_t*, int32_t*, dim_t);
  template void primitives<Device::XPU>::max(signed char, const signed char*, signed char*, dim_t);
  template void primitives<Device::XPU>::max(short, const short*, short*, dim_t);

  // max (array)
  template void primitives<Device::XPU>::max(const float*, const float*, float*, dim_t);
template void primitives<Device::XPU>::max(const float16_t*, const float16_t*, float16_t*, dim_t);
template void primitives<Device::XPU>::max(const bfloat16_t*, const bfloat16_t*, bfloat16_t*, dim_t);
  template void primitives<Device::XPU>::max(const int32_t*, const int32_t*, int32_t*, dim_t);
  template void primitives<Device::XPU>::max(const signed char*, const signed char*, signed char*, dim_t);
  template void primitives<Device::XPU>::max(const short*, const short*, short*, dim_t);

  // min (scalar)
  template void primitives<Device::XPU>::min(float, const float*, float*, dim_t);
template void primitives<Device::XPU>::min(float16_t, const float16_t*, float16_t*, dim_t);
template void primitives<Device::XPU>::min(bfloat16_t, const bfloat16_t*, bfloat16_t*, dim_t);
  template void primitives<Device::XPU>::min(int32_t, const int32_t*, int32_t*, dim_t);
  template void primitives<Device::XPU>::min(signed char, const signed char*, signed char*, dim_t);
  template void primitives<Device::XPU>::min(short, const short*, short*, dim_t);

  // min (array)
  template void primitives<Device::XPU>::min(const float*, const float*, float*, dim_t);
template void primitives<Device::XPU>::min(const float16_t*, const float16_t*, float16_t*, dim_t);
template void primitives<Device::XPU>::min(const bfloat16_t*, const bfloat16_t*, bfloat16_t*, dim_t);
  template void primitives<Device::XPU>::min(const int32_t*, const int32_t*, int32_t*, dim_t);
  template void primitives<Device::XPU>::min(const signed char*, const signed char*, signed char*, dim_t);
  template void primitives<Device::XPU>::min(const short*, const short*, short*, dim_t);

  // mul (scalar)
  template void primitives<Device::XPU>::mul(float, const float*, float*, dim_t);
template void primitives<Device::XPU>::mul(float16_t, const float16_t*, float16_t*, dim_t);
template void primitives<Device::XPU>::mul(bfloat16_t, const bfloat16_t*, bfloat16_t*, dim_t);
  template void primitives<Device::XPU>::mul(int32_t, const int32_t*, int32_t*, dim_t);
  template void primitives<Device::XPU>::mul(signed char, const signed char*, signed char*, dim_t);
  template void primitives<Device::XPU>::mul(short, const short*, short*, dim_t);

  // mul (array)
  template void primitives<Device::XPU>::mul(const float*, const float*, float*, dim_t);
template void primitives<Device::XPU>::mul(const float16_t*, const float16_t*, float16_t*, dim_t);
template void primitives<Device::XPU>::mul(const bfloat16_t*, const bfloat16_t*, bfloat16_t*, dim_t);
  template void primitives<Device::XPU>::mul(const int32_t*, const int32_t*, int32_t*, dim_t);
  template void primitives<Device::XPU>::mul(const signed char*, const signed char*, signed char*, dim_t);
  template void primitives<Device::XPU>::mul(const short*, const short*, short*, dim_t);

  // mul_batch_broadcast
  template void primitives<Device::XPU>::mul_batch_broadcast(
      const float*, const float*, float*, dim_t, dim_t);
template void primitives<Device::XPU>::mul_batch_broadcast(
      const float16_t*, const float16_t*, float16_t*, dim_t, dim_t);
template void primitives<Device::XPU>::mul_batch_broadcast(
      const bfloat16_t*, const bfloat16_t*, bfloat16_t*, dim_t, dim_t);
  template void primitives<Device::XPU>::mul_batch_broadcast(
      const int32_t*, const int32_t*, int32_t*, dim_t, dim_t);
  template void primitives<Device::XPU>::mul_batch_broadcast(
      const signed char*, const signed char*, signed char*, dim_t, dim_t);
  template void primitives<Device::XPU>::mul_batch_broadcast(
      const short*, const short*, short*, dim_t, dim_t);

  // penalize_previous_tokens
  template void primitives<Device::XPU>::penalize_previous_tokens(
      float*, const float*, const int32_t*, float, dim_t, dim_t, dim_t);
template void primitives<Device::XPU>::penalize_previous_tokens(
      float16_t*, const float16_t*, const int32_t*, float16_t, dim_t, dim_t, dim_t);
template void primitives<Device::XPU>::penalize_previous_tokens(
      bfloat16_t*, const bfloat16_t*, const int32_t*, bfloat16_t, dim_t, dim_t, dim_t);
  template void primitives<Device::XPU>::penalize_previous_tokens(
      int32_t*, const int32_t*, const int32_t*, int32_t, dim_t, dim_t, dim_t);
  template void primitives<Device::XPU>::penalize_previous_tokens(
      signed char*, const signed char*, const int32_t*, signed char, dim_t, dim_t, dim_t);
  template void primitives<Device::XPU>::penalize_previous_tokens(
      short*, const short*, const int32_t*, short, dim_t, dim_t, dim_t);

  // transpose_2d
  template void primitives<Device::XPU>::transpose_2d(const float*, const dim_t*, float*);
template void primitives<Device::XPU>::transpose_2d(const float16_t*, const dim_t*, float16_t*);
template void primitives<Device::XPU>::transpose_2d(const bfloat16_t*, const dim_t*, bfloat16_t*);
  template void primitives<Device::XPU>::transpose_2d(const int32_t*, const dim_t*, int32_t*);
  template void primitives<Device::XPU>::transpose_2d(const signed char*, const dim_t*, signed char*);
  template void primitives<Device::XPU>::transpose_2d(const short*, const dim_t*, short*);

  // transpose_3d
  template void primitives<Device::XPU>::transpose_3d(
      const float*, const dim_t*, const dim_t*, float*);
template void primitives<Device::XPU>::transpose_3d(
      const float16_t*, const dim_t*, const dim_t*, float16_t*);
template void primitives<Device::XPU>::transpose_3d(
      const bfloat16_t*, const dim_t*, const dim_t*, bfloat16_t*);
  template void primitives<Device::XPU>::transpose_3d(
      const int32_t*, const dim_t*, const dim_t*, int32_t*);
  template void primitives<Device::XPU>::transpose_3d(
      const signed char*, const dim_t*, const dim_t*, signed char*);
  template void primitives<Device::XPU>::transpose_3d(
      const short*, const dim_t*, const dim_t*, short*);

  // transpose_4d
  template void primitives<Device::XPU>::transpose_4d(
      const float*, const dim_t*, const dim_t*, float*);
template void primitives<Device::XPU>::transpose_4d(
      const float16_t*, const dim_t*, const dim_t*, float16_t*);
template void primitives<Device::XPU>::transpose_4d(
      const bfloat16_t*, const dim_t*, const dim_t*, bfloat16_t*);
  template void primitives<Device::XPU>::transpose_4d(
      const int32_t*, const dim_t*, const dim_t*, int32_t*);
  template void primitives<Device::XPU>::transpose_4d(
      const signed char*, const dim_t*, const dim_t*, signed char*);
  template void primitives<Device::XPU>::transpose_4d(
      const short*, const dim_t*, const dim_t*, short*);

  // logsumexp
  template float primitives<Device::XPU>::logsumexp(const float*, dim_t);
template float primitives<Device::XPU>::logsumexp(const float16_t*, dim_t);
template float primitives<Device::XPU>::logsumexp(const bfloat16_t*, dim_t);

  // exp
  template void primitives<Device::XPU>::exp(const float*, float*, dim_t);
template void primitives<Device::XPU>::exp(const float16_t*, float16_t*, dim_t);
template void primitives<Device::XPU>::exp(const bfloat16_t*, bfloat16_t*, dim_t);

  // log
  template void primitives<Device::XPU>::log(const float*, float*, dim_t);
template void primitives<Device::XPU>::log(const float16_t*, float16_t*, dim_t);
template void primitives<Device::XPU>::log(const bfloat16_t*, bfloat16_t*, dim_t);

  // cos
  template void primitives<Device::XPU>::cos(const float*, float*, dim_t);
template void primitives<Device::XPU>::cos(const float16_t*, float16_t*, dim_t);
template void primitives<Device::XPU>::cos(const bfloat16_t*, bfloat16_t*, dim_t);

  // sin
  template void primitives<Device::XPU>::sin(const float*, float*, dim_t);
template void primitives<Device::XPU>::sin(const float16_t*, float16_t*, dim_t);
template void primitives<Device::XPU>::sin(const bfloat16_t*, bfloat16_t*, dim_t);

  // tanh
  template void primitives<Device::XPU>::tanh(const float*, float*, dim_t);
template void primitives<Device::XPU>::tanh(const float16_t*, float16_t*, dim_t);
template void primitives<Device::XPU>::tanh(const bfloat16_t*, bfloat16_t*, dim_t);

  // relu
  template void primitives<Device::XPU>::relu(const float*, float*, dim_t);
template void primitives<Device::XPU>::relu(const float16_t*, float16_t*, dim_t);
template void primitives<Device::XPU>::relu(const bfloat16_t*, bfloat16_t*, dim_t);

  // gelu
  template void primitives<Device::XPU>::gelu(const float*, float*, dim_t);
template void primitives<Device::XPU>::gelu(const float16_t*, float16_t*, dim_t);
template void primitives<Device::XPU>::gelu(const bfloat16_t*, bfloat16_t*, dim_t);

  // gelu_tanh
  template void primitives<Device::XPU>::gelu_tanh(const float*, float*, dim_t);
template void primitives<Device::XPU>::gelu_tanh(const float16_t*, float16_t*, dim_t);
template void primitives<Device::XPU>::gelu_tanh(const bfloat16_t*, bfloat16_t*, dim_t);

  // gelu_sigmoid
  template void primitives<Device::XPU>::gelu_sigmoid(const float*, float*, dim_t);
template void primitives<Device::XPU>::gelu_sigmoid(const float16_t*, float16_t*, dim_t);
template void primitives<Device::XPU>::gelu_sigmoid(const bfloat16_t*, bfloat16_t*, dim_t);

  // sigmoid
  template void primitives<Device::XPU>::sigmoid(const float*, float*, dim_t);
template void primitives<Device::XPU>::sigmoid(const float16_t*, float16_t*, dim_t);
template void primitives<Device::XPU>::sigmoid(const bfloat16_t*, bfloat16_t*, dim_t);

  // swish
  template void primitives<Device::XPU>::swish(const float*, float*, dim_t);
template void primitives<Device::XPU>::swish(const float16_t*, float16_t*, dim_t);
template void primitives<Device::XPU>::swish(const bfloat16_t*, bfloat16_t*, dim_t);

  // gemm_pack_b
  template dim_t primitives<Device::XPU>::gemm_pack_b(
      const float*, bool, dim_t, dim_t, float, float*);
  template dim_t primitives<Device::XPU>::gemm_pack_b(
      const int16_t*, bool, dim_t, dim_t, float, int16_t*);
  template dim_t primitives<Device::XPU>::gemm_pack_b(
      const int8_t*, bool, dim_t, dim_t, float, int8_t*);

  // gemm
  template void primitives<Device::XPU>::gemm(
      bool, bool, bool, bool, dim_t, dim_t, dim_t,
      float, const float*, dim_t, const float*, dim_t,
      float, float*, dim_t, const float*);
template void primitives<Device::XPU>::gemm(
      bool, bool, bool, bool, dim_t, dim_t, dim_t,
      float, const float16_t*, dim_t, const float16_t*, dim_t,
      float, float16_t*, dim_t, const float16_t*);
template void primitives<Device::XPU>::gemm(
      bool, bool, bool, bool, dim_t, dim_t, dim_t,
      float, const bfloat16_t*, dim_t, const bfloat16_t*, dim_t,
      float, bfloat16_t*, dim_t, const bfloat16_t*);
  template void primitives<Device::XPU>::gemm(
      bool, bool, bool, bool, dim_t, dim_t, dim_t,
      float, const int8_t*, dim_t, const int8_t*, dim_t,
      float, float*, dim_t, const float*);
template void primitives<Device::XPU>::gemm(
      bool, bool, bool, bool, dim_t, dim_t, dim_t,
      float, const int8_t*, dim_t, const int8_t*, dim_t,
      float, float16_t*, dim_t, const float16_t*);
template void primitives<Device::XPU>::gemm(
      bool, bool, bool, bool, dim_t, dim_t, dim_t,
      float, const int8_t*, dim_t, const int8_t*, dim_t,
      float, bfloat16_t*, dim_t, const bfloat16_t*);
  template void primitives<Device::XPU>::gemm(
      bool, bool, bool, bool, dim_t, dim_t, dim_t,
      float, const int16_t*, dim_t, const int16_t*, dim_t,
      float, int32_t*, dim_t, const int32_t*);
  template void primitives<Device::XPU>::gemm(
      bool, bool, bool, bool, dim_t, dim_t, dim_t,
      float, const int8_t*, dim_t, const int8_t*, dim_t,
      float, int32_t*, dim_t, const int32_t*);

  // gemm_batch_strided
  template void primitives<Device::XPU>::gemm_batch_strided(
      bool, bool, dim_t, dim_t, dim_t,
      float, const float*, dim_t, dim_t,
      const float*, dim_t, dim_t,
      float, float*, dim_t, dim_t,
      dim_t);
template void primitives<Device::XPU>::gemm_batch_strided(
      bool, bool, dim_t, dim_t, dim_t,
      float, const float16_t*, dim_t, dim_t,
      const float16_t*, dim_t, dim_t,
      float, float16_t*, dim_t, dim_t,
      dim_t);
template void primitives<Device::XPU>::gemm_batch_strided(
      bool, bool, dim_t, dim_t, dim_t,
      float, const bfloat16_t*, dim_t, dim_t,
      const bfloat16_t*, dim_t, dim_t,
      float, bfloat16_t*, dim_t, dim_t,
      dim_t);

  // ==========================================================================
  // Explicit instantiations for cross_device_primitives
  // ==========================================================================

  template void cross_device_primitives<Device::XPU, Device::CPU>::copy(
      const float*, float*, dim_t);
  template void cross_device_primitives<Device::XPU, Device::CPU>::copy(
      const int32_t*, int32_t*, dim_t);
  template void cross_device_primitives<Device::CPU, Device::XPU>::copy(
      const float*, float*, dim_t);
  template void cross_device_primitives<Device::CPU, Device::XPU>::copy(
      const int32_t*, int32_t*, dim_t);

  template void cross_device_primitives<Device::XPU, Device::CPU>::copy(
      const float16_t*, float16_t*, dim_t);
  template void cross_device_primitives<Device::CPU, Device::XPU>::copy(
      const float16_t*, float16_t*, dim_t);
  template void cross_device_primitives<Device::XPU, Device::CPU>::copy(
      const bfloat16_t*, bfloat16_t*, dim_t);
  template void cross_device_primitives<Device::CPU, Device::XPU>::copy(
      const bfloat16_t*, bfloat16_t*, dim_t);

  template void cross_device_primitives<Device::XPU, Device::CPU>::copy(
      const int8_t*, int8_t*, dim_t);
  template void cross_device_primitives<Device::CPU, Device::XPU>::copy(
      const int8_t*, int8_t*, dim_t);
  template void cross_device_primitives<Device::XPU, Device::CPU>::copy(
      const int16_t*, int16_t*, dim_t);
  template void cross_device_primitives<Device::CPU, Device::XPU>::copy(
      const int16_t*, int16_t*, dim_t);

}  // namespace ctranslate2
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
