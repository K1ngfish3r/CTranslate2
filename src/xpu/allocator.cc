#include <cstdint>
#include <unordered_map>
#include <vector>

#include <sycl/sycl.hpp>

#include "ctranslate2/allocator.h"
#include "xpu/utils.h"

namespace ctranslate2 {

  // A caching allocator for SYCL device memory that reuses freed allocations.
  //
  // sycl::malloc_device and sycl::free are known to be slow system calls
  // (they go through the runtime driver layer). This allocator avoids calling
  // sycl::free on every deallocation and instead caches freed blocks for
  // reuse, dramatically reducing driver-level allocation pressure.
  //
  // Memory is binned by power-of-2 size.  On allocate(), if a cached block
  // of the right bin exists it is returned directly; otherwise a new
  // sycl::malloc_device call is made.  On free(), the block is placed back
  // into the cache instead of being freed, up to a configurable byte limit.
  // Actual sycl::free calls are deferred to clear_cache().
  //
  // Safety: this allocator makes the same assumption as the CUDA
  // CubCachingAllocator — that the caller has finished using the memory
  // before calling free().  Because the allocator is thread_local and the
  // SYCL queue for the current thread is in-order, any kernel submissions
  // that reference a block are guaranteed to have completed before a
  // subsequent allocate() reuses that same block.
  class XpuCachingAllocator : public Allocator {
  public:
    XpuCachingAllocator() {
      const char* env = std::getenv("CT2_XPU_ALLOCATOR_CACHE_SIZE");
      if (env)
        _max_cached_bytes = std::stoull(env);
    }

    ~XpuCachingAllocator() {
      clear_cache();
    }

    void* allocate(size_t size, int device_index) override {
      if (size == 0)
        return nullptr;

      int prev_device_index = -1;
      if (device_index >= 0) {
        prev_device_index = xpu::get_device_index();
        if (prev_device_index != device_index)
          xpu::set_device_index(device_index);
      }

      // Round up to the nearest power of 2 for bin lookup.
      size_t bin_key = round_up_pow2(size);

      // Try to reuse a cached block.
      auto it = _cache.find(bin_key);
      if (it != _cache.end() && !it->second.empty()) {
        auto& block = it->second.back();
        void* ptr = block.ptr;
        _cached_bytes -= block.size;
        it->second.pop_back();
        _allocation_sizes[ptr] = size;

        if (prev_device_index >= 0 && prev_device_index != device_index)
          xpu::set_device_index(prev_device_index);

        return ptr;
      }

      // No cached block available — allocate new device memory.
      auto queue = xpu::get_sycl_queue();
      void* ptr = sycl::malloc_device(size, queue);
      if (!ptr)
        throw std::runtime_error("XPU allocator: sycl::malloc_device failed");

      _allocation_sizes[ptr] = size;

      if (prev_device_index >= 0 && prev_device_index != device_index)
        xpu::set_device_index(prev_device_index);

      return ptr;
    }

    void free(void* ptr, int device_index) override {
      if (!ptr)
        return;

      int prev_device_index = -1;
      if (device_index >= 0) {
        prev_device_index = xpu::get_device_index();
        if (prev_device_index != device_index)
          xpu::set_device_index(device_index);
      }

      auto size_it = _allocation_sizes.find(ptr);
      if (size_it == _allocation_sizes.end()) {
        // Unknown pointer — this should not happen under normal operation.
        // Fall back to a safe deferred free.
        auto queue = xpu::get_sycl_queue();
        auto ctx = queue.get_context();
        queue.ext_oneapi_submit_barrier();
        queue.submit([ptr, ctx](sycl::handler& cgh) {
          cgh.host_task([=]() { sycl::free(ptr, ctx); });
        });
      } else {
        size_t size = size_it->second;
        size_t bin_key = round_up_pow2(size);
        _allocation_sizes.erase(size_it);

        if (_cached_bytes + size <= _max_cached_bytes) {
          // Stay within the cache budget — stash it.
          _cache[bin_key].push_back({ptr, size});
          _cached_bytes += size;
        } else {
          // Over the limit — free for real with deferred barrier.
          auto queue = xpu::get_sycl_queue();
          auto ctx = queue.get_context();
          queue.ext_oneapi_submit_barrier();
          queue.submit([ptr, ctx](sycl::handler& cgh) {
            cgh.host_task([=]() { sycl::free(ptr, ctx); });
          });
        }
      }

      if (prev_device_index >= 0 && prev_device_index != device_index)
        xpu::set_device_index(prev_device_index);
    }

    void clear_cache() override {
      // Wait for the queue to drain, then synchronously free all cached blocks.
      for (auto& [bin, blocks] : _cache) {
        if (blocks.empty())
          continue;
        auto queue = xpu::get_sycl_queue();
        auto ctx = queue.get_context();
        queue.wait();
        for (auto& block : blocks) {
          sycl::free(block.ptr, ctx);
        }
      }
      _cache.clear();
      _cached_bytes = 0;
      _allocation_sizes.clear();
    }

  private:
    struct CachedBlock {
      void* ptr;
      size_t size;
    };

    // Bin-indexed cache of free blocks.  The key is the power-of-2 rounded size.
    std::unordered_map<size_t, std::vector<CachedBlock>> _cache;

    // Tracks the original requested size for each active allocation so that
    // free() can determine the correct bin without an extra parameter.
    std::unordered_map<void*, size_t> _allocation_sizes;

    size_t _max_cached_bytes = 200 * (1 << 20);  // 200 MB default
    size_t _cached_bytes = 0;

    // ---- helpers -----------------------------------------------------------

    static size_t round_up_pow2(size_t v) {
      if (v == 0)
        return 0;
      v--;
      v |= v >> 1;
      v |= v >> 2;
      v |= v >> 4;
      v |= v >> 8;
      v |= v >> 16;
#if SIZE_MAX > 0xFFFFFFFFU
      v |= v >> 32;
#endif
      v++;
      return v;
    }
  };

  template<>
  Allocator& get_allocator<Device::XPU>() {
    static thread_local XpuCachingAllocator allocator;
    return allocator;
  }

}
