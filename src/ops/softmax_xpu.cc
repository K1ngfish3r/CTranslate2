#include "ctranslate2/ops/softmax.h"
#include "type_dispatch.h"

#ifdef CT2_WITH_XPU

#include <dnnl.hpp>
#include <sycl/sycl.hpp>
#include <limits>

#include "xpu/utils.h"

namespace ctranslate2 {
  namespace ops {

    namespace {

      // -------------------------------------------------------------------------
      // SYCL kernel for masked softmax.
      //
      // T  – host type (float / float16_t / bfloat16_t).  All computation is
      //      performed in float; values are widened on load and narrowed on store
      //      via `static_cast`.
      //
      // Layout:  (batch_size, depth)
      //
      // Each work-group processes one batch row.  `lengths[i]` gives the number of
      // valid (unmasked) elements in row `i`.  Positions >= lengths[i] are set to
      // 0 in the output (or 0 after softmax).  If lengths[i] == 0 the entire row
      // is masked and stays at zero.
      //
      // The kernel follows the numerically-stable softmax pattern:
      //   1. row_max = max_j (x[j])  over j < lengths[i]
      //   2. exp_sum = sum_j exp(x[j] - row_max)  over j < lengths[i]
      //   3. y[j] = exp(x[j] - row_max) / exp_sum  (softmax)
      //        or y[j] = x[j] - row_max - log(exp_sum)  (log-softmax)
      //      for j < lengths[i];  y[j] = 0 for j >= lengths[i].
      // -------------------------------------------------------------------------
      template <typename T>
      void masked_softmax_kernel(sycl::queue& queue,
                                 const T* input,
                                 const int32_t* lengths,
                                 T* output,
                                 dim_t batch_size,
                                 dim_t depth,
                                 bool log) {
        static constexpr size_t WG = 256;

        if (batch_size == 0 || depth == 0) return;

        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::nd_range<1>(
                  sycl::range<1>(static_cast<size_t>(batch_size) * WG),
                  sycl::range<1>(WG)),
              [=](sycl::nd_item<1> item) {
                const size_t row = item.get_group_linear_id();
                if (row >= static_cast<size_t>(batch_size)) return;

                const size_t tid = item.get_local_linear_id();
                const size_t n   = static_cast<size_t>(depth);
                const size_t off = row * n;
                const dim_t  raw_valid = lengths[row];

                // Entire row masked -- output is already zero.
                if (raw_valid <= 0) return;

                const size_t valid = static_cast<size_t>(raw_valid);

                // --- Step 1: row-wise max over valid elements ---
                float local_max = std::numeric_limits<float>::lowest();
                for (size_t j = tid; j < valid; j += WG) {
                  local_max = sycl::max(local_max,
                                        static_cast<float>(input[off + j]));
                }
                float row_max = sycl::reduce_over_group(
                    item.get_group(), local_max, sycl::maximum<float>());

                // --- Step 2: sum of exp(x - max) over valid elements ---
                float local_sum = 0.0f;
                for (size_t j = tid; j < valid; j += WG) {
                  local_sum += sycl::exp(static_cast<float>(input[off + j])
                                         - row_max);
                }
                float exp_sum = sycl::reduce_over_group(
                    item.get_group(), local_sum, sycl::plus<float>());

                // --- Step 3: write output ---
                // All work-items cooperatively write the full row (including the
                // masked region) to ensure correct coverage even when valid < depth.
                if (log) {
                  float log_sum = sycl::log(exp_sum);
                  for (size_t j = tid; j < n; j += WG) {
                    output[off + j] = (j < valid)
                        ? static_cast<T>(static_cast<float>(input[off + j])
                                         - row_max - log_sum)
                        : static_cast<T>(0.0f);
                  }
                } else {
                  float inv_sum = 1.0f / exp_sum;
                  for (size_t j = tid; j < n; j += WG) {
                    output[off + j] = (j < valid)
                        ? static_cast<T>(sycl::exp(
                            static_cast<float>(input[off + j]) - row_max)
                            * inv_sum)
                        : static_cast<T>(0.0f);
                  }
                }
              });
        }).wait();
      }

      // -----------------------------------------------------------------------
      // Shared implementation template for all float types.
      // Dispatches to the SYCL kernel when masking is needed and to oneDNN
      // otherwise.
      // -----------------------------------------------------------------------
      template <typename T>
      void softmax_impl(const StorageView& input,
                        const StorageView* lengths,
                        StorageView& output,
                        bool log) {
        const dim_t depth = input.dim(-1);
        const dim_t batch_size = input.size() / depth;

        if (lengths) {
          // Custom SYCL kernel for masked softmax (oneDNN softmax does not
          // support masking).
          auto queue = xpu::get_sycl_queue();
          masked_softmax_kernel<T>(queue,
                                   input.data<T>(),
                                   lengths->data<int32_t>(),
                                   output.data<T>(),
                                   batch_size,
                                   depth,
                                   log);
          return;
        }

        // Unmasked softmax via oneDNN.
        auto engine = xpu::get_dnnl_engine();
        auto stream = xpu::get_dnnl_stream();

        const auto dtype = xpu::get_dnnl_dtype(DataTypeToEnum<T>::value);

        auto src_md = dnnl::memory::desc({batch_size, depth}, dtype,
                                          dnnl::memory::format_tag::nc);
        auto dst_md = dnnl::memory::desc({batch_size, depth}, dtype,
                                          dnnl::memory::format_tag::nc);

        auto src_mem = dnnl::memory(src_md, engine,
                                     const_cast<void*>(input.buffer()));
        auto dst_mem = dnnl::memory(dst_md, engine, output.buffer());

        auto algorithm = log ? dnnl::algorithm::softmax_log
                             : dnnl::algorithm::softmax_accurate;
        auto softmax_pd = dnnl::softmax_forward::primitive_desc(
            engine, dnnl::prop_kind::forward_inference,
            algorithm, src_md, dst_md, 1);

        auto softmax = dnnl::softmax_forward(softmax_pd);
        softmax.execute(stream, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_DST, dst_mem}
        });
        stream.wait();
      }

    }  // anonymous namespace

    template <Device D, typename T>
    void SoftMax::compute(
        const StorageView& input,
        const StorageView* lengths,
        StorageView& output) const {
      softmax_impl<T>(input, lengths, output, _log);
    }

#define DEFINE_SOFTMAX_XPU(T)                                            \
    template void                                                           \
    SoftMax::compute<Device::XPU, T>(                                       \
        const StorageView& input,                                           \
        const StorageView* lengths,                                         \
        StorageView& output) const;

    DEFINE_SOFTMAX_XPU(float)
    DEFINE_SOFTMAX_XPU(float16_t)
    DEFINE_SOFTMAX_XPU(bfloat16_t)

  }
}

#endif
