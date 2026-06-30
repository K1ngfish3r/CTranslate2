#include "ctranslate2/ops/nccl_ops.h"

#ifdef CT2_WITH_XPU
#  ifdef CT2_WITH_TENSOR_PARALLEL
#    include <mpi.h>
#    include <oneapi/ccl.h>
#    include <optional>
#    include "ctranslate2/devices.h"
#  endif
#  include "xpu/utils.h"
#endif

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

#ifdef CT2_WITH_XPU
#ifdef CT2_WITH_TENSOR_PARALLEL

    // Map CTranslate2 data types to oneCCL C API data types.
    onecclDataType_t getCclDataType(DataType type) {
      switch (type) {
      case DataType::BFLOAT16:
        return onecclBfloat16;
      case DataType::FLOAT16:
        return onecclFloat16;
      case DataType::FLOAT32:
        return onecclFloat;
      case DataType::INT32:
        return onecclInt;
      case DataType::INT8:
        return onecclInt8;
      default:
        throw std::invalid_argument("The current datatype " + std::to_string(static_cast<int>(type)) +
                                    " is not supported for the mode tensor parallel on XPU");
      }
    }

    // Map ReduceAll reduction ops to oneCCL C API reduction ops.
    onecclRedOp_t redop_to_ccl_op(ReduceAll::RED_OP op) {
      switch (op) {
      case ReduceAll::RED_OP::SUM:
        return onecclSum;
      case ReduceAll::RED_OP::PROD:
        return onecclProd;
      case ReduceAll::RED_OP::MAX:
        return onecclMax;
      case ReduceAll::RED_OP::MIN:
        return onecclMin;
      case ReduceAll::RED_OP::AVG:
        return onecclAvg;
      default:
        throw std::runtime_error("the current reduce operation " + std::to_string(static_cast<int>(op)) +
                                 " is not supported on XPU");
      }
    }

    // Return-value-check helper for oneCCL calls.
#define ONECCL_CHECK(call) do {                                             \
      onecclResult_t _r = (call);                                           \
      if (_r != onecclSuccess)                                              \
        throw std::runtime_error(std::string(#call " failed: ")            \
                                 + onecclGetErrorString(_r));               \
    } while (0)

    // Shared thread-local storage for the oneCCL communicator.
    // Lives outside get_comm() so destroy_ccl_comm() can access the
    // same instance without a separate declaration.
    // thread_local implies static storage at namespace scope.
    thread_local std::optional<onecclComm_t> ccl_comm;

    // Thread-local oneCCL communicator using the new NCCL-compatible C API.
    // Mirrors the NCCL communicator pattern in ScopedMPISetter::getNcclComm().
    // Bootstraps using MPI (already initialized by ScopedMPISetter).
    //
    // The C API auto-initializes on first call — no explicit ccl::init() needed.
    // The KVS bootstrap and SYCL device/context wrapping from the legacy C++
    // API are eliminated: onecclGetUniqueId + onecclCommInitRank replace them.
    // The SYCL queue is passed directly to collective calls as a void* stream.
    onecclComm_t& get_comm() {
      if (!ccl_comm.has_value()) {
        int my_rank = ScopedMPISetter::getCurRank();
        int n_ranks = ScopedMPISetter::getNRanks();

        // Generate a unique ID on rank 0 and broadcast via MPI.
        // The C API uses a fixed-size unique ID (struct, not KVS address).
        onecclUniqueId uid{};
        if (my_rank == 0) {
          ONECCL_CHECK(onecclGetUniqueId(&uid));
        }
        MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Associate this rank with its local XPU device (not global rank).
        ONECCL_CHECK(onecclSetDevice(static_cast<uint32_t>(ScopedMPISetter::getLocalRank())));

        // Create communicator (ccl::init() is auto-invoked internally).
        onecclComm_t comm_handle = nullptr;
        ONECCL_CHECK(onecclCommInitRank(&comm_handle, static_cast<size_t>(n_ranks), uid, my_rank));

        ccl_comm.emplace(comm_handle);
      }
      return *ccl_comm;
    }

    // Destroy the thread-local oneCCL communicator.
    // Must be called before MPI_Finalize() to avoid use-after-free.
    // Intended to be called from ScopedMPISetter::finalize().
    void destroy_ccl_comm() {
      if (ccl_comm.has_value() && *ccl_comm != nullptr) {
        onecclCommDestroy(*ccl_comm);
        ccl_comm.reset();
      }
    }

#endif  // CT2_WITH_TENSOR_PARALLEL
#endif  // CT2_WITH_XPU


    template <Device D, typename T>
    void ReduceAll::compute(const StorageView& input, StorageView& output) const {
#ifdef CT2_WITH_XPU
#ifdef CT2_WITH_TENSOR_PARALLEL
      dim_t data_size = input.size();
      onecclDataType_t ccl_dtype = getCclDataType(input.dtype());
      onecclRedOp_t ccl_op = redop_to_ccl_op(_reduce_op);
      onecclComm_t& comm = get_comm();

      // oneCCL C API takes a void* that points to a SYCL queue.
      sycl::queue queue = xpu::get_sycl_queue();

      onecclResult_t result = onecclAllReduce(
          const_cast<T*>(input.data<T>()),
          static_cast<void*>(output.data<T>()),
          static_cast<size_t>(data_size), ccl_dtype, ccl_op,
          comm, &queue);

      if (result != onecclSuccess) {
        if (ccl_op == onecclAvg) {
          // AVG reduction may not be supported on the ESIMD/scheduler backend.
          // Fall back to SUM + submit a SYCL kernel to divide by n_ranks.
          result = onecclAllReduce(
              const_cast<T*>(input.data<T>()),
              static_cast<void*>(output.data<T>()),
              static_cast<size_t>(data_size), ccl_dtype, onecclSum,
              comm, &queue);
          if (result != onecclSuccess) {
            throw std::runtime_error(std::string("onecclAllReduce (SUM fallback from AVG) failed: ")
                                     + onecclGetErrorString(result));
          }
          int n_ranks = ScopedMPISetter::getNRanks();
          auto* out = output.data<T>();
          queue.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(static_cast<size_t>(data_size)),
                           [=](sycl::id<1> i) {
              out[i] = static_cast<T>(static_cast<float>(out[i]) / n_ranks);
            });
          }).wait();
        } else {
          throw std::runtime_error(std::string("onecclAllReduce failed: ")
                                   + onecclGetErrorString(result));
        }
      } else {
        queue.wait();
      }
#else
      (void)input;
      (void)output;
      throw std::runtime_error("reduce all is not supported on XPU (compiled without tensor parallel support)");
#endif
#else
      (void)input;
      (void)output;
#endif
    }

    template <Device D, typename T>
    void GatherAll::compute(const StorageView& input, StorageView& output) const {
#ifdef CT2_WITH_XPU
#ifdef CT2_WITH_TENSOR_PARALLEL
      dim_t data_size = input.size();
      onecclDataType_t ccl_dtype = getCclDataType(input.dtype());
      onecclComm_t& comm = get_comm();

      sycl::queue queue = xpu::get_sycl_queue();

      // onecclAllGather takes sendcount per rank (same for all ranks).
      // Unlike the legacy C++ API's allgatherv, no recvcounts vector is needed.
      onecclResult_t result = onecclAllGather(
          const_cast<T*>(input.data<T>()),
          static_cast<void*>(output.data<T>()),
          static_cast<size_t>(data_size), ccl_dtype,
          comm, &queue);

      if (result != onecclSuccess) {
        throw std::runtime_error(std::string("onecclAllGather failed: ")
                                 + onecclGetErrorString(result));
      }
      queue.wait();
#else
      (void)input;
      (void)output;
      throw std::runtime_error("gather all is not supported on XPU (compiled without tensor parallel support)");
#endif
#else
      (void)input;
      (void)output;
#endif
    }

#define DECLARE_IMPL(T)                                                 \
        template void ReduceAll::compute<Device::XPU, T>(const StorageView&, \
                                                          StorageView&) const; \
        template void GatherAll::compute<Device::XPU, T>(const StorageView&, \
                                                          StorageView&) const;
    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
