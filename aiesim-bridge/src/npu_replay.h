// NPU runtime-sequence (txn) op-stream decoder: the C++ twin of the Rust
// encode_npu wire format (crates/xdna-emu-ffi/src/aiesim/backend.rs, mod
// npu_tag). MATCHED PAIR -- tag constants and field order below must stay
// identical to the encoder; both sides cross-reference each other.
//
// This is the host-side command stream that drives shim-DMA transfers in/out of
// DDR and gates compute via locks. Unlike the CDO (static device config), it
// carries two ops that need resolution against runtime state:
//   * DdrPatch -- patch a host-buffer DDR address into a shim BD address word.
//   * Sync (dma_await_task) -- block until a shim DMA channel finishes.
// Everything runs on the driver SC_THREAD: register access is TIMED b_transport
// and Sync advances sim time (wait()) while polling, so the cluster's DMA/cores
// make concurrent progress (the genwrapper ps_main model).
//
// Addresses are NPU1 (partition-logical, base-less) and are translated to the
// absolute Versal cluster address via addr_remap.h before every access.
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

class ps_bridge;
class ddr_target;

namespace aiesim {

// Decode + replay an NPU runtime-sequence op-stream. `start_col` is the
// partition's physical start column; `host_buffers` are the registered (DDR
// addr, size) regions DdrPatch resolves against (arg_idx -> buffer). `ddr` is
// the host DDR model bound to the shim-DMA masters -- Sync watches its
// transaction counter for quiescence-based completion (see dma_wait). Returns 0
// on success; 1 on decode error, DdrPatch out-of-range, or Sync timeout.
int npu_replay(ps_bridge* ps, ddr_target* ddr, const uint8_t* ops, std::size_t len,
               uint8_t start_col,
               const std::vector<std::pair<uint64_t, std::size_t>>& host_buffers);

}  // namespace aiesim
