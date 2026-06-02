// CDO op-stream decoder: the C++ twin of the Rust encode_cdo wire format
// (crates/xdna-emu-ffi/src/aiesim/backend.rs, mod cdo_tag). MATCHED PAIR -- the
// tag constants and field order below must stay identical to the encoder; both
// sides cross-reference each other. A tag mismatch is encoder/decoder drift and
// fails loudly.
//
// Replays a parsed CDO (device configuration: BD descriptors, routing, locks,
// core-enables) onto the cluster's register space. Writes use the PS bridge's
// zero-time backdoor (transport_dbg): the service loop runs in sc_main, not an
// SC_THREAD, so a timed b_transport that wait()s would be illegal there -- and
// the II-B.1 selftest proved backdoor writes land in the cluster's registers.
// MASK_POLL advances the kernel (sc_start) between backdoor reads; DELAY steps
// time. (Whether backdoor config drives a live workload vs. needing timed writes
// from an SC_THREAD is the open question resolved by II-B.2b's end-to-end gate.)
#pragma once

#include <cstddef>
#include <cstdint>

class ps_bridge;

namespace aiesim {

// Decode + replay a CDO op-stream. Returns 0 on success; 1 on decode error
// (truncated stream / unknown tag) or MASK_POLL timeout.
int cdo_replay(ps_bridge* ps, const uint8_t* ops, std::size_t len);

}  // namespace aiesim
