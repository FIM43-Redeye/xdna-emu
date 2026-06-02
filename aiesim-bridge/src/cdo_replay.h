// CDO op-stream decoder: the C++ twin of the Rust encode_cdo wire format
// (crates/xdna-emu-ffi/src/aiesim/backend.rs, mod cdo_tag). MATCHED PAIR -- the
// tag constants and field order below must stay identical to the encoder; both
// sides cross-reference each other. A tag mismatch is encoder/decoder drift and
// fails loudly.
//
// Replays a parsed CDO (device configuration: BD descriptors, routing, locks,
// core-enables) onto the cluster's register space via TIMED b_transport
// (ps->write32/read32). Runs on the driver SC_THREAD, the only context where
// b_transport (which wait()s on the AXI handshake) and time advance (wait()) are
// legal -- the backdoor reaches only a shadow store, not live registers (see the
// feasibility findings doc, 2026-06-02). MASK_POLL advances the kernel (wait())
// between live reads; DELAY steps time.
#pragma once

#include <cstddef>
#include <cstdint>

class ps_bridge;

namespace aiesim {

// Decode + replay a CDO op-stream. Returns 0 on success; 1 on decode error
// (truncated stream / unknown tag) or MASK_POLL timeout.
int cdo_replay(ps_bridge* ps, const uint8_t* ops, std::size_t len);

}  // namespace aiesim
