// SPDX-License-Identifier: MIT
//
// Isolated translation unit: the ONLY place in the runner that includes
// XRT's internal shim header to reach hwctx_handle::get_slotidx().
//
// Build note: compiled with the xdna-driver XRT source tree on its include
// path (core/include + runtime_src) so that core/common/shim/hwctx_handle.h
// and its cascade resolve. The xrt::hw_context and xrt_core::hwctx_handle
// definitions used here are byte-identical to the installed XRT 2.23.0
// headers (verified by diffing the installed build commit against the source
// tree), so the type passed across the TU boundary is ABI-compatible.
#include "hwctx_slot.h"

#include "core/common/shim/hwctx_handle.h"

uint16_t bridge_hwctx_slot(const xrt::hw_context& ctx) {
    // Public conversion (xrt_hw_context.h) -> internal shim handle, then the
    // get_slotidx() virtual. slot_id is uint32_t; the AIE accessors take a
    // uint16_t context_id, so narrow it (slot ids are small).
    auto* handle = static_cast<xrt_core::hwctx_handle*>(ctx);
    return static_cast<uint16_t>(handle->get_slotidx());
}
