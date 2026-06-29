// SPDX-License-Identifier: MIT
//
// bridge_hwctx_slot() - return the hardware-context slot index for a
// given xrt::hw_context.
//
// The slot id is required as the `context_id` argument to
// xrt::aie::device's AIE register/memory accessors (read_aie_reg etc.).
// There is no public XRT API that exposes it, so this is obtained via the
// public `xrt::hw_context::operator xrt_core::hwctx_handle*()` followed by
// the internal `hwctx_handle::get_slotidx()` virtual.
//
// That internal call needs XRT's source-tree shim header
// (core/common/shim/hwctx_handle.h), which is not shipped with the
// installed /opt/xilinx/xrt headers. To avoid dragging the source-tree
// include path (and its slightly different xrt_kernel.h / xrt_bo.h) into
// the rest of the runner, the implementation is isolated in hwctx_slot.cpp;
// this header declares only the public-surface signature.
#pragma once

#include "xrt/xrt_hw_context.h"

#include <cstdint>

uint16_t bridge_hwctx_slot(const xrt::hw_context& ctx);
