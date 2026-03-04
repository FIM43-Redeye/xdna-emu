// SPDX-License-Identifier: MIT
//
// emu_debug.h -- Diagnostic logging macros for the XRT-emulator bridge.
//
// Two output paths:
//   1. xrt_core::message::send -- integrated with XRT logging, verbosity-gated
//      via xrt.ini [Runtime] verbosity=7.  Zero overhead when disabled.
//   2. XDNA_EMU_DEBUG env var -- force-enables stderr output without touching
//      xrt.ini.  Cached once on first use (no repeated getenv calls).
//
// Both paths can be active simultaneously.  By default (no env var, default
// xrt.ini verbosity=4/warning), EMU_DBG and EMU_INFO produce no output.

#pragma once

#include "core/common/message.h"

#include <cstdio>
#include <cstdlib>

namespace xdna_emu {
namespace detail {

inline bool emu_debug_enabled()
{
    static bool enabled = (std::getenv("XDNA_EMU_DEBUG") != nullptr);
    return enabled;
}

} // namespace detail
} // namespace xdna_emu

// Debug level -- only shown at xrt.ini verbosity >= 7.
#define EMU_DBG(fmt, ...) do { \
    xrt_core::message::send(xrt_core::message::severity_level::debug, \
                            "xdna-emu", fmt, ##__VA_ARGS__); \
    if (xdna_emu::detail::emu_debug_enabled()) \
        std::fprintf(stderr, "[EMU DBG] " fmt "\n", ##__VA_ARGS__); \
} while (0)

// Info level -- shown at xrt.ini verbosity >= 6.
#define EMU_INFO(fmt, ...) do { \
    xrt_core::message::send(xrt_core::message::severity_level::info, \
                            "xdna-emu", fmt, ##__VA_ARGS__); \
    if (xdna_emu::detail::emu_debug_enabled()) \
        std::fprintf(stderr, "[EMU INFO] " fmt "\n", ##__VA_ARGS__); \
} while (0)

// Warning level -- shown at default verbosity (>= 4).
#define EMU_WARN(fmt, ...) do { \
    xrt_core::message::send(xrt_core::message::severity_level::warning, \
                            "xdna-emu", fmt, ##__VA_ARGS__); \
    if (xdna_emu::detail::emu_debug_enabled()) \
        std::fprintf(stderr, "[EMU WARN] " fmt "\n", ##__VA_ARGS__); \
} while (0)
