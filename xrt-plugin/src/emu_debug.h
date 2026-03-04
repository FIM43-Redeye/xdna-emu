// SPDX-License-Identifier: MIT
//
// emu_debug.h -- Diagnostic logging macros for the XRT-emulator bridge.
//
// Two output paths:
//   1. xrt_core::message::send -- integrated with XRT logging, verbosity-gated
//      via xrt.ini [Runtime] verbosity=7.  Zero overhead when disabled.
//   2. XDNA_EMU_LOG_LEVEL env var -- enables stderr output at the specified
//      level without touching xrt.ini.  Accepted values: error, warn, info,
//      debug (case-insensitive).  Also passed through to the Rust emulator
//      by pdev_emu::on_first_open().
//
// Both paths can be active simultaneously.  By default (no env var, default
// xrt.ini verbosity=4/warning), EMU_DBG and EMU_INFO produce no output.
//
// All stderr messages are prefixed [emu-plugin] to distinguish them from
// Rust emulator messages (which use the env_logger format).

#pragma once

#include "core/common/message.h"

#include <cstdio>
#include <cstdlib>
#include <strings.h>

namespace xdna_emu {
namespace detail {

/// Logging threshold for stderr output, derived from XDNA_EMU_LOG_LEVEL.
/// 0 = off, 1 = error, 2 = warn, 3 = info, 4 = debug.
inline int emu_log_level()
{
    static int level = [] {
        const char* env = std::getenv("XDNA_EMU_LOG_LEVEL");
        if (!env)
            return 0;  // off by default
        if (strcasecmp(env, "debug") == 0) return 4;
        if (strcasecmp(env, "info") == 0)  return 3;
        if (strcasecmp(env, "warn") == 0)  return 2;
        if (strcasecmp(env, "error") == 0) return 1;
        // Legacy: XDNA_EMU_LOG_LEVEL=1 (or any unrecognized value) = info
        return 3;
    }();
    return level;
}

} // namespace detail
} // namespace xdna_emu

// Debug level -- only shown at xrt.ini verbosity >= 7.
#define EMU_DBG(fmt, ...) do { \
    xrt_core::message::send(xrt_core::message::severity_level::debug, \
                            "xdna-emu", fmt, ##__VA_ARGS__); \
    if (xdna_emu::detail::emu_log_level() >= 4) \
        std::fprintf(stderr, "[emu-plugin dbg] " fmt "\n", ##__VA_ARGS__); \
} while (0)

// Info level -- shown at xrt.ini verbosity >= 6.
#define EMU_INFO(fmt, ...) do { \
    xrt_core::message::send(xrt_core::message::severity_level::info, \
                            "xdna-emu", fmt, ##__VA_ARGS__); \
    if (xdna_emu::detail::emu_log_level() >= 3) \
        std::fprintf(stderr, "[emu-plugin info] " fmt "\n", ##__VA_ARGS__); \
} while (0)

// Warning level -- shown at default verbosity (>= 4).
#define EMU_WARN(fmt, ...) do { \
    xrt_core::message::send(xrt_core::message::severity_level::warning, \
                            "xdna-emu", fmt, ##__VA_ARGS__); \
    if (xdna_emu::detail::emu_log_level() >= 2) \
        std::fprintf(stderr, "[emu-plugin warn] " fmt "\n", ##__VA_ARGS__); \
} while (0)
