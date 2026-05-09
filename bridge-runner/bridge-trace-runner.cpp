// SPDX-License-Identifier: MIT
//
// bridge-trace-runner.cpp -- Generic XRT runner for trace-instrumented
// AIE xclbins.
//
// Reads kernel argument metadata from the xclbin via
// xrt::xclbin::kernel::get_args() and allocates/binds buffers by
// semantic position (not by hardcoded group_id(N)). Any xclbin that
// follows the mlir-aie kernarg convention (opcode, instr, instr_size,
// inputs, outputs, optional ctrlpkts, optional trace) works without
// per-test source changes.
//
// Trace buffer identification: the installed mlir-aie toolchain tags
// the trace slot with a generic name (e.g., "bo4") rather than "trace".
// Instead of relying on the name string, the runner identifies the
// trace buffer as the LAST buffer kernarg after inputs/outputs are
// accounted for. This is position-robust across toolchain versions.
//
// Two invocation modes:
//
//   Single-shot (legacy):
//     bridge-trace-runner --xclbin A.xclbin --instr insts.bin \
//                         --trace-out trace.bin [--input ...] [...]
//
//   Batch (session):
//     bridge-trace-runner --batch-stdin [--xclbin A.xclbin] [--kernel NAME]
//     # Each line on stdin is a set of per-run args in the same CLI syntax
//     # (minus --xclbin/--kernel which are inherited from the outer CLI).
//     # Per line, the runner emits one JSON status object on stdout.
//     # When --xclbin varies across lines, the runner caches prepared-kernel
//     # state per xclbin so device/xclbin/hw_context/kernel are each built
//     # exactly once per unique xclbin across the session.
//
// Output: raw trace buffer contents written to --trace-out, ready to be
// parsed by tools/parse-trace.py.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"

namespace {

// AIE kernel opcode used by mlir-aie's launch convention (see bridge test
// examples at mlir-aie/test/npu-xrt/*/test.cpp).
constexpr uint64_t AIE_KERNEL_OPCODE = 3;

// Classifier FFI struct (must match Rust XdnaEmuKernargRole).
struct ClassifierRole {
    uint8_t arg_idx;
    uint8_t role;
    uint8_t _pad[2];
    uint32_t bd_reg_addr;
};
constexpr uint8_t ROLE_DATA_MM2S = 0;
constexpr uint8_t ROLE_DATA_S2MM = 1;
constexpr uint8_t ROLE_CTRLPKT   = 2;
// ROLE_UNKNOWN = 255 is defined in the Rust FFI but we don't branch on it
// explicitly on the C++ side -- any non-Ctrlpkt role is treated as data.

using classify_fn_t = int64_t (*)(const void*, uint64_t, const void*, uint64_t,
                                  ClassifierRole*, uint64_t);

// CLI args split into two scopes:
//   OuterArgs: set once per process. In single-shot mode these fully describe
//              the one run; in batch mode they supply session-level defaults
//              (xclbin, kernel, verbose) that each batch line can override.
//   RunArgs:   per-run parameters. Built from the outer CLI in single-shot
//              mode, or rebuilt per stdin line in batch mode.
struct OuterArgs {
    std::string xclbin;            // default xclbin when --batch-stdin lines omit one
    std::string kernel_name;       // empty = auto-detect single kernel
    bool batch_stdin = false;
    bool verbose = false;
};

struct RunArgs {
    std::string xclbin;            // may override outer default in batch mode
    std::string kernel_name;       // may override outer default in batch mode
    std::string instr_bin;
    std::string trace_out;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<std::string> ctrlpkts;
    // Tiles to core-reset before each kernel launch. Format: "col:row".
    // When BRIDGE_RUNNER_REUSE_CONTEXT=1, the runner prepends a
    // disable/reset/enable sequence to every insts.bin so kernel state
    // doesn't bleed across runs. When the context is rebuilt per-run
    // anyway (default), the preamble is a no-op (bytes are still
    // injected but harmless).
    std::vector<std::pair<uint8_t, uint8_t>> reset_tiles;

    // Per-lock init values to write before the core is re-enabled.
    // Format: col:row:lock_id:value. Mirrors what the CDO's init
    // sequence does at hw_context creation: zero all locks, then
    // overwrite the ones with non-zero MLIR-declared init values.
    // Without these, a core-only reset starts the kernel from
    // (locks=0) which deadlocks any kernel whose first core-side
    // action is acquiring a prod lock (the prod=2 init lets the core
    // claim two output buffers up front; with prod=0 there's nothing
    // to acquire and no producer to release it).
    struct LockInit { uint8_t col; uint8_t row; uint8_t lock_id; int8_t value; };
    std::vector<LockInit> lock_inits;

    // CDO blobs (mlir-aie's main_aie_cdo_*.bin) to replay as a
    // userspace preamble before each kernel launch. Transcoded to
    // insts.bin opcodes (Write32/MaskWrite/BlockWrite) and spliced
    // ahead of the original instruction stream. Multiple flags allowed,
    // applied in order -- typical usage is two paths: the init blob
    // (zero locks, set lock init values, configure DMA BDs and stream
    // switch routing, holds core in reset) followed by the enable blob
    // (single MaskWrite to clear the reset bit and start the core).
    // This is the userspace-only equivalent of re-running aie2_hwctx_start
    // and lets us reuse a single hw_context across many launches without
    // trace state drifting between runs.
    std::vector<std::string> cdo_preambles;
    // Default 1 MiB. Large enough that event-heavy workloads (lock-
    // intensive kernels, full-event sweeps) don't silently truncate
    // mid-trace. The trace unit has no backpressure -- once the BO
    // fills, further events are dropped on the floor and the comparison
    // gets attributed to "emulator didn't emit the event" when the real
    // cause was "buffer ran out." 1 MiB still fits comfortably in host
    // memory and NPU BO allocation limits.
    uint64_t trace_size_bytes = 1u << 20;

    // Override for the runner's "last buffer kernarg = trace BO"
    // heuristic. Set to the 0-indexed position of the trace BO among
    // the data buffer kernargs (bo0..boN-1, NOT counting the instr
    // buffer). Matches manifest.json's trace_ddr_id field.
    //
    // Why this exists: mlir-aie's IRON trace injection allocates an
    // extra buffer slot in the kernel signature for trace, but it
    // doesn't always end up at the last position -- some tests have
    // trailing unused/ctrlpkt slots after it. The "last BO" heuristic
    // saves a zero buffer in those cases. -1 = use heuristic.
    int trace_buf_idx = -1;

    // When set, configure compute tile (perf_col, perf_row) Performance_Counter0
    // to count ACTIVE_CORE cycles before run.start(), then read after run.wait().
    // Provides a measurement independent of the trace controller -- useful for
    // diagnosing trace-induced artifacts in cycle measurements.
    bool read_perf_counter = false;
    uint16_t perf_col = 0;
    uint16_t perf_row = 2;
};

void print_usage(const char* argv0) {
    std::fprintf(stderr,
        "usage:\n"
        "  %s --xclbin <path> --instr <insts.bin> --trace-out <path> \\\n"
        "     [--kernel <name>] [--input <bin>]... [--output <path>]... \\\n"
        "     [--ctrlpkt <bin>]... [--cdo-preamble <cdo.bin>]... \\\n"
        "     [--reset-tile col:row]... [--reset-lock col:row:lock:val]... \\\n"
        "     [--trace-buf-idx N] [--trace-size N] [-v]\n"
        "\n"
        "  %s --batch-stdin [--xclbin <path>] [--kernel <name>] [-v]\n"
        "     # read one command line per stdin line in the same syntax\n"
        "     # as above (minus --xclbin/--kernel unless overriding); emits\n"
        "     # a single-line JSON status per run on stdout.\n"
        "\n"
        "Generic XRT runner for trace-instrumented AIE xclbins.\n"
        "Kernel args are discovered from xclbin metadata -- no hardcoded\n"
        "group_id mapping. The trace buffer is identified as the last\n"
        "buffer kernarg after inputs/outputs.\n",
        argv0, argv0);
}

// Tokenise a stdin command line into argv-style tokens. Supports simple
// whitespace splitting plus "double-quoted" substrings so paths with
// spaces pass through intact. No escape handling -- callers just avoid
// embedded quotes in paths, which the trace sweep never produces.
std::vector<std::string> tokenize_line(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool in_quote = false;
    for (char c : line) {
        if (c == '"') {
            in_quote = !in_quote;
            continue;
        }
        if (!in_quote && (c == ' ' || c == '\t')) {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
            continue;
        }
        cur.push_back(c);
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// Parse a token list into {outer, run}. In batch-stdin mode the caller
// passes tokens from one stdin line and sets is_batch_line=true, which
// skips the single-shot required-args check and allows omitted
// --xclbin/--kernel (they inherit from outer). Returns 0 on success.
int parse_tokens(const std::vector<std::string>& tokens,
                 OuterArgs& outer, RunArgs& run,
                 bool is_batch_line, const char* argv0) {
    for (size_t i = 0; i < tokens.size(); ++i) {
        const std::string& a = tokens[i];
        auto need_val = [&](const char* flag) -> std::string {
            if (i + 1 >= tokens.size()) {
                throw std::runtime_error(std::string("error: ") + flag + " needs a value");
            }
            return tokens[++i];
        };
        if (a == "--xclbin")            run.xclbin = need_val("--xclbin");
        else if (a == "--kernel")       run.kernel_name = need_val("--kernel");
        else if (a == "--instr")        run.instr_bin = need_val("--instr");
        else if (a == "--trace-out")    run.trace_out = need_val("--trace-out");
        else if (a == "--input")        run.inputs.push_back(need_val("--input"));
        else if (a == "--output")       run.outputs.push_back(need_val("--output"));
        else if (a == "--ctrlpkt")      run.ctrlpkts.push_back(need_val("--ctrlpkt"));
        else if (a == "--reset-tile") {
            // Format: "col:row". Multiple --reset-tile flags allowed,
            // one per active core tile in the kernel.
            std::string spec = need_val("--reset-tile");
            auto colon = spec.find(':');
            if (colon == std::string::npos) {
                throw std::runtime_error(
                    "error: --reset-tile expects col:row, got '" + spec + "'");
            }
            int col = std::atoi(spec.substr(0, colon).c_str());
            int row = std::atoi(spec.substr(colon + 1).c_str());
            if (col < 0 || col > 0x7F || row < 0 || row > 0x1F) {
                throw std::runtime_error(
                    "error: --reset-tile col/row out of range: " + spec);
            }
            run.reset_tiles.emplace_back(static_cast<uint8_t>(col),
                                         static_cast<uint8_t>(row));
        }
        else if (a == "--cdo-preamble") {
            // Path to a CDO blob (mlir-aie main_aie_cdo_*.bin) to
            // transcode into the insts.bin preamble. Repeatable;
            // applied in order. Typical: --cdo-preamble init.bin
            // --cdo-preamble enable.bin.
            run.cdo_preambles.push_back(need_val("--cdo-preamble"));
        }
        else if (a == "--trace-buf-idx") {
            // 0-indexed position of the trace BO among data buffer
            // kernargs (bo0..boN-1). Matches manifest.json's
            // trace_ddr_id. -1 (default) keeps the legacy "last buffer
            // = trace" heuristic.
            std::string v = need_val("--trace-buf-idx");
            if (v.empty() || (v[0] == '-' && v.size() == 1)) {
                throw std::runtime_error(
                    "error: --trace-buf-idx must be an integer, got '" + v + "'");
            }
            char* end = nullptr;
            long val = std::strtol(v.c_str(), &end, 0);
            if (end == v.c_str() || *end != '\0' || val < -1 || val > 31) {
                throw std::runtime_error(
                    "error: --trace-buf-idx out of range [-1,31]: " + v);
            }
            run.trace_buf_idx = static_cast<int>(val);
        }
        else if (a == "--reset-lock") {
            // Format: col:row:lock_id:value. Value is signed 6-bit
            // (-64..63), per the AIE2 lock value field width. Multiple
            // --reset-lock flags allowed.
            std::string spec = need_val("--reset-lock");
            auto parts = std::vector<std::string>{};
            std::string cur;
            for (char c : spec) {
                if (c == ':') { parts.push_back(cur); cur.clear(); }
                else cur.push_back(c);
            }
            if (!cur.empty()) parts.push_back(cur);
            if (parts.size() != 4) {
                throw std::runtime_error(
                    "error: --reset-lock expects col:row:lock_id:value, got '"
                    + spec + "'");
            }
            int col = std::atoi(parts[0].c_str());
            int row = std::atoi(parts[1].c_str());
            int lid = std::atoi(parts[2].c_str());
            int val = std::atoi(parts[3].c_str());
            if (col < 0 || col > 0x7F || row < 0 || row > 0x1F
                || lid < 0 || lid >= 16 || val < -64 || val > 63) {
                throw std::runtime_error(
                    "error: --reset-lock fields out of range: " + spec);
            }
            run.lock_inits.push_back({static_cast<uint8_t>(col),
                                      static_cast<uint8_t>(row),
                                      static_cast<uint8_t>(lid),
                                      static_cast<int8_t>(val)});
        }
        else if (a == "--batch-stdin") {
            if (is_batch_line) {
                throw std::runtime_error("error: --batch-stdin is not allowed on a batch input line");
            }
            outer.batch_stdin = true;
        }
        else if (a == "--trace-size") {
            // strtoull silently wraps negatives to UINT64_MAX and accepts any
            // leading garbage -- validate explicitly so "-1" or "foo" fail loud.
            std::string val = need_val("--trace-size");
            if (val.empty() || val[0] == '-') {
                throw std::runtime_error(
                    "error: --trace-size must be a non-negative integer, got '" + val + "'");
            }
            char* end = nullptr;
            run.trace_size_bytes = std::strtoull(val.c_str(), &end, 0);
            if (end == val.c_str() || *end != '\0') {
                throw std::runtime_error(
                    "error: --trace-size must be a non-negative integer, got '" + val + "'");
            }
        }
        else if (a == "--read-perf-counter") {
            run.read_perf_counter = true;
        }
        else if (a == "--perf-tile") {
            // Format: col:row, e.g. "0:2". Defaults to (0,2).
            std::string val = need_val("--perf-tile");
            auto colon = val.find(':');
            if (colon == std::string::npos) {
                throw std::runtime_error(
                    "error: --perf-tile must be col:row, got '" + val + "'");
            }
            run.perf_col = static_cast<uint16_t>(std::stoi(val.substr(0, colon)));
            run.perf_row = static_cast<uint16_t>(std::stoi(val.substr(colon + 1)));
        }
        else if (a == "-v" || a == "--verbose") outer.verbose = true;
        else if (a == "-h" || a == "--help") { print_usage(argv0); std::exit(0); }
        else {
            throw std::runtime_error(std::string("error: unknown arg: ") + a);
        }
    }
    if (!is_batch_line) {
        // Single-shot: all three paths required. In batch mode the outer
        // CLI may carry only --xclbin/--kernel/--verbose/--batch-stdin.
        if (outer.batch_stdin) {
            // Batch: instr/trace-out come on stdin, not the outer CLI.
            if (!run.instr_bin.empty() || !run.trace_out.empty()) {
                std::fprintf(stderr, "warning: --instr/--trace-out on outer CLI are ignored in --batch-stdin mode\n");
                run.instr_bin.clear();
                run.trace_out.clear();
            }
        } else {
            if (run.xclbin.empty() || run.instr_bin.empty() || run.trace_out.empty()) {
                std::fprintf(stderr, "error: --xclbin, --instr, and --trace-out are required\n");
                print_usage(argv0);
                return 1;
            }
        }
        // Promote run-level xclbin/kernel to outer defaults so the batch loop
        // (if any) inherits them. Single-shot doesn't care which it reads from.
        if (!run.xclbin.empty() && outer.xclbin.empty()) outer.xclbin = run.xclbin;
        if (!run.kernel_name.empty() && outer.kernel_name.empty())
            outer.kernel_name = run.kernel_name;
    } else {
        // Batch line: inherit xclbin/kernel if not specified on this line.
        if (run.xclbin.empty()) run.xclbin = outer.xclbin;
        if (run.kernel_name.empty()) run.kernel_name = outer.kernel_name;
        if (run.xclbin.empty() || run.instr_bin.empty() || run.trace_out.empty()) {
            throw std::runtime_error(
                "error: batch line needs --instr, --trace-out "
                "(and --xclbin if not inherited from outer CLI)");
        }
    }
    return 0;
}

int parse_cli(int argc, char** argv, OuterArgs& outer, RunArgs& run) {
    std::vector<std::string> tokens;
    tokens.reserve(static_cast<size_t>(argc - 1));
    for (int i = 1; i < argc; ++i) tokens.emplace_back(argv[i]);
    try {
        return parse_tokens(tokens, outer, run, /*is_batch_line=*/false, argv[0]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        print_usage(argv[0]);
        return 1;
    }
}

struct KernArgInfo {
    std::string name;
    std::string host_type;   // e.g., "uint32_t*", "uint32_t", "uint64_t*"
    size_t index = 0;
    uint64_t size = 0;       // in bytes; for pointers, the buffer size
    uint64_t offset = 0;
    bool is_scalar() const { return host_type.find('*') == std::string::npos; }
};

std::vector<KernArgInfo> read_kernel_args(
    const xrt::xclbin& xclbin,
    const std::string& kernel_name_hint,
    std::string& chosen_kernel_name,
    bool verbose)
{
    auto kernels = xclbin.get_kernels();
    if (kernels.empty()) {
        throw std::runtime_error("xclbin has no kernels");
    }
    const xrt::xclbin::kernel* picked = nullptr;
    for (const auto& k : kernels) {
        if (kernel_name_hint.empty() ||
            k.get_name().find(kernel_name_hint) != std::string::npos) {
            picked = &k;
            break;
        }
    }
    if (!picked) {
        throw std::runtime_error("no kernel matches --kernel hint: " + kernel_name_hint);
    }
    chosen_kernel_name = picked->get_name();
    std::vector<KernArgInfo> out;
    for (const auto& a : picked->get_args()) {
        KernArgInfo k;
        k.name = a.get_name();
        k.host_type = a.get_host_type();
        k.index = a.get_index();
        k.size = a.get_size();
        k.offset = a.get_offset();
        out.push_back(k);
        if (verbose) {
            std::fprintf(stderr,
                "  arg[%zu] name=%-12s host_type=%-16s size=%lu offset=%lu %s\n",
                k.index, k.name.c_str(), k.host_type.c_str(),
                (unsigned long)k.size, (unsigned long)k.offset,
                k.is_scalar() ? "(scalar)" : "(buffer)");
        }
    }
    return out;
}

// AIE2 NPU register addressing: addr = (col << 25) | (row << 20) | offset.
// Constants confirmed via xdna-archspec/aie-device-models.json column_shift
// and row_shift fields, and against the NPU instruction parser at
// src/npu/executor.rs decode_addr_to_tile_relative().
constexpr uint32_t AIE2_COL_SHIFT = 25;
constexpr uint32_t AIE2_ROW_SHIFT = 20;
constexpr uint32_t AIE2_CORE_CTRL_OFFSET = 0x00032000;
constexpr uint32_t AIE2_LOCK_VALUE_BASE = 0x0001F000;
constexpr uint32_t AIE2_LOCK_STRIDE = 0x10;
constexpr uint32_t AIE2_LOCKS_PER_TILE = 16;

inline uint32_t tile_addr(uint8_t col, uint8_t row, uint32_t offset) {
    return (static_cast<uint32_t>(col) << AIE2_COL_SHIFT)
         | (static_cast<uint32_t>(row) << AIE2_ROW_SHIFT)
         | offset;
}

// Append a MaskWrite (opcode 0x03) op to the instruction word vector.
// Layout (28 bytes = 7 u32s):
//   [0] hdr   = 0x00000003 (opcode in low byte, rest padding)
//   [1] pad   = 0
//   [2,3] reg_addr (u64 LE; high word zero for 32-bit NPU addrs)
//   [4] value
//   [5] mask
//   [6] size  = 28
// Authority: src/npu/parser.rs (decoder) and mlir-aie's
// AIETargetNPU.cpp::appendMaskWrite32() (encoder).
inline void emit_maskwrite(std::vector<uint32_t>& out,
                           uint32_t reg_addr, uint32_t value, uint32_t mask) {
    out.push_back(0x00000003);
    out.push_back(0);
    out.push_back(reg_addr);
    out.push_back(0);
    out.push_back(value);
    out.push_back(mask);
    out.push_back(28);
}

// Append a Write32 (opcode 0x00) op. Layout (24 bytes = 6 u32s):
//   [0] hdr   = 0x00000000
//   [1] pad   = 0
//   [2,3] reg_addr (u64 LE)
//   [4] value
//   [5] size  = 24
inline void emit_write32(std::vector<uint32_t>& out,
                         uint32_t reg_addr, uint32_t value) {
    out.push_back(0x00000000);
    out.push_back(0);
    out.push_back(reg_addr);
    out.push_back(0);
    out.push_back(value);
    out.push_back(24);
}

// Append a BlockWrite (opcode 0x01) op. Layout (16 + N*4 bytes):
//   [0] hdr     = 0x00000001
//   [1] pad     = 0
//   [2] reg_off (u32 -- BlockWrite uses a 32-bit offset, unlike
//                Write32/MaskWrite which use a u64-aligned reg_addr)
//   [3] size    = 16 + payload_bytes (the full instruction size)
//   [4..]       N data words written to consecutive 4-byte addresses.
// Authority: src/npu/parser.rs decode of NpuOpcode::BlockWrite (header
// is 8 bytes opcode+pad, then u32 reg_off, then u32 size, then payload).
inline void emit_blockwrite(std::vector<uint32_t>& out,
                            uint32_t reg_off,
                            const uint32_t* data, size_t n) {
    out.push_back(0x00000001);
    out.push_back(0);
    out.push_back(reg_off);
    out.push_back(static_cast<uint32_t>(16 + n * 4));
    out.insert(out.end(), data, data + n);
}

// Op size (in u32 words) for each emitter, used to count num_ops_extra.
constexpr uint32_t MASKWRITE_WORDS = 7;
constexpr uint32_t WRITE32_WORDS = 6;
// BlockWrite is variable-length; only the fixed-header word count is constant.
constexpr uint32_t BLOCKWRITE_HEADER_WORDS = 4;

// Build the reset preamble for a set of active core tiles. For each
// tile the sequence is the four mask-writes that aie-rt's CoreReset
// API performs against the Core Control register at offset 0x32000:
//   1. disable core (bit 0 = 0)
//   2. assert reset (bit 1 = 1)
//   3. deassert reset (bit 1 = 0)
//   4. re-enable core (bit 0 = 1)
// This eliminates the alternating-TIMEOUT pattern when reusing a
// single hw_context across kernel launches, EXCEPT for kernels that
// use cascade flows -- there's no user-space register to drain the
// cascade-FIFO state, so cascade kernels still need hw_context
// teardown. Detection of that case lives in the caller (typically
// trace-sweep.py builds the reset-tile list from MLIR analysis and
// omits it for cascade kernels).
//
// The returned words are appended to the existing instruction stream;
// the runner increments num_ops and total_size in the header before
// submission.
std::vector<uint32_t> build_reset_preamble(
    const std::vector<std::pair<uint8_t, uint8_t>>& tiles,
    const std::vector<RunArgs::LockInit>& lock_inits)
{
    std::vector<uint32_t> out;
    out.reserve(tiles.size() * (3 * MASKWRITE_WORDS
                                + AIE2_LOCKS_PER_TILE * WRITE32_WORDS)
                + lock_inits.size() * WRITE32_WORDS);
    // Mirror the CDO init sequence exactly (extracted from
    // main_aie_cdo_init.bin for add_one_objFifo and confirmed against
    // aie-rt's reginit table in xaiemlgbl_reginit.c:2454):
    //
    //   1. Assert core reset (and KEEP it asserted -- the CDO holds the
    //      core in reset throughout the entire init phase). The CDO
    //      doesn't even bother clearing the enable bit explicitly
    //      because the reset bit dominates.
    //   2. While core is in reset, write 0 to all 16 lock value
    //      registers in each tile's memory module.
    //   3. Apply per-lock non-zero init values from MLIR (typically
    //      prod-lock init=2 for objFifo double-buffer kernels).
    //   4. Deassert reset and re-enable the core.
    //
    // Earlier attempts that toggled reset on/off around lock writes,
    // or skipped lock writes entirely, either hung the kernel or
    // produced trace variance across runs. This sequence matches
    // exactly what hw_context creation does, so traces should be
    // bit-identical to a fresh-context baseline.
    for (auto [col, row] : tiles) {
        uint32_t ctrl = tile_addr(col, row, AIE2_CORE_CTRL_OFFSET);
        emit_maskwrite(out, ctrl, /*val=*/2, /*mask=*/2);   // assert reset
        for (uint32_t n = 0; n < AIE2_LOCKS_PER_TILE; ++n) {
            uint32_t reg = tile_addr(col, row,
                AIE2_LOCK_VALUE_BASE + n * AIE2_LOCK_STRIDE);
            emit_write32(out, reg, /*value=*/0);
        }
    }
    // Per-lock non-zero init values come AFTER the zero-pass, also
    // while core is held in reset. The 6-bit signed value is written
    // into the low 6 bits of the lock value register; we mask to that
    // width and reinterpret as unsigned for the wire-level write.
    for (const auto& li : lock_inits) {
        uint32_t reg = tile_addr(li.col, li.row,
            AIE2_LOCK_VALUE_BASE + li.lock_id * AIE2_LOCK_STRIDE);
        uint32_t v = static_cast<uint32_t>(li.value) & 0x3F;
        emit_write32(out, reg, v);
    }
    // Finally release reset and re-enable the core. We use mask=3 with
    // value=1 (enable=1, reset=0) in a single mask-write, which
    // matches what enable.bin does (one MaskWrite to switch both bits
    // simultaneously is safer than two separate writes).
    for (auto [col, row] : tiles) {
        uint32_t ctrl = tile_addr(col, row, AIE2_CORE_CTRL_OFFSET);
        emit_maskwrite(out, ctrl, /*val=*/1, /*mask=*/3);   // enable=1, reset=0
    }
    return out;
}

// Splice a freshly built reset preamble into a runtime-sequence
// (insts.bin) word stream loaded from disk. The original header
// reports num_ops and total_size_bytes; we update both.
//
// Header layout (16 bytes = 4 u32s):
//   [0] magic         = 0x06030100
//   [1] flags         (preserved)
//   [2] num_ops
//   [3] total_size_bytes
void splice_reset_preamble(std::vector<uint32_t>& instr_words,
                           const std::vector<uint32_t>& preamble_words)
{
    if (instr_words.size() < 4) return;     // bad input -- caller will error elsewhere
    if (preamble_words.empty()) return;
    constexpr size_t HEADER_WORDS = 4;
    constexpr uint32_t MAGIC = 0x06030100;
    if (instr_words[0] != MAGIC) return;    // not our format -- leave alone
    // Walk the preamble word-stream to count actual ops, since op size
    // varies (write32 = 6 words, maskwrite = 7 words). Each op's size
    // in BYTES is in its size field, which is at a different word
    // index per opcode -- but its position is always the *last* word
    // of that op (parser.rs:371-406 confirms). Easier and more robust:
    // walk by reading each op's size field at an opcode-dependent
    // position.
    uint32_t num_ops_extra = 0;
    size_t i = 0;
    while (i < preamble_words.size()) {
        uint32_t opcode = preamble_words[i] & 0xFF;
        size_t op_words;
        switch (opcode) {
            case 0x00: op_words = WRITE32_WORDS; break;
            case 0x03: op_words = MASKWRITE_WORDS; break;
            case 0x01: {
                // BlockWrite: size in bytes lives in the 4th word
                // (header, pad, reg_off, size_bytes, ...payload).
                if (i + BLOCKWRITE_HEADER_WORDS > preamble_words.size()) return;
                uint32_t size_bytes = preamble_words[i + 3];
                if (size_bytes < 16 || size_bytes % 4 != 0) return;
                op_words = size_bytes / 4;
                break;
            }
            default:
                // Unknown opcode -- bail on counting; let the FW
                // complain about the malformed stream rather than
                // silently miscounting.
                return;
        }
        ++num_ops_extra;
        i += op_words;
    }
    uint32_t bytes_extra = static_cast<uint32_t>(preamble_words.size() * 4);
    instr_words[2] += num_ops_extra;
    instr_words[3] += bytes_extra;
    // Insert preamble after the 4-word header, before the original ops.
    instr_words.insert(instr_words.begin() + HEADER_WORDS,
                       preamble_words.begin(), preamble_words.end());
}

// CDO opcodes used by mlir-aie's main_aie_cdo_*.bin blobs. Authority:
// xdna-emu/src/parser/cdo/syntax.rs CdoOpcode enum.
constexpr uint16_t CDO_OP_END_MARK    = 0x100;
constexpr uint16_t CDO_OP_MASK_WRITE  = 0x102;
constexpr uint16_t CDO_OP_WRITE       = 0x103;
constexpr uint16_t CDO_OP_DMA_WRITE   = 0x105;
constexpr uint16_t CDO_OP_MASK_WRITE64 = 0x107;
constexpr uint16_t CDO_OP_WRITE64     = 0x108;
constexpr uint16_t CDO_OP_NOP         = 0x111;
constexpr uint16_t CDO_OP_MARKER      = 0x119;

constexpr uint32_t CDO_HEADER_BYTES   = 20;
constexpr uint32_t CDO_MAGIC_CDO      = 0x004f4443; // "CDO\0"
constexpr uint32_t CDO_MAGIC_XLNX     = 0x584c4e58; // "XLNX"

// Transcode a CDO blob (the kind mlir-aie emits as main_aie_cdo_*.bin)
// into an insts.bin word stream. The point: the init CDO contains all
// the state setup that hw_context creation normally applies via the
// firmware mailbox -- core reset, lock zeroing + init values, DMA BD
// programming, stream-switch routing, and trace unit setup. By
// replaying it as user-space insts.bin ops we re-establish that state
// per kernel launch, which is the only way to hold trace output stable
// when reusing a single hw_context across many launches.
//
// Mapping:
//   CDO Write64  (0x108)   -> insts.bin Write32   (0x00)
//   CDO Write    (0x103)   -> insts.bin Write32   (0x00)
//   CDO MaskWrite64 (0x107) -> insts.bin MaskWrite (0x03)
//   CDO MaskWrite  (0x102)  -> insts.bin MaskWrite (0x03)
//   CDO DmaWrite (0x105)   -> insts.bin BlockWrite (0x01)
//   CDO Nop / EndMark / Marker -> skipped
//
// Unsupported CDO ops (MaskPoll, Delay) throw -- these don't appear in
// the static config CDOs we replay, and silently dropping them would
// produce a subtly-broken preamble.
//
// 64-bit CDO addresses with a non-zero high word are rejected: AIE NPU
// register space is 32 bits; a non-zero hi means the blob targets DDR
// above 4 GiB, which the AIE controller can't reach via insts.bin
// register writes anyway.
std::vector<uint32_t> transcode_cdo_to_insts(const std::vector<uint8_t>& cdo,
                                             const std::string& path_for_msg)
{
    if (cdo.size() < CDO_HEADER_BYTES) {
        throw std::runtime_error(path_for_msg + ": CDO blob too small ("
                                 + std::to_string(cdo.size()) + " bytes)");
    }
    uint32_t ident, cdo_length_words;
    std::memcpy(&ident, cdo.data() + 4, 4);
    std::memcpy(&cdo_length_words, cdo.data() + 12, 4);
    if (ident != CDO_MAGIC_CDO && ident != CDO_MAGIC_XLNX) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "0x%08x", ident);
        throw std::runtime_error(path_for_msg + ": bad CDO magic " + buf);
    }
    size_t end = std::min<size_t>(cdo.size(),
        static_cast<size_t>(CDO_HEADER_BYTES) + cdo_length_words * 4);

    std::vector<uint32_t> out;
    size_t off = CDO_HEADER_BYTES;
    while (off + 4 <= end) {
        uint32_t cmd_word;
        std::memcpy(&cmd_word, cdo.data() + off, 4);
        off += 4;
        uint16_t opcode = static_cast<uint16_t>(cmd_word & 0xFFFF);
        uint8_t inline_len = static_cast<uint8_t>((cmd_word >> 16) & 0xFF);
        uint32_t payload_len;
        if (inline_len == 0xFF) {
            // CDO v2 extended length: payload size lives in the next word.
            if (off + 4 > end) break;
            std::memcpy(&payload_len, cdo.data() + off, 4);
            off += 4;
        } else {
            payload_len = inline_len;
        }
        if (off + static_cast<size_t>(payload_len) * 4 > end) break;
        const uint32_t* payload =
            reinterpret_cast<const uint32_t*>(cdo.data() + off);
        off += static_cast<size_t>(payload_len) * 4;

        auto require = [&](uint32_t need, const char* op_name) {
            if (payload_len < need) {
                throw std::runtime_error(path_for_msg + ": CDO " + op_name
                    + " has payload_len=" + std::to_string(payload_len)
                    + ", expected >=" + std::to_string(need));
            }
        };
        auto check_addr_hi = [&](uint32_t addr_hi, const char* op_name) {
            if (addr_hi != 0) {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "0x%08x", addr_hi);
                throw std::runtime_error(path_for_msg + ": CDO " + op_name
                    + " addr_hi=" + buf + " (>4GiB) not supported");
            }
        };

        switch (opcode) {
        case CDO_OP_END_MARK:
        case CDO_OP_NOP:
        case CDO_OP_MARKER:
            break;

        case CDO_OP_WRITE:
            require(2, "Write");
            emit_write32(out, payload[0], payload[1]);
            break;

        case CDO_OP_WRITE64:
            require(3, "Write64");
            check_addr_hi(payload[0], "Write64");
            emit_write32(out, payload[1], payload[2]);
            break;

        case CDO_OP_MASK_WRITE:
            require(3, "MaskWrite");
            // CDO order is [addr, mask, value]; insts.bin emit order is
            // (addr, value, mask) per emit_maskwrite signature.
            emit_maskwrite(out, payload[0], payload[2], payload[1]);
            break;

        case CDO_OP_MASK_WRITE64:
            require(4, "MaskWrite64");
            check_addr_hi(payload[0], "MaskWrite64");
            // [addr_hi, addr_lo, mask, value]
            emit_maskwrite(out, payload[1], payload[3], payload[2]);
            break;

        case CDO_OP_DMA_WRITE:
            require(2, "DmaWrite");
            check_addr_hi(payload[0], "DmaWrite");
            // payload = [addr_hi, addr_lo, data...]
            emit_blockwrite(out, payload[1],
                            payload + 2,
                            static_cast<size_t>(payload_len) - 2);
            break;

        default: {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "0x%03x", opcode);
            throw std::runtime_error(path_for_msg + ": CDO opcode " + buf
                + " not supported in preamble replay");
        }
        }
    }
    return out;
}

// Slurp a CDO file from disk.
std::vector<uint8_t> load_cdo_blob(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open CDO file: " + path);
    std::streamsize bytes = f.tellg();
    if (bytes < 0) throw std::runtime_error("cannot stat CDO file: " + path);
    f.seekg(0);
    std::vector<uint8_t> data(static_cast<size_t>(bytes));
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

/// Load an instruction binary as a vector of uint32_t words.
std::vector<uint32_t> load_instr_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open instr file: " + path);
    std::streamsize bytes = f.tellg();
    if (bytes < 0) throw std::runtime_error("cannot stat instr file: " + path);
    if (bytes % 4 != 0) {
        throw std::runtime_error("instr file size not a multiple of 4 bytes: " + path);
    }
    f.seekg(0);
    std::vector<uint32_t> words(static_cast<size_t>(bytes) / 4);
    f.read(reinterpret_cast<char*>(words.data()), bytes);
    return words;
}

/// Walk insts.bin's DdrPatch ops and return the largest arg_idx, which is
/// the BO regmap slot the firmware patches for the trace shim DMA. The
/// trace BO is always added last by the inject flow, so its arg_idx
/// dominates. Returns -1 if no DdrPatch ops are found, the file isn't a
/// recognizable insts.bin, or the structure is malformed -- callers should
/// fall back to the legacy "last buffer kernarg = trace" heuristic in that
/// case. Mirrors tools/trace-sweep.py::_discover_trace_buf_idx so the
/// standalone runner default lines up with how trace-sweep already drives
/// --trace-buf-idx.
int discover_trace_buf_idx_from_insts(const std::string& path) {
    constexpr uint32_t INSTS_HEADER_BYTES = 16;
    constexpr uint32_t INSTS_MAGIC = 0x06030100u;
    constexpr uint8_t  OPCODE_WRITE32     = 0x00;
    constexpr uint8_t  OPCODE_BLOCKWRITE  = 0x01;
    constexpr uint8_t  OPCODE_MASKWRITE   = 0x03;
    constexpr uint8_t  OPCODE_DDR_PATCH   = 0x81;

    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return -1;
    std::streamsize bytes = f.tellg();
    if (bytes < static_cast<std::streamsize>(INSTS_HEADER_BYTES)) return -1;
    f.seekg(0);
    std::vector<uint8_t> data(static_cast<size_t>(bytes));
    if (!f.read(reinterpret_cast<char*>(data.data()), bytes)) return -1;

    uint32_t magic = static_cast<uint32_t>(data[0])
                   | (static_cast<uint32_t>(data[1]) << 8)
                   | (static_cast<uint32_t>(data[2]) << 16)
                   | (static_cast<uint32_t>(data[3]) << 24);
    if (magic != INSTS_MAGIC) return -1;

    uint32_t total_size = static_cast<uint32_t>(data[12])
                        | (static_cast<uint32_t>(data[13]) << 8)
                        | (static_cast<uint32_t>(data[14]) << 16)
                        | (static_cast<uint32_t>(data[15]) << 24);
    size_t end = std::min<size_t>(data.size(), total_size);
    size_t off = INSTS_HEADER_BYTES;
    int max_arg_idx = -1;
    while (off + 4 <= end) {
        uint8_t opcode = data[off];
        size_t op_size;
        if (opcode == OPCODE_WRITE32) {
            op_size = 24;
        } else if (opcode == OPCODE_BLOCKWRITE) {
            if (off + 16 > end) break;
            op_size = static_cast<uint32_t>(data[off + 12])
                    | (static_cast<uint32_t>(data[off + 13]) << 8)
                    | (static_cast<uint32_t>(data[off + 14]) << 16)
                    | (static_cast<uint32_t>(data[off + 15]) << 24);
        } else if (opcode == OPCODE_MASKWRITE) {
            op_size = 28;
        } else if (opcode == OPCODE_DDR_PATCH) {
            op_size = 48;
            // Payload starts at off+8; arg_idx is one byte at payload[24].
            if (off + 8 + 24 < end) {
                int arg_idx = data[off + 8 + 24];
                if (arg_idx > max_arg_idx) max_arg_idx = arg_idx;
            }
        } else {
            // Unknown opcode -- can't trust the rest, stop walking.
            break;
        }
        off += op_size;
    }
    return max_arg_idx;
}

/// Read the contents of a file into a BO's host-mapped memory, up to the BO's
/// size.  If the file is shorter than expected_size, the tail is left as
/// whatever the caller initialized the BO with (typically zero).  If the file
/// is LARGER than expected_size, the excess is silently discarded -- warn on
/// stderr so size-mismatched inputs (e.g. f64 data passed to an f32 kernarg)
/// surface as an obvious diagnostic rather than a silent half-load.
void load_bo_from_file(xrt::bo& bo, const std::string& path, size_t expected_size) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open input file: " + path);
    std::streamsize file_size = f.tellg();
    if (file_size < 0) throw std::runtime_error("cannot stat input file: " + path);
    if (static_cast<size_t>(file_size) > expected_size) {
        std::fprintf(stderr,
            "warning: %s is %lld bytes but kernarg expects %zu; discarding excess\n",
            path.c_str(), static_cast<long long>(file_size), expected_size);
    }
    f.seekg(0);
    f.read(static_cast<char*>(bo.map<void*>()), static_cast<std::streamsize>(expected_size));
}

/// Positional classification of kernargs into their semantic roles.
/// The installed mlir-aie toolchain uses generic names (bo0, bo1, ...) rather
/// than a distinguished "trace" name, so we dispatch by position:
///
///   scalars[0]  = opcode     (arg 0)
///   buffers[0]  = instr      (arg 1, first buffer)
///   scalars[1]  = ninstr     (arg 2)
///   buffers[1..N-2] = middle data buffers (inputs then outputs)
///   buffers[N-1] = trace     (last buffer)
struct KernArgPlan {
    size_t opcode_idx      = SIZE_MAX;  // first scalar
    size_t instr_idx       = SIZE_MAX;  // first buffer
    size_t instr_size_idx  = SIZE_MAX;  // second scalar
    std::vector<size_t> middle_buf_indices;  // buffers between instr and trace
    size_t trace_idx       = SIZE_MAX;  // last buffer
};

KernArgPlan classify_kernargs(const std::vector<KernArgInfo>& kargs,
                              int trace_buf_idx_override = -1) {
    KernArgPlan plan;
    std::vector<size_t> scalar_indices;
    std::vector<size_t> buffer_indices;
    for (const auto& k : kargs) {
        if (k.is_scalar()) scalar_indices.push_back(k.index);
        else buffer_indices.push_back(k.index);
    }
    // Expected layout (mlir-aie bridge-test convention):
    //   scalars: opcode (u64), ninstr (u32) -- exactly 2
    //   buffers: instr, [bo0..boK-1] (at least one of which is trace) -- >=2
    // Tightening scalar_indices to exactly 2: if a future test adds a third
    // scalar (tuning param, tile id, etc.), the runner needs to grow a new
    // positional slot for it rather than silently leave it unbound.
    if (scalar_indices.size() != 2) {
        throw std::runtime_error(
            "expected exactly 2 scalar kernargs (opcode, ninstr); got " +
            std::to_string(scalar_indices.size()) +
            " (runner needs updating to handle additional scalars)");
    }
    if (buffer_indices.size() < 2) {
        throw std::runtime_error(
            "expected at least 2 buffer kernargs (instr + trace); got " +
            std::to_string(buffer_indices.size()));
    }
    plan.opcode_idx     = scalar_indices[0];
    plan.instr_size_idx = scalar_indices[1];
    plan.instr_idx      = buffer_indices[0];

    // Trace BO selection. Data buffers are buffer_indices[1..N-1] (i.e.,
    // excluding instr at [0]), so a 0-indexed data-buffer override of N
    // selects buffer_indices[N+1].
    size_t trace_buf_arr_pos;  // index into buffer_indices[]
    if (trace_buf_idx_override >= 0) {
        size_t want = static_cast<size_t>(trace_buf_idx_override) + 1;
        if (want >= buffer_indices.size()) {
            throw std::runtime_error(
                "--trace-buf-idx " + std::to_string(trace_buf_idx_override)
                + " out of range; kernel has " +
                std::to_string(buffer_indices.size() - 1) + " data buffers");
        }
        trace_buf_arr_pos = want;
    } else {
        trace_buf_arr_pos = buffer_indices.size() - 1;  // legacy: last
    }
    plan.trace_idx = buffer_indices[trace_buf_arr_pos];

    // Middle buffers: every data buffer that isn't instr (idx 0) and
    // isn't trace.
    for (size_t i = 1; i < buffer_indices.size(); ++i) {
        if (i == trace_buf_arr_pos) continue;
        plan.middle_buf_indices.push_back(buffer_indices[i]);
    }
    return plan;
}

/// Try to load libxdna_emu.so and resolve xdna_emu_classify_kernargs.
/// Returns nullptr if the library is absent or the symbol is missing;
/// callers then fall back to positional binding.
classify_fn_t try_load_classifier(bool verbose) {
    auto try_path = [&](const std::string& path) -> classify_fn_t {
        void* h = dlopen(path.c_str(), RTLD_LAZY);
        if (!h) return nullptr;
        auto fn = reinterpret_cast<classify_fn_t>(dlsym(h, "xdna_emu_classify_kernargs"));
        if (verbose && fn) std::fprintf(stderr, "  classifier loaded from %s\n", path.c_str());
        return fn;
    };
    if (const char* dir = std::getenv("XDNA_EMU_DIR"); dir && *dir) {
        if (auto fn = try_path(std::string(dir) + "/libxdna_emu.so")) return fn;
    }
    return try_path("libxdna_emu.so");
}

/// Classify kernarg roles via the Rust FFI. Returns empty on failure.
/// Re-reads the xclbin bytes on each call -- cheap (~ms for a ~1 MB file)
/// and keeps the per-run classification independent of any outer caching.
std::vector<ClassifierRole> classify_kernargs_via_ffi(
    classify_fn_t fn,
    const std::string& xclbin_path,
    const std::vector<uint32_t>& instr_words,
    bool verbose)
{
    if (!fn) return {};
    std::ifstream xf(xclbin_path, std::ios::binary | std::ios::ate);
    if (!xf) return {};
    std::streamsize xsz = xf.tellg();
    if (xsz <= 0) return {};
    xf.seekg(0);
    std::vector<uint8_t> xbytes(static_cast<size_t>(xsz));
    xf.read(reinterpret_cast<char*>(xbytes.data()), xsz);

    const uint8_t* ibytes = reinterpret_cast<const uint8_t*>(instr_words.data());
    uint64_t isz = instr_words.size() * sizeof(uint32_t);

    int64_t count = fn(xbytes.data(), xbytes.size(), ibytes, isz, nullptr, 0);
    if (count < 0) {
        if (verbose) std::fprintf(stderr, "  classifier returned error %ld\n", (long)count);
        return {};
    }
    std::vector<ClassifierRole> out(static_cast<size_t>(count));
    int64_t filled = fn(xbytes.data(), xbytes.size(), ibytes, isz,
                        out.data(), out.size());
    if (filled < 0) return {};
    out.resize(static_cast<size_t>(filled));
    return out;
}

const char* role_name(uint8_t role) {
    switch (role) {
        case ROLE_DATA_MM2S: return "data_mm2s";
        case ROLE_DATA_S2MM: return "data_s2mm";
        case ROLE_CTRLPKT:   return "ctrlpkt";
        default:             return "unknown";
    }
}

// --------------------------------------------------------------------------
// Per-xclbin state held across runs in batch mode.
// --------------------------------------------------------------------------

// Reusing a single xrt::hw_context across repeated kernel launches in
// the same process produces an *alternating* TIMEOUT pattern: run 0
// succeeds, run 1 times out (state=8), run 2 succeeds, run 3 aborts
// (state=6), etc. This reproduces across tests (add_one_objFifo,
// add_one_using_dma), with BO pooling, with per-run BO alloc, and with
// inter-run sleeps -- the only thing that eliminates it is recreating
// the hw_context before each launch. Reference tests that loop over
// kernel invocations within a single hw_context (matrix_multiplication)
// sync inputs only once and re-launch; they don't exercise the
// re-sync + re-launch pattern that our trace sweep needs.
//
// Until the underlying XRT/driver behavior is understood we build a
// fresh hw_context + kernel on each call. This still skips the far
// more expensive per-process startup (device open, xclbin parse,
// classifier dlopen) and measures ~90ms/run on Phoenix vs ~228ms for
// a fresh process per run -- a 2.5x speedup. Set
// BRIDGE_RUNNER_REUSE_CONTEXT=1 to experiment with the unsafe-but-
// faster reuse path (not recommended; will hit the timeout pattern).
bool reuse_context_across_runs() {
    if (const char* v = std::getenv("BRIDGE_RUNNER_REUSE_CONTEXT")) {
        return std::atoi(v) != 0;
    }
    return false;
}

// Async context pipeline: pre-builds the next hw_context + kernel in a
// background thread and async-destroys used ones in a second thread, so
// the main thread doesn't pay the ~40 ms build + ~45 ms destroy cost
// inline. The runner becomes bottlenecked on combined builder-plus-
// destroyer throughput instead of their sum.
//
// Measured speedups are smaller than the math suggests, because the
// driver's aie2_lock mutex (grabbed by both aie2_hwctx_start and
// aie2_hwctx_stop in xdna-driver) serializes builder and destroyer:
// they can't actually run in parallel inside the kernel driver. What
// we do get is the main thread's kernel submission (which uses a
// per-context mailbox and doesn't grab aie2_lock) overlapping with
// whichever of build/destroy is running.
//
// On Phoenix + trace-sweep, this measures ~8% sweep wall-clock
// improvement (33.5 s -> 30.9 s averaged over 3 runs on
// add_one_objFifo/chess). Runner-level per-iter drops from ~125 ms
// sync to ~85 ms async, but parse-trace.py dominates at ~900 ms/batch
// so most of the runner savings are hidden in the total.
//
// Disable with BRIDGE_RUNNER_ASYNC_CTX=0 to fall back to the
// synchronous teardown path (useful for debugging or isolating
// regressions).
bool async_ctx_enabled() {
    if (const char* v = std::getenv("BRIDGE_RUNNER_ASYNC_CTX")) {
        return std::atoi(v) != 0;
    }
    return true;
}

// How many ready hw_contexts to keep pre-built. Default 1 is enough for
// the typical sweep where builder throughput (~40 ms/ctx) dominates
// main-thread consumption (~2 ms kernel + BO alloc). Higher values help
// if the main thread briefly runs faster than builder steady-state.
// Capped at 4 so we never approach the Phoenix per-process hw_context
// limit (empirically ~6) even with a destroyer queue also in flight.
size_t async_ctx_depth() {
    if (const char* v = std::getenv("BRIDGE_RUNNER_ASYNC_DEPTH")) {
        int n = std::atoi(v);
        if (n < 1) n = 1;
        if (n > 4) n = 4;
        return static_cast<size_t>(n);
    }
    return 1;
}

struct CtxBundle {
    xrt::hw_context ctx;
    xrt::kernel kernel;
};

// Per-(xclbin,kernel) pipeline. Owns a builder thread that produces
// ready CtxBundles and a destroyer thread that consumes retired ones.
// Thread-safe for a single consumer (the main thread) -- take_ready()
// and retire() are the only main-thread entry points.
class CtxPipeline {
public:
    CtxPipeline(xrt::device& device, xrt::xclbin xclbin,
                std::string kernel_name, bool verbose)
        : device_(device), xclbin_(std::move(xclbin)),
          kernel_name_(std::move(kernel_name)), verbose_(verbose),
          want_depth_(async_ctx_depth())
    {
        builder_ = std::thread([this]{ builder_loop(); });
        destroyer_ = std::thread([this]{ destroyer_loop(); });
    }

    ~CtxPipeline() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            shutdown_ = true;
        }
        cv_need_build_.notify_all();
        cv_need_destroy_.notify_all();
        cv_ready_.notify_all();
        if (builder_.joinable()) builder_.join();
        if (destroyer_.joinable()) destroyer_.join();
    }

    // Blocks until a ready bundle is available. Returns nullptr only on
    // shutdown. A builder may occasionally fail (network hiccup, NPU
    // transient), in which case it retries; take_ready() just waits.
    std::unique_ptr<CtxBundle> take_ready() {
        auto t0 = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lk(mu_);
        cv_ready_.wait(lk, [this]{ return !ready_.empty() || shutdown_; });
        if (ready_.empty()) return nullptr;
        auto b = std::move(ready_.front());
        ready_.pop_front();
        size_t gv = graveyard_.size();
        size_t rd = ready_.size();
        // Wake the builder to restock the queue. We notify under the
        // lock so the builder sees the updated ready_.size() when it
        // rechecks its predicate.
        cv_need_build_.notify_one();
        auto t1 = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        if (verbose_) {
            std::fprintf(stderr,
                "[pipeline] take_ready wait=%lldms (ready=%zu, graveyard=%zu)\n",
                (long long)ms, rd, gv);
        }
        return b;
    }

    // Hand a used bundle to the destroyer. Main thread returns without
    // waiting; the destroyer thread will tear it down in the
    // background.
    void retire(std::unique_ptr<CtxBundle> bundle) {
        if (!bundle) return;
        size_t gv;
        {
            std::lock_guard<std::mutex> lk(mu_);
            graveyard_.push_back(std::move(bundle));
            gv = graveyard_.size();
        }
        cv_need_destroy_.notify_one();
        if (verbose_) {
            std::fprintf(stderr,
                "[pipeline] retired (graveyard now %zu)\n", gv);
        }
    }

private:
    void builder_loop() {
        while (true) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_need_build_.wait(lk, [this]{
                    return shutdown_ || ready_.size() < want_depth_;
                });
                if (shutdown_) return;
            }
            // Build outside the lock -- hw_context construction can
            // take ~40 ms, we don't want to hold mu_ that long.
            std::unique_ptr<CtxBundle> b;
            try {
                b = std::make_unique<CtxBundle>();
                b->ctx = xrt::hw_context(device_, xclbin_.get_uuid());
                b->kernel = xrt::kernel(b->ctx, kernel_name_);
            } catch (const std::exception& e) {
                if (verbose_) {
                    std::fprintf(stderr,
                        "[ctx-builder] build failed: %s (retry)\n", e.what());
                }
                // Back off briefly and retry on the next loop iteration.
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (shutdown_) return;
                ready_.push_back(std::move(b));
            }
            cv_ready_.notify_one();
        }
    }

    void destroyer_loop() {
        while (true) {
            std::unique_ptr<CtxBundle> b;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_need_destroy_.wait(lk, [this]{
                    return shutdown_ || !graveyard_.empty();
                });
                if (shutdown_ && graveyard_.empty()) return;
                if (graveyard_.empty()) continue;
                b = std::move(graveyard_.front());
                graveyard_.pop_front();
            }
            // Explicit reset outside the lock -- kernel first (holds a
            // reference to ctx), then ctx.
            try {
                b->kernel = xrt::kernel{};
                b->ctx = xrt::hw_context{};
            } catch (const std::exception& e) {
                if (verbose_) {
                    std::fprintf(stderr,
                        "[ctx-destroyer] destroy threw: %s (continuing)\n",
                        e.what());
                }
            }
            // b goes out of scope here.
        }
    }

    xrt::device& device_;
    xrt::xclbin xclbin_;
    std::string kernel_name_;
    bool verbose_;
    size_t want_depth_;

    std::mutex mu_;
    std::condition_variable cv_ready_;
    std::condition_variable cv_need_build_;
    std::condition_variable cv_need_destroy_;
    std::deque<std::unique_ptr<CtxBundle>> ready_;
    std::deque<std::unique_ptr<CtxBundle>> graveyard_;
    bool shutdown_ = false;

    std::thread builder_;
    std::thread destroyer_;
};

struct PreparedKernel {
    std::string xclbin_path;
    std::string kernel_hint;   // empty means "any"
    xrt::xclbin xclbin;
    std::string kernel_name;
    xrt::hw_context context;
    xrt::kernel kernel;
    std::vector<KernArgInfo> kargs;
    KernArgPlan plan;
    // Cached trace buffer override applied at prepare time (-1 = none).
    // Used by get_or_prepare to detect callers passing inconsistent
    // overrides for the same xclbin.
    int trace_buf_idx = -1;

    // BO pool held across runs. Reallocating BOs between rapid-fire
    // runs against the same hw_context causes the scheduler to
    // intermittently time out (observed: every other run fails with
    // state=8 TIMEOUT on add_one_using_dma / add_one_objFifo). Pooling
    // BOs so the kernel sees a stable set of backing buffers -- and
    // only rewriting their *contents* between runs -- matches how the
    // reference test.cpp uses XRT (one alloc + many kernel invocations)
    // and eliminates the failures.
    //
    // The pool is lazily initialized on the first execute_run call.
    // Subsequent calls verify sizes still match; a size mismatch
    // triggers a full pool rebuild for that slot (rare in practice --
    // trace sweeps keep BO sizes constant).
    bool pool_ready = false;
    std::vector<xrt::bo> pool_bos;                 // index-parallel to pool_arg_to_bo
    std::vector<size_t> pool_arg_to_bo_idx;        // kargs.size(); SIZE_MAX for scalars
    std::vector<size_t> pool_bo_sizes;             // parallel to pool_bos
    size_t pool_trace_bo_idx = SIZE_MAX;
    // Last-run binding snapshot so we can detect when the --input/
    // --ctrlpkt mapping changes and need to reseed contents.
    std::vector<std::string> pool_middle_binding_tags;
    std::vector<std::string> pool_middle_binding_paths;
    uint64_t pool_trace_size = 0;
    // xrt::run is NOT pooled across calls -- reusing it breaks after a
    // timeout and is not the XRT-documented pattern. Each execute_run
    // creates a fresh run, sets args, starts, waits. The BOs it
    // references come from this pool so no BO churn happens per run.

    // Async pipeline: when enabled, each run takes a fresh
    // hw_context+kernel from the builder thread and retires the used
    // one to the destroyer thread. When this pipeline is active,
    // pool_ready is always false -- the BO pool is invalidated every
    // run because BOs are bound to the retiring kernel's group_ids.
    // Constructed lazily on the first async run.
    std::unique_ptr<CtxPipeline> pipeline;
};

std::unique_ptr<PreparedKernel> prepare_kernel(
    xrt::device& device,
    const std::string& xclbin_path,
    const std::string& kernel_hint,
    int trace_buf_idx,
    bool verbose)
{
    auto p = std::make_unique<PreparedKernel>();
    p->xclbin_path = xclbin_path;
    p->kernel_hint = kernel_hint;
    p->xclbin = xrt::xclbin(xclbin_path);
    p->kargs = read_kernel_args(p->xclbin, kernel_hint, p->kernel_name, verbose);
    std::fprintf(stderr, "bridge-trace-runner: kernel=%s, %zu args\n",
                 p->kernel_name.c_str(), p->kargs.size());
    p->plan = classify_kernargs(p->kargs, trace_buf_idx);
    p->trace_buf_idx = trace_buf_idx;
    if (verbose) {
        std::fprintf(stderr,
            "  plan: opcode=%zu instr=%zu ninstr=%zu middle=%zu trace=%zu"
            " (trace_buf_idx=%d)\n",
            p->plan.opcode_idx, p->plan.instr_idx, p->plan.instr_size_idx,
            p->plan.middle_buf_indices.size(), p->plan.trace_idx,
            trace_buf_idx);
    }
    device.register_xclbin(p->xclbin);
    p->context = xrt::hw_context(device, p->xclbin.get_uuid());
    p->kernel = xrt::kernel(p->context, p->kernel_name);
    return p;
}

// --------------------------------------------------------------------------
// Per-run execution.
// --------------------------------------------------------------------------

struct RunOutcome {
    bool ok = false;
    std::string error;
    uint64_t elapsed_ms = 0;
    uint64_t elapsed_us = 0;
    uint64_t kernel_us = 0;  // run.start() -> run.wait() returns
    // Tile (0,2) Performance_Counter0 reading after kernel completes.
    // Set when --read-perf-counter is enabled. Counter is configured before
    // run.start() to count ACTIVE_CORE cycles, then read after run.wait().
    // Independent of trace controller -- crosscheck for trace artifacts.
    uint32_t core_cycles = 0;
    bool perf_ok = false;
};

// Escape a string for single-line JSON. Handles the minimum set of chars
// that could appear in a stderr error message; we never embed arbitrary
// JSON-hostile data in these strings.
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c & 0xff);
                    out += buf;
                } else {
                    out.push_back(c);
                }
        }
    }
    return out;
}

// Helper: resolve the binding for each middle-buffer slot given the
// per-run ctrlpkt/input/output lists and the classifier output. Returns
// a parallel vector with one entry per middle slot.
enum class Bind { Zero, InputFile, OutputSlot, CtrlpktFile };
struct BindInfo { Bind kind; std::string path; size_t output_slot_index = 0; };

// Map a middle-buffer kernarg slot to the address_patch arg_idx the
// firmware uses for it. Slots 0/1/2 are opcode/instr/ninstr (handled by
// XRT), so kernarg 3 is BO arg_idx 0.
inline size_t kernarg_to_arg_idx(size_t kernarg) { return kernarg - 3; }

std::vector<BindInfo> resolve_bindings(
    const KernArgPlan& plan,
    const std::vector<ClassifierRole>& roles,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& ctrlpkts)
{
    std::vector<BindInfo> data_arg_binding(plan.middle_buf_indices.size(),
                                           BindInfo{Bind::Zero, {}, 0});

    // Build arg_idx -> role lookup. Only arg_idx values that appear in
    // the instruction stream's DdrPatch ops show up here; unused slots
    // (declared in the kernel template but never patched) fall through
    // to a Bind::Zero placeholder, which is correct -- the kernel
    // doesn't read them, so any binding works.
    std::unordered_map<size_t, uint8_t> role_by_arg_idx;
    for (const auto& r : roles) {
        role_by_arg_idx[r.arg_idx] = r.role;
    }

    // Classifier-aware path: assign --input/--output/--ctrlpkt files to
    // the slots whose role matches, in order. This replaces the legacy
    // round-robin which silently mis-bound --output for any kernel
    // signature where the s2mm slot wasn't immediately after the mm2s
    // slot (e.g., runtime_loop's `(mm2s, unused, s2mm)` shape).
    if (!roles.empty()) {
        size_t in_i = 0, out_i = 0, ctrl_i = 0;
        for (size_t slot = 0; slot < plan.middle_buf_indices.size(); ++slot) {
            size_t arg_idx = kernarg_to_arg_idx(plan.middle_buf_indices[slot]);
            auto it = role_by_arg_idx.find(arg_idx);
            if (it == role_by_arg_idx.end()) continue;  // unused -> Zero
            switch (it->second) {
            case ROLE_DATA_MM2S:
                if (in_i < inputs.size()) {
                    data_arg_binding[slot] = {Bind::InputFile, inputs[in_i++], 0};
                }
                break;
            case ROLE_DATA_S2MM:
                if (out_i < outputs.size()) {
                    data_arg_binding[slot] = {Bind::OutputSlot,
                                              outputs[out_i], out_i};
                    ++out_i;
                }
                break;
            case ROLE_CTRLPKT:
                if (ctrl_i < ctrlpkts.size()) {
                    data_arg_binding[slot] = {Bind::CtrlpktFile,
                                              ctrlpkts[ctrl_i++], 0};
                }
                break;
            default:
                break;  // ROLE_UNKNOWN -> Zero
            }
        }
        if (in_i < inputs.size()) {
            throw std::runtime_error(
                "more --input paths (" + std::to_string(inputs.size()) +
                ") than mm2s middle slots (" + std::to_string(in_i) + ")");
        }
        if (out_i < outputs.size()) {
            throw std::runtime_error(
                "more --output paths (" + std::to_string(outputs.size()) +
                ") than s2mm middle slots (" + std::to_string(out_i) + ")");
        }
        if (ctrl_i < ctrlpkts.size()) {
            throw std::runtime_error(
                "more --ctrlpkt paths (" + std::to_string(ctrlpkts.size()) +
                ") than ctrlpkt middle slots (" + std::to_string(ctrl_i) + ")");
        }
        return data_arg_binding;
    }

    // Fallback: classifier unavailable (no FFI library, or empty roles
    // result). Use the legacy round-robin: reserve any classifier-flagged
    // ctrlpkt slots first (none in this branch since roles is empty),
    // then assign inputs and outputs to remaining slots in order. This
    // preserves the pre-classifier behavior for setups without
    // libxdna_emu.so.
    size_t input_i = 0, output_i = 0;
    for (size_t slot = 0; slot < data_arg_binding.size(); ++slot) {
        if (input_i < inputs.size()) {
            data_arg_binding[slot] = {Bind::InputFile, inputs[input_i++], 0};
        } else if (output_i < outputs.size()) {
            data_arg_binding[slot] = {Bind::OutputSlot, outputs[output_i], output_i};
            ++output_i;
        }
    }
    return data_arg_binding;
}

// Stable string tag for a binding, used to detect when the per-slot
// role changed between runs (invalidates pooled BO content).
std::string binding_tag(const BindInfo& b) {
    switch (b.kind) {
        case Bind::Zero:        return "zero";
        case Bind::InputFile:   return "input:" + b.path;
        case Bind::OutputSlot:  return "output";
        case Bind::CtrlpktFile: return "ctrlpkt:" + b.path;
    }
    return "?";
}

// Determine the required BO size for a middle slot. File-backed slots
// must be sized >= their file (pointer kernargs declare size=8 for the
// pointer itself, so we take the file size when it's larger).
size_t middle_slot_size(const KernArgInfo& karg, const BindInfo& bind) {
    size_t declared = static_cast<size_t>(karg.size);
    if (bind.kind == Bind::InputFile || bind.kind == Bind::CtrlpktFile) {
        std::ifstream fs(bind.path, std::ios::binary | std::ios::ate);
        if (!fs) throw std::runtime_error("cannot open input file: " + bind.path);
        std::streamsize p = fs.tellg();
        if (p < 0) throw std::runtime_error("cannot stat input file: " + bind.path);
        return std::max(declared, static_cast<size_t>(p));
    }
    return declared;
}

RunOutcome execute_run(
    xrt::device& device,
    PreparedKernel& prep,
    classify_fn_t classify_fn,
    const RunArgs& args,
    bool verbose)
{
    RunOutcome result;
    auto t0 = std::chrono::steady_clock::now();
    // Hoisted above the try so the retirement block at the end can see
    // it regardless of where an exception was thrown.
    std::unique_ptr<CtxBundle> active_bundle;
    try {
        const auto& kargs = prep.kargs;
        const auto& plan = prep.plan;

        size_t expected_middle = args.inputs.size() + args.outputs.size();
        if (expected_middle > plan.middle_buf_indices.size()) {
            throw std::runtime_error(
                "too many --input/--output args: got " +
                std::to_string(expected_middle) + ", xclbin has " +
                std::to_string(plan.middle_buf_indices.size()) + " middle buffers");
        }

        // Load instr + classify kernargs to identify ctrlpkt slots.
        auto instr_words = load_instr_binary(args.instr_bin);
        if (verbose) {
            std::fprintf(stderr, "  loaded %zu instruction words from %s\n",
                         instr_words.size(), args.instr_bin.c_str());
        }
        // Inject the userspace preamble that re-establishes hw_context
        // setup state before the kernel runs. Two sources, additive:
        //
        //   --cdo-preamble <path>  : transcode a full CDO blob into
        //     insts.bin ops. This is the comprehensive path (replays
        //     core reset, lock zero+init, DMA BDs, stream-switch
        //     routing, trace setup) and is what we rely on for
        //     hwctx-reuse trace stability.
        //
        //   --reset-tile / --reset-lock : the older minimal path that
        //     only resets core+locks. Kept for the cases where a CDO
        //     blob isn't available or we want a hand-tuned preamble.
        //
        // Harmless (just extra register writes) when reuse_context is
        // off, but necessary when on -- without it, every-other-launch
        // hangs on drifted lock state and the trace unit state drifts
        // between launches.
        std::vector<uint32_t> combined_preamble;
        for (const auto& p : args.cdo_preambles) {
            auto blob = load_cdo_blob(p);
            auto words = transcode_cdo_to_insts(blob, p);
            if (verbose) {
                std::fprintf(stderr,
                    "  cdo preamble %s: %zu bytes -> %zu insts.bin words\n",
                    p.c_str(), blob.size(), words.size());
            }
            combined_preamble.insert(combined_preamble.end(),
                                     words.begin(), words.end());
        }
        if (!args.reset_tiles.empty() || !args.lock_inits.empty()) {
            auto words = build_reset_preamble(args.reset_tiles,
                                              args.lock_inits);
            if (verbose) {
                std::fprintf(stderr,
                    "  manual reset preamble: %zu tile(s), %zu lock-init(s)"
                    " -> %zu words\n",
                    args.reset_tiles.size(), args.lock_inits.size(),
                    words.size());
            }
            combined_preamble.insert(combined_preamble.end(),
                                     words.begin(), words.end());
        }
        if (!combined_preamble.empty()) {
            splice_reset_preamble(instr_words, combined_preamble);
            if (verbose) {
                std::fprintf(stderr,
                    "  preamble total: +%zu words spliced\n",
                    combined_preamble.size());
            }
        }
        auto roles = classify_kernargs_via_ffi(
            classify_fn, args.xclbin, instr_words, verbose);
        if (verbose && !roles.empty()) {
            std::fprintf(stderr, "  classifier roles:");
            for (const auto& r : roles) {
                std::fprintf(stderr, " arg%u=%s", r.arg_idx, role_name(r.role));
            }
            std::fprintf(stderr, "\n");
        }

        // When the classifier is unavailable (no FFI lib loaded or empty
        // roles), --ctrlpkt has no slot to attach to; degrade by treating
        // those paths as additional --input args (which is what the
        // pre-classifier runner did). resolve_bindings's empty-roles
        // branch then round-robins them across middle slots.
        std::vector<std::string> inputs = args.inputs;
        std::vector<std::string> ctrlpkts = args.ctrlpkts;
        if (classify_fn == nullptr || roles.empty()) {
            if (!ctrlpkts.empty() && verbose) {
                std::fprintf(stderr,
                    "  classifier unavailable; --ctrlpkt falls back to --input\n");
            }
            inputs.insert(inputs.begin(), ctrlpkts.begin(), ctrlpkts.end());
            ctrlpkts.clear();
        }

        auto data_arg_binding = resolve_bindings(
            plan, roles, inputs, args.outputs, ctrlpkts);

        // Rebuild hw_context + kernel per run to sidestep the
        // alternating-TIMEOUT pattern that hits on hw_context reuse.
        // Two paths: async pipeline (default) hands us a pre-built
        // bundle and retires the used one to a destroyer thread so
        // both costs overlap kernel execution; synchronous teardown
        // (legacy, for comparison) does it all inline.
        if (!reuse_context_across_runs()) {
            if (async_ctx_enabled()) {
                if (!prep.pipeline) {
                    prep.pipeline = std::make_unique<CtxPipeline>(
                        device, prep.xclbin, prep.kernel_name, verbose);
                }
                active_bundle = prep.pipeline->take_ready();
                if (!active_bundle) {
                    throw std::runtime_error(
                        "ctx pipeline shut down before producing a bundle");
                }
                // Swap the pre-built ctx+kernel into prep so the rest
                // of execute_run operates on a fresh, valid pair.
                prep.context = active_bundle->ctx;
                prep.kernel = active_bundle->kernel;
            } else {
                prep.kernel = xrt::kernel{};
                prep.context = xrt::hw_context{};
                prep.context = xrt::hw_context(device, prep.xclbin.get_uuid());
                prep.kernel = xrt::kernel(prep.context, prep.kernel_name);
            }
            prep.pool_ready = false;
        }

        // --- Pooled BO allocation ---
        //
        // First call or a BO-size mismatch triggers a fresh pool. When
        // the pool already matches current requirements, we reuse the
        // BOs directly -- only their *contents* are rewritten per run.
        // This matches the reference test.cpp's "one alloc, many runs"
        // pattern and avoids the alternating-run failures seen when
        // BOs were reallocated between launches in the same hw_context.
        size_t instr_bytes = instr_words.size() * sizeof(uint32_t);
        bool need_full_rebuild = !prep.pool_ready;
        std::vector<size_t> middle_sizes(plan.middle_buf_indices.size());
        for (size_t slot = 0; slot < plan.middle_buf_indices.size(); ++slot) {
            size_t karg_idx = plan.middle_buf_indices[slot];
            middle_sizes[slot] = middle_slot_size(kargs[karg_idx],
                                                  data_arg_binding[slot]);
        }

        // Size-mismatch check. We rebuild the whole pool rather than
        // per-slot reallocation so the xrt::run re-setarg is a clean
        // single pass.
        if (!need_full_rebuild) {
            if (prep.pool_trace_size != args.trace_size_bytes) need_full_rebuild = true;
        }
        if (!need_full_rebuild) {
            // Check each BO's current size matches what we'd want now.
            // instr BO:
            size_t instr_bo_idx = prep.pool_arg_to_bo_idx[plan.instr_idx];
            if (prep.pool_bo_sizes[instr_bo_idx] != instr_bytes) need_full_rebuild = true;
            // middle BOs:
            for (size_t slot = 0; !need_full_rebuild && slot < plan.middle_buf_indices.size(); ++slot) {
                size_t karg_idx = plan.middle_buf_indices[slot];
                size_t idx = prep.pool_arg_to_bo_idx[karg_idx];
                if (prep.pool_bo_sizes[idx] != middle_sizes[slot]) need_full_rebuild = true;
            }
        }

        if (need_full_rebuild) {
            if (verbose) {
                std::fprintf(stderr,
                    "bridge-trace-runner: %s BO pool\n",
                    prep.pool_ready ? "rebuilding" : "allocating");
            }
            prep.pool_bos.clear();
            prep.pool_bo_sizes.clear();
            prep.pool_arg_to_bo_idx.assign(kargs.size(), SIZE_MAX);
            prep.pool_bos.reserve(plan.middle_buf_indices.size() + 2);
            prep.pool_bo_sizes.reserve(plan.middle_buf_indices.size() + 2);

            auto make_buffer = [&](size_t karg_idx, size_t sz,
                                   xrt::bo::flags flags) -> xrt::bo& {
                prep.pool_bos.emplace_back(device, sz, flags,
                                           prep.kernel.group_id(static_cast<int>(karg_idx)));
                prep.pool_bo_sizes.push_back(sz);
                prep.pool_arg_to_bo_idx[karg_idx] = prep.pool_bos.size() - 1;
                return prep.pool_bos.back();
            };

            make_buffer(plan.instr_idx, instr_bytes, xrt::bo::flags::cacheable);
            for (size_t slot = 0; slot < plan.middle_buf_indices.size(); ++slot) {
                size_t karg_idx = plan.middle_buf_indices[slot];
                make_buffer(karg_idx, middle_sizes[slot], xrt::bo::flags::host_only);
            }
            make_buffer(plan.trace_idx, args.trace_size_bytes, xrt::bo::flags::host_only);
            prep.pool_trace_bo_idx = prep.pool_arg_to_bo_idx[plan.trace_idx];
            prep.pool_trace_size = args.trace_size_bytes;

            prep.pool_middle_binding_tags.assign(plan.middle_buf_indices.size(), {});
            prep.pool_middle_binding_paths.assign(plan.middle_buf_indices.size(), {});
            prep.pool_ready = true;
        } else if (verbose) {
            std::fprintf(stderr, "bridge-trace-runner: reusing pooled BOs\n");
        }

        // Fill instr BO with current instr contents.
        {
            xrt::bo& bo = prep.pool_bos[prep.pool_arg_to_bo_idx[plan.instr_idx]];
            std::memcpy(bo.map<void*>(), instr_words.data(), instr_bytes);
            bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        // Fill middle BOs per-binding. Re-seed content every run (we
        // can't tell without reading the file whether the prior run
        // left stale bytes in a slot we'll now treat differently, and
        // zero + optional file-load is cheap for the sizes we deal in).
        for (size_t slot = 0; slot < plan.middle_buf_indices.size(); ++slot) {
            size_t karg_idx = plan.middle_buf_indices[slot];
            xrt::bo& bo = prep.pool_bos[prep.pool_arg_to_bo_idx[karg_idx]];
            const auto& bind = data_arg_binding[slot];
            std::memset(bo.map<void*>(), 0, middle_sizes[slot]);
            if (bind.kind == Bind::InputFile || bind.kind == Bind::CtrlpktFile) {
                load_bo_from_file(bo, bind.path, middle_sizes[slot]);
            }
            bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            prep.pool_middle_binding_tags[slot] = binding_tag(bind);
            prep.pool_middle_binding_paths[slot] = bind.path;
        }

        // Trace BO: zero every launch (the trace unit writes a prefix;
        // stale bytes beyond the prefix would corrupt later parses).
        //
        // Note on cumulative offsets across batches: the trace shim DMA
        // BD (configured once per run by insts.bin via npu_writebd) is
        // valid but never completes -- the trace stream is open-ended
        // and there is no length check that fires. Re-pushing the same
        // BD on the next run does not reset the channel's internal
        // write-byte counter, so each batch's events land at the
        // cumulative offset where the previous batch's writes stopped.
        // Empty batches (no matching events) leave the pointer untouched,
        // which is why some sweep batches start exactly where the last
        // non-empty batch ended. Data integrity is preserved (the BO
        // here is genuinely zero before each run; new data is at fresh
        // offsets, never overlapping prior runs); only the buffer's
        // *effective capacity* shrinks linearly across the batch-stdin
        // session. With trace_size_bytes=1MB this is fine for sweeps in
        // the low thousands of batches. A future improvement could
        // inject an explicit DMA channel reset between runs.
        {
            xrt::bo& bo = prep.pool_bos[prep.pool_trace_bo_idx];
            std::memset(bo.map<void*>(), 0, args.trace_size_bytes);
            bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        // Fresh xrt::run per call; reusing one across starts is not the
        // documented pattern and a single timeout poisons the run object
        // for all future launches. Args come from the pooled BOs, so
        // construction is still cheap (no BO alloc). We use the
        // kernel() functor (matches the reference test.cpp pattern) --
        // it calls set_arg+start atomically and is what the programming
        // examples use in multi-iteration loops.
        xrt::run run(prep.kernel);
        for (size_t i = 0; i < kargs.size(); ++i) {
            if (i == plan.opcode_idx) {
                run.set_arg(static_cast<int>(i), AIE_KERNEL_OPCODE);
            } else if (i == plan.instr_size_idx) {
                run.set_arg(static_cast<int>(i),
                            static_cast<uint32_t>(instr_words.size()));
            } else if (prep.pool_arg_to_bo_idx[i] != SIZE_MAX) {
                run.set_arg(static_cast<int>(i),
                            prep.pool_bos[prep.pool_arg_to_bo_idx[i]]);
            } else {
                throw std::runtime_error(
                    "kernarg " + std::to_string(i) +
                    " was not classified (no scalar role and no BO bound)");
            }
        }
        if (verbose) {
            std::fprintf(stderr, "  launching kernel (timeout 30s)\n");
        }
        // Configure perf counter on (perf_col, perf_row) to count ACTIVE_CORE
        // cycles. Independent of the trace controller -- diagnostic for
        // trace-induced artifacts. Stop event = 0 means counter increments
        // on every cycle the start event has fired.
        //
        // Lifecycle gotcha (#359 / 2026-05-05-aie-rw-access-firmware-actually-supported.md):
        // hwctx->num_col is set inside the runqueue connect path, which fires
        // off a kernel workqueue (aie2_ctx_runqueue.c::part_ctx_start) only
        // after a kernel command has been queued. Until that workqueue runs,
        // num_col == 0 and the partition-range check at aie2_pci.c rejects
        // any read_aie_reg / write_aie_reg with EINVAL. So we cannot do this
        // setup before run.start(). Instead we run.start() first, then poll
        // read_aie_reg until it succeeds (signalling that the connect
        // workqueue has run and the partition is allocated), then perform
        // the writes. Race: the AIE core may have begun executing by the
        // time setup completes -- in practice the kernel sits in pipeline
        // fill for thousands of cycles before any ACTIVE_CORE cycle elapses,
        // so the missed-cycles window is bounded and small relative to the
        // measurement.
        // Performance counter register block layout (per aie-rt
        // xaiemlgbl_params.h). We use *counter 1*: counter 2 is the trace
        // anchor (set by mlir-trace-inject's runtime sequence), counter 0
        // is left for other userspace consumers; counter 1 is free.
        constexpr uint32_t PERF_CTRL0_OFFSET    = 0x00031500;
        constexpr uint32_t PERF_CTRL2_OFFSET    = 0x00031508;
        constexpr uint32_t PERF_COUNTER1_OFFSET = 0x00031524;
        constexpr uint8_t  EVENT_ACTIVE_CORE    = 0x1C;
        if (args.read_perf_counter) {
            // Force eager partition allocation BEFORE run.start(). Without
            // this, hwctx->num_col stays 0 until the runqueue dispatcher
            // fires part_ctx_start (lazy alloc), so any AIE register access
            // fails with EINVAL during the gap between hwctx_create and
            // first cmd_submit. The new DRM_AMDXDNA_FORCE_CONNECT_HWCTX
            // ioctl (in xdna-driver, exposed via xrt::hw_context::
            // force_connect) drives that connect synchronously.
            try {
                prep.context.force_connect(1000);  // 1s timeout
            } catch (const std::exception& e) {
                if (verbose) {
                    std::fprintf(stderr,
                        "  force_connect failed: %s (perf counter setup "
                        "will be skipped)\n", e.what());
                }
            }

            // Configure counter 1 pre-launch. Reads + writes succeed once
            // force_connect has set num_col. Read-modify-write so we
            // preserve cnt0/cnt2/cnt3 fields that other tooling owns.
            try {
                uint32_t ctrl0 = prep.context.read_aie_reg(
                    args.perf_col, args.perf_row, PERF_CTRL0_OFFSET);
                uint32_t ctrl2 = prep.context.read_aie_reg(
                    args.perf_col, args.perf_row, PERF_CTRL2_OFFSET);
                // PERF_CTRL0[22:16] = cnt1 start_event = ACTIVE_CORE
                // PERF_CTRL0[30:24] = cnt1 stop_event  = NONE
                ctrl0 &= ~uint32_t(0x7F7F0000);
                ctrl0 |= uint32_t(EVENT_ACTIVE_CORE) << 16;
                prep.context.write_aie_reg(
                    args.perf_col, args.perf_row, PERF_CTRL0_OFFSET, ctrl0);
                // PERF_CTRL2[14:8] = cnt1 reset_event = NONE
                ctrl2 &= ~uint32_t(0x7F00);
                prep.context.write_aie_reg(
                    args.perf_col, args.perf_row, PERF_CTRL2_OFFSET, ctrl2);
                // Zero counter 1 to start fresh. Lands BEFORE run.start()
                // so the kernel's full ACTIVE_CORE window counts from 0.
                prep.context.write_aie_reg(
                    args.perf_col, args.perf_row, PERF_COUNTER1_OFFSET, 0);
            } catch (const std::exception& e) {
                if (verbose) {
                    std::fprintf(stderr,
                        "  perf counter setup write failed: %s\n", e.what());
                }
            }
        }
        auto k0 = std::chrono::steady_clock::now();
        run.start();
        auto state = run.wait(std::chrono::seconds(30));
        auto k1 = std::chrono::steady_clock::now();
        result.kernel_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(k1 - k0).count());
        if (args.read_perf_counter) {
            try {
                // Read counter 1 to match the cnt1-based setup above.
                result.core_cycles = prep.context.read_aie_reg(
                    args.perf_col, args.perf_row, PERF_COUNTER1_OFFSET);
                result.perf_ok = true;
            } catch (const std::exception& e) {
                if (verbose) {
                    std::fprintf(stderr,
                        "  perf counter read failed: %s\n", e.what());
                }
            }
        }
        if (verbose) {
            std::fprintf(stderr, "  run state after wait: %d\n", (int)state);
        }
        if (state != ERT_CMD_STATE_COMPLETED) {
            throw std::runtime_error(
                "kernel did not complete (state=" +
                std::to_string(static_cast<int>(state)) + ")");
        }

        // Trace buffer: sync from device then write raw bytes.
        xrt::bo& trace_bo = prep.pool_bos[prep.pool_trace_bo_idx];
        trace_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        {
            std::ofstream out(args.trace_out, std::ios::binary);
            if (!out) {
                throw std::runtime_error("cannot open trace-out: " + args.trace_out);
            }
            out.write(static_cast<const char*>(trace_bo.map<const void*>()),
                      static_cast<std::streamsize>(args.trace_size_bytes));
        }
        if (verbose) {
            std::fprintf(stderr, "  wrote %lu bytes of trace to %s\n",
                         (unsigned long)args.trace_size_bytes, args.trace_out.c_str());
        }

        // Write any --output sinks back to their host paths.
        if (!args.outputs.empty()) {
            for (size_t slot = 0; slot < plan.middle_buf_indices.size(); ++slot) {
                if (data_arg_binding[slot].kind != Bind::OutputSlot) continue;
                size_t karg_idx = plan.middle_buf_indices[slot];
                xrt::bo& bo = prep.pool_bos[prep.pool_arg_to_bo_idx[karg_idx]];
                bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                const auto& out_path = data_arg_binding[slot].path;
                std::ofstream out(out_path, std::ios::binary);
                if (!out) {
                    throw std::runtime_error("cannot open output file: " + out_path);
                }
                out.write(static_cast<const char*>(bo.map<const void*>()),
                          static_cast<std::streamsize>(kargs[karg_idx].size));
            }
        }
        result.ok = true;
    } catch (const std::exception& e) {
        result.ok = false;
        result.error = e.what();
    }
    // Retire the async-acquired bundle regardless of success/failure.
    // Clear prep's copies first so the destroyer thread holds the last
    // reference and can tear down off the main thread's critical path.
    // BO pool is bound to this kernel's group_ids, so it has to go too.
    if (active_bundle) {
        prep.pool_bos.clear();
        prep.pool_bo_sizes.clear();
        prep.pool_arg_to_bo_idx.clear();
        prep.pool_ready = false;
        prep.kernel = xrt::kernel{};
        prep.context = xrt::hw_context{};
        prep.pipeline->retire(std::move(active_bundle));
    }
    auto t1 = std::chrono::steady_clock::now();
    result.elapsed_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
    result.elapsed_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    return result;
}

// --------------------------------------------------------------------------
// Session: manages a cache of PreparedKernel keyed by xclbin path. New
// xclbins trigger prepare_kernel; reused xclbins reuse the cached entry.
// --------------------------------------------------------------------------

struct Session {
    xrt::device device;
    classify_fn_t classify_fn = nullptr;
    bool verbose = false;
    // Cache key is (xclbin_path, kernel_hint). Different kernel hints
    // against the same xclbin select different kernels, so they need
    // separate PreparedKernel entries.
    std::unordered_map<std::string, std::unique_ptr<PreparedKernel>> cache;

    PreparedKernel& get_or_prepare(const std::string& xclbin_path,
                                   const std::string& kernel_hint,
                                   int trace_buf_idx = -1) {
        std::string key = xclbin_path + "::" + kernel_hint;
        auto it = cache.find(key);
        if (it != cache.end()) {
            // Surface inconsistent --trace-buf-idx across runs against
            // the same xclbin -- the cached plan's trace selection
            // would be wrong for the new value, so refuse rather than
            // silently using stale routing.
            if (it->second->trace_buf_idx != trace_buf_idx) {
                throw std::runtime_error(
                    "trace_buf_idx mismatch on cached xclbin "
                    + xclbin_path + ": cached=" +
                    std::to_string(it->second->trace_buf_idx) +
                    ", new=" + std::to_string(trace_buf_idx));
            }
            return *it->second;
        }
        auto prep = prepare_kernel(device, xclbin_path, kernel_hint,
                                   trace_buf_idx, verbose);
        auto& ref = *prep;
        cache.emplace(std::move(key), std::move(prep));
        return ref;
    }
};

int run_single_shot(Session& session, const RunArgs& args) {
    std::fprintf(stderr, "bridge-trace-runner: xclbin=%s instr=%s trace_out=%s\n",
                 args.xclbin.c_str(), args.instr_bin.c_str(), args.trace_out.c_str());
    if (session.verbose) {
        std::fprintf(stderr, "  kernel=%s trace_size=%lu inputs=%zu outputs=%zu\n",
                     args.kernel_name.empty() ? "<auto>" : args.kernel_name.c_str(),
                     (unsigned long)args.trace_size_bytes,
                     args.inputs.size(), args.outputs.size());
    }
    try {
        // If the user didn't pin --trace-buf-idx, scan insts.bin for the
        // max DdrPatch arg_idx (= the trace BO's BO regmap slot) so we
        // bind to the slot the firmware actually patches. The legacy
        // "last buffer = trace" fallback only kicks in if we can't read
        // the insts.bin (e.g., malformed file).
        int trace_buf_idx = args.trace_buf_idx;
        if (trace_buf_idx < 0) {
            int discovered = discover_trace_buf_idx_from_insts(args.instr_bin);
            if (discovered >= 0) {
                if (session.verbose) {
                    std::fprintf(stderr,
                        "  discovered trace_buf_idx=%d from insts.bin\n",
                        discovered);
                }
                trace_buf_idx = discovered;
            }
        }
        auto& prep = session.get_or_prepare(args.xclbin, args.kernel_name,
                                            trace_buf_idx);
        auto out = execute_run(session.device, prep, session.classify_fn,
                               args, session.verbose);
        if (!out.ok) {
            std::fprintf(stderr, "error: %s\n", out.error.c_str());
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}

int run_batch_stdin(Session& session, const OuterArgs& outer, const char* argv0) {
    // Emit a "ready" marker on stdout so the parent can synchronise before
    // sending the first command. This also doubles as a liveness check
    // (parent sees ready = session is up and classifier resolved).
    std::printf("{\"event\":\"ready\",\"pid\":%ld}\n", (long)getpid());
    std::fflush(stdout);

    std::string line;
    uint64_t run_idx = 0;
    while (std::getline(std::cin, line)) {
        // Trim trailing CR (Windows line endings) so the last token parses
        // cleanly on mixed-platform pipes.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        // Ignore blank lines and # comments so the parent can interleave
        // human-readable markers without breaking the protocol.
        if (line.empty() || line[0] == '#') continue;

        RunArgs args;
        // Inherit outer xclbin/kernel defaults.
        OuterArgs outer_copy = outer;
        try {
            auto tokens = tokenize_line(line);
            parse_tokens(tokens, outer_copy, args, /*is_batch_line=*/true, argv0);
        } catch (const std::exception& e) {
            std::printf(
                "{\"run_idx\":%llu,\"ok\":false,\"error\":\"parse: %s\"}\n",
                (unsigned long long)run_idx,
                json_escape(e.what()).c_str());
            std::fflush(stdout);
            ++run_idx;
            continue;
        }

        // Prep or reuse for this xclbin. Auto-discover trace_buf_idx
        // from insts.bin's max DdrPatch arg_idx when the user didn't
        // pin one (matches run_single_shot behavior); the cache key
        // includes the resolved value so different instr files with
        // different trace slots don't collide.
        int trace_buf_idx_resolved = args.trace_buf_idx;
        if (trace_buf_idx_resolved < 0 && !args.instr_bin.empty()) {
            int discovered = discover_trace_buf_idx_from_insts(args.instr_bin);
            if (discovered >= 0) trace_buf_idx_resolved = discovered;
        }
        PreparedKernel* prep = nullptr;
        try {
            prep = &session.get_or_prepare(args.xclbin, args.kernel_name,
                                           trace_buf_idx_resolved);
        } catch (const std::exception& e) {
            std::printf(
                "{\"run_idx\":%llu,\"ok\":false,\"error\":\"prepare: %s\"}\n",
                (unsigned long long)run_idx,
                json_escape(e.what()).c_str());
            std::fflush(stdout);
            ++run_idx;
            continue;
        }

        if (session.verbose) {
            std::fprintf(stderr,
                "bridge-trace-runner[batch]: run %llu xclbin=%s instr=%s trace_out=%s\n",
                (unsigned long long)run_idx,
                args.xclbin.c_str(), args.instr_bin.c_str(), args.trace_out.c_str());
        }

        auto out = execute_run(session.device, *prep, session.classify_fn,
                               args, session.verbose);
        if (out.ok) {
            std::printf(
                "{\"run_idx\":%llu,\"ok\":true,\"trace_out\":\"%s\","
                "\"elapsed_ms\":%llu,\"elapsed_us\":%llu,\"kernel_us\":%llu,"
                "\"perf_ok\":%s,\"core_cycles\":%lu}\n",
                (unsigned long long)run_idx,
                json_escape(args.trace_out).c_str(),
                (unsigned long long)out.elapsed_ms,
                (unsigned long long)out.elapsed_us,
                (unsigned long long)out.kernel_us,
                out.perf_ok ? "true" : "false",
                (unsigned long)out.core_cycles);
        } else {
            std::printf(
                "{\"run_idx\":%llu,\"ok\":false,\"trace_out\":\"%s\","
                "\"error\":\"%s\",\"elapsed_ms\":%llu,\"elapsed_us\":%llu,"
                "\"kernel_us\":%llu,\"perf_ok\":%s,\"core_cycles\":%lu}\n",
                (unsigned long long)run_idx,
                json_escape(args.trace_out).c_str(),
                json_escape(out.error).c_str(),
                (unsigned long long)out.elapsed_ms,
                (unsigned long long)out.elapsed_us,
                (unsigned long long)out.kernel_us,
                out.perf_ok ? "true" : "false",
                (unsigned long)out.core_cycles);
        }
        std::fflush(stdout);
        ++run_idx;
    }
    return 0;
}

} // anonymous namespace

int main(int argc, char** argv) {
    OuterArgs outer;
    RunArgs run;
    if (int rc = parse_cli(argc, argv, outer, run); rc != 0) return rc;

    try {
        Session session{xrt::device(0), try_load_classifier(outer.verbose),
                        outer.verbose, {}};
        if (outer.batch_stdin) {
            return run_batch_stdin(session, outer, argv[0]);
        }
        return run_single_shot(session, run);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
