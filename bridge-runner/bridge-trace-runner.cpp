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
// Output: raw trace buffer contents written to --trace-out, ready to be
// parsed by tools/trace-to-cycles.py.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <stdexcept>
#include <string>
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

struct CliArgs {
    std::string xclbin;
    std::string kernel_name;       // empty = auto-detect single kernel
    std::string instr_bin;         // path to insts.bin
    std::string trace_out;         // where to dump raw trace bytes
    std::vector<std::string> inputs;   // paths to input buffer binaries
    std::vector<std::string> outputs;  // paths where outputs get written
    std::vector<std::string> ctrlpkts; // paths to ctrlpkt blobs
    // Default 1 MiB. Large enough that event-heavy workloads (lock-
    // intensive kernels, full-event sweeps) don't silently truncate
    // mid-trace. The trace unit has no backpressure -- once the BO
    // fills, further events are dropped on the floor and the comparison
    // gets attributed to "emulator didn't emit the event" when the real
    // cause was "buffer ran out." 1 MiB still fits comfortably in host
    // memory and NPU BO allocation limits.
    uint64_t trace_size_bytes = 1u << 20;
    bool verbose = false;
};

void print_usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s --xclbin <path> --instr <insts.bin> "
        "--trace-out <path> [--kernel <name>] [--input <bin>]... "
        "[--output <path>]... [--trace-size N] [-v]\n"
        "\n"
        "Generic XRT runner for trace-instrumented AIE xclbins.\n"
        "Kernel args are discovered from xclbin metadata -- no hardcoded\n"
        "group_id mapping. The trace buffer is identified as the last\n"
        "buffer kernarg after inputs/outputs.\n",
        argv0);
}

int parse_cli(int argc, char** argv, CliArgs& out) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need_val = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s needs a value\n", flag);
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--xclbin")            out.xclbin = need_val("--xclbin");
        else if (a == "--kernel")       out.kernel_name = need_val("--kernel");
        else if (a == "--instr")        out.instr_bin = need_val("--instr");
        else if (a == "--trace-out")    out.trace_out = need_val("--trace-out");
        else if (a == "--input")        out.inputs.push_back(need_val("--input"));
        else if (a == "--output")       out.outputs.push_back(need_val("--output"));
        else if (a == "--ctrlpkt")      out.ctrlpkts.push_back(need_val("--ctrlpkt"));
        else if (a == "--trace-size") {
            // strtoull silently wraps negatives to UINT64_MAX and accepts any
            // leading garbage -- validate explicitly so "-1" or "foo" fail loud.
            const char* val = need_val("--trace-size");
            if (val[0] == '-' || val[0] == '\0') {
                std::fprintf(stderr, "error: --trace-size must be a non-negative integer, got '%s'\n", val);
                return 1;
            }
            char* end = nullptr;
            out.trace_size_bytes = std::strtoull(val, &end, 0);
            if (end == val || *end != '\0') {
                std::fprintf(stderr, "error: --trace-size must be a non-negative integer, got '%s'\n", val);
                return 1;
            }
        }
        else if (a == "-v" || a == "--verbose") out.verbose = true;
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); std::exit(0); }
        else {
            std::fprintf(stderr, "error: unknown arg: %s\n", a.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }
    if (out.xclbin.empty() || out.instr_bin.empty() || out.trace_out.empty()) {
        std::fprintf(stderr, "error: --xclbin, --instr, and --trace-out are required\n");
        print_usage(argv[0]);
        return 1;
    }
    return 0;
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

KernArgPlan classify_kernargs(const std::vector<KernArgInfo>& kargs) {
    KernArgPlan plan;
    std::vector<size_t> scalar_indices;
    std::vector<size_t> buffer_indices;
    for (const auto& k : kargs) {
        if (k.is_scalar()) scalar_indices.push_back(k.index);
        else buffer_indices.push_back(k.index);
    }
    // Expected layout (mlir-aie bridge-test convention):
    //   scalars: opcode (u64), ninstr (u32) -- exactly 2
    //   buffers: instr, [bo0..boK-1], trace -- at least 2
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
    plan.trace_idx      = buffer_indices.back();
    // Middle buffers: everything between the first buffer (instr) and the
    // last buffer (trace).
    for (size_t i = 1; i + 1 < buffer_indices.size(); ++i) {
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

} // anonymous namespace

int main(int argc, char** argv) {
    CliArgs args;
    if (int rc = parse_cli(argc, argv, args); rc != 0) return rc;
    std::fprintf(stderr, "bridge-trace-runner: xclbin=%s instr=%s trace_out=%s\n",
                 args.xclbin.c_str(), args.instr_bin.c_str(), args.trace_out.c_str());
    if (args.verbose) {
        std::fprintf(stderr, "  kernel=%s trace_size=%lu inputs=%zu outputs=%zu\n",
                     args.kernel_name.empty() ? "<auto>" : args.kernel_name.c_str(),
                     (unsigned long)args.trace_size_bytes,
                     args.inputs.size(), args.outputs.size());
    }
    // Load the xclbin and discover its kernel-arg layout.
    try {
        xrt::device device(0);
        xrt::xclbin xclbin(args.xclbin);
        std::string kernel_name;
        auto kargs = read_kernel_args(xclbin, args.kernel_name, kernel_name, args.verbose);
        std::fprintf(stderr, "bridge-trace-runner: kernel=%s, %zu args\n",
                     kernel_name.c_str(), kargs.size());

        // Classify kernargs into positional roles.
        auto plan = classify_kernargs(kargs);
        if (args.verbose) {
            std::fprintf(stderr,
                "  plan: opcode=%zu instr=%zu ninstr=%zu middle=%zu trace=%zu\n",
                plan.opcode_idx, plan.instr_idx, plan.instr_size_idx,
                plan.middle_buf_indices.size(), plan.trace_idx);
        }
        // Sanity-check --input/--output count against available middle buffers.
        size_t expected_middle = args.inputs.size() + args.outputs.size();
        if (expected_middle > plan.middle_buf_indices.size()) {
            throw std::runtime_error(
                "too many --input/--output args: got " +
                std::to_string(expected_middle) + ", xclbin has " +
                std::to_string(plan.middle_buf_indices.size()) + " middle buffers");
        }

        // Register xclbin with device and open a hardware context.
        device.register_xclbin(xclbin);
        xrt::hw_context context(device, xclbin.get_uuid());
        xrt::kernel kernel(context, kernel_name);

        // Load the instruction binary.
        auto instr_words = load_instr_binary(args.instr_bin);
        if (args.verbose) {
            std::fprintf(stderr, "  loaded %zu instruction words from %s\n",
                         instr_words.size(), args.instr_bin.c_str());
        }

        // Best-effort kernarg-role classification via libxdna_emu.so.
        // When the classifier is available and --ctrlpkt paths are supplied,
        // we bind each ctrlpkt path to the classifier-identified Ctrlpkt
        // data-arg (ascending arg_idx order). --input/--output flows
        // positionally through the remaining middle buffers exactly as
        // before, skipping any slot reserved for a ctrlpkt. Without the
        // classifier, --ctrlpkt falls back to being equivalent to --input
        // appended to the front of the input list, preserving legacy
        // positional behavior.
        static classify_fn_t g_classify = try_load_classifier(args.verbose);
        auto roles = classify_kernargs_via_ffi(
            g_classify, args.xclbin, instr_words, args.verbose);
        std::vector<size_t> ctrlpkt_data_args;
        for (const auto& r : roles) {
            if (r.role == ROLE_CTRLPKT) {
                ctrlpkt_data_args.push_back(static_cast<size_t>(r.arg_idx));
            }
        }
        std::sort(ctrlpkt_data_args.begin(), ctrlpkt_data_args.end());
        if (args.verbose && !roles.empty()) {
            std::fprintf(stderr, "  classifier roles:");
            for (const auto& r : roles) {
                std::fprintf(stderr, " arg%u=%s", r.arg_idx, role_name(r.role));
            }
            std::fprintf(stderr, "\n");
        }
        // If no classifier, treat --ctrlpkt as positional --input.
        if (g_classify == nullptr || roles.empty()) {
            if (!args.ctrlpkts.empty() && args.verbose) {
                std::fprintf(stderr,
                    "  classifier unavailable; --ctrlpkt falls back to --input\n");
            }
            args.inputs.insert(args.inputs.begin(),
                               args.ctrlpkts.begin(), args.ctrlpkts.end());
            args.ctrlpkts.clear();
        } else if (args.ctrlpkts.size() > ctrlpkt_data_args.size()) {
            throw std::runtime_error(
                "more --ctrlpkt paths (" + std::to_string(args.ctrlpkts.size()) +
                ") than classifier-identified ctrlpkt args (" +
                std::to_string(ctrlpkt_data_args.size()) + ")");
        }

        // Build a data-arg -> binding-source map:
        //   ctrlpkt path (if reserved for this slot), otherwise --input file
        //   in positional order skipping reserved slots, otherwise --output,
        //   otherwise zero-fill.
        enum class Bind { Zero, InputFile, OutputSlot, CtrlpktFile };
        struct BindInfo { Bind kind; std::string path; };
        std::vector<BindInfo> data_arg_binding(plan.middle_buf_indices.size(),
                                               BindInfo{Bind::Zero, {}});
        std::vector<bool> reserved(plan.middle_buf_indices.size(), false);
        for (size_t i = 0; i < args.ctrlpkts.size(); ++i) {
            size_t slot = ctrlpkt_data_args[i];
            if (slot >= data_arg_binding.size()) {
                throw std::runtime_error(
                    "classifier reports ctrlpkt at arg_idx " +
                    std::to_string(slot) + " but only " +
                    std::to_string(data_arg_binding.size()) + " middle buffers exist");
            }
            data_arg_binding[slot] = {Bind::CtrlpktFile, args.ctrlpkts[i]};
            reserved[slot] = true;
        }
        size_t input_i = 0, output_i = 0;
        for (size_t slot = 0; slot < data_arg_binding.size(); ++slot) {
            if (reserved[slot]) continue;
            if (input_i < args.inputs.size()) {
                data_arg_binding[slot] = {Bind::InputFile, args.inputs[input_i++]};
            } else if (output_i < args.outputs.size()) {
                data_arg_binding[slot] = {Bind::OutputSlot, args.outputs[output_i++]};
            }
        }

        // Per-arg BO storage.  Reserve upfront so push_back never reallocates
        // while we hold interior references.  `arg_to_bo_idx[k]` gives the index
        // in `bos` for kernarg k, or SIZE_MAX if k is a scalar -- Task 10 uses
        // this to call run.set_arg(k, bos[arg_to_bo_idx[k]]) without having to
        // re-derive the kernarg->BO mapping from the plan.
        std::vector<xrt::bo> bos;
        bos.reserve(kargs.size());
        std::vector<size_t> arg_to_bo_idx(kargs.size(), SIZE_MAX);
        xrt::bo* trace_bo_ptr = nullptr;

        // Helper: allocate a BO for a given kernarg index and record it.
        auto make_buffer = [&](size_t karg_idx, size_t sz,
                               xrt::bo::flags flags) -> xrt::bo& {
            bos.emplace_back(device, sz, flags,
                             kernel.group_id(static_cast<int>(karg_idx)));
            arg_to_bo_idx[karg_idx] = bos.size() - 1;
            return bos.back();
        };

        // Instr BO: cacheable (kernel reads only, no host-after-run access).
        {
            auto& bo = make_buffer(plan.instr_idx,
                                   instr_words.size() * sizeof(uint32_t),
                                   xrt::bo::flags::cacheable);
            std::memcpy(bo.map<void*>(),
                        instr_words.data(),
                        instr_words.size() * sizeof(uint32_t));
            bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        // Middle buffers: driven by the data_arg_binding table built above.
        // Each slot is either bound to a file (ctrlpkt or --input), reserved
        // for an --output, or zero-filled.
        for (size_t slot = 0; slot < plan.middle_buf_indices.size(); ++slot) {
            size_t karg_idx = plan.middle_buf_indices[slot];
            size_t karg_declared_sz = static_cast<size_t>(kargs[karg_idx].size);
            const auto& bind = data_arg_binding[slot];
            size_t sz = karg_declared_sz;
            if (bind.kind == Bind::InputFile || bind.kind == Bind::CtrlpktFile) {
                // For pointer kernargs (void*) the declared size is 8 bytes
                // regardless of payload; size the BO from the file's actual
                // bytes so the whole blob reaches the device.
                std::ifstream fs(bind.path, std::ios::binary | std::ios::ate);
                if (!fs) throw std::runtime_error("cannot open input file: " + bind.path);
                std::streamsize p = fs.tellg();
                if (p < 0) throw std::runtime_error("cannot stat input file: " + bind.path);
                sz = std::max(karg_declared_sz, static_cast<size_t>(p));
            }
            auto& bo = make_buffer(karg_idx, sz, xrt::bo::flags::host_only);
            std::memset(bo.map<void*>(), 0, sz);
            if (bind.kind == Bind::InputFile || bind.kind == Bind::CtrlpktFile) {
                load_bo_from_file(bo, bind.path, sz);
            }
            bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        // Trace BO: zero-filled, sized to --trace-size.
        {
            auto& bo = make_buffer(plan.trace_idx, args.trace_size_bytes,
                                   xrt::bo::flags::host_only);
            std::memset(bo.map<void*>(), 0, args.trace_size_bytes);
            bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            trace_bo_ptr = &bo;
        }
        std::fprintf(stderr, "bridge-trace-runner: allocated %zu BOs "
                             "(instr + %zu middle + trace)\n",
                     bos.size(), plan.middle_buf_indices.size());

        // Build the positional arg vector.  For each kernarg, either pass the
        // scalar value we know by role (opcode = 3 as the AIE kernel opcode,
        // ninstr = instr_words.size()) or reference the BO we allocated above
        // via arg_to_bo_idx.
        xrt::run run(kernel);
        for (size_t i = 0; i < kargs.size(); ++i) {
            if (i == plan.opcode_idx) {
                run.set_arg(static_cast<int>(i), AIE_KERNEL_OPCODE);
            } else if (i == plan.instr_size_idx) {
                run.set_arg(static_cast<int>(i),
                            static_cast<uint32_t>(instr_words.size()));
            } else if (arg_to_bo_idx[i] != SIZE_MAX) {
                run.set_arg(static_cast<int>(i), bos[arg_to_bo_idx[i]]);
            } else {
                throw std::runtime_error(
                    "kernarg " + std::to_string(i) +
                    " was not classified (no scalar role and no BO bound)");
            }
        }

        if (args.verbose) {
            std::fprintf(stderr, "  launching kernel (timeout 30s)\n");
        }
        run.start();
        auto state = run.wait(std::chrono::seconds(30));
        if (state != ERT_CMD_STATE_COMPLETED) {
            std::fprintf(stderr, "error: kernel did not complete (state=%d)\n",
                         static_cast<int>(state));
            return 1;
        }

        // Trace buffer: sync from device then write raw bytes to --trace-out.
        trace_bo_ptr->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        {
            std::ofstream out(args.trace_out, std::ios::binary);
            if (!out) {
                throw std::runtime_error("cannot open trace-out: " + args.trace_out);
            }
            out.write(static_cast<const char*>(trace_bo_ptr->map<const void*>()),
                      static_cast<std::streamsize>(args.trace_size_bytes));
        }
        if (args.verbose) {
            std::fprintf(stderr, "  wrote %lu bytes of trace to %s\n",
                         (unsigned long)args.trace_size_bytes, args.trace_out.c_str());
        }

        // Optional: write middle-buffer outputs back to --output paths, in
        // order.  We skipped the first args.inputs.size() middle slots (those
        // are inputs).  The outputs map to the next args.outputs.size() slots.
        if (!args.outputs.empty()) {
            for (size_t o = 0; o < args.outputs.size(); ++o) {
                size_t mb_slot = args.inputs.size() + o;
                if (mb_slot >= plan.middle_buf_indices.size()) break;
                size_t karg_idx = plan.middle_buf_indices[mb_slot];
                // Defensive: middle buffers are always allocated in Task 9's
                // make_buffer(), so arg_to_bo_idx[karg_idx] should never be
                // SIZE_MAX here.  Guard anyway so a future refactor that
                // skips allocation fails loudly instead of indexing bos[~0].
                if (arg_to_bo_idx[karg_idx] == SIZE_MAX) {
                    throw std::runtime_error(
                        "output kernarg " + std::to_string(karg_idx) +
                        " has no BO (classification skipped allocation?)");
                }
                xrt::bo& bo = bos[arg_to_bo_idx[karg_idx]];
                bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                std::ofstream out(args.outputs[o], std::ios::binary);
                if (!out) {
                    throw std::runtime_error("cannot open output file: " + args.outputs[o]);
                }
                out.write(static_cast<const char*>(bo.map<const void*>()),
                          static_cast<std::streamsize>(kargs[karg_idx].size));
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
