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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"

namespace {

struct CliArgs {
    std::string xclbin;
    std::string kernel_name;       // empty = auto-detect single kernel
    std::string instr_bin;         // path to insts.bin
    std::string trace_out;         // where to dump raw trace bytes
    std::vector<std::string> inputs;   // paths to input buffer binaries
    std::vector<std::string> outputs;  // paths where outputs get written
    uint64_t trace_size_bytes = 8192;
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
        // device is kept alive here to verify XRT device opening works end-to-end.
        // Task 9 will use it for BO allocation.
        (void)device;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    // Allocation + execution land in Tasks 9-10.
    return 0;
}
