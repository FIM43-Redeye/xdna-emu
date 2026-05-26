// txn-poll-probe: post-run snapshot of firmware-reported NPU controller state
// for a kernel run. Probes whether `txn_op_idx` / `ctx_pc` (new fields in the
// drivers/accel uapi) are populated after a normal completion, and whether
// they line up with EMU's view of how many control codes the NPU executed.
//
// Mode A only (this file): one ioctl post-`run.wait()`, no polling.
//
// The hwctx must be in scope at query time -- the driver tears the partition
// down on hwctx destruction, after which the entry disappears. We therefore
// snapshot before any RAII teardown.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"

#include <drm/amdxdna_accel.h>

namespace {

constexpr const char* DEFAULT_XCLBIN =
    "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/"
    "_diag_shim_chain_sweep/k8/chess/aie.xclbin";
constexpr const char* DEFAULT_INSTS =
    "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/"
    "_diag_shim_chain_sweep/k8/chess/insts.bin";
constexpr int DEFAULT_K = 8;
constexpr int DEFAULT_N = 64;

struct Args {
    std::string xclbin = DEFAULT_XCLBIN;
    std::string insts = DEFAULT_INSTS;
    int k = DEFAULT_K;
    int n = DEFAULT_N;
    bool verbose = false;
    bool pre_snapshot = false;
};

std::vector<uint32_t> load_instr_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "open insts %s: %s\n", path.c_str(), strerror(errno));
        std::exit(2);
    }
    auto sz = f.tellg();
    f.seekg(0);
    std::vector<uint32_t> data(static_cast<size_t>(sz) / sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(data.data()), sz);
    return data;
}

struct HwctxSnapshot {
    bool ioctl_ok = false;
    int err = 0;
    bool found = false;
    amdxdna_drm_hwctx_entry entry = {};
    uint32_t total_returned = 0;
};

// Driver walks xdna->client_list and returns one entry per (client, hwctx).
// Same-euid clients are visible without CAP_SYS_ADMIN per
// amdxdna_client_visible() in amdxdna_pci_drv.h.
HwctxSnapshot snapshot_hwctx_state() {
    HwctxSnapshot snap;
    int fd = open("/dev/accel/accel0", O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        snap.err = errno;
        return snap;
    }

    constexpr int MAX_ENTRIES = 32;
    std::vector<amdxdna_drm_hwctx_entry> entries(MAX_ENTRIES);
    std::memset(entries.data(), 0, entries.size() * sizeof(entries[0]));

    amdxdna_drm_get_array arg = {};
    arg.param = DRM_AMDXDNA_HW_CONTEXT_ALL;
    arg.element_size = sizeof(amdxdna_drm_hwctx_entry);
    arg.num_element = MAX_ENTRIES;
    arg.buffer = reinterpret_cast<__u64>(entries.data());

    int rc = ioctl(fd, DRM_IOCTL_AMDXDNA_GET_ARRAY, &arg);
    int e = errno;
    close(fd);

    if (rc < 0) {
        snap.err = e;
        return snap;
    }
    snap.ioctl_ok = true;
    snap.total_returned = arg.num_element;

    pid_t my_pid = getpid();
    for (uint32_t i = 0; i < arg.num_element && i < MAX_ENTRIES; ++i) {
        if (entries[i].pid == my_pid) {
            snap.entry = entries[i];
            snap.found = true;
            break;
        }
    }
    return snap;
}

void print_snapshot(const char* label, const HwctxSnapshot& snap) {
    if (!snap.ioctl_ok) {
        std::printf("[%s] ioctl FAILED errno=%d (%s)\n", label, snap.err,
                    strerror(snap.err));
        return;
    }
    if (!snap.found) {
        std::printf("[%s] no entry for pid=%d (driver returned %u total)\n",
                    label, getpid(), snap.total_returned);
        return;
    }
    const auto& e = snap.entry;
    std::printf("[%s] pid=%lld ctx_id=%u hwctx_id=%u start_col=%u num_col=%u state=%u\n",
                label, static_cast<long long>(e.pid), e.context_id, e.hwctx_id,
                e.start_col, e.num_col, e.state);
    std::printf("  submissions=%llu completions=%llu suspensions=%llu errors=%llu\n",
                static_cast<unsigned long long>(e.command_submissions),
                static_cast<unsigned long long>(e.command_completions),
                static_cast<unsigned long long>(e.suspensions),
                static_cast<unsigned long long>(e.errors));
    std::printf("  txn_op_idx=%u (0x%08x) ctx_pc=0x%08x\n",
                e.txn_op_idx, e.txn_op_idx, e.ctx_pc);
    std::printf("  fatal: type=%u exception_type=%u exception_pc=0x%08x app_module=%u\n",
                e.fatal_error_type, e.fatal_error_exception_type,
                e.fatal_error_exception_pc, e.fatal_error_app_module);
    std::printf("  migrations=%llu preemptions=%llu heap_usage=%llu\n",
                static_cast<unsigned long long>(e.migrations),
                static_cast<unsigned long long>(e.preemptions),
                static_cast<unsigned long long>(e.heap_usage));
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--xclbin") args.xclbin = next();
        else if (a == "--insts") args.insts = next();
        else if (a == "--k") args.k = std::stoi(next());
        else if (a == "--n") args.n = std::stoi(next());
        else if (a == "-v" || a == "--verbose") args.verbose = true;
        else if (a == "--pre-snapshot") args.pre_snapshot = true;
        else if (a == "-h" || a == "--help") {
            std::printf("Usage: txn-poll-probe [opts]\n"
                        "  --xclbin PATH      (default: %s)\n"
                        "  --insts PATH       (default: %s)\n"
                        "  --k K              (default: %d) -- chain-sweep K\n"
                        "  --n N              (default: %d) -- chain-sweep N\n"
                        "  --pre-snapshot     also query before run.start()\n"
                        "  -v, --verbose\n",
                        DEFAULT_XCLBIN, DEFAULT_INSTS, DEFAULT_K, DEFAULT_N);
            return 0;
        }
        else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 2;
        }
    }

    const int K = args.k;
    const int N = args.n;
    const int TOTAL = K * N;

    auto instr_v = load_instr_binary(args.insts);
    if (args.verbose) {
        std::fprintf(stderr, "K=%d N=%d TOTAL=%d instr_words=%zu\n", K, N, TOTAL,
                     instr_v.size());
    }

    xrt::device device(0);
    xrt::xclbin xclbin(args.xclbin);
    auto kernels = xclbin.get_kernels();
    if (kernels.empty()) {
        std::fprintf(stderr, "no kernels in xclbin\n");
        return 2;
    }
    // Match the mlir-aie test.cpp pattern: find by "MLIR_AIE" prefix rather
    // than blindly taking kernels[0]. The xclbin may carry auxiliary kernels
    // and picking the wrong one causes BO group_ids to map to a different
    // argument signature -- which manifests as IO_PAGE_FAULTs at NULL.
    const std::string kernel_prefix = "MLIR_AIE";
    std::string kernel_name;
    for (auto& k : kernels) {
        auto name = k.get_name();
        if (args.verbose) std::fprintf(stderr, "  kernel: %s\n", name.c_str());
        if (name.rfind(kernel_prefix, 0) == 0) {
            kernel_name = name;
            break;
        }
    }
    if (kernel_name.empty()) {
        std::fprintf(stderr, "no kernel matching prefix '%s'\n",
                     kernel_prefix.c_str());
        return 2;
    }
    device.register_xclbin(xclbin);
    xrt::hw_context ctx(device, xclbin.get_uuid());
    auto kernel = xrt::kernel(ctx, kernel_name);

    // K-sweep kernel signature (k1/test.cpp): opcode, instr, instr_count,
    // bo_inA[TOTAL*4], bo_inB[N*4], bo_out[TOTAL*4].
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_inA = xrt::bo(device, TOTAL * sizeof(int32_t),
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_inB = xrt::bo(device, N * sizeof(int32_t),
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_out = xrt::bo(device, TOTAL * sizeof(int32_t),
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
    // Trace BO. The chain-sweep xclbins are compiled with trace events
    // enabled in the MLIR runtime sequence; if no BO is bound to arg 6,
    // firmware writes trace events to NULL and triggers IOMMU faults at
    // 0x0-0x200. This matches what trace-prepare.py injects into the
    // build-dir test.cpp -- see tools/cpp_trace_patch.py.
    constexpr size_t TRACE_SIZE = 1u << 20;  // 1 MiB
    auto bo_trace = xrt::bo(device, TRACE_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                            kernel.group_id(6));
    std::memset(bo_trace.map<void*>(), 0, TRACE_SIZE);
    bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto* in_a = bo_inA.map<uint32_t*>();
    for (int i = 0; i < TOTAL; ++i) in_a[i] = static_cast<uint32_t>(i + 1);
    std::memcpy(bo_instr.map<void*>(), instr_v.data(),
                instr_v.size() * sizeof(uint32_t));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    if (args.pre_snapshot) {
        // Before run.start() the partition is not yet allocated, so we
        // expect no entry here. Captured for confirmation.
        auto pre = snapshot_hwctx_state();
        print_snapshot("pre", pre);
    }

    auto t0 = std::chrono::steady_clock::now();
    auto run = kernel(3u, bo_instr, instr_v.size(), bo_inA, bo_inB, bo_out, bo_trace);
    auto state = run.wait(std::chrono::seconds(10));
    auto t1 = std::chrono::steady_clock::now();
    uint64_t kernel_us = std::chrono::duration_cast<std::chrono::microseconds>(
                             t1 - t0).count();

    auto post = snapshot_hwctx_state();

    std::printf("kernel_us=%llu state=%d K=%d N=%d instr_words=%zu\n",
                static_cast<unsigned long long>(kernel_us), static_cast<int>(state),
                K, N, instr_v.size());
    print_snapshot("post", post);

    return (state == ERT_CMD_STATE_COMPLETED) ? 0 : 1;
}
