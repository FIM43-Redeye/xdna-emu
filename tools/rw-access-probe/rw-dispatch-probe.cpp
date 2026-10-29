// rw-dispatch-probe: pre-sample Timer_Low, dispatch a known kernel
// run, wait for completion, post-sample Timer_Low.  Pins down what
// Timer_Low's source clock and gating behavior actually are.
//
// Hardcoded to the add_one_using_dma kernel layout (IN_SIZE=64,
// signature: opcode=3, instr, instr_size, inA, inB, out, trace).
// Other kernels need their own probes -- this is a calibration tool,
// not a generic harness.
//
// Usage:
//   rw-dispatch-probe --xclbin <path> --instr <insts.bin> \
//                     [--col N --row N --reg 0xXXXXX]      \
//                     [--inner-reads N]
//
// --inner-reads N samples Timer_Low N times BEFORE and N times AFTER
// the kernel run (default 1). Useful to compare the per-AIE_RW_ACCESS
// idle delta (~11,870 ticks from T2) against the during-run delta.
//
// Requires root or CAP_SYS_ADMIN.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <sys/types.h>
#include <unistd.h>

#include "xrt/xrt_aie.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

namespace {

constexpr int IN_SIZE = 64;
constexpr int OUT_SIZE = 64;
constexpr size_t TRACE_SIZE = 1048576;
constexpr uint32_t REG_TIMER_LOW = 0x340F8;

struct Args {
    std::string xclbin;
    std::string instr;
    int col = 0;
    int row = 2;
    uint32_t reg = REG_TIMER_LOW;
    int inner_reads = 1;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if      (s == "--xclbin"      && i + 1 < argc) a.xclbin = argv[++i];
        else if (s == "--instr"       && i + 1 < argc) a.instr  = argv[++i];
        else if (s == "--col"         && i + 1 < argc) a.col    = std::stoi(argv[++i]);
        else if (s == "--row"         && i + 1 < argc) a.row    = std::stoi(argv[++i]);
        else if (s == "--reg"         && i + 1 < argc) a.reg    = std::stoul(argv[++i], nullptr, 0);
        else if (s == "--inner-reads" && i + 1 < argc) a.inner_reads = std::stoi(argv[++i]);
    }
    if (a.inner_reads < 1) a.inner_reads = 1;
    return a;
}

std::vector<uint32_t> load_instr(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open instr file: " + path);
    std::vector<uint32_t> v;
    uint32_t w;
    while (f.read(reinterpret_cast<char*>(&w), sizeof(w))) v.push_back(w);
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    if (a.xclbin.empty() || a.instr.empty()) {
        std::fprintf(stderr, "usage: %s --xclbin <path> --instr <path> "
                             "[--col N --row N --reg 0xXXXXX] [--inner-reads N]\n", argv[0]);
        return 2;
    }

    std::printf("== rw-dispatch-probe ==\n");
    std::printf("xclbin       : %s\n", a.xclbin.c_str());
    std::printf("instr        : %s\n", a.instr.c_str());
    std::printf("Timer_Low    : col=%d row=%d reg=0x%05x\n", a.col, a.row, a.reg);
    std::printf("inner-reads  : %d before + %d after kernel run\n", a.inner_reads, a.inner_reads);

    auto instr_v = load_instr(a.instr);
    std::printf("instr count  : %zu words\n", instr_v.size());

    xrt::device dev{0};
    xrt::aie::device aie_dev{dev};
    xrt::xclbin xclbin{a.xclbin};
    dev.register_xclbin(xclbin);
    xrt::hw_context ctx{dev, xclbin.get_uuid()};

    // Find the kernel (single kernel xclbins; if multiple, take first).
    auto kernels = xclbin.get_kernels();
    if (kernels.empty()) { std::fprintf(stderr, "no kernels in xclbin\n"); return 2; }
    xrt::kernel k{ctx, kernels[0].get_name()};
    std::printf("kernel       : %s\n", kernels[0].get_name().c_str());

    // Buffers per add_one_using_dma test.cpp layout.
    auto bo_instr = xrt::bo(dev, instr_v.size() * sizeof(uint32_t),
                            XCL_BO_FLAGS_CACHEABLE, k.group_id(1));
    auto bo_inA = xrt::bo(dev, IN_SIZE * sizeof(int32_t),
                          XRT_BO_FLAGS_HOST_ONLY, k.group_id(3));
    auto bo_inB = xrt::bo(dev, IN_SIZE * sizeof(int32_t),
                          XRT_BO_FLAGS_HOST_ONLY, k.group_id(4));
    auto bo_out = xrt::bo(dev, OUT_SIZE * sizeof(int32_t),
                          XRT_BO_FLAGS_HOST_ONLY, k.group_id(5));
    auto bo_trace = xrt::bo(dev, TRACE_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                            k.group_id(6));

    uint32_t* bufInA = bo_inA.map<uint32_t*>();
    for (int i = 0; i < IN_SIZE; ++i) bufInA[i] = i + 1;

    void* bufInstr = bo_instr.map<void*>();
    std::memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

    std::memset(bo_trace.map<void*>(), 0, TRACE_SIZE);

    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    pid_t pid = getpid();
    uint16_t ctx_id = 1;
    auto sample = [&](int n, std::vector<uint32_t>& out, std::vector<double>& rt_us_out) {
        for (int i = 0; i < n; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            uint32_t v = aie_dev.read_aie_reg(pid, ctx_id,
                                              static_cast<uint16_t>(a.col),
                                              static_cast<uint16_t>(a.row), a.reg);
            auto t1 = std::chrono::steady_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            out.push_back(v);
            rt_us_out.push_back(us);
        }
    };

    std::vector<uint32_t> pre, post;
    std::vector<double> pre_rt, post_rt;

    // Pre-sample.
    sample(a.inner_reads, pre, pre_rt);
    auto t_before = std::chrono::steady_clock::now();

    // Dispatch.
    constexpr unsigned int opcode = 3;
    auto run = k(opcode, bo_instr, static_cast<uint32_t>(instr_v.size()),
                 bo_inA, bo_inB, bo_out, bo_trace);
    ert_cmd_state st = run.wait();
    auto t_after = std::chrono::steady_clock::now();
    double run_us = std::chrono::duration<double, std::micro>(t_after - t_before).count();

    // Post-sample.
    sample(a.inner_reads, post, post_rt);

    // Also sample Timer_Control to see if the runtime_sequence's write
    // is sticking. Address depends on tile module (core/shim = 0x34000,
    // memory = 0x14000, memtile = 0x94000); pick by row heuristic.
    uint32_t timer_ctrl_addr = (a.row == 1) ? 0x94000 : 0x34000;
    uint32_t timer_ctrl_val = 0xDEADBEEF;
    try {
        timer_ctrl_val = aie_dev.read_aie_reg(pid, ctx_id,
                                              static_cast<uint16_t>(a.col),
                                              static_cast<uint16_t>(a.row),
                                              timer_ctrl_addr);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[warning] Timer_Control read failed: %s\n", e.what());
    }

    // Report.
    std::printf("\nkernel run   : %s  (wall-clock %.0f us)\n",
                st == ERT_CMD_STATE_COMPLETED ? "COMPLETED" : "FAILED",
                run_us);

    std::printf("\n-- Timer_Low samples --\n");
    for (int i = 0; i < a.inner_reads; ++i) {
        std::printf("  pre[%d]  = 0x%08x  (%10u)   roundtrip=%.1fus\n",
                    i, pre[i], pre[i], pre_rt[i]);
    }
    for (int i = 0; i < a.inner_reads; ++i) {
        std::printf("  post[%d] = 0x%08x  (%10u)   roundtrip=%.1fus\n",
                    i, post[i], post[i], post_rt[i]);
    }

    uint32_t bracket_delta = post[0] - pre.back();
    std::printf("\nbracket delta (post[0] - pre[last]) = %u ticks\n", bracket_delta);
    std::printf("                                       across kernel run\n");

    if (a.inner_reads >= 2) {
        uint32_t pre_step  = pre[1]  - pre[0];
        uint32_t post_step = post[1] - post[0];
        std::printf("\npre-step  (pre[1]  - pre[0])  = %u ticks  (idle baseline)\n", pre_step);
        std::printf("post-step (post[1] - post[0]) = %u ticks  (idle baseline)\n", post_step);
    }

    std::printf("\nTimer_Control (addr 0x%05x col=%d row=%d) post-run = 0x%08x  (%u)\n",
                timer_ctrl_addr, a.col, a.row, timer_ctrl_val, timer_ctrl_val);
    if (timer_ctrl_val == 0) {
        std::printf("  -- Timer_Control is ZERO: runtime_sequence's config did not stick\n");
    } else if (timer_ctrl_val != 0xDEADBEEF) {
        uint32_t reset_event = (timer_ctrl_val >> 8) & 0x7F;
        uint32_t reset_bit = (timer_ctrl_val >> 31) & 0x1;
        std::printf("  -- Reset_Event = %u, Reset bit = %u\n", reset_event, reset_bit);
    }

    // Verify output to confirm kernel really ran.
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    uint32_t* bufOut = bo_out.map<uint32_t*>();
    int errors = 0;
    for (int i = 0; i < OUT_SIZE; ++i) {
        if (bufOut[i] != static_cast<uint32_t>(i + 2)) ++errors;
    }
    std::printf("\noutput check : %s (%d / %d errors)\n",
                errors == 0 ? "PASS" : "FAIL", errors, OUT_SIZE);

    return st == ERT_CMD_STATE_COMPLETED && errors == 0 ? 0 : 1;
}
