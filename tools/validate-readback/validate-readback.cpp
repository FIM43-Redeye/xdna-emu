// validate-readback: probe xrt::hw_context::read_aie_reg for ground-truth honesty.
// Throwaway harness; see docs/superpowers/specs/2026-05-06-validate-read-aie-reg-design.md.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"

namespace {

constexpr uint32_t TIMER_LOW_OFFSET     = 0x000340F8;
constexpr uint32_t PERF_CTRL0_OFFSET    = 0x00031500;
constexpr uint32_t PERF_COUNTER0_OFFSET = 0x00031520;
constexpr uint8_t  EVENT_ACTIVE_CORE    = 0x1C;

constexpr const char* DEFAULT_XCLBIN =
    "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/peano/aie.xclbin";
constexpr const char* DEFAULT_INSTS =
    "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/peano/insts.bin";

enum class Verdict { Pass, Fail, Info, Skip };
struct TestResult {
    std::string id;
    Verdict v;
    std::string detail;
};

const char* verdict_str(Verdict v) {
    switch (v) {
        case Verdict::Pass: return "PASS";
        case Verdict::Fail: return "FAIL";
        case Verdict::Info: return "INFO";
        case Verdict::Skip: return "SKIP";
    }
    return "?";
}

void print_result(const TestResult& r) {
    std::printf("[%s] %-4s %s\n", r.id.c_str(), verdict_str(r.v), r.detail.c_str());
}

TestResult test_V1(xrt::hw_context& ctx, int col, int row) {
    constexpr uint32_t MAGIC = 0xDEADBEEFu;
    try {
        // Make sure the counter is not actively counting (clear start_event).
        ctx.write_aie_reg(static_cast<uint16_t>(col), static_cast<uint16_t>(row),
                          PERF_CTRL0_OFFSET, 0);
        ctx.write_aie_reg(static_cast<uint16_t>(col), static_cast<uint16_t>(row),
                          PERF_COUNTER0_OFFSET, MAGIC);
        uint32_t got = ctx.read_aie_reg(static_cast<uint16_t>(col),
                                        static_cast<uint16_t>(row),
                                        PERF_COUNTER0_OFFSET);
        char buf[128];
        std::snprintf(buf, sizeof(buf), "wrote 0x%08x, read 0x%08x", MAGIC, got);
        // Allow tiny advance if start_event leaked from prior state.
        Verdict v = (got == MAGIC || (got > MAGIC && got - MAGIC < 100))
                    ? Verdict::Pass : Verdict::Fail;
        return {"V1", v, buf};
    } catch (const std::exception& e) {
        return {"V1", Verdict::Fail, std::string("threw: ") + e.what()};
    }
}

TestResult test_V0(xrt::hw_context& ctx, int col, int row) {
    // Test that TIMER_LOW returns changing, wall-time-correlated values.
    // We do NOT assert a specific clock rate: the tile clock may be gated
    // when the core is idle, so the observed effective rate is informational.
    try {
        uint32_t t0 = ctx.read_aie_reg(static_cast<uint16_t>(col),
                                       static_cast<uint16_t>(row),
                                       TIMER_LOW_OFFSET);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        uint32_t t1 = ctx.read_aie_reg(static_cast<uint16_t>(col),
                                       static_cast<uint16_t>(row),
                                       TIMER_LOW_OFFSET);
        uint32_t delta = t1 - t0; // wraparound fine for any plausible delta
        double effective_mhz = static_cast<double>(delta) / 1000.0; // cycles per us
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "timer_lo: 0x%08x -> 0x%08x (delta=%u over ~1ms, ~%.1f MHz effective)",
                      t0, t1, delta, effective_mhz);
        // PASS: counter is live and advancing at any rate from "barely" to
        // "full AIE core clock." 100..2_000_000_000 covers idle clock-gating
        // through full-speed; 0 or wildly higher would be suspicious.
        Verdict v = (delta > 100u && delta < 2'000'000'000u) ? Verdict::Pass : Verdict::Fail;
        return {"V0", v, buf};
    } catch (const std::exception& e) {
        return {"V0", Verdict::Fail, std::string("threw: ") + e.what()};
    }
}

TestResult test_L1(xrt::hw_context& ctx, int col, int row) {
    // Same hwctx, AFTER the dummy run has completed and allocated the partition.
    try {
        uint32_t v = ctx.read_aie_reg(static_cast<uint16_t>(col),
                                      static_cast<uint16_t>(row),
                                      TIMER_LOW_OFFSET);
        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      "post-warmup pre-launch read OK, TIMER_LOW=0x%08x", v);
        return {"L1", Verdict::Pass, buf};
    } catch (const std::exception& e) {
        return {"L1", Verdict::Fail,
                std::string("post-warmup read still failed: ") + e.what()};
    }
}

TestResult test_L0(xrt::hw_context& ctx, int col, int row) {
    try {
        uint32_t v = ctx.read_aie_reg(static_cast<uint16_t>(col),
                                      static_cast<uint16_t>(row),
                                      TIMER_LOW_OFFSET);
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "pre-launch read SUCCEEDED (lifecycle bug not present?), value=0x%08x",
                      v);
        return {"L0", Verdict::Info, buf};
    } catch (const std::exception& e) {
        return {"L0", Verdict::Pass,
                std::string("pre-launch read threw as expected: ") + e.what()};
    }
}

struct Args {
    std::string xclbin = DEFAULT_XCLBIN;
    std::string insts  = DEFAULT_INSTS;
    int col = 0;
    int row = 2;
    bool verbose = false;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if      (s == "--xclbin" && i + 1 < argc) a.xclbin = argv[++i];
        else if (s == "--insts"  && i + 1 < argc) a.insts  = argv[++i];
        else if (s == "--col"    && i + 1 < argc) a.col    = std::atoi(argv[++i]);
        else if (s == "--row"    && i + 1 < argc) a.row    = std::atoi(argv[++i]);
        else if (s == "-v" || s == "--verbose")   a.verbose = true;
        else {
            std::fprintf(stderr, "unknown arg: %s\n", s.c_str());
            std::exit(2);
        }
    }
    return a;
}

std::vector<uint32_t> load_insts(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open insts: " + path);
    auto size = f.tellg();
    f.seekg(0);
    std::vector<uint32_t> v(size / 4);
    f.read(reinterpret_cast<char*>(v.data()), size);
    return v;
}

struct RunResult {
    uint64_t kernel_us = 0;
};

RunResult run_kernel_once(xrt::device& device, xrt::kernel& kernel,
                          const std::vector<uint32_t>& instr_v,
                          bool verbose) {
    // add_one_using_dma kernarg layout (per peano build of MLIR_AIE):
    //   0: opcode (3 = ELF kernel)
    //   1: instr_bo
    //   2: ninstrs
    //   3: input BO   (group_id 3)
    //   4: middle BO  (group_id 4, allocated by runtime_sequence but unused)
    //   5: output BO  (group_id 5)
    constexpr size_t IN_BYTES  = 64 * sizeof(int32_t);
    constexpr size_t MID_BYTES = 32 * sizeof(int32_t);
    constexpr size_t OUT_BYTES = 64 * sizeof(int32_t);

    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    std::memcpy(bo_instr.map<void*>(), instr_v.data(),
                instr_v.size() * sizeof(uint32_t));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto bo_in  = xrt::bo(device, IN_BYTES,  XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_mid = xrt::bo(device, MID_BYTES, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_out = xrt::bo(device, OUT_BYTES, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
    std::memset(bo_in.map<void*>(),  0, IN_BYTES);
    std::memset(bo_mid.map<void*>(), 0, MID_BYTES);
    std::memset(bo_out.map<void*>(), 0, OUT_BYTES);
    bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_mid.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = xrt::run(kernel);
    run.set_arg(0, 3u);
    run.set_arg(1, bo_instr);
    run.set_arg(2, static_cast<uint32_t>(instr_v.size()));
    run.set_arg(3, bo_in);
    run.set_arg(4, bo_mid);
    run.set_arg(5, bo_out);

    auto t0 = std::chrono::steady_clock::now();
    run.start();
    auto state = run.wait(std::chrono::seconds(30));
    auto t1 = std::chrono::steady_clock::now();
    if (state != ERT_CMD_STATE_COMPLETED) {
        throw std::runtime_error("kernel did not complete (state=" +
                                 std::to_string(static_cast<int>(state)) + ")");
    }
    RunResult r;
    r.kernel_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    if (verbose) std::fprintf(stderr, "  run completed in %lu us\n",
                              static_cast<unsigned long>(r.kernel_us));
    return r;
}

} // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    std::printf("validate-readback: xclbin=%s col=%d row=%d\n",
                args.xclbin.c_str(), args.col, args.row);

    auto device = xrt::device(0);
    auto xclbin = xrt::xclbin(args.xclbin);
    device.register_xclbin(xclbin);
    auto ctx = xrt::hw_context(device, xclbin.get_uuid());
    auto kernels = xclbin.get_kernels();
    if (kernels.empty()) throw std::runtime_error("no kernels in xclbin");
    auto kernel = xrt::kernel(ctx, kernels[0].get_name());

    std::printf("[INFO] loaded xclbin, kernel=%s\n", kernels[0].get_name().c_str());

    std::vector<TestResult> results;
    results.push_back(test_L0(ctx, args.col, args.row));
    print_result(results.back());

    auto instr_v = load_insts(args.insts);
    std::printf("[INFO] loaded %zu instr words\n", instr_v.size());

    std::printf("[INFO] running dummy kernel to allocate partition...\n");
    auto dummy = run_kernel_once(device, kernel, instr_v, args.verbose);
    std::printf("[INFO] dummy run kernel_us=%lu\n",
                static_cast<unsigned long>(dummy.kernel_us));

    results.push_back(test_L1(ctx, args.col, args.row));
    print_result(results.back());

    results.push_back(test_V0(ctx, args.col, args.row));
    print_result(results.back());

    results.push_back(test_V1(ctx, args.col, args.row));
    print_result(results.back());

    return 0;
}
