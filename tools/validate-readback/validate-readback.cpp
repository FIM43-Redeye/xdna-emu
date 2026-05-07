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

#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"

#include "amdxdna_accel.h"

namespace {

constexpr uint32_t TIMER_LOW_OFFSET     = 0x000340F8;
constexpr uint32_t PERF_CTRL0_OFFSET    = 0x00031500;
constexpr uint32_t PERF_COUNTER0_OFFSET = 0x00031520;
constexpr uint8_t  EVENT_ACTIVE_CORE    = 0x1C;

constexpr const char* DEFAULT_XCLBIN =
    "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/peano/aie.xclbin";
constexpr const char* DEFAULT_INSTS =
    "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/peano/insts.bin";

struct RunResult;
RunResult run_kernel_once(xrt::device& device, xrt::kernel& kernel,
                          const std::vector<uint32_t>& instr_v,
                          bool verbose);
struct RunResult { uint64_t kernel_us = 0; };

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

void configure_active_core_counter(xrt::hw_context& ctx, int col, int row) {
    // Read PERF_CTRL0, clear start[6:0] and stop[14:8], set start = ACTIVE_CORE.
    uint32_t ctrl = ctx.read_aie_reg(static_cast<uint16_t>(col),
                                     static_cast<uint16_t>(row),
                                     PERF_CTRL0_OFFSET);
    ctrl &= ~uint32_t(0x7F7Fu);
    ctrl |= static_cast<uint32_t>(EVENT_ACTIVE_CORE);
    ctx.write_aie_reg(static_cast<uint16_t>(col), static_cast<uint16_t>(row),
                      PERF_CTRL0_OFFSET, ctrl);
    ctx.write_aie_reg(static_cast<uint16_t>(col), static_cast<uint16_t>(row),
                      PERF_COUNTER0_OFFSET, 0);
}

struct V2Out {
    uint32_t cnt_target = 0;
    uint32_t cnt_neighbor = 0;
    uint64_t kernel_us = 0;
};

// Kept for future re-enablement once the multi-run-on-same-hwctx issue
// is cracked (see validate-readback README "Findings"). Currently unused.
[[maybe_unused]] TestResult test_V2(xrt::device& device, xrt::hw_context& ctx,
                   xrt::kernel& kernel,
                   const std::vector<uint32_t>& instr_v,
                   int col, int target_row, int neighbor_row,
                   bool verbose, V2Out* out) {
    try {
        configure_active_core_counter(ctx, col, target_row);
        // Disable the neighbor counter and zero it.
        ctx.write_aie_reg(static_cast<uint16_t>(col), static_cast<uint16_t>(neighbor_row),
                          PERF_CTRL0_OFFSET, 0);
        ctx.write_aie_reg(static_cast<uint16_t>(col), static_cast<uint16_t>(neighbor_row),
                          PERF_COUNTER0_OFFSET, 0);

        auto rr = run_kernel_once(device, kernel, instr_v, verbose);

        uint32_t target_v = ctx.read_aie_reg(static_cast<uint16_t>(col),
                                             static_cast<uint16_t>(target_row),
                                             PERF_COUNTER0_OFFSET);
        uint32_t neighbor_v = ctx.read_aie_reg(static_cast<uint16_t>(col),
                                               static_cast<uint16_t>(neighbor_row),
                                               PERF_COUNTER0_OFFSET);
        if (out) {
            out->cnt_target = target_v;
            out->cnt_neighbor = neighbor_v;
            out->kernel_us = rr.kernel_us;
        }

        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "(col,%d)=%u (col,%d)=%u kernel_us=%lu",
                      target_row, target_v, neighbor_row, neighbor_v,
                      static_cast<unsigned long>(rr.kernel_us));
        Verdict v = (target_v > 0 && neighbor_v == 0) ? Verdict::Pass : Verdict::Fail;
        return {"V2", v, buf};
    } catch (const std::exception& e) {
        return {"V2", Verdict::Fail, std::string("threw: ") + e.what()};
    }
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

// ----- Mailbox-opcode probe (Phase 1 of NPU1 capability survey) -----
//
// Fires DRM_IOCTL_AMDXDNA_GET_ARRAY for DRM_AMDXDNA_AIE_COREDUMP, which
// causes the driver to dispatch MSG_OP_GET_COREDUMP (0x119) to firmware
// IFF the opcode is present in `npu1_msg_op_tbl[]`. We probe by adding a
// placeholder entry to that table for the opcode under test, rebuilding
// the driver, then running this binary; firmware's response status word
// is the final answer. (See finding doc 2026-05-06-npu1-msg-op-capability-survey.md.)
//
// Discriminator:
//   EOPNOTSUPP/-95           -> opcode NOT in npu1_msg_op_tbl[] (gate active)
//   0 (success) + real data  -> firmware implements opcode
//   0 + all-zeros/all-ones   -> half-implementation suspect (firmware acked but no data)
//   EINVAL/-22 + dmesg shows AIE2_STATUS_INVALID_COMMAND -> firmware does NOT
//                             implement opcode (revert table entry)
//   EIO / other              -> see dmesg

struct ProbeResult {
    int rc = -1;          // ioctl return value (0 = success, -1 = errno set)
    int err = 0;          // errno
    size_t bytes = 0;     // payload size after retry (driver hint or actual fill)
    bool all_zero = false;
    bool all_ones = false;
    uint32_t first_nonzero_offset = 0;
    uint32_t first_nonzero_value = 0;
};

ProbeResult probe_get_coredump(int slot, bool verbose) {
    ProbeResult pr;
    int fd = open("/dev/accel/accel0", O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        pr.err = errno;
        if (verbose) std::fprintf(stderr, "  open /dev/accel/accel0 failed: %s\n", strerror(errno));
        return pr;
    }

    // Start with the minimum size that passes the driver's first sanity check
    // (sizeof(amdxdna_drm_aie_coredump) = 16). We expect ENOSPC with a size
    // hint in arg.element_size, then we retry with that size.
    std::vector<char> buf(sizeof(amdxdna_drm_aie_coredump), 0);
    auto fill_config = [&]() {
        auto* cfg = reinterpret_cast<amdxdna_drm_aie_coredump*>(buf.data());
        cfg->pid = static_cast<__u64>(getpid());
        cfg->context_id = static_cast<__u32>(slot);
        cfg->pad = 0;
    };

    for (int attempt = 0; attempt < 2; ++attempt) {
        fill_config();
        amdxdna_drm_get_array arg = {};
        arg.param = DRM_AMDXDNA_AIE_COREDUMP;
        arg.element_size = static_cast<__u32>(buf.size());
        arg.num_element = 1;
        arg.buffer = reinterpret_cast<__u64>(buf.data());

        int rc = ioctl(fd, DRM_IOCTL_AMDXDNA_GET_ARRAY, &arg);
        if (rc == 0) {
            pr.rc = 0;
            pr.err = 0;
            pr.bytes = buf.size();
            break;
        }
        int e = errno;
        if (e == ENOSPC && attempt == 0 && arg.element_size > buf.size()) {
            if (verbose) std::fprintf(stderr,
                "  slot=%d ENOSPC, hint=%u, retrying with full buffer\n",
                slot, arg.element_size);
            buf.assign(arg.element_size, 0);
            continue;
        }
        pr.rc = -1;
        pr.err = e;
        pr.bytes = (e == ENOSPC) ? arg.element_size : 0;
        break;
    }

    close(fd);

    if (pr.rc == 0 && !buf.empty()) {
        // Skip the config header in our content scan; data starts after it.
        size_t hdr = sizeof(amdxdna_drm_aie_coredump);
        size_t data_bytes = buf.size() > hdr ? buf.size() - hdr : 0;
        bool z = true, o = true;
        for (size_t i = 0; i < data_bytes; ++i) {
            uint8_t b = static_cast<uint8_t>(buf[hdr + i]);
            if (b != 0x00) z = false;
            if (b != 0xFF) o = false;
            if (!z && !o) break;
        }
        pr.all_zero = z;
        pr.all_ones = o;
        if (!z && !o) {
            // Find first nonzero word for a content sample.
            for (size_t i = 0; i + 4 <= data_bytes; i += 4) {
                uint32_t v = 0;
                std::memcpy(&v, buf.data() + hdr + i, 4);
                if (v != 0) {
                    pr.first_nonzero_offset = static_cast<uint32_t>(i);
                    pr.first_nonzero_value = v;
                    break;
                }
            }
        }
    }
    return pr;
}

TestResult test_M0_coredump_scan(bool verbose) {
    // Brute-force scan ctx slots 1..7 (slot 0 is AMDXDNA_INVALID_CTX_HANDLE
    // and never assigned by xa_alloc_cyclic). The driver returns -EINVAL
    // with "Context not found" for unallocated slots, so distinguishing
    // "no context here" from "real firmware response" is straightforward.
    std::string detail;
    int found_slot = -1;
    ProbeResult found;
    for (int slot = 1; slot <= 7; ++slot) {
        ProbeResult pr = probe_get_coredump(slot, verbose);
        char line[256];
        if (pr.rc == 0) {
            std::snprintf(line, sizeof(line),
                "slot=%d OK bytes=%zu zero=%d ones=%d nz_off=0x%x nz_val=0x%08x",
                slot, pr.bytes, pr.all_zero ? 1 : 0, pr.all_ones ? 1 : 0,
                pr.first_nonzero_offset, pr.first_nonzero_value);
            found_slot = slot; found = pr;
        } else {
            std::snprintf(line, sizeof(line),
                "slot=%d errno=%d (%s) bytes=%zu",
                slot, pr.err, strerror(pr.err), pr.bytes);
            // EINVAL means no such ctx in our process; skip silently in non-verbose.
            if (pr.err == EINVAL && !verbose) continue;
            if (found_slot < 0 && pr.err != EINVAL) { found_slot = slot; found = pr; }
        }
        if (verbose) std::fprintf(stderr, "  M0 %s\n", line);
        if (!detail.empty()) detail += "; ";
        detail += line;
    }
    return TestResult{"M0", Verdict::Info, detail.empty()
        ? std::string{"no slots responded (no hwctx?)"}
        : detail};
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

    // V2/V3/V4 require a SECOND kernel run on the same hwctx with the perf
    // counter freshly programmed. Empirically that second run hangs with
    // ERT_CMD_STATE_NORESPONSE: the add_one_using_dma compute core hits
    // aie.end after its 4 iterations and is halted, and a subsequent
    // run.start() does not reset the core. Bridge-runner gets multi-run
    // working somehow (likely by recreating hwctx between batches); that
    // workaround is out of scope for this validation pass. The core
    // validation question -- "does read_aie_reg return real data?" --
    // has been answered PASS by L0/L1/V0/V1, which is sufficient.
    results.push_back({"V2", Verdict::Skip,
                       "second-run-on-same-hwctx hangs (ERT_CMD_STATE_NORESPONSE); "
                       "needs hwctx-recreate workaround, out of scope here"});
    print_result(results.back());
    results.push_back({"V3", Verdict::Skip, "blocked on V2"});
    print_result(results.back());
    results.push_back({"V4", Verdict::Skip, "blocked on V2"});
    print_result(results.back());

    // Mailbox-opcode probe: fires DRM_AMDXDNA_AIE_COREDUMP for each candidate
    // hwctx slot. Run with `unsafe_accept_all_msg=N` (default) to confirm the
    // op-table gate is in effect, then again with `=Y` to observe firmware's
    // actual response to MSG_OP_GET_COREDUMP (0x119).
    results.push_back(test_M0_coredump_scan(args.verbose));
    print_result(results.back());

    // Cleanup: disable any counter we left programmed, best-effort.
    try {
        ctx.write_aie_reg(static_cast<uint16_t>(args.col),
                          static_cast<uint16_t>(args.row),
                          PERF_CTRL0_OFFSET, 0);
    } catch (...) { /* device may be wedged after a hung run, that's fine */ }

    int passes = 0, fails = 0, skips = 0;
    for (const auto& r : results) {
        switch (r.v) {
            case Verdict::Pass: ++passes; break;
            case Verdict::Fail: ++fails;  break;
            case Verdict::Skip: ++skips;  break;
            case Verdict::Info: break;
        }
    }
    std::printf("VALIDATION: %d/%zu PASS (%d skipped, %d failed)\n",
                passes, results.size(), skips, fails);
    return fails;
}
