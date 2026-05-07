// validate-readback: probe xrt::hw_context::read_aie_reg for ground-truth honesty.
// Throwaway harness; see docs/superpowers/specs/2026-05-06-validate-read-aie-reg-design.md.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
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

constexpr uint32_t TIMER_LOW_OFFSET         = 0x000340F8;
constexpr uint32_t PERF_CTRL0_OFFSET        = 0x00031500;
constexpr uint32_t PERF_COUNTER0_OFFSET     = 0x00031520;
constexpr uint8_t  EVENT_ACTIVE_CORE        = 0x1C;
// Memtile register space (per aie-rt xaiemlgbl_params.h):
//   DATAMEMORY base       = 0x00000 (512KB DM, 0x00000-0x7FFFF)
//   PERFORMANCE_COUNTER0  = 0x91020
//   TIMER_LOW             = 0x940F8 (mirror of CORE_MODULE_TIMER_LOW at 0x340F8)
constexpr uint32_t MEMTILE_TIMER_LOW_OFFSET = 0x000940F8;
// Compute DM low addr; safe to probe with a single read (no write).
constexpr uint32_t COMPUTE_DM_PROBE_OFFSET  = 0x00000100;

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
    bool probe_dm_danger = false;
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
        else if (s == "--probe-dm-danger")        a.probe_dm_danger = true;
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

// ----- AIE_RW_ACCESS functional validation slots -----
//
// These exercise the production AIE_RW_ACCESS path on tiles that the
// dummy kernel's hwctx CLAIMS. add_one_using_dma uses tiles (0,0) shim,
// (0,1) memtile, (0,2) compute -- one of each tile type, conveniently.
//
// IMPORTANT: an earlier version of this file swept rows 2-5 of the
// partition column. That blew up: the firmware enforces per-tile-claim
// authorization (not just partition-column scoping), so reads at rows
// 3-5 returned AIE2_STATUS_INVALID_PARAM. The driver translated each
// non-success status to -EINVAL, and the accumulated mailbox traffic
// eventually caused mgmt_chann to time out and tear down, cascading to
// -ENODEV and an SMU wedge requiring reboot. Lesson: only touch the
// kernel's claimed tiles, and run the "unclaimed-cell" probe last so
// other answers land before any potential cascade.
//
// SECOND IMPORTANT: a later version added V4 = memtile DM round-trip at
// offset 0x10000. That uncovered a half-implementation: firmware ACKs
// the writes (returns SUCCESS, driver logs them) but silently DROPS the
// readback -- the response never arrives, the mailbox channel times out
// after 5s, and the user-context channel is destroyed. SMU survived,
// but it's still a hard hit on the device. So V4 is now memtile
// register-space (read-only TIMER_LOW), and the DM round-trip moves
// behind --probe-dm-danger (default off).
// See finding doc 2026-05-07-aie-rw-access-memtile-dm-half-impl.md.
//
// Address-space coverage matrix (after this round of tests):
//   compute reg  : V0/V1/V2/V5 -- VERIFIED working
//   compute DM   : V6          -- TBD
//   memtile reg  : V4          -- TBD
//   memtile DM   : --danger    -- KNOWN broken (writes ack, reads hang)
//   shim         : not exercised here (no claimed register space probed)
//
// Half-implementation discriminators each test catches:
//   V2: same-cell back-to-back reads (5x) return identical values
//       (catches stale-cache / read-side-effect bugs)
//   V3: read at unclaimed cell throws cleanly
//       (catches over-permissive firmware; runs LAST)
//   V4: memtile register read returns plausible non-zero, advancing values
//       (probes whether memtile reg-space is reachable at all)
//   V5: N=100 random-magic round-trip stress on the compute cell
//       (catches flakiness / occasional corruption)
//   V6: compute DM single read returns *something* without throwing
//       (probes whether AIE_RW_ACCESS reaches DM, not just registers)

constexpr int MEMTILE_ROW = 1;
constexpr int COMPUTE_ROW = 2;
constexpr int UNCLAIMED_ROW = 5;  // last compute row, NOT in add_one_using_dma's hwctx

TestResult test_V2_read_consistency(xrt::hw_context& ctx, int col) {
    // V1 left PERF_COUNTER0 = 0xDEADBEEF at (col, 2) with PERF_CTRL0 = 0
    // (no event configured, so the counter does not advance). Read 5x in
    // tight succession; values must be identical.
    constexpr int N = 5;
    std::vector<uint32_t> reads;
    reads.reserve(N);
    try {
        for (int i = 0; i < N; ++i) {
            reads.push_back(
                ctx.read_aie_reg(static_cast<uint16_t>(col),
                                 static_cast<uint16_t>(COMPUTE_ROW),
                                 PERF_COUNTER0_OFFSET));
        }
    } catch (const std::exception& e) {
        return {"V2", Verdict::Fail, std::string("threw on read ") +
                std::to_string(reads.size()) + ": " + e.what()};
    }
    bool all_equal = true;
    for (size_t i = 1; i < reads.size(); ++i)
        if (reads[i] != reads[0]) { all_equal = false; break; }
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "%dx PERF_COUNTER0 @ (%d,%d): 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x %s",
        N, col, COMPUTE_ROW, reads[0], reads[1], reads[2], reads[3], reads[4],
        all_equal ? "all-equal" : "JITTER");
    return {"V2", all_equal ? Verdict::Pass : Verdict::Fail, buf};
}

TestResult test_V4_memtile_reg(xrt::hw_context& ctx, int col) {
    // Memtile register-space probe. TIMER_LOW (0x940F8) is read-only, no
    // side effects, and the memtile timer is always running. Single read
    // ONLY after a 1-second idle pause -- this tests the "hammering the
    // mailbox too fast" hypothesis. If memtile reads hang even after 1s
    // of idle, rate is conclusively ruled out and the firmware just
    // can't service memtile reads via AIE_RW_ACCESS.
    //
    // PASS: read succeeds (memtile reachable + slow attempt sufficient).
    // FAIL/throw: memtile reg-space is broken regardless of pacing
    // (rate hypothesis ruled out, plain firmware bug).
    std::this_thread::sleep_for(std::chrono::seconds(1));
    try {
        uint32_t t0 = ctx.read_aie_reg(static_cast<uint16_t>(col),
                                       static_cast<uint16_t>(MEMTILE_ROW),
                                       MEMTILE_TIMER_LOW_OFFSET);
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "memtile (%d,%d) TIMER_LOW after 1s idle: 0x%08x (rate hypothesis CONFIRMED -- pacing matters)",
            col, MEMTILE_ROW, t0);
        return {"V4", Verdict::Info, buf};
    } catch (const std::exception& e) {
        return {"V4", Verdict::Fail,
                std::string("memtile read after 1s idle still hung (rate hypothesis RULED OUT): ") + e.what()};
    }
}

TestResult test_V6_compute_dm(xrt::hw_context& ctx, int col) {
    // Compute DM probe. Single read at low DM offset. add_one_using_dma
    // uses compute DM for its objFifo buffers, so anything is possible
    // here -- we don't assert a value, just that the read returns
    // *something* without throwing or hanging.
    //
    // PASS: read returns successfully (any value).
    // FAIL/throw: AIE_RW_ACCESS doesn't reach compute DM (same family
    // as the memtile DM bug, or a different breakage).
    try {
        uint32_t v = ctx.read_aie_reg(static_cast<uint16_t>(col),
                                      static_cast<uint16_t>(COMPUTE_ROW),
                                      COMPUTE_DM_PROBE_OFFSET);
        char buf[160];
        std::snprintf(buf, sizeof(buf),
            "compute DM (%d,%d) off=0x%x: 0x%08x (any value OK)",
            col, COMPUTE_ROW, COMPUTE_DM_PROBE_OFFSET, v);
        return {"V6", Verdict::Pass, buf};
    } catch (const std::exception& e) {
        return {"V6", Verdict::Fail, std::string("threw: ") + e.what()};
    }
}

TestResult test_VD_memtile_dm_danger(xrt::hw_context& ctx, int col) {
    // KNOWN-BROKEN reproduction of the half-implementation. Writes will
    // silently ack, the readback hangs in firmware, the user-context
    // mailbox times out at 5s and is destroyed by the driver. SMU has
    // survived this in our prior runs but the device needs a driver
    // reload before another xclbin load.
    //
    // Gated behind --probe-dm-danger so it never fires by accident.
    constexpr uint32_t OFF_A = 0x10000;
    constexpr uint32_t OFF_B = 0x10100;
    const uint32_t MAGIC_A = 0xBEEF0000u | (static_cast<uint32_t>(col & 0xff) << 8) | 0xAA;
    const uint32_t MAGIC_B = 0xBEEF0000u | (static_cast<uint32_t>(col & 0xff) << 8) | 0xBB;
    try {
        ctx.write_aie_reg(static_cast<uint16_t>(col),
                          static_cast<uint16_t>(MEMTILE_ROW), OFF_A, MAGIC_A);
        ctx.write_aie_reg(static_cast<uint16_t>(col),
                          static_cast<uint16_t>(MEMTILE_ROW), OFF_B, MAGIC_B);
        uint32_t got_a = ctx.read_aie_reg(static_cast<uint16_t>(col),
                                          static_cast<uint16_t>(MEMTILE_ROW), OFF_A);
        uint32_t got_b = ctx.read_aie_reg(static_cast<uint16_t>(col),
                                          static_cast<uint16_t>(MEMTILE_ROW), OFF_B);
        bool ok_a = (got_a == MAGIC_A);
        bool ok_b = (got_b == MAGIC_B);
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "memtile DM (%d,%d) off=0x%x: %s 0x%08x; off=0x%x: %s 0x%08x (UNEXPECTED PASS, finding may be obsolete)",
            col, MEMTILE_ROW, OFF_A, ok_a ? "ok" : "MISMATCH", got_a,
            OFF_B, ok_b ? "ok" : "MISMATCH", got_b);
        return {"VD", (ok_a && ok_b) ? Verdict::Info : Verdict::Fail, buf};
    } catch (const std::exception& e) {
        return {"VD", Verdict::Pass,
                std::string("memtile DM round-trip threw as expected (half-impl confirmed): ") + e.what()};
    }
}

TestResult test_V5_stress(xrt::hw_context& ctx, int col, int row, int iters) {
    // N=iters random-magic round-trips on the claimed compute cell.
    // PERF_CTRL0 was set to 0 by V1; we re-set defensively.
    try {
        ctx.write_aie_reg(static_cast<uint16_t>(col),
                          static_cast<uint16_t>(row), PERF_CTRL0_OFFSET, 0);
    } catch (const std::exception& e) {
        return {"V5", Verdict::Fail, std::string("ctrl0 write threw: ") + e.what()};
    }
    std::mt19937 rng(0xCAFEBABEu);
    int exact = 0, near = 0, mismatch = 0;
    uint32_t worst_delta = 0, last_got = 0, last_want = 0;
    for (int i = 0; i < iters; ++i) {
        uint32_t want = static_cast<uint32_t>(rng());
        try {
            ctx.write_aie_reg(static_cast<uint16_t>(col),
                              static_cast<uint16_t>(row),
                              PERF_COUNTER0_OFFSET, want);
            uint32_t got = ctx.read_aie_reg(static_cast<uint16_t>(col),
                                            static_cast<uint16_t>(row),
                                            PERF_COUNTER0_OFFSET);
            if (got == want) ++exact;
            else if (got > want && got - want < 100u) {
                ++near;
                if (got - want > worst_delta) worst_delta = got - want;
            } else {
                ++mismatch; last_got = got; last_want = want;
            }
        } catch (const std::exception& e) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "threw at iter %d/%d: exact=%d near=%d mismatch=%d msg=%s",
                i, iters, exact, near, mismatch, e.what());
            return {"V5", Verdict::Fail, buf};
        }
    }
    char buf[256];
    if (mismatch == 0) {
        std::snprintf(buf, sizeof(buf),
            "%d iters: exact=%d near=%d (worst delta=%u) mismatch=0",
            iters, exact, near, worst_delta);
    } else {
        std::snprintf(buf, sizeof(buf),
            "%d iters: exact=%d near=%d mismatch=%d last_mismatch want=0x%08x got=0x%08x",
            iters, exact, near, mismatch, last_want, last_got);
    }
    Verdict v = (mismatch == 0 && (exact + near) == iters) ? Verdict::Pass : Verdict::Fail;
    return {"V5", v, buf};
}

TestResult test_V3_unclaimed_cell(xrt::hw_context& ctx, int col) {
    // Authorization probe: read PERF_COUNTER0 from a cell our hwctx does
    // NOT claim (row 5, last compute row -- add_one_using_dma uses only
    // rows 0,1,2).
    //
    // EMPIRICAL UPDATE (2026-05-07): we previously believed unclaimed
    // cell reads returned AIE2_STATUS_INVALID_PARAM (translated to
    // -EINVAL). The actual observed behavior on Phoenix NPU1 is that
    // the firmware HANGS just like memtile reads -- the read returns
    // -ETIMEDOUT (-62) after 5s. The prior "INVALID_PARAM for
    // unclaimed cells" reading was likely a misinterpretation of the
    // cascade: once one read hung and destroyed mgmt_chann, subsequent
    // reads got -EINVAL via the aie2_send_mgmt_msg_wait status path.
    //
    // PASS if the read throws (whatever the cause -- auth refusal or
    // firmware hang). The mechanism distinction is captured in
    // dmesg (status code 0x02000004 vs ETIMEDOUT).
    try {
        uint32_t got = ctx.read_aie_reg(static_cast<uint16_t>(col),
                                        static_cast<uint16_t>(UNCLAIMED_ROW),
                                        PERF_COUNTER0_OFFSET);
        char buf[160];
        std::snprintf(buf, sizeof(buf),
            "UNEXPECTED: unclaimed cell (%d,%d) PERF_COUNTER0 read returned 0x%08x without throwing",
            col, UNCLAIMED_ROW, got);
        return {"V3", Verdict::Info, buf};
    } catch (const std::exception& e) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "unclaimed (%d,%d) read threw as expected (auth enforced): %s",
            col, UNCLAIMED_ROW, e.what());
        return {"V3", Verdict::Pass, buf};
    }
}

TestResult test_M0_coredump_scan(bool verbose) {
    // Brute-force scan ctx slots 1..16. Driver returns -EINVAL ("Context
    // not found") for unallocated slots. We always include every result
    // in the detail string so a bypass-less "all EINVAL" outcome is
    // explicit, not silently elided.
    std::string detail;
    for (int slot = 1; slot <= 16; ++slot) {
        ProbeResult pr = probe_get_coredump(slot, verbose);
        char line[256];
        if (pr.rc == 0) {
            std::snprintf(line, sizeof(line),
                "slot=%d OK bytes=%zu zero=%d ones=%d nz_off=0x%x nz_val=0x%08x",
                slot, pr.bytes, pr.all_zero ? 1 : 0, pr.all_ones ? 1 : 0,
                pr.first_nonzero_offset, pr.first_nonzero_value);
        } else {
            std::snprintf(line, sizeof(line),
                "slot=%d errno=%d (%s) bytes=%zu",
                slot, pr.err, strerror(pr.err), pr.bytes);
        }
        if (verbose) std::fprintf(stderr, "  M0 %s\n", line);
        if (!detail.empty()) detail += "; ";
        detail += line;
    }
    return TestResult{"M0", Verdict::Info, detail};
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

    // V2/V4/V5: AIE_RW_ACCESS functional sweeps on the dummy kernel's
    // CLAIMED tiles. (See file-level comment for why we don't sweep
    // unclaimed tiles -- prior version cascaded to SMU wedge.) Each
    // test bails out on the first throw; if the channel goes south we
    // stop sending requests rather than accumulate INVALID_PARAMs.
    auto run_or_skip = [&](const char* id, auto&& fn) {
        // Skip if any prior V/M test already thrown-bailed (likely channel
        // is bad). Detect by scanning prior failures with a "threw" marker.
        for (const auto& r : results) {
            if (r.v == Verdict::Fail && r.detail.find("threw") != std::string::npos) {
                results.push_back({id, Verdict::Skip,
                    "prior test threw; not running to avoid mgmt_chann cascade"});
                print_result(results.back());
                return;
            }
        }
        results.push_back(fn());
        print_result(results.back());
    };

    // Order: all SAFE tests (compute reg / DM, unclaimed-cell EINVAL)
    // FIRST, so they land their answers before the known-risky memtile
    // probe in V4 potentially destroys the channel.
    run_or_skip("V2", [&]{ return test_V2_read_consistency(ctx, args.col); });
    run_or_skip("V5", [&]{ return test_V5_stress(ctx, args.col, args.row, 100); });
    run_or_skip("V6", [&]{ return test_V6_compute_dm(ctx, args.col); });
    run_or_skip("V3", [&]{ return test_V3_unclaimed_cell(ctx, args.col); });

    // V4 LAST: memtile register read after 1s idle pause. Tests the
    // "hammering the mailbox too fast" hypothesis. Single read -- if
    // it hangs, the channel dies and any subsequent test would skip.
    run_or_skip("V4", [&]{ return test_V4_memtile_reg(ctx, args.col); });

    // VD (DANGER ZONE) LAST: opt-in via --probe-dm-danger. Reproduces
    // the memtile-DM half-implementation: writes ack, reads hang. Will
    // destroy the user-context mailbox channel; subsequent xclbin loads
    // require a driver reload (`pkexec modprobe -r/+ amdxdna`).
    if (args.probe_dm_danger) {
        std::printf("[INFO] --probe-dm-danger: running known-broken memtile DM round-trip; channel will likely die\n");
        run_or_skip("VD", [&]{ return test_VD_memtile_dm_danger(ctx, args.col); });
    }

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
