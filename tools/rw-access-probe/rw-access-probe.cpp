// rw-access-probe: smallest possible end-to-end validation of
// MSG_OP_AIE_RW_ACCESS via xrt::aie::device::read_aie_reg on Phoenix.
//
// Reads core Timer_Low (col=0, row=2, reg_addr=0x340F8) twice with a
// configurable sleep between, prints both readings and the delta. With
// the core actively running (NPU clock at ~400 MHz), the delta should
// correlate with wall time: 100 ms ≈ 40M cycles.
//
// Requires root or CAP_SYS_ADMIN -- the new XRT API path is gated on
// admin per xrt_aie.h ("** This function works only in Admin mode **").
//
// Background: validates the 2026-05-26 driver patch to npu1_fw_feature_table
// (adds AIE2_RW_ACCESS bit) by hitting the production XRT path end-to-end.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <sys/types.h>
#include <unistd.h>

#include "xrt/xrt_aie.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

namespace {

// Default xclbin: any compute-tile kernel works; we just need a live hw_context.
constexpr const char* DEFAULT_XCLBIN =
    "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/"
    "_diag_shim_chain_sweep/k8/chess/aie.xclbin";
constexpr int DEFAULT_COL = 0;
constexpr int DEFAULT_ROW = 2;
constexpr uint32_t REG_TIMER_LOW = 0x340F8;  // core Timer_Low
constexpr int DEFAULT_SLEEP_MS = 100;

struct Args {
    std::string xclbin = DEFAULT_XCLBIN;
    int col = DEFAULT_COL;
    int row = DEFAULT_ROW;
    uint32_t reg = REG_TIMER_LOW;
    int sleep_ms = DEFAULT_SLEEP_MS;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--xclbin" && i + 1 < argc) a.xclbin = argv[++i];
        else if (s == "--col" && i + 1 < argc) a.col = std::stoi(argv[++i]);
        else if (s == "--row" && i + 1 < argc) a.row = std::stoi(argv[++i]);
        else if (s == "--reg" && i + 1 < argc) a.reg = std::stoul(argv[++i], nullptr, 0);
        else if (s == "--sleep-ms" && i + 1 < argc) a.sleep_ms = std::stoi(argv[++i]);
    }
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    std::printf("== rw-access-probe ==\n");
    std::printf("xclbin   : %s\n", a.xclbin.c_str());
    std::printf("target   : col=%d row=%d reg=0x%05x sleep=%dms\n",
                a.col, a.row, a.reg, a.sleep_ms);

    xrt::device dev{0};
    xrt::aie::device aie_dev{dev};

    xrt::xclbin xclbin{a.xclbin};
    dev.register_xclbin(xclbin);
    xrt::hw_context ctx{dev, xclbin.get_uuid()};

    pid_t pid = getpid();
    // The XRT API "context_id" maps to amdxdna_hwctx.id (the per-client
    // userspace-facing ID populated into amdxdna_drm_hwctx_entry.context_id).
    // From txn-poll-probe runs that's 1 for the first/only context.
    uint16_t ctx_id = 1;
    std::printf("pid=%d  ctx_id (assumed)=%u\n", pid, ctx_id);

    try {
        uint32_t v1 = aie_dev.read_aie_reg(pid, ctx_id, static_cast<uint16_t>(a.col),
                                            static_cast<uint16_t>(a.row), a.reg);
        std::this_thread::sleep_for(std::chrono::milliseconds(a.sleep_ms));
        uint32_t v2 = aie_dev.read_aie_reg(pid, ctx_id, static_cast<uint16_t>(a.col),
                                            static_cast<uint16_t>(a.row), a.reg);
        uint32_t delta = v2 - v1;  // unsigned wrap is fine for 32-bit counter

        std::printf("\nv1    = 0x%08x  (%10u)\n", v1, v1);
        std::printf("v2    = 0x%08x  (%10u)\n", v2, v2);
        std::printf("delta = 0x%08x  (%10u cycles)\n", delta, delta);
        std::printf("ratio = %.0f cycles/ms\n",
                    static_cast<double>(delta) / a.sleep_ms);

        if (v1 == 0 && v2 == 0) {
            std::printf("\nVERDICT: zero readings — read_aie_reg call may have silently failed\n");
            return 2;
        }
        if (delta == 0) {
            std::printf("\nVERDICT: counter did not advance — core may not be clocked\n");
            return 3;
        }
        std::printf("\nVERDICT: cycle counter advanced (delta > 0). Feature path live.\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nEXCEPTION: %s\n", e.what());
        return 1;
    }
}
