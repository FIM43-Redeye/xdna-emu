// rw-access-probe: smallest possible end-to-end validation of
// MSG_OP_AIE_RW_ACCESS via xrt::aie::device::read_aie_reg on Phoenix.
//
// Reads a tile register N times (default 2) with a sleep between, then
// prints each reading and (if num-reads >= 2) the v[last]-v[0] delta.
//
// Default target is core Timer_Low (col=0, row=2, addr=0x340F8). Use
// --col/--row/--reg to probe other tile types -- per the AM025 regdb,
// the tile-local Timer_Low offset differs per module:
//   core (compute row>=2)        : 0x340F8
//   memory (compute mem-side)    : 0x140F8
//   memory_tile (memtile row=1)  : 0x940F8
//   shim (row=0)                 : 0x340F8
// The Phoenix FW handler for opcode 0x203 does no row/col validation:
// supplying a tile-local offset that doesn't decode at the target tile
// hangs the AXI fabric, blocking the FW until TDR fires.
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
    int num_reads = 2;
    std::string label;     // optional, prepended to output
    std::string csv_path;  // if set, write per-read rows here and suppress per-read stdout
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
        else if (s == "--num-reads" && i + 1 < argc) a.num_reads = std::stoi(argv[++i]);
        else if (s == "--label" && i + 1 < argc) a.label = argv[++i];
        else if (s == "--csv" && i + 1 < argc) a.csv_path = argv[++i];
    }
    if (a.num_reads < 1) a.num_reads = 1;
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    std::printf("== rw-access-probe ==\n");
    if (!a.label.empty()) std::printf("label    : %s\n", a.label.c_str());
    std::printf("xclbin   : %s\n", a.xclbin.c_str());
    std::printf("target   : col=%d row=%d reg=0x%05x num_reads=%d sleep=%dms\n",
                a.col, a.row, a.reg, a.num_reads, a.sleep_ms);

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
        std::vector<uint32_t> vals;
        vals.reserve(a.num_reads);
        std::FILE* csv = nullptr;
        if (!a.csv_path.empty()) {
            csv = std::fopen(a.csv_path.c_str(), "w");
            if (!csv) {
                std::fprintf(stderr, "EXCEPTION: cannot open --csv path '%s'\n", a.csv_path.c_str());
                return 1;
            }
            std::fprintf(csv, "index,timestamp_ns,roundtrip_us,value\n");
        }
        auto t_origin = std::chrono::steady_clock::now();
        for (int i = 0; i < a.num_reads; ++i) {
            if (i > 0) std::this_thread::sleep_for(std::chrono::milliseconds(a.sleep_ms));
            auto t0 = std::chrono::steady_clock::now();
            uint32_t v = aie_dev.read_aie_reg(pid, ctx_id,
                                              static_cast<uint16_t>(a.col),
                                              static_cast<uint16_t>(a.row), a.reg);
            auto t1 = std::chrono::steady_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            int64_t t_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t0 - t_origin).count();
            if (csv) {
                std::fprintf(csv, "%d,%lld,%.3f,%u\n", i, static_cast<long long>(t_ns), us, v);
            } else {
                std::printf("v[%d]  = 0x%08x  (%10u)   roundtrip=%.1fus\n", i, v, v, us);
            }
            vals.push_back(v);
        }
        if (csv) {
            std::fclose(csv);
            std::printf("wrote %d rows to %s\n", a.num_reads, a.csv_path.c_str());
        }

        if (vals.size() >= 2) {
            uint32_t delta = vals.back() - vals.front();
            std::printf("delta = 0x%08x  (%10u)   over %dms sleep\n",
                        delta, delta, (a.num_reads - 1) * a.sleep_ms);
        }

        bool any_nonzero = false;
        for (auto v : vals) if (v != 0) { any_nonzero = true; break; }
        if (!any_nonzero) {
            std::printf("\nVERDICT: all readings zero — call may have silently failed\n");
            return 2;
        }
        std::printf("\nVERDICT: read returned non-zero. Feature path live for this (col,row,reg).\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nEXCEPTION: %s\n", e.what());
        return 1;
    }
}
