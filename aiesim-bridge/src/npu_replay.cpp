#include "npu_replay.h"

#include <systemc.h>

#include <cstdio>

#include "addr_remap.h"
#include "ps_bridge.h"

namespace aiesim {

namespace {

// Wire-format tags -- MUST match crates/xdna-emu-ffi/src/aiesim/backend.rs
// mod npu_tag exactly. Field layouts (little-endian, one tagged record per op):
enum NpuTag : uint8_t {
    WRITE32 = 1,      // [reg_off u32][val u32]
    BLOCK_WRITE = 2,  // [reg_off u32][count u32][vals u32...]
    MASK_WRITE = 3,   // [reg_off u32][val u32][mask u32]
    MASK_POLL = 4,    // [reg_off u32][val u32][mask u32]
    DDR_PATCH = 5,    // [reg_addr u32][arg_idx u8][arg_plus u32]
    SYNC = 6,         // [channel u8][column u8][direction u8][col_num u8][row u8][row_num u8]
};

// Sync (dma_await_task) "channel done" mask for the AIE-ML DMA channel status
// register, derived from aie-rt _XAieMl_DmaWaitForDone (xaie_dma_aieml.c) + the
// xaiemlgbl register layout: TaskQSize[22:20] | ChannelRunning[19] |
// StalledTCT[5] | StalledStreamStarve[4] | StalledLockRel[3] | StalledLockAcq[2].
// Done == all of these clear (queue empty, not running, not stalled). The
// TaskQSize bits also defeat the false-idle race a queued-but-not-started task
// would otherwise cause (no need for the interpreter's "started" flag).
constexpr uint32_t kDmaDoneMask = 0x007000FCu;

// Poll bound for Sync, like cdo_replay's MASK_POLL: advance the kernel in quanta
// so a never-completing DMA fails instead of hanging. The cluster's DMA runs in
// real sim-time, so this only ever trips on a genuine stall.
constexpr uint64_t kPollQuantumNs = 256;
constexpr uint64_t kPollMaxNs = 5'000'000;

// AIE-ML DMA channel STATUS register offset within a tile, by tile-type (from
// the NPU1 row), direction (0=S2MM, 1=MM2S) and channel. Offsets per the
// xaiemlgbl params header; per-channel stride is 4 in every block.
//   shim/NOC (row 0):     S2MM 0x1D220, MM2S 0x1D228   (2+2 ch)
//   memtile (row 1):      S2MM 0xA0660, MM2S 0xA0680   (6+6 ch)
//   compute (row >= 2):   S2MM 0x1DF00, MM2S 0x1DF10   (2+2 ch)
uint32_t dma_status_offset(uint8_t npu1_row, uint8_t direction, uint8_t channel) {
    uint32_t s2mm_base, mm2s_base;
    if (npu1_row == 0) {
        s2mm_base = 0x1D220;
        mm2s_base = 0x1D228;
    } else if (npu1_row == 1) {
        s2mm_base = 0xA0660;
        mm2s_base = 0xA0680;
    } else {
        s2mm_base = 0x1DF00;
        mm2s_base = 0x1DF10;
    }
    const uint32_t base = (direction == 0) ? s2mm_base : mm2s_base;
    return base + static_cast<uint32_t>(channel) * 4u;
}

// Little-endian cursor with bounds checking (twin of cdo_replay's Reader).
struct Reader {
    const uint8_t* p;
    std::size_t n;
    std::size_t i = 0;
    bool err = false;

    bool need(std::size_t k) {
        if (i + k > n) { err = true; return false; }
        return true;
    }
    uint8_t u8() { return need(1) ? p[i++] : 0; }
    uint32_t u32() {
        if (!need(4)) return 0;
        uint32_t v = uint32_t(p[i]) | (uint32_t(p[i + 1]) << 8) |
                     (uint32_t(p[i + 2]) << 16) | (uint32_t(p[i + 3]) << 24);
        i += 4;
        return v;
    }
};

// Poll (reg & kDmaDoneMask) == 0 -- the DMA channel is idle, queue-empty, not
// stalled. Advances sim time between checks so the cluster's DMA/cores progress.
bool dma_wait(ps_bridge* ps, uint64_t status_addr) {
    uint64_t elapsed = 0;
    for (;;) {
        if ((ps->read32(status_addr) & kDmaDoneMask) == 0) return true;
        if (elapsed >= kPollMaxNs) return false;
        sc_core::wait(sc_core::sc_time(double(kPollQuantumNs), sc_core::SC_NS));
        elapsed += kPollQuantumNs;
    }
}

}  // namespace

int npu_replay(ps_bridge* ps, const uint8_t* ops, std::size_t len, uint8_t start_col,
               const std::vector<std::pair<uint64_t, std::size_t>>& host_buffers) {
    Reader r{ops, len};
    while (r.i < r.n && !r.err) {
        const uint8_t tag = r.u8();
        switch (tag) {
            case WRITE32: {
                uint32_t a = r.u32(), v = r.u32();
                if (r.err) break;
                // Register and data-memory writes both go through the config
                // MMIO socket; the cluster decodes the tile address internally
                // (no DM-vs-register split needed on our side).
                ps->write32(cluster_addr(a, start_col), v);
                break;
            }
            case BLOCK_WRITE: {
                uint32_t a = r.u32(), count = r.u32();
                if (r.err || !r.need(std::size_t(count) * 4)) { r.err = true; break; }
                const uint64_t base = cluster_addr(a, start_col);
                for (uint32_t k = 0; k < count; ++k) {
                    uint32_t w = r.u32();
                    ps->write32(base + k * 4u, w);
                }
                break;
            }
            case MASK_WRITE: {
                uint32_t a = r.u32(), v = r.u32(), m = r.u32();  // NPU order: val, mask
                if (r.err) break;
                const uint64_t ca = cluster_addr(a, start_col);
                uint32_t cur = ps->read32(ca);
                ps->write32(ca, (cur & ~m) | (v & m));
                break;
            }
            case MASK_POLL: {
                uint32_t a = r.u32(), v = r.u32(), m = r.u32();  // NPU order: val, mask
                if (r.err) break;
                const uint64_t ca = cluster_addr(a, start_col);
                uint64_t elapsed = 0;
                bool ok = false;
                for (;;) {
                    if ((ps->read32(ca) & m) == v) { ok = true; break; }
                    if (elapsed >= kPollMaxNs) break;
                    sc_core::wait(sc_core::sc_time(double(kPollQuantumNs), sc_core::SC_NS));
                    elapsed += kPollQuantumNs;
                }
                if (!ok) {
                    std::fprintf(stderr,
                                 "[npu_replay] MASK_POLL timeout: reg=0x%x val=0x%x mask=0x%x\n",
                                 a, v, m);
                    return 1;
                }
                break;
            }
            case DDR_PATCH: {
                uint32_t reg_addr = r.u32();
                uint8_t arg_idx = r.u8();
                uint32_t arg_plus = r.u32();
                if (r.err) break;
                if (arg_idx >= host_buffers.size()) {
                    std::fprintf(stderr,
                                 "[npu_replay] DdrPatch arg_idx %u >= %zu host buffers\n",
                                 arg_idx, host_buffers.size());
                    return 1;
                }
                // Host-buffer mode (host buffers are registered by the plugin):
                // the patched DDR address is the buffer base + the BD's byte
                // offset. Write it into the shim BD address word pair (low 32 to
                // the addr-low word, high 16 to the addr-high word). The DDR
                // address is in the bridge's ddr_target space -- NOT translated.
                const uint64_t patched = host_buffers[arg_idx].first + arg_plus;
                const uint64_t lo_reg = cluster_addr(reg_addr, start_col);
                ps->write32(lo_reg, uint32_t(patched & 0xFFFFFFFFu));

                const uint8_t row = uint8_t((reg_addr >> kRowShift) & 0x1F);
                const uint32_t hi16 = uint32_t((patched >> 32) & 0xFFFFu);
                if (row == 0) {
                    // Shim BD addr-high word shares bits[31:16] with packet /
                    // out-of-order-id fields; RMW to preserve them (aie-rt
                    // _XAieMl_BdSetAddr uses XAie_MaskWrite32 here).
                    uint32_t cur = ps->read32(lo_reg + 4);
                    ps->write32(lo_reg + 4, (cur & 0xFFFF0000u) | hi16);
                } else {
                    ps->write32(lo_reg + 4, hi16);
                }
                break;
            }
            case SYNC: {
                uint8_t channel = r.u8();
                uint8_t column = r.u8();
                uint8_t direction = r.u8();
                (void)r.u8();  // column_num (partition extent, unused)
                uint8_t row = r.u8();
                (void)r.u8();  // row_num (partition extent, unused)
                if (r.err) break;
                // Build the NPU1 status-register address (col/row/offset) and
                // translate to the cluster. The cluster's DMA updates this status
                // as it runs; dma_wait advances sim time until the channel is done.
                const uint32_t off = dma_status_offset(row, direction, channel);
                const uint32_t npu_addr = (uint32_t(column) << kColShift) |
                                          (uint32_t(row) << kRowShift) | off;
                const uint64_t status_addr = cluster_addr(npu_addr, start_col);
                if (!dma_wait(ps, status_addr)) {
                    std::fprintf(stderr,
                                 "[npu_replay] Sync timeout: col=%u row=%u dir=%u ch=%u "
                                 "status=0x%llx\n",
                                 column, row, direction, channel,
                                 (unsigned long long)status_addr);
                    return 1;
                }
                break;
            }
            default:
                std::fprintf(stderr, "[npu_replay] unknown tag %u at offset %zu -- drift\n",
                             tag, r.i - 1);
                return 1;
        }
    }
    if (r.err) {
        std::fprintf(stderr, "[npu_replay] truncated op-stream (len=%zu)\n", len);
        return 1;
    }
    return 0;
}

}  // namespace aiesim
