#include "npu_replay.h"

#include <systemc.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

// Sync (dma_await_task) completion detection for the AIE-ML DMA channel status
// register. The runtime-sequence Sync is a Task-Completion-Token wait, which the
// interpreter models in is_sync_satisfied (src/npu/executor.rs) by watching the
// Channel_Running bit edge -- not an aie-rt-style "wait for done" poll. The
// decisive signal is Channel_Running (AM025 DMA_*_Status_0[19]) going 1 then 0
// (started-then-idle); a channel already idle with an empty Task_Queue at the
// first poll is fast completion. Task_Queue_Size[22:20] guards the
// never-started false-idle race.
//
// The stall flags (StalledTCT[5], StalledLockRel[3], StalledStreamStarve[4],
// StalledLockAcq[2]) are deliberately EXCLUDED from the done decision: a channel
// that finished its transfer but parks in a TCT/lock stall (the completion-token
// FIFO the bridge never drains) is data-done. Including them -- the old
// kDmaDoneMask=0x0078003C, modeled on aie-rt's poll-style DmaWaitForDone -- hung
// TCT-heavy (sync_task_complete_token) and BD-reuse (shim_dma_bd_reuse) kernels.
constexpr uint32_t kChannelRunningBit = 0x00080000u;  // Channel_Running[19]
constexpr uint32_t kTaskQueueSizeMask = 0x00700000u;  // Task_Queue_Size[22:20]

// Poll bound for Sync, like cdo_replay's MASK_POLL: advance the kernel in quanta
// so a never-completing DMA fails instead of hanging. The cluster's DMA runs in
// real sim-time, so this only ever trips on a genuine stall.
// The cluster sim advances ~1 sim-ns per ~1 ms wall, so the cap doubles as a
// wall-clock bound: 100k ns ~ 2 min before a stuck (starving) channel gives up.
// A real small DMA completes in a few thousand sim-ns. Tunable.
constexpr uint64_t kPollQuantumNs = 256;
// Default cap; XDNA_AIESIM_POLL_MAX_NS overrides (e.g. a small value for a
// fast-fail timeout probe when debugging a known stall).
inline uint64_t poll_max_ns() {
    if (const char* e = std::getenv("XDNA_AIESIM_POLL_MAX_NS")) {
        const uint64_t v = std::strtoull(e, nullptr, 0);
        if (v) return v;
    }
    return 100'000;
}

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

// Resolve a DdrPatch arg_idx to its DDR base address. For arg_idx within the
// registered host buffers, that buffer's address; beyond it, fabricate a 1 MiB
// trace-style buffer per extra index, placed after the last buffer -- mirroring
// the interpreter (executor.rs execute_ddr_patch), so an over-indexed BD (trace
// / instrumentation) gets a valid, non-overlapping DDR address instead of an
// error. The bridge's ddr_target is sparse-paged, so any such address is live.
uint64_t resolve_arg_base(const std::vector<std::pair<uint64_t, std::size_t>>& bufs,
                          uint8_t arg_idx) {
    if (arg_idx < bufs.size()) return bufs[arg_idx].first;
    constexpr uint64_t kTrace = 0x100000;  // 1 MiB, matching the interpreter
    uint64_t addr = bufs.empty() ? kTrace : (bufs.back().first + bufs.back().second);
    for (std::size_t idx = bufs.size(); idx < arg_idx; ++idx) addr += kTrace;
    return addr;
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

// Live sim-vs-wall ratio probe. Set XDNA_AIESIM_RATE_LOG=1 to print, every
// ~kRateEvery quanta, the absolute SystemC sim time (us) and the wall time (s)
// since the first poll -- so the cycle-accurate slowdown is directly visible.
void rate_tick(const char* where, uint64_t poll_elapsed_ns) {
    static const bool on = std::getenv("XDNA_AIESIM_RATE_LOG") != nullptr;
    if (!on) return;
    static const auto wall0 = std::chrono::steady_clock::now();
    static uint64_t n = 0;
    constexpr uint64_t kRateEvery = 16;  // 16 * 256ns = ~4us sim between prints
    if (n++ % kRateEvery != 0) return;
    const double sim_us = sc_core::sc_time_stamp().to_seconds() * 1e6;
    const double wall_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
    std::fprintf(stderr,
                 "[rate] %s sim=%.3f us wall=%.1f s  (this-wait poll=%.3f us)  "
                 "ratio=%.1f ms_wall/sim_us\n",
                 where, sim_us, wall_s, poll_elapsed_ns / 1000.0,
                 wall_s > 0 && sim_us > 0 ? (wall_s * 1000.0) / sim_us : 0.0);
}

// Mirror the interpreter's is_sync_satisfied: a sync is satisfied once
// Channel_Running has been observed high then low (started-then-idle), or the
// channel is already idle with an empty task queue at first poll (fast
// completion). Stall flags are excluded -- see kChannelRunningBit above.
// Advances sim time in quanta so a never-completing channel fails (returns
// false) instead of hanging; the cluster's DMA runs in real sim-time, so this
// only trips on a genuine stall.
bool dma_wait(ps_bridge* ps, uint64_t status_addr) {
    uint64_t elapsed = 0;
    bool started = false;
    for (;;) {
        const uint32_t st = ps->read32(status_addr);
        if (st & kChannelRunningBit) {
            started = true;  // running -- keep waiting for it to drain
        } else if (started || (st & kTaskQueueSizeMask) == 0) {
            return true;     // started-then-idle, or already-drained
        }
        if (elapsed >= poll_max_ns()) return false;
        sc_core::wait(sc_core::sc_time(double(kPollQuantumNs), sc_core::SC_NS));
        elapsed += kPollQuantumNs;
        rate_tick("dma_wait", elapsed);
    }
}

// Diagnostic: when XDNA_AIESIM_PROBE_TILE="col,row" (NPU1 logical) is set, dump
// the compute tile's core/clock/program/data state via timed live reads. Lets us
// tell "core never enabled" (clock-gated / no program loaded) from "core enabled
// but stalled" (input stream/lock starvation) -- the two failure modes that both
// leave the output S2MM channel never completing. Register offsets are the
// AIE-ML compute-tile module (xaiemlgbl_params.h):
//   Core_Control 0x32000 (Enable[0], Reset[1])
//   Core_Status  0x32004 (Enable[0], Done[20], stream-stall SS0[10], lock-stall..)
//   Core_PC      0x31100   Program_Memory 0x20000   Module_Clock_Control 0x60000
// `tag` labels the call site (entry / sync / timeout).
void probe_tile(ps_bridge* ps, uint8_t start_col, const char* tag) {
    const char* e = std::getenv("XDNA_AIESIM_PROBE_TILE");
    if (!e) return;
    int col = 0, row = 0;
    if (std::sscanf(e, "%d,%d", &col, &row) != 2) return;
    // Read NPU1 logical (col,r,off) through the cluster remap.
    auto rd = [&](int r, uint32_t off) -> uint32_t {
        const uint32_t npu = (uint32_t(col) << kColShift) | (uint32_t(r) << kRowShift) | off;
        return ps->read32(cluster_addr(npu, start_col));
    };
    // Compute-tile core + clock + program state.
    const uint32_t ctrl = rd(row, 0x32000), status = rd(row, 0x32004), pc = rd(row, 0x31100);
    const uint32_t clk = rd(row, 0x60000), pm0 = rd(row, 0x20000);
    std::fprintf(stderr,
                 "[probe %s] core(%d,%d) ctrl=0x%08x status=0x%08x pc=0x%05x clk=0x%08x pm0=0x%08x\n",
                 tag, col, row, ctrl, status, pc, clk, pm0);
    // TILE_CONTROL_PACKET_HANDLER_STATUS (0x3FF30, CORE_MODULE; xaiemlgbl_params.h):
    // bit0 first-header parity err, bit1 second-header parity err, bit2 slverr-on-
    // access, bit3 TLAST err. Nonzero => the compute tile's control-packet handler
    // saw a malformed REQUEST (our framing bug) -- the our-error check for why the
    // read response is unframed. Reads via the model's faithful register path.
    const uint32_t cph = rd(row, 0x3FF30);
    std::fprintf(stderr,
                 "[probe %s] ctrl_pkt_handler_status(%d,%d)=0x%08x  [hdr1_parity=%u hdr2_parity=%u slverr=%u tlast=%u]\n",
                 tag, col, row, cph, cph & 1u, (cph >> 1) & 1u, (cph >> 2) & 1u, (cph >> 3) & 1u);
    // Compute-tile memory-module locks 0..7 (0x1F000, stride 0x10). Signed value.
    std::fprintf(stderr, "[probe %s] locks ", tag);
    for (int l = 0; l < 8; ++l) std::fprintf(stderr, "L%d=%d ", l, (int)rd(row, 0x1F000 + l * 0x10));
    std::fprintf(stderr, "\n");
    // Compute-tile DMA channel status: S2MM (receives input from stream) 0x1DF00,
    // MM2S (sends output to stream) 0x1DF10.
    std::fprintf(stderr, "[probe %s] compute-dma s2mm0=0x%08x mm2s0=0x%08x\n",
                 tag, rd(row, 0x1DF00), rd(row, 0x1DF10));
    // Shim (row 0) DMA channel status: MM2S (input DDR->stream) 0x1D228,
    // S2MM (output stream->DDR) 0x1D220. Plus the input BD0 (len 0x1D000,
    // addr-lo 0x1D004, addr-hi 0x1D008) and MM2S ch0 ctrl 0x1D210 / task-queue
    // 0x1D214 -- to confirm the input descriptor is configured + queued.
    std::fprintf(stderr,
                 "[probe %s] shim-dma mm2s0=0x%08x s2mm0=0x%08x | bd0[len,lo,hi]=0x%08x,0x%08x,0x%08x "
                 "mm2s0_ctrl=0x%08x mm2s0_q=0x%08x\n",
                 tag, rd(0, 0x1D228), rd(0, 0x1D220), rd(0, 0x1D000), rd(0, 0x1D004), rd(0, 0x1D008),
                 rd(0, 0x1D210), rd(0, 0x1D214));
    // Did ANY beats land at shim S2MM ch0 (@ctrl0, the control-response sink)?
    // write_count 0x1DF18 / finish_tlast_fifo 0x1DF20 per the device DMA def. Also
    // the NoC-interface mux (0x1F000) / demux (0x1F004) config that steers SS south
    // streams to DMA channels by stream-id. write_count>0 => response arrived but
    // not drained (BD/consume issue); ==0 => never generated or never routed.
    std::fprintf(stderr,
                 "[probe %s] shim s2mm-ch0 write_count=0x%08x finish_tlast=0x%08x | mux=0x%08x demux=0x%08x\n",
                 tag, rd(0, 0x1DF18), rd(0, 0x1DF20), rd(0, 0x1F000), rd(0, 0x1F004));
    // Compute-tile data-memory scan -- look for the input pattern (0,1,2,3,...).
    // 64 KiB data memory; sample a few plausible buffer offsets.
    static const uint32_t kOff[] = {0x0000, 0x0400, 0x0800, 0x1000, 0x1400, 0x1800, 0x2000, 0x4000};
    std::fprintf(stderr, "[probe %s] dm", tag);
    for (uint32_t o : kOff) std::fprintf(stderr, " [0x%04x]=%u,%u", o, rd(row, o), rd(row, o + 4));
    std::fprintf(stderr, "\n");
    // Memtile (NPU1 row 1) -- the middle of the shim->memtile->compute pipeline.
    // If its S2MM (from shim) is done but MM2S (to compute) is running+stalled,
    // the downstream route to the compute tile is broken (the Fork A row-gap).
    // Memtile DMA status: S2MM 0xA0660, MM2S 0xA0680 (stride 4). Locks 0xC0000.
    std::fprintf(stderr, "[probe %s] memtile(0,1) s2mm0=0x%08x s2mm1=0x%08x mm2s0=0x%08x mm2s1=0x%08x locks ",
                 tag, rd(1, 0xA0660), rd(1, 0xA0664), rd(1, 0xA0680), rd(1, 0xA0684));
    for (int l = 0; l < 4; ++l) std::fprintf(stderr, "L%d=%d ", l, (int)rd(1, 0xC0000 + l * 0x10));
    std::fprintf(stderr, "\n");

    // Stream-switch master-config readback (XDNA_AIESIM_PROBE_SS): confirm whether
    // the packet-mode config the CDO programmed actually stuck in the cluster's SS
    // registers. Master config: bit31 = port enable, bit30 = packet-stream enable.
    // If a packet kernel's masters read back 0xC0000000-flavored, the config landed
    // (so a non-routing hang is a cluster-model gap, not a lost write); if they read
    // 0x8.../0x0, the write path dropped the packet bit. Offsets per AM025:
    //   compute/shim SS master base 0x3F000; memtile SS master base 0xB0000 (stride 4).
    if (std::getenv("XDNA_AIESIM_PROBE_SS")) {
        auto dump_ss = [&](int r, uint32_t base, int count, const char* label) {
            std::fprintf(stderr, "[probe %s] ss-master %s:", tag, label);
            int en = 0, pkt = 0;
            for (int i = 0; i < count; ++i) {
                uint32_t v = rd(r, base + uint32_t(i) * 4);
                if (v & 0x80000000u) ++en;
                if (v & 0x40000000u) ++pkt;
                if (v & 0xC0000000u)  // only print configured ports
                    std::fprintf(stderr, " m%d=0x%08x%s", i, v, (v & 0x40000000u) ? "(PKT)" : "");
            }
            std::fprintf(stderr, "  [%d enabled, %d packet]\n", en, pkt);
        };
        dump_ss(row, 0x3F000, 23, "compute");
        dump_ss(1, 0xB0000, 17, "memtile");
        dump_ss(0, 0x3F000, 22, "shim");

        // Slave SLOT config (compute, base 0x3F200): the packet-ID matcher. A
        // configured slot has Enable[8] with id[28:24]/mask[20:16]/msel[5:4]/
        // arbit[2:0]. If packet masters are enabled but NO slots are configured,
        // incoming packet IDs match nothing -> dropped (the routing gap).
        std::fprintf(stderr, "[probe %s] compute slots:", tag);
        int slots = 0;
        for (int i = 0; i < 128; ++i) {
            uint32_t v = rd(row, 0x3F200 + uint32_t(i) * 4);
            if (v & 0x100u) {  // Enable
                ++slots;
                std::fprintf(stderr, " s%d=0x%08x(id=%u msel=%u arb=%u)", i, v,
                             (v >> 24) & 0x1F, (v >> 4) & 0x3, v & 0x7);
            }
        }
        std::fprintf(stderr, "  [%d configured]\n", slots);

        // Deterministic-merge / arbiter (compute, base 0x3F800): the Versal-style
        // arbiter init that NPU1 CDOs may omit. If these are all zero while packet
        // masters are enabled, a missing arbiter init is the likely routing gap.
        std::fprintf(stderr, "[probe %s] compute det-merge:", tag);
        int dm = 0;
        for (int i = 0; i < 24; ++i) {
            uint32_t v = rd(row, 0x3F800 + uint32_t(i) * 4);
            if (v) { ++dm; std::fprintf(stderr, " [0x%03x]=0x%08x", 0x800 + i * 4, v); }
        }
        std::fprintf(stderr, "  [%d nonzero]\n", dm);

        // Shim (row 0) slave SLOT config: the demux matcher that decides whether a
        // control-response packet (id=2) routes to the shim DMA0 S2MM master. Flow
        // 0x3 (id=3 -> DMA1) is known-good, so a missing id=2 slot here would be the
        // control-read-response gap. Same register layout as compute (base 0x3F200).
        std::fprintf(stderr, "[probe %s] shim slots:", tag);
        int sslots = 0;
        for (int i = 0; i < 128; ++i) {
            uint32_t v = rd(0, 0x3F200 + uint32_t(i) * 4);
            if (v & 0x100u) {
                ++sslots;
                std::fprintf(stderr, " s%d=0x%08x(id=%u msel=%u arb=%u)", i, v,
                             (v >> 24) & 0x1F, (v >> 4) & 0x3, v & 0x7);
            }
        }
        std::fprintf(stderr, "  [%d configured]\n", sslots);
    }
}

}  // namespace

int npu_replay(ps_bridge* ps, const uint8_t* ops, std::size_t len, uint8_t start_col,
               const std::vector<std::pair<uint64_t, std::size_t>>& host_buffers) {
    Reader r{ops, len};
    probe_tile(ps, start_col, "entry");
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
                    if (elapsed >= poll_max_ns()) break;
                    sc_core::wait(sc_core::sc_time(double(kPollQuantumNs), sc_core::SC_NS));
                    elapsed += kPollQuantumNs;
                    rate_tick("mask_poll", elapsed);
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
                // Host-buffer mode (host buffers are registered by the plugin):
                // the patched DDR address is the buffer base + the BD's byte
                // offset. Write it into the shim BD address word pair (low 32 to
                // the addr-low word, high 16 to the addr-high word). The DDR
                // address is in the bridge's ddr_target space -- NOT translated.
                const uint64_t patched = resolve_arg_base(host_buffers, arg_idx) + arg_plus;
                const uint64_t lo_reg = cluster_addr(reg_addr, start_col);
                if (std::getenv("XDNA_AIESIM_TRACE")) {
                    std::fprintf(stderr,
                                 "[npu_replay] DdrPatch arg_idx=%u patched=0x%llx -> reg=0x%llx\n",
                                 arg_idx, (unsigned long long)patched, (unsigned long long)lo_reg);
                }
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
                if (std::getenv("XDNA_AIESIM_TRACE")) {
                    uint32_t first = ps->read32(status_addr);
                    std::fprintf(stderr,
                                 "[npu_replay] Sync col=%u row=%u dir=%u ch=%u status=0x%llx "
                                 "first_read=0x%08x (running&=0x%x qsize&=0x%x)\n",
                                 column, row, direction, channel,
                                 (unsigned long long)status_addr, first,
                                 first & kChannelRunningBit, first & kTaskQueueSizeMask);
                }
                probe_tile(ps, start_col, "sync");
                if (!dma_wait(ps, status_addr)) {
                    std::fprintf(stderr,
                                 "[npu_replay] Sync timeout: col=%u row=%u dir=%u ch=%u "
                                 "status=0x%llx\n",
                                 column, row, direction, channel,
                                 (unsigned long long)status_addr);
                    probe_tile(ps, start_col, "timeout");
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
