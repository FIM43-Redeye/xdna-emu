#include "npu_replay.h"

#include <systemc.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "addr_remap.h"
#include "ddr_target.h"
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
// Task_Queue_Size MSB (bit 22 on aie-ml). aie-rt's _XAieMl_DmaWaitForBdTaskQueue
// polls this bit before pushing a BD to the start queue: set => the depth-4
// (StartQSizeMax) queue is full. We replicate that pacing before task-queue
// pushes (see WRITE32) so we don't overrun the cluster model's queue.
constexpr uint32_t kTaskQueueFullBit = 0x00400000u;

// Sync advances the kernel in quanta so the cluster's DMA/cores make concurrent
// progress while we poll. Completion is decided by quiescence, not a fixed
// sim-time cap (see dma_wait): a single long BD (e.g. shim_dma_bd_reuse's 5120-
// word 2D-wrap recv) keeps Channel_Running=1 for its WHOLE duration, so a fixed
// cap can't tell "slow but progressing" from "wedged" -- it just cuts slow
// transfers off mid-flight. Instead we watch the shim-DMA transaction counter
// (ddr_target::dma_txn_count, the same "DMAs drained" signal the run() early-exit
// uses): as long as it advances, the transfer is live; only when it stalls for a
// settle window with the channel still not done is the DMA genuinely wedged.
constexpr uint64_t kPollQuantumNs = 256;

// Settle window: quanta of no shim-DMA transaction progress before a still-
// running channel is declared wedged. Must exceed the longest legitimate gap
// between transactions during a slow transfer (the core passthrough's lock
// ping-pong between consecutive output bursts). XDNA_AIESIM_SETTLE_QUANTA
// overrides. Default 512 (~131k sim-ns) is generous: a true wedge costs that
// much extra sim-time before giving up, but a live transfer never false-trips.
inline uint64_t settle_quanta() {
    if (const char* e = std::getenv("XDNA_AIESIM_SETTLE_QUANTA")) {
        const uint64_t v = std::strtoull(e, nullptr, 0);
        if (v) return v;
    }
    return 512;
}

// Hard safety backstop on total Sync sim-time, so a pathological channel that
// dribbles one transaction just inside every settle window can't loop forever.
// Far above any real transfer; XDNA_AIESIM_POLL_MAX_NS overrides (e.g. a small
// value for a fast-fail timeout probe when debugging a known stall). 0 from the
// env means "no fixed cap -- rely on quiescence alone".
inline uint64_t poll_max_ns() {
    if (const char* e = std::getenv("XDNA_AIESIM_POLL_MAX_NS")) {
        return std::strtoull(e, nullptr, 0);  // honor 0 as "no cap"
    }
    return 50'000'000;  // ~50 ms sim-time backstop
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

// If a SHIM (row 0) tile-local register offset is a DMA task-queue PUSH register,
// return the matching channel-status register offset (for the Task_Queue_Size
// poll); otherwise 0. Shim task queues (per aie-rt / AM025): S2MM ch0/1 =
// 0x1D204/0x1D20C, MM2S ch0/1 = 0x1D214/0x1D21C; each maps to its channel status
// via dma_status_offset(0, dir, ch). Runtime-sequence dma_start_task pushes only
// ever target the shim, so this shim-only mapping covers the replayed op stream
// (memtile/compute DMAs are CDO-configured and self-looping, never host-pushed).
uint32_t shim_taskqueue_status_off(uint32_t off) {
    switch (off) {
        case 0x1D204: return dma_status_offset(0, 0, 0);  // S2MM ch0 -> 0x1D220
        case 0x1D20C: return dma_status_offset(0, 0, 1);  // S2MM ch1 -> 0x1D224
        case 0x1D214: return dma_status_offset(0, 1, 0);  // MM2S ch0 -> 0x1D228
        case 0x1D21C: return dma_status_offset(0, 1, 1);  // MM2S ch1 -> 0x1D22C
        default:      return 0;
    }
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

// Mirror the interpreter's is_sync_satisfied for SUCCESS, but use DMA quiescence
// (not a fixed sim-time cap) to detect a genuine wedge.
//
// SUCCESS: Channel_Running observed high then low (started-then-idle), or the
// channel already idle with an empty task queue at first poll (fast completion).
// Stall flags are excluded -- see kChannelRunningBit above.
//
// WEDGE: the shim-DMA transaction counter (ddr_target::dma_txn_count) has not
// advanced for a settle window while the channel is still not done. A single
// long BD holds Channel_Running=1 for its entire duration, so the status alone
// cannot distinguish "slow but progressing" from "stuck" -- only the transaction
// counter can. As long as the DMA keeps committing reads/writes to host DDR the
// transfer is live and we keep waiting, with no artificial deadline. (`ddr` may
// be null in degenerate setups; then we fall back to the sim-time backstop only.)
bool dma_wait(ps_bridge* ps, ddr_target* ddr, uint64_t status_addr) {
    uint64_t elapsed = 0;
    bool started = false;
    const bool trace = std::getenv("XDNA_AIESIM_TRACE") != nullptr;
    uint64_t since_log = 0;
    uint32_t prev_st = 0xFFFFFFFFu;
    uint64_t last_txns = ddr ? ddr->dma_txn_count() : 0;
    uint64_t idle_quanta = 0;
    for (;;) {
        const uint32_t st = ps->read32(status_addr);
        if (trace && (st != prev_st || since_log >= 16384)) {
            std::fprintf(stderr,
                         "[dma_wait@%lluns] status=0x%08x (running=%u curbd=%u qsize=%u state=%u) "
                         "txns=%llu idle=%llu\n",
                         (unsigned long long)elapsed, st,
                         (st & kChannelRunningBit) ? 1u : 0u, (st >> 24) & 0xFu,
                         (st >> 20) & 0x7u, st & 0x3u,
                         (unsigned long long)(ddr ? ddr->dma_txn_count() : 0),
                         (unsigned long long)idle_quanta);
            prev_st = st;
            since_log = 0;
        }
        if (st & kChannelRunningBit) {
            started = true;  // running -- keep waiting for it to drain
        } else if (started || (st & kTaskQueueSizeMask) == 0) {
            return true;     // started-then-idle, or already-drained
        }
        // Quiescence-based wedge detection: progress resets the idle counter;
        // a full settle window with no shim-DMA activity = genuinely stuck.
        if (ddr) {
            const uint64_t txns = ddr->dma_txn_count();
            if (txns != last_txns) {
                last_txns = txns;
                idle_quanta = 0;
            } else if (++idle_quanta >= settle_quanta()) {
                if (trace)
                    std::fprintf(stderr,
                                 "[dma_wait] quiescent wedge at %lluns: status=0x%08x txns=%llu\n",
                                 (unsigned long long)elapsed, st, (unsigned long long)txns);
                return false;
            }
        }
        // Hard sim-time backstop (0 = disabled, rely on quiescence alone).
        const uint64_t cap = poll_max_ns();
        if (cap && elapsed >= cap) return false;
        sc_core::wait(sc_core::sc_time(double(kPollQuantumNs), sc_core::SC_NS));
        elapsed += kPollQuantumNs;
        since_log += kPollQuantumNs;
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
    // Shim BD8 full descriptor (the 2D-wrap S2MM recv in shim_dma_bd_reuse). Base
    // 0x1D100, 8 words @ stride 4 (AM025 DMA_BD8_0..7). Decode: W0=Buffer_Length,
    // W3[29:20]=D0_Wrap W3[19:0]=D0_Step(-1), W4[29:20]=D1_Wrap W4[19:0]=D1_Step(-1),
    // W6[25:20]=Iter_Wrap(-1), W7 bit25=Valid bit26=UseNextBD [30:27]=NextBD. Lets
    // us see whether the cluster received a correct 2D descriptor or a mangled one.
    {
        uint32_t b[8];
        for (int i = 0; i < 8; ++i) b[i] = rd(0, 0x1D100 + uint32_t(i) * 4);
        std::fprintf(stderr,
                     "[probe %s] shim bd8 W0-7=0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x "
                     "| len=%u d0wrap=%u d0step=%u d1wrap=%u d1step=%u iterwrap=%u valid=%u usenext=%u next=%u\n",
                     tag, b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                     b[0], (b[3] >> 20) & 0x3FF, (b[3] & 0xFFFFF) + 1,
                     (b[4] >> 20) & 0x3FF, (b[4] & 0xFFFFF) + 1, ((b[6] >> 20) & 0x3F) + 1,
                     (b[7] >> 25) & 1u, (b[7] >> 26) & 1u, (b[7] >> 27) & 0xFu);
    }
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

// Post-config array settle: model the config->host-submit gap real hardware
// has for free. After LOAD_CDO enables the cores, on silicon they run until
// they reach their first synchronization point (a lock/stream stall) BEFORE
// the host runtime sequence is ever submitted. The bridge otherwise fires
// every EXEC_NPU register write at the same sim-instant, before the cores have
// advanced a cycle -- so a runtime write that races a core's pre-lock work
// lands in the wrong order. The canonical victim is add_maskwrite: its core
// stores 0x37373737 into input_buffer, then blocks on a lock; the runtime then
// maskwrites two of those words and releases the lock. On HW (and the
// interpreter) the maskwrites land on top of the stored values and survive
// into the core's read; with no settle the bridge applied them first, the
// core's store then clobbered them, and the masked result was lost.
//
// We restore the ordering by advancing sim-time until the cores quiesce.
// Quiescence signal: per-core program-counter (Core_PC 0x31100) stability. A
// core stalled on a lock/stream holds a frozen PC; a running core's PC
// advances. PC-stability is bit-layout-agnostic (no Core_Status stall-bit
// decoding to get wrong) and is the ground-truth "is this core making
// progress" signal. We warm up a few quanta so the cores get going, then wait
// for every enabled core's PC to hold steady for a short window, with a
// sim-time cap as the backstop. The settle is purely an ordering aid -- if a
// core never blocks before the host sequence, hitting the cap and proceeding
// is harmless (the cores keep running concurrently during the sequence, as
// before). XDNA_AIESIM_POSTCFG_SETTLE=0 disables it (A/B knob).
void settle_array(ps_bridge* ps, uint8_t start_col) {
    if (const char* e = std::getenv("XDNA_AIESIM_POSTCFG_SETTLE"))
        if (std::strcmp(e, "0") == 0) return;

    constexpr uint32_t kCoreCtrl = 0x32000;  // Core_Control (Enable[0])
    constexpr uint32_t kCorePc = 0x31100;    // Core_PC
    // NPU1 geometry: 5 columns, compute cores on rows 2..5. Over-scanning is
    // harmless -- a non-enabled tile reads Core_Control[0]=0 and is skipped, so
    // the enable check bounds us to the real cores. (Could be device-JSON-
    // derived later; the emulated device is fixed NPU1 today.)
    constexpr int kNumCols = 5, kFirstComputeRow = 2, kLastComputeRow = 5;

    auto rd = [&](int col, int row, uint32_t off) -> uint32_t {
        const uint32_t npu = (uint32_t(col) << kColShift) |
                             (uint32_t(row) << kRowShift) | off;
        return ps->read32(cluster_addr(npu, start_col));
    };

    // Enumerate enabled compute cores once.
    std::vector<std::pair<int, int>> cores;
    for (int c = 0; c < kNumCols; ++c)
        for (int r = kFirstComputeRow; r <= kLastComputeRow; ++r)
            if (rd(c, r, kCoreCtrl) & 0x1u) cores.emplace_back(c, r);
    if (cores.empty()) return;  // DMA-only kernel: nothing to settle

    const bool trace = std::getenv("XDNA_AIESIM_TRACE") != nullptr;
    constexpr uint64_t kWarmupQuanta = 4;  // let cores start before checking
    constexpr uint64_t kStableQuanta = 2;  // consecutive stable quanta = settled
    // Backstop for the case where a core never reaches a stable PC -- e.g.
    // static_L1_init's degenerate empty core (`aie.core { aie.end }`), which the
    // cluster model runs off the end of its program memory (a stream of
    // [ERROR:101] PM-bank faults) so its PC never settles. The legit settle
    // fires via quiescence in a handful of quanta (real cores block on their
    // input lock within a few hundred cycles); this only bounds the non-
    // quiescing case. Kept generous enough for any real staggered-blocking core
    // yet small enough that a degenerate core costs ~tens of sim-us, not the
    // ~2M cycles (minutes of wall-clock at aiesim's rate) that timed the kernel
    // out. Hitting it just proceeds -- the cores keep running during the
    // sequence, as they did before the settle existed.
    constexpr uint64_t kSettleCapNs = 16'384;  // 64 quanta (~16k cycles)

    auto sample = [&]() {
        std::vector<uint32_t> pcs;
        pcs.reserve(cores.size());
        for (const auto& cr : cores) pcs.push_back(rd(cr.first, cr.second, kCorePc));
        return pcs;
    };

    std::vector<uint32_t> prev = sample();
    uint64_t elapsed = 0, advanced = 0, stable = 0;
    while (elapsed < kSettleCapNs) {
        sc_core::wait(sc_core::sc_time(double(kPollQuantumNs), sc_core::SC_NS));
        elapsed += kPollQuantumNs;
        ++advanced;
        std::vector<uint32_t> cur = sample();
        if (advanced <= kWarmupQuanta) {  // skip the "not started yet" window
            prev = std::move(cur);
            continue;
        }
        if (cur == prev) {
            if (++stable >= kStableQuanta) {
                if (trace)
                    std::fprintf(stderr,
                                 "[settle] array quiescent after %lluns (%zu core(s))\n",
                                 (unsigned long long)elapsed, cores.size());
                return;
            }
        } else {
            stable = 0;
        }
        prev = std::move(cur);
    }
    if (trace)
        std::fprintf(stderr,
                     "[settle] cap reached at %lluns (%zu core(s), not fully quiesced)\n",
                     (unsigned long long)elapsed, cores.size());
}

}  // namespace

int npu_replay(ps_bridge* ps, ddr_target* ddr, const uint8_t* ops, std::size_t len,
               uint8_t start_col,
               const std::vector<std::pair<uint64_t, std::size_t>>& host_buffers) {
    Reader r{ops, len};
    probe_tile(ps, start_col, "entry");
    // Let the just-configured cores reach their first lock/stream block before
    // replaying the host runtime sequence -- models the HW config->submit gap
    // so runtime writes can't race a core's pre-lock work (see settle_array).
    settle_array(ps, start_col);
    while (r.i < r.n && !r.err) {
        const uint8_t tag = r.u8();
        switch (tag) {
            case WRITE32: {
                uint32_t a = r.u32(), v = r.u32();
                if (r.err) break;
                // Task-queue PUSH pacing (faithful to aie-rt
                // _XAieMl_DmaWaitForBdTaskQueue): the shim DMA start queue is
                // depth 4 (StartQSizeMax). On real hardware the MMIO write to a
                // full queue stalls, so the driver fires dma_start_task back to
                // back and the hardware paces it. The cluster model instead
                // accepts-and-DROPS overflow pushes, so a batch of >5 fire-and-
                // forget tasks silently loses every task past the 5th (4 queued +
                // 1 in-flight) -- exactly the bd_id 5,6,7 drops seen on
                // shim_dma_bd_reuse. We restore the hardware pacing: before a
                // task-queue push, poll the channel's Task_Queue_Size MSB and
                // advance sim-time until the queue has room.
                if (const uint32_t row = (a >> kRowShift) & 0x1Fu; row == 0) {
                    if (const uint32_t qstat = shim_taskqueue_status_off(a & 0xFFFFFu)) {
                        const uint64_t stat_addr =
                            cluster_addr((a & ~uint64_t(0xFFFFF)) | qstat, start_col);
                        uint64_t waited = 0;
                        while (ps->read32(stat_addr) & kTaskQueueFullBit) {
                            if (waited >= poll_max_ns()) {
                                std::fprintf(stderr,
                                             "[npu_replay] task-queue pace timeout: off=0x%05x\n",
                                             a & 0xFFFFFu);
                                break;
                            }
                            sc_core::wait(sc_core::sc_time(double(kPollQuantumNs), sc_core::SC_NS));
                            waited += kPollQuantumNs;
                        }
                    }
                }
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
                const bool trace_bd = std::getenv("XDNA_AIESIM_TRACE") &&
                                      (a & 0xFFFFFu) >= 0x1D100u && (a & 0xFFFFFu) < 0x1D120u;
                if (trace_bd)
                    std::fprintf(stderr, "[npu_replay] BLOCK_WRITE BD8 a=0x%05x count=%u:", a & 0xFFFFFu, count);
                for (uint32_t k = 0; k < count; ++k) {
                    uint32_t w = r.u32();
                    ps->write32(base + k * 4u, w);
                    if (trace_bd) std::fprintf(stderr, " W%u=0x%08x", k, w);
                }
                if (trace_bd) std::fprintf(stderr, "\n");
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
                const uint64_t lo_reg = cluster_addr(reg_addr, start_col);

                // Two address-resolution modes, mirroring the interpreter
                // (executor.rs execute_ddr_patch / _prepatched). Both write the
                // resolved DDR address into the shim BD address word pair (low 32
                // to the addr-low word, high 16 to the addr-high word).
                //
                //  - Host-buffer mode (classic insts.bin path): the plugin
                //    registered the kernel-arg BOs, so the address is the buffer
                //    base + the BD's byte offset, already in the bridge's flat
                //    ddr_target space -- no translation.
                //
                //  - Pre-patched mode (XRT ELF/module path, opcode 20): the
                //    regmap carries no BOs, so host_buffers is empty. XRT's ELF
                //    relocation already wrote BO_addr + arg_plus + the DDR-AIE
                //    aperture offset into the BD via the preceding block-write.
                //    Read it back, strip the offset, write the flat-space address.
                //    The subtract stands in for the AIE's host-DDR aperture, which
                //    the bridge does not model as a translation (real silicon's
                //    memory controller absorbs this 2 GB offset; we keep one flat
                //    ddr_target instead -- see executor.rs execute_ddr_patch_
                //    prepatched). TODO: model the aperture properly so both flows
                //    share one translation rather than this special case.
                uint64_t patched;
                if (host_buffers.empty()) {
                    constexpr uint64_t kDdrAieAddrOffset = 0x8000'0000u;
                    const uint32_t word_lo = ps->read32(lo_reg);
                    const uint32_t word_hi = ps->read32(lo_reg + 4);
                    const uint64_t xrt_addr =
                        ((uint64_t(word_hi) & 0xFFFFu) << 32) | word_lo;
                    patched = xrt_addr - kDdrAieAddrOffset;
                } else {
                    patched = resolve_arg_base(host_buffers, arg_idx) + arg_plus;
                }

                if (std::getenv("XDNA_AIESIM_TRACE")) {
                    std::fprintf(stderr,
                                 "[npu_replay] DdrPatch arg_idx=%u patched=0x%llx -> reg=0x%llx%s\n",
                                 arg_idx, (unsigned long long)patched,
                                 (unsigned long long)lo_reg,
                                 host_buffers.empty() ? " (pre-patched)" : "");
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
                if (!dma_wait(ps, ddr, status_addr)) {
                    std::fprintf(stderr,
                                 "[npu_replay] Sync timeout: col=%u row=%u dir=%u ch=%u "
                                 "status=0x%llx\n",
                                 column, row, direction, channel,
                                 (unsigned long long)status_addr);
                    probe_tile(ps, start_col, "timeout");
                    if (ddr && std::getenv("XDNA_AIESIM_TRACE")) {
                        // Partial-transfer discriminator: how many 256-word
                        // slices of each host buffer physically reached DDR at the
                        // wedge? (Unwritten sparse pages read as 0.) Distinguishes
                        // a clean completion stall (all slices present, only the
                        // token missing) from dropped data upstream -- e.g. the
                        // start-queue overrun that left shim_dma_bd_reuse 6 slices
                        // short before the task-queue pacing above was added.
                        for (std::size_t bi = 0; bi < host_buffers.size(); ++bi) {
                            const uint64_t ba = host_buffers[bi].first;
                            const std::size_t nw = host_buffers[bi].second / 4;
                            std::size_t filled = 0;
                            std::fprintf(stderr, "[timeout] hostbuf[%zu]@0x%llx %zuw slices:",
                                         bi, (unsigned long long)ba, nw);
                            for (std::size_t s = 0; s * 256 < nw; ++s) {
                                const std::size_t rem = nw - s * 256;
                                const std::size_t cnt = rem < 256 ? rem : 256;
                                uint32_t w[256];
                                ddr->host_read(ba + uint64_t(s) * 256 * 4, w, cnt * 4);
                                bool nz = false;
                                for (std::size_t k = 0; k < cnt; ++k)
                                    if (w[k]) { nz = true; break; }
                                if (nz) { ++filled; std::fprintf(stderr, " s%zu=0x%08x", s, w[0]); }
                            }
                            std::fprintf(stderr, "  [%zu non-zero]\n", filled);
                        }
                    }
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
