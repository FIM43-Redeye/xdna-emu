//! DMA timing configuration.
//!
//! Provides cycle counts for each phase of a DMA transfer. The unified
//! channel FSM (`ChannelFsm`) uses these values directly as countdown
//! timers in its state variants -- no separate timing state machine needed.
//!
//! # Timing Model
//!
//! ```text
//! ┌──────────┐  ┌─────────┐  ┌──────────────┐  ┌─────────┐  ┌──────────┐
//! │ BD Setup │->│ Lock    │->│ Data         │->│ Lock    │->│ BD Chain │
//! │ (4 cyc)  │  │ Acquire │  │ Transfer     │  │ Release │  │ (2 cyc)  │
//! │          │  │ (1 cyc) │  │ (N/bandwidth)│  │ (1 cyc) │  │          │
//! └──────────┘  └─────────┘  └──────────────┘  └─────────┘  └──────────┘
//! ```

/// DMA timing configuration.
#[derive(Debug, Clone, Copy)]
pub struct DmaTimingConfig {
    /// Cycles to parse and setup a buffer descriptor
    pub bd_setup_cycles: u8,

    /// Cycles from channel start to first data movement
    pub channel_start_cycles: u8,

    /// Words (32-bit) transferred per cycle per channel
    pub words_per_cycle: u8,

    /// Memory access latency in cycles
    pub memory_latency_cycles: u8,

    /// Cycles to acquire a lock
    pub lock_acquire_cycles: u8,

    /// Cycles to release a lock
    pub lock_release_cycles: u8,

    /// Cycles between BD completion and next BD start
    pub bd_chain_cycles: u8,

    /// Throughput: words (32-bit) per cycle for shim DMA transfers
    /// touching host memory.  Separate from `words_per_cycle` because
    /// the shim AXI master / DDR interface is narrower than the tile
    /// data memory bus on Phoenix.  HW measurement: 1 word/cyc.
    pub shim_words_per_cycle: u8,

    /// Throughput: words (32-bit) per cycle across a stream-switch port.  The
    /// AIE2 inter-tile stream is a 32-bit AXI4-Stream (1 word/cyc/port), so a
    /// transfer that crosses a stream (memtile/compute MM2S egress, S2MM
    /// ingress) is rate-limited by this, not the wider `words_per_cycle` tile
    /// data bus.  Without this the MM2S bursts the 4-word memory-read rate into
    /// the shallow stream FIFO, fragmenting PORT_RUNNING at the opening; HW
    /// meters egress to 1 word/cyc and stays continuously asserted (#140).
    pub stream_words_per_cycle: u8,

    /// Extra pipeline latency for shim tile DDR access (NoC + DDR controller).
    /// Applied once per BD, between MemoryLatency and Transferring, only for
    /// shim tiles with host memory endpoints.  Folded into per-direction
    /// cold-start values as of 2026-05-25; default is 0.
    pub host_memory_latency_cycles: u16,

    /// True one-shot cold-start latency for shim MM2S DMA, once per
    /// channel per session.  Fires only on the FIRST task on this channel
    /// after the engine starts; subsequent tasks skip it (HW keeps the
    /// channel warm indefinitely).  See finding
    /// 2026-05-25-shim-bd-chain-amortization.
    pub shim_ddr_cold_start_mm2s_cycles: u16,

    /// True one-shot cold-start latency for shim S2MM DMA.  Near-zero on
    /// Phoenix -- pull direction never fills a DDR read pipeline.
    pub shim_ddr_cold_start_s2mm_cycles: u16,

    /// Per-task overhead on shim MM2S DMA touching host memory, paid on
    /// EVERY task.  Distinct from `shim_ddr_cold_start_mm2s_cycles` which
    /// is one-shot per session.
    pub shim_per_task_overhead_mm2s_cycles: u16,

    /// Per-task overhead on shim S2MM DMA touching host memory.
    pub shim_per_task_overhead_s2mm_cycles: u16,

    /// Geometric decay ratio (per-mille) of the shim MM2S warm-up
    /// transient.  The cold-start cost is charged as
    /// `shim_ddr_cold_start_mm2s_cycles * (permille/1000)^i` at task
    /// index `i` in a channel session, so it decays across the chain
    /// rather than firing once on task 0.  Calibrated to ~310 (r=0.31)
    /// against 2026-05-27 N=50 K=8 HW.  Phase 2d warm-up-transient model.
    pub shim_warmup_decay_mm2s_permille: u16,

    /// Geometric decay ratio (per-mille) of the shim S2MM warm-up
    /// transient.  ~0 on Phoenix -- S2MM has no measurable tail past
    /// task 0, so this preserves the pure one-shot cold-start.
    pub shim_warmup_decay_s2mm_permille: u16,

    /// Memtile lock-release pipeline latency, in cycles, from BD completion
    /// (FINISHED_BD) to the cross-lock release landing on the semaphore.  Even
    /// a swap that is immediately grantable does not release until this many
    /// cycles after the buffer finished filling/draining; under backpressure
    /// the swap dominates and this is absorbed.  Calibrated to ~63 on NPU1
    /// Phoenix from the tenant-4 producer-fill probe (HW: FINISHED_BD->full-REL
    /// gap 63-64 on the warmup/prompt releases, ~2071 once backpressured).
    /// Applied only on memtile tiles (cross-lock producer/consumer handoff);
    /// 0 elsewhere -- no compute/shim HW evidence for a non-zero value.
    pub memtile_lock_release_latency_cycles: u16,

    /// One-time-per-channel-session pipeline-FILL latency for memtile DMA
    /// channels, paid on the first task only (the non-shim analogue of the
    /// shim DDR cold-start).  The shim cold-start concentrates all modeled
    /// pipeline-fill in the shim; the memtile channels' deeper ObjectFifo
    /// pipeline fills over ~1200cy on HW that EMU otherwise collapses to
    /// ~55cy, leaving the shim S2MM duration ~1232cy short on add_one (#140).
    /// Grounded in the DMA `Status` STARTING state (AM025); magnitude is
    /// HW-observation.  Fires only when `warm_task_index == 0`, so steady-
    /// state ping-pong cadence is untouched.  Default 0 = no-op until
    /// calibrated against #140 layer-2 HW data.
    pub memtile_first_bd_startup_cycles: u16,

    /// One-time-per-channel-session pipeline-FILL latency for compute (core)
    /// DMA channels, paid on the first task only.  Separate knob from the
    /// memtile one (finer 8-word BDs over a different stream depth calibrate
    /// independently).  Default 0 = no-op (#140 layer-2).
    pub compute_first_bd_startup_cycles: u16,

    /// Minimum inter-BD bubble, in cycles, on the chained-BD prefetch fast
    /// path.  At each `next_bd` boundary the DMA channel deasserts its stream
    /// port (PORT_RUNNING) for this many cycles -- the next_bd-fetch + lock
    /// handshake costs a cycle even when the next buffer is immediately
    /// available.  HW-confirmed: NPU1 Phoenix add_one memtile slot0 traces
    /// `on16 off1 x4` (one 1-cycle bubble per 16-word S2MM BD execution), and
    /// slot4 (MM2S) shows the same.  Larger real inter-BD waits (lock stall,
    /// host pipeline) dominate and absorb this.  `0` restores the old
    /// back-to-back prefetch (no bubble).  Calibrated to 1 on Phoenix;
    /// env-overridable via `XDNA_EMU_BD_SWITCH_BUBBLE`.
    pub bd_switch_bubble_cycles: u16,
}

impl Default for DmaTimingConfig {
    fn default() -> Self {
        // Default to AIE2 for test convenience.  Production callers
        // should use from_model() with the arch-appropriate model.
        Self::from_model(&xdna_archspec::aie2::dma::AIE2_DMA_MODEL)
    }
}

impl DmaTimingConfig {
    /// Create timing config from a DmaModel's per-arch constants.
    ///
    /// The constants come from the arch impl (AIE2: reads from
    /// `xdna_archspec::aie2::timing::DMA_*`).  This is a cold path --
    /// called once per DmaEngine construction.
    pub fn from_model(model: &'static dyn xdna_archspec::dma::DmaModel) -> Self {
        let m = model.timing_config();
        Self {
            bd_setup_cycles: m.bd_setup_cycles,
            channel_start_cycles: m.channel_start_cycles,
            words_per_cycle: m.words_per_cycle,
            shim_words_per_cycle: m.shim_words_per_cycle,
            stream_words_per_cycle: m.stream_words_per_cycle,
            memory_latency_cycles: m.memory_latency_cycles,
            lock_acquire_cycles: m.lock_acquire_cycles,
            lock_release_cycles: m.lock_release_cycles,
            bd_chain_cycles: m.bd_chain_cycles,
            host_memory_latency_cycles: m.host_memory_latency_cycles,
            shim_ddr_cold_start_mm2s_cycles: m.shim_ddr_cold_start_mm2s_cycles,
            shim_ddr_cold_start_s2mm_cycles: m.shim_ddr_cold_start_s2mm_cycles,
            shim_per_task_overhead_mm2s_cycles: m.shim_per_task_overhead_mm2s_cycles,
            shim_per_task_overhead_s2mm_cycles: m.shim_per_task_overhead_s2mm_cycles,
            shim_warmup_decay_mm2s_permille: m.shim_warmup_decay_mm2s_permille,
            shim_warmup_decay_s2mm_permille: m.shim_warmup_decay_s2mm_permille,
            // Non-shim first-BD pipeline-fill startup (#140).  Env-overridable
            // so the layer-2 calibration loop can sweep without rebuilding the
            // archspec crate; absent env leaves the model value (0) in place.
            memtile_first_bd_startup_cycles: dma_u16_from_env(
                "XDNA_EMU_MEMTILE_FIRST_BD_STARTUP",
                m.memtile_first_bd_startup_cycles,
                |k| std::env::var(k).ok(),
            ),
            compute_first_bd_startup_cycles: dma_u16_from_env(
                "XDNA_EMU_COMPUTE_FIRST_BD_STARTUP",
                m.compute_first_bd_startup_cycles,
                |k| std::env::var(k).ok(),
            ),
            // HW-calibrated micro-timing (NPU1 Phoenix memtile, tenant-4 probe).
            // Not yet plumbed through the per-arch DmaModel -- a single memtile
            // observation; promote to the arch model if AIE2P measures different.
            memtile_lock_release_latency_cycles: 63,
            // Per-BD-switch bubble: 1 cycle on Phoenix (NPU1 add_one memtile
            // slot0 = on16/off1). Env-overridable for experiments.
            bd_switch_bubble_cycles: bd_switch_bubble_from_env(1, |k| std::env::var(k).ok()),
        }
    }

    /// Calculate total cycles for a transfer of given size.
    pub fn transfer_cycles(&self, bytes: u64, has_acquire_lock: bool, has_release_lock: bool) -> u64 {
        let words = (bytes + 3) / 4; // Round up to words
        let data_cycles = (words + self.words_per_cycle as u64 - 1) / self.words_per_cycle as u64;

        let mut total = self.bd_setup_cycles as u64
            + self.channel_start_cycles as u64
            + self.memory_latency_cycles as u64
            + data_cycles;

        if has_acquire_lock {
            total += self.lock_acquire_cycles as u64;
        }
        if has_release_lock {
            total += self.lock_release_cycles as u64;
        }

        total
    }

    /// Calculate cycles for BD chaining.
    pub fn chain_cycles(&self) -> u64 {
        self.bd_chain_cycles as u64 + self.bd_setup_cycles as u64
    }
}

/// Overlay the BD-switch bubble width from the environment onto a base.
///
/// Reads `XDNA_EMU_BD_SWITCH_BUBBLE` (via `get`, so it is testable without
/// touching process env). Absent or unparseable leaves `base` unchanged. Set
/// to `0` to restore the old back-to-back chained-BD prefetch (no bubble).
pub fn bd_switch_bubble_from_env(base: u16, get: impl Fn(&str) -> Option<String>) -> u16 {
    get("XDNA_EMU_BD_SWITCH_BUBBLE")
        .and_then(|v| v.trim().parse::<u16>().ok())
        .unwrap_or(base)
}

/// Overlay a `u16` DMA timing value named `key` from the environment onto a
/// base. Absent or unparseable leaves `base` unchanged. `get` is injected so
/// the overlay is testable without mutating process env. Used by the #140
/// layer-2 calibration loop to sweep the non-shim first-BD startup knobs.
pub fn dma_u16_from_env(key: &str, base: u16, get: impl Fn(&str) -> Option<String>) -> u16 {
    get(key).and_then(|v| v.trim().parse::<u16>().ok()).unwrap_or(base)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bd_switch_bubble_defaults_to_one_and_env_overrides() {
        // Phoenix default is a 1-cycle bubble (HW add_one memtile slot0 off1).
        let cfg = DmaTimingConfig::from_model(&xdna_archspec::aie2::dma::AIE2_DMA_MODEL);
        assert_eq!(cfg.bd_switch_bubble_cycles, 1, "default 1-cycle BD-switch bubble");
        // Env can widen or disable it.
        assert_eq!(bd_switch_bubble_from_env(1, |_| None), 1, "absent keeps base");
        assert_eq!(bd_switch_bubble_from_env(1, |_| Some("0".into())), 0, "0 disables bubble");
        assert_eq!(bd_switch_bubble_from_env(1, |_| Some("3".into())), 3, "override widens");
    }

    #[test]
    fn dma_u16_from_env_overlays_named_key() {
        // Absent -> base; present+parseable -> override; junk -> base.
        assert_eq!(dma_u16_from_env("K", 0, |_| None), 0, "absent keeps base");
        assert_eq!(dma_u16_from_env("K", 0, |_| Some("205".into())), 205, "override applies");
        assert_eq!(dma_u16_from_env("K", 7, |_| Some(" 42 ".into())), 42, "trims whitespace");
        assert_eq!(dma_u16_from_env("K", 7, |_| Some("nan".into())), 7, "junk keeps base");
    }

    #[test]
    fn test_timing_config_default() {
        let config = DmaTimingConfig::default();
        assert_eq!(config.bd_setup_cycles, 4);
        // AIE2: 128-bit data bus, which is 4 x 32-bit words per cycle.
        // Source: xdna_archspec::aie2::timing::DMA_WORDS_PER_CYCLE = 4
        // (generated by crates/xdna-archspec/build.rs from AM025
        // xaiemlgbl_params.h's DATAMEMORY_WIDTH=128).
        assert_eq!(config.words_per_cycle, 4);
        // AIE2 inter-tile stream is a 32-bit AXI4-Stream = 1 word/cyc/port.
        assert_eq!(config.stream_words_per_cycle, 1);
    }

    #[test]
    fn test_transfer_cycles_simple() {
        let config = DmaTimingConfig::default();

        // 16 bytes = 4 words at 4 words/cycle = 1 data cycle, no locks
        let cycles = config.transfer_cycles(16, false, false);
        // BD setup (4) + channel start (2) + memory latency (5) + data (1) = 12
        assert_eq!(cycles, 12);
    }

    #[test]
    fn test_transfer_cycles_with_locks() {
        let config = DmaTimingConfig::default();

        // 16 bytes = 4 words at 4 words/cycle = 1 data cycle, with locks
        let cycles = config.transfer_cycles(16, true, true);
        // BD setup (4) + channel start (2) + memory latency (5) + data (1) + acquire (1) + release (1) = 14
        assert_eq!(cycles, 14);
    }
}
