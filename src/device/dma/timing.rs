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

use super::burst::BurstParams;

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

    /// DDR burst-delivery parameters for shim host-memory transfers.  Models a
    /// bursty AXI-master/DDR delivery cadence at the *producer* (shim host
    /// read).  DISABLED by default: this was the wrong lever for the memtile
    /// PORT_RUNNING cadence (that gap is a consumer-side per-BD-switch bubble,
    /// `bd_switch_bubble_cycles`).  Parked, still env-enablable via
    /// `XDNA_EMU_DDR_BURST_WORDS` for experiments.  See `src/device/dma/burst.rs`.
    pub ddr_burst: BurstParams,

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
            // HW-calibrated micro-timing (NPU1 Phoenix memtile, tenant-4 probe).
            // Not yet plumbed through the per-arch DmaModel -- a single memtile
            // observation; promote to the arch model if AIE2P measures different.
            memtile_lock_release_latency_cycles: 63,
            // DDR burst delivery: DISABLED by default (uniform delivery).  The
            // shim-side burst gate was the wrong model for the memtile
            // PORT_RUNNING cadence -- that gap is a per-BD-switch bubble at the
            // *consumer* (bd_switch_bubble_cycles), not a producer-side AXI
            // burst.  A free-running 16-word gate fragments contiguous BDs and
            // smears chained transfers (the k8 regression).  BurstGate is parked
            // (still env-enablable via XDNA_EMU_DDR_BURST_WORDS) pending a
            // keep/delete call.  See src/device/dma/burst.rs.
            ddr_burst: ddr_burst_from_env(BurstParams::DISABLED, |k| std::env::var(k).ok()),
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

/// Overlay DDR burst parameters from the environment onto a base.
///
/// Reads (via `get`, so the parse is testable without touching process env):
/// - `XDNA_EMU_DDR_PROFILE`         -> named calibrated base (`phoenix` =
///   `AIE2_DDR_PHOENIX`; `off`/`none` = `DISABLED`); range vars below override it
/// - `XDNA_EMU_DDR_BURST_WORDS`     -> both `burst_words_{min,max}` (degenerate /
///   fixed alias; set to non-zero `max` to enable)
/// - `XDNA_EMU_DDR_BURST_WORDS_{MIN,MAX}`        -> each `burst_words_*` bound
/// - `XDNA_EMU_DDR_INTER_BURST_CYCLES` -> both `inter_burst_cycles_{min,max}`
/// - `XDNA_EMU_DDR_INTER_BURST_CYCLES_{MIN,MAX}` -> each `inter_burst_cycles_*`
/// - `XDNA_EMU_DDR_FIRST_LATENCY`   -> `first_access_latency`
///
/// Precedence per bound: explicit `_MIN`/`_MAX` range var > single-value alias
/// (sets both bounds = degenerate/fixed cadence) > base value. The PRNG seed is
/// separate (`XDNA_EMU_DDR_SEED`, resolved process-wide in `burst::master_seed`).
/// Any var that is absent or unparseable leaves the corresponding field at its
/// base value. Enabling is opt-in: with no env set and a `DISABLED` base the
/// result is `DISABLED` (uniform delivery).
pub fn ddr_burst_from_env(base: BurstParams, get: impl Fn(&str) -> Option<String>) -> BurstParams {
    // A named profile (XDNA_EMU_DDR_PROFILE) selects a calibrated base; explicit
    // range vars below still override individual bounds.
    let base = match get("XDNA_EMU_DDR_PROFILE").as_deref().map(str::trim) {
        Some("phoenix") => BurstParams::AIE2_DDR_PHOENIX,
        Some("off") | Some("none") => BurstParams::DISABLED,
        _ => base,
    };
    let parse = |key: &str| get(key).and_then(|v| v.trim().parse::<u16>().ok());
    // Single-value vars set BOTH bounds; _MIN/_MAX then override each bound.
    let bw = parse("XDNA_EMU_DDR_BURST_WORDS");
    let ib = parse("XDNA_EMU_DDR_INTER_BURST_CYCLES");
    BurstParams {
        burst_words_min: parse("XDNA_EMU_DDR_BURST_WORDS_MIN").or(bw).unwrap_or(base.burst_words_min),
        burst_words_max: parse("XDNA_EMU_DDR_BURST_WORDS_MAX").or(bw).unwrap_or(base.burst_words_max),
        inter_burst_cycles_min: parse("XDNA_EMU_DDR_INTER_BURST_CYCLES_MIN")
            .or(ib)
            .unwrap_or(base.inter_burst_cycles_min),
        inter_burst_cycles_max: parse("XDNA_EMU_DDR_INTER_BURST_CYCLES_MAX")
            .or(ib)
            .unwrap_or(base.inter_burst_cycles_max),
        first_access_latency: parse("XDNA_EMU_DDR_FIRST_LATENCY").unwrap_or(base.first_access_latency),
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
    fn ddr_burst_disabled_by_default() {
        // No env -> the model is off (uniform delivery), so default cycle
        // accuracy is untouched.
        let cfg = ddr_burst_from_env(BurstParams::DISABLED, |_| None);
        assert_eq!(cfg, BurstParams::DISABLED);
        assert!(!cfg.enabled());
    }

    #[test]
    fn ddr_burst_env_overlay_enables_and_overrides() {
        use std::collections::HashMap;
        let env: HashMap<&str, &str> = [
            ("XDNA_EMU_DDR_BURST_WORDS", "256"),
            ("XDNA_EMU_DDR_INTER_BURST_CYCLES", "1024"),
            ("XDNA_EMU_DDR_FIRST_LATENCY", "600"),
        ]
        .into_iter()
        .collect();
        let cfg = ddr_burst_from_env(BurstParams::DISABLED, |k| env.get(k).map(|s| s.to_string()));
        assert!(cfg.enabled());
        // Single-value vars set both bounds (degenerate / fixed alias).
        assert_eq!((cfg.burst_words_min, cfg.burst_words_max), (256, 256));
        assert_eq!((cfg.inter_burst_cycles_min, cfg.inter_burst_cycles_max), (1024, 1024));
        assert_eq!(cfg.first_access_latency, 600);

        // Partial env: unset fields keep the base value.
        let partial: HashMap<&str, &str> = [("XDNA_EMU_DDR_BURST_WORDS", "64")].into_iter().collect();
        let base = BurstParams {
            burst_words_min: 0,
            burst_words_max: 0,
            inter_burst_cycles_min: 999,
            inter_burst_cycles_max: 999,
            first_access_latency: 7,
        };
        let cfg2 = ddr_burst_from_env(base, |k| partial.get(k).map(|s| s.to_string()));
        assert_eq!((cfg2.burst_words_min, cfg2.burst_words_max), (64, 64));
        assert_eq!(
            (cfg2.inter_burst_cycles_min, cfg2.inter_burst_cycles_max),
            (999, 999),
            "unset field keeps base"
        );
        assert_eq!(cfg2.first_access_latency, 7, "unset field keeps base");
    }

    #[test]
    fn ddr_burst_profile_selects_calibrated_base_and_vars_override() {
        use std::collections::HashMap;
        // The phoenix profile selects the band-calibrated range.
        let env: HashMap<&str, &str> = [("XDNA_EMU_DDR_PROFILE", "phoenix")].into_iter().collect();
        let cfg = ddr_burst_from_env(BurstParams::DISABLED, |k| env.get(k).map(|s| s.to_string()));
        assert!(cfg.enabled());
        assert_eq!(cfg, BurstParams::AIE2_DDR_PHOENIX);

        // Explicit range vars still override a profile bound.
        let env2: HashMap<&str, &str> =
            [("XDNA_EMU_DDR_PROFILE", "phoenix"), ("XDNA_EMU_DDR_BURST_WORDS_MAX", "60")]
                .into_iter()
                .collect();
        let cfg2 = ddr_burst_from_env(BurstParams::DISABLED, |k| env2.get(k).map(|s| s.to_string()));
        assert_eq!(cfg2.burst_words_min, 36, "profile min retained");
        assert_eq!(cfg2.burst_words_max, 60, "explicit _MAX overrides profile");

        // 'off' forces DISABLED even over a non-disabled base.
        let env3: HashMap<&str, &str> = [("XDNA_EMU_DDR_PROFILE", "off")].into_iter().collect();
        let cfg3 = ddr_burst_from_env(BurstParams::AIE2_DDR_PHOENIX, |k| env3.get(k).map(|s| s.to_string()));
        assert!(!cfg3.enabled());
    }

    #[test]
    fn ddr_burst_env_min_max_ranges_and_precedence() {
        use std::collections::HashMap;
        // Explicit _MIN/_MAX set a stochastic range directly.
        let env: HashMap<&str, &str> = [
            ("XDNA_EMU_DDR_BURST_WORDS_MIN", "8"),
            ("XDNA_EMU_DDR_BURST_WORDS_MAX", "16"),
            ("XDNA_EMU_DDR_INTER_BURST_CYCLES_MIN", "7"),
            ("XDNA_EMU_DDR_INTER_BURST_CYCLES_MAX", "16"),
        ]
        .into_iter()
        .collect();
        let cfg = ddr_burst_from_env(BurstParams::DISABLED, |k| env.get(k).map(|s| s.to_string()));
        assert!(cfg.enabled());
        assert_eq!((cfg.burst_words_min, cfg.burst_words_max), (8, 16));
        assert_eq!((cfg.inter_burst_cycles_min, cfg.inter_burst_cycles_max), (7, 16));

        // Precedence: _MAX overrides the single-value alias for that bound only;
        // the un-overridden bound takes the single-value alias.
        let env2: HashMap<&str, &str> =
            [("XDNA_EMU_DDR_BURST_WORDS", "10"), ("XDNA_EMU_DDR_BURST_WORDS_MAX", "20")]
                .into_iter()
                .collect();
        let cfg2 = ddr_burst_from_env(BurstParams::DISABLED, |k| env2.get(k).map(|s| s.to_string()));
        assert_eq!(
            (cfg2.burst_words_min, cfg2.burst_words_max),
            (10, 20),
            "min from single-value alias, max from explicit _MAX"
        );
    }

    #[test]
    fn from_model_disables_ddr_burst_by_default() {
        // The shim-side DDR burst gate is parked (wrong lever for the memtile
        // PORT_RUNNING cadence -- that is a consumer-side per-BD-switch bubble).
        // It must be OFF by default so it never perturbs cycle baselines; the
        // env overlay can still enable it for experiments. (Relies on
        // XDNA_EMU_DDR_* being unset, as in CI.)
        let cfg = DmaTimingConfig::from_model(&xdna_archspec::aie2::dma::AIE2_DMA_MODEL);
        assert!(!cfg.ddr_burst.enabled(), "burst gate should be parked (off) by default");
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
