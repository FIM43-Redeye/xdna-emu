//! Configurable timing tolerance bands for VCD signal comparison.
//!
//! The emulator and aiesimulator will not agree on exact cycle timing for
//! every signal. Some differences are acceptable and expected:
//!
//! - DMA transfers may start 1-2 cycles apart due to pipeline depth modeling.
//! - Stream switch routing may have different latency models.
//! - Lock operations may complete at slightly different times.
//! - Core PC values are the accuracy target and should match exactly.
//!
//! [`ToleranceConfig`] lets the comparison engine say "DMA address signals
//! can differ by up to 2 cycles" and report them as "matched with timing
//! offset" rather than "mismatched".
//!
//! # Lookup priority
//!
//! Field-specific overrides take precedence over subsystem overrides, which
//! in turn take precedence over the default:
//!
//! ```
//! use xdna_emu::vcd::tolerance::ToleranceConfig;
//! use xdna_emu::vcd::state_path::{DmaDir, StatePath, Subsystem};
//!
//! let config = ToleranceConfig::new(3)
//!     .with_subsystem(Subsystem::Dma, 2)
//!     .with_field("address", 5);
//!
//! // Field override wins over subsystem override.
//! let addr = StatePath::DmaAddress { col: 0, row: 1, dir: DmaDir::S2mm, ch: 0 };
//! assert_eq!(config.tolerance_for(&addr), 5);
//!
//! // Subsystem override wins over default.
//! let fsm = StatePath::DmaFsmState { col: 0, row: 1, dir: DmaDir::S2mm, ch: 0 };
//! assert_eq!(config.tolerance_for(&fsm), 2);
//! ```

use crate::vcd::state_path::{StatePath, Subsystem};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ToleranceConfig
// ---------------------------------------------------------------------------

/// Configurable timing tolerance bands for VCD signal comparison.
///
/// Defines how many cycles of timing difference are acceptable for each
/// signal type. Signals within tolerance are reported as "matched with
/// timing offset" rather than "mismatched".
///
/// Construct via [`ToleranceConfig::new`], [`ToleranceConfig::aie2_default`],
/// [`ToleranceConfig::strict`], or [`ToleranceConfig::relaxed`]. Use the
/// builder methods [`with_subsystem`][ToleranceConfig::with_subsystem] and
/// [`with_field`][ToleranceConfig::with_field] to layer overrides.
#[derive(Debug, Clone)]
pub struct ToleranceConfig {
    /// Default tolerance for signals not explicitly configured.
    pub default_cycles: u64,
    /// Per-subsystem tolerance overrides.
    subsystem_tolerances: HashMap<Subsystem, u64>,
    /// Per-field tolerance overrides (keyed by [`StatePath::field_name`]).
    field_tolerances: HashMap<String, u64>,
}

impl ToleranceConfig {
    /// Create a tolerance config with the given default cycle tolerance.
    ///
    /// All signals use `default_cycles` unless overridden via
    /// [`with_subsystem`][Self::with_subsystem] or
    /// [`with_field`][Self::with_field].
    pub fn new(default_cycles: u64) -> Self {
        ToleranceConfig {
            default_cycles,
            subsystem_tolerances: HashMap::new(),
            field_tolerances: HashMap::new(),
        }
    }

    /// Set tolerance for an entire subsystem (builder method).
    ///
    /// Overrides the default for every signal in `subsystem`, unless a more
    /// specific field-level override also applies.
    pub fn with_subsystem(mut self, subsystem: Subsystem, cycles: u64) -> Self {
        self.subsystem_tolerances.insert(subsystem, cycles);
        self
    }

    /// Set tolerance for a specific field name (builder method).
    ///
    /// The `field_name` must match what [`StatePath::field_name`] returns for
    /// the signal of interest (e.g. `"address"`, `"fsm_state"`, `"pc"`).
    /// Field-level overrides take highest precedence.
    pub fn with_field(mut self, field_name: &str, cycles: u64) -> Self {
        self.field_tolerances.insert(field_name.to_string(), cycles);
        self
    }

    /// Look up the tolerance (in cycles) for a given [`StatePath`].
    ///
    /// Resolution priority: field-specific > subsystem > default.
    pub fn tolerance_for(&self, path: &StatePath) -> u64 {
        // Field-specific check first (highest priority).
        let field = path.field_name();
        if let Some(&t) = self.field_tolerances.get(field) {
            return t;
        }
        // Subsystem check next.
        let sub = path.subsystem();
        if let Some(&t) = self.subsystem_tolerances.get(&sub) {
            return t;
        }
        // Fall back to the global default.
        self.default_cycles
    }

    // -----------------------------------------------------------------------
    // Named presets
    // -----------------------------------------------------------------------

    /// Recommended default tolerance config for AIE2 VCD comparison.
    ///
    /// Bands are based on observed differences between the emulator and
    /// aiesimulator across subsystems:
    ///
    /// | Subsystem   | Tolerance | Rationale |
    /// |-------------|-----------|-----------|
    /// | Core        | 0 cycles  | Accuracy target; PC must match exactly |
    /// | Lock        | 0 cycles  | Lock ops are synchronisation points |
    /// | DMA         | 2 cycles  | Pipeline depth differences |
    /// | Stream      | 2 cycles  | Routing latency model differences |
    /// | Event       | 0 cycles  | Events are tied to core/DMA timing |
    /// | Memory      | 1 cycle   | Bank conflict modelling |
    /// | PerfCount   | 0 cycles  | Counters should track exactly |
    pub fn aie2_default() -> Self {
        ToleranceConfig::new(0)
            .with_subsystem(Subsystem::Core, 0)
            .with_subsystem(Subsystem::Lock, 0)
            .with_subsystem(Subsystem::Dma, 2)
            .with_subsystem(Subsystem::Stream, 2)
            .with_subsystem(Subsystem::Event, 0)
            .with_subsystem(Subsystem::Memory, 1)
            .with_subsystem(Subsystem::PerfCount, 0)
    }

    /// Strict tolerance: 0 cycles for every signal.
    ///
    /// Any timing difference at all is flagged as a mismatch. Use this to
    /// audit whether the emulator achieves exact cycle-accuracy.
    pub fn strict() -> Self {
        ToleranceConfig::new(0)
    }

    /// Relaxed tolerance: generous margins for initial development.
    ///
    /// Useful when bringing up a new subsystem and exact timing is not yet
    /// calibrated. Core and Lock tolerances are also loosened so that
    /// structural correctness can be verified before timing precision.
    pub fn relaxed() -> Self {
        ToleranceConfig::new(5)
            .with_subsystem(Subsystem::Core, 2)
            .with_subsystem(Subsystem::Lock, 2)
            .with_subsystem(Subsystem::Dma, 5)
            .with_subsystem(Subsystem::Stream, 5)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vcd::state_path::*;

    #[test]
    fn default_tolerance_returned_when_no_overrides() {
        let config = ToleranceConfig::new(3);
        let path = StatePath::LockValue { col: 0, row: 1, idx: 0 };
        assert_eq!(config.tolerance_for(&path), 3);
    }

    #[test]
    fn subsystem_override_takes_precedence_over_default() {
        let config = ToleranceConfig::new(3).with_subsystem(Subsystem::Lock, 0);
        let path = StatePath::LockValue { col: 0, row: 1, idx: 0 };
        assert_eq!(config.tolerance_for(&path), 0);
    }

    #[test]
    fn field_override_takes_precedence_over_subsystem() {
        let config = ToleranceConfig::new(3)
            .with_subsystem(Subsystem::Dma, 2)
            .with_field("address", 5);
        let path = StatePath::DmaAddress { col: 0, row: 1, dir: DmaDir::S2mm, ch: 0 };
        assert_eq!(config.tolerance_for(&path), 5);
    }

    #[test]
    fn field_override_does_not_affect_different_field_in_same_subsystem() {
        // "address" override should not change "fsm_state" in the same DMA subsystem.
        let config = ToleranceConfig::new(3)
            .with_subsystem(Subsystem::Dma, 2)
            .with_field("address", 5);
        let path = StatePath::DmaFsmState { col: 0, row: 1, dir: DmaDir::S2mm, ch: 0 };
        assert_eq!(config.tolerance_for(&path), 2);
    }

    #[test]
    fn aie2_default_core_is_zero() {
        let config = ToleranceConfig::aie2_default();
        let path = StatePath::CorePc { col: 0, row: 2, stage: 1 };
        assert_eq!(config.tolerance_for(&path), 0);
    }

    #[test]
    fn aie2_default_dma_is_two() {
        let config = ToleranceConfig::aie2_default();
        let path = StatePath::DmaFsmState { col: 0, row: 1, dir: DmaDir::S2mm, ch: 0 };
        assert_eq!(config.tolerance_for(&path), 2);
    }

    #[test]
    fn aie2_default_stream_is_two() {
        let config = ToleranceConfig::aie2_default();
        let path = StatePath::StreamPortData {
            col: 0,
            row: 1,
            port: PortId::named("sSouth0"),
        };
        assert_eq!(config.tolerance_for(&path), 2);
    }

    #[test]
    fn aie2_default_memory_is_one() {
        let config = ToleranceConfig::aie2_default();
        let path = StatePath::MemBankConflict { col: 0, row: 1, bank: 0 };
        assert_eq!(config.tolerance_for(&path), 1);
    }

    #[test]
    fn aie2_default_lock_is_zero() {
        let config = ToleranceConfig::aie2_default();
        let path = StatePath::LockValue { col: 0, row: 1, idx: 3 };
        assert_eq!(config.tolerance_for(&path), 0);
    }

    #[test]
    fn aie2_default_event_is_zero() {
        let config = ToleranceConfig::aie2_default();
        let path = StatePath::EventTrace {
            col: 0,
            row: 2,
            event_code: 0x42,
            event_name: "core_active".to_string(),
        };
        assert_eq!(config.tolerance_for(&path), 0);
    }

    #[test]
    fn aie2_default_perf_count_is_zero() {
        let config = ToleranceConfig::aie2_default();
        let path = StatePath::PerfCounter { col: 0, row: 1, idx: 0 };
        assert_eq!(config.tolerance_for(&path), 0);
    }

    #[test]
    fn strict_is_all_zero() {
        let config = ToleranceConfig::strict();
        // All subsystems should return 0.
        let paths: Vec<StatePath> = vec![
            StatePath::DmaAddress { col: 0, row: 1, dir: DmaDir::S2mm, ch: 0 },
            StatePath::LockValue { col: 0, row: 1, idx: 0 },
            StatePath::CorePc { col: 0, row: 2, stage: 0 },
            StatePath::StreamPortData {
                col: 0,
                row: 1,
                port: PortId::named("sSouth0"),
            },
            StatePath::MemBankConflict { col: 0, row: 1, bank: 0 },
            StatePath::PerfCounter { col: 0, row: 1, idx: 0 },
        ];
        for path in &paths {
            assert_eq!(config.tolerance_for(path), 0, "strict() failed for {:?}", path);
        }
    }

    #[test]
    fn relaxed_dma_is_five() {
        let config = ToleranceConfig::relaxed();
        let path = StatePath::DmaFsmState { col: 0, row: 1, dir: DmaDir::Mm2s, ch: 1 };
        assert_eq!(config.tolerance_for(&path), 5);
    }

    #[test]
    fn relaxed_core_is_two() {
        let config = ToleranceConfig::relaxed();
        let path = StatePath::CoreReset { col: 1, row: 3 };
        assert_eq!(config.tolerance_for(&path), 2);
    }

    #[test]
    fn relaxed_unconfigured_subsystem_uses_default() {
        // Memory is not explicitly set in relaxed(); should fall back to default=5.
        let config = ToleranceConfig::relaxed();
        let path = StatePath::MemBankConflict { col: 0, row: 1, bank: 2 };
        assert_eq!(config.tolerance_for(&path), 5);
    }

    #[test]
    fn builder_is_composable_multiple_subsystems() {
        let config = ToleranceConfig::new(10)
            .with_subsystem(Subsystem::Core, 0)
            .with_subsystem(Subsystem::Dma, 3);
        assert_eq!(
            config.tolerance_for(&StatePath::CorePc { col: 0, row: 1, stage: 0 }),
            0
        );
        assert_eq!(
            config.tolerance_for(&StatePath::DmaFsmState {
                col: 0,
                row: 1,
                dir: DmaDir::S2mm,
                ch: 0
            }),
            3
        );
        // Stream not configured -- should use default.
        assert_eq!(
            config.tolerance_for(&StatePath::StreamPortIdle {
                col: 0,
                row: 1,
                port: PortId::named("sNorth0"),
            }),
            10
        );
    }

    #[test]
    fn clone_is_independent() {
        let original = ToleranceConfig::new(1).with_subsystem(Subsystem::Dma, 2);
        let mut cloned = original.clone();
        cloned = cloned.with_subsystem(Subsystem::Dma, 99);
        // Original must be unchanged.
        let path = StatePath::DmaFsmState { col: 0, row: 1, dir: DmaDir::S2mm, ch: 0 };
        assert_eq!(original.tolerance_for(&path), 2);
        assert_eq!(cloned.tolerance_for(&path), 99);
    }
}
