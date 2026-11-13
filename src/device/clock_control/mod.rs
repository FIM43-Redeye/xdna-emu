//! AIE2 clock-control subsystem.
//!
//! Owns all clock-gating state for the array (column / module / adaptive
//! tiers).  Boots with every tile gated, matching silicon behavior per
//! aie-rt's XAie_PmRequestTiles documentation.  Tests opt out via
//! `ungate_all()` which exercises the same register-write path the
//! real CDO uses.
//!
//! Spec: xdna-emu-priv: research/specs/2026-05-24-clock-control-design.md

use std::collections::{HashMap, HashSet};

use xdna_archspec::dma::ClockControlLayout;
use xdna_archspec::types::TileKind;

use crate::device::regdb::device_reg_layout;

/// Which module within a tile is being queried.  Mirrors the bit-field
/// breakdown of Module_Clock_Control in AM025.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModuleKind {
    Core,
    Memory,
    Dma,
    StreamSwitch,
}

/// Per-tile module-gate state.  Stores the raw register value(s); the
/// per-module active queries decode bit fields on demand.
#[derive(Debug, Clone)]
struct TileGates {
    /// `Module_Clock_Control` (compute, memtile) or `Module_Clock_Control_0` (shim).
    raw_mcc_0: u32,
    /// `Module_Clock_Control_1` (shim only); None for other tile types.
    raw_mcc_1: Option<u32>,
}

/// Per-tile adaptive-gate state.  Tracks idle cycles separately for
/// the DMA and stream switch subsystems.  A gate engages when its idle
/// cycles reach 2^abort_period (AM025 supported range 3-12).
///
/// An abort period of `None` means nothing has configured it, so the
/// register database's reset value applies (7, i.e. 128 cycles).
#[derive(Debug, Clone, Default)]
struct AdaptiveState {
    dma_idle_cycles: u32,
    ss_idle_cycles: u32,
    /// Stream-switch gate threshold exponent, from the
    /// Stream_Switch_Adaptive_Clock_Gate_Abort_Period register.
    ss_abort_period: Option<u8>,
    /// DMA gate threshold exponent.  AM025 defines no register for it; only
    /// `set_adaptive_abort_period` sets it.
    dma_abort_period: Option<u8>,
}

/// Per-array clock-gating state.  Single source of truth for all
/// column / module / adaptive gates.
#[derive(Debug, Clone)]
pub struct ClockController {
    /// Per-column clock gate; index = col.  `false` = gated (inactive).
    columns: Vec<bool>,
    /// Number of rows in the array.  Stored so `ungate_all()` knows how
    /// far to iterate without needing to consult arch config.
    num_rows: u8,
    /// Per-tile module-gate state, keyed by (col, row).
    /// Tile entry is absent until first write to that tile's MCC register;
    /// is_module_active falls back to the AM025 reset values in that case
    /// (consistent with what a read of the register would return).
    tiles: HashMap<(u8, u8), TileGates>,
    /// Per-tile adaptive-gate idle counters and abort_period config.
    /// Entry created lazily on first tick or set_adaptive_abort_period.
    adaptive: HashMap<(u8, u8), AdaptiveState>,
    /// Set of (col, row, offset) tuples that have already been warned
    /// about as accesses to a gated tile.  Used to dedup the warning
    /// emitted by `warn_gated_access` so a single bug pattern does not
    /// flood the log on every cycle.
    warned_sites: HashSet<(u8, u8, u32)>,
}

/// AM025 offset for Column_Clock_Control (shim tiles only).
pub const COLUMN_CLOCK_CONTROL_OFFSET: u32 = 0x000FFF20;

/// AM025 offset for AIE_Tile_Column_Reset (shim tiles only).
pub const AIE_TILE_COLUMN_RESET_OFFSET: u32 = 0x000F_FF28;

/// Shim registers the host drives while the column clock is gated, so an
/// access to them is not the gated-access bug pattern. aie-rt's partition
/// init and teardown (`_XAie_PrivilegeInitPart`,
/// `_XAie_PrivilegeTeardownPart`) disable every column clock buffer, then
/// assert AIE_Tile_Column_Reset ("Gate all tiles before resetting columns
/// to quiet traffic"), and only then re-enable the clocks.
const REACHABLE_WHILE_GATED_SHIM_OFFSETS: [u32; 1] = [AIE_TILE_COLUMN_RESET_OFFSET];

/// AM025 offset for Module_Clock_Control_1 (shim only, NoC enable).
const MCC_SHIM_1_OFFSET: u32 = 0x000FFF04;

/// Internal tile-kind discriminator for bit-layout selection.
/// Uses a clock-control-local enum rather than xdna_archspec::types::TileKind
/// to avoid coupling the controller to the broader archspec model -- this
/// only needs three buckets (Compute / Memtile / Shim) regardless of how
/// many shim variants exist elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClockTileKind {
    Compute,
    Memtile,
    Shim,
}

fn tile_kind_layout(row: u8) -> &'static ClockControlLayout {
    clock_tile_kind_from_row(row).layout()
}

fn clock_tile_kind_from_row(row: u8) -> ClockTileKind {
    if row == 0 {
        ClockTileKind::Shim
    } else if row == 1 {
        ClockTileKind::Memtile
    } else {
        ClockTileKind::Compute
    }
}

impl ClockTileKind {
    fn archspec_kind(self) -> TileKind {
        match self {
            ClockTileKind::Compute => TileKind::Compute,
            ClockTileKind::Memtile => TileKind::Mem,
            ClockTileKind::Shim => TileKind::ShimNoc,
        }
    }

    /// This tile kind's clock-control registers, from the register database.
    fn layout(self) -> &'static ClockControlLayout {
        device_reg_layout().clock_control(self.archspec_kind())
    }
}

/// AM025 reset value for Module_Clock_Control (and MCC_0 for shim).
fn reset_value_for_mcc(tile_kind: ClockTileKind) -> u32 {
    tile_kind.layout().module_clock_control_reset
}

/// AM025 reset value for shim Module_Clock_Control_1 (NoC enable bit set).
fn reset_value_for_mcc_1() -> u32 {
    device_reg_layout()
        .reset_value(TileKind::ShimNoc, MCC_SHIM_1_OFFSET)
        .expect("register database has no reset value for the shim Module_Clock_Control_1")
}

/// The clock-control registers a tile decodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClockRegister {
    /// `Column_Clock_Control` (shim row only).
    Column,
    /// `Module_Clock_Control` (shim: `Module_Clock_Control_0`).
    ModuleControl0,
    /// `Module_Clock_Control_1` (shim only).
    ModuleControl1,
    /// `Stream_Switch_Adaptive_Clock_Gate_Abort_Period`.
    AbortPeriod,
    /// `Stream_Switch_Adaptive_Clock_Gate_Status` (read-only).
    Status,
}

impl ClockRegister {
    fn decode(row: u8, offset: u32) -> Option<Self> {
        let tile_kind = clock_tile_kind_from_row(row);
        let layout = tile_kind.layout();
        if offset == layout.module_clock_control {
            Some(Self::ModuleControl0)
        } else if offset == layout.abort_period_register {
            Some(Self::AbortPeriod)
        } else if offset == layout.status_register {
            Some(Self::Status)
        } else if tile_kind == ClockTileKind::Shim && offset == COLUMN_CLOCK_CONTROL_OFFSET {
            Some(Self::Column)
        } else if tile_kind == ClockTileKind::Shim && offset == MCC_SHIM_1_OFFSET {
            Some(Self::ModuleControl1)
        } else {
            None
        }
    }
}

/// Decode `kind` from a Module_Clock_Control register value.
/// Bit positions confirmed from aie_registers_aie2.json on 2026-05-24.
///
/// | ModuleKind    | Compute      | Memtile      | Shim              |
/// |---------------|--------------|--------------|-------------------|
/// | Core          | MCC bit 2    | always false | always false      |
/// | Memory        | MCC bit 1    | MCC bit 1    | always false      |
/// | Dma           | MCC bit 1    | MCC bit 1    | MCC_1 bit 0 (NoC) |
/// | StreamSwitch  | MCC bit 0    | MCC bit 0    | MCC_0 bit 0       |
///
/// On compute/memtile Dma and Memory share the same physical bit -- the
/// silicon does not separately clock-gate DMA from data memory.
fn mcc_module_active(
    raw_mcc_0: u32,
    raw_mcc_1: Option<u32>,
    tile_kind: ClockTileKind,
    kind: ModuleKind,
) -> bool {
    use ClockTileKind::*;
    use ModuleKind::*;
    let (reg, bit): (u32, u8) = match (tile_kind, kind) {
        // Compute tile: MCC bit 2 = Core, bit 1 = Memory (= Dma), bit 0 = SS.
        (Compute, Core) => (raw_mcc_0, 2),
        (Compute, Memory) => (raw_mcc_0, 1),
        (Compute, Dma) => (raw_mcc_0, 1), // same bit as Memory
        (Compute, StreamSwitch) => (raw_mcc_0, 0),
        // Memtile: no Core; bit 1 = Memory (= Dma), bit 0 = SS.
        (Memtile, Core) => return false,
        (Memtile, Memory) => (raw_mcc_0, 1),
        (Memtile, Dma) => (raw_mcc_0, 1),
        (Memtile, StreamSwitch) => (raw_mcc_0, 0),
        // Shim: no Core / Memory.  Dma lives in MCC_1 bit 0 (NoC module).
        (Shim, Core) => return false,
        (Shim, Memory) => return false,
        (Shim, Dma) => match raw_mcc_1 {
            Some(r) => (r, 0),
            None => return false,
        },
        (Shim, StreamSwitch) => (raw_mcc_0, 0),
    };
    (reg >> bit) & 0x1 != 0
}

impl ClockController {
    /// Construct a controller for an array of `num_cols` columns and
    /// `num_rows` rows.  All columns boot gated (silicon-accurate).
    pub fn new(num_cols: u8, num_rows: u8) -> Self {
        Self {
            columns: vec![false; num_cols as usize],
            num_rows,
            tiles: HashMap::new(),
            adaptive: HashMap::new(),
            warned_sites: HashSet::new(),
        }
    }

    /// Returns true iff column `col` has its clock enabled.
    /// Returns false for out-of-range columns.
    pub fn is_column_active(&self, col: u8) -> bool {
        self.columns.get(col as usize).copied().unwrap_or(false)
    }

    /// Returns true iff the named module on tile (col, row) is currently
    /// clocked.  Column gate dominates: a gated column means every module
    /// reports inactive regardless of MCC.  An ungated column with no MCC
    /// writes yet uses the AM025 reset value.
    pub fn is_module_active(&self, col: u8, row: u8, kind: ModuleKind) -> bool {
        if !self.is_column_active(col) {
            return false;
        }
        let tile_kind = clock_tile_kind_from_row(row);
        let (raw_mcc_0, raw_mcc_1) = match self.tiles.get(&(col, row)) {
            Some(gates) => (gates.raw_mcc_0, gates.raw_mcc_1),
            None => (
                reset_value_for_mcc(tile_kind),
                if matches!(tile_kind, ClockTileKind::Shim) {
                    Some(reset_value_for_mcc_1())
                } else {
                    None
                },
            ),
        };
        mcc_module_active(raw_mcc_0, raw_mcc_1, tile_kind, kind)
    }

    /// Returns true iff any module on this tile is active.
    pub fn is_tile_active(&self, col: u8, row: u8) -> bool {
        use ModuleKind::*;
        self.is_module_active(col, row, Core)
            || self.is_module_active(col, row, Memory)
            || self.is_module_active(col, row, Dma)
            || self.is_module_active(col, row, StreamSwitch)
    }

    /// Advance per-tile adaptive-gate idle counters.  Activity in DMA or
    /// SS resets the corresponding counter; idleness advances it.  Once
    /// a counter reaches 2^abort_period the adaptive gate engages.
    ///
    /// This is the raw, two-counter form used by existing tests and by
    /// higher-level callers that have computed both activity flags together.
    /// The step loop uses `tick_adaptive_dma` and `tick_adaptive_ss` instead
    /// so it can skip a gated module's counter without touching the other.
    pub fn tick_adaptive(&mut self, col: u8, row: u8, dma_active: bool, ss_active: bool) {
        let entry = self.adaptive.entry((col, row)).or_default();
        if dma_active {
            entry.dma_idle_cycles = 0;
        } else {
            entry.dma_idle_cycles = entry.dma_idle_cycles.saturating_add(1);
        }
        if ss_active {
            entry.ss_idle_cycles = 0;
        } else {
            entry.ss_idle_cycles = entry.ss_idle_cycles.saturating_add(1);
        }
    }

    /// Advance only the DMA adaptive-gate idle counter.
    ///
    /// Called from `step_data_movement` when the DMA module is ungated.
    /// Keeps the SS counter frozen (a gated SS module's counter should not
    /// advance -- silicon does not clock a gated module's idle detector).
    pub fn tick_adaptive_dma(&mut self, col: u8, row: u8, active: bool) {
        let entry = self.adaptive.entry((col, row)).or_default();
        if active {
            entry.dma_idle_cycles = 0;
        } else {
            entry.dma_idle_cycles = entry.dma_idle_cycles.saturating_add(1);
        }
    }

    /// Advance only the SS adaptive-gate idle counter.
    ///
    /// Called from `step_data_movement` when the StreamSwitch module is ungated.
    /// Keeps the DMA counter frozen (a gated DMA module's counter should not
    /// advance -- silicon does not clock a gated module's idle detector).
    pub fn tick_adaptive_ss(&mut self, col: u8, row: u8, active: bool) {
        let entry = self.adaptive.entry((col, row)).or_default();
        if active {
            entry.ss_idle_cycles = 0;
        } else {
            entry.ss_idle_cycles = entry.ss_idle_cycles.saturating_add(1);
        }
    }

    /// Reset the DMA adaptive idle counter for this tile.
    ///
    /// Called from emit sites where silicon would generate a wake event for the
    /// DMA module: register writes targeting DMA/lock space, lock-value changes
    /// reaching a tile with channels in `AcquiringLock`. Equivalent in effect to
    /// `tick_adaptive_dma(active=true)` but reads more clearly at call sites
    /// that want the wake semantics rather than the per-cycle accounting form.
    pub fn wake_adaptive_dma(&mut self, col: u8, row: u8) {
        let entry = self.adaptive.entry((col, row)).or_default();
        entry.dma_idle_cycles = 0;
    }

    /// Reset the SS adaptive idle counter for this tile.
    ///
    /// Called from emit sites where silicon would generate a wake event for the
    /// StreamSwitch module: stream beats arriving at a slave port of this tile,
    /// SS configuration register writes. Same call-site-clarity rationale as
    /// `wake_adaptive_dma`.
    pub fn wake_adaptive_ss(&mut self, col: u8, row: u8) {
        let entry = self.adaptive.entry((col, row)).or_default();
        entry.ss_idle_cycles = 0;
    }

    /// Reset the DMA idle counter for every tile in `col` to zero.
    ///
    /// Internal helper called when a column transitions from gated to ungated.
    fn reset_dma_counters_for_column(&mut self, col: u8) {
        for row in 0..self.num_rows {
            let entry = self.adaptive.entry((col, row)).or_default();
            entry.dma_idle_cycles = 0;
        }
    }

    /// Reset the SS idle counter for every tile in `col` to zero.
    ///
    /// Internal helper called when a column transitions from gated to ungated.
    fn reset_ss_counters_for_column(&mut self, col: u8) {
        for row in 0..self.num_rows {
            let entry = self.adaptive.entry((col, row)).or_default();
            entry.ss_idle_cycles = 0;
        }
    }

    /// Reset both adaptive idle counters (DMA + SS) for every tile in a
    /// column. Called on `AIE_Tile_Column_Reset` -- the adaptive counters
    /// are tile-resident state cleared by the column reset, unlike the
    /// shim-resident clock-gate enable bits which the reset leaves intact.
    pub(crate) fn reset_adaptive_counters_for_column(&mut self, col: u8) {
        self.reset_dma_counters_for_column(col);
        self.reset_ss_counters_for_column(col);
    }

    /// Configure the abort_period (= 2^N idle-cycle threshold) for both
    /// adaptive gates on a tile.  AM025 supported range is 3-12; values
    /// outside that produce undefined hardware behavior, but the
    /// emulator does not enforce the range.
    ///
    /// The register bus sets only the stream-switch threshold (see
    /// `write_register`); this API-level setter moves both.
    pub fn set_adaptive_abort_period(&mut self, col: u8, row: u8, period_2pow: u8) {
        let entry = self.adaptive.entry((col, row)).or_default();
        entry.ss_abort_period = Some(period_2pow);
        entry.dma_abort_period = Some(period_2pow);
    }

    /// The raw Module_Clock_Control value of a tile: what was written, else the reset value.
    fn raw_mcc_0(&self, col: u8, row: u8) -> u32 {
        match self.tiles.get(&(col, row)) {
            Some(gates) => gates.raw_mcc_0,
            None => reset_value_for_mcc(clock_tile_kind_from_row(row)),
        }
    }

    /// Whether the stream-switch adaptive gate is enabled (Module_Clock_Control
    /// `Stream_Switch_Adaptive_Clock_Gate`).  Database-derived, silicon check pending.
    pub fn is_adaptive_ss_enabled(&self, col: u8, row: u8) -> bool {
        clock_tile_kind_from_row(row)
            .layout()
            .stream_switch_adaptive_enable
            .extract_bool(self.raw_mcc_0(col, row))
    }

    /// Whether the DMA adaptive gate is enabled (Module_Clock_Control
    /// `DMA_Adaptive_Clock_Gate`).  Database-derived, silicon check pending.
    pub fn is_adaptive_dma_enabled(&self, col: u8, row: u8) -> bool {
        clock_tile_kind_from_row(row)
            .layout()
            .dma_adaptive_enable
            .extract_bool(self.raw_mcc_0(col, row))
    }

    /// Idle-cycle count at which a gate with exponent `period` engages: 2^period.
    /// Applied as written for every value the 4-bit field can hold, including
    /// those outside AM025's documented 3-12.
    fn threshold(period: Option<u8>, row: u8) -> u32 {
        let exponent =
            period.unwrap_or_else(|| clock_tile_kind_from_row(row).layout().abort_period_reset as u8);
        1u32.checked_shl(exponent as u32).unwrap_or(u32::MAX)
    }

    /// True iff `offset` is this tile's stream-switch Abort_Period register.  The
    /// register sits in the stream-switch address range, so the register bus
    /// routes it by offset rather than by subsystem.
    pub fn is_abort_period_register(&self, row: u8, offset: u32) -> bool {
        ClockRegister::decode(row, offset) == Some(ClockRegister::AbortPeriod)
    }

    /// Returns true iff the DMA adaptive gate is currently engaged
    /// (idle cycle counter has crossed 2^abort_period).
    ///
    /// Consumed by `step_dma` and `step_all_dma` to skip execution on
    /// engaged tiles.  Silicon stops the clocked domain on engagement
    /// and resumes it on external wake events; the emulator implements
    /// the wake paths through `wake_adaptive_dma`:
    ///
    /// - Register-bus accesses (DMA, Lock, DataMemory) wake via the
    ///   dispatcher in `DeviceState::wake_adaptive_for_subsystem`.
    /// - Stream beats arriving at a slave port set `cycle_active`,
    ///   which Phase 5 of `step_data_movement` converts to
    ///   `tick_adaptive_ss(active=true)` (Wake 2 -- SS counter, but the
    ///   same pattern covers DMA via the per-cycle ss_active /
    ///   dma_active checks).
    /// - Cross-tile lock changes arrive as control-packet writes, which
    ///   take the register-bus path above (Wake 3 reduces to Wake 1).
    ///
    /// History: consumption deferred 2026-05-25 (490a6b1), re-enabled
    /// after wake-on-event coverage shipped.  See
    /// xdna-emu-priv: research/coverage/cycle-accuracy-mission.md item #8.
    pub fn is_adaptive_dma_engaged(&self, col: u8, row: u8) -> bool {
        let Some(s) = self.adaptive.get(&(col, row)) else {
            return false;
        };
        self.is_adaptive_dma_enabled(col, row)
            && s.dma_idle_cycles >= Self::threshold(s.dma_abort_period, row)
    }

    /// Returns true iff the SS adaptive gate is currently engaged.
    /// Consumed by `step_tile_switches`; same wake-on-event guarantees as
    /// `is_adaptive_dma_engaged` -- see that docstring for the full path
    /// enumeration.
    pub fn is_adaptive_ss_engaged(&self, col: u8, row: u8) -> bool {
        let Some(s) = self.adaptive.get(&(col, row)) else {
            return false;
        };
        self.is_adaptive_ss_enabled(col, row) && s.ss_idle_cycles >= Self::threshold(s.ss_abort_period, row)
    }

    /// Record a gated-tile access for warning purposes.
    ///
    /// Returns `true` if this is the first time the (col, row, offset)
    /// site has been seen while the tile was gated -- the caller is
    /// then expected to emit a log warning.  Returns `false` if the
    /// tile is currently active (no warning needed) or if this exact
    /// site has already been warned about (dedup).
    ///
    /// Per the spec, accesses to gated tiles proceed (silicon does not
    /// block them); this method is purely for surfacing the bug
    /// pattern in the emulator.  The dispatch path should still serve
    /// the access regardless of this method's return value.
    pub fn warn_gated_access(&mut self, col: u8, row: u8, offset: u32) -> bool {
        if self.is_tile_active(col, row) || (row == 0 && REACHABLE_WHILE_GATED_SHIM_OFFSETS.contains(&offset))
        {
            return false;
        }
        self.warned_sites.insert((col, row, offset))
    }

    /// Test-only accessor for the number of distinct gated-access
    /// sites this controller has recorded.
    #[cfg(test)]
    pub fn warned_sites_len(&self) -> usize {
        self.warned_sites.len()
    }

    /// Handle a register write at the given tile / offset.  Silently
    /// ignores offsets that are not clock-control registers.
    ///
    /// Re-ungate transition semantics (silicon-accurate):
    /// - Column_Clock_Control: when bit 0 transitions from 0 -> 1, reset the
    ///   DMA and SS idle counters for every tile in that column to 0.  The
    ///   counter was frozen while gated; restarting from 0 matches silicon
    ///   behavior where the idle detector sees a "fresh" clock domain.
    /// - Module_Clock_Control: when a previously-gated module bit transitions
    ///   to 1, reset the corresponding counter for that tile.  Handles
    ///   simultaneous multi-bit transitions by comparing old vs new per module.
    pub fn write_register(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        match ClockRegister::decode(row, offset) {
            Some(ClockRegister::Column) => {
                // Bit 0 = column-clock-enable; per AM025.
                let was_active = self.is_column_active(col);
                let now_active = (value & 0x1) != 0;
                if let Some(slot) = self.columns.get_mut(col as usize) {
                    *slot = now_active;
                }
                // Reset adaptive counters on gated -> ungated transition.
                if !was_active && now_active {
                    self.reset_dma_counters_for_column(col);
                    self.reset_ss_counters_for_column(col);
                }
            }
            // Each tile kind's own Module_Clock_Control; all write raw_mcc_0.
            Some(ClockRegister::ModuleControl0) => {
                let tile_kind = clock_tile_kind_from_row(row);
                // Capture old module-active state before applying the write.
                let old_raw_mcc_0 = self
                    .tiles
                    .get(&(col, row))
                    .map(|g| g.raw_mcc_0)
                    .unwrap_or_else(|| reset_value_for_mcc(tile_kind));
                let old_mcc_1 = self.tiles.get(&(col, row)).and_then(|g| g.raw_mcc_1);

                let entry = self
                    .tiles
                    .entry((col, row))
                    .or_insert_with(|| TileGates { raw_mcc_0: 0, raw_mcc_1: None });
                entry.raw_mcc_0 = value;

                // Check for module ungate transitions and reset the
                // corresponding idle counter.  Only meaningful when the
                // column is ungated -- a gated column has no clocked counters.
                if self.is_column_active(col) {
                    let was_dma = mcc_module_active(old_raw_mcc_0, old_mcc_1, tile_kind, ModuleKind::Dma);
                    let now_dma = mcc_module_active(value, old_mcc_1, tile_kind, ModuleKind::Dma);
                    if !was_dma && now_dma {
                        let entry = self.adaptive.entry((col, row)).or_default();
                        entry.dma_idle_cycles = 0;
                    }
                    let was_ss =
                        mcc_module_active(old_raw_mcc_0, old_mcc_1, tile_kind, ModuleKind::StreamSwitch);
                    let now_ss = mcc_module_active(value, old_mcc_1, tile_kind, ModuleKind::StreamSwitch);
                    if !was_ss && now_ss {
                        let entry = self.adaptive.entry((col, row)).or_default();
                        entry.ss_idle_cycles = 0;
                    }
                }
            }
            Some(ClockRegister::ModuleControl1) => {
                let tile_kind = ClockTileKind::Shim;
                let old_raw_mcc_0 = self
                    .tiles
                    .get(&(col, row))
                    .map(|g| g.raw_mcc_0)
                    .unwrap_or_else(|| reset_value_for_mcc(tile_kind));
                let old_mcc_1 = self
                    .tiles
                    .get(&(col, row))
                    .and_then(|g| g.raw_mcc_1)
                    .or_else(|| Some(reset_value_for_mcc_1()));

                let entry = self
                    .tiles
                    .entry((col, row))
                    .or_insert_with(|| TileGates { raw_mcc_0: 0, raw_mcc_1: Some(0) });
                entry.raw_mcc_1 = Some(value);

                // Shim DMA (NoC module) lives in MCC_1 bit 0.
                if self.is_column_active(col) {
                    let new_mcc_1 = Some(value);
                    let was_dma = mcc_module_active(old_raw_mcc_0, old_mcc_1, tile_kind, ModuleKind::Dma);
                    let now_dma = mcc_module_active(old_raw_mcc_0, new_mcc_1, tile_kind, ModuleKind::Dma);
                    if !was_dma && now_dma {
                        let entry = self.adaptive.entry((col, row)).or_default();
                        entry.dma_idle_cycles = 0;
                    }
                }
            }
            // Database-derived, silicon check pending: the idle threshold of the
            // stream-switch adaptive gate is 2^N for the 4-bit Abort_Period field.
            // The DMA gate has no such register in AM025 and keeps its own threshold.
            Some(ClockRegister::AbortPeriod) => {
                let period = tile_kind_layout(row).abort_period.extract(value) as u8;
                self.adaptive.entry((col, row)).or_default().ss_abort_period = Some(period);
            }
            // Read-only status; a write has no effect.
            Some(ClockRegister::Status) => {}
            None => {} // not a clock-control offset
        }
    }

    /// Test helper: enable every column and module by writing
    /// all-active patterns via the same register-write path the CDO uses.
    /// Spec: serve-and-warn policy; tests opt out of silicon-accurate
    /// boot via this helper.
    ///
    /// The row-to-kind mapping (0=Shim, 1=Memtile, 2+=Compute) is
    /// NPU1-specific; acceptable for v1 since the controller is
    /// constructed per-array and AIE2P will need its own variant.
    pub fn ungate_all(&mut self) {
        let num_cols = self.columns.len() as u8;
        let rows = self.num_rows;
        for col in 0..num_cols {
            self.write_register(col, 0, COLUMN_CLOCK_CONTROL_OFFSET, 0x1);
            for row in 0..rows {
                self.write_register(col, row, tile_kind_layout(row).module_clock_control, 0xFFFF_FFFF);
                if matches!(clock_tile_kind_from_row(row), ClockTileKind::Shim) {
                    self.write_register(col, row, MCC_SHIM_1_OFFSET, 0xFFFF_FFFF);
                }
            }
        }
    }

    /// Read a clock-control register.  Returns the current value, or the
    /// AM025 reset value if the register has not been written yet.
    /// Returns None if the offset is not a known clock-control register.
    pub fn read_register(&self, col: u8, row: u8, offset: u32) -> Option<u32> {
        let tile_kind = clock_tile_kind_from_row(row);
        let layout = tile_kind.layout();
        match ClockRegister::decode(row, offset)? {
            ClockRegister::Column => {
                // Reflect current state: bit 0 = column enabled.
                Some(if self.is_column_active(col) { 0x1 } else { 0x0 })
            }
            ClockRegister::ModuleControl0 => Some(self.raw_mcc_0(col, row)),
            ClockRegister::ModuleControl1 => Some(
                self.tiles
                    .get(&(col, row))
                    .and_then(|g| g.raw_mcc_1)
                    .unwrap_or_else(reset_value_for_mcc_1),
            ),
            ClockRegister::AbortPeriod => Some(
                self.adaptive
                    .get(&(col, row))
                    .and_then(|s| s.ss_abort_period)
                    .map_or(layout.abort_period_reset, u32::from),
            ),
            // Database-derived, silicon check pending: "1 = adaptive gating active:
            // feature enabled and switch idle for time-out, data path not clocked".
            ClockRegister::Status => {
                Some(layout.status_active.insert(0, self.is_adaptive_ss_engaged(col, row) as u32))
            }
        }
    }
}

#[cfg(test)]
mod adaptive_register_tests;
#[cfg(test)]
mod integration_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_clock_controller_boots_with_all_columns_gated() {
        let clock = ClockController::new(5, 6);
        for col in 0..5 {
            assert!(!clock.is_column_active(col), "col {} should be gated at boot", col);
        }
    }

    #[test]
    fn is_column_active_returns_false_for_out_of_range_col() {
        let clock = ClockController::new(5, 6);
        assert!(!clock.is_column_active(5));
        assert!(!clock.is_column_active(99));
    }

    #[test]
    fn write_column_clock_control_bit0_enables_column() {
        let mut clock = ClockController::new(5, 6);
        // Column 2's shim tile is at (col=2, row=0).
        // Column_Clock_Control offset is 0x000FFF20.
        clock.write_register(2, 0, 0x000FFF20, 0x1);
        assert!(clock.is_column_active(2));
        assert!(!clock.is_column_active(0), "other cols unaffected");
    }

    #[test]
    fn write_column_clock_control_bit0_clear_disables_column() {
        let mut clock = ClockController::new(5, 6);
        clock.write_register(2, 0, 0x000FFF20, 0x1);
        assert!(clock.is_column_active(2));
        clock.write_register(2, 0, 0x000FFF20, 0x0);
        assert!(!clock.is_column_active(2));
    }

    #[test]
    fn read_column_clock_control_returns_reset_before_any_write() {
        let clock = ClockController::new(5, 6);
        // AM025 reset is 0x00000000 for Column_Clock_Control.
        assert_eq!(clock.read_register(2, 0, 0x000FFF20), Some(0x00000000));
    }

    #[test]
    fn read_column_clock_control_returns_written_value() {
        let mut clock = ClockController::new(5, 6);
        clock.write_register(3, 0, 0x000FFF20, 0x1);
        assert_eq!(clock.read_register(3, 0, 0x000FFF20), Some(0x1));
    }

    #[test]
    fn module_kind_variants_exist() {
        let _ = ModuleKind::Core;
        let _ = ModuleKind::Memory;
        let _ = ModuleKind::Dma;
        let _ = ModuleKind::StreamSwitch;
    }

    #[test]
    fn is_module_active_default_false_for_all_kinds() {
        let clock = ClockController::new(5, 6);
        for kind in [ModuleKind::Core, ModuleKind::Memory, ModuleKind::Dma, ModuleKind::StreamSwitch] {
            assert!(!clock.is_module_active(2, 2, kind), "module {:?} should be gated at boot", kind);
        }
    }

    #[test]
    fn column_reset_on_a_gated_shim_is_not_a_gated_access() {
        let mut clock = ClockController::new(5, 6);
        assert!(!clock.warn_gated_access(1, 0, AIE_TILE_COLUMN_RESET_OFFSET));
        assert!(clock.warn_gated_access(1, 0, 0x0003_6030), "other shim registers still warn");
        assert!(clock.warn_gated_access(1, 2, AIE_TILE_COLUMN_RESET_OFFSET), "only the shim row is exempt");
        assert_eq!(clock.warned_sites_len(), 2);
    }

    #[test]
    fn is_tile_active_default_false() {
        let clock = ClockController::new(5, 6);
        assert!(!clock.is_tile_active(2, 2));
    }

    // ---- Task 5: Module_Clock_Control bit-field decode ----

    const MCC_COMPUTE_OFFSET: u32 = 0x00060000;
    const MCC_MEMTILE_OFFSET: u32 = 0x000FFF00;
    const MCC_SHIM_0_OFFSET: u32 = 0x000FFF00;
    const MCC_SHIM_1_OFFSET: u32 = 0x000FFF04;

    #[test]
    fn mcc_compute_decodes_core_bit() {
        // Bit positions per aie_registers_aie2.json (2026-05-24 lookup).
        let mut clock = ClockController::new(5, 6);
        clock.write_register(2, 0, 0x000FFF20, 0x1); // ungate column 2
                                                     // Compute MCC bit 2 = Core_Module_Clock_Enable.
        clock.write_register(2, 3, MCC_COMPUTE_OFFSET, 1 << 2);
        assert!(clock.is_module_active(2, 3, ModuleKind::Core));
        assert!(!clock.is_module_active(2, 3, ModuleKind::Memory));
        assert!(!clock.is_module_active(2, 3, ModuleKind::Dma));
        assert!(!clock.is_module_active(2, 3, ModuleKind::StreamSwitch));
    }

    #[test]
    fn mcc_compute_decodes_memory_bit_as_both_memory_and_dma() {
        let mut clock = ClockController::new(5, 6);
        clock.write_register(2, 0, 0x000FFF20, 0x1);
        // Compute MCC bit 1 = Memory_Module_Clock_Enable.  Same bit clocks DMA.
        clock.write_register(2, 3, MCC_COMPUTE_OFFSET, 1 << 1);
        assert!(clock.is_module_active(2, 3, ModuleKind::Memory));
        assert!(clock.is_module_active(2, 3, ModuleKind::Dma));
        assert!(!clock.is_module_active(2, 3, ModuleKind::Core));
    }

    #[test]
    fn mcc_compute_decodes_ss_bit() {
        let mut clock = ClockController::new(5, 6);
        clock.write_register(2, 0, 0x000FFF20, 0x1);
        clock.write_register(2, 3, MCC_COMPUTE_OFFSET, 1 << 0);
        assert!(clock.is_module_active(2, 3, ModuleKind::StreamSwitch));
    }

    #[test]
    fn mcc_memtile_no_core() {
        let mut clock = ClockController::new(5, 6);
        clock.write_register(2, 0, 0x000FFF20, 0x1);
        // Set everything we can on memtile.
        clock.write_register(2, 1, MCC_MEMTILE_OFFSET, 0xFFFF_FFFF);
        // Memtile has no Core module -- always false regardless of writes.
        assert!(!clock.is_module_active(2, 1, ModuleKind::Core));
        // Memory and Dma both reflect bit 1.
        assert!(clock.is_module_active(2, 1, ModuleKind::Memory));
        assert!(clock.is_module_active(2, 1, ModuleKind::Dma));
        assert!(clock.is_module_active(2, 1, ModuleKind::StreamSwitch));
    }

    #[test]
    fn mcc_shim_dma_lives_in_mcc_1_not_mcc_0() {
        let mut clock = ClockController::new(5, 6);
        clock.write_register(2, 0, 0x000FFF20, 0x1);
        // MCC_0 with everything set; MCC_1 explicitly 0.
        clock.write_register(2, 0, MCC_SHIM_0_OFFSET, 0xFFFF_FFFF);
        clock.write_register(2, 0, MCC_SHIM_1_OFFSET, 0x0);
        // SS comes from MCC_0 bit 0 -> active.
        assert!(clock.is_module_active(2, 0, ModuleKind::StreamSwitch));
        // DMA on shim comes from MCC_1 bit 0 (NoC) -> inactive.
        assert!(!clock.is_module_active(2, 0, ModuleKind::Dma));
        // Shim has no Core or Memory.
        assert!(!clock.is_module_active(2, 0, ModuleKind::Core));
        assert!(!clock.is_module_active(2, 0, ModuleKind::Memory));
    }

    #[test]
    fn mcc_shim_dma_lives_in_mcc_1_bit_0_when_set() {
        let mut clock = ClockController::new(5, 6);
        clock.write_register(2, 0, 0x000FFF20, 0x1);
        clock.write_register(2, 0, MCC_SHIM_1_OFFSET, 1 << 0);
        assert!(clock.is_module_active(2, 0, ModuleKind::Dma));
    }

    #[test]
    fn module_inactive_when_column_gated_even_with_mcc_set() {
        let mut clock = ClockController::new(5, 6);
        // Column 2 NOT ungated.
        clock.write_register(2, 3, MCC_COMPUTE_OFFSET, 1 << 2); // try to enable Core
                                                                // Column gate dominates -- everything inactive.
        assert!(!clock.is_module_active(2, 3, ModuleKind::Core));
    }

    #[test]
    fn read_mcc_returns_am025_reset_value_before_any_write() {
        let clock = ClockController::new(5, 6);
        // Compute tile reset is 0x37.
        assert_eq!(clock.read_register(2, 3, MCC_COMPUTE_OFFSET), Some(0x37));
        // Memtile reset is 0x33.
        assert_eq!(clock.read_register(2, 1, MCC_MEMTILE_OFFSET), Some(0x33));
        // Shim Module_Clock_Control_0 reset is 0x3B.
        assert_eq!(clock.read_register(2, 0, MCC_SHIM_0_OFFSET), Some(0x3B));
        // Shim Module_Clock_Control_1 reset is 0x01.
        assert_eq!(clock.read_register(2, 0, MCC_SHIM_1_OFFSET), Some(0x01));
    }

    // ---- Task 6: AdaptiveState + tick_adaptive ----

    #[test]
    fn adaptive_gate_default_disengaged_on_fresh_tile() {
        let clock = ClockController::new(5, 6);
        // No writes yet; adaptive state is "permissive" because the
        // tile is also clock-gated (default).  Once ungated and ticked,
        // the adaptive gate will engage on sustained idle.
        assert!(!clock.is_adaptive_dma_engaged(2, 2));
        assert!(!clock.is_adaptive_ss_engaged(2, 2));
    }

    #[test]
    fn adaptive_dma_engages_after_idle_cycles() {
        let mut clock = ClockController::new(5, 6);
        // Set abort_period = 3 (engage after 2^3 = 8 idle cycles).
        clock.set_adaptive_abort_period(2, 2, 3);
        for _ in 0..8 {
            clock.tick_adaptive(2, 2, /*dma_active=*/ false, /*ss_active=*/ false);
        }
        assert!(clock.is_adaptive_dma_engaged(2, 2));
    }

    #[test]
    fn adaptive_dma_resets_on_activity() {
        let mut clock = ClockController::new(5, 6);
        clock.set_adaptive_abort_period(2, 2, 3);
        for _ in 0..7 {
            clock.tick_adaptive(2, 2, false, false);
        }
        // One active cycle resets the counter.
        clock.tick_adaptive(2, 2, true, false);
        assert!(!clock.is_adaptive_dma_engaged(2, 2));
    }

    #[test]
    fn ungate_all_makes_every_column_and_module_active() {
        let mut clock = ClockController::new(5, 6);
        // Pre: everything gated.
        assert!(!clock.is_column_active(0));
        assert!(!clock.is_module_active(2, 3, ModuleKind::Core));
        // Ungate all.
        clock.ungate_all();
        // Post: every column active.
        for col in 0..5 {
            assert!(clock.is_column_active(col));
        }
        // Every module that physically exists on a tile-kind reports
        // active.  Per Task 5 semantics (matches AM025): shim has only
        // StreamSwitch + Dma; memtile lacks Core; compute has all four.
        // Iterate that matrix explicitly so this test does not mask
        // future bugs by tolerating false-where-true-was-expected.
        for col in 0..5 {
            // row 0 = shim
            assert!(clock.is_module_active(col, 0, ModuleKind::StreamSwitch));
            assert!(clock.is_module_active(col, 0, ModuleKind::Dma));
            // row 1 = memtile
            assert!(clock.is_module_active(col, 1, ModuleKind::Memory));
            assert!(clock.is_module_active(col, 1, ModuleKind::Dma));
            assert!(clock.is_module_active(col, 1, ModuleKind::StreamSwitch));
            // rows 2..6 = compute
            for row in 2..6 {
                for kind in [ModuleKind::Core, ModuleKind::Memory, ModuleKind::Dma, ModuleKind::StreamSwitch]
                {
                    assert!(
                        clock.is_module_active(col, row, kind),
                        "tile ({}, {}) module {:?} should be active after ungate_all",
                        col,
                        row,
                        kind
                    );
                }
            }
        }
    }

    #[test]
    fn adaptive_ss_independent_from_dma() {
        let mut clock = ClockController::new(5, 6);
        clock.set_adaptive_abort_period(2, 2, 3);
        // DMA active, SS idle.
        for _ in 0..8 {
            clock.tick_adaptive(2, 2, /*dma=*/ true, /*ss=*/ false);
        }
        assert!(!clock.is_adaptive_dma_engaged(2, 2), "DMA active -> DMA gate stays disengaged");
        assert!(clock.is_adaptive_ss_engaged(2, 2), "SS idle long enough -> SS gate engages");
    }

    // ---- tick_adaptive_dma / tick_adaptive_ss split methods ----

    #[test]
    fn tick_adaptive_dma_only_advances_dma_counter() {
        let mut clock = ClockController::new(5, 6);
        clock.set_adaptive_abort_period(2, 2, 3);
        for _ in 0..8 {
            clock.tick_adaptive_dma(2, 2, false);
        }
        assert!(clock.is_adaptive_dma_engaged(2, 2), "DMA counter must engage after threshold idle cycles");
        // SS counter was never ticked; must remain disengaged.
        assert!(!clock.is_adaptive_ss_engaged(2, 2), "SS counter untouched by tick_adaptive_dma");
    }

    #[test]
    fn tick_adaptive_ss_only_advances_ss_counter() {
        let mut clock = ClockController::new(5, 6);
        clock.set_adaptive_abort_period(2, 2, 3);
        for _ in 0..8 {
            clock.tick_adaptive_ss(2, 2, false);
        }
        assert!(clock.is_adaptive_ss_engaged(2, 2), "SS counter must engage after threshold idle cycles");
        // DMA counter was never ticked; must remain disengaged.
        assert!(!clock.is_adaptive_dma_engaged(2, 2), "DMA counter untouched by tick_adaptive_ss");
    }

    #[test]
    fn tick_adaptive_dma_active_resets_counter() {
        let mut clock = ClockController::new(5, 6);
        clock.set_adaptive_abort_period(2, 2, 3);
        for _ in 0..7 {
            clock.tick_adaptive_dma(2, 2, false);
        }
        clock.tick_adaptive_dma(2, 2, true); // active -- resets counter
        assert!(!clock.is_adaptive_dma_engaged(2, 2), "active tick must reset DMA counter");
    }

    #[test]
    fn tick_adaptive_ss_active_resets_counter() {
        let mut clock = ClockController::new(5, 6);
        clock.set_adaptive_abort_period(2, 2, 3);
        for _ in 0..7 {
            clock.tick_adaptive_ss(2, 2, false);
        }
        clock.tick_adaptive_ss(2, 2, true); // active -- resets counter
        assert!(!clock.is_adaptive_ss_engaged(2, 2), "active tick must reset SS counter");
    }

    // ---- wake_adaptive_dma / wake_adaptive_ss ----

    #[test]
    fn wake_adaptive_dma_resets_engaged_counter() {
        let mut clock = ClockController::new(5, 6);
        clock.set_adaptive_abort_period(2, 2, 3);
        for _ in 0..8 {
            clock.tick_adaptive_dma(2, 2, false);
        }
        assert!(clock.is_adaptive_dma_engaged(2, 2), "precondition: gate engaged");
        clock.wake_adaptive_dma(2, 2);
        assert!(!clock.is_adaptive_dma_engaged(2, 2), "wake must release DMA gate");
    }

    #[test]
    fn wake_adaptive_ss_resets_engaged_counter() {
        let mut clock = ClockController::new(5, 6);
        clock.set_adaptive_abort_period(2, 2, 3);
        for _ in 0..8 {
            clock.tick_adaptive_ss(2, 2, false);
        }
        assert!(clock.is_adaptive_ss_engaged(2, 2), "precondition: gate engaged");
        clock.wake_adaptive_ss(2, 2);
        assert!(!clock.is_adaptive_ss_engaged(2, 2), "wake must release SS gate");
    }

    #[test]
    fn wake_adaptive_dma_does_not_affect_ss_counter() {
        let mut clock = ClockController::new(5, 6);
        clock.set_adaptive_abort_period(2, 2, 3);
        for _ in 0..8 {
            clock.tick_adaptive_dma(2, 2, false);
            clock.tick_adaptive_ss(2, 2, false);
        }
        assert!(clock.is_adaptive_dma_engaged(2, 2));
        assert!(clock.is_adaptive_ss_engaged(2, 2));
        clock.wake_adaptive_dma(2, 2);
        assert!(!clock.is_adaptive_dma_engaged(2, 2));
        assert!(clock.is_adaptive_ss_engaged(2, 2), "wake_adaptive_dma must leave SS counter alone");
    }

    #[test]
    fn wake_adaptive_ss_does_not_affect_dma_counter() {
        let mut clock = ClockController::new(5, 6);
        clock.set_adaptive_abort_period(2, 2, 3);
        for _ in 0..8 {
            clock.tick_adaptive_dma(2, 2, false);
            clock.tick_adaptive_ss(2, 2, false);
        }
        clock.wake_adaptive_ss(2, 2);
        assert!(clock.is_adaptive_dma_engaged(2, 2), "wake_adaptive_ss must leave DMA counter alone");
        assert!(!clock.is_adaptive_ss_engaged(2, 2));
    }

    #[test]
    fn wake_adaptive_creates_entry_for_fresh_tile() {
        // First wake on a tile that has never been ticked should not panic and
        // should leave the gate disengaged (counter at 0, threshold at 2^7=128).
        let mut clock = ClockController::new(5, 6);
        clock.wake_adaptive_dma(3, 4);
        clock.wake_adaptive_ss(3, 4);
        assert!(!clock.is_adaptive_dma_engaged(3, 4));
        assert!(!clock.is_adaptive_ss_engaged(3, 4));
    }

    // ---- reset-on-re-ungate (column) ----

    #[test]
    fn column_ungate_resets_adaptive_counters() {
        let mut clock = ClockController::new(5, 6);
        // Force the DMA counter near engagement while still column-gated
        // (tick_adaptive bypasses the column gate -- tests call raw form).
        clock.set_adaptive_abort_period(2, 3, 3);
        for _ in 0..7 {
            clock.tick_adaptive_dma(2, 3, false);
            clock.tick_adaptive_ss(2, 3, false);
        }
        // Pre-check: counters are advancing (not yet at threshold).
        {
            let s = clock.adaptive.get(&(2, 3)).expect("adaptive entry created");
            assert_eq!(s.dma_idle_cycles, 7);
            assert_eq!(s.ss_idle_cycles, 7);
        }
        // Ungate: write Column_Clock_Control bit 0 = 1 on the shim row.
        clock.write_register(2, 0, COLUMN_CLOCK_CONTROL_OFFSET, 0x1);
        // Both counters must be reset to 0.
        let s = clock
            .adaptive
            .get(&(2, 3))
            .expect("adaptive entry must still exist after ungate");
        assert_eq!(s.dma_idle_cycles, 0, "column ungate must reset dma_idle_cycles");
        assert_eq!(s.ss_idle_cycles, 0, "column ungate must reset ss_idle_cycles");
    }

    #[test]
    fn column_ungate_does_not_reset_counters_on_already_active_column() {
        let mut clock = ClockController::new(5, 6);
        // Ungate column first.
        clock.write_register(2, 0, COLUMN_CLOCK_CONTROL_OFFSET, 0x1);
        // Advance counters.
        clock.set_adaptive_abort_period(2, 3, 3);
        for _ in 0..5 {
            clock.tick_adaptive_dma(2, 3, false);
            clock.tick_adaptive_ss(2, 3, false);
        }
        // Write column active again (no state transition -- was already active).
        clock.write_register(2, 0, COLUMN_CLOCK_CONTROL_OFFSET, 0x1);
        // Counters must NOT be reset -- no transition.
        let s = clock.adaptive.get(&(2, 3)).expect("adaptive entry present");
        assert_eq!(s.dma_idle_cycles, 5, "re-write to already-active column must not reset DMA counter");
        assert_eq!(s.ss_idle_cycles, 5, "re-write to already-active column must not reset SS counter");
    }

    // ---- reset-on-re-ungate (module) ----

    #[test]
    fn module_ungate_resets_adaptive_counter() {
        let mut clock = ClockController::new(5, 6);
        // Ungate the column first so MCC writes are meaningful.
        clock.write_register(2, 0, COLUMN_CLOCK_CONTROL_OFFSET, 0x1);
        // Gate the DMA module (bit 1 = 0) by writing MCC with only SS active.
        clock.write_register(2, 3, MCC_COMPUTE_OFFSET, 0b001); // bit 0 = SS, bit 1 = Mem/DMA off
                                                               // Advance DMA idle counter directly (test API bypasses gate check).
        clock.set_adaptive_abort_period(2, 3, 3);
        for _ in 0..6 {
            clock.tick_adaptive_dma(2, 3, false);
        }
        {
            let s = clock.adaptive.get(&(2, 3)).expect("entry exists");
            assert_eq!(s.dma_idle_cycles, 6);
        }
        // Ungate DMA module (write bit 1 = 1).
        clock.write_register(2, 3, MCC_COMPUTE_OFFSET, 0b011); // bit 1 = Mem/DMA, bit 0 = SS
                                                               // DMA counter must be reset; SS counter untouched.
        let s = clock.adaptive.get(&(2, 3)).expect("entry still exists");
        assert_eq!(s.dma_idle_cycles, 0, "DMA module ungate must reset dma_idle_cycles");
        // SS counter was not reset (SS was already active before).
        assert_eq!(s.ss_idle_cycles, 0, "SS counter should still be 0 (never ticked)");
    }
}
