//! In-process NPU1 cluster VCD mapping tree.
//!
//! The in-process aiesim bridge (`aiesim-bridge/`) drives the MSM cluster model
//! directly and dumps a VCD via `add_sc_traces` (#87). That waveform is in
//! native NPU1 geometry -- the same coordinates as the trace BO / real hardware,
//! so no vc2802 row remap is needed -- which makes it the timing oracle for the
//! three-way calibration (#84) and the future interp<->aiesim diff (#86).
//!
//! # Why this is a third tree
//!
//! The in-process MSM VCD is its own layout, distinct from both
//! [`build_aie2_mapping_tree`](super::mapping::build_aie2_mapping_tree) (the
//! idealized NPU1 tree) and
//! [`build_vc2802_mapping_tree`](super::mapping::build_vc2802_mapping_tree) (the
//! standalone aiesimulator tree):
//!
//! - Scope root is `aiesim_top.math_engine`.
//! - Geometry: 5 columns (0-4), shim row 0, mem row 1, **array rows 2-7** (the
//!   MSM model emits 6 compute rows; NPU1 hardware populates 4, rows 2-5).
//! - The model dumps its *full internal signal set* -- ~39k signals / ~2.8k
//!   distinct templates, ~5x the idealized tree. The compute core alone
//!   (`cm.proc`) exposes the entire ISS internals (bus lanes, adaptors), none
//!   of which has an interpreter analog.
//!
//! # Typed vs raw (the design)
//!
//! See `docs/coverage/inproc-mapping-tree-design.md`. Signals split two ways:
//!
//! - **Typed tier** -- the seven subsystems with a real cross-source comparison
//!   partner (lock, dma, stream, core, memory, event, perf) map to the existing
//!   [`StatePath`](super::state_path::StatePath) variants. Most reuse the shared
//!   subsystem mappings directly, since the MSM signal names match aiesimulator
//!   ground truth (the same names the existing mappings were derived from).
//! - **Raw tier** -- everything else (ISS internals, bus adaptors, interrupts,
//!   broadcast nets, tile-control, PL interface, DMA channel-internal flags)
//!   resolves to [`StatePath::Raw`](super::state_path::StatePath::Raw) so the
//!   coverage audit accounts for it (~100% resolved) while the comparison engine
//!   skips it (no partner to diff).
//!
//! Two wrappers realize the split: [`RawFallback`] makes a typed mapping cover
//! its *entire* scope (typed where known, Raw otherwise), and [`RawMapping`]
//! covers a scope with no typed mapping at all. [`NestedScopeGroup`] bundles the
//! several sub-subsystems that live under the compute tile's `cm` and `mm`
//! scopes, with the same raw fallback.

use crate::vcd::core_mapping::{proc_iss_mapping, proc_pc_mapping};
use crate::vcd::dma_mapping::{dma_mapping, shim_dma_mapping};
use crate::vcd::event_mapping::event_mapping;
use crate::vcd::lock_mapping::lock_mapping;
use crate::vcd::mapping::{MappingTree, NestedScopeMapping, SubsystemMapping, TileMapping};
use crate::vcd::state_path::{StatePath, Subsystem};
use crate::vcd::stream_mapping::{memtile_stream_mapping, shim_stream_mapping};

// ---------------------------------------------------------------------------
// RawMapping -- resolve any leaf under a scope to StatePath::Raw
// ---------------------------------------------------------------------------

/// A [`TileMapping`] that resolves *every* signal under its scope to
/// [`StatePath::Raw`]. Used for scopes with no cross-source comparison partner
/// at all -- PL interface, tile control, interrupts, broadcast nets, the
/// compute tile's degenerate `stream_switch`/`column_reset_n`.
///
/// The Raw `signal` field preserves the scope-qualified leaf name verbatim so
/// the coverage audit can report exactly what was seen.
pub struct RawMapping {
    scope: String,
    subsystem: Subsystem,
}

impl RawMapping {
    /// Create a raw mapping for `scope`, bucketing its signals under
    /// `subsystem` in the coverage report (typically [`Subsystem::Other`], or
    /// [`Subsystem::Event`] for the broadcast/interface event nets).
    pub fn new(scope: &str, subsystem: Subsystem) -> Self {
        RawMapping { scope: scope.to_string(), subsystem }
    }
}

/// Join a scope and its remaining leaf segments into a Raw `signal` name.
fn raw_signal(scope: &str, segments: &[&str]) -> String {
    if segments.is_empty() {
        scope.to_string()
    } else {
        format!("{}.{}", scope, segments.join("."))
    }
}

impl TileMapping for RawMapping {
    fn scope_name(&self) -> &str {
        &self.scope
    }

    fn resolve(&self, segments: &[&str], col: u8, row: u8) -> Option<StatePath> {
        Some(StatePath::Raw {
            col,
            row,
            subsystem: self.subsystem,
            signal: raw_signal(&self.scope, segments),
        })
    }

    fn enumerate(&self, _col: u8, _row: u8) -> Vec<StatePath> {
        // Raw signals cannot be enumerated -- their names are only known at VCD
        // parse time. They are observe-only (never emitted), so an empty
        // enumeration is correct.
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// RawFallback -- wrap a typed mapping so unknown leaves resolve to Raw
// ---------------------------------------------------------------------------

/// Wraps a typed [`TileMapping`] so that any leaf the inner mapping does not
/// recognise resolves to [`StatePath::Raw`] instead of `None`. This makes the
/// inner mapping cover its *entire* scope: typed for the comparable signals,
/// Raw for the model-internal extras (e.g. the DMA channel's `channel_running`,
/// `start_task`, `memory_starvation` flags that have no `StatePath` variant).
///
/// `scope_name` and `enumerate` delegate to the inner mapping, so the typed
/// paths are still what gets enumerated for coverage; the Raw extras are
/// observe-only.
pub struct RawFallback {
    inner: Box<dyn TileMapping>,
    subsystem: Subsystem,
}

/// Wrap `inner` so unrecognised leaves under its scope resolve to
/// `StatePath::Raw` with the given `subsystem` bucket.
pub fn raw_fallback(inner: impl TileMapping + 'static, subsystem: Subsystem) -> RawFallback {
    RawFallback { inner: Box::new(inner), subsystem }
}

impl TileMapping for RawFallback {
    fn scope_name(&self) -> &str {
        self.inner.scope_name()
    }

    fn resolve(&self, segments: &[&str], col: u8, row: u8) -> Option<StatePath> {
        if let Some(path) = self.inner.resolve(segments, col, row) {
            return Some(path);
        }
        Some(StatePath::Raw {
            col,
            row,
            subsystem: self.subsystem,
            signal: raw_signal(self.inner.scope_name(), segments),
        })
    }

    fn enumerate(&self, col: u8, row: u8) -> Vec<StatePath> {
        self.inner.enumerate(col, row)
    }
}

// ---------------------------------------------------------------------------
// NestedScopeGroup -- a scope containing several sub-subsystems, with raw fallback
// ---------------------------------------------------------------------------

/// A VCD scope that contains multiple sub-subsystem scopes, routed by the inner
/// scope name, with a raw fallback for anything unrecognised.
///
/// The compute tile's `mm` scope holds `dma`, `locks`, `dm`, `event_trace`, and
/// `performance_counter`; its `cm` scope holds the core (`proc` / `proc.iss`)
/// plus `event_trace` and internal nets. A single [`NestedScopeGroup`] models
/// either: it tries each child whose `scope_name` matches the first remaining
/// segment (in order -- the core needs two `proc` children, a flat PC mapping
/// and a nested `iss` mapping), returning the first that resolves, and falls
/// back to Raw for inner scopes no child handles.
pub struct NestedScopeGroup {
    scope: String,
    children: Vec<Box<dyn TileMapping>>,
    fallback: Subsystem,
}

impl NestedScopeGroup {
    /// Create a nested scope group named `scope` containing `children`, with a
    /// raw fallback bucketed under `fallback` for unrecognised inner scopes.
    pub fn new(scope: &str, children: Vec<Box<dyn TileMapping>>, fallback: Subsystem) -> Self {
        NestedScopeGroup { scope: scope.to_string(), children, fallback }
    }
}

impl TileMapping for NestedScopeGroup {
    fn scope_name(&self) -> &str {
        &self.scope
    }

    fn resolve(&self, segments: &[&str], col: u8, row: u8) -> Option<StatePath> {
        if !segments.is_empty() {
            let inner_scope = segments[0];
            for child in &self.children {
                if child.scope_name() == inner_scope {
                    if let Some(path) = child.resolve(&segments[1..], col, row) {
                        return Some(path);
                    }
                }
            }
        }
        // No child resolved this inner scope -> raw fallback (full coverage).
        Some(StatePath::Raw { col, row, subsystem: self.fallback, signal: raw_signal(&self.scope, segments) })
    }

    fn enumerate(&self, col: u8, row: u8) -> Vec<StatePath> {
        let mut paths = Vec::new();
        for child in &self.children {
            paths.extend(child.enumerate(col, row));
        }
        paths
    }
}

// ---------------------------------------------------------------------------
// Small typed mappings specific to the in-process layout
// ---------------------------------------------------------------------------

/// Performance-counter mapping: `performance_counter.counter_{i}` ->
/// [`StatePath::PerfCounter`]. The `counter_event_value_{i}` companions are
/// configuration shadows with no comparison value and fall to the raw tier.
pub fn inproc_perf_mapping() -> SubsystemMapping {
    // 8 covers every tile type's counter count (shim/compute 2-4, mem 8); the
    // mapping only fires for counters actually present in the VCD.
    SubsystemMapping::new("performance_counter", Subsystem::PerfCount).indexed_signal(
        "counter",
        8,
        32,
        |col, row, idx| StatePath::PerfCounter { col, row, idx },
    )
}

/// Data-memory mapping: bank-conflict detection signals map to the typed
/// [`Memory`](Subsystem::Memory) variants; the dense per-port access nets
/// (`port_DMA_read_local0_b0`, ...) are model-internal wiring and fall to the
/// raw tier via [`raw_fallback`].
///
/// - `conflict_{bank}`      -> [`StatePath::MemBankConflict`]
/// - `conflict_addr_{bank}` -> [`StatePath::MemConflictAddr`]
pub fn inproc_dm_mapping() -> SubsystemMapping {
    // 16 banks per the AIE2 mem tile; compute tiles expose fewer and only fire
    // for those present.
    SubsystemMapping::new("dm", Subsystem::Memory)
        .indexed_signal("conflict_addr", 16, 32, |col, row, bank| StatePath::MemConflictAddr {
            col,
            row,
            bank,
        })
        .indexed_signal("conflict", 16, 1, |col, row, bank| StatePath::MemBankConflict { col, row, bank })
}

// ---------------------------------------------------------------------------
// Compute tile cm / mm group builders
// ---------------------------------------------------------------------------

/// The compute tile `cm` (core module) group: core pipeline + ISS bus under
/// `proc` / `proc.iss`, the core's `event_trace`, and a raw fallback for the
/// ISS internals (`proc.iss.*` bus lanes, `proc.core_status`, `proc.dm_*`,
/// adaptors) and `event_broadcast` / `proccore_status` nets.
fn inproc_cm_group() -> NestedScopeGroup {
    NestedScopeGroup::new(
        "cm",
        vec![
            // Two children both named "proc": the flat pipeline PCs, then the
            // nested iss bus. The group tries both until one resolves.
            Box::new(proc_pc_mapping()),
            Box::new(NestedScopeMapping::new("proc", Box::new(proc_iss_mapping()))),
            Box::new(event_mapping()),
        ],
        Subsystem::Other,
    )
}

/// The compute tile `mm` (memory module) group: the 2+2 DMA, 16 locks, data
/// memory, event trace, and performance counters, each raw-wrapped so its
/// channel-internal extras resolve; a raw fallback covers `event_broadcast`.
fn inproc_mm_group() -> NestedScopeGroup {
    NestedScopeGroup::new(
        "mm",
        vec![
            Box::new(raw_fallback(dma_mapping(2, 2), Subsystem::Dma)),
            Box::new(raw_fallback(lock_mapping(16), Subsystem::Lock)),
            Box::new(raw_fallback(inproc_dm_mapping(), Subsystem::Memory)),
            Box::new(raw_fallback(event_mapping(), Subsystem::Event)),
            Box::new(raw_fallback(inproc_perf_mapping(), Subsystem::PerfCount)),
        ],
        Subsystem::Other,
    )
}

// ---------------------------------------------------------------------------
// The tree
// ---------------------------------------------------------------------------

/// Build the complete VCD signal mapping tree for the in-process NPU1 cluster.
///
/// Scope root `aiesim_top.math_engine`; 5 columns; shim row 0, mem row 1,
/// compute rows 2-7. Every per-tile subsystem signal resolves (typed where it
/// has a `StatePath` partner, Raw otherwise). A handful of top-level oddments
/// outside the per-tile hierarchy (`pm_adapt.*`, `shim_reset_n_*`,
/// `aie_ctrl_*`) remain unmapped -- ~0.25% of signals -- since they belong to
/// no tile.
pub fn build_npu1_inproc_mapping_tree() -> MappingTree {
    let cols: Vec<u8> = (0..5).collect();
    // The MSM model emits 6 compute rows (2-7); NPU1 hardware populates 4 (2-5).
    let rows_compute: Vec<u8> = (2..8).collect();

    let shim_tiles: Vec<(u8, u8)> = cols.iter().map(|&c| (c, 0)).collect();
    let mem_tiles: Vec<(u8, u8)> = cols.iter().map(|&c| (c, 1)).collect();
    let compute_tiles: Vec<(u8, u8)> =
        cols.iter().flat_map(|&c| rows_compute.iter().map(move |&r| (c, r))).collect();

    MappingTree::builder()
        // The MSM vcd_trace_file_writer emits a single flat `$scope module
        // SystemC` whose vars carry the full dotted `aiesim_top.math_engine...`
        // name, so wellen's resolved hierarchy is `SystemC.aiesim_top.
        // math_engine...`. Match that outermost SystemC scope.
        .scope("SystemC")
        .scope("aiesim_top")
        .scope("math_engine")
        // -- Shim tiles (row 0) --
        .tile_group("shim", &shim_tiles)
        .mapping(raw_fallback(lock_mapping(16), Subsystem::Lock))
        .mapping(raw_fallback(shim_dma_mapping(2, 2), Subsystem::Dma))
        .mapping(raw_fallback(shim_stream_mapping(), Subsystem::Stream))
        .mapping(raw_fallback(event_mapping(), Subsystem::Event))
        .mapping(raw_fallback(inproc_perf_mapping(), Subsystem::PerfCount))
        .mapping(RawMapping::new("pl_interface", Subsystem::Other))
        .mapping(RawMapping::new("tile_control", Subsystem::Other))
        .mapping(RawMapping::new("event_broadcast_a", Subsystem::Event))
        .mapping(RawMapping::new("event_broadcast_b", Subsystem::Event))
        .mapping(RawMapping::new("event_interface_a", Subsystem::Event))
        .mapping(RawMapping::new("event_interface_b", Subsystem::Event))
        .mapping(RawMapping::new("first_level_interrupt_a", Subsystem::Other))
        .mapping(RawMapping::new("first_level_interrupt_b", Subsystem::Other))
        .mapping(RawMapping::new("second_level_interrupt", Subsystem::Other))
        .done_tile_group()
        // -- Mem tiles (row 1) --
        .tile_group("mem_row", &mem_tiles)
        .mapping(raw_fallback(lock_mapping(64), Subsystem::Lock))
        .mapping(raw_fallback(dma_mapping(6, 6), Subsystem::Dma))
        .mapping(raw_fallback(memtile_stream_mapping(), Subsystem::Stream))
        .mapping(raw_fallback(event_mapping(), Subsystem::Event))
        .mapping(raw_fallback(inproc_perf_mapping(), Subsystem::PerfCount))
        .mapping(raw_fallback(inproc_dm_mapping(), Subsystem::Memory))
        .mapping(RawMapping::new("tile_control", Subsystem::Other))
        .mapping(RawMapping::new("event_broadcast_a", Subsystem::Event))
        .mapping(RawMapping::new("event_broadcast_b", Subsystem::Event))
        .done_tile_group()
        // -- Compute tiles (rows 2-7) --
        .tile_group("array", &compute_tiles)
        .mapping(inproc_cm_group())
        .mapping(inproc_mm_group())
        .mapping(RawMapping::new("stream_switch", Subsystem::Stream))
        .mapping(RawMapping::new("tile_control", Subsystem::Other))
        .mapping(RawMapping::new("column_reset_n", Subsystem::Other))
        .done_tile_group()
        .build()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vcd::state_path::{DmaDir, PortId};

    /// Resolve an in-process VCD name through the tree. The `name` is given
    /// without the outermost `SystemC` scope (prepended here) so the test
    /// strings read as the natural `aiesim_top.math_engine...` hierarchy.
    fn resolve(tree: &MappingTree, name: &str) -> Option<StatePath> {
        let full = format!("SystemC.{}", name);
        let segments: Vec<&str> = full.split('.').collect();
        tree.resolve(&segments)
    }

    // -- Typed-tier reuse --

    #[test]
    fn shim_lock_typed() {
        let tree = build_npu1_inproc_mapping_tree();
        assert_eq!(
            resolve(&tree, "aiesim_top.math_engine.shim.tile_0_0.locks.value_3"),
            Some(StatePath::LockValue { col: 0, row: 0, idx: 3 })
        );
    }

    #[test]
    fn memtile_lock_64_typed() {
        let tree = build_npu1_inproc_mapping_tree();
        assert_eq!(
            resolve(&tree, "aiesim_top.math_engine.mem_row.tile_4_1.locks.value_63"),
            Some(StatePath::LockValue { col: 4, row: 1, idx: 63 })
        );
    }

    #[test]
    fn memtile_dma_typed() {
        let tree = build_npu1_inproc_mapping_tree();
        // s2mm_state5 is valid for the 6-channel mem tile.
        assert_eq!(
            resolve(&tree, "aiesim_top.math_engine.mem_row.tile_2_1.dma.s2mm_state5.cur_bd"),
            Some(StatePath::DmaCurrentBd { col: 2, row: 1, dir: DmaDir::S2mm, ch: 5 })
        );
    }

    #[test]
    fn event_trace_anchor_typed() {
        // The #84 timing anchor: dma s2mm start-task event.
        let tree = build_npu1_inproc_mapping_tree();
        assert_eq!(
            resolve(
                &tree,
                "aiesim_top.math_engine.mem_row.tile_0_1.event_trace.event21_dma_s2mm_sel0_start_task"
            ),
            Some(StatePath::EventTrace {
                col: 0,
                row: 1,
                event_code: 21,
                event_name: "dma_s2mm_sel0_start_task".to_string(),
            })
        );
    }

    #[test]
    fn compute_dma_under_mm_typed() {
        let tree = build_npu1_inproc_mapping_tree();
        assert_eq!(
            resolve(&tree, "aiesim_top.math_engine.array.tile_1_3.mm.dma.s2mm_state0.cur_bd"),
            Some(StatePath::DmaCurrentBd { col: 1, row: 3, dir: DmaDir::S2mm, ch: 0 })
        );
    }

    #[test]
    fn compute_lock_under_mm_typed() {
        let tree = build_npu1_inproc_mapping_tree();
        assert_eq!(
            resolve(&tree, "aiesim_top.math_engine.array.tile_1_3.mm.locks.value_5"),
            Some(StatePath::LockValue { col: 1, row: 3, idx: 5 })
        );
    }

    #[test]
    fn compute_core_pc_under_cm_proc_typed() {
        let tree = build_npu1_inproc_mapping_tree();
        assert_eq!(
            resolve(&tree, "aiesim_top.math_engine.array.tile_2_4.cm.proc.pc_E1"),
            Some(StatePath::CorePc { col: 2, row: 4, stage: 1 })
        );
    }

    #[test]
    fn compute_core_iss_under_cm_proc_iss_typed() {
        let tree = build_npu1_inproc_mapping_tree();
        assert_eq!(
            resolve(&tree, "aiesim_top.math_engine.array.tile_2_4.cm.proc.iss.pm_rd_in"),
            Some(StatePath::CorePmData { col: 2, row: 4 })
        );
        assert_eq!(
            resolve(&tree, "aiesim_top.math_engine.array.tile_2_4.cm.proc.iss.reset"),
            Some(StatePath::CoreReset { col: 2, row: 4 })
        );
    }

    #[test]
    fn compute_event_trace_under_cm_typed() {
        let tree = build_npu1_inproc_mapping_tree();
        assert_eq!(
            resolve(&tree, "aiesim_top.math_engine.array.tile_2_4.cm.event_trace.event0_none"),
            Some(StatePath::EventTrace { col: 2, row: 4, event_code: 0, event_name: "none".to_string() })
        );
    }

    #[test]
    fn memtile_dm_conflict_typed() {
        let tree = build_npu1_inproc_mapping_tree();
        assert_eq!(
            resolve(&tree, "aiesim_top.math_engine.mem_row.tile_0_1.dm.conflict_3"),
            Some(StatePath::MemBankConflict { col: 0, row: 1, bank: 3 })
        );
        assert_eq!(
            resolve(&tree, "aiesim_top.math_engine.mem_row.tile_0_1.dm.conflict_addr_3"),
            Some(StatePath::MemConflictAddr { col: 0, row: 1, bank: 3 })
        );
    }

    #[test]
    fn perf_counter_typed() {
        let tree = build_npu1_inproc_mapping_tree();
        assert_eq!(
            resolve(&tree, "aiesim_top.math_engine.mem_row.tile_0_1.performance_counter.counter_2"),
            Some(StatePath::PerfCounter { col: 0, row: 1, idx: 2 })
        );
    }

    #[test]
    fn shim_stream_slave_event_typed() {
        let tree = build_npu1_inproc_mapping_tree();
        assert_eq!(
            resolve(&tree, "aiesim_top.math_engine.shim.tile_0_0.stream_switch.event_idle_sSouth0"),
            Some(StatePath::StreamPortIdle { col: 0, row: 0, port: PortId::named("sSouth0") })
        );
    }

    // -- Raw tier --

    #[test]
    fn dma_channel_internal_flag_is_raw() {
        // channel_running has no StatePath variant -> RawFallback catches it.
        let tree = build_npu1_inproc_mapping_tree();
        let p = resolve(&tree, "aiesim_top.math_engine.mem_row.tile_0_1.dma.s2mm_state0.channel_running");
        match p {
            Some(StatePath::Raw { col: 0, row: 1, subsystem: Subsystem::Dma, signal }) => {
                assert_eq!(signal, "dma.s2mm_state0.channel_running");
            }
            other => panic!("expected Raw(Dma), got {:?}", other),
        }
    }

    #[test]
    fn pl_interface_is_raw() {
        let tree = build_npu1_inproc_mapping_tree();
        let p = resolve(&tree, "aiesim_top.math_engine.shim.tile_0_0.pl_interface.pl_to_shim0");
        assert!(matches!(p, Some(StatePath::Raw { subsystem: Subsystem::Other, .. })));
    }

    #[test]
    fn tile_control_is_raw() {
        let tree = build_npu1_inproc_mapping_tree();
        assert!(matches!(
            resolve(&tree, "aiesim_top.math_engine.array.tile_0_2.tile_control.some_leaf"),
            Some(StatePath::Raw { .. })
        ));
    }

    #[test]
    fn compute_stream_fifo_is_raw() {
        let tree = build_npu1_inproc_mapping_tree();
        let p = resolve(&tree, "aiesim_top.math_engine.array.tile_0_2.stream_switch.fifo0_used_size");
        match p {
            Some(StatePath::Raw { subsystem: Subsystem::Stream, signal, .. }) => {
                assert_eq!(signal, "stream_switch.fifo0_used_size");
            }
            other => panic!("expected Raw(Stream), got {:?}", other),
        }
    }

    #[test]
    fn column_reset_n_leaf_is_raw() {
        let tree = build_npu1_inproc_mapping_tree();
        let p = resolve(&tree, "aiesim_top.math_engine.array.tile_0_2.column_reset_n");
        match p {
            Some(StatePath::Raw { signal, .. }) => assert_eq!(signal, "column_reset_n"),
            other => panic!("expected Raw, got {:?}", other),
        }
    }

    #[test]
    fn cm_proc_internal_is_raw() {
        // ISS bus internals under cm.proc that are not pc_E/iss-typed -> Raw.
        let tree = build_npu1_inproc_mapping_tree();
        let p = resolve(&tree, "aiesim_top.math_engine.array.tile_0_2.cm.proc.core_status.reset");
        assert!(matches!(p, Some(StatePath::Raw { subsystem: Subsystem::Other, .. })), "got {:?}", p);
    }

    #[test]
    fn cm_event_broadcast_is_raw() {
        let tree = build_npu1_inproc_mapping_tree();
        assert!(matches!(
            resolve(&tree, "aiesim_top.math_engine.array.tile_0_2.cm.event_broadcast.x"),
            Some(StatePath::Raw { .. })
        ));
    }

    // -- Geometry --

    #[test]
    fn geometry_5_cols_rows_2_to_7() {
        let tree = build_npu1_inproc_mapping_tree();
        // col 4 exists (5 columns 0-4).
        assert!(resolve(&tree, "aiesim_top.math_engine.shim.tile_4_0.locks.value_0").is_some());
        // compute row 7 exists (MSM model has 6 compute rows 2-7).
        assert!(resolve(&tree, "aiesim_top.math_engine.array.tile_0_7.cm.proc.pc_E1").is_some());
        // col 5 does not exist.
        assert!(resolve(&tree, "aiesim_top.math_engine.shim.tile_5_0.locks.value_0").is_none());
        // compute row 8 does not exist.
        assert!(resolve(&tree, "aiesim_top.math_engine.array.tile_0_8.cm.proc.pc_E1").is_none());
    }

    #[test]
    fn wrong_root_scope_returns_none() {
        let tree = build_npu1_inproc_mapping_tree();
        // The vc2802 / aie2 root "top" must not match the in-process tree.
        assert!(resolve(&tree, "top.math_engine.shim.tile_0_0.locks.value_0").is_none());
    }
}
