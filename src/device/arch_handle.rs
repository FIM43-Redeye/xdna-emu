//! Process-global cache of architecture-level seams: lock value layout,
//! stream-switch topology, ISA execute model, and instruction latency table.
//!
//! Exposes each as a fast accessor without forcing every caller to hold a
//! `&dyn ArchConfig` or a `Tile`.
//! Lazy-initialized at first call; the resolved `&'static` references stay cached
//! for the process lifetime.
//!
//! This is a pragmatic bridge: the GUI runtime arch-switch direction flagged
//! in the refactor spec will eventually replace the `default_arch()` seed with
//! an explicit init at binary / arch selection time. For now, xdna-emu is
//! single-arch (AIE2), so each cache is effectively a compile-time constant
//! reached through one indirection.

use std::sync::OnceLock;
use xdna_archspec::isa_execute::IsaExecutor;
use xdna_archspec::locks::LockValueLayout;
use xdna_archspec::stream_switch::StreamSwitchTopology;
use crate::interpreter::timing::LatencyTable;

static LOCK_VALUE_LAYOUT: OnceLock<&'static LockValueLayout> = OnceLock::new();

/// Get the process-wide lock value layout.
///
/// First call lazily resolves through `default_arch().lock_model()`;
/// subsequent calls return the cached pointer directly.
pub fn lock_value_layout() -> &'static LockValueLayout {
    LOCK_VALUE_LAYOUT.get_or_init(|| {
        xdna_archspec::runtime::default_arch().lock_model().value_layout()
    })
}

static STREAM_SWITCH_TOPOLOGY: OnceLock<&'static StreamSwitchTopology> = OnceLock::new();

/// Get the process-wide stream-switch topology.
///
/// First call lazily resolves through `default_arch().stream_switch_model()`;
/// subsequent calls return the cached pointer directly.
pub fn stream_switch_topology() -> &'static StreamSwitchTopology {
    STREAM_SWITCH_TOPOLOGY.get_or_init(|| {
        xdna_archspec::runtime::default_arch().stream_switch_model().topology()
    })
}

static ISA_EXECUTOR: OnceLock<&'static dyn IsaExecutor> = OnceLock::new();

/// Per-arch ISA execute seam for operation-level behavioral divergence.
///
/// Returns the `&'static dyn IsaExecutor` for the runtime's default
/// arch. Ships empty per Subsystem 7's Approach A audit; anchored
/// for future seams.
pub fn isa_executor() -> &'static dyn IsaExecutor {
    *ISA_EXECUTOR.get_or_init(|| {
        xdna_archspec::runtime::default_arch().isa_executor()
    })
}

static HAS_CASCADE_LINK: OnceLock<bool> = OnceLock::new();

/// Whether the default architecture has a cascade link between adjacent compute tiles.
///
/// AIE2 and AIE2P return `true`; AIE1 returns `false`. Lazily resolved from
/// `default_arch().has_cascade_link()` on first call.
///
/// `cascade.rs` gates all cascade operations on this flag, so a future
/// AIE1 path gets no-op cascade handlers without touching execute-layer logic.
pub fn has_cascade_link() -> bool {
    *HAS_CASCADE_LINK.get_or_init(|| {
        xdna_archspec::runtime::default_arch().has_cascade_link()
    })
}

static LATENCY_TABLE: OnceLock<LatencyTable> = OnceLock::new();

/// Process-wide instruction latency table for the default architecture.
///
/// Lazily populated on first call by querying the LLVM MCDisassembler FFI
/// for all instruction itinerary latencies. Subsequent calls return the
/// cached reference with zero FFI overhead.
///
/// This seam replaces direct `LatencyTable::aie2()` calls at construction
/// sites so that the table is built once and shared across all executors.
/// When multi-arch support lands, this accessor will select the table based
/// on the runtime arch rather than the hardcoded AIE2 constructor.
pub fn latency_table() -> &'static LatencyTable {
    LATENCY_TABLE.get_or_init(LatencyTable::aie2)
}
