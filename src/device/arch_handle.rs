//! Process-global cache of the architecture's lock model and stream-switch topology.
//!
//! Exposes the lock value layout and stream-switch topology as fast accessors
//! without forcing every caller to hold a `&dyn ArchConfig` or a `Tile`.
//! Lazy-initialized at first call; the resolved `&'static` references stay cached
//! for the process lifetime.
//!
//! This is a pragmatic bridge: the GUI runtime arch-switch direction flagged
//! in the refactor spec will eventually replace the `default_arch()` seed with
//! an explicit init at binary / arch selection time. For now, xdna-emu is
//! single-arch (AIE2), so each cache is effectively a compile-time constant
//! reached through one indirection.

use std::sync::OnceLock;
use xdna_archspec::locks::LockValueLayout;
use xdna_archspec::stream_switch::StreamSwitchTopology;

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
