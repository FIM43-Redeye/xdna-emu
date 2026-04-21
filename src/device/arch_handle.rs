//! Process-global cache of the architecture's lock model.
//!
//! Exposes the lock value layout as a fast accessor without forcing
//! every caller to hold a `&dyn ArchConfig` or a `Tile`. Lazy-initialized
//! at first call; the resolved `&'static LockValueLayout` stays cached
//! for the process lifetime.
//!
//! This is a pragmatic bridge for Subsystem 4: the GUI runtime arch-switch
//! direction flagged in the spec will eventually replace the
//! `default_arch()` seed with an explicit init at binary / arch selection
//! time. For now, xdna-emu is single-arch (AIE2), so the cache is
//! effectively a compile-time constant reached through one indirection.

use std::sync::OnceLock;
use xdna_archspec::locks::LockValueLayout;

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
