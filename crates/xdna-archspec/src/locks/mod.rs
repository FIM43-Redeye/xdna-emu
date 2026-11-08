//! Lock model trait: feature flags and value-layout carrier for
//! per-arch lock behavior.
//!
//! Subsystem 4 of the device-family refactor introduces this trait
//! as a behavioral seam. AIE2 / AIE2P use the concrete
//! `aie2::locks::Aie2LockModel` impl (6-bit register field, acq-EQ
//! + dynamic value ops enabled). AIE1's eventual
//! `aie1::locks::Aie1LockModel` will reflect address-encoded locks
//! (no read-back field), binary-semaphore software bounds, and
//! no acq-EQ.
//!
//! Consumers access an impl via `ArchConfig::lock_model()`, which
//! returns a `&'static dyn LockModel`. Because every concrete impl
//! is zero-sized and stateless, the accessor returns a reference to
//! a `static` singleton -- no allocation, no lifetime bookkeeping.
//!
//! The trait is intentionally coarse (3 methods, all cold-path).
//! Hot-path consumers (none today) cache `&'static LockValueLayout`
//! at construction rather than re-dispatching per call.

#[cfg(test)]
mod tests;

/// Lock-value field layout: the register field that holds a lock's count.
///
/// - AIE2: a 6-bit unsigned count (bits [5:0], mask 0x3F), 0..=63, the
///   field shape `aie_registers_aie2.json` gives for
///   `memory.Lock0_value.Lock_value` and the maximum mlir-aie's AIE2
///   target model allows (`getMaxLockValue() == 0x3F`). Silicon confirms
///   it (lock request probe, 2026-11-08): 32..=63 are ordinary positive
///   counts, a write keeps bits [5:0], and a release wraps modulo 64.
///   aie-rt's LockValLowerBound/UpperBound (-64/63) bound the 7-bit
///   signed *change* value of an acquire or release, not the count.
/// - AIE1: no read-back value field. AIE1's lock hardware is
///   address-encoded (`Lock.LockVal * LockMod->LockValOff`); value reads
///   come via the per-lock `LockN` status bit in the all-locks-status
///   register (1 bit per lock, not a per-lock value field), so `width`
///   and `mask` are not meaningful there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LockValueLayout {
    /// Width of the Lock_value register field, in bits.
    pub width: u8,
    /// Mask that isolates the field within a raw 32-bit register value.
    pub mask: u32,
}

impl LockValueLayout {
    /// The count a raw register value stores: its low `width` bits.
    #[inline]
    pub fn decode(&self, raw: u32) -> u8 {
        (raw & self.mask) as u8
    }

    /// The largest count the field holds.
    #[inline]
    pub fn max(&self) -> u8 {
        self.mask as u8
    }
}

/// Per-arch lock behavior, consulted at construction / BD-parse
/// boundaries (never in a per-cycle hot path on today's AIE2-only
/// call sites).
pub trait LockModel: Send + Sync + core::fmt::Debug {
    /// Whether the arch supports the acquire-with-equality mode
    /// (lock.value == expected) in addition to the default
    /// acquire-with-greater-or-equal (lock.value >= expected).
    ///
    /// AIE2+: true (both modes). AIE1: false (single ACQ mode only).
    /// Evidence: AIE2 exposes per-lock event types for both ACQ_GE
    /// and ACQ_EQ at `xaiemlgbl_params.h:8048-8054` (offsets);
    /// AIE1's `xaie_locks_aie.c` has no such distinction.
    fn supports_acquire_eq(&self) -> bool;

    /// Whether the arch supports host-side `GetValue` / `SetValue`
    /// of the lock register.
    ///
    /// AIE2+: true. AIE1: false (returns `XAIE_FEATURE_NOT_SUPPORTED`
    /// at `xaie_locks_aie.c:169` for GetValue, `:195` for SetValue).
    fn supports_dynamic_value_ops(&self) -> bool;

    /// The value-field layout for this arch's locks.
    ///
    /// Returns a `'static` reference to a singleton layout so hot-path
    /// consumers can cache the pointer at construction.
    fn value_layout(&self) -> &'static LockValueLayout;
}
