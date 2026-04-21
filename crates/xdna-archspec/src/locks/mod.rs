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

/// Lock-value field layout -- register field shape plus logical value bounds.
///
/// Field shape and bounds diverge per-arch:
/// - AIE2: 6-bit register field (bits [5:0], mask 0x3F), min=-64, max=63.
///   Logical range is 7-bit signed; values outside the 6-bit field
///   alias when read back. aie-rt `xaiemlgbl_reginit.c:2452-2453` for
///   the bounds; `aie_registers_aie2.json` `memory.Lock0_value.Lock_value`
///   for the field shape.
/// - AIE1: no read-back value field. AIE1's lock hardware is
///   address-encoded (`Lock.LockVal * LockMod->LockValOff`); value reads
///   come via the per-lock `LockN` status bit in the all-locks-status
///   register (1 bit per lock, not a per-lock value field). The
///   `width` / `mask` / `sign_bit` fields are not meaningful for AIE1;
///   bounds (min=-1, max=1) are enforced software-side per aie-rt
///   `xaiegbl_reginit.c:1251-1252`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LockValueLayout {
    /// Width of the Lock_value register field, in bits.
    pub width: u8,
    /// Mask that isolates the field within a raw 32-bit register read.
    pub mask: u32,
    /// Position of the sign bit within the field (always `width - 1`
    /// for two's-complement representations).
    pub sign_bit: u8,
    /// Minimum logical lock value (aie-rt LockValLowerBound).
    pub min: i8,
    /// Maximum logical lock value (aie-rt LockValUpperBound).
    pub max: i8,
}

impl LockValueLayout {
    /// Sign-extend a raw register read to a signed 8-bit lock value.
    ///
    /// Identical formula on every arch; only the field width differs.
    #[inline]
    pub fn sign_extend(&self, raw: u32) -> i8 {
        let masked = (raw & self.mask) as u8;
        if masked & (1 << self.sign_bit) != 0 {
            masked as i8 | !(self.mask as i8)
        } else {
            masked as i8
        }
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
