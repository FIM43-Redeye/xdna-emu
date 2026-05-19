//! Tier B async-error subsystem.
//!
//! Sits parallel to `device::interrupts` (Tier A). When an error-category
//! event is generated, `state::effects::apply_tile_local_effects` calls
//! `AsyncErrorSink::record_error` here in addition to the existing
//! Tier A L1/L2 latch path.
//!
//! Three output surfaces: per-handle cache (consumed by plugin ioctl),
//! per-column 8KB mailbox rings in driver-wire format (reserved for future
//! kernel-driver attachment), optional push callback (FFI-registered).
//!
//! Design source:
//! `docs/superpowers/specs/2026-05-19-interrupt-tier-b-firmware-mailbox-design.md`

pub mod sink;
pub mod types;

pub use sink::AsyncErrorSink;
pub use types::{
    AieError, AieErrInfoHeader, AmdxdnaAsyncError, AsyncRing, ASYNC_BUF_SIZE, MAX_ERRORS_PER_RING,
    RET_CODE_OVERFLOW,
};
