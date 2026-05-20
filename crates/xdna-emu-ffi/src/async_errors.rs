//! FFI surface for Tier B async-error delivery.
//!
//! Five C symbols: cache reader, ring reader, ring-pending probe, callback
//! registration, clear helper. Plugin consumes `xdna_emu_get_last_async_error`
//! via `resolve_required`; the others are reserved for future consumers
//! (visual debugger, test harnesses, future kernel-driver attachment).
//!
//! Conventions match the rest of this crate (see `query.rs`):
//! - Opaque handle pointer; null handle returns sentinel.
//! - Last-error string set via `set_last_error` on failure paths.
//! - Buffers are copy-on-read; caller retains ownership.

use std::ffi::c_void;

use xdna_emu_core::device::async_errors::AmdxdnaAsyncError;

// `XdnaEmuHandle` is referenced by the FFI functions added in Tasks 10-12.
#[allow(unused_imports)]
use super::XdnaEmuHandle;

/// uapi-mirror of `struct amdxdna_async_error`. Exposed to C as the
/// out-parameter type for `xdna_emu_get_last_async_error` and the record
/// type passed to the push callback.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct XdnaEmuAsyncError {
    pub err_code: u64,
    pub ts_us: u64,
    pub ex_err_code: u64,
}

const _: () = assert!(std::mem::size_of::<XdnaEmuAsyncError>() == 24);

impl From<&AmdxdnaAsyncError> for XdnaEmuAsyncError {
    fn from(src: &AmdxdnaAsyncError) -> Self {
        Self { err_code: src.err_code, ts_us: src.ts_us, ex_err_code: src.ex_err_code }
    }
}

/// C callback signature for push-notification on error.
pub type XdnaEmuAsyncErrorCallback =
    unsafe extern "C" fn(record: *const XdnaEmuAsyncError, user_data: *mut c_void);
