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
use std::slice;

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

/// Read the last-recorded async-error record into `out`.
///
/// # Returns
/// - `1`: a record is populated and copied to `*out`
/// - `0`: no errors have been recorded since the last reset
/// - `-1`: `handle` is null
/// - `-2`: `out` is null
///
/// # Safety
/// `handle` must be a valid pointer from `xdna_emu_create`; `out` must point
/// to at least `sizeof(XdnaEmuAsyncError)` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_get_last_async_error(
    handle: *mut XdnaEmuHandle,
    out: *mut XdnaEmuAsyncError,
) -> i32 {
    if handle.is_null() {
        return -1;
    }
    if out.is_null() {
        return -2;
    }

    let handle = &mut *handle;
    match handle.engine.device().async_errors.last_cache() {
        Some(rec) => {
            *out = XdnaEmuAsyncError::from(rec);
            1
        }
        None => 0,
    }
}

/// Clear the async-error cache, all per-column rings, and the drain queue.
/// Does NOT touch Tier A L1/L2 latch state or any other tile state.
///
/// # Returns
/// - `0`: success
/// - `-1`: `handle` is null
///
/// # Safety
/// `handle` must be a valid pointer from `xdna_emu_create`.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_clear_async_errors(handle: *mut XdnaEmuHandle) -> i32 {
    if handle.is_null() {
        return -1;
    }

    let handle = &mut *handle;
    handle.engine.device_mut().async_errors.clear();
    0
}

/// Copy up to `buf_size` bytes from column `col`'s ring buffer into `buf`.
/// Bytes are driver-wire format: `AieErrInfoHeader` (12B) followed by
/// `err_cnt * AieError` (12B each).
///
/// # Returns
/// - `N >= 0`: number of bytes copied (always at least 12 for the header)
/// - `-1`: `handle` is null
/// - `-2`: `col` is out of range for this device
/// - `-3`: `buf` is null
///
/// # Safety
/// `handle` must be valid; `buf` must point to at least `buf_size` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_read_async_event_ring(
    handle: *mut XdnaEmuHandle,
    col: u32,
    buf: *mut u8,
    buf_size: u64,
) -> i64 {
    if handle.is_null() {
        return -1;
    }
    if buf.is_null() {
        return -3;
    }

    let handle = &mut *handle;
    let col_u8 = match u8::try_from(col) {
        Ok(c) => c,
        Err(_) => return -2,
    };
    let ring = match handle.engine.device().async_errors.ring(col_u8) {
        Some(r) => r,
        None => return -2,
    };
    let dst = slice::from_raw_parts_mut(buf, buf_size as usize);
    ring.read_into(dst) as i64
}

/// Probe whether column `col`'s ring has any pending records.
///
/// # Returns
/// - `1`: `err_cnt > 0`
/// - `0`: ring is empty
/// - `-1`: `handle` is null
/// - `-2`: `col` is out of range
///
/// # Safety
/// `handle` must be valid.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_async_event_pending(handle: *mut XdnaEmuHandle, col: u32) -> i32 {
    if handle.is_null() {
        return -1;
    }

    let handle = &mut *handle;
    let col_u8 = match u8::try_from(col) {
        Ok(c) => c,
        Err(_) => return -2,
    };
    match handle.engine.device().async_errors.ring(col_u8) {
        Some(r) if r.header().err_cnt > 0 => 1,
        Some(_) => 0,
        None => -2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{xdna_emu_create, xdna_emu_destroy};
    use xdna_archspec::aie2::async_errors::AieErrorOrigin;

    /// Helper: create a handle, run a closure, then destroy it.
    unsafe fn with_handle(f: impl FnOnce(*mut XdnaEmuHandle)) {
        let h = xdna_emu_create();
        assert!(!h.is_null());
        f(h);
        xdna_emu_destroy(h);
    }

    #[test]
    fn get_last_async_error_returns_zero_on_fresh_handle() {
        unsafe {
            with_handle(|h| {
                let mut out = XdnaEmuAsyncError::default();
                let rc = xdna_emu_get_last_async_error(h, &mut out);
                assert_eq!(rc, 0, "no record yet -> 0");
                assert_eq!(out.err_code, 0);
            });
        }
    }

    #[test]
    fn get_last_async_error_returns_one_after_record() {
        unsafe {
            with_handle(|h| {
                // Drive a record directly through the sink (no engine step needed).
                let dev = (*h).engine.device_mut();
                dev.array.set_dma_cycle(50_000);
                dev.async_errors.record_error(1, 2, AieErrorOrigin::Core, 69, 50_000);

                let mut out = XdnaEmuAsyncError::default();
                let rc = xdna_emu_get_last_async_error(h, &mut out);
                assert_eq!(rc, 1, "record present -> 1");
                assert_eq!(out.ts_us, 50);
                assert_eq!(out.ex_err_code, (2u64 << 8) | 1u64);
            });
        }
    }

    #[test]
    fn get_last_async_error_null_handle_returns_minus_one() {
        unsafe {
            let mut out = XdnaEmuAsyncError::default();
            let rc = xdna_emu_get_last_async_error(std::ptr::null_mut(), &mut out);
            assert_eq!(rc, -1);
        }
    }

    #[test]
    fn get_last_async_error_null_out_returns_minus_two() {
        unsafe {
            with_handle(|h| {
                let rc = xdna_emu_get_last_async_error(h, std::ptr::null_mut());
                assert_eq!(rc, -2);
            });
        }
    }

    #[test]
    fn clear_async_errors_resets_cache() {
        unsafe {
            with_handle(|h| {
                let dev = (*h).engine.device_mut();
                dev.async_errors.record_error(1, 2, AieErrorOrigin::Core, 69, 50_000);
                let rc = xdna_emu_clear_async_errors(h);
                assert_eq!(rc, 0);

                let mut out = XdnaEmuAsyncError::default();
                let rc = xdna_emu_get_last_async_error(h, &mut out);
                assert_eq!(rc, 0, "clear must drop the cache");
            });
        }
    }

    #[test]
    fn clear_async_errors_null_handle_returns_minus_one() {
        unsafe {
            let rc = xdna_emu_clear_async_errors(std::ptr::null_mut());
            assert_eq!(rc, -1);
        }
    }

    #[test]
    fn read_async_event_ring_returns_header_only_when_empty() {
        unsafe {
            with_handle(|h| {
                let mut buf = vec![0u8; 64];
                let n = xdna_emu_read_async_event_ring(h, 0, buf.as_mut_ptr(), buf.len() as u64);
                // Empty ring returns just the 12-byte header (err_cnt = 0).
                assert_eq!(n, 12);
                assert_eq!(&buf[0..4], &0u32.to_le_bytes());
            });
        }
    }

    #[test]
    fn read_async_event_ring_returns_payload_after_record() {
        unsafe {
            with_handle(|h| {
                let dev = (*h).engine.device_mut();
                dev.async_errors.record_error(1, 2, AieErrorOrigin::Core, 69, 1_000);

                let mut buf = vec![0u8; 64];
                let n = xdna_emu_read_async_event_ring(h, 1, buf.as_mut_ptr(), buf.len() as u64);
                // 12-byte header + 12-byte record = 24 bytes.
                assert_eq!(n, 24);
                assert_eq!(&buf[0..4], &1u32.to_le_bytes(), "err_cnt = 1");
                assert_eq!(buf[12], 2, "record row");
                assert_eq!(buf[13], 1, "record col");
                assert_eq!(buf[20], 69, "record event_id (offset 12+8)");
            });
        }
    }

    #[test]
    fn read_async_event_ring_invalid_col_returns_minus_two() {
        unsafe {
            with_handle(|h| {
                let mut buf = vec![0u8; 64];
                let n = xdna_emu_read_async_event_ring(h, 99, buf.as_mut_ptr(), buf.len() as u64);
                assert_eq!(n, -2);
            });
        }
    }

    #[test]
    fn read_async_event_ring_null_buf_returns_minus_three() {
        unsafe {
            with_handle(|h| {
                let n = xdna_emu_read_async_event_ring(h, 0, std::ptr::null_mut(), 64);
                assert_eq!(n, -3);
            });
        }
    }

    #[test]
    fn read_async_event_ring_null_handle_returns_minus_one() {
        unsafe {
            let mut buf = vec![0u8; 64];
            let n =
                xdna_emu_read_async_event_ring(std::ptr::null_mut(), 0, buf.as_mut_ptr(), buf.len() as u64);
            assert_eq!(n, -1);
        }
    }

    #[test]
    fn async_event_pending_zero_on_empty() {
        unsafe {
            with_handle(|h| {
                assert_eq!(xdna_emu_async_event_pending(h, 0), 0);
            });
        }
    }

    #[test]
    fn async_event_pending_one_after_record() {
        unsafe {
            with_handle(|h| {
                let dev = (*h).engine.device_mut();
                dev.async_errors.record_error(3, 2, AieErrorOrigin::Core, 69, 1_000);
                assert_eq!(xdna_emu_async_event_pending(h, 3), 1);
                assert_eq!(xdna_emu_async_event_pending(h, 0), 0, "col 0 still empty");
            });
        }
    }

    #[test]
    fn async_event_pending_invalid_col_returns_minus_two() {
        unsafe {
            with_handle(|h| {
                assert_eq!(xdna_emu_async_event_pending(h, 99), -2);
            });
        }
    }
}
