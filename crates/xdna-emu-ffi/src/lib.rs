//! Foreign Function Interface for xdna-emu.
//!
//! This crate produces `libxdna_emu.so` (a C-compatible shared library)
//! for integrating xdna-emu with C/C++ applications -- primarily the XRT
//! emulation plugin at `xrt-plugin/`.
//!
//! # Safety Contract
//!
//! All functions use `unsafe extern "C"` ABI. The C caller must uphold:
//!
//! ## Handle Invariants
//! - `XdnaEmuHandle` pointers returned by `xdna_emu_create` are opaque
//!   and must not be dereferenced or modified by the caller.
//! - A handle must be passed to exactly one `xdna_emu_destroy` call.
//!   Using a handle after destroy is undefined behavior.
//! - Handles are **not thread-safe**: concurrent calls on the same handle
//!   are data races. The XRT plugin serializes via its own mutex.
//!
//! ## Pointer and Buffer Requirements
//! - All `*const c_char` string parameters must be valid, NUL-terminated,
//!   UTF-8 C strings. Invalid UTF-8 returns an error (not UB).
//! - All `*const u8` / `*mut u8` buffer parameters must point to at
//!   least `len` accessible bytes. Null is checked where documented.
//! - Buffer data is copied during write/read operations; the caller
//!   retains ownership of their pointers.
//!
//! ## Error Propagation
//! - Errors are stored in a thread-local string (`LAST_ERROR`).
//! - After any non-Success return, call `xdna_emu_get_error` on the
//!   **same thread** to retrieve the message. The message is valid
//!   until the next FFI call on that thread.
//!
//! ## Lifetime Requirements
//! - Handles returned by `xdna_emu_create` live until `xdna_emu_destroy`.
//! - String pointers from `xdna_emu_get_error` / `xdna_emu_get_device_name`
//!   are borrowed from thread-local or handle state; they are invalidated
//!   by the next FFI call on the same thread or by destroying the handle.

mod async_errors;
mod backend;
mod classify;
mod config;
pub mod context;
mod execution;
mod firmware;
mod memory;
mod query;

pub use async_errors::*;
pub use backend::NpuBackend;
pub use classify::*;
pub use config::*;
pub use context::{xdna_emu_get_context_state, XdnaEmuContextState};
pub use execution::*;
pub use firmware::*;
pub use memory::*;
pub use query::*;

use std::cell::RefCell;
use std::ffi::c_char;
use std::sync::Mutex;

use xdna_emu_core::interpreter::engine::InterpreterEngine;
use xdna_emu_core::npu::NpuExecutor;

// Thread-local error storage for xdna_emu_get_error().
thread_local! {
    static LAST_ERROR: RefCell<String> = RefCell::new(String::new());
}

pub(crate) fn set_last_error(msg: String) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = msg;
    });
}

/// Wrapper around an FFI callback + user_data pointer pair.
///
/// `*mut c_void` is not `Send`; this struct asserts the caller follows
/// the documented "handles are not thread-safe" contract so we can store
/// the pair on the handle without inheriting `!Send` constraints on
/// `XdnaEmuHandle`.
#[derive(Clone, Copy)]
pub(crate) struct AsyncErrorCallback {
    pub func: async_errors::XdnaEmuAsyncErrorCallback,
    pub user_data: *mut std::ffi::c_void,
}
// SAFETY: Both Send and Sync rest on the same invariant: handle access
// is serialized by the plugin's mutex (see XdnaEmuHandle docs in this
// file). So neither concurrent ownership (Send) nor concurrent reference
// (Sync) is possible in practice. The user_data pointer is opaque to us
// and only ever passed back to the registered C callback on the same
// thread that registered it.
unsafe impl Send for AsyncErrorCallback {}
unsafe impl Sync for AsyncErrorCallback {}

/// Opaque handle to emulator state.
/// Wraps an NpuBackend and related state.
pub struct XdnaEmuHandle {
    pub(crate) backend: Box<dyn crate::backend::NpuBackend>,
    pub(crate) xclbin_path: Option<String>,
    pub(crate) npu_executor: NpuExecutor,
    pub(crate) max_cycles: u64,
    /// Next address to allocate for xdna_emu_alloc_buffer.
    /// Starts at a high address to avoid conflicts with user-specified regions.
    pub(crate) next_alloc_addr: u64,
    /// Free list of (addr, aligned_size) pairs for size-matched address reuse.
    /// Real HW kernel drivers' BO allocators reuse physical addresses across
    /// allocate/free cycles; the bridge runner's per-run pool teardown relies
    /// on that to keep cumulative shim-DMA channel pointers pointing at the
    /// same DDR location across batches. Without reuse, the EMU's monotonic
    /// next_alloc_addr produces a fresh BO per run and the trace channel's
    /// active transfer (which still holds the previous run's address) writes
    /// to a stale address. See bridge-trace-runner.cpp:1500-1738 for the
    /// teardown rationale (alternating-TIMEOUT workaround on real HW).
    pub(crate) free_list: Vec<(u64, u64)>,
    /// Optional FFI-registered push callback for async errors (Tier B).
    pub(crate) async_callback: Option<AsyncErrorCallback>,
}

/// Result codes for FFI operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XdnaEmuResult {
    Success = 0,
    InvalidHandle = 1,
    InvalidPath = 2,
    ParseError = 3,
    ExecutionError = 4,
    BufferError = 5,
    NullPointer = 6,
}

/// Why the emulator stopped running.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XdnaEmuHaltReason {
    /// Kernel ran to natural completion (cores halted, syncs satisfied).
    Completed = 0,
    /// Cycle budget (`max_cycles`) reached before natural completion.
    Budget = 1,
    /// Error during execution (FFI fault, executor error).
    Error = 2,
    /// A MaskPoll instruction could not be satisfied because the engine became
    /// quiescent (no cores running, no DMA progress) with the poll condition
    /// still unmet.
    ///
    /// This is the expected outcome for the debug_halt_probe's MASKPOLL
    /// halt-synchronization on the emulator: the probe polls `Core_Status[16]`
    /// (DEBUG_HALT), which never sets because the emulator's control-packet
    /// write path drops writes to core/debug registers into a catch-all.  The
    /// emulator terminates deterministically with this reason rather than
    /// spinning forever.  The polled register is NOT modified; no subsequent
    /// instruction is issued.
    ///
    /// On real hardware the poll satisfies (the core halts at the breakpoint);
    /// this reason is emulator-only.
    MaskPollUnsatisfied = 3,
    /// Tier C: the in-flight submission wedged; the per-context state is
    /// Failed. Caller should observe this, call `xdna_emu_reset_context`
    /// before the next submission, and translate to an EIO-shaped XRT
    /// command state in `run.wait()`.
    ///
    /// "Recovered" describes the contract: the device is ready to accept
    /// the next submission once reset is called; this status code is the
    /// signal that triggered recovery, not that recovery has already run.
    WedgeRecovered = 4,
}

/// Execution status returned by run functions.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct XdnaEmuExecStatus {
    pub result: XdnaEmuResult,
    pub cycles_executed: u64,
    /// True when cores have halted or all DMA syncs are satisfied.
    /// Kept for back-compat; `halt_reason` carries richer information.
    pub halted: bool,
    /// Structured reason for the halt.
    pub halt_reason: XdnaEmuHaltReason,
}

// Global lock for thread safety during initialization
static INIT_LOCK: Mutex<()> = Mutex::new(());

/// Create a new emulator instance.
///
/// # Safety
/// Returns a non-null handle on success, null on failure.
/// The returned handle must be freed with `xdna_emu_destroy`.
///
/// SAFETY: Box::into_raw transfers ownership to the caller. No raw
/// pointer dereferences occur; the handle is constructed from safe Rust
/// values and leaked via Box.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_create() -> *mut XdnaEmuHandle {
    let _lock = INIT_LOCK.lock().unwrap();

    // If XDNA_EMU_LOG_LEVEL is set but RUST_LOG is not, bridge the two
    // so that a single env var controls both C++ plugin and Rust logging.
    if std::env::var("RUST_LOG").is_err() {
        if let Ok(level) = std::env::var("XDNA_EMU_LOG_LEVEL") {
            std::env::set_var("RUST_LOG", &level);
        }
    }

    // Initialize logging if not already done
    let _ = env_logger::try_init();
    xdna_emu_core::debug::watch::init();

    let config = xdna_emu_core::config::Config::get();
    let mut engine = InterpreterEngine::new_npu1();
    engine.set_stall_threshold(config.stall_threshold());
    let handle = Box::new(XdnaEmuHandle {
        backend: Box::new(engine),
        xclbin_path: None,
        npu_executor: NpuExecutor::new(),
        max_cycles: config.max_cycles(),
        // Start auto-allocation at 0x8000_0000_0000 to avoid conflicts
        // with user-specified host regions (typically < 0x1_0000_0000).
        next_alloc_addr: 0x8000_0000_0000,
        free_list: Vec::new(),
        async_callback: None,
    });

    Box::into_raw(handle)
}

/// Destroy an emulator instance.
///
/// # Safety
/// `handle` must be a valid pointer returned by `xdna_emu_create`,
/// or null (in which case this is a no-op). Double-free is UB.
///
/// SAFETY: Box::from_raw reclaims the allocation made by Box::into_raw
/// in xdna_emu_create. The null check prevents UB from null pointers.
/// The caller must not use the handle after this call.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_destroy(handle: *mut XdnaEmuHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Reset a context for a fresh submission.
///
/// Transitions the named context from Failed -> Connected (no-op on already-
/// Connected), clears its Tier B async-error sink, and calls the engine's
/// per-context reset to wipe stale tile state.
///
/// Call this between submissions, and especially after observing a
/// `WedgeRecovered` halt -- the next `xdna_emu_run` entry rejects a non-
/// Connected context.
///
/// Returns:
/// - `Success` on a clean reset
/// - `InvalidHandle` for a null handle
/// - `ExecutionError` for an invalid context_id
///
/// # Safety
/// - `handle` must be valid
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_reset_context(
    handle: *mut XdnaEmuHandle,
    context_id: u32,
) -> XdnaEmuResult {
    if handle.is_null() {
        return XdnaEmuResult::InvalidHandle;
    }

    let handle = &mut *handle;
    use xdna_emu_core::device::context::ContextId;
    let cid = ContextId(context_id);
    if handle.backend.reset_context(cid).is_err() {
        log::error!("xdna_emu_reset_context: invalid context_id {}", context_id);
        return XdnaEmuResult::ExecutionError;
    }
    handle.backend.reset_for_new_context();
    log::debug!("xdna_emu_reset_context: cleared per-context tile state for ctx {}", context_id);
    XdnaEmuResult::Success
}

/// Get the last error message (for debugging).
///
/// Reads the thread-local error string set by failed FFI operations.
///
/// # Safety
/// - `buffer` must point to at least `buffer_size` bytes
/// - Returns the number of bytes written (excluding null terminator)
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_get_error(buffer: *mut c_char, buffer_size: u64) -> u64 {
    if buffer.is_null() || buffer_size == 0 {
        return 0;
    }

    LAST_ERROR.with(|e| {
        let msg = e.borrow();
        if msg.is_empty() {
            *buffer = 0;
            return 0;
        }
        let bytes = msg.as_bytes();
        let copy_len = bytes.len().min((buffer_size - 1) as usize);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer as *mut u8, copy_len);
        *buffer.add(copy_len) = 0; // null terminator
        copy_len as u64
    })
}

/// Get version information.
#[no_mangle]
pub extern "C" fn xdna_emu_version() -> u32 {
    // Version 0.1.0 = 0x000100
    0x000100
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    /// Helper: create a handle, run a closure, then destroy it.
    unsafe fn with_handle(f: impl FnOnce(*mut XdnaEmuHandle)) {
        let handle = xdna_emu_create();
        assert!(!handle.is_null(), "xdna_emu_create returned null");
        f(handle);
        xdna_emu_destroy(handle);
    }

    #[test]
    fn test_alloc_buffer_returns_page_aligned_address() {
        unsafe {
            with_handle(|h| {
                let addr = xdna_emu_alloc_buffer(h, 1024);
                assert_ne!(addr, 0, "alloc_buffer should return non-zero");
                assert_eq!(addr % 4096, 0, "address should be page-aligned");
            });
        }
    }

    #[test]
    fn test_alloc_buffer_successive_non_overlapping() {
        unsafe {
            with_handle(|h| {
                let a1 = xdna_emu_alloc_buffer(h, 4096);
                let a2 = xdna_emu_alloc_buffer(h, 8192);
                assert_ne!(a1, 0);
                assert_ne!(a2, 0);
                assert_ne!(a1, a2, "successive allocations must not overlap");
                // Second allocation should come after first.
                assert!(a2 >= a1 + 4096);
            });
        }
    }

    #[test]
    fn test_alloc_buffer_rounds_up_size() {
        unsafe {
            with_handle(|h| {
                // Allocating 1 byte should still consume a full page.
                let a1 = xdna_emu_alloc_buffer(h, 1);
                let a2 = xdna_emu_alloc_buffer(h, 1);
                assert_eq!(a2 - a1, 4096, "1-byte alloc should round up to 4096");
            });
        }
    }

    #[test]
    fn test_alloc_buffer_zero_size_returns_zero() {
        unsafe {
            with_handle(|h| {
                let addr = xdna_emu_alloc_buffer(h, 0);
                assert_eq!(addr, 0, "zero-size alloc should return 0");
            });
        }
    }

    #[test]
    fn test_alloc_buffer_null_handle() {
        unsafe {
            let addr = xdna_emu_alloc_buffer(std::ptr::null_mut(), 4096);
            assert_eq!(addr, 0, "null handle should return 0");
        }
    }

    #[test]
    fn test_free_buffer() {
        unsafe {
            with_handle(|h| {
                let addr = xdna_emu_alloc_buffer(h, 4096);
                assert_ne!(addr, 0);
                // Should not panic or error.
                xdna_emu_free_buffer(h, addr);
                // Freeing a non-existent address should be a no-op (just logs).
                xdna_emu_free_buffer(h, 0xDEAD_BEEF);
            });
        }
    }

    #[test]
    fn test_alloc_after_free_recycles_address() {
        // The bridge runner's per-run pool teardown (alloc -> free -> alloc
        // with the same size) needs the same address back; the trace
        // shim-DMA channel's persistent write pointer relies on it.
        unsafe {
            with_handle(|h| {
                let a1 = xdna_emu_alloc_buffer(h, 4096);
                xdna_emu_free_buffer(h, a1);
                let a2 = xdna_emu_alloc_buffer(h, 4096);
                assert_eq!(a1, a2, "freed addr should be recycled on same-size alloc");
            });
        }
    }

    #[test]
    fn test_recycle_only_on_size_match() {
        unsafe {
            with_handle(|h| {
                let small = xdna_emu_alloc_buffer(h, 4096);
                xdna_emu_free_buffer(h, small);
                // Different aligned size: must NOT recycle (would risk
                // returning an under-sized region).
                let big = xdna_emu_alloc_buffer(h, 8192);
                assert_ne!(big, small);
                // Now another 4K allocation should recycle the original.
                let recycled = xdna_emu_alloc_buffer(h, 4096);
                assert_eq!(recycled, small);
            });
        }
    }

    #[test]
    fn test_recycle_survives_repeated_pool_cycles() {
        // Models the bridge runner's batch-stdin loop: repeatedly allocate
        // a fixed pool, then free it, then allocate again. The address
        // allocator rotates within a stable working set rather than
        // drifting monotonically, so DMA channel state from a previous
        // run still resolves to a live physical location.
        //
        // Per-slot order may vary (free list is unordered), but:
        //   - the unique-size slot (trace BO) must round-trip exactly
        //   - the *set* of addresses across same-size slots must be stable
        unsafe {
            with_handle(|h| {
                let mut first_trace: Option<u64> = None;
                let mut first_4k: std::collections::HashSet<u64> = std::collections::HashSet::new();
                for _ in 0..3 {
                    let trace = xdna_emu_alloc_buffer(h, 1024 * 1024);
                    let instr = xdna_emu_alloc_buffer(h, 4096);
                    let input = xdna_emu_alloc_buffer(h, 4096);
                    if let Some(t) = first_trace {
                        assert_eq!(trace, t, "trace BO addr must round-trip");
                        let cycle_4k: std::collections::HashSet<u64> = [instr, input].into_iter().collect();
                        assert_eq!(cycle_4k, first_4k, "4K pool set must be stable");
                    } else {
                        first_trace = Some(trace);
                        first_4k = [instr, input].into_iter().collect();
                    }
                    xdna_emu_free_buffer(h, trace);
                    xdna_emu_free_buffer(h, instr);
                    xdna_emu_free_buffer(h, input);
                }
            });
        }
    }

    #[test]
    fn test_alloc_buffer_read_write_roundtrip() {
        unsafe {
            with_handle(|h| {
                let addr = xdna_emu_alloc_buffer(h, 256);
                assert_ne!(addr, 0);

                // Write data through host memory interface.
                let data: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
                let result = xdna_emu_write_host_memory(h, addr, data.as_ptr(), 16);
                assert_eq!(result, XdnaEmuResult::Success);

                // Read it back.
                let mut buf = [0u8; 16];
                let result = xdna_emu_read_host_memory(h, addr, buf.as_mut_ptr(), 16);
                assert_eq!(result, XdnaEmuResult::Success);
                assert_eq!(buf, data);
            });
        }
    }

    #[test]
    fn test_read_register_valid_tile() {
        unsafe {
            with_handle(|h| {
                // Write a known value to a lock register on tile (0, 2).
                // Lock_Value registers for compute tiles start at 0x1F100,
                // spaced 16 bytes apart.
                let lock_value_0 = 0x1F100u32; // Lock 0 value register

                let rc = xdna_emu_write_register(h, 0, 2, lock_value_0, 0x42);
                assert_eq!(rc, 0, "write_register should succeed");

                let val = xdna_emu_read_register(h, 0, 2, lock_value_0);
                assert_eq!(val, 0x42, "read back should match written value");
            });
        }
    }

    #[test]
    fn test_write_register_out_of_bounds_tile() {
        unsafe {
            with_handle(|h| {
                // Tile (99, 99) does not exist.
                let rc = xdna_emu_write_register(h, 99, 99, 0x1F100, 0);
                assert_eq!(rc, -2, "out-of-bounds tile should return -2");
            });
        }
    }

    #[test]
    fn test_read_register_null_handle() {
        unsafe {
            let val = xdna_emu_read_register(std::ptr::null_mut(), 0, 2, 0x1F100);
            assert_eq!(val, 0, "null handle should return 0");
        }
    }

    #[test]
    fn test_write_register_null_handle() {
        unsafe {
            let rc = xdna_emu_write_register(std::ptr::null_mut(), 0, 2, 0x1F100, 0);
            assert_eq!(rc, -1, "null handle should return -1");
        }
    }

    #[test]
    fn test_tile_memory_write_read_roundtrip() {
        unsafe {
            with_handle(|h| {
                // Tile (0, 2) is a compute tile with 64KB data memory.
                let pattern: [u8; 8] = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE];

                let rc = xdna_emu_write_tile_memory(h, 0, 2, 0, 8, pattern.as_ptr());
                assert_eq!(rc, 0, "write_tile_memory should succeed");

                let mut buf = [0u8; 8];
                let rc = xdna_emu_read_tile_memory(h, 0, 2, 0, 8, buf.as_mut_ptr());
                assert_eq!(rc, 0, "read_tile_memory should succeed");
                assert_eq!(buf, pattern, "read should match written data");
            });
        }
    }

    #[test]
    fn test_tile_memory_nonzero_offset() {
        unsafe {
            with_handle(|h| {
                let data: [u8; 4] = [0x11, 0x22, 0x33, 0x44];
                let offset = 1024u32;

                let rc = xdna_emu_write_tile_memory(h, 0, 2, offset, 4, data.as_ptr());
                assert_eq!(rc, 0);

                let mut buf = [0u8; 4];
                let rc = xdna_emu_read_tile_memory(h, 0, 2, offset, 4, buf.as_mut_ptr());
                assert_eq!(rc, 0);
                assert_eq!(buf, data);
            });
        }
    }

    #[test]
    fn test_tile_memory_out_of_bounds() {
        unsafe {
            with_handle(|h| {
                // Compute tile has 64KB (65536 bytes). Writing at offset 65535
                // with size 2 should fail.
                let data = [0u8; 2];
                let rc = xdna_emu_write_tile_memory(h, 0, 2, 65535, 2, data.as_ptr());
                assert_eq!(rc, -3, "exceeding bounds should return -3");

                let mut buf = [0u8; 2];
                let rc = xdna_emu_read_tile_memory(h, 0, 2, 65535, 2, buf.as_mut_ptr());
                assert_eq!(rc, -3, "exceeding bounds should return -3");
            });
        }
    }

    #[test]
    fn test_tile_memory_invalid_tile() {
        unsafe {
            with_handle(|h| {
                let mut buf = [0u8; 4];
                let rc = xdna_emu_read_tile_memory(h, 99, 99, 0, 4, buf.as_mut_ptr());
                assert_eq!(rc, -2, "invalid tile should return -2");
            });
        }
    }

    #[test]
    fn test_tile_memory_null_pointer() {
        unsafe {
            with_handle(|h| {
                let rc = xdna_emu_read_tile_memory(h, 0, 2, 0, 4, std::ptr::null_mut());
                assert_eq!(rc, -4, "null out pointer should return -4");

                let rc = xdna_emu_write_tile_memory(h, 0, 2, 0, 4, std::ptr::null());
                assert_eq!(rc, -4, "null data pointer should return -4");
            });
        }
    }

    #[test]
    fn test_tile_memory_null_handle() {
        unsafe {
            let mut buf = [0u8; 4];
            let rc = xdna_emu_read_tile_memory(std::ptr::null_mut(), 0, 2, 0, 4, buf.as_mut_ptr());
            assert_eq!(rc, -1, "null handle should return -1");

            let data = [0u8; 4];
            let rc = xdna_emu_write_tile_memory(std::ptr::null_mut(), 0, 2, 0, 4, data.as_ptr());
            assert_eq!(rc, -1, "null handle should return -1");
        }
    }

    #[test]
    fn test_get_columns_rows() {
        unsafe {
            with_handle(|h| {
                let cols = xdna_emu_get_columns(h);
                let rows = xdna_emu_get_rows(h);
                // NPU1 default: 5 columns, 6 rows.
                assert_eq!(cols, 5, "NPU1 should have 5 columns");
                assert_eq!(rows, 6, "NPU1 should have 6 rows");
            });
        }
    }

    #[test]
    fn test_get_columns_rows_null_handle() {
        unsafe {
            assert_eq!(xdna_emu_get_columns(std::ptr::null_mut()), 0);
            assert_eq!(xdna_emu_get_rows(std::ptr::null_mut()), 0);
        }
    }

    #[test]
    fn test_get_device_name() {
        unsafe {
            with_handle(|h| {
                let mut buf = [0i8; 256];
                let len = xdna_emu_get_device_name(h, buf.as_mut_ptr(), 256);
                assert!(len > 0, "should return positive length");
                let name = CStr::from_ptr(buf.as_ptr()).to_str().unwrap();
                assert!(name.contains("Emulated"), "name should contain 'Emulated': {}", name);
                assert!(name.contains("AIE2"), "name should contain 'AIE2': {}", name);
            });
        }
    }

    #[test]
    fn test_get_device_name_null_handle() {
        unsafe {
            let mut buf = [0i8; 64];
            let rc = xdna_emu_get_device_name(std::ptr::null_mut(), buf.as_mut_ptr(), 64);
            assert_eq!(rc, -1, "null handle should return -1");
        }
    }

    #[test]
    fn test_get_device_name_small_buffer() {
        unsafe {
            with_handle(|h| {
                // Buffer of 10 should truncate but not crash.
                let mut buf = [0i8; 10];
                let len = xdna_emu_get_device_name(h, buf.as_mut_ptr(), 10);
                assert_eq!(len, 9, "should truncate to buf_size - 1");
                // Should be null-terminated.
                assert_eq!(buf[9], 0);
            });
        }
    }

    // -- Diagnostic query tests -------------------------------------------

    #[test]
    fn test_get_lock_value_default_zero() {
        unsafe {
            with_handle(|h| {
                // Compute tile (0, 2) has 16 locks, all initially 0.
                let val = xdna_emu_get_lock_value(h, 0, 2, 0);
                assert_eq!(val, 0, "default lock value should be 0");
            });
        }
    }

    #[test]
    fn test_get_lock_value_after_write_register() {
        unsafe {
            with_handle(|h| {
                // Write lock 0 value via register interface.
                // Lock_Value registers for compute tiles start at 0x1F000
                // (Lock0_value), spaced 0x10 bytes apart.
                let lock_value_0 = 0x1F000u32;
                let rc = xdna_emu_write_register(h, 0, 2, lock_value_0, 3);
                assert_eq!(rc, 0);

                let val = xdna_emu_get_lock_value(h, 0, 2, 0);
                assert_eq!(val, 3, "lock should reflect written value");
            });
        }
    }

    #[test]
    fn test_get_lock_value_invalid_tile() {
        unsafe {
            with_handle(|h| {
                let val = xdna_emu_get_lock_value(h, 99, 99, 0);
                assert_eq!(val, -128, "invalid tile should return -128");
            });
        }
    }

    #[test]
    fn test_get_lock_value_null_handle() {
        unsafe {
            let val = xdna_emu_get_lock_value(std::ptr::null_mut(), 0, 2, 0);
            assert_eq!(val, -128);
        }
    }

    #[test]
    fn test_get_dma_channel_state_idle_by_default() {
        unsafe {
            with_handle(|h| {
                // All channels should be idle at startup.
                let state = xdna_emu_get_dma_channel_state(h, 0, 2, 1, 0);
                assert_eq!(state & 0xFF, 0, "s2mm ch0 should be idle");

                let state = xdna_emu_get_dma_channel_state(h, 0, 2, 0, 0);
                assert_eq!(state & 0xFF, 0, "mm2s ch0 should be idle");
            });
        }
    }

    #[test]
    fn test_get_dma_channel_stats_default_zeros() {
        unsafe {
            with_handle(|h| {
                let mut stats = XdnaEmuChannelStats {
                    transfers_completed: 0xFF,
                    bytes_transferred: 0xFF,
                    cycles_spent: 0xFF,
                    lock_wait_cycles: 0xFF,
                };
                let rc = xdna_emu_get_dma_channel_stats(h, 0, 2, 1, 0, &mut stats);
                assert_eq!(rc, 0, "should succeed for valid tile/channel");
                assert_eq!(stats.transfers_completed, 0);
                assert_eq!(stats.bytes_transferred, 0);
            });
        }
    }

    #[test]
    fn test_set_log_level() {
        unsafe {
            let level = b"debug\0";
            let rc = xdna_emu_set_log_level(level.as_ptr() as *const c_char);
            assert_eq!(rc, 0, "setting debug level should succeed");

            let bad = b"nonsense\0";
            let rc = xdna_emu_set_log_level(bad.as_ptr() as *const c_char);
            assert_eq!(rc, -1, "invalid level should return -1");
        }
    }

    #[test]
    fn test_dump_tile_state() {
        unsafe {
            with_handle(|h| {
                let mut buf = [0i8; 1024];
                let len = xdna_emu_dump_tile_state(h, 0, 2, buf.as_mut_ptr(), 1024);
                assert!(len > 0, "should produce non-empty output");
                let text = CStr::from_ptr(buf.as_ptr()).to_str().unwrap();
                assert!(text.contains("tile(0,2)"), "should mention tile coords");
                assert!(text.contains("locks:"), "should mention locks");
            });
        }
    }

    #[test]
    fn test_dump_tile_state_invalid_tile() {
        unsafe {
            with_handle(|h| {
                let mut buf = [0i8; 64];
                let rc = xdna_emu_dump_tile_state(h, 99, 99, buf.as_mut_ptr(), 64);
                assert_eq!(rc, -2);
            });
        }
    }

    #[test]
    fn test_get_error_returns_last_error() {
        unsafe {
            // Trigger an error by querying an invalid tile.
            let _ = xdna_emu_get_lock_value(
                // Need a valid handle but invalid tile.
                xdna_emu_create(),
                99,
                99,
                0,
            );

            let mut buf = [0i8; 256];
            let len = xdna_emu_get_error(buf.as_mut_ptr(), 256);
            assert!(len > 0, "should have an error message");
            let msg = CStr::from_ptr(buf.as_ptr()).to_str().unwrap();
            assert!(msg.contains("out of bounds"), "error should mention bounds: {}", msg);
        }
    }

    // =====================================================================
    // FFI interface completeness -- parsed from the C++ transport source.
    //
    // This test reads `xrt-plugin/src/transport_inprocess.cpp` and extracts
    // every symbol name passed to `resolve_required` or `resolve_optional`.
    // It then reads all Rust source files under this crate's `src/` and
    // extracts every `#[no_mangle]` exported function name. The test asserts
    // that every symbol the C++ side expects is present in our Rust exports.
    //
    // If someone adds a new FFI function to the C++ transport, this test
    // fails until the corresponding Rust function is implemented.
    // =====================================================================

    /// Extract FFI symbol names from the C++ transport source.
    /// Matches `resolve_required<...>("symbol_name")` and
    /// `resolve_optional<...>("symbol_name")` patterns.
    fn parse_cpp_expected_symbols(cpp_source: &str) -> Vec<(String, bool)> {
        let mut symbols = Vec::new();
        for line in cpp_source.lines() {
            let trimmed = line.trim();
            let required = trimmed.contains("resolve_required");
            let optional = trimmed.contains("resolve_optional");
            if !required && !optional {
                continue;
            }
            // Extract the string literal: find the quoted symbol name.
            if let Some(start) = trimmed.find('"') {
                if let Some(end) = trimmed[start + 1..].find('"') {
                    let name = &trimmed[start + 1..start + 1 + end];
                    symbols.push((name.to_string(), required));
                }
            }
        }
        symbols
    }

    /// Extract `#[no_mangle]` exported function names from Rust source text.
    fn parse_rust_exported_symbols(rust_source: &str) -> Vec<String> {
        let mut symbols = Vec::new();
        let mut next_is_export = false;
        for line in rust_source.lines() {
            let trimmed = line.trim();
            if trimmed == "#[no_mangle]" {
                next_is_export = true;
                continue;
            }
            if next_is_export {
                next_is_export = false;
                // Extract function name from lines like:
                //   pub unsafe extern "C" fn xdna_emu_create(...
                //   pub extern "C" fn xdna_emu_version() -> u32 {
                if let Some(fn_pos) = trimmed.find("fn ") {
                    let after_fn = &trimmed[fn_pos + 3..];
                    let name_end = after_fn.find('(').unwrap_or(after_fn.len());
                    let name = after_fn[..name_end].trim();
                    if !name.is_empty() {
                        symbols.push(name.to_string());
                    }
                }
            }
        }
        symbols
    }

    #[test]
    fn test_ffi_interface_completeness() {
        use std::path::PathBuf;

        // Navigate from the FFI crate back to the repo root.
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo_root = manifest_dir
            .join("../..")
            .canonicalize()
            .expect("cannot find repo root from FFI crate");
        let cpp_path = repo_root.join("xrt-plugin/src/transport_inprocess.cpp");
        let ffi_src_dir = manifest_dir.join("src");

        let cpp_source = std::fs::read_to_string(&cpp_path)
            .unwrap_or_else(|e| panic!("cannot read {}: {}", cpp_path.display(), e));

        // Scan all .rs files in this crate's src/ for exported symbols.
        let mut exported = Vec::new();
        for entry in std::fs::read_dir(&ffi_src_dir)
            .unwrap_or_else(|e| panic!("cannot read dir {}: {}", ffi_src_dir.display(), e))
        {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "rs") {
                let source = std::fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("cannot read {}: {}", path.display(), e));
                exported.extend(parse_rust_exported_symbols(&source));
            }
        }

        let expected = parse_cpp_expected_symbols(&cpp_source);

        assert!(!expected.is_empty(), "parsed zero symbols from C++ source -- parser broken?");
        assert!(!exported.is_empty(), "parsed zero symbols from Rust source -- parser broken?");

        let mut missing_required = Vec::new();
        let mut missing_optional = Vec::new();

        for (sym, required) in &expected {
            if !exported.iter().any(|e| e == sym) {
                if *required {
                    missing_required.push(sym.as_str());
                } else {
                    missing_optional.push(sym.as_str());
                }
            }
        }

        if !missing_required.is_empty() || !missing_optional.is_empty() {
            let mut msg = String::new();
            if !missing_required.is_empty() {
                msg.push_str(&format!(
                    "REQUIRED symbols missing from Rust FFI ({}):\n",
                    missing_required.len()
                ));
                for sym in &missing_required {
                    msg.push_str(&format!("  - {}\n", sym));
                }
            }
            if !missing_optional.is_empty() {
                msg.push_str(&format!(
                    "OPTIONAL symbols missing from Rust FFI ({}):\n",
                    missing_optional.len()
                ));
                for sym in &missing_optional {
                    msg.push_str(&format!("  - {}\n", sym));
                }
            }
            panic!(
                "FFI interface incomplete!\n\n{}\n\
                 C++ transport expects {} symbols, Rust exports {} symbols.\n\
                 Add the missing functions to crates/xdna-emu-ffi/src/.",
                msg,
                expected.len(),
                exported.len()
            );
        }

        // Informational: symbols we export but C++ doesn't consume.
        let extra: Vec<&str> = exported
            .iter()
            .filter(|e| !expected.iter().any(|(s, _)| s == *e))
            .map(|s| s.as_str())
            .collect();
        if !extra.is_empty() {
            eprintln!("Note: {} Rust FFI symbols not consumed by C++ transport: {:?}", extra.len(), extra);
        }
    }

    #[test]
    fn test_ffi_symbol_parser_helpers() {
        // Verify the parsers work on representative input.
        let cpp = r#"
            sym_create_ = resolve_required<fn_create>("xdna_emu_create");
            sym_version_ = resolve_required<fn_version>("xdna_emu_version");
            sym_alloc_buffer_ = resolve_optional<fn_alloc_buffer>("xdna_emu_alloc_buffer");
        "#;
        let symbols = parse_cpp_expected_symbols(cpp);
        assert_eq!(symbols.len(), 3);
        assert_eq!(symbols[0], ("xdna_emu_create".to_string(), true));
        assert_eq!(symbols[1], ("xdna_emu_version".to_string(), true));
        assert_eq!(symbols[2], ("xdna_emu_alloc_buffer".to_string(), false));

        let rust = r#"
            #[no_mangle]
            pub unsafe extern "C" fn xdna_emu_create() -> *mut XdnaEmuHandle {
            #[no_mangle]
            pub extern "C" fn xdna_emu_version() -> u32 {
        "#;
        let exports = parse_rust_exported_symbols(rust);
        assert_eq!(exports.len(), 2);
        assert_eq!(exports[0], "xdna_emu_create");
        assert_eq!(exports[1], "xdna_emu_version");
    }

    #[test]
    fn halt_reason_wedge_recovered_has_discriminant_four() {
        let r = XdnaEmuHaltReason::WedgeRecovered;
        assert_eq!(r as u32, 4);
    }
}
