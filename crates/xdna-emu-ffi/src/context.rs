//! Tier C FFI: per-context state accessor.

use super::XdnaEmuHandle;

/// Mirror of [`xdna_emu_core::device::context::ContextState`] discriminants
/// for the C ABI.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XdnaEmuContextState {
    Connected = 0,
    Stopped = 1,
    Failed = 2,
}

/// Read the current state and completion counter of a context.
///
/// # Returns
/// - `0` on success
/// - `-1` if `handle` is null or any out-pointer is null
/// - `-2` if `context_id` is out of range
///
/// # Safety
/// - `handle` must be valid
/// - `out_state` must point to writable `XdnaEmuContextState`
/// - `out_completed_counter` must point to writable `u64`
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_get_context_state(
    handle: *mut XdnaEmuHandle,
    context_id: u32,
    out_state: *mut XdnaEmuContextState,
    out_completed_counter: *mut u64,
) -> i32 {
    if handle.is_null() || out_state.is_null() || out_completed_counter.is_null() {
        return -1;
    }
    let handle = &*handle;
    let device = handle.engine.device();
    let ctx = match device.contexts.get(context_id as usize) {
        Some(c) => c,
        None => return -2,
    };

    use xdna_emu_core::device::context::ContextState;
    let state = match ctx.state {
        ContextState::Connected => XdnaEmuContextState::Connected,
        ContextState::Stopped => XdnaEmuContextState::Stopped,
        ContextState::Failed { .. } => XdnaEmuContextState::Failed,
    };
    *out_state = state;
    *out_completed_counter = ctx.completed_counter;
    0
}
