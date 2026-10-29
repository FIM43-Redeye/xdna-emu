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
    let device = handle.backend.as_interpreter().expect("Plan A: interpreter backend").device();
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

#[cfg(test)]
mod tests {
    use crate::{XdnaEmuResult, xdna_emu_create, xdna_emu_destroy, xdna_emu_reset_context};

    #[test]
    fn reset_context_transitions_failed_context_to_connected() {
        // After a wedge marks context Failed, reset_context restores Connected.
        let handle = unsafe { xdna_emu_create() };

        // Mark context 0 as Failed via direct device access (pub(crate) paths
        // are visible here since we are inside the xdna_emu_ffi crate).
        {
            let handle_mut = unsafe { &mut *handle };
            let device = handle_mut
                .backend
                .as_interpreter_mut()
                .expect("test interpreter backend")
                .device_mut();
            use xdna_emu_core::device::tdr::{TdrDiagnosis, WedgeReason};
            device.contexts[0].mark_failed(
                WedgeReason::Quiescent,
                TdrDiagnosis {
                    core_states: vec![],
                    dma_states: vec![],
                    data_in_flight: false,
                    pending_syncs: vec![],
                },
            );
            assert!(device.contexts[0].is_failed());
        }

        let rc = unsafe { xdna_emu_reset_context(handle, 0) };
        assert_eq!(rc, XdnaEmuResult::Success);

        {
            let handle_ref = unsafe { &*handle };
            assert!(handle_ref
                .backend
                .as_interpreter()
                .expect("test interpreter backend")
                .device()
                .contexts[0]
                .is_connected());
        }

        unsafe { xdna_emu_destroy(handle) };
    }
}
