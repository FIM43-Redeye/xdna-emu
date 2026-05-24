//! Firmware-emulation hooks.
//!
//! Real silicon's firmware runs on the mgmt-ERT and responds to driver
//! mailbox messages (MSG_OP_CREATE_CONTEXT, MSG_OP_DESTROY_CONTEXT,
//! MSG_OP_CONFIG_CU, etc.) by programming the array on the driver's
//! behalf.  Our plugin spans both the driver role (XRT requests) and the
//! firmware role (programming the emulated array), so the hooks in this
//! module replay the firmware-side register-write sequences that the
//! user-visible CDO never carries.
//!
//! All hooks go through `write_tile_register`, which is the same dispatch
//! path any other MMIO write uses.  The emulator cannot tell whether a
//! write originated in CDO, control packet, or firmware emulation.
//!
//! Each function returns `0` on success, a negative error code on failure,
//! and stores a human-readable message in `LAST_ERROR` for the caller to
//! retrieve via `xdna_emu_get_error`.

use super::{set_last_error, XdnaEmuHandle};

/// AM025 offset for Column_Clock_Control (shim row 0; bit 0 = enable).
/// Mirrors the private constant in `src/device/clock_control/mod.rs`;
/// kept hardcoded here so the FFI does not require re-exporting an
/// internal constant through `xdna-emu-core`'s public surface.
///
/// Source: aie-rt `xaiemlgbl_params.h` -- the same offset
/// `_XAieMl_RequestTiles` (device_aieml.c:309-385) writes when firmware
/// processes MSG_OP_CREATE_CONTEXT on real silicon.
const COLUMN_CLOCK_CONTROL_OFFSET: u32 = 0x000FFF20;

/// Emulate firmware's response to MSG_OP_CREATE_CONTEXT: ungate the
/// columns assigned to this context's partition.  On real silicon,
/// firmware issues `_XAieMl_RequestTiles` (aie-rt device_aieml.c:309)
/// which writes `Column_Clock_Control = 0x1` for each column in the
/// partition.  We mirror that exactly -- through `write_tile_register`
/// so the write goes through the normal dispatch path, identical to
/// any other MMIO from the emulator's POV.
///
/// Module_Clock_Control is intentionally NOT touched: per aie-rt
/// `_XAieMl_PmSetColumnClockBuffer` (device_aieml.c:272-295) firmware
/// only writes the column-level gate; per-tile module gates stay at
/// their AM025 reset values (compute 0x37, memtile 0x33, shim 0x3B),
/// which already enable the modules that boot active.
///
/// `num_tiles` is the XRT-native unit: SHIM's
/// `create_ctx_on_device` (xdna-driver `src/shim/hwctx.cpp:279`) packs
/// `num_tiles = num_cols * core_rows` into the ioctl arg.  We invert
/// that here: column count is `num_tiles / compute_rows`, where
/// `compute_rows = total_rows - 2` for AIE2 (1 shim row + 1 memtile
/// row; remainder are compute).
///
/// # Returns
/// - `0` on success
/// - `-1` if `handle` is null
/// - `-2` if `num_tiles` is not a multiple of `compute_rows` (malformed
///   request from upstream)
/// - `-3` if `start_col + num_col > total_cols` (partition spec
///   overflows the array; upstream should have allocated this on a
///   smaller-quantum)
///
/// # Safety
/// `handle` must be a valid pointer returned by `xdna_emu_create`.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_assign_partition(
    handle: *mut XdnaEmuHandle,
    start_col: u8,
    num_tiles: u32,
) -> i32 {
    if handle.is_null() {
        set_last_error("xdna_emu_assign_partition: null handle".to_string());
        return -1;
    }

    let handle = &mut *handle;
    let device = handle.engine.device_mut();
    let total_cols = device.cols();
    let total_rows = device.array.rows() as u32;
    // AIE2 layout: row 0 = shim, row 1 = memtile, rows 2.. = compute.
    // Matches the discriminator in clock_control's `clock_tile_kind_from_row`.
    let compute_rows = total_rows.saturating_sub(2);
    if compute_rows == 0 {
        set_last_error("xdna_emu_assign_partition: device has no compute rows".to_string());
        return -2;
    }
    if num_tiles % compute_rows != 0 {
        set_last_error(format!(
            "xdna_emu_assign_partition: num_tiles {} is not a multiple of compute_rows {}; \
             upstream SHIM packs num_tiles = num_cols * core_rows, so a non-multiple is malformed",
            num_tiles, compute_rows,
        ));
        return -2;
    }
    let num_col = num_tiles / compute_rows;
    let end = start_col as usize + num_col as usize;
    if end > total_cols {
        set_last_error(format!(
            "xdna_emu_assign_partition: partition spec overflows array \
             (start_col={}, num_tiles={} => num_col={}, end={}, array cols={})",
            start_col, num_tiles, num_col, end, total_cols,
        ));
        return -3;
    }

    for col in start_col..start_col + num_col as u8 {
        device.write_tile_register(col, 0, COLUMN_CLOCK_CONTROL_OFFSET, 0x1);
    }
    log::debug!(
        "xdna_emu_assign_partition: ungated {} cols [{}..{}] (num_tiles={}, compute_rows={})",
        num_col,
        start_col,
        end,
        num_tiles,
        compute_rows,
    );
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{xdna_emu_create, xdna_emu_destroy};

    // NPU1 has 4 compute rows, so num_tiles = num_cols * 4. The tests
    // below pass num_tiles values matching what real bridge tests emit.

    #[test]
    fn assign_partition_ungates_target_columns() {
        let handle = unsafe { xdna_emu_create() };
        {
            let h = unsafe { &*handle };
            for col in 0..5 {
                assert!(
                    !h.engine.device().array.clock().is_column_active(col),
                    "col {} should be gated at boot",
                    col
                );
            }
        }
        // num_tiles=16 = 4 cols * 4 compute rows. start_col=1, so cols 1..5.
        let rc = unsafe { xdna_emu_assign_partition(handle, 1, 16) };
        assert_eq!(rc, 0);
        {
            let h = unsafe { &*handle };
            assert!(!h.engine.device().array.clock().is_column_active(0), "col 0 not in partition");
            for col in 1..=4 {
                assert!(
                    h.engine.device().array.clock().is_column_active(col),
                    "col {} should be ungated by the partition",
                    col
                );
            }
        }
        unsafe { xdna_emu_destroy(handle) };
    }

    #[test]
    fn assign_partition_decodes_two_col_partition_from_num_tiles_8() {
        // Real bridge-test value: two_col uses num_tiles=8 -> 2 cols starting at 1.
        let handle = unsafe { xdna_emu_create() };
        let rc = unsafe { xdna_emu_assign_partition(handle, 1, 8) };
        assert_eq!(rc, 0);
        {
            let h = unsafe { &*handle };
            assert!(!h.engine.device().array.clock().is_column_active(0));
            assert!(h.engine.device().array.clock().is_column_active(1));
            assert!(h.engine.device().array.clock().is_column_active(2));
            assert!(!h.engine.device().array.clock().is_column_active(3), "col 3 outside partition");
            assert!(!h.engine.device().array.clock().is_column_active(4), "col 4 outside partition");
        }
        unsafe { xdna_emu_destroy(handle) };
    }

    #[test]
    fn assign_partition_rejects_non_multiple_of_compute_rows() {
        let handle = unsafe { xdna_emu_create() };
        let rc = unsafe { xdna_emu_assign_partition(handle, 0, 7) }; // 7 % 4 != 0
        assert_eq!(rc, -2);
        unsafe { xdna_emu_destroy(handle) };
    }

    #[test]
    fn assign_partition_rejects_partition_overflow() {
        let handle = unsafe { xdna_emu_create() };
        // num_tiles=24 -> num_col=6, but array has only 5 cols.
        let rc = unsafe { xdna_emu_assign_partition(handle, 0, 24) };
        assert_eq!(rc, -3);
        unsafe { xdna_emu_destroy(handle) };
    }

    #[test]
    fn assign_partition_null_handle() {
        let rc = unsafe { xdna_emu_assign_partition(std::ptr::null_mut(), 0, 4) };
        assert_eq!(rc, -1);
    }
}
