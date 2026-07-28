//! Firmware component API and legacy SHIM hook.
//!
//! Real silicon's firmware runs on the mgmt-ERT and responds to driver
//! mailbox messages (MSG_OP_CREATE_CONTEXT, MSG_OP_DESTROY_CONTEXT,
//! MSG_OP_CONFIG_CU, etc.) by programming the array on the driver's behalf.
//!
//! The byte-oriented functions below expose the real in-tree firmware
//! processor: explicit loading, natural boot against the interpreter engine's
//! existing array state, and direct access through firmware-programmed SRAM
//! aliases. They do not parse or synthesize mailbox commands.
//!
//! `xdna_emu_assign_partition` is retained separately for current XRT SHIM
//! bridge tests. It synthesizes one firmware side effect and is not the
//! unmodified-driver firmware path.

use super::{set_last_error, XdnaEmuHandle, XdnaEmuResult};
use xdna_emu_core::firmware::{FirmwareImage, FirmwareProcessor};

fn checked_firmware_size(value: u64) -> Option<usize> {
    usize::try_from(value).ok().filter(|&size| size <= isize::MAX as usize)
}

/// Load an explicit Phoenix management-firmware image into `handle`.
///
/// A successful load atomically replaces any prior processor. Failed parsing
/// or Phoenix load-map validation leaves the prior processor untouched.
///
/// # Safety
/// `handle` must be null or a live pointer returned by `xdna_emu_create`.
/// `firmware_data` must point to `firmware_size` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_load_firmware(
    handle: *mut XdnaEmuHandle,
    firmware_data: *const u8,
    firmware_size: u64,
) -> XdnaEmuResult {
    set_last_error(String::new());
    if handle.is_null() {
        set_last_error("xdna_emu_load_firmware: null handle".to_string());
        return XdnaEmuResult::InvalidHandle;
    }
    if firmware_data.is_null() {
        set_last_error("xdna_emu_load_firmware: null firmware_data".to_string());
        return XdnaEmuResult::NullPointer;
    }
    let Some(size) = checked_firmware_size(firmware_size) else {
        set_last_error("xdna_emu_load_firmware: firmware_size exceeds the Rust slice limit".to_string());
        return XdnaEmuResult::ParseError;
    };

    let handle = &mut *handle;
    if handle.backend.as_interpreter_mut().is_none() {
        set_last_error("xdna_emu_load_firmware: backend does not support firmware execution".to_string());
        return XdnaEmuResult::ExecutionError;
    }

    let bytes = std::slice::from_raw_parts(firmware_data, size);
    let processor = match FirmwareImage::parse(bytes).and_then(FirmwareProcessor::try_load_m2c) {
        Ok(processor) => processor,
        Err(error) => {
            set_last_error(format!("xdna_emu_load_firmware: {error}"));
            return XdnaEmuResult::ParseError;
        }
    };
    handle.firmware = Some(processor);
    XdnaEmuResult::Success
}

/// Run loaded firmware to its natural idle against the handle's array device.
///
/// # Safety
/// `handle` must be null or a live pointer returned by `xdna_emu_create`.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_boot_firmware(
    handle: *mut XdnaEmuHandle,
    max_instructions: u64,
) -> XdnaEmuResult {
    set_last_error(String::new());
    if handle.is_null() {
        set_last_error("xdna_emu_boot_firmware: null handle".to_string());
        return XdnaEmuResult::InvalidHandle;
    }
    if max_instructions == 0 {
        set_last_error("xdna_emu_boot_firmware: max_instructions must be nonzero".to_string());
        return XdnaEmuResult::ExecutionError;
    }

    let handle = &mut *handle;
    let XdnaEmuHandle { backend, firmware, .. } = handle;
    let Some(engine) = backend.as_interpreter_mut() else {
        set_last_error("xdna_emu_boot_firmware: backend does not support firmware execution".to_string());
        return XdnaEmuResult::ExecutionError;
    };
    let Some(processor) = firmware.as_mut() else {
        set_last_error("xdna_emu_boot_firmware: no firmware loaded".to_string());
        return XdnaEmuResult::ExecutionError;
    };

    let report = processor.boot_to_idle_with_device(engine.device_mut(), max_instructions);
    if !report.reached_idle {
        set_last_error(format!(
            "xdna_emu_boot_firmware: stopped before idle: pc={:#010x}, instructions={}, \
             unresolved_spin={:?}, unknown_op={:?}",
            report.last_pc, report.instrs_executed, report.unresolved_spin, report.unknown_op,
        ));
        return XdnaEmuResult::ExecutionError;
    }
    XdnaEmuResult::Success
}

/// Read a 32-bit word through the firmware-programmed host SRAM aliases.
///
/// # Safety
/// `handle` must be null or a live pointer returned by `xdna_emu_create`.
/// `value_out` must be null or writable for one `u32`.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_firmware_read_host_sram32(
    handle: *mut XdnaEmuHandle,
    device_address: u32,
    value_out: *mut u32,
) -> XdnaEmuResult {
    set_last_error(String::new());
    if handle.is_null() {
        set_last_error("xdna_emu_firmware_read_host_sram32: null handle".to_string());
        return XdnaEmuResult::InvalidHandle;
    }
    if value_out.is_null() {
        set_last_error("xdna_emu_firmware_read_host_sram32: null value_out".to_string());
        return XdnaEmuResult::NullPointer;
    }

    let handle = &mut *handle;
    if handle.backend.as_interpreter().is_none() {
        set_last_error(
            "xdna_emu_firmware_read_host_sram32: backend does not support firmware execution".to_string(),
        );
        return XdnaEmuResult::ExecutionError;
    }
    let Some(processor) = handle.firmware.as_ref() else {
        set_last_error("xdna_emu_firmware_read_host_sram32: no firmware loaded".to_string());
        return XdnaEmuResult::ExecutionError;
    };

    *value_out = processor.bus.host_sram_load32(device_address);
    XdnaEmuResult::Success
}

/// Write a 32-bit word through the firmware-programmed host SRAM aliases.
///
/// # Safety
/// `handle` must be null or a live pointer returned by `xdna_emu_create`.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_firmware_write_host_sram32(
    handle: *mut XdnaEmuHandle,
    device_address: u32,
    value: u32,
) -> XdnaEmuResult {
    set_last_error(String::new());
    if handle.is_null() {
        set_last_error("xdna_emu_firmware_write_host_sram32: null handle".to_string());
        return XdnaEmuResult::InvalidHandle;
    }

    let handle = &mut *handle;
    if handle.backend.as_interpreter().is_none() {
        set_last_error(
            "xdna_emu_firmware_write_host_sram32: backend does not support firmware execution".to_string(),
        );
        return XdnaEmuResult::ExecutionError;
    }
    let Some(processor) = handle.firmware.as_mut() else {
        set_last_error("xdna_emu_firmware_write_host_sram32: no firmware loaded".to_string());
        return XdnaEmuResult::ExecutionError;
    };

    processor.bus.host_sram_store32(device_address, value);
    XdnaEmuResult::Success
}

/// Emulate firmware's response to MSG_OP_CREATE_CONTEXT: ungate the
/// columns assigned to this context's partition.  On real silicon,
/// firmware issues `_XAieMl_RequestTiles` (aie-rt device_aieml.c:309)
/// which writes `Column_Clock_Control = 0x1` for each column in the
/// partition.  The actual register writes live in the core
/// (`DeviceState::assign_partition_columns`) so this XRT-plugin path and
/// the in-process `XclbinSuite` runner share one firmware implementation;
/// this hook only decodes the XRT-native `num_tiles` unit into a column
/// count and delegates.
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
    // Non-interpreter backends (aiesim) ungate partition columns themselves via
    // the cluster's CDO replay (Column_Clock_Control writes), so the
    // firmware-emulation hook is a no-op there -- only the in-process
    // interpreter needs the explicit ungate.
    let Some(device) = handle.backend.as_interpreter_mut().map(|i| i.device_mut()) else {
        return 0;
    };
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

    device.assign_partition_columns(start_col, num_col as u8);
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
    use crate::{xdna_emu_create, xdna_emu_destroy, xdna_emu_get_error, xdna_emu_reset_context, XdnaEmuResult};

    fn synthetic_m2c_image() -> Vec<u8> {
        let mut raw = vec![0u8; 0x2d101];
        raw[0x10..0x14].copy_from_slice(b"$PS1");
        let declared = raw.len() as u32;
        raw[0x14..0x18].copy_from_slice(&declared.to_le_bytes());
        raw
    }

    fn last_error() -> String {
        let mut buffer = [0i8; 256];
        let len = unsafe { xdna_emu_get_error(buffer.as_mut_ptr(), buffer.len() as u64) };
        let bytes = unsafe { std::slice::from_raw_parts(buffer.as_ptr().cast::<u8>(), len as usize) };
        String::from_utf8(bytes.to_vec()).expect("UTF-8 LAST_ERROR")
    }

    fn assert_last_error_contains(expected: &str) {
        let actual = last_error();
        assert!(actual.contains(expected), "LAST_ERROR {actual:?} does not contain {expected:?}");
    }

    #[test]
    fn firmware_size_rejects_lengths_above_rust_slice_limit() {
        assert_eq!(checked_firmware_size(isize::MAX as u64), Some(isize::MAX as usize),);
        assert_eq!(checked_firmware_size(isize::MAX as u64 + 1), None);
    }

    // NPU1 has 4 compute rows, so num_tiles = num_cols * 4. The tests
    // below pass num_tiles values matching what real bridge tests emit.

    #[test]
    fn assign_partition_ungates_target_columns() {
        let handle = unsafe { xdna_emu_create() };
        {
            let h = unsafe { &*handle };
            for col in 0..5 {
                assert!(
                    !h.backend
                        .as_interpreter()
                        .expect("test interpreter backend")
                        .device()
                        .array
                        .clock()
                        .is_column_active(col),
                    "col {} should be gated at boot",
                    col
                );
            }
        }
        // num_tiles=16 = 4 cols * 4 compute rows. start_col=1, so cols 1..=4.
        let rc = unsafe { xdna_emu_assign_partition(handle, 1, 16) };
        assert_eq!(rc, 0);
        {
            let h = unsafe { &*handle };
            assert!(
                !h.backend
                    .as_interpreter()
                    .expect("test interpreter backend")
                    .device()
                    .array
                    .clock()
                    .is_column_active(0),
                "col 0 not in partition"
            );
            for col in 1..=4 {
                assert!(
                    h.backend
                        .as_interpreter()
                        .expect("test interpreter backend")
                        .device()
                        .array
                        .clock()
                        .is_column_active(col),
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
            assert!(!h
                .backend
                .as_interpreter()
                .expect("test interpreter backend")
                .device()
                .array
                .clock()
                .is_column_active(0));
            assert!(h
                .backend
                .as_interpreter()
                .expect("test interpreter backend")
                .device()
                .array
                .clock()
                .is_column_active(1));
            assert!(h
                .backend
                .as_interpreter()
                .expect("test interpreter backend")
                .device()
                .array
                .clock()
                .is_column_active(2));
            assert!(
                !h.backend
                    .as_interpreter()
                    .expect("test interpreter backend")
                    .device()
                    .array
                    .clock()
                    .is_column_active(3),
                "col 3 outside partition"
            );
            assert!(
                !h.backend
                    .as_interpreter()
                    .expect("test interpreter backend")
                    .device()
                    .array
                    .clock()
                    .is_column_active(4),
                "col 4 outside partition"
            );
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

    #[test]
    fn firmware_api_rejects_nulls_missing_state_and_zero_budget() {
        let bytes = synthetic_m2c_image();
        assert_eq!(
            unsafe { xdna_emu_load_firmware(std::ptr::null_mut(), bytes.as_ptr(), bytes.len() as u64) },
            XdnaEmuResult::InvalidHandle,
        );
        assert_last_error_contains("xdna_emu_load_firmware: null handle");

        let handle = unsafe { xdna_emu_create() };
        assert_eq!(
            unsafe { xdna_emu_load_firmware(handle, std::ptr::null(), bytes.len() as u64) },
            XdnaEmuResult::NullPointer,
        );
        assert_last_error_contains("xdna_emu_load_firmware: null firmware_data");
        assert_eq!(
            unsafe {
                xdna_emu_load_firmware(
                    handle,
                    std::ptr::NonNull::<u8>::dangling().as_ptr(),
                    isize::MAX as u64 + 1,
                )
            },
            XdnaEmuResult::ParseError,
        );
        assert_last_error_contains("xdna_emu_load_firmware: firmware_size exceeds the Rust slice limit");

        let malformed = [0u8; 0x18];
        assert_eq!(
            unsafe { xdna_emu_load_firmware(handle, malformed.as_ptr(), malformed.len() as u64) },
            XdnaEmuResult::ParseError,
        );
        assert_last_error_contains("xdna_emu_load_firmware: bad firmware magic");

        let mut untouched = 0xfeed_face;
        assert_eq!(
            unsafe { xdna_emu_firmware_read_host_sram32(std::ptr::null_mut(), 0x030b_f000, &mut untouched) },
            XdnaEmuResult::InvalidHandle,
        );
        assert_last_error_contains("xdna_emu_firmware_read_host_sram32: null handle");
        assert_eq!(
            unsafe { xdna_emu_firmware_read_host_sram32(handle, 0x030b_f000, &mut untouched) },
            XdnaEmuResult::ExecutionError,
        );
        assert_last_error_contains("xdna_emu_firmware_read_host_sram32: no firmware loaded");
        assert_eq!(untouched, 0xfeed_face);

        assert_eq!(
            unsafe { xdna_emu_firmware_write_host_sram32(std::ptr::null_mut(), 0x030b_f000, 0) },
            XdnaEmuResult::InvalidHandle,
        );
        assert_last_error_contains("xdna_emu_firmware_write_host_sram32: null handle");
        assert_eq!(
            unsafe { xdna_emu_firmware_write_host_sram32(handle, 0x030b_f000, 0) },
            XdnaEmuResult::ExecutionError,
        );
        assert_last_error_contains("xdna_emu_firmware_write_host_sram32: no firmware loaded");

        assert_eq!(unsafe { xdna_emu_boot_firmware(std::ptr::null_mut(), 1) }, XdnaEmuResult::InvalidHandle,);
        assert_last_error_contains("xdna_emu_boot_firmware: null handle");
        assert_eq!(unsafe { xdna_emu_boot_firmware(handle, 1) }, XdnaEmuResult::ExecutionError,);
        assert_last_error_contains("xdna_emu_boot_firmware: no firmware loaded");

        assert_eq!(
            unsafe { xdna_emu_load_firmware(handle, bytes.as_ptr(), bytes.len() as u64) },
            XdnaEmuResult::Success,
        );
        assert_eq!(last_error(), "", "a successful firmware load must clear the prior error");
        assert_eq!(unsafe { xdna_emu_boot_firmware(handle, 0) }, XdnaEmuResult::ExecutionError,);
        assert_last_error_contains("xdna_emu_boot_firmware: max_instructions must be nonzero");
        assert_eq!(
            unsafe { xdna_emu_firmware_read_host_sram32(handle, 0x030b_f000, std::ptr::null_mut(),) },
            XdnaEmuResult::NullPointer,
        );
        assert_last_error_contains("xdna_emu_firmware_read_host_sram32: null value_out");
        unsafe { xdna_emu_destroy(handle) };
    }

    #[test]
    fn firmware_boot_routes_array_access_through_the_handles_device() {
        let mut bytes = synthetic_m2c_image();
        bytes[0x200..0x203].copy_from_slice(&[0x22, 0x61, 0x00]); // s32i a2, a1, 0
        let handle = unsafe { xdna_emu_create() };
        assert_eq!(
            unsafe { xdna_emu_load_firmware(handle, bytes.as_ptr(), bytes.len() as u64) },
            XdnaEmuResult::Success,
        );

        let address = 0x9c00_0000 + (1 << 25) + (2 << 20) + 0x0007_0000;
        unsafe {
            let processor = (*handle).firmware.as_mut().expect("loaded firmware");
            processor.cpu.regs.write_ar(1, address);
            processor.cpu.regs.write_ar(2, 0xabcd_1234);
        }

        assert_eq!(unsafe { xdna_emu_boot_firmware(handle, 1) }, XdnaEmuResult::ExecutionError);
        assert_last_error_contains("xdna_emu_boot_firmware: stopped before idle");
        assert_eq!(
            unsafe {
                (*handle)
                    .backend
                    .as_interpreter_mut()
                    .expect("interpreter")
                    .device_mut()
                    .read_tile_register(1, 2, 0x0007_0000)
            },
            0xabcd_1234,
        );
        unsafe { xdna_emu_destroy(handle) };
    }

    #[test]
    fn failed_load_preserves_the_existing_processor_and_array() {
        let bytes = synthetic_m2c_image();
        let handle = unsafe { xdna_emu_create() };
        unsafe {
            (*handle)
                .backend
                .as_interpreter_mut()
                .expect("interpreter")
                .device_mut()
                .write_tile_register(4, 0, 0x000f_ff20, 1);
        }

        assert_eq!(
            unsafe { xdna_emu_load_firmware(handle, bytes.as_ptr(), bytes.len() as u64) },
            XdnaEmuResult::Success,
        );
        assert_eq!(
            unsafe { xdna_emu_firmware_write_host_sram32(handle, 0x030b_f000, 0x1234_5678) },
            XdnaEmuResult::Success,
        );

        let malformed = [0u8; 0x18];
        assert_eq!(
            unsafe { xdna_emu_load_firmware(handle, malformed.as_ptr(), malformed.len() as u64) },
            XdnaEmuResult::ParseError,
        );
        assert_last_error_contains("xdna_emu_load_firmware: bad firmware magic");

        let mut value = 0;
        assert_eq!(
            unsafe { xdna_emu_firmware_read_host_sram32(handle, 0x030b_f000, &mut value) },
            XdnaEmuResult::Success,
        );
        assert_eq!(value, 0x1234_5678);
        assert_eq!(
            unsafe {
                (*handle)
                    .backend
                    .as_interpreter_mut()
                    .expect("interpreter")
                    .device_mut()
                    .read_tile_register(4, 0, 0x000f_ff20)
            },
            1,
        );
        unsafe { xdna_emu_destroy(handle) };
    }

    #[test]
    fn context_reset_preserves_loaded_firmware_state() {
        let bytes = synthetic_m2c_image();
        let handle = unsafe { xdna_emu_create() };
        assert_eq!(
            unsafe { xdna_emu_load_firmware(handle, bytes.as_ptr(), bytes.len() as u64) },
            XdnaEmuResult::Success,
        );
        assert_eq!(
            unsafe { xdna_emu_firmware_write_host_sram32(handle, 0x030b_f000, 0xcafe_babe) },
            XdnaEmuResult::Success,
        );
        assert_eq!(unsafe { xdna_emu_reset_context(handle, 0) }, XdnaEmuResult::Success,);
        let mut value = 0;
        assert_eq!(
            unsafe { xdna_emu_firmware_read_host_sram32(handle, 0x030b_f000, &mut value) },
            XdnaEmuResult::Success,
        );
        assert_eq!(value, 0xcafe_babe);
        unsafe { xdna_emu_destroy(handle) };
    }

    #[test]
    fn firmware_api_rejects_non_interpreter_backend_before_parsing() {
        let handle = unsafe { xdna_emu_create() };
        unsafe {
            (*handle).backend = Box::new(crate::backend::mock::MockBackend::default());
        }
        let malformed = [0u8; 1];
        assert_eq!(
            unsafe { xdna_emu_load_firmware(handle, malformed.as_ptr(), malformed.len() as u64) },
            XdnaEmuResult::ExecutionError,
        );
        assert_last_error_contains("xdna_emu_load_firmware: backend does not support firmware execution");
        assert_eq!(unsafe { xdna_emu_boot_firmware(handle, 1) }, XdnaEmuResult::ExecutionError);
        assert_last_error_contains("xdna_emu_boot_firmware: backend does not support firmware execution");

        let mut value = 0;
        assert_eq!(
            unsafe { xdna_emu_firmware_read_host_sram32(handle, 0x030b_f000, &mut value) },
            XdnaEmuResult::ExecutionError,
        );
        assert_last_error_contains(
            "xdna_emu_firmware_read_host_sram32: backend does not support firmware execution",
        );
        assert_eq!(
            unsafe { xdna_emu_firmware_write_host_sram32(handle, 0x030b_f000, 0) },
            XdnaEmuResult::ExecutionError,
        );
        assert_last_error_contains(
            "xdna_emu_firmware_write_host_sram32: backend does not support firmware execution",
        );
        unsafe { xdna_emu_destroy(handle) };
    }

    #[test]
    fn public_ffi_boots_pinned_firmware_and_exposes_genuine_alive_state() {
        let Ok(path) = std::env::var("XDNA_FIRMWARE") else {
            eprintln!("skip: set XDNA_FIRMWARE");
            return;
        };
        let digest = std::process::Command::new("sha256sum")
            .arg(&path)
            .output()
            .expect("run sha256sum");
        assert!(digest.status.success());
        let digest_stdout = String::from_utf8(digest.stdout).expect("sha256sum output");
        assert_eq!(
            digest_stdout.split_whitespace().next(),
            Some("d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e"),
        );
        let bytes = std::fs::read(path).expect("read firmware");
        let handle = unsafe { xdna_emu_create() };

        assert_eq!(
            unsafe { xdna_emu_load_firmware(handle, bytes.as_ptr(), bytes.len() as u64) },
            XdnaEmuResult::Success,
        );
        assert_eq!(unsafe { xdna_emu_boot_firmware(handle, 200_000) }, XdnaEmuResult::Success,);

        let mut value = 0;
        assert_eq!(
            unsafe { xdna_emu_firmware_read_host_sram32(handle, 0x030b_b020, &mut value) },
            XdnaEmuResult::Success,
        );
        assert_eq!(value, 0x5550_4e5f);
        assert_eq!(
            unsafe { xdna_emu_firmware_read_host_sram32(handle, 0x030b_f000, &mut value) },
            XdnaEmuResult::Success,
        );
        assert_eq!(value, 0x030b_b000);
        assert_eq!(
            unsafe { xdna_emu_firmware_write_host_sram32(handle, 0x030b_f000, 0) },
            XdnaEmuResult::Success,
        );
        assert_eq!(
            unsafe { xdna_emu_firmware_read_host_sram32(handle, 0x030b_f000, &mut value) },
            XdnaEmuResult::Success,
        );
        assert_eq!(value, 0);
        unsafe { xdna_emu_destroy(handle) };
    }
}
