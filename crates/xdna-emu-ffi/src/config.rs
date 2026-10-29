//! Loading and configuration functions for the FFI interface.
//!
//! Handles xclbin/PDI/ELF loading, core synchronization, and the XRT shim's
//! partition-assignment hook.

use std::ffi::{CStr, c_char};
use std::slice;

use xdna_emu_core::parser::{Xclbin, AiePartition, Cdo};
use xdna_emu_core::parser::xclbin::SectionKind;
use xdna_emu_core::parser::cdo::find_cdo_offset;

use super::{XdnaEmuHandle, XdnaEmuResult, set_last_error};

/// Load an xclbin file into the emulator.
///
/// This parses the xclbin, extracts the CDO configuration, and applies
/// it to the device state.
///
/// # Safety
/// - `handle` must be valid
/// - `xclbin_path` must be a valid null-terminated C string
/// - `uuid_out` must point to a 16-byte buffer (or be null to skip)
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_load_xclbin(
    handle: *mut XdnaEmuHandle,
    xclbin_path: *const c_char,
    uuid_out: *mut u8,
) -> XdnaEmuResult {
    if handle.is_null() {
        return XdnaEmuResult::InvalidHandle;
    }
    if xclbin_path.is_null() {
        return XdnaEmuResult::NullPointer;
    }

    let handle = &mut *handle;

    // Convert path
    let path_str = match CStr::from_ptr(xclbin_path).to_str() {
        Ok(s) => s,
        Err(_) => return XdnaEmuResult::InvalidPath,
    };

    // Parse xclbin
    let xclbin = match Xclbin::from_file(path_str) {
        Ok(x) => x,
        Err(e) => {
            let msg = format!("Failed to parse xclbin: {}", e);
            log::error!("{}", msg);
            set_last_error(msg);
            return XdnaEmuResult::ParseError;
        }
    };

    // Extract and write UUID if requested
    if !uuid_out.is_null() {
        let uuid_slice = slice::from_raw_parts_mut(uuid_out, 16);
        let uuid = xclbin.uuid();
        uuid_slice.copy_from_slice(uuid.as_bytes());
    }

    // Find and apply CDO
    let partition_section = match xclbin.find_section(SectionKind::AiePartition) {
        Some(s) => s,
        None => {
            log::error!("No AIE partition section in xclbin");
            return XdnaEmuResult::ParseError;
        }
    };

    let partition = match AiePartition::parse(partition_section.data()) {
        Ok(p) => p,
        Err(e) => {
            log::error!("Failed to parse AIE partition: {}", e);
            return XdnaEmuResult::ParseError;
        }
    };

    // Get primary PDI
    let pdi = match partition.primary_pdi() {
        Some(p) => p,
        None => {
            log::error!("No primary PDI in partition");
            return XdnaEmuResult::ParseError;
        }
    };

    let start_col = place_partition(handle.backend.as_mut());
    log::info!(
        "load_xclbin: physical start_col = {} (xclbin virtual candidates {:?})",
        start_col,
        partition.start_columns()
    );
    // Diagnostic override (XDNA_AIESIM_FORCE_START_COL): relocate the partition to
    // a chosen physical column. Used to land NPU1 logical col 0 on a NoC shim
    // column when running NPU1 xclbins over the genuine Versal (VC2802) geometry,
    // isolating geometry-specific cluster behavior. Off by default.
    if let Ok(v) = std::env::var("XDNA_AIESIM_FORCE_START_COL") {
        if let Ok(forced) = v.parse::<u16>() {
            log::warn!("XDNA_AIESIM_FORCE_START_COL: forcing start_col to {forced}");
            handle.backend.set_start_col(forced as u8);
        }
    }

    // Apply PDI through the shared golden path.
    let result = apply_pdi_data(handle, pdi.pdi_image);
    if result != XdnaEmuResult::Success {
        return result;
    }

    handle.xclbin_path = Some(path_str.to_string());
    log::info!("Loaded xclbin: {}", path_str);
    XdnaEmuResult::Success
}

/// Place a single partition where the xdna-driver would: its first
/// partition slot, at the architecture's first application column
/// (`rq_part_reinit`: `start_col = first_col + i * num_col`, i = 0).
/// The shift applies to CDO ops (apply_device_op) and NPU executor ops
/// (decode_npu_address / Sync.column) alike. Always set explicitly, so a
/// reused handle never keeps a previous context's column.
pub(crate) fn place_partition(backend: &mut dyn crate::backend::NpuBackend) -> u8 {
    let start_col = backend.partition_column_start();
    backend.set_start_col(start_col);
    start_col
}

/// Apply PDI data to the emulator device.  This is the ONE golden path
/// for device configuration -- both `load_xclbin` and `load_pdi` converge
/// here.  Finds CDO within the bootgen container, parses it, and applies
/// it to the device.
pub(super) fn apply_pdi_data(handle: &mut XdnaEmuHandle, data: &[u8]) -> XdnaEmuResult {
    log::info!("apply_pdi_data: {} bytes, head={:02x?}", data.len(), &data[..data.len().min(16)]);

    // Find CDO within bootgen container.
    let cdo_offset = match find_cdo_offset(data) {
        Some(off) => {
            log::info!("  CDO found at offset {}", off);
            off
        }
        None => {
            // Try offset 0 as fallback (raw CDO without bootgen header).
            log::warn!("  No bootgen header found, trying offset 0");
            0
        }
    };

    // Parse CDO.
    let cdo = match Cdo::parse(&data[cdo_offset..]) {
        Ok(c) => {
            log::info!("  CDO: {} command words", c.command_length_words());
            c
        }
        Err(e) => {
            let msg =
                format!("Failed to parse CDO from PDI ({} bytes, offset {}): {}", data.len(), cdo_offset, e);
            log::error!("{}", msg);
            if data.len() >= 8 {
                log::error!(
                    "  At offset {}: {:02x?}",
                    cdo_offset,
                    &data[cdo_offset..data.len().min(cdo_offset + 20)]
                );
            }
            set_last_error(msg);
            return XdnaEmuResult::ParseError;
        }
    };

    // Apply CDO to device.
    if let Err(e) = handle.backend.apply_cdo(&cdo) {
        let msg = format!("Failed to apply CDO: {}", e);
        log::error!("{}", msg);
        set_last_error(msg);
        return XdnaEmuResult::ExecutionError;
    }

    XdnaEmuResult::Success
}

/// Load a raw PDI (Programmable Device Image) into the emulator.
///
/// The PDI contains a CDO stream that configures DMA descriptors, routing,
/// and loads ELF programs via blockwrite instructions.  This is the same
/// data that `load_xclbin` extracts from the xclbin container -- this
/// entry point lets the XRT bridge pass PDI data directly without needing
/// a file path.
///
/// # Safety
/// - `handle` must be valid
/// - `pdi_data` must point to at least `pdi_size` bytes
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_load_pdi(
    handle: *mut XdnaEmuHandle,
    pdi_data: *const u8,
    pdi_size: u64,
) -> XdnaEmuResult {
    if handle.is_null() {
        return XdnaEmuResult::InvalidHandle;
    }
    if pdi_data.is_null() && pdi_size > 0 {
        return XdnaEmuResult::NullPointer;
    }

    let handle = &mut *handle;
    let data = slice::from_raw_parts(pdi_data, pdi_size as usize);

    // This entry point doesn't see the xclbin's partition section, so
    // it can't pick a start_col itself.  The bridge-plugin is expected
    // to call xdna_emu_set_start_col() *before* xdna_emu_load_pdi() to
    // configure the partition shift.  If the caller skips the setter,
    // start_col remains at whatever it was last set to (the backend's
    // architecture-native default on a fresh handle).

    let result = apply_pdi_data(handle, data);
    if result == XdnaEmuResult::Success {
        log::info!("Loaded PDI ({} bytes)", pdi_size);
    }
    result
}

/// Load an ELF file for a specific tile.
///
/// # Safety
/// - `handle` must be valid
/// - `elf_path` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_load_elf(
    handle: *mut XdnaEmuHandle,
    col: u8,
    row: u8,
    elf_path: *const c_char,
) -> XdnaEmuResult {
    if handle.is_null() {
        return XdnaEmuResult::InvalidHandle;
    }
    if elf_path.is_null() {
        return XdnaEmuResult::NullPointer;
    }

    let handle = &mut *handle;

    let path_str = match CStr::from_ptr(elf_path).to_str() {
        Ok(s) => s,
        Err(_) => return XdnaEmuResult::InvalidPath,
    };

    let elf_data = match std::fs::read(path_str) {
        Ok(d) => d,
        Err(e) => {
            log::error!("Failed to read ELF file {}: {}", path_str, e);
            return XdnaEmuResult::InvalidPath;
        }
    };

    if let Err(e) = handle.backend.load_elf_bytes(col as usize, row as usize, &elf_data) {
        log::error!("Failed to load ELF: {}", e);
        return XdnaEmuResult::ParseError;
    }

    log::info!("Loaded ELF for tile ({}, {}): {}", col, row, path_str);
    XdnaEmuResult::Success
}

/// Sync cores from device state (after CDO/ELF loading).
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_sync_cores(handle: *mut XdnaEmuHandle) -> XdnaEmuResult {
    if handle.is_null() {
        return XdnaEmuResult::InvalidHandle;
    }

    let handle = &mut *handle;
    handle.backend.sync_cores_from_device();

    XdnaEmuResult::Success
}

/// Set the partition's physical start column.
///
/// CDO streams and runtime_sequence ops use partition-relative (logical)
/// column indices; consumers that need physical addressing -- the device
/// state's CDO applier and the NPU executor -- shift `tile.col` by this
/// amount.  The xdna-driver assigns this at hw_context create time from
/// its partition layout, `first_col + slot * width`; the xclbin's
/// `start_col_list` holds mlir-aie virtual-frame candidates and plays no
/// part.  The plugin, standing in for the driver, allocates from
/// `xdna_emu_get_partition_column_start` and forwards its choice here.
/// The xclbin path (`xdna_emu_load_xclbin`) sets it internally.
///
/// Must be called *before* `xdna_emu_load_pdi` if the caller wants the
/// shift applied to the loaded CDO.  Idempotent: callers can re-set it
/// for each new hw_context.
///
/// # Safety
/// - `handle` must be valid
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_set_start_col(handle: *mut XdnaEmuHandle, start_col: u8) -> XdnaEmuResult {
    if handle.is_null() {
        return XdnaEmuResult::InvalidHandle;
    }

    let handle = &mut *handle;
    handle.backend.set_start_col(start_col);
    log::info!("xdna_emu_set_start_col: start_col = {}", start_col);
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
    use super::{place_partition, xdna_emu_assign_partition};
    use crate::{xdna_emu_create, xdna_emu_destroy};
    use crate::backend::{InterpreterBackend, NpuBackend};
    use xdna_emu_core::interpreter::engine::InterpreterEngine;

    #[test]
    fn place_partition_uses_the_drivers_first_column_over_a_stale_one() {
        let mut be = InterpreterBackend::new(InterpreterEngine::new_npu1());
        // What the xclbin start_columns[0] used to set: a virtual-frame column.
        be.set_start_col(0);
        assert_eq!(place_partition(&mut be), 1);
        assert_eq!(be.engine.device().start_col, 1);
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
}
