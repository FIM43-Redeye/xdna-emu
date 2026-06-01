//! Loading and configuration functions for the FFI interface.
//!
//! Handles xclbin/PDI/ELF loading and core synchronization.

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

    // Mirror the xdna-driver allocator's "first available" choice: pick
    // the first candidate from the partition's start_columns array as the
    // physical start column.  The same shift gets applied to CDO ops
    // (apply_device_op) and to NPU executor ops (decode_npu_address /
    // Sync.column), so the load_xclbin path lands partition-aware
    // addressing end to end.  Empty list (older xclbins): leave at 0.
    let start_col = partition.start_columns().first().copied().unwrap_or(0);
    handle.backend.set_start_col(start_col as u8);
    log::info!("load_xclbin: physical start_col = {}", start_col);

    // Apply PDI through the shared golden path.
    let result = apply_pdi_data(handle, pdi.pdi_image);
    if result != XdnaEmuResult::Success {
        return result;
    }

    handle.xclbin_path = Some(path_str.to_string());
    log::info!("Loaded xclbin: {}", path_str);
    XdnaEmuResult::Success
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
    // start_col remains at whatever it was last set to (default 0).

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
/// amount.  The xdna-driver allocates this from `aie_partition.start_col_list`
/// at hw_context create time; the bridge-plugin path forwards that choice
/// here so the emulator mirrors HW addressing.  The xclbin path
/// (`xdna_emu_load_xclbin`) sets this internally from the parsed partition.
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
