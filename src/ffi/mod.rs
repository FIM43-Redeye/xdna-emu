//! Foreign Function Interface for xdna-emu.
//!
//! This module provides C-callable functions for integrating xdna-emu
//! with C/C++ applications (like the mock XRT library).
//!
//! # Safety
//! All functions in this module use `unsafe` extern "C" ABI and must be
//! called with valid pointers. Null pointer checks are performed where
//! appropriate.
//!
//! # Memory Management
//! - Handles returned by `xdna_emu_*_create` functions must be freed
//!   with the corresponding `xdna_emu_*_destroy` function.
//! - Buffer data is copied during write/read operations; the caller
//!   retains ownership of their pointers.

use std::ffi::{CStr, c_char, c_void};
use std::path::Path;
use std::slice;
use std::sync::Mutex;

use crate::parser::{Xclbin, AiePartition, Cdo};
use crate::parser::xclbin::SectionKind;
use crate::parser::cdo::find_cdo_offset;
use crate::interpreter::engine::InterpreterEngine;
use crate::npu::{NpuInstructionStream, NpuExecutor};

/// Opaque handle to emulator state.
/// Wraps InterpreterEngine and related state.
pub struct XdnaEmuHandle {
    engine: InterpreterEngine,
    xclbin_path: Option<String>,
    npu_executor: NpuExecutor,
    max_cycles: u64,
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

/// Execution status returned by run functions.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct XdnaEmuExecStatus {
    pub result: XdnaEmuResult,
    pub cycles_executed: u64,
    pub halted: bool,
}

// Global lock for thread safety during initialization
static INIT_LOCK: Mutex<()> = Mutex::new(());

/// Create a new emulator instance.
///
/// # Safety
/// Returns a non-null handle on success, null on failure.
/// The returned handle must be freed with `xdna_emu_destroy`.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_create() -> *mut XdnaEmuHandle {
    let _lock = INIT_LOCK.lock().unwrap();

    // Initialize logging if not already done
    let _ = env_logger::try_init();

    let handle = Box::new(XdnaEmuHandle {
        engine: InterpreterEngine::new_npu1(),
        xclbin_path: None,
        npu_executor: NpuExecutor::new(),
        max_cycles: 100000,
    });

    Box::into_raw(handle)
}

/// Destroy an emulator instance.
///
/// # Safety
/// `handle` must be a valid pointer returned by `xdna_emu_create`,
/// or null (in which case this is a no-op).
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_destroy(handle: *mut XdnaEmuHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

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
            log::error!("Failed to parse xclbin: {}", e);
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

    // Find CDO offset
    let cdo_offset = match find_cdo_offset(pdi.pdi_image) {
        Some(o) => o,
        None => {
            log::error!("No CDO found in PDI");
            return XdnaEmuResult::ParseError;
        }
    };

    // Parse CDO
    let cdo = match Cdo::parse(&pdi.pdi_image[cdo_offset..]) {
        Ok(c) => c,
        Err(e) => {
            log::error!("Failed to parse CDO: {}", e);
            return XdnaEmuResult::ParseError;
        }
    };

    // Apply CDO to device
    if let Err(e) = handle.engine.device_mut().apply_cdo(&cdo) {
        log::error!("Failed to apply CDO: {}", e);
        return XdnaEmuResult::ExecutionError;
    }

    // Load ELF files from xclbin if present
    // The ELF files are embedded in the AIE partition
    // For now, we rely on the test directory containing ELF files separately

    handle.xclbin_path = Some(path_str.to_string());

    // Try to auto-load ELF files from the project directory
    // Convention: xclbin at <dir>/aie.xclbin, ELFs at <dir>/aie_arch.mlir.prj/*.elf
    let xclbin_path = std::path::Path::new(path_str);
    if let Some(parent) = xclbin_path.parent() {
        // Try common project directory names
        for prj_name in &["aie_arch.mlir.prj", "aie.mlir.prj"] {
            let prj_dir = parent.join(prj_name);
            if prj_dir.exists() {
                // Scan for ELF files named like: main_core_<col>_<row>.elf
                if let Ok(entries) = std::fs::read_dir(&prj_dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.extension().map(|e| e == "elf").unwrap_or(false) {
                            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                                // Parse tile coordinates from filename
                                // Pattern: main_core_<col>_<row> or core_<col>_<row>
                                let parts: Vec<&str> = stem.split('_').collect();
                                if parts.len() >= 4 && parts[parts.len()-3] == "core" {
                                    if let (Ok(col), Ok(row)) = (
                                        parts[parts.len()-2].parse::<u8>(),
                                        parts[parts.len()-1].parse::<u8>()
                                    ) {
                                        if let Ok(elf_data) = std::fs::read(&path) {
                                            if handle.engine.load_elf_bytes(col as usize, row as usize, &elf_data).is_ok() {
                                                log::info!("Loaded ELF for tile ({}, {}): {:?}", col, row, path);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            }
        }
    }

    log::info!("Loaded xclbin: {}", path_str);
    XdnaEmuResult::Success
}

/// Allocate a region in host memory.
///
/// # Safety
/// - `handle` must be valid
/// - `name` must be a valid null-terminated C string (or null for unnamed)
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_alloc_host_region(
    handle: *mut XdnaEmuHandle,
    name: *const c_char,
    address: u64,
    size: u64,
) -> XdnaEmuResult {
    if handle.is_null() {
        return XdnaEmuResult::InvalidHandle;
    }

    let handle = &mut *handle;

    let region_name = if name.is_null() {
        format!("region_{:x}", address)
    } else {
        match CStr::from_ptr(name).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => format!("region_{:x}", address),
        }
    };

    let host_mem = handle.engine.host_memory_mut();
    let _ = host_mem.allocate_region(&region_name, address, size as usize);

    // Also register with NPU executor for address patching
    handle.npu_executor.add_host_buffer(address, size as usize);

    log::debug!("Allocated host region '{}' at 0x{:x} size {}", region_name, address, size);
    XdnaEmuResult::Success
}

/// Write data to host memory at a specific address.
///
/// # Safety
/// - `handle` must be valid
/// - `data` must point to at least `size` bytes
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_write_host_memory(
    handle: *mut XdnaEmuHandle,
    address: u64,
    data: *const u8,
    size: u64,
) -> XdnaEmuResult {
    if handle.is_null() {
        return XdnaEmuResult::InvalidHandle;
    }
    if data.is_null() && size > 0 {
        return XdnaEmuResult::NullPointer;
    }

    let handle = &mut *handle;
    let data_slice = slice::from_raw_parts(data, size as usize);

    // Write as u32 words
    let host_mem = handle.engine.host_memory_mut();
    for (i, chunk) in data_slice.chunks(4).enumerate() {
        let mut word = [0u8; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        let value = u32::from_le_bytes(word);
        host_mem.write_u32(address + (i * 4) as u64, value);
    }

    log::debug!("Wrote {} bytes to host memory at 0x{:x}", size, address);
    XdnaEmuResult::Success
}

/// Read data from host memory at a specific address.
///
/// # Safety
/// - `handle` must be valid
/// - `data` must point to a buffer of at least `size` bytes
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_read_host_memory(
    handle: *mut XdnaEmuHandle,
    address: u64,
    data: *mut u8,
    size: u64,
) -> XdnaEmuResult {
    if handle.is_null() {
        return XdnaEmuResult::InvalidHandle;
    }
    if data.is_null() && size > 0 {
        return XdnaEmuResult::NullPointer;
    }

    let handle = &mut *handle;
    let data_slice = slice::from_raw_parts_mut(data, size as usize);

    // Read as u32 words
    let host_mem = handle.engine.host_memory_mut();
    for (i, chunk) in data_slice.chunks_mut(4).enumerate() {
        let value = host_mem.read_u32(address + (i * 4) as u64);
        let bytes = value.to_le_bytes();
        chunk.copy_from_slice(&bytes[..chunk.len()]);
    }

    log::debug!("Read {} bytes from host memory at 0x{:x}", size, address);
    XdnaEmuResult::Success
}

/// Clear host buffer list for NPU executor.
/// Call this before adding buffers for a new execution.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_clear_host_buffers(
    handle: *mut XdnaEmuHandle,
) -> XdnaEmuResult {
    if handle.is_null() {
        return XdnaEmuResult::InvalidHandle;
    }

    let handle = &mut *handle;
    handle.npu_executor = NpuExecutor::new();

    XdnaEmuResult::Success
}

/// Add a host buffer for NPU instruction address patching.
/// Buffers are added in order matching the runtime_sequence arguments.
///
/// # Safety
/// - `handle` must be valid
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_add_host_buffer(
    handle: *mut XdnaEmuHandle,
    address: u64,
    size: u64,
) -> XdnaEmuResult {
    if handle.is_null() {
        return XdnaEmuResult::InvalidHandle;
    }

    let handle = &mut *handle;
    handle.npu_executor.add_host_buffer(address, size as usize);

    log::debug!("Added host buffer: addr=0x{:x} size={}", address, size);
    XdnaEmuResult::Success
}

/// Execute NPU instructions.
///
/// This executes the instruction buffer which triggers DMA transfers
/// and configures the shim tiles.
///
/// # Safety
/// - `handle` must be valid
/// - `instr_data` must point to at least `instr_size` bytes
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_execute_npu_instructions(
    handle: *mut XdnaEmuHandle,
    instr_data: *const u8,
    instr_size: u64,
) -> XdnaEmuResult {
    if handle.is_null() {
        return XdnaEmuResult::InvalidHandle;
    }
    if instr_data.is_null() && instr_size > 0 {
        return XdnaEmuResult::NullPointer;
    }

    let handle = &mut *handle;
    let instr_slice = slice::from_raw_parts(instr_data, instr_size as usize);

    // Parse instruction stream
    let stream = match NpuInstructionStream::parse(instr_slice) {
        Ok(s) => s,
        Err(e) => {
            log::error!("Failed to parse NPU instructions: {}", e);
            return XdnaEmuResult::ParseError;
        }
    };

    log::info!("Executing {} NPU instructions", stream.instructions().len());

    // Execute instructions
    if let Err(e) = handle.npu_executor.execute(&stream, handle.engine.device_mut()) {
        log::error!("NPU instruction execution failed: {}", e);
        return XdnaEmuResult::ExecutionError;
    }

    XdnaEmuResult::Success
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

    if let Err(e) = handle.engine.load_elf_bytes(col as usize, row as usize, &elf_data) {
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
    handle.engine.sync_cores_from_device();

    XdnaEmuResult::Success
}

/// Set maximum cycles for execution.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_set_max_cycles(
    handle: *mut XdnaEmuHandle,
    max_cycles: u64,
) -> XdnaEmuResult {
    if handle.is_null() {
        return XdnaEmuResult::InvalidHandle;
    }

    let handle = &mut *handle;
    handle.max_cycles = max_cycles;

    XdnaEmuResult::Success
}

/// Run the emulator until completion or max cycles.
///
/// Returns execution status including whether the cores halted.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_run(handle: *mut XdnaEmuHandle) -> XdnaEmuExecStatus {
    if handle.is_null() {
        return XdnaEmuExecStatus {
            result: XdnaEmuResult::InvalidHandle,
            cycles_executed: 0,
            halted: false,
        };
    }

    let handle = &mut *handle;

    use crate::interpreter::engine::EngineStatus;

    let mut cycles = 0u64;
    let max = handle.max_cycles;

    log::info!("Running emulator (max {} cycles)", max);

    while cycles < max {
        handle.engine.step();
        cycles += 1;

        if handle.engine.status() == EngineStatus::Halted {
            log::info!("Cores halted after {} cycles", cycles);
            break;
        }

        // Check if DMA syncs are satisfied (execution complete)
        if handle.npu_executor.syncs_satisfied(handle.engine.device()) {
            log::info!("All DMA syncs satisfied after {} cycles", cycles);
            break;
        }
    }

    let halted = handle.engine.status() == EngineStatus::Halted
        || handle.npu_executor.syncs_satisfied(handle.engine.device());

    XdnaEmuExecStatus {
        result: XdnaEmuResult::Success,
        cycles_executed: cycles,
        halted,
    }
}

/// Get the last error message (for debugging).
///
/// # Safety
/// - `buffer` must point to at least `buffer_size` bytes
/// - Returns the number of bytes written (excluding null terminator)
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_get_error(
    buffer: *mut c_char,
    buffer_size: u64,
) -> u64 {
    // For now, just return empty - we'd need thread-local storage for proper error handling
    if !buffer.is_null() && buffer_size > 0 {
        *buffer = 0;
    }
    0
}

/// Get version information.
#[no_mangle]
pub extern "C" fn xdna_emu_version() -> u32 {
    // Version 0.1.0 = 0x000100
    0x000100
}
