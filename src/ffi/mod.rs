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

use std::ffi::{CStr, c_char};
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
    /// Next address to allocate for xdna_emu_alloc_buffer.
    /// Starts at a high address to avoid conflicts with user-specified regions.
    next_alloc_addr: u64,
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
        // Start auto-allocation at 0x8000_0000_0000 to avoid conflicts
        // with user-specified host regions (typically < 0x1_0000_0000).
        next_alloc_addr: 0x8000_0000_0000,
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

    // Load instructions for interleaved execution in xdna_emu_run().
    handle.npu_executor.load(&stream);

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
        // Advance NPU instruction execution (interleaved with engine step)
        {
            let (device, host_mem) = handle.engine.device_and_host_memory();
            if let crate::npu::AdvanceResult::Error(msg) = handle.npu_executor.try_advance(device, host_mem) {
                log::error!("NPU executor fatal: {}", msg);
                break;
            }
        }

        handle.engine.step();
        cycles += 1;

        if handle.engine.status() == EngineStatus::Halted {
            log::info!("Cores halted after {} cycles", cycles);
            break;
        }

        // Check if DMA syncs are satisfied (execution complete).
        // Only check after all NPU instructions have been processed.
        if handle.npu_executor.is_done()
            && handle.npu_executor.syncs_satisfied(handle.engine.device())
        {
            log::info!("All DMA syncs satisfied after {} cycles", cycles);
            break;
        }
    }

    let halted = handle.engine.status() == EngineStatus::Halted
        || (handle.npu_executor.is_done()
            && handle.npu_executor.syncs_satisfied(handle.engine.device()));

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

/// Allocate a host memory buffer of the given size.
///
/// Returns a page-aligned base address (u64) on success, or 0 on failure.
/// The address is automatically assigned from an internal allocator and
/// registered with the emulator's host memory system.
///
/// # Safety
/// - `handle` must be valid
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_alloc_buffer(
    handle: *mut XdnaEmuHandle,
    size: u64,
) -> u64 {
    if handle.is_null() || size == 0 {
        return 0;
    }

    let handle = &mut *handle;

    // Round size up to page boundary (4096 bytes).
    let page_size: u64 = 4096;
    let aligned_size = (size + page_size - 1) & !(page_size - 1);

    // The address is already page-aligned because next_alloc_addr starts
    // page-aligned and we always advance by page-aligned sizes.
    let addr = handle.next_alloc_addr;

    let host_mem = handle.engine.host_memory_mut();
    let name = format!("alloc_{:x}", addr);
    if host_mem.allocate_region(&name, addr, aligned_size as usize).is_err() {
        log::error!("Failed to allocate buffer at 0x{:x} size {}", addr, aligned_size);
        return 0;
    }

    // Also register with NPU executor for address patching.
    handle.npu_executor.add_host_buffer(addr, aligned_size as usize);

    handle.next_alloc_addr = addr + aligned_size;

    log::debug!("Allocated buffer at 0x{:x} size {}", addr, aligned_size);
    addr
}

/// Free a previously allocated host memory buffer.
///
/// Removes the region from host memory tracking. The underlying sparse
/// pages are not deallocated (they will be reclaimed when the emulator
/// handle is destroyed).
///
/// # Safety
/// - `handle` must be valid
/// - `addr` should be a value previously returned by `xdna_emu_alloc_buffer`
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_free_buffer(
    handle: *mut XdnaEmuHandle,
    addr: u64,
) {
    if handle.is_null() {
        return;
    }

    let handle = &mut *handle;
    let host_mem = handle.engine.host_memory_mut();
    if !host_mem.free_region(addr) {
        log::warn!("free_buffer: no region at 0x{:x}", addr);
    } else {
        log::debug!("Freed buffer at 0x{:x}", addr);
    }
}

/// Read a 32-bit register from a specific tile.
///
/// Uses the AIE address encoding to locate the tile and read the register
/// at the given offset within that tile.
///
/// # Safety
/// - `handle` must be valid
///
/// # Returns
/// The register value, or 0 if the tile coordinates are out of bounds.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_read_register(
    handle: *mut XdnaEmuHandle,
    col: u16,
    row: u16,
    reg_addr: u32,
) -> u32 {
    if handle.is_null() {
        return 0;
    }

    let handle = &mut *handle;
    let device = handle.engine.device_mut();

    if let Some(tile) = device.tile_mut(col as usize, row as usize) {
        tile.read_register(reg_addr)
    } else {
        log::warn!("read_register: tile ({}, {}) out of bounds", col, row);
        0
    }
}

/// Write a 32-bit value to a tile register.
///
/// Uses the AIE address encoding to locate the tile and write the value
/// at the given offset within that tile.
///
/// # Safety
/// - `handle` must be valid
///
/// # Returns
/// 0 on success, -1 if the handle is invalid, -2 if the tile is out of bounds.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_write_register(
    handle: *mut XdnaEmuHandle,
    col: u16,
    row: u16,
    reg_addr: u32,
    value: u32,
) -> i32 {
    if handle.is_null() {
        return -1;
    }

    let handle = &mut *handle;
    let device = handle.engine.device_mut();

    if let Some(tile) = device.tile_mut(col as usize, row as usize) {
        tile.write_register(reg_addr, value);
        0
    } else {
        log::warn!("write_register: tile ({}, {}) out of bounds", col, row);
        -2
    }
}

/// Read bytes from a tile's local data memory.
///
/// Copies `size` bytes starting at `offset` within the tile's data memory
/// into the caller-provided `out` buffer.
///
/// # Safety
/// - `handle` must be valid
/// - `out` must point to at least `size` bytes
///
/// # Returns
/// 0 on success, -1 if handle is invalid, -2 if tile is out of bounds,
/// -3 if the read would exceed memory bounds, -4 if `out` is null.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_read_tile_memory(
    handle: *mut XdnaEmuHandle,
    col: u16,
    row: u16,
    offset: u32,
    size: u32,
    out: *mut u8,
) -> i32 {
    if handle.is_null() {
        return -1;
    }
    if out.is_null() && size > 0 {
        return -4;
    }

    let handle = &mut *handle;
    let device = handle.engine.device_mut();

    let tile = match device.tile_mut(col as usize, row as usize) {
        Some(t) => t,
        None => {
            log::warn!("read_tile_memory: tile ({}, {}) out of bounds", col, row);
            return -2;
        }
    };

    let mem = tile.data_memory();
    let start = offset as usize;
    let end = start + size as usize;

    if end > mem.len() {
        log::warn!(
            "read_tile_memory: offset {} + size {} exceeds memory size {}",
            offset, size, mem.len()
        );
        return -3;
    }

    let out_slice = slice::from_raw_parts_mut(out, size as usize);
    out_slice.copy_from_slice(&mem[start..end]);
    0
}

/// Write bytes to a tile's local data memory.
///
/// Copies `size` bytes from the caller-provided `data` buffer into the
/// tile's data memory starting at `offset`.
///
/// # Safety
/// - `handle` must be valid
/// - `data` must point to at least `size` bytes
///
/// # Returns
/// 0 on success, -1 if handle is invalid, -2 if tile is out of bounds,
/// -3 if the write would exceed memory bounds, -4 if `data` is null.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_write_tile_memory(
    handle: *mut XdnaEmuHandle,
    col: u16,
    row: u16,
    offset: u32,
    size: u32,
    data: *const u8,
) -> i32 {
    if handle.is_null() {
        return -1;
    }
    if data.is_null() && size > 0 {
        return -4;
    }

    let handle = &mut *handle;
    let device = handle.engine.device_mut();

    let tile = match device.tile_mut(col as usize, row as usize) {
        Some(t) => t,
        None => {
            log::warn!("write_tile_memory: tile ({}, {}) out of bounds", col, row);
            return -2;
        }
    };

    let data_slice = slice::from_raw_parts(data, size as usize);
    if !tile.write_data(offset as usize, data_slice) {
        log::warn!(
            "write_tile_memory: offset {} + size {} exceeds memory bounds",
            offset, size
        );
        return -3;
    }

    0
}

/// Get the number of tile columns in the emulated device.
///
/// # Safety
/// - `handle` must be valid
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_get_columns(handle: *mut XdnaEmuHandle) -> u8 {
    if handle.is_null() {
        return 0;
    }
    let handle = &*handle;
    handle.engine.device().cols() as u8
}

/// Get the number of tile rows in the emulated device.
///
/// # Safety
/// - `handle` must be valid
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_get_rows(handle: *mut XdnaEmuHandle) -> u8 {
    if handle.is_null() {
        return 0;
    }
    let handle = &*handle;
    handle.engine.device().rows() as u8
}

/// Get the device name string (e.g. "NPU Phoenix (Emulated)").
///
/// Writes a null-terminated string into `buf`. Returns the number of
/// bytes written excluding the null terminator, or -1 on error.
///
/// # Safety
/// - `handle` must be valid
/// - `buf` must point to at least `buf_size` bytes
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_get_device_name(
    handle: *mut XdnaEmuHandle,
    buf: *mut c_char,
    buf_size: u32,
) -> i32 {
    if handle.is_null() || buf.is_null() || buf_size == 0 {
        return -1;
    }

    let handle = &*handle;
    let arch_name = handle.engine.device().arch_name();
    let name = format!("NPU Phoenix (Emulated) [{}]", arch_name);

    let bytes = name.as_bytes();
    let copy_len = bytes.len().min((buf_size - 1) as usize);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, copy_len);
    *buf.add(copy_len) = 0; // null terminator
    copy_len as i32
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
            let rc = xdna_emu_read_tile_memory(
                std::ptr::null_mut(), 0, 2, 0, 4, buf.as_mut_ptr(),
            );
            assert_eq!(rc, -1, "null handle should return -1");

            let data = [0u8; 4];
            let rc = xdna_emu_write_tile_memory(
                std::ptr::null_mut(), 0, 2, 0, 4, data.as_ptr(),
            );
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
}
