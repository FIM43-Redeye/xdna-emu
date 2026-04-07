//! Host memory operations for the FFI interface.
//!
//! Allocation, read, write, and buffer management for host memory regions.

use std::ffi::{CStr, c_char};
use std::slice;

use super::{XdnaEmuHandle, XdnaEmuResult};

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
///
/// SAFETY: slice::from_raw_parts requires `data` to be valid for `size`
/// bytes. The null+size check above prevents null dereference. The caller
/// must ensure the buffer is accessible for the given length.
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
///
/// SAFETY: slice::from_raw_parts_mut requires `data` to be valid for
/// `size` bytes of writable memory. The null check prevents null deref.
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
    handle.npu_executor.set_host_buffers(Vec::new());

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
