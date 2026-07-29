//! Host memory operations for the FFI interface.
//!
//! Allocation, read, write, and buffer management for host memory regions.

use std::ffi::{CStr, c_char};
use std::slice;

use super::{XdnaEmuHandle, XdnaEmuResult, set_last_error};

fn checked_external_range(address: u64, size: u64) -> Option<usize> {
    let size = usize::try_from(size)
        .ok()
        .filter(|&size| size != 0 && size <= isize::MAX as usize)?;
    address.checked_add(size as u64)?;
    Some(size)
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

    let host_mem = handle.backend.host_memory_mut();
    let _ = host_mem.allocate_region(&region_name, address, size as usize);

    // Also register with NPU executor for address patching
    handle.backend.add_host_buffer(address, size as usize);

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
    let host_mem = handle.backend.host_memory_mut();
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
    let host_mem = handle.backend.host_memory_mut();
    for (i, chunk) in data_slice.chunks_mut(4).enumerate() {
        let value = host_mem.read_u32(address + (i * 4) as u64);
        let bytes = value.to_le_bytes();
        chunk.copy_from_slice(&bytes[..chunk.len()]);
    }

    log::debug!("Read {} bytes from host memory at 0x{:x}", size, address);
    XdnaEmuResult::Success
}

/// Map live caller-owned memory into the emulator host address space.
///
/// # Safety
/// - `handle` must be valid
/// - `data` must remain valid for reads and writes of `size` bytes until the
///   matching unmap or handle destruction
/// - external access must not race emulator access to the same bytes
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_map_host_memory(
    handle: *mut XdnaEmuHandle,
    address: u64,
    data: *mut u8,
    size: u64,
) -> XdnaEmuResult {
    set_last_error(String::new());
    if handle.is_null() {
        set_last_error("xdna_emu_map_host_memory: null handle".to_string());
        return XdnaEmuResult::InvalidHandle;
    }
    if data.is_null() {
        set_last_error("xdna_emu_map_host_memory: null data".to_string());
        return XdnaEmuResult::NullPointer;
    }
    let Some(size) = checked_external_range(address, size) else {
        set_last_error("xdna_emu_map_host_memory: invalid address or size".to_string());
        return XdnaEmuResult::BufferError;
    };

    let handle = &mut *handle;
    if let Err(error) = unsafe { handle.backend.host_memory_mut().map_external(address, data, size) } {
        set_last_error(format!("xdna_emu_map_host_memory: {error}"));
        return XdnaEmuResult::BufferError;
    }
    XdnaEmuResult::Success
}

/// Remove the exact live host-memory mapping.
///
/// # Safety
/// `handle` must be null or a live pointer returned by `xdna_emu_create`.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_unmap_host_memory(
    handle: *mut XdnaEmuHandle,
    address: u64,
    size: u64,
) -> XdnaEmuResult {
    set_last_error(String::new());
    if handle.is_null() {
        set_last_error("xdna_emu_unmap_host_memory: null handle".to_string());
        return XdnaEmuResult::InvalidHandle;
    }
    let Some(size) = checked_external_range(address, size) else {
        set_last_error("xdna_emu_unmap_host_memory: invalid address or size".to_string());
        return XdnaEmuResult::BufferError;
    };

    let handle = &mut *handle;
    if let Err(error) = handle.backend.host_memory_mut().unmap_external(address, size) {
        set_last_error(format!("xdna_emu_unmap_host_memory: {error}"));
        return XdnaEmuResult::BufferError;
    }
    XdnaEmuResult::Success
}

/// Clear host buffer list for NPU executor.
/// Call this before adding buffers for a new execution.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_clear_host_buffers(handle: *mut XdnaEmuHandle) -> XdnaEmuResult {
    if handle.is_null() {
        return XdnaEmuResult::InvalidHandle;
    }

    let handle = &mut *handle;
    handle.backend.clear_host_buffers();

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
    handle.backend.add_host_buffer(address, size as usize);

    log::debug!("Added host buffer: addr=0x{:x} size={}", address, size);
    XdnaEmuResult::Success
}

/// Allocate a host memory buffer of the given size.
///
/// Returns a page-aligned base address (u64) on success, or 0 on failure.
/// The address is automatically assigned from an internal allocator and
/// registered with the emulator's host memory system.
///
/// **Address reuse:** prefers a previously-freed range whose aligned size
/// exactly matches the request; only falls back to advancing
/// `next_alloc_addr` when no match exists. This mirrors what real HW kernel
/// drivers do (recycle physical BO addresses) and keeps DMA channels whose
/// internal pointers persist across runs (e.g. the trace shim-DMA channel)
/// pointed at the same DDR location across the bridge runner's per-run pool
/// teardown. Without it, every batch in batch-stdin mode gets a fresh BO,
/// the channel keeps writing to the stale previous address, and the new
/// run's trace BO reads back as zeros.
///
/// # Safety
/// - `handle` must be valid
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_alloc_buffer(handle: *mut XdnaEmuHandle, size: u64) -> u64 {
    if handle.is_null() || size == 0 {
        return 0;
    }

    let handle = &mut *handle;

    // Round size up to page boundary (4096 bytes).
    let page_size: u64 = 4096;
    let aligned_size = (size + page_size - 1) & !(page_size - 1);

    // Prefer a recycled address whose aligned size matches exactly.
    // Exact-size match (rather than first-fit) keeps the recycled region
    // self-contained -- no leftover slack to manage and no risk of the
    // returned region overlapping a still-live allocation.
    let recycled = handle
        .free_list
        .iter()
        .position(|&(_, sz)| sz == aligned_size)
        .map(|i| handle.free_list.swap_remove(i).0);

    let addr = recycled.unwrap_or(handle.next_alloc_addr);

    let host_mem = handle.backend.host_memory_mut();
    let name = format!("alloc_{:x}", addr);
    if host_mem.allocate_region(&name, addr, aligned_size as usize).is_err() {
        log::error!("Failed to allocate buffer at 0x{:x} size {}", addr, aligned_size);
        // Recycled addresses go back on the free list; bumping
        // next_alloc_addr would have stranded a fresh range.
        if recycled.is_some() {
            handle.free_list.push((addr, aligned_size));
        }
        return 0;
    }

    // Also register with NPU executor for address patching.
    handle.backend.add_host_buffer(addr, aligned_size as usize);

    if recycled.is_none() {
        handle.next_alloc_addr = addr + aligned_size;
    }

    log::debug!(
        "Allocated buffer at 0x{:x} size {} ({})",
        addr,
        aligned_size,
        if recycled.is_some() { "recycled" } else { "fresh" }
    );
    addr
}

/// Free a previously allocated host memory buffer.
///
/// Returns the address range to the free list for size-matched reuse on a
/// later `xdna_emu_alloc_buffer` call. The underlying sparse pages are not
/// deallocated (they are reclaimed when the emulator handle is destroyed),
/// but the host_memory region tracking is removed so the next allocator call
/// can re-register cleanly.
///
/// Returns `XdnaEmuResult::Success` on a successful free, `BufferError` if
/// the address has no live region, or `InvalidHandle` for a null handle.
/// (The XRT plugin's `transport_inprocess` calls this through a function
/// pointer typed `Result (*)(XdnaEmuHandle*, uint64_t)` and bails on
/// non-success, so a void return would surface garbage from EAX as a fake
/// failure.)
///
/// # Safety
/// - `handle` must be valid
/// - `addr` should be a value previously returned by `xdna_emu_alloc_buffer`
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_free_buffer(handle: *mut XdnaEmuHandle, addr: u64) -> XdnaEmuResult {
    if handle.is_null() {
        return XdnaEmuResult::InvalidHandle;
    }

    let handle = &mut *handle;
    let host_mem = handle.backend.host_memory_mut();
    // Look up the size BEFORE freeing so we can return the range to the
    // free list with the same aligned size we registered it with.
    let region_size = host_mem.region_at(addr).map(|r| r.size as u64);
    if !host_mem.free_region(addr) {
        log::warn!("free_buffer: no region at 0x{:x}", addr);
        return XdnaEmuResult::BufferError;
    }
    log::debug!("Freed buffer at 0x{:x}", addr);

    if let Some(sz) = region_size {
        handle.free_list.push((addr, sz));
    }
    XdnaEmuResult::Success
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{xdna_emu_create, xdna_emu_destroy, xdna_emu_read_host_memory, xdna_emu_write_host_memory};

    #[test]
    fn external_host_mapping_rejects_invalid_ranges_and_requires_exact_unmap() {
        let mut bytes = [0u8; 8];
        assert_eq!(
            unsafe {
                xdna_emu_map_host_memory(
                    std::ptr::null_mut(),
                    0x6000_0000,
                    bytes.as_mut_ptr(),
                    bytes.len() as u64,
                )
            },
            XdnaEmuResult::InvalidHandle
        );

        let handle = unsafe { xdna_emu_create() };
        assert_eq!(
            unsafe {
                xdna_emu_map_host_memory(handle, 0x6000_0000, std::ptr::null_mut(), bytes.len() as u64)
            },
            XdnaEmuResult::NullPointer
        );
        assert_eq!(
            unsafe { xdna_emu_map_host_memory(handle, 0x6000_0000, bytes.as_mut_ptr(), 0) },
            XdnaEmuResult::BufferError
        );
        assert_eq!(
            unsafe { xdna_emu_map_host_memory(handle, u64::MAX - 1, bytes.as_mut_ptr(), bytes.len() as u64) },
            XdnaEmuResult::BufferError
        );
        assert_eq!(
            unsafe {
                xdna_emu_map_host_memory(
                    handle,
                    0x6000_0000,
                    std::ptr::NonNull::<u8>::dangling().as_ptr(),
                    isize::MAX as u64 + 1,
                )
            },
            XdnaEmuResult::BufferError
        );

        assert_eq!(
            unsafe { xdna_emu_map_host_memory(handle, 0x6000_0000, bytes.as_mut_ptr(), bytes.len() as u64,) },
            XdnaEmuResult::Success
        );
        assert_eq!(
            unsafe { xdna_emu_map_host_memory(handle, 0x6000_0004, bytes.as_mut_ptr(), bytes.len() as u64,) },
            XdnaEmuResult::BufferError
        );
        assert_eq!(unsafe { xdna_emu_unmap_host_memory(handle, 0x6000_0000, 4) }, XdnaEmuResult::BufferError);
        assert_eq!(
            unsafe { xdna_emu_unmap_host_memory(handle, 0x6000_0000, bytes.len() as u64) },
            XdnaEmuResult::Success
        );
        assert_eq!(
            unsafe { xdna_emu_unmap_host_memory(handle, 0x6000_0000, bytes.len() as u64) },
            XdnaEmuResult::BufferError
        );
        assert_eq!(
            unsafe { xdna_emu_unmap_host_memory(std::ptr::null_mut(), 0, 1) },
            XdnaEmuResult::InvalidHandle
        );
        unsafe { xdna_emu_destroy(handle) };
    }

    #[test]
    fn external_host_mapping_is_directly_coherent_with_existing_memory_io() {
        let handle = unsafe { xdna_emu_create() };
        let mut mapped = [0u8; 8];
        assert_eq!(
            unsafe {
                xdna_emu_map_host_memory(handle, 0x6000_0000, mapped.as_mut_ptr(), mapped.len() as u64)
            },
            XdnaEmuResult::Success
        );

        let input = [1u8, 2, 3, 4];
        assert_eq!(
            unsafe { xdna_emu_write_host_memory(handle, 0x6000_0004, input.as_ptr(), input.len() as u64,) },
            XdnaEmuResult::Success
        );
        assert_eq!(mapped[4..], input);

        mapped[..4].copy_from_slice(&[5, 6, 7, 8]);
        let mut output = [0u8; 4];
        assert_eq!(
            unsafe {
                xdna_emu_read_host_memory(handle, 0x6000_0000, output.as_mut_ptr(), output.len() as u64)
            },
            XdnaEmuResult::Success
        );
        assert_eq!(output, [5, 6, 7, 8]);

        assert_eq!(
            unsafe { xdna_emu_unmap_host_memory(handle, 0x6000_0000, mapped.len() as u64) },
            XdnaEmuResult::Success
        );
        unsafe { xdna_emu_destroy(handle) };
    }
}
