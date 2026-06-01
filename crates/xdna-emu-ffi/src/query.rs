//! Diagnostic and query functions for the FFI interface.
//!
//! Register access, tile memory, device info, lock/DMA state, and logging.

use std::ffi::{CStr, c_char};
use std::slice;

use super::{XdnaEmuHandle, set_last_error};

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
    let device = handle
        .backend
        .as_interpreter_mut()
        .expect("Plan A: interpreter backend")
        .device_mut();

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
    let device = handle
        .backend
        .as_interpreter_mut()
        .expect("Plan A: interpreter backend")
        .device_mut();

    if device.tile(col as usize, row as usize).is_some() {
        device.write_tile_register(col as u8, row as u8, reg_addr, value);
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
/// - `out` must point to at least `size` writable bytes
///
/// SAFETY: slice::from_raw_parts_mut on `out` requires the caller to
/// provide a valid, writable buffer of at least `size` bytes. Null is
/// checked. Tile bounds are validated before any memory access.
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
    let device = handle
        .backend
        .as_interpreter_mut()
        .expect("Plan A: interpreter backend")
        .device_mut();

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
        log::warn!("read_tile_memory: offset {} + size {} exceeds memory size {}", offset, size, mem.len());
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
    let device = handle
        .backend
        .as_interpreter_mut()
        .expect("Plan A: interpreter backend")
        .device_mut();

    let tile = match device.tile_mut(col as usize, row as usize) {
        Some(t) => t,
        None => {
            log::warn!("write_tile_memory: tile ({}, {}) out of bounds", col, row);
            return -2;
        }
    };

    let data_slice = slice::from_raw_parts(data, size as usize);
    if !tile.write_data(offset as usize, data_slice) {
        log::warn!("write_tile_memory: offset {} + size {} exceeds memory bounds", offset, size);
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
    handle.backend.cols() as u8
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
    handle.backend.rows() as u8
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
    let arch_name = handle.backend.arch_name();
    let name = format!("NPU Phoenix (Emulated) [{}]", arch_name);

    let bytes = name.as_bytes();
    let copy_len = bytes.len().min((buf_size - 1) as usize);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, copy_len);
    *buf.add(copy_len) = 0; // null terminator
    copy_len as i32
}

/// Set the emulator log level at runtime.
///
/// # Safety
/// - `level` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_set_log_level(level: *const c_char) -> i32 {
    if level.is_null() {
        return -1;
    }

    let level_str = match CStr::from_ptr(level).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let filter = match level_str {
        "error" => log::LevelFilter::Error,
        "warn" => log::LevelFilter::Warn,
        "info" => log::LevelFilter::Info,
        "debug" => log::LevelFilter::Debug,
        "trace" => log::LevelFilter::Trace,
        _ => {
            set_last_error(format!("set_log_level: unknown level '{}'", level_str));
            return -1;
        }
    };

    log::set_max_level(filter);
    0
}

/// Get the current value of a tile lock.
///
/// # Safety
/// - `handle` must be valid
///
/// # Returns
/// Lock value (-64..63), or -128 on error.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_get_lock_value(
    handle: *mut XdnaEmuHandle,
    col: u16,
    row: u16,
    lock_id: u8,
) -> i8 {
    if handle.is_null() {
        return -128;
    }

    let handle = &*handle;
    let device = handle.backend.as_interpreter().expect("Plan A: interpreter backend").device();

    let tile = match device.tile(col as usize, row as usize) {
        Some(t) => t,
        None => {
            set_last_error(format!("get_lock_value: tile ({}, {}) out of bounds", col, row));
            return -128;
        }
    };

    let id = lock_id as usize;
    if id < tile.locks.len() {
        tile.locks[id].value
    } else {
        set_last_error(format!(
            "get_lock_value: lock_id {} out of range (tile has {})",
            lock_id,
            tile.locks.len()
        ));
        -128
    }
}

/// Get the state of a DMA channel.
///
/// Returns a packed u32: bits 0-7 = state enum, bits 8-15 = lock_id
/// when state == WaitingForLock.
///
/// State values: 0=Idle, 1=Active, 2=Paused, 3=WaitingForLock, 4=Error.
///
/// The `is_s2mm` parameter selects the direction: 1 for S2MM channels
/// (indices 0..s2mm_count-1), 0 for MM2S channels (indices
/// s2mm_count..num_channels-1).  The `channel_index` is zero-based
/// within the selected direction.
///
/// # Safety
/// - `handle` must be valid
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_get_dma_channel_state(
    handle: *mut XdnaEmuHandle,
    col: u16,
    row: u16,
    is_s2mm: u8,
    channel_index: u8,
) -> u32 {
    use xdna_emu_core::device::dma::ChannelState;

    if handle.is_null() {
        return 0;
    }

    let handle = &*handle;
    let device = handle.backend.as_interpreter().expect("Plan A: interpreter backend").device();

    let engine = match device.array.dma_engine(col as u8, row as u8) {
        Some(e) => e,
        None => return 0,
    };

    // Convert direction + index to absolute channel ID.
    // S2MM channels come first, then MM2S.
    let abs_ch = if is_s2mm != 0 {
        channel_index
    } else {
        engine.s2mm_channel_count() as u8 + channel_index
    };

    let state = engine.channel_state(abs_ch);
    match state {
        ChannelState::Idle => 0,
        ChannelState::Active => 1,
        ChannelState::Paused => 2,
        ChannelState::WaitingForLock(lock_id) => 3 | ((lock_id as u32) << 8),
        ChannelState::Error => 4,
    }
}

/// DMA channel statistics (mirrors C struct XdnaEmuChannelStats).
#[repr(C)]
pub struct XdnaEmuChannelStats {
    pub transfers_completed: u64,
    pub bytes_transferred: u64,
    pub cycles_spent: u64,
    pub lock_wait_cycles: u64,
}

/// Get performance statistics for a DMA channel.
///
/// # Safety
/// - `handle` must be valid
/// - `out` must point to a valid XdnaEmuChannelStats
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_get_dma_channel_stats(
    handle: *mut XdnaEmuHandle,
    col: u16,
    row: u16,
    is_s2mm: u8,
    channel_index: u8,
    out: *mut XdnaEmuChannelStats,
) -> i32 {
    if handle.is_null() {
        return -1;
    }
    if out.is_null() {
        return -4;
    }

    let handle = &*handle;
    let device = handle.backend.as_interpreter().expect("Plan A: interpreter backend").device();

    let engine = match device.array.dma_engine(col as u8, row as u8) {
        Some(e) => e,
        None => return -2,
    };

    let abs_ch = if is_s2mm != 0 {
        channel_index
    } else {
        engine.s2mm_channel_count() as u8 + channel_index
    };

    match engine.channel_stats(abs_ch) {
        Some(stats) => {
            (*out).transfers_completed = stats.transfers_completed;
            (*out).bytes_transferred = stats.bytes_transferred;
            (*out).cycles_spent = stats.cycles_spent;
            (*out).lock_wait_cycles = stats.lock_wait_cycles;
            0
        }
        None => {
            set_last_error(format!("get_dma_channel_stats: channel {} out of range", abs_ch));
            -3
        }
    }
}

/// Dump a human-readable summary of a tile's state (locks + DMA).
///
/// # Safety
/// - `handle` must be valid
/// - `buf` must point to at least `buf_size` bytes
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_dump_tile_state(
    handle: *mut XdnaEmuHandle,
    col: u16,
    row: u16,
    buf: *mut c_char,
    buf_size: u32,
) -> i32 {
    use std::fmt::Write;

    if handle.is_null() || buf.is_null() || buf_size == 0 {
        return -1;
    }

    let handle = &*handle;
    let device = handle.backend.as_interpreter().expect("Plan A: interpreter backend").device();

    let tile = match device.tile(col as usize, row as usize) {
        Some(t) => t,
        None => {
            set_last_error(format!("dump_tile_state: tile ({}, {}) out of bounds", col, row));
            return -2;
        }
    };

    let mut out = String::with_capacity(512);

    // Lock values.
    let _ = write!(out, "tile({},{}) locks:", col, row);
    for (i, lock) in tile.locks.iter().enumerate() {
        if lock.value != 0 {
            let _ = write!(out, " [{}]={}", i, lock.value);
        }
    }
    out.push('\n');

    // DMA channel state.
    if let Some(engine) = device.array.dma_engine(col as u8, row as u8) {
        let s2mm_count = engine.s2mm_channel_count();
        let mm2s_count = engine.mm2s_channel_count();

        for i in 0..s2mm_count {
            let state = engine.channel_state(i as u8);
            let stats = engine.channel_stats(i as u8);
            let _ = write!(out, "  s2mm[{}]: {:?}", i, state);
            if let Some(s) = stats {
                let _ = write!(out, " xfr={} bytes={}", s.transfers_completed, s.bytes_transferred);
            }
            out.push('\n');
        }

        for i in 0..mm2s_count {
            let abs = s2mm_count as u8 + i as u8;
            let state = engine.channel_state(abs);
            let stats = engine.channel_stats(abs);
            let _ = write!(out, "  mm2s[{}]: {:?}", i, state);
            if let Some(s) = stats {
                let _ = write!(out, " xfr={} bytes={}", s.transfers_completed, s.bytes_transferred);
            }
            out.push('\n');
        }
    }

    let bytes = out.as_bytes();
    let needed = bytes.len();
    if needed >= buf_size as usize {
        // Buffer too small -- return required size.
        return needed as i32 + 1;
    }

    std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, needed);
    *buf.add(needed) = 0;
    needed as i32
}
