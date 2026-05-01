// SPDX-License-Identifier: MIT
//
// transport.h -- Abstract interface between the XRT driver plugin and
// the emulator backend.
//
// In-process mode: dlopen's libxdna_emu.so (transport_inprocess).
// Future server mode: connects via Unix socket to xdna-emu --serve.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace xdna_emu {

class emu_transport {
public:
    virtual ~emu_transport() = default;

    // -- Lifecycle -----------------------------------------------------------

    /// Load an xclbin into the emulator, receiving the 16-byte UUID back.
    virtual void load_xclbin(const std::string& path,
                             uint8_t uuid_out[16]) = 0;

    /// Load raw PDI data (CDO + ELFs) into the emulator.
    virtual void load_pdi(const void* data, size_t size) = 0;

    /// Reset per-hw-context tile state (locks, DMAs, stream switches, cores)
    /// to mirror a real-HW column reset on hw_context teardown / re-creation.
    /// Host memory contents are preserved -- callers re-upload via the BO
    /// sync path. No-op if the backing FFI symbol is not available.
    virtual void reset_context() {}

    // -- Buffer management ---------------------------------------------------

    /// Allocate a host-visible buffer.  Returns the device address.
    virtual uint64_t alloc_buffer(size_t size) = 0;

    /// Free a previously-allocated buffer.
    virtual void free_buffer(uint64_t addr) = 0;

    /// Write caller data into host memory at @addr.
    virtual void write_memory(uint64_t addr, const void* data,
                              size_t size) = 0;

    /// Read host memory at @addr into caller buffer.
    virtual void read_memory(uint64_t addr, void* data,
                             size_t size) = 0;

    // -- Host buffer registration --------------------------------------------

    /// Clear the NPU executor's host buffer list.
    /// Must be called before add_host_buffer() for a new submission.
    virtual void clear_host_buffers() = 0;

    /// Register a host buffer for DdrPatch address patching.
    /// Buffers must be added in order matching the runtime_sequence
    /// arguments: add_host_buffer[0] = DdrPatch arg_idx 0, etc.
    virtual void add_host_buffer(uint64_t addr, uint64_t size) = 0;

    // -- Execution -----------------------------------------------------------

    /// Submit NPU instructions (from host memory) and run to completion.
    virtual void execute(const void* instructions, size_t size) = 0;

    /// Execute NPU instructions already in emulator memory at @dev_addr.
    /// This is the normal XRT path: instructions were sync'd to device
    /// memory via sync_bo, and the ert_packet tells us the address.
    virtual void execute_from_device(uint64_t dev_addr, uint32_t size) = 0;

    /// Check whether the last execution has completed.
    virtual bool poll_completion() = 0;

    // -- Debug / AIE tile access ---------------------------------------------

    /// Read a 32-bit register from a tile.
    virtual uint32_t read_reg(uint16_t col, uint16_t row,
                              uint32_t addr) = 0;

    /// Write a 32-bit register in a tile.
    virtual void write_reg(uint16_t col, uint16_t row,
                           uint32_t addr, uint32_t val) = 0;

    /// Bulk-read tile-local memory.
    virtual void read_tile_memory(uint16_t col, uint16_t row,
                                  uint32_t offset, uint32_t size,
                                  void* out) = 0;

    /// Bulk-write tile-local memory.
    virtual void write_tile_memory(uint16_t col, uint16_t row,
                                   uint32_t offset, uint32_t size,
                                   const void* data) = 0;

    // -- Device metadata -----------------------------------------------------

    /// Number of tile columns in the emulated device.
    virtual uint8_t get_columns() = 0;

    /// Number of tile rows in the emulated device.
    virtual uint8_t get_rows() = 0;

    /// Device name string (e.g. "NPU Phoenix (Emulated) [AIE2]").
    virtual std::string get_device_name() = 0;

    // -- Diagnostics ---------------------------------------------------------
    // Default implementations return "not available" so that future
    // transport backends (socket, etc.) work without implementing these.

    struct DmaChannelStats {
        uint64_t transfers_completed = 0;
        uint64_t bytes_transferred   = 0;
        uint64_t cycles_spent        = 0;
        uint64_t lock_wait_cycles    = 0;
    };

    /// Get the current value of a tile lock (-128 = not available).
    virtual int8_t get_lock_value(uint16_t col, uint16_t row,
                                  uint8_t lock_id)
    { return -128; }

    /// Get packed DMA channel state (bits 0-7 = state enum, 8-15 = lock_id).
    virtual uint32_t get_dma_channel_state(uint16_t col, uint16_t row,
                                           uint8_t is_s2mm,
                                           uint8_t channel_index)
    { return 0; }

    /// Get DMA channel statistics.  Returns true on success.
    virtual bool get_dma_channel_stats(uint16_t col, uint16_t row,
                                       uint8_t is_s2mm,
                                       uint8_t channel_index,
                                       DmaChannelStats& out)
    { return false; }

    /// Set the emulator's internal log level.  Returns true on success.
    virtual bool set_log_level(const std::string& level)
    { return false; }

    /// Dump a human-readable summary of a tile's state.
    virtual std::string dump_tile_state(uint16_t col, uint16_t row)
    { return {}; }

    // -- Factory -------------------------------------------------------------

    /// Create an in-process transport by dlopen'ing the emulator shared lib.
    static std::unique_ptr<emu_transport> create_inprocess(
        const std::string& lib_path);
};

} // namespace xdna_emu
