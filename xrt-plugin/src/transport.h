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

    // -- Factory -------------------------------------------------------------

    /// Create an in-process transport by dlopen'ing the emulator shared lib.
    static std::unique_ptr<emu_transport> create_inprocess(
        const std::string& lib_path);
};

} // namespace xdna_emu
