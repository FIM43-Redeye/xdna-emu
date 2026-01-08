// SPDX-License-Identifier: Apache-2.0
// Emulator Bridge - Communication between mock XRT and xdna-emu
// This module handles invoking the Rust emulator and managing state

#ifndef MOCK_XRT_EMULATOR_BRIDGE_H_
#define MOCK_XRT_EMULATOR_BRIDGE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace xrt_emu {

/// Buffer descriptor for emulator communication
struct BufferDesc {
    uint64_t id;           // Unique buffer ID
    uint64_t device_addr;  // Address in device memory space
    size_t size;           // Buffer size in bytes
    void* host_ptr;        // Mapped host memory pointer
};

/// Execution result from emulator
struct ExecResult {
    bool success;
    int error_code;
    std::string error_message;
    uint64_t cycles_executed;
};

/// EmulatorBridge manages communication with the xdna-emu Rust emulator
class EmulatorBridge {
public:
    /// Get singleton instance
    static EmulatorBridge& instance();

    /// Initialize the emulator with device info
    ///
    /// @param device_name  Device name (e.g., "NPU Phoenix")
    /// @return true on success
    bool initialize(const std::string& device_name);

    /// Load an xclbin into the emulator
    ///
    /// @param xclbin_path  Path to xclbin file
    /// @param uuid  Output UUID of loaded xclbin
    /// @return true on success
    bool load_xclbin(const std::string& xclbin_path, uint8_t uuid[16]);

    /// Create a hardware context
    ///
    /// @param uuid  UUID of xclbin
    /// @return Context handle (0 on failure)
    uint32_t create_context(const uint8_t uuid[16]);

    /// Destroy a hardware context
    void destroy_context(uint32_t ctx_handle);

    /// Allocate a buffer
    ///
    /// @param size  Buffer size in bytes
    /// @param flags  Buffer flags
    /// @return Buffer descriptor
    BufferDesc allocate_buffer(size_t size, uint64_t flags);

    /// Free a buffer
    void free_buffer(uint64_t buffer_id);

    /// Sync buffer to device
    void sync_to_device(uint64_t buffer_id, size_t offset, size_t size);

    /// Sync buffer from device
    void sync_from_device(uint64_t buffer_id, size_t offset, size_t size);

    /// Execute a kernel
    ///
    /// @param ctx_handle  Context handle
    /// @param kernel_name  Kernel name
    /// @param instr_buffer  Instruction buffer ID
    /// @param instr_size  Number of instructions
    /// @param args  Buffer IDs for kernel arguments
    /// @return Execution result
    ExecResult execute(uint32_t ctx_handle,
                       const std::string& kernel_name,
                       uint64_t instr_buffer,
                       size_t instr_size,
                       const std::vector<uint64_t>& args);

    /// Get the path to the xdna-emu executable
    std::string get_emulator_path() const;

    /// Set the emulator path (for testing/development)
    void set_emulator_path(const std::string& path);

    /// Check if emulator is available
    bool is_available() const;

private:
    EmulatorBridge();
    ~EmulatorBridge();
    EmulatorBridge(const EmulatorBridge&) = delete;
    EmulatorBridge& operator=(const EmulatorBridge&) = delete;

    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace xrt_emu

#endif // MOCK_XRT_EMULATOR_BRIDGE_H_
