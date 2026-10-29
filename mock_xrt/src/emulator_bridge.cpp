// SPDX-License-Identifier: Apache-2.0
// Emulator Bridge implementation using xdna-emu FFI

#include "emulator_bridge.h"
#include "xdna_emu.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

namespace xrt_emu {

/// Implementation details
struct EmulatorBridge::Impl {
    XdnaEmuHandle* emu_handle = nullptr;
    std::string xclbin_path;
    uint64_t next_buffer_id = 1;
    uint64_t next_device_addr = 0x100000;  // Start of device address space
    std::unordered_map<uint64_t, BufferDesc> buffers;
    bool initialized = false;

    Impl() {
        // Create emulator instance
        emu_handle = xdna_emu_create();
        if (!emu_handle) {
            std::cerr << "[mock_xrt] ERROR: Failed to create emulator instance" << std::endl;
        }
    }

    ~Impl() {
        if (emu_handle) {
            xdna_emu_destroy(emu_handle);
        }
    }
};

EmulatorBridge& EmulatorBridge::instance() {
    static EmulatorBridge instance;
    return instance;
}

EmulatorBridge::EmulatorBridge() : m_impl(std::make_unique<Impl>()) {}

EmulatorBridge::~EmulatorBridge() = default;

bool EmulatorBridge::initialize(const std::string& device_name) {
    if (!m_impl->emu_handle) {
        return false;
    }
    m_impl->initialized = true;
    std::cerr << "[mock_xrt] Initialized emulator for device: " << device_name << std::endl;
    return true;
}

bool EmulatorBridge::load_xclbin(const std::string& xclbin_path, uint8_t uuid[16]) {
    if (!m_impl->emu_handle) {
        return false;
    }

    m_impl->xclbin_path = xclbin_path;

    XdnaEmuResult result = xdna_emu_load_xclbin(
        m_impl->emu_handle,
        xclbin_path.c_str(),
        uuid
    );

    if (result != XDNA_EMU_SUCCESS) {
        std::cerr << "[mock_xrt] Failed to load xclbin: " << xclbin_path
                  << " (error code " << result << ")" << std::endl;
        return false;
    }

    std::cerr << "[mock_xrt] Loaded xclbin: " << xclbin_path << std::endl;
    return true;
}

uint32_t EmulatorBridge::create_context(const uint8_t uuid[16]) {
    // Return a simple context handle
    static uint32_t next_ctx = 1;
    std::cerr << "[mock_xrt] Created hardware context" << std::endl;
    return next_ctx++;
}

void EmulatorBridge::destroy_context(uint32_t ctx_handle) {
    std::cerr << "[mock_xrt] Destroyed hardware context " << ctx_handle << std::endl;
}

BufferDesc EmulatorBridge::allocate_buffer(size_t size, uint64_t flags) {
    BufferDesc desc;
    desc.id = m_impl->next_buffer_id++;
    desc.size = size;
    desc.device_addr = m_impl->next_device_addr;
    m_impl->next_device_addr += (size + 4095) & ~4095ULL;  // Page-align

    // Allocate host memory
    desc.host_ptr = std::aligned_alloc(4096, (size + 4095) & ~4095ULL);
    std::memset(desc.host_ptr, 0, size);

    m_impl->buffers[desc.id] = desc;

    // Register region with emulator
    if (m_impl->emu_handle) {
        xdna_emu_alloc_host_region(
            m_impl->emu_handle,
            nullptr,
            desc.device_addr,
            size
        );
    }

    std::cerr << "[mock_xrt] Allocated buffer " << desc.id
              << " size=" << size
              << " device_addr=0x" << std::hex << desc.device_addr << std::dec
              << std::endl;

    return desc;
}

void EmulatorBridge::free_buffer(uint64_t buffer_id) {
    auto it = m_impl->buffers.find(buffer_id);
    if (it != m_impl->buffers.end()) {
        std::free(it->second.host_ptr);
        m_impl->buffers.erase(it);
        std::cerr << "[mock_xrt] Freed buffer " << buffer_id << std::endl;
    }
}

void EmulatorBridge::sync_to_device(uint64_t buffer_id, size_t offset, size_t size) {
    auto it = m_impl->buffers.find(buffer_id);
    if (it == m_impl->buffers.end()) {
        std::cerr << "[mock_xrt] ERROR: sync_to_device: buffer " << buffer_id << " not found" << std::endl;
        return;
    }

    const BufferDesc& desc = it->second;

    // Write buffer contents to emulator host memory
    if (m_impl->emu_handle) {
        const uint8_t* src = static_cast<const uint8_t*>(desc.host_ptr) + offset;
        xdna_emu_write_host_memory(
            m_impl->emu_handle,
            desc.device_addr + offset,
            src,
            size > 0 ? size : desc.size
        );
    }

    std::cerr << "[mock_xrt] Sync to device: buffer " << buffer_id
              << " offset=" << offset << " size=" << size << std::endl;
}

void EmulatorBridge::sync_from_device(uint64_t buffer_id, size_t offset, size_t size) {
    auto it = m_impl->buffers.find(buffer_id);
    if (it == m_impl->buffers.end()) {
        std::cerr << "[mock_xrt] ERROR: sync_from_device: buffer " << buffer_id << " not found" << std::endl;
        return;
    }

    BufferDesc& desc = it->second;

    // Read buffer contents from emulator host memory
    if (m_impl->emu_handle) {
        uint8_t* dst = static_cast<uint8_t*>(desc.host_ptr) + offset;
        size_t read_size = size > 0 ? size : desc.size;
        xdna_emu_read_host_memory(
            m_impl->emu_handle,
            desc.device_addr + offset,
            dst,
            read_size
        );
    }

    std::cerr << "[mock_xrt] Sync from device: buffer " << buffer_id
              << " offset=" << offset << " size=" << size << std::endl;
}

ExecResult EmulatorBridge::execute(uint32_t ctx_handle,
                                   const std::string& kernel_name,
                                   uint64_t instr_buffer_addr,
                                   size_t instr_size,
                                   const std::vector<uint64_t>& args) {
    ExecResult result;

    std::cerr << "[mock_xrt] Executing kernel: " << kernel_name << std::endl;
    std::cerr << "[mock_xrt]   instr_buffer=0x" << std::hex << instr_buffer_addr << std::dec
              << " instr_size=" << instr_size << std::endl;
    std::cerr << "[mock_xrt]   args: ";
    for (auto arg : args) {
        std::cerr << "0x" << std::hex << arg << std::dec << " ";
    }
    std::cerr << std::endl;

    if (!m_impl->emu_handle) {
        result.success = false;
        result.error_code = -1;
        result.error_message = "Emulator not initialized";
        return result;
    }

    // Find the instruction buffer by device address
    const BufferDesc* instr_buffer = nullptr;
    for (const auto& [id, desc] : m_impl->buffers) {
        if (desc.device_addr == instr_buffer_addr) {
            instr_buffer = &desc;
            break;
        }
    }

    if (!instr_buffer) {
        result.success = false;
        result.error_code = -2;
        result.error_message = "Instruction buffer not found";
        std::cerr << "[mock_xrt] ERROR: Instruction buffer at 0x" << std::hex
                  << instr_buffer_addr << std::dec << " not found" << std::endl;
        return result;
    }

    // Sync all input buffers to device
    for (auto& [id, desc] : m_impl->buffers) {
        sync_to_device(id, 0, desc.size);
    }

    // Set up host buffer mapping for DdrPatch instructions
    // args contains buffer addresses that correspond to arg_idx 0, 1, 2, ...
    // in the runtime_sequence's DdrPatch instructions
    xdna_emu_clear_host_buffers(m_impl->emu_handle);

    // Add data buffers - args now only contains buffer addresses
    for (size_t i = 0; i < args.size(); i++) {
        uint64_t buf_addr = args[i];
        // Find the buffer descriptor by device address
        for (const auto& [id, desc] : m_impl->buffers) {
            if (desc.device_addr == buf_addr) {
                xdna_emu_add_host_buffer(m_impl->emu_handle, desc.device_addr, desc.size);
                std::cerr << "[mock_xrt] Registered arg_idx " << i
                          << " -> buffer at 0x" << std::hex << desc.device_addr
                          << " size " << std::dec << desc.size << std::endl;
                break;
            }
        }
    }

    // Execute NPU instructions
    // The instruction buffer contains NPU commands that configure DMA and trigger transfers
    XdnaEmuResult emu_result = xdna_emu_execute_npu_instructions(
        m_impl->emu_handle,
        static_cast<const uint8_t*>(instr_buffer->host_ptr),
        instr_buffer->size
    );

    if (emu_result != XDNA_EMU_SUCCESS) {
        result.success = false;
        result.error_code = emu_result;
        result.error_message = "NPU instruction execution failed";
        std::cerr << "[mock_xrt] NPU instruction execution failed: " << emu_result << std::endl;
        return result;
    }

    // Sync cores from device state
    xdna_emu_sync_cores(m_impl->emu_handle);

    // Set max cycles and run
    xdna_emu_set_max_cycles(m_impl->emu_handle, 100000);
    XdnaEmuExecStatus status = xdna_emu_run(m_impl->emu_handle);

    if (status.result != XDNA_EMU_SUCCESS) {
        result.success = false;
        result.error_code = status.result;
        result.error_message = "Emulator run failed";
        std::cerr << "[mock_xrt] Emulator run failed: " << status.result << std::endl;
        return result;
    }

    // Sync output buffers from device
    for (auto& [id, desc] : m_impl->buffers) {
        sync_from_device(id, 0, desc.size);
    }

    result.success = true;
    result.error_code = 0;
    result.cycles_executed = status.cycles_executed;

    std::cerr << "[mock_xrt] Kernel execution completed: " << status.cycles_executed
              << " cycles, halted=" << (status.halted ? "yes" : "no") << std::endl;

    return result;
}

std::string EmulatorBridge::get_emulator_path() const {
    return "embedded";  // Using FFI, no separate executable
}

void EmulatorBridge::set_emulator_path(const std::string& path) {
    // No-op for FFI version
}

bool EmulatorBridge::is_available() const {
    return m_impl->emu_handle != nullptr;
}

} // namespace xrt_emu
