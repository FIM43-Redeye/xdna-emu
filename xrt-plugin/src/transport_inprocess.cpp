// SPDX-License-Identifier: MIT
//
// transport_inprocess.cpp -- In-process emulator transport via dlopen.

#include "transport_inprocess.h"

#include <dlfcn.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace xdna_emu {

// ---------------------------------------------------------------------------
// Factory (defined in emu_transport, implemented here)
// ---------------------------------------------------------------------------

std::unique_ptr<emu_transport>
emu_transport::create_inprocess(const std::string& lib_path)
{
    return std::make_unique<emu_transport_inprocess>(lib_path);
}

// ---------------------------------------------------------------------------
// Symbol resolution helpers
// ---------------------------------------------------------------------------

template <typename T>
T emu_transport_inprocess::resolve_required(const char* name)
{
    dlerror();  // clear previous error
    void* sym = dlsym(dl_handle_, name);
    const char* err = dlerror();
    if (err)
        throw std::runtime_error(
            std::string("dlsym failed for ") + name + ": " + err);
    return reinterpret_cast<T>(sym);
}

template <typename T>
T emu_transport_inprocess::resolve_optional(const char* name)
{
    dlerror();
    void* sym = dlsym(dl_handle_, name);
    dlerror();  // discard any error
    return reinterpret_cast<T>(sym);
}

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

std::string emu_transport_inprocess::last_error()
{
    if (!sym_get_error_)
        return "(no error function available)";

    char buf[512];
    uint64_t n = sym_get_error_(buf, sizeof(buf));
    if (n == 0)
        return "(unknown error)";
    return std::string(buf, n);
}

void emu_transport_inprocess::check(Result rc, const char* context)
{
    if (rc == 0)
        return;
    throw std::runtime_error(
        std::string(context) + " failed (rc=" + std::to_string(rc)
        + "): " + last_error());
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

emu_transport_inprocess::emu_transport_inprocess(const std::string& lib_path)
{
    // dlopen the emulator shared library.
    dl_handle_ = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!dl_handle_)
        throw std::runtime_error(
            std::string("dlopen failed: ") + dlerror());

    // Resolve required symbols -- these must be present.
    sym_create_             = resolve_required<fn_create>("xdna_emu_create");
    sym_destroy_            = resolve_required<fn_destroy>("xdna_emu_destroy");
    sym_load_xclbin_        = resolve_required<fn_load_xclbin>("xdna_emu_load_xclbin");
    sym_alloc_host_region_  = resolve_required<fn_alloc_host_region>("xdna_emu_alloc_host_region");
    sym_write_host_memory_  = resolve_required<fn_write_host_memory>("xdna_emu_write_host_memory");
    sym_read_host_memory_   = resolve_required<fn_read_host_memory>("xdna_emu_read_host_memory");
    sym_clear_host_buffers_ = resolve_required<fn_clear_host_buffers>("xdna_emu_clear_host_buffers");
    sym_add_host_buffer_    = resolve_required<fn_add_host_buffer>("xdna_emu_add_host_buffer");
    sym_exec_npu_instr_     = resolve_required<fn_exec_npu_instr>("xdna_emu_execute_npu_instructions");
    sym_sync_cores_         = resolve_required<fn_sync_cores>("xdna_emu_sync_cores");
    sym_set_max_cycles_     = resolve_required<fn_set_max_cycles>("xdna_emu_set_max_cycles");
    sym_run_                = resolve_required<fn_run>("xdna_emu_run");
    sym_get_error_          = resolve_required<fn_get_error>("xdna_emu_get_error");
    sym_version_            = resolve_required<fn_version>("xdna_emu_version");

    // Resolve optional/future symbols -- may not exist yet.
    sym_alloc_buffer_       = resolve_optional<fn_alloc_buffer>("xdna_emu_alloc_buffer");
    sym_free_buffer_        = resolve_optional<fn_free_buffer>("xdna_emu_free_buffer");
    sym_read_register_      = resolve_optional<fn_read_register>("xdna_emu_read_register");
    sym_write_register_     = resolve_optional<fn_write_register>("xdna_emu_write_register");
    sym_read_tile_mem_      = resolve_optional<fn_read_tile_mem>("xdna_emu_read_tile_memory");
    sym_write_tile_mem_     = resolve_optional<fn_write_tile_mem>("xdna_emu_write_tile_memory");
    sym_get_columns_        = resolve_optional<fn_get_columns>("xdna_emu_get_columns");
    sym_get_rows_           = resolve_optional<fn_get_rows>("xdna_emu_get_rows");
    sym_get_device_name_    = resolve_optional<fn_get_device_name>("xdna_emu_get_device_name");

    // Create the emulator instance.
    emu_ = sym_create_();
    if (!emu_) {
        dlclose(dl_handle_);
        dl_handle_ = nullptr;
        throw std::runtime_error("xdna_emu_create() returned null");
    }
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

emu_transport_inprocess::~emu_transport_inprocess()
{
    if (emu_ && sym_destroy_)
        sym_destroy_(emu_);

    if (dl_handle_)
        dlclose(dl_handle_);
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void emu_transport_inprocess::load_xclbin(const std::string& path,
                                          uint8_t uuid_out[16])
{
    Result rc = sym_load_xclbin_(emu_, path.c_str(), uuid_out);
    check(rc, "load_xclbin");
}

// ---------------------------------------------------------------------------
// Buffer management
// ---------------------------------------------------------------------------

uint64_t emu_transport_inprocess::alloc_buffer(size_t size)
{
    // Prefer the dedicated alloc_buffer FFI if available.
    if (sym_alloc_buffer_)
        return sym_alloc_buffer_(emu_, static_cast<uint64_t>(size));

    // Fallback: assign an address and create a host region manually.
    uint64_t addr = next_alloc_addr_;
    next_alloc_addr_ += (size + 0xFFF) & ~uint64_t(0xFFF);  // page-align

    Result rc = sym_alloc_host_region_(emu_, nullptr, addr,
                                       static_cast<uint64_t>(size));
    check(rc, "alloc_buffer (host_region fallback)");
    return addr;
}

void emu_transport_inprocess::free_buffer(uint64_t addr)
{
    if (sym_free_buffer_) {
        Result rc = sym_free_buffer_(emu_, addr);
        check(rc, "free_buffer");
        return;
    }
    // No fallback -- the existing FFI has no free_host_region.
    // Silently succeed; the memory will be reclaimed when the emulator
    // instance is destroyed.
}

void emu_transport_inprocess::write_memory(uint64_t addr, const void* data,
                                           size_t size)
{
    Result rc = sym_write_host_memory_(
        emu_, addr, static_cast<const uint8_t*>(data),
        static_cast<uint64_t>(size));
    check(rc, "write_memory");
}

void emu_transport_inprocess::read_memory(uint64_t addr, void* data,
                                          size_t size)
{
    Result rc = sym_read_host_memory_(
        emu_, addr, static_cast<uint8_t*>(data),
        static_cast<uint64_t>(size));
    check(rc, "read_memory");
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

void emu_transport_inprocess::execute(const void* instructions, size_t size)
{
    // Submit NPU instructions (patches addresses, configures shim DMA).
    Result rc = sym_exec_npu_instr_(
        emu_, static_cast<const uint8_t*>(instructions),
        static_cast<uint64_t>(size));
    check(rc, "execute (npu instructions)");

    // Sync cores so they see any new ELF/CDO state.
    rc = sym_sync_cores_(emu_);
    check(rc, "execute (sync cores)");

    // Run the emulator to completion.
    ExecStatus status = sym_run_(emu_);
    check(status.result, "execute (run)");
    last_run_complete_ = (status.halted != 0);
}

bool emu_transport_inprocess::poll_completion()
{
    return last_run_complete_;
}

// ---------------------------------------------------------------------------
// Debug / AIE tile access
// ---------------------------------------------------------------------------

uint32_t emu_transport_inprocess::read_reg(uint16_t col, uint16_t row,
                                           uint32_t addr)
{
    if (!sym_read_register_)
        throw std::runtime_error("read_reg: FFI function not available");
    return sym_read_register_(emu_, col, row, addr);
}

void emu_transport_inprocess::write_reg(uint16_t col, uint16_t row,
                                        uint32_t addr, uint32_t val)
{
    if (!sym_write_register_)
        throw std::runtime_error("write_reg: FFI function not available");
    Result rc = sym_write_register_(emu_, col, row, addr, val);
    check(rc, "write_reg");
}

void emu_transport_inprocess::read_tile_memory(uint16_t col, uint16_t row,
                                               uint32_t offset, uint32_t size,
                                               void* out)
{
    if (!sym_read_tile_mem_)
        throw std::runtime_error("read_tile_memory: FFI function not available");
    Result rc = sym_read_tile_mem_(emu_, col, row, offset, size, out);
    check(rc, "read_tile_memory");
}

void emu_transport_inprocess::write_tile_memory(uint16_t col, uint16_t row,
                                                uint32_t offset, uint32_t size,
                                                const void* data)
{
    if (!sym_write_tile_mem_)
        throw std::runtime_error("write_tile_memory: FFI function not available");
    Result rc = sym_write_tile_mem_(emu_, col, row, offset, size, data);
    check(rc, "write_tile_memory");
}

// ---------------------------------------------------------------------------
// Device metadata
// ---------------------------------------------------------------------------

uint8_t emu_transport_inprocess::get_columns()
{
    if (sym_get_columns_)
        return sym_get_columns_(emu_);
    return 5;  // Phoenix default
}

uint8_t emu_transport_inprocess::get_rows()
{
    if (sym_get_rows_)
        return sym_get_rows_(emu_);
    return 6;  // Phoenix default
}

std::string emu_transport_inprocess::get_device_name()
{
    if (sym_get_device_name_) {
        char buf[256];
        int32_t len = sym_get_device_name_(emu_, buf, sizeof(buf));
        if (len > 0)
            return std::string(buf, static_cast<size_t>(len));
    }
    return "NPU Phoenix (Emulated)";  // fallback
}

} // namespace xdna_emu
