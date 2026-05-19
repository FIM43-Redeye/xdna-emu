// SPDX-License-Identifier: MIT
//
// transport_inprocess.cpp -- In-process emulator transport via dlopen.

#include "transport_inprocess.h"

#include <dlfcn.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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
    sym_load_pdi_           = resolve_required<fn_load_pdi>("xdna_emu_load_pdi");
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
    sym_reset_context_      = resolve_optional<fn_reset_context>("xdna_emu_reset_context");
    sym_set_start_col_      = resolve_optional<fn_set_start_col>("xdna_emu_set_start_col");
    sym_read_register_      = resolve_optional<fn_read_register>("xdna_emu_read_register");
    sym_write_register_     = resolve_optional<fn_write_register>("xdna_emu_write_register");
    sym_read_tile_mem_      = resolve_optional<fn_read_tile_mem>("xdna_emu_read_tile_memory");
    sym_write_tile_mem_     = resolve_optional<fn_write_tile_mem>("xdna_emu_write_tile_memory");
    sym_get_columns_        = resolve_optional<fn_get_columns>("xdna_emu_get_columns");
    sym_get_rows_           = resolve_optional<fn_get_rows>("xdna_emu_get_rows");
    sym_get_device_name_    = resolve_optional<fn_get_device_name>("xdna_emu_get_device_name");

    // Diagnostic queries (optional -- may not exist in older builds).
    sym_get_lock_value_     = resolve_optional<fn_get_lock_value>("xdna_emu_get_lock_value");
    sym_get_dma_ch_state_   = resolve_optional<fn_get_dma_ch_state>("xdna_emu_get_dma_channel_state");
    sym_get_dma_ch_stats_   = resolve_optional<fn_get_dma_ch_stats>("xdna_emu_get_dma_channel_stats");
    sym_set_log_level_      = resolve_optional<fn_set_log_level>("xdna_emu_set_log_level");
    sym_dump_tile_state_    = resolve_optional<fn_dump_tile_state>("xdna_emu_dump_tile_state");

    // Create the emulator instance.
    emu_ = sym_create_();
    if (!emu_) {
        dlclose(dl_handle_);
        dl_handle_ = nullptr;
        throw std::runtime_error("xdna_emu_create() returned null");
    }

    // Apply cycle budget from XDNA_EMU_MAX_CYCLES (0 or unset = unbounded).
    if (const char* env = std::getenv("XDNA_EMU_MAX_CYCLES")) {
        char* end = nullptr;
        // Reject negative values (strtoull silently wraps them to UINT64_MAX,
        // which would be interpreted as a "huge but finite" budget rather than
        // the likely-intended "unlimited").
        if (env[0] == '-') {
            std::cerr << "[xdna-emu] XDNA_EMU_MAX_CYCLES='" << env
                      << "' is negative; ignoring (use 0 for unbounded)\n";
        } else {
            uint64_t val = std::strtoull(env, &end, 10);
            if (end == env || *end != '\0') {
                std::cerr << "[xdna-emu] XDNA_EMU_MAX_CYCLES='" << env
                          << "' unparseable; ignoring\n";
            } else {
                max_cycles_budget_ = val;
                sym_set_max_cycles_(emu_, val);
            }
        }
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
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
    Result rc = sym_load_xclbin_(emu_, path.c_str(), uuid_out);
    check(rc, "load_xclbin");
}

void emu_transport_inprocess::load_pdi(const void* data, size_t size)
{
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
    Result rc = sym_load_pdi_(emu_,
                              static_cast<const uint8_t*>(data),
                              static_cast<uint64_t>(size));
    check(rc, "load_pdi");
}

void emu_transport_inprocess::set_start_col(uint8_t start_col)
{
    if (!sym_set_start_col_)
        return;  // Older emulator builds: no-op fallback.
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
    Result rc = sym_set_start_col_(emu_, start_col);
    check(rc, "set_start_col");
}

// ---------------------------------------------------------------------------
// Buffer management
// ---------------------------------------------------------------------------

uint64_t emu_transport_inprocess::alloc_buffer(size_t size)
{
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
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
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
    if (sym_free_buffer_) {
        Result rc = sym_free_buffer_(emu_, addr);
        check(rc, "free_buffer");
        return;
    }
    // No fallback -- the existing FFI has no free_host_region.
    // Silently succeed; the memory will be reclaimed when the emulator
    // instance is destroyed.
}

void emu_transport_inprocess::reset_context()
{
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
    // Optional symbol -- older emulator builds will not export it. The
    // call site (create_ctx) has been wired up unconditionally; if the
    // symbol is missing we silently fall through, matching the prior
    // behavior where the EMU had no context-reset hook.
    if (!sym_reset_context_)
        return;
    Result rc = sym_reset_context_(emu_);
    check(rc, "reset_context");
}

void emu_transport_inprocess::write_memory(uint64_t addr, const void* data,
                                           size_t size)
{
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
    Result rc = sym_write_host_memory_(
        emu_, addr, static_cast<const uint8_t*>(data),
        static_cast<uint64_t>(size));
    check(rc, "write_memory");
}

void emu_transport_inprocess::read_memory(uint64_t addr, void* data,
                                          size_t size)
{
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
    Result rc = sym_read_host_memory_(
        emu_, addr, static_cast<uint8_t*>(data),
        static_cast<uint64_t>(size));
    check(rc, "read_memory");
}

// ---------------------------------------------------------------------------
// Host buffer registration
// ---------------------------------------------------------------------------

void emu_transport_inprocess::clear_host_buffers()
{
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
    Result rc = sym_clear_host_buffers_(emu_);
    check(rc, "clear_host_buffers");
}

void emu_transport_inprocess::add_host_buffer(uint64_t addr, uint64_t size)
{
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
    Result rc = sym_add_host_buffer_(emu_, addr, size);
    check(rc, "add_host_buffer");
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

void emu_transport_inprocess::execute(const void* instructions, size_t size)
{
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
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
    last_run_complete_ = (status.halted != 0);

    // Emit a structured status line so the bridge script can classify results.
    // Emit BEFORE the check(result) throw so bridges still see halt_reason on
    // error paths -- otherwise an ExecutionError loses all cycle/reason info.
    const char* reason_str = "error";
    switch (status.halt_reason) {
        case HALT_COMPLETED:            reason_str = "completed";            break;
        case HALT_BUDGET:               reason_str = "budget";               break;
        case HALT_ERROR:                reason_str = "error";                break;
        case HALT_MASKPOLL_UNSATISFIED: reason_str = "maskpoll_unsatisfied"; break;
    }
    std::cerr << "XDNA_EMU_STATUS: halt_reason=" << reason_str
              << " cycles=" << status.cycles_executed
              << " max_cycles=" << max_cycles_budget_ << "\n";

    check(status.result, "execute (run)");
}

void emu_transport_inprocess::execute_from_device(uint64_t dev_addr,
                                                   uint32_t size)
{
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
    // Read the instruction bytes from emulator memory, then execute them
    // through the normal path.  This is the XRT flow: the host wrote
    // instructions into a BO, sync'd it to device, and the ert_packet
    // tells us the device address.
    std::vector<uint8_t> instr(size);
    Result rc = sym_read_host_memory_(emu_, dev_addr, instr.data(),
                                       static_cast<uint64_t>(size));
    check(rc, "execute_from_device (read instructions)");
    execute(instr.data(), size);
}

bool emu_transport_inprocess::poll_completion()
{
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
    return last_run_complete_;
}

// ---------------------------------------------------------------------------
// Debug / AIE tile access
// ---------------------------------------------------------------------------

uint32_t emu_transport_inprocess::read_reg(uint16_t col, uint16_t row,
                                           uint32_t addr)
{
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
    if (!sym_read_register_)
        throw std::runtime_error("read_reg: FFI function not available");
    return sym_read_register_(emu_, col, row, addr);
}

void emu_transport_inprocess::write_reg(uint16_t col, uint16_t row,
                                        uint32_t addr, uint32_t val)
{
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
    if (!sym_write_register_)
        throw std::runtime_error("write_reg: FFI function not available");
    Result rc = sym_write_register_(emu_, col, row, addr, val);
    check(rc, "write_reg");
}

void emu_transport_inprocess::read_tile_memory(uint16_t col, uint16_t row,
                                               uint32_t offset, uint32_t size,
                                               void* out)
{
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
    if (!sym_read_tile_mem_)
        throw std::runtime_error("read_tile_memory: FFI function not available");
    Result rc = sym_read_tile_mem_(emu_, col, row, offset, size, out);
    check(rc, "read_tile_memory");
}

void emu_transport_inprocess::write_tile_memory(uint16_t col, uint16_t row,
                                                uint32_t offset, uint32_t size,
                                                const void* data)
{
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
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
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
    if (sym_get_columns_)
        return sym_get_columns_(emu_);
    return 5;  // Phoenix default
}

uint8_t emu_transport_inprocess::get_rows()
{
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
    if (sym_get_rows_)
        return sym_get_rows_(emu_);
    return 6;  // Phoenix default
}

std::string emu_transport_inprocess::get_device_name()
{
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
    if (sym_get_device_name_) {
        char buf[256];
        int32_t len = sym_get_device_name_(emu_, buf, sizeof(buf));
        if (len > 0)
            return std::string(buf, static_cast<size_t>(len));
    }
    return "NPU Phoenix (Emulated)";  // fallback
}

// ---------------------------------------------------------------------------
// Diagnostic queries
// ---------------------------------------------------------------------------

int8_t emu_transport_inprocess::get_lock_value(uint16_t col, uint16_t row,
                                               uint8_t lock_id)
{
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
    if (sym_get_lock_value_)
        return sym_get_lock_value_(emu_, col, row, lock_id);
    return -128;  // not available
}

uint32_t emu_transport_inprocess::get_dma_channel_state(uint16_t col,
                                                         uint16_t row,
                                                         uint8_t is_s2mm,
                                                         uint8_t channel_index)
{
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
    if (sym_get_dma_ch_state_)
        return sym_get_dma_ch_state_(emu_, col, row, is_s2mm, channel_index);
    return 0;  // Idle
}

bool emu_transport_inprocess::get_dma_channel_stats(uint16_t col,
                                                     uint16_t row,
                                                     uint8_t is_s2mm,
                                                     uint8_t channel_index,
                                                     DmaChannelStats& out)
{
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
    if (!sym_get_dma_ch_stats_)
        return false;

    // The Rust FFI uses the same layout as our DmaChannelStats struct:
    // 4 x uint64_t.  Pass the address directly.
    int32_t rc = sym_get_dma_ch_stats_(emu_, col, row, is_s2mm,
                                        channel_index, &out);
    return rc == 0;
}

bool emu_transport_inprocess::set_log_level(const std::string& level)
{
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
    if (!sym_set_log_level_)
        return false;
    return sym_set_log_level_(level.c_str()) == 0;
}

std::string emu_transport_inprocess::dump_tile_state(uint16_t col,
                                                      uint16_t row)
{
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
    if (!sym_dump_tile_state_)
        return {};

    char buf[4096];
    int32_t len = sym_dump_tile_state_(emu_, col, row, buf, sizeof(buf));
    if (len > 0 && static_cast<uint32_t>(len) < sizeof(buf))
        return std::string(buf, static_cast<size_t>(len));
    return {};
}

} // namespace xdna_emu
