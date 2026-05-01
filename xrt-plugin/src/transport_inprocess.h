// SPDX-License-Identifier: MIT
//
// transport_inprocess.h -- In-process emulator transport.
//
// dlopen's libxdna_emu.so and resolves the C FFI symbols defined in
// include/xdna_emu.h.  New FFI symbols that may not exist yet are
// resolved with graceful fallback (nullptr stored, exception thrown if
// the caller actually tries to use the missing function).

#pragma once

#include "transport.h"

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>

// Forward-declare the opaque handle so we don't pull in xdna_emu.h here.
struct XdnaEmuHandle;

namespace xdna_emu {

class emu_transport_inprocess final : public emu_transport {
public:
    /// Construct by loading @lib_path via dlopen.
    /// Throws std::runtime_error on dlopen/dlsym failure.
    explicit emu_transport_inprocess(const std::string& lib_path);

    ~emu_transport_inprocess() override;

    // Non-copyable, non-movable (owns dlopen handle + emulator state).
    emu_transport_inprocess(const emu_transport_inprocess&) = delete;
    emu_transport_inprocess& operator=(const emu_transport_inprocess&) = delete;

    // -- emu_transport interface ---------------------------------------------

    void load_xclbin(const std::string& path,
                     uint8_t uuid_out[16]) override;
    void load_pdi(const void* data, size_t size) override;

    uint64_t alloc_buffer(size_t size) override;
    void     free_buffer(uint64_t addr) override;
    void     reset_context() override;
    void     write_memory(uint64_t addr, const void* data,
                          size_t size) override;
    void     read_memory(uint64_t addr, void* data,
                         size_t size) override;

    void clear_host_buffers() override;
    void add_host_buffer(uint64_t addr, uint64_t size) override;

    void execute(const void* instructions, size_t size) override;
    void execute_from_device(uint64_t dev_addr, uint32_t size) override;
    bool poll_completion() override;

    uint32_t read_reg(uint16_t col, uint16_t row,
                      uint32_t addr) override;
    void     write_reg(uint16_t col, uint16_t row,
                       uint32_t addr, uint32_t val) override;
    void     read_tile_memory(uint16_t col, uint16_t row,
                              uint32_t offset, uint32_t size,
                              void* out) override;
    void     write_tile_memory(uint16_t col, uint16_t row,
                               uint32_t offset, uint32_t size,
                               const void* data) override;

    uint8_t     get_columns() override;
    uint8_t     get_rows() override;
    std::string get_device_name() override;

    // -- Diagnostic overrides ------------------------------------------------
    int8_t      get_lock_value(uint16_t col, uint16_t row,
                               uint8_t lock_id) override;
    uint32_t    get_dma_channel_state(uint16_t col, uint16_t row,
                                      uint8_t is_s2mm,
                                      uint8_t channel_index) override;
    bool        get_dma_channel_stats(uint16_t col, uint16_t row,
                                      uint8_t is_s2mm,
                                      uint8_t channel_index,
                                      DmaChannelStats& out) override;
    bool        set_log_level(const std::string& level) override;
    std::string dump_tile_state(uint16_t col, uint16_t row) override;

private:
    // -----------------------------------------------------------------------
    // FFI function-pointer types.
    //
    // "Existing" functions are guaranteed present in any libxdna_emu.so that
    // matches the current include/xdna_emu.h.  "Future" functions may not
    // exist yet -- dlsym may return nullptr.
    // -----------------------------------------------------------------------

    // -- Result / status types (mirrored from xdna_emu.h) -------------------
    // We only need these as typedefs for the function pointers.
    //
    // LAYOUT MUST MATCH `XdnaEmuExecStatus` in
    // crates/xdna-emu-ffi/src/lib.rs. Field order and sizes here are
    // load-bearing: the Rust cdylib returns this struct by value via
    // the C ABI. Appending new fields is safe; reordering is not.
    using Result = int;          // XdnaEmuResult enum (0 = success)
    enum HaltReason : int {      // Mirrors Rust `XdnaEmuHaltReason`.
        HALT_COMPLETED = 0,
        HALT_BUDGET    = 1,
        HALT_ERROR     = 2,
    };
    struct ExecStatus {
        Result     result;          // 4 bytes, offset 0
        uint64_t   cycles_executed; // 8 bytes, offset 8 (with 4-byte alignment padding at 4)
        uint8_t    halted;          // 1 byte, offset 16 -- matches Rust `bool` layout exactly
        HaltReason halt_reason;     // 4 bytes, offset 20 (compiler adds 3 bytes of pad at 17-19)
    };                              // total: 24 bytes

    // -- Existing FFI -------------------------------------------------------
    using fn_create             = XdnaEmuHandle* (*)();
    using fn_destroy            = void (*)(XdnaEmuHandle*);
    using fn_load_xclbin        = Result (*)(XdnaEmuHandle*, const char*, uint8_t*);
    using fn_load_pdi           = Result (*)(XdnaEmuHandle*, const uint8_t*, uint64_t);
    using fn_alloc_host_region  = Result (*)(XdnaEmuHandle*, const char*,
                                             uint64_t, uint64_t);
    using fn_write_host_memory  = Result (*)(XdnaEmuHandle*, uint64_t,
                                             const uint8_t*, uint64_t);
    using fn_read_host_memory   = Result (*)(XdnaEmuHandle*, uint64_t,
                                             uint8_t*, uint64_t);
    using fn_clear_host_buffers = Result (*)(XdnaEmuHandle*);
    using fn_add_host_buffer    = Result (*)(XdnaEmuHandle*, uint64_t, uint64_t);
    using fn_exec_npu_instr     = Result (*)(XdnaEmuHandle*, const uint8_t*, uint64_t);
    using fn_sync_cores         = Result (*)(XdnaEmuHandle*);
    using fn_set_max_cycles     = Result (*)(XdnaEmuHandle*, uint64_t);
    using fn_run                = ExecStatus (*)(XdnaEmuHandle*);
    using fn_get_error          = uint64_t (*)(char*, uint64_t);
    using fn_version            = uint32_t (*)();

    // -- Future FFI (may be nullptr) ----------------------------------------
    using fn_alloc_buffer       = uint64_t (*)(XdnaEmuHandle*, uint64_t);
    using fn_free_buffer        = Result (*)(XdnaEmuHandle*, uint64_t);
    using fn_reset_context      = Result (*)(XdnaEmuHandle*);
    using fn_read_register      = uint32_t (*)(XdnaEmuHandle*, uint16_t,
                                               uint16_t, uint32_t);
    using fn_write_register     = Result (*)(XdnaEmuHandle*, uint16_t,
                                             uint16_t, uint32_t, uint32_t);
    using fn_read_tile_mem      = Result (*)(XdnaEmuHandle*, uint16_t,
                                             uint16_t, uint32_t, uint32_t,
                                             void*);
    using fn_write_tile_mem     = Result (*)(XdnaEmuHandle*, uint16_t,
                                             uint16_t, uint32_t, uint32_t,
                                             const void*);
    using fn_get_columns        = uint8_t (*)(XdnaEmuHandle*);
    using fn_get_rows           = uint8_t (*)(XdnaEmuHandle*);
    using fn_get_device_name    = int32_t (*)(XdnaEmuHandle*, char*, uint32_t);

    // -- Diagnostic FFI (may be nullptr) -----------------------------------
    using fn_get_lock_value     = int8_t (*)(XdnaEmuHandle*, uint16_t,
                                             uint16_t, uint8_t);
    using fn_get_dma_ch_state   = uint32_t (*)(XdnaEmuHandle*, uint16_t,
                                               uint16_t, uint8_t, uint8_t);
    using fn_get_dma_ch_stats   = int32_t (*)(XdnaEmuHandle*, uint16_t,
                                              uint16_t, uint8_t, uint8_t,
                                              void*);
    using fn_set_log_level      = int32_t (*)(const char*);
    using fn_dump_tile_state    = int32_t (*)(XdnaEmuHandle*, uint16_t,
                                             uint16_t, char*, uint32_t);

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------

    void*           dl_handle_ = nullptr;   // dlopen handle
    XdnaEmuHandle*  emu_       = nullptr;   // emulator instance

    // Serialize every FFI call into the Rust handle. The Rust side has
    // no internal locking on XdnaEmuHandle's mutable state (free_list,
    // next_alloc_addr, engine, npu_executor, host_memory), and the
    // bridge runner's async ctx-builder/destroyer threads can call
    // through this transport concurrently with the main submit thread.
    // Without this lock, alloc_buffer racing destroy_bo's free_buffer
    // produced glibc heap corruption ("double free or corruption (!prev)")
    // and "free_buffer: no region at <addr>" warnings, both of which
    // were intermittent failures roughly 1-in-3 trace sweep runs.
    //
    // Recursive because execute_from_device locks, then calls execute()
    // which locks again -- keeping the lock held across the read+execute
    // pair preserves atomicity vs. a destroyer thread tearing down state
    // between the two calls.
    mutable std::recursive_mutex ffi_lock_;

    // Tracks whether the last execute() has finished.
    bool            last_run_complete_ = true;

    // Counter for generating unique buffer addresses in fallback mode.
    uint64_t        next_alloc_addr_ = 0x100000000ULL;

    // Cycle budget set from XDNA_EMU_MAX_CYCLES (0 = unbounded).
    uint64_t        max_cycles_budget_ = 0;

    // -- Resolved function pointers (existing) ------------------------------
    fn_create             sym_create_             = nullptr;
    fn_destroy            sym_destroy_            = nullptr;
    fn_load_xclbin        sym_load_xclbin_        = nullptr;
    fn_load_pdi           sym_load_pdi_           = nullptr;
    fn_alloc_host_region  sym_alloc_host_region_  = nullptr;
    fn_write_host_memory  sym_write_host_memory_  = nullptr;
    fn_read_host_memory   sym_read_host_memory_   = nullptr;
    fn_clear_host_buffers sym_clear_host_buffers_ = nullptr;
    fn_add_host_buffer    sym_add_host_buffer_    = nullptr;
    fn_exec_npu_instr     sym_exec_npu_instr_     = nullptr;
    fn_sync_cores         sym_sync_cores_         = nullptr;
    fn_set_max_cycles     sym_set_max_cycles_     = nullptr;
    fn_run                sym_run_                 = nullptr;
    fn_get_error          sym_get_error_          = nullptr;
    fn_version            sym_version_            = nullptr;

    // -- Resolved function pointers (future, nullable) ----------------------
    fn_alloc_buffer       sym_alloc_buffer_       = nullptr;
    fn_free_buffer        sym_free_buffer_        = nullptr;
    fn_reset_context      sym_reset_context_      = nullptr;
    fn_read_register      sym_read_register_      = nullptr;
    fn_write_register     sym_write_register_     = nullptr;
    fn_read_tile_mem      sym_read_tile_mem_      = nullptr;
    fn_write_tile_mem     sym_write_tile_mem_     = nullptr;
    fn_get_columns        sym_get_columns_        = nullptr;
    fn_get_rows           sym_get_rows_           = nullptr;
    fn_get_device_name    sym_get_device_name_    = nullptr;

    // -- Diagnostic (nullable) ---------------------------------------------
    fn_get_lock_value     sym_get_lock_value_     = nullptr;
    fn_get_dma_ch_state   sym_get_dma_ch_state_   = nullptr;
    fn_get_dma_ch_stats   sym_get_dma_ch_stats_   = nullptr;
    fn_set_log_level      sym_set_log_level_      = nullptr;
    fn_dump_tile_state    sym_dump_tile_state_     = nullptr;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Resolve a required symbol.  Throws std::runtime_error if missing.
    template <typename T>
    T resolve_required(const char* name);

    /// Resolve an optional symbol.  Returns nullptr if missing.
    template <typename T>
    T resolve_optional(const char* name);

    /// Throw if @rc indicates an error, pulling the message from get_error().
    void check(Result rc, const char* context);

    /// Retrieve the last error string from the emulator library.
    std::string last_error();
};

} // namespace xdna_emu
