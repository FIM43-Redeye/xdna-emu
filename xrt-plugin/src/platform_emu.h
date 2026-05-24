// SPDX-License-Identifier: MIT
//
// platform_emu.h -- Emulator platform driver.
//
// Overrides the virtual ioctl dispatch methods in shim_xdna::platform_drv,
// routing buffer and execution operations through emu_transport instead
// of through real DRM ioctls.

#pragma once

#include "shim/platform.h"
#include "transport.h"
#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace xdna_emu {

class platform_drv_emu : public shim_xdna::platform_drv
{
public:
  using platform_drv::platform_drv;

  // Override open/close to skip real device-node operations.
  void
  drv_open(const std::string& sysfs_name) const override;

  void
  drv_close() const override;

private:
  // -- Context management ----------------------------------------------------

  void
  create_ctx(shim_xdna::create_ctx_arg& arg) const override;

  void
  destroy_ctx(shim_xdna::destroy_ctx_arg& arg) const override;

  // -- Buffer management -----------------------------------------------------

  void
  create_bo(shim_xdna::bo_info& arg) const override;

  void
  create_uptr_bo(shim_xdna::bo_info& arg) const override;

  void
  destroy_bo(shim_xdna::destroy_bo_arg& arg) const override;

  void
  sync_bo(shim_xdna::sync_bo_arg& arg) const override;

  // -- Execution -------------------------------------------------------------

  void
  submit_cmd(shim_xdna::submit_cmd_arg& arg) const override;

  void
  wait_cmd_ioctl(shim_xdna::wait_cmd_arg& arg) const override;

  // -- Context configuration -------------------------------------------------

  void
  config_ctx_cu_config(shim_xdna::config_ctx_cu_config_arg& arg) const override;

  void
  config_ctx_debug_bo(shim_xdna::config_ctx_debug_bo_arg& arg) const override;

  // -- Buffer management (unsupported operations) ---------------------------

  void
  export_bo(shim_xdna::export_bo_arg& arg) const override;

  void
  import_bo(shim_xdna::import_bo_arg& arg) const override;

  // -- Execution (syncobj path) ---------------------------------------------

  void
  wait_cmd_syncobj(shim_xdna::wait_cmd_arg& arg) const override;

  // -- Info / sysfs / state -------------------------------------------------

  void
  get_info(amdxdna_drm_get_info& arg) const override;

  void
  get_info_array(amdxdna_drm_get_array& arg) const override;

  void
  set_state(amdxdna_drm_set_state& arg) const override;

  void
  get_sysfs(shim_xdna::get_sysfs_arg& arg) const override;

  void
  put_sysfs(shim_xdna::put_sysfs_arg& arg) const override;

  // -- Syncobj stubs (no real DRM fd available) ------------------------------

  void
  create_syncobj(shim_xdna::create_destroy_syncobj_arg& arg) const override;

  void
  destroy_syncobj(shim_xdna::create_destroy_syncobj_arg& arg) const override;

  void
  export_syncobj(shim_xdna::export_import_syncobj_arg& arg) const override;

  void
  import_syncobj(shim_xdna::export_import_syncobj_arg& arg) const override;

  void
  wait_syncobj(shim_xdna::wait_syncobj_arg& arg) const override;

  void
  signal_syncobj(shim_xdna::signal_syncobj_arg& arg) const override;

  // -- Internal state --------------------------------------------------------

  // The transport is owned by pdev_emu; we hold a non-owning pointer
  // set by pdev_emu::on_first_open() after creating the transport.
  mutable emu_transport* m_transport = nullptr;

public:
  void
  set_transport(emu_transport* t) const { m_transport = t; }

  /// Log all tracked BOs (EMU_DBG level).
  void dump_bo_table(const char* context) const;

  /// Log all active contexts (EMU_DBG level).
  void dump_ctx_table(const char* context) const;

private:

  // Monotonic counters for synthetic handles.
  mutable std::atomic<uint32_t> m_next_ctx_handle{1};
  mutable std::atomic<uint32_t> m_next_bo_handle{1};
  mutable std::atomic<uint64_t> m_next_seq{1};
  mutable std::atomic<uint32_t> m_next_syncobj{1};
  mutable std::atomic<uint64_t> m_memfd_size{0};  // Current memfd file size.

  // Map BO handle -> {device_addr, size, host_ptr, user_ptr}.
  struct bo_entry {
    uint64_t dev_addr  = 0;
    size_t   size      = 0;
    void*    host_ptr  = nullptr;
    bool     user_ptr  = false;  // true = caller-owned, do not free
  };
  mutable std::mutex m_bo_lock;
  mutable std::unordered_map<uint32_t, bo_entry> m_bo_map;

  // Active contexts: handle -> {start_col, num_tiles, pid, submissions, completions}.
  // num_tiles = num_cols * compute_rows (the XRT-native unit from arg.num_tiles).
  // Callers must overwrite num_tiles immediately from arg.num_tiles; the default
  // is a placeholder — do not rely on it.
  struct ctx_entry {
    uint32_t start_col  = 0;
    uint32_t num_tiles  = 5;  // placeholder; overwritten in create_ctx from arg.num_tiles
    int64_t  pid       = 0;
    uint64_t submissions = 0;
    uint64_t completions = 0;
    uint16_t num_cus   = 0;
    uint32_t debug_bo  = 0;  // 0 = no debug buffer attached
    // Stored PDI blobs for this hw_context. Captured in
    // config_ctx_cu_config and replayed at submit_cmd time after a
    // fresh reset_context, so each submit starts from a clean array
    // state regardless of what previous submits left behind. This is
    // what makes long trace sweeps (mode-1 and mode-2) stable past
    // the first few batches -- without per-submit reset, lock and
    // DMA-channel state from prior submits drift into the next
    // submit's run and stall on lock acquires.
    std::vector<std::vector<uint8_t>> pdi_blobs;
  };
  mutable std::mutex m_ctx_lock;
  mutable std::unordered_map<uint32_t, ctx_entry> m_ctx_map;

  // Settable state (round-trips through set_state / get_info).
  mutable uint8_t m_power_mode  = POWER_MODE_DEFAULT;
  mutable uint8_t m_preemption  = 0;  // disabled
  mutable uint8_t m_fbp_mode    = 0;  // disabled
};

} // namespace xdna_emu
