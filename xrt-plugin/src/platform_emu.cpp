// SPDX-License-Identifier: MIT
//
// platform_emu.cpp -- Emulator platform driver.
//
// Routes ioctl-like operations through emu_transport instead of
// through the real DRM device.

#include "platform_emu.h"
#include "shim/shim_debug.h"
#include "drm_local/amdxdna_accel.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace xdna_emu {

// ---------------------------------------------------------------------------
// Open / close -- skip the real device-node open.
// ---------------------------------------------------------------------------

void
platform_drv_emu::
drv_open(const std::string& /*sysfs_name*/) const
{
  // No real device node to open.  The base class drv_open() would try
  // to open /dev/accel/accelN which does not exist for emulation.
  //
  // The transport is owned by pdev_emu and will be set separately.
  // We do NOT call the base implementation.
}

void
platform_drv_emu::
drv_close() const
{
  // Nothing to close -- no real fd was opened.
  m_transport = nullptr;
}

// ---------------------------------------------------------------------------
// Context management -- assign monotonic IDs.
// ---------------------------------------------------------------------------

void
platform_drv_emu::
create_ctx(shim_xdna::create_ctx_arg& arg) const
{
  arg.ctx_handle = m_next_ctx_handle.fetch_add(1);
  arg.umq_doorbell = AMDXDNA_INVALID_DOORBELL_OFFSET;
  arg.syncobj_handle = AMDXDNA_INVALID_FENCE_HANDLE;
}

void
platform_drv_emu::
destroy_ctx(shim_xdna::destroy_ctx_arg& /*arg*/) const
{
  // Nothing to tear down -- context tracking is just an ID.
}

// ---------------------------------------------------------------------------
// Buffer management
// ---------------------------------------------------------------------------

void
platform_drv_emu::
create_bo(shim_xdna::bo_info& arg) const
{
  if (!m_transport)
    shim_err(ENODEV, "create_bo: transport not initialized");

  // Ask the emulator to allocate a buffer and return a device address.
  uint64_t dev_addr = m_transport->alloc_buffer(arg.size);

  // Allocate a host-side mapping for the caller.
  void* host_ptr = std::aligned_alloc(4096, (arg.size + 4095) & ~4095UL);
  if (!host_ptr)
    shim_err(ENOMEM, "create_bo: aligned_alloc failed for %zu bytes", arg.size);
  std::memset(host_ptr, 0, arg.size);

  // Assign a synthetic BO handle.
  uint32_t handle = m_next_bo_handle.fetch_add(1);

  // Populate the output fields.
  arg.bo.handle = handle;
  arg.bo.res_id = AMDXDNA_INVALID_BO_HANDLE;
  arg.xdna_addr = dev_addr;
  arg.vaddr = host_ptr;

  // Use AMDXDNA_INVALID_ADDR so that the shim's mmap_drm_bo() skips
  // the mmap path entirely (it checks for this sentinel).  The vaddr
  // is already usable via the host_ptr we allocated.
  arg.map_offset = AMDXDNA_INVALID_ADDR;

  // Track the allocation.
  {
    const std::lock_guard<std::mutex> lock(m_bo_lock);
    m_bo_map[handle] = {dev_addr, arg.size, host_ptr};
  }

  // Store in the base class BO info map for later lookup.
  save_bo_info(handle, arg);
}

void
platform_drv_emu::
destroy_bo(shim_xdna::destroy_bo_arg& arg) const
{
  if (!delete_bo_info(arg.bo.handle))
    return;

  bo_entry entry{};
  {
    const std::lock_guard<std::mutex> lock(m_bo_lock);
    auto it = m_bo_map.find(arg.bo.handle);
    if (it != m_bo_map.end()) {
      entry = it->second;
      m_bo_map.erase(it);
    }
  }

  if (entry.host_ptr)
    std::free(entry.host_ptr);

  if (m_transport && entry.dev_addr)
    m_transport->free_buffer(entry.dev_addr);
}

void
platform_drv_emu::
sync_bo(shim_xdna::sync_bo_arg& arg) const
{
  if (!m_transport)
    shim_err(ENODEV, "sync_bo: transport not initialized");

  // Look up the BO's host and device addresses.
  bo_entry entry{};
  {
    const std::lock_guard<std::mutex> lock(m_bo_lock);
    auto it = m_bo_map.find(arg.bo.handle);
    if (it == m_bo_map.end())
      shim_err(ENOENT, "sync_bo: unknown BO handle %u", arg.bo.handle);
    entry = it->second;
  }

  auto* host = static_cast<uint8_t*>(entry.host_ptr);
  uint64_t dev = entry.dev_addr;

  if (arg.direction == xrt_core::buffer_handle::direction::host2device) {
    m_transport->write_memory(dev + arg.offset,
                              host + arg.offset,
                              arg.size);
  } else {
    m_transport->read_memory(dev + arg.offset,
                             host + arg.offset,
                             arg.size);
  }
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

void
platform_drv_emu::
submit_cmd(shim_xdna::submit_cmd_arg& arg) const
{
  if (!m_transport)
    shim_err(ENODEV, "submit_cmd: transport not initialized");

  // Retrieve the command BO's host pointer to find the instruction buffer.
  bo_entry entry{};
  {
    const std::lock_guard<std::mutex> lock(m_bo_lock);
    auto it = m_bo_map.find(arg.cmd_bo.handle);
    if (it == m_bo_map.end())
      shim_err(ENOENT, "submit_cmd: unknown cmd BO handle %u", arg.cmd_bo.handle);
    entry = it->second;
  }

  // The command BO contains NPU instructions.
  m_transport->execute(entry.host_ptr, entry.size);

  arg.seq = m_next_seq.fetch_add(1);
}

void
platform_drv_emu::
wait_cmd_ioctl(shim_xdna::wait_cmd_arg& arg) const
{
  if (!m_transport)
    shim_err(ENODEV, "wait_cmd_ioctl: transport not initialized");

  // Poll until completion or timeout.
  auto deadline = std::chrono::steady_clock::now();
  if (arg.timeout_ms == 0) {
    // 0 means wait forever -- use a very large timeout.
    deadline += std::chrono::hours(24);
  } else {
    deadline += std::chrono::milliseconds(arg.timeout_ms);
  }

  while (!m_transport->poll_completion()) {
    if (std::chrono::steady_clock::now() >= deadline)
      shim_err(ETIMEDOUT, "wait_cmd_ioctl: timed out after %u ms", arg.timeout_ms);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

// ---------------------------------------------------------------------------
// Info / sysfs -- return canned Phoenix/NPU1 values.
// ---------------------------------------------------------------------------

void
platform_drv_emu::
get_info(amdxdna_drm_get_info& arg) const
{
  switch (arg.param) {
  case DRM_AMDXDNA_QUERY_AIE_METADATA: {
    if (arg.buffer_size < sizeof(amdxdna_drm_query_aie_metadata))
      shim_err(EINVAL, "get_info: buffer too small for AIE metadata");

    auto* md = reinterpret_cast<amdxdna_drm_query_aie_metadata*>(arg.buffer);
    std::memset(md, 0, sizeof(*md));

    // Phoenix / NPU1 topology: 5 columns, 6 rows (including shim row).
    md->cols = 5;
    md->rows = 6;
    md->col_size = 0;  // Not meaningful for emulation.
    md->version.major = 2;
    md->version.minor = 0;

    // Core tiles: rows 2-5 (4 rows), starting at row 2.
    md->core.row_count = 4;
    md->core.row_start = 2;
    md->core.dma_channel_count = 2;
    md->core.lock_count = 16;
    md->core.event_reg_count = 4;

    // Mem tiles: row 1 (1 row).
    md->mem.row_count = 1;
    md->mem.row_start = 1;
    md->mem.dma_channel_count = 2;
    md->mem.lock_count = 64;
    md->mem.event_reg_count = 2;

    // Shim tiles: row 0 (1 row).
    md->shim.row_count = 1;
    md->shim.row_start = 0;
    md->shim.dma_channel_count = 2;
    md->shim.lock_count = 16;
    md->shim.event_reg_count = 4;

    arg.buffer_size = sizeof(amdxdna_drm_query_aie_metadata);
    break;
  }

  case DRM_AMDXDNA_QUERY_AIE_VERSION: {
    if (arg.buffer_size < sizeof(amdxdna_drm_query_aie_version))
      shim_err(EINVAL, "get_info: buffer too small for AIE version");

    auto* ver = reinterpret_cast<amdxdna_drm_query_aie_version*>(arg.buffer);
    ver->major = 2;
    ver->minor = 0;
    arg.buffer_size = sizeof(amdxdna_drm_query_aie_version);
    break;
  }

  case DRM_AMDXDNA_QUERY_FIRMWARE_VERSION: {
    if (arg.buffer_size < sizeof(amdxdna_drm_query_firmware_version))
      shim_err(EINVAL, "get_info: buffer too small for firmware version");

    auto* fw = reinterpret_cast<amdxdna_drm_query_firmware_version*>(arg.buffer);
    fw->major = 0;
    fw->minor = 0;
    fw->patch = 0;
    fw->build = 0;
    arg.buffer_size = sizeof(amdxdna_drm_query_firmware_version);
    break;
  }

  case DRM_AMDXDNA_QUERY_CLOCK_METADATA: {
    if (arg.buffer_size < sizeof(amdxdna_drm_query_clock_metadata))
      shim_err(EINVAL, "get_info: buffer too small for clock metadata");

    auto* clk = reinterpret_cast<amdxdna_drm_query_clock_metadata*>(arg.buffer);
    std::memset(clk, 0, sizeof(*clk));
    std::strncpy(reinterpret_cast<char*>(clk->mp_npu_clock.name), "MP-NPU", 15);
    clk->mp_npu_clock.freq_mhz = 1000;
    std::strncpy(reinterpret_cast<char*>(clk->h_clock.name), "H-CLK", 15);
    clk->h_clock.freq_mhz = 1000;
    arg.buffer_size = sizeof(amdxdna_drm_query_clock_metadata);
    break;
  }

  case DRM_AMDXDNA_GET_POWER_MODE: {
    if (arg.buffer_size < sizeof(amdxdna_drm_get_power_mode))
      shim_err(EINVAL, "get_info: buffer too small for power mode");

    auto* pm = reinterpret_cast<amdxdna_drm_get_power_mode*>(arg.buffer);
    pm->power_mode = POWER_MODE_DEFAULT;
    arg.buffer_size = sizeof(amdxdna_drm_get_power_mode);
    break;
  }

  default:
    shim_not_supported_err("get_info: unsupported param");
  }
}

void
platform_drv_emu::
get_sysfs(shim_xdna::get_sysfs_arg& arg) const
{
  // Return empty data for any sysfs query -- no real sysfs nodes exist.
  arg.real_size = 0;
}

// ---------------------------------------------------------------------------
// Syncobj stubs -- emulation does not use DRM syncobjs.
// ---------------------------------------------------------------------------

void
platform_drv_emu::
create_syncobj(shim_xdna::create_destroy_syncobj_arg& arg) const
{
  arg.handle = m_next_syncobj.fetch_add(1);
}

void
platform_drv_emu::
destroy_syncobj(shim_xdna::create_destroy_syncobj_arg& /*arg*/) const
{
  // Nothing to destroy.
}

void
platform_drv_emu::
wait_syncobj(shim_xdna::wait_syncobj_arg& /*arg*/) const
{
  // In emulation, execution is synchronous -- nothing to wait on.
}

void
platform_drv_emu::
signal_syncobj(shim_xdna::signal_syncobj_arg& /*arg*/) const
{
  // No-op in emulation.
}

} // namespace xdna_emu
