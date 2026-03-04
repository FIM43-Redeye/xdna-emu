// SPDX-License-Identifier: MIT
//
// platform_emu.cpp -- Emulator platform driver.
//
// Routes ioctl-like operations through emu_transport instead of
// through the real DRM device.  Implements the complete platform_drv
// virtual interface so that XRT treats the emulator like real hardware.

#include "platform_emu.h"
#include "shim/shim_debug.h"
#include "drm_local/amdxdna_accel.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unistd.h>

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
// Context management -- assign monotonic IDs, track active contexts.
// ---------------------------------------------------------------------------

void
platform_drv_emu::
create_ctx(shim_xdna::create_ctx_arg& arg) const
{
  uint32_t handle = m_next_ctx_handle.fetch_add(1);
  arg.ctx_handle = handle;
  arg.umq_doorbell = AMDXDNA_INVALID_DOORBELL_OFFSET;
  arg.syncobj_handle = AMDXDNA_INVALID_FENCE_HANDLE;

  // Track the context for get_info_array(HW_CONTEXT_ALL).
  {
    const std::lock_guard<std::mutex> lock(m_ctx_lock);
    ctx_entry entry{};
    entry.num_col = arg.num_tiles;
    entry.pid = static_cast<int64_t>(getpid());
    m_ctx_map[handle] = entry;
  }
}

void
platform_drv_emu::
destroy_ctx(shim_xdna::destroy_ctx_arg& arg) const
{
  const std::lock_guard<std::mutex> lock(m_ctx_lock);
  m_ctx_map.erase(arg.ctx_handle);
}

void
platform_drv_emu::
config_ctx_cu_config(shim_xdna::config_ctx_cu_config_arg& /*arg*/) const
{
  // Accept kernel configuration silently.  The real driver would push
  // the configuration buffer to firmware; the emulator receives kernels
  // through the xclbin load path instead.
}

void
platform_drv_emu::
config_ctx_debug_bo(shim_xdna::config_ctx_debug_bo_arg& /*arg*/) const
{
  // Debug buffer attach/detach is a no-op for emulation.
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

void
platform_drv_emu::
export_bo(shim_xdna::export_bo_arg& /*arg*/) const
{
  // BO export requires a real DRM fd for dma-buf sharing.
  shim_not_supported_err("export_bo: not supported in emulation (no DRM fd)");
}

void
platform_drv_emu::
import_bo(shim_xdna::import_bo_arg& /*arg*/) const
{
  shim_not_supported_err("import_bo: not supported in emulation (no DRM fd)");
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

  // Update context stats.
  {
    const std::lock_guard<std::mutex> lock(m_ctx_lock);
    auto it = m_ctx_map.find(arg.ctx_handle);
    if (it != m_ctx_map.end())
      it->second.submissions++;
  }
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

void
platform_drv_emu::
wait_cmd_syncobj(shim_xdna::wait_cmd_arg& arg) const
{
  // Delegate to the ioctl-based wait since we don't have real syncobjs.
  wait_cmd_ioctl(arg);
}

// ---------------------------------------------------------------------------
// get_info -- device queries via DRM_AMDXDNA_GET_INFO.
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

    // Query the emulator for actual array dimensions.
    if (m_transport) {
      md->cols = m_transport->get_columns();
      md->rows = m_transport->get_rows();
    } else {
      md->cols = 5;
      md->rows = 6;
    }
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
    std::memset(pm, 0, sizeof(*pm));
    pm->power_mode = m_power_mode;
    arg.buffer_size = sizeof(amdxdna_drm_get_power_mode);
    break;
  }

  case DRM_AMDXDNA_QUERY_AIE_STATUS: {
    // Tile status query.  The caller provides a nested buffer pointer
    // inside amdxdna_drm_query_aie_status.  We zero-fill it (no tile
    // status data available from emulation).
    if (arg.buffer_size < sizeof(amdxdna_drm_query_aie_status))
      shim_err(EINVAL, "get_info: buffer too small for AIE status");

    auto* st = reinterpret_cast<amdxdna_drm_query_aie_status*>(arg.buffer);
    if (st->buffer && st->buffer_size > 0)
      std::memset(reinterpret_cast<void*>(st->buffer), 0, st->buffer_size);
    st->cols_filled = 0;
    arg.buffer_size = sizeof(amdxdna_drm_query_aie_status);
    break;
  }

  case DRM_AMDXDNA_QUERY_SENSORS: {
    // Sensor data not available in emulation.
    arg.buffer_size = 0;
    break;
  }

  case DRM_AMDXDNA_QUERY_HW_CONTEXTS: {
    // Legacy hardware context query (superseded by get_info_array).
    // Return the list of active contexts as amdxdna_drm_query_hwctx[].
    const std::lock_guard<std::mutex> lock(m_ctx_lock);
    uint32_t needed = static_cast<uint32_t>(
      m_ctx_map.size() * sizeof(amdxdna_drm_query_hwctx));

    if (arg.buffer_size < needed) {
      arg.buffer_size = needed;
      break;
    }

    auto* out = reinterpret_cast<amdxdna_drm_query_hwctx*>(arg.buffer);
    uint32_t idx = 0;
    for (const auto& [handle, ctx] : m_ctx_map) {
      auto& entry = out[idx++];
      std::memset(&entry, 0, sizeof(entry));
      entry.context_id = handle;
      entry.start_col = ctx.start_col;
      entry.num_col = ctx.num_col;
      entry.pid = ctx.pid;
      entry.command_submissions = ctx.submissions;
      entry.command_completions = ctx.completions;
    }
    arg.buffer_size = needed;
    break;
  }

  case DRM_AMDXDNA_QUERY_TELEMETRY: {
    // Telemetry not available in emulation.  Return an empty header.
    if (arg.buffer_size < sizeof(amdxdna_drm_query_telemetry_header))
      shim_err(EINVAL, "get_info: buffer too small for telemetry header");

    auto* hdr = reinterpret_cast<amdxdna_drm_query_telemetry_header*>(arg.buffer);
    std::memset(hdr, 0, sizeof(*hdr));
    arg.buffer_size = sizeof(amdxdna_drm_query_telemetry_header);
    break;
  }

  case DRM_AMDXDNA_GET_FORCE_PREEMPT_STATE: {
    if (arg.buffer_size < sizeof(amdxdna_drm_attribute_state))
      shim_err(EINVAL, "get_info: buffer too small for preemption state");

    auto* st = reinterpret_cast<amdxdna_drm_attribute_state*>(arg.buffer);
    std::memset(st, 0, sizeof(*st));
    st->state = m_preemption;
    arg.buffer_size = sizeof(amdxdna_drm_attribute_state);
    break;
  }

  case DRM_AMDXDNA_QUERY_RESOURCE_INFO: {
    if (arg.buffer_size < sizeof(amdxdna_drm_get_resource_info))
      shim_err(EINVAL, "get_info: buffer too small for resource info");

    auto* res = reinterpret_cast<amdxdna_drm_get_resource_info*>(arg.buffer);
    std::memset(res, 0, sizeof(*res));
    res->npu_clk_max = 1000;   // MHz
    res->npu_tops_max = 0;     // Not applicable for emulation.
    res->npu_task_max = 256;
    res->npu_tops_curr = 0;
    res->npu_task_curr = 0;
    arg.buffer_size = sizeof(amdxdna_drm_get_resource_info);
    break;
  }

  case DRM_AMDXDNA_GET_FRAME_BOUNDARY_PREEMPT_STATE: {
    if (arg.buffer_size < sizeof(amdxdna_drm_attribute_state))
      shim_err(EINVAL, "get_info: buffer too small for FBP state");

    auto* st = reinterpret_cast<amdxdna_drm_attribute_state*>(arg.buffer);
    std::memset(st, 0, sizeof(*st));
    st->state = m_fbp_mode;
    arg.buffer_size = sizeof(amdxdna_drm_attribute_state);
    break;
  }

  default:
    shim_not_supported_err("get_info: unsupported param");
  }
}

// ---------------------------------------------------------------------------
// get_info_array -- array queries via DRM_AMDXDNA_GET_ARRAY.
// ---------------------------------------------------------------------------

void
platform_drv_emu::
get_info_array(amdxdna_drm_get_array& arg) const
{
  switch (arg.param) {
  case DRM_AMDXDNA_HW_CONTEXT_ALL: {
    const std::lock_guard<std::mutex> lock(m_ctx_lock);
    uint32_t count = static_cast<uint32_t>(m_ctx_map.size());

    if (count == 0) {
      arg.num_element = 0;
      break;
    }

    uint32_t avail = std::min(arg.num_element, count);
    auto* out = reinterpret_cast<amdxdna_drm_hwctx_entry*>(arg.buffer);
    uint32_t idx = 0;
    for (const auto& [handle, ctx] : m_ctx_map) {
      if (idx >= avail)
        break;
      auto& entry = out[idx++];
      std::memset(&entry, 0, sizeof(entry));
      entry.context_id = handle;
      entry.hwctx_id = handle;
      entry.start_col = ctx.start_col;
      entry.num_col = ctx.num_col;
      entry.pid = ctx.pid;
      entry.command_submissions = ctx.submissions;
      entry.command_completions = ctx.completions;
      entry.state = AMDXDNA_HWCTX_STATE_IDLE;
    }
    arg.num_element = idx;
    break;
  }

  case DRM_AMDXDNA_HW_LAST_ASYNC_ERR: {
    // No async errors in emulation.
    arg.num_element = 0;
    break;
  }

  default:
    shim_not_supported_err("get_info_array: unsupported param");
  }
}

// ---------------------------------------------------------------------------
// set_state -- device state changes via DRM_AMDXDNA_SET_STATE.
// ---------------------------------------------------------------------------

void
platform_drv_emu::
set_state(amdxdna_drm_set_state& arg) const
{
  switch (arg.param) {
  case DRM_AMDXDNA_SET_POWER_MODE: {
    if (arg.buffer_size < sizeof(amdxdna_drm_set_power_mode))
      shim_err(EINVAL, "set_state: buffer too small for power mode");

    auto* pm = reinterpret_cast<const amdxdna_drm_set_power_mode*>(arg.buffer);
    m_power_mode = pm->power_mode;
    break;
  }

  case DRM_AMDXDNA_SET_FORCE_PREEMPT: {
    if (arg.buffer_size < sizeof(amdxdna_drm_attribute_state))
      shim_err(EINVAL, "set_state: buffer too small for preemption");

    auto* st = reinterpret_cast<const amdxdna_drm_attribute_state*>(arg.buffer);
    m_preemption = st->state;
    break;
  }

  case DRM_AMDXDNA_SET_FRAME_BOUNDARY_PREEMPT: {
    if (arg.buffer_size < sizeof(amdxdna_drm_attribute_state))
      shim_err(EINVAL, "set_state: buffer too small for FBP");

    auto* st = reinterpret_cast<const amdxdna_drm_attribute_state*>(arg.buffer);
    m_fbp_mode = st->state;
    break;
  }

  case DRM_AMDXDNA_WRITE_AIE_MEM:
  case DRM_AMDXDNA_WRITE_AIE_REG: {
    // Direct AIE memory/register writes could be routed through the
    // transport in the future.  For now, accept silently.
    break;
  }

  default:
    shim_not_supported_err("set_state: unsupported param");
  }
}

// ---------------------------------------------------------------------------
// sysfs -- respond to sysfs queries used for device identification.
// ---------------------------------------------------------------------------

void
platform_drv_emu::
get_sysfs(shim_xdna::get_sysfs_arg& arg) const
{
  // The xdna shim registers queries with empty subdev and lowercase
  // entry names.  The sysfs_node field contains just the entry name.
  std::string value;

  if (arg.sysfs_node == "vbnv") {
    // rom_vbnv query -- device name shown in xrt-smi Name column.
    if (m_transport)
      value = m_transport->get_device_name();
    else
      value = "NPU Phoenix (Emulated)";
  } else if (arg.sysfs_node == "device") {
    // PCI device ID -- Phoenix = 0x1502.
    value = "0x1502";
  } else if (arg.sysfs_node == "revision") {
    value = "0x0";
  } else if (arg.sysfs_node == "vendor") {
    value = "0x1022";
  } else if (arg.sysfs_node == "subsystem_device") {
    value = "0x0000";
  } else if (arg.sysfs_node == "subsystem_vendor") {
    value = "0x0000";
  } else if (arg.sysfs_node == "link_width" ||
             arg.sysfs_node == "link_width_max") {
    value = "0";
  } else if (arg.sysfs_node == "link_speed" ||
             arg.sysfs_node == "link_speed_max") {
    value = "0";
  } else {
    // Unknown sysfs node -- return empty.
    arg.real_size = 0;
    return;
  }

  size_t copy_len = std::min(value.size(), arg.data.size());
  std::memcpy(arg.data.data(), value.data(), copy_len);
  arg.real_size = copy_len;
}

void
platform_drv_emu::
put_sysfs(shim_xdna::put_sysfs_arg& /*arg*/) const
{
  // Sysfs writes are not meaningful for emulation.
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
export_syncobj(shim_xdna::export_import_syncobj_arg& /*arg*/) const
{
  shim_not_supported_err("export_syncobj: not supported in emulation (no DRM fd)");
}

void
platform_drv_emu::
import_syncobj(shim_xdna::export_import_syncobj_arg& /*arg*/) const
{
  shim_not_supported_err("import_syncobj: not supported in emulation (no DRM fd)");
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
