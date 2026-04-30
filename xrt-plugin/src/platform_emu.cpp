// SPDX-License-Identifier: MIT
//
// platform_emu.cpp -- Emulator platform driver.
//
// Routes ioctl-like operations through emu_transport instead of
// through the real DRM device.  Implements the complete platform_drv
// virtual interface so that XRT treats the emulator like real hardware.

#include "platform_emu.h"
#include "emu_debug.h"
#include "emu_ert_decode.h"
#include "shim/shim_debug.h"
#include "drm_local/amdxdna_accel.h"
#include "core/include/ert.h"

#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
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
  // Create an anonymous memory-backed fd via memfd_create.  The base
  // class drv_mmap() calls ::mmap(fd, offset) which works on memfds,
  // so the entire buffer mmap machinery works unchanged.  We grow the
  // memfd with ftruncate() each time a BO needs mmap space.
  int fd = memfd_create("xdna-emu", MFD_CLOEXEC);
  if (fd < 0)
    shim_err(errno, "drv_open: memfd_create failed");

  m_dev_fd = fd;
  EMU_DBG("drv_open: memfd_create -> fd=%d", fd);
}

void
platform_drv_emu::
drv_close() const
{
  EMU_DBG("drv_close: m_dev_fd=%d", m_dev_fd);
  m_transport = nullptr;

  if (m_dev_fd >= 0) {
    ::close(m_dev_fd);
    m_dev_fd = -1;
  }
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
config_ctx_cu_config(shim_xdna::config_ctx_cu_config_arg& arg) const
{
  // The conf_buf contains amdxdna_hwctx_param_config_cu: a count of CUs
  // followed by per-CU entries (cu_bo handle + cu_func).  On real hardware,
  // this pushes PDI data to firmware.  For the emulator, we read the PDI
  // data from each CU's BO and load it -- the PDI contains the CDO stream
  // that configures DMA descriptors, routing, and core ELF programs.

  if (arg.conf_buf.size() < sizeof(amdxdna_hwctx_param_config_cu))
    return;  // Malformed, nothing to do.

  auto* conf = reinterpret_cast<const amdxdna_hwctx_param_config_cu*>(
    arg.conf_buf.data());

  shim_debug("config_ctx_cu_config: ctx %u, %u CUs", arg.ctx_handle, conf->num_cus);

  // Track CU config per context for diagnostics.
  {
    const std::lock_guard<std::mutex> lock(m_ctx_lock);
    auto it = m_ctx_map.find(arg.ctx_handle);
    if (it != m_ctx_map.end()) {
      it->second.num_cus = conf->num_cus;
    }
  }

  // Load each CU's PDI into the emulator.
  for (uint16_t i = 0; i < conf->num_cus; i++) {
    auto& cu = conf->cu_configs[i];
    shim_debug("  CU[%u]: bo=%u func=%u", i, cu.cu_bo, cu.cu_func);

    if (!m_transport)
      continue;

    // Look up the BO's device address and size.
    bo_entry entry{};
    {
      const std::lock_guard<std::mutex> lock(m_bo_lock);
      auto it = m_bo_map.find(cu.cu_bo);
      if (it == m_bo_map.end())
        continue;
      entry = it->second;
    }

    if (entry.size == 0)
      continue;

    // Read PDI data from the memfd (not emulator memory).
    //
    // XRT's hwctx_kmq writes PDI bytes directly to the mmap'd BO
    // address, then calls buffer::sync(HOST2DEVICE).  On cache-coherent
    // devices (like NPU), sync is a no-op (just clflush or skip), so
    // our platform_drv_emu::sync_bo is never called and the data never
    // reaches the emulator's internal host memory.
    //
    // The data IS in the memfd because XRT wrote it via mmap.  Read it
    // with pread, then also copy it into the emulator so that later
    // accesses (e.g., from the NPU executor) can find it.
    std::vector<uint8_t> pdi_data(entry.size);
    shim_xdna::bo_info bo_info{};
    if (!load_bo_info(cu.cu_bo, bo_info)) {
      EMU_WARN("config_ctx_cu_config: no bo_info for BO %u", cu.cu_bo);
      continue;
    }

    ssize_t nr = pread(m_dev_fd, pdi_data.data(), entry.size,
                       static_cast<off_t>(bo_info.map_offset));
    if (nr <= 0) {
      EMU_WARN("config_ctx_cu_config: pread BO %u failed (nr=%zd)", cu.cu_bo, nr);
      continue;
    }

    // Log entry with hex dump for bridge debugging.
    {
      std::string hex;
      size_t dump_len = std::min(pdi_data.size(), static_cast<size_t>(16));
      for (size_t j = 0; j < dump_len; j++) {
        char buf[4];
        snprintf(buf, sizeof(buf), "%02x ", pdi_data[j]);
        hex += buf;
      }
      EMU_INFO("config_ctx_cu_config: PDI BO %u: %zu bytes (pread=%zd), "
               "first %zu: %s",
               cu.cu_bo, pdi_data.size(), nr, dump_len, hex.c_str());
    }

    m_transport->load_pdi(pdi_data.data(), pdi_data.size());
  }
}

void
platform_drv_emu::
config_ctx_debug_bo(shim_xdna::config_ctx_debug_bo_arg& arg) const
{
  // Track debug buffer association per context.  The emulator can use
  // this for diagnostic data output in the future.
  const std::lock_guard<std::mutex> lock(m_ctx_lock);
  auto it = m_ctx_map.find(arg.ctx_handle);
  if (it == m_ctx_map.end())
    return;

  if (arg.is_detach) {
    shim_debug("config_ctx_debug_bo: detach bo %u from ctx %u",
               arg.bo.handle, arg.ctx_handle);
    it->second.debug_bo = 0;
  } else {
    shim_debug("config_ctx_debug_bo: attach bo %u to ctx %u",
               arg.bo.handle, arg.ctx_handle);
    it->second.debug_bo = arg.bo.handle;
  }
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

  // Grow the memfd to accommodate this BO and record its mmap offset.
  size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  size_t aligned_size = (arg.size + page_size - 1) & ~(page_size - 1);
  uint64_t offset = m_memfd_size.fetch_add(aligned_size);
  EMU_DBG("create_bo: size=%zu dev_addr=0x%" PRIx64 " map_offset=0x%" PRIx64,
          arg.size, dev_addr, offset);

  if (ftruncate(m_dev_fd, static_cast<off_t>(offset + aligned_size)) < 0)
    shim_err(errno, "create_bo: ftruncate memfd to %zu failed", offset + aligned_size);

  // Assign a synthetic BO handle.
  uint32_t handle = m_next_bo_handle.fetch_add(1);

  // Populate the output fields.
  arg.bo.handle = handle;
  arg.bo.res_id = AMDXDNA_INVALID_BO_HANDLE;
  arg.xdna_addr = dev_addr;
  arg.vaddr = nullptr;       // Will be set by shim's mmap path.
  arg.map_offset = offset;   // Valid offset into our memfd.

  // Track the allocation.
  {
    const std::lock_guard<std::mutex> lock(m_bo_lock);
    m_bo_map[handle] = {dev_addr, arg.size, nullptr, /*user_ptr=*/false};
  }

  // Store in the base class BO info map for later lookup.
  save_bo_info(handle, arg);

}

void
platform_drv_emu::
create_uptr_bo(shim_xdna::bo_info& arg) const
{
  if (!m_transport)
    shim_err(ENODEV, "create_uptr_bo: transport not initialized");

  // User-pointer BOs wrap an existing host allocation.  The caller
  // already set arg.vaddr to their buffer.  We allocate a device
  // address but the shim handles the vaddr via its uptr path.
  uint64_t dev_addr = m_transport->alloc_buffer(arg.size);

  uint32_t handle = m_next_bo_handle.fetch_add(1);
  arg.bo.handle = handle;
  arg.bo.res_id = AMDXDNA_INVALID_BO_HANDLE;
  arg.xdna_addr = dev_addr;
  // uptr BOs skip mmap_drm_bo entirely (buffer::vaddr returns m_uptr).
  arg.map_offset = AMDXDNA_INVALID_ADDR;

  {
    const std::lock_guard<std::mutex> lock(m_bo_lock);
    m_bo_map[handle] = {dev_addr, arg.size, arg.vaddr, /*user_ptr=*/true};
  }

  save_bo_info(handle, arg);
}

void
platform_drv_emu::
destroy_bo(shim_xdna::destroy_bo_arg& arg) const
{
  // Drain the base class's refcount entry, but do NOT use its return
  // value to gate the actual free. The base class's delete_bo_info uses
  // a refcount that load_bo_info bumps on every lookup -- and the EMU
  // plugin calls load_bo_info during sync_bo, submit_cmd, and ctx
  // configuration without a matching decrement (those are pure reads in
  // our model). The result is that the base count never reaches 1, so
  // delete_bo_info's "true (erase) on refcount==1" path never fires,
  // and the legacy gate `if (!delete_bo_info(...)) return;` would skip
  // the free forever -- the bug that left run 2+ trace BOs allocated
  // at fresh addresses while the shim DMA channel still pointed at the
  // run 1 address.
  //
  // Treat m_bo_map as the authoritative ownership record instead. xrt::bo
  // destruction is the unambiguous "no more user-side refs" signal that
  // triggered this destroy_bo call, so the first destroy for a handle
  // should always free; subsequent destroys (e.g. from sub-handles XRT
  // duplicates internally) find no entry in m_bo_map and become no-ops.
  delete_bo_info(arg.bo.handle); // drain base-class entry; ignore result

  bo_entry entry{};
  bool present = false;
  {
    const std::lock_guard<std::mutex> lock(m_bo_lock);
    auto it = m_bo_map.find(arg.bo.handle);
    if (it != m_bo_map.end()) {
      entry = it->second;
      m_bo_map.erase(it);
      present = true;
    }
  }
  if (!present)
    return; // already freed by an earlier destroy_bo for this handle.

  // Regular BOs are backed by memfd (freed when fd is closed).
  // User-pointer BOs have caller-owned memory -- never freed by us.
  // Only the emulator-side device address needs explicit cleanup.
  if (m_transport && entry.dev_addr)
    m_transport->free_buffer(entry.dev_addr);
}

void
platform_drv_emu::
sync_bo(shim_xdna::sync_bo_arg& arg) const
{
  if (!m_transport)
    shim_err(ENODEV, "sync_bo: transport not initialized");

  // Look up the BO's device address and saved bo_info (for mmap offset).
  bo_entry entry{};
  {
    const std::lock_guard<std::mutex> lock(m_bo_lock);
    auto it = m_bo_map.find(arg.bo.handle);
    if (it == m_bo_map.end())
      shim_err(ENOENT, "sync_bo: unknown BO handle %u", arg.bo.handle);
    entry = it->second;
  }

  uint64_t dev = entry.dev_addr;

  if (entry.user_ptr) {
    // User-pointer BOs have a direct host pointer.
    auto* host = static_cast<uint8_t*>(entry.host_ptr);
    if (arg.direction == xrt_core::buffer_handle::direction::host2device)
      m_transport->write_memory(dev + arg.offset, host + arg.offset, arg.size);
    else
      m_transport->read_memory(dev + arg.offset, host + arg.offset, arg.size);
    return;
  }

  // Regular BOs are backed by the memfd.  Use the base-class bo_info
  // to find the mmap offset, then pread/pwrite the memfd directly.
  shim_xdna::bo_info info{};
  if (!load_bo_info(arg.bo.handle, info))
    shim_err(ENOENT, "sync_bo: no bo_info for handle %u", arg.bo.handle);

  std::vector<uint8_t> buf(arg.size);

  if (arg.direction == xrt_core::buffer_handle::direction::host2device) {
    // Read from memfd (host) -> write to emulator (device).
    ssize_t n = pread(m_dev_fd, buf.data(), arg.size,
                      static_cast<off_t>(info.map_offset + arg.offset));
    if (n < 0)
      shim_err(errno, "sync_bo: pread memfd failed");
    EMU_DBG("sync_bo h2d: bo=%u dev=0x%" PRIx64 " offset=0x%" PRIx64
            " pread=%zd first=0x%08x",
            arg.bo.handle, dev, info.map_offset + arg.offset, n,
            arg.size >= 4 ? *reinterpret_cast<uint32_t*>(buf.data()) : 0);
    m_transport->write_memory(dev + arg.offset, buf.data(), arg.size);
  } else {
    // Read from emulator (device) -> write to memfd (host).
    m_transport->read_memory(dev + arg.offset, buf.data(), arg.size);
    ssize_t n = pwrite(m_dev_fd, buf.data(), arg.size,
                       static_cast<off_t>(info.map_offset + arg.offset));
    if (n < 0)
      shim_err(errno, "sync_bo: pwrite memfd failed");

    // Log first words of d2h readback for quick zero-check.
    if (arg.size >= 4) {
      auto* words = reinterpret_cast<const uint32_t*>(buf.data());
      uint32_t nwords = std::min(static_cast<uint32_t>(arg.size / 4), 4u);
      if (nwords >= 4) {
        EMU_DBG("sync_bo d2h: bo=%u dev=0x%" PRIx64 " first4=[0x%08x 0x%08x 0x%08x 0x%08x]",
                arg.bo.handle, dev, words[0], words[1], words[2], words[3]);
      } else if (nwords >= 1) {
        EMU_DBG("sync_bo d2h: bo=%u dev=0x%" PRIx64 " first=0x%08x",
                arg.bo.handle, dev, words[0]);
      }
    }
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

  // Look up the command BO's mmap offset in the memfd so we can read
  // the instruction payload and write back the completion state.
  shim_xdna::bo_info info{};
  if (!load_bo_info(arg.cmd_bo.handle, info))
    shim_err(ENOENT, "submit_cmd: no bo_info for cmd BO handle %u", arg.cmd_bo.handle);

  bo_entry entry{};
  {
    const std::lock_guard<std::mutex> lock(m_bo_lock);
    auto it = m_bo_map.find(arg.cmd_bo.handle);
    if (it == m_bo_map.end())
      shim_err(ENOENT, "submit_cmd: unknown cmd BO handle %u", arg.cmd_bo.handle);
    entry = it->second;
  }

  // Read the command BO (ert_packet) from the memfd.
  std::vector<uint8_t> buf(entry.size);
  ssize_t n = pread(m_dev_fd, buf.data(), entry.size,
                    static_cast<off_t>(info.map_offset));
  if (n < 0)
    shim_err(errno, "submit_cmd: pread memfd failed");

  auto* pkt = reinterpret_cast<ert_start_kernel_cmd*>(buf.data());
  xdna_emu::detail::emu_log_ert_packet(pkt, "submit_cmd");
  dump_bo_table("submit_cmd");

  // -- Collect sub-commands --------------------------------------------------
  //
  // Build a list of (instr_addr, instr_size) pairs to execute.  For
  // single commands (ERT_START_NPU, ERT_START_CU) there is one entry.
  // For ERT_CMD_CHAIN, each chained BO contributes one entry.

  struct sub_cmd {
    uint64_t instr_addr;
    uint32_t instr_size;
    // Sub-command packet data (for ERT_START_CU host buffer registration).
    // Null for ERT_START_NPU.
    const ert_start_kernel_cmd* pkt;
  };
  std::vector<sub_cmd> sub_cmds;

  // Buffers to hold sub-command BO data read from memfd (keep alive
  // until after execution so that pkt pointers remain valid).
  std::vector<std::vector<uint8_t>> sub_cmd_bufs;

  // Sub-command BO memfd offsets for marking completion afterward.
  struct sub_cmd_completion {
    uint64_t map_offset;
    size_t   buf_size;
  };
  std::vector<sub_cmd_completion> sub_cmd_completions;

  // Helper: extract instr_addr/instr_size from a single ERT packet.
  // Handles all instruction-bearing opcodes used by the XDNA driver:
  //   ERT_START_NPU (20)             -- standard NPU submission
  //   ERT_START_DPU (18)             -- DPU with dtrace + chaining
  //   ERT_START_NPU_PREEMPT (21)     -- NPU with save/restore
  //   ERT_START_NPU_PREEMPT_ELF (22) -- same, ELF-based
  //   ERT_START_CU (0)               -- legacy kernel launch
  auto parse_single_cmd = [](const ert_start_kernel_cmd* p,
                             uint64_t& addr, uint32_t& size) -> bool {
    auto* mp = const_cast<ert_start_kernel_cmd*>(p);

    if (auto* npu = get_ert_npu_data(mp)) {
      addr = npu->instruction_buffer;
      size = npu->instruction_buffer_size;
      return true;
    }
    if (auto* dpu = get_ert_dpu_data(mp)) {
      addr = dpu->instruction_buffer;
      size = dpu->instruction_buffer_size;
      return true;
    }
    if (auto* pre = get_ert_npu_preempt_data(mp)) {
      // Preemption: we execute the main instruction buffer.
      // Save/restore buffers are for context switching -- not
      // relevant for emulation (single-context, no preemption).
      addr = pre->instruction_buffer;
      size = pre->instruction_buffer_size;
      return true;
    }
    if (auto* elf = get_ert_npu_elf_data(mp)) {
      addr = elf->instruction_buffer;
      size = elf->instruction_buffer_size;
      return true;
    }
    if (p->opcode == ERT_START_CU) {
      auto* regmap = reinterpret_cast<const uint8_t*>(
          p->data + p->extra_cu_masks);
      uint64_t a = 0;
      uint32_t n = 0;
      std::memcpy(&a, regmap + 0x08, sizeof(a));
      std::memcpy(&n, regmap + 0x10, sizeof(n));
      addr = a;
      size = n * sizeof(uint32_t);
      return true;
    }
    return false;
  };

  if (pkt->opcode == ERT_CMD_CHAIN) {
    // ERT_CMD_CHAIN: the payload is ert_cmd_chain_data with an array of
    // sub-command BO handles.  Each BO contains its own ERT packet.
    auto* chain = get_ert_cmd_chain_data(
        reinterpret_cast<ert_packet*>(buf.data()));
    if (!chain)
      shim_err(EINVAL, "submit_cmd: malformed ERT_CMD_CHAIN packet");

    EMU_INFO("submit_cmd: ERT_CMD_CHAIN with %u sub-command(s)",
             chain->command_count);

    sub_cmd_bufs.resize(chain->command_count);

    for (uint32_t i = 0; i < chain->command_count; i++) {
      // data[i] is the BO handle (kmhdl) of the sub-command BO.
      auto sub_handle = static_cast<uint32_t>(chain->data[i]);

      // Look up the sub-command BO.
      bo_entry sub_entry{};
      {
        const std::lock_guard<std::mutex> lock(m_bo_lock);
        auto it = m_bo_map.find(sub_handle);
        if (it == m_bo_map.end()) {
          EMU_WARN("submit_cmd: chain[%u] unknown BO handle %u, skipping",
                   i, sub_handle);
          continue;
        }
        sub_entry = it->second;
      }

      shim_xdna::bo_info sub_bi{};
      if (!load_bo_info(sub_handle, sub_bi)) {
        EMU_WARN("submit_cmd: chain[%u] no bo_info for handle %u, skipping",
                 i, sub_handle);
        continue;
      }

      // Read the sub-command BO from the memfd.
      sub_cmd_bufs[i].resize(sub_entry.size);
      ssize_t nr = pread(m_dev_fd, sub_cmd_bufs[i].data(),
                         sub_entry.size,
                         static_cast<off_t>(sub_bi.map_offset));
      if (nr <= 0) {
        EMU_WARN("submit_cmd: chain[%u] pread failed for handle %u",
                 i, sub_handle);
        continue;
      }

      auto* sub_pkt = reinterpret_cast<ert_start_kernel_cmd*>(
          sub_cmd_bufs[i].data());
      xdna_emu::detail::emu_log_ert_packet(sub_pkt,
          ("submit_cmd chain[" + std::to_string(i) + "]").c_str());

      uint64_t addr = 0;
      uint32_t size = 0;
      if (!parse_single_cmd(sub_pkt, addr, size)) {
        EMU_WARN("submit_cmd: chain[%u] unsupported sub-opcode %u",
                 i, sub_pkt->opcode);
        continue;
      }

      sub_cmds.push_back({addr, size, sub_pkt});
      sub_cmd_completions.push_back(
          {sub_bi.map_offset, sub_cmd_bufs[i].size()});

      EMU_INFO("submit_cmd: chain[%u] instr_addr=0x%" PRIx64
               " instr_size=%u opcode=%u",
               i, addr, size, (unsigned)sub_pkt->opcode);

      // Expand inline DPU chaining within the chain sub-command.
      if (sub_pkt->opcode == ERT_START_DPU) {
        auto* dpu = get_ert_dpu_data(
            const_cast<ert_start_kernel_cmd*>(sub_pkt));
        while (dpu && dpu->chained > 0) {
          dpu = get_ert_dpu_data_next(dpu);
          if (dpu) {
            EMU_INFO("submit_cmd: chain[%u] DPU chained instr_addr=0x%"
                     PRIx64 " instr_size=%u",
                     i, dpu->instruction_buffer,
                     dpu->instruction_buffer_size);
            sub_cmds.push_back({dpu->instruction_buffer,
                                dpu->instruction_buffer_size, sub_pkt});
          }
        }
      }
    }
  } else {
    // Single command: ERT_START_NPU, ERT_START_DPU, ERT_START_CU, etc.
    uint64_t addr = 0;
    uint32_t size = 0;
    if (!parse_single_cmd(pkt, addr, size)) {
      uint32_t op = pkt->opcode;
      shim_err(EINVAL, "submit_cmd: unsupported opcode %u (err=%d): %s",
               op, EINVAL, "Invalid argument");
    }

    EMU_INFO("submit_cmd: instr_addr=0x%" PRIx64 " instr_size=%u opcode=%u",
             addr, size, (unsigned)pkt->opcode);
    sub_cmds.push_back({addr, size, pkt});

    // ERT_START_DPU supports inline chaining: when chained > 0, there
    // are additional ert_dpu_data elements concatenated after the first.
    // Each has its own instruction buffer that must be executed.
    if (pkt->opcode == ERT_START_DPU) {
      auto* dpu = get_ert_dpu_data(
          const_cast<ert_start_kernel_cmd*>(pkt));
      while (dpu && dpu->chained > 0) {
        dpu = get_ert_dpu_data_next(dpu);
        if (dpu) {
          EMU_INFO("submit_cmd: DPU chained instr_addr=0x%" PRIx64
                   " instr_size=%u chained=%u",
                   dpu->instruction_buffer,
                   dpu->instruction_buffer_size,
                   dpu->chained);
          sub_cmds.push_back({dpu->instruction_buffer,
                              dpu->instruction_buffer_size, pkt});
        }
      }
    }
  }

  // -- Sync all BOs from memfd into emulator host memory ---------------------
  //
  // On real hardware the CPU and NPU share physical memory, so
  // buffer::sync() just flushes CPU caches (clflush_data).  In the
  // emulator the memory spaces are separate: the app writes via mmap
  // on the memfd, but the emulator has its own internal host memory.
  // We bridge the gap by copying every BO's content from the memfd
  // into the emulator before execution.
  {
    const std::lock_guard<std::mutex> lock(m_bo_lock);
    for (const auto& [handle, bo] : m_bo_map) {
      if (bo.user_ptr || bo.size == 0)
        continue;

      shim_xdna::bo_info bi{};
      if (!load_bo_info(handle, bi))
        continue;

      std::vector<uint8_t> tmp(bo.size);
      ssize_t nr = pread(m_dev_fd, tmp.data(), bo.size,
                         static_cast<off_t>(bi.map_offset));
      if (nr > 0)
        m_transport->write_memory(bo.dev_addr, tmp.data(),
                                  static_cast<size_t>(nr));
    }
    EMU_DBG("submit_cmd: synced %zu BOs from memfd to emulator",
            m_bo_map.size());
  }

  // -- Execute each sub-command ----------------------------------------------

  for (size_t cmd_idx = 0; cmd_idx < sub_cmds.size(); cmd_idx++) {
    const auto& sc = sub_cmds[cmd_idx];

    // Register host buffers for DdrPatch address patching.
    //
    // The NPU instruction stream contains DdrPatch instructions that
    // reference host buffers by index (arg_idx).  The emulator's NPU
    // executor looks up host_buffers[arg_idx] to get the device address
    // to write into DMA BD registers.
    //
    // The regmap follows the opcode-specific header in every ERT packet
    // type.  BO addresses are packed as 8-byte entries.  For ERT_START_CU
    // they start at offset 0x14 from regmap; for other opcodes the regmap
    // IS the BO argument list (offset 0x00).
    m_transport->clear_host_buffers();

    if (sc.pkt) {
      auto* mp = const_cast<ert_start_kernel_cmd*>(sc.pkt);
      auto* regmap_begin = get_ert_regmap_begin(mp);
      auto* regmap_end   = get_ert_regmap_end(mp);
      auto  regmap_bytes = static_cast<uint32_t>(
          (regmap_end - regmap_begin) * sizeof(uint32_t));

      // For ERT_START_CU, the regmap starts with opcode(4B), instr(8B),
      // ninstr(4B), then BO addresses at offset 0x14.  For all other
      // opcodes (START_NPU, START_DPU, etc.) the regmap IS the BO list
      // starting at offset 0x00.
      uint32_t bo_start = (sc.pkt->opcode == ERT_START_CU) ? 0x14 : 0x00;
      auto* regmap = reinterpret_cast<const uint8_t*>(regmap_begin);

      uint32_t arg_idx = 0;
      uint32_t valid_bo_count = 0;
      for (uint32_t off = bo_start; off + 8 <= regmap_bytes; off += 8) {
        uint64_t bo_addr = 0;
        std::memcpy(&bo_addr, regmap + off, sizeof(bo_addr));
        if (bo_addr == 0) {
          arg_idx++;
          continue;
        }
        uint64_t bo_size = 0;
        {
          const std::lock_guard<std::mutex> lock(m_bo_lock);
          for (const auto& [h, bo] : m_bo_map) {
            if (bo.dev_addr == bo_addr) {
              bo_size = bo.size;
              break;
            }
          }
        }
        if (bo_size == 0) {
          // Address not found in BO table -- likely not a real BO address
          // (e.g., ELF module path puts opcode in regmap).  Skip it.
          arg_idx++;
          continue;
        }
        m_transport->add_host_buffer(bo_addr, bo_size);
        EMU_DBG("submit_cmd[%zu]: registered host buffer arg_idx=%u "
                "addr=0x%" PRIx64 " size=%" PRIu64,
                cmd_idx, arg_idx, bo_addr, bo_size);
        arg_idx++;
        valid_bo_count++;
      }

      // ELF module path: when using xrt::ext::kernel with an ELF
      // module, XRT patches BO addresses directly into the control code
      // buffers (block-write payloads in .ctrltext) via ELF relocations
      // at the host, adding the DDR AIE address offset (0x80000000).
      // The firmware does NOT receive a host buffer table -- it just
      // executes the pre-patched instruction stream.
      //
      // We leave the host buffer table empty so the emulator's DdrPatch
      // handler knows to derive correct addresses from the XRT-patched
      // register values (subtracting the DDR offset) rather than looking
      // up arg_idx in a (wrong) fallback table.
      if (valid_bo_count == 0) {
        EMU_INFO("submit_cmd[%zu]: no valid BOs in regmap -- "
                 "ELF module path (addresses pre-patched by XRT)",
                 cmd_idx);
      }
    }

    // Execute the NPU instruction buffer from emulator host memory.
    m_transport->execute_from_device(sc.instr_addr, sc.instr_size);
  }

  // Post-execution diagnostics: dump DMA state for any non-idle channels.
  if (xdna_emu::detail::emu_log_level() >= 4) {
    uint8_t cols = m_transport->get_columns();
    uint8_t rows = m_transport->get_rows();
    for (uint8_t c = 0; c < cols; c++) {
      for (uint8_t r = 0; r < rows; r++) {
        for (uint8_t dir = 0; dir < 2; dir++) {
          for (uint8_t ch = 0; ch < 2; ch++) {
            auto state = m_transport->get_dma_channel_state(c, r, dir, ch);
            if (state != 0) {
              emu_transport::DmaChannelStats stats{};
              m_transport->get_dma_channel_stats(c, r, dir, ch, stats);
              EMU_DBG("  tile(%u,%u) %s ch%u: state=%u xfr=%" PRIu64
                      " bytes=%" PRIu64,
                      c, r, dir ? "s2mm" : "mm2s", ch,
                      state & 0xFF,
                      stats.transfers_completed,
                      stats.bytes_transferred);
            }
          }
        }
      }
    }
  }

  // -- Sync results back: emulator host memory -> memfd ----------------------
  //
  // The app will read results via mmap on the memfd, but the emulator
  // wrote them to its internal host memory.  Copy everything back so
  // the mmap'd view reflects what the emulator produced.
  {
    const std::lock_guard<std::mutex> lock(m_bo_lock);
    for (const auto& [handle, bo] : m_bo_map) {
      if (bo.user_ptr || bo.size == 0)
        continue;

      shim_xdna::bo_info bi{};
      if (!load_bo_info(handle, bi))
        continue;

      std::vector<uint8_t> tmp(bo.size);
      m_transport->read_memory(bo.dev_addr, tmp.data(), bo.size);
      pwrite(m_dev_fd, tmp.data(), bo.size,
             static_cast<off_t>(bi.map_offset));
    }
    EMU_DBG("submit_cmd: synced %zu BOs from emulator back to memfd",
            m_bo_map.size());
  }

  // Mark the outer packet as completed so that poll_command() sees it.
  auto* hdr = reinterpret_cast<ert_packet*>(buf.data());
  hdr->state = ERT_CMD_STATE_COMPLETED;
  pwrite(m_dev_fd, buf.data(), sizeof(ert_packet),
         static_cast<off_t>(info.map_offset));

  // For chains, mark each sub-command packet as completed too.
  for (size_t i = 0; i < sub_cmd_completions.size(); i++) {
    auto& sc = sub_cmd_completions[i];
    auto* sub_hdr = reinterpret_cast<ert_packet*>(sub_cmd_bufs[i].data());
    sub_hdr->state = ERT_CMD_STATE_COMPLETED;
    pwrite(m_dev_fd, sub_cmd_bufs[i].data(), sizeof(ert_packet),
           static_cast<off_t>(sc.map_offset));
  }

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

  case DRM_AMDXDNA_READ_AIE_MEM: {
    if (arg.buffer_size < sizeof(amdxdna_drm_aie_mem))
      shim_err(EINVAL, "get_info: buffer too small for AIE mem read");

    auto* mem = reinterpret_cast<amdxdna_drm_aie_mem*>(arg.buffer);
    if (!m_transport)
      shim_err(ENODEV, "get_info: transport not initialized for AIE mem read");

    m_transport->read_tile_memory(
      static_cast<uint16_t>(mem->col),
      static_cast<uint16_t>(mem->row),
      mem->addr, mem->size,
      reinterpret_cast<void*>(mem->buf_p));
    arg.buffer_size = sizeof(amdxdna_drm_aie_mem);
    break;
  }

  case DRM_AMDXDNA_READ_AIE_REG: {
    if (arg.buffer_size < sizeof(amdxdna_drm_aie_reg))
      shim_err(EINVAL, "get_info: buffer too small for AIE reg read");

    auto* reg = reinterpret_cast<amdxdna_drm_aie_reg*>(arg.buffer);
    if (!m_transport)
      shim_err(ENODEV, "get_info: transport not initialized for AIE reg read");

    reg->val = m_transport->read_reg(
      static_cast<uint16_t>(reg->col),
      static_cast<uint16_t>(reg->row),
      reg->addr);
    arg.buffer_size = sizeof(amdxdna_drm_aie_reg);
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

  case DRM_AMDXDNA_WRITE_AIE_MEM: {
    if (arg.buffer_size < sizeof(amdxdna_drm_aie_mem))
      shim_err(EINVAL, "set_state: buffer too small for AIE mem write");

    auto* mem = reinterpret_cast<const amdxdna_drm_aie_mem*>(arg.buffer);
    if (!m_transport)
      shim_err(ENODEV, "set_state: transport not initialized for AIE mem write");

    m_transport->write_tile_memory(
      static_cast<uint16_t>(mem->col),
      static_cast<uint16_t>(mem->row),
      mem->addr, mem->size,
      reinterpret_cast<const void*>(mem->buf_p));
    break;
  }

  case DRM_AMDXDNA_WRITE_AIE_REG: {
    if (arg.buffer_size < sizeof(amdxdna_drm_aie_reg))
      shim_err(EINVAL, "set_state: buffer too small for AIE reg write");

    auto* reg = reinterpret_cast<const amdxdna_drm_aie_reg*>(arg.buffer);
    if (!m_transport)
      shim_err(ENODEV, "set_state: transport not initialized for AIE reg write");

    m_transport->write_reg(
      static_cast<uint16_t>(reg->col),
      static_cast<uint16_t>(reg->row),
      reg->addr, reg->val);
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

// ---------------------------------------------------------------------------
// Diagnostic dumps
// ---------------------------------------------------------------------------

void
platform_drv_emu::
dump_bo_table(const char* context) const
{
  const std::lock_guard<std::mutex> lock(m_bo_lock);
  EMU_DBG("%s: %zu BOs tracked", context, m_bo_map.size());
  for (const auto& [handle, bo] : m_bo_map) {
    EMU_DBG("  BO %u: dev=0x%" PRIx64 " size=%zu uptr=%d",
            handle, bo.dev_addr, bo.size, bo.user_ptr ? 1 : 0);
  }
}

void
platform_drv_emu::
dump_ctx_table(const char* context) const
{
  const std::lock_guard<std::mutex> lock(m_ctx_lock);
  EMU_DBG("%s: %zu contexts", context, m_ctx_map.size());
  for (const auto& [handle, ctx] : m_ctx_map) {
    EMU_DBG("  ctx %u: cols=%u cus=%u submissions=%" PRIu64,
            handle, ctx.num_col, ctx.num_cus, ctx.submissions);
  }
}

} // namespace xdna_emu
