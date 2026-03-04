// SPDX-License-Identifier: MIT
//
// pdev_emu.cpp -- Synthetic PCI device for the emulator.

#include "pdev_emu.h"
#include "platform_emu.h"
#include "shim/shim_debug.h"
#include <cstdlib>

namespace xdna_emu {

void
pdev_emu::
on_first_open() const
{
  const std::lock_guard<std::mutex> lock(m_lock);

  // Determine library path from environment or use a default.
  const char* lib_env = std::getenv("XDNA_EMU_LIB");
  std::string lib_path = lib_env ? lib_env : "libxdna_emu.so";

  m_transport = emu_transport::create_inprocess(lib_path);

  // Wire the transport into the platform driver so that its ioctl
  // handlers (create_bo, sync_bo, submit_cmd, etc.) can reach it.
  if (m_platform)
    m_platform->set_transport(m_transport.get());
}

void
pdev_emu::
on_last_close() const
{
  const std::lock_guard<std::mutex> lock(m_lock);
  if (m_platform)
    m_platform->set_transport(nullptr);
  m_transport.reset();
}

bool
pdev_emu::
is_cache_coherent() const
{
  // Return false so that sync_bo is called by the shim.  The emulator
  // has its own internal memory -- sync_bo copies data between the
  // host-side memfd and the emulator's device memory.
  return false;
}

uint64_t
pdev_emu::
get_heap_paddr() const
{
  // No device heap in emulation mode.  Return zero; callers should
  // check is_umq() and avoid using the heap path.
  return 0;
}

void*
pdev_emu::
get_heap_vaddr() const
{
  return nullptr;
}

bool
pdev_emu::
is_umq() const
{
  // We use KMQ-style command submission (exec buf), not UMQ.
  return false;
}

void
pdev_emu::
create_drm_bo(shim_xdna::bo_info* arg) const
{
  // Delegate to the platform driver via the normal ioctl path.
  // platform_drv_emu::create_bo will route through the transport.
  drv_ioctl(shim_xdna::drv_ioctl_cmd::create_bo, arg);
}

emu_transport*
pdev_emu::
get_transport() const
{
  return m_transport.get();
}

} // namespace xdna_emu
