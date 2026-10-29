// SPDX-License-Identifier: MIT
//
// pdev_emu.h -- Synthetic PCI device for the emulator.
//
// Inherits shim_xdna::pdev and implements the pure virtual methods
// that would normally talk to a real DRM device. Instead, we route
// buffer operations through emu_transport.

#pragma once

#include "shim/pcidev.h"
#include "transport.h"
#include <memory>
#include <mutex>

namespace xdna_emu {

class platform_drv_emu;

class pdev_emu : public shim_xdna::pdev
{
public:
  pdev_emu(std::shared_ptr<const shim_xdna::platform_drv>& driver,
           const std::string& sysfs_name)
    : pdev(driver, sysfs_name)
    , m_platform(std::dynamic_pointer_cast<const platform_drv_emu>(driver))
  {}

  bool
  is_cache_coherent() const override;

  uint64_t
  get_heap_paddr() const override;

  void*
  get_heap_vaddr() const override;

  bool
  is_umq() const override;

  void
  create_drm_bo(shim_xdna::bo_info* arg) const override;

  // The transport owned by this device.  Created on first open,
  // destroyed on last close.
  emu_transport*
  get_transport() const;

private:
  void
  on_first_open() const override;

  void
  on_last_close() const override;

  std::shared_ptr<const platform_drv_emu> m_platform;
  mutable std::unique_ptr<emu_transport> m_transport;
  mutable std::mutex m_lock;
};

} // namespace xdna_emu
