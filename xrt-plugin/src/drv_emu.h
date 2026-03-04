// SPDX-License-Identifier: MIT
//
// drv_emu.h -- Emulator driver registration for XRT.
//
// Registers drv_emu with XRT's driver list via the same static-init
// pattern used by the real xdna shim (pcidrv_amdxdna.cpp).
//
// LIMITATION: The base xrt_core::pci::drv::scan_devices() is
// non-virtual and walks /sys/bus/pci/drivers/{name()}/. Since we have
// no real PCI device, that directory does not exist and scan_devices
// returns without adding anything. A future task will provide an
// alternative discovery path (e.g. XRT patch or environment-variable
// triggered direct device injection).

#pragma once

#include "shim/pcidrv.h"
#include <string>

namespace xdna_emu {

class drv_emu : public shim_xdna::drv
{
public:
  using drv::drv;

  std::string
  name() const override;

  std::string
  dev_node_prefix() const override;

  std::string
  dev_node_dir() const override;

  std::string
  sysfs_dev_node_dir() const override;

private:
  std::shared_ptr<xrt_core::pci::dev>
  create_pcidev(const std::string& sysfs) const override;
};

} // namespace xdna_emu
