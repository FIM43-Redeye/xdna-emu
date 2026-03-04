// SPDX-License-Identifier: MIT
//
// drv_emu.h -- Emulator driver registration for XRT.
//
// Registers drv_emu with XRT's driver list via the same static-init
// pattern used by the real xdna shim (pcidrv_amdxdna.cpp).
//
// When XDNA_EMU=1, scan_devices() injects a synthetic pdev_emu into
// the ready list. Otherwise the driver is loaded but invisible.

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

  bool
  is_user() const override;

  std::string
  dev_node_prefix() const override;

  std::string
  dev_node_dir() const override;

  std::string
  sysfs_dev_node_dir() const override;

  void
  scan_devices(std::vector<std::shared_ptr<xrt_core::pci::dev>>& ready_list,
               std::vector<std::shared_ptr<xrt_core::pci::dev>>& nonready_list) const override;

private:
  std::shared_ptr<xrt_core::pci::dev>
  create_pcidev(const std::string& sysfs) const override;
};

} // namespace xdna_emu
