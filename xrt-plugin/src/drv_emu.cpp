// SPDX-License-Identifier: MIT
//
// drv_emu.cpp -- Emulator driver registration for XRT.
//
// When this shared library is dlopen'd by XRT's driver_loader, the
// static constructor below calls xrt_core::pci::register_driver()
// exactly as the real xdna shim does.

#include "drv_emu.h"
#include "pdev_emu.h"
#include "platform_emu.h"
#include "core/pcie/linux/system_linux.h"
#include <cstdlib>

namespace {

// Static-init registration: runs when libxrt_driver_emu.so is dlopen'd.
struct X
{
  X() { xrt_core::pci::register_driver(std::make_shared<xdna_emu::drv_emu>()); }
} x;

} // anonymous namespace

namespace xdna_emu {

std::string
drv_emu::
name() const
{
  return "xdna_emu";
}

std::string
drv_emu::
dev_node_prefix() const
{
  return "accel";
}

std::string
drv_emu::
dev_node_dir() const
{
  return "accel";
}

std::string
drv_emu::
sysfs_dev_node_dir() const
{
  return "accel";
}

bool
drv_emu::
is_user() const
{
  return true;
}

void
drv_emu::
scan_devices(std::vector<std::shared_ptr<xrt_core::pci::dev>>& ready_list,
             std::vector<std::shared_ptr<xrt_core::pci::dev>>& /*nonready_list*/) const
{
  // Only inject the emulator device when XDNA_EMU is set.  Presence is the
  // trigger; the value is ignored ("0" still maps to off as a small
  // affordance).  XDNA_EMU_RUNTIME picks debug vs release; see pdev_emu.cpp.
  const char* env = std::getenv("XDNA_EMU");
  if (!env || std::string(env) == "0")
    return;

  // Insert the emulator device at index 0 so that `xrt::device(0)` -- the
  // convention used by all mlir-aie test binaries -- selects the emulator
  // instead of real hardware.  This way `XDNA_EMU=1 ./test.exe` just works
  // without any BDF environment variable or patched test sources.
  //
  // XRT calls scan_devices() on each registered driver in load order,
  // appending to the shared ready_list.  By inserting at the front, the
  // emulator stays at index 0 regardless of whether other drivers have
  // already added real devices.  (The synthetic BDF below is just an
  // identifier; it is no longer part of the user-facing contract.)
  ready_list.insert(ready_list.begin(), create_pcidev("ffff:ff:1f.0"));
}

std::shared_ptr<xrt_core::pci::dev>
drv_emu::
create_pcidev(const std::string& sysfs) const
{
  // Build the platform driver chain:
  //   drv_emu -> platform_drv_emu -> pdev_emu
  //
  // This mirrors pcidrv_amdxdna.cpp but without reading device_type
  // from sysfs (we have no real PCI device).
  auto driver = std::dynamic_pointer_cast<const shim_xdna::drv>(shared_from_this());
  auto platform_driver = std::dynamic_pointer_cast<const shim_xdna::platform_drv>(
    std::make_shared<const platform_drv_emu>(driver));
  return std::make_shared<pdev_emu>(platform_driver, sysfs);
}

} // namespace xdna_emu
