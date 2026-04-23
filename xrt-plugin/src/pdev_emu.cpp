// SPDX-License-Identifier: MIT
//
// pdev_emu.cpp -- Synthetic PCI device for the emulator.

#include "pdev_emu.h"
#include "emu_debug.h"
#include "platform_emu.h"
#include "shim/shim_debug.h"
#include <cstdlib>

namespace xdna_emu {

void
pdev_emu::
on_first_open() const
{
  const std::lock_guard<std::mutex> lock(m_lock);

  // Determine library path.
  //
  // Resolution order:
  //   1. XDNA_EMU_DIR + XDNA_EMU profile -- e.g. $XDNA_EMU_DIR/target/debug/libxdna_emu.so
  //      (explicit path override; honored as-is, no further fallbacks).
  //      XDNA_EMU="debug" or "release" selects the Cargo profile.
  //      XDNA_EMU="1" (or any other truthy value) defaults to "debug".
  //   2. Profile-named lib in the standard search path --
  //      libxdna_emu_debug.so or libxdna_emu_release.so (installed as symlinks
  //      in /opt/xilinx/xrt/lib/ by rebuild-plugin.sh).
  //   3. Plain libxdna_emu.so via ldconfig/LD_LIBRARY_PATH (legacy fallback).
  const char* dir_env = std::getenv("XDNA_EMU_DIR");
  const char* emu_env = std::getenv("XDNA_EMU");
  std::string profile = "debug";  // default
  if (emu_env) {
    std::string val(emu_env);
    if (val == "release")
      profile = "release";
    // "debug", "1", or any other truthy value -> debug
  }

  if (dir_env && dir_env[0] != '\0') {
    // Explicit override: use it verbatim, no fallbacks.
    std::string lib_path = std::string(dir_env) + "/target/" + profile +
                           "/libxdna_emu.so";
    EMU_INFO("Loading emulator library: %s (profile=%s, XDNA_EMU_DIR override)",
             lib_path.c_str(), profile.c_str());
    m_transport = emu_transport::create_inprocess(lib_path);
  } else {
    // No override: try profile-suffixed name first, then plain name.
    std::string profiled = "libxdna_emu_" + profile + ".so";
    std::string plain    = "libxdna_emu.so";
    EMU_INFO("Loading emulator library: %s (profile=%s)",
             profiled.c_str(), profile.c_str());
    try {
      m_transport = emu_transport::create_inprocess(profiled);
    } catch (const std::runtime_error& e) {
      EMU_WARN("Profile-named lib %s not found (%s); falling back to %s",
               profiled.c_str(), e.what(), plain.c_str());
      m_transport = emu_transport::create_inprocess(plain);
    }
  }

  // Pass XDNA_EMU_LOG_LEVEL through to the Rust emulator so that both
  // the C++ plugin and Rust emulator use the same verbosity setting.
  const char* log_level = std::getenv("XDNA_EMU_LOG_LEVEL");
  if (log_level && m_transport) {
    if (m_transport->set_log_level(log_level))
      EMU_INFO("Set emulator log level: %s", log_level);
    else
      EMU_WARN("Failed to set emulator log level: %s", log_level);
  }

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
