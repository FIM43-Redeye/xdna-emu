# XRT Driver Plugin Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task.

**Goal:** Build `libxrt_driver_emu.so.2`, an XRT driver plugin that routes
NPU operations to the xdna-emu Rust emulator, giving 100% XRT API coverage
and coexistence with real hardware.

**Architecture:** XRT auto-discovers the plugin at startup. The plugin
registers a synthetic emulator device activated by `XDNA_EMU=1`. It compiles
the xdna-driver shim source (device, hwctx, buffer, hwq, fence) unmodified,
replacing only the platform driver backend with one that routes to xdna-emu
via a transport abstraction (in-process dlopen initially, socket-based later).

**Tech Stack:** C++17, CMake, XRT 2.21.0 internal headers (from xdna-driver
submodule), Rust FFI (xdna_emu.h), dlopen/dlsym for emulator loading.

**Reference**: Design at `docs/plans/2026-03-03-xrt-plugin-design.md`

---

### Task 1: Add xdna-driver Git Submodule

**Files:**
- Create: `.gitmodules` entry for xdna-driver
- Create: `xdna-driver/` (submodule checkout)

**Step 1: Add submodule**

```bash
cd /home/triple/npu-work/xdna-emu
git submodule add https://github.com/amd/xdna-driver.git xdna-driver
```

**Step 2: Initialize XRT nested submodule**

XRT is a submodule inside xdna-driver:

```bash
cd xdna-driver
git submodule update --init xrt
cd ..
```

**Step 3: Pin to a known-good commit**

The locally installed XRT is version 2.21.0. Pin the submodule to the
matching tag or commit:

```bash
cd xdna-driver
git log --oneline -5  # find the commit matching our installed version
cd ..
```

**Step 4: Verify headers exist**

```bash
ls xdna-driver/xrt/src/runtime_src/core/pcie/linux/pcidrv.h
ls xdna-driver/xrt/src/runtime_src/core/common/device.h
ls xdna-driver/xrt/src/runtime_src/core/common/ishim.h
ls xdna-driver/src/shim/platform.h
ls xdna-driver/src/shim/device.h
```

All should exist.

**Step 5: Commit**

```bash
git add .gitmodules xdna-driver
git commit -m "build: add xdna-driver as submodule for XRT plugin headers"
```

---

### Task 2: CMake Build Skeleton

**Files:**
- Create: `xrt-plugin/CMakeLists.txt`
- Create: `xrt-plugin/src/stub.cpp` (temporary, just to verify build)

**Step 1: Create directory structure**

```bash
mkdir -p xrt-plugin/src
```

**Step 2: Write CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.16)
project(xrt_driver_emu VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# XRT installation
set(XRT_INSTALL_DIR "/opt/xilinx/xrt" CACHE PATH "XRT installation directory")
set(XRT_INCLUDE_DIR "${XRT_INSTALL_DIR}/include")

# xdna-driver source (submodule)
set(XDNA_DRV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../xdna-driver" CACHE PATH
    "xdna-driver source directory")
set(XRT_SRC_DIR "${XDNA_DRV_DIR}/xrt/src/runtime_src")
set(SHIM_SRC_DIR "${XDNA_DRV_DIR}/src/shim")

# Verify critical paths exist
if(NOT EXISTS "${XRT_SRC_DIR}/core/common/device.h")
  message(FATAL_ERROR "XRT source not found at ${XRT_SRC_DIR}. "
    "Did you initialize the xdna-driver submodule?")
endif()

# Find XRT libraries
find_library(XRT_CORE_LIB xrt_core PATHS "${XRT_INSTALL_DIR}/lib" REQUIRED)
find_library(XRT_COREUTIL_LIB xrt_coreutil PATHS "${XRT_INSTALL_DIR}/lib" REQUIRED)

# Include paths
include_directories(
  ${XRT_INCLUDE_DIR}
  ${XRT_SRC_DIR}
  ${XDNA_DRV_DIR}/src
  ${SHIM_SRC_DIR}
  # DRM local headers from xdna-driver
  ${XDNA_DRV_DIR}/src/driver/amdxdna
)

# Plugin sources -- our new code
set(EMU_SOURCES
  src/stub.cpp  # temporary placeholder
)

# Shim sources from xdna-driver (reused as-is)
# These will be added incrementally as we verify they compile
set(SHIM_SOURCES
  # Added in later tasks as dependencies resolve
)

# Build shared library
add_library(xrt_driver_emu SHARED ${EMU_SOURCES} ${SHIM_SOURCES})

# Link against XRT core
target_link_libraries(xrt_driver_emu PRIVATE ${XRT_CORE_LIB} ${XRT_COREUTIL_LIB})

# Version the shared library to match XRT convention
set_target_properties(xrt_driver_emu PROPERTIES
  VERSION 2.21.0
  SOVERSION 2
  OUTPUT_NAME "xrt_driver_emu"
)

# Defines matching xdna-driver build
target_compile_definitions(xrt_driver_emu PRIVATE
  XRT_ENABLE_AIE
  XRT_AIE_BUILD
  XRT_BUILD
)

# Install to XRT lib directory
install(TARGETS xrt_driver_emu LIBRARY DESTINATION "${XRT_INSTALL_DIR}/lib")
```

**Step 3: Write stub.cpp**

```cpp
// Temporary stub to verify the build system works.
// Will be replaced by real plugin source files.
namespace { struct stub {}; }
```

**Step 4: Build and verify**

```bash
cd xrt-plugin
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

Expected: `libxrt_driver_emu.so.2.21.0` built successfully.

**Step 5: Commit**

```bash
git add xrt-plugin/CMakeLists.txt xrt-plugin/src/stub.cpp
git commit -m "build: xrt-plugin CMake skeleton with xdna-driver includes"
```

---

### Task 3: Transport Abstraction

**Files:**
- Create: `xrt-plugin/src/transport.h`

**Step 1: Write abstract transport interface**

```cpp
// xrt-plugin/src/transport.h
//
// Abstract interface between the XRT plugin and the emulator.
// Initial implementation: in-process via dlopen (transport_inprocess).
// Future: socket-based for GUI attachment (transport_socket).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace xdna_emu {

class emu_transport {
public:
    virtual ~emu_transport() = default;

    // Lifecycle
    virtual void load_xclbin(const void* data, size_t size,
                             uint8_t uuid_out[16]) = 0;

    // Buffer management
    virtual uint64_t alloc_buffer(size_t size) = 0;
    virtual void free_buffer(uint64_t addr) = 0;
    virtual void write_memory(uint64_t addr, const void* data,
                              size_t size) = 0;
    virtual void read_memory(uint64_t addr, void* data,
                             size_t size) = 0;

    // Execution
    virtual void execute(const void* instructions, size_t size) = 0;
    virtual bool poll_completion() = 0;

    // Debug / AIE access
    virtual uint32_t read_reg(uint16_t col, uint16_t row,
                              uint32_t addr) = 0;
    virtual void write_reg(uint16_t col, uint16_t row,
                           uint32_t addr, uint32_t val) = 0;
    virtual void read_tile_memory(uint16_t col, uint16_t row,
                                  uint32_t offset, uint32_t size,
                                  void* out) = 0;
    virtual void write_tile_memory(uint16_t col, uint16_t row,
                                   uint32_t offset, uint32_t size,
                                   const void* data) = 0;

    // Factory
    static std::unique_ptr<emu_transport> create_inprocess(
        const std::string& lib_path);
};

} // namespace xdna_emu
```

**Step 2: Commit**

```bash
git add xrt-plugin/src/transport.h
git commit -m "feat: transport abstraction interface for emulator communication"
```

---

### Task 4: In-Process Transport (dlopen)

**Files:**
- Create: `xrt-plugin/src/transport_inprocess.h`
- Create: `xrt-plugin/src/transport_inprocess.cpp`

**Step 1: Write header**

```cpp
// xrt-plugin/src/transport_inprocess.h
#pragma once

#include "transport.h"
#include <dlfcn.h>

namespace xdna_emu {

// Function pointer types matching include/xdna_emu.h C API
using XdnaEmuHandle = void;
using create_fn = XdnaEmuHandle* (*)();
using destroy_fn = void (*)(XdnaEmuHandle*);
using load_xclbin_fn = int (*)(XdnaEmuHandle*, const char*, uint8_t*);
using alloc_buffer_fn = uint64_t (*)(XdnaEmuHandle*, size_t);
using free_buffer_fn = void (*)(XdnaEmuHandle*, uint64_t);
using write_mem_fn = int (*)(XdnaEmuHandle*, uint64_t, const void*, size_t);
using read_mem_fn = int (*)(XdnaEmuHandle*, uint64_t, void*, size_t);
using execute_fn = int (*)(XdnaEmuHandle*, const void*, size_t);
using run_fn = int (*)(XdnaEmuHandle*);
using read_reg_fn = uint32_t (*)(XdnaEmuHandle*, uint16_t, uint16_t, uint32_t);
using write_reg_fn = int (*)(XdnaEmuHandle*, uint16_t, uint16_t, uint32_t, uint32_t);
using read_tile_mem_fn = int (*)(XdnaEmuHandle*, uint16_t, uint16_t,
                                 uint32_t, uint32_t, void*);
using write_tile_mem_fn = int (*)(XdnaEmuHandle*, uint16_t, uint16_t,
                                  uint32_t, uint32_t, const void*);

class emu_transport_inprocess : public emu_transport {
public:
    explicit emu_transport_inprocess(const std::string& lib_path);
    ~emu_transport_inprocess() override;

    void load_xclbin(const void* data, size_t size,
                     uint8_t uuid_out[16]) override;
    uint64_t alloc_buffer(size_t size) override;
    void free_buffer(uint64_t addr) override;
    void write_memory(uint64_t addr, const void* data, size_t size) override;
    void read_memory(uint64_t addr, void* data, size_t size) override;
    void execute(const void* instructions, size_t size) override;
    bool poll_completion() override;
    uint32_t read_reg(uint16_t col, uint16_t row, uint32_t addr) override;
    void write_reg(uint16_t col, uint16_t row, uint32_t addr,
                   uint32_t val) override;
    void read_tile_memory(uint16_t col, uint16_t row,
                          uint32_t offset, uint32_t size,
                          void* out) override;
    void write_tile_memory(uint16_t col, uint16_t row,
                           uint32_t offset, uint32_t size,
                           const void* data) override;

private:
    void* m_lib_handle = nullptr;
    XdnaEmuHandle* m_emu_handle = nullptr;

    // Resolved function pointers
    destroy_fn m_destroy = nullptr;
    load_xclbin_fn m_load_xclbin = nullptr;
    alloc_buffer_fn m_alloc_buffer = nullptr;
    free_buffer_fn m_free_buffer = nullptr;
    write_mem_fn m_write_mem = nullptr;
    read_mem_fn m_read_mem = nullptr;
    execute_fn m_execute = nullptr;
    run_fn m_run = nullptr;
    read_reg_fn m_read_reg = nullptr;
    write_reg_fn m_write_reg = nullptr;
    read_tile_mem_fn m_read_tile_mem = nullptr;
    write_tile_mem_fn m_write_tile_mem = nullptr;

    void resolve_symbols();
};

} // namespace xdna_emu
```

**Step 2: Write implementation**

The implementation dlopen's `libxdna_emu.so`, resolves all function symbols
via dlsym, creates an emulator handle, and forwards each transport method
to the corresponding FFI function.

Key logic in the constructor:
```cpp
emu_transport_inprocess::emu_transport_inprocess(const std::string& lib_path) {
    m_lib_handle = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!m_lib_handle)
        throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
    resolve_symbols();
    auto create = (create_fn)dlsym(m_lib_handle, "xdna_emu_create");
    m_emu_handle = create();
}
```

Each method delegates to the resolved FFI function pointer.

**Step 3: Add to CMakeLists.txt**

Add `src/transport_inprocess.cpp` to `EMU_SOURCES`. Link against `dl`
for dlopen/dlsym.

**Step 4: Verify it compiles**

```bash
cd xrt-plugin/build && cmake .. && make -j$(nproc)
```

**Step 5: Commit**

```bash
git add xrt-plugin/src/transport_inprocess.h xrt-plugin/src/transport_inprocess.cpp
git commit -m "feat: in-process transport via dlopen for emulator FFI"
```

---

### Task 5: New Rust FFI Functions

**Files:**
- Modify: `src/ffi/mod.rs`
- Modify: `include/xdna_emu.h`

The transport calls FFI functions that don't exist yet. Add them to the
Rust side first.

**Step 1: Write failing tests**

In `src/ffi/mod.rs` test module, add tests that call the new FFI entry
points through Rust (no C needed). Test that:

- `xdna_emu_alloc_buffer(handle, 4096)` returns a non-zero address
- `xdna_emu_free_buffer(handle, addr)` succeeds
- `xdna_emu_read_register(handle, 0, 2, 0x00032000)` returns a value
  (lock register read)
- `xdna_emu_write_register(handle, 0, 2, 0x00032000, 1)` succeeds
- `xdna_emu_read_tile_memory(handle, 0, 2, 0, 64, buf)` reads tile memory
- `xdna_emu_write_tile_memory(handle, 0, 2, 0, 4, data)` writes tile memory

**Step 2: Run tests to verify they fail**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib ffi -- --nocapture
```

Expected: compilation errors (functions don't exist yet).

**Step 3: Implement the FFI functions**

Each function follows the existing pattern in `src/ffi/mod.rs`:
- Validate handle (null check)
- Lock the emulator state
- Perform the operation
- Return status code

`alloc_buffer`: Allocate a region in host memory, return address.
Uses `DeviceState::host_memory` to track allocations.

`free_buffer`: Deallocate a previously allocated host memory region.

`read_register`/`write_register`: Route through `DeviceState::read_register`
/ `DeviceState::write_register` using `TileAddress::encode(col, row, offset)`.

`read_tile_memory`/`write_tile_memory`: Access tile local memory directly
via `Tile::memory` byte arrays.

**Step 4: Update C header**

Add declarations to `include/xdna_emu.h`:

```c
uint64_t xdna_emu_alloc_buffer(XdnaEmuHandle* handle, size_t size);
void xdna_emu_free_buffer(XdnaEmuHandle* handle, uint64_t addr);
uint32_t xdna_emu_read_register(XdnaEmuHandle* handle,
    uint16_t col, uint16_t row, uint32_t reg_addr);
int xdna_emu_write_register(XdnaEmuHandle* handle,
    uint16_t col, uint16_t row, uint32_t reg_addr, uint32_t value);
int xdna_emu_read_tile_memory(XdnaEmuHandle* handle,
    uint16_t col, uint16_t row, uint32_t offset, uint32_t size, void* out);
int xdna_emu_write_tile_memory(XdnaEmuHandle* handle,
    uint16_t col, uint16_t row, uint32_t offset, uint32_t size,
    const void* data);
```

**Step 5: Run tests**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib ffi -- --nocapture
```

Expected: All new tests pass.

**Step 6: Build the shared library**

```bash
cargo build --release
ls target/release/libxdna_emu.so
```

**Step 7: Commit**

```bash
git add src/ffi/mod.rs include/xdna_emu.h
git commit -m "feat: add FFI functions for register/memory/buffer access"
```

---

### Task 6: Driver Registration (drv_emu)

**Files:**
- Create: `xrt-plugin/src/drv_emu.h`
- Create: `xrt-plugin/src/drv_emu.cpp`

**Step 1: Write drv_emu**

This is the entry point. When XRT dlopen's our plugin, the static
constructor registers `drv_emu` with the XRT driver registry.

```cpp
// drv_emu.h
#pragma once
#include "shim/pcidrv.h"  // shim_xdna::drv base class

namespace xdna_emu {

class drv_emu : public shim_xdna::drv {
public:
    std::string name() const override;
    std::string dev_node_prefix() const override;
    std::string dev_node_dir() const override;
    std::string sysfs_dev_node_dir() const override;

    // Override scan_devices to return synthetic emulator device
    void scan_devices(
        std::vector<std::shared_ptr<xrt_core::pci::dev>>& ready,
        std::vector<std::shared_ptr<xrt_core::pci::dev>>& nonready
    ) const;

    std::shared_ptr<xrt_core::pci::dev>
    create_pcidev(const std::string& sysfs) const override;
};

} // namespace xdna_emu
```

```cpp
// drv_emu.cpp
#include "drv_emu.h"
#include "pdev_emu.h"
#include "core/pcie/linux/pcidrv.h"
#include <cstdlib>

namespace xdna_emu {

std::string drv_emu::name() const { return "xdna_emu"; }
std::string drv_emu::dev_node_prefix() const { return "accel"; }
std::string drv_emu::dev_node_dir() const { return "accel"; }
std::string drv_emu::sysfs_dev_node_dir() const { return "accel"; }

void drv_emu::scan_devices(
    std::vector<std::shared_ptr<xrt_core::pci::dev>>& ready,
    std::vector<std::shared_ptr<xrt_core::pci::dev>>& /*nonready*/
) const {
    const char* env = std::getenv("XDNA_EMU");
    if (!env)
        return;  // No emulator device when env var not set

    auto dev = std::make_shared<pdev_emu>(shared_from_this(), env);
    ready.push_back(std::move(dev));
}

std::shared_ptr<xrt_core::pci::dev>
drv_emu::create_pcidev(const std::string&) const {
    // Not used -- scan_devices creates devices directly
    return nullptr;
}

} // namespace xdna_emu

// Static registration: runs when XRT dlopen's this plugin
namespace {
struct register_emu_driver {
    register_emu_driver() {
        xrt_core::pci::register_driver(
            std::make_shared<xdna_emu::drv_emu>());
    }
} s_register;
}
```

**Step 2: Verify the static registration compiles**

This requires linking against `libxrt_core.so` which exports
`xrt_core::pci::register_driver()`.

**Step 3: Commit**

```bash
git add xrt-plugin/src/drv_emu.h xrt-plugin/src/drv_emu.cpp
git commit -m "feat: drv_emu registers emulator device with XRT"
```

---

### Task 7: Synthetic PCI Device (pdev_emu)

**Files:**
- Create: `xrt-plugin/src/pdev_emu.h`
- Create: `xrt-plugin/src/pdev_emu.cpp`

**Step 1: Write pdev_emu**

This represents the emulator as a PCI device to XRT. It inherits from
`shim_xdna::pdev` and provides emulator-specific implementations.

Key overrides:
- `is_cache_coherent()` -> true (emulator memory IS host memory)
- `is_umq()` -> false (use KMQ path, simpler)
- `on_first_open()` -> create emu_transport instance
- `on_last_close()` -> destroy transport
- `create_drm_bo()` -> allocate via transport
- `mmap()` / `munmap()` -> redirect to heap (no kernel mmap)

The pdev_emu holds the `emu_transport` instance and a
`platform_drv_emu` that routes ioctl dispatch to the transport.

Sysfs reads (`get_sysfs`) return canned values for device properties:
device_type="emu", vbnv="NPU Phoenix Emulator", etc.

**Step 2: Handle mmap without kernel**

The critical challenge: `shim_xdna::pdev::mmap()` calls
`m_driver->drv_mmap()` which calls `::mmap(device_fd, ...)`. Since we
have no device fd, we need to intercept this.

Approach: `pdev_emu` overrides `mmap()` and `munmap()` to use `malloc`
and `free` respectively. The `map_offset` field in `bo_info` becomes
an index into our buffer table rather than a DRM GEM offset.

If `mmap` is not virtual on the base class, we may need to compile a
patched version of `platform.cpp` that makes `drv_mmap` virtual, or
compile a custom `buffer_emu.cpp` that avoids the mmap path entirely.

**Investigation step**: Check whether `pdev::mmap()` is virtual. If not,
determine the minimal patch needed.

**Step 3: Commit**

```bash
git add xrt-plugin/src/pdev_emu.h xrt-plugin/src/pdev_emu.cpp
git commit -m "feat: synthetic PCI device for emulator"
```

---

### Task 8: Platform Driver (platform_drv_emu)

**Files:**
- Create: `xrt-plugin/src/platform_emu.h`
- Create: `xrt-plugin/src/platform_emu.cpp`

**Step 1: Write platform_drv_emu**

Inherits `shim_xdna::platform_drv`. Overrides all 26 virtual methods.
Holds a pointer to the `emu_transport` (owned by pdev_emu).

Key method implementations:

**Context management**:
- `create_ctx()`: Assign a local context ID. Store in a map.
  Set `arg.ctx_handle = next_ctx_id++`.
- `destroy_ctx()`: Remove from map.

**Buffer management**:
- `create_bo()`: Call `transport->alloc_buffer(size)`. Fill `bo_info`
  with the returned address. Assign a local handle ID.
  `arg.xdna_addr = addr; arg.vaddr = malloc(size); arg.bo.handle = id;`
- `destroy_bo()`: Call `transport->free_buffer(addr)`. Free local memory.
- `sync_bo()`: For host-to-device: `transport->write_memory()`.
  For device-to-host: `transport->read_memory()`.

**Command execution**:
- `submit_cmd()`: Extract NPU instruction buffer from the command BO.
  Call `transport->execute(instructions, size)`. Assign sequence number.
- `wait_cmd_ioctl()`: Call `transport->poll_completion()`. Block until done.

**Device info**:
- `get_info()`: Return canned topology data (5 columns, 6 rows, etc.).
- `get_info_array()`: Return tile descriptions.

**Stubs** (initially):
- `export_bo()`, `import_bo()`: throw not_supported
- `create_syncobj()`, `destroy_syncobj()`, etc.: throw not_supported
- `set_state()`: no-op
- `get_sysfs()`, `put_sysfs()`: return canned values

**Step 2: Commit**

```bash
git add xrt-plugin/src/platform_emu.h xrt-plugin/src/platform_emu.cpp
git commit -m "feat: platform_drv_emu routes ioctl dispatch to emulator"
```

---

### Task 9: Compile Shim Source and Resolve Dependencies

**Files:**
- Modify: `xrt-plugin/CMakeLists.txt`
- Possibly create: `xrt-plugin/src/platform_base_emu.cpp` (mmap patch)

**Step 1: Add shim sources to CMakeLists.txt**

```cmake
set(SHIM_SOURCES
  ${SHIM_SRC_DIR}/device.cpp
  ${SHIM_SRC_DIR}/hwctx.cpp
  ${SHIM_SRC_DIR}/hwq.cpp
  ${SHIM_SRC_DIR}/buffer.cpp
  ${SHIM_SRC_DIR}/fence.cpp
  ${SHIM_SRC_DIR}/pcidev.cpp
  ${SHIM_SRC_DIR}/pcidrv.cpp
  ${SHIM_SRC_DIR}/platform.cpp
)
```

Do NOT include: `host/platform_host.cpp`, `host/pcidrv_amdxdna.cpp`,
any `kmq/` or `umq/` files.

**Step 2: Attempt build, identify failures**

```bash
cd xrt-plugin/build && cmake .. && make -j$(nproc) 2>&1 | head -100
```

Expected failures:
- Missing DRM headers (`drm/drm.h`, `amdxdna_accel.h`)
- Non-virtual mmap issues
- Missing KMQ/UMQ symbols

**Step 3: Resolve DRM header dependencies**

The shim files include `drm_local/amdxdna_accel.h` (UAPI header from
the kernel driver). This header defines ioctl structures but no kernel
code. It should compile in userspace. Add the include path:

```cmake
include_directories(${XDNA_DRV_DIR}/src/driver/amdxdna)
```

For `drm/drm.h`, install `libdrm-dev` or add to include path from
the system.

**Step 4: Handle mmap abstraction**

If `platform_drv::drv_mmap()` is non-virtual and buffer.cpp calls it:

Option A: Write `platform_base_emu.cpp` that compiles instead of
`platform.cpp`, making `drv_mmap` virtual.

Option B: Override at the pdev level if pdev::mmap() is virtual.

Option C: Write `buffer_emu.cpp` that replaces `buffer.cpp` with
malloc-based buffer allocation.

Determine which option works by examining the compilation errors.

**Step 5: Iterate until build succeeds**

The goal is `libxrt_driver_emu.so.2` linking without undefined symbols
(except for XRT core symbols resolved at runtime via dlopen).

**Step 6: Commit**

```bash
git add xrt-plugin/
git commit -m "build: compile xdna shim sources into emulator plugin"
```

---

### Task 10: Integration Test -- Device Discovery

**Step 1: Install the plugin**

```bash
sudo cp xrt-plugin/build/libxrt_driver_emu.so.2.21.0 /opt/xilinx/xrt/lib/
sudo ln -sf libxrt_driver_emu.so.2.21.0 /opt/xilinx/xrt/lib/libxrt_driver_emu.so.2
```

**Step 2: Verify XRT discovers the emulator device**

```bash
XDNA_EMU=1 xrt-smi examine 2>&1
```

Expected: Device list includes an emulator entry alongside real NPU.

If xrt-smi crashes or doesn't show the device, debug with:
```bash
XDNA_EMU=1 LD_DEBUG=libs xrt-smi examine 2>&1 | grep xrt_driver
```

This confirms XRT found and loaded our plugin.

**Step 3: Verify without env var**

```bash
xrt-smi examine 2>&1
```

Expected: Only real NPU device, no emulator. Plugin loaded but
scan_devices returned empty.

**Step 4: Commit any fixes**

```bash
git add -A && git commit -m "fix: integration fixes for XRT device discovery"
```

---

### Task 11: Integration Test -- Simple Buffer Test

**Step 1: Write a minimal test program**

Create `xrt-plugin/tests/test_basic.cpp`:

```cpp
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <iostream>

int main() {
    // Open emulator device (should be device 0 if no real NPU,
    // or device 1 if real NPU present)
    xrt::device device(0);
    std::cout << "Device: " << device.get_name() << std::endl;

    // Allocate a buffer
    auto bo = xrt::bo(device, 4096, 0, 0);
    auto* ptr = bo.map<int>();
    ptr[0] = 42;
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "Buffer allocated, size=" << bo.size() << std::endl;
    std::cout << "PASS" << std::endl;
    return 0;
}
```

**Step 2: Compile against real XRT**

```bash
g++ -std=c++17 -I/opt/xilinx/xrt/include \
    -L/opt/xilinx/xrt/lib -lxrt_coreutil \
    -o test_basic xrt-plugin/tests/test_basic.cpp
```

**Step 3: Run against emulator**

```bash
XDNA_EMU=1 ./test_basic
```

Expected: "Device: NPU Phoenix Emulator", "Buffer allocated, size=4096",
"PASS".

**Step 4: Run against real NPU** (if available)

```bash
./test_basic
```

Expected: Works against real hardware. Same binary, different backend.

**Step 5: Commit**

```bash
git add xrt-plugin/tests/test_basic.cpp
git commit -m "test: basic XRT buffer test through emulator plugin"
```

---

### Task 12: Integration Test -- Kernel Execution

**Step 1: Run an mlir-aie test through the plugin**

Pick a simple passing test (e.g., `passthrough`). The test.exe was
previously compiled against mock_xrt. We need to recompile it against
real XRT headers:

```bash
cd /path/to/mlir-aie/test/npu-xrt/passthrough/
g++ -std=c++17 -I/opt/xilinx/xrt/include \
    -L/opt/xilinx/xrt/lib -lxrt_coreutil \
    -o test.exe test.cpp
```

Then run:

```bash
XDNA_EMU=1 ./test.exe
```

Debug any failures by comparing behavior with the old mock_xrt path.

**Step 2: Compare with real NPU**

```bash
./test.exe  # runs on real hardware
```

Compare outputs. They should be identical for simple tests.

**Step 3: Commit any fixes**

```bash
git add -A && git commit -m "fix: kernel execution through XRT plugin"
```

---

### Task 13: Delete mock_xrt

**Files:**
- Delete: `mock_xrt/` (entire directory)
- Modify: `CLAUDE.md` (update references)
- Modify: any other files referencing mock_xrt

**Step 1: Remove mock_xrt**

```bash
rm -rf mock_xrt/
```

**Step 2: Update documentation**

Replace all references to `mock_xrt` with `xrt-plugin` in:
- `CLAUDE.md`
- `ROADMAP.md`
- `.claude/components/testing.md` (if it references mock_xrt)

**Step 3: Verify nothing depends on mock_xrt**

```bash
grep -r "mock_xrt" --include="*.rs" --include="*.toml" --include="*.md" .
```

Fix any remaining references.

**Step 4: Commit**

```bash
git add -A && git commit -m "chore: delete mock_xrt, replaced by xrt-plugin"
```

---

### Task 14: Control Packet Verification

**Step 1: Compile add_one_ctrl_packet test against real XRT**

```bash
cd /path/to/mlir-aie/test/npu-xrt/add_one_ctrl_packet/
g++ -std=c++17 -I/opt/xilinx/xrt/include \
    -L/opt/xilinx/xrt/lib -lxrt_coreutil \
    -o test.exe test.cpp
```

**Step 2: Run through emulator plugin**

```bash
XDNA_EMU=1 ./test.exe
```

This exercises the full control packet path:
- XRT API -> plugin -> emulator FFI
- Emulator loads xclbin, configures DMA/routing
- test.exe sends control packets (OP_WRITE, OP_READ)
- Emulator processes packets via ctrl_packet_write (Section 1 fix)
- OP_READ generates response via TileCtrl (Section 2 fix)
- Results returned through S2MM -> host buffer -> test.exe comparison

**Step 3: Debug and fix**

The first run will likely fail. Use `RUST_LOG=xdna_emu=debug` to trace
the execution path. Compare with expected behavior from the design doc.

**Step 4: Commit fixes**

```bash
git add -A && git commit -m "fix: control packet path through XRT plugin"
```

---

### Task 15: Update XFAIL Expectations and Documentation

**Files:**
- Modify: `tests/test_overrides.toml`
- Modify: `ROADMAP.md`
- Modify: `CLAUDE.md`

**Step 1: Update test expectations**

If control packet tests now pass through the XRT plugin path, update
`test_overrides.toml` to remove XFAIL markers for:
- `add_one_ctrl_packet`
- `add_one_ctrl_packet_4_cores`
- `add_one_ctrl_packet_col_overlay`

**Step 2: Update roadmap**

Update Phase 2 (Toolchain Integration) status in `ROADMAP.md` to reflect
that XRT integration is now real (plugin-based, not mock).

**Step 3: Update CLAUDE.md**

Replace mock_xrt references with xrt-plugin documentation. Update the
testing component doc to describe the new XRT plugin workflow.

**Step 4: Commit**

```bash
git add -A && git commit -m "docs: update for XRT plugin, remove mock_xrt references"
```
