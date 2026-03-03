# XRT Driver Plugin Design

**Date**: 2026-03-03
**Status**: Design approved, implementation pending

## Goal

Replace the hand-written mock_xrt library with a proper XRT driver plugin
(`libxrt_driver_emu.so.2`) that registers with the real XRT runtime. This
gives us 100% XRT API coverage for free (XRT handles the API, we handle
the backend), coexistence with real NPU hardware, and the same test.exe
binary running on either silicon or emulator without recompilation.

## Context

### Current State: mock_xrt

The emulator currently ships a `mock_xrt/` directory containing 1,187 lines
of C++ that manually reimplement ~25% of XRT's API surface (device, bo,
kernel, run, hw_context, xclbin, plus experimental APIs). Test binaries
link against `libxrt_mock.so` instead of real XRT. This means:

- Different binary for emulator vs hardware (different link target)
- ~180 missing API methods (75% of XRT surface not covered)
- Every new XRT feature requires manual reimplementation
- No path to coexistence (can't test on both in the same session)

### XRT Driver Plugin Architecture

XRT has a clean plugin system. At startup, `libxrt_core.so` scans
`/opt/xilinx/xrt/lib/` for libraries matching `libxrt_driver_*.so.2` and
loads them via `dlopen(RTLD_NOW | RTLD_GLOBAL)`. Each plugin self-registers
via a static constructor:

```cpp
namespace {
  struct X {
    X() { xrt_core::pci::register_driver(std::make_shared<drv_emu>()); }
  } x;
}
```

The xdna driver (`libxrt_driver_xdna.so.2`) uses this exact pattern. Its
shim layer is beautifully abstracted: all hardware communication goes through
a `platform_drv` base class with ~20 virtual methods mapping to DRM ioctls.
The upper layers (device, hwctx, buffer, hwq, fence) are backend-agnostic.

### The Opportunity

Instead of reimplementing XRT's API, we implement only the platform_drv
backend. XRT handles all 240+ API methods. We handle the ~20 ioctl
equivalents by routing them to the Rust emulator.

## Design

### Architecture

```
Application (test.exe, linked against real XRT)
  |
Real XRT (libxrt_core.so, libxrt_coreutil.so)
  |
  +-- libxrt_driver_xdna.so.2    (real hardware, from xdna-driver)
  |     shim_xdna::platform_drv_host -> DRM ioctls -> amdxdna.ko -> silicon
  |
  +-- libxrt_driver_emu.so.2     (emulator, from xdna-emu)
        shim_xdna::platform_drv_emu -> emu_transport -> Rust emulator
```

Both plugins coexist. Real NPU appears as device 0 (via xdna driver).
Emulator appears as device 1 (or device 0 if no real hardware present).
Applications select via `xrt::device(index)`.

### Activation

Environment variable `XDNA_EMU` controls emulator device visibility:

- `XDNA_EMU=1`: Emulator device visible. Plugin searches standard paths
  for `libxdna_emu.so`.
- `XDNA_EMU=/path/to/libxdna_emu.so`: Explicit library path.
- Unset: Plugin loaded by XRT but `scan_devices()` returns empty. Zero
  overhead, no emulator device in device list.

### Transport Abstraction

The plugin communicates with the emulator through an abstract transport
interface. This enables two future modes without restructuring:

```cpp
class emu_transport {
public:
    virtual ~emu_transport() = default;

    // Lifecycle
    virtual void load_xclbin(const void* data, size_t size, uuid& out) = 0;

    // Buffer management
    virtual uint64_t alloc_buffer(size_t size) = 0;
    virtual void free_buffer(uint64_t addr) = 0;
    virtual void write_memory(uint64_t addr, const void* data, size_t size) = 0;
    virtual void read_memory(uint64_t addr, void* data, size_t size) = 0;

    // Execution
    virtual void execute(const void* instructions, size_t size) = 0;
    virtual bool poll_completion() = 0;

    // Debug / AIE access
    virtual uint32_t read_reg(uint16_t col, uint16_t row, uint32_t addr) = 0;
    virtual void write_reg(uint16_t col, uint16_t row, uint32_t addr, uint32_t val) = 0;
    virtual void read_tile_memory(uint16_t col, uint16_t row,
                                  uint32_t offset, uint32_t size, void* out) = 0;
    virtual void write_tile_memory(uint16_t col, uint16_t row,
                                   uint32_t offset, uint32_t size, const void* data) = 0;
};
```

**Initial implementation**: `emu_transport_inprocess` -- dlopen's
`libxdna_emu.so` and calls the C FFI functions directly. Emulator runs
inside the application's process.

**Future**: `emu_transport_socket` -- connects to a running `xdna-emu
--serve` instance via Unix socket. The emulator server owns the state and
exposes a GUI for interactive debugging. The application drives execution
while the GUI visualizes it in real-time.

The transport selection is an implementation detail behind `platform_drv_emu`.
Upper layers never know which mode is active.

### Plugin Components

**`drv_emu`** (inherits `shim_xdna::drv`)
- `name()` returns `"xdna_emu"`
- `scan_devices()` overridden: checks `XDNA_EMU` env var, returns one
  synthetic `pdev_emu` if set, empty otherwise
- `create_pcidev()` returns `pdev_emu` instance
- Self-registers via static constructor calling `xrt_core::pci::register_driver()`

**`pdev_emu`** (inherits `shim_xdna::pdev`)
- Synthetic PCI identity (no real BDF)
- `is_cache_coherent()` returns true (emulator memory is host memory)
- `is_umq()` returns false (kernel-mode queue path)
- `on_first_open()` creates `emu_transport` instance
- `on_last_close()` destroys it
- `create_drm_bo()` delegates to transport's `alloc_buffer()`
- Sysfs reads return canned device properties

**`platform_drv_emu`** (inherits `shim_xdna::platform_drv`)
- Overrides ~20 ioctl dispatch methods, mapping each to transport calls:

| platform_drv method | Transport call |
|---------------------|----------------|
| `create_ctx()` | Local context ID allocation |
| `destroy_ctx()` | Free context |
| `create_bo()` | `transport->alloc_buffer(size)` |
| `destroy_bo()` | `transport->free_buffer(addr)` |
| `sync_bo()` | `transport->write_memory()` / `read_memory()` |
| `submit_cmd()` | `transport->execute(instructions, size)` |
| `wait_cmd_ioctl()` | `transport->poll_completion()` |
| `get_info()` | Return canned topology/frequency data |
| `get_info_array()` | Return canned tile info |

**Reused from xdna-driver shim** (compiled into plugin, unmodified):
- `device.cpp` -- XRT device interface, AIE reg/mem, xclbin loading
- `hwctx.cpp` -- hardware context lifecycle, queue binding
- `buffer.cpp` -- buffer object management (map/unmap/sync)
- `hwq.cpp` -- hardware queue, command submission
- `fence.cpp` -- synchronization primitives

If specific shim files have DRM coupling too tight for emulation, the
device_emu class can override individual methods from the ishim interface
as escape hatches. The default path is full shim reuse.

### FFI Changes

The Rust FFI (`include/xdna_emu.h`) evolves to match the transport interface:

**Keep**: `xdna_emu_create`, `xdna_emu_destroy`, `xdna_emu_load_xclbin`,
`xdna_emu_write_host_memory`, `xdna_emu_read_host_memory`,
`xdna_emu_execute_npu_instructions`, `xdna_emu_run`, `xdna_emu_set_max_cycles`,
`xdna_emu_sync_cores`.

**Add**:
- `xdna_emu_alloc_buffer(handle, size) -> address`
- `xdna_emu_free_buffer(handle, address)`
- `xdna_emu_read_register(handle, col, row, addr) -> value`
- `xdna_emu_write_register(handle, col, row, addr, value)`
- `xdna_emu_read_tile_memory(handle, col, row, offset, size, out)`
- `xdna_emu_write_tile_memory(handle, col, row, offset, size, data)`

**Remove** (mock_xrt-specific):
- `xdna_emu_add_host_buffer` (replaced by `alloc_buffer`)
- `xdna_emu_clear_host_buffers` (replaced by individual `free_buffer`)
- `xdna_emu_alloc_host_region` (subsumed by `alloc_buffer`)

### Build System

**Directory structure**:
```
xdna-emu/
  xrt-plugin/                       # Replaces mock_xrt/
    CMakeLists.txt
    src/
      drv_emu.h / drv_emu.cpp       # Driver registration + discovery
      pdev_emu.h / pdev_emu.cpp     # Synthetic PCI device
      platform_emu.h / .cpp         # Ioctl dispatch -> transport
      transport.h                   # Abstract transport interface
      transport_inprocess.h / .cpp  # dlopen-based in-process backend
  xdna-driver/                      # Git submodule
    xrt/src/runtime_src/            # XRT internal headers
    src/shim/                       # Shim source compiled into plugin
```

**CMake**:
- Compiles plugin source files (~5 new .cpp files)
- Compiles xdna-driver shim files (device, hwctx, buffer, hwq, fence, pdev)
- Links against installed `libxrt_core.so` and `libxrt_coreutil.so`
- Does NOT link against `libxdna_emu.so` (dlopen'd at runtime)
- Output: `libxrt_driver_emu.so.2`
- Install target: `/opt/xilinx/xrt/lib/`

**Submodule**:
```bash
git submodule add https://github.com/amd/xdna-driver.git xdna-driver
cd xdna-driver && git submodule update --init xrt
```

### Testing Strategy

**Unit tests** (Rust): New FFI functions get Rust-side tests against
DeviceState instances.

**Plugin build**: CMake build succeeds, `nm -D` confirms expected symbols.

**Integration** (manual, iterative):
1. Install plugin to `/opt/xilinx/xrt/lib/`
2. `XDNA_EMU=1 xrt-smi examine` -- emulator device appears
3. Compile mlir-aie test.exe against real XRT (not mock_xrt)
4. `XDNA_EMU=1 ./test.exe` -- runs against emulator
5. `./test.exe` (no env var) -- runs against real NPU
6. Compare outputs

**Control packet verification**: Once plugin works,
`XDNA_EMU=1 ./add_one_ctrl_packet/test.exe` exercises the control packet
path end-to-end through real XRT -> plugin -> emulator.

### Migration from mock_xrt

1. Build plugin, verify alongside real XRT
2. Run mlir-aie tests through plugin, confirm equivalent results
3. Delete `mock_xrt/` directory entirely
4. Update documentation references

## Preparing for Future Work

These components are not built now but the design explicitly avoids
blocking them:

**Server mode transport**: The `emu_transport` abstraction accepts a
socket-based implementation without changing any upper layer. The
emulator server (`xdna-emu --serve`) owns state and exposes a GUI;
the plugin becomes a thin transport client.

**Multi-device**: `drv_emu::scan_devices()` returns a vector. Currently
one synthetic device. Extending to multiple (different architectures,
different xclbin loads) requires only changing the scan logic and
transport multiplexing.

**Syncobj / fence emulation**: `platform_drv_emu` stubs syncobj methods
initially. The interface is in place for real timeline semantics when
async execution support is needed.

**GUI attachment**: The transport abstraction's `read_reg` / `read_tile_memory`
methods give the GUI read access to emulator state. In server mode,
the GUI runs in the emulator process alongside the state. In in-process
mode, a debug thread could expose the same interface.

## Files to Create

| File | Purpose |
|------|---------|
| `xrt-plugin/CMakeLists.txt` | Build system for the plugin |
| `xrt-plugin/src/drv_emu.h/.cpp` | Driver registration and device discovery |
| `xrt-plugin/src/pdev_emu.h/.cpp` | Synthetic PCI device |
| `xrt-plugin/src/platform_emu.h/.cpp` | Ioctl dispatch to transport |
| `xrt-plugin/src/transport.h` | Abstract transport interface |
| `xrt-plugin/src/transport_inprocess.h/.cpp` | dlopen-based in-process transport |

## Files to Modify

| File | Change |
|------|--------|
| `src/ffi/mod.rs` | Add new FFI functions (alloc_buffer, read_register, etc.) |
| `include/xdna_emu.h` | Add new C function declarations, remove mock_xrt-specific ones |
| `.gitmodules` | Add xdna-driver submodule |
| `CLAUDE.md` | Update references from mock_xrt to xrt-plugin |
| `ROADMAP.md` | Update integration status |

## Files to Delete

| File | Reason |
|------|--------|
| `mock_xrt/` (entire directory) | Replaced by xrt-plugin |
