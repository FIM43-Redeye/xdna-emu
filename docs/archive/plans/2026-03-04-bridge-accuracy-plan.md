# XRT-Emulator Bridge: Accuracy Plan

Date: 2026-03-04
Status: Investigation complete, implementation pending

## What We Have Working

1. **Device discovery**: `XDNA_EMU=1` injects synthetic device at BDF `ffff:ff:1f.0`
2. **BDF-based device selection**: `XRT_DEVICE_BDF` env var in patched test.cpp
3. **memfd-backed BOs**: `drv_open()` creates memfd, `create_bo()` grows it,
   base class `drv_mmap()` maps regions -- fully working after ABI fix
4. **Host buffer registration**: `clear_host_buffers()` + `add_host_buffer()`
   from ERT regmap for DdrPatch address patching
5. **Pre/post sync**: `submit_cmd()` copies all BOs between memfd and emulator
   host memory before and after execution
6. **ERT packet parsing**: START_CU opcode, instruction address + size extraction
7. **NPU instruction execution**: `execute_from_device()` reads instructions
   from emulator host memory, parses, and runs
8. **rebuild-plugin.sh**: One-command build + install (plugin + Rust lib)
9. **emu-bridge-test.sh**: Automated 45-test comparator (bridge vs npu-test)

## Bugs Fixed This Session

### 1. mmap EBADF (ABI violation)

**Root cause**: The xdna-driver patch to `platform.h` moved `m_dev_fd` from
its original position (after `m_driver`) to before it. This changed the class
memory layout. The pre-compiled XRT library used the original layout, so
`dev_fd()` and `drv_mmap()` (compiled into XRT) read `m_dev_fd` at the old
offset (still -1), while our code wrote to the new offset.

**Diagnostic**: `m_dev_fd=3` but `dev_fd()=-1` -- two different memory locations.

**Fix**: Changed the patch to only modify the access specifier (`private:` ->
`protected:`) without reordering members. The new patch:
```diff
-private:
+protected:
   std::shared_ptr<const drv> m_driver;
   mutable int m_dev_fd = -1;
   mutable std::string m_sysfs_root;
+private:
```

### 2. clear_host_buffers() nuked emulator state

**Root cause**: `xdna_emu_clear_host_buffers()` created a brand new
`NpuExecutor`, destroying all data previously synced into host memory.

**Fix**: Changed to `handle.npu_executor.set_host_buffers(Vec::new())` which
only clears the buffer list.

### 3. Plugin not loading from local build

**Root cause**: XRT's driver_loader dlopen's plugins from `/opt/xilinx/xrt/lib/`,
not from `LD_LIBRARY_PATH`. Local builds weren't being used.

**Fix**: `rebuild-plugin.sh` copies both the plugin SO and the Rust emulator
lib to `/opt/xilinx/xrt/lib/` via pkexec.

## Current Blocker: Xclbin/PDI Not Loaded

### The Problem

The emulator runs NPU instructions (DdrPatch, write_register, etc.) but the
**xclbin is never loaded** into the emulator. This means:
- No ELF programs in compute tiles (cores halt after 1 cycle)
- No CDO configuration applied (DMA descriptors, routing)
- The NPU instruction stream alone is not enough -- it assumes the xclbin
  was already loaded by firmware

### How Real Hardware Does It

1. `xrt::device::register_xclbin(xclbin)` -- no-op in the shim, just stores UUID
2. `xrt::hw_context(device, uuid)` -> `hwctx_kmq` constructor:
   - Parses xclbin via `xclbin_parser`, extracts PDI binary per CU
   - Creates CACHEABLE BO per CU, copies PDI data into it
   - Syncs BO to device (`sync(host2device)`)
   - Calls `config_ctx_cu_config` with BO handles
3. On real NPU, firmware receives the PDI data and loads it (CDO + ELFs)

### What The Emulator Needs

The emulator's `load_xclbin(path)` FFI function does the full job:
1. Parses xclbin container
2. Finds AIE partition section
3. Extracts primary PDI
4. Finds CDO offset within PDI (bootgen header search)
5. Parses CDO (write_register, blockwrite, etc.)
6. Applies CDO to device (configures DMA, routing, writes ELFs to tile memory)
7. Auto-loads ELF files from the project directory

### Attempted Fix (Incomplete)

Added `load_pdi()` FFI function that takes raw PDI bytes and tries to find/parse
CDO within them. Called from `config_ctx_cu_config` by reading PDI data from
the synced BO.

**Failed because**: The PDI data (2608 bytes) doesn't start with a recognizable
CDO header. `find_cdo_offset()` couldn't find the magic bytes. The PDI format
from `xclbin_parser::get_cu_pdi()` may be:
- Raw CDO without bootgen header (needs offset 0 parsing)
- A different binary format entirely
- Missing the CDO magic because of format differences

## Plan: Correct PDI/Xclbin Loading

### Investigation Needed

Before implementing, we need to understand:

1. **What exactly is in `get_cu_pdi()` data?**
   - Hex dump the first 32+ bytes of the PDI BO
   - Compare with known CDO headers (magic 0x004F4443 "CDO\0" at offset 4)
   - Check if it's a bootgen image, raw CDO, or something else
   - Reference: `aietools/` may have PDI format documentation

2. **How does the NPU firmware process PDIs?**
   - The xdna-driver sends PDI data to firmware via `config_ctx_cu_config`
   - Firmware calls the PDI loader which handles bootgen headers
   - Check xdna-driver source for PDI loading path
   - Check if firmware strips headers before CDO processing

3. **What does `xclbin_parser::get_cu_pdi()` actually extract?**
   - Trace through `hwctx.cpp` to see how `m_pdi` is populated
   - The AIE_PARTITION JSON has `PDIs[].file_name` and `cdo_groups[]`
   - The binary PDI data is embedded somewhere in the xclbin container
   - `xclbinutil --dump-section` can extract it for inspection

4. **Is the PDI the same as what `load_xclbin` processes?**
   - `load_xclbin` extracts `primary_pdi().pdi_image` from the AIE partition
   - `get_cu_pdi()` extracts per-CU PDI data
   - Are these the same binary blob?

### Implementation Options (in order of preference)

#### Option A: Load xclbin by path (simplest, most reliable)

Intercept at a point where the xclbin file path is known and call the
existing `transport->load_xclbin(path)`.

Possible interception points:
- Store xclbin path when BO is created with xclbin data
- Use `XDNA_EMU_XCLBIN` environment variable (fragile but works)
- Override at the device level (harder, needs custom device class)

Advantages: reuses existing, tested code path
Disadvantages: needs the file path, not just in-memory data

#### Option B: Load xclbin from in-memory data

Add `load_xclbin_from_data(handle, data, size)` FFI function that takes
the raw xclbin bytes (not a file path) and processes them.

The xclbin data IS available in XRT's core infrastructure (stored when
`register_xclbin` is called). We'd need to extract it from the
`xrt::xclbin` object.

Advantages: doesn't need file path
Disadvantages: need to find where to intercept the xclbin object

#### Option C: Parse PDI/CDO directly (current attempt)

Process the raw PDI data from the CU config BO.

Requires understanding the exact PDI binary format. May need:
- Bootgen header stripping
- Different CDO magic detection
- Handling of PDI-specific wrapping

Advantages: uses the data that's already flowing through the system
Disadvantages: PDI format is complex and poorly documented

#### Option D: Load xclbin in hwctx creation (cleanest architecture)

Override `create_hw_context` in the device to intercept the `xrt::xclbin`
object. This requires either:
- A custom device class (pdev_emu already exists, but device creation
  goes through the shim's `device` class)
- Adding a hook in the shim's device class

The `xrt::xclbin` object has `get_axlf()` which returns the raw xclbin
data. This could be passed to `load_xclbin_from_data()`.

### Recommended Approach

**Start with Option A** (xclbin path) as a quick unblock, then move to
**Option D** (intercept at hwctx creation) for proper architecture.

For Option A:
- The test already passes `-x aie.xclbin` as an argument
- The `xrt::xclbin` object is constructed from a file path
- We could pass the path via environment variable as a quick hack
- OR: intercept in hwctx constructor where the xclbin object is available

For Option D (proper fix):
- Study how `device::create_hw_context()` works
- See if we can override it via pdev_emu or a custom device class
- Extract raw xclbin data from `xrt::xclbin::get_axlf()`
- Pass to a new `load_xclbin_from_data()` FFI function

## Files Modified (Uncommitted)

| File | Status | Description |
|------|--------|-------------|
| `src/ffi/mod.rs` | Modified | `clear_host_buffers` fix + `load_pdi` function |
| `xrt-plugin/src/platform_emu.cpp` | Modified | PDI loading in config_ctx_cu_config, cleanup diag prints |
| `xrt-plugin/src/platform_emu.h` | Unchanged | |
| `xrt-plugin/src/pdev_emu.cpp` | Modified | Removed diag print |
| `xrt-plugin/src/transport.h` | Modified | Added `load_pdi()` virtual method |
| `xrt-plugin/src/transport_inprocess.h` | Modified | Added load_pdi types + override |
| `xrt-plugin/src/transport_inprocess.cpp` | Modified | load_pdi implementation |
| `include/xdna_emu.h` | Modified | `xdna_emu_load_pdi` declaration |
| `scripts/rebuild-plugin.sh` | New | One-command build+install script |
| `xdna-driver/src/shim/platform.h` | Modified | ABI-safe protected patch |

## Test Results Context

The `emu-bridge-test.sh` results from earlier in the session (before BDF fix)
were running on **real hardware**, not the emulator. All 36 "bridge PASS"
results were false positives. Once we correctly target the emulated device
via BDF, the test produces all-zeros because the xclbin isn't loaded.

The npu-test runner works correctly for this test (add_one_using_dma: 362
cycles, 64/64 correct) because it calls `load_xclbin()` directly.

## Next Session Checklist

1. Commit the ABI fix and clear_host_buffers fix (these are correct and tested)
2. Commit rebuild-plugin.sh
3. Investigate PDI format (hex dump, compare with CDO headers)
4. Choose and implement xclbin loading approach
5. Get first genuine bridge PASS on add_one_using_dma
6. Run full three-way comparison suite
