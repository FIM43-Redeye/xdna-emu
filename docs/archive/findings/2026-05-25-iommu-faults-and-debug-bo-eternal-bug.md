---
name: IOMMU faults during kernel runs trace to missing trace-BO binding; "Get bo 4 failed" is unrelated SHIM type-mismatch bug
description: Two distinct root causes for spurious dmesg noise on every kernel run. (1) IOMMU faults at 0x0-0x200 during chain-sweep runs come from kernels whose MLIR has trace events enabled but where no BO is bound to kernel.group_id(6) -- firmware writes trace events to NULL. trace-prepare.py patches build-dir test.cpp to inject the trace BO, hiding this from anyone using the patched binaries. (2) "Get bo 4 failed" in dmesg is a SHIM bug -- src/shim/buffer.cpp:875 allocates an internal metadata BO with type AMDXDNA_BO_CMD but the kernel driver requires AMDXDNA_BO_DEV for DRM_AMDXDNA_HWCTX_ASSIGN_DBG_BUF. Non-fatal but eternal.
type: project
---

# IOMMU faults and "Get bo 4 failed" -- two unrelated bugs

## TL;DR

Building a fresh test runner (`tools/txn-poll-probe`) that wraps the
chain-sweep kernel surfaced two distinct dmesg signatures that look
related but aren't:

1. **9 IOMMU faults** at addresses `0x0, 0x40, 0x80, ..., 0x200` during
   every kernel run.
2. **`aie2_hwctx_cfg_debug_bo: Get bo 4 failed`** -- an "ETERNAL" error
   that has appeared since project start.

Diagnosing them required ~3 hours of comparing my probe to `test.exe`,
which doesn't trigger the IOMMU faults. The diagnoses:

- **IOMMU faults**: our fault. The chain-sweep xclbins compile MLIR
  with trace events enabled, expecting a BO bound to `kernel.group_id(6)`.
  `test.exe` (built from the build-dir test.cpp, which is patched by
  `trace-prepare.py`) binds a 1MB trace BO. Hand-written runners
  (probe, validate-readback, etc.) that don't bind a trace BO leave
  the firmware writing trace events to NULL -> IOMMU faults.

- **"Get bo 4 failed"**: AMD's SHIM bug. `src/shim/buffer.cpp:875`
  allocates `m_metadata_bo` as a `dbg_buffer` with type
  `AMDXDNA_BO_CMD`, but the driver's
  `aie2_hwctx_cfg_debug_bo()` requires `AMDXDNA_BO_DEV`. The driver
  returns -EINVAL; XRT silently proceeds without the firmware
  debug/log/trace BO attached.

Both are real but neither blocks anything we're doing.

## How they manifested

Symptom in dmesg, every run of `txn-poll-probe` or any new chain-sweep
runner:

```
[T] amdxdna [...] [drm] *ERROR* aie2_hwctx_cfg_debug_bo: Get bo 4 failed
[T] amdxdna [...] cmdbuf: 00000060: 00000000 00000000 00000000 00000000
[T135] amdxdna [...] AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0001 address=0x0   flags=0x0027]
[T135] amdxdna [...] AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0001 address=0x40  flags=0x0027]
[T135] amdxdna [...] AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0001 address=0x80  flags=0x0027]
... (9 faults total, 0x40 apart) ...
[T] amdxdna [...] xdna_mailbox.131: Mailbox channel stopped, irq: 131
```

The run still reports `state=ERT_CMD_STATE_COMPLETED` (4) and
`submissions=1 completions=1` -- the kernel completes despite the
faults.

## Why the diagnosis took so long

`test.exe` (mlir-aie's built test binary) showed the "Get bo 4 failed"
line but **not** the IOMMU faults. My fresh build of the same `test.cpp`
source showed both. That pointed to compile-time differences, but the
binaries linked the same libxrt_coreutil.so and used the same XRT
headers.

Comparing demangled symbols revealed the actual difference:

- `test.exe`: `xrt::run::set_arg<uint&, bo&, ulong, bo&, bo&, bo&, bo&>`
  -- **4 BOs** after instr_size.
- my build: `xrt::run::set_arg<uint&, bo&, ulong, bo&, bo&, bo&>` --
  **3 BOs** after instr_size.

That meant `test.exe` was built from a DIFFERENT test.cpp than the one
in the source tree. Finding the difference: `build/test/...
_diag_shim_chain_sweep/k8/test.cpp` (the build-dir copy compiled by
`run.lit`) contains injected code from `trace-prepare.py`:

```cpp
// Trace buffer (injected by trace-prepare.py)
constexpr size_t _xdna_trace_size = 1048576;  // 1MB
auto bo_trace = xrt::bo(device, _xdna_trace_size, XRT_BO_FLAGS_HOST_ONLY,
                         kernel.group_id(6));
memset(bo_trace.map<void*>(), 0, _xdna_trace_size);
bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);
...
auto run = kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_inB, bo_out, bo_trace);
```

The source-tree `test.cpp` has only 3 BOs. `trace-prepare.py` (called
by `scripts/emu-bridge-test.sh`'s compile phase) tree-sitter-patches
the build-dir copy to add `bo_trace` as the 7th kernel arg. The
patched version is what gets compiled into `test.exe`.

## Bug 1: IOMMU faults from missing trace BO

The chain-sweep MLIR includes a TraceEvent runtime sequence that
configures shim DMA to write trace events to a host BO. The kernel
expects that BO at `kernel.group_id(6)` (arg index 6 in the kernel
signature, after opcode/instr/ninstr/bo0/bo1/bo2).

Without a BO bound to arg 6:
- The cmdbuf's `bo3` slot (cmdbuf offset 0x60) is zero.
- Firmware programs the trace shim DMA BD with a NULL DRAM address.
- When the kernel fires trace events, shim DMA writes to NULL.
- IOMMU faults at addresses 0x0-0x200 (9 events, each at a 0x40 stride
  -- one per packet header / shim DMA BD entry boundary).

The faults are non-fatal because the trace writes aren't observed by
the actual kernel data path -- the kernel still completes its
specified work.

**Fix for txn-poll-probe**: allocate a 1MB trace BO at `kernel.group_id(6)`
and pass it as the 7th kernel arg. Confirmed: IOMMU faults
disappear cleanly after the fix.

**Broader implication**: any new test runner built against
chain-sweep-style xclbins (or any aiecc-compiled xclbin with trace
enabled) needs to bind a trace BO. Possible options for systematizing
this:

1. Make the trace BO requirement explicit in the xclbin metadata so
   tools can detect and refuse to run without binding.
2. Add a "discover trace requirement" helper to xdna-emu tools.
3. Document the convention in xdna-emu/CLAUDE.md so future Claude
   sessions don't rediscover it.

## Bug 2: "Get bo 4 failed" -- SHIM type mismatch

Tracing via `strace -e ioctl -v`:

```
ioctl(DRM_IOCTL_AMDXDNA_CREATE_BO, {size=0x58, type=4 [BO_CMD]}) -> handle=4
ioctl(DRM_IOCTL_AMDXDNA_CONFIG_HWCTX, {ctx=1, param_type=1 [ASSIGN_DBG_BUF],
       param_val=4}) -> -1 EINVAL
ioctl(DRM_IOCTL_GEM_CLOSE, {handle=4})
ioctl(DRM_IOCTL_GEM_CLOSE, {handle=3})
```

So the SHIM:
1. Creates an 88-byte BO with `type=AMDXDNA_BO_CMD` (handle 4).
2. Calls `DRM_AMDXDNA_HWCTX_ASSIGN_DBG_BUF` with that handle.
3. Driver responds with -EINVAL.
4. SHIM closes handles 4 and 3 (also cleans up an earlier 32KB BO_SHARE).
5. Userspace BO allocation continues, reusing handles starting at 3.

The driver's enforcement is in `src/driver/amdxdna/aie2_ctx.c:996`:

```c
abo = amdxdna_gem_get_obj(client, bo_hdl, AMDXDNA_BO_DEV);
if (!abo) {
    XDNA_ERR(xdna, "Get bo %d failed", bo_hdl);
    ret = -EINVAL;
    goto free_cmd;
}
```

`amdxdna_gem_get_obj` with the `AMDXDNA_BO_DEV` filter only returns
non-null if the BO's type matches. BO_CMD (type 4) != BO_DEV (type 3)
-> returns NULL -> EINVAL.

The SHIM-side bug is in `src/shim/buffer.cpp:875`:

```cpp
void uc_dbg_buffer::config(const xrt_core::hwctx_handle* hwctx, ...) {
    ...
    m_metadata_bo = std::make_unique<dbg_buffer>(m_pdev, meta_buf_size,
                                                  AMDXDNA_BO_CMD);  // <-- BUG
    ...
    m_metadata_bo->bind_hwctx(*ctx);  // triggers ASSIGN_DBG_BUF
}
```

This path fires whenever XRT instantiates an `XRT_BO_USE_LOG` /
`USE_DTRACE` / `USE_DEBUG_QUEUE` / `USE_UC_DEBUG` buffer. Those are
all the firmware-side telemetry / DPT buffers. The intent appears to
be: SHIM creates a metadata descriptor BO that the firmware reads
out-of-band to set up debug/log/trace writes; the descriptor needs to
be a device-side BO so firmware can DMA-read it. The SHIM allocates
it as `BO_CMD`; the driver expects `BO_DEV`.

**Likely candidate fix**: change `AMDXDNA_BO_CMD` -> `AMDXDNA_BO_DEV`
on line 875.

### Tree investigation -- the "is our SHIM stale?" check

After Maya's instinct ("legacy code mixed up before"), verified the
tree picture carefully before considering a fix:

1. **SHIM source tree.** Only one exists: `src/shim/`. The
   `drivers/accel/` subtree is kernel-driver-only (Linux upstream
   submission target); userspace SHIM lives only in `src/`. Both
   `CMake/upstream.cmake` and `CMake/native.cmake` `add_subdirectory(src/shim)`
   -- there's no "modern" SHIM source we missed.

2. **The loaded SHIM was built from src/shim/buffer.cpp@HEAD.** Confirmed:
   - `/opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2.23.0` size and mtime
     match `xdna-driver/build/Release/src/shim/libxrt_driver_xdna.so.2.23.0`
     exactly (built same wallclock second, 2026-05-24 15:06).
   - `strings` shows `_GLOBAL__sub_I_buffer.cpp` in the .so symbols.
   - Source has `AMDXDNA_BO_CMD` on line 875 of HEAD.

3. **Driver-side enforcement is the same on BOTH kernel trees**:
   - `src/driver/amdxdna/aie2_ctx.c:996`: `amdxdna_gem_get_obj(client,
     bo_hdl, AMDXDNA_BO_DEV)`
   - `drivers/accel/amdxdna/aie2_ctx.c:994`: identical.
   Both require BO_DEV. No "fix already upstream that we missed."

### Git archaeology -- this is a regression

`git log -S "AMDXDNA_BO_CMD" -- src/shim/buffer.cpp`:

- **2024-05-09 (PR #72, "write buffer support - part 1")**: driver
  begins requiring `AMDXDNA_BO_DEV` for `amdxdna_gem_get_obj` in
  `aie2_hwctx_cfg_debug_bo`. At this time, the SHIM passed
  `AMDXDNA_BO_SHARE`. **Already broken** -- BO_SHARE != BO_DEV.
- **2025-07-16 (commit 4cadd9d8)**: Brian Xu introduces the
  `uc_dbg_buffer::config()` method with the `m_metadata_bo`
  allocation, still as `AMDXDNA_BO_SHARE`. Bug persists.
- **2025-08-04 (PR #674, commit eae07ad0)**: David Zhang's "Enable
  in-memory fw log for debug type" changes `AMDXDNA_BO_SHARE` ->
  `AMDXDNA_BO_CMD`. **Still broken** (different broken value, same
  failure mode).

So the debug-BO attach path has been **broken since 2024-05-09**.
Every kernel run that goes through `XRT_BO_USE_LOG` /
`USE_DTRACE` / `USE_DEBUG_QUEUE` / `USE_UC_DEBUG` (which is approx
every kernel run XRT spawns) has hit the EINVAL silently.

The 2025-08-04 commit's intent was to "enable in-memory fw log for
debug type" -- but the change couldn't have ever worked at the driver
boundary. Possible explanations:
- AMD's internal QA passes either don't exercise this path or don't
  flag "Get bo %d failed" in dmesg as a failure.
- The PR was tested on a tree where the driver had a different
  enforcement (e.g., NPU4-only? `drivers/accel` only? a SHIM-only
  smoke test?). Worth checking if the AIE4/NPU4 driver path has a
  different gem_get_obj filter.

### Implication for our work

The Phoenix DPT (firmware Debug/Profile/Trace) infrastructure -- one
of the planned avenues for our cycle-accuracy investigation -- relies
on XRT being able to attach FW_LOG / FW_TRACE buffers. If this attach
path returns EINVAL silently, **we'd see no DPT data on Phoenix even
if firmware were producing it correctly**. The "Get bo 4 failed" is
not just dmesg noise -- it's blocking the entire firmware-side
observability surface.

This is also consistent with the earlier 2026-05-25 findings on
Phoenix's `txn_op_idx` field appearing as zero: the AIE2_APP_HEALTH
feature bit is off because Phoenix's feature table doesn't enable it.
Different proximate cause, same end state (firmware-side observability
is unwired on our setup).

## Impact

**On our work**:
- Bug 1 (IOMMU faults) was a concern; now solved with a one-line BO
  allocation. No emulator-side code affected.
- Bug 2 (Get bo 4) is noise; doesn't block anything but means firmware
  DPT/FW_LOG/FW_TRACE buffers never get attached -- which may explain
  why those paths return empty data on Phoenix even when we expect
  events. Worth re-investigating that direction once Bug 2 is fixed.

**On the cycle-accuracy calibration story**:
- Bug 1 was silently present in every hand-built test runner. The
  bridge-trace-runner uses `trace-prepare.py`-patched test.cpp variants,
  so its measurements have always had the trace BO bound correctly.
  Our K-sweep calibration numbers are NOT compromised.
- Bug 2 is unrelated to cycle accuracy; it only affects firmware
  observability paths we haven't been using.

## Follow-ups

- **Verify drivers/accel SHIM state.** Does `drivers/accel`-aligned
  SHIM source still have the BO_CMD/BO_DEV type mismatch? Where does
  the loaded `libxrt_driver_xdna.so.2.23.0` come from -- src/shim
  or drivers/accel?
- **One-line SHIM patch experiment.** If the bug is still there
  upstream, patch line 875 to `AMDXDNA_BO_DEV` locally, rebuild
  xdna-driver, reinstall, check that "Get bo 4" disappears AND that
  FW_LOG/FW_TRACE buffer attachment starts succeeding. Confirmation
  experiment, not a production fix.
- **Document trace-BO requirement convention.** Either in
  xdna-emu/CLAUDE.md or in `tools/README.md`. Anyone building a new
  test runner needs to know that aiecc xclbins with trace enabled
  require `bo_trace` at `kernel.group_id(6)`.
- **Submit to AMD?** If the SHIM bug is confirmed unfixed in
  drivers/accel, file an issue with reproduction steps. Low-priority
  but obviously broken.

## See also

- `xdna-driver/src/shim/buffer.cpp:875` -- bug site
- `xdna-driver/src/driver/amdxdna/aie2_ctx.c:996` -- driver
  type-enforcement (the side that's strict)
- `xdna-driver/drivers/accel/amdxdna/aie.c:319+` -- the
  drivers/accel-tree alternative checks (uses different framework)
- `xdna-emu/tools/cpp_trace_patch.py` -- the tree-sitter patcher
  that injects `bo_trace` into build-dir test.cpp
- `xdna-emu/tools/txn-poll-probe/txn-poll-probe.cpp` -- the runner
  whose IOMMU faults surfaced this
- 2026-05-25-npu-controller-dispatch-overhead.md -- the parent task
  that led here
