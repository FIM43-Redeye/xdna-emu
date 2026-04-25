# Cycle-Budget Testing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Instrument bridge tests so that cycle-count regressions (EMU spending more emulated cycles than HW) can be classified separately from wall-clock slowdowns (EMU's host process running slow), turning today's opaque TIMEOUT results into actionable `BUDGET` vs `TIMEOUT` signals.

## Phase B Pivot Note (2026-04-22, mid-execution)

The original Phase B built a `CycleCounterHelper` that called a patched `xrt::hw_context::read_aie_reg`. The XRT patch (Task 1) is correct and committed, but during Task 4 verification on real HW we discovered the underlying ioctl is gated by `MSG_OP_AIE_RW_ACCESS` — a firmware capability that Phoenix firmware 1.5.5.391 does not advertise. The kernel driver returns `-EOPNOTSUPP` from `aie2_rw_aie_reg` at `aie2_message.c:1774`.

No software-only workaround: `DRM_AMDXDNA_READ_AIE_REG` is `#ifdef AMDXDNA_AIE2_PRIV` (dev-build only, and still firmware-gated). Strix/NPU2/NPU4 and Versal likely ship with the capability, but there's no known firmware update path for our Phoenix hardware.

**Revised Phase B approach:** use mlir-aie's declarative trace infrastructure (`aie.utils.trace.configure_trace()`, `start_trace()`, `parse_trace()` plus the `aie-opt --aie-insert-trace-flows` / `--aie-inline-trace-config` passes — declarative trace IRON API landed via PR #2988). Capture `CoreEvent.ACTIVE` (= 28 = 0x1C) with timestamps via the HW's per-tile timer; trace unit writes packets to a DMA buffer; post-run `parse_trace()` yields cycle counts. No firmware RPC needed — trace config is set at compile time.

**Bonus:** this retires our atrophied in-tree trace tools (`tools/trace-inject.py`, `trace-sweep.py`, `trace-trim.py`, `trace-merge.py`, `trace-prepare.py`, and `src/bin/trace_compare.rs`) and delegates to mlir-aie's single source of truth. (Trace *comparison* may need a small replacement later — deferred.)

> **Update (2026-04-25):** the deferred replacement landed.  `tools/trace_decoder/` is an in-tree, MIT-licensed byte-stream decoder covering modes 0/1/2 (mlir-aie's `parse_trace` covers mode 0 only, and we declined to upstream new modes as an external contributor).  `tools/parse-trace.py --decoder=ours` is now the default and authoritative for the cycle-diff pipeline; `--decoder=mlir-aie` remains the cross-validation oracle for mode 0.  Not a permanent fork — when mlir-aie's `parse_trace` covers all three modes upstream, we swap back.  Trace decoding is post-mortem so the swap-back has no hot-path cost.

**Status of the old Phase B:**
- **Task 1 (XRT patch)**: remains committed on `xdna-driver` branch `xdna-emu-cycle-budget` as future-firmware-ready code. Dormant.
- **Task 2 (CycleCounterHelper)**: remains committed on `mlir-aie` branch `xdna-emu-cycle-budget` as dormant companion code. Not currently used.
- **Task 3 (bridge sed injection)**: reverted from xdna-emu `dev` in commit `4b13a9b`.
- **Task 4 (HW verification)**: superseded; will be re-done against the trace-based Phase B.

**Execution reorder:** Phase A (FFI + plugin) and Phase D (PerfCounterBank level-event fix) are independent of Phase B and proceed now. Phase B gets a separate brainstorm once Phase A + D land, then Phase C follows Phase B.

**Phase B status (2026-04-22 update):** **Done.** Phase B was re-planned and executed as a standalone document at [`2026-04-22-phase-b-trace-cycle-capture.md`](2026-04-22-phase-b-trace-cycle-capture.md). All 15 tasks landed. Validation results across 7 representative bridge tests are recorded at [`2026-04-22-phase-b-trace-cycle-capture-validation.md`](2026-04-22-phase-b-trace-cycle-capture-validation.md): pipeline produces valid `cycles.HW.<test>.<compiler>.txt` files for vector-bearing tests (`vector_scalar_using_dma` = 41181 cycles as the reference success case). Four limitations documented (scalar-kernel empty traces, single-event-per-tile degenerate deltas, `.py`-only MLIR sources, ctrlpkt-flow incompatibility) — none block Phase C. Phase C tasks below can now consume the cycles files this pipeline produces.

---

**Architecture:** Four phases land in order (B prerequisite: XRT patch) → B → A → D → C.
**XRT patch prerequisite** — stock `xrt::aie::device::read_aie_reg` takes `(pid, context_id, col, row, addr)` and opens its own AIE context on construction, conflicting with `xrt::hw_context`. We add a small additive method `xrt::hw_context::read_aie_reg(col, row, addr)` that reuses the context's own slot_idx + core_device, applies partition-relative-to-absolute col translation, and issues the shim's existing 3-arg register read. Patch lives in our local `xdna-driver/xrt/` build; upstream eventually.
**B** adds HW perf counter capture in mlir-aie test.exe binaries using the patched API.
**A** plumbs an `XDNA_EMU_MAX_CYCLES` budget through the FFI and plugin, with a structured `XDNA_EMU_STATUS:` log line the bridge parses.
**D** fixes EMU's `PerfCounterBank` to tick only on level-event assertion (matching HW `ACTIVE_CORE` semantics) so the comparison in C is trustworthy.
**C** ties it together: bridge reads HW cycles, applies a 3× tolerance (with per-test overrides), exports the budget, and classifies EMU results as `BUDGET` when exceeded.

**Tech Stack:**
- Rust (xdna-emu core, FFI crate) — add halt_reason to `XdnaEmuExecStatus`, fix level-event tick semantics
- C++ (xrt source, xrt-plugin, mlir-aie helpers) — XRT patch, env var read, status emit, HW counter helper
- Bash (`scripts/emu-bridge-test.sh`) — status parse, budget calc, results merge
- XRT (patched `xrt_hw_context.h`, `xrt_hw_context.cpp`) — exposes tile-register access scoped to the context

---

## Branching Strategy

This work spans three repositories. **No git worktree** — we want changes to flow into the existing work stream.

- `xdna-emu/` — continue on current `dev` branch. Each task commits directly.
- `mlir-aie/` — create a new local branch `xdna-emu-cycle-budget` from current HEAD. Phase B touches this branch only. Upstream the helper later, once stable.
- `xdna-driver/` (contains the `xrt/` submodule/source) — create a new local branch `xdna-emu-cycle-budget` from current HEAD. Task 1 touches this branch only. Upstream the XRT patch eventually; until then, the patched XRT is built + installed locally.

```bash
cd /home/triple/npu-work/mlir-aie
git checkout -b xdna-emu-cycle-budget
cd /home/triple/npu-work/xdna-driver
git checkout -b xdna-emu-cycle-budget
```

Rebuild requirements vary per task — each task's "Commit" step is followed by a "Rebuild" step where needed (after Task 1: XRT + shim rebuild + pkexec install; after FFI changes: rebuild libxdna_emu; after plugin changes: rebuild-plugin.sh; after mlir-aie helper changes: none, header-only).

---

## File Structure

### `xdna-driver/xrt/` (Task 1: XRT patch)

- **Modify:** `src/runtime_src/core/include/xrt/xrt_hw_context.h` — public `read_aie_reg`/`write_aie_reg` declarations on `class hw_context`.
- **Modify:** `src/runtime_src/core/common/api/xrt_hw_context.cpp` — impl on `hw_context_impl` + public forwarders.

### `mlir-aie/` (Phase B)

- **Create:** `runtime_lib/test_lib/include/cycle_counter.h` — header-only RAII helper class `test_utils::CycleCounterHelper`. Constructor takes `xrt::hw_context&`. `start()` programs perf counter 0 on every compute tile in the partition via the patched `hw_context::{read,write}_aie_reg`. Destructor, if env var `XDNA_CYCLES_OUT` is set, reads counters and writes the file.
- **Modify:** `runtime_lib/test_lib/include/test_utils.h` — `#include "cycle_counter.h"` so tests pick it up transparently.

### `xdna-emu/`

**Phase A — FFI + plugin:**
- **Modify:** `crates/xdna-emu-ffi/src/lib.rs:82-99` — add `XdnaEmuHaltReason` enum (`Completed`, `Budget`, `Error`), add `halt_reason` field to `XdnaEmuExecStatus`.
- **Modify:** `crates/xdna-emu-ffi/src/execution.rs:70-182` — loop condition to treat `max_cycles=0` as unbounded, populate `halt_reason` at exit points.
- **Create:** `crates/xdna-emu-ffi/tests/max_cycles.rs` — integration test via FFI symbols.
- **Modify:** `xrt-plugin/src/transport_inprocess.cpp` — read `XDNA_EMU_MAX_CYCLES` env var, call `sym_set_max_cycles_`, emit `XDNA_EMU_STATUS:` line after `sym_run_`.
- **Modify:** `scripts/emu-bridge-test.sh` — parse status line, add `BUDGET` classification, relax `timeout 120` → `timeout 600`, add `--no-timeout` flag, extend test.cpp sed patching to inject cycle counter helper.

**Phase D — EMU perf counter semantics:**
- **Create:** `docs/arch/perfcnt-level-events.md` — short audit listing ACTIVE_CORE + stall events, current EMU behavior, proposed refactor.
- **Modify:** `src/device/perf_counters/mod.rs` — refactor `tick()` to take level-event state (core active); only counters whose `start_event` is asserted tick.
- **Modify:** `src/device/perf_counters/tests.rs` — new tests for level-event semantics.
- **Modify:** `src/interpreter/engine/coordinator.rs:961-966` — pass core/stall state to `tick()`.

**Phase C — bridge gate:**
- **Modify:** `scripts/emu-bridge-test.sh` — `parse_hw_cycles()` function, `load_cycle_overrides()` function, budget calc, export `XDNA_EMU_MAX_CYCLES` before EMU run, merge cycle counts into results table.
- **Create:** `scripts/cycle-budget-overrides.txt` — empty override file with header comment.
- **Create:** `scripts/show-cycle-drift.sh` — diagnostic helper, sorts tests by `EMU/HW` ratio.

---

## Task 1: Patch XRT — add `hw_context::read_aie_reg` / `write_aie_reg`

**Files:**
- Modify: `/home/triple/npu-work/xdna-driver/xrt/src/runtime_src/core/include/xrt/xrt_hw_context.h`
- Modify: `/home/triple/npu-work/xdna-driver/xrt/src/runtime_src/core/common/api/xrt_hw_context.cpp`

### Why

The stock public API for AIE tile register access is `xrt::aie::device::read_aie_reg(pid, context_id, col, row, addr)`. Two problems for our use case:

1. **Constructing `xrt::aie::device` opens a new AIE context** (`open_context(access_mode)` in its constructor), which conflicts with the `xrt::hw_context` the test already holds on the same device.
2. **`context_id` is not publicly accessible** from `xrt::hw_context` — the slot_idx lives in `hw_context_impl::m_slot_idx` internally but no getter is exposed.

Rather than a second context + a new accessor, we add `read_aie_reg` / `write_aie_reg` directly on `xrt::hw_context`. The context already knows its slot_idx and core_device; the method uses those plus `getpid()` internally. No new context, no conflict, partition-relative cols handled by reusing XRT's existing `get_abs_col` logic.

- [ ] **Step 1: Create the local xdna-driver branch**

```bash
cd /home/triple/npu-work/xdna-driver
git checkout -b xdna-emu-cycle-budget
git status
```

Expected: `On branch xdna-emu-cycle-budget`, clean tree (the submodule-modified / untracked-file noise from prior build runs is pre-existing and fine to ignore).

- [ ] **Step 2: Patch the public header**

Edit `/home/triple/npu-work/xdna-driver/xrt/src/runtime_src/core/include/xrt/xrt_hw_context.h`. Inside `class hw_context`, near the existing `get_aie_coredump()` declaration (around line 272), add these public methods:

```cpp
  /**
   * read_aie_reg() - Read an AIE tile register within this hw_context's partition.
   *
   * @param col   Column index, relative to the start of this context's partition.
   * @param row   Absolute row index (row 0 = shim, row 1 = memtile, rows 2+ = compute).
   * @param reg_addr  Tile-local register offset (byte address).
   * @return      32-bit register value.
   *
   * Uses the context's own slot_idx and the caller's process id. Does not
   * require a separate xrt::aie::device.
   */
  XCL_DRIVER_DLLESPEC
  uint32_t
  read_aie_reg(uint16_t col, uint16_t row, uint32_t reg_addr) const;

  /**
   * write_aie_reg() - Write an AIE tile register within this hw_context's partition.
   *
   * See read_aie_reg() for argument semantics. Returns true on success.
   */
  XCL_DRIVER_DLLESPEC
  bool
  write_aie_reg(uint16_t col, uint16_t row, uint32_t reg_addr, uint32_t reg_val);
```

- [ ] **Step 3: Patch the implementation**

Edit `/home/triple/npu-work/xdna-driver/xrt/src/runtime_src/core/common/api/xrt_hw_context.cpp`. The file has two relevant regions: the `hw_context_impl` class (starts around line 63) and the public `xrt::hw_context` forwarders (near the end of the file, before the closing `} // xrt` around line 524).

Locate the relevant `#include` block near the top and ensure these are present (add any missing):

```cpp
#include "core/common/api/hw_context_int.h"
#include "core/common/query_requests.h"
#include "core/common/unistd.h"   // xrt_core::utils::get_pid()
```

Inside `hw_context_impl` (after the existing `get_aie_coredump()` method, around line 512), add a private helper and two new public methods:

```cpp
  // Partition-relative (col) -> absolute (col) translation. Same logic as
  // the private get_abs_col() in xrt_device.cpp:422 but scoped locally so
  // we don't need a cross-TU dependency.
  uint16_t
  partition_abs_col(uint16_t col) const
  {
    auto pid = static_cast<uint64_t>(xrt_core::utils::get_pid());
    auto slot = static_cast<uint32_t>(m_hdl->get_slotidx());
    auto data = xrt_core::device_query_default<xrt_core::query::aie_partition_info>(m_core_device.get(), {});
    for (const auto& entry : data) {
      if (entry.pid != pid || std::stoi(entry.metadata.id) != static_cast<int>(slot))
        continue;
      auto abs_col = static_cast<uint16_t>(col + entry.start_col);
      if (abs_col >= entry.start_col + entry.num_cols)
        throw std::out_of_range("read/write_aie_reg: col index out of range");
      return abs_col;
    }
    throw std::runtime_error("read/write_aie_reg: partition not found for this hw_context");
  }

  uint32_t
  read_aie_reg(uint16_t col, uint16_t row, uint32_t reg_addr) const
  {
    if ((reg_addr & 0x3) != 0)
      throw std::runtime_error("read_aie_reg: address is not 4 byte aligned");
    auto abs_col = partition_abs_col(col);
    return m_core_device->read_aie_reg(abs_col, row, reg_addr);
  }

  bool
  write_aie_reg(uint16_t col, uint16_t row, uint32_t reg_addr, uint32_t reg_val)
  {
    if ((reg_addr & 0x3) != 0)
      throw std::runtime_error("write_aie_reg: address is not 4 byte aligned");
    auto abs_col = partition_abs_col(col);
    return m_core_device->write_aie_reg(abs_col, row, reg_addr, reg_val);
  }
```

Then at the end of the file, inside the outer `namespace xrt {` block (before the `} // xrt` closer), add the public forwarders:

```cpp
uint32_t
hw_context::
read_aie_reg(uint16_t col, uint16_t row, uint32_t reg_addr) const
{
  return get_handle()->read_aie_reg(col, row, reg_addr);
}

bool
hw_context::
write_aie_reg(uint16_t col, uint16_t row, uint32_t reg_addr, uint32_t reg_val)
{
  return get_handle()->write_aie_reg(col, row, reg_addr, reg_val);
}
```

Verify `m_core_device` and `m_hdl->get_slotidx()` are accessible from the context where you're adding the impl methods (they're existing private members of `hw_context_impl`, used elsewhere in the same class — see around `xrt_hw_context.cpp:502-503`).

- [ ] **Step 4: Rebuild XRT (release)**

```bash
cd /home/triple/npu-work/xdna-driver/build
nice -n 19 ./build.sh -release -nokmod -nocmake
```

`-nokmod` skips the kernel module (we don't need to rebuild amdxdna.ko for this), `-nocmake` reuses existing cmake config. Expected: build completes in ~5-15 minutes producing a .deb under `build/Release/_CPack_Packages/...`.

If `-nocmake` fails because the build tree is stale, drop the flag and rerun (takes longer — cmake regen adds a few minutes).

- [ ] **Step 5: Install the rebuilt XRT libraries**

The simplest approach: copy just the rebuilt `libxrt_core.so*` and the updated header into `/opt/xilinx/xrt/`:

```bash
cd /home/triple/npu-work/xdna-driver/build/Release
pkexec install -m 0644 opt/xilinx/xrt/lib/libxrt_core.so.2.23.0 /opt/xilinx/xrt/lib/
pkexec install -m 0644 opt/xilinx/xrt/include/xrt/xrt_hw_context.h /opt/xilinx/xrt/include/xrt/
sudo ldconfig 2>/dev/null || pkexec ldconfig
```

If pkexec prompts and gets dismissed, retry once — per project convention, pkexec dismissals are often mis-clicks.

- [ ] **Step 6: Smoke-test the patched XRT via a small C++ snippet**

Create `/tmp/claude-1000/test-patched-xrt.cpp`:

```cpp
#include <iostream>
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"

int main() {
  // Just verify the new methods are declared and the installed header is up to date.
  // This is a compile-only test — we're not actually running against HW here.
  auto ptr_read = &xrt::hw_context::read_aie_reg;
  auto ptr_write = &xrt::hw_context::write_aie_reg;
  std::cout << "read_aie_reg ptr: " << (void*)ptr_read
            << " write_aie_reg ptr: " << (void*)ptr_write << "\n";
  return 0;
}
```

Compile with the installed XRT:

```bash
/usr/bin/clang++ -std=c++17 -I/opt/xilinx/xrt/include \
  /tmp/claude-1000/test-patched-xrt.cpp \
  -L/opt/xilinx/xrt/lib -lxrt_coreutil \
  -o /tmp/claude-1000/test-patched-xrt
/tmp/claude-1000/test-patched-xrt
```

Expected: compile succeeds; runs and prints two non-null pointers. If compile fails with "no member named 'read_aie_reg' in 'xrt::hw_context'", the installed header didn't update — redo Step 5.

- [ ] **Step 7: Commit the XRT patch**

```bash
cd /home/triple/npu-work/xdna-driver
git add xrt/src/runtime_src/core/include/xrt/xrt_hw_context.h \
        xrt/src/runtime_src/core/common/api/xrt_hw_context.cpp
git commit -m "xrt: expose read_aie_reg/write_aie_reg on hw_context

Adds public methods xrt::hw_context::{read,write}_aie_reg(col, row,
reg_addr) that reuse the context's own slot_idx and core_device.
Partition-relative col is translated to absolute via an aie_partition_info
query matching on pid + slot_idx.

This avoids forcing callers to construct an xrt::aie::device (which
opens a second AIE context via open_context() on construction and
conflicts with the existing hw_context).

To upstream: PR this to upstream XRT once xdna-emu's cycle-budget
instrumentation has validated the API in practice."
```

## Task 2: Phase B — Build the HW cycle counter helper

**Files:**
- Create: `/home/triple/npu-work/mlir-aie/runtime_lib/test_lib/include/cycle_counter.h`
- Modify: `/home/triple/npu-work/mlir-aie/runtime_lib/test_lib/include/test_utils.h`

**Depends on:** Task 1 (patched XRT installed). The helper calls `xrt::hw_context::read_aie_reg`/`write_aie_reg`, which only exists after Task 1.

- [ ] **Step 1: Create the local mlir-aie branch**

```bash
cd /home/triple/npu-work/mlir-aie
git checkout -b xdna-emu-cycle-budget
git status
```

Expected: `On branch xdna-emu-cycle-budget`, clean tree.

- [ ] **Step 2: Write the helper header**

Create `/home/triple/npu-work/mlir-aie/runtime_lib/test_lib/include/cycle_counter.h`:

```cpp
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Per-tile AIE2 performance-counter helper for xdna-emu cycle-budget tests.
//
// RAII: start() programs counter 0 on every compute tile in this
// hw_context's partition with start_event = ACTIVE_CORE, no stop, no reset.
// Destructor, when the env var XDNA_CYCLES_OUT is set, reads each counter
// and writes a text file with one "col row cycles" line per tile (plus a
// header comment).
//
// Uses the patched xrt::hw_context::{read,write}_aie_reg methods (see the
// XRT patch under xdna-driver/xrt/). (col, row) is partition-relative: the
// context's own slot_idx is used internally to resolve the partition
// range; we iterate cols 0..7 and rows 2..5, letting XRT throw out-of-range
// for cols outside the actual partition.
//
// Register offsets: Core module perf counter 0 control = 0x31500,
// value = 0x31520. ACTIVE_CORE event ID on AIE2 = 0x1C (28 decimal), per
// AM025 event enumeration and aie-rt xaiemlgbl_reginit.c.
//
// Failures on individual tiles (tile not in partition, XRT ioctl error)
// are caught and logged to stderr; the tile is simply omitted from the
// output.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "xrt/xrt_hw_context.h"

namespace test_utils {

class CycleCounterHelper {
public:
    static constexpr uint32_t PERF_CTRL0_OFFSET     = 0x00031500;
    static constexpr uint32_t PERF_COUNTER0_OFFSET  = 0x00031520;
    static constexpr uint8_t  EVENT_ACTIVE_CORE     = 0x1C;

    // NPU1 partition-relative cols: any context has start_col relative to
    // its own partition, so we probe cols 0..7 (covers the max partition
    // width on Phoenix). Rows are absolute: row 0 = shim, row 1 = memtile,
    // rows 2..5 = compute.
    static constexpr uint16_t COL_MIN = 0;
    static constexpr uint16_t COL_MAX = 7;
    static constexpr uint16_t ROW_MIN = 2;
    static constexpr uint16_t ROW_MAX = 5;

    explicit CycleCounterHelper(xrt::hw_context& ctx)
        : ctx_(ctx) {}

    // Configure perf counter 0 on every probeable compute tile:
    //   1. Read the control register, clear start/stop fields for cnt 0,
    //      set start = ACTIVE_CORE. Write back.
    //   2. Zero the counter value register.
    void start() {
        tiles_.clear();
        for (uint16_t col = COL_MIN; col <= COL_MAX; ++col) {
            for (uint16_t row = ROW_MIN; row <= ROW_MAX; ++row) {
                try {
                    uint32_t ctrl = ctx_.read_aie_reg(col, row, PERF_CTRL0_OFFSET);
                    // Counter 0 occupies bits [6:0] (start) and [14:8] (stop).
                    ctrl &= ~uint32_t(0x7F7F);
                    ctrl |= uint32_t(EVENT_ACTIVE_CORE);  // start only, stop = 0
                    ctx_.write_aie_reg(col, row, PERF_CTRL0_OFFSET, ctrl);
                    ctx_.write_aie_reg(col, row, PERF_COUNTER0_OFFSET, 0);
                    tiles_.emplace_back(col, row);
                } catch (const std::exception& e) {
                    // Tile not in partition, out-of-range col, or XRT ioctl
                    // error. Skip silently — many cols will be OOR for narrow
                    // partitions and that's expected.
                }
            }
        }
    }

    // Destructor writes the output file if XDNA_CYCLES_OUT is set.
    // No-op otherwise (e.g., when test.exe is run against real HW outside
    // the bridge script, or when the helper was never started).
    ~CycleCounterHelper() {
        const char* path = std::getenv("XDNA_CYCLES_OUT");
        if (!path || tiles_.empty()) return;

        std::ofstream out(path);
        if (!out) {
            std::cerr << "[cycle_counter] failed to open " << path << "\n";
            return;
        }
        out << "# col row cycles\n";
        for (auto [col, row] : tiles_) {
            try {
                uint32_t val = ctx_.read_aie_reg(col, row, PERF_COUNTER0_OFFSET);
                out << col << " " << row << " " << val << "\n";
            } catch (const std::exception& e) {
                std::cerr << "[cycle_counter] read failed at ("
                          << col << "," << row << "): " << e.what() << "\n";
            }
        }
    }

private:
    xrt::hw_context& ctx_;
    std::vector<std::pair<uint16_t, uint16_t>> tiles_;
};

} // namespace test_utils
```

- [ ] **Step 3: Include the helper from test_utils.h**

Add near the top of `/home/triple/npu-work/mlir-aie/runtime_lib/test_lib/include/test_utils.h` (after the existing XRT includes):

```cpp
#include "cycle_counter.h"
```

Verify the include path by searching for existing `#include` lines in `test_utils.h` and placing the new include with them.

- [ ] **Step 4: Commit**

```bash
cd /home/triple/npu-work/mlir-aie
git add runtime_lib/test_lib/include/cycle_counter.h \
        runtime_lib/test_lib/include/test_utils.h
git commit -m "test: add CycleCounterHelper for HW perf counter readout

Header-only RAII helper that programs AIE2 perf counter 0 with
start_event = ACTIVE_CORE on every compute tile. Destructor writes
tile,row,cycles lines to the path named by \$XDNA_CYCLES_OUT.

Uses the patched xrt::hw_context::{read,write}_aie_reg API
(xrt_hw_context.h), so no aie-rt linkage is required in test.exe.
The patch lives in xdna-driver/xrt/ on a parallel
xdna-emu-cycle-budget branch; upstream eventually.

Register offsets and event ID derived from aie-rt xaiemlgbl_params.h
and xaiemlgbl_reginit.c."
```

---

## Task 3: Phase B — Wire the helper into bridge test.cpp patches

**Files:**
- Modify: `/home/triple/npu-work/xdna-emu/scripts/emu-bridge-test.sh:1037-1048` (sed patching block in `compile_one`)
- Modify: `/home/triple/npu-work/xdna-emu/scripts/emu-bridge-test.sh:1141-1146` (HW run invocation — export `XDNA_CYCLES_OUT`)

- [ ] **Step 1: Extend the sed patch in compile_one to inject helper setup**

In `scripts/emu-bridge-test.sh`, locate the existing sed block at line 1043-1046 that patches `device_index` / `xrt::device(device_index)`. Extend it to also inject the helper instantiation right after the `xrt::hw_context` declaration:

Replace:

```bash
      sed \
        -e 's/unsigned int device_index = 0;/const char* _bdf = std::getenv("XRT_DEVICE_BDF");/' \
        -e 's/auto device = xrt::device(device_index);/auto device = _bdf ? xrt::device(std::string(_bdf)) : xrt::device(0);/' \
        "$src_dir/test.cpp" > "$build_dir/test.cpp"
```

with:

```bash
      sed \
        -e 's/unsigned int device_index = 0;/const char* _bdf = std::getenv("XRT_DEVICE_BDF");/' \
        -e 's/auto device = xrt::device(device_index);/auto device = _bdf ? xrt::device(std::string(_bdf)) : xrt::device(0);/' \
        -e '/xrt::hw_context context(device, xclbin.get_uuid());/a\
  test_utils::CycleCounterHelper _xdna_cyc(context); _xdna_cyc.start();' \
        "$src_dir/test.cpp" > "$build_dir/test.cpp"
```

The `a\` suffix appends the helper line *after* every `xrt::hw_context context(...)` declaration. Tests that use a different variable name or construction style won't match — those will silently skip the cycle counter injection and still produce a clean run. We accept that partial coverage; the majority of tests follow the common template.

- [ ] **Step 2: Export XDNA_CYCLES_OUT around the HW run**

In `run_one_hardware()` around line 1141-1146, add an export of `XDNA_CYCLES_OUT` alongside the existing `XRT_DEVICE_BDF` / `XDNA_TRACE_DIR` exports. Replace:

```bash
  (
    cd "$build_dir"
    export XRT_DEVICE_BDF="$bdf"
    export XDNA_TRACE_DIR="$trace_out_dir"
    timeout 30 bash -c "$run_cmd"
  ) > "$log_file" 2>&1 || rc=$?
```

with:

```bash
  local cycles_file="$build_dir/${safe}${vsuffix}.hw.cycles.txt"
  (
    cd "$build_dir"
    export XRT_DEVICE_BDF="$bdf"
    export XDNA_TRACE_DIR="$trace_out_dir"
    export XDNA_CYCLES_OUT="$cycles_file"
    timeout 30 bash -c "$run_cmd"
  ) > "$log_file" 2>&1 || rc=$?
```

(Path convention: `{build_dir}/{safe_name}{.variant}.hw.cycles.txt` lives next to test.exe and the xclbin, co-located with the build so it gets cached the same way.)

- [ ] **Step 3: Rebuild a single test.exe to verify the patch flow**

```bash
cd /home/triple/npu-work/xdna-emu
./scripts/emu-bridge-test.sh --compile --chess-only --no-hw add_one_using_dma
```

Expected: compile succeeds; `mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/test.cpp` contains `test_utils::CycleCounterHelper _xdna_cyc(context);` one line after the `xrt::hw_context` declaration.

- [ ] **Step 4: Inspect the patched test.cpp**

```bash
grep -A1 'xrt::hw_context context' /home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/test.cpp
```

Expected:
```
  xrt::hw_context context(device, xclbin.get_uuid());
  test_utils::CycleCounterHelper _xdna_cyc(context); _xdna_cyc.start();
```

- [ ] **Step 5: Commit**

```bash
cd /home/triple/npu-work/xdna-emu
git add scripts/emu-bridge-test.sh
git commit -m "bridge: inject CycleCounterHelper into test.cpp + export XDNA_CYCLES_OUT

Extends the existing compile-time sed patch to append a
CycleCounterHelper instantiation after xrt::hw_context construction.
run_one_hardware now exports XDNA_CYCLES_OUT so the helper's RAII
destructor writes {build_dir}/{name}.hw.cycles.txt on HW-path runs."
```

---

## Task 4: Phase B — Verify cycle capture on real hardware

This task produces no commit; it's a gate that Phase B produces sensible data before we build the comparison infrastructure on top.

- [ ] **Step 1: Run a single-compiler HW-path test**

```bash
cd /home/triple/npu-work/xdna-emu
./scripts/emu-bridge-test.sh --chess-only --no-emu add_one_using_dma
```

Expected: HW run succeeds (PASS), a cycles file exists at
`/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/add_one_using_dma.hw.cycles.txt`.

- [ ] **Step 2: Inspect the cycles file**

```bash
cat /home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/add_one_using_dma.hw.cycles.txt
```

Expected format:
```
# col row cycles
0 2 <nonzero>
```

One line per probed compute tile. For a single-column add_one test, expect one or two entries with small but nonzero cycles counts (hundreds to low thousands).

- [ ] **Step 3: Run one more test with known larger footprint**

```bash
./scripts/emu-bridge-test.sh --chess-only --no-emu matrix_multiplication_using_dma
```

Check `.hw.cycles.txt` — should have multiple tile entries with larger cycle counts than add_one.

- [ ] **Step 4: If step 3 shows zero or missing values, stop and investigate**

Possible causes:
- Helper include not found → check test_utils.h modification from Task 2
- Reg write silently fails → add a `std::cerr` diagnostic in the helper's start() catch block
- ACTIVE_CORE event ID is different → cross-check against aie-rt's `xaiemlgbl_reginit.c` event enum

Report findings before proceeding.

---

## Task 5: Phase A.1 — FFI halt_reason enum + unbounded loop fix

**Files:**
- Modify: `/home/triple/npu-work/xdna-emu/crates/xdna-emu-ffi/src/lib.rs:80-99`
- Modify: `/home/triple/npu-work/xdna-emu/crates/xdna-emu-ffi/src/execution.rs:70-182`
- Create: `/home/triple/npu-work/xdna-emu/crates/xdna-emu-ffi/tests/max_cycles.rs`

- [ ] **Step 1: Write the failing integration test**

Create `/home/triple/npu-work/xdna-emu/crates/xdna-emu-ffi/tests/max_cycles.rs`:

```rust
// SPDX-License-Identifier: MIT
//
// Integration tests for xdna_emu_set_max_cycles + halt_reason.

use xdna_emu_ffi::{
    xdna_emu_create, xdna_emu_destroy, xdna_emu_run, xdna_emu_set_max_cycles,
    XdnaEmuHaltReason, XdnaEmuResult,
};

#[test]
fn max_cycles_zero_is_unbounded() {
    unsafe {
        let h = xdna_emu_create();
        assert!(!h.is_null());
        xdna_emu_set_max_cycles(h, 0);
        let status = xdna_emu_run(h);
        assert_eq!(status.result, XdnaEmuResult::Success);
        // No xclbin loaded -> no cores enabled -> engine halts immediately,
        // so we just expect Completed, not Budget.
        assert_eq!(status.halt_reason, XdnaEmuHaltReason::Completed);
        xdna_emu_destroy(h);
    }
}

#[test]
fn max_cycles_one_hits_budget() {
    // TODO(plan-phase): requires a loaded xclbin with an enabled core;
    // currently no-op without a fixture. Left as a placeholder integration
    // test; the unit-level guarantees are enforced via the loop condition
    // in execution.rs.
}
```

- [ ] **Step 2: Run the test to confirm it fails to compile**

```bash
cd /home/triple/npu-work/xdna-emu
TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --test max_cycles max_cycles_zero_is_unbounded
```

Expected: compile failure — `XdnaEmuHaltReason` undefined, `halt_reason` field not on `XdnaEmuExecStatus`.

- [ ] **Step 3: Add the halt_reason enum and field**

Edit `/home/triple/npu-work/xdna-emu/crates/xdna-emu-ffi/src/lib.rs`. Replace the `XdnaEmuExecStatus` block (lines 92-99) with:

```rust
/// Why the emulator stopped running.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XdnaEmuHaltReason {
    /// Kernel ran to natural completion (cores halted, syncs satisfied).
    Completed = 0,
    /// Cycle budget (`max_cycles`) reached before natural completion.
    Budget = 1,
    /// Error during execution (FFI fault, executor error).
    Error = 2,
}

/// Execution status returned by run functions.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct XdnaEmuExecStatus {
    pub result: XdnaEmuResult,
    pub cycles_executed: u64,
    pub halted: bool,
    pub halt_reason: XdnaEmuHaltReason,
}
```

Keep the `halted` bool — downstream callers may depend on it, and the new field adds information without subtracting.

- [ ] **Step 4: Fix the run loop to treat max=0 as unbounded and populate halt_reason**

Edit `/home/triple/npu-work/xdna-emu/crates/xdna-emu-ffi/src/execution.rs`. Replace lines 72-182 (the entire `xdna_emu_run` function body) with:

```rust
pub unsafe extern "C" fn xdna_emu_run(handle: *mut XdnaEmuHandle) -> XdnaEmuExecStatus {
    use xdna_emu::interpreter::engine::EngineStatus;

    if handle.is_null() {
        return XdnaEmuExecStatus {
            result: XdnaEmuResult::InvalidHandle,
            cycles_executed: 0,
            halted: false,
            halt_reason: XdnaEmuHaltReason::Error,
        };
    }

    let handle = &mut *handle;

    let mut cycles = 0u64;
    let max = handle.max_cycles;
    let unbounded = max == 0;

    log::info!(
        "Running emulator (max {})",
        if unbounded { "unbounded".to_string() } else { max.to_string() }
    );

    // Warm-up: let cores reach their first blocking point before NPU
    // instructions arrive. See execution.rs history for rationale.
    if handle.engine.enabled_cores() > 0 && !handle.npu_executor.is_done() {
        const MAX_WARMUP: u64 = 100_000;
        while cycles < MAX_WARMUP {
            handle.engine.step();
            cycles += 1;
            if handle.engine.all_cores_blocked() {
                break;
            }
        }
        log::info!("Core warm-up: {} cycles (all cores at first blocking point)", cycles);
    }

    let mut natural_halt = false;

    while unbounded || cycles < max {
        // Advance NPU instruction execution.
        let npu_progressed;
        {
            let (device, host_mem) = handle.engine.device_and_host_memory();
            let result = handle.npu_executor.try_advance(device, host_mem);
            if let xdna_emu::npu::AdvanceResult::Error(msg) = result {
                log::error!("NPU executor fatal: {}", msg);
                return XdnaEmuExecStatus {
                    result: XdnaEmuResult::ExecutionError,
                    cycles_executed: cycles,
                    halted: false,
                    halt_reason: XdnaEmuHaltReason::Error,
                };
            }
            npu_progressed = matches!(result, xdna_emu::npu::AdvanceResult::Progressed);
        }
        if npu_progressed {
            handle.engine.flush_ctrl_packets();
        }

        handle.engine.step();
        cycles += 1;

        if handle.engine.status() == EngineStatus::Halted {
            let executor_pending = !handle.npu_executor.is_done()
                || !handle.npu_executor.syncs_satisfied(handle.engine.device());
            if executor_pending || handle.engine.device().array.any_dma_active() {
                handle.engine.force_running();
            } else {
                log::info!("Cores halted after {} cycles", cycles);
                natural_halt = true;
                break;
            }
        }

        if handle.engine.status() == EngineStatus::Stalled {
            log::warn!("Stall detected after {} cycles: no monotonic progress", cycles);
            natural_halt = true;
            break;
        }

        if handle.npu_executor.is_done()
            && handle.npu_executor.syncs_satisfied(handle.engine.device())
        {
            log::info!("All DMA syncs satisfied after {} cycles", cycles);
            natural_halt = true;
            break;
        }
    }

    handle.engine.flush_trace_to_host();

    let halted = handle.engine.status() == EngineStatus::Halted
        || (handle.npu_executor.is_done()
            && handle.npu_executor.syncs_satisfied(handle.engine.device()));

    // If we fell out of the while-loop with cycles >= max and didn't halt
    // naturally, budget was hit.
    let halt_reason = if natural_halt || halted {
        XdnaEmuHaltReason::Completed
    } else {
        XdnaEmuHaltReason::Budget
    };

    XdnaEmuExecStatus {
        result: XdnaEmuResult::Success,
        cycles_executed: cycles,
        halted,
        halt_reason,
    }
}
```

Re-add the `use XdnaEmuHaltReason` import near the top of `execution.rs`:

```rust
use super::{XdnaEmuHandle, XdnaEmuResult, XdnaEmuExecStatus, XdnaEmuHaltReason};
```

- [ ] **Step 5: Run the test**

```bash
cd /home/triple/npu-work/xdna-emu
TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --test max_cycles max_cycles_zero_is_unbounded
```

Expected: PASS.

- [ ] **Step 6: Run the existing Rust test suite**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib
```

Expected: all existing tests still pass. Any regressions indicate that downstream callers are reading `XdnaEmuExecStatus` by field order and have broken — that's a plan-writer bug, fix by reverting to prepending the new field or investigate consumers.

- [ ] **Step 7: Rebuild the FFI library for the plugin**

```bash
cargo build -p xdna-emu-ffi
```

Expected: `target/debug/libxdna_emu.so` updated.

- [ ] **Step 8: Commit**

```bash
git add crates/xdna-emu-ffi/src/lib.rs \
        crates/xdna-emu-ffi/src/execution.rs \
        crates/xdna-emu-ffi/tests/max_cycles.rs
git commit -m "ffi: add XdnaEmuHaltReason enum + treat max_cycles=0 as unbounded

XdnaEmuExecStatus now carries an explicit halt_reason (Completed,
Budget, Error) in addition to the existing halted: bool. Run loop
condition is 'unbounded || cycles < max' so set_max_cycles(0) is
explicit unbounded mode."
```

---

## Task 6: Phase A.2 — Plugin env-var reader + XDNA_EMU_STATUS emit

**Files:**
- Modify: `/home/triple/npu-work/xdna-emu/xrt-plugin/src/transport_inprocess.cpp`

- [ ] **Step 1: Locate the plugin's run invocation**

```bash
grep -n 'sym_run_\|sym_set_max_cycles_' /home/triple/npu-work/xdna-emu/xrt-plugin/src/transport_inprocess.cpp
```

Expected: `sym_run_` is resolved at line 101 but only called from a run method elsewhere in the file. Locate the method that calls `sym_run_` and wrap it.

- [ ] **Step 2: Read the run invocation to understand its current shape**

```bash
grep -n 'sym_run_(' /home/triple/npu-work/xdna-emu/xrt-plugin/src/transport_inprocess.cpp
```

Note the line number; read the function it's in (±20 lines around the call) with the `Read` tool.

- [ ] **Step 3: Add env-var read to constructor**

At the end of the constructor (after line 130, just before the closing `}`), insert:

```cpp
    // Apply cycle budget from XDNA_EMU_MAX_CYCLES (0 or unset = unbounded).
    if (const char* env = std::getenv("XDNA_EMU_MAX_CYCLES")) {
        char* end = nullptr;
        uint64_t val = std::strtoull(env, &end, 10);
        if (end == env || *end != '\0') {
            std::cerr << "[xdna-emu] XDNA_EMU_MAX_CYCLES='" << env
                      << "' unparseable; ignoring\n";
        } else {
            max_cycles_budget_ = val;
            sym_set_max_cycles_(emu_, val);
        }
    }
```

Add the member to the class (in the header, `xrt-plugin/src/transport_inprocess.h` or wherever `emu_transport_inprocess` is declared):

```cpp
    uint64_t max_cycles_budget_ = 0;
```

Add the header include if not present:

```cpp
#include <cstdlib>
#include <cstring>
```

- [ ] **Step 4: Emit XDNA_EMU_STATUS after run**

Find the method that calls `sym_run_()`. Immediately after it, before returning, add:

```cpp
    XdnaEmuExecStatus status = sym_run_(emu_);

    // Emit a parseable status line for the bridge script.
    const char* reason_str = "error";
    switch (status.halt_reason) {
        case XDNA_EMU_HALT_COMPLETED: reason_str = "completed"; break;
        case XDNA_EMU_HALT_BUDGET:    reason_str = "budget";    break;
        case XDNA_EMU_HALT_ERROR:     reason_str = "error";     break;
    }
    std::cerr << "XDNA_EMU_STATUS: halt_reason=" << reason_str
              << " cycles=" << status.cycles_executed
              << " max_cycles=" << max_cycles_budget_ << "\n";
```

(Note: the enum names in C land depend on how the FFI crate exports them. If the generated C header names them differently — e.g., `XdnaEmuHaltReason_Completed` — adjust accordingly. Run the next step to see what's actually generated.)

- [ ] **Step 5: Check the generated C header for the halt_reason enum names**

```bash
find /home/triple/npu-work/xdna-emu -name 'xdna_emu_ffi.h' -o -name 'xdna_emu.h' 2>/dev/null
```

If a generated header exists, grep for `halt_reason` and `HaltReason` to see the exact C identifier spellings. Fix the `switch` arms in Step 4 if needed.

If no header is auto-generated, the plugin's own C++ wrapper likely declares the enum manually. Search `xrt-plugin/` for any existing enum or struct declarations matching `XdnaEmuExecStatus` and update them alongside:

```bash
grep -rn 'XdnaEmuExecStatus\|halt_reason' /home/triple/npu-work/xdna-emu/xrt-plugin/
```

Add the enum declaration to the plugin's FFI shim header if missing:

```cpp
enum XdnaEmuHaltReason : int {
    XDNA_EMU_HALT_COMPLETED = 0,
    XDNA_EMU_HALT_BUDGET    = 1,
    XDNA_EMU_HALT_ERROR     = 2,
};

struct XdnaEmuExecStatus {
    int       result;
    uint64_t  cycles_executed;
    bool      halted;
    XdnaEmuHaltReason halt_reason;
};
```

Keep field order identical to the Rust `#[repr(C)] struct` (Task 5 step 3).

- [ ] **Step 6: Build the plugin**

```bash
cd /home/triple/npu-work/xdna-emu
./scripts/rebuild-plugin.sh 2>&1 | tee /tmp/claude-1000/plugin-rebuild.log
```

Expected: "Plugin build: OK" at the end. If build fails, inspect `/tmp/claude-1000/plugin-rebuild.log`.

- [ ] **Step 7: Smoke-test the plugin with no budget**

```bash
cd /home/triple/npu-work/xdna-emu
./scripts/emu-bridge-test.sh --chess-only --no-hw add_one_using_dma 2>&1 | tee /tmp/claude-1000/smoke.log
grep -A1 XDNA_EMU_STATUS /home/triple/npu-work/xdna-emu/build/bridge-test-results/latest/*.bridge.log | head
```

Expected: the bridge log for add_one_using_dma contains a line of the form
`XDNA_EMU_STATUS: halt_reason=completed cycles=<N> max_cycles=0`.

- [ ] **Step 8: Smoke-test the plugin with a tiny budget**

```bash
XDNA_EMU_MAX_CYCLES=10 ./scripts/emu-bridge-test.sh --chess-only --no-hw add_one_using_dma
grep -A1 XDNA_EMU_STATUS /home/triple/npu-work/xdna-emu/build/bridge-test-results/latest/*.bridge.log | head
```

Expected: `XDNA_EMU_STATUS: halt_reason=budget cycles=10 max_cycles=10` (or cycles slightly over 10 depending on bundle boundary behavior; budget is what matters).

- [ ] **Step 9: Commit**

```bash
git add xrt-plugin/
git commit -m "plugin: wire XDNA_EMU_MAX_CYCLES env var and emit status line

emu_transport_inprocess now reads XDNA_EMU_MAX_CYCLES on
construction and calls xdna_emu_set_max_cycles. After each
sym_run_() invocation it prints 'XDNA_EMU_STATUS: halt_reason=...
cycles=... max_cycles=...' to stderr so the bridge script can
classify BUDGET vs completed runs."
```

---

## Task 7: Phase A.3 — Bridge script status parsing + timeout relax + --no-timeout flag

**Files:**
- Modify: `/home/triple/npu-work/xdna-emu/scripts/emu-bridge-test.sh`

- [ ] **Step 1: Add --no-timeout option to CLI parser**

In the `while [[ $# -gt 0 ]]; do case "$1" in` block (around line 96-148), add a new case before `--help`:

```bash
    --no-timeout)  NO_TIMEOUT=true; shift ;;
```

And near the top (around line 91) initialize the flag:

```bash
NO_TIMEOUT=false
```

Add a line to the `--help` USAGE block listing it:

```
  --no-timeout    Run EMU without wall-clock timeout (use for very long runs)
```

- [ ] **Step 2: Relax the EMU timeout and honor --no-timeout**

In `run_one_bridge` (around line 1245-1255), replace:

```bash
    timeout 120 bash -c "$run_cmd"
```

with:

```bash
    if [[ "${NO_TIMEOUT:-false}" == "true" ]]; then
      bash -c "$run_cmd"
    else
      timeout 600 bash -c "$run_cmd"
    fi
```

Export `NO_TIMEOUT` alongside the other exports (around line 168) so parallel subshells see it:

```bash
export NO_TIMEOUT
```

- [ ] **Step 3: Parse XDNA_EMU_STATUS and add BUDGET result classification**

In `run_one_bridge` after the existing `result=` classification block (around line 1256-1263), insert:

```bash
  # Override with BUDGET if the plugin reported budget-exceeded.
  local status_line
  status_line="$(grep 'XDNA_EMU_STATUS:' "$log_file" | tail -1 || true)"
  if [[ -n "$status_line" ]]; then
    local hr
    hr="$(echo "$status_line" | grep -oP 'halt_reason=\K\w+' || true)"
    if [[ "$hr" == "budget" ]]; then
      result="BUDGET"
    fi
  fi
```

Place this after `result` is first set by the rc + grep PASS logic, but before the EMU_MISS check — so BUDGET overrides FAIL/TIMEOUT, but EMU_MISS still takes priority as it indicates infrastructure failure.

- [ ] **Step 4: Update print_report to recognize BUDGET**

In `print_report`, around the `case "$br"` block (line ~1494-1505), add a case for BUDGET:

```bash
        BUDGET)   bridge_budget[$compiler]=$(( ${bridge_budget[$compiler]} + 1 ))
                  bridge_fail[$compiler]=$(( ${bridge_fail[$compiler]} + 1 )) ;;
```

And initialize the counter in the counter-declarations block above (around line 1376-1396):

```bash
    bridge_budget[$compiler]=0
```

In the Summary block (around line 1587), add:

```bash
    if [[ ${bridge_budget[$compiler]} -gt 0 ]]; then
      echo "  (${bridge_budget[$compiler]} BUDGET)"
    fi
```

- [ ] **Step 5: Smoke-test the classification**

```bash
cd /home/triple/npu-work/xdna-emu
XDNA_EMU_MAX_CYCLES=10 ./scripts/emu-bridge-test.sh --chess-only --no-hw add_one_using_dma
```

Expected: the results table shows `Chess/EMU: BUDGET` for add_one_using_dma. The summary reports `(1 BUDGET)`.

- [ ] **Step 6: Smoke-test --no-timeout doesn't break normal runs**

```bash
./scripts/emu-bridge-test.sh --no-timeout --chess-only --no-hw add_one_using_dma
```

Expected: normal PASS result. Run time matches the default (no hang).

- [ ] **Step 7: Commit**

```bash
git add scripts/emu-bridge-test.sh
git commit -m "bridge: parse XDNA_EMU_STATUS and add BUDGET result class

Bridge now greps each EMU log for the plugin's status line and
reclassifies runs with halt_reason=budget as BUDGET (distinct from
FAIL/TIMEOUT). Wall-clock timeout relaxed from 120s to 600s; new
--no-timeout flag disables the wall-clock cap for long runs."
```

---

## Task 8: Phase D.1 — Audit doc for level-event perf counter semantics

**Files:**
- Create: `/home/triple/npu-work/xdna-emu/docs/arch/perfcnt-level-events.md`

- [ ] **Step 1: Write the audit**

Create the file with this content:

```markdown
# Performance Counter Level-Event Semantics Audit

## Problem

HW AIE2 perf counters configured with `XAIE_EVENT_ACTIVE_CORE` as start
event tick only during cycles when the core is in Execute state. This is
a *level* signal, not a pulse — the counter advances while the signal is
asserted, not on signal transitions.

EMU's current `PerfCounterBank` (`src/device/perf_counters/mod.rs:326-383`)
implements a pulse model: `handle_event(start)` transitions the counter
to `Active`, and `tick()` (called unconditionally every cycle from
`coordinator.rs:964`) increments every `Active` counter regardless of
whether the core is still active.

Consequence: once started, an ACTIVE_CORE-configured EMU counter ticks
every cycle until stopped, even when the core is stalled on lock-wait,
cascade-wait, or disabled. The counter overcounts.

## Which events are affected

Level-valued events reachable from EMU state, ordered by how close their
semantics are to "tick only while X":

| Event | Level semantic | EMU state source |
|-------|----------------|------------------|
| `ACTIVE_CORE` | Core in Execute state | `Core::is_running()` / `!is_blocked()` |
| `ACTIVE_MEMORY_STALL` | Core blocked on memory-bank contention | Not modeled in EMU (no bank contention) |
| `ACTIVE_LOCK_STALL` | Core blocked on lock acquire | `Core::is_waiting_on_lock()` |
| `ACTIVE_CASCADE_STALL` | Core blocked on cascade full/empty | `Core::is_waiting_on_cascade()` |
| `DISABLED_CORE` | Core is disabled | `!Core::is_enabled()` |

`ACTIVE_MEMORY_STALL` has no EMU backing — skip. The others map to
existing blocked-state checks.

## Pulse vs level today

`handle_event()` is called for pulse events — `LOCK_ACQUIRE`,
`PORT_RUNNING`, `BRANCH_TAKEN`, etc. — and correctly transitions
`Idle`/`Stopped` <-> `Active`. That path stays as-is for pulse events.

The fix targets only the `tick()` path: rather than ticking every
`Active` counter unconditionally, a counter with a level-valued
start_event should tick only when that level is asserted *this cycle*.

## Fix: Option 2 (selected)

Decision (from spec open questions): **Option 2 — move the tick gate
into the caller.** Instead of changing `tick()`'s signature, the
coordinator checks core state before calling `tick()`:

- Core is in Execute state → `core_perf_counters.tick_active_cycles()`
- Core is blocked/disabled → `core_perf_counters.tick_idle_cycles()`

`tick_active_cycles()` is the current behavior: increment all Active
counters, return threshold fires.

`tick_idle_cycles()` is new: increment only counters whose `start_event`
is NOT a level-valued event (i.e., preserves pulse-counter behavior for
non-ACTIVE_CORE counters that happen to be running on the core module),
but does not increment counters started by `ACTIVE_CORE`. Threshold
checks still apply.

This is the narrowest possible change: zero diff to `handle_event()`,
zero diff to pulse-event counters, no new plumbing.

## Out-of-scope for this work

- Stall events (`ACTIVE_LOCK_STALL`, etc.) as start events. Spec says:
  "If fixing these falls out of the same refactor (same plumbing, same
  predicate), include them. If any needs significant separate plumbing,
  defer to a follow-up." For Option 2, adding stall-event tick gating
  is symmetric work: the coordinator would need to query `is_waiting_on_lock()`
  etc. and conditionally tick a different subset. Leave as a follow-up.
- DMA delay modeling — orthogonal.
- NoC latency — orthogonal.
```

- [ ] **Step 2: Commit**

```bash
cd /home/triple/npu-work/xdna-emu
git add docs/arch/perfcnt-level-events.md
git commit -m "docs: audit of level-event perf counter semantics (Phase D.1)

Documents the drift between HW ACTIVE_CORE (level) and EMU's
current pulse-based PerfCounterBank. Selects Option 2 (move tick
gate into the coordinator caller) as the narrowest fix. Stall
events left as a follow-up."
```

---

## Task 9: Phase D.2 — Refactor PerfCounterBank.tick() for level-event semantics

**Files:**
- Modify: `/home/triple/npu-work/xdna-emu/src/device/perf_counters/mod.rs`
- Modify: `/home/triple/npu-work/xdna-emu/src/device/perf_counters/tests.rs`
- Modify: `/home/triple/npu-work/xdna-emu/src/interpreter/engine/coordinator.rs:961-966`

- [ ] **Step 1: Write the failing unit test**

Add to `/home/triple/npu-work/xdna-emu/src/device/perf_counters/tests.rs` (near the bottom, before any trailing `}`):

```rust
/// ACTIVE_CORE event ID on AIE2 (per aie-rt xaiemlgbl_reginit.c).
const EVENT_ACTIVE_CORE: u8 = 0x1C;

#[test]
fn active_core_ticks_only_while_core_active() {
    let mut bank = PerfCounterBank::new(4);
    bank.write_control_start_stop(
        EVENT_ACTIVE_CORE as u32,  // counter 0 start
        0, 1, 7,
    );
    // Fire start event -> counter 0 becomes Active.
    bank.handle_event(EVENT_ACTIVE_CORE);
    assert!(bank.is_active(0));

    // 10 active cycles.
    for _ in 0..10 {
        bank.tick_active_cycles();
    }
    assert_eq!(bank.read_counter(0), 10);

    // 5 idle cycles (core stalled / blocked / disabled).
    for _ in 0..5 {
        bank.tick_idle_cycles();
    }
    assert_eq!(bank.read_counter(0), 10, "counter must not tick while idle");

    // 10 more active cycles.
    for _ in 0..10 {
        bank.tick_active_cycles();
    }
    assert_eq!(bank.read_counter(0), 20);
}

#[test]
fn pulse_event_counter_ticks_in_both_states() {
    // A counter started by a non-ACTIVE_CORE event (e.g., LOCK_ACQUIRE = 0x2E)
    // should behave as before: tick whenever Active, regardless of core state.
    const EVENT_LOCK_ACQUIRE: u8 = 0x2E;
    let mut bank = PerfCounterBank::new(4);
    bank.write_control_start_stop(EVENT_LOCK_ACQUIRE as u32, 0, 1, 7);
    bank.handle_event(EVENT_LOCK_ACQUIRE);

    // tick_idle_cycles should still increment pulse-started counters.
    for _ in 0..10 {
        bank.tick_idle_cycles();
    }
    assert_eq!(bank.read_counter(0), 10);
}
```

- [ ] **Step 2: Run the failing test**

```bash
cd /home/triple/npu-work/xdna-emu
TMPDIR=/tmp/claude-1000 cargo test --lib active_core_ticks_only_while_core_active
```

Expected: compile failure — `tick_active_cycles` / `tick_idle_cycles` don't exist.

- [ ] **Step 3: Refactor tick() into two variants**

In `/home/triple/npu-work/xdna-emu/src/device/perf_counters/mod.rs`, replace the existing `tick()` method (lines 365-383) with:

```rust
    /// ACTIVE_CORE event ID on AIE2 (per aie-rt xaiemlgbl_reginit.c).
    /// A counter whose start_event is this value is level-gated on the core
    /// Execute state.
    const EVENT_ACTIVE_CORE: u8 = 0x1C;

    /// Advance counters for a cycle during which the core is in Execute state.
    ///
    /// Returns a vector of counter indices that reached their event value
    /// threshold this cycle.
    pub fn tick_active_cycles(&mut self) -> Vec<usize> {
        self.tick_internal(true)
    }

    /// Advance counters for a cycle during which the core is NOT in Execute
    /// state (stalled, blocked, disabled).
    ///
    /// Counters whose start_event is level-valued (currently just
    /// ACTIVE_CORE) do not tick. Other active counters still tick — they're
    /// pulse-started and remain driven by their pulse-event semantics.
    pub fn tick_idle_cycles(&mut self) -> Vec<usize> {
        self.tick_internal(false)
    }

    fn tick_internal(&mut self, core_active: bool) -> Vec<usize> {
        let mut threshold_events = Vec::new();
        for i in 0..self.num_counters {
            if self.state[i] != CounterState::Active {
                continue;
            }
            if self.start_event[i] == Self::EVENT_ACTIVE_CORE && !core_active {
                continue; // level-gated, core not asserting
            }
            self.counter_value[i] = self.counter_value[i].wrapping_add(1);
            if self.event_value[i] != 0
                && self.counter_value[i] == self.event_value[i]
            {
                threshold_events.push(i);
            }
        }
        threshold_events
    }

    /// Back-compat shim: old call site. Equivalent to `tick_active_cycles()`.
    /// Prefer the explicit variants at new call sites.
    #[deprecated(note = "use tick_active_cycles or tick_idle_cycles")]
    pub fn tick(&mut self) -> Vec<usize> {
        self.tick_active_cycles()
    }
```

- [ ] **Step 4: Run the tests again**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib active_core_ticks_only_while_core_active pulse_event_counter_ticks_in_both_states
```

Expected: both PASS.

- [ ] **Step 5: Update the coordinator to call the right variant per cycle**

In `/home/triple/npu-work/xdna-emu/src/interpreter/engine/coordinator.rs` around line 961-966, replace:

```rust
        for tile in &mut self.device.array.tiles {
            tile.core_timer.tick();
            tile.mem_timer.tick();
            tile.core_perf_counters.tick();
            tile.mem_perf_counters.tick();
        }
```

with:

```rust
        for tile in &mut self.device.array.tiles {
            tile.core_timer.tick();
            tile.mem_timer.tick();
            let core_active = tile.core.is_running_this_cycle();
            if core_active {
                tile.core_perf_counters.tick_active_cycles();
            } else {
                tile.core_perf_counters.tick_idle_cycles();
            }
            // Memory module perf counters aren't level-gated on core state;
            // keep existing semantics.
            tile.mem_perf_counters.tick_active_cycles();
        }
```

If `tile.core.is_running_this_cycle()` doesn't exist in the current API, grep for the nearest existing predicate:

```bash
grep -n 'fn is_running\|fn is_enabled\|fn is_blocked\|enabled_cores\|EngineStatus' /home/triple/npu-work/xdna-emu/src/device/tile.rs /home/triple/npu-work/xdna-emu/src/interpreter/engine/mod.rs 2>/dev/null | head
```

Use whichever boolean most closely mirrors "core advanced an instruction this cycle." The executing-plans worker may need to add a thin accessor on `Core` or `Tile` if none fits. Name it `is_running_this_cycle()`.

- [ ] **Step 6: Run the full lib test suite to catch any regressions**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1 | tee /tmp/claude-1000/d2-lib-tests.log
```

Expected: all prior tests still pass. Watch for perf-counter-related tests that may have been relying on the old tick-every-cycle semantic; those need review.

If any test fails and the failure is "counter was supposed to tick but didn't," that's real signal — a test assumed the old unconditional tick. Update the test to use `tick_active_cycles()` and call it once per active cycle it was simulating.

- [ ] **Step 7: Rebuild the FFI**

```bash
cargo build -p xdna-emu-ffi
```

- [ ] **Step 8: Commit**

```bash
git add src/device/perf_counters/mod.rs \
        src/device/perf_counters/tests.rs \
        src/interpreter/engine/coordinator.rs
git commit -m "perf_counters: gate ACTIVE_CORE ticks on core Execute state (Phase D.2)

Splits tick() into tick_active_cycles() and tick_idle_cycles().
Counters started by ACTIVE_CORE (event 0x1C) only advance during
cycles when the core is running an instruction; pulse-started
counters tick in both. Coordinator picks the variant per tile per
cycle based on core state. Old tick() kept as a deprecated shim."
```

---

## Task 10: Phase D.3 — HW integration spot-check

No commit unless a bug is found. This gate confirms that the EMU counter values are now close enough to HW to proceed with Phase C's comparison gate.

- [ ] **Step 1: Pick two or three simple tests with known HW cycles files**

After Phase B landed, every HW run produces `{name}.hw.cycles.txt`. Pick 2-3 small tests:

```bash
ls /home/triple/npu-work/mlir-aie/build/test/npu-xrt/*/chess/*.hw.cycles.txt | head
```

- [ ] **Step 2: Run each test through EMU and read emulator-side counters**

No native "dump counter values after run" path exists yet — easiest approach: add a temporary `log::info!` line in `execution.rs` just before the final `XdnaEmuExecStatus` return that dumps counter 0 for each tile, then run via the bridge:

```bash
XDNA_EMU_LOG_LEVEL=info ./scripts/emu-bridge-test.sh --chess-only --no-hw add_one_using_dma
```

Look for the dump line in the bridge log. Alternatively, write a small standalone binary (`src/bin/dump-perfctr.rs`) that loads an xclbin via the ffi and prints counter values — heavier but reusable for later work.

Either way, compare EMU counter values to the HW cycles file for the same tiles.

- [ ] **Step 3: Expected outcome**

EMU ≈ HW in magnitude (same order; ratio within a small constant factor). If a test shows EMU counter 10× higher than HW, that's a real cycle-modeling drift — flag it but don't fix in this task (it's Phase C's "show-cycle-drift.sh" use case). Proceed to Phase C regardless.

If a test shows EMU counter 0 while HW shows meaningful cycles, Phase D did not propagate correctly — investigate before moving on.

- [ ] **Step 4: Remove the temporary log line if used**

If Step 2 added a debug log, revert it before moving on:

```bash
git checkout -- crates/xdna-emu-ffi/src/execution.rs
```

---

## Task 11: Phase C.1 — Bridge script parses hw.cycles.txt → hw_max

**Files:**
- Modify: `/home/triple/npu-work/xdna-emu/scripts/emu-bridge-test.sh`

- [ ] **Step 1: Add parser function near the other bash helpers**

After `sanitize_name()` (around line 614), add:

```bash
# Parse a HW cycles file produced by the Phase B CycleCounterHelper.
# File format: comment lines starting with '#', otherwise "col row cycles".
# Returns the maximum cycles value across all tiles, or empty if no file.
parse_hw_cycles_max() {
  local file="$1"
  [[ -f "$file" ]] || return 0
  awk '
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*$/ { next }
    { if ($3 > max) max = $3 }
    END { print max }
  ' "$file"
}
export -f parse_hw_cycles_max
```

- [ ] **Step 2: Smoke-test the parser**

```bash
cd /home/triple/npu-work/xdna-emu
source scripts/emu-bridge-test.sh 2>/dev/null  # will fail but funcs load
# If sourcing doesn't work due to early exits, test by invoking in a sub-shell:
bash -c "$(declare -f parse_hw_cycles_max); parse_hw_cycles_max /home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/add_one_using_dma.hw.cycles.txt"
```

Expected: a single number on stdout matching the largest cycle value in the file.

- [ ] **Step 3: Commit**

```bash
git add scripts/emu-bridge-test.sh
git commit -m "bridge: add parse_hw_cycles_max helper (Phase C.1)

Reads Phase B's {t}.hw.cycles.txt files and returns the max
cycles across compute tiles. Foundation for the per-test cycle
budget computed in Phase C.3."
```

---

## Task 12: Phase C.2 — Override file + loader

**Files:**
- Create: `/home/triple/npu-work/xdna-emu/scripts/cycle-budget-overrides.txt`
- Modify: `/home/triple/npu-work/xdna-emu/scripts/emu-bridge-test.sh`

- [ ] **Step 1: Create the empty override file**

`/home/triple/npu-work/xdna-emu/scripts/cycle-budget-overrides.txt`:

```
# Cycle-budget per-test overrides for emu-bridge-test.sh.
#
# Format (one per line, whitespace-separated):
#   test-name   multiplier   reason
#
# Multiplier is an integer >= 1 replacing the default 3x tolerance.
# 'reason' runs to end of line (may contain spaces).
#
# Every override MUST include a reason. No exceptions -- the override
# file is where perf knowledge lives, and a line without context is
# opaque to reviewers.
#
# Example:
#   dense_matmul_large   5   high cross-crate PRMX traffic; 3x too tight as of 2026-04
```

- [ ] **Step 2: Add loader function**

After `parse_hw_cycles_max` (from Task 11), add:

```bash
# Look up the override multiplier for a test. Prints the multiplier
# (integer), or empty if no override. Returns 0 always.
lookup_cycle_override() {
  local test_name="$1"
  local override_file="${SCRIPT_DIR}/cycle-budget-overrides.txt"
  [[ -f "$override_file" ]] || return 0
  awk -v name="$test_name" '
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*$/ { next }
    $1 == name { print $2; exit }
  ' "$override_file"
}
export -f lookup_cycle_override
```

- [ ] **Step 3: Smoke-test the loader**

```bash
echo "add_one_using_dma 5 test-reason" >> scripts/cycle-budget-overrides.txt
bash -c "$(declare -f lookup_cycle_override); SCRIPT_DIR=/home/triple/npu-work/xdna-emu/scripts lookup_cycle_override add_one_using_dma"
```

Expected: `5` on stdout.

Undo the test entry:

```bash
git checkout -- scripts/cycle-budget-overrides.txt
```

- [ ] **Step 4: Commit**

```bash
git add scripts/cycle-budget-overrides.txt scripts/emu-bridge-test.sh
git commit -m "bridge: add cycle-budget override file + loader (Phase C.2)

Empty override file at scripts/cycle-budget-overrides.txt with
header documenting the format. lookup_cycle_override() returns
the multiplier for a named test (empty = no override)."
```

---

## Task 13: Phase C.3 — Budget calculation + env var export

**Files:**
- Modify: `/home/triple/npu-work/xdna-emu/scripts/emu-bridge-test.sh`

- [ ] **Step 1: Compute and export budget in run_one_bridge**

In `run_one_bridge` (around line 1245-1255), just before the EMU invocation subshell, add:

```bash
  # Compute cycle budget from Phase B HW cycles file (if present).
  local hw_cycles_file="$BUILD_BASE/$name/$compiler/${safe}${vsuffix}.hw.cycles.txt"
  local hw_max
  hw_max="$(parse_hw_cycles_max "$hw_cycles_file")"
  local budget=0   # 0 = unbounded
  if [[ -n "$hw_max" ]] && [[ "$hw_max" -gt 0 ]]; then
    local mult
    mult="$(lookup_cycle_override "$name")"
    [[ -z "$mult" ]] && mult=3
    budget=$(( hw_max * mult ))
  fi
```

Then in the subshell block, add the export alongside the existing ones:

```bash
    export XDNA_EMU_MAX_CYCLES="$budget"
```

- [ ] **Step 2: Smoke-test with a test that has a HW cycles file**

```bash
cd /home/triple/npu-work/xdna-emu
./scripts/emu-bridge-test.sh --chess-only add_one_using_dma
grep 'XDNA_EMU_STATUS\|XDNA_EMU_MAX_CYCLES' /home/triple/npu-work/xdna-emu/build/bridge-test-results/latest/*.bridge.log | head
```

Expected: `XDNA_EMU_STATUS: ...` line with `max_cycles` equal to 3 × the HW max from the cycles file.

- [ ] **Step 3: Smoke-test with a test that has no HW cycles file**

```bash
rm -f /home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/add_one_using_dma.hw.cycles.txt
./scripts/emu-bridge-test.sh --chess-only --no-hw add_one_using_dma
grep 'XDNA_EMU_STATUS' /home/triple/npu-work/xdna-emu/build/bridge-test-results/latest/*.bridge.log
```

Expected: `max_cycles=0` (unbounded — no reference point).

- [ ] **Step 4: Commit**

```bash
git add scripts/emu-bridge-test.sh
git commit -m "bridge: compute per-test cycle budget + export XDNA_EMU_MAX_CYCLES (Phase C.3)

run_one_bridge now parses {t}.hw.cycles.txt (if present), applies
the override multiplier (default 3x), and exports the result as
XDNA_EMU_MAX_CYCLES before launching the EMU run. Missing cycles
file -> unbounded (0), so EMU-only tests remain unaffected."
```

---

## Task 14: Phase C.5 — Results column merge + show-cycle-drift helper

**Files:**
- Modify: `/home/triple/npu-work/xdna-emu/scripts/emu-bridge-test.sh` (print_report)
- Create: `/home/triple/npu-work/xdna-emu/scripts/show-cycle-drift.sh`

- [ ] **Step 1: Merge cycle counts into results table**

In `print_report`, find the inner loop that prints each test row. In the HW column print (`printf "  %-${col_width}s" "$hr"`, around line 1474), precede it with a lookup of the HW cycles:

```bash
        local hw_cycles_file="$BUILD_BASE/$name/$compiler/${safe}${vsuffix}.hw.cycles.txt"
        local hw_max=""
        hw_max="$(parse_hw_cycles_max "$hw_cycles_file" 2>/dev/null)"
        local hw_cell="$hr"
        [[ -n "$hw_max" ]] && [[ "$hw_max" -gt 0 ]] && hw_cell="$hr $hw_max"
        printf "  %-16s" "$hw_cell"
```

(Widen the column to `%-16s` so the suffix fits.)

For the EMU column, similarly extract cycles from the status line:

```bash
      local emu_status_line
      emu_status_line="$(grep 'XDNA_EMU_STATUS:' "$RESULTS_DIR/${safe}${vsuffix}.${compiler}.bridge.log" 2>/dev/null | tail -1)"
      local emu_cycles=""
      [[ -n "$emu_status_line" ]] && emu_cycles="$(echo "$emu_status_line" | grep -oP '\bcycles=\K\d+')"
      local emu_cell="$br"
      if [[ -n "$emu_cycles" ]] && [[ -n "$hw_max" ]] && [[ "$hw_max" -gt 0 ]]; then
        emu_cell="$br $emu_cycles/$hw_max"
      fi
      printf "  %-22s" "$emu_cell"
```

Update the header printing block at top of `print_report` (around line 1340) to widen the column labels accordingly.

- [ ] **Step 2: Create the drift helper**

`/home/triple/npu-work/xdna-emu/scripts/show-cycle-drift.sh`:

```bash
#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Show tests sorted by EMU_cycles / HW_cycles ratio (highest first).
# Reads the latest bridge-test results directory.
#
# Usage:
#   ./scripts/show-cycle-drift.sh              # latest results, all tests
#   ./scripts/show-cycle-drift.sh -n 20        # top 20 drift tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EMU_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${BRIDGE_TEST_RESULTS:-${EMU_ROOT}/build/bridge-test-results/latest}"

TOP_N=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) TOP_N="$2"; shift 2 ;;
    *)  echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "No results directory at $RESULTS_DIR" >&2
  exit 1
fi

# Walk every bridge.log, extract cycles= and find the matching HW cycles file.
declare -a rows
for log in "$RESULTS_DIR"/*.bridge.log; do
  [[ -f "$log" ]] || continue
  base="$(basename "$log" .bridge.log)"  # e.g. add_one_using_dma.chess
  safe="${base%.*}"
  compiler="${base##*.}"

  emu_cycles="$(grep 'XDNA_EMU_STATUS:' "$log" | tail -1 | grep -oP '\bcycles=\K\d+' || true)"
  [[ -z "$emu_cycles" ]] && continue

  # Find matching HW cycles file by walking the build dir.
  mlir_aie="${EMU_ROOT}/../mlir-aie"
  hw_file="${mlir_aie}/build/test/npu-xrt/${safe}/${compiler}/${safe}.hw.cycles.txt"
  [[ -f "$hw_file" ]] || continue
  hw_max="$(awk '/^[[:space:]]*#/{next} /^[[:space:]]*$/{next} {if($3>max)max=$3} END{print max}' "$hw_file")"
  [[ -z "$hw_max" ]] || [[ "$hw_max" -eq 0 ]] && continue

  # Use awk for float ratio (bash can't do float).
  ratio="$(awk -v e="$emu_cycles" -v h="$hw_max" 'BEGIN{ printf "%.3f", e/h }')"
  rows+=("$ratio $base $emu_cycles $hw_max")
done

printf "%-10s  %-50s  %-12s  %-12s\n" RATIO TEST EMU HW
printf "%-10s  %-50s  %-12s  %-12s\n" "----------" "$(printf '%.0s-' {1..50})" "------------" "------------"
printf '%s\n' "${rows[@]}" \
  | sort -rg \
  | { [[ -n "$TOP_N" ]] && head -n "$TOP_N" || cat; } \
  | awk '{ printf "%-10s  %-50s  %-12s  %-12s\n", $1, $2, $3, $4 }'
```

Make it executable:

```bash
chmod +x /home/triple/npu-work/xdna-emu/scripts/show-cycle-drift.sh
```

- [ ] **Step 3: Smoke-test the helper against a recent bridge run**

```bash
cd /home/triple/npu-work/xdna-emu
./scripts/show-cycle-drift.sh -n 10
```

Expected: a table with RATIO, TEST, EMU, HW columns, sorted descending by ratio. If no recent bridge-test results exist, it prints an empty table with just headers.

- [ ] **Step 4: Commit**

```bash
git add scripts/emu-bridge-test.sh scripts/show-cycle-drift.sh
git commit -m "bridge: merge cycle counts into results + add show-cycle-drift.sh (Phase C.5)

HW and EMU columns in print_report now include cycle counts:
'PASS 8421/8400' format (status + EMU/HW). The new
show-cycle-drift.sh walks the latest results dir and ranks tests
by EMU/HW ratio — the go-to tool for the perf-hunt that follows
this workstream."
```

---

## Task 15: Full-suite verification

No commit. This is the gate that the cycle-budget workstream is ready to feed the perf hunt.

- [ ] **Step 1: Run the full bridge suite**

```bash
cd /home/triple/npu-work/xdna-emu
./scripts/emu-bridge-test.sh 2>&1 | tee /tmp/claude-1000/full-bridge-test.log
```

Expected runtime: 15-30 minutes (per the CLAUDE.md note).

- [ ] **Step 2: Confirm cycle counts show in the results table**

```bash
grep -A3 'TEST.*Chess/HW\|TEST.*Chess/EMU' /tmp/claude-1000/full-bridge-test.log | head
```

Expected: column headers and a few data rows show `PASS <N>` / `PASS <N>/<M>` format.

- [ ] **Step 3: Confirm today's regressions are now classified**

The 20 tests that went PASS → TIMEOUT in Subsystem 7 should now each show as either:
- `BUDGET` (cycle-modeled slowdown — perf counter shows > 3× HW cycles), or
- `TIMEOUT` (host-side wall-clock slowdown — plugin never reported halt_reason=budget, process hit the 600s wall)

Grep the summary:

```bash
grep -E 'BUDGET|TIMEOUT' /tmp/claude-1000/full-bridge-test.log
```

Expected: some combination of BUDGET and TIMEOUT counts, but no longer opaque — each is one class or the other, ready for Phase 2 investigation.

- [ ] **Step 4: Run show-cycle-drift.sh**

```bash
./scripts/show-cycle-drift.sh -n 20
```

Expected: a ranked list with the top 20 drift tests. The tests that became BUDGET in step 3 should appear near the top of this list.

- [ ] **Step 5: If any unexpected FAIL or EMU_MISS appears**

Grep for them and investigate individually. An EMU_MISS indicates the plugin isn't being loaded; a FAIL on a previously-passing test indicates the cycle-budget plumbing introduced a bug. Fix before declaring this workstream done.

- [ ] **Step 6: Report out**

Post a short summary:
- N tests PASS
- N tests BUDGET (cycle-modeled slowdown candidates)
- N tests TIMEOUT (wall-clock slowdown candidates)
- Top 5 drift tests from `show-cycle-drift.sh`

That summary is the handoff to the perf-hunt workstream.

---

## Self-Review

**Spec coverage check:**

| Spec section | Covered by | Gaps |
|---|---|---|
| XRT patch prerequisite (discovered during execution) | Task 1 | none |
| Phase B — HW cycle capture | Tasks 2-4 | none |
| Phase A — EMU cycle budget | Tasks 5-7 | none |
| Phase D — EMU counter semantic fix | Tasks 8-10 | Stall events deferred per spec |
| Phase C — Bridge comparison gate | Tasks 11-14 | C.4 (BUDGET classification) rolled into Task 7 per spec's note |
| Success criteria | Task 15 | none |

**Placeholder scan:** Task 5's `max_cycles_one_hits_budget` test is a documented TODO — left intentionally as a placeholder for future fixture-based work. All other steps contain concrete code or commands.

**Type consistency:** `CycleCounterHelper` name used consistently (header, sed injection, docs). `XdnaEmuHaltReason` Rust enum and `XdnaEmuHaltReason` C++ enum share the same variant names (`Completed=0`, `Budget=1`, `Error=2`). `tick_active_cycles` / `tick_idle_cycles` used consistently across Tasks 8 and coordinator.rs. Status key `XDNA_EMU_STATUS:` consistent across plugin emit (Task 6) and bridge parser (Task 7). Env var `XDNA_EMU_MAX_CYCLES` consistent across plugin, bridge, docs.

**Known deviations from spec:**
- **Task 1 (XRT patch) was discovered during execution, not in the spec.** The spec assumed `xrt::device::read_aie_reg(col, row, addr)` existed as a public 3-arg API. Investigation revealed the real API is `xrt::aie::device::read_aie_reg(pid, context_id, col, row, addr)` whose constructor conflicts with `xrt::hw_context` and whose `context_id` is not publicly accessible. The patch adds `xrt::hw_context::{read,write}_aie_reg(col, row, addr)` to our locally-built XRT. Planned to upstream eventually.
- The helper header lives under `runtime_lib/test_lib/include/` rather than `test/npu-xrt/` as the spec outlined. This is because `test_utils.h` is the natural include entry point for tests, and the helper is pulled in transparently via that header — no CMake change needed.
- `bridge_budget` counter and summary entry added to `print_report` in Task 7 (Phase A.3) rather than Phase C.4 because the classification machinery is already in place once the status parser lands; splitting it would create dead code during A-land/C-land gap.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-22-cycle-budget-testing.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**

---

## Post-Phase B status update (2026-04-23)

Phase C ("cycle budget enforcement") and Phase D.3 ("HW integration
spot-check") are superseded by Phase E -- see
[`2026-04-23-phase-e-trace-diff-cycle-budget.md`](2026-04-23-phase-e-trace-diff-cycle-budget.md)
for the replacement design and
[`2026-04-23-phase-e-validation.md`](2026-04-23-phase-e-validation.md)
for empirical results.
