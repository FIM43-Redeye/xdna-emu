# debug_halt Hardware Probe (Phase A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a permanent, checked-in NPU1 control-packet probe that derives, from real silicon, (G1) whether a breakpoint/single-step halt takes effect before or after the trap bundle commits, and (G2) the observable behavior of the count-based single-step register `Debug_Control0[5:2]`.

**Architecture:** A new bridge test `debug_halt_probe` under `mlir-aie/test/npu-xrt/`, modeled on the verified `add_one_ctrl_packet` control-packet plumbing. A minimal straight-line compute core writes distinctly-valued sentinel markers to an output buffer; the runtime sequence issues control-packet register writes (`aiex.npu.write32` to compute-tile debug registers) to arm breakpoints / count-step, then DMAs the marker buffer back to the host. The decisive observation is plain output-BO readback; control-packet register readback (hand-assembled, as `add_one_ctrl_packet` already does) is corroborating. Every experiment runs EMU-first through the real bridge flow, then once on hardware. Findings are recorded in a durable findings doc; the probe stays checked in as a re-runnable regression. Phase B (the implementation) is a separate, later plan written against the recorded findings.

**Tech Stack:** mlir-aie (IRON / `aiex` runtime-sequence dialect), XRT bridge plugin, the xdna-emu emulator, `scripts/emu-bridge-test.sh`, `tools/llvm-objdump-aie`.

**Spec:** `docs/superpowers/specs/2026-05-18-debug-halt-design.md` (Phase A = Section 4; this plan implements Section 4 only).

---

## Conventions for every task

- Workspace root: `/home/triple/npu-work`. Emulator repo: `/home/triple/npu-work/xdna-emu`.
- Probe test source dir (created in Task 1): `/home/triple/npu-work/mlir-aie/test/npu-xrt/debug_halt_probe/`.
- Compute tile under test is `(0, 2)` (column 0, row 2) — matches the `add_one_ctrl_packet` template and `npu1_1col`.
- Debug register offsets (tile-local, AIE2, from the spec audit): Debug_Control0 `0x32010`, Debug_Control2 `0x32018`, Debug_Status `0x3201C`, PC_Event0 `0x32020`, Core_Status `0x32004`. PC_Event* = bit 31 VALID, bits [13:0] 14-bit PC_ADDRESS.
- **EMU-first, always.** Run on the emulator and confirm the harness behaves before any hardware run. Never run two hardware suites concurrently.
- **Hardware-run guard (applies to every HW step):** before the HW run, confirm no other HW suite is active; stage NPU recovery — if the NPU wedges, first recovery is `pkexec sh -c 'modprobe -r amdxdna && modprobe amdxdna'` (single combined pkexec call), then smoke-test with `xrt-smi validate` (not a bridge test). Do not run `xrt-smi` while a HW test is active. If processes are stuck in D-state, stop and hand a reboot to Maya — do not self-reboot.
- EMU invocation pattern (from the built test dir): `XDNA_EMU=1 XDNA_EMU_RUNTIME=debug ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin`
- HW invocation pattern: `env -u XDNA_EMU ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin`
- Bridge driver: `cd /home/triple/npu-work/xdna-emu && ./scripts/emu-bridge-test.sh debug_halt_probe --chess-only --no-hw` (EMU only) / drop `--no-hw` for the HW run. Results: `build/bridge-test-results/latest/`.
- Disassembler: `/home/triple/npu-work/xdna-emu/tools/llvm-objdump-aie -d <core elf>`.
- After any Rust change: `cargo build` (none expected in Phase A — this plan authors a test artifact and records findings; it does not modify emulator Rust).
- Commit after every task. No emoji. End commit messages with `Generated using Claude Code.`
- Findings doc (created in Task 4, appended in Task 6, finalized in Task 7): `docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md`.

---

## File structure

| File | Responsibility | Tasks |
|------|----------------|-------|
| `mlir-aie/test/npu-xrt/debug_halt_probe/aie.mlir` | Probe kernel + device: minimal straight-line marker core, output DMA, control-packet runtime sequence | 1,2,3,5 |
| `mlir-aie/test/npu-xrt/debug_halt_probe/test.cpp` | Host harness: BO alloc, marker readback + verdict print | 1,2,3,5 |
| `mlir-aie/test/npu-xrt/debug_halt_probe/run.lit` | Build recipe (copied from `add_one_ctrl_packet`, retargeted) | 1 |
| `mlir-aie/test/npu-xrt/debug_halt_probe/README.md` | What this probe is, how to re-run it, what each experiment proves | 1,7 |
| `docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md` | Recorded observed truth (G1, G2) with raw evidence | 4,6,7 |

The reference template (read, do not modify): `/home/triple/npu-work/mlir-aie/test/npu-xrt/add_one_ctrl_packet/{aie.mlir,test.cpp,run.lit}`.

---

## Task 1: De-risking authoring spike — control-packet write to a compute-tile register, observed on EMU

**Goal:** Prove the unknown mechanism cheaply before building the experiments: a minimal NPU1 kernel whose core writes 3 sentinel markers to an output buffer, plus a runtime sequence that issues one `aiex.npu.write32` to a compute-tile debug register (the innocuous `Debug_Control0 = 0`). Success = the marker buffer reads back correctly on EMU **and** the emulator's control-packet path shows the write reaching tile (0,2) offset `0x32010`.

**Files:**
- Create: `mlir-aie/test/npu-xrt/debug_halt_probe/aie.mlir`
- Create: `mlir-aie/test/npu-xrt/debug_halt_probe/test.cpp`
- Create: `mlir-aie/test/npu-xrt/debug_halt_probe/run.lit`
- Create: `mlir-aie/test/npu-xrt/debug_halt_probe/README.md`
- Reference (read-only): `mlir-aie/test/npu-xrt/add_one_ctrl_packet/{aie.mlir,test.cpp,run.lit}`

- [ ] **Step 1: Read the reference template in full**

Read all three of `/home/triple/npu-work/mlir-aie/test/npu-xrt/add_one_ctrl_packet/aie.mlir`, `test.cpp`, `run.lit`. The verified facts you depend on:
- `run.lit` recipe (copy verbatim, only the `%S` test name changes): `cp %S/aie.mlir aie_arch.mlir` → `sed 's/NPUDEVICE/npu1_1col/g'` → `aiecc.py --no-aiesim --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=basic-sequential --xclbin-name=aie.xclbin --npu-insts-name=insts.bin aie_arch.mlir` → `clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags` → `%run_on_npu1% ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin`.
- The verified `aiex.npu.write32` op syntax (from `add_one_ctrl_packet/aie.mlir:131`): `aiex.npu.write32 {address = 0x1d214 : ui32, column = 0 : i32, row = 0 : i32, value = 0x80000000 : ui32}` — `address` is a raw ui32 tile-local offset; `column`/`row` select the tile. Targeting compute tile (0,2) is `column = 0 : i32, row = 2 : i32`.
- The shim/DMA/output plumbing pattern: `aie.shim_dma_allocation`, `aiex.npu.dma_memcpy_nd(...) {metadata = @sym}`, `aiex.npu.dma_wait {symbol = @sym}`, and the `aie.mem`/`aie.dma_bd` MM2S path (lines 96-108, 121-122, 140).

- [ ] **Step 2: Write the spike `aie.mlir`**

Create `mlir-aie/test/npu-xrt/debug_halt_probe/aie.mlir`. Keep the **shim allocation, packet flows, `aie.mem` MM2S output path, and runtime-sequence DMA plumbing structurally identical to `add_one_ctrl_packet`** (copy those regions and adjust symbols), but replace the core with a minimal straight-line marker core (no infinite loop, no input locks):

```mlir
//===- aie.mlir --- debug_halt_probe (Phase A spike) -----------*- MLIR -*-===//
// Probe kernel: minimal straight-line core writes 3 sentinel markers to
// output_buffer, then ends. Runtime sequence issues one control-packet
// register write to a compute-tile debug register to prove the authoring
// mechanism. Marker buffer is DMA'd back to the host.
//===----------------------------------------------------------------------===//
module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
    %tile_0_2 = aie.tile(0, 2)

    %output_lock4 = aie.lock(%tile_0_2, 4) {init = 0 : i32, sym_name = "output_lock4"}
    %output_lock5 = aie.lock(%tile_0_2, 5) {init = 1 : i32, sym_name = "output_lock5"}

    %output_buffer = aie.buffer(%tile_0_2) {sym_name = "output_buffer"} : memref<8xi32>

    aie.packet_flow(0x3) {
      aie.packet_source<%tile_0_2, DMA : 0>
      aie.packet_dest<%tile_0_0, DMA : 1>
    }
    aie.flow(%tile_0_0, DMA : 1, %tile_0_2, DMA : 1)

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %mAA = arith.constant 170 : i32   // 0xAA pre-trap marker
      %mBB = arith.constant 187 : i32   // 0xBB trap-bundle marker
      %mCC = arith.constant 204 : i32   // 0xCC post-trap marker
      aie.use_lock(%output_lock5, AcquireGreaterEqual, 1)
      memref.store %mAA, %output_buffer[%c0] : memref<8xi32>
      memref.store %mBB, %output_buffer[%c1] : memref<8xi32>
      memref.store %mCC, %output_buffer[%c2] : memref<8xi32>
      aie.use_lock(%output_lock4, Release, 1)
      aie.end
    }

    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      aie.use_lock(%output_lock4, AcquireGreaterEqual, 1)
      aie.dma_bd(%output_buffer : memref<8xi32>, 0, 8) {packet = #aie.packet_info<pkt_id = 3, pkt_type = 0>}
      aie.use_lock(%output_lock5, Release, 1)
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }

    aie.shim_dma_allocation @out0 (%tile_0_0, S2MM, 1)

    aie.runtime_sequence @seq(%arg0: memref<8xi32>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c8_i64 = arith.constant 8 : i64
      // SPIKE: prove a control-packet write to a compute-tile debug register
      // is expressible. Debug_Control0 = 0 is innocuous (no-op semantics).
      aiex.npu.write32 {address = 0x32010 : ui32, column = 0 : i32, row = 2 : i32, value = 0 : ui32}
      // DMA the marker buffer back to the host.
      aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c8_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, issue_token = true, metadata = @out0} : memref<8xi32>
      aiex.npu.dma_wait {symbol = @out0}
    }
  }
}
```

If `aiecc.py` rejects any region (it will report the offending op), reconcile that region against the exact `add_one_ctrl_packet/aie.mlir` text — the plumbing there is known-good on `npu1_1col`.

- [ ] **Step 3: Write the spike `test.cpp`**

Create `mlir-aie/test/npu-xrt/debug_halt_probe/test.cpp`. Model BO setup and arg order on `add_one_ctrl_packet/test.cpp` (read it for the exact `xrt::bo` / `kernel.group_id` / `run.wait()` idioms and the `test_utils` includes). The probe harness only needs the instruction BO and one output BO:

```cpp
// debug_halt_probe host harness. Reads 8 marker words back and prints them.
#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include <cstdint>
#include <iostream>
#include <vector>

int main(int argc, const char *argv[]) {
  cxxopts::Options options("debug_halt_probe");
  test_utils::add_default_options(options);
  cxxopts::ParseResult vm = test_utils::parse_options(argc, argv, options);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  auto kname = xclbin.get_kernels()[0].get_name();
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kname);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_out = xrt::bo(device, 8 * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));

  memcpy(bo_instr.map<void *>(), instr_v.data(),
         instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = kernel(3, bo_instr, (uint32_t)instr_v.size(), bo_out);
  run.wait();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  uint32_t *out = bo_out.map<uint32_t *>();
  std::cout << "MARKERS:";
  for (int i = 0; i < 8; i++)
    std::cout << " " << out[i];
  std::cout << "\n";

  bool pass = (out[0] == 0xAA && out[1] == 0xBB && out[2] == 0xCC);
  std::cout << (pass ? "SPIKE_PASS" : "SPIKE_FAIL") << "\n";
  return pass ? 0 : 1;
}
```

The kernel arg/group-id indices (`kernel.group_id(1)` instr, `group_id(3)` output) must match what `add_one_ctrl_packet/test.cpp` uses for the analogous BOs after the runtime-sequence arg-count change — verify against that file and the aiecc-generated kernel signature; adjust the indices if aiecc assigns differently for a single-output sequence.

- [ ] **Step 4: Write `run.lit` and `README.md`**

`run.lit` — copy `add_one_ctrl_packet/run.lit` verbatim; it already references `%S/aie.mlir` and `%S/test.cpp`, so no path edits are needed. Confirm it keeps `// REQUIRES: ryzen_ai` and the `%run_on_npu1%` / `sed 's/NPUDEVICE/npu1_1col/g'` lines.

`README.md` — short: what the probe is (Phase A of debug_halt; derives halt timing G1 + count-step G2), how to re-run (`./scripts/emu-bridge-test.sh debug_halt_probe --chess-only` from `xdna-emu/`), and that experiment specifics live in the spec `docs/superpowers/specs/2026-05-18-debug-halt-design.md`.

- [ ] **Step 5: Build + run on EMU; verify markers and the control-packet write reach the tile**

Run: `cd /home/triple/npu-work/xdna-emu && RUST_LOG=info ./scripts/emu-bridge-test.sh debug_halt_probe --chess-only --no-hw 2>&1 | tee /tmp/claude-1000/probe-spike.log`

Expected: the test builds (xclbin + insts.bin produced), runs under the emulator, and `test.exe` prints `MARKERS: 170 187 204 ...` and `SPIKE_PASS`. In `/tmp/claude-1000/probe-spike.log` (or the per-test EMU log under `build/bridge-test-results/latest/debug_halt_probe.chess/emu/`), confirm a control-packet / register-write log line showing a write to tile `(0,2)` offset `0x32010`. (The just-shipped control_packets path logs dispatched register writes; if not visible at `info`, grep the EMU log for `32010` or `WriteRegister`.)

If `SPIKE_FAIL` or no write reaches the tile: the mechanism is not yet proven — debug the authoring (most likely the runtime-sequence arg count / group-id mapping or a plumbing op rejected by aiecc) before proceeding. This task's whole purpose is to find that out cheaply.

- [ ] **Step 6: Commit**

```bash
cd /home/triple/npu-work/mlir-aie
git add test/npu-xrt/debug_halt_probe/
git commit -m "debug_halt_probe: authoring spike -- control-packet write to compute-tile debug reg, EMU-verified

Minimal NPU1 straight-line marker kernel + runtime sequence that issues
aiex.npu.write32 to compute-tile (0,2) Debug_Control0. Proves the
control-packet-to-debug-register authoring mechanism on the emulator
before building the G1/G2 experiments. Phase A of the debug_halt spec.

Generated using Claude Code."
```

(The probe lives in the `mlir-aie` working tree — commit there. If `mlir-aie` is not the intended home for checked-in probes, instead place the test dir under `xdna-emu/tools/experiments/debug_halt_probe/` and symlink/copy into the bridge discovery path; confirm with Maya at the regroup. Default: commit in `mlir-aie` alongside the other `npu-xrt` tests.)

---

## Task 2: Experiment 1 kernel — separable pre/trap/post marker bundles

**Goal:** Shape the core so the trap bundle is unambiguously identifiable in disassembly and the three markers occupy distinct buffer slots, so post-run readback yields a clean before/after verdict.

**Files:**
- Modify: `mlir-aie/test/npu-xrt/debug_halt_probe/aie.mlir` (core region only)
- Modify: `mlir-aie/test/npu-xrt/debug_halt_probe/test.cpp` (verdict logic)

- [ ] **Step 1: Make markers individually traceable in the core**

The Task 1 core already stores `0xAA, 0xBB, 0xCC` to `output_buffer[0..2]`. Make each store depend on a distinct constant so the three `memref.store` bundles do not get coalesced/reordered by the compiler into one bundle. Replace the core body with:

```mlir
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %mAA = arith.constant 170 : i32
      %mBB = arith.constant 187 : i32
      %mCC = arith.constant 204 : i32
      %done = arith.constant 1 : i32
      aie.use_lock(%output_lock5, AcquireGreaterEqual, 1)
      memref.store %mAA, %output_buffer[%c0] : memref<8xi32>   // PRE-trap
      memref.store %mBB, %output_buffer[%c1] : memref<8xi32>   // TRAP bundle
      memref.store %mCC, %output_buffer[%c2] : memref<8xi32>   // POST-trap
      memref.store %done, %output_buffer[%c3] : memref<8xi32>  // sentinel: core ran to end
      aie.use_lock(%output_lock4, Release, 1)
      aie.end
    }
```

- [ ] **Step 2: Encode the before/after verdict in `test.cpp`**

Replace the `pass`/print block in `test.cpp` with verdict logic that interprets the markers for Experiment 1 (the trap is armed in Task 3; until then all four markers land):

```cpp
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  uint32_t *out = bo_out.map<uint32_t *>();
  std::cout << "MARKERS: out0=" << out[0] << " out1=" << out[1]
            << " out2=" << out[2] << " out3=" << out[3] << "\n";
  // No-trap baseline: all four markers present, core reached aie.end.
  bool baseline = (out[0] == 0xAA && out[1] == 0xBB &&
                   out[2] == 0xCC && out[3] == 1);
  // Trap armed (Task 3): out0 always lands (pre-trap).
  //  - out1==0xBB && out2==0    -> trap bundle COMMITTED -> HALT AFTER COMMIT
  //  - out1==0    && out2==0    -> trap bundle DID NOT commit -> HALT BEFORE COMMIT
  // out3 (done sentinel) must be 0 when trapped (core halted, never reached end).
  std::cout << "BASELINE:" << (baseline ? "YES" : "NO") << "\n";
  std::cout << "TRAP_VERDICT:";
  if (out[0] == 0xAA && out[3] == 0) {
    if (out[1] == 0xBB && out[2] == 0)      std::cout << "AFTER_COMMIT\n";
    else if (out[1] == 0 && out[2] == 0)    std::cout << "BEFORE_COMMIT\n";
    else                                     std::cout << "AMBIGUOUS\n";
  } else {
    std::cout << "NO_TRAP_OR_RAN_TO_END\n";
  }
  return 0;
```

- [ ] **Step 3: Build + EMU run, confirm clean no-trap baseline**

Run: `cd /home/triple/npu-work/xdna-emu && ./scripts/emu-bridge-test.sh debug_halt_probe --chess-only --no-hw 2>&1 | tee /tmp/claude-1000/probe-exp1-baseline.log`

Expected: `MARKERS: out0=170 out1=187 out2=204 out3=1`, `BASELINE:YES`, `TRAP_VERDICT:NO_TRAP_OR_RAN_TO_END`. This confirms the un-trapped kernel runs to completion and all four markers are observable — the control for Experiment 1.

- [ ] **Step 4: Commit**

```bash
cd /home/triple/npu-work/mlir-aie
git add test/npu-xrt/debug_halt_probe/
git commit -m "debug_halt_probe: Exp1 kernel -- separable pre/trap/post/done markers + verdict logic

Four distinct marker stores (0xAA pre, 0xBB trap-bundle, 0xCC post,
done-sentinel) and test.cpp before/after-commit verdict. EMU no-trap
baseline confirmed: all four markers land, core runs to aie.end.

Generated using Claude Code."
```

---

## Task 3: Experiment 1 — arm the PC_Event breakpoint at the trap bundle; record EMU's observed timing

**Goal:** Disassemble the compiled core, pin the trap bundle's 14-bit PC, arm `PC_Event0`+`Debug_Control2[0]` via control packets, and record what the **emulator** does (this also validates the already-wired PC_Event path through the real bridge flow — a divergence here is itself a finding).

**Files:**
- Modify: `mlir-aie/test/npu-xrt/debug_halt_probe/aie.mlir` (runtime sequence: add PC_Event0 + Debug_Control2 writes)

- [ ] **Step 1: Disassemble the compiled core ELF and locate the trap bundle PC**

After Task 2's build, the Chess core ELF is at:
`/home/triple/npu-work/mlir-aie/build/test/npu-xrt/debug_halt_probe/chess/aie_arch.mlir.prj/main_core_0_2.elf`
(if the `.prj` subdir name differs, find it: `ls /home/triple/npu-work/mlir-aie/build/test/npu-xrt/debug_halt_probe/chess/*.prj/`).

Run: `/home/triple/npu-work/xdna-emu/tools/llvm-objdump-aie -d /home/triple/npu-work/mlir-aie/build/test/npu-xrt/debug_halt_probe/chess/aie_arch.mlir.prj/main_core_0_2.elf > /tmp/claude-1000/probe-disasm.txt`

Read `/tmp/claude-1000/probe-disasm.txt`. Identify the three store bundles. The markers are immediates `170 (0xAA)`, `187 (0xBB)`, `204 (0xCC)` — find the bundle that materializes `0xBB`/`187` and stores it to `output_buffer + 4` (slot 1). Record that bundle's instruction address (left-column hex). This is `TRAP_PC`. PC_Event matches the **low 14 bits** of the PC: `TRAP_PC14 = TRAP_PC & 0x3FFF`.

If the disasm is ambiguous (immediates folded), bump the trap marker to a distinctive value like `0xB1B2` and rebuild Task 2 so it is unmistakable in the instruction stream; re-disassemble.

- [ ] **Step 2: Arm the breakpoint in the runtime sequence**

In `aie.mlir`, in `@seq`, **before** the `dma_memcpy_nd`, replace the Task 1 spike `write32` with the breakpoint arming (substitute the real `TRAP_PC14` you recorded — shown here as `0xNNNN`):

```mlir
      // Arm PC_Event0 at the trap bundle (VALID bit 31 | 14-bit PC).
      aiex.npu.write32 {address = 0x32020 : ui32, column = 0 : i32, row = 2 : i32, value = 0x8000NNNN : ui32}
      // Debug_Control2[0] = PC_Event_Halt: halt when a PC event fires.
      aiex.npu.write32 {address = 0x32018 : ui32, column = 0 : i32, row = 2 : i32, value = 0x1 : ui32}
```

`0x8000NNNN` = `0x80000000 | TRAP_PC14`. These writes are issued before the core is released to run (the runtime sequence executes before the kernel's lock is released by the host op sequence; if ordering proves wrong on EMU in Step 3, this is the thing to fix — the arming must land before the core executes the trap bundle).

- [ ] **Step 3: EMU run — record the emulator's observed timing**

Run: `cd /home/triple/npu-work/xdna-emu && ./scripts/emu-bridge-test.sh debug_halt_probe --chess-only --no-hw 2>&1 | tee /tmp/claude-1000/probe-exp1-emu.log`

Read the `MARKERS:` / `TRAP_VERDICT:` lines. Expected on EMU, per the current model (halt evaluated after `update_pc`, i.e. post-commit): `out0=170 out1=187 out2=0 out3=0`, `TRAP_VERDICT:AFTER_COMMIT`. Record the exact observed line. If the verdict is `BEFORE_COMMIT`, `AMBIGUOUS`, or the core ran to end, that is a real finding about the emulator's wired PC_Event path — note it; do not "fix" it here (Phase B owns changes, parameterized on hardware truth).

- [ ] **Step 4: Commit**

```bash
cd /home/triple/npu-work/mlir-aie
git add test/npu-xrt/debug_halt_probe/aie.mlir
git commit -m "debug_halt_probe: Exp1 -- arm PC_Event0 breakpoint at trap bundle (TRAP_PC14=0xNNNN)

PC_Event0 = VALID|<trap bundle low-14 PC from llvm-objdump-aie>,
Debug_Control2[0]=1. EMU observed verdict recorded in
/tmp/claude-1000/probe-exp1-emu.log (to be transcribed into the
findings doc in Task 4).

Generated using Claude Code."
```

(Replace `0xNNNN` in the message with the real recorded value before committing.)

---

> **REVISION (2026-05-18, post-Task-3 discovery).** Tasks 1–3 are done and remain valid (kernel, breakpoint arming at the disasm-derived TRAP_PC, the TRAP_PC re-derivation discipline). Task 3 exposed two fatal flaws in the original Exp1 observation/verdict (spec §4.2 "Discovery"): (1) the compiler reorders the marker stores so a source-order verdict is inverted; (2) the lock-gated marker DMA hangs forever on a halted core (NPU-wedge). Force-ordering was investigated and rejected (no MLIR-level pin; empirically confirmed). The redesign — confirmed with Maya — is: observe via **control-packet OP_READ while the core is halted** (core-independent; prior art `add_one_ctrl_packet`), and compute the verdict from the **disassembled schedule** + `Core_Status`, not source order. Tasks 4–8 below replace the old Tasks 4–7.

## Task 4: Rebuild Exp1 observation — control-packet readback + schedule-derived verdict (EMU)

**Goal:** Replace the lock-gated-DMA observation with control-packet OP_READ of the marker buffer + `Core_Status` (works on a halted core), and a verdict computed from the disassembled schedule. EMU-validate the readback *mechanism* and the no-trap baseline (EMU cannot halt — it drops the breakpoint-arming writes — so read-while-halted is first exercised on hardware in Task 5).

**Files:**
- Modify: `mlir-aie/test/npu-xrt/debug_halt_probe/aie.mlir` (add control-packet read plumbing; keep the Task 2 4-marker core and the Task 3 breakpoint-arming writes)
- Modify: `mlir-aie/test/npu-xrt/debug_halt_probe/test.cpp` (assemble read control packets, parse responses, schedule-derived verdict)
- Modify: `mlir-aie/test/npu-xrt/debug_halt_probe/README.md` (observation-path + verdict description)
- Reference (read-only, the verified template): `/home/triple/npu-work/mlir-aie/test/npu-xrt/add_one_ctrl_packet/{aie.mlir,test.cpp}`

- [ ] **Step 1: Study the control-packet read template**

Read `add_one_ctrl_packet/aie.mlir` and `test.cpp` fully. The verified facts to reuse:
- aie.mlir control-packet read plumbing: `aie.packet_flow(0x1)` shim DMA:0 → tile TileControl:0 (host→tile ctrl packets); `aie.packet_flow(0x2)` tile TileControl:0 → shim DMA:0 (tile→host responses); `aie.shim_dma_allocation @ctrl0 (%tile_0_0, S2MM, 0)`; runtime-sequence `dma_memcpy_nd(... metadata=@ctrl0)` + `dma_wait{symbol=@ctrl0}` and the blockwrite/address_patch/sync push sequence (lines 110–155).
- Control-packet header word (xdna-emu `src/device/control_packets/parser.rs:9-14`): `[31] parity`, `[30:24] response/stream id`, `[23:22] operation` (OP_READ = `0x1`), `[21:20] length` (value = beats−1; beats = words to read), `[19:0] tile-local address`. OP_READ returns the addressed words back through the TileControl→shim response path; for addresses below the tile data-memory size (0x10000 for a compute tile) it reads **tile data memory** (`registers.rs:98-110`). `add_one_ctrl_packet/test.cpp` (read-packet assembly + parity, ~lines 93-136 and the ctrlOut validation ~164-186) is the concrete model — read it verbatim and reuse its packet-assembly idiom.

- [ ] **Step 2: Determine `output_buffer`'s tile-local control-packet address**

From the Task 3 disasm (`/tmp/claude-1000/probe-disasm.txt` or re-run `tools/llvm-objdump-aie -d` on the core ELF), `p0` = the `output_buffer` base in the core's 20-bit address space (recorded as `0x70400` in Task 3). The control-packet OP_READ address is the **tile-local data-memory offset** (the low bits within the 0x0000–0xFFFF compute data-memory window — i.e. `0x70400 & 0xFFFF = 0x0400`, the same form `add_one_ctrl_packet` uses when it reads `other_buffer` at `0x440`). Record `OUTBUF_ADDR` (expected `0x0400`; confirm against the aiecc allocation / by the EMU readback in Step 5 returning the known marker values — if the read returns garbage, the address is wrong, re-derive). Slots: `output_buffer[k]` at `OUTBUF_ADDR + 4*k`. `Core_Status` is register `0x32004`.

- [ ] **Step 3: Add the control-packet read plumbing to aie.mlir**

Add (modeled verbatim on `add_one_ctrl_packet/aie.mlir`): the `aie.packet_flow(0x1)` and `aie.packet_flow(0x2)` TileControl flows for tile (0,2), `aie.shim_dma_allocation @ctrl0 (%tile_0_0, S2MM, 0)`, and a second runtime-sequence argument for the ctrl-response BO. Keep the existing Task 2 4-marker core and the Task 3 `aiex.npu.write32` breakpoint-arming writes (PC_Event0=0x80000114, Debug_Control2=0x1) unchanged and still before the core is released. The `@seq` now also issues the read-control-packet DMA push (the blockwrite/address_patch/sync idiom from the template) and `dma_wait{symbol=@ctrl0}` so responses land in the ctrl-response BO. **Remove the `@out0` marker-DMA path entirely** (`dma_memcpy_nd`/`dma_wait @out0`, the lock-gated `aie.mem` MM2S block, `@out0` shim_dma_allocation, `packet_flow(0x3)`, and the `bo_out` arg in test.cpp/kernel): it is fully redundant (control-packet OP_READ is the sole observation) and a `dma_wait @out0` on a halted core blocks forever and wedges the NPU. The EMU no-trap baseline still passes because the OP_READ returns the markers regardless. Preserve SPDX headers and the aie-rt citation comments.

- [ ] **Step 4: Rewrite test.cpp — assemble read packets, parse responses, schedule-derived verdict**

Allocate the ctrl-in (read-request packets) and ctrl-out (response) BOs per the `add_one_ctrl_packet/test.cpp` group-id idiom (verify indices against the aiecc-generated signature). Assemble OP_READ control packets (parity per the template) requesting: the 4 `output_buffer` slots (`OUTBUF_ADDR + 4*k`, k=0..3) and `Core_Status` (0x32004). Parse the response words from the ctrl-out BO.

Verdict — computed from the **committed disassembled schedule** (Task 3 / spec §4.2), NOT source order. For the current build: trap bundle `0x114` stores `output_buffer[1]=0xBB`; strictly-later `0x11c` stores `output_buffer[0]=0xAA`; later still `[3]=1`, `[2]=0xCC`. Let `s = output_buffer` slots, `cs = Core_Status`. `HALTED` ≡ `cs` bit16 (Debug_Halt) set AND bit0 (Enable) set:

```cpp
  // Schedule (from llvm-objdump-aie, see aie.mlir TRAP_PC comment):
  //   trap bundle 0x114 -> s[1]=0xBB ; strictly-later 0x11c -> s[0]=0xAA
  //   later -> s[3]=1 ; later -> s[2]=0xCC
  bool halted = (cs & (1u<<16)) != 0;  // DEBUG_HALT bit alone -- ENABLE-stays-1 is an unverified HW assumption; DEBUG_HALT=1 already proves the core ran and is halted
  std::cout << "SLOTS: s0=" << s[0] << " s1=" << s[1]
            << " s2=" << s[2] << " s3=" << s[3]
            << " CORE_STATUS=0x" << std::hex << cs << std::dec
            << " HALTED=" << (halted?1:0) << "\n";
  std::cout << "TRAP_VERDICT:";
  if (!halted) {
    if (s[0]==0xAA && s[1]==0xBB && s[2]==0xCC && s[3]==1)
      std::cout << "NO_TRAP_OR_RAN_TO_END\n";        // breakpoint never fired
    else
      std::cout << "ANOMALY_NOT_HALTED\n";           // record verbatim
  } else if (s[1]==0xBB && s[0]==0 && s[2]==0 && s[3]==0) {
    std::cout << "AFTER_COMMIT\n";   // trap store committed, nothing later did
  } else if (s[1]==0 && s[0]==0 && s[2]==0 && s[3]==0) {
    std::cout << "BEFORE_COMMIT\n";  // halted, trap store did NOT commit
  } else {
    std::cout << "AMBIGUOUS\n";      // unexpected combo -- record verbatim
  }
  std::cout << "PASS\n";             // bridge-harness pass token (emu-bridge-test.sh grep)
  return 0;
```

- [ ] **Step 5: EMU run — validate the readback mechanism + no-trap baseline**

Run: `cd /home/triple/npu-work/xdna-emu && ./scripts/emu-bridge-test.sh debug_halt_probe --chess-only --no-hw 2>&1 | tee /tmp/claude-1000/probe-exp1-readback-emu.log`

EMU drops the breakpoint-arming writes (`write_core_register` catch-all), so the core runs to completion: expected `SLOTS: s0=170 s1=187 s2=204 s3=1`, `CORE_STATUS` not debug-halted, `TRAP_VERDICT:NO_TRAP_OR_RAN_TO_END`, bridge `PASS`. This proves the control-packet readback mechanism returns correct tile-data-memory + Core_Status values end-to-end. If the slots read back as garbage, `OUTBUF_ADDR` is wrong (Step 2) — re-derive. Record the exact lines. (Read-while-halted is unverifiable on EMU by construction; that is Task 5's job on hardware.)

- [ ] **Step 6: Commit**

```bash
cd /home/triple/npu-work/mlir-aie
git add test/npu-xrt/debug_halt_probe/
git commit -m "debug_halt_probe: Exp1 redesign -- control-packet readback + schedule-derived verdict

Lock-gated DMA readback hangs on a halted core (Task 3 discovery);
source-order verdict is inverted by compiler store-reordering. Replace
with control-packet OP_READ of output_buffer + Core_Status (works on a
halted core, prior art add_one_ctrl_packet) and a verdict computed from
the disassembled schedule. EMU validates the readback mechanism + the
no-trap baseline; read-while-halted is hardware-first (Task 5).

Generated using Claude Code."
```

---

## Task 5: Experiment 1 — objectfifo-gate redesign + re-derive TRAP_PC (EMU)

> **REVISION (Task 5 HW attempt 1 failed: arming race).** The first HW run did not halt — the core is enabled by the CDO before `@seq` runs, so it completed before the `@seq` arming `write32`s landed (`Core_Status=0x100000` CORE_DONE). Toolchain grounding (spec §4.2) rejected host-lock-release, CDO debug-reg init, and memory-poll gates; the robust fix is a **blocking objectfifo gate**. This Task 5 implements that gate (EMU); Task 5b re-runs on hardware.
>
> **REVISION 2 (Task 5 Step 3: shim-channel collision).** The first gated EMU run timed out at 600s. Grounded root cause (spec §4.2 shim-channel-disjointness): `@gate` defaulted its shim feed to shim MM2S ch0, which the hand-rolled ctrl-in OP_READ push also targets, so the pathfinder compiled a single circuit broadcast (`switchbox(0,2): South:1 → {TileControl:0, DMA:0}`) — ctrl-in headers fan into the gate buffer and the gate token into TileControl, `dma_wait @ctrl0` never satisfies. Not an EMU bug (faithfully reproduced) and HW-unsafe (would wedge). Pinning `@gate` to ch1 via `aie.shim_dma_allocation` was found **not toolchain-honored** (grounded, spec §4.2: `DMAChannelAnalysis` never reads `ShimDMAAllocationOp`; no per-objectfifo channel attr; objectfifo always takes lowest free channel). **Adopted fix:** leave `@gate` on its natural default shim MM2S **ch0**; repoint the hand-rolled ctrl-in OP_READ push to shim MM2S **ch1** (the controllable side). Step 1 below incorporates this; the Step-2 TRAP_PC=0x184 re-derivation from the interrupted partial is independent of the shim channel but is re-verified after the corrected build.
>
> **REVISION 3 (Task 5 Step 3: no happens-after — MASKPOLL halt-sync).** With the ch1 fix the EMU run no longer collides, but came back all-zero slots / `ANOMALY_NOT_HALTED`. Grounded (spec §4.2 Halt-synchronization, P2/P3): the static-`@seq` OP_READ has **no synchronization** ordering "core halted at trap" before "OP_READ reads" — pure relative latency (on EMU the gated core's stores land after the run loop terminates at `dma_wait @ctrl0`; on HW the on-die halt and host/NoC OP_READ have no primitive between them). Grounding established no on-device halt→lock/event actuation exists and `@seq` is strictly static, so the robust fix is a firmware **`XAIE_IO_MASKPOLL`** (opcode 4) post-compile-injected into `insts.bin` (no MLIR op emits it), blocking the stream until `Core_Status[16]` (`DEBUG_HALT`) before the OP_READ. The streamed MASKPOLL has **no timeout**; EMU never satisfies it (write-side gap → core never halts), so the emulator gains a **graceful poll-termination** contract (deterministic, honest: distinct `MaskPollUnsatisfied` terminal reason, clean `run.wait()`, no register fakery, no pretend-halt) and `test.cpp` treats MASKPOLL-unsatisfied as the **expected EMU baseline**. EMU/HW run the identical injected binary. Steps 3-4 below are this redesign; Steps 1-2 stand.

**Status:** Steps 1-2 **DONE**, committed mlir-aie `9a12651d99` (objectfifo `@gate` on shim MM2S ch0; ctrl-in repointed to ch1; route disjoint; TRAP_PC=0x184, core ELF unchanged by the repoint). Steps 3-4 below are the **REVISION 3 MASKPOLL halt-sync redesign** (replaces the obsolete static-`@seq` no-trap-baseline Step 3).

**Goal:** A host→core blocking objectfifo gate guarantees arming-before-core-run (Steps 1-2, done). A post-compile-injected MASKPOLL on `Core_Status[16]` then makes the OP_READ provably happen-after the core halts (synchronization-ordered, not latency-ordered). The emulator is hardened so the unsatisfiable-on-EMU poll terminates deterministically; EMU validates the gate-feed, disjoint route, injector, and poll-termination contract (expected EMU outcome: `MASKPOLL_UNSATISFIED_EMU`). The G1 answer is HW-only (Task 5b).

**Files:** Modify `mlir-aie/test/npu-xrt/debug_halt_probe/{aie.mlir,test.cpp,README.md}`. Reference patterns (read-only): `/home/triple/npu-work/mlir-aie/test/aiecc/cpp_basic.mlir` and `/home/triple/npu-work/mlir-aie/programming_examples/vision/vision_passthrough/aie2_lineBased_8b_tiny.mlir` (canonical host→core objectfifo-feed via runtime-sequence `dma_memcpy_nd`).

- [ ] **Step 1: Add the gate objectfifo.** Define a 1-element shim→compute objectfifo `@gate` (host/shim producer → tile (0,2) consumer, `memref<1xi32>`), modeled on the cited examples. The core's FIRST op becomes `aie.objectfifo.acquire @gate(Consume,1)` (blocks on `llvm.aie2.acquire`, a HW pipeline stall) then `aie.objectfifo.release @gate(Consume,1)`, then the existing 4 marker stores + `aie.end`. Keep the Task-3 breakpoint-arming `write32`s in `@seq` and the Task-4 control-packet OP_READ readback plumbing. In `@seq`, order: arming `write32`s (0x32020/0x32018) → `dma_memcpy_nd` feeding `@gate` → the OP_READ readback push → `dma_wait{@ctrl0}`. (Runtime-sequence order is an in-order guarantee per `AIEToConfiguration`.) Preserve SPDX, aie-rt citations, TRAP_PC/OUTBUF_ADDR re-derivation warnings; no `@out0`/lock-gated-DMA reintroduction. Add a host gate-buffer BO arg. **Shim-channel disjointness (REVISION 2, mandatory):** `@gate` stays on its natural default shim MM2S **ch0** (the objectfifo channel is not pinnable — grounded, spec §4.2). Repoint the hand-rolled ctrl-in OP_READ push from shim MM2S **ch0** to **ch1**. The toolchain-derived delta (per-channel stride `0x8`, aie-rt `xaiemlgbl_reginit.c .ChIdxOffset = 0x8`; confirm against the regdb before editing): ctrl-in CTRL `0x1d210→0x1d218`; ctrl-in TASK_QUEUE `0x1d214→0x1d21c` (all five occurrences); the ctrl-in `aie.packet_flow` source `<%tile_0_0, DMA : 0>→<%tile_0_0, DMA : 1>`; the `aiex.npu.sync` MM2S ops `channel = 0→1` (all five). **Do not change** any BD-layout/BD-address writes (`0x1d000`, `0x1d004` address_patch, `blockwrite_data_0`) — BDs are channel-independent. Do not touch the `@gate` objectfifo's channel (leave it default). The two host→compute flows must take independent physical shim paths or the pathfinder broadcasts both from a shared slave (spec §4.2). **Mandatory gate before the EMU run:** compile EMU-only (`--no-hw --compile`) and verify in the lowered `input_physical.mlir` that `switchbox(0,2)` routes `@gate` and the ctrl-in flow on *distinct* slave ports (no shared `South:1 → {TileControl:0, DMA:0}` broadcast); quote the relevant `aie.connect`/`aie.packet_rules` lines. If still shared, STOP and report.

- [ ] **Step 2: Re-derive TRAP_PC + slot⇄schedule map.** The gate changes the compiled core. Disassemble the new core ELF (`tools/llvm-objdump-aie -d <chess prj>/main_core_0_2.elf`), identify the bundle that stores `0xBB(187)` to the slot-1 store (the trap), record the new `TRAP_PC`, `TRAP_PC14 = TRAP_PC & 0x3FFF`, new `PC_Event0 value = 0x80000000 | TRAP_PC14`, and the new strictly-later slot/PC. Update the `write32 0x32020` value and the test.cpp schedule-comment + verdict slot mapping accordingly. Update the in-artifact TRAP_PC warning text with the new derivation. If the disasm can't unambiguously identify the 0xBB-to-slot1 store, STOP and report.

- [ ] **Step 3: Build the MASKPOLL `insts.bin` injector.** A standalone tool under `xdna-emu/tools/` (do NOT extend the rewrite-only `trace-patch-events.py`). Given a compiled `insts.bin`, insert the 28-byte `XAIE_IO_MASKPOLL` instruction and fix the header. Byte-exact form (LE; grounded, spec §4.2): `[0]`opcode=`0x04`+3 pad; `[4]`zero word; `[8]`reg_off lo = `0x00232004` (= `(col0<<25)|(row2<<20)|0x32004`, Core_Status); `[12]`reg_off hi=0; `[16]`value=`0x00010000`; `[20]`mask=`0x00010000`; `[24]`size=`28`. Anchor: insert immediately before the **first ctrl-in MaskWrite32 targeting tile-local `0x1d218`** (the ch1 channel-type setup preceding the first OP_READ push) — channel-robust, not instruction-count-based. Bump `insts.bin` header op-count `+1` and total byte-size `+28`. Wire into the bridge flow so EMU and HW run the **identical patched binary** (byte-parity). **TDD** (`cargo test`/script test): round-trip a known `insts.bin` — headers correct, the inserted op parses back as `MaskPoll` with exact operands, anchor located, idempotent (refuses or no-ops on double-inject). If the anchor (`0x1d218` MaskWrite32) is not found in a compiled stream, STOP and report.

- [ ] **Step 4: Emulator graceful poll-termination (TDD) + `test.cpp` verdict.**
  - **Emulator (xdna-emu Rust):** when a `BlockedOnPoll` cannot be satisfied and the engine is otherwise quiescent (core halted/idle, no monotonic progress), end the run with a distinct terminal reason `MaskPollUnsatisfied`; the FFI/`run.wait()` path returns a **clean completion (not a hang)**. It **must not** fake the polled register, pretend the core halted, or skip to the OP_READ. Read the actual run-loop / stall-detector / `syncs_satisfied` / FFI-completion interaction (`crates/xdna-emu-ffi/src/execution.rs:~205`, `src/npu/executor.rs` `BlockedOnPoll`) — **do not assume**; grounding indicates the loop already breaks on stall and the real hang is host `run.wait()` blocked on the never-completed ctrl-out DMA, so the fix likely centers on FFI completion + surfacing the terminal reason. **TDD first**, `cargo test --lib`: (a) poll satisfied immediately → proceeds; (b) poll satisfied after N cycles → proceeds; (c) poll never satisfied + engine quiescent → deterministic `MaskPollUnsatisfied`, no hang, polled register untouched, no OP_READ issued.
  - **`test.cpp` verdict:** recognize "OP_READ responses absent because MASKPOLL unsatisfied" as a distinct **expected EMU** outcome → emit `TRAP_VERDICT:MASKPOLL_UNSATISFIED_EMU` → bridge `PASS` for EMU. The HW path (core halts → MASKPOLL satisfies → OP_READ runs) computes the existing schedule-derived G1 verdict **unchanged**. Never misreport MASKPOLL-unsatisfied as `BEFORE_COMMIT`/`AFTER_COMMIT`/`NO_TRAP_OR_RAN_TO_END`.

- [ ] **Step 5: EMU validation.** `cd /home/triple/npu-work/xdna-emu && ./scripts/emu-bridge-test.sh debug_halt_probe --chess-only --no-hw 2>&1 | tee /tmp/claude-1000/probe-exp1-maskpoll-emu.log`. Expect: identical injected binary; gate fed; route disjoint; MASKPOLL never satisfied (EMU can't arm — write-side gap); emulator terminates **deterministically, no hang/timeout**; `TRAP_VERDICT:MASKPOLL_UNSATISFIED_EMU`; bridge `PASS`. Re-confirm TRAP_PC=`0x184` (the injector adds an instruction but does not touch the core ELF). `cargo test --lib` green (emulator + injector tests). Record exact `SLOTS:`/`CORE_STATUS`/`TRAP_VERDICT`/PASS lines. If EMU hangs/timeouts, the graceful-termination contract is unmet — debug before any HW; do not proceed to Task 5b.

- [ ] **Step 6: README + commits.** Update the probe README (gate + ch1 + MASKPOLL halt-sync mechanism; the `MASKPOLL_UNSATISFIED_EMU` baseline meaning; that the G1 answer is the Task 5b HW run). Two commits, messages ending `Generated using Claude Code.` (internal — no pre-approval): (a) **xdna-emu (dev)**: injector tool + emulator graceful-poll-termination + tests; (b) **mlir-aie (`xdna-emu-cycle-budget`)**: `test.cpp` verdict + README (`aie.mlir` gate/ch1 already committed `9a12651d99`).

**Report:** injector design + byte-exact MASKPOLL emitted + anchor; emulator change locus + the run.wait/terminal-reason mechanism found *in code*; TDD test list + `cargo test --lib` green; exact EMU `SLOTS:`/`CORE_STATUS`/`TRAP_VERDICT:MASKPOLL_UNSATISFIED_EMU`/PASS lines + no-hang confirmed; TRAP_PC=0x184 re-confirmed; commit SHAs; preservation confirmations (SPDX, aie-rt citations, no @out0/lock-gated-DMA, re-derivation warnings, @gate on ch0); concerns.

---

## Task 5b: Experiment 1 — hardware run, derive G1, open the findings doc [HARDWARE FORK]

**Goal:** Run the gated+armed probe on the real NPU; the gate guarantees arming-before-core-run; the core halts at the trap; control-packet OP_READ reads `output_buffer` + `Core_Status` while halted; the schedule-derived verdict yields the authoritative G1 (before/after-commit). This is the first exercise of read-while-halted (EMU could not validate it).

**Files:** Create `xdna-emu/docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md`

- [ ] **Step 1: Pre-flight the hardware.** No other HW suite running. `xrt-smi validate` (expect pass). If it fails: `pkexec sh -c 'modprobe -r amdxdna && modprobe amdxdna'`, re-validate. Do not proceed until healthy.

- [ ] **Step 2: Run the armed probe on hardware**

`cd /home/triple/npu-work/xdna-emu && ./scripts/emu-bridge-test.sh debug_halt_probe --chess-only 2>&1 | tee /tmp/claude-1000/probe-exp1-hw.log`

The injected MASKPOLL blocks the instruction stream until `Core_Status[16]` (`DEBUG_HALT`); the OP_READ therefore executes **only after the core has halted** (synchronization-ordered). Record the exact `SLOTS:` / `CORE_STATUS` / `TRAP_VERDICT:` lines from the **hardware** run. Interpretation:
- `AFTER_COMMIT` → silicon halts after the trap bundle commits → the emulator's post-`update_pc` model is **proven correct** for sync traps.
- `BEFORE_COMMIT` → silicon halts before the trap bundle commits → emulator after-commit model is a **real Phase B fidelity fix**.
- **MASKPOLL never satisfies (core does not halt on HW) → the stream blocks forever (no-timeout MASKPOLL) and the device wedges; there is no clean `NO_TRAP` verdict in this design.** This is the §7-bounded path: recover (`pkexec sh -c 'modprobe -r amdxdna && modprobe amdxdna'`, staged per CLAUDE.md), retry **once**. A **second** hang → STOP; do **not** redesign the probe again; scope G1 as the **§8 forward-commitment** (ship the emulator's documented after-commit model as the explicit assumption) and surface to Maya with raw logs.
- `ANOMALY_NOT_HALTED` / `AMBIGUOUS` (core halted, MASKPOLL satisfied, but slots inconsistent) → record raw `SLOTS:`/`CORE_STATUS` verbatim; do not force a conclusion.

Wedge protocol: with MASKPOLL the OP_READ is **gated by the core halting** (it no longer completes independently of the core). The success path is: core halts → MASKPOLL satisfies → OP_READ + verdict land before any `run.wait()` timeout. A non-halting core manifests as a MASKPOLL hang/wedge (handled above), not a benign incomplete read. Pre-flight `xrt-smi validate`; never run a second HW suite concurrently; no `xrt-smi` during the run. One recovery + one retry only; second wedge → stop, §8, surface to Maya.

- [ ] **Step 3: Write the findings doc (G1 section)**

Create the findings doc:

```markdown
# Findings: debug_halt halt-timing (G1) and single-step-count (G2)

Source: Phase A hardware probe (`mlir-aie/test/npu-xrt/debug_halt_probe`),
spec `docs/superpowers/specs/2026-05-18-debug-halt-design.md`. Ground
truth = real NPU1 (Phoenix) hardware. Observation: control-packet
OP_READ of output_buffer + Core_Status while the core is halted.
Verdict: computed from the disassembled schedule (not source order).

## G1 — Breakpoint / single-step halt timing

PC_Event0 armed at trap bundle 0x114 (stores 0xBB to output_buffer[1]),
Debug_Control2[0]=1. Schedule: trap 0x114->s[1]=0xBB; strictly-later
0x11c->s[0]=0xAA; later s[3]=1; later s[2]=0xCC.

- HW observed: `<paste exact SLOTS / CORE_STATUS / TRAP_VERDICT line>`
- EMU baseline (mechanism check, cannot halt): `<paste exact line>`

**Conclusion:** On silicon a synchronous PC-event breakpoint halts
**<BEFORE|AFTER>** the trap bundle commits. <One sentence: emulator
model proven correct / Phase B fidelity fix; plus the recorded EMU
finding that control-packet writes to debug regs are dropped by the
write_core_register catch-all -- a separate Phase B routing input.>

Raw logs: /tmp/claude-1000/probe-exp1-{readback-emu,hw}.log
(transcribed here; logs are ephemeral).

## G2 — Single_Step_Count (Debug_Control0[5:2])

(Filled in Task 7.)
```

No placeholders in the committed doc except the explicitly-marked G2 stub; fill every `<...>` with real observed data.

- [ ] **Step 4: Commit**

```bash
cd /home/triple/npu-work/xdna-emu
git add docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md
git commit -m "debug_halt findings: G1 halt-timing derived from hardware

Redesigned probe on real NPU1: control-packet read of output_buffer +
Core_Status while halted; schedule-derived verdict shows silicon halts
<BEFORE|AFTER> the trap bundle commits. Determines the synchronous-trap
halt boundary for Phase B; also records the EMU control-packet-write
routing gap as a Phase B input.

Generated using Claude Code."
```

(Substitute the real BEFORE/AFTER before committing.)

---

## Task 6: Experiment 2 kernel + config matrix — count-based single-step (EMU)

**Goal:** Reshape the probe for `Debug_Control0[5:2]` (`Single_Step_Count`) characterization, inheriting the Exp1 **`@gate`** (arming-race fix) and **ctrl-in on shim MM2S ch1** (collision fix), but **NOT the MASKPOLL** — the happens-after is a no-poll double-read OP_READ (spec §4.3; Task-7 HW-grounded the MASKPOLL out: the no-timeout `CORE_DONE` poll SMU-wedged the NPU and was an unproven happens-after). Without `@gate`, a `LANDED:8 not-halted` result is ambiguous between "count-step inert on silicon" and "armed too late", which would void G2.

**Files:** Modify `mlir-aie/test/npu-xrt/debug_halt_probe/{aie.mlir,test.cpp,README.md}`; parameterize `xdna-emu/tools/inject-maskpoll.py`.

- [ ] **Step 1: 8-marker straight-line core, gated by the Exp1 `@gate`.** Keep the Exp1 `@gate` objectfifo, packet flows, `output_lock4/5`, and structure verbatim; only the core body changes from 4 markers to 8 sequential distinct stores (`output_buffer[k]=101+k`, k=0..7). `output_buffer` stays `memref<8xi32>`. The core still acquires `@gate` first (hardware-blocking stall), then the 8 stores, then `aie.use_lock(%output_lock4, Release, 1)`, `aie.end`:

```mlir
      %gv = aie.objectfifo.acquire @gate(Consume, 1) : !aie.objectfifosubview<memref<1xi32>>
      %g = aie.objectfifo.subview.access %gv[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
      aie.objectfifo.release @gate(Consume, 1)
      %c0..%c7 = arith.constant 0..7 : index
      %v1..%v8 = arith.constant 101..108 : i32
      aie.use_lock(%output_lock5, AcquireGreaterEqual, 1)
      memref.store %v1, %output_buffer[%c0] : memref<8xi32>   // ... through
      memref.store %v8, %output_buffer[%c7] : memref<8xi32>
      aie.use_lock(%output_lock4, Release, 1)
      aie.end
```

Markers are NOT assumed to be in source order on-chip — Step 3's verdict counts committed markers, order-independent (source binds value 101+k to slot k; the scheduler only reorders commit timing, not value-to-slot). The happens-after is the double OP_READ's snapshot agreement (Step 3), NOT an injected poll and NOT a memory sentinel — no new derived constant, no device-wedge hazard.

- [ ] **Step 2: Count-step config in @seq.** Replace *only* the Exp1 PC_Event0 (`0x32020`) + Debug_Control2 (`0x32018`) arming writes with a single Debug_Control0 count write. Keep the `@gate` feed (`dma_memcpy_nd @gate`) and the entire Exp1 ctrl-in-on-ch1 OP_READ readback plumbing intact, extended to 8 marker slots (bump the read count 5→9: 8 markers + Core_Status; add 4 packet blocks mirroring the existing ones; `%c5_i64`→`%c9_i64`). Start `count=4`, no halt bit: `Debug_Control0 = (4<<2) = 0x10`:

```mlir
      // Debug_Control0: Single_Step_Count = N at bits [5:2] (aie-rt
      // XAIEMLGBL_CORE_MODULE_DEBUG_CONTROL0, xaiemlgbl_params.h:2452;
      // SINGLE_STEP_COUNT bits 2-5, DEBUG_HALT_BIT bit 0). Matrix swept
      // by editing this value (see README / findings).
      aiex.npu.write32 {address = 0x32010 : ui32, column = 0 : i32, row = 2 : i32, value = 0x10 : ui32}
```

- [ ] **Step 3: Double-read LANDED verdict in test.cpp (no MASKPOLL).** `@seq` issues the 9-packet OP_READ readback **twice** (18 packets: pass A = 8 markers + Core_Status, pass B = same; pass B is strictly after pass A in the in-order stream and after pass A's shim round-trips). `test.cpp` parses both snapshots and requires agreement (settled-state self-validation; spec §4.3). `NUM_READ_PKTS` 5→18; `sA[8]`,`csA`,`sB[8]`,`csB`:

```cpp
  auto count = [](const uint32_t* s){ int n=0; for(int k=0;k<8;k++) if(s[k]==(uint32_t)(101+k)) n++; return n; };
  int landedA = count(sA), landedB = count(sB);
  bool settled = (landedA == landedB) && (csA == csB) &&
                 !memcmp(sA, sB, sizeof(sA));
  bool halted = (csB & (1u<<16)) != 0;  // DEBUG_HALT (bit 16)
  bool done   = (csB & (1u<<20)) != 0;  // CORE_DONE  (bit 20)
  std::cout << "SLOTS:";
  for (int k=0;k<8;k++) std::cout << " " << sB[k];
  std::cout << " CORE_STATUS=0x" << std::hex << csB << std::dec
            << " HALTED=" << (halted?1:0) << " DONE=" << (done?1:0)
            << " SETTLED=" << (settled?1:0)
            << "\nLANDED:" << landedB << "\n";
  if (!settled) { std::cout << "UNSETTLED\n"; return 1; }  // core mid-exec: clean re-run, NOT a wedge
  // settled && landed==8 && !halted     -> count-step inert (ran to completion)
  // settled && halted && landed==N      -> core stopped after N committed stores (count-step active)
  // other settled                       -> record SLOTS+CORE_STATUS verbatim
  std::cout << "PASS\n";
  return 0;
```

No injected poll, so nothing can block the instruction stream forever; the OP_READ reads tile memory/registers directly regardless of core state (Task-3 finding). This is the transient Exp2 verdict; Task 8 git-restores the Exp1 `BEFORE_COMMIT` verdict. README documents the Exp1↔Exp2 switch.

- [ ] **Step 4: No-injection bridge wiring, EMU run, document.** The injector keeps its `--witness done|halt` param for Exp1, but Exp2 must inject **nothing**. Add a `none` sentinel to the bridge wiring: `_inject_maskpoll_if_probe` treats `DEBUG_HALT_PROBE_WITNESS=none` as "skip injection entirely" (echo `no MASKPOLL (Exp2 double-read)`); Exp1 default `halt` unchanged (still injects, byte-identical). Run: `cd /home/triple/npu-work/xdna-emu && DEBUG_HALT_PROBE_WITNESS=none ./scripts/emu-bridge-test.sh debug_halt_probe --chess-only --no-hw 2>&1 | tee /tmp/claude-1000/probe-exp2-emu.log`. **Expected EMU (no write-side gap):** `aiex.npu.write32` to `Debug_Control0` (`0x32010`) reaches `core_debug` via `apply_tile_local_effects` → `write_debug_control0` (mod.rs:787); EMU count-step is inert because the §5.2 state machine is unimplemented (nothing reads the stored count), NOT a dropped write. Core runs all 8 → terminal state stable → pass A == pass B → `SETTLED=1`, `LANDED:8`, `DONE=1`/`HALTED=0`, bridge `PASS`. No poll ⇒ no MASKPOLL-satisfy dependency on EMU. **If `SETTLED=0` on EMU** (snapshots disagree — EMU readback racing core completion), surface it: tune the inter-pass spacing (e.g. a benign extra OP_READ between passes), do not hand-wave. Append an "Experiment 2" section to README: minimal set (`count=0 control 0x00`, `count=4 no-halt 0x10`), the double-read no-poll mechanism, `DEBUG_HALT_PROBE_WITNESS=none`, and the Exp1↔Exp2 switch.

- [ ] **Step 5: Commit**

```bash
cd /home/triple/npu-work/xdna-emu && git add scripts/emu-bridge-test.sh
cd /home/triple/npu-work/mlir-aie && git add test/npu-xrt/debug_halt_probe/
git commit -m "debug_halt_probe: Exp2 -- count-step double-read (no MASKPOLL)

8 distinct marker stores behind the Exp1 @gate (arming-race immunity)
on ch1; @seq writes Debug_Control0 Single_Step_Count; LANDED via a
DOUBLE ctrl-in OP_READ readback, settled-state self-validated by
snapshot agreement (pass A == pass B). NO injected poll -- the Task-7
HW-grounded redesign: the no-timeout CORE_DONE MASKPOLL SMU-wedged the
NPU and was an unproven happens-after; OP_READ reads tile state
core-independently (Task-3) and is ms >> the core's us, so a stable
terminal state needs no synchronization. Cannot wedge the device.
Bridge: DEBUG_HALT_PROBE_WITNESS=none skips injection (Exp1 halt
default unchanged). EMU baseline: count-step inert (Phase B 5.2
unimplemented; write DOES reach core_debug -- no write-side gap).

Generated using Claude Code."
```

(Separate repos: the bridge-wiring change commits on xdna-emu `dev`; the probe `test.cpp`/`aie.mlir`/README on mlir-aie `xdna-emu-cycle-budget`. The injector itself is unchanged this step — its `--witness` param already serves Exp1.)

---

## Task 7: Experiment 2 — hardware sweep, derive G2 [HARDWARE FORK]

**Goal:** Run the **minimal** count-step set on the real NPU with the no-poll double-read probe; record what `Debug_Control0[5:2]` does on silicon as far as observable.

**Files:** Modify `mlir-aie/test/npu-xrt/debug_halt_probe/aie.mlir` (one value per point); `xdna-emu/docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md` (G2 section)

**Prerequisite: a reboot.** Task-7's first HW attempt SMU-wedged the NPU (the now-retired no-timeout `CORE_DONE` MASKPOLL); the device needs a reboot to clear it (kernel is healthy — the timed-out-message driver fix is verified). Maya triggers the reboot. The redesigned probe (no injected poll) **cannot** SMU-wedge the device — worst case is a TDR with clean driver recovery.

- [ ] **Step 1: Pre-flight hardware** (`xrt-smi validate`; recover per CLAUDE.md if needed). Confirm the fixed `amdxdna` is loaded (`./build.sh -release -refresh_dkms` if a fresh build is needed).

- [ ] **Step 2: Run the minimal set on hardware (no injection).** Two points only: `count=0 control 0x00` and `count=4 no-halt 0x10` (spec §4.3 — the decisive inert-vs-active contrast; the full 5-point sweep is explicitly *not* run). For each: set the `Debug_Control0` value in `aie.mlir`, run `DEBUG_HALT_PROBE_WITNESS=none ./scripts/emu-bridge-test.sh debug_halt_probe --chess-only --compile 2>&1 | tee /tmp/claude-1000/probe-exp2-hw-<cfg>.log` from `xdna-emu/`. Record `SLOTS:`/`CORE_STATUS`/`HALTED`/`DONE`/`SETTLED`/`LANDED:`.
  - `SETTLED=1` is required for a usable datapoint; `UNSETTLED` ⇒ re-run once (clean, not a wedge — see Step 4 spacing note). **Operator note:** an `UNSETTLED` run exits non-zero, so the bridge result column shows `FAIL` even though nothing is wrong — always read the log: `FAIL` + `UNSETTLED` in stdout = the benign re-run signal, distinct from a genuine failure. (3-state outcome mapped onto the bridge's 2-state token; the log carries the real verdict.)
  - **`0x00` settled, `LANDED:8`, not halted** (core ran to completion) AND **`0x10` settled, `LANDED:N<8` + halted/stopped** ⇒ count-step is **active** on silicon (the partial signal from Task-7 attempt 1, now cleanly grounded).
  - **Both settled, `LANDED:8`, not halted** ⇒ count-step **inert** on silicon (the spec §1/§7 hypothesis).
  - A TDR (firmware hang unrelated to a poll) ⇒ clean driver recovery (verified), `xrt-smi validate`, retry once; second TDR ⇒ record "indeterminate", §8 posture, do not redesign again.

- [ ] **Step 3: Fill the G2 findings section.** Table (`0x00`/`0x10` → EMU LANDED → HW LANDED/HALTED/DONE/SETTLED → interpretation) + the active-vs-inert conclusion. Whatever the minimal set does not resolve (decrement/expire/re-arm/halt-bit interaction — the unrun points) is an explicit Phase B documented modeling decision + the **§8 count-step forward-commitment** (the expected disposition: substrate-only, no binary depends on it). Also record the Task-7 detour: the no-timeout `CORE_DONE` MASKPOLL HW-wedge, the kernel NULL-deref/UAF it exposed, the verified `amdxdna` fix, and the resulting probe redesign (this is significant durable content — cross-link the driver fix).

- [ ] **Step 4: Commit**

```bash
cd /home/triple/npu-work/xdna-emu
git add docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md
cd /home/triple/npu-work/mlir-aie
git add test/npu-xrt/debug_halt_probe/aie.mlir
git commit -m "debug_halt findings: G2 count-step -- minimal HW set (no-poll double-read)

0x00 vs 0x10 on real NPU1 via the redesigned no-injection double-read
probe. Active-vs-inert conclusion; remaining edges -> Phase B modeling
decisions + Section 8 forward-commitment. Records the Task-7 detour:
no-timeout CORE_DONE MASKPOLL HW-wedge -> amdxdna NULL-deref/UAF (fixed,
verified) -> probe redesign.

Generated using Claude Code."
```

---

## Task 8: Finalize findings, restore the probe to a re-runnable state, regroup checkpoint

**Goal:** Leave the probe a clean permanent regression artifact, findings complete, explicit handoff to the Phase B planning regroup.

**Files:** Modify `mlir-aie/test/npu-xrt/debug_halt_probe/{aie.mlir,README.md}`; `xdna-emu/docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md`

- [ ] **Step 1: Restore the probe to the post-Unit-1b Exp1 self-checking state.** *Corrected — the real regression-valuable Exp1 state is NOT the pre-Unit-1b "Task 4 end-state":* it is the MASKPOLL-synchronized `BEFORE_COMMIT` config (`aie.mlir` at `9a12651d99`, `test.cpp` at `830972966c`, with the injected `--witness halt` DEBUG_HALT MASKPOLL). Since Task 6 only transiently edits these files, restore by `git checkout 830972966c -- test/npu-xrt/debug_halt_probe/test.cpp` and `git checkout 9a12651d99 -- test/npu-xrt/debug_halt_probe/aie.mlir` (verify against `git log` HEADs at restore time). Rationale: Exp1 is the deterministic self-checking EMU+HW regression; Exp2's matrix is a swept investigation. README documents how to switch to the Exp2 core for re-investigation.

- [ ] **Step 2: EMU re-run** confirms the restored probe reproduces its recorded post-Unit-1b EMU baseline (`SLOTS: s0=0 s1=0 s2=0 s3=0`, `CORE_STATUS=0x10003`, `HALTED=1`, `TRAP_VERDICT:BEFORE_COMMIT`, `PASS`): `./scripts/emu-bridge-test.sh debug_halt_probe --chess-only --no-hw 2>&1 | tee /tmp/claude-1000/probe-final-emu.log`.

- [ ] **Step 3: Finalize the findings doc.** Closing "Phase B inputs" section, direct no-placeholder instructions to Phase B: (a) the synchronous-trap halt boundary from G1 — **shipped** by Phase B Units 1/1b (record as done, not open); (b) count-step semantics from G2 + the explicit documented-modeling-decision list for unobservable edges; (c) *Corrected — there is **no** control-packet→core_debug write-side routing gap:* Units 1/1b already closed the read-path reconciliation (the writes always reached `core_debug` via `apply_tile_local_effects`; spec §4.2 "Mechanism correction"). The real remaining Phase B work item is the **§5.2 count-step state machine** (the stored `Debug_Control0[5:2]` drives nothing — no consumer) plus the **§5.1 single-step halt boundary** (the deferred, G2-coupled half); (d) whether the §8 count-step forward-commitment is triggered (likely — see Task 7 Step 3).

- [ ] **Step 4: Commit and stop at the regroup**

```bash
cd /home/triple/npu-work/mlir-aie && git add test/npu-xrt/debug_halt_probe/ && \
git commit -m "debug_halt_probe: restore to Exp1 redesigned regression state

Probe left in the deterministic Exp1 control-packet-readback config as a
permanent re-runnable regression; README documents the Exp2 switch.

Generated using Claude Code." && \
cd /home/triple/npu-work/xdna-emu && \
git add docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md && \
git commit -m "debug_halt findings: finalize -- Phase B inputs section

Closes Phase A. No-placeholder Phase B inputs: G1 halt boundary (shipped
Units 1/1b), G2 count-step + modeling decisions, the open work items
(5.2 count-step state machine + 5.1 single-step halt boundary; no
write-side routing gap -- Units 1/1b closed read reconciliation),
Section 8 trigger status. Regroup before Phase B remainder.

Generated using Claude Code."
```

Then **stop**. Phase A is complete. Do not begin the Phase B remainder. Surface to Maya: Phase A findings (G1 verdict — already shipped via Units 1/1b; G2 characterization; the §5.2/§5.1 open Phase B work items; §8 status), and that the next step is the regroup before the Phase B remainder plan — her call when, per the plan→execute→regroup→next-plan rhythm.

---

## Self-review

**(Revised 2026-05-18 for the post-Task-3 redesign; spec §4 updated in lockstep.)**

**Spec coverage (spec Section 4 = Phase A):**
- 4.1 instrument (control-packet readback while halted is load-bearing per revised §4.1; EMU-first then HW; Exp1 before Exp2; recovery staged) → Tasks 1 (authoring mechanism), 4 (readback mechanism + no-trap baseline), 5 (Exp1 HW), 6 (Exp2 EMU), 7 (Exp2 HW). Covered.
- 4.2 Experiment 1, redesigned (4 markers; trap PC from `llvm-objdump-aie`; PC_Event0+Debug_Control2; **schedule-derived** verdict + Core_Status disambiguation; control-packet OP_READ while halted) → Tasks 2 (kernel), 3 (arming + disasm-derived TRAP_PC + re-derivation discipline), 4 (readback + verdict), 5 (HW derive G1). The "force store order" idea was investigated and rejected (no toolchain pin; empirically confirmed by our own disasm); schedule-derived verdict is the spec-blessed approach.
- 4.3 Experiment 2 (8 markers, Debug_Control0 minimal set) → Tasks 6,7. Inherits the Exp1 `@gate` (arming-race) + ctrl-in-on-ch1 (collision) fixes but **NOT the MASKPOLL** — Task-7 HW-grounded the no-timeout `CORE_DONE` poll out (it SMU-wedged the NPU, exposed the `amdxdna` NULL-deref/UAF, and was an unproven happens-after). Happens-after = no injected poll; latency-ordered **double OP_READ** self-validated by snapshot agreement (stable terminal state; OP_READ is core-state-independent — Task 3 — and ms ≫ the core's µs). Cannot wedge the device. HW = minimal `0x00`/`0x10` set, not the full sweep. No write-side routing gap (Units 1/1b; spec §4.2).
- 4.4 output (findings doc + permanent re-runnable probe) → Tasks 5,7,8. Covered.

**Superseded scope reductions (no longer open):** the earlier "ReadRegisters omitted as YAGNI" reduction is reversed — now the primary observation, justified by the discovered flaw. The post-resume hardware check stays out of Phase A but is a *tracked spec §8 forward-commitment* (resume hardware-verification), not an unflagged drop. Both are resolved in the spec, not merely "surfaced at regroup".

**New Phase B input captured:** the EMU control-packet→`core_debug` register-write routing gap (debug-reg writes hit the `write_core_register` catch-all and are dropped) is recorded as an explicit Phase B work item in the findings-doc Phase B inputs (Task 8 Step 3) — discovered, not designed-around.

**Placeholder scan:** No "TBD/TODO/handle appropriately". Intentional fill-ins (`<BEFORE|AFTER>` verdict, `OUTBUF_ADDR`, LANDED counts) are *experimental observations the executor records / derives*, each with an exact derivation step and a "substitute the real value before committing" instruction — a probe plan's outputs are data, not code.

**Type/identifier consistency:** Test dir `debug_halt_probe`, tile `(0,2)`, offsets `0x32010/0x32018/0x32020`, `Core_Status` `0x32004`, `OUTBUF_ADDR` (derived, expected `0x0400`), trap bundle `0x114`→slot1=0xBB / strictly-later `0x11c`→slot0=0xAA consistent across spec §4.2, the aie.mlir TRAP_PC comment, and `test.cpp`; verdict tokens (`AFTER_COMMIT`/`BEFORE_COMMIT`/`NO_TRAP_OR_RAN_TO_END`/`AMBIGUOUS`/`ANOMALY_NOT_HALTED`/`LANDED`/`PASS`) consistent between `test.cpp` and the findings doc. Bridge invocation + recovery commands match CLAUDE.md.

---

## Phase B Unit 1: synchronous PC-event halt boundary (before-commit) + core_debug control-packet routing gaps

> **Gate crossed deliberately.** Phase B was "Plan 2, later." This is one scoped, fully-derived, hardware-free unit pulled forward because G1 (Task 5b) unambiguously derived *before-commit* and this is the highest-leverage follow-through. Single-step / count-step / G2 remain deferred. Authority: spec §5.1, §5.3, §6 (committed coherence). Subagent-driven, TDD, two-stage review — same discipline as Tasks 4-5.

**Goal:** Make a synchronous PC_Event/breakpoint halt take effect *before* the trap bundle's side effects commit (matching silicon, per the G1 finding), and wire the coupled `core_debug` control-packet register routing gaps so the behavior is testable end-to-end on EMU. PC_Event/breakpoint origin only; async paths provably untouched; single-step boundary explicitly out of scope.

- [ ] **Step 1: Routing-gap fix (enables the end-to-end test).** Dispatch control-packet debug-register writes (`0x32010`–`0x3202C`) into `core_debug` instead of the silent `write_core_register` `_ => {}` (`compute.rs:554`), and the symmetric read path into `core_debug` (`read_register_pure`, `registers.rs:95`). TDD: control-packet write of PC_Event0/Debug_Control2 then read-back round-trips through `core_debug`; non-debug regs unaffected. Cite aie-rt offsets.

- [ ] **Step 2: Pre-execute PC_Event seam.** In `coordinator.rs` (before `step_with_neighbor_locks`, ~:624): if the next PC matches an armed PC_Event with `PC_Event_Halt`, halt **without executing the bundle** (the trap store never lands). Add a non-committing `core_debug` query (e.g. `has_sync_pc_trap_at(pc)`); condition the existing post-execute `update_pc` (:638) so the same match does not re-fire after resume. Async halt paths and the `interpreter.rs:181` gate unchanged — verify by argument and by the existing async tests staying green.

- [ ] **Step 3: Guarding test (approach decided up front).** Coordinator-level: arm PC_Event0 at a `TRAP_PC` whose bundle stores to a known tile data address, step, assert `DebugHalt` AND the store did **not** land (before-commit). Resolve the encoded-store-bundle dependency per spec §6 in order (a) reuse an existing compiled fixture; (b) hand-encode from llvm-aie ISA + document; (c) fall back to a state-machine-level assertion and explicitly record the literal store-not-landed claim as hardware-probe-covered. **Surface which of (a)/(b)/(c) before settling for (c).**

- [ ] **Step 4: Validate + coverage + commit.** `cargo test --lib` green (xdna-emu + xdna-archspec); existing async/SSTEP tests (`core_debug/tests.rs:989,1046`, the watchpoint test) still pass (they assert untouched behavior). Regenerate coverage artifacts (`gen_coverage_artifacts`), zero drift; completeness stays `< Full` (G1 + routing closed; G2/single-step still open). Commit (xdna-emu `dev`), message ending `Generated using Claude Code.`. Then two-stage review (spec-compliance + code-quality, fresh sonnet) + controller adjudication, same as Task 5.

**STOP at:** any need to move single-step/count-step (out of scope — defer to §5.2/G2); guarding-test fallback (c) without surfacing first; async-path behavior change (must be provably untouched).

---

## Phase B Unit 1b: reconcile the mutable read path — EMU reproduces G1 (self-checking regression)

> **Grounded follow-on.** The Unit-1 review + a dedicated grounding established the spec's "write-side gap" was false (control-packet *and* `@seq npu.write32` debug-reg writes always reached `core_debug` via `apply_tile_local_effects`; arming was wired). The Task-5 `MASKPOLL_UNSATISFIED_EMU` was caused solely by the **mutable `tile.read_register`** path (the injected-MASKPOLL `Core_Status` poll) falling back to the raw HashMap and never reflecting the dynamically-computed `DEBUG_HALT`. Unit 1 fixed only `read_register_pure` (control-packet OP_READ path). Unit 1b reconciles the mutable path the same way, closing the recorded §4.2 "inconsistent register-access paths" input. Consequence: arming (wired) + Unit-1 before-commit seam + Unit-1b read fix ⇒ **EMU reproduces the HW `BEFORE_COMMIT`**, turning the probe into a self-checking EMU+HW regression of the G1 fidelity fix. Authority: spec §4.2 "Mechanism correction"/"Phase B Unit 1b", §5.1/§5.3/§6 (coherence committed this cycle). Subagent-driven, TDD, two-stage review.

- [ ] **Step 1: Reconcile the mutable read path.** Dispatch `tile.read_register` (mutable) `Core_Status` (0x32004) + debug-reg (`0x32010`–`0x3202C`) reads into `core_debug` (live-computed status, DEBUG_HALT bit 16), mirroring exactly what Unit 1 (`e0ec922`) did for `read_register_pure`. Do NOT diverge the two paths' semantics — they must now agree. TDD: mutable `read_register(0x32004)` reflects `core_debug` halt state and equals `read_register_pure(0x32004)` for the same state; non-debug regs unaffected.

- [ ] **Step 2: Flip the probe's EMU verdict expectation.** `mlir-aie .../debug_halt_probe/test.cpp`: the EMU path now reaches the same schedule-derived verdict as HW (`BEFORE_COMMIT`) — the `MASKPOLL_UNSATISFIED_EMU` branch is no longer the probe's EMU outcome. Update the verdict logic/comments so EMU and HW both assert `BEFORE_COMMIT` (sentinel/`MASKPOLL_UNSATISFIED_EMU` handling may remain as a defensive branch but must NOT be the expected EMU result). HW path unchanged. Keep SPDX/citations.

- [ ] **Step 3: EMU bridge validation.** Rebuild FFI; `TMPDIR=/tmp/claude-1000 ./scripts/emu-bridge-test.sh debug_halt_probe --chess-only --no-hw 2>&1 | tee /tmp/claude-1000/probe-unit1b-emu.log`. Expect: identical injected binary, gate fed, MASKPOLL now **satisfies** on EMU (DEBUG_HALT observable), OP_READ issues, all marker slots zero + DEBUG_HALT ⇒ `TRAP_VERDICT:BEFORE_COMMIT`, bridge `PASS`, deterministic (no hang). Record exact `SLOTS:`/`CORE_STATUS`/`TRAP_VERDICT`/PASS lines. The emulator graceful-poll-termination unit tests must still be green (contract retained, just not this probe's path). If EMU does not reach `BEFORE_COMMIT`, STOP and report (read reconciliation or seam interaction issue).

- [ ] **Step 4: Findings-doc coherence + coverage + commit.** Update `findings/2026-05-18-debug-halt-timing-and-single-step-count.md` EMU section: retire the `MASKPOLL_UNSATISFIED_EMU`/HW-only-by-construction framing; record that EMU now reproduces `BEFORE_COMMIT` and the probe is a self-checking EMU+HW regression (HW remains ground truth). `cargo test --lib` green (xdna-emu + xdna-archspec); regenerate coverage artifacts, zero non-narrative drift; completeness stays `< Full` (G2/single-step still open). Two commits: xdna-emu `dev` (read reconciliation + tests + findings doc + coverage); mlir-aie `xdna-emu-cycle-budget` (`test.cpp` verdict). Messages end `Generated using Claude Code.` Then two-stage review (spec-compliance + code-quality, fresh sonnet) + controller adjudication.

**STOP at:** the two read paths ending up with *different* semantics (they must agree — that is the whole point); any change to single-step/count-step/async; EMU not reproducing `BEFORE_COMMIT` (do not force the verdict — surface it).
