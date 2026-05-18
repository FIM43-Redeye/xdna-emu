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

## Task 4: Experiment 1 — hardware run, derive G1, open the findings doc

**Goal:** Run the armed probe on the real NPU, read the markers back, determine before-vs-after-commit on silicon, and record it as the authoritative G1 finding.

**Files:**
- Create: `xdna-emu/docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md`

- [ ] **Step 1: Pre-flight the hardware**

Confirm no other HW suite is running. Smoke-test the device: `xrt-smi validate` (expect pass). If it fails, recover: `pkexec sh -c 'modprobe -r amdxdna && modprobe amdxdna'`, re-validate. Do not proceed until the device is healthy.

- [ ] **Step 2: Run the armed probe on hardware**

Run: `cd /home/triple/npu-work/xdna-emu && ./scripts/emu-bridge-test.sh debug_halt_probe --chess-only 2>&1 | tee /tmp/claude-1000/probe-exp1-hw.log`
(no `--no-hw` → HW run; the script also reruns EMU for comparison.)

Read the HW `test.exe` output (`build/bridge-test-results/latest/debug_halt_probe.chess/hw/` or the tee'd log). Record the exact `MARKERS:` and `TRAP_VERDICT:` lines from the **hardware** run.

Interpretation:
- `TRAP_VERDICT:AFTER_COMMIT` → silicon halts after the trap bundle commits. The emulator's current model is **proven correct**.
- `TRAP_VERDICT:BEFORE_COMMIT` → silicon halts before the trap bundle commits. The emulator's after-commit model is a **real fidelity bug** for synchronous traps (Phase B fix).
- `AMBIGUOUS` / core ran to end → the breakpoint did not take effect on HW; debug the arming (ordering, PC mask, Debug_Control2 bit) and rerun. Do not record an ambiguous result as the finding.

If the NPU wedges during the run, recover per the hardware-run guard and rerun once; if it wedges again, stop and surface to Maya.

- [ ] **Step 3: Write the findings doc (G1 section)**

Create `xdna-emu/docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md`:

```markdown
# Findings: debug_halt halt-timing (G1) and single-step-count (G2)

Source: Phase A hardware probe (`mlir-aie/test/npu-xrt/debug_halt_probe`),
derived per spec `docs/superpowers/specs/2026-05-18-debug-halt-design.md`.
Ground truth = real NPU1 (Phoenix) hardware.

## G1 — Breakpoint / single-step halt timing

Experiment: PC_Event0 armed at the trap bundle (the bundle that stores
0xBB to output slot 1), Debug_Control2[0]=1. Markers: out0=pre-trap,
out1=trap-bundle, out2=post-trap, out3=ran-to-end sentinel.

- EMU observed (current model): `<paste exact MARKERS/TRAP_VERDICT line>`
- HW observed (ground truth):  `<paste exact MARKERS/TRAP_VERDICT line>`

**Conclusion:** On silicon, a synchronous PC-event breakpoint halts
**<BEFORE|AFTER>** the trap bundle commits. <One sentence: emulator
model is proven correct / is a fidelity bug to fix in Phase B.>

Raw logs: /tmp/claude-1000/probe-exp1-{emu,hw}.log (transcribed here;
logs are ephemeral).

## G2 — Single_Step_Count (Debug_Control0[5:2])

(Filled in Task 6.)
```

Fill every `<...>` with the real observed data — no placeholders left in the committed doc except the explicitly-marked G2 stub.

- [ ] **Step 4: Commit**

```bash
cd /home/triple/npu-work/xdna-emu
git add docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md
git commit -m "debug_halt findings: G1 halt-timing derived from hardware

PC_Event breakpoint probe on real NPU1: silicon halts <BEFORE|AFTER>
the trap bundle commits. EMU vs HW verdicts recorded. Determines the
synchronous-trap halt boundary for Phase B.

Generated using Claude Code."
```

(Substitute the real BEFORE/AFTER before committing.)

---

## Task 5: Experiment 2 kernel + config matrix — count-based single-step

**Goal:** Reshape the probe to characterize `Debug_Control0[5:2]` (`Single_Step_Count`) as far as silicon reveals: a known-length straight-line marker run, plus a small config matrix written via control packets.

**Files:**
- Modify: `mlir-aie/test/npu-xrt/debug_halt_probe/aie.mlir` (core: N sequential markers; runtime sequence: Debug_Control0 count write)
- Modify: `mlir-aie/test/npu-xrt/debug_halt_probe/test.cpp` (count how many markers landed)

- [ ] **Step 1: N-marker straight-line core**

Replace the core body with 8 sequential distinct stores so "how many committed before halt" is directly countable:

```mlir
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %v1 = arith.constant 101 : i32
      %v2 = arith.constant 102 : i32
      %v3 = arith.constant 103 : i32
      %v4 = arith.constant 104 : i32
      %v5 = arith.constant 105 : i32
      %v6 = arith.constant 106 : i32
      %v7 = arith.constant 107 : i32
      %v8 = arith.constant 108 : i32
      aie.use_lock(%output_lock5, AcquireGreaterEqual, 1)
      memref.store %v1, %output_buffer[%c0] : memref<8xi32>
      memref.store %v2, %output_buffer[%c1] : memref<8xi32>
      memref.store %v3, %output_buffer[%c2] : memref<8xi32>
      memref.store %v4, %output_buffer[%c3] : memref<8xi32>
      memref.store %v5, %output_buffer[%c4] : memref<8xi32>
      memref.store %v6, %output_buffer[%c5] : memref<8xi32>
      memref.store %v7, %output_buffer[%c6] : memref<8xi32>
      memref.store %v8, %output_buffer[%c7] : memref<8xi32>
      aie.use_lock(%output_lock4, Release, 1)
      aie.end
    }
```

- [ ] **Step 2: Count-step config in the runtime sequence**

Replace the Task 3 breakpoint arming in `@seq` with a single Debug_Control0 count write. The matrix is swept by editing this one value and rebuilding (documented in Step 4); start with `count=4`, count alone (no halt bit): `Debug_Control0 = (4 << 2) = 0x10`:

```mlir
      // Debug_Control0: Single_Step_Count = N at bits [5:2], halt bit [0]=0.
      // Matrix is swept by editing this value (see README / findings).
      aiex.npu.write32 {address = 0x32010 : ui32, column = 0 : i32, row = 2 : i32, value = 0x10 : ui32}
```

- [ ] **Step 3: Marker-count verdict in `test.cpp`**

Replace the verdict block:

```cpp
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  uint32_t *out = bo_out.map<uint32_t *>();
  int landed = 0;
  for (int i = 0; i < 8; i++)
    if (out[i] == (uint32_t)(101 + i)) landed++;
  std::cout << "MARKERS:";
  for (int i = 0; i < 8; i++) std::cout << " " << out[i];
  std::cout << "\nLANDED:" << landed << "\n";
  // landed==8 -> count-step had no effect (all bundles ran)
  // landed==N -> core halted after exactly N committed stores
  // 0<landed<8, !=N -> partial/other; record verbatim
  return 0;
```

- [ ] **Step 4: EMU run + document the sweep matrix in README**

Run: `cd /home/triple/npu-work/xdna-emu && ./scripts/emu-bridge-test.sh debug_halt_probe --chess-only --no-hw 2>&1 | tee /tmp/claude-1000/probe-exp2-emu.log`

Record `LANDED:` for the `count=4, no-halt-bit` config. Append a "Experiment 2 sweep matrix" section to `debug_halt_probe/README.md` listing the configs to run on HW in Task 6, each with the exact `Debug_Control0` value:
- `count=4, halt bit 0`: `0x10`
- `count=4, halt bit 1`: `0x11`
- `count=2, halt bit 0`: `0x08`
- `count=8 (max-ish), halt bit 0`: `0x20`
- `count=0`: `0x00` (control — must run all 8)

- [ ] **Step 5: Commit**

```bash
cd /home/triple/npu-work/mlir-aie && git add test/npu-xrt/debug_halt_probe/ && \
cd /home/triple/npu-work/xdna-emu && git add docs/ 2>/dev/null; \
cd /home/triple/npu-work/mlir-aie && git commit -m "debug_halt_probe: Exp2 -- N-marker count-step kernel + config matrix

8 sequential distinct marker stores; runtime sequence writes
Debug_Control0 Single_Step_Count. test.cpp reports LANDED count.
EMU baseline recorded; HW sweep matrix documented in README.

Generated using Claude Code."
```

(README lives with the probe in `mlir-aie`; the `xdna-emu` git add is a no-op safety if the README were mirrored — the canonical README is in the probe dir.)

---

## Task 6: Experiment 2 — hardware sweep, derive G2

**Goal:** Run the count-step config matrix on the real NPU and record what `Debug_Control0[5:2]` actually does on silicon, as far as it is observable.

**Files:**
- Modify: `mlir-aie/test/npu-xrt/debug_halt_probe/aie.mlir` (one value per sweep point)
- Modify: `xdna-emu/docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md` (G2 section)

- [ ] **Step 1: Pre-flight hardware** (same as Task 4 Step 1: `xrt-smi validate`, recover if needed.)

- [ ] **Step 2: Run each matrix point on hardware**

For each config in the README matrix: set the `Debug_Control0` value in `aie.mlir`, then run `cd /home/triple/npu-work/xdna-emu && ./scripts/emu-bridge-test.sh debug_halt_probe --chess-only 2>&1 | tee /tmp/claude-1000/probe-exp2-hw-<cfg>.log`. Record `LANDED:` and the full `MARKERS:` line for each. Recover the NPU between points only if it wedges (per the hardware-run guard); a clean run needs no recovery. If a wedge recurs on the same config, record "config <X> wedges the device" as itself a finding and skip to the next.

- [ ] **Step 3: Fill the G2 section of the findings doc**

Replace the G2 stub with a results table (config → EMU LANDED → HW LANDED → interpretation) and a conclusion stating, in plain terms, what is established about arm/decrement/expire/halt-bit-interaction and what remains unobservable. Explicitly mark the unobservable parts as "to be a documented modeling decision in Phase B" and note the Section 8 forward-commitment (silicon-fidelity revisit) applies if behavior is under-characterized.

- [ ] **Step 4: Commit**

```bash
cd /home/triple/npu-work/xdna-emu
git add docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md
cd /home/triple/npu-work/mlir-aie
git add test/npu-xrt/debug_halt_probe/aie.mlir
git commit -m "debug_halt findings: G2 count-step characterized on hardware

Debug_Control0[5:2] config matrix swept on real NPU1. Results table +
conclusion recorded; unobservable edges flagged for Phase B documented
modeling decisions / Section 8 forward-commitment.

Generated using Claude Code."
```

---

## Task 7: Finalize findings, restore the probe to a re-runnable state, regroup checkpoint

**Goal:** Leave the probe as a clean permanent regression artifact, the findings doc complete, and an explicit handoff to the Phase B planning regroup.

**Files:**
- Modify: `mlir-aie/test/npu-xrt/debug_halt_probe/{aie.mlir,README.md}`
- Modify: `xdna-emu/docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md`

- [ ] **Step 1: Restore the probe to the Experiment 1 armed state**

Set `aie.mlir` back to the Experiment 1 configuration (Task 3: PC_Event0 armed at `TRAP_PC14`, Debug_Control2[0]=1, the 4-marker pre/trap/post/done core). Rationale: Exp 1 is the deterministic, low-risk, regression-valuable configuration; Exp 2's matrix is a swept investigation, not a single regression state. README documents how to switch to the Exp 2 core for re-investigation.

- [ ] **Step 2: EMU re-run confirms the restored probe still passes its known verdict**

Run: `cd /home/triple/npu-work/xdna-emu && ./scripts/emu-bridge-test.sh debug_halt_probe --chess-only --no-hw 2>&1 | tee /tmp/claude-1000/probe-final-emu.log`

Confirm the `TRAP_VERDICT:` line matches the EMU verdict recorded in the findings doc G1 section (the probe is now a stable regression: re-running it reproduces the recorded EMU behavior).

- [ ] **Step 3: Finalize the findings doc**

Add a closing "Phase B inputs" section that states, as direct instructions to the Phase B plan: (a) the synchronous-trap halt boundary to implement (from G1); (b) the count-step semantics to implement + the explicit list of documented modeling decisions for unobservable edges (from G2); (c) whether the Section 8 forward-commitment is triggered. No placeholders — every item resolved to a concrete instruction or an explicit "deferred per Section 8, tracked".

- [ ] **Step 4: Commit and stop at the regroup**

```bash
cd /home/triple/npu-work/mlir-aie && git add test/npu-xrt/debug_halt_probe/ && \
git commit -m "debug_halt_probe: restore to Exp1 armed regression state + finalize findings

Probe left in the deterministic Exp1 breakpoint configuration as a
permanent re-runnable regression. Findings doc finalized with explicit
Phase B inputs (G1 halt boundary, G2 count-step semantics + documented
modeling decisions, Section 8 trigger status).

Generated using Claude Code." && \
cd /home/triple/npu-work/xdna-emu && \
git add docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md && \
git commit -m "debug_halt findings: finalize -- Phase B inputs section

Closes Phase A. Findings doc now carries direct, no-placeholder inputs
for the Phase B implementation plan. Regroup before Phase B planning.

Generated using Claude Code."
```

Then **stop**. Phase A is complete. Do not begin Phase B. Surface to Maya: Phase A findings (G1 verdict, G2 characterization, Section 8 status), and that the next step is to regroup and write the Phase B implementation plan against the recorded findings — her call when to start, per the plan→execute→regroup→next-plan rhythm.

---

## Self-review

**Spec coverage (spec Section 4 = Phase A):**
- 4.1 instrument (control-packet, marker-BO observation, EMU-first then HW, Exp1-before-Exp2, recovery staged) → Tasks 1 (mechanism), 2-4 (Exp1), 5-6 (Exp2); EMU-first and recovery in the per-task Conventions/guards; Exp1 banked (Task 4) before Exp2 HW (Task 6). Covered.
- 4.2 Experiment 1 (3 markers + done sentinel, trap PC from objdump, PC_Event0+Debug_Control2, before/after verdict, resume bonus) → Tasks 2,3,4. Resume-bonus: the spec's post-resume `mem[2]==0xCC` check is **not** implemented (the runtime sequence cannot deassert the breakpoint mid-run and re-observe without a second core pass). Recorded here as a known scope reduction: Phase A derives the halt-timing fact; resume is already wired/tested in-emulator and is re-validated in Phase B's interpreter-level tests, not via this probe. (Surfaced to Maya at the regroup.)
- 4.3 Experiment 2 (N markers, Debug_Control0 matrix, ReadRegisters corroboration) → Task 5,6. `ReadRegisters` corroboration via hand-assembled read control packets is **omitted** — marker-count is decisive per the spec and adding a ctrlOut read-packet path is extra surface for no decisive gain; noted as a deliberate YAGNI reduction, revisit only if marker-count proves ambiguous.
- 4.4 output (findings doc + permanent re-runnable probe) → Tasks 4,6,7. Covered.

**Two scope reductions to surface at the regroup** (both already noted above): the post-resume bonus check and the ReadRegisters corroboration are dropped as non-decisive; the marker-BO observation carries the spec's decisive load. Flag both to Maya rather than silently dropping.

**Placeholder scan:** No "TBD/TODO/handle appropriately". The intentional fill-ins (`0xNNNN` trap PC, BEFORE/AFTER verdict, LANDED counts) are *experimental observations the executor records*, not unspecified design — each has an exact derivation step and a "substitute the real value before committing" instruction. Acceptable: a probe plan's outputs are data, not code.

**Type/identifier consistency:** Test dir `debug_halt_probe`, tile `(0,2)`, offsets `0x32010/0x32018/0x32020`, marker values consistent across kernel and `test.cpp` per task (Exp1: 170/187/204/1; Exp2: 101..108), verdict tokens (`AFTER_COMMIT`/`BEFORE_COMMIT`/`LANDED`) consistent between `test.cpp` and the findings doc. Bridge invocation and recovery commands match CLAUDE.md.
