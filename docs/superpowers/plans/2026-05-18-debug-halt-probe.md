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

**Goal:** Add a host→core blocking objectfifo gate so the core cannot execute its first real instruction until the host has (1) issued the arming `write32`s then (2) fed the gate. Re-derive `TRAP_PC` + the slot⇄schedule map from the gated build's fresh disasm (the schedule changes). EMU-validate the no-trap baseline (EMU still can't arm — write-side gap — so EMU expects `NO_TRAP_OR_RAN_TO_END` once the gate is fed; this validates the gate plumbing + readback, not halt).

**Files:** Modify `mlir-aie/test/npu-xrt/debug_halt_probe/{aie.mlir,test.cpp,README.md}`. Reference patterns (read-only): `/home/triple/npu-work/mlir-aie/test/aiecc/cpp_basic.mlir` and `/home/triple/npu-work/mlir-aie/programming_examples/vision/vision_passthrough/aie2_lineBased_8b_tiny.mlir` (canonical host→core objectfifo-feed via runtime-sequence `dma_memcpy_nd`).

- [ ] **Step 1: Add the gate objectfifo.** Define a 1-element shim→compute objectfifo `@gate` (host/shim producer → tile (0,2) consumer, `memref<1xi32>`), modeled on the cited examples. The core's FIRST op becomes `aie.objectfifo.acquire @gate(Consume,1)` (blocks on `llvm.aie2.acquire`, a HW pipeline stall) then `aie.objectfifo.release @gate(Consume,1)`, then the existing 4 marker stores + `aie.end`. Keep the Task-3 breakpoint-arming `write32`s in `@seq` and the Task-4 control-packet OP_READ readback plumbing. In `@seq`, order: arming `write32`s (0x32020/0x32018) → `dma_memcpy_nd` feeding `@gate` → the OP_READ readback push → `dma_wait{@ctrl0}`. (Runtime-sequence order is an in-order guarantee per `AIEToConfiguration`.) Preserve SPDX, aie-rt citations, TRAP_PC/OUTBUF_ADDR re-derivation warnings; no `@out0`/lock-gated-DMA reintroduction. Add a host gate-buffer BO arg. **Shim-channel disjointness (REVISION 2, mandatory):** `@gate` stays on its natural default shim MM2S **ch0** (the objectfifo channel is not pinnable — grounded, spec §4.2). Repoint the hand-rolled ctrl-in OP_READ push from shim MM2S **ch0** to **ch1**. The toolchain-derived delta (per-channel stride `0x8`, aie-rt `xaiemlgbl_reginit.c .ChIdxOffset = 0x8`; confirm against the regdb before editing): ctrl-in CTRL `0x1d210→0x1d218`; ctrl-in TASK_QUEUE `0x1d214→0x1d21c` (all five occurrences); the ctrl-in `aie.packet_flow` source `<%tile_0_0, DMA : 0>→<%tile_0_0, DMA : 1>`; the `aiex.npu.sync` MM2S ops `channel = 0→1` (all five). **Do not change** any BD-layout/BD-address writes (`0x1d000`, `0x1d004` address_patch, `blockwrite_data_0`) — BDs are channel-independent. Do not touch the `@gate` objectfifo's channel (leave it default). The two host→compute flows must take independent physical shim paths or the pathfinder broadcasts both from a shared slave (spec §4.2). **Mandatory gate before the EMU run:** compile EMU-only (`--no-hw --compile`) and verify in the lowered `input_physical.mlir` that `switchbox(0,2)` routes `@gate` and the ctrl-in flow on *distinct* slave ports (no shared `South:1 → {TileControl:0, DMA:0}` broadcast); quote the relevant `aie.connect`/`aie.packet_rules` lines. If still shared, STOP and report.

- [ ] **Step 2: Re-derive TRAP_PC + slot⇄schedule map.** The gate changes the compiled core. Disassemble the new core ELF (`tools/llvm-objdump-aie -d <chess prj>/main_core_0_2.elf`), identify the bundle that stores `0xBB(187)` to the slot-1 store (the trap), record the new `TRAP_PC`, `TRAP_PC14 = TRAP_PC & 0x3FFF`, new `PC_Event0 value = 0x80000000 | TRAP_PC14`, and the new strictly-later slot/PC. Update the `write32 0x32020` value and the test.cpp schedule-comment + verdict slot mapping accordingly. Update the in-artifact TRAP_PC warning text with the new derivation. If the disasm can't unambiguously identify the 0xBB-to-slot1 store, STOP and report.

- [ ] **Step 3: EMU run — validate gate plumbing + no-trap baseline.** `cd /home/triple/npu-work/xdna-emu && ./scripts/emu-bridge-test.sh debug_halt_probe --chess-only --no-hw 2>&1 | tee /tmp/claude-1000/probe-exp1-gate-emu.log`. EMU: the gate must be fed by `@seq` so the core unblocks and runs (confirm it does — a stuck gate hangs EMU too); EMU still drops the arming writes so expect the markers all written, `CORE_STATUS=0x0` (read-side gap), `HALTED=0`, `TRAP_VERDICT:NO_TRAP_OR_RAN_TO_END`, bridge `PASS`. This validates the gate + readback plumbing end-to-end on EMU. Record exact lines. **With the REVISION-2 shim-channel fix the run must now COMPLETE (not time out):** the 600s timeout in the first attempt *was* the shim-channel collision; disjoint channels let the ctrl-out S2MM assemble its 5 OP_READ responses so `dma_wait @ctrl0` satisfies. A timeout now means the channel-disjointness fix did not take (re-check the ctrl-in ch1 register repoint and the `input_physical.mlir` switchbox slave ports) — debug before any HW; do not proceed to Task 5b on a timeout.

- [ ] **Step 4: Update README** (gate mechanism, new TRAP_PC, EMU baseline meaning, that Task 5b is the HW run). Commit (single, from `/home/triple/npu-work/mlir-aie`, `git add test/npu-xrt/debug_halt_probe/`):
```
debug_halt_probe: Exp1 -- blocking objectfifo gate (fix arming race) + re-derive TRAP_PC

HW attempt 1 raced (core CDO-enabled before @seq arming landed). Add a
host->core blocking objectfifo gate (llvm.aie2.acquire HW stall, immune
to load-elimination); @seq arms then feeds the gate (in-order
guarantee). Re-derived TRAP_PC/slot map from the gated build's disasm.
EMU validates gate+readback plumbing (no-trap baseline); HW is Task 5b.

Generated using Claude Code.
```

**Report:** the new TRAP_PC/TRAP_PC14/PC_Event0 value and how the 0xBB-slot1 store was identified; exact EMU `SLOTS:`/`CORE_STATUS`/`TRAP_VERDICT` lines; gate-feed confirmed (core unblocks, no EMU hang); preservation confirmations; commit SHA; concerns.

---

## Task 5b: Experiment 1 — hardware run, derive G1, open the findings doc [HARDWARE FORK]

**Goal:** Run the gated+armed probe on the real NPU; the gate guarantees arming-before-core-run; the core halts at the trap; control-packet OP_READ reads `output_buffer` + `Core_Status` while halted; the schedule-derived verdict yields the authoritative G1 (before/after-commit). This is the first exercise of read-while-halted (EMU could not validate it).

**Files:** Create `xdna-emu/docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md`

- [ ] **Step 1: Pre-flight the hardware.** No other HW suite running. `xrt-smi validate` (expect pass). If it fails: `pkexec sh -c 'modprobe -r amdxdna && modprobe amdxdna'`, re-validate. Do not proceed until healthy.

- [ ] **Step 2: Run the armed probe on hardware**

`cd /home/triple/npu-work/xdna-emu && ./scripts/emu-bridge-test.sh debug_halt_probe --chess-only 2>&1 | tee /tmp/claude-1000/probe-exp1-hw.log`

Record the exact `SLOTS:` / `CORE_STATUS` / `TRAP_VERDICT:` lines from the **hardware** run. Interpretation:
- `AFTER_COMMIT` → silicon halts after the trap bundle commits → the emulator's post-`update_pc` model is **proven correct** for sync traps.
- `BEFORE_COMMIT` → silicon halts before the trap bundle commits → emulator after-commit model is a **real Phase B fidelity fix**.
- `NO_TRAP_OR_RAN_TO_END` → breakpoint did not fire on HW: debug arming (PC mask, Debug_Control2 bit, ordering of the arming writes vs core release). Do **not** record a non-halted result as G1.
- `ANOMALY_NOT_HALTED` / `AMBIGUOUS` → record the raw `SLOTS:`/`CORE_STATUS` verbatim; do not force a conclusion.

Wedge protocol: if the core halts and `run.wait()` times out leaving the device wedged, the control-packet read should still have completed before the timeout (it does not depend on the core). If the NPU wedges, recover per the hardware-run guard, rerun once; second wedge → stop and surface to Maya with the raw logs.

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

**Goal:** Reshape the probe for `Debug_Control0[5:2]` (`Single_Step_Count`) characterization, **reusing the Task 4 control-packet readback infra** (count-step may also halt the core mid-sequence, so the lock-gated DMA is equally unusable here).

**Files:** Modify `mlir-aie/test/npu-xrt/debug_halt_probe/{aie.mlir,test.cpp,README.md}`

- [ ] **Step 1: 8-marker straight-line core.** Replace the core body with 8 sequential distinct stores (`output_buffer[0..7] = 101..108`):

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

Markers are NOT assumed to be in source order on-chip — Step 3's verdict counts *how many distinct markers landed*, which is order-independent (each value is unique), so count-step characterization does not depend on schedule order.

- [ ] **Step 2: Count-step config in @seq.** Replace the Task 3 PC_Event0/Debug_Control2 arming writes with a single Debug_Control0 count write (keep the control-packet read plumbing from Task 4 intact). Start `count=4`, no halt bit: `Debug_Control0 = (4<<2) = 0x10`:

```mlir
      // Debug_Control0: Single_Step_Count = N at bits [5:2] (aie-rt
      // XAIEMLGBL_CORE_MODULE_DEBUG_CONTROL0, xaiemlgbl_params.h:2452;
      // SINGLE_STEP_COUNT bits 2-5, DEBUG_HALT_BIT bit 0). Matrix swept
      // by editing this value (see README / findings).
      aiex.npu.write32 {address = 0x32010 : ui32, column = 0 : i32, row = 2 : i32, value = 0x10 : ui32}
```

- [ ] **Step 3: LANDED verdict via control-packet readback in test.cpp.** Reuse the Task 4 read-packet machinery. Read `output_buffer[0..7]` (8 slots, `OUTBUF_ADDR + 4*k`) and `Core_Status` via OP_READ. Verdict:

```cpp
  int landed = 0;
  for (int k = 0; k < 8; k++) if (s[k] == (uint32_t)(101+k)) landed++;
  bool halted = (cs & (1u<<16)) != 0;  // DEBUG_HALT bit alone -- ENABLE-stays-1 is an unverified HW assumption; DEBUG_HALT=1 already proves the core ran and is halted
  std::cout << "SLOTS:";
  for (int k=0;k<8;k++) std::cout << " " << s[k];
  std::cout << " CORE_STATUS=0x" << std::hex << cs << std::dec
            << " HALTED=" << (halted?1:0) << "\nLANDED:" << landed << "\n";
  // landed==8 && !halted -> count-step had no effect (ran to completion)
  // halted && landed==N   -> core halted after exactly N committed stores
  // other                 -> partial/other; record SLOTS+CORE_STATUS verbatim
  std::cout << "PASS\n";
  return 0;
```

- [ ] **Step 4: EMU run + document sweep matrix.** `cd /home/triple/npu-work/xdna-emu && ./scripts/emu-bridge-test.sh debug_halt_probe --chess-only --no-hw 2>&1 | tee /tmp/claude-1000/probe-exp2-emu.log`. EMU drops the Debug_Control0 write (catch-all), so expect `LANDED:8`, not halted (baseline: count-step inert in EMU — itself the recorded G2 EMU finding). Record exact lines. Append an "Experiment 2 sweep matrix" section to README listing the HW configs with exact `Debug_Control0` values: `count=4 no-halt 0x10`, `count=4 halt 0x11`, `count=2 no-halt 0x08`, `count=8 no-halt 0x20`, `count=0 control 0x00`.

- [ ] **Step 5: Commit**

```bash
cd /home/triple/npu-work/mlir-aie
git add test/npu-xrt/debug_halt_probe/
git commit -m "debug_halt_probe: Exp2 -- count-step kernel + matrix (reuses ctrl-pkt readback)

8 distinct marker stores; @seq writes Debug_Control0 Single_Step_Count;
LANDED counted via the Task 4 control-packet readback (count-step can
also halt the core, so lock-gated DMA is equally unusable). EMU baseline
(count-step inert in EMU) recorded; HW sweep matrix in README.

Generated using Claude Code."
```

---

## Task 7: Experiment 2 — hardware sweep, derive G2 [HARDWARE FORK]

**Goal:** Sweep the count-step matrix on the real NPU; record what `Debug_Control0[5:2]` does on silicon as far as observable.

**Files:** Modify `mlir-aie/test/npu-xrt/debug_halt_probe/aie.mlir` (one value per point); `xdna-emu/docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md` (G2 section)

- [ ] **Step 1: Pre-flight hardware** (as Task 5 Step 1: `xrt-smi validate`, recover if needed).

- [ ] **Step 2: Run each matrix point on hardware.** For each README-matrix config: set the `Debug_Control0` value in aie.mlir, run `./scripts/emu-bridge-test.sh debug_halt_probe --chess-only 2>&1 | tee /tmp/claude-1000/probe-exp2-hw-<cfg>.log` from `xdna-emu/`. Record `LANDED:` + full `SLOTS:`/`CORE_STATUS` per config. Recover only on wedge (hardware-run guard); recurring wedge on a config → record "config <X> wedges device" as a finding, skip to next.

- [ ] **Step 3: Fill the G2 findings section.** Results table (config → EMU LANDED → HW LANDED/HALTED → interpretation) + a conclusion on arm/decrement/expire/halt-bit interaction and what stays unobservable. Mark unobservable edges "documented modeling decision in Phase B"; note the §8 forward-commitment (silicon-fidelity revisit) applies if under-characterized.

- [ ] **Step 4: Commit**

```bash
cd /home/triple/npu-work/xdna-emu
git add docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md
cd /home/triple/npu-work/mlir-aie
git add test/npu-xrt/debug_halt_probe/aie.mlir
git commit -m "debug_halt findings: G2 count-step characterized on hardware

Debug_Control0[5:2] matrix swept on real NPU1 (control-packet readback).
Results table + conclusion; unobservable edges flagged for Phase B
documented modeling decisions / Section 8 forward-commitment.

Generated using Claude Code."
```

---

## Task 8: Finalize findings, restore the probe to a re-runnable state, regroup checkpoint

**Goal:** Leave the probe a clean permanent regression artifact, findings complete, explicit handoff to the Phase B planning regroup.

**Files:** Modify `mlir-aie/test/npu-xrt/debug_halt_probe/{aie.mlir,README.md}`; `xdna-emu/docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md`

- [ ] **Step 1: Restore the probe to the Exp1 redesigned armed state** (Task 4 end-state: 4-marker core, Task 3 breakpoint-arming writes, control-packet readback + schedule-derived verdict). Rationale: Exp1 is the deterministic regression-valuable config; Exp2's matrix is a swept investigation. README documents how to switch to the Exp2 core for re-investigation.

- [ ] **Step 2: EMU re-run** confirms the restored probe reproduces its recorded EMU baseline (`SLOTS: 170 187 204 1`, `NO_TRAP_OR_RAN_TO_END`, `PASS`): `./scripts/emu-bridge-test.sh debug_halt_probe --chess-only --no-hw 2>&1 | tee /tmp/claude-1000/probe-final-emu.log`.

- [ ] **Step 3: Finalize the findings doc.** Closing "Phase B inputs" section, direct no-placeholder instructions to Phase B: (a) the synchronous-trap halt boundary from G1; (b) count-step semantics from G2 + the explicit documented-modeling-decision list for unobservable edges; (c) the **control-packet → core_debug register-write routing gap** (EMU drops debug-reg writes via the `write_core_register` catch-all) as an explicit Phase B work item; (d) whether the §8 forward-commitment is triggered.

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

Closes Phase A. No-placeholder Phase B inputs: G1 halt boundary, G2
count-step + modeling decisions, the control-packet->core_debug routing
gap work item, Section 8 trigger status. Regroup before Phase B.

Generated using Claude Code."
```

Then **stop**. Phase A is complete. Do not begin Phase B. Surface to Maya: Phase A findings (G1 verdict, G2 characterization, the routing-gap Phase B input, §8 status), and that the next step is the regroup before writing the Phase B plan — her call when, per the plan→execute→regroup→next-plan rhythm.

---

## Self-review

**(Revised 2026-05-18 for the post-Task-3 redesign; spec §4 updated in lockstep.)**

**Spec coverage (spec Section 4 = Phase A):**
- 4.1 instrument (control-packet readback while halted is load-bearing per revised §4.1; EMU-first then HW; Exp1 before Exp2; recovery staged) → Tasks 1 (authoring mechanism), 4 (readback mechanism + no-trap baseline), 5 (Exp1 HW), 6 (Exp2 EMU), 7 (Exp2 HW). Covered.
- 4.2 Experiment 1, redesigned (4 markers; trap PC from `llvm-objdump-aie`; PC_Event0+Debug_Control2; **schedule-derived** verdict + Core_Status disambiguation; control-packet OP_READ while halted) → Tasks 2 (kernel), 3 (arming + disasm-derived TRAP_PC + re-derivation discipline), 4 (readback + verdict), 5 (HW derive G1). The "force store order" idea was investigated and rejected (no toolchain pin; empirically confirmed by our own disasm); schedule-derived verdict is the spec-blessed approach.
- 4.3 Experiment 2 (8 markers, Debug_Control0 matrix) → Tasks 6,7. Control-packet readback is now **load-bearing, not omitted** — the redesign promotes it from the earlier YAGNI-deferred corroboration because the lock-gated DMA cannot observe a halted core (Task 3 discovery). Exp2 reuses the Task 4 readback infra.
- 4.4 output (findings doc + permanent re-runnable probe) → Tasks 5,7,8. Covered.

**Superseded scope reductions (no longer open):** the earlier "ReadRegisters omitted as YAGNI" reduction is reversed — now the primary observation, justified by the discovered flaw. The post-resume hardware check stays out of Phase A but is a *tracked spec §8 forward-commitment* (resume hardware-verification), not an unflagged drop. Both are resolved in the spec, not merely "surfaced at regroup".

**New Phase B input captured:** the EMU control-packet→`core_debug` register-write routing gap (debug-reg writes hit the `write_core_register` catch-all and are dropped) is recorded as an explicit Phase B work item in the findings-doc Phase B inputs (Task 8 Step 3) — discovered, not designed-around.

**Placeholder scan:** No "TBD/TODO/handle appropriately". Intentional fill-ins (`<BEFORE|AFTER>` verdict, `OUTBUF_ADDR`, LANDED counts) are *experimental observations the executor records / derives*, each with an exact derivation step and a "substitute the real value before committing" instruction — a probe plan's outputs are data, not code.

**Type/identifier consistency:** Test dir `debug_halt_probe`, tile `(0,2)`, offsets `0x32010/0x32018/0x32020`, `Core_Status` `0x32004`, `OUTBUF_ADDR` (derived, expected `0x0400`), trap bundle `0x114`→slot1=0xBB / strictly-later `0x11c`→slot0=0xAA consistent across spec §4.2, the aie.mlir TRAP_PC comment, and `test.cpp`; verdict tokens (`AFTER_COMMIT`/`BEFORE_COMMIT`/`NO_TRAP_OR_RAN_TO_END`/`AMBIGUOUS`/`ANOMALY_NOT_HALTED`/`LANDED`/`PASS`) consistent between `test.cpp` and the findings doc. Bridge invocation + recovery commands match CLAUDE.md.
