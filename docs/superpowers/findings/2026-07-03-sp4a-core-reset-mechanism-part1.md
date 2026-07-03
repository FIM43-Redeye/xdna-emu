# SP-4a cold-start ROOT CAUSE: CORE_CONTROL reset bit (Part 1 landed)

**Date:** 2026-07-03  **Issue:** #140 (SP-4a)  **Commit:** `1e9e6700` (Part 1)
**Kernel:** of_q0_lean (traced); EMU via bridge-trace-runner, HW via same runner
with `XDNA_EMU` unset. Supersedes the "gate dataflow on dispatch" framing and
the drain-throttle-as-primary framing for the OFFSET symptom.

## The one-line answer

The EMU ran the compute cores at **CDO/config time (cyc 0)**; real silicon holds
them **enabled-but-in-reset** until the **firmware deasserts reset at kernel
launch**. The EMU derived core run-state from the ENABLE bit alone and ignored
the RESET bit. That single fact is the SP-4a cold-start root: cores ran ~673cy
(6779 traced) ahead of the shim dispatch, filling the objectfifo pipeline to a
full depth-2 equilibrium, so at dispatch the pipe was warm (HW's is cold).

## Mechanism, fully grounded (three independent sources agree)

1. **CDO bytes** (`build_q0_lean_trace/aie_traced.mlir.prj/main_aie_cdo_*.bin`,
   decoded word-by-word; command format `[op=0x40107=MaskWrite64][hi][addr]
   [value][mask]`, `0x_32000` = `CORE_CONTROL`):
   - `cdo_init` masked-writes `CORE_CONTROL` value 0x2 / mask 0x2 -> **asserts
     RESET (bit 1)**.
   - `cdo_enable` masked-writes value 0x1 / mask 0x1 -> sets **ENABLE (bit 0)**,
     mask leaves RESET untouched. Net after CDO: `CORE_CONTROL = 0x03`
     (ENABLE=1, RESET=1) = **enabled but held in reset = not executing**.
   - Nothing in the CDO *or* `insts.bin` ever clears the reset bit. The
     reset-deassert is a **pure firmware side effect** at kernel launch.
2. **aie-rt** (`xaie_core_aieml.c`): power-on `CORE_CONTROL = 0x02` (reset
   asserted); `XAie_CoreEnable` sets only the enable bit and does **not** clear
   reset (separate `XAie_CoreUnreset`). No HAL interlock couples core-enable to
   DMA dispatch -- run-ahead is HAL-legal, so the cold-at-dispatch behavior is a
   firmware/reset property, not a HAL gate.
3. **mlir-aie flow** (default npu1 objectfifo): core-enable, memtile relay DMA,
   BDs, lock init are ALL in the CDO (`AIETargetCDODirect`, `enableCores=true`);
   `insts.bin` is only the shim `dma_memcpy_nd` dispatch + `dma_wait`. So on HW
   the cores are "enabled" at CDO load, well before the runtime dispatch.

## The decisive HW probe (why we trust "cold at dispatch")

Added an env-gated **post-load sleep** to the bridge runner (`bridge-trace-
runner.cpp`, `SP4A_POST_LOAD_SLEEP_MS`): sleep between `xrt::hw_context`
creation (CDO/PDI applied) and the kernel run (dispatch). If cores ran from
CDO-load they would fill the pipe during the sleep.

| post-load sleep | offset | starvation | of_out onset (span) |
|---|---|---|---|
| 0 ms | +2 | +13 | 1123 |
| 100 ms | +2 | +13 | 1104 |
| 1000 ms | +2 | +13 | 1121 |

**Pipe state at dispatch is INVARIANT to a 1-second sleep** (~1e9 AIE cycles).
Cores do not produce during the config->dispatch window; they engage at ~the
kernel dispatch. Cold-at-dispatch is the faithful model, measured not assumed.

## The EMU bug (exact locus)

`src/device/state/compute.rs` -- both `write_core_register` and
`mask_write_core_register` CORE_CONTROL branches derived
`enabled = control & 1` (bit 0 only), ignoring RESET (bit 1). So at `0x03` the
EMU marked the core runnable while silicon holds it.

## Part 1 (landed, `1e9e6700`, byte-identical, 3585 lib tests pass)

- **Reset-honoring run-state:** `enabled = (control & 1 != 0) && (control & 0x2
  == 0)` in both branches. `CORE_CONTROL_RESET = 0x2` constant.
- **`DeviceState::release_core_resets()`** -- the **firmware-launch anchor**:
  clears the reset bit on enable-intent cores, recomputes run-state, pushes
  pending core-enables. Models the firmware reset-deassert that no register
  write in CDO/insts.bin performs.
- Called at the **end of `apply_cdo`** (config-completion) FOR NOW -> cores
  start ~at CDO time exactly as before, so behavior is byte-identical (lean
  offset -62, starvation +1491 unchanged). This is a pure scaffolding step:
  introduces the reset state machine + the named firmware seam, changes nothing.

## The symptom decomposition (confirmed by experiment)

Two independent, additive levers + a third residual. Verified by env-gated
experiments on the lean kernel (both hacks live UNCOMMITTED in the working tree):

- **Core-hold** (`XDNA_EMU_SP4A_COLD_UNTIL_DISPATCH=1`, coordinator gate: skip
  compute-core stepping until the of_out shim S2MM `(start_col,0)` ch0 arms) ->
  owns the **OFFSET**: -62 -> -12, and prod cold-starts at **+45 ~= HW +43**.
- **Drain throttle** (`XDNA_EMU_S2MM_COLD_COOLDOWN=341 _DECAY=0`, the earlier
  shim cold-start / 179-relocation work `ad4ef6ac`) -> owns **STARVATION**:
  +1491 -> **+11 ~= HW +13**. Untouched by the core-hold.
- Run both together: offset -12, starvation +11. Both near HW at once.
- **Third lever (residual):** offset -12 vs HW +2, and of_out onset +363 vs HW
  **+1120** -- the EMU forward-fills the cold pipe **~3x too fast**. This is the
  intermediate-stage relay/cold-start RATE, independent of the reset mechanism.

## What Part 1 is NOT (Part 2 = firmware timing)

Part 1 preserves behavior. The FIX (cold-at-dispatch) needs the firmware
reset-deassert **relocated from config-completion to the true launch timing**,
which couples to an unmeasured HW fact: how cheap is config-register MMIO in
AIE-cycles (the ~170cy/instr the EMU currently charges gives cores runway HW
does not). Pin with a **runtime-sequence padding HW probe** (add N dummy config
writes before the dispatch; if the pipe stays cold, config MMIO is ~free in
AIE-cycles and cores effectively start at dispatch).

## The endgame: real firmware on a simulated management processor

`release_core_resets()` is the seam. The reset-deassert, the ~8000cy mailbox
latency (executor), the dispatch gates, and the warm-up are ALL hardcoded
firmware behaviors. The "true simulator" (aiesim alternative) route is to load
the real XDNA management-processor firmware (Maya: "almost certain MicroBlaze",
binaries location TBD, partial RE done) and run it on a simulated mgmt processor
-- then these timings emerge instead of being magic numbers. MicroBlaze is a
documented 32-bit RISC with existing open tooling (QEMU target, open ISA sims);
the hard part is the peripheral/memory-map model (mailbox, ert command queue,
AIE array interface), which the partial RE maps.

## Prior falsification (do not repeat)

`806125d2` already proved `XDNA_EMU_WARMUP_CAP=0` (warm-up off) is byte-identical
-- the warm-up (`backend.rs:225`) is NOT the fill source; the main-loop
run-ahead is. The warm-up's stated premise ("cores run thousands of cycles
before insts arrive") is FALSIFIED by the HW sleep sweep. But it was added
(`a21806eb`) to fix a real init-before-insts-writes bug on some kernel -- find
that kernel before removing the warm-up in Part 2.

## Reproduce

```
cargo build -p xdna-emu-ffi
cmake --build bridge-runner/build   # for the post-load-sleep probe
cd build/experiments/sp4a-drainthrottle
# HW mechanism sweep (cores cold regardless of sleep):
env -u XDNA_EMU SP4A_POST_LOAD_SLEEP_MS=1000 <runner> ... ; python3 measure.py ...
# EMU decomposition:
XDNA_EMU_SP4A_COLD_UNTIL_DISPATCH=1 XDNA_EMU_S2MM_COLD_COOLDOWN=341 <runner> ...
```
