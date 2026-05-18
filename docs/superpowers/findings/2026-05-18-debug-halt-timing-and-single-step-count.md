# Debug Halt Timing and Single-Step Count

**Source**: Phase A debug_halt_probe, Task 5 (G1) / Task 7 (G2 -- pending).
**Ground truth**: Real NPU1 (Phoenix, XDNA/AIE2, 0000:c6:00.1).
**Observation method**: Control-packet OP_READ of output_buffer slots + Core_Status
register (0x32004) while core is halted (or after it completes). Five sequential
OP_READ packets, responses collected by shim DMA S2MM channel 0.
**Verdict signal**: Core_Status bit 16 = DEBUG_HALT (mask 0x00010000, per
XAIEMLGBL_CORE_MODULE_CORE_STATUS_DEBUG_HALT_MASK in aie-rt
driver/src/global/xaiemlgbl_params.h). HALTED = ((cs & (1<<16)) != 0).

Raw log: `/tmp/claude-1000/probe-exp1-hw.log`
Hardware result file: `build/bridge-test-results/20260518/debug_halt_probe.chess.hw.log`
Probe source: `mlir-aie/test/npu-xrt/debug_halt_probe/` (branch xdna-emu-cycle-budget,
HEAD 8095fbf79b at probe design time).

---

## G1 -- Breakpoint / single-step halt timing

### Probe configuration (Exp1, Task 5)

Kernel: straight-line core writes four sentinel markers to `output_buffer[0..3]`
then ends. Lock 5 (init=1) is acquired on entry and lock 4 is released on exit.
No lock-gated DMA path; observation is solely via control-packet OP_READ.

Compiler: Chess (xchesscc). PC addresses from `llvm-objdump` of
`main_core_0_2.elf` (chess build, confirmed in Task 3 and re-verified here):

```
core_0_2 disassembly (AIE2 VLIW, delay slots of j #release at 0x104):
  0x10a  mova dj0, #0xbb ; movxm p0, #0x70400   (load constants)
  0x114  st dj0, [p0, #4] ; mov m0, #0xaa        (TRAP bundle: store 0xBB -> output_buffer[1])
  0x11c  st m0, [p0], #8 ; movx r1, #0x1         (store 0xAA -> output_buffer[0])
  0x122  nopa ; st r1, [p0, #4] ; mov dj0, #0xcc (store 1 -> output_buffer[3])
  0x130  paddb ; nopa ; st dj0, [p0, #0] ; ...   (store 0xCC -> output_buffer[2])
```

Arming sequence (in `@seq`, before OP_READ readback):
- `npu.write32` addr=0x32020 value=0x80000114 -- PC_Event0: VALID | PC_ADDRESS=0x114
- `npu.write32` addr=0x32018 value=0x1 -- Debug_Control2[0] = PC_Event_Halt

Expected verdict if breakpoint fires at 0x114:
- `AFTER_COMMIT` (s1==0xBB, s0==s2==s3==0, HALTED=1): silicon halts AFTER trap bundle commits
- `BEFORE_COMMIT` (all slots 0, HALTED=1): silicon halts BEFORE trap bundle commits

### Hardware observed (NPU1, 2026-05-18)

```
SLOTS: s0=170 s1=187 s2=204 s3=1 CORE_STATUS=0x100000 HALTED=0
TRAP_VERDICT:NO_TRAP_OR_RAN_TO_END
```

Core_Status=0x100000 = bit 20 = CORE_DONE (XAIEMLGBL_CORE_MODULE_CORE_STATUS_CORE_DONE_MASK).
All four markers are fully written; the core ran to completion without halting.

### EMU baseline observed (Task 5 rerun, same session)

```
SLOTS: s0=170 s1=187 s2=204 s3=1 CORE_STATUS=0x0 HALTED=0
TRAP_VERDICT:NO_TRAP_OR_RAN_TO_END
```

EMU Core_Status=0x0: the emulator's control-packet read path does not route
OP_READ packets targeting the core/debug register space to the core_debug module
(known gap: symmetric to the write-side gap that drops the breakpoint-arming
writes). HALTED=0 in EMU reflects this read-side gap, not actual core state.
The NO_TRAP_OR_RAN_TO_END verdict on EMU is expected for this reason.

### Conclusion

**The breakpoint did NOT fire on hardware.** CORE_STATUS=0x100000 (DONE bit, not
DEBUG_HALT bit) and all four markers written confirm the core ran to completion
before the breakpoint arm took effect.

**Root cause -- arming race condition**: The core is enabled during PDI loading
(before `@seq` runs). Lock 5 is initialized to 1 (immediately available), so the
core acquires the lock and runs the entire kernel without blocking. By the time
the two `npu.write32` NPU commands in `@seq` travel from the host through the
NPU command path and reach the tile's debug registers (PC_Event0, Debug_Control2),
the core has already executed past PC 0x114 and reached DONE state.

The TRAP_PC address 0x114 is confirmed correct (re-verified against the chess ELF
disassembly in this task). The arming register offsets and values are also
correct (cross-checked against aie-rt xaiemlgbl_params.h). The failure is
purely a timing issue: the `npu.write32` path has insufficient command latency
to outrace the core startup.

**G1 is not derived from this run.** The breakpoint arming must be restructured to
guarantee the debug registers are armed before the core executes past the trap
bundle. Two options:

1. **Pre-arm via CDO**: Write PC_Event0 and Debug_Control2 as CDO commands in the
   PDI (not in `@seq`), so the debug registers are set before the core is enabled.
   This is the correct hardware approach and eliminates the race entirely.

2. **Block-before-arm**: Add a lock gate that holds the core before the trap bundle
   until the host sets a flag via a separate control-packet write32 (complex, less
   clean).

Option 1 (CDO pre-arm) is the recommended fix. It requires adding two
`aiex.npu.write32` ops to `@seq` to be moved into the CDO (or using a
`memref.global` / `aiex.npu.write32`-at-startup mechanism). This is a probe
redesign, not a verdict-level ambiguity -- the result is unambiguous: the race
means no halt, so no G1 observation yet.

**Phase B implication**: G1 remains undetermined. The emulator's current
post-`update_pc` synchronous trap model is neither proven correct nor incorrect by
this run. A probe redesign (CDO pre-arm) is required before a hardware G1
verdict can be recorded.

The EMU control-packet write-side gap (arming writes dropped) and read-side gap
(core_debug registers not reachable via OP_READ) are confirmed as separate Phase B
inputs, independent of the G1 result.

---

## G2 -- Single_Step_Count (Debug_Control0[5:2])

(Filled in Task 7.)
