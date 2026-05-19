# Debug Halt Timing and Single-Step Count

**Source**: Phase A `debug_halt_probe`, Task 5/5b (G1) / Task 7 (G2 -- pending).
**Ground truth**: Real NPU1 (Phoenix, XDNA/AIE2, `0000:c6:00.1`), 2026-05-18.
**Observation method**: A firmware `XAIE_IO_MASKPOLL` (opcode 4) post-compile-
injected into the probe `insts.bin` blocks the instruction stream until
`Core_Status & (1<<16)` (`DEBUG_HALT`) before the control-packet OP_READ push.
The OP_READ therefore reads `output_buffer[0..3]` + `Core_Status` (0x32004)
*provably after* the core has halted (synchronization-ordered, not
latency-ordered). Five OP_READ packets; responses via shim DMA S2MM ch0.
**Verdict signal**: `Core_Status` bit 16 = `DEBUG_HALT` (mask `0x00010000`,
`XAIEMLGBL_CORE_MODULE_CORE_STATUS_DEBUG_HALT_MASK`, aie-rt
`driver/src/global/xaiemlgbl_params.h`). `HALTED = ((cs & (1<<16)) != 0)`.
Verdict computed from the disassembled store *schedule*, not source order.

Raw logs: `/tmp/claude-1000/probe-exp1-hw.log`,
`build/bridge-test-results/20260518/debug_halt_probe.chess.hw.log` (HW),
`.chess.bridge.log` (EMU). Transcribed below; the `/tmp` log is ephemeral.
Probe source: `mlir-aie/test/npu-xrt/debug_halt_probe/` (branch
`xdna-emu-cycle-budget`): `aie.mlir`/`README` at `9a12651d99`, `test.cpp`
verdict at `8546397987`; MASKPOLL injector + emulator graceful poll-
termination at xdna-emu `8be784a` (+ post-review fixups `4cd02a1`).

---

## G1 -- Breakpoint / single-step halt timing

### Probe configuration (final, Exp1)

Straight-line core: acquire a blocking objectfifo `@gate`, write four sentinel
markers to `output_buffer[0..3]`, `aie.end`. No lock-gated DMA path;
observation is solely control-packet OP_READ. Compiler: Chess (xchesscc).

Three grounded redesigns were required before a valid G1 could be observed --
each a real hardware/toolchain fact surfaced by grounding, recorded here so the
derivation is durable:

1. **Arming race** (attempt 1): the core is CDO-enabled before `@seq` runs, so
   it completed before the `@seq` breakpoint-arming `write32`s landed
   (`Core_Status=0x100000` CORE_DONE, all markers written). Fix: a host->core
   **blocking objectfifo `@gate`** (`llvm.aie2.acquire`, a HW pipeline stall
   immune to load-elimination); `@seq` arms *then* feeds the gate (in-order
   guarantee, `AIEToConfiguration`).
2. **Shim-channel collision** (attempt 2): `@gate` defaulted its shim feed to
   shim MM2S ch0, which the hand-rolled ctrl-in OP_READ push also used, so the
   pathfinder compiled a single circuit broadcast
   (`switchbox(0,2): South:1 -> {TileControl:0, DMA:0}`) -- HW-unsafe. The
   objectfifo channel is not toolchain-pinnable; fix: leave `@gate` on default
   ch0 and **repoint the ctrl-in push to shim MM2S ch1** (stride `0x8`).
3. **No happens-after** (attempt 3, P2/P3): the static-`@seq` OP_READ had no
   synchronization ordering "core halted at trap" before "OP_READ reads" --
   pure relative latency. No on-device halt->lock/event actuation exists and
   `@seq` is strictly static, so fix: post-compile-inject a firmware
   **`XAIE_IO_MASKPOLL`** on `Core_Status[16]` before the OP_READ push.

Disassembly (`llvm-objdump-aie` of `main_core_0_2.elf`, gated chess build;
re-confirmed post-injector -- the injector patches only `insts.bin`, never the
core ELF):

```
0x17a  mova dj0, #0xbb ; movxm p0, #0x70400      (load constants; p0=output_buffer)
0x184  st   dj0, [p0, #4] ; mov m0, #0xaa        (TRAP bundle: store 0xBB=187 -> output_buffer[1] = s[1])
0x18c  st   m0,  [p0], #8 ; movx r1, #0x1        (strictly-later: store 0xAA=170 -> output_buffer[0] = s[0])
 ...   later bundles: store 1 -> s[3]; store 0xCC=204 -> s[2]
```

`TRAP_PC = 0x184` (the slot-1 `0xBB` store). Arming (in `@seq`, before the
gate-feed and OP_READ):
- `npu.write32` `0x32020 = 0x80000184` -- PC_Event0: VALID(31) | PC_ADDRESS=0x184
- `npu.write32` `0x32018 = 0x1` -- Debug_Control2[0] = PC_Event_Halt
- injected `MASKPOLL` `0x00232004`, mask=value=`0x00010000` -- block until DEBUG_HALT

Verdict map (schedule-derived; spec §4.2): trap slot = `s[1]`, strictly-later =
`s[0]`. `halted ∧ s[1] committed ∧ no strictly-later` -> AFTER_COMMIT;
`halted ∧ all slots zero` -> BEFORE_COMMIT (DEBUG_HALT=1 disambiguates from
"core never ran" -- a core that never ran is not debug-halted).

### Hardware observed (NPU1, 2026-05-18)

```
SLOTS: s0=0 s1=0 s2=0 s3=0 CORE_STATUS=0x10001 HALTED=1
TRAP_VERDICT:BEFORE_COMMIT
```

HW run: 1.9 s, bridge PASS, no wedge. `Core_Status = 0x10001` =
`DEBUG_HALT`(bit 16) | `ENABLE`(bit 0): the core ran and is debug-halted. All
four marker slots -- including the trap slot `s[1]` (the `0xBB`/187 store at
`TRAP_PC=0x184`) -- are zero. The MASKPOLL satisfied (DEBUG_HALT became set),
so the OP_READ provably read a *stable, halted* state.

### EMU baseline observed (same byte-identical injected binary, same session)

```
XDNA_EMU_STATUS: halt_reason=maskpoll_unsatisfied cycles=24304 max_cycles=0
SLOTS: s0..s3 = 0xDEADC0DE  CORE_STATUS=0xdeadc0de  HALTED=0
TRAP_VERDICT:MASKPOLL_UNSATISFIED_EMU
```

EMU drops the breakpoint-arming control-packet writes (write-side
`write_core_register` catch-all), so `Core_Status[16]` is never set and the
MASKPOLL is structurally unsatisfiable on EMU. The emulator's graceful poll-
termination contract fired exactly as designed: deterministic termination at
24304 cycles (3.3 s wall, *no hang*), the polled register never faked, the core
never pretend-halted, the OP_READ never issued -- `output_buffer` stays at the
`0xDEADC0DE` sentinel `test.cpp` pre-filled. This validates the injector + the
poll-termination contract; G1 itself is HW-only by construction (EMU cannot
halt -- a recorded Phase B input, not a flaw).

### Conclusion

**On NPU1 (Phoenix/AIE2) silicon, a synchronous PC-event breakpoint halts the
core BEFORE the trap bundle commits.** Evidence: `DEBUG_HALT=1` proves the core
executed and is debug-halted; every marker slot is zero, including the trap
slot `s[1]` whose store *is* the trap bundle at `0x184` -- so that store had
not committed when the core halted. `DEBUG_HALT=1` rules out the "core never
ran" reading (a non-running core is not debug-halted). Incidental
corroboration: `ENABLE`(bit 0)=1 alongside `DEBUG_HALT`, consistent with
ENABLE-stays-1-while-halted -- though the verdict deliberately relies on
`DEBUG_HALT` alone (spec §4.2; ENABLE-stays-1 was an unverified assumption,
here merely observed, not depended on).

**Phase B implication (real, scoped fidelity fix -- spec §5.1).** The emulator
currently evaluates the trap and the `DebugHalt` gate *after* `update_pc`
(after the bundle's side effects commit). Silicon halts *before* the trap
bundle commits for synchronous-origin halts. Phase B must evaluate the trap +
`DebugHalt` boundary **before the trap bundle commits** for synchronous traps
(PC_Event / breakpoint / single-step); asynchronous halts (host
`Debug_Control0[0]`, stall-halt, external event-halt) take effect at the next
boundary and are unchanged. This is the named sync/async halt-boundary seam,
not an ad hoc tweak. A guarding interpreter-level test must assert the observed
before-commit timing (breakpoint at a known PC; assert the trap-bundle side
effect did *not* land).

The probe stays checked in as a permanent re-runnable regression: on HW it
derives G1; on EMU the `MASKPOLL_UNSATISFIED_EMU` contract guards the injector
+ graceful poll-termination path. The EMU control-packet write-side gap (arming
writes dropped) and read-side gap (`core_debug` registers unreachable via
OP_READ) remain separate recorded Phase B inputs, independent of the G1 result.

---

## G2 -- Single_Step_Count (Debug_Control0[5:2])

(Filled in Task 7.)
