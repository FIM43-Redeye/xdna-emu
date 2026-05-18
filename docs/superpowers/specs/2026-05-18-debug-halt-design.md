# debug_halt — design

Status: design (brainstormed, pre-plan)
Date: 2026-05-18
Subsystem: `debug_halt` (AIE2 core debug — halt/resume, breakpoints,
single-step), coverage gap-queue entry.

## 1. Context and the true gap

The coverage seed (`crates/xdna-archspec/src/coverage/units.rs`,
`debug_halt`) reads "Halt + status bits modeled. Programmable
breakpoints and single-step PC trap not wired through interpreter."
A code+toolchain audit (2026-05-18) showed that narrative is **stale**:
most of it has since been wired and is toolchain-faithful.

Confirmed wired and faithful (these stay as-is — do not redesign):

- Halt gate: `src/interpreter/core/interpreter.rs:181` returns
  `StepResult::DebugHalt` while `tile.core_debug.is_halted()`.
- Event-halt + resume: `Debug_Halt_Core_Event0/1`,
  `Debug_Resume_Core_Event` (`check_event_halt`, mod.rs:626), driven
  per step. Matches aie-rt `XAie_CoreConfigDebugControl1`
  (`aie-rt/driver/src/core/xaie_core.c:692-749`).
- PC_Event0..3 breakpoints + `PC_Range_0_1/2_3`, gated by
  `Debug_Control2[0]` (`check_pc_events`, mod.rs:700), driven via
  `update_pc` (`coordinator.rs:638`).
- Stall-halt (`Debug_Control2` mem/lock/stream, `check_stall_halt`).
- Event-driven single-step (`Debug_Control1[14:8]`,
  `pending_single_step` latch drained at `coordinator.rs:643`).
- Debug_Status layout (no single-step cause bit; aggregate only) —
  matches AM025 `core_module/debug/misc.txt:7-14` and aie-rt
  `xaie_core.h:36-42`.
- halt/unhalt = `Debug_Control0[0]` write — matches aie-rt
  `_XAie_CoreDebugCtrlHalt`.

The genuine remaining gap is exactly two items:

- **G1 — Halt timing unverified.** Breakpoint / single-step halts
  currently take effect *after* the trap bundle commits, purely by
  construction (`update_pc` runs post-step in the coordinator, the
  `is_halted` gate trips on the *next* step). The open-source toolchain
  does **not** specify whether silicon halts before or after the bundle
  at the trap PC commits. This is a latent fidelity unknown beneath
  every wired trap path.
- **G2 — Count-based single-step is dead state.**
  `Debug_Control0[5:2]` `Single_Step_Count` ("Number of instruction to
  single-step", AM025 `core_module/debug/control.txt:8`) is stored by
  `write_debug_control0` (mod.rs:787) and read by **nothing**. No
  toolchain API writes this field (aie-rt has none; the toolchain's
  single-step is exclusively the event-driven `Debug_Control1[14:8]`
  path). It is therefore not a binary-compatibility hole — no compiled
  `.xclbin` configures it — but it is the natural register substrate
  for xdna-emu's own visual step-debugger.

## 2. Audit findings (derived facts)

From the 2026-05-18 toolchain-derivation pass (aie-rt
`driver/src/core/`, AM025 register JSON
`mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json`, AM025 extracted
docs `docs/xdna/am025-compact/core_module/`):

- A. `Debug_Control0[5:2]` Single_Step_Count: AM025 verbatim "Number of
  instruction to single-step". 4-bit (0-15). **Hardware arm / decrement
  / auto-clear semantics NOT SPECIFIED in open-source toolchain.** No
  aie-rt API writes it.
- B. `Debug_Control1[14:8]` Debug_SingleStep_Core_Event: "Event No to
  Single-Step the AI Engine". aie-rt-exposed
  (`XAie_CoreConfigDebugControl1`). Independent mechanism from A.
- C. **Halt timing (before/after the trap bundle): NOT SPECIFIED in
  open-source toolchain.** No aie-rt comment, AM025 text, or test pins
  the instruction boundary. Requires hardware observation.
- D. Debug_Status: bits 0 Debug_halted, 1 PC_Event_halted, 2
  Memory_Stall_Halted, 3 Lock_Stall_Halted, 4 Stream_stall_Halted, 5
  Debug_Event0_Halted, 6 Debug_Event1_Halted. **No single-step cause
  bit** — aggregate halt is the only signal. Our model already matches.
- E. PC_Event* = bit 31 VALID, bits [13:0] 14-bit PC_ADDRESS.
  Core_Status[16] = Debug_Halt ("Debug has stopped the AI Engine").
  Our model already matches.
- F. Resume: aie-rt halt/unhalt write `Debug_Control0[0]`;
  `Debug_Resume_Core_Event` is event-based resume. aie-rt clears no
  latches and never touches the count field on unhalt.

Two items (A arm/clear semantics, C halt timing) are unspecified by the
open-source toolchain *because no toolchain path exercises them*. Per
the project source hierarchy, the next ground truth is hardware
observation.

## 3. Scope

In scope:

- Phase A: a hardware probe that derives C (halt timing) and
  characterizes A (count-step behavior) as far as silicon reveals.
- Phase B: implement the sync-vs-async halt-boundary distinction
  (parameterized on C) and the count-step state machine as the
  emulator's debugger substrate (parameterized on A, with documented
  modeling decisions for unobservable edges).
- Coverage narrative rewrite to the true state; verdict → Full once
  both gaps are closed.

Out of scope (explicitly):

- Redesigning any confirmed-wired-and-faithful path (Section 1 list).
- A visual-debugger UI consuming the count-step substrate (separate
  later work; Phase B only provides the faithful register-level
  substrate it will sit on).
- AIE2P/AIE2PS debug specifics (AIE2/NPU1 only, per project focus).

## 4. Phase A — hardware probe (permanent artifact)

### 4.1 Instrument

A hand-authored NPU1 mlir-aie probe kernel + runtime sequence,
checked in under the bridge/experiments infra (not a throwaway patch).
Control packets — the just-shipped `control_packets` subsystem
(`CtrlPacketAction::WriteRegister` / `ReadRegisters`) — inject the
debug-register configuration and read back debug state. Control-packet
register writes are faithful: they are the same class of write the real
NPU instruction stream carries, so the identical probe binary runs on
EMU and HW. **No `AIE_RW_ACCESS`** anywhere (avoids the
mailbox-poisoning path the calibration-feedback memory warns against).

The decisive observation is plain output-BO readback: the kernel writes
sentinel markers to a buffer; the host reads the buffer back through the
normal XRT path. Control-packet `ReadRegisters` of Core_PC /
Core_Status / Debug_Status is corroborating, not load-bearing.

Run order: EMU-first (harness shakeout, expected-behavior baseline),
then a single HW session. NPU recovery staged per CLAUDE.md
(`modprobe -r amdxdna && modprobe amdxdna` first). **Experiment 1 runs
and is banked before Experiment 2** — high-value/low-risk result
secured before the exploratory pokes.

### 4.2 Experiment 1 — halt timing (G1)

Kernel writes distinct markers at three known bundles:
`mem[0]=0xAA` (pre-trap), `mem[1]=0xBB` (the trap bundle),
`mem[2]=0xCC` (post-trap). The trap bundle's 14-bit PC is obtained from
`llvm-objdump -d` of the compiled core ELF (map the `mem[1]` store to
its bundle PC; PC_Event matches the low 14 bits).

Control packets, before core start: `PC_Event0 = VALID | addr(bundle_1)`;
`Debug_Control2[0]=1` (PC_Event_Halt). Start core. Readback:

- `mem[1]==0xBB` → trap bundle committed → **halt is after-commit**.
- `mem[1]==0` and `mem[0]==0xAA` → **halt is before-commit**.
- `mem[2]` must be `0` either way (core halted).

Then resume (clear `Debug_Control2[0]` or drive a
`Debug_Resume_Core_Event`) and confirm `mem[2]==0xCC` post-resume —
validates the resume path end-to-end on silicon as a bonus.

### 4.3 Experiment 2 — Single_Step_Count characterization (G2)

Kernel = N distinct sequential marker stores (`mem[i]=i` for a known
N). Control packets write `Debug_Control0=(N<<2)` across a small matrix:
count alone; count + halt-bit; count with core enabled first vs before
enable. After each, `ReadRegisters` Debug_Status/Core_Status and count
landed markers. This characterizes arm / decrement / expire / re-arm as
far as silicon reveals. Whatever stays ambiguous becomes an explicit
documented modeling decision in Phase B (it is debugger-substrate-only;
no binary path depends on it).

### 4.4 Output

`docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md`
recording observed truth with raw evidence. The probe kernel stays
checked in as a re-runnable regression for the subsystem.

## 5. Phase B — implementation (parameterized on Phase A)

### 5.1 Halt boundary: synchronous vs asynchronous

Introduce the distinction the hardware actually has:

- **Synchronous trap halts** — PC_Event/breakpoint and single-step:
  bound to a specific instruction boundary. Their halt boundary is set
  to whatever Experiment 1 observes. If silicon halts before-commit and
  we currently halt after-commit, this is a real, scoped fidelity fix:
  evaluate the trap and the `DebugHalt` gate *before* the trap bundle
  commits for sync-origin halts. If silicon halts after-commit, the
  current model is *proven* and documented (no code change, only the
  finding + a guarding test).
- **Asynchronous halts** — host `Debug_Control0[0]` write, stall-halt,
  external event-halt: take effect at the next instruction boundary.
  Already correct; unchanged.

The split is the elegant core of the fix: a single, named seam between
"this halt is tied to *this* instruction" and "this halt happens
*soon*", rather than an ad hoc timing tweak.

### 5.2 Count-step state machine (debugger substrate)

A small arm / decrement / expire state machine local to
`core_debug/mod.rs`: pure logic, unit-testable in isolation, no
interpreter dependency — preserves the module's documented "projection
+ latch, does not drive the interpreter" character. `write_debug_control0`
arms a step budget from `[5:2]`; the coordinator decrements it per
committed bundle (one new consumer call adjacent to the existing
`consume_pending_single_step` at `coordinator.rs:643`) and requests
halt at expiry. Behavior follows Experiment 2 findings; unobservable
edges (re-arm on resume, count=0, halt-bit interaction) are implemented
to the most natural reading and documented inline as explicit modeling
decisions, citing the finding.

### 5.3 Component boundaries

- `src/device/core_debug/mod.rs` — count-step state machine; new
  synchronous-trap-pending concept if 5.1 needs it. Self-contained.
- `src/interpreter/engine/coordinator.rs` — single per-bundle drive
  point: `tick_single_step()` + correct sync-trap halt boundary.
- `src/interpreter/core/interpreter.rs` — refine the `DebugHalt`
  boundary for sync traps **only if Experiment 1 demands it**.
- probe kernel + findings doc — the derivation, durable.

## 6. Testing

- Unit (`core_debug/tests.rs`): arm-N, tick-N, expire-at-N, resume
  behavior, re-arm, count=0 no-op, halt-bit interaction. Pure state
  machine, no hardware.
- Interpreter-level: a sync-trap halt-boundary test asserting the
  *observed* timing (breakpoint at a known PC; assert the trap-bundle
  side effect did/did not land per Experiment 1); count-step end-to-end
  through the coordinator.
- Hardware probe = the derivation itself and a permanent re-runnable
  regression.
- `cargo test --lib` green (xdna-emu + xdna-archspec). Coverage
  regenerated in lockstep: `units.rs` narrative → true state, verdict →
  `Modeled { completeness: Full }` once G1+G2 closed, artifacts
  regenerated via
  `cargo run -p xdna-archspec --example gen_coverage_artifacts`,
  zero drift, committed with the seed change.

## 7. Risks

- Exotic count-step pokes could wedge the NPU. Mitigation:
  control-packet config writes (not `AIE_RW_ACCESS`), EMU-first,
  Experiment 1 banked before Experiment 2, recovery staged.
- Experiment 1 may invalidate the current after-commit model. That is
  the point — a scoped, derived fidelity fix, not scope creep.
- `Single_Step_Count` may be inert on silicon (effectively reserved).
  Then it is honestly documented as host-debugger-only emulator
  semantics; the verdict is still a defensible Full because the
  binary-reachable surface is complete and the unreachable surface is
  explicitly characterized, not silently faked.

## 8. Forward-commitment (tracked, deferred)

**Count-step silicon-fidelity.** If Experiment 2 leaves
`Debug_Control0[5:2]` behavior under-characterized (likely — silicon
never normally exercises it, and a full reverse-engineer may exceed
this round's hardware budget), Phase B ships the documented
debugger-substrate semantics now and records a tracked
forward-commitment to revisit *silicon-faithful* count-step
characterization when better register-poke tooling, AM025 RTL
documentation, or a dedicated hardware-observation budget is available.

This mirrors the established `AIE_AXIMM_Config.SLVERR_Block` precedent
(NoC-gated, forward-linked from the `noc` coverage gap): the path is
left open as a recorded goal, not closed. The commitment is recorded
here in Section 8 and surfaced in the `debug_halt` coverage narrative
so it is discoverable, not lost.

## 9. Open questions resolved by Phase A

These are the spec's explicit parameterization points — Phase B is
written against the answers, not guesses:

- Q1: Does silicon halt before or after the trap bundle commits?
  → Experiment 1. Sets the synchronous-trap halt boundary (5.1).
- Q2: What arms `Debug_Control0[5:2]` count-step, does it decrement,
  does it auto-clear/re-arm, does it require the halt bit?
  → Experiment 2 as far as observable; remainder → documented modeling
  decisions (5.2) + Section 8 forward-commitment.
