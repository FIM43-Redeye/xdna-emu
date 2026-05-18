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

**Observation is control-packet readback while the core is halted, and
this is load-bearing (revised — see 4.2 discovery).** The original
design used lock-gated DMA / plain output-BO readback as the decisive
path with control-packet `ReadRegisters` as mere corroboration. Phase A
execution (Task 3) proved that wrong: a working breakpoint halts the
core *before* it releases the lock that gates the marker DMA, so any
output-BO/DMA readback hangs forever and wedges the device. The only
core-independent way to read tile state while halted is the
control-packet OP_READ path (it reads tile data memory and registers
directly; emulator `registers.rs` `read_register_pure`, with prior art
in `add_one_ctrl_packet`). Control-packet readback of the marker buffer
*and* Core_Status is therefore the primary observation. (This promotes
what 4.3 earlier deferred as YAGNI to load-bearing — justified by the
discovered flaw.)

Run order: EMU-first (harness shakeout, expected-behavior baseline),
then a single HW session. NPU recovery staged per CLAUDE.md
(`modprobe -r amdxdna && modprobe amdxdna` first). **Experiment 1 runs
and is banked before Experiment 2** — high-value/low-risk result
secured before the exploratory pokes.

### 4.2 Experiment 1 — halt timing (G1)

**Discovery (Phase A, Task 3).** Two fatal flaws in the original Exp1
mechanism surfaced once the compiled core was disassembled, both masked
on EMU (the emulator drops control-packet writes to debug registers
into a `write_core_register` catch-all, so EMU never arms the
breakpoint — itself a recorded Phase B input: the control-packet →
`core_debug` register-write routing is unwired). The flaws:

1. *Source order ≠ schedule order.* The compiler reorders the marker
   stores; the trap-marker store was scheduled **first**, not in the
   middle. A source-order before/after model is inverted and misreads a
   real trap as "no trap."
2. *Lock-gated readback hangs on a halted core.* The marker DMA is
   gated by a lock the core only releases after finishing; a working
   breakpoint halts the core before that release, so DMA/output-BO
   readback blocks forever and wedges the NPU.

Forcing the compiler to preserve source store order was investigated
and rejected: no MLIR-level mechanism reliably pins it (`chess_separator`
is a Peano no-op; mlir-aie does not lower `volatile` memref semantics;
RAW chains fall to alias analysis), and our own disasm empirically
confirms reordering. The redesign therefore does **not** fight the
scheduler.

**Redesigned mechanism.** The kernel writes four distinct markers to
distinct `output_buffer` slots. The trap bundle (where `PC_Event0` is
armed, `Debug_Control2[0]=1`) is identified from `llvm-objdump-aie` of
the compiled core, with the slot⇄schedule-order map *derived from that
disassembly* (not source order) and kept fresh via the TRAP_PC
re-derivation discipline already committed in Task 3. For the current
build the schedule is: `0x114` trap → `output_buffer[1]=0xBB`; strictly
later `0x11c` → `output_buffer[0]=0xAA`; then `[3]=done`; then
`[2]=0xCC`.

Observation is **control-packet OP_READ while the core is halted**
(core-independent; the lock-gated DMA is unusable per the discovery):
read the four `output_buffer` slots *and* `Core_Status` (0x32004).
**The lock-gated marker-DMA path (`@out0`/`aie.mem` MM2S) is removed
entirely from the probe** — it is fully redundant (the OP_READ is the
sole observation) and a `dma_wait @out0` on a halted core blocks
forever and wedges the NPU (a second pre-hardware wedge bug caught at
Task-4 code review; the earlier "leave @out0, harmless" framing was
wrong). Nothing in the probe depends on the halted core releasing a
lock.

The verdict is computed from the disassembled schedule, using the
`Core_Status` **DEBUG_HALT bit (16) alone** — `halted ≡ Core_Status &
(1<<16)`. Requiring the ENABLE bit was rejected: ENABLE-stays-1-while-
halted is an unverified hardware assumption, and DEBUG_HALT=1 already
proves the core ran and is debug-halted (you cannot be debug-halted
without having executed):

- `halted` and trap slot (`[1]`) committed, no strictly-later slot
  written → **halt is after-commit**.
- `halted` and trap slot not committed (all slots zero) → **halt is
  before-commit**. (DEBUG_HALT=1 itself disambiguates this from "core
  never ran": a core that never ran is not debug-halted, so it lands in
  the not-halted branch as an anomaly, never as before-commit.)
- not `halted`, all four slots written → breakpoint did not fire
  (`NO_TRAP_OR_RAN_TO_END`).
- not `halted`, slots not the full set → `ANOMALY_NOT_HALTED` (record
  raw).

**EMU limitation (accepted).** Because EMU drops the breakpoint-arming
writes (write-side `write_core_register` catch-all), EMU cannot produce
a halted state. Additionally there is a *symmetric read-side gap*:
control-packet OP_READ of core/debug registers (e.g. `Core_Status`
0x32004) is not wired into `core_debug` (`read_register_pure` has no
core_debug dispatch), so EMU returns `Core_Status = 0x0` even for a
completed core. Both are recorded Phase B inputs (control-packet ⇄
`core_debug` register routing, read and write sides; possibly multiple
inconsistent register-access paths to reconcile). Consequently EMU
validates only the readback *mechanism* and the no-trap baseline; the
read-*while-halted* semantics and the G1 answer itself are first
exercised on hardware (where Core_Status is populated correctly). This
is understood, not a flaw — HW is the ground truth for G1 regardless.

Resume is not exercised by this probe (a runtime sequence cannot
deassert the breakpoint mid-run and re-observe); it stays in-emulator-
tested + Phase-B-revalidated, with hardware resume-verification a
tracked Section 8 forward-commitment.

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

**Resume hardware-verification.** Phase A's probe derives halt timing
but does not hardware-confirm *resume* (a runtime sequence cannot
deassert the breakpoint mid-run and re-observe a second core pass).
Resume is wired and tested in-emulator and is re-validated in Phase B's
interpreter-level tests; the missing piece is a real-silicon
confirmation. Given how many debug behaviors on this NPU have proven
to be unknowns until probed, hardware-verifying resume is a tracked
later-consideration: worth a dedicated probe pass eventually, not a
Phase A/B blocker. Recorded here so it is not lost.

**Probe-artifact robustness: `OUTBUF_ADDR`.** The redesigned Exp1/Exp2
observation reads `output_buffer` via control-packet OP_READ at a
tile-local address (`OUTBUF_ADDR`, expected `0x0400`) derived from the
disassembly / aiecc allocation — a fragile magic constant of the same
class as `TRAP_PC`. Phase A makes it self-documenting (an in-artifact
"derived, re-verify if the kernel/allocation changes" warning, parallel
to the committed TRAP_PC discipline) and self-checking (the EMU no-trap
readback must return the known marker values, else re-derive). That is
sufficient for Phase A correctness, but it is not *robust*: a future
maintainer editing the kernel could silently desync it. Tracked
later-consideration: make `OUTBUF_ADDR` non-fragile (derive it
programmatically — e.g. from the aiecc allocation map / a symbol — so
it cannot rot), not a Phase A blocker. Recorded here so it is not
ignored.

## 9. Open questions resolved by Phase A

These are the spec's explicit parameterization points — Phase B is
written against the answers, not guesses:

- Q1: Does silicon halt before or after the trap bundle commits?
  → Experiment 1. Sets the synchronous-trap halt boundary (5.1).
- Q2: What arms `Debug_Control0[5:2]` count-step, does it decrement,
  does it auto-clear/re-arm, does it require the halt bit?
  → Experiment 2 as far as observable; remainder → documented modeling
  decisions (5.2) + Section 8 forward-commitment.
