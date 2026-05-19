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
`[2]=0xCC` (this schedule is for the pre-gate build; the objectfifo
gate below changes it — re-derived per the TRAP_PC discipline).

**Arming-race discovery (Phase A, Task 5 hardware run) + the gate.**
The first hardware run did not halt: `Core_Status=0x100000` (CORE_DONE),
all markers written. Root cause, confirmed by toolchain grounding: the
compute core is enabled by the **CDO**, *before* the host
`aie.runtime_sequence @seq` executes (`AIETargetCDODirect.cpp`,
`AIERT.cpp addCoreEnable()`). With the entry lock immediately
satisfied, the tiny core ran to completion in nanoseconds — long before
the `@seq` control-packet arming `write32`s reached the tile. No
`@seq`-only ordering can win this race. Two framed fixes were grounded
and **rejected**: there is no host lock-release op (`aiex` dialect has
none), and no mlir-aie construct emits arbitrary debug-register writes
into the CDO. A naive memory-poll gate was also rejected: AIE2 has no
honored volatile/fence, so a spin-wait load is hoisted/eliminated (the
same wall that killed force-ordering).

The supported, robust fix is a **blocking objectfifo gate**: the core's
first op is `aie.objectfifo.acquire(@gate, Consume, 1)`, which lowers to
the `llvm.aie2.acquire` hardware-blocking intrinsic (a real pipeline
stall, immune to load-elimination — same primitive class as the core's
existing `use_lock`). `@seq` issues the arming `write32`s **then** a
`dma_memcpy_nd` that feeds `@gate`; runtime-sequence ordering is a
documented in-order guarantee (control-packet then DMA, preserved by
`AIEToConfiguration`). The core therefore cannot execute its first real
instruction until the host has armed the breakpoint and then released
the gate. Adding the gate changes the compiled schedule, so `TRAP_PC`
and the slot⇄order map are re-derived from a fresh disasm (the Task-3
re-derivation discipline is exactly this mechanism). Bound: if a HW run
*still* does not halt despite the documented ordering guarantee, G1
becomes a tracked §8 forward-commitment rather than a further redesign.

**Shim-channel-disjointness constraint (Task 5 Step 3 discovery).** The
gate's shim feed and the ctrl-in OP_READ push **must use disjoint shim
MM2S channels**. The first gated EMU run timed out: `@gate` defaulted
its shim feed to shim MM2S channel 0, and the hand-rolled ctrl-in push
(register block `0x1d210/0x1d214`, inherited from the
`add_one_ctrl_packet` idiom) also targets shim MM2S channel 0. With one
shim source feeding two compute-tile destinations, the pathfinder
correctly compiles a single **circuit broadcast** (`switchbox(0,2):
South:1 → {TileControl:0, DMA:0}`, visible in `input_physical.mlir`);
aie-rt confirms circuit masters forward their slave unconditionally
(`xaie_ss.c:128-164,187-261`, `MstrPktEn=0`, no header parsing). On
silicon this byte-copies every ctrl-in header into the `@gate` S2MM
buffer and every gate token into TileControl — corruption,
`dma_wait @ctrl0` never satisfies, `run.wait()` timeout, NPU wedge. The
emulator faithfully reproduced this (no EMU bug; the probe was at
fault).

The first-considered fix — pinning `@gate` to shim MM2S ch1 via an
explicit `aie.shim_dma_allocation` — is **not honored by the
toolchain** (grounded): the objectfifo lowering's `DMAChannelAnalysis`
seeds channel occupancy only from `ShimDMAOp`/`MemOp`/`MemTileDMAOp`/
`FlowOp` and *never* from `ShimDMAAllocationOp`
(`AIEObjectFifoStatefulTransform.cpp:84-120`); there is no
per-objectfifo channel attribute (`AIEOps.td:1825-1848`); the
objectfifo unconditionally takes the lowest free channel
(`getDMAChannelIndex`, `:125-152`), and a pre-placed
`@gate_shim_alloc` collides with the symbol the lowering itself emits
(`:1752`). The only objectfifo-side lever is a dummy `aie.shim_dma`
occupier, which emits real BD/channel config into the CDO (undocumented,
no in-tree precedent).

**Adopted fix:** leave `@gate` on its natural default shim MM2S **ch0**
and repoint the hand-rolled ctrl-in OP_READ push to shim MM2S **ch1**
(the controllable side — plain register writes, fully characterized).
The toolchain-derived delta is mechanical: shim DMA per-channel stride
is `0x8` (aie-rt `xaiemlgbl_reginit.c` `.ChIdxOffset = 0x8`), so the
ctrl-in CTRL `0x1d210→0x1d218` and TASK_QUEUE `0x1d214→0x1d21c`, the
ctrl-in `aie.packet_flow` source `<%shim, DMA:0>→<%shim, DMA:1>`, and
the `aiex.npu.sync` MM2S `channel 0→1`; no BD-layout or BD-address
changes (BDs are channel-independent). `add_one_ctrl_packet` already
proves hand-rolled-MM2S + objectfifo channels coexist disjointly on one
shim tile. This is a durable constraint on the probe, not a one-off:
the gate's and the ctrl-packet's shim MM2S channels must always be
disjoint, and because the objectfifo channel is not pinnable, the
ctrl-packet side is the one that must move.

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

**Halt-synchronization (Task 5 P2/P3 discovery + MASKPOLL fix).** The
static-`@seq` OP_READ has *no happens-after* against the core halt:
nothing orders "core halted at the trap PC" before "OP_READ reads
`output_buffer`" — it was a pure relative-latency assumption (grounded
P2/P3: the gated core's marker stores land *after* the EMU run loop
terminates at `dma_wait @ctrl0`, `execution.rs:205-209`; and on silicon
the on-die halt and the host/NoC OP_READ have no synchronization
primitive between them — the same failure class that broke arming
(attempt 1) and shim routing (attempt 2)). Grounding established there
is **no on-device halt→lock/event actuation** on AIE2 (the event system
is observational, not actuating: regdb has lock→event only, no
event→lock-set; `DMA_*_Start_Queue` has no event field), and
`aie.runtime_sequence` is strictly static (no poll/conditional op;
`AIEX.td`). aie-rt's own debugger precedent is a host-side
`XAie_MaskPoll` spin on the halt bit *then* read (`xaie_core.c:65-99`).
The robust, synchronization-ordered fix is a firmware-level
**`XAIE_IO_MASKPOLL`** instruction (opcode 4) post-compile-injected into
the probe's `insts.bin`, positioned after the gate-feed `dma_memcpy_nd`
and before the first OP_READ push: it blocks the instruction stream
until `Core_Status & (1<<16)` (`DEBUG_HALT`) is set, so the OP_READ
provably executes *after* the core has halted. No MLIR dialect emits
this (`NpuMaskPollOp` does not exist; `AIETargetNPU.cpp`), so a
post-compile binary patch is the only route — consistent with the
probe's existing hand-rolled control-register writes and the trace
stack's `insts.bin` patching. Byte-exact streamed form (28 bytes, no
timeout field — `TimeOutUs` is dropped at serialization,
`xaie_controlcode.c:620`): `reg_off = (col<<25)|(row<<20)|0x32004`
(absolute NPU address; `col=0` logical, `row=2`), `value = mask =
0x00010000`. The injector must bump the `insts.bin` header op-count
(+1) and byte-size (+28); anchor the injection point to the first
ctrl-in MaskWrite32 at tile-local `0x1d218` (channel-robust, not
instruction-count-based).

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

**Mechanism correction (grounded post-Phase-B-Unit-1).** Earlier
revisions of this section attributed the EMU's inability to observe a
halt to a *write-side* gap ("EMU drops the breakpoint-arming writes,
`write_core_register` catch-all, `compute.rs:554`"). **That is false.**
Both the control-packet path *and* the probe's `@seq npu.write32`
arming path share the same sink: `write_tile_register → dispatch →
apply_tile_local_effects (effects.rs:296-298) → core_debug.write_register`.
The debug/PC offsets `0x32010`–`0x3202C` are classified
`SubsystemKind::Debug`/`ProgramCounter` by `subsystem_from_offset` and
**bypass `write_core_register` entirely** — its `_ => {}` arm is
unreachable for them. `pc_event0`/`debug_ctrl2` have been settable in
`core_debug` since `8f52355`/`ed647b0` (pre-Task-5). **The breakpoint
was armed on EMU all along.**

The true reason the Task-5 EMU run produced `MASKPOLL_UNSATISFIED_EMU`
is a **read-side gap on the mutable `tile.read_register` path** — the
one the injected MASKPOLL polls `Core_Status` (0x32004) through. It
falls back to the raw register HashMap, which never reflects the
*dynamically computed* `Core_Status[16]=DEBUG_HALT` (derived from
`core_debug.read_status()`), so the MASKPOLL always read 0 and never
satisfied. This is **distinct from** the `read_register_pure` gap
(control-packet OP_READ path) that Phase B Unit 1 (`e0ec922`) fixed —
the two read paths diverging *is* the "possibly multiple inconsistent
register-access paths to reconcile" recorded here as a Phase B input.

**Phase B Unit 1b — reconcile the read paths; EMU reproduces G1.**
Dispatch the mutable `tile.read_register` `Core_Status`/debug-register
reads into `core_debug` (mirroring the `read_register_pure` fix Unit 1
made on the other path), closing that recorded inconsistent-paths
input. With arming already wired, the Unit-1 pre-execute seam halting
*before* the trap bundle commits, and the MASKPOLL now able to observe
`DEBUG_HALT`, the **EMU reproduces the hardware result**: MASKPOLL
satisfies → OP_READ issues → all marker slots zero + `DEBUG_HALT` →
`TRAP_VERDICT:BEFORE_COMMIT` on EMU, matching HW. The probe is thereby a
**self-checking EMU+HW regression of the G1 before-commit fidelity
fix**, not a HW-only derivation with an EMU placeholder. The earlier
"G1 is HW-only by construction" framing is retired; HW remains ground
truth, EMU is now a faithful regression of it.

**Retained: emulator graceful poll-termination (independent hardening).**
The contract added in Phase A — an unsatisfiable `BlockedOnPoll` with a
quiescent engine terminates **deterministically and honestly** (distinct
`MaskPollUnsatisfied` reason, clean `run.wait()`, **no** register
fakery, no pretend-halt, no skipped OP_READ) — remains valid and is
independently unit-tested (poll satisfied immediately / after N /
never). It is no longer the *probe's* EMU baseline (Unit 1b makes this
probe's MASKPOLL satisfy on EMU), but it stays as general emulator
robustness for any genuinely-unsatisfiable poll. EMU and HW still run
the *identical* injected binary (byte-parity).

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

- **Synchronous PC-event/breakpoint halts** — **G1 derived (Task 5b):
  silicon halts BEFORE the trap bundle commits**; the emulator currently
  evaluates the trap + `DebugHalt` gate *after* `update_pc` (after the
  bundle's side effects commit). Scoped fidelity fix: a **pre-execute
  seam** in the coordinator — for PC_Event/breakpoint origin, detect the
  PC match and apply `DebugHalt` *before* the bundle's side effects
  execute, so the trap bundle's store never lands. PC_Event/breakpoint
  origin **only** (see single-step note).
- **Single-step halts — deferred to §5.2/G2, not this unit.** Single-
  step is synchronous in principle, but before-commit is only well-
  defined for PC_Event-wired single-step; an `SSTEP_EVENT` armed by a
  watchpoint/mid-execution event has *no* meaningful before-commit point
  (its arming condition is only known after the bundle runs — a genuine
  §5.1 design ambiguity, not just a hardware question). Single-step is
  also naturally coupled to the count-step substrate (§5.2), which is
  G2-dependent. The single-step halt boundary is therefore scoped out of
  this unit and addressed with §5.2/G2. Two existing tests
  (`core_debug/tests.rs:989,1046`) currently assert the after-commit
  single-step model and stay valid until then.
- **Asynchronous halts** — host `Debug_Control0[0]` write, stall-halt,
  external event-halt: take effect at the next instruction boundary.
  Already correct; the pre-execute seam is additive and provably does
  not touch these (they keep setting `halted`, caught at the existing
  `interpreter.rs:181` gate).
- **Coupled read-path reconciliation (Unit 1 + Unit 1b).** *Correction:*
  there is **no write-side gap** — control-packet *and* `@seq npu.write32`
  debug-reg writes both reach `core_debug` via `apply_tile_local_effects`
  (see "Mechanism correction", §4.2). The real recorded input is the
  divergent **read** paths: Unit 1 dispatched `read_register_pure`
  (control-packet OP_READ path) into `core_debug`; **Unit 1b** dispatches
  the mutable `tile.read_register` `Core_Status`/debug-reg reads (the
  injected-MASKPOLL path) the same way, reconciling the two and closing
  the "inconsistent register-access paths" Phase B input. With arming
  (always wired) + the Unit-1 before-commit seam + Unit-1b read
  reconciliation, the **EMU reproduces the HW `BEFORE_COMMIT` result** —
  the probe becomes a self-checking EMU+HW regression of this fidelity
  fix.

The async/sync split remains the elegant core; this unit lands the
PC-event half (the part fully derived by G1 and unentangled with G2).

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

**This unit (PC-event before-commit + routing gaps):**
- `src/interpreter/engine/coordinator.rs` — add the **pre-execute
  PC_Event seam** before the bundle executes (before
  `step_with_neighbor_locks`, ~:624): if the next PC matches an armed
  PC_Event with `PC_Event_Halt`, halt *without* executing the bundle;
  condition the existing post-execute `update_pc` (:638) so it does not
  re-fire the same match after resume.
- `src/device/core_debug/mod.rs` — a `sync_trap_pending`-style query
  (`has_sync_pc_trap_at(pc)`) that does not itself commit the halt;
  control-packet register dispatch reachable from the routing-gap fix.
- `src/device/tile/registers.rs` — **Unit 1:** dispatch
  `read_register_pure` (control-packet OP_READ path) `Core_Status`/
  debug-reg reads into `core_debug` (live-computed status). Writes need
  no change (already wired via `apply_tile_local_effects`).
- `src/interpreter/core/interpreter.rs` — `is_halted()` gate at :181
  unchanged (the coordinator sets `halted` pre-execute; the existing
  gate still catches it). No change expected here.
- findings doc — the derivation, durable (committed `96ecb6b`;
  EMU section updated by Unit 1b to record EMU reproduces `BEFORE_COMMIT`).

**Unit 1b (read-path reconciliation; EMU reproduces G1):**
- `src/device/tile/registers.rs` — dispatch the **mutable**
  `tile.read_register` `Core_Status`/debug-reg reads into `core_debug`,
  mirroring the Unit-1 `read_register_pure` fix (the injected MASKPOLL
  polls via this path). Closes the §4.2 inconsistent-paths input.
- `mlir-aie .../debug_halt_probe/test.cpp` — EMU verdict expectation
  flips from `MASKPOLL_UNSATISFIED_EMU` to the schedule-derived
  `BEFORE_COMMIT` (EMU now reproduces HW); HW path unchanged.
- findings doc EMU section — updated to "EMU reproduces `BEFORE_COMMIT`;
  probe is a self-checking EMU+HW regression" (retire the HW-only/
  `MASKPOLL_UNSATISFIED_EMU`-baseline framing).
- The emulator graceful-poll-termination contract + its unit tests stay
  (independent hardening; no longer the probe's path).

**Deferred (with §5.2/G2):** count-step state machine; single-step halt
boundary; `tick_single_step()` per-bundle drive — out of this unit.

## 6. Testing

- Unit (`core_debug/tests.rs`): pre-execute sync-PC-trap query
  (`has_sync_pc_trap_at`) arm/match/no-match; control-packet debug-reg
  write/read dispatch round-trips through `core_debug` (routing-gap
  fix). Pure, no hardware. (Count-step arm-N/tick-N/expire/resume unit
  tests are deferred with §5.2/G2.)
- **Guarding test (this unit): a coordinator-level before-commit
  assertion** — arm PC_Event0 at a known `TRAP_PC` whose bundle stores
  to a known tile data address, step, assert the core is `DebugHalt`
  **and the store did NOT land** (before-commit, per G1). Now reachable
  end-to-end because the routing-gap fix lets the test arm PC_Event via
  the control-packet path. **Known risk / approach decision:** this
  needs a real encoded AIE2 store-bundle in program memory (existing
  interpreter tests use raw NOP bytes; no pre-baked store encoding is
  known). Resolve up front, in order of preference: (a) reuse an
  encoded store bundle from an existing compiled test binary/fixture;
  (b) hand-encode one from the llvm-aie ISA and document the encoding;
  (c) only if neither is feasible, fall back to a coordinator-level
  state-machine assertion (halt fires pre-execute; PC not advanced)
  and explicitly record that the literal store-did-not-land assertion
  is covered by the hardware probe, not the unit suite. The implementer
  must surface which of (a)/(b)/(c) before settling for a weaker test.
- Hardware probe = the durable derivation + permanent regression. Post
  Unit 1b it is **self-checking on EMU too**: the EMU bridge run
  reproduces the HW `TRAP_VERDICT:BEFORE_COMMIT` (arming wired + Unit-1
  before-commit seam + Unit-1b read reconciliation), so a regression in
  the before-commit fidelity fix fails the EMU bridge, not only HW.
- Unit-1b read-path test (`registers.rs`/integration): mutable
  `tile.read_register` of `Core_Status` reflects `core_debug` halt state
  (DEBUG_HALT bit 16), consistent with `read_register_pure`; the
  emulator graceful-poll-termination unit tests remain green (the
  contract is retained, just not the probe's path).
- `cargo test --lib` green (xdna-emu + xdna-archspec). Coverage
  regenerated in lockstep. **Completeness stays < Full after this
  unit** — it closes the G1 (sync PC-event halt-boundary) surface and
  the core_debug routing gaps; `Modeled { completeness: Full }` still
  requires G2 + the deferred single-step/count-step substrate.
  Regenerate artifacts via
  `cargo run -p xdna-archspec --example gen_coverage_artifacts`, zero
  drift, committed with the change.

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
- The streamed MASKPOLL has **no timeout** (no watchdog found in open
  sources — a hardware-fork unknown; assume infinite poll). If the core
  does **not** halt on HW (arming still fails despite the gate), the
  MASKPOLL blocks the instruction stream forever and wedges the NPU.
  Mitigation: hardware wedges are greenlit-survivable (recover per
  CLAUDE.md `modprobe -r amdxdna && modprobe amdxdna`, staged); the G1
  HW step is **bounded** — a second hang after one recovery+retry stops
  the experiment and scopes G1 as the §8 forward-commitment rather than
  a further redesign.

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

**G1 halt-timing (bounded escalation).** The Phase A probe derives G1
(synchronous-trap halt boundary, §5.1) on hardware via the MASKPOLL
halt-synchronized OP_READ. If the HW core does not halt — MASKPOLL never
satisfies — even after one recovery+retry (the §7-bounded HW step),
G1 is **not** force-concluded and the probe is **not** redesigned a
further time. Instead Phase B ships the emulator's current
after-commit synchronous-trap model as the **explicit documented
assumption** (§5.1: model proven only if observed; otherwise stated as
the unverified default), and a tracked forward-commitment records
deriving G1 when better arming/observation tooling or a dedicated
hardware-observation budget is available. Same posture as count-step
above: open recorded goal, not closed; surfaced in the `debug_halt`
coverage narrative.

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

**`Core_Status` RESET-bit EMU/HW divergence.** Surfaced by the Unit 1b
code-quality review: when debug-halted at the trap, HW reports
`Core_Status = 0x10001` (DEBUG_HALT|ENABLE) but EMU reports `0x10003`
(DEBUG_HALT|RESET|ENABLE) — EMU leaves bit 1 (`RESET`) set.
`CoreDebugState.reset` defaults to `true` and `Coordinator::enable_core()`
calls `set_enabled(true)` directly, bypassing `write_control()` (which
would clear `reset`); a CDO `Core_Control=0x1` write *does* route through
`write_control` and clear it, so the residual `RESET` in the observed
run indicates the enable path, not the CDO write, set `enabled`. This
does **not** affect G1 or the probe verdict (which keys solely on
DEBUG_HALT bit 16; before-commit also requires all marker slots zero),
and the findings doc documents the divergence. Tracked
later-consideration: reconcile `enable_core()` with `write_control()` so
EMU `Core_Status` matches silicon bit-for-bit when halted — improves the
self-checking regression's fidelity, not a Phase A/B blocker. Recorded
here so it is not lost.

## 9. Open questions resolved by Phase A

These are the spec's explicit parameterization points — Phase B is
written against the answers, not guesses:

- Q1: Does silicon halt before or after the trap bundle commits?
  → Experiment 1. Sets the synchronous-trap halt boundary (5.1).
- Q2: What arms `Debug_Control0[5:2]` count-step, does it decrement,
  does it auto-clear/re-arm, does it require the halt bit?
  → Experiment 2 as far as observable; remainder → documented modeling
  decisions (5.2) + Section 8 forward-commitment.
