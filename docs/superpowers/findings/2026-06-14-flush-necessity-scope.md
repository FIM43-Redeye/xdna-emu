# Scope: Is `flush_ctrl_packets` necessary? (and can we delete it for true correctness)

**Date:** 2026-06-14
**Task:** #133 (tenant-4 tail-collapse) -> reframed
**Status:** SCOPE / investigation complete, awaiting decision on execution

## TL;DR

The tenant-4 tail-collapse is a symptom of `flush_ctrl_packets`, an
inter-instruction control-packet "fast-forward" that runs the stream router up
to 8x at a frozen simulation cycle. The prior fix (`route_only_step`) stopped it
re-stepping the DMA *engines* but left it advancing *circuit-switched producer
data*, which over-delivers the memtile fill at end-of-stream (~2 words/cycle vs
the physical 1 word/cycle wire rate).

**The decisive finding of this scope: the flush is almost certainly vestigial.**
Its entire reason to exist -- "the emulator's NPU executor fires instructions
faster than the fabric can deliver control packets" -- was true under the OLD
default cost profile (`legacy_one_per_packet`, 1 cycle/instruction). It is no
longer true: the active default is `provisional_npu1`, an HW-calibrated
firmware-latency model in which every control-packet-emitting instruction costs
>=100 cycles, and the normal per-cycle `step()` path already delivers and
applies control packets continuously. The firmware-latency model we were about
to "add" is **already built and live**.

So "model firmware latency so the flush disappears" is not a from-scratch
recalibration monster. The model exists. The work is: **verify the flush is
removable, delete it, and re-confirm the existing calibration still holds.**

## What the investigation established

### 1. The firmware-latency model already exists and is active

- The executor has a `RetiringInstruction` state driven by `CycleCostModel`
  (`src/npu/cycle_cost.rs`). After an instruction *executes* (emitting any
  control packet), the executor parks for `cost-1` cycles before the next
  instruction issues. During those cycles the engine steps normally.
- The active default profile is `provisional_npu1()` (`cycle_cost.rs:455`,
  consumed via `CycleCostModel::default()` at `executor.rs:236`). It is
  HW-calibrated (the 2026-05-27 N=50 campaign): per-tile-type BD config 100 cyc,
  MaskWrite +110, BlockWrite +13/word, fabric hops 3-4, post-sync mailbox 8000,
  and the per-direction Task_Queue dispatch gate (MM2S 1086/1964/3050, S2MM flat
  3050).
- `legacy_one_per_packet()` (every instruction = 1 cyc) still exists but is NOT
  the default. Under it, the flush is genuinely load-bearing.

### 2. The normal per-cycle path already delivers AND applies control packets

`InterpreterEngine::step()` (`coordinator.rs:563`), Phase 3, every cycle:
- `step_data_movement()` -> `route_streams()` advances control packets one hop
  through the fabric (`coordinator.rs:969-970`);
- then `drain_ctrl_packet_actions()` + `dispatch_ctrl_action()` apply the
  completed register writes (`coordinator.rs:985-988`).

So control-packet application is NOT flush-exclusive. It runs continuously during
normal stepping. The flush only does *additional* frozen-cycle routing passes
when an instruction progressed.

### 3. Cost audit: no control-packet emitter is cheap enough to need the flush

`CycleCostModel::cost_of` (`cycle_cost.rs:363`) under `provisional_npu1`:

| Instruction | Emits ctrl packet? | Cost | Fabric delivery latency |
|---|---|---|---|
| `Write32` (BD/reg) | yes | ~104-120 cyc | <=~23 cyc (5 row-hops x4 +3) |
| `BlockWrite` | yes | >=114 cyc | <=~23 cyc |
| `MaskWrite` | yes | >=214 cyc | <=~23 cyc |
| `DdrPatch` | yes | 100 cyc | <=~23 cyc |
| `MaskPoll` | no (status read) | 1 cyc | n/a |
| `Sync` | no (status read) | 1 cyc | n/a |

Because the retirement cost is charged *after* the packet is emitted, every
control packet gets >=100 cycles of normal routing+dispatch before the next
instruction issues -- a >4x margin over the worst-case fabric traversal. The
only 1-cycle instructions (`MaskPoll`/`Sync`) read status written by earlier,
already-delivered instructions. **No instruction sequence under
`provisional_npu1` requires the flush for correct ordering.**

## Why the flush still hurts: the tail-collapse

At end-of-stream the executor walks teardown/sync (`MaskPoll`/`Sync`, cost 1),
so it *progresses every cycle*, firing the flush every cycle. Each flush runs
`route_only_step` up to 8x at a frozen cycle, and although it no longer steps the
DMA engines, each routing pass still advances *circuit-switched producer data*,
pre-staging the memtile S2MM input FIFO from upstream fabric slack (shim output
FIFO cap 4, switch FIFOs). The next real `step_all_dma` drains that at the wide
internal bus rate, so the fill sustains ~2 words/cycle until the slack drains
(~11 cycles) -- faster than the 1 word/cycle stream wire physically allows.
aiesim's VCD confirms HW holds 1 word/cycle for every buffer including the last.

The flush thus contributes nothing positive under the active profile and
actively introduces an unphysical end-of-stream burst.

## Proposed change

**Delete `flush_ctrl_packets` (the in-run flush at `backend.rs:293-295`); rely on
the existing firmware-latency model + normal per-cycle control-packet delivery.**

- `flush_trace_to_host` (post-run trace drain, `coordinator.rs:278`) is SEPARATE
  and unaffected -- it runs after the loop exits, when timestamps are already
  baked, and legitimately needs DMA stepping to drain trace bytes to DDR. Leave
  it alone.
- `route_only_step` becomes dead code once the in-run flush is gone (only caller
  is `flush_ctrl_packets`). Remove it too, or keep as a tested utility. TBD.

This is the "true correctness" path: the flush was a proxy for firmware latency;
the real firmware-latency model supersedes it; remove the proxy.

## Refinement: executor config writes are IMMEDIATE, not fabric-routed

`execute_write32` (`executor.rs:935`) applies the register synchronously via
`device.write_tile_register(col,row,offset,value)` -- the executor's BD/config
writes are NOT packet-switched stream traffic. They land in-place, instantly, in
sim. So the "control-packet emitter cost window" framing below is imprecise: those
writes need no delivery window at all.

The flush's ONLY real remaining function is fast-forwarding **fabric-routed
control packets** -- genuine packet-switched stream data (the control-packet
kernel category, tasks #65/#69), whose delivery is paced by stream routing
(1 hop/cycle via `route_streams`), not by executor instruction cost. The flush
runs `route_only_step` up to 8x per progression, advancing such a packet up to 8
hops in zero sim-time vs the physical 1 hop/cycle.

Consequence for deletion:
- **Executor-direct-config kernels (most of the corpus, incl. tenant-4):** the
  configs never went through the flush. Deletion is trivially safe; the flush's
  only effect there is the parasitic tail-collapse.
- **Fabric-control-packet kernels:** deletion changes their control-packet
  delivery from "fast-forwarded" to physical 1 hop/cycle. More correct, but
  slower. Risk = a kernel that fired a dependent executor read inside the window
  the flush used to mask. This is exactly what the hazard detector watches and
  the bridge (with its control-packet kernels) validates.

## Feasibility risks (what could make deletion wrong)

1. **A control-packet emitter cheaper than fabric latency.** Audit says none
   exist under `provisional_npu1`. RISK: LOW, but the bridge is the safety net.
2. **Fabric congestion delaying a control packet past the retirement window.**
   Control packets are setup-time and/or packet-switched (separate from circuit
   data routes); the >4x margin absorbs normal latency. RISK: LOW; bridge covers
   it.
3. **A non-default profile path.** If any production path runs
   `legacy_one_per_packet`, the flush is load-bearing there. Need to confirm the
   bridge/FFI path always uses `provisional_npu1` (it does via
   `CycleCostModel::default()`, but verify no override). RISK: LOW.
4. **A kernel that depends on the flush's fast-forward for a legitimate ordering
   not covered by instruction cost** (e.g., a control read immediately after a
   fast write to the same tile). Audit suggests impossible, but only the 169
   kernel bridge can prove the corpus is clean. RISK: this is exactly what the
   bridge validates.

## Blast radius (reframed -- much smaller than a from-scratch K)

The earlier blast-radius survey assumed we were *adding* firmware latency from a
legacy-1-cycle baseline. We are NOT: the latency is already live in the active
calibration. Removing the flush changes only end-of-stream circuit-data timing
(the tail stops collapsing -- the DESIRED change, matching HW's flat fill).

| Subsystem | Effect of deleting the in-run flush |
|---|---|
| DMA FSM / lock / memory timing (`timing.rs`) | none -- intra-DMA, flush-independent |
| Dispatch gate / mailbox / warmup calibration | re-VERIFY only; the firmware latency itself is unchanged. The flush was present during their calibration, so end-of-stream tails shift slightly; re-confirm gaps still match HW. |
| Trace per-anchor timing | tail anchors shift to the physically-correct (flat) values; this should IMPROVE the producer-fill axis. Re-run the three-way timing to confirm no regression elsewhere. |
| Golden/baseline traces | re-verify against HW; the tenant-4 producer probe should go flat (the target). |
| Unit tests asserting cycle counts | ~3-5 may need updating; `route_only_step` test removed/retargeted. |

This is "re-verify the existing calibration with a vestigial shortcut removed,"
not "recalibrate the firmware model from scratch."

## Validation plan (TDD + HW)

1. **TDD**: replace the `route_only_step` contract test with one asserting that,
   under `provisional_npu1`, a BD-config write's control packet reaches the
   target register via normal stepping within its retirement window -- i.e., the
   ordering the flush used to guarantee is now guaranteed by instruction cost.
2. **Delete** the in-run flush; remove now-dead `route_only_step` (decide).
3. `cargo test --lib` green.
4. Rebuild the FFI `.so`; re-run the tenant-4 producer probe. **Success = the
   last fill goes flat at 1 word/cycle** (FINISHED_BD/REL tail matches steady),
   matching aiesim/HW.
5. **Full 169-kernel bridge** (the corpus-wide safety net for control-packet
   ordering). Compare against the pre-change baseline trace.log per the
   verify-against-baseline discipline.
6. Re-confirm the dispatch-gate / warmup / mailbox gaps still match the
   2026-05-27 HW campaign (spot-check the calibration kernels).

## Fallback

If the bridge surfaces a real flush dependency (some kernel where deletion
breaks control-packet ordering), fall back to the **packet-only flush**: restrict
the flush to packet-switched traffic only (the stream switch already separates
packet vs circuit routing -- `step_packet_routes`, the bit-30 packet-mode slave
flag). That fixes the tail-collapse without removing control-packet fast-forward.
It is the safe, smaller fix; deletion is the correct, cleaner one. The bridge run
decides between them on evidence.

## Execution results (2026-06-14)

Implemented: flush deleted; `route_only_step` removed; control-packet ordering
hazard detector added (`InterpreterEngine::note_ctrl_packet_ordering_hazard` +
`ctrl_packet_hazard_count`, screening on `TileArray::has_pending_control_packet`).

- **Lib tests:** 3497 pass / 0 fail (TDD: detector primitive + engine counter,
  plus a trace-exclusion assertion).
- **Tenant-4 producer probe (PRIMARY GOAL -- tail flat):** the last fill now
  paces at the steady cadence, eliminating the collapse entirely.

  | | steady gap | last-fill gap |
  |---|---|---|
  | HW (NPU1 truth) | 2119 | **2119 flat** |
  | route_only_step (prior patch) | 2125 | 2105 (collapse -20) |
  | **flush deleted (this work)** | 2125 | **2125 flat** |

  Deletion is *more correct* than the route_only_step patch. Residual EMU-vs-HW
  steady offset (2125 vs 2119) is the deferred component-(1) core-loop sub-noise
  calibration, out of scope here.
- **Detector correctness:** a broad packet signal first fired ~41x/run purely on
  packet-switched TRACE traffic. Narrowed to the control delivery path; now fires
  **0x** on tenant-4 (a kernel whose config is executor-direct, no fabric control
  packets). Detector validated as control-specific and silent where it should be.

### Full bridge result (2026-06-14)

169 kernels, both compilers, EMU + HW:
- **0 bridge (EMU) fail, 0 HW fail, both compilers.**
- **0 trace divergences** (Chess 124 clean / Peano 45 clean) -- the end-of-stream
  timing change disturbed no other kernel's trace vs HW (threshold ~10 cyc).
- 1 XFAIL (Peano `objectfifo_repeat_distribute_repeat`, expected).
- 1 uncompiled: `vec_mul_trace_distribute_lateral` on Chess -- a pre-existing
  `xchesscc` segfault (loop-count-overflow), byte-identical in the prior day's
  run. NOT XFAIL in mlir-aie: its RUN line is `aiecc.py --no-xchesscc`, i.e.
  Peano-only by design. Peano builds + HW-runs it (PASS). The bridge harness
  over-compiles (both compilers regardless of the kernel's declared set); the
  correct fix is to honor `--no-xchesscc` and SKIP Chess (separate harness task).

### Hazard-detector harvest (corpus-wide, from the sweep's info-level EMU logs)

The detector fired ONLY on the `ctrl_packet_reconfig*` family -- the kernels that
route fabric control packets to reconfigure tiles mid-run, so a control packet is
genuinely in flight at an instruction boundary:

| kernel | fires (chess/peano) | EMU | HW |
|---|---|---|---|
| ctrl_packet_reconfig | 36 / 36 | PASS | PASS |
| ctrl_packet_reconfig_1x4_cores | 84 / 84 | PASS | PASS |
| ctrl_packet_reconfig_4x1_cores | 4 / 4 | PASS | PASS |
| ctrl_packet_reconfig_elf | 102 / 93 | PASS | PASS |

Everything else (incl. `add_one_ctrl_packet`, `packet_flow*`, tenant-4, the
executor-direct majority): **0 fires**. Every kernel that fired the detector
PASSED on EMU and silicon with 0 trace divergence. This is the empirical proof:
the detector pinpoints exactly where the flush's precondition is live, and those
kernels are correct WITHOUT it -- the flush was unnecessary even where it would
have acted. The backstop is meaningful (not dead) and will localize any future
control-packet-delivery regression to a red `ctrl_packet_reconfig`.

## Open decisions for Maya

- **Go for deletion** (validate-first, bridge-gated) vs **start with packet-only
  flush** (safer, smaller) and only attempt deletion if the corpus proves clean?
- **`route_only_step`**: delete as dead code, or keep as a tested utility?
- Anything that should block on re-confirming the dispatch-gate calibration
  before we touch the flush at all?
