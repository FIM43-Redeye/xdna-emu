---
name: '#355a Phase C: per-stage attribution -- 99% of the gap is shim DDR cold-start'
description: With memtile DMA SEL trace events working (after fixing two EMU infra bugs -- DMA_Event_Channel_Selection register modeling and broadcast hw_id per-module translation), decomposed the chain from shim dispatch to kernel acquire into 5 stages on add_one_using_dma. Result -- 2537 of 2566 unattributed cycles (98.9%) live in stages 1+2 (shim dispatch -> first chunk in memtile SRAM). HW takes 2699 cyc, EMU 162 cyc. The driver is shim DMA cold-start latency for the first DDR access. Stages 3-5 (memtile internal, memtile->compute, compute pipeline fill) are within 6/12/11 cyc of HW respectively -- already well-calibrated.
type: project
---

# `#355a` Phase C: per-stage attribution

## TL;DR

The 3000-cyc structural pipeline-fill gap from Phase A
(compute LOCK_STALL -> first INSTR_LOCK_ACQUIRE_REQ; HW=3458,
EMU=427) decomposes as:

| Stage                                            | HW (cyc) | EMU (cyc) |  Gap | Ratio |
| ------------------------------------------------ | -------: | --------: | ---: | ----: |
| 1+2: shim dispatch -> memtile S2MM done          | **2699** |   **162** | 2537 | 16.7x |
| 3: memtile S2MM -> memtile MM2S done             |       13 |         7 |    6 |  1.9x |
| 4: memtile MM2S -> compute S2MM done             |       21 |         9 |   12 |  2.3x |
| 5: compute S2MM done -> kernel acquired          |       15 |         4 |   11 |  3.8x |
| **TOTAL: shim dispatch -> kernel acquired**      | **2748** |   **182** | 2566 | 15.1x |

**Stages 1+2 own 2537 of 2566 unattributed cycles (98.9%).**
The other three stages are within 6/12/11 cyc of HW respectively --
already well-calibrated to within a noise floor that approaches the
trace decoder's cycle-delta granularity.

The qualitative driver: HW shim DMA has a one-shot ~2500-cyc DDR
controller setup latency on the first access. EMU's shim DMA model
has effectively zero cold-start cost, just the per-cycle FSM
machinery (`MemoryLatency=5` for memtile, similar for shim).

## Two infra bugs surfaced en route

Phase A and Phase B claimed to use memtile data but were actually
flying blind for memtile traces -- both bugs masked the symptom by
producing 0 memtile events on EMU. This invalidates any prior
finding-doc claim that compared HW vs EMU on memtile-side anchors.

### Bug 1: DMA_Event_Channel_Selection register (0xA06A0) was unmodeled

MemTiles broadcast only 4 DMA event lines (`S2MM_SEL0/SEL1`,
`MM2S_SEL0/SEL1`) regardless of how many physical channels (6+6) are
running. Real HW selects which physical channel feeds each SEL slot
via the DMA_Event_Channel_Selection register at offset 0xA06A0
(per AM020 / aie-rt `xaiemlgbl_params.h`):

| Bits   | Field             | Default |
| ------ | ----------------- | ------- |
| 2:0    | S2MM_Sel0_Channel | 0       |
| 10:8   | S2MM_Sel1_Channel | 0       |
| 18:16  | MM2S_Sel0_Channel | 0       |
| 26:24  | MM2S_Sel1_Channel | 0       |

Pre-fix EMU silently dropped flat channels >= 4 in
`memtile_event_to_hw_id` and never modeled the register, so:

- Multi-channel kernels showed only channel-0 events (which would
  have matched a register-default HW configuration, but invisible
  to test authors expecting per-channel attribution).
- Channels 4-11 (S2MM 4-5, MM2S 0-5 in flat indexing) were silently
  dropped. Wrong if a SEL slot is later programmed to target them.

**Fix** (commit `6a1d8a6`): added `Tile::memtile_dma_event_chan_sel`
state, hooked offset 0xA06A0 in `apply_tile_local_effects`, replaced
`memtile_event_to_hw_id` with `memtile_event_to_hw_ids` (returns
`[Option<u8>; 2]` because both SEL slots can be aimed at the same
channel and fire simultaneously).

### Bug 2: Broadcast hw_id was propagated source-side, not per-receiver

Each event module sees BROADCAST_N at a different hw_id base:

| Module                | BROADCAST_0 hw_id |
| --------------------- | ----------------- |
| Compute core / mem    | 107               |
| Shim PL_A             | 110               |
| MemTile event module  | 142               |

`propagate_broadcasts` was pushing the source tile's hw_id (e.g.,
core BROADCAST_15 = 122) directly to every receiving tile. So when
a memtile trace unit's start_event was correctly programmed to
157 (=MEM_TILE_BROADCAST_15), the propagator delivered hw_id 122
and the trace unit never armed.

Compute traces accidentally worked because core and mem modules
share base 107. Shim trace worked because it's armed via direct
USER_EVENT_1 from npu_write32 (no broadcast involved). MemTile
trace **never armed** in any prior measurement.

**Fix** (commit `28a92c5`): store the broadcast channel number
(0..15) in `pending_broadcasts` instead of hw_id; translate to
each receiving module's hw_id when calling
`notify_*_trace_event`.

## Test setup (the third gotcha)

`add_one_using_dma`'s memtile uses **circular** DMA chains
(objectFifo ping-pong: `next_bd` always set), so
`DMA_*_SEL*_FINISHED_TASK` never fires. Phase B's task list
included these and would have produced empty traces even with the
infra bugs fixed.

Use `FINISHED_BD` instead. It fires per-BD-complete and is the
only meaningful boundary event for circular DMA chains.

Plumbing for this: added `--memtile-sel-channels`,
`XDNA_TRACE_MEMTILE_SEL_CHANNELS`, and `XDNA_TRACE_MEMMOD_EVENTS`
env passthroughs (commits `32e8dc5` and `5b3e916`).

## Measurement methodology

Bridge test invocation:

```bash
XDNA_TRACE_MODE=event_time \
XDNA_TRACE_MEMTILE_EVENTS="DMA_S2MM_SEL0_START_TASK,DMA_S2MM_SEL0_FINISHED_BD,DMA_S2MM_SEL1_START_TASK,DMA_S2MM_SEL1_FINISHED_BD,DMA_MM2S_SEL0_START_TASK,DMA_MM2S_SEL0_FINISHED_BD,DMA_MM2S_SEL1_START_TASK,DMA_MM2S_SEL1_FINISHED_BD" \
XDNA_TRACE_MEMTILE_SEL_CHANNELS="S2MM_SEL0:0,S2MM_SEL1:1,MM2S_SEL0:0,MM2S_SEL1:1" \
XDNA_TRACE_MEMMOD_EVENTS="DMA_S2MM_0_FINISHED_BD,DMA_MM2S_0_FINISHED_BD,DMA_S2MM_0_START_TASK,DMA_MM2S_0_START_TASK,CONFLICT_DM_BANK_0,CONFLICT_DM_BANK_1,EDGE_DETECTION_EVENT_0,EDGE_DETECTION_EVENT_1" \
./scripts/emu-bridge-test.sh --chess-only -v _diag_phase_b_add_one_instrumented
```

First-iteration anchor cycles (relative to absolute trace timer):

| Anchor                                             | HW       | EMU    |
| -------------------------------------------------- | -------: | -----: |
| `kernel_blocks` (compute LOCK_STALL)               | 341035   |  4135  |
| `shim_dispatch` (shim DMA_MM2S_0_START_TASK)       | 341302   |  4272  |
| `shim_done` (shim DMA_MM2S_0_FINISHED_TASK)        | 342984   |  5211  |
| `memtile_s2mm_done` (DMA_S2MM_SEL0_FINISHED_BD)    | 344001   |  4434  |
| `memtile_mm2s_done` (DMA_MM2S_SEL0_FINISHED_BD)    | 344014   |  4441  |
| `compute_s2mm_done` (memmod DMA_S2MM_0_FINISHED_BD)| 344035   |  4450  |
| `kernel_acquired` (INSTR_LOCK_ACQUIRE_REQ)         | 344050   |  4454  |

`shim_done` (FINISHED_TASK) is informative but **not a clean stage
boundary** because the shim DMA streams data continuously -- the
first BD's payload reaches the memtile before the shim task
finishes streaming the rest. Hence the negative `1b` subset on
EMU below.

| Sub-stage breakdown                                | HW   | EMU  | Note                              |
| -------------------------------------------------- | ---: | ---: | --------------------------------- |
| 1a: shim_dispatch -> shim FINISHED_TASK            | 1682 |  939 | "shim done streaming"             |
| 1b: shim FINISHED_TASK -> memtile S2MM FINISHED_BD | 1017 | -777 | EMU first chunk arrives BEFORE shim done |

EMU's negative delta is the smoking-gun signature of zero cold-start
latency. On HW the shim's DDR-controller setup dominates the first
chunk's latency, so the chunk lands at the memtile **after** the
shim task itself has finished streaming. On EMU the shim has no
cold-start; the chunk lands well before the shim reports task done.

## Stage analysis

### Stage 1+2 (shim dispatch -> first chunk in memtile SRAM)

HW=2699, EMU=162, gap=2537 cyc.

This is the entire DDR-to-array path: shim DMA reads from DDR,
sends through stream switch, memtile S2MM writes to SRAM.

Under-modeled component: **shim DMA cold-start**. HW DDR controllers
have a 2000-3000 cyc setup latency on a fresh access (precharge,
activate, CAS), then sustained throughput. EMU's shim model has
effectively no cold-start -- only the per-cycle FSM machinery.

**Calibration target**: introduce a one-shot "DDR access cold-start"
cost in the shim DMA model, fired once when transitioning Idle ->
AcquiringLock (or Idle -> MemoryLatency for shim where there's no
lock), **not** per-BD. ~2500 cyc.

This is structural -- a new modeling primitive, not a knob bump.
Per-BD chains in steady state should remain at ~5 cyc (the existing
`MemoryLatency` value), only the first BD pays the setup cost.

### Stage 3 (memtile S2MM done -> memtile MM2S done)

HW=13, EMU=7, gap=6 cyc.

Tight enough to be in the EMU's noise floor. Likely contributors:

- `lock_acquire_cycles=1` may be 1 short of HW arbiter round-trip.
- `bd_chain_cycles` / chain-rearm overhead may be 1-2 cyc short.
- Memtile SRAM read latency (MM2S sees data after S2MM commit) may
  need 1-2 extra `MemoryLatency` cyc.

**Priority**: low. 6 cyc absolute is below where calibration buys
much. Defer until a workload demands sub-30-cyc fidelity here.

### Stage 4 (memtile MM2S -> compute S2MM done)

HW=21, EMU=9, gap=12 cyc.

This is the meaningful one of the small-stage trio. The path
involves stream switch hops (memtile master out -> memtile north
slot -> compute south input -> compute slave -> compute S2MM DMA
into LM).

Known under-modeling: `channel_start_cycles=2` is **defined in
DmaTimingConfig but not consumed** anywhere in the FSM stepper
(found in pre-Phase-A audit, not yet wired). Plumbing this through
would account for some of the 12 cyc.

Other candidates: `LocalRoute.latency=3` per hop may need bumping
to 4-5 if the path actually has 2 hops being collapsed; master-port
FIFO drain time isn't explicitly modeled (we recently corrected the
depths from 8/4 to 4/2 per AM020, but drain is still implicit).

**Priority**: medium. Worth fixing alongside stage 1+2 since both
involve shim/DMA FSM timing config. The `channel_start_cycles`
plumbing is a specific actionable lead.

### Stage 5 (compute S2MM done -> kernel acquired)

HW=15, EMU=4, gap=11 cyc.

Decomposable using existing events on the compute mem trace:
`LOCK_<N>_REL` (id 46+N\*4) -> `LOCK_<N>_ACQ_GE` (id 45+N\*4) ->
core `LOCK_STALL` falling edge -> kernel resumes.

Splits into "lock arbiter resolution" + "kernel pipeline restart
from stall." HW likely has 5-10 cyc of pipeline restart cost
(instruction fetch refill, register restore from stall context),
which EMU models as zero.

**Priority**: low. 11 cyc absolute, similar to stage 3.

## Closure plan for `#355a`

In rough priority:

1. **Stage 1+2 fix**: introduce `shim_ddr_cold_start_cycles` (or
   similar) in `DmaTimingConfig`, fire once per task transition
   from Idle on shim engines. Default ~2500 cyc per AM020 DDR
   timing tables (precise value to be measured by varying access
   patterns; this is one calibration knob to tune against HW).
2. **Stage 4 channel_start_cycles plumb-through**: thread the
   already-defined config field into the actual FSM transitions.
   Should account for ~2-4 of the 12-cyc gap.
3. **Re-measure**: same harness, same kernel. Confirm stages 1+2
   close to within stage-3-noise floor; confirm stage 4 closes to
   within ~5 cyc.
4. **Optional micro-tuning**: if a future workload demands sub-30
   cyc fidelity on the memtile->kernel path, revisit stages 3 and 5.

## See also

- `2026-05-10-phase-a-trace-cycles-measurement.md` -- original
  3000-cyc gap measurement; this finding decomposes its single
  number into per-stage components.
- `2026-05-10-phase-b-runtime-seq-instrumentation.md` -- coarser
  T0..T3 anchor decomposition. Note: phase B's claim about EMU
  memtile data was based on traces produced before the broadcast
  hw_id fix; some of its numbers were derived from EMU compute
  events alone with no memtile cross-check.
- `2026-05-10-dma-waterfall-pipeline-fill-decomp.md` -- EMU-side
  FSM phase residency; complementary to this HW-vs-EMU comparison.
- Commits: `6a1d8a6` (memtile SEL register modeling),
  `32e8dc5` (inject script + bridge-test plumbing),
  `28a92c5` (broadcast hw_id translation),
  `5b3e916` (XDNA_TRACE_MEMMOD_EVENTS env passthrough).
- Test variant: `mlir-aie/test/npu-xrt/_diag_phase_b_add_one_instrumented/`.
