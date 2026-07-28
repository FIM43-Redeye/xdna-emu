# Brief: derive the AIE2 core-vs-DMA memory-bank arbitration rule that produces MEMORY_STALL (event 23)

## The point (read this first)

The xdna-emu emulator does not model a real hardware cost: on a memory-heavy
compute loop, real NPU1 (Phoenix, AIE2/aieml) fires **~220-244 MEMORY_STALL
events per consumer core**; the emulator fires **0**. This is a genuine
cycle-cost gap (each missing stall is ~1 lost HW cycle; the deficit
back-pressures upstream). We are opening a campaign to close it, **derive-first**
(no fitting).

Your job is the **hardware-fact derivation**: find, from the open-source
toolchain, the mechanism by which the AIE2 compute-tile memory port stalls, and
propose a *derived* cost rule. This is a bounded "what does the silicon actually
do, and where is it written down" question with a live hardware oracle to check
against -- not an open-ended design task.

## The smoking-gun evidence (already captured, do not re-run)

Reproduction kernel `mlir-aie/test/npu-xrt/spike_bringup/of_q0_rich.py`: a Q=0
vertical objectfifo spine. Scalar int32 loops, 32 elements x 16 reps. Verified
HW MEMORY_STALL counts (`build/experiments/sp3-spike-trace/spike.events.json`
and `task3/run_*/events.json`, stable across 20 runs):

| Core | Compute body | DMA touching its local mem during compute | MEMORY_STALL (HW) |
|------|--------------|-------------------------------------------|-------------------|
| Producer (1,2) | `eo[i] = index_cast(i)` (store only, no input read) | output drain only (~1 stream) | **2** |
| ConsA (1,3) | `eo[i] = ei[i] + 1` (load `ei` + store `eo`) | inbound relay fill of `ei` + output drain of `eo` (~2 streams) | **220** |
| ConsB (2,3) | `eo[i] = ei[i] + 2` (cross-column) | same, cross-column relay | **244** |

The producer and consumers run near-identical scalar loops over the same 32x16
elements. The producer fires ~2; the consumers fire ~220+. The structural
difference: **the consumer's loop reads a local-memory buffer (`ei`) that a
relay DMA is concurrently filling** (and writes `eo` while a DMA drains it); the
producer only stores, and its input isn't DMA-fed.

## The leading hypothesis (confirm, refute, or refine it -- do not assume it)

**MEMORY_STALL = arbitration contention between the compute core's load/store
port and the tile DMA engine (and/or neighbor cores) for access to the same
local-memory bank.** When the core's access lands on a bank the DMA is using
that cycle, the core port stalls one cycle and fires event 23. The producer sees
almost none because its compute isn't racing an inbound DMA on the same banks.

This is (probably) *documented hardware structure* -- AIE2 tile data memory is
multi-bank and the core / DMA / cascade / neighbors arbitrate for bank access
with a defined priority. If so, the stall cost is **derivable**, not fitted.

## The questions to answer (with source anchors)

1. **Bank structure.** How many banks does an AIE2 compute-tile data memory
   have, and what is the bank size / word interleave? (aie-rt register defs,
   AM025 register DB, or the mlir-aie device model.) Which addresses map to
   which bank for this kernel's `ei`/`eo` objectfifo buffers?

2. **The arbitration mechanism.** When the core and the DMA (or a neighbor
   core, or cascade) request the *same bank* the *same cycle*, what does the
   hardware do? Is there a documented arbitration priority and a 1-cycle
   (or N-cycle) stall charged to the loser? Is that stall exactly the
   `CORE_MEMORY_STALL` event (id 23,
   `aie-rt/driver/src/events/xaie_events_aieml.h:58`)? Quote the register /
   doc text (e.g. `Event_Group_Core_Stall_Enable` bit `Memory_Stall`,
   `Memory_Stall_Halt` "from DM or TM (processor-bus)", the per-direction
   `Memory_Stall_E/N/W/S` registers in `aie_registers_aie2.json`).

3. **The derived cost rule.** From the bank structure + arbitration, derive a
   rule that predicts stall *count* for a given (core access pattern,
   concurrent DMA access pattern, bank mapping). It must plausibly yield ~220
   for a consumer doing 512 load+store element-ops against a concurrent
   fill+drain, and ~2 for a store-only producer with no inbound DMA -- **as a
   consequence of the derived rule, NOT by fitting a constant to 220.** If the
   rule needs a hardware-measured parameter the toolchain does not expose (e.g.
   the exact DMA-vs-core bank duty cycle), say so precisely and name the one HW
   measurement that would pin it.

4. **Where the toolchain is silent.** If aie-rt / AM025 / mlir-aie do NOT
   document the arbitration cost (only that the event exists), state that
   plainly. Do not invent a mechanism. Distinguish "documented and derivable"
   from "event exists but cost is undocumented -> needs HW characterization."

## Deliverables

A written finding (present it as your final message; also acceptable to write it
to `docs/superpowers/findings/2026-07-13-memory-stall-bank-arbitration.md`) with:

1. The AIE2 compute-tile bank structure, with source anchors.
2. The arbitration mechanism and whether it maps to event 23, quoted from source.
3. The proposed derived cost rule (or an explicit "undocumented, needs HW"
   verdict), and how it explains the 2 / 220 / 244 asymmetry.
4. A ranked single next step (derive-only, or the one HW measurement needed).

## Ground rules (important)

- **Derive, do NOT calibrate.** The rule comes from the bank structure +
  arbitration semantics, not from fitting to 220. "It reproduces 220" is a
  consequence to check, never the derivation criterion. A hardcoded per-access
  probability tuned to hit 220 is a failure, not a solution.
- **Sources, in priority order:** (1) aie-rt `/home/triple/npu-work/aie-rt/driver/src/`
  (branch `xlnx_rel_v2025.2` -- the official Xilinx clone, NOT mlir-aie's
  vendored fork); (2) AM025 register DB
  `/home/triple/npu-work/mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json`;
  (3) mlir-aie device model; (4) AM020/AM025 prose. aietools
  (`/home/triple/npu-work/amd-unified-software/aietools`) is a **read-only
  reading reference** -- understand the hardware fact, never copy code/data.
- **Read-only.** Do not modify emulator source. This is a derivation/finding
  task; the model implementation is Claude's follow-up.
- **Do NOT commit.** Present findings for review.

## Execution discipline (prior Codex jobs hung here -- READ)

- Do **NOT** dispatch an independent-reviewer subagent and do **NOT** enter any
  `wait` / collaboration poll loop. A prior job soft-hung 40+ minutes doing
  exactly that. Verify inline, write the finding, TERMINATE. Your final message
  is the deliverable.
- If any command is long / CPU-bound, background-and-block in ONE shell command
  (`cmd & wait $!`); never poll in a check->sleep loop.
- Reconstruct context from this brief + the anchors below; do not go spelunking
  the whole repo.

## Anchors

- Event def: `/home/triple/npu-work/aie-rt/driver/src/events/xaie_events_aieml.h:58`
  (`XAIEML_EVENTS_CORE_MEMORY_STALL 23U`).
- Register DB: `/home/triple/npu-work/mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json`
  (search `Memory_Stall`, `Event_Group_Core_Stall`, bank/DM registers).
- Kernel: `/home/triple/npu-work/mlir-aie/test/npu-xrt/spike_bringup/of_q0_rich.py`
  (compute bodies at lines 55-57 producer, 65-66 ConsA, 75-76 ConsB).
- Existing HW captures: `build/experiments/sp3-spike-trace/spike.events.json`,
  `task3/run_*/events.json`.
- Emulator's current (partial) model: `src/interpreter/execute/cycle_accurate.rs:206-245`
  (`record_memory_access`, intra-bundle same-bank conflict only; scalar-only at
  `:230`); event type `src/interpreter/state/event_trace.rs:58-60`.
- Gap doc: `docs/fidelity-gaps/core-compute-timing.md`.
