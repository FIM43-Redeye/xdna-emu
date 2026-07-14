# MEMORY_STALL is per-physical-bank round-robin arbitration loss; the emulator's model has the mechanism inverted

Date: 2026-07-13
Branch: `feat/core-memory-stall-model`
Kernel: `mlir-aie/test/npu-xrt/spike_bringup/of_q0_rich.py` (Q=0 vertical objectfifo spine)
Derivation: Codex/Sol dig of AM020/AM025/aie-rt/Peano; every hardware claim
verified against the cited source by Claude.

## The gap

Real NPU1 (Phoenix, AIE2) fires ~220 `MEMORY_STALL` (core event 23) per
*consumer* core on a memory-heavy scalar loop and only ~2 on the *producer*.
The emulator was believed to fire 0. This finding root-causes the mechanism and
shows the emulator's existing model is not merely under-calibrated -- it encodes
the **wrong physical mechanism**.

HW counts (stable across 20 runs, `build/experiments/sp3-spike-trace/`):

| Core | Compute body | HW MEMORY_STALL |
|------|--------------|-----------------|
| Producer (row 2) | `eo[i] = index_cast(i)` (store only) | 2 |
| ConsA (row 3) | `eo[i] = ei[i] + 1` (load + store) | 220 |
| ConsB (row 3, X-col) | `eo[i] = ei[i] + 2` | 244 |

## The mechanism (derived, verbatim from AM020)

`MEMORY_STALL` is the core-side consequence of a **denied data-memory
request** under **per-physical-bank round-robin arbitration**. AM020 ch.2
("AIE-ML Memory Module", `docs/xdna/am020-aie-ml/chapter-2-aie-ml-tile-architecture.md:166`),
verbatim:

> Each memory bank has its own arbitrator to arbitrate between all requesters.
> The memory bank arbitration is round-robin to avoid starving any requester. It
> handles a new request every clock cycle. When there are multiple requests in
> the same cycle to the same memory bank, only one request per cycle is allowed
> to access the memory. The other requesters are stalled for one cycle and the
> hardware retries the memory request in the next cycle.

So each arbitration loss costs exactly **one cycle**, the request **retries**
next cycle, and a core loss stalls the whole datapath (AM020 ch.4:69 -- two load
ports, one store port; a bank conflict on any port stalls the datapath). The
event mapping (`aie-rt/driver/src/events/xaie_events_aieml.h:57`,
`XAIEML_EVENTS_CORE_MEMORY_STALL 23`; AM025 `Core_Status.Memory_Stall_E/N/W/S`
= a directional memory access "is stalling the AI Engine"; local DM is the east
`0x70000` quadrant) confirms event 23 is the core-lost-memory-service signal.
Event 23 says the core was denied; it does not say which bank or requester --
the memory-module `CONFLICT_DM_BANK_n` events (77-84) and the DMA
backpressure/starvation events (39-42) are the complementary signals.

### Two refinements to the original hypothesis

1. **Load and store do NOT self-conflict.** Current Peano
   (`llvm-aie/.../AIEHazardRecognizer.cpp:746`, commit `a03628cba8cc`) states a
   same-bank load and store use separate hardware ports and never conflict. So
   the consumer's bundled load+store is *not* an intra-core explanation; the
   contender is **external** -- in this topology, the tile S2MM/MM2S DMA. This
   matches the load/store asymmetry: the producer only stores (few losses); the
   consumers load from banks the DMA is also touching (220 losses).
2. **220 is not a per-access probability.** The consumer steady-state loop (PC
   `0x210`) is 24 iterations; the trace shape is `ConsA = 26 + 8x24 + 1 + 1 =
   220`, one event every 2 cycles -- exactly "each iteration's bundle is denied
   once and succeeds on retry." The trace starts mid-execution, so `220/512`
   is meaningless. ConsB's extra 24-burst accounts for 244 vs 220.

## Bank structure and the emulator bug

AM020 ch.2:164: data memory is 64 KB as **eight** 8 KB physical banks
(512 word x 128-bit, single-port); "every two banks are interleaved to form one
bank, a total of **four** banks of 16 KB. Bank[0] starts at address 0." The
compiler agrees: `AIETargetModel.h:595` `getNumBanks` returns **4** for compute
tiles, and `AIEAssignBuffers.cpp` divides the space into four *contiguous*
16 KB ranges.

The emulator's bank selector is **inconsistent with this**
(`src/interpreter/timing/memory.rs:305`):

```rust
pub fn bank(&self) -> u8 { ((self.address >> 4) & 0x7) as u8 }
```

`(addr >> 4) & 7` is an 8-way, 16-byte-granularity interleave across the entire
address space. It scatters this kernel's `0x400..0x5ff` buffers across all 8
physical banks instead of confining them to the low logical bank. This is
directly why the emulator sees the wrong contention (below).

The correct layout is four contiguous 16 KB logical banks, each an interleaved
pair of physical banks. The exact sub-bank interleave granularity (Sol's
inference: `physical = 2*((a>>14)&3) + ((a>>4)&1)`, i.e. 16-byte word interleave
within the pair) is **not explicitly documented** and is one of the facts the
HW capture below pins.

## Emulator evidence: the model is inverted (in-process census)

Running the traced kernel in-process on current code
(`of_q0_rich_bank_overlap_census`, env `XDNA_EMU_STALL_DEBUG=1`) over the full
compute (17473 cycles), aggregated over enabled compute tiles:

```
core_nz=2660  dma_nz=2798  both_nz=684  conflict=32
```

All **32** detected conflicts are on the **producer** (row 2), walking physical
banks 0->7 during its startup fill; **zero** on the consumers. So vs HW:

| Tile | EMU conflicts | HW MEMORY_STALL |
|------|---------------|-----------------|
| Producer | 32 | 2 |
| Consumer | 0 | ~220 |

Inverted. The emulator over-charges the producer's store-vs-drain sweep (which
silicon does not stall on) and misses the consumers' load-vs-DMA losses
entirely -- the consumers have 652 same-cycle core+DMA overlaps that never land
on the same bank because the `(addr>>4)&7` selector spread them across 8 banks.

Two more emulator faults, both confirmed in code:
- The stall does **not** feed cycle cost. Coordinator Phase 4
  (`src/interpreter/engine/coordinator.rs`, the `core_banks & dma_banks` block)
  only bumps a `memory_stalls` stat and emits a held-level trace edge; there is
  no `record_stall` / `ctx.cycles` effect, so a detected conflict is free.
- The conflict check is symmetric over loads and stores, contradicting the
  Peano load/store-port compatibility.

## The derived cost rule (deterministic, no fitted constant)

Per cycle, per physical bank: enumerate the incompatible requesters targeting it
(core's required load/store, DMA S2MM/MM2S, neighbors), apply persistent
round-robin state, grant one, and hold+retry every loser. If any request the
current core bundle requires loses, freeze the core datapath for one cycle and
assert `CORE_MEMORY_STALL` once; raise `CONFLICT_DM_BANK_n` for each contended
bank and the appropriate DMA memory-pressure event. Do not manufacture a
load-vs-store self-conflict. The core cost is the count of cycles in which a
required core memory request is denied -- recursive (a denial repeats the bundle
next cycle), deterministic, stateful, with no probability anywhere.

## What is undocumented -> the one HW measurement

The static sources give the bank structure, round-robin policy, retry, 1-cycle
cost, and event identities. They do **not** give: the initial round-robin owner,
the requester enumeration / pointer-advance rule, the cycle each DMA channel
presents its bank request, or the exact core-port/DMA-arbiter merge. These shape
the count and must be measured, not fitted.

**Ranked next step (in progress):** one cycle-correlated HW capture of the
unchanged frozen kernel adding compute-memory trace slots on both consumer tiles
for `CONFLICT_DM_BANK_0`/`_1` (77/78), `DMA_S2MM_0_MEMORY_BACKPRESSURE` (39),
`DMA_MM2S_0_MEMORY_STARVATION` (41), retaining core `MEMORY_STALL` and a PC
marker covering `0x210`. This pins: event-23-coincident-with-bank-0/1-conflict
(proves the bank mechanism and the physical-bank mapping), which side loses on
intervening cycles, and the round-robin phase -- or refutes the hypothesis and
redirects to another DM/TM processor-bus stall.

## HW confirmation (2026-07-13 capture) -- mechanism proven, not just plausible

The ranked HW capture was run: one shot of the frozen kernel on real NPU1,
compute-tile core trace (MEMORY_STALL) + memmod trace (CONFLICT_DM_BANK_0-3,
S2MM_0_MEMORY_BACKPRESSURE, MM2S_0_MEMORY_STARVATION), event_time. Capture
preserved at `build/experiments/memory-stall-bankcap/`. Per-tile event counts
(tiles HW-shifted col0->col1):

| Tile | MEMORY_STALL | CONFLICT_BANK_0 | CONFLICT_BANK_1 | CONFLICT_BANK_2-7 | S2MM_0_MEM_BACKPRESSURE | MM2S_0_MEM_STARVATION |
|------|---:|---:|---:|---:|---:|---:|
| Producer (1,2) | 1 | 2 | 1 | 0 | ~0 | 0 |
| ConsA (1,3) | 220 | 115 | 109 | 0 | 344 | 0 |
| ConsB (2,3) | 245 | 128 | 121 | 0 | 396 | 0 |

Every derived claim is confirmed:

1. **Bank mapping.** Conflicts fire on physical banks **0 and 1 only**; banks
   2-7 are silent. The kernel's `0x400..0x5ff` buffers are confined to the low
   logical bank exactly as derived -- the emulator's `(addr>>4)&7` 8-way spread
   is definitively wrong.
2. **Mechanism = the core losing arbitration.** `MEMORY_STALL` (220 / 245)
   almost equals `CONFLICT_DM_BANK_0 + _1` (224 / 249). It is slightly *less*
   because a bank conflict raises `CONFLICT_DM_BANK_n` regardless of who loses,
   but `MEMORY_STALL` only when the *core* loses; round-robin grants the core ~98%
   of the time here but occasionally grants the DMA instead (the ~4-event gap).
   This is the round-robin mechanism, dead on.
3. **The contender is the S2MM fill DMA.** `S2MM_0_MEMORY_BACKPRESSURE` fires
   heavily on both consumers (344 / 396) and `MM2S_0_MEMORY_STARVATION` not at
   all -- the incoming fill is the requester being denied bank 0/1 on the
   off-cycles, not the outgoing drain.
4. **Producer asymmetry.** 1 stall / 3 conflicts, matching HW's ~2: a store-only
   stream with one MM2S drain rarely loses arbitration.

No parameter was fitted; the capture measured the previously-undocumented
dynamics (physical-bank occupancy, which side loses) and every number fell in
line with the AM020-derived mechanism.

## Cycle-level causality (2026-07-14) -- the mechanism is EXACT, not approximate

The counts above prove correlation. A cycle-level re-analysis of the same
capture proves **causation**, and corrects three numbers.

**Every MEMORY_STALL cycle is preceded by a bank conflict in the immediately
previous cycle. Zero exceptions.**

| Tile | MEMORY_STALL cycles with NO conflict at t-1 | contended cycles followed by a stall | core-won cycles |
|------|---:|---:|---:|
| ConsA (1,3) | **0 / 220** | 220 / 232 (94.8%) | 12 |
| ConsB (2,3) | **0 / 245** | 245 / 255 (96.1%) | 10 |
| Producer (1,2) | 0 / 1 | 1 / 3 | 2 |

A best-fit offset scan over +/-12 cycles peaks sharply at **-1** (220/220 and
245/245 hits; offset 0 scores only 6 and 5). The -1 is a clean causal ordering:
the conflict is observed in the memory module at cycle t, the core's stall is
asserted at t+1. (The mem-module trace anchors sit exactly +2 cycles after the
core anchors on every tile, so 1 cycle of "pipeline latency" vs "trace-unit
anchor skew" is not separable from this capture -- it does not affect the
conclusion.)

Also: `CONFLICT_BANK_0 ∩ CONFLICT_BANK_1 = ∅` -- the two banks are never
contended in the same cycle in this workload.

### Three corrections

1. **A decoder bug invalidated the previous timestamp reading.**
   `tools/trace_decoder`'s `soc` field is `timer - cmd_index` (`decode.py:220`),
   which subtracts an implicit `+1` that `decode.py:217` adds per command. That
   `+1` is **physical**; `soc` is a decoder artifact. Proof: mode-0 has a
   dedicated `Multiple` opcode for co-occurring events (and this capture uses
   it), so two separate frames can never share a cycle -- yet `soc` places 7
   CONFLICT frames and 12 BACKPRESSURE frames on the identical cycle 555681.
   Use `ts`, not `soc`. On the true timebase, ConsA's MEMORY_STALL fires **every
   other cycle** (555685, 687, 689, 691...): 216 singletons, dominant gap 2 --
   exactly the "denied once, succeeds on retry" shape predicted above.

2. **Decoded RECORD counts are an encoding artifact. Never compare them.** A
   held signal encodes as `Event(cycles=0) + Repeat(N)`, and the record emitter
   silently drops the expansion (`decode.py:242-249`). S2MM_0_BACKPRESSURE shows
   **344 records for 756 cycles-high** -- a 2.2x undercount. Compare **interval
   area** (total cycles asserted), which `rebuild_perfetto_mode0` already
   computes and which integrates identically across both encodings.

3. **Corrected HW ground truth (interval area, i.e. cycles asserted):**

| Tile | MEMORY_STALL | BANK_0 | BANK_1 | BANKS 2-3 | S2MM_0_BP | MM2S_0_STARV |
|------|---:|---:|---:|---:|---:|---:|
| Producer (1,2) | 1 | 2 | 1 | 0 | 0 | 0 |
| ConsA (1,3) | 220 | **121** | **110** | 0 | **756** | 0 |
| ConsB (2,3) | 245 | **132** | **122** | 0 | **853** | 0 |

   True contended cycles on ConsA are **232** (not the 224 that the record
   counts suggested), so the core loses **94.8%** of contentions, not ~98%.

### Two traps for any validation against this capture

- **HW's MEMORY_STALL is a 1-cycle burst at period 2, not a contiguous stall.**
  A model that charges a multi-cycle contiguous stall could total ~220 cycles
  with a completely wrong *shape*. Compare the run-length and gap distribution
  (HW: 216 singletons, dominant gap 2), not just the total.
- **Banks 4-7 were never traced.** The mem slot table covers only BANK_0..3
  (`events.json:slot_names.mem`). "Banks 2-7 = 0" is *observed* zero for banks
  2-3 and **unmeasured** for 4-7. Do not let a model "match" an unmeasured zero.

## Emulator vs HW (2026-07-14, Task 8) -- mechanism reproduced, magnitude ~2x high

The derived arbitration model (Tasks 1-7: per-physical-bank round-robin,
REQUEST -> ARBITRATE -> COMMIT, `CoreStatus::WaitBank` costing a real cycle) was
run against the traced kernel in-process
(`of_q0_rich_bank_arbitration_vs_hw` in `src/testing/xclbin_suite.rs`, per-cycle
recorder `src/interpreter/engine/bank_census.rs`). No constant was tuned. Cycles
compared, not decoded record counts; emulator cols are HW cols minus 1.

| Tile | MEMORY_STALL (EMU / HW) | BANK_0 | BANK_1 | BANKS 2-3 | BANKS 4-7 | S2MM_0_BP | MM2S_0_STARV |
|------|---:|---:|---:|---:|---:|---:|---:|
| Producer (0,2) | **102** / 1 | 145 / 2 | 85 / 1 | 0 / 0 | 0 / (untraced) | 0 / 0 | 128 / 0 |
| ConsA (0,3) | **425** / 220 | 268 / 121 | 230 / 110 | 0 / 0 | 0 / (untraced) | 56 / 756 | 40 / 0 |
| ConsB (1,3) | **440** / 245 | 280 / 132 | 250 / 122 | 0 / 0 | 0 / (untraced) | 76 / 853 | 53 / 0 |

Shape (the trap the brief warns about):

| Tile | EMU stall runs | singletons | dominant gap | HW |
|------|---:|---:|---:|---|
| ConsA | 400 | 383 (96%) | 2 (369x) | 216 singletons, gap 2 |
| ConsB | 399 | 375 (94%) | 2 (369x) | same |
| Producer | 41 | 9 | 4 | 1 stall total |

Also: EMU contended cycles ConsA 498 / ConsB 530, of which the core loses 85.3% /
83.0% (HW: 232 / 255 contended, core loses 94.8% / 96.1%). `CONFLICT_BANK_0` and
`CONFLICT_BANK_1` are **never** contended in the same cycle in the emulator
either -- matching HW's disjointness exactly.

### What reproduces

1. **The mechanism, not just the number.** The stall is a 1-cycle deny-and-retry
   at period 2 -- 96% singletons, dominant gap 2 -- the identical shape HW shows.
   A model that charged one contiguous multi-cycle stall would have been a
   failure at this line; this one is not.
2. **The right tiles.** Consumers stall, the producer mostly does not. The
   pre-arc model was exactly INVERTED (32 conflicts on the producer, 0 on the
   consumers).
3. **The right banks.** Conflicts land on physical banks 0 and 1 only; 2-3 are
   silent (HW-observed) and 4-7 are silent too (HW never traced those, so this is
   consistency, not validation). The `(addr>>4)&7` 8-way spread is gone.
4. **The cost is real.** Every stall cycle here is a cycle in which the core's
   bundle did not retire.

### Where it diverges, and why (honest, unfitted)

**The model over-produces stall cycles by ~1.9x on the consumers and ~100x on the
producer, and it produces MM2S starvation where HW has none.** Single root cause,
and it is the DMA side, not the arbiter:

> The emulator's DMA presents a bank request on **every cycle** of a transfer
> (`words_per_cycle_for` -> 1 stream word = 4 bytes per cycle for a streaming
> channel, `src/device/dma/engine/stepping.rs:1357`). Silicon's DMA aggregates
> stream beats into **128-bit (16-byte) bank accesses** (AM020 ch.2:164, bank
> width 128 bits), so it only touches a bank once per four stream beats.

A ~4x denser DMA bank demand is a ~4x denser collision opportunity against the
same core traffic, which is exactly what the numbers show, moderated on the
consumers by the fact that a stalled core also stops requesting:

- Consumers: HW 232 contended cycles -> EMU 498 (2.1x). Stalls 220 -> 425 (1.9x).
- Producer: HW 3 contended cycles -> EMU 230. The producer's core stores are
  sparse, so in HW its MM2S drain almost never collides with a store; in EMU the
  MM2S demands the bank every cycle of the drain and collides constantly. This is
  the same defect, amplified because the producer's true collision rate is near
  zero -- a dense-demand model has nothing to hide behind.
- `MM2S_0_MEMORY_STARVATION`: HW 0 everywhere, EMU 40 / 53 / 128. Same cause: the
  emulator's MM2S loses arbitrations it should never have been in.

This is precisely the **idealized-fast-memory / DMA-burst-cadence gap** the design
spec called out and deliberately did not model (DMA cadence varies with installed
silicon). It is recorded here, not fitted away. Closing it means modelling the
DMA's bank access as a 16-byte burst once per 4 stream beats -- a DMA-timing
change, not an arbiter change, and one that would move DMA timing everywhere else
in the emulator, so it is a separate decision.

**A second, independent gap (semantic, not timing):** HW's
`S2MM_0_MEMORY_BACKPRESSURE` asserts for **756 / 853** cycles -- far more than the
232 / 255 cycles in which any bank was contended at all. The event therefore
cannot mean only "this channel lost a bank arbitration this cycle", which is what
the emulator emits it for (56 / 76 cycles). On silicon it evidently asserts
whenever the S2MM write path cannot drain into memory, of which arbitration loss
is only one cause. The emulator's emission is a strict subset of HW's, so the
event's *semantics* are under-modelled independently of the cadence gap above.

### Verdict

The AM020-derived round-robin arbiter reproduces the **mechanism** of
`MEMORY_STALL` -- right tiles, right banks, right per-cycle shape, real cycle
cost, and it fixed a model that previously had the phenomenon exactly backwards.
Its **magnitude** is ~2x high on the consumers and badly high on the producer,
traceable to one specific, documented, unfitted cause on the DMA side.

## Provenance

Brief: `docs/superpowers/BRIEF-memory-stall-bank-arbitration.md`. Prior context
that side-stepped rather than root-caused this gap:
`build/experiments/sp3-spike-trace/SP4A-LEAN-FINDING.md` (removing compute drops
ConsA 220 -> 3). Gap registry row: `docs/fidelity-gaps/core-compute-timing.md`.
