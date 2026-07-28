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

1. ~~**Load and store do NOT self-conflict.**~~ **REFUTED by the HW capture
   (2026-07-14). Load and store DO self-conflict; the arbiter must model it.**

   The original claim leaned on Peano
   (`llvm-aie/.../AIEHazardRecognizer.cpp:746`, commit `a03628cba8cc`): "load-vs-
   store uses separate HW ports", therefore the consumer's bundled load+store is
   not an intra-core explanation and the contender must be the tile DMA. Silicon
   says otherwise:

   - **Arithmetic.** HW denies the core on **all 24** steady-state bundles of
     every rep. The only other requesters are the tile DMAs, and per rep the S2MM
     writes 128 B while the MM2S reads 128 B. At the **measured** 16 B/access
     granule (`2026-07-14-dma-bank-access-width.md`) that is 8 + 8 = **16 bank
     accesses per rep** -- arithmetically incapable of causing 24 denials even if
     every single access collided. The DMA cannot be the contender.
   - **The load/store asymmetry is the self-conflict's fingerprint, not the
     DMA's.** The producer's loop is **store-only** -- no load, hence no possible
     self-conflict -- and it stalls ~0 on BOTH sides (HW 1, EMU 4), despite
     running a DMA drain the whole time. The load+store consumers stall on every
     bundle on BOTH sides. If the DMA were the contender, the producer's drain
     would collide too.

   Peano's comment describes what its **scheduler** bothers to track -- a bank
   conflict costs one cycle, a performance matter, so it models only the
   load-vs-load case it can actually schedule around. It is not a claim about
   what the DM bank arbiter does, and the emulator must not follow it.
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
bank and the appropriate DMA memory-pressure event. (An earlier version of this
rule said "do not manufacture a load-vs-store self-conflict"; that was wrong --
see the REFUTED refinement 1 above. A bundle's load and store DO contend when
they land in the same physical bank, and on this kernel that is what drives every
consumer stall.) The core cost is the count of cycles in which a
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

### Traps for any validation against this capture

- **HW's MEMORY_STALL is a 1-cycle burst at period 2, not a contiguous stall.**
  A model that charges a multi-cycle contiguous stall could total ~220 cycles
  with a completely wrong *shape*. Compare the run-length and gap distribution
  (HW: 216 singletons, dominant gap 2), not just the total.
- **Banks 4-7 were never traced.** The mem slot table covers only BANK_0..3
  (`events.json:slot_names.mem`). "Banks 2-7 = 0" is *observed* zero for banks
  2-3 and **unmeasured** for 4-7. Do not let a model "match" an unmeasured zero.
- **The capture holds ~9 of the kernel's 16 pipeline iterations. NEVER compare
  run totals against it -- compare per-iteration shape.** `of_q0_rich` is Q=0, so
  the cores need no DDR input and start executing the moment the CDO enables
  them, long before the host instruction stream arms the trace unit. Several reps
  therefore run before the window opens. Every run total in this capture is a
  fraction of the kernel's true total, and the fraction differs per tile (ConsA
  caught 9 reps, ConsB 10). Comparing totals against a full 16-rep emulator run
  manufactures a ratio of `16/9 = 1.78` out of nothing -- which is exactly what
  happened, and is the whole content of the retracted "~1.9x over-production"
  below.

  **The cheap coverage check:** count `INSTR_LOCK_ACQUIRE_REQ` and divide by
  acquires-per-rep. The consumers acquire twice per rep, so a full 16-rep capture
  would hold 32; this one holds **16** on ConsA and 18 on ConsB. (Corroborating:
  the trace buffer is only 15% full, so nothing was truncated at the end, and
  `PERF_CNT_2` keeps ticking 8200 cycles past the last core event -- the trace
  started LATE, it did not stop early.)

## Emulator vs HW (2026-07-14) -- the mechanism AND the magnitude match, per rep

The derived arbitration model (Tasks 1-7: per-physical-bank round-robin,
REQUEST -> ARBITRATE -> COMMIT, `CoreStatus::WaitBank` costing a real cycle) was
run against the traced kernel in-process
(`of_q0_rich_bank_arbitration_vs_hw` in `src/testing/xclbin_suite.rs`, per-cycle
recorder `src/interpreter/engine/bank_census.rs`). No constant was tuned. Cycles
compared, not decoded record counts; emulator cols are HW cols minus 1.

**Compare PER REP, not run totals.** The capture holds ~9 of the kernel's 16
pipeline iterations (see the coverage trap above); the emulator runs all 16. Each
outer-loop rep is one stall burst, and the burst is the unit both sides measure
identically:

| per burst (one outer-loop rep) | EMU | HW |
|--------------------------------|----:|---:|
| stall cycles                   |  24 | 24 |
| burst span (cycles)            |  47 | 47 |
| dominant gap inside the burst  |   2 |  2 |
| contended banks                | 0/1, never both | 0/1, never both |
| **rep-bursts counted**         | **16** | **9** (ConsA) / **10** (ConsB) |
| stalls per rep                 | 24.4 | 24.4 |

Whole-run figures, for the record only -- they are NOT a fidelity comparison,
because the two intervals differ: EMU MEMORY_STALL 4 / 389 / 392 (Producer /
ConsA / ConsB) against a 9-and-10-rep HW window's 1 / 220 / 245. Contended cycles
EMU 8 / 404 / 407, core loses 96.3% (HW 94.8% / 96.1%). Of ConsA's 404 contended
cycles only 26 involve a bank a DMA demanded -- **378 are the core conflicting
with itself**, which is the mechanism, not an error (see refuted refinement 1).

### RETRACTED: "the model over-produces consumer stall cycles by ~1.9x"

An earlier version of this section compared the emulator's 16-rep run totals
against this capture's 9-rep window and concluded the model was "magnitude ~2x
high" on the consumers, opening a fidelity-gap row for a nonexistent core-port
bug. **That verdict was false and is withdrawn.** The 1.78 was `16 / 9`, nothing
else.

ConsB is the discriminator: its ratio is `394/245 = 1.61`, and its own burst count
gives `16 / 10 = 1.60`. A genuine magnitude error scales **uniformly** across both
consumers; a coverage gap scales with each capture's own rep count, which is what
is observed -- while the burst *content* (24 stalls, 47 cycles, gap 2, banks 0/1)
is byte-for-byte identical on both sides.

The full disproof, with the five independent proofs that the HW window opens
mid-run, is in `.superpowers/sdd/coreports-report.md`.

### What the model gets right

1. **The mechanism, not just the number.** A 1-cycle deny-and-retry at period 2 --
   96% singleton runs, dominant gap 2 -- the identical shape HW shows. A model
   charging one contiguous multi-cycle stall would have failed here.
2. **The magnitude, per rep.** 24 stalls per rep against HW's 24, in the same
   47-cycle span. The steady-state loop is a single VLIW bundle
   (`lda`+`st`, PC `0x210`, `LC=24`) whose load and store land in the same
   physical bank on every iteration, so the core is denied exactly once per
   bundle.
3. **The right tiles.** Consumers stall, the producer does not (EMU 4, HW 1). The
   pre-arc model was exactly INVERTED (32 conflicts on the producer, 0 on the
   consumers).
4. **The right banks.** Physical banks 0 and 1 only, never both in the same cycle
   -- matching HW's disjointness exactly. Banks 2-3 silent (HW-observed); 4-7
   silent too, but HW never traced those, so that is consistency, not validation.
5. **The cost is real.** Every stall cycle is a cycle in which the core's bundle
   did not retire.

### Residual real gaps

* **Pipeline-fill depth.** EMU free-runs 5 reps before blocking on the un-armed
  DDR drain; HW runs ~7. A ~2-rep fifo/host-arm timing question, unrelated to bank
  arbitration.
* **DMA memory-pressure events.** Closed separately by the staging-FIFO model
  (Stage 2, `924c47f3`) -- both events are FIFO-occupancy signals, not
  arbitration signals. Its one residual is a one-cycle fill difference (EMU's
  backpressure opens +16 cycles into a lock stall, silicon's +15), deliberately
  not fitted away.

## Provenance

Brief: `docs/superpowers/BRIEF-memory-stall-bank-arbitration.md`. Prior context
that side-stepped rather than root-caused this gap:
`build/experiments/sp3-spike-trace/SP4A-LEAN-FINDING.md` (removing compute drops
ConsA 220 -> 3). Gap registry row: `docs/fidelity-gaps/core-compute-timing.md`.
