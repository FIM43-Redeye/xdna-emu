# Design: compute-core memory-bank arbitration (MEMORY_STALL)

Date: 2026-07-13
Branch: `feat/core-memory-stall-model`
Status: approved (Maya), ready for implementation plan
Finding this implements: [`2026-07-13-memory-stall-bank-arbitration.md`](../findings/2026-07-13-memory-stall-bank-arbitration.md)

## Problem

The emulator does not model the cycle cost of compute-core memory-bank
contention. Real NPU1 fires ~220 `MEMORY_STALL` per consumer core on a
memory-heavy loop; the emulator's existing model fires them on the **wrong tile
entirely** (32 on the producer, 0 on the consumers) and charges **zero cycles**
for the ones it does detect. The mechanism is inverted, not under-calibrated.

## The hardware truth (derived + HW-confirmed)

**Bank structure.** AIE2 compute-tile data memory is 64 KB as **eight physical
banks** (512 word x 128-bit, **single-port**, 8 KB each). "From a programmer's
perspective, every two banks are interleaved to form one bank, a total of four
banks of 16 KB" (AM020 ch.2:164). Bank[0] starts at address 0.

**Arbitration is PHYSICAL, and this is settled.** AM020 ch.2:166: "**Each memory
bank has its own arbitrator** to arbitrate between all requesters. The memory
bank arbitration is **round-robin**... only one request per cycle is allowed to
access the memory. The other requesters are **stalled for one cycle and the
hardware retries** the memory request in the next cycle." The arbiters are per
*physical* bank (the single-port SRAMs); the four-bank view is explicitly an
address-map abstraction ("from a programmer's perspective").

Three independent confirmations that arbitration is physical, not logical:
1. aie-rt exposes **eight** `CONFLICT_DM_BANK_0..7` events (77-84), not four.
   Conflict events are per-arbiter.
2. Our HW capture lit up **both** bank 0 and bank 1, near-evenly (115/109), for
   a kernel whose buffers all live in a **single logical bank**. One logical
   bank cannot split conflicts across two independently-counted arbiters unless
   the arbiters are physical.
3. Banks 2-7 were silent -- consistent with one logical bank = exactly two
   physical banks.

**Peano is not a counter-example.** Peano's "a same-bank load and store never
conflict (separate ports)" (`AIEHazardRecognizer.cpp:746`) is a compiler
scheduling heuristic at *logical* granularity: because a logical bank
interleaves two physical banks every 16 bytes, a load and a store to the same
logical bank land on *different physical banks* whenever their addresses differ
in bit 4 -- usually true for strided access. The rule **emerges from** the
physical model; it does not contradict it.

**Event semantics.** `CORE_MEMORY_STALL` (core event 23) = the core lost
arbitration and its datapath is stalled. AM020 ch.4:69: the core has two load
ports and one store port, and a bank conflict on **any** port stalls the **whole
datapath** -- so the stall granularity is the bundle, not the slot.

## HW evidence (capture `build/experiments/memory-stall-bankcap/`)

| Tile | MEMORY_STALL | CONFLICT bank 0 | bank 1 | banks 2-7 | S2MM_0_MEM_BACKPRESSURE | MM2S_0_MEM_STARVATION |
|------|---:|---:|---:|---:|---:|---:|
| Producer (1,2) | 1 | 2 | 1 | 0 | ~0 | 0 |
| ConsA (1,3) | 220 | 115 | 109 | 0 | 344 | 0 |
| ConsB (2,3) | 245 | 128 | 121 | 0 | 396 | 0 |

MEMORY_STALL ~= CONFLICT(bank0+bank1), slightly less: a conflict raises
`CONFLICT_DM_BANK_n` regardless of who loses, but `MEMORY_STALL` only when the
*core* loses. The core loses ~98% of collisions here.

## Design

### 1. Fix the bank mapping (foundation)

`MemoryAccess::bank()` (`src/interpreter/timing/memory.rs:305`) and
`banks_for_access()` (`src/device/banking.rs`) both use a flat
`(addr >> 4) & 7` -- an 8-way 16-byte interleave across the entire address
space. This is wrong: it scatters one logical bank across all eight physical
banks (and is why the emulator's conflicts "walk" banks 0-7 on the producer).

Replace with the derived, HW-confirmed compute-tile layout:

```
logical  = (addr >> 14) & 0x3              // four contiguous 16 KB banks
physical = 2 * logical + ((addr >> 4) & 1) // pair interleaved every 16 bytes
```

Confirmed by the capture: buffers at `0x400..0x5ff` -> logical 0 -> physical
banks 0/1, split near-evenly (the 16-byte interleave).

**Scope:** compute tiles only. The **memtile** has different, unvalidated
geometry (512 KB, 8 banks); leave its mapping untouched and document it. Do not
guess.

### 2. The arbiter

A per-tile bank arbiter over **8 physical banks**, each **single-port**, each
with a **persistent round-robin pointer** (one arbiter per bank, per AM020).

Requesters enumerated per bank per cycle:
- the core's load ports and store port;
- the tile DMA's S2MM x2 and MM2S x2 channels;
- *(extension point, not wired in this pass)* neighbor-core access ports.

**Granularity: physical.** A physical bank is genuinely single-port, so any two
requesters targeting the same physical bank in the same cycle contend --
including the core's own load and store when they land on the same physical
bank. Peano's logical-granularity rule falls out naturally (see above). This is
falsifiable: if it over-produces stalls, that is a signal to revisit.

**Excluded, with reason:** cascade is a separate FIFO-backed stream with its own
core stall event (25), not a bank requester. Memory-mapped AXI config access is
not modeled during execution.

### 3. Per-cycle phase order (request -> arbitrate -> commit)

The current loop is retroactive: the core commits its bundle (Phase 2), the DMA
commits its transfer (Phase 3), and only then does Phase 4 *observe* the
collision -- charging nothing and emitting a trace edge a cycle late. Replace
with:

```
A. DMA demand    compute each channel's intended bank(s) for cycle T from
                 descriptor state. No transfer yet.
B. Core demand   if the core is lock/dma/stream-stalled: no demand.
                 else decode the next bundle and run get_address /
                 get_store_address over its load/store slots -> banks.
                 No commit. (These are already pure functions of ctx.)
C. Arbitrate     per physical bank, round-robin among contenders.
D. Commit        core wins every bank it needs -> execute the bundle normally.
                 core loses any bank -> WaitBank stall: record_stall(1),
                   PC unchanged, retry next cycle.
                 DMA channel wins -> transfer. Loses -> hold, retry next cycle.
E. Emit          MEMORY_STALL on core loss; CONFLICT_DM_BANK_n per contended
                 bank; S2MM backpressure / MM2S starvation on DMA loss.
```

The stall reuses the **existing** per-cycle stall-retry machinery
(`try_resume_stall`, `interpreter.rs:660`) -- the same shape as
`WaitLock`/`WaitDma`/`WaitStream`, which are re-entered every coordinator cycle
and consume exactly one stall cycle per tick. This keeps the stall pinned to the
cycle it actually occurs, so the core's MEMORY_STALL edge aligns with the
memmod's CONFLICT edge exactly as on silicon. **No debt-carry, no timeline
drift.**

Rejected alternative: splitting the bundle into demand/commit phases for
per-slot arbitration. AM020 says a conflict on *any* port stalls the whole
datapath, so bundle-granularity gating **is** the faithful semantics; per-slot
granularity would be major surgery in the hot executor to model a granularity
the hardware does not have.

### 4. Cycle cost and events

**The cycle cost is the point and is currently absent.** Today Phase 4 only
bumps a `memory_stalls` stat; a detected conflict is free. `record_stall(1)` on
a lost arbitration (`context.rs:740`, which advances `ctx.cycles`) is what
finally makes a bank conflict cost a cycle and back-pressure upstream.

**MEMORY_STALL encoding is unchanged** (held level via `mem_stall_edge`).
Isolated one-cycle stalls each produce their own rising edge, so ~220 discrete
stalls should decode as ~220 events -- matching the capture. Verify, do not
assume.

**New emissions:** `CONFLICT_DM_BANK_n` must fire on any contention (not only
core-loses), and DMA backpressure/starvation events on DMA loss.

## Success criteria

**Target: faithful mechanism, ballpark counts** -- not byte-match.

- Conflicts land on the **right tiles** (consumers, not the producer) and the
  **right banks** (0/1 only, none on 2-7).
- Consumer MEMORY_STALL in the right order of magnitude (~200s, not 0, and not
  32-on-the-wrong-tile); producer in the low single digits.
- Lost arbitrations **cost cycles**.
- `cargo test --lib` green (3899 baseline).

**Why not exact reproduction:** the precise count depends on the emulator's
DMA cycle-level bank-access burst pattern matching silicon's. **We deliberately
model an idealized fast memory rather than any one silicon's DMA timing** (Maya,
2026-07-13) -- DMA behaviour differs with installed silicon, so chasing a
specific part's burst pattern would be calibration to a moving target, not
fidelity. Count deltas that trace to DMA burst timing are **documented as a
separate gap, never fitted away.**

**If faithful round-robin under-produces** (e.g. ~110 instead of ~220), that is
a *genuine finding* about the real arbiter's dynamics (request phasing,
pointer-advance rule, requester count) -- report it; do **not** install a
priority constant to hit 220. A likely benign explanation: the core is one
requester among several (S2MM + MM2S both contending on banks 0/1), so a fair
round-robin already loses most collisions for the core. We find out by
implementing and measuring.

## Risks

- **Blast radius.** Charging real stall cycles shifts cycle counts across
  *every* kernel. This is the intended effect but is wide. Assess explicitly:
  correctness regressions are blockers; timing shifts are expected and reviewed.
- **Physical-granularity assumption** (§2) is the most likely to bite. It is
  falsifiable via stall over-production.
- **Performance.** Per-cycle per-bank arbitration in the hot loop costs time.
  Correctness before performance (project rule); measure, do not pre-optimize.

## Out of scope

Memtile bank mapping; neighbor-core requesters (extension point); cascade;
memory-mapped AXI; the DMA burst-timing residual (separate gap).
