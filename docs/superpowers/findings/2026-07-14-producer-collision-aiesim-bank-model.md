# Producer bank-collision: aiesimulator oracle confirms the bank model and core-priority arbitration; refutes symmetric round-robin

Date: 2026-07-14
Branch: master
Hardware: real Phoenix NPU1 (AIE2) for HW counts; AMD aiesimulator (cycle-accurate AIE2 ISS, `aie2simmsm_dbg`, VC2802/xcve2802 device) as a per-cycle mechanism oracle.

Sequel to [`2026-07-14-producer-overcollision-phase-antilock.md`](2026-07-14-producer-overcollision-phase-antilock.md).
When no single-port model could reproduce the HW K-sweep, we ran the probe under
AMD's own cycle-accurate simulator to *observe* the arbitration instead of inferring it.

## The question that forced this

The producer collide probe (dense core marching logical bank 0 while the tile MM2S
drains the same logical bank) gives, per run, at dwell K=4/8/16/32:

| K | core MEMORY_STALL | CONFLICT_DM_BANK |
|---|---:|---:|
| 4  | 20  | 1103 |
| 8  | 29  | 98  |
| 16 | 46  | 141 |
| 32 | 136 | 443 |

(HW, 3 reps each, run-to-run identical; `tools/experiments/producer_probe.py`
variants `collide`/`collide_sticky{8,16,32}`, measured with `producer_probe_measure.py`.)

No single-port DMA + local-memory model reproduces this. ~20 microsim configs across
four mechanism families were falsified (`tools/experiments/producer_microsim.py`):
in-order greedy, in-order retry-persist + anti-starvation, out-of-order reorder, and
core-priority + free-bank prefetch + anti-starvation. The invariant failure: HW's
1103 conflicts at K=4 cannot coexist with low stalls + `MM2S_STARVATION=0` in any
single-port model. AM020 (ch.2:166) says arbitration is "round-robin," but the 55:1
CONFLICT:STALL ratio at K=4 implies the core has effective priority, which symmetric
round-robin cannot produce.

## The aiesimulator oracle (recipe is reusable)

`aiecc --xchesscc --xbridge --aiesim <probe>.mlir` + a host `test.cpp` (the NPU
`runtime_sequence` flow needs a C++ host `main()` driving the array via the generated
`aie_inc.cpp` XAie API; NPU-style probes provide none, so we supply one) builds a
cycle-accurate AIE2 sim. `aiesimulator --pkg-dir=.../sim --dump-vcd foo` runs it and
dumps a VCD exposing, per cycle, per physical bank b0..b7: `conflict_bN`,
`conflict_addr_bN`, every requestor port (`port_DMA_read0_bN`, `port_W_write_bN` =
the owning core, `port_{N,S,E,W}_*`, `port_AXI_*`), plus the HW events
`event23_memory_stall` (core) and `event77..84_conflict_dm_bank`. Full recipe +
tooling in `tools/experiments/aiesim/` (`repro.sh`, `test.cpp`, `collide_sim.mlir`,
`collide_read.mlir`, `vcd_probe.py`, `cycle_coincide.py`, `SIGNALS_tile_7_3.txt`).
Reduce REPS to ~3 to keep the VCD manageable.

## What the ISS settled

**1. Bank interleave CONFIRMED `physical = 2*((addr>>14)&3) + ((addr>>4)&1)`
(16-byte, bit-4 sub-bank).** Read directly off which `port_*_bN` each buffer's
accesses lit: `0x0400` and `0x2400` both hit bank-pair {b0,b1} (high-term 0),
alternating by bit 4; `0x8000` hits b4 (high-term 2). This was previously "Sol's
inference" (`2026-07-13-memory-stall-bank-arbitration.md`), unverified against
hardware; the ISS map matches it exactly. **`src/device/banking.rs` is correct**,
and the collide probe creates a *real* physical-bank collision -- the phase anti-lock
was not a bank-model artifact.

**2. Arbitration is CORE-PRIORITY (DMA yields), not symmetric round-robin.** In the
read-read run (core densely *reads* bank 0 in ~68% of cycles, MM2S needs 192 reads)
the two collide **0 times** -- the DMA schedules every read into a core-free cycle,
with 3 brief `memory_starvation` pulses (DMA waiting) and **zero** core stalls.
Symmetric round-robin would split ~50/50; this is the DMA deferring to the core.
Direction matches HW's 55:1 ratio. **The emulator's symmetric per-bank rotor
(`src/device/bank_arbiter.rs`) is the confirmed defect**; AM020's "round-robin to
avoid starving" is the *anti-starvation* layer on top of core-priority, not symmetric
fairness. The DMA's "defer accesses into bank-free cycles" behavior (FIFO-mediated)
is the free-cycle deferral the refined model needed, and the emulator's
fixed-schedule DMA fetch (the anti-lock, `stepping.rs:311` `next_granule_fetch`) is
what fails to do it.

**3. aiesimulator is NOT a faithful oracle for our probe's conflict accounting.** The
ISS models each bank's read and write ports as *independent*: 186 proven same-cycle,
same-bank core-**write** vs DMA-**read** events fired **zero** conflicts and zero core
stalls. But AM020 ch.2:164 says the bank is *single-port* and HW shows 1103
CONFLICT_DM_BANK for exactly that write-vs-read scenario. So the ISS abstracts
write-vs-read contention away (models 1R1W where silicon is 1RW); it only models
read-read contention (resolved by silent core-priority serialization, no event
emitted). It cannot reproduce the collide probe's 1103/20 accounting.

## The mechanism (grounded, two-part), and the open detail

- **Bank layout:** correct (verified).
- **Arbitration:** core-priority + anti-starvation (DMA yields; verified in the ISS
  read-read run and consistent with HW's 55:1). -> replace the emulator's symmetric
  rotor.
- **DMA:** opportunistically defers memory accesses into bank-free cycles via its
  egress FIFO (0 collisions when sparse). -> replace the emulator's fixed-schedule
  fetch (the anti-lock).

## HW port-model experiment (2026-07-14): the rule is DENSITY, not access type

Two experiments, same DMA read, `producer_probe.py` variants:

| core access | density | MEMORY_STALL | CONFLICT_DM_BANK |
|---|---|---:|---:|
| collide (core WRITE, 1 store/cy) | ~100% | 18 | 1098 |
| collide_read (core READ, serial accumulator) | ~68% | 0 | ~0 |
| collide_read_dense (core READ, 4 indep lanes) | ~100% | 24.5 | 1162 |

**First read (WRONG, superseded): "conflict is WRITE-vs-READ specific."** The first two
rows (write-dense=1098, read-serial=0) looked like an access-type rule -- a bank read
and a bank write contend, two reads don't. That fit the aiesim ISS too (ISS models the
bank as 1R1W, so its write-read = 0; its read-read = 0 by DMA deferral). But it also fit
a second hypothesis, and both had to be tested before building.

**Corrected read (CONFIRMED): the rule is DENSITY + DMA DEFERRAL, access-type-BLIND.**
The discriminator was a *dense* read-march (`collide_read_dense`: four independent
accumulator lanes, ELF-verified to sustain 1 load/cycle with zero free bundles). It gives
**~1162 conflicts / 24.5 stalls -- as much as, slightly more than, the dense WRITE.**
Rock-solid across 10 reps, no jitter. So a dense read collides exactly like a dense write:
**reads are NOT immune.** `collide_read`'s earlier 0 was because its serial
accumulator chain throttled it to ~68% density, leaving free cycles the elastic DMA read
deferred into -- not because read-read cannot contend. (Self-check: the fresh pipeline
reproduced collide 1098-1103/18-20 and collide_read ~0/0, byte-identical Chess ELF, before
the dense number was trusted. Note collide_read showed a small stable bimodal 16-conflict/
4-stall blip in ~7/10 reps -- genuine minor HW nondeterminism, two orders below collide,
consistent with the shim-bimodal HW-determinism ceiling recorded elsewhere.)

**The mechanism, finalized:** the physical bank is single-port (1RW). ANY two same-bank
same-cycle accesses contend, regardless of type or requester. Arbitration is
CORE-PRIORITY (the core wins; the DMA is denied and CONFLICT_DM_BANK counts the denial).
The DMA is ELASTIC (egress/ingress FIFO slack): it fetches memory granules
opportunistically and DEFERS into bank-free cycles. When the core is sub-dense
(free cycles exist) the DMA slots every access into a gap -> 0 conflicts; when the core is
100% dense (no gaps) the DMA collides every cycle it needs the bank -> ~1100 conflicts, and
the core stalls only at the floor (~20) where the DMA must force a grant to avoid its FIFO
underflowing the stream. read-vs-read is not special -- it is the same single-port
contention, and its 0 count at sub-density is the deferral, nothing else.

**Implication for the fix (CORRECTED -- access-type-blind):** the emulator's arbiter
(`bank_arbiter.rs`) must replace symmetric round-robin with CORE-PRIORITY over the DMA
(with anti-starvation) -- but it needs NO read/write awareness; do NOT thread an access
type through `Requester`. The elastic free-cycle deferral belongs in the DMA model
(`stepping.rs` `next_granule_fetch`, the fixed-schedule anti-lock): the DMA must defer its
granule fetch into bank-free cycles and force a grant only near FIFO underflow. That
FIFO-underflow force is where the ~18-24 core stalls come from -- an emergent property of
the DMA's elasticity, not a fixed arbiter threshold.

## Density sweep (2026-07-15): NOT a clean knee -- it is PHASE, confirming 2522be2b

To pin the "DMA fits in free cycles until ~75% core density" prediction, a write-march was
throttled to a spread of measured bank-0 store densities (store-port-limited by G stores/iter,
S to bank 0 + G-S to a bank-2 filler; `producer_probe.py` `collide_d*`). Prediction: conflicts
~0 until ~75%, then climb. RESULT: **no clean knee, non-monotonic, and run-to-run bimodal.**

| measured density | CONFLICT_DM_BANK (median/3 reps) | MEMORY_STALL |
|---:|---:|---:|
| 0% (d50 compiler-deferred its 1 store) | 52 | 14 |
| 33% | 2 | 1 |
| 50% | 402 | 106 |
| 67% | 201 (bimodal: 201/201/**0**) | 45 |
| 80% | 621 | 124 |
| 89% | 528 | 96 |
| 100% (collide) | 1103 | 20 |

80% > 89%; 67% < 50%; a 67% rep collapsed 201->0. Conflict is NOT a function of density alone.
This **confirms the repo's existing diagnosis** (`2522be2b`,
[`2026-07-14-producer-overcollision-phase-antilock.md`](2026-07-14-producer-overcollision-phase-antilock.md)):
the core (period-8 physical-bank march) and DMA (period-8 granule cadence) phase-LOCK, and the
outcome is a near-binary "every granule collides" vs "the DMA dodges every granule" set by
loop-entry transients. At intermediate densities that phase is unstable run-to-run (bimodal) --
i.e. **below the HW's own determinism ceiling**, so chasing the intermediate-density curve is
futile. Only the FULL-density anchors are robust: dense write 1098/18-20, dense read 1162/24,
sparse ~0.

**Reconciliation of "density/deferral" vs "phase":** both describe the same physics from
different ends. Sparse core genuinely leaves free slots -> DMA defers -> ~0 (robust). Full-dense
core occupies one physical sub-bank every cycle -> the DMA cannot reliably dodge and contends
~once per granule -> ~1100 (robust). The messy middle is the phase-lock knife-edge and is not a
fidelity target. The earlier "conflict is access-type-specific" read was wrong (dense read = 1162);
the correct invariant is: **a dense same-logical-bank core denies the DMA regardless of sub-bank
phase** -- 2522be2b's framing (c).

**NEXT (fix, unchanged from 2522be2b, refined here):** ONE two-part change --
(1) break the DMA phase anti-lock so it contends at ~HW rate against a dense core, PHASE-INDEPENDENTLY
(framing c: dense core denies the DMA every granule via contend-and-retry, NOT a fixed phase-offset
constant, NOT a jitter model); (2) core-CLASS-priority arbiter (access-type-BLIND, round-robin
WITHIN a class) so of those ~1100 conflicts the core loses only ~1.6% -> ~18-20 stalls. Validate
against the robust anchors (dense write 1098/18-20, dense read 1162/24, sparse/apart 0), NOT the
phase-locked intermediate densities. `collide_sticky`'s non-monotonic K-sweep is the same
phase-lock artifact.

## Traps for the next investigator

- The bank interleave is now HW/ISS-verified 16-byte; do not re-open it.
- aiesim's ISS is a faithful oracle for the bank MAP only. Its conflict ACCOUNTING is
  not faithful: it models the bank as 1R1W (independent read/write ports), so it fires
  no write-read conflicts, and it defers reads into free cycles, so its sub-dense
  read-read run also reported 0. HW proves a *dense* read-read genuinely contends
  (~1162, `collide_read_dense`). Do not use the ISS as the oracle for any conflict count.
- Access type does NOT gate bank contention. An early two-point read suggested
  "write-vs-read contends, read-vs-read doesn't"; the dense-read probe refuted it. The
  rule is density + DMA free-cycle deferral, access-type-blind. Do not re-introduce a
  read/write distinction in the arbiter.
- CONFLICT_DM_BANK is a per-bank coincidence-adjacent counter; MEMORY_STALL is the
  core losing arbitration. They are decoupled (1103 vs 20 at K=4).
- "AM020 says round-robin" is not "symmetric round-robin" -- it is core-priority with
  anti-starvation. The emulator's symmetric rotor is the bug.
