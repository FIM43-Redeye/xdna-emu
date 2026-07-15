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

**Open (single-port-specific, unresolved):** the exact HW conflict accounting for
write-vs-read -- why 1103 conflicts but only 20 stalls at K=4, when the ISS's
read-read defers with zero conflicts. Neither the ISS (models 1R1W) nor the microsims
pin it. NEXT is a cheap HW port-model experiment: compare CONFLICT_DM_BANK across
core-write-vs-DMA-read (have: 1103), core-read-vs-DMA-read, and (via S2MM)
write-vs-write, to characterize how the single port accounts conflicts -- then
implement the core-priority + free-cycle-deferral fix and validate against the HW
collide probe (target ~20 stalls / ~1103 conflicts).

## Traps for the next investigator

- The bank interleave is now HW/ISS-verified 16-byte; do not re-open it.
- aiesim's ISS does not model write-vs-read single-port contention -- do not use it as
  the oracle for the producer probe's conflict count. It IS a faithful oracle for the
  bank map and for read-read core-priority behavior.
- CONFLICT_DM_BANK is a per-bank coincidence-adjacent counter; MEMORY_STALL is the
  core losing arbitration. They are decoupled (1103 vs 20 at K=4).
- "AM020 says round-robin" is not "symmetric round-robin" -- it is core-priority with
  anti-starvation. The emulator's symmetric rotor is the bug.
