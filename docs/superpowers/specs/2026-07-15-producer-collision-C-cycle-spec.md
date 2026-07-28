# Producer bank-collision fix -- C-cycle build spec

Date: 2026-07-15
Status: approved design, ready to plan
Supersedes the fix design in `2026-07-15-producer-collision-fix-design.md` (whose Part 1 was
BLOCKED). That doc holds the full journey (diagnosis, adversarial review, N=10 ceiling check,
C feasibility) -- the WHY. This doc is the WHAT/HOW.

## One-paragraph summary

The emulator under-counts producer core-vs-DMA bank conflicts ~16x (68 vs HW ~1100) because its
fixed-schedule DMA phase-LOCKS anti-correlated with the dense core (a deterministic anti-lock),
and its symmetric round-robin arbiter over-loses the few conflicts it has (44% vs HW ~1.6%). A
cycle-level microsim (`producer_csim_feasibility.py`) proved a faithful cycle-level mechanism --
**core-class-priority arbiter + DMA present-greedily / back-off-when-denied**, keeping the
existing egress FIFO depth 12 -- reproduces the HW STRUCTURE deterministically with NO phase
constant, NO RNG, NO sub-cycle re-architecture: dense -> phase-robust ~1000 conflicts; intermediate
-> phase-bimodal (matching HW's coin-flip); starvation 0. Backoff-on-denial is the deterministic
phase-drift that breaks the anti-lock. With conflicts at ~1000, core-priority yields ~1000 x 1.6%
~= 16-20 stalls -- in the HW band, so both errors are fixed by one coherent change (they were
coupled all along).

## The mechanism (C-cycle)

Three coupled pieces, all cycle-level:

1. **Core-class-priority arbiter** (`bank_arbiter.rs`). On a contended physical bank: if any
   core port wants it, a core port wins (round-robin WITHIN the core ports); a DMA channel wins
   only if no core port wants it (round-robin within DMA channels). Access-type-BLIND (the
   dense-READ > dense-WRITE ordering, #3, comes from the core presenting BOTH load ports' demands
   -- LoadA+LoadB occupy both sub-banks -> the DMA has no free slot -> more conflicts; the arbiter
   already models the 3 ports as independent requesters, so this emerges from faithful per-port
   demand, not a new rule).
2. **DMA urgency override.** A DMA demand flagged `urgent` (egress FIFO near underflow) beats the
   core. This is the ONLY way the core loses to the DMA -> the ~18-20 core stalls. Without it,
   core-priority zeroes DMA-caused stalls.
3. **DMA backoff-on-denial** (`stepping.rs` `next_granule_fetch` / the egress-staging fetch).
   When core-priority denies the granule fetch, the DMA does NOT hammer every cycle -- it backs
   off and re-presents later. The backoff SHIFTS the fetch phase relative to the core each denial,
   so the DMA cannot stay pinned in the dodge phase; it drifts and collides ~1/granule. This is
   what breaks the anti-lock. (Current EMU effectively "hammers/dodges" on a fixed schedule and
   stays anti-locked at 68.)

The FIFO depth stays 12 (the existing value, `model_builder.rs:~235`). Do NOT retune it -- the
microsim shows fifo=12 + backoff lands at the anchor; changing it is the calibration we reject.

## Validation targets

Robust anchors ONLY (the N=10 same-binary check proved intermediate densities are genuinely
HW-nondeterministic -- run-to-run bimodal on byte-identical binaries -- so they are NOT targets;
`collide_read`'s "~0" is itself a bimodal 16/0 and is DROPPED as an anchor):

| probe (`producer_probe.py`, EMU side) | HW CONFLICT | HW MEMORY_STALL |
|---|---:|---:|
| `apart` (dense write, different logical bank) | 0 | 0 |
| `collide` (dense write, same logical bank) | 1098-1103 | 18-20 |
| `collide_read_dense` (dense read, same bank) | ~1162 | ~24 |

Success:
- CONFLICT lands order-1000 for both dense probes (not 68), phase-robust (stable run-to-run in
  the deterministic emulator), starvation 0.
- **STALL is the PRIMARY DERISK**: must land ~18-24 ROBUSTLY (the microsim's toy stall model was
  phase-fragile: phi 0-2 -> 16, a degenerate phi=3 -> 1024, phi 4-7 -> 0; the real emulator's
  faithful FIFO/urgency model must not have that resonance). The physical story: stalls are a
  per-transfer FIFO-refill effect (~1-2/transfer x 16 ~= 18-20) -- the DMA force-grants a few
  times per transfer while filling its FIFO against the dense core, then coasts. If the faithful
  model reproduces ~18-24 robustly -> full success. If not -> fall back (see Gate below).
- dense READ conflict/stall >= dense WRITE (the port-occupancy ordering, #3).
- `apart` stays 0 (different logical bank never contends).
- Consumer-side core-vs-core self-collision (~91%) unchanged.

## Derisk-first GATE (do this before full implementation)

The stall count is the one open risk. FIRST, prototype the C-cycle mechanism far enough to
measure, in the EMULATOR, whether `collide` produces ~18-24 stalls ROBUSTLY (not just ~1000
conflicts). If yes -> proceed to full build. If the stall count is phase-fragile or off-band in
the real emulator too -> STOP and reassess: either refine the urgency/FIFO-refill model, or fall
back to "faithful conflict count + core-priority + a DOCUMENTED residual stall gap" (still ahead
of today's compensating-errors state). Do not grind on stall calibration -- if it needs a knob,
that is a finding to surface, not to hide.

## Interface changes

- `bank_arbiter.rs`: winner selection (symmetric rotor -> class priority + within-class rotor +
  urgency override). Demand must carry urgency: widen to `(Requester, bank_mask, urgent)` OR pass
  a parallel urgent set into `arbitrate` (planning decides). Contended-bank counting unchanged.
- `stepping.rs`: `next_granule_fetch` / egress-staging fetch gains backoff-on-denial state and
  surfaces an `urgent` flag (FIFO-occupancy driven) through `peek_bank_demand`.
- `coordinator.rs` `arbitrate_memory_banks` (~2059): thread urgency from the DMA demand into the
  arbiter call; unchanged otherwise.
- No change to `banking.rs` (bank map HW/ISS-verified) or the 1-in-4 granule width.

## Test plan

1. `bank_arbiter.rs` unit tests: re-express the symmetric-fairness tests as class-priority (core
   beats DMA; within-class rotation; urgency beats core; a non-urgent DMA does not starve because
   urgency escalates before underflow). KEEP the multi-bank sticky retry proofs. Add a >=2
   concurrent-urgent-DMA-channel starvation sweep (the review flagged this case as uncovered).
2. A coordinator/stepping test pinning the anti-lock break: dense same-logical-bank core + MM2S ->
   conflicts order ~1000 (not 68), stalls ~18-24, starvation 0; and dense READ >= dense WRITE.
3. Full re-verification: ISA 4815/4815; `cargo test --lib`; bridge suite.
4. EMU-side anchor validation via `producer_probe.py`: `apart` 0/0, `collide` ~1000/~18-24,
   `collide_read_dense` >= collide. Compare to the HW bands; do NOT compare intermediate densities.

## Non-goals

- Reproducing intermediate-density or `collide_sticky` conflict/stall counts (HW-nondeterministic,
  proven by the N=10 same-binary check).
- Retuning the egress FIFO depth (calibration).
- Matching the exact 1162-vs-1098 gap numerically -- only the read>=write DIRECTION.

## Blast radius

`bank_arbiter.rs` (winner selection + demand interface + several tests re-expressed; module docs:
anti-starvation becomes urgency-escalation, not rotor fairness), `stepping.rs` (fetch/backoff),
`coordinator.rs` (thread urgency). The retry-contract invariants and multi-bank sticky proofs are
preserved; the rotor starvation bound is replaced by an urgency-escalation bound that MUST be
built + swept (incl. multi-urgent-DMA) the way the old one was.
