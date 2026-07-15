# Producer bank-collision fix: core-class-priority arbiter + phase-independent DMA contention

Date: 2026-07-15
Status: design, pending review
Supersedes the "NEXT" sections of the two producer findings; this is the build spec.

## Problem

On a dense same-logical-bank producer (compute core marching logical bank 0 while
that tile's MM2S drains the same bank), the emulator diverges from real Phoenix HW
in two compensating ways (diagnosed in
[`2026-07-14-producer-overcollision-phase-antilock.md`](../findings/2026-07-14-producer-overcollision-phase-antilock.md),
grounded further in
[`2026-07-14-producer-collision-aiesim-bank-model.md`](../findings/2026-07-14-producer-collision-aiesim-bank-model.md)):

1. **Phase anti-lock (~16x too few conflicts).** The emulator steps the core and DMA
   in rigid lockstep. Both alternate the two 16-byte-interleaved physical sub-banks of
   logical bank 0 with period 8 at the same rate; the fixed step schedule pins them in
   a stable half-period ANTI-phase, so the DMA's granule fetch lands on the sub-bank the
   core is NOT using. EMU registers 68 CONFLICT_DM_BANK cycles/run; HW registers ~1100.
2. **Symmetric round-robin (over-loss).** `bank_arbiter.rs` is a symmetric per-bank
   round-robin over all requesters. On a contended bank the core loses ~44% of the time;
   HW gives the core near-priority and loses only ~1.6% (the DMA absorbs the loss via its
   egress-FIFO slack, `MM2S_STARVATION = 0`).

Few-but-lossy (EMU) net-cancels to ~1.6x on this probe, but the cancellation is fragile
(short-object boundaries stack it). Both halves must be fixed together: core-priority
alone undershoots (68 x 1.6% -> ~1 stall); dense-conflicts alone with the symmetric rotor
overshoots (1100 x 44% -> ~480 stalls); both together give 1100 x 1.6% -> ~18, matching HW.

## Validation targets (robust anchors ONLY)

The density sweep proved conflict is NOT a clean function of core density: it is
phase-locked and run-to-run BIMODAL at intermediate densities (a 67%-density rep
collapsed 201->0), i.e. **below the HW's own determinism ceiling.** Do NOT calibrate
against intermediate densities or the `collide_sticky` K-sweep -- those are phase-lock
artifacts. Calibrate ONLY against the robust, run-to-run-stable anchors:

| probe (`producer_probe.py`) | core | HW CONFLICT | HW MEMORY_STALL |
|---|---|---:|---:|
| `apart` | dense write, different logical bank | 0 | 0 |
| `collide` | dense write, same logical bank | 1098-1103 | 18-20 |
| `collide_read_dense` | dense read, same logical bank | ~1162 | ~24 |
| `collide_read` | ~68%-dense read (serial acc) | ~0 | ~0 |

Success = the emulator lands in these bands (order-of-magnitude on CONFLICT, within a few
on STALL), `MM2S_STARVATION` stays 0, AND the existing consumer-side core-vs-core results
(the ~91% core self-conflict path) are unchanged.

## The fix: ONE two-part change

### Part 2 -- arbiter: core-class priority with a DMA urgency override (`bank_arbiter.rs`)

Replace the symmetric per-bank round-robin winner selection with **class priority**:

- On a contended physical bank, if any **core** port (`Requester::Core`) wants it, a core
  port wins; rotate the round-robin pointer WITHIN the core ports only.
- A **DMA** channel wins a contended bank only if no core port wants it that cycle; rotate
  within the DMA channels.
- **Urgency override:** a DMA demand flagged `urgent` (its egress FIFO is near underflow --
  it cannot afford to wait or the stream starves) beats the core. This is the ONLY way the
  core loses to the DMA, and it is what produces the ~1.6% core stalls (~18-20/run). Without
  it, core-priority would zero DMA-caused core stalls.

Interface change: the arbiter demand must carry urgency. Options (decide in planning):
either widen the demand tuple to `(Requester, bank_mask, urgent: bool)`, or pass a parallel
`urgent` requester set into `arbitrate`. The urgency signal originates in the DMA engine
(FIFO occupancy), surfaced through `peek_bank_demand`.

WITHIN-class round-robin preserves the retry contract and the existing multi-bank sticky
proofs unchanged (the module's starvation sweeps are re-expressed as within-class, plus a
new proof that a non-urgent DMA never starves BECAUSE urgency escalates it before its FIFO
underflows -- the anti-starvation mechanism moves from "the rotor rotates" to "urgency
escalates," which is what HW does).

Contended-bank counting (drives CONFLICT_DM_BANK) is unchanged: any bank with >=2 wanters is
contended, regardless of who wins.

### Part 1 -- break the anti-lock, phase-independently (mechanism C, `stepping.rs`)

**Principle (framing c, HW-grounded):** a dense core owning a logical bank uses one of its
two interleaved physical sub-banks EVERY cycle, so an MM2S reading the same logical bank
cannot reliably find a free slot -- it contends and re-presents ~once per granule. Model the
NET physical fact, NOT the exact sub-cycle phase (which is at the determinism ceiling).

**Hard invariants the mechanism MUST satisfy** (these are the guardrails; the exact code is
derived against the anchors during implementation, not hand-fitted):

- **No phase constant.** No fixed sub-cycle offset that aligns the DMA to the core -- that is
  calibration and the real phase is bimodal.
- **No RNG / jitter.** The emulator stays deterministic (rules out framing b).
- **No DMA starvation.** The DMA must still win enough granules to keep the stream fed
  (`MM2S_STARVATION = 0`). A dense-core logical-level "deny always" is WRONG -- it starves the
  DMA. The DMA wins on cycles the core is on the OTHER physical sub-bank; physical-sub-bank
  independence is real.
- **Sparse core -> ~0 conflicts.** A sub-dense core (e.g. `collide_read`'s ~68%) genuinely
  leaves free slots the DMA defers into; the mechanism must NOT fabricate conflicts there.
- **Preserve core-vs-core.** The consumer-side per-physical-bank arbitration for core self-
  collision is unchanged.

**Candidate mechanism (starting point, to be validated against the anchors):** the anti-lock
is born because the emulator's DMA fetches a whole 16-byte granule in the single cycle it
wins and then idles ~4 cycles paced by drain, and that one winning cycle is pinned to a free
sub-bank phase. Break it by making the DMA's granule fetch **contend at the phase the stream
demands the granule** (drain-driven, in-order) and hold/re-present under denial, rather than
opportunistically pre-fetching into whichever cycle its sub-bank happens to be free. Against a
dense core the fetch then coincides with the core's same-sub-bank cycles ~half the time
(~1 contended cycle per granule -> ~1100/run); the core wins by priority; the DMA completes
the granule on a following cycle when the core is on the other sub-bank; it escalates to
`urgent` only when its FIFO nears underflow (~18-20 forced grants -> core stalls). Whether
this reproduces the anchors WITHOUT a phase constant is the core implementation question; if
the drain-driven-present framing still pins a benign phase, the fallback is an explicit
granule-level contention model (the DMA's per-granule bank acquisition against a
same-logical-bank core costs its expected ~1 contended cycle by construction).

Anchors in code: `stepping.rs:311` `next_granule_fetch` (where the fetch phase is born),
`:156` `channel_bank_mask` (routes demand to the peek), `coordinator.rs:2059`
`arbitrate_memory_banks` (where core + DMA demands meet the arbiter).

## Non-goals

- Reproducing intermediate-density or `collide_sticky` conflict counts (phase-lock artifacts).
- A perfectly-accurate per-cycle phase model. The target is the robust anchors and the
  correct MECHANISM, to the HW's own determinism limit (the project's zero-calibration
  north-star).

## Test plan

1. Unit: `bank_arbiter.rs` -- rewrite the symmetric-fairness tests as class-priority
   (core beats DMA; within-class rotation; urgency beats core; non-urgent DMA does not starve
   because urgency escalates). Keep the multi-bank sticky retry proofs.
2. Unit: a coordinator/stepping test pinning the anti-lock break -- a dense same-logical-bank
   core + MM2S produces conflicts of order ~1000 (not ~68), core stalls of order ~20 (not
   ~480), starvation 0; a sparse core produces ~0.
3. Full re-verification: ISA 4815/4815; `cargo test --lib`; bridge suite.
4. HW-anchor validation via `producer_probe.py` (EMU side): `apart` 0/0, `collide`
   ~1100/~18-20, `collide_read_dense` ~1162/~24, `collide_read` ~0. Compare EMU to the HW
   bands above; do NOT compare intermediate densities.

## Blast radius

- `bank_arbiter.rs`: winner selection + demand interface (urgency); several tests re-expressed
  from symmetric fairness to class priority. The retry-contract module docs updated (anti-
  starvation now = urgency escalation, not rotor fairness).
- `stepping.rs`: `next_granule_fetch` / the egress-staging fetch timing.
- `coordinator.rs`: `arbitrate_memory_banks` demand assembly (thread urgency through).
- No change to the bank map (`banking.rs` -- HW/ISS-verified) or the granule width (1-in-4).
