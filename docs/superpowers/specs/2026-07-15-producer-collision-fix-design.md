# Producer bank-collision fix: core-class-priority arbiter + phase-independent DMA contention

Date: 2026-07-15
Status: **SUPERSEDED** by `2026-07-15-producer-collision-C-cycle-spec.md` (the C-cycle build
spec). This doc is retained as the JOURNEY record -- the diagnosis, the adversarial review that
broke Part 1, the N=10 determinism-ceiling check, and the C feasibility probe (the WHY). Do NOT
build the fix design in THIS doc; build the C-cycle spec.

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

## Adversarial review outcome (Opus, 2026-07-15)

An adversarial reviewer (tasked to break the design) found Part 1 as written rests on a
contradiction and the anchor set cannot validate the committed model. Accepted findings,
most-threatening first:

1. **KILL -- "phase-independent ~1100" is incoherent for a deterministic stepper.** The
   ~1100 conflict count is a sub-cycle phase-AVERAGE (uniform-phase sampling gives
   P(land on core's sub-bank)=0.5 x expected-wait ~2 cy = ~1 conflict/granule ~= 1100/run).
   A lockstep cycle-deterministic emulator is itself an oscillator: it phase-LOCKS to an
   EXTREME (our current 68 = the dodging phase; or a present-until-won overshoot of
   ~2000-4000), never the average. Reaching ~1100 therefore requires one of: a fixed phase
   offset (forbidden invariant c), RNG/jitter (forbidden invariant d), or injecting ~1
   conflict/granule "by construction" (the spec's own fallback -- which hardcodes the answer =
   calibration). There is no fourth way. The conflict count is below the emulator's
   STRUCTURAL fidelity (it needs the sub-cycle async the fixed-schedule model abstracts away).
2. **RESHAPE -- the ~18 stall target is FIFO-depth-set, and that depth is NOT HW-pinned.**
   `types.rs:~1316` states the egress FIFO depth is unconstrained by any HW capture (any
   depth >=~5 satisfies starvation=0); it is currently 12. "1100 x 1.6% -> 18" is an
   identity (1.6% := observed 18/1100), not a prediction, and the urgency-firing rate that
   would produce the stalls is a function of that unpinned depth. First-order it may not fire
   at all (a 12-word FIFO drains 4 words over the core's 4-cycle sub-bank dwell and never
   underflows -> ~0 stalls, undershoot).
3. **RESHAPE -- the mechanism is per-port sub-bank occupancy, NOT scalar density.** dense
   READ (1162/24, rock-solid) > dense WRITE (1098/18) because a dense read uses BOTH load
   ports and can occupy BOTH interleaved sub-banks in one cycle (no free sub-bank for the
   DMA -> more conflicts AND more forced grants), while a dense write uses the one store port
   and always leaves the other sub-bank free. An access-blind scalar-density model predicts
   these EQUAL and cannot order write<read. The arbiter already models 3 independent core
   ports -- the model must use that, not a density scalar.
4. **RESHAPE -- the "sparse->0" anchor sits at the density the spec disowns.** collide_read
   (~68%)=0 but collide_d67 (~67%)=201: same density, opposite verdict, differing only by
   free-cycle DISTRIBUTION (serial-accumulator spreads free cycles; gap_body clumps them).
   And collide_read is not a clean 0 (a 16/4 bimodal blip in ~7/10 reps). The discriminator
   is free-cycle spacing, not a density scalar.
5. RISK -- Part 1 and Part 2 are coupled (core-priority shifts the DMA's re-present timing ->
   shifts phase -> changes the conflict count); "1100 x 1.6%" is not a valid factorization.
6. RISK -- core-class priority removes the arbiter's PROVEN rotor starvation bound; the
   "urgency escalation" replacement is asserted, has a rare-yet-sufficient tension, and the
   multi-concurrent-DMA-channel case is unaddressed. New proof must be built + swept.
7. RISK -- the "determinism ceiling" rests on ONE bimodal rep in 3 (collide_d67 201/201/0)
   and does not exclude compile variance. Settle with a same-binary N=10 capture before
   locking validation scope.
8. RISK -- the AM020->core-priority reinterpretation is HW-backed for DIRECTION (55:1
   CONFLICT:STALL) but the urgency-as-anti-starvation MECHANISM is a one-probe inference.

**Reframe (the honest conclusion):** the CONFLICT count (~1100) is a sub-cycle phase-average
a lockstep cycle-emulator cannot reproduce without calibration -> it is a STRUCTURAL fidelity
gap, to be documented, not fitted. Part 2 (core-class priority) remains a genuine, HW-backed
arbiter bug worth fixing (it corrects the 44% over-loss), BUT the stall count it feeds is
entangled with the unreproducible conflict count and the unpinned FIFO depth, so exact 18-20
is likely not faithfully reachable either. The per-port sub-bank-occupancy mechanism (#3) is a
real refinement to fold in regardless. NEXT: (i) cheap same-binary N=10 HW capture across
intermediate densities to confirm/refute the determinism ceiling (#7) and the
collide_read/collide_d67 conflict (#4); (ii) rescope to "fix core-class priority + document the
conflict-count as a micro-timing/determinism-ceiling fidelity gap," NOT "reproduce 1100."

## N=10 same-binary determinism check (2026-07-15) -- ceiling CONFIRMED real

Re-ran the EXACT existing xclbins (no recompiles) 10x each. Genuine same-binary run-to-run
HW nondeterminism is PROVEN:

| variant | CONFLICT_DM_BANK across 10 same-binary runs | verdict |
|---|---|---|
| collide (control) | 1098/1103 (tight), stall 18/20 | stable -- pipeline trustworthy |
| collide_read | **16 x5, 0 x5** | BIMODAL coin-flip |
| collide_d75 | **201 x3, 0 x7** | BIMODAL coin-flip |
| collide_d50 | 52/57 | stable |
| collide_d67 | 402/405 | stable (this on-disk binary; != the prior 201/201/0 program) |
| collide_d83 | 621 x10 | stable |

**The determinism ceiling is REAL** -- `collide_read` and `collide_d75` collapse to 0 run-to-run
on a byte-identical binary (trace content genuinely differs; not a pipeline artifact). This is
the SAME "~50/50 coin flip" signature already documented for the shim MM2S DDR-read
(`project_tct_completion_inflight`). So NOT fitting the knife-edge conflict counts is empirically
justified, and **`collide_read`'s "~0" is NOT a clean anchor -- it is a bimodal 16/0; DROP it
from the anchor set.** The robust anchors are: `apart`=0, `collide`=1098-1103/18-20,
`collide_read_dense`=1162/24.

## The coupling consequence (worked out post-review, decision-forcing)

Because the conflict count is structurally stuck at EMU's phase-locked ~68 (unreproducible) and
stalls = conflicts x core-loss-rate, fixing the arbiter to faithful core-priority takes the EMU
stall count 30 -> ~1 (the finding's own math: 68 x 1.6% ~= 1), UN-cancelling the two compensating
errors. Current EMU 30 vs HW 18-20 is a FRAGILE coincidence on this probe (balloons to the
original "17 vs 1 ~10x" on of_q0_rich's short-object boundaries). So a pure arbiter fix REGRESSES
the observable stall count (abs error 12 -> ~19) while making the MECHANISM faithful and the
error CONSISTENT (steady undershoot vs variable overshoot). This trade -- and whether to ship any
code change at all vs bank the understanding as a documented structural gap -- is the open scope
decision.

## Option C feasibility probe (2026-07-15) -- cycle-level microsim (`producer_csim_feasibility.py`)

Cycle-stepped model of the producer scenario: dense/throttled core (sub-bank period 8),
DMA egress FIFO, core-priority arbiter, swept over the initial phase offset phi=0..7 and DMA
presentation/retry policies. Question: can a FAITHFUL cycle-level model hit the robust HW
anchor phase-independently, or does it need sub-cycle resolution (re-architecture)?

**Headline: sub-cycle re-architecture is NOT required.** A faithful cycle-level policy --
core-priority + DMA present-greedily + BACK-OFF-when-denied (keeping the existing FIFO depth
12) -- reproduces the right STRUCTURE deterministically, no phase constant, no RNG:

- **Full density -> phase-ROBUST ~976-1008 conflicts** across ALL phi (matches HW `collide`'s
  rock-stable ~1100), starvation 0.
- **Intermediate density (0.5-0.75) -> phase-BIMODAL** (16 or ~976 depending on phi), which
  structurally MATCHES HW's coin-flip at `collide_read`/`collide_d75` (16/0, 201/0).
- Mechanism: back-off-on-denial SHIFTS the DMA fetch phase each time core-priority denies it,
  so it cannot stay pinned in the dodge phase -- it drifts and collides ~1/granule on average.
  This is deterministic phase-drift at CYCLE granularity. It partially REFUTES the review's
  "incoherent" claim (#1): that holds for FIXED-cadence policies (hammer overshoots ~3000, jit
  dodges to 0), but a backoff retry deterministically breaks the lock.

**Honest caveats (why this is not a calibration-free silver bullet):**
- The absolute conflict count rides on FIFO depth (fifo=5 -> 0; 12 -> ~1000; 16/bo1 -> ~1500)
  and backoff length. FIFO=12 is the EXISTING value (`model_builder.rs:235`), so this is not
  NEW calibration -- but the number is FIFO-sensitive, and the review's point that FIFO depth
  is HW-unpinned (#2) stands: the producer anchor would implicitly PIN it to ~12.
- The STALL count (~18-20) is NOT cleanly reproduced: phase-fragile (phi 0-2 -> 16 ~= HW 18-20,
  but phi=3 -> a degenerate 1024 resonance, phi 4-7 -> 0). The toy's stall mechanism is too
  crude; a faithful FIFO-drain/urgency model is needed and is the real OPEN RISK.
- "Backoff on denial" is a plausible faithful DMA policy (a FIFO-buffered DMA with slack has no
  reason to hammer) but is INFERRED from the observable, not derived from an HW doc.

**Reframed verdict:** Option C = a buildable CYCLE-LEVEL fix (core-priority + backoff retry +
existing FIFO), NOT a re-architecture. It reproduces the fidelity STRUCTURE (robust-at-dense,
bimodal-at-intermediate, starvation-free) and the conflict count with the existing FIFO=12 --
strictly more faithful than the current state or A-strict. Its stall-count fidelity is the open
risk to derisk in implementation. This supersedes A-strict (C-cycle is more faithful AND
buildable). The decision: pursue C-cycle (accept the stall-count risk + the implicit FIFO=12
assertion), or fall back to A-lite (document) if the stall risk proves unresolvable.
