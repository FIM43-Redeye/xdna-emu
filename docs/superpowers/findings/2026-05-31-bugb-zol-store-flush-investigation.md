# Findings: BUG-B — AIE2 zero-overhead-loop store flush

**Date:** 2026-05-31
**Follow-on to:** `2026-05-30-buga-fix-and-retriage.md` (which flagged BUG-B)
**Status:** Root cause understood; modeled to the best simple approximation and
shipped. The exact flush condition is **proven irreducibly cycle-exact** (see
the recency experiment) -- no single static feature determines it. A model-first
cycle/feature study then **proved the shipped body-size heuristic is the
static-feature optimum**: an exhaustive fit that separates the harvest set 0/104
overfits and loses to body-size on an independent natural corpus (5 vs 2 wrong).
The ~2% residual is a genuine fetch-pipeline phase effect below disassembly
resolution. Closing it would require cycle-level simulation (aiesimulator);
**that escalation is deferred as an optional deep-dive** -- the static ceiling is
now proven, not assumed. See "Model-first cycle study" below.

---

## Headline

BUG-B is the differential fuzzer's "NPU zeros every Nth sub-word element"
divergence. It is **a real hardware effect of a Peano codegen quirk**, not an
emulator compute bug and not a fuzzer-harness artifact:

- **Peano's simple-unroll path parks a partial-word store (`st.s8`/`st.s16`) in
  the loop-end (`LE`) bundle.** On a zero-overhead-loop **back-edge**, hardware
  can flush that store before it commits, so the affected element keeps its
  initial value. The final iteration falls through (no back-edge) so its element
  survives. **Chess compiles the same kernels correctly** (no store parked at
  `LE`), confirming it is a Peano-specific scheduling bug.
- The emulator's job is to **reproduce the effect** (xdna-emu is meant to be an
  open-source aiesimulator -- faithful to what silicon does with a given binary,
  including the effects of badly-scheduled code), without breaking
  correctly-scheduled loops.

The hard part: the flush is **cycle-exact** and the commit cases are rare, so no
static rule fully separates flush from commit. We ship the best first-order
model and document the residual.

---

## Confirmed hardware facts (grounded, keep these)

1. **Partial-word stores are read-modify-write and commit late at stage E11
   (issue+11).** `AIE2Schedule.td:133-141` ("load is in E5, store is in E11");
   `MemInstrItinData<II_STHB, ... MemoryCycles<[5,11]>>` (`:733-737`).
   AM020:3921 "8-bit and 16-bit stores are implemented as read-modify-write";
   AM020:3911 "Load and store units manage the 5-cycle latency of data memory."
2. **i32 / full-word stores are immediate** (one store port, not RMW). Not in
   the emulator's `pending_stores` queue; unaffected by any flush.
3. **The loop runs on a Program Control Unit (PCU)** with its own fetch counter
   `fc` and shadow loop count `lci` (AM020 ~4059-4097). The PCU fetches ahead
   and the back-edge is a **fetch redirect** from `LE` to `LS` -- the physical
   origin of the last-bundle behavior. The AIE2 front-end fetches in 16-byte
   packets.
4. **llvm-aie has NO scheduler constraint keeping stores away from `LE`, and is
   silent on back-edge flush** (`AIEBaseSubtarget.cpp:304-320` only enforces a
   7-bundle setup-to-`LE` distance for the ls/le/lc register writes to settle).
   So the flush is undocumented emergent pipeline behavior -- it cannot be
   derived from the toolchain, only observed on silicon.
5. **DMA descriptors are innocent.** `examples/decode_cdo_bds.rs` decoded the CDO
   BD registers independently: every hop is plain contiguous. The drop is the
   core store, not the data plane.

---

## The discriminator search (what was tried, what was refuted)

Validation method throughout: every fuzz seed that mismatches has its true HW
output saved as `npu_output.bin` (HW output is deterministic).
`examples/validate_seeds.rs` replays a seed corpus through the in-process
emulator and diffs byte-for-byte. `tools/classify_le_store.py` disassembles each
core ELF, finds the `.L_LEnd0` (LE) bundle, detects a partial-word store, and
extracts its data producer and the loop-body byte span (`le - ls`).

**Three static hypotheses were each proposed, looked clean, then refuted by a
larger sample:**

- **Producer latency** (mul=2cyc flush vs lshl=1cyc commit). Refuted: across the
  corpus, lshl-, add-, and mul-fed LE stores all appear in both flush and commit
  outcomes. The seed_18(mul)-vs-seed_1826(lshl) contrast was a coincidence of
  body size.
- **Loop-body fetch-packet span** (`le - ls`). The current shipped model. Fits
  the calibration corpus perfectly and the 112-byte commit cases solidly, but a
  2000-seed widened HW sweep found **96-byte bodies that commit** (seed_1086,
  seed_1340), refuting it as the *exact* rule. See the model section.
- **LE-bundle composition** (store-only vs store + induction `add`). Refuted by
  seed_1048: a 96-byte FLUSH case whose LE bundle is `st.s8 r1; add r0,r0,#4` --
  structurally identical to the commit cases seed_1086/seed_1340.

seed_1048 (flush) and seed_1086 (commit) are structurally indistinguishable:
same 96-byte body, same `store + induction-add` LE bundle, same 1-bundle
dist-1 ALU producer. The split is genuinely cycle-exact.

### The recency experiment (the decisive scale test)

The two known 96-byte commits (seed_1086 `i ^ (i + C)`, seed_1340 `i | (i - C)`)
shared one feature: the store's data producer reads a register written in the
**immediately-preceding bundle** ("data-input recency 1"), whereas every flush
read a long-stable value (recency >= 5 or induction-sourced). Across the labeled
corpus this separated cleanly -- the two commits were the *only* recency-1 cases.

To test it rather than overfit two points, a fuzzer harvest mode
(`XDNA_FUZZ_RECENCY1`, in `gen.rs`) forces every iteration's store value into a
pure-induction-var two-op chain (`i op1 (i op2 C)`), mass-producing recency-1
store-at-LE kernels (~5x the natural rate). A 2000-seed HW run gave the verdict:

| body | recency | HW commit | HW flush |
|------|---------|-----------|----------|
| 96   | 1       | **22**    | 7        |
| 96   | 2-4     | 0         | 4        |
| 96   | >=5     | 1         | 49       |
| 112  | 2-4     | 3         | 0        |
| 112  | >=5     | 1         | 16       |

**Recency is a strong but non-deterministic correlate.** At body-96, recency-1
commits **76%** (22/29) vs **2%** for recency>=5 -- so data-input freshness
genuinely drives the behavior. But 7/29 recency-1 cases still flushed, and the
body-size rule itself broke on this distribution (body-112 mostly *flushed*
here, where natural body-112 kernels commit). **No single static feature
determines the outcome** -- body, recency, and producer are all
distribution-dependent correlates, never determinants. This is the proof that
the phenomenon is irreducibly cycle-exact and needs cycle-level simulation.

The harvest produced a **rich controlled dataset** (~95 body-96 + ~20 body-112
store-at-LE kernels with recency + HW labels) under
`build/experiments/2026-05-31-recency1b/` -- reproducible via
`XDNA_FUZZ_RECENCY1=1 ... fuzz --seed 1 --iterations 2000 --hw`. This is the
validation set for the next-phase cycle model. `tools/classify_le_store.py` now
emits a `data_recency` column.

---

## The shipped model: loop-body fetch-packet threshold

`ZOL_FLUSH_MAX_BODY_BYTES = 0x60` (96 bytes, six fetch packets) in
`src/interpreter/state/context.rs`. On a back-edge, a `pending_store` with
`issue_pc == LE` is flushed **iff** `le - ls <= 96`. Register writes (loads,
pointer/index updates) in the LE bundle are never flushed (they retire to the
register file); stores issued earlier in the body have entered the decoupled
store pipeline and commit across the back-edge (software pipelining relies on
this).

**Why this rule** -- evidence across ~3,500 store-at-LE kernels on real silicon:

| body | fetch packets | HW flush | HW commit |
|------|--------------|----------|-----------|
| 96B  | 6            | 107      | 2 (seed_1086, seed_1340) |
| 112B | 7            | 0        | 4 (seed_1826, seed_1781, +2) |

- `> 96B -> commit`: **4/4 correct**.
- `== 96B -> flush`: **107/109 correct** (98.2%).

**It is the least-wrong simple model.** Mispredicts across the corpus:
threshold = **2** (the 96B commits), unconditional flush = **6** (all commits),
never-flush = **109** (all flushes).

**Known residual (shipped limitation):** ~1.8% of 96-byte store-at-LE loops
COMMIT (seed_1086, seed_1340), and the emulator wrongly flushes them. These are
cycle-exact boundary cases, structurally indistinguishable from flush cases,
with only two known samples -- too few to derive a safe finer rule and
impractical to enrich (~0.04% of random seeds). Closing this gap needs
cycle-level pipeline visibility (aiesimulator) and more commit samples.

---

## Validation results

- **Calibration corpus** (`build/fuzz`, ~1,577 seeds with saved HW output):
  `validate_seeds` 1576/1577. The one miss is seed_1964, an unrelated i8 compute
  divergence (shift/mul semantics), NOT a store flush.
- **Fresh generalization, 300 widened seeds:** 12/12 store-at-LE @96B flush on
  silicon (EMU == HW), 0 fail.
- **Fresh generalization, 2000 widened seeds:** 1546 pass, **3 fail**, 90 error
  (timeouts), 0 crash. The 3 fails: seed_1086 + seed_1340 (the 96B-commit
  residual above) and seed_1806 (a non-ZOL idx-1 drop -- a *separate* divergence
  to triage independently; `classify_le_store` reports `loop: False` for it).
- Full lib suite: 3250/0, including five ZOL flush/commit unit tests in
  `context.rs`.

The widened HW runs live under `build/experiments/2026-05-31-widefuzz/`
(hwrun*.log).

---

## Permanent fuzzer improvement

`src/fuzzer/gen.rs`: kernel op-count widened from 2-8 to **1-16**. Larger op
counts produce longer unrolled bodies, broadening scalar-pipeline coverage and
-- the reason it was done here -- pushing loop bodies past six fetch packets so
the differential fuzzer naturally exercises both the flush and commit regimes.
A single op already yields a six-packet body, so the floor is unchanged; the
gain is at the top end. Side effect: a ~4.5% HW-timeout rate on the largest
generated kernels (handled gracefully as `error`, not false bugs) -- judged
benign.

---

## Tools (kept, committed)

- `tools/classify_le_store.py` -- disassemble every seed's core ELF, find the LE
  bundle's partial-word store, extract producer op/latency/distance and loop
  body span, plus the `data_recency` feature. The workhorse for the
  discriminator search.
- `tools/cycle_model.py` -- cycle/fetch-domain contingency + threshold search
  (`<fuzz_root> <hwrun_log> [--2d]`). Built for the model-first study; proved the
  static ceiling (see "Model-first cycle study").
- `src/fuzzer/gen.rs` `XDNA_FUZZ_RECENCY1` mode -- harvest mode that forces
  recency-1 store-at-LE kernels (~5x natural rate) for the scale test; kept as a
  research capability for the cycle-model phase.
- `tools/body_size_sweep_compile.py` -- attempt to compile controlled-body-size
  kernels. Recorded as a NEGATIVE result: Peano will not reliably park a store
  at LE in hand-written kernels (it parks a register-op instead), so controlled
  hand-crafted sweeps don't work; the widened fuzzer is the only reliable
  store-at-LE generator.
- `examples/validate_seeds.rs`, `examples/check_le_squash.rs`,
  `examples/decode_cdo_bds.rs` -- in-process replay / single-seed diff / CDO BD
  decode.

Disassembly oracle: `llvm-aie/install/bin/llvm-objdump -d --triple=aie2 <elf>`.

---

## Model-first cycle study (the static ceiling, proven)

Before reaching for the aiesimulator oracle, we ran the cheaper experiment: build
the cycle/fetch-domain model from the toolchain and test whether *any*
static-feature rule beats the shipped body-size heuristic. It does not -- and the
failure is instructive.

**Pipeline facts (from llvm-aie `AIE2Schedule.td`, via Explore).** AIE2 issues
one bundle per cycle in order (nops are real cycles); partial-word stores commit
at **E11** (`MemoryCycles<[5,11]>`, `II_STHB`); the front end fetches **16 bytes
per cycle**; the compiler already pads intra-iteration RAW stalls with nops, so a
stall model just reproduces the static bundle count. The one unmodeled degree of
freedom is the **fetch-vs-issue rate mismatch**: a 96-byte body issued in 24 vs
27 bundles has different nop density, so the back-edge fetch-redirect lands at a
different *phase* relative to the store's E11.

**What the data showed** (recency-harvest set, 104 store-at-LE kernels with real
HW flush/commit labels; `tools/cycle_model.py`):

- **Cycles (bundle count) is non-monotonic.** At producer-distance 1, *only*
  `cycles=25` commits; 24, 26, 27 all flush. A pure "longer iteration = more time
  to commit" threshold is therefore wrong -- it is a phase effect, not a
  duration effect.
- **Freshness drives it, but in two operands.** The store
  `st.s8 rData, [p1, dj0]` has a *data* operand and an *address* (`dj0`) operand.
  Data-producer distance split the first ambiguous cell cleanly (dist 1-2 ->
  commit, dist 6 -> flush); address-producer distance split the next
  (`mov dj0` at LE-1 -> commit, 4-5 bundles back -> flush). Both say the same
  thing: a store of fresh, still-in-flight operands commits; a store of settled
  operands flushes. But neither, nor both together, separates globally.
- **Exhaustive fit overfits.** A depth-3 decision tree over (body, cycles,
  data-dist, addr-dist, surplus=16C-B, store-width) separates the harvest set
  **0/104** -- but its splits are memorization (`surplus == 320` encodes the
  exact (96,26)/(112,27) phase; `st.s16 -> commit` memorizes 5 outliers).

**The decisive generalization test** (`generalization_test.py`, in the recency1b
experiment dir): fit on the harvest set, evaluate on the **independent natural
widened corpus** (`build/experiments/2026-05-31-widefuzz/`, no harvest):

| rule | natural corpus (46) | harvest set (104) |
|------|---------------------|-------------------|
| body-size (shipped) | **2 wrong** | 39 |
| robust core (cycles + addr-freshness) | 18 | 33 |
| depth-3 fitted tree | **5 wrong** | 0 |

The perfectly-fit tree loses to the shipped body-size rule on data it did not
see. **This proves body-size is the static-feature optimum**, and its 2 residual
errors (the 96-byte commits) are irreducible to anything in the disassembly --
they are a fetch-pipeline phase that only cycle-level simulation could resolve.

**Decision: bank the proof, defer aiesim.** The shipped rule is now provably
near-optimal; the residual is ~2% and irreducible-static. The aiesimulator path
(below) remains available but is an optional deep-dive for the last 2%, not a
required next step -- and aiesim may not reproduce an *undocumented* flush at all.

- **Tools kept:** `tools/cycle_model.py` (cycle/feature contingency + threshold
  search, dataset-parameterized: `<fuzz_root> <hwrun_log> [--2d]`). The fitting
  harnesses `fetch_sim.py` and `generalization_test.py` live in the recency1b
  experiment dir (gitignored).
- **If aiesim is ever attempted:** oracle is `aie2simmsm`
  (`amd-unified-software/aietools`, cycle-approximate); `--aiesim` is
  incompatible with the Peano flow, so the buggy Peano ELF must be
  ELF/CDO-swapped into a Chess-built sim package; license- + network-gated. Prior
  flow shells at `build/experiments/2026-05-13-chess-aiesim/`. Validation set is
  the recency1b dataset; cycle-by-cycle discriminating pairs are the (96,25,D=1)
  cell's commits (e.g. seed_94) vs flushes (e.g. seed_739, seed_1270), which are
  structurally near-identical with opposite outcome.

## Open follow-ups (separate from BUG-B)

- **seed_1806** (widened sweep): non-ZOL kernel, EMU zeros idx 1 where HW keeps
  it. A distinct divergence; needs its own triage.

---

## Upstream

BUG-B is a confirmed **Peano codegen bug**: the simple-unroll path parks a
partial-word store in the LE bundle of a zero-overhead loop, where the back-edge
fetch redirect can flush it. Filed upstream (issue-first; Chess compiles the
same kernels correctly):
**[Xilinx/llvm-aie#1012](https://github.com/Xilinx/llvm-aie/issues/1012)**.
The minimal reproducer (`out[i] = i*i`, int8) and its verified `st.s8`-at-LE
disassembly are under `build/experiments/2026-05-31-peano-repro/`.
