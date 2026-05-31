# Findings: BUG-B — AIE2 zero-overhead-loop store flush

**Date:** 2026-05-31
**Follow-on to:** `2026-05-30-buga-fix-and-retriage.md` (which flagged BUG-B)
**Status:** Root cause understood; modeled to the best simple approximation and
shipped. The exact flush condition is cycle-exact and under-sampled; a small
known residual remains (documented below). Resolving it fully would need the
aiesimulator oracle plus more commit samples than hardware naturally produces.

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
  body span. The workhorse for the discriminator search.
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

## Open follow-ups (separate from BUG-B)

- **seed_1806** (widened sweep): non-ZOL kernel, EMU zeros idx 1 where HW keeps
  it. A distinct divergence; needs its own triage.
- **The 96-byte flush/commit split**: if the aiesimulator oracle is ever stood
  up, seed_1048 (flush) vs seed_1086 (commit) -- both 96B, store+induction at LE
  -- are the canonical discriminating pair to trace cycle-by-cycle.

---

## Upstream

BUG-B is a confirmed **Peano codegen bug**: the simple-unroll path parks a
partial-word store in the LE bundle of a zero-overhead loop, where the back-edge
fetch redirect can flush it. Filed upstream (issue-first; Chess compiles the
same kernels correctly):
**[Xilinx/llvm-aie#1012](https://github.com/Xilinx/llvm-aie/issues/1012)**.
The minimal reproducer (`out[i] = i*i`, int8) and its verified `st.s8`-at-LE
disassembly are under `build/experiments/2026-05-31-peano-repro/`.
