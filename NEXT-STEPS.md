# NEXT-STEPS — SP-5b R3b (resume here)

**Date:** 2026-07-01
**Role of this doc:** the single pickup point for resuming SP-5b R3b work as the
orchestrator. Read this first. It carries the ruling and the exact rollback to
execute, and points into the dense history
(`docs/superpowers/2026-07-01-sp5b-r3b-NEXT-STEPS.md`) and the findings only
where depth is needed. Nothing here needs re-deriving — the identifiability
decision that was blocking is now made.

---

## 1. One-paragraph situation

SP-5b builds the *measurement apparatus* for per-tile clock-skew calibration of
the timer-sync broadcast model (#140). The 4-knob skew model is
`origin_X = dn_v*d_v + dn_h*d_h + intra(kind)`, and it bakes in four structural
assumptions: (1) per-hop uniformity, (2) within-axis direction isotropy (one
`d_v` for North+South, one `d_h` for East+West), (3) cross-axis additivity (no
turn penalty), (4) per-channel hop-cost uniformity. Phase 1 (R1, within-column
`d_v` + intra) is merged and its silicon gate is green. The next build was R3b,
a two-flood perf-counter instrument intended as the *primary* `d_h/d_v`
instrument and — per the soundness audit's Q1 remediation — as the hook that
would *falsify* the isotropy assumption via "enriched geometry" (signed N/S +
E/W solver columns). During TDD execution of that plan, Task 3's rank-check
proved the enriched geometry is unrealizable. **That blocker is now resolved
(Section 2). The work in front of the resuming session is executing the
resulting rollback (Section 3).**

## 2. The resolved decision

**Ruling: Option 1 — accept the rank-2 reality, revise the solver to `{d_h,
d_v}` (+ optional turn term for non-corner sources), and mark within-axis
direction isotropy as ASSUMED, not measured.** This converges with the prior
Opus recommendation and was independently adjudicated fresh. The adjudication
also *sharpened* it in three ways that the original finding
(`docs/superpowers/findings/2026-07-01-r3b-two-source-identifiability-limit.md`)
did not have:

**(a) The theorem is correct; one illustrative line in the finding is wrong.**
The load-bearing algebra holds: `max(dx,0) − max(−dx,0) = dx` gives
`e − w = tile.col − src.col` unconditionally, so `a_hE − a_hW = s1.col − s2.col`
is a per-pair constant, E/W columns collapse after referencing (likewise N/S),
rank ceiling 3, and corner sources add a third null direction → rank 2. **But
the finding's illustrative corner-collapse coefficient line
`a_turn = 5*a_hW + 2*a_vS − 10` is wrong** — it is convention-dependent on the
exact hop definitions in `r3b_observe.py`, and independent re-derivations
produce different constants (the two array dimensions appear transposed, and the
additive constant does not survive a clean derivation). **Action:** in the
finding doc, restate that line generically as "a linear combination of the axis
columns plus a constant" rather than as a bare numeric identity, or re-derive it
carefully from `r3b_observe.py`'s actual `_hops` sign conventions and pin it.
The rank-2 conclusion is unaffected either way.

**(b) Within-axis direction isotropy splits — vertical is measurable, only
horizontal must be assumed.** The audit's "put a same-kind tile on the opposite
side of the source along each axis" instinct was right but filed under the wrong
instrument. R3b (interval-based) can never realize it — the anisotropy term
enters every interval as a per-pair constant and is nulled at *any* source count
(this is *why* a "third source", Option 2, is not merely expensive but
self-defeating: separating within-axis directions requires an *absolute*
timing observable, which for the horizontal axis is exactly the ~30-cycle
cross-column jitter R3b's interval design exists to dodge). **But R1 measures
absolute within-column arrival offsets, and HW shows those pairs are jitter-free
(range-0).** So a *two-sided R1 spine* — source placed mid-column with measured
cores both North and South — makes **vertical** direction anisotropy
*falsifiable for free* on the already-built R1 instrument (inject `d_vN ≠ d_vS`,
require `fit_residual` to grow). Phoenix has only 4 core rows (2–5) plus
memtile/shim, so the two-sided span is thin — enough to *falsify*, not to
*precisely identify* `d_vN` vs `d_vS`, but strictly better than R3b's structural
zero. **Net:** the genuinely-assumed surface collapses from "all within-axis
isotropy" to **horizontal direction isotropy alone**, a second-order asymmetry
on a single-digit-cycle per-hop cost.

**(c) The honesty guard.** Horizontal direction isotropy (`d_hE = d_hW`) is
ASSUMED, structurally unmeasurable by any two-source interval instrument,
physically defensible as a second-order asymmetry. The emulator model already
assumes exactly this (`src/device/state/effects.rs:491-494`, one `d_v` for N/S,
one `d_h` for E/W) — so R3b's blind spot and the model's assumption are *the
same assumption*, and feeding an isotropic `d_h` forward launders nothing **as
long as the provenance says so**. The `calibrated` flip provenance (SP-5c) must
carry the line "horizontal direction isotropy ASSUMED, not measured" at the
point of use, not buried in a design doc.

## 3. Execution — the R3b rollback (this is the work)

TDD, one testable deliverable per step, on branch `feat/sp5b-r3b-pc-software`
(Tasks 1-2 are already committed+pushed there; this rolls back Task 1's
over-reach and adds the R1 check).

**Solver / code:**
1. Revert `tools/calibration/skew/r3b_extract.py` to the `{d_h, d_v}` fit against
   signed net displacement `(dn_h, dn_v)` (essentially its pre-Task-1 form). Add
   an **optional** `d_turn` column, enabled only for non-corner source configs
   (so the turn column is not collinear with the axis columns + const). Remove
   the 5-param form and the rank-5 guard.
2. Delete `test_recovers_all_five_params` — it passes only on hand-synthesized
   rows that violate `a_hE − a_hW = const`, i.e. data the bridge can never emit;
   a test asserting an impossibility is worse than none. Replace with (a) a
   `{d_h, d_v}` recovery test on realizable corner-source rows, and (b) a
   negative test asserting the E−W / N−S null directions are detected and
   reported as *assumed*, not silently averaged.
3. Add a two-sided vertical falsification test to the **R1** solver
   (`tools/calibration/skew/r1_diff_extract.py` and its test): mid-column source,
   cores both sides; inject `d_vN ≠ d_vS`; require `fit_residual` to grow.

**Doc rollback (upgrade existing errata pointers from "OPEN decision" to
"RESOLVED: Option 1 + R1 reallocation"):**
- `docs/superpowers/specs/2026-06-30-sp5b-measurement-apparatus-design.md`
  **Sec.2**: rewrite the "enriched geometry falsifies all four assumptions"
  claim to the four-way split — per-hop uniformity ✓ (≥3 collinear), additivity
  ✓ (non-corner sources), channel uniformity ✓ (separate dual-channel probe),
  vertical isotropy ✓ (two-sided R1 spine), horizontal isotropy ✗ ASSUMED.
  **Sec.5.4**: strike the signed N/S+E/W + interaction solver columns for R3b;
  replace with `{d_h, d_v}` (+ turn non-corner); redirect vertical-anisotropy
  falsification to R1. **Sec.4.5**: add the two-sided (mid-column source) R1
  spine as the vertical hook, with the 4-core-row precision caveat.
- `docs/superpowers/specs/2026-06-30-sp5b-kernel-hw-bringup-design.md`
  **Sec.5.1**: remove the "two-sided per axis" bullet as an *R3b* isotropy hook;
  keep the off-axis diagonal tile for additivity (non-corner) and keep the
  channel-uniformity control; fix or genericize the `a_turn` coefficient line.
  **Sec.5.2**: revert `r3b_observe`/`r3b_extract` to signed net displacement
  `{d_h, d_v}` (+ optional turn); drop the signed-indicator/interaction column
  requirement.
- `docs/superpowers/findings/2026-07-01-r3b-two-source-identifiability-limit.md`:
  fix the corner-collapse coefficient line (2a above); add the two sharpenings —
  interval methods cannot recover within-axis direction at *any* source count
  (strengthens the Option-2 rejection), and the vertical check belongs on a
  two-sided R1 spine; record the ruling (Option 1 adopted, vertical reallocated
  to R1, horizontal isotropy assumed).
- `docs/superpowers/findings/2026-07-01-sp5b-soundness-audit.md`: mark Q1
  remediation RESOLVED per this ruling (its "enriched geometry falsifies
  isotropy" over-reach is conceded for the within-axis-*horizontal* case;
  realized for vertical via R1).
- `docs/superpowers/plans/2026-07-01-sp5b-r3b-pc-enriched.md`: update the BLOCKED
  header to RESOLVED; adjust Task 1's parameter set to `{d_h, d_v}` (+ optional
  turn) and drop Task 3's enriched-geometry rank-5 guard.

**Then** the R3b build itself (design Sec.5, still valid on everything except the
enriched-geometry point) can proceed: the hand-authored MLIR flood kernel, the
counter-readback host (the hard part is readout, not config), `r3b_observe.py`,
and the `r3b_pc_gate.sh` gate. SP-5b still produces *no number* — correctness is
SP-5c's human causal-vs-HW gate.

## 4. Current committed state (verify with `git log`, not memory)

- `master` carries this doc + the identifiability finding + errata pointers on
  the audit finding, both rev3 design docs, and the plan header.
- Branch `feat/sp5b-r3b-pc-software` (pushed): Tasks 1-2 green — the 5-param
  `r3b_extract.py` (to be reverted per Section 3) and `r3b_observe.py`. Task 3
  was reverted; branch sits clean at Task 2.
- The mlir-aie sibling repo (`/home/triple/npu-work/mlir-aie`, branch
  `xdna-emu-cycle-budget`) is clean of the flawed corner-source geometry.json.
- SDD ledger: `.superpowers/sdd/progress.md` (gitignored) — Task 1 complete,
  Task 2 complete (Minors), Task 3 BLOCKED (now resolved by this doc).

## 5. Pointers and guardrails

**Depth docs:** `docs/superpowers/2026-07-01-sp5b-r3b-NEXT-STEPS.md` (full
history, R3b build detail Sec.C, verified register anchors Sec.C —
`Performance_Control0`@0x31500, `Performance_Counter0`@0x31520,
`Timer_Control`@0x34000, `BROADCAST_15`=122, `USER_EVENT_2`=126; note
`npu1_4col` does NOT exist, use `npu1_3col`); the two rev3 design docs; the two
findings named above; `docs/trace/cross-domain-skew-limit.md`.

**Live memory:** `memory/project_timer_sync_arc_inflight.md` (whole #140 arc).

**Run the core:** `python3 -m pytest tools/test_skew_*.py -q`; `cargo test --lib`
(seam unset = byte-identical). Rebuild the FFI `.so`
(`cargo build -p xdna-emu-ffi`) before any plugin/gate use — never bare
`cargo build`.

**HW-safety (non-negotiable):** HW is the cheap oracle — err toward more HW runs.
Never run two HW suites concurrently. No `xrt-smi` during a HW run (segfaults
this devbox). `pkexec` not `sudo`. TDR recovery:
`pkexec sh -c 'modprobe -r amdxdna && modprobe amdxdna'`. Reboot is handed to the
user, never self-run. Never pipe build/test through `tail`/`grep`. Never put
persistent work in `/tmp` (reboots wipe it).
