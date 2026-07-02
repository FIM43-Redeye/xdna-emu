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

**(a) The theorem is correct, and so is the coefficient line.** The load-bearing
algebra holds: `max(dx,0) − max(−dx,0) = dx` gives `e − w = tile.col − src.col`
unconditionally, so `a_hE − a_hW = s1.col − s2.col` is a per-pair constant, E/W
columns collapse after referencing (likewise N/S), rank ceiling 3, and corner
sources add a third null direction → rank 2. The Fable-agent adjudication claimed
the finding's illustrative line `a_turn = 5*a_hW + 2*a_vS − 10` was wrong, but
that was a false positive — it derived the turn coefficient under a different
coordinate/hop convention than `r3b_observe.py` actually uses. Re-verified
against the real conventions (brute force over all 18 `npu1_3col` tiles,
`test_skew_r3b_identifiability.py`): the line is exact, and the referenced
enriched design matrix is rank 2. No coefficient fix needed; the guard test now
locks this in.

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

## 3. Execution — the R3b rollback (DONE, code; residual, docs)

The software rollback landed on branch `feat/sp5b-r3b-isotropy-rollback`
(commit `77180706`), full skew suite **33 passed**. It supersedes the abandoned
`feat/sp5b-r3b-pc-software` (Task 1's 5-param enrichment is not merged anywhere).

**Solver / code — DONE:**
1. No solver change was needed: master's `r3b_extract.py` was already the correct
   `{d_h, d_v}` fit (Task 1 had enriched it; that enrichment is simply not
   carried forward). `dn_h`/`dn_v` are the change in *absolute* hop count between
   floods (`|X.col−s2.col| − |X.col−s1.col|`), not signed net displacement.
2. `r3b_observe.py` ported and adjusted to emit `{dn_h, dn_v, r}` instead of the
   per-direction `a_*` coefficients; `test_skew_r3b_observe.py` rewritten with an
   observe→extract round-trip recovering a known `{d_h, d_v}`.
3. `test_skew_r3b_identifiability.py` (new) locks the theorem: E/W and N/S columns
   are constant offsets, the enriched 5-col design on corner sources is rank 2,
   and cross-axis `{d_h, d_v}` is rank 2 (identifiable).
4. `test_skew_r1_diff_extract.py`: two-sided mid-column R1 spine falsifies
   vertical anisotropy (`d_vN ≠ d_vS` → residual grows), with isotropic and
   one-sided-blindness controls.
5. **Deferred (YAGNI):** the optional `d_turn` (cross-axis additivity) column —
   identifiable only with non-corner sources, which don't exist yet. Add it when
   that geometry is built; the mechanism is documented and identifiable.

**Doc rollback — residual (light-touch; authoritative corrected content lives in
this doc + the finding):**
- Finding `docs/superpowers/findings/2026-07-01-r3b-two-source-identifiability-limit.md`:
  add the ruling (Option 1, vertical reallocated to R1, horizontal isotropy
  assumed) and the two valid sharpenings (interval methods can't recover
  within-axis direction at *any* source count; vertical check → two-sided R1).
  Do **not** touch the coefficient line — it is verified correct (Sec.2a).
- The two rev3 design docs, the audit finding Q1, and the R3b-PC plan carry
  errata pointers to the finding: upgrade each from "OPEN decision" to
  "RESOLVED: Option 1 + R1 reallocation." Full section rewrites of the specs'
  Sec.5 kernel-construction prose are best folded into the actual R3b build, not
  done speculatively now — the errata pointer keeps them honest in the interim.

**Then** the R3b build itself (design Sec.5, still valid except the
enriched-geometry point) can proceed: the hand-authored MLIR flood kernel, the
counter-readback host (the hard part is readout, not config), and the
`r3b_pc_gate.sh` gate — consuming the `{dn_h, dn_v}` bridge already built. SP-5b
still produces *no number* — correctness is SP-5c's human causal-vs-HW gate.

## 4. Current committed state (verify with `git log`, not memory)

- `master` carries this doc + the identifiability finding + the Option-1 code
  rollback (once `feat/sp5b-r3b-isotropy-rollback` merges) + errata pointers on
  the audit finding, both rev3 design docs, and the plan header.
- Branch `feat/sp5b-r3b-isotropy-rollback` (commit `77180706`): the Option-1
  rollback — `{d_h, d_v}` observe bridge + identifiability guard + two-sided R1
  spine test. Full skew suite 33 passed.
- Branch `feat/sp5b-r3b-pc-software` (pushed): **SUPERSEDED** — held Task 1's
  5-param enrichment (abandoned) and the original Task-2 observe (reworked in the
  rollback branch). Not merged; keep for history or delete.
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
