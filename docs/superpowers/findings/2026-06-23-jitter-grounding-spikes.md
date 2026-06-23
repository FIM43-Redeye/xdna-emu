# Jitter-grounding spikes -- measured foundation for #140's grounding rule

**Date:** 2026-06-23
**Reproduce:** `build/experiments/spike-jitter-grounding/{spike.py,spike2.py}`
(gitignored; HW, add_one_using_dma, chess). Run with
`env -u XDNA_EMU PYTHONPATH=tools <ironenv>/python build/.../spikeN.py`.

These two HW spikes settled the model for the explicit-jitter-robust-grounding
design (`docs/superpowers/specs/2026-06-23-explicit-jitter-robust-grounding-design.md`)
*before* committing the plan. They overturned the design's first premise twice.

## Why we spiked

An Opus adversarial review of the first design draft returned NEEDS-REDESIGN with
3 verified CRITICALs, the central one being: cross-run offsets subtract two
**unsynchronized per-tile timers**, so "Q=0 / single-run exactness" was unsound.
Rather than redesign on speculation, we measured.

## Spike 1 (10 runs): the jitter model was inverted

The draft assumed the deterministic part of the through-core span is the
post-delivery remainder (bracket at `MM2S_FINISHED`) and the jitter is delivery.
Measurement showed the opposite:

- `MM2S_START -> S2MM_START` (both shim, one domain) was stable ~935 in 9/10 runs
  while the FINISHED/delivery events scattered ~3700 cycles. The deterministic
  thing is the **issue cadence**, not the delivery; FINISHED **is** the jitter.
- One run's whole shim pipeline issued +3000 cycles late vs the core PERF_CNT_2,
  yet its within-shim span was still 937 -> **cross-domain timer skew is real but
  cancels within a domain** (CRITICAL-1 scoped: cross-tile only; through-core,
  being within-shim, is unaffected).
- One within-domain span outlier remained (890 vs 935): the open question.

## Spike 2 (20 runs, core lock/stall/contention signature): DECISIVE

Added core `INSTR_LOCK_ACQUIRE_REQ/RELEASE_REQ`, `ACTIVE`, `LOCK_STALL`,
`MEMORY_STALL`, `LOCK_ACCESS_TO_UNAVAILABLE`, `GROUP_STALL` to read each run's
structure.

**The deterministic unit is a within-domain milestone-bounded segment:**

```
core INSTR_LOCK_ACQUIRE_REQ -> INSTR_LOCK_RELEASE_REQ = 22 cycles
ALL 20 runs, range 0 -- including run_01 (span 3354), run_12 (acquire@5029),
run_16 (span 900).
```

- Lock STRUCTURE identical every run (15 acquires / 16 releases / 0
  lock-unavailable / 0 memory-stall): the span outliers are NOT an explainable
  kernel-level structural class. They are delivery jitter on the WAIT (when the
  acquire fires: absolute 990..5029) plus trace-load perturbation.
- **Observer effect:** spike 1 (1 core trace event) -> span stable ~935; spike 2
  (7 core events incl. high-frequency ACTIVE ~132x / GROUP_STALL ~149x) -> span
  scattered 900..3354. Heavy trace perturbs timing. BUT the compute segment held
  exactly 22 under heavy trace -> **within-domain exact-agreement grounding is
  observer-effect-robust**; low-frequency event selection is a refinement, not a
  correctness requirement.

## The model the plan is built on

- **Deterministic, cycle-accurate, Q=0 literally:** within-timer-domain (same
  tile) segments whose per-run offset agrees EXACTLY (range 0). Exact agreement is
  equality, not statistics -- it both classifies and verifies magnitude.
- **Jitter = named gaps:** within-domain offsets that do NOT agree exactly (they
  bundle a wait) and all cross-domain offsets. Existence + orientation only.
- **Through-core deliverable:** `gap + (exact 22-cycle compute) + gap`, never one
  deterministic number for the bundled 935 span.
- **Deferred (own plans):** cross-domain timer-sync (BROADCAST_15 timer-reset);
  active low-frequency event selection.

## Opus review findings, re-adjudicated against measurement

- CRITICAL-1 (timer skew): confirmed real; scoped to cross-tile, cancels
  within-domain -> does not block within-shim through-core.
- CRITICAL-2 (decoder Q justification stale): upheld -> dropped from the design.
- CRITICAL-3 (classify_edge DDR-source absent from dump): premise moot -- the real
  classifier is same-domain + exact-agreement, derived from the event key.
- IMPORTANT-5 (single-run can't verify magnitude): dissolved -- grounding uses
  exact cross-run agreement (range 0, proven), which verifies magnitude without
  statistics. "Single-run" relaxes to "exact agreement across the run set."
