# Trace-Accuracy Corpus Re-measure — Design

**Date:** 2026-06-26
**Status:** Approved (design); pending implementation plan
**Issue:** #140 (true-accuracy arc)

## Motivation

The emulator's last full trace-accuracy tally was "15 CLEAN / 125 DIVERGE"
across the npu1 bridge corpus (~70 kernels x 2 compilers ~= 140 `(kernel,
compiler)` points). That number is **stale and untrustworthy** for two reasons:

1. **The encoder layer has since been closed.** The cont.6 fixes
   (`70b25060`, `edf72f68`, `52649b88`, `924ad3c0`, `8b82dabf`) repaired four
   trace encode/decode/model bugs; every memtile `PORT_RUNNING` port now decodes
   to exactly `words` (the `sum(PORT_RUNNING)==words` oracle holds). Many of the
   125 DIVERGEs were almost certainly *encoding artifacts* that no longer exist.
2. **The capture-load canary now exists.** We proved (capture-load-sensitivity.md)
   that host load contaminates within-domain cycle measurements, and built an
   engine that *warns* on unaccounted within-domain nondeterminism instead of
   swallowing it. A tally taken on a loaded host records load artifacts as real
   divergences; we now have the means to certify a capture session clean.

So our **true accuracy baseline is unknown.** Before attacking any specific
divergence (e.g. the one documented open model-timing gap, the shim->memtile
relay-fill burst structure), we re-measure honestly to learn where we actually
stand. Triage, then target.

## Goal / Definition of Done

A fresh, **certified-clean** CLEAN/DIVERGE baseline for the full npu1 bridge
corpus (both compilers), plus a ranked list of the *real* divergences — written
to a dated baseline doc and feeding the next accuracy goal.

"Certified-clean" means: the capture session is bracketed by a canary witness
that confirms a known cycle-exact within-domain span stayed cycle-exact
throughout. If it did not, the tally is void.

## Non-Goals

- **Fixing divergences.** This task *measures*; closing the real divergences is
  the next goal, scoped from this baseline's ranked list.
- **Per-kernel canary integration** (Approach 3). Wiring the inference-engine
  canary into the bridge comparator so every DIVERGE is auto-classified
  within-domain / structural / load is deferred — revisit only if the baseline
  shows many within-domain DIVERGEs worth auto-triaging.
- **Merging Chess compile sessions.** The Chess RAM wall (~5.5 GB/job, ~48 GB
  ceiling) caps compile parallelism. A shared/merged Chess compile session would
  lift that cap, but it is its own project. This campaign lives within the
  existing `-j4` default clamp.

## Components

### 1. Canary witness — `tools/canary-witness.py` (new)

A cleanliness certifier built on the canary we just shipped.

**Core verdict logic (offline-testable):** given the sentinel kernel's
multi-run capture dirs, ground the known within-domain core-lock span and decide
clean vs dirty:

- Sentinel: `add_one_using_dma` (chess), the core-lock span
  `INSTR_LOCK_ACQUIRE_REQ -> INSTR_LOCK_RELEASE_REQ` on the compute tile — the
  exact span proven cycle-exact-under-clean / flicker-under-load.
- Clean iff `ground_edge(run_dirs, REL, ACQ)` returns a `Segment` (cycle-exact,
  range 0). Dirty iff it returns a `Gap` with reason `within_domain_nonexact`
  (the unaccounted anomaly). Equivalently, at the engine level: the sentinel's
  `warnings` list is empty **and** the core-lock segment is present (present
  proves the span was actually captured and grounded, not merely absent).

**Live front-end (HW-gated, thin):** capture the sentinel on HW (reuse
`inference/run_experiment.py` / `hw_instrument.py`), then apply the core verdict.

**Why this is honest:** the verdict is the canary's own rule (range <= Q == 0,
threshold-free), not a new statistical test. The witness does not *classify*
load vs HW — it only flags "this session was not clean"; a human re-runs.

### 2. Corpus run — existing tooling, no new code

`scripts/emu-bridge-test.sh --trace` over the full npu1 set, both compilers, on
the quiet box. Per `(kernel, compiler)` it emits:

- `${safe}.${compiler}.trace.summary` — `CLEAN` / `DIVERGE` / `ERROR` (the
  authoritative `TRACE_VERDICT` token, `src/trace/compare.rs:2228`).
- `${safe}.${compiler}.trace.log` — full comparator report with the edge/level
  `diverged` + `count_mismatch` magnitudes.

Chess compile parallelism stays at the script default (`-j4`, ~22 GB); never
overridden upward (RAM wall).

### 3. Harvester + baseline doc — `tools/accuracy-baseline.py` (new)

Reads a `build/bridge-test-results/<date>/` dir and produces a structured
baseline:

- Per-`(kernel, compiler)` verdict from `.trace.summary`.
- Divergence magnitude parsed from `.trace.log` (sum of edge/level diverged +
  count_mismatch) for ranking.
- Ranked DIVERGE list, cross-referenced against `docs/known-fidelity-gaps.md`
  (so a DIVERGE already explained by a documented gap is tagged, and the
  unexplained ones surface as new targets).
- Emits a dated markdown: `docs/trace/accuracy-baseline-YYYYMMDD.md`.

Offline-testable against fixture result dirs (synthetic `.trace.summary` +
`.trace.log` files).

## Data Flow

```
witness(start) --clean?--> [no] -> abort, host not quiet
       | clean
       v
emu-bridge-test.sh --trace  (full corpus, both compilers, quiet box)
       |
       v
harvest .trace.summary + .trace.log
       |
       v
witness(end) --clean?--> [no] -> VOID tally, flag for re-run
       | clean
       v
accuracy-baseline.py -> docs/trace/accuracy-baseline-YYYYMMDD.md
                        + ranked real-divergence target list
```

## Build Order

1. **Witness verdict logic** (offline, TDD): test the core check against the
   persisted fixtures — `build/experiments/lock-jitter-clean` must certify
   **clean** (Segment), `build/experiments/lock-jitter-loaded` must flag
   **dirty** (within_domain_nonexact Gap). These are real captures from the
   load-sensitivity proof, so the test exercises the genuine signal.
2. **Harvester** (offline, TDD): test tally + ranking + known-gap
   cross-reference against fixture result dirs.
3. **Witness live front-end** (HW-gated, code-only in subagent; controller runs).
4. **HW campaign** (controller/Maya): one full corpus run after the code is
   green. Expensive (~30–120 min); no progress re-runs.
5. **Generate baseline doc + ranked list** from the campaign results.

## Error Handling / HW Realities

- **NPU TDR mid-run:** recover via `modprobe -r amdxdna && modprobe amdxdna`
  (one `pkexec`); mark affected kernels `ERROR`; re-run them isolated. Never
  trust a verdict straddling a TDR. Smoke-test recovery with `xrt-smi validate`,
  not a bridge run.
- **Cleanliness is bracketed, not continuous.** A transient load spike *between*
  the start and end witnesses could slip through undetected. That is the
  acknowledged limit of "we can try it"; the dedicated mini-PC removes it later
  by construction.
- **Box must stay quiet:** no gaming, no second HW suite (ISA/bridge), no other
  Claude HW session, no `xrt-smi` during the run (segfaults). The witness
  certifies quiet post-hoc; these are the operational preconditions.
- **Chess RAM wall:** default `-j4` clamp only; do not raise.

## Testing Strategy

This is a measurement campaign, so the **witness is itself the test of capture
validity**. The two new scripts are TDD'd offline:

- Witness verdict logic — against the `lock-jitter-{clean,loaded}` fixtures
  (clean -> Segment -> certified; loaded -> within_domain_nonexact -> flagged).
- Harvester — against synthetic fixture result dirs (known verdict mix -> known
  tally + ranking).

No Rust changes; pure Python (`tools/`). `cargo test --lib` not required.

## Deliverables

- `tools/canary-witness.py` (+ offline tests)
- `tools/accuracy-baseline.py` (+ offline tests)
- `docs/trace/accuracy-baseline-YYYYMMDD.md` (generated from the campaign)
- A ranked real-divergence target list that scopes the *next* accuracy goal

## Future Work (noted, out of scope)

- **Merge/share Chess compile sessions** to lift the RAM-wall parallelism cap.
- **Per-kernel canary integration** (Approach 3) if the baseline warrants it.
- **Mini-PC dedicated capture box** — removes the bracketed-cleanliness limit by
  running captures on an idle host by construction.
