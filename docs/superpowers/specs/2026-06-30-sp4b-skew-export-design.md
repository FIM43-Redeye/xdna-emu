# SP-4b -- Engine skew-export + causal decomposition (design)

**Arc:** #140 timer-sync faithful-broadcast
([`2026-06-28-timer-sync-faithful-broadcast-arc.md`](2026-06-28-timer-sync-faithful-broadcast-arc.md) §4, SP-4b).
**Depends on:** SP-1 (landed), SP-3 (landed). **Not** SP-2.
**Status:** design approved 2026-06-30; awaiting adversarial review + implementation plan.

---

## 1. What SP-4b delivers

SP-4b is the engine-facing half of SP-4. It adds a **new grounded quantity** to the
trace inference engine: `causal_offset = raw_offset − skew(A,B) = Δwall`, the
physical-separation cycle count between two events in *different* timer domains,
distinct from the existing `reproduction_offset` (the raw fused `Δwall + skew`
that reproduces the trace).

It does this by:

1. **Exporting the emulator's modeled per-module timer-reset arrival (`origin_D`)**
   as a sidecar artifact on the existing trace run, plus an explicit `calibrated`
   flag.
2. **Solving `skew(A,B)` in the engine** as a trivial `origin_D[B] − origin_D[A]`
   lookup, and emitting `causal_offset` -- but **only when the broadcast model is
   calibrated**.
3. **Relaxing the `timeline.py` cross-domain fail-loud guard** so a cross-domain
   frame may carry a cycle value *iff* it has a calibrated `causal_offset`.
4. **Amending skew-limit doc Sec.9** to permit a model-derived cross-domain
   causal segment under calibration.

Until SP-5 measures the per-hop constants, the model is **uncalibrated**:
`skew(A,B) = 0`, `causal_offset` is **withheld** (`None`), and runtime behavior is
byte-identical to today. The full path is built and unit-tested via *synthetic*
non-zero constants; it flips live the moment SP-5 sets `calibrated = true`.

### Non-goals

- **No SP-2 dependency.** SP-4b reads `origin_D` (modeled) and the
  PERF_CNT_2-anchored wall time directly, not the encoded trace's reconciled
  origins. Trace-origin reconciliation (SP-2) is irrelevant to this decomposition.
- **No direction-diverse rank-2 `d_h`/`d_v` solve in the engine.** See §6 -- this
  folds into SP-5, because exporting finished `origin_D` already bakes in the
  directional sum.
- **Async DDR/NoC (`is_async_cdc`) pairs stay existence-only gaps**, always. Their
  skew is non-deterministic; out of scope by design.

---

## 2. Background: the boundary the spec glossed

The arc spec (§4) says SP-4b's solver "reads `origin_D` (SP-1) and `W_sim` ... and
exports skew to the inference engine." A touchpoint audit (2026-06-30) found those
live on opposite sides of a **Rust→Python boundary that does not yet exist**:

- `origin_D`, `d_h`/`d_v`, and the flood-delay constants are **SP-1 Rust artifacts**
  (`src/device/state/effects.rs`, `crates/xdna-archspec/src/model_builder.rs`). The
  constants are **all 0** today (placeholders; real values are SP-5).
- The Python engine (`tools/inference/`) sees **only captured trace timestamps**
  (PERF_CNT_2-anchored `soc`). There is no `W_sim` field and no skew code anywhere
  in the engine -- only prose in docstrings.

So the decomposition spans a boundary that has to be built. The clean fact that
makes it cheap: **`skew(A,B)` is a deterministic function of the broadcast model**,
fully known emulator-side --

```
tau_X(W)   = (W − T0) − origin_D(X)              # a tile's reset-synced timer reading
raw_offset = soc(x) − soc(y)
           = (W_x − W_y) − (origin_D(A) − origin_D(B))
           = Δwall − (origin_D(A) − origin_D(B))
=>  Δwall  = raw_offset + (origin_D(A) − origin_D(B))
           = raw_offset − skew(A,B)               with  skew(A,B) := origin_D(B) − origin_D(A)
```

So the engine never needs to *infer* skew. The emulator exports `origin_D`; the
engine subtracts. (Sign convention: `skew(A,B) = origin_D(B) − origin_D(A)`, chosen
so `causal_offset = raw − skew = Δwall`. Pinned by unit test against this
derivation.)

---

## 3. Architecture

```
EMULATOR (Rust)                         INFERENCE ENGINE (Python)
---------------                         -------------------------
effects.rs broadcast_origin_d           loader_model.py  (NEW)
  per-module origin_D  ───────────────►   reads sidecar -> MODEL_DERIVED facts:
model_builder.rs                            origin_d(module, D)
  BroadcastTiming.calibrated (NEW)          skew_calibrated(bool)
        │                                         │
        ▼                                         ▼
  origin_d.json sidecar               grounding.ground_edge  (cross-domain branch)
  written on the existing               skew = origin_D[B] − origin_D[A]
  trace run, keyed col|row|module       causal_offset = (raw − skew) if calibrated else None
                                        Gap += {skew, causal_offset, skew_calibrated}
                                                  │
                                                  ▼
                                        timeline.py guard (relaxed, dormant pre-SP-5)
                                        rules.py / facts.py / engine.py  (tuple + report)
```

The emulator's job ends at "write an honest sidecar." The engine's job is to
consume it **without ever laundering modeled data into a measured claim**.

---

## 4. The Rust export

### 4a. `calibrated` flag on `BroadcastTiming`

Add a `calibrated: bool` field to `BroadcastTiming`
(`crates/xdna-archspec/src/types.rs:1353`), set `false` in `model_builder.rs:270`.
It is an **explicit** flag, **not** inferred from "constants != 0" -- a genuinely
measured `d_v = 0` at SP-5 must not read as uncalibrated. SP-5 flips it to `true`
when it writes real constants. This is the single source of truth for the engine's
`skew_calibrated`.

This is an SP-1-adjacent addition: SP-1 left the constants as placeholders without
marking them *as* placeholders, and the keystone needs that mark.

### 4b. Per-module `origin_D` export

`broadcast_origin_d` (`effects.rs:443`) already produces `(col, row, origin_D)` per
reached tile via a Dijkstra min-cost wavefront; `:582` splits it into
`core_delay`/`mem_delay` with the intra-tile offsets. SP-4b serializes that table,
keyed by the engine's domain identity `col|row|module` (matching the `_domain_of`
prefix the guard uses), into a sidecar:

```json
{
  "calibrated": false,
  "modules": { "1|2|core": 0, "1|2|mem": 0, "1|1|mem": 0, ... }
}
```

The table is a pure function of the configured array + flood source: deterministic
and run-invariant (it does not need the 8-batch trace sweep).

### 4c. Hook point

The export rides the existing trace-emission path -- the run that already writes
`trace.events.json` also writes `origin_d.json` beside it. One EMU run, two
artifacts, no new entry point.

---

## 5. The Python engine

Throughline: **modeled data never launders into a measured claim**, and **pre-SP-5
runtime behavior is byte-identical to today** (so the existing trace-sweep
regression gate cannot regress).

### 5a. Third provenance category + model-derived loader

Today the engine has two leaf sources: `loader.py` (measured `fired` facts) and
`ledger.py` (structural "binary contains" facts). The sidecar is neither -- it is
**model-derived**. Add:

- A new `MODEL_DERIVED` provenance kind in `facts.py`. `provenance_ok` accepts it
  but keeps it distinguishable, so any causal claim built on it is permanently
  traceable to "modeled" -- never silently upgraded to "measured."
- A new loader (`loader_model.py`) emitting `origin_d(module, D)` and
  `skew_calibrated(bool)` facts under `MODEL_DERIVED`.

### 5b. Decomposition in `grounding.ground_edge` (cross-domain branch, `:105`)

Where `raw = offset_exact(...)` is computed today:

```python
skew = origin_D[B] − origin_D[A]                 # sign per §2 derivation
causal_offset = (raw − skew) if skew_calibrated else None
```

`Gap` (`grounding.py:79`) gains three fields -- `skew`, `causal_offset`,
`skew_calibrated` (default `False`) -- with `reproduction_offset` **untouched**. The
`is_async_cdc` early-out (`:98`) stays existence-only: those pairs never receive a
`causal_offset`, even when calibrated.

### 5c. Guard relaxation, dormant until SP-5 (`timeline.py:410`)

Today: `assert` a cluster frame is single-domain; raise `AssertionError` on
cross-domain. Relaxed: a cross-domain frame may emit a cycle value **iff** it
carries a calibrated `causal_offset` (and is not async-CDC).

Pre-SP-5, `calibrated == false` -> `causal_offset is None` everywhere -> the guard
fires **exactly as it does now**. The relaxation is real code but inert until
silicon; pre-SP-5 behavior is provably unchanged. `CrossTrackEdge`
(`timeline.py:148`) gains a `causal_offset` field beside its existing
`reproduction_offset`.

### 5d. Fact tuple + report surfacing

The derives-fact tuple at `rules.py:57` extends with the new trailing args; the
existing `<5`/`<6`-arg backward-compat guards in `facts.py` get a `<7` sibling, so
old facts still parse. `engine.py:63` and the convergence report surface
`causal_offset` beside `reproduction_offset`, explicitly labeled and shown as
**withheld (uncalibrated)** when `None` -- the report never reads as if a causal
cycle exists when one does not.

---

## 6. Scope: the three spec bullets under this design

The arc spec lists three SP-4b bullets as co-equal. Exporting finished `origin_D`
(rather than solving in the engine) collapses two of them:

| Spec bullet | Disposition | Rationale |
|---|---|---|
| **P3 -- export skew, add `causal_offset`, relax guard, amend Sec.9** | **IN -- the build** (§4-5, §7). Gated on `skew_calibrated`. | The concrete engine deliverable. |
| **P2 -- direction-diverse rank-2 solve for separate `d_h`/`d_v`** | **DEFER to SP-5** (note here). | Exporting finished `origin_D` already bakes in `n_h·d_h + n_v·d_v` (the wavefront did the directional sum). The engine never solves for `d_h`/`d_v`; that solve is a *silicon measurement* technique, and SP-5 already owns it (spec route-3b: two-source broadcast trigger). Round-3 problem 2 is sidestepped at the source, not re-solved engine-side. |
| **P1 -- in-domain round-trip coupling-latency gate** | **IN, as a test** (§7.3), not new engine code. | The engine already proves within-domain segments cycle-exact (`Segment`). P1 = a targeted regression test asserting SP-3's same-domain A→B→A round-trip reproduces Q=0-exact, so that at SP-5 the cross-domain residual is trustworthy as *pure* skew. |

**Deliberate deferred gap:** deferring P2 means SP-4b never validates `origin_D`'s
directional decomposition against a real direction-diverse *measurement* -- it
trusts SP-1's wavefront. That validation cannot happen until SP-5 has silicon
`d_h`/`d_v` to check against (pre-SP-5 everything is zero, nothing to validate).
This is correct sequencing, recorded here so the deferral is deliberate, not an
oversight.

---

## 7. Testing (TDD)

The key testability move: **synthetic non-zero constants** exercise the full live
path without silicon.

1. **Rust** -- `origin_D` table export keyed `col|row|module`; `calibrated` flag
   plumbs end-to-end; sidecar round-trips.
2. **Python (core):**
   - *Skew sign/value:* inject synthetic table (calibrated, non-zero) -> assert
     `skew = origin_D[B] − origin_D[A]` and `causal_offset = raw − skew = Δwall`.
   - *Withhold gate:* `calibrated = false` -> `causal_offset is None` for every
     cross-domain pair; output identical to pre-SP-4b.
   - *async-CDC exclusion:* async pairs never get a `causal_offset`, even calibrated.
   - *Guard:* a calibrated cross-domain frame is permitted (no raise); an
     uncalibrated one raises exactly as today.
   - *Provenance:* `causal_offset` traces to `MODEL_DERIVED`; old `<6`-arg
     derives-facts still parse.
3. **P1 round-trip gate** -- run SP-3's kernel through the engine; assert the
   same-domain A→B→A round-trip reproduces Q=0-exact (existing `Segment` machinery;
   new targeted assertion).
4. **Regression** -- the existing trace-sweep gate stays green (load-bearing
   "reproduction didn't break" check); `cargo test --lib` green.

---

## 8. Epistemic framing (skew-limit Sec.9 amendment)

Sec.9 today: *"Engine: record the exact raw cross-domain offset as a
reproduction-target annotation. Never emit a cross-domain causal segment."*
SP-4b amends only the Engine bullet:

> Engine: record the exact raw cross-domain offset as `reproduction_offset`
> (**unchanged**). *Additionally*, when a **calibrated** broadcast model is
> available, emit the decomposed `causal_offset = raw − skew(A,B)` as a
> **model-derived** causal cycle. Until calibrated, withhold it (gap-only, status
> quo). Async DDR/NoC egress stays gap-only **always**.

The amendment stays honest about *why* this is now allowed: Sec.5-6 (the
trace-in-isolation underdetermination -- three walls) is **untouched and still
true**. The decomposition becomes possible only because the emulator supplies a
*verified forward model* (Sec.7's "emulator-plus-verification" tier), not because
the trace gained information. The trace alone still cannot split the offset; the
calibrated emulator can. `causal_offset` is tagged model-derived and
cross-references Sec.7.

---

## 9. Risks and open questions

- **Sign convention.** `skew(A,B) = origin_D(B) − origin_D(A)`. Easy to flip;
  pinned by a unit test against the §2 derivation. Verify once on a synthetic case
  with known `Δwall`.
- **Domain-key alignment.** The sidecar's `col|row|module` keys must match exactly
  what `_domain_of` produces in `timeline.py`. A key mismatch would silently leave
  `origin_D` lookups empty -> `causal_offset` always `None` even when calibrated.
  Test asserts every traced cross-domain module resolves to a sidecar entry.
- **`origin_D` for untraced modules.** The export should cover every module that can
  appear in a cross-domain pair; a missing entry must fail loud (calibrated) rather
  than silently withhold.
- **Backward-compat fact arity.** The `<7`-arg guard must not mis-parse existing
  6-arg derives-facts. Covered by a regression test.
