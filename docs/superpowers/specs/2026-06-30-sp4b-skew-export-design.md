# SP-4b -- Engine skew-export + causal decomposition (design)

**Arc:** #140 timer-sync faithful-broadcast
([`2026-06-28-timer-sync-faithful-broadcast-arc.md`](2026-06-28-timer-sync-faithful-broadcast-arc.md) §4, SP-4b).
**Depends on:** SP-1 (landed), SP-3 (landed). **Not** SP-2.
**Status:** design approved 2026-06-30; revised after adversarial review (7 findings
folded in); awaiting user-review gate + implementation plan.

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
2. **Solving `skew(A,B)` in the engine** as a trivial `origin_D` subtraction, and
   emitting `causal_offset` as a **separate, model-grounded fact** -- but **only
   when the broadcast model is calibrated**.
3. **Adding `causal_offset` additively on the cross-domain edge path** (`Gap` ->
   `CrossTrackEdge` -> renderer), where the renderer presents it as a model-derived
   cross-track quantity, never a local tile cycle. No runtime guard is relaxed (see
   §5c -- the arc's "relax the guard" phrasing was imprecise).
4. **Amending skew-limit doc Sec.9** to permit a model-derived cross-domain causal
   segment under calibration.

Until SP-5 measures the per-hop constants, the model is **uncalibrated**:
`skew(A,B) = 0`, no `causal_offset` fact is emitted, and all existing engine
outputs are byte-identical to today. The full path is built and unit-tested via
*synthetic* non-zero constants; it flips live the moment SP-5 sets
`calibrated = true`.

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

## 2. Background and the sign derivation

The arc spec (§4) says SP-4b's solver "reads `origin_D` (SP-1) and `W_sim` ... and
exports skew to the inference engine." A touchpoint audit (2026-06-30) found those
live on opposite sides of a **Rust→Python boundary that does not yet exist**:

- `origin_D`, `d_h`/`d_v`, and the flood-delay constants are **SP-1 Rust artifacts**
  (`src/device/state/effects.rs`, `crates/xdna-archspec/src/model_builder.rs`). The
  constants are **all 0** today (placeholders; real values are SP-5).
- The Python engine (`tools/inference/`) sees **only captured trace timestamps**
  (PERF_CNT_2-anchored `soc`). There is no `W_sim` field and no skew code anywhere
  in the engine -- only prose in docstrings.

The fact that makes the decomposition cheap: **`skew(A,B)` is a deterministic
function of the broadcast model**, fully known emulator-side. Derivation, for two
events `x` (domain A) and `y` (domain B):

```
tau_X(W)   = (W − T0) − origin_D(X)              # a tile's reset-synced timer reading
soc(x)     = (W_x − T0) − origin_D(A)
soc(y)     = (W_y − T0) − origin_D(B)
raw        = soc(x) − soc(y)
           = (W_x − W_y) − (origin_D(A) − origin_D(B))
           = Δwall − (origin_D(A) − origin_D(B))
=>  Δwall  = raw + (origin_D(A) − origin_D(B))
           = raw − skew(A,B)        with  skew(A,B) := origin_D(B) − origin_D(A)
```

So the engine never *infers* skew -- the emulator exports `origin_D`, the engine
subtracts.

### 2a. The A/B ↔ child/parent binding (the flip-prone line)

`ground_edge(run_dirs, child, parent)` computes `raw = offset_exact(child, parent)`,
and `offset_exact(a, b)` returns `a − b` (`verifier.py:59`). So
`raw = soc(child) − soc(parent)` ⇒ `x = child`, `y = parent` ⇒ **A = domain(child),
B = domain(parent)**. The implementation must therefore compute:

```
skew = origin_D[domain(parent)] − origin_D[domain(child)]
causal_offset = raw − skew = Δwall
```

This binding is the single most error-prone line in SP-4b. It is pinned by a test
that constructs **asymmetric synthetic `origin_D` and a known nonzero `Δwall`** and
asserts `causal_offset == Δwall` (§7) -- *not* by asserting
`skew == origin_D[B] − origin_D[A]`, which is circular (it restates the
implementation's own A/B binding and passes regardless of the physical sign).

---

## 3. Architecture

```
EMULATOR (Rust)                         INFERENCE ENGINE (Python)
---------------                         -------------------------
effects.rs broadcast_origin_d           loader_model.py  (NEW)
  per-module core_delay/mem_delay ─────►   reads sidecar -> MODEL_DERIVED facts:
  (= origin_D; NOT *_target)                 origin_d(domain_key, D)   [cites sidecar]
model_builder.rs                            skew_calibrated(bool)      [cites sidecar]
  BroadcastTiming.calibrated (NEW)                │
        │                                         ▼
        ▼                              grounding.ground_edge  (cross-domain branch)
  origin_d.json sidecar                 skew = origin_D[parent_dom] − origin_D[child_dom]
  written on the existing               causal_offset = (raw − skew) if calibrated else None
  trace run, keyed                      Gap += {causal_offset}   (reproduction_offset unchanged)
  col|row|<pkt_type>                              │
  (numeric pkt_type)                              ▼
                                        weave -> CrossTrackEdge += causal_offset
                                        rules.py: SEPARATE causal fact w/ MODEL_DERIVED premises
                                        facts.py: provenance_ok gains a MODEL_DERIVED branch
                                        renderer: prints causal_offset only when present, tagged
```

The emulator's job ends at "write an honest sidecar." The engine's job is to
consume it **without ever laundering modeled data into a measured claim** -- which,
after review, requires the model dependency to appear in the causal fact's
*premises*, not merely its data args (§5a).

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

`broadcast_origin_d` (`effects.rs:443`) produces `(col, row, origin_D)` per reached
tile via a Dijkstra min-cost wavefront; `:582-584` splits it into
`core_delay = origin_d + core_off` / `mem_delay = origin_d + mem_off` -- **these are
the physical `origin_D` values to export.**

> **Export-quantity caution.** Export `core_delay`/`mem_delay` (`:583-584`), **not**
> the adjacent `core_target`/`mem_target` (`:585-586` = `max_delay − delay`), which
> are the *complement* of `origin_D` (fed to the trace units at `:606-607`). Grabbing
> the `*_target` neighbor inverts the skew sign undetectably -- the engine-side
> synthetic test would not catch an export-side flip. A Rust-side test asserts the
> exported value equals `origin_d + intra_off`, not its complement.

**Keying.** The sidecar keys each module by the engine's domain identity
`col|row|<pkt_type>`, where **`pkt_type` is the numeric trace packet-type code**
(core/mem/memtile/shim), matching exactly what `same_domain` / `_domain_of`
(`grounding.py:51`, `timeline.py`) derive via `key.rsplit("|", 1)[0]`. The anchor
`1|2|0|PERF_CNT_2` confirms the form: col=1, row=2, pkt_type=0 (core). **Not**
`col|row|core` -- a module-name key would never resolve against the engine's numeric
keys and the live path would be silently dead (review finding 3).

The table must cover **every module type that can appear in a cross-domain pair** --
core, mem, memtile, shim -- not only compute core/mem. A core/mem-only table is
incomplete. Either the Rust side emits numeric `pkt_type` directly, or
`loader_model.py` owns the `module → pkt_type` translation (the exact pkt_type
enumeration is resolved in the plan; `dma-fill-measure.py:35-36` and the anchor give
core=0, mem=1).

```json
{
  "calibrated": false,
  "modules": { "1|2|0": 0, "1|2|1": 0, "1|1|<memtile_pkt>": 0, ... }
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

Throughline: **modeled data never launders into a measured claim**, and **all
existing engine outputs are byte-identical pre-SP-5** (so the trace-sweep
regression gate cannot regress).

### 5a. Third provenance category, as a real premise (not a data arg)

Today the engine has two leaf sources: `loader.py` (measured `fired` facts) and
`ledger.py` (structural "binary contains" facts, cited into `kb.ledger`). The
sidecar is neither -- it is **model-derived**. Add:

- A new `ModelDerived` support kind in `facts.py`, carrying a **citation to the
  model artifact** (the `origin_d.json` sidecar + its `calibrated` flag), mirroring
  how `Structural` cites `kb.ledger`.
- A real branch in `provenance_ok` (`facts.py:108`): a `ModelDerived` leaf is
  accepted **iff** its citation is a registered model source (e.g. `kb.model`),
  exactly paralleling the `Structural`/`kb.ledger` check at `:119-121`. Today
  `provenance_ok` rejects any non-Measured/non-Structural leaf at `:122-123`; this
  branch is a deliberate, audited keystone extension to a third leaf class -- it
  does **not** silently bless model data as measured.
- A new loader (`loader_model.py`) emitting `origin_d(domain_key, D)` and
  `skew_calibrated(bool)` facts under `ModelDerived`, cited to the sidecar.

**The causal fact is a separate fact with model premises.** Review finding 1: the
existing gap `derives` fact (`rules.py:56-58`) carries `reproduction_offset` as a
*trailing data arg* whose premises are only the measured `fired` facts + the
structural placement -- the model dependency is invisible to `leaves()`, so
`provenance_ok` would bless a model-derived number as fully measured. SP-4b must
therefore emit `causal_offset` as its **own** fact whose `Derived` premises
explicitly include the `origin_d(child_dom)`, `origin_d(parent_dom)`, and
`skew_calibrated` `ModelDerived` facts (plus the measured raw). Then the fact's DAG
genuinely bottoms out on model leaves, `leaves()` surfaces the `ModelDerived`
support, and any consumer can detect and label it. `reproduction_offset` stays
exactly where it is, untouched.

### 5b. Decomposition in `grounding.ground_edge` (cross-domain branch, `:105`)

Where `raw = offset_exact(child, parent)` is computed today:

```python
skew = origin_D[parent_dom] − origin_D[child_dom]   # A=child, B=parent (§2a)
causal_offset = (raw − skew) if skew_calibrated else None
```

`Gap` (`grounding.py:74`) gains one field -- `causal_offset: Optional[int] = None` --
with `reproduction_offset` **untouched**. The `is_async_cdc` early-out (`:98`) stays
existence-only: those pairs return before any offset computation, so they receive
neither `reproduction_offset` nor `causal_offset`, calibrated or not.

Lookups use `origin_D[k]` (not `.get`): a **calibrated** model with a missing module
entry must **fail loud**, not silently withhold (review finding 3). Pre-calibration,
`skew_calibrated` is false and the lookup is not reached.

### 5c. No guard relaxation -- the change is additive on the edge path

Review finding 2: the arc spec's "relax the `timeline.py` cross-domain fail-loud
guard" is imprecise. The only runtime guard, `internal_cycles:410`, asserts a
**`ClusterFrame` is single-domain** -- and frames are built per-domain
(`assemble_timeline` partitions keys `by_domain`), so a cross-domain frame **never
reaches it**. It is a fatal-A "wiring slip" protection; relaxing it would weaken
that check while doing nothing for SP-4b.

The actual cross-domain path is the **edge path** `weave → ground_edge → Gap →
CrossTrackEdge` (`timeline.py:548-558`), which has **no runtime guard** -- the
"no cross-domain cycle" invariant is enforced *structurally* by `CrossTrackEdge`
simply lacking a cycle field. SP-4b's change is therefore purely additive:

- `CrossTrackEdge` (`timeline.py:144`) gains `causal_offset: Optional[int]` beside
  its existing `reproduction_offset`.
- `weave` (`:555-557`) is updated to copy `g.causal_offset` into the edge (review
  finding 6 -- the current `weave` copies only `reason` and `reproduction_offset`).
- The **renderer** presents `causal_offset` as a model-derived cross-track quantity
  (clearly tagged), never a local tile cycle -- and **omits the field entirely when
  it is `None`** (review finding 5). Absence already signals "no causal cycle";
  printing "withheld" would change pre-SP-5 render output and break byte-identity.

`internal_cycles:410` is left untouched.

### 5d. Fact tuple + report surfacing

`causal_offset` rides its **own** fact (§5a), not the gap tuple, so the gap
`derives` tuple at `rules.py:57` is unchanged and no arity migration is forced on
existing facts. `engine.py` and the convergence report surface the new causal fact
when present, explicitly tagged model-derived; pre-SP-5 (none emitted) the report is
byte-identical.

### 5e. Precise byte-identity claim

Pre-SP-5 (`calibrated == false`): **all existing engine outputs** -- gap/segment
placements, every `reproduction_offset`, the convergence report's existing lines,
the trace-sweep regression gate -- and the **EMU trace bytes** are byte-identical to
today (the EMU trace because all constants are 0 ⇒ `propagate_broadcasts` floods
with zero delay, `max_delay = 0`, every target 0, exactly as now). The *only*
additions to the KB are inert `ModelDerived` input leaves (which `provenance_ok`'s
new branch accepts via citation) and **zero** causal facts. The byte-identity claim
is scoped to existing outputs + trace bytes -- it is **not** a claim that the raw KB
fact *set* is unchanged (the inert model leaves are new).

---

## 6. Scope: the three spec bullets under this design

The arc spec lists three SP-4b bullets as co-equal. Exporting finished `origin_D`
(rather than solving in the engine) collapses two of them:

| Spec bullet | Disposition | Rationale |
|---|---|---|
| **P3 -- export skew, add `causal_offset`, emit model-grounded causal fact, amend Sec.9** | **IN -- the build** (§4-5, §8). Gated on `skew_calibrated`. | The concrete engine deliverable. |
| **P2 -- direction-diverse rank-2 solve for separate `d_h`/`d_v`** | **DEFER to SP-5** (signed off by Maya 2026-06-30). | Exporting finished `origin_D` already bakes in `n_h·d_h + n_v·d_v` -- `broadcast_origin_d` does min-cost Dijkstra (`effects.rs:486-500`), and the engine only ever needs the scalar `origin_D` per module, never `d_h`/`d_v` separately. Round-3 problem 2 (rank-deficient hop-count fit) cannot arise because the spec exports finished `origin_D` rather than fitting it. The rank-2 *measurement* is SP-5's job (spec route-3b: two-source broadcast trigger). |
| **P1 -- in-domain round-trip coupling-latency gate** | **IN, as a test** (§7.3), not new engine code. | The engine already proves within-domain segments cycle-exact (`Segment`). P1 = a targeted regression test asserting SP-3's same-domain A→B→A round-trip reproduces Q=0-exact, so that at SP-5 the cross-domain residual is trustworthy as *pure* skew. |

**Deliberate deferred gap (explicitly blessed):** the arc (§4 SP-4b) assigns the
"rank-2 solve for separate `d_h`/`d_v`" to SP-4b, so deferring P2 is a real scope
move *out* of SP-4b, not merely a re-derivation -- and it was signed off when this
scope split was presented and approved (2026-06-30). The consequence: SP-4b never
validates `origin_D`'s directional decomposition against a real direction-diverse
*measurement* -- it trusts SP-1's wavefront. That validation cannot happen until
SP-5 has silicon `d_h`/`d_v` to check against (pre-SP-5 everything is zero, nothing
to validate). Correct sequencing, recorded so the deferral is deliberate.

---

## 7. Testing (TDD)

The key testability move: **synthetic non-zero constants** exercise the full live
path without silicon.

1. **Rust** -- `origin_D` table export keyed `col|row|<pkt_type>` covering
   core/mem/memtile/shim; the exported value equals `origin_d + intra_off` (not the
   `*_target` complement, finding 7); `calibrated` flag plumbs end-to-end; sidecar
   round-trips.
2. **Python (core):**
   - *Sign-pinning (the load-bearing one):* construct asymmetric synthetic
     `origin_D` **and** a known nonzero synthetic `Δwall` -> assert
     `causal_offset == Δwall`. Do **not** rely on the circular
     `skew == origin_D[B] − origin_D[A]` assertion (§2a).
   - *Withhold gate:* `calibrated = false` -> **no causal fact emitted**; every
     existing output identical to pre-SP-4b.
   - *async-CDC exclusion:* async pairs get neither offset, even calibrated.
   - *Provenance, both directions:* a calibrated `causal_offset` fact's `leaves()`
     includes a `ModelDerived` leaf and `provenance_ok` accepts it via model
     citation; a `ModelDerived` leaf with an **unregistered** citation makes
     `provenance_ok` return `False` (the keystone still bites).
   - *Domain-key resolution:* every traced cross-domain module resolves to a sidecar
     entry; a calibrated-but-missing entry fails loud.
   - *Edge threading:* `weave` copies `causal_offset` into `CrossTrackEdge`; renderer
     omits the field when `None`, prints it tagged when present.
3. **P1 round-trip gate** -- run SP-3's kernel through the engine; assert the
   same-domain A→B→A round-trip reproduces Q=0-exact (existing `Segment` machinery;
   new targeted assertion).
4. **Regression** -- the trace-sweep gate stays green; `cargo test --lib` green; an
   explicit byte-identity check on an existing convergence report pre-SP-5.

---

## 8. Epistemic framing (skew-limit Sec.9 amendment)

Sec.9 today: *"Engine: record the exact raw cross-domain offset as a
reproduction-target annotation. Never emit a cross-domain causal segment."*
SP-4b amends only the Engine bullet:

> Engine: record the exact raw cross-domain offset as `reproduction_offset`
> (**unchanged**). *Additionally*, when a **calibrated** broadcast model is
> available, emit the decomposed `causal_offset = raw − skew(A,B)` as a
> **model-derived** causal fact -- grounded on the emulator's broadcast model
> (`origin_D`), provenance-tagged `ModelDerived`, never presented as measured.
> Until calibrated, emit nothing (gap-only, status quo). Async DDR/NoC egress stays
> gap-only **always**.

The amendment stays honest about *why* this is now allowed: Sec.5-6 (the
trace-in-isolation underdetermination -- the three walls) is **untouched and still
true**. The decomposition becomes possible only because the emulator supplies a
*verified forward model* (Sec.7's "emulator-plus-verification" tier), not because
the trace gained information. The trace alone still cannot split the offset; the
calibrated emulator can -- and the `ModelDerived` provenance kind is exactly what
records that distinction in the engine's audit DAG.

---

## 9. Risks and open questions

- **Sign / A-B binding.** `skew = origin_D[parent_dom] − origin_D[child_dom]` (§2a).
  Pinned by the known-Δwall test, not the circular skew assertion.
- **Export-quantity flip.** Export `core_delay`/`mem_delay`, not `*_target` (§4b).
  Rust-side test guards it.
- **pkt_type enumeration.** The exact numeric codes for memtile and shim modules are
  resolved in the implementation plan (core=0, mem=1 confirmed). The table must be
  complete over cross-domain-reachable modules; calibrated-but-missing fails loud.
- **Keystone extension surface.** `provenance_ok` gains a `ModelDerived` branch.
  This is the first leaf class beyond Measured/Structural -- the test asserts it
  still bites on an unregistered citation, so the extension is audited, not a hole.
