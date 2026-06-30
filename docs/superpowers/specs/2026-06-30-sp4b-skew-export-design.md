# SP-4b -- Engine skew-export + causal decomposition (design)

**Arc:** #140 timer-sync faithful-broadcast
([`2026-06-28-timer-sync-faithful-broadcast-arc.md`](2026-06-28-timer-sync-faithful-broadcast-arc.md) §4, SP-4b).
**Depends on:** SP-1 (landed). **Not** SP-2, **not** SP-3 (see §6 -- the P1
round-trip gate moved to SP-5, removing the SP-3 kernel dependency).
**Status:** design approved 2026-06-30; revised after two review passes
(adversarial code-binding review + deeper modeling-assumption review); awaiting
user-review gate + implementation plan.

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
   flag, under a single-flood-source precondition (§4d).
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
  origins.
- **No SP-3 dependency.** The in-domain round-trip coupling-latency gate (arc
  bullet P1) moves to SP-5 (§6): it only validates a *nonzero* cross-domain residual
  as pure skew, which cannot exist pre-calibration.
- **No direction-diverse rank-2 `d_h`/`d_v` solve in the engine.** Exporting
  finished `origin_D` already bakes in the directional sum; the rank-2 *measurement*
  is SP-5 (§6).
- **No multi-flood-source reconciliation.** SP-4b asserts a single reset source and
  fails loud otherwise (§4d/§5d); per-source `T0` reconciliation is out of scope.
- **Async DDR/NoC (`is_async_cdc`) pairs stay existence-only gaps**, always.

---

## 2. Background, the sign derivation, and what `causal_offset` claims

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
events `x` (domain A) and `y` (domain B), both reset by the same flood at a single
`T0` (§4d precondition):

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
`skew == origin_D[B] − origin_D[A]`, which is circular.

### 2b. What `causal_offset` actually claims (exact-in-model, estimate-of-silicon)

`causal_offset` is **exact within the model**: given the broadcast model and its
constants, the subtraction `raw − skew` is an exact integer. Its fidelity to
*silicon* equals the **calibration fidelity** -- on a HW trace,
`causal_offset = HW_raw − model_skew` equals the true silicon `Δwall` only insofar
as `model_skew == HW_skew`, i.e. only as well as SP-5 measured the constants. On an
EMU trace it is self-consistent (the model's own `Δwall`). The `ModelDerived`
provenance tag (§5a) records the dependency on the model; the design treats
`causal_offset` as model-exact and silicon-approximate, and the report must never
present it as a measured silicon fact. (This is *why* it is withheld until
calibrated: an uncalibrated model is not even an estimate.)

### 2c. Verified: the PERF_CNT_2 anchor cancels

Every `soc` is anchored to one event in domain `1|2|0`:
`soc_anchored(x) = (W_x − W_anchor) − origin_D(A) + origin_D(anchor_dom)`. When
`raw = soc(child) − soc(parent)` is formed, the `origin_D(anchor_dom)` term appears
in both and **cancels** -- so the derivation holds regardless of which domain the
anchor lives in. (Checked during the deeper review; the anchoring does not
contaminate the decomposition.)

---

## 3. Architecture

```
EMULATOR (Rust)                         INFERENCE ENGINE (Python)
---------------                         -------------------------
effects.rs broadcast_origin_d           loader_model.py  (NEW)
  per-module core_delay/mem_delay ─────►   reads sidecar -> MODEL_DERIVED facts:
  (= origin_D; NOT *_target)                 origin_d(domain_key, D)   [cites sidecar]
  + single-source assertion (§4d)            skew_calibrated(bool)     [cites sidecar]
model_builder.rs                            flood_source(key)         [cites sidecar]
  BroadcastTiming.calibrated (NEW)                │
        │                                         ▼
        ▼                              grounding.ground_edge(.., origin_D, skew_calibrated)
  origin_d.json sidecar                 (single-source check: both domains reset from
  written on the existing                the one flood, else fail loud -- §5d)
  trace run, keyed                       skew = origin_D[parent_dom] − origin_D[child_dom]
  col|row|<pkt_type>                     causal_offset = (raw − skew) if calibrated else None
  (numeric pkt_type)                     Gap += {causal_offset}  (reproduction_offset unchanged)
                                                  │
                                                  ▼
                                        weave -> CrossTrackEdge += causal_offset
                                        rules.py: SEPARATE causal fact w/ MODEL_DERIVED premises
                                        facts.py: provenance_ok gains a MODEL_DERIVED branch
                                        renderer: prints causal_offset only when present, tagged
```

The emulator's job ends at "write an honest sidecar." The engine's job is to
consume it **without ever laundering modeled data into a measured claim** -- the
model dependency must appear in the causal fact's *premises*, not merely its data
args (§5a).

---

## 4. The Rust export

### 4a. `calibrated` flag on `BroadcastTiming`

Add a `calibrated: bool` field to `BroadcastTiming`
(`crates/xdna-archspec/src/types.rs:1353`), set `false` in `model_builder.rs:270`.
It is an **explicit** flag, **not** inferred from "constants != 0" -- a genuinely
measured `d_v = 0` at SP-5 must not read as uncalibrated. SP-5 flips it to `true`.
This is the single source of truth for the engine's `skew_calibrated`.

### 4b. Per-module `origin_D` export

`broadcast_origin_d` (`effects.rs:443`) produces `(col, row, origin_D)` per reached
tile via a Dijkstra min-cost wavefront; `:582-584` splits it into
`core_delay = origin_d + core_off` / `mem_delay = origin_d + mem_off` -- **these are
the physical `origin_D` values to export.**

> **Export-quantity caution.** Export `core_delay`/`mem_delay` (`:583-584`), **not**
> the adjacent `core_target`/`mem_target` (`:585-586` = `max_delay − delay`), which
> are the *complement* of `origin_D`. Grabbing the `*_target` neighbor inverts the
> skew sign undetectably. A Rust-side test asserts the exported value equals
> `origin_d + intra_off`, not its complement.

**Keying.** The sidecar keys each module by the engine's domain identity
`col|row|<pkt_type>`, where **`pkt_type` is the numeric trace packet-type code**,
matching `same_domain`/`_domain_of` (`grounding.py:51`) `key.rsplit("|", 1)[0]`. The
anchor `1|2|0|PERF_CNT_2` confirms the form (col=1,row=2,pkt_type=0=core). **Not**
`col|row|core`. The table must cover **every module type that can appear in a
cross-domain pair** -- core, mem, memtile, shim. Either Rust emits numeric
`pkt_type` directly, or `loader_model.py` owns the `module → pkt_type` translation
(exact enumeration resolved in the plan; core=0, mem=1 confirmed).

```json
{
  "calibrated": false,
  "flood_source": "0|0|0",
  "modules": { "1|2|0": 0, "1|2|1": 0, "1|1|<memtile_pkt>": 0, ... }
}
```

### 4c. Hook point

The export rides the existing trace-emission path -- the run that writes
`trace.events.json` also writes `origin_d.json` beside it. One EMU run, two
artifacts, no new entry point.

### 4d. Single-flood-source precondition + assertion

`skew(A,B) = origin_D(B) − origin_D(A)` is the *complete* skew **only if** every
traced module's timer was reset by **one** flood from **one** source at a single
`T0`. The emulator does not structurally guarantee this: `propagate_broadcasts`
floods from a single `(col, source_row)` per call, but
`propagate_broadcasts_fixpoint` (`effects.rs:642`) re-floods from *every* tile with
pending broadcasts, and the reset path overwrites `set_origin_offset` last-writer-
wins (`effects.rs:606-609`, `timer.rs:203`) -- the inter-source `T0_B − T0_A` term
is silently dropped. So a config with two independent reset sources would yield a
wrong `causal_offset` with no warning.

SP-4b therefore **asserts a single source** at export time: `broadcast_origin_d`'s
seed is one tile, and the export records that `flood_source` and asserts no other
tile generated a timer-reset broadcast over the run. The engine cross-checks
(`§5d`) and **fails loud** on any cross-domain pair whose two domains did not both
reset from `flood_source`. Per-source reconciliation (tracking each `T0` and adding
`T0_B − T0_A`) is explicitly out of scope (it is itself an SP-5 calibration
quantity).

---

## 5. The Python engine

Throughline: **modeled data never launders into a measured claim**, and **all
existing engine outputs are byte-identical pre-SP-5**.

### 5a. Third provenance category, as a real premise (not a data arg)

Today the engine has two leaf sources: `loader.py` (measured `fired` facts) and
`ledger.py` (structural facts cited into `kb.ledger`). The sidecar is neither -- it
is **model-derived**. Add:

- A new `ModelDerived` support kind in `facts.py`, carrying a **citation to the
  model artifact** (the `origin_d.json` sidecar + its `calibrated` flag), mirroring
  how `Structural` cites `kb.ledger`.
- A real branch in `provenance_ok` (`facts.py:108`): a `ModelDerived` leaf is
  accepted **iff** its citation is a registered model source (e.g. `kb.model`),
  exactly paralleling the `Structural`/`kb.ledger` check at `:119-121`. Today
  `provenance_ok` rejects any non-Measured/non-Structural leaf at `:122-123`; this
  is a deliberate, audited keystone extension to a third leaf class -- it does
  **not** silently bless model data as measured.
- A new loader (`loader_model.py`) emitting `origin_d(domain_key, D)`,
  `skew_calibrated(bool)`, and `flood_source(key)` facts under `ModelDerived`, cited
  to the sidecar.

**The causal fact is a separate fact with model premises.** The existing gap
`derives` fact (`rules.py:56-58`) carries `reproduction_offset` as a *trailing data
arg* whose premises are only the measured `fired` facts + structural placement -- the
model dependency is invisible to `leaves()`, so `provenance_ok` would bless a
model-derived number as fully measured. SP-4b emits `causal_offset` as its **own**
fact whose `Derived` premises explicitly include the `origin_d(child_dom)`,
`origin_d(parent_dom)`, and `skew_calibrated` `ModelDerived` facts (plus the
measured raw). Then the fact's DAG genuinely bottoms out on model leaves,
`leaves()` surfaces the `ModelDerived` support, and any consumer can detect and
label it. `reproduction_offset` stays untouched.

### 5b. Decomposition in `grounding.ground_edge` (cross-domain branch, `:105`)

`ground_edge` gains parameters: the `origin_D` table and `skew_calibrated` (threaded
from the KB's `ModelDerived` facts -- a signature change; `rules.py` is the single
caller that supplies them and builds the separate causal fact from the same
computation, so the render-path value and the provenance-path value never diverge).
In the cross-domain branch:

```python
skew = origin_D[parent_dom] − origin_D[child_dom]   # A=child, B=parent (§2a)
causal_offset = (raw − skew) if skew_calibrated else None
```

`Gap` (`grounding.py:74`) gains one field -- `causal_offset: Optional[int] = None` --
with `reproduction_offset` **untouched**. The `is_async_cdc` early-out (`:98`) stays
existence-only. Lookups use `origin_D[k]` (not `.get`): a **calibrated** model with
a missing module entry must **fail loud**. Pre-calibration, the lookup is not
reached.

### 5c. No guard relaxation -- the change is additive on the edge path

The arc's "relax the `timeline.py` cross-domain fail-loud guard" is imprecise. The
only runtime guard, `internal_cycles:410`, asserts a **`ClusterFrame` is
single-domain** -- and frames are built per-domain, so a cross-domain frame **never
reaches it**. It is a fatal-A wiring-slip protection; relaxing it would weaken that
check while doing nothing for SP-4b. The actual cross-domain path is the **edge
path** `weave → ground_edge → Gap → CrossTrackEdge` (`timeline.py:548-558`), which
has **no runtime guard** -- the invariant is enforced *structurally* by
`CrossTrackEdge` lacking a cycle field. SP-4b's change is purely additive:

- `CrossTrackEdge` (`timeline.py:144`) gains `causal_offset: Optional[int]`.
- `weave` (`:555-557`) is updated to copy `g.causal_offset` (it currently copies
  only `reason` and `reproduction_offset`).
- The **renderer** presents `causal_offset` as a model-derived cross-track quantity
  (tagged), never a local tile cycle, and **omits the field entirely when `None`**
  -- absence already signals "no causal cycle"; printing "withheld" would change
  pre-SP-5 render output and break byte-identity.

`internal_cycles:410` is left untouched.

### 5d. Single-source fail-loud cross-check

Using the `flood_source` fact (§4d), the engine refuses any cross-domain
decomposition whose two domains are not both reachable from `flood_source` in the
exported table -- a hard error, never a silent `causal_offset`. Pre-calibration this
path is dormant (no decomposition runs); under calibration it guarantees the skew
formula is only applied where it is complete.

### 5e. Byte-identity: the single load-bearing spot

Pre-SP-5 (`calibrated == false`): the loop, planner, and chainer are unaffected --
they read the KB only via `by_predicate` filters
(`loop.py` `ranking()`, `planner.py`, `chainer.py`), never all-facts enumeration, so
inert `origin_d`/`skew_calibrated`/`flood_source` facts cannot shift any
count/measure/fixpoint. The convergence report enumerates only predicate-specific
lists (`derives`/`segments`/`gaps`/...), not raw leaves. The **one** all-facts
enumeration in the engine is `provenance_ok` (`facts.py:112`) -- and it is an `AND`,
not a count, so N extra facts do not shift it. It stays `True` **iff** the new
`ModelDerived` leaves are accepted by the new registered-citation branch (§5a).
That branch is therefore the single load-bearing precondition for byte-identity, and
§7 tests it directly (the report's `provenance_ok` field must stay `True` pre-SP-5).
The byte-identity claim is scoped to existing outputs + EMU trace bytes -- not to the
raw KB fact *set* (the inert model leaves are new).

---

## 6. Scope: the three arc SP-4b bullets under this design

| Arc bullet | Disposition | Rationale |
|---|---|---|
| **P3 -- export skew, add `causal_offset`, emit model-grounded causal fact, amend Sec.9** | **IN -- the build** (§4-5, §8). Gated on `skew_calibrated`. | The concrete engine deliverable. |
| **P2 -- direction-diverse rank-2 solve for separate `d_h`/`d_v`** | **DEFER to SP-5** (signed off 2026-06-30). | Exporting finished `origin_D` already bakes in `n_h·d_h + n_v·d_v` -- `broadcast_origin_d` does min-cost Dijkstra (`effects.rs:486-500`) and the engine needs only the scalar `origin_D`. The rank-2 *measurement* is SP-5 (route-3b). |
| **P1 -- in-domain round-trip coupling-latency gate** | **DEFER to SP-5** (signed off 2026-06-30). | P1 pins coupling latency in-domain so the cross-domain residual is *pure skew* -- but that residual is nonzero only once constants are calibrated (SP-5). Pre-SP-5 (skew=0) there is nothing to validate. Its round-trip kernel does not exist as committed code (the SP-3 validation kernel was designed + spiked but not landed; only an engine-completeness spike exists), and the silicon it guards lives in SP-5. So P1 belongs with SP-5, removing the SP-3 dependency from SP-4b entirely. |

**Net SP-4b scope:** P3 (the decomposition machinery) + the §4d/§5d single-source
guards, validated by synthetic-constant unit tests and the byte-identity regression
-- **no kernel dependency**. P1 and P2 are SP-5's; the deferrals were presented and
signed off 2026-06-30.

---

## 7. Testing (TDD)

The key testability move: **synthetic non-zero constants** exercise the full live
path without silicon.

1. **Rust** -- `origin_D` table export keyed `col|row|<pkt_type>` over
   core/mem/memtile/shim; exported value equals `origin_d + intra_off` (not the
   `*_target` complement); `calibrated` flag plumbs end-to-end; single-source
   assertion fires on a synthetic two-source config; sidecar round-trips.
2. **Python (core):**
   - *Sign-pinning (load-bearing):* asymmetric synthetic `origin_D` **and** a known
     nonzero synthetic `Δwall` -> assert `causal_offset == Δwall`. Not the circular
     `skew == origin_D[B] − origin_D[A]`.
   - *Withhold gate:* `calibrated = false` -> **no causal fact emitted**; every
     existing output identical to pre-SP-4b; **`provenance_ok` stays `True`** with the
     inert `ModelDerived` leaves present (§5e).
   - *Provenance, both directions:* a calibrated `causal_offset` fact's `leaves()`
     includes a `ModelDerived` leaf accepted via model citation; a `ModelDerived`
     leaf with an **unregistered** citation makes `provenance_ok` return `False`.
   - *Single-source fail-loud:* a cross-domain pair whose domains reset from
     different sources raises, never yields a `causal_offset` (§5d).
   - *async-CDC exclusion:* async pairs get neither offset, even calibrated.
   - *Domain-key resolution:* every traced cross-domain module resolves to a sidecar
     entry; calibrated-but-missing fails loud.
   - *Edge threading:* `weave` copies `causal_offset`; renderer omits the field when
     `None`, prints it tagged when present; render output byte-identical pre-SP-5.
3. **Regression** -- the trace-sweep gate stays green; `cargo test --lib` green; an
   explicit byte-identity check on an existing convergence report pre-SP-5.

(The P1 in-domain round-trip gate is **SP-5's** test, not SP-4b's -- §6.)

---

## 8. Epistemic framing (skew-limit Sec.9 amendment)

Sec.9 today: *"Engine: record the exact raw cross-domain offset as a
reproduction-target annotation. Never emit a cross-domain causal segment."*
SP-4b amends only the Engine bullet:

> Engine: record the exact raw cross-domain offset as `reproduction_offset`
> (**unchanged**). *Additionally*, when a **calibrated, single-source** broadcast
> model is available, emit the decomposed `causal_offset = raw − skew(A,B)` as a
> **model-derived** causal fact -- exact in the model, an estimate of silicon with
> error equal to the calibration error (§2b), provenance-tagged `ModelDerived`,
> never presented as measured. Until calibrated, emit nothing (gap-only, status
> quo). Multi-source pairs fail loud; async DDR/NoC egress stays gap-only **always**.

The amendment stays honest about *why* this is allowed: Sec.5-6 (the
trace-in-isolation underdetermination -- the three walls) is **untouched and still
true**. The decomposition becomes possible only because the emulator supplies a
*verified forward model* (Sec.7's "emulator-plus-verification" tier), not because
the trace gained information. The `ModelDerived` provenance kind records exactly
that distinction in the engine's audit DAG.

---

## 9. Risks and open questions

- **Sign / A-B binding.** `skew = origin_D[parent_dom] − origin_D[child_dom]` (§2a);
  pinned by the known-`Δwall` test, not the circular skew assertion.
- **Export-quantity flip.** Export `core_delay`/`mem_delay`, not `*_target` (§4b);
  Rust-side test guards it.
- **Single-source completeness.** The skew formula is complete only under one flood
  source (§4d). Guarded by export-side assertion + engine-side fail-loud (§5d);
  per-source reconciliation deferred (SP-5+). Open: confirm the actual traced
  kernels use a single BROADCAST_15 reset (expected, but the guard makes a violation
  loud rather than silent).
- **pkt_type enumeration.** Exact numeric codes for memtile/shim modules resolved in
  the plan (core=0, mem=1 confirmed); table complete over cross-domain-reachable
  modules; calibrated-but-missing fails loud.
- **Keystone extension surface.** `provenance_ok` gains a `ModelDerived` branch --
  the first leaf class beyond Measured/Structural. The test asserts it still bites on
  an unregistered citation, so the extension is audited, not a hole. This same branch
  is the single load-bearing spot for pre-SP-5 byte-identity (§5e).
- **Verified, holds (deeper review):** the PERF_CNT_2 anchor cancels in the
  child−parent difference (§2c); the loop/planner/chainer use `by_predicate`, not
  all-facts enumeration, so inert model facts cannot perturb them (§5e).
