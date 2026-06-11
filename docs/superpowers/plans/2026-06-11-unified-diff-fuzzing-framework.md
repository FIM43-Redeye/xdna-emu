# Unified differential-fuzzing framework

**Date:** 2026-06-11
**Status:** design, for review.
**Goal:** one framework, many domains -- a single differential-fuzzing engine
(generate -> lower -> run on N backends -> diff -> localize -> ledger -> bank/
replay) that any subsystem plugs into as a *tenant*. The vehicle for verifying
the whole device, not just compute, before Phoenix retires.

## Why now, and why it's the retirement gate

The release gate `clean_release(Aie2)` (in-code: *"Green == safe to retire
NPU1"*) is a **two-axis** predicate over the perishable queue:

```
- aietools-modeled (e.g. vector compute): UNVERIFIED   <- vector campaign clears this
- doc-specified  (e.g. DMA/stream side effects): UNVERIFIED   <- only the framework clears this
```

Vector verification (now silicon-clean, 218/218) clears axis 1. Axis 2 -- DMA,
stream, lock, and the other doc-specified side effects -- stays red until those
subsystems are differentially silicon-verified. **The framework is literally the
machine that drives axis 2 to green.** "Framework before we call it on Phoenix"
is not a preference; it is how the gate is built.

Second forcing function: Phoenix silicon is a one-way door (Strix swap). Anything
whose only oracle is real silicon is verify-now-or-never. The framework makes
that breadth tractable *and* -- because `Backend` is pluggable and includes
aiesim -- banks each domain as a permanent Interpreter-vs-aiesim oracle that
survives the swap.

## What we already have (two tenants, not one)

Two differential fuzzers evolved separately and prove the abstraction is real:

- **scalar** (`src/fuzzer/{ast,gen,lower_cpp,runner,params}.rs`, `fuzz`) --
  scalar logic AST, EMU-vs-HW.
- **vector** (`src/fuzzer/vector/`, `fuzz-vector`) -- the mature engine: coverage
  ledger with self-healing `--reverify`, durable pool+table-version banking,
  table-independent replay, type-aware NaN-tolerant comparator, parallel-compile
  / serial-HW campaign loop.

The vector engine is ~70% domain-agnostic already. Extraction = lift it to
generic, make **both** existing fuzzers tenants of it, *then* add new domains.
Extracting against two real instances (not one) is the guard against baking in
vector's assumptions.

## Core architecture

A tenant provides a `Domain`; the engine owns everything cross-cutting.

```rust
trait Domain {
    type Case;                                  // the AST: scalar prog, vector Chain, DMA BD-set, lock dance
    fn universe(&self) -> Vec<Key>;             // coverage-key space (one ledger spans all domains, namespaced)
    fn generate(&self, seed: u64, target: &Key) -> Self::Case;
    fn coverage_keys(&self, c: &Self::Case) -> Vec<Key>;
    fn lower(&self, c: &Self::Case) -> Artifact;        // kernel.cc | CDO config | inst sequence
    fn observe(&self, b: Backend, a: &Artifact, c: &Self::Case) -> Result<Observation, RunError>;
    fn compare(&self, lhs: &Observation, rhs: &Observation, c: &Self::Case) -> Option<Divergence>;
}

enum Backend { Interpreter, Aiesim, Hardware }  // differential = any two; post-swap = Interpreter vs Aiesim
```

**Engine (shared, lifted from vector):** the campaign loop (generate ->
parallel-compile -> run-on-backends -> compare -> credit/bank), the `Ledger`
(hits / divergent / resolved / unreachable, self-heal via `--reverify`, durable
bank format, table-version), replay, reporting, the backend dispatch.

**Genuinely shared (~70%):** ledger, banking, replay, campaign orchestration,
backend dispatch, coverage accounting, the CLI surface.

**Domain-polymorphic (~30%, honest):** `Case`/`generate`/`lower`/`observe`/
`compare`. The crux is `Observation` -- it is *not* uniform:
- compute -> output bytes (value diff; today's slice comparator)
- DMA / data-movement -> destination buffer + completion order
- timing -> cycle counts / latency curve (a tolerance compare, not equality)
- trace -> event sequence (held-level diff -- the M-series machinery)
- error/fault -> fault classification (which fault, when)

`compare` is therefore domain-supplied. The engine never assumes "equal bytes."

## Tenant roadmap (sequence = de-risk the abstraction first)

1. **Lift + port vector** (tenant 1). Move the engine out of `vector/` into
   `src/fuzzer/core/`; re-express vector as `Domain`. Acceptance: the 218/218
   ledger, banked corpus, and replay all still pass *unchanged*. This is a
   refactor with a hard regression bar -- nothing about vector fidelity moves.
2. **Port scalar** (tenant 2). Re-express the existing scalar fuzzer as a
   `Domain`. If the trait survives two structurally-different tenants without
   creaking, it will hold for the rest. **Gate: do not add new domains until
   1+2 are green on the shared engine.**
3. **DMA / data-movement** (tenant 3, the axis-2 payoff). Cases = random valid
   n-D BD programs (strides, wraps, padding, packet vs circuit, chaining).
   Observation = moved bytes + completion order. Oracle = HW + aiesim. This is
   the first new coverage that bites into `clean_release` axis 2.
4. **Locks / streams under contention** (tenant 4). Multi-actor cases; observe
   final state + (where modeled) grant order.
5. **Multi-tile concurrency** (tenant 5, the capstone). Cases = small
   multi-core producer/consumer programs; observe result correctness across the
   emulator's interleavings. Hardest localization -- deferred until 1-4 give
   clean per-subsystem signal first (matches the "finish each subsystem to 100%
   before moving on" policy).

**Timing and error-behavior are *modes*, not co-equal domains (proposal).** A
compute or DMA case can be observed for value *or* for cycles *or* for fault
response -- same case, different `Observation`/`compare`. Folding them in as a
`mode` axis on existing tenants beats minting separate domains whose generators
would duplicate the value tenants'.

## Migration safety

- Vector's committed behavior is the regression bar: the lift is done only when
  `cargo test --lib`, the 218/218 ledger, and the 24-seed replay are bit-identical
  to today. The banked corpus and `ledger.json` formats are preserved (the
  durable format already versions itself).
- One PR per step; each leaves a working `fuzz` and `fuzz-vector`.

## Open forks (for review)

1. **Engine home / naming.** `src/fuzzer/core/` + `src/fuzzer/domains/{...}/`
   (keep the `fuzzer` roof) vs. a fresh `src/diffuzz/`. Proposal: keep `fuzzer/`,
   add `core/` + `domains/` -- least churn, both CLIs stay.
2. **Timing/error as modes vs domains.** Proposal: modes (above). The risk is a
   mode axis bloating the `Domain` trait; if it does, promote them to domains.
3. **One ledger, namespaced keys** (`vector/...`, `scalar/...`, `dma/...`) vs.
   per-domain ledgers. Proposal: one namespaced ledger -- a single
   `clean_release`-feeding coverage truth, one replay corpus.
4. **Coupling to `clean_release`.** Should a domain reaching ledger-complete
   *auto-emit* the `Verified` override that clears its slice of the perishable
   queue, or stay a manual Maya-gated step (as #113 is)? Proposal: keep it
   manual/evidence-gated -- verification credit and release-gate flipping are
   different decisions, and the gate is WEIGHTY.

## Verification philosophy: mechanistic fidelity, not I/O matching

Differential `compare` *finds* divergences; the goal of resolving one is to model
the real **bit movement** and the real **deterministic hidden state**, not to
curve-fit outputs. Config words (e.g. the `r0=0x1c` FP-config), accumulator and
pipeline latches, rounding state, residual datapath state -- all in scope to
model precisely. A tenant is "done" when its model reproduces the mechanism, not
merely the sweep.

Worked example (sets the bar): the bf16/fp32 NaN payload is *deterministic
residual-state arithmetic*. We model it as actual bit movement -- the significands
run through the same fixed-point mantissa-adder + normalize datapath as normal
arithmetic, exponent pinned at the special-value 255 -- validated 8160/8160 on
silicon and corroborated by AMD's cluster model computing the identical `r`. That
is the standard: model the datapath, not the answer.

## Out of scope

- The everything-in-one-case mixed-domain fuzzer (real programs do all
  subsystems at once) -- maximal realism, but it destroys per-subsystem
  divergence localization. It is the eventual capstone *after* each domain is
  independently solid, not a starting point.
- **Genuinely nondeterministic** state only: *what selects* the NaN
  canonical-vs-datapath regime in a session (driver-reload-invariant,
  session-varying -- no oracle), micro-timing below aiesim resolution,
  analog/thermal. Note the line: the regime *selector* is parked; the NaN
  *arithmetic* is modeled (above). Deterministic residual state is never out of
  scope -- only the nondeterministic selector is.
