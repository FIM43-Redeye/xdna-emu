# Timing-model derivability: where AIE2/AIE2P cycle-timing data comes from

**Date:** 2026-06-11
**Status:** design input (reconnaissance-backed), for the eventual timing-mode tenant.
**Related:** the unified diff-fuzzing framework (`2026-06-11-unified-diff-fuzzing-framework.md`,
"timing as a mode"); existing timing work (#83/#84/#85 three-way timing campaigns,
#107/#108 MAC pipeline-latency modeling, #86 interp<->aiesim VCD timing leg);
the source-derivation policy in `xdna-emu/CLAUDE.md`.

## Why this note exists

The framework treats **timing as a mode** on each tenant: the same generated
kernel, observed for *cycles* instead of *values*, compared with a *tolerance*
instead of byte-equality. Before we build that mode we need to know where the
timing ground truth lives, what is legitimately usable, and what is off-limits.
A read-only reconnaissance of the locally-installed aietools tree (#123) plus a
broad documentation sweep across the whole install and our own tree answers that.

**Correction (2026-06-11, after the broad doc sweep).** An earlier draft of this
note headlined "there is no latency table to read." That was an artifact of a
first pass scoped narrowly to aietools *data files*, and it undersold the
picture. The accurate headline: **the timing *model* is well-documented and
already partly built; only the last-mile exact per-instruction micro-latencies
are proprietary -- and those we close by measurement.** Specifically:

- **AM020 / AM027 ch.4** document the AIE-ML / AIE-ML-v2 timing model (pipeline
  stages, a stated 7-cycle load-result latency, lock-arbitration overhead, DMA
  channel timing) -- and we have both PDFs locally under `docs/xdna/pdfs/`.
- **We already have an AM020-derived timing subsystem in-tree**:
  `src/interpreter/timing/` (`latency.rs` cites AM020 ch.4 at 1 GHz with concrete
  per-category cycle counts, plus `arbitration`/`hazards`/`memory`/`slots`/`sync`).
  The vector fuzzer's seven silicon-only *latency-class* catches (#112/#114) and
  the MAC pipeline-latency work (#107/#108) are this model in action. Timing as a
  framework mode therefore *extends an existing, AM020-grounded model* -- it is
  not greenfield.
- The proprietary `.sfg`/`.so` tables hold the exact micro-latencies the open
  docs leave qualitative; those are the *edge*, not the main story, and we get
  them by measurement (source 2), not extraction.

The policy below is unchanged; only the "how much is available" framing is
corrected upward.

## The three sources, in priority order (mirrors the value-side policy)

### 1. Open toolchain -- PRIMARY, derive from it first

- **llvm-aie (Peano) TableGen scheduling models**
  (`../llvm-aie/llvm/lib/Target/AIE/AIE2*.td`, and the AIE2P equivalents). These
  define instruction classes, issue slots, bundle/VLIW packing rules, and
  per-class latencies -- the scheduler's own machine model. This is the same kind
  of data a cycle-accurate core model needs, and it is Apache-licensed and
  already in-tree as a reference. **Start every timing question here.** Our ISA
  decode is already fully TableGen-driven; the scheduling model is the next layer
  to consume.
- **aie-rt** (`../aie-rt/driver/src/`) for the *structure* of DMA/lock/stream
  timing -- channel state machines, BD processing order, lock semantics. It gives
  the control-flow skeleton whose latencies we then measure.

Gaps the open toolchain does **not** fully pin: exact resource-hazard / bank-
conflict penalties, bypass/forwarding timings, and per-lane vector pipelining
beyond what the `.td` latency classes express. Those go to source 2.

### 2. Measurement / observation -- GROUND TRUTH for what (1) doesn't cover

This is the doctrine we already live by ("the real NPU is always ground truth").
For timing it means **microbenchmark + measure**, never extract:

- **Real NPU1 silicon** -- the authoritative clock. Trace-based cycle measurement
  (the M-series held-level trace pipeline) already gives us per-anchor and
  total-cycle numbers on hardware (#83/#84/#85).
- **The cycle-accurate simulator** (`aie2simmsm`, the MSM cycle-accurate variant;
  `aie2simmsm_func` is functional-only) as a second, always-available oracle that
  survives the Strix swap. The in-process aiesim bridge + VCD machinery (#87/#88,
  #86 pending) is the hook.

The timing mode is then literally a differential fit: generate a kernel, measure
its cycle profile on the interpreter vs the sim vs HW, and refine the
interpreter's latency model where they disagree -- the same generate -> observe
-> compare -> localize loop as the value side, with a cycle-count `Observation`
and a tolerance `compare`.

### 3. aietools -- READ-ONLY reference for *shape*, never a data source

Per the licensing policy, aietools is a reading reference only. For timing
specifically, the #123 recon found that the *useful* timing knowledge there is
locked in two non-readable forms, and the *readable* parts give only structure:

- **Readable -- architectural shape only (no latency values):**
  `tps/.../iss_pipeline.tcl` (pipeline stage names / decode-stage indices, plain
  Tcl); the SystemC MSM scheduler skeleton under `data/systemc/simlibs/`; DDR3
  functional-model timing constants; a memory-hazard checker config. Useful for
  *confirming the pipeline structure* we model, not for numbers.
- **NOT usable -- proprietary latency data:** per-instruction latencies exist
  only as (a) cycle annotations buried in Synopsys-generated `.sfg` files
  (proprietary format, tens of MB, machine-generated) and (b) compiled closed
  libraries (`libxv_timing.so` / `libxv_timingdata.so`). **Extracting either --
  reverse-engineering the `.sfg` format or disassembling the libs -- is over the
  line.** That is lifting proprietary *data*, not reading a hardware fact and
  reimplementing it. We do not do it.

The recon's own conclusion agrees: the legally-clean and practically-better path
is empirical measurement (source 2), not source extraction.

## The line, stated plainly

- **Do:** read llvm-aie TableGen scheduling models (primary); read aie-rt for
  control structure; read aietools for *pipeline shape* (stage counts, FU names);
  measure cycles on real silicon and the cycle-accurate simulator and fit the
  model to those.
- **Don't:** parse/extract `.sfg` latency annotations, disassemble
  `libxv_timing*.so`, or otherwise copy proprietary timing tables. If the open
  toolchain doesn't have a number, we *measure* it -- we never lift it.

## How this lands in the framework

A **timing mode** on the value tenants (proposal from the framework doc, not a
separate domain): a compute or DMA case is observed for cycles via a cycle-count
`Observation` and a tolerance `compare`, reusing each tenant's generator. The
backends are the same `Backend` enum -- `Interpreter` (our model) vs `Hardware`
and/or `Aiesim` (the measurement oracles). The model we are fitting is sourced
from (1), with its gaps closed by (2); (3) only ever confirms structure.

This is downstream of the value tenants reaching solid per-subsystem signal
(finish each subsystem to 100% first). It is recorded now so the derivation
discipline is fixed before the first cycle number is written.
