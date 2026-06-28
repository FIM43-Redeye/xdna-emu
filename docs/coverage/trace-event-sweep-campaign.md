# Trace-Event-Sweep Characterization Campaign

**Status:** PLAN (decisions locked 2026-06-06); pending: (1) build the sweep aggregator (cross-kernel fidelity report, NEEDS BUILDING), (2) three-way harness integration completes (step 9 of three-way-timing-calibration.md), (3) execute full-corpus sweep.
**Date:** 2026-06-06

## Goal

Characterize **every traceable hardware trace event** on real Phoenix NPU1 and
measure the emulator's event-level fidelity against silicon -- fire count *and*
cycle timing, per event, per tile -- across the test corpus, while Phoenix is
still available.

This is the finest-grained emulator validation we can do: not "does the kernel
produce the right output" but "does every hardware trace event fire, and fire
at the same cycle, on the emulator as on silicon." The sweep capability
(`tools/trace-sweep.py`, Phase 5b) was built for exactly this.

## Why now

Phoenix is a one-way door at the Strix swap. Event-level HW ground truth is
irreplaceable once the hardware is gone -- the aiesimulator models a generic
device, not Phoenix, and our emulator's event fidelity can only be *anchored* to
real silicon while we still have it. This campaign banks that ground truth.

## What the sweep produces (mechanism exists)

`trace-sweep.py` compiles a kernel once, then rotates every traceable event
through the tile's 8 trace slots, batch by batch, running HW (serial) and EMU
(parallel) per batch -- no recompilation. Per `(kernel, compiler, tile)` it
emits an event matrix:

```
event_id -> { hw_cycles, hw_fire_count, emu_cycles, emu_fire_count }
```

plus a cross-batch `.merged.json` (anchored timeline) and `sweep-manifest.json`
(grounding + PC-invariance metadata). Already wired into the harness:
`./scripts/emu-bridge-test.sh --sweep`, Phase 5b, which auto-discovers traceable
tiles from each test's `aie.mlir` (row >= 2 -> core + memmod; row 1 -> memtile;
row 0 -> shim).

## The event space ("everything")

Authoritative source: `xaie_events_aieml.h` (the same enum aie-rt and the MLIR
trace lowering use). Distinct traceable events and resulting batch counts:

| Tile type | Events | Slots (after grounding) | Batches / kernel-tile |
|-----------|--------|-------------------------|-----------------------|
| core      | 127    | 5 (3 ground)            | ~24                   |
| memmod    | 122    | 7 (1 ground)            | ~18                   |
| memtile   | 161    | 7 (1 ground)            | ~23                   |
| shim (PL) | 128    | 7 (1 ground)            | ~18                   |

A given kernel only *fires* a subset of these; the rest record fire-count 0 --
which is itself a data point (it confirms the emulator also does not fire them).

## Scope decisions (proposed; see "Open decisions")

- **Events: ALL.** Fire-count 0 is meaningful (negative confirmation). "Every
  traceable event" is the stated goal.
- **Tiles: all traceable tiles per kernel** (Phase 5b already does this).
- **Kernels:** full corpus vs diverse subset -- the main cost lever (below).
- **Compilers:** chess is ground truth; peano secondary.

## Magnitude

Driven by batch count, not per-batch time (each batch is a fast patch + run +
parse, ~0.3-1.5s all-in on HW).

- **Full corpus** (~72 PASS kernels x ~3 traceable tile-types x ~20 batches x 2
  compilers) ~= **8-9k batch-runs** -> roughly **45 min - 3.5 hr** of HW,
  depending on per-batch overhead. Bounded; one to a few sessions.
- **Diverse subset** (~18 kernels chosen to span DMA / lock / cascade / memtile
  / control-packet / objectfifo) ~= **2-2.5k batch-runs** -> **~20-45 min**.
  Event coverage saturates quickly: a handful of kernels per category exercises
  nearly the whole *fired*-event space; the long tail is never-fired events,
  which are corpus-independent.

## Execution

`./scripts/emu-bridge-test.sh --sweep [--with-cycle-diff] [-v <filter>]`
(serial HW, parallel EMU). Output under `RESULTS_DIR/<safe>.<compiler>.sweep/`.

## Analysis (NEEDS BUILDING)

`trace-sweep.py` emits per-tile matrices + `.merged.json` + `sweep-manifest.json`,
but there is **no cross-kernel aggregator**. Deliverable to build:

- An aggregator that rolls every sweep matrix into an **event-fidelity coverage
  report**: for each `(tile-type, event)`, across the corpus --
  HW-fired? EMU-fired? cycle-match within tolerance? -> one of
  `MATCH / DRIFT(dt) / HW-ONLY / EMU-ONLY / NEVER-FIRED`.
- Output: a coverage matrix answering both questions at once -- "did we
  characterize every event?" and "does the emulator match silicon on each?"
  This is the campaign's actual product (the per-event fidelity ledger), and it
  becomes a durable regression oracle for the emulator refactor.

## Relationship to the three-way timing campaign

Complementary, not redundant:

- **Three-way timing** (#85): a *narrow* event set (DMA task anchors), *three*
  sources (HW / interp / aiesim), focused on per-anchor cycle drift.
- **Trace-event sweep** (this): the *full* event vocabulary, *two* sources (HW
  vs EMU -- aiesim is not in the sweep path), focused on per-event fire +
  timing fidelity.

They could fold: run the sweep first for comprehensive HW-vs-EMU event fidelity,
then the three-way to add the aiesim leg on the DMA anchors specifically.

## Decisions (locked 2026-06-06)

1. **Kernel scope: FULL CORPUS.** Completeness over cost -- this is the durable
   Phoenix record. (Diverse-subset shakeout dropped; we just want all the data.)
2. **Compiler: chess + peano** (both).
3. **MATCH tolerance: trace-compare default (`|dt| > 10`) for now.** We need the
   data first; tighten the verdict later if warranted.
4. **Sequencing: AFTER the three-way re-run.** Both need Phoenix and cannot
   overlap. The three-way re-run is ready (B-full fixed its cycle leg) and runs
   first; the sweep campaign follows once the aggregator is built.

## Future throughput idea (park until framework exists)

HW is *very fast* per batch. Worth examining -- after the aggregator/analysis
framework is built -- whether multiple kernels can be driven on the NPU
concurrently (multi-column / multi-context) to cut wall-clock on the full-corpus
sweep. Not now; it changes nothing about the data we gather, only how fast.

## Build order

1. (in flight) three-way timing re-run -- closes #85, exercises B-full on HW.
2. Build the sweep aggregator (no HW): roll per-tile matrices into the
   per-event HW-vs-EMU fidelity coverage report.
3. Run the full-corpus sweep campaign (chess + peano, all tiles, all events).
4. (later) Examine multi-kernel HW concurrency for throughput.
