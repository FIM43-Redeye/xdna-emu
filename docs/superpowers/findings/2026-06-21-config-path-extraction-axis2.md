# Config-Path Extraction (Plan 2) — Findings & Axis-2 Close-Out

**Date:** 2026-06-21  **Branch:** `framework-tenant4-lock-trace`  **Issue:** #140

Plan 2 replaced Plan 1's hand-authored structural ledger with one **auto-generated from the
loaded binary**, and validated it end-to-end through the inference engine on captured NPU1
trace data. This documents the outcome, the bugs found, and the boundary of what config alone
can derive.

## What now lives in the emulator

1. **Static stream-switch route graph** — `DeviceState::resolve_route_graph()`
   (`src/device/stream_switch/route_graph.rs`), distilled from the runtime routing logic and
   reusing `xdna_archspec::aie2::stream_switch` topology constants. Validated (A5) as a clean
   **superset** of the 376 inter-tile hops add_one actually enacts.
2. **DMA dataflow edges** added to that graph (Tier E):
   - `EdgeKind::DmaBufferRelay` — intra-tile, S2MM (master DMA port, buffer writer) → MM2S
     (slave DMA port, buffer reader) when their BD buffer byte-ranges overlap.
   - `EdgeKind::LockPair` — intra-tile, S2MM releases lock N → MM2S acquires lock N (data-ready
     direction only; back-pressure excluded). Corroborates `DmaBufferRelay` for add_one.
   - Both validated (E4) against runtime lock enactment: a gated lock-handoff recorder shows
     every enacted dataflow handoff (`release@cy < acquire@cy`) is covered by a static edge,
     with matching orientation.
3. **Full physical dump** — `examples/dump_config_json.rs` quotes CDO + applied instruction
   stream (route graph, ports, `event_port_selection`, BDs, locks, shim mux).
4. **Sound generator** — `tools/config_extract/` turns route-graph reachability over fired
   events into a `config_path` ledger compatible with `inference/ledger.py`.

## The caveat that is now closed

Plan 1's Axis-2 used a **hand-authored** ledger. Plan 2's
`test_inference_hw_smoke.py::test_generated_ledger_*` generate the ledger from the config dump,
run the engine on the captured runs, and confirm it soundly derives the config-justified timing
relationships — **`provenance_ok`, replication clean**, on real silicon data.

## Bugs found (and the lessons)

- **DMA port-direction comments were inverted** (`stream_switch/mod.rs`): `dma_master()`/
  `dma_slave()` said "for MM2S"/"for S2MM" — backwards. Authoritative per
  `device/array/routing.rs`: master DMA port = S2MM (writer), slave = MM2S (reader). Fixed
  (`cbd7b1b7`) with the convention spelled out so it cannot be re-inverted.
- **Memtile/compute BDs read out of bounds** (`d822d193`): BD registers live in the DMA
  subsystem address space (`0xA0000`), above the tile's `data_memory` slice, so
  `BufferDescriptor::from_memory()` returned all-zero ("invalid") BDs. Now read from the decoded
  `DmaEngine.bd_configs[]` (CDO-populated, valid pre-run).
- **`config_path` a/b orientation was flipped** (`ab3107cf`): the generator emitted
  `a=child, b=parent`, but the engine (`rules.py:40`) and `ledger.py` require `a=parent,
  b=child`. Fed to the engine, the generated ledger derived **nothing**. The Tier C audit only
  checked `provenance_ok` (does the cite *resolve*) — never the engine's *derive* path — so the
  flip was invisible until Tier D ran the generated ledger through the engine. **Lesson:** a
  ledger that "loads" is not a ledger that "derives"; the regression guard now runs the full
  derive path.

## The boundary: config ceiling vs through-core

With the orientation fixed, the generated ledger soundly derives the two **memtile** buffer
relays (`PORT_RUNNING_4 ← PORT_RUNNING_0`, offset ≈30; `PORT_RUNNING_5 ← PORT_RUNNING_1`,
offset ≈88) and correctly **declines** the co-firing pairs (`PR0→PR1`, `PR0→PR5`). It does NOT
derive the shim-DMA cross-pipeline causality (`DMA_S2MM_0 ← DMA_MM2S_0`) or `STREAM_STARVATION`
that the hand ledger had, because shim `MM2S → S2MM` is **not route-reachable**: the forward
path **breaks at the compute tile**, where `S2MM(in1) → core(+1) → MM2S(out1)` uses *different*
buffers (no `DmaBufferRelay`) and the **core** (not a DMA) performs the lock handoff (no
`LockPair`).

That gap is **exactly** the through-core relay, which needs the compute **ELF lock
instructions** — a different source (program, not config). It is slated as the immediate
follow-on plan: `docs/superpowers/plans/2026-06-21-program-path-through-core.md`.

**So Tier E sits at its correct config ceiling:** it derives everything the configuration
justifies, refuses co-firing, and the only residual is program-derived. The hand ledger reached
further only via human domain knowledge of the full pipeline.

## Status

- **Config-path extraction (Plan 2 / Tier E): ready at the config ceiling.**
- **#140 overall: not done** — `program_path` (through-core) is the next tractable plan, and
  further tiers follow. Splitting #140 across tiers/plans is the deliberate strategy.
