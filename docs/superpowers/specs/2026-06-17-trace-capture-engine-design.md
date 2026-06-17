# Self-owned NPU trace-capture engine (task #140) — design

**Date:** 2026-06-17
**Status:** approved design, pre-plan
**Predecessor:** `2026-06-16-cross-batch-trace-join-design.md` and its plan. The
join library (`tools/trace_join.py`, Tasks 1-6) is built and reviewed; this
engine is the execution layer that feeds it correctly-labeled HW data.

## Goal

Build a self-owned capture engine that takes a **batch plan** (which events to
trace on which tile/module in each batch) and produces correctly-labeled
`events.json` per batch per run on real hardware — with reliable coverage,
candidates co-traced in one execution, and repeatable N times.

The engine replaces the existing `trace-sweep.py` *orchestration* (its batching,
grounding, and `_relabel_events` guessing), which proved unreliable for the
join: a restricted sweep silently produced one incomplete batch and labeled
events from guessed slot-names rather than the actual configuration. The engine
keeps only the audited, toolchain-derived hardware primitives.

## Why this shape

The capture pipeline separates cleanly into toolchain-derived hardware
primitives (sound, kept) and trace-sweep's own orchestration (murky, discarded).
Three primitives were audited on evidence (2026-06-17):

- **`tools/trace-patch-events.py`** (register patcher) — writes 8 event IDs into
  a tile module's `Trace_Event0/1` registers. Register offsets
  (`core 0x340E0/4, memmod 0x140E0/4, memtile 0x940E0/4, shim 0x340E0/4`) match
  aie-rt's `xaiemlgbl_params.h` exactly; NPU address `(col<<25)|(row<<20)|offset`
  is correct; explicit `ValueError` on a missing target write (no silent
  corruption); standalone CLI + importable pure functions. **Sound, reuse.**
- **`bridge-runner/bridge-trace-runner.cpp`** (XRT runner) — runs the kernel,
  syncs the trace BO, writes a raw `trace.bin`; standalone stdin protocol, no
  trace-sweep coupling. **Sound, reuse — with one operational rule:** the trace
  shim-DMA byte counter is cumulative across runs, so the engine must issue
  `RESET` between batches/runs for clean independent windows.
- **`tools/trace_decoder/`** (in-tree decoder) — decodes `trace.bin` to raw
  `(col,row,pkt_type,slot,ts,soc,mode)` with `parse_trace(words,
  slot_names=None)`. Raw mode-0 parity with mlir-aie is covered by
  `test_mode0_decode_matches_oracle_byte_for_byte` (19/19 passing,
  2026-06-17). **Sound for the raw layer, reuse.**

**Decoder choice — in-tree raw + live parity guard.** The decoder-divergence
history is real but confined to the **held-level span / timeline** layer (e.g.
commit `70b25060`, 2026-06-16: span builder merged `PORT_RUNNING` bursts), where
`decode.py` declares mlir-aie authoritative. The **raw milestone layer**
(`pkt_type`, slot, `soc`) — all the milestone join needs — is the parity-clean
part. We use the in-tree decoder for raw decode (it is decoupled from naming and
gives us the exact-labeling win), and add a **live parity guard**: decode a real
`trace.bin` with both the in-tree decoder and mlir-aie's `parse_trace`, assert
raw-layer agreement, so drift on our actual data is caught — not just the frozen
fixture. The span layer (the deferred I-2 work) routes through the
upstream-authoritative decoder if and when we reach it.

## Module-explicit keys (collision elimination)

The whole chain — active events, derivability graph, plan, capture — uses the
key **`"{col}|{row}|{pkt_type}|{name}"`**. Row 2 carries two trace modules
(core `pkt_type=0`, memmod `pkt_type=1`); an event name like `PERF_CNT_2` exists
in *every* module's table, so a `(col,row,name)` key is ambiguous between
modules. Including `pkt_type` makes core-`PERF_CNT_2` and memmod-`PERF_CNT_2`
distinct **by construction** — no name-table guessing, no collision guard, and
it also fixes a latent bug where the library's `(col,row,name)` grouping would
silently merge two physically distinct events. (Verified: add_one has 0 such
collisions today, but tile `(1,2)` does trace both modules, so the ambiguity is
latent — we eliminate the class rather than rely on luck.) The anchor key
becomes `"1|2|0|PERF_CNT_2"`.

Module is resolved from the **observed `pkt_type`** in the discovery decode (the
event fired on a specific module), never from name-table membership.

## Engine architecture — five components

`tools/trace_capture.py`, a self-owned executor. Input: a batch plan, test name,
run count N, output dir. Output: `run_NN/batch_MM/hw/trace.events.json` in the
schema `trace_join` consumes (`col,row,pkt_type,name,slot,ts,soc,mode`).

1. **Name->ID resolver** — parses aie-rt's `xaie_events_aieml.h`
   (`#define XAIEML_EVENTS_<MOD>_<NAME> <id>U`) to map
   `(tile_type, event_name) -> numeric ID`. Small, self-owned (no trace-sweep
   dependency). Module prefixes: `CORE->core, MEM->memmod, MEM_TILE->memtile,
   PL->shim`.
2. **Slot configurator** — per batch, builds each tile-module's 8-slot event-ID
   list (the events we chose), calls `trace-patch-events.py --multi-tile` ->
   `insts.patched.bin`. We own the slot assignment; slot order is fixed
   (anchor/grounding first), and the `(pkt_type,col,row,slot)->name` map is
   recorded as the labeling oracle.
3. **Runner driver** — drives `bridge-trace-runner` over its stdin protocol
   (`--instr --trace-out --trace-size --input --output --ctrlpkt`); issues
   `RESET` between batches/runs -> `trace.bin`.
4. **Raw decoder + exact labeler** — `trace_decoder.parse_trace(words,
   slot_names=None)` -> raw events; labels each by the recorded
   `(pkt_type,col,row,slot)->name` map, with a column-offset reconcile for HW's
   runtime `start_col`. An event decoding to an unconfigured slot is a hard
   error.
5. **Parity guard** (test, not runtime) — decode a representative real
   `trace.bin` with both the in-tree decoder and mlir-aie's `parse_trace`,
   assert raw `(pkt_type, slot, ts)` agreement.

## Per-batch data flow

For each run (1..N), for each batch in the plan:

1. **Resolve events to modules** — the plan's per-tile event lists carry
   `pkt_type` (module-explicit keys), so produce per-`(col,row,module)` slot
   lists directly. Shim (row 0) and memtile (row 1) are single-module.
2. **Assign slots** — fixed order, anchor first (slot 0); record the
   `(pkt_type,col,row,slot)->name` map.
3. **Configure** — build the multi-tile patch spec (event IDs per module),
   call the patcher -> `insts.patched.bin`.
4. **Run** — `RESET` the runner, then run the kernel with the patched insts ->
   `trace.bin`.
5. **Decode raw** — in-tree decoder, `slot_names=None`.
6. **Label exactly** — `(pkt_type,col,row,slot)` lookup in the recorded map;
   column-offset reconcile; hard error on an unconfigured slot.
7. **Write** `run_NN/batch_MM/hw/trace.events.json`.

The inversion from trace-sweep: it decoded first and *guessed* names from MLIR;
we **record the name->slot map at config time and apply it after decode**, so
labeling cannot drift from what we actually traced.

## Coverage & co-tracing guarantees

- **Coverage:** every planned event is configured in a real slot; after decode,
  every configured slot is accounted for (fired-and-labeled, or recorded
  silent-for-this-run). A decode to an unconfigured slot is a hard error. A
  coverage gap is *visible*, not inferred from a relabel mismatch.
- **Co-tracing:** all four tiles' modules are configured in one patched
  `insts.bin` per batch and run in one kernel execution, so every event in a
  batch is co-traced in one silicon run — exactly what derivability measurement
  needs (shim DMA and core DMA in the same execution). The plan controls which
  events share a batch; the engine guarantees they share an execution.

## Wiring — the full join loop

The engine is the executor; `trace_join` is the planner/analyzer. They compose:

```
1. Discovery   -> engine runs a broad per-module plan -> active events
2. Derivability-> engine captures a candidates-co-traced plan, N runs -> graph
3. Plan        -> trace_join.synthesize_plan -> a planned batch plan
4. Capture     -> engine runs the planned plan, N runs -> labeled events.json
5. Join        -> trace_join.join_run -> one merged every-event trace
6. Validate    -> trace_join.cross_run_skeleton across runs
```

The engine's machinery is identical at steps 1/2/4; only the plan differs.

**Discovery bootstrapping:** the engine runs whatever plan it is given, so
discovery is "run a broad plan, union what fires" — no special orchestrator. For
the add_one validation we **seed from the known active set** (already in hand);
catalog-discovery is a capability the engine enables, not built out now (YAGNI).

## Deliverables this cycle

1. **Library module-explicit retrofit** — `trace_join.py` and its tests keyed by
   `(col,row,pkt_type,name)` (the foundation; the anchor becomes `1|2|0|PERF_CNT_2`).
2. **The capture engine** — `tools/trace_capture.py` + `tools/test_trace_capture.py`
   (the five components).
3. **Loop driver + add_one HW validation** — a thin driver wiring
   library<->engine through the six steps, and cross-run validation on real
   planned captures.

## Testing

- **Unit (synthetic, no HW):** module resolution, slot assignment,
  record-then-apply labeling, configured-vs-observed reconciliation — synthetic
  raw-event fixtures, deterministic.
- **Parity guard:** in-tree vs mlir-aie raw-layer agreement on one real
  `trace.bin`.
- **HW smoke (cheap, disposable):** a tiny known batch plan on the NPU; confirm
  full coverage and exact labeling.

## Boundaries and non-goals

- The three audited primitives stay external black boxes; we do not rewrite the
  binary decoder, the XRT runner, or the register patcher.
- The held-level span layer (I-2) stays deferred; when reached, it uses the
  upstream-authoritative decoder, not the in-tree span builder.
- Catalog-discovery orchestration is not built out (the engine enables it; we
  seed add_one from the known active set).
- Single column, NPU1/AIE2, `add_one_using_dma` as the validation kernel.
