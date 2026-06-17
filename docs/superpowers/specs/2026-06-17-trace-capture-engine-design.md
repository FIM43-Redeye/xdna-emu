# Self-owned NPU trace-capture engine (task #140) — design

**Date:** 2026-06-17
**Status:** approved design, pre-plan
**Predecessor:** `2026-06-16-cross-batch-trace-join-design.md` and its plan. The
join library (`tools/trace_join.py`, Tasks 1-6) is built and reviewed; this
engine is the execution layer that feeds it correctly-labeled HW data.

## Goal

Build a self-owned capture engine that takes a **batch plan** (which events to
trace on which tile/module in each batch) and produces correctly-labeled
`events.json` per batch per run on real hardware — with observable coverage
(N-run union, gaps named not hidden), candidates co-traced in one execution, and
repeatable N times.

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

## Labeling key is column-free

The decoder reads an **absolute** column from the packet header; the patcher
writes the **absolute** column it is given. They agree only if the engine
patches at the column XRT's runtime allocator picks — and `bridge-trace-runner`
does not expose `start_col`. Rather than guess or "reconcile" a column offset
(an unspecified hand-wave that would turn every event into an "unconfigured
slot" error when `start_col != patch_col`), the engine keys the labeling map on
**`(pkt_type, row, slot)`** and ignores column — the same boundary trace-sweep
already chose for the same reason. Within a single traced column this is
unambiguous (no two configured tiles share a `(pkt_type, row, slot)`). The
decoded `col` is used only as a **sanity guard**: every event in one capture
must carry the one column we traced; a foreign column is a hard error. Multi-
column capture is out of scope (single column, NPU1); it would extend the key,
not break it.

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
   (anchor/grounding first), and the **`(pkt_type, row, slot) -> name`** map is
   recorded as the labeling oracle. **The map is column-free by design** (see
   "Labeling key is column-free" below) — the patch is written at an absolute
   column, but the label lookup keys on `(pkt_type, row, slot)`.
3. **Runner driver** — drives `bridge-trace-runner` over its stdin protocol
   (`--instr --trace-out --trace-size --input --output --ctrlpkt`); issues
   `RESET` between batches/runs -> `trace.bin`. Passes an **explicit generous
   `--trace-size`** (the BO has no backpressure: once full, events are silently
   dropped, and a full co-traced batch is the high-volume case most likely to
   truncate). The driver **detects truncation** (buffer-full / missing
   end-marker) and surfaces it as a *distinct* error from "events did not fire"
   — truncation must never masquerade as a coverage gap. **RESET serializes
   batches:** each RESET tears down and rebuilds the hw_context (the only way to
   zero the cumulative shim-DMA write counter), so batches run sequentially, not
   pipelined. Skipping RESET to "speed up" silently reintroduces the cumulative
   offset — do not.
4. **Raw decoder + exact labeler** — `trace_decoder.parse_trace(words,
   slot_names=None)` -> raw events; labels each by the recorded
   `(pkt_type, row, slot) -> name` map (column-free). Two hard-error guards: a
   decoded `(pkt_type, row, slot)` that is not in the map (we never configured
   it), and a decoded `col` that is not the single column we traced (a
   sanity check on the start_col assumption). The engine **replaces two
   separate trace-sweep labeling paths** — the `pkt 0/1` relabel and the
   MLIR-slot-names path for `pkt 2/3` — with one uniform column-free map across
   all four `pkt_type`s.
5. **Parity guard** (test, not runtime) — decode a representative real
   `trace.bin` with both the in-tree decoder and mlir-aie's `parse_trace`,
   assert raw `(pkt_type, slot, ts)` agreement.

## Per-batch data flow

For each run (1..N), for each batch in the plan:

1. **Resolve events to modules** — the plan's per-tile event lists carry
   `pkt_type` (module-explicit keys), so produce per-`(col,row,module)` slot
   lists directly. Shim (row 0) and memtile (row 1) are single-module.
2. **Assign slots** — fixed order, anchor first (slot 0); record the
   `(pkt_type, row, slot) -> name` map (column-free).
3. **Configure** — build the multi-tile patch spec (event IDs per module),
   call the patcher -> `insts.patched.bin`.
4. **Run** — `RESET` the runner (zeroes the cumulative trace-DMA counter), then
   run the kernel with the patched insts and an explicit generous `--trace-size`
   -> `trace.bin`; detect truncation as a distinct error.
5. **Decode raw** — in-tree decoder, `slot_names=None`.
6. **Label exactly** — `(pkt_type, row, slot)` lookup in the recorded map; hard
   error on a slot we did not configure, and on a foreign column (start_col
   sanity guard).
7. **Write** `run_NN/batch_MM/hw/trace.events.json`.

The inversion from trace-sweep: it decoded first and *guessed* names from MLIR;
we **record the name->slot map at config time and apply it after decode**, so
labeling cannot drift from what we actually traced.

## Coverage & co-tracing guarantees

- **Observable coverage via N-run union (not "reliable").** A configured event
  is not guaranteed to fire in any single run — DMA milestones are stochastic
  and a short trace window may miss some. The engine does **not** pretend
  otherwise. Per run, every configured slot is accounted for (fired-and-labeled,
  or recorded silent-for-this-run); across the N runs the engine **unions** the
  observed events per configured slot and **reports any slot never seen in any
  run** as an explicit coverage gap. A decode to an unconfigured slot, or a
  truncated trace, is a hard error — distinct from a silent-this-run slot. The
  point is that a gap is *visible and named*, never inferred from a relabel
  mismatch — but the engine surfaces the gap, it does not magically fill it.
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

**Discovery bootstrapping:** step 1 (discovery) is **not exercised this cycle.**
For the add_one validation we **seed from the known active set** (already in
hand). Catalog-discovery — "run a broad plan covering each tile-type's full
event catalog in `ceil(catalog/8)` batches, union what fires" — is a capability
the engine structurally enables (it runs whatever plan it is given), but we do
not build a discovery orchestrator now (YAGNI). The wiring diagram lists it for
completeness; this cycle starts from a seeded active set.

## Deliverables this cycle

1. **Library module-explicit retrofit** — `trace_join.py` and its tests keyed by
   `(col,row,pkt_type,name)` (the foundation; the anchor becomes
   `1|2|0|PERF_CNT_2`). This is a **key-layer rewrite, not a find-replace**:
   `_key`/`_split_key` (which hardcodes `split("|", 2)` for 3 fields),
   `anchored_firsts`, `join_run`, ~21 `anchor_key` literals, and ~33 test
   assertions all assume 3-part keys; `sweep_lists`/`load_active_events` group by
   tile and infer module from row, which now must use `pkt_type`. Sequence it as
   real, separately-reviewed work with a full test re-baseline — not a warm-up.
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

## Design-review notes (2026-06-17, Opus senior review)

The core mechanism was verified end-to-end on a real `trace.bin` (114 events
decoded raw, then labeled). Two facts could not be statically proven and are
covered by hard guards rather than assumed:
- **start_col is always the single traced column.** Observed = 1 on the
  FFI/bridge path, but the allocator could differ. Covered by the foreign-column
  hard-error guard (labeling is column-free regardless).
- **A full co-traced batch fits one trace BO.** Untested at full volume.
  Covered by the truncation-detection hard error and an explicit generous
  `--trace-size`.
The review's verdict was sound-with-fixes; all Critical/Important findings are
folded into the sections above.
