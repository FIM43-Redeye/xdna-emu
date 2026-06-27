# Multi-Column Trace Capture (SP1) — Design

**Status:** Approved (design), pending implementation plan.

**Goal:** Make the trace-capture pipeline capture kernels that span more than one
hardware column, validated end-to-end on `two_col`, and fold in the mechanical
kernel-agnostic bits (insts discovery, `start_col` from the dump, dump fallback)
so capture stops being wired to a single kernel's placement.

**Non-goal (deferred to SP2):** kernel-agnostic *anchoring* — auto-selecting and
threading the cross-run anchor for kernels without a standard core anchor. SP1
keeps the existing anchor (`1|2|0|PERF_CNT_2`), which is valid for `two_col`
(it has an active core at absolute `1|2`).

---

## Background

The trace-capture pipeline (`tools/`) currently assumes a single traced column.
A real-HW shakedown of the integrated timeline engine surfaced this: `two_col`
emits trace events on absolute columns 1 **and** 2, and capture aborts on the
first column-2 event with `CaptureError: foreign column 2 (traced 1)`.

Investigation showed the single-column assumption is **narrow**. The NPU already
routes every column's trace into one shared DDR buffer (packet-routed; each
traced tile gets a unique packet ID, ≤31 tiles total — a hardware ceiling, not
ours to build). And almost the whole pipeline is already multi-column:

- **Decoder** extracts `col` from packet-header bits and de-interleaves per tile
  (`tools/trace_decoder/`). That is *why* the probe decoded a column-2 event at
  all — only the label layer rejected it.
- **Patcher** (`trace-patch-events.py --multi-tile`) already patches tiles across
  columns.
- **Event enumeration** (`inference/selfmodel.py::enumerate_configured_events`)
  already spans all columns of the dump.
- **Trace routing** is baked into the xclbin at compile time; `two_col`'s binary
  already egresses column 2.

The entire blocker is the labeling layer in `tools/trace_capture.py`.

## The three single-column assumptions

1. **`label_events` foreign-column guard** (`trace_capture.py`): raises
   `CaptureError` when a decoded event's `col != traced_col`.
2. **`label_map` is column-free** — keyed `(pkt_type, row, slot)`. This was a
   deliberate choice to bridge relative↔absolute column spaces without
   re-deriving `start_col`. It **collides** when two columns share a
   `(pkt_type, row)`: `two_col` has cores at absolute `1|2` and `2|2`, both
   writing slot 0 → key `(0, 2, 0)` written twice, second clobbers first.
3. **Single `traced_col` parameter** threaded through `capture()` and
   `HwInstrument`, plus the relative↔absolute reconcile that the column-free map
   currently hides.

---

## Column spaces

Two column spaces coexist and must be reconciled:

- **Relative col** — the MLIR/insts.bin logical column. What the **patcher**
  consumes. (`two_col`'s columns are relative 0 and 1.)
- **Absolute col** — the hardware partition column. What the **decoder** reports.
  (`two_col` placed at the partition starting `start_col=1` → absolute 1 and 2.)

`absolute = relative + start_col`.

## Architecture (Approach A): absolute-keyed map, single reconcile point

The batch is expressed in **absolute** column space (the engine's natural space —
`enumerate_configured_events` already emits absolute keys). The reconcile happens
in exactly one place, `configure_batch`, which emits:

- the **`patch_spec`** in **relative** col (`abs - start_col`) for the patcher, and
- the **`label_map`** keyed **`(pkt_type, row, abs_col, slot)`** for the decoder side.

`label_events` then does a pure absolute-col lookup — no `traced_col`, no guard,
no `start_col` arithmetic. `HwInstrument` stops doing its current absolute→relative
conversion (it passes absolute batches straight through; `configure_batch` owns
the conversion).

```
batch (ABSOLUTE col)
        │
        ▼
configure_batch(batch, start_col)
   ├── patch_spec  (RELATIVE col = abs - start_col)  ──► patcher ──► HW run ──► trace.bin
   └── label_map   keyed (pkt, row, ABS col, slot)                              │
                                                          decoder (parse_trace) ▼
                                                          raw events (ABSOLUTE col)
                                                                    │
                                                                    ▼
                                              label_events(raw, label_map)
                                              pure (pkt,row,abs_col,slot) lookup
                                                                    │
                                                                    ▼
                                              trace.events.json (ABSOLUTE col)
```

### Why not the alternatives

- **Relative-keyed map, reconcile at lookup** (rejected): `label_events` would
  convert each decoded `abs→rel` via `start_col`, spreading the col-space logic
  across two functions. Approach A keeps it in one.
- **Keep the column-free map** (non-viable): collides on `two_col`. This is the
  root defect, not an option.

---

## Components

### 1. Multi-column label layer
`tools/trace_capture.py`, `tools/inference/hw_instrument.py`.

- `configure_batch(batch, anchor, mode, start_col)`: parse the absolute col from
  each `"col|row|pkt"` key; emit `patch_spec` entries with relative col
  (`abs - start_col`); build `label_map[(pkt, row, abs_col, slot)] = name`.
- `label_events(raw_events, label_map)`: drop the `traced_col` parameter and the
  foreign-column guard; look up each decoded event by `(pkt, row, col, slot)`
  (col is already absolute from the decoder).
- `capture(plan, runner, *, test, out_dir, start_col, trace_size, instr, ...)`:
  replace `traced_col` with `start_col`; pass it to `configure_batch`; call
  `label_events(raw, lmap)`.
- `HwInstrument`: pass absolute batches to `capture()` unchanged; pass `start_col`
  (drop the absolute→relative tile conversion now owned by `configure_batch`).

### 2. insts discovery
`tools/trace_capture.py::_discover_xclbin_insts(test, compiler)`.

Resolution order: (a) `insts.bin` if present; (b) parse the test's `run.lit`
for `--npu-insts-name=<name>`; (c) glob for a single `*.bin` in the build dir.
Error clearly if none / ambiguous. (mlir-aie uses ~5 insts filenames.)

### 3. start_col in the dump
`examples/dump_config_json.rs`, `tools/config_extract/dump_model.py`,
`tools/inference/run_experiment.py`.

- Rust: emit the partition `start_col` into the `ConfigDump` JSON.
- Python `dump_model`: read `start_col` (backward-compatible — older fixtures
  without the field fall back to `None`).
- `run_experiment`: prefer `dump.start_col` when present, else the config value
  (default 1). Removes the hardcoded placement assumption at its source.

### 4. dump fallback
`tools/inference/run_experiment.py`.

When `dump_path` is None, try `config_extract/fixtures/{test}.config.json` before
degrading to `dump=None`.

### 5. Validation
Offline unit tests + one HW integration run (see Testing).

---

## Error handling

The foreign-column guard is removed; the **unconfigured-slot invariant stays and
becomes the single uniform rule**: every decoded event must resolve in
`label_map`, or `label_events` raises `CaptureError` naming `(col, row, pkt,
slot)`. This preserves the safety net that caught the mode-1 EVENT_PC
phantom-fire bug (a mis-decoded stream surfaces as unconfigured slots), now as
one rule instead of two guards.

If `two_col` surfaces a partition tile with compile-time trace that we did not
configure (so its events don't resolve in `label_map`), we handle it then by
NONE-disabling that tile's slots — we do **not** pre-build a tolerant
drop-unconfigured path (YAGNI; a silent drop would mask real decode errors).

---

## Testing

**Offline (no HW), the bulk of the gate:**
- `configure_batch` / `label_events` with synthetic decoded events spanning two
  columns sharing a `(pkt,row)` (the `two_col` collision case): assert both
  columns' events label correctly, no collision, absolute-col lookup, and that an
  unconfigured `(col,row,pkt,slot)` raises `CaptureError`.
- `configure_batch` emits `patch_spec` in relative col and `label_map` in
  absolute col for a given `start_col`.
- `_discover_xclbin_insts`: synthetic build dirs exercising each resolution branch
  (`insts.bin`, `run.lit` parse, single `*.bin`, ambiguous/none error).
- `dump_model` reads `start_col`; absent field → `None` (backward-compat).
- `run_experiment` dump fallback by test name.

**Integration (HW, box-agnostic, structural):**
- One `two_col` capture through `run_experiment`: assert a well-formed
  multi-column `IntegratedTimeline` — tracks spanning both columns, cross-domain
  edges only between fired events, no dangling endpoints, census computed. This is
  structural (not determinism-fidelity), so it passes on any box.

**Regression:** `cargo test --lib` (Rust dump change) and the offline Python
inference suite stay green; no new failures.

---

## Definition of Done

`two_col` captures with no foreign-column error and assembles a well-formed
multi-column `IntegratedTimeline` (column-spanning tracks, valid cross-domain
edges, no dangling endpoints, census computed); offline suite green;
`cargo test --lib` green. Box-agnostic — no quiet-host dependency.
