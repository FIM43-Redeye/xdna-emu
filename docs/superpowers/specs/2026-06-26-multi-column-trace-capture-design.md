# Multi-Column Trace Capture (SP1) — Design

**Status:** Approved (design), revised after agentic review, pending implementation
plan.

**Goal:** Make the trace-capture pipeline capture kernels that span more than one
hardware column, validated end-to-end on `two_col`, and fold in the mechanical
kernel-agnostic bits (insts discovery, `start_col` from the dump, dump fallback)
so capture stops being wired to a single kernel's placement.

**Non-goal (deferred to SP2):** kernel-agnostic *anchoring* — auto-selecting and
threading the cross-run anchor for kernels without a standard core anchor. SP1
keeps the existing anchor (`1|2|0|PERF_CNT_2`), which is valid for `two_col`
(it has an active core at absolute `1|2`).

**Immediate follow-up (its own sub-project, right after SP1):** cross-column edge
*grounding* — making the two columns causally connect in the timeline (see
"Connectivity" below). SP1 produces a multi-column timeline that is honest about
the columns being disconnected; connecting them is the next step, not someday.

---

## Background

The trace-capture pipeline (`tools/`) currently assumes a single traced column.
A real-HW shakedown of the integrated timeline engine surfaced this: `two_col`
emits trace events on absolute columns 1 **and** 2, and capture aborts on the
first column-2 event with `CaptureError: foreign column 2 (traced 1)`.

Investigation showed the single-column assumption is **narrow**. The NPU already
routes every column's trace into one shared DDR buffer (packet-routed; each
traced tile gets a unique packet ID). Almost the whole pipeline is already
multi-column:

- **Decoder** extracts `col` from packet-header bits and de-interleaves per tile
  (`tools/trace_decoder/`). That is *why* the probe decoded a column-2 event at
  all — only the label layer rejected it.
- **Patcher** (`trace-patch-events.py --multi-tile`) already patches tiles across
  columns.
- **Event enumeration** (`inference/selfmodel.py::enumerate_configured_events`)
  already spans all columns of the dump (emits absolute keys `col+start_col`).
- **Trace routing** is baked into the xclbin at compile time; `two_col`'s binary
  already egresses column 2.
- **The timeline engine** is already multi-column-ready: `coupling_oracle` applies
  `start_col` to both endpoints, `weave`/`ground_edge` treat every `(col,row,pkt)`
  as an independent domain, and `generate_ledger` reachability spans columns.

The blocker is the labeling layer in `tools/trace_capture.py`.

## The three single-column assumptions

1. **`label_events` foreign-column guard** (`trace_capture.py`): raises
   `CaptureError` when a decoded event's `col != traced_col`.
2. **`label_map` is column-free** — keyed `(pkt_type, row, slot)`. This was a
   deliberate choice to bridge relative↔absolute column spaces without
   re-deriving `start_col`. It **collides** when two columns share a
   `(pkt_type, row)`. Confirmed in `two_col`: cores at absolute `1|2` and `2|2`
   (both pkt 0, slot 0 → key `(0,2,0)` written twice), and likewise shims at
   `1|0`/`2|0` and memtiles at `1|1`/`2|1`. The absolute-keyed map is genuinely
   necessary.
3. **Single `traced_col` parameter** threaded through `capture()` and
   `HwInstrument` (and `KernelConfig`/CLI), plus the relative↔absolute reconcile
   that the column-free map currently hides.

---

## Column spaces

Two column spaces coexist and must be reconciled:

- **Relative col** — the MLIR/insts.bin logical column. What the **patcher** and
  the **`probe_slot_capacity`** gate consume. (`two_col`'s columns are relative 0
  and 1.)
- **Absolute col** — the hardware partition column. What the **decoder** reports.
  (`two_col` placed at the partition starting `start_col=1` → absolute 1 and 2.)

`absolute = relative + start_col`.

## Architecture (Approach A): absolute-keyed map, single reconcile point

The batch is expressed in **absolute** column space (the engine's natural space —
`enumerate_configured_events` already emits absolute keys). The
**patcher-facing** reconcile happens in exactly one place, `configure_batch`,
which emits:

- the **`patch_spec`** in **relative** col (`abs - start_col`) for the patcher, and
- the **`label_map`** keyed **`(pkt_type, row, abs_col, slot)`** for the decoder side.

`label_events` then does a pure absolute-col lookup — no `traced_col`, no guard,
no `start_col` arithmetic.

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

**Critical: `HwInstrument` keeps a *separate* abs→rel mapping for the
`probe_slot_capacity` gate.** `HwInstrument` uses its current abs→rel conversion
for two independent purposes: (a) building the plan/anchor for `build_active_plan`
— this moves into `configure_batch`; and (b) the drop-untraceable-tile gate,
which calls `probe_slot_capacity(insts_bytes, col, row, tile_type)` with the
**relative** col because `insts.bin` is in relative-col space. Purpose (b) must
**stay** in `HwInstrument` (it has `self._start_col`). Dropping it wholesale would
feed absolute cols to the probe → wrong offsets → every tile returns capacity 0 →
silently dropped → empty capture. The patcher-facing conversion moves to
`configure_batch`; the probe-facing rel-col mapping stays.

### Why not the alternatives

- **Relative-keyed map, reconcile at lookup** (rejected): `label_events` would
  convert each decoded `abs→rel` via `start_col`, spreading the col-space logic
  across two functions. Approach A keeps the patcher-facing reconcile in one place.
- **Keep the column-free map** (non-viable): collides on `two_col`. This is the
  root defect, not an option.

---

## Components

### 1. Multi-column label layer
`tools/trace_capture.py`, `tools/inference/hw_instrument.py`.

- `configure_batch(batch, anchor="PERF_CNT_2", mode=0, start_col=0)`: parse the
  absolute col from each `"col|row|pkt"` key; emit `patch_spec` entries with
  relative col (`abs - start_col`); build `label_map[(pkt, row, abs_col, slot)]
  = name`. **`start_col` keeps a default** so `selfmodel.legal_batch`'s
  `configure_batch(batch.tiles)` call (slot/name legality only — column-agnostic)
  keeps working.
- `label_events(raw_events, label_map)`: drop the `traced_col` parameter and the
  foreign-column guard; look up each decoded event by `(pkt, row, col, slot)`
  (col is already absolute from the decoder).
- `capture(plan, runner, *, test, out_dir, start_col, trace_size, instr, ...)`:
  replace `traced_col` with `start_col`; pass it to `configure_batch`; call
  `label_events(raw, lmap)`.
- `HwInstrument`: pass absolute batches to `capture()` and pass `start_col`;
  **retain** the abs→rel mapping for the `probe_slot_capacity` gate (see the
  Critical note above).

### 2. insts discovery
`tools/trace_capture.py::_discover_xclbin_insts(test, compiler)`.

Resolution order: (a) `insts.bin` if present; (b) parse the test's `run.lit`
for `--npu-insts-name=<name>`; (c) glob for a single `*.bin` in the build dir.
Error clearly if none / ambiguous. (mlir-aie uses ~5 insts filenames.)

### 3. start_col in the dump
`examples/dump_config_json.rs`, `tools/config_extract/dump_model.py`,
`tools/inference/run_experiment.py`.

- Rust: source `start_col` from the **xclbin partition metadata**
  (`AiePartition::start_columns()[0]`, the leftmost candidate placement), NOT from
  `state.start_col` (which `load_state_from_xclbin` leaves at its `0` default — a
  trap: emitting 0 and "preferring dump" would override the correct default and
  break anchoring). Emit it into the `ConfigDump` JSON.
- Python `dump_model`: read `start_col` via `.get()` (backward-compatible — older
  fixtures without the field → `None`).
- `run_experiment`: use `dump.start_col` only when it is **present and non-zero**;
  otherwise fall back to the config value (default 1). Document that the dump
  value is the candidate-list head, not a runtime-placement guarantee — the
  driver picks the actual column at load time (leftmost-available, = 1 in
  practice for all current tests).

### 4. dump fallback
`tools/inference/run_experiment.py`.

When `dump_path` is None, try `config_extract/fixtures/{test}.config.json` before
degrading to `dump=None`.

### 5. Deprecate the relative-col `run_loop` path
`tools/trace_capture.py::run_loop`, `tools/capture_infer_smoke.py`,
`SEED_ACTIVE_PLAN`.

`run_loop` is a live caller of `capture()` that builds its plan from
`SEED_ACTIVE_PLAN` (relative-col, add_one-specific) and passes `traced_col`.
After Approach A, `capture()` expects absolute batches + `start_col`; feeding it
relative batches would compute `abs - start_col = -1`. `run_loop`/`SEED_ACTIVE_PLAN`
are dead on the live `run_experiment` path (only `capture_infer_smoke.py` uses
them). Deprecate/remove them rather than leave a silent breakage. (This also
clears the `SEED_ACTIVE_PLAN` cleanup that was nominally SP2's.)

### 6. Validation
Offline unit tests + a synthetic two-column timeline fixture + one HW integration
run (see Testing).

---

## Connectivity (the substantive multi-column question)

`coupling_oracle(dump, start_col)` surfaces `two_col`'s cross-column couplings
(`inter_tile` core-port edges → absolute pairs `1|2~2|2`, `1|4~2|4`). But `weave`
only forms a `CrossTrackEdge` when both endpoints fire **and** resolve to the
relevant ports, and the events that actually fire on compute tiles
(`INSTR_VECTOR`, stalls, DMA) mostly do **not** resolve to the core ports those
cross-column couplings use. The likely outcome (HW-gated — unconfirmed until a
real `two_col` capture) is that the two columns appear as causally **disconnected
island-clusters** of tracks.

**SP1 policy (decided): disconnected-but-honest.** SP1 requires that every
cross-column coupling from the oracle is either grounded as a `CrossTrackEdge`
**or** surfaced as a `connectivity_defect:<a>~<b>` flag — **never silently
dropped**. SP1 does not require the columns to causally connect. Actually
*connecting* them (cross-column edge grounding — understanding `two_col`'s real
cross-column dataflow and fixing port resolution for `inter_tile` edges so `weave`
can bridge) is the **immediate follow-up sub-project**, taken up right after SP1.

---

## Error handling

The foreign-column guard is removed; the **unconfigured-slot invariant stays and
becomes the single uniform rule**: every decoded event must resolve in
`label_map`, or `label_events` raises `CaptureError` naming `(col, row, pkt,
slot)`. This preserves the safety net that caught the mode-1 EVENT_PC
phantom-fire bug (a mis-decoded stream surfaces as unconfigured slots), now as
one rule instead of two guards.

`build_active_plan` includes every route-graph-active tile in every batch and
`configure_batch` pads to 8 NONE slots, so any route-graph-active tile's
compile-time trace is wiped. **Residual risk (HW-gated, expected-possible
first-run failure, not a settled outcome):** a tile carrying compile-time trace
that is *not* in the route graph would never be NONE-disabled, so its events
would decode and trip the unconfigured-slot error on the first `two_col` run. We
handle that reactively by NONE-disabling the offending tile — we do **not**
pre-build a tolerant drop-unconfigured path (YAGNI; a silent drop would mask real
decode errors).

---

## Anchoring note (cross-column, single buffer)

One core anchor suffices for multi-column — no SP2 dependency. All columns
packet-route into one shared DDR buffer in a single HW run, so column-2 events
are captured in the *same batch* as the column-1 anchor and anchored against the
same anchor soc. The only residual: a batch where the anchor (`PERF_CNT_2`)
doesn't fire drops that batch's column-2 events too (handled as
dropout/capture-health, not a correctness bug). The hardcoded
`ANCHOR = "1|2|0|PERF_CNT_2"` (in `verifier.py`, `loader.py`, `generator.py`,
`trace_join.py` defaults) is correct for `two_col` and stays; generalizing it is
SP2's concern.

## Capacity note

`two_col` has ~11 active tiles / ~18 candidate trace modules — comfortably under
the 31-packet-ID ceiling. `build_active_plan` yields **2 batches** (the shim
`_MENU` has 9 events > 8 slots; everything else ≤ 8). The real per-tile limiter
is `probe_slot_capacity` (only tiles the xclbin compiled trace on are traceable),
not the 31-packet ceiling. The 2 MB default trace buffer is plausibly sufficient
(~4× add_one's module count) but is HW-unverified.

---

## Testing

**Offline (no HW) — the regression gate:**
- **Multi-column label layer:** `configure_batch` / `label_events` with synthetic
  decoded events spanning two columns sharing a `(pkt,row)` (the `two_col`
  collision): assert both columns' events label correctly, no collision,
  absolute-col lookup, `patch_spec` in relative col, `label_map` in absolute col
  for a given `start_col`, and that an unconfigured `(col,row,pkt,slot)` raises
  `CaptureError`.
- **Synthetic two-column timeline fixture (makes the headline DoD offline-provable):**
  hand-built two-column `trace.events.json` run dirs (cols 1 and 2, with a
  cross-column coupling in the dump) → `assemble_timeline` → assert the
  **discriminating predicate**: `{t.domain.split("|")[0] for t in tl.tracks}`
  contains both `"1"` and `"2"`; intra-column tracks well-formed; and each
  cross-column oracle coupling is either a `CrossTrackEdge` or a
  `connectivity_defect` flag (never absent). This turns the multi-column claim
  into something the offline gate proves and regression-locks; HW is then only
  for the capture smoke.
- `_discover_xclbin_insts`: synthetic build dirs exercising each resolution branch
  (`insts.bin`, `run.lit` parse, single `*.bin`, ambiguous/none error).
- `dump_model` reads `start_col`; absent field → `None` (backward-compat);
  `run_experiment` uses it only when present-and-non-zero, else config default.
- `run_experiment` dump fallback by test name.

**Blast radius to update in lockstep** (the rename must be complete, not half):
`KernelConfig.traced_col` + `--traced-col` CLI (`run_experiment.py`),
`HwInstrument.__init__` (`hw_instrument.py`), `canary_witness.py` (constructs
`KernelConfig`), and tests `test_trace_capture.py` (3-tuple `label_map` asserts +
the foreign-column-hard-error test being deleted), `test_hw_instrument.py`
(`fake_capture` signature), `test_experiment_loop_hw.py`,
`test_experiment_report.py`, `test_timeline.py`.

**Integration (HW, box-agnostic, structural):**
- One `two_col` capture through `run_experiment`: completes with no foreign-column
  or unconfigured-slot error and assembles a timeline whose tracks cover **both**
  columns 1 and 2, with cross-column couplings grounded-or-flagged. Structural
  (not determinism-fidelity), so it passes on any box.

**Regression:** `cargo test --lib` (Rust dump change) and the offline Python
inference suite stay green; no new failures. The 4 committed fixtures need **no**
regeneration to keep working (absent `start_col` → config default 1, correct for
all); `two_col.config.json` is regenerated **with `start_col=1`** (sourced per
Component 3) so the "dump.start_col preferred when present" path is exercised.

---

## Definition of Done

`two_col` captures with no foreign-column or unconfigured-slot error and assembles
a well-formed `IntegratedTimeline` whose **track set covers both column 1 and
column 2**, with intra-column structure correct and every cross-column oracle
coupling grounded-or-flagged (never silently dropped); the discriminating
multi-column predicate is **also proven offline** against a synthetic two-column
fixture; offline suite green; `cargo test --lib` green. Box-agnostic — no
quiet-host dependency. (Cross-column edge *grounding* is the immediate follow-up
sub-project, not part of SP1's DoD.)
