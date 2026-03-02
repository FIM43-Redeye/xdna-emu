# Design: Automated NPU Ground Truth Comparison

**Date**: 2026-03-01
**Status**: Approved

## Problem

The emulator needs automated comparison against real NPU behavior. Currently,
trace comparison is manual (visual inspection in Perfetto) or requires the
full 13-group sweep (slow). We need a lightweight, always-on ground truth
pipeline that captures what the NPU does, strips nondeterministic noise, and
compares emulator output against it -- every run, automatically.

## Three Layers of Ground Truth

1. **Functional output** -- DMA output buffer contents (the computation result)
2. **Event sequence** -- ordered list of deterministic events per tile
3. **Stable timing deltas** -- inter-event cycle gaps for deterministic events

Each layer is compared independently. A test can match on output but diverge
on events (semantic correctness, wrong instruction path) or match on events
but diverge on timing (correct behavior, wrong cycle model).

## Canonical Trace Format

Stored at `build/ground-truth/<test>/`:

```
canonical.json     -- events + deltas, nondeterminism stripped
output.bin         -- raw DMA output buffer(s)
metadata.json      -- capture context (timestamp, driver, kernel hash, config)
```

### canonical.json structure

```json
{
  "version": 1,
  "event_config": "base8",
  "tiles": {
    "core(2,0)": {
      "events": [
        {"name": "INSTR_VECTOR", "delta": 0},
        {"name": "INSTR_LOAD", "delta": 3},
        {"name": "LOCK_ACQUIRE_REQ", "delta": 1}
      ],
      "filtered_events": ["LOCK_STALL", "MEMORY_STALL"]
    }
  }
}
```

### metadata.json structure

```json
{
  "captured_at": "2026-03-01T14:30:00Z",
  "kernel_hash": "sha256:abc123...",
  "driver_version": "6.19.0-custom+",
  "event_config": "base8",
  "npu_device": "NPU1",
  "capture_reps": 1
}
```

## Workflow

Fully automatic, no special flags needed:

```
For each test in npu-test:
  1. Run on NPU -> raw trace packets + output buffers
  2. Canonicalize: filter known-nondeterministic events, compute deltas
  3. If prior ground truth exists:
     a. Compare new NPU canonical vs stored (detect firmware/driver drift)
     b. Report drift (informational, not failure)
  4. Overwrite ground truth with new capture
  5. Run emulator on same binary
  6. Canonicalize emulator trace (same pipeline)
  7. Compare emulator canonical vs ground truth
  8. Report verdict per layer
```

### Event Configuration

- **Default (base 8)**: INSTR_VECTOR, INSTR_LOAD, INSTR_STORE, INSTR_CALL,
  PORT_RUNNING_0, PORT_RUNNING_1, LOCK_STALL, ACTIVE. Single NPU run.
- **Full sweep (`--full`)**: All 13 event groups, statistical classification
  via existing `check_determinism()` from trace_sweep.rs.

## Canonicalization Pipeline

```
Raw trace packets (binary from NPU or emulator trace buffer)
  |  parse.py (NPU) or emulator trace emitter
  v
Perfetto JSON events per tile
  |  load into Vec<TraceEvent>
  v
Filter: remove known-nondeterministic events
  |  hardcoded list for single-run: LOCK_STALL, MEMORY_STALL,
  |  STREAM_STALL, CASCADE_STALL
  |  (--full mode: statistical classification via SlotClass)
  v
Keep only Active events, sorted by timestamp
  |
  v
Compute deltas: delta[i] = ts[i] - ts[i-1], delta[0] = 0
  |
  v
CanonicalTrace { tiles: HashMap<TileId, TileTrace> }
```

## Comparison Logic

Three independent comparisons, each reported separately:

| Layer   | Compared                      | Pass criteria            |
|---------|-------------------------------|--------------------------|
| Output  | byte-for-byte DMA buffers     | Identical                |
| Events  | ordered event names per tile  | Same sequence            |
| Timing  | inter-event deltas            | Within tolerance (default 0) |

## Reporting

### Per-test verdict line

```
matrix_multiply .............. PASS  [GT: MATCH]
vec_vec_add .................. PASS  [GT: MATCH (drift: timing +2cy core(2,0))]
add_314_using_dma ............ VFAIL [GT: DIVERGE output]
passthrough .................. PASS  [GT: NEW (first capture)]
```

### Verdicts

- **MATCH** -- all three layers identical
- **MATCH (drift: ...)** -- emulator matches, but NPU-vs-stored shows changes
- **DIVERGE output** -- DMA buffers differ (most serious)
- **DIVERGE events** -- event sequence differs (semantic bug)
- **DIVERGE timing** -- stable deltas differ beyond tolerance
- **NEW** -- no prior ground truth, captured fresh

### Divergence detail (on failure)

```
  DIVERGE events in core(2,0):
    ground truth: INSTR_VECTOR, INSTR_LOAD, LOCK_ACQ, INSTR_VECTOR
    emulator:     INSTR_VECTOR, INSTR_LOAD, INSTR_VECTOR
    ^ missing LOCK_ACQ at position 2

  DIVERGE output:
    first difference at byte 128: expected 0x42, got 0x00
    (16 of 1024 bytes differ)
```

## Code Location

- `src/testing/ground_truth.rs` -- canonical trace types, serde, comparison
- `src/testing/xclbin_suite.rs` -- capture/compare integration
- `src/bin/npu_test.rs` -- `--full` flag, verdict reporting

## Design Decisions

- **JSON via serde** -- `Serialize`/`Deserialize` on canonical types. Swap to
  bincode by changing one line if perf matters later.
- **Always capture from NPU** -- no "if exists skip" logic. Runs are cheap.
- **Ground truth = latest NPU capture** -- overwritten each run. Promotion to
  version-controlled `tests/ground-truth/` is a future option.
- **Reuse trace_sweep.rs** -- SlotClass enum and classification logic already
  exist. Extract and reuse for `--full` mode.
- **Hardcoded nondeterministic list for single-run** -- avoids needing multiple
  NPU runs for the common case. The four stall events are empirically known
  to be nondeterministic.
