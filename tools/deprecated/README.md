# Deprecated Trace Tools

These tools predate mlir-aie's declarative trace IRON API (PR #2988 / commit
e4f35d643c, merged 2026-03-30). The mainline cycle-capture pipeline now uses
`tools/mlir-trace-inject.py` + `bridge-runner/bridge-trace-runner` +
`tools/parse-trace.py`, which delegate the heavy lifting to mlir-aie's
`aie.utils.trace` module (`configure_trace`, `start_trace`, `parse_trace`,
`get_cycles_summary`).

## Why we kept these files

The declarative API covers our needs but may not cover every one-off use case
the old tools served. We keep the source here, outside the default tooling
path, so anyone needing a dropped capability can:

1. Open the old tool to see how it did the thing
2. Either port that capability into the mainline pipeline or run the old tool
   directly (it still works -- it just isn't invoked by default)

## Contents

The `-v1` suffix marks tools whose base name is also used by a current
gen-2 tool at the top of `tools/`. The names are otherwise easy to confuse;
the v1 ones here are the originals, not the active ones.

| File | Original purpose |
|------|------------------|
| `trace-inject.py` | Inject trace routing into MLIR (capacity planner, collision-aware IDs) |
| `trace-sweep-v1.py` | Multi-batch event sweep orchestrator (superseded by `tools/trace-sweep.py`) |
| `trace-trim.py` | Strip sentinel padding from raw trace buffers |
| `trace-merge.py` | Merge per-batch Perfetto JSON with TRUE anchor alignment |
| `trace-patch-events-v1.py` | Patch event slots in compiled insts.bin without recompilation (superseded by `tools/trace-patch-events.py`) |
| `trace-bridge.sh` | End-to-end trace comparison shell driver, superseded by `scripts/emu-bridge-test.sh` |
| `trace-compare.py` | Python HW/EMU comparator, superseded by `src/bin/trace_compare.rs` (the Rust binary at `target/release/trace-compare`) |

## Do not add new callers

Any new tracing feature belongs in the mainline pipeline, not here. If the
mainline pipeline lacks a capability you need, extend it -- don't reach back
for these.
