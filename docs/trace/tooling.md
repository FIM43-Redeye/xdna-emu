# Tracing Ecosystem -- Tool Reference

Binary trace comparison between emulator and real NPU hardware. All traces
converge to Perfetto JSON (viewable at ui.perfetto.dev).

For the division-of-labor summary and strategy context, see CLAUDE.md's
"Tracing Ecosystem" section and [`strategy.md`](strategy.md). This file is
the per-tool inventory of the active pipeline.

## Active pipeline (six layers, executed top to bottom)

1. **Pre-build** -- one-shot per test before aiecc runs.
   | Tool | Purpose |
   |------|---------|
   | `tools/trace-prepare.py` | Compile-side prep: injects trace MLIR via mlir-trace-inject, patches `test.cpp` via cpp_trace_patch. Used by `scripts/emu-bridge-test.sh`. |
   | `tools/mlir-trace-inject.py` | Declarative MLIR injector, calls mlir-aie's IRON API. |
   | `tools/cpp_trace_patch.py` | Tree-sitter C++ transformer for `test.cpp` (trace BO alloc + set_arg). |
   | `tools/trace_config.py`, `tools/trace_config_schema.json`, `tools/trace_config_examples/` | Config layer (events per tile, mode, packet IDs). |

2. **Run** -- per event-set, swaps event slots without recompiling.
   | Tool | Purpose |
   |------|---------|
   | `bridge-runner/bridge-trace-runner` | C++ multi-batch orchestrator (HW + EMU). Supports `--batch-stdin` (RESET command for worker reuse) and `--snapshot-on-timeout <dir>` (captures CORE/DMA/lock register state on `run.wait` timeout, before driver recovery wipes it). |
   | `tools/trace-sweep.py` | Gen-2 multi-tile sweep, 8-event batches per tile. |
   | `tools/trace-patch-events.py` | Gen-2 patcher: rewrites event-slot bytes in compiled `insts.bin`. |

3. **Decode** -- raw trace buffer to events JSON / cycles / Perfetto.
   | Tool | Purpose |
   |------|---------|
   | `tools/parse-trace.py` | Single-source decoder, wraps mlir-aie's `parse_trace`. Emits any combination of flat events JSON, cycles scalar, raw Perfetto, raw command stream. |
   | `tools/trace_decoder/` | In-tree decoder backend (default; alternate to upstream parser). |

4. **Compare** -- HW vs EMU events.
   | Tool | Purpose |
   |------|---------|
   | `src/bin/trace_compare.rs` | Rust comparator (event-sequence diff with anchor alignment, configurable tolerance, per-tile MATCH/DRIFT/ORDER_MISMATCH/MISSING_EVENT verdict). Consumes events JSON from parse-trace.py. |

5. **Matrix / regression** -- drive across (test x compiler x tile).
   | Tool | Purpose |
   |------|---------|
   | `scripts/trace-sweep-all.py` | Drives trace-sweep.py across the bridge matrix. |
   | `scripts/show-sweep-matrix.py` | Renders + diffs sweep result trees. |
   | `scripts/merge-sweep-results.py` | Post-hoc merge of sweep result dirs. |
   | `scripts/trace-quarantine.txt` | Tests to skip in trace mode (deadlocks, IOMMU faults). |
   | `scripts/trace-incompat-tests.txt` | Tests structurally incompatible with trace injection. |

6. **Glue** -- end-to-end orchestration + EMU-side emission.
   | Tool | Purpose |
   |------|---------|
   | `scripts/emu-bridge-test.sh` | End-to-end bridge driver (`--sweep`, `--trace=pc-anchored`, etc.). |
   | `src/device/trace_unit/` | Emulator-side trace unit -- writes the same packet-stream format HW does. |

Build the Rust comparator: `cargo build --release --bin trace-compare`.

## Deprecated tools

Archive at `tools/deprecated/`, kept for reference; do not add new callers.
Files with a `-v1` suffix were renamed during the gen-2 rollout so their base
names stop colliding with current tools at the top of `tools/`:

| Tool | Original purpose |
|------|------------------|
| `tools/deprecated/trace-inject.py` | Inject trace routing into MLIR (capacity planner, collision-aware IDs) |
| `tools/deprecated/trace-sweep-v1.py` | Multi-batch event sweep orchestrator (superseded by `tools/trace-sweep.py`) |
| `tools/deprecated/trace-trim.py` | Strip sentinel padding from raw trace buffers |
| `tools/deprecated/trace-merge.py` | Merge per-batch Perfetto JSON with TRUE anchor alignment |
| `tools/deprecated/trace-patch-events-v1.py` | Patch event slots in compiled insts.bin (superseded by `tools/trace-patch-events.py`) |
| `tools/deprecated/trace-bridge.sh` | End-to-end shell driver, superseded by `scripts/emu-bridge-test.sh` |
| `tools/deprecated/trace-compare.py` | Python HW/EMU comparator, superseded by `src/bin/trace_compare.rs` |

See `tools/deprecated/README.md` for the standdown rationale.
