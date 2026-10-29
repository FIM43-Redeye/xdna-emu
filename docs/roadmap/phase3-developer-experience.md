# Phase 3: Developer Experience

**Goal**: Make debugging and profiling excellent.

**Status**: GUI exists (needs debugging features, disconnected from live emulator state)

---

## Overview

The visual debugger should provide:
- Breakpoints, watchpoints, stepping
- Performance profiling and hot path identification
- Data flow visualization
- Trace recording and replay

---

## 3.1 Debugging

| Task | Status | Notes |
|------|--------|-------|
| Breakpoints (PC, memory access, lock) | TODO | |
| Watchpoints (memory write triggers) | TODO | |
| Step into/over/out | TODO | |
| Call stack visualization | TODO | |
| Register inspection with symbolic names | TODO | |

---

## 3.2 Profiling

| Task | Status | Notes |
|------|--------|-------|
| Cycle counts per function | TODO | |
| IPC (instructions per cycle) analysis | TODO | |
| Stall analysis (memory, lock, DMA) | TODO | |
| Hot path identification | TODO | |
| Vector utilization metrics | TODO | |

---

## 3.3 Visualization

| Task | Status | Notes |
|------|--------|-------|
| Execution timeline (all cores) | TODO | |
| DMA transfer visualization | TODO | |
| Lock contention graph | TODO | |
| Memory access heatmap | TODO | |
| Data flow animation | TODO | |

---

## 3.4 Trace & Replay

| Task | Status | Notes |
|------|--------|-------|
| Record execution trace | TODO | Emulator produces Perfetto JSON via --trace |
| Replay from trace file | TODO | |
| Compare traces (emulator vs hardware) | VERIFIED | Rust `trace-compare` binary; gen-2 trace stack (`tools/mlir-trace-inject.py` + `cpp_trace_patch.py` + `parse-trace.py` + `trace-sweep.py` + `trace_decoder/`); legacy v1 in `tools/deprecated/` |
| Export to VCD/FST for waveform viewers | TODO | |

---

## Current State

The visual layer exists in `src/visual/`:
- `app.rs` - Main application
- `tile_grid.rs` - Tile array view
- `tile_detail.rs` - Detail panels
- `controls.rs` - Run/Step/Reset
- `memory_view.rs` - Hex viewer

Uses `egui` (eframe) for cross-platform GUI.

---

## Resources

- **egui**: https://github.com/emilk/egui
- **VCD format**: Value Change Dump for waveform viewers
