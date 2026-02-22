# Design: Trace All Tile Types

**Date**: 2026-02-22
**Status**: Approved
**Goal**: Make trace injection tile-type-agnostic so any NPU workload can be
instrumented for emulator-vs-hardware comparison.

## Context

trace-inject.py currently only traces compute tiles (row >= 2), rejecting
tests that lack them.  The mlir-aie trace API already supports memtile, shim,
and core memory tracing with dedicated event enums, register bases, and packet
types.  We just need to pass all tiles through.

## Approach: "Trace Everything"

Collect every tile in the design.  Designate one shim tile (row 0) as the
trace destination (routes packets to DDR).  Pass all remaining tiles --
compute, memtile, additional shims -- to the existing
`configure_packet_tracing_flow` / `configure_packet_tracing_aie2` API pair.
The API dispatches per tile type internally; no branching in our code.

### Event Selection

Each tile type gets 8 trace events (hardware limit per trace unit).  We use
mlir-aie's defaults, which are well-chosen for general observability:

- **Core**: vector instructions, lock acquire/release, stalls
- **Memtile**: port activity, DMA S2MM/MM2S, lock operations
- **Shim**: port activity, DMA transfers

The manifest records which event set was used.  A future "retrace" mode can
cycle through alternative event sets to capture different aspects across
multiple runs.

## Changes

### trace-inject.py

Replace tile classification logic:

```
Before:  shim_tile + compute_tiles[] -> error if no compute_tiles
After:   shim_tile (destination) + tiles_to_trace[] (everything else)
         -> error only if tiles_to_trace is empty
```

Update manifest `tiles_traced` entries to include `tile_type` field
(`"core"`, `"memtile"`, `"shim"`) and `events` field (event set identifier).

### trace-run.py

No execution changes needed.  Buffer allocation is tile-type-agnostic.
`parse_trace` already handles mixed tile types.  `SystemExit` catch handles
coordinate mismatches from the parser.

### lit_trace.rs

Remove "memtile-only" from inject failure expectations.  No other changes --
the Rust pipeline is tile-type-agnostic.

### Unchanged

- `has_existing_trace` detection
- Blocklist (ctrl_packet, loadpdi, etc.)
- Staleness/hash system
- Kernel compilation pipeline
- Buffer size defaults (1MB)

## Expected Results

- 49 success -> ~53-54 success (recovering 4 memtile_dmas tests + possibly simple_repeat)
- 5 inject failures -> ~0-1 (only if a test has literally nothing to trace)
- 16 skipped unchanged (blocklisted for non-trace reasons)

## Future Work

- Retrace mode: cycle through event sets across runs for full coverage
- Packet ID deconfliction: detect existing `aie.packet_flow` IDs and offset
  trace IDs to avoid collisions
- Arbitrary workload injection: accept any .xclbin or MLIR, not just npu-xrt tests
