# VCD Deep Extraction and Cycle-Accurate Comparison

**Date:** 2026-03-12
**Status:** Design approved, pending implementation plan

## Problem

aiesimulator produces VCD files containing ~124,000 signals that fully
describe the hardware state of every tile at every cycle: DMA state machines,
lock values, stream switch flow control, program counter pipeline stages,
memory port transactions, and more. Our current VCD parser extracts only the
~200 event_trace signals per tile (0.2% of available data) and discards
everything else.

The emulator's goal is to be a cycle-accurate reproduction of the hardware.
The VCD data is the richest validation source available -- far richer than
hardware trace (which only captures configured events) or halt-and-read
snapshots. We need to use all of it.

## Goals

1. Parse ALL 124,000 VCD signals from aiesimulator output, mapped into the
   emulator's own state representation.
2. Have the emulator emit VCD output using the same signal hierarchy, so both
   sides produce directly comparable files.
3. Compare the two VCDs at the subsystem level with drill-down to individual
   signals, reporting the first divergence point per signal.
4. Integrate comparison into the bridge test suite as an `--aiesim` phase.
5. Use the VCD signal taxonomy as a completeness audit: every signal aiesim
   exposes that the emulator cannot produce is a gap to fill.

## Non-Goals

- Automated causal analysis ("lock diverged, causing DMA to diverge"). This
  requires understanding inter-subsystem dependencies and is better done by
  human analysis with two VCD files open in a waveform viewer.
- Perfetto JSON output. The emulator's eventual GUI will be the trace viewer;
  Perfetto is a stopgap.
- Peano compiler support for aiesim. aiesim requires Chess builds. This is a
  hardware/toolchain constraint, not something we solve.
- Real-time VCD emission during emulation. VCD recording is a compile-time
  feature, not a runtime toggle.

## Architecture Overview

```
aiesimulator                         xdna-emu (vcd-recording build)
     |                                       |
     v                                       v
  aiesim.vcd                           emulator.vcd
     |                                       |
     v                                       v
  wellen loads VCD                     wellen loads VCD
     |                                       |
     v                                       v
  Mapping tree resolves                Mapping tree resolves
  signal names -> StatePaths           signal names -> StatePaths
     |                                       |
     +------ matched by StatePath -----------+
                      |
                      v
              Subsystem Comparator
              (value + timing check)
                      |
                      v
              Divergence Report
              (text + JSON)
```

Both sides produce VCD. The `wellen` crate loads both files with compressed,
transition-based storage. The signal mapping tree translates VCD signal names
to `StatePath` identifiers. The comparator matches signals from both files by
`StatePath` and reports value divergences and timing deltas.

## Design Decisions

### The emulator IS the canonical state model

There is no third "neutral" state representation. The emulator's internal
state (`DeviceState`, `Tile`, `CoreState`, `DmaEngine`, `Lock`,
`StreamSwitch`, etc.) defines what state exists. The VCD adapter's job is to
map aiesim's SystemC signal names into the emulator's state structure.

This means:
- One adapter (VCD-to-emulator-state), not two.
- The emulator doesn't change its internal representation to accommodate VCD.
- Unmapped VCD signals represent gaps in the emulator, not gaps in the adapter.

### wellen for reading, vcd crate for writing

Instead of building a custom transition store, we use existing Rust crates:

- **`wellen`** (https://github.com/ekiwi/wellen) for **reading** VCD files:
  multi-threaded parsing, compressed signal storage (transition-based, not
  snapshot-based), optimized for subset access (load 124K signals, query only
  what's needed), also reads FST format (potential future optimization).

- **`vcd`** (https://crates.io/crates/vcd) for **writing** VCD files: the
  emulator's VCD emission uses this crate's `Writer` API to produce
  IEEE 1364-compliant output. wellen is a reader-focused library (built for
  the Surfer waveform viewer) and does not provide a write API.

This eliminates all custom signal storage and serialization code. We focus on
mapping and comparison, not data management.

### Cargo feature flag for VCD recording

VCD emission is behind `cfg(feature = "vcd-recording")`:

```toml
[features]
vcd-recording = ["dep:wellen", "dep:vcd"]
```

When the feature is off, all recording call sites are compiled out entirely.
No runtime cost, no optimizer trust, no conditional branches. Two binaries:

- `cargo build` -- fast emulator, no VCD capability
- `cargo build --features vcd-recording` -- VCD-capable, used with `--aiesim`

This avoids both the runtime overhead of always-on recording and the viral
generics infection of a trait-based approach with monomorphization.

### Tolerance model for nondeterministic timing

DDR/NoC access, memory bank conflicts, and DMA scheduling introduce
nondeterministic delays that differ between aiesim and the emulator. A naive
cycle-exact comparison would produce false positives on every DDR-touching
operation.

The comparison separates two concerns:

1. **Value correctness** -- Did the signal reach the same values in the same
   order? Exact match, order-sensitive, cycle-insensitive.
2. **Timing accuracy** -- Did transitions happen at the same cycle? Approximate
   match with configurable tolerance bands per subsystem.

Tolerance is hierarchical with most-specific-wins:

```toml
[vcd_compare.tolerance]
# Global default: exact match
default = 0

[vcd_compare.tolerance.core]
default = 0              # deterministic pipeline

[vcd_compare.tolerance.dma]
default = 1000            # DDR/NoC latency variation
fsm_state = 1000          # state transitions shift with latency
current_bd = 0            # BD selection is deterministic

[vcd_compare.tolerance.lock]
default = 2               # small jitter from arbitration

[vcd_compare.tolerance.stream]
default = 5               # routing pipeline differences

# Per-tile overrides when chasing specific issues
[vcd_compare.tolerance.dma.tile_overrides]
# "col,row" = tolerance
# "0,0" = 1500
```

Resolution order: per-tile-per-field > per-field > per-subsystem > global.

Value correctness is the primary pass/fail gate. Timing accuracy is
informational and tracked as a metric. As the emulator's timing model
improves, tolerance bands tighten. The bands become a timing accuracy
scoreboard.

### VCD timescale reconciliation

aiesimulator VCD uses `1 ps` timescale with ~952 ps clock period (AIE2 at
~1.05 GHz). The emulator operates in cycles, not picoseconds.

The emulator's VCD output uses cycle-based timestamps: each VCD `#N` is a
cycle number, with timescale set to the clock period (`952 ps` for AIE2).
This makes both VCDs semantically equivalent -- aiesim's `#952` = cycle 1,
emulator's `#1` at `952 ps` timescale = also cycle 1.

The comparator normalizes both files to cycles during loading:
- aiesim VCD: `cycle = timestamp_ps / clock_period_ps` (integer division)
- emulator VCD: `cycle = timestamp` (already in cycles)

Clock period is configurable (default 952 ps for AIE2, will differ for
AIE2P). The comparator reads it from the mapping tree's device configuration.

### StatePath: canonical signal identity

`StatePath` is the key type that bridges VCD signal names and emulator state.
It is a structured enum, not a string, to enable type-safe matching:

```rust
enum StatePath {
    // Lock subsystem
    LockValue { col: u8, row: u8, idx: u8 },
    LockOp { col: u8, row: u8, idx: u8 },

    // DMA subsystem
    DmaFsmState { col: u8, row: u8, dir: DmaDir, ch: u8 },
    DmaCurrentBd { col: u8, row: u8, dir: DmaDir, ch: u8 },
    DmaBdLength { col: u8, row: u8, dir: DmaDir, ch: u8 },
    DmaAddress { col: u8, row: u8, dir: DmaDir, ch: u8 },
    DmaProcessedBytes { col: u8, row: u8, dir: DmaDir, ch: u8 },
    // ... per BD field

    // Stream subsystem
    StreamPortData { col: u8, row: u8, port: PortId },
    StreamPortIdle { col: u8, row: u8, port: PortId },
    StreamPortRunning { col: u8, row: u8, port: PortId },
    StreamPortStalled { col: u8, row: u8, port: PortId },
    StreamPortTlast { col: u8, row: u8, port: PortId },

    // Core subsystem
    CorePc { col: u8, row: u8, stage: u8 },  // pipeline stages E1-E7
    CorePmAddress { col: u8, row: u8 },
    CoreReset { col: u8, row: u8 },
    // ... load/store unit fields

    // Memory subsystem
    MemBankConflict { col: u8, row: u8, bank: u8 },
    MemPortAccess { col: u8, row: u8, port: MemPortId },

    // Events (existing coverage)
    EventTrace { col: u8, row: u8, event_code: u16 },

    // Performance counters
    PerfCounter { col: u8, row: u8, idx: u8 },
}
```

The mapping tree produces `StatePath` values from VCD signal names (parsing)
and from emulator state field identifiers (emission). The comparator matches
signals from the two VCD files by `StatePath` equality, not by string name.

### Robustness: partial and oversized VCDs

aiesim VCD files can be hundreds of megabytes for long simulations, or
truncated if aiesim hits its cycle timeout or crashes.

- **Memory budget:** wellen's compressed storage is efficient (transition-based),
  but 124K signals over 100K+ cycles can still grow. The comparator loads
  signals lazily by subsystem during the sweep phase, not all at once.
- **Truncated files:** If one VCD is shorter than the other, comparison runs
  up to the shorter file's last timestamp and reports "truncated at cycle N"
  in the output. Signals that match up to the truncation point are reported
  as partial matches.
- **aietools version drift:** Signal names may change across aietools releases.
  Unmapped signals appear in the coverage audit, making drift visible
  immediately. The mapping tree is updated to track new naming conventions.

## Component Design

### 1. Hierarchical Signal Mapping Tree

The mapping tree is the heart of the system. It declaratively describes the
correspondence between VCD signal names and emulator state, using ~200
reusable templates to cover 124K signals.

**Structure:**

```rust
enum MappingNode {
    /// Parameterized group: instantiated per tile, channel, lock, etc.
    Group {
        name_pattern: &'static str,  // e.g., "tile_{col}_{row}"
        params: ParamExtractor,       // extracts col, row from name
        children: Vec<MappingNode>,
    },
    /// Leaf: one signal type, instantiated per parent parameters.
    Signal {
        name_pattern: &'static str,  // e.g., "value_{idx}"
        width: u16,
        subsystem: Subsystem,
        binding: SignalBinding,       // VCD name <-> emulator state path
    },
}

enum Subsystem {
    Core,        // PC, pipeline stages, execution state
    Dma,         // Channels, BDs, transfers, lock operations
    Lock,        // Lock values and pending operations
    Stream,      // Switch ports, routing, flow control
    Memory,      // Bank access, conflict detection
    Event,       // Trace events (existing parser coverage)
    PerfCount,   // Performance counters
}
```

**Declaration mirrors hardware hierarchy:**

```rust
fn build_mapping_tree() -> MappingNode {
    device_root()
        .tile_group("shim", TileType::Shim, |tile| {
            tile.dma(shim_dma_mapping()) // 18-bit BD length, DDR addressing
                .stream_switch(stream_switch_mapping())
                .locks(lock_mapping())
                .event_trace(event_mapping())
                .perf_counters(perf_mapping())
        })
        .tile_group("mem_row", TileType::MemTile, |tile| {
            tile.dma(dma_mapping())
                .stream_switch(stream_switch_mapping())
                .locks(lock_mapping())
                .memory(memory_port_mapping())
                .event_trace(event_mapping())
        })
        .tile_group("array", TileType::Compute, |tile| {
            tile.core(core_mapping())
                .dma(dma_mapping())
                .stream_switch(stream_switch_mapping())
                .locks(lock_mapping())
                .memory(memory_port_mapping())
                .event_trace(event_mapping())
        })
}
```

Reusable subtrees (`dma_mapping()`, `lock_mapping()`, etc.) are shared across
tile types. A lock is a lock whether it's in a shim, memtile, or compute tile.

**Coverage audit** falls out naturally. After parsing a VCD header, the tree
reports:

```
Signal Coverage Report:
  MAPPED:    118,420 / 123,327 (96.0%)
  UNMAPPED:    4,907 signals (no emulator counterpart)

  Unmapped by subsystem:
    memory/conflict_addr_*:  480 signals  -- emulator tracks conflicts, not addresses
    core/dme_*_w_*:          128 signals  -- wide load port B not yet modeled
    perf_counters/*:         200 signals  -- perf counters not implemented
```

Unmapped signals become a prioritized backlog for emulator completeness.

**Future:** The mapping tree is a natural candidate for integration with
xdna-archspec, which already aggregates hardware topology from multiple
toolchain sources. The signal mapping tree is the same concept applied to
runtime state observability.

### 2. VCD Emission (Emulator Side)

When built with `--features vcd-recording`, the emulator records state
transitions to VCD during execution.

**Signal declaration:** At startup, the mapping tree is walked to emit VCD
`$var` declarations for every signal the emulator can produce. Signals that
the emulator does not yet implement are declared but never transition (visible
in comparison as "no transitions" vs aiesim's actual transitions).

**Recording call sites:** Guarded by `#[cfg(feature = "vcd-recording")]`
throughout the device code. When the feature is off, the lines do not exist
in the binary.

```rust
// In DMA channel state machine transition:
#[cfg(feature = "vcd-recording")]
self.recorder.record_word(
    cycle,
    StatePath::dma_fsm_state(self.col, self.row, self.channel_idx),
    new_state as u32,
);
```

**Performance:** Recording cost is proportional to actual state changes, same
as aiesim. Since aiesim itself runs 10-100x slower than the emulator, VCD
recording overhead is noise in the comparison workflow.

### 3. Comparison Engine

Takes two VCD files (aiesim + emulator), loads both with `wellen`, and runs
subsystem-level comparison with drill-down.

**Phase 1: Signal alignment.** Walk both VCD headers through the mapping tree.
Match signals by `StatePath`, not by VCD name string (the emulator's hierarchy
may use slightly different naming than aiesim's SystemC hierarchy). Report
signals present in one file but not the other.

**Phase 2: Subsystem sweep.** For each subsystem, iterate through all matched
signal pairs. For each pair, walk both transition lists:

- **Value comparison:** Same values in same order? If not, report the
  transition index and cycle where values first diverge.
- **Timing comparison:** For value-matched signals, compute the cycle delta
  at each corresponding transition. Report max delta. Compare against
  tolerance band for this signal's subsystem (and any per-field or per-tile
  overrides).

**Phase 3: Report generation.** Structured output in two formats:

- **Text report** for terminal display: summary pass/fail per subsystem,
  first-divergence table, timing delta statistics.
- **JSON report** for bridge test integration: machine-readable, parseable
  by `emu-bridge-test.sh` for automated pass/fail.

Example text output:

```
=== VCD Comparison: aiesim.vcd vs emulator.vcd ===

Subsystem    Signals  Values   Timing (max delta)  Status
---------    -------  ------   ------------------  ------
Core              14  MATCH    0 cycles            PASS
DMA               92  MATCH    47 cycles (tol=1000) PASS
Locks            128  MATCH    2 cycles (tol=2)    PASS
Streams          245  MATCH    3 cycles (tol=5)    PASS
Memory           320  MATCH    1 cycle (tol=0)     PASS
Events           161  MATCH    0 cycles            PASS

Overall: PASS (all values match, timing within tolerance)
```

### 4. Standalone Tool

A new binary or subcommand:

```bash
# Compare two existing VCD files
xdna-emu vcd-compare aiesim.vcd emulator.vcd

# Or: compile, run both, compare (integrated flow)
xdna-emu compare-aiesim --prj-dir=foo.prj --timeout=100000
```

The standalone tool is useful for ad-hoc debugging. The integrated flow is
used by the bridge test suite.

### 5. Bridge Test Integration

New `--aiesim` flag for `emu-bridge-test.sh`:

```bash
# Add aiesim comparison to bridge tests
./scripts/emu-bridge-test.sh --aiesim

# Combine with existing flags
./scripts/emu-bridge-test.sh --aiesim --no-hw    # EMU vs aiesim only
./scripts/emu-bridge-test.sh --aiesim -v add_one  # single test
```

**Chess-only constraint:** aiesimulator requires Chess-compiled builds (the
`sim/` directory with `ps.so` is only generated by `aiecc --aiesim
--xchesscc`). The `--aiesim` flag operates exclusively on Chess builds. In a
dual-compiler run, aiesim comparison applies only to the Chess column. Passing
`--aiesim --peano-only` is an error (no Chess builds to compare against).

**Per-test flow when `--aiesim` is active:**

1. Chess build already produces `.prj/sim/` -- no extra compilation.
2. Run aiesimulator with `--dump-vcd` (existing `run_unit_simulation()`).
3. Run emulator (VCD-recording build) with `--dump-vcd`.
4. Run `vcd-compare` on both outputs.
5. Report per-subsystem pass/fail.

**Results directory structure:**

```
/tmp/emu-bridge-results-YYYYMMDD/
  add_one_using_dma/
    chess/
      hw/               # existing
      emu/              # existing
      aiesim/           # NEW
        aiesim.vcd
        emulator.vcd
        comparison.json
        comparison.txt
```

**Build:** The bridge script uses the VCD-recording emulator build
exclusively when `--aiesim` is passed.

## VCD Signal Taxonomy

Based on analysis of aiesimulator VCD output (123,327 signals):

| Category | Signals | Emulator Coverage | Notes |
|----------|---------|-------------------|-------|
| DMA state machines | ~13,000 | High | `DmaEngine`, `ChannelContext`, `ChannelFsm` |
| Lock values/ops | ~10,400 | High | `Lock`, `LockArbiter` |
| Stream switch | ~22,841 | High | `StreamSwitch`, `StreamPort` |
| Data memory ports | ~24,800 | Partial | Bank access tracked, port-level not yet |
| Event trace | ~17,900 | Complete | Existing parser, `TraceUnit` |
| Core pipeline (PC) | ~14 | Partial | Single PC, not pipeline stages E1-E7 |
| Core load/store | ~100 | Partial | Load pipeline modeled, not port-level |
| Performance counters | ~796 | None | Not implemented |
| Tile control | ~1,385 | Partial | Registers exist, task_complete not modeled |
| Event broadcast | ~160 | None | Broadcast direction masks not modeled |
| Memory conflicts | ~960 | Partial | Bitmask tracking, not address-level |
| Control packets | 2 | High | `ControlPacketState` |
| Shim PL interface | ~640 | None | PL fabric not relevant to NPU |
| Global/resets | ~40 | Low | Reset signals partially modeled |

This table serves as the initial completeness audit. "None" and "Partial"
entries are the emulator feature backlog, prioritized by how much they affect
functional correctness.

## Dependencies

| Crate | Purpose | Status |
|-------|---------|--------|
| `wellen` | VCD reading, compressed signal storage, subset access | New dependency |
| `vcd` | VCD writing (emulator emission via `Writer` API) | New dependency |
| `toml` | Tolerance config parsing | Already in use |
| `serde_json` | JSON report output | Already in use |

## Files to Create/Modify

| File | Purpose |
|------|---------|
| `src/vcd/mod.rs` | Module root for VCD deep extraction |
| `src/vcd/mapping.rs` | Hierarchical signal mapping tree |
| `src/vcd/emit.rs` | Emulator VCD emission (behind feature flag) |
| `src/vcd/compare.rs` | Comparison engine |
| `src/vcd/tolerance.rs` | Tolerance config parsing and resolution |
| `src/vcd/report.rs` | Text and JSON report generation |
| `src/bin/vcd_compare.rs` | Standalone comparison binary |
| `Cargo.toml` | `vcd-recording` feature, wellen dependency |
| `scripts/emu-bridge-test.sh` | `--aiesim` flag |
| `src/trace/vcd.rs` | Existing parser -- evaluate migration to wellen |

The existing `src/trace/vcd.rs` (event_trace-only parser) continues to work
for its current use case (Perfetto JSON conversion). It may eventually migrate
to use wellen internally, but that is not a goal of this work.

## Implementation Order

1. Add wellen dependency, build mapping tree with lock and DMA subtrees
2. VCD parsing through mapping tree, coverage audit output
3. Emulator VCD emission (feature flag, recording call sites)
4. Comparison engine (subsystem sweep, tolerance model)
5. Standalone comparison binary
6. Bridge test integration (`--aiesim` flag)
7. Expand mapping tree to remaining subsystems (streams, memory, core)

Each step is independently testable. Step 1-2 can be validated against the
existing aiesim VCD files at `/tmp/aiesim-test2/`.

## Verification

1. Parse aiesim VCD, print coverage audit -- confirms mapping tree works
2. Run emulator with `--dump-vcd`, verify signal declarations match tree
3. Compare aiesim vs emulator VCD on a simple test (e.g., lock acquire/release)
4. Run bridge test with `--aiesim` on `01_precompiled_core_function`
5. Tighten tolerance bands, verify timing accuracy improves with emulator fixes

## Sources

- [wellen](https://github.com/ekiwi/wellen) -- Rust VCD/FST waveform library
- [vcd crate](https://crates.io/crates/vcd) -- Rust VCD reader/writer
- [vcddiff](https://crates.io/crates/vcddiff) -- VCD comparison tool
- [Surfer waveform viewer](https://kevinlaeufer.com/pdfs/surfer_cav2025.pdf) -- Built on wellen
