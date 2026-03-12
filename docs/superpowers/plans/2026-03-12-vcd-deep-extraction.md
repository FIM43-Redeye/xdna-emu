# VCD Deep Extraction Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parse all 124K aiesimulator VCD signals, emit VCD from the emulator, and compare both at the subsystem level with configurable timing tolerance.

**Architecture:** Both aiesim and the emulator produce VCD files. The `wellen` crate reads both; the `vcd` crate writes the emulator's output. A hierarchical signal mapping tree translates VCD signal names to `StatePath` identifiers. A subsystem comparator walks matched signal pairs, checking value sequences and timing deltas against configurable tolerance bands.

**Tech Stack:** Rust, wellen (VCD reading), vcd (VCD writing), toml (tolerance config), serde_json (JSON reports)

**Spec:** `docs/superpowers/specs/2026-03-12-vcd-deep-extraction-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| **`src/vcd/mod.rs`** | Module root. Re-exports public types. |
| **`src/vcd/state_path.rs`** | `StatePath` enum, `Subsystem` enum, `DmaDir`/`PortId` types. |
| **`src/vcd/mapping.rs`** | `MappingNode` tree, builder API, `resolve()` for VCD name -> StatePath. |
| **`src/vcd/lock_mapping.rs`** | Lock subsystem subtree definition. |
| **`src/vcd/dma_mapping.rs`** | DMA subsystem subtree definition (tile + shim variants). |
| **`src/vcd/stream_mapping.rs`** | Stream switch subsystem subtree definition. |
| **`src/vcd/core_mapping.rs`** | Core/pipeline subsystem subtree definition. |
| **`src/vcd/event_mapping.rs`** | Event trace subsystem subtree definition. |
| **`src/vcd/coverage.rs`** | Coverage audit: mapped vs unmapped signals. |
| **`src/vcd/emit.rs`** | VCD writer (behind `cfg(feature = "vcd-recording")`). |
| **`src/vcd/tolerance.rs`** | Tolerance config parsing, hierarchical resolution. |
| **`src/vcd/compare.rs`** | Comparison engine: alignment, subsystem sweep. |
| **`src/vcd/report.rs`** | Text and JSON report generation. |
| **`src/bin/vcd_compare.rs`** | Standalone comparison binary. |
| **`Cargo.toml`** | `vcd-recording` feature flag, wellen + vcd dependencies. |
| **`src/lib.rs`** | Add `pub mod vcd;` declaration. |

Existing files that are NOT modified: `src/trace/vcd.rs` (continues to handle event_trace -> Perfetto JSON conversion independently).

---

## Chunk 1: Foundation -- StatePath, Mapping Tree, Coverage Audit

This chunk builds the core data model and mapping infrastructure. It can be
fully tested against the existing aiesim VCD files at `/tmp/aiesim-test2/`
without touching the emulator's device code at all.

### Task 1: Add dependencies and module skeleton

**Files:**
- Modify: `Cargo.toml`
- Create: `src/vcd/mod.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Add wellen and vcd dependencies to Cargo.toml**

In `Cargo.toml`, add under `[dependencies]`:

```toml
# VCD deep extraction (comparison engine always available; emission gated by feature)
wellen = "0.20"
```

Add under `[features]`:

```toml
vcd-recording = ["dep:vcd"]
```

Add under `[dependencies]` with `optional = true`:

```toml
vcd = { version = "0.7", optional = true }
```

wellen is always available (needed for VCD reading/comparison). The `vcd` writing
crate is only pulled in when emitting VCD from the emulator.

- [ ] **Step 2: Create module skeleton**

Create `src/vcd/mod.rs`:

```rust
//! VCD deep extraction: parse, emit, and compare aiesimulator VCD files.
//!
//! This module provides:
//! - [`state_path::StatePath`]: canonical signal identity bridging VCD names and emulator state
//! - [`mapping`]: hierarchical signal mapping tree (VCD signal names <-> StatePaths)
//! - [`coverage`]: coverage audit (mapped vs unmapped signals)
//! - [`compare`]: subsystem-level comparison engine
//! - [`tolerance`]: configurable timing tolerance bands
//! - [`report`]: text and JSON report generation
//! - [`emit`]: VCD emission from emulator (behind `vcd-recording` feature flag)

pub mod state_path;
pub mod mapping;
pub mod coverage;
pub mod tolerance;
pub mod compare;
pub mod report;

#[cfg(feature = "vcd-recording")]
pub mod emit;

// Subsystem mapping subtrees
pub mod lock_mapping;
pub mod dma_mapping;
pub mod stream_mapping;
pub mod core_mapping;
pub mod event_mapping;
```

- [ ] **Step 3: Add module declaration to lib.rs**

In `src/lib.rs`, add after the `pub mod archspec;` line:

```rust
pub mod vcd;
```

- [ ] **Step 4: Verify it compiles**

Run: `cargo check 2>&1 | head -5`

Expected: Compilation errors for missing files (the submodules). That's fine --
we'll create them in subsequent tasks. The dependency resolution should succeed.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/vcd/mod.rs src/lib.rs
git commit -m "feat(vcd): add module skeleton and wellen/vcd dependencies"
```

---

### Task 2: StatePath enum and Subsystem types

**Files:**
- Create: `src/vcd/state_path.rs`

This is the canonical signal identity type. Every VCD signal and every emulator
state field maps to a `StatePath` value. The comparison engine matches signals
by `StatePath` equality.

- [ ] **Step 1: Write tests for StatePath**

At the bottom of `src/vcd/state_path.rs`, write the tests first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_path_subsystem_classification() {
        let lock = StatePath::LockValue { col: 0, row: 1, idx: 3 };
        assert_eq!(lock.subsystem(), Subsystem::Lock);

        let dma = StatePath::DmaFsmState {
            col: 1, row: 2, dir: DmaDir::S2mm, ch: 0,
        };
        assert_eq!(dma.subsystem(), Subsystem::Dma);

        let stream = StatePath::StreamPortData {
            col: 0, row: 0, port: PortId::named("sSouth3"),
        };
        assert_eq!(stream.subsystem(), Subsystem::Stream);

        let core = StatePath::CorePc { col: 0, row: 3, stage: 1 };
        assert_eq!(core.subsystem(), Subsystem::Core);
    }

    #[test]
    fn state_path_display_roundtrip() {
        let path = StatePath::LockValue { col: 2, row: 1, idx: 7 };
        let s = path.to_string();
        assert!(s.contains("lock"));
        assert!(s.contains("2"));
        assert!(s.contains("1"));
        assert!(s.contains("7"));
    }

    #[test]
    fn state_path_equality_and_hash() {
        use std::collections::HashSet;
        let a = StatePath::LockValue { col: 0, row: 1, idx: 3 };
        let b = StatePath::LockValue { col: 0, row: 1, idx: 3 };
        let c = StatePath::LockValue { col: 0, row: 1, idx: 4 };
        assert_eq!(a, b);
        assert_ne!(a, c);

        let mut set = HashSet::new();
        set.insert(a);
        assert!(set.contains(&b));
        assert!(!set.contains(&c));
    }

    #[test]
    fn port_id_construction() {
        let named = PortId::named("sSouth3");
        assert_eq!(named.name(), "sSouth3");

        let indexed = PortId::indexed(PortBundle::South, 3);
        assert_eq!(indexed.name(), "sSouth3");

        assert_eq!(named, indexed);
    }

    #[test]
    fn dma_dir_display() {
        assert_eq!(DmaDir::S2mm.as_str(), "s2mm");
        assert_eq!(DmaDir::Mm2s.as_str(), "mm2s");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib vcd::state_path 2>&1 | tail -5`

Expected: FAIL -- module doesn't have the types yet.

- [ ] **Step 3: Implement StatePath and supporting types**

Write the implementation above the tests in `src/vcd/state_path.rs`:

```rust
//! Canonical signal identity for VCD deep extraction.
//!
//! `StatePath` bridges VCD signal names and emulator internal state.
//! Two signals from different VCD files (aiesim vs emulator) that map
//! to the same `StatePath` are compared by the comparison engine.

use std::fmt;

/// Hardware subsystem classification for comparison grouping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Subsystem {
    Core,
    Dma,
    Lock,
    Stream,
    Memory,
    Event,
    PerfCount,
}

impl Subsystem {
    /// All subsystem variants in display order.
    pub const ALL: &[Subsystem] = &[
        Subsystem::Core,
        Subsystem::Dma,
        Subsystem::Lock,
        Subsystem::Stream,
        Subsystem::Memory,
        Subsystem::Event,
        Subsystem::PerfCount,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            Subsystem::Core => "core",
            Subsystem::Dma => "dma",
            Subsystem::Lock => "lock",
            Subsystem::Stream => "stream",
            Subsystem::Memory => "memory",
            Subsystem::Event => "event",
            Subsystem::PerfCount => "perf_count",
        }
    }
}

impl fmt::Display for Subsystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// DMA transfer direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DmaDir {
    /// Stream to memory.
    S2mm,
    /// Memory to stream.
    Mm2s,
}

impl DmaDir {
    pub fn as_str(&self) -> &'static str {
        match self {
            DmaDir::S2mm => "s2mm",
            DmaDir::Mm2s => "mm2s",
        }
    }
}

impl fmt::Display for DmaDir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Stream switch port bundle direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PortBundle {
    North,
    South,
    East,
    West,
    Dma,
    Core,
    Fifo,
    Trace,
    TileCtrl,
}

impl PortBundle {
    /// VCD prefix for this bundle (e.g., "sSouth", "sDMA").
    pub fn vcd_prefix(&self) -> &'static str {
        match self {
            PortBundle::North => "sNorth",
            PortBundle::South => "sSouth",
            PortBundle::East => "sEast",
            PortBundle::West => "sWest",
            PortBundle::Dma => "sDMA",
            PortBundle::Core => "sCore",
            PortBundle::Fifo => "sFIFO",
            PortBundle::Trace => "sTrace",
            PortBundle::TileCtrl => "sTileCtrl",
        }
    }
}

/// Stream port identifier.
///
/// Wraps the VCD port name (e.g., "sSouth3") for type-safe matching.
/// Can be constructed from a raw name or from bundle + index.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PortId(String);

impl PortId {
    /// Construct from a raw VCD port name.
    pub fn named(name: &str) -> Self {
        PortId(name.to_string())
    }

    /// Construct from bundle direction and index.
    pub fn indexed(bundle: PortBundle, idx: u8) -> Self {
        PortId(format!("{}{}", bundle.vcd_prefix(), idx))
    }

    /// Get the port name string.
    pub fn name(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PortId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Memory port identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemPortId {
    /// Port name from VCD (e.g., "Port_S_read_A_b0").
    pub name: String,
}

impl MemPortId {
    pub fn new(name: &str) -> Self {
        MemPortId { name: name.to_string() }
    }
}

/// Canonical signal identity.
///
/// Every VCD signal and every emulator state field maps to a `StatePath`.
/// The comparison engine matches signals from two VCD files by `StatePath`
/// equality, not by VCD signal name string.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StatePath {
    // -- Lock subsystem --
    LockValue { col: u8, row: u8, idx: u8 },
    LockOp { col: u8, row: u8, idx: u8 },

    // -- DMA subsystem --
    DmaFsmState { col: u8, row: u8, dir: DmaDir, ch: u8 },
    DmaCurrentBd { col: u8, row: u8, dir: DmaDir, ch: u8 },
    DmaBdLength { col: u8, row: u8, dir: DmaDir, ch: u8 },
    DmaAddress { col: u8, row: u8, dir: DmaDir, ch: u8 },
    DmaData { col: u8, row: u8, dir: DmaDir, ch: u8 },
    DmaProcessedStream { col: u8, row: u8, dir: DmaDir, ch: u8 },
    DmaProcessedMem { col: u8, row: u8, dir: DmaDir, ch: u8 },
    DmaLockAcqId { col: u8, row: u8, dir: DmaDir, ch: u8 },
    DmaLockAcqValue { col: u8, row: u8, dir: DmaDir, ch: u8 },
    DmaLockRelValue { col: u8, row: u8, dir: DmaDir, ch: u8 },
    DmaStatus { col: u8, row: u8, dir: DmaDir, ch: u8 },
    DmaBdValid { col: u8, row: u8, dir: DmaDir, ch: u8 },
    DmaPacketId { col: u8, row: u8, dir: DmaDir, ch: u8 },
    DmaEnablePacket { col: u8, row: u8, dir: DmaDir, ch: u8 },
    DmaNextBd { col: u8, row: u8, dir: DmaDir, ch: u8 },
    DmaUseNextBd { col: u8, row: u8, dir: DmaDir, ch: u8 },
    DmaTlastSuppress { col: u8, row: u8, dir: DmaDir, ch: u8 },
    DmaIterStepsize { col: u8, row: u8, dir: DmaDir, ch: u8 },
    DmaIterCurrent { col: u8, row: u8, dir: DmaDir, ch: u8 },
    DmaIterWrap { col: u8, row: u8, dir: DmaDir, ch: u8 },

    // -- Stream subsystem --
    StreamPortData { col: u8, row: u8, port: PortId },
    StreamPortIdle { col: u8, row: u8, port: PortId },
    StreamPortRunning { col: u8, row: u8, port: PortId },
    StreamPortStalled { col: u8, row: u8, port: PortId },
    StreamPortTlast { col: u8, row: u8, port: PortId },

    // -- Core subsystem --
    CorePc { col: u8, row: u8, stage: u8 },
    CorePmAddress { col: u8, row: u8 },
    CorePmData { col: u8, row: u8 },
    CoreTmAddress { col: u8, row: u8 },
    CoreTmReadData { col: u8, row: u8 },
    CoreTmWriteData { col: u8, row: u8 },
    CoreTmLoad { col: u8, row: u8 },
    CoreTmStore { col: u8, row: u8 },
    CoreReset { col: u8, row: u8 },
    CoreBreakpointHalted { col: u8, row: u8 },

    // -- Memory subsystem --
    MemBankConflict { col: u8, row: u8, bank: u8 },
    MemConflictAddr { col: u8, row: u8, bank: u8 },
    MemPortAccess { col: u8, row: u8, port: MemPortId },

    // -- Events (existing coverage) --
    EventTrace { col: u8, row: u8, event_code: u16, event_name: String },

    // -- Performance counters --
    PerfCounter { col: u8, row: u8, idx: u8 },
}

impl StatePath {
    /// Which subsystem this signal belongs to.
    pub fn subsystem(&self) -> Subsystem {
        match self {
            StatePath::LockValue { .. } | StatePath::LockOp { .. } => Subsystem::Lock,

            StatePath::DmaFsmState { .. } | StatePath::DmaCurrentBd { .. }
            | StatePath::DmaBdLength { .. } | StatePath::DmaAddress { .. }
            | StatePath::DmaData { .. } | StatePath::DmaProcessedStream { .. }
            | StatePath::DmaProcessedMem { .. } | StatePath::DmaLockAcqId { .. }
            | StatePath::DmaLockAcqValue { .. } | StatePath::DmaLockRelValue { .. }
            | StatePath::DmaStatus { .. } | StatePath::DmaBdValid { .. }
            | StatePath::DmaPacketId { .. } | StatePath::DmaEnablePacket { .. }
            | StatePath::DmaNextBd { .. } | StatePath::DmaUseNextBd { .. }
            | StatePath::DmaTlastSuppress { .. } | StatePath::DmaIterStepsize { .. }
            | StatePath::DmaIterCurrent { .. } | StatePath::DmaIterWrap { .. } => Subsystem::Dma,

            StatePath::StreamPortData { .. } | StatePath::StreamPortIdle { .. }
            | StatePath::StreamPortRunning { .. } | StatePath::StreamPortStalled { .. }
            | StatePath::StreamPortTlast { .. } => Subsystem::Stream,

            StatePath::CorePc { .. } | StatePath::CorePmAddress { .. }
            | StatePath::CorePmData { .. } | StatePath::CoreTmAddress { .. }
            | StatePath::CoreTmReadData { .. } | StatePath::CoreTmWriteData { .. }
            | StatePath::CoreTmLoad { .. } | StatePath::CoreTmStore { .. }
            | StatePath::CoreReset { .. }
            | StatePath::CoreBreakpointHalted { .. } => Subsystem::Core,

            StatePath::MemBankConflict { .. } | StatePath::MemConflictAddr { .. }
            | StatePath::MemPortAccess { .. } => Subsystem::Memory,

            StatePath::EventTrace { .. } => Subsystem::Event,
            StatePath::PerfCounter { .. } => Subsystem::PerfCount,
        }
    }

    /// Tile coordinates (col, row) for this signal.
    pub fn tile(&self) -> (u8, u8) {
        match self {
            StatePath::LockValue { col, row, .. }
            | StatePath::LockOp { col, row, .. }
            | StatePath::DmaFsmState { col, row, .. }
            | StatePath::DmaCurrentBd { col, row, .. }
            | StatePath::DmaBdLength { col, row, .. }
            | StatePath::DmaAddress { col, row, .. }
            | StatePath::DmaData { col, row, .. }
            | StatePath::DmaProcessedStream { col, row, .. }
            | StatePath::DmaProcessedMem { col, row, .. }
            | StatePath::DmaLockAcqId { col, row, .. }
            | StatePath::DmaLockAcqValue { col, row, .. }
            | StatePath::DmaLockRelValue { col, row, .. }
            | StatePath::DmaStatus { col, row, .. }
            | StatePath::DmaBdValid { col, row, .. }
            | StatePath::DmaPacketId { col, row, .. }
            | StatePath::DmaEnablePacket { col, row, .. }
            | StatePath::DmaNextBd { col, row, .. }
            | StatePath::DmaUseNextBd { col, row, .. }
            | StatePath::DmaTlastSuppress { col, row, .. }
            | StatePath::DmaIterStepsize { col, row, .. }
            | StatePath::DmaIterCurrent { col, row, .. }
            | StatePath::DmaIterWrap { col, row, .. }
            | StatePath::StreamPortData { col, row, .. }
            | StatePath::StreamPortIdle { col, row, .. }
            | StatePath::StreamPortRunning { col, row, .. }
            | StatePath::StreamPortStalled { col, row, .. }
            | StatePath::StreamPortTlast { col, row, .. }
            | StatePath::CorePc { col, row, .. }
            | StatePath::CorePmAddress { col, row, .. }
            | StatePath::CorePmData { col, row, .. }
            | StatePath::CoreTmAddress { col, row, .. }
            | StatePath::CoreTmReadData { col, row, .. }
            | StatePath::CoreTmWriteData { col, row, .. }
            | StatePath::CoreTmLoad { col, row, .. }
            | StatePath::CoreTmStore { col, row, .. }
            | StatePath::CoreReset { col, row, .. }
            | StatePath::CoreBreakpointHalted { col, row, .. }
            | StatePath::MemBankConflict { col, row, .. }
            | StatePath::MemConflictAddr { col, row, .. }
            | StatePath::MemPortAccess { col, row, .. }
            | StatePath::EventTrace { col, row, .. }
            | StatePath::PerfCounter { col, row, .. } => (*col, *row),
        }
    }

    /// Signal-specific name for tolerance config lookup (e.g., "fsm_state", "value").
    pub fn field_name(&self) -> &'static str {
        match self {
            StatePath::LockValue { .. } => "value",
            StatePath::LockOp { .. } => "op",
            StatePath::DmaFsmState { .. } => "fsm_state",
            StatePath::DmaCurrentBd { .. } => "current_bd",
            StatePath::DmaBdLength { .. } => "bd_length",
            StatePath::DmaAddress { .. } => "address",
            StatePath::DmaData { .. } => "data",
            StatePath::DmaProcessedStream { .. } => "processed_stream",
            StatePath::DmaProcessedMem { .. } => "processed_mem",
            StatePath::DmaLockAcqId { .. } => "lock_acq_id",
            StatePath::DmaLockAcqValue { .. } => "lock_acq_value",
            StatePath::DmaLockRelValue { .. } => "lock_rel_value",
            StatePath::DmaStatus { .. } => "status",
            StatePath::DmaBdValid { .. } => "bd_valid",
            StatePath::DmaPacketId { .. } => "packet_id",
            StatePath::DmaEnablePacket { .. } => "enable_packet",
            StatePath::DmaNextBd { .. } => "next_bd",
            StatePath::DmaUseNextBd { .. } => "use_next_bd",
            StatePath::DmaTlastSuppress { .. } => "tlast_suppress",
            StatePath::DmaIterStepsize { .. } => "iter_stepsize",
            StatePath::DmaIterCurrent { .. } => "iter_current",
            StatePath::DmaIterWrap { .. } => "iter_wrap",
            StatePath::StreamPortData { .. } => "data",
            StatePath::StreamPortIdle { .. } => "idle",
            StatePath::StreamPortRunning { .. } => "running",
            StatePath::StreamPortStalled { .. } => "stalled",
            StatePath::StreamPortTlast { .. } => "tlast",
            StatePath::CorePc { .. } => "pc",
            StatePath::CorePmAddress { .. } => "pm_address",
            StatePath::CorePmData { .. } => "pm_data",
            StatePath::CoreTmAddress { .. } => "tm_address",
            StatePath::CoreTmReadData { .. } => "tm_read_data",
            StatePath::CoreTmWriteData { .. } => "tm_write_data",
            StatePath::CoreTmLoad { .. } => "tm_load",
            StatePath::CoreTmStore { .. } => "tm_store",
            StatePath::CoreReset { .. } => "reset",
            StatePath::CoreBreakpointHalted { .. } => "breakpoint_halted",
            StatePath::MemBankConflict { .. } => "conflict",
            StatePath::MemConflictAddr { .. } => "conflict_addr",
            StatePath::MemPortAccess { .. } => "port_access",
            StatePath::EventTrace { .. } => "event",
            StatePath::PerfCounter { .. } => "counter",
        }
    }
}

impl fmt::Display for StatePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (col, row) = self.tile();
        write!(f, "tile({},{}).{}.{}", col, row, self.subsystem(), self.field_name())?;
        // Add distinguishing suffix for indexed signals
        match self {
            StatePath::LockValue { idx, .. } | StatePath::LockOp { idx, .. } => {
                write!(f, "[{}]", idx)
            }
            StatePath::DmaFsmState { dir, ch, .. }
            | StatePath::DmaCurrentBd { dir, ch, .. }
            | StatePath::DmaBdLength { dir, ch, .. }
            | StatePath::DmaAddress { dir, ch, .. }
            | StatePath::DmaData { dir, ch, .. }
            | StatePath::DmaProcessedStream { dir, ch, .. }
            | StatePath::DmaProcessedMem { dir, ch, .. }
            | StatePath::DmaLockAcqId { dir, ch, .. }
            | StatePath::DmaLockAcqValue { dir, ch, .. }
            | StatePath::DmaLockRelValue { dir, ch, .. }
            | StatePath::DmaStatus { dir, ch, .. }
            | StatePath::DmaBdValid { dir, ch, .. }
            | StatePath::DmaPacketId { dir, ch, .. }
            | StatePath::DmaEnablePacket { dir, ch, .. }
            | StatePath::DmaNextBd { dir, ch, .. }
            | StatePath::DmaUseNextBd { dir, ch, .. }
            | StatePath::DmaTlastSuppress { dir, ch, .. }
            | StatePath::DmaIterStepsize { dir, ch, .. }
            | StatePath::DmaIterCurrent { dir, ch, .. }
            | StatePath::DmaIterWrap { dir, ch, .. } => {
                write!(f, "[{}.{}]", dir, ch)
            }
            StatePath::StreamPortData { port, .. }
            | StatePath::StreamPortIdle { port, .. }
            | StatePath::StreamPortRunning { port, .. }
            | StatePath::StreamPortStalled { port, .. }
            | StatePath::StreamPortTlast { port, .. } => {
                write!(f, "[{}]", port)
            }
            StatePath::CorePc { stage, .. } => write!(f, "[E{}]", stage),
            StatePath::MemBankConflict { bank, .. }
            | StatePath::MemConflictAddr { bank, .. } => {
                write!(f, "[{}]", bank)
            }
            StatePath::MemPortAccess { port, .. } => write!(f, "[{}]", port.name),
            StatePath::EventTrace { event_code, event_name, .. } => {
                write!(f, "[{}:{}]", event_code, event_name)
            }
            StatePath::PerfCounter { idx, .. } => write!(f, "[{}]", idx),
            _ => Ok(()),
        }
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --lib vcd::state_path 2>&1 | tail -10`

Expected: All 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/vcd/state_path.rs
git commit -m "feat(vcd): add StatePath enum and supporting types"
```

---

### Task 3: Mapping tree infrastructure

**Files:**
- Create: `src/vcd/mapping.rs`

The mapping tree is the hierarchical, declarative structure that translates
between VCD signal names and StatePaths. This task builds the tree node types
and the resolution algorithm. Subtree definitions (lock, DMA, etc.) come in
subsequent tasks.

- [ ] **Step 1: Write tests for mapping tree resolution**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::vcd::state_path::*;

    #[test]
    fn resolve_lock_signal() {
        let tree = build_test_tree();
        // aiesim VCD: "top.math_engine.mem_row.tile_0_1.locks.value_3"
        let segments = ["top", "math_engine", "mem_row", "tile_0_1", "locks", "value_3"];
        let result = tree.resolve(&segments);
        assert_eq!(
            result,
            Some(StatePath::LockValue { col: 0, row: 1, idx: 3 })
        );
    }

    #[test]
    fn resolve_unknown_signal_returns_none() {
        let tree = build_test_tree();
        let segments = ["top", "math_engine", "totally_unknown"];
        assert_eq!(tree.resolve(&segments), None);
    }

    #[test]
    fn enumerate_all_paths() {
        let tree = build_test_tree();
        let paths = tree.enumerate_paths();
        // Should contain at least the lock signals
        assert!(paths.iter().any(|p| matches!(p, StatePath::LockValue { .. })));
    }

    /// Minimal tree for testing: just locks in one mem_row tile.
    fn build_test_tree() -> MappingTree {
        MappingTree::new()
            .scope("top")
            .scope("math_engine")
            .tile_group("mem_row", &[(0, 1)])
            .locks(64)
            .build()
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib vcd::mapping 2>&1 | tail -5`

Expected: FAIL.

- [ ] **Step 3: Implement MappingTree and MappingNode**

Write the implementation in `src/vcd/mapping.rs`. The key types:

- `MappingNode`: enum with `Scope`, `TileGroup`, `Subsystem`, `Signal` variants.
- `MappingTree`: wraps the root node, provides `resolve(&[&str]) -> Option<StatePath>`.
- Builder API: fluent methods for constructing the tree declaratively.
- `resolve()`: walks the VCD signal name segments through the tree, extracting
  parameters (col, row, idx) at `TileGroup` and `Signal` levels, and returns
  the matching `StatePath`.
- `enumerate_paths()`: walks the entire tree and returns all possible `StatePath`
  values (for VCD emission header generation).

Key implementation details:
- Tile group nodes use pattern `tile_{col}_{row}` -- extract col/row by parsing
  the VCD name segment with a simple `tile_(\d+)_(\d+)` regex or string split.
- Signal leaf nodes use patterns like `value_{idx}` -- extract the index
  parameter from the segment suffix.
- The tree is immutable after construction.

This is a substantial implementation (~200-300 lines). Focus on getting the
lock subtree working end-to-end first (resolve + enumerate), then expand.

This task must also define the `SubsystemMapping` type used by Tasks 4-7.
`SubsystemMapping` is a builder for one subsystem's signal set within a tile:

```rust
/// Builder for one subsystem's signal mappings.
pub struct SubsystemMapping {
    /// Scope name in VCD (e.g., "locks", "stream_switch")
    scope_name: &'static str,
    subsystem: Subsystem,
    /// Signal definitions: (name_pattern, count, width, StatePath factory)
    signals: Vec<SignalDef>,
}

impl SubsystemMapping {
    pub fn new(scope_name: &'static str, subsystem: Subsystem) -> Self { ... }

    /// Add an indexed signal: matches "{name}_{idx}" for idx 0..count.
    pub fn indexed_signal(
        self,
        name_prefix: &'static str,
        count: u8,
        width: u16,
        factory: fn(col: u8, row: u8, idx: u8) -> StatePath,
    ) -> Self { ... }

    /// Can this subsystem resolve the given VCD segment name?
    pub fn can_resolve(&self, segment: &str) -> bool { ... }

    /// Enumerate all StatePaths for a tile at (col, row).
    pub fn enumerate(&self, col: u8, row: u8) -> Vec<StatePath> { ... }
}
```

Tasks 4-7 all construct `SubsystemMapping` instances via this builder.

- [ ] **Step 4: Run tests**

Run: `cargo test --lib vcd::mapping 2>&1 | tail -10`

Expected: All 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/vcd/mapping.rs
git commit -m "feat(vcd): hierarchical signal mapping tree with resolve/enumerate"
```

---

### Task 4: Lock mapping subtree

**Files:**
- Create: `src/vcd/lock_mapping.rs`

First concrete subsystem mapping. Locks are the simplest subsystem: just
`value_{idx}` and `lock_op_{idx}` per tile. This validates the mapping tree
end-to-end against real VCD data.

- [ ] **Step 1: Write test against real VCD file**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_mapping_resolves_value_signal() {
        let lock_tree = lock_mapping(64); // 64 locks per memtile
        // This should match: "value_3" segment -> LockValue with idx=3
        assert!(lock_tree.can_resolve("value_3"));
    }

    #[test]
    fn lock_mapping_resolves_op_signal() {
        let lock_tree = lock_mapping(64);
        assert!(lock_tree.can_resolve("lock_op_7"));
    }

    #[test]
    fn lock_mapping_enumerates_all() {
        let lock_tree = lock_mapping(16); // 16 locks for compute tile
        let paths = lock_tree.enumerate(0, 3); // tile (0,3)
        // 16 values + 16 ops = 32
        assert_eq!(paths.len(), 32);
    }
}
```

- [ ] **Step 2: Implement lock_mapping()**

In `src/vcd/lock_mapping.rs`:

```rust
//! Lock subsystem signal mapping.
//!
//! VCD hierarchy: `tile_X_Y.locks.value_N` and `tile_X_Y.locks.lock_op_N`
//! Maps to: `StatePath::LockValue` and `StatePath::LockOp`

use crate::vcd::mapping::SubsystemMapping;
use crate::vcd::state_path::{StatePath, Subsystem};

/// Build the lock subsystem mapping for a tile with `num_locks` locks.
pub fn lock_mapping(num_locks: u8) -> SubsystemMapping {
    SubsystemMapping::new("locks", Subsystem::Lock)
        .indexed_signal("value", num_locks, 32, |col, row, idx| {
            StatePath::LockValue { col, row, idx }
        })
        .indexed_signal("lock_op", num_locks, 32, |col, row, idx| {
            StatePath::LockOp { col, row, idx }
        })
}
```

The `SubsystemMapping` builder (part of `mapping.rs`) handles the
`{name}_{idx}` pattern matching and enumeration.

- [ ] **Step 3: Run tests**

Run: `cargo test --lib vcd::lock_mapping 2>&1 | tail -5`

Expected: All 3 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/vcd/lock_mapping.rs
git commit -m "feat(vcd): lock subsystem signal mapping"
```

---

### Task 5: DMA mapping subtree

**Files:**
- Create: `src/vcd/dma_mapping.rs`

DMA is the most complex subsystem mapping (~450 signals per tile per the VCD
survey). Each tile has S2MM and MM2S channels, each with ~20 BD state fields.

- [ ] **Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::vcd::state_path::DmaDir;

    #[test]
    fn dma_mapping_resolves_fsm_state() {
        let dma = dma_mapping(2, 2); // 2 S2MM, 2 MM2S
        // VCD: "s2mm_state0.cur_bd" within tile's DMA scope
        assert!(dma.can_resolve_nested(&["s2mm_state0", "cur_bd"]));
    }

    #[test]
    fn dma_mapping_enumerates_channels() {
        let dma = dma_mapping(2, 2);
        let paths = dma.enumerate(0, 1);
        // 2 S2MM + 2 MM2S channels, each with ~20 fields
        assert!(paths.len() >= 80);
    }

    #[test]
    fn shim_dma_mapping_different_from_tile() {
        let shim = shim_dma_mapping(2, 2);
        let tile = dma_mapping(2, 2);
        // Shim has same structure but may have additional fields
        let shim_paths = shim.enumerate(0, 0);
        let tile_paths = tile.enumerate(0, 1);
        assert!(!shim_paths.is_empty());
        assert!(!tile_paths.is_empty());
    }
}
```

- [ ] **Step 2: Implement dma_mapping() and shim_dma_mapping()**

Define channel-level signal mapping for all BD state fields observed in the
VCD survey:

```
cur_bd, cur_bd_valid, cur_bd_lock_acq_ID, cur_bd_acquire_value,
cur_bd_release_value, cur_bd_length, cur_bd_next_BD, cur_bd_use_next_BD,
cur_bd_tlast_suppress, cur_bd_iteration_stepsize, cur_bd_iteration_current,
cur_bd_iteration_wrap, cur_bd_packet_ID, cur_bd_enable_packet,
status, processed_stream, processed_mem, address, data, lanes,
requested_lock_acq, requested_lock_rel, release_request
```

Each channel is a `ChannelMapping` containing these signal definitions. The
`dma_mapping()` function creates an S2MM + MM2S group, each with N channels.

`shim_dma_mapping()` reuses the same channel structure (shim DMA has the same
VCD signal hierarchy, just with 18-bit BD length fields -- that's a value
range difference, not a structural one).

- [ ] **Step 3: Run tests**

Run: `cargo test --lib vcd::dma_mapping 2>&1 | tail -5`

Expected: All 3 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/vcd/dma_mapping.rs
git commit -m "feat(vcd): DMA subsystem signal mapping (tile + shim)"
```

---

### Task 6: Coverage audit

**Files:**
- Create: `src/vcd/coverage.rs`

Loads a real aiesim VCD file via wellen, walks its signal hierarchy through
the mapping tree, and reports mapped vs unmapped signals. This is the first
real integration test using wellen.

- [ ] **Step 1: Write test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_report_from_real_vcd() {
        // Skip if test VCD not available.
        // To obtain this file: run aiesimulator on a Chess-compiled test
        // with --dump-vcd. See docs/aiesimulator.md for instructions.
        let vcd_path = "/tmp/aiesim-test2/trace.vcd";
        if !std::path::Path::new(vcd_path).exists() {
            eprintln!("Skipping: {} not found (run aiesim to generate)", vcd_path);
            return;
        }

        // Build a partial tree from the subtrees available so far
        // (lock + DMA at minimum). build_aie2_mapping_tree() comes in Task 8.
        let tree = crate::vcd::mapping::MappingTree::new()
            .scope("top")
            .scope("math_engine")
            .tile_group("mem_row", &[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)])
            .locks(64)
            .dma(2, 2)
            .tile_group("array", &[(0, 2), (0, 3), (1, 2), (1, 3)])
            .locks(16)
            .dma(2, 2)
            .build();

        let report = coverage_audit(vcd_path, &tree).unwrap();

        // Should map at least the lock + DMA signals
        assert!(report.mapped_count > 0, "Expected some mapped signals");
        assert!(report.total_count > 100_000, "Expected 100K+ total signals");

        // Print the report for manual inspection
        eprintln!("{}", report);
    }
}
```

- [ ] **Step 2: Implement coverage_audit()**

```rust
//! Coverage audit: compare VCD signals against mapping tree.

use crate::vcd::mapping::MappingTree;
use crate::vcd::state_path::Subsystem;
use std::collections::BTreeMap;
use std::fmt;

/// Result of a coverage audit.
pub struct CoverageReport {
    pub total_count: usize,
    pub mapped_count: usize,
    pub unmapped_count: usize,
    /// Unmapped signals grouped by prefix (first 3 hierarchy levels).
    pub unmapped_groups: BTreeMap<String, usize>,
    /// Mapped signal counts per subsystem.
    pub mapped_by_subsystem: BTreeMap<Subsystem, usize>,
}

/// Run a coverage audit on a VCD file against a mapping tree.
pub fn coverage_audit(
    vcd_path: &str,
    tree: &MappingTree,
) -> Result<CoverageReport, String> {
    let waveform = wellen::simple::read(vcd_path)
        .map_err(|e| format!("Failed to read VCD: {}", e))?;
    let hierarchy = waveform.hierarchy();

    let mut total = 0usize;
    let mut mapped = 0usize;
    let mut unmapped_groups: BTreeMap<String, usize> = BTreeMap::new();
    let mut mapped_by_subsystem: BTreeMap<Subsystem, usize> = BTreeMap::new();

    // iter_vars() yields &Var directly (not VarRef).
    // Each Var has .full_name() and .signal_ref() methods.
    for var in hierarchy.iter_vars() {
        total += 1;
        let full_name = var.full_name(hierarchy);
        let segments: Vec<&str> = full_name.split('.').collect();

        if let Some(path) = tree.resolve(&segments) {
            mapped += 1;
            *mapped_by_subsystem.entry(path.subsystem()).or_insert(0) += 1;
        } else {
            // Group by first 3 segments for readable output
            let prefix = segments.iter().take(3).copied().collect::<Vec<_>>().join(".");
            *unmapped_groups.entry(prefix).or_insert(0) += 1;
        }
    }

    Ok(CoverageReport {
        total_count: total,
        mapped_count: mapped,
        unmapped_count: total - mapped,
        unmapped_groups,
        mapped_by_subsystem,
    })
}
```

Implement `Display` for `CoverageReport` to produce the text output shown in
the spec.

- [ ] **Step 3: Run tests**

Run: `cargo test --lib vcd::coverage -- --nocapture 2>&1 | tail -20`

Expected: If `/tmp/aiesim-test2/trace.vcd` exists, prints coverage report.
Lock and DMA signals should show as mapped. The unmapped count will be high
initially (streams, memory, core not yet mapped). That's expected.

- [ ] **Step 4: Commit**

```bash
git add src/vcd/coverage.rs
git commit -m "feat(vcd): coverage audit -- mapped vs unmapped VCD signals"
```

---

### Task 7: Stream, core, and event mapping subtrees

**Files:**
- Create: `src/vcd/stream_mapping.rs`
- Create: `src/vcd/core_mapping.rs`
- Create: `src/vcd/event_mapping.rs`

These three subtrees round out the major signal categories. They follow the
same pattern as lock and DMA mappings.

- [ ] **Step 1: Write stream_mapping tests, then implement**

Follow the TDD pattern from Tasks 4-5: write tests in `stream_mapping.rs`
first (can_resolve for port signals, enumerate counts), verify they fail,
then implement.

Stream switch signals per port: `data` (32-bit), `event_idle`, `event_running`,
`event_stalled`, `event_tlast` (all 1-bit). Port names from VCD:
`from_sSouth0..N`, `from_sNorth0..N`, `from_sDMA0..N`, `from_sTileCtrl`, etc.

The mapping takes a list of port names (derived from the device model or
hardcoded per tile type) and creates signal entries for each port.

- [ ] **Step 2: Write core_mapping tests, then implement**

Compute tile only. Signals: `pc_E1..pc_E7` (32-bit pipeline stages),
`pm_rd_in` (128-bit), `pm_ad_out` (20-bit), `tm_rd_in` (32-bit),
`tm_ad_out` (20-bit), `tm_wr_out` (32-bit), `tm_ld_out` (1-bit),
`tm_st_out` (1-bit), `reset` (1-bit), `pc_breakpoint_halted` (1-bit).

All under the `cm` (compute module) scope in the VCD hierarchy.

- [ ] **Step 3: Write event_mapping tests, then implement**

Event trace signals: `event{N}_{name}` where N is the event code and name
is the event name from VCD (e.g., `event73_INSTR_VECTOR`). These are 1-bit
level signals.

The mapping parses the event code from the signal name suffix.

- [ ] **Step 4: Run all three test suites**

Run: `cargo test --lib vcd::stream_mapping vcd::core_mapping vcd::event_mapping`

Expected: All tests pass.

- [ ] **Step 5: Re-run coverage audit**

Run: `cargo test --lib vcd::coverage -- --nocapture 2>&1 | tail -20`

Expected: Coverage percentage should jump significantly (streams alone are
~22K signals). Target: >80% mapped.

**Note:** Memory port mapping (~24K signals) is deferred to a follow-up task.
The emulator's memory port model is partial (bitmask tracking, not
port-level), so mapping these signals now would result in a large unmapped
block. Add memory port mapping once the emulator's memory model is extended.

- [ ] **Step 6: Commit**

```bash
git add src/vcd/stream_mapping.rs src/vcd/core_mapping.rs src/vcd/event_mapping.rs
git commit -m "feat(vcd): stream, core, and event mapping subtrees"
```

---

### Task 8: Wire up the full AIE2 mapping tree

**Files:**
- Modify: `src/vcd/mapping.rs`

Add `build_aie2_mapping_tree()` that assembles all subtrees into the complete
device-level mapping tree, matching the spec's declarative structure.

- [ ] **Step 1: Write test**

```rust
#[test]
fn aie2_tree_covers_all_tile_types() {
    let tree = build_aie2_mapping_tree();
    let paths = tree.enumerate_all();
    // Should have paths for shim, memtile, and compute tiles
    let has_shim = paths.iter().any(|p| p.tile().1 == 0);
    let has_memtile = paths.iter().any(|p| p.tile().1 == 1);
    let has_compute = paths.iter().any(|p| p.tile().1 >= 2);
    assert!(has_shim, "Missing shim tile paths");
    assert!(has_memtile, "Missing memtile paths");
    assert!(has_compute, "Missing compute tile paths");
}
```

- [ ] **Step 2: Implement build_aie2_mapping_tree()**

Uses the NPU1/Phoenix array layout: 5 columns x 6 rows.
- Row 0: shim tiles (5 tiles)
- Row 1: memtiles (5 tiles)
- Rows 2-5: compute tiles (20 tiles)

Assembles subtrees per tile type exactly as shown in the spec.

- [ ] **Step 3: Run coverage audit with full tree**

Expected: >90% mapped signals for the lock+DMA+stream+core+event set.
Memory port signals will remain partially unmapped (deferred to later).

- [ ] **Step 4: Commit**

```bash
git add src/vcd/mapping.rs
git commit -m "feat(vcd): complete AIE2 mapping tree for NPU1/Phoenix"
```

---

## Chunk 2: Tolerance Model and Comparison Engine

This chunk builds the comparison infrastructure. Once complete, you can
compare two VCD files and get a structured divergence report.

### Task 9: Tolerance configuration

**Files:**
- Create: `src/vcd/tolerance.rs`

Hierarchical tolerance resolution: per-tile-per-field > per-field >
per-subsystem > global default.

- [ ] **Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::vcd::state_path::*;

    #[test]
    fn default_tolerance_is_zero() {
        let config = ToleranceConfig::default();
        let path = StatePath::CorePc { col: 0, row: 3, stage: 1 };
        assert_eq!(config.resolve(&path), 0);
    }

    #[test]
    fn subsystem_override() {
        let config = ToleranceConfig::default()
            .with_subsystem(Subsystem::Dma, 1000);
        let dma_path = StatePath::DmaFsmState {
            col: 0, row: 1, dir: DmaDir::S2mm, ch: 0,
        };
        assert_eq!(config.resolve(&dma_path), 1000);

        // Core should still be 0
        let core_path = StatePath::CorePc { col: 0, row: 3, stage: 1 };
        assert_eq!(config.resolve(&core_path), 0);
    }

    #[test]
    fn field_override_beats_subsystem() {
        let config = ToleranceConfig::default()
            .with_subsystem(Subsystem::Dma, 1000)
            .with_field(Subsystem::Dma, "current_bd", 0);
        let bd_path = StatePath::DmaCurrentBd {
            col: 0, row: 1, dir: DmaDir::S2mm, ch: 0,
        };
        assert_eq!(config.resolve(&bd_path), 0);

        let fsm_path = StatePath::DmaFsmState {
            col: 0, row: 1, dir: DmaDir::S2mm, ch: 0,
        };
        assert_eq!(config.resolve(&fsm_path), 1000);
    }

    #[test]
    fn tile_override_beats_field() {
        let config = ToleranceConfig::default()
            .with_subsystem(Subsystem::Dma, 1000)
            .with_tile_override(Subsystem::Dma, 0, 0, 1500);
        let shim_path = StatePath::DmaFsmState {
            col: 0, row: 0, dir: DmaDir::S2mm, ch: 0,
        };
        assert_eq!(config.resolve(&shim_path), 1500);

        let memtile_path = StatePath::DmaFsmState {
            col: 0, row: 1, dir: DmaDir::S2mm, ch: 0,
        };
        assert_eq!(config.resolve(&memtile_path), 1000);
    }

    #[test]
    fn parse_from_toml() {
        let toml_str = r#"
            [vcd_compare.tolerance]
            default = 0
            [vcd_compare.tolerance.dma]
            default = 1000
            current_bd = 0
        "#;
        let config = ToleranceConfig::from_toml(toml_str).unwrap();
        let dma_path = StatePath::DmaFsmState {
            col: 0, row: 1, dir: DmaDir::S2mm, ch: 0,
        };
        assert_eq!(config.resolve(&dma_path), 1000);
    }
}
```

- [ ] **Step 2: Implement ToleranceConfig**

```rust
//! Configurable timing tolerance for VCD comparison.

use crate::vcd::state_path::{StatePath, Subsystem};
use std::collections::HashMap;

/// Hierarchical tolerance configuration.
///
/// Resolution order: tile override > field override > subsystem > global.
pub struct ToleranceConfig {
    global_default: u64,
    subsystem_defaults: HashMap<Subsystem, u64>,
    field_overrides: HashMap<(Subsystem, &'static str), u64>,
    tile_overrides: HashMap<(Subsystem, u8, u8), u64>,
}
```

Implement `resolve(&StatePath) -> u64`, builder methods, and TOML parsing.

- [ ] **Step 3: Run tests**

Run: `cargo test --lib vcd::tolerance 2>&1 | tail -10`

Expected: All 5 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/vcd/tolerance.rs
git commit -m "feat(vcd): hierarchical tolerance config with TOML parsing"
```

---

### Task 10: Comparison engine -- signal alignment

**Files:**
- Create: `src/vcd/compare.rs`

Phase 1 of comparison: load two VCD files via wellen, walk both signal
hierarchies through the mapping tree, match by StatePath.

- [ ] **Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_identical_files() {
        let vcd_path = "/tmp/aiesim-test2/trace.vcd";
        if !std::path::Path::new(vcd_path).exists() {
            eprintln!("Skipping: {} not found", vcd_path);
            return;
        }

        let tree = crate::vcd::mapping::build_aie2_mapping_tree();
        let alignment = align_signals(vcd_path, vcd_path, &tree).unwrap();

        // Comparing a file to itself: all mapped signals should align
        assert_eq!(alignment.only_in_a.len(), 0);
        assert_eq!(alignment.only_in_b.len(), 0);
        assert!(alignment.matched.len() > 0);
    }
}
```

- [ ] **Step 2: Implement align_signals()**

```rust
//! VCD comparison engine.

use crate::vcd::mapping::MappingTree;
use crate::vcd::state_path::StatePath;
use std::collections::HashMap;
use wellen::SignalRef;

/// Timescale normalization config.
pub struct TimescaleConfig {
    /// Clock period in picoseconds (952 for AIE2).
    pub clock_period_ps: u64,
}

impl Default for TimescaleConfig {
    fn default() -> Self {
        TimescaleConfig { clock_period_ps: 952 }
    }
}

/// Result of signal alignment between two VCD files.
pub struct SignalAlignment {
    /// Signals matched by StatePath: (path, signal_ref_a, signal_ref_b).
    pub matched: Vec<(StatePath, SignalRef, SignalRef)>,
    /// Signals only in file A (aiesim).
    pub only_in_a: Vec<StatePath>,
    /// Signals only in file B (emulator).
    pub only_in_b: Vec<StatePath>,
}

/// Full comparison context: owns both waveforms so they stay alive
/// through alignment and sweep phases.
pub struct CompareContext {
    pub wave_a: wellen::Waveform,
    pub wave_b: wellen::Waveform,
    pub alignment: SignalAlignment,
}

/// Load two VCD files and align signals by StatePath.
///
/// Returns a `CompareContext` that owns both waveforms. The caller
/// uses this context for the sweep phase (loading signal data lazily).
pub fn align_signals(
    path_a: &str,
    path_b: &str,
    tree: &MappingTree,
) -> Result<CompareContext, String> {
    // read() loads header + raw body into memory. Individual signal data
    // is decoded lazily via load_signals() in the sweep phase.
    let wave_a = wellen::simple::read(path_a)
        .map_err(|e| format!("Failed to read VCD A: {}", e))?;
    let wave_b = wellen::simple::read(path_b)
        .map_err(|e| format!("Failed to read VCD B: {}", e))?;

    // Map each VCD's signals to StatePaths
    let map_a = resolve_hierarchy(&wave_a, tree);
    let map_b = resolve_hierarchy(&wave_b, tree);

    // Match by StatePath
    let mut matched = Vec::new();
    let mut only_in_a = Vec::new();

    for (path, sig_ref) in &map_a {
        if let Some(&sig_ref_b) = map_b.get(path) {
            matched.push((path.clone(), *sig_ref, sig_ref_b));
        } else {
            only_in_a.push(path.clone());
        }
    }

    let only_in_b: Vec<_> = map_b.keys()
        .filter(|p| !map_a.contains_key(p))
        .cloned()
        .collect();

    let alignment = SignalAlignment { matched, only_in_a, only_in_b };
    Ok(CompareContext { wave_a, wave_b, alignment })
}

/// Resolve all signals in a waveform to StatePaths via the mapping tree.
fn resolve_hierarchy(
    waveform: &wellen::Waveform,
    tree: &MappingTree,
) -> HashMap<StatePath, SignalRef> {
    let hierarchy = waveform.hierarchy();
    let mut map = HashMap::new();

    // iter_vars() yields &Var directly. Each Var has signal_ref()
    // and full_name() methods.
    for var in hierarchy.iter_vars() {
        let full_name = var.full_name(hierarchy);
        let segments: Vec<&str> = full_name.split('.').collect();

        if let Some(path) = tree.resolve(&segments) {
            map.insert(path, var.signal_ref());
        }
    }

    map
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test --lib vcd::compare 2>&1 | tail -10`

Expected: Pass (self-comparison shows all matched, none asymmetric).

- [ ] **Step 4: Commit**

```bash
git add src/vcd/compare.rs
git commit -m "feat(vcd): signal alignment -- match VCD signals by StatePath"
```

---

### Task 11: Comparison engine -- subsystem sweep

**Files:**
- Modify: `src/vcd/compare.rs`

Phase 2: for each matched signal pair, load transitions from both waveforms,
compare value sequences and timing deltas.

- [ ] **Step 1: Write tests**

```rust
#[test]
fn compare_identical_files_all_match() {
    let vcd_path = "/tmp/aiesim-test2/trace.vcd";
    if !std::path::Path::new(vcd_path).exists() {
        eprintln!("Skipping: {} not found", vcd_path);
        return;
    }

    let tree = crate::vcd::mapping::build_aie2_mapping_tree();
    let tolerance = ToleranceConfig::default();
    let timescale = TimescaleConfig::default();
    let result = compare_vcds(vcd_path, vcd_path, &tree, &tolerance, &timescale)
        .unwrap();

    // Self-comparison: all signals should match values and timing
    assert!(result.value_divergences.is_empty());
    for sub_result in result.by_subsystem.values() {
        assert_eq!(sub_result.max_timing_delta, 0);
    }
}
```

- [ ] **Step 2: Implement compare_vcds()**

Key types:

```rust
/// Per-signal comparison result.
pub struct SignalComparison {
    pub path: StatePath,
    pub values_match: bool,
    /// Index of first divergent transition (if values don't match).
    pub first_diverge_idx: Option<usize>,
    /// Cycle of first divergent transition.
    pub first_diverge_cycle: Option<u64>,
    /// Maximum timing delta across matching transitions.
    pub max_timing_delta: u64,
    /// Whether timing is within tolerance.
    pub timing_within_tolerance: bool,
}

/// Per-subsystem aggregated result.
pub struct SubsystemResult {
    pub subsystem: Subsystem,
    pub signal_count: usize,
    pub values_match: bool,
    pub max_timing_delta: u64,
    pub timing_within_tolerance: bool,
    pub first_value_divergence: Option<SignalComparison>,
}

/// Full comparison result.
pub struct CompareResult {
    pub by_subsystem: BTreeMap<Subsystem, SubsystemResult>,
    pub value_divergences: Vec<SignalComparison>,
    pub alignment: SignalAlignment,
}
```

Implementation:
1. Call `align_signals()` to get matched pairs.
2. Group matched pairs by subsystem.
3. For each subsystem, load signals lazily via `waveform.load_signals()`.
4. For each matched pair, iterate transitions using `signal.iter_changes()`.
5. Normalize timestamps to cycles (aiesim: divide by clock period).
6. Compare value sequences (order-sensitive, cycle-insensitive).
7. For value-matched signals, compute timing deltas at each transition.
8. Aggregate per-subsystem results.
9. Unload signals after each subsystem to limit memory usage.

- [ ] **Step 3: Run tests**

Run: `cargo test --lib vcd::compare 2>&1 | tail -10`

Expected: All compare tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/vcd/compare.rs
git commit -m "feat(vcd): subsystem sweep -- value + timing comparison"
```

---

### Task 12: Report generation

**Files:**
- Create: `src/vcd/report.rs`

Text and JSON output from CompareResult.

- [ ] **Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_report_shows_subsystems() {
        let result = make_test_result();
        let text = format_text_report(&result);
        assert!(text.contains("Subsystem"));
        assert!(text.contains("PASS") || text.contains("FAIL"));
    }

    #[test]
    fn json_report_is_valid() {
        let result = make_test_result();
        let json = format_json_report(&result);
        // Should parse as valid JSON
        let _: serde_json::Value = serde_json::from_str(&json).unwrap();
    }
}
```

- [ ] **Step 2: Implement format_text_report() and format_json_report()**

Text format matches the spec example. JSON format uses serde_json::to_string_pretty()
on a serializable `CompareReport` struct.

- [ ] **Step 3: Run tests**

Run: `cargo test --lib vcd::report 2>&1 | tail -5`

Expected: Pass.

- [ ] **Step 4: Commit**

```bash
git add src/vcd/report.rs
git commit -m "feat(vcd): text and JSON report generation"
```

---

## Chunk 3: Standalone Tool, VCD Emission, and Bridge Integration

### Task 13: Standalone vcd-compare binary

**Files:**
- Create: `src/bin/vcd_compare.rs`
- Modify: `Cargo.toml` (add `[[bin]]` section)

- [ ] **Step 1: Add binary to Cargo.toml**

```toml
[[bin]]
name = "vcd-compare"
path = "src/bin/vcd_compare.rs"
```

- [ ] **Step 2: Implement CLI**

```rust
//! Standalone VCD comparison tool.
//!
//! Usage:
//!   vcd-compare <aiesim.vcd> <emulator.vcd> [--tolerance=config.toml] [--json]

use xdna_emu::vcd::{
    compare::compare_vcds,
    mapping::build_aie2_mapping_tree,
    tolerance::ToleranceConfig,
    compare::TimescaleConfig,
    report::{format_text_report, format_json_report},
};

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: vcd-compare <file_a.vcd> <file_b.vcd> [--tolerance=config.toml] [--json]");
        std::process::exit(1);
    }

    let path_a = &args[1];
    let path_b = &args[2];

    let json_mode = args.iter().any(|a| a == "--json");
    let tolerance = args.iter()
        .find(|a| a.starts_with("--tolerance="))
        .map(|a| {
            let path = a.strip_prefix("--tolerance=").unwrap();
            let content = std::fs::read_to_string(path)
                .unwrap_or_else(|e| {
                    eprintln!("Failed to read tolerance config: {}", e);
                    std::process::exit(1);
                });
            ToleranceConfig::from_toml(&content)
                .unwrap_or_else(|e| {
                    eprintln!("Failed to parse tolerance config: {}", e);
                    std::process::exit(1);
                })
        })
        .unwrap_or_default();

    let tree = build_aie2_mapping_tree();
    let timescale = TimescaleConfig::default();

    let result = compare_vcds(path_a, path_b, &tree, &tolerance, &timescale)
        .unwrap_or_else(|e| {
            eprintln!("Comparison failed: {}", e);
            std::process::exit(1);
        });

    if json_mode {
        println!("{}", format_json_report(&result));
    } else {
        println!("{}", format_text_report(&result));
    }

    // Exit code: 0 if all values match, 1 if divergence
    if result.value_divergences.is_empty() {
        std::process::exit(0);
    } else {
        std::process::exit(1);
    }
}
```

- [ ] **Step 3: Build and smoke test**

Run: `cargo build --bin vcd-compare 2>&1 | tail -3`

Expected: Build succeeds.

If `/tmp/aiesim-test2/trace.vcd` exists:

Run: `cargo run --bin vcd-compare -- /tmp/aiesim-test2/trace.vcd /tmp/aiesim-test2/trace.vcd 2>&1`

Expected: Self-comparison shows all PASS with 0 timing deltas.

- [ ] **Step 4: Commit**

```bash
git add src/bin/vcd_compare.rs Cargo.toml
git commit -m "feat(vcd): standalone vcd-compare binary"
```

---

### Task 14: VCD emission from emulator (feature-flagged)

**Files:**
- Create: `src/vcd/emit.rs`

This is behind `cfg(feature = "vcd-recording")`. Uses the `vcd` crate's
Writer API to produce VCD output during emulator execution.

- [ ] **Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn emits_valid_vcd_header() {
        let tree = crate::vcd::mapping::build_aie2_mapping_tree();
        let mut buf = Vec::new();
        let mut recorder = VcdRecorder::new(&mut buf, &tree, 952).unwrap();
        recorder.finish().unwrap();

        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("$timescale"));
        assert!(output.contains("$scope"));
        assert!(output.contains("$var"));
        assert!(output.contains("$enddefinitions"));
    }

    #[test]
    fn records_lock_transition() {
        let tree = crate::vcd::mapping::build_aie2_mapping_tree();
        let mut buf = Vec::new();
        {
            let mut recorder = VcdRecorder::new(&mut buf, &tree, 952).unwrap();
            let path = StatePath::LockValue { col: 0, row: 1, idx: 0 };
            recorder.record_word(0, &path, 0);
            recorder.record_word(100, &path, 1);
            recorder.finish().unwrap();
        }
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("#0"));
        assert!(output.contains("#100"));
    }
}
```

- [ ] **Step 2: Implement VcdRecorder**

```rust
//! VCD emission from the emulator.
//!
//! Behind `cfg(feature = "vcd-recording")`. Uses the `vcd` crate Writer.

use crate::vcd::mapping::MappingTree;
use crate::vcd::state_path::StatePath;
use std::collections::HashMap;
use std::io::Write;

/// VCD recorder that translates emulator state changes to VCD output.
pub struct VcdRecorder<W: Write> {
    writer: vcd::Writer<W>,
    /// Map from StatePath to VCD IdCode for value change writing.
    path_to_id: HashMap<StatePath, vcd::IdCode>,
}

impl<W: Write> VcdRecorder<W> {
    /// Create a new recorder, writing VCD header from the mapping tree.
    pub fn new(
        output: W,
        tree: &MappingTree,
        clock_period_ps: u32,
    ) -> Result<Self, String> {
        let mut writer = vcd::Writer::new(output);
        writer.timescale(clock_period_ps, vcd::TimescaleUnit::PS)
            .map_err(|e| format!("VCD timescale: {}", e))?;

        let mut path_to_id = HashMap::new();
        // Walk the mapping tree to emit $scope/$var declarations
        // and build the path -> IdCode lookup table.
        tree.emit_vcd_header(&mut writer, &mut path_to_id)?;

        writer.enddefinitions()
            .map_err(|e| format!("VCD enddefinitions: {}", e))?;

        Ok(VcdRecorder { writer, path_to_id })
    }

    /// Record a 1-bit state change.
    ///
    /// Recording errors are silently discarded (via `let _ = ...`) because
    /// VCD recording failures must not be fatal to emulation. The VCD file
    /// may be truncated but the emulation continues correctly.
    pub fn record_bit(&mut self, cycle: u64, path: &StatePath, value: bool) {
        if let Some(&id) = self.path_to_id.get(path) {
            let _ = self.writer.timestamp(cycle);
            let v = if value { vcd::Value::V1 } else { vcd::Value::V0 };
            let _ = self.writer.change_scalar(id, v);
        }
    }

    /// Record a multi-bit state change.
    ///
    /// Emits all 32 bits regardless of actual signal width. This produces
    /// valid VCD with leading zeros for narrower signals. A future
    /// optimization could use the signal width from the mapping tree to
    /// emit only the relevant bits.
    pub fn record_word(&mut self, cycle: u64, path: &StatePath, value: u32) {
        if let Some(&id) = self.path_to_id.get(path) {
            let _ = self.writer.timestamp(cycle);
            let bits: Vec<vcd::Value> = (0..32).rev()
                .map(|i| if (value >> i) & 1 == 1 { vcd::Value::V1 } else { vcd::Value::V0 })
                .collect();
            let _ = self.writer.change_vector(id, bits);
        }
    }

    /// Flush and finalize VCD output.
    pub fn finish(mut self) -> Result<(), String> {
        self.writer.flush().map_err(|e| format!("VCD flush: {}", e))
    }
}
```

This requires `emit_vcd_header()` on `MappingTree` (added in Task 3's
implementation). That method walks the tree, calls `writer.add_module()`,
`writer.add_wire()`, `writer.upscope()` for each level, and populates the
`path_to_id` map.

- [ ] **Step 3: Run tests with feature flag**

Run: `cargo test --lib --features vcd-recording vcd::emit 2>&1 | tail -10`

Expected: Both tests pass.

- [ ] **Step 4: Verify default build excludes emit module**

Run: `cargo test --lib vcd::emit 2>&1 | tail -5`

Expected: No tests found (module not compiled without feature).

- [ ] **Step 5: Commit**

```bash
git add src/vcd/emit.rs
git commit -m "feat(vcd): VCD emission from emulator (behind vcd-recording feature)"
```

---

### Task 15: Bridge test integration

**Files:**
- Modify: `scripts/emu-bridge-test.sh`

Add `--aiesim` flag that runs aiesimulator + emulator (VCD build) + vcd-compare
on each Chess test.

- [ ] **Step 1: Add --aiesim flag parsing**

In the argument parsing section of `emu-bridge-test.sh`, add:

```bash
--aiesim)
    AIESIM=true
    # aiesim requires Chess builds
    if [[ "$PEANO_ONLY" == "true" ]]; then
        echo "ERROR: --aiesim requires Chess builds (incompatible with --peano-only)"
        exit 1
    fi
    shift ;;
```

- [ ] **Step 2: Add aiesim phase after EMU phase**

After the EMU run phase, add a new phase that:
1. Checks if Chess `.prj/sim/` directory exists.
2. Runs aiesimulator with `--dump-vcd` (using existing `run_unit_simulation`
   or direct invocation).
3. Builds the emulator with `--features vcd-recording` if not already built.
4. Runs the emulator with `--dump-vcd` flag.
5. Runs `vcd-compare` on the two VCD outputs.
6. Copies results to `$RESULTS_DIR/$test/chess/aiesim/`.

- [ ] **Step 3: Test with a single test**

Run (outside sandbox, requires license):

```bash
./scripts/emu-bridge-test.sh --aiesim --no-hw -v 01_precompiled_core_function
```

Expected: aiesim phase runs, VCD comparison report printed.

- [ ] **Step 4: Commit**

```bash
git add scripts/emu-bridge-test.sh
git commit -m "feat(vcd): bridge test --aiesim flag for VCD comparison"
```

---

### Task 16: Create stub files for compilation

**Files:**
- Create remaining stub files needed for `cargo check` to pass

Before any implementation begins, create minimal stub files so the module
compiles. Each file just needs enough content to satisfy the module
declarations in `mod.rs`.

- [ ] **Step 1: Create all stub files**

Each file gets a doc comment and minimal placeholder content:

- `src/vcd/compare.rs`: empty module with doc comment
- `src/vcd/coverage.rs`: empty module with doc comment
- `src/vcd/tolerance.rs`: empty module with doc comment
- `src/vcd/report.rs`: empty module with doc comment
- `src/vcd/lock_mapping.rs`: empty module with doc comment
- `src/vcd/dma_mapping.rs`: empty module with doc comment
- `src/vcd/stream_mapping.rs`: empty module with doc comment
- `src/vcd/core_mapping.rs`: empty module with doc comment
- `src/vcd/event_mapping.rs`: empty module with doc comment

- [ ] **Step 2: Verify cargo check passes**

Run: `cargo check 2>&1 | tail -5`

Expected: Compiles cleanly (no errors). Warnings about unused imports are OK.

- [ ] **Step 3: Commit**

```bash
git add src/vcd/
git commit -m "feat(vcd): compilation stubs for all submodules"
```

**Important:** This task should be done FIRST (before Task 2) so that
`cargo check` passes throughout development. The plan lists it last for
logical flow, but execute it as part of Task 1.

---

## Implementation Notes

### Execution order

The recommended execution order is:

1. Task 1 + Task 16 (skeleton + stubs -- get cargo check passing)
2. Task 2 (StatePath)
3. Task 3 (mapping tree infrastructure)
4. Task 4 (lock mapping -- first end-to-end validation)
5. Task 5 (DMA mapping)
6. Task 6 (coverage audit -- first real VCD test)
7. Task 7 (stream, core, event mappings)
8. Task 8 (full AIE2 tree)
9. Task 9 (tolerance config)
10. Task 10 (signal alignment)
11. Task 11 (subsystem sweep)
12. Task 12 (report generation)
13. Task 13 (standalone binary)
14. Task 14 (VCD emission)
15. Task 15 (bridge integration)

### Testing strategy

- Tasks 2-5, 7-9: Pure unit tests (no VCD files needed)
- Tasks 6, 8: Integration tests using `/tmp/aiesim-test2/trace.vcd`
  (from previous aiesim run). Tests skip gracefully if file is missing.
- Tasks 10-12: Integration tests comparing VCD files
- Task 13: Build + smoke test
- Task 14: Unit tests with feature flag (`--features vcd-recording`)
- Task 15: Manual end-to-end test (requires license, outside sandbox)

### Key reference files in the emulator

When implementing recording call sites (Task 14), these are the emulator
files where VCD recording calls need to be added:

| Emulator file | What to record |
|---------------|----------------|
| `src/device/tile.rs` (Lock) | `LockValue` on acquire/release |
| `src/device/dma/channel.rs` (ChannelFsm) | `DmaFsmState` on FSM transitions |
| `src/device/dma/engine.rs` (DmaEngine) | `DmaAddress`, `DmaProcessedBytes` during transfer |
| `src/device/stream_switch.rs` (StreamPort) | `StreamPortIdle/Running/Stalled/Tlast` per cycle |
| `src/interpreter/state/context.rs` (ExecutionContext) | `CorePc` on PC update |

### wellen API usage pattern

```rust
// Load VCD -- reads header + raw body into memory.
// Individual signal data is decoded lazily via load_signals().
let mut waveform = wellen::simple::read("file.vcd")?;

// Access hierarchy
let hierarchy = waveform.hierarchy();
let timescale = hierarchy.timescale(); // Option<Timescale>

// Find signals by walking hierarchy.
// iter_vars() yields &Var directly (not VarRef).
for var in hierarchy.iter_vars() {
    let name = var.full_name(hierarchy);
    let signal_ref = var.signal_ref();
    // ... resolve through mapping tree
}

// Load specific signals (decodes from raw body -- this is the lazy part)
waveform.load_signals(&[signal_ref_a, signal_ref_b]);

// Read transitions.
// iter_changes() yields (TimeTableIdx, SignalValue) tuples.
// TimeTableIdx is u32, index into time_table.
if let Some(signal) = waveform.get_signal(signal_ref_a) {
    let time_table = waveform.time_table();
    for (time_idx, value) in signal.iter_changes() {
        let timestamp = time_table[time_idx as usize];
        // ...
    }
}

// Free memory
waveform.unload_signals(&[signal_ref_a, signal_ref_b]);
```

### vcd crate Writer usage pattern

```rust
use vcd::{Writer, Value, TimescaleUnit, SimulationCommand};

let mut writer = Writer::new(output);
writer.timescale(952, TimescaleUnit::PS)?;

// Hierarchy
writer.add_module("math_engine")?;
  writer.add_module("mem_row")?;
    writer.add_module("tile_0_1")?;
      writer.add_module("locks")?;
        let id = writer.add_wire(32, "value_0")?;
      writer.upscope()?;
    writer.upscope()?;
  writer.upscope()?;
writer.upscope()?;
writer.enddefinitions()?;

// Value changes
writer.timestamp(0)?;
writer.change_vector(id, (0..32).map(|_| Value::V0))?;

writer.timestamp(100)?;
let value: u32 = 1;
let bits: Vec<Value> = (0..32).rev()
    .map(|i| if (value >> i) & 1 == 1 { Value::V1 } else { Value::V0 })
    .collect();
writer.change_vector(id, bits)?;
```
