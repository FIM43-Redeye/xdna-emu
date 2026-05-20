# Interrupt Tier B Spec 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`../specs/2026-05-19-interrupt-tier-b-firmware-mailbox-design.md`](../specs/2026-05-19-interrupt-tier-b-firmware-mailbox-design.md)

**Goal:** Build the firmware async-error pipeline plumbing (per-handle cache + per-column mailbox rings + push callback + driver-mirror categorization + FFI + plugin ioctl wire-up), with `INSTR_ERROR` (event 69) as the demonstrating producer. Tier A's L1/L2 latch path stays untouched; Tier B fires in parallel.

**Architecture:** New `device::async_errors` subsystem (sink + types) with categorization tables in archspec. Hook lives in `state::effects::apply_tile_local_effects` next to existing Tier A `seed_broadcasts_for_event` + `tap_l1_interrupt` calls. Five FFI symbols expose the three surfaces (cache, ring, callback). Plugin replaces the no-op `DRM_AMDXDNA_HW_LAST_ASYNC_ERR` case with a `resolve_required` cache read.

**Tech Stack:** Rust (workspace crates `xdna-emu-core`, `xdna-archspec`, `xdna-emu-ffi`), C++ (XRT plugin), POSIX FFI.

---

## File map

**Create:**
- `crates/xdna-archspec/src/aie2/async_errors.rs` — categorization tables, lookups, encoding helpers, enums
- `src/device/async_errors/mod.rs` — module root, re-exports
- `src/device/async_errors/types.rs` — wire-format types (`AieError`, `AieErrInfoHeader`, `AmdxdnaAsyncError`), `AsyncRing`
- `src/device/async_errors/sink.rs` — `AsyncErrorSink` (cache + per-column rings + drain queue + `record_error`)
- `crates/xdna-emu-ffi/src/async_errors.rs` — FFI symbols

**Modify:**
- `crates/xdna-archspec/src/aie2/mod.rs` — `pub mod async_errors;`
- `src/device/mod.rs` — `pub mod async_errors;`
- `src/device/state/mod.rs` — add `pub async_errors: AsyncErrorSink` field + `new()` init + clear in any context-reset path
- `src/device/state/effects.rs` — restructure the `is_event_generate` match to return `Option<AieErrorOrigin>` instead of bool; add Tier B hook
- `crates/xdna-emu-ffi/src/lib.rs` — `mod async_errors; pub use async_errors::*;` + `async_callback` field on `XdnaEmuHandle`
- `crates/xdna-emu-ffi/src/execution.rs` — drain `newly_recorded` after each `engine.step()` and fire callback if registered
- `xrt-plugin/src/transport_inprocess.h` — typedef + member
- `xrt-plugin/src/transport_inprocess.cpp` — `resolve_required` call
- `xrt-plugin/src/platform_emu.cpp` — replace `DRM_AMDXDNA_HW_LAST_ASYNC_ERR` no-op

---

## Conventions for this plan

**TDD discipline:** every task starts with a failing test, then minimal impl. Run `cargo build` (no `--release`) and `cargo test --lib` bare — never piped through `tail`/`grep`/`head` (project CLAUDE.md, "Long-running commands" section).

**Sandbox-safe test runs:** `TMPDIR=/tmp/claude-1000 cargo test --lib --no-fail-fast <filter>`.

**Workspace lookup tips for subagents:**
- Existing FFI pattern: read `crates/xdna-emu-ffi/src/query.rs` (null-handle convention, last_error pattern, repr-C return types)
- Existing FFI tests pattern: read `crates/xdna-emu-ffi/src/lib.rs` tests (`with_handle` helper, `xdna_emu_create`/`destroy` lifecycle)
- Existing subsystem pattern: read `src/device/interrupts/mod.rs` + `l1.rs` + `l2.rs` for two-file subsystem layout
- Existing Tier A hook site: `src/device/state/effects.rs:367-406` (the `is_event_generate` block) — Tier B hook lives directly adjacent
- Existing test fixture for event generation: `src/device/state/effects.rs:638-647` (`fire_event_generate_for_test`)
- `TileKind` enum: `crates/xdna-archspec/src/types.rs:71` (variants: `Compute, Mem, ShimNoc, ShimPl`)

**Commit message style:** match recent Tier A commits (`git log -10 --oneline`). Subject form: `interrupt: <imperative summary>`. Body ends with literally:

```
Generated using Claude Code.
```

No emoji anywhere (project + global CLAUDE.md).

---

## Task 1: Archspec enums + module skeleton

Establishes the type vocabulary the rest of the plan uses.

**Files:**
- Create: `crates/xdna-archspec/src/aie2/async_errors.rs`
- Modify: `crates/xdna-archspec/src/aie2/mod.rs`

- [ ] **Step 1: Write failing tests for the enums**

Append to `crates/xdna-archspec/src/aie2/async_errors.rs` (this is the new file; start with this content):

```rust
//! Async-error categorization tables and encoding helpers.
//!
//! Direct port of `xdna-driver/src/driver/amdxdna/aie2_error.c` static tables
//! and `xdna-driver/src/driver/amdxdna/amdxdna_error.h` encoding macros.
//! Used by Tier B's `AsyncErrorSink` to turn an `(event_id, origin, row, col)`
//! tuple into the `amdxdna_async_error` record the host ioctl returns.
//!
//! Design source: `docs/superpowers/specs/2026-05-19-interrupt-tier-b-firmware-mailbox-design.md`.

use crate::types::TileKind;

/// Origin module of an AIE error event.
///
/// This is the emu-internal enum used for categorization-table dispatch
/// (4 variants). The driver's on-wire `enum aie_module_type` has only 3
/// values (MEM=0, CORE=1, PL=2) and disambiguates memtile from compute-mem
/// at the receiver via the row field. See `wire_mod_type()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AieErrorOrigin {
    Core,
    Mem,
    MemTile,
    Pl,
}

impl AieErrorOrigin {
    /// Map to the driver's wire-format `enum aie_module_type` value.
    /// Mirrors `aie2_error.c:34-37`: MEM=0, CORE=1, PL=2 (memtile shares MEM).
    pub fn wire_mod_type(self) -> u32 {
        match self {
            AieErrorOrigin::Mem | AieErrorOrigin::MemTile => 0,
            AieErrorOrigin::Core => 1,
            AieErrorOrigin::Pl => 2,
        }
    }

    /// Derive from `TileKind`. Compute -> Core (mem-module of compute tiles
    /// must use `AieErrorOrigin::Mem` explicitly; this helper covers the
    /// common case where the caller knows only the tile, not which module
    /// of it fired). ShimNoc/ShimPl both map to Pl.
    pub fn from_tile_kind(kind: TileKind) -> Self {
        match kind {
            TileKind::Compute => AieErrorOrigin::Core,
            TileKind::Mem => AieErrorOrigin::MemTile,
            TileKind::ShimNoc | TileKind::ShimPl => AieErrorOrigin::Pl,
        }
    }
}

/// Driver `enum aie_error_category` (aie2_error.c:39-53).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AieErrorCategory {
    Saturation,
    Fp,
    Stream,
    Access,
    Bus,
    Instruction,
    Ecc,
    Lock,
    Dma,
    MemParity,
    Unknown,
}

/// Driver `enum amdxdna_error_num`. Values mirror the driver enum
/// (`amdxdna_error.h:37-53`) verbatim so encoded `err_code` bytes match
/// what a real driver would emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum AmdxdnaErrorNum {
    FirewallTrip = 1,
    TempHigh = 2,
    AieSaturation = 3,
    AieFp = 4,
    AieStream = 5,
    AieAccess = 6,
    AieBus = 7,
    AieInstruction = 8,
    AieEcc = 9,
    AieLock = 10,
    AieDma = 11,
    AieMemParity = 12,
    KdsCu = 13,
    KdsExec = 14,
    Unknown = 15,
}

/// Driver `enum amdxdna_error_driver` (`amdxdna_error.h:55-60`). Tier B always
/// emits `Aie` (via the convenience `build_critical_aie_error_code` helper).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum AmdxdnaErrorDriver {
    Xocl = 1,
    Xclmgmt = 2,
    Zocl = 3,
    Aie = 4,
    Unknown = 5,
}

/// Driver `enum amdxdna_error_severity` (`amdxdna_error.h:63-72`). Tier B always
/// emits `Critical` for AIE async errors -- matches the driver's hardcoded
/// `AMDXDNA_CRITICAL_ERROR_CODE_BUILD` macro path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum Severity {
    Emergency = 1,
    Alert = 2,
    Critical = 3,
    Error = 4,
    Warning = 5,
    Notice = 6,
    Info = 7,
    Debug = 8,
    Unknown = 9,
}

/// Driver `enum amdxdna_error_module` (`amdxdna_error.h:75-83`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum AmdxdnaErrorModule {
    Firewall = 1,
    Cmc = 2,
    AieCore = 3,
    AieMemory = 4,
    AieShim = 5,
    AieNoc = 6,
    AiePl = 7,
    Unknown = 8,
}

/// Driver `enum amdxdna_error_class` (`amdxdna_error.h:86-92`). Tier B always
/// emits `Aie` for AIE async errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum Class {
    System = 1,
    Aie = 2,
    Hardware = 3,
    Unknown = 4,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_mod_type_matches_driver_enum() {
        assert_eq!(AieErrorOrigin::Mem.wire_mod_type(), 0);
        assert_eq!(AieErrorOrigin::Core.wire_mod_type(), 1);
        assert_eq!(AieErrorOrigin::Pl.wire_mod_type(), 2);
        // MemTile shares MEM_MOD per driver (disambiguated by row at receiver).
        assert_eq!(AieErrorOrigin::MemTile.wire_mod_type(), 0);
    }

    #[test]
    fn origin_from_tile_kind() {
        assert_eq!(AieErrorOrigin::from_tile_kind(TileKind::Compute), AieErrorOrigin::Core);
        assert_eq!(AieErrorOrigin::from_tile_kind(TileKind::Mem), AieErrorOrigin::MemTile);
        assert_eq!(AieErrorOrigin::from_tile_kind(TileKind::ShimNoc), AieErrorOrigin::Pl);
        assert_eq!(AieErrorOrigin::from_tile_kind(TileKind::ShimPl), AieErrorOrigin::Pl);
    }
}
```

- [ ] **Step 2: Wire the new module into archspec**

Open `crates/xdna-archspec/src/aie2/mod.rs` and add the module declaration alongside the existing `pub mod` lines (alphabetical order if file is alphabetized; else at the end of the `pub mod` block):

```rust
pub mod async_errors;
```

- [ ] **Step 3: Run the tests to verify they pass**

```bash
cargo test -p xdna-archspec --lib async_errors
```

Expected: `test result: ok. N passed; 0 failed` for the two tests in step 1.

- [ ] **Step 4: Commit**

```bash
git add crates/xdna-archspec/src/aie2/async_errors.rs crates/xdna-archspec/src/aie2/mod.rs
git commit -m "$(cat <<'EOF'
interrupt: add Tier B async_errors enum scaffolding in archspec

AieErrorOrigin (4 variants for categorization dispatch, wire_mod_type
maps to driver's 3-value aie_module_type). AieErrorCategory,
AmdxdnaErrorNum, AmdxdnaErrorModule, Severity, Class enums for
async-error record encoding. Empty bodies; tables and helpers land in
the next tasks.

Generated using Claude Code.
EOF
)"
```

---

## Task 2: Categorization tables + lookup

Ports the driver's static event-category tables and the dispatch function.

**Files:**
- Modify: `crates/xdna-archspec/src/aie2/async_errors.rs`

- [ ] **Step 1: Write the failing categorization tests**

Append to `crates/xdna-archspec/src/aie2/async_errors.rs` (inside the existing `#[cfg(test)] mod tests`):

```rust
    #[test]
    fn core_event_categories_match_driver_table() {
        // Direct port of aie2_error.c:105-120 core table.
        let cases: &[(u8, AieErrorCategory)] = &[
            (55, AieErrorCategory::Access),
            (56, AieErrorCategory::Stream),
            (57, AieErrorCategory::Stream),
            (58, AieErrorCategory::Bus),
            (59, AieErrorCategory::Instruction),
            (60, AieErrorCategory::Access),
            (62, AieErrorCategory::Ecc),
            (64, AieErrorCategory::Ecc),
            (65, AieErrorCategory::Access),
            (66, AieErrorCategory::Access),
            (67, AieErrorCategory::Lock),
            (70, AieErrorCategory::Instruction),
            (71, AieErrorCategory::Stream),
            (72, AieErrorCategory::Bus),
            // Emu-specific extension: INSTR_ERROR=69 (mlir-aie naming).
            // Driver table omits this; spec section 2 decisions table
            // explains the divergence.
            (69, AieErrorCategory::Instruction),
        ];
        for &(event_id, expected) in cases {
            assert_eq!(
                event_to_category(event_id, AieErrorOrigin::Core),
                Some(expected),
                "core event {event_id} should map to {expected:?}"
            );
        }
    }

    #[test]
    fn mem_event_categories_match_driver_table() {
        // Direct port of aie2_error.c:89-103 mem table.
        let cases: &[(u8, AieErrorCategory)] = &[
            (88, AieErrorCategory::Ecc),
            (90, AieErrorCategory::Ecc),
            (91, AieErrorCategory::MemParity),
            (92, AieErrorCategory::MemParity),
            (93, AieErrorCategory::MemParity),
            (94, AieErrorCategory::MemParity),
            (95, AieErrorCategory::MemParity),
            (96, AieErrorCategory::MemParity),
            (97, AieErrorCategory::Dma),
            (98, AieErrorCategory::Dma),
            (99, AieErrorCategory::Dma),
            (100, AieErrorCategory::Dma),
            (101, AieErrorCategory::Lock),
        ];
        for &(event_id, expected) in cases {
            assert_eq!(
                event_to_category(event_id, AieErrorOrigin::Mem),
                Some(expected),
                "mem event {event_id} should map to {expected:?}"
            );
        }
    }

    #[test]
    fn memtile_event_categories_match_driver_table() {
        // Direct port of aie2_error.c:122-132 memtile table.
        let cases: &[(u8, AieErrorCategory)] = &[
            (130, AieErrorCategory::Ecc),
            (132, AieErrorCategory::Ecc),
            (133, AieErrorCategory::Dma),
            (134, AieErrorCategory::Dma),
            (135, AieErrorCategory::Stream),
            (136, AieErrorCategory::Stream),
            (137, AieErrorCategory::Stream),
            (138, AieErrorCategory::Bus),
            (139, AieErrorCategory::Lock),
        ];
        for &(event_id, expected) in cases {
            assert_eq!(
                event_to_category(event_id, AieErrorOrigin::MemTile),
                Some(expected),
                "memtile event {event_id} should map to {expected:?}"
            );
        }
    }

    #[test]
    fn shim_event_categories_match_driver_table() {
        // Direct port of aie2_error.c:134-146 shim table (11 entries).
        let cases: &[(u8, AieErrorCategory)] = &[
            (64, AieErrorCategory::Bus),
            (65, AieErrorCategory::Stream),
            (66, AieErrorCategory::Stream),
            (67, AieErrorCategory::Bus),
            (68, AieErrorCategory::Bus),
            (69, AieErrorCategory::Bus),
            (70, AieErrorCategory::Bus),
            (71, AieErrorCategory::Bus),
            (72, AieErrorCategory::Dma),
            (73, AieErrorCategory::Dma),
            (74, AieErrorCategory::Lock),
        ];
        for &(event_id, expected) in cases {
            assert_eq!(
                event_to_category(event_id, AieErrorOrigin::Pl),
                Some(expected),
                "shim event {event_id} should map to {expected:?}"
            );
        }
    }

    #[test]
    fn wrong_table_lookup_returns_none() {
        // Event 99 is a mem error, not core.
        assert_eq!(event_to_category(99, AieErrorOrigin::Core), None);
        // Event 56 is core, not memtile.
        assert_eq!(event_to_category(56, AieErrorOrigin::MemTile), None);
    }

    #[test]
    fn is_error_event_agrees_with_lookup() {
        assert!(is_error_event(69, AieErrorOrigin::Core));
        assert!(is_error_event(91, AieErrorOrigin::Mem));
        assert!(!is_error_event(0, AieErrorOrigin::Core));
        assert!(!is_error_event(99, AieErrorOrigin::Core)); // wrong table
    }
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test -p xdna-archspec --lib async_errors
```

Expected: compile errors (`event_to_category`, `is_error_event` not found).

- [ ] **Step 3: Implement the tables and lookups**

Append to `crates/xdna-archspec/src/aie2/async_errors.rs`, BEFORE the `#[cfg(test)]` block:

```rust
/// Single entry in a category lookup table.
pub struct EventCategory {
    pub event_id: u8,
    pub category: AieErrorCategory,
}

/// Core-module error events. Direct port of `aie2_error.c:105-120`, plus the
/// emu-specific entry for INSTR_ERROR=69 (mlir-aie names event 69 INSTR_ERROR;
/// driver table omits it). See spec section 2 INSTR_ERROR row for rationale.
pub const CORE_EVENT_CAT: &[EventCategory] = &[
    EventCategory { event_id: 55, category: AieErrorCategory::Access },
    EventCategory { event_id: 56, category: AieErrorCategory::Stream },
    EventCategory { event_id: 57, category: AieErrorCategory::Stream },
    EventCategory { event_id: 58, category: AieErrorCategory::Bus },
    EventCategory { event_id: 59, category: AieErrorCategory::Instruction },
    EventCategory { event_id: 60, category: AieErrorCategory::Access },
    EventCategory { event_id: 62, category: AieErrorCategory::Ecc },
    EventCategory { event_id: 64, category: AieErrorCategory::Ecc },
    EventCategory { event_id: 65, category: AieErrorCategory::Access },
    EventCategory { event_id: 66, category: AieErrorCategory::Access },
    EventCategory { event_id: 67, category: AieErrorCategory::Lock },
    EventCategory { event_id: 69, category: AieErrorCategory::Instruction }, // emu-specific
    EventCategory { event_id: 70, category: AieErrorCategory::Instruction },
    EventCategory { event_id: 71, category: AieErrorCategory::Stream },
    EventCategory { event_id: 72, category: AieErrorCategory::Bus },
];

/// Mem-module error events (mem module of compute tiles).
/// Direct port of `aie2_error.c:89-103`.
pub const MEM_EVENT_CAT: &[EventCategory] = &[
    EventCategory { event_id: 88, category: AieErrorCategory::Ecc },
    EventCategory { event_id: 90, category: AieErrorCategory::Ecc },
    EventCategory { event_id: 91, category: AieErrorCategory::MemParity },
    EventCategory { event_id: 92, category: AieErrorCategory::MemParity },
    EventCategory { event_id: 93, category: AieErrorCategory::MemParity },
    EventCategory { event_id: 94, category: AieErrorCategory::MemParity },
    EventCategory { event_id: 95, category: AieErrorCategory::MemParity },
    EventCategory { event_id: 96, category: AieErrorCategory::MemParity },
    EventCategory { event_id: 97, category: AieErrorCategory::Dma },
    EventCategory { event_id: 98, category: AieErrorCategory::Dma },
    EventCategory { event_id: 99, category: AieErrorCategory::Dma },
    EventCategory { event_id: 100, category: AieErrorCategory::Dma },
    EventCategory { event_id: 101, category: AieErrorCategory::Lock },
];

/// Memtile (row==1 mem) error events.
/// Direct port of `aie2_error.c:122-132`.
pub const MEMTILE_EVENT_CAT: &[EventCategory] = &[
    EventCategory { event_id: 130, category: AieErrorCategory::Ecc },
    EventCategory { event_id: 132, category: AieErrorCategory::Ecc },
    EventCategory { event_id: 133, category: AieErrorCategory::Dma },
    EventCategory { event_id: 134, category: AieErrorCategory::Dma },
    EventCategory { event_id: 135, category: AieErrorCategory::Stream },
    EventCategory { event_id: 136, category: AieErrorCategory::Stream },
    EventCategory { event_id: 137, category: AieErrorCategory::Stream },
    EventCategory { event_id: 138, category: AieErrorCategory::Bus },
    EventCategory { event_id: 139, category: AieErrorCategory::Lock },
];

/// Shim-PL error events. Direct port of `aie2_error.c:134-146` (11 entries).
pub const SHIM_EVENT_CAT: &[EventCategory] = &[
    EventCategory { event_id: 64, category: AieErrorCategory::Bus },
    EventCategory { event_id: 65, category: AieErrorCategory::Stream },
    EventCategory { event_id: 66, category: AieErrorCategory::Stream },
    EventCategory { event_id: 67, category: AieErrorCategory::Bus },
    EventCategory { event_id: 68, category: AieErrorCategory::Bus },
    EventCategory { event_id: 69, category: AieErrorCategory::Bus },
    EventCategory { event_id: 70, category: AieErrorCategory::Bus },
    EventCategory { event_id: 71, category: AieErrorCategory::Bus },
    EventCategory { event_id: 72, category: AieErrorCategory::Dma },
    EventCategory { event_id: 73, category: AieErrorCategory::Dma },
    EventCategory { event_id: 74, category: AieErrorCategory::Lock },
];

/// Look up the error category for a given `(event_id, origin)` pair.
/// Returns `None` if the event is not classified as an error in this origin's
/// table (matches the driver's `AIE_ERROR_UNKNOWN` fallback at
/// `aie2_error.c:211, 221`).
pub fn event_to_category(event_id: u8, origin: AieErrorOrigin) -> Option<AieErrorCategory> {
    let table = match origin {
        AieErrorOrigin::Core => CORE_EVENT_CAT,
        AieErrorOrigin::Mem => MEM_EVENT_CAT,
        AieErrorOrigin::MemTile => MEMTILE_EVENT_CAT,
        AieErrorOrigin::Pl => SHIM_EVENT_CAT,
    };
    table.iter().find(|e| e.event_id == event_id).map(|e| e.category)
}

/// Gate function: does this `(event_id, origin)` pair represent an error?
/// Tier B's effects.rs hook uses this to decide whether to call
/// `AsyncErrorSink::record_error`.
pub fn is_error_event(event_id: u8, origin: AieErrorOrigin) -> bool {
    event_to_category(event_id, origin).is_some()
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test -p xdna-archspec --lib async_errors
```

Expected: all tests (the 2 enum tests from Task 1 + 6 new tests) pass.

- [ ] **Step 5: Commit**

```bash
git add crates/xdna-archspec/src/aie2/async_errors.rs
git commit -m "$(cat <<'EOF'
interrupt: port driver async-error categorization tables to archspec

CORE/MEM/MEMTILE/SHIM event category tables, direct ports of
aie2_error.c:89-150, plus the emu-specific INSTR_ERROR=69 entry
(spec section 2 decisions). event_to_category dispatches by origin;
is_error_event is the gate the Tier B hook will use.

Generated using Claude Code.
EOF
)"
```

---

## Task 3: Encoding helpers + module/category lookups

Driver-macro ports for `err_code` / `ex_err_code` plus the two small lookup
tables (`aie_cat_err_num_map`, `aie_mod_amdxdna_err_mod_map`) needed to
fully populate an `amdxdna_async_error` record.

**Files:**
- Modify: `crates/xdna-archspec/src/aie2/async_errors.rs`

- [ ] **Step 1: Write failing encoding/lookup tests**

Append to the `#[cfg(test)] mod tests` block:

```rust
    #[test]
    fn build_err_code_round_trips_via_bit_unpack() {
        let code = build_err_code(
            AmdxdnaErrorNum::AieInstruction,
            AmdxdnaErrorDriver::Aie,
            Severity::Critical,
            AmdxdnaErrorModule::AieCore,
            Class::Aie,
        );
        // Per amdxdna_error.h:95-104 layout: num at [15:0] (16-bit mask),
        // driver at [19:16], severity at [27:24], module at [35:32], class
        // at [43:40] -- all 4-bit masks.
        let num = code & 0xFFFF;
        let driver = (code >> 16) & 0xF;
        let severity = (code >> 24) & 0xF;
        let module = (code >> 32) & 0xF;
        let class = (code >> 40) & 0xF;
        assert_eq!(num, AmdxdnaErrorNum::AieInstruction as u64);
        assert_eq!(driver, AmdxdnaErrorDriver::Aie as u64);
        assert_eq!(severity, Severity::Critical as u64);
        assert_eq!(module, AmdxdnaErrorModule::AieCore as u64);
        assert_eq!(class, Class::Aie as u64);
    }

    #[test]
    fn build_critical_aie_error_code_matches_driver_macro() {
        // AMDXDNA_CRITICAL_ERROR_CODE_BUILD hardcodes driver=Aie, severity=Critical,
        // class=Aie (amdxdna_error.h:113-115). This convenience helper is what
        // Tier B uses for AIE async errors.
        let code = build_critical_aie_error_code(
            AmdxdnaErrorNum::AieInstruction,
            AmdxdnaErrorModule::AieCore,
        );
        let expected = build_err_code(
            AmdxdnaErrorNum::AieInstruction,
            AmdxdnaErrorDriver::Aie,
            Severity::Critical,
            AmdxdnaErrorModule::AieCore,
            Class::Aie,
        );
        assert_eq!(code, expected);
    }

    #[test]
    fn build_ex_err_code_packs_row_and_col() {
        // Per amdxdna_error.h:139-141: col in [3:0] (mask 0xF), row in [11:8].
        assert_eq!(build_ex_err_code(2, 3), (2 << 8) | 3);
        assert_eq!(build_ex_err_code(0, 0), 0);
        assert_eq!(build_ex_err_code(0xF, 0xF), (0xF << 8) | 0xF);
    }

    #[test]
    fn category_to_error_num_maps_known_categories() {
        assert_eq!(category_to_error_num(AieErrorCategory::Saturation), AmdxdnaErrorNum::AieSaturation);
        assert_eq!(category_to_error_num(AieErrorCategory::Instruction), AmdxdnaErrorNum::AieInstruction);
        assert_eq!(category_to_error_num(AieErrorCategory::Dma), AmdxdnaErrorNum::AieDma);
        assert_eq!(category_to_error_num(AieErrorCategory::Unknown), AmdxdnaErrorNum::Unknown);
    }

    #[test]
    fn mod_to_amdxdna_module_maps_known_origins() {
        assert_eq!(mod_type_to_amdxdna_module(AieErrorOrigin::Core), AmdxdnaErrorModule::AieCore);
        assert_eq!(mod_type_to_amdxdna_module(AieErrorOrigin::Mem), AmdxdnaErrorModule::AieMemory);
        // Memtile reuses the AIE memory module ID (driver behavior:
        // memtile categorization happens via row==1 at the receiver, but
        // the wire-format module field is AIE_MEMORY).
        assert_eq!(mod_type_to_amdxdna_module(AieErrorOrigin::MemTile), AmdxdnaErrorModule::AieMemory);
        assert_eq!(mod_type_to_amdxdna_module(AieErrorOrigin::Pl), AmdxdnaErrorModule::AiePl);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test -p xdna-archspec --lib async_errors
```

Expected: compile errors for `build_err_code`, `build_ex_err_code`, `category_to_error_num`, `mod_type_to_amdxdna_module`.

- [ ] **Step 3: Implement encoders and lookups**

Append to `crates/xdna-archspec/src/aie2/async_errors.rs`, BEFORE the `#[cfg(test)]` block:

```rust
// Bit layout of amdxdna_async_error.err_code (amdxdna_error.h:95-104).
// 5 fields: num (16-bit) at [15:0]; driver, severity, module, class each
// 4-bit at [19:16], [27:24], [35:32], [43:40].
const NUM_SHIFT: u32 = 0;
const DRIVER_SHIFT: u32 = 16;
const SEVERITY_SHIFT: u32 = 24;
const MODULE_SHIFT: u32 = 32;
const CLASS_SHIFT: u32 = 40;
const NUM_MASK: u64 = 0xFFFF;
const DRIVER_MASK: u64 = 0xF;
const SEVERITY_MASK: u64 = 0xF;
const MODULE_MASK: u64 = 0xF;
const CLASS_MASK: u64 = 0xF;

/// Build the `err_code` field of `amdxdna_async_error`.
/// Mirrors `AMDXDNA_ERROR_CODE_BUILD` (`amdxdna_error.h:106-111`). Caller
/// supplies all 5 fields; for the common AIE async-error path use
/// `build_critical_aie_error_code` instead.
pub const fn build_err_code(
    num: AmdxdnaErrorNum,
    driver: AmdxdnaErrorDriver,
    severity: Severity,
    module: AmdxdnaErrorModule,
    class: Class,
) -> u64 {
    ((num as u64 & NUM_MASK) << NUM_SHIFT)
        | ((driver as u64 & DRIVER_MASK) << DRIVER_SHIFT)
        | ((severity as u64 & SEVERITY_MASK) << SEVERITY_SHIFT)
        | ((module as u64 & MODULE_MASK) << MODULE_SHIFT)
        | ((class as u64 & CLASS_MASK) << CLASS_SHIFT)
}

/// Convenience that hardcodes `driver=Aie`, `severity=Critical`, `class=Aie`.
/// Mirrors the driver's `AMDXDNA_CRITICAL_ERROR_CODE_BUILD`
/// (`amdxdna_error.h:113-115`), which is what `aie2_error.c:239` uses to
/// build the `err_code` for every AIE async error.
pub const fn build_critical_aie_error_code(
    num: AmdxdnaErrorNum,
    module: AmdxdnaErrorModule,
) -> u64 {
    build_err_code(
        num,
        AmdxdnaErrorDriver::Aie,
        Severity::Critical,
        module,
        Class::Aie,
    )
}

// Bit layout of amdxdna_async_error.ex_err_code (amdxdna_error.h:127-141).
// col in [3:0] (mask 0xF, shift 0); row in [11:8] (mask 0xF, shift 8).
const EXTRA_COL_SHIFT: u32 = 0;
const EXTRA_ROW_SHIFT: u32 = 8;
const EXTRA_COL_MASK: u64 = 0xF;
const EXTRA_ROW_MASK: u64 = 0xF;

/// Build the `ex_err_code` field of `amdxdna_async_error`.
/// Mirrors `AMDXDNA_ERROR_EXTRA_CODE_BUILD` (amdxdna_error.h:139-141).
pub const fn build_ex_err_code(row: u8, col: u8) -> u64 {
    (((col as u64) & EXTRA_COL_MASK) << EXTRA_COL_SHIFT)
        | (((row as u64) & EXTRA_ROW_MASK) << EXTRA_ROW_SHIFT)
}

/// Map error category to the driver's `amdxdna_error_num`.
/// Mirrors `aie_cat_err_num_map` lookup (`aie2_error.c`). Unknown is the
/// fallback (driver default at `aie2_error.c:182`).
pub fn category_to_error_num(cat: AieErrorCategory) -> AmdxdnaErrorNum {
    match cat {
        AieErrorCategory::Saturation => AmdxdnaErrorNum::AieSaturation,
        AieErrorCategory::Fp => AmdxdnaErrorNum::AieFp,
        AieErrorCategory::Stream => AmdxdnaErrorNum::AieStream,
        AieErrorCategory::Access => AmdxdnaErrorNum::AieAccess,
        AieErrorCategory::Bus => AmdxdnaErrorNum::AieBus,
        AieErrorCategory::Instruction => AmdxdnaErrorNum::AieInstruction,
        AieErrorCategory::Ecc => AmdxdnaErrorNum::AieEcc,
        AieErrorCategory::Lock => AmdxdnaErrorNum::AieLock,
        AieErrorCategory::Dma => AmdxdnaErrorNum::AieDma,
        AieErrorCategory::MemParity => AmdxdnaErrorNum::AieMemParity,
        AieErrorCategory::Unknown => AmdxdnaErrorNum::Unknown,
    }
}

/// Map internal origin to driver's `amdxdna_error_module`.
/// Mirrors `aie_mod_amdxdna_err_mod_map` (`aie2_error.c:161-165`). Memtile
/// reuses the AIE_MEMORY module ID; receiver disambiguates by row.
pub fn mod_type_to_amdxdna_module(origin: AieErrorOrigin) -> AmdxdnaErrorModule {
    match origin {
        AieErrorOrigin::Core => AmdxdnaErrorModule::AieCore,
        AieErrorOrigin::Mem | AieErrorOrigin::MemTile => AmdxdnaErrorModule::AieMemory,
        AieErrorOrigin::Pl => AmdxdnaErrorModule::AiePl,
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test -p xdna-archspec --lib async_errors
```

Expected: all archspec async_errors tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/xdna-archspec/src/aie2/async_errors.rs
git commit -m "$(cat <<'EOF'
interrupt: add async-error encoding helpers and lookup maps

build_err_code / build_ex_err_code port the driver's
AMDXDNA_ERROR_CODE_BUILD / AMDXDNA_ERROR_EXTRA_CODE_BUILD macros.
category_to_error_num and mod_type_to_amdxdna_module mirror the
driver's aie_cat_err_num_map and aie_mod_amdxdna_err_mod_map lookups.

Generated using Claude Code.
EOF
)"
```

---

## Task 4: Wire-format types

`AieError`, `AieErrInfoHeader`, `AmdxdnaAsyncError`, `AsyncRing` — the byte
layouts that match the driver's structs. Size assertions are compile-time;
round-trip is runtime.

**Files:**
- Create: `src/device/async_errors/mod.rs`
- Create: `src/device/async_errors/types.rs`
- Modify: `src/device/mod.rs`

- [ ] **Step 1: Create the module stubs and wire them in**

Create `src/device/async_errors/mod.rs`:

```rust
//! Tier B async-error subsystem.
//!
//! Sits parallel to `device::interrupts` (Tier A). When an error-category
//! event is generated, `state::effects::apply_tile_local_effects` calls
//! `AsyncErrorSink::record_error` here in addition to the existing
//! Tier A L1/L2 latch path.
//!
//! Three output surfaces: per-handle cache (consumed by plugin ioctl),
//! per-column 8KB mailbox rings in driver-wire format (reserved for future
//! kernel-driver attachment), optional push callback (FFI-registered).
//!
//! Design source:
//! `docs/superpowers/specs/2026-05-19-interrupt-tier-b-firmware-mailbox-design.md`

pub mod types;
```

Open `src/device/mod.rs` and add (next to other `pub mod` lines, alphabetical
or at end of the block — match the local convention):

```rust
pub mod async_errors;
```

- [ ] **Step 2: Write failing tests for the wire-format types**

Create `src/device/async_errors/types.rs`:

```rust
//! Wire-format types for async-error delivery.
//!
//! Direct mirrors of driver structs; bytes are what firmware would DMA into
//! the host async-event buffer. Compile-time size assertions pin the layout;
//! runtime tests verify offsets via byte round-trip.
//!
//! References:
//! - `aie_error`        -> xdna-driver/src/driver/amdxdna/aie2_error.c:56-64
//! - `aie_err_info`     -> xdna-driver/src/driver/amdxdna/aie2_error.c:66-71
//! - `amdxdna_async_error` -> xdna-driver/include/uapi/drm/amdxdna_accel.h:610-617

/// Mirrors driver `struct aie_error` (12 bytes). NOT packed -- driver
/// comment "Don't pack, unless XAIE side changed" (aie2_error.c:55).
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct AieError {
    pub row: u8,
    pub col: u8,
    pub reserved_0: u16,
    pub mod_type: u32,
    pub event_id: u8,
    pub reserved_1: u8,
    pub reserved_2: u16,
}

const _: () = assert!(std::mem::size_of::<AieError>() == 12);

/// Header preceding `err_cnt` `AieError` entries in the ring.
/// Mirrors `struct aie_err_info` (aie2_error.c:66-71).
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct AieErrInfoHeader {
    pub err_cnt: u32,
    pub ret_code: u32,
    pub rsvd: u32,
}

const _: () = assert!(std::mem::size_of::<AieErrInfoHeader>() == 12);

/// uapi async-error record returned by `DRM_AMDXDNA_HW_LAST_ASYNC_ERR`.
/// Mirrors `struct amdxdna_async_error` (amdxdna_accel.h:610-617).
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct AmdxdnaAsyncError {
    pub err_code: u64,
    pub ts_us: u64,
    pub ex_err_code: u64,
}

const _: () = assert!(std::mem::size_of::<AmdxdnaAsyncError>() == 24);

/// 8 KB per driver `ASYNC_BUF_SIZE` (aie2_msg_priv.h:406, SZ_8K).
pub const ASYNC_BUF_SIZE: usize = 8 * 1024;

/// Max errors that fit after the header.
pub const MAX_ERRORS_PER_RING: usize =
    (ASYNC_BUF_SIZE - std::mem::size_of::<AieErrInfoHeader>())
        / std::mem::size_of::<AieError>();

/// Emu-defined `ret_code` value set when a push overflows. Driver treats
/// any nonzero `ret_code` as an error (aie2_error.c worker handling).
pub const RET_CODE_OVERFLOW: u32 = 1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_errors_per_ring_is_681() {
        // (8192 - 12) / 12 = 681 (integer division).
        assert_eq!(MAX_ERRORS_PER_RING, 681);
    }

    #[test]
    fn aie_error_field_offsets_via_byte_inspection() {
        // Construct a record with distinguishable values in each field,
        // transmute to bytes, and verify each field's position.
        let e = AieError {
            row: 0xA1,
            col: 0xB2,
            reserved_0: 0,
            mod_type: 0xDEAD_BEEF,
            event_id: 0xC3,
            reserved_1: 0,
            reserved_2: 0,
        };
        let bytes: [u8; 12] = unsafe { std::mem::transmute(e) };
        assert_eq!(bytes[0], 0xA1, "row at offset 0");
        assert_eq!(bytes[1], 0xB2, "col at offset 1");
        // bytes[2..4] is reserved_0
        // bytes[4..8] is mod_type, little-endian on x86_64
        assert_eq!(&bytes[4..8], &0xDEAD_BEEF_u32.to_le_bytes());
        assert_eq!(bytes[8], 0xC3, "event_id at offset 8");
    }

    #[test]
    fn amdxdna_async_error_field_offsets() {
        let r = AmdxdnaAsyncError {
            err_code: 0x1111_2222_3333_4444,
            ts_us: 0x5555_6666_7777_8888,
            ex_err_code: 0x9999_AAAA_BBBB_CCCC,
        };
        let bytes: [u8; 24] = unsafe { std::mem::transmute(r) };
        assert_eq!(&bytes[0..8], &r.err_code.to_le_bytes());
        assert_eq!(&bytes[8..16], &r.ts_us.to_le_bytes());
        assert_eq!(&bytes[16..24], &r.ex_err_code.to_le_bytes());
    }
}
```

- [ ] **Step 3: Run tests to verify they pass**

```bash
cargo build -p xdna-emu-core
TMPDIR=/tmp/claude-1000 cargo test --lib async_errors::types
```

Expected: 3 new tests pass; compile-time assertions verified at build.

- [ ] **Step 4: Commit**

```bash
git add src/device/async_errors/mod.rs src/device/async_errors/types.rs src/device/mod.rs
git commit -m "$(cat <<'EOF'
interrupt: add Tier B wire-format types (AieError, header, async record)

AieError (12B), AieErrInfoHeader (12B), AmdxdnaAsyncError (24B) with
compile-time size assertions and byte-layout round-trip tests.
ASYNC_BUF_SIZE = 8K, MAX_ERRORS_PER_RING = 681. Sink and ring land in
the next tasks.

Generated using Claude Code.
EOF
)"
```

---

## Task 5: AsyncRing

Per-column 8 KB byte buffer with header overlay and `push` / `read_into` /
`clear` operations.

**Files:**
- Modify: `src/device/async_errors/types.rs`

- [ ] **Step 1: Write failing tests for AsyncRing**

Append to `src/device/async_errors/types.rs` (inside the existing `#[cfg(test)] mod tests`):

```rust
    #[test]
    fn ring_starts_empty() {
        let ring = AsyncRing::new();
        assert_eq!(ring.header().err_cnt, 0);
        assert_eq!(ring.header().ret_code, 0);
    }

    #[test]
    fn push_increments_err_cnt_and_stores_record() {
        let mut ring = AsyncRing::new();
        let e = AieError { row: 2, col: 1, event_id: 69, mod_type: 1, ..Default::default() };
        ring.push(e).expect("first push must succeed");
        assert_eq!(ring.header().err_cnt, 1);
        assert_eq!(ring.records()[0], e);
        let e2 = AieError { row: 3, col: 2, event_id: 70, mod_type: 1, ..Default::default() };
        ring.push(e2).expect("second push must succeed");
        assert_eq!(ring.header().err_cnt, 2);
        assert_eq!(ring.records()[0], e);
        assert_eq!(ring.records()[1], e2);
    }

    #[test]
    fn push_at_capacity_returns_overflow() {
        let mut ring = AsyncRing::new();
        // Fill to capacity.
        for i in 0..MAX_ERRORS_PER_RING as u32 {
            ring.push(AieError {
                row: (i & 0xFF) as u8,
                col: ((i >> 8) & 0xFF) as u8,
                event_id: 69,
                mod_type: 1,
                ..Default::default()
            })
            .expect("push within capacity must succeed");
        }
        assert_eq!(ring.header().err_cnt as usize, MAX_ERRORS_PER_RING);
        // One more must overflow.
        let next = AieError { row: 0xFF, col: 0xFF, event_id: 69, mod_type: 1, ..Default::default() };
        assert!(matches!(ring.push(next), Err(Overflow)));
        // err_cnt unchanged.
        assert_eq!(ring.header().err_cnt as usize, MAX_ERRORS_PER_RING);
    }

    #[test]
    fn read_into_copies_header_then_records() {
        let mut ring = AsyncRing::new();
        ring.push(AieError { row: 5, col: 1, event_id: 69, mod_type: 1, ..Default::default() }).unwrap();
        let mut dst = vec![0u8; 64];
        let n = ring.read_into(&mut dst);
        // 12 byte header + 1 * 12 byte record = 24 bytes.
        assert_eq!(n, 24);
        // Header err_cnt = 1 (little-endian u32 at offset 0).
        assert_eq!(&dst[0..4], &1u32.to_le_bytes());
        // First record's row at offset 12.
        assert_eq!(dst[12], 5);
        assert_eq!(dst[13], 1);
    }

    #[test]
    fn read_into_returns_zero_on_empty_ring() {
        let ring = AsyncRing::new();
        let mut dst = vec![0u8; 64];
        // Header is always present; an empty ring still copies the 12-byte header.
        let n = ring.read_into(&mut dst);
        assert_eq!(n, 12, "empty ring read returns just the header");
        assert_eq!(&dst[0..4], &0u32.to_le_bytes(), "err_cnt == 0");
    }

    #[test]
    fn read_into_truncates_to_dst_size() {
        let mut ring = AsyncRing::new();
        ring.push(AieError::default()).unwrap();
        ring.push(AieError::default()).unwrap();
        // dst smaller than the 36 bytes the ring contains: should copy only dst.len().
        let mut dst = vec![0u8; 20];
        assert_eq!(ring.read_into(&mut dst), 20);
    }

    #[test]
    fn clear_zeros_header_and_records() {
        let mut ring = AsyncRing::new();
        ring.push(AieError { row: 5, col: 1, event_id: 69, mod_type: 1, ..Default::default() }).unwrap();
        ring.set_ret_code(RET_CODE_OVERFLOW);
        ring.clear();
        assert_eq!(ring.header().err_cnt, 0);
        assert_eq!(ring.header().ret_code, 0);
        let mut dst = vec![0u8; ASYNC_BUF_SIZE];
        ring.read_into(&mut dst);
        // Only header (12B) is "used" -- all zero.
        assert!(dst[..12].iter().all(|&b| b == 0));
    }
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib async_errors::types
```

Expected: compile errors for `AsyncRing`, `Overflow`.

- [ ] **Step 3: Implement AsyncRing**

Append to `src/device/async_errors/types.rs`, BEFORE the `#[cfg(test)]` block:

```rust
/// Error type returned by `AsyncRing::push` when the ring is at capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Overflow;

/// Per-column 8 KB ring in driver-wire format.
///
/// Layout: 12-byte `AieErrInfoHeader` at offset 0, followed by `err_cnt`
/// `AieError` records (12 bytes each) starting at offset 12. Byte-compatible
/// with what firmware would DMA into the driver's `dma_hdl` async-event
/// buffer.
pub struct AsyncRing {
    bytes: Box<[u8; ASYNC_BUF_SIZE]>,
}

impl AsyncRing {
    pub fn new() -> Self {
        Self { bytes: Box::new([0u8; ASYNC_BUF_SIZE]) }
    }

    pub fn header(&self) -> &AieErrInfoHeader {
        // SAFETY: bytes is a 8192-byte aligned heap allocation; AieErrInfoHeader
        // is 12 bytes of POD with no padding-sensitive fields. The cast aliases
        // the same bytes as a typed view.
        unsafe { &*(self.bytes.as_ptr() as *const AieErrInfoHeader) }
    }

    fn header_mut(&mut self) -> &mut AieErrInfoHeader {
        // SAFETY: as above; &mut self gives exclusive access.
        unsafe { &mut *(self.bytes.as_mut_ptr() as *mut AieErrInfoHeader) }
    }

    /// Append a record. Returns `Err(Overflow)` if the ring is at capacity;
    /// in that case the ring state is unchanged.
    pub fn push(&mut self, e: AieError) -> Result<(), Overflow> {
        let cnt = self.header().err_cnt as usize;
        if cnt >= MAX_ERRORS_PER_RING {
            return Err(Overflow);
        }
        let header_size = std::mem::size_of::<AieErrInfoHeader>();
        let rec_size = std::mem::size_of::<AieError>();
        let offset = header_size + cnt * rec_size;
        // SAFETY: bounds-checked: offset + rec_size <= ASYNC_BUF_SIZE iff
        // cnt < MAX_ERRORS_PER_RING (the gate above).
        unsafe {
            let dst = self.bytes.as_mut_ptr().add(offset) as *mut AieError;
            std::ptr::write_unaligned(dst, e);
        }
        self.header_mut().err_cnt += 1;
        Ok(())
    }

    /// Read-only slice view of stored records.
    pub fn records(&self) -> &[AieError] {
        let cnt = self.header().err_cnt as usize;
        let header_size = std::mem::size_of::<AieErrInfoHeader>();
        // SAFETY: bytes is contiguous; cnt is bounded by MAX_ERRORS_PER_RING
        // (push enforces). AieError is repr(C) POD.
        unsafe {
            let p = self.bytes.as_ptr().add(header_size) as *const AieError;
            std::slice::from_raw_parts(p, cnt)
        }
    }

    /// Copy header + valid records into `dst`. Returns the number of bytes
    /// copied (at most `dst.len()` and at most `12 + err_cnt * 12`).
    pub fn read_into(&self, dst: &mut [u8]) -> usize {
        let used = std::mem::size_of::<AieErrInfoHeader>()
            + (self.header().err_cnt as usize) * std::mem::size_of::<AieError>();
        let n = used.min(dst.len());
        dst[..n].copy_from_slice(&self.bytes[..n]);
        n
    }

    /// Set the ret_code field (used to signal Overflow to consumers).
    pub fn set_ret_code(&mut self, code: u32) {
        self.header_mut().ret_code = code;
    }

    /// Zero the entire buffer (header + payload area).
    pub fn clear(&mut self) {
        self.bytes.fill(0);
    }
}

impl Default for AsyncRing {
    fn default() -> Self {
        Self::new()
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib async_errors::types
```

Expected: all 9 types tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/device/async_errors/types.rs
git commit -m "$(cat <<'EOF'
interrupt: add AsyncRing for per-column wire-format error storage

8KB ring with 12-byte AieErrInfoHeader followed by AieError records.
push/read_into/clear/records/set_ret_code operations. Overflow at
capacity (681 records) returns Err(Overflow) without mutating state.

Generated using Claude Code.
EOF
)"
```

---

## Task 6: AsyncErrorSink

The single-owner subsystem that ties cache + rings + drain queue + push
trigger together. `record_error` is the only mutation entry point.

**Files:**
- Create: `src/device/async_errors/sink.rs`
- Modify: `src/device/async_errors/mod.rs`

- [ ] **Step 1: Write failing tests for AsyncErrorSink**

Create `src/device/async_errors/sink.rs`:

```rust
//! Tier B sink: cache + per-column rings + drain queue.
//!
//! Single owner of all async-error state. `record_error` is the only
//! mutation entry point; reads (cache, ring bytes, drain) are read-only
//! aside from `drain_newly_recorded` (which empties the queue).

use std::collections::VecDeque;

use xdna_archspec::aie2::async_errors::{self, AieErrorOrigin};

use super::types::{AieError, AmdxdnaAsyncError, AsyncRing, RET_CODE_OVERFLOW};

/// Async-error subsystem state. Lives on `DeviceState`.
pub struct AsyncErrorSink {
    /// Last-error cache. `None` until any error fires; mirrors the driver's
    /// `amdxdna_async_err_cache` last-write-wins behavior.
    cache: Option<AmdxdnaAsyncError>,
    /// Per-column ring buffers. Indexed by physical column. Sized to match
    /// the driver's per-column `async_event` slot allocation (5 cols on NPU1,
    /// but we size dynamically to support any width).
    rings: Vec<AsyncRing>,
    /// Records added since last drain. Drained by the FFI layer between
    /// engine steps to fire the registered push callback.
    newly_recorded: VecDeque<AmdxdnaAsyncError>,
}

impl AsyncErrorSink {
    /// Create a sink with `num_cols` independent ring buffers.
    pub fn new(num_cols: usize) -> Self {
        Self {
            cache: None,
            rings: (0..num_cols).map(|_| AsyncRing::new()).collect(),
            newly_recorded: VecDeque::new(),
        }
    }

    /// Record an error. Updates the cache (last-write-wins), appends to
    /// the column's ring (or sets RET_CODE_OVERFLOW if full), and queues
    /// the cache record for FFI drain.
    pub fn record_error(
        &mut self,
        col: u8,
        row: u8,
        origin: AieErrorOrigin,
        event_id: u8,
        cycle: u64,
    ) {
        // 1. Append to the column's ring.
        let col_idx = col as usize;
        if let Some(ring) = self.rings.get_mut(col_idx) {
            let record = AieError {
                row,
                col,
                reserved_0: 0,
                mod_type: origin.wire_mod_type(),
                event_id,
                reserved_1: 0,
                reserved_2: 0,
            };
            if ring.push(record).is_err() {
                ring.set_ret_code(RET_CODE_OVERFLOW);
            }
        }

        // 2. Categorize and update cache. `is_error_event` gates the call
        // site, so `event_to_category` returning None here would be a bug;
        // we expect Some and unwrap with a clear message.
        let category = async_errors::event_to_category(event_id, origin)
            .expect("record_error: event_to_category returned None; caller must gate via is_error_event");
        let num = async_errors::category_to_error_num(category);
        let module = async_errors::mod_type_to_amdxdna_module(origin);
        // Convenience helper hardcodes driver=Aie, severity=Critical, class=Aie,
        // matching driver's AMDXDNA_CRITICAL_ERROR_CODE_BUILD (aie2_error.c:239).
        let err_code = async_errors::build_critical_aie_error_code(num, module);
        let ex_err_code = async_errors::build_ex_err_code(row, col);
        // Cycle-as-nanoseconds at ~1 GHz silicon -> microseconds = cycle / 1000.
        // Deterministic; spec section 2 ts_us decision.
        let ts_us = cycle / 1000;
        let record = AmdxdnaAsyncError { err_code, ts_us, ex_err_code };
        self.cache = Some(record);
        self.newly_recorded.push_back(record);
    }

    /// Read the last-error cache. `None` until any error fires.
    pub fn last_cache(&self) -> Option<&AmdxdnaAsyncError> {
        self.cache.as_ref()
    }

    /// Read-only view of column `col`'s ring. `None` if `col` is out of range.
    pub fn ring(&self, col: u8) -> Option<&AsyncRing> {
        self.rings.get(col as usize)
    }

    /// Drain and return records queued since the last drain. FIFO order.
    /// FFI layer calls this between engine steps to fire push callbacks.
    pub fn drain_newly_recorded(&mut self) -> Vec<AmdxdnaAsyncError> {
        self.newly_recorded.drain(..).collect()
    }

    /// Zero the cache, all rings, and the drain queue. Called from
    /// `xdna_emu_reset_context` and `xdna_emu_clear_async_errors`.
    pub fn clear(&mut self) {
        self.cache = None;
        for r in self.rings.iter_mut() {
            r.clear();
        }
        self.newly_recorded.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_sink_has_empty_cache_and_rings() {
        let sink = AsyncErrorSink::new(5);
        assert!(sink.last_cache().is_none());
        for c in 0u8..5 {
            assert_eq!(sink.ring(c).unwrap().header().err_cnt, 0);
        }
    }

    #[test]
    fn record_error_populates_cache_and_ring() {
        use xdna_archspec::aie2::async_errors::{AmdxdnaErrorModule, AmdxdnaErrorNum};
        let mut sink = AsyncErrorSink::new(5);
        sink.record_error(1, 2, AieErrorOrigin::Core, 69, 50_000);
        let cache = sink.last_cache().expect("cache must be populated");
        assert_eq!(
            cache.err_code,
            async_errors::build_critical_aie_error_code(
                AmdxdnaErrorNum::AieInstruction,
                AmdxdnaErrorModule::AieCore,
            )
        );
        assert_eq!(cache.ts_us, 50); // 50_000 cycles / 1000
        assert_eq!(cache.ex_err_code, (2u64 << 8) | 1u64);
        let ring = sink.ring(1).unwrap();
        assert_eq!(ring.header().err_cnt, 1);
        let rec = &ring.records()[0];
        assert_eq!(rec.event_id, 69);
        assert_eq!(rec.row, 2);
        assert_eq!(rec.col, 1);
        assert_eq!(rec.mod_type, AieErrorOrigin::Core.wire_mod_type());
    }

    #[test]
    fn second_record_overwrites_cache_appends_ring() {
        let mut sink = AsyncErrorSink::new(5);
        sink.record_error(1, 2, AieErrorOrigin::Core, 69, 1_000);
        sink.record_error(1, 3, AieErrorOrigin::Core, 70, 2_000);
        let cache = sink.last_cache().unwrap();
        assert_eq!(cache.ts_us, 2);
        assert_eq!(cache.ex_err_code, (3u64 << 8) | 1u64);
        assert_eq!(sink.ring(1).unwrap().header().err_cnt, 2);
    }

    #[test]
    fn per_column_rings_independent() {
        let mut sink = AsyncErrorSink::new(5);
        sink.record_error(1, 2, AieErrorOrigin::Core, 69, 1_000);
        sink.record_error(3, 2, AieErrorOrigin::Core, 69, 2_000);
        assert_eq!(sink.ring(1).unwrap().header().err_cnt, 1);
        assert_eq!(sink.ring(3).unwrap().header().err_cnt, 1);
        // Column 2 untouched.
        assert_eq!(sink.ring(2).unwrap().header().err_cnt, 0);
    }

    #[test]
    fn out_of_range_col_is_silent_noop_on_ring_but_still_updates_cache() {
        let mut sink = AsyncErrorSink::new(5);
        sink.record_error(99, 2, AieErrorOrigin::Core, 69, 1_000);
        // Cache populated even though no ring matched (categorization still ran).
        assert!(sink.last_cache().is_some());
        // No rings updated.
        for c in 0u8..5 {
            assert_eq!(sink.ring(c).unwrap().header().err_cnt, 0);
        }
    }

    #[test]
    fn overflow_at_capacity_sets_ret_code() {
        use super::super::types::MAX_ERRORS_PER_RING;
        let mut sink = AsyncErrorSink::new(5);
        for _ in 0..MAX_ERRORS_PER_RING {
            sink.record_error(0, 2, AieErrorOrigin::Core, 69, 1_000);
        }
        assert_eq!(sink.ring(0).unwrap().header().err_cnt as usize, MAX_ERRORS_PER_RING);
        // One more triggers overflow.
        sink.record_error(0, 2, AieErrorOrigin::Core, 69, 1_000);
        assert_eq!(sink.ring(0).unwrap().header().ret_code, RET_CODE_OVERFLOW);
        // err_cnt unchanged.
        assert_eq!(sink.ring(0).unwrap().header().err_cnt as usize, MAX_ERRORS_PER_RING);
    }

    #[test]
    fn drain_returns_records_in_fifo_order_and_empties_queue() {
        let mut sink = AsyncErrorSink::new(5);
        sink.record_error(1, 2, AieErrorOrigin::Core, 69, 1_000);
        sink.record_error(2, 3, AieErrorOrigin::Core, 70, 2_000);
        let drained = sink.drain_newly_recorded();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].ts_us, 1);
        assert_eq!(drained[1].ts_us, 2);
        // Second drain returns empty.
        assert!(sink.drain_newly_recorded().is_empty());
    }

    #[test]
    fn clear_resets_cache_rings_and_drain_queue() {
        let mut sink = AsyncErrorSink::new(5);
        sink.record_error(0, 2, AieErrorOrigin::Core, 69, 1_000);
        sink.clear();
        assert!(sink.last_cache().is_none());
        assert_eq!(sink.ring(0).unwrap().header().err_cnt, 0);
        assert!(sink.drain_newly_recorded().is_empty());
    }
}
```

Open `src/device/async_errors/mod.rs` and add the `sink` module + re-exports:

```rust
pub mod sink;
pub mod types;

pub use sink::AsyncErrorSink;
pub use types::{AieError, AieErrInfoHeader, AmdxdnaAsyncError, AsyncRing, ASYNC_BUF_SIZE, MAX_ERRORS_PER_RING, RET_CODE_OVERFLOW};
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib async_errors::sink
```

Expected: 7 sink tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/device/async_errors/sink.rs src/device/async_errors/mod.rs
git commit -m "$(cat <<'EOF'
interrupt: add AsyncErrorSink (cache + per-col rings + drain queue)

Single mutation entry point: record_error. Cache last-write-wins,
ring appends, drain queue for FFI push-callback delivery. Categorization
via archspec helpers. Out-of-range col silently skips the ring update
but still populates the cache. Overflow sets RET_CODE_OVERFLOW.

Generated using Claude Code.
EOF
)"
```

---

## Task 7: Integrate AsyncErrorSink into DeviceState

Adds the field, initializes in `new`, clears in any context-reset path.

**Files:**
- Modify: `src/device/state/mod.rs`
- Read for reference: `src/interpreter/engine/coordinator.rs:1538` (`reset_for_new_context`) to confirm whether array-reset already covers per-tile state vs whether DeviceState fields need explicit clearing

- [ ] **Step 1: Write a failing integration test**

Append to `src/device/state/mod.rs` (or to its existing `#[cfg(test)] mod tests` block; create one if absent):

```rust
#[cfg(test)]
mod async_errors_integration_tests {
    use super::*;
    use xdna_archspec::aie2::async_errors::AieErrorOrigin;

    #[test]
    fn device_state_exposes_async_error_sink() {
        let dev = DeviceState::new_npu1();
        assert!(dev.async_errors.last_cache().is_none());
    }

    #[test]
    fn record_error_through_sink_reaches_cache() {
        let mut dev = DeviceState::new_npu1();
        dev.async_errors.record_error(1, 2, AieErrorOrigin::Core, 69, 10_000);
        assert!(dev.async_errors.last_cache().is_some());
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib async_errors_integration_tests
```

Expected: compile error (`async_errors` field doesn't exist on `DeviceState`).

- [ ] **Step 3: Add the field and constructor init**

Open `src/device/state/mod.rs`. In the `pub struct DeviceState { ... }` block (around line 68), add the field next to `pub array`:

```rust
    /// Tier B async-error subsystem: cache + per-column rings + drain queue.
    /// Populated from `state::effects::apply_tile_local_effects` when an
    /// error-category event is generated.
    pub async_errors: crate::device::async_errors::AsyncErrorSink,
```

In the `pub fn new(arch: Arc<dyn ArchConfig>) -> Self` constructor (around line 97), initialize the field. Use `array.cols()` to size the rings:

```rust
    pub fn new(arch: Arc<dyn ArchConfig>) -> Self {
        let array = TileArray::new(arch);
        let num_cols = array.cols() as usize;
        Self {
            array,
            stats: CdoStats::default(),
            pending_core_enables: Vec::new(),
            start_col: 0,
            async_errors: crate::device::async_errors::AsyncErrorSink::new(num_cols),
        }
    }
```

- [ ] **Step 4: Wire async_errors clearing into the context-reset path**

Open `src/interpreter/engine/coordinator.rs` and locate `reset_for_new_context` (around line 1538). At the start of the function body, before the existing per-tile reset loop, add:

```rust
        self.device.async_errors.clear();
```

(Confirm the engine field name is `device` — adjust to actual name if different.)

- [ ] **Step 5: Run tests to verify they pass**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib async_errors
TMPDIR=/tmp/claude-1000 cargo test --lib reset_for_new_context
```

Expected: all async_errors tests pass; reset tests still green.

- [ ] **Step 6: Commit**

```bash
git add src/device/state/mod.rs src/interpreter/engine/coordinator.rs
git commit -m "$(cat <<'EOF'
interrupt: integrate AsyncErrorSink into DeviceState

Field on DeviceState sized to array.cols() at construction.
reset_for_new_context clears the sink alongside other per-context state
so a fresh hw_context starts with an empty error cache + rings.

Generated using Claude Code.
EOF
)"
```

---

## Task 8: Tier B hook in effects.rs + end-to-end integration test

Restructures the `is_event_generate` match to return `Option<AieErrorOrigin>`,
adds the Tier B `record_error` call, and verifies via a control-packet-driven
test that the full chain works for INSTR_ERROR.

**Files:**
- Modify: `src/device/state/effects.rs`

- [ ] **Step 1: Write the failing end-to-end test**

In `src/device/state/effects.rs`, append to the existing `#[cfg(test)] mod interrupt_path_tests` block (around line 627). It already has `fire_event_generate_for_test` defined on `DeviceState` — reuse it:

```rust
    #[test]
    fn event_generate_for_instr_error_populates_async_cache_and_ring() {
        use xdna_archspec::aie2::async_errors::{
            self, AieErrorOrigin, AmdxdnaErrorModule, AmdxdnaErrorNum,
        };
        let mut dev = DeviceState::new_npu1();
        // Drive simulated time so ts_us is nonzero -- proves the cycle-as-ts
        // conversion is wired (not a literal 0 from absent cycle plumbing).
        dev.array.set_dma_cycle(50_000);

        // Fire INSTR_ERROR on a compute tile via the real production path.
        let (col, row) = (1u8, 2u8);
        dev.fire_event_generate_for_test(col, row, 69);

        // Cache populated with the right decode.
        let cache = dev.async_errors.last_cache().expect("cache must populate");
        let expected_err = async_errors::build_critical_aie_error_code(
            AmdxdnaErrorNum::AieInstruction,
            AmdxdnaErrorModule::AieCore,
        );
        assert_eq!(cache.err_code, expected_err, "err_code must decode INSTR_ERROR");
        assert_eq!(cache.ex_err_code, ((row as u64) << 8) | col as u64, "ex_err_code packs row|col");
        assert_eq!(cache.ts_us, 50, "ts_us = 50_000 cycles / 1000 = 50 us");

        // Ring at col 1 has the wire-format record.
        let ring = dev.async_errors.ring(col).expect("ring must exist");
        assert_eq!(ring.header().err_cnt, 1);
        let rec = &ring.records()[0];
        assert_eq!(rec.event_id, 69);
        assert_eq!(rec.row, row);
        assert_eq!(rec.col, col);
        assert_eq!(rec.mod_type, AieErrorOrigin::Core.wire_mod_type());
    }

    #[test]
    fn event_generate_for_non_error_event_does_not_record_async() {
        let mut dev = DeviceState::new_npu1();
        let (col, row) = (1u8, 2u8);
        // Event 7 is NOT in any error table (it's a generic user event).
        dev.fire_event_generate_for_test(col, row, 7);
        assert!(
            dev.async_errors.last_cache().is_none(),
            "non-error event must not populate the async cache"
        );
        assert_eq!(dev.async_errors.ring(col).unwrap().header().err_cnt, 0);
    }

    #[test]
    fn tier_a_fires_independently_of_tier_b_for_non_error_shim_event() {
        use crate::device::interrupts::{L1_REG_ENABLE_A, L1_REG_IRQ_NO_A, SwitchId};
        let mut dev = DeviceState::new_npu1();
        // Configure shim L1 to latch on event 7 (use 7 not 69 so this test
        // doesn't depend on INSTR_ERROR being in any shim event-slot mapping).
        // Then drive event 7 on a SHIM tile and verify L1 latched.
        let (col, row) = (0u8, 0u8);
        {
            let t = dev.array.get_mut(col, row).unwrap();
            let l1 = t.l1_irq.as_mut().unwrap();
            l1.set_irq_event_slot(SwitchId::A, 0, 7);
            l1.write_register(L1_REG_ENABLE_A, 1 << 16);
            l1.write_register(L1_REG_IRQ_NO_A, 5);
        }
        dev.fire_event_generate_for_test(col, row, 7);

        // Tier A: L1 latched + IRQ_NO queued.
        let t = dev.array.get(col, row).unwrap();
        let l1 = t.l1_irq.as_ref().unwrap();
        assert_ne!(l1.read_status(SwitchId::A) & (1 << 16), 0, "Tier A L1 must latch");
        assert!(t.pending_broadcasts.contains(&5), "Tier A IRQ_NO must queue");

        // Tier B: shim event 7 is NOT in the SHIM_EVENT_CAT table, so no
        // async record. This proves Tier A fires independently of Tier B
        // (a Tier-A-only event leaves Tier B's cache empty).
        assert!(
            dev.async_errors.last_cache().is_none(),
            "non-error shim event must not populate async cache; Tier B is independent"
        );
    }
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib interrupt_path_tests::event_generate_for_instr_error
```

Expected: tests fail — assertion failures (cache stays `None` because the Tier B hook isn't wired yet).

- [ ] **Step 3: Restructure the is_event_generate match and add the hook**

Open `src/device/state/effects.rs`. Replace the existing block at lines 362-406 with:

```rust
        // 11. Event_Generate: fire on trace units, broadcast, and (Tier B)
        // async-error pipeline.
        //
        // The EventModule above handles the event register write, but the
        // trace units, broadcast propagation, and the firmware async-error
        // path also need to be notified. Event_Generate offset is the first
        // register in the event block; the offset selects which module
        // fired (core vs mem on compute tiles), which in turn determines
        // the Tier B origin used for categorization.
        let origin = match tile.tile_kind {
            TileKind::Compute if offset == ce.event_generate => {
                Some(xdna_archspec::aie2::async_errors::AieErrorOrigin::Core)
            }
            TileKind::Compute if offset == me.event_generate => {
                Some(xdna_archspec::aie2::async_errors::AieErrorOrigin::Mem)
            }
            TileKind::Mem if offset == mte.event_generate => {
                Some(xdna_archspec::aie2::async_errors::AieErrorOrigin::MemTile)
            }
            TileKind::ShimNoc | TileKind::ShimPl if offset == ce.event_generate => {
                Some(xdna_archspec::aie2::async_errors::AieErrorOrigin::Pl)
            }
            _ => None,
        };
        if let Some(origin) = origin {
            let event_id = (value & 0x7F) as u8;
            log::info!(
                "Tile({},{}) Event_Generate: event_id={} (offset=0x{:X}) cycle={}",
                col,
                row,
                event_id,
                offset,
                current_cycle
            );

            // Fire the event directly on local trace units. Use the array's
            // current_cycle so trace unit deltas reflect real simulation time;
            // passing a hardcoded 0 here causes every generated event to look
            // like it fired at cycle 0.
            tile.core_trace.notify_event(event_id, current_cycle, None);
            tile.mem_trace.notify_event(event_id, current_cycle, None);

            // Check broadcast channel mapping in the EventModule: if the
            // generated event matches any broadcast channel's configured
            // event, queue the channel number for column propagation.
            //
            // Note: `pending_broadcasts` stores the channel number (0..15),
            // not a hw_id. Per-module hw_id translation happens at the
            // receiving tile in `propagate_broadcasts`, since each module
            // type sees BROADCAST_N at a different hw_id (compute core/mem
            // = 107+N, shim PL_A = 110+N, memtile = 142+N). Shared with the
            // hardware error path so the scan logic cannot drift.
            tile.seed_broadcasts_for_event(event_id);
            // Tier A interrupt path: a software-generated event is also
            // offered to this tile's L1 interrupt controller (shim only).
            // On latch, L1 queues its IRQ_NO into pending_broadcasts so the
            // existing propagate_broadcasts transport carries it to L2.
            tile.tap_l1_interrupt(event_id);

            // Tier B firmware async-error path: parallel to Tier A. On real
            // silicon, firmware delivers errors via mailbox regardless of
            // AIE L1/L2 enable state -- so this hook fires at event-generation,
            // not after L1 latches. The two paths are independent: an error
            // populates Tier B's cache + ring whether or not L1 was enabled.
            if xdna_archspec::aie2::async_errors::is_error_event(event_id, origin) {
                self.async_errors.record_error(col, row, origin, event_id, current_cycle);
            }
        }
```

Note: `self.async_errors.record_error` is called outside the `tile` borrow scope. Refactor the surrounding code if needed so that the `tile` mutable borrow ends before the `self.async_errors` mutable borrow begins. The existing code at line 362-406 already borrows `tile` extensively; extract the Tier A calls into a scoped block if the borrow checker complains.

If borrow-checker issues arise, hoist the `tile` operations into a self-contained block, capture `(col, row, current_cycle, event_id, origin, was_error)` into locals, then call `self.async_errors.record_error(...)` after the block ends.

- [ ] **Step 4: Run tests to verify they pass**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib interrupt_path_tests
```

Expected: all interrupt_path_tests pass — the 3 new Tier B tests plus the existing Tier A tests.

- [ ] **Step 5: Run the full library test suite to verify no regression**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib
```

Expected: all tests pass; test count is (previous total + the new tests in this plan so far).

- [ ] **Step 6: Commit**

```bash
git add src/device/state/effects.rs
git commit -m "$(cat <<'EOF'
interrupt: wire Tier B async-error hook into event-generate path

apply_tile_local_effects restructures is_event_generate to return an
AieErrorOrigin (Core / Mem / MemTile / Pl). When the origin is set and
the event is in the archspec error tables, record_error is called on
DeviceState::async_errors. Parallel to Tier A's L1/L2 latch path:
an error fires both, neither hides the other. INSTR_ERROR end-to-end
test verifies cache + ring populate; non-error-event test verifies
the gate does not record; independence test verifies Tier A fires
even when Tier B doesn't.

Generated using Claude Code.
EOF
)"
```

---

## Task 9: FFI module skeleton + handle field

Sets up the file structure for FFI symbols and adds the callback storage on
`XdnaEmuHandle`.

**Files:**
- Create: `crates/xdna-emu-ffi/src/async_errors.rs`
- Modify: `crates/xdna-emu-ffi/src/lib.rs`

- [ ] **Step 1: Create the FFI module with the shared types**

Create `crates/xdna-emu-ffi/src/async_errors.rs`:

```rust
//! FFI surface for Tier B async-error delivery.
//!
//! Five C symbols: cache reader, ring reader, ring-pending probe, callback
//! registration, clear helper. Plugin consumes `xdna_emu_get_last_async_error`
//! via `resolve_required`; the others are reserved for future consumers
//! (visual debugger, test harnesses, future kernel-driver attachment).
//!
//! Conventions match the rest of this crate (see `query.rs`):
//! - Opaque handle pointer; null handle returns sentinel.
//! - Last-error string set via `set_last_error` on failure paths.
//! - Buffers are copy-on-read; caller retains ownership.

use std::ffi::c_void;
use std::slice;

use xdna_emu_core::device::async_errors::AmdxdnaAsyncError;

use super::XdnaEmuHandle;

/// uapi-mirror of `struct amdxdna_async_error`. Exposed to C as the
/// out-parameter type for `xdna_emu_get_last_async_error` and the record
/// type passed to the push callback.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct XdnaEmuAsyncError {
    pub err_code: u64,
    pub ts_us: u64,
    pub ex_err_code: u64,
}

const _: () = assert!(std::mem::size_of::<XdnaEmuAsyncError>() == 24);

impl From<&AmdxdnaAsyncError> for XdnaEmuAsyncError {
    fn from(src: &AmdxdnaAsyncError) -> Self {
        Self { err_code: src.err_code, ts_us: src.ts_us, ex_err_code: src.ex_err_code }
    }
}

/// C callback signature for push-notification on error.
pub type XdnaEmuAsyncErrorCallback =
    unsafe extern "C" fn(record: *const XdnaEmuAsyncError, user_data: *mut c_void);
```

- [ ] **Step 2: Add the module declaration and re-exports in lib.rs**

Open `crates/xdna-emu-ffi/src/lib.rs`. In the `mod` block near the top (around lines 39-44), add:

```rust
mod async_errors;
```

In the `pub use` block (around lines 45-49), add:

```rust
pub use async_errors::*;
```

In the `XdnaEmuHandle` struct definition (around line 71), add the callback storage field. The pointer-pair shape matches the FFI registration pattern. The pointer must be wrapped in a small Rust-friendly wrapper so the field is `Send`-compatible enough for the existing usage (single-threaded via the plugin's mutex — see lib.rs:16-18).

Add a wrapper struct near the top of `crates/xdna-emu-ffi/src/lib.rs` (after the `LAST_ERROR` thread_local):

```rust
/// Wrapper around an FFI callback + user_data pointer pair.
///
/// `*mut c_void` is not `Send`; this struct asserts the caller follows
/// the documented "handles are not thread-safe" contract so we can store
/// the pair on the handle without inheriting `!Send` constraints on
/// `XdnaEmuHandle`.
#[derive(Clone, Copy)]
pub(crate) struct AsyncErrorCallback {
    pub func: async_errors::XdnaEmuAsyncErrorCallback,
    pub user_data: *mut std::ffi::c_void,
}
// SAFETY: handle access is serialized by the plugin's mutex; the user_data
// pointer is opaque to us and only ever passed back to the registered C
// callback on the same thread that registered it.
unsafe impl Send for AsyncErrorCallback {}
unsafe impl Sync for AsyncErrorCallback {}
```

Then in the `pub struct XdnaEmuHandle { ... }` body, add the field:

```rust
    /// Optional FFI-registered push callback for async errors (Tier B).
    pub(crate) async_callback: Option<AsyncErrorCallback>,
```

And in `xdna_emu_create` (around line 175), initialize the field to `None`:

```rust
    let handle = Box::new(XdnaEmuHandle {
        engine,
        xclbin_path: None,
        npu_executor: NpuExecutor::new(),
        max_cycles: config.max_cycles(),
        next_alloc_addr: 0x8000_0000_0000,
        free_list: Vec::new(),
        async_callback: None,
    });
```

- [ ] **Step 3: Build to verify the scaffolding compiles**

```bash
cargo build -p xdna-emu-ffi
```

Expected: clean build, no warnings about unused items beyond the obvious "this is a skeleton" set.

- [ ] **Step 4: Commit**

```bash
git add crates/xdna-emu-ffi/src/async_errors.rs crates/xdna-emu-ffi/src/lib.rs
git commit -m "$(cat <<'EOF'
interrupt: scaffold Tier B FFI module + handle callback storage

New crates/xdna-emu-ffi/src/async_errors.rs with XdnaEmuAsyncError
(24B repr-C mirror of amdxdna_async_error) and the callback typedef.
XdnaEmuHandle gains an async_callback: Option<AsyncErrorCallback>
field, initialized to None in xdna_emu_create. Actual FFI symbols
land in the next tasks.

Generated using Claude Code.
EOF
)"
```

---

## Task 10: FFI cache reader + clear helper

`xdna_emu_get_last_async_error` and `xdna_emu_clear_async_errors`. The
two simplest FFI symbols, perfect for shaking out the convention.

**Files:**
- Modify: `crates/xdna-emu-ffi/src/async_errors.rs`

- [ ] **Step 1: Write failing FFI tests**

Append to `crates/xdna-emu-ffi/src/async_errors.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{xdna_emu_create, xdna_emu_destroy};
    use xdna_archspec::aie2::async_errors::AieErrorOrigin;

    /// Helper: create a handle, run a closure, then destroy it.
    unsafe fn with_handle(f: impl FnOnce(*mut XdnaEmuHandle)) {
        let h = xdna_emu_create();
        assert!(!h.is_null());
        f(h);
        xdna_emu_destroy(h);
    }

    #[test]
    fn get_last_async_error_returns_zero_on_fresh_handle() {
        unsafe {
            with_handle(|h| {
                let mut out = XdnaEmuAsyncError::default();
                let rc = xdna_emu_get_last_async_error(h, &mut out);
                assert_eq!(rc, 0, "no record yet -> 0");
                assert_eq!(out.err_code, 0);
            });
        }
    }

    #[test]
    fn get_last_async_error_returns_one_after_record() {
        unsafe {
            with_handle(|h| {
                // Drive a record directly through the sink (no engine step needed).
                let dev = (*h).engine.device_mut();
                dev.array.set_dma_cycle(50_000);
                dev.async_errors.record_error(1, 2, AieErrorOrigin::Core, 69, 50_000);

                let mut out = XdnaEmuAsyncError::default();
                let rc = xdna_emu_get_last_async_error(h, &mut out);
                assert_eq!(rc, 1, "record present -> 1");
                assert_eq!(out.ts_us, 50);
                assert_eq!(out.ex_err_code, (2u64 << 8) | 1u64);
            });
        }
    }

    #[test]
    fn get_last_async_error_null_handle_returns_minus_one() {
        unsafe {
            let mut out = XdnaEmuAsyncError::default();
            let rc = xdna_emu_get_last_async_error(std::ptr::null_mut(), &mut out);
            assert_eq!(rc, -1);
        }
    }

    #[test]
    fn get_last_async_error_null_out_returns_minus_two() {
        unsafe {
            with_handle(|h| {
                let rc = xdna_emu_get_last_async_error(h, std::ptr::null_mut());
                assert_eq!(rc, -2);
            });
        }
    }

    #[test]
    fn clear_async_errors_resets_cache() {
        unsafe {
            with_handle(|h| {
                let dev = (*h).engine.device_mut();
                dev.async_errors.record_error(1, 2, AieErrorOrigin::Core, 69, 50_000);
                let rc = xdna_emu_clear_async_errors(h);
                assert_eq!(rc, 0);

                let mut out = XdnaEmuAsyncError::default();
                let rc = xdna_emu_get_last_async_error(h, &mut out);
                assert_eq!(rc, 0, "clear must drop the cache");
            });
        }
    }

    #[test]
    fn clear_async_errors_null_handle_returns_minus_one() {
        unsafe {
            let rc = xdna_emu_clear_async_errors(std::ptr::null_mut());
            assert_eq!(rc, -1);
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --lib async_errors
```

Expected: compile errors for `xdna_emu_get_last_async_error`, `xdna_emu_clear_async_errors`.

- [ ] **Step 3: Implement the FFI symbols**

Append to `crates/xdna-emu-ffi/src/async_errors.rs`, BEFORE the `#[cfg(test)]` block:

```rust
/// Read the last-recorded async-error record into `out`.
///
/// Returns:
///   1 if a record is populated and copied to `*out`
///   0 if no errors have been recorded since the last reset
///  -1 if `handle` is null
///  -2 if `out` is null
///
/// # Safety
/// `handle` must be a valid pointer from `xdna_emu_create`; `out` must point
/// to at least `sizeof(XdnaEmuAsyncError)` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_get_last_async_error(
    handle: *mut XdnaEmuHandle,
    out: *mut XdnaEmuAsyncError,
) -> i32 {
    if handle.is_null() {
        return -1;
    }
    if out.is_null() {
        return -2;
    }
    let handle = &mut *handle;
    match handle.engine.device().async_errors.last_cache() {
        Some(rec) => {
            *out = XdnaEmuAsyncError::from(rec);
            1
        }
        None => 0,
    }
}

/// Clear the async-error cache, all per-column rings, and the drain queue.
/// Does NOT touch Tier A L1/L2 latch state or any other tile state.
///
/// Returns:
///   0 on success
///  -1 if `handle` is null
///
/// # Safety
/// `handle` must be a valid pointer from `xdna_emu_create`.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_clear_async_errors(handle: *mut XdnaEmuHandle) -> i32 {
    if handle.is_null() {
        return -1;
    }
    let handle = &mut *handle;
    handle.engine.device_mut().async_errors.clear();
    0
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --lib async_errors
```

Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/xdna-emu-ffi/src/async_errors.rs
git commit -m "$(cat <<'EOF'
interrupt: add FFI cache reader + clear for async errors

xdna_emu_get_last_async_error returns 1/0/-1/-2 (record-present /
empty / null-handle / null-out). xdna_emu_clear_async_errors drops
the cache + all rings + drain queue without touching tile state.

Generated using Claude Code.
EOF
)"
```

---

## Task 11: FFI ring reader + pending probe

`xdna_emu_read_async_event_ring` and `xdna_emu_async_event_pending` for the
future kernel-driver attachment path.

**Files:**
- Modify: `crates/xdna-emu-ffi/src/async_errors.rs`

- [ ] **Step 1: Write failing tests**

Append to the `#[cfg(test)] mod tests` block:

```rust
    #[test]
    fn read_async_event_ring_returns_header_only_when_empty() {
        unsafe {
            with_handle(|h| {
                let mut buf = vec![0u8; 64];
                let n = xdna_emu_read_async_event_ring(h, 0, buf.as_mut_ptr(), buf.len() as u64);
                // Empty ring returns just the 12-byte header (err_cnt = 0).
                assert_eq!(n, 12);
                assert_eq!(&buf[0..4], &0u32.to_le_bytes());
            });
        }
    }

    #[test]
    fn read_async_event_ring_returns_payload_after_record() {
        unsafe {
            with_handle(|h| {
                let dev = (*h).engine.device_mut();
                dev.async_errors.record_error(1, 2, AieErrorOrigin::Core, 69, 1_000);

                let mut buf = vec![0u8; 64];
                let n = xdna_emu_read_async_event_ring(h, 1, buf.as_mut_ptr(), buf.len() as u64);
                // 12-byte header + 12-byte record = 24 bytes.
                assert_eq!(n, 24);
                assert_eq!(&buf[0..4], &1u32.to_le_bytes(), "err_cnt = 1");
                assert_eq!(buf[12], 2, "record row");
                assert_eq!(buf[13], 1, "record col");
                assert_eq!(buf[20], 69, "record event_id (offset 12+8)");
            });
        }
    }

    #[test]
    fn read_async_event_ring_invalid_col_returns_minus_two() {
        unsafe {
            with_handle(|h| {
                let mut buf = vec![0u8; 64];
                let n = xdna_emu_read_async_event_ring(h, 99, buf.as_mut_ptr(), buf.len() as u64);
                assert_eq!(n, -2);
            });
        }
    }

    #[test]
    fn read_async_event_ring_null_buf_returns_minus_three() {
        unsafe {
            with_handle(|h| {
                let n = xdna_emu_read_async_event_ring(h, 0, std::ptr::null_mut(), 64);
                assert_eq!(n, -3);
            });
        }
    }

    #[test]
    fn read_async_event_ring_null_handle_returns_minus_one() {
        unsafe {
            let mut buf = vec![0u8; 64];
            let n = xdna_emu_read_async_event_ring(std::ptr::null_mut(), 0, buf.as_mut_ptr(), buf.len() as u64);
            assert_eq!(n, -1);
        }
    }

    #[test]
    fn async_event_pending_zero_on_empty() {
        unsafe {
            with_handle(|h| {
                assert_eq!(xdna_emu_async_event_pending(h, 0), 0);
            });
        }
    }

    #[test]
    fn async_event_pending_one_after_record() {
        unsafe {
            with_handle(|h| {
                let dev = (*h).engine.device_mut();
                dev.async_errors.record_error(3, 2, AieErrorOrigin::Core, 69, 1_000);
                assert_eq!(xdna_emu_async_event_pending(h, 3), 1);
                assert_eq!(xdna_emu_async_event_pending(h, 0), 0, "col 0 still empty");
            });
        }
    }

    #[test]
    fn async_event_pending_invalid_col_returns_minus_two() {
        unsafe {
            with_handle(|h| {
                assert_eq!(xdna_emu_async_event_pending(h, 99), -2);
            });
        }
    }
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --lib async_errors
```

Expected: compile errors for `xdna_emu_read_async_event_ring`, `xdna_emu_async_event_pending`.

- [ ] **Step 3: Implement the symbols**

Append to `crates/xdna-emu-ffi/src/async_errors.rs`, BEFORE the `#[cfg(test)]` block:

```rust
/// Copy up to `buf_size` bytes from column `col`'s ring buffer into `buf`.
/// Bytes are driver-wire format: `AieErrInfoHeader` (12B) followed by
/// `err_cnt * AieError` (12B each).
///
/// Returns the number of bytes copied (always at least 12 for the header;
/// 0 only on error), or:
///   -1 if `handle` is null
///   -2 if `col` is out of range for this device
///   -3 if `buf` is null
///
/// # Safety
/// `handle` must be valid; `buf` must point to at least `buf_size` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_read_async_event_ring(
    handle: *mut XdnaEmuHandle,
    col: u32,
    buf: *mut u8,
    buf_size: u64,
) -> i64 {
    if handle.is_null() {
        return -1;
    }
    if buf.is_null() {
        return -3;
    }
    let handle = &mut *handle;
    let col_u8 = match u8::try_from(col) {
        Ok(c) => c,
        Err(_) => return -2,
    };
    let ring = match handle.engine.device().async_errors.ring(col_u8) {
        Some(r) => r,
        None => return -2,
    };
    let dst = slice::from_raw_parts_mut(buf, buf_size as usize);
    ring.read_into(dst) as i64
}

/// Probe whether column `col`'s ring has any pending records.
///
/// Returns 1 if `err_cnt > 0`, 0 if empty, or:
///   -1 if `handle` is null
///   -2 if `col` is out of range
///
/// # Safety
/// `handle` must be valid.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_async_event_pending(
    handle: *mut XdnaEmuHandle,
    col: u32,
) -> i32 {
    if handle.is_null() {
        return -1;
    }
    let handle = &mut *handle;
    let col_u8 = match u8::try_from(col) {
        Ok(c) => c,
        Err(_) => return -2,
    };
    match handle.engine.device().async_errors.ring(col_u8) {
        Some(r) if r.header().err_cnt > 0 => 1,
        Some(_) => 0,
        None => -2,
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --lib async_errors
```

Expected: all FFI async_errors tests pass (Task 10's 6 + Task 11's 8 = 14 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/xdna-emu-ffi/src/async_errors.rs
git commit -m "$(cat <<'EOF'
interrupt: add FFI ring reader + pending probe

xdna_emu_read_async_event_ring returns header + payload bytes in
driver-wire format for the given column (0 ring -> 12B header only).
xdna_emu_async_event_pending probes err_cnt > 0 without copying.
Both return -1/-2/-3 for null-handle / invalid-col / null-buf.

Generated using Claude Code.
EOF
)"
```

---

## Task 12: FFI callback registration + drain integration

`xdna_emu_set_async_event_callback` plus the per-step drain loop in
`xdna_emu_run` that fires the registered callback for each newly-recorded
error.

**Files:**
- Modify: `crates/xdna-emu-ffi/src/async_errors.rs`
- Modify: `crates/xdna-emu-ffi/src/execution.rs`

- [ ] **Step 1: Write failing tests for callback registration**

Append to the `#[cfg(test)] mod tests` block in `crates/xdna-emu-ffi/src/async_errors.rs`:

```rust
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Mutex;

    /// Test-only callback that increments a counter and records the last
    /// observed err_code into a Mutex<Option<u64>>.
    static OBSERVED: Mutex<Option<u64>> = Mutex::new(None);
    static FIRE_COUNT: AtomicU32 = AtomicU32::new(0);

    unsafe extern "C" fn test_callback(rec: *const XdnaEmuAsyncError, _ud: *mut std::ffi::c_void) {
        FIRE_COUNT.fetch_add(1, Ordering::SeqCst);
        if !rec.is_null() {
            *OBSERVED.lock().unwrap() = Some((*rec).err_code);
        }
    }

    #[test]
    fn set_async_event_callback_registers_and_fires_on_drain() {
        unsafe {
            with_handle(|h| {
                FIRE_COUNT.store(0, Ordering::SeqCst);
                *OBSERVED.lock().unwrap() = None;

                let rc = xdna_emu_set_async_event_callback(h, Some(test_callback), std::ptr::null_mut());
                assert_eq!(rc, 0);

                // Drive a record, then call the drain helper directly.
                let dev = (*h).engine.device_mut();
                dev.async_errors.record_error(1, 2, AieErrorOrigin::Core, 69, 50_000);

                // The drain happens inside xdna_emu_run between engine steps.
                // For this unit test, exercise the helper directly.
                fire_async_callbacks_for(&mut *h);

                assert_eq!(FIRE_COUNT.load(Ordering::SeqCst), 1);
                assert!(OBSERVED.lock().unwrap().is_some());
            });
        }
    }

    #[test]
    fn set_async_event_callback_with_none_unregisters() {
        unsafe {
            with_handle(|h| {
                FIRE_COUNT.store(0, Ordering::SeqCst);
                xdna_emu_set_async_event_callback(h, Some(test_callback), std::ptr::null_mut());
                xdna_emu_set_async_event_callback(h, None, std::ptr::null_mut());

                let dev = (*h).engine.device_mut();
                dev.async_errors.record_error(1, 2, AieErrorOrigin::Core, 69, 50_000);

                fire_async_callbacks_for(&mut *h);
                assert_eq!(FIRE_COUNT.load(Ordering::SeqCst), 0, "unregistered callback must not fire");
            });
        }
    }

    #[test]
    fn set_async_event_callback_null_handle_returns_minus_one() {
        unsafe {
            assert_eq!(
                xdna_emu_set_async_event_callback(std::ptr::null_mut(), Some(test_callback), std::ptr::null_mut()),
                -1
            );
        }
    }
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --lib async_errors
```

Expected: compile errors for `xdna_emu_set_async_event_callback`, `fire_async_callbacks_for`.

- [ ] **Step 3: Implement registration and the drain helper**

Append to `crates/xdna-emu-ffi/src/async_errors.rs`, BEFORE the `#[cfg(test)]` block:

```rust
/// Register a C callback fired synchronously when an async error is recorded.
/// Pass `None` to unregister. `user_data` is round-tripped to each invocation.
///
/// Thread-safety: the callback fires from whichever thread drives `xdna_emu_run`.
/// Per the lib.rs handle-safety contract, that is expected to be a single
/// thread per handle (the XRT plugin serializes via its own mutex).
///
/// Returns 0 on success, -1 if `handle` is null.
///
/// # Safety
/// `handle` must be valid; the `callback` (if Some) must be a valid C function
/// pointer matching the `XdnaEmuAsyncErrorCallback` signature.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_set_async_event_callback(
    handle: *mut XdnaEmuHandle,
    callback: Option<XdnaEmuAsyncErrorCallback>,
    user_data: *mut c_void,
) -> i32 {
    if handle.is_null() {
        return -1;
    }
    let handle = &mut *handle;
    handle.async_callback = callback.map(|func| crate::AsyncErrorCallback { func, user_data });
    0
}

/// Drain newly-recorded async-error records and fire the registered callback
/// (if any) for each. Called from the run loop between engine steps.
///
/// # Safety
/// `handle` must be a valid mutable reference.
pub(crate) unsafe fn fire_async_callbacks_for(handle: &mut XdnaEmuHandle) {
    let Some(cb) = handle.async_callback else { return };
    let drained = handle.engine.device_mut().async_errors.drain_newly_recorded();
    for rec in drained {
        let xrec = XdnaEmuAsyncError::from(&rec);
        (cb.func)(&xrec as *const _, cb.user_data);
    }
}
```

- [ ] **Step 4: Wire the drain into xdna_emu_run**

Open `crates/xdna-emu-ffi/src/execution.rs`. Inside `xdna_emu_run`, locate the line:

```rust
        handle.engine.step();
```

(around line 237). Immediately after it, add:

```rust
        // Tier B: drain newly-recorded async errors and fire the registered
        // callback (if any). Mirrors the flush_trace_to_host pattern -- FFI
        // layer observes between engine steps.
        crate::async_errors::fire_async_callbacks_for(handle);
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --lib async_errors
TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --lib  # full crate, no regression
```

Expected: callback tests pass (3 new); other FFI tests still green.

- [ ] **Step 6: Commit**

```bash
git add crates/xdna-emu-ffi/src/async_errors.rs crates/xdna-emu-ffi/src/execution.rs
git commit -m "$(cat <<'EOF'
interrupt: add FFI callback registration + drain in xdna_emu_run

xdna_emu_set_async_event_callback stores an opt-in C callback on the
handle. fire_async_callbacks_for drains newly-recorded records and
invokes the callback for each. Wired into xdna_emu_run after every
engine.step() -- same observation cadence as flush_trace_to_host.

Generated using Claude Code.
EOF
)"
```

---

## Task 13: Plugin C++ wire-up

Replaces `platform_emu.cpp`'s `DRM_AMDXDNA_HW_LAST_ASYNC_ERR` no-op with
a `resolve_required` call into the new FFI symbol.

**Files:**
- Modify: `xrt-plugin/src/transport_inprocess.h`
- Modify: `xrt-plugin/src/transport_inprocess.cpp`
- Modify: `xrt-plugin/src/platform_emu.cpp`

- [ ] **Step 1: Read existing plugin patterns for context**

Before editing, read the existing patterns so the new code matches:

```bash
grep -n "resolve_required\|resolve_optional" xrt-plugin/src/transport_inprocess.cpp | head -10
grep -n "sym_" xrt-plugin/src/transport_inprocess.h | head -10
grep -n "DRM_AMDXDNA_HW_" xrt-plugin/src/platform_emu.cpp | head -10
```

These show the exact pattern (typedef declaration, member declaration, `resolve_required` call, member invocation).

- [ ] **Step 2: Add the FFI typedef and member to the transport header**

Open `xrt-plugin/src/transport_inprocess.h`. Find the block of `fn_*` typedefs and add (matching the style of nearby typedefs):

```cpp
// Tier B async-error cache reader. Returns 1 if a record is copied to *out,
// 0 if empty, -1 null-handle, -2 null-out. See crates/xdna-emu-ffi/src/async_errors.rs.
struct XdnaEmuAsyncError {  // 24-byte mirror of amdxdna_async_error
    uint64_t err_code;
    uint64_t ts_us;
    uint64_t ex_err_code;
};
using fn_get_last_async_error = int32_t (*)(XdnaEmuHandle*, XdnaEmuAsyncError*);
```

(`XdnaEmuHandle` is already declared opaquely in this file — adjust the snippet to match the exact declaration style.)

In the member-variable block (where other `sym_*_` fields live), add:

```cpp
    fn_get_last_async_error sym_get_last_async_error_;
```

- [ ] **Step 3: Add the resolve_required call in the transport ctor**

Open `xrt-plugin/src/transport_inprocess.cpp`. In the constructor / init block where other symbols are resolved, add:

```cpp
sym_get_last_async_error_ = resolve_required<fn_get_last_async_error>(
    "xdna_emu_get_last_async_error");
```

Place it in the same alphabetical / logical grouping as the other `resolve_required` calls.

- [ ] **Step 4: Replace the no-op ioctl case in platform_emu**

Open `xrt-plugin/src/platform_emu.cpp`. Find `case DRM_AMDXDNA_HW_LAST_ASYNC_ERR:` (around line 1189) and replace the body:

```cpp
  case DRM_AMDXDNA_HW_LAST_ASYNC_ERR: {
    if (arg.buffer_size < sizeof(amdxdna_async_error))
      shim_err(EINVAL, "get_info_array: buffer too small for async error");

    amdxdna_async_error rec{};
    auto& transport = m_drv->transport();
    int32_t got = transport.sym_get_last_async_error_(
        transport.handle(),
        reinterpret_cast<XdnaEmuAsyncError*>(&rec));

    if (got == 1) {
      std::memcpy(arg.buffer, &rec, sizeof(rec));
      arg.num_element = 1;
    } else {
      arg.num_element = 0;  // no record yet, or FFI error; same surface as old no-op
    }
    break;
  }
```

(Confirm exact accessor names on `m_drv` / `transport` by reading the existing surrounding cases in `platform_emu.cpp`. If the plugin exposes the transport differently, adapt the access path.)

- [ ] **Step 5: Build the FFI cdylib first**

Bridge tests load `libxdna_emu.so`. The FFI symbol must exist in the cdylib before the plugin tries to resolve it. Build both:

```bash
cargo build -p xdna-emu-ffi
cargo build -p xdna-emu-ffi --release
```

Expected: clean builds. The new symbol `xdna_emu_get_last_async_error` is present in both `target/debug/libxdna_emu.so` and `target/release/libxdna_emu.so`.

- [ ] **Step 6: Build the plugin**

```bash
./scripts/rebuild-plugin.sh
```

Expected: clean build; the new resolve_required call succeeds against the rebuilt `.so`.

- [ ] **Step 7: Run the FFI completeness test**

```bash
TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu-ffi --lib test_ffi_interface_completeness
```

Expected: PASS. The auto-discovery test scans `xrt-plugin/src/transport_inprocess.cpp` for `resolve_required` and `resolve_optional` calls, then checks every named symbol is exported from this crate. `xdna_emu_get_last_async_error` should now be both expected (from C++) and exported (from Rust).

- [ ] **Step 8: Run the full library test suite**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib
```

Expected: all tests pass. Compare the count to the pre-Tier-B baseline (3094 passed before this plan started, per the post-Tier-A push) plus the new tests added across this plan (Tasks 1-12 add roughly 35 tests in total).

- [ ] **Step 9: Commit**

```bash
git add xrt-plugin/src/transport_inprocess.h xrt-plugin/src/transport_inprocess.cpp xrt-plugin/src/platform_emu.cpp
git commit -m "$(cat <<'EOF'
interrupt: wire DRM_AMDXDNA_HW_LAST_ASYNC_ERR ioctl to Tier B cache

Plugin resolve_required's xdna_emu_get_last_async_error and surfaces
the populated amdxdna_async_error to userspace. Replaces the longstanding
"No async errors in emulation" no-op (platform_emu.cpp:1189). Stale
emu .so without the symbol fails to load loudly (resolve_required, not
optional) -- matches the rebuild-plugin / refresh-dkms discipline.

Generated using Claude Code.
EOF
)"
```

---

## Final verification

After all 13 tasks land:

- [ ] Run the full library suite and confirm pass count and no regressions:

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib
```

- [ ] Confirm the FFI .so contains the new symbol:

```bash
nm -D target/debug/libxdna_emu.so | grep -E "xdna_emu_(get_last_async|read_async|async_event_pending|set_async_event|clear_async)"
```

Expected: 5 symbols listed.

- [ ] Confirm the plugin links cleanly against the new .so:

```bash
ldd /opt/xilinx/xrt/lib/libxrt_driver_emu.so.2 | grep -v "not found"
```

Expected: no "not found" lines.

- [ ] Push to dev (only after final code review):

```bash
git push origin dev
```

---

## Plan self-review notes

- **Spec coverage:** every locked decision in spec section 2 has a corresponding task. Cache (Task 6,7,10), per-column rings (Task 5,6,11), push callback (Task 12), driver-mirror categorization (Task 2,3), wire format (Task 4), trigger at event-generation (Task 8), `resolve_required` plugin wire-up (Task 13), cycle/1000 timestamp (Task 6), INSTR_ERROR as Instruction emu-specific entry (Task 2). ts_us scaling tested in Task 6+8. FFI completeness test gating in Task 13.
- **No placeholders:** every code step has complete copy-pasteable code with exact function signatures and field names. The only "look at existing code first" guidance is in Task 13 step 1, which is intentional — the C++ plugin patterns vary between accessor styles and the safest path is to grep for the actual idiom before adapting.
- **Type consistency:** `AieErrorOrigin` (Task 1) → used in Task 2's lookup tables, Task 3's encoding, Task 6's sink, Task 8's hook. `AsyncErrorSink::record_error(col, row, origin, event_id, cycle)` signature consistent from Task 6 onward. `AsyncRing::push -> Result<(), Overflow>` consistent in Task 5 + Task 6. FFI return code convention (1/0/-1/-2/-3) consistent across Tasks 10-12.

---

## Out of scope reminder (from spec §1.2)

- Per-class detection producers (DMA, parity, ECC, lock, stream) — separate Tier B follow-up specs.
- Wall-clock timestamp mode.
- Bridge test addition (manual ad-hoc test after this lands; bridge fixture in first follow-up).
- Tier C TDR / context-recovery.

These do NOT get tasks in this plan.
