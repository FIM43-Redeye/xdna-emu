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

    /// Derive from `TileKind`. Compute -> Core. ShimNoc/ShimPl both map to Pl.
    /// Does not cover the mem-module of compute tiles; callers that know
    /// which module fired should construct `AieErrorOrigin` directly.
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

    #[test]
    fn core_table_has_expected_entries() {
        assert!(CORE_EVENT_CAT
            .iter()
            .any(|e| e.event_id == 55 && e.category == AieErrorCategory::Access));
        assert!(CORE_EVENT_CAT
            .iter()
            .any(|e| e.event_id == 70 && e.category == AieErrorCategory::Instruction));
        // emu-specific INSTR_ERROR entry
        assert!(CORE_EVENT_CAT
            .iter()
            .any(|e| e.event_id == 69 && e.category == AieErrorCategory::Instruction));
    }

    #[test]
    fn mem_table_has_expected_entries() {
        assert!(MEM_EVENT_CAT
            .iter()
            .any(|e| e.event_id == 88 && e.category == AieErrorCategory::Ecc));
        assert!(MEM_EVENT_CAT
            .iter()
            .any(|e| e.event_id == 97 && e.category == AieErrorCategory::Dma));
        assert!(MEM_EVENT_CAT
            .iter()
            .any(|e| e.event_id == 101 && e.category == AieErrorCategory::Lock));
    }

    #[test]
    fn memtile_table_has_expected_entries() {
        assert!(MEMTILE_EVENT_CAT
            .iter()
            .any(|e| e.event_id == 130 && e.category == AieErrorCategory::Ecc));
        assert!(MEMTILE_EVENT_CAT
            .iter()
            .any(|e| e.event_id == 135 && e.category == AieErrorCategory::Stream));
        assert!(MEMTILE_EVENT_CAT
            .iter()
            .any(|e| e.event_id == 139 && e.category == AieErrorCategory::Lock));
    }

    #[test]
    fn shim_table_has_11_entries_and_expected_values() {
        assert_eq!(SHIM_EVENT_CAT.len(), 11);
        assert!(SHIM_EVENT_CAT
            .iter()
            .any(|e| e.event_id == 64 && e.category == AieErrorCategory::Bus));
        assert!(SHIM_EVENT_CAT
            .iter()
            .any(|e| e.event_id == 72 && e.category == AieErrorCategory::Dma));
        assert!(SHIM_EVENT_CAT
            .iter()
            .any(|e| e.event_id == 74 && e.category == AieErrorCategory::Lock));
    }

    #[test]
    fn event_to_category_returns_none_for_unknown() {
        assert!(event_to_category(0xFF, AieErrorOrigin::Core).is_none());
        assert!(event_to_category(0xFF, AieErrorOrigin::Mem).is_none());
        assert!(event_to_category(0xFF, AieErrorOrigin::MemTile).is_none());
        assert!(event_to_category(0xFF, AieErrorOrigin::Pl).is_none());
    }

    #[test]
    fn is_error_event_gate() {
        assert!(is_error_event(55, AieErrorOrigin::Core));
        assert!(!is_error_event(0xFF, AieErrorOrigin::Core));
    }
}
