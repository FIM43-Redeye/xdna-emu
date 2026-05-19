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
