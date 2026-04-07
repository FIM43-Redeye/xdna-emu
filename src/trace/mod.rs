//! Hardware trace event ID mappings and trace infrastructure.
//!
//! Maps emulator `EventType` values to AIE2 hardware event IDs used by the
//! binary trace unit (`TraceUnit`). The emulator produces the same binary
//! trace format as real NPU hardware, decoded by mlir-aie's `parse.py`.
//!
//! Also provides build-time generated event code tables, a GUI trace store,
//! and VCD-to-Perfetto conversion for aiesimulator integration.

pub mod compare;
pub mod store;
pub mod vcd;

/// Generated trace event codes from mlir-aie's canonical enums.
///
/// Provides per-tile-type event code constants and name lookup functions.
/// Generated at build time by `build.rs` from `tools/mlir-aie-bridge.py trace-events`.
/// Falls back to stubs if mlir-aie is not available.
pub mod event_codes {
    include!(concat!(env!("OUT_DIR"), "/trace_event_codes.rs"));
}

use crate::interpreter::state::EventType;

/// Validate compiled-in trace event tables against current mlir-aie.
///
/// Spot-checks key event codes from the build-time generated tables against
/// live bridge output. Reports any mismatches, which indicate mlir-aie was
/// updated after the last build. Returns Ok(()) if all checks pass, or an
/// error if the bridge invocation fails (bridge unavailable is not an error).
///
/// Requires the `tooling` feature (needs the integration::bridge module).
#[cfg(feature = "tooling")]
pub fn validate_trace_events(
    bridge: &crate::integration::bridge::BridgePath,
) -> Result<Vec<String>, String> {
    let json = crate::integration::bridge::invoke_bridge(bridge, "trace-events", &[])?;

    let mut warnings = Vec::new();

    // Spot-check core events.
    let core = &json["enums"]["CoreEvent"];
    let core_checks: &[(&str, u8)] = &[
        ("INSTR_VECTOR", event_codes::core_events::INSTR_VECTOR),
        ("MEMORY_STALL", event_codes::core_events::MEMORY_STALL),
        ("LOCK_STALL", event_codes::core_events::LOCK_STALL),
        ("ACTIVE", event_codes::core_events::ACTIVE),
        ("DISABLED", event_codes::core_events::DISABLED),
        ("INSTR_LOAD", event_codes::core_events::INSTR_LOAD),
    ];

    for (name, expected) in core_checks {
        if let Some(actual) = core[*name].as_u64() {
            if actual as u8 != *expected {
                warnings.push(format!(
                    "CoreEvent::{} compiled={} mlir-aie={}",
                    name, expected, actual,
                ));
            }
        }
    }

    // Spot-check mem events.
    let mem = &json["enums"]["MemEvent"];
    let mem_checks: &[(&str, u8)] = &[
        ("DMA_S2MM_0_START_TASK", event_codes::mem_events::DMA_S2MM_0_START_TASK),
        ("DMA_MM2S_0_START_TASK", event_codes::mem_events::DMA_MM2S_0_START_TASK),
    ];

    for (name, expected) in mem_checks {
        if let Some(actual) = mem[*name].as_u64() {
            if actual as u8 != *expected {
                warnings.push(format!(
                    "MemEvent::{} compiled={} mlir-aie={}",
                    name, expected, actual,
                ));
            }
        }
    }

    Ok(warnings)
}

/// Map an emulator `EventType` to its AIE2 hardware event ID.
///
/// These IDs come from mlir-aie `python/utils/trace/events/aie2.py`.
/// Core module events and memory module events use different ID spaces.
///
/// # Core Module Events
///
/// These are the event IDs for events originating in the Core module
/// (instruction events, stalls, core status).
///
/// # Memory Module Events
///
/// DMA and lock events use per-channel IDs. The `channel` field in DMA
/// events encodes S2MM (0-1) vs MM2S (2-3) for compute tiles.
pub fn core_event_to_hw_id(event: &EventType) -> Option<u8> {
    match event {
        // User-defined events (Core module)
        EventType::InstrEvent { id: 0, .. }    => Some(33),
        EventType::InstrEvent { id: 1, .. }    => Some(34),
        EventType::InstrEvent { .. }           => None, // id 2/3 are AIE2P only
        // Stall events (Core module)
        EventType::MemoryStall { .. }          => Some(23),
        EventType::StreamStall { .. }          => Some(24),
        EventType::LockStall { .. }            => Some(26),
        // Core status
        EventType::CoreActive                  => Some(28),
        EventType::CoreDisabled                => Some(29),
        // Program flow events
        EventType::InstrCall { .. }            => Some(35),
        EventType::InstrReturn { .. }          => Some(36),
        EventType::InstrVector { .. }          => Some(37),
        EventType::InstrLoad { .. }            => Some(38),
        EventType::InstrStore { .. }           => Some(39),
        EventType::InstrStreamGet { .. }       => Some(40),
        EventType::InstrStreamPut { .. }       => Some(41),
        EventType::InstrLockAcquireReq { .. }  => Some(44),
        EventType::InstrLockReleaseReq { .. }  => Some(45),
        // Branch is an emulator-internal event, no hardware equivalent
        EventType::BranchTaken { .. }          => None,
        // Memory module events don't have core event IDs
        _ => None,
    }
}

/// Map an emulator `EventType` to its AIE2 compute tile memory module event ID.
///
/// DMA events are per-channel. The channel field encodes:
/// - 0 = S2MM ch0, 1 = S2MM ch1, 2 = MM2S ch0, 3 = MM2S ch1
///
/// Lock events are per-lock (lock IDs 0-7 have dedicated event IDs).
///
/// Event IDs from mlir-aie `python/utils/trace/events/aie2.py` MemEvent.
pub fn mem_event_to_hw_id(event: &EventType) -> Option<u8> {
    match event {
        // DMA start task: S2MM_0=19, S2MM_1=20, MM2S_0=21, MM2S_1=22
        EventType::DmaStartTask { channel } => match channel {
            0 => Some(19),
            1 => Some(20),
            2 => Some(21),
            3 => Some(22),
            _ => None,
        },
        // DMA finished BD: S2MM_0=23, S2MM_1=24, MM2S_0=25, MM2S_1=26
        EventType::DmaFinishedBd { channel } => match channel {
            0 => Some(23),
            1 => Some(24),
            2 => Some(25),
            3 => Some(26),
            _ => None,
        },
        // DMA finished task: S2MM_0=27, S2MM_1=28, MM2S_0=29, MM2S_1=30
        EventType::DmaFinishedTask { channel } => match channel {
            0 => Some(27),
            1 => Some(28),
            2 => Some(29),
            3 => Some(30),
            _ => None,
        },
        // DMA stalled lock: S2MM_0=31, S2MM_1=32, MM2S_0=33, MM2S_1=34
        EventType::DmaStalledLock { channel } => match channel {
            0 => Some(31),
            1 => Some(32),
            2 => Some(33),
            3 => Some(34),
            _ => None,
        },
        // DMA stream starvation: S2MM_0=35, S2MM_1=36, MM2S_0=37(backpressure), MM2S_1=38
        EventType::DmaStreamStarvation { channel } => match channel {
            0 => Some(35),
            1 => Some(36),
            2 => Some(37),
            3 => Some(38),
            _ => None,
        },
        // Lock acquire: LOCK_SEL0_ACQ_GE=45, stride 4 per lock
        // Lock 0 acq_ge=45, Lock 1 acq_ge=49, ..., Lock 7 acq_ge=73
        EventType::LockAcquire { lock_id } => {
            if *lock_id <= 7 {
                Some(45 + (*lock_id as u8) * 4)
            } else {
                None
            }
        },
        // Lock release: LOCK_0_REL=46, stride 4 per lock
        // Lock 0 rel=46, Lock 1 rel=50, ..., Lock 7 rel=74
        EventType::LockRelease { lock_id } => {
            if *lock_id <= 7 {
                Some(46 + (*lock_id as u8) * 4)
            } else {
                None
            }
        },
        // Core events don't have memory module IDs
        _ => None,
    }
}

/// Map an emulator `EventType` to its AIE2 MemTile hardware event ID.
///
/// MemTiles use a different event ID namespace than compute tile memory modules.
/// DMA events are offset by +2 from compute tile IDs, and use SEL0/SEL1 naming
/// for the two channel groups.
///
/// Event IDs from mlir-aie `python/utils/trace/events/aie2.py` MemTileEvent.
pub fn memtile_event_to_hw_id(event: &EventType) -> Option<u8> {
    match event {
        // DMA start task: S2MM_SEL0=21, S2MM_SEL1=22, MM2S_SEL0=23, MM2S_SEL1=24
        EventType::DmaStartTask { channel } => match channel {
            0 => Some(21),
            1 => Some(22),
            2 => Some(23),
            3 => Some(24),
            _ => None,
        },
        // DMA finished BD: S2MM_SEL0=25, S2MM_SEL1=26, MM2S_SEL0=27, MM2S_SEL1=28
        EventType::DmaFinishedBd { channel } => match channel {
            0 => Some(25),
            1 => Some(26),
            2 => Some(27),
            3 => Some(28),
            _ => None,
        },
        // DMA finished task: S2MM_SEL0=29, S2MM_SEL1=30, MM2S_SEL0=31, MM2S_SEL1=32
        EventType::DmaFinishedTask { channel } => match channel {
            0 => Some(29),
            1 => Some(30),
            2 => Some(31),
            3 => Some(32),
            _ => None,
        },
        // DMA stalled lock: S2MM_SEL0=33, S2MM_SEL1=34, MM2S_SEL0=35, MM2S_SEL1=36
        EventType::DmaStalledLock { channel } => match channel {
            0 => Some(33),
            1 => Some(34),
            2 => Some(35),
            3 => Some(36),
            _ => None,
        },
        // DMA stream starvation: S2MM_SEL0=37, S2MM_SEL1=38, MM2S_SEL0=39(bp), MM2S_SEL1=40
        EventType::DmaStreamStarvation { channel } => match channel {
            0 => Some(37),
            1 => Some(38),
            2 => Some(39),
            3 => Some(40),
            _ => None,
        },
        // Lock acquire: LOCK_SEL0_ACQ_GE=47, stride 4 per lock
        EventType::LockAcquire { lock_id } => {
            if *lock_id <= 7 {
                Some(47 + (*lock_id as u8) * 4)
            } else {
                None
            }
        },
        // Lock release: LOCK_SEL0_REL=48, stride 4 per lock
        EventType::LockRelease { lock_id } => {
            if *lock_id <= 7 {
                Some(48 + (*lock_id as u8) * 4)
            } else {
                None
            }
        },
        _ => None,
    }
}

/// Map an emulator `EventType` to its AIE2 shim tile (PL module) hardware event ID.
///
/// Shim tiles have their own event namespace distinct from both compute tile
/// memory modules and MemTiles. DMA events are in a different ID range.
///
/// Event IDs from xaie_events_aieml.h (XAIEML_EVENTS_PL_*).
pub fn shim_event_to_hw_id(event: &EventType) -> Option<u8> {
    match event {
        // DMA start task: S2MM_0=14, S2MM_1=15, MM2S_0=16, MM2S_1=17
        EventType::DmaStartTask { channel } => match channel {
            0 => Some(14),
            1 => Some(15),
            2 => Some(16),
            3 => Some(17),
            _ => None,
        },
        // DMA finished BD: S2MM_0=18, S2MM_1=19, MM2S_0=20, MM2S_1=21
        EventType::DmaFinishedBd { channel } => match channel {
            0 => Some(18),
            1 => Some(19),
            2 => Some(20),
            3 => Some(21),
            _ => None,
        },
        // DMA finished task: S2MM_0=22, S2MM_1=23, MM2S_0=24, MM2S_1=25
        EventType::DmaFinishedTask { channel } => match channel {
            0 => Some(22),
            1 => Some(23),
            2 => Some(24),
            3 => Some(25),
            _ => None,
        },
        // DMA stalled lock: S2MM_0=26, S2MM_1=27, MM2S_0=28, MM2S_1=29
        EventType::DmaStalledLock { channel } => match channel {
            0 => Some(26),
            1 => Some(27),
            2 => Some(28),
            3 => Some(29),
            _ => None,
        },
        // DMA stream starvation: S2MM_0=30, S2MM_1=31
        // DMA stream backpressure: MM2S_0=32, MM2S_1=33
        EventType::DmaStreamStarvation { channel } => match channel {
            0 => Some(30),
            1 => Some(31),
            2 => Some(32),
            3 => Some(33),
            _ => None,
        },
        // Lock acquire: LOCK_0_ACQ_GE=40, stride 4 per lock (6 locks max)
        EventType::LockAcquire { lock_id } => {
            if *lock_id <= 5 {
                Some(40 + (*lock_id as u8) * 4)
            } else {
                None
            }
        },
        // Lock release: LOCK_0_REL=41, stride 4 per lock
        EventType::LockRelease { lock_id } => {
            if *lock_id <= 5 {
                Some(41 + (*lock_id as u8) * 4)
            } else {
                None
            }
        },
        _ => None,
    }
}

/// ShimTileEvent EDGE_DETECTION_EVENT_N hardware event ID.
/// Shim PL module: IDs 11-12 for detectors 0-1.
pub fn shim_edge_detection_event_hw_id(detector: u8) -> u8 {
    11 + detector
}

/// Hardware event IDs for stream switch port monitoring events.
///
/// Each event port (0-7) has four event types at stride 4:
/// PORT_IDLE, PORT_RUNNING, PORT_STALLED, PORT_TLAST.
///
/// | Module     | PORT_IDLE_0 | PORT_RUNNING_0 | PORT_STALLED_0 | PORT_TLAST_0 |
/// |------------|-------------|----------------|----------------|--------------|
/// | CoreEvent  | 74          | 75             | 76             | 77           |
/// | MemEvent   | (none)      | (none)         | (none)         | (none)       |
/// | MemTile    | 79          | 80             | 81             | 82           |
///
/// The physical port (master/slave + index) is configured separately via
/// Event Port Selection registers (0x3FF00/0x3FF04 or 0xB0F00/0xB0F04).
///
/// Source: mlir-aie `python/utils/trace/events/aie2.py`

pub fn core_port_idle_hw_id(event_port: u8) -> u8 {
    74 + (event_port * 4)
}

pub fn core_port_running_hw_id(event_port: u8) -> u8 {
    75 + (event_port * 4)
}

pub fn core_port_stalled_hw_id(event_port: u8) -> u8 {
    76 + (event_port * 4)
}

pub fn core_port_tlast_hw_id(event_port: u8) -> u8 {
    77 + (event_port * 4)
}

/// Hardware event ID for PORT_RUNNING_N in the memory module (compute tiles).
///
/// WARNING: MemEvent does NOT have PORT events. IDs 78+ in MemEvent are
/// CONFLICT_DM_BANK events. This function exists for symmetry but should
pub fn memtile_port_idle_hw_id(event_port: u8) -> u8 {
    79 + (event_port * 4)
}

pub fn memtile_port_running_hw_id(event_port: u8) -> u8 {
    80 + (event_port * 4)
}

pub fn memtile_port_stalled_hw_id(event_port: u8) -> u8 {
    81 + (event_port * 4)
}

pub fn memtile_port_tlast_hw_id(event_port: u8) -> u8 {
    82 + (event_port * 4)
}

/// Shim tile port event IDs (ShimTileEvent).
/// Base: PORT_IDLE_0=77, stride 4.
pub fn shim_port_idle_hw_id(event_port: u8) -> u8 {
    77 + (event_port * 4)
}

pub fn shim_port_running_hw_id(event_port: u8) -> u8 {
    78 + (event_port * 4)
}

pub fn shim_port_stalled_hw_id(event_port: u8) -> u8 {
    79 + (event_port * 4)
}

pub fn shim_port_tlast_hw_id(event_port: u8) -> u8 {
    80 + (event_port * 4)
}

// ============================================================================
// Memory bank conflict event IDs
// ============================================================================
//
// CONFLICT_DM_BANK events fire when two agents (core + DMA, or two DMA
// channels) access the same memory bank in the same cycle. The hardware
// generates one event per conflicting bank per cycle.
//
// Source: xaie_events_aieml.h

/// MemEvent CONFLICT_DM_BANK_N hardware event ID.
/// Compute tile memory module: IDs 77-84 for banks 0-7.
pub fn mem_conflict_dm_bank_hw_id(bank: u8) -> u8 {
    77 + bank
}

/// MemTileEvent CONFLICT_DM_BANK_N hardware event ID.
/// MemTile memory module: IDs 112-127 for banks 0-15.
pub fn memtile_conflict_dm_bank_hw_id(bank: u8) -> u8 {
    112 + bank
}

// ============================================================================
// Edge detection event IDs
// ============================================================================
//
// Edge detection monitors event signal transitions (rising/falling edges).
// Each module has two independent edge detectors that produce derived events.
//
// Source: xaie_events_aieml.h

/// CoreEvent EDGE_DETECTION_EVENT_N hardware event ID.
/// Core module: IDs 13-14 for detectors 0-1.
pub fn core_edge_detection_event_hw_id(detector: u8) -> u8 {
    13 + detector
}

/// MemEvent EDGE_DETECTION_EVENT_N hardware event ID.
/// Compute tile memory module: IDs 11-12 for detectors 0-1.
pub fn mem_edge_detection_event_hw_id(detector: u8) -> u8 {
    11 + detector
}

/// MemTileEvent EDGE_DETECTION_EVENT_N hardware event ID.
/// MemTile memory module: IDs 13-14 for detectors 0-1.
pub fn memtile_edge_detection_event_hw_id(detector: u8) -> u8 {
    13 + detector
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Hardware event ID mapping tests --

    #[test]
    fn test_core_event_instr_event_ids() {
        // INSTR_EVENT_0=33, INSTR_EVENT_1=34
        assert_eq!(core_event_to_hw_id(&EventType::InstrEvent { pc: 0, id: 0 }), Some(33));
        assert_eq!(core_event_to_hw_id(&EventType::InstrEvent { pc: 0, id: 1 }), Some(34));
        // id >= 2 is AIE2P only
        assert_eq!(core_event_to_hw_id(&EventType::InstrEvent { pc: 0, id: 2 }), None);
        assert_eq!(core_event_to_hw_id(&EventType::InstrEvent { pc: 0, id: 3 }), None);
    }

    #[test]
    fn test_core_event_ids() {
        // Verify key core event IDs match mlir-aie aie2.py
        assert_eq!(core_event_to_hw_id(&EventType::InstrVector { pc: 0 }), Some(37));
        assert_eq!(core_event_to_hw_id(&EventType::InstrLoad { pc: 0 }), Some(38));
        assert_eq!(core_event_to_hw_id(&EventType::InstrStore { pc: 0 }), Some(39));
        assert_eq!(core_event_to_hw_id(&EventType::InstrCall { pc: 0 }), Some(35));
        assert_eq!(core_event_to_hw_id(&EventType::InstrReturn { pc: 0 }), Some(36));
        assert_eq!(core_event_to_hw_id(&EventType::MemoryStall { cycles: 1 }), Some(23));
        assert_eq!(core_event_to_hw_id(&EventType::LockStall { cycles: 1 }), Some(26));
        assert_eq!(core_event_to_hw_id(&EventType::StreamStall { cycles: 1 }), Some(24));
        assert_eq!(core_event_to_hw_id(&EventType::CoreActive), Some(28));
        assert_eq!(core_event_to_hw_id(&EventType::CoreDisabled), Some(29));
        assert_eq!(core_event_to_hw_id(&EventType::InstrStreamGet { pc: 0 }), Some(40));
        assert_eq!(core_event_to_hw_id(&EventType::InstrStreamPut { pc: 0 }), Some(41));
        assert_eq!(core_event_to_hw_id(&EventType::InstrLockAcquireReq { pc: 0 }), Some(44));
        assert_eq!(core_event_to_hw_id(&EventType::InstrLockReleaseReq { pc: 0 }), Some(45));
    }

    #[test]
    fn test_core_event_branch_has_no_hw_id() {
        assert_eq!(core_event_to_hw_id(&EventType::BranchTaken { from_pc: 0, to_pc: 0 }), None);
    }

    #[test]
    fn test_mem_event_dma_ids() {
        // DMA start task: S2MM_0=19, S2MM_1=20, MM2S_0=21, MM2S_1=22
        assert_eq!(mem_event_to_hw_id(&EventType::DmaStartTask { channel: 0 }), Some(19));
        assert_eq!(mem_event_to_hw_id(&EventType::DmaStartTask { channel: 1 }), Some(20));
        assert_eq!(mem_event_to_hw_id(&EventType::DmaStartTask { channel: 2 }), Some(21));
        assert_eq!(mem_event_to_hw_id(&EventType::DmaStartTask { channel: 3 }), Some(22));

        // DMA finished BD: S2MM_0=23, S2MM_1=24, MM2S_0=25, MM2S_1=26
        assert_eq!(mem_event_to_hw_id(&EventType::DmaFinishedBd { channel: 0 }), Some(23));
        assert_eq!(mem_event_to_hw_id(&EventType::DmaFinishedBd { channel: 3 }), Some(26));

        // DMA finished task: S2MM_0=27, ..., MM2S_1=30
        assert_eq!(mem_event_to_hw_id(&EventType::DmaFinishedTask { channel: 0 }), Some(27));
        assert_eq!(mem_event_to_hw_id(&EventType::DmaFinishedTask { channel: 3 }), Some(30));
    }

    #[test]
    fn test_mem_event_lock_ids() {
        // Lock 0: acq_ge=45, rel=46
        assert_eq!(mem_event_to_hw_id(&EventType::LockAcquire { lock_id: 0 }), Some(45));
        assert_eq!(mem_event_to_hw_id(&EventType::LockRelease { lock_id: 0 }), Some(46));

        // Lock 3: acq_ge=45+12=57, rel=46+12=58
        assert_eq!(mem_event_to_hw_id(&EventType::LockAcquire { lock_id: 3 }), Some(57));
        assert_eq!(mem_event_to_hw_id(&EventType::LockRelease { lock_id: 3 }), Some(58));

        // Lock 7: acq_ge=45+28=73, rel=46+28=74
        assert_eq!(mem_event_to_hw_id(&EventType::LockAcquire { lock_id: 7 }), Some(73));
        assert_eq!(mem_event_to_hw_id(&EventType::LockRelease { lock_id: 7 }), Some(74));

        // Lock 8+: out of range
        assert_eq!(mem_event_to_hw_id(&EventType::LockAcquire { lock_id: 8 }), None);
    }

    #[test]
    fn test_core_events_not_in_mem_space() {
        // Core events should return None from mem_event_to_hw_id
        assert_eq!(mem_event_to_hw_id(&EventType::InstrVector { pc: 0 }), None);
        assert_eq!(mem_event_to_hw_id(&EventType::CoreActive), None);
    }

    #[test]
    fn test_mem_events_not_in_core_space() {
        // Memory events should return None from core_event_to_hw_id
        assert_eq!(core_event_to_hw_id(&EventType::DmaStartTask { channel: 0 }), None);
        assert_eq!(core_event_to_hw_id(&EventType::LockAcquire { lock_id: 0 }), None);
    }

    // -- MemTile event ID tests --

    #[test]
    fn test_memtile_dma_ids_offset_from_compute() {
        // MemTile DMA events are offset +2 from compute tile memory module.
        // Compute: S2MM_0_START_TASK=19, MemTile: S2MM_SEL0_START_TASK=21
        assert_eq!(memtile_event_to_hw_id(&EventType::DmaStartTask { channel: 0 }), Some(21));
        assert_eq!(memtile_event_to_hw_id(&EventType::DmaStartTask { channel: 1 }), Some(22));
        assert_eq!(memtile_event_to_hw_id(&EventType::DmaStartTask { channel: 2 }), Some(23));
        assert_eq!(memtile_event_to_hw_id(&EventType::DmaStartTask { channel: 3 }), Some(24));
    }

    #[test]
    fn test_memtile_dma_finished_ids() {
        assert_eq!(memtile_event_to_hw_id(&EventType::DmaFinishedBd { channel: 0 }), Some(25));
        assert_eq!(memtile_event_to_hw_id(&EventType::DmaFinishedBd { channel: 3 }), Some(28));
        assert_eq!(memtile_event_to_hw_id(&EventType::DmaFinishedTask { channel: 0 }), Some(29));
        assert_eq!(memtile_event_to_hw_id(&EventType::DmaFinishedTask { channel: 3 }), Some(32));
    }

    #[test]
    fn test_memtile_lock_ids_offset_from_compute() {
        // Compute: LOCK_SEL0_ACQ_GE=45, MemTile: LOCK_SEL0_ACQ_GE=47
        assert_eq!(memtile_event_to_hw_id(&EventType::LockAcquire { lock_id: 0 }), Some(47));
        assert_eq!(memtile_event_to_hw_id(&EventType::LockRelease { lock_id: 0 }), Some(48));
        // Lock 7: 47 + 28 = 75
        assert_eq!(memtile_event_to_hw_id(&EventType::LockAcquire { lock_id: 7 }), Some(75));
        assert_eq!(memtile_event_to_hw_id(&EventType::LockRelease { lock_id: 7 }), Some(76));
    }

    #[test]
    fn test_memtile_core_events_return_none() {
        assert_eq!(memtile_event_to_hw_id(&EventType::InstrVector { pc: 0 }), None);
        assert_eq!(memtile_event_to_hw_id(&EventType::CoreActive), None);
    }

    // -- Port event ID tests --
    //
    // Each event port has 4 IDs at stride 4: IDLE, RUNNING, STALLED, TLAST.

    #[test]
    fn test_core_port_event_ids_stride() {
        // CoreEvent: PORT_IDLE_0=74, PORT_RUNNING_0=75, PORT_STALLED_0=76, PORT_TLAST_0=77
        assert_eq!(core_port_idle_hw_id(0), 74);
        assert_eq!(core_port_running_hw_id(0), 75);
        assert_eq!(core_port_stalled_hw_id(0), 76);
        assert_eq!(core_port_tlast_hw_id(0), 77);

        // Port 1: +4
        assert_eq!(core_port_idle_hw_id(1), 78);
        assert_eq!(core_port_running_hw_id(1), 79);
        assert_eq!(core_port_stalled_hw_id(1), 80);
        assert_eq!(core_port_tlast_hw_id(1), 81);

        // Port 7: +28
        assert_eq!(core_port_idle_hw_id(7), 102);
        assert_eq!(core_port_running_hw_id(7), 103);
        assert_eq!(core_port_stalled_hw_id(7), 104);
        assert_eq!(core_port_tlast_hw_id(7), 105);
    }

    #[test]
    fn test_shim_port_event_ids_stride() {
        // ShimTileEvent: PORT_IDLE_0=77, PORT_RUNNING_0=78, PORT_STALLED_0=79, PORT_TLAST_0=80
        assert_eq!(shim_port_idle_hw_id(0), 77);
        assert_eq!(shim_port_running_hw_id(0), 78);
        assert_eq!(shim_port_stalled_hw_id(0), 79);
        assert_eq!(shim_port_tlast_hw_id(0), 80);

        // Port 7: +28
        assert_eq!(shim_port_idle_hw_id(7), 105);
        assert_eq!(shim_port_running_hw_id(7), 106);
        assert_eq!(shim_port_stalled_hw_id(7), 107);
        assert_eq!(shim_port_tlast_hw_id(7), 108);
    }

    #[test]
    fn test_memtile_port_event_ids_stride() {
        // MemTileEvent: PORT_IDLE_0=79, PORT_RUNNING_0=80, PORT_STALLED_0=81, PORT_TLAST_0=82
        assert_eq!(memtile_port_idle_hw_id(0), 79);
        assert_eq!(memtile_port_running_hw_id(0), 80);
        assert_eq!(memtile_port_stalled_hw_id(0), 81);
        assert_eq!(memtile_port_tlast_hw_id(0), 82);

        // Port 1: +4
        assert_eq!(memtile_port_idle_hw_id(1), 83);
        assert_eq!(memtile_port_running_hw_id(1), 84);
        assert_eq!(memtile_port_stalled_hw_id(1), 85);
        assert_eq!(memtile_port_tlast_hw_id(1), 86);

        // Port 7: +28
        assert_eq!(memtile_port_idle_hw_id(7), 107);
        assert_eq!(memtile_port_running_hw_id(7), 108);
        assert_eq!(memtile_port_stalled_hw_id(7), 109);
        assert_eq!(memtile_port_tlast_hw_id(7), 110);
    }

    #[test]
    fn test_conflict_dm_bank_hw_ids() {
        // Compute tile MemEvent: GROUP_MEMORY_CONFLICT=76, BANK_0=77..BANK_7=84
        assert_eq!(mem_conflict_dm_bank_hw_id(0), 77);
        assert_eq!(mem_conflict_dm_bank_hw_id(7), 84);

        // MemTile: GROUP_MEMORY_CONFLICT=111, BANK_0=112..BANK_15=127
        assert_eq!(memtile_conflict_dm_bank_hw_id(0), 112);
        assert_eq!(memtile_conflict_dm_bank_hw_id(15), 127);
    }

    #[test]
    fn test_edge_detection_event_hw_ids() {
        // Core module: EDGE_DETECTION_EVENT_0=13, EVENT_1=14
        assert_eq!(core_edge_detection_event_hw_id(0), 13);
        assert_eq!(core_edge_detection_event_hw_id(1), 14);

        // Memory module: EDGE_DETECTION_EVENT_0=11, EVENT_1=12
        assert_eq!(mem_edge_detection_event_hw_id(0), 11);
        assert_eq!(mem_edge_detection_event_hw_id(1), 12);

        // MemTile: EDGE_DETECTION_EVENT_0=13, EVENT_1=14
        assert_eq!(memtile_edge_detection_event_hw_id(0), 13);
        assert_eq!(memtile_edge_detection_event_hw_id(1), 14);

        // Shim PL module: EDGE_DETECTION_EVENT_0=11, EVENT_1=12
        assert_eq!(shim_edge_detection_event_hw_id(0), 11);
        assert_eq!(shim_edge_detection_event_hw_id(1), 12);
    }

    #[test]
    #[cfg(feature = "tooling")]
    fn test_validate_trace_events_passes() {
        // Validate compiled-in tables against live mlir-aie (if available).
        let bridge = crate::integration::bridge::BridgePath::discover();
        if bridge.is_none() { return; }
        let warnings = validate_trace_events(&bridge.unwrap());
        assert!(warnings.is_ok(), "validation failed: {:?}", warnings.err());
        let warnings = warnings.unwrap();
        assert!(
            warnings.is_empty(),
            "trace event mismatches: {:?}",
            warnings,
        );
    }

    #[test]
    fn test_shim_event_to_hw_id() {
        use crate::trace::EventType;

        // DMA start task: S2MM_0=14, S2MM_1=15, MM2S_0=16, MM2S_1=17
        assert_eq!(shim_event_to_hw_id(&EventType::DmaStartTask { channel: 0 }), Some(14));
        assert_eq!(shim_event_to_hw_id(&EventType::DmaStartTask { channel: 1 }), Some(15));
        assert_eq!(shim_event_to_hw_id(&EventType::DmaStartTask { channel: 2 }), Some(16));
        assert_eq!(shim_event_to_hw_id(&EventType::DmaStartTask { channel: 3 }), Some(17));

        // DMA finished BD: S2MM_0=18, S2MM_1=19, MM2S_0=20, MM2S_1=21
        assert_eq!(shim_event_to_hw_id(&EventType::DmaFinishedBd { channel: 0 }), Some(18));
        assert_eq!(shim_event_to_hw_id(&EventType::DmaFinishedBd { channel: 3 }), Some(21));

        // DMA finished task: S2MM_0=22, S2MM_1=23, MM2S_0=24, MM2S_1=25
        assert_eq!(shim_event_to_hw_id(&EventType::DmaFinishedTask { channel: 0 }), Some(22));
        assert_eq!(shim_event_to_hw_id(&EventType::DmaFinishedTask { channel: 3 }), Some(25));

        // DMA stalled lock: S2MM_0=26, S2MM_1=27, MM2S_0=28, MM2S_1=29
        assert_eq!(shim_event_to_hw_id(&EventType::DmaStalledLock { channel: 0 }), Some(26));
        assert_eq!(shim_event_to_hw_id(&EventType::DmaStalledLock { channel: 3 }), Some(29));

        // DMA stream starvation/backpressure: S2MM_0=30, S2MM_1=31, MM2S_0=32, MM2S_1=33
        assert_eq!(shim_event_to_hw_id(&EventType::DmaStreamStarvation { channel: 0 }), Some(30));
        assert_eq!(shim_event_to_hw_id(&EventType::DmaStreamStarvation { channel: 3 }), Some(33));

        // Lock events: 6 locks, stride 4, ACQ_GE base=40, REL base=41
        assert_eq!(shim_event_to_hw_id(&EventType::LockAcquire { lock_id: 0 }), Some(40));
        assert_eq!(shim_event_to_hw_id(&EventType::LockAcquire { lock_id: 5 }), Some(60));
        assert_eq!(shim_event_to_hw_id(&EventType::LockAcquire { lock_id: 6 }), None); // Only 6 locks
        assert_eq!(shim_event_to_hw_id(&EventType::LockRelease { lock_id: 0 }), Some(41));
        assert_eq!(shim_event_to_hw_id(&EventType::LockRelease { lock_id: 5 }), Some(61));
    }
}
