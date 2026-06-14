//! Hardware trace event ID mappings and trace infrastructure.
//!
//! Maps emulator `EventType` values to AIE2 hardware event IDs used by the
//! binary trace unit (`TraceUnit`). The emulator produces the same binary
//! trace format as real NPU hardware, decoded by mlir-aie's `parse.py`.
//!
//! Also provides build-time generated event code tables, a GUI trace store,
//! and VCD-to-Perfetto conversion for aiesimulator integration.

pub mod compare;
pub mod compare_mode2;
pub mod mode2_decode;
pub mod stages;
pub mod stochastic;
pub mod store;
pub mod vcd;

/// Trace event codes from mlir-aie's canonical enums.
///
/// Provides per-tile-type event code constants and name lookup functions.
/// Generated at build time by `xdna-archspec/build.rs` from
/// `tools/mlir-aie-bridge.py trace-events`. Falls back to stubs if
/// mlir-aie is not available.
pub mod event_codes {
    pub use xdna_archspec::aie2::trace_events::*;
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
pub fn validate_trace_events(bridge: &crate::integration::bridge::BridgePath) -> Result<Vec<String>, String> {
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
                warnings.push(format!("CoreEvent::{} compiled={} mlir-aie={}", name, expected, actual,));
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
                warnings.push(format!("MemEvent::{} compiled={} mlir-aie={}", name, expected, actual,));
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
        EventType::InstrEvent { id: 0, .. } => Some(33),
        EventType::InstrEvent { id: 1, .. } => Some(34),
        EventType::InstrEvent { .. } => None, // id 2/3 are AIE2P only
        // Stall events (Core module)
        EventType::MemoryStall { .. } => Some(23),
        EventType::StreamStall { .. } => Some(24),
        EventType::LockStall { .. } => Some(26),
        // Core status
        EventType::CoreActive => Some(28),
        EventType::CoreDisabled => Some(29),
        // Program flow events
        EventType::InstrCall { .. } => Some(35),
        EventType::InstrReturn { .. } => Some(36),
        EventType::InstrVector { .. } => Some(37),
        EventType::InstrLoad { .. } => Some(38),
        EventType::InstrStore { .. } => Some(39),
        EventType::InstrStreamGet { .. } => Some(40),
        EventType::InstrStreamPut { .. } => Some(41),
        EventType::InstrLockAcquireReq { .. } => Some(44),
        EventType::InstrLockReleaseReq { .. } => Some(45),
        // Branch is an emulator-internal event, no hardware equivalent
        EventType::BranchTaken { .. } => None,
        // LockStallLevel is a level-EDGE event routed via core_level_edge /
        // set_event_level, not the pulse notify_event path -- return None here
        // so the coordinator's drain falls through to the level route.
        EventType::LockStallLevel { .. } => None,
        // Memory module events don't have core event IDs
        _ => None,
    }
}

/// Map a core-module LEVEL-edge event to its `(hw_id, active)` for the trace
/// unit's held-level path (`set_event_level`). Returns `None` for events that
/// are not level edges -- the caller routes those through
/// `core_event_to_hw_id` + `notify_event` (the pulse path) instead.
///
/// hw_id 26 mirrors `core_event_to_hw_id(LockStall)` -- both are the LOCK_STALL
/// event; the difference is pulse (notify_event) vs level (set_event_level).
pub(crate) fn core_level_edge(event: &EventType) -> Option<(u8, bool)> {
    match event {
        EventType::LockStallLevel { active } => Some((26, *active)),
        EventType::StreamStallLevel { active } => Some((24, *active)),
        // CASCADE_STALL (25) is distinct from STREAM_STALL (24): cascade
        // backpressure has its own event enum (xaie_events_aieml.h:60).
        EventType::CascadeStallLevel { active } => Some((25, *active)),
        _ => None,
    }
}

/// For a memory-module DMA trace event, return its held-level polarity:
/// `Some(true)` on the asserting edge, `Some(false)` on the deasserting edge.
/// Returns `None` for one-cycle pulse events (start/finished task, finished BD),
/// which route through the pulse path. The hw_id is resolved separately by the
/// module-specific `*_event_to_hw_id` mapping; this only carries the edge.
pub(crate) fn dma_level_active(event: &EventType) -> Option<bool> {
    match event {
        EventType::DmaStalledLock { active, .. } => Some(*active),
        EventType::DmaStreamStarvation { active, .. } => Some(*active),
        _ => None,
    }
}

/// Extract the PC field from an EventType variant, if it carries one.
///
/// Returns `Some(pc)` for instruction-class events (InstrVector, InstrLoad,
/// InstrStore, InstrCall, InstrReturn, InstrLockAcquireReq, InstrLockReleaseReq,
/// InstrStreamGet, InstrStreamPut, InstrEvent). These are the 10 variants that
/// carry an actual program counter at the point the instruction retired.
///
/// Returns `None` for core-state events (CoreActive, CoreDisabled), stall events
/// (MemoryStall, LockStall, StreamStall), memory-module events (DMA, lock, port),
/// branch events, and any other variant whose PC is either not meaningful or not
/// directly available at notify time.
pub(crate) fn event_pc(event: &EventType) -> Option<u32> {
    match event {
        EventType::InstrVector { pc }
        | EventType::InstrLoad { pc }
        | EventType::InstrStore { pc }
        | EventType::InstrCall { pc }
        | EventType::InstrReturn { pc }
        | EventType::InstrLockAcquireReq { pc }
        | EventType::InstrLockReleaseReq { pc }
        | EventType::InstrStreamGet { pc }
        | EventType::InstrStreamPut { pc } => Some(*pc),
        EventType::InstrEvent { pc, .. } => Some(*pc),
        // Stall variants carry an optional PC; emission sites that can
        // snapshot the stalled instruction's PC pass Some, others pass None.
        // Mode-1 trace encoding uses this so the encoded frame's PC matches
        // the issuing instruction (and HW's per-instruction-PC view).
        EventType::MemoryStall { pc, .. }
        | EventType::LockStall { pc, .. }
        | EventType::StreamStall { pc, .. } => *pc,
        // If you add a new InstrXxx variant carrying `pc: u32`, add a matching
        // arm above and extend `event_pc_extracts_from_instruction_variants` --
        // otherwise the new variant will silently fall through to None and
        // PC threading will break for it.
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
        EventType::DmaStalledLock { channel, .. } => match channel {
            0 => Some(31),
            1 => Some(32),
            2 => Some(33),
            3 => Some(34),
            _ => None,
        },
        // DMA stream starvation: S2MM_0=35, S2MM_1=36, MM2S_0=37(backpressure), MM2S_1=38
        EventType::DmaStreamStarvation { channel, .. } => match channel {
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
        }
        // Lock release: LOCK_0_REL=46, stride 4 per lock
        // Lock 0 rel=46, Lock 1 rel=50, ..., Lock 7 rel=74
        EventType::LockRelease { lock_id } => {
            if *lock_id <= 7 {
                Some(46 + (*lock_id as u8) * 4)
            } else {
                None
            }
        }
        // Core events don't have memory module IDs
        _ => None,
    }
}

/// MemTile DMA Event Channel Selection register state (offset 0xA06A0).
///
/// Per AM020 / aie-rt `xaiemlgbl_params.h`, MemTiles have only 4 DMA event
/// signals broadcast to the trace network: S2MM_SEL0, S2MM_SEL1, MM2S_SEL0,
/// MM2S_SEL1. The DMA_Event_Channel_Selection register (0xA06A0) selects
/// which physical DMA channel feeds each SEL slot:
///
/// | Bits   | Field             | Selects which physical channel fires SEL_N |
/// |--------|-------------------|--------------------------------------------|
/// | 2:0    | S2MM_Sel0_Channel | 0..5 (default 0)                           |
/// | 10:8   | S2MM_Sel1_Channel | 0..5 (default 0)                           |
/// | 18:16  | MM2S_Sel0_Channel | 0..5 (default 0)                           |
/// | 26:24  | MM2S_Sel1_Channel | 0..5 (default 0)                           |
///
/// At reset, all SEL slots point at physical channel 0 -- so by default,
/// channel 0 fires both SEL0 and SEL1 events for its direction, while
/// channels 1-5 fire nothing. To capture activity on a non-zero channel,
/// software must program 0xA06A0 to redirect a SEL slot at it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemtileDmaEventSel {
    pub s2mm_sel0: u8,
    pub s2mm_sel1: u8,
    pub mm2s_sel0: u8,
    pub mm2s_sel1: u8,
}

impl MemtileDmaEventSel {
    /// Decode the 32-bit register value into per-SEL channel selectors.
    ///
    /// Field layout per AM020 / aie-rt `xaiemlgbl_params.h`:
    ///   S2MM_SEL0 = bits 2:0, S2MM_SEL1 = bits 10:8,
    ///   MM2S_SEL0 = bits 18:16, MM2S_SEL1 = bits 26:24.
    /// All fields are 3 bits wide (memtile has 6 channels per direction; the
    /// 7th value is reserved).
    pub fn from_register(value: u32) -> Self {
        Self {
            s2mm_sel0: (value & 0x7) as u8,
            s2mm_sel1: ((value >> 8) & 0x7) as u8,
            mm2s_sel0: ((value >> 16) & 0x7) as u8,
            mm2s_sel1: ((value >> 24) & 0x7) as u8,
        }
    }
}

/// Number of S2MM channels on a MemTile (per AM020 / mlir-aie device model).
/// Used to decode the EMU's flat channel index (0..5 = S2MM, 6..11 = MM2S).
const MEMTILE_S2MM_CHANNELS: u8 = 6;

/// Map an emulator `EventType` to its AIE2 MemTile hardware event ID(s).
///
/// MemTiles use a different event ID namespace than compute tile memory modules.
/// DMA events are offset by +2 from compute tile IDs, and use SEL0/SEL1 naming
/// for the two channel groups.
///
/// `sel` is the current value of the DMA_Event_Channel_Selection register
/// (0xA06A0). For DMA events on flat channel `N`, the SEL slot whose
/// configured channel equals `N` fires the corresponding event ID. If no
/// SEL slot is targeted at this channel, the event is dropped (returns
/// `[None, None]`). Both SEL slots can be aimed at the same channel
/// (e.g., both default to 0) -- in that case both event IDs fire.
///
/// Returns up to two HW event IDs in a fixed-size array. Non-DMA events
/// (locks, etc.) always produce at most one entry. Unused slots are `None`.
///
/// Event IDs from mlir-aie `python/utils/trace/events/aie2.py` MemTileEvent.
pub fn memtile_event_to_hw_ids(event: &EventType, sel: MemtileDmaEventSel) -> [Option<u8>; 2] {
    let dma_ids = |channel: u8, s2mm_base: u8, mm2s_base: u8| -> [Option<u8>; 2] {
        if channel < MEMTILE_S2MM_CHANNELS {
            // S2MM direction: flat channel = per-direction channel.
            let phys = channel;
            [(phys == sel.s2mm_sel0).then_some(s2mm_base), (phys == sel.s2mm_sel1).then_some(s2mm_base + 1)]
        } else {
            // MM2S direction: subtract S2MM count to get per-direction channel.
            let phys = channel - MEMTILE_S2MM_CHANNELS;
            [(phys == sel.mm2s_sel0).then_some(mm2s_base), (phys == sel.mm2s_sel1).then_some(mm2s_base + 1)]
        }
    };
    match event {
        // DMA start task: S2MM_SEL0=21, S2MM_SEL1=22, MM2S_SEL0=23, MM2S_SEL1=24
        EventType::DmaStartTask { channel } => dma_ids(*channel, 21, 23),
        // DMA finished BD: S2MM_SEL0=25, S2MM_SEL1=26, MM2S_SEL0=27, MM2S_SEL1=28
        EventType::DmaFinishedBd { channel } => dma_ids(*channel, 25, 27),
        // DMA finished task: S2MM_SEL0=29, S2MM_SEL1=30, MM2S_SEL0=31, MM2S_SEL1=32
        EventType::DmaFinishedTask { channel } => dma_ids(*channel, 29, 31),
        // DMA stalled lock: S2MM_SEL0=33, S2MM_SEL1=34, MM2S_SEL0=35, MM2S_SEL1=36
        EventType::DmaStalledLock { channel, .. } => dma_ids(*channel, 33, 35),
        // DMA stream starvation: S2MM_SEL0=37, S2MM_SEL1=38, MM2S_SEL0=39(bp), MM2S_SEL1=40
        EventType::DmaStreamStarvation { channel, .. } => dma_ids(*channel, 37, 39),
        // Lock acquire: LOCK_SEL0_ACQ_GE=47, stride 4 per lock.
        // Lock events are not gated by the DMA SEL register.
        EventType::LockAcquire { lock_id } if *lock_id <= 7 => [Some(47 + (*lock_id as u8) * 4), None],
        // Lock release: LOCK_SEL0_REL=48, stride 4 per lock.
        EventType::LockRelease { lock_id } if *lock_id <= 7 => [Some(48 + (*lock_id as u8) * 4), None],
        _ => [None, None],
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
        EventType::DmaStalledLock { channel, .. } => match channel {
            0 => Some(26),
            1 => Some(27),
            2 => Some(28),
            3 => Some(29),
            _ => None,
        },
        // DMA stream starvation: S2MM_0=30, S2MM_1=31
        // DMA stream backpressure: MM2S_0=32, MM2S_1=33
        EventType::DmaStreamStarvation { channel, .. } => match channel {
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
        }
        // Lock release: LOCK_0_REL=41, stride 4 per lock
        EventType::LockRelease { lock_id } => {
            if *lock_id <= 5 {
                Some(41 + (*lock_id as u8) * 4)
            } else {
                None
            }
        }
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
        assert_eq!(core_event_to_hw_id(&EventType::MemoryStall { cycles: 1, pc: None }), Some(23));
        assert_eq!(core_event_to_hw_id(&EventType::LockStall { cycles: 1, pc: None }), Some(26));
        assert_eq!(core_event_to_hw_id(&EventType::StreamStall { cycles: 1, pc: None }), Some(24));
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

    /// Helper: SEL register state with each SEL slot pointing at a distinct
    /// channel. Convenient for tests that want every channel 0..3 to fire
    /// exactly one SEL event ID.
    fn sel_distinct_4ch() -> MemtileDmaEventSel {
        MemtileDmaEventSel { s2mm_sel0: 0, s2mm_sel1: 1, mm2s_sel0: 0, mm2s_sel1: 1 }
    }

    #[test]
    fn test_memtile_dma_ids_with_sel_register() {
        // With SEL programmed for distinct channels (S2MM_SEL0=0, S2MM_SEL1=1,
        // MM2S_SEL0=0, MM2S_SEL1=1), each S2MM channel 0/1 and MM2S channel
        // 0/1 fires exactly one event ID.
        let sel = sel_distinct_4ch();
        // S2MM channels are flat 0..5; MM2S channels are flat 6..11.
        // S2MM ch0 -> SEL0 (id=21), S2MM ch1 -> SEL1 (id=22).
        assert_eq!(memtile_event_to_hw_ids(&EventType::DmaStartTask { channel: 0 }, sel), [Some(21), None]);
        assert_eq!(memtile_event_to_hw_ids(&EventType::DmaStartTask { channel: 1 }, sel), [None, Some(22)]);
        // MM2S ch0 -> SEL0 (id=23), MM2S ch1 -> SEL1 (id=24).
        assert_eq!(memtile_event_to_hw_ids(&EventType::DmaStartTask { channel: 6 }, sel), [Some(23), None]);
        assert_eq!(memtile_event_to_hw_ids(&EventType::DmaStartTask { channel: 7 }, sel), [None, Some(24)]);
    }

    #[test]
    fn test_memtile_dma_default_sel_fires_both_slots_on_ch0() {
        // Default SEL (all zero) means both SEL0 and SEL1 point at channel 0.
        // A single channel-0 event fires both event IDs simultaneously --
        // matching real hardware where the broadcast network sees both lines.
        let sel = MemtileDmaEventSel::default();
        assert_eq!(
            memtile_event_to_hw_ids(&EventType::DmaStartTask { channel: 0 }, sel),
            [Some(21), Some(22)]
        );
        // Channel 1 fires nothing because no SEL slot is aimed at it.
        assert_eq!(memtile_event_to_hw_ids(&EventType::DmaStartTask { channel: 1 }, sel), [None, None]);
        // MM2S channel 0 (flat=6) likewise fires both MM2S SEL slots.
        assert_eq!(
            memtile_event_to_hw_ids(&EventType::DmaStartTask { channel: 6 }, sel),
            [Some(23), Some(24)]
        );
        // MM2S channel 1 (flat=7) fires nothing.
        assert_eq!(memtile_event_to_hw_ids(&EventType::DmaStartTask { channel: 7 }, sel), [None, None]);
    }

    #[test]
    fn test_memtile_dma_high_channels_supported() {
        // Channels 4-5 (S2MM) and 10-11 (flat, = MM2S 4-5) must work when
        // a SEL slot points at them. Pre-fix code silently dropped these.
        let sel = MemtileDmaEventSel { s2mm_sel0: 5, s2mm_sel1: 4, mm2s_sel0: 5, mm2s_sel1: 4 };
        assert_eq!(memtile_event_to_hw_ids(&EventType::DmaStartTask { channel: 5 }, sel), [Some(21), None]);
        assert_eq!(memtile_event_to_hw_ids(&EventType::DmaStartTask { channel: 4 }, sel), [None, Some(22)]);
        // Flat 11 = MM2S ch5; flat 10 = MM2S ch4.
        assert_eq!(memtile_event_to_hw_ids(&EventType::DmaStartTask { channel: 11 }, sel), [Some(23), None]);
        assert_eq!(memtile_event_to_hw_ids(&EventType::DmaStartTask { channel: 10 }, sel), [None, Some(24)]);
    }

    #[test]
    fn test_memtile_dma_finished_ids_with_sel() {
        let sel = sel_distinct_4ch();
        assert_eq!(memtile_event_to_hw_ids(&EventType::DmaFinishedBd { channel: 0 }, sel), [Some(25), None]);
        assert_eq!(memtile_event_to_hw_ids(&EventType::DmaFinishedBd { channel: 1 }, sel), [None, Some(26)]);
        assert_eq!(
            memtile_event_to_hw_ids(&EventType::DmaFinishedTask { channel: 0 }, sel),
            [Some(29), None]
        );
        assert_eq!(
            memtile_event_to_hw_ids(&EventType::DmaFinishedTask { channel: 7 }, sel),
            [None, Some(32)]
        );
    }

    #[test]
    fn test_memtile_lock_ids_offset_from_compute() {
        // Lock events are not gated by the DMA SEL register.
        let sel = MemtileDmaEventSel::default();
        // Compute: LOCK_SEL0_ACQ_GE=45, MemTile: LOCK_SEL0_ACQ_GE=47
        assert_eq!(memtile_event_to_hw_ids(&EventType::LockAcquire { lock_id: 0 }, sel), [Some(47), None]);
        assert_eq!(memtile_event_to_hw_ids(&EventType::LockRelease { lock_id: 0 }, sel), [Some(48), None]);
        // Lock 7: 47 + 28 = 75
        assert_eq!(memtile_event_to_hw_ids(&EventType::LockAcquire { lock_id: 7 }, sel), [Some(75), None]);
        assert_eq!(memtile_event_to_hw_ids(&EventType::LockRelease { lock_id: 7 }, sel), [Some(76), None]);
    }

    #[test]
    fn test_memtile_core_events_return_none() {
        let sel = MemtileDmaEventSel::default();
        assert_eq!(memtile_event_to_hw_ids(&EventType::InstrVector { pc: 0 }, sel), [None, None]);
        assert_eq!(memtile_event_to_hw_ids(&EventType::CoreActive, sel), [None, None]);
    }

    #[test]
    fn test_memtile_dma_event_sel_decode() {
        // S2MM_SEL0=3 (bits 2:0), S2MM_SEL1=5 (bits 10:8),
        // MM2S_SEL0=2 (bits 18:16), MM2S_SEL1=4 (bits 26:24)
        let raw = 0u32 | 3 | (5 << 8) | (2 << 16) | (4 << 24);
        let sel = MemtileDmaEventSel::from_register(raw);
        assert_eq!(sel, MemtileDmaEventSel { s2mm_sel0: 3, s2mm_sel1: 5, mm2s_sel0: 2, mm2s_sel1: 4 });
        // 3-bit fields ignore high bits.
        let sel = MemtileDmaEventSel::from_register(0xFFFF_FFFF);
        assert_eq!(sel, MemtileDmaEventSel { s2mm_sel0: 7, s2mm_sel1: 7, mm2s_sel0: 7, mm2s_sel1: 7 });
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
        if bridge.is_none() {
            return;
        }
        let warnings = validate_trace_events(&bridge.unwrap());
        assert!(warnings.is_ok(), "validation failed: {:?}", warnings.err());
        let warnings = warnings.unwrap();
        assert!(warnings.is_empty(), "trace event mismatches: {:?}", warnings,);
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
        assert_eq!(shim_event_to_hw_id(&EventType::DmaStalledLock { channel: 0, active: true }), Some(26));
        assert_eq!(shim_event_to_hw_id(&EventType::DmaStalledLock { channel: 3, active: true }), Some(29));

        // DMA stream starvation/backpressure: S2MM_0=30, S2MM_1=31, MM2S_0=32, MM2S_1=33
        assert_eq!(
            shim_event_to_hw_id(&EventType::DmaStreamStarvation { channel: 0, active: true }),
            Some(30)
        );
        assert_eq!(
            shim_event_to_hw_id(&EventType::DmaStreamStarvation { channel: 3, active: true }),
            Some(33)
        );

        // Lock events: 6 locks, stride 4, ACQ_GE base=40, REL base=41
        assert_eq!(shim_event_to_hw_id(&EventType::LockAcquire { lock_id: 0 }), Some(40));
        assert_eq!(shim_event_to_hw_id(&EventType::LockAcquire { lock_id: 5 }), Some(60));
        assert_eq!(shim_event_to_hw_id(&EventType::LockAcquire { lock_id: 6 }), None); // Only 6 locks
        assert_eq!(shim_event_to_hw_id(&EventType::LockRelease { lock_id: 0 }), Some(41));
        assert_eq!(shim_event_to_hw_id(&EventType::LockRelease { lock_id: 5 }), Some(61));
    }

    #[test]
    fn dma_level_active_classifies_stall_starvation_edges() {
        // Held-level DMA events carry the assert/deassert polarity...
        assert_eq!(dma_level_active(&EventType::DmaStalledLock { channel: 0, active: true }), Some(true));
        assert_eq!(dma_level_active(&EventType::DmaStalledLock { channel: 1, active: false }), Some(false));
        assert_eq!(
            dma_level_active(&EventType::DmaStreamStarvation { channel: 0, active: true }),
            Some(true)
        );
        assert_eq!(
            dma_level_active(&EventType::DmaStreamStarvation { channel: 1, active: false }),
            Some(false)
        );
        // ...while one-cycle DMA pulses return None (routed through notify_event).
        assert_eq!(dma_level_active(&EventType::DmaStartTask { channel: 0 }), None);
        assert_eq!(dma_level_active(&EventType::DmaFinishedBd { channel: 0 }), None);
        assert_eq!(dma_level_active(&EventType::DmaFinishedTask { channel: 0 }), None);
    }

    #[test]
    fn core_level_edge_classifies_stall_family() {
        // The core-module stall family is held LEVEL: each carries assert/deassert
        // polarity and routes through set_event_level, NOT the pulse notify_event
        // path. hw_ids derive from xaie_events_aieml.h (AIE2 core module):
        // MEMORY_STALL=23, STREAM_STALL=24, CASCADE_STALL=25, LOCK_STALL=26.
        assert_eq!(core_level_edge(&EventType::LockStallLevel { active: true }), Some((26, true)));
        assert_eq!(core_level_edge(&EventType::LockStallLevel { active: false }), Some((26, false)));
        assert_eq!(core_level_edge(&EventType::StreamStallLevel { active: true }), Some((24, true)));
        assert_eq!(core_level_edge(&EventType::StreamStallLevel { active: false }), Some((24, false)));
        // CASCADE_STALL (25) is a DISTINCT event from STREAM_STALL (24): cascade
        // backpressure (SCD-read / MCD-write stalls) is its own enum, not folded
        // into the stream stall. See xaie_events_aieml.h:60.
        assert_eq!(core_level_edge(&EventType::CascadeStallLevel { active: true }), Some((25, true)));
        assert_eq!(core_level_edge(&EventType::CascadeStallLevel { active: false }), Some((25, false)));
        // Pulse / non-level events return None (routed via core_event_to_hw_id).
        assert_eq!(core_level_edge(&EventType::InstrLoad { pc: 0 }), None);
        assert_eq!(core_level_edge(&EventType::MemoryStall { cycles: 1, pc: None }), None);
    }

    #[test]
    fn event_pc_extracts_from_instruction_variants() {
        // All 10 instruction-class variants carry a PC.
        assert_eq!(event_pc(&EventType::InstrVector { pc: 0x100 }), Some(0x100));
        assert_eq!(event_pc(&EventType::InstrLoad { pc: 0x104 }), Some(0x104));
        assert_eq!(event_pc(&EventType::InstrStore { pc: 0x108 }), Some(0x108));
        assert_eq!(event_pc(&EventType::InstrCall { pc: 0x10C }), Some(0x10C));
        assert_eq!(event_pc(&EventType::InstrReturn { pc: 0x110 }), Some(0x110));
        assert_eq!(event_pc(&EventType::InstrLockAcquireReq { pc: 0x114 }), Some(0x114));
        assert_eq!(event_pc(&EventType::InstrLockReleaseReq { pc: 0x118 }), Some(0x118));
        assert_eq!(event_pc(&EventType::InstrStreamGet { pc: 0x11C }), Some(0x11C));
        assert_eq!(event_pc(&EventType::InstrStreamPut { pc: 0x120 }), Some(0x120));
        assert_eq!(event_pc(&EventType::InstrEvent { pc: 0x300, id: 1 }), Some(0x300));

        // Stall variants thread the optional pc through.
        assert_eq!(event_pc(&EventType::MemoryStall { cycles: 5, pc: None }), None);
        assert_eq!(event_pc(&EventType::LockStall { cycles: 3, pc: None }), None);
        assert_eq!(event_pc(&EventType::StreamStall { cycles: 1, pc: None }), None);
        assert_eq!(event_pc(&EventType::MemoryStall { cycles: 5, pc: Some(0x340) }), Some(0x340));
        assert_eq!(event_pc(&EventType::LockStall { cycles: 3, pc: Some(0x340) }), Some(0x340));
        assert_eq!(event_pc(&EventType::StreamStall { cycles: 1, pc: Some(0x340) }), Some(0x340));
        assert_eq!(event_pc(&EventType::CoreActive), None);
        assert_eq!(event_pc(&EventType::CoreDisabled), None);
        assert_eq!(event_pc(&EventType::DmaStartTask { channel: 0 }), None);
    }
}
