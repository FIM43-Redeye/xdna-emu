//! Trace event sweep: multi-group NPU trace capture and comparison.
//!
//! The NPU hardware trace unit has 8 event slots configured by two registers
//! (Trace_Event0 and Trace_Event1). Each slot monitors one hardware event.
//! With 128 possible core events and only 8 slots per run, a full picture
//! requires sweeping through multiple event group configurations.
//!
//! This module provides:
//! - Complete AIE2 core event catalogue (128 events with categories)
//! - 13 sweep groups covering all 100 sweepable events (sequential order)
//! - Binary patching of `insts.bin` to swap trace event configurations
//!   without recompilation
//! - Decoding of binary trace packets into event sequences
//! - Multi-group trace merging into unified all-event traces
//! - Comparison of NPU vs emulator event sequences per group

use std::path::Path;

// -------------------------------------------------------------------------
// Event group definitions
// -------------------------------------------------------------------------

/// A trace event group: 8 hardware event IDs assigned to the 8 trace slots.
///
/// The 8 IDs are packed into two 32-bit register values:
/// - Trace_Event0: slots 0-3 (low byte = slot 0)
/// - Trace_Event1: slots 4-7 (low byte = slot 4)
pub struct TraceEventGroup {
    /// Human-readable group name.
    pub name: &'static str,
    /// Hardware event IDs in slot order (0-7).
    pub event_ids: [u8; 8],
}

impl TraceEventGroup {
    /// Compute the Trace_Event0 register value (slots 0-3, little-endian).
    pub fn event0_value(&self) -> u32 {
        u32::from_le_bytes([self.event_ids[0], self.event_ids[1], self.event_ids[2], self.event_ids[3]])
    }

    /// Compute the Trace_Event1 register value (slots 4-7, little-endian).
    pub fn event1_value(&self) -> u32 {
        u32::from_le_bytes([self.event_ids[4], self.event_ids[5], self.event_ids[6], self.event_ids[7]])
    }
}

// -------------------------------------------------------------------------
// AIE2 core trace event catalogue
// -------------------------------------------------------------------------
// Source: aie-rt xaie_events_aieml.h (lines 35-161),
// cross-referenced with mlir-aie aie2.py (CoreEvent enum).

/// Categories for core trace events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventCategory {
    /// Null event (ID 0) -- never fires.
    None,
    /// Always-true trigger (ID 1) -- fires every cycle, floods trace.
    Trigger,
    /// Group meta-event -- fires when any sub-event in its group fires.
    /// Useful for single-slot summaries; redundant when sub-events are
    /// captured individually in a full sweep.
    Group,
    /// Timer synchronization and threshold events.
    Timer,
    /// Performance counter overflow (requires counter pre-configuration).
    PerfCounter,
    /// Combinational logic events (requires combo register config).
    Combo,
    /// Edge detection events (requires edge detector config).
    EdgeDetect,
    /// Program counter watchpoint events (requires PC watchpoint config).
    PcWatchpoint,
    /// Pipeline stall events.
    Stall,
    /// Core debug state.
    Debug,
    /// Core activity status.
    Status,
    /// ECC error/scrubbing events.
    Ecc,
    /// Instruction type events (fires on decode of matching instruction).
    Instruction,
    /// Arithmetic and floating-point error events.
    ComputeError,
    /// Memory, bus, and access error events.
    AccessError,
    /// Stream switch port status events (idle/running/stalled/tlast x 8 ports).
    StreamPort,
    /// Inter-tile broadcast events (16 channels).
    Broadcast,
    /// Application-defined user events (4 channels).
    UserEvent,
    /// Reserved/undefined hardware event.
    Reserved,
}

/// A core trace event from the AIE2 hardware event catalogue.
#[derive(Debug, Clone, Copy)]
pub struct CoreTraceEvent {
    /// Human-readable event name (matches aie-rt / mlir-aie naming).
    pub name: &'static str,
    /// Event category.
    pub category: EventCategory,
    /// Whether this event is included in automatic sweep groups.
    ///
    /// Events excluded from sweep:
    /// - NONE (0): never fires
    /// - TRUE (1): fires every cycle, floods trace buffer
    /// - GROUP_* meta-events: redundant when sub-events are captured individually
    /// - Config-dependent (perf counters, combos, edge detectors, PC watchpoints):
    ///   require hardware pre-configuration to fire
    /// - RESERVED (54): undefined
    pub sweepable: bool,
}

/// Sweepable event helper (included in automatic sweep).
const fn sw(name: &'static str, c: EventCategory) -> CoreTraceEvent {
    CoreTraceEvent { name, category: c, sweepable: true }
}
/// Non-sweepable event helper (excluded from automatic sweep).
const fn ns(name: &'static str, c: EventCategory) -> CoreTraceEvent {
    CoreTraceEvent { name, category: c, sweepable: false }
}

/// Complete AIE2 core module trace event catalogue (128 events, indexed by ID).
///
/// Source: `aie-rt/driver/src/events/xaie_events_aieml.h` lines 35-161,
/// cross-referenced with `mlir-aie/.../trace/events/aie2.py`.
#[rustfmt::skip]
pub const CORE_EVENTS: [CoreTraceEvent; 128] = {
    use EventCategory as E;
    [
        ns("NONE",                     E::None),          //   0
        ns("TRUE",                     E::Trigger),       //   1
        ns("GROUP_0",                  E::Group),         //   2
        sw("TIMER_SYNC",              E::Timer),         //   3
        sw("TIMER_VALUE_REACHED",     E::Timer),         //   4
        ns("PERF_CNT_0",              E::PerfCounter),   //   5
        ns("PERF_CNT_1",              E::PerfCounter),   //   6
        ns("PERF_CNT_2",              E::PerfCounter),   //   7
        ns("PERF_CNT_3",              E::PerfCounter),   //   8
        ns("COMBO_EVENT_0",           E::Combo),         //   9
        ns("COMBO_EVENT_1",           E::Combo),         //  10
        ns("COMBO_EVENT_2",           E::Combo),         //  11
        ns("COMBO_EVENT_3",           E::Combo),         //  12
        ns("EDGE_DETECTION_0",        E::EdgeDetect),    //  13
        ns("EDGE_DETECTION_1",        E::EdgeDetect),    //  14
        ns("GROUP_PC_EVENT",          E::Group),         //  15
        ns("PC_0",                     E::PcWatchpoint),  //  16
        ns("PC_1",                     E::PcWatchpoint),  //  17
        ns("PC_2",                     E::PcWatchpoint),  //  18
        ns("PC_3",                     E::PcWatchpoint),  //  19
        ns("PC_RANGE_0_1",            E::PcWatchpoint),  //  20
        ns("PC_RANGE_2_3",            E::PcWatchpoint),  //  21
        ns("GROUP_STALL",             E::Group),         //  22
        sw("MEMORY_STALL",            E::Stall),         //  23
        sw("STREAM_STALL",            E::Stall),         //  24
        sw("CASCADE_STALL",           E::Stall),         //  25
        sw("LOCK_STALL",              E::Stall),         //  26
        sw("DEBUG_HALTED",            E::Debug),         //  27
        sw("ACTIVE",                   E::Status),        //  28
        sw("DISABLED",                 E::Status),        //  29
        sw("ECC_ERROR_STALL",         E::Ecc),           //  30
        sw("ECC_SCRUBBING_STALL",     E::Ecc),           //  31
        ns("GROUP_PROGRAM_FLOW",      E::Group),         //  32
        sw("INSTR_EVENT_0",           E::Instruction),   //  33
        sw("INSTR_EVENT_1",           E::Instruction),   //  34
        sw("INSTR_CALL",              E::Instruction),   //  35
        sw("INSTR_RETURN",            E::Instruction),   //  36
        sw("INSTR_VECTOR",            E::Instruction),   //  37
        sw("INSTR_LOAD",              E::Instruction),   //  38
        sw("INSTR_STORE",             E::Instruction),   //  39
        sw("INSTR_STREAM_GET",        E::Instruction),   //  40
        sw("INSTR_STREAM_PUT",        E::Instruction),   //  41
        sw("INSTR_CASCADE_GET",       E::Instruction),   //  42
        sw("INSTR_CASCADE_PUT",       E::Instruction),   //  43
        sw("INSTR_LOCK_ACQUIRE_REQ",  E::Instruction),   //  44
        sw("INSTR_LOCK_RELEASE_REQ",  E::Instruction),   //  45
        ns("GROUP_ERRORS_0",          E::Group),         //  46
        ns("GROUP_ERRORS_1",          E::Group),         //  47
        sw("SRS_OVERFLOW",            E::ComputeError),  //  48
        sw("UPS_OVERFLOW",            E::ComputeError),  //  49
        sw("FP_HUGE",                  E::ComputeError),  //  50
        sw("INT_FP_0",                E::ComputeError),  //  51
        sw("FP_INVALID",              E::ComputeError),  //  52
        sw("FP_INF",                   E::ComputeError),  //  53
        ns("RESERVED_54",             E::Reserved),      //  54
        sw("PM_REG_ACCESS_FAILURE",   E::AccessError),   //  55
        sw("STREAM_PKT_PARITY_ERR",   E::AccessError),   //  56
        sw("CONTROL_PKT_ERROR",       E::AccessError),   //  57
        sw("AXI_MM_SLAVE_ERROR",      E::AccessError),   //  58
        sw("INSTR_DECOMPRSN_ERROR",   E::AccessError),   //  59
        sw("DM_ADDR_OUT_OF_RANGE",    E::AccessError),   //  60
        sw("PM_ECC_SCRUB_CORRECTED",  E::AccessError),   //  61
        sw("PM_ECC_SCRUB_2BIT",       E::AccessError),   //  62
        sw("PM_ECC_ERROR_1BIT",       E::AccessError),   //  63
        sw("PM_ECC_ERROR_2BIT",       E::AccessError),   //  64
        sw("PM_ADDR_OUT_OF_RANGE",    E::AccessError),   //  65
        sw("DM_ACCESS_UNAVAILABLE",   E::AccessError),   //  66
        sw("LOCK_ACCESS_UNAVAILABLE", E::AccessError),   //  67
        sw("INSTR_WARNING",           E::AccessError),   //  68
        sw("INSTR_ERROR",             E::AccessError),   //  69
        sw("DECOMPRESSION_UNDERFLOW", E::AccessError),   //  70
        sw("SS_PORT_PARITY_ERROR",    E::AccessError),   //  71
        sw("PROCESSOR_BUS_ERROR",     E::AccessError),   //  72
        ns("GROUP_STREAM_SWITCH",     E::Group),         //  73
        sw("PORT_IDLE_0",             E::StreamPort),    //  74
        sw("PORT_RUNNING_0",          E::StreamPort),    //  75
        sw("PORT_STALLED_0",          E::StreamPort),    //  76
        sw("PORT_TLAST_0",            E::StreamPort),    //  77
        sw("PORT_IDLE_1",             E::StreamPort),    //  78
        sw("PORT_RUNNING_1",          E::StreamPort),    //  79
        sw("PORT_STALLED_1",          E::StreamPort),    //  80
        sw("PORT_TLAST_1",            E::StreamPort),    //  81
        sw("PORT_IDLE_2",             E::StreamPort),    //  82
        sw("PORT_RUNNING_2",          E::StreamPort),    //  83
        sw("PORT_STALLED_2",          E::StreamPort),    //  84
        sw("PORT_TLAST_2",            E::StreamPort),    //  85
        sw("PORT_IDLE_3",             E::StreamPort),    //  86
        sw("PORT_RUNNING_3",          E::StreamPort),    //  87
        sw("PORT_STALLED_3",          E::StreamPort),    //  88
        sw("PORT_TLAST_3",            E::StreamPort),    //  89
        sw("PORT_IDLE_4",             E::StreamPort),    //  90
        sw("PORT_RUNNING_4",          E::StreamPort),    //  91
        sw("PORT_STALLED_4",          E::StreamPort),    //  92
        sw("PORT_TLAST_4",            E::StreamPort),    //  93
        sw("PORT_IDLE_5",             E::StreamPort),    //  94
        sw("PORT_RUNNING_5",          E::StreamPort),    //  95
        sw("PORT_STALLED_5",          E::StreamPort),    //  96
        sw("PORT_TLAST_5",            E::StreamPort),    //  97
        sw("PORT_IDLE_6",             E::StreamPort),    //  98
        sw("PORT_RUNNING_6",          E::StreamPort),    //  99
        sw("PORT_STALLED_6",          E::StreamPort),    // 100
        sw("PORT_TLAST_6",            E::StreamPort),    // 101
        sw("PORT_IDLE_7",             E::StreamPort),    // 102
        sw("PORT_RUNNING_7",          E::StreamPort),    // 103
        sw("PORT_STALLED_7",          E::StreamPort),    // 104
        sw("PORT_TLAST_7",            E::StreamPort),    // 105
        ns("GROUP_BROADCAST",         E::Group),         // 106
        sw("BROADCAST_0",             E::Broadcast),     // 107
        sw("BROADCAST_1",             E::Broadcast),     // 108
        sw("BROADCAST_2",             E::Broadcast),     // 109
        sw("BROADCAST_3",             E::Broadcast),     // 110
        sw("BROADCAST_4",             E::Broadcast),     // 111
        sw("BROADCAST_5",             E::Broadcast),     // 112
        sw("BROADCAST_6",             E::Broadcast),     // 113
        sw("BROADCAST_7",             E::Broadcast),     // 114
        sw("BROADCAST_8",             E::Broadcast),     // 115
        sw("BROADCAST_9",             E::Broadcast),     // 116
        sw("BROADCAST_10",            E::Broadcast),     // 117
        sw("BROADCAST_11",            E::Broadcast),     // 118
        sw("BROADCAST_12",            E::Broadcast),     // 119
        sw("BROADCAST_13",            E::Broadcast),     // 120
        sw("BROADCAST_14",            E::Broadcast),     // 121
        sw("BROADCAST_15",            E::Broadcast),     // 122
        ns("GROUP_USER_EVENT",        E::Group),         // 123
        sw("USER_EVENT_0",            E::UserEvent),     // 124
        sw("USER_EVENT_1",            E::UserEvent),     // 125
        sw("USER_EVENT_2",            E::UserEvent),     // 126
        sw("USER_EVENT_3",            E::UserEvent),     // 127
    ]
};

/// Look up an event's human-readable name by hardware ID (0-127).
pub fn event_name(id: u8) -> &'static str {
    CORE_EVENTS.get(id as usize).map(|e| e.name).unwrap_or("UNKNOWN")
}

/// Return event IDs included in the automatic sweep (sequential order).
///
/// 100 events across 13 groups of 8. Excludes: NONE, TRUE, GROUP_* meta-events,
/// config-dependent events (perf counters, combos, edge detectors, PC watchpoints),
/// and RESERVED.
pub fn sweepable_event_ids() -> Vec<u8> {
    CORE_EVENTS
        .iter()
        .enumerate()
        .filter(|(_, e)| e.sweepable)
        .map(|(id, _)| id as u8)
        .collect()
}

// -------------------------------------------------------------------------
// Sweep groups (13 groups, sequential order through sweepable events)
// -------------------------------------------------------------------------

/// Group 0: timer + pipeline stalls + debug/status.
const SWEEP_00: TraceEventGroup =
    TraceEventGroup { name: "timer_stall_debug", event_ids: [3, 4, 23, 24, 25, 26, 27, 28] };
/// Group 1: status/ECC + instruction events (begin).
const SWEEP_01: TraceEventGroup =
    TraceEventGroup { name: "status_instr_a", event_ids: [29, 30, 31, 33, 34, 35, 36, 37] };
/// Group 2: instruction events (load/store/stream/cascade/lock).
const SWEEP_02: TraceEventGroup =
    TraceEventGroup { name: "instr_mem_sync", event_ids: [38, 39, 40, 41, 42, 43, 44, 45] };
/// Group 3: compute errors + first access errors.
const SWEEP_03: TraceEventGroup =
    TraceEventGroup { name: "error_compute", event_ids: [48, 49, 50, 51, 52, 53, 55, 56] };
/// Group 4: access/system errors (a).
const SWEEP_04: TraceEventGroup =
    TraceEventGroup { name: "error_access_a", event_ids: [57, 58, 59, 60, 61, 62, 63, 64] };
/// Group 5: access/system errors (b).
const SWEEP_05: TraceEventGroup =
    TraceEventGroup { name: "error_access_b", event_ids: [65, 66, 67, 68, 69, 70, 71, 72] };
/// Group 6: stream port events (ports 0-1).
const SWEEP_06: TraceEventGroup =
    TraceEventGroup { name: "port_01", event_ids: [74, 75, 76, 77, 78, 79, 80, 81] };
/// Group 7: stream port events (ports 2-3).
const SWEEP_07: TraceEventGroup =
    TraceEventGroup { name: "port_23", event_ids: [82, 83, 84, 85, 86, 87, 88, 89] };
/// Group 8: stream port events (ports 4-5).
const SWEEP_08: TraceEventGroup =
    TraceEventGroup { name: "port_45", event_ids: [90, 91, 92, 93, 94, 95, 96, 97] };
/// Group 9: stream port events (ports 6-7).
const SWEEP_09: TraceEventGroup =
    TraceEventGroup { name: "port_67", event_ids: [98, 99, 100, 101, 102, 103, 104, 105] };
/// Group 10: broadcast channels 0-7.
const SWEEP_10: TraceEventGroup =
    TraceEventGroup { name: "broadcast_0to7", event_ids: [107, 108, 109, 110, 111, 112, 113, 114] };
/// Group 11: broadcast channels 8-15.
const SWEEP_11: TraceEventGroup =
    TraceEventGroup { name: "broadcast_8to15", event_ids: [115, 116, 117, 118, 119, 120, 121, 122] };
/// Group 12: user-defined events (padded with NONE).
const SWEEP_12: TraceEventGroup =
    TraceEventGroup { name: "user_event", event_ids: [124, 125, 126, 127, 0, 0, 0, 0] };

/// All sweep groups in order. Used by the trace sweep orchestration.
pub const TRACE_EVENT_GROUPS: &[&TraceEventGroup] = &[
    &SWEEP_00, &SWEEP_01, &SWEEP_02, &SWEEP_03, &SWEEP_04, &SWEEP_05, &SWEEP_06, &SWEEP_07, &SWEEP_08,
    &SWEEP_09, &SWEEP_10, &SWEEP_11, &SWEEP_12,
];

/// Number of sweep groups.
pub const NUM_GROUPS: usize = 13;

/// The fuzz_template.py default trace event configuration.
///
/// Register values: Trace_Event0 = 0x26300221, Trace_Event1 = 0x4F477A2C.
/// Kept for reference, sanity-checking, and non-sweep use cases.
pub const LEGACY_DEFAULT_GROUP: TraceEventGroup = TraceEventGroup {
    name: "legacy_default",
    event_ids: [0x21, 0x02, 0x30, 0x26, 0x2C, 0x7A, 0x47, 0x4F],
};

// -------------------------------------------------------------------------
// NPU address computation
// -------------------------------------------------------------------------

/// Compute the NPU register address from tile coordinates and module offset.
///
/// NPU address encoding: `(col << 25) | (row << 20) | tile_offset`.
fn npu_address(col: u32, row: u32, tile_offset: u32) -> u32 {
    (col << 25) | (row << 20) | tile_offset
}

/// Trace_Event0 register offset within the core trace module.
const TRACE_EVENT0_OFFSET: u32 = 0x340E0;

/// Trace_Event1 register offset within the core trace module.
const TRACE_EVENT1_OFFSET: u32 = 0x340E4;

/// Default tile coordinates for trace configuration (column 0, row 2).
/// Matches fuzz_template.py which configures tile (0,2).
const TRACE_TILE_COL: u32 = 0;
const TRACE_TILE_ROW: u32 = 2;

// -------------------------------------------------------------------------
// insts.bin patching
// -------------------------------------------------------------------------

/// Scan `insts.bin` for a Write32 instruction targeting `target_reg_off`,
/// returning the byte offset of the 4-byte value field within the file.
///
/// Write32 layout (24 bytes total):
///   [0]:    opcode byte (0x00)
///   [1-3]:  padding (zeros)
///   [4-7]:  padding (zeros)
///   [8-15]: reg_off as u64 LE
///   [16-19]: value as u32 LE  <-- returned offset points here
///   [20-23]: size as u32 LE
fn find_write32_value_offset(insts: &[u8], target_reg_off: u32) -> Option<usize> {
    // Scan at 4-byte boundaries looking for Write32 instructions.
    // Each instruction is at least 24 bytes for Write32.
    let mut i = 0;
    while i + 24 <= insts.len() {
        // Check opcode byte (0x00 = Write32)
        if insts[i] == 0x00 {
            // Verify padding bytes are zero (bytes 1-7)
            let pad_ok = insts[i + 1..i + 8].iter().all(|&b| b == 0);
            if pad_ok {
                // Read reg_off (u64 LE at offset 8)
                let reg_off_lo =
                    u32::from_le_bytes([insts[i + 8], insts[i + 9], insts[i + 10], insts[i + 11]]);
                let reg_off_hi =
                    u32::from_le_bytes([insts[i + 12], insts[i + 13], insts[i + 14], insts[i + 15]]);

                if reg_off_lo == target_reg_off && reg_off_hi == 0 {
                    return Some(i + 16);
                }
            }
        }
        i += 4;
    }
    None
}

/// Patch `insts.bin` bytes to use the given trace event group.
///
/// Finds the Write32 instructions targeting Trace_Event0 and Trace_Event1
/// for tile (0,2) and replaces their value fields. Returns `Err` if either
/// target instruction is not found.
pub fn patch_insts_for_group(insts: &[u8], group: &TraceEventGroup) -> Result<Vec<u8>, String> {
    let event0_addr = npu_address(TRACE_TILE_COL, TRACE_TILE_ROW, TRACE_EVENT0_OFFSET);
    let event1_addr = npu_address(TRACE_TILE_COL, TRACE_TILE_ROW, TRACE_EVENT1_OFFSET);

    let off0 = find_write32_value_offset(insts, event0_addr).ok_or_else(|| {
        format!("Write32 for Trace_Event0 (addr 0x{:08X}) not found in insts.bin", event0_addr)
    })?;
    let off1 = find_write32_value_offset(insts, event1_addr).ok_or_else(|| {
        format!("Write32 for Trace_Event1 (addr 0x{:08X}) not found in insts.bin", event1_addr)
    })?;

    let mut patched = insts.to_vec();
    patched[off0..off0 + 4].copy_from_slice(&group.event0_value().to_le_bytes());
    patched[off1..off1 + 4].copy_from_slice(&group.event1_value().to_le_bytes());
    Ok(patched)
}

// -------------------------------------------------------------------------
// Binary trace decoding
// -------------------------------------------------------------------------

/// A decoded trace event: slot ID and absolute cycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedEvent {
    /// Event slot (0-7).
    pub slot: u8,
    /// Absolute cycle (from start marker + accumulated deltas).
    pub abs_cycle: u64,
}

/// Trim a raw trace buffer to just the region containing real data.
///
/// NPU trace buffers are typically 1MB, mostly filled with padding (0xFE
/// bytes) and zeros. This function finds the end of actual trace data using
/// the same heuristic as mlir-aie's `trim_trace_pkts`: the first uint32
/// word equal to `0xFEFEFEFE` followed by two `0x00000000` words marks the
/// boundary. Everything from that point onward (inclusive of the sentinel)
/// is padding.
///
/// Returns the byte length of the valid data prefix (always a multiple of 4).
/// If no trim point is found, returns the full buffer length.
pub fn trim_trace_buffer(data: &[u8]) -> usize {
    // Scan uint32 words looking for 0xFEFEFEFE followed by two 0x00000000.
    let word_count = data.len() / 4;
    for i in 0..word_count {
        let off = i * 4;
        let word = u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        if word == 0xFEFEFEFE {
            // Check if next two words are 0x00000000.
            if i + 2 < word_count {
                let w1 = u32::from_le_bytes([data[off + 4], data[off + 5], data[off + 6], data[off + 7]]);
                let w2 = u32::from_le_bytes([data[off + 8], data[off + 9], data[off + 10], data[off + 11]]);
                if w1 == 0 && w2 == 0 {
                    // Include the sentinel word itself (matches mlir-aie behavior).
                    return (i + 1) * 4;
                }
            }
        }
    }
    data.len()
}

/// Check if a 32-bit word is a valid trace packet header.
///
/// Per mlir-aie `parse_pkt_hdr_in_stream`, a valid header must have:
/// - Odd parity (odd number of 1-bits)
/// - Bits [11:5] = 0 (reserved)
/// - Bit 19 = 0 (reserved)
/// - Bits [30:28] = 0 (reserved)
///
/// All-zero words (empty buffers) fail the parity check.
fn is_valid_trace_header(word: u32) -> bool {
    // Odd parity: popcount must be odd.
    if word.count_ones() % 2 == 0 {
        return false;
    }
    // Reserved fields must be zero.
    if ((word >> 5) & 0x7F) != 0 {
        return false;
    }
    if ((word >> 19) & 0x1) != 0 {
        return false;
    }
    if ((word >> 28) & 0x7) != 0 {
        return false;
    }
    true
}

/// Decode binary trace data (raw bytes from the trace buffer) into events.
///
/// The buffer is first trimmed to remove padding/zeros (see
/// [`trim_trace_buffer`]), then decoded packet-by-packet.
///
/// The trace buffer contains 32-byte packets. Word 0 is the header; words
/// 1-7 contain 28 bytes of encoded events (big-endian packed into u32s).
///
/// Hardware event encoding (from mlir-aie parse.py / AM025):
///
/// | Type     | Discriminant       | Bytes | Slot bits | Delta bits |
/// |----------|--------------------|-------|-----------|------------|
/// | Single0  | bit7=0             | 1     | [6:4]     | [3:0]      |
/// | Single1  | bits[7:5]=100      | 2     | [4:2]     | [1:0]+[7:0]|
/// | Single2  | bits[7:5]=101      | 3     | [4:2]     | [1:0]+[15:0]|
/// | Multiple0| bits[7:4]=1100     | 2     | bitmask   | [3:0]      |
/// | Multiple1| bits[7:2]=110100   | 3     | bitmask   | [9:0]      |
/// | Multiple2| bits[7:2]=110101   | 4     | bitmask   | [17:0]     |
/// | Start    | (byte&0xFB)==0xF0  | 8     | N/A       | timer 56b  |
/// | Repeat0  | bits[7:4]=1110     | 1     | N/A       | N/A        |
/// | Padding  | 0xFE               | 1     | N/A       | N/A        |
/// | Sync     | 0xFF               | 1     | N/A       | N/A        |
///
/// Note: the emulator's trace_unit.rs encoder uses a different bit layout
/// (slot in [6:4] for Single1/2), which is a known divergence from hardware.
/// This decoder implements the hardware format for NPU trace compatibility.
pub fn decode_binary_trace(data: &[u8]) -> Vec<DecodedEvent> {
    let trimmed_len = trim_trace_buffer(data);
    let data = &data[..trimmed_len];

    let mut events = Vec::new();

    // Each packet is 32 bytes (8 words of 4 bytes each).
    let packet_size = 32;
    let mut pkt_offset = 0;

    let mut abs_cycle: u64 = 0;
    // For repeat commands: replay the last event set.
    let mut last_events: Vec<DecodedEvent> = Vec::new();
    let mut last_delta: u64 = 0;

    while pkt_offset + packet_size <= data.len() {
        // Word 0 is the packet header. Validate it before decoding.
        // A valid trace packet header has odd parity and specific reserved
        // bits clear (matching mlir-aie's parse_pkt_hdr_in_stream).
        let header = u32::from_le_bytes([
            data[pkt_offset],
            data[pkt_offset + 1],
            data[pkt_offset + 2],
            data[pkt_offset + 3],
        ]);
        if !is_valid_trace_header(header) {
            break;
        }

        // Words 1-7 are data: extract 28 bytes.
        // Trace buffers are stored as little-endian uint32 words. Each word
        // packs 4 event bytes MSB-first (byte 3 = most significant position),
        // matching mlir-aie's convert_to_byte_stream which extracts with
        // `event >> (byte * 8)` for byte = 3,2,1,0.
        let mut payload = [0u8; 28];
        for word_idx in 0..7 {
            let word_offset = pkt_offset + (word_idx + 1) * 4;
            let word = u32::from_le_bytes([
                data[word_offset],
                data[word_offset + 1],
                data[word_offset + 2],
                data[word_offset + 3],
            ]);
            payload[word_idx * 4] = (word >> 24) as u8;
            payload[word_idx * 4 + 1] = (word >> 16) as u8;
            payload[word_idx * 4 + 2] = (word >> 8) as u8;
            payload[word_idx * 4 + 3] = word as u8;
        }

        // Decode events from the 28-byte payload.
        let mut i = 0;
        while i < 28 {
            let b = payload[i];

            // Padding (0xFE)
            if b == 0xFE {
                i += 1;
                continue;
            }
            // Sync / end marker (0xFF)
            if b == 0xFF {
                i += 1;
                continue;
            }
            // Start marker: (byte & 0xFB) == 0xF0
            if (b & 0xFB) == 0xF0 {
                if i + 8 > 28 {
                    break;
                }
                let mut timer: u64 = 0;
                for j in 1..8 {
                    timer = (timer << 8) | (payload[i + j] as u64);
                }
                abs_cycle = timer;
                i += 8;
                continue;
            }
            // Repeat0: bits[7:4] = 1110
            if b & 0xF0 == 0xE0 {
                let repeats = (b & 0x0F) as u64;
                for _ in 0..repeats {
                    abs_cycle += last_delta;
                    for ev in &last_events {
                        events.push(DecodedEvent { slot: ev.slot, abs_cycle });
                    }
                }
                i += 1;
                continue;
            }
            // Filler: bits[7:2] = 110111
            if b & 0xFC == 0xDC {
                i += 4;
                continue;
            }
            // Repeat1: bits[7:2] = 110110
            if b & 0xFC == 0xD8 {
                if i + 2 > 28 {
                    break;
                }
                let repeats = (((b & 0x03) as u64) << 8) | (payload[i + 1] as u64);
                for _ in 0..repeats {
                    abs_cycle += last_delta;
                    for ev in &last_events {
                        events.push(DecodedEvent { slot: ev.slot, abs_cycle });
                    }
                }
                i += 2;
                continue;
            }
            // Multiple2: bits[7:2] = 110101
            if b & 0xFC == 0xD4 {
                if i + 4 > 28 {
                    break;
                }
                let event_mask = ((b as u16 & 0x03) << 6) | ((payload[i + 1] as u16) >> 2);
                let delta = (((payload[i + 1] & 0x03) as u64) << 16)
                    | ((payload[i + 2] as u64) << 8)
                    | (payload[i + 3] as u64);
                abs_cycle += delta;
                last_delta = delta;
                last_events.clear();
                for slot in 0..8u8 {
                    if event_mask & (1 << slot) != 0 {
                        let ev = DecodedEvent { slot, abs_cycle };
                        last_events.push(ev.clone());
                        events.push(ev);
                    }
                }
                i += 4;
                continue;
            }
            // Multiple1: bits[7:2] = 110100
            if b & 0xFC == 0xD0 {
                if i + 3 > 28 {
                    break;
                }
                let event_mask = ((b as u16 & 0x03) << 6) | ((payload[i + 1] as u16) >> 2);
                let delta = (((payload[i + 1] & 0x03) as u64) << 8) | (payload[i + 2] as u64);
                abs_cycle += delta;
                last_delta = delta;
                last_events.clear();
                for slot in 0..8u8 {
                    if event_mask & (1 << slot) != 0 {
                        let ev = DecodedEvent { slot, abs_cycle };
                        last_events.push(ev.clone());
                        events.push(ev);
                    }
                }
                i += 3;
                continue;
            }
            // Multiple0: bits[7:4] = 1100
            if b & 0xF0 == 0xC0 {
                if i + 2 > 28 {
                    break;
                }
                let event_mask = ((b as u16 & 0x0F) << 4) | ((payload[i + 1] as u16) >> 4);
                let delta = (payload[i + 1] & 0x0F) as u64;
                abs_cycle += delta;
                last_delta = delta;
                last_events.clear();
                for slot in 0..8u8 {
                    if event_mask & (1 << slot) != 0 {
                        let ev = DecodedEvent { slot, abs_cycle };
                        last_events.push(ev.clone());
                        events.push(ev);
                    }
                }
                i += 2;
                continue;
            }
            // Single2: bits[7:5] = 101
            if b & 0xE0 == 0xA0 {
                if i + 3 > 28 {
                    break;
                }
                let slot = (b >> 2) & 0x07;
                let delta =
                    (((b & 0x03) as u64) << 16) | ((payload[i + 1] as u64) << 8) | (payload[i + 2] as u64);
                abs_cycle += delta;
                last_delta = delta;
                last_events.clear();
                let ev = DecodedEvent { slot, abs_cycle };
                last_events.push(ev.clone());
                events.push(ev);
                i += 3;
                continue;
            }
            // Single1: bits[7:5] = 100
            if b & 0xE0 == 0x80 {
                if i + 2 > 28 {
                    break;
                }
                let slot = (b >> 2) & 0x07;
                let delta = (((b & 0x03) as u64) << 8) | (payload[i + 1] as u64);
                abs_cycle += delta;
                last_delta = delta;
                last_events.clear();
                let ev = DecodedEvent { slot, abs_cycle };
                last_events.push(ev.clone());
                events.push(ev);
                i += 2;
                continue;
            }
            // Single0: bit7 = 0
            if b & 0x80 == 0 {
                let slot = (b >> 4) & 0x07;
                let delta = (b & 0x0F) as u64;
                abs_cycle += delta;
                last_delta = delta;
                last_events.clear();
                let ev = DecodedEvent { slot, abs_cycle };
                last_events.push(ev.clone());
                events.push(ev);
                i += 1;
                continue;
            }

            // Unknown byte -- skip.
            i += 1;
        }

        pkt_offset += packet_size;
    }

    events
}

// -------------------------------------------------------------------------
// Trace comparison
// -------------------------------------------------------------------------

/// Result of comparing NPU and emulator event sequences for one group.
#[derive(Debug)]
pub struct TraceComparison {
    /// Name of the trace event group.
    pub group_name: String,
    /// Number of events decoded from NPU trace.
    pub npu_event_count: usize,
    /// Number of events decoded from emulator trace.
    pub emu_event_count: usize,
    /// Whether the slot sequences match exactly.
    pub sequence_match: bool,
    /// First divergence point: (index, npu_slot, emu_slot).
    pub first_divergence: Option<(usize, u8, u8)>,
    /// Number of events filtered during canonicalization (0 = raw comparison).
    pub filtered_count: usize,
}

/// Compare NPU and emulator event sequences by slot ID only.
///
/// Absolute cycles are ignored (emulator and hardware have different timing).
/// Two sequences match if they have the same length and identical slot IDs
/// at every position.
pub fn compare_event_sequences(
    npu_events: &[DecodedEvent],
    emu_events: &[DecodedEvent],
    group_name: &str,
) -> TraceComparison {
    let npu_count = npu_events.len();
    let emu_count = emu_events.len();

    // Find first divergence in slot sequences.
    let min_len = npu_count.min(emu_count);
    let mut first_divergence = None;

    for i in 0..min_len {
        if npu_events[i].slot != emu_events[i].slot {
            first_divergence = Some((i, npu_events[i].slot, emu_events[i].slot));
            break;
        }
    }

    // If lengths differ and no slot mismatch was found, the divergence is
    // at the end of the shorter sequence.
    if first_divergence.is_none() && npu_count != emu_count {
        let idx = min_len;
        let npu_slot = if idx < npu_count {
            npu_events[idx].slot
        } else {
            u8::MAX
        };
        let emu_slot = if idx < emu_count {
            emu_events[idx].slot
        } else {
            u8::MAX
        };
        first_divergence = Some((idx, npu_slot, emu_slot));
    }

    let sequence_match = first_divergence.is_none();

    TraceComparison {
        group_name: group_name.to_string(),
        npu_event_count: npu_count,
        emu_event_count: emu_count,
        sequence_match,
        first_divergence,
        filtered_count: 0,
    }
}

/// Format a TraceComparison as a human-readable string.
impl std::fmt::Display for TraceComparison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let suffix = if self.filtered_count > 0 {
            format!(" (filtered {} timing-dep)", self.filtered_count)
        } else {
            String::new()
        };
        let label = if self.filtered_count > 0 {
            "canon"
        } else {
            "events"
        };

        if self.sequence_match {
            write!(f, "{}: MATCH ({} {}){}", self.group_name, self.npu_event_count, label, suffix,)
        } else if let Some((idx, npu_s, emu_s)) = self.first_divergence {
            write!(
                f,
                "{}: DIVERGE at index {} (npu_slot={}, emu_slot={}) \
                 [npu={} {}, emu={} {}]{}",
                self.group_name,
                idx,
                if npu_s == u8::MAX {
                    "END".to_string()
                } else {
                    npu_s.to_string()
                },
                if emu_s == u8::MAX {
                    "END".to_string()
                } else {
                    emu_s.to_string()
                },
                self.npu_event_count,
                label,
                self.emu_event_count,
                label,
                suffix,
            )
        } else {
            write!(f, "{}: EMPTY (no events)", self.group_name)
        }
    }
}

// -------------------------------------------------------------------------
// Sweep orchestration helpers
// -------------------------------------------------------------------------

/// A trace command skeleton element, stripped of timing and repeat counts.
///
/// Used for determinism comparison: two traces are considered deterministic
/// if they produce the same sequence of event transitions (same slots firing
/// in the same order), regardless of how many cycles each phase lasts.
#[derive(Debug, Clone, PartialEq, Eq)]
enum SkeletonEntry {
    /// A single event slot fired.
    Single(u8),
    /// Multiple event slots fired simultaneously (sorted).
    Multiple(Vec<u8>),
}

/// Extract a command skeleton from decoded events.
///
/// Groups consecutive events at the same abs_cycle into Multiple entries
/// and single events into Single entries. Repeat runs (same slots at
/// consecutive cycles) collapse into one entry. This strips timing
/// completely, leaving only the ordered sequence of event transitions.
fn extract_skeleton(events: &[DecodedEvent]) -> Vec<SkeletonEntry> {
    if events.is_empty() {
        return Vec::new();
    }

    let mut skeleton = Vec::new();
    let mut i = 0;

    while i < events.len() {
        // Collect all events at the same abs_cycle.
        let cycle = events[i].abs_cycle;
        let mut slots: Vec<u8> = Vec::new();
        while i < events.len() && events[i].abs_cycle == cycle {
            slots.push(events[i].slot);
            i += 1;
        }
        slots.sort();
        slots.dedup();

        let entry = if slots.len() == 1 {
            SkeletonEntry::Single(slots[0])
        } else {
            SkeletonEntry::Multiple(slots)
        };

        // Collapse consecutive identical entries (repeat compression).
        if skeleton.last() != Some(&entry) {
            skeleton.push(entry);
        }
    }

    skeleton
}

// -------------------------------------------------------------------------
// Dimension-driven determinism analysis
// -------------------------------------------------------------------------
//
// Every analysis angle is a named "dimension" -- a projection of the raw
// event stream into a comparable value (scalar or sequence). Adding a new
// angle requires one call to `scalar_dim()` or `sequence_dim()` inside
// `analyze_dimensions()`. Per-slot dimensions auto-generate for all 8
// slots, filtered to those that actually fire.

/// Result of comparing one analysis dimension across repetitions.
#[derive(Debug, Clone)]
pub struct DimensionVerdict {
    /// Dimension name (e.g., "deltas", "slot5_fires", "skeleton").
    pub name: String,
    /// Whether this dimension is identical across all repetitions.
    pub deterministic: bool,
    /// Human-readable summary of the values or variance.
    /// Scalars: "42" or "3-7". Sequences: "match (8)" or "diverge@5/12".
    pub summary: String,
}

// -------------------------------------------------------------------------
// Slot classification and trace canonicalization
// -------------------------------------------------------------------------

/// Classification of a trace slot's determinism behavior, derived from
/// the N-rep determinism check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotClass {
    /// Same fire count across all reps (and > 0) -- include in canonical trace.
    Deterministic,
    /// Fires 0 or 1 times (trace window boundary race) -- include.
    Boundary,
    /// Fire count varies significantly across reps -- exclude from canonical trace.
    TimingDependent,
    /// Never fires in any rep -- exclude.
    Inactive,
}

/// Classify each of the 8 trace slots based on multi-rep fire count analysis.
///
/// Takes the decoded events from each determinism-check rep and determines
/// which slots fire deterministically vs. which depend on runtime timing.
pub fn classify_slots_from_reps(reps: &[Vec<DecodedEvent>]) -> [SlotClass; 8] {
    let mut classes = [SlotClass::Inactive; 8];
    for slot in 0..8u8 {
        let counts: Vec<usize> = reps
            .iter()
            .map(|events| events.iter().filter(|e| e.slot == slot).count())
            .collect();
        let min = counts.iter().copied().min().unwrap_or(0);
        let max = counts.iter().copied().max().unwrap_or(0);

        classes[slot as usize] = if max == 0 {
            SlotClass::Inactive
        } else if min == max {
            SlotClass::Deterministic
        } else if min == 0 && max <= 1 {
            SlotClass::Boundary
        } else {
            SlotClass::TimingDependent
        };
    }
    classes
}

/// A trace with non-deterministic slots filtered out.
///
/// Suitable for direct emulator-vs-hardware comparison: only events that the
/// NPU itself is deterministic about are included. The filtered event counts
/// are preserved for reporting.
pub struct CanonicalTrace {
    /// Events after filtering out TimingDependent and Inactive slots,
    /// aligned to cycle 0.
    pub events: Vec<DecodedEvent>,
    /// Per-slot classification (for reporting).
    pub slot_classes: [SlotClass; 8],
    /// How many events were filtered per slot (for reporting).
    pub filtered_counts: [usize; 8],
}

/// Produce a canonical trace by filtering out non-deterministic slots
/// and aligning to cycle 0.
pub fn canonicalize(events: &[DecodedEvent], slot_classes: &[SlotClass; 8]) -> CanonicalTrace {
    let mut filtered_counts = [0usize; 8];

    // Count filtered events per slot.
    for e in events {
        match slot_classes[e.slot as usize] {
            SlotClass::TimingDependent | SlotClass::Inactive => {
                filtered_counts[e.slot as usize] += 1;
            }
            _ => {}
        }
    }

    // Keep only Deterministic and Boundary events, align to cycle 0.
    let kept: Vec<DecodedEvent> = events
        .iter()
        .filter(|e| matches!(slot_classes[e.slot as usize], SlotClass::Deterministic | SlotClass::Boundary,))
        .cloned()
        .collect();

    let base = kept.first().map(|e| e.abs_cycle).unwrap_or(0);
    let aligned = kept
        .into_iter()
        .map(|e| DecodedEvent { slot: e.slot, abs_cycle: e.abs_cycle.saturating_sub(base) })
        .collect();

    CanonicalTrace { events: aligned, slot_classes: *slot_classes, filtered_counts }
}

/// Compare NPU and emulator traces after canonicalizing both.
///
/// Returns the comparison result plus the total number of filtered events
/// (from both sides) for reporting context.
pub fn compare_canonical(
    npu_events: &[DecodedEvent],
    emu_events: &[DecodedEvent],
    slot_classes: &[SlotClass; 8],
    group_name: &str,
) -> (TraceComparison, usize) {
    let npu_canon = canonicalize(npu_events, slot_classes);
    let emu_canon = canonicalize(emu_events, slot_classes);
    let total_filtered =
        npu_canon.filtered_counts.iter().sum::<usize>() + emu_canon.filtered_counts.iter().sum::<usize>();
    let mut comp = compare_event_sequences(&npu_canon.events, &emu_canon.events, group_name);
    comp.filtered_count = total_filtered;
    (comp, total_filtered)
}

/// Complete determinism analysis across multiple trace buffer repetitions.
///
/// Instead of hardcoding specific analysis angles, this report contains a
/// list of independently-evaluated dimensions. Each dimension compares one
/// projection of the event data across all reps. The `compute_deterministic`
/// verdict is derived from the `deltas` dimension.
#[derive(Debug, Clone)]
pub struct DeterminismReport {
    /// Number of repetitions analyzed.
    pub num_reps: usize,
    /// All dimension verdicts (filtered to non-trivial dimensions).
    pub dimensions: Vec<DimensionVerdict>,
    /// Overall compute determinism verdict (derived from deltas dimension).
    pub compute_deterministic: bool,
    /// Per-slot determinism classification (derived from fire count variance).
    pub slot_classes: [SlotClass; 8],
}

impl DeterminismReport {
    /// Look up a dimension verdict by name.
    pub fn dimension(&self, name: &str) -> Option<&DimensionVerdict> {
        self.dimensions.iter().find(|d| d.name == name)
    }

    /// Check if a specific dimension is deterministic.
    /// Returns true if the dimension was not measured (no data to compare).
    pub fn is_deterministic(&self, name: &str) -> bool {
        self.dimension(name).map(|d| d.deterministic).unwrap_or(true)
    }
}

impl std::fmt::Display for DeterminismReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.num_reps < 2 {
            return write!(f, "SKIP (only {} rep)", self.num_reps);
        }

        let total = self.dimensions.len();
        let det_count = self.dimensions.iter().filter(|d| d.deterministic).count();
        let varying: Vec<&DimensionVerdict> = self.dimensions.iter().filter(|d| !d.deterministic).collect();

        if self.compute_deterministic {
            write!(f, "PASS (compute deterministic, {} reps, {}/{} dims)", self.num_reps, det_count, total)?;
        } else {
            // List the failing sequence dimensions for quick diagnosis.
            let failing_seqs: Vec<&str> = self
                .dimensions
                .iter()
                .filter(|d| !d.deterministic && d.summary.contains("diverge"))
                .map(|d| d.name.as_str())
                .collect();
            if failing_seqs.is_empty() {
                write!(f, "FAIL ({} reps, {}/{} dims)", self.num_reps, det_count, total)?;
            } else {
                write!(
                    f,
                    "FAIL ({} diverge, {} reps, {}/{} dims)",
                    failing_seqs.join("+"),
                    self.num_reps,
                    det_count,
                    total
                )?;
            }
        }

        if !varying.is_empty() {
            write!(
                f,
                " [varying: {}]",
                varying
                    .iter()
                    .map(|d| format!("{}={}", d.name, d.summary))
                    .collect::<Vec<_>>()
                    .join(", ")
            )?;
        }

        Ok(())
    }
}

/// Align decoded events so the first event is at cycle 0.
fn align_events(events: &[DecodedEvent]) -> Vec<DecodedEvent> {
    if events.is_empty() {
        return Vec::new();
    }
    let base = events[0].abs_cycle;
    events
        .iter()
        .map(|e| DecodedEvent { slot: e.slot, abs_cycle: e.abs_cycle.saturating_sub(base) })
        .collect()
}

/// Count state-change edges for a single slot in an event stream.
///
/// An "edge" is a transition from the slot being absent to present or
/// vice versa across consecutive time steps. This measures behavioral
/// structure independent of duration.
fn count_slot_edges(events: &[DecodedEvent], slot: u8) -> usize {
    if events.is_empty() {
        return 0;
    }

    // Build a set of cycles where this slot fires.
    let mut active_cycles = std::collections::BTreeSet::new();
    for e in events {
        if e.slot == slot {
            active_cycles.insert(e.abs_cycle);
        }
    }
    if active_cycles.is_empty() {
        return 0;
    }

    // Get all unique cycles in the event stream.
    let mut all_cycles = std::collections::BTreeSet::new();
    for e in events {
        all_cycles.insert(e.abs_cycle);
    }

    // Count transitions between active and inactive.
    let mut edges = 0;
    let mut was_active = false;
    for &cycle in &all_cycles {
        let is_active = active_cycles.contains(&cycle);
        if is_active != was_active {
            edges += 1;
            was_active = is_active;
        }
    }

    edges
}

/// Extract per-cycle slot sets from a decoded event stream.
///
/// Returns a vector of (abs_cycle, sorted slot set) pairs in cycle order,
/// with consecutive duplicates collapsed (repeat compression). This is the
/// behavioral transition sequence.
fn extract_transitions(events: &[DecodedEvent]) -> Vec<Vec<u8>> {
    if events.is_empty() {
        return Vec::new();
    }

    let mut transitions = Vec::new();
    let mut i = 0;

    while i < events.len() {
        let cycle = events[i].abs_cycle;
        let mut slots = Vec::new();
        while i < events.len() && events[i].abs_cycle == cycle {
            slots.push(events[i].slot);
            i += 1;
        }
        slots.sort();
        slots.dedup();

        // Collapse consecutive identical slot sets.
        if transitions.last() != Some(&slots) {
            transitions.push(slots);
        }
    }

    transitions
}

/// Extract inter-event deltas from decoded events.
///
/// Returns the cycle delta between each pair of consecutive time-step
/// changes (not individual events, but distinct cycles). This captures
/// instruction-level timing independent of which slots fire.
fn extract_deltas(events: &[DecodedEvent]) -> Vec<u64> {
    let mut unique_cycles: Vec<u64> = Vec::new();
    for e in events {
        if unique_cycles.last() != Some(&e.abs_cycle) {
            unique_cycles.push(e.abs_cycle);
        }
    }
    if unique_cycles.len() < 2 {
        return Vec::new();
    }
    unique_cycles.windows(2).map(|w| w[1].saturating_sub(w[0])).collect()
}

// -------------------------------------------------------------------------
// Dimension extraction helpers
// -------------------------------------------------------------------------

/// Convert a sorted slot set to a bitmask (slot N -> bit N).
fn slot_bitmask(slots: &[u8]) -> u64 {
    let mut mask = 0u64;
    for &s in slots {
        mask |= 1 << s;
    }
    mask
}

/// Extract inter-fire cycle deltas for a single slot.
///
/// Returns the cycle gaps between consecutive fires of the given slot,
/// capturing per-slot timing behavior independent of other slots.
fn extract_slot_deltas(events: &[DecodedEvent], slot: u8) -> Vec<u64> {
    let mut cycles: Vec<u64> = Vec::new();
    for e in events {
        if e.slot == slot {
            if cycles.last() != Some(&e.abs_cycle) {
                cycles.push(e.abs_cycle);
            }
        }
    }
    if cycles.len() < 2 {
        return Vec::new();
    }
    cycles.windows(2).map(|w| w[1].saturating_sub(w[0])).collect()
}

/// Find the longest consecutive run of a slot being active.
///
/// A "run" is a sequence of consecutive global time steps where the slot
/// fires without interruption. Measures duty cycle patterns.
fn max_consecutive_run(events: &[DecodedEvent], slot: u8) -> usize {
    let mut active: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
    for e in events {
        if e.slot == slot {
            active.insert(e.abs_cycle);
        }
    }
    if active.is_empty() {
        return 0;
    }

    let mut all: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
    for e in events {
        all.insert(e.abs_cycle);
    }

    let mut max_run = 0;
    let mut current_run = 0;
    for &cycle in &all {
        if active.contains(&cycle) {
            current_run += 1;
            max_run = max_run.max(current_run);
        } else {
            current_run = 0;
        }
    }
    max_run
}

// -------------------------------------------------------------------------
// Dimension framework
// -------------------------------------------------------------------------

/// Add a scalar dimension: one u64 value per rep, compared for equality.
///
/// Suppresses dimensions where all reps produce zero (non-firing slots).
fn scalar_dim(dims: &mut Vec<DimensionVerdict>, name: &str, values: &[u64]) {
    if values.len() < 2 {
        return;
    }
    // Skip if all values are zero (non-firing slot, uninteresting).
    if values.iter().all(|&v| v == 0) {
        return;
    }
    let deterministic = values.iter().all(|&v| v == values[0]);
    let summary = if deterministic {
        format!("{}", values[0])
    } else {
        let min = *values.iter().min().unwrap();
        let max = *values.iter().max().unwrap();
        format!("{}-{}", min, max)
    };
    dims.push(DimensionVerdict { name: name.to_string(), deterministic, summary });
}

/// Add a sequence dimension: one Vec<u64> per rep, compared element-by-element.
///
/// Uses prefix matching: sequences of different lengths compare their shared
/// prefix. Suppresses dimensions where all reps produce empty sequences.
fn sequence_dim(dims: &mut Vec<DimensionVerdict>, name: &str, sequences: &[Vec<u64>]) {
    if sequences.len() < 2 {
        return;
    }
    if sequences.iter().all(|s| s.is_empty()) {
        return;
    }

    let min_len = sequences.iter().map(|s| s.len()).min().unwrap_or(0);
    let max_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);

    // Compare shared prefix.
    let mut match_count = 0;
    for i in 0..min_len {
        if sequences.iter().all(|s| s[i] == sequences[0][i]) {
            match_count += 1;
        } else {
            break;
        }
    }

    // Prefix match: if all shared elements match, the dimension is
    // deterministic. The shorter trace may have captured less data
    // (trace buffer filled), not different behavior.
    let deterministic = match_count == min_len;
    let summary = if deterministic && min_len == max_len {
        format!("match ({})", max_len)
    } else if deterministic {
        format!("prefix {}/{}", min_len, max_len)
    } else {
        format!("diverge@{}/{}", match_count, max_len)
    };

    dims.push(DimensionVerdict { name: name.to_string(), deterministic, summary });
}

/// Analyze all dimensions from aligned event streams.
///
/// Automatically generates global scalar, global sequence, and per-slot
/// dimensions. Per-slot dimensions are only included for slots that fire
/// in at least one rep.
fn analyze_dimensions(reps: &[Vec<DecodedEvent>], start_times: &[u64]) -> Vec<DimensionVerdict> {
    let mut dims = Vec::new();
    if reps.len() < 2 {
        return dims;
    }

    // --- Start time (pre-alignment, host scheduling jitter) ---
    scalar_dim(&mut dims, "start_time", start_times);

    // --- Global scalar dimensions ---
    let total_events: Vec<u64> = reps.iter().map(|e| e.len() as u64).collect();
    scalar_dim(&mut dims, "total_events", &total_events);

    let unique_cycles: Vec<u64> = reps
        .iter()
        .map(|e| {
            let mut set = std::collections::BTreeSet::new();
            for ev in e {
                set.insert(ev.abs_cycle);
            }
            set.len() as u64
        })
        .collect();
    scalar_dim(&mut dims, "unique_cycles", &unique_cycles);

    let durations: Vec<u64> = reps
        .iter()
        .map(|e| {
            if e.len() < 2 {
                0
            } else {
                e.last().unwrap().abs_cycle - e.first().unwrap().abs_cycle
            }
        })
        .collect();
    scalar_dim(&mut dims, "duration", &durations);

    // --- Global sequence dimensions ---
    let all_deltas: Vec<Vec<u64>> = reps.iter().map(|e| extract_deltas(e)).collect();
    sequence_dim(&mut dims, "deltas", &all_deltas);

    let all_transitions: Vec<Vec<u64>> = reps
        .iter()
        .map(|e| extract_transitions(e).iter().map(|slots| slot_bitmask(slots)).collect())
        .collect();
    sequence_dim(&mut dims, "transitions", &all_transitions);

    let all_skeletons: Vec<Vec<u64>> = reps
        .iter()
        .map(|e| {
            extract_skeleton(e)
                .iter()
                .map(|entry| match entry {
                    SkeletonEntry::Single(s) => 1u64 << *s,
                    SkeletonEntry::Multiple(ss) => slot_bitmask(ss),
                })
                .collect()
        })
        .collect();
    sequence_dim(&mut dims, "skeleton", &all_skeletons);

    // --- Per-slot dimensions (auto-generated for all 8 slots) ---
    for slot in 0..8u8 {
        let fires: Vec<u64> = reps
            .iter()
            .map(|e| e.iter().filter(|ev| ev.slot == slot).count() as u64)
            .collect();

        // Skip slots that never fire in any rep.
        if fires.iter().all(|&f| f == 0) {
            continue;
        }

        let slot_name = format!("slot{}", slot);

        scalar_dim(&mut dims, &format!("{}_fires", slot_name), &fires);

        let edges: Vec<u64> = reps.iter().map(|e| count_slot_edges(e, slot) as u64).collect();
        scalar_dim(&mut dims, &format!("{}_edges", slot_name), &edges);

        let first_cycles: Vec<u64> = reps
            .iter()
            .map(|e| e.iter().find(|ev| ev.slot == slot).map(|ev| ev.abs_cycle).unwrap_or(0))
            .collect();
        scalar_dim(&mut dims, &format!("{}_first_cycle", slot_name), &first_cycles);

        let last_cycles: Vec<u64> = reps
            .iter()
            .map(|e| e.iter().rev().find(|ev| ev.slot == slot).map(|ev| ev.abs_cycle).unwrap_or(0))
            .collect();
        scalar_dim(&mut dims, &format!("{}_last_cycle", slot_name), &last_cycles);

        let max_runs: Vec<u64> = reps.iter().map(|e| max_consecutive_run(e, slot) as u64).collect();
        scalar_dim(&mut dims, &format!("{}_max_run", slot_name), &max_runs);

        let slot_dts: Vec<Vec<u64>> = reps.iter().map(|e| extract_slot_deltas(e, slot)).collect();
        sequence_dim(&mut dims, &format!("{}_deltas", slot_name), &slot_dts);
    }

    dims
}

/// Analyze trace determinism across multiple repetitions.
///
/// Decodes binary traces, aligns to cycle 0, then evaluates every
/// analysis dimension automatically. The `compute_deterministic` verdict
/// is derived from the `deltas` dimension (inter-event cycle timing).
pub fn check_determinism(traces: &[Vec<u8>]) -> DeterminismReport {
    let num_reps = traces.len();

    // Decode all traces.
    let all_events: Vec<Vec<DecodedEvent>> = traces.iter().map(|t| decode_binary_trace(t)).collect();

    // Start times (pre-alignment).
    let start_times: Vec<u64> = all_events
        .iter()
        .map(|events| events.first().map(|e| e.abs_cycle).unwrap_or(0))
        .collect();

    // Align all events to start at cycle 0.
    let aligned: Vec<Vec<DecodedEvent>> = all_events.iter().map(|events| align_events(events)).collect();

    // Classify slots from the aligned events.
    let slot_classes = classify_slots_from_reps(&aligned);

    // Analyze all dimensions.
    let dimensions = analyze_dimensions(&aligned, &start_times);

    // Compute determinism = deltas match across the shared prefix.
    // Inter-event deltas capture instruction-level timing: the cycle gap
    // between successive unique trace ticks. If these are identical across
    // reps, the core executed the same instruction sequence at the same
    // relative timing. All other dimensions are informational diagnostics.
    let compute_deterministic = dimensions
        .iter()
        .find(|d| d.name == "deltas")
        .map(|d| d.deterministic)
        .unwrap_or(true);

    DeterminismReport { num_reps, dimensions, compute_deterministic, slot_classes }
}

/// Write a sweep summary file.
pub fn write_sweep_summary(
    sweep_dir: &Path,
    determinism: &DeterminismReport,
    comparisons: &[TraceComparison],
) -> std::io::Result<()> {
    use std::io::Write;
    let path = sweep_dir.join("sweep_summary.txt");
    let mut f = std::fs::File::create(&path)?;

    writeln!(f, "Determinism: {}", determinism)?;

    for comp in comparisons {
        writeln!(f, "{}", comp)?;
    }
    Ok(())
}

/// Delete large binary trace files after analysis, keeping only text summaries.
///
/// Trace buffers are typically 1MB each; a 13-group sweep with 5 determinism
/// reps produces ~18MB per seed. This removes the .bin files while preserving
/// .txt comparison results and the sweep summary.
pub fn cleanup_trace_binaries(sweep_dir: &Path) {
    if let Ok(entries) = std::fs::read_dir(sweep_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("bin") {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
}

// -------------------------------------------------------------------------
// Multi-group trace merge
// -------------------------------------------------------------------------

/// An event from a merged multi-group trace, identified by hardware event ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergedEvent {
    /// Hardware event ID (0-127), resolved from the trace slot + group config.
    pub event_id: u8,
    /// Absolute cycle from the trace start marker.
    pub abs_cycle: u64,
}

/// Merge decoded traces from multiple groups into a unified event stream.
///
/// Each input pair maps a group (for slot-to-event-ID translation) to its
/// decoded events. The output contains all events with their actual hardware
/// event IDs (not slot numbers), sorted by absolute cycle.
///
/// This produces an "all-event trace" when given traces from all sweep groups.
/// Requires deterministic execution across groups (verified separately).
pub fn merge_sweep_traces(group_traces: &[(&TraceEventGroup, &[DecodedEvent])]) -> Vec<MergedEvent> {
    let mut merged = Vec::new();
    for &(group, events) in group_traces {
        for ev in events {
            let slot = ev.slot as usize;
            if slot < 8 {
                let event_id = group.event_ids[slot];
                // Skip NONE (0) padding slots in the last group.
                if event_id != 0 {
                    merged.push(MergedEvent { event_id, abs_cycle: ev.abs_cycle });
                }
            }
        }
    }
    merged.sort_by_key(|e| e.abs_cycle);
    merged
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_legacy_group_matches_fuzz_template() {
        // Verify legacy default group register values match fuzz_template.py.
        assert_eq!(
            LEGACY_DEFAULT_GROUP.event0_value(),
            640680481,
            "Legacy event0 should be 0x{:08X} (640680481), got 0x{:08X}",
            640680481u32,
            LEGACY_DEFAULT_GROUP.event0_value(),
        );
        assert_eq!(
            LEGACY_DEFAULT_GROUP.event1_value(),
            1330084396,
            "Legacy event1 should be 0x{:08X} (1330084396), got 0x{:08X}",
            1330084396u32,
            LEGACY_DEFAULT_GROUP.event1_value(),
        );
    }

    #[test]
    fn test_event_group_packing() {
        // Verify packing of known event IDs into register values.
        let group = TraceEventGroup { name: "test", event_ids: [1, 2, 3, 4, 5, 6, 7, 8] };
        assert_eq!(group.event0_value(), u32::from_le_bytes([1, 2, 3, 4]));
        assert_eq!(group.event1_value(), u32::from_le_bytes([5, 6, 7, 8]));
    }

    #[test]
    fn test_npu_address() {
        // Tile (0,2), Trace_Event0 offset 0x340E0.
        assert_eq!(npu_address(0, 2, 0x340E0), 0x002340E0);
        // Tile (0,2), Trace_Event1 offset 0x340E4.
        assert_eq!(npu_address(0, 2, 0x340E4), 0x002340E4);
        // Tile (1,3): col=1 -> bit 25, row=3 -> bits 22:20.
        assert_eq!(npu_address(1, 3, 0x340E0), 0x023340E0);
    }

    /// Build a fake Write32 instruction for testing.
    fn make_write32(reg_off: u32, value: u32) -> Vec<u8> {
        let mut instr = Vec::with_capacity(24);
        // Opcode header: opcode=0x00 + 3 pad bytes
        instr.push(0x00);
        instr.extend_from_slice(&[0, 0, 0]);
        // 4 zero bytes
        instr.extend_from_slice(&[0, 0, 0, 0]);
        // reg_off as u64 LE
        instr.extend_from_slice(&reg_off.to_le_bytes());
        instr.extend_from_slice(&0u32.to_le_bytes()); // high 32 bits
                                                      // value as u32 LE
        instr.extend_from_slice(&value.to_le_bytes());
        // size as u32 LE
        instr.extend_from_slice(&24u32.to_le_bytes());
        assert_eq!(instr.len(), 24);
        instr
    }

    #[test]
    fn test_find_write32_value_offset() {
        let event0_addr = npu_address(0, 2, TRACE_EVENT0_OFFSET);
        let event1_addr = npu_address(0, 2, TRACE_EVENT1_OFFSET);

        // Build a fake insts.bin with two Write32 instructions.
        let mut insts = Vec::new();
        insts.extend_from_slice(&make_write32(event0_addr, 0xDEADBEEF));
        insts.extend_from_slice(&make_write32(event1_addr, 0xCAFEBABE));

        let off0 = find_write32_value_offset(&insts, event0_addr);
        assert_eq!(off0, Some(16)); // value at offset 16 in first instruction

        let off1 = find_write32_value_offset(&insts, event1_addr);
        assert_eq!(off1, Some(24 + 16)); // value at offset 16 in second instruction

        // Verify the found offsets point to the correct values.
        let val0 = u32::from_le_bytes([
            insts[off0.unwrap()],
            insts[off0.unwrap() + 1],
            insts[off0.unwrap() + 2],
            insts[off0.unwrap() + 3],
        ]);
        assert_eq!(val0, 0xDEADBEEF);
    }

    #[test]
    fn test_patch_round_trip() {
        let event0_addr = npu_address(0, 2, TRACE_EVENT0_OFFSET);
        let event1_addr = npu_address(0, 2, TRACE_EVENT1_OFFSET);

        // Build fake insts with legacy default values.
        let mut insts = Vec::new();
        insts.extend_from_slice(&make_write32(event0_addr, LEGACY_DEFAULT_GROUP.event0_value()));
        insts.extend_from_slice(&make_write32(event1_addr, LEGACY_DEFAULT_GROUP.event1_value()));

        // Patching with legacy group should be a no-op.
        let patched0 = patch_insts_for_group(&insts, &LEGACY_DEFAULT_GROUP).unwrap();
        assert_eq!(insts, patched0, "Legacy group patch should be identity");

        // Patching with sweep group 0 should change both value fields.
        let patched1 = patch_insts_for_group(&insts, TRACE_EVENT_GROUPS[0]).unwrap();
        assert_ne!(insts, patched1, "Sweep group 0 patch should differ from legacy");

        // Verify the patched values match sweep group 0.
        let val0 = u32::from_le_bytes(patched1[16..20].try_into().unwrap());
        let val1 = u32::from_le_bytes(patched1[24 + 16..24 + 20].try_into().unwrap());
        assert_eq!(val0, TRACE_EVENT_GROUPS[0].event0_value());
        assert_eq!(val1, TRACE_EVENT_GROUPS[0].event1_value());

        // Patch back to legacy.
        let round_trip = patch_insts_for_group(&patched1, &LEGACY_DEFAULT_GROUP).unwrap();
        assert_eq!(insts, round_trip, "Round-trip should restore original");
    }

    #[test]
    fn test_patch_missing_target() {
        // insts.bin with no matching Write32 should return Err.
        let insts = make_write32(0x12345678, 0);
        let result = patch_insts_for_group(&insts, &LEGACY_DEFAULT_GROUP);
        assert!(result.is_err());
    }

    /// Build a trace packet from raw event bytes (helper for tests).
    fn make_trace_packet(event_bytes: &[u8]) -> Vec<u8> {
        // Header (word 0): valid header for tile (2,1), core trace, packet_id=1.
        // Matches the format from real NPU traces: col in [27:21], row in [20:16],
        // type in [13:12], id in [4:0], reserved bits clear, odd parity.
        let header: u32 = 0x00220001; // col=1, row=2, type=0, id=1, popcount=3 (odd)
        let mut packet = Vec::with_capacity(32);
        packet.extend_from_slice(&header.to_le_bytes());

        // Words 1-7: pack event_bytes into LE uint32 words, pad with 0xFE.
        // Event bytes are packed MSB-first within each word, then the word is
        // stored little-endian -- matching real NPU trace buffer format.
        let mut padded = [0xFEu8; 28];
        let copy_len = event_bytes.len().min(28);
        padded[..copy_len].copy_from_slice(&event_bytes[..copy_len]);

        for word_idx in 0..7 {
            let base = word_idx * 4;
            let word =
                u32::from_be_bytes([padded[base], padded[base + 1], padded[base + 2], padded[base + 3]]);
            packet.extend_from_slice(&word.to_le_bytes());
        }

        assert_eq!(packet.len(), 32);
        packet
    }

    #[test]
    fn test_decode_single0_events() {
        // Single0: bit7=0, slot in [6:4], delta in [3:0]
        // slot=1, delta=3: byte = (1 << 4) | 3 = 0x13
        // slot=2, delta=5: byte = (2 << 4) | 5 = 0x25
        let events = vec![
            0xF0, 0, 0, 0, 0, 0, 0, 100,  // Start marker: timer=100
            0x13, // slot=1, delta=3 -> abs=103
            0x25, // slot=2, delta=5 -> abs=108
        ];
        let packet = make_trace_packet(&events);
        let decoded = decode_binary_trace(&packet);
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0], DecodedEvent { slot: 1, abs_cycle: 103 });
        assert_eq!(decoded[1], DecodedEvent { slot: 2, abs_cycle: 108 });
    }

    #[test]
    fn test_decode_single1_event() {
        // Single1 (hardware format): bits[7:5]=100, slot in [4:2], delta in [1:0]+byte1
        // slot=3, delta=500 (0x1F4):
        //   byte0 = 0x80 | (3 << 2) | ((500 >> 8) & 0x03) = 0x80 | 0x0C | 0x01 = 0x8D
        //   byte1 = 500 & 0xFF = 0xF4
        let events = vec![
            0xF0, 0, 0, 0, 0, 0, 0, 0, // Start marker: timer=0
            0x8D, 0xF4, // slot=3, delta=500 -> abs=500
        ];
        let packet = make_trace_packet(&events);
        let decoded = decode_binary_trace(&packet);
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0], DecodedEvent { slot: 3, abs_cycle: 500 });
    }

    #[test]
    fn test_decode_single2_event() {
        // Single2 (hardware format): bits[7:5]=101, slot in [4:2], delta in [1:0]+byte1+byte2
        // slot=5, delta=100000 (0x186A0):
        //   byte0 = 0xA0 | (5 << 2) | ((100000 >> 16) & 0x03) = 0xA0 | 0x14 | 0x01 = 0xB5
        //   byte1 = (100000 >> 8) & 0xFF = 0x86
        //   byte2 = 100000 & 0xFF = 0xA0
        let events = vec![
            0xF0, 0, 0, 0, 0, 0, 0, 0, // Start marker: timer=0
            0xB5, 0x86, 0xA0, // slot=5, delta=100000 -> abs=100000
        ];
        let packet = make_trace_packet(&events);
        let decoded = decode_binary_trace(&packet);
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0], DecodedEvent { slot: 5, abs_cycle: 100000 });
    }

    #[test]
    fn test_decode_empty_trace() {
        // Packet with only padding bytes.
        let events = vec![0xFE; 28];
        let packet = make_trace_packet(&events);
        let decoded = decode_binary_trace(&packet);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_decode_empty_buffer() {
        let decoded = decode_binary_trace(&[]);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_decode_trims_padding() {
        // One packet of real data followed by padding sentinel + zeros.
        let events = vec![
            0xF0, 0, 0, 0, 0, 0, 0, 100,  // Start marker: timer=100
            0x13, // slot=1, delta=3
        ];
        let mut buf = make_trace_packet(&events);

        // Append 0xFEFEFEFE sentinel + two zero words (as little-endian u32s).
        buf.extend_from_slice(&0xFEFEFEFEu32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());

        // Append a second packet full of zeros (would decode as fake events).
        buf.extend_from_slice(&[0u8; 32]);

        // Without trimming we'd decode the zeros as slot-0 events.
        // With trimming, only the real packet is decoded.
        let decoded = decode_binary_trace(&buf);
        assert_eq!(decoded.len(), 1, "Expected 1 real event, got {}", decoded.len());
        assert_eq!(decoded[0].slot, 1);
    }

    #[test]
    fn test_compare_matching() {
        let events = vec![
            DecodedEvent { slot: 1, abs_cycle: 100 },
            DecodedEvent { slot: 3, abs_cycle: 200 },
            DecodedEvent { slot: 5, abs_cycle: 300 },
        ];
        // Same slots, different cycles (cycles are ignored in comparison).
        let emu_events = vec![
            DecodedEvent { slot: 1, abs_cycle: 50 },
            DecodedEvent { slot: 3, abs_cycle: 150 },
            DecodedEvent { slot: 5, abs_cycle: 250 },
        ];
        let comp = compare_event_sequences(&events, &emu_events, "test");
        assert!(comp.sequence_match);
        assert!(comp.first_divergence.is_none());
        assert_eq!(comp.npu_event_count, 3);
        assert_eq!(comp.emu_event_count, 3);
    }

    #[test]
    fn test_compare_divergent_slot() {
        let npu = vec![DecodedEvent { slot: 1, abs_cycle: 100 }, DecodedEvent { slot: 3, abs_cycle: 200 }];
        let emu = vec![
            DecodedEvent { slot: 1, abs_cycle: 100 },
            DecodedEvent { slot: 4, abs_cycle: 200 }, // slot mismatch
        ];
        let comp = compare_event_sequences(&npu, &emu, "test");
        assert!(!comp.sequence_match);
        assert_eq!(comp.first_divergence, Some((1, 3, 4)));
    }

    #[test]
    fn test_compare_divergent_length() {
        let npu = vec![
            DecodedEvent { slot: 1, abs_cycle: 100 },
            DecodedEvent { slot: 3, abs_cycle: 200 },
            DecodedEvent { slot: 5, abs_cycle: 300 },
        ];
        let emu = vec![DecodedEvent { slot: 1, abs_cycle: 100 }, DecodedEvent { slot: 3, abs_cycle: 200 }];
        let comp = compare_event_sequences(&npu, &emu, "test");
        assert!(!comp.sequence_match);
        assert_eq!(comp.first_divergence, Some((2, 5, u8::MAX)));
    }

    #[test]
    fn test_compare_both_empty() {
        let comp = compare_event_sequences(&[], &[], "test");
        assert!(comp.sequence_match);
        assert!(comp.first_divergence.is_none());
    }

    // --- Trim tests ---

    #[test]
    fn test_trim_finds_sentinel() {
        // 16 bytes of data, then 0xFEFEFEFE + two 0x00000000.
        let mut buf = vec![0xABu8; 16];
        buf.extend_from_slice(&0xFEFEFEFEu32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 100]); // trailing zeros

        // trim includes the sentinel word itself: 16 + 4 = 20 bytes.
        assert_eq!(trim_trace_buffer(&buf), 20);
    }

    #[test]
    fn test_trim_no_sentinel_returns_full() {
        let buf = vec![0x42u8; 64];
        assert_eq!(trim_trace_buffer(&buf), 64);
    }

    #[test]
    fn test_trim_empty() {
        assert_eq!(trim_trace_buffer(&[]), 0);
    }

    #[test]
    fn test_trim_fefe_without_zeros_not_trimmed() {
        // 0xFEFEFEFE followed by non-zero data: not a trim point.
        let mut buf = vec![0xABu8; 16];
        buf.extend_from_slice(&0xFEFEFEFEu32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // non-zero
        buf.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(trim_trace_buffer(&buf), buf.len());
    }

    // --- Skeleton + determinism tests ---

    #[test]
    fn test_skeleton_collapses_repeats() {
        // Same slot firing at consecutive cycles should collapse to one entry.
        let events = vec![
            DecodedEvent { slot: 1, abs_cycle: 100 },
            DecodedEvent { slot: 1, abs_cycle: 101 },
            DecodedEvent { slot: 1, abs_cycle: 102 },
            DecodedEvent { slot: 2, abs_cycle: 103 },
        ];
        let skel = extract_skeleton(&events);
        assert_eq!(skel.len(), 2);
        assert_eq!(skel[0], SkeletonEntry::Single(1));
        assert_eq!(skel[1], SkeletonEntry::Single(2));
    }

    #[test]
    fn test_skeleton_groups_simultaneous() {
        // Multiple slots at the same cycle -> Multiple entry.
        let events = vec![
            DecodedEvent { slot: 5, abs_cycle: 100 },
            DecodedEvent { slot: 7, abs_cycle: 100 },
            DecodedEvent { slot: 7, abs_cycle: 101 },
        ];
        let skel = extract_skeleton(&events);
        assert_eq!(skel.len(), 2);
        assert_eq!(skel[0], SkeletonEntry::Multiple(vec![5, 7]));
        assert_eq!(skel[1], SkeletonEntry::Single(7));
    }

    #[test]
    fn test_determinism_pass_same_transitions_different_start_time() {
        // Same event transitions and deltas, different absolute start timer.
        // This is the real-world case: host scheduling jitter shifts the
        // start time but the kernel executes identically.
        let trace_a = make_trace_packet(&[
            0xF0, 0, 0, 0, 0, 0, 0, 100,  // timer=100
            0x13, // slot=1, delta=3
            0x25, // slot=2, delta=5
        ]);
        let trace_b = make_trace_packet(&[
            0xF0, 0, 0, 0, 0, 0, 0, 200,  // timer=200 (different start!)
            0x13, // slot=1, delta=3 (same)
            0x25, // slot=2, delta=5 (same)
        ]);
        let report = check_determinism(&[trace_a, trace_b]);
        assert!(report.is_deterministic("transitions"));
        assert!(report.is_deterministic("deltas"));
        assert!(report.compute_deterministic);
        assert!(!report.is_deterministic("start_time"), "start time should vary");
    }

    #[test]
    fn test_determinism_fail_different_deltas() {
        // Same transitions but different inter-event deltas: this is a real
        // timing difference in instruction execution, not just DMA jitter.
        let trace_a = make_trace_packet(&[
            0xF0, 0, 0, 0, 0, 0, 0, 0, 0x13, // slot=1, delta=3
            0x25, // slot=2, delta=5
        ]);
        let trace_b = make_trace_packet(&[
            0xF0, 0, 0, 0, 0, 0, 0, 0, 0x1A, // slot=1, delta=10 (different!)
            0x2F, // slot=2, delta=15 (different!)
        ]);
        let report = check_determinism(&[trace_a, trace_b]);
        assert!(report.is_deterministic("transitions"), "transitions should still match");
        assert!(!report.is_deterministic("deltas"), "deltas should NOT match");
        assert!(!report.compute_deterministic, "compute determinism requires matching deltas");
    }

    #[test]
    fn test_determinism_pass_prefix_match() {
        // One trace has fewer events but is a prefix of the other.
        let trace_long = make_trace_packet(&[
            0xF0, 0, 0, 0, 0, 0, 0, 0, 0x13, // slot=1
            0x25, // slot=2
            0x37, // slot=3
        ]);
        let trace_short = make_trace_packet(&[
            0xF0, 0, 0, 0, 0, 0, 0, 0, 0x13, // slot=1
            0x25, // slot=2
        ]);
        let report = check_determinism(&[trace_long, trace_short]);
        assert!(report.compute_deterministic);
    }

    #[test]
    fn test_determinism_fail_different_deltas_real() {
        // Two traces with the same number of events but different timing.
        // This represents a genuine instruction-level divergence.
        let trace_a = make_trace_packet(&[
            0xF0, 0, 0, 0, 0, 0, 0, 0, 0x13, // slot=1, delta=3
            0x25, // slot=2, delta=5 -> delta between events = 5
        ]);
        let trace_b = make_trace_packet(&[
            0xF0, 0, 0, 0, 0, 0, 0, 0, 0x13, // slot=1, delta=3
            0x29, // slot=2, delta=9 -> delta between events = 9 (different!)
        ]);
        let report = check_determinism(&[trace_a, trace_b]);
        assert!(!report.is_deterministic("deltas"));
        assert!(!report.compute_deterministic);
        let display = format!("{}", report);
        assert!(display.contains("FAIL"));
    }

    #[test]
    fn test_determinism_different_slots_no_deltas_is_inconclusive() {
        // Single-event traces with different slots: transitions diverge
        // but there are no deltas to compare, so compute_deterministic
        // is vacuously true (insufficient data to determine).
        let trace_a = make_trace_packet(&[
            0xF0, 0, 0, 0, 0, 0, 0, 0, 0x13, // slot=1
        ]);
        let trace_b = make_trace_packet(&[
            0xF0, 0, 0, 0, 0, 0, 0, 0, 0x23, // slot=2 (different!)
        ]);
        let report = check_determinism(&[trace_a, trace_b]);
        assert!(!report.is_deterministic("transitions"), "transitions should diverge");
        // With only one event each, there are no inter-event deltas.
        assert!(report.compute_deterministic, "no deltas to compare = vacuously true");
    }

    #[test]
    fn test_determinism_single_trace() {
        let trace = make_trace_packet(&[0xF0, 0, 0, 0, 0, 0, 0, 42, 0x13]);
        let report = check_determinism(&[trace]);
        assert!(report.compute_deterministic);
        assert_eq!(report.num_reps, 1);
        assert!(report.dimensions.is_empty(), "single rep produces no dimensions");
    }

    #[test]
    fn test_determinism_compensates_dma_latency() {
        // Simulates DMA latency variation: same skeleton but different repeat
        // counts for the initial stall phase. Both traces show:
        //   Multiple[5,7] (stall+active) -> Single[7] (active only)
        // but with different numbers of repeated stall cycles.

        // Trace A: stall phase with delta=0, repeated via repeat0.
        let trace_a = make_trace_packet(&[
            0xF0, 0, 0, 0, 0, 0, 0, 0, // START timer=0
            0xCA, 0x00, // Multiple0: slots 5+7, delta=0
            0xE5, // Repeat0 x5
            0x70, // Single0: slot=7, delta=0
            0xE3, // Repeat0 x3
        ]);
        // Trace B: same pattern but longer stall (more repeats).
        let trace_b = make_trace_packet(&[
            0xF0, 0, 0, 0, 0, 0, 0, 0, // START timer=0
            0xCA, 0x00, // Multiple0: slots 5+7, delta=0
            0xEF, // Repeat0 x15  (much longer stall!)
            0x70, // Single0: slot=7, delta=0
            0xE3, // Repeat0 x3
        ]);
        let report = check_determinism(&[trace_a, trace_b]);
        assert!(report.is_deterministic("transitions"), "transitions should match");
        assert!(report.compute_deterministic, "compute should be deterministic");
        // Slot 5 (LOCK_STALL proxy) should have different fire counts.
        assert!(
            !report.is_deterministic("slot5_fires"),
            "slot 5 fire counts should differ due to repeat count variance"
        );
    }

    #[test]
    fn test_determinism_report_per_slot_verdicts() {
        // Two traces where slot 1 fires identically but slot 2 fires different
        // amounts (via repeat count variation).
        let trace_a = make_trace_packet(&[
            0xF0, 0, 0, 0, 0, 0, 0, 0, 0x11, // slot=1, delta=1
            0x21, // slot=2, delta=1
            0xE3, // Repeat0 x3 (replays slot=2 three more times)
        ]);
        let trace_b = make_trace_packet(&[
            0xF0, 0, 0, 0, 0, 0, 0, 0, 0x11, // slot=1, delta=1
            0x21, // slot=2, delta=1
            0xE7, // Repeat0 x7 (replays slot=2 seven more times)
        ]);
        let report = check_determinism(&[trace_a, trace_b]);

        // Transitions should match: [slot1] -> [slot2] in both.
        assert!(report.is_deterministic("transitions"));
        assert!(report.compute_deterministic);

        // Slot 1 fires exactly once in both.
        assert!(report.is_deterministic("slot1_fires"), "slot 1 should be count-deterministic");

        // Slot 2 fires different counts due to repeat variation.
        assert!(!report.is_deterministic("slot2_fires"), "slot 2 should NOT be count-deterministic");
    }

    #[test]
    fn test_dimension_framework_auto_generates_per_slot() {
        // Verify that the dimension framework produces per-slot dimensions
        // for active slots and skips inactive ones.
        let trace_a = make_trace_packet(&[
            0xF0, 0, 0, 0, 0, 0, 0, 0, 0x13, // slot=1, delta=3
            0x25, // slot=2, delta=5
            0x13, // slot=1, delta=3
        ]);
        let trace_b = make_trace_packet(&[
            0xF0, 0, 0, 0, 0, 0, 0, 0, 0x13, // slot=1, delta=3
            0x25, // slot=2, delta=5
            0x13, // slot=1, delta=3
        ]);
        let report = check_determinism(&[trace_a, trace_b]);

        // Slot 1 and 2 should have dimensions.
        assert!(report.dimension("slot1_fires").is_some(), "slot1 fires should exist");
        assert!(report.dimension("slot2_fires").is_some(), "slot2 fires should exist");
        assert!(report.dimension("slot1_edges").is_some(), "slot1 edges should exist");
        assert!(report.dimension("slot1_max_run").is_some(), "slot1 max_run should exist");
        assert!(report.dimension("slot1_deltas").is_some(), "slot1 deltas should exist");

        // Slot 0 never fires, should have no dimensions.
        assert!(report.dimension("slot0_fires").is_none(), "slot0 should be absent");

        // Global dimensions should exist.
        assert!(report.dimension("deltas").is_some());
        assert!(report.dimension("transitions").is_some());
        assert!(report.dimension("skeleton").is_some());
        assert!(report.dimension("total_events").is_some());
        assert!(report.dimension("duration").is_some());
    }

    #[test]
    fn test_dimension_display_shows_varying() {
        // Verify the Display impl lists varying dimensions.
        let trace_a = make_trace_packet(&[
            0xF0, 0, 0, 0, 0, 0, 0, 100, 0x13, // slot=1, delta=3
        ]);
        let trace_b = make_trace_packet(&[
            0xF0, 0, 0, 0, 0, 0, 0, 200, 0x13, // slot=1, delta=3
        ]);
        let report = check_determinism(&[trace_a, trace_b]);
        let display = format!("{}", report);
        assert!(display.contains("PASS"));
        assert!(display.contains("start_time="), "should show varying start_time");
    }

    #[test]
    fn test_display_match() {
        let comp = TraceComparison {
            group_name: "default".to_string(),
            npu_event_count: 42,
            emu_event_count: 42,
            sequence_match: true,
            first_divergence: None,
            filtered_count: 0,
        };
        let s = format!("{}", comp);
        assert!(s.contains("MATCH"));
        assert!(s.contains("42"));
    }

    #[test]
    fn test_display_diverge() {
        let comp = TraceComparison {
            group_name: "test".to_string(),
            npu_event_count: 10,
            emu_event_count: 8,
            sequence_match: false,
            first_divergence: Some((5, 3, 4)),
            filtered_count: 0,
        };
        let s = format!("{}", comp);
        assert!(s.contains("DIVERGE at index 5"));
    }

    // --- Event catalogue tests ---

    #[test]
    fn test_sweepable_count() {
        let ids = sweepable_event_ids();
        assert_eq!(ids.len(), 100, "Expected 100 sweepable events, got {}", ids.len());
        // Verify they're in ascending order.
        for w in ids.windows(2) {
            assert!(w[0] < w[1], "Sweepable IDs not sorted: {} >= {}", w[0], w[1]);
        }
    }

    #[test]
    fn test_excluded_events_not_sweepable() {
        // NONE, TRUE, RESERVED, all GROUP_* meta-events.
        let excluded = [0, 1, 2, 15, 22, 32, 46, 47, 54, 73, 106, 123];
        for id in excluded {
            assert!(
                !CORE_EVENTS[id].sweepable,
                "Event {} ({}) should not be sweepable",
                id, CORE_EVENTS[id].name,
            );
        }
        // Config-dependent events.
        for id in 5..=8 {
            assert!(!CORE_EVENTS[id].sweepable, "PERF_CNT_{}", id - 5);
        }
        for id in 9..=12 {
            assert!(!CORE_EVENTS[id].sweepable, "COMBO_{}", id - 9);
        }
        for id in 13..=14 {
            assert!(!CORE_EVENTS[id].sweepable, "EDGE_{}", id - 13);
        }
        for id in 16..=21 {
            assert!(!CORE_EVENTS[id].sweepable, "PC_{}", id - 16);
        }
    }

    #[test]
    fn test_sweep_groups_cover_all_sweepable() {
        let expected = sweepable_event_ids();
        let mut actual: Vec<u8> = Vec::new();
        for group in TRACE_EVENT_GROUPS {
            for &id in &group.event_ids {
                if id != 0 {
                    // Skip NONE padding
                    actual.push(id);
                }
            }
        }
        assert_eq!(
            actual,
            expected,
            "Sweep groups should cover all sweepable events in order.\n\
             Missing: {:?}\nExtra: {:?}",
            expected.iter().filter(|id| !actual.contains(id)).collect::<Vec<_>>(),
            actual.iter().filter(|id| !expected.contains(id)).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_event_name_lookup() {
        assert_eq!(event_name(0), "NONE");
        assert_eq!(event_name(28), "ACTIVE");
        assert_eq!(event_name(37), "INSTR_VECTOR");
        assert_eq!(event_name(127), "USER_EVENT_3");
    }

    #[test]
    fn test_num_groups_consistent() {
        assert_eq!(TRACE_EVENT_GROUPS.len(), NUM_GROUPS);
        assert_eq!(NUM_GROUPS, 13);
    }

    // --- Merge tests ---

    #[test]
    fn test_merge_sweep_traces() {
        // Two groups, each with some events at different cycles.
        let group_a = TraceEventGroup { name: "a", event_ids: [10, 20, 30, 40, 50, 60, 70, 80] };
        let group_b = TraceEventGroup { name: "b", event_ids: [11, 21, 31, 41, 51, 61, 71, 81] };
        let events_a = vec![
            DecodedEvent { slot: 0, abs_cycle: 100 }, // event_id=10
            DecodedEvent { slot: 2, abs_cycle: 300 }, // event_id=30
        ];
        let events_b = vec![
            DecodedEvent { slot: 1, abs_cycle: 200 }, // event_id=21
            DecodedEvent { slot: 3, abs_cycle: 400 }, // event_id=41
        ];
        let merged = merge_sweep_traces(&[(&group_a, &events_a), (&group_b, &events_b)]);
        assert_eq!(merged.len(), 4);
        // Sorted by abs_cycle.
        assert_eq!(merged[0], MergedEvent { event_id: 10, abs_cycle: 100 });
        assert_eq!(merged[1], MergedEvent { event_id: 21, abs_cycle: 200 });
        assert_eq!(merged[2], MergedEvent { event_id: 30, abs_cycle: 300 });
        assert_eq!(merged[3], MergedEvent { event_id: 41, abs_cycle: 400 });
    }

    #[test]
    fn test_merge_skips_none_padding() {
        // Group with NONE padding slots (like SWEEP_12).
        let group = TraceEventGroup { name: "padded", event_ids: [124, 125, 0, 0, 0, 0, 0, 0] };
        let events = vec![
            DecodedEvent { slot: 0, abs_cycle: 100 }, // event_id=124
            DecodedEvent { slot: 1, abs_cycle: 200 }, // event_id=125
            DecodedEvent { slot: 2, abs_cycle: 300 }, // event_id=0 -> skipped
        ];
        let merged = merge_sweep_traces(&[(&group, &events)]);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].event_id, 124);
        assert_eq!(merged[1].event_id, 125);
    }

    // --- Slot classification and canonicalization tests ---

    #[test]
    fn test_classify_slots_deterministic() {
        // All reps have the same fire counts -> Deterministic.
        let reps = vec![
            vec![
                DecodedEvent { slot: 0, abs_cycle: 10 },
                DecodedEvent { slot: 0, abs_cycle: 20 },
                DecodedEvent { slot: 1, abs_cycle: 15 },
            ],
            vec![
                DecodedEvent { slot: 0, abs_cycle: 10 },
                DecodedEvent { slot: 0, abs_cycle: 20 },
                DecodedEvent { slot: 1, abs_cycle: 15 },
            ],
        ];
        let classes = classify_slots_from_reps(&reps);
        assert_eq!(classes[0], SlotClass::Deterministic);
        assert_eq!(classes[1], SlotClass::Deterministic);
        assert_eq!(classes[2], SlotClass::Inactive);
    }

    #[test]
    fn test_classify_slots_timing_dependent() {
        // Slot 5 fires different amounts across reps -> TimingDependent.
        let reps = vec![
            vec![DecodedEvent { slot: 5, abs_cycle: 10 }, DecodedEvent { slot: 5, abs_cycle: 20 }],
            vec![
                DecodedEvent { slot: 5, abs_cycle: 10 },
                DecodedEvent { slot: 5, abs_cycle: 20 },
                DecodedEvent { slot: 5, abs_cycle: 30 },
                DecodedEvent { slot: 5, abs_cycle: 40 },
                DecodedEvent { slot: 5, abs_cycle: 50 },
            ],
        ];
        let classes = classify_slots_from_reps(&reps);
        assert_eq!(classes[5], SlotClass::TimingDependent);
    }

    #[test]
    fn test_classify_slots_boundary() {
        // Slot 2 fires 0 or 1 times across reps -> Boundary.
        let reps = vec![
            vec![DecodedEvent { slot: 2, abs_cycle: 10 }],
            vec![], // slot 2 doesn't fire
            vec![DecodedEvent { slot: 2, abs_cycle: 30 }],
        ];
        let classes = classify_slots_from_reps(&reps);
        assert_eq!(classes[2], SlotClass::Boundary);
    }

    #[test]
    fn test_classify_slots_inactive() {
        // Slot 6 never fires in any rep -> Inactive.
        let reps = vec![
            vec![DecodedEvent { slot: 0, abs_cycle: 10 }],
            vec![DecodedEvent { slot: 0, abs_cycle: 20 }],
        ];
        let classes = classify_slots_from_reps(&reps);
        assert_eq!(classes[6], SlotClass::Inactive);
    }

    #[test]
    fn test_canonicalize_filters_timing_dependent() {
        // 9 events total: 6 from timing-dependent slot 5, 3 from deterministic slot 1.
        let events = vec![
            DecodedEvent { slot: 5, abs_cycle: 100 },
            DecodedEvent { slot: 1, abs_cycle: 105 },
            DecodedEvent { slot: 5, abs_cycle: 110 },
            DecodedEvent { slot: 1, abs_cycle: 115 },
            DecodedEvent { slot: 5, abs_cycle: 120 },
            DecodedEvent { slot: 5, abs_cycle: 125 },
            DecodedEvent { slot: 5, abs_cycle: 130 },
            DecodedEvent { slot: 5, abs_cycle: 135 },
            DecodedEvent { slot: 1, abs_cycle: 140 },
        ];
        let mut classes = [SlotClass::Inactive; 8];
        classes[1] = SlotClass::Deterministic;
        classes[5] = SlotClass::TimingDependent;

        let canon = canonicalize(&events, &classes);
        assert_eq!(canon.events.len(), 3, "Should keep only slot 1 events");
        assert_eq!(canon.filtered_counts[5], 6, "6 slot-5 events filtered");
        assert_eq!(canon.filtered_counts[1], 0, "No slot-1 events filtered");
        // All kept events should be slot 1.
        for e in &canon.events {
            assert_eq!(e.slot, 1);
        }
    }

    #[test]
    fn test_canonicalize_aligns_to_zero() {
        let events = vec![
            DecodedEvent { slot: 0, abs_cycle: 5000 },
            DecodedEvent { slot: 0, abs_cycle: 5100 },
            DecodedEvent { slot: 0, abs_cycle: 5200 },
        ];
        let mut classes = [SlotClass::Inactive; 8];
        classes[0] = SlotClass::Deterministic;

        let canon = canonicalize(&events, &classes);
        assert_eq!(canon.events[0].abs_cycle, 0, "First event should be at cycle 0");
        assert_eq!(canon.events[1].abs_cycle, 100);
        assert_eq!(canon.events[2].abs_cycle, 200);
    }

    #[test]
    fn test_compare_canonical_ignores_stalls() {
        // NPU and emulator produce the same deterministic events (slot 1)
        // but NPU also has non-deterministic stall events (slot 5).
        // Canonical comparison should MATCH.
        let npu_events = vec![
            DecodedEvent { slot: 5, abs_cycle: 100 }, // stall (filtered)
            DecodedEvent { slot: 1, abs_cycle: 105 },
            DecodedEvent { slot: 5, abs_cycle: 110 }, // stall (filtered)
            DecodedEvent { slot: 1, abs_cycle: 115 },
            DecodedEvent { slot: 5, abs_cycle: 120 }, // stall (filtered)
        ];
        let emu_events =
            vec![DecodedEvent { slot: 1, abs_cycle: 10 }, DecodedEvent { slot: 1, abs_cycle: 20 }];
        let mut classes = [SlotClass::Inactive; 8];
        classes[1] = SlotClass::Deterministic;
        classes[5] = SlotClass::TimingDependent;

        let (comp, filtered) = compare_canonical(&npu_events, &emu_events, &classes, "test");
        assert!(comp.sequence_match, "Should match on deterministic slots only");
        assert_eq!(comp.npu_event_count, 2, "NPU canonical: 2 slot-1 events");
        assert_eq!(comp.emu_event_count, 2, "EMU canonical: 2 slot-1 events");
        assert_eq!(filtered, 3, "3 NPU stall events filtered");
        assert!(comp.filtered_count > 0, "filtered_count should be set");
    }
}
