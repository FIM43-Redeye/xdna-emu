//! Perfetto-compatible trace exporter.
//!
//! Converts the emulator's event log into Chrome Trace Event Format JSON,
//! compatible with mlir-aie's trace parser (`python/utils/trace/parse.py`)
//! so that emulator and hardware traces can be opened side-by-side in
//! [Perfetto UI](https://ui.perfetto.dev/).
//!
//! # Format
//!
//! The output is a JSON array of trace event objects. Each object has:
//! - `name`: Event name (matches hardware enum names, e.g. `INSTR_VECTOR`)
//! - `ph`: Phase (`"B"` = begin, `"E"` = end, `"M"` = metadata)
//! - `pid`: Process ID (unique per tile + trace type)
//! - `tid`: Thread ID (event slot 0-7)
//! - `ts`: Timestamp (cycle count, treated as microseconds)
//! - `args`: Arbitrary metadata
//!
//! # Tile-to-PID mapping
//!
//! Following mlir-aie conventions:
//! - Core trace tiles get sequential PIDs starting from 0
//! - Memory module traces get PIDs after all core traces
//! - Each tile's events are assigned to thread IDs 0-7 based on event type
//!
//! # Usage
//!
//! ```ignore
//! use xdna_emu::trace::export_perfetto;
//! use std::fs::File;
//!
//! let events = engine.trace_log();
//! let mut file = File::create("trace.json")?;
//! export_perfetto(events, &mut file)?;
//! ```

pub mod vcd;

use crate::interpreter::engine::TileTracedEvent;
use crate::interpreter::state::EventType;
use std::collections::BTreeMap;
use std::io::Write;

/// Event category for PID grouping (matches mlir-aie PacketType).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum TraceType {
    Core = 0,
    Mem = 1,
}

/// Map an EventType to its trace type (Core or Memory module).
fn event_trace_type(event: &EventType) -> TraceType {
    match event {
        // Instruction and stall events are Core module traces
        EventType::InstrVector { .. }
        | EventType::InstrLoad { .. }
        | EventType::InstrStore { .. }
        | EventType::InstrCall { .. }
        | EventType::InstrReturn { .. }
        | EventType::InstrLockAcquireReq { .. }
        | EventType::InstrLockReleaseReq { .. }
        | EventType::InstrStreamGet { .. }
        | EventType::InstrStreamPut { .. }
        | EventType::MemoryStall { .. }
        | EventType::LockStall { .. }
        | EventType::StreamStall { .. }
        | EventType::CoreActive
        | EventType::CoreDisabled
        | EventType::BranchTaken { .. } => TraceType::Core,

        // DMA and lock events are Memory module traces
        EventType::DmaStartTask { .. }
        | EventType::DmaFinishedBd { .. }
        | EventType::DmaFinishedTask { .. }
        | EventType::DmaStalledLock { .. }
        | EventType::DmaStreamStarvation { .. }
        | EventType::LockAcquire { .. }
        | EventType::LockRelease { .. } => TraceType::Mem,
    }
}

/// Map an EventType to its hardware event name string.
/// These names match the AIE2 enum names in mlir-aie's trace event
/// definitions (`aie/utils/trace/events/aie2.py`).
fn event_name(event: &EventType) -> &'static str {
    match event {
        EventType::InstrVector { .. } => "INSTR_VECTOR",
        EventType::InstrLoad { .. } => "INSTR_LOAD",
        EventType::InstrStore { .. } => "INSTR_STORE",
        EventType::InstrCall { .. } => "INSTR_CALL",
        EventType::InstrReturn { .. } => "INSTR_RETURN",
        EventType::InstrLockAcquireReq { .. } => "INSTR_LOCK_ACQUIRE_REQ",
        EventType::InstrLockReleaseReq { .. } => "INSTR_LOCK_RELEASE_REQ",
        EventType::InstrStreamGet { .. } => "INSTR_STREAM_GET",
        EventType::InstrStreamPut { .. } => "INSTR_STREAM_PUT",
        EventType::MemoryStall { .. } => "MEMORY_STALL",
        EventType::LockStall { .. } => "LOCK_STALL",
        EventType::StreamStall { .. } => "STREAM_STALL",
        EventType::DmaStartTask { .. } => "DMA_START_TASK",
        EventType::DmaFinishedBd { .. } => "DMA_FINISHED_BD",
        EventType::DmaFinishedTask { .. } => "DMA_FINISHED_TASK",
        EventType::DmaStalledLock { .. } => "DMA_STALLED_LOCK",
        EventType::DmaStreamStarvation { .. } => "DMA_STREAM_STARVATION",
        EventType::LockAcquire { .. } => "LOCK_ACQ",
        EventType::LockRelease { .. } => "LOCK_REL",
        EventType::CoreActive => "ACTIVE",
        EventType::CoreDisabled => "DISABLED",
        EventType::BranchTaken { .. } => "BRANCH_TAKEN",
    }
}

/// Assign a stable thread ID (0-7) for a given event type within a tile.
/// This determines which "lane" the event appears on in Perfetto's timeline.
/// We group similar events on the same tid for visual clarity.
fn event_tid(event: &EventType) -> u32 {
    match event {
        // Core module: spread across tids 0-6
        EventType::InstrVector { .. } => 0,
        EventType::InstrLoad { .. } => 1,
        EventType::InstrStore { .. } => 2,
        EventType::InstrCall { .. } | EventType::InstrReturn { .. } => 3,
        EventType::InstrLockAcquireReq { .. } | EventType::InstrLockReleaseReq { .. } => 4,
        EventType::InstrStreamGet { .. } | EventType::InstrStreamPut { .. } => 5,
        EventType::MemoryStall { .. } | EventType::LockStall { .. }
        | EventType::StreamStall { .. } => 6,
        EventType::CoreActive | EventType::CoreDisabled | EventType::BranchTaken { .. } => 7,

        // Memory module: spread across tids 0-4
        EventType::DmaStartTask { .. } => 0,
        EventType::DmaFinishedBd { .. } | EventType::DmaFinishedTask { .. } => 1,
        EventType::DmaStalledLock { .. } | EventType::DmaStreamStarvation { .. } => 2,
        EventType::LockAcquire { .. } => 3,
        EventType::LockRelease { .. } => 4,
    }
}

/// Key for identifying a unique tile + trace type combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct TileTraceKey {
    trace_type: TraceType,
    col: u8,
    row: u8,
}

/// Export trace events as Perfetto-compatible JSON.
///
/// The output format matches mlir-aie's trace parser so emulator and hardware
/// traces can be compared directly in Perfetto UI.
///
/// Events are grouped by tile and trace type (Core vs Memory module).
/// Each group gets a unique PID, with metadata events naming the process
/// and threads.
pub fn export_perfetto(
    events: &[TileTracedEvent],
    output: &mut dyn Write,
) -> std::io::Result<()> {
    // Discover all unique tile+trace_type combinations and assign PIDs.
    // BTreeMap gives deterministic ordering.
    let mut pid_map: BTreeMap<TileTraceKey, u32> = BTreeMap::new();
    for evt in events {
        let key = TileTraceKey {
            trace_type: event_trace_type(&evt.event),
            col: evt.col,
            row: evt.row,
        };
        let next_pid = pid_map.len() as u32;
        pid_map.entry(key).or_insert(next_pid);
    }

    // Discover which tids are used per PID (for thread_name metadata)
    let mut tid_names: BTreeMap<(u32, u32), &'static str> = BTreeMap::new();
    for evt in events {
        let key = TileTraceKey {
            trace_type: event_trace_type(&evt.event),
            col: evt.col,
            row: evt.row,
        };
        let pid = pid_map[&key];
        let tid = event_tid(&evt.event);
        tid_names.entry((pid, tid)).or_insert(event_name(&evt.event));
    }

    output.write_all(b"[\n")?;
    let mut first = true;

    // Write metadata events: process_name for each tile
    for (key, &pid) in &pid_map {
        let trace_type_name = match key.trace_type {
            TraceType::Core => "core",
            TraceType::Mem => "mem",
        };
        if !first { output.write_all(b",\n")?; }
        first = false;
        write!(
            output,
            r#"{{"name":"process_name","ph":"M","pid":{},"args":{{"name":"{}_trace for tile{},{}"}}}}"#,
            pid, trace_type_name, key.row, key.col
        )?;
    }

    // Write metadata events: thread_name for each tid
    for (&(pid, tid), &name) in &tid_names {
        if !first { output.write_all(b",\n")?; }
        first = false;
        write!(
            output,
            r#"{{"name":"thread_name","ph":"M","pid":{},"tid":{},"args":{{"name":"{}"}}}}"#,
            pid, tid, name
        )?;
    }

    // Write actual trace events.
    // Instant events (most of ours) are emitted as Begin+End pairs with
    // duration of 1 cycle, matching mlir-aie's convention.
    for evt in events {
        let key = TileTraceKey {
            trace_type: event_trace_type(&evt.event),
            col: evt.col,
            row: evt.row,
        };
        let pid = pid_map[&key];
        let tid = event_tid(&evt.event);
        let name = event_name(&evt.event);
        let ts = evt.cycle;

        // Begin event
        if !first { output.write_all(b",\n")?; }
        first = false;
        write!(
            output,
            r#"{{"name":"{}","ph":"B","pid":{},"tid":{},"ts":{},"args":{{}}}}"#,
            name, pid, tid, ts
        )?;

        // End event (1 cycle duration for instant events,
        // stall events use their cycle count as duration)
        let duration = match &evt.event {
            EventType::MemoryStall { cycles } => *cycles as u64,
            EventType::LockStall { cycles } => *cycles as u64,
            EventType::StreamStall { cycles } => *cycles as u64,
            _ => 1,
        };

        output.write_all(b",\n")?;
        write!(
            output,
            r#"{{"name":"{}","ph":"E","pid":{},"tid":{},"ts":{},"args":{{}}}}"#,
            name, pid, tid, ts + duration
        )?;
    }

    output.write_all(b"\n]\n")?;
    Ok(())
}

/// Offset all `pid` fields in a Perfetto JSON string and prefix process_name args.
///
/// Used by the trace merge pipeline to separate traces from different sources
/// (NPU hardware, emulator, aiesimulator) into distinct PID ranges within a
/// single combined Perfetto JSON file.
///
/// This operates on raw JSON text via line-by-line regex replacement rather
/// than full JSON parsing, keeping the dependency footprint minimal and
/// handling the simple, regular structure of Perfetto trace event arrays.
///
/// # Arguments
///
/// * `json` - Perfetto JSON string (array of trace event objects)
/// * `pid_offset` - Value to add to every `"pid":N` field
/// * `name_prefix` - Prefix for process_name metadata args (e.g. "Emulator: ")
pub fn offset_perfetto_pids(json: &str, pid_offset: i64, name_prefix: &str) -> String {
    // Match "pid":N where N is a non-negative integer
    let pid_re = regex::Regex::new(r#""pid"\s*:\s*(\d+)"#).unwrap();
    // Match "name":"..." inside process_name metadata args
    let pname_re = regex::Regex::new(
        r#"("name"\s*:\s*"process_name".*?"args"\s*:\s*\{[^}]*"name"\s*:\s*")([^"]*)"#
    ).unwrap();

    let mut result = String::with_capacity(json.len() + json.len() / 10);

    for line in json.lines() {
        let mut modified = pid_re.replace_all(line, |caps: &regex::Captures| {
            let old_pid: i64 = caps[1].parse().unwrap_or(0);
            let new_pid = old_pid + pid_offset;
            format!(r#""pid":{}"#, new_pid)
        }).to_string();

        // Prefix process_name if this line is a process_name metadata event
        if modified.contains("\"process_name\"") && !name_prefix.is_empty() {
            modified = pname_re.replace(&modified, |caps: &regex::Captures| {
                format!("{}{}{}", &caps[1], name_prefix, &caps[2])
            }).to_string();
        }

        result.push_str(&modified);
        result.push('\n');
    }

    result
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

/// Map an emulator `EventType` to its AIE2 memory module hardware event ID.
///
/// DMA events are per-channel. The channel field encodes:
/// - 0 = S2MM ch0, 1 = S2MM ch1, 2 = MM2S ch0, 3 = MM2S ch1
///
/// Lock events are per-lock (lock IDs 0-7 have dedicated event IDs).
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::engine::TileTracedEvent;

    #[test]
    fn test_event_name_mapping() {
        // Verify key event names match mlir-aie conventions
        assert_eq!(event_name(&EventType::InstrVector { pc: 0 }), "INSTR_VECTOR");
        assert_eq!(event_name(&EventType::InstrLoad { pc: 0 }), "INSTR_LOAD");
        assert_eq!(event_name(&EventType::DmaStartTask { channel: 0 }), "DMA_START_TASK");
        assert_eq!(event_name(&EventType::CoreDisabled), "DISABLED");
        assert_eq!(event_name(&EventType::LockAcquire { lock_id: 0 }), "LOCK_ACQ");
    }

    #[test]
    fn test_event_trace_type() {
        assert_eq!(event_trace_type(&EventType::InstrVector { pc: 0 }), TraceType::Core);
        assert_eq!(event_trace_type(&EventType::DmaStartTask { channel: 0 }), TraceType::Mem);
        assert_eq!(event_trace_type(&EventType::LockAcquire { lock_id: 0 }), TraceType::Mem);
    }

    #[test]
    fn test_export_empty() {
        let mut buf = Vec::new();
        export_perfetto(&[], &mut buf).unwrap();
        let json = String::from_utf8(buf).unwrap();
        // Empty array with newlines
        assert!(json.starts_with("["));
        assert!(json.trim_end().ends_with("]"));
        // No events
        assert!(!json.contains(r#""ph":"B""#));
    }

    #[test]
    fn test_export_basic_events() {
        let events = vec![
            TileTracedEvent { col: 0, row: 2, cycle: 10, event: EventType::InstrLoad { pc: 0x100 } },
            TileTracedEvent { col: 0, row: 2, cycle: 11, event: EventType::InstrVector { pc: 0x104 } },
            TileTracedEvent { col: 0, row: 2, cycle: 15, event: EventType::CoreDisabled },
        ];

        let mut buf = Vec::new();
        export_perfetto(&events, &mut buf).unwrap();
        let json = String::from_utf8(buf).unwrap();

        // Should be valid JSON array
        assert!(json.starts_with("["));
        assert!(json.trim_end().ends_with("]"));

        // Should contain metadata
        assert!(json.contains("process_name"));
        assert!(json.contains("core_trace for tile2,0"));

        // Should contain event names
        assert!(json.contains("INSTR_LOAD"));
        assert!(json.contains("INSTR_VECTOR"));
        assert!(json.contains("DISABLED"));

        // Should contain Begin and End phases
        assert!(json.contains(r#""ph":"B""#));
        assert!(json.contains(r#""ph":"E""#));
    }

    #[test]
    fn test_export_multi_tile() {
        let events = vec![
            TileTracedEvent { col: 0, row: 2, cycle: 10, event: EventType::InstrLoad { pc: 0 } },
            TileTracedEvent { col: 1, row: 2, cycle: 10, event: EventType::InstrLoad { pc: 0 } },
            TileTracedEvent { col: 0, row: 2, cycle: 10, event: EventType::DmaStartTask { channel: 0 } },
        ];

        let mut buf = Vec::new();
        export_perfetto(&events, &mut buf).unwrap();
        let json = String::from_utf8(buf).unwrap();

        // Should have 3 process_name metadata entries:
        // core_trace for tile(2,0), core_trace for tile(2,1), mem_trace for tile(2,0)
        let process_count = json.matches("process_name").count();
        assert_eq!(process_count, 3, "Should have 3 tile/trace-type combos");
    }

    #[test]
    fn test_export_stall_duration() {
        let events = vec![
            TileTracedEvent { col: 0, row: 2, cycle: 100, event: EventType::LockStall { cycles: 5 } },
        ];

        let mut buf = Vec::new();
        export_perfetto(&events, &mut buf).unwrap();
        let json = String::from_utf8(buf).unwrap();

        // Begin at ts=100, End at ts=105 (5 cycle duration)
        assert!(json.contains(r#""ts":100"#));
        assert!(json.contains(r#""ts":105"#));
    }

    #[test]
    fn test_offset_perfetto_pids_basic() {
        let json = r#"[
{"name":"process_name","ph":"M","pid":0,"args":{"name":"core_trace for tile2,0"}},
{"name":"INSTR_LOAD","ph":"B","pid":0,"tid":1,"ts":10,"args":{}}
]"#;
        let result = offset_perfetto_pids(json, 100, "Emulator: ");

        // PID should be shifted to 100
        assert!(result.contains(r#""pid":100"#));
        assert!(!result.contains(r#""pid":0"#));

        // process_name should be prefixed
        assert!(result.contains("Emulator: core_trace for tile2,0"));
    }

    #[test]
    fn test_offset_perfetto_pids_no_prefix() {
        let json = r#"{"name":"INSTR_LOAD","ph":"B","pid":5,"tid":1,"ts":10,"args":{}}"#;
        let result = offset_perfetto_pids(json, 0, "");
        // PID unchanged when offset is 0
        assert!(result.contains(r#""pid":5"#));
    }

    #[test]
    fn test_offset_perfetto_pids_multiple() {
        let json = r#"[
{"name":"process_name","ph":"M","pid":0,"args":{"name":"core_trace for tile2,0"}},
{"name":"process_name","ph":"M","pid":1,"args":{"name":"mem_trace for tile2,0"}},
{"name":"INSTR_LOAD","ph":"B","pid":0,"tid":1,"ts":10,"args":{}},
{"name":"DMA_START_TASK","ph":"B","pid":1,"tid":0,"ts":15,"args":{}}
]"#;
        let result = offset_perfetto_pids(json, 200, "aiesimulator: ");

        assert!(result.contains(r#""pid":200"#));
        assert!(result.contains(r#""pid":201"#));
        assert!(!result.contains(r#""pid":0"#));
        assert!(!result.contains(r#""pid":1,"#));
        assert!(result.contains("aiesimulator: core_trace"));
        assert!(result.contains("aiesimulator: mem_trace"));
    }

    // -- Hardware event ID mapping tests --

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
}
