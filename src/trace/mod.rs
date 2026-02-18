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
}
