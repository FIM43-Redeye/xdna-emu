//! VCD-to-Perfetto converter for aiesimulator trace files.
//!
//! Parses IEEE 1364 VCD (Value Change Dump) files produced by `aiesimulator
//! --dump-vcd` and converts the self-describing `event_trace` signals into
//! Perfetto-compatible Chrome Trace Event Format JSON.
//!
//! # Why not eventanalyze?
//!
//! AMD's `eventanalyze` tool requires `compiler_report.json` from the full
//! Vitis flow, which our aiecc.py/mlir-aie builds don't produce. It also
//! has no Perfetto JSON output format. This parser bypasses eventanalyze
//! entirely by reading the VCD directly.
//!
//! # VCD format specifics (aiesimulator)
//!
//! aiesimulator produces SystemC 1.0 VCD with these characteristics:
//! - All signals in a flat `$scope module SystemC $end` (no nested scopes)
//! - Signal hierarchy encoded in dotted signal names
//! - Signal IDs are plain integers (not typical short ASCII codes)
//! - Timescale is 1 ps; clock period is ~952 ps for AIE2 simulation
//! - 129K+ signals declared, but only ~200 per tile are `event_trace`
//!
//! # Event trace signal naming
//!
//! Event signals are self-describing, with the event code and name embedded:
//! ```text
//! shim:    ...math_engine.shim.tile_X_Y.event_trace.eventN_name
//! memtile: ...math_engine.mem_row.tile_X_Y.event_trace.eventN_name
//! compute: ...math_engine.array.tile_X_Y.cm.event_trace.eventN_name
//! ```
//!
//! These are 1-bit level signals: `b1` = event active, `b0` = event ended.
//! Transitions map directly to Perfetto Begin/End phases.
//!
//! # Usage
//!
//! ```ignore
//! use xdna_emu::trace::vcd::vcd_to_perfetto;
//! use std::fs::File;
//! use std::io::BufReader;
//!
//! let vcd = BufReader::new(File::open("trace.vcd")?);
//! let mut out = File::create("trace.json")?;
//! vcd_to_perfetto(vcd, &mut out, None)?;
//! ```

use std::collections::BTreeMap;
use std::io::{BufRead, Write};

use regex::Regex;

/// VCD-trace-local tile-type enum. NOT the archspec `TileKind` -- this
/// is a private enum used exclusively for VCD signal-name dispatch in
/// trace output. Do not expose it publicly or import `TileKind` in
/// parallel without considering which one the call site wants.
///
/// Tile type classification derived from VCD signal hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TileType {
    /// Shim tile (row 0): `math_engine.shim.tile_X_Y`
    Shim,
    /// Memory tile (row 1-2): `math_engine.mem_row.tile_X_Y`
    MemTile,
    /// Compute tile (row 3+): `math_engine.array.tile_X_Y.cm`
    Compute,
}

/// A parsed event_trace signal from the VCD header.
#[derive(Debug, Clone)]
pub struct EventSignal {
    pub tile_type: TileType,
    pub col: u8,
    pub row: u8,
    pub event_code: u16,
    /// Event name from the VCD signal suffix, uppercased to match mlir-aie
    /// conventions (e.g. "INSTR_VECTOR", "DMA_S2MM_0_START_TASK").
    pub event_name: String,
}

/// Trace type for PID grouping, matching our emulator's convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum TraceType {
    Core = 0,
    Mem = 1,
}

/// Key for identifying a unique tile + trace type combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct TileTraceKey {
    trace_type: TraceType,
    col: u8,
    row: u8,
}

/// A single event transition extracted from the VCD body.
#[derive(Debug)]
struct Transition {
    /// Timestamp in simulation cycles (converted from picoseconds).
    cycle: u64,
    /// Perfetto phase: "B" for begin (signal went high), "E" for end (went low).
    phase: &'static str,
    /// Reference to the parsed signal info.
    signal_idx: usize,
}

/// Parsed VCD header information.
pub struct VcdHeader {
    /// Timescale in picoseconds (e.g. 1 for "1 ps").
    pub timescale_ps: u64,
    /// Map from VCD signal ID (integer) to index in `signals`.
    pub signal_map: BTreeMap<u32, usize>,
    /// All parsed event_trace signals.
    pub signals: Vec<EventSignal>,
}

/// Parse the VCD header, extracting timescale and event_trace signals.
///
/// Reads lines until `$enddefinitions $end` is found. Only signals matching
/// the `event_trace.eventN_name` pattern are retained; all others are
/// skipped for performance.
pub fn parse_vcd_header(reader: &mut impl BufRead) -> std::io::Result<VcdHeader> {
    // Pattern for event_trace signals in any tile type.
    //
    // Three variants of the signal hierarchy:
    //   shim:    ...math_engine.shim.tile_X_Y.event_trace.eventN_name
    //   memtile: ...math_engine.mem_row.tile_X_Y.event_trace.eventN_name
    //   compute: ...math_engine.array.tile_X_Y.cm.event_trace.eventN_name
    let signal_re = Regex::new(
        r"math_engine\.(shim|mem_row|array)\.tile_(\d+)_(\d+)(?:\.cm)?\.event_trace\.event(\d+)_(.+)",
    )
    .expect("valid regex");

    let mut timescale_ps: u64 = 1;
    let mut signal_map: BTreeMap<u32, usize> = BTreeMap::new();
    let mut signals: Vec<EventSignal> = Vec::new();
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            break; // EOF
        }

        let trimmed = line.trim();

        // Parse timescale
        if trimmed.starts_with("$timescale") {
            // May be on same line or next line. Look for a number + unit.
            if let Some(ps) = parse_timescale(trimmed) {
                timescale_ps = ps;
            } else {
                // Read next line for the actual value
                line.clear();
                reader.read_line(&mut line)?;
                if let Some(ps) = parse_timescale(line.trim()) {
                    timescale_ps = ps;
                }
            }
            continue;
        }

        // End of header
        if trimmed.starts_with("$enddefinitions") {
            break;
        }

        // Parse signal declarations
        // Format: $var wire <width> <id> <signal_name> $end
        if trimmed.starts_with("$var") {
            // Split on whitespace. Fields: $var wire width id name $end
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 6 {
                let id_str = parts[3];
                let signal_name = parts[4];

                if let Some(caps) = signal_re.captures(signal_name) {
                    if let Ok(id) = id_str.parse::<u32>() {
                        let tile_type = match &caps[1] {
                            "shim" => TileType::Shim,
                            "mem_row" => TileType::MemTile,
                            "array" => TileType::Compute,
                            _ => continue,
                        };
                        let col: u8 = caps[2].parse().unwrap_or(0);
                        let row: u8 = caps[3].parse().unwrap_or(0);
                        let event_code: u16 = caps[4].parse().unwrap_or(0);
                        let event_name = caps[5].to_uppercase();

                        // Skip sentinel events that fire on every tile
                        // and provide no useful information.
                        if event_name == "NONE" || event_name == "TRUE" {
                            continue;
                        }

                        let idx = signals.len();
                        signals.push(EventSignal { tile_type, col, row, event_code, event_name });
                        signal_map.insert(id, idx);
                    }
                }
            }
            continue;
        }
    }

    Ok(VcdHeader { timescale_ps, signal_map, signals })
}

/// Parse a timescale string like "1 ps", "1ps", or "$timescale 1 ps $end".
/// Returns the timescale in picoseconds.
fn parse_timescale(s: &str) -> Option<u64> {
    // Extract number and unit from the string
    let s = s.trim_start_matches("$timescale").trim();
    let s = s.trim_end_matches("$end").trim();

    // Find the numeric part
    let num_end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    if num_end == 0 {
        return None;
    }
    let value: u64 = s[..num_end].parse().ok()?;
    let unit = s[num_end..].trim().to_lowercase();

    let multiplier = match unit.as_str() {
        "ps" => 1,
        "ns" => 1_000,
        "us" => 1_000_000,
        "ms" => 1_000_000_000,
        "s" => 1_000_000_000_000,
        _ => return None,
    };

    Some(value * multiplier)
}

/// Classify an event into Core or Memory trace type based on its name.
///
/// This determines which PID group the event belongs to in the Perfetto
/// output, matching the same logic as our emulator's `trace.rs`.
fn event_trace_type(signal: &EventSignal) -> TraceType {
    let name = &signal.event_name;

    // Core module events: instruction events, stalls, core state
    if name.starts_with("INSTR_")
        || name.ends_with("_STALL")
        || name.starts_with("MEMORY_STALL")
        || name.starts_with("LOCK_STALL")
        || name.starts_with("STREAM_STALL")
        || name == "ACTIVE"
        || name == "DISABLED"
        || name.starts_with("GROUP_CORE_")
        || name.starts_with("DEBUG_")
        || name.starts_with("FP_")
        || name == "TLAST_IN_WSS"
    {
        return TraceType::Core;
    }

    // Memory module events: DMA, locks
    if name.starts_with("DMA_") || name.starts_with("LOCK_") {
        return TraceType::Mem;
    }

    // Default to core for unknown events
    TraceType::Core
}

/// Assign a stable thread ID for visual grouping in Perfetto.
///
/// Events of the same category appear on the same timeline lane.
fn event_tid(signal: &EventSignal) -> u32 {
    let name = &signal.event_name;

    if signal.tile_type == TileType::Compute {
        // Core tile: spread instruction events across tids 0-7
        if name == "INSTR_VECTOR" {
            return 0;
        }
        if name == "INSTR_LOAD" {
            return 1;
        }
        if name == "INSTR_STORE" {
            return 2;
        }
        if name == "INSTR_CALL" || name == "INSTR_RETURN" {
            return 3;
        }
        if name.starts_with("INSTR_LOCK_") {
            return 4;
        }
        if name.starts_with("INSTR_STREAM_") {
            return 5;
        }
        if name.ends_with("_STALL")
            || name.starts_with("MEMORY_STALL")
            || name.starts_with("LOCK_STALL")
            || name.starts_with("STREAM_STALL")
        {
            return 6;
        }
        if name == "ACTIVE" || name == "DISABLED" {
            return 7;
        }
    }

    // DMA events
    if name.starts_with("DMA_") {
        if name.contains("START_TASK") {
            return 0;
        }
        if name.contains("FINISHED") {
            return 1;
        }
        if name.contains("STALLED") || name.contains("STARVATION") {
            return 2;
        }
        return 3;
    }

    // Lock events
    if name.starts_with("LOCK_") {
        if name.contains("ACQ") {
            return 3;
        }
        if name.contains("REL") {
            return 4;
        }
        return 5;
    }

    // Group and other events: use high tid to avoid cluttering main lanes
    if name.starts_with("GROUP_") {
        return 8;
    }

    // Catch-all
    9
}

/// AIE2 default clock period in picoseconds (~1.05 GHz).
///
/// This is the simulation clock period observed in aiesimulator VCD output.
/// Used as default when no explicit clock period is provided.
const DEFAULT_CLOCK_PERIOD_PS: u64 = 952;

/// Convert an aiesimulator VCD file to Perfetto-compatible JSON.
///
/// Reads the VCD from `reader`, extracts event_trace signal transitions,
/// converts timestamps from picoseconds to cycles, and writes Perfetto
/// JSON to `output`.
///
/// # Arguments
///
/// * `reader` - Buffered reader over the VCD file
/// * `output` - Writer for the Perfetto JSON output
/// * `clock_period_ps` - Optional clock period in picoseconds. If `None`,
///   uses the default AIE2 simulation period (952 ps).
pub fn vcd_to_perfetto(
    mut reader: impl BufRead,
    output: &mut dyn Write,
    clock_period_ps: Option<u64>,
) -> std::io::Result<VcdConvertResult> {
    let clock_ps = clock_period_ps.unwrap_or(DEFAULT_CLOCK_PERIOD_PS);

    // Phase 1: Parse header
    let header = parse_vcd_header(&mut reader)?;

    if header.signals.is_empty() {
        // No event_trace signals found -- write empty trace
        output.write_all(b"[\n]\n")?;
        return Ok(VcdConvertResult { signal_count: 0, transition_count: 0, tile_count: 0 });
    }

    // Phase 2: Scan body for transitions on tracked signals
    let transitions = parse_vcd_body(&mut reader, &header, clock_ps)?;

    // Phase 3: Emit Perfetto JSON
    let stats = emit_perfetto(&header, &transitions, output)?;

    Ok(stats)
}

/// Result statistics from a VCD-to-Perfetto conversion.
#[derive(Debug)]
pub struct VcdConvertResult {
    /// Number of event_trace signals found in the VCD header.
    pub signal_count: usize,
    /// Number of transitions (Begin + End events) written.
    pub transition_count: usize,
    /// Number of unique tiles found.
    pub tile_count: usize,
}

/// Parse the VCD body, collecting transitions on tracked event_trace signals.
fn parse_vcd_body(
    reader: &mut impl BufRead,
    header: &VcdHeader,
    clock_period_ps: u64,
) -> std::io::Result<Vec<Transition>> {
    let mut transitions = Vec::new();
    let mut current_time_ps: u64 = 0;
    let mut line = String::new();
    let mut in_dumpvars = false;

    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            break; // EOF
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Track $dumpvars/$end sections (skip initial values)
        if trimmed.starts_with("$dumpvars") {
            in_dumpvars = true;
            continue;
        }
        if in_dumpvars && trimmed.starts_with("$end") {
            in_dumpvars = false;
            continue;
        }
        if in_dumpvars {
            continue;
        }

        // Timestamp line: #<picoseconds>
        if let Some(ts_str) = trimmed.strip_prefix('#') {
            if let Ok(ts) = ts_str.parse::<u64>() {
                current_time_ps = ts;
            }
            continue;
        }

        // Value change: b<value> <id>
        // For 1-bit event_trace signals, we only care about b0/b1.
        if let Some(rest) = trimmed.strip_prefix('b') {
            // Split "0 12345" or "1 12345"
            if let Some((value_str, id_str)) = rest.split_once(' ') {
                if let Ok(id) = id_str.trim().parse::<u32>() {
                    if let Some(&sig_idx) = header.signal_map.get(&id) {
                        let phase = match value_str {
                            "1" => "B",    // Signal went high -> Begin
                            "0" => "E",    // Signal went low -> End
                            _ => continue, // Multi-bit value, skip
                        };
                        let cycle = current_time_ps / clock_period_ps;
                        transitions.push(Transition { cycle, phase, signal_idx: sig_idx });
                    }
                }
            }
        }
    }

    Ok(transitions)
}

/// Emit Perfetto JSON from parsed transitions.
fn emit_perfetto(
    header: &VcdHeader,
    transitions: &[Transition],
    output: &mut dyn Write,
) -> std::io::Result<VcdConvertResult> {
    // Build PID map from signals that actually have transitions
    let mut active_signals: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for t in transitions {
        active_signals.insert(t.signal_idx);
    }

    let mut pid_map: BTreeMap<TileTraceKey, u32> = BTreeMap::new();
    for &sig_idx in &active_signals {
        let signal = &header.signals[sig_idx];
        let key = TileTraceKey { trace_type: event_trace_type(signal), col: signal.col, row: signal.row };
        let next_pid = pid_map.len() as u32;
        pid_map.entry(key).or_insert(next_pid);
    }

    // Discover which tids are used per PID (for thread_name metadata)
    let mut tid_names: BTreeMap<(u32, u32), &str> = BTreeMap::new();
    for &sig_idx in &active_signals {
        let signal = &header.signals[sig_idx];
        let key = TileTraceKey { trace_type: event_trace_type(signal), col: signal.col, row: signal.row };
        let pid = pid_map[&key];
        let tid = event_tid(signal);
        tid_names.entry((pid, tid)).or_insert(&signal.event_name);
    }

    output.write_all(b"[\n")?;
    let mut first = true;

    // Write process_name metadata for each tile
    for (key, &pid) in &pid_map {
        let trace_type_name = match key.trace_type {
            TraceType::Core => "core",
            TraceType::Mem => "mem",
        };
        if !first {
            output.write_all(b",\n")?;
        }
        first = false;
        write!(
            output,
            r#"{{"name":"process_name","ph":"M","pid":{},"args":{{"name":"{}_trace for tile{},{}"}}}}"#,
            pid, trace_type_name, key.row, key.col
        )?;
    }

    // Write thread_name metadata for each tid
    for (&(pid, tid), &name) in &tid_names {
        if !first {
            output.write_all(b",\n")?;
        }
        first = false;
        write!(
            output,
            r#"{{"name":"thread_name","ph":"M","pid":{},"tid":{},"args":{{"name":"{}"}}}}"#,
            pid, tid, name
        )?;
    }

    // Write transition events
    let mut transition_count = 0;
    for t in transitions {
        let signal = &header.signals[t.signal_idx];
        let key = TileTraceKey { trace_type: event_trace_type(signal), col: signal.col, row: signal.row };
        let pid = pid_map[&key];
        let tid = event_tid(signal);

        if !first {
            output.write_all(b",\n")?;
        }
        first = false;
        write!(
            output,
            r#"{{"name":"{}","ph":"{}","pid":{},"tid":{},"ts":{},"args":{{}}}}"#,
            signal.event_name, t.phase, pid, tid, t.cycle
        )?;
        transition_count += 1;
    }

    output.write_all(b"\n]\n")?;

    Ok(VcdConvertResult { signal_count: header.signals.len(), transition_count, tile_count: pid_map.len() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufReader;

    /// Minimal VCD with one event_trace signal that transitions.
    const MINIMAL_VCD: &str = "\
$timescale
    1 ps
$end
$scope module SystemC $end
$var wire 	1		100		tl.aie_logical.aie_xtlm.math_engine.array.tile_0_3.cm.event_trace.event37_instr_vector	 $end
$var wire 	1		101		tl.aie_logical.aie_xtlm.math_engine.array.tile_0_3.cm.event_trace.event28_active	 $end
$var wire 	1		200		tl.aie_logical.aie_xtlm.math_engine.shim.tile_0_0.event_trace.event14_dma_s2mm_0_start_task	 $end
$var wire 	32		999		tl.aie_logical.aie_xtlm.math_engine.shim.tile_0_0.stream_switch.from_sFIFO0.data	 $end
$upscope $end
$enddefinitions $end
$dumpvars
b0 100
b0 101
b0 200
b0 999
$end

#952
b1 101

#1904
b1 100

#2856
b0 100

#3808
b1 100
b0 101

#4760
b0 100
";

    #[test]
    fn test_parse_timescale() {
        assert_eq!(parse_timescale("1 ps"), Some(1));
        assert_eq!(parse_timescale("1ps"), Some(1));
        assert_eq!(parse_timescale("10 ns"), Some(10_000));
        assert_eq!(parse_timescale("1 us"), Some(1_000_000));
        assert_eq!(parse_timescale("$timescale 1 ps $end"), Some(1));
    }

    #[test]
    fn test_parse_header() {
        let mut reader = BufReader::new(MINIMAL_VCD.as_bytes());
        let header = parse_vcd_header(&mut reader).unwrap();

        assert_eq!(header.timescale_ps, 1);
        // Should have 3 event_trace signals (not the stream_switch one)
        assert_eq!(header.signals.len(), 3);

        // Verify signal 100 -> event37_instr_vector on compute tile (0,3)
        let sig_idx = header.signal_map[&100];
        let sig = &header.signals[sig_idx];
        assert_eq!(sig.tile_type, TileType::Compute);
        assert_eq!(sig.col, 0);
        assert_eq!(sig.row, 3);
        assert_eq!(sig.event_code, 37);
        assert_eq!(sig.event_name, "INSTR_VECTOR");

        // Verify signal 101 -> event28_active on compute tile (0,3)
        let sig_idx = header.signal_map[&101];
        let sig = &header.signals[sig_idx];
        assert_eq!(sig.event_code, 28);
        assert_eq!(sig.event_name, "ACTIVE");

        // Verify signal 200 -> DMA event on shim tile (0,0)
        let sig_idx = header.signal_map[&200];
        let sig = &header.signals[sig_idx];
        assert_eq!(sig.tile_type, TileType::Shim);
        assert_eq!(sig.col, 0);
        assert_eq!(sig.row, 0);
        assert_eq!(sig.event_code, 14);
        assert_eq!(sig.event_name, "DMA_S2MM_0_START_TASK");

        // Signal 999 (stream_switch) should NOT be in the map
        assert!(!header.signal_map.contains_key(&999));
    }

    #[test]
    fn test_full_conversion() {
        let reader = BufReader::new(MINIMAL_VCD.as_bytes());
        let mut output = Vec::new();
        let result = vcd_to_perfetto(reader, &mut output, Some(952)).unwrap();

        assert_eq!(result.signal_count, 3);
        assert!(result.transition_count > 0);

        let json = String::from_utf8(output).unwrap();

        // Should be valid JSON structure
        assert!(json.starts_with("["));
        assert!(json.trim_end().ends_with("]"));

        // Should contain metadata
        assert!(json.contains("process_name"));
        assert!(json.contains("core_trace for tile3,0"));

        // Should contain events with correct names
        assert!(json.contains("INSTR_VECTOR"));
        assert!(json.contains("ACTIVE"));

        // Should contain Begin and End phases
        assert!(json.contains(r#""ph":"B""#));
        assert!(json.contains(r#""ph":"E""#));

        // Verify timestamps are in cycles, not picoseconds
        // #952 / 952 = cycle 1, #1904 / 952 = cycle 2, etc.
        assert!(json.contains(r#""ts":1"#)); // ACTIVE begins at cycle 1
        assert!(json.contains(r#""ts":2"#)); // INSTR_VECTOR begins at cycle 2
        assert!(json.contains(r#""ts":3"#)); // INSTR_VECTOR ends at cycle 3
    }

    #[test]
    fn test_empty_vcd() {
        let vcd = "\
$timescale 1 ps $end
$enddefinitions $end
$dumpvars
$end
";
        let reader = BufReader::new(vcd.as_bytes());
        let mut output = Vec::new();
        let result = vcd_to_perfetto(reader, &mut output, None).unwrap();

        assert_eq!(result.signal_count, 0);
        assert_eq!(result.transition_count, 0);

        let json = String::from_utf8(output).unwrap();
        assert_eq!(json.trim(), "[\n]");
    }

    #[test]
    fn test_memtile_signals() {
        let vcd = "\
$timescale 1 ps $end
$scope module SystemC $end
$var wire 	1		50		tl.aie_logical.aie_xtlm.math_engine.mem_row.tile_1_1.event_trace.event21_dma_s2mm_sel0_start_task	 $end
$upscope $end
$enddefinitions $end
$dumpvars
b0 50
$end

#952
b1 50

#1904
b0 50
";
        let mut reader = BufReader::new(vcd.as_bytes());
        let header = parse_vcd_header(&mut reader).unwrap();

        assert_eq!(header.signals.len(), 1);
        let sig = &header.signals[0];
        assert_eq!(sig.tile_type, TileType::MemTile);
        assert_eq!(sig.col, 1);
        assert_eq!(sig.row, 1);
        assert_eq!(sig.event_name, "DMA_S2MM_SEL0_START_TASK");
    }

    #[test]
    fn test_event_classification() {
        // Core events
        let core_signal = EventSignal {
            tile_type: TileType::Compute,
            col: 0,
            row: 3,
            event_code: 37,
            event_name: "INSTR_VECTOR".to_string(),
        };
        assert_eq!(event_trace_type(&core_signal), TraceType::Core);

        let active_signal = EventSignal {
            tile_type: TileType::Compute,
            col: 0,
            row: 3,
            event_code: 28,
            event_name: "ACTIVE".to_string(),
        };
        assert_eq!(event_trace_type(&active_signal), TraceType::Core);

        // Memory events
        let dma_signal = EventSignal {
            tile_type: TileType::Shim,
            col: 0,
            row: 0,
            event_code: 14,
            event_name: "DMA_S2MM_0_START_TASK".to_string(),
        };
        assert_eq!(event_trace_type(&dma_signal), TraceType::Mem);

        let lock_signal = EventSignal {
            tile_type: TileType::Compute,
            col: 0,
            row: 3,
            event_code: 44,
            event_name: "LOCK_SEL0_ACQ_EQ".to_string(),
        };
        assert_eq!(event_trace_type(&lock_signal), TraceType::Mem);
    }

    /// Integration test against real aiesimulator VCD output.
    ///
    /// Ignored by default (file may not exist). Run explicitly with:
    /// `cargo test --lib trace::vcd::tests::test_real_vcd -- --ignored`
    #[test]
    #[ignore]
    fn test_real_vcd() {
        use std::fs::File;

        let vcd_path = "build/unit_tests/08_tile_locks/--simulation-cycle-timeout.vcd";
        let file = match File::open(vcd_path) {
            Ok(f) => f,
            Err(_) => {
                eprintln!("Skipping: {vcd_path} not found (run aiesimulator first)");
                return;
            }
        };

        let reader = BufReader::new(file);
        let mut output = Vec::new();
        let result = vcd_to_perfetto(reader, &mut output, None).unwrap();

        eprintln!("Real VCD conversion results:");
        eprintln!("  Event trace signals: {}", result.signal_count);
        eprintln!("  Transitions written: {}", result.transition_count);
        eprintln!("  Unique tiles:        {}", result.tile_count);

        // The 08_tile_locks test should have many event_trace signals
        // (sentinel events TRUE/NONE are filtered, leaving ~17K useful signals)
        assert!(
            result.signal_count > 1000,
            "Expected 1000+ event_trace signals, got {}",
            result.signal_count
        );
        assert!(result.transition_count > 0, "Expected transitions");
        assert!(result.tile_count > 0, "Expected at least one tile");

        let json = String::from_utf8(output).unwrap();

        // Basic structure checks
        assert!(json.starts_with("["));
        assert!(json.trim_end().ends_with("]"));
        assert!(json.contains("process_name"));
        assert!(json.contains(r#""ph":"B""#));
        assert!(json.contains(r#""ph":"E""#));

        // Should have events from the active compute tile (7,3)
        assert!(json.contains("tile3,7"), "Expected tile3,7 (row 3, col 7)");

        // Should contain core events
        assert!(json.contains("ACTIVE") || json.contains("DISABLED"), "Expected core state events");

        // Write output for manual inspection
        let out_path = "build/unit_tests/08_tile_locks/trace_from_vcd.json";
        std::fs::write(out_path, &json).unwrap();
        eprintln!("  Perfetto JSON written to: {out_path}");
        eprintln!("  Open at https://ui.perfetto.dev/");
    }

    #[test]
    fn test_tid_assignment() {
        let make_signal = |name: &str| EventSignal {
            tile_type: TileType::Compute,
            col: 0,
            row: 3,
            event_code: 0,
            event_name: name.to_string(),
        };

        // Core instruction events get distinct tids
        assert_eq!(event_tid(&make_signal("INSTR_VECTOR")), 0);
        assert_eq!(event_tid(&make_signal("INSTR_LOAD")), 1);
        assert_eq!(event_tid(&make_signal("INSTR_STORE")), 2);
        assert_eq!(event_tid(&make_signal("INSTR_CALL")), 3);
        assert_eq!(event_tid(&make_signal("INSTR_RETURN")), 3);
        assert_eq!(event_tid(&make_signal("ACTIVE")), 7);
        assert_eq!(event_tid(&make_signal("DISABLED")), 7);
    }
}
