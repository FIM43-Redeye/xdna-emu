//! VCD emission from emulator state.
//!
//! Records emulator state changes as a VCD file, using the mapping tree
//! to generate signal headers that are compatible with the aiesimulator
//! VCD format. This enables cycle-by-cycle comparison.
//!
//! # Usage
//!
//! ```ignore
//! use crate::vcd::emit::VcdRecorder;
//! use crate::vcd::mapping::build_aie2_mapping_tree;
//!
//! let tree = build_aie2_mapping_tree();
//! let mut file = BufWriter::new(File::create("trace.vcd")?);
//! let mut recorder = VcdRecorder::new(&mut file, &tree)?;
//! recorder.record(0, &StatePath::LockValue { col: 0, row: 1, idx: 3 }, 0)?;
//! recorder.record(10, &StatePath::LockValue { col: 0, row: 1, idx: 3 }, 1)?;
//! recorder.finish()?;
//! ```
//!
//! # VCD signal naming
//!
//! All signals are placed in a flat `top.emu` scope. Each signal's VCD
//! reference name is the [`StatePath`] display string with non-identifier
//! characters replaced by `_`. This avoids parser errors in downstream tools
//! while remaining human-readable.
//!
//! The comparison engine matches signals by [`StatePath`] equality, not by
//! the VCD identifier string, so the exact naming scheme does not affect
//! comparison accuracy.
//!
//! # Gating
//!
//! This module is only compiled when the `vcd-recording` feature is enabled.

use crate::vcd::mapping::MappingTree;
use crate::vcd::state_path::StatePath;
use std::collections::HashMap;
use std::io::Write;

// ---------------------------------------------------------------------------
// VcdRecorder
// ---------------------------------------------------------------------------

/// VCD recorder that writes emulator state changes to a VCD file.
///
/// # Lifecycle
///
/// 1. Create with [`VcdRecorder::new`] -- writes the VCD header immediately.
/// 2. Call [`VcdRecorder::record`] for each state change.
/// 3. Call [`VcdRecorder::finish`] to flush and finalise the file.
///
/// The recorder silently ignores out-of-order timestamps: if a call to
/// `record` provides a `time` less than the current timestamp the event
/// is still written but no new `#timestamp` line is emitted.
pub struct VcdRecorder<W: Write> {
    writer: vcd::Writer<W>,
    /// Map from StatePath to VCD variable ID.
    path_to_id: HashMap<StatePath, vcd::IdCode>,
    /// Current simulation time (most recent `#timestamp` written).
    current_time: u64,
    /// Whether any `#timestamp` line has been written yet.
    ///
    /// Needed because `current_time` starts at 0, so we cannot use
    /// `time > current_time` alone to decide whether to emit `#0`.
    timestamp_written: bool,
}

impl<W: Write> VcdRecorder<W> {
    /// Create a new VCD recorder and write the VCD header.
    ///
    /// The header declares all signals that the mapping tree can enumerate,
    /// placed in a flat `top.emu` scope. After this call the file is ready
    /// to receive value-change events via [`record`](Self::record).
    ///
    /// Returns `Err` if any I/O error occurs while writing the header.
    pub fn new(writer: W, tree: &MappingTree) -> Result<Self, String> {
        let mut vcd_writer = vcd::Writer::new(writer);
        let mut path_to_id = HashMap::new();

        // Write standard VCD header fields.
        vcd_writer
            .timescale(1, vcd::TimescaleUnit::NS)
            .map_err(|e| format!("VCD timescale: {}", e))?;

        // Outer scope: top
        vcd_writer
            .add_module("top")
            .map_err(|e| format!("VCD scope top: {}", e))?;

        // Inner scope: emu (emulator -- distinguishes our VCDs from aiesimulator VCDs)
        vcd_writer
            .add_module("emu")
            .map_err(|e| format!("VCD scope emu: {}", e))?;

        // Enumerate every StatePath the tree can produce.
        let paths = tree.enumerate_all();

        for path in paths {
            let vcd_name = sanitize_vcd_name(&path.to_string());
            let width = signal_width(&path);
            let id = vcd_writer
                .add_wire(width, &vcd_name)
                .map_err(|e| format!("VCD wire {}: {}", vcd_name, e))?;
            path_to_id.insert(path, id);
        }

        // Close the two scopes.
        vcd_writer
            .upscope()
            .map_err(|e| format!("VCD upscope emu: {}", e))?;
        vcd_writer
            .upscope()
            .map_err(|e| format!("VCD upscope top: {}", e))?;

        // End of header declarations.
        vcd_writer
            .enddefinitions()
            .map_err(|e| format!("VCD enddefinitions: {}", e))?;

        Ok(VcdRecorder {
            writer: vcd_writer,
            path_to_id,
            current_time: 0,
            timestamp_written: false,
        })
    }

    /// Record a state change at the given simulation time.
    ///
    /// If `time` is greater than the current timestamp, a `#time` line is
    /// written first. If `time` equals the current timestamp, no new timestamp
    /// line is written. If `time` is less than the current timestamp the event
    /// is written at the current time (VCD files must be monotonically
    /// increasing; do not rely on this behaviour for correctness).
    ///
    /// Returns `Err` if `path` was not in the mapping tree passed to `new`, or
    /// if any I/O error occurs.
    pub fn record(&mut self, time: u64, path: &StatePath, value: u128) -> Result<(), String> {
        // Advance the simulation clock when needed.
        // We must also emit the first timestamp even when time == 0, because
        // `current_time` starts at 0 and `time > current_time` would be false.
        let need_timestamp = !self.timestamp_written || time > self.current_time;
        if need_timestamp {
            self.writer
                .timestamp(time)
                .map_err(|e| format!("VCD timestamp {}: {}", time, e))?;
            self.current_time = time;
            self.timestamp_written = true;
        }

        let id = *self
            .path_to_id
            .get(path)
            .ok_or_else(|| format!("Unknown StatePath: {}", path))?;

        let width = signal_width(path);

        if width == 1 {
            // Scalar: write a single-bit value change.
            let bit = if value != 0 {
                vcd::Value::V1
            } else {
                vcd::Value::V0
            };
            self.writer
                .change_scalar(id, bit)
                .map_err(|e| format!("VCD change_scalar for {}: {}", path, e))?;
        } else {
            // Vector: emit bits MSB-first as required by VCD.
            self.writer
                .change_vector(
                    id,
                    (0..width)
                        .rev()
                        .map(|i| if (value >> i) & 1 == 1 { vcd::Value::V1 } else { vcd::Value::V0 }),
                )
                .map_err(|e| format!("VCD change_vector for {}: {}", path, e))?;
        }

        Ok(())
    }

    /// Flush the writer and finalise the VCD file.
    ///
    /// Must be called to ensure all buffered data is written to the underlying
    /// `Write` implementation.
    pub fn finish(mut self) -> Result<(), String> {
        self.writer
            .flush()
            .map_err(|e| format!("VCD flush: {}", e))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// signal_width -- bit width per StatePath variant
// ---------------------------------------------------------------------------

/// Return the bit width to use for a VCD `$var` declaration for this signal.
///
/// Width assignments follow the signal semantics:
/// - 1 bit: boolean flags (reset, stall, valid, TLAST, …)
/// - 8 bits: small indices (BD index, lock ID, packet ID, …)
/// - 20 bits: AIE2 address bus width (program counter, memory address)
/// - 32 bits: most data registers and counters
/// - 128 bits: full 256-bit vector lane expressed as two 128-bit halves
///   (DMA data bus, program-memory instruction word)
pub fn signal_width(path: &StatePath) -> u32 {
    match path {
        // ------------------------------------------------------------------
        // 1-bit: boolean status flags
        // ------------------------------------------------------------------
        StatePath::CoreReset { .. }
        | StatePath::CoreBreakpointHalted { .. }
        | StatePath::CoreTmLoad { .. }
        | StatePath::CoreTmStore { .. }
        | StatePath::StreamPortIdle { .. }
        | StatePath::StreamPortRunning { .. }
        | StatePath::StreamPortStalled { .. }
        | StatePath::StreamPortTlast { .. }
        | StatePath::DmaBdValid { .. }
        | StatePath::DmaEnablePacket { .. }
        | StatePath::DmaUseNextBd { .. }
        | StatePath::DmaTlastSuppress { .. }
        | StatePath::EventTrace { .. } => 1,

        // ------------------------------------------------------------------
        // 8-bit: small index fields
        // ------------------------------------------------------------------
        StatePath::DmaCurrentBd { .. }
        | StatePath::DmaLockAcqId { .. }
        | StatePath::DmaLockAcqValue { .. }
        | StatePath::DmaLockRelValue { .. }
        | StatePath::DmaNextBd { .. }
        | StatePath::DmaPacketId { .. } => 8,

        // ------------------------------------------------------------------
        // 20-bit: AIE2 address bus (program counter, memory address)
        // ------------------------------------------------------------------
        StatePath::CorePmAddress { .. } | StatePath::CoreTmAddress { .. } => 20,

        // ------------------------------------------------------------------
        // 32-bit: general-purpose registers, counters, and status words
        // ------------------------------------------------------------------
        StatePath::LockValue { .. }
        | StatePath::LockOp { .. }
        | StatePath::CorePc { .. }
        | StatePath::CoreTmReadData { .. }
        | StatePath::CoreTmWriteData { .. }
        | StatePath::StreamPortData { .. }
        | StatePath::DmaFsmState { .. }
        | StatePath::DmaBdLength { .. }
        | StatePath::DmaAddress { .. }
        | StatePath::DmaProcessedStream { .. }
        | StatePath::DmaProcessedMem { .. }
        | StatePath::DmaStatus { .. }
        | StatePath::DmaIterStepsize { .. }
        | StatePath::DmaIterCurrent { .. }
        | StatePath::DmaIterWrap { .. }
        | StatePath::MemBankConflict { .. }
        | StatePath::MemConflictAddr { .. }
        | StatePath::MemPortAccess { .. }
        | StatePath::PerfCounter { .. } => 32,

        // ------------------------------------------------------------------
        // 128-bit: full-width data bus (DMA data, instruction word)
        // ------------------------------------------------------------------
        StatePath::CorePmData { .. } | StatePath::DmaData { .. } => 128,
    }
}

// ---------------------------------------------------------------------------
// sanitize_vcd_name -- make a StatePath display string a valid VCD identifier
// ---------------------------------------------------------------------------

/// Convert a [`StatePath`] display string to a valid VCD reference identifier.
///
/// VCD identifiers may not contain whitespace or the special characters used in
/// VCD syntax (`$`, `#`, `/`). The `StatePath` display format uses `(`, `)`,
/// `.`, `[`, `]` for readability. This function replaces all such characters
/// with `_` so the string is accepted by VCD parsers without quoting.
///
/// # Example
///
/// `tile(0,1).lock.value[3]` becomes `tile_0_1__lock_value_3_`
fn sanitize_vcd_name(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' => c,
            _ => '_',
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vcd::lock_mapping::lock_mapping;
    use crate::vcd::state_path::DmaDir;

    /// Build a small but valid mapping tree covering one tile with 4 locks.
    fn build_small_tree() -> MappingTree {
        MappingTree::builder()
            .scope("top")
            .scope("emu")
            .tile_group("test", &[(0, 1)])
            .mapping(lock_mapping(4))
            .done_tile_group()
            .build()
    }

    // ------------------------------------------------------------------
    // Header generation
    // ------------------------------------------------------------------

    #[test]
    fn recorder_creates_vcd_header() {
        let tree = build_small_tree();
        let mut buf = Vec::new();
        let recorder = VcdRecorder::new(&mut buf, &tree).unwrap();
        recorder.finish().unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("$timescale"), "missing $timescale in: {}", output);
        assert!(output.contains("$enddefinitions"), "missing $enddefinitions in: {}", output);
    }

    #[test]
    fn recorder_header_contains_scope_hierarchy() {
        let tree = build_small_tree();
        let mut buf = Vec::new();
        let recorder = VcdRecorder::new(&mut buf, &tree).unwrap();
        recorder.finish().unwrap();
        let output = String::from_utf8(buf).unwrap();
        // Should have a scope for "top" and "emu".
        assert!(output.contains("top"), "missing top scope");
        assert!(output.contains("emu"), "missing emu scope");
    }

    #[test]
    fn recorder_header_contains_signal_declarations() {
        let tree = build_small_tree();
        let mut buf = Vec::new();
        let recorder = VcdRecorder::new(&mut buf, &tree).unwrap();
        recorder.finish().unwrap();
        let output = String::from_utf8(buf).unwrap();
        // lock_mapping(4) for tile (0,1) produces 4 value + 4 op = 8 signals.
        // Each is declared with $var.
        let var_count = output.matches("$var").count();
        assert_eq!(var_count, 8, "expected 8 $var declarations, got {}", var_count);
    }

    // ------------------------------------------------------------------
    // Value-change recording
    // ------------------------------------------------------------------

    #[test]
    fn recorder_records_value_change_timestamps() {
        let tree = build_small_tree();
        let mut buf = Vec::new();
        {
            let mut recorder = VcdRecorder::new(&mut buf, &tree).unwrap();
            let path = StatePath::LockValue { col: 0, row: 1, idx: 0 };
            recorder.record(0, &path, 0).unwrap();
            recorder.record(10, &path, 42).unwrap();
            recorder.finish().unwrap();
        }
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("#0"), "missing #0 timestamp");
        assert!(output.contains("#10"), "missing #10 timestamp");
    }

    #[test]
    fn recorder_scalar_at_time_zero() {
        let tree = build_small_tree();
        let mut buf = Vec::new();
        {
            let mut recorder = VcdRecorder::new(&mut buf, &tree).unwrap();
            let path = StatePath::LockValue { col: 0, row: 1, idx: 0 };
            recorder.record(0, &path, 1).unwrap();
            recorder.finish().unwrap();
        }
        let output = String::from_utf8(buf).unwrap();
        // Time 0 should produce a #0 line followed by change data.
        assert!(output.contains("#0"), "missing #0 timestamp after record at t=0");
    }

    #[test]
    fn recorder_does_not_duplicate_timestamp_line() {
        // Two records at the same time should emit one timestamp line.
        let tree = build_small_tree();
        let mut buf = Vec::new();
        {
            let mut recorder = VcdRecorder::new(&mut buf, &tree).unwrap();
            let p0 = StatePath::LockValue { col: 0, row: 1, idx: 0 };
            let p1 = StatePath::LockValue { col: 0, row: 1, idx: 1 };
            recorder.record(5, &p0, 0).unwrap();
            recorder.record(5, &p1, 1).unwrap();
            recorder.finish().unwrap();
        }
        let output = String::from_utf8(buf).unwrap();
        let ts_count = output.matches("#5").count();
        assert_eq!(ts_count, 1, "expected exactly one #5 timestamp, got {}", ts_count);
    }

    // ------------------------------------------------------------------
    // Unknown path rejection
    // ------------------------------------------------------------------

    #[test]
    fn recorder_rejects_unknown_path() {
        let tree = build_small_tree();
        let mut buf = Vec::new();
        let mut recorder = VcdRecorder::new(&mut buf, &tree).unwrap();
        // Tile (9, 9) is not in this tree.
        let unknown = StatePath::LockValue { col: 9, row: 9, idx: 0 };
        assert!(
            recorder.record(0, &unknown, 0).is_err(),
            "should reject unknown StatePath"
        );
    }

    // ------------------------------------------------------------------
    // signal_width classification
    // ------------------------------------------------------------------

    #[test]
    fn signal_width_boolean_flags_are_1_bit() {
        assert_eq!(signal_width(&StatePath::CoreReset { col: 0, row: 0 }), 1);
        assert_eq!(signal_width(&StatePath::CoreBreakpointHalted { col: 0, row: 0 }), 1);
        assert_eq!(signal_width(&StatePath::DmaBdValid { col: 0, row: 0, dir: DmaDir::S2mm, ch: 0 }), 1);
        assert_eq!(signal_width(&StatePath::DmaEnablePacket { col: 0, row: 0, dir: DmaDir::S2mm, ch: 0 }), 1);
        assert_eq!(signal_width(&StatePath::DmaUseNextBd { col: 0, row: 0, dir: DmaDir::S2mm, ch: 0 }), 1);
        assert_eq!(signal_width(&StatePath::DmaTlastSuppress { col: 0, row: 0, dir: DmaDir::S2mm, ch: 0 }), 1);
        use crate::vcd::state_path::PortId;
        assert_eq!(signal_width(&StatePath::StreamPortIdle { col: 0, row: 0, port: PortId::named("sSouth0") }), 1);
    }

    #[test]
    fn signal_width_small_indices_are_8_bit() {
        assert_eq!(signal_width(&StatePath::DmaCurrentBd { col: 0, row: 0, dir: DmaDir::S2mm, ch: 0 }), 8);
        assert_eq!(signal_width(&StatePath::DmaNextBd { col: 0, row: 0, dir: DmaDir::S2mm, ch: 0 }), 8);
        assert_eq!(signal_width(&StatePath::DmaPacketId { col: 0, row: 0, dir: DmaDir::S2mm, ch: 0 }), 8);
    }

    #[test]
    fn signal_width_address_bus_is_20_bit() {
        assert_eq!(signal_width(&StatePath::CorePmAddress { col: 0, row: 0 }), 20);
        assert_eq!(signal_width(&StatePath::CoreTmAddress { col: 0, row: 0 }), 20);
    }

    #[test]
    fn signal_width_general_registers_are_32_bit() {
        assert_eq!(signal_width(&StatePath::LockValue { col: 0, row: 0, idx: 0 }), 32);
        assert_eq!(signal_width(&StatePath::LockOp { col: 0, row: 0, idx: 0 }), 32);
        assert_eq!(signal_width(&StatePath::CorePc { col: 0, row: 0, stage: 0 }), 32);
        assert_eq!(signal_width(&StatePath::PerfCounter { col: 0, row: 0, idx: 0 }), 32);
    }

    #[test]
    fn signal_width_data_bus_is_128_bit() {
        assert_eq!(signal_width(&StatePath::CorePmData { col: 0, row: 0 }), 128);
        assert_eq!(signal_width(&StatePath::DmaData { col: 0, row: 0, dir: DmaDir::S2mm, ch: 0 }), 128);
    }

    // ------------------------------------------------------------------
    // VCD name sanitisation
    // ------------------------------------------------------------------

    #[test]
    fn sanitize_vcd_name_replaces_special_chars() {
        // The StatePath display format uses parens, dots, and brackets.
        let raw = "tile(0,1).lock.value[3]";
        let clean = sanitize_vcd_name(raw);
        // Must not contain VCD-unfriendly characters.
        assert!(!clean.contains('('), "paren not replaced: {}", clean);
        assert!(!clean.contains(')'), "paren not replaced: {}", clean);
        assert!(!clean.contains('.'), "dot not replaced: {}", clean);
        assert!(!clean.contains('['), "bracket not replaced: {}", clean);
        assert!(!clean.contains(']'), "bracket not replaced: {}", clean);
        assert!(!clean.contains(','), "comma not replaced: {}", clean);
        // Alphanumeric content must be preserved.
        assert!(clean.contains('0'), "digit lost: {}", clean);
        assert!(clean.contains('1'), "digit lost: {}", clean);
        assert!(clean.contains("lock"), "word lost: {}", clean);
        assert!(clean.contains("value"), "word lost: {}", clean);
    }

    #[test]
    fn sanitize_vcd_name_preserves_alnum_and_underscore() {
        let raw = "abc_XYZ_123";
        let clean = sanitize_vcd_name(raw);
        assert_eq!(clean, raw, "alphanumeric/underscore content must not change");
    }
}
