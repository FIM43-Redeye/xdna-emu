//! Trace event sweep: multi-group NPU trace capture and comparison.
//!
//! The NPU hardware trace unit has 8 event slots configured by two registers
//! (Trace_Event0 and Trace_Event1). Each slot monitors one hardware event.
//! Since many interesting events exist (instruction types, stalls, locks,
//! ports), a single 8-slot configuration can only observe a subset.
//!
//! This module provides:
//! - Event group definitions (3 groups of 8 events = 24 events total)
//! - Binary patching of `insts.bin` to swap trace event configurations
//!   without recompilation
//! - Decoding of binary trace packets into event sequences
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
        u32::from_le_bytes([
            self.event_ids[0],
            self.event_ids[1],
            self.event_ids[2],
            self.event_ids[3],
        ])
    }

    /// Compute the Trace_Event1 register value (slots 4-7, little-endian).
    pub fn event1_value(&self) -> u32 {
        u32::from_le_bytes([
            self.event_ids[4],
            self.event_ids[5],
            self.event_ids[6],
            self.event_ids[7],
        ])
    }
}

/// The default trace event group matching fuzz_template.py.
///
/// These exact event IDs produce register values 640680481 (0x26300221) and
/// 1330084396 (0x4F477A2C), matching the hardcoded values in fuzz_template.py.
/// The event IDs were derived from the mlir-aie vec_mul_event_trace reference.
const GROUP_0: TraceEventGroup = TraceEventGroup {
    name: "default",
    // Trace_Event0 = 0x26300221, Trace_Event1 = 0x4F477A2C
    event_ids: [0x21, 0x02, 0x30, 0x26, 0x2C, 0x7A, 0x47, 0x4F],
};

/// Group 1: memory and flow control events.
const GROUP_1: TraceEventGroup = TraceEventGroup {
    name: "memory_flow",
    // INSTR_STORE(39), INSTR_CALL(35), INSTR_RETURN(36), INSTR_STREAM_GET(40),
    // INSTR_STREAM_PUT(41), LOCK_RELEASE_REQ(45), MEMORY_STALL(23), STREAM_STALL(24)
    event_ids: [39, 35, 36, 40, 41, 45, 23, 24],
};

/// Group 2: status and port events.
const GROUP_2: TraceEventGroup = TraceEventGroup {
    name: "status_ports",
    // CASCADE_STALL(25), ACTIVE(28), DISABLED(29), PORT_IDLE_0(74),
    // PORT_STALLED_0(76), PORT_TLAST_0(77), PORT_RUNNING_2(83), PORT_RUNNING_3(87)
    event_ids: [25, 28, 29, 74, 76, 77, 83, 87],
};

/// All trace event groups, indexed by group number.
pub const TRACE_EVENT_GROUPS: &[&TraceEventGroup] = &[&GROUP_0, &GROUP_1, &GROUP_2];

/// Number of trace event groups.
pub const NUM_GROUPS: usize = 3;

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
                let reg_off_lo = u32::from_le_bytes([
                    insts[i + 8],
                    insts[i + 9],
                    insts[i + 10],
                    insts[i + 11],
                ]);
                let reg_off_hi = u32::from_le_bytes([
                    insts[i + 12],
                    insts[i + 13],
                    insts[i + 14],
                    insts[i + 15],
                ]);

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
        format!(
            "Write32 for Trace_Event0 (addr 0x{:08X}) not found in insts.bin",
            event0_addr
        )
    })?;
    let off1 = find_write32_value_offset(insts, event1_addr).ok_or_else(|| {
        format!(
            "Write32 for Trace_Event1 (addr 0x{:08X}) not found in insts.bin",
            event1_addr
        )
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

/// Decode binary trace data (raw bytes from the trace buffer) into events.
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
    let mut events = Vec::new();

    // Each packet is 32 bytes (8 words of 4 bytes each).
    let packet_size = 32;
    let mut pkt_offset = 0;

    let mut abs_cycle: u64 = 0;
    // For repeat commands: replay the last event set.
    let mut last_events: Vec<DecodedEvent> = Vec::new();
    let mut last_delta: u64 = 0;

    while pkt_offset + packet_size <= data.len() {
        // Word 0 is header -- skip it.
        // Words 1-7 are data: extract 28 bytes (big-endian packed in u32s).
        let mut payload = [0u8; 28];
        for word_idx in 0..7 {
            let word_offset = pkt_offset + (word_idx + 1) * 4;
            let word = u32::from_be_bytes([
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
                        events.push(DecodedEvent {
                            slot: ev.slot,
                            abs_cycle,
                        });
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
                        events.push(DecodedEvent {
                            slot: ev.slot,
                            abs_cycle,
                        });
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
                let event_mask =
                    ((b as u16 & 0x03) << 6) | ((payload[i + 1] as u16) >> 2);
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
                let event_mask =
                    ((b as u16 & 0x03) << 6) | ((payload[i + 1] as u16) >> 2);
                let delta = (((payload[i + 1] & 0x03) as u64) << 8)
                    | (payload[i + 2] as u64);
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
                let event_mask =
                    ((b as u16 & 0x0F) << 4) | ((payload[i + 1] as u16) >> 4);
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
                let delta = (((b & 0x03) as u64) << 16)
                    | ((payload[i + 1] as u64) << 8)
                    | (payload[i + 2] as u64);
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
                let delta =
                    (((b & 0x03) as u64) << 8) | (payload[i + 1] as u64);
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
    }
}

/// Format a TraceComparison as a human-readable string.
impl std::fmt::Display for TraceComparison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.sequence_match {
            write!(
                f,
                "{}: MATCH ({} events)",
                self.group_name, self.npu_event_count,
            )
        } else if let Some((idx, npu_s, emu_s)) = self.first_divergence {
            write!(
                f,
                "{}: DIVERGE at index {} (npu_slot={}, emu_slot={}) \
                 [npu={} events, emu={} events]",
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
                self.emu_event_count,
            )
        } else {
            write!(f, "{}: EMPTY (no events)", self.group_name)
        }
    }
}

// -------------------------------------------------------------------------
// Sweep orchestration helpers
// -------------------------------------------------------------------------

/// Check bitwise determinism across multiple binary trace buffers.
///
/// Returns `true` if all buffers are byte-identical. Returns a description
/// of the first difference found if they diverge.
pub fn check_determinism(traces: &[Vec<u8>]) -> Result<(), String> {
    if traces.len() < 2 {
        return Ok(());
    }
    let reference = &traces[0];
    for (rep, trace) in traces.iter().enumerate().skip(1) {
        if trace.len() != reference.len() {
            return Err(format!(
                "rep {} length {} != rep 0 length {}",
                rep,
                trace.len(),
                reference.len()
            ));
        }
        for (byte_idx, (a, b)) in reference.iter().zip(trace.iter()).enumerate() {
            if a != b {
                return Err(format!(
                    "rep {} differs at byte {}: 0x{:02X} vs 0x{:02X}",
                    rep, byte_idx, a, b
                ));
            }
        }
    }
    Ok(())
}

/// Write a sweep summary file.
pub fn write_sweep_summary(
    sweep_dir: &Path,
    determinism: &Result<(), String>,
    comparisons: &[TraceComparison],
) -> std::io::Result<()> {
    use std::io::Write;
    let path = sweep_dir.join("sweep_summary.txt");
    let mut f = std::fs::File::create(&path)?;

    match determinism {
        Ok(()) => writeln!(f, "Determinism: PASS")?,
        Err(e) => writeln!(f, "Determinism: FAIL ({})", e)?,
    }

    for comp in comparisons {
        writeln!(f, "{}", comp)?;
    }
    Ok(())
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group0_matches_fuzz_template() {
        // Verify group 0 register values match fuzz_template.py hardcoded values.
        assert_eq!(
            GROUP_0.event0_value(),
            640680481,
            "Group 0 event0 should be 0x{:08X} (640680481), got 0x{:08X}",
            640680481u32,
            GROUP_0.event0_value(),
        );
        assert_eq!(
            GROUP_0.event1_value(),
            1330084396,
            "Group 0 event1 should be 0x{:08X} (1330084396), got 0x{:08X}",
            1330084396u32,
            GROUP_0.event1_value(),
        );
    }

    #[test]
    fn test_event_group_packing() {
        // Verify packing of known event IDs into register values.
        let group = TraceEventGroup {
            name: "test",
            event_ids: [1, 2, 3, 4, 5, 6, 7, 8],
        };
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

        // Build fake insts with group 0 values.
        let mut insts = Vec::new();
        insts.extend_from_slice(&make_write32(event0_addr, GROUP_0.event0_value()));
        insts.extend_from_slice(&make_write32(event1_addr, GROUP_0.event1_value()));

        // Patching with group 0 should be a no-op.
        let patched0 = patch_insts_for_group(&insts, &GROUP_0).unwrap();
        assert_eq!(insts, patched0, "Group 0 patch should be identity");

        // Patching with group 1 should change both value fields.
        let patched1 = patch_insts_for_group(&insts, &GROUP_1).unwrap();
        assert_ne!(insts, patched1, "Group 1 patch should differ");

        // Verify the patched values match group 1.
        let val0 = u32::from_le_bytes(patched1[16..20].try_into().unwrap());
        let val1 = u32::from_le_bytes(patched1[24 + 16..24 + 20].try_into().unwrap());
        assert_eq!(val0, GROUP_1.event0_value());
        assert_eq!(val1, GROUP_1.event1_value());

        // Patch back to group 0.
        let round_trip = patch_insts_for_group(&patched1, &GROUP_0).unwrap();
        assert_eq!(insts, round_trip, "Round-trip should restore original");
    }

    #[test]
    fn test_patch_missing_target() {
        // insts.bin with no matching Write32 should return Err.
        let insts = make_write32(0x12345678, 0);
        let result = patch_insts_for_group(&insts, &GROUP_0);
        assert!(result.is_err());
    }

    /// Build a trace packet from raw event bytes (helper for tests).
    fn make_trace_packet(event_bytes: &[u8]) -> Vec<u8> {
        // Header (word 0): minimal valid header.
        let header: u32 = 0x80000001; // parity=1, packet_id=1
        let mut packet = Vec::with_capacity(32);
        packet.extend_from_slice(&header.to_be_bytes());

        // Words 1-7: pack event_bytes big-endian, pad with 0xFE.
        let mut padded = [0xFEu8; 28];
        let copy_len = event_bytes.len().min(28);
        padded[..copy_len].copy_from_slice(&event_bytes[..copy_len]);

        for word_idx in 0..7 {
            let base = word_idx * 4;
            let word = u32::from_be_bytes([
                padded[base],
                padded[base + 1],
                padded[base + 2],
                padded[base + 3],
            ]);
            packet.extend_from_slice(&word.to_be_bytes());
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
            0xF0, 0, 0, 0, 0, 0, 0, 100, // Start marker: timer=100
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
        let npu = vec![
            DecodedEvent { slot: 1, abs_cycle: 100 },
            DecodedEvent { slot: 3, abs_cycle: 200 },
        ];
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
        let emu = vec![
            DecodedEvent { slot: 1, abs_cycle: 100 },
            DecodedEvent { slot: 3, abs_cycle: 200 },
        ];
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

    #[test]
    fn test_determinism_pass() {
        let traces = vec![vec![1, 2, 3], vec![1, 2, 3], vec![1, 2, 3]];
        assert!(check_determinism(&traces).is_ok());
    }

    #[test]
    fn test_determinism_fail_content() {
        let traces = vec![vec![1, 2, 3], vec![1, 2, 4]];
        let err = check_determinism(&traces).unwrap_err();
        assert!(err.contains("rep 1 differs at byte 2"));
    }

    #[test]
    fn test_determinism_fail_length() {
        let traces = vec![vec![1, 2, 3], vec![1, 2]];
        let err = check_determinism(&traces).unwrap_err();
        assert!(err.contains("length"));
    }

    #[test]
    fn test_determinism_single_trace() {
        assert!(check_determinism(&[vec![1, 2, 3]]).is_ok());
    }

    #[test]
    fn test_display_match() {
        let comp = TraceComparison {
            group_name: "default".to_string(),
            npu_event_count: 42,
            emu_event_count: 42,
            sequence_match: true,
            first_divergence: None,
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
        };
        let s = format!("{}", comp);
        assert!(s.contains("DIVERGE at index 5"));
    }
}
