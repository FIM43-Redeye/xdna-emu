//! VCD comparison engine: signal alignment and timeline extraction.
//!
//! Loads two VCD files (emulator output and aiesimulator output), resolves
//! every variable through the [`MappingTree`], aligns them by [`StatePath`],
//! and extracts value-change timelines for downstream comparison.
//!
//! This module handles only *loading and alignment*. The actual subsystem-level
//! comparison logic lives in the sweep module (Task 11); report generation is
//! in the report module (Task 12).
//!
//! # Usage
//!
//! ```ignore
//! use xdna_emu::vcd::compare::load_and_align;
//! use xdna_emu::vcd::mapping::MappingTree;
//!
//! let tree = build_mapping_tree();
//! let input = load_and_align("emu.vcd", "sim.vcd", &tree)?;
//! println!("aligned: {} pairs, {} emu-only, {} sim-only",
//!     input.pairs.len(), input.emu_only.len(), input.sim_only.len());
//! ```

use crate::vcd::mapping::MappingTree;
use crate::vcd::state_path::StatePath;
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// SignalValue
// ---------------------------------------------------------------------------

/// Possible signal values in a VCD timeline.
///
/// Most hardware signals are purely binary and fit in a `u128`. Multi-state
/// values (containing X or Z bits) are stored as a human-readable bit string
/// since their exact representation matters for debugging but not for
/// arithmetic comparison.
#[derive(Debug, Clone, PartialEq)]
pub enum SignalValue {
    /// A concrete integer value (covers 1-bit through 128-bit signals).
    Integer(u128),
    /// Multi-state (contains X or Z bits). Stored as a bit string.
    FourState(String),
}

impl fmt::Display for SignalValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SignalValue::Integer(v) => write!(f, "{}", v),
            SignalValue::FourState(s) => write!(f, "{}", s),
        }
    }
}

// ---------------------------------------------------------------------------
// ValueChange
// ---------------------------------------------------------------------------

/// A single value change at a specific timestamp.
#[derive(Debug, Clone)]
pub struct ValueChange {
    /// VCD timestamp in timescale units (from the time table).
    pub time: u64,
    /// The signal value at this timestamp.
    pub value: SignalValue,
}

// ---------------------------------------------------------------------------
// SignalTimeline
// ---------------------------------------------------------------------------

/// Timeline of value changes for a single signal.
///
/// Changes are sorted by ascending timestamp. Constructed by
/// [`extract_timelines`] from a VCD file.
#[derive(Debug, Clone)]
pub struct SignalTimeline {
    /// The canonical identity of this signal.
    pub path: StatePath,
    /// Value changes in ascending time order.
    pub changes: Vec<ValueChange>,
}

impl SignalTimeline {
    /// Number of value changes in the timeline.
    pub fn len(&self) -> usize {
        self.changes.len()
    }

    /// Whether the timeline has no value changes.
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }
}

// ---------------------------------------------------------------------------
// AlignedPair
// ---------------------------------------------------------------------------

/// A pair of timelines for the same signal from two different VCD sources.
///
/// Both fields are `Option` because a signal may appear in one VCD but not
/// the other (the mapping tree resolved it, but the variable was absent from
/// the other file).
#[derive(Debug)]
pub struct AlignedPair {
    /// The canonical identity shared by both timelines.
    pub path: StatePath,
    /// Emulator timeline (absent if the signal was not found in the emulator VCD).
    pub emu: Option<SignalTimeline>,
    /// Simulator timeline (absent if the signal was not found in the sim VCD).
    pub sim: Option<SignalTimeline>,
}

// ---------------------------------------------------------------------------
// ComparisonInput
// ---------------------------------------------------------------------------

/// All aligned signal pairs from two VCD files, ready for comparison.
///
/// Produced by [`load_and_align`]. Contains:
/// - `pairs`: signals found in at least one VCD, aligned by [`StatePath`].
/// - `emu_only` / `sim_only`: signals present in one VCD but entirely absent
///   from the other (these are coverage gaps, not comparison failures).
pub struct ComparisonInput {
    /// Aligned signal pairs, one per mapped [`StatePath`] found in either VCD.
    pub pairs: Vec<AlignedPair>,
    /// StatePaths found in emulator VCD but not in sim VCD.
    pub emu_only: Vec<StatePath>,
    /// StatePaths found in sim VCD but not in emulator VCD.
    pub sim_only: Vec<StatePath>,
}

impl ComparisonInput {
    /// Total number of aligned pairs (signals present in both VCDs).
    pub fn both_count(&self) -> usize {
        self.pairs
            .iter()
            .filter(|p| p.emu.is_some() && p.sim.is_some())
            .count()
    }
}

// ---------------------------------------------------------------------------
// Conversion: wellen SignalValue -> our SignalValue
// ---------------------------------------------------------------------------

/// Convert a wellen `SignalValue` to our owned `SignalValue`.
///
/// Binary data (2-state) is converted to a `u128`. Four-state and nine-state
/// values (containing X/Z/etc.) are stored as bit strings since they cannot
/// be represented as integers.
///
/// Signals wider than 128 bits are stored as `FourState` bit strings to avoid
/// overflow; this is a pragmatic choice since no AIE2 signal exceeds 256 bits
/// and the comparison engine works on string equality for four-state values.
fn convert_signal_value(sv: &wellen::SignalValue<'_>) -> SignalValue {
    match sv {
        wellen::SignalValue::Binary(data, bits) => {
            if *bits == 0 {
                return SignalValue::Integer(0);
            }
            if *bits > 128 {
                // Too wide for u128; fall back to bit string.
                let s = sv.to_bit_string().unwrap_or_default();
                return SignalValue::FourState(s);
            }
            // Convert big-endian bit-packed bytes to u128.
            // wellen stores binary data as 1-bit-per-bit packed into bytes,
            // MSB-first, with the first byte potentially containing fewer
            // than 8 bits.
            let mut result: u128 = 0;
            for &byte in data.iter() {
                result = (result << 8) | (byte as u128);
            }
            // Mask to the exact bit width. The first byte may have had
            // high bits set from a previous encoding; mask them off.
            let mask = if *bits >= 128 {
                u128::MAX
            } else {
                (1u128 << *bits) - 1
            };
            SignalValue::Integer(result & mask)
        }
        wellen::SignalValue::FourValue(_, _)
        | wellen::SignalValue::NineValue(_, _) => {
            let s = sv.to_bit_string().unwrap_or_default();
            SignalValue::FourState(s)
        }
        wellen::SignalValue::Real(r) => {
            // Reals are uncommon in hardware VCD; store the bit pattern.
            SignalValue::Integer(r.to_bits() as u128)
        }
        wellen::SignalValue::String(s) => {
            SignalValue::FourState(s.to_string())
        }
        wellen::SignalValue::Event => {
            // Events have no data value; record as zero.
            SignalValue::Integer(0)
        }
    }
}

// ---------------------------------------------------------------------------
// extract_timelines
// ---------------------------------------------------------------------------

/// Extract mapped signal timelines from a single VCD file.
///
/// Walks every variable in the VCD hierarchy, resolves its full dotted name
/// through the mapping tree, and (for mapped signals) loads the signal's
/// value-change data and builds a [`SignalTimeline`].
///
/// Unmapped variables are silently skipped.
pub fn extract_timelines(
    vcd_path: &str,
    tree: &MappingTree,
) -> Result<HashMap<StatePath, SignalTimeline>, String> {
    // Guard against missing files before calling wellen (which may panic).
    if !std::path::Path::new(vcd_path).exists() {
        return Err(format!("VCD file not found: {}", vcd_path));
    }

    let mut waveform = wellen::simple::read(vcd_path)
        .map_err(|e| format!("Failed to read VCD '{}': {:?}", vcd_path, e))?;

    let hierarchy = waveform.hierarchy();

    // Phase 1: Walk the hierarchy and collect (StatePath, SignalRef) pairs
    // for all mapped variables. We must collect them first because loading
    // signal data requires a mutable borrow on the waveform.
    let mut mapped: Vec<(StatePath, wellen::SignalRef)> = Vec::new();

    for var in hierarchy.iter_vars() {
        let full_name = var.full_name(hierarchy);
        let segments: Vec<&str> = full_name.split('.').collect();

        if let Some(state_path) = tree.resolve(&segments) {
            mapped.push((state_path, var.signal_ref()));
        }
    }

    if mapped.is_empty() {
        return Ok(HashMap::new());
    }

    // Phase 2: Load all mapped signals in a single batch for efficiency.
    let signal_refs: Vec<wellen::SignalRef> = mapped.iter().map(|(_, sr)| *sr).collect();
    waveform.load_signals(&signal_refs);

    // Phase 3: Build timelines from the loaded signal data.
    let time_table = waveform.time_table();
    let mut timelines = HashMap::new();

    for (state_path, signal_ref) in &mapped {
        let signal = match waveform.get_signal(*signal_ref) {
            Some(s) => s,
            None => continue, // Signal data could not be loaded; skip.
        };

        let mut changes = Vec::new();
        for (time_idx, sv) in signal.iter_changes() {
            let time = time_table
                .get(time_idx as usize)
                .copied()
                .unwrap_or(time_idx as u64);
            let value = convert_signal_value(&sv);
            changes.push(ValueChange { time, value });
        }

        let timeline = SignalTimeline {
            path: state_path.clone(),
            changes,
        };

        timelines.insert(state_path.clone(), timeline);
    }

    Ok(timelines)
}

// ---------------------------------------------------------------------------
// load_and_align
// ---------------------------------------------------------------------------

/// Load two VCD files and align signals by [`StatePath`].
///
/// For each [`StatePath`] that appears in at least one VCD, an [`AlignedPair`]
/// is created with the corresponding timelines. Paths present in only one
/// VCD are also reported via [`ComparisonInput::emu_only`] and
/// [`ComparisonInput::sim_only`].
///
/// # Errors
///
/// Returns a descriptive error string if either VCD file cannot be read.
pub fn load_and_align(
    emu_path: &str,
    sim_path: &str,
    tree: &MappingTree,
) -> Result<ComparisonInput, String> {
    let emu_signals = extract_timelines(emu_path, tree)?;
    let sim_signals = extract_timelines(sim_path, tree)?;

    let mut pairs = Vec::new();
    let mut emu_only = Vec::new();
    let mut sim_only = Vec::new();

    // Collect all unique StatePaths from both maps.
    let mut all_paths: Vec<StatePath> = Vec::new();
    for key in emu_signals.keys() {
        all_paths.push(key.clone());
    }
    for key in sim_signals.keys() {
        if !emu_signals.contains_key(key) {
            all_paths.push(key.clone());
        }
    }
    // Sort for deterministic output order (StatePath derives Eq+Hash but
    // not Ord; sort by Display string as a stable fallback).
    all_paths.sort_by(|a, b| a.to_string().cmp(&b.to_string()));

    for path in all_paths {
        let emu_tl = emu_signals.get(&path).cloned();
        let sim_tl = sim_signals.get(&path).cloned();

        match (&emu_tl, &sim_tl) {
            (Some(_), None) => emu_only.push(path.clone()),
            (None, Some(_)) => sim_only.push(path.clone()),
            _ => {}
        }

        pairs.push(AlignedPair {
            path,
            emu: emu_tl,
            sim: sim_tl,
        });
    }

    Ok(ComparisonInput {
        pairs,
        emu_only,
        sim_only,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vcd::state_path::*;

    // -------------------------------------------------------------------
    // SignalValue unit tests
    // -------------------------------------------------------------------

    #[test]
    fn signal_value_integer_equality() {
        assert_eq!(SignalValue::Integer(42), SignalValue::Integer(42));
        assert_ne!(SignalValue::Integer(42), SignalValue::Integer(43));
    }

    #[test]
    fn signal_value_four_state_equality() {
        let a = SignalValue::FourState("01xz".to_string());
        let b = SignalValue::FourState("01xz".to_string());
        let c = SignalValue::FourState("0000".to_string());
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn signal_value_mixed_types_not_equal() {
        let int_val = SignalValue::Integer(1);
        let fs_val = SignalValue::FourState("1".to_string());
        assert_ne!(int_val, fs_val);
    }

    #[test]
    fn signal_value_display() {
        assert_eq!(format!("{}", SignalValue::Integer(255)), "255");
        assert_eq!(format!("{}", SignalValue::FourState("01xz".to_string())), "01xz");
    }

    // -------------------------------------------------------------------
    // SignalTimeline unit tests
    // -------------------------------------------------------------------

    #[test]
    fn signal_timeline_len_and_empty() {
        let empty_tl = SignalTimeline {
            path: StatePath::LockValue { col: 0, row: 1, idx: 0 },
            changes: vec![],
        };
        assert!(empty_tl.is_empty());
        assert_eq!(empty_tl.len(), 0);

        let nonempty_tl = SignalTimeline {
            path: StatePath::LockValue { col: 0, row: 1, idx: 0 },
            changes: vec![ValueChange { time: 0, value: SignalValue::Integer(0) }],
        };
        assert!(!nonempty_tl.is_empty());
        assert_eq!(nonempty_tl.len(), 1);
    }

    // -------------------------------------------------------------------
    // AlignedPair unit tests
    // -------------------------------------------------------------------

    #[test]
    fn aligned_pair_both_present() {
        let path = StatePath::LockValue { col: 0, row: 1, idx: 0 };
        let tl = SignalTimeline {
            path: path.clone(),
            changes: vec![ValueChange { time: 0, value: SignalValue::Integer(0) }],
        };
        let pair = AlignedPair {
            path: path.clone(),
            emu: Some(tl.clone()),
            sim: Some(tl),
        };
        assert!(pair.emu.is_some());
        assert!(pair.sim.is_some());
    }

    #[test]
    fn aligned_pair_emu_only() {
        let path = StatePath::LockValue { col: 0, row: 1, idx: 0 };
        let pair = AlignedPair {
            path: path.clone(),
            emu: Some(SignalTimeline { path, changes: vec![] }),
            sim: None,
        };
        assert!(pair.emu.is_some());
        assert!(pair.sim.is_none());
    }

    #[test]
    fn aligned_pair_sim_only() {
        let path = StatePath::LockValue { col: 0, row: 1, idx: 0 };
        let pair = AlignedPair {
            path: path.clone(),
            emu: None,
            sim: Some(SignalTimeline { path, changes: vec![] }),
        };
        assert!(pair.emu.is_none());
        assert!(pair.sim.is_some());
    }

    // -------------------------------------------------------------------
    // ComparisonInput unit tests
    // -------------------------------------------------------------------

    #[test]
    fn comparison_input_both_count() {
        let path_a = StatePath::LockValue { col: 0, row: 1, idx: 0 };
        let path_b = StatePath::LockValue { col: 0, row: 1, idx: 1 };
        let tl_a = SignalTimeline {
            path: path_a.clone(),
            changes: vec![],
        };
        let tl_b = SignalTimeline {
            path: path_b.clone(),
            changes: vec![],
        };

        let input = ComparisonInput {
            pairs: vec![
                AlignedPair {
                    path: path_a.clone(),
                    emu: Some(tl_a.clone()),
                    sim: Some(tl_a),
                },
                AlignedPair {
                    path: path_b.clone(),
                    emu: Some(tl_b),
                    sim: None,
                },
            ],
            emu_only: vec![path_b],
            sim_only: vec![],
        };

        assert_eq!(input.both_count(), 1);
    }

    // -------------------------------------------------------------------
    // convert_signal_value unit tests
    // -------------------------------------------------------------------

    #[test]
    fn convert_binary_single_bit_zero() {
        let data = [0u8];
        let sv = wellen::SignalValue::Binary(&data, 1);
        let result = convert_signal_value(&sv);
        assert_eq!(result, SignalValue::Integer(0));
    }

    #[test]
    fn convert_binary_single_bit_one() {
        let data = [1u8];
        let sv = wellen::SignalValue::Binary(&data, 1);
        let result = convert_signal_value(&sv);
        assert_eq!(result, SignalValue::Integer(1));
    }

    #[test]
    fn convert_binary_eight_bits() {
        let data = [0xAB_u8];
        let sv = wellen::SignalValue::Binary(&data, 8);
        let result = convert_signal_value(&sv);
        assert_eq!(result, SignalValue::Integer(0xAB));
    }

    #[test]
    fn convert_binary_32_bits() {
        // 0x00000042 in big-endian bytes
        let data = [0x00u8, 0x00, 0x00, 0x42];
        let sv = wellen::SignalValue::Binary(&data, 32);
        let result = convert_signal_value(&sv);
        assert_eq!(result, SignalValue::Integer(0x42));
    }

    #[test]
    fn convert_binary_zero_bits() {
        let data = [];
        let sv = wellen::SignalValue::Binary(&data, 0);
        let result = convert_signal_value(&sv);
        assert_eq!(result, SignalValue::Integer(0));
    }

    #[test]
    fn convert_four_state_to_string() {
        // Four-state values contain X/Z; stored as bit strings.
        let data = [0b11_10_01_00u8]; // z x 1 0 in 4-state encoding
        let sv = wellen::SignalValue::FourValue(&data, 4);
        let result = convert_signal_value(&sv);
        match result {
            SignalValue::FourState(s) => {
                assert_eq!(s, "zx10", "four-state bit string mismatch");
            }
            other => panic!("expected FourState, got {:?}", other),
        }
    }

    #[test]
    fn convert_event_to_zero() {
        let sv = wellen::SignalValue::Event;
        let result = convert_signal_value(&sv);
        assert_eq!(result, SignalValue::Integer(0));
    }

    #[test]
    fn convert_real_preserves_bits() {
        let sv = wellen::SignalValue::Real(3.14);
        let result = convert_signal_value(&sv);
        let expected = 3.14f64.to_bits() as u128;
        assert_eq!(result, SignalValue::Integer(expected));
    }

    #[test]
    fn convert_string_to_four_state() {
        let sv = wellen::SignalValue::String("hello");
        let result = convert_signal_value(&sv);
        assert_eq!(result, SignalValue::FourState("hello".to_string()));
    }

    // -------------------------------------------------------------------
    // extract_timelines: missing file
    // -------------------------------------------------------------------

    #[test]
    fn extract_timelines_missing_file() {
        let tree = crate::vcd::mapping::MappingTree::builder().build();
        let result = extract_timelines("/nonexistent/path.vcd", &tree);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    // -------------------------------------------------------------------
    // extract_timelines: synthetic VCD
    // -------------------------------------------------------------------

    #[test]
    fn extract_timelines_from_synthetic_vcd() {
        use crate::vcd::lock_mapping::lock_mapping;
        use std::io::Write;

        let dir = std::env::temp_dir().join("vcd_compare_test");
        std::fs::create_dir_all(&dir).unwrap();
        let vcd_path = dir.join("test.vcd");

        // Write a minimal VCD with one lock value signal that changes
        // three times: 0 at t=0, 1 at t=10, 2 at t=20.
        let mut f = std::fs::File::create(&vcd_path).unwrap();
        writeln!(f, "$timescale 1ns $end").unwrap();
        writeln!(f, "$scope module top $end").unwrap();
        writeln!(f, "$scope module math_engine $end").unwrap();
        writeln!(f, "$scope module mem_row $end").unwrap();
        writeln!(f, "$scope module tile_0_1 $end").unwrap();
        writeln!(f, "$scope module locks $end").unwrap();
        writeln!(f, "$var wire 32 a value_0 $end").unwrap();
        writeln!(f, "$upscope $end").unwrap(); // locks
        writeln!(f, "$upscope $end").unwrap(); // tile_0_1
        writeln!(f, "$upscope $end").unwrap(); // mem_row
        writeln!(f, "$upscope $end").unwrap(); // math_engine
        writeln!(f, "$upscope $end").unwrap(); // top
        writeln!(f, "$enddefinitions $end").unwrap();
        writeln!(f, "#0").unwrap();
        writeln!(f, "b00000000000000000000000000000000 a").unwrap();
        writeln!(f, "#10").unwrap();
        writeln!(f, "b00000000000000000000000000000001 a").unwrap();
        writeln!(f, "#20").unwrap();
        writeln!(f, "b00000000000000000000000000000010 a").unwrap();
        drop(f);

        // Build a mapping tree that covers this signal.
        let tree = crate::vcd::mapping::MappingTree::builder()
            .scope("top")
            .scope("math_engine")
            .tile_group("mem_row", &[(0, 1)])
            .subsystem(lock_mapping(64))
            .done_tile_group()
            .build();

        let timelines = extract_timelines(vcd_path.to_str().unwrap(), &tree).unwrap();

        let path = StatePath::LockValue { col: 0, row: 1, idx: 0 };
        assert!(
            timelines.contains_key(&path),
            "expected LockValue(0,1,0) in timelines, got keys: {:?}",
            timelines.keys().collect::<Vec<_>>()
        );

        let tl = &timelines[&path];
        assert_eq!(tl.changes.len(), 3, "expected 3 value changes");
        assert_eq!(tl.changes[0].time, 0);
        assert_eq!(tl.changes[0].value, SignalValue::Integer(0));
        assert_eq!(tl.changes[1].time, 10);
        assert_eq!(tl.changes[1].value, SignalValue::Integer(1));
        assert_eq!(tl.changes[2].time, 20);
        assert_eq!(tl.changes[2].value, SignalValue::Integer(2));

        std::fs::remove_dir_all(&dir).ok();
    }

    // -------------------------------------------------------------------
    // extract_timelines: unmapped signals are skipped
    // -------------------------------------------------------------------

    #[test]
    fn extract_timelines_skips_unmapped() {
        use std::io::Write;

        let dir = std::env::temp_dir().join("vcd_compare_unmapped");
        std::fs::create_dir_all(&dir).unwrap();
        let vcd_path = dir.join("test.vcd");

        let mut f = std::fs::File::create(&vcd_path).unwrap();
        writeln!(f, "$timescale 1ns $end").unwrap();
        writeln!(f, "$scope module unmapped_scope $end").unwrap();
        writeln!(f, "$var wire 1 x some_signal $end").unwrap();
        writeln!(f, "$upscope $end").unwrap();
        writeln!(f, "$enddefinitions $end").unwrap();
        writeln!(f, "#0").unwrap();
        writeln!(f, "0x").unwrap();
        drop(f);

        // Empty tree: nothing maps.
        let tree = crate::vcd::mapping::MappingTree::builder().build();
        let timelines = extract_timelines(vcd_path.to_str().unwrap(), &tree).unwrap();
        assert!(timelines.is_empty(), "expected no mapped timelines");

        std::fs::remove_dir_all(&dir).ok();
    }

    // -------------------------------------------------------------------
    // load_and_align: synthetic dual-VCD test
    // -------------------------------------------------------------------

    #[test]
    fn load_and_align_synthetic() {
        use crate::vcd::lock_mapping::lock_mapping;
        use std::io::Write;

        let dir = std::env::temp_dir().join("vcd_compare_align");
        std::fs::create_dir_all(&dir).unwrap();

        // Emulator VCD: lock value_0 and value_1.
        let emu_path = dir.join("emu.vcd");
        {
            let mut f = std::fs::File::create(&emu_path).unwrap();
            writeln!(f, "$timescale 1ns $end").unwrap();
            writeln!(f, "$scope module top $end").unwrap();
            writeln!(f, "$scope module math_engine $end").unwrap();
            writeln!(f, "$scope module mem_row $end").unwrap();
            writeln!(f, "$scope module tile_0_1 $end").unwrap();
            writeln!(f, "$scope module locks $end").unwrap();
            writeln!(f, "$var wire 32 a value_0 $end").unwrap();
            writeln!(f, "$var wire 32 b value_1 $end").unwrap();
            writeln!(f, "$upscope $end").unwrap();
            writeln!(f, "$upscope $end").unwrap();
            writeln!(f, "$upscope $end").unwrap();
            writeln!(f, "$upscope $end").unwrap();
            writeln!(f, "$upscope $end").unwrap();
            writeln!(f, "$enddefinitions $end").unwrap();
            writeln!(f, "#0").unwrap();
            writeln!(f, "b00000000000000000000000000000000 a").unwrap();
            writeln!(f, "b00000000000000000000000000000000 b").unwrap();
            writeln!(f, "#5").unwrap();
            writeln!(f, "b00000000000000000000000000000001 a").unwrap();
        }

        // Simulator VCD: lock value_0 and value_2 (value_1 absent, value_2 extra).
        let sim_path = dir.join("sim.vcd");
        {
            let mut f = std::fs::File::create(&sim_path).unwrap();
            writeln!(f, "$timescale 1ns $end").unwrap();
            writeln!(f, "$scope module top $end").unwrap();
            writeln!(f, "$scope module math_engine $end").unwrap();
            writeln!(f, "$scope module mem_row $end").unwrap();
            writeln!(f, "$scope module tile_0_1 $end").unwrap();
            writeln!(f, "$scope module locks $end").unwrap();
            writeln!(f, "$var wire 32 a value_0 $end").unwrap();
            writeln!(f, "$var wire 32 c value_2 $end").unwrap();
            writeln!(f, "$upscope $end").unwrap();
            writeln!(f, "$upscope $end").unwrap();
            writeln!(f, "$upscope $end").unwrap();
            writeln!(f, "$upscope $end").unwrap();
            writeln!(f, "$upscope $end").unwrap();
            writeln!(f, "$enddefinitions $end").unwrap();
            writeln!(f, "#0").unwrap();
            writeln!(f, "b00000000000000000000000000000000 a").unwrap();
            writeln!(f, "b00000000000000000000000000000000 c").unwrap();
            writeln!(f, "#7").unwrap();
            writeln!(f, "b00000000000000000000000000000001 a").unwrap();
        }

        let tree = crate::vcd::mapping::MappingTree::builder()
            .scope("top")
            .scope("math_engine")
            .tile_group("mem_row", &[(0, 1)])
            .subsystem(lock_mapping(64))
            .done_tile_group()
            .build();

        let result = load_and_align(
            emu_path.to_str().unwrap(),
            sim_path.to_str().unwrap(),
            &tree,
        )
        .unwrap();

        // value_0: present in both -> aligned pair with both timelines.
        // value_1: emu only.
        // value_2: sim only.
        assert_eq!(result.pairs.len(), 3, "expected 3 pairs total");
        assert_eq!(result.both_count(), 1, "expected 1 pair with both present");
        assert_eq!(result.emu_only.len(), 1, "expected 1 emu-only path");
        assert_eq!(result.sim_only.len(), 1, "expected 1 sim-only path");

        // Check that value_0 pair has both timelines with correct data.
        let path_0 = StatePath::LockValue { col: 0, row: 1, idx: 0 };
        let pair_0 = result.pairs.iter().find(|p| p.path == path_0).unwrap();
        let emu_tl = pair_0.emu.as_ref().unwrap();
        let sim_tl = pair_0.sim.as_ref().unwrap();
        // Emu: 0 at t=0, 1 at t=5
        assert_eq!(emu_tl.changes.len(), 2);
        assert_eq!(emu_tl.changes[0].time, 0);
        assert_eq!(emu_tl.changes[1].time, 5);
        assert_eq!(emu_tl.changes[1].value, SignalValue::Integer(1));
        // Sim: 0 at t=0, 1 at t=7
        assert_eq!(sim_tl.changes.len(), 2);
        assert_eq!(sim_tl.changes[0].time, 0);
        assert_eq!(sim_tl.changes[1].time, 7);
        assert_eq!(sim_tl.changes[1].value, SignalValue::Integer(1));

        // Check emu_only contains value_1.
        let path_1 = StatePath::LockValue { col: 0, row: 1, idx: 1 };
        assert!(result.emu_only.contains(&path_1));

        // Check sim_only contains value_2.
        let path_2 = StatePath::LockValue { col: 0, row: 1, idx: 2 };
        assert!(result.sim_only.contains(&path_2));

        std::fs::remove_dir_all(&dir).ok();
    }

    // -------------------------------------------------------------------
    // load_and_align: missing file error
    // -------------------------------------------------------------------

    #[test]
    fn load_and_align_missing_emu_file() {
        let tree = crate::vcd::mapping::MappingTree::builder().build();
        let result = load_and_align("/nonexistent/emu.vcd", "/nonexistent/sim.vcd", &tree);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------
    // Real VCD integration test (skips if files absent)
    // -------------------------------------------------------------------

    #[test]
    fn extract_timelines_from_real_vcd() {
        let vcd_path = "/tmp/aiesim-test2/trace.vcd";
        if !std::path::Path::new(vcd_path).exists() {
            eprintln!("Skipping: {} not found (run aiesim to generate)", vcd_path);
            return;
        }

        use crate::vcd::lock_mapping::lock_mapping;
        use crate::vcd::dma_mapping::dma_mapping;

        let tree = crate::vcd::mapping::MappingTree::builder()
            .scope("top")
            .scope("math_engine")
            .tile_group("mem_row", &[(0, 1)])
            .subsystem(lock_mapping(64))
            .subsystem(dma_mapping(6, 6))
            .done_tile_group()
            .build();

        let timelines = extract_timelines(vcd_path, &tree).unwrap();

        assert!(
            !timelines.is_empty(),
            "expected at least some mapped timelines from real VCD"
        );

        // Print a summary of what we found.
        eprintln!("Extracted {} timelines from real VCD:", timelines.len());
        for (path, tl) in &timelines {
            eprintln!("  {} -- {} changes", path, tl.changes.len());
        }
    }
}
