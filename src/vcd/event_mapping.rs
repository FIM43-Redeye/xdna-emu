//! Event trace subsystem signal mapping.
//!
//! VCD hierarchy for event trace signals (from aiesimulator output):
//!
//! ```text
//! tile_0_1.event_trace.event0_none             (1-bit)
//! tile_0_1.event_trace.event1_true             (1-bit)
//! tile_0_1.event_trace.event2_group_0_error    (1-bit)
//! tile_0_1.event_trace.event73_INSTR_VECTOR    (1-bit)
//! tile_0_1.event_trace.event85_DMA_S2MM_0_START_TASK  (1-bit)
//! ```
//!
//! The scope name is `event_trace`. Each signal is named
//! `event{code}_{name}` where `{code}` is a decimal event code and
//! `{name}` is the human-readable event name (with underscores replacing
//! spaces). Both are embedded in the signal name.
//!
//! # Design
//!
//! Unlike locks or DMA, the set of event signals is not fixed at compile
//! time -- it varies by tile type and aietools version. The signal names are
//! self-describing (they embed the code and name), so resolution is done by
//! parsing the `event{N}_{name}` format dynamically.
//!
//! This mapping does not use [`SubsystemMapping`]'s builder methods (which
//! require a known-at-construction-time signal list). Instead, [`EventMapping`]
//! is a standalone type that implements pattern-based resolution via the
//! `event{code}_{name}` naming convention.
//!
//! # Maps to
//!
//! - `event{code}_{name}` -> [`StatePath::EventTrace`] with `event_code` and
//!   `event_name` parsed from the signal name.
//!
//! # Integration note
//!
//! Like [`StreamPortMapping`](super::stream_mapping::StreamPortMapping), this
//! type cannot be directly passed to the standard mapping tree builder
//! (which only accepts [`SubsystemMapping`]). Task 8 will extend the tree to
//! accept both types.

use crate::vcd::mapping::{SubsystemMapping, TileMapping};
use crate::vcd::state_path::{StatePath, Subsystem};

// ---------------------------------------------------------------------------
// EventMapping -- pattern-based event signal resolution
// ---------------------------------------------------------------------------

/// Pattern-based mapping for event trace signals.
///
/// Resolves any signal of the form `event{code}_{name}` to a
/// [`StatePath::EventTrace`] by parsing the signal name. Both the event code
/// (decimal integer) and event name (the rest of the string after the first
/// underscore) are extracted from the signal name.
///
/// The scope name in the VCD is `event_trace`.
pub struct EventMapping;

impl EventMapping {
    /// The VCD scope name for the event trace subsystem.
    pub fn scope_name(&self) -> &str {
        "event_trace"
    }

    /// Attempt to resolve a single VCD segment to a [`StatePath::EventTrace`].
    ///
    /// `segments` must contain exactly one element with the pattern
    /// `event{N}_{name}`. Returns `None` if the segment does not match.
    ///
    /// # Parsing rules
    ///
    /// - The signal name must start with the literal prefix `"event"`.
    /// - After the prefix, a decimal integer (the event code) follows.
    /// - After the integer, an underscore separates the code from the name.
    /// - The remainder (everything after the first underscore) is the event
    ///   name, which may itself contain underscores.
    ///
    /// # Examples
    ///
    /// ```
    /// // "event0_none"  -> event_code: 0, event_name: "none"
    /// // "event73_INSTR_VECTOR" -> event_code: 73, event_name: "INSTR_VECTOR"
    /// // "event85_DMA_S2MM_0_START_TASK" -> event_code: 85, event_name: "DMA_S2MM_0_START_TASK"
    /// ```
    pub fn resolve(&self, segments: &[&str], col: u8, row: u8) -> Option<StatePath> {
        if segments.len() != 1 {
            return None;
        }
        parse_event_signal(segments[0], col, row)
    }

    /// Enumerate all possible [`StatePath::EventTrace`] values for a tile.
    ///
    /// This is not implementable without knowing the exact set of events
    /// defined in the VCD (since event codes and names are tile-type-specific
    /// and aietools-version-specific). Returns an empty vec.
    ///
    /// For the tree, event enumeration is handled by reading the VCD header
    /// and resolving signals dynamically, rather than by static enumeration.
    pub fn enumerate(&self, _col: u8, _row: u8) -> Vec<StatePath> {
        // Event enumeration is not static -- the set of events is determined
        // at VCD parse time from the signal names. See module docs.
        Vec::new()
    }
}

impl TileMapping for EventMapping {
    fn scope_name(&self) -> &str {
        EventMapping::scope_name(self)
    }

    fn resolve(&self, segments: &[&str], col: u8, row: u8) -> Option<StatePath> {
        EventMapping::resolve(self, segments, col, row)
    }

    fn enumerate(&self, col: u8, row: u8) -> Vec<StatePath> {
        EventMapping::enumerate(self, col, row)
    }
}

// ---------------------------------------------------------------------------
// Signal name parser
// ---------------------------------------------------------------------------

/// Parse a VCD signal name of the form `event{code}_{name}` into a
/// [`StatePath::EventTrace`].
///
/// Returns `None` if the name does not match the expected pattern.
fn parse_event_signal(name: &str, col: u8, row: u8) -> Option<StatePath> {
    // Strip the "event" prefix.
    let rest = name.strip_prefix("event")?;

    // Find the first underscore to separate code from name.
    let underscore_pos = rest.find('_')?;
    let code_str = &rest[..underscore_pos];
    let event_name = &rest[underscore_pos + 1..];

    // Parse the event code as a decimal integer.
    let event_code: u16 = code_str.parse().ok()?;

    Some(StatePath::EventTrace { col, row, event_code, event_name: event_name.to_string() })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build the event trace subsystem mapping for a tile.
///
/// Returns an [`EventMapping`] that resolves VCD signals of the form
/// `event{code}_{name}` under the `event_trace` scope to
/// [`StatePath::EventTrace`] values.
///
/// # Pattern matching
///
/// Unlike other subsystem mappings, event signals cannot be enumerated
/// statically because the event code/name set varies by tile type and
/// aietools version. Resolution is done dynamically by parsing the signal
/// name at resolution time.
///
/// # VCD scope
///
/// The VCD scope name is `"event_trace"`.
pub fn event_mapping() -> EventMapping {
    EventMapping
}

/// Build a [`SubsystemMapping`] adapter for the event trace scope.
///
/// This adapter allows the event trace scope to be registered with the
/// standard mapping tree builder (which expects [`SubsystemMapping`]).
/// However, since events are dynamically resolved, the returned mapping has
/// no statically-defined signals. Resolution of event signals is handled by
/// [`EventMapping`] in a separate code path.
///
/// Task 8 will wire both types into the mapping tree.
pub fn event_mapping_as_subsystem() -> SubsystemMapping {
    // Empty mapping with the correct scope name. Event signals are handled
    // by EventMapping via dynamic pattern matching.
    SubsystemMapping::new("event_trace", Subsystem::Event)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vcd::state_path::StatePath;

    // -- Resolution tests --

    #[test]
    fn event_resolves_event0_none() {
        let mapping = event_mapping();
        let result = mapping.resolve(&["event0_none"], 0, 1);
        assert_eq!(
            result,
            Some(StatePath::EventTrace { col: 0, row: 1, event_code: 0, event_name: "none".to_string() })
        );
    }

    #[test]
    fn event_resolves_event1_true() {
        let mapping = event_mapping();
        let result = mapping.resolve(&["event1_true"], 0, 1);
        assert_eq!(
            result,
            Some(StatePath::EventTrace { col: 0, row: 1, event_code: 1, event_name: "true".to_string() })
        );
    }

    #[test]
    fn event_resolves_event73_instr_vector() {
        let mapping = event_mapping();
        let result = mapping.resolve(&["event73_INSTR_VECTOR"], 0, 3);
        assert_eq!(
            result,
            Some(StatePath::EventTrace {
                col: 0,
                row: 3,
                event_code: 73,
                event_name: "INSTR_VECTOR".to_string(),
            })
        );
    }

    #[test]
    fn event_resolves_event85_dma_s2mm_start_task() {
        // Event names can contain multiple underscores.
        let mapping = event_mapping();
        let result = mapping.resolve(&["event85_DMA_S2MM_0_START_TASK"], 1, 2);
        assert_eq!(
            result,
            Some(StatePath::EventTrace {
                col: 1,
                row: 2,
                event_code: 85,
                event_name: "DMA_S2MM_0_START_TASK".to_string(),
            })
        );
    }

    #[test]
    fn event_code_extracted_correctly() {
        // Verify the event_code field holds the correct parsed integer.
        let mapping = event_mapping();
        let result = mapping.resolve(&["event73_INSTR_VECTOR"], 0, 3);
        assert!(
            matches!(result, Some(StatePath::EventTrace { event_code: 73, .. })),
            "Expected event_code=73, got {:?}",
            result
        );
    }

    #[test]
    fn event_name_extracted_correctly() {
        // Verify the event_name is everything after the first underscore.
        let mapping = event_mapping();
        if let Some(StatePath::EventTrace { event_name, .. }) =
            mapping.resolve(&["event73_INSTR_VECTOR"], 0, 3)
        {
            assert_eq!(event_name, "INSTR_VECTOR");
        } else {
            panic!("Expected EventTrace for event73_INSTR_VECTOR");
        }
    }

    #[test]
    fn event_name_with_underscores_preserves_full_name() {
        // The event name is the full string after the first underscore,
        // including any subsequent underscores.
        let mapping = event_mapping();
        if let Some(StatePath::EventTrace { event_name, .. }) =
            mapping.resolve(&["event85_DMA_S2MM_0_START_TASK"], 0, 1)
        {
            assert_eq!(event_name, "DMA_S2MM_0_START_TASK");
        } else {
            panic!("Expected EventTrace for event85_DMA_S2MM_0_START_TASK");
        }
    }

    #[test]
    fn event_tile_coordinates_propagated() {
        let mapping = event_mapping();
        let a = mapping.resolve(&["event0_none"], 0, 1).unwrap();
        let b = mapping.resolve(&["event0_none"], 3, 5).unwrap();
        assert_ne!(a, b);
        if let (
            StatePath::EventTrace { col: ca, row: ra, .. },
            StatePath::EventTrace { col: cb, row: rb, .. },
        ) = (&a, &b)
        {
            assert_eq!((*ca, *ra), (0u8, 1u8));
            assert_eq!((*cb, *rb), (3u8, 5u8));
        } else {
            panic!("Expected EventTrace variants");
        }
    }

    #[test]
    fn event_scope_name_is_event_trace() {
        let mapping = event_mapping();
        assert_eq!(mapping.scope_name(), "event_trace");
    }

    // -- Rejection tests --

    #[test]
    fn event_rejects_no_event_prefix() {
        let mapping = event_mapping();
        // Signal does not start with "event".
        assert_eq!(mapping.resolve(&["lock_value_3"], 0, 1), None);
    }

    #[test]
    fn event_rejects_event_without_code() {
        // "event_none" has no numeric code between prefix and underscore.
        let mapping = event_mapping();
        assert_eq!(mapping.resolve(&["event_none"], 0, 1), None);
    }

    #[test]
    fn event_rejects_event_with_no_name() {
        // "event73" has no underscore separator and no name.
        let mapping = event_mapping();
        assert_eq!(mapping.resolve(&["event73"], 0, 1), None);
    }

    #[test]
    fn event_rejects_non_numeric_code() {
        // "eventXX_foo" has a non-numeric code.
        let mapping = event_mapping();
        assert_eq!(mapping.resolve(&["eventXX_foo"], 0, 1), None);
    }

    #[test]
    fn event_rejects_wrong_segment_count() {
        let mapping = event_mapping();
        // Too many segments (events are flat, not nested).
        assert_eq!(mapping.resolve(&["event0_none", "extra"], 0, 1), None);
        // Empty segments.
        assert_eq!(mapping.resolve(&[], 0, 1), None);
    }

    #[test]
    fn event_rejects_code_out_of_u16_range() {
        // Code larger than u16::MAX (65535).
        let mapping = event_mapping();
        assert_eq!(mapping.resolve(&["event99999_foo"], 0, 1), None);
    }

    // -- Enumeration test --

    #[test]
    fn event_enumerate_returns_empty() {
        // Event enumeration is not static -- events are discovered from the VCD.
        let mapping = event_mapping();
        let paths = mapping.enumerate(0, 1);
        assert!(
            paths.is_empty(),
            "Expected empty enumeration for event mapping, got {} entries",
            paths.len()
        );
    }

    // -- Standalone parser tests --

    #[test]
    fn parse_event_signal_basic() {
        let result = parse_event_signal("event0_none", 0, 1);
        assert!(result.is_some());
    }

    #[test]
    fn parse_event_signal_high_code() {
        // Codes up to 65535 are valid.
        let result = parse_event_signal("event65535_max", 0, 0);
        assert_eq!(
            result,
            Some(StatePath::EventTrace { col: 0, row: 0, event_code: 65535, event_name: "max".to_string() })
        );
    }

    #[test]
    fn parse_event_signal_lowercase_name() {
        // Event names can be lowercase.
        let result = parse_event_signal("event2_group_0_error", 0, 0);
        assert_eq!(
            result,
            Some(StatePath::EventTrace {
                col: 0,
                row: 0,
                event_code: 2,
                event_name: "group_0_error".to_string(),
            })
        );
    }
}
