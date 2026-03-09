//! Piecewise-linear alignment between HW and EMU cycle timelines.
//!
//! Anchor pairs define known correspondences between HW and EMU cycles
//! (e.g., both sides agree that "DMA_START happened here"). Between
//! anchors, cycles are linearly interpolated. Before the first or after
//! the last anchor, a constant offset from the nearest anchor is applied.
//!
//! The unified coordinate system uses EMU cycles as the basis. HW cycles
//! are mapped into this space by applying the piecewise offset derived
//! from the anchors.

/// A single anchor: a known (hw_cycle, emu_cycle) correspondence.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Anchor {
    hw: u64,
    emu: u64,
}

/// Piecewise-linear mapping between HW and EMU cycle domains.
///
/// With zero anchors, both `hw_to_unified` and `emu_to_unified` are
/// identity (return the input cycle as f64). With one anchor, a constant
/// offset is applied. With two or more, interpolation occurs between
/// adjacent anchors and constant extrapolation outside the range.
#[derive(Debug, Clone)]
pub struct AlignmentMap {
    /// Sorted by hw_cycle (ascending). Invariant: no duplicate hw values.
    anchors: Vec<Anchor>,
}

impl AlignmentMap {
    /// Create an empty alignment map (identity mapping).
    pub fn identity() -> Self {
        Self {
            anchors: Vec::new(),
        }
    }

    /// Create an alignment map from a single anchor pair.
    pub fn from_single_anchor(hw: u64, emu: u64) -> Self {
        Self {
            anchors: vec![Anchor { hw, emu }],
        }
    }

    /// Insert an anchor, maintaining sort order by hw_cycle.
    ///
    /// If an anchor with the same hw_cycle already exists, it is replaced.
    pub fn add_anchor(&mut self, hw: u64, emu: u64) {
        let anchor = Anchor { hw, emu };
        match self.anchors.binary_search_by_key(&hw, |a| a.hw) {
            Ok(idx) => self.anchors[idx] = anchor,
            Err(idx) => self.anchors.insert(idx, anchor),
        }
    }

    /// Number of anchor pairs in this map.
    pub fn anchor_count(&self) -> usize {
        self.anchors.len()
    }

    /// Map a HW cycle into the unified (EMU-based) coordinate system.
    ///
    /// - Zero anchors: identity (returns `hw_cycle` as f64).
    /// - One anchor `(h0, e0)`: constant offset `hw_cycle + (e0 - h0)`.
    /// - Multiple anchors: piecewise-linear interpolation between adjacent
    ///   anchors, constant extrapolation outside the range.
    pub fn hw_to_unified(&self, hw_cycle: u64) -> f64 {
        let hw = hw_cycle as f64;

        match self.anchors.len() {
            0 => hw,
            1 => {
                let a = &self.anchors[0];
                hw + (a.emu as f64 - a.hw as f64)
            }
            _ => {
                let first = &self.anchors[0];
                let last = &self.anchors[self.anchors.len() - 1];

                if hw_cycle <= first.hw {
                    // Before first anchor: constant offset from first anchor.
                    hw + (first.emu as f64 - first.hw as f64)
                } else if hw_cycle >= last.hw {
                    // After last anchor: constant offset from last anchor.
                    hw + (last.emu as f64 - last.hw as f64)
                } else {
                    // Between two anchors: find the segment and interpolate.
                    // Binary search for the right segment.
                    let idx = self
                        .anchors
                        .partition_point(|a| a.hw <= hw_cycle)
                        .saturating_sub(1);
                    let a = &self.anchors[idx];
                    let b = &self.anchors[idx + 1];

                    let hw_span = (b.hw - a.hw) as f64;
                    let emu_span = b.emu as f64 - a.emu as f64;
                    let t = (hw - a.hw as f64) / hw_span;

                    a.emu as f64 + t * emu_span
                }
            }
        }
    }

    /// Map an EMU cycle into the unified coordinate system.
    ///
    /// EMU cycles ARE the unified coordinate system, so this is identity.
    pub fn emu_to_unified(&self, emu_cycle: u64) -> f64 {
        emu_cycle as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_map_is_identity() {
        let map = AlignmentMap::identity();
        assert_eq!(map.anchor_count(), 0);
        assert_eq!(map.hw_to_unified(0), 0.0);
        assert_eq!(map.hw_to_unified(1000), 1000.0);
        assert_eq!(map.emu_to_unified(500), 500.0);
    }

    #[test]
    fn single_anchor_constant_offset() {
        // HW started at cycle 100, EMU started at cycle 200.
        // Offset = 200 - 100 = +100.
        let map = AlignmentMap::from_single_anchor(100, 200);
        assert_eq!(map.anchor_count(), 1);

        // hw_cycle 100 -> unified 200 (the anchor itself)
        assert_eq!(map.hw_to_unified(100), 200.0);
        // hw_cycle 0 -> unified 100 (before anchor, constant offset)
        assert_eq!(map.hw_to_unified(0), 100.0);
        // hw_cycle 300 -> unified 400 (after anchor, constant offset)
        assert_eq!(map.hw_to_unified(300), 400.0);

        // EMU is always identity
        assert_eq!(map.emu_to_unified(200), 200.0);
    }

    #[test]
    fn two_anchors_interpolation_midpoint() {
        let mut map = AlignmentMap::identity();
        // HW 100 = EMU 200, HW 200 = EMU 400.
        // At midpoint hw=150, we expect emu=300 (linear interpolation).
        map.add_anchor(100, 200);
        map.add_anchor(200, 400);
        assert_eq!(map.anchor_count(), 2);

        // At the anchors themselves
        assert_eq!(map.hw_to_unified(100), 200.0);
        assert_eq!(map.hw_to_unified(200), 400.0);

        // Midpoint interpolation
        assert_eq!(map.hw_to_unified(150), 300.0);

        // Quarter point
        assert_eq!(map.hw_to_unified(125), 250.0);

        // Before first anchor: constant offset from first (200 - 100 = +100)
        assert_eq!(map.hw_to_unified(50), 150.0);

        // After last anchor: constant offset from last (400 - 200 = +200)
        assert_eq!(map.hw_to_unified(300), 500.0);
    }

    #[test]
    fn add_anchor_maintains_sort_order() {
        let mut map = AlignmentMap::identity();
        // Insert out of order
        map.add_anchor(300, 600);
        map.add_anchor(100, 200);
        map.add_anchor(200, 400);
        assert_eq!(map.anchor_count(), 3);

        // Verify they are sorted: anchors at hw 100, 200, 300
        assert_eq!(map.hw_to_unified(100), 200.0);
        assert_eq!(map.hw_to_unified(200), 400.0);
        assert_eq!(map.hw_to_unified(300), 600.0);

        // Interpolation between first two: hw 150 -> emu 300
        assert_eq!(map.hw_to_unified(150), 300.0);

        // Interpolation between last two: hw 250 -> emu 500
        assert_eq!(map.hw_to_unified(250), 500.0);
    }

    #[test]
    fn add_anchor_replaces_duplicate_hw() {
        let mut map = AlignmentMap::identity();
        map.add_anchor(100, 200);
        assert_eq!(map.anchor_count(), 1);
        assert_eq!(map.hw_to_unified(100), 200.0);

        // Replace the anchor at hw=100 with a different emu value
        map.add_anchor(100, 300);
        assert_eq!(map.anchor_count(), 1);
        assert_eq!(map.hw_to_unified(100), 300.0);
    }

    #[test]
    fn before_first_anchor_extrapolation() {
        let mut map = AlignmentMap::identity();
        map.add_anchor(500, 1000);
        map.add_anchor(1000, 1500);

        // Before first anchor: offset = 1000 - 500 = +500
        assert_eq!(map.hw_to_unified(0), 500.0);
        assert_eq!(map.hw_to_unified(250), 750.0);
        assert_eq!(map.hw_to_unified(500), 1000.0);
    }

    #[test]
    fn emu_to_unified_always_identity() {
        let map = AlignmentMap::from_single_anchor(100, 200);
        assert_eq!(map.emu_to_unified(0), 0.0);
        assert_eq!(map.emu_to_unified(100), 100.0);
        assert_eq!(map.emu_to_unified(999), 999.0);
    }

    #[test]
    fn three_segments_with_different_rates() {
        let mut map = AlignmentMap::identity();
        // Segment 1: HW 0..100 -> EMU 0..100 (1:1 rate, offset 0)
        // Segment 2: HW 100..200 -> EMU 100..300 (1:2 rate, EMU runs 2x faster)
        // Segment 3: after HW 200, constant offset +100
        map.add_anchor(0, 0);
        map.add_anchor(100, 100);
        map.add_anchor(200, 300);

        // Within segment 1: 1:1
        assert_eq!(map.hw_to_unified(50), 50.0);

        // Within segment 2: 1:2
        assert_eq!(map.hw_to_unified(150), 200.0);

        // At boundaries
        assert_eq!(map.hw_to_unified(0), 0.0);
        assert_eq!(map.hw_to_unified(100), 100.0);
        assert_eq!(map.hw_to_unified(200), 300.0);

        // After last: constant offset from last (300 - 200 = +100)
        assert_eq!(map.hw_to_unified(250), 350.0);
    }
}
