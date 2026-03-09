//! Viewport state for the trace timeline.
//!
//! Maps between cycle numbers and screen pixel coordinates. Handles zoom
//! (with anchor preservation) and pan operations for smooth navigation.

/// Minimum zoom level (pixels per cycle). At this level, one pixel spans
/// 10,000 cycles -- suitable for viewing millions of cycles at once.
const MIN_PX_PER_CYCLE: f64 = 0.0001;

/// Maximum zoom level (pixels per cycle). At this level, each cycle is
/// 100 pixels wide -- suitable for inspecting individual events.
const MAX_PX_PER_CYCLE: f64 = 100.0;

/// Viewport mapping between cycle space and pixel space.
///
/// The viewport defines a window into the cycle timeline. `start_cycle` is
/// the left edge, and the right edge is determined by `start_cycle + width /
/// px_per_cycle`. Zooming adjusts `px_per_cycle`; panning adjusts
/// `start_cycle`.
#[derive(Debug, Clone)]
pub struct Viewport {
    /// First visible cycle (left edge of the viewport).
    pub start_cycle: f64,
    /// Pixels per cycle (zoom level). Higher values = more zoomed in.
    pub px_per_cycle: f64,
    /// Total viewport width in pixels (updated each frame from available width).
    pub width_px: f32,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            start_cycle: 0.0,
            px_per_cycle: 0.1,
            width_px: 1000.0,
        }
    }
}

impl Viewport {
    /// Cycle at the right edge of the viewport.
    pub fn end_cycle(&self) -> f64 {
        self.start_cycle + self.width_px as f64 / self.px_per_cycle
    }

    /// Number of cycles visible in the viewport.
    pub fn visible_cycles(&self) -> f64 {
        self.width_px as f64 / self.px_per_cycle
    }

    /// Convert a cycle number to a pixel x-offset relative to the viewport
    /// left edge.
    pub fn cycle_to_px(&self, cycle: f64) -> f32 {
        ((cycle - self.start_cycle) * self.px_per_cycle) as f32
    }

    /// Convert a pixel x-offset (relative to viewport left edge) to a cycle
    /// number.
    pub fn px_to_cycle(&self, px: f32) -> f64 {
        self.start_cycle + px as f64 / self.px_per_cycle
    }

    /// Whether a cycle falls within the visible range.
    pub fn is_visible(&self, cycle: f64) -> bool {
        cycle >= self.start_cycle && cycle <= self.end_cycle()
    }

    /// Zoom by `factor` around a pixel anchor point.
    ///
    /// The cycle under `anchor_px` stays fixed on screen. A factor > 1.0
    /// zooms in (more pixels per cycle); < 1.0 zooms out.
    /// `px_per_cycle` is clamped to [`MIN_PX_PER_CYCLE`, `MAX_PX_PER_CYCLE`].
    pub fn zoom_at(&mut self, factor: f64, anchor_px: f32) {
        // Cycle under the anchor before zoom.
        let anchor_cycle = self.px_to_cycle(anchor_px);

        // Apply zoom.
        self.px_per_cycle = (self.px_per_cycle * factor).clamp(MIN_PX_PER_CYCLE, MAX_PX_PER_CYCLE);

        // Adjust start_cycle so the anchor cycle stays at the same pixel.
        self.start_cycle = anchor_cycle - anchor_px as f64 / self.px_per_cycle;
    }

    /// Pan by a pixel delta (positive = scroll right, revealing later cycles).
    ///
    /// Internally this shifts `start_cycle` by `-delta / px_per_cycle` so
    /// that dragging right shows earlier cycles (standard scroll behavior).
    pub fn pan_px(&mut self, delta_px: f32) {
        self.start_cycle -= delta_px as f64 / self.px_per_cycle;
    }

    /// Fit a cycle range into the viewport, with a small margin.
    ///
    /// If `min == max`, uses a default span of 100 cycles centered on the
    /// value. A 5% margin is added on each side for visual breathing room.
    pub fn fit_range(&mut self, min: f64, max: f64) {
        let (min, max) = if (max - min).abs() < 1.0 {
            // Degenerate range -- show 100 cycles centered on the value.
            (min - 50.0, max + 50.0)
        } else {
            let margin = (max - min) * 0.05;
            (min - margin, max + margin)
        };

        let span = max - min;
        self.start_cycle = min;
        self.px_per_cycle = (self.width_px as f64 / span).clamp(MIN_PX_PER_CYCLE, MAX_PX_PER_CYCLE);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values() {
        let vp = Viewport::default();
        assert_eq!(vp.start_cycle, 0.0);
        assert!((vp.px_per_cycle - 0.1).abs() < f64::EPSILON);
        assert_eq!(vp.width_px, 1000.0);
    }

    #[test]
    fn end_cycle_and_visible_cycles() {
        let vp = Viewport {
            start_cycle: 100.0,
            px_per_cycle: 0.5,
            width_px: 500.0,
        };
        // visible_cycles = 500 / 0.5 = 1000
        assert!((vp.visible_cycles() - 1000.0).abs() < f64::EPSILON);
        // end_cycle = 100 + 1000 = 1100
        assert!((vp.end_cycle() - 1100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn roundtrip_cycle_to_px_and_back() {
        let vp = Viewport {
            start_cycle: 50.0,
            px_per_cycle: 2.0,
            width_px: 800.0,
        };
        let cycle = 175.0;
        let px = vp.cycle_to_px(cycle);
        let back = vp.px_to_cycle(px);
        assert!((back - cycle).abs() < 1e-9, "roundtrip failed: {} != {}", back, cycle);
    }

    #[test]
    fn cycle_to_px_basic() {
        let vp = Viewport {
            start_cycle: 100.0,
            px_per_cycle: 2.0,
            width_px: 800.0,
        };
        // cycle 100 -> px 0 (left edge)
        assert!((vp.cycle_to_px(100.0) - 0.0).abs() < f32::EPSILON);
        // cycle 200 -> px 200 (200-100)*2 = 200
        assert!((vp.cycle_to_px(200.0) - 200.0).abs() < f32::EPSILON);
        // cycle 50 -> px -100 (before viewport, negative)
        assert!((vp.cycle_to_px(50.0) - (-100.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn px_to_cycle_basic() {
        let vp = Viewport {
            start_cycle: 100.0,
            px_per_cycle: 2.0,
            width_px: 800.0,
        };
        // px 0 -> cycle 100
        assert!((vp.px_to_cycle(0.0) - 100.0).abs() < f64::EPSILON);
        // px 200 -> cycle 200
        assert!((vp.px_to_cycle(200.0) - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn is_visible_range() {
        let vp = Viewport {
            start_cycle: 100.0,
            px_per_cycle: 1.0,
            width_px: 500.0,
        };
        // Visible range: [100, 600]
        assert!(vp.is_visible(100.0));
        assert!(vp.is_visible(300.0));
        assert!(vp.is_visible(600.0));
        assert!(!vp.is_visible(99.9));
        assert!(!vp.is_visible(600.1));
    }

    #[test]
    fn zoom_preserves_anchor() {
        let mut vp = Viewport {
            start_cycle: 0.0,
            px_per_cycle: 1.0,
            width_px: 1000.0,
        };

        // Place anchor at pixel 400 -> cycle 400.
        let anchor_px = 400.0_f32;
        let anchor_cycle_before = vp.px_to_cycle(anchor_px);

        // Zoom in by 2x.
        vp.zoom_at(2.0, anchor_px);

        let anchor_cycle_after = vp.px_to_cycle(anchor_px);
        assert!(
            (anchor_cycle_after - anchor_cycle_before).abs() < 1e-9,
            "Anchor cycle shifted: {} -> {}",
            anchor_cycle_before,
            anchor_cycle_after,
        );

        // px_per_cycle should have doubled.
        assert!((vp.px_per_cycle - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn zoom_clamps_to_limits() {
        let mut vp = Viewport::default();

        // Zoom way in -- should clamp to MAX_PX_PER_CYCLE.
        vp.zoom_at(1e10, 0.0);
        assert!((vp.px_per_cycle - MAX_PX_PER_CYCLE).abs() < f64::EPSILON);

        // Zoom way out -- should clamp to MIN_PX_PER_CYCLE.
        vp.zoom_at(1e-20, 0.0);
        assert!((vp.px_per_cycle - MIN_PX_PER_CYCLE).abs() < f64::EPSILON);
    }

    #[test]
    fn pan_moves_start_cycle() {
        let mut vp = Viewport {
            start_cycle: 100.0,
            px_per_cycle: 2.0,
            width_px: 800.0,
        };

        // Pan right by 100 pixels -> start_cycle decreases by 100/2 = 50.
        vp.pan_px(100.0);
        assert!((vp.start_cycle - 50.0).abs() < f64::EPSILON);

        // Pan left by 200 pixels -> start_cycle increases by 200/2 = 100.
        vp.pan_px(-200.0);
        assert!((vp.start_cycle - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn fit_range_normal() {
        let mut vp = Viewport {
            start_cycle: 0.0,
            px_per_cycle: 1.0,
            width_px: 1000.0,
        };

        vp.fit_range(200.0, 400.0);

        // With 5% margin: range becomes 190..410, span=220.
        let expected_start = 200.0 - (200.0 * 0.05);
        assert!(
            (vp.start_cycle - expected_start).abs() < 1e-9,
            "start_cycle {} != expected {}",
            vp.start_cycle,
            expected_start,
        );

        // px_per_cycle = 1000 / 220 ~= 4.545...
        let expected_ppc = 1000.0 / 220.0;
        assert!(
            (vp.px_per_cycle - expected_ppc).abs() < 1e-9,
            "px_per_cycle {} != expected {}",
            vp.px_per_cycle,
            expected_ppc,
        );
    }

    #[test]
    fn fit_range_degenerate() {
        let mut vp = Viewport {
            start_cycle: 999.0,
            px_per_cycle: 50.0,
            width_px: 500.0,
        };

        // Same min and max -> should default to 100-cycle span.
        vp.fit_range(1000.0, 1000.0);

        assert!((vp.start_cycle - 950.0).abs() < f64::EPSILON);
        // px_per_cycle = 500 / 100 = 5.0
        assert!((vp.px_per_cycle - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn multiple_roundtrips_at_different_zooms() {
        let mut vp = Viewport::default();
        for &ppc in &[0.001, 0.1, 1.0, 10.0, 50.0] {
            vp.px_per_cycle = ppc;
            vp.start_cycle = 12345.0;
            let cycle = 54321.0;
            let px = vp.cycle_to_px(cycle);
            let back = vp.px_to_cycle(px);
            // At low px_per_cycle the intermediate f32 representation of
            // the pixel offset loses precision (f32 has ~7 significant
            // digits). Allow a tolerance proportional to 1/px_per_cycle.
            let tolerance = 1.0 / ppc * 1e-3;
            assert!(
                (back - cycle).abs() < tolerance,
                "roundtrip failed at ppc={}: {} != {} (tolerance {})",
                ppc,
                back,
                cycle,
                tolerance,
            );
        }
    }
}
