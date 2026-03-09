//! Color and layout constants for the trace comparison visualizer.
//!
//! All visual styling is centralized here so the rendering code stays clean
//! and palette changes only require editing one file.

use eframe::egui::Color32;

// ============================================================================
// Lane backgrounds
// ============================================================================

/// Background tint for HW (hardware) trace lanes -- subtle blue.
pub const HW_LANE_BG: Color32 = Color32::from_rgb(30, 40, 60);

/// Background tint for EMU (emulator) trace lanes -- subtle green.
pub const EMU_LANE_BG: Color32 = Color32::from_rgb(30, 55, 40);

// ============================================================================
// Event bars and edges
// ============================================================================

/// Edge tick mark color (instantaneous events).
pub const EDGE_TICK: Color32 = Color32::from_rgb(200, 200, 200);

/// HW level bar fill (sustained/interval events).
pub const HW_LEVEL_BAR: Color32 = Color32::from_rgb(70, 130, 200);

/// EMU level bar fill (sustained/interval events).
pub const EMU_LEVEL_BAR: Color32 = Color32::from_rgb(70, 180, 100);

// ============================================================================
// Divergence indicators
// ============================================================================

/// Line connecting divergent event pairs.
pub const DIVERGE_LINE: Color32 = Color32::from_rgb(220, 80, 80);

/// Line connecting matching (non-divergent) event pairs.
pub const MATCH_LINE: Color32 = Color32::from_rgb(80, 80, 80);

/// Highlight around a selected event or tile.
pub const SELECTED_HIGHLIGHT: Color32 = Color32::from_rgb(255, 200, 60);

/// Semi-transparent wash over divergent regions.
pub const DIVERGE_WASH: Color32 = Color32::from_rgba_premultiplied(220, 80, 80, 40);

/// Small marker dot/diamond at a divergence point.
pub const DIVERGE_MARKER: Color32 = Color32::from_rgb(255, 100, 100);

// ============================================================================
// Minimap
// ============================================================================

/// Minimap viewport rectangle border color.
pub const MINIMAP_VIEWPORT: Color32 = Color32::from_rgb(200, 200, 200);

/// Minimap background fill.
pub const MINIMAP_BG: Color32 = Color32::from_rgb(20, 20, 25);

// ============================================================================
// Layout constants
// ============================================================================

/// Height of one event lane in pixels.
pub const LANE_HEIGHT: f32 = 20.0;

/// Vertical gap between event blocks (tile groups).
pub const BLOCK_GAP: f32 = 8.0;

/// Height of the minimap strip at the top.
pub const MINIMAP_HEIGHT: f32 = 24.0;

/// Minimum pixels per cycle tick -- controls zoom floor.
pub const MIN_PX_PER_TICK: f32 = 2.0;

/// Cycle delta threshold: deltas above this magnitude count as divergence.
/// Matches the constant in `trace::compare`.
pub const DIVERGE_THRESHOLD: i64 = 10;
