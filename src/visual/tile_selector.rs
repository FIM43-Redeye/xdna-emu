//! Tile selector sidebar for the trace comparison visualizer.
//!
//! Lists tiles sorted by divergence severity (most divergent first).
//! Each entry shows (col, row), tile type, and divergence count.
//! Clicking a tile selects it for timeline display.

use eframe::egui;

use crate::trace::compare::TileKey;

use super::data::{divergence_count, TraceSource};

// ============================================================================
// Tile type display name
// ============================================================================

/// Map pkt_type to a human-readable tile type label.
///
/// The pkt_type field in TileKey encodes the tile class:
/// - 0: Core tile (compute + local memory)
/// - 1: Memory module (adjacent to core)
/// - 3: MemTile (shared memory between columns)
/// - anything else: Shim tile (DDR interface)
fn tile_type_label(pkt_type: u8) -> &'static str {
    match pkt_type {
        0 => "Core",
        1 => "Mem",
        3 => "MTile",
        _ => "Shim",
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Render the tile selector sidebar.
///
/// Returns `Some(TileKey)` if the user clicked a new tile, `None` otherwise.
/// The sidebar is wrapped in a vertical scroll area so it handles any number
/// of tiles without overflowing the panel.
pub fn show_tile_selector(
    ui: &mut egui::Ui,
    source: &dyn TraceSource,
    selected: Option<&TileKey>,
) -> Option<TileKey> {
    let mut clicked = None;

    ui.heading("Tiles");
    ui.separator();

    egui::ScrollArea::vertical()
        .auto_shrink([false; 2])
        .show(ui, |ui| {
            let batch = source.batch_result();
            for key in source.tile_keys() {
                let div_count = divergence_count(batch, key);
                let is_selected = selected == Some(key);

                let label = format!(
                    "({},{}) {} [{}]",
                    key.col,
                    key.row,
                    tile_type_label(key.pkt_type),
                    div_count,
                );

                let response = ui.selectable_label(is_selected, &label);
                if response.clicked() && !is_selected {
                    clicked = Some(*key);
                }
            }
        });

    clicked
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_type_labels() {
        assert_eq!(tile_type_label(0), "Core");
        assert_eq!(tile_type_label(1), "Mem");
        assert_eq!(tile_type_label(3), "MTile");
        assert_eq!(tile_type_label(2), "Shim");
        assert_eq!(tile_type_label(255), "Shim");
    }
}
