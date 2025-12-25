//! Tile array grid visualization.
//!
//! Shows the NPU tile array as a colored grid where each cell represents
//! a tile and its status.

use eframe::egui::{self, Color32, Rect, Sense, Stroke, StrokeKind, Vec2};

use crate::device::tile::TileType;
use crate::interpreter::CoreStatus;

use super::app::{EmulatorApp, SelectedTile};

/// Tile cell size in pixels.
const TILE_SIZE: f32 = 40.0;
/// Spacing between tiles.
const TILE_SPACING: f32 = 4.0;

/// Get the color for a tile based on its type and status.
fn tile_color(app: &EmulatorApp, col: u8, row: u8) -> Color32 {
    let tile = match app.engine.device().array.get(col, row) {
        Some(t) => t,
        None => return Color32::DARK_GRAY,
    };

    match tile.tile_type {
        TileType::Shim => Color32::from_rgb(60, 60, 80), // Dark blue-gray
        TileType::MemTile => Color32::from_rgb(70, 100, 70), // Dark green
        TileType::Compute => {
            // Check core status
            if let Some(status) = app.engine.core_status(col as usize, row as usize) {
                match status {
                    CoreStatus::Running => Color32::from_rgb(50, 200, 50),      // Bright green
                    CoreStatus::WaitingLock { .. } => Color32::from_rgb(200, 200, 50), // Yellow
                    CoreStatus::WaitingDma { .. } => Color32::from_rgb(200, 150, 50),  // Orange
                    CoreStatus::Halted => Color32::from_rgb(150, 50, 50),       // Dark red
                    CoreStatus::Error => Color32::from_rgb(255, 0, 0),          // Bright red
                    CoreStatus::Ready => {
                        if tile.core.enabled {
                            Color32::from_rgb(80, 120, 80) // Ready (light green)
                        } else {
                            Color32::from_rgb(60, 60, 60) // Disabled (gray)
                        }
                    }
                }
            } else if tile.core.enabled {
                Color32::from_rgb(80, 120, 80)
            } else {
                Color32::from_rgb(60, 60, 60)
            }
        }
    }
}

/// Show the tile grid.
pub fn show_tile_grid(ui: &mut egui::Ui, app: &mut EmulatorApp) {
    let cols = app.engine.device().array.cols();
    let rows = app.engine.device().array.rows();

    // Grid dimensions
    let grid_width = cols as f32 * (TILE_SIZE + TILE_SPACING);
    let grid_height = rows as f32 * (TILE_SIZE + TILE_SPACING);

    // Legend
    ui.horizontal_wrapped(|ui| {
        show_legend_item(ui, "Shim", Color32::from_rgb(60, 60, 80));
        show_legend_item(ui, "Mem", Color32::from_rgb(70, 100, 70));
        show_legend_item(ui, "Run", Color32::from_rgb(50, 200, 50));
        show_legend_item(ui, "Wait", Color32::from_rgb(200, 200, 50));
        show_legend_item(ui, "Halt", Color32::from_rgb(150, 50, 50));
    });
    ui.separator();

    // Allocate space for grid
    let (response, painter) = ui.allocate_painter(Vec2::new(grid_width, grid_height), Sense::click());
    let origin = response.rect.min;

    // Draw tiles (bottom-up: row 0 at bottom, row 5 at top)
    for col in 0..cols {
        for row in 0..rows {
            // Invert Y so row 0 is at bottom
            let visual_row = rows - 1 - row;

            let x = origin.x + col as f32 * (TILE_SIZE + TILE_SPACING);
            let y = origin.y + visual_row as f32 * (TILE_SIZE + TILE_SPACING);

            let rect = Rect::from_min_size(
                egui::pos2(x, y),
                Vec2::new(TILE_SIZE, TILE_SIZE),
            );

            let color = tile_color(app, col, row);
            let is_selected = app.selected_tile == Some(SelectedTile { col, row });

            // Draw tile background
            painter.rect_filled(rect, 4.0, color);

            // Draw selection border
            if is_selected {
                painter.rect_stroke(rect, 4.0, Stroke::new(3.0, Color32::WHITE), StrokeKind::Outside);
            }

            // Draw tile label
            let label = match app.engine.device().array.get(col, row) {
                Some(tile) => {
                    let type_char = match tile.tile_type {
                        TileType::Shim => "S",
                        TileType::MemTile => "M",
                        TileType::Compute => "C",
                    };
                    format!("{}{},{}", type_char, col, row)
                }
                None => format!("{},{}", col, row),
            };

            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                label,
                egui::FontId::proportional(10.0),
                Color32::WHITE,
            );

            // Handle click
            if response.clicked() {
                if let Some(pos) = response.interact_pointer_pos() {
                    if rect.contains(pos) {
                        app.selected_tile = Some(SelectedTile { col, row });
                    }
                }
            }
        }
    }

    ui.add_space(TILE_SPACING);

    // Tile info summary
    ui.separator();
    ui.heading("Summary");

    let (shim, mem, compute) = app.engine.device().array.count_by_type();
    ui.label(format!("Tiles: {} shim, {} mem, {} compute", shim, mem, compute));
    ui.label(format!("Enabled cores: {}", app.engine.enabled_cores()));
    ui.label(format!("Active cores: {}", app.engine.active_cores()));

    // Selected tile quick info
    if let Some(sel) = app.selected_tile {
        ui.separator();
        ui.label(format!("Selected: ({}, {})", sel.col, sel.row));

        if let Some(tile) = app.engine.device().array.get(sel.col, sel.row) {
            let type_name = match tile.tile_type {
                TileType::Shim => "Shim Tile",
                TileType::MemTile => "Memory Tile (512KB)",
                TileType::Compute => "Compute Tile",
            };
            ui.label(type_name);

            if tile.is_compute() {
                ui.label(format!("PC: 0x{:04X}", tile.core.pc));
                if let Some(status) = app.engine.core_status(sel.col as usize, sel.row as usize) {
                    ui.label(format!("Status: {:?}", status));
                }
            }
        }
    }
}

/// Show a legend item.
fn show_legend_item(ui: &mut egui::Ui, label: &str, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(12.0, 12.0), Sense::hover());
    ui.painter().rect_filled(rect, 2.0, color);
    ui.label(label);
}
