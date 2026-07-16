//! Array grid rendering (Task 6): the NPU array as a spatial grid of
//! type-colored, clickable tiles.

use eframe::egui::{self, Color32, Rect, Sense, Stroke, StrokeKind, Vec2};

use crate::debugger::model::{tile_grid, TileKindDisplay};
use crate::device::array::TileArray;
use crate::visual::theme;

fn tile_color(kind: TileKindDisplay) -> Color32 {
    match kind {
        TileKindDisplay::Shim => theme::TILE_SHIM,
        TileKindDisplay::Mem => theme::TILE_MEM,
        TileKindDisplay::Core => theme::TILE_CORE,
    }
}

/// Draw the array as a grid of clickable tiles. Hardware row 0 (shim) is
/// flipped to the bottom so the layout reads like the physical array.
pub fn show(ui: &mut egui::Ui, array: &TileArray, selected: &mut Option<(u8, u8)>) {
    let cells = tile_grid(array);
    let rows = array.rows();
    let cell = Vec2::new(64.0, 44.0);
    let pad = 6.0;
    let origin = ui.min_rect().min + Vec2::new(pad, pad);

    // Compute each tile's rect and handle clicks first, then paint once --
    // keeps the `&mut Ui` borrow (allocate_rect) and the `&Painter` borrow
    // (below) from overlapping.
    let mut rects: Vec<(Rect, TileKindDisplay, (u8, u8))> = Vec::with_capacity(cells.len());
    for c in &cells {
        let vis_row = (rows - 1 - c.row) as f32;
        let pos = origin + Vec2::new(c.col as f32 * (cell.x + pad), vis_row * (cell.y + pad));
        let rect = Rect::from_min_size(pos, cell);
        let resp = ui.allocate_rect(rect, Sense::click());
        if resp.clicked() {
            *selected = Some((c.col, c.row));
        }
        rects.push((rect, c.kind, (c.col, c.row)));
    }

    let painter = ui.painter();
    for (rect, kind, pos) in rects {
        painter.rect_filled(rect, 4.0, tile_color(kind));
        if *selected == Some(pos) {
            painter.rect_stroke(rect, 4.0, Stroke::new(3.0, theme::TILE_SELECTED), StrokeKind::Outside);
        }
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            format!("{},{}", pos.0, pos.1),
            egui::FontId::monospace(12.0),
            theme::TILE_LABEL,
        );
    }

    // Reserve the space we painted into so the panel sizes correctly.
    let width = array.cols() as f32 * (cell.x + pad) + pad;
    let height = rows as f32 * (cell.y + pad) + pad;
    ui.allocate_space(Vec2::new(width, height));
}
