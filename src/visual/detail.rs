//! Selected-tile detail rows (Task 7).

use eframe::egui;

use crate::debugger::engine_host::EngineHost;
use crate::debugger::model::{tile_ports, tile_snapshot, tile_state};
use crate::visual::theme::Palette;
use crate::visual::tile::{tile, DetailTier, TilePresentation};

/// Lock-bank colour by semaphore value. Redundant with the number drawn on the
/// cell (WCAG 1.4.1: never colour alone). Positive = tokens available, negative
/// = over-acquired, zero = idle. Overflow/underflow flags aren't in the snapshot
/// yet -- a loud fault colour drops in here once they are.
fn lock_color(value: i8, palette: &Palette) -> egui::Color32 {
    if value > 0 {
        palette.band_green
    } else if value < 0 {
        palette.band_amber
    } else {
        palette.band_pale
    }
}

/// Paint the tile's own lock bank as a compact colour+number grid.
///
/// Directional (West/Own/East) framing is deliberately not applied here: a tile's
/// own locks are all "Own". The cross-tile address space a mem tile *references*
/// spans three tiles' banks and is a separate, mem-tile-only view.
fn lock_map(ui: &mut egui::Ui, locks: &[i8], palette: &Palette) {
    const PER_ROW: usize = 8;
    const CELL: egui::Vec2 = egui::vec2(30.0, 20.0);
    const GAP: f32 = 3.0;

    if locks.is_empty() {
        ui.label("no locks");
        return;
    }

    let rows = locks.len().div_ceil(PER_ROW);
    let cols = locks.len().min(PER_ROW);
    let size = egui::vec2(cols as f32 * (CELL.x + GAP) - GAP, rows as f32 * (CELL.y + GAP) - GAP);
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter();

    for (i, &value) in locks.iter().enumerate() {
        let (col, row) = (i % PER_ROW, i / PER_ROW);
        let min = rect.min + egui::vec2(col as f32 * (CELL.x + GAP), row as f32 * (CELL.y + GAP));
        let cell = egui::Rect::from_min_size(min, CELL);
        painter.rect_filled(cell, 3.0, lock_color(value, palette));
        painter.text(
            cell.center(),
            egui::Align2::CENTER_CENTER,
            format!("{value}"),
            egui::FontId::monospace(11.0),
            palette.text,
        );
    }
}

fn port_color(active: bool, stalled: bool, palette: &Palette) -> egui::Color32 {
    if stalled {
        palette.route_stalled
    } else if active {
        palette.route_moving
    } else {
        palette.text
    }
}

pub fn show(ui: &mut egui::Ui, host: &EngineHost, selected: Option<(u8, u8)>, palette: &Palette) {
    let Some((col, row)) = selected else {
        ui.label("Select a tile to inspect it.");
        return;
    };
    let Some(snap) = tile_snapshot(&host.engine, col, row) else {
        ui.label(format!("No tile at ({col},{row})"));
        return;
    };

    ui.heading(format!("Tile ({},{})  [{:?}]", snap.col, snap.row, snap.kind));

    let state = tile_state(&host.engine, col, row);
    let ports = tile_ports(&host.engine.device().array, col, row);
    let diagram_size = egui::vec2(ui.available_width().min(360.0).max(180.0), 180.0);
    let (rect, _) = ui.allocate_exact_size(diagram_size, egui::Sense::hover());
    tile(
        ui,
        rect,
        &snap,
        &state,
        &ports,
        &TilePresentation { tier: DetailTier::Diagram, selected: true, palette },
    );

    ui.separator();
    ui.label(format!(
        "core: {}   pc: {}",
        snap.core_status.as_deref().unwrap_or("-"),
        snap.pc.map(|p| format!("0x{p:05x}")).unwrap_or_else(|| "-".into())
    ));

    ui.separator();
    ui.label("DMA channels:");
    for ch in &snap.dma {
        ui.monospace(format!(
            "  ch{}: {}  cur_bd={:?} queued_bd={:?} queue={}",
            ch.index, ch.state, ch.current_bd, ch.queued_bd, ch.queue_len
        ));
    }

    ui.separator();
    egui::CollapsingHeader::new(format!("locks ({})", snap.locks.len()))
        .default_open(true)
        .show(ui, |ui| lock_map(ui, &snap.locks, palette));

    ui.separator();
    egui::CollapsingHeader::new(format!("memory ({} bytes)", snap.mem_size)).show(ui, |ui| {
        for (i, w) in snap.mem_peek.iter().enumerate() {
            ui.monospace(format!("  [0x{:04x}] 0x{:08x}", i * 4, w));
        }
    });

    ui.separator();
    ui.label("stream ports:");
    ui.horizontal_wrapped(|ui| {
        for p in &snap.ports {
            ui.colored_label(port_color(p.active, p.stalled, palette), &p.label);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visual::theme::Palette;

    #[test]
    fn lock_color_encodes_value_sign() {
        let palette = Palette::dark();
        assert_eq!(lock_color(3, &palette), palette.band_green);
        assert_eq!(lock_color(0, &palette), palette.band_pale);
        assert_eq!(lock_color(-2, &palette), palette.band_amber);
    }

    #[test]
    fn port_color_prioritizes_stalls_then_activity() {
        let palette = Palette::dark();

        assert_eq!(port_color(false, false, &palette), palette.text);
        assert_eq!(port_color(true, false, &palette), palette.route_moving);
        assert_eq!(port_color(true, true, &palette), palette.route_stalled);
    }
}
