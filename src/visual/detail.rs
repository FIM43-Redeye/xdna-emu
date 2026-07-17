//! Selected-tile detail rows (Task 7).

use eframe::egui;

use crate::debugger::engine_host::EngineHost;
use crate::debugger::model::{tile_ports, tile_snapshot, tile_state};
use crate::visual::floorplan::{floorplan, lock_map, FloorplanPresentation};
use crate::visual::theme::Palette;

fn port_color(active: bool, stalled: bool, palette: &Palette) -> egui::Color32 {
    if stalled {
        palette.route_stalled
    } else if active {
        palette.route_moving
    } else {
        palette.text
    }
}

pub fn show(
    ui: &mut egui::Ui,
    host: &EngineHost,
    selected: Option<(u8, u8)>,
    palette: &Palette,
    mem_texture: Option<egui::TextureId>,
) {
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
    let diagram_size = egui::vec2(ui.available_width().min(360.0).max(180.0), 280.0);
    let (rect, _) = ui.allocate_exact_size(diagram_size, egui::Sense::hover());
    floorplan(ui, rect, &snap, &state, &ports, &FloorplanPresentation { palette, mem_texture });

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
    fn port_color_prioritizes_stalls_then_activity() {
        let palette = Palette::dark();

        assert_eq!(port_color(false, false, &palette), palette.text);
        assert_eq!(port_color(true, false, &palette), palette.route_moving);
        assert_eq!(port_color(true, true, &palette), palette.route_stalled);
    }
}
