//! Selected-tile detail rows (Task 7).

use eframe::egui;

use crate::debugger::engine_host::EngineHost;
use crate::debugger::model::tile_snapshot;
use crate::visual::theme;

pub fn show(ui: &mut egui::Ui, host: &EngineHost, selected: Option<(u8, u8)>) {
    let Some((col, row)) = selected else {
        ui.label("Select a tile to inspect it.");
        return;
    };
    let Some(snap) = tile_snapshot(&host.engine, col, row) else {
        ui.label(format!("No tile at ({col},{row})"));
        return;
    };

    ui.heading(format!("Tile ({},{})  [{:?}]", snap.col, snap.row, snap.kind));

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
    egui::CollapsingHeader::new("locks (64)").show(ui, |ui| {
        // 8 per row for compactness.
        for chunk in snap.locks.chunks(8) {
            ui.monospace(chunk.iter().map(|v| format!("{v:>3}")).collect::<Vec<_>>().join(" "));
        }
    });

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
            let color = if p.stalled {
                theme::PORT_STALLED
            } else if p.active {
                theme::PORT_ACTIVE
            } else {
                theme::TILE_LABEL
            };
            ui.colored_label(color, &p.label);
        }
    });
}
