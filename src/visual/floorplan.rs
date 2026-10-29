//! Reusable tile floor-plan schematic for architecture views.

use eframe::egui::{
    self, pos2, vec2, Align2, FontId, Rect, Sense, Stroke, StrokeKind, UiBuilder, WidgetInfo, WidgetType,
};

use crate::debugger::model::{ChannelSnapshot, PortWire, TileKindDisplay, TileSnapshot, TileState};
use crate::device::dma::DmaStall;
use crate::visual::theme::Palette;
use crate::visual::tile::paint_port;

pub struct FloorplanPresentation<'a> {
    pub palette: &'a Palette,
    pub mem_texture: Option<egui::TextureId>,
    pub selected_dma: Option<u8>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FloorplanResponse {
    pub dma_channel_clicked: Option<u8>,
}

#[derive(Clone, Copy, Debug)]
struct FloorplanBlocks {
    core: Option<Rect>,
    memory_module: Option<Rect>,
    banks: Option<Rect>,
    ddr_noc: Option<Rect>,
    dma: Rect,
    locks: Rect,
}

fn module_blocks(module: Rect, gap: f32) -> (Rect, Rect, Rect) {
    let header = (module.height() * 0.13).clamp(14.0, 18.0);
    let content = Rect::from_min_max(
        pos2(module.left() + gap, module.top() + header),
        pos2(module.right() - gap, module.bottom() - gap),
    );
    let banks_height = (content.height() - gap) * 0.45;
    let banks = Rect::from_min_max(content.min, pos2(content.right(), content.top() + banks_height));
    let lower_top = banks.bottom() + gap;
    let half_width = (content.width() - gap) * 0.5;
    let dma = Rect::from_min_max(
        pos2(content.left(), lower_top),
        pos2(content.left() + half_width, content.bottom()),
    );
    let locks = Rect::from_min_max(pos2(dma.right() + gap, lower_top), content.max);
    (banks, dma, locks)
}

/// Pure schematic geometry. The Memory Module intentionally contains its
/// banks, DMA, and lock blocks; leaf blocks never overlap one another.
fn block_rects(rect: Rect, kind: TileKindDisplay) -> FloorplanBlocks {
    let margin = (rect.width().min(rect.height()) * 0.06).clamp(6.0, 10.0);
    let interior = rect.shrink(margin);
    let gap = (interior.width().min(interior.height()) * 0.04).clamp(4.0, 6.0);

    match kind {
        TileKindDisplay::Core => {
            let core_width = interior.width() * 0.31;
            let core =
                Rect::from_min_max(interior.min, pos2(interior.left() + core_width, interior.bottom()));
            let module = Rect::from_min_max(pos2(core.right() + gap, interior.top()), interior.max);
            let (banks, dma, locks) = module_blocks(module, gap);
            FloorplanBlocks {
                core: Some(core),
                memory_module: Some(module),
                banks: Some(banks),
                ddr_noc: None,
                dma,
                locks,
            }
        }
        TileKindDisplay::Mem => {
            let (banks, dma, locks) = module_blocks(interior, gap);
            FloorplanBlocks {
                core: None,
                memory_module: Some(interior),
                banks: Some(banks),
                ddr_noc: None,
                dma,
                locks,
            }
        }
        TileKindDisplay::Shim => {
            let upper_height = (interior.height() - gap) * 0.48;
            let ddr_noc =
                Rect::from_min_max(interior.min, pos2(interior.right(), interior.top() + upper_height));
            let lower_top = ddr_noc.bottom() + gap;
            let half_width = (interior.width() - gap) * 0.5;
            let dma = Rect::from_min_max(
                pos2(interior.left(), lower_top),
                pos2(interior.left() + half_width, interior.bottom()),
            );
            let locks = Rect::from_min_max(pos2(dma.right() + gap, lower_top), interior.max);
            FloorplanBlocks {
                core: None,
                memory_module: None,
                banks: None,
                ddr_noc: Some(ddr_noc),
                dma,
                locks,
            }
        }
    }
}

/// Lock-bank colour by semaphore value. The number in each cell keeps the
/// state readable without relying on colour alone.
pub(crate) fn lock_color(value: i8, palette: &Palette) -> egui::Color32 {
    if value > 0 {
        palette.band_green
    } else if value < 0 {
        palette.band_amber
    } else {
        palette.band_pale
    }
}

/// Paint the tile's own lock bank as a compact, responsive colour+number grid.
/// Directional West/Own/East framing does not apply: these are the tile's own
/// locks, while a mem tile's cross-tile address space is a separate view.
pub(crate) fn lock_map(ui: &mut egui::Ui, locks: &[i8], palette: &Palette) {
    const PER_ROW: usize = 8;
    const PREFERRED_CELL: egui::Vec2 = vec2(30.0, 20.0);

    if locks.is_empty() {
        ui.label("no locks");
        return;
    }

    let rows = locks.len().div_ceil(PER_ROW);
    let cols = locks.len().min(PER_ROW);
    let available = ui.available_size();
    let gap = if available.x < 160.0 { 1.0 } else { 3.0 };
    let cell = vec2(
        ((available.x - gap * (cols - 1) as f32) / cols as f32).clamp(1.0, PREFERRED_CELL.x),
        ((available.y - gap * (rows - 1) as f32) / rows as f32).clamp(1.0, PREFERRED_CELL.y),
    );
    let size =
        vec2(cols as f32 * cell.x + (cols - 1) as f32 * gap, rows as f32 * cell.y + (rows - 1) as f32 * gap);
    let (_, rect) = ui.allocate_space(size);
    let font = FontId::monospace((cell.y * 0.55).clamp(5.0, 11.0));

    for (i, &value) in locks.iter().enumerate() {
        let (col, row) = (i % PER_ROW, i / PER_ROW);
        let min = rect.min + vec2(col as f32 * (cell.x + gap), row as f32 * (cell.y + gap));
        let cell = Rect::from_min_size(min, cell);
        ui.painter().rect_filled(cell, 3.0, lock_color(value, palette));
        ui.painter()
            .text(cell.center(), Align2::CENTER_CENTER, value, font.clone(), palette.text);
    }
}

fn block_frame(ui: &egui::Ui, rect: Rect, palette: &Palette) {
    ui.painter().rect_filled(rect, 3.0, palette.bg);
    ui.painter()
        .rect_stroke(rect, 3.0, Stroke::new(1.0_f32, palette.text), StrokeKind::Inside);
}

fn block_label(ui: &egui::Ui, rect: Rect, label: impl ToString, palette: &Palette) {
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        label,
        FontId::monospace((rect.height() * 0.18).clamp(8.0, 12.0)),
        palette.text,
    );
}

fn dma_color(channel: &ChannelSnapshot, palette: &Palette) -> egui::Color32 {
    if channel.phase == "Error" {
        return palette.band_red;
    }
    match channel.stall {
        Some(DmaStall::LockWait) => palette.band_amber,
        Some(DmaStall::Backpressure) => palette.route_stalled,
        Some(DmaStall::Starved) => palette.dma_starved,
        Some(DmaStall::Other) => palette.band_pale,
        None => match channel.phase {
            "Transferring" => palette.band_green,
            "ReleasingLock" => palette.band_blue,
            "Idle" => palette.band_grey,
            "Paused" => palette.band_amber,
            _ => palette.band_pale,
        },
    }
}

fn paint_dma(
    ui: &egui::Ui,
    rect: Rect,
    channels: &[ChannelSnapshot],
    palette: &Palette,
    id: egui::Id,
    selected_dma: Option<u8>,
) -> Option<u8> {
    block_label(ui, Rect::from_min_max(rect.min, pos2(rect.right(), rect.top() + 13.0)), "DMA", palette);
    if channels.is_empty() {
        return None;
    }

    let content = Rect::from_min_max(pos2(rect.left() + 3.0, rect.top() + 14.0), rect.max - vec2(3.0, 3.0));
    let gap = 1.0;
    let bar_height = (content.height() - gap * (channels.len() - 1) as f32) / channels.len() as f32;
    let mut clicked = None;

    for (i, channel) in channels.iter().enumerate() {
        let min = content.min + vec2(0.0, i as f32 * (bar_height + gap));
        let bar = Rect::from_min_size(min, vec2(content.width(), bar_height));
        let response = ui.interact(bar, id.with(channel.index), Sense::click());
        response.widget_info(|| {
            WidgetInfo::labeled(
                WidgetType::Button,
                true,
                format!(
                    "DMA channel {}, phase {}, stall {:?}, progress {:.0} percent, current BD {:?}, {} queued",
                    channel.index,
                    channel.phase,
                    channel.stall,
                    channel.progress * 100.0,
                    channel.current_bd,
                    channel.queue_len
                ),
            )
        });
        if response.clicked() {
            clicked = Some(channel.index);
        }

        let color = dma_color(channel, palette);
        ui.painter().rect_filled(bar, 1.0, color.gamma_multiply(0.3));
        let progress = channel.progress.clamp(0.0, 1.0);
        if progress > 0.0 {
            let fill = Rect::from_min_max(bar.min, pos2(bar.left() + bar.width() * progress, bar.bottom()));
            ui.painter().rect_filled(fill, 1.0, color);
        }

        let font = FontId::monospace((bar.height() * 0.5).clamp(5.0, 9.0));
        ui.painter().text(
            pos2(bar.left() + 2.0, bar.center().y),
            Align2::LEFT_CENTER,
            format!("ch{}", channel.index),
            font.clone(),
            palette.text,
        );

        let slot_width = (bar.height() * 1.2).clamp(10.0, 16.0);
        let max_slots = ((bar.width() - 30.0) / (slot_width + 1.0)).floor().max(0.0) as usize;
        let current_slots = usize::from(channel.current_bd.is_some());
        if bar.height() < 9.0 || max_slots <= current_slots {
            ui.painter().text(
                pos2(bar.right() - 2.0, bar.center().y),
                Align2::RIGHT_CENTER,
                format!("q{}", channel.queue_len),
                font.clone(),
                palette.text,
            );
        } else {
            let mut shown = channel.queue_bds.len().min(max_slots - current_slots);
            if channel.queue_len > shown && shown > 0 {
                shown -= 1;
            }
            let hidden = channel.queue_len.saturating_sub(shown);
            let mut right = bar.right() - 1.0;
            if hidden > 0 {
                let overflow =
                    Rect::from_min_size(pos2(right - slot_width, bar.top()), vec2(slot_width, bar.height()));
                ui.painter().text(
                    overflow.center(),
                    Align2::CENTER_CENTER,
                    format!("+{hidden}"),
                    font.clone(),
                    palette.text,
                );
                right = overflow.left() - 1.0;
            }
            for &bd in channel.queue_bds.iter().take(shown).rev() {
                let cell = Rect::from_min_max(
                    pos2(right - slot_width, bar.top() + 1.0),
                    pos2(right, bar.bottom() - 1.0),
                );
                ui.painter().rect_filled(cell, 1.0, palette.band_pale);
                ui.painter()
                    .text(cell.center(), Align2::CENTER_CENTER, bd, font.clone(), palette.text);
                right = cell.left() - 1.0;
            }
            if let Some(bd) = channel.current_bd {
                let cell = Rect::from_min_max(
                    pos2(right - slot_width, bar.top() + 1.0),
                    pos2(right, bar.bottom() - 1.0),
                );
                ui.painter().rect_filled(cell, 1.0, palette.selected);
                ui.painter()
                    .text(cell.center(), Align2::CENTER_CENTER, bd, font.clone(), palette.bg);
            }
        }

        if selected_dma == Some(channel.index) {
            ui.painter()
                .rect_stroke(bar, 1.0, Stroke::new(1.5_f32, palette.selected), StrokeKind::Inside);
        }
    }

    clicked
}

/// Paint a kind-aware tile floor plan with live state layered over its static
/// component grouping.
pub fn floorplan(
    ui: &mut egui::Ui,
    rect: Rect,
    snap: &TileSnapshot,
    state: &TileState,
    ports: &[PortWire],
    pres: &FloorplanPresentation<'_>,
) -> FloorplanResponse {
    let id = ui.make_persistent_id(("architecture-floorplan", snap.col, snap.row));
    let response = ui.interact(rect, id, Sense::click());
    let label = format!("{:?} tile floor plan, state {}", snap.kind, state.code());
    response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, true, &label));

    let blocks = block_rects(rect, snap.kind);

    if let Some(module) = blocks.memory_module {
        block_frame(ui, module, pres.palette);
        ui.painter().text(
            pos2(module.left() + 5.0, module.top() + 4.0),
            Align2::LEFT_TOP,
            "MEMORY MODULE",
            FontId::monospace(9.0),
            pres.palette.text,
        );
    }
    for block in [blocks.core, blocks.banks, blocks.ddr_noc].into_iter().flatten() {
        block_frame(ui, block, pres.palette);
    }
    block_frame(ui, blocks.dma, pres.palette);
    block_frame(ui, blocks.locks, pres.palette);

    if let Some(core) = blocks.core {
        ui.painter().rect_filled(core.shrink(1.0), 2.0, pres.palette.band(state));
        block_label(ui, core, format!("CORE\n{}", state.code()), pres.palette);
    }
    if let Some(banks) = blocks.banks {
        if let Some(texture) = pres.mem_texture {
            ui.painter().rect_filled(banks, 3.0, pres.palette.bg);
            ui.painter().image(
                texture,
                banks,
                Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        } else {
            ui.painter().rect_filled(banks.shrink(1.0), 2.0, pres.palette.kind_mem);
            block_label(ui, banks, format!("MEM {}K", snap.mem_size / 1024), pres.palette);
        }
    }
    if let Some(ddr_noc) = blocks.ddr_noc {
        ui.painter().rect_filled(ddr_noc.shrink(1.0), 2.0, pres.palette.kind_shim);
        block_label(ui, ddr_noc, "DDR/NoC", pres.palette);
    }
    let dma_channel_clicked =
        paint_dma(ui, blocks.dma, &snap.dma, pres.palette, id.with("dma"), pres.selected_dma);
    ui.painter().text(
        pos2(blocks.locks.center().x, blocks.locks.top() + 7.0),
        Align2::CENTER_CENTER,
        "LOCKS",
        FontId::monospace(9.0),
        pres.palette.text,
    );
    let lock_rect = Rect::from_min_max(
        pos2(blocks.locks.left() + 3.0, blocks.locks.top() + 14.0),
        blocks.locks.max - vec2(3.0, 3.0),
    );
    let mut lock_ui = ui.new_child(
        UiBuilder::new()
            .id_salt(("floorplan-locks", snap.col, snap.row))
            .max_rect(lock_rect),
    );
    lock_map(&mut lock_ui, &snap.locks, pres.palette);

    for port in ports.iter().filter(|port| port.port_type.is_external()) {
        paint_port(ui, rect, port, pres.palette);
    }

    FloorplanResponse { dma_channel_clicked }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::{pos2, vec2, Event, Modifiers, PointerButton, RawInput, Rect};

    use crate::debugger::model::{ChannelSnapshot, TileKindDisplay, TileSnapshot, TileState};
    use crate::device::dma::DmaStall;
    use crate::visual::theme::Palette;

    fn snapshot(kind: TileKindDisplay) -> TileSnapshot {
        TileSnapshot {
            col: 2,
            row: 3,
            kind,
            core_status: None,
            pc: None,
            dma: vec![ChannelSnapshot {
                index: 0,
                state: "Idle".into(),
                current_bd: None,
                queued_bd: None,
                queue_len: 0,
                progress: 0.0,
                phase: "Idle",
                stall: None,
                queue_bds: Vec::new(),
            }],
            locks: vec![0; if kind == TileKindDisplay::Mem { 64 } else { 16 }],
            mem_size: if kind == TileKindDisplay::Mem {
                512 * 1024
            } else {
                64 * 1024
            },
            mem_peek: Vec::new(),
            ports: Vec::new(),
        }
    }

    fn channel(index: u8, phase: &'static str, stall: Option<DmaStall>) -> ChannelSnapshot {
        ChannelSnapshot {
            index,
            state: phase.into(),
            current_bd: Some(1),
            queued_bd: None,
            queue_len: 2,
            progress: 0.5,
            phase,
            stall,
            queue_bds: vec![3, 5],
        }
    }

    fn inside(outer: Rect, inner: Rect) -> bool {
        outer.contains(inner.min) && outer.contains(inner.max)
    }

    fn overlaps(a: Rect, b: Rect) -> bool {
        a.left() < b.right() && b.left() < a.right() && a.top() < b.bottom() && b.top() < a.bottom()
    }

    #[test]
    fn block_rects_fit_without_overlapping_for_every_tile_kind() {
        let rect = Rect::from_min_max(pos2(20.0, 30.0), pos2(380.0, 210.0));

        for kind in [TileKindDisplay::Core, TileKindDisplay::Mem, TileKindDisplay::Shim] {
            let blocks = block_rects(rect, kind);
            let mut all = vec![blocks.dma, blocks.locks];
            all.extend(blocks.core);
            all.extend(blocks.memory_module);
            all.extend(blocks.banks);
            all.extend(blocks.ddr_noc);
            assert!(all.into_iter().all(|block| inside(rect, block)), "{kind:?} block outside tile");

            let mut leaves = vec![blocks.dma, blocks.locks];
            leaves.extend(blocks.core);
            leaves.extend(blocks.banks);
            leaves.extend(blocks.ddr_noc);
            for (i, &a) in leaves.iter().enumerate() {
                for &b in &leaves[i + 1..] {
                    assert!(!overlaps(a, b), "{kind:?} leaf blocks overlap: {a:?} and {b:?}");
                }
            }

            if let Some(module) = blocks.memory_module {
                assert!(blocks.banks.is_some_and(|banks| inside(module, banks)));
                assert!(inside(module, blocks.dma));
                assert!(inside(module, blocks.locks));
            }
        }
    }

    #[test]
    fn block_rects_expose_kind_specific_blocks_and_locks_for_all_kinds() {
        let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(360.0, 180.0));
        let core = block_rects(rect, TileKindDisplay::Core);
        let mem = block_rects(rect, TileKindDisplay::Mem);
        let shim = block_rects(rect, TileKindDisplay::Shim);

        assert!(core.core.is_some());
        assert!(mem.core.is_none());
        assert!(shim.core.is_none());
        assert!(core.banks.is_some() && mem.banks.is_some());
        assert!(shim.banks.is_none() && shim.ddr_noc.is_some());
        assert!(core.locks.is_positive() && mem.locks.is_positive() && shim.locks.is_positive());
    }

    #[test]
    fn floorplan_renders_every_tile_kind() {
        for kind in [TileKindDisplay::Core, TileKindDisplay::Mem, TileKindDisplay::Shim] {
            eframe::egui::__run_test_ui(|ui| {
                let rect = Rect::from_min_size(ui.min_rect().min, vec2(360.0, 180.0));
                let palette = Palette::dark();
                floorplan(
                    ui,
                    rect,
                    &snapshot(kind),
                    &TileState::Running,
                    &[],
                    &FloorplanPresentation { palette: &palette, mem_texture: None, selected_dma: None },
                );
            });
        }
    }

    #[test]
    fn floorplan_renders_memory_texture() {
        eframe::egui::__run_test_ui(|ui| {
            let rect = Rect::from_min_size(ui.min_rect().min, vec2(360.0, 280.0));
            let palette = Palette::dark();
            floorplan(
                ui,
                rect,
                &snapshot(TileKindDisplay::Core),
                &TileState::Running,
                &[],
                &FloorplanPresentation {
                    palette: &palette,
                    mem_texture: Some(egui::TextureId::Managed(1)),
                    selected_dma: None,
                },
            );
        });
    }

    #[test]
    fn dma_color_maps_progress_phases_and_stall_reasons() {
        let palette = Palette::dark();

        assert_eq!(dma_color(&channel(0, "Transferring", None), &palette), palette.band_green);
        assert_eq!(
            dma_color(&channel(0, "Transferring", Some(DmaStall::LockWait)), &palette),
            palette.band_amber
        );
        assert_eq!(
            dma_color(&channel(0, "Transferring", Some(DmaStall::Backpressure)), &palette),
            palette.route_stalled
        );
        assert_eq!(
            dma_color(&channel(0, "Transferring", Some(DmaStall::Starved)), &palette),
            palette.dma_starved
        );
        assert_eq!(dma_color(&channel(0, "ReleasingLock", None), &palette), palette.band_blue);
        assert_eq!(dma_color(&channel(0, "Idle", None), &palette), palette.band_grey);
        assert_eq!(
            dma_color(&channel(0, "MemoryLatency", Some(DmaStall::Other)), &palette),
            palette.band_pale
        );
        assert_eq!(dma_color(&channel(0, "Error", None), &palette), palette.band_red);
    }

    #[test]
    fn dma_bars_render_every_fsm_phase_and_stall_without_panic() {
        let cases = [
            ("Idle", None),
            ("BdSetup", Some(DmaStall::Other)),
            ("AcquiringLock", Some(DmaStall::LockWait)),
            ("MemoryLatency", Some(DmaStall::Other)),
            ("HostPipelineLatency", Some(DmaStall::Other)),
            ("BdSwitchBubble", Some(DmaStall::Other)),
            ("Transferring", None),
            ("Transferring", Some(DmaStall::Backpressure)),
            ("Transferring", Some(DmaStall::Starved)),
            ("StartupHold", Some(DmaStall::Other)),
            ("DrainingEgress", Some(DmaStall::Backpressure)),
            ("ReleasingLock", None),
            ("BdChaining", Some(DmaStall::Other)),
            ("Paused", None),
            ("Error", None),
        ];

        eframe::egui::__run_test_ui(|ui| {
            let mut snap = snapshot(TileKindDisplay::Core);
            snap.dma = cases
                .iter()
                .enumerate()
                .map(|(index, &(phase, stall))| channel(index as u8, phase, stall))
                .collect();
            let palette = Palette::dark();
            let rect = Rect::from_min_size(ui.min_rect().min, vec2(360.0, 320.0));
            floorplan(
                ui,
                rect,
                &snap,
                &TileState::Running,
                &[],
                &FloorplanPresentation { palette: &palette, mem_texture: None, selected_dma: Some(8) },
            );
        });
    }

    #[test]
    fn dma_bar_click_returns_the_channel_id() {
        let ctx = egui::Context::default();
        let palette = Palette::dark();
        let rect = Rect::from_min_size(pos2(20.0, 20.0), vec2(360.0, 280.0));
        let dma = block_rects(rect, TileKindDisplay::Core).dma;
        let click_pos =
            Rect::from_min_max(pos2(dma.left() + 3.0, dma.top() + 14.0), dma.max - vec2(3.0, 3.0)).center();
        let mut snap = snapshot(TileKindDisplay::Core);
        snap.dma = vec![channel(7, "Transferring", None)];
        let mut clicked = None;

        let mut render = |ctx: &egui::Context| {
            egui::CentralPanel::default().show(ctx, |ui| {
                clicked = floorplan(
                    ui,
                    rect,
                    &snap,
                    &TileState::Running,
                    &[],
                    &FloorplanPresentation { palette: &palette, mem_texture: None, selected_dma: None },
                )
                .dma_channel_clicked;
            });
        };
        let _ = ctx.run(
            RawInput {
                screen_rect: Some(Rect::from_min_size(pos2(0.0, 0.0), vec2(500.0, 400.0))),
                ..Default::default()
            },
            &mut render,
        );
        let events = vec![
            Event::PointerMoved(click_pos),
            Event::PointerButton {
                pos: click_pos,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: Modifiers::default(),
            },
            Event::PointerButton {
                pos: click_pos,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: Modifiers::default(),
            },
        ];
        let _ = ctx.run(
            RawInput {
                screen_rect: Some(Rect::from_min_size(pos2(0.0, 0.0), vec2(500.0, 400.0))),
                events,
                ..Default::default()
            },
            &mut render,
        );

        assert_eq!(clicked, Some(7));
    }

    #[test]
    fn lock_color_encodes_value_sign() {
        let palette = Palette::dark();
        assert_eq!(lock_color(3, &palette), palette.band_green);
        assert_eq!(lock_color(0, &palette), palette.band_pale);
        assert_eq!(lock_color(-2, &palette), palette.band_amber);
    }
}
