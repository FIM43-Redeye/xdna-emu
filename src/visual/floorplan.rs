//! Reusable tile floor-plan schematic for architecture views.

use eframe::egui::{
    self, pos2, vec2, Align2, FontId, Rect, Sense, Stroke, StrokeKind, UiBuilder, WidgetInfo, WidgetType,
};

use crate::debugger::model::{ChannelSnapshot, PortWire, TileKindDisplay, TileSnapshot, TileState};
use crate::visual::theme::Palette;
use crate::visual::tile::paint_port;

pub struct FloorplanPresentation<'a> {
    pub palette: &'a Palette,
    pub mem_texture: Option<egui::TextureId>,
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

fn paint_dma(ui: &egui::Ui, rect: Rect, channels: &[ChannelSnapshot], palette: &Palette) {
    block_label(ui, Rect::from_min_max(rect.min, pos2(rect.right(), rect.top() + 13.0)), "DMA", palette);
    if channels.is_empty() {
        return;
    }

    let content = Rect::from_min_max(pos2(rect.left() + 3.0, rect.top() + 14.0), rect.max - vec2(3.0, 3.0));
    let cols = if channels.len() > 3 { 2 } else { 1 };
    let rows = channels.len().div_ceil(cols);
    let gap = 1.0;
    let cell_size = vec2(
        (content.width() - gap * (cols - 1) as f32) / cols as f32,
        (content.height() - gap * (rows - 1) as f32) / rows as f32,
    );

    for (i, channel) in channels.iter().enumerate() {
        let col = i % cols;
        let row = i / cols;
        let min = content.min + vec2(col as f32 * (cell_size.x + gap), row as f32 * (cell_size.y + gap));
        let cell = Rect::from_min_size(min, cell_size);
        let color = match channel.state.as_str() {
            "Idle" => palette.band_grey,
            "Error" => palette.band_red,
            "Paused" => palette.band_amber,
            state if state.starts_with("WaitingForLock") => palette.band_amber,
            _ => palette.band_green,
        };
        let bd = channel
            .current_bd
            .or(channel.queued_bd)
            .map_or_else(|| "-".into(), |bd| bd.to_string());
        let state = channel.state.chars().next().unwrap_or('-');
        ui.painter().rect_filled(cell, 1.0, color);
        ui.painter().text(
            cell.center(),
            Align2::CENTER_CENTER,
            format!("ch{} {state} b{bd} q{}", channel.index, channel.queue_len),
            FontId::monospace((cell.height() * 0.35).clamp(6.0, 9.0)),
            palette.text,
        );
    }
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
) {
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
    paint_dma(ui, blocks.dma, &snap.dma, pres.palette);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::{pos2, vec2, Rect};

    use crate::debugger::model::{ChannelSnapshot, TileKindDisplay, TileSnapshot, TileState};
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
                    &FloorplanPresentation { palette: &palette, mem_texture: None },
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
                &FloorplanPresentation { palette: &palette, mem_texture: Some(egui::TextureId::Managed(1)) },
            );
        });
    }

    #[test]
    fn lock_color_encodes_value_sign() {
        let palette = Palette::dark();
        assert_eq!(lock_color(3, &palette), palette.band_green);
        assert_eq!(lock_color(0, &palette), palette.band_pale);
        assert_eq!(lock_color(-2, &palette), palette.band_amber);
    }
}
