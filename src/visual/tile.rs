//! Reusable tile atom for both the array overview and the detail inspector.

use eframe::egui::{
    self, Align2, FontId, Pos2, Rect, Response, Sense, Stroke, StrokeKind, Vec2, WidgetInfo, WidgetType,
};

use crate::debugger::model::{accessible_label, PortWire, TileKindDisplay, TileSnapshot, TileState};
use crate::device::{PortDirection, PortType};
use crate::visual::theme::Palette;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DetailTier {
    Dot,
    ColorCode,
    LetterCode,
    Diagram,
}

pub struct TilePresentation<'a> {
    pub tier: DetailTier,
    pub selected: bool,
    pub palette: &'a Palette,
}

#[derive(Clone, Copy, Debug)]
pub struct PortAnchor {
    pub index: u8,
    pub port_type: PortType,
    pub direction: PortDirection,
    pub pos: Pos2,
}

pub struct TileOutput {
    pub response: Response,
    pub anchors: Vec<PortAnchor>,
}

/// Normalized position for a stream port. External ports land on their
/// physical edge; local ports occupy stable positions inside the tile.
pub(crate) fn port_offset(port_type: PortType, index: u8) -> Vec2 {
    // External bundles have at most eight ports on the current architecture.
    // Port indices are global within a direction, so modulo preserves distinct
    // lanes within each contiguous bundle without baking in a device layout.
    let lane = f32::from(index % 8 + 1) / 9.0;
    match port_type {
        PortType::North => Vec2::new(lane, 0.0),
        PortType::South => Vec2::new(lane, 1.0),
        PortType::East => Vec2::new(1.0, lane),
        PortType::West => Vec2::new(0.0, lane),
        PortType::Dma(channel) => Vec2::new(0.2 + 0.12 * f32::from(channel % 5), 0.72),
        PortType::Core => Vec2::new(0.5, 0.5),
        PortType::TileCtrl => Vec2::new(0.5, 0.24),
        PortType::Cascade => Vec2::new(0.76, 0.5),
        PortType::Fifo => Vec2::new(0.24, 0.5),
        PortType::Trace => Vec2::new(0.76, 0.24),
    }
}

fn anchor_pos(rect: Rect, port_type: PortType, index: u8) -> Pos2 {
    let offset = port_offset(port_type, index);
    Pos2::new(rect.left() + offset.x * rect.width(), rect.top() + offset.y * rect.height())
}

fn kind_color(kind: TileKindDisplay, palette: &Palette) -> egui::Color32 {
    match kind {
        TileKindDisplay::Shim => palette.kind_shim,
        TileKindDisplay::Mem => palette.kind_mem,
        TileKindDisplay::Core => palette.kind_core,
    }
}

fn inward_vector(port_type: PortType) -> Vec2 {
    match port_type {
        PortType::North => Vec2::new(0.0, 6.0),
        PortType::South => Vec2::new(0.0, -6.0),
        PortType::East => Vec2::new(-6.0, 0.0),
        PortType::West => Vec2::new(6.0, 0.0),
        _ => Vec2::ZERO,
    }
}

fn paint_code(ui: &egui::Ui, rect: Rect, text: &str, state: &TileState, palette: &Palette) {
    let size = (rect.height() * 0.3).clamp(8.0, 18.0);
    let font = FontId::monospace(size);
    ui.painter()
        .text(rect.center(), Align2::CENTER_CENTER, text, font.clone(), palette.text);
    if state.is_error() {
        // egui's default font API has no weight knob; a second close pass gives
        // ERR/! the heavier treatment required for the fault state.
        ui.painter().text(
            rect.center() + Vec2::new(0.6, 0.0),
            Align2::CENTER_CENTER,
            text,
            font,
            palette.text,
        );
    }
}

fn paint_error_frame(ui: &egui::Ui, rect: Rect, palette: &Palette) {
    let stroke = Stroke::new(3.0_f32, palette.err_border);
    ui.painter().rect_stroke(rect, 3.0, stroke, StrokeKind::Outside);

    // Short diagonal marks make the alarm recognizable without relying on red.
    let mark = 5.0;
    for x in [rect.left() + mark, rect.right() - mark] {
        ui.painter()
            .line_segment([Pos2::new(x - mark, rect.top()), Pos2::new(x, rect.top() + mark)], stroke);
        ui.painter()
            .line_segment([Pos2::new(x - mark, rect.bottom() - mark), Pos2::new(x, rect.bottom())], stroke);
    }
}

pub(crate) fn paint_port(ui: &egui::Ui, rect: Rect, port: &PortWire, palette: &Palette) -> Pos2 {
    let pos = anchor_pos(rect, port.port_type, port.index);
    let color = if port.stalled {
        palette.route_stalled
    } else if port.moving {
        palette.route_moving
    } else {
        palette.route_idle
    };
    let width: f32 = if port.active || port.stalled { 2.0 } else { 1.0 };
    let stroke = Stroke::new(width, color);

    if port.port_type.is_external() {
        let inward = inward_vector(port.port_type);
        let (from, to) = match port.direction {
            PortDirection::Master => (pos + inward, pos),
            PortDirection::Slave => (pos, pos + inward),
        };
        ui.painter().arrow(from, to - from, stroke);
    }

    pos
}

/// Paint one tile and return its accessible interaction plus route anchors.
pub fn tile(
    ui: &mut egui::Ui,
    rect: Rect,
    snap: &TileSnapshot,
    state: &TileState,
    ports: &[PortWire],
    pres: &TilePresentation<'_>,
) -> TileOutput {
    let id = ui.make_persistent_id(("architecture-tile", snap.col, snap.row));
    let response = ui.interact(rect, id, Sense::click());
    let label = accessible_label(snap.col, snap.row, snap.kind, state);
    // egui 0.31 maps WidgetType::Button to AccessKit's Role::Button.
    response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, true, &label));

    let frame = (rect.width().min(rect.height()) * 0.08).clamp(1.5, 4.0);
    ui.painter().rect_filled(rect, 3.0, kind_color(snap.kind, pres.palette));
    let band_rect = rect.shrink(frame);
    ui.painter().rect_filled(band_rect, 2.0, pres.palette.band(state));

    let mut anchors = Vec::new();
    match pres.tier {
        DetailTier::Diagram => {
            paint_code(ui, band_rect, state.code(), state, pres.palette);
            anchors.extend(ports.iter().filter_map(|port| {
                let pos = paint_port(ui, rect, port, pres.palette);
                port.port_type.is_external().then_some(PortAnchor {
                    index: port.index,
                    port_type: port.port_type,
                    direction: port.direction,
                    pos,
                })
            }));
        }
        DetailTier::ColorCode => {
            paint_code(ui, band_rect, state.code(), state, pres.palette);
        }
        DetailTier::LetterCode => {
            let mut code = [0; 4];
            paint_code(ui, band_rect, state.letter().encode_utf8(&mut code), state, pres.palette);
        }
        DetailTier::Dot if state.is_error() => paint_code(ui, band_rect, "!", state, pres.palette),
        DetailTier::Dot => {}
    }

    if pres.tier != DetailTier::Diagram {
        anchors.push(PortAnchor {
            index: 0,
            port_type: PortType::Core,
            direction: PortDirection::Master,
            pos: rect.center(),
        });
    }

    if pres.selected {
        ui.painter()
            .rect_stroke(rect, 3.0, Stroke::new(3.0_f32, pres.palette.selected), StrokeKind::Outside);
    }
    if state.is_error() {
        paint_error_frame(ui, rect, pres.palette);
    }

    TileOutput { response, anchors }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debugger::model::{PortWire, TileKindDisplay, TileSnapshot, TileState};
    use crate::device::{PortDirection, PortType};
    use crate::visual::theme::Palette;

    fn snapshot() -> TileSnapshot {
        TileSnapshot {
            col: 2,
            row: 3,
            kind: TileKindDisplay::Core,
            core_status: None,
            pc: None,
            dma: Vec::new(),
            locks: Vec::new(),
            mem_size: 0,
            mem_peek: Vec::new(),
            ports: Vec::new(),
        }
    }

    #[test]
    fn external_port_offsets_land_on_their_physical_edges() {
        assert_eq!(port_offset(PortType::North, 2).y, 0.0);
        assert_eq!(port_offset(PortType::South, 2).y, 1.0);
        assert_eq!(port_offset(PortType::East, 2).x, 1.0);
        assert_eq!(port_offset(PortType::West, 2).x, 0.0);
        let local = port_offset(PortType::Core, 0);
        assert!(local.x > 0.0 && local.x < 1.0);
        assert!(local.y > 0.0 && local.y < 1.0);
    }

    #[test]
    fn indices_spread_ports_along_an_edge() {
        assert_ne!(port_offset(PortType::North, 1), port_offset(PortType::North, 2));
    }

    #[test]
    fn tile_is_clickable_and_diagram_returns_external_anchors() {
        eframe::egui::__run_test_ui(|ui| {
            let rect = eframe::egui::Rect::from_min_size(ui.min_rect().min, eframe::egui::vec2(80.0, 60.0));
            let ports = [PortWire {
                index: 2,
                direction: PortDirection::Master,
                port_type: PortType::North,
                moving: true,
                active: true,
                stalled: false,
                route_to: Some((2, 4, 5)),
            }];
            let palette = Palette::dark();
            let output = tile(
                ui,
                rect,
                &snapshot(),
                &TileState::Running,
                &ports,
                &TilePresentation { tier: DetailTier::Diagram, selected: false, palette: &palette },
            );

            assert!(output.response.sense.senses_click());
            assert_eq!(output.anchors.len(), 1);
            assert_eq!(output.anchors[0].pos.y, rect.top());
        });
    }

    #[test]
    fn coarse_tier_returns_one_center_anchor() {
        eframe::egui::__run_test_ui(|ui| {
            let rect = eframe::egui::Rect::from_min_size(ui.min_rect().min, eframe::egui::vec2(40.0, 30.0));
            let palette = Palette::dark();
            let output = tile(
                ui,
                rect,
                &snapshot(),
                &TileState::Ready,
                &[],
                &TilePresentation { tier: DetailTier::LetterCode, selected: false, palette: &palette },
            );

            assert_eq!(output.anchors.len(), 1);
            assert_eq!(output.anchors[0].pos, rect.center());
        });
    }
}
