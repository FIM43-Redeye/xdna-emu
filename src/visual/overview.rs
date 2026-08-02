//! Array architecture-view composer: layout, zoom/pan, tiles, axes, and routes.

use eframe::egui::{self, pos2, vec2, Align2, FontId, Pos2, Rect, Sense, Vec2};

use crate::debugger::model::{tile_axes, tile_grid, tile_ports, tile_snapshot, tile_state, PortWire};
use crate::device::array::TileArray;
use crate::device::PortDirection;
use crate::interpreter::InterpreterEngine;
use crate::visual::routes::{draw_routes, RouteLine};
use crate::visual::theme::Palette;
use crate::visual::tile::{tile, DetailTier, PortAnchor, TilePresentation};

const GAP: f32 = 6.0;
const OUTER_PAD: f32 = 8.0;
const AXIS_LEFT: f32 = 42.0;
const AXIS_TOP: f32 = 36.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ZoomLevel {
    #[default]
    Fit,
    Close,
}

struct RenderedTile {
    col: u8,
    row: u8,
    rect: Rect,
    ports: Vec<PortWire>,
    anchors: Vec<PortAnchor>,
}

fn detail_tier(zoom: ZoomLevel, cell: Vec2) -> DetailTier {
    match zoom {
        ZoomLevel::Close => DetailTier::Diagram,
        ZoomLevel::Fit if cell.x >= 44.0 && cell.y >= 26.0 => DetailTier::ColorCode,
        ZoomLevel::Fit if cell.x >= 18.0 && cell.y >= 14.0 => DetailTier::LetterCode,
        ZoomLevel::Fit => DetailTier::Dot,
    }
}

fn grid_size(cell: Vec2, cols: usize, rows: usize) -> Vec2 {
    vec2(
        cell.x * cols as f32 + GAP * cols.saturating_sub(1) as f32,
        cell.y * rows as f32 + GAP * rows.saturating_sub(1) as f32,
    )
}

fn fit_cell_size(available: Vec2, cols: usize, rows: usize) -> Vec2 {
    if cols == 0 || rows == 0 {
        return Vec2::ZERO;
    }
    let base = vec2(64.0, 44.0);
    let width = (available.x - GAP * cols.saturating_sub(1) as f32).max(0.0);
    let height = (available.y - GAP * rows.saturating_sub(1) as f32).max(0.0);
    let scale = (width / (base.x * cols as f32)).min(height / (base.y * rows as f32)).max(0.0);
    base * scale
}

/// Locate a tile from the actual axis values. Rows are deliberately reversed
/// so hardware row 0 (shim where present) remains at the bottom.
fn cell_rect(origin: Pos2, cell: Vec2, cols: &[u8], rows: &[u8], col: u8, row: u8) -> Option<Rect> {
    let col_index = cols.iter().position(|value| *value == col)?;
    let row_index = rows.iter().rev().position(|value| *value == row)?;
    let min = origin + vec2(col_index as f32 * (cell.x + GAP), row_index as f32 * (cell.y + GAP));
    Some(Rect::from_min_size(min, cell))
}

fn route_lines(tiles: &[RenderedTile], tier: DetailTier) -> Vec<RouteLine> {
    let mut routes = Vec::new();
    for source in tiles {
        for port in &source.ports {
            if port.direction != PortDirection::Master {
                continue;
            }
            let Some((dst_col, dst_row, dst_index)) = port.route_to else {
                continue;
            };
            let Some(destination) = tiles.iter().find(|tile| tile.col == dst_col && tile.row == dst_row)
            else {
                continue;
            };

            let (from, to) = if tier == DetailTier::Diagram {
                let Some(from) = source
                    .anchors
                    .iter()
                    .find(|anchor| anchor.index == port.index && anchor.direction == PortDirection::Master)
                    .map(|anchor| anchor.pos)
                else {
                    continue;
                };
                let Some(to) = destination
                    .anchors
                    .iter()
                    .find(|anchor| anchor.index == dst_index && anchor.direction == PortDirection::Slave)
                    .map(|anchor| anchor.pos)
                else {
                    continue;
                };
                (from, to)
            } else {
                let Some(from) = source.anchors.first().map(|anchor| anchor.pos) else {
                    continue;
                };
                let to = destination
                    .anchors
                    .first()
                    .map(|anchor| anchor.pos)
                    .unwrap_or_else(|| destination.rect.center());
                (from, to)
            };

            routes.push(RouteLine { from, to, moving: port.moving, stalled: port.stalled });
        }
    }
    routes
}

fn paint_axes(
    painter: &egui::Painter,
    origin: Pos2,
    cell: Vec2,
    cols: &[u8],
    rows: &[u8],
    palette: &Palette,
) {
    let size = grid_size(cell, cols.len(), rows.len());
    let font = FontId::monospace(11.0);
    painter.text(
        pos2(origin.x + size.x * 0.5, origin.y - 20.0),
        Align2::CENTER_BOTTOM,
        "col",
        font.clone(),
        palette.text,
    );
    for (index, col) in cols.iter().enumerate() {
        painter.text(
            pos2(origin.x + index as f32 * (cell.x + GAP) + cell.x * 0.5, origin.y - 4.0),
            Align2::CENTER_BOTTOM,
            col,
            font.clone(),
            palette.text,
        );
    }

    painter.text(
        pos2(origin.x - 38.0, origin.y + size.y * 0.5),
        Align2::LEFT_CENTER,
        "row",
        font.clone(),
        palette.text,
    );
    for (index, row) in rows.iter().rev().enumerate() {
        painter.text(
            pos2(origin.x - 4.0, origin.y + index as f32 * (cell.y + GAP) + cell.y * 0.5),
            Align2::RIGHT_CENTER,
            row,
            font.clone(),
            palette.text,
        );
    }
}

pub fn show(
    ui: &mut egui::Ui,
    engine: &InterpreterEngine,
    array: &TileArray,
    palette: &Palette,
    selected: &mut Option<(u8, u8)>,
    zoom: &mut ZoomLevel,
    pan: &mut Vec2,
) {
    ui.horizontal(|ui| {
        ui.label("Zoom:");
        ui.selectable_value(zoom, ZoomLevel::Fit, "Fit");
        ui.selectable_value(zoom, ZoomLevel::Close, "Close");
    });

    let cells = tile_grid(array);
    let (cols, rows) = tile_axes(&cells);
    if cells.is_empty() {
        ui.label("No tiles in the array.");
        return;
    }

    let viewport = ui.available_rect_before_wrap();
    let pan_response = ui.allocate_rect(
        viewport,
        if *zoom == ZoomLevel::Close {
            Sense::drag()
        } else {
            Sense::hover()
        },
    );
    let previous_clip = ui.clip_rect();
    ui.set_clip_rect(previous_clip.intersect(viewport));
    ui.painter().rect_filled(viewport, 0.0, palette.bg);

    let plot_size = vec2(
        (viewport.width() - AXIS_LEFT - OUTER_PAD).max(0.0),
        (viewport.height() - AXIS_TOP - OUTER_PAD).max(0.0),
    );
    let plot_origin = viewport.min + vec2(AXIS_LEFT, AXIS_TOP);
    let cell = match zoom {
        ZoomLevel::Fit => fit_cell_size(plot_size, cols.len(), rows.len()),
        ZoomLevel::Close => vec2(112.0, 76.0),
    };
    let grid = grid_size(cell, cols.len(), rows.len());
    let origin = match zoom {
        ZoomLevel::Fit => plot_origin + (plot_size - grid) * 0.5,
        ZoomLevel::Close => plot_origin + *pan,
    };
    let tier = detail_tier(*zoom, cell);
    paint_axes(ui.painter(), origin, cell, &cols, &rows, palette);

    let mut pan_delta = pan_response.drag_delta();
    let mut rendered = Vec::with_capacity(cells.len());
    for cell_data in cells {
        let Some(rect) = cell_rect(origin, cell, &cols, &rows, cell_data.col, cell_data.row) else {
            continue;
        };
        let Some(snapshot) = tile_snapshot(engine, cell_data.col, cell_data.row) else {
            continue;
        };
        let state = tile_state(engine, cell_data.col, cell_data.row);
        let ports = tile_ports(array, cell_data.col, cell_data.row);
        let output = tile(
            ui,
            rect,
            &snapshot,
            &state,
            &ports,
            &TilePresentation { tier, selected: *selected == Some((cell_data.col, cell_data.row)), palette },
        );
        let response = if *zoom == ZoomLevel::Close {
            output.response.interact(Sense::drag())
        } else {
            output.response.clone()
        };
        if response.clicked() {
            *selected = Some((cell_data.col, cell_data.row));
        }
        if response.dragged() {
            pan_delta = response.drag_delta();
        }
        rendered.push(RenderedTile {
            col: cell_data.col,
            row: cell_data.row,
            rect,
            ports,
            anchors: output.anchors,
        });
    }

    let routes = route_lines(&rendered, tier);
    draw_routes(&ui.painter().with_clip_rect(viewport), &routes, palette);
    if *zoom == ZoomLevel::Close {
        *pan += pan_delta;
    }
    ui.set_clip_rect(previous_clip);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debugger::model::PortWire;
    use crate::device::{PortDirection, PortType};
    use crate::visual::tile::{DetailTier, PortAnchor};
    use eframe::egui::{pos2, vec2};

    #[test]
    fn zoom_and_cell_size_select_the_detail_tier() {
        assert_eq!(detail_tier(ZoomLevel::Close, vec2(1.0, 1.0)), DetailTier::Diagram);
        assert_eq!(detail_tier(ZoomLevel::Fit, vec2(60.0, 40.0)), DetailTier::ColorCode);
        assert_eq!(detail_tier(ZoomLevel::Fit, vec2(30.0, 22.0)), DetailTier::LetterCode);
        assert_eq!(detail_tier(ZoomLevel::Fit, vec2(12.0, 8.0)), DetailTier::Dot);
    }

    #[test]
    fn cell_rect_flips_actual_row_axis_values() {
        let cols = [3, 7];
        let rows = [2, 9];
        let top = cell_rect(pos2(10.0, 20.0), vec2(40.0, 30.0), &cols, &rows, 7, 9).unwrap();
        let bottom = cell_rect(pos2(10.0, 20.0), vec2(40.0, 30.0), &cols, &rows, 7, 2).unwrap();

        assert_eq!(top.left(), bottom.left());
        assert!(top.top() < bottom.top());
    }

    #[test]
    fn route_lines_use_diagram_anchors_and_fit_centers() {
        let source_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(20.0, 20.0));
        let destination_rect = Rect::from_min_size(pos2(40.0, 0.0), vec2(20.0, 20.0));
        let tiles = [
            RenderedTile {
                col: 0,
                row: 0,
                rect: source_rect,
                ports: vec![PortWire {
                    index: 12,
                    direction: PortDirection::Master,
                    port_type: PortType::East,
                    moving: true,
                    active: true,
                    stalled: false,
                    route_to: Some((1, 0, 7)),
                }],
                anchors: vec![PortAnchor {
                    index: 12,
                    port_type: PortType::East,
                    direction: PortDirection::Master,
                    pos: pos2(20.0, 8.0),
                }],
            },
            RenderedTile {
                col: 1,
                row: 0,
                rect: destination_rect,
                ports: Vec::new(),
                anchors: vec![PortAnchor {
                    index: 7,
                    port_type: PortType::West,
                    direction: PortDirection::Slave,
                    pos: pos2(40.0, 8.0),
                }],
            },
        ];

        let diagram = route_lines(&tiles, DetailTier::Diagram);

        assert_eq!(diagram.len(), 1);
        assert_eq!(diagram[0].from, pos2(20.0, 8.0));
        assert_eq!(diagram[0].to, pos2(40.0, 8.0));
        assert!(diagram[0].moving);
        assert!(!diagram[0].stalled);

        let mut fit_tiles = tiles;
        fit_tiles[0].anchors[0].pos = pos2(9.0, 10.0);
        fit_tiles[1].anchors[0].pos = pos2(51.0, 10.0);
        let fit = route_lines(&fit_tiles, DetailTier::LetterCode);

        assert_eq!(fit[0].from, pos2(9.0, 10.0));
        assert_eq!(fit[0].to, pos2(51.0, 10.0));
    }
}
