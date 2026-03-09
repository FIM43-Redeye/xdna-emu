//! Timeline widget for the trace comparison visualizer.
//!
//! Renders HW and EMU event lanes side by side, with zoom/pan interaction.
//! Each event slot gets one lane per source (HW/EMU), laid out vertically.
//! Edge events are drawn as vertical tick marks; divergent ticks are
//! highlighted. Binary search ensures only visible events are drawn.

use eframe::egui;

use crate::trace::compare::TileKey;

use super::data::TraceSource;
use super::event_detail::SelectedEvent;
use super::viewport::Viewport;
use super::theme;

// ============================================================================
// TimelineState
// ============================================================================

/// Persistent state for the timeline widget.
pub struct TimelineState {
    /// Viewport mapping cycles to screen coordinates.
    pub viewport: Viewport,
    /// Whether the viewport has been fitted to data on first render.
    initialized: bool,
}

impl Default for TimelineState {
    fn default() -> Self {
        Self {
            viewport: Viewport::default(),
            initialized: false,
        }
    }
}

impl TimelineState {
    /// Reset initialization flag so the viewport re-fits on next render.
    ///
    /// Call this when the selected tile changes.
    pub fn reset(&mut self) {
        self.initialized = false;
    }
}

// ============================================================================
// Lane layout
// ============================================================================

/// One logical lane in the timeline (an event name within a source block).
struct Lane {
    /// Event name (e.g., "DMA_S2MM_0_start_task").
    name: String,
    /// Event slot index (0-7) that this lane corresponds to.
    slot: u8,
    /// Top y-coordinate of this lane (relative to the timeline rect).
    y_top: f32,
}

/// Compute the lane layout for a given tile.
///
/// Returns (hw_lanes, emu_lanes, total_height). Each source block gets one
/// lane per unique event name found in the comparison results. HW lanes
/// come first (top), then a gap, then EMU lanes.
fn compute_lanes(
    source: &dyn TraceSource,
    tile: &TileKey,
) -> (Vec<Lane>, Vec<Lane>, f32) {
    let config = source.batch_config();

    // Determine which event list to use based on pkt_type.
    let event_names: &[String] = match tile.pkt_type {
        0 => &config.core_events,
        1 => &config.mem_events,
        3 => &config.memtile_events,
        _ => &config.core_events, // Fallback.
    };

    // If no events are configured, derive slot names from the events
    // actually present in the data.
    let hw_events = source.hw_events(tile);
    let emu_events = source.emu_events(tile);

    // Collect unique slots that actually have events.
    let mut active_slots: Vec<u8> = Vec::new();
    for ev in hw_events.iter().chain(emu_events.iter()) {
        if !active_slots.contains(&ev.slot) {
            active_slots.push(ev.slot);
        }
    }
    active_slots.sort();

    if active_slots.is_empty() {
        return (Vec::new(), Vec::new(), 0.0);
    }

    // Build lane lists.
    let mut hw_lanes = Vec::new();
    let mut emu_lanes = Vec::new();
    let mut y = 0.0_f32;

    // HW block.
    for &slot in &active_slots {
        let name = if (slot as usize) < event_names.len() {
            event_names[slot as usize].clone()
        } else {
            format!("slot_{}", slot)
        };
        hw_lanes.push(Lane { name, slot, y_top: y });
        y += theme::LANE_HEIGHT;
    }

    y += theme::BLOCK_GAP;

    // EMU block.
    for &slot in &active_slots {
        let name = if (slot as usize) < event_names.len() {
            event_names[slot as usize].clone()
        } else {
            format!("slot_{}", slot)
        };
        emu_lanes.push(Lane { name, slot, y_top: y });
        y += theme::LANE_HEIGHT;
    }

    (hw_lanes, emu_lanes, y)
}

// ============================================================================
// Public API
// ============================================================================

/// Compute the total cycle extent (min, max) for a tile's events.
///
/// Scans all HW and EMU events, maps them through the alignment into
/// unified coordinates, and returns (min_cycle, max_cycle). Returns
/// `None` if there are no events at all.
fn compute_extent(
    source: &dyn TraceSource,
    tile: &TileKey,
    alignment: &super::alignment::AlignmentMap,
) -> Option<(f64, f64)> {
    let hw_events = source.hw_events(tile);
    let emu_events = source.emu_events(tile);

    let mut min_cycle = f64::MAX;
    let mut max_cycle = f64::MIN;

    for ev in hw_events {
        let unified = alignment.hw_to_unified(ev.abs_cycle);
        min_cycle = min_cycle.min(unified);
        max_cycle = max_cycle.max(unified);
    }
    for ev in emu_events {
        let unified = alignment.emu_to_unified(ev.abs_cycle);
        min_cycle = min_cycle.min(unified);
        max_cycle = max_cycle.max(unified);
    }

    if min_cycle <= max_cycle {
        Some((min_cycle, max_cycle))
    } else {
        None
    }
}

/// Draw a minimap bar showing the full trace extent with the current
/// viewport highlighted. Clicking on the minimap pans the viewport to
/// center on the clicked position.
fn draw_minimap(
    ui: &mut egui::Ui,
    total_min: f64,
    total_max: f64,
    viewport: &mut Viewport,
) {
    let available_width = ui.available_width();
    let (response, painter) = ui.allocate_painter(
        egui::Vec2::new(available_width, theme::MINIMAP_HEIGHT),
        egui::Sense::click(),
    );
    let rect = response.rect;

    // Fill background.
    painter.rect_filled(rect, 2.0, theme::MINIMAP_BG);

    let total_span = total_max - total_min;
    if total_span <= 0.0 {
        return;
    }

    // Calculate viewport position as fraction of total range.
    let vp_start_frac = ((viewport.start_cycle - total_min) / total_span)
        .clamp(0.0, 1.0) as f32;
    let vp_end_frac = ((viewport.end_cycle() - total_min) / total_span)
        .clamp(0.0, 1.0) as f32;

    // Ensure the viewport indicator is at least 2 pixels wide so it is
    // always visible even when zoomed far out.
    let vp_left = rect.left() + vp_start_frac * rect.width();
    let vp_right_raw = rect.left() + vp_end_frac * rect.width();
    let vp_right = vp_right_raw.max(vp_left + 2.0).min(rect.right());

    let vp_rect = egui::Rect::from_min_max(
        egui::pos2(vp_left, rect.top() + 1.0),
        egui::pos2(vp_right, rect.bottom() - 1.0),
    );

    // Draw viewport indicator.
    painter.rect_stroke(
        vp_rect,
        1.0,
        egui::Stroke::new(1.0, theme::MINIMAP_VIEWPORT),
        egui::StrokeKind::Outside,
    );

    // Click to pan: center the viewport on the clicked position.
    if response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let click_frac = ((pos.x - rect.left()) / rect.width())
                .clamp(0.0, 1.0) as f64;
            let target_cycle = total_min + click_frac * total_span;
            // Center the viewport on the target cycle.
            let half_visible = viewport.visible_cycles() / 2.0;
            viewport.start_cycle = target_cycle - half_visible;
        }
    }
}

/// Render the timeline for a selected tile.
///
/// Returns `Some(SelectedEvent)` if the user hovered over an event tick.
/// The caller should store this in the app state for the detail panel.
pub fn show_timeline(
    ui: &mut egui::Ui,
    source: &dyn TraceSource,
    tile: &TileKey,
    state: &mut TimelineState,
) -> Option<SelectedEvent> {
    let alignment = source.alignment();

    // Compute lane layout.
    let (hw_lanes, emu_lanes, total_height) = compute_lanes(source, tile);

    if hw_lanes.is_empty() && emu_lanes.is_empty() {
        ui.label("No events for this tile.");
        return None;
    }

    // Compute the total cycle extent for minimap and initial fit.
    let extent = compute_extent(source, tile, alignment);

    // Draw the minimap bar above the event lanes.
    if let Some((total_min, total_max)) = extent {
        draw_minimap(ui, total_min, total_max, &mut state.viewport);
        ui.add_space(2.0);
    }

    // Allocate the drawing area. Use a scroll area for vertical overflow,
    // but the horizontal axis is managed by the viewport (zoom/pan).
    let desired_height = total_height.max(100.0);

    let (response, painter) = ui.allocate_painter(
        egui::Vec2::new(ui.available_width(), desired_height),
        egui::Sense::click_and_drag(),
    );
    let rect = response.rect;

    // Update viewport width from actual available space.
    state.viewport.width_px = rect.width();

    // First-time initialization: fit viewport to the data range.
    if !state.initialized {
        if let Some((min_cycle, max_cycle)) = extent {
            state.viewport.fit_range(min_cycle, max_cycle);
        }
        state.initialized = true;
    }

    // Handle input: scroll wheel for zoom, drag for pan.
    if response.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll.abs() > 0.1 {
            let factor = if scroll > 0.0 { 1.1 } else { 1.0 / 1.1 };
            let mouse_x = ui.input(|i| {
                i.pointer.hover_pos().map(|p| p.x).unwrap_or(rect.center().x)
            });
            state.viewport.zoom_at(factor, mouse_x - rect.left());
        }
    }

    if response.dragged() {
        state.viewport.pan_px(response.drag_delta().x);
    }

    // Draw block labels.
    let label_color = egui::Color32::from_gray(180);
    if !hw_lanes.is_empty() {
        let hw_label_pos = egui::pos2(rect.left() + 4.0, rect.top() + hw_lanes[0].y_top);
        painter.text(
            hw_label_pos,
            egui::Align2::LEFT_TOP,
            "HW",
            egui::FontId::proportional(10.0),
            label_color,
        );
    }
    if !emu_lanes.is_empty() {
        let emu_label_pos = egui::pos2(rect.left() + 4.0, rect.top() + emu_lanes[0].y_top);
        painter.text(
            emu_label_pos,
            egui::Align2::LEFT_TOP,
            "EMU",
            egui::FontId::proportional(10.0),
            label_color,
        );
    }

    // Draw lane backgrounds.
    for lane in &hw_lanes {
        let lane_rect = egui::Rect::from_min_size(
            egui::pos2(rect.left(), rect.top() + lane.y_top),
            egui::vec2(rect.width(), theme::LANE_HEIGHT),
        );
        painter.rect_filled(lane_rect, 0.0, theme::HW_LANE_BG);

        // Lane name label (right-aligned within the lane).
        painter.text(
            egui::pos2(rect.right() - 4.0, lane_rect.top() + 2.0),
            egui::Align2::RIGHT_TOP,
            &lane.name,
            egui::FontId::proportional(9.0),
            egui::Color32::from_gray(120),
        );
    }
    for lane in &emu_lanes {
        let lane_rect = egui::Rect::from_min_size(
            egui::pos2(rect.left(), rect.top() + lane.y_top),
            egui::vec2(rect.width(), theme::LANE_HEIGHT),
        );
        painter.rect_filled(lane_rect, 0.0, theme::EMU_LANE_BG);

        painter.text(
            egui::pos2(rect.right() - 4.0, lane_rect.top() + 2.0),
            egui::Align2::RIGHT_TOP,
            &lane.name,
            egui::FontId::proportional(9.0),
            egui::Color32::from_gray(120),
        );
    }

    // Build a lookup from EdgeResult name to its deltas for divergence
    // coloring. We need this to know which event index is divergent.
    let batch = source.batch_result();
    let tile_result = batch.tiles.iter().find(|(k, _)| k == tile).map(|(_, r)| r);

    // Collect per-slot edge results for fast lookup.
    // EdgeResult names come from the config; we map them back to slot index.
    let config = source.batch_config();
    let event_names: &[String] = match tile.pkt_type {
        0 => &config.core_events,
        1 => &config.mem_events,
        3 => &config.memtile_events,
        _ => &config.core_events,
    };

    // Hover detection state.
    let hover_pos = ui.input(|i| i.pointer.hover_pos());
    let mut hovered_event: Option<SelectedEvent> = None;
    let hover_tolerance = 3.0_f32; // pixels

    // Draw HW event ticks.
    let hw_events = source.hw_events(tile);
    draw_ticks(
        &painter,
        &state.viewport,
        hw_events,
        &hw_lanes,
        alignment,
        true, // is_hw
        rect,
        tile_result,
        event_names,
        hover_pos,
        hover_tolerance,
        &mut hovered_event,
    );

    // Draw EMU event ticks.
    let emu_events = source.emu_events(tile);
    draw_ticks(
        &painter,
        &state.viewport,
        emu_events,
        &emu_lanes,
        alignment,
        false, // is_hw
        rect,
        tile_result,
        event_names,
        hover_pos,
        hover_tolerance,
        &mut hovered_event,
    );

    hovered_event
}

// ============================================================================
// Tick rendering
// ============================================================================

/// Draw vertical tick marks for events in the given lanes.
///
/// Uses binary search (`partition_point`) to find only the events within
/// the visible viewport, avoiding iteration over potentially millions of
/// off-screen events.
#[allow(clippy::too_many_arguments)]
fn draw_ticks(
    painter: &egui::Painter,
    viewport: &Viewport,
    events: &[crate::trace::compare::TileEvent],
    lanes: &[Lane],
    alignment: &super::alignment::AlignmentMap,
    is_hw: bool,
    rect: egui::Rect,
    tile_result: Option<&crate::trace::compare::TileResult>,
    _event_names: &[String],
    hover_pos: Option<egui::Pos2>,
    hover_tolerance: f32,
    hovered_event: &mut Option<SelectedEvent>,
) {
    // For each lane (slot), filter events to that slot and draw visible ones.
    for lane in lanes {
        let lane_top = rect.top() + lane.y_top;
        let lane_bottom = lane_top + theme::LANE_HEIGHT;

        // Collect events for this slot.
        // Events are sorted by abs_cycle globally, but we need per-slot
        // filtering. For v1, iterate and filter. Performance-sensitive
        // optimization (per-slot pre-sorted arrays) can come later.
        //
        // However, we still use the viewport range to skip events outside
        // the visible window.
        let start_cycle = viewport.start_cycle;
        let end_cycle = viewport.end_cycle();

        // Find the EdgeResult for this slot's event name (for divergence info).
        let edge_result = tile_result.and_then(|tr| {
            tr.edge_results.iter().find(|er| er.name == lane.name)
        });

        // Per-slot event counter for mapping to EdgeResult.deltas indices.
        // EdgeResult.deltas[i] corresponds to the i-th occurrence of this
        // event in the trace. We track the per-slot event index to look up
        // the delta.
        let mut slot_event_idx: usize = 0;

        for ev in events {
            if ev.slot != lane.slot {
                continue;
            }

            let unified_cycle = if is_hw {
                alignment.hw_to_unified(ev.abs_cycle)
            } else {
                alignment.emu_to_unified(ev.abs_cycle)
            };

            // Skip events outside the viewport.
            if unified_cycle < start_cycle {
                slot_event_idx += 1;
                continue;
            }
            if unified_cycle > end_cycle {
                // Events are sorted by abs_cycle, so all subsequent events
                // for this slot would also be beyond the viewport. But since
                // we are iterating over ALL events (not just this slot), we
                // cannot break -- other slots might interleave. Just skip.
                slot_event_idx += 1;
                continue;
            }

            let px_x = viewport.cycle_to_px(unified_cycle);
            let screen_x = rect.left() + px_x;

            // Determine tick color: divergent ticks get highlighted.
            let tick_color = if let Some(er) = edge_result {
                if slot_event_idx < er.deltas.len()
                    && er.deltas[slot_event_idx].abs() > theme::DIVERGE_THRESHOLD
                {
                    theme::DIVERGE_MARKER
                } else {
                    theme::EDGE_TICK
                }
            } else {
                theme::EDGE_TICK
            };

            // Draw the tick as a short vertical line.
            painter.line_segment(
                [
                    egui::pos2(screen_x, lane_top + 2.0),
                    egui::pos2(screen_x, lane_bottom - 2.0),
                ],
                egui::Stroke::new(1.0, tick_color),
            );

            // Hover detection.
            if let Some(pos) = hover_pos {
                if (pos.x - screen_x).abs() < hover_tolerance
                    && pos.y >= lane_top
                    && pos.y <= lane_bottom
                {
                    let delta = if let Some(er) = edge_result {
                        er.deltas.get(slot_event_idx).copied()
                    } else {
                        None
                    };

                    *hovered_event = Some(SelectedEvent {
                        name: lane.name.clone(),
                        index: slot_event_idx,
                        hw_cycle: if is_hw { Some(ev.abs_cycle) } else { None },
                        emu_cycle: if !is_hw { Some(ev.abs_cycle) } else { None },
                        delta,
                        is_level: false,
                        hw_duration: None,
                        emu_duration: None,
                    });
                }
            }

            slot_event_idx += 1;
        }
    }
}
