//! Main application shell for the trace comparison visualizer.
//!
//! [`TraceViewerApp`] implements [`eframe::App`] and orchestrates the
//! top-level layout: menu bar, tile sidebar, event detail panel, status
//! bar, and the central timeline rendering area.

use eframe::egui;

use crate::trace::compare::TileKey;

use super::data::{LoadedComparison, TraceSource};
use super::event_detail::{self, SelectedEvent};
use super::tile_selector;
use super::timeline;

// ============================================================================
// TraceViewerApp
// ============================================================================

/// Top-level application state for the trace comparison visualizer.
pub struct TraceViewerApp {
    /// Loaded trace comparison data, if any.
    source: Option<LoadedComparison>,
    /// Currently selected tile in the sidebar.
    selected_tile: Option<TileKey>,
    /// Currently selected event (from timeline interaction).
    selected_event: Option<SelectedEvent>,
    /// Persistent state for the timeline widget (viewport, initialization).
    timeline_state: timeline::TimelineState,
    /// Status bar message.
    status: String,
    /// Error message to display in a popup window.
    error: Option<String>,
    /// Names of available trace batches (for sweep directories).
    batch_names: Vec<String>,
    /// Currently selected batch index.
    selected_batch: usize,
}

impl Default for TraceViewerApp {
    fn default() -> Self {
        Self {
            source: None,
            selected_tile: None,
            selected_event: None,
            timeline_state: timeline::TimelineState::default(),
            status: "Ready. Open a trace pair to begin.".to_string(),
            error: None,
            batch_names: vec!["Batch 0".to_string()],
            selected_batch: 0,
        }
    }
}

impl TraceViewerApp {
    /// Open a file dialog to select HW and EMU trace directories, then load.
    fn open_trace_pair(&mut self) {
        let hw_dir = rfd::FileDialog::new()
            .set_title("Select HW trace directory")
            .pick_folder();

        let hw_dir = match hw_dir {
            Some(d) => d,
            None => return, // User cancelled.
        };

        let emu_dir = rfd::FileDialog::new()
            .set_title("Select EMU trace directory")
            .pick_folder();

        let emu_dir = match emu_dir {
            Some(d) => d,
            None => return, // User cancelled.
        };

        self.load_trace_pair(&hw_dir, &emu_dir);
    }

    /// Load and compare traces from the given directories.
    ///
    /// On success, auto-selects the first tile (most divergent).
    /// On failure, stores the error for popup display.
    fn load_trace_pair(&mut self, hw_dir: &std::path::Path, emu_dir: &std::path::Path) {
        match LoadedComparison::from_trace_dirs(hw_dir, emu_dir) {
            Ok(comparison) => {
                // Auto-select the most divergent tile.
                let first_tile = comparison.tile_keys().first().copied();
                let tile_count = comparison.tile_keys().len();

                self.source = Some(comparison);
                self.selected_tile = first_tile;
                self.selected_event = None;
                self.status = format!(
                    "Loaded {} tiles from {} / {}",
                    tile_count,
                    hw_dir.display(),
                    emu_dir.display(),
                );
                self.error = None;
            }
            Err(e) => {
                self.error = Some(e);
            }
        }
    }
}

impl eframe::App for TraceViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // ====================================================================
        // Menu bar (top panel)
        // ====================================================================
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open Trace Pair...").clicked() {
                        ui.close_menu();
                        self.open_trace_pair();
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                // Batch selector: only shown when multiple batches are
                // available (sweep directories with different event
                // configurations). Single-batch loads hide this.
                if self.batch_names.len() > 1 {
                    ui.separator();
                    egui::ComboBox::from_label("Batch")
                        .selected_text(&self.batch_names[self.selected_batch])
                        .show_ui(ui, |ui| {
                            for (i, name) in self.batch_names.iter().enumerate() {
                                ui.selectable_value(&mut self.selected_batch, i, name);
                            }
                        });
                }
            });
        });

        // ====================================================================
        // Status bar (bottom panel, below everything)
        // ====================================================================
        egui::TopBottomPanel::bottom("status_bar")
            .max_height(20.0)
            .show(ctx, |ui| {
                ui.label(&self.status);
            });

        // ====================================================================
        // Event detail panel (bottom, above status bar)
        // ====================================================================
        egui::TopBottomPanel::bottom("detail_panel")
            .max_height(40.0)
            .show(ctx, |ui| {
                event_detail::show_event_detail(ui, self.selected_event.as_ref());
            });

        // ====================================================================
        // Error popup (if any)
        // ====================================================================
        if self.error.is_some() {
            let mut open = true;
            egui::Window::new("Error")
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    if let Some(ref msg) = self.error {
                        ui.label(msg);
                    }
                });
            if !open {
                self.error = None;
            }
        }

        // ====================================================================
        // Tile sidebar (left panel, only when data is loaded)
        // ====================================================================
        if let Some(ref source) = self.source {
            egui::SidePanel::left("tile_sidebar")
                .default_width(160.0)
                .show(ctx, |ui| {
                    if let Some(new_tile) = tile_selector::show_tile_selector(
                        ui,
                        source,
                        self.selected_tile.as_ref(),
                    ) {
                        self.selected_tile = Some(new_tile);
                        self.selected_event = None;
                        self.timeline_state.reset();
                    }
                });
        }

        // ====================================================================
        // Central panel (timeline)
        // ====================================================================
        egui::CentralPanel::default().show(ctx, |ui| {
            if let (Some(ref source), Some(ref tile)) = (&self.source, &self.selected_tile) {
                if let Some(event) = timeline::show_timeline(
                    ui,
                    source as &dyn TraceSource,
                    tile,
                    &mut self.timeline_state,
                ) {
                    self.selected_event = Some(event);
                }
            } else if self.source.is_some() {
                ui.centered_and_justified(|ui| {
                    ui.label("Select a tile from the sidebar.");
                });
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("No traces loaded. Use File > Open Trace Pair...");
                });
            }
        });
    }
}
