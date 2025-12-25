//! Main emulator application.

use eframe::egui;
use std::path::PathBuf;

use crate::interpreter::{InterpreterEngine, EngineStatus};
use crate::parser::{Xclbin, AiePartition, Cdo};
use crate::parser::xclbin::SectionKind;
use crate::parser::cdo::find_cdo_offset;

use super::controls;
use super::tile_grid;
use super::tile_detail;

/// Selected tile for detail view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectedTile {
    pub col: u8,
    pub row: u8,
}

/// Main application state.
pub struct EmulatorApp {
    /// The emulation engine.
    pub engine: InterpreterEngine,
    /// Currently selected tile for detail view.
    pub selected_tile: Option<SelectedTile>,
    /// Path to loaded xclbin file.
    pub loaded_file: Option<PathBuf>,
    /// Status message to display.
    pub status_message: String,
    /// Auto-run mode (continuously step).
    pub auto_run: bool,
    /// Steps per frame in auto-run mode.
    pub steps_per_frame: u64,
    /// Show memory view.
    pub show_memory: bool,
    /// Memory view offset.
    pub memory_offset: usize,
    /// Show program memory (vs data memory).
    pub show_program_memory: bool,
    /// Show lock panel.
    pub show_locks: bool,
    /// Show DMA panel.
    pub show_dma: bool,
    /// Show registers panel.
    pub show_registers: bool,
    /// Error message (if any).
    pub error_message: Option<String>,
    /// Dropped file path (for drag & drop).
    dropped_file: Option<PathBuf>,
}

impl Default for EmulatorApp {
    fn default() -> Self {
        Self {
            engine: InterpreterEngine::new_npu1(),
            selected_tile: Some(SelectedTile { col: 0, row: 2 }),
            loaded_file: None,
            status_message: "Ready. Drag & drop an .xclbin file or use File > Open".to_string(),
            auto_run: false,
            steps_per_frame: 100,
            show_memory: true,
            memory_offset: 0,
            show_program_memory: false,
            show_locks: true,
            show_dma: true,
            show_registers: true,
            error_message: None,
            dropped_file: None,
        }
    }
}

impl EmulatorApp {
    /// Create a new app with an engine.
    pub fn with_engine(engine: InterpreterEngine) -> Self {
        Self {
            engine,
            ..Default::default()
        }
    }

    /// Load an xclbin file.
    pub fn load_xclbin(&mut self, path: &std::path::Path) -> anyhow::Result<()> {
        self.status_message = format!("Loading: {}", path.display());

        // Parse xclbin
        let xclbin = Xclbin::from_file(path)?;

        // Find AIE partition
        let section = xclbin
            .find_section(SectionKind::AiePartition)
            .ok_or_else(|| anyhow::anyhow!("No AIE partition found in xclbin"))?;

        let partition = AiePartition::parse(section.data())?;

        // Get PDI and parse CDO
        let pdi = partition
            .primary_pdi()
            .ok_or_else(|| anyhow::anyhow!("No primary PDI in partition"))?;

        let cdo_offset = find_cdo_offset(pdi.pdi_image)
            .ok_or_else(|| anyhow::anyhow!("No CDO found in PDI"))?;

        let cdo = Cdo::parse(&pdi.pdi_image[cdo_offset..])?;

        // Apply to device
        self.engine = InterpreterEngine::new_npu1();
        self.engine.device_mut().apply_cdo(&cdo)?;

        self.loaded_file = Some(path.to_path_buf());
        self.status_message = format!(
            "Loaded: {} ({} commands, {} bytes program)",
            path.file_name().unwrap_or_default().to_string_lossy(),
            self.engine.device().stats.commands,
            self.engine.device().stats.program_bytes
        );
        self.error_message = None;

        Ok(())
    }

    /// Handle file drop.
    fn handle_file_drop(&mut self, ctx: &egui::Context) {
        // Check for dropped files
        ctx.input(|i| {
            if !i.raw.dropped_files.is_empty() {
                if let Some(file) = i.raw.dropped_files.first() {
                    if let Some(path) = &file.path {
                        self.dropped_file = Some(path.clone());
                    }
                }
            }
        });

        // Process dropped file
        if let Some(path) = self.dropped_file.take() {
            if let Err(e) = self.load_xclbin(&path) {
                self.error_message = Some(format!("Failed to load: {}", e));
                self.status_message = "Load failed".to_string();
            }
        }
    }
}

impl eframe::App for EmulatorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Handle file drops
        self.handle_file_drop(ctx);

        // Auto-run mode
        if self.auto_run && self.engine.status() == EngineStatus::Running {
            self.engine.run(self.steps_per_frame);
            ctx.request_repaint();
        }

        // Menu bar
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open...").clicked() {
                        // Use native file dialog if available
                        if let Some(path) = rfd_open_xclbin() {
                            if let Err(e) = self.load_xclbin(&path) {
                                self.error_message = Some(format!("Failed to load: {}", e));
                            }
                        }
                        ui.close_menu();
                    }
                    if ui.button("Reset").clicked() {
                        self.engine.reset();
                        self.status_message = "Device reset".to_string();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.show_memory, "Memory");
                    ui.checkbox(&mut self.show_locks, "Locks");
                    ui.checkbox(&mut self.show_dma, "DMA");
                    ui.checkbox(&mut self.show_registers, "Registers");
                });

                ui.menu_button("Help", |ui| {
                    if ui.button("About").clicked() {
                        self.status_message = "xdna-emu v0.1.0 - AMD XDNA NPU Emulator".to_string();
                        ui.close_menu();
                    }
                });
            });
        });

        // Control panel at top
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            controls::show_controls(ui, self);
        });

        // Status bar at bottom
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status_message);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!(
                        "Cycles: {} | Instructions: {}",
                        self.engine.total_cycles(),
                        self.engine.total_instructions()
                    ));
                });
            });
        });

        // Error popup
        if let Some(error) = self.error_message.clone() {
            egui::Window::new("Error")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(&error);
                    if ui.button("OK").clicked() {
                        self.error_message = None;
                    }
                });
        }

        // Left panel: tile grid
        egui::SidePanel::left("tile_grid")
            .default_width(280.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.heading("Tile Array");
                ui.separator();
                tile_grid::show_tile_grid(ui, self);
            });

        // Right panel: tile details
        egui::CentralPanel::default().show(ctx, |ui| {
            tile_detail::show_tile_detail(ui, self);
        });
    }
}

/// Open file dialog for xclbin files.
/// Returns None if dialog was cancelled or rfd is not available.
fn rfd_open_xclbin() -> Option<PathBuf> {
    // Try to use rfd if available, otherwise return None
    // For now we rely on drag & drop since rfd adds complexity
    None
}
