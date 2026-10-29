use std::path::PathBuf;

use eframe::egui;

use crate::debugger::engine_host::{self, EngineHost};
use crate::visual::memviz;
use crate::visual::overview::ZoomLevel;
use crate::visual::theme::Palette;

pub struct DebuggerApp {
    host: Option<EngineHost>,
    load_error: Option<String>,
    pub selected: Option<(u8, u8)>,
    pub selected_dma: Option<u8>,
    /// Cycles advanced per frame while running (single tunable; a speed slider
    /// drops straight in here later).
    pub run_budget: u32,
    pub overview_zoom: ZoomLevel,
    pub overview_pan: egui::Vec2,
    pub high_contrast: bool,
    mem_textures: memviz::MemoryTextures,
}

impl DebuggerApp {
    pub fn new(xclbin: Option<PathBuf>) -> Self {
        let (host, load_error) = match xclbin {
            Some(p) => match engine_host::load(&p) {
                Ok(h) => (Some(h), None),
                Err(e) => (None, Some(e)),
            },
            None => (None, None),
        };
        Self {
            host,
            load_error,
            selected: None,
            selected_dma: None,
            run_budget: 32,
            overview_zoom: ZoomLevel::Fit,
            overview_pan: egui::Vec2::ZERO,
            high_contrast: false,
            mem_textures: memviz::MemoryTextures::default(),
        }
    }
}

fn palette(ui: &egui::Ui, high_contrast: bool) -> Palette {
    if high_contrast {
        Palette::high_contrast()
    } else if ui.visuals().dark_mode {
        Palette::dark()
    } else {
        Palette::light()
    }
}

fn reset_dma_selection(
    previous_tile: Option<(u8, u8)>,
    selected_tile: Option<(u8, u8)>,
    selected_dma: &mut Option<u8>,
) {
    if previous_tile != selected_tile {
        *selected_dma = None;
    }
}

impl eframe::App for DebuggerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Advance while running, bounded per frame; request continuous repaint.
        if let Some(h) = self.host.as_mut() {
            if h.run_state == engine_host::RunState::Running {
                let status = h.step_bounded(self.run_budget);
                use crate::interpreter::EngineStatus;
                if matches!(status, EngineStatus::Halted | EngineStatus::Stalled | EngineStatus::Error) {
                    h.run_state = engine_host::RunState::Paused;
                } else {
                    ctx.request_repaint();
                }
            }
        }

        egui::TopBottomPanel::top("controls").show(ctx, |ui| match self.host.as_mut() {
            Some(h) => crate::visual::controls::show(ui, h, &mut self.run_budget),
            None => {
                ui.label(self.load_error.clone().unwrap_or_else(|| "No design loaded".into()));
            }
        });

        // A design with no control program loads and steps fine but can never
        // move data -- every channel stalls for real. Say so, loudly: silence
        // here is indistinguishable from a deadlocked kernel.
        if let Some(warning) = self.host.as_ref().and_then(|h| h.control_program.warning()) {
            let high_contrast = self.high_contrast;
            egui::TopBottomPanel::top("control_program_warning").show(ctx, |ui| {
                let color = palette(ui, high_contrast).band_amber;
                ui.horizontal_wrapped(|ui| {
                    // Redundant coding: the word carries the meaning even where
                    // the amber does not read (WCAG 1.4.1).
                    ui.colored_label(color, "WARNING:");
                    ui.colored_label(color, warning);
                });
            });
        }

        let previous_selected = self.selected;
        egui::SidePanel::left("overview").resizable(true).show(ctx, |ui| {
            if let Some(h) = self.host.as_ref() {
                let palette = palette(ui, self.high_contrast);
                crate::visual::overview::show(
                    ui,
                    &h.engine,
                    &h.engine.device().array,
                    &palette,
                    &mut self.selected,
                    &mut self.overview_zoom,
                    &mut self.overview_pan,
                );
            } else {
                ui.label("No design loaded");
            }
        });
        reset_dma_selection(previous_selected, self.selected, &mut self.selected_dma);

        let mem_texture = {
            let host = self.host.as_ref();
            let mem_textures = &mut self.mem_textures;
            if let (Some(host), Some((col, row))) = (host, self.selected) {
                host.engine.device().array.get(col, row).and_then(|tile| {
                    mem_textures.texture(ctx, col, row, tile.data_memory(), tile.data_memory_gen())
                })
            } else {
                None
            }
        };

        egui::CentralPanel::default().show(ctx, |ui| match self.host.as_ref() {
            Some(h) => {
                let palette = palette(ui, self.high_contrast);
                crate::visual::detail::show(
                    ui,
                    h,
                    self.selected,
                    &mut self.selected_dma,
                    &palette,
                    mem_texture,
                );
            }
            None => {
                ui.label("No design loaded");
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dma_selection_resets_only_when_the_tile_changes() {
        let mut app = DebuggerApp::new(None);
        assert_eq!(app.selected_dma, None);

        app.selected_dma = Some(2);
        reset_dma_selection(Some((0, 2)), Some((0, 2)), &mut app.selected_dma);
        assert_eq!(app.selected_dma, Some(2));

        reset_dma_selection(Some((0, 2)), Some((1, 2)), &mut app.selected_dma);
        assert_eq!(app.selected_dma, None);
    }
}
