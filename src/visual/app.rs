use std::path::PathBuf;

use eframe::egui;

use crate::debugger::engine_host::{self, EngineHost};

pub struct DebuggerApp {
    host: Option<EngineHost>,
    load_error: Option<String>,
    pub selected: Option<(u8, u8)>,
    /// Cycles advanced per frame while running (single tunable; a speed slider
    /// drops straight in here later).
    pub run_budget: u32,
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
        Self { host, load_error, selected: None, run_budget: 32 }
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
            Some(h) => crate::visual::controls::show(ui, h, self.run_budget),
            None => {
                ui.label(self.load_error.clone().unwrap_or_else(|| "No design loaded".into()));
            }
        });

        egui::SidePanel::left("overview").resizable(true).show(ctx, |ui| {
            if let Some(h) = self.host.as_ref() {
                crate::visual::overview::show(ui, &h.engine.device().array, &mut self.selected);
            } else {
                ui.label("No design loaded");
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| match self.host.as_ref() {
            Some(h) => crate::visual::detail::show(ui, h, self.selected),
            None => {
                ui.label("No design loaded");
            }
        });
    }
}
