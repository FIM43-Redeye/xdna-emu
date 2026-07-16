//! Control bar (Task 8): cycle/status readout + Step / Step 100 / Run / Pause / Reset.

use eframe::egui;

use crate::debugger::engine_host::{EngineHost, RunState};

pub fn show(ui: &mut egui::Ui, host: &mut EngineHost, run_budget: u32) {
    ui.horizontal(|ui| {
        ui.monospace(format!("cycle {:>8}", host.total_cycles()));
        ui.monospace(format!("{:?}", host.status()));
        ui.separator();
        if ui.button("Step").clicked() {
            host.step_one();
        }
        if ui.button("Step 100").clicked() {
            host.step_bounded(100);
        }
        match host.run_state {
            RunState::Paused => {
                if ui.button("Run").clicked() {
                    host.run_state = RunState::Running;
                }
            }
            RunState::Running => {
                if ui.button("Pause").clicked() {
                    host.run_state = RunState::Paused;
                }
            }
        }
        if ui.button("Reset").clicked() {
            host.reset();
        }
        // run_budget shown for context; a slider replaces this later.
        ui.label(format!("budget/frame: {run_budget}"));
    });
}
