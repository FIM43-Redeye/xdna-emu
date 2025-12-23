//! Emulation control panel.

use eframe::egui;

use crate::emu::EngineStatus;

use super::app::EmulatorApp;

/// Show the emulation controls.
pub fn show_controls(ui: &mut egui::Ui, app: &mut EmulatorApp) {
    ui.horizontal(|ui| {
        // Run/Pause button
        let (run_text, run_enabled) = match app.engine.status {
            EngineStatus::Running => ("Pause", true),
            EngineStatus::Paused | EngineStatus::Idle => ("Run", true),
            EngineStatus::Breakpoint { .. } => ("Resume", true),
            EngineStatus::AllHalted => ("Run", false),
        };

        if ui.add_enabled(run_enabled, egui::Button::new(run_text)).clicked() {
            match app.engine.status {
                EngineStatus::Running => {
                    app.auto_run = false;
                    app.engine.pause();
                }
                EngineStatus::Paused | EngineStatus::Idle | EngineStatus::Breakpoint { .. } => {
                    app.auto_run = true;
                    app.engine.resume();
                }
                _ => {}
            }
        }

        // Step button
        let step_enabled = !matches!(app.engine.status, EngineStatus::Running);
        if ui.add_enabled(step_enabled, egui::Button::new("Step")).clicked() {
            app.engine.step();
        }

        // Step N button
        if ui.add_enabled(step_enabled, egui::Button::new("Step 100")).clicked() {
            app.engine.run(100);
        }

        // Reset button
        if ui.button("Reset").clicked() {
            app.auto_run = false;
            app.engine.reset();
            app.status_message = "Device reset".to_string();
        }

        ui.separator();

        // Status display
        let status_color = match app.engine.status {
            EngineStatus::Idle => egui::Color32::GRAY,
            EngineStatus::Running => egui::Color32::GREEN,
            EngineStatus::Paused => egui::Color32::YELLOW,
            EngineStatus::Breakpoint { .. } => egui::Color32::from_rgb(200, 50, 200),
            EngineStatus::AllHalted => egui::Color32::RED,
        };

        ui.colored_label(status_color, app.engine.status_string());

        ui.separator();

        // Speed control
        ui.label("Speed:");
        ui.add(egui::Slider::new(&mut app.steps_per_frame, 1..=10000).logarithmic(true));
        ui.label("steps/frame");

        ui.separator();

        // Stats
        ui.label(format!("Active: {}", app.engine.active_cores()));
        ui.label(format!("Enabled: {}", app.engine.enabled_cores()));
    });
}
