//! Event detail panel for the trace comparison visualizer.
//!
//! Bottom panel showing details for a selected or hovered event.
//! Displays event name, index, HW/EMU cycles, delta, and durations
//! for level (interval) events.

use eframe::egui;

// ============================================================================
// SelectedEvent
// ============================================================================

/// Information about the currently selected or hovered event.
///
/// Populated by the timeline renderer when the user interacts with an event
/// bar or edge tick. The detail panel reads this to display a summary.
#[derive(Debug, Clone)]
pub struct SelectedEvent {
    /// Event slot name (e.g., "DMA_S2MM_0_start_task").
    pub name: String,
    /// Paired event index within this event type.
    pub index: usize,
    /// HW cycle at which this event occurred (None if HW side is missing).
    pub hw_cycle: Option<u64>,
    /// EMU cycle at which this event occurred (None if EMU side is missing).
    pub emu_cycle: Option<u64>,
    /// Signed delta (hw_cycle - emu_cycle), if both sides present.
    pub delta: Option<i64>,
    /// True for level (interval) events, false for edge (instantaneous).
    pub is_level: bool,
    /// HW-side interval duration (only meaningful for level events).
    pub hw_duration: Option<u64>,
    /// EMU-side interval duration (only meaningful for level events).
    pub emu_duration: Option<u64>,
}

// ============================================================================
// Public API
// ============================================================================

/// Render the event detail panel.
///
/// If `event` is `Some`, displays a horizontal row of labeled values.
/// If `None`, displays a hint message directing the user to click an event.
pub fn show_event_detail(ui: &mut egui::Ui, event: Option<&SelectedEvent>) {
    match event {
        Some(ev) => {
            ui.horizontal(|ui| {
                ui.strong(&ev.name);
                ui.separator();

                ui.label(format!("#{}", ev.index));
                ui.separator();

                if let Some(hw) = ev.hw_cycle {
                    ui.label(format!("HW: {}", hw));
                } else {
                    ui.label("HW: --");
                }

                if let Some(emu) = ev.emu_cycle {
                    ui.label(format!("EMU: {}", emu));
                } else {
                    ui.label("EMU: --");
                }

                if let Some(delta) = ev.delta {
                    ui.separator();
                    let sign = if delta > 0 { "+" } else { "" };
                    ui.label(format!("delta: {}{}", sign, delta));
                }

                if ev.is_level {
                    ui.separator();
                    if let Some(hw_dur) = ev.hw_duration {
                        ui.label(format!("HW dur: {}", hw_dur));
                    }
                    if let Some(emu_dur) = ev.emu_duration {
                        ui.label(format!("EMU dur: {}", emu_dur));
                    }
                }
            });
        }
        None => {
            ui.label("Select an event in the timeline to see details.");
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_event_clone_and_debug() {
        let event = SelectedEvent {
            name: "DMA_start".to_string(),
            index: 42,
            hw_cycle: Some(1000),
            emu_cycle: Some(1010),
            delta: Some(-10),
            is_level: false,
            hw_duration: None,
            emu_duration: None,
        };
        let cloned = event.clone();
        assert_eq!(cloned.name, "DMA_start");
        assert_eq!(cloned.index, 42);
        assert_eq!(cloned.delta, Some(-10));
        // Debug formatting should not panic.
        let _ = format!("{:?}", cloned);
    }

    #[test]
    fn selected_event_level_with_durations() {
        let event = SelectedEvent {
            name: "lock_stall".to_string(),
            index: 0,
            hw_cycle: Some(500),
            emu_cycle: Some(510),
            delta: Some(-10),
            is_level: true,
            hw_duration: Some(200),
            emu_duration: Some(190),
        };
        assert!(event.is_level);
        assert_eq!(event.hw_duration, Some(200));
        assert_eq!(event.emu_duration, Some(190));
    }

    #[test]
    fn selected_event_missing_sides() {
        let event = SelectedEvent {
            name: "orphan".to_string(),
            index: 3,
            hw_cycle: None,
            emu_cycle: Some(100),
            delta: None,
            is_level: false,
            hw_duration: None,
            emu_duration: None,
        };
        assert!(event.hw_cycle.is_none());
        assert!(event.delta.is_none());
    }
}
