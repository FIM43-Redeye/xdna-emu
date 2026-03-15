//! Edge detection circuit for event signal transitions.

/// Single edge detection circuit.
///
/// Monitors one event signal and generates an EDGE_DETECTION_EVENT when
/// the signal transitions (rising, falling, or both). Each module has two
/// independent edge detectors (SelectId 0 and 1).
///
/// Register layout (Edge_Detection_event_control):
/// - Event 0: bits [6:0] event select, bit 9 rising, bit 10 falling
/// - Event 1: bits [22:16] event select, bit 25 rising, bit 26 falling
/// - MemTile: 8-bit event fields (bits [7:0] and [23:16])
#[derive(Debug, Clone, Copy)]
pub struct EdgeDetector {
    /// Hardware event ID to monitor (0 = disabled).
    pub input_event: u8,
    /// Fire on 0->1 transition.
    pub trigger_rising: bool,
    /// Fire on 1->0 transition.
    pub trigger_falling: bool,
    /// Whether the monitored event was active last cycle.
    pub(super) prev_active: bool,
    /// Whether the monitored event was active this cycle (accumulates
    /// during event notification, reset at end of cycle).
    pub(super) curr_active: bool,
}

impl Default for EdgeDetector {
    fn default() -> Self {
        Self {
            input_event: 0,
            trigger_rising: false,
            trigger_falling: false,
            prev_active: false,
            curr_active: false,
        }
    }
}
