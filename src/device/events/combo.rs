// Combo event logic for AIE2 event subsystem.
//
// Each module has 4 combo events (COMBO_EVENT_0 through COMBO_EVENT_3).
// Combo events apply boolean logic to pairs of input events:
//
//   Combo 0: logic(EventA, EventB)
//   Combo 1: logic(EventC, EventD)
//   Combo 2: logic(Combo0_output, Combo1_output)
//
// Per aie-rt xaie_events.c _XAie_EventComboControl():
// - Combo 0 and 1 each take two configurable input events (A/B or C/D).
// - Combo 2 is a meta-combo: it combines the outputs of Combo 0 and Combo 1
//   (no separate input events -- the input register write is skipped).
// - Combo 3 is available as an additional combo event slot.
//
// Register layout (from xaiemlgbl_params.h):
//   COMBO_EVENT_INPUTS (one 32-bit register):
//     [6:0]   EventA  (combo 0 input 1)
//     [14:8]  EventB  (combo 0 input 2)
//     [22:16] EventC  (combo 1 input 1)
//     [30:24] EventD  (combo 1 input 2)
//
//   COMBO_EVENT_CONTROL (one 32-bit register):
//     [1:0]   Combo0 logic operation
//     [9:8]   Combo1 logic operation
//     [17:16] Combo2 logic operation
//
// Logic operations (per XAie_EventComboOps):
//   0 = E1 AND E2
//   1 = E1 AND (NOT E2)
//   2 = E1 OR E2
//   3 = E1 OR (NOT E2)

/// Hardware event ID type (7-bit for core/mem/PL, 8-bit for memtile).
pub type EventId = u8;

/// Combo logic operation.
///
/// Per aie-rt XAie_EventComboOps enum. These are the 2-bit values written
/// to the COMBO_EVENT_CONTROL register fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ComboLogic {
    /// E1 AND E2
    And = 0,
    /// E1 AND (NOT E2)
    AndNot = 1,
    /// E1 OR E2
    Or = 2,
    /// E1 OR (NOT E2)
    OrNot = 3,
}

impl ComboLogic {
    /// Decode a 2-bit register value to ComboLogic.
    pub fn from_bits(val: u8) -> Self {
        match val & 0x3 {
            0 => ComboLogic::And,
            1 => ComboLogic::AndNot,
            2 => ComboLogic::Or,
            3 => ComboLogic::OrNot,
            _ => unreachable!(),
        }
    }

    /// Evaluate this logic operation on two boolean inputs.
    pub fn evaluate(self, e1: bool, e2: bool) -> bool {
        match self {
            ComboLogic::And => e1 && e2,
            ComboLogic::AndNot => e1 && !e2,
            ComboLogic::Or => e1 || e2,
            ComboLogic::OrNot => e1 || !e2,
        }
    }
}

/// A single combo event unit.
///
/// Each combo event takes two input events and applies a logic operation.
/// Combo 2 is special: its inputs are the outputs of Combo 0 and Combo 1.
#[derive(Debug, Clone)]
pub struct ComboEvent {
    /// First input event ID.
    pub input_a: EventId,
    /// Second input event ID.
    pub input_b: EventId,
    /// Logic operation to apply.
    pub logic: ComboLogic,
}

impl ComboEvent {
    /// Create a new combo event with default (disabled) state.
    pub fn new() -> Self {
        Self { input_a: 0, input_b: 0, logic: ComboLogic::And }
    }

    /// Configure this combo event with input events and logic.
    pub fn configure(&mut self, input_a: EventId, input_b: EventId, logic: ComboLogic) {
        self.input_a = input_a;
        self.input_b = input_b;
        self.logic = logic;
    }

    /// Evaluate this combo event given a set of currently active events.
    ///
    /// For regular combos (0 and 1), checks if input_a and input_b are
    /// in the active set, then applies the logic operation.
    pub fn evaluate(&self, is_active: &dyn Fn(EventId) -> bool) -> bool {
        let a = is_active(self.input_a);
        let b = is_active(self.input_b);
        self.logic.evaluate(a, b)
    }

    /// Reset to default state.
    pub fn reset(&mut self) {
        self.input_a = 0;
        self.input_b = 0;
        self.logic = ComboLogic::And;
    }
}

impl Default for ComboEvent {
    fn default() -> Self {
        Self::new()
    }
}

/// All 4 combo events for a module, plus the meta-combo (combo 2) logic.
///
/// Per aie-rt:
/// - Combos 0 and 1 have independently configurable input events.
/// - Combo 2 combines the outputs of combo 0 and combo 1 (its input
///   events are ignored; the hardware wires combo 0/1 outputs directly).
/// - Combo 3 has its own independent input events.
#[derive(Debug, Clone)]
pub struct ComboEventConfig {
    /// The 4 combo event units. Index matches combo ID (0-3).
    pub combos: [ComboEvent; 4],
}

impl ComboEventConfig {
    /// Create a new combo event configuration in reset state.
    pub fn new() -> Self {
        Self {
            combos: [ComboEvent::new(), ComboEvent::new(), ComboEvent::new(), ComboEvent::new()],
        }
    }

    /// Configure a combo event.
    ///
    /// For combo_id 2 (meta-combo), input_a and input_b are ignored --
    /// the hardware always uses combo 0 and combo 1 outputs.
    pub fn configure(&mut self, combo_id: usize, input_a: EventId, input_b: EventId, logic: ComboLogic) {
        if combo_id >= 4 {
            return;
        }
        self.combos[combo_id].logic = logic;
        // Per aie-rt: combo 2 skips input event register config.
        if combo_id != 2 {
            self.combos[combo_id].input_a = input_a;
            self.combos[combo_id].input_b = input_b;
        }
    }

    /// Evaluate all combo events, returning which combo event IDs fire.
    ///
    /// The `is_active` callback checks whether a given event ID is
    /// currently asserted. The `combo_event_base` is the hardware event
    /// ID of COMBO_EVENT_0 in this module's event space.
    ///
    /// Returns a list of hardware event IDs for fired combo events.
    pub fn evaluate(&self, is_active: &dyn Fn(EventId) -> bool, combo_event_base: EventId) -> Vec<EventId> {
        let mut fired = Vec::new();

        // Evaluate combo 0.
        let c0 = self.combos[0].evaluate(is_active);
        // Evaluate combo 1.
        let c1 = self.combos[1].evaluate(is_active);
        // Evaluate combo 2 (meta-combo: inputs are c0 and c1 outputs).
        let c2 = self.combos[2].logic.evaluate(c0, c1);
        // Evaluate combo 3 (independent).
        let c3 = self.combos[3].evaluate(is_active);

        if c0 {
            fired.push(combo_event_base);
        }
        if c1 {
            fired.push(combo_event_base.wrapping_add(1));
        }
        if c2 {
            fired.push(combo_event_base.wrapping_add(2));
        }
        if c3 {
            fired.push(combo_event_base.wrapping_add(3));
        }

        fired
    }

    /// Read the COMBO_EVENT_INPUTS register value.
    ///
    /// Layout: [6:0] EventA, [14:8] EventB, [22:16] EventC, [30:24] EventD
    /// where A/B are combo 0 inputs, C/D are combo 1 inputs.
    pub fn read_input_register(&self) -> u32 {
        let a = self.combos[0].input_a as u32;
        let b = (self.combos[0].input_b as u32) << 8;
        let c = (self.combos[1].input_a as u32) << 16;
        let d = (self.combos[1].input_b as u32) << 24;
        a | b | c | d
    }

    /// Write the COMBO_EVENT_INPUTS register value.
    pub fn write_input_register(&mut self, value: u32) {
        self.combos[0].input_a = (value & 0xFF) as u8;
        self.combos[0].input_b = ((value >> 8) & 0xFF) as u8;
        self.combos[1].input_a = ((value >> 16) & 0xFF) as u8;
        self.combos[1].input_b = ((value >> 24) & 0xFF) as u8;
    }

    /// Read the COMBO_EVENT_CONTROL register value.
    ///
    /// Layout: [1:0] combo0_logic, [9:8] combo1_logic, [17:16] combo2_logic
    pub fn read_control_register(&self) -> u32 {
        let c0 = self.combos[0].logic as u32;
        let c1 = (self.combos[1].logic as u32) << 8;
        let c2 = (self.combos[2].logic as u32) << 16;
        c0 | c1 | c2
    }

    /// Write the COMBO_EVENT_CONTROL register value.
    pub fn write_control_register(&mut self, value: u32) {
        self.combos[0].logic = ComboLogic::from_bits((value & 0x3) as u8);
        self.combos[1].logic = ComboLogic::from_bits(((value >> 8) & 0x3) as u8);
        self.combos[2].logic = ComboLogic::from_bits(((value >> 16) & 0x3) as u8);
    }

    /// Reset all combo events to default state.
    pub fn reset(&mut self) {
        for combo in &mut self.combos {
            combo.reset();
        }
    }
}

impl Default for ComboEventConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- ComboLogic tests --

    #[test]
    fn test_combo_logic_from_bits() {
        assert_eq!(ComboLogic::from_bits(0), ComboLogic::And);
        assert_eq!(ComboLogic::from_bits(1), ComboLogic::AndNot);
        assert_eq!(ComboLogic::from_bits(2), ComboLogic::Or);
        assert_eq!(ComboLogic::from_bits(3), ComboLogic::OrNot);
        // High bits ignored.
        assert_eq!(ComboLogic::from_bits(0xFF), ComboLogic::OrNot);
    }

    #[test]
    fn test_combo_logic_and() {
        assert!(ComboLogic::And.evaluate(true, true));
        assert!(!ComboLogic::And.evaluate(true, false));
        assert!(!ComboLogic::And.evaluate(false, true));
        assert!(!ComboLogic::And.evaluate(false, false));
    }

    #[test]
    fn test_combo_logic_and_not() {
        assert!(!ComboLogic::AndNot.evaluate(true, true));
        assert!(ComboLogic::AndNot.evaluate(true, false));
        assert!(!ComboLogic::AndNot.evaluate(false, true));
        assert!(!ComboLogic::AndNot.evaluate(false, false));
    }

    #[test]
    fn test_combo_logic_or() {
        assert!(ComboLogic::Or.evaluate(true, true));
        assert!(ComboLogic::Or.evaluate(true, false));
        assert!(ComboLogic::Or.evaluate(false, true));
        assert!(!ComboLogic::Or.evaluate(false, false));
    }

    #[test]
    fn test_combo_logic_or_not() {
        // OrNot(e1, e2) = e1 || !e2
        assert!(ComboLogic::OrNot.evaluate(true, true)); // true || !true = true
        assert!(ComboLogic::OrNot.evaluate(true, false)); // true || !false = true
        assert!(!ComboLogic::OrNot.evaluate(false, true)); // false || !true = false
                                                           // false || !false == false || true == true
        assert!(ComboLogic::OrNot.evaluate(false, false));
    }

    // -- ComboEvent tests --

    #[test]
    fn test_combo_event_new() {
        let ce = ComboEvent::new();
        assert_eq!(ce.input_a, 0);
        assert_eq!(ce.input_b, 0);
        assert_eq!(ce.logic, ComboLogic::And);
    }

    #[test]
    fn test_combo_event_configure() {
        let mut ce = ComboEvent::new();
        ce.configure(10, 20, ComboLogic::Or);
        assert_eq!(ce.input_a, 10);
        assert_eq!(ce.input_b, 20);
        assert_eq!(ce.logic, ComboLogic::Or);
    }

    #[test]
    fn test_combo_event_evaluate_and() {
        let mut ce = ComboEvent::new();
        ce.configure(5, 10, ComboLogic::And);

        // Both active.
        assert!(ce.evaluate(&|id| id == 5 || id == 10));
        // Only first active.
        assert!(!ce.evaluate(&|id| id == 5));
        // Neither active.
        assert!(!ce.evaluate(&|_| false));
    }

    #[test]
    fn test_combo_event_evaluate_or() {
        let mut ce = ComboEvent::new();
        ce.configure(5, 10, ComboLogic::Or);

        assert!(ce.evaluate(&|id| id == 5 || id == 10));
        assert!(ce.evaluate(&|id| id == 5));
        assert!(ce.evaluate(&|id| id == 10));
        assert!(!ce.evaluate(&|_| false));
    }

    #[test]
    fn test_combo_event_reset() {
        let mut ce = ComboEvent::new();
        ce.configure(5, 10, ComboLogic::Or);
        ce.reset();
        assert_eq!(ce.input_a, 0);
        assert_eq!(ce.input_b, 0);
        assert_eq!(ce.logic, ComboLogic::And);
    }

    // -- ComboEventConfig tests --

    #[test]
    fn test_combo_config_new() {
        let cfg = ComboEventConfig::new();
        for combo in &cfg.combos {
            assert_eq!(combo.input_a, 0);
            assert_eq!(combo.input_b, 0);
            assert_eq!(combo.logic, ComboLogic::And);
        }
    }

    #[test]
    fn test_combo_config_configure_regular() {
        let mut cfg = ComboEventConfig::new();
        cfg.configure(0, 5, 10, ComboLogic::And);
        assert_eq!(cfg.combos[0].input_a, 5);
        assert_eq!(cfg.combos[0].input_b, 10);
        assert_eq!(cfg.combos[0].logic, ComboLogic::And);
    }

    #[test]
    fn test_combo_config_configure_meta() {
        let mut cfg = ComboEventConfig::new();
        // Combo 2 is the meta-combo: inputs are ignored.
        cfg.configure(2, 99, 99, ComboLogic::Or);
        // Input events should NOT be updated (they stay at 0).
        assert_eq!(cfg.combos[2].input_a, 0);
        assert_eq!(cfg.combos[2].input_b, 0);
        // But the logic should be updated.
        assert_eq!(cfg.combos[2].logic, ComboLogic::Or);
    }

    #[test]
    fn test_combo_config_evaluate_and_requires_both() {
        let mut cfg = ComboEventConfig::new();
        // Combo 0: event 5 AND event 10.
        cfg.configure(0, 5, 10, ComboLogic::And);

        let base = 9; // COMBO_EVENT_0 base ID
                      // Both active -> combo 0 fires.
        let fired = cfg.evaluate(&|id| id == 5 || id == 10, base);
        assert!(fired.contains(&9));

        // Only one active -> combo 0 does not fire.
        let fired = cfg.evaluate(&|id| id == 5, base);
        assert!(!fired.contains(&9));
    }

    #[test]
    fn test_combo_config_evaluate_or_fires_on_any() {
        let mut cfg = ComboEventConfig::new();
        cfg.configure(0, 5, 10, ComboLogic::Or);

        let base = 9;
        let fired = cfg.evaluate(&|id| id == 5, base);
        assert!(fired.contains(&9));

        let fired = cfg.evaluate(&|id| id == 10, base);
        assert!(fired.contains(&9));

        let fired = cfg.evaluate(&|_| false, base);
        assert!(!fired.contains(&9));
    }

    #[test]
    fn test_combo_config_meta_combo2() {
        let mut cfg = ComboEventConfig::new();
        // Combo 0: event 5 AND event 10 -> fires when both active.
        cfg.configure(0, 5, 10, ComboLogic::And);
        // Combo 1: event 20 AND event 30 -> fires when both active.
        cfg.configure(1, 20, 30, ComboLogic::And);
        // Combo 2 (meta): combo0 OR combo1.
        cfg.configure(2, 0, 0, ComboLogic::Or);

        let base = 9;

        // Only combo 0 inputs active -> combo 0 fires, combo 2 fires.
        let fired = cfg.evaluate(&|id| id == 5 || id == 10, base);
        assert!(fired.contains(&9)); // combo 0
        assert!(!fired.contains(&10)); // combo 1
        assert!(fired.contains(&11)); // combo 2 (meta OR)

        // Only combo 1 inputs active -> combo 1 fires, combo 2 fires.
        let fired = cfg.evaluate(&|id| id == 20 || id == 30, base);
        assert!(!fired.contains(&9)); // combo 0
        assert!(fired.contains(&10)); // combo 1
        assert!(fired.contains(&11)); // combo 2 (meta OR)

        // Neither -> combo 2 does not fire.
        let fired = cfg.evaluate(&|_| false, base);
        assert!(!fired.contains(&11));
    }

    #[test]
    fn test_combo_config_meta_combo2_and() {
        let mut cfg = ComboEventConfig::new();
        cfg.configure(0, 5, 10, ComboLogic::And);
        cfg.configure(1, 20, 30, ComboLogic::And);
        // Combo 2: combo0 AND combo1 -- both must fire.
        cfg.configure(2, 0, 0, ComboLogic::And);

        let base = 9;

        // Only combo 0 fires -> combo 2 does not.
        let fired = cfg.evaluate(&|id| id == 5 || id == 10, base);
        assert!(fired.contains(&9)); // combo 0
        assert!(!fired.contains(&11)); // combo 2

        // Both combo 0 and combo 1 fire -> combo 2 fires.
        let fired = cfg.evaluate(&|id| id == 5 || id == 10 || id == 20 || id == 30, base);
        assert!(fired.contains(&9)); // combo 0
        assert!(fired.contains(&10)); // combo 1
        assert!(fired.contains(&11)); // combo 2
    }

    #[test]
    fn test_combo_config_combo3_independent() {
        let mut cfg = ComboEventConfig::new();
        cfg.configure(3, 42, 43, ComboLogic::Or);

        let base = 9;
        let fired = cfg.evaluate(&|id| id == 42, base);
        assert!(fired.contains(&12)); // combo 3 = base + 3
    }

    #[test]
    fn test_combo_config_out_of_bounds() {
        let mut cfg = ComboEventConfig::new();
        // combo_id 4 is out of bounds -- should be silently ignored.
        cfg.configure(4, 5, 10, ComboLogic::And);
        // Nothing changed.
        for combo in &cfg.combos {
            assert_eq!(combo.logic, ComboLogic::And);
            assert_eq!(combo.input_a, 0);
        }
    }

    // -- Register interface tests --

    #[test]
    fn test_input_register_rw() {
        let mut cfg = ComboEventConfig::new();
        cfg.configure(0, 0x12, 0x34, ComboLogic::And);
        cfg.configure(1, 0x56, 0x78, ComboLogic::And);

        let reg = cfg.read_input_register();
        assert_eq!(reg & 0xFF, 0x12); // EventA
        assert_eq!((reg >> 8) & 0xFF, 0x34); // EventB
        assert_eq!((reg >> 16) & 0xFF, 0x56); // EventC
        assert_eq!((reg >> 24) & 0xFF, 0x78); // EventD

        // Write a different value.
        cfg.write_input_register(0xAB_CD_EF_01);
        assert_eq!(cfg.combos[0].input_a, 0x01);
        assert_eq!(cfg.combos[0].input_b, 0xEF);
        assert_eq!(cfg.combos[1].input_a, 0xCD);
        assert_eq!(cfg.combos[1].input_b, 0xAB);
    }

    #[test]
    fn test_control_register_rw() {
        let mut cfg = ComboEventConfig::new();
        cfg.combos[0].logic = ComboLogic::Or; // 2
        cfg.combos[1].logic = ComboLogic::AndNot; // 1
        cfg.combos[2].logic = ComboLogic::OrNot; // 3

        let reg = cfg.read_control_register();
        assert_eq!(reg & 0x3, 2); // combo 0
        assert_eq!((reg >> 8) & 0x3, 1); // combo 1
        assert_eq!((reg >> 16) & 0x3, 3); // combo 2

        cfg.write_control_register(0x0003_0000); // combo2=OrNot(3), combo1=And(0), combo0=And(0)
        assert_eq!(cfg.combos[0].logic, ComboLogic::And);
        assert_eq!(cfg.combos[1].logic, ComboLogic::And);
        assert_eq!(cfg.combos[2].logic, ComboLogic::OrNot);
    }

    #[test]
    fn test_combo_config_reset() {
        let mut cfg = ComboEventConfig::new();
        cfg.configure(0, 5, 10, ComboLogic::Or);
        cfg.configure(1, 20, 30, ComboLogic::AndNot);
        cfg.reset();
        for combo in &cfg.combos {
            assert_eq!(combo.input_a, 0);
            assert_eq!(combo.input_b, 0);
            assert_eq!(combo.logic, ComboLogic::And);
        }
    }

    #[test]
    fn test_multiple_combos_simultaneous() {
        let mut cfg = ComboEventConfig::new();
        cfg.configure(0, 5, 10, ComboLogic::And);
        cfg.configure(1, 5, 20, ComboLogic::And);
        cfg.configure(3, 5, 30, ComboLogic::Or);

        let base = 9;
        // Event 5 is active, event 10 is active, event 20 is not.
        let fired = cfg.evaluate(&|id| id == 5 || id == 10, base);
        // Combo 0: 5 AND 10 -> true.
        assert!(fired.contains(&9));
        // Combo 1: 5 AND 20 -> false.
        assert!(!fired.contains(&10));
        // Combo 3: 5 OR 30 -> true (5 is active).
        assert!(fired.contains(&12));
    }
}
