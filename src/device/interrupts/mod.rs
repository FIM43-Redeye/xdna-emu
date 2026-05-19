//! AIE2 interrupt controller emulation.
//!
//! The AIE2 interrupt system has two levels:
//!
//! - **Level 1 (L1)**: Located in each shim tile's PL module stream switch.
//!   Maps internal events to interrupt lines via IRQ event registers. Has two
//!   independent switches (A and B) with identical register layouts offset by
//!   0x30. Each switch has 20 interrupt IDs, 4 IRQ event slots, and 16
//!   broadcast channels. Events are wire-OR'd into interrupt output lines.
//!
//! - **Level 2 (L2)**: Located in shim NoC tiles only. Aggregates L1
//!   interrupt outputs from the column (via broadcast network). Has 16
//!   broadcast input channels and routes to up to 4 NoC/host interrupt
//!   outputs.
//!
//! # Register Interface (from aie-rt xaiemlgbl_params.h)
//!
//! ## L1 Registers (PL_MODULE, per switch, Switch B = Switch A + 0x30)
//!
//! | Register          | Offset (A) | Width | Description                          |
//! |-------------------|------------|-------|--------------------------------------|
//! | Mask              | 0x35000    | 20    | Read-only: current enable state      |
//! | Enable            | 0x35004    | 20    | Write-to-set bits in mask            |
//! | Disable           | 0x35008    | 20    | Write-to-clear bits in mask          |
//! | Status            | 0x3500C    | 20    | Latched interrupts; write-to-clear   |
//! | IRQ_NO            | 0x35010    | 4     | Broadcast ID for interrupt output    |
//! | IRQ_EVENT         | 0x35014    | 28    | 4 event slots, 7 bits each           |
//! | Block_North_Set   | 0x35018    | 16    | Block broadcast from north (set)     |
//! | Block_North_Clear | 0x3501C    | 16    | Block broadcast from north (clear)   |
//! | Block_North_Value | 0x35020    | 16    | Block broadcast from north (current) |
//!
//! ## L2 Registers (NOC_MODULE, shim NoC tiles only)
//!
//! | Register  | Offset  | Width | Description                          |
//! |-----------|---------|-------|--------------------------------------|
//! | Mask      | 0x15000 | 16    | Read-only: current enable state      |
//! | Enable    | 0x15004 | 16    | Write-to-set bits in mask            |
//! | Disable   | 0x15008 | 16    | Write-to-clear bits in mask          |
//! | Status    | 0x1500C | 16    | Latched interrupts; write-to-clear   |
//! | Interrupt | 0x15010 | 2     | NoC interrupt output status          |
//!
//! # Hardware Behavior (per aie-rt)
//!
//! - Enable register: writing a 1 to bit N sets bit N in the mask register.
//! - Disable register: writing a 1 to bit N clears bit N in the mask register.
//! - Status register: reading returns latched interrupt state. Writing a 1 to
//!   bit N clears (acknowledges) that bit.
//! - L1 IRQ event register: 4 event slots packed at 7 bits each (bits [6:0],
//!   [14:8], [22:16], [30:24]). When the configured event fires, the
//!   corresponding interrupt ID (slot index + 16) latches in the status register.
//! - L1 broadcast block: prevents broadcast signals from propagating through
//!   the switch after being captured by the interrupt controller. Set/Clear
//!   registers modify the block value register.

pub mod l1;
pub mod l2;

pub use l1::{L1InterruptController, SwitchId};
pub use l2::L2InterruptController;

/// Number of interrupt IDs in the L1 controller per switch.
/// Per aie-rt: `NumIntrIds = 20`.
pub const L1_NUM_INTERRUPT_IDS: u8 = 20;

/// Number of IRQ event mapping slots in the L1 controller per switch.
/// Per aie-rt: `NumIrqEvents = 4`.
pub const L1_NUM_IRQ_EVENTS: u8 = 4;

/// Number of broadcast IDs in the L1 controller.
/// Per aie-rt: `NumBroadcastIds = 16`.
pub const L1_NUM_BROADCAST_IDS: u8 = 16;

/// Bit width of each IRQ event field in the packed IRQ_EVENT register.
/// Per aie-rt: field width = 7 (from `BaseIrqEventMask = 0x7F`).
pub const L1_IRQ_EVENT_FIELD_WIDTH: u8 = 7;

/// Stride between IRQ event fields in the packed IRQ_EVENT register.
/// Per aie-rt: `IrqEventOff = 8`.
pub const L1_IRQ_EVENT_FIELD_STRIDE: u8 = 8;

/// Mask for a single IRQ event field (7 bits).
pub const L1_IRQ_EVENT_FIELD_MASK: u32 = 0x7F;

/// Offset between Switch A and Switch B register banks.
/// Per aie-rt: `SwOff = 0x30`.
pub const L1_SWITCH_OFFSET: u32 = 0x30;

/// Number of broadcast IDs in the L2 controller.
/// Per aie-rt: `NumBroadcastIds = 16`.
pub const L2_NUM_BROADCAST_IDS: u8 = 16;

/// Number of NoC interrupt output lines from L2.
/// Per aie-rt: `NumNoCIntr = 4`.
pub const L2_NUM_NOC_INTERRUPTS: u8 = 4;

// -- L1 register offsets (PL_MODULE, Switch A base) --
// Per aie-rt xaiemlgbl_params.h

/// L1 Mask register offset (Switch A).
pub const L1_REG_MASK_A: u32 = 0x0003_5000;
/// L1 Enable register offset (Switch A).
pub const L1_REG_ENABLE_A: u32 = 0x0003_5004;
/// L1 Disable register offset (Switch A).
pub const L1_REG_DISABLE_A: u32 = 0x0003_5008;
/// L1 Status register offset (Switch A).
pub const L1_REG_STATUS_A: u32 = 0x0003_500C;
/// L1 IRQ number (broadcast ID) register offset (Switch A).
pub const L1_REG_IRQ_NO_A: u32 = 0x0003_5010;
/// L1 IRQ event mapping register offset (Switch A).
pub const L1_REG_IRQ_EVENT_A: u32 = 0x0003_5014;
/// L1 Block north-in set register offset (Switch A).
pub const L1_REG_BLOCK_NORTH_SET_A: u32 = 0x0003_5018;
/// L1 Block north-in clear register offset (Switch A).
pub const L1_REG_BLOCK_NORTH_CLEAR_A: u32 = 0x0003_501C;
/// L1 Block north-in value register offset (Switch A).
pub const L1_REG_BLOCK_NORTH_VALUE_A: u32 = 0x0003_5020;

// -- L2 register offsets (NOC_MODULE) --
// Per aie-rt xaiemlgbl_params.h

/// L2 Mask register offset.
pub const L2_REG_MASK: u32 = 0x0001_5000;
/// L2 Enable register offset.
pub const L2_REG_ENABLE: u32 = 0x0001_5004;
/// L2 Disable register offset.
pub const L2_REG_DISABLE: u32 = 0x0001_5008;
/// L2 Status register offset.
pub const L2_REG_STATUS: u32 = 0x0001_500C;
/// L2 Interrupt output register offset.
pub const L2_REG_INTERRUPT: u32 = 0x0001_5010;

/// Valid mask for L1 interrupt IDs (20 bits).
pub const L1_VALID_MASK: u32 = 0x000F_FFFF;
/// Valid mask for L1 broadcast block (16 bits).
pub const L1_BLOCK_VALID_MASK: u32 = 0x0000_FFFF;
/// Valid mask for L2 channels (16 bits).
pub const L2_VALID_MASK: u32 = 0x0000_FFFF;
/// Valid mask for L2 NoC interrupt output (2 bits).
pub const L2_NOC_INTERRUPT_MASK: u32 = 0x0000_0003;

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // L1: initial state
    // ---------------------------------------------------------------

    #[test]
    fn l1_initial_state() {
        let ctrl = L1InterruptController::new();
        assert_eq!(ctrl.read_status(SwitchId::A), 0);
        assert_eq!(ctrl.read_mask(SwitchId::A), 0);
        assert_eq!(ctrl.read_status(SwitchId::B), 0);
        assert_eq!(ctrl.read_mask(SwitchId::B), 0);
        assert_eq!(ctrl.read_irq_no(SwitchId::A), 0);
        assert_eq!(ctrl.read_irq_no(SwitchId::B), 0);
        assert_eq!(ctrl.read_irq_event(SwitchId::A), 0);
        assert_eq!(ctrl.read_irq_event(SwitchId::B), 0);
        assert_eq!(ctrl.read_block_north_value(SwitchId::A), 0);
        assert_eq!(ctrl.read_block_north_value(SwitchId::B), 0);
    }

    // ---------------------------------------------------------------
    // L1: enable / disable / mask
    // ---------------------------------------------------------------

    #[test]
    fn l1_enable_sets_mask_bits() {
        let mut ctrl = L1InterruptController::new();
        ctrl.write_enable(SwitchId::A, 0b1001);
        assert_eq!(ctrl.read_mask(SwitchId::A), 0b1001);
    }

    #[test]
    fn l1_enable_accumulates() {
        let mut ctrl = L1InterruptController::new();
        ctrl.write_enable(SwitchId::A, 0b1001);
        ctrl.write_enable(SwitchId::A, 1 << 5);
        assert_eq!(ctrl.read_mask(SwitchId::A), 0b1001 | (1 << 5));
    }

    #[test]
    fn l1_disable_clears_mask_bits() {
        let mut ctrl = L1InterruptController::new();
        ctrl.write_enable(SwitchId::A, 0b1001 | (1 << 5));
        ctrl.write_disable(SwitchId::A, 0b0001);
        assert_eq!(ctrl.read_mask(SwitchId::A), 0b1000 | (1 << 5));
    }

    #[test]
    fn l1_switches_have_independent_masks() {
        let mut ctrl = L1InterruptController::new();
        ctrl.write_enable(SwitchId::A, 0xFF);
        assert_eq!(ctrl.read_mask(SwitchId::B), 0);
    }

    #[test]
    fn l1_enable_clamps_to_valid_mask() {
        let mut ctrl = L1InterruptController::new();
        ctrl.write_enable(SwitchId::A, 0xFFFF_FFFF);
        assert_eq!(ctrl.read_mask(SwitchId::A), L1_VALID_MASK);
    }

    // ---------------------------------------------------------------
    // L1: event signaling
    // ---------------------------------------------------------------

    #[test]
    fn l1_event_maps_to_interrupt_line() {
        let mut ctrl = L1InterruptController::new();
        // Map event 42 to IRQ event slot 0 on Switch A.
        // Slot 0 corresponds to interrupt ID 16.
        ctrl.set_irq_event_slot(SwitchId::A, 0, 42);
        ctrl.write_enable(SwitchId::A, 1 << 16);

        let result = ctrl.signal_event(SwitchId::A, 42);
        assert_eq!(result, Some(16));
        assert_ne!(ctrl.read_status(SwitchId::A) & (1 << 16), 0);
    }

    #[test]
    fn l1_disabled_interrupt_does_not_latch() {
        let mut ctrl = L1InterruptController::new();
        // Map event 10 to slot 1 (interrupt ID 17) but do NOT enable ID 17.
        ctrl.set_irq_event_slot(SwitchId::A, 1, 10);

        let result = ctrl.signal_event(SwitchId::A, 10);
        assert_eq!(result, None);
        assert_eq!(ctrl.read_status(SwitchId::A), 0);
    }

    #[test]
    fn l1_unmapped_event_does_nothing() {
        let mut ctrl = L1InterruptController::new();
        ctrl.write_enable(SwitchId::A, L1_VALID_MASK);
        // Signal event 99 -- not mapped to any slot (all slots are 0).
        let result = ctrl.signal_event(SwitchId::A, 99);
        assert_eq!(result, None);
        assert_eq!(ctrl.read_status(SwitchId::A), 0);
    }

    #[test]
    fn l1_multiple_events_different_slots() {
        let mut ctrl = L1InterruptController::new();
        ctrl.set_irq_event_slot(SwitchId::A, 0, 10); // slot 0 -> ID 16
        ctrl.set_irq_event_slot(SwitchId::A, 1, 20); // slot 1 -> ID 17
        ctrl.write_enable(SwitchId::A, (1 << 16) | (1 << 17));

        ctrl.signal_event(SwitchId::A, 10);
        ctrl.signal_event(SwitchId::A, 20);

        let status = ctrl.read_status(SwitchId::A);
        assert_ne!(status & (1 << 16), 0, "slot 0 / ID 16 should be latched");
        assert_ne!(status & (1 << 17), 0, "slot 1 / ID 17 should be latched");
    }

    #[test]
    fn l1_switches_signal_independently() {
        let mut ctrl = L1InterruptController::new();
        ctrl.set_irq_event_slot(SwitchId::A, 0, 5);
        ctrl.set_irq_event_slot(SwitchId::B, 0, 5);
        ctrl.write_enable(SwitchId::A, 1 << 16);
        // Only A is enabled, not B.

        assert_eq!(ctrl.signal_event(SwitchId::A, 5), Some(16));
        assert_eq!(ctrl.signal_event(SwitchId::B, 5), None);
    }

    // ---------------------------------------------------------------
    // L1: status clear (ack)
    // ---------------------------------------------------------------

    #[test]
    fn l1_status_clear() {
        let mut ctrl = L1InterruptController::new();
        ctrl.set_irq_event_slot(SwitchId::A, 0, 5);
        ctrl.write_enable(SwitchId::A, 1 << 16);
        ctrl.signal_event(SwitchId::A, 5);
        assert_ne!(ctrl.read_status(SwitchId::A), 0);

        ctrl.clear_status(SwitchId::A, 1 << 16);
        assert_eq!(ctrl.read_status(SwitchId::A), 0);
    }

    #[test]
    fn l1_status_clear_is_selective() {
        let mut ctrl = L1InterruptController::new();
        ctrl.set_irq_event_slot(SwitchId::A, 0, 1);
        ctrl.set_irq_event_slot(SwitchId::A, 1, 2);
        ctrl.write_enable(SwitchId::A, (1 << 16) | (1 << 17));
        ctrl.signal_event(SwitchId::A, 1);
        ctrl.signal_event(SwitchId::A, 2);

        // Clear only ID 16, leave ID 17.
        ctrl.clear_status(SwitchId::A, 1 << 16);
        assert_eq!(ctrl.read_status(SwitchId::A) & (1 << 16), 0);
        assert_ne!(ctrl.read_status(SwitchId::A) & (1 << 17), 0);
    }

    // ---------------------------------------------------------------
    // L1: IRQ event register
    // ---------------------------------------------------------------

    #[test]
    fn l1_irq_event_register_round_trip() {
        let mut ctrl = L1InterruptController::new();
        // Pack events: slot0=0x11, slot1=0x22, slot2=0x33, slot3=0x44
        let packed: u32 = 0x11 | (0x22 << 8) | (0x33 << 16) | (0x44 << 24);
        ctrl.write_irq_event(SwitchId::A, packed);

        assert_eq!(ctrl.read_irq_event(SwitchId::A), packed & 0x7F7F7F7F);
        assert_eq!(ctrl.get_irq_event_slot(SwitchId::A, 0), 0x11);
        assert_eq!(ctrl.get_irq_event_slot(SwitchId::A, 1), 0x22);
        assert_eq!(ctrl.get_irq_event_slot(SwitchId::A, 2), 0x33);
        assert_eq!(ctrl.get_irq_event_slot(SwitchId::A, 3), 0x44);
    }

    #[test]
    fn l1_irq_event_masks_to_7_bits() {
        let mut ctrl = L1InterruptController::new();
        ctrl.set_irq_event_slot(SwitchId::A, 0, 0xFF);
        // 0xFF masked to 7 bits = 0x7F
        assert_eq!(ctrl.get_irq_event_slot(SwitchId::A, 0), 0x7F);
    }

    #[test]
    fn l1_set_irq_event_slot_preserves_others() {
        let mut ctrl = L1InterruptController::new();
        ctrl.set_irq_event_slot(SwitchId::B, 0, 10);
        ctrl.set_irq_event_slot(SwitchId::B, 2, 30);
        // Setting slot 1 should not affect slot 0 or 2.
        ctrl.set_irq_event_slot(SwitchId::B, 1, 20);
        assert_eq!(ctrl.get_irq_event_slot(SwitchId::B, 0), 10);
        assert_eq!(ctrl.get_irq_event_slot(SwitchId::B, 1), 20);
        assert_eq!(ctrl.get_irq_event_slot(SwitchId::B, 2), 30);
    }

    // ---------------------------------------------------------------
    // L1: IRQ number (broadcast ID)
    // ---------------------------------------------------------------

    #[test]
    fn l1_irq_no_register() {
        let mut ctrl = L1InterruptController::new();
        ctrl.write_irq_no(SwitchId::A, 7);
        assert_eq!(ctrl.read_irq_no(SwitchId::A), 7);
    }

    #[test]
    fn l1_irq_no_clamps_to_4_bits() {
        let mut ctrl = L1InterruptController::new();
        ctrl.write_irq_no(SwitchId::B, 0xFF);
        assert_eq!(ctrl.read_irq_no(SwitchId::B), 0x0F);
    }

    // ---------------------------------------------------------------
    // L1: broadcast block north
    // ---------------------------------------------------------------

    #[test]
    fn l1_block_north_set_clear_value() {
        let mut ctrl = L1InterruptController::new();
        ctrl.write_block_north_set(SwitchId::A, 0x00FF);
        assert_eq!(ctrl.read_block_north_value(SwitchId::A), 0x00FF);

        ctrl.write_block_north_set(SwitchId::A, 0xFF00);
        assert_eq!(ctrl.read_block_north_value(SwitchId::A), 0xFFFF);

        ctrl.write_block_north_clear(SwitchId::A, 0x0F0F);
        assert_eq!(ctrl.read_block_north_value(SwitchId::A), 0xF0F0);
    }

    #[test]
    fn l1_block_north_switches_independent() {
        let mut ctrl = L1InterruptController::new();
        ctrl.write_block_north_set(SwitchId::A, 0xFFFF);
        assert_eq!(ctrl.read_block_north_value(SwitchId::B), 0);
    }

    #[test]
    fn l1_block_north_clamps_to_16_bits() {
        let mut ctrl = L1InterruptController::new();
        ctrl.write_block_north_set(SwitchId::A, 0xFFFF_FFFF);
        assert_eq!(ctrl.read_block_north_value(SwitchId::A), L1_BLOCK_VALID_MASK);
    }

    // ---------------------------------------------------------------
    // L1: register interface
    // ---------------------------------------------------------------

    #[test]
    fn l1_register_interface_enable() {
        let mut ctrl = L1InterruptController::new();
        assert!(ctrl.write_register(L1_REG_ENABLE_A, 0b1010));
        assert_eq!(ctrl.read_register(L1_REG_MASK_A), Some(0b1010));
    }

    #[test]
    fn l1_register_interface_switch_b() {
        let mut ctrl = L1InterruptController::new();
        let enable_b = L1_REG_ENABLE_A + L1_SWITCH_OFFSET;
        let mask_b = L1_REG_MASK_A + L1_SWITCH_OFFSET;
        assert!(ctrl.write_register(enable_b, 0b0101));
        assert_eq!(ctrl.read_register(mask_b), Some(0b0101));
        // Switch A unaffected.
        assert_eq!(ctrl.read_register(L1_REG_MASK_A), Some(0));
    }

    #[test]
    fn l1_register_interface_status_write_clears() {
        let mut ctrl = L1InterruptController::new();
        ctrl.set_irq_event_slot(SwitchId::A, 0, 1);
        ctrl.write_enable(SwitchId::A, 1 << 16);
        ctrl.signal_event(SwitchId::A, 1);
        assert_ne!(ctrl.read_register(L1_REG_STATUS_A), Some(0));

        ctrl.write_register(L1_REG_STATUS_A, 1 << 16);
        assert_eq!(ctrl.read_register(L1_REG_STATUS_A), Some(0));
    }

    #[test]
    fn l1_register_interface_irq_event() {
        let mut ctrl = L1InterruptController::new();
        let value = 0x0A | (0x14 << 8);
        assert!(ctrl.write_register(L1_REG_IRQ_EVENT_A, value));
        assert_eq!(ctrl.read_register(L1_REG_IRQ_EVENT_A), Some(value));
    }

    #[test]
    fn l1_register_interface_irq_no() {
        let mut ctrl = L1InterruptController::new();
        assert!(ctrl.write_register(L1_REG_IRQ_NO_A, 5));
        assert_eq!(ctrl.read_register(L1_REG_IRQ_NO_A), Some(5));
    }

    #[test]
    fn l1_register_interface_block_north() {
        let mut ctrl = L1InterruptController::new();
        assert!(ctrl.write_register(L1_REG_BLOCK_NORTH_SET_A, 0xAA));
        assert_eq!(ctrl.read_register(L1_REG_BLOCK_NORTH_VALUE_A), Some(0xAA));
        assert!(ctrl.write_register(L1_REG_BLOCK_NORTH_CLEAR_A, 0x0A));
        assert_eq!(ctrl.read_register(L1_REG_BLOCK_NORTH_VALUE_A), Some(0xA0));
    }

    #[test]
    fn l1_register_interface_unknown_offset() {
        let ctrl = L1InterruptController::new();
        assert_eq!(ctrl.read_register(0x0003_FFFF), None);
    }

    #[test]
    fn l1_register_interface_write_unknown_returns_false() {
        let mut ctrl = L1InterruptController::new();
        assert!(!ctrl.write_register(0x0003_FFFF, 1));
    }

    #[test]
    fn l1_all_switch_a_register_offsets_readable() {
        let ctrl = L1InterruptController::new();
        let offsets = [
            L1_REG_MASK_A,
            L1_REG_ENABLE_A,
            L1_REG_DISABLE_A,
            L1_REG_STATUS_A,
            L1_REG_IRQ_NO_A,
            L1_REG_IRQ_EVENT_A,
            L1_REG_BLOCK_NORTH_SET_A,
            L1_REG_BLOCK_NORTH_CLEAR_A,
            L1_REG_BLOCK_NORTH_VALUE_A,
        ];
        for off in offsets {
            assert!(ctrl.read_register(off).is_some(), "offset {:#x} not readable", off);
        }
    }

    #[test]
    fn l1_all_switch_b_register_offsets_readable() {
        let ctrl = L1InterruptController::new();
        let offsets = [
            L1_REG_MASK_A + L1_SWITCH_OFFSET,
            L1_REG_ENABLE_A + L1_SWITCH_OFFSET,
            L1_REG_DISABLE_A + L1_SWITCH_OFFSET,
            L1_REG_STATUS_A + L1_SWITCH_OFFSET,
            L1_REG_IRQ_NO_A + L1_SWITCH_OFFSET,
            L1_REG_IRQ_EVENT_A + L1_SWITCH_OFFSET,
            L1_REG_BLOCK_NORTH_SET_A + L1_SWITCH_OFFSET,
            L1_REG_BLOCK_NORTH_CLEAR_A + L1_SWITCH_OFFSET,
            L1_REG_BLOCK_NORTH_VALUE_A + L1_SWITCH_OFFSET,
        ];
        for off in offsets {
            assert!(ctrl.read_register(off).is_some(), "offset {:#x} not readable", off);
        }
    }

    // ---------------------------------------------------------------
    // L1: mask register is read-only
    // ---------------------------------------------------------------

    #[test]
    fn l1_mask_register_write_is_noop() {
        let mut ctrl = L1InterruptController::new();
        ctrl.write_enable(SwitchId::A, 0b1111);
        // Writing directly to Mask should not change the state.
        ctrl.write_register(L1_REG_MASK_A, 0);
        assert_eq!(ctrl.read_mask(SwitchId::A), 0b1111);
    }

    // ---------------------------------------------------------------
    // L1: default trait
    // ---------------------------------------------------------------

    #[test]
    fn l1_default_matches_new() {
        let a = L1InterruptController::new();
        let b = L1InterruptController::default();
        assert_eq!(a.read_mask(SwitchId::A), b.read_mask(SwitchId::A));
        assert_eq!(a.read_status(SwitchId::A), b.read_status(SwitchId::A));
    }

    // ---------------------------------------------------------------
    // L2: initial state
    // ---------------------------------------------------------------

    #[test]
    fn l2_initial_state() {
        let ctrl = L2InterruptController::new();
        assert_eq!(ctrl.read_mask(), 0);
        assert_eq!(ctrl.read_status(), 0);
        assert!(!ctrl.pending_host_interrupt());
    }

    // ---------------------------------------------------------------
    // L2: enable / disable
    // ---------------------------------------------------------------

    #[test]
    fn l2_enable_sets_mask_bits() {
        let mut ctrl = L2InterruptController::new();
        ctrl.write_enable(0b1100);
        assert_eq!(ctrl.read_mask(), 0b1100);
    }

    #[test]
    fn l2_enable_accumulates() {
        let mut ctrl = L2InterruptController::new();
        ctrl.write_enable(0b1100);
        ctrl.write_enable(0b0011);
        assert_eq!(ctrl.read_mask(), 0b1111);
    }

    #[test]
    fn l2_disable_clears_mask_bits() {
        let mut ctrl = L2InterruptController::new();
        ctrl.write_enable(0b1111);
        ctrl.write_disable(0b0101);
        assert_eq!(ctrl.read_mask(), 0b1010);
    }

    #[test]
    fn l2_enable_clamps_to_valid_mask() {
        let mut ctrl = L2InterruptController::new();
        ctrl.write_enable(0xFFFF_FFFF);
        assert_eq!(ctrl.read_mask(), L2_VALID_MASK);
    }

    // ---------------------------------------------------------------
    // L2: interrupt signaling
    // ---------------------------------------------------------------

    #[test]
    fn l2_signal_and_status() {
        let mut ctrl = L2InterruptController::new();
        ctrl.write_enable(0b0010);
        ctrl.signal_interrupt(1);
        assert_ne!(ctrl.read_status() & 0b0010, 0);
        assert!(ctrl.pending_host_interrupt());
    }

    #[test]
    fn l2_masked_signal_does_not_latch() {
        let mut ctrl = L2InterruptController::new();
        // Channel 3 is not enabled.
        ctrl.signal_interrupt(3);
        assert_eq!(ctrl.read_status(), 0);
        assert!(!ctrl.pending_host_interrupt());
    }

    #[test]
    fn l2_multiple_channels_pending() {
        let mut ctrl = L2InterruptController::new();
        ctrl.write_enable(0xFF);
        for ch in 0..8 {
            ctrl.signal_interrupt(ch);
        }
        assert_eq!(ctrl.read_status(), 0xFF);
        assert!(ctrl.pending_host_interrupt());
    }

    // ---------------------------------------------------------------
    // L2: status clear (ack)
    // ---------------------------------------------------------------

    #[test]
    fn l2_status_clear() {
        let mut ctrl = L2InterruptController::new();
        ctrl.write_enable(0b0110);
        ctrl.signal_interrupt(1);
        ctrl.signal_interrupt(2);
        assert_eq!(ctrl.read_status(), 0b0110);

        ctrl.clear_status(0b0010);
        assert_eq!(ctrl.read_status(), 0b0100);
        assert!(ctrl.pending_host_interrupt());

        ctrl.clear_status(0b0100);
        assert_eq!(ctrl.read_status(), 0);
        assert!(!ctrl.pending_host_interrupt());
    }

    // ---------------------------------------------------------------
    // L2: mask prevents host notification
    // ---------------------------------------------------------------

    #[test]
    fn l2_mask_prevents_host_notification() {
        let mut ctrl = L2InterruptController::new();
        ctrl.write_enable(1 << 5);
        ctrl.write_disable(1 << 5);

        ctrl.signal_interrupt(5);
        assert!(!ctrl.pending_host_interrupt());
    }

    // ---------------------------------------------------------------
    // L2: integration with L1
    // ---------------------------------------------------------------

    #[test]
    fn l2_aggregates_l1_outputs() {
        let mut l1 = L1InterruptController::new();
        let mut l2 = L2InterruptController::new();

        // L1: map event 7 to slot 0 (interrupt ID 16), output on broadcast 3.
        l1.set_irq_event_slot(SwitchId::A, 0, 7);
        l1.write_enable(SwitchId::A, 1 << 16);
        l1.write_irq_no(SwitchId::A, 3);

        // L2: enable channel 3 (matching L1's broadcast ID).
        l2.write_enable(1 << 3);

        // Fire event through L1 -> L2.
        if let Some(_interrupt_id) = l1.signal_event(SwitchId::A, 7) {
            let broadcast_id = l1.read_irq_no(SwitchId::A);
            l2.signal_interrupt(broadcast_id as u8);
        }

        assert!(l2.pending_host_interrupt());
        assert_ne!(l2.read_status() & (1 << 3), 0);
    }

    #[test]
    fn l2_l1_disabled_does_not_reach_l2() {
        let mut l1 = L1InterruptController::new();
        let mut l2 = L2InterruptController::new();

        // L1: map event but do NOT enable the interrupt.
        l1.set_irq_event_slot(SwitchId::A, 0, 7);
        l1.write_irq_no(SwitchId::A, 3);

        l2.write_enable(1 << 3);

        // Event fires on L1, but L1 does not latch -> no signal to L2.
        if let Some(_id) = l1.signal_event(SwitchId::A, 7) {
            l2.signal_interrupt(l1.read_irq_no(SwitchId::A) as u8);
        }

        assert!(!l2.pending_host_interrupt());
    }

    // ---------------------------------------------------------------
    // L2: channel-identity invariant
    // ---------------------------------------------------------------

    #[test]
    fn channel_identity_l1_irq_no_equals_l2_input_channel() {
        // Invariant probe: the broadcast id an L1 switch outputs (IRQ_NO) is
        // the same numeric channel L2 latches on. If a future change inserts
        // a remap, this fails loudly.
        let mut l2 = L2InterruptController::new();
        for ch in 0u8..16 {
            l2.write_enable(1 << ch);
        }
        for irq_no in 0u8..16 {
            l2.signal_interrupt(irq_no); // L1 IRQ_NO fed directly as L2 channel
            assert_ne!(
                l2.read_status() & (1 << irq_no),
                0,
                "L2 must latch the same channel index L1 output as IRQ_NO ({irq_no})"
            );
        }
    }

    // ---------------------------------------------------------------
    // L2: register interface
    // ---------------------------------------------------------------

    #[test]
    fn l2_register_interface_enable_disable() {
        let mut ctrl = L2InterruptController::new();
        assert!(ctrl.write_register(L2_REG_ENABLE, 0b1010));
        assert_eq!(ctrl.read_register(L2_REG_MASK), Some(0b1010));

        assert!(ctrl.write_register(L2_REG_DISABLE, 0b0010));
        assert_eq!(ctrl.read_register(L2_REG_MASK), Some(0b1000));
    }

    #[test]
    fn l2_register_interface_status_write_clears() {
        let mut ctrl = L2InterruptController::new();
        ctrl.write_enable(0b0001);
        ctrl.signal_interrupt(0);
        assert_ne!(ctrl.read_register(L2_REG_STATUS), Some(0));

        ctrl.write_register(L2_REG_STATUS, 0b0001);
        assert_eq!(ctrl.read_register(L2_REG_STATUS), Some(0));
    }

    #[test]
    fn l2_noc_interrupt_register_reflects_pending() {
        let mut ctrl = L2InterruptController::new();
        assert_eq!(ctrl.read_register(L2_REG_INTERRUPT), Some(0));

        ctrl.write_enable(0b0001);
        ctrl.signal_interrupt(0);
        let irq = ctrl.read_register(L2_REG_INTERRUPT).unwrap();
        assert_ne!(irq, 0, "NoC interrupt should be asserted when status pending");
    }

    #[test]
    fn l2_register_interface_unknown_offset() {
        let ctrl = L2InterruptController::new();
        assert_eq!(ctrl.read_register(0xDEAD), None);
    }

    #[test]
    fn l2_register_interface_write_unknown_returns_false() {
        let mut ctrl = L2InterruptController::new();
        assert!(!ctrl.write_register(0xDEAD, 1));
    }

    #[test]
    fn l2_all_register_offsets_readable() {
        let ctrl = L2InterruptController::new();
        let offsets = [L2_REG_MASK, L2_REG_ENABLE, L2_REG_DISABLE, L2_REG_STATUS, L2_REG_INTERRUPT];
        for off in offsets {
            assert!(ctrl.read_register(off).is_some(), "offset {:#x} not readable", off);
        }
    }

    // ---------------------------------------------------------------
    // L2: mask register is read-only
    // ---------------------------------------------------------------

    #[test]
    fn l2_mask_register_write_is_noop() {
        let mut ctrl = L2InterruptController::new();
        ctrl.write_enable(0b1111);
        ctrl.write_register(L2_REG_MASK, 0);
        assert_eq!(ctrl.read_mask(), 0b1111);
    }

    // ---------------------------------------------------------------
    // L2: default trait
    // ---------------------------------------------------------------

    #[test]
    fn l2_default_matches_new() {
        let a = L2InterruptController::new();
        let b = L2InterruptController::default();
        assert_eq!(a.read_mask(), b.read_mask());
        assert_eq!(a.read_status(), b.read_status());
    }
}
