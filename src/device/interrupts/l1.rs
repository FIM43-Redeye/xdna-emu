//! Level 1 (L1) interrupt controller emulation.
//!
//! The L1 interrupt controller lives in each shim tile's PL module stream
//! switch. It has two independent switches (A and B) with identical register
//! layouts offset by 0x30 bytes.
//!
//! # Event-to-Interrupt Flow
//!
//! 1. Software configures IRQ event slots (4 per switch) via the IRQ_EVENT
//!    register, mapping physical event IDs to slot indices.
//! 2. When an event fires, the controller checks all slots. If the event
//!    matches a slot, the corresponding interrupt ID (slot_index + 16) is
//!    looked up in the enable mask.
//! 3. If enabled, the interrupt ID bit latches in the status register and
//!    the controller outputs an interrupt on the configured broadcast line
//!    (IRQ_NO register).
//!
//! # Broadcast Blocking
//!
//! The L1 controller can block broadcast signals arriving from the north
//! (from the AIE array) after they have been captured. This prevents
//! pollution of the broadcast network. Block state is managed via
//! set/clear/value registers.
//!
//! # Register Behavior (per aie-rt)
//!
//! - **Enable**: Write-to-set. Writing 1 to bit N sets bit N in the mask.
//! - **Disable**: Write-to-clear. Writing 1 to bit N clears bit N in the mask.
//! - **Mask**: Read-only reflection of the current enable state.
//! - **Status**: Read returns latched state. Write-to-clear (ack): writing 1
//!   to bit N clears that bit.
//! - **IRQ_EVENT**: Packed register, 4 slots at 7 bits each with 8-bit stride.
//! - **IRQ_NO**: 4-bit broadcast ID for the interrupt output.
//! - **Block_North_Set/Clear**: Write-to-set/clear bits in Block_North_Value.

use super::{
    L1_BLOCK_VALID_MASK, L1_IRQ_EVENT_FIELD_MASK, L1_IRQ_EVENT_FIELD_STRIDE,
    L1_NUM_IRQ_EVENTS, L1_REG_BLOCK_NORTH_VALUE_A, L1_REG_MASK_A,
    L1_SWITCH_OFFSET, L1_VALID_MASK,
};

/// Identifies which switch (A or B) within the L1 interrupt controller.
///
/// Each shim tile has two independent L1 switches. Switch B's registers
/// are at Switch A's offsets plus `L1_SWITCH_OFFSET` (0x30).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchId {
    A = 0,
    B = 1,
}

/// Per-switch state within the L1 interrupt controller.
#[derive(Debug, Clone)]
struct SwitchState {
    /// Enable mask (20 bits). Reflects which interrupt IDs are enabled.
    /// Modified by writes to Enable (set bits) and Disable (clear bits).
    mask: u32,
    /// Latched interrupt status (20 bits). Bits set when enabled interrupts
    /// fire. Cleared by writing 1s to the status register (ack).
    status: u32,
    /// Broadcast ID for the interrupt output (4 bits).
    /// Per aie-rt: `BaseIrqRegOff`, written via `XAie_IntrCtrlL1IrqSet`.
    irq_no: u32,
    /// Packed IRQ event mapping register.
    /// 4 event slots, 7 bits each at 8-bit stride.
    /// Slot N holds the physical event ID that triggers interrupt ID (N + 16).
    irq_event: u32,
    /// Broadcast block north value (16 bits).
    /// Modified by writes to Block_North_Set (set bits) and
    /// Block_North_Clear (clear bits).
    block_north_value: u32,
}

impl Default for SwitchState {
    fn default() -> Self {
        Self {
            mask: 0,
            status: 0,
            irq_no: 0,
            irq_event: 0,
            block_north_value: 0,
        }
    }
}

/// Level 1 interrupt controller for a shim tile.
///
/// Contains two independent switches (A and B). Each switch has its own
/// enable mask, status, IRQ event mapping, broadcast ID, and north-blocking
/// state. All register offsets and field widths are derived from aie-rt
/// `xaiemlgbl_params.h` and `xaiemlgbl_reginit.c`.
#[derive(Debug, Clone)]
pub struct L1InterruptController {
    switches: [SwitchState; 2],
}

impl L1InterruptController {
    /// Create a new L1 interrupt controller with all state zeroed.
    ///
    /// Both switches start disabled (mask=0, status=0).
    pub fn new() -> Self {
        Self {
            switches: [SwitchState::default(), SwitchState::default()],
        }
    }

    // -- Mask (enable state) --

    /// Read the current enable mask for a switch.
    ///
    /// This is the read-only Mask register -- it reflects the cumulative
    /// effect of Enable and Disable writes.
    pub fn read_mask(&self, sw: SwitchId) -> u32 {
        self.switches[sw as usize].mask
    }

    /// Write to the Enable register: sets bits in the mask.
    ///
    /// Per aie-rt `_XAie_IntrCtrlL1Config` with `XAIE_ENABLE`:
    /// writes `(1 << IntrId)` to the enable register. The hardware ORs
    /// the written value into the mask.
    pub fn write_enable(&mut self, sw: SwitchId, value: u32) {
        let s = &mut self.switches[sw as usize];
        s.mask |= value & L1_VALID_MASK;
    }

    /// Write to the Disable register: clears bits in the mask.
    ///
    /// Per aie-rt `_XAie_IntrCtrlL1Config` with `XAIE_DISABLE`:
    /// writes `(1 << IntrId)` to the disable register. The hardware
    /// clears those bits from the mask.
    pub fn write_disable(&mut self, sw: SwitchId, value: u32) {
        let s = &mut self.switches[sw as usize];
        s.mask &= !(value & L1_VALID_MASK);
    }

    // -- Status --

    /// Read the latched interrupt status for a switch.
    ///
    /// Per aie-rt `XAie_IntrCtrlL1Status`: reads the Status register.
    pub fn read_status(&self, sw: SwitchId) -> u32 {
        self.switches[sw as usize].status
    }

    /// Clear (acknowledge) status bits by writing a mask.
    ///
    /// Per aie-rt `XAie_IntrCtrlL1Ack`: writes `ChannelBitMap` to the
    /// Status register, which clears the corresponding bits.
    pub fn clear_status(&mut self, sw: SwitchId, mask: u32) {
        let s = &mut self.switches[sw as usize];
        s.status &= !(mask & L1_VALID_MASK);
    }

    // -- IRQ Number (broadcast ID) --

    /// Read the broadcast ID for the interrupt output.
    ///
    /// Per aie-rt `XAie_IntrCtrlL1IrqSet`: 4-bit field.
    pub fn read_irq_no(&self, sw: SwitchId) -> u32 {
        self.switches[sw as usize].irq_no
    }

    /// Write the broadcast ID for the interrupt output.
    ///
    /// Per aie-rt: 4-bit field (mask 0xF).
    pub fn write_irq_no(&mut self, sw: SwitchId, value: u32) {
        self.switches[sw as usize].irq_no = value & 0x0F;
    }

    // -- IRQ Event Mapping --

    /// Read the packed IRQ event register.
    ///
    /// Returns 4 event IDs packed at 7 bits each with 8-bit stride:
    /// `[6:0]=slot0, [14:8]=slot1, [22:16]=slot2, [30:24]=slot3`.
    pub fn read_irq_event(&self, sw: SwitchId) -> u32 {
        self.switches[sw as usize].irq_event
    }

    /// Write the packed IRQ event register.
    ///
    /// Per aie-rt `XAie_IntrCtrlL1Event`: uses `XAie_MaskWrite32` to update
    /// individual slot fields. We store the full register value, masked to
    /// valid bits (7 bits per slot = 0x7F7F7F7F).
    pub fn write_irq_event(&mut self, sw: SwitchId, value: u32) {
        self.switches[sw as usize].irq_event = value & 0x7F7F_7F7F;
    }

    /// Set a single IRQ event slot.
    ///
    /// `slot` is 0..3, `event_id` is the 7-bit physical event ID.
    /// Per aie-rt `XAie_IntrCtrlL1Event`: slot N is at bit position
    /// `N * IrqEventOff` (IrqEventOff=8), field width 7.
    pub fn set_irq_event_slot(&mut self, sw: SwitchId, slot: u8, event_id: u8) {
        assert!(
            slot < L1_NUM_IRQ_EVENTS,
            "IRQ event slot {} out of range (max {})",
            slot,
            L1_NUM_IRQ_EVENTS - 1
        );
        let s = &mut self.switches[sw as usize];
        let shift = slot as u32 * L1_IRQ_EVENT_FIELD_STRIDE as u32;
        let field_mask = L1_IRQ_EVENT_FIELD_MASK << shift;
        s.irq_event = (s.irq_event & !field_mask)
            | (((event_id as u32) & L1_IRQ_EVENT_FIELD_MASK) << shift);
    }

    /// Read a single IRQ event slot.
    ///
    /// Returns the 7-bit physical event ID mapped to `slot` (0..3).
    pub fn get_irq_event_slot(&self, sw: SwitchId, slot: u8) -> u8 {
        assert!(
            slot < L1_NUM_IRQ_EVENTS,
            "IRQ event slot {} out of range (max {})",
            slot,
            L1_NUM_IRQ_EVENTS - 1
        );
        let shift = slot as u32 * L1_IRQ_EVENT_FIELD_STRIDE as u32;
        ((self.switches[sw as usize].irq_event >> shift) & L1_IRQ_EVENT_FIELD_MASK) as u8
    }

    // -- Broadcast Blocking --

    /// Write to Block_North_Set: sets bits in block_north_value.
    pub fn write_block_north_set(&mut self, sw: SwitchId, value: u32) {
        let s = &mut self.switches[sw as usize];
        s.block_north_value |= value & L1_BLOCK_VALID_MASK;
    }

    /// Write to Block_North_Clear: clears bits in block_north_value.
    pub fn write_block_north_clear(&mut self, sw: SwitchId, value: u32) {
        let s = &mut self.switches[sw as usize];
        s.block_north_value &= !(value & L1_BLOCK_VALID_MASK);
    }

    /// Read the current Block_North_Value.
    pub fn read_block_north_value(&self, sw: SwitchId) -> u32 {
        self.switches[sw as usize].block_north_value
    }

    // -- Event Signaling --

    /// Signal that a physical event has fired on a switch.
    ///
    /// Checks all 4 IRQ event slots. If the event matches a slot AND the
    /// corresponding interrupt ID (slot_index + 16) is enabled in the mask,
    /// the interrupt latches in the status register.
    ///
    /// Returns `Some(interrupt_id)` if the interrupt was enabled and latched,
    /// or `None` if the event was not mapped or the interrupt was disabled.
    ///
    /// The interrupt ID for slot N is N + 16, per aie-rt documentation:
    /// "Value 0 causes IRQ16, value 1 causes IRQ17, and so on."
    pub fn signal_event(&mut self, sw: SwitchId, event_id: u8) -> Option<u8> {
        let s = &mut self.switches[sw as usize];
        for slot in 0..L1_NUM_IRQ_EVENTS {
            let shift = slot as u32 * L1_IRQ_EVENT_FIELD_STRIDE as u32;
            let mapped_event =
                ((s.irq_event >> shift) & L1_IRQ_EVENT_FIELD_MASK) as u8;
            if mapped_event == event_id {
                let interrupt_id = slot + 16;
                let bit = 1u32 << interrupt_id;
                if s.mask & bit != 0 {
                    s.status |= bit;
                    return Some(interrupt_id);
                }
            }
        }
        None
    }

    // -- Register Interface --

    /// Resolve a raw register offset to a (SwitchId, local_offset) pair.
    ///
    /// Switch A registers start at `L1_REG_MASK_A` (0x35000).
    /// Switch B registers are at the same local offsets + `L1_SWITCH_OFFSET`.
    fn resolve_switch(offset: u32) -> Option<(SwitchId, u32)> {
        let base_a = L1_REG_MASK_A;
        let base_b = L1_REG_MASK_A + L1_SWITCH_OFFSET;
        let end_a = L1_REG_BLOCK_NORTH_VALUE_A + 4;
        let end_b = end_a + L1_SWITCH_OFFSET;

        if offset >= base_a && offset < end_a {
            Some((SwitchId::A, offset - base_a))
        } else if offset >= base_b && offset < end_b {
            Some((SwitchId::B, offset - base_b))
        } else {
            None
        }
    }

    /// Read a register by raw offset within the PL module address space.
    ///
    /// Returns `None` if the offset does not correspond to an L1 interrupt
    /// controller register.
    pub fn read_register(&self, offset: u32) -> Option<u32> {
        let (sw, local) = Self::resolve_switch(offset)?;
        match local {
            0x00 => Some(self.read_mask(sw)),                  // Mask
            0x04 => Some(self.read_mask(sw)),                  // Enable (reads as mask)
            0x08 => Some(0),                                   // Disable (write-only)
            0x0C => Some(self.read_status(sw)),                // Status
            0x10 => Some(self.read_irq_no(sw)),                // IRQ_NO
            0x14 => Some(self.read_irq_event(sw)),             // IRQ_EVENT
            0x18 => Some(self.read_block_north_value(sw)),     // Block_Set (reads value)
            0x1C => Some(0),                                   // Block_Clear (write-only)
            0x20 => Some(self.read_block_north_value(sw)),     // Block_Value
            _ => None,
        }
    }

    /// Write a register by raw offset within the PL module address space.
    ///
    /// Returns `true` if the offset was recognized and the write was applied,
    /// `false` if the offset does not correspond to an L1 register.
    pub fn write_register(&mut self, offset: u32, value: u32) -> bool {
        let Some((sw, local)) = Self::resolve_switch(offset) else {
            return false;
        };
        match local {
            0x00 => { /* Mask is read-only */ }
            0x04 => self.write_enable(sw, value),
            0x08 => self.write_disable(sw, value),
            0x0C => self.clear_status(sw, value),              // Write-to-clear (ack)
            0x10 => self.write_irq_no(sw, value),
            0x14 => self.write_irq_event(sw, value),
            0x18 => self.write_block_north_set(sw, value),
            0x1C => self.write_block_north_clear(sw, value),
            0x20 => { /* Value is read-only */ }
            _ => return false,
        }
        true
    }
}

impl Default for L1InterruptController {
    fn default() -> Self {
        Self::new()
    }
}
