//! Level 2 (L2) interrupt controller emulation.
//!
//! The L2 interrupt controller is located in shim NoC tiles only. It
//! aggregates L1 interrupt outputs from the column via the broadcast
//! network and routes them to up to 4 NoC/host interrupt output lines.
//!
//! # Register Behavior (per aie-rt)
//!
//! - **Enable**: Write-to-set. Writing 1 to bit N sets bit N in the mask.
//! - **Disable**: Write-to-clear. Writing 1 to bit N clears bit N in mask.
//! - **Mask**: Read-only reflection of current enable state (16 bits).
//! - **Status**: Read returns latched state. Write-to-clear (ack): writing 1
//!   to bit N clears that bit. Per aie-rt `XAie_IntrCtrlL2Ack` and
//!   `_XAie_LIntrCtrlL2Ack`, writing to status clears bits.
//! - **Interrupt**: 2-bit NoC interrupt output register. Reflects whether
//!   any unmasked interrupts are pending.
//!
//! # Channel Model
//!
//! The L2 controller has 16 broadcast input channels (matching the L1's
//! 16 broadcast IDs). Each channel can be independently enabled/disabled.
//! When an L1 interrupt fires, it drives a broadcast line. If the
//! corresponding L2 channel is enabled, the status bit latches and the
//! NoC interrupt output is asserted.
//!
//! Per aie-rt `xaiemlgbl_reginit.c`:
//! - `NumBroadcastIds = 16`
//! - `NumNoCIntr = 4`

use super::{
    L2_REG_DISABLE, L2_REG_ENABLE, L2_REG_INTERRUPT, L2_REG_MASK, L2_REG_STATUS, L2_VALID_MASK,
    L2_NOC_INTERRUPT_MASK,
};

/// Level 2 interrupt controller for a shim NoC tile.
///
/// Aggregates L1 interrupt outputs (via broadcast channels) and routes
/// to NoC/host interrupt lines. All register offsets and field widths
/// are derived from aie-rt `xaiemlgbl_params.h` and `xaiemlgbl_reginit.c`.
#[derive(Debug, Clone)]
pub struct L2InterruptController {
    /// Enable mask (16 bits). Reflects which broadcast channels are enabled.
    /// Modified by writes to Enable (set bits) and Disable (clear bits).
    mask: u32,
    /// Latched interrupt status (16 bits). Bits set when enabled channels
    /// receive an interrupt. Cleared by writing 1s to the status register.
    status: u32,
}

impl L2InterruptController {
    /// Create a new L2 interrupt controller with all state zeroed.
    ///
    /// All channels start disabled (mask=0, status=0).
    pub fn new() -> Self {
        Self { mask: 0, status: 0 }
    }

    // -- Mask (enable state) --

    /// Read the current enable mask.
    ///
    /// Per aie-rt `XAie_IntrCtrlL2Mask`: reads the Mask register at
    /// `MaskRegOff` (0x15000).
    pub fn read_mask(&self) -> u32 {
        self.mask
    }

    /// Write to the Enable register: sets bits in the mask.
    ///
    /// Per aie-rt `_XAie_IntrCtrlL2Config` with `XAIE_ENABLE`:
    /// writes the bitmap to the enable register. The hardware ORs
    /// the written value into the mask.
    pub fn write_enable(&mut self, value: u32) {
        self.mask |= value & L2_VALID_MASK;
    }

    /// Write to the Disable register: clears bits in the mask.
    ///
    /// Per aie-rt `_XAie_IntrCtrlL2Config` with `XAIE_DISABLE`:
    /// writes the bitmap to the disable register. The hardware clears
    /// those bits from the mask.
    pub fn write_disable(&mut self, value: u32) {
        self.mask &= !(value & L2_VALID_MASK);
    }

    // -- Status --

    /// Read the latched interrupt status.
    ///
    /// Per aie-rt `XAie_IntrCtrlL2Status`: reads the Status register at
    /// `StatusRegOff` (0x1500C).
    pub fn read_status(&self) -> u32 {
        self.status
    }

    /// Clear (acknowledge) status bits by writing a mask.
    ///
    /// Per aie-rt `XAie_IntrCtrlL2Ack` and `_XAie_LIntrCtrlL2Ack`:
    /// writes to the Status register, which clears the corresponding bits.
    pub fn clear_status(&mut self, mask: u32) {
        self.status &= !(mask & L2_VALID_MASK);
    }

    // -- Interrupt Signaling --

    /// Signal an interrupt on a broadcast channel.
    ///
    /// If the channel is enabled (bit set in mask), the corresponding
    /// status bit latches. If the channel is disabled, the signal is
    /// ignored.
    ///
    /// `channel` is 0..15 (the broadcast ID from the L1 controller's
    /// IRQ_NO register).
    pub fn signal_interrupt(&mut self, channel: u8) {
        let bit = 1u32 << (channel as u32);
        if self.mask & bit != 0 {
            self.status |= bit;
        }
    }

    /// Check whether any unmasked interrupt is pending.
    ///
    /// Returns `true` if any bit is set in the status register,
    /// indicating at least one enabled channel has a latched interrupt
    /// that has not been acknowledged.
    pub fn pending_host_interrupt(&self) -> bool {
        self.status != 0
    }

    // -- Register Interface --

    /// Read a register by raw offset within the NoC module address space.
    ///
    /// Returns `None` if the offset does not correspond to an L2 interrupt
    /// controller register.
    pub fn read_register(&self, offset: u32) -> Option<u32> {
        match offset {
            o if o == L2_REG_MASK => Some(self.read_mask()),
            o if o == L2_REG_ENABLE => Some(self.read_mask()), // Enable reads as mask
            o if o == L2_REG_DISABLE => Some(0),               // Disable is write-only
            o if o == L2_REG_STATUS => Some(self.read_status()),
            o if o == L2_REG_INTERRUPT => {
                // NoC interrupt output: bit 0 indicates any pending interrupt.
                // Per aie-rt, this is a 2-bit field reflecting pending state.
                let pending = if self.pending_host_interrupt() { 1u32 } else { 0u32 };
                Some(pending & L2_NOC_INTERRUPT_MASK)
            }
            _ => None,
        }
    }

    /// Write a register by raw offset within the NoC module address space.
    ///
    /// Returns `true` if the offset was recognized and the write was applied,
    /// `false` if the offset does not correspond to an L2 register.
    pub fn write_register(&mut self, offset: u32, value: u32) -> bool {
        match offset {
            o if o == L2_REG_MASK => { /* Mask is read-only */ }
            o if o == L2_REG_ENABLE => self.write_enable(value),
            o if o == L2_REG_DISABLE => self.write_disable(value),
            o if o == L2_REG_STATUS => self.clear_status(value), // Write-to-clear (ack)
            o if o == L2_REG_INTERRUPT => {
                // NoC interrupt routing register. On hardware this is the
                // single privileged L2 register (aie-rt _XAie_PrivilegeSetL2IrqId).
                // Privilege is a driver-side concern; per the project policy
                // applied to noc/shim_mux, the emulator gives unrestricted
                // access and does not model privilege gating. Output state is
                // derived, so the write is accepted and ignored. Scoped out by
                // design (spec 2026-05-19-interrupt-l2-closeout, Tier A).
            }
            _ => return false,
        }
        true
    }
}

impl Default for L2InterruptController {
    fn default() -> Self {
        Self::new()
    }
}
