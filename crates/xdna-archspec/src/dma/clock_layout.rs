//! Clock-control register layout, resolved from the register database.
//!
//! Each tile kind has one `Module_Clock_Control` register (named
//! `Module_Clock_Control_0` on the shim) holding the two adaptive clock-gate
//! enables, plus a stream-switch adaptive-gate Abort_Period register and a
//! status register. Offsets and field positions come from the database; none
//! are written out by hand.

use crate::regdb::{BitField, ModuleDef, RegisterDb};

/// Clock-control registers of one tile kind.
#[derive(Debug, Clone)]
pub struct ClockControlLayout {
    /// `Module_Clock_Control` (shim: `Module_Clock_Control_0`) tile-local offset.
    pub module_clock_control: u32,
    /// Reset value of `Module_Clock_Control`.
    pub module_clock_control_reset: u32,
    /// `Stream_Switch_Adaptive_Clock_Gate` enable field of `Module_Clock_Control`.
    pub stream_switch_adaptive_enable: BitField,
    /// `DMA_Adaptive_Clock_Gate` enable field of `Module_Clock_Control`.
    pub dma_adaptive_enable: BitField,
    /// `Stream_Switch_Adaptive_Clock_Gate_Abort_Period` offset.
    pub abort_period_register: u32,
    /// `Abort_Period` field (the idle threshold is 2^value cycles).
    pub abort_period: BitField,
    /// Reset value of the Abort_Period register.
    pub abort_period_reset: u32,
    /// `Stream_Switch_Adaptive_Clock_Gate_Status` offset.
    pub status_register: u32,
    /// `Active` field of the status register (1 = gate engaged).
    pub status_active: BitField,
}

impl ClockControlLayout {
    pub(super) fn from_regdb(db: &RegisterDb, module: &str, mcc_name: &str) -> Result<Self, String> {
        let m: &ModuleDef = db.module(module).ok_or_else(|| format!("Module '{}' not found", module))?;
        let reg = |name: &str| m.register(name).ok_or_else(|| format!("{}.{} not found", module, name));
        let field = |reg_name: &str, field_name: &str| -> Result<BitField, String> {
            reg(reg_name)?
                .field(field_name)
                .cloned()
                .ok_or_else(|| format!("{}.{}.{} not found", module, reg_name, field_name))
        };
        const ABORT: &str = "Stream_Switch_Adaptive_Clock_Gate_Abort_Period";
        const STATUS: &str = "Stream_Switch_Adaptive_Clock_Gate_Status";
        let mcc = reg(mcc_name)?;
        Ok(Self {
            module_clock_control: mcc.offset,
            module_clock_control_reset: mcc.reset_value,
            stream_switch_adaptive_enable: field(mcc_name, "Stream_Switch_Adaptive_Clock_Gate")?,
            dma_adaptive_enable: field(mcc_name, "DMA_Adaptive_Clock_Gate")?,
            abort_period_register: reg(ABORT)?.offset,
            abort_period: field(ABORT, "Abort_Period")?,
            abort_period_reset: reg(ABORT)?.reset_value,
            status_register: reg(STATUS)?.offset,
            status_active: field(STATUS, "Active")?,
        })
    }
}
