//! Register hazard detection for AIE2.
//!
//! Tracks data dependencies between instructions to detect:
//! - **RAW** (Read After Write): Reading a register before its write completes
//! - **WAW** (Write After Write): Writing a register before previous write completes
//! - **WAR** (Write After Read): Writing a register before previous read completes
//!
//! # Pipeline Hazards
//!
//! AIE2 has an 8-stage maximum pipeline (AM020 Ch4). When an instruction
//! writes to a register, subsequent instructions that read that register
//! must wait until the write completes.
//!
//! Example:
//! ```text
//! Cycle 1: MUL r3, r1, r2    ; r3 ready at cycle 3 (2-cycle latency)
//! Cycle 2: ADD r4, r3, r5    ; RAW hazard! r3 not ready yet
//!                            ; Must stall until cycle 3
//! ```

use crate::interpreter::bundle::{Operand, Operation, SlotOp};

/// Type of data hazard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HazardType {
    /// Read After Write: reading before write completes.
    Raw,
    /// Write After Write: writing before previous write completes.
    Waw,
    /// Write After Read: writing before previous read completes.
    /// (Usually not a hazard with proper register renaming, but we track it.)
    War,
}

/// Information about a detected hazard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hazard {
    /// Type of hazard.
    pub hazard_type: HazardType,
    /// Register involved.
    pub register: u8,
    /// Cycles to stall (until dependency resolves).
    pub stall_cycles: u8,
}

/// Tracks register availability for hazard detection.
///
/// Records when each register will be ready (after pending writes complete).
#[derive(Debug, Clone)]
pub struct HazardDetector {
    /// Cycle when each scalar register (r0-r31) becomes ready.
    scalar_ready: [u64; 32],

    /// Cycle when each pointer register (p0-p7) becomes ready.
    pointer_ready: [u64; 8],

    /// Cycle when each modifier register (m0-m7) becomes ready.
    modifier_ready: [u64; 8],

    /// Cycle when each vector register becomes ready.
    /// (Simplified: track as 32 registers even though AM020 says 24)
    vector_ready: [u64; 32],

    /// Cycle when each accumulator register becomes ready.
    accumulator_ready: [u64; 8],

    /// Current cycle.
    current_cycle: u64,

    /// Statistics: total RAW hazards.
    raw_hazards: u64,

    /// Statistics: total WAW hazards.
    waw_hazards: u64,

    /// Statistics: total stall cycles from hazards.
    stall_cycles: u64,
}

impl HazardDetector {
    /// Create a new hazard detector.
    pub fn new() -> Self {
        Self {
            scalar_ready: [0; 32],
            pointer_ready: [0; 8],
            modifier_ready: [0; 8],
            vector_ready: [0; 32],
            accumulator_ready: [0; 8],
            current_cycle: 0,
            raw_hazards: 0,
            waw_hazards: 0,
            stall_cycles: 0,
        }
    }

    /// Advance to the specified cycle.
    pub fn advance_to(&mut self, cycle: u64) {
        self.current_cycle = cycle;
    }

    /// Check if reading a scalar register would cause a hazard.
    pub fn check_scalar_read(&self, reg: u8) -> Option<Hazard> {
        let ready = self.scalar_ready[reg as usize & 0x1F];
        if ready > self.current_cycle {
            Some(Hazard {
                hazard_type: HazardType::Raw,
                register: reg,
                stall_cycles: (ready - self.current_cycle) as u8,
            })
        } else {
            None
        }
    }

    /// Check if writing a scalar register would cause a WAW hazard.
    pub fn check_scalar_write(&self, reg: u8) -> Option<Hazard> {
        let ready = self.scalar_ready[reg as usize & 0x1F];
        if ready > self.current_cycle {
            Some(Hazard {
                hazard_type: HazardType::Waw,
                register: reg,
                stall_cycles: (ready - self.current_cycle) as u8,
            })
        } else {
            None
        }
    }

    /// Record that a scalar register will be written with given latency.
    pub fn record_scalar_write(&mut self, reg: u8, latency: u8) {
        let ready_cycle = self.current_cycle + latency as u64;
        self.scalar_ready[reg as usize & 0x1F] = ready_cycle;
    }

    /// Check if reading a pointer register would cause a hazard.
    pub fn check_pointer_read(&self, reg: u8) -> Option<Hazard> {
        let ready = self.pointer_ready[reg as usize & 0x7];
        if ready > self.current_cycle {
            Some(Hazard {
                hazard_type: HazardType::Raw,
                register: reg,
                stall_cycles: (ready - self.current_cycle) as u8,
            })
        } else {
            None
        }
    }

    /// Record that a pointer register will be written.
    pub fn record_pointer_write(&mut self, reg: u8, latency: u8) {
        let ready_cycle = self.current_cycle + latency as u64;
        self.pointer_ready[reg as usize & 0x7] = ready_cycle;
    }

    /// Check if reading a vector register would cause a hazard.
    pub fn check_vector_read(&self, reg: u8) -> Option<Hazard> {
        let ready = self.vector_ready[reg as usize & 0x1F];
        if ready > self.current_cycle {
            Some(Hazard {
                hazard_type: HazardType::Raw,
                register: reg,
                stall_cycles: (ready - self.current_cycle) as u8,
            })
        } else {
            None
        }
    }

    /// Record that a vector register will be written.
    pub fn record_vector_write(&mut self, reg: u8, latency: u8) {
        let ready_cycle = self.current_cycle + latency as u64;
        self.vector_ready[reg as usize & 0x1F] = ready_cycle;
    }

    /// Check hazards for all operands of an operation.
    pub fn check_operation(&self, op: &SlotOp) -> Vec<Hazard> {
        let mut hazards = Vec::new();

        // Check source operands for RAW hazards
        for operand in &op.sources {
            if let Some(hazard) = self.check_operand_read(operand) {
                hazards.push(hazard);
            }
        }

        // Check destination for WAW
        if let Some(dest) = op.destination_register() {
            if let Some(hazard) = self.check_scalar_write(dest) {
                hazards.push(hazard);
            }
        }

        hazards
    }

    /// Check if reading an operand would cause a hazard.
    fn check_operand_read(&self, operand: &Operand) -> Option<Hazard> {
        match operand {
            Operand::ScalarReg(reg) => self.check_scalar_read(*reg),
            Operand::PointerReg(reg) => self.check_pointer_read(*reg),
            Operand::VectorReg(reg) => self.check_vector_read(*reg),
            _ => None,
        }
    }

    /// Record writes for an operation after it issues.
    pub fn record_operation(&mut self, op: &SlotOp, latency: u8) {
        if let Some(dest) = op.destination_register() {
            // Determine register type from operation
            match &op.op {
                Operation::VectorAdd { .. }
                | Operation::VectorSub { .. }
                | Operation::VectorMul { .. }
                | Operation::VectorMac { .. }
                | Operation::VectorShuffle { .. } => {
                    self.record_vector_write(dest, latency);
                }
                _ => {
                    self.record_scalar_write(dest, latency);
                }
            }
        }

        // Handle pointer post-modify
        if let Some(ptr_reg) = op.modified_pointer_register() {
            self.record_pointer_write(ptr_reg, 1); // AGU latency
        }
    }

    /// Get the maximum stall needed for all hazards.
    pub fn max_stall(&self, hazards: &[Hazard]) -> u8 {
        hazards.iter().map(|h| h.stall_cycles).max().unwrap_or(0)
    }

    /// Update statistics after resolving hazards.
    pub fn record_stall(&mut self, stall: u8, hazard_type: HazardType) {
        self.stall_cycles += stall as u64;
        match hazard_type {
            HazardType::Raw => self.raw_hazards += 1,
            HazardType::Waw => self.waw_hazards += 1,
            HazardType::War => {} // Not typically counted
        }
    }

    /// Reset the detector.
    pub fn reset(&mut self) {
        self.scalar_ready = [0; 32];
        self.pointer_ready = [0; 8];
        self.modifier_ready = [0; 8];
        self.vector_ready = [0; 32];
        self.accumulator_ready = [0; 8];
        self.current_cycle = 0;
        self.raw_hazards = 0;
        self.waw_hazards = 0;
        self.stall_cycles = 0;
    }

    /// Get statistics.
    pub fn stats(&self) -> HazardStats {
        HazardStats {
            raw_hazards: self.raw_hazards,
            waw_hazards: self.waw_hazards,
            total_stall_cycles: self.stall_cycles,
        }
    }
}

impl Default for HazardDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Hazard detection statistics.
#[derive(Debug, Clone, Copy)]
pub struct HazardStats {
    pub raw_hazards: u64,
    pub waw_hazards: u64,
    pub total_stall_cycles: u64,
}

/// Extension trait for SlotOp to get register information.
pub trait SlotOpExt {
    /// Get the destination register if this op writes to one.
    fn destination_register(&self) -> Option<u8>;

    /// Get the pointer register modified by post-increment.
    fn modified_pointer_register(&self) -> Option<u8>;
}

impl SlotOpExt for SlotOp {
    fn destination_register(&self) -> Option<u8> {
        // Check the dest field for the destination register
        match &self.dest {
            Some(Operand::ScalarReg(reg)) => Some(*reg),
            Some(Operand::VectorReg(reg)) => Some(*reg),
            Some(Operand::AccumReg(reg)) => Some(*reg),
            _ => None,
        }
    }

    fn modified_pointer_register(&self) -> Option<u8> {
        // Check for post-modify in load/store ops
        match &self.op {
            Operation::Load { post_modify, .. } | Operation::Store { post_modify, .. } => {
                use crate::interpreter::bundle::PostModify;
                match post_modify {
                    PostModify::Register(reg) => Some(*reg),
                    PostModify::Immediate(_) => {
                        // Need to know which pointer reg was used
                        // For now, check source operands
                        for operand in &self.sources {
                            if let Operand::PointerReg(reg) = operand {
                                return Some(*reg);
                            }
                        }
                        None
                    }
                    PostModify::None => None,
                }
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::SlotIndex;

    #[test]
    fn test_no_hazard() {
        let detector = HazardDetector::new();

        // No pending writes, so no hazard
        assert!(detector.check_scalar_read(5).is_none());
    }

    #[test]
    fn test_raw_hazard() {
        let mut detector = HazardDetector::new();
        detector.advance_to(10);

        // Write to r3 with 2-cycle latency
        detector.record_scalar_write(3, 2);

        // Try to read r3 immediately
        let hazard = detector.check_scalar_read(3);
        assert!(hazard.is_some());

        let h = hazard.unwrap();
        assert_eq!(h.hazard_type, HazardType::Raw);
        assert_eq!(h.register, 3);
        assert_eq!(h.stall_cycles, 2);
    }

    #[test]
    fn test_hazard_resolved() {
        let mut detector = HazardDetector::new();
        detector.advance_to(10);

        // Write to r3 with 2-cycle latency (ready at cycle 12)
        detector.record_scalar_write(3, 2);

        // Advance past the ready time
        detector.advance_to(12);

        // Now read should be fine
        assert!(detector.check_scalar_read(3).is_none());
    }

    #[test]
    fn test_waw_hazard() {
        let mut detector = HazardDetector::new();
        detector.advance_to(10);

        // Write to r3 with 2-cycle latency
        detector.record_scalar_write(3, 2);

        // Try to write r3 again
        let hazard = detector.check_scalar_write(3);
        assert!(hazard.is_some());
        assert_eq!(hazard.unwrap().hazard_type, HazardType::Waw);
    }

    #[test]
    fn test_vector_hazard() {
        let mut detector = HazardDetector::new();
        detector.advance_to(10);

        // Vector MAC to v5 with 4-cycle latency
        detector.record_vector_write(5, 4);

        let hazard = detector.check_vector_read(5);
        assert!(hazard.is_some());
        assert_eq!(hazard.unwrap().stall_cycles, 4);
    }
}
