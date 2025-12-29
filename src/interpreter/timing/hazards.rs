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

/// Detailed reason for a pipeline stall.
///
/// Used for profiling and debugging to understand where cycles are lost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StallReason {
    /// Register data hazard (RAW, WAW, or WAR).
    RegisterHazard {
        hazard_type: HazardType,
        register: u8,
        cycles: u8,
    },
    /// Memory bank conflict.
    MemoryConflict {
        bank: u8,
        cycles: u8,
    },
    /// Branch taken penalty (pipeline flush).
    BranchPenalty {
        cycles: u8,
    },
    /// Structural hazard (resource conflict between slots).
    StructuralHazard {
        resource: &'static str,
        cycles: u8,
    },
    /// Lock contention (waiting to acquire lock).
    LockContention {
        lock_id: u8,
        cycles: u64,
    },
    /// DMA wait (waiting for transfer to complete).
    DmaWait {
        channel: u8,
        cycles: u64,
    },
}

impl StallReason {
    /// Get the number of stall cycles for this reason.
    pub fn cycles(&self) -> u64 {
        match self {
            StallReason::RegisterHazard { cycles, .. } => *cycles as u64,
            StallReason::MemoryConflict { cycles, .. } => *cycles as u64,
            StallReason::BranchPenalty { cycles } => *cycles as u64,
            StallReason::StructuralHazard { cycles, .. } => *cycles as u64,
            StallReason::LockContention { cycles, .. } => *cycles,
            StallReason::DmaWait { cycles, .. } => *cycles,
        }
    }

    /// Check if this is a short stall (pipeline hazard) vs long stall (lock/DMA).
    pub fn is_pipeline_stall(&self) -> bool {
        matches!(
            self,
            StallReason::RegisterHazard { .. }
                | StallReason::MemoryConflict { .. }
                | StallReason::BranchPenalty { .. }
                | StallReason::StructuralHazard { .. }
        )
    }
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

impl Hazard {
    /// Convert this hazard to a StallReason for tracking.
    pub fn to_stall_reason(&self) -> StallReason {
        StallReason::RegisterHazard {
            hazard_type: self.hazard_type,
            register: self.register,
            cycles: self.stall_cycles,
        }
    }
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
    ///
    /// Note: HazardDetector only tracks register hazards. Memory conflicts,
    /// branch penalties, and structural hazards are tracked elsewhere.
    pub fn stats(&self) -> HazardStats {
        HazardStats {
            raw_hazards: self.raw_hazards,
            waw_hazards: self.waw_hazards,
            register_stall_cycles: self.stall_cycles,
            memory_stall_cycles: 0,
            branch_stall_cycles: 0,
            structural_stall_cycles: 0,
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
#[derive(Debug, Clone, Copy, Default)]
pub struct HazardStats {
    /// Count of RAW hazards detected.
    pub raw_hazards: u64,
    /// Count of WAW hazards detected.
    pub waw_hazards: u64,
    /// Total stall cycles from register hazards.
    pub register_stall_cycles: u64,
    /// Total stall cycles from memory conflicts.
    pub memory_stall_cycles: u64,
    /// Total stall cycles from branch penalties.
    pub branch_stall_cycles: u64,
    /// Total stall cycles from structural hazards.
    pub structural_stall_cycles: u64,
    /// Total stall cycles (sum of all above).
    pub total_stall_cycles: u64,
}

impl HazardStats {
    /// Record a stall with its reason.
    ///
    /// Pipeline stalls (register, memory, branch, structural) contribute to
    /// `total_stall_cycles`. Lock/DMA waits are external synchronization and
    /// are tracked separately (not counted in pipeline stall totals).
    pub fn record(&mut self, reason: &StallReason) {
        let cycles = reason.cycles();

        match reason {
            StallReason::RegisterHazard { hazard_type, .. } => {
                self.register_stall_cycles += cycles;
                self.total_stall_cycles += cycles;
                match hazard_type {
                    HazardType::Raw => self.raw_hazards += 1,
                    HazardType::Waw => self.waw_hazards += 1,
                    HazardType::War => {}
                }
            }
            StallReason::MemoryConflict { .. } => {
                self.memory_stall_cycles += cycles;
                self.total_stall_cycles += cycles;
            }
            StallReason::BranchPenalty { .. } => {
                self.branch_stall_cycles += cycles;
                self.total_stall_cycles += cycles;
            }
            StallReason::StructuralHazard { .. } => {
                self.structural_stall_cycles += cycles;
                self.total_stall_cycles += cycles;
            }
            StallReason::LockContention { .. } | StallReason::DmaWait { .. } => {
                // External synchronization - tracked separately, not in pipeline totals
            }
        }
    }

    /// Merge another stats instance into this one.
    pub fn merge(&mut self, other: &HazardStats) {
        self.raw_hazards += other.raw_hazards;
        self.waw_hazards += other.waw_hazards;
        self.register_stall_cycles += other.register_stall_cycles;
        self.memory_stall_cycles += other.memory_stall_cycles;
        self.branch_stall_cycles += other.branch_stall_cycles;
        self.structural_stall_cycles += other.structural_stall_cycles;
        self.total_stall_cycles += other.total_stall_cycles;
    }
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

    // --- StallReason tests ---

    #[test]
    fn test_stall_reason_cycles() {
        // Pipeline stalls (u8 cycles)
        let reg_hazard = StallReason::RegisterHazard {
            hazard_type: HazardType::Raw,
            register: 5,
            cycles: 3,
        };
        assert_eq!(reg_hazard.cycles(), 3);

        let mem_conflict = StallReason::MemoryConflict { bank: 2, cycles: 4 };
        assert_eq!(mem_conflict.cycles(), 4);

        let branch = StallReason::BranchPenalty { cycles: 7 };
        assert_eq!(branch.cycles(), 7);

        let structural = StallReason::StructuralHazard {
            resource: "VectorUnit",
            cycles: 2,
        };
        assert_eq!(structural.cycles(), 2);

        // Long stalls (u64 cycles)
        let lock = StallReason::LockContention {
            lock_id: 3,
            cycles: 1000,
        };
        assert_eq!(lock.cycles(), 1000);

        let dma = StallReason::DmaWait {
            channel: 1,
            cycles: 5000,
        };
        assert_eq!(dma.cycles(), 5000);
    }

    #[test]
    fn test_stall_reason_is_pipeline_stall() {
        // Pipeline stalls
        assert!(StallReason::RegisterHazard {
            hazard_type: HazardType::Raw,
            register: 0,
            cycles: 1
        }
        .is_pipeline_stall());
        assert!(StallReason::MemoryConflict { bank: 0, cycles: 1 }.is_pipeline_stall());
        assert!(StallReason::BranchPenalty { cycles: 3 }.is_pipeline_stall());
        assert!(StallReason::StructuralHazard {
            resource: "ALU",
            cycles: 1
        }
        .is_pipeline_stall());

        // Non-pipeline stalls (external waits)
        assert!(!StallReason::LockContention {
            lock_id: 0,
            cycles: 100
        }
        .is_pipeline_stall());
        assert!(!StallReason::DmaWait {
            channel: 0,
            cycles: 100
        }
        .is_pipeline_stall());
    }

    #[test]
    fn test_hazard_to_stall_reason() {
        let hazard = Hazard {
            hazard_type: HazardType::Raw,
            register: 7,
            stall_cycles: 3,
        };

        let reason = hazard.to_stall_reason();
        match reason {
            StallReason::RegisterHazard {
                hazard_type,
                register,
                cycles,
            } => {
                assert_eq!(hazard_type, HazardType::Raw);
                assert_eq!(register, 7);
                assert_eq!(cycles, 3);
            }
            _ => panic!("Expected RegisterHazard"),
        }
    }

    // --- HazardStats tests ---

    #[test]
    fn test_hazard_stats_record_register_hazards() {
        let mut stats = HazardStats::default();

        // Record RAW hazard
        stats.record(&StallReason::RegisterHazard {
            hazard_type: HazardType::Raw,
            register: 5,
            cycles: 2,
        });
        assert_eq!(stats.raw_hazards, 1);
        assert_eq!(stats.register_stall_cycles, 2);
        assert_eq!(stats.total_stall_cycles, 2);

        // Record WAW hazard
        stats.record(&StallReason::RegisterHazard {
            hazard_type: HazardType::Waw,
            register: 3,
            cycles: 3,
        });
        assert_eq!(stats.raw_hazards, 1);
        assert_eq!(stats.waw_hazards, 1);
        assert_eq!(stats.register_stall_cycles, 5);
        assert_eq!(stats.total_stall_cycles, 5);
    }

    #[test]
    fn test_hazard_stats_record_all_types() {
        let mut stats = HazardStats::default();

        stats.record(&StallReason::RegisterHazard {
            hazard_type: HazardType::Raw,
            register: 0,
            cycles: 2,
        });
        stats.record(&StallReason::MemoryConflict { bank: 1, cycles: 3 });
        stats.record(&StallReason::BranchPenalty { cycles: 7 });
        stats.record(&StallReason::StructuralHazard {
            resource: "VectorUnit",
            cycles: 4,
        });

        assert_eq!(stats.register_stall_cycles, 2);
        assert_eq!(stats.memory_stall_cycles, 3);
        assert_eq!(stats.branch_stall_cycles, 7);
        assert_eq!(stats.structural_stall_cycles, 4);
        assert_eq!(stats.total_stall_cycles, 16);
    }

    #[test]
    fn test_hazard_stats_lock_dma_not_totaled() {
        // Lock/DMA waits are external and not counted in pipeline stall total
        let mut stats = HazardStats::default();

        stats.record(&StallReason::LockContention {
            lock_id: 5,
            cycles: 100,
        });
        stats.record(&StallReason::DmaWait {
            channel: 0,
            cycles: 200,
        });

        // These should NOT affect pipeline stall counts
        assert_eq!(stats.register_stall_cycles, 0);
        assert_eq!(stats.memory_stall_cycles, 0);
        // But they are still counted in total (for overall cycle accounting)
        // Actually, looking at the code, these are NOT added to total_stall_cycles
        // That's intentional - they're tracked separately
        assert_eq!(stats.total_stall_cycles, 0);
    }

    #[test]
    fn test_hazard_stats_merge() {
        let mut stats1 = HazardStats::default();
        stats1.raw_hazards = 5;
        stats1.waw_hazards = 2;
        stats1.register_stall_cycles = 10;
        stats1.memory_stall_cycles = 5;
        stats1.total_stall_cycles = 15;

        let stats2 = HazardStats {
            raw_hazards: 3,
            waw_hazards: 1,
            register_stall_cycles: 6,
            memory_stall_cycles: 4,
            branch_stall_cycles: 7,
            structural_stall_cycles: 2,
            total_stall_cycles: 19,
        };

        stats1.merge(&stats2);

        assert_eq!(stats1.raw_hazards, 8);
        assert_eq!(stats1.waw_hazards, 3);
        assert_eq!(stats1.register_stall_cycles, 16);
        assert_eq!(stats1.memory_stall_cycles, 9);
        assert_eq!(stats1.branch_stall_cycles, 7);
        assert_eq!(stats1.structural_stall_cycles, 2);
        assert_eq!(stats1.total_stall_cycles, 34);
    }
}
