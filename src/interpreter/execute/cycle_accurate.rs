//! Cycle-accurate executor implementation.
//!
//! The `CycleAccurateExecutor` models pipeline stages, register hazards, and
//! memory bank conflicts to provide accurate cycle counts. This executor
//! provides the accurate timing behavior that matches real hardware.
//!
//! # Timing Model
//!
//! Based on AM020 (AIE-ML Architecture Manual):
//! - Scalar add/sub/logic: 1 cycle
//! - Scalar multiply: 2 cycles
//! - Memory access: 5 cycles
//! - Vector MAC: 4 cycles
//! - Branch taken: 3 cycles
//!
//! # Hazard Detection
//!
//! - **RAW** (Read After Write): Reading a register before its write completes
//! - **WAW** (Write After Write): Writing a register before previous write completes
//!
//! Stalls are inserted automatically to resolve hazards.
//!
//! # Memory Bank Conflicts
//!
//! The AIE2 has 8 memory banks. Accessing the same bank from multiple ports
//! in the same cycle causes a stall.

use crate::device::aie2_spec::BRANCH_PENALTY_CYCLES;
use crate::device::tile::Tile;
use crate::interpreter::bundle::{Operation, SlotOp, VliwBundle};
use crate::interpreter::state::{EventType, ExecutionContext};
use crate::interpreter::timing::{
    check_bundle_conflicts, HazardDetector, HazardStats, LatencyTable, MemoryAccess, MemoryModel,
    StallReason,
};
use crate::interpreter::traits::{ExecuteResult, Executor};

use super::control::ControlUnit;
use super::memory::MemoryUnit;
use super::scalar::ScalarAlu;
use super::vector::VectorAlu;

/// Cycle-accurate executor that models pipeline timing.
///
/// This executor tracks:
/// - Instruction latencies
/// - Register hazards (RAW, WAW)
/// - Memory bank conflicts
/// - Pipeline stalls
///
/// Use this for:
/// - Performance analysis
/// - Cycle-accurate simulation
/// - Validation against hardware
pub struct CycleAccurateExecutor {
    /// Track call return address.
    pending_call_return_addr: Option<u32>,

    /// Latency lookup table.
    latencies: LatencyTable,

    /// Register hazard detector.
    hazards: HazardDetector,

    /// Memory bank conflict detector.
    memory: MemoryModel,

    /// Total stall cycles from hazards.
    pub total_hazard_stalls: u64,

    /// Total stall cycles from memory conflicts.
    pub total_memory_stalls: u64,

    /// Total stall cycles from branch penalties.
    pub total_branch_stalls: u64,

    /// Detailed stall breakdown by reason.
    detailed_stats: HazardStats,
}

impl CycleAccurateExecutor {
    /// Create a new cycle-accurate executor.
    pub fn new() -> Self {
        Self {
            pending_call_return_addr: None,
            latencies: LatencyTable::aie2(),
            hazards: HazardDetector::new(),
            memory: MemoryModel::new(),
            total_hazard_stalls: 0,
            total_memory_stalls: 0,
            total_branch_stalls: 0,
            detailed_stats: HazardStats::default(),
        }
    }

    /// Calculate the execution cycles for a slot operation.
    fn operation_cycles(&self, op: &SlotOp) -> u8 {
        let key = LatencyTable::key_from_operation(&op.op);
        self.latencies.latency(key)
    }

    /// Check for register hazards and return stall cycles needed.
    /// Reserved for future cycle-accurate pipeline hazard integration.
    #[allow(dead_code)]
    fn check_hazards(&self, op: &SlotOp) -> u8 {
        let hazards = self.hazards.check_operation(op);
        self.hazards.max_stall(&hazards)
    }

    /// Check for memory bank conflicts.
    fn check_memory_conflict(&self, op: &SlotOp) -> u8 {
        match &op.op {
            Operation::Load { width, .. } | Operation::Store { width, .. } => {
                // Get address from pointer register source
                let addr = self.get_memory_address(op);
                let access = MemoryAccess {
                    address: addr,
                    width: width.bytes(),
                    is_write: matches!(op.op, Operation::Store { .. }),
                    port: if matches!(op.op, Operation::Load { .. }) { 0 } else { 2 },
                };
                self.memory.check_conflict(&access).stall_cycles
            }
            _ => 0,
        }
    }

    /// Extract memory address from operation (simplified).
    fn get_memory_address(&self, _op: &SlotOp) -> u32 {
        // In a real implementation, we'd read the pointer register value
        // from the execution context. For now, return 0 as a placeholder.
        // The actual address would be resolved during execution.
        0
    }

    /// Execute a single slot operation with timing tracking.
    fn execute_slot(
        &mut self,
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
    ) -> Option<ExecuteResult> {
        self.execute_slot_with_mem_locks(op, ctx, tile, None)
    }

    /// Execute a single slot operation with optional memory tile lock routing.
    fn execute_slot_with_mem_locks(
        &mut self,
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
        mem_tile_locks: Option<&mut [crate::device::tile::Lock; 64]>,
    ) -> Option<ExecuteResult> {
        // Check for call - save return address
        if matches!(op.op, Operation::Call) {
            self.pending_call_return_addr = Some(ctx.pc());
        }

        // Execute using the functional units
        if ScalarAlu::execute(op, ctx) {
            return None;
        }

        if VectorAlu::execute(op, ctx) {
            return None;
        }

        if MemoryUnit::execute(op, ctx, tile) {
            return None;
        }

        if let Some(result) = ControlUnit::execute_with_mem_locks(op, ctx, tile, mem_tile_locks) {
            return Some(result);
        }

        // Unknown operation - fail loudly to prevent silent incorrect behavior
        if let Operation::Unknown { opcode } = &op.op {
            return Some(ExecuteResult::Error {
                message: format!(
                    "Unknown instruction opcode 0x{:08X} at slot {:?}",
                    opcode, op.slot
                ),
            });
        }

        None
    }

    /// Record register writes for hazard tracking.
    fn record_writes(&mut self, op: &SlotOp) {
        let latency = self.operation_cycles(op);
        self.hazards.record_operation(op, latency);
    }

    /// Record memory access for bank conflict tracking.
    fn record_memory_access(&mut self, op: &SlotOp) {
        match &op.op {
            Operation::Load { width, .. } | Operation::Store { width, .. } => {
                let addr = self.get_memory_address(op);
                let access = MemoryAccess {
                    address: addr,
                    width: width.bytes(),
                    is_write: matches!(op.op, Operation::Store { .. }),
                    port: if matches!(op.op, Operation::Load { .. }) { 0 } else { 2 },
                };
                self.memory.record_access(&access);
            }
            _ => {}
        }
    }

    /// Reset executor state (for new program).
    pub fn reset(&mut self) {
        self.pending_call_return_addr = None;
        self.hazards.reset();
        self.memory.reset();
        self.total_hazard_stalls = 0;
        self.total_memory_stalls = 0;
        self.total_branch_stalls = 0;
        self.detailed_stats = HazardStats::default();
    }

    /// Get statistics.
    pub fn stats(&self) -> CycleAccurateStats {
        // Merge detailed stats with hazard detector stats
        let mut combined_stats = self.detailed_stats;
        combined_stats.merge(&self.hazards.stats());

        CycleAccurateStats {
            hazard_stalls: self.total_hazard_stalls,
            memory_stalls: self.total_memory_stalls,
            branch_stalls: self.total_branch_stalls,
            hazard_stats: combined_stats,
            memory_stats: self.memory.stats(),
        }
    }

    /// Get detailed stall breakdown.
    pub fn detailed_stats(&self) -> &HazardStats {
        &self.detailed_stats
    }
}

impl Default for CycleAccurateExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl CycleAccurateExecutor {
    /// Execute a bundle with optional memory tile lock routing.
    ///
    /// For compute tiles, pass the adjacent memory tile's locks to enable
    /// proper routing of lock IDs 48-63 (memory module locks).
    pub fn execute_with_mem_tile(
        &mut self,
        bundle: &VliwBundle,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
        mem_tile_locks: Option<&mut [crate::device::tile::Lock; 64]>,
    ) -> ExecuteResult {
        self.execute_internal(bundle, ctx, tile, mem_tile_locks)
    }

    /// Internal execution with optional memory tile locks.
    fn execute_internal(
        &mut self,
        bundle: &VliwBundle,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
        mut mem_tile_locks: Option<&mut [crate::device::tile::Lock; 64]>,
    ) -> ExecuteResult {
        self.pending_call_return_addr = None;

        let pc = ctx.pc();
        let start_cycle = ctx.cycles;

        // Record instruction start event
        ctx.timing_context_mut().record_event(start_cycle, EventType::InstructionStart { pc });

        // Advance timing models to current cycle
        self.hazards.advance_to(ctx.cycles);
        self.memory.advance_to(ctx.cycles);

        // Phase 1: Calculate stalls for all operations and collect StallReasons
        let mut total_stall = 0u8;
        let mut stall_reasons: Vec<StallReason> = Vec::new();

        for op in bundle.active_slots() {
            // Check register hazards
            let hazards = self.hazards.check_operation(op);
            for hazard in &hazards {
                let stall = hazard.stall_cycles;
                if stall > 0 {
                    total_stall = total_stall.max(stall);
                    stall_reasons.push(hazard.to_stall_reason());
                }
            }

            // Check memory conflicts
            let memory_stall = self.check_memory_conflict(op);
            if memory_stall > 0 {
                total_stall = total_stall.max(memory_stall);
                // Get bank from address (simplified - uses address bits)
                let bank = (self.get_memory_address(op) / 8192) as u8 & 0x7;
                stall_reasons.push(StallReason::MemoryConflict {
                    bank,
                    cycles: memory_stall,
                });
            }
        }

        // Check structural hazards (slot conflicts)
        let structural_conflicts = check_bundle_conflicts(bundle);
        for conflict in &structural_conflicts {
            total_stall = total_stall.max(conflict.penalty_cycles);
            stall_reasons.push(StallReason::StructuralHazard {
                resource: conflict.resource.name(),
                cycles: conflict.penalty_cycles,
            });
        }

        // Record stalls and their reasons
        if total_stall > 0 {
            self.total_hazard_stalls += total_stall as u64;
            ctx.record_stall(total_stall as u64);

            // Record detailed breakdown
            for reason in &stall_reasons {
                self.detailed_stats.record(reason);
            }

            // Emit stall events for profiling
            let timing = ctx.timing_context_mut();
            for reason in &stall_reasons {
                match reason {
                    StallReason::RegisterHazard { hazard_type, register, cycles } => {
                        timing.record_event(
                            start_cycle,
                            EventType::RegisterHazard {
                                hazard_type: *hazard_type,
                                register: *register,
                                cycles: *cycles,
                            },
                        );
                    }
                    StallReason::MemoryConflict { bank, cycles } => {
                        timing.record_event(
                            start_cycle,
                            EventType::MemoryConflict {
                                bank: *bank,
                                cycles: *cycles,
                            },
                        );
                    }
                    StallReason::BranchPenalty { cycles } => {
                        timing.record_event(
                            start_cycle,
                            EventType::BranchPenalty { cycles: *cycles },
                        );
                    }
                    StallReason::StructuralHazard { .. }
                    | StallReason::LockContention { .. }
                    | StallReason::DmaWait { .. } => {
                        // These don't have direct event mappings yet
                    }
                }
            }
        }

        // Phase 2: Execute all slot operations
        //
        // VLIW semantics require all reads to happen before any writes within
        // the same instruction word. We achieve this by executing slots in a
        // specific order:
        // 1. Load slots - read from memory
        // 2. Store slots - read from registers (must happen before scalar writes!)
        // 3. Scalar/Vector/Control - may write to registers
        //
        // This ensures that when a bundle contains both "st r7, [p1]" and
        // "add r7, r8, #1", the store captures r7's value BEFORE the add modifies it.
        use crate::interpreter::bundle::SlotIndex;

        let mut final_result = ExecuteResult::Continue;

        // Execution order: Load(4), Store(5), then Scalar0(0), Scalar1(1), Vector(2), Accumulator(3), Control(6)
        let execution_order = [
            SlotIndex::Load,       // Memory reads first
            SlotIndex::Store,      // Register reads for stores (before scalar writes!)
            SlotIndex::Scalar0,    // Scalar operations
            SlotIndex::Scalar1,
            SlotIndex::Vector,
            SlotIndex::Accumulator,
            SlotIndex::Control,
        ];

        for slot_idx in &execution_order {
            if let Some(ref op) = bundle.slots()[*slot_idx as usize] {
                // Reborrow mem_tile_locks for each slot operation
                let slot_mem_locks = mem_tile_locks.as_mut().map(|locks| &mut **locks);
                if let Some(result) = self.execute_slot_with_mem_locks(op, ctx, tile, slot_mem_locks) {
                    match &result {
                        ExecuteResult::Branch { .. }
                        | ExecuteResult::Halt
                        | ExecuteResult::WaitLock { .. }
                        | ExecuteResult::WaitDma { .. }
                        | ExecuteResult::WaitStream { .. } => {
                            final_result = result;
                        }
                        ExecuteResult::Continue => {}
                        ExecuteResult::Error { .. } => {
                            return result;
                        }
                    }
                }
            }
        }

        // Phase 3: Record writes and memory accesses for future hazard detection
        for op in bundle.active_slots() {
            self.record_writes(op);
            self.record_memory_access(op);
        }

        // Handle call return address
        if let Some(return_addr) = self.pending_call_return_addr {
            ctx.set_lr(return_addr.wrapping_add(bundle.size() as u32));
        }

        // Calculate execution cycles (max latency of all ops in bundle)
        let mut max_latency = 1u8;
        for op in bundle.active_slots() {
            let op_latency = self.operation_cycles(op);
            max_latency = max_latency.max(op_latency);
        }

        // Apply branch penalty if a branch was taken
        // (Pipeline flush due to mispredicted/unconditional branch)
        if let ExecuteResult::Branch { target } = final_result {
            let penalty = BRANCH_PENALTY_CYCLES;
            self.total_branch_stalls += penalty as u64;
            ctx.record_stall(penalty as u64);

            // Record in detailed stats
            self.detailed_stats.record(&StallReason::BranchPenalty { cycles: penalty });

            // Emit branch events
            let branch_cycle = ctx.cycles;
            let timing = ctx.timing_context_mut();
            timing.record_event(branch_cycle, EventType::BranchPenalty { cycles: penalty });
            timing.record_event(
                branch_cycle,
                EventType::BranchTaken { from_pc: pc, to_pc: target },
            );
        }

        // Update statistics with actual cycles
        ctx.record_instruction(max_latency as u64);

        // Sync timing context and record instruction complete
        let end_cycle = ctx.cycles;
        let timing = ctx.timing_context_mut();
        timing.hazard_stalls = self.total_hazard_stalls;
        timing.memory_stalls = self.total_memory_stalls;

        // Record instruction completion
        timing.record_event(
            end_cycle,
            EventType::InstructionComplete { pc, latency: max_latency },
        );

        final_result
    }
}

impl Executor for CycleAccurateExecutor {
    fn execute(
        &mut self,
        bundle: &VliwBundle,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
    ) -> ExecuteResult {
        // Delegate to internal implementation with no memory tile locks
        self.execute_internal(bundle, ctx, tile, None)
    }

    fn is_cycle_accurate(&self) -> bool {
        true
    }
}

/// Statistics from cycle-accurate execution.
#[derive(Debug, Clone)]
pub struct CycleAccurateStats {
    /// Total cycles stalled due to register hazards.
    pub hazard_stalls: u64,
    /// Total cycles stalled due to memory conflicts.
    pub memory_stalls: u64,
    /// Total cycles stalled due to branch penalties.
    pub branch_stalls: u64,
    /// Detailed hazard statistics.
    pub hazard_stats: crate::interpreter::timing::hazards::HazardStats,
    /// Detailed memory statistics.
    pub memory_stats: crate::interpreter::timing::memory::MemoryStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::{MemWidth, Operand, PostModify, SlotIndex};

    fn make_bundle(ops: Vec<SlotOp>) -> VliwBundle {
        let mut bundle = VliwBundle::empty();
        for op in ops {
            bundle.set_slot(op);
        }
        bundle
    }

    #[test]
    fn test_cycle_accurate_creation() {
        let executor = CycleAccurateExecutor::new();
        assert!(executor.is_cycle_accurate());
    }

    #[test]
    fn test_empty_bundle() {
        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        let bundle = make_bundle(vec![]);
        let result = executor.execute(&bundle, &mut ctx, &mut tile);

        assert!(matches!(result, ExecuteResult::Continue));
    }

    #[test]
    fn test_scalar_add_timing() {
        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        ctx.scalar.write(0, 10);
        ctx.scalar.write(1, 20);

        // Scalar add has 1 cycle latency
        let bundle = make_bundle(vec![
            SlotOp::new(SlotIndex::Scalar0, Operation::ScalarAdd)
                .with_dest(Operand::ScalarReg(2))
                .with_source(Operand::ScalarReg(0))
                .with_source(Operand::ScalarReg(1)),
        ]);

        executor.execute(&bundle, &mut ctx, &mut tile);

        assert_eq!(ctx.scalar.read(2), 30);
        assert_eq!(ctx.cycles, 1); // 1 cycle for scalar add
    }

    #[test]
    fn test_scalar_mul_timing() {
        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        ctx.scalar.write(0, 5);
        ctx.scalar.write(1, 6);

        // Scalar mul has 2 cycle latency
        let bundle = make_bundle(vec![
            SlotOp::new(SlotIndex::Scalar1, Operation::ScalarMul)
                .with_dest(Operand::ScalarReg(2))
                .with_source(Operand::ScalarReg(0))
                .with_source(Operand::ScalarReg(1)),
        ]);

        executor.execute(&bundle, &mut ctx, &mut tile);

        assert_eq!(ctx.scalar.read(2), 30);
        assert_eq!(ctx.cycles, 2); // 2 cycles for scalar mul
    }

    #[test]
    fn test_memory_load_timing() {
        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        tile.write_data_u32(0x100, 42);
        ctx.pointer.write(0, 0x100);

        // Memory load has 5 cycle latency
        let bundle = make_bundle(vec![SlotOp::new(
            SlotIndex::Load,
            Operation::Load {
                width: MemWidth::Word,
                post_modify: PostModify::None,
            },
        )
        .with_dest(Operand::ScalarReg(0))
        .with_source(Operand::PointerReg(0))]);

        executor.execute(&bundle, &mut ctx, &mut tile);

        assert_eq!(ctx.scalar.read(0), 42);
        assert_eq!(ctx.cycles, 5); // 5 cycles for memory load
    }

    #[test]
    fn test_reset() {
        let mut executor = CycleAccurateExecutor::new();
        executor.total_hazard_stalls = 10;
        executor.total_memory_stalls = 5;
        executor.total_branch_stalls = 3;

        executor.reset();

        assert_eq!(executor.total_hazard_stalls, 0);
        assert_eq!(executor.total_memory_stalls, 0);
        assert_eq!(executor.total_branch_stalls, 0);
    }

    #[test]
    fn test_stats() {
        let executor = CycleAccurateExecutor::new();
        let stats = executor.stats();

        assert_eq!(stats.hazard_stalls, 0);
        assert_eq!(stats.memory_stalls, 0);
        assert_eq!(stats.branch_stalls, 0);
    }

    #[test]
    fn test_branch_penalty() {
        use crate::interpreter::bundle::BranchCondition;

        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        // Create a bundle with an unconditional branch instruction
        let bundle = make_bundle(vec![
            SlotOp::new(SlotIndex::Control, Operation::Branch {
                condition: BranchCondition::Always,
            })
            .with_source(Operand::Immediate(0x100)), // branch target
        ]);

        let result = executor.execute(&bundle, &mut ctx, &mut tile);

        // Should return Branch result
        assert!(matches!(result, ExecuteResult::Branch { target: 0x100 }));

        // Should have recorded branch penalty (3 cycles)
        assert_eq!(executor.total_branch_stalls, 3);

        // Context should have penalty in stall count
        // (1 cycle for instruction + 3 for branch penalty = 4 total cycles,
        // but stalls are 3)
        assert!(ctx.stall_cycles >= 3);

        // Detailed stats should track branch stalls
        let detailed = executor.detailed_stats();
        assert_eq!(detailed.branch_stall_cycles, 3);
        assert_eq!(detailed.total_stall_cycles, 3);
    }

    #[test]
    fn test_detailed_stats_integration() {
        use crate::interpreter::bundle::BranchCondition;

        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        // Execute a branch to generate branch penalty stalls
        let bundle = make_bundle(vec![
            SlotOp::new(SlotIndex::Control, Operation::Branch {
                condition: BranchCondition::Always,
            })
            .with_source(Operand::Immediate(0x200)),
        ]);

        executor.execute(&bundle, &mut ctx, &mut tile);

        // Get full stats
        let stats = executor.stats();

        // Branch stalls should be tracked
        assert_eq!(stats.branch_stalls, 3);

        // Hazard stats should include branch stalls in breakdown
        assert_eq!(stats.hazard_stats.branch_stall_cycles, 3);
        assert_eq!(stats.hazard_stats.total_stall_cycles, 3);

        // Register and memory stalls should be zero (no hazards in this bundle)
        assert_eq!(stats.hazard_stats.register_stall_cycles, 0);
        assert_eq!(stats.hazard_stats.memory_stall_cycles, 0);
    }

    // --- Event Recording Tests ---

    #[test]
    fn test_event_recording_instruction_cycle() {
        use crate::interpreter::state::EventType;

        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new_with_timing();
        let mut tile = Tile::compute(0, 2);

        // Execute a simple scalar add
        ctx.scalar.write(0, 10);
        ctx.scalar.write(1, 20);

        let bundle = make_bundle(vec![
            SlotOp::new(SlotIndex::Scalar0, Operation::ScalarAdd)
                .with_dest(Operand::ScalarReg(2))
                .with_source(Operand::ScalarReg(0))
                .with_source(Operand::ScalarReg(1)),
        ]);

        executor.execute(&bundle, &mut ctx, &mut tile);

        // Check events were recorded
        let timing = ctx.timing_context();
        let events = timing.events.events();

        // Should have at least InstructionStart and InstructionComplete
        assert!(events.len() >= 2, "Expected at least 2 events, got {}", events.len());

        // First event should be InstructionStart
        assert!(
            matches!(events[0].event, EventType::InstructionStart { pc: 0 }),
            "First event should be InstructionStart at PC 0"
        );

        // Last event should be InstructionComplete
        let last = &events[events.len() - 1].event;
        assert!(
            matches!(last, EventType::InstructionComplete { pc: 0, .. }),
            "Last event should be InstructionComplete"
        );
    }

    #[test]
    fn test_event_recording_branch() {
        use crate::interpreter::bundle::BranchCondition;
        use crate::interpreter::state::EventType;

        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        // Execute a branch instruction
        let bundle = make_bundle(vec![
            SlotOp::new(SlotIndex::Control, Operation::Branch {
                condition: BranchCondition::Always,
            })
            .with_source(Operand::Immediate(0x200)),
        ]);

        executor.execute(&bundle, &mut ctx, &mut tile);

        // Check for branch events
        let timing = ctx.timing_context();
        let events = timing.events.events();

        // Should have BranchPenalty and BranchTaken events
        let has_branch_penalty = events.iter().any(|e| matches!(e.event, EventType::BranchPenalty { .. }));
        let has_branch_taken = events.iter().any(|e| {
            matches!(e.event, EventType::BranchTaken { from_pc: 0, to_pc: 0x200 })
        });

        assert!(has_branch_penalty, "Should have BranchPenalty event");
        assert!(has_branch_taken, "Should have BranchTaken event");
    }

    #[test]
    fn test_all_execution_has_timing() {
        // All contexts now have timing enabled
        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        let bundle = make_bundle(vec![
            SlotOp::new(SlotIndex::Scalar0, Operation::ScalarAdd)
                .with_dest(Operand::ScalarReg(2))
                .with_source(Operand::ScalarReg(0))
                .with_source(Operand::ScalarReg(1)),
        ]);

        executor.execute(&bundle, &mut ctx, &mut tile);

        // Should always have timing context
        assert!(ctx.has_timing());
        // Events should be recorded
        assert!(!ctx.timing_context().events.events().is_empty());
    }
}
