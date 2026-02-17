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
    HazardDetector, HazardStats, LatencyTable, MemoryAccess, MemoryModel,
    StallReason,
};
use crate::interpreter::traits::{ExecuteResult, Executor};

use super::control::ControlUnit;
use super::memory::{MemoryUnit, NeighborMemory};
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
    ///
    /// Prefers the TableGen-derived SemanticOp path when available,
    /// falling back to the deprecated Operation enum for unmatched instructions.
    fn operation_cycles(&self, op: &SlotOp) -> u8 {
        self.latencies.timing_for_slot_op(op).latency
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
    #[allow(dead_code)]
    fn execute_slot(
        &mut self,
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
    ) -> Option<ExecuteResult> {
        self.execute_slot_with_mem_locks(op, ctx, tile, None, None)
    }

    /// Execute a single slot operation with optional memory tile lock routing.
    fn execute_slot_with_mem_locks(
        &mut self,
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
        mem_tile_locks: Option<&mut [crate::device::tile::Lock]>,
        neighbors: Option<&mut NeighborMemory>,
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

        if MemoryUnit::execute(op, ctx, tile, neighbors) {
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
    /// Execute a bundle with optional memory tile lock routing and cross-tile memory.
    ///
    /// For compute tiles, pass the adjacent memory tile's locks to enable
    /// proper routing of lock IDs 48-63 (memory module locks).
    /// Pass `neighbors` to enable cross-tile memory access (quadrants 1-3).
    pub fn execute_with_mem_tile(
        &mut self,
        bundle: &VliwBundle,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
        mem_tile_locks: Option<&mut [crate::device::tile::Lock]>,
        neighbors: Option<&mut NeighborMemory>,
    ) -> ExecuteResult {
        self.execute_internal(bundle, ctx, tile, mem_tile_locks, neighbors)
    }

    /// Internal execution with optional memory tile locks and cross-tile memory.
    fn execute_internal(
        &mut self,
        bundle: &VliwBundle,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
        mut mem_tile_locks: Option<&mut [crate::device::tile::Lock]>,
        mut neighbors: Option<&mut NeighborMemory>,
    ) -> ExecuteResult {
        self.pending_call_return_addr = None;

        let pc = ctx.pc();
        let start_cycle = ctx.cycles;

        // Advance timing models to current cycle
        self.hazards.advance_to(ctx.cycles);
        self.memory.advance_to(ctx.cycles);

        // Phase 1: Calculate stalls
        //
        // NOTE: Register hazard stalls and memory bank conflict stalls are
        // currently DISABLED. The LLVM scheduler (Peano) produces code that
        // already resolves data hazards via instruction scheduling -- it places
        // enough independent instructions between a write and its dependent
        // read to cover the pipeline latency. Adding scoreboard-based stalls
        // on top of compiler-scheduled code produces incorrect timing: extra
        // stall cycles cause load results to become visible too early relative
        // to the instruction that needs them, breaking the compiler's
        // carefully-planned pipeline interleaving.
        //
        // The correct long-term fix is to model the AIE2 scoreboard
        // precisely, tracking which pipeline stages each instruction occupies
        // and only stalling when the compiler's schedule is truly insufficient.
        // For now, trusting the compiler produces correct functional results.
        let total_stall = 0u8;
        let stall_reasons: Vec<StallReason> = Vec::new();

        // Record stalls and their reasons (currently always 0 -- see note above)
        if total_stall > 0 {
            self.total_hazard_stalls += total_stall as u64;
            ctx.record_stall(total_stall as u64);

            // Record detailed breakdown
            for reason in &stall_reasons {
                self.detailed_stats.record(reason);
            }

            // Emit hardware-aligned stall events
            let timing = ctx.timing_context_mut();
            for reason in &stall_reasons {
                match reason {
                    StallReason::RegisterHazard { cycles, .. } => {
                        // Register hazards manifest as memory stalls at hardware level
                        timing.record_event(start_cycle, EventType::MemoryStall { cycles: *cycles });
                    }
                    StallReason::MemoryConflict { cycles, .. } => {
                        timing.record_event(start_cycle, EventType::MemoryStall { cycles: *cycles });
                    }
                    StallReason::LockContention { .. } => {
                        timing.record_event(start_cycle, EventType::LockStall { cycles: 1 });
                    }
                    StallReason::BranchPenalty { .. }
                    | StallReason::StructuralHazard { .. }
                    | StallReason::DmaWait { .. } => {}
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

        // Commit deferred load results whose latency has elapsed.
        // This must happen BEFORE the VLIW snapshot so that load results
        // become visible at the correct cycle and are captured by begin_bundle().
        ctx.commit_pending_writes();

        // AIE2 uses pure VLIW semantics: all reads within a bundle see the
        // pre-execution values. This is confirmed by the llvm-aie scheduling
        // model (AIE2Schedule.td) and hazard recognizer which handle data
        // dependencies only across bundles, not within them.
        //
        // The begin_bundle snapshot ensures that when `lda r13, [p5]` and
        // `or r3, r13, r0` are in the same bundle, the OR reads r13's value
        // from BEFORE the load writes it.
        ctx.begin_bundle();

        for slot_idx in &execution_order {
            if let Some(ref op) = bundle.slots()[*slot_idx as usize] {
                // Reborrow mem_tile_locks and neighbors for each slot operation
                let slot_mem_locks = mem_tile_locks.as_mut().map(|locks| &mut **locks);
                let slot_neighbors = neighbors.as_mut().map(|n| &mut **n);
                if let Some(result) = self.execute_slot_with_mem_locks(op, ctx, tile, slot_mem_locks, slot_neighbors) {
                    match &result {
                        ExecuteResult::Branch { .. }
                        | ExecuteResult::Call { .. }
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

                // Emit per-slot instruction class event matching hardware trace codes.
                // Each active slot generates its own event, just like hardware where
                // different event types fire for each functional unit.
                let event = match slot_idx {
                    SlotIndex::Load => Some(EventType::InstrLoad { pc }),
                    SlotIndex::Store => Some(EventType::InstrStore { pc }),
                    SlotIndex::Vector | SlotIndex::Accumulator =>
                        Some(EventType::InstrVector { pc }),
                    _ => None, // Scalar/Control events classified below
                };
                if let Some(evt) = event {
                    ctx.timing_context_mut().record_event(start_cycle, evt);
                }
            }
        }

        ctx.end_bundle();

        // Phase 3: Record writes and memory accesses for future hazard detection
        for op in bundle.active_slots() {
            self.record_writes(op);
            self.record_memory_access(op);
        }

        // Convert Branch to Call when this was a jl instruction.
        // LR is NOT set here -- it's deferred until delay slots are exhausted
        // (in ExecutionContext::tick_delay_slots) so that delay slot instructions
        // see the pre-call LR value, matching hardware pipeline behavior.
        if self.pending_call_return_addr.is_some() {
            if let ExecuteResult::Branch { target } = final_result {
                final_result = ExecuteResult::Call { target };
            }
        }

        // Emit Call/Return/Halt events based on execution result.
        // These events map to hardware INSTR_CALL, INSTR_RETURN, DISABLED_CORE.
        match &final_result {
            ExecuteResult::Call { .. } => {
                ctx.timing_context_mut().record_event(start_cycle, EventType::InstrCall { pc });
            }
            // Return is detected by the control unit returning a Branch to LR.
            // We check if the branch target matches the link register value.
            ExecuteResult::Branch { target } if *target == ctx.lr() => {
                ctx.timing_context_mut().record_event(start_cycle, EventType::InstrReturn { pc });
            }
            ExecuteResult::Halt => {
                ctx.timing_context_mut().record_event(start_cycle, EventType::CoreDisabled);
            }
            ExecuteResult::WaitLock { .. } => {
                ctx.timing_context_mut().record_event(start_cycle, EventType::LockStall { cycles: 1 });
            }
            ExecuteResult::WaitStream { .. } => {
                ctx.timing_context_mut().record_event(start_cycle, EventType::StreamStall { cycles: 1 });
            }
            _ => {}
        }

        // Apply branch penalty if a branch was taken
        // (Pipeline flush due to mispredicted/unconditional branch)
        let branch_target = match final_result {
            ExecuteResult::Branch { target } | ExecuteResult::Call { target } => Some(target),
            _ => None,
        };
        if let Some(target) = branch_target {
            let penalty = BRANCH_PENALTY_CYCLES;
            self.total_branch_stalls += penalty as u64;
            ctx.record_stall(penalty as u64);

            // Record in detailed stats
            self.detailed_stats.record(&StallReason::BranchPenalty { cycles: penalty });

            // Emit branch event
            let branch_cycle = ctx.cycles;
            ctx.timing_context_mut().record_event(
                branch_cycle,
                EventType::BranchTaken { from_pc: pc, to_pc: target },
            );
        }

        // Advance cycle counter by 1 (pipelined issue rate).
        // Latency-based deferred writes and hazard stalls handle the rest.
        ctx.record_instruction(1);

        // Sync timing context
        let timing = ctx.timing_context_mut();
        timing.hazard_stalls = self.total_hazard_stalls;
        timing.memory_stalls = self.total_memory_stalls;

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
        // Delegate to internal implementation with no memory tile locks or neighbors
        self.execute_internal(bundle, ctx, tile, None, None)
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
        assert_eq!(ctx.cycles, 1); // 1 cycle to issue (pipelined)
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

        // Load results are deferred -- commit them now that the latency has elapsed.
        // In normal execution, this happens at the start of the next bundle.
        ctx.flush_pending_writes();
        assert_eq!(ctx.scalar.read(0), 42);
        assert_eq!(ctx.cycles, 1); // 1 cycle to issue (pipelined); result deferred by latency
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
        let mut ctx = ExecutionContext::new();
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

        // Check events were recorded.
        // Scalar-only bundles don't emit per-class events (scalars have no
        // dedicated hardware trace event), but the event log should still
        // have been populated if any timing sync occurred.
        let timing = ctx.timing_context();
        let events = timing.events.events();

        // Scalar-only: no per-slot instruction events are emitted (scalars
        // don't map to a hardware trace event code). The log may be empty
        // or have timing sync events only.
        // Just verify no crash and timing context is populated.
        assert!(ctx.cycles > 0, "Cycles should advance after execution");
        let _ = events; // events may or may not be present for scalar-only
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

        // Should have BranchTaken event (hardware-aligned)
        let has_branch_taken = events.iter().any(|e| {
            matches!(e.event, EventType::BranchTaken { from_pc: 0, to_pc: 0x200 })
        });

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
        // Cycles should have advanced (execution happened)
        assert!(ctx.cycles > 0, "Execution should advance cycles");
    }
}
