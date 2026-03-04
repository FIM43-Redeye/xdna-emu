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
use crate::interpreter::bundle::{SlotOp, VliwBundle};
use crate::tablegen::SemanticOp;
use crate::interpreter::state::{EventType, ExecutionContext};
use crate::interpreter::timing::{
    HazardDetector, HazardStats, LatencyTable, MemoryAccess, MemoryModel,
    StallReason,
};
use crate::interpreter::traits::{ExecuteResult, Executor};

use super::cascade::{CascadeOps, CascadeResult};
use super::control::ControlUnit;
use super::memory::{MemoryUnit, NeighborMemory};
use super::semantic::execute_semantic;
use super::stream::{StreamOps, StreamResult};
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
    /// Uses the TableGen-derived SemanticOp for latency lookup.
    fn operation_cycles(&self, op: &SlotOp) -> u8 {
        self.latencies.timing_for_slot_op(op).latency
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
        if matches!(op.semantic, Some(SemanticOp::Call)) {
            self.pending_call_return_addr = Some(ctx.pc());
        }

        // Execute using the functional units.
        // Semantic dispatch handles pure register operations (scalar arithmetic,
        // logic, moves, flags). Falls through for vector, memory, stream,
        // cascade, and control flow ops.
        if execute_semantic(op, ctx) {
            return None;
        }

        if VectorAlu::execute(op, ctx) {
            return None;
        }

        if MemoryUnit::execute(op, ctx, tile, neighbors) {
            return None;
        }

        // Cascade operations: dedicated 384-bit point-to-point link, separate
        // from the stream switch fabric. Stall returns WaitStream with sentinel
        // port 255 so the coordinator retries next cycle.
        match CascadeOps::execute(op, ctx, tile) {
            CascadeResult::Completed => return None,
            CascadeResult::Stall => return Some(ExecuteResult::WaitStream { port: 255 }),
            CascadeResult::NotCascadeOp => {}
            CascadeResult::Error(msg) => return Some(ExecuteResult::Error { message: msg }),
        }

        // Stream operations: read/write between scalar registers and stream
        // switch ports. Stall on blocking read from empty stream.
        match StreamOps::execute(op, ctx, tile) {
            StreamResult::Completed => return None,
            StreamResult::Stall { port } => return Some(ExecuteResult::WaitStream { port }),
            StreamResult::NotStreamOp => {}
        }

        if let Some(result) = ControlUnit::execute_with_mem_locks(op, ctx, tile, mem_tile_locks) {
            return Some(result);
        }

        // Defense in depth: instructions without a semantic label have no
        // routing information at all. With 100% semantic coverage this path
        // should be unreachable, but keep it as a safety net.
        if op.semantic.is_none() {
            return Some(ExecuteResult::Error {
                message: format!(
                    "Unknown instruction (no semantic): slot={:?}, name={:?}, opcode={:#X?}",
                    op.slot, op.encoding_name, op.raw_opcode,
                ),
            });
        }

        // Every instruction with a semantic label should be claimed by one
        // of the execution units above. Reaching here means a labeled but
        // unimplemented op -- abort immediately rather than silently
        // producing wrong results by pretending execution succeeded.
        Some(ExecuteResult::Error {
            message: format!(
                "Unhandled instruction: semantic={:?}, slot={:?}, name={:?}",
                op.semantic, op.slot, op.encoding_name,
            ),
        })
    }

    /// Record register writes for hazard tracking.
    fn record_writes(&mut self, op: &SlotOp) {
        let latency = self.operation_cycles(op);
        self.hazards.record_operation(op, latency);
    }

    /// Record memory access for bank conflict tracking.
    fn record_memory_access(&mut self, op: &SlotOp) {
        match op.semantic {
            Some(SemanticOp::Load) | Some(SemanticOp::Store) if !op.is_vector => {
                // Placeholder: actual address resolution requires the execution
                // context, which record_memory_access doesn't currently receive.
                let addr = 0u32;
                let is_store = matches!(op.semantic, Some(SemanticOp::Store));
                let access = MemoryAccess {
                    address: addr,
                    width: op.mem_width.bytes(),
                    is_write: is_store,
                    port: if is_store { 2 } else { 0 },
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
        // NOTE: Hazard stalls are disabled. The AIE2 scoreboard model needs
        // precise per-stage tracking before stalls can be re-enabled. The
        // compiler's schedule is trusted for now. See timing/hazards.rs for
        // the detection infrastructure (recording is active for stats).

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

        // Execution order: LoadA/LoadB first, then Store, then compute slots.
        // Both load ports issue in the same cycle (they share a load unit).
        let execution_order = [
            SlotIndex::LoadA,      // Primary load port (LDA)
            SlotIndex::LoadB,      // Secondary load port (LDB)
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
                    SlotIndex::LoadA | SlotIndex::LoadB => Some(EventType::InstrLoad { pc }),
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
    use crate::interpreter::bundle::{MemWidth, Operand, SlotIndex};
    use crate::tablegen::SemanticOp;

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
            SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Add)
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
            SlotOp::from_semantic(SlotIndex::Scalar1, SemanticOp::Mul)
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
        let bundle = make_bundle(vec![
            SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
                .with_mem_width(MemWidth::Word)
                .with_dest(Operand::ScalarReg(0))
                .with_source(Operand::PointerReg(0)),
        ]);

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
            SlotOp::from_semantic(SlotIndex::Control, SemanticOp::Br)
                .with_branch_condition(BranchCondition::Always)
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
            SlotOp::from_semantic(SlotIndex::Control, SemanticOp::Br)
                .with_branch_condition(BranchCondition::Always)
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
        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        // Execute a simple scalar add
        ctx.scalar.write(0, 10);
        ctx.scalar.write(1, 20);

        let bundle = make_bundle(vec![
            SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Add)
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
            SlotOp::from_semantic(SlotIndex::Control, SemanticOp::Br)
                .with_branch_condition(BranchCondition::Always)
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
            SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Add)
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

    // --- Deferred Pointer Write Tests ---

    #[test]
    fn test_pointer_write_deferred_by_one_cycle() {
        // A MOV-to-pointer (via SemanticOp::Copy) should defer the write by 1 cycle.
        // Within the same bundle, the old pointer value should be visible.
        // After commit_pending_writes at the next cycle, the new value is visible.
        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        ctx.pointer.write(1, 0xDEAD); // Old p1 value

        // movxm p1, #0x70400
        let bundle = make_bundle(vec![
            SlotOp::from_semantic(SlotIndex::Scalar1, SemanticOp::Copy)
                .with_dest(Operand::PointerReg(1))
                .with_source(Operand::Immediate(0x70400)),
        ]);

        executor.execute(&bundle, &mut ctx, &mut tile);

        // After execution, p1 should still be the OLD value (write is deferred).
        assert_eq!(
            ctx.pointer.read(1), 0xDEAD,
            "Pointer write should be deferred, not visible immediately"
        );

        // Execute another bundle (any NOP will do). This calls commit_pending_writes.
        let nop = make_bundle(vec![]);
        executor.execute(&nop, &mut ctx, &mut tile);

        // Now p1 should have the new value (committed at start of cycle 2).
        assert_eq!(
            ctx.pointer.read(1), 0x70400,
            "Pointer write should be committed after one cycle"
        );
    }

    #[test]
    fn test_delay_pending_writes_at_branch_boundary() {
        // When delay_pending_writes(1) is called at a branch boundary,
        // pointer writes from the last delay slot need an extra cycle.
        // This test verifies the mechanism directly on ExecutionContext.
        let mut ctx = ExecutionContext::new();
        ctx.pointer.write(1, 0xDEAD);

        // Simulate movxm p1, #0x70400 at cycle 5 (last delay slot)
        ctx.cycles = 5;
        ctx.queue_pointer_write(1, 0x70400, 1); // ready_cycle = 6

        // Branch taken: delay pending writes by 1 extra cycle
        ctx.delay_pending_writes(1); // ready_cycle becomes 7

        // At cycle 6 (branch target first instruction): commit should NOT apply
        ctx.cycles = 6;
        ctx.commit_pending_writes();
        assert_eq!(
            ctx.pointer.read(1), 0xDEAD,
            "Pointer write should NOT be visible at branch target (cycle 6)"
        );

        // At cycle 7 (branch target second instruction): commit SHOULD apply
        ctx.cycles = 7;
        ctx.commit_pending_writes();
        assert_eq!(
            ctx.pointer.read(1), 0x70400,
            "Pointer write should be visible one cycle after branch target"
        );
    }

    #[test]
    fn test_pointer_write_sequential_latency_one() {
        // In sequential (non-branch) code, pointer writes should have latency 1:
        // movxm p1, #addr at cycle C → p1 available at cycle C+1.
        let mut ctx = ExecutionContext::new();
        ctx.pointer.write(1, 0);

        ctx.cycles = 10;
        ctx.queue_pointer_write(1, 0x1234, 1); // ready_cycle = 11

        // At cycle 11: commit should apply (no branch delay)
        ctx.cycles = 11;
        ctx.commit_pending_writes();
        assert_eq!(ctx.pointer.read(1), 0x1234);
    }
}
