//! Cycle-accurate executor implementation.
//!
//! The `CycleAccurateExecutor` models pipeline stages, register hazards, and
//! memory bank conflicts to provide accurate cycle counts. This is slower
//! than `FastExecutor` but essential for performance analysis.
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

use crate::device::tile::Tile;
use crate::interpreter::bundle::{Operation, SlotOp, VliwBundle};
use crate::interpreter::state::ExecutionContext;
use crate::interpreter::timing::{
    LatencyTable, MemoryAccess, MemoryModel, HazardDetector,
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
        }
    }

    /// Calculate the execution cycles for a slot operation.
    fn operation_cycles(&self, op: &SlotOp) -> u8 {
        let key = LatencyTable::key_from_operation(&op.op);
        self.latencies.latency(key)
    }

    /// Check for register hazards and return stall cycles needed.
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

    /// Execute a single slot operation (same as FastExecutor but with timing).
    fn execute_slot(
        &mut self,
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
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

        if let Some(result) = ControlUnit::execute(op, ctx, tile) {
            return Some(result);
        }

        if matches!(op.op, Operation::Unknown { .. }) {
            return None;
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
    }

    /// Get statistics.
    pub fn stats(&self) -> CycleAccurateStats {
        CycleAccurateStats {
            hazard_stalls: self.total_hazard_stalls,
            memory_stalls: self.total_memory_stalls,
            hazard_stats: self.hazards.stats(),
            memory_stats: self.memory.stats(),
        }
    }
}

impl Default for CycleAccurateExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl Executor for CycleAccurateExecutor {
    fn execute(
        &mut self,
        bundle: &VliwBundle,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
    ) -> ExecuteResult {
        self.pending_call_return_addr = None;

        // Advance timing models to current cycle
        self.hazards.advance_to(ctx.cycles);
        self.memory.advance_to(ctx.cycles);

        // Phase 1: Calculate stalls for all operations
        let mut total_stall = 0u8;
        for op in bundle.active_slots() {
            // Check register hazards
            let hazard_stall = self.check_hazards(op);
            total_stall = total_stall.max(hazard_stall);

            // Check memory conflicts
            let memory_stall = self.check_memory_conflict(op);
            total_stall = total_stall.max(memory_stall);
        }

        // Record stalls
        if total_stall > 0 {
            self.total_hazard_stalls += total_stall as u64;
            ctx.record_stall(total_stall as u64);
        }

        // Phase 2: Execute all slot operations
        let mut final_result = ExecuteResult::Continue;
        for op in bundle.active_slots() {
            if let Some(result) = self.execute_slot(op, ctx, tile) {
                match &result {
                    ExecuteResult::Branch { .. }
                    | ExecuteResult::Halt
                    | ExecuteResult::WaitLock { .. }
                    | ExecuteResult::WaitDma { .. } => {
                        final_result = result;
                    }
                    ExecuteResult::Continue => {}
                    ExecuteResult::Error { .. } => {
                        return result;
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

        // Update statistics with actual cycles
        ctx.record_instruction(max_latency as u64);

        // Sync timing context if enabled
        if let Some(timing) = ctx.timing_context_mut() {
            timing.hazard_stalls = self.total_hazard_stalls;
            timing.memory_stalls = self.total_memory_stalls;
        }

        final_result
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

        executor.reset();

        assert_eq!(executor.total_hazard_stalls, 0);
        assert_eq!(executor.total_memory_stalls, 0);
    }

    #[test]
    fn test_stats() {
        let executor = CycleAccurateExecutor::new();
        let stats = executor.stats();

        assert_eq!(stats.hazard_stalls, 0);
        assert_eq!(stats.memory_stalls, 0);
    }
}
