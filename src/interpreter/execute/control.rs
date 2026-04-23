//! Control unit execution.
//!
//! Handles control flow and synchronization operations.
//!
//! # Architecture Note
//!
//! Like [`MemoryUnit`](super::MemoryUnit), the control unit is NOT a legacy
//! fallback - it handles operations that have side effects beyond register
//! writes (branches change PC, locks affect tile state, etc.).
//!
//! ## Execution Flow
//!
//! ```text
//! CycleAccurateExecutor::execute_slot()
//!         |
//!         v
//!   execute_semantic(op, ctx)  <-- Pure register ops only
//!         |
//!         v
//!   VectorAlu / MemoryUnit / CascadeOps / StreamOps
//!         |
//!         v
//!   ControlUnit::execute(op, ctx, tile)  <-- Control flow (this module)
//! ```
//!
//! Control operations return `ExecuteResult` variants (Branch, Halt, WaitLock,
//! etc.) that affect the interpreter's execution flow.
//!
//! # Operations
//!
//! - **Branch**: Conditional and unconditional branches
//! - **Call/Return**: Subroutine calls with link register
//! - **Lock**: Acquire/release synchronization primitives
//! - **DMA**: Start/wait for DMA transfers
//! - **Halt**: Stop core execution
//!
//! # Lock Operations (AIE2 Semaphore Model)
//!
//! AIE2 uses semaphore locks with 6-bit unsigned values (0-63).
//! Lock operations can specify expected values and deltas:
//!
//! - **Acquire**: Waits until `value >= expected`, then applies delta
//!   - Operands: lock_id, [expected_value], [delta]
//!   - Default: expected=1, delta=-1 (simple binary semaphore)
//!
//! - **Release**: Applies delta (non-blocking)
//!   - Operands: lock_id, [delta]
//!   - Default: delta=+1 (simple binary semaphore)
//!
//! See AM020 Ch2 and AM025 Lock_Request register for details.

use crate::device::tile::{Lock, Tile, LockResult};
use crate::interpreter::bundle::{BranchCondition, Operand, SlotOp};
use xdna_archspec::aie2::isa::SemanticOp;
use xdna_archspec::aie2::locks::quadrants;
use crate::interpreter::state::ExecutionContext;

/// Neighbor lock slices for cross-tile lock routing.
///
/// AIE2 compute tiles can access locks in all four cardinal neighbors:
/// - South (IDs 0-15): row-1 neighbor (typically MemTile)
/// - West  (IDs 16-31): col-1 neighbor
/// - North (IDs 32-47): row+1 neighbor
/// - East/Internal (IDs 48-63): own tile (handled separately)
///
/// Per mlir-aie AIETargetModel::getLockLocalBaseIndex().
pub struct NeighborLocks<'a> {
    pub south: Option<&'a mut [Lock]>,
    pub west: Option<&'a mut [Lock]>,
    pub north: Option<&'a mut [Lock]>,
}

impl<'a> NeighborLocks<'a> {
    /// Create with no neighbors (all quadrants fall back to own tile).
    pub fn none() -> Self {
        Self { south: None, west: None, north: None }
    }

    /// Create from just a South neighbor (backward-compatible with old API).
    pub fn south_only(locks: Option<&'a mut [Lock]>) -> Self {
        Self { south: locks, west: None, north: None }
    }
}
use crate::interpreter::traits::{ExecuteResult, Flags, StateAccess};

/// Control unit for branches, calls, and synchronization.
pub struct ControlUnit;

impl ControlUnit {
    /// Execute a control operation (without cross-tile lock routing).
    pub fn execute(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
    ) -> Option<ExecuteResult> {
        Self::execute_with_neighbor_locks(op, ctx, tile, &mut NeighborLocks::none())
    }

    /// Execute a control operation with optional cross-tile lock routing.
    ///
    /// AIE2 lock quadrant mapping (per mlir-aie getLockLocalBaseIndex):
    /// - IDs 0-15  (South): row-1 neighbor locks
    /// - IDs 16-31 (West):  col-1 neighbor locks
    /// - IDs 32-47 (North): row+1 neighbor locks
    /// - IDs 48-63 (East=Internal): own tile's memory module locks
    pub fn execute_with_neighbor_locks(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
        neighbor_locks: &mut NeighborLocks,
    ) -> Option<ExecuteResult> {
        match op.semantic {
            Some(SemanticOp::Br) | Some(SemanticOp::BrCond) => {
                let condition = op.branch_condition.unwrap_or(BranchCondition::Always);
                let target = Self::get_branch_target(op, ctx);
                let should_branch = match condition {
                    // For jz/jnz, the test register is in dest (decoder maps
                    // the single register operand there). Try dest, then sources.
                    BranchCondition::Zero | BranchCondition::NotZero => {
                        let reg_val = if let Some(Operand::ScalarReg(r)) = &op.dest {
                            let v = ctx.read_scalar(*r) as i32;
                            log::debug!("JNZ/JZ: dest=ScalarReg({}) val={} target=0x{:X}", r, v, target);
                            v
                        } else if let Some(Operand::ScalarReg(r)) = op.sources.first() {
                            let v = ctx.read_scalar(*r) as i32;
                            log::debug!("JNZ/JZ: src[0]=ScalarReg({}) val={} target=0x{:X}", r, v, target);
                            v
                        } else {
                            log::error!(
                                "JNZ/JZ: no register operand found, defaulting to 0 -- \
                                 likely a decoder bug (dest={:?}, sources={:?})",
                                op.dest, op.sources
                            );
                            0
                        };
                        match condition {
                            BranchCondition::Zero => reg_val == 0,
                            BranchCondition::NotZero => reg_val != 0,
                            _ => unreachable!(),
                        }
                    }
                    // JNZD: test sources[0] (mRx0), write dest = mRx0 - 1,
                    // branch to sources[1] (mPm) if mRx0 was nonzero.
                    // Decrement always happens regardless of branch direction.
                    //
                    // TableGen: (outs eR:$mRx), (ins eR:$mRx0, eP:$mPm)
                    // Encoding: {mRx0, mRx, mPm, opcode} -- separate fields.
                    //
                    // When mRx == mRx0 (common case from Peano), this is a
                    // simple decrement-in-place. When they differ (Chess
                    // compiler uses this for stack-bridged loop counters),
                    // dest gets source - 1 while source is unchanged.
                    BranchCondition::NotZeroDecrement => {
                        // Read the test register (sources[0] = mRx0)
                        let test_val = if let Some(Operand::ScalarReg(r)) = op.sources.first() {
                            let v = ctx.read_scalar(*r) as i32;
                            log::debug!("JNZD: test src[0]=r{} val={} target=0x{:X}", r, v, target);
                            v
                        } else if let Some(Operand::ScalarReg(r)) = &op.dest {
                            // Fallback: some decoder paths may put the register in dest
                            let v = ctx.read_scalar(*r) as i32;
                            log::debug!("JNZD: test dest=r{} val={} target=0x{:X}", r, v, target);
                            v
                        } else {
                            log::debug!("JNZD: no test register found, defaulting to 0");
                            0
                        };
                        // Write dest = test_val - 1 (the SOURCE value minus 1).
                        // Critical: when dest != source, we must decrement the
                        // source value, not re-read the dest register.
                        if let Some(Operand::ScalarReg(r)) = &op.dest {
                            let decremented = (test_val - 1) as u32;
                            ctx.write_scalar(*r, decremented);
                            log::debug!("JNZD: r{} = {} - 1 = {}", r, test_val, decremented);
                        }
                        test_val != 0
                    }
                    // For other conditions, check flags
                    _ => Self::evaluate_condition(condition, ctx.flags()),
                };
                if should_branch {
                    Some(ExecuteResult::Branch { target })
                } else {
                    Some(ExecuteResult::Continue)
                }
            }

            Some(SemanticOp::Call) => {
                let target = Self::get_branch_target(op, ctx);
                // Save return address (PC + instruction size handled by caller)
                // Link register will be set by the executor after this returns
                Some(ExecuteResult::Branch { target })
            }

            Some(SemanticOp::Ret) => {
                let target = ctx.lr();
                Some(ExecuteResult::Branch { target })
            }

            Some(SemanticOp::LockAcquire) => {
                let (raw_lock_id, expected, delta) = Self::get_lock_acquire_params(op, ctx);
                // AIE2 lock quadrant mapping (per mlir-aie getLockLocalBaseIndex
                // and AIE2TargetModel::isMemEast = isInternal):
                //
                // - IDs 0-15  (South): row-1 neighbor (MemTile for compute row 2)
                // - IDs 16-31 (West):  col-1 neighbor
                // - IDs 32-47 (North): row+1 neighbor
                // - IDs 48-63 (East=Internal): OWN tile's memory module
                //
                // The DMA BD lock field (4-bit) directly addresses tile-local
                // locks 0-15, matching the East/Internal quadrant (IDs 48-63).
                let (locks, lock_id, is_own_tile) = Self::route_lock(
                    raw_lock_id, tile, neighbor_locks,
                );

                let current_value = locks[lock_id as usize].value;
                let lock = &mut locks[lock_id as usize];

                // Choose acquire mode based on delta sign:
                //   delta < 0: acq_ge (wait until value >= expected)
                //   delta > 0: this would be acq_eq derived from positive change_value
                // The get_lock_acquire_params derivation ensures delta < 0 for acq_ge
                // and delta < 0 (= -expected) for acq_eq, so both use the same path.
                // The difference is the threshold: acq_ge checks >=, acq_eq checks ==.
                // We distinguish by checking if delta == -expected (acq_eq pattern).
                let result = if delta < 0 && expected > 0 && delta == -expected {
                    // Could be either acq_ge or acq_eq -- both derived the same way.
                    // Use acq_ge which is the more common mode. acq_eq would only
                    // matter when the lock value overshoots the expected value, which
                    // is rare in ObjectFIFO-generated code.
                    lock.acquire_with_value(expected, delta)
                } else {
                    lock.acquire_with_value(expected, delta)
                };

                match result {
                    LockResult::Success => {
                        log::trace!(
                            "[WATCH-ACQ] pc=0x{:03X} cycle={} lock={} value={}->{} SUCCESS",
                            ctx.pc(), ctx.cycles, raw_lock_id, current_value, lock.value
                        );
                        log::info!("LockAcquire raw={} mapped={} expected={} delta={} current={} -> {} SUCCESS (own_tile={})",
                            raw_lock_id, lock_id, expected, delta, current_value, lock.value, is_own_tile);
                        // Trace event: INSTR_LOCK_ACQUIRE_REQ (hw_id 44). HW fires
                        // this once per acquire instruction issued. We record only
                        // on Success so retries during WaitLock stalls don't inflate
                        // the count -- for tests that complete, every acquire
                        // eventually succeeds exactly once, matching HW 1:1.
                        let pc = ctx.pc();
                        let cycle = ctx.cycles;
                        ctx.timing_context_mut().record_event(
                            cycle,
                            crate::interpreter::state::EventType::InstrLockAcquireReq { pc },
                        );
                        Some(ExecuteResult::Continue)
                    }
                    LockResult::PreconditionNotMet => {
                        log::info!("LockAcquire raw={} mapped={} expected={} current={} -> WAIT (own_tile={})",
                            raw_lock_id, lock_id, expected, current_value, is_own_tile);
                        Some(ExecuteResult::WaitLock { raw_lock_id })
                    }
                    LockResult::WouldOverflow => {
                        // This shouldn't happen for acquire, but handle it
                        Some(ExecuteResult::Continue)
                    }
                }
            }

            Some(SemanticOp::LockRelease) => {
                let (raw_lock_id, delta) = Self::get_lock_release_params(op, ctx);
                let tile_col = tile.col;
                let tile_row = tile.row;

                // Same quadrant routing as LockAcquire.
                let (locks, lock_id, is_own_tile) = Self::route_lock(
                    raw_lock_id, tile, neighbor_locks,
                );

                if is_own_tile {
                    // Own-tile release: defer by 1 cycle to model hardware lock
                    // arbiter pipeline. The request is submitted to the arbiter
                    // and resolved in Phase 3, making it visible to DMA next cycle.
                    let old_value = locks[lock_id as usize].value;
                    tile.defer_core_lock_release(lock_id as usize, delta);
                    log::info!("LockRelease tile({},{}) raw={} mapped={} delta={} value={} DEFERRED (own_tile)",
                        tile_col, tile_row, raw_lock_id, lock_id, delta, old_value);
                } else {
                    // Cross-tile release (memtile path): the clone-modify-writeback
                    // mechanism in the coordinator handles deferral separately.
                    let old_value = locks[lock_id as usize].value;
                    let lock = &mut locks[lock_id as usize];
                    lock.release_with_value(delta);
                    log::info!("LockRelease tile({},{}) raw={} mapped={} delta={} {} -> {} (cross_tile)",
                        tile_col, tile_row, raw_lock_id, lock_id, delta, old_value, lock.value);
                }
                // Trace event: INSTR_LOCK_RELEASE_REQ (hw_id 45). HW fires this
                // once per release instruction issued; releases always complete
                // in the same cycle, so a single record matches HW 1:1.
                let pc = ctx.pc();
                let cycle = ctx.cycles;
                ctx.timing_context_mut().record_event(
                    cycle,
                    crate::interpreter::state::EventType::InstrLockReleaseReq { pc },
                );
                Some(ExecuteResult::Continue)
            }

            Some(SemanticOp::DmaStart) => {
                let channel = Self::get_dma_channel(op);
                let bd_index = Self::get_dma_bd(op);
                Self::start_dma(tile, channel, bd_index);
                Some(ExecuteResult::Continue)
            }

            Some(SemanticOp::DmaWait) => {
                let channel = Self::get_dma_channel(op);
                if Self::is_dma_complete(tile, channel) {
                    Some(ExecuteResult::Continue)
                } else {
                    Some(ExecuteResult::WaitDma { channel })
                }
            }

            Some(SemanticOp::Halt) | Some(SemanticOp::Done) => Some(ExecuteResult::Halt),

            Some(SemanticOp::Nop) => Some(ExecuteResult::Continue),

            _ => None, // Not a control operation
        }
    }

    /// Get branch target address from operand.
    ///
    /// For conditional branches (jnz/jz), the sources contain both a condition
    /// register and an immediate target address. We prefer the Immediate operand
    /// since that's the branch target; the register is the condition (already
    /// handled by the caller). For unconditional branches (jl), the first source
    /// is the Immediate target directly. For register-indirect branches (jl pN,
    /// ret lr, jnzd), a PointerReg operand provides the target.
    fn get_branch_target(op: &SlotOp, ctx: &ExecutionContext) -> u32 {
        // Prefer Immediate operand (the branch target address)
        for src in &op.sources {
            if let Operand::Immediate(v) = src {
                return *v as u32;
            }
        }
        // Check for pointer register (register-indirect branch: jl pN, jnzd ..pN)
        for src in &op.sources {
            if let Operand::PointerReg(r) = src {
                return ctx.pointer.read(*r);
            }
        }
        // Last resort: scalar register (e.g., jr rX)
        for src in &op.sources {
            if let Operand::ScalarReg(r) = src {
                return ctx.scalar_read(*r);
            }
        }
        log::error!(
            "[CONTROL] get_branch_target: no target operand found, defaulting to 0x0 -- \
             likely a decoder bug (sources: {:?})",
            op.sources
        );
        0
    }

    /// Evaluate a branch condition against flags.
    ///
    /// Note: Zero and NotZero are register-based conditions handled in execute(),
    /// not flag-based. They should not reach this function.
    pub fn evaluate_condition(condition: BranchCondition, flags: Flags) -> bool {
        match condition {
            BranchCondition::Always => true,
            BranchCondition::Equal => flags.z,
            BranchCondition::NotEqual => !flags.z,
            BranchCondition::Less => flags.n != flags.v, // Signed less
            BranchCondition::GreaterEqual => flags.n == flags.v, // Signed >=
            BranchCondition::LessEqual => flags.z || (flags.n != flags.v), // Signed <=
            BranchCondition::Greater => !flags.z && (flags.n == flags.v), // Signed >
            BranchCondition::Negative => flags.n,
            BranchCondition::PositiveOrZero => !flags.n,
            BranchCondition::CarrySet => flags.c,
            BranchCondition::CarryClear => !flags.c,
            BranchCondition::OverflowSet => flags.v,
            BranchCondition::OverflowClear => !flags.v,
            // These are handled directly in execute() using register values
            BranchCondition::Zero
            | BranchCondition::NotZero
            | BranchCondition::NotZeroDecrement => {
                panic!(
                    "Zero/NotZero/NotZeroDecrement condition reached evaluate_condition() -- \
                     must be handled in execute() using register values"
                );
            }
        }
    }

    /// Get lock acquire parameters from operands.
    ///
    /// Returns (lock_id, expected_value, delta).
    /// Default: expected=1, delta=-1 (simple binary semaphore).
    ///
    /// AIE2 has two ACQ instruction variants:
    /// - ACQ_mLockId_imm: lock ID is a 6-bit immediate (Operand::Immediate or Lock)
    /// - ACQ_mLockId_reg: lock ID is in a scalar register (Operand::ScalarReg)
    /// Both variants also take a register (mRy) for the lock value parameter.
    fn get_lock_acquire_params(op: &SlotOp, ctx: &ExecutionContext) -> (u8, i8, i8) {
        // Lock ID from first operand -- immediate or register
        let lock_id = op.sources.first().map_or(0, |src| match src {
            Operand::Lock(id) => *id,
            Operand::Immediate(v) => *v as u8,
            Operand::ScalarReg(r) => ctx.scalar_read(*r) as u8,
            other => {
                log::warn!("[LOCK] ACQ: unexpected lock_id operand {:?}, defaulting to 0", other);
                0
            }
        });

        // Change value from second operand.
        //
        // AIE2 lock acquire instructions (ACQ_mLockId_imm, ACQ_mLockId_reg)
        // encode the change value in the second operand. The semantics match
        // the BD lock field and handle_lock_request:
        //
        //   change < 0: acq_ge -- expected = |change|, delta = change
        //   change > 0: acq_eq -- expected = change, delta = -change
        //   change == 0: simple acquire -- expected = 1, delta = -1
        //
        // The operand is either an Immediate (for constant values) or a
        // ScalarReg (when the compiler loads the value into a register).
        let change_value = op.sources.get(1).map_or(-1_i8, |src| match src {
            Operand::Immediate(v) => *v as i8,
            Operand::ScalarReg(r) => ctx.scalar_read(*r) as i8,
            other => {
                log::warn!("[LOCK] ACQ: unexpected change_value operand {:?}, defaulting to -1", other);
                -1
            }
        });

        let (expected, delta) = if change_value < 0 {
            ((-change_value) as i8, change_value)
        } else if change_value > 0 {
            (change_value, -change_value)
        } else {
            (1_i8, -1_i8)
        };

        (lock_id, expected, delta)
    }

    /// Get lock release parameters from operands.
    ///
    /// Returns (lock_id, delta).
    ///
    /// AIE2 lock release instructions (REL_mLockId_imm, REL_mLockId_reg)
    /// encode the delta in the second operand. The operand is either an
    /// Immediate (for constant values like +1) or a ScalarReg (when the
    /// compiler loads the delta into a register, e.g., for repeat_count > 1
    /// where delta = repeat_count).
    fn get_lock_release_params(op: &SlotOp, ctx: &ExecutionContext) -> (u8, i8) {
        // Lock ID from first operand -- immediate or register
        let lock_id = op.sources.first().map_or(0, |src| match src {
            Operand::Lock(id) => *id,
            Operand::Immediate(v) => *v as u8,
            Operand::ScalarReg(r) => ctx.scalar_read(*r) as u8,
            other => {
                log::warn!("[LOCK] REL: unexpected lock_id operand {:?}, defaulting to 0", other);
                0
            }
        });

        // Delta from second operand (default: +1)
        let delta = op.sources.get(1).map_or(1, |src| match src {
            Operand::Immediate(v) => *v as i8,
            Operand::ScalarReg(r) => ctx.scalar_read(*r) as i8,
            other => {
                log::warn!("[LOCK] REL: unexpected delta operand {:?}, defaulting to +1", other);
                1
            }
        });

        (lock_id, delta)
    }

    /// Get DMA channel from operand.
    fn get_dma_channel(op: &SlotOp) -> u8 {
        op.sources.first().map_or(0, |src| match src {
            Operand::DmaChannel(ch) => *ch,
            Operand::Immediate(v) => *v as u8,
            other => {
                log::warn!("[DMA] get_dma_channel: unexpected operand {:?}, defaulting to ch 0", other);
                0
            }
        })
    }

    /// Route a core lock ID to the correct lock array and local index.
    ///
    /// AIE2 lock quadrant mapping (per mlir-aie getLockLocalBaseIndex
    /// and AIE2TargetModel::isMemEast which returns isInternal):
    ///
    /// - IDs 0-15  (South): row-1 neighbor's locks
    /// - IDs 16-31 (West):  col-1 neighbor's locks
    /// - IDs 32-47 (North): row+1 neighbor's locks
    /// - IDs 48-63 (East=Internal): OWN tile's memory module locks
    ///
    /// The DMA BD lock field (4-bit, values 0-15) addresses the same
    /// physical locks as the East/Internal quadrant (IDs 48-63).
    ///
    /// Returns (lock_slice, local_lock_id, is_own_tile).
    fn route_lock<'a>(
        raw_lock_id: u8,
        tile: &'a mut Tile,
        neighbor_locks: &'a mut NeighborLocks,
    ) -> (&'a mut [Lock], u8, bool) {
        if raw_lock_id >= quadrants::EAST_START {
            // East = Internal = own tile's memory module locks.
            let id = (raw_lock_id - quadrants::EAST_START) % tile.locks.len() as u8;
            (&mut tile.locks, id, true)
        } else if raw_lock_id < quadrants::SOUTH_END {
            // South = row-1 neighbor.
            if let Some(ref mut locks) = neighbor_locks.south {
                let id = raw_lock_id % locks.len() as u8;
                (&mut **locks, id, false)
            } else {
                log::warn!("Lock ID {} targets South neighbor but no South locks available", raw_lock_id);
                let id = raw_lock_id % tile.locks.len() as u8;
                (&mut tile.locks, id, true)
            }
        } else if raw_lock_id < quadrants::WEST_END {
            // West = col-1 neighbor.
            if let Some(ref mut locks) = neighbor_locks.west {
                let id = (raw_lock_id - quadrants::WEST_START) % locks.len() as u8;
                (&mut **locks, id, false)
            } else {
                log::warn!("Lock ID {} targets West neighbor but no West locks available", raw_lock_id);
                let id = (raw_lock_id - quadrants::WEST_START) % tile.locks.len() as u8;
                (&mut tile.locks, id, true)
            }
        } else {
            // North = row+1 neighbor (IDs 32-47).
            if let Some(ref mut locks) = neighbor_locks.north {
                let id = (raw_lock_id - quadrants::NORTH_START) % locks.len() as u8;
                (&mut **locks, id, false)
            } else {
                log::warn!("Lock ID {} targets North neighbor but no North locks available", raw_lock_id);
                let id = (raw_lock_id - quadrants::NORTH_START) % tile.locks.len() as u8;
                (&mut tile.locks, id, true)
            }
        }
    }

    /// Start a DMA transfer.
    ///
    /// This sets up the start request in tile.dma_channels[]. The interpreter
    /// is responsible for syncing this to the actual DmaEngine and stepping it.
    ///
    /// The start_queue field holds the BD index (from immediate operand).
    /// When the interpreter sees running=true and start_queue is set, it calls
    /// DmaEngine.start_channel(channel, start_queue).
    fn start_dma(tile: &mut Tile, channel: u8, bd_index: u8) {
        let ch = &mut tile.dma_channels[channel as usize & 0x03];
        ch.running = true;
        ch.start_queue = bd_index as u32;
    }

    /// Get the BD index for a DMA start operation.
    fn get_dma_bd(op: &SlotOp) -> u8 {
        // BD index is typically in second operand or immediate
        op.sources.get(1).map_or(0, |src| match src {
            Operand::Immediate(v) => *v as u8,
            other => {
                log::warn!("[DMA] get_dma_bd: unexpected operand {:?}, defaulting to BD 0", other);
                0
            }
        })
    }

    /// Check if DMA transfer is complete.
    ///
    /// Returns true if channel is not marked as running.
    /// The interpreter is responsible for clearing running=false when
    /// the DmaEngine reports the channel as complete.
    fn is_dma_complete(tile: &Tile, channel: u8) -> bool {
        !tile.dma_channels[channel as usize & 0x03].running
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::SlotIndex;
    use xdna_archspec::aie2::isa::SemanticOp;

    fn make_ctx() -> ExecutionContext {
        ExecutionContext::new()
    }

    fn make_tile() -> Tile {
        Tile::compute(0, 2)
    }

    #[test]
    fn test_unconditional_branch() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::Br)
            .with_branch_condition(BranchCondition::Always)
            .with_source(Operand::Immediate(0x1000));

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Branch { target: 0x1000 })));
    }

    #[test]
    fn test_conditional_branch_taken() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Set zero flag
        ctx.set_flags(Flags {
            z: true,
            n: false,
            c: false,
            v: false,
        });

        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::BrCond)
            .with_branch_condition(BranchCondition::Equal)
            .with_source(Operand::Immediate(0x2000));

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Branch { target: 0x2000 })));
    }

    #[test]
    fn test_conditional_branch_not_taken() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Zero flag not set
        ctx.set_flags(Flags {
            z: false,
            n: false,
            c: false,
            v: false,
        });

        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::BrCond)
            .with_branch_condition(BranchCondition::Equal)
            .with_source(Operand::Immediate(0x2000));

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));
    }

    #[test]
    fn test_return() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        ctx.set_lr(0x5000);

        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::Ret);

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Branch { target: 0x5000 })));
    }

    #[test]
    fn test_jnzd_same_register_nonzero() {
        // jnzd r5, r5, p0 -- common case (Peano compiler)
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        ctx.write_scalar(5, 3); // r5 = 3
        ctx.pointer.write(0, 0x1b0); // p0 = branch target

        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::BrCond)
            .with_branch_condition(BranchCondition::NotZeroDecrement)
            .with_dest(Operand::ScalarReg(5))          // mRx = r5
            .with_source(Operand::ScalarReg(5))         // mRx0 = r5
            .with_source(Operand::PointerReg(0));       // mPm = p0

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Branch { target: 0x1b0 })));
        assert_eq!(ctx.read_scalar(5), 2); // r5 = 3 - 1 = 2
    }

    #[test]
    fn test_jnzd_same_register_zero() {
        // jnzd r5, r5, p0 -- falls through when r5 = 0
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        ctx.write_scalar(5, 0); // r5 = 0
        ctx.pointer.write(0, 0x1b0);

        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::BrCond)
            .with_branch_condition(BranchCondition::NotZeroDecrement)
            .with_dest(Operand::ScalarReg(5))
            .with_source(Operand::ScalarReg(5))
            .with_source(Operand::PointerReg(0));

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));
        assert_eq!(ctx.read_scalar(5), 0xFFFFFFFF); // 0 - 1 wraps
    }

    #[test]
    fn test_jnzd_split_register() {
        // jnzd r20, r24, p1 -- Chess compiler pattern: different dest/source.
        // dest gets source - 1; source is tested but NOT modified.
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        ctx.write_scalar(24, 3);     // r24 = 3 (source, tested)
        ctx.write_scalar(20, 0x99);  // r20 = 0x99 (dest, gets r24 - 1)
        ctx.pointer.write(1, 0x190); // p1 = branch target

        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::BrCond)
            .with_branch_condition(BranchCondition::NotZeroDecrement)
            .with_dest(Operand::ScalarReg(20))           // mRx = r20
            .with_source(Operand::ScalarReg(24))          // mRx0 = r24
            .with_source(Operand::PointerReg(1));         // mPm = p1

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Branch { target: 0x190 })));
        assert_eq!(ctx.read_scalar(20), 2, "dest = source - 1 = 3 - 1 = 2");
        assert_eq!(ctx.read_scalar(24), 3, "source unchanged");
    }

    #[test]
    fn test_jnzd_split_register_zero() {
        // jnzd r20, r24, p1 -- source = 0, falls through
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        ctx.write_scalar(24, 0);      // r24 = 0 (source)
        ctx.write_scalar(20, 0x42);   // r20 = junk (dest)
        ctx.pointer.write(1, 0x190);

        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::BrCond)
            .with_branch_condition(BranchCondition::NotZeroDecrement)
            .with_dest(Operand::ScalarReg(20))
            .with_source(Operand::ScalarReg(24))
            .with_source(Operand::PointerReg(1));

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));
        assert_eq!(ctx.read_scalar(20), 0xFFFFFFFF, "dest = 0 - 1 wraps");
        assert_eq!(ctx.read_scalar(24), 0, "source unchanged");
    }

    #[test]
    fn test_lock_acquire_success() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Initialize lock with value 1 (available)
        tile.locks[5].value = 1;

        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::LockAcquire)
            .with_source(Operand::Lock(5));

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));
        assert_eq!(tile.locks[5].value, 0); // Lock acquired
    }

    #[test]
    fn test_lock_acquire_blocked() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Lock value 0 (unavailable)
        tile.locks[5].value = 0;

        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::LockAcquire)
            .with_source(Operand::Lock(5));

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::WaitLock { raw_lock_id: 5 })));
    }

    #[test]
    fn test_lock_release() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        tile.locks[3].value = 0;

        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::LockRelease)
            .with_source(Operand::Lock(3));

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));
        // Release is deferred: submitted to arbiter, not yet applied
        assert_eq!(tile.locks[3].value, 0, "Not yet committed (pending in arbiter)");
        // Resolve the arbiter to apply the deferred release
        tile.resolve_lock_requests(0);
        assert_eq!(tile.locks[3].value, 1, "Committed after arbiter resolve");
    }

    #[test]
    fn test_lock_acquire_with_change_value() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Initialize lock with value 5
        tile.locks[2].value = 5;

        // Acquire with change_value=-3 (acq_ge: expected=3, delta=-3)
        // Matches hardware: acq #lock_id, r_change where r_change = -3
        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::LockAcquire)
            .with_source(Operand::Lock(2))
            .with_source(Operand::Immediate(-3)); // change_value

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));
        assert_eq!(tile.locks[2].value, 2); // 5 + (-3) = 2
    }

    #[test]
    fn test_lock_acquire_with_change_value_blocked() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Initialize lock with value 2
        tile.locks[7].value = 2;

        // Acquire with change_value=-5 (acq_ge: expected=5, delta=-5)
        // Should fail since lock value 2 < 5
        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::LockAcquire)
            .with_source(Operand::Lock(7))
            .with_source(Operand::Immediate(-5)); // change_value

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::WaitLock { raw_lock_id: 7 })));
        assert_eq!(tile.locks[7].value, 2); // Unchanged
    }

    #[test]
    fn test_lock_acquire_register_change_value() {
        // Test the exact pattern from compute_repeat:
        // acq #0x32, r17 where r17 = -4
        // This acquires lock 2 with acq_ge(4), delta=-4
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Lock 2 initialized to 4 (matching compute_repeat CDO)
        tile.locks[2].value = 4;
        ctx.scalar.write(17, (-4_i32) as u32); // r17 = -4

        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::LockAcquire)
            .with_source(Operand::Lock(2))
            .with_source(Operand::ScalarReg(17)); // change_value from register

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));
        assert_eq!(tile.locks[2].value, 0, "Lock should go from 4 to 0 (4 + (-4))");
    }

    #[test]
    fn test_lock_release_register_delta() {
        // Test the exact pattern from compute_repeat:
        // rel #0x33, r20 where r20 = 4
        // This releases lock 3 with delta=+4
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        tile.locks[3].value = 0;
        ctx.scalar.write(20, 4); // r20 = 4

        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::LockRelease)
            .with_source(Operand::Lock(3))
            .with_source(Operand::ScalarReg(20)); // delta from register

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));
        assert_eq!(tile.locks[3].value, 0, "Not yet committed (pending in arbiter)");
        tile.resolve_lock_requests(0);
        assert_eq!(tile.locks[3].value, 4, "Committed after arbiter resolve");
    }

    #[test]
    fn test_lock_release_with_delta() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        tile.locks[4].value = 5;

        // Release with delta=3
        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::LockRelease)
            .with_source(Operand::Lock(4))
            .with_source(Operand::Immediate(3)); // delta

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));
        assert_eq!(tile.locks[4].value, 5, "Not yet committed (pending in arbiter)");
        tile.resolve_lock_requests(0);
        assert_eq!(tile.locks[4].value, 8, "5 + 3 = 8 after arbiter resolve");
    }

    #[test]
    fn test_lock_release_saturates() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        tile.locks[6].value = 60;

        // Release with delta=10 (should saturate at 63)
        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::LockRelease)
            .with_source(Operand::Lock(6))
            .with_source(Operand::Immediate(10)); // delta

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));
        assert_eq!(tile.locks[6].value, 60, "Not yet committed (pending in arbiter)");
        tile.resolve_lock_requests(0);
        assert_eq!(tile.locks[6].value, 63, "Committed saturated at MAX");
        assert!(tile.locks[6].overflow, "Overflow flag set");
    }

    #[test]
    fn test_lock_acquire_register_lock_id() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Set r0 = 49 (memory module lock 1, maps to tile.locks[1])
        ctx.scalar.write(0, 49);
        tile.locks[1].value = 1;

        // ACQ_mLockId_reg uses ScalarReg for lock ID
        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::LockAcquire)
            .with_source(Operand::ScalarReg(0)); // lock ID from register r0

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));
        assert_eq!(tile.locks[1].value, 0); // Lock 1 acquired (mapped from 49)
    }

    #[test]
    fn test_lock_release_register_lock_id() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Set r0 = 48 (memory module lock 0, maps to tile.locks[0])
        ctx.scalar.write(0, 48);
        tile.locks[0].value = 0;

        // REL_mLockId_reg uses ScalarReg for lock ID
        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::LockRelease)
            .with_source(Operand::ScalarReg(0)); // lock ID from register r0

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));
        // Lock ID 48 maps to own-tile lock 0 (East/Internal quadrant), deferred
        assert_eq!(tile.locks[0].value, 0, "Not yet committed (pending in arbiter)");
        tile.resolve_lock_requests(0);
        assert_eq!(tile.locks[0].value, 1, "Committed after arbiter resolve");
    }

    #[test]
    fn test_lock_release_and_acquire_same_cycle() {
        // Verify that a core lock release and DMA acquire targeting the
        // same lock both succeed in the same arbiter resolution cycle.
        // Releases are non-blocking on real hardware and must not prevent
        // a same-cycle acquire from seeing the updated value.
        use crate::device::tile::{LockRequest, LockRequestor};

        let mut tile = make_tile();
        let mut ctx = make_ctx();

        tile.locks[5].value = 0;

        // Phase 2: Core releases lock 5 (submits to arbiter, not yet applied)
        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::LockRelease)
            .with_source(Operand::Lock(5));
        ControlUnit::execute(&op, &mut ctx, &mut tile);

        // Lock value is still 0 (release pending in arbiter)
        assert_eq!(tile.locks[5].value, 0, "Release pending, not yet applied");

        // Simulate DMA submitting an acquire for lock 5 (needs >= 1)
        // This goes into the same arbiter batch as the core release
        tile.submit_lock_request(LockRequest {
            requestor: LockRequestor::DmaS2mm(0),
            lock_id: 5,
            is_acquire: true,
            expected: 1,
            delta: -1,
            equal_mode: false,
        });

        // Resolve arbiter: release applied first (lock 0 -> 1), then
        // acquire sees updated value (1 >= 1, granted, lock 1 -> 0).
        tile.resolve_lock_requests(0);

        // Both succeeded in the same cycle: release +1 then acquire -1
        assert_eq!(tile.locks[5].value, 0, "Release then acquire in same cycle");
    }

    #[test]
    fn test_halt() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::Halt);

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Halt)));
    }

    #[test]
    fn test_nop() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let op = SlotOp::nop(SlotIndex::Control);

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));
    }

    #[test]
    fn test_condition_evaluation() {
        // Equal
        assert!(ControlUnit::evaluate_condition(
            BranchCondition::Equal,
            Flags { z: true, n: false, c: false, v: false }
        ));
        assert!(!ControlUnit::evaluate_condition(
            BranchCondition::Equal,
            Flags { z: false, n: false, c: false, v: false }
        ));

        // Less (signed: n != v)
        assert!(ControlUnit::evaluate_condition(
            BranchCondition::Less,
            Flags { z: false, n: true, c: false, v: false }
        ));
        assert!(!ControlUnit::evaluate_condition(
            BranchCondition::Less,
            Flags { z: false, n: true, c: false, v: true }
        ));

        // Greater (signed: !z && n == v)
        assert!(ControlUnit::evaluate_condition(
            BranchCondition::Greater,
            Flags { z: false, n: false, c: false, v: false }
        ));
        assert!(!ControlUnit::evaluate_condition(
            BranchCondition::Greater,
            Flags { z: true, n: false, c: false, v: false }
        ));
    }

    #[test]
    fn test_dma_start() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Start DMA on channel 0 with BD 5
        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::DmaStart)
            .with_source(Operand::DmaChannel(0))
            .with_source(Operand::Immediate(5)); // BD index

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));

        // Verify channel state was set up for interpreter to sync
        assert!(tile.dma_channels[0].running);
        assert_eq!(tile.dma_channels[0].start_queue, 5);
    }

    #[test]
    fn test_dma_wait_blocks() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Set channel as running (simulates DMA in progress)
        tile.dma_channels[1].running = true;

        // DmaWait should return WaitDma since channel is busy
        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::DmaWait)
            .with_source(Operand::DmaChannel(1));

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::WaitDma { channel: 1 })));
    }

    #[test]
    fn test_dma_wait_completes() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Channel not running (transfer complete)
        tile.dma_channels[2].running = false;

        // DmaWait should succeed
        let op = SlotOp::from_semantic(SlotIndex::Control, SemanticOp::DmaWait)
            .with_source(Operand::DmaChannel(2));

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));
    }
}
