//! Core interpreter implementation.
//!
//! The interpreter manages the fetch-decode-execute loop for a single AIE2 core.

use crate::device::tile::Tile;
use crate::interpreter::bundle::VliwBundle;
use crate::interpreter::decode::InstructionDecoder;
use crate::interpreter::execute::FastExecutor;
use crate::interpreter::state::ExecutionContext;
use crate::interpreter::traits::{DecodeError, Decoder, ExecuteResult, Executor};

/// Core execution status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum CoreStatus {
    /// Core is ready to execute.
    #[default]
    Ready,
    /// Core is running.
    Running,
    /// Core is waiting on lock acquisition.
    WaitingLock { lock_id: u8 },
    /// Core is waiting on DMA completion.
    WaitingDma { channel: u8 },
    /// Core is waiting on stream data (blocking read with empty buffer).
    WaitingStream { port: u8 },
    /// Core has halted (normal termination).
    Halted,
    /// Core encountered an error.
    Error,
}


/// Result of a single step execution.
#[derive(Debug, Clone)]
pub enum StepResult {
    /// Continue executing next instruction.
    Continue,
    /// Stalled waiting on lock.
    WaitLock { lock_id: u8 },
    /// Stalled waiting on DMA.
    WaitDma { channel: u8 },
    /// Stalled waiting on stream data.
    WaitStream { port: u8 },
    /// Core halted.
    Halt,
    /// Decode error.
    DecodeError(DecodeError),
    /// Execution error.
    ExecError(String),
}

/// Per-core interpreter.
///
/// Manages the execution of a single AIE2 compute core by coordinating
/// the decoder and executor.
///
/// # Default Configuration
///
/// The default configuration uses:
/// - `InstructionDecoder`: O(1) instruction decoder from llvm-aie TableGen files
/// - `FastExecutor`: Non-cycle-accurate fast executor
///
/// If llvm-aie is not available, the decoder will be empty and return
/// unknown operations for all instructions.
pub struct CoreInterpreter<D = InstructionDecoder, E = FastExecutor>
where
    D: Decoder,
    E: Executor,
{
    /// Instruction decoder.
    decoder: D,
    /// Execution unit.
    executor: E,
    /// Current core status.
    status: CoreStatus,
    /// Last decoded bundle (for debugging).
    last_bundle: Option<VliwBundle>,
}

impl CoreInterpreter<InstructionDecoder, FastExecutor> {
    /// Create a new interpreter with default decoder and executor.
    ///
    /// Uses InstructionDecoder loaded from llvm-aie at ../llvm-aie.
    /// Falls back to an empty decoder if llvm-aie is not found.
    pub fn default_new() -> Self {
        Self::new(InstructionDecoder::load_default(), FastExecutor::new())
    }
}

impl<D, E> CoreInterpreter<D, E>
where
    D: Decoder,
    E: Executor,
{
    /// Create a new interpreter with the given decoder and executor.
    pub fn new(decoder: D, executor: E) -> Self {
        Self {
            decoder,
            executor,
            status: CoreStatus::Ready,
            last_bundle: None,
        }
    }

    /// Get the current core status.
    pub fn status(&self) -> CoreStatus {
        self.status
    }

    /// Check if the core is halted.
    pub fn is_halted(&self) -> bool {
        matches!(self.status, CoreStatus::Halted)
    }

    /// Check if the core is stalled (waiting on lock or DMA).
    pub fn is_stalled(&self) -> bool {
        matches!(
            self.status,
            CoreStatus::WaitingLock { .. } | CoreStatus::WaitingDma { .. }
        )
    }

    /// Get the last decoded bundle (for debugging).
    pub fn last_bundle(&self) -> Option<&VliwBundle> {
        self.last_bundle.as_ref()
    }

    /// Execute a single instruction cycle.
    ///
    /// Returns the result of execution which indicates how to proceed.
    pub fn step(&mut self, ctx: &mut ExecutionContext, tile: &mut Tile) -> StepResult {
        // Check if halted
        if self.is_halted() {
            return StepResult::Halt;
        }

        // Try to resume from stall
        if let Some(result) = self.try_resume_stall(ctx, tile) {
            return result;
        }

        // Fetch instruction bytes from program memory
        let pc = ctx.pc();
        let program_mem = match tile.program_memory() {
            Some(mem) => mem,
            None => {
                self.status = CoreStatus::Error;
                return StepResult::ExecError("No program memory".to_string());
            }
        };

        // Check PC bounds
        let pc_offset = pc as usize;
        if pc_offset >= program_mem.len() {
            self.status = CoreStatus::Halted;
            return StepResult::Halt;
        }

        // Get instruction bytes (maximum 16 for full VLIW)
        let end = (pc_offset + 16).min(program_mem.len());
        let bytes = &program_mem[pc_offset..end];

        // Decode instruction
        let bundle = match self.decoder.decode(bytes, pc) {
            Ok(b) => b,
            Err(e) => {
                self.status = CoreStatus::Error;
                return StepResult::DecodeError(e);
            }
        };

        let bundle_size = bundle.size();

        // Execute bundle
        self.status = CoreStatus::Running;
        let result = self.executor.execute(&bundle, ctx, tile);

        // Save bundle for debugging
        self.last_bundle = Some(bundle);

        // Handle execution result
        match result {
            ExecuteResult::Continue => {
                ctx.advance_pc(bundle_size as u32);
                self.status = CoreStatus::Ready;
                StepResult::Continue
            }

            ExecuteResult::Branch { target } => {
                ctx.set_pc(target);
                self.status = CoreStatus::Ready;
                StepResult::Continue
            }

            ExecuteResult::WaitLock { lock_id } => {
                self.status = CoreStatus::WaitingLock { lock_id };
                ctx.record_stall(1);
                StepResult::WaitLock { lock_id }
            }

            ExecuteResult::WaitDma { channel } => {
                self.status = CoreStatus::WaitingDma { channel };
                ctx.record_stall(1);
                StepResult::WaitDma { channel }
            }

            ExecuteResult::WaitStream { port } => {
                self.status = CoreStatus::WaitingStream { port };
                ctx.record_stall(1);
                StepResult::WaitStream { port }
            }

            ExecuteResult::Halt => {
                self.status = CoreStatus::Halted;
                ctx.halted = true;
                StepResult::Halt
            }

            ExecuteResult::Error { message } => {
                self.status = CoreStatus::Error;
                StepResult::ExecError(message)
            }
        }
    }

    /// Try to resume from a stall condition.
    ///
    /// Returns `Some(result)` if still stalled, `None` if resumed.
    fn try_resume_stall(&mut self, ctx: &mut ExecutionContext, tile: &mut Tile) -> Option<StepResult> {
        match self.status {
            CoreStatus::WaitingLock { lock_id } => {
                // Check if lock is now available (don't acquire yet - instruction will do that)
                let lock_value = tile.locks[lock_id as usize].value;
                log::trace!("try_resume_stall: lock {} value = {}", lock_id, lock_value);
                if lock_value > 0 {
                    log::info!("Lock {} available (value={}), resuming execution", lock_id, lock_value);
                    self.status = CoreStatus::Ready;
                    // Re-execute the instruction - it will acquire the lock
                    None
                } else {
                    ctx.record_stall(1);
                    Some(StepResult::WaitLock { lock_id })
                }
            }

            CoreStatus::WaitingDma { channel } => {
                // Check if DMA is complete
                if !tile.dma_channels[channel as usize].running {
                    self.status = CoreStatus::Ready;
                    None
                } else {
                    ctx.record_stall(1);
                    Some(StepResult::WaitDma { channel })
                }
            }

            CoreStatus::WaitingStream { port } => {
                // Check if stream data is available
                if tile.has_stream_input(port) {
                    self.status = CoreStatus::Ready;
                    // Re-execute the instruction now that data is available
                    None
                } else {
                    ctx.record_stall(1);
                    Some(StepResult::WaitStream { port })
                }
            }

            _ => None,
        }
    }

    /// Run for up to `max_cycles` cycles or until halt/error.
    ///
    /// Returns the final step result and number of cycles executed.
    pub fn run(&mut self, ctx: &mut ExecutionContext, tile: &mut Tile, max_cycles: u64) -> (StepResult, u64) {
        let start_cycles = ctx.cycles;

        for _ in 0..max_cycles {
            match self.step(ctx, tile) {
                StepResult::Continue => continue,
                result @ StepResult::Halt => return (result, ctx.cycles - start_cycles),
                result @ StepResult::DecodeError(_) => return (result, ctx.cycles - start_cycles),
                result @ StepResult::ExecError(_) => return (result, ctx.cycles - start_cycles),
                // On stall, count as one cycle and continue
                StepResult::WaitLock { .. } | StepResult::WaitDma { .. } | StepResult::WaitStream { .. } => continue,
            }
        }

        (StepResult::Continue, ctx.cycles - start_cycles)
    }

    /// Reset the interpreter state.
    pub fn reset(&mut self) {
        self.status = CoreStatus::Ready;
        self.last_bundle = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::{Operation, Operand, SlotIndex, SlotOp, MemWidth, PostModify, BranchCondition};

    fn make_interpreter() -> CoreInterpreter {
        CoreInterpreter::default_new()
    }

    fn make_tile_with_program(program: &[u8]) -> Tile {
        let mut tile = Tile::compute(0, 2);
        assert!(tile.write_program(0, program));
        tile
    }

    #[test]
    fn test_step_nop() {
        let mut interpreter = make_interpreter();
        let mut ctx = ExecutionContext::new();
        let mut tile = make_tile_with_program(&[0x00, 0x00, 0x00, 0x00]);

        let result = interpreter.step(&mut ctx, &mut tile);

        assert!(matches!(result, StepResult::Continue));
        assert_eq!(ctx.pc(), 4);
        assert_eq!(ctx.instructions, 1);
        assert_eq!(interpreter.status(), CoreStatus::Ready);
    }

    #[test]
    fn test_step_multiple_nops() {
        let mut interpreter = make_interpreter();
        let mut ctx = ExecutionContext::new();
        // 4 NOP instructions
        let mut tile = make_tile_with_program(&[
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ]);

        for i in 0..4 {
            let result = interpreter.step(&mut ctx, &mut tile);
            assert!(matches!(result, StepResult::Continue));
            assert_eq!(ctx.pc(), (i + 1) * 4);
        }

        assert_eq!(ctx.instructions, 4);
    }

    #[test]
    fn test_run_with_limit() {
        let mut interpreter = make_interpreter();
        let mut ctx = ExecutionContext::new();
        // Many NOPs
        let mut tile = make_tile_with_program(&[0x00u8; 1024]);

        let (result, cycles) = interpreter.run(&mut ctx, &mut tile, 10);

        assert!(matches!(result, StepResult::Continue));
        assert_eq!(cycles, 10);
        assert_eq!(ctx.pc(), 40); // 10 instructions * 4 bytes
    }

    #[test]
    fn test_pc_out_of_bounds_halts() {
        let mut interpreter = make_interpreter();
        let mut ctx = ExecutionContext::new();
        ctx.set_pc(0x20000); // Beyond program memory

        let mut tile = make_tile_with_program(&[0x00; 16]);

        let result = interpreter.step(&mut ctx, &mut tile);

        assert!(matches!(result, StepResult::Halt));
        assert!(interpreter.is_halted());
    }

    #[test]
    fn test_halted_stays_halted() {
        let mut interpreter = make_interpreter();
        let mut ctx = ExecutionContext::new();
        let mut tile = make_tile_with_program(&[0x00; 16]);

        // Manually halt
        ctx.set_pc(0xFFFFFF);
        let _ = interpreter.step(&mut ctx, &mut tile);

        // Should remain halted
        let result = interpreter.step(&mut ctx, &mut tile);
        assert!(matches!(result, StepResult::Halt));
    }

    #[test]
    fn test_status_transitions() {
        let mut interpreter = make_interpreter();

        assert_eq!(interpreter.status(), CoreStatus::Ready);
        assert!(!interpreter.is_halted());
        assert!(!interpreter.is_stalled());
    }

    #[test]
    fn test_reset() {
        let mut interpreter = make_interpreter();
        let mut ctx = ExecutionContext::new();
        let mut tile = make_tile_with_program(&[0x00; 16]);

        // Run some steps
        interpreter.step(&mut ctx, &mut tile);

        // Reset
        interpreter.reset();

        assert_eq!(interpreter.status(), CoreStatus::Ready);
        assert!(interpreter.last_bundle().is_none());
    }

    #[test]
    fn test_last_bundle_preserved() {
        let mut interpreter = make_interpreter();
        let mut ctx = ExecutionContext::new();
        let mut tile = make_tile_with_program(&[0x00, 0x00, 0x00, 0x00]);

        interpreter.step(&mut ctx, &mut tile);

        let bundle = interpreter.last_bundle().expect("Should have last bundle");
        assert_eq!(bundle.size(), 4);
    }

    #[test]
    fn test_no_program_memory_error() {
        let mut interpreter = make_interpreter();
        let mut ctx = ExecutionContext::new();
        // Shim tile has no program memory
        let mut tile = Tile::shim(0, 0);

        let result = interpreter.step(&mut ctx, &mut tile);

        assert!(matches!(result, StepResult::ExecError(_)));
        assert!(matches!(interpreter.status(), CoreStatus::Error));
    }
}
