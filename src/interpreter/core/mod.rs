//! Per-core interpreter.
//!
//! The `CoreInterpreter` ties together the decoder and executor to run a
//! single AIE2 compute core. It manages the execution loop, handles stalls,
//! and tracks core status.
//!
//! # Execution Model
//!
//! Each core runs independently:
//!
//! 1. Fetch instruction at PC from program memory
//! 2. Decode into VLIW bundle
//! 3. Execute all slot operations
//! 4. Handle result (advance PC, branch, stall, or halt)
//!
//! # Example
//!
//! ```ignore
//! use xdna_emu::interpreter::{CoreInterpreter, InstructionDecoder, CycleAccurateExecutor};
//!
//! let decoder = InstructionDecoder::load_default();
//! let executor = CycleAccurateExecutor::new();
//! let mut interpreter = CoreInterpreter::new(decoder, executor);
//!
//! // Run 100 cycles or until stall/halt
//! let result = interpreter.run(&mut ctx, &mut tile, 100);
//! ```

mod interpreter;

pub use interpreter::{CoreInterpreter, CoreStatus, StepResult};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::tile::Tile;
    use crate::interpreter::decode::InstructionDecoder;
    use crate::interpreter::execute::CycleAccurateExecutor;
    use crate::interpreter::state::ExecutionContext;

    #[test]
    fn test_core_interpreter_basic() {
        let decoder = InstructionDecoder::load_default();
        let executor = CycleAccurateExecutor::new();
        let mut interpreter = CoreInterpreter::new(decoder, executor);

        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        // Write NOP instruction to program memory
        // 0x00000000 is detected as NOP
        assert!(tile.write_program(0, &[0x00, 0x00, 0x00, 0x00]));

        // Run one step
        let result = interpreter.step(&mut ctx, &mut tile);
        assert!(matches!(result, StepResult::Continue));
        assert_eq!(ctx.pc(), 4); // Moved to next instruction
        assert_eq!(ctx.instructions, 1);
    }
}
