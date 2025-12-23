//! AIE core executor.
//!
//! Each compute tile has an AIE core that executes independently.
//! This module provides the execution logic for a single core.

use super::instruction::{Instruction, InstructionKind, ArithOp, BranchCond, DecodeError};
use crate::device::tile::{Tile, Lock, PROGRAM_MEMORY_SIZE};

/// Core execution status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreStatus {
    /// Core is idle (not enabled).
    Idle,
    /// Core is running normally.
    Running,
    /// Core is waiting on a lock.
    WaitingLock { lock_id: u8 },
    /// Core is waiting for DMA completion.
    WaitingDma { channel: u8 },
    /// Core has halted (reached halt instruction or error).
    Halted,
    /// Core hit a breakpoint.
    Breakpoint,
    /// Decode error occurred.
    DecodeError,
}

/// Result of executing one instruction.
#[derive(Debug, Clone)]
pub enum ExecuteResult {
    /// Continue to next instruction.
    Continue,
    /// Branch to a new address.
    Branch { target: u32 },
    /// Core is now waiting on a lock.
    WaitLock { lock_id: u8 },
    /// Core is now waiting for DMA.
    WaitDma { channel: u8 },
    /// Core has halted.
    Halt,
    /// Error during execution.
    Error { message: String },
}

/// Scalar register file.
#[derive(Debug, Clone)]
pub struct ScalarRegs {
    /// General purpose registers (r0-r15).
    pub gpr: [u32; 16],
}

impl Default for ScalarRegs {
    fn default() -> Self {
        Self { gpr: [0; 16] }
    }
}

/// Vector register file (simplified).
#[derive(Debug, Clone)]
pub struct VectorRegs {
    /// Vector registers (v0-v7), each 256 bits (8 x u32).
    pub vr: [[u32; 8]; 8],
}

impl Default for VectorRegs {
    fn default() -> Self {
        Self { vr: [[0; 8]; 8] }
    }
}

/// Core executor state.
///
/// This maintains the execution state for a single AIE core,
/// separate from the tile hardware state.
#[derive(Debug)]
pub struct CoreExecutor {
    /// Column position.
    pub col: u8,
    /// Row position.
    pub row: u8,
    /// Current status.
    pub status: CoreStatus,
    /// Cycle count.
    pub cycles: u64,
    /// Instructions executed.
    pub instructions: u64,
    /// Scalar register file.
    pub scalar: ScalarRegs,
    /// Vector register file.
    pub vector: VectorRegs,
    /// Last decoded instruction (for display).
    pub last_instruction: Option<Instruction>,
    /// Last execution result.
    pub last_result: Option<ExecuteResult>,
}

impl CoreExecutor {
    /// Create a new core executor for the given tile position.
    pub fn new(col: u8, row: u8) -> Self {
        Self {
            col,
            row,
            status: CoreStatus::Idle,
            cycles: 0,
            instructions: 0,
            scalar: ScalarRegs::default(),
            vector: VectorRegs::default(),
            last_instruction: None,
            last_result: None,
        }
    }

    /// Reset the executor state.
    pub fn reset(&mut self) {
        self.status = CoreStatus::Idle;
        self.cycles = 0;
        self.instructions = 0;
        self.scalar = ScalarRegs::default();
        self.vector = VectorRegs::default();
        self.last_instruction = None;
        self.last_result = None;
    }

    /// Execute one cycle on this core.
    ///
    /// Returns true if the core is still running.
    pub fn step(&mut self, tile: &mut Tile) -> bool {
        // Check if core is enabled
        if !tile.core.enabled {
            self.status = CoreStatus::Idle;
            return false;
        }

        // Handle waiting states
        match self.status {
            CoreStatus::Idle => {
                if tile.core.enabled {
                    self.status = CoreStatus::Running;
                } else {
                    return false;
                }
            }
            CoreStatus::WaitingLock { lock_id } => {
                // Try to acquire the lock
                if lock_id < 64 && tile.locks[lock_id as usize].value > 0 {
                    tile.locks[lock_id as usize].acquire();
                    self.status = CoreStatus::Running;
                } else {
                    self.cycles += 1;
                    return true; // Still waiting
                }
            }
            CoreStatus::WaitingDma { channel } => {
                // Check if DMA is complete
                if channel < 4 && !tile.dma_channels[channel as usize].running {
                    self.status = CoreStatus::Running;
                } else {
                    self.cycles += 1;
                    return true; // Still waiting
                }
            }
            CoreStatus::Halted | CoreStatus::Breakpoint | CoreStatus::DecodeError => {
                return false;
            }
            CoreStatus::Running => {}
        }

        // Fetch instruction
        let pc = tile.core.pc as usize;
        let program = match tile.program_memory() {
            Some(pm) => pm,
            None => {
                self.status = CoreStatus::Halted;
                return false;
            }
        };

        if pc >= PROGRAM_MEMORY_SIZE {
            self.status = CoreStatus::Halted;
            return false;
        }

        // Decode instruction
        let inst = match Instruction::decode(&program[pc..]) {
            Ok(i) => i,
            Err(_) => {
                self.status = CoreStatus::DecodeError;
                return false;
            }
        };

        // Execute instruction
        let result = self.execute(&inst, tile);

        // Update PC based on result
        match &result {
            ExecuteResult::Continue => {
                tile.core.pc += inst.size as u32;
            }
            ExecuteResult::Branch { target } => {
                tile.core.pc = *target;
            }
            ExecuteResult::WaitLock { lock_id } => {
                self.status = CoreStatus::WaitingLock { lock_id: *lock_id };
            }
            ExecuteResult::WaitDma { channel } => {
                self.status = CoreStatus::WaitingDma { channel: *channel };
            }
            ExecuteResult::Halt => {
                self.status = CoreStatus::Halted;
            }
            ExecuteResult::Error { .. } => {
                self.status = CoreStatus::Halted;
            }
        }

        // Update stats
        self.cycles += 1;
        self.instructions += 1;
        self.last_instruction = Some(inst);
        self.last_result = Some(result);

        self.status == CoreStatus::Running
            || matches!(self.status, CoreStatus::WaitingLock { .. } | CoreStatus::WaitingDma { .. })
    }

    /// Execute a single instruction.
    fn execute(&mut self, inst: &Instruction, tile: &mut Tile) -> ExecuteResult {
        match &inst.kind {
            InstructionKind::Nop => ExecuteResult::Continue,

            InstructionKind::Move { dst, src } => {
                if (*dst as usize) < 16 && (*src as usize) < 16 {
                    self.scalar.gpr[*dst as usize] = self.scalar.gpr[*src as usize];
                }
                ExecuteResult::Continue
            }

            InstructionKind::Load { dst, base, offset } => {
                let addr = (self.scalar.gpr[*base as usize] as i32 + *offset as i32) as usize;
                if let Some(value) = tile.read_data_u32(addr) {
                    if (*dst as usize) < 16 {
                        self.scalar.gpr[*dst as usize] = value;
                    }
                }
                ExecuteResult::Continue
            }

            InstructionKind::Store { src, base, offset } => {
                let addr = (self.scalar.gpr[*base as usize] as i32 + *offset as i32) as usize;
                if (*src as usize) < 16 {
                    let value = self.scalar.gpr[*src as usize];
                    tile.write_data_u32(addr, value);
                }
                ExecuteResult::Continue
            }

            InstructionKind::Arith { op, dst, src1, src2 } => {
                if (*dst as usize) < 16 && (*src1 as usize) < 16 && (*src2 as usize) < 16 {
                    let a = self.scalar.gpr[*src1 as usize];
                    let b = self.scalar.gpr[*src2 as usize];

                    let result = match op {
                        ArithOp::Add => a.wrapping_add(b),
                        ArithOp::Sub => a.wrapping_sub(b),
                        ArithOp::Mul => a.wrapping_mul(b),
                        ArithOp::And => a & b,
                        ArithOp::Or => a | b,
                        ArithOp::Xor => a ^ b,
                        ArithOp::Shl => a << (b & 31),
                        ArithOp::Shr => a >> (b & 31),
                    };

                    self.scalar.gpr[*dst as usize] = result;
                }
                ExecuteResult::Continue
            }

            InstructionKind::Branch { target, condition } => {
                let take_branch = match condition {
                    BranchCond::Always => true,
                    BranchCond::Equal => self.scalar.gpr[0] == 0,
                    BranchCond::NotEqual => self.scalar.gpr[0] != 0,
                    BranchCond::LessThan => (self.scalar.gpr[0] as i32) < 0,
                    BranchCond::GreaterEqual => (self.scalar.gpr[0] as i32) >= 0,
                };

                if take_branch {
                    ExecuteResult::Branch { target: *target }
                } else {
                    ExecuteResult::Continue
                }
            }

            InstructionKind::Call { target } => {
                // Save return address to link register
                tile.core.lr = tile.core.pc + inst.size as u32;
                ExecuteResult::Branch { target: *target }
            }

            InstructionKind::Return => {
                ExecuteResult::Branch { target: tile.core.lr }
            }

            InstructionKind::LockAcquire { lock_id, value } => {
                let lid = *lock_id as usize;
                if lid < 64 {
                    // Check if we can acquire (value > 0)
                    if tile.locks[lid].value > 0 {
                        tile.locks[lid].value = tile.locks[lid].value.saturating_sub(*value as u32);
                        ExecuteResult::Continue
                    } else {
                        // Need to wait
                        ExecuteResult::WaitLock { lock_id: *lock_id }
                    }
                } else {
                    ExecuteResult::Error {
                        message: format!("Invalid lock ID: {}", lock_id),
                    }
                }
            }

            InstructionKind::LockRelease { lock_id, value } => {
                let lid = *lock_id as usize;
                if lid < 64 {
                    tile.locks[lid].value = tile.locks[lid].value.saturating_add(*value as u32);
                    ExecuteResult::Continue
                } else {
                    ExecuteResult::Error {
                        message: format!("Invalid lock ID: {}", lock_id),
                    }
                }
            }

            InstructionKind::Vector { op } => {
                // Simplified vector operation - just increment cycle count
                // Real implementation would need full vector ISA
                ExecuteResult::Continue
            }

            InstructionKind::Dma { channel, bd, start } => {
                if *channel < 4 {
                    if *start {
                        // Start DMA transfer
                        tile.dma_channels[*channel as usize].current_bd = *bd;
                        tile.dma_channels[*channel as usize].running = true;
                    }
                    // In a real emulator, we'd simulate the DMA here
                    // For now, just mark as complete immediately
                    tile.dma_channels[*channel as usize].running = false;
                }
                ExecuteResult::Continue
            }

            InstructionKind::Unknown { .. } => {
                // Treat unknown as NOP for now
                ExecuteResult::Continue
            }
        }
    }

    /// Check if this core is active (running or waiting).
    pub fn is_active(&self) -> bool {
        matches!(
            self.status,
            CoreStatus::Running | CoreStatus::WaitingLock { .. } | CoreStatus::WaitingDma { .. }
        )
    }

    /// Get a status string for display.
    pub fn status_string(&self) -> &'static str {
        match self.status {
            CoreStatus::Idle => "Idle",
            CoreStatus::Running => "Running",
            CoreStatus::WaitingLock { .. } => "Wait Lock",
            CoreStatus::WaitingDma { .. } => "Wait DMA",
            CoreStatus::Halted => "Halted",
            CoreStatus::Breakpoint => "Breakpoint",
            CoreStatus::DecodeError => "Decode Err",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::tile::Tile;

    fn make_test_tile() -> Tile {
        let mut tile = Tile::compute(1, 2);
        tile.core.enabled = true;
        tile
    }

    #[test]
    fn test_core_executor_creation() {
        let exec = CoreExecutor::new(1, 2);
        assert_eq!(exec.col, 1);
        assert_eq!(exec.row, 2);
        assert_eq!(exec.status, CoreStatus::Idle);
    }

    #[test]
    fn test_step_disabled_core() {
        let mut exec = CoreExecutor::new(1, 2);
        let mut tile = Tile::compute(1, 2);
        tile.core.enabled = false;

        let running = exec.step(&mut tile);
        assert!(!running);
        assert_eq!(exec.status, CoreStatus::Idle);
    }

    #[test]
    fn test_step_nop() {
        let mut exec = CoreExecutor::new(1, 2);
        let mut tile = make_test_tile();

        // Write NOP instruction at PC=0
        assert!(tile.write_program(0, &[0x00, 0x00, 0x00, 0x00]));

        let running = exec.step(&mut tile);
        assert!(running);
        assert_eq!(tile.core.pc, 4);
        assert_eq!(exec.instructions, 1);
    }

    #[test]
    fn test_arith_add() {
        let mut exec = CoreExecutor::new(1, 2);
        let mut tile = make_test_tile();

        // Set up registers
        exec.scalar.gpr[1] = 10;
        exec.scalar.gpr[2] = 20;

        // Create ADD instruction manually
        let inst = Instruction {
            kind: InstructionKind::Arith {
                op: ArithOp::Add,
                dst: 0,
                src1: 1,
                src2: 2,
            },
            size: 4,
            raw: vec![0; 4],
        };

        let result = exec.execute(&inst, &mut tile);
        assert!(matches!(result, ExecuteResult::Continue));
        assert_eq!(exec.scalar.gpr[0], 30);
    }

    #[test]
    fn test_lock_acquire_available() {
        let mut exec = CoreExecutor::new(1, 2);
        let mut tile = make_test_tile();

        // Set lock 5 to value 2
        tile.locks[5].value = 2;

        let inst = Instruction {
            kind: InstructionKind::LockAcquire { lock_id: 5, value: 1 },
            size: 4,
            raw: vec![0; 4],
        };

        let result = exec.execute(&inst, &mut tile);
        assert!(matches!(result, ExecuteResult::Continue));
        assert_eq!(tile.locks[5].value, 1);
    }

    #[test]
    fn test_lock_acquire_wait() {
        let mut exec = CoreExecutor::new(1, 2);
        let mut tile = make_test_tile();

        // Lock 5 is already at 0
        tile.locks[5].value = 0;

        let inst = Instruction {
            kind: InstructionKind::LockAcquire { lock_id: 5, value: 1 },
            size: 4,
            raw: vec![0; 4],
        };

        let result = exec.execute(&inst, &mut tile);
        assert!(matches!(result, ExecuteResult::WaitLock { lock_id: 5 }));
    }

    #[test]
    fn test_memory_load_store() {
        let mut exec = CoreExecutor::new(1, 2);
        let mut tile = make_test_tile();

        // Store value
        exec.scalar.gpr[0] = 0x100; // base address
        exec.scalar.gpr[1] = 0xDEADBEEF; // value to store

        let store = Instruction {
            kind: InstructionKind::Store { src: 1, base: 0, offset: 0 },
            size: 4,
            raw: vec![0; 4],
        };
        exec.execute(&store, &mut tile);

        // Load value back
        let load = Instruction {
            kind: InstructionKind::Load { dst: 2, base: 0, offset: 0 },
            size: 4,
            raw: vec![0; 4],
        };
        exec.execute(&load, &mut tile);

        assert_eq!(exec.scalar.gpr[2], 0xDEADBEEF);
    }
}
