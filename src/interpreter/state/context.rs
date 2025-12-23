//! Execution context for an AIE2 core.
//!
//! The `ExecutionContext` holds all processor state needed to execute
//! instructions: registers, program counter, flags, and execution statistics.
//!
//! This replaces the placeholder `ExecutionContext` in `traits.rs`.

use super::registers::{
    AccumulatorRegisterFile, ModifierRegisterFile, PointerRegisterFile, ScalarRegisterFile,
    VectorRegisterFile,
};
use crate::interpreter::traits::{Flags, StateAccess};

/// Complete execution context for an AIE2 core.
///
/// Contains all register files and execution state needed for instruction
/// execution. Designed for efficient access patterns:
///
/// - Hot path: scalar regs, PC, flags (accessed every instruction)
/// - Warm path: pointer/modifier regs (accessed on memory ops)
/// - Cold path: vector/accumulator regs (accessed on vector ops)
#[derive(Clone)]
pub struct ExecutionContext {
    // === Hot path (accessed every cycle) ===
    /// Program counter.
    pc: u32,

    /// Condition flags.
    flags: Flags,

    /// Scalar general purpose registers (r0-r31).
    pub scalar: ScalarRegisterFile,

    // === Warm path (memory operations) ===
    /// Pointer registers (p0-p7).
    pub pointer: PointerRegisterFile,

    /// Modifier registers (m0-m7).
    pub modifier: ModifierRegisterFile,

    // === Cold path (vector operations) ===
    /// Vector registers (v0-v31).
    pub vector: VectorRegisterFile,

    /// Accumulator registers (acc0-acc7).
    pub accumulator: AccumulatorRegisterFile,

    // === Statistics ===
    /// Total cycles executed.
    pub cycles: u64,

    /// Total instructions executed.
    pub instructions: u64,

    /// Stall cycles (waiting on locks, DMA, etc.).
    pub stall_cycles: u64,

    // === Control ===
    /// Core is halted.
    pub halted: bool,

    /// Stack pointer register (alias to a scalar or pointer reg).
    /// By convention, often p0 or r13.
    sp_reg: SpRegister,

    /// Link register (alias to a scalar reg).
    /// By convention, often r0 or r14.
    lr_reg: u8,
}

/// Which register to use as stack pointer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpRegister {
    /// Use pointer register (p0-p7).
    Pointer(u8),
    /// Use scalar register (r0-r31).
    Scalar(u8),
}

impl Default for SpRegister {
    fn default() -> Self {
        SpRegister::Pointer(0) // p0 is typical stack pointer
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionContext {
    /// Create a new execution context with all state zeroed.
    pub fn new() -> Self {
        Self {
            pc: 0,
            flags: Flags::default(),
            scalar: ScalarRegisterFile::new(),
            pointer: PointerRegisterFile::new(),
            modifier: ModifierRegisterFile::new(),
            vector: VectorRegisterFile::new(),
            accumulator: AccumulatorRegisterFile::new(),
            cycles: 0,
            instructions: 0,
            stall_cycles: 0,
            halted: false,
            sp_reg: SpRegister::default(),
            lr_reg: 0, // r0 as link register
        }
    }

    /// Create a new context with initial stack pointer.
    pub fn with_stack(stack_addr: u32) -> Self {
        let mut ctx = Self::new();
        ctx.set_sp(stack_addr);
        ctx
    }

    /// Get the program counter.
    #[inline]
    pub fn pc(&self) -> u32 {
        self.pc
    }

    /// Set the program counter.
    #[inline]
    pub fn set_pc(&mut self, pc: u32) {
        self.pc = pc;
    }

    /// Advance PC by the given offset.
    #[inline]
    pub fn advance_pc(&mut self, offset: u32) {
        self.pc = self.pc.wrapping_add(offset);
    }

    /// Get the condition flags.
    #[inline]
    pub fn flags(&self) -> Flags {
        self.flags
    }

    /// Set the condition flags.
    #[inline]
    pub fn set_flags(&mut self, flags: Flags) {
        self.flags = flags;
    }

    /// Get the stack pointer value.
    #[inline]
    pub fn sp(&self) -> u32 {
        match self.sp_reg {
            SpRegister::Pointer(r) => self.pointer.read(r),
            SpRegister::Scalar(r) => self.scalar.read(r),
        }
    }

    /// Set the stack pointer value.
    #[inline]
    pub fn set_sp(&mut self, value: u32) {
        match self.sp_reg {
            SpRegister::Pointer(r) => self.pointer.write(r, value),
            SpRegister::Scalar(r) => self.scalar.write(r, value),
        }
    }

    /// Get the link register value.
    #[inline]
    pub fn lr(&self) -> u32 {
        self.scalar.read(self.lr_reg)
    }

    /// Set the link register value.
    #[inline]
    pub fn set_lr(&mut self, value: u32) {
        self.scalar.write(self.lr_reg, value);
    }

    /// Push a value onto the stack (decrement SP, write value).
    pub fn push(&mut self, value: u32) {
        let new_sp = self.sp().wrapping_sub(4);
        self.set_sp(new_sp);
        // Note: actual memory write must be done by caller
    }

    /// Pop a value from the stack (read value, increment SP).
    /// Returns the stack address where the value should be read from.
    pub fn pop(&mut self) -> u32 {
        let sp = self.sp();
        self.set_sp(sp.wrapping_add(4));
        sp
    }

    /// Record one instruction executed.
    #[inline]
    pub fn record_instruction(&mut self, cycles: u64) {
        self.instructions += 1;
        self.cycles += cycles;
    }

    /// Record stall cycles.
    #[inline]
    pub fn record_stall(&mut self, cycles: u64) {
        self.stall_cycles += cycles;
        self.cycles += cycles;
    }

    /// Reset execution statistics.
    pub fn reset_stats(&mut self) {
        self.cycles = 0;
        self.instructions = 0;
        self.stall_cycles = 0;
    }

    /// Get instructions per cycle (IPC) ratio.
    pub fn ipc(&self) -> f64 {
        if self.cycles == 0 {
            0.0
        } else {
            self.instructions as f64 / self.cycles as f64
        }
    }

    /// Reset all state (registers, PC, flags, stats).
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Configure which register is used as stack pointer.
    pub fn set_sp_register(&mut self, reg: SpRegister) {
        self.sp_reg = reg;
    }

    /// Configure which scalar register is used as link register.
    pub fn set_lr_register(&mut self, reg: u8) {
        self.lr_reg = reg & 0x1F;
    }
}

impl StateAccess for ExecutionContext {
    fn read_scalar(&self, reg: u8) -> u32 {
        self.scalar.read(reg)
    }

    fn write_scalar(&mut self, reg: u8, value: u32) {
        self.scalar.write(reg, value);
    }

    fn read_vector(&self, reg: u8) -> [u32; 8] {
        self.vector.read(reg)
    }

    fn write_vector(&mut self, reg: u8, value: [u32; 8]) {
        self.vector.write(reg, value);
    }

    fn pc(&self) -> u32 {
        self.pc
    }

    fn set_pc(&mut self, pc: u32) {
        self.pc = pc;
    }

    fn flags(&self) -> Flags {
        self.flags
    }

    fn set_flags(&mut self, flags: Flags) {
        self.flags = flags;
    }
}

impl std::fmt::Debug for ExecutionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionContext")
            .field("pc", &format_args!("0x{:08X}", self.pc))
            .field("flags", &self.flags)
            .field("cycles", &self.cycles)
            .field("instructions", &self.instructions)
            .field("halted", &self.halted)
            .field("scalar", &self.scalar)
            .field("pointer", &self.pointer)
            .field("vector", &self.vector)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = ExecutionContext::new();
        assert_eq!(ctx.pc(), 0);
        assert_eq!(ctx.cycles, 0);
        assert!(!ctx.halted);
    }

    #[test]
    fn test_context_with_stack() {
        let ctx = ExecutionContext::with_stack(0x7_0000);
        assert_eq!(ctx.sp(), 0x7_0000);
    }

    #[test]
    fn test_pc_operations() {
        let mut ctx = ExecutionContext::new();

        ctx.set_pc(0x1000);
        assert_eq!(ctx.pc(), 0x1000);

        ctx.advance_pc(4);
        assert_eq!(ctx.pc(), 0x1004);

        ctx.advance_pc(0xFFFF_FFFC); // -4 wrapping
        assert_eq!(ctx.pc(), 0x1000);
    }

    #[test]
    fn test_flags_operations() {
        let mut ctx = ExecutionContext::new();

        let flags = Flags {
            z: true,
            n: false,
            c: true,
            v: false,
        };
        ctx.set_flags(flags);

        let read = ctx.flags();
        assert!(read.z);
        assert!(read.c);
        assert!(!read.n);
    }

    #[test]
    fn test_stack_operations() {
        let mut ctx = ExecutionContext::with_stack(0x1000);

        ctx.push(0xDEAD);
        assert_eq!(ctx.sp(), 0x0FFC);

        let addr = ctx.pop();
        assert_eq!(addr, 0x0FFC);
        assert_eq!(ctx.sp(), 0x1000);
    }

    #[test]
    fn test_link_register() {
        let mut ctx = ExecutionContext::new();

        ctx.set_lr(0x2000);
        assert_eq!(ctx.lr(), 0x2000);
        assert_eq!(ctx.scalar.read(0), 0x2000); // lr_reg defaults to r0
    }

    #[test]
    fn test_statistics() {
        let mut ctx = ExecutionContext::new();

        ctx.record_instruction(1);
        ctx.record_instruction(2);
        ctx.record_stall(3);

        assert_eq!(ctx.instructions, 2);
        assert_eq!(ctx.cycles, 6); // 1 + 2 + 3
        assert_eq!(ctx.stall_cycles, 3);
        assert!((ctx.ipc() - 0.333).abs() < 0.01);

        ctx.reset_stats();
        assert_eq!(ctx.cycles, 0);
    }

    #[test]
    fn test_state_access_trait() {
        let mut ctx = ExecutionContext::new();

        // Test through trait methods
        <ExecutionContext as StateAccess>::write_scalar(&mut ctx, 10, 42);
        assert_eq!(<ExecutionContext as StateAccess>::read_scalar(&ctx, 10), 42);

        let vec_data = [1, 2, 3, 4, 5, 6, 7, 8];
        <ExecutionContext as StateAccess>::write_vector(&mut ctx, 5, vec_data);
        assert_eq!(
            <ExecutionContext as StateAccess>::read_vector(&ctx, 5),
            vec_data
        );
    }

    #[test]
    fn test_sp_register_config() {
        let mut ctx = ExecutionContext::new();

        // Default: pointer register p0
        ctx.set_sp(0x1000);
        assert_eq!(ctx.pointer.read(0), 0x1000);

        // Switch to scalar register r13
        ctx.set_sp_register(SpRegister::Scalar(13));
        ctx.set_sp(0x2000);
        assert_eq!(ctx.scalar.read(13), 0x2000);
        assert_eq!(ctx.sp(), 0x2000);
    }

    #[test]
    fn test_reset() {
        let mut ctx = ExecutionContext::new();

        ctx.set_pc(0x1000);
        ctx.scalar.write(5, 42);
        ctx.cycles = 100;
        ctx.halted = true;

        ctx.reset();

        assert_eq!(ctx.pc(), 0);
        assert_eq!(ctx.scalar.read(5), 0);
        assert_eq!(ctx.cycles, 0);
        assert!(!ctx.halted);
    }
}
