//! Execution context for an AIE2 core.
//!
//! The `ExecutionContext` holds all processor state needed to execute
//! instructions: registers, program counter, flags, and execution statistics.
//!
//! This replaces the placeholder `ExecutionContext` in `traits.rs`.
//!
//! # Timing Support
//!
//! For cycle-accurate execution, use `TimingContext` which bundles:
//! - `HazardDetector`: Tracks RAW/WAW/WAR register hazards
//! - `MemoryModel`: Tracks memory bank conflicts
//! - `LatencyTable`: Operation latency lookup
//!
//! ```ignore
//! let mut ctx = ExecutionContext::new();
//! ctx.enable_timing(); // Enable cycle-accurate tracking
//! ```

use super::registers::{
    AccumulatorRegisterFile, ModifierRegisterFile, PointerRegisterFile, ScalarRegisterFile,
    VectorRegisterFile,
};
use crate::interpreter::timing::{HazardDetector, HazardType, LatencyTable, MemoryModel};
use crate::interpreter::traits::{Flags, StateAccess};

// ============================================================================
// Event Tracing
// ============================================================================

/// Types of events that can be recorded for profiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    /// Instruction execution started.
    InstructionStart { pc: u32 },
    /// Instruction execution completed.
    InstructionComplete { pc: u32, latency: u8 },
    /// Pipeline stall due to register hazard.
    RegisterHazard { hazard_type: HazardType, register: u8, cycles: u8 },
    /// Pipeline stall due to memory bank conflict.
    MemoryConflict { bank: u8, cycles: u8 },
    /// Pipeline stall due to branch penalty.
    BranchPenalty { cycles: u8 },
    /// Branch taken.
    BranchTaken { from_pc: u32, to_pc: u32 },
    /// DMA transfer started.
    DmaStart { channel: u8 },
    /// DMA transfer completed.
    DmaComplete { channel: u8 },
    /// Lock acquire started.
    LockAcquireStart { lock_id: u8 },
    /// Lock acquired successfully.
    LockAcquired { lock_id: u8 },
    /// Lock contention (waiting).
    LockContention { lock_id: u8 },
    /// Lock released.
    LockReleased { lock_id: u8 },
    /// Core halted.
    Halt,
}

/// A timestamped event for profiling.
#[derive(Debug, Clone, Copy)]
pub struct TimestampedEvent {
    /// Cycle when the event occurred.
    pub cycle: u64,
    /// The event type and details.
    pub event: EventType,
}

/// Event log for recording execution events.
#[derive(Clone)]
pub struct EventLog {
    /// Recorded events.
    events: Vec<TimestampedEvent>,
    /// Maximum events to keep (circular buffer behavior).
    max_events: usize,
    /// Whether tracing is enabled.
    enabled: bool,
}

impl EventLog {
    /// Create a new event log with default capacity.
    pub fn new() -> Self {
        Self::with_capacity(10000)
    }

    /// Create a new event log with specified capacity.
    pub fn with_capacity(max_events: usize) -> Self {
        Self {
            events: Vec::with_capacity(max_events.min(1000)),
            max_events,
            enabled: false,
        }
    }

    /// Enable event recording.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable event recording.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if recording is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Record an event at the given cycle.
    #[inline]
    pub fn record(&mut self, cycle: u64, event: EventType) {
        if !self.enabled {
            return;
        }
        if self.events.len() >= self.max_events {
            // Drop oldest events (circular buffer)
            self.events.remove(0);
        }
        self.events.push(TimestampedEvent { cycle, event });
    }

    /// Get all recorded events.
    pub fn events(&self) -> &[TimestampedEvent] {
        &self.events
    }

    /// Clear all events.
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Get the number of recorded events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if the log is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

impl Default for EventLog {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Timing Context
// ============================================================================

/// Timing context for cycle-accurate execution.
///
/// Bundles all timing-related state into one structure that can be
/// optionally attached to an `ExecutionContext` for accurate cycle counting.
#[derive(Clone)]
pub struct TimingContext {
    /// Register hazard detector (RAW, WAW, WAR).
    pub hazards: HazardDetector,

    /// Memory bank conflict detector.
    pub memory: MemoryModel,

    /// Operation latency lookup table.
    pub latencies: LatencyTable,

    /// Total hazard stall cycles.
    pub hazard_stalls: u64,

    /// Total memory conflict stall cycles.
    pub memory_stalls: u64,

    /// Event log for profiling/tracing.
    pub events: EventLog,
}

impl TimingContext {
    /// Create a new timing context with AIE2 defaults.
    pub fn new() -> Self {
        Self {
            hazards: HazardDetector::new(),
            memory: MemoryModel::new(),
            latencies: LatencyTable::aie2(),
            hazard_stalls: 0,
            memory_stalls: 0,
            events: EventLog::new(),
        }
    }

    /// Advance all timing models to the given cycle.
    pub fn advance_to(&mut self, cycle: u64) {
        self.hazards.advance_to(cycle);
        self.memory.advance_to(cycle);
    }

    /// Reset timing state but keep latency table.
    pub fn reset(&mut self) {
        self.hazards.reset();
        self.memory.reset();
        self.hazard_stalls = 0;
        self.memory_stalls = 0;
        self.events.clear();
    }

    /// Enable event tracing.
    pub fn enable_tracing(&mut self) {
        self.events.enable();
    }

    /// Disable event tracing.
    pub fn disable_tracing(&mut self) {
        self.events.disable();
    }

    /// Record an event at the given cycle.
    #[inline]
    pub fn record_event(&mut self, cycle: u64, event: EventType) {
        self.events.record(cycle, event);
    }

    /// Get combined timing statistics.
    pub fn total_stall_cycles(&self) -> u64 {
        self.hazard_stalls + self.memory_stalls
    }
}

impl Default for TimingContext {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for TimingContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TimingContext")
            .field("hazard_stalls", &self.hazard_stalls)
            .field("memory_stalls", &self.memory_stalls)
            .finish()
    }
}

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

    // === Timing (optional) ===
    /// Timing context for cycle-accurate execution.
    /// When None, uses fast mode (1 cycle per instruction).
    pub timing: Option<TimingContext>,
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
    /// Uses fast mode (no timing) by default.
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
            timing: None,
        }
    }

    /// Create a new context with cycle-accurate timing enabled.
    ///
    /// This enables:
    /// - Hazard detection (RAW, WAW, WAR)
    /// - Memory bank conflict modeling
    /// - Event tracing for profiling
    ///
    /// Use `disable_tracing()` on the timing context if you need
    /// cycle-accurate execution without the overhead of event recording.
    pub fn new_with_timing() -> Self {
        let mut ctx = Self::new();
        ctx.enable_timing();
        // Enable event tracing by default for profiling
        if let Some(timing) = ctx.timing_context_mut() {
            timing.enable_tracing();
        }
        ctx
    }

    /// Create a new context with initial stack pointer.
    pub fn with_stack(stack_addr: u32) -> Self {
        let mut ctx = Self::new();
        ctx.set_sp(stack_addr);
        ctx
    }

    /// Enable cycle-accurate timing.
    /// This adds overhead but provides accurate cycle counts.
    pub fn enable_timing(&mut self) {
        if self.timing.is_none() {
            self.timing = Some(TimingContext::new());
        }
    }

    /// Disable cycle-accurate timing (switch to fast mode).
    pub fn disable_timing(&mut self) {
        self.timing = None;
    }

    /// Check if cycle-accurate timing is enabled.
    #[inline]
    pub fn has_timing(&self) -> bool {
        self.timing.is_some()
    }

    /// Get the timing context if enabled.
    #[inline]
    pub fn timing_context(&self) -> Option<&TimingContext> {
        self.timing.as_ref()
    }

    /// Get mutable timing context if enabled.
    #[inline]
    pub fn timing_context_mut(&mut self) -> Option<&mut TimingContext> {
        self.timing.as_mut()
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
    pub fn push(&mut self, _value: u32) {
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

    #[test]
    fn test_timing_context() {
        // Default: no timing
        let ctx = ExecutionContext::new();
        assert!(!ctx.has_timing());
        assert!(ctx.timing.is_none());

        // Create with timing
        let ctx_timed = ExecutionContext::new_with_timing();
        assert!(ctx_timed.has_timing());
        assert!(ctx_timed.timing.is_some());
    }

    #[test]
    fn test_enable_disable_timing() {
        let mut ctx = ExecutionContext::new();

        // Enable timing
        ctx.enable_timing();
        assert!(ctx.has_timing());

        // Access timing context
        if let Some(timing) = ctx.timing_context_mut() {
            timing.hazard_stalls = 5;
        }
        assert_eq!(ctx.timing_context().unwrap().hazard_stalls, 5);

        // Disable timing
        ctx.disable_timing();
        assert!(!ctx.has_timing());
        assert!(ctx.timing_context().is_none());
    }

    #[test]
    fn test_timing_context_reset() {
        let mut timing = TimingContext::new();
        timing.hazard_stalls = 10;
        timing.memory_stalls = 5;

        timing.reset();
        assert_eq!(timing.hazard_stalls, 0);
        assert_eq!(timing.memory_stalls, 0);
        assert_eq!(timing.total_stall_cycles(), 0);
    }

    // --- Event Log tests ---

    #[test]
    fn test_event_log_disabled_by_default() {
        let log = EventLog::new();
        assert!(!log.is_enabled());
        assert!(log.is_empty());
    }

    #[test]
    fn test_event_log_record_when_disabled() {
        let mut log = EventLog::new();

        // Recording when disabled should not add events
        log.record(10, EventType::InstructionStart { pc: 0x100 });
        assert!(log.is_empty());
    }

    #[test]
    fn test_event_log_record_when_enabled() {
        let mut log = EventLog::new();
        log.enable();

        log.record(10, EventType::InstructionStart { pc: 0x100 });
        log.record(11, EventType::InstructionComplete { pc: 0x100, latency: 1 });

        assert_eq!(log.len(), 2);
        assert_eq!(log.events()[0].cycle, 10);
        assert_eq!(log.events()[1].cycle, 11);
    }

    #[test]
    fn test_event_log_clear() {
        let mut log = EventLog::new();
        log.enable();

        log.record(1, EventType::Halt);
        log.record(2, EventType::Halt);
        assert_eq!(log.len(), 2);

        log.clear();
        assert!(log.is_empty());
    }

    #[test]
    fn test_event_log_circular_buffer() {
        let mut log = EventLog::with_capacity(3);
        log.enable();

        log.record(1, EventType::InstructionStart { pc: 0x100 });
        log.record(2, EventType::InstructionStart { pc: 0x104 });
        log.record(3, EventType::InstructionStart { pc: 0x108 });

        // At capacity, next record should drop oldest
        log.record(4, EventType::InstructionStart { pc: 0x10C });

        assert_eq!(log.len(), 3);
        // First event (cycle 1) should be dropped
        assert_eq!(log.events()[0].cycle, 2);
        assert_eq!(log.events()[2].cycle, 4);
    }

    #[test]
    fn test_timing_context_event_tracing() {
        let mut timing = TimingContext::new();

        // Events disabled by default
        timing.record_event(10, EventType::BranchTaken { from_pc: 0x100, to_pc: 0x200 });
        assert!(timing.events.is_empty());

        // Enable tracing
        timing.enable_tracing();
        timing.record_event(20, EventType::BranchTaken { from_pc: 0x200, to_pc: 0x300 });
        assert_eq!(timing.events.len(), 1);

        // Disable tracing
        timing.disable_tracing();
        timing.record_event(30, EventType::Halt);
        assert_eq!(timing.events.len(), 1); // No new event recorded
    }

    #[test]
    fn test_event_type_variants() {
        use crate::interpreter::timing::HazardType;

        // Test that all event variants can be created
        let events = vec![
            EventType::InstructionStart { pc: 0x100 },
            EventType::InstructionComplete { pc: 0x100, latency: 2 },
            EventType::RegisterHazard { hazard_type: HazardType::Raw, register: 5, cycles: 2 },
            EventType::MemoryConflict { bank: 3, cycles: 1 },
            EventType::BranchPenalty { cycles: 3 },
            EventType::BranchTaken { from_pc: 0x100, to_pc: 0x200 },
            EventType::DmaStart { channel: 0 },
            EventType::DmaComplete { channel: 0 },
            EventType::LockAcquireStart { lock_id: 5 },
            EventType::LockAcquired { lock_id: 5 },
            EventType::LockContention { lock_id: 5 },
            EventType::LockReleased { lock_id: 5 },
            EventType::Halt,
        ];

        let mut log = EventLog::new();
        log.enable();

        for (i, event) in events.into_iter().enumerate() {
            log.record(i as u64, event);
        }

        assert_eq!(log.len(), 13);
    }
}
