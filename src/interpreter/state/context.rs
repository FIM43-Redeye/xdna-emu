//! Execution context for an AIE2 core.
//!
//! The `ExecutionContext` holds all processor state needed to execute
//! instructions: registers, program counter, flags, and execution statistics.
//!
//! This replaces the placeholder `ExecutionContext` in `traits.rs`.
//!
//! # Timing Support
//!
//! All execution is cycle-accurate. The `TimingContext` (always present) includes:
//! - `HazardDetector`: Tracks RAW/WAW/WAR register hazards
//! - `MemoryModel`: Tracks memory bank conflicts
//! - `LatencyTable`: Operation latency lookup
//!
//! ```ignore
//! let ctx = ExecutionContext::new();
//! assert!(ctx.has_timing()); // Always true
//! ```

use super::registers::{
    AccumulatorRegisterFile, ModifierRegisterFile, PointerRegisterFile, ScalarRegisterFile,
    VectorRegisterFile,
};
use crate::interpreter::bundle::Operand;
use crate::interpreter::timing::{HazardDetector, LatencyTable, MemoryModel};
use crate::interpreter::traits::{Flags, StateAccess};

// ============================================================================
// Load Latency: Pending Write Queue
// ============================================================================

/// A deferred register write from a memory load operation.
///
/// AIE2 memory loads have a 7-cycle pipeline latency (AIE2Schedule.td
/// operandcycles[0] = 7 for all load itineraries). The compiler pipelines
/// multiple loads to the same register, relying on this latency to keep
/// earlier values alive until they are consumed. Without deferred writes,
/// later loads instantly overwrite earlier values, breaking the pipeline.
#[derive(Debug, Clone)]
pub struct PendingWrite {
    /// Destination register to write when ready.
    pub dest: Operand,
    /// Value to write (scalar: lower 32 bits; vector: use vec_value).
    pub scalar_value: u32,
    /// Vector value for vector loads (None for scalar loads).
    pub vec_value: Option<[u32; 8]>,
    /// Cycle at which this write becomes visible.
    pub ready_cycle: u64,
}

// ============================================================================
// Event Tracing
// ============================================================================

/// Types of events that can be recorded for profiling and trace export.
///
/// Each variant maps to a hardware trace event code from the AIE2 tile trace
/// units (Core module and Memory module). This alignment allows direct
/// comparison between emulator traces and hardware traces in Perfetto.
///
/// See AM025 Trace Event Codes and AIE2Schedule.td CoreEvent definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    // -- Instruction events (Core module trace) --
    // Per-class events matching hardware CoreEvent codes from AIE2Schedule.td.

    /// Vector instruction executed (VMAC, VADD, VCMP, etc.).
    /// Maps to hardware INSTR_VECTOR.
    InstrVector { pc: u32 },
    /// Load instruction executed.
    /// Maps to hardware INSTR_LOAD.
    InstrLoad { pc: u32 },
    /// Store instruction executed.
    /// Maps to hardware INSTR_STORE.
    InstrStore { pc: u32 },
    /// Call instruction executed (jl).
    /// Maps to hardware INSTR_CALL.
    InstrCall { pc: u32 },
    /// Return instruction executed (ret).
    /// Maps to hardware INSTR_RETURN.
    InstrReturn { pc: u32 },
    /// Lock acquire request instruction.
    /// Maps to hardware INSTR_LOCK_ACQUIRE_REQ.
    InstrLockAcquireReq { pc: u32 },
    /// Lock release request instruction.
    /// Maps to hardware INSTR_LOCK_RELEASE_REQ.
    InstrLockReleaseReq { pc: u32 },
    /// Stream get instruction.
    /// Maps to hardware INSTR_STREAM_GET.
    InstrStreamGet { pc: u32 },
    /// Stream put instruction.
    /// Maps to hardware INSTR_STREAM_PUT.
    InstrStreamPut { pc: u32 },
    /// User-defined event instruction (`event #0` or `event #1`).
    /// Maps to hardware INSTR_EVENT_0 (id=0) or INSTR_EVENT_1 (id=1).
    InstrEvent { pc: u32, id: u8 },

    // -- Stall events (Core module trace) --

    /// Memory access stall.
    /// Maps to hardware MEMORY_STALL.
    MemoryStall { cycles: u8 },
    /// Lock acquire stall.
    /// Maps to hardware LOCK_STALL.
    LockStall { cycles: u8 },
    /// Stream interface stall.
    /// Maps to hardware STREAM_STALL.
    StreamStall { cycles: u8 },

    // -- DMA events (Memory module trace) --
    // Channel encodes direction: 0-1 = S2MM (input), 2-3 = MM2S (output)
    // for compute tiles. Shim/memtile may have more channels.

    /// DMA channel started a task.
    /// Maps to hardware DMA_x_START_TASK.
    DmaStartTask { channel: u8 },
    /// DMA channel finished one buffer descriptor.
    /// Maps to hardware DMA_x_FINISHED_BD.
    DmaFinishedBd { channel: u8 },
    /// DMA channel finished an entire task (all BDs and repeats).
    /// Maps to hardware DMA_x_FINISHED_TASK.
    DmaFinishedTask { channel: u8 },
    /// DMA channel stalled waiting for a lock.
    /// Maps to hardware DMA_x_STALLED_LOCK.
    DmaStalledLock { channel: u8 },
    /// DMA channel stalled waiting for stream data.
    /// Maps to hardware DMA_x_STREAM_STARVATION.
    DmaStreamStarvation { channel: u8 },

    // -- Lock events (Memory module trace) --

    /// Lock acquired.
    /// Maps to hardware LOCK_n_ACQ.
    LockAcquire { lock_id: u8 },
    /// Lock released.
    /// Maps to hardware LOCK_n_REL.
    LockRelease { lock_id: u8 },

    // -- Core state events --

    /// Core is actively executing.
    /// Maps to hardware ACTIVE_CORE.
    CoreActive,
    /// Core has halted (done instruction).
    /// Maps to hardware DISABLED_CORE.
    CoreDisabled,

    // -- Branch events (emulator-internal, no direct HW trace event) --

    /// Branch taken with source and target PCs.
    BranchTaken { from_pc: u32, to_pc: u32 },
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

/// Pending branch for delay slot handling.
///
/// AIE2 has 5-cycle branch delay slots - after a branch instruction,
/// the next 5 instructions still execute before the branch takes effect.
///
/// The delay_slots counter starts at 6 because `tick()` is called on the
/// same cycle as the branch instruction itself (in the interpreter loop).
/// The first tick brings it to 5, then 5 subsequent instruction cycles
/// bring it to 0, giving exactly 5 executed delay slot instructions.
///
/// For `jl` (call) instructions, `is_call` is set and LR is updated
/// WHEN the delay slots are exhausted (not immediately). This matches
/// hardware behavior where delay slot instructions see the pre-call LR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PendingBranch {
    /// The target address to branch to.
    pub target: u32,
    /// Number of delay slots remaining.
    /// Starts at 6: first tick (on branch cycle) -> 5, then 5 DS instructions -> 0.
    pub delay_slots: u8,
    /// Whether this branch is a call (jl) that should update LR on completion.
    pub is_call: bool,
}

impl PendingBranch {
    /// Create a new pending branch with 5 delay slots.
    ///
    /// Uses initial count of 6 because tick() is called on the branch cycle.
    pub fn new(target: u32) -> Self {
        Self {
            target,
            delay_slots: 6,
            is_call: false,
        }
    }

    /// Create a new pending call (jl) with 5 delay slots.
    ///
    /// When delay slots are exhausted, the caller should set LR = current PC.
    pub fn new_call(target: u32) -> Self {
        Self {
            target,
            delay_slots: 6,
            is_call: true,
        }
    }

    /// Decrement delay slot counter. Returns true if branch should now be taken.
    pub fn tick(&mut self) -> bool {
        if self.delay_slots > 0 {
            self.delay_slots -= 1;
        }
        self.delay_slots == 0
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

    // === Timing ===
    /// Timing context for cycle-accurate execution.
    /// Always present - all execution is cycle-accurate.
    pub timing: TimingContext,

    // === VLIW Bundle Support ===
    /// Snapshot of register files for VLIW parallel read semantics.
    /// When set, reads use the snapshot instead of live registers.
    /// This ensures all reads in a bundle see pre-execution values.
    scalar_snapshot: Option<ScalarRegisterFile>,
    pointer_snapshot: Option<PointerRegisterFile>,
    modifier_snapshot: Option<ModifierRegisterFile>,

    // === Branch Delay Slot Support ===
    /// Pending branch waiting for delay slots to complete.
    /// AIE2 has 5-cycle branch delay slots - after a branch is decided,
    /// the next 5 instructions still execute before the branch takes effect.
    pending_branch: Option<PendingBranch>,

    // === Load Latency Pipeline ===
    /// Deferred register writes from memory load operations.
    /// Loads have a 7-cycle pipeline latency (AIE2Schedule.td). The write to
    /// the destination register is deferred until `ready_cycle` is reached.
    pending_writes: Vec<PendingWrite>,
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
    ///
    /// All execution is cycle-accurate with:
    /// - Hazard detection (RAW, WAW, WAR)
    /// - Memory bank conflict modeling
    /// - Event tracing (can be disabled via `timing.disable_tracing()`)
    pub fn new() -> Self {
        let mut timing = TimingContext::new();
        timing.enable_tracing(); // Enable event tracing by default for profiling
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
            lr_reg: super::registers::LR_REG_INDEX, // Dedicated special register index
            timing,
            scalar_snapshot: None,
            pointer_snapshot: None,
            modifier_snapshot: None,
            pending_branch: None,
            pending_writes: Vec::new(),
        }
    }

    /// Create a new context with cycle-accurate timing enabled.
    ///
    /// This is an alias for `new()` - all execution is now cycle-accurate.
    #[deprecated(since = "0.2.0", note = "All execution is now cycle-accurate. Use new() instead.")]
    pub fn new_with_timing() -> Self {
        Self::new()
    }

    /// Create a new context with initial stack pointer.
    pub fn with_stack(stack_addr: u32) -> Self {
        let mut ctx = Self::new();
        ctx.set_sp(stack_addr);
        ctx
    }

    /// Enable cycle-accurate timing.
    ///
    /// This is now a no-op - all execution is cycle-accurate.
    #[deprecated(since = "0.2.0", note = "All execution is now cycle-accurate. This method is a no-op.")]
    pub fn enable_timing(&mut self) {
        // No-op - timing is always enabled
    }

    /// Disable cycle-accurate timing.
    ///
    /// This is now a no-op - all execution is cycle-accurate.
    #[deprecated(since = "0.2.0", note = "All execution is now cycle-accurate. This method is a no-op.")]
    pub fn disable_timing(&mut self) {
        // No-op - timing cannot be disabled
    }

    /// Check if cycle-accurate timing is enabled.
    ///
    /// Always returns `true` - all execution is cycle-accurate.
    #[inline]
    pub fn has_timing(&self) -> bool {
        true
    }

    /// Get the timing context.
    #[inline]
    pub fn timing_context(&self) -> &TimingContext {
        &self.timing
    }

    /// Get mutable timing context.
    #[inline]
    pub fn timing_context_mut(&mut self) -> &mut TimingContext {
        &mut self.timing
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

    // === VLIW Bundle Support ===

    /// Begin a VLIW bundle execution.
    ///
    /// Takes a snapshot of scalar, pointer, and modifier registers so that
    /// all reads within the bundle see the pre-execution values, implementing
    /// VLIW parallel semantics where all operations execute "simultaneously".
    #[inline]
    pub fn begin_bundle(&mut self) {
        self.scalar_snapshot = Some(self.scalar.clone());
        self.pointer_snapshot = Some(self.pointer.clone());
        self.modifier_snapshot = Some(self.modifier.clone());
    }

    /// End a VLIW bundle execution.
    ///
    /// Clears register snapshots. Writes that occurred during bundle
    /// execution are already committed to the live registers.
    #[inline]
    pub fn end_bundle(&mut self) {
        self.scalar_snapshot = None;
        self.pointer_snapshot = None;
        self.modifier_snapshot = None;
    }

    // === Load Latency Methods ===

    /// Queue a scalar load result for deferred register write.
    ///
    /// The write will become visible after `latency` cycles, modeling
    /// the AIE2 memory load pipeline.
    pub fn queue_scalar_load(&mut self, dest: Operand, value: u32, latency: u64) {
        self.pending_writes.push(PendingWrite {
            dest,
            scalar_value: value,
            vec_value: None,
            ready_cycle: self.cycles + latency,
        });
    }

    /// Queue a vector load result for deferred register write.
    pub fn queue_vector_load(&mut self, dest: Operand, value: [u32; 8], latency: u64) {
        self.pending_writes.push(PendingWrite {
            dest,
            scalar_value: 0,
            vec_value: Some(value),
            ready_cycle: self.cycles + latency,
        });
    }

    /// Flush all pending writes immediately, ignoring cycle timing.
    ///
    /// Used in unit tests where memory operations are tested in isolation
    /// without the cycle-accurate executor. Production code should use
    /// `commit_pending_writes()` instead.
    #[cfg(test)]
    pub fn flush_pending_writes(&mut self) {
        let writes: Vec<_> = self.pending_writes.drain(..).collect();
        for pw in &writes {
            self.apply_pending_write(pw);
        }
    }

    /// Commit all pending writes whose ready_cycle has been reached.
    ///
    /// Call this at the start of each cycle, BEFORE begin_bundle(), so that
    /// load results become visible at the correct time and are captured by
    /// the VLIW snapshot.
    pub fn commit_pending_writes(&mut self) {
        let current = self.cycles;
        // Commit writes whose ready_cycle has been reached.
        //
        // Using `<=` matches LLVM's LoadLatency semantics: a load with
        // latency L issued at cycle C has ready_cycle = C + L, and a
        // dependent instruction at cycle C + L can read the new value.
        //
        // When multiple pending writes target the SAME register, we must
        // apply them in ready_cycle order so the latest one wins. We sort
        // the ready writes by ready_cycle before applying.
        let mut ready_indices: Vec<usize> = Vec::new();
        for (i, pw) in self.pending_writes.iter().enumerate() {
            if pw.ready_cycle <= current {
                ready_indices.push(i);
            }
        }
        // Sort by ready_cycle so earlier writes are applied first
        // (later writes to the same register overwrite them correctly)
        ready_indices.sort_by_key(|&i| self.pending_writes[i].ready_cycle);
        // Drain in reverse index order to avoid invalidating indices
        // We collect the writes first, then remove them
        let mut ready_writes: Vec<PendingWrite> = Vec::with_capacity(ready_indices.len());
        // Remove from highest index to lowest to preserve indices
        ready_indices.sort_unstable_by(|a, b| b.cmp(a));
        for &idx in &ready_indices {
            ready_writes.push(self.pending_writes.swap_remove(idx));
        }
        // Sort by ready_cycle ascending and apply
        ready_writes.sort_by_key(|pw| pw.ready_cycle);
        for pw in &ready_writes {
            self.apply_pending_write(pw);
        }
    }

    /// Apply a single pending write to the register file.
    fn apply_pending_write(&mut self, pw: &PendingWrite) {
        match &pw.dest {
            Operand::ScalarReg(r) => self.scalar.write(*r, pw.scalar_value),
            Operand::PointerReg(r) => self.pointer.write(*r, pw.scalar_value),
            Operand::ModifierReg(r) => self.modifier.write(*r, pw.scalar_value),
            Operand::VectorReg(r) => {
                if let Some(vec) = &pw.vec_value {
                    self.vector.write(*r, *vec);
                }
            }
            _ => {}
        }
    }

    /// Read a scalar register with VLIW semantics.
    ///
    /// If inside a bundle (snapshot exists), returns the pre-execution value.
    /// Otherwise, returns the current (live) value.
    #[inline]
    pub fn scalar_read(&self, reg: u8) -> u32 {
        if let Some(snapshot) = &self.scalar_snapshot {
            snapshot.read(reg)
        } else {
            self.scalar.read(reg)
        }
    }

    /// Read a pointer register with VLIW semantics.
    #[inline]
    pub fn pointer_read(&self, reg: u8) -> u32 {
        if let Some(snapshot) = &self.pointer_snapshot {
            snapshot.read(reg)
        } else {
            self.pointer.read(reg)
        }
    }

    /// Read a modifier register with VLIW semantics.
    #[inline]
    pub fn modifier_read(&self, reg: u8) -> u32 {
        if let Some(snapshot) = &self.modifier_snapshot {
            snapshot.read(reg)
        } else {
            self.modifier.read(reg)
        }
    }

    // === Branch Delay Slot Support ===

    /// Set a pending branch with 5 delay slots.
    ///
    /// Called when a branch instruction is executed. The branch won't
    /// actually change the PC until 5 more instructions have executed.
    #[inline]
    pub fn set_pending_branch(&mut self, target: u32) {
        // If there's already a pending branch, the new one replaces it
        // (this matches hardware behavior for back-to-back branches)
        self.pending_branch = Some(PendingBranch::new(target));
        log::debug!("Branch to 0x{:X} pending at PC=0x{:X}, 5 delay slots", target, self.pc);
    }

    /// Set a pending call (jl) with 5 delay slots.
    ///
    /// Like `set_pending_branch`, but also defers the LR update until
    /// delay slots are exhausted. At that point, LR is set to the current
    /// PC (the first instruction after all delay slots). This matches
    /// hardware behavior where delay slot instructions see the pre-call LR.
    #[inline]
    pub fn set_pending_call(&mut self, target: u32) {
        self.pending_branch = Some(PendingBranch::new_call(target));
        log::debug!("Call to 0x{:X} pending, 5 delay slots (LR deferred)", target);
    }

    /// Check if there's a pending branch.
    #[inline]
    pub fn has_pending_branch(&self) -> bool {
        self.pending_branch.is_some()
    }

    /// Get the pending branch target (if any).
    #[inline]
    pub fn pending_branch_target(&self) -> Option<u32> {
        self.pending_branch.map(|b| b.target)
    }

    /// Tick the delay slot counter after executing an instruction.
    ///
    /// Returns `Some(target)` if delay slots are exhausted and branch
    /// should now be taken, `None` otherwise.
    ///
    /// For call instructions (is_call=true), also updates LR to the
    /// current PC when delay slots are exhausted. This is the address
    /// of the first instruction after all delay slots, which is the
    /// correct return address.
    #[inline]
    pub fn tick_delay_slots(&mut self) -> Option<u32> {
        if let Some(ref mut pending) = self.pending_branch {
            if pending.tick() {
                let target = pending.target;
                let is_call = pending.is_call;
                self.pending_branch = None;
                if is_call {
                    // LR = current PC = first instruction after all delay slots.
                    // This is deferred from the jl execution to match hardware
                    // pipeline behavior (delay slots see the old LR).
                    let return_addr = self.pc;
                    self.set_lr(return_addr);
                    log::debug!(
                        "Call delay slots exhausted, LR=0x{:X}, branching to 0x{:X}",
                        return_addr, target
                    );
                } else {
                    log::debug!("Delay slots exhausted, branching to 0x{:X}", target);
                }
                return Some(target);
            }
        }
        None
    }

    /// Clear any pending branch (used on halt or error).
    #[inline]
    pub fn clear_pending_branch(&mut self) {
        self.pending_branch = None;
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
    ///
    /// Accepts indices 0-47 (including special register slots 32-47).
    pub fn set_lr_register(&mut self, reg: u8) {
        assert!((reg as usize) < super::registers::NUM_SCALAR_REGS,
                "lr register index {} out of range", reg);
        self.lr_reg = reg;
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
        use crate::interpreter::state::LR_REG_INDEX;
        let mut ctx = ExecutionContext::new();

        ctx.set_lr(0x2000);
        assert_eq!(ctx.lr(), 0x2000);
        // lr is stored at dedicated index 32, NOT r0
        assert_eq!(ctx.scalar.read(LR_REG_INDEX), 0x2000);
        // r0 should be unaffected
        assert_eq!(ctx.scalar.read(0), 0);
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
    fn test_timing_always_enabled() {
        // All contexts now have timing enabled
        let ctx = ExecutionContext::new();
        assert!(ctx.has_timing());

        // Legacy constructor also has timing
        #[allow(deprecated)]
        let ctx_timed = ExecutionContext::new_with_timing();
        assert!(ctx_timed.has_timing());
    }

    #[test]
    fn test_timing_context_access() {
        let mut ctx = ExecutionContext::new();

        // Can always access timing context
        assert!(ctx.has_timing());

        // Access timing context directly (no longer Option)
        ctx.timing_context_mut().hazard_stalls = 5;
        assert_eq!(ctx.timing_context().hazard_stalls, 5);

        // Deprecated methods are no-ops
        #[allow(deprecated)]
        ctx.enable_timing();
        assert!(ctx.has_timing());

        #[allow(deprecated)]
        ctx.disable_timing();
        assert!(ctx.has_timing()); // Still enabled - disable is now a no-op
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
        log.record(10, EventType::InstrLoad { pc: 0x100 });
        assert!(log.is_empty());
    }

    #[test]
    fn test_event_log_record_when_enabled() {
        let mut log = EventLog::new();
        log.enable();

        log.record(10, EventType::InstrLoad { pc: 0x100 });
        log.record(11, EventType::InstrVector { pc: 0x100 });

        assert_eq!(log.len(), 2);
        assert_eq!(log.events()[0].cycle, 10);
        assert_eq!(log.events()[1].cycle, 11);
    }

    #[test]
    fn test_event_log_clear() {
        let mut log = EventLog::new();
        log.enable();

        log.record(1, EventType::CoreDisabled);
        log.record(2, EventType::CoreDisabled);
        assert_eq!(log.len(), 2);

        log.clear();
        assert!(log.is_empty());
    }

    #[test]
    fn test_event_log_circular_buffer() {
        let mut log = EventLog::with_capacity(3);
        log.enable();

        log.record(1, EventType::InstrLoad { pc: 0x100 });
        log.record(2, EventType::InstrLoad { pc: 0x104 });
        log.record(3, EventType::InstrLoad { pc: 0x108 });

        // At capacity, next record should drop oldest
        log.record(4, EventType::InstrLoad { pc: 0x10C });

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
        timing.record_event(30, EventType::CoreDisabled);
        assert_eq!(timing.events.len(), 1); // No new event recorded
    }

    #[test]
    fn test_event_type_variants() {
        // Test that all event variants can be created and recorded.
        // Each maps to a hardware trace event code.
        let events = vec![
            // Instruction events
            EventType::InstrVector { pc: 0x100 },
            EventType::InstrLoad { pc: 0x104 },
            EventType::InstrStore { pc: 0x108 },
            EventType::InstrCall { pc: 0x10C },
            EventType::InstrReturn { pc: 0x110 },
            EventType::InstrLockAcquireReq { pc: 0x114 },
            EventType::InstrLockReleaseReq { pc: 0x118 },
            EventType::InstrStreamGet { pc: 0x11C },
            EventType::InstrStreamPut { pc: 0x120 },
            EventType::InstrEvent { pc: 0x124, id: 0 },
            EventType::InstrEvent { pc: 0x128, id: 1 },
            // Stall events
            EventType::MemoryStall { cycles: 2 },
            EventType::LockStall { cycles: 3 },
            EventType::StreamStall { cycles: 1 },
            // DMA events
            EventType::DmaStartTask { channel: 0 },
            EventType::DmaFinishedBd { channel: 1 },
            EventType::DmaFinishedTask { channel: 2 },
            EventType::DmaStalledLock { channel: 0 },
            EventType::DmaStreamStarvation { channel: 1 },
            // Lock events
            EventType::LockAcquire { lock_id: 5 },
            EventType::LockRelease { lock_id: 5 },
            // Core state
            EventType::CoreActive,
            EventType::CoreDisabled,
            // Branch (emulator-internal)
            EventType::BranchTaken { from_pc: 0x100, to_pc: 0x200 },
        ];

        let mut log = EventLog::new();
        log.enable();

        for (i, event) in events.into_iter().enumerate() {
            log.record(i as u64, event);
        }

        assert_eq!(log.len(), 24);
    }
}
