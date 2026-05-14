//! Core interpreter implementation.
//!
//! The interpreter manages the fetch-decode-execute loop for a single AIE2 core.

use crate::device::tile::Tile;
use crate::interpreter::bundle::VliwBundle;
use crate::interpreter::decode::InstructionDecoder;
use crate::interpreter::execute::CycleAccurateExecutor;
use crate::interpreter::state::{EventType, ExecutionContext};
use crate::interpreter::traits::{DecodeError, Decoder, ExecuteResult, Executor};

/// Core execution status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoreStatus {
    /// Core is ready to execute.
    #[default]
    Ready,
    /// Core is running.
    Running,
    /// Core is waiting on lock acquisition.
    /// Stores raw lock ID (0-63) preserving quadrant routing information.
    WaitingLock { raw_lock_id: u8 },
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
    /// Stalled waiting on lock. Stores raw lock ID (0-63) with quadrant.
    WaitLock { raw_lock_id: u8 },
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

/// Period (in cycles) at which a LOCK_STALL trace event is re-emitted
/// while the core remains in `WaitingLock`. HW emits LOCK_STALL on every
/// cycle the WaitLock instruction is unresolved (the trace controller's
/// LOCK_STALL signal is level, not edge, so the core trace unit emits a
/// frame each cycle the signal is asserted). Measured on
/// `add_one_using_dma`: HW=4394 LOCK_STALL events over the stall window.
///
/// Previous value was 1024, based on the assumption that LOCK_STALL was
/// perf-counter-driven (PERF_CTRL0 counting ACTIVE_CORE with threshold
/// 1024). That model was wrong; the bridge auto-mode trace decode in
/// 2026-05-11 surfaced that HW LOCK_STALL is per-cycle of held stall.
/// See `2026-05-11-emu-dma-pipeline-too-fast-misses-stalls.md`.
const LOCK_STALL_TRACE_PERIOD: u64 = 1;

/// Per-core interpreter.
///
/// Manages the execution of a single AIE2 compute core by coordinating
/// the decoder and executor.
///
/// # Default Configuration
///
/// The default configuration uses:
/// - `InstructionDecoder`: O(1) instruction decoder from llvm-aie TableGen files
/// - `CycleAccurateExecutor`: Cycle-accurate executor with AM020 timing
///
/// If llvm-aie is not available, the decoder will be empty and return
/// unknown operations for all instructions.
pub struct CoreInterpreter<D = InstructionDecoder, E = CycleAccurateExecutor>
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
    /// Cycles accumulated since the last periodic LOCK_STALL trace
    /// emission while the core is in `WaitingLock`. Reset on acquire
    /// and on `reset()`.
    lock_stall_periodic: u64,
}

impl CoreInterpreter<InstructionDecoder, CycleAccurateExecutor> {
    /// Create a new interpreter with default decoder and executor.
    ///
    /// Uses InstructionDecoder loaded from llvm-aie at ../llvm-aie.
    /// Falls back to an empty decoder if llvm-aie is not found.
    pub fn default_new() -> Self {
        Self::new(InstructionDecoder::load_default(), CycleAccurateExecutor::new())
    }

    /// Execute a single instruction cycle with full neighbor lock routing
    /// and optional cross-tile memory access.
    ///
    /// All four lock quadrants per AIE2 getLockLocalBaseIndex:
    /// South (0-15), West (16-31), North (32-47), East/Internal (48-63).
    pub fn step_with_neighbor_locks(
        &mut self,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
        neighbor_locks: &mut crate::interpreter::execute::NeighborLocks,
        neighbors: Option<&mut crate::interpreter::execute::NeighborMemory>,
        view: Option<&crate::device::state::NeighborView>,
    ) -> StepResult {
        self.step_internal(ctx, tile, Some(neighbor_locks), neighbors, view)
    }

    /// Execute a single instruction cycle with South-only lock routing
    /// and optional cross-tile memory access (backward-compatible).
    pub fn step_with_mem_locks(
        &mut self,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
        mem_tile_locks: Option<&mut [crate::device::tile::Lock]>,
        neighbors: Option<&mut crate::interpreter::execute::NeighborMemory>,
        view: Option<&crate::device::state::NeighborView>,
    ) -> StepResult {
        let mut nlocks = crate::interpreter::execute::NeighborLocks::south_only(mem_tile_locks);
        self.step_internal(ctx, tile, Some(&mut nlocks), neighbors, view)
    }

    /// Internal step implementation.
    fn step_internal(
        &mut self,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
        neighbor_locks: Option<&mut crate::interpreter::execute::NeighborLocks>,
        neighbors: Option<&mut crate::interpreter::execute::NeighborMemory>,
        view: Option<&crate::device::state::NeighborView>,
    ) -> StepResult {
        // Check if halted
        if self.is_halted() {
            return StepResult::Halt;
        }

        // Try to resume from stall (pass neighbor locks for cross-tile routing)
        if let Some(result) = self.try_resume_stall(ctx, tile, neighbor_locks.as_ref().map(|nl| &**nl)) {
            return result;
        }

        // Drain any deferred-PC events whose target cycle has been reached,
        // stamping each with the current issue PC. Models the HW trace
        // controller's PC sampling pipeline depth; see TRACE_PC_PIPELINE_DEPTH.
        let current_cycle = ctx.cycles;
        let current_pc = ctx.pc();
        ctx.timing_context_mut().drain_deferred_pc(current_cycle, current_pc);

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
                log::debug!(
                    "DecodeError at PC=0x{:x}, {} bytes avail, prog_len=0x{:x}, \
                     pending_branch={:?}: {:?}",
                    pc,
                    bytes.len(),
                    program_mem.len(),
                    ctx.pending_branch_target(),
                    e
                );
                self.status = CoreStatus::Error;
                return StepResult::DecodeError(e);
            }
        };

        let bundle_size = bundle.size();

        // Execute bundle with neighbor locks and neighbor memory for proper routing
        self.status = CoreStatus::Running;
        let result = if let Some(nlocks) = neighbor_locks {
            self.executor
                .execute_with_neighbor_locks(&bundle, ctx, tile, nlocks, neighbors, view)
        } else {
            self.executor.execute_with_mem_tile(&bundle, ctx, tile, None, neighbors, view)
        };

        // Save bundle for debugging
        self.last_bundle = Some(bundle);

        // Handle execution result (same as step())
        match result {
            crate::interpreter::traits::ExecuteResult::Continue => {
                ctx.advance_pc(bundle_size as u32);
                if let Some(branch_target) = ctx.tick_delay_slots() {
                    ctx.set_pc(branch_target);
                } else {
                    if let Some(info) = ctx.check_hardware_loop(pc) {
                        let cycle = ctx.cycles;
                        ctx.timing_context_mut().record_event(
                            cycle,
                            EventType::LoopBoundary {
                                lc_before: info.lc_before,
                                lc_after: info.lc_after,
                                le_pc: info.le_pc,
                            },
                        );
                        tile.core_trace.notify_loop_boundary(cycle, info.lc_before, info.lc_after);
                    }
                }
                self.status = CoreStatus::Ready;
                StepResult::Continue
            }

            crate::interpreter::traits::ExecuteResult::Branch { target } => {
                ctx.set_pending_branch(target);
                ctx.advance_pc(bundle_size as u32);
                if let Some(branch_target) = ctx.tick_delay_slots() {
                    ctx.set_pc(branch_target);
                    ctx.delay_pending_writes(1);
                } else {
                    if let Some(info) = ctx.check_hardware_loop(pc) {
                        let cycle = ctx.cycles;
                        ctx.timing_context_mut().record_event(
                            cycle,
                            EventType::LoopBoundary {
                                lc_before: info.lc_before,
                                lc_after: info.lc_after,
                                le_pc: info.le_pc,
                            },
                        );
                        tile.core_trace.notify_loop_boundary(cycle, info.lc_before, info.lc_after);
                    }
                }
                self.status = CoreStatus::Ready;
                StepResult::Continue
            }

            crate::interpreter::traits::ExecuteResult::Call { target } => {
                ctx.set_pending_call(target);
                ctx.advance_pc(bundle_size as u32);
                if let Some(branch_target) = ctx.tick_delay_slots() {
                    ctx.set_pc(branch_target);
                    ctx.delay_pending_writes(1);
                } else {
                    if let Some(info) = ctx.check_hardware_loop(pc) {
                        let cycle = ctx.cycles;
                        ctx.timing_context_mut().record_event(
                            cycle,
                            EventType::LoopBoundary {
                                lc_before: info.lc_before,
                                lc_after: info.lc_after,
                                le_pc: info.le_pc,
                            },
                        );
                        tile.core_trace.notify_loop_boundary(cycle, info.lc_before, info.lc_after);
                    }
                }
                self.status = CoreStatus::Ready;
                StepResult::Continue
            }

            crate::interpreter::traits::ExecuteResult::WaitLock { raw_lock_id } => {
                self.status = CoreStatus::WaitingLock { raw_lock_id };
                StepResult::WaitLock { raw_lock_id }
            }

            crate::interpreter::traits::ExecuteResult::WaitDma { channel } => {
                self.status = CoreStatus::WaitingDma { channel };
                StepResult::WaitDma { channel }
            }

            crate::interpreter::traits::ExecuteResult::WaitStream { port } => {
                self.status = CoreStatus::WaitingStream { port };
                StepResult::WaitStream { port }
            }

            crate::interpreter::traits::ExecuteResult::Halt => {
                self.status = CoreStatus::Halted;
                StepResult::Halt
            }

            crate::interpreter::traits::ExecuteResult::Error { message } => {
                self.status = CoreStatus::Error;
                StepResult::ExecError(message)
            }
        }
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
            lock_stall_periodic: 0,
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
            CoreStatus::WaitingLock { .. } | CoreStatus::WaitingDma { .. } | CoreStatus::WaitingStream { .. }
        )
    }

    /// Get the last decoded bundle (for debugging).
    pub fn last_bundle(&self) -> Option<&VliwBundle> {
        self.last_bundle.as_ref()
    }

    /// Compute the trace-visible PC for a core stalled in a lock-acquire.
    ///
    /// AIE2's pipeline fetches and decodes several instructions ahead of
    /// execute. When ACQ stalls in execute, fetch+decode have already
    /// advanced through the function epilog (RET + delay slots) and pin
    /// at the start of the last delay slot, where the pipeline buffer
    /// fills (the next instruction can't be issued until the branch
    /// commits, and the branch can't commit until ACQ resolves). The
    /// trace unit reports this fetch PC as the core's PC -- not the
    /// execute PC where ACQ is parked.
    ///
    /// This helper looks ahead from the WaitLock-stall PC, finds the
    /// next branch (typically the RET in the lock-acquire stub), and
    /// returns the start of that branch's last delay slot. Falls back
    /// to `stall_pc` if the lookahead can't decode or no branch is
    /// found within a small lookahead window.
    pub fn lock_stall_pipeline_pc(&self, program_mem: &[u8], stall_pc: u32) -> u32 {
        use xdna_archspec::aie2::isa::SemanticOp;

        let decode_at = |pc: u32| -> Option<VliwBundle> {
            let off = pc as usize;
            if off >= program_mem.len() {
                return None;
            }
            let end = (off + 16).min(program_mem.len());
            self.decoder.decode(&program_mem[off..end], pc).ok()
        };

        let is_branch = |bundle: &VliwBundle| {
            bundle.slots().iter().flatten().any(|op| {
                matches!(
                    op.semantic,
                    Some(SemanticOp::Br | SemanticOp::BrCond | SemanticOp::Call | SemanticOp::Ret)
                )
            })
        };

        // Decode the stall instruction itself (typically the ACQ); advance
        // past it so we can scan the function epilog for a branch.
        let stall_bundle = match decode_at(stall_pc) {
            Some(b) => b,
            None => return stall_pc,
        };
        let mut pc = stall_pc + stall_bundle.size() as u32;

        // Walk forward up to 5 instructions looking for a branch (covers
        // the common acquire-stub layout: ACQ, then immediately RET).
        for _ in 0..5 {
            let bundle = match decode_at(pc) {
                Some(b) => b,
                None => return stall_pc,
            };
            if is_branch(&bundle) {
                // Found the branch -- walk forward 4 more bundles past it
                // (delay slots 1..=4); the start of the 5th delay slot is
                // where HW's fetch PC pins. AIE2 has 5 branch delay slots.
                let mut delay_pc = pc + bundle.size() as u32;
                for _ in 0..4 {
                    let dbundle = match decode_at(delay_pc) {
                        Some(b) => b,
                        None => return stall_pc,
                    };
                    delay_pc += dbundle.size() as u32;
                }
                return delay_pc;
            }
            pc += bundle.size() as u32;
        }
        stall_pc
    }

    /// Execute a single instruction cycle.
    ///
    /// Returns the result of execution which indicates how to proceed.
    pub fn step(&mut self, ctx: &mut ExecutionContext, tile: &mut Tile) -> StepResult {
        // Check if halted
        if self.is_halted() {
            return StepResult::Halt;
        }

        // Try to resume from stall (no neighbor locks in simple step path)
        if let Some(result) = self.try_resume_stall(ctx, tile, None) {
            return result;
        }

        // Drain any deferred-PC events whose target cycle has been reached.
        // See step_internal for rationale.
        let current_cycle = ctx.cycles;
        let current_pc = ctx.pc();
        ctx.timing_context_mut().drain_deferred_pc(current_cycle, current_pc);

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
                // Check if pending branch should now be taken
                if let Some(branch_target) = ctx.tick_delay_slots() {
                    ctx.set_pc(branch_target);
                    ctx.delay_pending_writes(1);
                } else {
                    if let Some(info) = ctx.check_hardware_loop(pc) {
                        let cycle = ctx.cycles;
                        ctx.timing_context_mut().record_event(
                            cycle,
                            EventType::LoopBoundary {
                                lc_before: info.lc_before,
                                lc_after: info.lc_after,
                                le_pc: info.le_pc,
                            },
                        );
                        tile.core_trace.notify_loop_boundary(cycle, info.lc_before, info.lc_after);
                    }
                }
                self.status = CoreStatus::Ready;
                StepResult::Continue
            }

            ExecuteResult::Branch { target } => {
                // Set pending branch with 5 delay slots
                // The next 5 instructions will still execute before the branch
                ctx.set_pending_branch(target);
                ctx.advance_pc(bundle_size as u32);
                // Check if this was a back-to-back branch and delay slots exhausted
                if let Some(branch_target) = ctx.tick_delay_slots() {
                    ctx.set_pc(branch_target);
                    ctx.delay_pending_writes(1);
                } else {
                    if let Some(info) = ctx.check_hardware_loop(pc) {
                        let cycle = ctx.cycles;
                        ctx.timing_context_mut().record_event(
                            cycle,
                            EventType::LoopBoundary {
                                lc_before: info.lc_before,
                                lc_after: info.lc_after,
                                le_pc: info.le_pc,
                            },
                        );
                        tile.core_trace.notify_loop_boundary(cycle, info.lc_before, info.lc_after);
                    }
                }
                self.status = CoreStatus::Ready;
                StepResult::Continue
            }

            ExecuteResult::Call { target } => {
                // Call (jl): like branch, but LR update is deferred until
                // delay slots are exhausted. set_pending_call sets is_call=true
                // so tick_delay_slots will set LR = current PC at that time.
                ctx.set_pending_call(target);
                ctx.advance_pc(bundle_size as u32);
                if let Some(branch_target) = ctx.tick_delay_slots() {
                    ctx.set_pc(branch_target);
                    ctx.delay_pending_writes(1);
                } else {
                    if let Some(info) = ctx.check_hardware_loop(pc) {
                        let cycle = ctx.cycles;
                        ctx.timing_context_mut().record_event(
                            cycle,
                            EventType::LoopBoundary {
                                lc_before: info.lc_before,
                                lc_after: info.lc_after,
                                le_pc: info.le_pc,
                            },
                        );
                        tile.core_trace.notify_loop_boundary(cycle, info.lc_before, info.lc_after);
                    }
                }
                self.status = CoreStatus::Ready;
                StepResult::Continue
            }

            ExecuteResult::WaitLock { raw_lock_id } => {
                log::info!(
                    "Core({},{}) stall: WaitingLock raw_lock={} pc=0x{:X}",
                    tile.col,
                    tile.row,
                    raw_lock_id,
                    ctx.pc()
                );
                self.status = CoreStatus::WaitingLock { raw_lock_id };
                ctx.record_stall(1);
                StepResult::WaitLock { raw_lock_id }
            }

            ExecuteResult::WaitDma { channel } => {
                log::info!(
                    "Core({},{}) stall: WaitingDma ch={} pc=0x{:X}",
                    tile.col,
                    tile.row,
                    channel,
                    ctx.pc()
                );
                self.status = CoreStatus::WaitingDma { channel };
                ctx.record_stall(1);
                StepResult::WaitDma { channel }
            }

            ExecuteResult::WaitStream { port } => {
                log::info!(
                    "Core({},{}) stall: WaitingStream port={} pc=0x{:X}",
                    tile.col,
                    tile.row,
                    port,
                    ctx.pc()
                );
                self.status = CoreStatus::WaitingStream { port };
                ctx.record_stall(1);
                StepResult::WaitStream { port }
            }

            ExecuteResult::Halt => {
                self.status = CoreStatus::Halted;
                ctx.halted = true;
                ctx.clear_pending_branch(); // Clear any pending branch on halt
                StepResult::Halt
            }

            ExecuteResult::Error { message } => {
                self.status = CoreStatus::Error;
                ctx.clear_pending_branch(); // Clear any pending branch on error
                StepResult::ExecError(message)
            }
        }
    }

    /// Try to resume from a stall condition.
    ///
    /// Returns `Some(result)` if still stalled, `None` if resumed.
    /// For cross-tile lock stalls, `neighbor_locks` provides access to the
    /// correct neighbor tile's lock array via quadrant routing.
    fn try_resume_stall(
        &mut self,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
        neighbor_locks: Option<&crate::interpreter::execute::NeighborLocks>,
    ) -> Option<StepResult> {
        match self.status {
            CoreStatus::WaitingLock { raw_lock_id } => {
                // Route the raw lock ID to the correct tile's lock array,
                // matching the quadrant logic in ControlUnit::route_lock.
                let lock_value = if raw_lock_id >= 48 {
                    // East/Internal = own tile
                    let id = (raw_lock_id - 48) as usize % tile.locks.len();
                    tile.locks[id].value
                } else if raw_lock_id < 16 {
                    // South = row-1 neighbor
                    neighbor_locks
                        .and_then(|nl| nl.south.as_ref())
                        .map(|locks| locks[raw_lock_id as usize % locks.len()].value)
                        .unwrap_or(tile.locks[raw_lock_id as usize % tile.locks.len()].value)
                } else if raw_lock_id < 32 {
                    // West = col-1 neighbor
                    let id = (raw_lock_id - 16) as usize;
                    neighbor_locks
                        .and_then(|nl| nl.west.as_ref())
                        .map(|locks| locks[id % locks.len()].value)
                        .unwrap_or(tile.locks[id % tile.locks.len()].value)
                } else {
                    // North = row+1 neighbor (IDs 32-47)
                    let id = (raw_lock_id - 32) as usize;
                    neighbor_locks
                        .and_then(|nl| nl.north.as_ref())
                        .map(|locks| locks[id % locks.len()].value)
                        .unwrap_or(tile.locks[id % tile.locks.len()].value)
                };

                log::trace!("try_resume_stall: raw_lock {} value = {}", raw_lock_id, lock_value);
                if lock_value > 0 {
                    log::info!("Lock {} available (value={}), resuming execution", raw_lock_id, lock_value);
                    self.status = CoreStatus::Ready;
                    self.lock_stall_periodic = 0;
                    // Re-execute the instruction - it will acquire the lock
                    None
                } else {
                    ctx.record_stall(1);
                    // Periodic LOCK_STALL re-emission. HW emits one LOCK_STALL
                    // event each time the trace controller's cycle-driven
                    // sampler fires while the core is held in stall (typically
                    // every 1024 cycles, driven by a perf counter counting
                    // ACTIVE_CORE -- which keeps incrementing during stall).
                    // The initial event is emitted by the executor on entry to
                    // WaitLock; this path covers the held-stall window.
                    self.lock_stall_periodic += 1;
                    if self.lock_stall_periodic >= LOCK_STALL_TRACE_PERIOD {
                        let cycle = ctx.cycles;
                        let pc = ctx.pc();
                        ctx.timing_context_mut()
                            .record_event(cycle, EventType::LockStall { cycles: 1, pc: Some(pc) });
                        self.lock_stall_periodic = 0;
                    }
                    Some(StepResult::WaitLock { raw_lock_id })
                }
            }

            CoreStatus::WaitingDma { channel } => {
                // Check if DMA is complete
                if !tile.dma_channels[channel as usize].running {
                    log::info!("DMA ch{} complete, resuming execution", channel);
                    self.status = CoreStatus::Ready;
                    None
                } else {
                    ctx.record_stall(1);
                    Some(StepResult::WaitDma { channel })
                }
            }

            CoreStatus::WaitingStream { port } => {
                if port == 254 {
                    // Cascade READ stall (sentinel port 254): SCD empty.
                    // Resume only when cascade input data is available.
                    if tile.has_cascade_input() {
                        log::debug!("Cascade SCD data available, resuming execution");
                        self.status = CoreStatus::Ready;
                        None
                    } else {
                        ctx.record_stall(1);
                        Some(StepResult::WaitStream { port })
                    }
                } else if port == 255 {
                    // Cascade WRITE stall (sentinel port 255): MCD full.
                    // Resume only when cascade output has been drained.
                    if tile.cascade_output.is_empty() {
                        log::debug!("Cascade MCD drained, resuming execution");
                        self.status = CoreStatus::Ready;
                        None
                    } else {
                        ctx.record_stall(1);
                        Some(StepResult::WaitStream { port })
                    }
                } else {
                    // Regular stream stall: check if stream data is available
                    if tile.has_stream_input(port) {
                        log::info!("Stream port {} data available, resuming execution", port);
                        self.status = CoreStatus::Ready;
                        // Re-execute the instruction now that data is available
                        None
                    } else {
                        ctx.record_stall(1);
                        Some(StepResult::WaitStream { port })
                    }
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
                StepResult::WaitLock { .. } | StepResult::WaitDma { .. } | StepResult::WaitStream { .. } => {
                    continue
                }
            }
        }

        (StepResult::Continue, ctx.cycles - start_cycles)
    }

    /// Reset the interpreter state.
    pub fn reset(&mut self) {
        self.status = CoreStatus::Ready;
        self.last_bundle = None;
        self.lock_stall_periodic = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
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
        let interpreter = make_interpreter();

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
