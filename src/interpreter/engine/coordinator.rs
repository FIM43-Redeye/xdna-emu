//! Multi-core coordinator implementation.
//!
//! The engine manages multiple core interpreters and coordinates their execution.
//! It also coordinates DMA engines and host memory for data transfers.
//!
//! # Execution Mode
//!
//! All execution is cycle-accurate using `CycleAccurateExecutor` with full
//! AM020 timing model - hazard detection, memory bank conflicts, branch
//! penalties, and event tracing. This ensures consistent, accurate behavior
//! that matches the real hardware.

use crate::device::dma::ChannelState;
use crate::device::host_memory::HostMemory;
use crate::device::tile::Lock;
use crate::device::DeviceState;
use crate::interpreter::bundle::VliwBundle;
use crate::interpreter::core::{CoreInterpreter, CoreStatus, StepResult};
use crate::interpreter::decode::InstructionDecoder;
use crate::interpreter::execute::{CycleAccurateExecutor, NeighborLocks, NeighborMemory};
use crate::interpreter::state::{EventType, ExecutionContext};
use crate::parser::AieElf;
use xdna_archspec::aie2::SHIM_ROW;
use xdna_archspec::types::TileKind;

/// Build a `NeighborLocks` from the executing tile's isolation byte and
/// per-direction lock slices. A set bit in `isolation` (per
/// [`crate::device::tile::isolation`]) hides the corresponding direction's
/// slice from the executor; `route_lock` then falls back to the own-tile
/// lock array. East/Internal quadrant locks always live on the executing
/// tile, so they're unaffected by isolation.
pub(crate) fn build_neighbor_locks_with_isolation<'a>(
    isolation: u8,
    south: Option<&'a mut [Lock]>,
    west: Option<&'a mut [Lock]>,
    north: Option<&'a mut [Lock]>,
) -> NeighborLocks<'a> {
    use crate::device::tile::isolation as iso;
    NeighborLocks {
        south: if isolation & iso::SOUTH != 0 { None } else { south },
        west: if isolation & iso::WEST != 0 { None } else { west },
        north: if isolation & iso::NORTH != 0 { None } else { north },
    }
}

/// Engine execution status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EngineStatus {
    /// Engine is ready to run.
    #[default]
    Ready,
    /// Engine is running (at least one core active).
    Running,
    /// Engine is paused.
    Paused,
    /// All cores have halted.
    Halted,
    /// No monotonic progress for stall_threshold cycles.
    Stalled,
    /// Engine encountered an error.
    Error,
}

/// Type alias for the cycle-accurate interpreter.
type Interpreter = CoreInterpreter<InstructionDecoder, CycleAccurateExecutor>;

/// Per-core state managed by the engine.
struct CoreState {
    /// Core interpreter (cycle-accurate).
    interpreter: Interpreter,
    /// Execution context (registers, PC, flags).
    context: ExecutionContext,
    /// Cross-tile neighbor memory snapshots, hoisted across steps.
    /// `ensure_snapshot` is gen-aware, so persisting this across steps
    /// turns most per-step calls into cheap cache hits (see `NeighborMemory`).
    neighbors: NeighborMemory,
    /// Is this core enabled?
    enabled: bool,
    /// Absolute count of trace events already consumed from the
    /// EventLog. Tracked in EventLog's monotonic `total_recorded`
    /// space, NOT as an index into the current `events()` slice --
    /// the latter shifts down each time the circular buffer drops its
    /// oldest entry, silently stranding the cursor past end-of-slice
    /// once the buffer wraps.
    trace_events_consumed: u64,
    /// True if this core executed an instruction (StepResult::Continue)
    /// during the current cycle's execute phase. Used by the tick phase to
    /// route ACTIVE_CORE-configured perf counters to tick_active_cycles vs
    /// tick_idle_cycles. Reset to false at the start of each cycle.
    active_this_cycle: bool,
}

impl CoreState {
    /// Create a new core state for a specific tile position.
    ///
    /// The tile's column and row are used to initialize the read-only
    /// CORE_ID register, so the core knows its position in the array.
    fn new(col: u8, row: u8) -> Self {
        Self {
            interpreter: CoreInterpreter::new(
                InstructionDecoder::load_default(),
                CycleAccurateExecutor::new(),
            ),
            context: ExecutionContext::new_for_tile(col, row),
            neighbors: NeighborMemory::new(col as usize, row as usize),
            enabled: false,
            trace_events_consumed: 0,
            active_this_cycle: false,
        }
    }
}

/// Multi-core interpreter engine.
///
/// Coordinates execution across all compute cores in the device.
/// Also manages DMA engines and host memory for data transfers.
///
/// All execution is cycle-accurate with AM020-based timing - register hazard
/// detection (RAW, WAW, WAR), memory bank conflict modeling, branch penalties,
/// and event tracing for profiling.
pub struct InterpreterEngine {
    /// Device state (tiles, memory, locks, DMA engines).
    device: DeviceState,
    /// Host memory for DMA transfers to/from external DDR.
    host_memory: HostMemory,
    /// Per-core execution state (indexed by column * max_rows + row).
    cores: Vec<CoreState>,
    /// Number of columns.
    cols: usize,
    /// Number of rows.
    rows: usize,
    /// Compute row start (first row with compute tiles).
    compute_row_start: usize,
    /// Engine status.
    status: EngineStatus,
    /// Total cycles executed.
    total_cycles: u64,
    /// Total instructions executed across all cores.
    total_instructions: u64,
    /// Auto-run mode.
    auto_run: bool,
    /// Stall detection: last observed total DMA bytes transferred.
    last_dma_bytes: u64,
    /// Stall detection: last observed total lock releases.
    last_lock_releases: u64,
    /// Stall detection: last observed total instructions executed.
    last_instructions: u64,
    /// Consecutive cycles with no monotonic progress.
    stall_cycles: u64,
    /// Cycles of no progress before declaring stall. 0 = disabled.
    stall_threshold: u64,
}

impl InterpreterEngine {
    /// Create a new engine from device state.
    ///
    /// Uses `CycleAccurateExecutor` with full AM020 timing model:
    /// - Register hazard detection (RAW, WAW, WAR)
    /// - Memory bank conflict modeling
    /// - Branch penalty tracking
    /// - Event tracing for profiling
    pub fn new(device: DeviceState) -> Self {
        let cols = device.cols();
        let rows = device.rows();
        let compute_row_start = 2; // Rows 0=shim, 1=memtile, 2+=compute

        // Create core states for all possible positions, each initialized
        // with its tile coordinates so CORE_ID is correct.
        let num_cores = cols * rows;
        let cores = (0..num_cores)
            .map(|idx| {
                let col = (idx / rows) as u8;
                let row = (idx % rows) as u8;
                CoreState::new(col, row)
            })
            .collect();

        Self {
            device,
            host_memory: HostMemory::new(),
            cores,
            cols,
            rows,
            compute_row_start,
            status: EngineStatus::Ready,
            total_cycles: 0,
            total_instructions: 0,
            auto_run: false,
            last_dma_bytes: 0,
            last_lock_releases: 0,
            last_instructions: 0,
            stall_cycles: 0,
            stall_threshold: 0,
        }
    }

    /// Create a new engine with cycle-accurate timing.
    /// Create engine for NPU1 (Phoenix).
    pub fn new_npu1() -> Self {
        Self::new(DeviceState::new_npu1())
    }

    /// Create engine for NPU2 (Strix).
    pub fn new_npu2() -> Self {
        Self::new(DeviceState::new_npu2())
    }

    /// All execution is cycle-accurate.
    pub fn is_cycle_accurate(&self) -> bool {
        true
    }

    /// Get the engine status.
    pub fn status(&self) -> EngineStatus {
        self.status
    }

    /// Set the stall detection threshold. 0 disables stall detection.
    pub fn set_stall_threshold(&mut self, threshold: u64) {
        self.stall_threshold = threshold;
    }

    /// Reset status from Halted to Running.
    ///
    /// The engine halts when no cores are enabled and no DMA is active,
    /// which is correct for its local view. But the NPU instruction
    /// executor may still be configuring and triggering DMA from the
    /// host command stream. The run loop calls this when it knows
    /// external work is still in progress, so `step()` continues
    /// advancing DMA engines and stream switches.
    pub fn force_running(&mut self) {
        if self.status == EngineStatus::Halted {
            self.status = EngineStatus::Running;
        }
    }

    /// Get total cycles executed.
    pub fn total_cycles(&self) -> u64 {
        self.total_cycles
    }

    /// Get total instructions executed across all cores.
    pub fn total_instructions(&self) -> u64 {
        self.total_instructions
    }

    /// Flush trace units and route final trace packets to host DDR.
    ///
    /// Call after execution completes. Flushes all tile trace units so partial
    /// packets get emitted with padding, then runs additional routing passes
    /// to deliver them through the stream switch network to host memory.
    pub fn flush_trace_to_host(&mut self) {
        // Flush all tile trace units so partial packets get emitted with padding.
        // This ensures the final trace data is available for stream routing.
        for col in 0..self.device.array.cols() {
            for row in 0..self.device.array.rows() {
                if let Some(tile) = self.device.array.get_mut(col as u8, row as u8) {
                    tile.core_trace.flush();
                    tile.mem_trace.flush();
                }
            }
        }

        // Route flushed trace packets through the stream switch to host DDR.
        // During normal execution, step_data_movement() runs every step() cycle.
        // But flush() above creates final packets AFTER the execution loop exits,
        // so we need additional routing passes to deliver them through the
        // multi-hop stream switch network.
        //
        // Each word traverses multiple hops (e.g., Compute -> MemTile -> Shim
        // -> DMA = 3-4 hops). Each hop takes one routing pass. Two 8-word
        // packets need ~64 passes in the worst case. Use a generous limit
        // and stop early when nothing moves.
        let mut total_flush_words = 0;
        let mut flush_iters = 0;
        for _ in 0..100 {
            let (_, _streams_moved, words_routed) =
                self.device.array.step_data_movement(&mut self.host_memory);
            total_flush_words += words_routed;
            flush_iters += 1;

            // Check for fatal errors from data movement during flush too
            let fatal_errors = self.device.array.drain_fatal_errors();
            if !fatal_errors.is_empty() {
                for err in &fatal_errors {
                    log::error!("Data movement fatal (flush): {}", err);
                }
                self.status = EngineStatus::Error;
                return;
            }

            // Dispatch any ctrl packet actions produced during flush routing
            {
                let ctrl_actions = self.device.array.drain_ctrl_packet_actions();
                for action in ctrl_actions {
                    self.dispatch_ctrl_action(action);
                }
                self.drain_core_enables();
            }

            // Keep routing as long as trace data is anywhere in the pipeline:
            // pending in trace units, in stream switch slave/master FIFOs, or
            // awaiting inter-tile propagation.
            let trace_in_flight = self.device.array.tiles.iter().any(|t| {
                t.core_trace.has_pending_words()
                    || t.mem_trace.has_pending_words()
                    || t.stream_switch.has_pending_packet()
                    || t.stream_switch.has_pending_data()
            });
            if !trace_in_flight && words_routed == 0 {
                break;
            }
        }
        if total_flush_words > 0 {
            log::info!("Post-flush routing: {} words in {} iterations", total_flush_words, flush_iters);
        }
    }

    /// Flush in-flight control packet data through the stream switch.
    ///
    /// On real hardware, stream switch latency is 1-3 cycles per hop -- far
    /// shorter than the NPU firmware's instruction-to-instruction latency.
    /// So when firmware observes a shim DMA completion (Sync) and then writes
    /// a START_QUEUE register, the control packet data has long since arrived
    /// at the destination tile.
    ///
    /// In our emulator, the NPU executor and stream routing are interleaved
    /// within the same cycle loop. Without explicit flushing, the executor
    /// can fire START_QUEUE before the previous cycle's routing delivers
    /// the control packet's final words to the tile's DMA BD registers.
    ///
    /// Call this between NPU executor advance and engine step to ensure
    /// all in-flight control packet data reaches its destination before
    /// subsequent NPU instructions take effect.
    pub fn flush_ctrl_packets(&mut self) {
        // Run routing passes to deliver pending stream data, then dispatch
        // any control packet actions that were generated.  Limit to a few
        // iterations; control packets traverse at most 4-5 hops.
        for _ in 0..8 {
            let (_, _, words) = self.device.array.step_data_movement(&mut self.host_memory);

            let ctrl_actions = self.device.array.drain_ctrl_packet_actions();
            if ctrl_actions.is_empty() && words == 0 {
                break;
            }

            for action in ctrl_actions {
                self.dispatch_ctrl_action(action);
            }
            self.drain_core_enables();
        }
    }

    /// Get reference to device state.
    pub fn device(&self) -> &DeviceState {
        &self.device
    }

    /// Get mutable reference to device state.
    pub fn device_mut(&mut self) -> &mut DeviceState {
        &mut self.device
    }

    /// Get reference to host memory.
    pub fn host_memory(&self) -> &HostMemory {
        &self.host_memory
    }

    /// Get mutable reference to host memory.
    pub fn host_memory_mut(&mut self) -> &mut HostMemory {
        &mut self.host_memory
    }

    /// Get mutable references to both device state and host memory.
    ///
    /// Needed when NPU instruction execution must interleave DMA stepping
    /// (backpressure on full task queues requires stepping DMA with host memory).
    pub fn device_and_host_memory(&mut self) -> (&mut DeviceState, &mut HostMemory) {
        (&mut self.device, &mut self.host_memory)
    }

    /// Enable a core at (col, row).
    pub fn enable_core(&mut self, col: usize, row: usize) {
        if let Some(core) = self.get_core_mut(col, row) {
            core.enabled = true;
        }
        // Mirror to CoreDebugState so Core_Status register shows enabled.
        if let Some(tile) = self.device.tile_mut(col, row) {
            tile.core_debug.set_enabled(true);
        }
    }

    /// Disable a core at (col, row).
    pub fn disable_core(&mut self, col: usize, row: usize) {
        if let Some(core) = self.get_core_mut(col, row) {
            core.enabled = false;
        }
        if let Some(tile) = self.device.tile_mut(col, row) {
            tile.core_debug.set_enabled(false);
        }
    }

    /// Check if a core is enabled.
    pub fn is_core_enabled(&self, col: usize, row: usize) -> bool {
        self.get_core(col, row).map(|c| c.enabled).unwrap_or(false)
    }

    /// Drain pending core enable/disable events from device state.
    ///
    /// Core_Control register writes (from NPU instructions, CDO, or control
    /// packets) push events to `device.pending_core_enables`. This syncs
    /// those to the engine's internal core state, matching how real hardware
    /// immediately reacts to the register write.
    fn drain_core_enables(&mut self) {
        let events: Vec<_> = self.device.pending_core_enables.drain(..).collect();
        for (col, row, enabled) in events {
            if enabled {
                self.enable_core(col as usize, row as usize);
            } else {
                self.disable_core(col as usize, row as usize);
            }
        }
    }

    /// Get core status at (col, row).
    pub fn core_status(&self, col: usize, row: usize) -> Option<CoreStatus> {
        self.get_core(col, row).map(|c| c.interpreter.status())
    }

    /// Get core execution context at (col, row).
    pub fn core_context(&self, col: usize, row: usize) -> Option<&ExecutionContext> {
        self.get_core(col, row).map(|c| &c.context)
    }

    /// Get mutable core execution context at (col, row).
    pub fn core_context_mut(&mut self, col: usize, row: usize) -> Option<&mut ExecutionContext> {
        self.get_core_mut(col, row).map(|c| &mut c.context)
    }

    /// Set the program counter for a core.
    pub fn set_core_pc(&mut self, col: usize, row: usize, pc: u32) {
        if let Some(core) = self.get_core_mut(col, row) {
            core.context.set_pc(pc);
        }
    }

    /// Set a pointer register for a core.
    ///
    /// AIE2 cores have 8 pointer registers (p0-p7) used for address generation.
    /// This method allows pre-initializing buffer base addresses before kernel
    /// execution, as kernels expect the runtime/CDO to configure these pointers.
    pub fn set_core_pointer(&mut self, col: usize, row: usize, reg: u8, value: u32) {
        if let Some(core) = self.get_core_mut(col, row) {
            core.context.pointer.write(reg, value);
        }
    }

    /// Set a modifier register for a core.
    ///
    /// AIE2 cores have 8 modifier registers (m0-m7) used for addressing modes.
    /// These are typically set to stride values for loop-based memory access.
    pub fn set_core_modifier(&mut self, col: usize, row: usize, reg: u8, value: u32) {
        if let Some(core) = self.get_core_mut(col, row) {
            core.context.modifier.write(reg, value);
        }
    }

    /// Get the last decoded bundle for a core (for debugging).
    ///
    /// Returns the last VLIW bundle that was decoded for the core at (col, row).
    /// Useful for diagnosing decode/execution errors.
    pub fn core_last_bundle(&self, col: usize, row: usize) -> Option<&VliwBundle> {
        self.get_core(col, row).and_then(|c| c.interpreter.last_bundle())
    }

    /// Load an ELF file into a core's memory.
    ///
    /// Parses the ELF, loads program and data segments into the tile's memory,
    /// sets the entry point as the PC, and enables the core.
    ///
    /// Returns the entry point address on success.
    pub fn load_elf_bytes(&mut self, col: usize, row: usize, data: &[u8]) -> Result<u32, String> {
        let elf = AieElf::parse(data).map_err(|e| format!("ELF parse error: {}", e))?;
        let entry = elf.entry_point();

        // Load segments into tile memory
        {
            let tile = self
                .device
                .tile_mut(col, row)
                .ok_or_else(|| format!("Invalid tile coordinates ({}, {})", col, row))?;
            elf.load_into(tile);
        }

        // Set entry point and enable core
        self.set_core_pc(col, row, entry);
        self.enable_core(col, row);

        Ok(entry)
    }

    /// Execute one cycle on all enabled cores and DMA engines.
    ///
    /// The execution order is:
    /// 1. Sync DMA start requests from tiles to DMA engines
    /// 2. Step all DMA engines (so lock releases are visible to cores)
    /// 3. Step all compute cores (with fresh lock snapshot from MemTile)
    /// 4. Update tile DMA channel state from engine state
    ///
    /// Note: DMA steps before cores so that when a core copies MemTile locks
    /// for memory module access (lock IDs 48-63), it sees the locks that
    /// DMA just released. This is critical for producer/consumer sync.
    pub fn step(&mut self) {
        if matches!(self.status, EngineStatus::Halted | EngineStatus::Error) {
            return;
        }

        self.status = EngineStatus::Running;
        let mut any_running = false;
        let mut all_halted = true;

        // Reset per-core active flag. Set to true only for StepResult::Continue
        // in Phase 2; the perf counter tick in Phase 3e reads it to pick
        // tick_active_cycles vs tick_idle_cycles for ACTIVE_CORE-gated counters.
        for core in &mut self.cores {
            core.active_this_cycle = false;
        }

        // Publish the current simulation cycle to the tile array so that
        // register-write side effects (trace unit start/stop, broadcast
        // propagation) can time-stamp events with the right cycle. Also
        // propagates to DMA engines for DMA-side trace events.
        self.device.array.set_dma_cycle(self.total_cycles);

        // Phase 1: Sync DMA start requests from tiles to DMA engines
        self.sync_dma_start_requests();

        // Phase 2: Step each enabled core.
        //
        // Core lock releases are DEFERRED: they write to core_lock_deltas
        // instead of modifying lock.value directly. This models the hardware
        // lock arbiter pipeline -- on real AIE2, the arbiter handles one
        // request per clock cycle (AM020), so a core release at cycle N
        // becomes visible to DMA at cycle N+1.
        //
        // The deferred requests are resolved by the arbiter during Phase 3,
        // so the NEXT cycle's DMA operations will see the updated values.
        //
        // Core lock acquires still read lock.value directly (committed state
        // from previous cycles). This is correct: the core stalls if the
        // precondition isn't met, and the acquire decrement is immediate.
        //
        // DMA-to-core lock visibility (the reverse direction) is handled by
        // step_data_movement's internal lock snapshot/commit, which commits
        // DMA lock releases at the end of the data movement phase. Cores
        // read committed lock state, so DMA releases from cycle N-1 are
        // visible to cores at cycle N -- the same 1-cycle latency as the
        // hardware's lock pipeline.
        for col in 0..self.cols {
            for row in self.compute_row_start..self.rows {
                let idx = col * self.rows + row;

                if !self.cores[idx].enabled {
                    continue;
                }

                // AIE2 lock quadrant routing: copy neighbor locks before stepping,
                // then write changes back after. Per mlir-aie getLockLocalBaseIndex:
                //   South (IDs 0-15):  row-1 neighbor
                //   West  (IDs 16-31): col-1 neighbor
                //   North (IDs 32-47): row+1 neighbor
                //   East  (IDs 48-63): own tile (handled internally)
                let mut south_locks: Option<Vec<crate::device::tile::Lock>> = None;
                let mut west_locks: Option<Vec<crate::device::tile::Lock>> = None;
                let mut north_locks: Option<Vec<crate::device::tile::Lock>> = None;

                if row > SHIM_ROW as usize {
                    if let Some(south) = self.device.tile(col, row - 1) {
                        south_locks = Some(south.locks.clone());
                    }
                }
                if col > 0 {
                    if let Some(west) = self.device.tile(col - 1, row) {
                        west_locks = Some(west.locks.clone());
                    }
                }
                if row + 1 < self.rows {
                    if let Some(north) = self.device.tile(col, row + 1) {
                        north_locks = Some(north.locks.clone());
                    }
                }

                // Split the device so the executing tile and a read-through
                // view of all other tiles come from one borrow. The view
                // is threaded into `step_with_neighbor_locks` so the read
                // sites can `ensure_snapshot` lazily -- only the quadrants
                // the kernel actually accesses get snapshotted, and only
                // on the first read. Both `tile` and `view` drop at the
                // end of this body, freeing `&mut self.device` for
                // `drain_writes` below.
                let Some((tile, view)) = self.device.split_tile_mut(col, row) else {
                    continue;
                };
                if !tile.is_compute() {
                    continue;
                }
                let core = &mut self.cores[idx];

                // Tile_Control isolation: refresh the cached byte on
                // NeighborMemory and gate cross-tile lock slices before
                // they reach the executor. A set bit in tile.isolation
                // blocks every cross-tile transit on that direction.
                // Per-call sites in execute/memory then short-circuit
                // via NeighborMemory's own `is_blocked`. See
                // `crate::device::tile::isolation` for the bit layout.
                core.neighbors.set_isolation(tile.isolation);
                let mut nlocks = build_neighbor_locks_with_isolation(
                    tile.isolation,
                    south_locks.as_mut().map(|v| v.as_mut_slice()),
                    west_locks.as_mut().map(|v| v.as_mut_slice()),
                    north_locks.as_mut().map(|v| v.as_mut_slice()),
                );

                let result = core.interpreter.step_with_neighbor_locks(
                    &mut core.context,
                    tile,
                    &mut nlocks,
                    Some(&mut core.neighbors),
                    Some(&view),
                );

                // Update CoreDebugState with current PC and stall info.
                // This mirrors the interpreter state into the register
                // space so host reads of Core_Status/Core_PC return
                // correct values (matching real hardware behavior).
                // update_pc also drives PC_Event0..3 matching, which can
                // set the pending single-step latch through check_event_halt.
                tile.core_debug.update_pc(core.context.pc());
                // Drain the SSTEP_EVENT latch: if any event during this
                // step matched Debug_Control1.SSTEP_EVENT, request halt
                // now (interpretation (a): the triggering bundle is the
                // last to commit before halt).
                tile.core_debug.consume_pending_single_step();

                // Mirror the live PC onto tile.core.pc as well, so any
                // path that reads tile-side state (broadcast
                // propagation, register effects) sees the running PC
                // instead of the stale CDO-loaded entry value. Without
                // this sync, mode-2's Start anchor_pc reads 0
                // regardless of what the core was actually doing when
                // start_event fired.
                tile.core.pc = core.context.pc();
                match result {
                    StepResult::Continue => {
                        tile.core_debug.update_stalls(false, false, false, false);
                        all_halted = false;
                        any_running = true;
                        self.total_instructions += 1;
                        core.active_this_cycle = true;
                    }
                    StepResult::WaitLock { .. } => {
                        tile.core_debug.update_stalls(false, true, false, false);
                        all_halted = false;
                        any_running = true;
                        // Pipeline-PC adjustment: HW's fetch+decode is
                        // several stages ahead of execute. When ACQ stalls
                        // in execute, the fetch PC pins at the start of
                        // the last delay slot of the next branch (RET in
                        // the acquire stub). Override tile.core.pc so the
                        // trace unit sees the same PC HW would, instead
                        // of the execute PC where ACQ is parked.
                        let stall_pc = core.context.pc();
                        if let Some(prog_mem) = tile.program_memory() {
                            let pipeline_pc = core.interpreter.lock_stall_pipeline_pc(prog_mem, stall_pc);
                            tile.core.pc = pipeline_pc;
                            tile.core_debug.update_pc(pipeline_pc);
                        }
                    }
                    StepResult::WaitDma { .. } => {
                        tile.core_debug.update_stalls(true, false, false, false);
                        all_halted = false;
                        any_running = true;
                    }
                    StepResult::WaitStream { .. } => {
                        tile.core_debug.update_stalls(false, false, true, false);
                        all_halted = false;
                        any_running = true;
                    }
                    StepResult::Halt => {
                        tile.core_debug.set_done(true);
                        tile.core_debug.update_stalls(false, false, false, false);
                    }
                    StepResult::DebugHalt => {
                        // Debug halt is a transient pause -- the program isn't
                        // done. Don't call set_done; just clear stall flags
                        // (the previous instruction has already retired) and
                        // mark the engine as still active. Leaving all_halted
                        // false keeps the engine ticking so DMAs continue and
                        // the host can write Debug_Control0=0 to resume.
                        tile.core_debug.update_stalls(false, false, false, false);
                        all_halted = false;
                    }
                    StepResult::DecodeError(ref e) => {
                        log::error!(
                            "Core({},{}) DecodeError at cycle {}: {:?}",
                            col,
                            row,
                            self.total_cycles,
                            e
                        );
                        self.status = EngineStatus::Error;
                        return;
                    }
                    StepResult::ExecError(ref e) => {
                        log::error!(
                            "Core({},{}) ExecError at cycle {}: {:?}",
                            col,
                            row,
                            self.total_cycles,
                            e
                        );
                        self.status = EngineStatus::Error;
                        return;
                    }
                }

                // Notify core trace unit with any new events since last cycle.
                //
                // Use the coordinator's global cycle (self.total_cycles) rather
                // than the per-core `evt.cycle`. The event log stores the
                // core-local retire counter (ctx.cycles), which freezes during
                // stalls and thus does not reflect the tile clock that real HW
                // trace units observe. Events drained here happened in the
                // current simulation step, so total_cycles is the right stamp.
                let event_log = &core.context.timing_context().events;
                let total = event_log.total_recorded();
                let new_start = core.trace_events_consumed;
                if new_start < total {
                    let cycle = self.total_cycles;
                    for evt in event_log.since(new_start) {
                        if let Some(hw_id) = crate::trace::core_event_to_hw_id(&evt.event) {
                            let pc = crate::trace::event_pc(&evt.event);
                            tile.notify_core_trace_event(hw_id, cycle, pc);
                        }
                    }
                    core.trace_events_consumed = total;
                }

                // Drain buffered cross-tile writes to neighbor tiles.
                // Snapshots stay live; only the writes vec is drained.
                // NLL: `tile` and `view` are no longer in use after this
                // point, so `&mut self.device` is free to be re-borrowed.
                if core.neighbors.has_pending_writes() {
                    core.neighbors.drain_writes(&mut self.device);
                }

                // Submit modified neighbor locks back to neighbor tiles.
                //
                // Instead of writing back directly (which would make core lock
                // releases visible to neighbor DMA in the same cycle), we compute
                // the diff and submit it as a core release to the arbiter. The
                // arbiter resolves in Phase 3, providing the 1-cycle delay.
                fn writeback_locks(
                    device: &mut DeviceState,
                    locks: Option<Vec<crate::device::tile::Lock>>,
                    ncol: usize,
                    nrow: usize,
                ) {
                    if let Some(modified) = locks {
                        if let Some(neighbor) = device.tile_mut(ncol, nrow) {
                            for i in 0..modified.len().min(neighbor.locks.len()) {
                                let delta = (modified[i].value as i16) - (neighbor.locks[i].value as i16);
                                if delta != 0 {
                                    neighbor.defer_core_lock_release(i, delta as i8);
                                }
                            }
                        }
                    }
                }
                // South
                if row > SHIM_ROW as usize {
                    writeback_locks(&mut self.device, south_locks, col, row - 1);
                }
                // West
                if col > 0 {
                    writeback_locks(&mut self.device, west_locks, col - 1, row);
                }
                // North
                if row + 1 < self.rows {
                    writeback_locks(&mut self.device, north_locks, col, row + 1);
                }
            }
        }

        // Phase 3: Step all DMA engines and stream routing.
        //
        // Core lock releases from Phase 2 are pending in tile arbiters.
        // step_data_movement() does: submit DMA lock requests -> resolve
        // all tile arbiters (core + DMA together) -> step DMA channels
        // checking arbiter results -> route streams.
        //
        // Set cycle timestamp on DMA engines before stepping so trace
        // events get the correct cycle number.
        self.device.array.set_dma_cycle(self.total_cycles);

        let (dma_active, streams_moved, _words_routed) =
            self.device.array.step_data_movement(&mut self.host_memory);

        // Check for fatal errors from data movement (impossible-on-hardware
        // conditions like missing packet routes or stream buffer overflows).
        let fatal_errors = self.device.array.drain_fatal_errors();
        if !fatal_errors.is_empty() {
            for err in &fatal_errors {
                log::error!("Data movement fatal: {}", err);
            }
            self.status = EngineStatus::Error;
            return;
        }

        // Dispatch control packet actions through DeviceState for full
        // module dispatch (MemTile DMA BDs, stream switch, etc.).
        let ctrl_actions = self.device.array.drain_ctrl_packet_actions();
        for action in ctrl_actions {
            self.dispatch_ctrl_action(action);
        }
        self.drain_core_enables();

        // Drain memory-module trace events (DMA + locks) into the global
        // trace log and notify trace units so they can produce binary trace
        // packets. Each tile type uses a different event ID namespace:
        // - Compute tiles: mem_event_to_hw_id -> mem_trace
        // - MemTiles: memtile_event_to_hw_id -> mem_trace
        // - Shim tiles: shim_event_to_hw_id -> core_trace (PL module)
        for (col, row, cycle, event) in self.device.array.drain_mem_trace_events() {
            if let Some(tile) = self.device.array.get_mut(col, row) {
                if tile.is_shim() {
                    if let Some(id) = crate::trace::shim_event_to_hw_id(&event) {
                        // Shim DMA/lock events have no program counter.
                        tile.notify_core_trace_event(id, cycle, None);
                    }
                } else if tile.is_mem() {
                    // Memtile DMA events are gated by the DMA_Event_Channel_Selection
                    // register (0xA06A0). Each SEL slot can map to a different
                    // physical channel and a single channel may fire both SEL slots
                    // when both are aimed at it (e.g., reset default = both at ch0).
                    let sel =
                        crate::trace::MemtileDmaEventSel::from_register(tile.memtile_dma_event_chan_sel);
                    for id in crate::trace::memtile_event_to_hw_ids(&event, sel).into_iter().flatten() {
                        tile.notify_mem_trace_event(id, cycle, None);
                    }
                } else {
                    if let Some(id) = crate::trace::mem_event_to_hw_id(&event) {
                        tile.notify_mem_trace_event(id, cycle, None);
                    }
                }
            }
        }

        // Phase 3b: Generate stream switch port events on signal transitions.
        //
        // Four event types per monitored port, all edge-triggered on real
        // silicon (rising edge of the corresponding port signal):
        // - PORT_RUNNING: fires when port becomes active (idle -> active)
        // - PORT_IDLE: fires when port becomes idle (active -> idle)
        // - PORT_STALLED: fires when stall asserts (rising edge only)
        // - PORT_TLAST: fires when TLAST asserts (rising edge only)
        //
        // step_data_movement() seeds these flags from FIFO state at routing
        // start and updates them as data flows. We compare against
        // `tile.prev_port_state` from the previous cycle and only emit on
        // transitions, matching HW behavior. Emitting every active cycle
        // (the old behavior) floods small trace BDs in milliseconds.
        {
            let cycle = self.total_cycles;
            // Collect (tile_idx, hw_event_id, event_type, trace_target) tuples.
            // trace_target: which trace unit to notify (core_trace vs mem_trace).
            let mut port_events: Vec<(usize, u8, EventType, TileKind)> = Vec::new();
            // Per-tile updates to prev_port_state, applied after collection.
            let mut prev_updates: Vec<(usize, u8, bool, bool, bool)> = Vec::new();
            for idx in 0..self.device.array.tiles.len() {
                let tile = &self.device.array.tiles[idx];
                if !tile.core_trace.is_configured() && !tile.mem_trace.is_configured() {
                    continue;
                }
                let tt = tile.tile_kind;

                for event_port in 0..8u8 {
                    if let Some((port_idx, is_master)) = tile.event_port_selection[event_port as usize] {
                        let port = if is_master {
                            tile.stream_switch.masters.get(port_idx as usize)
                        } else {
                            tile.stream_switch.slaves.get(port_idx as usize)
                        };
                        let Some(port) = port else { continue };

                        let (prev_active, prev_stalled, prev_tlast) =
                            tile.prev_port_state[event_port as usize];
                        let cur_active = port.cycle_active;
                        let cur_stalled = port.cycle_stalled;
                        let cur_tlast = port.cycle_tlast;

                        // PORT_RUNNING fires on idle -> active transition;
                        // PORT_IDLE fires on active -> idle transition.
                        if cur_active && !prev_active {
                            let hw_id = match tt {
                                TileKind::Compute => crate::trace::core_port_running_hw_id(event_port),
                                TileKind::Mem => crate::trace::memtile_port_running_hw_id(event_port),
                                TileKind::ShimNoc | TileKind::ShimPl => {
                                    crate::trace::shim_port_running_hw_id(event_port)
                                }
                            };
                            port_events.push((idx, hw_id, EventType::PortRunning { port: event_port }, tt));
                        } else if !cur_active && prev_active {
                            let hw_id = match tt {
                                TileKind::Compute => crate::trace::core_port_idle_hw_id(event_port),
                                TileKind::Mem => crate::trace::memtile_port_idle_hw_id(event_port),
                                TileKind::ShimNoc | TileKind::ShimPl => {
                                    crate::trace::shim_port_idle_hw_id(event_port)
                                }
                            };
                            port_events.push((idx, hw_id, EventType::PortIdle { port: event_port }, tt));
                        }

                        // PORT_STALLED and PORT_TLAST fire only on rising edge.
                        if cur_stalled && !prev_stalled {
                            let hw_id = match tt {
                                TileKind::Compute => crate::trace::core_port_stalled_hw_id(event_port),
                                TileKind::Mem => crate::trace::memtile_port_stalled_hw_id(event_port),
                                TileKind::ShimNoc | TileKind::ShimPl => {
                                    crate::trace::shim_port_stalled_hw_id(event_port)
                                }
                            };
                            port_events.push((idx, hw_id, EventType::PortStalled { port: event_port }, tt));
                        }

                        if cur_tlast && !prev_tlast {
                            let hw_id = match tt {
                                TileKind::Compute => crate::trace::core_port_tlast_hw_id(event_port),
                                TileKind::Mem => crate::trace::memtile_port_tlast_hw_id(event_port),
                                TileKind::ShimNoc | TileKind::ShimPl => {
                                    crate::trace::shim_port_tlast_hw_id(event_port)
                                }
                            };
                            port_events.push((idx, hw_id, EventType::PortTlast { port: event_port }, tt));
                        }

                        prev_updates.push((idx, event_port, cur_active, cur_stalled, cur_tlast));
                    }
                }
            }
            for (idx, hw_id, _evt, tt) in port_events {
                let tile = &mut self.device.array.tiles[idx];
                // Compute tiles: core_trace (CoreEvent namespace)
                // Shim tiles: core_trace (PL module, single trace unit)
                // MemTiles: mem_trace (MemTileEvent namespace)
                // Port events have no PC -- they are I/O fabric events.
                if tt.is_mem() {
                    tile.notify_mem_trace_event(hw_id, cycle, None);
                } else {
                    tile.notify_core_trace_event(hw_id, cycle, None);
                }
            }
            for (idx, event_port, active, stalled, tlast) in prev_updates {
                self.device.array.tiles[idx].prev_port_state[event_port as usize] = (active, stalled, tlast);
            }
        }

        if dma_active || streams_moved {
            any_running = true;
        }

        // Phase 4: Detect memory bank conflicts (core vs DMA).
        //
        // When the core and DMA access the same memory bank in the same
        // cycle, HW fires two classes of trace event:
        //   * CONFLICT_DM_BANK_N on the memory module trace unit (per-bank,
        //     mem event IDs 77-84 for compute tiles / 112-119 for memtiles).
        //   * MEMORY_STALL on the core module trace unit (core event 23),
        //     since the core's load/store bundle is the agent that stalls.
        //
        // Both events target the same cycle, so we emit them together and
        // bump the core's `memory_stalls` cycle-stat counter. This mirrors
        // how `LOCK_STALL` is emitted on the core side for lock conflicts.
        {
            let cycle = self.total_cycles;
            let mut bank_events: Vec<(usize, u8)> = Vec::new();
            // Cores whose bundle stalled this cycle due to a DMA-bank conflict.
            let mut stalled_cores: Vec<usize> = Vec::new();
            for col in 0..self.cols {
                for row in self.compute_row_start..self.rows {
                    let idx = col * self.rows + row;
                    if !self.cores[idx].enabled {
                        continue;
                    }
                    let core_banks = self.cores[idx].context.cycle_core_banks;
                    let tile_idx = self.device.array.tile_index(col as u8, row as u8);
                    let dma_banks = self.device.array.tiles[tile_idx].cycle_dma_banks;

                    let conflicts = core_banks & dma_banks;
                    if conflicts != 0 {
                        let num_banks = self.device.array.tiles[tile_idx].num_banks();
                        for bank in 0..num_banks as u8 {
                            if conflicts & (1 << bank) != 0 {
                                bank_events.push((tile_idx, bank));
                            }
                        }
                        stalled_cores.push(idx);
                    }

                    // Reset core bank tracking for next cycle
                    self.cores[idx].context.reset_bank_tracking();
                }
            }
            for (tile_idx, bank) in bank_events {
                let tile = &mut self.device.array.tiles[tile_idx];
                let hw_id = if tile.is_mem() {
                    crate::trace::memtile_conflict_dm_bank_hw_id(bank)
                } else {
                    crate::trace::mem_conflict_dm_bank_hw_id(bank)
                };
                // Memory-module event; no program counter.
                tile.notify_mem_trace_event(hw_id, cycle, None);
            }
            // Emit MEMORY_STALL (core event 23) for each core that lost a
            // bank to the DMA this cycle. Bump the core's stall counter too.
            for core_idx in stalled_cores {
                // Snapshot the core's PC before any &mut borrow so the
                // trace-unit notify and the timing-context record can both
                // stamp the stalled instruction's PC (matches HW's
                // per-instruction-PC view of MEMORY_STALL windows).
                let pc = self.cores[core_idx].context.pc();
                // Map core index -> tile coordinates for the trace unit.
                let col = core_idx / self.rows;
                let row = core_idx % self.rows;
                if let Some(tile) = self.device.array.get_mut(col as u8, row as u8) {
                    let hw_id = crate::trace::core_event_to_hw_id(
                        &crate::interpreter::state::EventType::MemoryStall { cycles: 1, pc: Some(pc) },
                    );
                    if let Some(id) = hw_id {
                        tile.notify_core_trace_event(id, cycle, Some(pc));
                    }
                }
                let ctx = &mut self.cores[core_idx].context;
                ctx.timing_context_mut().memory_stalls += 1;
                ctx.timing_context_mut().record_event(
                    cycle,
                    crate::interpreter::state::EventType::MemoryStall { cycles: 1, pc: Some(pc) },
                );
            }
        }

        // Phase 3c: Fire TRUE (event code 1) on every configured trace unit.
        //
        // Hardware fires TRUE unconditionally every cycle. When a trace
        // unit has TRUE in one of its 8 event slots, it records it --
        // otherwise notify_event() returns immediately (no slot match).
        // This enables the "metronome" alignment strategy for multi-run
        // trace sweeps.
        {
            let cycle = self.total_cycles;
            const TRUE_EVENT: u8 = 1;
            for tile in &mut self.device.array.tiles {
                if tile.core_trace.is_configured() {
                    // Pass the core's current PC so mode-2's Start frame
                    // captures the real anchor PC instead of defaulting to
                    // 0. Mode 0/1 ignore the PC arg here (mode 0 has no
                    // PC field, mode 1 only records PCs on slot fires).
                    let pc = Some(tile.core.pc);
                    tile.core_trace.notify_event(TRUE_EVENT, cycle, pc);
                }
                if tile.mem_trace.is_configured() {
                    tile.mem_trace.notify_event(TRUE_EVENT, cycle, None);
                }
            }
        }

        // Phase 3d: Evaluate edge detectors.
        //
        // After all raw events have been generated (DMA, port, core, bank
        // conflict, TRUE), check edge detectors for rising/falling
        // transitions and fire EDGE_DETECTION_EVENT_0/1 as needed.
        {
            let cycle = self.total_cycles;
            for tile in &mut self.device.array.tiles {
                tile.evaluate_edge_detectors(cycle);
            }
        }

        // Phase 3e: Tick tile timers and performance counters; route firings.
        //
        // Each tile has two 64-bit timers (core module and memory module)
        // that increment every cycle.
        //
        // Core module perf counters are dispatched to tick_active_cycles or
        // tick_idle_cycles based on whether the core executed an instruction
        // (StepResult::Continue) this cycle. Counters configured with
        // ACTIVE_CORE (0x1C) as start_event are level-gated: they tick only
        // when the core is in Execute state, matching hardware behavior.
        //
        // Memory module perf counters are not gated on core Execute state;
        // they always use tick_active_cycles (existing semantics preserved).
        //
        // Tiles are indexed identically to self.cores (col * rows + row), so
        // cores[i] is the CoreState for tiles[i].
        //
        // Per aie-rt xaie_events_aieml.h, PERF_CNT_N hw event id is 5+N in
        // every module type (core 5..8, memmod 5..6, memtile 5..8, shim/PL
        // 5..6). Each counter that crossed its threshold this cycle is fed
        // back through handle_event (so self-resetting configs can recycle)
        // and forwarded to the owning module's trace unit.
        //
        // ORDERING: this phase runs *before* Phase 3f (commit_cycle) so that
        // perfcnt notify_event() calls accumulate into the same cycle's
        // pending_slot_mask as TRUE/edge-detector events. AM020 specifies
        // one frame per cycle: when multiple events fire in cycle N, they
        // must be coalesced into a single Multiple frame, not split across
        // two frames.
        {
            // PERF_CNT_0 has the same numeric ID (5) across every module type
            // per aie-rt xaie_events_aieml.h, so core_events::PERF_CNT_0 is
            // safe to use as the base for both core and memmod perf counters.
            use xdna_archspec::aie2::trace_events::core_events::PERF_CNT_0 as PERF_CNT_BASE;
            // TRUE (event 1) and ACTIVE_CORE (event 0x1C) are level-asserted
            // start events on real hardware: TRUE is always asserted, and
            // ACTIVE_CORE is asserted while the core is in Execute state.
            // A counter configured with either as its start event needs to
            // see the corresponding handle_event() each cycle to transition
            // Idle->Active before the tick logic runs. Without this, the
            // perfcnt anchor scheme emitted by mlir-trace-inject (start =
            // ACTIVE_CORE, reset = PERF_CNT_0, threshold = period) would
            // never fire because the counter stays stuck in Idle (#354).
            const TRUE_EVENT: u8 = 1;
            const ACTIVE_CORE_EVENT: u8 = 0x1C;
            let cycle = self.total_cycles;
            for (i, tile) in self.device.array.tiles.iter_mut().enumerate() {
                tile.core_timer.tick();
                tile.mem_timer.tick();

                let core_active = self.cores.get(i).map_or(false, |c| c.active_this_cycle);

                // Re-assert TRUE every cycle and ACTIVE_CORE only when the
                // core is executing. handle_event() is idempotent: an
                // already-Active counter stays Active; counters whose start
                // event doesn't match are unaffected. Stop and reset events
                // still take priority inside handle_event() per its existing
                // ordering.
                //
                // ACTIVE_CORE is an edge-style start event: it pulses on
                // transition to Execute and arms the counter. Once Active,
                // the counter ticks every cycle regardless of whether the
                // core is still in Execute -- HW perf counters are duration
                // counters started by ACTIVE_CORE and stopped by
                // DISABLED_CORE, not gated per cycle on Execute state.
                tile.core_perf_counters.handle_event(TRUE_EVENT);
                if core_active {
                    tile.core_perf_counters.handle_event(ACTIVE_CORE_EVENT);
                }
                tile.mem_perf_counters.handle_event(TRUE_EVENT);

                let core_fired = tile.core_perf_counters.tick();
                if !core_fired.is_empty() {
                    // Snapshot the core's pipeline-adjusted PC for trace
                    // stamping. tile.core.pc is updated each cycle from the
                    // core context (Phase 2) and includes the stall-PC
                    // pipeline adjustment, so it matches what HW's trace
                    // controller would sample when the perf-counter
                    // threshold fires.
                    let core_pc = tile.core.pc;
                    for cnt_idx in core_fired {
                        let hw_id = PERF_CNT_BASE + cnt_idx as u8;
                        // Feed back so self-reset configs work, then trace-notify.
                        tile.core_perf_counters.handle_event(hw_id);
                        tile.notify_core_trace_event(hw_id, cycle, Some(core_pc));
                    }
                }

                // Memory module perf counters tick unconditionally.
                let mem_fired = tile.mem_perf_counters.tick();
                for cnt_idx in mem_fired {
                    let hw_id = PERF_CNT_BASE + cnt_idx as u8;
                    tile.mem_perf_counters.handle_event(hw_id);
                    tile.notify_mem_trace_event(hw_id, cycle, None);
                }
            }
        }

        // Phase 3f: Commit per-cycle trace frames.
        //
        // Per AM020 event-time mode, a trace unit creates at most one frame
        // per cycle. `notify_event` accumulates slot activity into a bitmask;
        // committing here emits one Single/Multiple frame for this cycle.
        // Without this step the mask only commits lazily on the next cycle's
        // event, which can leave the last cycle's frame uncommitted past the
        // simulation's natural end and inflate routing pressure for the
        // uncommitted bytes when they eventually land at flush time.
        //
        // This phase MUST run after all event-emitting phases (3a-3e) so
        // every event for `cycle` is already in pending_slot_mask. Otherwise
        // events that fire after the commit get carried into the next
        // cycle's frame, splitting a HW Multiple frame into two.
        //
        // Mode-2 atoms are NOT emitted here. Per AM020 ch.2 mode 2 records
        // only branches and ZOL LC, not per-cycle execution status.
        // notify_atom / notify_branch_taken fire from the executor at
        // branch resolution; this phase just drains them via commit_cycle.
        {
            let cycle = self.total_cycles;
            for tile in &mut self.device.array.tiles {
                tile.core_trace.commit_cycle(cycle);
                tile.mem_trace.commit_cycle(cycle);
            }
        }

        // Phase 4: Update tile DMA channel state from engine state
        self.sync_dma_completion();

        self.total_cycles += 1;

        // Determine if we should halt the engine.
        //
        // The engine halts when all work is done:
        // 1. All cores have halted (or no cores were enabled), AND
        // 2. Either:
        //    a. No DMA activity at all, OR
        //    b. DMA is active but no progress for multiple cycles (deadlock)
        //
        // For DMA-only tests (no cores enabled), NPU instructions from the
        // host command stream trigger DMA transfers. The engine must keep
        // stepping so DMA engines and stream switches advance. The run loop
        // calls force_running() while NPU instructions are still pending;
        // once DMA is triggered, the deadlock detection here handles
        // completion and stall detection normally.
        let any_cores_enabled = self.cores.iter().any(|c| c.enabled);

        // All compute work is done when cores have halted (or none exist).
        let cores_done = if any_cores_enabled {
            all_halted
        } else {
            !any_running
        };

        if cores_done && !dma_active {
            self.status = EngineStatus::Halted;
            return;
        }

        // -- Monotonic progress detection --
        //
        // Three signals count as forward progress, any one of which resets
        // the stall counter:
        //
        // 1. DMA bytes transferred: a DMA channel moved data.
        // 2. Lock releases: an arbiter released a lock to a waiter.
        // 3. Core instructions executed: a core advanced its PC at least
        //    once outside of WaitLock/WaitDma/WaitStream.
        //
        // The third signal exists to avoid false stalls during heavy scalar
        // computation (e.g. byte-by-byte memcpy in compute kernels) that can
        // run for tens of thousands of cycles between any DMA or lock event.
        // True deadlocks are still caught: a deadlocked core is in WaitLock
        // (etc.) and does not advance total_instructions.
        if self.stall_threshold > 0 {
            let dma_bytes = self.device.array.total_dma_bytes_transferred();
            let lock_releases = self.device.array.total_lock_releases();
            let instructions = self.total_instructions;

            if dma_bytes > self.last_dma_bytes
                || lock_releases > self.last_lock_releases
                || instructions > self.last_instructions
            {
                self.stall_cycles = 0;
                self.last_dma_bytes = dma_bytes;
                self.last_lock_releases = lock_releases;
                self.last_instructions = instructions;
            } else {
                self.stall_cycles += 1;
                if self.stall_cycles >= self.stall_threshold {
                    log::warn!(
                        "Stall detected: no progress for {} cycles (dma_bytes={}, lock_releases={}, instructions={})",
                        self.stall_cycles, dma_bytes, lock_releases, instructions,
                    );
                    self.status = EngineStatus::Stalled;
                }
            }
        } else if cores_done && dma_active {
            // Stall detection disabled -- fall back to simple deadlock check.
            // Only runs while cores are done but DMA is still active.
            let dma_bytes = self.device.array.total_dma_bytes_transferred();
            if dma_bytes > self.last_dma_bytes {
                self.stall_cycles = 0;
                self.last_dma_bytes = dma_bytes;
            } else {
                self.stall_cycles += 1;
                if self.stall_cycles >= 50 {
                    log::info!(
                        "Engine halting: all cores done, DMA stalled for {} cycles",
                        self.stall_cycles,
                    );
                    self.status = EngineStatus::Halted;
                }
            }
        } else {
            // Cores still running, stall detection disabled -- reset counter
            // so it doesn't carry over stale counts into the post-halt check.
            self.stall_cycles = 0;
        }
    }

    /// Sync DMA start requests from tile.dma_channels to DmaEngine.
    ///
    /// When a core executes DmaStart, it sets tile.dma_channels[ch].running = true
    /// and tile.dma_channels[ch].start_queue = bd_index. This method reads those
    /// values and calls DmaEngine.start_channel().
    fn sync_dma_start_requests(&mut self) {
        for col in 0..self.cols as u8 {
            for row in 0..self.rows as u8 {
                // Get tile and DMA engine together
                if let Some((tile, engine)) = self.device.array.tile_and_dma(col, row) {
                    for ch in 0..tile.dma_channels.len() {
                        let channel = &mut tile.dma_channels[ch];

                        // Check for pending start request:
                        // - running is true (set by DmaStart instruction)
                        // - engine channel is not already active
                        if channel.running && !engine.channel_active(ch as u8) {
                            let bd_index = channel.start_queue as u8;

                            // Start the channel on the engine
                            if engine.start_channel(ch as u8, bd_index).is_ok() {
                                // Clear start_queue to indicate we processed it
                                channel.start_queue = 0xFF;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Sync DMA completion state from DmaEngine to tile.dma_channels.
    ///
    /// When DmaEngine reports a channel as complete (not active), we clear
    /// tile.dma_channels[ch].running so that DmaWait instructions can proceed.
    fn sync_dma_completion(&mut self) {
        for col in 0..self.cols as u8 {
            for row in 0..self.rows as u8 {
                if let Some((tile, engine)) = self.device.array.tile_and_dma(col, row) {
                    for ch in 0..tile.dma_channels.len() {
                        let channel = &mut tile.dma_channels[ch];

                        // If engine reports channel complete, clear running flag
                        if channel.running {
                            let state = engine.channel_state(ch as u8);
                            if matches!(state, ChannelState::Idle) {
                                channel.running = false;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Run for up to `max_cycles` cycles.
    ///
    /// Stops early if all cores halt or an error occurs.
    /// Returns the number of cycles actually executed.
    pub fn run(&mut self, max_cycles: u64) -> u64 {
        let start = self.total_cycles;

        for _ in 0..max_cycles {
            self.step();

            if matches!(self.status, EngineStatus::Halted | EngineStatus::Stalled | EngineStatus::Error) {
                break;
            }
        }

        self.total_cycles - start
    }

    /// Pause execution.
    pub fn pause(&mut self) {
        if self.status == EngineStatus::Running {
            self.status = EngineStatus::Paused;
        }
    }

    /// Resume execution.
    pub fn resume(&mut self) {
        if self.status == EngineStatus::Paused {
            self.status = EngineStatus::Ready;
        }
    }

    /// Reset all cores.
    pub fn reset(&mut self) {
        let rows = self.rows;
        for (idx, core) in self.cores.iter_mut().enumerate() {
            let col = (idx / rows) as u8;
            let row = (idx % rows) as u8;
            core.interpreter.reset();
            // ExecutionContext::reset() resets via Self::new() which sets
            // CORE_ID to (0,0). Re-tag each core's identity so reads of
            // CORE_ID continue to return this core's actual (col, row)
            // -- kernels routinely branch on it.
            core.context = ExecutionContext::new_for_tile(col, row);
            core.enabled = false;
            core.trace_events_consumed = 0;
            core.active_this_cycle = false;
        }

        self.status = EngineStatus::Ready;
        self.total_cycles = 0;
        self.total_instructions = 0;
        self.last_dma_bytes = 0;
        self.last_lock_releases = 0;
        self.last_instructions = 0;
        self.stall_cycles = 0;
    }

    /// Reset all per-context tile state on hw_context teardown / re-creation.
    ///
    /// Mirrors what real HW does on hw_context destroy: a column reset that
    /// clears locks, DMA channel queues + BD configs, stream switch routing,
    /// and trace state. Host memory (BO contents) is intentionally NOT
    /// touched -- the bridge runner zeroes the trace BO via sync_to_device
    /// before each new run, and input BOs are re-uploaded the same way.
    ///
    /// Called from the FFI on a fresh xrt::hw_context creation so the new
    /// run starts on a clean column. Without this, lock state from a prior
    /// run can leave an acquire stuck (target_value reached but the queued
    /// release never fires for the new BD), and the new run stalls at the
    /// stall threshold without making forward progress. Each reset also
    /// nudges the trace_unit clear via Trace_Control0 rewrites in the new
    /// run's insts.bin, but locks/BDs need this explicit reset because no
    /// existing register write paths reach them.
    pub fn reset_for_new_context(&mut self) {
        self.reset();
        self.device.array.reset();
    }

    /// Sync core enabled state and PC from device tiles.
    ///
    /// After applying CDO, the tile.core.enabled and tile.core.pc fields
    /// are set by register writes. This method syncs those values to the
    /// engine's internal core state, enabling execution.
    ///
    /// Call this after `device_mut().apply_cdo()` to start execution.
    pub fn sync_cores_from_device(&mut self) {
        for col in 0..self.cols {
            for row in self.compute_row_start..self.rows {
                if let Some(tile) = self.device.array.get(col as u8, row as u8) {
                    if tile.is_compute() {
                        let idx = col * self.rows + row;
                        if let Some(core) = self.cores.get_mut(idx) {
                            // Sync enabled state from tile
                            core.enabled = tile.core.enabled;
                            // Sync PC from tile (if set by CDO)
                            if tile.core.pc != 0 || tile.core.enabled {
                                core.context.set_pc(tile.core.pc);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Get the number of enabled cores.
    pub fn enabled_cores(&self) -> usize {
        self.cores.iter().filter(|c| c.enabled).count()
    }

    /// Get the number of running (non-halted) cores.
    pub fn active_cores(&self) -> usize {
        self.cores.iter().filter(|c| c.enabled && !c.interpreter.is_halted()).count()
    }

    /// Check if all enabled cores are blocked (stalled on lock/DMA or halted).
    ///
    /// Returns true when no enabled core can make forward progress. This is
    /// used by the NPU executor warm-up phase: on real hardware, cores run
    /// for thousands of cycles before NPU instructions arrive through the
    /// NoC, so they always reach their first blocking point (typically a lock
    /// acquire) before any host-issued writes modify tile memory.
    pub fn all_cores_blocked(&self) -> bool {
        let enabled: Vec<_> = self.cores.iter().filter(|c| c.enabled).collect();
        !enabled.is_empty() && enabled.iter().all(|c| c.interpreter.is_stalled() || c.interpreter.is_halted())
    }

    /// Set auto-run mode.
    pub fn set_auto_run(&mut self, auto: bool) {
        self.auto_run = auto;
    }

    /// Check if auto-run is enabled.
    pub fn is_auto_run(&self) -> bool {
        self.auto_run
    }

    /// Get status as string for display.
    pub fn status_string(&self) -> &'static str {
        match self.status {
            EngineStatus::Ready => "Ready",
            EngineStatus::Running => "Running",
            EngineStatus::Paused => "Paused",
            EngineStatus::Halted => "Halted",
            EngineStatus::Stalled => "Stalled",
            EngineStatus::Error => "Error",
        }
    }

    // Private helpers

    fn core_index(&self, col: usize, row: usize) -> Option<usize> {
        if col < self.cols && row < self.rows {
            Some(col * self.rows + row)
        } else {
            None
        }
    }

    fn get_core(&self, col: usize, row: usize) -> Option<&CoreState> {
        self.core_index(col, row).map(|idx| &self.cores[idx])
    }

    fn get_core_mut(&mut self, col: usize, row: usize) -> Option<&mut CoreState> {
        self.core_index(col, row).map(|idx| &mut self.cores[idx])
    }

    /// Single dispatch point for a drained `CtrlPacketAction`. Owns the
    /// SLVERR decode-gate: a control-packet access whose offset does not
    /// decode is a faithful AXI slave error -- latch bit 2, suppress the
    /// access, and continue (poll-only sticky; never engine-fatal). The
    /// structural-rejection `Error` arm is unchanged (out of scope).
    fn dispatch_ctrl_action(&mut self, action: crate::device::tile::CtrlPacketAction) {
        use crate::device::tile::CtrlPacketAction;
        match action {
            CtrlPacketAction::WriteRegister { col, row, offset, value } => {
                if self.device.ctrl_pkt_offset_decodes(row, offset) {
                    self.device.write_tile_register(col, row, offset, value);
                } else {
                    log::error!(
                        "Tile ({},{}) ctrl_pkt SLVERR: write to undecoded offset \
                         0x{:05X} (sets Control_Packet_Handler_Status bit 0x4)",
                        col,
                        row,
                        offset
                    );
                    self.device.latch_ctrl_slverr(col, row);
                }
            }
            CtrlPacketAction::ReadRegisters { col, row, offset, count, response_id } => {
                if self.device.ctrl_pkt_read_range_decodes(row, offset, count) {
                    self.device.array.handle_read_registers(col, row, offset, count, response_id);
                } else {
                    log::error!(
                        "Tile ({},{}) ctrl_pkt SLVERR: read of {} regs from undecoded \
                         offset 0x{:05X} (sets Control_Packet_Handler_Status bit 0x4)",
                        col,
                        row,
                        count,
                        offset
                    );
                    self.device.latch_ctrl_slverr(col, row);
                }
            }
            CtrlPacketAction::Error(msg) => {
                self.device.array.fatal_errors.push(msg);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_npu1() {
        let engine = InterpreterEngine::new_npu1();

        assert_eq!(engine.status(), EngineStatus::Ready);
        assert_eq!(engine.total_cycles(), 0);
        assert_eq!(engine.enabled_cores(), 0);
    }

    #[test]
    fn test_enable_disable_cores() {
        let mut engine = InterpreterEngine::new_npu1();

        // Enable core at (0, 2) - first compute tile
        engine.enable_core(0, 2);
        assert!(engine.is_core_enabled(0, 2));
        assert_eq!(engine.enabled_cores(), 1);

        // Disable it
        engine.disable_core(0, 2);
        assert!(!engine.is_core_enabled(0, 2));
        assert_eq!(engine.enabled_cores(), 0);
    }

    #[test]
    fn test_set_core_pc() {
        let mut engine = InterpreterEngine::new_npu1();

        engine.set_core_pc(0, 2, 0x1000);
        let ctx = engine.core_context(0, 2).unwrap();
        assert_eq!(ctx.pc(), 0x1000);
    }

    #[test]
    fn test_step_no_enabled_cores() {
        let mut engine = InterpreterEngine::new_npu1();

        engine.step();

        // Should halt since no cores enabled
        assert_eq!(engine.status(), EngineStatus::Halted);
        assert_eq!(engine.total_cycles(), 1);
    }

    #[test]
    fn test_step_with_enabled_core() {
        let mut engine = InterpreterEngine::new_npu1();

        // Enable a core and write NOP to program memory
        engine.enable_core(0, 2);

        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00, 0x00, 0x00, 0x00]);
        }

        engine.step();

        assert_eq!(engine.status(), EngineStatus::Running);
        assert_eq!(engine.total_cycles(), 1);

        // Check that PC advanced
        let ctx = engine.core_context(0, 2).unwrap();
        assert_eq!(ctx.pc(), 4);
    }

    #[test]
    fn test_run_limited_cycles() {
        let mut engine = InterpreterEngine::new_npu1();

        engine.enable_core(0, 2);

        // Write many NOPs
        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00u8; 1024]);
        }

        let cycles = engine.run(10);

        assert_eq!(cycles, 10);
        assert_eq!(engine.total_cycles(), 10);
    }

    #[test]
    fn test_pause_resume() {
        let mut engine = InterpreterEngine::new_npu1();

        engine.enable_core(0, 2);
        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00u8; 1024]);
        }

        engine.step();
        assert_eq!(engine.status(), EngineStatus::Running);

        engine.pause();
        assert_eq!(engine.status(), EngineStatus::Paused);

        engine.resume();
        assert_eq!(engine.status(), EngineStatus::Ready);
    }

    #[test]
    fn test_reset() {
        let mut engine = InterpreterEngine::new_npu1();

        engine.enable_core(0, 2);
        engine.set_core_pc(0, 2, 0x1000);

        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00u8; 64]);
        }

        engine.run(5);

        engine.reset();

        assert_eq!(engine.status(), EngineStatus::Ready);
        assert_eq!(engine.total_cycles(), 0);
        assert_eq!(engine.enabled_cores(), 0);
    }

    #[test]
    fn test_reset_for_new_context_preserves_core_id() {
        // ExecutionContext::reset() does `*self = Self::new()`, which
        // calls `new_for_tile(0, 0)` and zeros the CORE_ID register for
        // every core regardless of position. Real HW's CORE_ID is wired
        // to the tile's position and never changes -- and AIE kernels
        // routinely branch on it. Forgetting to re-tag the identity
        // after reset turned every post-reset run into a "this is the
        // (0,0) tile" simulation, which suppressed real INSTR_VECTOR /
        // LOCK_STALL events whose firing depended on the kernel's
        // per-tile branch -- the j>=8 sweep caught this when fresh
        // workers (correct CORE_ID) produced different traces than
        // post-reset workers (zeroed CORE_ID).
        use crate::interpreter::state::CORE_ID_REG_INDEX;
        let mut engine = InterpreterEngine::new_npu1();

        let pre = engine
            .core_context(1, 2)
            .expect("core (1,2) exists")
            .scalar
            .read(CORE_ID_REG_INDEX);
        assert_eq!(pre, (1 << 16) | 2, "fresh CORE_ID for (1,2) should be 0x10002");

        engine.reset_for_new_context();

        let post = engine
            .core_context(1, 2)
            .expect("core (1,2) exists")
            .scalar
            .read(CORE_ID_REG_INDEX);
        assert_eq!(post, (1 << 16) | 2, "CORE_ID for (1,2) must survive reset; got 0x{post:08x}");
    }

    #[test]
    fn test_reset_for_new_context_clears_perf_counter_values() {
        // Performance counters (PERF_CNT_N) emit trace events when their
        // running count reaches `event_value`. On real hw_context reset
        // they zero out -- if we leave them at their prior count, the
        // next run hits the threshold N cycles earlier than a fresh run
        // would, emitting phantom PERF_CNT events before the new run's
        // workload has actually generated any counted events. The j1
        // sweep saw this as 3 extra PERF_CNT_2 events in batches that
        // followed a high-activity batch on the same runner session
        // (e.g., j1 batch_01=11 vs j4 batch_01=8 in add_one_using_dma).
        let mut engine = InterpreterEngine::new_npu1();

        let tile = engine.device_mut().tile_mut(0, 2).expect("tile (0,2) exists on NPU1");
        tile.core_perf_counters.write_counter(0, 12345);
        tile.mem_perf_counters.write_counter(1, 67890);

        engine.reset_for_new_context();

        let tile = engine.device_mut().tile_mut(0, 2).expect("tile (0,2) exists on NPU1");
        assert_eq!(
            tile.core_perf_counters.read_counter(0),
            0,
            "core_perf_counters[0] must zero on reset_for_new_context"
        );
        assert_eq!(
            tile.mem_perf_counters.read_counter(1),
            0,
            "mem_perf_counters[1] must zero on reset_for_new_context"
        );
    }

    #[test]
    fn test_reset_for_new_context_clears_pending_mem_trace() {
        // mem_trace_pending buffers memory-module trace events (lock
        // acquire/release, DMA) until the coordinator drains them at
        // each cycle boundary. Events pushed near end-of-run can outlive
        // the final drain; if reset_for_new_context leaves them in place
        // they reappear at cycle 0 of the next hw_context, producing
        // phantom events that bias trace-comparison event counts. This
        // showed up as a 3-event divergence between j=1 and j>1 sweep
        // runs whenever a small batch followed a high-activity batch
        // on the same runner session.
        use crate::interpreter::state::EventType;
        let mut engine = InterpreterEngine::new_npu1();

        let tile = engine.device_mut().tile_mut(0, 2).expect("tile (0,2) exists on NPU1");
        tile.mem_trace_pending.push((1000, EventType::LockAcquire { lock_id: 0 }));
        tile.mem_trace_pending.push((1001, EventType::LockRelease { lock_id: 0 }));

        engine.reset_for_new_context();

        let drained = engine.device_mut().array.drain_mem_trace_events();
        assert!(
            drained.is_empty(),
            "reset_for_new_context must clear pending mem-trace events, but found {drained:?}"
        );
    }

    #[test]
    fn test_multiple_cores() {
        let mut engine = InterpreterEngine::new_npu1();

        // Enable multiple cores
        engine.enable_core(0, 2);
        engine.enable_core(1, 2);
        engine.enable_core(0, 3);

        assert_eq!(engine.enabled_cores(), 3);

        // Write NOPs to all their program memories
        for &(col, row) in &[(0, 2), (1, 2), (0, 3)] {
            if let Some(tile) = engine.device_mut().tile_mut(col, row) {
                tile.write_program(0, &[0x00u8; 64]);
            }
        }

        engine.step();

        assert_eq!(engine.active_cores(), 3);
    }

    #[test]
    fn test_auto_run() {
        let mut engine = InterpreterEngine::new_npu1();

        assert!(!engine.is_auto_run());

        engine.set_auto_run(true);
        assert!(engine.is_auto_run());

        engine.set_auto_run(false);
        assert!(!engine.is_auto_run());
    }

    #[test]
    fn test_status_string() {
        let engine = InterpreterEngine::new_npu1();
        assert_eq!(engine.status_string(), "Ready");
    }

    #[test]
    fn test_dma_integration() {
        use crate::device::dma::BdConfig;

        let mut engine = InterpreterEngine::new_npu1();

        // Configure a BD on tile (1, 2) using MM2S channel (channel 2)
        // Use 64-byte transfer: at 4 bytes/cycle, takes 16 cycles to complete
        // This ensures the transfer is still in progress after the first step
        if let Some(dma) = engine.device_mut().array.dma_engine_mut(1, 2) {
            dma.configure_bd(0, BdConfig::simple_1d(0x100, 64)).unwrap();
        }

        // Simulate DmaStart by setting tile channel state (channel 2 = MM2S)
        if let Some(tile) = engine.device_mut().tile_mut(1, 2) {
            tile.dma_channels[2].running = true;
            tile.dma_channels[2].start_queue = 0; // BD index 0
        }

        // Initial state: DMA should not be active on engine yet
        {
            let dma = engine.device().array.dma_engine(1, 2).unwrap();
            assert!(!dma.channel_active(2));
        }

        // Step once - should sync start request and begin transfer
        engine.step();

        // DMA engine should now be active (transfer in progress)
        {
            let dma = engine.device().array.dma_engine(1, 2).unwrap();
            assert!(dma.channel_active(2), "DMA should be active after first step");
        }

        // Step more - 64 bytes at 4 bytes/cycle = 16 cycles
        // (Currently no setup overhead since timing isn't integrated yet).
        // Drain MM2S output each step; without it the 4-deep slave-port
        // FIFO fills after a few cycles and the DMA stalls indefinitely
        // because no consumer is wired up in this isolated test.
        for i in 0..50 {
            engine.step();
            {
                let dma = engine.device_mut().array.dma_engine_mut(1, 2).unwrap();
                while dma.pop_stream_out().is_some() {}
            }

            let dma = engine.device().array.dma_engine(1, 2).unwrap();
            let state = dma.channel_state(2);

            // Break when complete
            if matches!(state, ChannelState::Idle) {
                break;
            }

            // Safety check - shouldn't take more than ~20 cycles
            if i > 30 {
                let stats = dma.channel_stats(2).unwrap();
                panic!(
                    "DMA taking too long. State: {:?}, bytes transferred: {}, active: {}",
                    state,
                    stats.bytes_transferred,
                    dma.channel_active(2)
                );
            }
        }

        // Tile channel should show complete
        {
            let tile = engine.device().array.tile(1, 2);
            assert!(!tile.dma_channels[2].running, "Tile channel should show complete");
        }
    }

    #[test]
    fn test_host_memory_access() {
        let mut engine = InterpreterEngine::new_npu1();

        // Write some data to host memory
        engine.host_memory_mut().write_bytes(0x1000, &[1, 2, 3, 4]);

        // Read it back
        let mut buf = [0u8; 4];
        engine.host_memory().read_bytes(0x1000, &mut buf);
        assert_eq!(buf, [1, 2, 3, 4]);
    }

    // --- Cycle-accurate mode tests ---

    #[test]
    fn test_engine_is_always_cycle_accurate() {
        let engine = InterpreterEngine::new_npu1();

        assert!(engine.is_cycle_accurate());
        assert_eq!(engine.status(), EngineStatus::Ready);
        assert_eq!(engine.total_cycles(), 0);
    }

    #[test]
    fn test_all_execution_is_cycle_accurate() {
        let engine1 = InterpreterEngine::new_npu1();
        assert!(engine1.is_cycle_accurate());

        let engine2 = InterpreterEngine::new_npu2();
        assert!(engine2.is_cycle_accurate());
    }

    #[test]
    fn test_timing_context_enabled() {
        let engine = InterpreterEngine::new_npu1();

        // Core contexts should have timing enabled
        let ctx = engine.core_context(0, 2).unwrap();
        assert!(ctx.has_timing(), "Cores should have timing context");
    }

    #[test]
    fn test_cycle_accurate_step() {
        let mut engine = InterpreterEngine::new_npu1();

        // Enable a core and write NOP to program memory
        engine.enable_core(0, 2);

        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00, 0x00, 0x00, 0x00]);
        }

        engine.step();

        assert_eq!(engine.status(), EngineStatus::Running);
        assert_eq!(engine.total_cycles(), 1);

        // Check that PC advanced
        let ctx = engine.core_context(0, 2).unwrap();
        assert_eq!(ctx.pc(), 4);
        assert_eq!(ctx.cycles, 1);
    }

    #[test]
    fn test_npu2_is_cycle_accurate() {
        let engine = InterpreterEngine::new_npu2();
        assert!(engine.is_cycle_accurate());
    }

    // --- Perf counter → trace unit wiring test (Task 1, A.2) ---

    /// Verifies that threshold firings from PerfCounterBank are routed to the
    /// tile's trace unit.
    ///
    /// Before Task 1, the coordinator discards the Vec<usize> returned by
    /// tick_*_cycles(). This test catches that gap: it configures a perf
    /// counter to fire at threshold=5, maps PERF_CNT_0 (hw_id=5) to trace
    /// slot 0, and asserts the trace unit has recorded data after running.
    ///
    /// The test should FAIL before the wiring is in place (Step 1.3).
    #[test]
    fn test_perfcnt_threshold_routes_to_trace_unit() {
        let mut engine = InterpreterEngine::new_npu1();

        // Enable core (0,2) with enough NOPs to run 20+ cycles without halting.
        // NOPs advance PC by 4 bytes; 512 bytes -> 128 cycles of runtime.
        engine.enable_core(0, 2);
        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00u8; 512]);
        }

        // Configure perf counter 0 on tile (0,2) core module:
        //   start_event = TRUE (id=1), stop_event = NONE (id=0), threshold = 5.
        // write_control_start_stop(value, counter_lo, counter_hi, event_width)
        // bits [6:0]=start0, [14:8]=stop0
        //
        // No manual handle_event(1) here: the coordinator's Phase 3e fires
        // TRUE every cycle into perf counter banks, which is what arms a
        // TRUE-started counter from Idle->Active. Verifying the fix for
        // #354 means relying on that wiring rather than poking it manually.
        {
            let tile = engine.device_mut().array.tile_mut(0, 2);
            let ctrl0 = 1u32 | (0u32 << 8); // start=TRUE(1), stop=NONE(0)
            tile.core_perf_counters.write_control_start_stop(ctrl0, 0, 1, 7);
            tile.core_perf_counters.write_event_value(0, 5);
        }

        // Configure trace unit on tile (0,2) core module:
        //   mode = EventTime (0), start_event = TRUE (1), stop_event = NONE (0)
        //   slot 0 = PERF_CNT_0 (hw_id = 5)
        {
            let tile = engine.device_mut().array.tile_mut(0, 2);
            // Trace_Control0: [31:24]=stop=0, [23:16]=start=1, [1:0]=mode=0
            let ctrl0 = (0u32 << 24) | (1u32 << 16) | 0u32;
            tile.core_trace.write_register(0x00, ctrl0);
            // Trace_Event0: slot0 = PERF_CNT_0 = 5
            tile.core_trace.write_register(0x10, 5u32);
        }

        // The coordinator fires TRUE(1) on the trace unit in Phase 3c, which
        // will transition it from Idle to Running on the first step.

        // Run 20 cycles: threshold=5 fires at cycle 5 (counter started at 0),
        // and the wiring should route PERF_CNT_0(5) into the trace unit.
        let _cycles = engine.run(20);

        // After Task 1, the trace unit byte_buffer must be non-empty: at least
        // one frame was encoded for the cycle-5 threshold event.
        // After Task 1 wiring, the trace must hold MORE than just the 8-byte
        // start marker (0xF0 + 7 timer bytes that encode_start() pushes).
        // A threshold event at cycle 5 adds a Single0 frame (1 byte), so
        // wired correctly the byte count will be > 8.
        //
        // Without the wiring (coordinator discards Vec return), only the
        // start marker is in the buffer: exactly 8 bytes, never a slot frame.
        let tile = engine.device().array.tile(0, 2);
        let encoded = tile.core_trace.encoded_bytes_len();
        assert!(
            encoded > 8,
            "trace has only {} encoded bytes (just the start marker): \
             perfcnt threshold was not routed to trace unit",
            encoded
        );
    }

    /// Regression for #354: a perf counter configured with start_event =
    /// ACTIVE_CORE (the perfcnt anchor scheme that mlir-trace-inject emits)
    /// must transition Idle->Active and reach its threshold without any
    /// manual handle_event() poke. Before the Phase 3e wiring fix, the
    /// counter sat in Idle forever and never fired, which is why HW
    /// captured PERF_CNT_0 anchor pulses but EMU captured zero.
    ///
    /// Setup matches the mlir-trace-inject perfcnt config block on a
    /// shorter threshold:
    ///   Performance_Control0 = 0x1C  -> Cnt0_Start_Event = ACTIVE_CORE
    ///   Performance_Control2 = 0x05  -> Cnt0_Reset_Event = PERF_CNT_0 (self-reset)
    ///   Performance_Counter0_Event_Value = 5 (vs production 1024)
    /// Then run 12 cycles and assert the trace unit recorded at least
    /// two PERF_CNT_0 firings (the self-reset cycle gives 5-cycle period:
    /// fires at cycle 5 and cycle 11 -- exact count depends on stop ordering).
    #[test]
    fn test_perfcnt_active_core_start_fires_without_manual_arm() {
        let mut engine = InterpreterEngine::new_npu1();

        engine.enable_core(0, 2);
        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00u8; 512]);
        }

        // Perf counter 0: start=ACTIVE_CORE(0x1C), reset=PERF_CNT_0(5), threshold=5.
        // No manual handle_event() call -- the coordinator's Phase 3e wiring
        // is what arms the counter. This is the bug reproducer for #354.
        {
            let tile = engine.device_mut().array.tile_mut(0, 2);
            let ctrl0 = 0x1Cu32 | (0u32 << 8); // start=ACTIVE_CORE, stop=NONE
            tile.core_perf_counters.write_control_start_stop(ctrl0, 0, 1, 7);
            tile.core_perf_counters.write_event_value(0, 5);
            // reset_event[0] = PERF_CNT_0 (5)
            tile.core_perf_counters.write_control_reset(5u32, 7);
        }

        // Trace unit on tile (0,2) core: mode=0, start=TRUE(1), slot 0 = PERF_CNT_0(5).
        {
            let tile = engine.device_mut().array.tile_mut(0, 2);
            let ctrl0 = (0u32 << 24) | (1u32 << 16) | 0u32;
            tile.core_trace.write_register(0x00, ctrl0);
            tile.core_trace.write_register(0x10, 5u32);
        }

        // Run 12 cycles. With period=5 and self-reset, the counter should
        // fire PERF_CNT_0 at cycle 5 and cycle 10 (5-cycle interval after
        // reset).
        let _cycles = engine.run(12);

        // The trace unit should have encoded both PERF_CNT_0 firings as
        // Single0 frames (1 byte each) on top of the 8-byte Start marker.
        // Pre-fix this would be exactly 8 bytes (no firings ever land).
        let tile = engine.device().array.tile(0, 2);
        let encoded = tile.core_trace.encoded_bytes_len();
        assert!(
            encoded >= 10,
            "expected >= 10 encoded bytes (start marker + 2x Single0 PERF_CNT_0 \
             frames); got {} bytes -- ACTIVE_CORE-started counter never fired \
             (#354 regression)",
            encoded
        );
    }

    /// Verifies AM020 "one frame per cycle" invariant when a perf counter
    /// fires in the same cycle as another slot event (here, TRUE).
    ///
    /// HW emits a single Multiple frame whose mask carries both slot bits.
    /// The emulator must do the same; if Phase 3e (perfcnt route) ran AFTER
    /// Phase 3f (commit_cycle), the perfcnt notify would land in the next
    /// cycle's pending_slot_mask, splitting one HW frame into two emulator
    /// frames.
    ///
    /// Setup: counter 0 fires at cycle 4 (threshold=5, 5 ticks across cycles
    /// 0..4). Trace slots: slot 0 = PERF_CNT_0 (hw_id=5), slot 1 = TRUE
    /// (hw_id=1). After 6 steps the byte buffer contains an 8-byte Start
    /// marker plus a stream of Single0 TRUE frames, with one Multiple0
    /// frame at cycle 4 carrying mask=0b11.
    ///
    /// Multiple0 byte format (from encode_multiple):
    ///   byte0 = 0xC0 | (mask >> 4)
    ///   byte1 = ((mask & 0x0F) << 4) | (delta & 0x0F)
    /// For mask=0b00000011: byte0=0xC0, byte1=0x30 | delta.
    #[test]
    fn test_perfcnt_and_true_same_cycle_coalesce_into_multiple_frame() {
        let mut engine = InterpreterEngine::new_npu1();

        engine.enable_core(0, 2);
        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00u8; 256]);
        }

        // Perf counter 0: start=TRUE(1), threshold=5. Fires at cycle 4.
        {
            let tile = engine.device_mut().array.tile_mut(0, 2);
            let ctrl0 = 1u32 | (0u32 << 8);
            tile.core_perf_counters.write_control_start_stop(ctrl0, 0, 1, 7);
            tile.core_perf_counters.write_event_value(0, 5);
            tile.core_perf_counters.handle_event(1); // arm
        }

        // Trace: start=TRUE(1), slot 0 = PERF_CNT_0 (5), slot 1 = TRUE (1).
        {
            let tile = engine.device_mut().array.tile_mut(0, 2);
            let ctrl0 = (0u32 << 24) | (1u32 << 16) | 0u32;
            tile.core_trace.write_register(0x00, ctrl0);
            // Trace_Event0: slot0=5 (PERF_CNT_0), slot1=1 (TRUE), slot2=0, slot3=0
            let evt0 = 5u32 | (1u32 << 8);
            tile.core_trace.write_register(0x10, evt0);
        }

        // 6 steps covers: trace start at cycle 0, TRUE frames at 1..3, the
        // coalesced frame at cycle 4, plus one more cycle so cycle 4's frame
        // is committed (commit_cycle in same step runs after notify, so the
        // cycle-4 frame is committed at end of cycle 4 itself).
        let _ = engine.run(6);

        let bytes = engine.device().array.tile(0, 2).core_trace.encoded_bytes().to_vec();

        // Skip the 8-byte Start marker (byte 0 = 0xF0 + 7 timer bytes).
        assert_eq!(
            bytes.first().copied(),
            Some(0xF0),
            "expected Start marker as first byte; got buffer {:02X?}",
            bytes
        );
        assert!(bytes.len() > 8, "buffer must hold more than just the Start marker; got {:02X?}", bytes);

        // Walk the post-marker bytes and locate any Multiple0 frame.
        // Multiple0 has top 4 bits = 0b1100, so byte0 & 0xF0 == 0xC0.
        // Single0 has top bit = 0 (byte < 0x80), Single1 = 0x80..0x9F,
        // Single2 = 0xA0..0xBF.
        //
        // We expect exactly one Multiple0 frame (at cycle 4) with mask=0b11.
        let mut multiple_count = 0usize;
        let mut multiple_with_mask_3 = 0usize;
        let mut i = 8; // past Start marker
        while i < bytes.len() {
            let b = bytes[i];
            if b & 0xF0 == 0xC0 {
                // Multiple0: 2 bytes
                multiple_count += 1;
                let mask_hi = b & 0x0F;
                let byte1 = bytes.get(i + 1).copied().unwrap_or(0);
                let mask_lo = (byte1 >> 4) & 0x0F;
                let mask = (mask_hi << 4) | mask_lo;
                if mask == 0b0000_0011 {
                    multiple_with_mask_3 += 1;
                }
                i += 2;
            } else if b & 0xFC == 0xD0 {
                // Multiple1: 3 bytes
                multiple_count += 1;
                i += 3;
            } else if b & 0xFC == 0xD4 {
                // Multiple2: 4 bytes
                multiple_count += 1;
                i += 4;
            } else if b & 0x80 == 0 {
                // Single0: 1 byte
                i += 1;
            } else if b & 0xE0 == 0x80 {
                // Single1: 2 bytes
                i += 2;
            } else if b & 0xE0 == 0xA0 {
                // Single2: 3 bytes
                i += 3;
            } else {
                // Unknown -- stop walking to avoid runaway.
                break;
            }
        }

        assert_eq!(
            multiple_with_mask_3, 1,
            "expected exactly one Multiple frame with mask=0b11 (TRUE+PERF_CNT_0 \
             coalesced at cycle 4); got {} Multiple frames total, {} with mask=0b11. \
             Buffer: {:02X?}",
            multiple_count, multiple_with_mask_3, bytes
        );
    }

    /// Verifies that PC values are threaded from the core EventLog through
    /// the coordinator drain loop into the trace unit's mode-1 (EventPc) encoder.
    ///
    /// Flow under test:
    ///   EventType::InstrVector { pc: 0x100 }
    ///     -> core.context.timing_context().events
    ///     -> coordinator Phase 2 drain
    ///     -> trace::event_pc(&evt.event) = Some(0x100)
    ///     -> notify_core_trace_event(37, cycle, Some(0x100))
    ///     -> TraceUnit::notify_event(37, 0, Some(0x100))
    ///     -> pending_pc = 0x100, pending_slot_mask = 0b1
    ///     -> Phase 3f commit_cycle -> encode_event_pc(mask=1, pc=0x100)
    ///     -> byte_buffer[8..12] = [0xC4, 0x04, 0x01, 0x00]
    ///
    /// The test locks in that PC threading actually flows end-to-end from the
    /// EventType variant through event_pc() to the encoded EventPC frame.
    #[test]
    fn test_instr_vector_pc_threads_into_mode1_event_pc_frame() {
        use crate::interpreter::state::EventType;

        let mut engine = InterpreterEngine::new_npu1();

        // Enable core (0,2) with NOPs so the step doesn't halt.
        engine.enable_core(0, 2);
        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00u8; 256]);
        }

        // Configure core_trace on tile (0,2) in EventPc mode (mode=1):
        //   start_event = TRUE (hw_id=1), stop_event = NONE (hw_id=0)
        //   slot 0 = INSTR_VECTOR (hw_id=37)
        {
            let tile = engine.device_mut().array.tile_mut(0, 2);
            // Trace_Control0: stop[31:24]=0, start[23:16]=1, mode[1:0]=1
            let ctrl0 = (0u32 << 24) | (1u32 << 16) | 1u32;
            tile.core_trace.write_register(0x00, ctrl0);
            // Trace_Event0: slot 0 = INSTR_VECTOR = 37
            tile.core_trace.write_register(0x10, 37u32);

            // Pre-arm: fire the start event (TRUE=1) directly so the trace
            // unit transitions Idle->Running before the coordinator's Phase 2.
            // This mirrors what Phase 3c would do one cycle later, but we
            // need it done before Phase 2 drains the synthetic event below.
            //
            // HW pipelines Idle->Running by 1 cycle: notify_event(start, 0)
            // arms the unit; only events from cycle > 0 are recorded. The
            // unmatched-event call at cycle 1 trips the cycle-advance
            // promotion path so state is Running before the drain runs.
            tile.core_trace.notify_event(1, 0, None);
            tile.core_trace.notify_event(0xFF, 1, None);
        }

        // Inject a synthetic InstrVector event at cycle 0 into core (0,2).
        // The coordinator drain loop reads from trace_events_consumed=0,
        // which points to this event, and will call notify_core_trace_event
        // with the pc extracted by trace::event_pc().
        {
            let ctx = engine.core_context_mut(0, 2).unwrap();
            ctx.timing_context_mut().events.record(0, EventType::InstrVector { pc: 0x100 });
        }

        // Run 1 cycle: the drain loop processes the synthetic InstrVector event,
        // calls notify_core_trace_event(37, 0, Some(0x100)), which sets
        // pending_slot_mask=0b1 and pending_pc=0x100. Phase 3f commits the
        // frame: encode_event_pc(mask=1, pc=0x0100) appends 4 bytes.
        let _cycles = engine.run(1);

        let tile = engine.device().array.tile(0, 2);
        let bytes = tile.core_trace.encoded_bytes();

        // Start marker for mode 1 is 0xF1 (not 0xF0).
        assert_eq!(
            bytes.first().copied(),
            Some(0xF1),
            "expected mode-1 Start marker 0xF1 as first byte; got {:02X?}",
            bytes
        );
        assert!(
            bytes.len() >= 12,
            "expected at least 12 bytes (8 start marker + 4 EventPC frame); \
             got {} bytes: {:02X?}",
            bytes.len(),
            bytes
        );

        // bytes 0-7: Start marker (0xF1 + 7 timer bytes in EventPc mode).
        // bytes 8-11: the EventPC frame for the InstrVector event.
        // encode_event_pc(mask=0b00000001, pc14=0x0100):
        //   byte0 = 0b1100_0100 | (mask >> 6 & 0b11) = 0xC4 | 0 = 0xC4
        //   byte1 = (mask & 0b0011_1111) << 2 = (1 & 63) << 2 = 0x04
        //   byte2 = (pc14 >> 8) & 0x3F = (0x100 >> 8) & 0x3F = 0x01
        //   byte3 = pc14 & 0xFF = 0x00
        let frame = &bytes[8..12];
        assert_eq!(
            frame,
            &[0xC4, 0x04, 0x01, 0x00],
            "EventPC frame mismatch: expected [0xC4, 0x04, 0x01, 0x00] \
             (mask=0b1, pc=0x100), got {:02X?}. \
             Full buffer: {:02X?}",
            frame,
            bytes
        );
    }

    /// End-to-end perfcnt -> mode-1 round-trip.
    ///
    /// Drives the full chain that the test_perfcnt_threshold_routes_to_trace_unit
    /// test only covers in mode-0:
    ///   perfcnt threshold fires
    ///     -> coordinator Phase 3e drain
    ///     -> notify_core_trace_event(PERF_CNT_0, cycle, None)  [no PC]
    ///     -> TraceUnit::notify_event in EventPc mode
    ///     -> pending_pc = 0 (sentinel for None)
    ///     -> commit_cycle -> encode_event_pc(mask, pc=0)
    ///     -> 4-byte EventPC frame in byte_buffer
    ///
    /// PERF_CNT_N events stamp the frame with the core's current PC (the
    /// same value HW samples when the threshold fires), so a counter that
    /// crosses threshold mid-execution shows up in mode-1 frames at the
    /// in-flight PC rather than as a PC=0 sentinel. HW emits the same:
    /// real captures of `add_one_using_dma` show PERF_CNT_2 frames stamped
    /// with PCs in the compute body and in the lock-stall pipeline PC
    /// (e.g., 204 and 832), never 0.
    #[test]
    fn perfcnt_threshold_routes_to_mode1_event_pc_frame_with_core_pc() {
        use xdna_archspec::aie2::trace_events::core_events::PERF_CNT_0;

        let mut engine = InterpreterEngine::new_npu1();

        engine.enable_core(0, 2);
        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00u8; 256]);
        }

        // Perf counter 0: start=TRUE, threshold=5.  Fires at cycle 4 (5 ticks
        // across cycles 0..4 with active-cycle tick).
        {
            let tile = engine.device_mut().array.tile_mut(0, 2);
            let ctrl0 = 1u32 | (0u32 << 8); // start=TRUE(1), stop=NONE(0)
            tile.core_perf_counters.write_control_start_stop(ctrl0, 0, 1, 7);
            tile.core_perf_counters.write_event_value(0, 5);
            tile.core_perf_counters.handle_event(1); // arm
        }

        // Trace: mode=EventPc(1), start=TRUE(1), slot 0 = PERF_CNT_0.
        {
            let tile = engine.device_mut().array.tile_mut(0, 2);
            // Trace_Control0: stop[31:24]=0, start[23:16]=1, mode[1:0]=1
            let ctrl0 = (0u32 << 24) | (1u32 << 16) | 1u32;
            tile.core_trace.write_register(0x00, ctrl0);
            tile.core_trace.write_register(0x10, PERF_CNT_0 as u32);
        }

        // Run enough cycles for the threshold to fire and Phase 3f to commit.
        let _cycles = engine.run(10);

        let tile = engine.device().array.tile(0, 2);
        let bytes = tile.core_trace.encoded_bytes();

        // Start marker = 0xF1 for mode 1.
        assert_eq!(
            bytes.first().copied(),
            Some(0xF1),
            "mode-1 Start marker 0xF1 expected; got {:02X?}",
            bytes
        );

        // After 8-byte start marker, expect at least one EventPC frame.
        // Discriminator: (b & 0b1111_1100) == 0b1100_0100, mask=0b1 (slot 0).
        // The 5th NOP issues at PC=0x14 (each NOP advances PC by 4 bytes,
        // 5 cycles after PC=0 -> PC=0x14), so byte3 = 0x14.
        assert!(
            bytes.len() >= 12,
            "expected >= 12 bytes (8-byte Start + 4-byte EventPC); got {}: {:02X?}",
            bytes.len(),
            bytes
        );
        let frame = &bytes[8..12];
        assert_eq!(
            frame,
            &[0xC4, 0x04, 0x00, 0x14],
            "perfcnt threshold did not produce expected mode-1 EventPC frame \
             (mask=0b1, pc=0x14 at firing cycle); got {:02X?}.  Full buffer: {:02X?}",
            frame,
            bytes
        );
    }

    // ----------------------------------------------------------------------
    // Tile_Control isolation -> NeighborLocks gating
    //
    // The coordinator constructs NeighborLocks by calling
    // build_neighbor_locks_with_isolation. These tests pin the mapping from
    // isolation byte to per-direction Some/None state, which is what the
    // executor's route_lock fallback consumes. East quadrant locks live on
    // the executing tile and are unaffected by isolation.
    // ----------------------------------------------------------------------

    fn lock_vec(n: usize) -> Vec<Lock> {
        (0..n).map(|_| Lock::new(0)).collect()
    }

    #[test]
    fn build_neighbor_locks_no_isolation_passes_all_slices() {
        let mut s = lock_vec(16);
        let mut w = lock_vec(16);
        let mut n = lock_vec(16);
        let nlocks = build_neighbor_locks_with_isolation(
            0,
            Some(s.as_mut_slice()),
            Some(w.as_mut_slice()),
            Some(n.as_mut_slice()),
        );
        assert!(nlocks.south.is_some());
        assert!(nlocks.west.is_some());
        assert!(nlocks.north.is_some());
    }

    #[test]
    fn build_neighbor_locks_each_isolation_bit_hides_only_its_direction() {
        use crate::device::tile::isolation as iso;
        let cases = [(iso::SOUTH, "south"), (iso::WEST, "west"), (iso::NORTH, "north")];
        for (bit, dir) in cases {
            let mut s = lock_vec(16);
            let mut w = lock_vec(16);
            let mut n = lock_vec(16);
            let nlocks = build_neighbor_locks_with_isolation(
                bit,
                Some(s.as_mut_slice()),
                Some(w.as_mut_slice()),
                Some(n.as_mut_slice()),
            );
            // The direction matching the bit must be hidden; others stay
            // available. East/Internal isn't in NeighborLocks at all (own
            // tile), so it's implicitly unaffected.
            match dir {
                "south" => {
                    assert!(nlocks.south.is_none(), "SOUTH bit must hide south slice");
                    assert!(nlocks.west.is_some());
                    assert!(nlocks.north.is_some());
                }
                "west" => {
                    assert!(nlocks.south.is_some());
                    assert!(nlocks.west.is_none(), "WEST bit must hide west slice");
                    assert!(nlocks.north.is_some());
                }
                "north" => {
                    assert!(nlocks.south.is_some());
                    assert!(nlocks.west.is_some());
                    assert!(nlocks.north.is_none(), "NORTH bit must hide north slice");
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn build_neighbor_locks_all_directions_hides_every_slice() {
        use crate::device::tile::isolation as iso;
        let mut s = lock_vec(16);
        let mut w = lock_vec(16);
        let mut n = lock_vec(16);
        let nlocks = build_neighbor_locks_with_isolation(
            iso::ALL_DIRECTIONS,
            Some(s.as_mut_slice()),
            Some(w.as_mut_slice()),
            Some(n.as_mut_slice()),
        );
        assert!(nlocks.south.is_none());
        assert!(nlocks.west.is_none());
        assert!(nlocks.north.is_none());
    }

    #[test]
    fn build_neighbor_locks_passes_through_none_inputs() {
        // When the coordinator has no neighbor at all (edge tile), the
        // input slices are already None. Isolation must not turn None
        // into Some -- it can only hide a present slice.
        let nlocks = build_neighbor_locks_with_isolation(0, None, None, None);
        assert!(nlocks.south.is_none());
        assert!(nlocks.west.is_none());
        assert!(nlocks.north.is_none());
    }

    /// End-to-end: bring up the engine, set tile.isolation = SOUTH on a
    /// compute tile, and verify the gate engages without crashing the
    /// step loop. Pre-isolation behavior would let the executor see the
    /// south neighbor's locks; post-isolation it shouldn't. We can't
    /// directly inspect the in-flight NeighborLocks (the closure owns
    /// it), but a clean step + correct status proves the wiring holds.
    #[test]
    fn engine_step_survives_with_isolation_set() {
        use crate::device::tile::isolation as iso;
        let mut engine = InterpreterEngine::new_npu1();
        engine.enable_core(0, 2);
        if let Some(tile) = engine.device_mut().tile_mut(0, 2) {
            tile.write_program(0, &[0x00, 0x00, 0x00, 0x00]); // NOP
            tile.isolation = iso::ALL_DIRECTIONS;
        }
        engine.step();
        assert_eq!(engine.status(), EngineStatus::Running);
        assert_eq!(engine.total_cycles(), 1);
    }

    #[test]
    fn ctrl_write_to_unknown_offset_sets_slverr_and_suppresses_write() {
        use crate::device::tile::CtrlPacketAction;
        let mut engine = InterpreterEngine::new_npu1();
        // 0x1F200 is verified SubsystemKind::Unknown on a compute tile.
        engine.dispatch_ctrl_action(CtrlPacketAction::WriteRegister {
            col: 0,
            row: 2,
            offset: 0x1F200,
            value: 0xABCD_1234,
        });
        let tile = engine.device.array.get(0, 2).expect("compute tile (0,2)");
        assert_eq!(tile.pkt_handler_status & 0x4, 0x4, "SLVERR bit must latch");
        assert!(
            tile.registers_ref().get(&0x1F200).is_none(),
            "undecoded write must be suppressed (not stored)"
        );
        assert!(
            !matches!(engine.status(), EngineStatus::Error),
            "SLVERR is poll-only sticky -- engine must not abort"
        );
    }

    #[test]
    fn ctrl_read_of_unknown_offset_sets_slverr_no_response() {
        use crate::device::tile::CtrlPacketAction;
        let mut engine = InterpreterEngine::new_npu1();
        engine.dispatch_ctrl_action(CtrlPacketAction::ReadRegisters {
            col: 0,
            row: 2,
            offset: 0x1F200,
            count: 2,
            response_id: 7,
        });
        let tile = engine.device.array.get(0, 2).expect("compute tile (0,2)");
        assert_eq!(tile.pkt_handler_status & 0x4, 0x4, "SLVERR bit must latch");
        assert!(tile.pending_ctrl_response.is_empty(), "undecoded read must not queue a response");
        assert!(!matches!(engine.status(), EngineStatus::Error));
    }

    #[test]
    fn ctrl_write_to_valid_offset_no_slverr_and_applies() {
        use crate::device::tile::CtrlPacketAction;
        let mut engine = InterpreterEngine::new_npu1();
        // 0x400 is compute data memory -- decodes, must NOT SLVERR.
        engine.dispatch_ctrl_action(CtrlPacketAction::WriteRegister {
            col: 0,
            row: 2,
            offset: 0x400,
            value: 0x0000_0001,
        });
        let tile = engine.device.array.get(0, 2).expect("compute tile (0,2)");
        assert_eq!(
            tile.pkt_handler_status & 0x4,
            0,
            "decodable offset must NOT set SLVERR (false-positive guard)"
        );
    }
}
