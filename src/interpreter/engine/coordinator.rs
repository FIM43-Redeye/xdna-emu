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
use xdna_archspec::types::TileKind;
use crate::device::DeviceState;
use crate::parser::AieElf;
use crate::interpreter::bundle::VliwBundle;
use crate::interpreter::core::{CoreInterpreter, CoreStatus, StepResult};
use crate::interpreter::decode::InstructionDecoder;
use crate::interpreter::execute::{CycleAccurateExecutor, NeighborMemory};
use crate::interpreter::state::{EventType, ExecutionContext};
use crate::interpreter::timing::MemoryQuadrant;
use xdna_archspec::aie2::SHIM_ROW;

/// Engine execution status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
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
    /// Is this core enabled?
    enabled: bool,
    /// Number of trace events already consumed from the EventLog.
    /// Used to incrementally drain new events each cycle for the
    /// hardware trace unit without needing a drain() method.
    trace_events_consumed: usize,
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
                use crate::device::tile::CtrlPacketAction;
                let ctrl_actions = self.device.array.drain_ctrl_packet_actions();
                for action in ctrl_actions {
                    match action {
                        CtrlPacketAction::WriteRegister { col, row, offset, value } => {
                            self.device.write_tile_register(col, row, offset, value);
                        }
                        CtrlPacketAction::ReadRegisters { col, row, offset, count, response_id } => {
                            self.device.array.handle_read_registers(
                                col, row, offset, count, response_id,
                            );
                        }
                        CtrlPacketAction::Error(msg) => {
                            self.device.array.fatal_errors.push(msg);
                        }
                    }
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
            log::info!(
                "Post-flush routing: {} words in {} iterations",
                total_flush_words, flush_iters
            );
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
            let (_, _, words) =
                self.device.array.step_data_movement(&mut self.host_memory);

            let ctrl_actions = self.device.array.drain_ctrl_packet_actions();
            if ctrl_actions.is_empty() && words == 0 {
                break;
            }

            use crate::device::tile::CtrlPacketAction;
            for action in ctrl_actions {
                match action {
                    CtrlPacketAction::WriteRegister { col, row, offset, value } => {
                        self.device.write_tile_register(col, row, offset, value);
                    }
                    CtrlPacketAction::ReadRegisters { col, row, offset, count, response_id } => {
                        self.device.array.handle_read_registers(
                            col, row, offset, count, response_id,
                        );
                    }
                    CtrlPacketAction::Error(msg) => {
                        self.device.array.fatal_errors.push(msg);
                    }
                }
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
        self.get_core(col, row)
            .map(|c| c.enabled)
            .unwrap_or(false)
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
            let tile = self.device.tile_mut(col, row)
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

                // Build cross-tile neighbor memory context.
                let mut neighbors = NeighborMemory::new(col, row);
                neighbors.ensure_snapshot(MemoryQuadrant::South, &self.device);
                neighbors.ensure_snapshot(MemoryQuadrant::West, &self.device);
                neighbors.ensure_snapshot(MemoryQuadrant::North, &self.device);

                // Get tile for this core
                if let Some(tile) = self.device.tile_mut(col, row) {
                    if !tile.is_compute() {
                        continue;
                    }

                    let core = &mut self.cores[idx];

                    // Build neighbor locks struct
                    let mut nlocks = crate::interpreter::execute::NeighborLocks {
                        south: south_locks.as_mut().map(|v| v.as_mut_slice()),
                        west: west_locks.as_mut().map(|v| v.as_mut_slice()),
                        north: north_locks.as_mut().map(|v| v.as_mut_slice()),
                    };

                    let result = core.interpreter.step_with_neighbor_locks(
                        &mut core.context, tile,
                        &mut nlocks,
                        Some(&mut neighbors),
                    );

                    // Update CoreDebugState with current PC and stall info.
                    // This mirrors the interpreter state into the register
                    // space so host reads of Core_Status/Core_PC return
                    // correct values (matching real hardware behavior).
                    tile.core_debug.update_pc(core.context.pc());
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
                        StepResult::DecodeError(ref e) => {
                            log::error!("Core({},{}) DecodeError at cycle {}: {:?}", col, row, self.total_cycles, e);
                            self.status = EngineStatus::Error;
                            return;
                        }
                        StepResult::ExecError(ref e) => {
                            log::error!("Core({},{}) ExecError at cycle {}: {:?}", col, row, self.total_cycles, e);
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
                    let events = core.context.timing_context().events.events();
                    let new_start = core.trace_events_consumed;
                    if new_start < events.len() {
                        let cycle = self.total_cycles;
                        for evt in &events[new_start..] {
                            if let Some(hw_id) = crate::trace::core_event_to_hw_id(&evt.event) {
                                tile.notify_core_trace_event(hw_id, cycle);
                            }
                        }
                        core.trace_events_consumed = events.len();
                    }
                }

                // Apply buffered cross-tile writes to neighbor tiles
                if neighbors.has_pending_writes() {
                    neighbors.apply_writes(&mut self.device);
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
                    ncol: usize, nrow: usize,
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
        use crate::device::tile::CtrlPacketAction;
        let ctrl_actions = self.device.array.drain_ctrl_packet_actions();
        for action in ctrl_actions {
            match action {
                CtrlPacketAction::WriteRegister { col, row, offset, value } => {
                    self.device.write_tile_register(col, row, offset, value);
                }
                CtrlPacketAction::ReadRegisters { col, row, offset, count, response_id } => {
                    self.device.array.handle_read_registers(
                        col, row, offset, count, response_id,
                    );
                }
                CtrlPacketAction::Error(msg) => {
                    self.device.array.fatal_errors.push(msg);
                }
            }
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
                        tile.notify_core_trace_event(id, cycle);
                    }
                } else if tile.is_mem() {
                    if let Some(id) = crate::trace::memtile_event_to_hw_id(&event) {
                        tile.notify_mem_trace_event(id, cycle);
                    }
                } else {
                    if let Some(id) = crate::trace::mem_event_to_hw_id(&event) {
                        tile.notify_mem_trace_event(id, cycle);
                    }
                }
            }
        }

        // Phase 3b: Generate stream switch port events from cycle flags.
        //
        // Four event types per monitored port, all level events:
        // - PORT_IDLE: port had no data this cycle
        // - PORT_RUNNING: port was active (had data flowing)
        // - PORT_STALLED: port had data but couldn't forward (backpressure)
        // - PORT_TLAST: a TLAST was seen on this port
        //
        // step_data_movement() seeds these flags from FIFO state at routing
        // start and updates them as data flows. We read them here to notify
        // trace units.
        {
            let cycle = self.total_cycles;
            // Collect (tile_idx, hw_event_id, event_type, trace_target) tuples.
            // trace_target: which trace unit to notify (core_trace vs mem_trace).
            let mut port_events: Vec<(usize, u8, EventType, TileKind)> = Vec::new();
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

                        // PORT_RUNNING or PORT_IDLE (mutually exclusive)
                        let (hw_id, evt) = if port.cycle_active {
                            let hw_id = match tt {
                                TileKind::Compute => crate::trace::core_port_running_hw_id(event_port),
                                TileKind::Mem => crate::trace::memtile_port_running_hw_id(event_port),
                                TileKind::ShimNoc | TileKind::ShimPl => crate::trace::shim_port_running_hw_id(event_port),
                            };
                            (hw_id, EventType::PortRunning { port: event_port })
                        } else {
                            let hw_id = match tt {
                                TileKind::Compute => crate::trace::core_port_idle_hw_id(event_port),
                                TileKind::Mem => crate::trace::memtile_port_idle_hw_id(event_port),
                                TileKind::ShimNoc | TileKind::ShimPl => crate::trace::shim_port_idle_hw_id(event_port),
                            };
                            (hw_id, EventType::PortIdle { port: event_port })
                        };
                        port_events.push((idx, hw_id, evt, tt));

                        if port.cycle_stalled {
                            let hw_id = match tt {
                                TileKind::Compute => crate::trace::core_port_stalled_hw_id(event_port),
                                TileKind::Mem => crate::trace::memtile_port_stalled_hw_id(event_port),
                                TileKind::ShimNoc | TileKind::ShimPl => crate::trace::shim_port_stalled_hw_id(event_port),
                            };
                            port_events.push((idx, hw_id, EventType::PortStalled { port: event_port }, tt));
                        }

                        if port.cycle_tlast {
                            let hw_id = match tt {
                                TileKind::Compute => crate::trace::core_port_tlast_hw_id(event_port),
                                TileKind::Mem => crate::trace::memtile_port_tlast_hw_id(event_port),
                                TileKind::ShimNoc | TileKind::ShimPl => crate::trace::shim_port_tlast_hw_id(event_port),
                            };
                            port_events.push((idx, hw_id, EventType::PortTlast { port: event_port }, tt));
                        }
                    }
                }
            }
            for (idx, hw_id, _evt, tt) in port_events {
                let tile = &mut self.device.array.tiles[idx];
                // Compute tiles: core_trace (CoreEvent namespace)
                // Shim tiles: core_trace (PL module, single trace unit)
                // MemTiles: mem_trace (MemTileEvent namespace)
                if tt.is_mem() {
                    tile.notify_mem_trace_event(hw_id, cycle);
                } else {
                    tile.notify_core_trace_event(hw_id, cycle);
                }
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
                tile.notify_mem_trace_event(hw_id, cycle);
            }
            // Emit MEMORY_STALL (core event 23) for each core that lost a
            // bank to the DMA this cycle. Bump the core's stall counter too.
            for core_idx in stalled_cores {
                // Map core index -> tile coordinates for the trace unit.
                let col = core_idx / self.rows;
                let row = core_idx % self.rows;
                if let Some(tile) = self.device.array.get_mut(col as u8, row as u8) {
                    let hw_id = crate::trace::core_event_to_hw_id(
                        &crate::interpreter::state::EventType::MemoryStall { cycles: 1 },
                    );
                    if let Some(id) = hw_id {
                        tile.notify_core_trace_event(id, cycle);
                    }
                }
                let ctx = &mut self.cores[core_idx].context;
                ctx.timing_context_mut().memory_stalls += 1;
                ctx.timing_context_mut().record_event(
                    cycle,
                    crate::interpreter::state::EventType::MemoryStall { cycles: 1 },
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
                    tile.core_trace.notify_event(TRUE_EVENT, cycle);
                }
                if tile.mem_trace.is_configured() {
                    tile.mem_trace.notify_event(TRUE_EVENT, cycle);
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

        // Phase 3f: Commit per-cycle trace frames.
        //
        // Per AM020 event-time mode, a trace unit creates at most one frame
        // per cycle. `notify_event` accumulates slot activity into a bitmask;
        // committing here emits one Single/Multiple frame for this cycle.
        // Without this step the mask only commits lazily on the next cycle's
        // event, which can leave the last cycle's frame uncommitted past the
        // simulation's natural end and inflate routing pressure for the
        // uncommitted bytes when they eventually land at flush time.
        {
            let cycle = self.total_cycles;
            for tile in &mut self.device.array.tiles {
                tile.core_trace.commit_cycle(cycle);
                tile.mem_trace.commit_cycle(cycle);
            }
        }

        // Phase 3e: Tick tile timers and performance counters.
        //
        // Each tile has two 64-bit timers (core module and memory module)
        // that increment every cycle. The timer can generate a trigger
        // event when it reaches a programmed threshold.
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
        for (i, tile) in self.device.array.tiles.iter_mut().enumerate() {
            tile.core_timer.tick();
            tile.mem_timer.tick();
            let core_active = self.cores.get(i).map_or(false, |c| c.active_this_cycle);
            if core_active {
                tile.core_perf_counters.tick_active_cycles();
            } else {
                tile.core_perf_counters.tick_idle_cycles();
            }
            // Memory module perf counters tick unconditionally.
            tile.mem_perf_counters.tick_active_cycles();
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
        let cores_done = if any_cores_enabled { all_halted } else { !any_running };

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
        for core in &mut self.cores {
            core.interpreter.reset();
            core.context.reset();
            core.enabled = false;
        }

        self.status = EngineStatus::Ready;
        self.total_cycles = 0;
        self.total_instructions = 0;
        self.last_dma_bytes = 0;
        self.last_lock_releases = 0;
        self.last_instructions = 0;
        self.stall_cycles = 0;
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
        self.cores
            .iter()
            .filter(|c| c.enabled && !c.interpreter.is_halted())
            .count()
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
        !enabled.is_empty()
            && enabled.iter().all(|c| {
                c.interpreter.is_stalled() || c.interpreter.is_halted()
            })
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
        // (Currently no setup overhead since timing isn't integrated yet)
        for i in 0..50 {
            engine.step();

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
                    state, stats.bytes_transferred, dma.channel_active(2)
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
}
