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
use crate::device::tile::TileType;
use crate::device::DeviceState;
use crate::parser::{AieElf, MemoryRegion};
use crate::interpreter::bundle::VliwBundle;
use crate::interpreter::core::{CoreInterpreter, CoreStatus, StepResult};
use crate::interpreter::decode::InstructionDecoder;
use crate::interpreter::execute::{CycleAccurateExecutor, NeighborMemory};
use crate::interpreter::state::{EventType, ExecutionContext};

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
}

impl CoreState {
    /// Create a new core state with cycle-accurate executor.
    fn new() -> Self {
        Self {
            interpreter: CoreInterpreter::new(
                InstructionDecoder::load_default(),
                CycleAccurateExecutor::new(),
            ),
            context: ExecutionContext::new(),
            enabled: false,
            trace_events_consumed: 0,
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
    /// Counter for cycles with no progress while all cores halted.
    /// Used to detect deadlock where DMAs are stalled waiting for resources.
    no_progress_cycles: u32,
    /// Last cycle's words routed (to detect progress).
    last_words_routed: usize,
    /// Last cycle's total DMA bytes transferred (to detect DMA-level progress
    /// even when no stream words are routed -- e.g., during lock operations).
    last_dma_bytes: u64,
    /// Global trace log collecting events from all tiles.
    /// DMA events are drained from engines each cycle; instruction events
    /// come from per-core TimingContexts. Events include (col, row) source
    /// info embedded in the TileTracedEvent wrapper.
    trace_log: Vec<TileTracedEvent>,
}

/// A trace event tagged with its source tile coordinates.
/// This wraps TimestampedEvent to add spatial information needed for
/// Perfetto export (each tile maps to a separate Perfetto process).
#[derive(Debug, Clone, Copy)]
pub struct TileTracedEvent {
    /// Source tile column.
    pub col: u8,
    /// Source tile row.
    pub row: u8,
    /// Cycle when the event occurred.
    pub cycle: u64,
    /// The event type.
    pub event: EventType,
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

        // Create core states for all possible positions
        let num_cores = cols * rows;
        let cores = (0..num_cores).map(|_| CoreState::new()).collect();

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
            no_progress_cycles: 0,
            last_words_routed: 0,
            last_dma_bytes: 0,
            trace_log: Vec::new(),
        }
    }

    /// Create a new engine with cycle-accurate timing.
    ///
    /// This is an alias for `new()` - all execution is now cycle-accurate.
    /// Preserved for backward compatibility.
    #[deprecated(since = "0.2.0", note = "All execution is now cycle-accurate. Use new() instead.")]
    pub fn new_cycle_accurate(device: DeviceState) -> Self {
        Self::new(device)
    }

    /// Create engine for NPU1 (Phoenix).
    pub fn new_npu1() -> Self {
        Self::new(DeviceState::new_npu1())
    }

    /// Create engine for NPU2 (Strix).
    pub fn new_npu2() -> Self {
        Self::new(DeviceState::new_npu2())
    }

    /// Create engine for NPU1 (Phoenix) with cycle-accurate timing.
    ///
    /// This is an alias for `new_npu1()` - all execution is now cycle-accurate.
    #[deprecated(since = "0.2.0", note = "All execution is now cycle-accurate. Use new_npu1() instead.")]
    pub fn new_cycle_accurate_npu1() -> Self {
        Self::new_npu1()
    }

    /// Create engine for NPU2 (Strix) with cycle-accurate timing.
    ///
    /// This is an alias for `new_npu2()` - all execution is now cycle-accurate.
    #[deprecated(since = "0.2.0", note = "All execution is now cycle-accurate. Use new_npu2() instead.")]
    pub fn new_cycle_accurate_npu2() -> Self {
        Self::new_npu2()
    }

    /// Check if this engine is running in cycle-accurate mode.
    ///
    /// Always returns `true` - all execution is now cycle-accurate.
    pub fn is_cycle_accurate(&self) -> bool {
        true
    }

    /// Get the engine status.
    pub fn status(&self) -> EngineStatus {
        self.status
    }

    /// Get total cycles executed.
    pub fn total_cycles(&self) -> u64 {
        self.total_cycles
    }

    /// Get total instructions executed across all cores.
    pub fn total_instructions(&self) -> u64 {
        self.total_instructions
    }

    /// Get the global trace log (DMA and core events from all tiles).
    pub fn trace_log(&self) -> &[TileTracedEvent] {
        &self.trace_log
    }

    /// Drain all per-core instruction trace events into the global trace log.
    /// Call this after execution completes (before export) to collect
    /// instruction-level events from each core's TimingContext.
    pub fn collect_core_trace_events(&mut self) {
        for col in 0..self.cols {
            for row in self.compute_row_start..self.rows {
                let idx = col * self.rows + row;
                if !self.cores[idx].enabled {
                    continue;
                }
                let events = self.cores[idx].context.timing_context().events.events();
                for evt in events {
                    self.trace_log.push(TileTracedEvent {
                        col: col as u8,
                        row: row as u8,
                        cycle: evt.cycle,
                        event: evt.event,
                    });
                }
            }
        }

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

    /// Enable a core at (col, row).
    pub fn enable_core(&mut self, col: usize, row: usize) {
        if let Some(core) = self.get_core_mut(col, row) {
            core.enabled = true;
        }
    }

    /// Disable a core at (col, row).
    pub fn disable_core(&mut self, col: usize, row: usize) {
        if let Some(core) = self.get_core_mut(col, row) {
            core.enabled = false;
        }
    }

    /// Check if a core is enabled.
    pub fn is_core_enabled(&self, col: usize, row: usize) -> bool {
        self.get_core(col, row)
            .map(|c| c.enabled)
            .unwrap_or(false)
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

            for seg in elf.load_segments() {
                let vaddr = seg.vaddr as usize;

                match seg.region {
                    MemoryRegion::Program => {
                        tile.write_program(vaddr, seg.data);
                    }
                    MemoryRegion::Data => {
                        // Data memory starts at 0x00070000 in AIE address space
                        let dm_offset = vaddr.saturating_sub(0x00070000);
                        let dm = tile.data_memory_mut();
                        if dm_offset + seg.data.len() <= dm.len() {
                            dm[dm_offset..dm_offset + seg.data.len()].copy_from_slice(seg.data);
                        }
                    }
                    _ => {}
                }
            }
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

        // Phase 1: Sync DMA start requests from tiles to DMA engines
        self.sync_dma_start_requests();

        // Phase 2: Step all DMA engines and stream routing.
        // This ensures lock releases from DMA are visible when cores
        // snapshot MemTile locks for memory module access.
        //
        // Set cycle timestamp on DMA engines before stepping so trace
        // events get the correct cycle number.
        self.device.array.set_dma_cycle(self.total_cycles);

        let (dma_active, streams_moved, words_routed) =
            self.device.array.step_data_movement(&mut self.host_memory);

        // Drain DMA trace events into the global trace log and notify
        // memory module trace units so they can produce binary trace packets.
        // MemTiles use a different event ID namespace than compute tiles.
        for (col, row, cycle, event) in self.device.array.drain_dma_trace_events() {
            if let Some(tile) = self.device.array.get_mut(col, row) {
                let hw_id = if tile.is_mem_tile() {
                    crate::trace::memtile_event_to_hw_id(&event)
                } else {
                    crate::trace::mem_event_to_hw_id(&event)
                };
                if let Some(id) = hw_id {
                    tile.mem_trace.notify_event(id, cycle);
                }
            }
            self.trace_log.push(TileTracedEvent { col, row, cycle, event });
        }

        // Phase 2b: Generate PORT_RUNNING events from cycle_active flags.
        //
        // PORT_RUNNING is a level event: it fires every cycle a monitored port
        // is active. step_data_movement() seeds cycle_active from pre-existing
        // FIFO state and marks ports active as data flows through during routing.
        // We read those flags here to notify trace units.
        {
            let cycle = self.total_cycles;
            let mut port_events: Vec<(usize, u8, bool)> = Vec::new();
            for idx in 0..self.device.array.tiles.len() {
                let tile = &self.device.array.tiles[idx];
                if !tile.core_trace.is_configured() && !tile.mem_trace.is_configured() {
                    continue;
                }

                for event_port in 0..8u8 {
                    if let Some((port_idx, is_master)) = tile.event_port_selection[event_port as usize] {
                        let port_active = if is_master {
                            tile.stream_switch.masters.get(port_idx as usize)
                                .map_or(false, |p| p.cycle_active)
                        } else {
                            tile.stream_switch.slaves.get(port_idx as usize)
                                .map_or(false, |p| p.cycle_active)
                        };

                        if port_active {
                            let hw_id = if tile.is_compute() {
                                crate::trace::core_port_running_hw_id(event_port)
                            } else {
                                crate::trace::memtile_port_running_hw_id(event_port)
                            };
                            port_events.push((idx, hw_id, tile.is_compute()));
                        }
                    }
                }
            }
            for (idx, hw_id, is_compute) in port_events {
                let tile = &mut self.device.array.tiles[idx];
                if is_compute {
                    tile.core_trace.notify_event(hw_id, cycle);
                } else {
                    tile.mem_trace.notify_event(hw_id, cycle);
                }
            }
        }

        if dma_active || streams_moved {
            any_running = true;
        }

        // Phase 3: Step each enabled core (after DMA, so locks are up-to-date)
        for col in 0..self.cols {
            for row in self.compute_row_start..self.rows {
                let idx = col * self.rows + row;

                if !self.cores[idx].enabled {
                    continue;
                }

                // For compute tiles, we need memory module lock routing.
                // Memory module locks (48-63) should access the MemTile (row 1) locks.
                // Copy MemTile locks before stepping, then copy changes back after.
                let mem_tile_row = 1; // MemTile is always row 1
                let mut mem_tile_locks_copy: Option<Vec<crate::device::tile::Lock>> = None;

                // Copy memory tile locks if we're in a compute row
                if row > mem_tile_row {
                    if let Some(mem_tile) = self.device.tile(col, mem_tile_row) {
                        mem_tile_locks_copy = Some(mem_tile.locks.clone());
                    }
                }

                // Build cross-tile neighbor memory context.
                // Snapshots are loaded eagerly before the mutable tile borrow
                // because the memory functions can't access DeviceState during
                // execution. Only existing neighbors are cloned (edge tiles
                // have fewer). This costs up to 192KB per core per cycle.
                //
                // Optimization paths if profiling shows this matters:
                // 1. Pre-allocate per-core snapshot buffers and reuse across cycles
                // 2. Use shared immutable &[u8] instead of clones (requires proving
                //    no cross-tile writes within the same cycle)
                // 3. Track which cores use cross-tile addressing and skip others
                let mut neighbors = NeighborMemory::new(col, row);
                neighbors.ensure_snapshot(1, &self.device); // West
                neighbors.ensure_snapshot(2, &self.device); // North
                neighbors.ensure_snapshot(3, &self.device); // East

                // Get tile for this core
                if let Some(tile) = self.device.tile_mut(col, row) {
                    if tile.tile_type != TileType::Compute {
                        continue;
                    }

                    let core = &mut self.cores[idx];

                    // Step with memory tile locks and neighbor memory
                    let result = if let Some(ref mut mem_locks) = mem_tile_locks_copy {
                        core.interpreter.step_with_mem_locks(
                            &mut core.context, tile,
                            Some(mem_locks.as_mut_slice()),
                            Some(&mut neighbors),
                        )
                    } else {
                        core.interpreter.step_with_mem_locks(
                            &mut core.context, tile,
                            None,
                            Some(&mut neighbors),
                        )
                    };

                    match result {
                        StepResult::Continue => {
                            all_halted = false;
                            any_running = true;
                            self.total_instructions += 1;
                        }
                        StepResult::WaitLock { .. } | StepResult::WaitDma { .. } | StepResult::WaitStream { .. } => {
                            // Stalled, but still active - not halted
                            all_halted = false;
                            any_running = true;
                        }
                        StepResult::Halt => {
                            // This core halted - don't set all_halted=false
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
                    // We track how many events have been consumed to avoid re-processing
                    // the full EventLog history every cycle.
                    let events = core.context.timing_context().events.events();
                    let new_start = core.trace_events_consumed;
                    if new_start < events.len() {
                        for evt in &events[new_start..] {
                            if let Some(hw_id) = crate::trace::core_event_to_hw_id(&evt.event) {
                                tile.core_trace.notify_event(hw_id, evt.cycle);
                            }
                        }
                        core.trace_events_consumed = events.len();
                    }
                }

                // Apply buffered cross-tile writes to neighbor tiles
                if neighbors.has_pending_writes() {
                    neighbors.apply_writes(&mut self.device);
                }

                // Copy modified memory tile locks back to the MemTile
                if let Some(mem_locks) = mem_tile_locks_copy {
                    if let Some(mem_tile) = self.device.tile_mut(col, mem_tile_row) {
                        mem_tile.locks.copy_from_slice(&mem_locks);
                    }
                }
            }
        }

        // Phase 4: Update tile DMA channel state from engine state
        self.sync_dma_completion();

        self.total_cycles += 1;

        // Determine if we should halt the engine.
        //
        // The engine halts when:
        // 1. All cores have halted (executed `done` instruction), AND
        // 2. Either:
        //    a. No DMA activity at all, OR
        //    b. DMA is active but no progress for multiple cycles (deadlock)
        //
        // Deadlock detection: If all cores are halted and DMA makes no progress
        // (no words routed) for several consecutive cycles, the system is stuck.
        // This handles cases like:
        // - DMA waiting for locks that will never be released
        // - DMA waiting for stream data that will never arrive
        // - Circular dependencies in the data flow
        //
        // Check if any cores were enabled (to distinguish DMA-only tests from real programs)
        let any_cores_enabled = self.cores.iter().any(|c| c.enabled);

        // Case 1: No cores enabled - halt if no activity at all
        if !any_cores_enabled && !any_running {
            self.status = EngineStatus::Halted;
            return;
        }

        // Case 2: Cores enabled and all halted - check for deadlock
        if all_halted && any_cores_enabled {
            let dma_waiting = self.device.array.any_dma_waiting_for_lock();

            // Check if we're making progress.
            // Progress means EITHER stream words were routed OR DMA transferred
            // more bytes. During pipeline drainage after core completion, DMA may
            // spend several cycles on lock acquire/release without routing stream
            // words -- but bytes_transferred will still increase when data moves.
            let dma_bytes = self.device.array.total_dma_bytes_transferred();
            let making_progress = words_routed > 0
                || dma_bytes > self.last_dma_bytes;
            self.last_words_routed = words_routed;
            self.last_dma_bytes = dma_bytes;

            if !dma_active {
                // No DMA activity at all - clean halt
                self.status = EngineStatus::Halted;
            } else if !making_progress {
                // DMA active but no progress this cycle
                self.no_progress_cycles += 1;

                // After 50 cycles of no progress with all cores halted, give up.
                // Lock acquire/release cycles and multi-hop stream routing can
                // cause gaps of 10-20 cycles with no visible byte progress, so
                // we need a generous threshold to avoid false positives.
                if self.no_progress_cycles >= 50 {
                    if dma_waiting {
                        log::info!(
                            "Engine halting after {} cycles: all cores done, DMAs stalled on unreleased locks",
                            self.no_progress_cycles
                        );
                    } else {
                        log::info!(
                            "Engine halting after {} cycles: all cores done, DMA deadlock detected",
                            self.no_progress_cycles
                        );
                    }
                    self.status = EngineStatus::Halted;
                }
            } else {
                // Making progress - reset counter
                self.no_progress_cycles = 0;
            }
        } else {
            // Cores still running - reset counter
            self.no_progress_cycles = 0;
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
                            if matches!(state, ChannelState::Complete | ChannelState::Idle) {
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

            if matches!(self.status, EngineStatus::Halted | EngineStatus::Error) {
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
                    if tile.tile_type == TileType::Compute {
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
            if matches!(state, ChannelState::Complete | ChannelState::Idle) {
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
        // All engines are now cycle-accurate
        let engine1 = InterpreterEngine::new_npu1();
        assert!(engine1.is_cycle_accurate());

        let engine2 = InterpreterEngine::new_npu2();
        assert!(engine2.is_cycle_accurate());

        // Legacy constructors (deprecated) also return cycle-accurate engines
        #[allow(deprecated)]
        let engine3 = InterpreterEngine::new_cycle_accurate_npu1();
        assert!(engine3.is_cycle_accurate());
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
