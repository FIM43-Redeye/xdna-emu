//! Multi-core coordinator implementation.
//!
//! The engine manages multiple core interpreters and coordinates their execution.
//! It also coordinates DMA engines and host memory for data transfers.
//!
//! # Execution Modes
//!
//! The engine supports two execution modes:
//!
//! - **Fast mode** (`InterpreterEngine::new()`): Uses `FastExecutor` for quick
//!   functional simulation. All instructions execute in 1 cycle.
//!
//! - **Cycle-accurate mode** (`InterpreterEngine::new_cycle_accurate()`): Uses
//!   `CycleAccurateExecutor` with full timing model - hazard detection, memory
//!   bank conflicts, branch penalties, and event tracing.

use crate::device::dma::ChannelState;
use crate::device::host_memory::HostMemory;
use crate::device::tile::TileType;
use crate::device::DeviceState;
use crate::parser::{AieElf, MemoryRegion};
use crate::interpreter::bundle::VliwBundle;
use crate::interpreter::core::{CoreInterpreter, CoreStatus, StepResult};
use crate::interpreter::decode::InstructionDecoder;
use crate::interpreter::execute::{CycleAccurateExecutor, FastExecutor};
use crate::interpreter::state::ExecutionContext;

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

/// Type alias for fast interpreter (no timing).
type FastInterpreter = CoreInterpreter<InstructionDecoder, FastExecutor>;

/// Type alias for cycle-accurate interpreter (full timing).
type CycleAccurateInterpreter = CoreInterpreter<InstructionDecoder, CycleAccurateExecutor>;

/// Interpreter variant - either fast or cycle-accurate.
///
/// This enum allows the engine to switch between execution modes at creation time.
/// Fast mode is used for quick functional testing, while cycle-accurate mode
/// provides detailed timing information for performance analysis.
enum InterpreterKind {
    /// Fast execution mode (1 cycle per instruction, no hazard tracking).
    Fast(FastInterpreter),
    /// Cycle-accurate execution mode (models pipeline, hazards, memory timing).
    CycleAccurate(CycleAccurateInterpreter),
}

impl InterpreterKind {
    /// Create a new fast interpreter.
    fn new_fast() -> Self {
        Self::Fast(CoreInterpreter::new(
            InstructionDecoder::load_default(),
            FastExecutor::new(),
        ))
    }

    /// Create a new cycle-accurate interpreter.
    fn new_cycle_accurate() -> Self {
        Self::CycleAccurate(CoreInterpreter::new(
            InstructionDecoder::load_default(),
            CycleAccurateExecutor::new(),
        ))
    }

    /// Execute a single step.
    fn step(&mut self, ctx: &mut ExecutionContext, tile: &mut crate::device::tile::Tile) -> StepResult {
        match self {
            Self::Fast(interp) => interp.step(ctx, tile),
            Self::CycleAccurate(interp) => interp.step(ctx, tile),
        }
    }

    /// Get interpreter status.
    fn status(&self) -> CoreStatus {
        match self {
            Self::Fast(interp) => interp.status(),
            Self::CycleAccurate(interp) => interp.status(),
        }
    }

    /// Check if halted.
    fn is_halted(&self) -> bool {
        match self {
            Self::Fast(interp) => interp.is_halted(),
            Self::CycleAccurate(interp) => interp.is_halted(),
        }
    }

    /// Reset the interpreter.
    fn reset(&mut self) {
        match self {
            Self::Fast(interp) => interp.reset(),
            Self::CycleAccurate(interp) => interp.reset(),
        }
    }

    /// Check if this is cycle-accurate mode.
    fn is_cycle_accurate(&self) -> bool {
        matches!(self, Self::CycleAccurate(_))
    }

    /// Get the last decoded bundle (for debugging).
    fn last_bundle(&self) -> Option<&VliwBundle> {
        match self {
            Self::Fast(interp) => interp.last_bundle(),
            Self::CycleAccurate(interp) => interp.last_bundle(),
        }
    }
}

/// Per-core state managed by the engine.
struct CoreState {
    /// Core interpreter (fast or cycle-accurate).
    interpreter: InterpreterKind,
    /// Execution context (registers, PC, flags).
    context: ExecutionContext,
    /// Is this core enabled?
    enabled: bool,
}

impl CoreState {
    /// Create a new core state with fast executor.
    fn new() -> Self {
        Self {
            interpreter: InterpreterKind::new_fast(),
            context: ExecutionContext::new(),
            enabled: false,
        }
    }

    /// Create a new core state with cycle-accurate executor.
    fn new_cycle_accurate() -> Self {
        Self {
            interpreter: InterpreterKind::new_cycle_accurate(),
            context: ExecutionContext::new_with_timing(),
            enabled: false,
        }
    }
}

/// Multi-core interpreter engine.
///
/// Coordinates execution across all compute cores in the device.
/// Also manages DMA engines and host memory for data transfers.
///
/// # Execution Modes
///
/// Create with `new()` for fast functional simulation, or `new_cycle_accurate()`
/// for detailed timing with hazard detection, memory conflicts, and event tracing.
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
    /// Whether cycle-accurate timing is enabled.
    cycle_accurate: bool,
    /// Counter for cycles with no progress while all cores halted.
    /// Used to detect deadlock where DMAs are stalled waiting for resources.
    no_progress_cycles: u32,
    /// Last cycle's words routed (to detect progress).
    last_words_routed: usize,
}

impl InterpreterEngine {
    /// Create a new engine from device state (fast mode).
    ///
    /// Uses `FastExecutor` - all instructions execute in 1 cycle with no
    /// pipeline modeling. Ideal for quick functional testing.
    pub fn new(device: DeviceState) -> Self {
        Self::with_mode(device, false)
    }

    /// Create a new engine with cycle-accurate timing.
    ///
    /// Uses `CycleAccurateExecutor` with full timing model:
    /// - Register hazard detection (RAW, WAW, WAR)
    /// - Memory bank conflict modeling
    /// - Branch penalty tracking
    /// - Event tracing for profiling
    pub fn new_cycle_accurate(device: DeviceState) -> Self {
        Self::with_mode(device, true)
    }

    /// Internal constructor with timing mode flag.
    fn with_mode(device: DeviceState, cycle_accurate: bool) -> Self {
        let cols = device.cols();
        let rows = device.rows();
        let compute_row_start = 2; // Rows 0=shim, 1=memtile, 2+=compute

        // Create core states for all possible positions
        let num_cores = cols * rows;
        let cores = if cycle_accurate {
            (0..num_cores).map(|_| CoreState::new_cycle_accurate()).collect()
        } else {
            (0..num_cores).map(|_| CoreState::new()).collect()
        };

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
            cycle_accurate,
            no_progress_cycles: 0,
            last_words_routed: 0,
        }
    }

    /// Create engine for NPU1 (Phoenix) in fast mode.
    pub fn new_npu1() -> Self {
        Self::new(DeviceState::new_npu1())
    }

    /// Create engine for NPU2 (Strix) in fast mode.
    pub fn new_npu2() -> Self {
        Self::new(DeviceState::new_npu2())
    }

    /// Create engine for NPU1 (Phoenix) with cycle-accurate timing.
    pub fn new_cycle_accurate_npu1() -> Self {
        Self::new_cycle_accurate(DeviceState::new_npu1())
    }

    /// Create engine for NPU2 (Strix) with cycle-accurate timing.
    pub fn new_cycle_accurate_npu2() -> Self {
        Self::new_cycle_accurate(DeviceState::new_npu2())
    }

    /// Check if this engine is running in cycle-accurate mode.
    pub fn is_cycle_accurate(&self) -> bool {
        self.cycle_accurate
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
    /// 2. Step all compute cores
    /// 3. Step all DMA engines
    /// 4. Update tile DMA channel state from engine state
    pub fn step(&mut self) {
        if matches!(self.status, EngineStatus::Halted | EngineStatus::Error) {
            return;
        }

        self.status = EngineStatus::Running;
        let mut any_running = false;
        let mut all_halted = true;

        // Phase 1: Sync DMA start requests from tiles to DMA engines
        self.sync_dma_start_requests();

        // Phase 2: Step each enabled core
        for col in 0..self.cols {
            for row in self.compute_row_start..self.rows {
                let idx = col * self.rows + row;

                if !self.cores[idx].enabled {
                    continue;
                }

                // Get tile for this core
                if let Some(tile) = self.device.tile_mut(col, row) {
                    if tile.tile_type != TileType::Compute {
                        continue;
                    }

                    let core = &mut self.cores[idx];
                    let result = core.interpreter.step(&mut core.context, tile);

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
                        StepResult::DecodeError(_) | StepResult::ExecError(_) => {
                            self.status = EngineStatus::Error;
                            return;
                        }
                    }
                }
            }
        }

        // Phase 3: Step all DMA engines and stream routing
        // This includes: DMA transfers, DMA-to-stream routing, and stream propagation
        let (dma_active, streams_moved, words_routed) =
            self.device.array.step_data_movement(&mut self.host_memory);
        if dma_active || streams_moved {
            any_running = true;
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

            // Check if we're making progress
            let making_progress = words_routed > 0 || words_routed != self.last_words_routed;
            self.last_words_routed = words_routed;

            if !dma_active {
                // No DMA activity at all - clean halt
                self.status = EngineStatus::Halted;
            } else if !making_progress {
                // DMA active but no progress this cycle
                self.no_progress_cycles += 1;

                // After 10 cycles of no progress with all cores halted, give up
                // This is generous - real deadlocks are detected quickly
                if self.no_progress_cycles >= 10 {
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
    fn test_cycle_accurate_engine_creation() {
        let engine = InterpreterEngine::new_cycle_accurate_npu1();

        assert!(engine.is_cycle_accurate());
        assert_eq!(engine.status(), EngineStatus::Ready);
        assert_eq!(engine.total_cycles(), 0);
    }

    #[test]
    fn test_fast_vs_cycle_accurate_mode() {
        // Verify fast mode creates fast executor
        let fast_engine = InterpreterEngine::new_npu1();
        assert!(!fast_engine.is_cycle_accurate());

        // Verify cycle-accurate mode creates cycle-accurate executor
        let accurate_engine = InterpreterEngine::new_cycle_accurate_npu1();
        assert!(accurate_engine.is_cycle_accurate());
    }

    #[test]
    fn test_cycle_accurate_timing_context_enabled() {
        let engine = InterpreterEngine::new_cycle_accurate_npu1();

        // Core contexts should have timing enabled
        let ctx = engine.core_context(0, 2).unwrap();
        assert!(ctx.has_timing(), "Cycle-accurate cores should have timing context");
    }

    #[test]
    fn test_cycle_accurate_step() {
        let mut engine = InterpreterEngine::new_cycle_accurate_npu1();

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
    fn test_cycle_accurate_npu2() {
        let engine = InterpreterEngine::new_cycle_accurate_npu2();
        assert!(engine.is_cycle_accurate());
    }
}
