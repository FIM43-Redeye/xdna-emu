//! Main emulation engine.
//!
//! The engine coordinates execution across all tiles, handling:
//! - Core execution (one step = one cycle per core)
//! - DMA transfers (simulated)
//! - Breakpoints and debug controls
//!
//! # Usage
//!
//! ```ignore
//! let mut engine = Engine::new(device_state);
//! engine.step();      // Execute one cycle
//! engine.run(1000);   // Run for up to 1000 cycles
//! ```

use super::core::{CoreExecutor, CoreStatus};
use crate::device::DeviceState;

/// Breakpoint definition.
#[derive(Debug, Clone)]
pub struct Breakpoint {
    /// Tile column.
    pub col: u8,
    /// Tile row.
    pub row: u8,
    /// Program counter address.
    pub pc: u32,
    /// Whether this breakpoint is enabled.
    pub enabled: bool,
}

impl Breakpoint {
    /// Create a new breakpoint.
    pub fn new(col: u8, row: u8, pc: u32) -> Self {
        Self {
            col,
            row,
            pc,
            enabled: true,
        }
    }
}

/// Engine execution status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineStatus {
    /// Engine is idle (no cores running).
    Idle,
    /// Engine is running.
    Running,
    /// Engine is paused.
    Paused,
    /// Hit a breakpoint.
    Breakpoint { col: u8, row: u8 },
    /// All cores have halted.
    AllHalted,
}

/// Main emulation engine.
///
/// Coordinates execution across all cores in the device.
pub struct Engine {
    /// Device state being emulated.
    pub state: DeviceState,
    /// Core executors (one per compute tile).
    pub executors: Vec<CoreExecutor>,
    /// Breakpoints.
    pub breakpoints: Vec<Breakpoint>,
    /// Current engine status.
    pub status: EngineStatus,
    /// Total cycles executed.
    pub total_cycles: u64,
    /// Maximum cycles before auto-stop (0 = unlimited).
    pub max_cycles: u64,
}

impl Engine {
    /// Create a new engine with the given device state.
    pub fn new(state: DeviceState) -> Self {
        let mut executors = Vec::new();

        // Create an executor for each compute tile
        let cols = state.array.cols();
        let rows = state.array.rows();

        for col in 0..cols {
            for row in 2..rows {
                // Rows 2-5 are compute tiles
                executors.push(CoreExecutor::new(col, row));
            }
        }

        Self {
            state,
            executors,
            breakpoints: Vec::new(),
            status: EngineStatus::Idle,
            total_cycles: 0,
            max_cycles: 0,
        }
    }

    /// Create an engine for NPU1.
    pub fn new_npu1() -> Self {
        Self::new(DeviceState::new_npu1())
    }

    /// Reset the engine to initial state.
    pub fn reset(&mut self) {
        self.state.array.reset();
        for exec in &mut self.executors {
            exec.reset();
        }
        self.status = EngineStatus::Idle;
        self.total_cycles = 0;
    }

    /// Add a breakpoint.
    pub fn add_breakpoint(&mut self, col: u8, row: u8, pc: u32) {
        self.breakpoints.push(Breakpoint::new(col, row, pc));
    }

    /// Remove all breakpoints.
    pub fn clear_breakpoints(&mut self) {
        self.breakpoints.clear();
    }

    /// Check if any breakpoint is hit.
    fn check_breakpoints(&self) -> Option<(u8, u8)> {
        for bp in &self.breakpoints {
            if !bp.enabled {
                continue;
            }

            if let Some(tile) = self.state.array.get(bp.col, bp.row) {
                if tile.core.pc == bp.pc {
                    return Some((bp.col, bp.row));
                }
            }
        }
        None
    }

    /// Execute one cycle across all cores.
    ///
    /// Returns true if any core is still running.
    pub fn step(&mut self) -> bool {
        if self.status == EngineStatus::Paused {
            return false;
        }

        // Check for breakpoints before stepping
        if let Some((col, row)) = self.check_breakpoints() {
            self.status = EngineStatus::Breakpoint { col, row };
            // Mark the executor as at breakpoint
            for exec in &mut self.executors {
                if exec.col == col && exec.row == row {
                    exec.status = CoreStatus::Breakpoint;
                }
            }
            return false;
        }

        let mut any_running = false;

        // Step each executor
        for exec in &mut self.executors {
            if let Some(tile) = self.state.array.get_mut(exec.col, exec.row) {
                if exec.step(tile) {
                    any_running = true;
                }
            }
        }

        self.total_cycles += 1;
        self.status = if any_running {
            EngineStatus::Running
        } else {
            EngineStatus::AllHalted
        };

        // Check max cycles
        if self.max_cycles > 0 && self.total_cycles >= self.max_cycles {
            self.status = EngineStatus::Paused;
            return false;
        }

        any_running
    }

    /// Run for up to `max_cycles` cycles.
    ///
    /// Stops early if all cores halt or a breakpoint is hit.
    /// Returns the number of cycles actually executed.
    pub fn run(&mut self, max_cycles: u64) -> u64 {
        let start = self.total_cycles;
        let limit = self.total_cycles + max_cycles;

        self.status = EngineStatus::Running;

        while self.total_cycles < limit {
            if !self.step() {
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

    /// Resume execution from paused/breakpoint state.
    pub fn resume(&mut self) {
        match self.status {
            EngineStatus::Paused | EngineStatus::Breakpoint { .. } => {
                self.status = EngineStatus::Running;
                // Clear breakpoint status on executors
                for exec in &mut self.executors {
                    if exec.status == CoreStatus::Breakpoint {
                        exec.status = CoreStatus::Running;
                    }
                }
            }
            _ => {}
        }
    }

    /// Get the executor for a specific tile.
    pub fn get_executor(&self, col: u8, row: u8) -> Option<&CoreExecutor> {
        self.executors.iter().find(|e| e.col == col && e.row == row)
    }

    /// Get mutable executor for a specific tile.
    pub fn get_executor_mut(&mut self, col: u8, row: u8) -> Option<&mut CoreExecutor> {
        self.executors.iter_mut().find(|e| e.col == col && e.row == row)
    }

    /// Count active cores.
    pub fn active_cores(&self) -> usize {
        self.executors.iter().filter(|e| e.is_active()).count()
    }

    /// Count enabled cores.
    pub fn enabled_cores(&self) -> usize {
        self.executors
            .iter()
            .filter(|e| {
                self.state
                    .array
                    .get(e.col, e.row)
                    .map(|t| t.core.enabled)
                    .unwrap_or(false)
            })
            .count()
    }

    /// Get total instructions executed across all cores.
    pub fn total_instructions(&self) -> u64 {
        self.executors.iter().map(|e| e.instructions).sum()
    }

    /// Print engine status.
    pub fn print_status(&self) {
        println!("Engine Status: {:?}", self.status);
        println!("Total Cycles: {}", self.total_cycles);
        println!("Total Instructions: {}", self.total_instructions());
        println!("Enabled Cores: {}", self.enabled_cores());
        println!("Active Cores: {}", self.active_cores());

        println!();
        println!("Core Status:");
        for exec in &self.executors {
            if let Some(tile) = self.state.array.get(exec.col, exec.row) {
                if tile.core.enabled || exec.is_active() {
                    print!(
                        "  ({},{}) PC=0x{:04X} {} cycles={} instr={}",
                        exec.col,
                        exec.row,
                        tile.core.pc,
                        exec.status_string(),
                        exec.cycles,
                        exec.instructions
                    );
                    if let Some(inst) = &exec.last_instruction {
                        print!(" last: {}", inst.disassemble());
                    }
                    println!();
                }
            }
        }
    }

    /// Get a status string for the engine.
    pub fn status_string(&self) -> &'static str {
        match self.status {
            EngineStatus::Idle => "Idle",
            EngineStatus::Running => "Running",
            EngineStatus::Paused => "Paused",
            EngineStatus::Breakpoint { .. } => "Breakpoint",
            EngineStatus::AllHalted => "Halted",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceState;

    #[test]
    fn test_engine_creation() {
        let engine = Engine::new_npu1();
        // 5 cols × 4 rows (rows 2-5) = 20 compute tiles
        assert_eq!(engine.executors.len(), 20);
    }

    #[test]
    fn test_engine_step_no_cores_enabled() {
        let mut engine = Engine::new_npu1();
        let running = engine.step();
        assert!(!running);
        assert_eq!(engine.status, EngineStatus::AllHalted);
    }

    #[test]
    fn test_engine_step_with_enabled_core() {
        let mut engine = Engine::new_npu1();

        // Enable core at (1,2) and write a NOP
        {
            let tile = engine.state.array.tile_mut(1, 2);
            tile.core.enabled = true;
            assert!(tile.write_program(0, &[0x00, 0x00, 0x00, 0x00]));
        }

        let running = engine.step();
        assert!(running);
        assert_eq!(engine.status, EngineStatus::Running);
        assert_eq!(engine.total_cycles, 1);
    }

    #[test]
    fn test_engine_run() {
        let mut engine = Engine::new_npu1();

        // Enable core and write some NOPs followed by halt (all zeros runs off end)
        {
            let tile = engine.state.array.tile_mut(1, 2);
            tile.core.enabled = true;
            for i in 0..10 {
                assert!(tile.write_program(i * 4, &[0x00, 0x00, 0x00, 0x00]));
            }
        }

        let cycles = engine.run(5);
        assert_eq!(cycles, 5);
        assert!(engine.total_cycles >= 5);
    }

    #[test]
    fn test_breakpoint() {
        let mut engine = Engine::new_npu1();

        // Enable core and write NOPs
        {
            let tile = engine.state.array.tile_mut(1, 2);
            tile.core.enabled = true;
            for i in 0..10 {
                assert!(tile.write_program(i * 4, &[0x00, 0x00, 0x00, 0x00]));
            }
        }

        // Add breakpoint at PC=8
        engine.add_breakpoint(1, 2, 8);

        engine.run(100);

        // Should have stopped at breakpoint
        assert!(matches!(
            engine.status,
            EngineStatus::Breakpoint { col: 1, row: 2 }
        ));
    }

    #[test]
    fn test_pause_resume() {
        let mut engine = Engine::new_npu1();
        engine.status = EngineStatus::Running;

        engine.pause();
        assert_eq!(engine.status, EngineStatus::Paused);

        engine.resume();
        assert_eq!(engine.status, EngineStatus::Running);
    }

    #[test]
    fn test_reset() {
        let mut engine = Engine::new_npu1();
        engine.total_cycles = 1000;
        engine.status = EngineStatus::Running;

        engine.reset();

        assert_eq!(engine.total_cycles, 0);
        assert_eq!(engine.status, EngineStatus::Idle);
    }
}
