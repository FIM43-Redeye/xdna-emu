//! `EngineHost` -- loads an xclbin into an `InterpreterEngine`, finds a
//! companion NPU instruction stream, and steps both together with a
//! per-frame budget. The GUI (src/visual) drives this without touching the
//! interpreter directly.

use std::fs;
use std::path::Path;

use crate::interpreter::{EngineStatus, InterpreterEngine};
use crate::loading::{default_host_buffers, load_engine};
use crate::npu::{AdvanceResult, NpuExecutor, NpuInstructionStream};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RunState {
    Paused,
    Running,
}

pub struct EngineHost {
    pub engine: InterpreterEngine,
    executor: Option<NpuExecutor>,
    pub run_state: RunState,
}

/// Look for a companion instruction stream next to the xclbin. v1 uses the
/// simple convention (insts.bin / insts.elf in the same directory); zeroed
/// host memory is a valid default input, so a missing insts file just means
/// no control program runs.
fn find_companion_insts(xclbin_path: &Path) -> Option<NpuExecutor> {
    let dir = xclbin_path.parent()?;
    let data = ["insts.bin", "insts.elf"].iter().find_map(|n| fs::read(dir.join(n)).ok())?;
    let stream = NpuInstructionStream::parse(&data).ok()?;
    let mut ex = NpuExecutor::new();
    ex.set_host_buffers(default_host_buffers());
    ex.load(&stream);
    Some(ex)
}

pub fn load(xclbin_path: &Path) -> Result<EngineHost, String> {
    let engine = load_engine(xclbin_path)?;
    let executor = find_companion_insts(xclbin_path);
    Ok(EngineHost { engine, executor, run_state: RunState::Paused })
}

impl EngineHost {
    pub fn total_cycles(&self) -> u64 {
        self.engine.total_cycles()
    }

    pub fn status(&self) -> EngineStatus {
        self.engine.status()
    }

    /// One executor-interleave + one `engine.step()`. Mirrors
    /// `xclbin_suite::run_engine`'s per-cycle order: advance the NPU
    /// instruction stream (DMA config/triggers) before stepping cores, so a
    /// full system step sees this cycle's DMA state.
    ///
    /// Returns `false` if a fatal executor error paused the run instead of
    /// stepping the engine. The error condition doesn't clear itself, so
    /// `step_bounded` uses this to stop instead of re-hitting the same
    /// failing instruction for the rest of its budget.
    pub fn step_one(&mut self) -> bool {
        if let Some(ex) = self.executor.as_mut() {
            let (device, host_mem) = self.engine.device_and_host_memory();
            if let AdvanceResult::Error(msg) = ex.try_advance(device, host_mem) {
                log::error!("NPU executor fatal: {}", msg);
                self.run_state = RunState::Paused;
                return false;
            }
        }
        self.engine.step();
        true
    }

    /// Up to `budget` steps; stops early on a terminal engine status or a
    /// fatal executor error (see `step_one`).
    pub fn step_bounded(&mut self, budget: u32) -> EngineStatus {
        for _ in 0..budget {
            if !self.step_one() {
                break;
            }
            match self.engine.status() {
                EngineStatus::Halted | EngineStatus::Stalled | EngineStatus::Error => break,
                _ => {}
            }
        }
        self.engine.status()
    }

    pub fn reset(&mut self) {
        self.engine.reset();
        self.run_state = RunState::Paused;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin")
    }

    #[test]
    fn step_bounded_advances_cycles() {
        let path = fixture();
        if !path.exists() {
            eprintln!("SKIP step_bounded_advances_cycles: fixture not built at {}", path.display());
            return;
        }
        let mut host = load(&path).expect("load");
        let before = host.total_cycles();
        host.step_bounded(50);
        assert!(host.total_cycles() > before, "stepping must advance the cycle count");
    }

    #[test]
    fn reset_returns_to_zero_cycles() {
        let path = fixture();
        if !path.exists() {
            eprintln!("SKIP reset_returns_to_zero_cycles: fixture not built at {}", path.display());
            return;
        }
        let mut host = load(&path).expect("load");
        host.step_bounded(50);
        host.reset();
        assert_eq!(host.total_cycles(), 0);
    }
}
