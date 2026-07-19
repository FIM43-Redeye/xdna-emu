//! `EngineHost` -- loads an xclbin into an `InterpreterEngine`, finds a
//! companion NPU instruction stream, and steps both together with a
//! per-frame budget. The GUI (src/visual) drives this without touching the
//! interpreter directly.

use std::fs;
use std::path::{Path, PathBuf};

use crate::interpreter::{EngineStatus, InterpreterEngine};
use crate::loading::{default_host_buffers, load_engine};
use crate::npu::{AdvanceResult, NpuExecutor, NpuInstructionStream};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RunState {
    Paused,
    Running,
}

/// Where a design's control program came from, or why there isn't one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlProgram {
    Loaded(PathBuf),
    Missing,
    ParseFailed { path: PathBuf, error: String },
}

impl ControlProgram {
    /// Human-readable reason this design will not move data, if it will not.
    pub fn warning(&self) -> Option<String> {
        match self {
            ControlProgram::Loaded(_) => None,
            ControlProgram::Missing => Some(
                "No control program found next to this xclbin (looked for insts.bin, insts.elf, \
                 and a variant-matched insts*). DMAs are never triggered, so nothing will move."
                    .to_string(),
            ),
            ControlProgram::ParseFailed { path, error } => Some(format!(
                "Control program {} failed to parse ({error}). DMAs are never triggered, \
                 so nothing will move.",
                path.display()
            )),
        }
    }
}

/// Locate a companion instruction stream next to the xclbin.
///
/// Two conventions exist in the mlir-aie tree:
///   - the common one: a plain `insts.bin` / `insts.elf` (229 of 232 kernels);
///   - a variant-suffixed one, where one directory holds several designs --
///     e.g. `aie2_plain.xclbin` beside `insts2_plain.txt`. Those files are
///     binary despite the `.txt` extension: byte-identical framing to
///     `insts.bin`, so they need no separate parser, only to be found.
///
/// The plain names are tried first so the common case cannot regress. The
/// variant key is the xclbin stem's suffix after its first `_` ("plain"), and
/// a candidate must be an `insts*` file whose own stem ends with that key --
/// which is what keeps `aie2_cascade.xclbin` off `insts2_plain.txt`.
fn find_insts_file(xclbin_path: &Path) -> Option<PathBuf> {
    let dir = xclbin_path.parent()?;

    for name in ["insts.bin", "insts.elf"] {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    let variant = xclbin_path.file_stem()?.to_str()?.split_once('_').map(|(_, v)| v)?;
    let mut matches: Vec<PathBuf> = fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("insts"))
                && p.file_stem().and_then(|s| s.to_str()).is_some_and(|s| s.ends_with(variant))
        })
        .collect();
    matches.sort();
    matches.into_iter().next()
}

pub struct EngineHost {
    pub engine: InterpreterEngine,
    /// How the control program was resolved -- surfaced in the GUI so a design
    /// that cannot move data says so instead of looking deadlocked.
    pub control_program: ControlProgram,
    executor: Option<NpuExecutor>,
    pub run_state: RunState,
    /// Source xclbin, kept so `reset()` can reload the whole design from
    /// scratch instead of trying to un-apply engine-side state piecemeal.
    xclbin_path: PathBuf,
}

/// Look for a companion instruction stream next to the xclbin. v1 uses the
/// simple convention (insts.bin / insts.elf in the same directory); zeroed
/// host memory is a valid default input, so a missing insts file just means
/// no control program runs.
fn find_companion_insts(xclbin_path: &Path) -> (Option<NpuExecutor>, ControlProgram) {
    let Some(path) = find_insts_file(xclbin_path) else {
        return (None, ControlProgram::Missing);
    };
    let data = match fs::read(&path) {
        Ok(d) => d,
        Err(e) => return (None, ControlProgram::ParseFailed { path, error: e.to_string() }),
    };
    let stream = match NpuInstructionStream::parse(&data) {
        Ok(s) => s,
        Err(e) => return (None, ControlProgram::ParseFailed { path, error: format!("{e:?}") }),
    };
    let mut ex = NpuExecutor::new();
    ex.set_host_buffers(default_host_buffers());
    ex.load(&stream);
    (Some(ex), ControlProgram::Loaded(path))
}

pub fn load(xclbin_path: &Path) -> Result<EngineHost, String> {
    let engine = load_engine(xclbin_path)?;
    let (executor, control_program) = find_companion_insts(xclbin_path);
    if let Some(w) = control_program.warning() {
        log::warn!("{} : {w}", xclbin_path.display());
    }
    Ok(EngineHost {
        engine,
        control_program,
        executor,
        run_state: RunState::Paused,
        xclbin_path: xclbin_path.to_path_buf(),
    })
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

    /// Reload the whole design from `xclbin_path` rather than trying to
    /// un-apply engine-side state piecemeal: `InterpreterEngine::reset()`
    /// alone rewinds cycle bookkeeping but never re-derives core-enabled
    /// state from `DeviceState` (that only happens via
    /// `sync_cores_from_device()`, called once at load time), so a bare
    /// engine reset left every core permanently disabled. Reloading reuses
    /// `load()` wholesale, which reapplies the CDO, reloads ELFs, and
    /// re-syncs cores, so the design actually runs again after reset.
    pub fn reset(&mut self) {
        let path = self.xclbin_path.clone();
        match load(&path) {
            Ok(fresh) => *self = fresh,
            Err(e) => {
                log::error!(
                    "EngineHost::reset: reload of {} failed ({e}), falling back to partial reset",
                    path.display()
                );
                self.engine.reset();
                self.run_state = RunState::Paused;
            }
        }
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

    /// Directory holding three designs that share one folder and use the
    /// variant-suffixed convention (`aie2_plain.xclbin` / `insts2_plain.txt`).
    fn variant_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../mlir-aie/build/test/npu-xrt/matrix_multiplication_using_cascade/chess")
    }

    fn total_dma_bytes(host: &EngineHost) -> u64 {
        let array = &host.engine.device().array;
        let mut bytes = 0;
        for col in 0..array.cols() {
            for row in 0..array.rows() {
                if let Some(eng) = array.dma_engine(col, row) {
                    for ch in 0..eng.channel_count() as u8 {
                        if let Some(st) = eng.channel_stats(ch) {
                            bytes += st.bytes_transferred;
                        }
                    }
                }
            }
        }
        bytes
    }

    #[test]
    fn finds_the_conventional_insts_bin() {
        let path = fixture();
        if !path.exists() {
            eprintln!("SKIP finds_the_conventional_insts_bin: fixture not built");
            return;
        }
        let found = find_insts_file(&path).expect("insts.bin next to the xclbin must be found");
        assert_eq!(found.file_name().unwrap(), "insts.bin");
    }

    #[test]
    fn matches_the_insts_variant_to_the_xclbin_variant() {
        let dir = variant_dir();
        if !dir.join("aie2_plain.xclbin").exists() {
            eprintln!("SKIP matches_the_insts_variant_to_the_xclbin_variant: fixture not built");
            return;
        }
        // Each xclbin must resolve to ITS OWN insts file, not merely any of the
        // three sitting in the directory.
        for variant in ["plain", "cascade", "buffer"] {
            let xclbin = dir.join(format!("aie2_{variant}.xclbin"));
            let found = find_insts_file(&xclbin)
                .unwrap_or_else(|| panic!("no control program resolved for variant {variant}"));
            assert_eq!(
                found.file_name().unwrap(),
                format!("insts2_{variant}.txt").as_str(),
                "variant {variant} resolved to the wrong control program"
            );
        }
    }

    #[test]
    fn reports_missing_when_no_control_program_exists() {
        let dir = std::env::temp_dir().join("xdna-emu-test-no-insts");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        let xclbin = dir.join("aie.xclbin");
        std::fs::write(&xclbin, b"not a real xclbin").expect("write");

        assert_eq!(find_insts_file(&xclbin), None);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_control_program_produces_a_warning_and_loaded_does_not() {
        assert!(ControlProgram::Missing.warning().is_some());
        assert!(ControlProgram::Loaded(PathBuf::from("insts.bin")).warning().is_none());
        assert!(ControlProgram::ParseFailed { path: PathBuf::from("insts.bin"), error: "bad".into() }
            .warning()
            .is_some());
    }

    #[test]
    fn variant_design_loads_its_control_program_and_moves_data() {
        let xclbin = variant_dir().join("aie2_plain.xclbin");
        if !xclbin.exists() {
            eprintln!("SKIP variant_design_loads_its_control_program_and_moves_data: not built");
            return;
        }
        let mut host = load(&xclbin).expect("load");
        assert!(
            matches!(host.control_program, ControlProgram::Loaded(_)),
            "variant design must resolve a control program, got {:?}",
            host.control_program
        );
        host.step_bounded(20_000);
        assert!(
            total_dma_bytes(&host) > 0,
            "with its control program loaded the design must actually move data"
        );
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

    #[test]
    fn reset_then_step_reruns_the_design() {
        let path = fixture();
        if !path.exists() {
            eprintln!("SKIP reset_then_step_reruns_the_design: fixture not built at {}", path.display());
            return;
        }
        let mut host = load(&path).expect("load");
        host.step_bounded(50);
        assert!(host.total_cycles() > 0);
        host.reset();
        assert_eq!(host.total_cycles(), 0);
        // After reset the SAME design must run again: stepping advances cycles from 0.
        host.step_bounded(50);
        assert!(host.total_cycles() > 0, "design must re-execute after reset, not sit dead");
    }
}
