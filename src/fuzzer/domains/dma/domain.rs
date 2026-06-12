//! DmaDomain: observation, buffer spec, localization, banking, and the
//! `impl Domain`. Mirrors scalar/domain.rs.
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::chain::{Dtype, DmaChain};
use super::gen::generate;
use super::lower::lower_chain;
use super::table::universe_keys;
use crate::fuzzer::core::domain::{Backend, Banked, Domain};
use crate::fuzzer::core::toolchain::{compile_dma_mlir, ToolPaths};
use crate::testing::npu_runner;
use crate::testing::test_cpp_parser::{BufferDef, BufferDir, BufferSpec, ElementType, InputPattern};
use crate::testing::xclbin_suite::{XclbinSuite, XclbinTest};

pub struct DmaObs {
    pub output: Vec<u8>,
}

fn elem_type(d: Dtype) -> ElementType {
    match d {
        Dtype::I32 => ElementType::I32,
        Dtype::I16 => ElementType::I16,
        Dtype::I8 => ElementType::I8,
    }
}

/// 2-buffer spec: in (group_id 3, Sequential{1,1}), out (group_id 4, Zeros).
/// The runtime_sequence arg order is (in, out); group_id = 3 + arg_index.
pub(crate) fn make_dma_buffer_spec(chain: &DmaChain) -> BufferSpec {
    let et = elem_type(chain.dtype);
    BufferSpec {
        buffers: vec![
            BufferDef {
                name: "buf_in".to_string(),
                group_id: 3,
                size_elements: chain.in_words(),
                element_type: et,
                direction: BufferDir::Input,
                input_pattern: InputPattern::Sequential { start: 1, step: 1 },
            },
            BufferDef {
                name: "buf_out".to_string(),
                group_id: 4,
                size_elements: chain.out_words(),
                element_type: et,
                direction: BufferDir::Output,
                input_pattern: InputPattern::Zeros,
            },
        ],
        multi_kernel: false,
    }
}

pub(crate) fn observe_impl(
    backend: Backend,
    xclbin: &Path,
    insts: &Path,
    chain: &DmaChain,
    max_cycles: u64,
) -> Result<DmaObs, String> {
    match backend {
        Backend::Interpreter => {
            let spec = make_dma_buffer_spec(chain);
            let test = XclbinTest::from_path(xclbin).with_buffer_spec(spec);
            let suite = XclbinSuite::new().with_max_cycles(max_cycles);
            let (outcome, raw_output, _trace) = suite.run_single_with_trace(&test);
            if !outcome.is_pass() {
                return Err(format!("emulator outcome not pass: {outcome:?}"));
            }
            let output = raw_output.ok_or_else(|| "Emulator produced no output".to_string())?;
            Ok(DmaObs { output })
        }
        Backend::Hardware => {
            if !npu_runner::npu_available() {
                return Err("NPU hardware not available".into());
            }
            let spec = make_dma_buffer_spec(chain);
            let test_name = format!("dmafuzz_seed_{}", chain.seed);
            match npu_runner::run_on_npu(&spec, &test_name, xclbin, insts, 30) {
                Ok(result) => Ok(DmaObs { output: result.output }),
                Err(e) => Err(format!("{e:?}")),
            }
        }
        Backend::Aiesim => Err("aiesim backend not wired for the DMA domain (Step 3)".into()),
    }
}

/// First transfer-region whose bytes differ, equal-split over `n` regions
/// (exact here because all chain output regions are equal-sized). Scalar-style.
pub(crate) fn first_divergent_region(emu: &[u8], hw: &[u8], n: usize) -> Option<usize> {
    if n == 0 {
        return if emu == hw { None } else { Some(0) };
    }
    let common = emu.len().min(hw.len());
    let region_bytes = (common / n).max(1);
    for r in 0..n {
        let start = r * region_bytes;
        let end = ((r + 1) * region_bytes).min(common);
        if start >= end {
            break;
        }
        if emu[start..end] != hw[start..end] {
            return Some(r);
        }
    }
    if emu.len() != hw.len() {
        return Some((common / region_bytes).min(n.saturating_sub(1)));
    }
    None
}

pub struct DmaDomain;

#[derive(Serialize, Deserialize)]
pub(crate) struct DmaChainRecord {
    pub(crate) chain: DmaChain,
    pub(crate) keys: Vec<String>,
}

pub(crate) fn bank_case(
    case_dir: &Path,
    chain: &DmaChain,
    keys: &[String],
    npu_output: Option<&[u8]>,
) -> Result<PathBuf, String> {
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    let bank_dir =
        PathBuf::from(home).join(format!("npu-work/experiments/phoenix-survival/dma/seed_{}", chain.seed));
    std::fs::create_dir_all(&bank_dir).map_err(|e| format!("create {}: {e}", bank_dir.display()))?;
    // best-effort copy of the generated MLIR for post-mortem
    std::fs::copy(case_dir.join("aie.mlir"), bank_dir.join("aie.mlir")).ok();
    let record = DmaChainRecord { chain: chain.clone(), keys: keys.to_vec() };
    let json = serde_json::to_string_pretty(&record).map_err(|e| format!("serialize: {e}"))?;
    std::fs::write(bank_dir.join("chain.json"), json).map_err(|e| format!("write chain.json: {e}"))?;
    if let Some(out) = npu_output {
        std::fs::write(bank_dir.join("npu_output.bin"), out).map_err(|e| format!("write npu_output: {e}"))?;
    }
    Ok(bank_dir)
}

impl Domain for DmaDomain {
    type Case = DmaChain;
    type Obs = DmaObs;

    fn name(&self) -> &str {
        "dma"
    }
    fn universe(&self) -> Vec<String> {
        universe_keys()
    }
    fn generate(&self, seed: u64, target: &str) -> DmaChain {
        generate(seed, target)
    }
    fn coverage_keys(&self, c: &DmaChain) -> Vec<String> {
        c.keys()
    }
    fn target_key(&self, c: &DmaChain) -> String {
        c.target_key.clone()
    }
    fn lower(&self, c: &DmaChain) -> String {
        lower_chain(c)
    }
    fn buffer_words(&self, c: &DmaChain) -> usize {
        c.out_words()
    }
    fn dtype(&self, c: &DmaChain) -> &str {
        c.dtype.template_dtype()
    }

    /// Override: write aie.mlir and run aiecc directly (no kernel object).
    fn compile(&self, tools: &ToolPaths, case_dir: &Path, c: &DmaChain) -> Result<(), String> {
        std::fs::write(case_dir.join("aie.mlir"), lower_chain(c))
            .map_err(|e| format!("write aie.mlir: {e}"))?;
        compile_dma_mlir(tools, case_dir)
    }

    fn observe(
        &self,
        backend: Backend,
        xclbin: &Path,
        insts: &Path,
        c: &DmaChain,
        max_cycles: u64,
    ) -> Result<DmaObs, String> {
        observe_impl(backend, xclbin, insts, c, max_cycles)
    }

    fn warnings(&self, obs: &DmaObs) -> Vec<String> {
        if !obs.output.is_empty() && obs.output.iter().all(|&b| b == 0) {
            vec!["vacuous output (all zero) -- degenerate transfer".into()]
        } else {
            Vec::new()
        }
    }

    fn compare(&self, emu: &DmaObs, reference: &DmaObs, keys: &[String]) -> Option<String> {
        let n = keys.len().max(1);
        first_divergent_region(&emu.output, &reference.output, n).map(|r| keys[r.min(n - 1)].clone())
    }

    fn bank(
        &self,
        case_dir: &Path,
        c: &DmaChain,
        reference: Option<&DmaObs>,
        _emu: Option<&DmaObs>,
    ) -> Result<PathBuf, String> {
        bank_case(case_dir, c, &c.keys(), reference.map(|o| o.output.as_slice()))
    }

    fn load_banked(&self, seed_dir: &Path) -> Result<Banked<DmaChain, DmaObs>, String> {
        let record: DmaChainRecord = std::fs::read_to_string(seed_dir.join("chain.json"))
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string()))?;
        let npu_output =
            std::fs::read(seed_dir.join("npu_output.bin")).map_err(|e| format!("npu_output.bin: {e}"))?;
        Ok(Banked::Replayable {
            case: record.chain,
            reference: DmaObs { output: npu_output },
            keys: record.keys,
        })
    }

    fn dump_divergent_observation(&self, case_dir: &Path, emu: &DmaObs) -> Result<(), String> {
        std::fs::write(case_dir.join("emu_output.bin"), &emu.output)
            .map_err(|e| format!("emu_output.bin write error: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn localizes_to_the_divergent_region() {
        // 3 equal regions of 16 i32 = 64 bytes each
        let mut emu = vec![7u8; 3 * 64];
        let hw = vec![7u8; 3 * 64];
        emu[70] = 9; // corrupt region 1 (bytes 64..128)
        assert_eq!(first_divergent_region(&emu, &hw, 3), Some(1));
    }

    #[test]
    fn equal_observations_match() {
        let buf = vec![3u8; 3 * 64];
        assert_eq!(first_divergent_region(&buf, &buf, 3), None);
    }

    #[test]
    fn length_mismatch_flags_a_region() {
        let emu = vec![1u8; 128];
        let hw = vec![1u8; 192];
        assert!(first_divergent_region(&emu, &hw, 3).is_some());
    }

    #[test]
    #[ignore = "requires toolchain; run with --ignored"]
    fn end_to_end_emu_run_produces_output() {
        use crate::fuzzer::core::domain::Domain;
        let tools = match crate::fuzzer::core::toolchain::ToolPaths::discover() {
            Ok(t) => t,
            Err(_) => return,
        };
        let dom = DmaDomain;
        let c = dom.generate(1, "transpose/memtile/mm2s/I32");
        let dir = std::env::temp_dir().join(format!("dma_e2e_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dom.compile(&tools, &dir, &c).expect("compile");
        let obs = dom
            .observe(Backend::Interpreter, &dir.join("aie.xclbin"), &dir.join("insts.bin"), &c, 2_000_000)
            .expect("emu run");
        assert!(!obs.output.is_empty());
        assert!(obs.output.iter().any(|&b| b != 0), "expected a non-zero reshuffle");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[ignore = "full-universe EMU smoke (81 aiecc compiles + 81 emu runs); run with --ignored"]
    fn universe_emu_smoke() {
        use crate::fuzzer::core::domain::Domain;
        let tools = match crate::fuzzer::core::toolchain::ToolPaths::discover() {
            Ok(t) => t,
            Err(_) => {
                eprintln!("SKIP: toolchain not discoverable");
                return;
            }
        };
        let dom = DmaDomain;
        let universe = dom.universe();
        let total = universe.len();
        let mut compile_fails: Vec<(String, String)> = Vec::new();
        let mut run_fails: Vec<(String, String)> = Vec::new();
        let mut ran = 0usize;
        for (i, key) in universe.iter().enumerate() {
            let c = dom.generate(i as u64 + 1, key);
            let dir = std::env::temp_dir().join(format!("dma_smoke_{i}_{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            match dom.compile(&tools, &dir, &c) {
                Ok(()) => {
                    match dom.observe(
                        Backend::Interpreter,
                        &dir.join("aie.xclbin"),
                        &dir.join("insts.bin"),
                        &c,
                        2_000_000,
                    ) {
                        Ok(obs) if !obs.output.is_empty() => ran += 1,
                        Ok(_) => run_fails.push((key.clone(), "empty output".into())),
                        Err(e) => run_fails.push((key.clone(), e)),
                    }
                }
                Err(e) => {
                    let tail: String = e
                        .lines()
                        .rev()
                        .take(3)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect::<Vec<_>>()
                        .join(" | ");
                    compile_fails.push((key.clone(), tail));
                }
            }
            std::fs::remove_dir_all(&dir).ok();
        }
        let compiled = total - compile_fails.len();
        eprintln!("=== DMA universe smoke: compiled {compiled}/{total}, ran {ran}/{total} ===");
        for (k, e) in &compile_fails {
            eprintln!("COMPILE FAIL {k}: {e}");
        }
        for (k, e) in &run_fails {
            eprintln!("RUN FAIL {k}: {e}");
        }
        assert!(compile_fails.is_empty(), "{} keys failed to compile", compile_fails.len());
        assert_eq!(ran, total, "{} keys failed to run on the emulator", total - ran);
    }
}
