//! DmaDomain: observation, buffer spec, localization (first half). Banking +
//! impl Domain land in the next task. Mirrors scalar/domain.rs.
use std::path::Path;

use super::chain::{Dtype, DmaChain};
use crate::fuzzer::core::domain::Backend;
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
#[allow(dead_code)] // consumed by impl Domain in the next task
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

#[allow(dead_code)] // consumed by impl Domain in the next task
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
#[allow(dead_code)] // consumed by impl Domain in the next task
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
}
