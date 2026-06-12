//! The scalar fuzzer as a `core::Domain` tenant. Owns chain execution on
//! EMU/HW, the exact-byte per-region comparator, and durable bank/replay.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::fuzzer::core::domain::{Backend, Banked, Domain};
use crate::fuzzer::core::toolchain::TRACE_BUFFER_ELEMENTS;
use crate::fuzzer::domains::scalar::chain::{Dtype, ScalarChain};
use crate::fuzzer::domains::scalar::gen::generate;
use crate::fuzzer::domains::scalar::lower::lower_chain;
use crate::fuzzer::domains::scalar::table::universe_keys;
use crate::testing::test_cpp_parser::{BufferDef, BufferDir, BufferSpec, ElementType, InputPattern};
use crate::testing::xclbin_suite::{XclbinSuite, XclbinTest};

/// One execution's result: the full output buffer (N regions concatenated) and
/// an optional binary trace. Scalar has no executed-op recorder (vector's
/// `fuzz_recorder` tracks vector ops); folding/vacuity is detected from the
/// output being all-zero in `warnings` (added in a later task).
pub struct ScalarObs {
    pub output: Vec<u8>,
    pub trace: Option<Vec<u8>>,
}

/// The scalar fuzzer tenant. Zero-sized: all state is per-case.
pub struct ScalarDomain;

/// Buffer size in dtype elements: the output spans `stages * region_len`; the
/// input is sized the same (only the first `region_len` elements are read).
pub(crate) fn buffer_words(chain: &ScalarChain) -> usize {
    chain.out_elems()
}

/// Map a scalar dtype to the buffer-spec element type.
pub(crate) fn elem_type(dtype: Dtype) -> ElementType {
    match dtype {
        Dtype::I32 => ElementType::I32,
        Dtype::I16 => ElementType::I16,
        Dtype::I8 => ElementType::I8,
    }
}

/// Buffer spec for a scalar chain: Sequential input (1,2,3,...), zero scratch,
/// zero output, standard trace buffer. All data buffers are `out_elems` wide.
pub(crate) fn make_scalar_buffer_spec(chain: &ScalarChain) -> BufferSpec {
    let words = buffer_words(chain);
    let et = elem_type(chain.dtype);
    BufferSpec {
        buffers: vec![
            BufferDef {
                name: "buf_in".to_string(),
                group_id: 3,
                size_elements: words,
                element_type: et,
                direction: BufferDir::Input,
                input_pattern: InputPattern::Sequential { start: 1, step: 1 },
            },
            BufferDef {
                name: "buf_scratch".to_string(),
                group_id: 4,
                size_elements: words,
                element_type: et,
                direction: BufferDir::Input,
                input_pattern: InputPattern::Zeros,
            },
            BufferDef {
                name: "buf_out".to_string(),
                group_id: 5,
                size_elements: words,
                element_type: et,
                direction: BufferDir::Output,
                input_pattern: InputPattern::Zeros,
            },
            BufferDef {
                name: "buf_trace".to_string(),
                group_id: 6,
                size_elements: TRACE_BUFFER_ELEMENTS,
                element_type: ElementType::I32,
                direction: BufferDir::Output,
                input_pattern: InputPattern::Zeros,
            },
        ],
        multi_kernel: false,
    }
}

/// Execute a compiled chain on `backend`, returning a [`ScalarObs`].
pub(crate) fn observe_impl(
    backend: Backend,
    xclbin: &Path,
    insts: &Path,
    chain: &ScalarChain,
    max_cycles: u64,
) -> Result<ScalarObs, String> {
    match backend {
        Backend::Interpreter => {
            let spec = make_scalar_buffer_spec(chain);
            let test = XclbinTest::from_path(xclbin).with_buffer_spec(spec);
            let suite = XclbinSuite::new().with_max_cycles(max_cycles);
            let (outcome, raw_output, trace) = suite.run_single_with_trace(&test);
            if !outcome.is_pass() {
                return Err(format!("emulator outcome not pass: {outcome:?}"));
            }
            let output = raw_output.ok_or_else(|| "Emulator produced no output".to_string())?;
            Ok(ScalarObs { output, trace })
        }
        Backend::Hardware => {
            use crate::testing::npu_runner;
            if !npu_runner::npu_available() {
                return Err("NPU hardware not available".into());
            }
            let spec = make_scalar_buffer_spec(chain);
            let test_name = format!("scalarfuzz_seed_{}", chain.seed);
            match npu_runner::run_on_npu(&spec, &test_name, xclbin, insts, 30) {
                Ok(result) => Ok(ScalarObs {
                    output: result.output,
                    trace: result.extra_outputs.get("buf_trace").cloned(),
                }),
                Err(e) => Err(format!("{:?}", e)),
            }
        }
        Backend::Aiesim => Err("aiesim backend not wired for the scalar domain (Step 2)".into()),
    }
}

/// The region (localizable) keys: coverage keys that are not the case-level
/// loop-style key. Order-preserving, so `region_keys(keys)[k]` is stage k's key.
pub(crate) fn region_keys(keys: &[String]) -> Vec<String> {
    keys.iter().filter(|k| !k.starts_with("loop_")).cloned().collect()
}

/// First differing output region between two buffers split into `n` equal
/// regions, or `None` if equal. Exact-byte (scalar is integer; no tolerance).
/// A length mismatch counts as a divergence at the first region past the
/// common length.
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
    use crate::fuzzer::domains::scalar::gen::generate;
    use crate::testing::test_cpp_parser::{BufferDir, ElementType, InputPattern};

    #[test]
    fn buffer_spec_has_sequential_input_and_zero_out() {
        let c = generate(1, "add/I8");
        let spec = make_scalar_buffer_spec(&c);
        assert_eq!(spec.buffers.len(), 4); // in, scratch, out, trace
        let buf_in = &spec.buffers[0];
        assert_eq!(buf_in.direction, BufferDir::Input);
        assert_eq!(buf_in.element_type, ElementType::I8);
        assert_eq!(buf_in.input_pattern, InputPattern::Sequential { start: 1, step: 1 });
        assert_eq!(buf_in.size_elements, c.out_elems());
        assert_eq!(spec.buffers[2].direction, BufferDir::Output);
        assert_eq!(spec.buffers[2].input_pattern, InputPattern::Zeros);
    }

    #[test]
    fn buffer_words_is_out_elems() {
        let c = generate(2, "add/I32");
        assert_eq!(buffer_words(&c), c.out_elems());
    }

    #[test]
    fn element_type_maps_each_dtype() {
        use crate::fuzzer::domains::scalar::chain::Dtype;
        assert_eq!(elem_type(Dtype::I32), ElementType::I32);
        assert_eq!(elem_type(Dtype::I16), ElementType::I16);
        assert_eq!(elem_type(Dtype::I8), ElementType::I8);
    }

    #[test]
    fn region_keys_filter_drops_loop_keys() {
        let keys = vec!["add/I32".to_string(), "branch/I32".to_string(), "loop_hw/I32".to_string()];
        assert_eq!(region_keys(&keys), vec!["add/I32".to_string(), "branch/I32".to_string()]);
    }

    #[test]
    fn first_divergent_region_localizes_to_the_changed_region() {
        // 3 regions of 8 bytes; corrupt a byte in region 1.
        let n = 3;
        let emu = vec![0xAAu8; n * 8];
        let mut hw = emu.clone();
        hw[8 + 3] ^= 0x40;
        assert_eq!(first_divergent_region(&emu, &hw, n), Some(1));
    }

    #[test]
    fn equal_buffers_do_not_diverge() {
        let emu = vec![0x5Au8; 4 * 16];
        assert_eq!(first_divergent_region(&emu, &emu.clone(), 4), None);
    }

    #[test]
    fn length_mismatch_diverges() {
        let emu = vec![0u8; 3 * 8];
        let hw = vec![0u8; 2 * 8];
        assert!(first_divergent_region(&emu, &hw, 3).is_some());
    }
}
