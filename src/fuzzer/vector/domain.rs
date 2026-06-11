//! The vector fuzzer as a `core::Domain` tenant. Owns the vector-specific half
//! of the engine: chain generation, lowering, execution on EMU/HW, the
//! type-aware NaN-tolerant slice comparator, and durable bank/replay.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::fuzzer::core::domain::{Backend, Banked, Domain};
use crate::fuzzer::core::toolchain::TRACE_BUFFER_ELEMENTS;
use crate::fuzzer::vector::chain::{Chain, Stage};
use crate::fuzzer::vector::gen::generate;
use crate::fuzzer::vector::lower::lower_chain;
use crate::fuzzer::vector::table::{universe_keys, VecType};
#[cfg(test)]
use crate::fuzzer::vector::table::table;
use crate::interpreter::execute::fuzz_recorder;
use crate::testing::test_cpp_parser::{BufferDef, BufferDir, BufferSpec, ElementType, InputPattern};
use crate::testing::xclbin_suite::{XclbinSuite, XclbinTest};

/// One execution's result: output bytes, optional trace, executed coverage keys.
pub struct VecObs {
    pub output: Vec<u8>,
    pub trace: Option<Vec<u8>>,
    pub executed: Vec<String>,
}

/// The vector fuzzer tenant. Zero-sized: all state is per-case.
pub struct VectorDomain;

/// Serialized form of a banked chain: enough to replay without the live table.
/// The banked `pool` (input bytes) and `keys` make replay self-contained -- it
/// reconstructs and runs the banked xclbin even after the op table evolves and
/// shifts `entry_idx`. `table_version` stamps which table the bank was cut under
/// so replay can tell a same-table regeneration apart from a reconstruction.
#[derive(Serialize, Deserialize)]
struct ChainRecord {
    seed: u64,
    target_key: String,
    keys: Vec<String>,
    stages: Vec<StageRecord>,
    /// Input pool bytes the kernel loaded operands from. Defaulted empty for
    /// legacy banks cut before this field (those fall back to regeneration).
    #[serde(default)]
    pool: Vec<u8>,
    /// Hash of the coverage-key universe the bank was cut under (0 = legacy).
    #[serde(default)]
    table_version: u64,
}

#[derive(Serialize, Deserialize)]
struct StageRecord {
    entry_idx: usize,
    mode: u8,
    second_pool_slot: Option<usize>,
}

impl ChainRecord {
    fn from_chain(chain: &Chain) -> Self {
        Self {
            seed: chain.seed,
            target_key: chain.target_key.clone(),
            keys: chain.keys(),
            stages: chain
                .stages
                .iter()
                .map(|s| StageRecord {
                    entry_idx: s.entry_idx,
                    mode: s.mode,
                    second_pool_slot: s.second_pool_slot,
                })
                .collect(),
            pool: chain.pool.clone(),
            table_version: current_table_version(),
        }
    }

    /// Reconstruct the runnable [`Chain`] from banked artifacts alone -- no live
    /// table lookup, so it is correct even when the table has shifted `entry_idx`.
    /// Execution (`run_emulator_vec`/`run_npu_vec`) uses only `pool` and the stage
    /// count, both preserved here; localization uses the banked `keys`.
    fn to_chain(&self) -> Chain {
        Chain {
            seed: self.seed,
            target_key: self.target_key.clone(),
            stages: self
                .stages
                .iter()
                .map(|s| Stage { entry_idx: s.entry_idx, mode: s.mode, second_pool_slot: s.second_pool_slot })
                .collect(),
            pool: self.pool.clone(),
        }
    }
}

/// Stable FNV-1a hash of the coverage-key universe -- the bank's table-version
/// stamp. Deterministic across runs and independent of std hasher internals, so
/// a bank cut today still compares equal after a rebuild of the same table.
fn current_table_version() -> u64 {
    let joined = universe_keys().join("\n");
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in joined.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Per-slice result types parsed from banked coverage keys (`name/Type/mMode`).
/// Table-independent, so replay applies the right per-slice comparator tolerance
/// even when `entry_idx` no longer resolves against the current table. Used by
/// `compare` so both campaign and replay remain table-independent.
fn out_types_from_keys(keys: &[String]) -> Vec<VecType> {
    keys.iter()
        .filter_map(|k| k.split('/').nth(1).and_then(VecType::from_debug))
        .collect()
}

/// Buffer size in i32 words: pool and output both live in `--size`-word
/// buffers, so take the max of the two and let the shorter side zero-pad.
fn buffer_words(chain: &Chain) -> usize {
    (chain.pool_slots() * 16).max(chain.out_bytes() / 4)
}

/// Per-stage result types for a chain, in output-slice order. Selects the
/// per-slice tolerance the comparator applies (bf16 NaN payload is don't-care).
#[cfg(test)]
fn chain_out_types(chain: &Chain) -> Vec<VecType> {
    let t = table();
    chain.stages.iter().map(|s| t[s.entry_idx].out_type).collect()
}

/// True when a bf16 bit pattern is a NaN: exponent all-ones, mantissa nonzero.
fn bf16_is_nan(x: u16) -> bool {
    (x & 0x7F80) == 0x7F80 && (x & 0x007F) != 0
}

/// Equality for one 64-byte output slice, with type-aware tolerance.
///
/// For bf16 results the NaN *payload* mantissa bits are functionally dead and
/// silicon produces two residual-state-dependent values for them (the #115
/// datapath vs. canonical regimes -- same binary, same lane, different session).
/// Gating differential credit on those bits tests residual hardware state, not
/// emulator correctness, so two NaNs with matching sign compare equal regardless
/// of payload. Everything else compares exactly: Inf-vs-Inf (sign+exp),
/// Inf-vs-NaN, NaN-vs-finite, sign flips, and all integer types still register
/// as real divergences.
fn slice_equal(a: &[u8], b: &[u8], vt: Option<VecType>) -> bool {
    if a == b {
        return true;
    }
    if vt != Some(VecType::Bf16x32) || a.len() < 64 || b.len() < 64 {
        return false;
    }
    for lane in 0..32 {
        let av = u16::from_le_bytes([a[lane * 2], a[lane * 2 + 1]]);
        let bv = u16::from_le_bytes([b[lane * 2], b[lane * 2 + 1]]);
        if av == bv {
            continue;
        }
        // Tolerate only a dead NaN payload: both NaN, same sign.
        if !(bf16_is_nan(av) && bf16_is_nan(bv) && (av >> 15) == (bv >> 15)) {
            return false;
        }
    }
    true
}

/// First differing 64-byte slice index between two buffers, or None if equal.
/// `out_types[i]` is stage i's result type and selects per-slice tolerance (see
/// [`slice_equal`]). Slices past `out_types` (zero padding) compare exactly. A
/// length mismatch beyond the common prefix counts as a divergence at the first
/// slice past the common length.
fn first_divergent_slice(a: &[u8], b: &[u8], out_types: &[VecType]) -> Option<usize> {
    let common = a.len().min(b.len());
    for (i, (sa, sb)) in a[..common].chunks(64).zip(b[..common].chunks(64)).enumerate() {
        if !slice_equal(sa, sb, out_types.get(i).copied()) {
            return Some(i);
        }
    }
    if a.len() != b.len() {
        return Some(common / 64);
    }
    None
}

/// Map a divergent slice index to a coverage key. Slices past the last stage
/// (zero padding) clamp to the final stage -- a diff there still means the kernel
/// wrote where it should not have, attributed to the chain. Takes keys directly
/// so replay can localize against banked keys without the live table.
fn slice_to_key(keys: &[String], slice: usize) -> String {
    let idx = slice.min(keys.len().saturating_sub(1));
    keys[idx].clone()
}

/// Buffer spec for a vector chain: pool bytes in, zero-filled out, both sized
/// to the common `--size` word count; standard scratch + trace buffers.
fn make_vec_buffer_spec(chain: &Chain, words: usize) -> BufferSpec {
    BufferSpec {
        buffers: vec![
            BufferDef {
                name: "buf_in".to_string(),
                group_id: 3,
                size_elements: words,
                element_type: ElementType::I32,
                direction: BufferDir::Input,
                input_pattern: InputPattern::Bytes(chain.pool.clone()),
            },
            BufferDef {
                name: "buf_scratch".to_string(),
                group_id: 4,
                size_elements: words,
                element_type: ElementType::I32,
                direction: BufferDir::Input,
                input_pattern: InputPattern::Zeros,
            },
            BufferDef {
                name: "buf_out".to_string(),
                group_id: 5,
                size_elements: words,
                element_type: ElementType::I32,
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

/// Bank a divergent/crashed case for post-mortem and replay.
fn bank_case(
    case_dir: &Path,
    chain: &Chain,
    npu_output: Option<&[u8]>,
    npu_trace: Option<&[u8]>,
    executed: &[String],
) -> Result<PathBuf, String> {
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    let bank_dir =
        PathBuf::from(home).join(format!("npu-work/experiments/phoenix-survival/vector/seed_{}", chain.seed));
    std::fs::create_dir_all(&bank_dir).map_err(|e| format!("create {}: {e}", bank_dir.display()))?;

    std::fs::copy(case_dir.join("fuzz_kernel.cc"), bank_dir.join("fuzz_kernel.cc"))
        .map_err(|e| format!("copy fuzz_kernel.cc: {e}"))?;

    let record = ChainRecord::from_chain(chain);
    let json = serde_json::to_string_pretty(&record).map_err(|e| format!("serialize chain: {e}"))?;
    std::fs::write(bank_dir.join("chain.json"), json).map_err(|e| format!("write chain.json: {e}"))?;

    if let Some(out) = npu_output {
        std::fs::write(bank_dir.join("npu_output.bin"), out).map_err(|e| format!("write npu_output: {e}"))?;
    }
    if let Some(t) = npu_trace {
        std::fs::write(bank_dir.join("npu_trace.bin"), t).map_err(|e| format!("write npu_trace: {e}"))?;
    }
    let executed_json =
        serde_json::to_string_pretty(executed).map_err(|e| format!("serialize executed: {e}"))?;
    std::fs::write(bank_dir.join("executed.json"), executed_json)
        .map_err(|e| format!("write executed.json: {e}"))?;
    Ok(bank_dir)
}

impl Domain for VectorDomain {
    type Case = Chain;
    type Obs = VecObs;

    fn name(&self) -> &str {
        "vector"
    }
    fn universe(&self) -> Vec<String> {
        universe_keys()
    }
    fn generate(&self, seed: u64, target: &str) -> Chain {
        generate(seed, target)
    }
    fn coverage_keys(&self, c: &Chain) -> Vec<String> {
        c.keys()
    }
    fn target_key(&self, c: &Chain) -> String {
        c.target_key.clone()
    }
    fn lower(&self, c: &Chain) -> String {
        lower_chain(c)
    }
    fn buffer_words(&self, c: &Chain) -> usize {
        buffer_words(c)
    }
    fn dtype(&self) -> &str {
        "i32"
    }

    fn observe(
        &self,
        backend: Backend,
        xclbin: &Path,
        insts: &Path,
        c: &Chain,
        max_cycles: u64,
    ) -> Result<VecObs, String> {
        match backend {
            Backend::Interpreter => {
                let spec = make_vec_buffer_spec(c, buffer_words(c));
                let test = XclbinTest::from_path(xclbin).with_buffer_spec(spec);
                let suite = XclbinSuite::new().with_max_cycles(max_cycles);
                fuzz_recorder::arm();
                let (outcome, raw_output, trace) = suite.run_single_with_trace(&test);
                let executed = fuzz_recorder::take().unwrap_or_default();
                // A non-pass outcome means the output buffer is stale zeros, not
                // computed data; comparing it would mis-attribute the failure to
                // vector compute.
                if !outcome.is_pass() {
                    return Err(format!("emulator outcome not pass: {outcome:?}"));
                }
                let output = raw_output.ok_or_else(|| "Emulator produced no output".to_string())?;
                Ok(VecObs { output, trace, executed })
            }
            Backend::Hardware => {
                use crate::testing::npu_runner;
                if !npu_runner::npu_available() {
                    return Err("NPU hardware not available".into());
                }
                let spec = make_vec_buffer_spec(c, buffer_words(c));
                let test_name = format!("vecfuzz_seed_{}", c.seed);
                match npu_runner::run_on_npu(&spec, &test_name, xclbin, insts, 30) {
                    Ok(result) => Ok(VecObs {
                        output: result.output,
                        trace: result.extra_outputs.get("buf_trace").cloned(),
                        executed: Vec::new(),
                    }),
                    Err(e) => Err(format!("{:?}", e)),
                }
            }
            Backend::Aiesim => Err("aiesim backend not wired for the vector domain (Step 1)".into()),
        }
    }

    fn warnings(&self, obs: &VecObs) -> Vec<String> {
        if obs.executed.is_empty() {
            vec!["no vector ops executed (chain folded by compiler)".into()]
        } else {
            Vec::new()
        }
    }

    fn compare(&self, emu: &VecObs, reference: &VecObs, keys: &[String]) -> Option<String> {
        first_divergent_slice(&emu.output, &reference.output, &out_types_from_keys(keys))
            .map(|slice| slice_to_key(keys, slice))
    }

    fn bank(
        &self,
        case_dir: &Path,
        c: &Chain,
        reference: Option<&VecObs>,
        emu_obs: Option<&VecObs>,
    ) -> Result<PathBuf, String> {
        let npu_output = reference.map(|o| o.output.as_slice());
        let npu_trace = reference.and_then(|o| o.trace.as_deref());
        let executed = emu_obs.map(|o| o.executed.as_slice()).unwrap_or(&[]);
        bank_case(case_dir, c, npu_output, npu_trace, executed)
    }

    fn load_banked(&self, seed_dir: &Path) -> Result<Banked<Chain, VecObs>, String> {
        let record: ChainRecord = std::fs::read_to_string(seed_dir.join("chain.json"))
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string()))?;
        let chain = if !record.pool.is_empty() {
            record.to_chain()
        } else {
            let c = generate(record.seed, &record.target_key);
            if c.keys() != record.keys {
                return Ok(Banked::Skip(
                    "legacy bank under a changed table (no pool) -- re-bank to replay".into(),
                ));
            }
            c
        };
        let npu_output =
            std::fs::read(seed_dir.join("npu_output.bin")).map_err(|e| format!("npu_output.bin: {e}"))?;
        Ok(Banked::Replayable {
            case: chain,
            reference: VecObs { output: npu_output, trace: None, executed: Vec::new() },
            keys: record.keys,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build synthetic 3-stage outputs differing only in slice 1 and verify
    /// the divergence localizes to stage 1's coverage key.
    #[test]
    fn slice_localization_maps_to_stage_key() {
        let chain = generate(1, "add/I32x16/m0");
        assert!(chain.stages.len() >= 3);
        let n = chain.out_bytes();
        let emu = vec![0xAAu8; n];
        let mut npu = emu.clone();
        npu[64 + 7] ^= 0x40; // corrupt one byte inside slice 1
        let slice = first_divergent_slice(&emu, &npu, &chain_out_types(&chain)).expect("must diverge");
        assert_eq!(slice, 1);
        assert_eq!(slice_to_key(&chain.keys(), slice), chain.keys()[1]);
    }

    #[test]
    fn equal_buffers_with_zero_padding_do_not_diverge() {
        // out_bytes shorter than buffer: both sides zero-padded to buffer size.
        let mut emu = vec![0u8; 4 * 64];
        emu[..128].fill(0x5A); // two written stages
        let npu = emu.clone();
        assert_eq!(first_divergent_slice(&emu, &npu, &[]), None);
    }

    #[test]
    fn length_mismatch_diverges_at_first_extra_slice() {
        let emu = vec![0u8; 3 * 64];
        let npu = vec![0u8; 2 * 64];
        assert_eq!(first_divergent_slice(&emu, &npu, &[]), Some(2));
    }

    /// One 64-byte bf16 slice (32 lanes), one lane set on each side.
    fn bf16_slice(lane: usize, value: u16) -> Vec<u8> {
        let mut s = vec![0u8; 64];
        let b = value.to_le_bytes();
        s[lane * 2] = b[0];
        s[lane * 2 + 1] = b[1];
        s
    }

    #[test]
    fn bf16_nan_payload_is_tolerated_when_sign_matches() {
        // Datapath regime 0xFF8C vs canonical regime 0xFF81: same sign, both NaN.
        let emu = bf16_slice(29, 0xFF8C);
        let npu = bf16_slice(29, 0xFF81);
        let types = [VecType::Bf16x32];
        assert_eq!(first_divergent_slice(&emu, &npu, &types), None);
    }

    #[test]
    fn bf16_opposite_sign_nan_still_diverges() {
        // Same payload, opposite sign -- a real sign divergence, not dead bits.
        let emu = bf16_slice(29, 0xFF8C);
        let npu = bf16_slice(29, 0x7F8C);
        let types = [VecType::Bf16x32];
        assert_eq!(first_divergent_slice(&emu, &npu, &types), Some(0));
    }

    #[test]
    fn bf16_inf_vs_nan_still_diverges() {
        // +Inf (0x7F80, mantissa 0) vs a NaN -- not both NaN, must flag.
        let emu = bf16_slice(5, 0x7F80);
        let npu = bf16_slice(5, 0x7F8C);
        let types = [VecType::Bf16x32];
        assert_eq!(first_divergent_slice(&emu, &npu, &types), Some(0));
    }

    #[test]
    fn bf16_nan_vs_finite_still_diverges() {
        let emu = bf16_slice(5, 0x7F8C); // NaN
        let npu = bf16_slice(5, 0x4048); // finite ~3.125
        let types = [VecType::Bf16x32];
        assert_eq!(first_divergent_slice(&emu, &npu, &types), Some(0));
    }

    #[test]
    fn bf16_tolerance_does_not_leak_to_int_slices() {
        // The same byte pattern, typed as int, must compare exactly.
        let emu = bf16_slice(29, 0xFF8C);
        let npu = bf16_slice(29, 0xFF81);
        let types = [VecType::I16x32];
        assert_eq!(first_divergent_slice(&emu, &npu, &types), Some(0));
    }

    #[test]
    fn slice_past_last_stage_clamps_to_final_key() {
        let chain = generate(2, "add/I32x16/m0");
        let last = chain.keys().len() - 1;
        assert_eq!(slice_to_key(&chain.keys(), last + 5), chain.keys()[last]);
    }

    #[test]
    fn buffer_words_covers_pool_and_output() {
        let chain = generate(3, "add/I32x16/m0");
        let words = buffer_words(&chain);
        assert!(words * 4 >= chain.pool.len());
        assert!(words * 4 >= chain.out_bytes());
    }

    #[test]
    fn buffer_spec_embeds_pool_bytes_and_zero_out() {
        let chain = generate(4, "add/I32x16/m0");
        let words = buffer_words(&chain);
        let spec = make_vec_buffer_spec(&chain, words);
        assert_eq!(spec.buffers.len(), 4);
        let buf_in = &spec.buffers[0];
        assert_eq!(buf_in.size_elements, words);
        assert_eq!(buf_in.input_pattern, InputPattern::Bytes(chain.pool.clone()));
        assert_eq!(spec.buffers[2].input_pattern, InputPattern::Zeros);
    }

    #[test]
    fn chain_record_round_trip_matches_regeneration() {
        let chain = generate(5, "add/I32x16/m0");
        let record = ChainRecord::from_chain(&chain);
        let json = serde_json::to_string(&record).unwrap();
        let loaded: ChainRecord = serde_json::from_str(&json).unwrap();
        let regen = generate(loaded.seed, &loaded.target_key);
        assert_eq!(regen.keys(), loaded.keys);
        assert_eq!(regen.stages.len(), loaded.stages.len());
    }

    #[test]
    fn durable_bank_reconstructs_chain_without_the_table() {
        // A banked record reconstructs an identical runnable chain via to_chain
        // -- pool and stage structure preserved, no generate()/table involved.
        let chain = generate(6, "add/Bf16x32/m0");
        let record = ChainRecord::from_chain(&chain);
        assert!(!record.pool.is_empty(), "pool is banked");
        assert_eq!(record.table_version, current_table_version());

        let rebuilt = record.to_chain();
        assert_eq!(rebuilt.seed, chain.seed);
        assert_eq!(rebuilt.pool, chain.pool);
        assert_eq!(rebuilt.stages, chain.stages);
        // Execution inputs are reconstructed identically.
        assert_eq!(buffer_words(&rebuilt), buffer_words(&chain));
    }

    #[test]
    fn out_types_parsed_from_keys_match_the_table() {
        let chain = generate(7, "add/Bf16x32/m0");
        let from_keys = out_types_from_keys(&chain.keys());
        assert_eq!(from_keys, chain_out_types(&chain));
        assert!(from_keys.contains(&VecType::Bf16x32));
    }

    #[test]
    fn table_version_is_stable_across_calls() {
        assert_eq!(current_table_version(), current_table_version());
        assert_ne!(current_table_version(), 0);
    }
}
