//! Vector fuzz runner: compile, execute (EMU and optionally HW), compare,
//! ledger bookkeeping, divergence banking, and replay.
//!
//! Targets are picked round-robin over the ledger's uncovered keys, so the
//! campaign always works the least-covered part of the universe. Each case
//! reuses the scalar fuzzer's compile pipeline (Peano clang -> fuzz_template
//! -> aiecc.py) with dtype i32. The differential compare is per 64-byte slice;
//! slice k maps back to stage k of the chain, so the first divergent slice
//! identifies the divergent coverage key directly.
//!
//! Silicon-verified credit only: keys are credited when EMU == HW for the full
//! output buffer. Without `--hw`, the run is a smoke/EMU sweep with no credit.
//!
//! FAIL banking goes to `~/npu-work/experiments/phoenix-survival/vector/seed_N/`
//! (never `/tmp`); the bank dir is self-contained for replay via `--replay`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::fuzzer::runner::{catch_panic, compile_kernel_case, ToolPaths, TRACE_BUFFER_ELEMENTS};
use crate::fuzzer::vector::chain::Chain;
use crate::fuzzer::vector::gen::generate;
use crate::fuzzer::vector::ledger::Ledger;
use crate::fuzzer::vector::lower::lower_chain;
use crate::interpreter::execute::fuzz_recorder;
use crate::testing::test_cpp_parser::{BufferDef, BufferDir, BufferSpec, ElementType, InputPattern};
use crate::testing::xclbin_suite::{XclbinSuite, XclbinTest};

/// Options for the vector fuzz runner.
pub struct VecFuzzOptions {
    pub iterations: usize,
    /// Base seed (None = wall clock); case i runs seed base+i.
    pub seed: Option<u64>,
    pub jobs: usize,
    pub hw: bool,
    pub max_cycles: u64,
    /// Hits per key needed for completion (ledger target).
    pub target_hits: u32,
    pub verbose: bool,
    /// Print the coverage report and exit.
    pub report_only: bool,
    /// Replay banked divergences from this directory (EMU vs banked npu_output.bin).
    pub replay: Option<PathBuf>,
}

/// Serialized form of a banked chain: enough to regenerate and audit.
/// Stages are stored explicitly so a bank survives table evolution detectably:
/// replay regenerates from (seed, target_key) and verifies keys match.
#[derive(Serialize, Deserialize)]
struct ChainRecord {
    seed: u64,
    target_key: String,
    keys: Vec<String>,
    stages: Vec<StageRecord>,
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
        }
    }
}

/// Buffer size in i32 words: pool and output both live in `--size`-word
/// buffers, so take the max of the two and let the shorter side zero-pad.
fn buffer_words(chain: &Chain) -> usize {
    (chain.pool_slots() * 16).max(chain.out_bytes() / 4)
}

/// First differing 64-byte slice index between two buffers, or None if equal.
/// A length mismatch beyond the common prefix counts as a divergence at the
/// first slice past the common length.
fn first_divergent_slice(a: &[u8], b: &[u8]) -> Option<usize> {
    let common = a.len().min(b.len());
    for (i, (sa, sb)) in a[..common].chunks(64).zip(b[..common].chunks(64)).enumerate() {
        if sa != sb {
            return Some(i);
        }
    }
    if a.len() != b.len() {
        return Some(common / 64);
    }
    None
}

/// Map a divergent slice index to the chain's coverage key. Slices past the
/// last stage (zero padding) clamp to the final stage -- a diff there still
/// means the kernel wrote where it should not have, attributed to the chain.
fn slice_to_key(chain: &Chain, slice: usize) -> String {
    let keys = chain.keys();
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

/// Run the chain's xclbin through the emulator with the recorder armed.
/// Returns (output, trace, executed-keys).
fn run_emulator_vec(
    xclbin_path: &Path,
    chain: &Chain,
    max_cycles: u64,
) -> Result<(Vec<u8>, Option<Vec<u8>>, Vec<String>), String> {
    let spec = make_vec_buffer_spec(chain, buffer_words(chain));
    let test = XclbinTest::from_path(xclbin_path).with_buffer_spec(spec);
    let suite = XclbinSuite::new().with_max_cycles(max_cycles);
    fuzz_recorder::arm();
    let (_outcome, raw_output, trace) = suite.run_single_with_trace(&test);
    let executed = fuzz_recorder::take().unwrap_or_default();
    let output = raw_output.ok_or_else(|| "Emulator produced no output".to_string())?;
    Ok((output, trace, executed))
}

/// Run the chain's xclbin on real NPU hardware. Returns (output, trace).
fn run_npu_vec(
    xclbin_path: &Path,
    insts_path: &Path,
    chain: &Chain,
) -> Result<(Vec<u8>, Option<Vec<u8>>), String> {
    use crate::testing::npu_runner;
    if !npu_runner::npu_available() {
        return Err("NPU hardware not available".into());
    }
    let spec = make_vec_buffer_spec(chain, buffer_words(chain));
    let test_name = format!("vecfuzz_seed_{}", chain.seed);
    match npu_runner::run_on_npu(&spec, &test_name, xclbin_path, insts_path, 30) {
        Ok(result) => {
            let trace = result.extra_outputs.get("buf_trace").cloned();
            Ok((result.output, trace))
        }
        Err(e) => Err(format!("{:?}", e)),
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

/// Last lines of a (possibly long) error message, for unreachable reasons.
fn tail_lines(msg: &str, n: usize) -> String {
    let lines: Vec<&str> = msg.lines().collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].join("\n")
}

/// One generated case awaiting compile/execute.
struct VecCase {
    chain: Chain,
    case_dir: PathBuf,
}

/// Run the vector fuzz campaign (or report/replay, per options).
pub fn run_vector_fuzz(opts: &VecFuzzOptions) {
    if let Some(ref dir) = opts.replay {
        run_replay(dir, opts);
        return;
    }

    let fuzz_dir = std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("build/fuzz-vector");
    std::fs::create_dir_all(&fuzz_dir).ok();
    let ledger_path = fuzz_dir.join("ledger.json");

    let mut ledger = match Ledger::load(&ledger_path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Error: failed to load ledger: {e}");
            std::process::exit(1);
        }
    };

    if opts.report_only {
        print!("{}", ledger.report(opts.target_hits));
        return;
    }

    let uncovered = ledger.uncovered(opts.target_hits);
    if uncovered.is_empty() {
        println!("Vector fuzz: coverage complete (target {} hits/key)", opts.target_hits);
        print!("{}", ledger.report(opts.target_hits));
        return;
    }
    if opts.iterations == 0 {
        println!("Vector fuzz: 0 iterations requested ({} keys uncovered)", uncovered.len());
        return;
    }

    let base_seed = opts.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    });
    println!(
        "Vector fuzz: {} iterations, base seed {}, {} uncovered keys, hw={}",
        opts.iterations,
        base_seed,
        uncovered.len(),
        opts.hw
    );

    let tools = match ToolPaths::discover() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error: failed to discover build tools: {e}");
            std::process::exit(1);
        }
    };

    // Generate cases: round-robin over the least-covered keys.
    let cases: Vec<VecCase> = (0..opts.iterations)
        .map(|i| {
            let seed = base_seed.wrapping_add(i as u64);
            let target = &uncovered[i % uncovered.len()];
            let chain = generate(seed, target);
            let case_dir = fuzz_dir.join(format!("seed_{seed}"));
            VecCase { chain, case_dir }
        })
        .collect();

    for case in &cases {
        std::fs::create_dir_all(&case.case_dir).ok();
        let cpp = lower_chain(&case.chain);
        if let Err(e) = std::fs::write(case.case_dir.join("fuzz_kernel.cc"), &cpp) {
            eprintln!("seed {} write error: {e}", case.chain.seed);
        }
    }

    // Compile phase (parallel work-stealing, like the scalar runner).
    let compile_start = Instant::now();
    let compiled: Vec<std::sync::Mutex<Option<Result<(), String>>>> =
        (0..cases.len()).map(|_| std::sync::Mutex::new(None)).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..opts.jobs.max(1) {
            let cases = &cases;
            let compiled = &compiled;
            let next = &next;
            let done = &done;
            let tools = &tools;
            s.spawn(move || loop {
                let idx = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if idx >= cases.len() {
                    break;
                }
                let case = &cases[idx];
                let words = buffer_words(&case.chain);
                let r = compile_kernel_case(tools, &case.case_dir, words, "i32");
                *compiled[idx].lock().unwrap() = Some(r);
                let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if !opts.verbose {
                    eprint!("\rCompiling [{}/{}]", n, cases.len());
                }
            });
        }
    });
    if !opts.verbose {
        eprintln!();
    }
    let compile_results: Vec<Result<(), String>> =
        compiled.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect();
    let compile_errors = compile_results.iter().filter(|r| r.is_err()).count();
    println!(
        "Compile: {} ok, {} error ({:.1}s, {} threads)",
        cases.len() - compile_errors,
        compile_errors,
        compile_start.elapsed().as_secs_f64(),
        opts.jobs
    );

    // Two distinct seeds failing the same target key => the key is unreachable.
    let mut compile_fail_seeds: HashMap<String, Vec<u64>> = HashMap::new();
    for (case, result) in cases.iter().zip(&compile_results) {
        if let Err(e) = result {
            if opts.verbose {
                eprintln!("seed {} compile error: {e}", case.chain.seed);
            }
            let seeds = compile_fail_seeds.entry(case.chain.target_key.clone()).or_default();
            if !seeds.contains(&case.chain.seed) {
                seeds.push(case.chain.seed);
            }
            if seeds.len() >= 2 {
                println!(
                    "key {} unreachable ({} distinct seeds fail to compile)",
                    case.chain.target_key,
                    seeds.len()
                );
                ledger.mark_unreachable(&case.chain.target_key, &tail_lines(e, 5));
            }
        }
    }

    // Execute phase: sequential (HW is a single device; EMU cases are small).
    let exec_start = Instant::now();
    let (mut pass, mut fail, mut error, mut crash, mut folded) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut since_save = 0usize;
    for (case, compile_result) in cases.iter().zip(&compile_results) {
        if compile_result.is_err() {
            continue;
        }
        let seed = case.chain.seed;
        let xclbin = case.case_dir.join("aie.xclbin");
        let insts = case.case_dir.join("insts.bin");

        let emu = catch_panic(|| run_emulator_vec(&xclbin, &case.chain, opts.max_cycles));
        let (emu_out, _emu_trace, executed) = match emu {
            Err(panic_msg) => {
                crash += 1;
                println!("seed {seed} CRASH (emulator panic): {panic_msg}");
                ledger.mark_divergent(&case.chain.target_key);
                if let Err(e) = bank_case(&case.case_dir, &case.chain, None, None, &[]) {
                    eprintln!("seed {seed} bank error: {e}");
                }
                continue;
            }
            Ok(Err(e)) => {
                error += 1;
                if opts.verbose {
                    println!("seed {seed} emulator error: {e}");
                }
                continue;
            }
            Ok(Ok(t)) => t,
        };
        if executed.is_empty() {
            folded += 1;
            println!("seed {seed} WARNING: no vector ops executed (chain folded by compiler)");
        }

        if !opts.hw {
            pass += 1;
            if opts.verbose {
                println!(
                    "seed {seed} emu ok ({} bytes, {} vector ops executed)",
                    emu_out.len(),
                    executed.len()
                );
            }
            continue;
        }

        let npu = catch_panic(|| run_npu_vec(&xclbin, &insts, &case.chain));
        let (npu_out, npu_trace) = match npu {
            Err(panic_msg) => {
                error += 1;
                println!("seed {seed} hw panic: {panic_msg}");
                continue;
            }
            Ok(Err(e)) => {
                error += 1;
                if opts.verbose {
                    println!("seed {seed} hw error: {e}");
                }
                continue;
            }
            Ok(Ok(pair)) => pair,
        };

        match first_divergent_slice(&emu_out, &npu_out) {
            None => {
                pass += 1;
                ledger.credit_keys(&case.chain.keys());
                if opts.verbose {
                    println!("seed {seed} MATCH ({} stages)", case.chain.stages.len());
                }
            }
            Some(slice) => {
                fail += 1;
                let key = slice_to_key(&case.chain, slice);
                println!("seed {seed} MISMATCH at slice {slice} -> key {key}");
                ledger.mark_divergent(&key);
                match bank_case(&case.case_dir, &case.chain, Some(&npu_out), npu_trace.as_deref(), &executed)
                {
                    Ok(dir) => println!("  banked to {}", dir.display()),
                    Err(e) => eprintln!("  bank error: {e}"),
                }
            }
        }

        since_save += 1;
        if since_save >= 10 {
            since_save = 0;
            if let Err(e) = ledger.save(&ledger_path) {
                eprintln!("ledger save error: {e}");
            }
        }
    }

    if let Err(e) = ledger.save(&ledger_path) {
        eprintln!("ledger save error: {e}");
    }

    println!(
        "Execute: {pass} pass, {fail} fail, {error} error, {crash} CRASH ({:.1}s)",
        exec_start.elapsed().as_secs_f64()
    );
    if folded > 0 {
        println!("  ({folded} chains executed no vector ops -- folded)");
    }
    println!(
        "Vector fuzz complete: {pass} pass, {fail} fail, {} error, {crash} CRASH",
        error + compile_errors
    );
    println!("  uncovered keys remaining: {}", ledger.uncovered(opts.target_hits).len());
    if fail > 0 || crash > 0 {
        std::process::exit(1);
    }
}

/// Replay banked divergences: regenerate, recompile if needed, run EMU, and
/// compare against the banked silicon output.
fn run_replay(dir: &Path, opts: &VecFuzzOptions) {
    let mut entries: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.is_dir() && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("seed_"))
            })
            .collect(),
        Err(e) => {
            eprintln!("Error: cannot read replay dir {}: {e}", dir.display());
            std::process::exit(1);
        }
    };
    entries.sort();
    if entries.is_empty() {
        println!("Replay: no seed_* directories in {}", dir.display());
        return;
    }

    let mut tools: Option<ToolPaths> = None;
    let (mut matched, mut mismatched, mut errors) = (0usize, 0usize, 0usize);

    for case_dir in &entries {
        let name = case_dir.file_name().unwrap().to_string_lossy().to_string();
        let record: ChainRecord = match std::fs::read_to_string(case_dir.join("chain.json"))
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string()))
        {
            Ok(r) => r,
            Err(e) => {
                errors += 1;
                println!("{name}: chain.json error: {e}");
                continue;
            }
        };

        let chain = generate(record.seed, &record.target_key);
        if chain.keys() != record.keys {
            errors += 1;
            println!("{name}: regenerated keys differ from banked keys (table changed?) -- skipping");
            continue;
        }

        let xclbin = case_dir.join("aie.xclbin");
        if !xclbin.exists() {
            if tools.is_none() {
                match ToolPaths::discover() {
                    Ok(t) => tools = Some(t),
                    Err(e) => {
                        eprintln!("Error: failed to discover build tools: {e}");
                        std::process::exit(1);
                    }
                }
            }
            if let Err(e) =
                compile_kernel_case(tools.as_ref().unwrap(), case_dir, buffer_words(&chain), "i32")
            {
                errors += 1;
                println!("{name}: recompile failed: {}", tail_lines(&e, 3));
                continue;
            }
        }

        let npu_out = match std::fs::read(case_dir.join("npu_output.bin")) {
            Ok(b) => b,
            Err(e) => {
                errors += 1;
                println!("{name}: npu_output.bin: {e}");
                continue;
            }
        };

        match catch_panic(|| run_emulator_vec(&xclbin, &chain, opts.max_cycles)) {
            Err(panic_msg) => {
                errors += 1;
                println!("{name}: CRASH (emulator panic): {panic_msg}");
            }
            Ok(Err(e)) => {
                errors += 1;
                println!("{name}: emulator error: {e}");
            }
            Ok(Ok((emu_out, _trace, _executed))) => match first_divergent_slice(&emu_out, &npu_out) {
                None => {
                    matched += 1;
                    println!("{name}: MATCH (divergence resolved)");
                }
                Some(slice) => {
                    mismatched += 1;
                    println!(
                        "{name}: still divergent at slice {slice} -> key {}",
                        slice_to_key(&chain, slice)
                    );
                }
            },
        }
    }

    println!("Replay complete: {matched} match, {mismatched} divergent, {errors} error");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzzer::vector::table::table;

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
        let slice = first_divergent_slice(&emu, &npu).expect("must diverge");
        assert_eq!(slice, 1);
        assert_eq!(slice_to_key(&chain, slice), chain.keys()[1]);
    }

    #[test]
    fn equal_buffers_with_zero_padding_do_not_diverge() {
        // out_bytes shorter than buffer: both sides zero-padded to buffer size.
        let mut emu = vec![0u8; 4 * 64];
        emu[..128].fill(0x5A); // two written stages
        let npu = emu.clone();
        assert_eq!(first_divergent_slice(&emu, &npu), None);
    }

    #[test]
    fn length_mismatch_diverges_at_first_extra_slice() {
        let emu = vec![0u8; 3 * 64];
        let npu = vec![0u8; 2 * 64];
        assert_eq!(first_divergent_slice(&emu, &npu), Some(2));
    }

    #[test]
    fn slice_past_last_stage_clamps_to_final_key() {
        let chain = generate(2, "add/I32x16/m0");
        let last = chain.keys().len() - 1;
        assert_eq!(slice_to_key(&chain, last + 5), chain.keys()[last]);
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
    fn tail_lines_keeps_last_n() {
        let msg = "a\nb\nc\nd";
        assert_eq!(tail_lines(msg, 2), "c\nd");
        assert_eq!(tail_lines(msg, 10), msg);
    }

    /// 200-seed Peano compile-clean: every lowered chain must compile with
    /// `clang -O2 -c` (no aiecc). Catches bad emit strings (the bf16 sel/bcast
    /// spellings were extrapolated, not spike-probed). Run once per table
    /// change:
    /// `cargo test --lib --features tooling vector_compile_clean -- --ignored`
    #[test]
    #[ignore = "needs Peano toolchain; run explicitly after table changes"]
    fn vector_compile_clean_200_seeds() {
        let tools = ToolPaths::discover().expect("tool discovery");
        let universe = Ledger::universe();
        let dir = std::env::current_dir().unwrap().join("build/fuzz-vector/compile-clean");
        std::fs::create_dir_all(&dir).unwrap();

        let cases: Vec<(u64, String, PathBuf)> = (0..200u64)
            .map(|seed| {
                let key = universe[(seed as usize) % universe.len()].clone();
                let case_dir = dir.join(format!("seed_{seed}"));
                std::fs::create_dir_all(&case_dir).unwrap();
                let chain = generate(seed, &key);
                std::fs::write(case_dir.join("fuzz_kernel.cc"), lower_chain(&chain)).unwrap();
                (seed, key, case_dir)
            })
            .collect();

        // Sanity: round-robin over 200 seeds covers every table entry.
        let entry_count = table().len();
        assert!(universe.len() >= entry_count);

        let failures = std::sync::Mutex::new(Vec::new());
        let next = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..8 {
                let cases = &cases;
                let tools = &tools;
                let failures = &failures;
                let next = &next;
                s.spawn(move || loop {
                    let idx = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if idx >= cases.len() {
                        break;
                    }
                    let (seed, key, case_dir) = &cases[idx];
                    let mut cmd = std::process::Command::new(&tools.peano_clang);
                    cmd.arg("--target=aie2-none-unknown-elf")
                        .arg("-O2")
                        .arg("-std=c++20") // aie_api headers are C++20 (concepts)
                        .arg("-c")
                        .arg(case_dir.join("fuzz_kernel.cc"))
                        .arg("-o")
                        .arg(case_dir.join("fuzz_kernel.cc.o"));
                    if let Some(inc) = tools.aie_api_include() {
                        cmd.arg("-I").arg(inc);
                    }
                    tools.apply_env(&mut cmd);
                    match cmd.output() {
                        Ok(out) if out.status.success() => {}
                        Ok(out) => failures.lock().unwrap().push(format!(
                            "seed {seed} key {key}:\n{}",
                            tail_lines(&String::from_utf8_lossy(&out.stderr), 8)
                        )),
                        Err(e) => failures.lock().unwrap().push(format!("seed {seed} key {key}: spawn {e}")),
                    }
                });
            }
        });

        let failures = failures.into_inner().unwrap();
        assert!(failures.is_empty(), "{} compile failures:\n{}", failures.len(), failures.join("\n---\n"));
    }
}
