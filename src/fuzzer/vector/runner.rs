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

use crate::fuzzer::core::domain::{Backend, Banked, Domain};
use crate::fuzzer::core::ledger::Ledger;
use crate::fuzzer::core::toolchain::{catch_panic, compile_kernel_case, ToolPaths};
use crate::fuzzer::vector::chain::Chain;
use crate::fuzzer::vector::domain::VectorDomain;

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
    /// Clear divergent flags and re-earn them against silicon this run. Requires
    /// `--hw` (credit is only honest with a live silicon comparison).
    pub reverify: bool,
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

    let dom = VectorDomain;

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
    let universe = crate::fuzzer::vector::table::universe_keys();

    if opts.report_only {
        print!("{}", ledger.report(&universe, opts.target_hits));
        return;
    }

    // Re-verify: clear divergent flags and re-earn them against silicon this run.
    // Credit is only honest with a live silicon comparison, so require --hw.
    let reverify_keys: Vec<String> = if opts.reverify {
        if !opts.hw {
            eprintln!("Error: --reverify requires --hw (clearing flags needs a silicon re-check)");
            std::process::exit(1);
        }
        let cleared = ledger.take_divergent();
        if cleared.is_empty() {
            println!("Re-verify: no divergent keys to clear.");
        } else {
            println!("Re-verify: cleared {} divergent flags to re-earn:", cleared.len());
            for k in &cleared {
                println!("  {k}");
            }
        }
        cleared
    } else {
        Vec::new()
    };

    let uncovered = ledger.uncovered(&universe, opts.target_hits);
    if uncovered.is_empty() {
        println!("Vector fuzz: coverage complete (target {} hits/key)", opts.target_hits);
        print!("{}", ledger.report(&universe, opts.target_hits));
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
            let chain = dom.generate(seed, target);
            let case_dir = fuzz_dir.join(format!("seed_{seed}"));
            VecCase { chain, case_dir }
        })
        .collect();

    for case in &cases {
        std::fs::create_dir_all(&case.case_dir).ok();
        let cpp = dom.lower(&case.chain);
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
            let dom = &dom;
            s.spawn(move || loop {
                let idx = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if idx >= cases.len() {
                    break;
                }
                let case = &cases[idx];
                let words = dom.buffer_words(&case.chain);
                let r = compile_kernel_case(tools, &case.case_dir, words, dom.dtype());
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

        let emu_res =
            catch_panic(|| dom.observe(Backend::Interpreter, &xclbin, &insts, &case.chain, opts.max_cycles));
        let emu = match emu_res {
            Err(panic_msg) => {
                crash += 1;
                println!("seed {seed} CRASH (emulator panic): {panic_msg}");
                ledger.mark_divergent(&case.chain.target_key);
                if let Err(e) = dom.bank(&case.case_dir, &case.chain, None, None) {
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
        let warnings = dom.warnings(&emu);
        if !warnings.is_empty() {
            folded += 1;
            for w in &warnings {
                println!("seed {seed} WARNING: {w}");
            }
        }

        if !opts.hw {
            pass += 1;
            if opts.verbose {
                println!(
                    "seed {seed} emu ok ({} bytes, {} vector ops executed)",
                    emu.output.len(),
                    emu.executed.len()
                );
            }
            continue;
        }

        let npu_res =
            catch_panic(|| dom.observe(Backend::Hardware, &xclbin, &insts, &case.chain, opts.max_cycles));
        let npu = match npu_res {
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
            Ok(Ok(obs)) => obs,
        };

        match dom.compare(&emu, &npu, &case.chain) {
            None => {
                pass += 1;
                ledger.credit_keys(&case.chain.keys());
                if opts.verbose {
                    println!("seed {seed} MATCH ({} stages)", case.chain.stages.len());
                }
            }
            Some(key) => {
                fail += 1;
                println!("seed {seed} MISMATCH -> key {key}");
                ledger.mark_divergent(&key);
                match dom.bank(&case.case_dir, &case.chain, Some(&npu), Some(&emu)) {
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

    // Record re-verify outcomes. A cleared key is *resolved* only if it actually
    // earned a silicon-clean hit this run; re-flagged keys stay divergent; keys
    // that were never exercised are left uncovered (not falsely resolved).
    if !reverify_keys.is_empty() {
        let (mut healed, mut still, mut untested) = (0usize, 0usize, 0usize);
        for key in &reverify_keys {
            if ledger.is_divergent(key) {
                still += 1;
            } else if ledger.hit_count(key) >= 1 {
                ledger.mark_resolved(key, &format!("re-verified clean, base seed {base_seed}"));
                healed += 1;
            } else {
                untested += 1;
            }
        }
        println!(
            "Re-verify: {healed} resolved, {still} still divergent, {untested} not exercised (of {})",
            reverify_keys.len()
        );
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
    println!("  uncovered keys remaining: {}", ledger.uncovered(&universe, opts.target_hits).len());
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

    let dom = VectorDomain;
    let mut tools: Option<ToolPaths> = None;
    let (mut matched, mut mismatched, mut errors) = (0usize, 0usize, 0usize);

    for case_dir in &entries {
        let name = case_dir.file_name().unwrap().to_string_lossy().to_string();

        // Reconstruct the runnable chain + banked reference observation. A durable
        // bank (with pool) is replayed directly and is table-independent; a legacy
        // bank under a changed table is reported and skipped.
        let (chain, reference) = match dom.load_banked(case_dir) {
            Ok(Banked::Replayable { case, reference, keys: _ }) => (case, reference),
            Ok(Banked::Skip(why)) => {
                errors += 1;
                println!("{name}: {why}");
                continue;
            }
            Err(e) => {
                errors += 1;
                println!("{name}: {e}");
                continue;
            }
        };

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
                compile_kernel_case(tools.as_ref().unwrap(), case_dir, dom.buffer_words(&chain), dom.dtype())
            {
                errors += 1;
                println!("{name}: recompile failed: {}", tail_lines(&e, 3));
                continue;
            }
        }

        let insts = case_dir.join("insts.bin");
        match catch_panic(|| dom.observe(Backend::Interpreter, &xclbin, &insts, &chain, opts.max_cycles)) {
            Err(panic_msg) => {
                errors += 1;
                println!("{name}: CRASH (emulator panic): {panic_msg}");
            }
            Ok(Err(e)) => {
                errors += 1;
                println!("{name}: emulator error: {e}");
            }
            Ok(Ok(emu)) => {
                match dom.compare(&emu, &reference, &chain) {
                    None => {
                        matched += 1;
                        println!("{name}: MATCH (divergence resolved)");
                    }
                    Some(slice_key) => {
                        mismatched += 1;
                        println!("{name}: still divergent -> key {slice_key}");
                        // Dump the EMU side next to the banked HW output for byte diffs.
                        if let Err(e) = std::fs::write(case_dir.join("emu_output.bin"), &emu.output) {
                            eprintln!("{name}: emu_output.bin write error: {e}");
                        }
                    }
                }
            }
        }
    }

    println!("Replay complete: {matched} match, {mismatched} divergent, {errors} error");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzzer::vector::gen::generate;
    use crate::fuzzer::vector::lower::lower_chain;
    use crate::fuzzer::vector::table::table;

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
        let universe = crate::fuzzer::vector::table::universe_keys();
        let dir = std::env::current_dir().unwrap().join("build/fuzz-vector/compile-clean");
        std::fs::create_dir_all(&dir).unwrap();

        // At least 200 seeds, and at least one seed per universe key so every
        // table entry/mode gets compiled.
        let n_seeds = 200u64.max(universe.len() as u64);
        let cases: Vec<(u64, String, PathBuf)> = (0..n_seeds)
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
                        Ok(out) if out.status.success() => {
                            // A successful compile can still fail to LINK: some
                            // aie_api paths (i8 mmul 4x8x4, vector::set on 8-bit
                            // lanes) reference an undefined runtime-mode
                            // ::shuffle(v8DB64) symbol Peano's lib never defines.
                            // Scan undefined symbols so link gaps fail here, not
                            // in the aiecc pipeline. memset/memcpy resolve from
                            // compiler-rt and are expected.
                            if let Some(und) = undefined_symbols(tools, &case_dir.join("fuzz_kernel.cc.o")) {
                                failures
                                    .lock()
                                    .unwrap()
                                    .push(format!("seed {seed} key {key}: undefined symbols:\n{und}"));
                            }
                        }
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

    /// Undefined symbols in an object file (excluding compiler-rt intrinsics:
    /// mem* and `__`-prefixed soft-float builtins like __floatunsisf, which
    /// aiecc resolves from libclang_rt), via Peano llvm-objdump. Real library
    /// gaps are C++-mangled (e.g. _Z7shuffleDv8_DB64_S0_j). None = clean.
    fn undefined_symbols(tools: &ToolPaths, obj: &Path) -> Option<String> {
        let objdump = tools.peano_clang.parent()?.join("llvm-objdump");
        let out = std::process::Command::new(objdump).arg("-t").arg(obj).output().ok()?;
        let text = String::from_utf8_lossy(&out.stdout);
        let und: Vec<&str> = text
            .lines()
            .filter(|l| l.contains("*UND*"))
            .filter(|l| {
                let sym = l.rsplit(' ').next().unwrap_or("");
                !sym.starts_with("mem") && !sym.starts_with("__")
            })
            .collect();
        if und.is_empty() {
            None
        } else {
            Some(und.join("\n"))
        }
    }
}
