//! Domain-agnostic differential-fuzzing engine: the campaign loop, ledger
//! bookkeeping, parallel compile, divergence banking, reverify, and replay,
//! generic over a [`Domain`] tenant.
//!
//! Targets are picked round-robin over the ledger's uncovered keys, so the
//! campaign always works the least-covered part of the universe. Each case is
//! lowered to `fuzz_kernel.cc`, compiled via the shared compile pipeline, run on
//! EMU (and optionally HW), compared by the domain's differential comparator,
//! and credited/banked accordingly. The work dir is `build/fuzz-{name}` and the
//! bank/replay persistence is the domain's own.
//!
//! Silicon-verified credit only: keys are credited when EMU == HW for the full
//! output buffer. Without `--hw`, the run is a smoke/EMU sweep with no credit.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::fuzzer::core::domain::{Backend, Banked, CampaignOptions, Domain};
use crate::fuzzer::core::ledger::Ledger;
use crate::fuzzer::core::toolchain::{catch_panic, compile_kernel_case, ToolPaths};

/// Last lines of a (possibly long) error message, for unreachable reasons.
fn tail_lines(msg: &str, n: usize) -> String {
    let lines: Vec<&str> = msg.lines().collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].join("\n")
}

/// One generated case awaiting compile/execute.
struct Case<D: Domain> {
    case: D::Case,
    case_dir: PathBuf,
}

/// Run a coverage-driven differential fuzz campaign for `dom` (or report/replay,
/// per options).
///
/// The `Sync` bounds are required only by the parallel compile pool, which
/// shares `&dom` and `&[Case<D>]` across `thread::scope` threads (the engine
/// never moves an owned case into a thread, so `Send` is not needed). A
/// zero-sized domain like `VectorDomain` and a plain-data case like `Chain`
/// satisfy them automatically.
pub fn run_campaign<D>(dom: &D, opts: &CampaignOptions)
where
    D: Domain + Sync,
    D::Case: Sync,
{
    if let Some(ref dir) = opts.replay {
        run_replay(dom, dir, opts);
        return;
    }

    let fuzz_dir = std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(format!("build/fuzz-{}", dom.name()));
    std::fs::create_dir_all(&fuzz_dir).ok();
    let ledger_path = fuzz_dir.join("ledger.json");

    let mut ledger = match Ledger::load(&ledger_path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Error: failed to load ledger: {e}");
            std::process::exit(1);
        }
    };
    let universe = dom.universe();

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
    let cases: Vec<Case<D>> = (0..opts.iterations)
        .map(|i| {
            let seed = base_seed.wrapping_add(i as u64);
            let target = &uncovered[i % uncovered.len()];
            let case = dom.generate(seed, target);
            let case_dir = fuzz_dir.join(format!("seed_{seed}"));
            Case { case, case_dir }
        })
        .collect();

    for (i, case) in cases.iter().enumerate() {
        std::fs::create_dir_all(&case.case_dir).ok();
        let cpp = dom.lower(&case.case);
        if let Err(e) = std::fs::write(case.case_dir.join("fuzz_kernel.cc"), &cpp) {
            eprintln!("seed {} write error: {e}", base_seed.wrapping_add(i as u64));
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
                let words = dom.buffer_words(&case.case);
                let r = compile_kernel_case(tools, &case.case_dir, words, dom.dtype(&case.case));
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
    for (i, (case, result)) in cases.iter().zip(&compile_results).enumerate() {
        if let Err(e) = result {
            let seed = base_seed.wrapping_add(i as u64);
            if opts.verbose {
                eprintln!("seed {} compile error: {e}", seed);
            }
            let target_key = dom.target_key(&case.case);
            let seeds = compile_fail_seeds.entry(target_key.clone()).or_default();
            if !seeds.contains(&seed) {
                seeds.push(seed);
            }
            if seeds.len() >= 2 {
                println!("key {} unreachable ({} distinct seeds fail to compile)", target_key, seeds.len());
                ledger.mark_unreachable(&target_key, &tail_lines(e, 5));
            }
        }
    }

    // Execute phase: sequential (HW is a single device; EMU cases are small).
    let exec_start = Instant::now();
    let (mut pass, mut fail, mut error, mut crash, mut folded) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut since_save = 0usize;
    for (i, (case, compile_result)) in cases.iter().zip(&compile_results).enumerate() {
        if compile_result.is_err() {
            continue;
        }
        let seed = base_seed.wrapping_add(i as u64);
        let xclbin = case.case_dir.join("aie.xclbin");
        let insts = case.case_dir.join("insts.bin");

        let emu_res =
            catch_panic(|| dom.observe(Backend::Interpreter, &xclbin, &insts, &case.case, opts.max_cycles));
        let emu = match emu_res {
            Err(panic_msg) => {
                crash += 1;
                println!("seed {seed} CRASH (emulator panic): {panic_msg}");
                ledger.mark_divergent(&dom.target_key(&case.case));
                if let Err(e) = dom.bank(&case.case_dir, &case.case, None, None) {
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
                println!("seed {seed} emu ok");
            }
            continue;
        }

        let npu_res =
            catch_panic(|| dom.observe(Backend::Hardware, &xclbin, &insts, &case.case, opts.max_cycles));
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

        let keys = dom.coverage_keys(&case.case);
        match dom.compare(&emu, &npu, &keys) {
            None => {
                pass += 1;
                ledger.credit_keys(&keys);
                if opts.verbose {
                    println!("seed {seed} MATCH ({} stages)", keys.len());
                }
            }
            Some(key) => {
                fail += 1;
                println!("seed {seed} MISMATCH -> key {key}");
                ledger.mark_divergent(&key);
                match dom.bank(&case.case_dir, &case.case, Some(&npu), Some(&emu)) {
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
fn run_replay<D: Domain>(dom: &D, dir: &Path, opts: &CampaignOptions) {
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

        // Reconstruct the runnable case + banked reference observation. A durable
        // bank (with pool) is replayed directly and is table-independent; a legacy
        // bank under a changed table is reported and skipped.
        let (case, reference, banked_keys) = match dom.load_banked(case_dir) {
            Ok(Banked::Replayable { case, reference, keys }) => (case, reference, keys),
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
            if let Err(e) = compile_kernel_case(
                tools.as_ref().unwrap(),
                case_dir,
                dom.buffer_words(&case),
                dom.dtype(&case),
            ) {
                errors += 1;
                println!("{name}: recompile failed: {}", tail_lines(&e, 3));
                continue;
            }
        }

        let insts = case_dir.join("insts.bin");
        match catch_panic(|| dom.observe(Backend::Interpreter, &xclbin, &insts, &case, opts.max_cycles)) {
            Err(panic_msg) => {
                errors += 1;
                println!("{name}: CRASH (emulator panic): {panic_msg}");
            }
            Ok(Err(e)) => {
                errors += 1;
                println!("{name}: emulator error: {e}");
            }
            Ok(Ok(emu)) => {
                match dom.compare(&emu, &reference, &banked_keys) {
                    None => {
                        matched += 1;
                        println!("{name}: MATCH (divergence resolved)");
                    }
                    Some(slice_key) => {
                        mismatched += 1;
                        println!("{name}: still divergent -> key {slice_key}");
                        // Dump the EMU side next to the banked HW output for byte diffs.
                        if let Err(e) = dom.dump_divergent_observation(case_dir, &emu) {
                            eprintln!("{name}: {e}");
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

    #[test]
    fn tail_lines_keeps_last_n() {
        let msg = "a\nb\nc\nd";
        assert_eq!(tail_lines(msg, 2), "c\nd");
        assert_eq!(tail_lines(msg, 10), msg);
    }
}
