//! Fuzz iteration loop: generate, compile, run, compare.
//!
//! Orchestrates the full fuzzing pipeline. Each iteration:
//! 1. Generate random `FuzzParams` from a seed
//! 2. Lower to C++ kernel source
//! 3. Compile to xclbin via IRON template + aiecc.py
//! 4. Run on emulator
//! 5. Run on real NPU hardware
//! 6. Compare outputs

use std::io::Write;
use std::path::PathBuf;

use crate::fuzzer::gen;
use crate::fuzzer::lower_cpp;
use crate::testing::runner_config::Options;

/// Run the fuzz loop: generate, compile, run on emulator + NPU, compare.
pub fn run_fuzz(opts: &Options) {
    let iterations = opts.fuzz_iterations;
    let base_seed = opts.fuzz_seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    });

    println!("Fuzzing {} iterations, base seed {}", iterations, base_seed);

    let fuzz_dir = PathBuf::from("build/fuzz");
    std::fs::create_dir_all(&fuzz_dir).ok();

    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut error = 0usize;

    for i in 0..iterations {
        let seed = base_seed.wrapping_add(i as u64);
        let params = gen::generate(seed);
        let case_dir = fuzz_dir.join(format!("seed_{}", seed));
        std::fs::create_dir_all(&case_dir).ok();

        // 1. Lower to C++
        let cpp = lower_cpp::lower_to_cpp(&params);
        let kernel_path = case_dir.join("fuzz_kernel.cc");
        if let Err(e) = std::fs::write(&kernel_path, &cpp) {
            error += 1;
            if opts.verbose {
                eprintln!("[{}/{}] seed {} write error: {}", i + 1, iterations, seed, e);
            }
            continue;
        }

        if opts.verbose {
            println!("[{}/{}] seed {} generated {} ops, {} bytes C++",
                i + 1, iterations, seed, params.body.ops.len(), cpp.len());
        }

        // 2-5. Compile + run + compare (Task 6 implements these)
        // For now, count successful generation as a pass.
        match compile_and_compare(opts, &params, &case_dir) {
            Ok(true) => pass += 1,
            Ok(false) => {
                fail += 1;
                eprintln!("MISMATCH at seed {}", seed);
            }
            Err(e) => {
                error += 1;
                if opts.verbose {
                    eprintln!("[{}/{}] seed {} error: {}", i + 1, iterations, seed, e);
                }
            }
        }

        // Progress (overwrite line in non-verbose mode)
        if !opts.verbose {
            print!("\r[{}/{}] {} pass, {} fail, {} error",
                i + 1, iterations, pass, fail, error);
            std::io::stdout().flush().ok();
        }
    }

    if !opts.verbose {
        println!(); // final newline after progress line
    }
    println!("Fuzz complete: {} pass, {} fail, {} error", pass, fail, error);

    if fail > 0 {
        std::process::exit(1);
    }
}

/// Compile a fuzz case and compare emulator vs NPU output.
///
/// Returns Ok(true) for match, Ok(false) for mismatch, Err for compile/run errors.
/// TODO(Task 6): Implement real compilation, emulation, and hardware execution.
fn compile_and_compare(
    _opts: &Options,
    _params: &crate::fuzzer::params::FuzzParams,
    _case_dir: &std::path::Path,
) -> Result<bool, String> {
    // Placeholder: generation-only mode. Task 6 wires in real compilation
    // and execution. For now, report success if we got this far.
    Err("compile/run not yet implemented (Task 6)".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_fuzz_zero_iterations() {
        // Zero iterations should complete without error.
        let opts = crate::testing::runner_config::Options {
            mode: crate::testing::runner_config::RunMode::Fuzz,
            verbose: false,
            jobs: 1,
            filters: vec![],
            chess_only: false,
            elfanalyze: false,
            chess_build: false,
            chess_emulator: false,
            chess_hardware: false,
            hw: false,
            aiesim: false,
            build_nice: 19,
            max_cycles: 100_000,
            unit_tests: false,
            unit_tests_aiesim: false,
            unit_tests_aiesim_timeout: 0,
            no_build: false,
            hw_only: false,
            examples: false,
            rebuild: false,
            timeout_secs: None,
            build_dir: None,
            max_failures: None,
            lit_args: vec![],
            watchdog_secs: 3600,
            trace_size: 0,
            aiesim_trace: false,
            list_only: false,
            output_path: std::path::PathBuf::from("/dev/null"),
            format_json: false,
            fuzz_iterations: 0,
            fuzz_seed: Some(1),
        };
        run_fuzz(&opts);
    }
}
