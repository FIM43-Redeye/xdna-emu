//! CLI argument parsing for the `fuzz` and `fuzz-vector` subcommands.
//!
//! Pure mapping from process args to `FuzzOptions` / `VecFuzzOptions`, kept in
//! the library so it is unit-testable (binaries are not). `main.rs` is a thin
//! caller.

use std::path::PathBuf;

use crate::fuzzer::runner::FuzzOptions;
use crate::fuzzer::vector::runner::VecFuzzOptions;

/// Runaway-guard ceiling for emulator cycles per fuzz case. Fuzz kernels are
/// tiny (buffers 16-256 elems, 2-8 ops), so this only bounds pathological runs.
const DEFAULT_MAX_CYCLES: u64 = 10_000_000;

fn default_jobs() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

fn parse_next<'a, I, T>(iter: &mut I, flag: &str) -> Result<T, String>
where
    I: Iterator<Item = &'a String>,
    T: std::str::FromStr,
{
    let raw = iter.next().ok_or_else(|| format!("{} requires a value", flag))?;
    raw.parse::<T>().map_err(|_| format!("invalid value for {}: {}", flag, raw))
}

/// Parse `fuzz` subcommand args (full argv, including argv[0] and the `fuzz`
/// token) into `FuzzOptions`. Unknown flags are errors so typos fail loudly.
pub fn parse_fuzz_args(args: &[String]) -> Result<FuzzOptions, String> {
    let mut opts = FuzzOptions {
        verbose: false,
        jobs: default_jobs(),
        hw: false,
        max_cycles: DEFAULT_MAX_CYCLES,
        fuzz_iterations: 0,
        fuzz_seed: None,
        trace_sweep: false,
        trace_sweep_reps: 5,
    };

    // Skip argv[0] (binary name); the `fuzz` token is consumed by its match arm.
    // Items borrow from `args`, not from `iter`, so calling parse_next(&mut iter)
    // inside the loop body after iter.next() is borrow-clean.
    let mut iter = args.iter().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "fuzz" => {}
            "--iterations" | "-n" => opts.fuzz_iterations = parse_next(&mut iter, "--iterations")?,
            "--seed" => opts.fuzz_seed = Some(parse_next(&mut iter, "--seed")?),
            "--jobs" | "-j" => opts.jobs = parse_next(&mut iter, "--jobs")?,
            "--max-cycles" => opts.max_cycles = parse_next(&mut iter, "--max-cycles")?,
            "--hw" => opts.hw = true,
            "--no-hw" => opts.hw = false,
            "--trace-sweep" => opts.trace_sweep = true,
            "--trace-sweep-reps" => opts.trace_sweep_reps = parse_next(&mut iter, "--trace-sweep-reps")?,
            "--verbose" | "-v" => opts.verbose = true,
            other => return Err(format!("unknown fuzz argument: {}", other)),
        }
    }
    Ok(opts)
}

/// Parse `fuzz-vector` subcommand args (full argv, including argv[0] and the
/// `fuzz-vector` token) into `VecFuzzOptions`. Unknown flags are errors.
pub fn parse_vector_fuzz_args(args: &[String]) -> Result<VecFuzzOptions, String> {
    let mut opts = VecFuzzOptions {
        iterations: 0,
        seed: None,
        jobs: default_jobs(),
        hw: false,
        max_cycles: DEFAULT_MAX_CYCLES,
        target_hits: 10,
        verbose: false,
        report_only: false,
        replay: None,
    };

    let mut iter = args.iter().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "fuzz-vector" => {}
            "--iterations" | "-n" => opts.iterations = parse_next(&mut iter, "--iterations")?,
            "--seed" => opts.seed = Some(parse_next(&mut iter, "--seed")?),
            "--jobs" | "-j" => opts.jobs = parse_next(&mut iter, "--jobs")?,
            "--hw" => opts.hw = true,
            "--no-hw" => opts.hw = false,
            "--max-cycles" => opts.max_cycles = parse_next(&mut iter, "--max-cycles")?,
            "--target-hits" => opts.target_hits = parse_next(&mut iter, "--target-hits")?,
            "--report" => opts.report_only = true,
            "--replay" => {
                let dir = iter.next().ok_or("--replay requires a directory")?;
                opts.replay = Some(PathBuf::from(dir));
            }
            "--verbose" | "-v" => opts.verbose = true,
            other => return Err(format!("unknown fuzz-vector argument: {}", other)),
        }
    }
    Ok(opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(rest: &[&str]) -> Vec<String> {
        let mut v = vec!["xdna-emu".to_string(), "fuzz".to_string()];
        v.extend(rest.iter().map(|s| s.to_string()));
        v
    }

    fn vargv(rest: &[&str]) -> Vec<String> {
        let mut v = vec!["xdna-emu".to_string(), "fuzz-vector".to_string()];
        v.extend(rest.iter().map(|s| s.to_string()));
        v
    }

    #[test]
    fn vector_defaults_when_only_subcommand() {
        let o = parse_vector_fuzz_args(&vargv(&[])).unwrap();
        assert_eq!(o.iterations, 0);
        assert_eq!(o.seed, None);
        assert!(!o.hw);
        assert_eq!(o.target_hits, 10);
        assert!(!o.report_only);
        assert!(o.replay.is_none());
        assert!(!o.verbose);
    }

    #[test]
    fn vector_parses_all_flags() {
        let o = parse_vector_fuzz_args(&vargv(&[
            "--iterations",
            "50",
            "--seed",
            "7",
            "--jobs",
            "4",
            "--hw",
            "--max-cycles",
            "123456",
            "--target-hits",
            "3",
            "--verbose",
        ]))
        .unwrap();
        assert_eq!(o.iterations, 50);
        assert_eq!(o.seed, Some(7));
        assert_eq!(o.jobs, 4);
        assert!(o.hw);
        assert_eq!(o.max_cycles, 123_456);
        assert_eq!(o.target_hits, 3);
        assert!(o.verbose);
    }

    #[test]
    fn vector_no_hw_clears_hw_last_wins() {
        assert!(!parse_vector_fuzz_args(&vargv(&["--hw", "--no-hw"])).unwrap().hw);
    }

    #[test]
    fn vector_report_and_replay() {
        let o = parse_vector_fuzz_args(&vargv(&["--report"])).unwrap();
        assert!(o.report_only);
        let o = parse_vector_fuzz_args(&vargv(&["--replay", "/some/dir"])).unwrap();
        assert_eq!(o.replay, Some(PathBuf::from("/some/dir")));
    }

    #[test]
    fn vector_unknown_flag_and_missing_value_are_errors() {
        assert!(parse_vector_fuzz_args(&vargv(&["--bogus"])).is_err());
        assert!(parse_vector_fuzz_args(&vargv(&["--replay"])).is_err());
        assert!(parse_vector_fuzz_args(&vargv(&["--iterations", "lots"])).is_err());
    }

    #[test]
    fn defaults_when_only_subcommand() {
        let o = parse_fuzz_args(&argv(&[])).unwrap();
        assert_eq!(o.fuzz_iterations, 0);
        assert_eq!(o.fuzz_seed, None);
        assert!(!o.hw);
        assert!(!o.trace_sweep);
        assert_eq!(o.trace_sweep_reps, 5);
        assert!(!o.verbose);
    }

    #[test]
    fn parses_iterations_and_seed() {
        let o = parse_fuzz_args(&argv(&["--iterations", "100", "--seed", "42"])).unwrap();
        assert_eq!(o.fuzz_iterations, 100);
        assert_eq!(o.fuzz_seed, Some(42));
    }

    #[test]
    fn hw_flag_sets_hw_and_no_hw_clears_it_last_wins() {
        assert!(parse_fuzz_args(&argv(&["--hw"])).unwrap().hw);
        assert!(!parse_fuzz_args(&argv(&["--hw", "--no-hw"])).unwrap().hw);
    }

    #[test]
    fn parses_jobs_max_cycles_trace_sweep() {
        let o = parse_fuzz_args(&argv(&[
            "--jobs",
            "4",
            "--max-cycles",
            "500000",
            "--trace-sweep",
            "--trace-sweep-reps",
            "3",
        ]))
        .unwrap();
        assert_eq!(o.jobs, 4);
        assert_eq!(o.max_cycles, 500_000);
        assert!(o.trace_sweep);
        assert_eq!(o.trace_sweep_reps, 3);
    }

    #[test]
    fn unknown_flag_is_an_error() {
        assert!(parse_fuzz_args(&argv(&["--bogus"])).is_err());
    }

    #[test]
    fn missing_value_is_an_error() {
        assert!(parse_fuzz_args(&argv(&["--iterations"])).is_err());
    }

    #[test]
    fn non_numeric_value_is_an_error() {
        assert!(parse_fuzz_args(&argv(&["--iterations", "lots"])).is_err());
    }
}
