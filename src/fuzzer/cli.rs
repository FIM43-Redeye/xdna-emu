//! CLI argument parsing for the `fuzz` subcommand.
//!
//! Pure mapping from process args to `FuzzOptions`, kept in the library so it
//! is unit-testable (binaries are not). `main.rs` is a thin caller.

use crate::fuzzer::runner::FuzzOptions;

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

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(rest: &[&str]) -> Vec<String> {
        let mut v = vec!["xdna-emu".to_string(), "fuzz".to_string()];
        v.extend(rest.iter().map(|s| s.to_string()));
        v
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
