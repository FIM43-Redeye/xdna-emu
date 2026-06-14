//! Efficient trace comparison: HW vs EMU.
//!
//! Consumes per-side events JSON produced by tools/parse-trace.py.  The
//! decoder backend that produced those JSONs is opaque to this tool --
//! the in-tree tools/trace_decoder package and mlir-aie's parse_trace
//! emit the same record shape.  All binary-level work happens upstream;
//! this tool only does the structural comparison.
//!
//! Usage:
//!   trace-compare --hw hw.events.json --emu emu.events.json [--events-json slot-names.json] [-o report.txt]
//!   trace-compare --sweep /path/to/sweep-dir [-o report.txt]
//!   trace-compare --sweep /path/to/sweep-dir --extended [-o report.txt]

use std::path::Path;
use std::process;

use xdna_emu::trace::compare::{self, AnalysisOptions, EventsConfig};
use xdna_emu::trace::stages;
use xdna_emu::trace::stochastic;

fn usage() -> ! {
    eprintln!("Usage:");
    eprintln!("  trace-compare --hw <events.json> --emu <events.json> \\");
    eprintln!("                [--events-json <slot-names.json>] [-o <file>]");
    eprintln!("  trace-compare --sweep <dir> [-o <file>]");
    eprintln!("  trace-compare --hw-dir <dir-of-run_*.json> --emu <events.json> \\");
    eprintln!("                [--band-sigma <k=2.0>] [--remap-columns]");
    eprintln!();
    eprintln!("  Stochastic-aware mode (--hw-dir / --hw-runs a,b,c) compares one EMU");
    eprintln!("  capture against a HW *distribution* (N captures), with a per-event");
    eprintln!("  tolerance band (mean +/- k*std) derived from HW run-to-run variance.");
    eprintln!();
    eprintln!("  The --hw/--emu inputs are events JSON produced by");
    eprintln!("  tools/parse-trace.py; the optional --events-json is a legacy");
    eprintln!("  aiecc slot-name override (mlir-aie events.json format).");
    eprintln!();
    eprintln!("Extended analysis (appended after standard report):");
    eprintln!("  --extended       Enable all extended analyses");
    eprintln!("  --iterations     Per-iteration period breakdown for recurring events");
    eprintln!("  --stalls         Stall attribution (level stalls -> resolving events)");
    eprintln!("  --cross-tile     Cross-tile event correlation (edge-to-edge pairing)");
    eprintln!("  --remap-columns  Normalize physical cols to logical 0-indexed");
    eprintln!("  --pc-anchored    PC-set/multiset diff + perfcnt cycle bands (mode-1 traces)");
    eprintln!("  --stages         Stage decomposition table (soc-based per-stage HW vs EMU");
    eprintln!("                   deltas; requires --hw/--emu, or --emu alone for single-side)");
    process::exit(1);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut hw_path: Option<String> = None;
    let mut emu_path: Option<String> = None;
    let mut sweep_path: Option<String> = None;
    let mut events_json_path: Option<String> = None;
    let mut output_path: Option<String> = None;
    let mut opts = AnalysisOptions::default();
    let mut stages_mode = false;
    let mut hw_dir: Option<String> = None;
    let mut hw_runs_list: Option<String> = None;
    let mut band_sigma: f64 = 2.0;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--hw" => {
                i += 1;
                hw_path = Some(args.get(i).unwrap_or_else(|| usage()).clone());
            }
            "--emu" => {
                i += 1;
                emu_path = Some(args.get(i).unwrap_or_else(|| usage()).clone());
            }
            "--sweep" => {
                i += 1;
                sweep_path = Some(args.get(i).unwrap_or_else(|| usage()).clone());
            }
            "--events-json" => {
                i += 1;
                events_json_path = Some(args.get(i).unwrap_or_else(|| usage()).clone());
            }
            "-o" | "--output" => {
                i += 1;
                output_path = Some(args.get(i).unwrap_or_else(|| usage()).clone());
            }
            "--extended" => {
                // OR-in the extended analyses rather than replacing the whole
                // struct, so flags like --remap-columns parsed earlier on the
                // same command line aren't silently clobbered.
                opts.iterations = true;
                opts.stalls = true;
                opts.cross_tile = true;
            }
            "--iterations" => {
                opts.iterations = true;
            }
            "--stalls" => {
                opts.stalls = true;
            }
            "--cross-tile" => {
                opts.cross_tile = true;
            }
            "--remap-columns" => {
                opts.remap_columns = true;
            }
            "--pc-anchored" => {
                opts.pc_anchored = true;
            }
            "--stages" => {
                stages_mode = true;
            }
            "--hw-dir" => {
                i += 1;
                hw_dir = Some(args.get(i).unwrap_or_else(|| usage()).clone());
            }
            "--hw-runs" => {
                i += 1;
                hw_runs_list = Some(args.get(i).unwrap_or_else(|| usage()).clone());
            }
            "--band-sigma" => {
                i += 1;
                band_sigma = args.get(i).unwrap_or_else(|| usage()).parse().unwrap_or_else(|_| {
                    eprintln!("Error: --band-sigma expects a number");
                    process::exit(1);
                });
            }
            "--help" | "-h" => usage(),
            other => {
                eprintln!("Unknown argument: {}", other);
                usage();
            }
        }
        i += 1;
    }

    // Validate: either --sweep or --hw+--emu. Stages mode is the one
    // exception that accepts --emu alone (HW-side optional for single-side
    // probing during campaign prep).
    if sweep_path.is_some() && hw_path.is_some() {
        eprintln!("Error: --sweep and --hw are mutually exclusive");
        process::exit(1);
    }
    if hw_path.is_some() && emu_path.is_none() {
        eprintln!("Error: --emu required when using --hw");
        process::exit(1);
    }
    let stochastic_mode = hw_dir.is_some() || hw_runs_list.is_some();
    if sweep_path.is_none() && hw_path.is_none() && !stochastic_mode && !(stages_mode && emu_path.is_some()) {
        eprintln!("Error: either --sweep, --hw/--emu, or --hw-dir/--emu required");
        usage();
    }

    // Stages mode is a standalone analysis (does not invoke the standard
    // sequence comparator). Reads events JSON on each side using soc,
    // reports per-stage cycle deltas, exits.
    if stages_mode {
        if sweep_path.is_some() {
            eprintln!("Error: --stages is incompatible with --sweep");
            process::exit(1);
        }
        let stage_rows = match (hw_path.as_deref(), emu_path.as_deref()) {
            (Some(hw), Some(emu)) => {
                stages::compute_stages_from_paths(Path::new(hw), Path::new(emu), stages::default_stages())
            }
            (None, Some(emu)) => {
                stages::compute_stages_single_from_path(Path::new(emu), stages::default_stages())
            }
            _ => {
                eprintln!("Error: --stages requires --emu (and optionally --hw)");
                process::exit(1);
            }
        };
        let stage_rows = stage_rows.unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            process::exit(1);
        });
        let report = stages::format_stages_report(&stage_rows);
        print!("{}", report);
        if let Some(out) = output_path {
            let path = Path::new(&out);
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Err(e) = std::fs::write(path, &report) {
                eprintln!("Error writing {}: {}", out, e);
                process::exit(1);
            }
            eprintln!("\nReport written to {}", out);
        }
        return;
    }

    // Stochastic-aware mode: --hw-dir/--hw-runs (N HW captures) + --emu (one).
    // Derives a per-event tolerance band from HW run-to-run variance and
    // band-compares the EMU. Standalone path; exits after.
    if hw_dir.is_some() || hw_runs_list.is_some() {
        if hw_path.is_some() || sweep_path.is_some() {
            eprintln!("Error: --hw-dir/--hw-runs is incompatible with --hw/--sweep");
            process::exit(1);
        }
        let emu = emu_path.unwrap_or_else(|| {
            eprintln!("Error: --emu required with --hw-dir/--hw-runs");
            process::exit(1);
        });
        let mut hw_paths: Vec<std::path::PathBuf> = Vec::new();
        if let Some(dir) = hw_dir {
            let entries = std::fs::read_dir(&dir).unwrap_or_else(|e| {
                eprintln!("Error reading --hw-dir {}: {}", dir, e);
                process::exit(1);
            });
            for e in entries.flatten() {
                let p = e.path();
                let is_json = p.extension().map(|x| x == "json").unwrap_or(false);
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                // Accept run_*.json (the capture convention) or any *.json.
                if is_json && (name.starts_with("run_") || name.ends_with(".json")) {
                    hw_paths.push(p);
                }
            }
            hw_paths.sort();
        }
        if let Some(list) = hw_runs_list {
            for s in list.split(',').filter(|s| !s.is_empty()) {
                hw_paths.push(std::path::PathBuf::from(s));
            }
        }
        if hw_paths.is_empty() {
            eprintln!("Error: no HW captures found for --hw-dir/--hw-runs");
            process::exit(1);
        }
        let config = EventsConfig::default();
        let report = match stochastic::compare_stochastic(
            &hw_paths,
            Path::new(&emu),
            &config,
            band_sigma,
            opts.remap_columns,
        ) {
            Ok(r) => stochastic::format_report(&r),
            Err(e) => {
                eprintln!("Error: {}", e);
                process::exit(1);
            }
        };
        print!("{}", report);
        if let Some(out) = output_path {
            let path = Path::new(&out);
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Err(e) = std::fs::write(path, &report) {
                eprintln!("Error writing {}: {}", out, e);
                process::exit(1);
            }
            eprintln!("\nReport written to {}", out);
        }
        return;
    }

    let report = if let Some(sweep) = sweep_path {
        match compare::compare_sweep_dir_with_opts(Path::new(&sweep), &opts) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Error: {}", e);
                process::exit(1);
            }
        }
    } else {
        let hw = hw_path.unwrap();
        let emu = emu_path.unwrap();

        let config = if let Some(ej) = events_json_path {
            let path = Path::new(&ej);
            if path.exists() {
                let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
                    eprintln!("Error reading {}: {}", ej, e);
                    process::exit(1);
                });
                serde_json::from_str(&text).unwrap_or_else(|e| {
                    eprintln!("Error parsing {}: {}", ej, e);
                    process::exit(1);
                })
            } else {
                EventsConfig::default()
            }
        } else {
            EventsConfig::default()
        };

        match compare::compare_batch_with_opts(Path::new(&hw), Path::new(&emu), &config, 0, &opts) {
            Ok(result) => compare::format_report(&[result]),
            Err(e) => {
                eprintln!("Error: {}", e);
                process::exit(1);
            }
        }
    };

    print!("{}", report);

    if let Some(out) = output_path {
        let path = Path::new(&out);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) = std::fs::write(path, &report) {
            eprintln!("Error writing {}: {}", out, e);
            process::exit(1);
        }
        eprintln!("\nReport written to {}", out);
    }
}
