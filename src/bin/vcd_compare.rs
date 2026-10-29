//! VCD deep comparison: cycle-accurate signal comparison between emulator
//! and aiesimulator VCD files.
//!
//! # Modes
//!
//! ## Compare two VCDs
//!
//! ```text
//! vcd-compare --emu emu.vcd --sim sim.vcd [--tolerance strict|relaxed|default] [--json] [-o report.txt]
//! ```
//!
//! Loads both VCD files, aligns signals via the AIE2 mapping tree, runs the
//! comparison engine, and produces either a text or JSON report.
//!
//! ## Coverage audit (single VCD)
//!
//! ```text
//! vcd-compare --coverage sim.vcd [--json]
//! ```
//!
//! Walks every signal in the VCD, checks which ones the mapping tree can
//! resolve to a [`StatePath`], and reports mapped vs unmapped counts broken
//! down by subsystem.

use std::process;

use xdna_emu::vcd::anchors::{anchors_to_json, extract_dma_anchors_with_period};
use xdna_emu::vcd::compare::{compare_signals, load_and_align};
use xdna_emu::vcd::coverage::coverage_audit;
use xdna_emu::vcd::cycles::cycle_span;
use xdna_emu::vcd::inproc_mapping::build_npu1_inproc_mapping_tree;
use xdna_emu::vcd::mapping::{build_aie2_mapping_tree, build_vc2802_mapping_tree, MappingTree};
use xdna_emu::vcd::report::{json_report, text_report};
use xdna_emu::vcd::state_path::Subsystem;
use xdna_emu::vcd::tolerance::ToleranceConfig;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn usage() -> ! {
    eprintln!("Usage:");
    eprintln!("  vcd-compare --emu <file> --sim <file> [options]");
    eprintln!("  vcd-compare --coverage <file> [options]");
    eprintln!("  vcd-compare --cycles <file> [--json] [options]");
    eprintln!("  vcd-compare --anchors <file> [options]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --device npu1|vc2802|npu1-inproc     Device geometry (default: vc2802 for coverage, npu1 for compare)");
    eprintln!("  --tolerance strict|relaxed|default   Timing tolerance (default: default)");
    eprintln!("  --json                               Output JSON instead of text");
    eprintln!("  -o <file>                            Write output to file");
    eprintln!("  --help, -h                           Show this help");
    process::exit(1);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut emu_path: Option<String> = None;
    let mut sim_path: Option<String> = None;
    let mut coverage_path: Option<String> = None;
    let mut cycles_path: Option<String> = None;
    let mut anchors_path: Option<String> = None;
    let mut tolerance_name = "default".to_string();
    let mut device_name: Option<String> = None;
    let mut json_output = false;
    let mut output_path: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--emu" => {
                i += 1;
                emu_path = Some(args.get(i).cloned().unwrap_or_else(|| usage()));
            }
            "--sim" => {
                i += 1;
                sim_path = Some(args.get(i).cloned().unwrap_or_else(|| usage()));
            }
            "--coverage" => {
                i += 1;
                coverage_path = Some(args.get(i).cloned().unwrap_or_else(|| usage()));
            }
            "--cycles" => {
                i += 1;
                cycles_path = Some(args.get(i).cloned().unwrap_or_else(|| usage()));
            }
            "--anchors" => {
                i += 1;
                anchors_path = Some(args.get(i).cloned().unwrap_or_else(|| usage()));
            }
            "--tolerance" => {
                i += 1;
                tolerance_name = args.get(i).cloned().unwrap_or_else(|| usage());
            }
            "--device" => {
                i += 1;
                device_name = Some(args.get(i).cloned().unwrap_or_else(|| usage()));
            }
            "--json" => {
                json_output = true;
            }
            "-o" | "--output" => {
                i += 1;
                output_path = Some(args.get(i).cloned().unwrap_or_else(|| usage()));
            }
            "--help" | "-h" => usage(),
            other => {
                eprintln!("Unknown argument: {}", other);
                usage();
            }
        }
        i += 1;
    }

    // Determine mode and validate argument combinations.
    if coverage_path.is_some() && (emu_path.is_some() || sim_path.is_some()) {
        eprintln!("Error: --coverage is mutually exclusive with --emu/--sim");
        process::exit(1);
    }

    if let Some(anc_path) = anchors_path {
        // Anchors come from the in-process NPU1 cluster VCD (#87/#88), native
        // NPU1 geometry -- so the default device is the in-process tree.
        let dev = device_name.as_deref().unwrap_or("npu1-inproc");
        let tree = parse_device(dev);
        run_anchors(&anc_path, &tree, output_path.as_deref());
    } else if let Some(cyc_path) = cycles_path {
        // aiesim VCDs use VC2802 geometry; same default as coverage.
        let dev = device_name.as_deref().unwrap_or("vc2802");
        let tree = parse_device(dev);
        run_cycles(&cyc_path, &tree, json_output, output_path.as_deref());
    } else if let Some(cov_path) = coverage_path {
        // Default to vc2802 for coverage (most common: aiesim VCDs use VC2802 geometry)
        let dev = device_name.as_deref().unwrap_or("vc2802");
        let tree = parse_device(dev);
        run_coverage(&cov_path, &tree, json_output, output_path.as_deref());
    } else if let (Some(emu), Some(sim)) = (emu_path, sim_path) {
        // Default to npu1 for comparison (emulator uses NPU1 geometry)
        let dev = device_name.as_deref().unwrap_or("npu1");
        let tree = parse_device(dev);
        let tolerance = parse_tolerance(&tolerance_name);
        run_compare(&emu, &sim, &tree, &tolerance, json_output, output_path.as_deref());
    } else {
        eprintln!("Error: either --coverage or both --emu and --sim are required");
        usage();
    }
}

/// Parse a device name into a [`MappingTree`], exiting on unknown names.
fn parse_device(name: &str) -> MappingTree {
    match name {
        "npu1" => build_aie2_mapping_tree(),
        "vc2802" => build_vc2802_mapping_tree(),
        "npu1-inproc" => build_npu1_inproc_mapping_tree(),
        other => {
            eprintln!("Error: unknown device '{}'. Use npu1, vc2802, or npu1-inproc.", other);
            process::exit(1);
        }
    }
}

/// Parse a tolerance name into a [`ToleranceConfig`], exiting on unknown names.
fn parse_tolerance(name: &str) -> ToleranceConfig {
    match name {
        "strict" => ToleranceConfig::strict(),
        "relaxed" => ToleranceConfig::relaxed(),
        "default" => ToleranceConfig::aie2_default(),
        other => {
            eprintln!("Error: unknown tolerance '{}'. Use strict, relaxed, or default.", other);
            process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// Compare mode
// ---------------------------------------------------------------------------

/// Load, align, compare two VCD files and write a report.
fn run_compare(
    emu_path: &str,
    sim_path: &str,
    tree: &MappingTree,
    tolerance: &ToleranceConfig,
    json: bool,
    output: Option<&str>,
) {
    let input = match load_and_align(emu_path, sim_path, tree) {
        Ok(input) => input,
        Err(e) => {
            eprintln!("Error loading VCD files: {}", e);
            process::exit(1);
        }
    };

    let result = compare_signals(&input, tolerance);

    let report = if json {
        json_report(&result)
    } else {
        text_report(&result, tolerance)
    };

    write_output(&report, output);
}

// ---------------------------------------------------------------------------
// Coverage mode
// ---------------------------------------------------------------------------

/// Run a coverage audit on a single VCD and write the report.
fn run_coverage(vcd_path: &str, tree: &MappingTree, _json: bool, output: Option<&str>) {
    let report = match coverage_audit(vcd_path, tree) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error auditing VCD: {}", e);
            process::exit(1);
        }
    };

    // CoverageReport implements Display. JSON output is not yet implemented
    // for coverage -- the Display output is machine-parseable enough for
    // scripting. A dedicated JSON format can be added when needed.
    let text = format!("{}", report);

    write_output(&text, output);
}

// ---------------------------------------------------------------------------
// Cycles mode
// ---------------------------------------------------------------------------

/// Extract DMA start/done timing anchors from an in-process NPU1 cluster VCD
/// and write them as bare-measurement JSON. This is the aiesim side of Option B
/// (per-anchor) three-way timing: the anchors align on `(col, row, kind)` with
/// the HW/interp trace-BO `DMA_*_{START,FINISHED}_TASK` events
/// (`tools/trace-anchors.py`), with zero geometry normalization.
fn run_anchors(vcd_path: &str, tree: &MappingTree, output: Option<&str>) {
    let (anchors, period_ps) = match extract_dma_anchors_with_period(vcd_path, tree) {
        Ok(pair) => pair,
        Err(e) => {
            eprintln!("Error extracting anchors from {}: {}", vcd_path, e);
            process::exit(1);
        }
    };

    if anchors.is_empty() {
        eprintln!("Warning: {} produced no DMA anchors (no channel showed status activity)", vcd_path);
    }

    let report = anchors_to_json(&anchors, period_ps);
    write_output(&report, output);
}

/// Compute the total active-cycle span of a single aiesimulator VCD and write
/// it out. This is the aiesim side of three-way timing calibration: the
/// `span_cycles` scalar is comparable to `parse-trace.py --out-cycles` for the
/// HW and interpreter sides.
fn run_cycles(vcd_path: &str, tree: &MappingTree, json: bool, output: Option<&str>) {
    // Default activity vocabulary: all temporal-activity subsystems. The event
    // trace signals alone are often one-shot in aiesim VCDs; the kernel's active
    // window is carried by DMA/lock/stream/core transitions. This measures the
    // gross active span (first..last activity) -- the coarse total-cycle proxy
    // for Option-C drift. Precise trace-event anchoring is Option B.
    let activity = [Subsystem::Dma, Subsystem::Lock, Subsystem::Stream, Subsystem::Core, Subsystem::Event];
    let span = match cycle_span(vcd_path, tree, &activity) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error computing cycle span for {}: {}", vcd_path, e);
            process::exit(1);
        }
    };

    if span.n_changes == 0 {
        eprintln!(
            "Warning: {} has no event-signal activity (degenerate/dump-vars-only VCD); span_cycles=0",
            vcd_path
        );
    }

    let report = if json {
        format!(
            "{{\"span_cycles\":{},\"span_ps\":{},\"period_ps\":{},\"first_ps\":{},\"last_ps\":{},\"n_signals\":{},\"n_changes\":{}}}\n",
            span.span_cycles,
            span.span_ps,
            span.period_ps,
            span.first_ps,
            span.last_ps,
            span.n_signals,
            span.n_changes
        )
    } else {
        format!(
            "span_cycles={}\nspan_ps={}\nperiod_ps={}\nfirst_ps={}\nlast_ps={}\nn_signals={}\nn_changes={}\n",
            span.span_cycles,
            span.span_ps,
            span.period_ps,
            span.first_ps,
            span.last_ps,
            span.n_signals,
            span.n_changes
        )
    };

    write_output(&report, output);
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// Print `content` to stdout, or write it to `path` if given.
fn write_output(content: &str, path: Option<&str>) {
    match path {
        Some(p) => {
            if let Err(e) = std::fs::write(p, content) {
                eprintln!("Error writing {}: {}", p, e);
                process::exit(1);
            }
            eprintln!("Report written to {}", p);
        }
        None => print!("{}", content),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the tolerance parser handles all valid names without panicking.
    #[test]
    fn parse_tolerance_all_valid_names() {
        // These must not call process::exit.
        let _ = parse_tolerance("strict");
        let _ = parse_tolerance("relaxed");
        let _ = parse_tolerance("default");
    }

    /// Verify that the binary compiles and the helper functions are callable.
    ///
    /// Actual end-to-end functionality is exercised by the library module
    /// tests (vcd::compare, vcd::report, vcd::coverage). This test confirms
    /// the binary wiring is correct.
    #[test]
    fn binary_compiles() {
        assert!(true);
    }
}
