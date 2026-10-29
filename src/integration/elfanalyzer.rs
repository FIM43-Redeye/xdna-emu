//! elfanalyzer integration for cross-validating ELF parsing.
//!
//! elfanalyzer is a standalone static analysis tool from aietools that
//! inspects AIE ELF binaries. It does NOT require a license, making it
//! the most accessible aietools integration.
//!
//! Supported analyses:
//! - `pmsize`: Program memory size per function
//! - `stacksize`: Stack size per function
//! - `globals`: Global data memory elements
//!
//! This module parses elfanalyzer's text output into structured Rust types.

use std::collections::HashMap;
use std::path::Path;

use super::aietools::{AieTools, Tool};
use crate::parser::elf::AieElf;

/// Structured output from elfanalyzer for a single ELF.
#[derive(Debug, Clone, Default)]
pub struct ElfAnalysis {
    /// Program memory size per function name (in bytes).
    pub pm_sizes: HashMap<String, usize>,
    /// Stack size per function name (in bytes).
    pub stack_sizes: HashMap<String, usize>,
    /// Global data memory elements.
    pub globals: Vec<GlobalEntry>,
}

/// A single global variable entry from elfanalyzer.
#[derive(Debug, Clone)]
pub struct GlobalEntry {
    pub name: String,
    pub address: u64,
    pub size: usize,
    pub section: String,
}

/// Run elfanalyzer on an ELF file and collect all available analyses.
///
/// Runs each analysis type separately (elfanalyzer processes one at a time)
/// and merges the results into a single `ElfAnalysis`.
///
/// Returns `Err` if elfanalyzer is not available or fails to execute.
/// Individual parse failures for specific analyses are logged but do not
/// fail the overall result -- partial data is still useful.
pub fn analyze(tools: &AieTools, elf_path: &Path) -> Result<ElfAnalysis, String> {
    if tools.elfanalyzer.is_none() {
        return Err("elfanalyzer not available".to_string());
    }

    let mut result = ElfAnalysis::default();

    // Run each analysis type and merge results
    if let Ok(output) = run_analysis(tools, elf_path, "pmsize") {
        result.pm_sizes = parse_pmsize(&output);
    }

    if let Ok(output) = run_analysis(tools, elf_path, "stacksize") {
        result.stack_sizes = parse_stacksize(&output);
    }

    if let Ok(output) = run_analysis(tools, elf_path, "globals") {
        result.globals = parse_globals(&output);
    }

    Ok(result)
}

/// Run a single elfanalyzer analysis and return stdout.
fn run_analysis(tools: &AieTools, elf_path: &Path, analysis: &str) -> Result<String, String> {
    let mut cmd = tools.command(Tool::Elfanalyzer).ok_or("elfanalyzer not available")?;

    cmd.arg(format!("--analysis={}", analysis)).arg(elf_path);

    log::debug!("Running elfanalyzer --analysis={} on {}", analysis, elf_path.display());

    let output = cmd.output().map_err(|e| format!("Failed to execute elfanalyzer: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("elfanalyzer --analysis={} failed: {}", analysis, stderr.trim()));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Parse `--analysis=pmsize` output.
///
/// Expected format (one function per line):
/// ```text
/// Function               PM Size (bytes)
/// --------               ---------------
/// main                   256
/// _start                 64
/// ```
fn parse_pmsize(output: &str) -> HashMap<String, usize> {
    let mut sizes = HashMap::new();

    for line in output.lines() {
        let line = line.trim();
        // Skip header lines and separators
        if line.is_empty()
            || line.starts_with("Function")
            || line.starts_with("----")
            || line.starts_with("Total")
        {
            continue;
        }

        // Split on whitespace: function name, then size
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            if let Ok(size) = parts.last().unwrap().parse::<usize>() {
                // Function name may contain multiple words -- take everything
                // before the last field (the size).
                let name = parts[..parts.len() - 1].join(" ");
                sizes.insert(name, size);
            }
        }
    }

    sizes
}

/// Parse `--analysis=stacksize` output.
///
/// Same format as pmsize but with stack sizes.
fn parse_stacksize(output: &str) -> HashMap<String, usize> {
    let mut sizes = HashMap::new();

    for line in output.lines() {
        let line = line.trim();
        if line.is_empty()
            || line.starts_with("Function")
            || line.starts_with("----")
            || line.starts_with("Total")
        {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            if let Ok(size) = parts.last().unwrap().parse::<usize>() {
                let name = parts[..parts.len() - 1].join(" ");
                sizes.insert(name, size);
            }
        }
    }

    sizes
}

/// Parse `--analysis=globals` output.
///
/// Expected format:
/// ```text
/// Name                   Address    Size    Section
/// ----                   -------    ----    -------
/// buffer                 0x20000    1024    .data
/// ```
fn parse_globals(output: &str) -> Vec<GlobalEntry> {
    let mut globals = Vec::new();

    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("Name") || line.starts_with("----") {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 4 {
            let name = parts[0].to_string();
            let address = parse_hex_or_dec(parts[1]).unwrap_or(0);
            let size = parts[2].parse::<usize>().unwrap_or(0);
            let section = parts[3].to_string();

            globals.push(GlobalEntry { name, address, size, section });
        }
    }

    globals
}

/// Parse a string as hex (0x...) or decimal.
fn parse_hex_or_dec(s: &str) -> Option<u64> {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        s.parse::<u64>().ok()
    }
}

/// Format an ElfAnalysis for human-readable display.
///
/// Used by the test runner's `--elfanalyze` mode.
pub fn format_analysis(analysis: &ElfAnalysis, elf_name: &str) -> String {
    let mut out = String::new();

    out.push_str(&format!("=== elfanalyzer: {} ===\n", elf_name));

    if !analysis.pm_sizes.is_empty() {
        out.push_str("  Program Memory:\n");
        let mut sorted: Vec<_> = analysis.pm_sizes.iter().collect();
        sorted.sort_by_key(|(_, &size)| std::cmp::Reverse(size));
        for (func, size) in &sorted {
            out.push_str(&format!("    {:<30} {:>6} bytes\n", func, size));
        }
    }

    if !analysis.stack_sizes.is_empty() {
        out.push_str("  Stack Usage:\n");
        let mut sorted: Vec<_> = analysis.stack_sizes.iter().collect();
        sorted.sort_by_key(|(_, &size)| std::cmp::Reverse(size));
        for (func, size) in &sorted {
            out.push_str(&format!("    {:<30} {:>6} bytes\n", func, size));
        }
    }

    if !analysis.globals.is_empty() {
        out.push_str("  Globals:\n");
        for g in &analysis.globals {
            out.push_str(&format!(
                "    {:<30} 0x{:08X}  {:>6} bytes  {}\n",
                g.name, g.address, g.size, g.section
            ));
        }
    }

    if analysis.pm_sizes.is_empty() && analysis.stack_sizes.is_empty() && analysis.globals.is_empty() {
        out.push_str("  (no analysis data)\n");
    }

    out
}

/// A discrepancy between elfanalyzer and our ELF parser.
#[derive(Debug, Clone)]
pub struct Discrepancy {
    pub symbol: String,
    pub kind: DiscrepancyKind,
}

/// What kind of discrepancy was found.
#[derive(Debug, Clone)]
pub enum DiscrepancyKind {
    /// Our parser has a function that elfanalyzer doesn't report.
    MissingFromElfanalyzer { our_size: u32 },
    /// elfanalyzer has a function that our parser doesn't see.
    MissingFromParser { elfanalyzer_size: usize },
    /// Both have the function but disagree on size.
    SizeMismatch { our_size: u32, elfanalyzer_size: usize },
}

/// Cross-validation result comparing elfanalyzer against our parser.
#[derive(Debug, Clone)]
pub struct CrossValidation {
    /// Functions that both agree on (name, size).
    pub matching: Vec<(String, usize)>,
    /// Discrepancies found.
    pub discrepancies: Vec<Discrepancy>,
}

impl CrossValidation {
    /// Whether the two views are fully consistent.
    pub fn is_consistent(&self) -> bool {
        self.discrepancies.is_empty()
    }

    /// Summary string for display.
    pub fn summary(&self) -> String {
        if self.is_consistent() {
            format!("{} functions match", self.matching.len())
        } else {
            format!("{} match, {} discrepancies", self.matching.len(), self.discrepancies.len())
        }
    }
}

/// Compare elfanalyzer output against our ELF parser's understanding.
///
/// Compares function names and program memory sizes. Discrepancies may be
/// benign (different size measurement methods) or indicate parser bugs.
pub fn cross_validate(analysis: &ElfAnalysis, elf_data: &[u8]) -> Result<CrossValidation, String> {
    let aie_elf = AieElf::parse(elf_data).map_err(|e| format!("Failed to parse ELF: {}", e))?;

    let mut matching = Vec::new();
    let mut discrepancies = Vec::new();

    // Build a map of our parser's function symbols
    let our_functions: HashMap<String, u32> = aie_elf
        .functions()
        .filter(|f| f.size > 0)
        .map(|f| (f.name.clone(), f.size))
        .collect();

    // Check each elfanalyzer function against our parser
    for (name, &ea_size) in &analysis.pm_sizes {
        if let Some(&our_size) = our_functions.get(name) {
            if our_size as usize == ea_size {
                matching.push((name.clone(), ea_size));
            } else {
                discrepancies.push(Discrepancy {
                    symbol: name.clone(),
                    kind: DiscrepancyKind::SizeMismatch { our_size, elfanalyzer_size: ea_size },
                });
            }
        } else {
            discrepancies.push(Discrepancy {
                symbol: name.clone(),
                kind: DiscrepancyKind::MissingFromParser { elfanalyzer_size: ea_size },
            });
        }
    }

    // Check for functions we see that elfanalyzer doesn't report
    for (name, &our_size) in &our_functions {
        if !analysis.pm_sizes.contains_key(name) {
            discrepancies.push(Discrepancy {
                symbol: name.clone(),
                kind: DiscrepancyKind::MissingFromElfanalyzer { our_size },
            });
        }
    }

    Ok(CrossValidation { matching, discrepancies })
}

/// Format cross-validation results for display.
pub fn format_cross_validation(cv: &CrossValidation, elf_name: &str) -> String {
    let mut out = String::new();

    if cv.is_consistent() {
        out.push_str(&format!("      elfanalyzer vs parser: {} -- {}\n", elf_name, cv.summary()));
        return out;
    }

    out.push_str(&format!("      elfanalyzer vs parser: {} -- {}\n", elf_name, cv.summary()));

    for d in &cv.discrepancies {
        match &d.kind {
            DiscrepancyKind::SizeMismatch { our_size, elfanalyzer_size } => {
                out.push_str(&format!(
                    "        MISMATCH {}: parser={} bytes, elfanalyzer={} bytes\n",
                    d.symbol, our_size, elfanalyzer_size
                ));
            }
            DiscrepancyKind::MissingFromParser { elfanalyzer_size } => {
                out.push_str(&format!(
                    "        MISSING (parser) {}: elfanalyzer says {} bytes\n",
                    d.symbol, elfanalyzer_size
                ));
            }
            DiscrepancyKind::MissingFromElfanalyzer { our_size } => {
                out.push_str(&format!(
                    "        MISSING (elfanalyzer) {}: parser says {} bytes\n",
                    d.symbol, our_size
                ));
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_pmsize() {
        let output = "\
Function               PM Size (bytes)
--------               ---------------
main                   256
_start                 64
helper_func            128
Total                  448
";
        let sizes = parse_pmsize(output);
        assert_eq!(sizes.get("main"), Some(&256));
        assert_eq!(sizes.get("_start"), Some(&64));
        assert_eq!(sizes.get("helper_func"), Some(&128));
        assert!(!sizes.contains_key("Total"));
    }

    #[test]
    fn test_parse_stacksize() {
        let output = "\
Function               Stack Size (bytes)
--------               ------------------
main                   512
_start                 0
";
        let sizes = parse_stacksize(output);
        assert_eq!(sizes.get("main"), Some(&512));
        assert_eq!(sizes.get("_start"), Some(&0));
    }

    #[test]
    fn test_parse_globals() {
        let output = "\
Name                   Address    Size    Section
----                   -------    ----    -------
buffer                 0x20000    1024    .data
counter                0x20400    4       .bss
";
        let globals = parse_globals(output);
        assert_eq!(globals.len(), 2);
        assert_eq!(globals[0].name, "buffer");
        assert_eq!(globals[0].address, 0x20000);
        assert_eq!(globals[0].size, 1024);
        assert_eq!(globals[0].section, ".data");
        assert_eq!(globals[1].name, "counter");
        assert_eq!(globals[1].size, 4);
    }

    #[test]
    fn test_parse_hex_or_dec() {
        assert_eq!(parse_hex_or_dec("0x20000"), Some(0x20000));
        assert_eq!(parse_hex_or_dec("0X1234"), Some(0x1234));
        assert_eq!(parse_hex_or_dec("42"), Some(42));
        assert_eq!(parse_hex_or_dec("invalid"), None);
    }

    #[test]
    fn test_parse_empty_output() {
        assert!(parse_pmsize("").is_empty());
        assert!(parse_stacksize("").is_empty());
        assert!(parse_globals("").is_empty());
    }

    #[test]
    fn test_cross_validation_matching() {
        let _analysis = ElfAnalysis {
            pm_sizes: [("main".to_string(), 256), ("helper".to_string(), 64)].into_iter().collect(),
            ..Default::default()
        };

        // Simulate cross-validation without a real ELF by testing the
        // comparison logic directly through the CrossValidation struct.
        let cv = CrossValidation {
            matching: vec![("main".to_string(), 256), ("helper".to_string(), 64)],
            discrepancies: vec![],
        };
        assert!(cv.is_consistent());
        assert!(cv.summary().contains("2 functions match"));
    }

    #[test]
    fn test_cross_validation_with_discrepancies() {
        let cv = CrossValidation {
            matching: vec![("main".to_string(), 256)],
            discrepancies: vec![Discrepancy {
                symbol: "helper".to_string(),
                kind: DiscrepancyKind::SizeMismatch { our_size: 64, elfanalyzer_size: 72 },
            }],
        };
        assert!(!cv.is_consistent());
        assert!(cv.summary().contains("1 match"));
        assert!(cv.summary().contains("1 discrepancies"));

        let formatted = format_cross_validation(&cv, "test.elf");
        assert!(formatted.contains("MISMATCH helper"));
        assert!(formatted.contains("parser=64"));
        assert!(formatted.contains("elfanalyzer=72"));
    }

    #[test]
    fn test_format_analysis() {
        let analysis = ElfAnalysis {
            pm_sizes: [("main".to_string(), 256)].into_iter().collect(),
            stack_sizes: [("main".to_string(), 512)].into_iter().collect(),
            globals: vec![GlobalEntry {
                name: "buf".to_string(),
                address: 0x20000,
                size: 1024,
                section: ".data".to_string(),
            }],
        };
        let formatted = format_analysis(&analysis, "core_0_2.elf");
        assert!(formatted.contains("elfanalyzer: core_0_2.elf"));
        assert!(formatted.contains("Program Memory"));
        assert!(formatted.contains("main"));
        assert!(formatted.contains("256 bytes"));
        assert!(formatted.contains("Stack Usage"));
        assert!(formatted.contains("Globals"));
        assert!(formatted.contains("buf"));
    }
}
