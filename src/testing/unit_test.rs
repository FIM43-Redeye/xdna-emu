//! Discovery and representation of mlir-aie chess compiler unit tests.
//!
//! These tests live in `mlir-aie/test/unit_tests/chess_compiler_tests_aie2/`
//! and follow a different model from the NPU/xclbin tests: they use direct
//! tile memory writes and lock-based synchronization instead of DDR, shim
//! DMA, and CDO configuration. Each test has:
//!
//! - An MLIR file (`aie.mlir` or `aie_row.mlir`) describing the tile array
//! - A `test.cpp` host harness that writes buffers, releases locks, and
//!   checks results via `mlir_aie_check()`
//! - Optional `kernel.cc`/`kernel.h` for precompiled core functions
//!
//! The build flow compiles everything via `aiecc.py --aiesim --xchesscc`
//! into a `.prj/` directory, then aiesimulator executes it. ps.so (the
//! compiled test.cpp) handles all I/O -- no separate input/output files.
//!
//! # Example
//!
//! ```ignore
//! let tests = discover(&mlir_aie_path);
//! for test in &tests {
//!     println!("{}: mlir={}", test.name, test.mlir_file.display());
//! }
//! ```

use std::path::{Path, PathBuf};

/// A single unit test from the chess_compiler_tests_aie2 suite.
#[derive(Debug, Clone)]
pub struct UnitTest {
    /// Test name (directory name, e.g. "01_precompiled_core_function").
    pub name: String,
    /// Absolute path to the test source directory.
    pub source_dir: PathBuf,
    /// Path to the MLIR file (aie.mlir or aie_row.mlir).
    pub mlir_file: PathBuf,
    /// Path to the test.cpp host harness.
    pub test_cpp: PathBuf,
    /// Optional kernel source files (kernel.cc, kernel.h).
    pub kernel_sources: Vec<PathBuf>,
    /// If set, the test should be skipped with this reason.
    pub skip_reason: Option<String>,
    /// Build steps extracted from `// RUN:` lines in the MLIR file.
    ///
    /// These are the raw RUN line text (after stripping the `// RUN:` prefix),
    /// filtered to exclude simulation commands (those containing `aiesim.sh`).
    /// Each entry is a shell command with lit substitution variables
    /// (`%PYTHON`, `%s`, `%S`, `%test_lib_flags`) still unexpanded.
    pub build_steps: Vec<String>,
}

/// Result of building a unit test via aiecc.py.
#[derive(Debug)]
pub struct UnitTestBuildResult {
    /// Path to the .prj directory containing sim/ artifacts.
    pub prj_dir: PathBuf,
    /// Build output (stdout + stderr) for diagnostics.
    pub build_log: String,
}

/// Discover unit tests in the chess_compiler_tests_aie2 directory.
///
/// Scans subdirectories of `{mlir_aie_path}/test/unit_tests/chess_compiler_tests_aie2/`.
/// A valid test must have both an MLIR file and a `test.cpp`. Tests with
/// XFAIL annotations are marked for skipping (unless an alternative MLIR
/// file exists). Tests without `test.cpp` (compile-only) are excluded.
///
/// Returns tests sorted by name for deterministic ordering.
pub fn discover(mlir_aie_path: &Path) -> Vec<UnitTest> {
    let test_root = mlir_aie_path
        .join("test/unit_tests/chess_compiler_tests_aie2");

    if !test_root.is_dir() {
        log::warn!(
            "Unit test directory not found: {}",
            test_root.display()
        );
        return Vec::new();
    }

    let mut tests = Vec::new();

    let entries = match std::fs::read_dir(&test_root) {
        Ok(e) => e,
        Err(e) => {
            log::warn!("Failed to read unit test directory: {}", e);
            return Vec::new();
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        // test.cpp is required (compile-only tests like 00 are excluded)
        let test_cpp = path.join("test.cpp");
        if !test_cpp.exists() {
            continue;
        }

        // Find the best MLIR file. Prefer aie.mlir, but if it has XFAIL
        // and aie_row.mlir exists, use aie_row.mlir instead.
        // The chosen file's RUN lines determine the build steps.
        let aie_mlir = path.join("aie.mlir");
        let aie_row_mlir = path.join("aie_row.mlir");

        let (mlir_file, skip_reason, run_lines) = if aie_mlir.exists() {
            let ann = parse_mlir_annotations(&aie_mlir);
            if ann.xfail {
                // Try the alternative
                if aie_row_mlir.exists() {
                    let alt = parse_mlir_annotations(&aie_row_mlir);
                    if alt.xfail {
                        (aie_row_mlir, Some("XFAIL in both aie.mlir and aie_row.mlir".to_string()), alt.run_lines)
                    } else if alt.fixme {
                        (aie_row_mlir, Some(format!("FIXME: {}", alt.fixme_reason)), alt.run_lines)
                    } else {
                        (aie_row_mlir, None, alt.run_lines)
                    }
                } else {
                    (aie_mlir, Some("XFAIL".to_string()), ann.run_lines)
                }
            } else if ann.fixme {
                (aie_mlir, Some(format!("FIXME: {}", ann.fixme_reason)), ann.run_lines)
            } else {
                (aie_mlir, None, ann.run_lines)
            }
        } else if aie_row_mlir.exists() {
            let ann = parse_mlir_annotations(&aie_row_mlir);
            if ann.xfail {
                (aie_row_mlir, Some("XFAIL".to_string()), ann.run_lines)
            } else if ann.fixme {
                (aie_row_mlir, Some(format!("FIXME: {}", ann.fixme_reason)), ann.run_lines)
            } else {
                (aie_row_mlir, None, ann.run_lines)
            }
        } else {
            // No MLIR file at all
            continue;
        };

        // Build steps: RUN lines excluding simulation commands.
        // Simulation (aiesim.sh) is handled separately by the runner.
        let build_steps: Vec<String> = run_lines.into_iter()
            .filter(|line| !line.contains("aiesim.sh"))
            .collect();

        // Collect kernel source files
        let mut kernel_sources = Vec::new();
        for filename in &["kernel.cc", "kernel.h"] {
            let p = path.join(filename);
            if p.exists() {
                kernel_sources.push(p);
            }
        }

        tests.push(UnitTest {
            name,
            source_dir: path,
            mlir_file,
            test_cpp,
            kernel_sources,
            skip_reason,
            build_steps,
        });
    }

    tests.sort_by(|a, b| a.name.cmp(&b.name));
    tests
}

/// Annotations parsed from an MLIR file's comment header.
struct MlirAnnotations {
    /// Test is expected to fail (XFAIL: in a comment).
    xfail: bool,
    /// Test has a FIXME annotation (often means it hangs or is broken).
    fixme: bool,
    /// The FIXME reason text, if any.
    fixme_reason: String,
    /// Raw `// RUN:` lines collected from the file (prefix stripped, trimmed).
    ///
    /// These encode the exact build and simulation commands that upstream
    /// intended, in the same format LLVM's `lit` test runner uses.
    run_lines: Vec<String>,
}

/// Parse LLVM lit-style annotations from an MLIR file.
///
/// Collects:
/// - `// RUN:` lines -- build and simulation commands (from entire file)
/// - `// XFAIL:` -- test expected to fail (first 30 lines only)
/// - `// FIXME` -- test has known issues (first 30 lines only)
/// - `// REQUIRES:` -- required features (informational only for now)
///
/// RUN lines are scanned from the entire file because LLVM lit allows them
/// anywhere. XFAIL and FIXME are only meaningful in the header.
fn parse_mlir_annotations(path: &Path) -> MlirAnnotations {
    let mut annotations = MlirAnnotations {
        xfail: false,
        fixme: false,
        fixme_reason: String::new(),
        run_lines: Vec::new(),
    };

    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return annotations,
    };

    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();

        // RUN lines: collect from entire file
        if let Some(rest) = trimmed.strip_prefix("// RUN:") {
            annotations.run_lines.push(rest.trim().to_string());
        }

        // XFAIL and FIXME: only check first 30 lines
        if idx < 30 {
            if trimmed.contains("XFAIL:") || trimmed.contains("XFAIL :") {
                annotations.xfail = true;
            }

            if trimmed.contains("FIXME") {
                annotations.fixme = true;
                if let Some(pos) = trimmed.find("FIXME") {
                    let after = trimmed[pos + 5..].trim();
                    let reason = after.trim_start_matches(':').trim();
                    if !reason.is_empty() {
                        annotations.fixme_reason = reason.to_string();
                    }
                }
            }
        }
    }

    annotations
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Create a unique temporary directory for tests.
    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join("xdna_emu_unit_test_tests")
            .join(name);
        // Clean up from any previous run
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_parse_annotations_xfail() {
        let dir = test_dir("xfail");
        let mlir = dir.join("test.mlir");
        let mut f = std::fs::File::create(&mlir).unwrap();
        writeln!(f, "// REQUIRES: valid_xchess_license").unwrap();
        writeln!(f, "// XFAIL: *").unwrap();
        writeln!(f, "module {{}}").unwrap();

        let ann = parse_mlir_annotations(&mlir);
        assert!(ann.xfail);
        assert!(!ann.fixme);
        assert!(ann.run_lines.is_empty());
    }

    #[test]
    fn test_parse_annotations_fixme() {
        let dir = test_dir("fixme");
        let mlir = dir.join("test.mlir");
        let mut f = std::fs::File::create(&mlir).unwrap();
        writeln!(f, "// FIXME: hangs in simulation").unwrap();
        writeln!(f, "module {{}}").unwrap();

        let ann = parse_mlir_annotations(&mlir);
        assert!(!ann.xfail);
        assert!(ann.fixme);
        assert_eq!(ann.fixme_reason, "hangs in simulation");
    }

    #[test]
    fn test_parse_annotations_clean() {
        let dir = test_dir("clean");
        let mlir = dir.join("test.mlir");
        let mut f = std::fs::File::create(&mlir).unwrap();
        writeln!(f, "// REQUIRES: valid_xchess_license, peano").unwrap();
        writeln!(f, "// RUN: %PYTHON aiecc.py %s").unwrap();
        writeln!(f, "module {{}}").unwrap();

        let ann = parse_mlir_annotations(&mlir);
        assert!(!ann.xfail);
        assert!(!ann.fixme);
        assert_eq!(ann.run_lines.len(), 1);
        assert_eq!(ann.run_lines[0], "%PYTHON aiecc.py %s");
    }

    #[test]
    fn test_parse_annotations_run_lines() {
        let dir = test_dir("run_lines");
        let mlir = dir.join("test.mlir");
        let mut f = std::fs::File::create(&mlir).unwrap();
        writeln!(f, "// REQUIRES: valid_xchess_license").unwrap();
        writeln!(f, "// RUN: xchesscc_wrapper aie2 -c %S/kernel.cc").unwrap();
        writeln!(f, "// RUN: %PYTHON aiecc.py --aiesim --xchesscc %s %test_lib_flags %S/test.cpp").unwrap();
        writeln!(f, "// RUN: aie.mlir.prj/aiesim.sh | FileCheck %s").unwrap();
        writeln!(f, "module {{}}").unwrap();

        let ann = parse_mlir_annotations(&mlir);
        assert_eq!(ann.run_lines.len(), 3);
        assert!(ann.run_lines[0].contains("xchesscc_wrapper"));
        assert!(ann.run_lines[1].contains("aiecc.py"));
        assert!(ann.run_lines[2].contains("aiesim.sh"));
    }

    #[test]
    fn test_parse_annotations_disabled_run_not_collected() {
        let dir = test_dir("disabled_run");
        let mlir = dir.join("test.mlir");
        let mut f = std::fs::File::create(&mlir).unwrap();
        // These are NOT valid RUN lines (disabled variants used in lit)
        writeln!(f, "// UN: this should not be collected").unwrap();
        writeln!(f, "// RUNX: this should not be collected").unwrap();
        writeln!(f, "//RUN: no space after //").unwrap();
        writeln!(f, "// RUN: this one IS valid").unwrap();
        writeln!(f, "module {{}}").unwrap();

        let ann = parse_mlir_annotations(&mlir);
        assert_eq!(ann.run_lines.len(), 1);
        assert_eq!(ann.run_lines[0], "this one IS valid");
    }

    #[test]
    fn test_parse_annotations_mixed_xfail_and_run() {
        let dir = test_dir("mixed");
        let mlir = dir.join("test.mlir");
        let mut f = std::fs::File::create(&mlir).unwrap();
        writeln!(f, "// XFAIL: *").unwrap();
        writeln!(f, "// RUN: %PYTHON aiecc.py %s").unwrap();
        writeln!(f, "// RUN: aie.mlir.prj/aiesim.sh | FileCheck %s").unwrap();
        writeln!(f, "module {{}}").unwrap();

        let ann = parse_mlir_annotations(&mlir);
        assert!(ann.xfail);
        assert_eq!(ann.run_lines.len(), 2);
    }

    #[test]
    fn test_discover_empty_dir() {
        let dir = test_dir("empty");
        let tests = discover(&dir);
        assert!(tests.is_empty());
    }

    #[test]
    fn test_discover_skips_no_test_cpp() {
        let dir = test_dir("no_cpp");
        let test_root = dir
            .join("test/unit_tests/chess_compiler_tests_aie2/00_itsalive");
        std::fs::create_dir_all(&test_root).unwrap();
        std::fs::write(test_root.join("aie.mlir"), "module {}").unwrap();
        // No test.cpp -- should be excluded

        let tests = discover(&dir);
        assert!(tests.is_empty());
    }

    #[test]
    fn test_discover_finds_valid_test_with_build_steps() {
        let dir = test_dir("valid");
        let test_root = dir
            .join("test/unit_tests/chess_compiler_tests_aie2/01_test");
        std::fs::create_dir_all(&test_root).unwrap();
        std::fs::write(
            test_root.join("aie.mlir"),
            "// RUN: xchesscc_wrapper aie2 -c %S/kernel.cc\n\
             // RUN: %PYTHON aiecc.py --aiesim --xchesscc %s %test_lib_flags %S/test.cpp\n\
             // RUN: aie.mlir.prj/aiesim.sh | FileCheck %s\n\
             module {}\n",
        ).unwrap();
        std::fs::write(test_root.join("test.cpp"), "int main() {}").unwrap();
        std::fs::write(test_root.join("kernel.cc"), "void f() {}").unwrap();

        let tests = discover(&dir);
        assert_eq!(tests.len(), 1);
        assert_eq!(tests[0].name, "01_test");
        assert!(tests[0].skip_reason.is_none());
        assert_eq!(tests[0].kernel_sources.len(), 1);
        // Build steps: aiesim.sh line is filtered out, 2 remain
        assert_eq!(tests[0].build_steps.len(), 2);
        assert!(tests[0].build_steps[0].contains("xchesscc_wrapper"));
        assert!(tests[0].build_steps[1].contains("aiecc.py"));
    }

    #[test]
    fn test_discover_build_steps_from_aie_row_on_xfail() {
        let dir = test_dir("xfail_fallback");
        let test_root = dir
            .join("test/unit_tests/chess_compiler_tests_aie2/04_shared");
        std::fs::create_dir_all(&test_root).unwrap();
        std::fs::write(
            test_root.join("aie.mlir"),
            "// XFAIL: *\n\
             // RUN: %PYTHON aiecc.py --xchesscc %s\n\
             module {}\n",
        ).unwrap();
        std::fs::write(
            test_root.join("aie_row.mlir"),
            "// RUN: %PYTHON aiecc.py --aiesim --xchesscc %s %test_lib_flags %S/test.cpp\n\
             // RUN: aie_row.mlir.prj/aiesim.sh | FileCheck %s\n\
             module {}\n",
        ).unwrap();
        std::fs::write(test_root.join("test.cpp"), "int main() {}").unwrap();

        let tests = discover(&dir);
        assert_eq!(tests.len(), 1);
        assert!(tests[0].skip_reason.is_none());
        assert!(tests[0].mlir_file.ends_with("aie_row.mlir"));
        // Build steps come from aie_row.mlir (chosen file), not aie.mlir
        assert_eq!(tests[0].build_steps.len(), 1);
        assert!(tests[0].build_steps[0].contains("--aiesim"));
    }

    #[test]
    fn test_discover_bcf_pattern_build_steps() {
        let dir = test_dir("bcf_pattern");
        let test_root = dir
            .join("test/unit_tests/chess_compiler_tests_aie2/02_precompiled");
        std::fs::create_dir_all(&test_root).unwrap();
        std::fs::write(
            test_root.join("aie.mlir"),
            "// RUN: %PYTHON aiecc.py --aiesim --xchesscc --xbridge --no-compile-host %s %test_lib_flags %S/test.cpp\n\
             // RUN: xchesscc_wrapper aie2 +l aie.mlir.prj/main_core_1_3.bcf %S/kernel.cc -o custom_1_3.elf\n\
             // RUN: aie.mlir.prj/aiesim.sh | FileCheck %s\n\
             module {}\n",
        ).unwrap();
        std::fs::write(test_root.join("test.cpp"), "int main() {}").unwrap();
        std::fs::write(test_root.join("kernel.cc"), "void f() {}").unwrap();

        let tests = discover(&dir);
        assert_eq!(tests.len(), 1);
        // BCF pattern: aiecc.py first, then xchesscc with BCF (reverse of pre-compile)
        assert_eq!(tests[0].build_steps.len(), 2);
        assert!(tests[0].build_steps[0].contains("aiecc.py"));
        assert!(tests[0].build_steps[1].contains("+l"));
        assert!(tests[0].build_steps[1].contains("main_core_1_3.bcf"));
    }
}
