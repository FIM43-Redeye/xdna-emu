//! Source-driven discovery and representation of mlir-aie NPU/xclbin tests.
//!
//! These tests live in `mlir-aie/test/npu-xrt/` and follow the LLVM lit test
//! format: each test directory contains a `run.lit` or `aie2.py` file with
//! `// RUN:` or `# RUN:` lines that encode the build recipe.
//!
//! Unlike build-tree discovery (which found only pre-built xclbins), source-
//! driven discovery finds ALL tests and builds them from source. This covers:
//! - Static MLIR tests (`aie.mlir` + `run.lit`)
//! - Python-generated MLIR tests (`aie2.py` generates MLIR at build time)
//! - Subdirectory tests (`core_dmas/writebd/`, `dynamic_object_fifo/ping_pong/`)
//!
//! # Example
//!
//! ```ignore
//! let tests = discover(&mlir_aie_path);
//! for test in &tests {
//!     println!("{}: {} build steps, requires: {:?}",
//!         test.name, test.build_steps.len(), test.requires);
//! }
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::test_cpp_parser::BufferSpec;
use super::native_hw::TestCppPattern;

/// A single NPU test discovered from the mlir-aie source tree.
///
/// Represents everything needed to build and run a test, before any
/// compilation has happened. The build phase consumes this to produce
/// xclbin artifacts that feed into the existing `XclbinSuite` pipeline.
#[derive(Debug, Clone)]
pub struct NpuTestSource {
    /// Test name, relative to `test/npu-xrt/`.
    /// Flat tests: e.g. "add_one_using_dma"
    /// Nested tests: e.g. "core_dmas/dma_configure_task_lock"
    pub name: String,
    /// Absolute path to the test source directory.
    pub source_dir: PathBuf,
    /// File that contains the RUN lines (run.lit, aie2.py, etc.).
    pub entry_file: PathBuf,
    /// Build-only RUN lines (execution commands filtered out).
    /// Lit substitution variables (%S, %s, %python, etc.) are unexpanded.
    pub build_steps: Vec<String>,
    /// Features required by this test (from REQUIRES: line).
    /// e.g. ["ryzen_ai", "chess"], ["ryzen_ai_npu2", "peano"]
    pub requires: Vec<String>,
    /// Test is expected to fail (XFAIL: annotation present).
    pub xfail: bool,
    /// Buffer metadata parsed from test.cpp (sizes, types, group IDs, input patterns).
    pub buffer_spec: Option<BufferSpec>,
    /// Override: skip this test with the given reason.
    pub skip_reason: Option<String>,
    /// Override: test is expected to fail with the given reason.
    pub expected_fail_reason: Option<String>,
    /// Path to test.cpp for native hardware execution (if present).
    pub test_cpp_path: Option<PathBuf>,
    /// Which argument pattern the test.cpp uses (cxxopts vs #define).
    pub test_cpp_pattern: Option<TestCppPattern>,
    /// Artifact names parsed from RUN lines (xclbin, insts, test binary).
    /// Used by discovery and the build system to know exactly what files
    /// a test produces, instead of guessing with hardcoded fallback lists.
    pub artifact_names: ArtifactNames,
}

impl NpuTestSource {
    /// Whether this test requires the Chess compiler.
    pub fn requires_chess(&self) -> bool {
        self.requires.iter().any(|r| r == "chess" || r == "valid_xchess_license")
    }

    /// Whether this test targets npu2 (Strix Point) hardware.
    pub fn requires_npu2(&self) -> bool {
        self.requires.iter().any(|r| r == "ryzen_ai_npu2")
    }

    /// Whether this test is suppressed (dont_run annotation).
    pub fn is_suppressed(&self) -> bool {
        self.requires.iter().any(|r| r == "dont_run")
    }
}

/// Annotations parsed from a test file's comment header.
pub struct Annotations {
    /// Raw RUN lines (prefix stripped, trimmed).
    pub run_lines: Vec<String>,
    /// Required features from REQUIRES: line.
    pub requires: Vec<String>,
    /// Test expected to fail.
    pub xfail: bool,
}

/// Discover NPU tests from the mlir-aie source tree.
///
/// Walks `{mlir_aie_path}/test/npu-xrt/` recursively, finding leaf test
/// directories. A leaf directory contains a test entry point file:
/// `run.lit`, `aie2.py`, or another Python file with `# RUN:` lines.
///
/// Returns tests sorted by name for deterministic ordering.
pub fn discover(mlir_aie_path: &Path) -> Vec<NpuTestSource> {
    let test_root_raw = mlir_aie_path.join("test/npu-xrt");

    if !test_root_raw.is_dir() {
        log::warn!(
            "NPU test source directory not found: {}",
            test_root_raw.display()
        );
        return Vec::new();
    }

    // Canonicalize so source_dir paths are absolute. Build commands run
    // in the output directory, so relative paths would not resolve.
    let test_root = match test_root_raw.canonicalize() {
        Ok(p) => p,
        Err(e) => {
            log::warn!("Failed to canonicalize {}: {}", test_root_raw.display(), e);
            return Vec::new();
        }
    };

    let mut tests = Vec::new();
    discover_recursive(&test_root, &test_root, &mut tests);

    // Filter out suppressed tests (dont_run) and sort
    tests.retain(|t| !t.is_suppressed());
    tests.sort_by(|a, b| a.name.cmp(&b.name));
    tests
}

/// Recursively discover test directories.
fn discover_recursive(base: &Path, dir: &Path, tests: &mut Vec<NpuTestSource>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            log::warn!("Failed to read directory {}: {}", dir.display(), e);
            return;
        }
    };

    // Collect subdirectories
    let mut subdirs = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            // Skip known non-test directories
            if name == "makefile-common" || name.starts_with('.') || name == "__pycache__" {
                continue;
            }
            subdirs.push(path);
        }
    }

    // Check if this directory is a leaf test (has an entry point file)
    if let Some((entry_file, annotations)) = find_entry_point(dir) {
        let name = match dir.strip_prefix(base) {
            Ok(rel) => rel.to_string_lossy().to_string(),
            Err(_) => dir.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string()),
        };

        let build_steps = filter_build_steps(&annotations.run_lines);
        let artifact_names = parse_artifact_names(&annotations.run_lines);

        // Parse buffer metadata from test.cpp if present
        let buffer_spec = super::test_cpp_parser::parse_test_cpp(dir);

        // Detect test.cpp and its argument pattern for native HW execution
        let (test_cpp_path, test_cpp_pattern) =
            match super::native_hw::detect_test_cpp(dir) {
                Some((path, pattern)) => (Some(path), Some(pattern)),
                None => (None, None),
            };

        tests.push(NpuTestSource {
            name,
            source_dir: dir.to_path_buf(),
            entry_file,
            build_steps,
            requires: annotations.requires,
            xfail: annotations.xfail,
            buffer_spec,
            skip_reason: None,
            expected_fail_reason: None,
            test_cpp_path,
            test_cpp_pattern,
            artifact_names,
        });
        // Leaf directory -- do not recurse further
        return;
    }

    // Not a leaf -- recurse into subdirectories
    for subdir in subdirs {
        discover_recursive(base, &subdir, tests);
    }
}

/// Find the test entry point file in a directory.
///
/// Priority: `run.lit` > `aie2.py` > other `.py` with RUN lines > `aie.mlir`.
/// Returns the entry file path and its parsed annotations.
fn find_entry_point(dir: &Path) -> Option<(PathBuf, Annotations)> {
    // 1. run.lit (highest priority -- always the driver when present)
    let run_lit = dir.join("run.lit");
    if run_lit.exists() {
        let ann = parse_annotations(&run_lit);
        if !ann.run_lines.is_empty() {
            return Some((run_lit, ann));
        }
    }

    // 2. aie2.py (standard Python MLIR generator)
    let aie2_py = dir.join("aie2.py");
    if aie2_py.exists() {
        let ann = parse_annotations(&aie2_py);
        if !ann.run_lines.is_empty() {
            return Some((aie2_py, ann));
        }
    }

    // 3. Other .py files with RUN lines (e.g. ext_to_core_L2_placed.py)
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "py") {
                let name = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");
                // Skip known non-entry-point Python files
                if name == "test.py" || name == "aie2.py" {
                    continue;
                }
                let ann = parse_annotations(&path);
                if !ann.run_lines.is_empty() {
                    return Some((path, ann));
                }
            }
        }
    }

    // 4. aie.mlir (fallback, rarely used as entry point in npu-xrt)
    let aie_mlir = dir.join("aie.mlir");
    if aie_mlir.exists() {
        let ann = parse_annotations(&aie_mlir);
        if !ann.run_lines.is_empty() {
            return Some((aie_mlir, ann));
        }
    }

    None
}

/// Parse LLVM lit-style annotations from a file.
///
/// Handles both comment styles:
/// - `// RUN:` (C++ style, used in .mlir and .lit files)
/// - `# RUN:` (Python style, used in .py files)
///
/// Also parses REQUIRES: and XFAIL: annotations from the first 40 lines.
pub fn parse_annotations(path: &Path) -> Annotations {
    let mut annotations = Annotations {
        run_lines: Vec::new(),
        requires: Vec::new(),
        xfail: false,
    };

    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return annotations,
    };

    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();

        // RUN lines: collect from entire file, both comment styles
        if let Some(rest) = trimmed.strip_prefix("// RUN:") {
            annotations.run_lines.push(rest.trim().to_string());
        } else if let Some(rest) = trimmed.strip_prefix("# RUN:") {
            annotations.run_lines.push(rest.trim().to_string());
        }

        // REQUIRES, XFAIL: only check first 40 lines (header area)
        if idx < 40 {
            let requires_text = trimmed.strip_prefix("// REQUIRES:")
                .or_else(|| trimmed.strip_prefix("# REQUIRES:"));
            if let Some(rest) = requires_text {
                for feature in rest.split(',') {
                    let f = feature.trim();
                    if !f.is_empty() {
                        annotations.requires.push(f.to_string());
                    }
                }
            }

            if trimmed.contains("XFAIL:") || trimmed.contains("XFAIL :") {
                annotations.xfail = true;
            }
        }
    }

    annotations
}

/// Filter RUN lines to keep only build commands.
///
/// The raw RUN lines include both build steps (aiecc.py, sed, cp, Python
/// MLIR generation) and execution steps (./test.exe, test.py). We keep only
/// build steps since the emulator has its own execution pipeline.
///
/// Filtering rules:
/// - Drop all `%run_on_npu2%` lines (we build for npu1 only)
/// - For `%run_on_npu1%` lines: keep build commands (sed), drop execution
/// - Drop lines starting with `clang` (host compilation)
/// - Drop lines with `./test.exe` or `test.py` (test execution)
/// - Strip `%run_on_npu1%` prefix from kept lines
/// - Strip `| FileCheck` pipeline suffix
/// Check if a line starts with a versioned GCC compiler command (e.g., `g++-13`).
fn starts_with_gcc_variant(line: &str) -> bool {
    // Match g++-N, gcc-N patterns (versioned GCC compilers)
    if let Some(rest) = line.strip_prefix("g++") {
        return rest.starts_with('-') || rest.starts_with(' ');
    }
    if let Some(rest) = line.strip_prefix("gcc") {
        return rest.starts_with('-') || rest.starts_with(' ');
    }
    false
}

pub fn filter_build_steps(run_lines: &[String]) -> Vec<String> {
    let mut steps = Vec::new();

    for line in run_lines {
        let trimmed = line.trim();

        // Drop all npu2 lines (we build for npu1 only)
        if trimmed.starts_with("%run_on_npu2%") {
            continue;
        }

        // Strip npu1 prefix (keep the command itself)
        let effective = if let Some(rest) = trimmed.strip_prefix("%run_on_npu1%") {
            rest.trim()
        } else {
            trimmed
        };

        // Drop execution and host-compilation lines.
        // Host compilers (clang, g++, etc.) produce test.exe which we don't
        // need for emulation -- we only need xclbin + insts.
        if effective.contains("./test.exe")
            || effective.contains("test.py")
            || effective.starts_with("clang ")
            || effective.starts_with("clang++ ")
            || effective.starts_with("g++ ")
            || starts_with_gcc_variant(effective)
        {
            continue;
        }

        // Strip FileCheck pipeline
        let mut step = effective.to_string();
        if let Some(pos) = step.find("| FileCheck") {
            step.truncate(pos);
        }

        let step = step.trim().to_string();
        if !step.is_empty() {
            steps.push(step);
        }
    }

    steps
}

/// Artifact names extracted from a test's RUN lines.
///
/// Parsed from `--xclbin-name=`, `--npu-insts-name=`, `--elf-name=`, and
/// `-o <binary>` patterns in the raw RUN lines. These tell the build system
/// and discovery exactly what files a test produces and consumes, rather than
/// guessing with hardcoded fallback lists.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArtifactNames {
    /// XCLBIN filenames produced by aiecc.py (e.g., "aie.xclbin", "final.xclbin").
    pub xclbin_names: Vec<String>,
    /// Instruction filenames produced by aiecc.py (e.g., "insts.bin", "insts.elf").
    pub insts_names: Vec<String>,
    /// Host test binary name from clang/g++ `-o` flag (e.g., "test.exe", "test").
    /// None if the test uses a Python script instead of compiled binary.
    pub test_exe_name: Option<String>,
}

/// Parse artifact names from raw RUN lines.
///
/// Extracts filenames from aiecc.py flags (`--xclbin-name=`, `--npu-insts-name=`,
/// `--elf-name=`) and host compiler `-o` flags. This is the single source of
/// truth for what files a test produces, replacing hardcoded name guessing in
/// discovery code.
pub fn parse_artifact_names(run_lines: &[String]) -> ArtifactNames {
    let mut names = ArtifactNames::default();

    for line in run_lines {
        let trimmed = line.trim();

        // Parse aiecc.py lines for xclbin and insts names
        if trimmed.contains("aiecc.py") {
            for part in trimmed.split_whitespace() {
                if let Some(name) = part.strip_prefix("--xclbin-name=") {
                    if !names.xclbin_names.contains(&name.to_string()) {
                        names.xclbin_names.push(name.to_string());
                    }
                }
                if let Some(name) = part.strip_prefix("--npu-insts-name=") {
                    if !names.insts_names.contains(&name.to_string()) {
                        names.insts_names.push(name.to_string());
                    }
                }
                if let Some(name) = part.strip_prefix("--elf-name=") {
                    if !names.insts_names.contains(&name.to_string()) {
                        names.insts_names.push(name.to_string());
                    }
                }
            }
        }

        // Parse host compiler lines for binary name: clang/g++/gcc -o <name>
        let is_host_compile = trimmed.starts_with("clang ")
            || trimmed.starts_with("clang++ ")
            || trimmed.starts_with("g++ ")
            || starts_with_gcc_variant(trimmed)
            || trimmed.starts_with("gcc ");
        if is_host_compile {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            for i in 0..parts.len().saturating_sub(1) {
                if parts[i] == "-o" {
                    names.test_exe_name = Some(parts[i + 1].to_string());
                    break;
                }
            }
        }
    }

    names
}

/// Test overrides loaded from `tests/test_overrides.toml`.
///
/// Replaces 68 hand-maintained manifest files with a minimal set of
/// emulator-specific metadata: skip gates and expected-fail markers.
/// Buffer metadata comes from test.cpp parsing, not overrides.
#[derive(Debug, Default, serde::Deserialize)]
#[serde(default)]
pub struct TestOverrides {
    /// Tests to skip entirely, with reason string.
    pub skip: HashMap<String, String>,
    /// Tests expected to fail, with reason string.
    pub expected_fail: HashMap<String, String>,
}

impl TestOverrides {
    /// Load overrides from a TOML file.
    ///
    /// Returns default (empty) overrides if the file is missing or unparseable.
    pub fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(content) => {
                match toml::from_str(&content) {
                    Ok(overrides) => overrides,
                    Err(e) => {
                        log::warn!("Failed to parse {}: {}", path.display(), e);
                        Self::default()
                    }
                }
            }
            Err(_) => Self::default(),
        }
    }
}

/// Load test overrides and propagate source annotations to runner metadata.
///
/// Call after `discover()`. This does two things:
///
/// 1. Applies emulator-specific gates from `test_overrides.toml` (skip,
///    expected_fail entries for emulator-specific issues).
/// 2. Propagates source-level annotations that the runner should act on:
///    - `XFAIL: *` -> `expected_fail_reason` (unless overridden by TOML)
///    - `REQUIRES: ryzen_ai_npu2` -> `skip_reason` (wrong hardware)
///
/// TOML overrides take precedence over source annotations, so
/// emulator-specific reasons are preserved when both are present.
pub fn load_overrides(tests: &mut [NpuTestSource], overrides_path: &Path) {
    let overrides = TestOverrides::load(overrides_path);

    for test in tests.iter_mut() {
        // Source annotations (lower priority, applied first)
        if test.xfail && test.expected_fail_reason.is_none() {
            test.expected_fail_reason = Some("XFAIL upstream".to_string());
        }
        if test.requires_npu2() && test.skip_reason.is_none() {
            test.skip_reason = Some("Requires NPU2 (Strix Point)".to_string());
        }

        // TOML overrides (higher priority, overwrite source annotations)
        if let Some(reason) = overrides.skip.get(&test.name) {
            test.skip_reason = Some(reason.clone());
        }
        if let Some(reason) = overrides.expected_fail.get(&test.name) {
            test.expected_fail_reason = Some(reason.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Create a unique temporary directory for tests.
    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join("xdna_emu_npu_test_tests")
            .join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_parse_annotations_run_lit() {
        let dir = test_dir("run_lit");
        let lit = dir.join("run.lit");
        let mut f = std::fs::File::create(&lit).unwrap();
        writeln!(f, "// REQUIRES: ryzen_ai").unwrap();
        writeln!(f, "//").unwrap();
        writeln!(f, "// RUN: cp %S/aie.mlir aie_arch.mlir").unwrap();
        writeln!(f, "// RUN: %python aiecc.py --no-aiesim %s").unwrap();
        writeln!(f, "// RUN: clang %S/test.cpp -o test.exe").unwrap();
        writeln!(f, "// RUN: ./test.exe -x aie.xclbin").unwrap();

        let ann = parse_annotations(&lit);
        assert_eq!(ann.run_lines.len(), 4);
        assert_eq!(ann.requires, vec!["ryzen_ai"]);
        assert!(!ann.xfail);
    }

    #[test]
    fn test_parse_annotations_python() {
        let dir = test_dir("python");
        let py = dir.join("aie2.py");
        let mut f = std::fs::File::create(&py).unwrap();
        writeln!(f, "# REQUIRES: ryzen_ai, valid_xchess_license").unwrap();
        writeln!(f, "#").unwrap();
        writeln!(f, "# RUN: %python %S/aie2.py > ./aie2.mlir").unwrap();
        writeln!(f, "# RUN: %python aiecc.py --no-aiesim %s").unwrap();
        writeln!(f, "# RUN: clang %S/test.cpp -o test.exe").unwrap();
        writeln!(f, "# RUN: %run_on_npu1% ./test.exe").unwrap();

        let ann = parse_annotations(&py);
        assert_eq!(ann.run_lines.len(), 4);
        assert_eq!(ann.requires, vec!["ryzen_ai", "valid_xchess_license"]);
    }

    #[test]
    fn test_parse_annotations_xfail() {
        let dir = test_dir("xfail");
        let lit = dir.join("run.lit");
        let mut f = std::fs::File::create(&lit).unwrap();
        writeln!(f, "// XFAIL: *").unwrap();
        writeln!(f, "// REQUIRES: ryzen_ai_npu1, chess").unwrap();
        writeln!(f, "// RUN: %python aiecc.py %s").unwrap();

        let ann = parse_annotations(&lit);
        assert!(ann.xfail);
        assert_eq!(ann.requires, vec!["ryzen_ai_npu1", "chess"]);
        assert_eq!(ann.run_lines.len(), 1);
    }

    #[test]
    fn test_filter_build_steps_typical_run_lit() {
        let run_lines = vec![
            "cp %S/aie.mlir aie_arch.mlir".to_string(),
            "%run_on_npu1% sed 's/NPUDEVICE/npu1_1col/g' -i aie_arch.mlir".to_string(),
            "%run_on_npu2% sed 's/NPUDEVICE/npu2_1col/g' -i aie_arch.mlir".to_string(),
            "%python aiecc.py --no-aiesim --aie-generate-xclbin --xclbin-name=aie.xclbin --npu-insts-name=insts.bin ./aie_arch.mlir".to_string(),
            "clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags".to_string(),
            "%run_on_npu1% ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin".to_string(),
            "%run_on_npu2% ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin".to_string(),
        ];

        let steps = filter_build_steps(&run_lines);
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0], "cp %S/aie.mlir aie_arch.mlir");
        assert!(steps[1].contains("sed") && steps[1].contains("npu1_1col"));
        // npu2 sed should be dropped
        assert!(steps.iter().all(|s| !s.contains("npu2_1col")));
        assert!(steps[2].contains("aiecc.py"));
    }

    #[test]
    fn test_filter_build_steps_aie2_py() {
        let run_lines = vec![
            "%python %S/aie2.py > ./aie2.mlir".to_string(),
            "%python aiecc.py --no-aiesim --xclbin-name=final.xclbin --npu-insts-name=insts.bin ./aie2.mlir".to_string(),
            "clang %S/test.cpp -o test.exe -std=c++17".to_string(),
            "%run_on_npu1% ./test.exe".to_string(),
        ];

        let steps = filter_build_steps(&run_lines);
        assert_eq!(steps.len(), 2);
        assert!(steps[0].contains("aie2.py"));
        assert!(steps[1].contains("aiecc.py"));
    }

    #[test]
    fn test_filter_build_steps_chess_kernel() {
        let run_lines = vec![
            "xchesscc_wrapper aie2 -I %aietools/include -c %S/kernel.cc -o kernel.o".to_string(),
            "%python %S/aie2.py > ./aie2.mlir".to_string(),
            "%python aiecc.py --no-aiesim --xclbin-name=final.xclbin ./aie2.mlir".to_string(),
            "clang %S/test.cpp -o test.exe".to_string(),
            "%run_on_npu1% ./test.exe".to_string(),
        ];

        let steps = filter_build_steps(&run_lines);
        assert_eq!(steps.len(), 3);
        assert!(steps[0].contains("xchesscc_wrapper"));
        assert!(steps[1].contains("aie2.py"));
        assert!(steps[2].contains("aiecc.py"));
    }

    #[test]
    fn test_filter_build_steps_filecheck() {
        let run_lines = vec![
            "%python aiecc.py %s | FileCheck %s".to_string(),
        ];

        let steps = filter_build_steps(&run_lines);
        assert_eq!(steps.len(), 1);
        assert!(!steps[0].contains("FileCheck"));
        assert_eq!(steps[0], "%python aiecc.py %s");
    }

    #[test]
    fn test_filter_build_steps_drops_test_py() {
        let run_lines = vec![
            "%python aiecc.py %s".to_string(),
            "%run_on_npu1% %python %S/test.py -x aie.xclbin".to_string(),
        ];

        let steps = filter_build_steps(&run_lines);
        assert_eq!(steps.len(), 1);
        assert!(steps[0].contains("aiecc.py"));
    }

    #[test]
    fn test_filter_build_steps_chess_only() {
        // Test without NPUDEVICE substitution (chess-only, direct %S/aie.mlir)
        let run_lines = vec![
            "%python aiecc.py --no-aiesim --xclbin-name=aie.xclbin --npu-insts-name=insts.bin %S/aie.mlir".to_string(),
            "clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags".to_string(),
            "%run_on_npu1% ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin".to_string(),
        ];

        let steps = filter_build_steps(&run_lines);
        assert_eq!(steps.len(), 1);
        assert!(steps[0].contains("aiecc.py"));
    }

    #[test]
    fn test_filter_build_steps_drops_gpp_host_compile() {
        // Tests like matrix_multiplication_using_cascade use g++-13 for
        // host compilation. These lines must be filtered out.
        let run_lines = vec![
            "xchesscc_wrapper aie2 -I %aietools/include -c %S/mm.cc -o ./mm.o".to_string(),
            "g++-13 %S/test.cpp -o test.exe -std=c++23 -Wall %xrt_flags".to_string(),
            "%python aiecc.py --xclbin-name=aie2_plain.xclbin --npu-insts-name=insts2_plain.txt %S/aie_plainx4.mlir".to_string(),
            "%run_on_npu1% ./test.exe -x aie2_plain.xclbin".to_string(),
            "g++ %S/other.cpp -o other.exe".to_string(),
            "gcc -c %S/helper.c -o helper.o".to_string(),
        ];

        let steps = filter_build_steps(&run_lines);
        assert_eq!(steps.len(), 2);
        assert!(steps[0].contains("xchesscc_wrapper"));
        assert!(steps[1].contains("aiecc.py"));
    }

    #[test]
    fn test_filter_build_steps_elf_instructions() {
        // add_one_objFifo_elf uses --aie-generate-elf instead of
        // --aie-generate-npu-insts. The build step must preserve these
        // flags so the build produces insts.elf, not insts.bin.
        let run_lines = vec![
            "cp %S/aie.mlir aie_arch.mlir".to_string(),
            "%run_on_npu1% sed 's/NPUDEVICE/npu1_1col/g' -i aie_arch.mlir".to_string(),
            "%run_on_npu2% sed 's/NPUDEVICE/npu2_1col/g' -i aie_arch.mlir".to_string(),
            "%python aiecc.py --aie-generate-xclbin --xclbin-name=aie.xclbin --aie-generate-elf --elf-name=insts.elf --no-compile-host ./aie_arch.mlir".to_string(),
            "clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags".to_string(),
            "%run_on_npu1% ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.elf".to_string(),
            "%run_on_npu2% ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.elf".to_string(),
            "%run_on_npu1% %python %S/test.py -x aie.xclbin -k MLIR_AIE -i insts.elf".to_string(),
            "%run_on_npu2% %python %S/test.py -x aie.xclbin -k MLIR_AIE -i insts.elf".to_string(),
        ];

        let steps = filter_build_steps(&run_lines);
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0], "cp %S/aie.mlir aie_arch.mlir");
        assert!(steps[1].contains("sed") && steps[1].contains("npu1_1col"));
        // aiecc.py line must preserve --aie-generate-elf and --elf-name
        let aiecc = &steps[2];
        assert!(aiecc.contains("aiecc.py"));
        assert!(aiecc.contains("--aie-generate-elf"));
        assert!(aiecc.contains("--elf-name=insts.elf"));
        assert!(!aiecc.contains("--aie-generate-npu-insts"));
    }

    #[test]
    fn test_requires_chess() {
        let test = NpuTestSource {
            name: "test".to_string(),
            source_dir: PathBuf::from("/test"),
            entry_file: PathBuf::from("/test/run.lit"),
            build_steps: Vec::new(),
            requires: vec!["ryzen_ai".to_string(), "chess".to_string()],
            xfail: false,
            buffer_spec: None,
            skip_reason: None,
            expected_fail_reason: None,
            test_cpp_path: None,
            test_cpp_pattern: None,
            artifact_names: ArtifactNames::default(),
        };
        assert!(test.requires_chess());
        assert!(!test.requires_npu2());
    }

    #[test]
    fn test_requires_npu2() {
        let test = NpuTestSource {
            name: "test".to_string(),
            source_dir: PathBuf::from("/test"),
            entry_file: PathBuf::from("/test/run.lit"),
            build_steps: Vec::new(),
            requires: vec!["ryzen_ai_npu2".to_string(), "peano".to_string()],
            xfail: false,
            buffer_spec: None,
            skip_reason: None,
            expected_fail_reason: None,
            test_cpp_path: None,
            test_cpp_pattern: None,
            artifact_names: ArtifactNames::default(),
        };
        assert!(test.requires_npu2());
        assert!(!test.requires_chess());
    }

    #[test]
    fn test_requires_valid_xchess_license() {
        let test = NpuTestSource {
            name: "test".to_string(),
            source_dir: PathBuf::from("/test"),
            entry_file: PathBuf::from("/test/run.lit"),
            build_steps: Vec::new(),
            requires: vec!["ryzen_ai".to_string(), "valid_xchess_license".to_string()],
            xfail: false,
            buffer_spec: None,
            skip_reason: None,
            expected_fail_reason: None,
            test_cpp_path: None,
            test_cpp_pattern: None,
            artifact_names: ArtifactNames::default(),
        };
        assert!(test.requires_chess());
    }

    #[test]
    fn test_is_suppressed() {
        let test = NpuTestSource {
            name: "test".to_string(),
            source_dir: PathBuf::from("/test"),
            entry_file: PathBuf::from("/test/test.py"),
            build_steps: Vec::new(),
            requires: vec!["dont_run".to_string()],
            xfail: false,
            buffer_spec: None,
            skip_reason: None,
            expected_fail_reason: None,
            test_cpp_path: None,
            test_cpp_pattern: None,
            artifact_names: ArtifactNames::default(),
        };
        assert!(test.is_suppressed());
    }

    #[test]
    fn test_discover_empty_dir() {
        let dir = test_dir("discover_empty");
        let test_root = dir.join("test/npu-xrt");
        std::fs::create_dir_all(&test_root).unwrap();

        let tests = discover(&dir);
        assert!(tests.is_empty());
    }

    #[test]
    fn test_discover_flat_run_lit() {
        let dir = test_dir("discover_flat");
        let test_root = dir.join("test/npu-xrt/add_one_using_dma");
        std::fs::create_dir_all(&test_root).unwrap();

        std::fs::write(
            test_root.join("run.lit"),
            "// REQUIRES: ryzen_ai\n\
             //\n\
             // RUN: cp %S/aie.mlir aie_arch.mlir\n\
             // RUN: %python aiecc.py --no-aiesim %s\n\
             // RUN: clang %S/test.cpp -o test.exe\n\
             // RUN: %run_on_npu1% ./test.exe\n",
        ).unwrap();
        std::fs::write(test_root.join("aie.mlir"), "module {}").unwrap();
        std::fs::write(test_root.join("test.cpp"), "int main() {}").unwrap();

        let tests = discover(&dir);
        assert_eq!(tests.len(), 1);
        assert_eq!(tests[0].name, "add_one_using_dma");
        assert_eq!(tests[0].requires, vec!["ryzen_ai"]);
        // Build steps: cp + aiecc.py (clang and test.exe filtered out)
        assert_eq!(tests[0].build_steps.len(), 2);
    }

    #[test]
    fn test_discover_nested_subdirectory() {
        let dir = test_dir("discover_nested");
        let test_root = dir.join("test/npu-xrt/core_dmas/writebd");
        std::fs::create_dir_all(&test_root).unwrap();

        std::fs::write(
            test_root.join("run.lit"),
            "// REQUIRES: ryzen_ai_npu1, chess\n\
             //\n\
             // RUN: %python aiecc.py --no-aiesim %S/aie.mlir\n\
             // RUN: clang %S/test.cpp -o test.exe\n\
             // RUN: %run_on_npu1% ./test.exe\n",
        ).unwrap();
        std::fs::write(test_root.join("aie.mlir"), "module {}").unwrap();
        std::fs::write(test_root.join("test.cpp"), "int main() {}").unwrap();

        let tests = discover(&dir);
        assert_eq!(tests.len(), 1);
        assert_eq!(tests[0].name, "core_dmas/writebd");
        assert!(tests[0].requires_chess());
    }

    #[test]
    fn test_discover_aie2_py() {
        let dir = test_dir("discover_aie2py");
        let test_root = dir.join("test/npu-xrt/nd_memcpy_test");
        std::fs::create_dir_all(&test_root).unwrap();

        std::fs::write(
            test_root.join("aie2.py"),
            "# REQUIRES: ryzen_ai, valid_xchess_license\n\
             #\n\
             # RUN: %python %S/aie2.py > ./aie2.mlir\n\
             # RUN: %python aiecc.py --no-aiesim --xclbin-name=final.xclbin ./aie2.mlir\n\
             # RUN: clang %S/test.cpp -o test.exe\n\
             # RUN: %run_on_npu1% ./test.exe\n\
             \n\
             import numpy as np\n",
        ).unwrap();
        std::fs::write(test_root.join("test.cpp"), "int main() {}").unwrap();

        let tests = discover(&dir);
        assert_eq!(tests.len(), 1);
        assert_eq!(tests[0].name, "nd_memcpy_test");
        assert!(tests[0].requires_chess());
        // Build steps: aie2.py + aiecc.py (clang and test.exe filtered out)
        assert_eq!(tests[0].build_steps.len(), 2);
        assert!(tests[0].build_steps[0].contains("aie2.py"));
    }

    #[test]
    fn test_discover_other_py_entry_point() {
        let dir = test_dir("discover_other_py");
        let test_root = dir.join("test/npu-xrt/custom_test");
        std::fs::create_dir_all(&test_root).unwrap();

        std::fs::write(
            test_root.join("ext_to_core.py"),
            "# REQUIRES: ryzen_ai\n\
             # RUN: %python %S/ext_to_core.py > ./aie2.mlir\n\
             # RUN: %python aiecc.py --no-aiesim --xclbin-name=final.xclbin ./aie2.mlir\n\
             # RUN: clang %S/test.cpp -o test.exe\n\
             # RUN: ./test.exe\n",
        ).unwrap();
        std::fs::write(test_root.join("test.cpp"), "int main() {}").unwrap();

        let tests = discover(&dir);
        assert_eq!(tests.len(), 1);
        assert_eq!(tests[0].name, "custom_test");
        assert_eq!(tests[0].build_steps.len(), 2);
    }

    #[test]
    fn test_discover_filters_dont_run() {
        let dir = test_dir("discover_dont_run");
        let test_root = dir.join("test/npu-xrt/suppressed");
        std::fs::create_dir_all(&test_root).unwrap();

        std::fs::write(
            test_root.join("run.lit"),
            "// REQUIRES: dont_run\n\
             // RUN: echo hello\n",
        ).unwrap();

        let tests = discover(&dir);
        assert!(tests.is_empty());
    }

    #[test]
    fn test_discover_skips_makefile_common() {
        let dir = test_dir("discover_makefile");
        let mk = dir.join("test/npu-xrt/makefile-common");
        std::fs::create_dir_all(&mk).unwrap();
        std::fs::write(mk.join("Makefile"), "all:\n\techo hi").unwrap();

        let test_root = dir.join("test/npu-xrt/real_test");
        std::fs::create_dir_all(&test_root).unwrap();
        std::fs::write(
            test_root.join("run.lit"),
            "// REQUIRES: ryzen_ai\n// RUN: %python aiecc.py %s\n",
        ).unwrap();

        let tests = discover(&dir);
        assert_eq!(tests.len(), 1);
        assert_eq!(tests[0].name, "real_test");
    }

    #[test]
    fn test_discover_sorted_output() {
        let dir = test_dir("discover_sorted");
        for name in &["zebra", "alpha", "middle"] {
            let test_root = dir.join(format!("test/npu-xrt/{}", name));
            std::fs::create_dir_all(&test_root).unwrap();
            std::fs::write(
                test_root.join("run.lit"),
                "// REQUIRES: ryzen_ai\n// RUN: %python aiecc.py %s\n",
            ).unwrap();
        }

        let tests = discover(&dir);
        assert_eq!(tests.len(), 3);
        assert_eq!(tests[0].name, "alpha");
        assert_eq!(tests[1].name, "middle");
        assert_eq!(tests[2].name, "zebra");
    }

    #[test]
    fn test_discover_multiple_nested() {
        let dir = test_dir("discover_multi_nested");

        // Two sibling tests under a parent directory
        for name in &["dma_configure_task_lock", "dma_configure_task_token"] {
            let test_root = dir.join(format!("test/npu-xrt/core_dmas/{}", name));
            std::fs::create_dir_all(&test_root).unwrap();
            std::fs::write(
                test_root.join("run.lit"),
                "// REQUIRES: ryzen_ai\n// RUN: %python aiecc.py %s\n",
            ).unwrap();
        }

        let tests = discover(&dir);
        assert_eq!(tests.len(), 2);
        assert_eq!(tests[0].name, "core_dmas/dma_configure_task_lock");
        assert_eq!(tests[1].name, "core_dmas/dma_configure_task_token");
    }

    #[test]
    fn test_run_lit_preferred_over_aie2_py() {
        let dir = test_dir("prefer_run_lit");
        let test_root = dir.join("test/npu-xrt/both_files");
        std::fs::create_dir_all(&test_root).unwrap();

        // Both run.lit and aie2.py exist -- run.lit should win
        std::fs::write(
            test_root.join("run.lit"),
            "// REQUIRES: ryzen_ai\n// RUN: echo from_run_lit\n",
        ).unwrap();
        std::fs::write(
            test_root.join("aie2.py"),
            "# REQUIRES: ryzen_ai\n# RUN: echo from_aie2_py\n",
        ).unwrap();

        let tests = discover(&dir);
        assert_eq!(tests.len(), 1);
        assert!(tests[0].entry_file.ends_with("run.lit"));
        assert!(tests[0].build_steps[0].contains("from_run_lit"));
    }

    #[test]
    fn test_discover_populates_artifact_names() {
        let dir = test_dir("discover_artifacts");
        let test_root = dir.join("test/npu-xrt/add_one_using_dma");
        std::fs::create_dir_all(&test_root).unwrap();

        std::fs::write(
            test_root.join("run.lit"),
            "// REQUIRES: ryzen_ai\n\
             //\n\
             // RUN: cp %S/aie.mlir aie_arch.mlir\n\
             // RUN: %run_on_npu1% sed 's/NPUDEVICE/npu1_1col/g' -i aie_arch.mlir\n\
             // RUN: %python aiecc.py --no-aiesim --aie-generate-xclbin --xclbin-name=aie.xclbin --npu-insts-name=insts.bin ./aie_arch.mlir\n\
             // RUN: clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags\n\
             // RUN: %run_on_npu1% ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin\n",
        ).unwrap();
        std::fs::write(test_root.join("aie.mlir"), "module {}").unwrap();
        std::fs::write(test_root.join("test.cpp"), "int main() {}").unwrap();

        let tests = discover(&dir);
        assert_eq!(tests.len(), 1);
        // Artifact names should be populated from RUN lines
        assert_eq!(tests[0].artifact_names.xclbin_names, vec!["aie.xclbin"]);
        assert_eq!(tests[0].artifact_names.insts_names, vec!["insts.bin"]);
        assert_eq!(tests[0].artifact_names.test_exe_name, Some("test.exe".to_string()));
    }

    #[test]
    fn test_discover_populates_elf_artifacts() {
        let dir = test_dir("discover_elf_artifacts");
        let test_root = dir.join("test/npu-xrt/add_one_objFifo_elf");
        std::fs::create_dir_all(&test_root).unwrap();

        std::fs::write(
            test_root.join("run.lit"),
            "// REQUIRES: ryzen_ai\n\
             //\n\
             // RUN: cp %S/aie.mlir aie_arch.mlir\n\
             // RUN: %python aiecc.py --aie-generate-xclbin --xclbin-name=aie.xclbin --aie-generate-elf --elf-name=insts.elf --no-compile-host ./aie_arch.mlir\n\
             // RUN: clang %S/test.cpp -o test.exe -std=c++17\n\
             // RUN: %run_on_npu1% ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.elf\n",
        ).unwrap();
        std::fs::write(test_root.join("aie.mlir"), "module {}").unwrap();
        std::fs::write(test_root.join("test.cpp"), "int main() {}").unwrap();

        let tests = discover(&dir);
        assert_eq!(tests.len(), 1);
        assert_eq!(tests[0].artifact_names.xclbin_names, vec!["aie.xclbin"]);
        assert_eq!(tests[0].artifact_names.insts_names, vec!["insts.elf"]);
        assert_eq!(tests[0].artifact_names.test_exe_name, Some("test.exe".to_string()));
    }

    #[test]
    fn test_load_overrides_applies_skip_and_expected_fail() {
        let dir = test_dir("load_overrides");
        let overrides_path = dir.join("test_overrides.toml");
        std::fs::write(&overrides_path, r#"
[skip]
skip_me = "Platform not supported"

[expected_fail]
flaky_test = "Known emulator limitation"
"#).unwrap();

        let mut tests = vec![
            NpuTestSource {
                name: "skip_me".to_string(),
                source_dir: dir.clone(),
                entry_file: dir.join("run.lit"),
                build_steps: Vec::new(),
                requires: Vec::new(),
                xfail: false,
                buffer_spec: None,
                skip_reason: None,
                expected_fail_reason: None,
                test_cpp_path: None,
                test_cpp_pattern: None,
                artifact_names: ArtifactNames::default(),
            },
            NpuTestSource {
                name: "flaky_test".to_string(),
                source_dir: dir.clone(),
                entry_file: dir.join("run.lit"),
                build_steps: Vec::new(),
                requires: Vec::new(),
                xfail: false,
                buffer_spec: None,
                skip_reason: None,
                expected_fail_reason: None,
                test_cpp_path: None,
                test_cpp_pattern: None,
                artifact_names: ArtifactNames::default(),
            },
            NpuTestSource {
                name: "normal_test".to_string(),
                source_dir: dir.clone(),
                entry_file: dir.join("run.lit"),
                build_steps: Vec::new(),
                requires: Vec::new(),
                xfail: false,
                buffer_spec: None,
                skip_reason: None,
                expected_fail_reason: None,
                test_cpp_path: None,
                test_cpp_pattern: None,
                artifact_names: ArtifactNames::default(),
            },
        ];

        load_overrides(&mut tests, &overrides_path);

        assert_eq!(tests[0].skip_reason.as_deref(), Some("Platform not supported"));
        assert!(tests[0].expected_fail_reason.is_none());

        assert!(tests[1].skip_reason.is_none());
        assert_eq!(tests[1].expected_fail_reason.as_deref(), Some("Known emulator limitation"));

        assert!(tests[2].skip_reason.is_none());
        assert!(tests[2].expected_fail_reason.is_none());
    }

    #[test]
    fn test_load_overrides_missing_file_returns_empty() {
        let dir = test_dir("overrides_missing");
        let overrides_path = dir.join("does_not_exist.toml");

        let mut tests = vec![NpuTestSource {
            name: "test".to_string(),
            source_dir: dir.clone(),
            entry_file: dir.join("run.lit"),
            build_steps: Vec::new(),
            requires: Vec::new(),
            xfail: false,
            buffer_spec: None,
            skip_reason: None,
            expected_fail_reason: None,
            test_cpp_path: None,
            test_cpp_pattern: None,
            artifact_names: ArtifactNames::default(),
        }];

        load_overrides(&mut tests, &overrides_path);
        assert!(tests[0].skip_reason.is_none());
        assert!(tests[0].expected_fail_reason.is_none());
    }

    #[test]
    fn test_load_overrides_nested_test_name() {
        let dir = test_dir("overrides_nested");
        let overrides_path = dir.join("test_overrides.toml");
        std::fs::write(&overrides_path, r#"
[skip]
"adjacent_memtile_access/two_memtiles" = "Requires NPU2"
"#).unwrap();

        let mut tests = vec![NpuTestSource {
            name: "adjacent_memtile_access/two_memtiles".to_string(),
            source_dir: dir.clone(),
            entry_file: dir.join("run.lit"),
            build_steps: Vec::new(),
            requires: Vec::new(),
            xfail: false,
            buffer_spec: None,
            skip_reason: None,
            expected_fail_reason: None,
            test_cpp_path: None,
            test_cpp_pattern: None,
            artifact_names: ArtifactNames::default(),
        }];

        load_overrides(&mut tests, &overrides_path);
        assert_eq!(tests[0].skip_reason.as_deref(), Some("Requires NPU2"));
    }

    // ---------------------------------------------------------------
    // parse_artifact_names tests
    // ---------------------------------------------------------------

    #[test]
    fn test_artifact_names_typical_run_lit() {
        // Standard pattern: aie.xclbin, insts.bin, test.exe
        let run_lines = vec![
            "cp %S/aie.mlir aie_arch.mlir".to_string(),
            "%run_on_npu1% sed 's/NPUDEVICE/npu1_1col/g' -i aie_arch.mlir".to_string(),
            "%python aiecc.py --no-aiesim --aie-generate-xclbin --xclbin-name=aie.xclbin --npu-insts-name=insts.bin ./aie_arch.mlir".to_string(),
            "clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags".to_string(),
            "%run_on_npu1% ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin".to_string(),
        ];

        let artifacts = parse_artifact_names(&run_lines);
        assert_eq!(artifacts.xclbin_names, vec!["aie.xclbin"]);
        assert_eq!(artifacts.insts_names, vec!["insts.bin"]);
        assert_eq!(artifacts.test_exe_name, Some("test.exe".to_string()));
    }

    #[test]
    fn test_artifact_names_elf_instructions() {
        // add_one_objFifo_elf: uses --aie-generate-elf, insts.elf
        let run_lines = vec![
            "cp %S/aie.mlir aie_arch.mlir".to_string(),
            "%python aiecc.py --aie-generate-xclbin --xclbin-name=aie.xclbin --aie-generate-elf --elf-name=insts.elf --no-compile-host ./aie_arch.mlir".to_string(),
            "clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags".to_string(),
            "%run_on_npu1% ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.elf".to_string(),
        ];

        let artifacts = parse_artifact_names(&run_lines);
        assert_eq!(artifacts.xclbin_names, vec!["aie.xclbin"]);
        assert_eq!(artifacts.insts_names, vec!["insts.elf"]);
        assert_eq!(artifacts.test_exe_name, Some("test.exe".to_string()));
    }

    #[test]
    fn test_artifact_names_final_xclbin() {
        // aie2.py tests typically use final.xclbin
        let run_lines = vec![
            "%python %S/aie2.py > ./aie2.mlir".to_string(),
            "%python aiecc.py --no-aiesim --xclbin-name=final.xclbin --npu-insts-name=insts.bin ./aie2.mlir".to_string(),
            "clang %S/test.cpp -o test.exe -std=c++17".to_string(),
            "%run_on_npu1% ./test.exe -x final.xclbin -k MLIR_AIE -i insts.bin".to_string(),
        ];

        let artifacts = parse_artifact_names(&run_lines);
        assert_eq!(artifacts.xclbin_names, vec!["final.xclbin"]);
        assert_eq!(artifacts.insts_names, vec!["insts.bin"]);
    }

    #[test]
    fn test_artifact_names_multi_kernel() {
        // matrix_multiplication_using_cascade: 3 xclbins, 3 insts files
        let run_lines = vec![
            "xchesscc_wrapper aie2 -I %aietools/include -c %S/mm.cc -o ./mm.o".to_string(),
            "%python aiecc.py --xclbin-name=aie2_buffer.xclbin --npu-insts-name=insts2_buffer.txt %S/aie_plainx1.mlir".to_string(),
            "%python aiecc.py --xclbin-name=aie2_cascade.xclbin --npu-insts-name=insts2_cascade.txt %S/aie_cascadex4.mlir".to_string(),
            "%python aiecc.py --xclbin-name=aie2_plain.xclbin --npu-insts-name=insts2_plain.txt %S/aie_plainx4.mlir".to_string(),
            "g++-13 %S/test.cpp -o test.exe -std=c++23 -Wall %xrt_flags".to_string(),
            "%run_on_npu1% ./test.exe -x aie2_buffer.xclbin -k MLIR_AIE -i insts2_buffer.txt".to_string(),
        ];

        let artifacts = parse_artifact_names(&run_lines);
        assert_eq!(artifacts.xclbin_names.len(), 3);
        assert!(artifacts.xclbin_names.contains(&"aie2_buffer.xclbin".to_string()));
        assert!(artifacts.xclbin_names.contains(&"aie2_cascade.xclbin".to_string()));
        assert!(artifacts.xclbin_names.contains(&"aie2_plain.xclbin".to_string()));
        assert_eq!(artifacts.insts_names.len(), 3);
        assert!(artifacts.insts_names.contains(&"insts2_buffer.txt".to_string()));
        assert_eq!(artifacts.test_exe_name, Some("test.exe".to_string()));
    }

    #[test]
    fn test_artifact_names_test_binary_no_extension() {
        // dma_task_large_linear: binary named "test" not "test.exe"
        let run_lines = vec![
            "%python aiecc.py --xclbin-name=final.xclbin --npu-insts-name=insts.bin %S/aie.mlir".to_string(),
            "clang %S/test.cpp -o test -std=c++17 -Wall %xrt_flags".to_string(),
            "%run_on_npu1% ./test -x final.xclbin -k MLIR_AIE -i insts.bin".to_string(),
        ];

        let artifacts = parse_artifact_names(&run_lines);
        assert_eq!(artifacts.test_exe_name, Some("test".to_string()));
        assert_eq!(artifacts.xclbin_names, vec!["final.xclbin"]);
    }

    #[test]
    fn test_artifact_names_python_test() {
        // Some tests use test.py instead of compiled binary
        let run_lines = vec![
            "%python aiecc.py --xclbin-name=aie.xclbin --npu-insts-name=insts.bin %S/aie.mlir".to_string(),
            "%run_on_npu1% %python %S/test.py -x aie.xclbin -i insts.bin".to_string(),
        ];

        let artifacts = parse_artifact_names(&run_lines);
        // Python tests don't produce a compiled binary
        assert_eq!(artifacts.test_exe_name, None);
        assert_eq!(artifacts.xclbin_names, vec!["aie.xclbin"]);
    }

    #[test]
    fn test_artifact_names_no_explicit_names() {
        // Minimal aiecc.py invocation without --xclbin-name or --npu-insts-name
        let run_lines = vec![
            "%python aiecc.py --no-aiesim %S/aie.mlir".to_string(),
            "clang %S/test.cpp -o test.exe".to_string(),
            "%run_on_npu1% ./test.exe".to_string(),
        ];

        let artifacts = parse_artifact_names(&run_lines);
        // No explicit names found
        assert!(artifacts.xclbin_names.is_empty());
        assert!(artifacts.insts_names.is_empty());
        assert_eq!(artifacts.test_exe_name, Some("test.exe".to_string()));
    }

    #[test]
    fn test_artifact_names_multi_instr_flags() {
        // multi_kernel tests use --instr0 and --instr1 in execution
        let run_lines = vec![
            "%python aiecc.py --xclbin-name=add_one.xclbin --npu-insts-name=add_one_insts.bin %S/aie_add_one.mlir".to_string(),
            "%python aiecc.py --xclbin-name=add_two.xclbin --npu-insts-name=add_two_insts.bin %S/aie_add_two.mlir".to_string(),
            "clang %S/test.cpp -o test.exe".to_string(),
            "%run_on_npu1% ./test.exe --instr0 add_one_insts.bin --instr1 add_two_insts.bin".to_string(),
        ];

        let artifacts = parse_artifact_names(&run_lines);
        assert_eq!(artifacts.xclbin_names.len(), 2);
        assert!(artifacts.xclbin_names.contains(&"add_one.xclbin".to_string()));
        assert!(artifacts.xclbin_names.contains(&"add_two.xclbin".to_string()));
        assert_eq!(artifacts.insts_names.len(), 2);
    }
}
