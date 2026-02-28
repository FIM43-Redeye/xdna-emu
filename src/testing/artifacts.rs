//! Canonical filesystem helpers for discovering test build artifacts.
//!
//! This module centralizes the low-level artifact lookup functions that
//! multiple callers need: finding xclbin files, NPU instruction binaries,
//! aiesimulator project directories, and recursively walking build output
//! directories with the standard 0/1/N xclbin branching logic.
//!
//! Source discovery (finding entry points like .lit/.py/.mlir in the
//! mlir-aie source tree) lives in [`super::npu_test`]. This module is
//! strictly about finding **build outputs** on disk.
//!
//! Also provides `ExampleSource` and `discover_buildable_examples()` for
//! programming_examples that can be built via Makefile.

use std::path::{Path, PathBuf};

use super::test_cpp_parser::BufferSpec;
use super::native_hw::TestCppPattern;
use crate::build_progress::{Buildable, BuiltArtifact};
use crate::integration::chess_build::{BuildEnv, BuildOpts, BuildResult};

// ---------------------------------------------------------------------------
// Individual artifact helpers
// ---------------------------------------------------------------------------

/// Find the NPU instructions file in a directory.
///
/// Checks for `insts.bin`, `insts.elf`, and `insts.txt` (in that order).
/// These contain host-to-NPU commands that configure shim DMAs and trigger
/// execution. The `.txt` variant appears in some programming_examples builds.
pub fn find_insts(dir: &Path) -> Option<PathBuf> {
    for name in &["insts.bin", "insts.elf", "insts.txt"] {
        let p = dir.join(name);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Find the matching insts file for a multi-variant xclbin.
///
/// In multi-xclbin test directories, the `aie` prefix in the xclbin name
/// maps to an `insts` prefix in the instructions file:
///
///   `aie2_buffer.xclbin` -> `insts2_buffer.txt`
///   `aie2_cascade.xclbin` -> `insts2_cascade.txt`
///
/// Search order:
/// 1. `aie`->`insts` prefix swap, with `.txt` then `.bin` extension
/// 2. `<stem>.txt`, `<stem>.bin` (literal stem match)
/// 3. Shared `insts.bin` fallback
pub fn find_matching_insts(dir: &Path, xclbin_stem: &str) -> Option<PathBuf> {
    // Convention: swap "aie" prefix to "insts", keep the rest
    if let Some(suffix) = xclbin_stem.strip_prefix("aie") {
        let insts_stem = format!("insts{}", suffix);
        for ext in &["txt", "bin"] {
            let candidate = dir.join(format!("{}.{}", insts_stem, ext));
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    // Fallback: try <stem>.txt and <stem>.bin directly
    for ext in &["txt", "bin"] {
        let candidate = dir.join(format!("{}.{}", xclbin_stem, ext));
        if candidate.exists() {
            return Some(candidate);
        }
    }

    // Last fallback: shared insts.bin
    let bin = dir.join("insts.bin");
    if bin.exists() {
        return Some(bin);
    }

    None
}

/// Collect all `.xclbin` files in a directory, sorted alphabetically.
pub fn collect_xclbins(dir: &Path) -> Vec<PathBuf> {
    let mut xclbins = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().is_some_and(|e| e == "xclbin") {
                xclbins.push(p);
            }
        }
    }
    xclbins.sort();
    xclbins
}

/// Find a `.prj` directory (aiesimulator project) in a directory.
pub fn find_prj_dir(dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() && path.extension().is_some_and(|e| e == "prj") {
            return Some(path);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Recursive build artifact discovery
// ---------------------------------------------------------------------------

/// A build artifact discovered by walking a build output directory.
#[derive(Debug, Clone)]
pub struct BuildArtifact {
    /// Test name (relative path from root, e.g. "add_one_using_dma"
    /// or "matrix_multiplication_using_cascade/aie2_buffer").
    pub name: String,
    /// Path to the .xclbin file.
    pub xclbin: PathBuf,
    /// Path to the NPU instructions file (if found).
    pub insts: Option<PathBuf>,
    /// Path to the .prj directory (if found).
    pub prj_dir: Option<PathBuf>,
}

/// Walk a build output directory and discover all xclbin-based test artifacts.
///
/// Handles three directory layouts:
/// 1. **Single xclbin** in dir -> one entry named after the directory.
/// 2. **Multiple xclbins** -> one entry per variant (`dir/stem`).
/// 3. **No xclbins** -> recurse into subdirectories.
pub fn discover_build_artifacts(root: &Path) -> Vec<BuildArtifact> {
    let mut results = Vec::new();
    if root.exists() {
        walk(root, "", &mut results);
    }
    results
}

/// Recursive walker implementing the 0/1/N xclbin branching logic.
fn walk(dir: &Path, prefix: &str, results: &mut Vec<BuildArtifact>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let name = entry.file_name().to_string_lossy().to_string();
        let full_name = if prefix.is_empty() {
            name
        } else {
            format!("{}/{}", prefix, name)
        };

        let xclbins = collect_xclbins(&path);
        match xclbins.len() {
            0 => {
                // No xclbins -- recurse into subdirectory
                walk(&path, &full_name, results);
            }
            1 => {
                // Single xclbin: one entry named after directory
                let insts = find_insts(&path);
                let prj = find_prj_dir(&path);
                results.push(BuildArtifact {
                    name: full_name,
                    xclbin: xclbins.into_iter().next().unwrap(),
                    insts,
                    prj_dir: prj,
                });
            }
            _ => {
                // Multiple xclbins: one entry per variant
                for xclbin in &xclbins {
                    let stem = xclbin
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_default();
                    let variant_name = format!("{}/{}", full_name, stem);
                    let insts = find_matching_insts(&path, &stem);
                    let prj = find_prj_dir(&path);
                    results.push(BuildArtifact {
                        name: variant_name,
                        xclbin: xclbin.clone(),
                        insts,
                        prj_dir: prj,
                    });
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Programming examples discovery
// ---------------------------------------------------------------------------

/// Discover pre-built test artifacts from programming_examples/.
///
/// Unlike npu-xrt (where xclbin lives directly in the test dir),
/// programming_examples put build outputs in a `build/` subdirectory:
///
///   examples/basic/passthrough_dmas/build/final.xclbin
///   examples/basic/passthrough_dmas/build/insts.bin
///
/// Only directories with both a `build/*.xclbin` and a corresponding
/// instructions file are included. Test names are prefixed with `examples/`
/// and use the path relative to `examples_root`.
pub fn discover_examples(examples_root: &Path) -> Vec<BuildArtifact> {
    let mut results = Vec::new();
    if examples_root.exists() {
        walk_examples(examples_root, examples_root, &mut results);
    }
    results
}

/// Recursive walker for programming_examples directories.
///
/// Looks for directories containing a `build/` subdirectory with xclbin
/// files. When found, creates BuildArtifact entries with the `examples/`
/// name prefix.
fn walk_examples(dir: &Path, root: &Path, results: &mut Vec<BuildArtifact>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut subdirs = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            subdirs.push(path);
        }
    }
    subdirs.sort();

    for subdir in subdirs {
        let dir_name = match subdir.file_name() {
            Some(n) => n.to_string_lossy().to_string(),
            None => continue,
        };

        // Skip hidden dirs, build dirs, and common non-test directories.
        if dir_name.starts_with('.') || dir_name.starts_with('_') {
            continue;
        }

        let build_dir = subdir.join("build");
        if build_dir.is_dir() {
            let xclbins = collect_xclbins(&build_dir);
            if !xclbins.is_empty() {
                // Compute name relative to root, prefixed with "examples/"
                let rel = subdir.strip_prefix(root).unwrap_or(&subdir);
                let name = format!("examples/{}", rel.to_string_lossy());

                // Use the first xclbin (typically final.xclbin)
                let insts = find_insts(&build_dir);

                // Only include if we have instructions -- an xclbin without
                // insts can't be executed by the emulator.
                if insts.is_some() {
                    let prj = find_prj_dir(&build_dir);
                    results.push(BuildArtifact {
                        name,
                        xclbin: xclbins.into_iter().next().unwrap(),
                        insts,
                        prj_dir: prj,
                    });
                }
            }
        }

        // Always recurse (examples are nested: basic/passthrough_dmas/)
        walk_examples(&subdir, root, results);
    }
}

// ---------------------------------------------------------------------------
// Buildable programming examples
// ---------------------------------------------------------------------------

/// A programming_example that can be built via its Makefile.
///
/// Unlike `NpuTestSource` (which uses RUN-line build steps from lit files),
/// examples use `make` with `makefile-common` conventions: `CHESS=true|false`
/// controls the compiler, and the default target produces the xclbin output
/// (usually `build/final.xclbin`, but some examples use other names).
#[derive(Debug, Clone)]
pub struct ExampleSource {
    /// Test name, prefixed with "examples/".
    /// e.g. "examples/basic/passthrough_dmas"
    pub name: String,
    /// Absolute path to the example source directory.
    pub source_dir: PathBuf,
    /// The primary Python source file (cache invalidation key).
    pub python_source: PathBuf,
    /// Buffer metadata parsed from test.cpp (sizes, types, group IDs).
    pub buffer_spec: Option<BufferSpec>,
    /// Which argument pattern the test.cpp uses.
    pub test_cpp_pattern: Option<TestCppPattern>,
}

impl Buildable for ExampleSource {
    fn name(&self) -> &str {
        &self.name
    }

    fn requires_chess(&self) -> bool {
        false // all examples support both compilers
    }

    fn skip_reason(&self) -> Option<&str> {
        None
    }

    fn can_build(&self) -> bool {
        true // discovery only finds buildable dirs
    }

    fn build(
        &self,
        build_env: &BuildEnv,
        output_dir: &Path,
        opts: &BuildOpts,
    ) -> Result<BuildResult, String> {
        build_env.build_example(&self.source_dir, &self.python_source, output_dir, opts)
    }

    fn output_dir(&self, use_chess: bool) -> PathBuf {
        if use_chess {
            // Chess: separate workspace to avoid clobbering Peano build
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("build/chess_examples")
                .join(&self.name)
        } else {
            // Peano: build in-place (Makefile convention)
            self.source_dir.clone()
        }
    }

    fn collect_artifacts(
        &self,
        _output_dir: &Path,
        result: &BuildResult,
    ) -> Vec<BuiltArtifact> {
        vec![BuiltArtifact {
            test_name: self.name.clone(),
            xclbin: result.xclbin.clone(),
            insts: result.insts.clone(),
            prj_dir: result.prj_dir.clone(),
            build_log: result.build_log.clone(),
        }]
    }

    fn enrich_test(&self, test: &mut crate::testing::xclbin_suite::XclbinTest) {
        test.buffer_spec = self.buffer_spec.clone();
        test.source_dir = Some(self.source_dir.clone());
        test.test_cpp_pattern = self.test_cpp_pattern;
    }
}

/// Discover buildable programming_examples from the source tree.
///
/// Walks `examples_root` recursively. For each directory with both a
/// `Makefile` and at least one `*.py` file, creates an `ExampleSource`.
/// Skips hidden dirs, `_`-prefixed dirs, and `makefile-common`.
pub fn discover_buildable_examples(examples_root: &Path) -> Vec<ExampleSource> {
    let mut results = Vec::new();
    if examples_root.exists() {
        walk_buildable_examples(examples_root, examples_root, &mut results);
    }
    results.sort_by(|a, b| a.name.cmp(&b.name));
    results
}

/// Recursive walker for buildable programming_examples.
fn walk_buildable_examples(
    dir: &Path,
    root: &Path,
    results: &mut Vec<ExampleSource>,
) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut subdirs = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            subdirs.push(path);
        }
    }
    subdirs.sort();

    for subdir in subdirs {
        let dir_name = match subdir.file_name() {
            Some(n) => n.to_string_lossy().to_string(),
            None => continue,
        };

        // Skip hidden dirs, _-prefixed dirs, build dirs, and makefile-common
        if dir_name.starts_with('.')
            || dir_name.starts_with('_')
            || dir_name == "build"
            || dir_name == "makefile-common"
            || dir_name == "utils"
            || dir_name == "mlir"
        {
            continue;
        }

        let makefile = subdir.join("Makefile");
        if makefile.exists() {
            // Find Python source files
            if let Some(py_source) = find_python_source(&subdir) {
                let rel = subdir.strip_prefix(root).unwrap_or(&subdir);
                let name = format!("examples/{}", rel.to_string_lossy());

                // Parse test.cpp for buffer metadata
                let buffer_spec = super::test_cpp_parser::parse_test_cpp(&subdir);
                let test_cpp_pattern = super::native_hw::detect_test_cpp(&subdir)
                    .map(|(_, pattern)| pattern);

                results.push(ExampleSource {
                    name,
                    source_dir: subdir.clone(),
                    python_source: py_source,
                    buffer_spec,
                    test_cpp_pattern,
                });

                // Don't recurse into directories that are themselves examples
                continue;
            }
        }

        // Recurse into subdirectories
        walk_buildable_examples(&subdir, root, results);
    }
}

/// Find the primary Python source file in a directory.
///
/// Looks for `*.py` files, preferring the one that matches the directory
/// name (e.g. `passthrough_dmas.py` in `passthrough_dmas/`).
fn find_python_source(dir: &Path) -> Option<PathBuf> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return None,
    };

    let dir_stem = dir.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    let mut py_files: Vec<PathBuf> = entries.flatten()
        .filter_map(|e| {
            let p = e.path();
            if p.extension().is_some_and(|ext| ext == "py") && p.is_file() {
                Some(p)
            } else {
                None
            }
        })
        .collect();
    py_files.sort();

    if py_files.is_empty() {
        return None;
    }

    // Prefer the file matching the directory name
    let preferred = py_files.iter().find(|p| {
        p.file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .is_some_and(|s| s == dir_stem)
    });

    preferred.cloned().or_else(|| py_files.into_iter().next())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Create a temp directory with a unique suffix for test isolation.
    fn test_dir(suffix: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xdna_emu_artifacts_test_{}", suffix));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_find_insts_bin() {
        let dir = test_dir("find_insts_bin");
        fs::write(dir.join("insts.bin"), b"data").unwrap();
        let result = find_insts(&dir);
        assert_eq!(result, Some(dir.join("insts.bin")));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_find_insts_elf() {
        let dir = test_dir("find_insts_elf");
        fs::write(dir.join("insts.elf"), b"data").unwrap();
        let result = find_insts(&dir);
        assert_eq!(result, Some(dir.join("insts.elf")));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_find_insts_prefers_bin() {
        let dir = test_dir("find_insts_prefer");
        fs::write(dir.join("insts.bin"), b"bin").unwrap();
        fs::write(dir.join("insts.elf"), b"elf").unwrap();
        let result = find_insts(&dir);
        assert_eq!(result, Some(dir.join("insts.bin")));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_find_insts_none() {
        let dir = test_dir("find_insts_none");
        assert_eq!(find_insts(&dir), None);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_find_insts_txt() {
        let dir = test_dir("find_insts_txt");
        fs::write(dir.join("insts.txt"), b"data").unwrap();
        let result = find_insts(&dir);
        assert_eq!(result, Some(dir.join("insts.txt")));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_find_insts_prefers_bin_over_txt() {
        let dir = test_dir("find_insts_bin_over_txt");
        fs::write(dir.join("insts.bin"), b"bin").unwrap();
        fs::write(dir.join("insts.txt"), b"txt").unwrap();
        let result = find_insts(&dir);
        assert_eq!(result, Some(dir.join("insts.bin")));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_find_insts_nonexistent_dir() {
        assert_eq!(find_insts(Path::new("/nonexistent_dir_12345")), None);
    }

    #[test]
    fn test_find_matching_insts_prefix_swap() {
        let dir = test_dir("matching_prefix");
        fs::write(dir.join("insts2_buffer.txt"), b"data").unwrap();
        let result = find_matching_insts(&dir, "aie2_buffer");
        assert_eq!(result, Some(dir.join("insts2_buffer.txt")));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_find_matching_insts_bin_fallback() {
        let dir = test_dir("matching_bin");
        fs::write(dir.join("insts2_buffer.bin"), b"data").unwrap();
        let result = find_matching_insts(&dir, "aie2_buffer");
        assert_eq!(result, Some(dir.join("insts2_buffer.bin")));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_find_matching_insts_stem_fallback() {
        let dir = test_dir("matching_stem");
        // No aie prefix -- falls through to stem.txt
        fs::write(dir.join("custom.txt"), b"data").unwrap();
        let result = find_matching_insts(&dir, "custom");
        assert_eq!(result, Some(dir.join("custom.txt")));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_find_matching_insts_shared_fallback() {
        let dir = test_dir("matching_shared");
        fs::write(dir.join("insts.bin"), b"data").unwrap();
        let result = find_matching_insts(&dir, "aie2_something");
        assert_eq!(result, Some(dir.join("insts.bin")));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_find_matching_insts_none() {
        let dir = test_dir("matching_none");
        assert_eq!(find_matching_insts(&dir, "aie2_buffer"), None);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_collect_xclbins_sorted() {
        let dir = test_dir("collect_sorted");
        fs::write(dir.join("c.xclbin"), b"").unwrap();
        fs::write(dir.join("a.xclbin"), b"").unwrap();
        fs::write(dir.join("b.xclbin"), b"").unwrap();
        fs::write(dir.join("not_xclbin.txt"), b"").unwrap();
        let result = collect_xclbins(&dir);
        let names: Vec<_> = result.iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap())
            .collect();
        assert_eq!(names, vec!["a.xclbin", "b.xclbin", "c.xclbin"]);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_collect_xclbins_empty() {
        let dir = test_dir("collect_empty");
        assert!(collect_xclbins(&dir).is_empty());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_collect_xclbins_nonexistent() {
        assert!(collect_xclbins(Path::new("/nonexistent_12345")).is_empty());
    }

    #[test]
    fn test_find_prj_dir_found() {
        let dir = test_dir("prj_found");
        fs::create_dir_all(dir.join("aie.mlir.prj")).unwrap();
        let result = find_prj_dir(&dir);
        assert!(result.is_some());
        assert!(result.unwrap().ends_with("aie.mlir.prj"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_find_prj_dir_none() {
        let dir = test_dir("prj_none");
        fs::create_dir_all(dir.join("not_a_prj")).unwrap();
        assert_eq!(find_prj_dir(&dir), None);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_find_prj_dir_nonexistent() {
        assert_eq!(find_prj_dir(Path::new("/nonexistent_12345")), None);
    }

    #[test]
    fn test_discover_single_xclbin() {
        let root = test_dir("discover_single");
        let test_dir_path = root.join("add_one");
        fs::create_dir_all(&test_dir_path).unwrap();
        fs::write(test_dir_path.join("aie.xclbin"), b"xclbin").unwrap();
        fs::write(test_dir_path.join("insts.bin"), b"insts").unwrap();

        let arts = discover_build_artifacts(&root);
        assert_eq!(arts.len(), 1);
        assert_eq!(arts[0].name, "add_one");
        assert!(arts[0].xclbin.ends_with("aie.xclbin"));
        assert!(arts[0].insts.as_ref().unwrap().ends_with("insts.bin"));
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_discover_multi_xclbin() {
        let root = test_dir("discover_multi");
        let test_dir_path = root.join("cascade");
        fs::create_dir_all(&test_dir_path).unwrap();
        fs::write(test_dir_path.join("aie2_buffer.xclbin"), b"").unwrap();
        fs::write(test_dir_path.join("aie2_cascade.xclbin"), b"").unwrap();
        fs::write(test_dir_path.join("insts2_buffer.txt"), b"").unwrap();
        fs::write(test_dir_path.join("insts2_cascade.txt"), b"").unwrap();

        let mut arts = discover_build_artifacts(&root);
        arts.sort_by(|a, b| a.name.cmp(&b.name));
        assert_eq!(arts.len(), 2);
        assert_eq!(arts[0].name, "cascade/aie2_buffer");
        assert_eq!(arts[1].name, "cascade/aie2_cascade");
        assert!(arts[0].insts.is_some());
        assert!(arts[1].insts.is_some());
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_discover_nested() {
        let root = test_dir("discover_nested");
        // Parent with no xclbin, child with xclbin
        let child = root.join("core_dmas").join("writebd");
        fs::create_dir_all(&child).unwrap();
        fs::write(child.join("aie.xclbin"), b"").unwrap();

        let arts = discover_build_artifacts(&root);
        assert_eq!(arts.len(), 1);
        assert_eq!(arts[0].name, "core_dmas/writebd");
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_discover_empty() {
        let root = test_dir("discover_empty");
        let arts = discover_build_artifacts(&root);
        assert!(arts.is_empty());
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_discover_nonexistent() {
        let arts = discover_build_artifacts(Path::new("/nonexistent_12345"));
        assert!(arts.is_empty());
    }

    #[test]
    fn test_discover_with_prj() {
        let root = test_dir("discover_prj");
        let test_dir_path = root.join("sim_test");
        fs::create_dir_all(&test_dir_path).unwrap();
        fs::write(test_dir_path.join("aie.xclbin"), b"").unwrap();
        fs::create_dir_all(test_dir_path.join("aie.mlir.prj")).unwrap();

        let arts = discover_build_artifacts(&root);
        assert_eq!(arts.len(), 1);
        assert!(arts[0].prj_dir.is_some());
        fs::remove_dir_all(&root).ok();
    }

    // -----------------------------------------------------------------------
    // discover_examples tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_discover_examples_basic() {
        let root = test_dir("examples_basic");
        // Simulate: basic/passthrough_dmas/build/final.xclbin + insts.bin
        let build = root.join("basic/passthrough_dmas/build");
        fs::create_dir_all(&build).unwrap();
        fs::write(build.join("final.xclbin"), b"xclbin").unwrap();
        fs::write(build.join("insts.bin"), b"insts").unwrap();

        let arts = discover_examples(&root);
        assert_eq!(arts.len(), 1);
        assert_eq!(arts[0].name, "examples/basic/passthrough_dmas");
        assert!(arts[0].xclbin.ends_with("final.xclbin"));
        assert!(arts[0].insts.as_ref().unwrap().ends_with("insts.bin"));
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_discover_examples_with_elf() {
        let root = test_dir("examples_elf");
        let build = root.join("basic/memcpy/build");
        fs::create_dir_all(&build).unwrap();
        fs::write(build.join("final.xclbin"), b"xclbin").unwrap();
        fs::write(build.join("insts.elf"), b"elf").unwrap();

        let arts = discover_examples(&root);
        assert_eq!(arts.len(), 1);
        assert_eq!(arts[0].name, "examples/basic/memcpy");
        assert!(arts[0].insts.as_ref().unwrap().ends_with("insts.elf"));
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_discover_examples_skips_no_insts() {
        let root = test_dir("examples_no_insts");
        // xclbin without instructions -- should be skipped
        let build = root.join("basic/broken/build");
        fs::create_dir_all(&build).unwrap();
        fs::write(build.join("final.xclbin"), b"xclbin").unwrap();

        let arts = discover_examples(&root);
        assert!(arts.is_empty());
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_discover_examples_multiple() {
        let root = test_dir("examples_multi");
        // Two examples at different nesting levels
        for (subpath, insts_name) in &[
            ("basic/dma_transpose/build", "insts.bin"),
            ("ml/matmul/build", "insts.elf"),
        ] {
            let build = root.join(subpath);
            fs::create_dir_all(&build).unwrap();
            fs::write(build.join("final.xclbin"), b"xclbin").unwrap();
            fs::write(build.join(insts_name), b"data").unwrap();
        }

        let mut arts = discover_examples(&root);
        arts.sort_by(|a, b| a.name.cmp(&b.name));
        assert_eq!(arts.len(), 2);
        assert_eq!(arts[0].name, "examples/basic/dma_transpose");
        assert_eq!(arts[1].name, "examples/ml/matmul");
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_discover_examples_empty() {
        let root = test_dir("examples_empty");
        let arts = discover_examples(&root);
        assert!(arts.is_empty());
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn test_discover_examples_nonexistent() {
        let arts = discover_examples(Path::new("/nonexistent_examples_12345"));
        assert!(arts.is_empty());
    }

    #[test]
    fn test_discover_examples_skips_hidden_dirs() {
        let root = test_dir("examples_hidden");
        // Hidden dir should be skipped
        let build = root.join(".hidden/test/build");
        fs::create_dir_all(&build).unwrap();
        fs::write(build.join("final.xclbin"), b"xclbin").unwrap();
        fs::write(build.join("insts.bin"), b"data").unwrap();

        // _build dir should be skipped
        let build2 = root.join("_build/test/build");
        fs::create_dir_all(&build2).unwrap();
        fs::write(build2.join("final.xclbin"), b"xclbin").unwrap();
        fs::write(build2.join("insts.bin"), b"data").unwrap();

        let arts = discover_examples(&root);
        assert!(arts.is_empty());
        fs::remove_dir_all(&root).ok();
    }
}
