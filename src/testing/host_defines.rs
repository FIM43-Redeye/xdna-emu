//! Hardcoded host-side preprocessor defines for examples without CMakeLists.txt.
//!
//! Some mlir-aie programming examples define host-side `-D` flags only in their
//! Makefiles (e.g., `-DDTYPE_i32`, `-DSIZE_M=768`). Since robust Makefile
//! parsing is impractical (Make is Turing-complete), we record the defaults here
//! so our g++ fallback can compile their test.cpp.
//!
//! Each entry mirrors the Makefile's default variable values. If upstream adds a
//! CMakeLists.txt for any of these, the cmake path takes priority and the entry
//! here becomes dead code -- remove it when that happens.
//!
//! To add an entry: read the Makefile, find the variable defaults and the g++
//! invocation line, then transcribe the `-D` flags with their default values.

/// Return extra `-D` flags needed by a test's host compilation, if any.
///
/// `source_dir` is the directory containing test.cpp (and Makefile).
/// Returns an empty slice for tests that don't need special defines.
pub fn extra_defines(source_dir: &std::path::Path) -> &'static [&'static str] {
    // Match on the last path components to stay independent of absolute prefix.
    let path = source_dir.to_string_lossy();

    if path.ends_with("basic/combined_transpose") {
        // Makefile: dtype?=i32, HOST_DEFINES=-DDTYPE_${dtype}
        return &["-DDTYPE_i32"];
    }

    if path.ends_with("basic/row_wise_bias_add") {
        // Makefile: M=768, N=2304, g++ ... -DSIZE_M=$M -DSIZE_N=$N
        return &["-DSIZE_M=768", "-DSIZE_N=2304"];
    }

    &[]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_combined_transpose() {
        let defines = extra_defines(Path::new("/any/prefix/basic/combined_transpose"));
        assert_eq!(defines, &["-DDTYPE_i32"]);
    }

    #[test]
    fn test_row_wise_bias_add() {
        let defines = extra_defines(Path::new("/any/prefix/basic/row_wise_bias_add"));
        assert_eq!(defines, &["-DSIZE_M=768", "-DSIZE_N=2304"]);
    }

    #[test]
    fn test_unknown_returns_empty() {
        let defines = extra_defines(Path::new("/some/other/test"));
        assert!(defines.is_empty());
    }
}
