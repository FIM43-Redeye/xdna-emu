//! Build script for xdna-emu.
//!
//! Most arch-data code generation has moved to `crates/xdna-archspec/build.rs`.
//! What remains here:
//!
//! - XRT plugin install logic (always belongs in xdna-emu).
//!
//! Moved to xdna-archspec/build.rs in Subsystem 6:
//! - `extract_aiert` + helpers + `gen_aiert_*` (Task 11).
//!   Generated files exposed via `xdna_archspec::aie2::aiert::*`;
//!   consumed by `src/device/aiert_validation.rs`.
//! - `compile_llvm_decoder_ffi` + LLVM link (Task 9).
//! - `gen_tablegen.rs` (Task 7); consumed via
//!   `xdna_archspec::aie2::isa::load_from_generated()`.
//! - All other gen_* functions (Tasks 4-8).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

// AM025 register database types come from the archspec crate.
// Used here only for the processor-slot cross-validation.
use xdna_archspec::regdb::RegisterDb;

// ============================================================================
// Entry point
// ============================================================================

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // Resolve AM025 JSON path: MLIR_AIE_PATH env or sibling directory
    let mlir_aie = env::var("MLIR_AIE_PATH").unwrap_or_else(|_| {
        manifest_dir
            .parent()
            .expect("Cargo manifest has no parent directory")
            .join("mlir-aie")
            .to_string_lossy()
            .to_string()
    });
    let am025_path =
        Path::new(&mlir_aie).join("lib/Dialect/AIE/Util/aie_registers_aie2.json");

    // Device model is in-repo
    let device_model_path = manifest_dir.join("tools/aie-device-models.json");

    // Rebuild triggers
    println!("cargo:rerun-if-changed={}", am025_path.display());
    println!("cargo:rerun-if-changed={}", device_model_path.display());
    println!("cargo:rerun-if-env-changed=MLIR_AIE_PATH");
    println!("cargo:rerun-if-changed=build.rs");

    // Load AM025 register database via the archspec crate's parser.
    let regdb = RegisterDb::from_file(&am025_path).unwrap_or_else(|e| {
        panic!(
            "Cannot load AM025 register database at {}:\n  {}\n\
             Set MLIR_AIE_PATH to override the mlir-aie location.",
            am025_path.display(),
            e
        )
    });

    // Build the full ArchModel (device topology + register data, cross-validated).
    // This is the graph as compile-time truth: Confirmed<T> panics on conflicts.
    let mut arch_model = xdna_archspec::build_arch_model(&device_model_path, &regdb, "npu1")
        .unwrap_or_else(|e| panic!("Failed to build ArchModel: {}", e));

    // Note: aie-rt extraction and cross-validation (extract_aiert + gen_aiert_*)
    // now runs in xdna-archspec's build.rs. The generated files are exposed via
    // xdna_archspec::aie2::aiert::{dma,locks,ports}, which aiert_validation.rs
    // imports directly.

    // Cross-validate processor model slot widths against llvm-aie TableGen.
    // The manual constants (from populate_aie2_manual_constants) are the baseline;
    // this extraction from AIE2Slots.td confirms them at compile time.
    let llvm_aie = env::var("LLVM_AIE_PATH").unwrap_or_else(|_| {
        manifest_dir
            .parent()
            .expect("Cargo manifest has no parent directory")
            .join("llvm-aie")
            .to_string_lossy()
            .to_string()
    });
    let llvm_aie_path = Path::new(&llvm_aie);
    let slots_td = llvm_aie_path.join("llvm/lib/Target/AIE/AIE2Slots.td");
    println!("cargo:rerun-if-env-changed=LLVM_AIE_PATH");
    if slots_td.exists() {
        println!("cargo:rerun-if-changed={}", slots_td.display());
        match xdna_archspec::tablegen::confirm_processor_slots(
            &mut arch_model,
            llvm_aie_path,
        ) {
            Ok(count) => {
                println!(
                    "cargo:warning=TableGen: confirmed {} slot widths from {}",
                    count,
                    slots_td.display()
                );
            }
            Err(e) => {
                panic!(
                    "TableGen slot validation failed:\n  {}\n\
                     Set LLVM_AIE_PATH to override the llvm-aie location.",
                    e
                );
            }
        }
    } else {
        println!(
            "cargo:warning=llvm-aie not found at {} -- using manual slot widths only",
            llvm_aie_path.display()
        );
    }

    // gen_arch(), gen_subsystems(), gen_core_module(), gen_lock_request(),
    // gen_stream_ports(), gen_stream_ranges(), and gen_trace_events() have all
    // moved to xdna-archspec/build.rs (Tasks 4-8).
    //
    // gen_tablegen.rs now emits from xdna-archspec/build.rs (Task 7).
    // load_from_generated() is forwarded via pub use in src/tablegen/mod.rs.

    // ========================================================================
    // Post-codegen: rebuild and install XRT plugin
    // ========================================================================
    //
    // The C++ plugin (libxrt_driver_emu.so) loads the Rust emulator at
    // runtime via dlopen -- there is no link-time dependency. This lets
    // us build and install the plugin from build.rs without circularity.
    //
    // The plugin .so is installed to /opt/xilinx/xrt/lib/ so XRT can
    // find it. The Rust lib is NOT copied -- the plugin resolves it at
    // runtime via XDNA_EMU_DIR/target/$profile/libxdna_emu.so.

    // Rebuild triggers for plugin C++ sources
    let plugin_src = manifest_dir.join("xrt-plugin/src");
    if plugin_src.exists() {
        for entry in fs::read_dir(&plugin_src).unwrap() {
            let entry = entry.unwrap();
            println!("cargo:rerun-if-changed={}", entry.path().display());
        }
        println!("cargo:rerun-if-changed=xrt-plugin/CMakeLists.txt");
    }

    // Only build the plugin if the cmake build directory exists.
    // First-time setup still requires: mkdir -p xrt-plugin/build && cd xrt-plugin/build && cmake ..
    let plugin_build = manifest_dir.join("xrt-plugin/build");
    let xrt_lib = Path::new("/opt/xilinx/xrt/lib");
    if plugin_build.join("CMakeCache.txt").exists() && xrt_lib.exists() {
        // Incremental cmake build (~2s when nothing changed).
        // Use .output() to capture stderr for diagnostics on failure.
        let result = std::process::Command::new("cmake")
            .args(["--build", "."])
            .current_dir(&plugin_build)
            .output();

        match result {
            Ok(output) if output.status.success() => {
                // Install plugin .so to XRT lib directory.
                // If the destination is already a symlink to the source,
                // skip the copy -- fs::copy follows symlinks and would
                // truncate the source file to 0 bytes.
                let src = plugin_build.join("libxrt_driver_emu.so.2.21.0");
                let dst = xrt_lib.join("libxrt_driver_emu.so.2.21.0");
                let link = xrt_lib.join("libxrt_driver_emu.so.2");
                if src.exists() {
                    let dst_resolves_to_src = dst.read_link()
                        .ok()
                        .and_then(|target| fs::canonicalize(xrt_lib.join(target)).ok())
                        .map_or(false, |resolved| {
                            fs::canonicalize(&src).map_or(false, |src_canon| resolved == src_canon)
                        });
                    if dst_resolves_to_src {
                        // Already symlinked -- nothing to do.
                    } else if let Err(e) = fs::copy(&src, &dst) {
                        println!("cargo:warning=Plugin install failed: {e}");
                    } else {
                        // Create/update symlink
                        let _ = fs::remove_file(&link);
                        #[cfg(unix)]
                        {
                            use std::os::unix::fs::symlink;
                            let _ = symlink("libxrt_driver_emu.so.2.21.0", &link);
                        }
                    }
                }
            }
            Ok(output) => {
                let code = output.status.code().unwrap_or(-1);
                let stderr = String::from_utf8_lossy(&output.stderr);
                println!("cargo:warning=Plugin cmake build failed (exit {code})");
                // Print first few lines of stderr for diagnostics.
                for line in stderr.lines().take(10) {
                    println!("cargo:warning=  plugin: {line}");
                }
            }
            Err(e) => {
                println!("cargo:warning=Plugin cmake build failed: {e}");
            }
        }
    }
}
