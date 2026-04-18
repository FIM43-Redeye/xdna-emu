//! Build script for xdna-emu.
//!
//! After Subsystem 6, all arch-data codegen and the LLVM MCDisassembler
//! FFI compile live in `crates/xdna-archspec/build.rs`.  What remains
//! here is the XRT plugin install: the C++ plugin (libxrt_driver_emu.so)
//! loads xdna-emu's Rust cdylib at runtime via dlopen, so we (re)build
//! the plugin and copy it to /opt/xilinx/xrt/lib/ as a post-build step.
//!
//! xdna-emu no longer generates any Rust source of its own.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    println!("cargo:rerun-if-changed=build.rs");

    // ========================================================================
    // Rebuild and install the XRT plugin
    // ========================================================================
    //
    // The C++ plugin (libxrt_driver_emu.so) loads the Rust emulator at
    // runtime via dlopen -- there is no link-time dependency.  This lets
    // us build and install the plugin from build.rs without circularity.
    //
    // The plugin .so is installed to /opt/xilinx/xrt/lib/ so XRT can find
    // it.  The Rust lib is NOT copied -- the plugin resolves it at
    // runtime via XDNA_EMU_DIR/target/$profile/libxdna_emu.so.

    let plugin_src = manifest_dir.join("xrt-plugin/src");
    if plugin_src.exists() {
        for entry in fs::read_dir(&plugin_src).unwrap() {
            let entry = entry.unwrap();
            println!("cargo:rerun-if-changed={}", entry.path().display());
        }
        println!("cargo:rerun-if-changed=xrt-plugin/CMakeLists.txt");
    }

    // Only build the plugin if the cmake build directory exists.
    // First-time setup still requires:
    //   mkdir -p xrt-plugin/build && cd xrt-plugin/build && cmake ..
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
