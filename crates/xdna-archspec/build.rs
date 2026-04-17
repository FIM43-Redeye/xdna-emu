//! Build script for xdna-archspec.
//!
//! Drives all AIE2 arch-data code generation from the validated
//! ArchModel (device-model + AM025 JSON, cross-validated via
//! Confirmed<T>). Output files land in $OUT_DIR and are included
//! by modules under `src/aie2/`.
//!
//! Because this script lives inside the crate it is generating for,
//! it cannot declare `xdna-archspec` as a build-dep. Module source
//! files are `#[path]`-included so the same types and parsers used
//! at runtime are available at build time.

#[path = "src/types.rs"]
mod types;
#[path = "src/regdb.rs"]
mod regdb;
#[path = "src/device_model.rs"]
mod device_model;
#[path = "src/regdb_extractor.rs"]
mod regdb_extractor;
#[path = "src/tablegen.rs"]
mod tablegen;
#[path = "src/model_builder.rs"]
mod model_builder;

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let _out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Workspace root = crate's parent's parent (crates/xdna-archspec -> crates -> root).
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("xdna-archspec manifest has no grandparent (expected <workspace-root>/crates/xdna-archspec)");

    // Resolve AM025 JSON: MLIR_AIE_PATH env var or sibling dir.
    let mlir_aie = env::var("MLIR_AIE_PATH").unwrap_or_else(|_| {
        workspace_root
            .parent()
            .expect("workspace root has no parent -- set MLIR_AIE_PATH to override")
            .join("mlir-aie")
            .to_string_lossy()
            .to_string()
    });
    let am025_path = Path::new(&mlir_aie).join("lib/Dialect/AIE/Util/aie_registers_aie2.json");

    // Device model is in the workspace root.
    let device_model_path = workspace_root.join("tools/aie-device-models.json");

    // Rebuild triggers.
    println!("cargo:rerun-if-changed={}", am025_path.display());
    println!("cargo:rerun-if-changed={}", device_model_path.display());
    println!("cargo:rerun-if-env-changed=MLIR_AIE_PATH");
    println!("cargo:rerun-if-changed=build.rs");

    // Tasks 4-11 add actual codegen calls here.

    // Silence unused warnings until the codegen steps land.
    let _ = (am025_path, device_model_path);
}
