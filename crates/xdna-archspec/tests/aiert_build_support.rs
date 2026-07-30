use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use tempfile::TempDir;

#[path = "../build_helpers/aiert.rs"]
mod aiert;

const DMA: &[&str] = &["compute_dma", "memtile_dma", "shim_dma"];
const LOCKS: &[&str] = &["compute_locks", "memtile_locks", "shim_locks"];
const PORTS: &[&str] = &[
    "COMPUTE_MASTER_PORTS",
    "COMPUTE_SLAVE_PORTS",
    "MEMTILE_MASTER_PORTS",
    "MEMTILE_SLAVE_PORTS",
    "SHIM_MASTER_PORTS",
    "SHIM_SLAVE_PORTS",
];

fn create_reginit(driver_src: &Path) {
    let path = driver_src.join("global/xaiemlgbl_reginit.c");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, "").unwrap();
}

fn write_executable(path: &Path, body: &str) {
    fs::write(path, format!("#!/bin/sh\n{body}\n")).unwrap();
    let mut permissions = fs::metadata(path).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions).unwrap();
}

#[test]
fn missing_reginit_is_an_error() {
    let temp = TempDir::new().unwrap();
    let driver_src = temp.path().join("driver/src");
    fs::create_dir_all(&driver_src).unwrap();

    let error = aiert::preprocess(&driver_src, Path::new("gcc")).unwrap_err();

    assert!(error.contains("global/xaiemlgbl_reginit.c"));
    assert!(error.contains(&driver_src.display().to_string()));
}

#[test]
fn unavailable_preprocessor_is_an_error() {
    let temp = TempDir::new().unwrap();
    let driver_src = temp.path().join("driver/src");
    create_reginit(&driver_src);
    let missing = temp.path().join("missing-gcc");

    let error = aiert::preprocess(&driver_src, &missing).unwrap_err();

    assert!(error.contains("cannot run"));
    assert!(error.contains(&missing.display().to_string()));
}

#[test]
fn nonzero_preprocessor_status_preserves_stderr() {
    let temp = TempDir::new().unwrap();
    let driver_src = temp.path().join("driver/src");
    let compiler = temp.path().join("failing-gcc");
    create_reginit(&driver_src);
    write_executable(&compiler, "echo 'fixture preprocessing failed' >&2\nexit 7");

    let error = aiert::preprocess(&driver_src, &compiler).unwrap_err();

    assert!(error.contains("status 7"));
    assert!(error.contains("fixture preprocessing failed"));
}

#[test]
fn successful_preprocessor_receives_source_and_include_arguments() {
    let temp = TempDir::new().unwrap();
    let driver_src = temp.path().join("driver/src");
    let compiler = temp.path().join("recording-gcc");
    create_reginit(&driver_src);
    write_executable(&compiler, "printf '%s\\n' \"$@\"");

    let output = aiert::preprocess(&driver_src, &compiler).unwrap();

    assert!(output.lines().any(|line| line == "-E"));
    assert!(output
        .lines()
        .any(|line| line == driver_src.join("global/xaiemlgbl_reginit.c").to_string_lossy()));
    assert!(output.lines().any(|line| line == "-I"));
    assert!(output.lines().any(|line| line == driver_src.join("dma").to_string_lossy()));
}

#[test]
fn complete_required_categories_are_accepted() {
    aiert::require_generated_categories(DMA, LOCKS, PORTS).unwrap();
}

#[test]
fn every_required_dma_category_is_enforced() {
    for missing in DMA {
        let present = DMA.iter().copied().filter(|name| name != missing).collect::<Vec<_>>();

        let error = aiert::require_generated_categories(&present, LOCKS, PORTS).unwrap_err();

        assert!(error.contains("DMA"), "{error}");
        assert!(error.contains(missing), "{error}");
    }
}

#[test]
fn every_required_lock_category_is_enforced() {
    for missing in LOCKS {
        let present = LOCKS.iter().copied().filter(|name| name != missing).collect::<Vec<_>>();

        let error = aiert::require_generated_categories(DMA, &present, PORTS).unwrap_err();

        assert!(error.contains("lock"), "{error}");
        assert!(error.contains(missing), "{error}");
    }
}

#[test]
fn every_required_port_category_is_enforced() {
    for missing in PORTS {
        let present = PORTS.iter().copied().filter(|name| name != missing).collect::<Vec<_>>();

        let error = aiert::require_generated_categories(DMA, LOCKS, &present).unwrap_err();

        assert!(error.contains("port"), "{error}");
        assert!(error.contains(missing), "{error}");
    }
}

#[test]
fn empty_parser_output_is_rejected() {
    let error = aiert::require_generated_categories(&[], &[], &[]).unwrap_err();

    assert!(error.contains("DMA"));
    assert!(error.contains("compute_dma"));
}
