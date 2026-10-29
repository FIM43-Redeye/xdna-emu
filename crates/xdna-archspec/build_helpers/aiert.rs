//! Fail-loud aie-rt preprocessing and generated-category validation.

use std::path::Path;
use std::process::Command;

const INCLUDE_SUBDIRS: &[&str] = &[
    "",
    "common",
    "core",
    "device",
    "dma",
    "events",
    "global",
    "interrupt",
    "io_backend",
    "lite",
    "locks",
    "memory",
    "npi",
    "perfcnt",
    "pl",
    "pm",
    "routing",
    "stream_switch",
    "timer",
    "trace",
    "util",
];

const REQUIRED_DMA: &[&str] = &["compute_dma", "memtile_dma", "shim_dma"];
const REQUIRED_LOCKS: &[&str] = &["compute_locks", "memtile_locks", "shim_locks"];
const REQUIRED_PORTS: &[&str] = &[
    "COMPUTE_MASTER_PORTS",
    "COMPUTE_SLAVE_PORTS",
    "MEMTILE_MASTER_PORTS",
    "MEMTILE_SLAVE_PORTS",
    "SHIM_MASTER_PORTS",
    "SHIM_SLAVE_PORTS",
];

pub fn preprocess(driver_src: &Path, compiler: &Path) -> Result<String, String> {
    let reginit = driver_src.join("global/xaiemlgbl_reginit.c");
    if !reginit.is_file() {
        return Err(format!(
            "aie-rt source is missing required global/xaiemlgbl_reginit.c: {}",
            reginit.display()
        ));
    }

    let mut command = Command::new(compiler);
    command.arg("-E");
    for subdir in INCLUDE_SUBDIRS {
        let include = if subdir.is_empty() {
            driver_src.to_path_buf()
        } else {
            driver_src.join(subdir)
        };
        command.arg("-I").arg(include);
    }
    command.arg(&reginit);

    let output = command
        .output()
        .map_err(|error| format!("cannot run aie-rt preprocessor {}: {error}", compiler.display()))?;
    if !output.status.success() {
        let status = output
            .status
            .code()
            .map_or_else(|| output.status.to_string(), |code| code.to_string());
        return Err(format!(
            "aie-rt preprocessor {} exited with status {}: {}",
            compiler.display(),
            status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

pub fn require_generated_categories(dma: &[&str], locks: &[&str], ports: &[&str]) -> Result<(), String> {
    require_categories("DMA", dma, REQUIRED_DMA)?;
    require_categories("lock", locks, REQUIRED_LOCKS)?;
    require_categories("port", ports, REQUIRED_PORTS)
}

fn require_categories(kind: &str, actual: &[&str], required: &[&str]) -> Result<(), String> {
    let missing = required
        .iter()
        .copied()
        .filter(|name| !actual.contains(name))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(());
    }

    Err(format!(
        "aie-rt extraction missing required {} categories: {}; parsed: {}",
        kind,
        missing.join(", "),
        actual.join(", ")
    ))
}
