//! Shared xclbin load path. Both the in-process test suite (xclbin_suite) and
//! the visual debugger use load_engine so there is exactly one place that
//! knows the parse -> CDO -> ELF -> sync incantation.

use std::path::Path;

use crate::interpreter::engine::InterpreterEngine;
use crate::npu::HostBuffer;
use crate::parser::cdo::find_cdo_offset;
use crate::parser::xclbin::SectionKind;
use crate::parser::{AiePartition, Cdo, Xclbin};

/// Parse `xclbin_path`, apply its CDO to a fresh NPU1 engine, load every core
/// ELF found alongside it, and sync core-enabled state. Returns a
/// ready-to-step engine. Does not set a stall threshold, populate host
/// memory, or build an `NpuExecutor` -- those are the caller's concern.
pub fn load_engine(xclbin_path: &Path) -> Result<InterpreterEngine, String> {
    let xclbin = Xclbin::from_file(xclbin_path).map_err(|e| format!("Failed to load xclbin: {}", e))?;

    let section = xclbin
        .find_section(SectionKind::AiePartition)
        .ok_or_else(|| "No AIE partition in xclbin".to_string())?;

    let partition =
        AiePartition::parse(section.data()).map_err(|e| format!("Failed to parse AIE partition: {}", e))?;

    let pdi = partition
        .primary_pdi()
        .ok_or_else(|| "No primary PDI in partition".to_string())?;

    let cdo_offset = find_cdo_offset(pdi.pdi_image).ok_or_else(|| "No CDO found in PDI".to_string())?;

    let cdo = Cdo::parse(&pdi.pdi_image[cdo_offset..]).map_err(|e| format!("Failed to parse CDO: {}", e))?;

    let mut engine = InterpreterEngine::new_npu1();

    // Emulate firmware's MSG_OP_CREATE_CONTEXT response before applying the
    // CDO: the driver/firmware ungates the partition's columns at context
    // create (aie-rt `_XAieMl_RequestTiles` writes `Column_Clock_Control =
    // 0x1`), which the user CDO never carries. Without this the columns
    // boot gated, `step_all_dma` skips them, and the shim DMA never moves
    // data (BUG-A). The XRT-plugin path does the equivalent via the
    // `xdna_emu_assign_partition` FFI hook; the in-process runner applies
    // the CDO unrelocated, so the partition occupies physical columns
    // [0, column_width).
    engine.device_mut().assign_partition_columns(0, partition.column_width() as u8);

    engine
        .device_mut()
        .apply_cdo(&cdo)
        .map_err(|e| format!("Failed to apply CDO: {}", e))?;

    // Load ELF files from the project directory alongside the xclbin.
    // IMPORTANT: load ELFs BEFORE sync_cores_from_device() so the engine
    // sees the cores as enabled after loading program code.
    for (col, row, path) in find_elf_files(xclbin_path) {
        let data = std::fs::read(&path).map_err(|e| format!("Failed to read ELF {:?}: {}", path, e))?;
        engine
            .load_elf_bytes(col as usize, row as usize, &data)
            .map_err(|e| format!("Failed to load ELF into ({},{}): {}", col, row, e))?;
    }

    // Sync core enabled state from device tiles to engine, called AFTER
    // loading ELFs so the engine sees the loaded cores.
    engine.sync_cores_from_device();

    Ok(engine)
}

/// The default DDR layout used when a test has no explicit buffer spec:
/// input (0x0, 4KB), middle (0x1000, 256B), output (0x2000, 4KB).
pub fn default_host_buffers() -> Vec<HostBuffer> {
    vec![
        HostBuffer { address: 0x0000, size: 4096 },
        HostBuffer { address: 0x1000, size: 256 },
        HostBuffer { address: 0x2000, size: 4096 },
    ]
}

/// Find core ELF files alongside `xclbin_path`, mirroring
/// `XclbinTest::project_dir` discovery + `find_elf_files` in
/// `src/testing/xclbin_suite.rs`. Duplicated here (not called through
/// `testing`) because `loading` must stay buildable without the `tooling`
/// feature that gates the test harness.
// ponytail: small mirror of testing::xclbin_suite's project-dir/ELF scan;
// unify if a third caller needs it.
fn find_elf_files(xclbin_path: &Path) -> Vec<(u8, u8, std::path::PathBuf)> {
    let Some(parent) = xclbin_path.parent() else {
        return Vec::new();
    };

    let project_dir =
        ["aie_arch.mlir.prj", "aie.mlir.prj"]
            .iter()
            .map(|name| parent.join(name))
            .find(|p| p.is_dir())
            .or_else(|| {
                std::fs::read_dir(parent).ok()?.flatten().map(|e| e.path()).find(|p| {
                    p.is_dir() && p.file_name().is_some_and(|n| n.to_string_lossy().ends_with(".prj"))
                })
            });

    let Some(project_dir) = project_dir else {
        return Vec::new();
    };

    let mut elfs = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&project_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name() {
                let name_str = name.to_string_lossy();
                if name_str.ends_with(".elf") && name_str.contains("core_") {
                    if let Some((col, row)) = parse_core_coords(&name_str) {
                        elfs.push((col, row, path));
                    }
                }
            }
        }
    }
    elfs
}

/// Parse core coordinates from filename like "main_core_0_2.elf".
fn parse_core_coords(name: &str) -> Option<(u8, u8)> {
    let core_idx = name.find("core_")?;
    let after_core = &name[core_idx + 5..];
    let parts: Vec<&str> = after_core.split('_').take(2).collect();
    if parts.len() >= 2 {
        let col: u8 = parts[0].parse().ok()?;
        let row_str = parts[1].trim_end_matches(".elf");
        let row: u8 = row_str.parse().ok()?;
        return Some((col, row));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin")
    }

    #[test]
    fn load_engine_produces_a_ready_engine() {
        let path = fixture();
        if !path.exists() {
            eprintln!("SKIP load_engine_produces_a_ready_engine: fixture not built at {}", path.display());
            return;
        }
        let engine = load_engine(&path).expect("load should succeed");
        // NPU1 array is instantiated and cores were enabled by ELF load.
        assert!(engine.device().array.cols() >= 1);
        assert!(engine.enabled_cores() >= 1, "at least one core should be enabled after ELF load");
    }

    #[test]
    fn default_host_buffers_match_the_default_ddr_layout() {
        let b = default_host_buffers();
        assert_eq!(b.len(), 3);
        assert_eq!((b[0].address, b[0].size), (0x0000, 4096));
        assert_eq!((b[1].address, b[1].size), (0x1000, 256));
        assert_eq!((b[2].address, b[2].size), (0x2000, 4096));
    }
}
