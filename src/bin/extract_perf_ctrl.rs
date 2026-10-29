//! Extract per-tile PERF_CTRL writes from an xclbin's CDO stream.
//!
//! Walks every CDO Write / MaskWrite command, decodes the tile-local
//! address, filters by the AM025 PERF_CTRL register offset windows, and
//! emits JSON keyed by (col, row).
//!
//! Tile-kind classification by row: shim = row 0, memtile = row 1,
//! compute = row >= 2. The compute tile addresses both the core
//! module (0x031500-0x03158C) and the core-memory module
//! (0x011000-0x011084); both are reported under the same (col, row).
//!
//! **Limitation, as found 2026-05-25**: IRON-style tests (the bulk of
//! the calibration corpus) configure perf counters via the runtime
//! NPU instruction stream (insts.bin), not via CDO at xclbin-load.
//! This extractor returns empty for those. The item #9 analysis path
//! (perf-counter-driven LOCK_STALL emission) uses inter-event
//! interval analysis on events.json instead -- see
//! `tools/lock-stall-intervals.py` -- which derives the effective
//! PERF_THRESHOLD empirically without needing the CDO readout.
//! Keep this tool for the minority of tests that DO configure PERF
//! at xclbin load.

use std::collections::BTreeMap;
use std::process;

use xdna_emu::parser::aie_partition::AiePartition;
use xdna_emu::parser::cdo::syntax::{Cdo, CdoRaw};
use xdna_emu::parser::cdo::find_cdo_offset;
use xdna_emu::parser::xclbin::{SectionKind, Xclbin};

#[derive(Clone, Copy)]
enum TileKind {
    Shim,
    Memtile,
    Compute,
}

impl TileKind {
    fn from_row(row: u8) -> Self {
        match row {
            0 => TileKind::Shim,
            1 => TileKind::Memtile,
            _ => TileKind::Compute,
        }
    }
    fn as_str(self) -> &'static str {
        match self {
            TileKind::Shim => "shim",
            TileKind::Memtile => "memtile",
            TileKind::Compute => "compute",
        }
    }
}

/// (tile-local offset, register name). The lookup is module-scoped:
/// shim uses the 0x031xxx core offsets but maps them to its own
/// Performance_Ctrl0/1 (no Control2). Compute tiles also have the
/// core-memory module at 0x011xxx. Memtile uses 0x091xxx.
fn perf_ctrl_register_name(tile: TileKind, offset: u32) -> Option<&'static str> {
    match tile {
        TileKind::Shim => match offset {
            0x031000 => Some("Performance_Ctrl0"),
            0x031008 => Some("Performance_Ctrl1"),
            0x031020 => Some("Performance_Counter0"),
            0x031024 => Some("Performance_Counter1"),
            0x031080 => Some("Performance_Counter0_Event_Value"),
            0x031084 => Some("Performance_Counter1_Event_Value"),
            _ => None,
        },
        TileKind::Memtile => match offset {
            0x091000 => Some("Performance_Control0"),
            0x091004 => Some("Performance_Control1"),
            0x091008 => Some("Performance_Control2"),
            0x091020 => Some("Performance_Counter0"),
            0x091024 => Some("Performance_Counter1"),
            0x091028 => Some("Performance_Counter2"),
            0x09102C => Some("Performance_Counter3"),
            0x091080 => Some("Performance_Counter0_Event_Value"),
            0x091084 => Some("Performance_Counter1_Event_Value"),
            0x091088 => Some("Performance_Counter2_Event_Value"),
            0x09108C => Some("Performance_Counter3_Event_Value"),
            _ => None,
        },
        TileKind::Compute => match offset {
            // Core module
            0x031500 => Some("core.Performance_Control0"),
            0x031504 => Some("core.Performance_Control1"),
            0x031508 => Some("core.Performance_Control2"),
            0x031520 => Some("core.Performance_Counter0"),
            0x031524 => Some("core.Performance_Counter1"),
            0x031528 => Some("core.Performance_Counter2"),
            0x03152C => Some("core.Performance_Counter3"),
            0x031580 => Some("core.Performance_Counter0_Event_Value"),
            0x031584 => Some("core.Performance_Counter1_Event_Value"),
            0x031588 => Some("core.Performance_Counter2_Event_Value"),
            0x03158C => Some("core.Performance_Counter3_Event_Value"),
            // Core-memory module
            0x011000 => Some("mem.Performance_Control0"),
            0x011008 => Some("mem.Performance_Control1"),
            0x011020 => Some("mem.Performance_Counter0"),
            0x011024 => Some("mem.Performance_Counter1"),
            0x011080 => Some("mem.Performance_Counter0_Event_Value"),
            0x011084 => Some("mem.Performance_Counter1_Event_Value"),
            _ => None,
        },
    }
}

struct PerfWrite {
    col: u8,
    row: u8,
    tile_kind: TileKind,
    offset: u32,
    reg_name: &'static str,
    value: u32,
    /// True for MaskWrite; the value reflects the masked-write result
    /// of new bits applied to a zero baseline, which is what the
    /// downstream tile state would be after the first such write.
    masked: bool,
    /// For MaskWrite, the mask itself; for Write, all-1s.
    mask: u32,
}

/// Returns (mask, value, masked?) for a write-style CDO command.
/// Non-write commands return None.
fn write_payload(cmd: &CdoRaw) -> Option<(u32, u32, bool)> {
    match cmd {
        CdoRaw::Write { value, .. } => Some((!0u32, *value, false)),
        CdoRaw::MaskWrite { mask, value, .. } => Some((*mask, *value, true)),
        CdoRaw::Write64 { value, .. } => Some((!0u32, *value, false)),
        CdoRaw::MaskWrite64 { mask, value, .. } => Some((*mask, *value, true)),
        _ => None,
    }
}

fn scan_pdi(pdi_image: &[u8]) -> Vec<PerfWrite> {
    // PDIs in xclbins are bootgen-wrapped (magic 0x11223344); strip the
    // wrapper to find the embedded CDO container.
    let cdo_offset = find_cdo_offset(pdi_image).unwrap_or(0);
    let cdo = match Cdo::parse(&pdi_image[cdo_offset..]) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for cmd in cdo.commands() {
        let Some((mask, value, masked)) = write_payload(&cmd) else {
            continue;
        };
        let Some((col, row, offset)) = cmd.decode_aie_address() else {
            continue;
        };
        let tile_kind = TileKind::from_row(row);
        let Some(reg_name) = perf_ctrl_register_name(tile_kind, offset) else {
            continue;
        };
        out.push(PerfWrite { col, row, tile_kind, offset, reg_name, value, masked, mask });
    }
    out
}

fn emit_json(xclbin: &str, writes: &[PerfWrite]) -> String {
    use serde_json::{json, Map, Value};

    let writes_arr: Vec<Value> = writes
        .iter()
        .map(|w| {
            json!({
                "col": w.col,
                "row": w.row,
                "tile_kind": w.tile_kind.as_str(),
                "offset": format!("0x{:06X}", w.offset),
                "reg_name": w.reg_name,
                "value": format!("0x{:08X}", w.value),
                "masked": w.masked,
                "mask": format!("0x{:08X}", w.mask),
            })
        })
        .collect();

    let mut by_tile: BTreeMap<(u8, u8), Map<String, Value>> = BTreeMap::new();
    for w in writes {
        let entry = by_tile.entry((w.col, w.row)).or_default();
        // For MaskWrite, accumulate by ORing into any existing bits at this
        // register so multiple partial writes compose. For full Write, the
        // value replaces what was there.
        let cur = entry
            .get(w.reg_name)
            .and_then(|v| v.as_str())
            .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
            .unwrap_or(0);
        let new = if w.masked {
            (cur & !w.mask) | (w.value & w.mask)
        } else {
            w.value
        };
        entry.insert(w.reg_name.to_string(), json!(format!("0x{:08X}", new)));
    }
    let mut tiles_arr: Vec<Value> = Vec::new();
    for ((col, row), regs) in by_tile {
        tiles_arr.push(json!({
            "col": col,
            "row": row,
            "tile_kind": TileKind::from_row(row).as_str(),
            "registers": Value::Object(regs),
        }));
    }

    let out = json!({
        "xclbin": xclbin,
        "writes": writes_arr,
        "by_tile": tiles_arr,
    });
    serde_json::to_string_pretty(&out).unwrap_or_else(|_| "{}".to_string())
}

fn usage() -> ! {
    eprintln!("Usage: extract-perf-ctrl <xclbin-path> [-o <out.json>]");
    eprintln!();
    eprintln!("Walks the xclbin's CDO stream, extracts every Write/MaskWrite");
    eprintln!("targeting a PERF_CTRL register, and emits per-tile JSON.");
    eprintln!("Tile-kind by row: shim=0, memtile=1, compute>=2.");
    process::exit(1);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut xclbin_path: Option<String> = None;
    let mut out_path: Option<String> = None;
    let mut debug = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                out_path = Some(args.get(i).unwrap_or_else(|| usage()).clone());
            }
            "--debug" => {
                debug = true;
            }
            "--help" | "-h" => usage(),
            other if other.starts_with('-') => {
                eprintln!("Unknown argument: {}", other);
                usage();
            }
            other => {
                if xclbin_path.is_some() {
                    eprintln!("Unexpected positional: {}", other);
                    usage();
                }
                xclbin_path = Some(other.to_string());
            }
        }
        i += 1;
    }
    let Some(xclbin_path) = xclbin_path else { usage() };

    let xclbin = Xclbin::from_file(&xclbin_path).unwrap_or_else(|e| {
        eprintln!("Error loading {}: {}", xclbin_path, e);
        process::exit(1);
    });

    let partition_section = xclbin.find_section(SectionKind::AiePartition).unwrap_or_else(|| {
        eprintln!("Error: xclbin has no AIE_PARTITION section");
        process::exit(1);
    });
    let partition = AiePartition::parse(partition_section.data()).unwrap_or_else(|e| {
        eprintln!("Error parsing AIE_PARTITION: {}", e);
        process::exit(1);
    });

    let mut all_writes: Vec<PerfWrite> = Vec::new();
    let mut debug_per_tile_offsets: BTreeMap<(u8, u8), BTreeMap<u32, u32>> = BTreeMap::new();
    let mut pdi_count = 0;
    for pdi in partition.pdis() {
        pdi_count += 1;
        if debug {
            eprintln!(
                "--- PDI #{}: uuid={} type={:?} image_len={} bytes ---",
                pdi_count,
                pdi.uuid,
                pdi.cdo_type,
                pdi.pdi_image.len()
            );
            let cdo_offset = find_cdo_offset(pdi.pdi_image).unwrap_or(0);
            match Cdo::parse(&pdi.pdi_image[cdo_offset..]) {
                Ok(cdo) => {
                    cdo.print_summary();
                    for cmd in cdo.commands() {
                        if let Some((col, row, off)) = cmd.decode_aie_address() {
                            if let Some((_, value, _)) = write_payload(&cmd) {
                                debug_per_tile_offsets.entry((col, row)).or_default().insert(off, value);
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("  Cdo::parse failed: {}", e);
                }
            }
        }
        all_writes.extend(scan_pdi(pdi.pdi_image));
    }
    if debug {
        eprintln!();
        eprintln!("--- Tile-write summary (all write-style commands) ---");
        for ((col, row), offsets) in &debug_per_tile_offsets {
            eprintln!("  tile ({}, {}): {} distinct offsets:", col, row, offsets.len());
            for (off, val) in offsets {
                let in_perf = (0x011000..=0x011084).contains(off)
                    || (0x031000..=0x03158C).contains(off)
                    || (0x091000..=0x09108C).contains(off);
                let mark = if in_perf { " <-- PERF" } else { "" };
                eprintln!("    0x{:06X} = 0x{:08X}{}", off, val, mark);
            }
        }
    }

    let report = emit_json(&xclbin_path, &all_writes);

    if let Some(out) = out_path {
        std::fs::write(&out, &report).unwrap_or_else(|e| {
            eprintln!("Error writing {}: {}", out, e);
            process::exit(1);
        });
        eprintln!("Wrote {} ({} perf-ctrl writes across {} tiles)", out, all_writes.len(), {
            let mut tiles = std::collections::HashSet::new();
            for w in &all_writes {
                tiles.insert((w.col, w.row));
            }
            tiles.len()
        });
    } else {
        print!("{}", report);
    }
}
