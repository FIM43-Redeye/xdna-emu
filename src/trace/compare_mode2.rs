//! Mode-2 trace comparator: three-layer HW/EMU comparison.
//!
//! See spec section 8 of `docs/archive/specs/2026-04-29-a2b-mode2-design.md`.
//! The Layer 1 + Layer 2 results gate test pass/fail; Layer 3 (atom windows)
//! is informational only -- DMA-stall nondeterminism makes it noisy in
//! aggregate, so it never gates a test.

use crate::trace::compare::TileKey;
use crate::trace::mode2_decode::{decode, Mode2Frame};

/// Outcome of comparing one tile's mode-2 streams.
#[derive(Debug, Clone)]
pub struct Mode2CompareResult {
    pub tile: TileKey,
    pub layer1: PcSequenceLayer,
    pub layer2: LcLayer,
    pub layer3: AtomWindowLayer,
    /// True iff layer1 + layer2 both have zero divergences.
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct PcSequenceLayer {
    pub hw_count: usize,
    pub emu_count: usize,
    /// Index of first divergence; None means streams agreed up through
    /// min(hw_count, emu_count).
    pub first_diverge: Option<usize>,
    pub samples: Vec<PcSampleRow>,
}

#[derive(Debug, Clone)]
pub struct PcSampleRow {
    pub idx: usize,
    pub hw_pc: u16,
    pub emu_pc: u16,
}

#[derive(Debug, Clone)]
pub struct LcLayer {
    pub hw_count: usize,
    pub emu_count: usize,
    pub first_diverge: Option<usize>,
    pub samples: Vec<LcSampleRow>,
}

#[derive(Debug, Clone)]
pub struct LcSampleRow {
    pub idx: usize,
    pub hw_flag: u8,
    pub hw_count_value: u32,
    pub emu_flag: u8,
    pub emu_count_value: u32,
}

#[derive(Debug, Clone, Default)]
pub struct AtomWindowLayer {
    pub windows: Vec<AtomWindow>,
}

#[derive(Debug, Clone)]
pub struct AtomWindow {
    pub from_pc: u16,
    pub to_pc: u16,
    pub hw_atoms: usize,
    pub emu_atoms: usize,
}

fn extract_pcs(frames: &[Mode2Frame]) -> Vec<u16> {
    frames
        .iter()
        .filter_map(|f| match f {
            Mode2Frame::NewPc { pc } => Some(*pc),
            _ => None,
        })
        .collect()
}

fn compare_layer1(hw: &[u16], emu: &[u16]) -> PcSequenceLayer {
    let mut samples = Vec::new();
    let mut first = None;
    let n = hw.len().min(emu.len());
    for i in 0..n {
        if hw[i] != emu[i] {
            samples.push(PcSampleRow { idx: i, hw_pc: hw[i], emu_pc: emu[i] });
            if first.is_none() {
                first = Some(i);
            }
        }
    }
    if hw.len() != emu.len() && first.is_none() {
        first = Some(n);
    }
    PcSequenceLayer { hw_count: hw.len(), emu_count: emu.len(), first_diverge: first, samples }
}

fn extract_lcs(frames: &[Mode2Frame]) -> Vec<(u8, u32)> {
    frames
        .iter()
        .filter_map(|f| match f {
            Mode2Frame::Lc { flag, count } => Some((*flag, *count)),
            _ => None,
        })
        .collect()
}

fn compare_layer2(hw: &[(u8, u32)], emu: &[(u8, u32)]) -> LcLayer {
    let mut samples = Vec::new();
    let mut first = None;
    let n = hw.len().min(emu.len());
    for i in 0..n {
        if hw[i] != emu[i] {
            samples.push(LcSampleRow {
                idx: i,
                hw_flag: hw[i].0,
                hw_count_value: hw[i].1,
                emu_flag: emu[i].0,
                emu_count_value: emu[i].1,
            });
            if first.is_none() {
                first = Some(i);
            }
        }
    }
    if hw.len() != emu.len() && first.is_none() {
        first = Some(n);
    }
    LcLayer { hw_count: hw.len(), emu_count: emu.len(), first_diverge: first, samples }
}

fn extract_pc_segments(frames: &[Mode2Frame]) -> Vec<(u16, u16, usize)> {
    let mut out = Vec::new();
    let mut current_anchor: Option<u16> = None;
    let mut atom_count_in_segment: usize = 0;
    for f in frames {
        match f {
            Mode2Frame::Atom { .. } => {
                atom_count_in_segment += 1;
            }
            Mode2Frame::Repeat0 { count } => {
                atom_count_in_segment += *count as usize;
            }
            Mode2Frame::Repeat1 { count } => {
                atom_count_in_segment += *count as usize;
            }
            Mode2Frame::NewPc { pc } => {
                if let Some(prev) = current_anchor {
                    out.push((prev, *pc, atom_count_in_segment));
                }
                current_anchor = Some(*pc);
                atom_count_in_segment = 0;
            }
            _ => {}
        }
    }
    out
}

fn compare_layer3(hw: &[Mode2Frame], emu: &[Mode2Frame]) -> AtomWindowLayer {
    let hw_segs = extract_pc_segments(hw);
    let emu_segs = extract_pc_segments(emu);
    let mut windows = Vec::new();
    let n = hw_segs.len().min(emu_segs.len());
    for i in 0..n {
        windows.push(AtomWindow {
            from_pc: hw_segs[i].0,
            to_pc: hw_segs[i].1,
            hw_atoms: hw_segs[i].2,
            emu_atoms: emu_segs[i].2,
        });
    }
    AtomWindowLayer { windows }
}

/// Compare HW and EMU mode-2 byte streams for one tile.
///
/// Use this entry point only for synthetic single-tile streams (unit
/// tests, hand-encoded fixtures) -- real captures from the runner
/// arrive packet-framed and must be demuxed via parse-trace.py first.
/// For sweep results, use [`compare_mode2_from_events_files`].
pub fn compare_mode2_for_tile(hw_stream: &[u8], emu_stream: &[u8], tile: TileKey) -> Mode2CompareResult {
    let hw_frames = decode(hw_stream);
    let emu_frames = decode(emu_stream);
    compare_mode2_for_tile_from_frames(&hw_frames, &emu_frames, tile)
}

/// Compare HW and EMU mode-2 frame streams for one tile.
///
/// Returns a result with all three layers populated. `passed` is `true`
/// iff Layer 1 (PC sequence) and Layer 2 (LC count + flag) both have
/// zero divergences. Layer 3 (atom windows) is informational only.
pub fn compare_mode2_for_tile_from_frames(
    hw_frames: &[Mode2Frame],
    emu_frames: &[Mode2Frame],
    tile: TileKey,
) -> Mode2CompareResult {
    let hw_pcs = extract_pcs(hw_frames);
    let emu_pcs = extract_pcs(emu_frames);
    let layer1 = compare_layer1(&hw_pcs, &emu_pcs);

    let hw_lcs = extract_lcs(hw_frames);
    let emu_lcs = extract_lcs(emu_frames);
    let layer2 = compare_layer2(&hw_lcs, &emu_lcs);
    let layer3 = compare_layer3(hw_frames, emu_frames);
    let passed = layer1.first_diverge.is_none() && layer2.first_diverge.is_none();
    Mode2CompareResult { tile, layer1, layer2, layer3, passed }
}

/// Outcome of loading + comparing a sweep's mode-2 events JSON pair.
#[derive(Debug, Clone)]
pub struct Mode2EventsCompareReport {
    /// Per-tile results for tiles present in BOTH HW and EMU.
    pub per_tile: Vec<Mode2CompareResult>,
    /// Tile keys that appear in HW only.
    pub hw_only: Vec<TileKey>,
    /// Tile keys that appear in EMU only.
    pub emu_only: Vec<TileKey>,
}

#[derive(Debug)]
pub enum Mode2EventsCompareError {
    Read {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    Parse {
        path: std::path::PathBuf,
        message: String,
    },
}

impl std::fmt::Display for Mode2EventsCompareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read { path, source } => write!(f, "read {}: {}", path.display(), source),
            Self::Parse { path, message } => write!(f, "parse {}: {}", path.display(), message),
        }
    }
}

impl std::error::Error for Mode2EventsCompareError {}

/// Load HW and EMU events-JSON files (as produced by `parse-trace.py
/// --trace-mode inst_exec`), pair tiles, and run the three-layer
/// comparator on each pair.
///
/// `remap_columns`: when true, the column field of each side's tile keys
/// is independently dense-remapped to 0..N-1. This lets HW (whose trace
/// packets carry the absolute physical column from the kernel placement,
/// e.g. col=1) compare against EMU (whose trace_unit emits the
/// array-local column, always starting from 0) without spurious
/// "HW only" / "EMU only" reports for what is logically the same tile.
/// Mirrors `crate::trace::compare::remap_tile_columns` for the mode-1
/// path. See note in `xdna-emu/CLAUDE.md` ("Trace column offset:
/// emulator col=0 vs HW col=start_col").
pub fn compare_mode2_from_events_files(
    hw_path: &std::path::Path,
    emu_path: &std::path::Path,
    remap_columns: bool,
) -> Result<Mode2EventsCompareReport, Mode2EventsCompareError> {
    let mut hw_tiles = load_events_file(hw_path)?;
    let mut emu_tiles = load_events_file(emu_path)?;
    if remap_columns {
        hw_tiles = remap_tile_columns(&hw_tiles);
        emu_tiles = remap_tile_columns(&emu_tiles);
    }

    let mut per_tile = Vec::new();
    let mut hw_only = Vec::new();
    let mut emu_only = Vec::new();

    let mut emu_keys: std::collections::BTreeSet<TileKey> = emu_tiles.keys().copied().collect();
    let mut hw_keys: Vec<TileKey> = hw_tiles.keys().copied().collect();
    hw_keys.sort();
    for tile in hw_keys {
        match emu_tiles.get(&tile) {
            Some(emu_frames) => {
                let hw_frames = &hw_tiles[&tile];
                per_tile.push(compare_mode2_for_tile_from_frames(hw_frames, emu_frames, tile));
                emu_keys.remove(&tile);
            }
            None => hw_only.push(tile),
        }
    }
    emu_only.extend(emu_keys);

    Ok(Mode2EventsCompareReport { per_tile, hw_only, emu_only })
}

/// Independently dense-remap each tile key's column to 0..N-1, preserving
/// the row + pkt_type within each side. Per-side remap is intentional: HW
/// and EMU may report different absolute columns for the same logical
/// tile, so collapsing each side independently is what makes them line up.
fn remap_tile_columns(
    tiles: &std::collections::BTreeMap<TileKey, Vec<Mode2Frame>>,
) -> std::collections::BTreeMap<TileKey, Vec<Mode2Frame>> {
    let cols: std::collections::BTreeSet<u8> = tiles.keys().map(|k| k.col).collect();
    let col_map: std::collections::HashMap<u8, u8> =
        cols.into_iter().enumerate().map(|(i, c)| (c, i as u8)).collect();
    tiles
        .iter()
        .map(|(k, v)| {
            let new_key = TileKey { col: col_map[&k.col], row: k.row, pkt_type: k.pkt_type };
            (new_key, v.clone())
        })
        .collect()
}

/// Parse a parse-trace events JSON into per-tile frame lists.
///
/// The JSON shape is:
/// ```text
/// { "tiles": { "<pkt_type>,<row>,<col>": [ {"type": "..."}, ... ], ... } }
/// ```
fn load_events_file(
    path: &std::path::Path,
) -> Result<std::collections::BTreeMap<TileKey, Vec<Mode2Frame>>, Mode2EventsCompareError> {
    let bytes = std::fs::read(path)
        .map_err(|e| Mode2EventsCompareError::Read { path: path.to_path_buf(), source: e })?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| Mode2EventsCompareError::Parse { path: path.to_path_buf(), message: e.to_string() })?;
    let tiles =
        value
            .get("tiles")
            .and_then(|v| v.as_object())
            .ok_or_else(|| Mode2EventsCompareError::Parse {
                path: path.to_path_buf(),
                message: "missing or non-object \"tiles\" field".into(),
            })?;
    let mut out = std::collections::BTreeMap::new();
    for (key, events) in tiles {
        let tile = parse_tile_key(key)
            .map_err(|message| Mode2EventsCompareError::Parse { path: path.to_path_buf(), message })?;
        let arr = events.as_array().ok_or_else(|| Mode2EventsCompareError::Parse {
            path: path.to_path_buf(),
            message: format!("tile {} value is not an array", key),
        })?;
        let mut frames = Vec::with_capacity(arr.len());
        for ev in arr {
            if let Some(frame) = event_to_frame(ev) {
                frames.push(frame);
            }
        }
        out.insert(tile, frames);
    }
    Ok(out)
}

/// "<pkt_type>,<row>,<col>" -> TileKey.
fn parse_tile_key(s: &str) -> Result<TileKey, String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        return Err(format!("tile key {:?} not 'pkt_type,row,col'", s));
    }
    let pkt_type: u8 = parts[0].trim().parse().map_err(|_| format!("tile key {:?}: bad pkt_type", s))?;
    let row: u8 = parts[1].trim().parse().map_err(|_| format!("tile key {:?}: bad row", s))?;
    let col: u8 = parts[2].trim().parse().map_err(|_| format!("tile key {:?}: bad col", s))?;
    Ok(TileKey { col, row, pkt_type })
}

/// Map one parse-trace event dict to a [`Mode2Frame`].
///
/// Returns None for event types we don't model (currently none reach
/// here -- Filler frames are absorbed by the upstream decoder before
/// they reach this JSON shape).
fn event_to_frame(ev: &serde_json::Value) -> Option<Mode2Frame> {
    let ty = ev.get("type")?.as_str()?;
    match ty {
        "Start" => {
            let anchor_pc = ev.get("anchor_pc").and_then(|v| v.as_u64()).unwrap_or(0) as u16;
            Some(Mode2Frame::Start { anchor_pc })
        }
        "New_PC" => {
            let pc = ev.get("pc").and_then(|v| v.as_u64()).unwrap_or(0) as u16;
            Some(Mode2Frame::NewPc { pc })
        }
        "E_atom" => Some(Mode2Frame::Atom { executed: true }),
        "N_atom" => Some(Mode2Frame::Atom { executed: false }),
        "LC" => {
            let flag = ev.get("flag").and_then(|v| v.as_u64()).unwrap_or(0) as u8;
            let count = ev.get("count").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            Some(Mode2Frame::Lc { flag, count })
        }
        "Repeat" => {
            // Upstream decoder collapses Repeat0 (4-bit count) and
            // Repeat1 (10-bit count) into one "Repeat" event. The
            // distinction doesn't matter to any layer of the
            // comparator, so just pick one variant by count.
            let count = ev.get("count").and_then(|v| v.as_u64()).unwrap_or(0);
            if count <= 0xF {
                Some(Mode2Frame::Repeat0 { count: count as u8 })
            } else {
                Some(Mode2Frame::Repeat1 { count: count as u16 })
            }
        }
        "Sync" => Some(Mode2Frame::Sync),
        "Stop" => Some(Mode2Frame::Stop),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::compare::TileKey;

    /// Encode a sequence of New_PC frames + Filler0 padding.
    fn encode_pcs(pcs: &[u16]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut bit_pos: u32 = 0;
        let mut word: u32 = 0;
        let push_bits = |val: u32, count: u32, out: &mut Vec<u8>, word: &mut u32, bit_pos: &mut u32| {
            for i in (0..count).rev() {
                let b = (val >> i) & 1;
                *word = (*word << 1) | b;
                *bit_pos += 1;
                if *bit_pos == 32 {
                    out.push((*word >> 24) as u8);
                    out.push((*word >> 16) as u8);
                    out.push((*word >> 8) as u8);
                    out.push(*word as u8);
                    *word = 0;
                    *bit_pos = 0;
                }
            }
        };
        for &pc in pcs {
            let frame = (0b10u32 << 14) | (pc as u32 & 0x3FFF);
            push_bits(frame, 16, &mut out, &mut word, &mut bit_pos);
        }
        // Pad with Filler0 nibbles
        while bit_pos != 0 {
            push_bits(0b0010, 4, &mut out, &mut word, &mut bit_pos);
        }
        out
    }

    /// Encode a sequence of LC long frames.
    /// Each (flag, count) pair becomes one 32-bit aligned LC frame.
    fn encode_lcs(pairs: &[(u8, u32)]) -> Vec<u8> {
        let mut out = Vec::new();
        for &(flag, count) in pairs {
            let word = (0b010u32 << 29) | ((flag as u32 & 1) << 28) | (count & 0x0FFFFFFF);
            out.push((word >> 24) as u8);
            out.push((word >> 16) as u8);
            out.push((word >> 8) as u8);
            out.push(word as u8);
        }
        out
    }

    fn key() -> TileKey {
        TileKey { col: 0, row: 2, pkt_type: 0 }
    }

    #[test]
    fn layer1_no_divergence_when_pcs_match() {
        let hw = encode_pcs(&[0x100, 0x200, 0x300]);
        let emu = encode_pcs(&[0x100, 0x200, 0x300]);
        let r = compare_mode2_for_tile(&hw, &emu, key());
        assert!(r.layer1.first_diverge.is_none());
        assert_eq!(r.layer1.hw_count, 3);
        assert_eq!(r.layer1.emu_count, 3);
        assert!(r.passed);
    }

    #[test]
    fn layer1_finds_divergence() {
        let hw = encode_pcs(&[0x100, 0x200, 0x300]);
        let emu = encode_pcs(&[0x100, 0x250, 0x300]);
        let r = compare_mode2_for_tile(&hw, &emu, key());
        assert_eq!(r.layer1.first_diverge, Some(1));
        assert!(!r.passed);
    }

    #[test]
    fn layer1_handles_count_mismatch() {
        let hw = encode_pcs(&[0x100, 0x200, 0x300]);
        let emu = encode_pcs(&[0x100, 0x200]);
        let r = compare_mode2_for_tile(&hw, &emu, key());
        assert_eq!(r.layer1.hw_count, 3);
        assert_eq!(r.layer1.emu_count, 2);
        assert!(r.layer1.first_diverge.is_some() || !r.passed);
    }

    #[test]
    fn layer2_lc_count_match() {
        let hw = encode_lcs(&[(1, 8), (1, 4)]);
        let emu = encode_lcs(&[(1, 8), (1, 4)]);
        let r = compare_mode2_for_tile(&hw, &emu, key());
        assert!(r.layer2.first_diverge.is_none());
        assert!(r.passed);
    }

    #[test]
    fn layer2_lc_count_mismatch() {
        let hw = encode_lcs(&[(1, 8), (1, 4)]);
        let emu = encode_lcs(&[(1, 8), (0, 4)]); // flag differs
        let r = compare_mode2_for_tile(&hw, &emu, key());
        assert_eq!(r.layer2.first_diverge, Some(1));
        assert!(!r.passed);
    }

    #[test]
    fn layer3_reports_atoms_per_pc_segment() {
        // Build streams: PC(0x100) + 5 E_atoms + PC(0x200) + 3 E_atoms + PC(0x300)
        // For both HW and EMU. Atom counts include only inter-PC region.
        let hw = build_pc_atom_stream(&[(0x100, 5), (0x200, 3), (0x300, 0)]);
        let emu = hw.clone();
        let r = compare_mode2_for_tile(&hw, &emu, key());
        // Expect 2 windows: 0x100->0x200 (5 atoms) and 0x200->0x300 (3 atoms).
        assert_eq!(r.layer3.windows.len(), 2);
        assert_eq!(r.layer3.windows[0].from_pc, 0x100);
        assert_eq!(r.layer3.windows[0].to_pc, 0x200);
        assert_eq!(r.layer3.windows[0].hw_atoms, 5);
        assert_eq!(r.layer3.windows[0].emu_atoms, 5);
        assert_eq!(r.layer3.windows[1].from_pc, 0x200);
        assert_eq!(r.layer3.windows[1].to_pc, 0x300);
        assert_eq!(r.layer3.windows[1].hw_atoms, 3);
        assert_eq!(r.layer3.windows[1].emu_atoms, 3);
    }

    /// Build a stream that interleaves PC anchors and trailing E_atoms.
    /// Each tuple is (PC, atom_count) — atom_count atoms get emitted after
    /// the PC anchor. Stream ends with Filler0 padding to word boundary.
    fn build_pc_atom_stream(specs: &[(u16, usize)]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut bit_pos: u32 = 0;
        let mut word: u32 = 0;
        let push_bits = |val: u32, count: u32, out: &mut Vec<u8>, word: &mut u32, bit_pos: &mut u32| {
            for i in (0..count).rev() {
                let b = (val >> i) & 1;
                *word = (*word << 1) | b;
                *bit_pos += 1;
                if *bit_pos == 32 {
                    out.push((*word >> 24) as u8);
                    out.push((*word >> 16) as u8);
                    out.push((*word >> 8) as u8);
                    out.push(*word as u8);
                    *word = 0;
                    *bit_pos = 0;
                }
            }
        };
        for &(pc, atoms) in specs {
            let frame = (0b10u32 << 14) | (pc as u32 & 0x3FFF);
            push_bits(frame, 16, &mut out, &mut word, &mut bit_pos);
            for _ in 0..atoms {
                push_bits(0b0001, 4, &mut out, &mut word, &mut bit_pos); // E_atom
            }
        }
        while bit_pos != 0 {
            push_bits(0b0010, 4, &mut out, &mut word, &mut bit_pos);
        }
        out
    }

    fn write_events_json(path: &std::path::Path, tile_key: &str, events_json: &str) {
        let body = format!(
            "{{\"schema_version\":1,\"trace_mode\":\"inst_exec\",\"tiles\":{{\"{}\":{}}}}}",
            tile_key, events_json
        );
        std::fs::write(path, body).unwrap();
    }

    #[test]
    fn events_json_round_trips_pc_and_lc() {
        let tmpdir = std::path::PathBuf::from(std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string()));
        let dir = tmpdir.join("compare_mode2_events_json_round_trip");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hw = dir.join("hw.json");
        let emu = dir.join("emu.json");
        let body = r#"[
            {"type":"Start","anchor_pc":256},
            {"type":"New_PC","pc":256},
            {"type":"E_atom"},
            {"type":"New_PC","pc":512},
            {"type":"LC","flag":0,"count":8},
            {"type":"Stop"}
        ]"#;
        write_events_json(&hw, "0,2,0", body);
        write_events_json(&emu, "0,2,0", body);

        let report = compare_mode2_from_events_files(&hw, &emu, false).unwrap();
        assert_eq!(report.per_tile.len(), 1);
        assert_eq!(report.hw_only, vec![]);
        assert_eq!(report.emu_only, vec![]);
        let r = &report.per_tile[0];
        assert!(r.passed);
        assert_eq!(r.tile, TileKey { col: 0, row: 2, pkt_type: 0 });
        assert_eq!(r.layer1.hw_count, 2);
        assert_eq!(r.layer2.hw_count, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn events_json_reports_tiles_only_in_one_side() {
        let tmpdir = std::path::PathBuf::from(std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string()));
        let dir = tmpdir.join("compare_mode2_events_json_one_side");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hw = dir.join("hw.json");
        let emu = dir.join("emu.json");
        let pc = r#"[{"type":"New_PC","pc":256}]"#;
        std::fs::write(&hw, format!(r#"{{"tiles":{{"0,2,0":{},"0,3,0":{}}}}}"#, pc, pc)).unwrap();
        std::fs::write(&emu, format!(r#"{{"tiles":{{"0,2,0":{}}}}}"#, pc)).unwrap();

        let report = compare_mode2_from_events_files(&hw, &emu, false).unwrap();
        assert_eq!(report.per_tile.len(), 1);
        assert_eq!(report.per_tile[0].tile.row, 2);
        assert_eq!(report.hw_only, vec![TileKey { col: 0, row: 3, pkt_type: 0 }]);
        assert_eq!(report.emu_only, vec![]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn events_json_handles_repeat_count() {
        let tmpdir = std::path::PathBuf::from(std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string()));
        let dir = tmpdir.join("compare_mode2_events_json_repeat");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hw = dir.join("hw.json");
        let emu = dir.join("emu.json");
        let body = r#"[
            {"type":"New_PC","pc":256},
            {"type":"E_atom"},
            {"type":"Repeat","count":7},
            {"type":"New_PC","pc":512}
        ]"#;
        write_events_json(&hw, "0,2,0", body);
        write_events_json(&emu, "0,2,0", body);

        let report = compare_mode2_from_events_files(&hw, &emu, false).unwrap();
        let r = &report.per_tile[0];
        assert_eq!(r.layer3.windows.len(), 1);
        assert_eq!(r.layer3.windows[0].hw_atoms, 8);
        assert_eq!(r.layer3.windows[0].emu_atoms, 8);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn events_json_remap_columns_pairs_offset_tiles() {
        // HW reports the absolute placement column (e.g., col=1 when the
        // kernel was placed starting at column 1). EMU emits its
        // array-local column, always starting at 0. Without remap, the
        // mode-2 comparator would log "HW only (1,2)" + "EMU only (0,2)"
        // for the same logical tile. With remap, both sides dense-remap
        // to col=0 and the comparator pairs them.
        let tmpdir = std::path::PathBuf::from(std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string()));
        let dir = tmpdir.join("compare_mode2_remap_columns");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hw = dir.join("hw.json");
        let emu = dir.join("emu.json");
        let body = r#"[{"type":"Start","anchor_pc":256},{"type":"New_PC","pc":256}]"#;
        write_events_json(&hw, "0,2,1", body);
        write_events_json(&emu, "0,2,0", body);

        // Without remap: no common tile.
        let no_remap = compare_mode2_from_events_files(&hw, &emu, false).unwrap();
        assert_eq!(no_remap.per_tile.len(), 0);
        assert_eq!(no_remap.hw_only.len(), 1);
        assert_eq!(no_remap.emu_only.len(), 1);

        // With remap: both sides collapse to col=0 and pair up.
        let with_remap = compare_mode2_from_events_files(&hw, &emu, true).unwrap();
        assert_eq!(with_remap.per_tile.len(), 1);
        assert_eq!(with_remap.hw_only, vec![]);
        assert_eq!(with_remap.emu_only, vec![]);
        let r = &with_remap.per_tile[0];
        assert_eq!(r.tile, TileKey { col: 0, row: 2, pkt_type: 0 });
        assert!(r.passed);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
