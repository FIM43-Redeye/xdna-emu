//! Data abstraction for the trace comparison visualizer.
//!
//! [`TraceSource`] is the trait that the rendering code programs against.
//! [`LoadedComparison`] is the concrete implementation that loads HW and
//! EMU trace directories, decodes them, runs the comparison, and stores
//! the results for rendering.

use std::collections::HashMap;
use std::path::Path;

use crate::trace::compare::{
    AnalysisOptions, BatchResult, EventsConfig, TileEvent, TileKey, compare_batch_with_opts,
    load_events_json, remap_tile_columns, shift_tile_columns,
};

use super::alignment::AlignmentMap;

// ============================================================================
// TraceSource trait
// ============================================================================

/// Abstraction over trace comparison data.
///
/// The rendering code uses this trait so it can work with different data
/// sources (file-loaded, streamed, synthetic test data) without coupling
/// to any particular loading mechanism.
pub trait TraceSource {
    /// All tile keys present in the comparison, sorted by divergence
    /// severity (most divergent first).
    fn tile_keys(&self) -> &[TileKey];

    /// The event slot name configuration for this batch.
    fn batch_config(&self) -> &EventsConfig;

    /// HW-side decoded events for a given tile.
    fn hw_events(&self, tile: &TileKey) -> &[TileEvent];

    /// EMU-side decoded events for a given tile.
    fn emu_events(&self, tile: &TileKey) -> &[TileEvent];

    /// The full comparison result for this batch.
    fn batch_result(&self) -> &BatchResult;

    /// The current alignment map (immutable access).
    fn alignment(&self) -> &AlignmentMap;

    /// The current alignment map (mutable access, for adding anchors).
    fn alignment_mut(&mut self) -> &mut AlignmentMap;
}

// ============================================================================
// LoadedComparison
// ============================================================================

/// Concrete [`TraceSource`] implementation that loads traces from disk.
///
/// Loads HW and EMU `trace_raw.bin` files, decodes them into per-tile
/// event lists, runs the batch comparison, and stores everything needed
/// for rendering.
pub struct LoadedComparison {
    /// Tile keys sorted by divergence count (descending).
    sorted_keys: Vec<TileKey>,
    /// Events configuration loaded from events.json.
    config: EventsConfig,
    /// Per-tile HW events.
    hw_events: HashMap<TileKey, Vec<TileEvent>>,
    /// Per-tile EMU events.
    emu_events: HashMap<TileKey, Vec<TileEvent>>,
    /// Comparison result.
    batch: BatchResult,
    /// Piecewise cycle alignment.
    alignment: AlignmentMap,
}

impl LoadedComparison {
    /// Load and compare traces from HW and EMU directories.
    ///
    /// Each directory must contain a `trace.events.json` file produced by
    /// `tools/parse-trace.py` (which invokes mlir-aie's decoder). The
    /// slot-name configuration is read from `events.json` in the HW dir if
    /// present (legacy aiecc format), otherwise recovered from the events
    /// JSON's own `slot_names` field.
    pub fn from_trace_dirs(hw_dir: &Path, emu_dir: &Path) -> Result<Self, String> {
        let hw_events_path = hw_dir.join("trace.events.json");
        let emu_events_path = emu_dir.join("trace.events.json");

        if !hw_events_path.exists() {
            return Err(format!(
                "HW trace events not found: {}\n\
                 (run tools/parse-trace.py --out-events on the HW bin first)",
                hw_events_path.display(),
            ));
        }
        if !emu_events_path.exists() {
            return Err(format!(
                "EMU trace events not found: {}\n\
                 (run tools/parse-trace.py --out-events on the EMU bin first)",
                emu_events_path.display(),
            ));
        }

        // Legacy slot-name override (aiecc events.json) if present; otherwise
        // slot_names from the HW events JSON wins (see compare_batch_with_opts).
        let config_override = load_events_config(hw_dir)?;

        // Run comparison.
        let batch = compare_batch_with_opts(
            &hw_events_path,
            &emu_events_path,
            &config_override,
            0, // batch index
            &AnalysisOptions::default(),
        )?;

        // Re-load events for rendering (comparison consumed them but didn't
        // return the tile maps; fine for now since traces are small). The
        // raw events.json keys carry physical (col, row), but the comparison
        // (and therefore sorted_keys below) normalizes by the per-side
        // placement origin -- mirror that here so HashMap lookups against
        // sorted_keys hit the right entries.
        let opts = AnalysisOptions::default();
        let (hw_events_raw, hw_config, hw_placement, _hw_modes) = load_events_json(&hw_events_path)?;
        let (emu_events_raw, _, emu_placement, _emu_modes) = load_events_json(&emu_events_path)?;
        let hw_events = match (hw_placement, opts.remap_columns) {
            (Some(p), _) => shift_tile_columns(&hw_events_raw, p),
            (None, true) => remap_tile_columns(&hw_events_raw),
            (None, false) => hw_events_raw,
        };
        let emu_events = match (emu_placement, opts.remap_columns) {
            (Some(p), _) => shift_tile_columns(&emu_events_raw, p),
            (None, true) => remap_tile_columns(&emu_events_raw),
            (None, false) => emu_events_raw,
        };
        let config = if !config_override.core_events.is_empty()
            || !config_override.mem_events.is_empty()
            || !config_override.memtile_events.is_empty()
        {
            config_override
        } else {
            hw_config
        };

        // Sort tile keys by divergence count (most divergent first).
        let mut sorted_keys: Vec<TileKey> = batch.tiles.iter().map(|(key, _)| *key).collect();
        sorted_keys.sort_by(|a, b| {
            let da = divergence_count(&batch, a);
            let db = divergence_count(&batch, b);
            db.cmp(&da).then_with(|| a.cmp(b))
        });

        // Build initial alignment from the first tile's t0 values.
        let alignment = if let Some((_, result)) = batch.tiles.first() {
            if result.hw_t0 != 0 || result.emu_t0 != 0 {
                AlignmentMap::from_single_anchor(result.hw_t0, result.emu_t0)
            } else {
                AlignmentMap::identity()
            }
        } else {
            AlignmentMap::identity()
        };

        Ok(Self { sorted_keys, config, hw_events, emu_events, batch, alignment })
    }
}

impl TraceSource for LoadedComparison {
    fn tile_keys(&self) -> &[TileKey] {
        &self.sorted_keys
    }

    fn batch_config(&self) -> &EventsConfig {
        &self.config
    }

    fn hw_events(&self, tile: &TileKey) -> &[TileEvent] {
        self.hw_events.get(tile).map(|v| v.as_slice()).unwrap_or(&[])
    }

    fn emu_events(&self, tile: &TileKey) -> &[TileEvent] {
        self.emu_events.get(tile).map(|v| v.as_slice()).unwrap_or(&[])
    }

    fn batch_result(&self) -> &BatchResult {
        &self.batch
    }

    fn alignment(&self) -> &AlignmentMap {
        &self.alignment
    }

    fn alignment_mut(&mut self) -> &mut AlignmentMap {
        &mut self.alignment
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Count divergent event types for a tile in a batch result.
///
/// A tile's divergence count is the number of edge results and level results
/// that have a `diverge_idx` set (i.e., at least one pair exceeds the
/// divergence threshold).
pub fn divergence_count(batch: &BatchResult, key: &TileKey) -> usize {
    let tile_result = match batch.tiles.iter().find(|(k, _)| k == key) {
        Some((_, result)) => result,
        None => return 0,
    };

    let edge_divergences = tile_result.edge_results.iter().filter(|er| er.diverge_idx.is_some()).count();

    let level_divergences = tile_result.level_results.iter().filter(|lr| lr.diverge_idx.is_some()).count();

    edge_divergences + level_divergences
}

/// Load an [`EventsConfig`] from an `events.json` file.
///
/// Searches for `events.json` in the given directory, then in its parent
/// directory. Returns `EventsConfig::default()` if not found.
pub fn load_events_config(dir: &Path) -> Result<EventsConfig, String> {
    // Try dir/events.json first.
    let path = dir.join("events.json");
    if path.exists() {
        let text = std::fs::read_to_string(&path).map_err(|e| format!("read {}: {}", path.display(), e))?;
        return serde_json::from_str(&text).map_err(|e| format!("parse {}: {}", path.display(), e));
    }

    // Try parent/events.json as fallback.
    if let Some(parent) = dir.parent() {
        let path = parent.join("events.json");
        if path.exists() {
            let text =
                std::fs::read_to_string(&path).map_err(|e| format!("read {}: {}", path.display(), e))?;
            return serde_json::from_str(&text).map_err(|e| format!("parse {}: {}", path.display(), e));
        }
    }

    // No events.json found -- use defaults.
    Ok(EventsConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn divergence_count_empty_batch() {
        let batch = BatchResult {
            batch_idx: 0,
            config: EventsConfig::default(),
            tiles: Vec::new(),
            stall_attributions: Vec::new(),
            cross_tile: None,
            pc_anchored: std::collections::HashMap::new(),
        };
        let key = TileKey { col: 0, row: 2, pkt_type: 0 };
        assert_eq!(divergence_count(&batch, &key), 0);
    }

    #[test]
    fn divergence_count_tile_not_found() {
        let batch = BatchResult {
            batch_idx: 0,
            config: EventsConfig::default(),
            tiles: vec![(
                TileKey { col: 1, row: 2, pkt_type: 0 },
                crate::trace::compare::TileResult {
                    hw_t0: 0,
                    emu_t0: 0,
                    edge_results: Vec::new(),
                    level_results: Vec::new(),
                    iteration_results: Vec::new(),
                },
            )],
            stall_attributions: Vec::new(),
            cross_tile: None,
            pc_anchored: std::collections::HashMap::new(),
        };
        let missing_key = TileKey { col: 99, row: 99, pkt_type: 0 };
        assert_eq!(divergence_count(&batch, &missing_key), 0);
    }

    #[test]
    fn load_events_config_missing_dir_returns_default() {
        let dir = Path::new("/nonexistent/path/that/does/not/exist");
        let config = load_events_config(dir).unwrap();
        assert!(config.core_events.is_empty());
        assert!(config.mem_events.is_empty());
        assert!(config.memtile_events.is_empty());
    }

    /// Regression guard: when events.json carries a placement origin > 0
    /// (e.g., HW with start_col=1), the compare path normalizes tile
    /// coords by subtracting origin_col. LoadedComparison must apply the
    /// same shift to its hw_events / emu_events HashMaps -- otherwise
    /// sorted_keys (normalized) miss every entry (raw) and the viewer
    /// reports "no events" for every tile.
    #[test]
    fn loaded_comparison_per_tile_events_nonempty_after_shift() {
        let tmp = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into());
        let root = std::path::PathBuf::from(tmp).join(format!(
            "xdna-emu-data-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let hw_dir = root.join("hw");
        let emu_dir = root.join("emu");
        std::fs::create_dir_all(&hw_dir).unwrap();
        std::fs::create_dir_all(&emu_dir).unwrap();

        // Single trace event at physical (col=1, row=2). With placement
        // origin_col=1, this should normalize to (col=0, row=2) in
        // both batch.tiles and the events HashMaps.
        let events_json = r#"{
            "schema_version": 1,
            "events": [
              {"col": 1, "row": 2, "pkt_type": 0, "slot": 0,
               "name": "PERF_CNT_2", "ts": 100, "soc": 100},
              {"col": 1, "row": 2, "pkt_type": 0, "slot": 1,
               "name": "INSTR_EVENT_0", "ts": 200, "soc": 200}
            ],
            "slot_names": {"core": ["PERF_CNT_2", "INSTR_EVENT_0"],
                           "mem": [], "memtile": []},
            "placement": {"origin_col": 1, "origin_row": 0}
        }"#;
        std::fs::write(hw_dir.join("trace.events.json"), events_json).unwrap();
        std::fs::write(emu_dir.join("trace.events.json"), events_json).unwrap();

        let comp = LoadedComparison::from_trace_dirs(&hw_dir, &emu_dir)
            .expect("LoadedComparison::from_trace_dirs should succeed");
        let keys = comp.tile_keys();
        assert!(!keys.is_empty(), "expected at least one tile key");
        for k in keys {
            assert!(
                !comp.hw_events(k).is_empty(),
                "hw_events({:?}) was empty -- sorted_keys do not match the events HashMap (forgot the placement shift?)",
                k,
            );
            assert!(
                !comp.emu_events(k).is_empty(),
                "emu_events({:?}) was empty -- sorted_keys do not match the events HashMap",
                k,
            );
            // Sanity: the normalized key has col=0 (after subtracting origin_col=1).
            assert_eq!(k.col, 0, "key {:?} should be normalized to col=0", k);
        }
        std::fs::remove_dir_all(&root).ok();
    }
}
