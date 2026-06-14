//! Stochastic-aware trace comparison: EMU vs a HW *distribution*.
//!
//! DDR-boundary trace events (`DMA_S2MM_*_STREAM_STARVATION`, `LOCK_STALL`) are
//! genuinely stochastic on real silicon -- they jitter run-to-run with no
//! exploitable structure (memoryless; CV ~6-25%), because DDR controller
//! arbitration/refresh timing is non-deterministic. Compute-structure events
//! (`PORT_RUNNING`) are deterministic (run-to-run std ~= 0). See
//! `docs/superpowers/findings/2026-06-14-ddr-delivery-stochasticity.md`.
//!
//! Comparing a deterministic EMU run against a *single* HW capture therefore
//! manufactures phantom divergence on the stochastic events. This module
//! compares EMU against the HW *distribution* (N captures of the same binary):
//! per event it derives a tolerance band from the HW run-to-run variance, so
//! the band width *emerges from the data* -- no hardcoded "these events are
//! stochastic" table. An event with HW std 0 collapses to an exact band
//! (a mismatch is a real bug); an event with HW std > 0 gets a `mean +/- k*std`
//! band.
//!
//! The band is HW-derived (the silicon's own variance is ground truth); a future
//! stochastic emulator DDR model earns its keep by reproducing that variance.
//! See `docs/superpowers/specs/2026-06-14-stochastic-aware-trace-comparison.md`.
//!
//! METRIC CAVEAT: this MVP compares raw event-*record* counts per slot. For
//! occurrence/edge events (DMA START/FINISHED tasks) that is exactly the
//! occurrence count. For *level* events (PORT_RUNNING, STREAM_STARVATION) the
//! record count is the held-level frame count, which can differ from the
//! semantic interval count (HW and EMU may encode the same intervals with a
//! different number of frames). The EMU-vs-HW comparison is internally
//! consistent (both raw), but for level events a future revision should band on
//! the reconstructed interval count (as `compare`'s level analysis does) to
//! separate encoding differences from behavioral ones. The band comparison
//! still correctly ranks EMU against the HW distribution either way.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use super::compare::{load_events_json, names_for_pkt, remap_tile_columns, EventsConfig, TileKey};

/// Classification of an event's run-to-run behavior, derived from HW variance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    /// HW count is identical across all runs (std == 0): compute structure.
    Deterministic,
    /// HW count varies across runs (std > 0): DDR-boundary stochasticity.
    Stochastic,
}

/// One event's HW count distribution and the EMU point's standing against it.
#[derive(Debug, Clone)]
pub struct EventBand {
    pub key: TileKey,
    pub slot: u8,
    pub name: String,
    /// HW count per run (length == number of HW captures).
    pub hw_counts: Vec<usize>,
    pub mean: f64,
    pub std: f64,
    pub min: usize,
    pub max: usize,
    pub emu_count: usize,
    pub band_lo: f64,
    pub band_hi: f64,
    pub regime: Regime,
    pub in_band: bool,
}

impl EventBand {
    /// Signed distance of the EMU count from the band, in standard deviations.
    /// Zero when in-band or when the band is exact and matched. For a
    /// deterministic event the "sigma" is undefined (std 0); we report the raw
    /// count delta instead via `count_delta`.
    pub fn sigma_out(&self) -> f64 {
        if self.in_band || self.std == 0.0 {
            return 0.0;
        }
        let d = if (self.emu_count as f64) < self.band_lo {
            self.band_lo - self.emu_count as f64
        } else {
            self.emu_count as f64 - self.band_hi
        };
        d / self.std
    }

    /// Raw EMU-minus-mean count delta (useful for deterministic events).
    pub fn count_delta(&self) -> f64 {
        self.emu_count as f64 - self.mean
    }
}

/// Full stochastic comparison result.
#[derive(Debug, Clone)]
pub struct StochasticReport {
    pub bands: Vec<EventBand>,
    pub n_runs: usize,
    pub sigma_k: f64,
}

impl StochasticReport {
    pub fn out_of_band(&self) -> usize {
        self.bands.iter().filter(|b| !b.in_band).count()
    }
    pub fn is_clean(&self) -> bool {
        self.out_of_band() == 0
    }
}

/// Population mean and standard deviation of a count sample.
fn mean_std(xs: &[usize]) -> (f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let n = xs.len() as f64;
    let mean = xs.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = xs.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// Count events per `(TileKey, slot)` in one loaded side.
///
/// Core tiles (pkt_type 0) are excluded: they are traced EVENT_PC (mode 1),
/// where the per-event scalar is a program counter and the record *count* is a
/// PC-repeat-expansion artifact, not a delivery-timing signal. The stochastic
/// band comparison is about mode-0 shim/memtile DMA-delivery events. (Same
/// rationale as the mode-aware cycle-drift exclusion in `compare`.)
fn count_by_slot(tiles: &super::compare::TileEvents) -> BTreeMap<(TileKey, u8), usize> {
    let mut out: BTreeMap<(TileKey, u8), usize> = BTreeMap::new();
    for (key, evs) in tiles {
        if key.pkt_type == 0 {
            continue;
        }
        for e in evs {
            *out.entry((*key, e.slot)).or_insert(0) += 1;
        }
    }
    out
}

/// Build per-event bands from a sample of HW count maps plus the EMU count map.
///
/// Pure over the already-counted inputs (no IO), so the band math is unit
/// testable without fixtures. `k` is the band half-width in standard
/// deviations. Deterministic events (std 0) collapse to an exact band.
pub fn build_bands(
    hw_runs: &[BTreeMap<(TileKey, u8), usize>],
    emu: &BTreeMap<(TileKey, u8), usize>,
    config: &EventsConfig,
    k: f64,
) -> Vec<EventBand> {
    // Every (key, slot) observed on any HW run, plus any EMU-only events.
    let mut keys: BTreeSet<(TileKey, u8)> = BTreeSet::new();
    for run in hw_runs {
        keys.extend(run.keys().copied());
    }
    keys.extend(emu.keys().copied());

    let mut bands = Vec::new();
    for (key, slot) in keys {
        let hw_counts: Vec<usize> =
            hw_runs.iter().map(|r| r.get(&(key, slot)).copied().unwrap_or(0)).collect();
        let emu_count = emu.get(&(key, slot)).copied().unwrap_or(0);
        let (mean, std) = mean_std(&hw_counts);
        let min = hw_counts.iter().copied().min().unwrap_or(0);
        let max = hw_counts.iter().copied().max().unwrap_or(0);
        let band_lo = mean - k * std;
        let band_hi = mean + k * std;
        let regime = if std == 0.0 {
            Regime::Deterministic
        } else {
            Regime::Stochastic
        };
        // Inclusive band; a tiny epsilon guards float edges of mean+/-k*std.
        let eps = 1e-9;
        let in_band = (emu_count as f64) >= band_lo - eps && (emu_count as f64) <= band_hi + eps;
        let names = names_for_pkt(config, key.pkt_type);
        let name = super::compare::slot_name(slot, names);
        bands.push(EventBand {
            key,
            slot,
            name,
            hw_counts,
            mean,
            std,
            min,
            max,
            emu_count,
            band_lo,
            band_hi,
            regime,
            in_band,
        });
    }
    bands
}

/// Compare a deterministic EMU capture against a HW distribution.
///
/// `hw_paths` are >= 1 HW events.json of the *same* binary; `emu_path` is the
/// single EMU events.json. `k` is the band half-width in standard deviations.
pub fn compare_stochastic(
    hw_paths: &[PathBuf],
    emu_path: &Path,
    config: &EventsConfig,
    k: f64,
    remap_columns: bool,
) -> Result<StochasticReport, String> {
    if hw_paths.is_empty() {
        return Err("compare_stochastic: need at least one HW capture".into());
    }
    // Each events.json carries its own slot_names; prefer the JSON's config for
    // labeling (the passed `config` is a fallback / caller override). The first
    // HW file with non-empty names wins, else EMU, else the fallback.
    let has_names = |c: &EventsConfig| {
        !c.core_events.is_empty()
            || !c.mem_events.is_empty()
            || !c.memtile_events.is_empty()
            || !c.shim_events.is_empty()
    };
    let mut resolved_config: Option<EventsConfig> = None;
    let load =
        |p: &Path, cfg_out: &mut Option<EventsConfig>| -> Result<BTreeMap<(TileKey, u8), usize>, String> {
            let (mut tiles, cfg, _placement, _modes) = load_events_json(p)?;
            if cfg_out.is_none() && has_names(&cfg) {
                *cfg_out = Some(cfg);
            }
            if remap_columns {
                tiles = remap_tile_columns(&tiles);
            }
            Ok(count_by_slot(&tiles))
        };
    let hw_runs: Vec<_> = hw_paths
        .iter()
        .map(|p| load(p, &mut resolved_config))
        .collect::<Result<_, _>>()?;
    let emu = load(emu_path, &mut resolved_config)?;
    let eff_config = resolved_config.as_ref().filter(|c| has_names(c)).unwrap_or(config);
    let bands = build_bands(&hw_runs, &emu, eff_config, k);
    Ok(StochasticReport { bands, n_runs: hw_paths.len(), sigma_k: k })
}

/// Render a human-readable report plus a `STOCHASTIC_VERDICT` token.
pub fn format_report(report: &StochasticReport) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "============================================================================");
    let _ = writeln!(out, "Stochastic-aware trace comparison (EMU vs HW distribution)");
    let _ =
        writeln!(out, "  HW runs: {}   band: mean +/- {:.1}*std (HW-derived)", report.n_runs, report.sigma_k);
    let _ = writeln!(out, "============================================================================");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "{:<34} {:>4} {:>7} {:>6} {:>9} {:>6} {:>15} {}",
        "event", "reg", "HWmean", "HWstd", "band", "EMU", "HWrange", "verdict"
    );
    // Stochastic events first (the interesting ones), then deterministic.
    let mut bands: Vec<&EventBand> = report.bands.iter().collect();
    bands.sort_by_key(|b| (b.regime == Regime::Deterministic, b.key, b.slot));
    for b in bands {
        let reg = match b.regime {
            Regime::Deterministic => "DET",
            Regime::Stochastic => "STO",
        };
        let verdict = if b.in_band {
            "in-band".to_string()
        } else if b.regime == Regime::Deterministic {
            format!("OUT (d={:+.0})", b.count_delta())
        } else {
            format!(
                "OUT ({:+.1} sigma)",
                if (b.emu_count as f64) < b.band_lo {
                    -b.sigma_out()
                } else {
                    b.sigma_out()
                }
            )
        };
        let _ = writeln!(
            out,
            "{:<34} {:>4} {:>7.1} {:>6.1} [{:>4.0},{:>4.0}] {:>6} [{:>3},{:>3}]    {}",
            b.name,
            reg,
            b.mean,
            b.std,
            b.band_lo.max(0.0),
            b.band_hi,
            b.emu_count,
            b.min,
            b.max,
            verdict
        );
    }
    let _ = writeln!(out);
    let oob = report.out_of_band();
    if oob == 0 {
        let _ = writeln!(
            out,
            "STOCHASTIC_VERDICT: CLEAN ({} events, all in-band over {} HW runs)",
            report.bands.len(),
            report.n_runs
        );
    } else {
        let det_oob = report
            .bands
            .iter()
            .filter(|b| !b.in_band && b.regime == Regime::Deterministic)
            .count();
        let sto_oob = oob - det_oob;
        let _ = writeln!(
            out,
            "STOCHASTIC_VERDICT: DIVERGE ({} of {} events out of band: {} deterministic, {} stochastic)",
            oob,
            report.bands.len(),
            det_oob,
            sto_oob
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(col: u8, row: u8, pt: u8, slot: u8) -> (TileKey, u8) {
        (TileKey { col, row, pkt_type: pt }, slot)
    }

    #[test]
    fn deterministic_event_collapses_to_exact_band() {
        // HW count identical across runs -> std 0 -> exact band.
        let hw: Vec<_> = (0..4)
            .map(|_| {
                let mut m = BTreeMap::new();
                m.insert(k(0, 1, 3, 0), 11usize); // PORT_RUNNING_0 always 11
                m
            })
            .collect();
        // EMU matches exactly -> in band.
        let mut emu = BTreeMap::new();
        emu.insert(k(0, 1, 3, 0), 11usize);
        let cfg = EventsConfig { memtile_events: vec!["PORT_RUNNING_0".into()], ..Default::default() };
        let bands = build_bands(&hw, &emu, &cfg, 2.0);
        assert_eq!(bands.len(), 1);
        assert_eq!(bands[0].regime, Regime::Deterministic);
        assert_eq!(bands[0].std, 0.0);
        assert!(bands[0].in_band, "exact match is in band");

        // EMU off-by-one -> out of band (a real bug for a deterministic event).
        let mut emu2 = BTreeMap::new();
        emu2.insert(k(0, 1, 3, 0), 12usize);
        let bands2 = build_bands(&hw, &emu2, &cfg, 2.0);
        assert!(!bands2[0].in_band, "deterministic off-by-one diverges");
        assert_eq!(bands2[0].count_delta(), 1.0);
    }

    #[test]
    fn stochastic_event_uses_mean_plus_k_std_band() {
        // HW counts 24,28,29,29,31,33 -> mean 29.0, std ~2.77. k=2 band ~[23.5, 34.5].
        let counts = [24usize, 28, 29, 29, 31, 33];
        let hw: Vec<_> = counts
            .iter()
            .map(|&c| {
                let mut m = BTreeMap::new();
                m.insert(k(0, 0, 2, 6), c); // shim slot 6 STREAM_STARVATION
                m
            })
            .collect();
        let cfg = EventsConfig {
            shim_events: vec![
                "a".into(),
                "b".into(),
                "c".into(),
                "d".into(),
                "e".into(),
                "f".into(),
                "DMA_S2MM_0_STREAM_STARVATION".into(),
                "h".into(),
            ],
            ..Default::default()
        };
        // EMU=28 is inside the band -> CLEAN.
        let mut emu_in = BTreeMap::new();
        emu_in.insert(k(0, 0, 2, 6), 28usize);
        let b_in = build_bands(&hw, &emu_in, &cfg, 2.0);
        assert_eq!(b_in[0].regime, Regime::Stochastic);
        assert_eq!(b_in[0].name, "DMA_S2MM_0_STREAM_STARVATION");
        assert!(b_in[0].in_band, "28 within mean+/-2std of ~[23.5,34.5]");

        // EMU=20 (the off-baseline) is BELOW band -> the model gap is real, not
        // a sampling artifact.
        let mut emu_low = BTreeMap::new();
        emu_low.insert(k(0, 0, 2, 6), 20usize);
        let b_low = build_bands(&hw, &emu_low, &cfg, 2.0);
        assert!(!b_low[0].in_band, "20 is below the HW band -> genuine under-emission");
        assert!(b_low[0].sigma_out() > 0.0);
    }

    #[test]
    fn emu_only_event_is_out_of_band() {
        // Event absent on HW (all-zero distribution) but present on EMU.
        let hw: Vec<_> = (0..3).map(|_| BTreeMap::<(TileKey, u8), usize>::new()).collect();
        let mut emu = BTreeMap::new();
        emu.insert(k(0, 0, 2, 0), 5usize);
        let cfg = EventsConfig::default();
        let bands = build_bands(&hw, &emu, &cfg, 2.0);
        assert_eq!(bands.len(), 1);
        // HW mean 0, std 0 -> deterministic exact band [0,0]; EMU 5 -> out.
        assert!(!bands[0].in_band);
        assert_eq!(bands[0].regime, Regime::Deterministic);
    }

    #[test]
    fn verdict_token_clean_and_diverge() {
        let counts = [28usize, 29, 30];
        let hw: Vec<_> = counts
            .iter()
            .map(|&c| {
                let mut m = BTreeMap::new();
                m.insert(k(0, 0, 2, 6), c);
                m
            })
            .collect();
        let cfg = EventsConfig::default();
        let mut emu = BTreeMap::new();
        emu.insert(k(0, 0, 2, 6), 29usize);
        let rep = StochasticReport { bands: build_bands(&hw, &emu, &cfg, 2.0), n_runs: 3, sigma_k: 2.0 };
        assert!(rep.is_clean());
        assert!(format_report(&rep).contains("STOCHASTIC_VERDICT: CLEAN"));

        let mut emu_bad = BTreeMap::new();
        emu_bad.insert(k(0, 0, 2, 6), 50usize);
        let rep_bad =
            StochasticReport { bands: build_bands(&hw, &emu_bad, &cfg, 2.0), n_runs: 3, sigma_k: 2.0 };
        assert!(!rep_bad.is_clean());
        assert!(format_report(&rep_bad).contains("STOCHASTIC_VERDICT: DIVERGE"));
    }
}
