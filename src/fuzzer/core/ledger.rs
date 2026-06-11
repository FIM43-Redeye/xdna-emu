//! Persistent coverage ledger for differential fuzzers.
//!
//! Domain-agnostic: holds only coverage key strings. The domain supplies its
//! key universe (e.g. `table::universe_keys()` for the vector fuzzer) to
//! `uncovered`, `complete`, and `report`. A passing chain credits all the stage
//! keys it executed; keys with confirmed emulator-vs-silicon divergence are
//! excluded from credit (and from completion) but never block the campaign;
//! keys whose kernels never compile are marked unreachable with a reason.
//! Timing-only divergence is tracked separately and does not withhold
//! functional credit. The campaign is complete when every key that is neither
//! divergent nor unreachable has at least `target` hits.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Persistent coverage state, serialized as pretty JSON between fuzz runs.
#[derive(Default, Serialize, Deserialize)]
pub struct Ledger {
    /// Verified-against-silicon hit count per coverage key.
    hits: HashMap<String, u32>,
    /// Keys with confirmed functional emulator-vs-silicon divergence.
    divergent: HashSet<String>,
    /// Keys with timing-only divergence (functional credit still allowed).
    timing_fail: HashSet<String>,
    /// Keys whose kernels never compile, with the reason.
    unreachable: HashMap<String, String>,
    /// Keys cleared from `divergent` by a clean re-verify, with evidence. Kept
    /// for audit; never withholds credit. Defaulted so pre-field ledgers load.
    #[serde(default)]
    resolved: HashMap<String, String>,
}

impl Ledger {
    /// Credit one verified chain: +1 to each executed key, skipping keys
    /// already marked divergent or unreachable.
    pub fn credit_keys(&mut self, keys: &[String]) {
        for key in keys {
            if self.divergent.contains(key) || self.unreachable.contains_key(key) {
                continue;
            }
            *self.hits.entry(key.clone()).or_insert(0) += 1;
        }
    }

    /// Mark a key as functionally divergent (excluded from credit/completion).
    /// A divergent key is never also `resolved`.
    pub fn mark_divergent(&mut self, key: &str) {
        self.resolved.remove(key);
        self.divergent.insert(key.to_string());
    }

    /// Drain the divergent set for re-verification, returning the cleared keys
    /// (sorted). They rejoin the uncovered pool until a run re-credits them (and
    /// `mark_resolved`) or re-flags them (`mark_divergent`).
    pub fn take_divergent(&mut self) -> Vec<String> {
        let mut keys: Vec<String> = self.divergent.drain().collect();
        keys.sort();
        keys
    }

    /// True if the key is currently flagged divergent.
    pub fn is_divergent(&self, key: &str) -> bool {
        self.divergent.contains(key)
    }

    /// Silicon-verified hit count for a key (0 if never credited).
    pub fn hit_count(&self, key: &str) -> u32 {
        self.hits.get(key).copied().unwrap_or(0)
    }

    /// Record that a previously-divergent key was re-verified clean against
    /// silicon, with an evidence note. Audit-only; does not affect credit.
    pub fn mark_resolved(&mut self, key: &str, evidence: &str) {
        self.resolved.insert(key.to_string(), evidence.to_string());
    }

    /// Flag timing-only divergence on a key; functional credit continues.
    pub fn mark_timing(&mut self, key: &str) {
        self.timing_fail.insert(key.to_string());
    }

    /// Mark a key as unreachable (kernel never compiles), with the reason.
    pub fn mark_unreachable(&mut self, key: &str, reason: &str) {
        self.unreachable.insert(key.to_string(), reason.to_string());
    }

    /// Universe keys still needing credit: not divergent, not unreachable,
    /// fewer than `target` hits. Sorted. `universe` is supplied by the domain.
    pub fn uncovered(&self, universe: &[String], target: u32) -> Vec<String> {
        universe
            .iter()
            .filter(|k| !self.divergent.contains(*k) && !self.unreachable.contains_key(*k))
            .filter(|k| self.hits.get(*k).copied().unwrap_or(0) < target)
            .cloned()
            .collect()
    }

    /// True when every key not divergent/unreachable has >= `target` hits.
    pub fn complete(&self, universe: &[String], target: u32) -> bool {
        self.uncovered(universe, target).is_empty()
    }

    /// Persist as pretty JSON, atomically (temp file + rename).
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| format!("serialize ledger: {e}"))?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, json).map_err(|e| format!("write {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, path)
            .map_err(|e| format!("rename {} -> {}: {e}", tmp.display(), path.display()))
    }

    /// Load from JSON; a missing file yields a fresh default ledger.
    pub fn load(path: &Path) -> Result<Self, String> {
        let data = match std::fs::read_to_string(path) {
            Ok(d) => d,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => return Err(format!("read {}: {e}", path.display())),
        };
        serde_json::from_str(&data).map_err(|e| format!("parse {}: {e}", path.display()))
    }

    /// Human-readable campaign status over `universe`.
    pub fn report(&self, universe: &[String], target: u32) -> String {
        let uncovered = self.uncovered(universe, target);
        let covered = universe
            .iter()
            .filter(|k| !self.divergent.contains(*k) && !self.unreachable.contains_key(*k))
            .filter(|k| self.hits.get(*k).copied().unwrap_or(0) >= target)
            .count();

        let mut out = String::new();
        let _ = writeln!(out, "vector fuzzer coverage (target {target} hits/key)");
        let _ = writeln!(out, "  universe:    {}", universe.len());
        let _ = writeln!(out, "  covered:     {covered}");
        let _ = writeln!(out, "  uncovered:   {}", uncovered.len());
        let _ = writeln!(out, "  divergent:   {}", self.divergent.len());
        let _ = writeln!(out, "  timing-only: {}", self.timing_fail.len());
        let _ = writeln!(out, "  unreachable: {}", self.unreachable.len());
        if !self.resolved.is_empty() {
            let _ = writeln!(out, "  resolved:    {} (re-verified clean)", self.resolved.len());
        }

        if !uncovered.is_empty() {
            let _ = writeln!(out, "uncovered (first 40):");
            for key in uncovered.iter().take(40) {
                let _ = writeln!(out, "  {key} ({} hits)", self.hits.get(key).copied().unwrap_or(0));
            }
            if uncovered.len() > 40 {
                let _ = writeln!(out, "  ... and {} more", uncovered.len() - 40);
            }
        }
        if !self.divergent.is_empty() {
            let mut div: Vec<&String> = self.divergent.iter().collect();
            div.sort();
            let _ = writeln!(out, "divergent (excluded from credit):");
            for key in div {
                let _ = writeln!(out, "  {key}");
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A synthetic coverage universe for ledger unit tests -- the ledger is
    /// table-agnostic, so tests supply their own key space.
    fn test_universe() -> Vec<String> {
        (0..20).map(|i| format!("op{i}/I32x16/m0")).collect()
    }

    fn tempdir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "xdna-emu-ledger-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn fresh_ledger_uncovered_is_full_universe() {
        let ledger = Ledger::default();
        let u = test_universe();
        assert_eq!(ledger.uncovered(&u, 10), u);
    }

    #[test]
    fn credit_increments_and_target_hits_remove_from_uncovered() {
        let mut ledger = Ledger::default();
        let u = test_universe();
        let key = u[0].clone();
        let keys = vec![key.clone()];
        ledger.credit_keys(&keys);
        assert!(ledger.uncovered(&u, 10).contains(&key), "1 hit < target");
        assert!(!ledger.uncovered(&u, 1).contains(&key), "1 hit meets target 1");
        for _ in 0..9 {
            ledger.credit_keys(&keys);
        }
        assert!(!ledger.uncovered(&u, 10).contains(&key), "10 hits meet target 10");
        assert_eq!(ledger.uncovered(&u, 10).len(), u.len() - 1);
    }

    #[test]
    fn credit_on_divergent_key_does_nothing() {
        let mut ledger = Ledger::default();
        let u = test_universe();
        let key = u[0].clone();
        ledger.mark_divergent(&key);
        for _ in 0..10 {
            ledger.credit_keys(std::slice::from_ref(&key));
        }
        assert_eq!(ledger.hits.get(&key), None);
        assert!(!ledger.uncovered(&u, 10).contains(&key), "divergent never uncovered");
    }

    #[test]
    fn unreachable_excluded_from_uncovered_and_credit() {
        let mut ledger = Ledger::default();
        let u = test_universe();
        let key = u[0].clone();
        ledger.mark_unreachable(&key, "kernel never compiles");
        assert!(!ledger.uncovered(&u, 10).contains(&key));
        ledger.credit_keys(std::slice::from_ref(&key));
        assert_eq!(ledger.hits.get(&key), None);
    }

    #[test]
    fn timing_flag_does_not_withhold_credit() {
        let mut ledger = Ledger::default();
        let u = test_universe();
        let key = u[0].clone();
        ledger.mark_timing(&key);
        for _ in 0..10 {
            ledger.credit_keys(std::slice::from_ref(&key));
        }
        assert!(!ledger.uncovered(&u, 10).contains(&key));
    }

    #[test]
    fn save_load_round_trip() {
        let dir = tempdir();
        let path = dir.join("ledger.json");
        let mut ledger = Ledger::default();
        let u = test_universe();
        ledger.credit_keys(&u[..5]);
        ledger.mark_divergent(&u[6]);
        ledger.mark_timing(&u[7]);
        ledger.mark_unreachable(&u[8], "no compile");

        ledger.save(&path).unwrap();
        let loaded = Ledger::load(&path).unwrap();
        assert_eq!(loaded.hits, ledger.hits);
        assert_eq!(loaded.divergent, ledger.divergent);
        assert_eq!(loaded.timing_fail, ledger.timing_fail);
        assert_eq!(loaded.unreachable, ledger.unreachable);
        assert_eq!(loaded.uncovered(&u, 10), ledger.uncovered(&u, 10));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn load_missing_path_yields_default() {
        let path = std::env::temp_dir().join("xdna-emu-ledger-test-does-not-exist.json");
        let ledger = Ledger::load(&path).unwrap();
        let u = test_universe();
        assert_eq!(ledger.uncovered(&u, 10).len(), u.len());
    }

    #[test]
    fn take_divergent_clears_flags_and_returns_keys_to_uncovered() {
        let mut ledger = Ledger::default();
        let u = test_universe();
        let key = u[0].clone();
        ledger.mark_divergent(&key);
        assert!(!ledger.uncovered(&u, 10).contains(&key), "flagged: not uncovered");

        let taken = ledger.take_divergent();
        assert_eq!(taken, vec![key.clone()]);
        assert!(!ledger.is_divergent(&key), "flag cleared");
        assert!(ledger.uncovered(&u, 10).contains(&key), "back in uncovered pool");

        // Re-earning it now credits normally.
        let keys = vec![key.clone()];
        for _ in 0..10 {
            ledger.credit_keys(&keys);
        }
        assert!(!ledger.uncovered(&u, 10).contains(&key), "re-earned to target");
    }

    #[test]
    fn mark_resolved_and_mark_divergent_are_mutually_exclusive() {
        let mut ledger = Ledger::default();
        let u = test_universe();
        let key = u[0].clone();
        ledger.mark_resolved(&key, "re-verified clean, seed 5000");
        ledger.mark_divergent(&key); // re-flagging drops the resolved record
        assert!(ledger.is_divergent(&key));
        let report = ledger.report(&u, 10);
        assert!(!report.contains("resolved:"), "no resolved entries: {report}");
    }

    #[test]
    fn resolved_field_round_trips_and_pre_field_ledger_loads() {
        let dir = tempdir();
        let path = dir.join("ledger.json");
        // A ledger.json written before the `resolved` field existed.
        std::fs::write(&path, r#"{"hits":{},"divergent":["k"],"timing_fail":[],"unreachable":{}}"#).unwrap();
        let mut ledger = Ledger::load(&path).expect("pre-field ledger loads via serde default");
        assert!(ledger.is_divergent("k"));

        ledger.take_divergent();
        ledger.mark_resolved("k", "re-verified clean");
        ledger.save(&path).unwrap();
        let loaded = Ledger::load(&path).unwrap();
        assert_eq!(loaded.resolved, ledger.resolved);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn complete_flips_when_all_credited() {
        let mut ledger = Ledger::default();
        let u = test_universe();
        assert!(!ledger.complete(&u, 1));
        ledger.credit_keys(&u[..u.len() - 1]);
        assert!(!ledger.complete(&u, 1), "one key still missing");
        ledger.credit_keys(&u);
        assert!(ledger.complete(&u, 1));
        assert!(!ledger.complete(&u, 2), "needs two hits each for target 2");
    }

    #[test]
    fn divergent_unreachable_do_not_block_completion() {
        let mut ledger = Ledger::default();
        let u = test_universe();
        ledger.mark_divergent(&u[0]);
        ledger.mark_unreachable(&u[1], "no compile");
        ledger.credit_keys(&u[2..]);
        assert!(ledger.complete(&u, 1));
    }

    #[test]
    fn report_contains_counts_and_divergent_list() {
        let mut ledger = Ledger::default();
        let u = test_universe();
        ledger.mark_divergent(&u[0]);
        ledger.credit_keys(&u[1..]);
        let report = ledger.report(&u, 1);
        assert!(report.contains("divergent"), "report: {report}");
        assert!(report.contains("divergent:   1"), "report: {report}");
        let covered = format!("covered:     {}", u.len() - 1);
        assert!(report.contains(&covered), "report: {report}");
        assert!(report.contains("uncovered:   0"), "report: {report}");
        assert!(report.contains(&u[0]), "divergent key listed: {report}");
    }
}
