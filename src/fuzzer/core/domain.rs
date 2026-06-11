//! The `Domain` contract: everything a fuzzer subsystem must provide for the
//! shared engine to run a coverage-driven differential campaign over it. The
//! engine owns the campaign loop, ledger, compile dispatch, banking/replay, and
//! reporting; a `Domain` owns the case AST, generation, lowering, execution,
//! comparison, and its own bank/replay persistence (the serialized form is
//! domain-specific — vector banks a `ChainRecord`, DMA would bank a BD program).

use std::path::{Path, PathBuf};

/// Which executor produced an observation. Differential = any two backends; the
/// vector domain compares `Interpreter` vs `Hardware`. `Aiesim` is enumerated
/// for forward-compatibility but not wired for vector in Step 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Interpreter,
    Aiesim,
    Hardware,
}

/// Campaign knobs — domain-agnostic. Lifted from the vector `VecFuzzOptions`.
pub struct CampaignOptions {
    pub iterations: usize,
    /// Base seed (None = wall clock); case i runs seed base+i.
    pub seed: Option<u64>,
    pub jobs: usize,
    pub hw: bool,
    pub max_cycles: u64,
    /// Hits per key needed for completion (ledger target).
    pub target_hits: u32,
    pub verbose: bool,
    /// Print the coverage report and exit.
    pub report_only: bool,
    /// Replay banked divergences from this directory.
    pub replay: Option<PathBuf>,
    /// Clear divergent flags and re-earn them against silicon this run.
    pub reverify: bool,
}

/// Result of loading one banked case for replay.
pub enum Banked<C, O> {
    /// A replayable case: reconstructed case + the banked reference observation
    /// (e.g. the silicon output) + the banked coverage keys for localization.
    Replayable {
        case: C,
        reference: O,
        keys: Vec<String>,
    },
    /// The bank cannot be replayed (e.g. a legacy bank under a changed table);
    /// the engine reports the reason and skips it.
    Skip(String),
}

/// A fuzzer subsystem the engine can drive. `Case` is the domain's AST; `Obs`
/// is whatever an execution yields (output bytes, moved buffers, cycle counts —
/// opaque to the engine, which only ever round-trips it through `compare`/`bank`).
pub trait Domain {
    type Case;
    type Obs;

    /// Stable short name: ledger lives at `build/fuzz-{name}`, banks at
    /// `phoenix-survival/{name}`. For vector this is `"vector"`.
    fn name(&self) -> &str;

    /// The full coverage-key universe (sorted), supplied to the ledger.
    fn universe(&self) -> Vec<String>;

    /// Generate the deterministic case for `(seed, target_key)`.
    fn generate(&self, seed: u64, target: &str) -> Self::Case;

    /// Per-stage coverage keys a passing case credits, in output order.
    fn coverage_keys(&self, case: &Self::Case) -> Vec<String>;

    /// The case's target key (the one generation aimed at).
    fn target_key(&self, case: &Self::Case) -> String;

    /// Lower the case to kernel source text written as `fuzz_kernel.cc`.
    fn lower(&self, case: &Self::Case) -> String;

    /// Buffer size in dtype words for the compile (`buf_in`/`scratch`/`out`).
    fn buffer_words(&self, case: &Self::Case) -> usize;

    /// Element dtype string passed to `compile_kernel_case` (vector: `"i32"`).
    fn dtype(&self) -> &str;

    /// Execute the compiled case on `backend`, returning a domain observation.
    fn observe(
        &self,
        backend: Backend,
        xclbin: &Path,
        insts: &Path,
        case: &Self::Case,
        max_cycles: u64,
    ) -> Result<Self::Obs, String>;

    /// Non-fatal warnings about an observation (e.g. "no vector ops executed —
    /// chain folded"). The engine prints them; default none.
    fn warnings(&self, _obs: &Self::Obs) -> Vec<String> {
        Vec::new()
    }

    /// Differential compare: `None` = match (credit the case); `Some(key)` =
    /// the first divergent coverage key (flag it, bank the case). Localization
    /// lives here because only the domain knows the observation's structure.
    /// Takes the coverage `keys` (not the case) so replay can compare against
    /// banked keys -- keeping per-slice type tolerance and localization
    /// table-independent. Campaign passes the live `coverage_keys(case)`; replay
    /// passes the banked keys.
    fn compare(&self, emu: &Self::Obs, reference: &Self::Obs, keys: &[String]) -> Option<String>;

    /// Bank a divergent/crashed case under `phoenix-survival/{name}/seed_N/` for
    /// post-mortem and replay. `reference` is None on an emulator crash (no HW
    /// observation). `case_dir` is the live build dir to copy artifacts from.
    fn bank(
        &self,
        case_dir: &Path,
        case: &Self::Case,
        reference: Option<&Self::Obs>,
        emu_obs: Option<&Self::Obs>,
    ) -> Result<PathBuf, String>;

    /// Load one banked seed dir for replay: reconstruct the case + reference
    /// observation, or report why it cannot be replayed.
    fn load_banked(&self, seed_dir: &Path) -> Result<Banked<Self::Case, Self::Obs>, String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockDomain;
    impl Domain for MockDomain {
        type Case = u64;
        type Obs = Vec<u8>;
        fn name(&self) -> &str {
            "mock"
        }
        fn universe(&self) -> Vec<String> {
            vec!["a/I32x16/m0".into()]
        }
        fn generate(&self, seed: u64, _t: &str) -> u64 {
            seed
        }
        fn coverage_keys(&self, _c: &u64) -> Vec<String> {
            vec!["a/I32x16/m0".into()]
        }
        fn target_key(&self, _c: &u64) -> String {
            "a/I32x16/m0".into()
        }
        fn lower(&self, _c: &u64) -> String {
            String::new()
        }
        fn buffer_words(&self, _c: &u64) -> usize {
            16
        }
        fn dtype(&self) -> &str {
            "i32"
        }
        fn observe(&self, _b: Backend, _x: &Path, _i: &Path, _c: &u64, _m: u64) -> Result<Vec<u8>, String> {
            Ok(vec![])
        }
        fn compare(&self, a: &Vec<u8>, b: &Vec<u8>, _keys: &[String]) -> Option<String> {
            if a == b {
                None
            } else {
                Some("a/I32x16/m0".into())
            }
        }
        fn bank(
            &self,
            _d: &Path,
            _c: &u64,
            _r: Option<&Vec<u8>>,
            _e: Option<&Vec<u8>>,
        ) -> Result<PathBuf, String> {
            Ok(PathBuf::new())
        }
        fn load_banked(&self, _d: &Path) -> Result<Banked<u64, Vec<u8>>, String> {
            Ok(Banked::Skip("mock".into()))
        }
    }

    fn drive<D: Domain>(d: &D) -> Vec<String> {
        d.universe()
    }

    #[test]
    fn mock_domain_satisfies_the_trait_behind_a_generic_bound() {
        let m = MockDomain;
        assert_eq!(drive(&m), vec!["a/I32x16/m0".to_string()]);
        assert_eq!(m.compare(&vec![1], &vec![2], &[]), Some("a/I32x16/m0".into()));
        assert_eq!(m.compare(&vec![1], &vec![1], &[]), None);
    }
}
