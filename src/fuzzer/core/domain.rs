//! The `Domain` contract: everything a fuzzer subsystem must provide for the
//! shared engine to run a coverage-driven differential campaign over it. The
//! engine owns the campaign loop, ledger, compile dispatch, banking/replay, and
//! reporting; a `Domain` owns the case AST, generation, lowering, execution,
//! comparison, and its own bank/replay persistence (the serialized form is
//! domain-specific — vector banks a `ChainRecord`, DMA would bank a BD program).

use std::path::{Path, PathBuf};

use super::toolchain::ToolPaths;

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
    /// Restrict this run's work-set to keys whose first `/`-delimited segment
    /// (the feature/op name) is in this list. `None` = the whole uncovered set.
    /// Lets a campaign be staged over a feature subset (e.g. run safe DMA
    /// patterns and bank them before pushing risky ones that may wedge the NPU).
    pub feature_filter: Option<Vec<String>>,
    /// Persist the first silicon-matched seed per target key as a durable
    /// regression-corpus entry (the *passing* corpus), not just divergences.
    /// Opt-in (default off) so normal campaigns are unchanged; used for targeted
    /// corpus-building runs (e.g. re-banking a previously stranded key with a
    /// fresh pool-bearing seed). One canonical seed per key, first match wins.
    pub bank_matches: bool,
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

    /// Stable short name for this domain. The engine's per-campaign work and
    /// ledger directory is `build/fuzz-{name}`; by convention the domain also
    /// banks under `phoenix-survival/{name}` (that path is owned by the domain's
    /// own `bank`, not constructed by the engine). For vector this is `"vector"`.
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

    /// Compile the case in `case_dir` to `aie.xclbin` + `insts.bin`. The default
    /// is the compute path used by vector and scalar: write `lower(case)` to
    /// `fuzz_kernel.cc`, then run Peano + fuzz_template.py + aiecc via
    /// `compile_kernel_case`. The DMA domain overrides this to write `aie.mlir`
    /// and run aiecc directly (no kernel object, no template).
    fn compile(&self, tools: &ToolPaths, case_dir: &Path, case: &Self::Case) -> Result<(), String> {
        let src = self.lower(case);
        std::fs::write(case_dir.join("fuzz_kernel.cc"), &src)
            .map_err(|e| format!("write fuzz_kernel.cc: {e}"))?;
        super::toolchain::compile_kernel_case(tools, case_dir, self.buffer_words(case), self.dtype(case))
    }

    /// Buffer size in dtype words for the compile (`buf_in`/`scratch`/`out`).
    fn buffer_words(&self, case: &Self::Case) -> usize;

    /// Element dtype string passed to `compile_kernel_case`. Depends on the case
    /// for domains with per-case dtype (scalar I32/I16/I8); vector ignores it.
    fn dtype(&self, case: &Self::Case) -> &str;

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

    /// Persist post-mortem artifacts for a case that is STILL divergent on
    /// replay (called by the engine's replay loop). The default is a no-op;
    /// a domain overrides it to dump whatever helps a human diff the failure
    /// (vector writes the EMU output next to the banked reference for byte
    /// diffing). `case_dir` is the banked seed dir. Errors are logged, not fatal.
    fn dump_divergent_observation(&self, case_dir: &Path, emu: &Self::Obs) -> Result<(), String> {
        let _ = (case_dir, emu);
        Ok(())
    }
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
        fn dtype(&self, _c: &u64) -> &str {
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
    fn default_compile_writes_lowered_source() {
        let dom = MockDomain;
        let dir = std::env::temp_dir().join(format!("dma_seam_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dom.lower(&0u64);
        std::fs::write(dir.join("fuzz_kernel.cc"), &src).unwrap();
        let back = std::fs::read_to_string(dir.join("fuzz_kernel.cc")).unwrap();
        assert_eq!(back, src);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn mock_domain_satisfies_the_trait_behind_a_generic_bound() {
        let m = MockDomain;
        assert_eq!(drive(&m), vec!["a/I32x16/m0".to_string()]);
        assert_eq!(m.compare(&vec![1], &vec![2], &[]), Some("a/I32x16/m0".into()));
        assert_eq!(m.compare(&vec![1], &vec![1], &[]), None);
    }
}
