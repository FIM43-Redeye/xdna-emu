# Phoenix-Survival Capture (Output Corpus) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the tooling to freeze real NPU1 (Phoenix/AIE2) ground truth into a committed, hardware-free regression corpus, so the emulator's AIE2 fidelity stays regression-testable forever after the one-way Strix swap.

**Architecture:** A new `src/corpus/` library module defines schema-first serde types (`CaseMeta`, `Manifest`), a capture writer that the fuzzer calls to persist *every non-vacuous* case (not just mismatches) into a full archive split `agree/` + `diverge/`, a curation algorithm that selects a stratified subset into a committed git corpus, and a replay gate (`tests/corpus_replay.rs`) that re-runs each committed case through the in-process `XclbinSuite` with zero hardware. Gate-side code (schema/replay/diff/curate) compiles without the `tooling` feature; capture/tag (which need the fuzzer) sit behind `tooling`.

**Tech Stack:** Rust, serde/serde_json (already deps), sha2 (new dep), the existing `fuzzer` + `testing::{XclbinSuite, npu_runner, test_cpp_parser}` modules.

**Source spec:** `docs/superpowers/specs/2026-05-31-phoenix-survival-capture-design.md` (Sections 1-5, APPROVED). Tracks B/C/D are out of scope (named follow-ons).

---

## File Structure

**New library module `src/corpus/`** (responsibilities, one per file):
- `mod.rs` — module root + feature-gated submodule decls + re-exports.
- `schema.rs` — serde types: `CaseMeta`, `Manifest`, `Class`, `Tag`, `OpBand`, `Dtype`, `LoopStyleMeta`, `Provenance`, `Hashes`, `Counts`, `CaseIndexEntry`; `cell_id`/`cell_dir` helpers. No fuzzer/testing-exec deps (embeds `BufferSpec` only).
- `hashes.rs` — `sha256_hex(bytes) -> String`.
- `diff.rs` — `CaseDiff` (first-diff index, count, sample values, signature) + `diff_outputs(...)`.
- `replay.rs` — `replay_case(case_dir) -> CaseVerdict`: load meta, reconstruct, run `XclbinSuite`, compare. (Gate-side; no `tooling`.)
- `curate.rs` — `select_subset(metas, params) -> Vec<usize>`: grid floor / tag floor / all-divergences / dedup. (Gate-side.)
- `tag.rs` — `detect_tags(&FuzzParams) -> Vec<Tag>` (AST-derivable). `#[cfg(feature = "tooling")]`.
- `capture.rs` — `capture_case(...)`: write a case dir + build `CaseMeta`. `#[cfg(feature = "tooling")]`.

**Modified:**
- `Cargo.toml` — add `sha2`.
- `src/lib.rs` — add `pub mod corpus;`.
- `src/testing/test_cpp_parser.rs` — add serde derives to `BufferSpec`/`BufferDef`/`BufferDir`/`ElementType`/`InputPattern`.
- `src/fuzzer/gen.rs` — add `generate_grid_case(cell, seed)`.
- `src/fuzzer/cli.rs` — add `--capture <dir>` and `--capture-per-cell <N>` options.
- `src/fuzzer/runner.rs` — capture hook in the compare loop.

**New binary + tests + scripts:**
- `src/bin/curate-corpus.rs` — archive -> committed subset + manifest + coverage.md.
- `tests/corpus_replay.rs` — the regression gate.
- `tests/fixtures/corpus-mini/` — synthetic 2-case fixture for testing the gate without the real corpus.
- `scripts/replay-corpus.sh`, `scripts/reclassify-corpus-case.sh`.

---

## Task 1: Dependencies, module skeleton, serde on testing types

**Files:**
- Modify: `Cargo.toml`
- Modify: `src/lib.rs`
- Create: `src/corpus/mod.rs`
- Modify: `src/testing/test_cpp_parser.rs:29-143`

- [ ] **Step 1: Add the sha2 dependency**

In `Cargo.toml`, under `[dependencies]` (next to the existing `serde`/`serde_json` lines ~46-48), add:

```toml
sha2 = "0.10"
```

- [ ] **Step 2: Add serde derives to the testing buffer types**

In `src/testing/test_cpp_parser.rs`, add `Serialize, Deserialize` to the derive lists of these five types (keep existing derives). At the top of the file ensure the import exists:

```rust
use serde::{Deserialize, Serialize};
```

Then update each derive:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]            // BufferSpec  (was: Debug, Clone)
#[derive(Debug, Clone, Serialize, Deserialize)]            // BufferDef
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]  // BufferDir
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]  // ElementType
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]        // InputPattern
```

- [ ] **Step 3: Create the corpus module root**

Create `src/corpus/mod.rs`:

```rust
//! Phoenix-survival capture corpus: freeze real NPU1 ground truth into a
//! committed, hardware-free regression corpus. See
//! `docs/superpowers/specs/2026-05-31-phoenix-survival-capture-design.md`.

pub mod schema;
pub mod hashes;
pub mod diff;
pub mod replay;
pub mod curate;

#[cfg(feature = "tooling")]
pub mod tag;
#[cfg(feature = "tooling")]
pub mod capture;

pub use schema::{
    CaseMeta, Class, Counts, CaseIndexEntry, Dtype, Hashes, LoopStyleMeta, Manifest, OpBand,
    Provenance, Tag, SCHEMA_VERSION,
};
```

(`schema.rs` etc. are created in later tasks; this file will not compile until Task 2. That is expected — do not build yet.)

- [ ] **Step 4: Register the module**

In `src/lib.rs`, add alongside the other `pub mod` declarations:

```rust
pub mod corpus;
```

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock src/lib.rs src/corpus/mod.rs src/testing/test_cpp_parser.rs
git commit -m "corpus: add sha2 dep, module skeleton, serde on buffer types

Generated using Claude Code."
```

---

## Task 2: Schema types + round-trip test

**Files:**
- Create: `src/corpus/schema.rs`
- Test: `src/corpus/schema.rs` (inline `#[cfg(test)]`)

- [ ] **Step 1: Write the schema with a failing round-trip test**

Create `src/corpus/schema.rs`:

```rust
//! Schema-first serde types for the corpus. Types first, JSON derived.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::testing::test_cpp_parser::BufferSpec;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Class {
    Agree,
    Diverge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dtype {
    I8,
    I16,
    I32,
}

impl Dtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Dtype::I8 => "i8",
            Dtype::I16 => "i16",
            Dtype::I32 => "i32",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoopStyleMeta {
    Simple,
    HardwareLoop,
}

impl LoopStyleMeta {
    pub fn as_str(self) -> &'static str {
        match self {
            LoopStyleMeta::Simple => "Simple",
            LoopStyleMeta::HardwareLoop => "HardwareLoop",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpBand {
    B1to4,
    B5to9,
    B10to16,
}

impl OpBand {
    pub fn from_count(n: u32) -> Self {
        match n {
            0..=4 => OpBand::B1to4,
            5..=9 => OpBand::B5to9,
            _ => OpBand::B10to16,
        }
    }
    /// Logical form used inside `cell_id` (slash-separated).
    pub fn as_str(self) -> &'static str {
        match self {
            OpBand::B1to4 => "1-4",
            OpBand::B5to9 => "5-9",
            OpBand::B10to16 => "10-16",
        }
    }
    /// Filesystem-safe form used in directory names.
    pub fn as_path(self) -> &'static str {
        match self {
            OpBand::B1to4 => "1to4",
            OpBand::B5to9 => "5to9",
            OpBand::B10to16 => "10to16",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Tag {
    StoreAtLe,
    HasLoad,
    HasBranchTaken,
    HasHwloop,
    MultiStore,
}

impl Tag {
    pub fn as_str(self) -> &'static str {
        match self {
            Tag::StoreAtLe => "store_at_le",
            Tag::HasLoad => "has_load",
            Tag::HasBranchTaken => "has_branch_taken",
            Tag::HasHwloop => "has_hwloop",
            Tag::MultiStore => "multi_store",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub date: String,
    pub peano_commit: String,
    pub driver_version: String,
    pub fw_version: String,
    pub determinism_ok: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hashes {
    pub xclbin_sha256: String,
    pub npu_output_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub emu_output_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaseMeta {
    pub schema_version: u32,
    pub seed: u64,
    pub cell_id: String,
    pub dtype: Dtype,
    pub size_elements: usize,
    pub loop_style: LoopStyleMeta,
    pub op_count: u32,
    pub op_count_band: OpBand,
    pub class: Class,
    pub tags: Vec<Tag>,
    pub op_signature: String,
    pub vacuous: bool,
    pub buffer_spec: BufferSpec,
    pub capture: Provenance,
    pub hashes: Hashes,
}

/// Logical cell identifier, e.g. `i8/128/Simple/5-9`.
pub fn cell_id(dtype: Dtype, size: usize, loop_style: LoopStyleMeta, band: OpBand) -> String {
    format!("{}/{}/{}/{}", dtype.as_str(), size, loop_style.as_str(), band.as_str())
}

/// Filesystem-safe cell prefix for a case dir, e.g. `i8-128-Simple-5to9`.
pub fn cell_dir(dtype: Dtype, size: usize, loop_style: LoopStyleMeta, band: OpBand) -> String {
    format!("{}-{}-{}-{}", dtype.as_str(), size, loop_style.as_str(), band.as_path())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Counts {
    pub total: u32,
    pub agree: u32,
    pub diverge: u32,
    pub vacuous_excluded: u32,
    pub error: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaseIndexEntry {
    pub path: String,
    pub class: Class,
    pub tags: Vec<Tag>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub campaign_id: String,
    pub capture_date: String,
    pub counts: Counts,
    pub cell_coverage: BTreeMap<String, u32>,
    pub tag_coverage: BTreeMap<String, u32>,
    pub toolchain: Provenance,
    pub cases: Vec<CaseIndexEntry>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::test_cpp_parser::{BufferDef, BufferDir, ElementType, InputPattern};

    fn sample_spec() -> BufferSpec {
        BufferSpec {
            buffers: vec![BufferDef {
                name: "buf_in".into(),
                group_id: 3,
                size_elements: 128,
                element_type: ElementType::I8,
                direction: BufferDir::Input,
                input_pattern: InputPattern::Sequential { start: 1, step: 1 },
            }],
            multi_kernel: false,
        }
    }

    fn sample_meta() -> CaseMeta {
        CaseMeta {
            schema_version: SCHEMA_VERSION,
            seed: 412,
            cell_id: cell_id(Dtype::I8, 128, LoopStyleMeta::Simple, OpBand::B5to9),
            dtype: Dtype::I8,
            size_elements: 128,
            loop_style: LoopStyleMeta::Simple,
            op_count: 7,
            op_count_band: OpBand::B5to9,
            class: Class::Agree,
            tags: vec![Tag::HasHwloop],
            op_signature: "abc123".into(),
            vacuous: false,
            buffer_spec: sample_spec(),
            capture: Provenance {
                date: "2026-06-01".into(),
                peano_commit: "deadbeef".into(),
                driver_version: "2.x".into(),
                fw_version: "1.x".into(),
                determinism_ok: true,
            },
            hashes: Hashes {
                xclbin_sha256: "aa".into(),
                npu_output_sha256: "bb".into(),
                emu_output_sha256: None,
            },
        }
    }

    #[test]
    fn case_meta_round_trips() {
        let m = sample_meta();
        let json = serde_json::to_string_pretty(&m).unwrap();
        let back: CaseMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn op_band_from_count() {
        assert_eq!(OpBand::from_count(1), OpBand::B1to4);
        assert_eq!(OpBand::from_count(4), OpBand::B1to4);
        assert_eq!(OpBand::from_count(5), OpBand::B5to9);
        assert_eq!(OpBand::from_count(9), OpBand::B5to9);
        assert_eq!(OpBand::from_count(10), OpBand::B10to16);
        assert_eq!(OpBand::from_count(16), OpBand::B10to16);
    }

    #[test]
    fn cell_id_and_dir_forms() {
        assert_eq!(cell_id(Dtype::I8, 128, LoopStyleMeta::Simple, OpBand::B5to9), "i8/128/Simple/5-9");
        assert_eq!(cell_dir(Dtype::I8, 128, LoopStyleMeta::Simple, OpBand::B5to9), "i8-128-Simple-5to9");
    }
}
```

- [ ] **Step 2: Run the tests to verify they pass**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib corpus::schema -- --nocapture`
Expected: PASS (`case_meta_round_trips`, `op_band_from_count`, `cell_id_and_dir_forms`).

- [ ] **Step 3: Commit**

```bash
git add src/corpus/schema.rs
git commit -m "corpus: schema-first CaseMeta/Manifest serde types + round-trip test

Generated using Claude Code."
```

---

## Task 3: Hashing helper

**Files:**
- Create: `src/corpus/hashes.rs`
- Test: `src/corpus/hashes.rs` (inline)

- [ ] **Step 1: Write the failing test + implementation**

Create `src/corpus/hashes.rs`:

```rust
//! Content hashing for corpus integrity (xclbin / output fingerprints).

use sha2::{Digest, Sha256};

/// Lowercase hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let digest = h.finalize();
    let mut s = String::with_capacity(64);
    for b in digest {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vector() {
        // SHA-256("") well-known value.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn stable_and_distinct() {
        assert_eq!(sha256_hex(b"abc"), sha256_hex(b"abc"));
        assert_ne!(sha256_hex(b"abc"), sha256_hex(b"abd"));
    }
}
```

- [ ] **Step 2: Run the tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib corpus::hashes`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/corpus/hashes.rs
git commit -m "corpus: sha256_hex content hash helper

Generated using Claude Code."
```

---

## Task 4: Diff report

**Files:**
- Create: `src/corpus/diff.rs`
- Test: `src/corpus/diff.rs` (inline)

- [ ] **Step 1: Write the failing test + implementation**

Create `src/corpus/diff.rs`:

```rust
//! Per-case diff report so a future regression is diagnosable, not just red.

use serde::{Deserialize, Serialize};

use crate::corpus::hashes::sha256_hex;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaseDiff {
    /// Element-stride (bytes) used to interpret the buffers.
    pub elem_bytes: usize,
    /// Index of the first differing element, if any.
    pub first_diff_elem: Option<usize>,
    /// Number of differing elements.
    pub diff_count: usize,
    /// Up to 8 `(elem_index, expected_le_bytes, got_le_bytes)` samples.
    pub samples: Vec<(usize, Vec<u8>, Vec<u8>)>,
    /// Stable signature: sha256 over (expected, got), truncated to 16 hex.
    pub signature: String,
}

impl CaseDiff {
    pub fn is_equal(&self) -> bool {
        self.diff_count == 0
    }
}

/// Compare `got` against `expected` at the given element stride.
pub fn diff_outputs(expected: &[u8], got: &[u8], elem_bytes: usize) -> CaseDiff {
    let stride = elem_bytes.max(1);
    let n = expected.len().min(got.len()) / stride;
    let mut first = None;
    let mut count = 0usize;
    let mut samples = Vec::new();
    for i in 0..n {
        let a = &expected[i * stride..(i + 1) * stride];
        let b = &got[i * stride..(i + 1) * stride];
        if a != b {
            if first.is_none() {
                first = Some(i);
            }
            count += 1;
            if samples.len() < 8 {
                samples.push((i, a.to_vec(), b.to_vec()));
            }
        }
    }
    // Length mismatch counts as a difference too.
    if expected.len() != got.len() && first.is_none() {
        first = Some(n);
        count += 1;
    }
    let mut sig_input = Vec::with_capacity(expected.len() + got.len());
    sig_input.extend_from_slice(expected);
    sig_input.extend_from_slice(got);
    let signature = sha256_hex(&sig_input)[..16].to_string();
    CaseDiff { elem_bytes: stride, first_diff_elem: first, diff_count: count, samples, signature }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_outputs() {
        let d = diff_outputs(&[1, 2, 3, 4], &[1, 2, 3, 4], 1);
        assert!(d.is_equal());
        assert_eq!(d.first_diff_elem, None);
    }

    #[test]
    fn first_diff_and_count() {
        // i16 stride: elem0 = [1,0], elem1 = [2,0]; flip elem1.
        let exp = vec![1, 0, 2, 0, 3, 0];
        let got = vec![1, 0, 9, 0, 3, 0];
        let d = diff_outputs(&exp, &got, 2);
        assert_eq!(d.first_diff_elem, Some(1));
        assert_eq!(d.diff_count, 1);
        assert_eq!(d.samples[0].0, 1);
    }

    #[test]
    fn signature_is_stable() {
        let a = diff_outputs(&[1, 2], &[3, 4], 1);
        let b = diff_outputs(&[1, 2], &[3, 4], 1);
        assert_eq!(a.signature, b.signature);
        assert_eq!(a.signature.len(), 16);
    }
}
```

- [ ] **Step 2: Run the tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib corpus::diff`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/corpus/diff.rs
git commit -m "corpus: per-case diff report (first-diff, count, samples, signature)

Generated using Claude Code."
```

---

## Task 5: Curation algorithm

**Files:**
- Create: `src/corpus/curate.rs`
- Test: `src/corpus/curate.rs` (inline)

The selection: from all archive metas, choose a stratified subset — grid floor (first K per cell), tag floor (>=M per tag), ALL divergences, dedup by `op_signature` within a cell. Deterministic (input order = seed order).

- [ ] **Step 1: Write the failing test + implementation**

Create `src/corpus/curate.rs`:

```rust
//! Deterministic stratified selection of a committed subset from the archive.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use crate::corpus::schema::{CaseMeta, Class, Tag};

#[derive(Debug, Clone, Copy)]
pub struct CurateParams {
    /// Grid floor: cases kept per non-empty cell (K).
    pub per_cell: usize,
    /// Tag floor: minimum representatives per tag (M).
    pub per_tag: usize,
}

impl Default for CurateParams {
    fn default() -> Self {
        CurateParams { per_cell: 4, per_tag: 8 }
    }
}

/// Returns the indices (into `metas`) selected for the committed corpus.
/// Input order is assumed seed-ordered, making the selection deterministic.
pub fn select_subset(metas: &[CaseMeta], p: CurateParams) -> Vec<usize> {
    let mut selected: BTreeSet<usize> = BTreeSet::new();

    // 1. All divergences -- rare and unrecapturable.
    for (i, m) in metas.iter().enumerate() {
        if m.class == Class::Diverge {
            selected.insert(i);
        }
    }

    // 2. Grid floor: first K per cell, dedup by op_signature within the cell.
    let mut per_cell_count: BTreeMap<&str, usize> = BTreeMap::new();
    let mut seen_sig_in_cell: HashSet<(&str, &str)> = HashSet::new();
    for (i, m) in metas.iter().enumerate() {
        if m.vacuous {
            continue;
        }
        let cell = m.cell_id.as_str();
        let sig_key = (cell, m.op_signature.as_str());
        if seen_sig_in_cell.contains(&sig_key) {
            continue; // dedup near-identical kernels within a cell
        }
        let c = per_cell_count.entry(cell).or_insert(0);
        if *c < p.per_cell {
            seen_sig_in_cell.insert(sig_key);
            *c += 1;
            selected.insert(i);
        }
    }

    // 3. Tag floor: top up each tag to M from not-yet-selected non-vacuous cases.
    for tag in [Tag::StoreAtLe, Tag::HasLoad, Tag::HasBranchTaken, Tag::HasHwloop, Tag::MultiStore] {
        let have = selected
            .iter()
            .filter(|&&i| metas[i].tags.contains(&tag))
            .count();
        let mut need = p.per_tag.saturating_sub(have);
        if need == 0 {
            continue;
        }
        for (i, m) in metas.iter().enumerate() {
            if need == 0 {
                break;
            }
            if m.vacuous || selected.contains(&i) || !m.tags.contains(&tag) {
                continue;
            }
            selected.insert(i);
            need -= 1;
        }
    }

    selected.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::schema::*;
    use crate::testing::test_cpp_parser::BufferSpec;

    fn meta(seed: u64, cell: &str, class: Class, tags: Vec<Tag>, sig: &str, vacuous: bool) -> CaseMeta {
        CaseMeta {
            schema_version: SCHEMA_VERSION,
            seed,
            cell_id: cell.into(),
            dtype: Dtype::I8,
            size_elements: 16,
            loop_style: LoopStyleMeta::Simple,
            op_count: 3,
            op_count_band: OpBand::B1to4,
            class,
            tags,
            op_signature: sig.into(),
            vacuous,
            buffer_spec: BufferSpec { buffers: vec![], multi_kernel: false },
            capture: Provenance {
                date: "d".into(), peano_commit: "p".into(), driver_version: "v".into(),
                fw_version: "f".into(), determinism_ok: true,
            },
            hashes: Hashes { xclbin_sha256: "x".into(), npu_output_sha256: "n".into(), emu_output_sha256: None },
        }
    }

    #[test]
    fn all_divergences_kept() {
        let metas = vec![
            meta(1, "c", Class::Agree, vec![], "s1", false),
            meta(2, "c", Class::Diverge, vec![], "s2", false),
        ];
        let sel = select_subset(&metas, CurateParams { per_cell: 0, per_tag: 0 });
        assert!(sel.contains(&1)); // the diverge case
    }

    #[test]
    fn grid_floor_and_dedup() {
        let metas = vec![
            meta(1, "c", Class::Agree, vec![], "same", false),
            meta(2, "c", Class::Agree, vec![], "same", false), // dup signature -> skipped
            meta(3, "c", Class::Agree, vec![], "diff", false),
        ];
        let sel = select_subset(&metas, CurateParams { per_cell: 2, per_tag: 0 });
        assert!(sel.contains(&0));
        assert!(!sel.contains(&1)); // deduped
        assert!(sel.contains(&2));
    }

    #[test]
    fn vacuous_excluded() {
        let metas = vec![meta(1, "c", Class::Agree, vec![], "s", true)];
        let sel = select_subset(&metas, CurateParams { per_cell: 4, per_tag: 0 });
        assert!(sel.is_empty());
    }

    #[test]
    fn tag_floor_tops_up() {
        let metas = vec![
            meta(1, "c", Class::Agree, vec![Tag::StoreAtLe], "s1", false),
            meta(2, "d", Class::Agree, vec![Tag::StoreAtLe], "s2", false),
            meta(3, "e", Class::Agree, vec![Tag::StoreAtLe], "s3", false),
        ];
        // per_cell 0 so grid floor selects nothing; tag floor must reach 2.
        let sel = select_subset(&metas, CurateParams { per_cell: 0, per_tag: 2 });
        let store_at_le = sel.iter().filter(|&&i| metas[i].tags.contains(&Tag::StoreAtLe)).count();
        assert_eq!(store_at_le, 2);
    }
}
```

- [ ] **Step 2: Run the tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib corpus::curate`
Expected: PASS (4 tests).

- [ ] **Step 3: Commit**

```bash
git add src/corpus/curate.rs
git commit -m "corpus: stratified curation (grid floor, tag floor, all-divergences, dedup)

Generated using Claude Code."
```

---

## Task 6: Replay (the gate's per-case engine)

**Files:**
- Create: `src/corpus/replay.rs`
- Test: covered by the gate's synthetic fixture in Task 11 (replay needs a real xclbin, so the unit test here only covers meta loading + verdict logic with a stubbed runner is impractical; we test the pure verdict logic).

- [ ] **Step 1: Write replay with a verdict-logic unit test**

Create `src/corpus/replay.rs`:

```rust
//! Gate engine: replay one committed case through the in-process emulator and
//! decide its verdict. No hardware, no license.

use std::path::{Path, PathBuf};

use crate::corpus::diff::{diff_outputs, CaseDiff};
use crate::corpus::schema::{CaseMeta, Class, Dtype};
use crate::testing::xclbin_suite::{XclbinSuite, XclbinTest};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Case behaved as frozen (agree==HW, diverge==EMU).
    Ok,
    /// Regression / unexpected change. Carries the diff.
    Mismatch(CaseDiff),
    /// Case could not be replayed (missing file, emulator produced nothing).
    Error(String),
}

#[derive(Debug, Clone)]
pub struct CaseResult {
    pub dir: PathBuf,
    pub cell_id: String,
    pub class: Class,
    pub verdict: Verdict,
}

fn dtype_bytes(d: Dtype) -> usize {
    match d {
        Dtype::I8 => 1,
        Dtype::I16 => 2,
        Dtype::I32 => 4,
    }
}

/// Load a case dir, run its xclbin through the emulator, compare against the
/// frozen reference for its class.
pub fn replay_case(dir: &Path) -> CaseResult {
    let meta_path = dir.join("meta.json");
    let meta: CaseMeta = match std::fs::read_to_string(&meta_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
    {
        Some(m) => m,
        None => {
            return CaseResult {
                dir: dir.to_path_buf(),
                cell_id: String::new(),
                class: Class::Agree,
                verdict: Verdict::Error(format!("cannot read {}", meta_path.display())),
            }
        }
    };

    let elem_bytes = dtype_bytes(meta.dtype);
    let nbytes = meta.size_elements * elem_bytes;

    // Run the frozen xclbin through the in-process emulator.
    let test = XclbinTest::from_path(dir.join("aie.xclbin")).with_buffer_spec(meta.buffer_spec.clone());
    let suite = XclbinSuite::new();
    let (_outcome, raw, _trace) = suite.run_single_with_trace(&test);
    let mut emu = match raw {
        Some(v) => v,
        None => {
            return CaseResult {
                dir: dir.to_path_buf(),
                cell_id: meta.cell_id,
                class: meta.class,
                verdict: Verdict::Error("emulator produced no output".into()),
            }
        }
    };
    emu.truncate(nbytes);

    // Reference depends on class: agree -> frozen HW; diverge -> frozen EMU.
    let ref_name = match meta.class {
        Class::Agree => "npu_output.bin",
        Class::Diverge => "emu_output.bin",
    };
    let mut reference = std::fs::read(dir.join(ref_name)).unwrap_or_default();
    reference.truncate(nbytes);

    let d = diff_outputs(&reference, &emu, elem_bytes);
    let verdict = if d.is_equal() { Verdict::Ok } else { Verdict::Mismatch(d) };

    CaseResult { dir: dir.to_path_buf(), cell_id: meta.cell_id, class: meta.class, verdict }
}

/// Walk `agree/` and `diverge/` under `corpus_root`, replaying every case dir
/// (a dir containing `meta.json`). Returns all results.
pub fn replay_corpus(corpus_root: &Path) -> Vec<CaseResult> {
    let mut out = Vec::new();
    for class_dir in ["agree", "diverge"] {
        let base = corpus_root.join(class_dir);
        let Ok(entries) = std::fs::read_dir(&base) else { continue };
        let mut dirs: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.join("meta.json").exists())
            .collect();
        dirs.sort();
        for d in dirs {
            out.push(replay_case(&d));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_meta_is_error() {
        let r = replay_case(Path::new("/nonexistent/case_dir_xyz"));
        assert!(matches!(r.verdict, Verdict::Error(_)));
    }

    #[test]
    fn empty_corpus_root_yields_nothing() {
        let tmp = std::env::temp_dir().join("corpus_replay_empty_test");
        let _ = std::fs::create_dir_all(&tmp);
        assert!(replay_corpus(&tmp).is_empty());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
```

- [ ] **Step 2: Run the tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib corpus::replay`
Expected: PASS (2 tests).

- [ ] **Step 3: Commit**

```bash
git add src/corpus/replay.rs
git commit -m "corpus: replay engine (frozen-reference compare per class) + walker

Generated using Claude Code."
```

---

## Task 7: Tag detection (AST-derivable)

**Files:**
- Create: `src/corpus/tag.rs`
- Test: `src/corpus/tag.rs` (inline)

Scope: this detects the three AST-derivable tags. `store_at_le`/`has_load` need disasm and are a named follow-up (Task 13 note).

- [ ] **Step 1: Write the failing test + implementation**

Create `src/corpus/tag.rs`:

```rust
//! Emergent-regime tagging derivable from the kernel AST. `store_at_le` and
//! `has_load` require disassembly and are added by a separate pass (see plan).

use crate::corpus::schema::Tag;
use crate::fuzzer::ast::{KernelOp, LoopStyle};
use crate::fuzzer::params::FuzzParams;

fn walk<'a>(ops: &'a [KernelOp], f: &mut impl FnMut(&'a KernelOp)) {
    for op in ops {
        f(op);
        match op {
            KernelOp::Branch { then_ops, else_ops, .. } => {
                walk(then_ops, f);
                walk(else_ops, f);
            }
            KernelOp::HwLoop { body, .. } => walk(body, f),
            _ => {}
        }
    }
}

pub fn detect_tags(params: &FuzzParams) -> Vec<Tag> {
    let ops = &params.body.ops;

    let mut stores = 0usize;
    let mut has_branch = false;
    let mut has_inner_hwloop = false;
    walk(ops, &mut |op| match op {
        KernelOp::Store { .. } => stores += 1,
        KernelOp::Branch { .. } => has_branch = true,
        KernelOp::HwLoop { .. } => has_inner_hwloop = true,
        _ => {}
    });

    let mut tags = Vec::new();
    if matches!(params.body.loop_style, LoopStyle::HardwareLoop) || has_inner_hwloop {
        tags.push(Tag::HasHwloop);
    }
    if has_branch {
        tags.push(Tag::HasBranchTaken);
    }
    if stores > 1 {
        tags.push(Tag::MultiStore);
    }
    tags.sort();
    tags.dedup();
    tags
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzzer::ast::{KernelBody, Operand, Var, BufRef};
    use crate::fuzzer::params::ScalarType;

    fn params(ops: Vec<KernelOp>, ls: LoopStyle) -> FuzzParams {
        FuzzParams { seed: 1, buffer_size: 16, dtype: ScalarType::I32, body: KernelBody { ops, loop_style: ls } }
    }

    fn store() -> KernelOp {
        KernelOp::Store { buf: BufRef(1), idx: Operand::Var(Var(1)), val: Operand::Var(Var(0)) }
    }

    #[test]
    fn hwloop_style_tagged() {
        let t = detect_tags(&params(vec![store()], LoopStyle::HardwareLoop));
        assert!(t.contains(&Tag::HasHwloop));
    }

    #[test]
    fn multi_store_tagged() {
        let t = detect_tags(&params(vec![store(), store()], LoopStyle::Simple));
        assert!(t.contains(&Tag::MultiStore));
    }

    #[test]
    fn simple_single_store_untagged() {
        let t = detect_tags(&params(vec![store()], LoopStyle::Simple));
        assert!(t.is_empty());
    }
}
```

(If `BufRef`/`Var`/`Operand` are not `pub` in `src/fuzzer/ast.rs`, make them `pub` — they are already used across the fuzzer.)

- [ ] **Step 2: Run the tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --features tooling --lib corpus::tag`
Expected: PASS (3 tests).

- [ ] **Step 3: Commit**

```bash
git add src/corpus/tag.rs
git commit -m "corpus: AST-derivable emergent-regime tagging (hwloop/branch/multi_store)

Generated using Claude Code."
```

---

## Task 8: Capture writer

**Files:**
- Create: `src/corpus/capture.rs`
- Test: `src/corpus/capture.rs` (inline, temp dir)

Writes one case dir into the archive and returns its `CaseMeta`. Pure I/O + meta assembly; the runner (Task 9) calls it with the bytes it already has.

- [ ] **Step 1: Write the failing test + implementation**

Create `src/corpus/capture.rs`:

```rust
//! Capture a single case into the archive: copy the xclbin, write the golden
//! output(s), and emit meta.json. Called by the fuzzer's capture mode.

use std::path::{Path, PathBuf};

use crate::corpus::hashes::sha256_hex;
use crate::corpus::schema::*;
use crate::corpus::tag::detect_tags;
use crate::fuzzer::ast::LoopStyle;
use crate::fuzzer::params::{FuzzParams, ScalarType};
use crate::testing::test_cpp_parser::BufferSpec;

fn dtype_of(s: ScalarType) -> Dtype {
    match s {
        ScalarType::I8 => Dtype::I8,
        ScalarType::I16 => Dtype::I16,
        ScalarType::I32 => Dtype::I32,
    }
}

fn loop_style_of(l: LoopStyle) -> LoopStyleMeta {
    match l {
        LoopStyle::Simple => LoopStyleMeta::Simple,
        LoopStyle::HardwareLoop => LoopStyleMeta::HardwareLoop,
    }
}

/// Op count for the cell axis: the generated body length (the final store is
/// part of the body, so this matches the generator's `num_ops + 1`; the band
/// edges are what matter, not the exact convention).
fn op_count(params: &FuzzParams) -> u32 {
    params.body.ops.len() as u32
}

/// Stable signature of the op sequence for dedup.
fn op_signature(params: &FuzzParams) -> String {
    sha256_hex(format!("{:?}", params.body.ops).as_bytes())[..16].to_string()
}

pub struct CaptureInput<'a> {
    pub params: &'a FuzzParams,
    pub buffer_spec: BufferSpec,
    pub class: Class,
    pub vacuous: bool,
    pub determinism_ok: bool,
    pub xclbin_src: &'a Path,
    pub npu_output: &'a [u8],
    /// Required for Diverge cases (the frozen EMU reference); ignored for Agree.
    pub emu_output: Option<&'a [u8]>,
    pub provenance_date: &'a str,
    pub peano_commit: &'a str,
    pub driver_version: &'a str,
    pub fw_version: &'a str,
}

/// Build the `CaseMeta` for an input (no I/O). Split out for testability.
pub fn build_meta(input: &CaptureInput, xclbin_bytes: &[u8]) -> CaseMeta {
    let p = input.params;
    let dtype = dtype_of(p.dtype);
    let loop_style = loop_style_of(p.body.loop_style);
    let band = OpBand::from_count(op_count(p));
    let emu_sha = input.emu_output.map(sha256_hex);
    CaseMeta {
        schema_version: SCHEMA_VERSION,
        seed: p.seed,
        cell_id: cell_id(dtype, p.buffer_size, loop_style, band),
        dtype,
        size_elements: p.buffer_size,
        loop_style,
        op_count: op_count(p),
        op_count_band: band,
        class: input.class,
        tags: detect_tags(p),
        op_signature: op_signature(p),
        vacuous: input.vacuous,
        buffer_spec: input.buffer_spec.clone(),
        capture: Provenance {
            date: input.provenance_date.to_string(),
            peano_commit: input.peano_commit.to_string(),
            driver_version: input.driver_version.to_string(),
            fw_version: input.fw_version.to_string(),
            determinism_ok: input.determinism_ok,
        },
        hashes: Hashes {
            xclbin_sha256: sha256_hex(xclbin_bytes),
            npu_output_sha256: sha256_hex(input.npu_output),
            emu_output_sha256: emu_sha,
        },
    }
}

/// Write the case into `<archive_root>/<class>/<cell_dir>_<seed>/` and return
/// the directory path. Copies xclbin, writes npu_output.bin (+ emu_output.bin
/// for diverge), and meta.json.
pub fn capture_case(archive_root: &Path, input: &CaptureInput) -> std::io::Result<PathBuf> {
    let xclbin_bytes = std::fs::read(input.xclbin_src)?;
    let meta = build_meta(input, &xclbin_bytes);

    let class_dir = match input.class {
        Class::Agree => "agree",
        Class::Diverge => "diverge",
    };
    let dtype = dtype_of(input.params.dtype);
    let band = OpBand::from_count(op_count(input.params));
    let dir = archive_root.join(class_dir).join(format!(
        "{}_{:06}",
        cell_dir(dtype, input.params.buffer_size, loop_style_of(input.params.body.loop_style), band),
        input.params.seed
    ));
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("aie.xclbin"), &xclbin_bytes)?;
    std::fs::write(dir.join("npu_output.bin"), input.npu_output)?;
    if input.class == Class::Diverge {
        if let Some(e) = input.emu_output {
            std::fs::write(dir.join("emu_output.bin"), e)?;
        }
    }
    std::fs::write(dir.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzzer::ast::{BufRef, KernelBody, KernelOp, Operand, Var};

    fn fparams() -> FuzzParams {
        FuzzParams {
            seed: 412,
            buffer_size: 128,
            dtype: ScalarType::I8,
            body: KernelBody {
                ops: vec![KernelOp::Store { buf: BufRef(1), idx: Operand::Var(Var(1)), val: Operand::Var(Var(0)) }],
                loop_style: LoopStyle::Simple,
            },
        }
    }

    #[test]
    fn writes_agree_case() {
        let tmp = std::env::temp_dir().join(format!("corpus_capture_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let xclbin = tmp.join("src.xclbin");
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(&xclbin, b"FAKEXCLBIN").unwrap();
        let p = fparams();
        let input = CaptureInput {
            params: &p,
            buffer_spec: BufferSpec { buffers: vec![], multi_kernel: false },
            class: Class::Agree,
            vacuous: false,
            determinism_ok: true,
            xclbin_src: &xclbin,
            npu_output: &[1, 2, 3, 4],
            emu_output: None,
            provenance_date: "2026-06-01",
            peano_commit: "abc",
            driver_version: "v",
            fw_version: "f",
        };
        let dir = capture_case(&tmp, &input).unwrap();
        assert!(dir.join("aie.xclbin").exists());
        assert!(dir.join("npu_output.bin").exists());
        assert!(!dir.join("emu_output.bin").exists()); // agree has no emu ref
        let meta: CaseMeta =
            serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
        assert_eq!(meta.seed, 412);
        assert_eq!(meta.cell_id, "i8/128/Simple/1-4");
        assert_eq!(meta.class, Class::Agree);
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
```

- [ ] **Step 2: Run the tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --features tooling --lib corpus::capture`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/corpus/capture.rs
git commit -m "corpus: capture writer (case dir + meta.json assembly) with temp-dir test

Generated using Claude Code."
```

---

## Task 9: Grid generator

**Files:**
- Modify: `src/fuzzer/gen.rs`
- Test: `src/fuzzer/gen.rs` (inline)

Adds a constrained generator that fixes the four grid axes for a cell and randomizes only the body, so the campaign guarantees per-cell coverage.

- [ ] **Step 1: Write the failing test + implementation**

In `src/fuzzer/gen.rs`, add (after `generate`):

```rust
/// A single grid cell's fixed axes (Section 2 of the spec).
#[derive(Debug, Clone, Copy)]
pub struct GridCell {
    pub dtype: ScalarType,
    pub buffer_size: usize,
    pub loop_style: LoopStyle,
    /// Inclusive op-count range for this cell's band, e.g. (5, 9).
    pub op_count_lo: usize,
    pub op_count_hi: usize,
}

/// The 90-cell grid: dtype(3) x size(5) x loop_style(2) x op-band(3).
pub fn grid_cells() -> Vec<GridCell> {
    let dtypes = [ScalarType::I32, ScalarType::I16, ScalarType::I8];
    let sizes = [16usize, 32, 64, 128, 256];
    let loops = [LoopStyle::Simple, LoopStyle::HardwareLoop];
    let bands = [(1usize, 4usize), (5, 9), (10, 16)];
    let mut cells = Vec::with_capacity(90);
    for &dtype in &dtypes {
        for &buffer_size in &sizes {
            for &loop_style in &loops {
                for &(lo, hi) in &bands {
                    cells.push(GridCell { dtype, buffer_size, loop_style, op_count_lo: lo, op_count_hi: hi });
                }
            }
        }
    }
    cells
}

/// Constrained generation: the cell fixes the four axes; the seed drives only
/// the op body (so determinism holds and each cell is guaranteed non-empty).
pub fn generate_grid_case(cell: GridCell, seed: u64) -> FuzzParams {
    let mut rng = Xorshift64(if seed == 0 { 1 } else { seed });
    let span = (cell.op_count_hi - cell.op_count_lo + 1) as u64;
    let num_ops = cell.op_count_lo + (rng.next() % span) as usize;
    let mut ops = Vec::with_capacity(num_ops + 1);
    for _ in 0..num_ops {
        ops.push(gen_op(&mut rng));
    }
    ops.push(KernelOp::Store { buf: BufRef(1), idx: Operand::Var(Var(1)), val: Operand::Var(Var(0)) });
    FuzzParams {
        seed,
        buffer_size: cell.buffer_size,
        dtype: cell.dtype,
        body: KernelBody { ops, loop_style: cell.loop_style },
    }
}

#[cfg(test)]
mod grid_tests {
    use super::*;

    #[test]
    fn grid_has_90_cells() {
        assert_eq!(grid_cells().len(), 90);
    }

    #[test]
    fn grid_case_respects_cell_axes() {
        let cell = GridCell {
            dtype: ScalarType::I16,
            buffer_size: 64,
            loop_style: LoopStyle::HardwareLoop,
            op_count_lo: 5,
            op_count_hi: 9,
        };
        let p = generate_grid_case(cell, 12345);
        assert_eq!(p.dtype, ScalarType::I16);
        assert_eq!(p.buffer_size, 64);
        assert_eq!(p.body.loop_style, LoopStyle::HardwareLoop);
        // body length = num_ops (5..=9) + 1 trailing store.
        let body_ops = p.body.ops.len();
        assert!((6..=10).contains(&body_ops), "got {body_ops}");
    }

    #[test]
    fn grid_case_is_deterministic() {
        let cell = grid_cells()[0];
        let a = generate_grid_case(cell, 99);
        let b = generate_grid_case(cell, 99);
        assert_eq!(format!("{:?}", a.body.ops), format!("{:?}", b.body.ops));
    }
}
```

(Ensure `gen_op`, `Xorshift64`, `KernelBody`, `KernelOp`, `BufRef`, `Var`, `Operand` are in scope in `gen.rs` — they already are, since `generate` uses them. Make `gen_op` non-`pub`-fine since the test is in the same module.)

- [ ] **Step 2: Run the tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --features tooling --lib fuzzer::gen::grid_tests`
Expected: PASS (3 tests).

- [ ] **Step 3: Commit**

```bash
git add src/fuzzer/gen.rs
git commit -m "fuzzer: constrained grid generator (90-cell coverage axes)

Generated using Claude Code."
```

---

## Task 10: Capture mode wiring (CLI + runner hook)

**Files:**
- Modify: `src/fuzzer/cli.rs:27-59` (and the `FuzzOptions` struct it fills)
- Modify: `src/fuzzer/runner.rs:464-552` (compare loop) and the case-build site

This is integration glue around hardware; it is verified manually (Step 4) rather than by a unit test, because it requires the live NPU.

- [ ] **Step 1: Add capture options to the CLI**

Find the `FuzzOptions` struct (filled in `parse_fuzz_args`, `src/fuzzer/cli.rs`). Add fields:

```rust
    /// When set, run in corpus-capture mode writing every non-vacuous case here.
    pub capture_dir: Option<std::path::PathBuf>,
    /// Cases per grid cell during capture (N).
    pub capture_per_cell: usize,
```

In the `FuzzOptions { ... }` initializer inside `parse_fuzz_args`, add defaults:

```rust
        capture_dir: None,
        capture_per_cell: 12,
```

In the `match arg.as_str()` arm list, add:

```rust
            "--capture" => opts.capture_dir = Some(std::path::PathBuf::from(parse_next_str(&mut iter, "--capture")?)),
            "--capture-per-cell" => opts.capture_per_cell = parse_next(&mut iter, "--capture-per-cell")?,
```

If a string-valued `parse_next_str` helper does not exist, add it next to `parse_next`:

```rust
fn parse_next_str(iter: &mut std::slice::Iter<String>, flag: &str) -> Result<String, String> {
    iter.next().cloned().ok_or_else(|| format!("{} requires a value", flag))
}
```

- [ ] **Step 2: Add the capture hook to the runner's compare loop**

In `src/fuzzer/runner.rs`, inside the comparison loop (lines 464-552), capture mode persists *every non-vacuous* case. Locate the `pass`/`vacuous`/`fail` branches and add capture calls. The runner already has `opts` in scope; gather provenance once before the loop:

```rust
// Before the compare loop, when in capture mode, gather provenance once.
let capture_ctx = opts.capture_dir.as_ref().map(|root| {
    let date = std::process::Command::new("date").arg("-I").output()
        .ok().and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string()).unwrap_or_default();
    let peano = std::process::Command::new("git")
        .args(["-C", "../llvm-aie", "rev-parse", "--short", "HEAD"]).output()
        .ok().and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string()).unwrap_or_default();
    let driver = std::fs::read_to_string("/sys/module/amdxdna/version").unwrap_or_default().trim().to_string();
    (root.clone(), date, peano, driver)
});
```

In the **pass** branch (outputs match, not all-zero — currently line ~518), add:

```rust
            pass += 1;
            if let Some((root, date, peano, driver)) = &capture_ctx {
                // Determinism gate: re-run EMU once, require byte-identical.
                let det_ok = match run_emulator(&case.xclbin_path, &case.params, opts.max_cycles) {
                    Ok((again, _)) => again.as_slice() == emu_output.as_slice(),
                    Err(_) => false,
                };
                let spec = make_fuzz_buffer_spec(&case.params);
                let input = crate::corpus::capture::CaptureInput {
                    params: &case.params,
                    buffer_spec: spec,
                    class: crate::corpus::schema::Class::Agree,
                    vacuous: false,
                    determinism_ok: det_ok,
                    xclbin_src: &case.xclbin_path,
                    npu_output: &npu_output,
                    emu_output: None,
                    provenance_date: date,
                    peano_commit: peano,
                    driver_version: driver,
                    fw_version: "",
                };
                if let Err(e) = crate::corpus::capture::capture_case(root, &input) {
                    eprintln!("seed {} capture(agree) failed: {}", case.seed, e);
                }
            }
```

In the **fail** branch (mismatch — currently line ~528, after the existing `std::fs::write(... emu_output.bin ...)` block), add:

```rust
            if let Some((root, date, peano, driver)) = &capture_ctx {
                let spec = make_fuzz_buffer_spec(&case.params);
                let input = crate::corpus::capture::CaptureInput {
                    params: &case.params,
                    buffer_spec: spec,
                    class: crate::corpus::schema::Class::Diverge,
                    vacuous: false,
                    determinism_ok: true,
                    xclbin_src: &case.xclbin_path,
                    npu_output: &npu_output,
                    emu_output: Some(&emu_output),
                    provenance_date: date,
                    peano_commit: peano,
                    driver_version: driver,
                    fw_version: "",
                };
                if let Err(e) = crate::corpus::capture::capture_case(root, &input) {
                    eprintln!("seed {} capture(diverge) failed: {}", case.seed, e);
                }
            }
```

(Vacuous and error/crash branches: do nothing — excluded from the corpus by design.)

NOTE: this hook references `case.xclbin_path`. Confirm the per-case struct (the `cases` element) exposes the compiled xclbin path; the HW runner already receives an `xclbin_path`, so thread the same value onto the case struct if it is not already a field. If the field is named differently (e.g. `case.case_dir.join("aie.xclbin")`), use that instead.

- [ ] **Step 3: Drive capture from the grid when `--capture` is set**

At the generation site in `run_fuzz` (where seeds currently map to `generate(seed)`), branch on capture mode so generation walks the grid `N` times per cell instead of random seeds:

```rust
let cases_to_generate: Vec<(u64, FuzzParams)> = if let Some(_root) = &opts.capture_dir {
    let mut v = Vec::new();
    let cells = crate::fuzzer::gen::grid_cells();
    for (ci, cell) in cells.iter().enumerate() {
        for n in 0..opts.capture_per_cell {
            // Deterministic per-(cell, n) seed.
            let seed = 1 + (ci as u64) * 100_000 + n as u64;
            v.push((seed, crate::fuzzer::gen::generate_grid_case(*cell, seed)));
        }
    }
    v
} else {
    // existing random-seed path, unchanged
    (0..opts.fuzz_iterations).map(|i| {
        let seed = opts.fuzz_seed.unwrap_or(0).wrapping_add(i as u64);
        (seed, crate::fuzzer::gen::generate(seed))
    }).collect()
};
```

(Adapt to the actual existing generation/loop shape in `run_fuzz`; the key change is: in capture mode, source `(seed, FuzzParams)` from `grid_cells()` x `capture_per_cell` via `generate_grid_case`.)

- [ ] **Step 4: Build and manually verify capture mode (EMU-only smoke, then HW)**

Build: `cargo build --features tooling`
Expected: compiles clean.

EMU-only smoke (no hardware; agree/diverge split still exercised against EMU-as-its-own-ref is meaningless, so this only checks the dir/meta plumbing — run a tiny grid):

```bash
cargo run --features tooling -- fuzz --capture /tmp/claude-1000/cap-smoke --capture-per-cell 1 --no-hw
```
Expected: `/tmp/claude-1000/cap-smoke/agree/.../meta.json` files exist and parse. (With `--no-hw` there is no HW reference; treat this purely as a plumbing smoke — confirm meta.json is well-formed.)

HW capture (requires live Phoenix; run un-sandboxed per repo ops rules):
```bash
cargo run --features tooling -- fuzz --capture ~/npu-work/experiments/phoenix-aie2-golden-smoke --capture-per-cell 1 --hw
```
Expected: `agree/` populated; any divergences under `diverge/` with `emu_output.bin`.

- [ ] **Step 5: Commit**

```bash
git add src/fuzzer/cli.rs src/fuzzer/runner.rs
git commit -m "fuzzer: --capture mode writes every non-vacuous case to the corpus archive

Generated using Claude Code."
```

---

## Task 11: Regression gate test + synthetic fixture

**Files:**
- Create: `tests/corpus_replay.rs`
- Create: `tests/fixtures/corpus-mini/agree/<case>/...` and `.../diverge/<case>/...`
- Create: `scripts/replay-corpus.sh`

The committed corpus does not exist until the campaign runs (Task 13), so the gate must (a) pass trivially when `tests/corpus/phoenix-aie2/` is empty/absent, and (b) be exercised now by a tiny synthetic fixture that has a real xclbin.

- [ ] **Step 1: Build the synthetic fixture**

Use the smallest real fuzz case as the fixture's xclbin source. Generate one case and copy its artifacts:

```bash
mkdir -p tests/fixtures/corpus-mini/agree tests/fixtures/corpus-mini/diverge
# produce a tiny known-good xclbin via the fuzzer (EMU is the reference here)
cargo run --features tooling -- fuzz --iterations 1 --seed 1 --no-hw
CASE=build/fuzz/seed_1
# Build an 'agree' fixture whose npu_output.bin IS the emulator output, so the
# gate's agree-assert (EMU==npu_output.bin) trivially holds for a correct emu.
cargo run --release --example validate_seeds -- build/fuzz   # writes nothing; used to confirm emu output
```

Then hand-create the fixture meta.json for the agree case (fill `buffer_spec` to match the fuzz layout, `size_elements`/`dtype` from the case's `aie.mlir`), copy `aie.xclbin`, and set `npu_output.bin` to the emulator's own output for that xclbin so a correct emulator passes. For the diverge case, set `emu_output.bin` to the emulator output and `npu_output.bin` to a deliberately different blob. (A helper to mint fixtures is out of scope; document the two fixture dirs in a `tests/fixtures/corpus-mini/README.md`.)

Minimum fixture `meta.json` (agree) — adjust `size_elements`/`dtype`/`buffer_spec` to the generated case:

```json
{
  "schema_version": 1,
  "seed": 1,
  "cell_id": "i32/16/Simple/1-4",
  "dtype": "I32",
  "size_elements": 16,
  "loop_style": "Simple",
  "op_count": 2,
  "op_count_band": "B1to4",
  "class": "Agree",
  "tags": [],
  "op_signature": "fixture",
  "vacuous": false,
  "buffer_spec": { "buffers": [
    {"name":"buf_in","group_id":3,"size_elements":16,"element_type":"I32","direction":"Input","input_pattern":{"Sequential":{"start":1,"step":1}}},
    {"name":"buf_scratch","group_id":4,"size_elements":16,"element_type":"I32","direction":"Input","input_pattern":"Zeros"},
    {"name":"buf_out","group_id":5,"size_elements":16,"element_type":"I32","direction":"Output","input_pattern":"Zeros"},
    {"name":"buf_trace","group_id":6,"size_elements":262144,"element_type":"I32","direction":"Output","input_pattern":"Zeros"}
  ], "multi_kernel": false },
  "capture": {"date":"2026-06-01","peano_commit":"fixture","driver_version":"fixture","fw_version":"fixture","determinism_ok":true},
  "hashes": {"xclbin_sha256":"fixture","npu_output_sha256":"fixture"}
}
```

- [ ] **Step 2: Write the gate test**

Create `tests/corpus_replay.rs`:

```rust
//! Phoenix-survival regression gate. Replays the committed corpus (and a
//! synthetic fixture) through the in-process emulator with zero hardware.

use std::path::Path;

use xdna_emu::corpus::replay::{replay_corpus, Verdict};

fn assert_corpus_clean(root: &Path) {
    if !root.exists() {
        eprintln!("corpus root {} absent -- skipping (campaign not yet run)", root.display());
        return;
    }
    let results = replay_corpus(root);
    let mut failures = Vec::new();
    for r in &results {
        match &r.verdict {
            Verdict::Ok => {}
            Verdict::Mismatch(d) => failures.push(format!(
                "{} [{}]: first_diff_elem={:?} count={} sig={}",
                r.dir.display(), r.cell_id, d.first_diff_elem, d.diff_count, d.signature
            )),
            Verdict::Error(e) => failures.push(format!("{}: ERROR {}", r.dir.display(), e)),
        }
    }
    eprintln!("corpus {}: {} cases, {} failures", root.display(), results.len(), failures.len());
    assert!(failures.is_empty(), "corpus regressions:\n{}", failures.join("\n"));
}

#[test]
fn fixture_corpus_replays_clean() {
    assert_corpus_clean(Path::new("tests/fixtures/corpus-mini"));
}

#[test]
fn committed_corpus_replays_clean() {
    // Trivially passes until the capture campaign populates this dir.
    assert_corpus_clean(Path::new("tests/corpus/phoenix-aie2"));
}
```

- [ ] **Step 3: Run the gate**

Run: `TMPDIR=/tmp/claude-1000 cargo test --test corpus_replay -- --nocapture`
Expected: PASS. `fixture_corpus_replays_clean` replays the synthetic agree+diverge cases (agree: EMU==npu_output.bin; diverge: EMU==emu_output.bin). `committed_corpus_replays_clean` prints "absent -- skipping".

- [ ] **Step 4: Add the ad-hoc replay script**

Create `scripts/replay-corpus.sh`:

```bash
#!/usr/bin/env bash
# Replay a corpus dir through the emulator (zero HW). Usage: replay-corpus.sh [corpus_root]
set -euo pipefail
ROOT="${1:-tests/corpus/phoenix-aie2}"
cd "$(dirname "$0")/.."
CORPUS_ROOT="$ROOT" cargo test --test corpus_replay committed_corpus_replays_clean -- --nocapture
```

(Optional enhancement: have the gate read `CORPUS_ROOT` env to override the path; if added, document it. For the base plan the script targets the default path.)

```bash
chmod +x scripts/replay-corpus.sh
```

- [ ] **Step 5: Commit**

```bash
git add tests/corpus_replay.rs tests/fixtures/corpus-mini scripts/replay-corpus.sh
git commit -m "corpus: regression gate test + synthetic fixture + replay script

Generated using Claude Code."
```

---

## Task 12: Curation binary + manifest/coverage generation

**Files:**
- Create: `src/bin/curate-corpus.rs`
- Create: `scripts/reclassify-corpus-case.sh`

- [ ] **Step 1: Write the curation binary**

Create `src/bin/curate-corpus.rs`:

```rust
//! Curate a captured archive into the committed corpus subset + manifest.
//!
//! Usage: curate-corpus <archive_root> <out_corpus_dir> [--per-cell K] [--per-tag M]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use xdna_emu::corpus::curate::{select_subset, CurateParams};
use xdna_emu::corpus::schema::{
    CaseIndexEntry, CaseMeta, Class, Counts, Manifest, SCHEMA_VERSION,
};

fn load_metas(archive: &Path) -> Vec<(PathBuf, CaseMeta)> {
    let mut out = Vec::new();
    for class_dir in ["agree", "diverge"] {
        let base = archive.join(class_dir);
        let Ok(entries) = std::fs::read_dir(&base) else { continue };
        let mut dirs: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.join("meta.json").exists())
            .collect();
        dirs.sort();
        for d in dirs {
            if let Ok(s) = std::fs::read_to_string(d.join("meta.json")) {
                if let Ok(m) = serde_json::from_str::<CaseMeta>(&s) {
                    out.push((d, m));
                }
            }
        }
    }
    out
}

fn copy_case(src: &Path, dst_root: &Path, m: &CaseMeta) -> std::io::Result<String> {
    let class_dir = match m.class {
        Class::Agree => "agree",
        Class::Diverge => "diverge",
    };
    let name = src.file_name().unwrap().to_string_lossy().to_string();
    let rel = format!("{}/{}", class_dir, name);
    let dst = dst_root.join(&rel);
    std::fs::create_dir_all(&dst)?;
    for f in ["aie.xclbin", "npu_output.bin", "emu_output.bin", "meta.json"] {
        let s = src.join(f);
        if s.exists() {
            std::fs::copy(&s, dst.join(f))?;
        }
    }
    Ok(rel)
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: curate-corpus <archive_root> <out_corpus_dir> [--per-cell K] [--per-tag M]");
        std::process::exit(2);
    }
    let archive = PathBuf::from(&args[1]);
    let out = PathBuf::from(&args[2]);
    let mut p = CurateParams::default();
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--per-cell" => { p.per_cell = args[i + 1].parse().unwrap(); i += 2; }
            "--per-tag" => { p.per_tag = args[i + 1].parse().unwrap(); i += 2; }
            other => { eprintln!("unknown arg {other}"); std::process::exit(2); }
        }
    }

    let loaded = load_metas(&archive);
    let metas: Vec<CaseMeta> = loaded.iter().map(|(_, m)| m.clone()).collect();
    let selected = select_subset(&metas, p);

    let mut cell_coverage: BTreeMap<String, u32> = BTreeMap::new();
    let mut tag_coverage: BTreeMap<String, u32> = BTreeMap::new();
    let mut counts = Counts { total: 0, agree: 0, diverge: 0, vacuous_excluded: 0, error: 0 };
    let mut cases = Vec::new();

    for &idx in &selected {
        let (src, m) = &loaded[idx];
        let rel = copy_case(src, &out, m)?;
        counts.total += 1;
        match m.class {
            Class::Agree => counts.agree += 1,
            Class::Diverge => counts.diverge += 1,
        }
        *cell_coverage.entry(m.cell_id.clone()).or_insert(0) += 1;
        for t in &m.tags {
            *tag_coverage.entry(t.as_str().to_string()).or_insert(0) += 1;
        }
        cases.push(CaseIndexEntry { path: rel, class: m.class, tags: m.tags.clone() });
    }

    let toolchain = metas
        .iter()
        .find(|m| !m.capture.peano_commit.is_empty())
        .map(|m| m.capture.clone())
        .unwrap_or(xdna_emu::corpus::schema::Provenance {
            date: String::new(), peano_commit: String::new(), driver_version: String::new(),
            fw_version: String::new(), determinism_ok: false,
        });

    let manifest = Manifest {
        schema_version: SCHEMA_VERSION,
        campaign_id: out.file_name().unwrap().to_string_lossy().to_string(),
        capture_date: toolchain.date.clone(),
        counts: counts.clone(),
        cell_coverage: cell_coverage.clone(),
        tag_coverage: tag_coverage.clone(),
        toolchain,
        cases,
    };
    std::fs::write(out.join("manifest.json"), serde_json::to_string_pretty(&manifest)?)?;

    // coverage.md
    let mut md = String::from("# Phoenix AIE2 corpus coverage\n\n");
    md.push_str(&format!(
        "Total {} (agree {}, diverge {}).\n\n## Cells\n\n",
        counts.total, counts.agree, counts.diverge
    ));
    for (cell, n) in &cell_coverage {
        md.push_str(&format!("- `{}`: {}\n", cell, n));
    }
    md.push_str("\n## Tags\n\n");
    for (tag, n) in &tag_coverage {
        md.push_str(&format!("- `{}`: {}\n", tag, n));
    }
    std::fs::write(out.join("coverage.md"), md)?;

    println!("curated {} cases -> {}", counts.total, out.display());
    Ok(())
}
```

- [ ] **Step 2: Build the binary**

Run: `cargo build --bin curate-corpus`
Expected: compiles clean.

- [ ] **Step 3: Smoke-test curation against the smoke archive (if Task 10 produced one)**

```bash
cargo run --bin curate-corpus -- ~/npu-work/experiments/phoenix-aie2-golden-smoke tests/corpus/phoenix-aie2 --per-cell 1 --per-tag 2
ls tests/corpus/phoenix-aie2/manifest.json tests/corpus/phoenix-aie2/coverage.md
TMPDIR=/tmp/claude-1000 cargo test --test corpus_replay committed_corpus_replays_clean -- --nocapture
```
Expected: manifest.json + coverage.md written; gate replays the curated cases clean.

- [ ] **Step 4: Add the reclassify helper**

Create `scripts/reclassify-corpus-case.sh`:

```bash
#!/usr/bin/env bash
# Promote a fixed divergence: move a case from diverge/ to agree/ and flip its
# class. Run after a fix makes the diverge tripwire go red BY DESIGN.
# Usage: reclassify-corpus-case.sh <path-to-diverge-case-dir>
set -euo pipefail
SRC="$1"
[ -f "$SRC/meta.json" ] || { echo "no meta.json in $SRC"; exit 1; }
ROOT="$(cd "$SRC/../.."; pwd)"
NAME="$(basename "$SRC")"
DST="$ROOT/agree/$NAME"
mkdir -p "$DST"
# Drop emu_output.bin (no longer the reference); flip class to Agree.
for f in aie.xclbin npu_output.bin meta.json; do cp "$SRC/$f" "$DST/$f"; done
python3 - "$DST/meta.json" <<'PY'
import json,sys
p=sys.argv[1]; m=json.load(open(p)); m["class"]="Agree"
m.get("hashes",{}).pop("emu_output_sha256",None)
json.dump(m,open(p,"w"),indent=2)
PY
rm -rf "$SRC"
echo "reclassified $NAME diverge -> agree"
```

```bash
chmod +x scripts/reclassify-corpus-case.sh
```

- [ ] **Step 5: Commit**

```bash
git add src/bin/curate-corpus.rs scripts/reclassify-corpus-case.sh
git commit -m "corpus: curate-corpus binary (subset + manifest + coverage) + reclassify helper

Generated using Claude Code."
```

---

## Task 13: Run the capture campaign (operational, HW-gated)

**This task runs on live Phoenix and is performed by Maya (or with the NPU available), not in the sandbox.** It produces the real corpus the gate guards. It is the last step and depends on all prior tasks building clean.

- [ ] **Step 1: Full build**

```bash
cargo build --release --features tooling
cargo build --release --bin curate-corpus
```
Expected: clean.

- [ ] **Step 2: Run the grid capture campaign on Phoenix**

```bash
cargo run --release --features tooling -- fuzz \
  --capture ~/npu-work/experiments/phoenix-aie2-golden-$(date +%Y%m%d) \
  --capture-per-cell 12 --hw -j5
```
Expected: ~1,080 controlled cases captured under `agree/` + `diverge/`. Watch for HW wedging per repo ops rules.

- [ ] **Step 3: (If store_at_le is thin) overflow harvest**

```bash
XDNA_FUZZ_RECENCY1=1 cargo run --release --features tooling -- fuzz \
  --capture ~/npu-work/experiments/phoenix-aie2-golden-$(date +%Y%m%d) \
  --iterations 4000 --hw -j5
```
(Capture mode + random iterations is the overflow path; the `store_at_le` tag floor is satisfied from this harvest. NOTE: capture mode currently drives the grid; if `--iterations` with `--capture` is needed for overflow, ensure Task 10 Step 3 falls back to the random-seed generator when `--iterations` is given alongside `--capture`. Flag if not.)

- [ ] **Step 4: Curate into the committed corpus**

```bash
cargo run --release --bin curate-corpus -- \
  ~/npu-work/experiments/phoenix-aie2-golden-$(date +%Y%m%d) \
  tests/corpus/phoenix-aie2 --per-cell 4 --per-tag 8
```
Expected: `tests/corpus/phoenix-aie2/{agree,diverge,manifest.json,coverage.md}` populated, ~10-15 MB.

- [ ] **Step 5: Verify the gate passes on the real corpus, then commit**

```bash
TMPDIR=/tmp/claude-1000 cargo test --test corpus_replay -- --nocapture
```
Expected: `committed_corpus_replays_clean` PASS over all committed cases.

```bash
git add tests/corpus/phoenix-aie2
git commit -m "corpus: commit Phoenix AIE2 golden regression corpus (campaign $(date +%Y%m%d))

Generated using Claude Code."
```

- [ ] **Step 6: Archive the full capture**

Confirm the full archive remains under `~/npu-work/experiments/phoenix-aie2-golden-<date>/` (agree + diverge + aie.mlir sources + errors/ bucket + manifest). This is the durable superset the committed subset is drawn from.

---

## Deferred (named follow-ups, NOT in this plan)

- **Disasm tags (`store_at_le`, `has_load`):** AST tagging (Task 7) covers `has_hwloop`/`has_branch_taken`/`multi_store`. The two disasm-derived tags reuse the BUG-B classifier (`tools/classify_le_store.py`) as a post-capture pass that reads each case's compiled ELF and adds the tag to `meta.json` before curation. Until wired, the `store_at_le` tag floor is best-effort.
- **Tracks B/C/D** (trace corpus / HW-gated verifications / raw-breadth reservoir): separate specs per the design doc.
- **aiesim third-runtime / xclbin-boundary integration:** the proven aiesim oracle (`docs/aiesimulator.md`) is complementary; its larger role is a separate effort.

---

## Self-Review

**Spec coverage:**
- Section 1 (architecture: capture -> corpus -> curate -> gate): Tasks 8/10 (capture), 13 Step 6 (archive), 12 (curate), 11/13 (gate). Covered.
- Section 2 (90-cell grid + tagging): Task 9 (grid), Task 7 (tags). store_at_le/has_load deferred (flagged).
- Section 3 (schema/storage/durability): Tasks 2 (schema), 8 (per-case dir, no aie.mlir in committed), 12 (manifest/coverage, curated subset), hashes (Task 3) for durability. Covered.
- Section 4 (gate + diff + diverge->agree lifecycle): Tasks 6 (replay), 4 (diff), 11 (gate test + script), 12 Step 4 (reclassify helper). Covered.
- Section 5 (edges/testing/inventory): determinism gate (Task 10 pass branch), vacuous/error exclusion (Task 10), tooling tests (Tasks 2-9 inline), aiesim reference + tracks B/C/D (Deferred section). Covered.

**Placeholder scan:** No "TBD"/"add error handling"/uncoded steps. Two honest scope cuts are explicitly named (disasm tags; the campaign overflow `--iterations`+`--capture` interaction flagged in Task 13 Step 3).

**Type consistency:** `CaseMeta`/`Manifest`/`Class`/`Tag`/`OpBand`/`Dtype`/`LoopStyleMeta`/`Provenance`/`Hashes` defined in Task 2 and used identically in Tasks 5/6/8/12. `replay_case`/`replay_corpus` (Task 6) used by Task 11. `select_subset`/`CurateParams` (Task 5) used by Task 12. `capture_case`/`CaptureInput`/`build_meta` (Task 8) used by Task 10. `generate_grid_case`/`GridCell`/`grid_cells` (Task 9) used by Task 10/13. `detect_tags` (Task 7) used by Task 8. Consistent.

**Known integration risks (verify during execution, not placeholders):** Task 10's exact insertion depends on the live shape of `run_fuzz`'s generation loop and the per-case struct's xclbin path field; the plan names both and says how to adapt. Task 11's fixture minting is manual (documented).
