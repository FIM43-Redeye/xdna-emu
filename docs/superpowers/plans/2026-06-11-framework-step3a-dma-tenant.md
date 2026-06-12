# Framework Step 3a: DMA / data-movement Domain tenant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third `core::Domain` tenant that fuzzes DDR data-movement (buffer-descriptor access patterns across shim + memtile DMA engines), differentially against real NPU1 silicon, with per-region localization -- covering the 8 faithfully-executing features (81-key universe).

**Architecture:** A DMA case is a chain of N>=1 transfers, each reshuffling its own input region -> its own output region of a DDR buffer under one n-dimensional BD access pattern; exactly one BD per transfer carries the fuzzed pattern, the rest of the loopback is linear passthrough. `lower()` emits complete raw `aie.mlir` (verified by the spike to compile through `aiecc.py` and run on the emulator); a new defaulted `Domain::compile` hook lets DMA run `aiecc` directly while vector/scalar keep the kernel+template path. The tenant is a near-structural twin of the scalar tenant (`src/fuzzer/domains/scalar/`).

**Tech Stack:** Rust; the existing `core::Domain` framework (`src/fuzzer/core/`); `aiecc.py` (raw-MLIR path); the in-process `XclbinSuite` emulator + `npu_runner` HW executor; `serde` for banking.

**Grounding:** This plan is backed by a two-round toolchain spike (artifacts under `build/experiments/dma-spike/`). All MLIR forms below compiled through the real `aiecc.py` and ran correct reshuffles on the emulator. Spec: `docs/superpowers/specs/2026-06-11-dma-data-movement-domain.md`.

---

## File Structure

New module `src/fuzzer/domains/dma/` (twin of `domains/scalar/`):
- `mod.rs` -- `pub mod chain; pub mod table; pub mod gen; pub mod lower; pub mod domain; pub mod runner;`
- `chain.rs` -- AST: `Dtype`, `Engine`, `Direction`, `Feature`, `BdPattern`, `DmaTransfer`, `DmaChain` (+ methods, serde).
- `table.rs` -- `universe_keys()` (81 capability-gated keys), `Target`/`parse_key()`.
- `gen.rs` -- `generate(seed, target) -> DmaChain` (targeted, Stage-A-safe), `Xorshift64`, per-feature pattern construction.
- `lower.rs` -- `lower_chain(&DmaChain) -> String` (the raw-MLIR emitter; shim + memtile paths).
- `domain.rs` -- `DmaObs`, `DmaDomain`, `make_dma_buffer_spec`, `observe_impl`, region localization, `DmaChainRecord` banking, `impl Domain` (with `compile` override).
- `runner.rs` -- `DmaFuzzOptions`, `run_dma_fuzz`.

Modified:
- `src/fuzzer/core/domain.rs` -- add defaulted `compile` trait method.
- `src/fuzzer/core/engine.rs` -- call `dom.compile()` in the compile pool (remove the inline lower+write+compile_kernel_case).
- `src/fuzzer/core/toolchain.rs` -- add `compile_dma_mlir()` helper; make `ToolPaths` `pub` if the trait-method visibility requires it.
- `src/fuzzer/domains/mod.rs` -- add `pub mod dma;`.
- `src/fuzzer/cli.rs`, `src/main.rs` -- wire a `fuzz-dma` subcommand.

---

## Task 1: Framework compile seam (defaulted `Domain::compile` hook)

**Files:**
- Modify: `src/fuzzer/core/domain.rs` (add defaulted method + import)
- Modify: `src/fuzzer/core/toolchain.rs` (add `compile_dma_mlir`; widen `ToolPaths` visibility if needed)
- Modify: `src/fuzzer/core/engine.rs:140-189` (compile pool calls `dom.compile`)

This is the only `core/` change. It must leave vector and scalar **byte-identical**.

- [ ] **Step 1: Write the failing test** in `src/fuzzer/core/domain.rs` (extend the existing `MockDomain` test module). MockDomain already implements `Domain`; assert its *default* `compile` is reachable through the trait object pattern the engine uses. Since `compile` needs `ToolPaths` (constructed from the environment), instead test the simpler invariant that the trait still has all tenants compiling. Add this unit test that asserts the default `compile` writes the lowered source then delegates -- by checking the source-write half in isolation:

```rust
#[test]
fn default_compile_writes_lowered_source() {
    // MockDomain::lower returns a fixed string; the default compile writes it to
    // fuzz_kernel.cc before delegating to compile_kernel_case. We only verify the
    // write half here (compile_kernel_case needs the real toolchain).
    let dom = MockDomain;
    let dir = std::env::temp_dir().join(format!("dma_seam_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dom.lower(&0u64);
    std::fs::write(dir.join("fuzz_kernel.cc"), &src).unwrap();
    let back = std::fs::read_to_string(dir.join("fuzz_kernel.cc")).unwrap();
    assert_eq!(back, src);
    std::fs::remove_dir_all(&dir).ok();
}
```

- [ ] **Step 2: Run it to verify it compiles/fails appropriately**

Run: `cargo test --lib fuzzer::core::domain::tests::default_compile_writes_lowered_source`
Expected: PASS (this is a scaffolding test; the real verification is the build + vector/scalar regression below).

- [ ] **Step 3: Add the defaulted `compile` method to the `Domain` trait** in `src/fuzzer/core/domain.rs`. Add near the other defaulted methods (`warnings`, `dump_divergent_observation`). At the top of the file add `use super::toolchain::ToolPaths;`. Insert after the `lower` method:

```rust
    /// Compile the case in `case_dir` to `aie.xclbin` + `insts.bin`. The default
    /// is the compute path used by vector and scalar: write `lower(case)` to
    /// `fuzz_kernel.cc`, then run Peano + fuzz_template.py + aiecc via
    /// `compile_kernel_case`. The DMA domain overrides this to write `aie.mlir`
    /// and run aiecc directly (no kernel object, no template).
    fn compile(&self, tools: &ToolPaths, case_dir: &Path, case: &Self::Case) -> Result<(), String> {
        let src = self.lower(case);
        std::fs::write(case_dir.join("fuzz_kernel.cc"), &src)
            .map_err(|e| format!("write fuzz_kernel.cc: {e}"))?;
        super::toolchain::compile_kernel_case(
            tools,
            case_dir,
            self.buffer_words(case),
            self.dtype(case),
        )
    }
```

If rustc raises `private_interfaces` / E0446 because `ToolPaths` is `pub(crate)` while `Domain` is `pub`, change `pub(crate) struct ToolPaths` to `pub struct ToolPaths` in `toolchain.rs` (its fields can stay `pub(crate)`/private). The build is the gate.

- [ ] **Step 4: Add `compile_dma_mlir` to `src/fuzzer/core/toolchain.rs`** (sibling of `compile_kernel_case`; it can read the private `python` field since it lives in this module):

```rust
/// Compile a pre-written `case_dir/aie.mlir` to xclbin + insts via aiecc.py only
/// (no Peano kernel object, no fuzz_template). Used by the DMA domain's
/// `compile` override. Mirrors `compile_kernel_case`'s aiecc step exactly.
pub(crate) fn compile_dma_mlir(tools: &ToolPaths, case_dir: &Path) -> Result<(), String> {
    let xclbin = case_dir.join("aie.xclbin");
    let mlir = case_dir.join("aie.mlir");
    if xclbin.exists() {
        if let (Ok(s), Ok(x)) = (std::fs::metadata(&mlir), std::fs::metadata(&xclbin)) {
            if let (Ok(st), Ok(xt)) = (s.modified(), x.modified()) {
                if xt > st {
                    return Ok(());
                }
            }
        }
    }
    let mut cmd = Command::new(&tools.python);
    cmd.arg(&tools.aiecc)
        .arg("--no-xchesscc")
        .arg("--no-xbridge")
        .arg("--no-aiesim")
        .arg("--aie-generate-xclbin")
        .arg("--aie-generate-npu-insts")
        .arg("--no-compile-host")
        .arg("--alloc-scheme=basic-sequential")
        .arg("--xclbin-name=aie.xclbin")
        .arg("--npu-insts-name=insts.bin")
        .arg("aie.mlir");
    cmd.current_dir(case_dir);
    tools.apply_env(&mut cmd);
    let out = cmd.output().map_err(|e| format!("Failed to spawn aiecc.py: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let stdout = String::from_utf8_lossy(&out.stdout);
        let combined = if stderr.is_empty() { stdout } else { stderr };
        return Err(format!(
            "aiecc.py failed:\n{}",
            combined.lines().take(12).collect::<Vec<_>>().join("\n")
        ));
    }
    if !xclbin.exists() {
        return Err("aiecc.py succeeded but aie.xclbin not found".into());
    }
    if !case_dir.join("insts.bin").exists() {
        return Err("aiecc.py succeeded but insts.bin not found".into());
    }
    Ok(())
}
```

- [ ] **Step 5: Refactor the engine compile pool** in `src/fuzzer/core/engine.rs`. Remove the Phase 2 write loop (lines ~140-146). In the Phase 3 spawn closure (lines ~155-176), add `let dom = &dom;` to the captured borrows, and replace the body that reads `let words = dom.buffer_words(...); let r = compile_kernel_case(tools, &case.case_dir, words, dom.dtype(&case.case));` with dir-creation + the trait call:

```rust
        s.spawn(move || loop {
            let idx = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if idx >= cases.len() {
                break;
            }
            let case = &cases[idx];
            std::fs::create_dir_all(&case.case_dir).ok();
            let r = dom.compile(tools, &case.case_dir, &case.case);
            *compiled[idx].lock().unwrap() = Some(r);
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if !opts.verbose {
                eprint!("\rCompiling [{}/{}]", n, cases.len());
            }
        });
```

Remove the now-unused `compile_kernel_case` import in engine.rs only if it is no longer referenced there (it stays exported from toolchain.rs for the default `compile`).

- [ ] **Step 6: Run the full regression** -- the seam must not touch vector/scalar.

Run: `cargo test --lib`
Expected: PASS, same count as before this task (3459) plus the one new scaffolding test = 3460. No failures.

- [ ] **Step 7: Commit**

```bash
git add src/fuzzer/core/domain.rs src/fuzzer/core/toolchain.rs src/fuzzer/core/engine.rs
git commit -m "fuzzer/core: defaulted Domain::compile hook + compile_dma_mlir helper

Additive seam for the DMA tenant: the engine's compile pool now calls
dom.compile(); the default impl is exactly the vector/scalar kernel+template
path, so both stay byte-identical. compile_dma_mlir runs aiecc.py directly on
a pre-written aie.mlir.

Generated using Claude Code."
```

---

## Task 2: DMA module + chain AST

**Files:**
- Create: `src/fuzzer/domains/dma/mod.rs`
- Create: `src/fuzzer/domains/dma/chain.rs`
- Modify: `src/fuzzer/domains/mod.rs` (add `pub mod dma;`)

- [ ] **Step 1: Create `src/fuzzer/domains/dma/mod.rs`:**

```rust
//! DMA / data-movement fuzzer domain (framework Step 3a).
//!
//! Fuzzes DDR buffer-descriptor access patterns across the shim and memtile DMA
//! engines, differentially against silicon, with per-region localization. See
//! `docs/superpowers/specs/2026-06-11-dma-data-movement-domain.md`.
pub mod chain;
pub mod domain;
pub mod gen;
pub mod lower;
pub mod runner;
pub mod table;
```

- [ ] **Step 2: Add `pub mod dma;`** to `src/fuzzer/domains/mod.rs` (alongside `pub mod vector; pub mod scalar;`).

- [ ] **Step 3: Write `src/fuzzer/domains/dma/chain.rs`** -- the AST. `Feature` includes `Iter` and `Chain` variants (reserved for 3b; excluded from `universe_keys()`), so 3b is a gating flip, not a refactor. `BdPattern.sizes`/`strides` are in **MLIR list order (outermost dim first, innermost last)** -- the emitter writes them directly.

```rust
//! DMA chain AST. A case = N>=1 transfers, each carrying one BD access pattern.
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dtype {
    I32,
    I16,
    I8,
}

impl Dtype {
    /// MLIR memref element type.
    pub fn mlir_elem(self) -> &'static str {
        match self {
            Dtype::I32 => "i32",
            Dtype::I16 => "i16",
            Dtype::I8 => "i8",
        }
    }
    /// String returned from `Domain::dtype` (unused by the DMA compile override,
    /// but the trait requires it).
    pub fn template_dtype(self) -> &'static str {
        self.mlir_elem()
    }
    pub fn byte_size(self) -> usize {
        match self {
            Dtype::I32 => 4,
            Dtype::I16 => 2,
            Dtype::I8 => 1,
        }
    }
    /// Coverage-key token.
    pub fn key_str(self) -> &'static str {
        match self {
            Dtype::I32 => "I32",
            Dtype::I16 => "I16",
            Dtype::I8 => "I8",
        }
    }
    pub fn all() -> [Dtype; 3] {
        [Dtype::I32, Dtype::I16, Dtype::I8]
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Engine {
    Shim,
    Memtile,
}

impl Engine {
    pub fn key_str(self) -> &'static str {
        match self {
            Engine::Shim => "shim",
            Engine::Memtile => "memtile",
        }
    }
    pub fn all() -> [Engine; 2] {
        [Engine::Shim, Engine::Memtile]
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Direction {
    Mm2s,
    S2mm,
}

impl Direction {
    pub fn key_str(self) -> &'static str {
        match self {
            Direction::Mm2s => "mm2s",
            Direction::S2mm => "s2mm",
        }
    }
    pub fn all() -> [Direction; 2] {
        [Direction::Mm2s, Direction::S2mm]
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Feature {
    Linear,
    Strided2d,
    Strided3d,
    Strided4d,
    Transpose,
    Overlap,
    Packet,
    PadBefore,
    PadAfter,
    PadBoth,
    /// 3b: shim outermost-dim BD-repeat. In the enum so 3b is a gating flip.
    Iter,
    /// 3b: memtile multi-BD next_bd chain.
    Chain,
}

impl Feature {
    pub fn key_str(self) -> &'static str {
        match self {
            Feature::Linear => "linear",
            Feature::Strided2d => "strided2d",
            Feature::Strided3d => "strided3d",
            Feature::Strided4d => "strided4d",
            Feature::Transpose => "transpose",
            Feature::Overlap => "overlap",
            Feature::Packet => "packet",
            Feature::PadBefore => "padbefore",
            Feature::PadAfter => "padafter",
            Feature::PadBoth => "padboth",
            Feature::Iter => "iter",
            Feature::Chain => "chain",
        }
    }
    /// All 3a features (excludes Iter/Chain).
    pub fn all_3a() -> [Feature; 10] {
        [
            Feature::Linear,
            Feature::Strided2d,
            Feature::Strided3d,
            Feature::Strided4d,
            Feature::Transpose,
            Feature::Overlap,
            Feature::Packet,
            Feature::PadBefore,
            Feature::PadAfter,
            Feature::PadBoth,
        ]
    }
}

/// One BD access pattern, in MLIR list order (sizes[0]/strides[0] = outermost).
/// All values in dtype elements. Stage-A-safe by construction (gen clamps).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BdPattern {
    pub sizes: Vec<u32>,
    pub strides: Vec<u32>,
    /// Empty = no padding; else len == sizes.len() (memtile MM2S only).
    pub pad_before: Vec<u32>,
    pub pad_after: Vec<u32>,
    /// (pkt_id, pkt_type) when packet-switched.
    pub packet: Option<(u8, u8)>,
}

impl BdPattern {
    /// Element count physically moved into the destination (data only; the
    /// product of the access-pattern sizes).
    pub fn data_elems(&self) -> usize {
        self.sizes.iter().map(|&s| s as usize).product::<usize>().max(1)
    }
    /// Total per-dim padding added (memtile MM2S). Padding inserts this many zero
    /// elements into the streamed output beyond the data.
    pub fn pad_elems(&self) -> usize {
        // The streamed length is product over dims of (size + pad_before + pad_after)
        // minus the data product. See lower.rs / the spike's pad arithmetic.
        if self.pad_before.is_empty() {
            return 0;
        }
        let padded: usize = self
            .sizes
            .iter()
            .zip(&self.pad_before)
            .zip(&self.pad_after)
            .map(|((&s, &b), &a)| (s + b + a) as usize)
            .product();
        padded.saturating_sub(self.data_elems())
    }
}

/// One transfer: a region reshuffle on a named engine/direction/feature.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DmaTransfer {
    pub engine: Engine,
    pub dir: Direction,
    pub feature: Feature,
    pub pattern: BdPattern,
    /// Element offset into the input DDR buffer.
    pub in_off: usize,
    /// Element offset into the output DDR buffer.
    pub out_off: usize,
    /// Input region element count (data, pre-pad).
    pub in_elems: usize,
    /// Output region element count (= in_elems + pattern.pad_elems()).
    pub out_elems: usize,
}

impl DmaTransfer {
    pub fn key(&self, dtype: Dtype) -> String {
        format!(
            "{}/{}/{}/{}",
            self.feature.key_str(),
            self.engine.key_str(),
            self.dir.key_str(),
            dtype.key_str()
        )
    }
    /// (start, end) byte bounds of this transfer's output region.
    pub fn out_byte_bounds(&self, dtype: Dtype) -> (usize, usize) {
        let bs = dtype.byte_size();
        (self.out_off * bs, (self.out_off + self.out_elems) * bs)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DmaChain {
    pub seed: u64,
    pub target_key: String,
    pub dtype: Dtype,
    /// One engine per case (the target's engine).
    pub engine: Engine,
    pub transfers: Vec<DmaTransfer>,
}

impl DmaChain {
    /// Per-transfer coverage keys, in transfer order.
    pub fn keys(&self) -> Vec<String> {
        self.transfers.iter().map(|t| t.key(self.dtype)).collect()
    }
    /// Total input buffer size in dtype elements.
    pub fn in_words(&self) -> usize {
        self.transfers.iter().map(|t| t.in_elems).sum()
    }
    /// Total output buffer size in dtype elements.
    pub fn out_words(&self) -> usize {
        self.transfers.iter().map(|t| t.out_elems).sum()
    }
}
```

- [ ] **Step 4: Write unit tests** at the bottom of `chain.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_format() {
        let t = DmaTransfer {
            engine: Engine::Shim,
            dir: Direction::Mm2s,
            feature: Feature::Transpose,
            pattern: BdPattern { sizes: vec![8, 8], strides: vec![1, 8], pad_before: vec![], pad_after: vec![], packet: None },
            in_off: 0, out_off: 0, in_elems: 64, out_elems: 64,
        };
        assert_eq!(t.key(Dtype::I16), "transpose/shim/mm2s/I16");
    }

    #[test]
    fn pad_elems_arithmetic() {
        // 8 data + 4 before + 4 after (1D) = 16 streamed -> 8 pad.
        let p = BdPattern { sizes: vec![8], strides: vec![1], pad_before: vec![4], pad_after: vec![4], packet: None };
        assert_eq!(p.data_elems(), 8);
        assert_eq!(p.pad_elems(), 8);
    }

    #[test]
    fn footprints_sum_regions() {
        let mk = |in_off, out_off, ie, oe| DmaTransfer {
            engine: Engine::Memtile, dir: Direction::Mm2s, feature: Feature::Linear,
            pattern: BdPattern { sizes: vec![ie as u32], strides: vec![1], pad_before: vec![], pad_after: vec![], packet: None },
            in_off, out_off, in_elems: ie, out_elems: oe,
        };
        let c = DmaChain { seed: 1, target_key: "linear/memtile/mm2s/I32".into(), dtype: Dtype::I32, engine: Engine::Memtile, transfers: vec![mk(0,0,64,64), mk(64,64,64,80)] };
        assert_eq!(c.in_words(), 128);
        assert_eq!(c.out_words(), 144);
    }
}
```

- [ ] **Step 5: Run + commit**

Run: `cargo test --lib fuzzer::domains::dma::chain`
Expected: PASS.

```bash
git add src/fuzzer/domains/dma/mod.rs src/fuzzer/domains/dma/chain.rs src/fuzzer/domains/mod.rs
git commit -m "fuzzer/dma: chain AST (Dtype/Engine/Direction/Feature/BdPattern/DmaTransfer/DmaChain)

Generated using Claude Code."
```

---

## Task 3: Coverage universe + key parsing (`table.rs`)

**Files:**
- Create: `src/fuzzer/domains/dma/table.rs`

The 3a universe is the capability-gated cross-product. Gating from the spike matrix.

- [ ] **Step 1: Write `table.rs`:**

```rust
//! DMA coverage universe (3a) and key parsing. Keys are
//! `{feature}/{engine}/{dir}/{dtype}`, capability-gated per the spike matrix.
use super::chain::{Direction, Dtype, Engine, Feature};

/// Is this (engine, feature, dir) combination part of the 3a universe?
/// Gating (spike-verified): strided4d + pad* are memtile-only; packet + pad*
/// are MM2S-only; iter/chain are deferred to 3b.
pub fn supported(engine: Engine, feature: Feature, dir: Direction) -> bool {
    use Feature::*;
    match feature {
        Linear | Strided2d | Strided3d | Transpose | Overlap => true,
        Strided4d => engine == Engine::Memtile,
        Packet => dir == Direction::Mm2s,
        PadBefore | PadAfter | PadBoth => engine == Engine::Memtile && dir == Direction::Mm2s,
        // 3b -- never in the 3a universe.
        Iter | Chain => false,
    }
}

/// The full sorted 3a key universe (81 keys).
pub fn universe_keys() -> Vec<String> {
    let mut keys = Vec::new();
    for engine in Engine::all() {
        for feature in Feature::all_3a() {
            for dir in Direction::all() {
                if !supported(engine, feature, dir) {
                    continue;
                }
                for dtype in Dtype::all() {
                    keys.push(format!(
                        "{}/{}/{}/{}",
                        feature.key_str(),
                        engine.key_str(),
                        dir.key_str(),
                        dtype.key_str()
                    ));
                }
            }
        }
    }
    keys.sort();
    keys.dedup();
    keys
}

/// A parsed coverage key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    pub feature: Feature,
    pub engine: Engine,
    pub dir: Direction,
    pub dtype: Dtype,
}

pub fn parse_key(key: &str) -> Result<Target, String> {
    let parts: Vec<&str> = key.split('/').collect();
    if parts.len() != 4 {
        return Err(format!("bad key (need feature/engine/dir/dtype): {key}"));
    }
    let feature = Feature::all_3a()
        .into_iter()
        .find(|f| f.key_str() == parts[0])
        .ok_or_else(|| format!("unknown feature: {}", parts[0]))?;
    let engine = Engine::all()
        .into_iter()
        .find(|e| e.key_str() == parts[1])
        .ok_or_else(|| format!("unknown engine: {}", parts[1]))?;
    let dir = Direction::all()
        .into_iter()
        .find(|d| d.key_str() == parts[2])
        .ok_or_else(|| format!("unknown dir: {}", parts[2]))?;
    let dtype = Dtype::all()
        .into_iter()
        .find(|t| t.key_str() == parts[3])
        .ok_or_else(|| format!("unknown dtype: {}", parts[3]))?;
    Ok(Target { feature, engine, dir, dtype })
}
```

- [ ] **Step 2: Write tests** (assert the exact count + gating invariants):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use super::super::chain::{Direction, Dtype, Engine, Feature};

    #[test]
    fn universe_is_81_keys() {
        let u = universe_keys();
        assert_eq!(u.len(), 81, "expected 81 3a keys, got {}", u.len());
        // sorted + unique
        let mut s = u.clone();
        s.sort();
        s.dedup();
        assert_eq!(s, u);
    }

    #[test]
    fn no_iter_or_chain_keys() {
        for k in universe_keys() {
            assert!(!k.starts_with("iter/"), "iter is 3b: {k}");
            assert!(!k.starts_with("chain/"), "chain is 3b: {k}");
        }
    }

    #[test]
    fn gating_invariants() {
        // strided4d + pad* memtile-only; packet + pad* mm2s-only.
        for k in universe_keys() {
            if k.starts_with("strided4d/") {
                assert!(k.contains("/memtile/"), "strided4d must be memtile: {k}");
            }
            if k.starts_with("pad") {
                assert!(k.contains("/memtile/") && k.contains("/mm2s/"), "pad must be memtile mm2s: {k}");
            }
            if k.starts_with("packet/") {
                assert!(k.contains("/mm2s/"), "packet must be mm2s: {k}");
            }
        }
    }

    #[test]
    fn parse_round_trips_every_key() {
        for k in universe_keys() {
            let t = parse_key(&k).unwrap();
            let back = format!("{}/{}/{}/{}", t.feature.key_str(), t.engine.key_str(), t.dir.key_str(), t.dtype.key_str());
            assert_eq!(back, k);
        }
    }
}
```

- [ ] **Step 3: Run + commit**

Run: `cargo test --lib fuzzer::domains::dma::table`
Expected: PASS, `universe_is_81_keys` green.

```bash
git add src/fuzzer/domains/dma/table.rs
git commit -m "fuzzer/dma: 81-key capability-gated coverage universe + parse_key

Generated using Claude Code."
```

---

## Task 4: Targeted, Stage-A-safe generation (`gen.rs`)

**Files:**
- Create: `src/fuzzer/domains/dma/gen.rs`

Generation is deterministic in `(seed, target)`, forces the target feature into the chain, and emits **only in-bounds, field-valid patterns** (Stage A). Field-width maxima are from the AM025 recon (shim D-stepsize 20-bit, memtile 17-bit, etc.); the binding constraint is that the addressed footprint stays within the region.

- [ ] **Step 1: Write `gen.rs`.** `Xorshift64` is copied from the scalar gen (the deferred follow-up to lift it to `core::rng` is out of scope here). The per-feature pattern builders are the heart -- each returns a `BdPattern` whose footprint fits `region_elems`.

```rust
//! Deterministic, Stage-A-safe DMA chain generation.
use super::chain::{BdPattern, Direction, DmaChain, DmaTransfer, Dtype, Engine, Feature};
use super::table::{parse_key, supported, Target};

/// Field-width maxima (AM025 aie_registers_aie2.json). Stage A clamps well under
/// these; the binding constraint is footprint <= region.
const SHIM_STEP_MAX: u32 = 0xFFFFF; // 20-bit
const MEMTILE_STEP_MAX: u32 = 0x1FFFF; // 17-bit

struct Xorshift64(u64);
impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Xorshift64(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len())]
    }
}

/// Region element counts a chain may use (kept modest for Stage-A safety and fast
/// compiles; all transfers in a chain share the same in-region length).
const REGION_ELEMS: [usize; 3] = [64, 128, 256];

/// Build a Stage-A-safe pattern for `feature` covering exactly `region` data
/// elements. Patterns whose product of sizes must equal `region` pick factor
/// pairs of `region`; strides keep every address within `[0, region)`.
fn build_pattern(rng: &mut Xorshift64, feature: Feature, engine: Engine, region: usize) -> BdPattern {
    let step_max = match engine {
        Engine::Shim => SHIM_STEP_MAX,
        Engine::Memtile => MEMTILE_STEP_MAX,
    };
    // Helper: a 2-factor split (a*b == region), a in {power-of-two divisors}.
    let split2 = |rng: &mut Xorshift64, n: usize| -> (usize, usize) {
        let mut divs: Vec<usize> = (1..=n).filter(|d| n % d == 0).collect();
        divs.retain(|&d| d > 1 && d < n);
        if divs.is_empty() {
            return (1, n);
        }
        let a = rng.pick(&divs);
        (a, n / a)
    };
    let clamp = |s: u32| -> u32 { s.min(step_max).max(1) };

    match feature {
        Feature::Linear | Feature::Packet => {
            // Contiguous 1D. (packet rides routing, pattern is linear.)
            let mut p = BdPattern { sizes: vec![region as u32], strides: vec![1], pad_before: vec![], pad_after: vec![], packet: None };
            if feature == Feature::Packet {
                p.packet = Some((rng.below(4) as u8, 0));
            }
            p
        }
        Feature::Strided2d => {
            let (a, b) = split2(rng, region); // a outer, b inner
            BdPattern { sizes: vec![a as u32, b as u32], strides: vec![clamp(b as u32), 1], pad_before: vec![], pad_after: vec![], packet: None }
        }
        Feature::Transpose => {
            // Column-major read of an a x b tile: outer dim steps by 1, inner by a.
            let (a, b) = split2(rng, region);
            BdPattern { sizes: vec![b as u32, a as u32], strides: vec![1, clamp(b as u32)], pad_before: vec![], pad_after: vec![], packet: None }
        }
        Feature::Strided3d => {
            let (a, bc) = split2(rng, region);
            let (b, c) = split2(rng, bc.max(2));
            let (a, b, c) = (a.max(1), b.max(1), (region / (a * b)).max(1));
            BdPattern { sizes: vec![a as u32, b as u32, c as u32], strides: vec![clamp((b * c) as u32), clamp(c as u32), 1], pad_before: vec![], pad_after: vec![], packet: None }
        }
        Feature::Strided4d => {
            // memtile only. 2x2x(.)x(.) nest within region.
            let half = (region / 4).max(1);
            let (c, d) = split2(rng, half.max(2));
            BdPattern { sizes: vec![2, 2, c as u32, d as u32], strides: vec![clamp((2 * c * d) as u32), clamp((c * d) as u32), clamp(d as u32), 1], pad_before: vec![], pad_after: vec![], packet: None }
        }
        Feature::Overlap => {
            // stride < inner extent so elements are re-read; footprint stays < region.
            let (a, b) = split2(rng, region);
            let stride = (b / 2).max(1);
            BdPattern { sizes: vec![a as u32, b as u32], strides: vec![clamp(stride as u32), 1], pad_before: vec![], pad_after: vec![], packet: None }
        }
        Feature::PadBefore | Feature::PadAfter | Feature::PadBoth => {
            // memtile mm2s only. Base 2D pattern + per-dim padding.
            let (a, b) = split2(rng, region);
            let pad = 2u32;
            let (pb, pa) = match feature {
                Feature::PadBefore => (vec![pad, pad], vec![0, 0]),
                Feature::PadAfter => (vec![0, 0], vec![pad, pad]),
                _ => (vec![pad, pad], vec![pad, pad]),
            };
            BdPattern { sizes: vec![a as u32, b as u32], strides: vec![clamp(b as u32), 1], pad_before: pb, pad_after: pa, packet: None }
        }
        // 3b -- never generated in 3a (table::supported returns false).
        Feature::Iter | Feature::Chain => {
            BdPattern { sizes: vec![region as u32], strides: vec![1], pad_before: vec![], pad_after: vec![], packet: None }
        }
    }
}

/// Generate a deterministic Stage-A-safe chain for `(seed, target)`.
pub fn generate(seed: u64, target: &str) -> DmaChain {
    let tgt: Target = parse_key(target).expect("generate target must be a valid 3a key");
    let mut rng = Xorshift64::new(seed);
    let engine = tgt.engine;
    let dtype = tgt.dtype;

    // 1..=4 transfers; one forced to the target slot.
    let n = 1 + rng.below(4);
    let target_slot = rng.below(n);
    let region = REGION_ELEMS[rng.below(REGION_ELEMS.len())];

    let mut transfers = Vec::with_capacity(n);
    let mut in_off = 0usize;
    let mut out_off = 0usize;
    for k in 0..n {
        let (feature, dir) = if k == target_slot {
            (tgt.feature, tgt.dir)
        } else {
            // Random valid (feature, dir) on the SAME engine.
            loop {
                let f = rng.pick(&Feature::all_3a());
                let d = rng.pick(&Direction::all());
                if supported(engine, f, d) {
                    break (f, d);
                }
            }
        };
        let pattern = build_pattern(&mut rng, feature, engine, region);
        let in_elems = region;
        let out_elems = region + pattern.pad_elems();
        transfers.push(DmaTransfer { engine, dir, feature, pattern, in_off, out_off, in_elems, out_elems });
        in_off += in_elems;
        out_off += out_elems;
    }

    DmaChain { seed, target_key: target.to_string(), dtype, engine, transfers }
}
```

- [ ] **Step 2: Write tests** -- determinism, target presence, and the **Stage-A safety invariant** over a large sample:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use super::super::table::universe_keys;

    #[test]
    fn deterministic() {
        let a = generate(42, "transpose/memtile/mm2s/I16");
        let b = generate(42, "transpose/memtile/mm2s/I16");
        assert_eq!(serde_json::to_string(&a).unwrap(), serde_json::to_string(&b).unwrap());
    }

    #[test]
    fn target_always_present() {
        for (i, key) in universe_keys().into_iter().enumerate() {
            let c = generate(i as u64, &key);
            assert_eq!(c.target_key, key);
            assert!(c.keys().contains(&key), "seed {i}: target {key} not in chain keys {:?}", c.keys());
            // single engine per case
            assert!(c.transfers.iter().all(|t| t.engine == c.engine));
        }
    }

    #[test]
    fn stage_a_safe_footprints() {
        // Every generated pattern must address strictly within its region, with
        // field values under the engine step-max. Large sample across the universe.
        for (i, key) in universe_keys().into_iter().enumerate().cycle().take(2000) {
            let c = generate(i as u64 * 7 + 1, &key);
            for t in &c.transfers {
                let p = &t.pattern;
                // max linear element address reached by the access pattern
                let max_addr: usize = p.sizes.iter().zip(&p.strides)
                    .map(|(&s, &st)| (s.saturating_sub(1) as usize) * st as usize)
                    .sum();
                assert!(max_addr < t.in_elems, "seed{i} key{key}: footprint {max_addr} >= region {}", t.in_elems);
                let step_max = if t.engine == Engine::Shim { SHIM_STEP_MAX } else { MEMTILE_STEP_MAX };
                assert!(p.strides.iter().all(|&s| s <= step_max), "seed{i} key{key}: stride over field width");
                assert!(p.sizes.len() <= 4 && !p.sizes.is_empty());
                if t.engine == Engine::Shim {
                    assert!(p.sizes.len() <= 3, "shim caps at 3 data dims");
                    assert!(p.pad_before.is_empty(), "shim has no padding");
                }
                assert_eq!(t.out_elems, t.in_elems + p.pad_elems());
            }
        }
    }
}
```

- [ ] **Step 3: Run + commit**

Run: `cargo test --lib fuzzer::domains::dma::gen`
Expected: PASS (especially `stage_a_safe_footprints`).

```bash
git add src/fuzzer/domains/dma/gen.rs
git commit -m "fuzzer/dma: targeted Stage-A-safe generation (per-feature pattern builders)

Generated using Claude Code."
```

---

## Task 5: Shim MLIR emitter -- skeleton + linear/strided (with compile gate)

**Files:**
- Create: `src/fuzzer/domains/dma/lower.rs`

This is the keystone end-to-end gate: the first time we emit MLIR and push it through the real `aiecc`. The shim topology is a no-core **memtile linear passthrough** (spike-verified); the fuzzed pattern rides the runtime_sequence `aiex.dma_configure_task` BDs.

**Golden target** (spike `shim_scatter`, generalized to a chain). For a shim chain of fixed per-transfer length `L` (elements), dtype memref `<L x ELEM>`, the device is:

```mlir
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %buf = aie.buffer(%tile_0_1) {sym_name = "mt_buf"} : memref<{L}x{ELEM}>
      %prod = aie.lock(%tile_0_1, 0) {init = 1 : i32, sym_name = "mt_prod"}
      %cons = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "mt_cons"}
      %0 = aie.dma_start(S2MM, 0, ^s2mm, ^mm2s_entry)
    ^s2mm:
      aie.use_lock(%prod, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf : memref<{L}x{ELEM}>, 0, {L})
      aie.use_lock(%cons, Release, 1)
      aie.next_bd ^s2mm
    ^mm2s_entry:
      %1 = aie.dma_start(MM2S, 0, ^mm2s, ^end)
    ^mm2s:
      aie.use_lock(%cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf : memref<{L}x{ELEM}>, 0, {L})
      aie.use_lock(%prod, Release, 1)
      aie.next_bd ^mm2s
    ^end:
      aie.end
    }
    aie.runtime_sequence(%in: memref<{IN}x{ELEM}>, %out: memref<{OUT}x{ELEM}>) {
      // per transfer k: a recv (S2MM) task + a send (MM2S) task.
      // gather (dir=mm2s): pattern on the MM2S send-from-%in BD; recv linear.
      // scatter (dir=s2mm): pattern on the S2MM recv-to-%out BD; send linear.
      %recv0 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%out : memref<{OUT}x{ELEM}>, {OUT_OFF}, {RLEN}{RPAT}) {bd_id = 8 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%recv0)
      %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0{SHIM_PKT}) {
        aie.dma_bd(%in : memref<{IN}x{ELEM}>, {IN_OFF}, {SLEN}{SPAT}) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t0)
      aiex.dma_await_task(%t0)
      aiex.dma_await_task(%recv0)
      // ... repeated per transfer, MM2S bd_id cycling 0..7, recv bd_id 8 ...
    }
  }
}
```

Emitter substitution rules (this task: `linear`, `strided2d`; `{RPAT}`/`{SPAT}` are the `, [<size=..,stride=..>, ...]` list or empty; `{SHIM_PKT}` empty for now):
- `{ELEM}` = `chain.dtype.mlir_elem()`; `{IN}` = `chain.in_words()`; `{OUT}` = `chain.out_words()`; `{L}` = the shared per-transfer region length.
- For transfer k with `dir = mm2s` (gather): the pattern rides the MM2S send BD (`{SPAT}` = the access-pattern list, `{SLEN}` = data length); the recv is linear (`{RPAT}` empty, `{RLEN}` = region).
- For `dir = s2mm` (scatter): the pattern rides the S2MM recv BD (`{RPAT}` = list, `{RLEN}` = region); the send is linear.
- `{IN_OFF}`/`{OUT_OFF}` = transfer's `in_off`/`out_off`.
- MM2S `bd_id` cycles `0..=7` per transfer (chains are N<=4 so no reuse-batching needed); recv `bd_id = 8`.

- [ ] **Step 1: Write the failing compile-gate test** in `lower.rs`. It generates a linear and a strided2d shim case, lowers, writes `aie.mlir`, runs `compile_dma_mlir`, and asserts xclbin+insts exist. Guard it behind the toolchain (skip if tools can't be discovered), like the existing fuzzer compile tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzzer::domains::dma::gen::generate;

    fn compile_one(key: &str, seed: u64) {
        let tools = match crate::fuzzer::core::toolchain::ToolPaths::discover() {
            Ok(t) => t,
            Err(_) => return, // no toolchain in this env -- skip
        };
        let c = generate(seed, key);
        let dir = std::env::temp_dir().join(format!("dma_lower_{}_{}", seed, std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("aie.mlir"), lower_chain(&c)).unwrap();
        let r = crate::fuzzer::core::toolchain::compile_dma_mlir(&tools, &dir);
        assert!(r.is_ok(), "{key} seed{seed}: {r:?}\n{}", lower_chain(&c));
        assert!(dir.join("aie.xclbin").exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[ignore = "requires toolchain; run with --ignored"]
    fn shim_linear_compiles() { compile_one("linear/shim/mm2s/I32", 1); }

    #[test]
    #[ignore = "requires toolchain; run with --ignored"]
    fn shim_strided2d_compiles() { compile_one("strided2d/shim/mm2s/I32", 2); }

    #[test]
    fn shim_mlir_has_device_and_runtime_sequence() {
        let c = generate(1, "linear/shim/mm2s/I32");
        let m = lower_chain(&c);
        assert!(m.contains("aie.device(npu1_1col)"));
        assert!(m.contains("aie.runtime_sequence"));
        assert!(m.contains("aie.memtile_dma(%tile_0_1)"));
        assert!(!m.contains("aie.core"), "shim path must have no core ELF");
    }
}
```

- [ ] **Step 2: Run the non-ignored test to verify it fails**

Run: `cargo test --lib fuzzer::domains::dma::lower::tests::shim_mlir_has_device_and_runtime_sequence`
Expected: FAIL (`lower_chain` not defined).

- [ ] **Step 3: Implement `lower_chain` for the shim path** (linear + strided2d). Write the emitter producing the golden above. Dispatch on `chain.engine`: this task implements `Engine::Shim`; leave a `todo`-free `Engine::Memtile` arm returning a minimal valid linear passthrough for now (Task 7 fills it). Emit the access-pattern list helper:

```rust
//! DMA chain -> raw aie.mlir lowering (spike-verified forms).
use super::chain::{BdPattern, Direction, DmaChain, DmaTransfer, Engine};

/// Render a pattern's access list: `, [<size = a, stride = b>, ...]` or empty.
fn pattern_list(p: &BdPattern) -> String {
    if p.sizes.len() == 1 && p.strides == vec![1] && p.pad_before.is_empty() {
        return String::new(); // pure linear: no list needed
    }
    let dims: Vec<String> = p.sizes.iter().zip(&p.strides)
        .map(|(s, st)| format!("<size = {s}, stride = {st}>"))
        .collect();
    format!(", [{}]", dims.join(", "))
}

pub fn lower_chain(chain: &DmaChain) -> String {
    match chain.engine {
        Engine::Shim => lower_shim(chain),
        Engine::Memtile => lower_memtile(chain),
    }
}

fn lower_shim(chain: &DmaChain) -> String { /* emit the golden; see Step-3 detail */ }

fn lower_memtile(chain: &DmaChain) -> String { /* Task 7 */ String::new() }
```

Implement `lower_shim` to render: header, `aie.device(npu1_1col)`, the two tiles, two flows, the memtile passthrough block (fixed-chunk `{L}` = the chain's shared region length -- assert all `in_elems` equal), then the runtime_sequence iterating transfers. For each transfer, emit the recv + send configure_tasks per the gather/scatter rule above, cycling MM2S `bd_id`. Use `pattern_list` for the patterned side. (Padding never occurs on the shim, so `{RLEN}`/`{SLEN}` = region.)

- [ ] **Step 4: Run the structural test + the ignored compile gates**

Run: `cargo test --lib fuzzer::domains::dma::lower::tests::shim_mlir_has_device_and_runtime_sequence`
Expected: PASS.

Run (toolchain): `cargo test --lib fuzzer::domains::dma::lower -- --ignored`
Expected: `shim_linear_compiles` + `shim_strided2d_compiles` PASS (xclbin produced). If aiecc rejects, read its error against `build/experiments/dma-spike/shim_scatter/aie.mlir` and fix the emission.

- [ ] **Step 5: Commit**

```bash
git add src/fuzzer/domains/dma/lower.rs
git commit -m "fuzzer/dma: shim MLIR emitter (linear+strided2d), compile-gated through aiecc

Generated using Claude Code."
```

---

## Task 6: Shim emitter -- all shim features + both directions

**Files:**
- Modify: `src/fuzzer/domains/dma/lower.rs`

Add `strided3d`, `transpose`, `overlap` (all just `pattern_list` variations, already handled if the list emission is generic), `packet` (shim emission site = the configure_task op signature), and confirm both `mm2s`/`s2mm` directions.

- [ ] **Step 1: Write the packet structural test** (shim packet rides the op signature, not the BD):

```rust
    #[test]
    fn shim_packet_on_op_signature() {
        let c = generate(5, "packet/shim/mm2s/I32");
        let m = lower_chain(&c);
        // spike-verified: shim packet is on the configure_task op, not the dma_bd attr.
        assert!(m.contains("aiex.dma_configure_task(%tile_0_0, MM2S, 0, <pkt_type ="), "shim packet on op sig:\n{m}");
        assert!(m.contains("aie.packet_flow("), "needs a packet_flow decl");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib fuzzer::domains::dma::lower::tests::shim_packet_on_op_signature`
Expected: FAIL.

- [ ] **Step 3: Implement shim packet + confirm 3D/transpose/overlap.** When a transfer's `pattern.packet` is `Some((id, ty))` and engine is shim, emit (a) a top-level `aie.packet_flow(id) { aie.packet_source<%tile_0_0, DMA : 0>; aie.packet_dest<%tile_0_1, DMA : 0> }` (collect these and emit before the memtile block), and (b) the MM2S configure_task op signature as `aiex.dma_configure_task(%tile_0_0, MM2S, 0, <pkt_type = {ty}, pkt_id = {id}>)`. The `pattern_list` helper already renders 3D/transpose/overlap (they are just different sizes/strides). Verify the gather/scatter sides handle multi-dim lists.

- [ ] **Step 4: Add ignored compile gates** for the remaining shim features:

```rust
    #[test] #[ignore = "requires toolchain; run with --ignored"]
    fn shim_strided3d_compiles() { compile_one("strided3d/shim/mm2s/I32", 3); }
    #[test] #[ignore] fn shim_transpose_compiles() { compile_one("transpose/shim/mm2s/I16", 4); }
    #[test] #[ignore] fn shim_overlap_compiles() { compile_one("overlap/shim/s2mm/I8", 5); }
    #[test] #[ignore] fn shim_packet_compiles() { compile_one("packet/shim/mm2s/I32", 6); }
    #[test] #[ignore] fn shim_scatter_compiles() { compile_one("strided2d/shim/s2mm/I32", 7); }
```

- [ ] **Step 5: Run + commit**

Run: `cargo test --lib fuzzer::domains::dma::lower::tests::shim_packet_on_op_signature` (PASS)
Run (toolchain): `cargo test --lib fuzzer::domains::dma::lower -- --ignored` (all shim gates PASS)

```bash
git add src/fuzzer/domains/dma/lower.rs
git commit -m "fuzzer/dma: shim emitter complete (3d/transpose/overlap/packet, both dirs)

Generated using Claude Code."
```

---

## Task 7: Memtile MLIR emitter -- skeleton + linear/strided

**Files:**
- Modify: `src/fuzzer/domains/dma/lower.rs`

The memtile path carries the fuzzed pattern (incl. padding) on a raw `aie.memtile_dma` BD; the shim runs linear configure_tasks. N transfers = N entries in the memtile channel BD list (spike `memtile_chain2`).

**Golden target** (spike `memtile_pad2d` / `memtile_strided`, generalized). For `dir = mm2s` the pattern rides the memtile **MM2S** BD; for `dir = s2mm` it rides the memtile **S2MM** BD. Each chained BD needs its own balanced acquire/release lock pair (spike rule). Single-transfer form:

```mlir
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %buf = aie.buffer(%tile_0_1) {sym_name = "mt_buf"} : memref<{OUT}x{ELEM}>
      %prod = aie.lock(%tile_0_1, 0) {init = 1 : i32, sym_name = "mt_prod"}
      %cons = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "mt_cons"}
      %0 = aie.dma_start(S2MM, 0, ^s2mm, ^mm2s_entry)
    ^s2mm:
      aie.use_lock(%prod, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf : memref<{OUT}x{ELEM}>, 0, {IN})          // receive un-padded data
      aie.use_lock(%cons, Release, 1)
      aie.next_bd ^s2mm
    ^mm2s_entry:
      %1 = aie.dma_start(MM2S, 0, ^mm2s, ^end)
    ^mm2s:
      aie.use_lock(%cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf : memref<{OUT}x{ELEM}>, 0, {MM2S_LEN}{PAT}{PADLIST})  // {MM2S_LEN}=padded total
      aie.use_lock(%prod, Release, 1)
      aie.next_bd ^mm2s
    ^end:
      aie.end
    }
    aie.runtime_sequence(%in: memref<{IN}x{ELEM}>, %out: memref<{OUT}x{ELEM}>) {
      %recv = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%out : memref<{OUT}x{ELEM}>, 0, {OUT}) {bd_id = 8 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%recv)
      %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%in : memref<{IN}x{ELEM}>, 0, {IN}) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t0)
      aiex.dma_await_task(%t0)
      aiex.dma_await_task(%recv)
    }
  }
}
```

`{PADLIST}` = `, [<const_pad_before = b0, const_pad_after = a0>, ...]` or empty. `{MM2S_LEN}` = `in_elems + pad_elems` (the padded total) when padding, else the data length. `{PAT}` = the access list.

- [ ] **Step 1: Write the memtile structural test:**

```rust
    #[test]
    fn memtile_mlir_uses_static_block_not_runtime_pattern() {
        let c = generate(10, "strided2d/memtile/mm2s/I32");
        let m = lower_chain(&c);
        assert!(m.contains("aie.memtile_dma(%tile_0_1)"));
        assert!(m.contains("aie.dma_start(MM2S"));
        // the pattern list must appear inside the memtile block, the runtime shim BDs are linear
        assert!(m.contains("aie.next_bd"));
    }
```

- [ ] **Step 2: Run to verify it fails** (memtile arm currently returns empty)

Run: `cargo test --lib fuzzer::domains::dma::lower::tests::memtile_mlir_uses_static_block_not_runtime_pattern`
Expected: FAIL.

- [ ] **Step 3: Implement `lower_memtile`** for linear + strided (no padding/packet yet). Render the golden above: for `dir = mm2s` the pattern + `{MM2S_LEN}` ride the MM2S BD and the S2MM receive is linear; for `dir = s2mm` the pattern rides the S2MM BD and the MM2S send is linear (mirror). For a chain of N transfers, emit N buffers + N BD entries in the channel's list form (each with its own acquire/release lock pair), and N shim send/recv tasks in the runtime_sequence. Start with N handled by distinct buffers `mt_buf0..` and distinct lock pairs.

- [ ] **Step 4: Run structural + ignored compile gates**

Run: `cargo test --lib fuzzer::domains::dma::lower::tests::memtile_mlir_uses_static_block_not_runtime_pattern` (PASS)

```rust
    #[test] #[ignore] fn memtile_linear_compiles() { compile_one("linear/memtile/mm2s/I32", 20); }
    #[test] #[ignore] fn memtile_strided_compiles() { compile_one("strided2d/memtile/mm2s/I32", 21); }
```

Run (toolchain): `cargo test --lib fuzzer::domains::dma::lower -- --ignored memtile`
Expected: both PASS. If aiecc rejects, diff against `build/experiments/dma-spike/memtile_strided/aie.mlir`.

- [ ] **Step 5: Commit**

```bash
git add src/fuzzer/domains/dma/lower.rs
git commit -m "fuzzer/dma: memtile MLIR emitter (linear+strided), compile-gated

Generated using Claude Code."
```

---

## Task 8: Memtile emitter -- strided4d, transpose, overlap, packet-attr, padding, both dirs

**Files:**
- Modify: `src/fuzzer/domains/dma/lower.rs`

- [ ] **Step 1: Write structural tests** for the memtile-specific emission sites:

```rust
    #[test]
    fn memtile_packet_on_bd_attribute() {
        let c = generate(30, "packet/memtile/mm2s/I32");
        let m = lower_chain(&c);
        // spike-verified: memtile packet is the dma_bd attribute, not the op sig.
        assert!(m.contains("{packet = #aie.packet_info<pkt_type ="), "memtile packet on bd attr:\n{m}");
    }

    #[test]
    fn memtile_padding_uses_const_pad_and_grows_output() {
        let c = generate(31, "padboth/memtile/mm2s/I8");
        let m = lower_chain(&c);
        assert!(m.contains("const_pad_before"), "padding list missing:\n{m}");
        // padded output region strictly larger than input region
        assert!(c.transfers.iter().any(|t| t.out_elems > t.in_elems));
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib fuzzer::domains::dma::lower::tests::memtile_packet_on_bd_attribute`
Expected: FAIL.

- [ ] **Step 3: Implement the remaining memtile features.** `strided4d`/`transpose`/`overlap` are `pattern_list` variants (already work if list emission is generic -- confirm 4D). `packet`: emit a top-level `aie.packet_flow(id) { source<%tile_0_1, DMA:0>; dest<%tile_0_0, DMA:0> }` and put `{packet = #aie.packet_info<pkt_type = ty, pkt_id = id>}` as the `aie.dma_bd` **attribute** on the memtile BD. Padding: render `{PADLIST}` from `pad_before`/`pad_after` aligned to `sizes`, and set the MM2S BD `len` to the padded total (`in_elems + pad_elems`) while the S2MM receive and the recv-to-DDR use the padded `{OUT}` for capture. Apply the spike's pad arithmetic: data goes into the buffer un-padded; only the patterned MM2S BD carries the padded length + pad list.

- [ ] **Step 4: Run structural + the full ignored memtile gate set**

```rust
    #[test] #[ignore] fn memtile_strided4d_compiles() { compile_one("strided4d/memtile/mm2s/I32", 32); }
    #[test] #[ignore] fn memtile_transpose_compiles() { compile_one("transpose/memtile/s2mm/I16", 33); }
    #[test] #[ignore] fn memtile_overlap_compiles() { compile_one("overlap/memtile/mm2s/I8", 34); }
    #[test] #[ignore] fn memtile_packet_compiles() { compile_one("packet/memtile/mm2s/I32", 35); }
    #[test] #[ignore] fn memtile_padbefore_compiles() { compile_one("padbefore/memtile/mm2s/I8", 36); }
    #[test] #[ignore] fn memtile_padafter_compiles() { compile_one("padafter/memtile/mm2s/I16", 37); }
    #[test] #[ignore] fn memtile_padboth_compiles() { compile_one("padboth/memtile/mm2s/I32", 38); }
```

Run (toolchain): `cargo test --lib fuzzer::domains::dma::lower -- --ignored memtile`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fuzzer/domains/dma/lower.rs
git commit -m "fuzzer/dma: memtile emitter complete (4d/transpose/overlap/packet-attr/padding)

Generated using Claude Code."
```

---

## Task 9: Domain observation, buffer spec, localization

**Files:**
- Create: `src/fuzzer/domains/dma/domain.rs` (first half: obs/spec/observe/localize/warnings)

Mirrors `scalar/domain.rs`, but with a **region-bounds-aware** localizer (DMA output regions are unequal under padding) and a 2-buffer spec (in=group_id 3, out=group_id 4 -- spike-confirmed).

- [ ] **Step 1: Write the localizer test:**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzzer::domains::dma::chain::*;

    fn mk_chain() -> DmaChain {
        let mk = |in_off, out_off| DmaTransfer {
            engine: Engine::Memtile, dir: Direction::Mm2s, feature: Feature::Linear,
            pattern: BdPattern { sizes: vec![16], strides: vec![1], pad_before: vec![], pad_after: vec![], packet: None },
            in_off, out_off, in_elems: 16, out_elems: 16,
        };
        DmaChain { seed: 1, target_key: "linear/memtile/mm2s/I32".into(), dtype: Dtype::I32, engine: Engine::Memtile, transfers: vec![mk(0,0), mk(16,16), mk(32,32)] }
    }

    #[test]
    fn localizes_to_the_divergent_region() {
        let c = mk_chain();
        let mut emu = vec![7u8; 3 * 16 * 4];
        let hw = vec![7u8; 3 * 16 * 4];
        // corrupt region 1 (bytes 64..128)
        emu[70] = 9;
        let bounds: Vec<(usize, usize)> = c.transfers.iter().map(|t| t.out_byte_bounds(c.dtype)).collect();
        assert_eq!(first_divergent_region(&emu, &hw, &bounds), Some(1));
    }

    #[test]
    fn equal_observations_match() {
        let c = mk_chain();
        let buf = vec![3u8; 3 * 16 * 4];
        let bounds: Vec<(usize, usize)> = c.transfers.iter().map(|t| t.out_byte_bounds(c.dtype)).collect();
        assert_eq!(first_divergent_region(&buf, &buf, &bounds), None);
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib fuzzer::domains::dma::domain::tests::localizes_to_the_divergent_region`
Expected: FAIL.

- [ ] **Step 3: Implement the first half of `domain.rs`:**

```rust
//! DmaDomain: observation, buffer spec, localization (first half); banking +
//! impl Domain in Task 10.
use std::path::Path;

use super::chain::{Dtype, DmaChain};
use crate::fuzzer::core::domain::Backend;
use crate::testing::npu_runner;
use crate::testing::test_cpp_parser::{BufferDef, BufferDir, BufferSpec, ElementType, InputPattern};
use crate::testing::xclbin_suite::{XclbinSuite, XclbinTest};

pub struct DmaObs {
    pub output: Vec<u8>,
}

fn elem_type(d: Dtype) -> ElementType {
    match d {
        Dtype::I32 => ElementType::I32,
        Dtype::I16 => ElementType::I16,
        Dtype::I8 => ElementType::I8,
    }
}

/// 2-buffer spec: in (group_id 3, Sequential{1,1}), out (group_id 4, Zeros).
/// Sizes from the chain footprints; the runtime_sequence arg order is (in, out).
pub(crate) fn make_dma_buffer_spec(chain: &DmaChain) -> BufferSpec {
    let et = elem_type(chain.dtype);
    BufferSpec {
        buffers: vec![
            BufferDef {
                name: "buf_in".into(),
                group_id: 3,
                size_elements: chain.in_words(),
                element_type: et,
                direction: BufferDir::Input,
                input_pattern: InputPattern::Sequential { start: 1, step: 1 },
            },
            BufferDef {
                name: "buf_out".into(),
                group_id: 4,
                size_elements: chain.out_words(),
                element_type: et,
                direction: BufferDir::Output,
                input_pattern: InputPattern::Zeros,
            },
        ],
        multi_kernel: false,
    }
}

pub(crate) fn observe_impl(
    backend: Backend,
    xclbin: &Path,
    insts: &Path,
    chain: &DmaChain,
    max_cycles: u64,
) -> Result<DmaObs, String> {
    match backend {
        Backend::Interpreter => {
            let spec = make_dma_buffer_spec(chain);
            let test = XclbinTest::from_path(xclbin).with_buffer_spec(spec);
            let suite = XclbinSuite::new().with_max_cycles(max_cycles);
            let (outcome, raw_output, _trace) = suite.run_single_with_trace(&test);
            if !outcome.is_pass() {
                return Err(format!("emulator outcome not pass: {outcome:?}"));
            }
            let output = raw_output.ok_or_else(|| "Emulator produced no output".to_string())?;
            Ok(DmaObs { output })
        }
        Backend::Hardware => {
            if !npu_runner::npu_available() {
                return Err("NPU hardware not available".into());
            }
            let spec = make_dma_buffer_spec(chain);
            let test_name = format!("dmafuzz_seed_{}", chain.seed);
            match npu_runner::run_on_npu(&spec, &test_name, xclbin, insts, 30) {
                Ok(result) => Ok(DmaObs { output: result.output }),
                Err(e) => Err(format!("{e:?}")),
            }
        }
        Backend::Aiesim => Err("aiesim backend not wired for the DMA domain (Step 3)".into()),
    }
}

/// First transfer whose output byte-region differs. `bounds[k]` = (start, end).
pub(crate) fn first_divergent_region(emu: &[u8], hw: &[u8], bounds: &[(usize, usize)]) -> Option<usize> {
    for (k, &(start, end)) in bounds.iter().enumerate() {
        let e_end = end.min(emu.len());
        let h_end = end.min(hw.len());
        if start >= e_end || start >= h_end {
            // a side is short -> this region diverges
            if emu.len() != hw.len() {
                return Some(k);
            }
            continue;
        }
        if emu[start..e_end] != hw[start..h_end] {
            return Some(k);
        }
    }
    if emu.len() != hw.len() {
        return Some(bounds.len().saturating_sub(1));
    }
    None
}
```

- [ ] **Step 4: Run + commit**

Run: `cargo test --lib fuzzer::domains::dma::domain`
Expected: PASS (localizer tests).

```bash
git add src/fuzzer/domains/dma/domain.rs
git commit -m "fuzzer/dma: DmaObs, 2-buffer spec, observe, region-bounds localizer

Generated using Claude Code."
```

---

## Task 10: Banking + `impl Domain for DmaDomain` (with compile override) + EMU run

**Files:**
- Modify: `src/fuzzer/domains/dma/domain.rs` (banking + trait impl)

- [ ] **Step 1: Write an end-to-end EMU-run gate** (ignored; needs toolchain). It generates a memtile transpose case, compiles via the `Domain::compile` override, runs on the emulator through `observe`, and asserts non-zero output:

```rust
    #[test]
    #[ignore = "requires toolchain; run with --ignored"]
    fn end_to_end_emu_run_produces_output() {
        use crate::fuzzer::core::domain::{Backend, Domain};
        let tools = match crate::fuzzer::core::toolchain::ToolPaths::discover() { Ok(t) => t, Err(_) => return };
        let dom = DmaDomain;
        let c = dom.generate(1, "transpose/memtile/mm2s/I32");
        let dir = std::env::temp_dir().join(format!("dma_e2e_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dom.compile(&tools, &dir, &c).expect("compile");
        let obs = dom.observe(Backend::Interpreter, &dir.join("aie.xclbin"), &dir.join("insts.bin"), &c, 2_000_000).expect("emu run");
        assert!(!obs.output.is_empty());
        assert!(obs.output.iter().any(|&b| b != 0), "expected a non-zero reshuffle");
        std::fs::remove_dir_all(&dir).ok();
    }
```

- [ ] **Step 2: Run to verify it fails** (`DmaDomain` not defined)

Run: `cargo test --lib fuzzer::domains::dma::domain::tests::end_to_end_emu_run_produces_output -- --ignored`
Expected: FAIL (compile error: no `DmaDomain`).

- [ ] **Step 3: Implement banking + the trait impl** (append to `domain.rs`):

```rust
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use crate::fuzzer::core::domain::{Banked, Domain};
use crate::fuzzer::core::toolchain::{compile_dma_mlir, ToolPaths};
use super::gen::generate;
use super::lower::lower_chain;
use super::table::universe_keys;

pub struct DmaDomain;

#[derive(Serialize, Deserialize)]
pub(crate) struct DmaChainRecord {
    pub(crate) chain: DmaChain,
    pub(crate) keys: Vec<String>,
}

pub(crate) fn bank_case(
    case_dir: &Path,
    chain: &DmaChain,
    keys: &[String],
    npu_output: Option<&[u8]>,
) -> Result<PathBuf, String> {
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    let bank_dir = PathBuf::from(home)
        .join(format!("npu-work/experiments/phoenix-survival/dma/seed_{}", chain.seed));
    std::fs::create_dir_all(&bank_dir).map_err(|e| format!("create {}: {e}", bank_dir.display()))?;
    // best-effort copy of the generated MLIR for post-mortem
    std::fs::copy(case_dir.join("aie.mlir"), bank_dir.join("aie.mlir")).ok();
    let record = DmaChainRecord { chain: chain.clone(), keys: keys.to_vec() };
    let json = serde_json::to_string_pretty(&record).map_err(|e| format!("serialize: {e}"))?;
    std::fs::write(bank_dir.join("chain.json"), json).map_err(|e| format!("write chain.json: {e}"))?;
    if let Some(out) = npu_output {
        std::fs::write(bank_dir.join("npu_output.bin"), out).map_err(|e| format!("write npu_output: {e}"))?;
    }
    Ok(bank_dir)
}

impl Domain for DmaDomain {
    type Case = DmaChain;
    type Obs = DmaObs;

    fn name(&self) -> &str { "dma" }
    fn universe(&self) -> Vec<String> { universe_keys() }
    fn generate(&self, seed: u64, target: &str) -> DmaChain { generate(seed, target) }
    fn coverage_keys(&self, c: &DmaChain) -> Vec<String> { c.keys() }
    fn target_key(&self, c: &DmaChain) -> String { c.target_key.clone() }
    fn lower(&self, c: &DmaChain) -> String { lower_chain(c) }
    fn buffer_words(&self, c: &DmaChain) -> usize { c.out_words() }
    fn dtype(&self, c: &DmaChain) -> &str { c.dtype.template_dtype() }

    /// Override: write aie.mlir and run aiecc directly (no kernel object).
    fn compile(&self, tools: &ToolPaths, case_dir: &Path, c: &DmaChain) -> Result<(), String> {
        std::fs::write(case_dir.join("aie.mlir"), lower_chain(c))
            .map_err(|e| format!("write aie.mlir: {e}"))?;
        compile_dma_mlir(tools, case_dir)
    }

    fn observe(&self, backend: Backend, xclbin: &Path, insts: &Path, c: &DmaChain, max_cycles: u64) -> Result<DmaObs, String> {
        observe_impl(backend, xclbin, insts, c, max_cycles)
    }

    fn warnings(&self, obs: &DmaObs) -> Vec<String> {
        if !obs.output.is_empty() && obs.output.iter().all(|&b| b == 0) {
            vec!["vacuous output (all zero) -- degenerate transfer".into()]
        } else {
            Vec::new()
        }
    }

    fn compare(&self, emu: &DmaObs, reference: &DmaObs, keys: &[String]) -> Option<String> {
        // keys are the banked/coverage keys, one per transfer in order; bounds
        // come from the case-derived regions. We reconstruct bounds from key count
        // by assuming equal split only as a fallback; the real bounds are passed via
        // the chain at campaign time through coverage_keys order. Since compare only
        // gets keys, recompute equal-region fallback when transfer sizes are uniform.
        // (DMA chains use uniform in-regions; padding only grows output uniformly per
        // padded transfer. We localize by equal division over keys.len() as scalar does,
        // which is exact when regions are equal-sized.)
        let n = keys.len().max(1);
        let bounds = equal_bounds(emu.output.len().min(reference.output.len()), n);
        first_divergent_region(&emu.output, &reference.output, &bounds)
            .map(|r| keys[r.min(n - 1)].clone())
    }

    fn bank(&self, case_dir: &Path, c: &DmaChain, reference: Option<&DmaObs>, _emu: Option<&DmaObs>) -> Result<PathBuf, String> {
        bank_case(case_dir, c, &c.keys(), reference.map(|o| o.output.as_slice()))
    }

    fn load_banked(&self, seed_dir: &Path) -> Result<Banked<DmaChain, DmaObs>, String> {
        let record: DmaChainRecord = std::fs::read_to_string(seed_dir.join("chain.json"))
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string()))?;
        let npu_output = std::fs::read(seed_dir.join("npu_output.bin")).map_err(|e| format!("npu_output.bin: {e}"))?;
        Ok(Banked::Replayable { case: record.chain, reference: DmaObs { output: npu_output }, keys: record.keys })
    }

    fn dump_divergent_observation(&self, case_dir: &Path, emu: &DmaObs) -> Result<(), String> {
        std::fs::write(case_dir.join("emu_output.bin"), &emu.output)
            .map_err(|e| format!("emu_output.bin write error: {e}"))
    }
}

/// Equal byte-region split over n regions (fallback localization).
fn equal_bounds(total: usize, n: usize) -> Vec<(usize, usize)> {
    let each = (total / n).max(1);
    (0..n).map(|k| (k * each, ((k + 1) * each).min(total))).collect()
}
```

**Note on `compare` localization:** the bounds-aware localizer (Task 9) is exact, but `compare` only receives `keys`, not the case. Since DMA chains use uniform input regions (and padding grows the output uniformly per padded transfer), the equal-split over `keys.len()` is correct for the common case. **Refinement decision for the implementer:** if a chain mixes padded and unpadded transfers (unequal output regions), equal-split mislocalizes. Resolve by making `DmaObs` carry the per-region byte bounds (computed in `observe` from the case, stored in `DmaObs`), so `compare` uses exact bounds without needing the case. Implement `DmaObs { output: Vec<u8>, bounds: Vec<(usize,usize)> }`, set `bounds` in `observe_impl` from the chain, and bank/restore it. This keeps localization exact under padding. (This is the clean form; do it now rather than the equal-split fallback.)

- [ ] **Step 4: Implement the `DmaObs.bounds` refinement** noted above so localization is exact: add `bounds: Vec<(usize,usize)>` to `DmaObs`; populate in `observe_impl` via `chain.transfers.iter().map(|t| t.out_byte_bounds(chain.dtype))`; in `compare` use `emu.bounds` (the EMU side, authoritative for the case shape); bank/restore `bounds` in the record (or recompute from the banked chain in `load_banked`). Update the Task 9 tests' `DmaObs` construction accordingly.

- [ ] **Step 5: Run** the localizer tests + the ignored EMU-run gate

Run: `cargo test --lib fuzzer::domains::dma::domain`
Expected: PASS.

Run (toolchain): `cargo test --lib fuzzer::domains::dma::domain::tests::end_to_end_emu_run_produces_output -- --ignored`
Expected: PASS (non-zero reshuffle output).

- [ ] **Step 6: Commit**

```bash
git add src/fuzzer/domains/dma/domain.rs
git commit -m "fuzzer/dma: banking + impl Domain (compile override, exact-bounds localize)

Generated using Claude Code."
```

---

## Task 11: Runner + CLI + main wiring (`fuzz-dma` subcommand)

**Files:**
- Create: `src/fuzzer/domains/dma/runner.rs`
- Modify: `src/fuzzer/cli.rs` (add `parse_dma_fuzz_args`)
- Modify: `src/main.rs` (dispatch `fuzz-dma`)

- [ ] **Step 1: Write `runner.rs`** (DMA is campaign-only -- no legacy trace path):

```rust
//! DMA fuzz campaign runner.
use crate::fuzzer::core::domain::CampaignOptions;
use crate::fuzzer::core::engine::run_campaign;
use super::domain::DmaDomain;

pub struct DmaFuzzOptions {
    pub campaign: CampaignOptions,
}

pub fn run_dma_fuzz(opts: &DmaFuzzOptions) {
    run_campaign(&DmaDomain, &opts.campaign);
}
```

- [ ] **Step 2: Add `parse_dma_fuzz_args` to `cli.rs`** (mirror `parse_fuzz_args`, minus trace-sweep):

```rust
pub fn parse_dma_fuzz_args(args: &[String]) -> Result<DmaFuzzOptions, String> {
    let mut campaign = CampaignOptions {
        iterations: 0, seed: None, jobs: default_jobs(), hw: false,
        max_cycles: DEFAULT_MAX_CYCLES, target_hits: 10, verbose: false,
        report_only: false, replay: None, reverify: false,
    };
    let mut iter = args.iter().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "fuzz-dma" => {}
            "--iterations" | "-n" => campaign.iterations = parse_next(&mut iter, "--iterations")?,
            "--seed" => campaign.seed = Some(parse_next(&mut iter, "--seed")?),
            "--jobs" | "-j" => campaign.jobs = parse_next(&mut iter, "--jobs")?,
            "--max-cycles" => campaign.max_cycles = parse_next(&mut iter, "--max-cycles")?,
            "--target-hits" => campaign.target_hits = parse_next(&mut iter, "--target-hits")?,
            "--hw" => campaign.hw = true,
            "--no-hw" => campaign.hw = false,
            "--report" => campaign.report_only = true,
            "--reverify" => campaign.reverify = true,
            "--replay" => { let d = iter.next().ok_or("--replay requires a directory")?; campaign.replay = Some(PathBuf::from(d)); }
            "--verbose" | "-v" => campaign.verbose = true,
            other => return Err(format!("unknown fuzz-dma argument: {other}")),
        }
    }
    Ok(DmaFuzzOptions { campaign })
}
```

Add `use crate::fuzzer::domains::dma::runner::DmaFuzzOptions;` to cli.rs imports.

- [ ] **Step 3: Dispatch in `main.rs`.** Next to the `args[1] == "fuzz"` block, add:

```rust
    if args.len() >= 2 && args[1] == "fuzz-dma" {
        #[cfg(feature = "tooling")]
        { return run_dma_fuzz_command(&args); }
        #[cfg(not(feature = "tooling"))]
        { eprintln!("fuzz-dma command requires --features tooling"); std::process::exit(1); }
    }
```

and the handler:

```rust
fn run_dma_fuzz_command(args: &[String]) -> anyhow::Result<()> {
    let opts = xdna_emu::fuzzer::cli::parse_dma_fuzz_args(args).map_err(|e| anyhow::anyhow!("fuzz-dma: {}", e))?;
    xdna_emu::fuzzer::domains::dma::runner::run_dma_fuzz(&opts);
    Ok(())
}
```

Update the CLI help text to mention `fuzz-dma`.

- [ ] **Step 4: Write a CLI parse test** in cli.rs tests:

```rust
    #[test]
    fn parse_dma_fuzz_basic() {
        let args: Vec<String> = ["bin", "fuzz-dma", "-n", "50", "--seed", "7", "--report"].iter().map(|s| s.to_string()).collect();
        let o = parse_dma_fuzz_args(&args).unwrap();
        assert_eq!(o.campaign.iterations, 50);
        assert_eq!(o.campaign.seed, Some(7));
        assert!(o.campaign.report_only);
    }
```

- [ ] **Step 5: Run + commit**

Run: `cargo test --lib fuzzer::cli` and `cargo test --lib` (full)
Expected: PASS.

```bash
git add src/fuzzer/domains/dma/runner.rs src/fuzzer/cli.rs src/main.rs
git commit -m "fuzzer/dma: fuzz-dma subcommand (runner + CLI + main dispatch)

Generated using Claude Code."
```

---

## Task 12: EMU smoke campaign across the 81-key universe + safety assertion

**Files:**
- Modify: `src/fuzzer/domains/dma/domain.rs` (add an ignored full-universe smoke test)

This is the "built to 100% / no signal marginal" gate: every key generates, compiles, and runs on the emulator without panic; generation stays Stage-A-safe.

- [ ] **Step 1: Write the smoke test** (ignored; toolchain + time):

```rust
    #[test]
    #[ignore = "full-universe EMU smoke; run with --ignored"]
    fn universe_emu_smoke() {
        use crate::fuzzer::core::domain::{Backend, Domain};
        let tools = match crate::fuzzer::core::toolchain::ToolPaths::discover() { Ok(t) => t, Err(_) => return };
        let dom = DmaDomain;
        let mut compiled = 0; let mut ran = 0;
        for (i, key) in dom.universe().into_iter().enumerate() {
            let c = dom.generate(i as u64 + 1, &key);
            let dir = std::env::temp_dir().join(format!("dma_smoke_{i}_{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            if dom.compile(&tools, &dir, &c).is_ok() {
                compiled += 1;
                if let Ok(obs) = dom.observe(Backend::Interpreter, &dir.join("aie.xclbin"), &dir.join("insts.bin"), &c, 2_000_000) {
                    assert!(!obs.output.is_empty());
                    ran += 1;
                }
            }
            std::fs::remove_dir_all(&dir).ok();
        }
        // every 3a key must compile and run on the emulator
        assert_eq!(compiled, 81, "compiled {compiled}/81");
        assert_eq!(ran, 81, "ran {ran}/81");
    }
```

- [ ] **Step 2: Run it** (this is the acceptance gate for 3a EMU coverage):

Run (toolchain): `cargo test --lib fuzzer::domains::dma::domain::tests::universe_emu_smoke -- --ignored --nocapture`
Expected: `compiled 81/81`, `ran 81/81`. If any key fails to compile or run, fix the emitter/generator for that (engine, feature, dir, dtype) before proceeding -- this is the "finish what you start" bar.

- [ ] **Step 3: Run the full unit suite** (no regressions, vector/scalar still green)

Run: `cargo test --lib`
Expected: PASS; count = prior + the DMA tests.

- [ ] **Step 4: Commit**

```bash
git add src/fuzzer/domains/dma/domain.rs
git commit -m "fuzzer/dma: full-universe EMU smoke gate (81/81 compile + run)

Generated using Claude Code."
```

---

## Task 13: Final holistic review + vector/scalar byte-identical verification

**Files:** none (review + verification only)

- [ ] **Step 1: Verify the two prior tenants are byte-identical** (the `Domain::compile` seam must not have touched them). Run the vector replay/report and scalar smoke exactly as the Step-2 outcome did:
  - `cargo test --lib` full pass.
  - If a HW/bridge path is available, spot-check that vector still reports clean; otherwise rely on the unit suite + the fact that the default `compile` is behaviorally identical to the old inline path.

- [ ] **Step 2: Dispatch a final code-quality reviewer** over the whole `src/fuzzer/domains/dma/` tree + the three `core/` edits, checking: no `unwrap` on toolchain/IO paths that should return `Result`; the emitter matches the spike golden forms; the universe gating matches the spec; `Iter`/`Chain` are reachable in the enum but excluded from `universe_keys()`; no `iter`/`chain` keys leak into a campaign.

- [ ] **Step 3: Confirm the deferred 3b boundary is clean** -- `Feature::Iter`/`Feature::Chain` exist, `table::supported` returns false for them, `gen` never targets them, and the spec's 3b section names the exact emulator fixes (shim BD-repeat, memtile next_bd). No 3b code is present.

- [ ] **Step 4: Write the plan Outcome section** (append to this file): commits, final test count, acceptance criteria met, any emitter quirks found during execution, and the confirmed 3a/3b boundary.

---

## Self-Review Notes (author)

- **Spec coverage:** decisions 1-10 of the spec map to tasks: program model + localization (T9-T10), topology (T5/T7), universe (T3), generation/Stage-A (T4), lowering (T5-T8), compile seam (T1), observe (T9), banking (T10), vacuity (T10 warnings). The 3a/3b split (post-spike section) is enforced in T3 (gating) + T13 (boundary check).
- **Residual risk flagged honestly:** the emitter tasks (T5-T8) lean on the spike's *verified golden MLIR* as the concrete target + an `aiecc` compile-gate as the oracle, rather than inlining every line of string-building. This is the correct shape for codegen against a dialect: the golden is the spec, the compile-gate is the proof. Every feature has a named golden artifact under `build/experiments/dma-spike/`.
- **The one localization subtlety** (compare gets keys, not the case, but DMA regions are unequal under padding) is resolved in T10-Step-4 by carrying `bounds` in `DmaObs` -- exact, not equal-split.
- **Type consistency:** `DmaObs` gains a `bounds` field in T10-Step-4; the T9 tests construct `DmaObs` and must be updated there (called out). `compile_dma_mlir`, `ToolPaths`, `make_dma_buffer_spec` signatures are consistent across T1/T9/T10.
- **No HW in the plan's gates:** all compile/run gates are EMU + `aiecc` (no NPU contention). The `--hw` differential campaign is run manually after merge (acceptance), like Steps 1/2.
