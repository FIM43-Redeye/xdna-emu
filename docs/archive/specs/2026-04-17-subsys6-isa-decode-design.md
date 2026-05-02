# Subsystem 6 -- ISA Decode -- Design

**Subsystem:** 6 of 8 (Phase 1b of the device-family refactor)
**Date:** 2026-04-17
**Parent refactor:** [docs/superpowers/specs/2026-04-16-device-family-refactor-design.md](2026-04-16-device-family-refactor-design.md)
**Prior subsystem:** [docs/superpowers/specs/2026-04-17-subsys1-regs-mem-design.md](2026-04-17-subsys1-regs-mem-design.md)
**Planned tag:** `phase1-subsys-isa-decode`

---

## Goal

Complete the arch-agnostic migration of the ISA decode infrastructure out of
`xdna-emu` into `xdna-archspec`, split the LLVM MCDisassembler FFI cleanly
along its interpreter-coupling boundary, rewrite the remaining `crate::arch::*`
and `crate::tablegen::*` consumers to direct `xdna_archspec` imports, and
dissolve both `mod arch` and `mod tablegen` forwarder blocks. This subsystem
consumes Subsystem 1's deferred Tasks 9 and 10 and finishes the Subsystem 1
follow-up list.

## Non-goals

- **No `IsaDecoder` trait.** Decoding behavior is invariant across AIE2/AIE2P/AIE1
  given the TableGen-driven bundle formats; slot count and format layout are data,
  not shape. A trait seam is a documented future candidate when AIE1 decoder
  data materializes, not a Subsystem 6 commitment.
- **No `Operand` move.** `interpreter::bundle::slot::Operand` is the emulator's
  execution-state model, referenced by 38 files across `interpreter/execute/*`.
  It is not arch data. Keeping it in xdna-emu preserves the correct ownership
  split: archspec describes silicon; xdna-emu models execution.
- **No second-arch implementation.** No AIE1 or AIE2P decoder data lands.
  Phase 1 is seams + relocation; arch population is future work.

---

## Context

Subsystem 1's Part A moved `gen_arch`, `gen_subsystems`, `gen_core_module`,
`gen_stream_ports`/`gen_stream_ranges`, and `gen_trace_events` into
`xdna-archspec/build.rs`. Two tasks were blocked and deferred:

- **Task 9** (move `build_helpers/` and `gen_tablegen` to archspec) was blocked
  by `gen_tablegen.rs`'s emitted paths (`use super::super::types::*;`
  `use super::super::resolver::*;`) that resolve only inside xdna-emu's module
  tree.
- **Task 10** (move `decoder_ffi/` and `compile_llvm_decoder_ffi` to archspec)
  was blocked by `src/tablegen/decoder_ffi.rs` using
  `crate::interpreter::bundle::slot::Operand`,
  `crate::interpreter::state::{LR_REG_INDEX, ...}`, and the
  `MappedOperand`/`RegisterMap`/`classify_reg_name` layer.

Both deferrals traced to the same root: the tablegen runtime types
(`src/tablegen/types.rs`, `src/tablegen/resolver/`) and the decoder FFI
consumer layer live inside `xdna-emu`, so any build-side move would leave
dangling cross-crate paths.

Audit of the actual coupling surface (see `## Coupling Audit` below) shows
the deferral is narrower than the initial investigation suggested. Only the
bottom half of `decoder_ffi.rs` (lines 346-1185) is interpreter-coupled. The
top half, all of `types.rs`, all of `resolver/`, and `decoder_bytecode.rs` are
arch-agnostic and move cleanly. This design splits `decoder_ffi.rs` at line
346 and moves everything above to archspec.

---

## Coupling audit (actual state, not narrative)

| File | Lines | Coupling |
|------|-------|----------|
| `src/tablegen/types.rs` | 1510 | None (only `std::collections`) |
| `src/tablegen/resolver/mod.rs` | 1188 | None (only `super::types`, `std::collections`) |
| `src/tablegen/resolver/operand_classification.rs` | 577 | None |
| `src/tablegen/resolver/semantic_inference.rs` | 515 | None |
| `src/tablegen/decoder_bytecode.rs` | 344 | None (uses `super::types`) |
| `src/tablegen/decoder_ffi.rs` lines 1-345 | 345 | None -- pure FFI: `OpKind`, `RawOperand`, `Slot`, `DecodedOperand`, `DecodeResult`, `InstrInfo`, `mcid`, `query_all_instr_info`, `init` |
| `src/tablegen/decoder_ffi.rs` lines 346-1185 | 839 | **Hard**: `crate::interpreter::bundle::slot::Operand` + `crate::interpreter::state::{LR_REG_INDEX, LS_REG_INDEX, LE_REG_INDEX, LC_REG_INDEX, DP_REG_INDEX, CORE_ID_REG_INDEX, SP_PTR_INDEX, MOD_BASE_M, MOD_BASE_DN, MOD_BASE_DJ, MOD_BASE_DC}` |
| `build_helpers/*.rs` | ~3340 | Build-time only; emits `super::super::types::*` path strings (in `build_helpers/codegen.rs:27-28, 56-57`) |
| `decoder_ffi/*.cpp`, `*.h`, `*.inc` | ~466 | Arch-agnostic |

**Consumer counts (for rewrite scope):**
- `crate::arch::*` imports: 36 files (was 37; `registers_spec.rs` self-consumer dissolved in Subsystem 1 Part B).
- `crate::tablegen::*` imports: 38 files (found via `rg 'use crate::tablegen'` in src/).

Overlap between the two sets is expected but not enumerated here -- each file
rewrites independently.

---

## Approach: Option A (wholesale lift, split at interpreter boundary)

### Section 1: Architecture

Three concerns folded into one subsystem:

**(a) Relocation.** Move the arch-agnostic slice of the ISA infrastructure
from xdna-emu into `xdna_archspec::aie2::isa::`. Specifically:

- `build_helpers/` -> `crates/xdna-archspec/build_helpers/`
- `src/tablegen/types.rs` -> `crates/xdna-archspec/src/aie2/isa/types.rs`
- `src/tablegen/resolver/` -> `crates/xdna-archspec/src/aie2/isa/resolver/`
- `src/tablegen/decoder_bytecode.rs` -> `crates/xdna-archspec/src/aie2/isa/decoder_bytecode.rs`
- `src/tablegen/decoder_ffi.rs` lines 1-345 -> `crates/xdna-archspec/src/aie2/isa/decoder_ffi.rs`
- `decoder_ffi/` -> `crates/xdna-archspec/decoder_ffi/`
- `build_helpers/element_type_logic.rs` (shared build+runtime) -> `crates/xdna-archspec/src/aie2/isa/element_type_logic.rs`, with `#[path]` re-inclusion from archspec's build.rs to keep the canonical-source property
- `gen_tablegen.rs` emission -> archspec's `OUT_DIR`, path strings rewrite `super::super::` -> `super::`
- `compile_llvm_decoder_ffi` + `run_llvm_config` + `llvm_aie_path` resolution -> archspec's build.rs
- `extract_aiert` + `gen_aiert_dma`/`gen_aiert_locks`/`gen_aiert_ports` -> archspec's build.rs (archspec already has the extract half for cross-validation; this subsystem adds the file-writing half)

**(b) Splitting `decoder_ffi.rs` at line 346.** The cut boundary falls where
interpreter types first appear. Above the line: pure FFI, raw decode,
`AccumWidth`, `DecodedOperand`, `DecodeResult`, `InstrInfo` + `mcid` flags,
`query_all_instr_info`, the `extern "C"` block, the `aie2_*` function
wrappers, the `Once` init. Below the line: `MappedOperand`, `parse_reg_name`,
`classify_reg_name`, `RegisterMap`. The bottom half becomes
`xdna-emu/src/interpreter/decode/register_map.rs` -- a thin adapter that
consumes archspec's raw decoder output and produces the interpreter's
`Operand` enum.

**(c) Consumer cleanup and forwarder dissolution.**

- Rewrite 36 `crate::arch::*` consumers to `xdna_archspec::aie2::*`.
- Rewrite 38 `crate::tablegen::*` consumers to `xdna_archspec::aie2::isa::*`
  (for types/resolver), or to `crate::interpreter::decode::register_map::*`
  (for `MappedOperand`/`RegisterMap`), or to
  `xdna_archspec::aie2::isa::decoder_ffi::*` (for FFI types like
  `AccumWidth`, `Slot`, `DecodedOperand`).
- Delete `pub mod arch { ... }` in `xdna-emu/src/lib.rs`.
- Delete `xdna-emu/src/tablegen/` entirely.
- xdna-emu's `src/device/aiert_validation.rs` rewrites to import from
  `xdna_archspec::aie2::aiert::{dma, locks, ports}`. The `include!()`s
  themselves move to archspec's side of the fence, where each generated
  `gen_aiert_*.rs` is wrapped in a `pub mod` that archspec re-exports. The
  xdna-emu side becomes a plain consumer of archspec's public API.

### Section 2: Components

**New: `crates/xdna-archspec/src/aie2/isa/`**

```
isa/
|-- mod.rs                  # re-exports + load_from_generated() entry
|-- types.rs                # (moved from src/tablegen/types.rs, unchanged body)
|-- resolver/
|   |-- mod.rs
|   |-- operand_classification.rs
|   `-- semantic_inference.rs
|-- decoder_bytecode.rs     # (moved)
|-- decoder_ffi.rs          # (top half of original; pure FFI)
`-- element_type_logic.rs   # (moved; shared build+runtime via #[path])
```

`mod.rs` `include!()`s `gen_tablegen.rs` from `OUT_DIR`. Codegen strings
emit `super::types::*` and `super::resolver::*` (one level less than the
old `super::super::*`, because archspec's module tree is shallower than
xdna-emu's was).

**New: `crates/xdna-archspec/build_helpers/`**

Verbatim move. Its `mod.rs`, `extract.rs`, `codegen.rs`, `semantics.rs`,
`records.rs`, `bytecode.rs`, `element_type_logic.rs`, `cpp_switch.rs`.
Referenced from `crates/xdna-archspec/build.rs` via
`#[path = "../build_helpers/mod.rs"] mod build_helpers;` (mirrors the current
xdna-emu sibling-to-crate layout).

**New: `crates/xdna-archspec/decoder_ffi/`**

Verbatim move of `aie2_decoder.cpp`, `aie2_decoder.h`, and the
LLVM-`tblgen`-produced `.inc` files. archspec's build.rs gains the
`compile_llvm_decoder_ffi` logic.

**New: `xdna-emu/src/interpreter/decode/register_map.rs`**

Bottom half of `decoder_ffi.rs` (lines 346-1185). Exports `MappedOperand`,
`classify_reg_name`, `RegisterMap`. Imports
`xdna_archspec::aie2::isa::decoder_ffi::{DecodedOperand, AccumWidth, init,
aie2_decode_slot, ...}` for input and `crate::interpreter::bundle::slot::Operand`
for output. Pure adapter. No new indirection vs. current code.

**Dissolved: `xdna-emu/src/tablegen/`**

Directory and `mod tablegen` declaration in `src/lib.rs` both go away. All
38 consumer files get their imports rewritten.

**Dissolved: `pub mod arch` in `xdna-emu/src/lib.rs`**

Forwarder block deletes. The 36 `crate::arch::*` consumers rewrite to
`xdna_archspec::aie2::*`. The `mod arch::subsystem` compat shim goes with it.

**Dissolved: ISA codegen in `xdna-emu/build.rs`**

Remaining after Subsystem 1: `extract_aiert` + ~10 parsing helpers +
`gen_aiert_dma`/`gen_aiert_locks`/`gen_aiert_ports`, the
`#[path = "build_helpers/mod.rs"]` include + `extract_all` +
`generate_tablegen_file`, `compile_llvm_decoder_ffi` + `run_llvm_config` +
`llvm_aie_path`. All of those migrate to archspec's build.rs. xdna-emu's
build.rs reduces to `gen_header` + the XRT plugin install block -- ~80 lines,
which is Subsystem 1's original Task 11 target.

**Dissolved from `xdna-emu/Cargo.toml` `[build-dependencies]`:** `cc`,
`tblgen`, `serde`, `serde_json` (all needed only by the moved codegen).

### Section 3: Data flow

**Build time (single `cargo build`):**

1. `xdna-archspec/build.rs` runs (xdna-emu depends on archspec; cargo orders it
   first).
2. Archspec build phases:
   - `extract_all()` invokes `tblgen` on llvm-aie TableGen sources -> raw
     extraction structs.
   - `generate_tablegen_file()` writes `gen_tblgen_slot_*.rs` + orchestrator
     `gen_tablegen.rs` into archspec's `OUT_DIR`. Emitted paths use
     `super::types::*` / `super::resolver::*`.
   - `extract_aiert()` runs C-preprocessor on aie-rt headers -> feeds
     cross-validation (already present) AND writes
     `gen_aiert_{dma,locks,ports}.rs` into archspec's `OUT_DIR` (new behavior).
   - `compile_llvm_decoder_ffi()` `cc`-compiles `decoder_ffi/aie2_decoder.cpp`
     + `.inc` files, links via `llvm-config --libs aie`. Emits the static lib
     archspec exposes.
3. `xdna-emu/build.rs` runs. XRT plugin install only.
4. archspec's `src/aie2/isa/mod.rs` `include!()`s `gen_tablegen.rs`; paths
   resolve in the same module tree.
5. xdna-emu's `src/device/aiert_validation.rs` imports from
   `xdna_archspec::aie2::aiert::{dma, locks, ports}` (new re-export
   modules; the `include!()` lives on the archspec side).

**Runtime decode (hot path, mechanically unchanged):**

1. Interpreter holds a `RegisterMap` (xdna-emu) built once via
   `RegisterMap::new()`.
2. `RegisterMap::new()` calls
   `xdna_archspec::aie2::isa::decoder_ffi::init()` then iterates FFI
   register names, running each through `classify_reg_name` to populate
   `HashMap<String, MappedOperand>`. Same one-time cost as today.
3. Per-instruction decode: interpreter calls
   `xdna_archspec::aie2::isa::decoder_ffi::aie2_decode_slot(slot, bits)` ->
   `DecodeResult { operands: Vec<DecodedOperand>, ... }`. Import path
   changes; execution path is byte-identical.
4. Interpreter looks up names via `RegisterMap::lookup(name)` ->
   `MappedOperand { operand: Operand, accum_width }`.

Zero new dyn dispatch. Zero new wrapper layers. Zero new HashMap allocs. The
module split is a namespace boundary, not an abstraction boundary.

### Section 4: Type ownership after the move

| Type | Crate | Why |
|------|-------|-----|
| `SemanticOp`, `InstrEncoding`, `OperandField`, `OperandType`, `RegisterKind`, `CompositeFormatDef`, `ProcessorModel`, `ItineraryInfo`, `RegisterModel`, etc. | archspec | Arch data; describes what silicon does |
| `DecodedOperand`, `DecodeResult`, `InstrInfo`, `AccumWidth`, `Slot`, `OpKind`, `RawOperand`, `RawDecodeResult`, `RawInstrInfo` | archspec | FFI surface; describes LLVM decoder output (arch-specific only in the sense of which TableGen fed it, which is covered by the `aie2::` namespace) |
| `mcid` module (MCID flag constants) | archspec | LLVM protocol constants |
| `Operand` (`ScalarReg`, `VectorReg`, `Memory { .. }`, etc.) | xdna-emu | Emulator execution state, not arch data. 38 consumers across `interpreter/execute/*`. |
| `MappedOperand`, `RegisterMap`, `classify_reg_name`, `parse_reg_name` | xdna-emu | Adapter: LLVM register name -> emulator `Operand`. |
| `LR_REG_INDEX`, `LS_REG_INDEX`, `LE_REG_INDEX`, `LC_REG_INDEX`, `DP_REG_INDEX`, `CORE_ID_REG_INDEX`, `SP_PTR_INDEX`, `MOD_BASE_*` | xdna-emu | Emulator register-file indexing conventions, not arch data. |

The split answers NEXT-STEPS.md's third question ("Does `RegisterMap` /
`MappedOperand` / `classify_reg_name` want a trait or a direct move?"):
direct move of `Operand` + constants is the cleaner-but-bigger move and the
answer is we don't do it -- the layer is fundamentally emulator-side, so it
lives in xdna-emu.

### Section 5: Testing and verification

**Global invariants (every commit):**

- `cargo test --lib` green (baseline: 2797 passed; 0 failed; 5 ignored).
- `cargo test -p xdna-archspec --lib` green (baseline: 138 passed; 1
  pre-existing fail).
- `cargo build` and `cargo build --release` green.
- `./scripts/emu-bridge-test.sh --no-hw -v add_one` green (fast smoke).

**Per-relocation truth check:**

Sharpest verification for a pure-relocation task is bit-identical generated
output. Procedure per task:

```bash
# before the task
cargo build
cp "$(find target/debug/build -name gen_tablegen.rs | head -1)" /tmp/claude-1000/gen_tablegen_before.rs

# after the task (path may now live under xdna-archspec-*/out)
cargo build
diff /tmp/claude-1000/gen_tablegen_before.rs "$(find target/debug/build -name gen_tablegen.rs | head -1)"
```

Expected diff: only the emitted `super::super::` -> `super::` path
rewrites (at most a handful of lines). Anything else is a regression.

Same procedure for `gen_aiert_{dma,locks,ports}.rs` during their move.

**Subsystem unit tests that follow the code:**

- The 7-test block in `src/tablegen/mod.rs` (`test_acq_instruction_disambiguation`,
  `test_acq_lock_id_field_type`, `test_structural_semantic_inference`,
  `test_processor_model`, `test_itinerary_data`, `test_register_model`,
  `test_composite_formats`) moves to `crates/xdna-archspec/src/aie2/isa/mod.rs`.
- FFI-level tests in `src/tablegen/decoder_ffi.rs` (top half) move with the
  file. Classifier tests (bottom half) stay with
  `xdna-emu/src/interpreter/decode/register_map.rs`.

**Per-subsystem gate (at each tag):**

1. `./scripts/emu-bridge-test.sh 2>&1 | tee /tmp/claude-1000/bridge-phase1-subsys6-<partA|partB>.log`
   -- full HW bridge run, ~30 min. Compare pass/fail matrix against
   `phase1-subsys-regs-mem` baseline; no new regressions.
2. `./scripts/isa-test.sh 2>&1 | tee /tmp/claude-1000/isa-subsys6-<partA|partB>.log`
   -- ISA test suite, ~10 min. Expect `FAIL: 0` across all 4815 test points.
3. Bridge and ISA run sequentially (never concurrently -- both target the
   NPU and would race for the device).

Expected verification cost: ~45 min per tag.

### Section 6: Scope gating -- two-part structure

This subsystem is ~9,300 lines relocated + ~74 consumer-file rewrites + the
`decoder_ffi.rs` split + the `extract_aiert` migration. Single-part would be
one long linear sequence. Two-part mirrors Subsystem 1's proven cadence:

**Part A -- Relocation (tag: `phase1-subsys-isa-decode-partA`).** Move
code. Keep `pub mod arch` and `pub mod tablegen` alive as forwarder blocks
(`pub use xdna_archspec::aie2::*`, `pub use xdna_archspec::aie2::isa::*`).
Consumers still use `crate::arch::*` / `crate::tablegen::*` paths; they
compile against the forwarders. Tasks: relocations + the decoder_ffi split
+ the `extract_aiert` migration. Full HW + ISA gate at tag.

**Part B -- Consumer cleanup (tag: `phase1-subsys-isa-decode`).** Atomic
sed-rewrite of all 36 + 38 = 74 consumer imports to direct
`xdna_archspec::aie2[::isa]::*` paths. Delete `pub mod arch` and
`pub mod tablegen` forwarders. Delete `src/tablegen/` directory.
`xdna-emu/build.rs` shrinks to ~80 lines (Subsystem 1's Task 11 target).
`xdna-emu/Cargo.toml` build-deps drop to empty (or keep only `xdna-archspec`
if the plugin install ends up needing it, which it currently doesn't). Full
HW + ISA gate at tag.

**Rationale for two parts:** Part A is a mechanical move with bit-identical
codegen output -- cheap to verify, cheap to recover from. Part B is the
atomic rewrite that must happen in one commit to avoid transient broken
states where some consumers use forwarder paths and others don't. Splitting
the risk aligns with Subsystem 1's "reduced-scope partial tag" precedent:
if Part B hits trouble, Part A is still a useful resting point.

---

## Why no trait seam

The parent refactor spec's "coarse first" and "what would AIE1 look like?"
principles apply. Concrete answer for ISA decode:

AIE1 has a 3-slot VLIW bundle vs. AIE2's 6-slot (+ nop) bundle. That sounds
like shape, but the decoder doesn't hardcode slot count -- it walks a
`composite_formats: Vec<CompositeFormatDef>` table and applies a bytecode
interpreter per slot. Change the data, and the same algorithm decodes a
3-slot bundle:

- `SlotDef` entries populate based on AIE1's `AIE1Slots.td` instead of AIE2.
- `CompositeFormatDef` entries describe the 3-slot bundle layout.
- The decoder bytecode (`decoder_bytecode.rs`) walks those tables
  unchanged.

A trait would be justified if decoding an AIE1 instruction required a
fundamentally different algorithm -- e.g., a non-TableGen-driven decoder,
or a different operand-extraction discipline. It does not. AIE1's decoder
is the same algorithm on different data.

**If that assessment is wrong,** the trait introduces itself naturally
when AIE1 decoder data lands: if the algorithm diverges, we extract
`IsaDecoder::decode_slot(&self, slot: Slot, bits: u64) -> DecodeResult`
then. The build-time infrastructure this subsystem establishes does not
prevent that; it only prevents *premature* introduction of the trait.

---

## Forward pointers

- **Subsystem 7 (ISA Execute).** Semantic-op handlers, intrinsic dispatch.
  `Operand` stays in xdna-emu during this subsystem and remains in xdna-emu
  through Subsystem 7 -- execution semantics are not arch data. A future
  `IsaExecutor` trait lands there if vector compute behavior diverges across
  archs (likely around rounding/saturation configuration words), which is
  orthogonal to decode.
- **Future: `IsaDecoder` trait.** If AIE1 lands and the decode algorithm
  diverges, extract here. Design note to be added at
  `docs/arch/isa-decode-trait.md` when that happens.

---

## Deliverables checklist

- [ ] All 9,300-ish lines relocated with bit-identical codegen output.
- [ ] `decoder_ffi.rs` split at line 346; bottom half becomes
      `src/interpreter/decode/register_map.rs`.
- [ ] `xdna-emu/src/tablegen/` directory deleted.
- [ ] `pub mod arch` and `pub mod tablegen` forwarder blocks deleted from
      `xdna-emu/src/lib.rs`.
- [ ] 36 `crate::arch::*` consumer rewrites + 38 `crate::tablegen::*`
      consumer rewrites (74 file edits; atomic in Part B).
- [ ] `extract_aiert` + `gen_aiert_*` dissolved from `xdna-emu/build.rs`
      into archspec's build.rs; `src/device/aiert_validation.rs` imports
      from archspec.
- [ ] `xdna-emu/build.rs` reduced to ~80 lines (XRT plugin install + header).
- [ ] `xdna-emu/Cargo.toml` `[build-dependencies]` emptied (except
      `xdna-archspec` if retained for plugin install).
- [ ] Part A tag: `phase1-subsys-isa-decode-partA`. Part B tag:
      `phase1-subsys-isa-decode`.
- [ ] Full HW bridge + ISA suite green at both tags.
- [ ] Design note at `docs/arch/isa-decode.md` (mandatory per-seam note from
      the parent refactor; explains the no-trait decision).
- [ ] `docs/arch/subsys6-audit.md` documenting baseline, audit facts,
      relocations, commits, verification, and any deferrals.
- [ ] `NEXT-STEPS.md` updated to point at Subsystem 2 as up next.

---

## Success criteria (must all hold at the final tag)

1. `cargo test --lib` passes; count is >= 2797 (Subsystem 6 tests that
   move join archspec's count and leave xdna-emu's count at >= baseline).
2. `cargo test -p xdna-archspec --lib` passes (>= 138 + newly-moved tests).
3. `cargo build --release` clean.
4. Full HW bridge run shows no new regressions vs. `phase1-subsys-regs-mem`
   baseline. `bd_chain_repeat_on_memtile` remains a known pre-existing
   failure on real HW, as at prior tags.
5. ISA test suite: `FAIL: 0 / 4815`.
6. `crate::arch::*` imports: 0 occurrences in `src/`.
7. `crate::tablegen::*` imports: 0 occurrences in `src/`.
8. `xdna-emu/build.rs` line count: ~80 +/- 10.
9. No `TODO`/`FIXME`/`unimplemented!()` without an open-issue reference.
10. All commits land on `dev`; no master merges during the subsystem.

---

## Ground rules (inherited from the parent refactor spec)

- **No master merges during the refactor.** Everything lands on `dev`.
- **`cargo test --lib` green at every commit.** Non-negotiable.
- **Bridge test smoke green at every subsystem tag.** Full HW run before
  each of Part A and Part B tags.
- **One authoritative source per concept.** Once ISA decode lives in
  archspec, it lives only in archspec. No parallel re-introduction in
  xdna-emu.
- **Traits decode/step/check; they do not hold mutable state.** State lives
  in plain structs. (Not exercised this subsystem since no trait is added;
  preserved for discipline.)
- **Coarse first.** No trait in Subsystem 6 -- documented in the design
  note.
- **No second-arch implementation during the refactor.** No AIE1 or AIE2P
  population.
