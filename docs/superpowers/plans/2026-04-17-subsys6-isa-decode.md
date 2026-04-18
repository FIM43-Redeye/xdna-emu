# Subsystem 6 -- ISA Decode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the arch-agnostic ISA decode infrastructure (tablegen types + resolver + decoder bytecode, build_helpers/, decoder_ffi/ C++ glue, gen_tablegen codegen, compile_llvm_decoder_ffi, extract_aiert + gen_aiert_*) from xdna-emu into `xdna_archspec::aie2::isa::*`, split `decoder_ffi.rs` at its interpreter-coupling boundary, rewrite the remaining 36 `crate::arch::*` and 25+ `crate::tablegen::*` consumers to direct archspec imports, and dissolve both `pub mod arch` and `pub mod tablegen` forwarders.

**Architecture:** Two-part subsystem. Part A (Tasks 1-12) relocates all infrastructure while keeping both forwarder blocks alive so consumers compile unchanged. Part B (Tasks 13-19) atomically rewrites all consumers and dissolves both forwarders, shrinking `xdna-emu/build.rs` to ~80 lines (XRT plugin install only).

**Tech Stack:** Rust 2021 workspace, `xdna-archspec` workspace crate, `#[path]` build-script module includes, `tblgen` LLVM TableGen Rust bindings, `cc` crate for C++ compile, LLVM 21 MCDisassembler via FFI.

**Spec:** [docs/superpowers/specs/2026-04-17-subsys6-isa-decode-design.md](../specs/2026-04-17-subsys6-isa-decode-design.md)

**Parent refactor:** [docs/superpowers/specs/2026-04-16-device-family-refactor-design.md](../specs/2026-04-16-device-family-refactor-design.md)

---

## Scope Note

**Part A (Tasks 1-12)** is pure relocation. Bit-identical codegen output at every step. Forwarder blocks (`pub mod arch`, `pub mod tablegen`) keep existing consumers compiling. Any regression in Part A is a missed import or a path mismatch, never a semantic change.

**Part B (Tasks 13-19)** is the atomic consumer rewrite + forwarder dissolution + build.rs shrinkage. Runs only after Part A's HW gate passes.

Branch: `dev`. Part A tag: `phase1-subsys-isa-decode-partA`. Part B tag: `phase1-subsys-isa-decode`.

---

## Global Invariants (every task, every commit)

- `cargo test --lib` green (baseline: 2797 passed; 0 failed; 5 ignored).
- `cargo test -p xdna-archspec --lib` green (baseline: 138 passed; 1 pre-existing fail `test_full_parse_all_devices`).
- `cargo build` green (release build required before each tag, not every commit).
- `./scripts/emu-bridge-test.sh --no-hw -v add_one` green (fast smoke, ~30 s).
- Full bridge HW run + `./scripts/isa-test.sh` green at each of Part A's and Part B's tags with no regressions vs. `phase1-subsys-regs-mem` baseline.
- No commit introduces `TODO`/`FIXME`/`unimplemented!()` without an open-issue reference.
- Commit messages: lowercase type prefix (`refactor:`, `docs:`, `test:`, `build:`); no emoji; ends with "Generated using Claude Code.".
- All work on `dev`. No merges to `master` during this plan.

---

## File Structure

**Current layout (post-Subsystem 1):**

```
xdna-emu/
├── build.rs                    # ~995 lines; extract_aiert + TableGen + FFI compile + XRT plugin install
├── build_helpers/              # AIE2 TableGen extractor (8 files, ~3,340 lines)
├── decoder_ffi/                # aie2_decoder.cpp + .h + .inc files (~466 lines)
├── src/
│   ├── lib.rs                  # has `pub mod arch { pub use xdna_archspec::aie2::*; ... }`
│   ├── tablegen/               # mod.rs + types.rs + resolver/ + decoder_bytecode.rs + decoder_ffi.rs
│   └── device/
│       └── aiert_validation.rs # include!()s gen_aiert_{dma,locks,ports}.rs
├── Cargo.toml                  # build-deps: serde_json, xdna-archspec, tblgen, cc
└── crates/xdna-archspec/
    ├── build.rs                # Subsystem 1's codegen: gen_arch, gen_subsystems, gen_trace_events, etc.
    ├── Cargo.toml              # build-deps: serde, serde_json
    └── src/
        ├── lib.rs
        ├── model_builder.rs
        ├── types.rs / device_model.rs / regdb.rs / regdb_extractor.rs / tablegen.rs / runtime.rs
        └── aie2/
            ├── mod.rs
            ├── registers.rs / subsystems.rs / stream_switch.rs / trace_events.rs / memory_map.rs
```

**Target layout after Part A:**

```
xdna-emu/
├── build.rs                    # unchanged from current (~995 lines); Part B shrinks it
├── (build_helpers/ deleted)
├── (decoder_ffi/ deleted)
├── src/
│   ├── lib.rs                  # `pub mod arch` forwarder + `pub mod tablegen` forwarder
│   ├── tablegen/
│   │   ├── mod.rs              # forwarder: `pub use xdna_archspec::aie2::isa::*;` + register_map sub-mod
│   │   └── register_map.rs     # bottom half of old decoder_ffi.rs (MappedOperand, classify_reg_name, RegisterMap)
│   └── device/
│       └── aiert_validation.rs # unchanged; Part B rewrites to archspec imports
└── crates/xdna-archspec/
    ├── build.rs                # +extract_aiert, +gen_aiert_*, +TableGen block, +compile_llvm_decoder_ffi
    ├── build_helpers/          # moved from xdna-emu
    ├── decoder_ffi/            # moved from xdna-emu
    ├── Cargo.toml              # build-deps: serde, serde_json, tblgen, cc
    └── src/
        ├── aie2/
        │   ├── mod.rs          # `pub mod isa; pub mod aiert;`
        │   ├── isa/
        │   │   ├── mod.rs               # include!(gen_tablegen.rs); re-exports
        │   │   ├── types.rs             # (moved)
        │   │   ├── resolver/            # (moved)
        │   │   ├── decoder_bytecode.rs  # (moved)
        │   │   ├── decoder_ffi.rs       # top half of old file
        │   │   └── element_type_logic.rs # (moved from build_helpers/)
        │   └── aiert/
        │       └── mod.rs      # pub mod dma/locks/ports each with include!()
```

**Target layout after Part B:**

```
xdna-emu/
├── build.rs                    # ~80 lines: XRT plugin install + header only
├── Cargo.toml                  # [build-dependencies] empty
├── src/
│   ├── lib.rs                  # no `pub mod arch`, no `pub mod tablegen`
│   ├── (tablegen/ deleted)
│   ├── interpreter/
│   │   └── decode/
│   │       └── register_map.rs # moved from src/tablegen/register_map.rs
│   └── device/
│       └── aiert_validation.rs # imports from xdna_archspec::aie2::aiert::*
```

---

## Consumer-Import Rewrite Reference

Part B's consumer rewrites follow a fixed substitution table. Document here for single-source-of-truth; subagents executing Task 14 and Task 15 reference this.

**`crate::arch::*` rewrites** (36 consumer files):

| Old path | New path |
|----------|----------|
| `crate::arch::COLUMNS` (and other top-level consts) | `xdna_archspec::aie2::COLUMNS` |
| `crate::arch::compute::*` | `xdna_archspec::aie2::compute::*` |
| `crate::arch::memtile::*` | `xdna_archspec::aie2::memtile::*` |
| `crate::arch::shim::*` | `xdna_archspec::aie2::shim::*` |
| `crate::arch::cardinal::*` | `xdna_archspec::aie2::cardinal::*` |
| `crate::arch::timing::*` | `xdna_archspec::aie2::timing::*` |
| `crate::arch::packet::*` | `xdna_archspec::aie2::packet::*` |
| `crate::arch::ctrl_packet::*` | `xdna_archspec::aie2::ctrl_packet::*` |
| `crate::arch::fot::*` | `xdna_archspec::aie2::fot::*` |
| `crate::arch::processor::*` | `xdna_archspec::aie2::processor::*` |
| `crate::arch::port_type::*` | `xdna_archspec::aie2::port_type::*` |
| `crate::arch::subsystem::*` | `xdna_archspec::aie2::subsystems::*` |
| `crate::arch::registers::*` | `xdna_archspec::aie2::registers::*` |
| `crate::arch::stream_switch::*` | `xdna_archspec::aie2::stream_switch::*` |
| `crate::arch::trace_events::*` | `xdna_archspec::aie2::trace_events::*` |
| `crate::arch::memory_map::*` | `xdna_archspec::aie2::memory_map::*` |
| `crate::arch::{COMPUTE,MEMTILE,SHIM}_MASTER_PORTS` | `xdna_archspec::aie2::{COMPUTE,MEMTILE,SHIM}_MASTER_PORTS` |
| `crate::arch::{COMPUTE,MEMTILE,SHIM}_SLAVE_PORTS` | `xdna_archspec::aie2::{COMPUTE,MEMTILE,SHIM}_SLAVE_PORTS` |

**`crate::tablegen::*` rewrites** (25+ consumer files):

| Old path | New path |
|----------|----------|
| `crate::tablegen::{types}::*` | `xdna_archspec::aie2::isa::*` (re-exported at `isa/mod.rs`) |
| `crate::tablegen::{SemanticOp, ImplicitReg, BranchCondition, ElementType, SelectVariant, ...}` | `xdna_archspec::aie2::isa::*` |
| `crate::tablegen::{OperandField, InstrEncoding, OperandType, RegisterKind, AddressingMode, InstrMemWidth, ...}` | `xdna_archspec::aie2::isa::*` |
| `crate::tablegen::{ProcessorModel, ItineraryInfo, RegisterModel, CompositeFormatDef, ...}` | `xdna_archspec::aie2::isa::*` |
| `crate::tablegen::decoder_bytecode::*` | `xdna_archspec::aie2::isa::decoder_bytecode::*` |
| `crate::tablegen::decoder_ffi::{OpKind, Slot, DecodedOperand, DecodeResult, InstrInfo, AccumWidth, mcid, init, aie2_decode_slot, aie2_opcode_name, query_all_instr_info, ...}` | `xdna_archspec::aie2::isa::decoder_ffi::*` |
| `crate::tablegen::decoder_ffi::{MappedOperand, RegisterMap, classify_reg_name, parse_reg_name}` | `crate::interpreter::decode::register_map::*` |

---

## Baseline to Preserve

Before Task 1, capture current numbers so later regression checks have a target:

```bash
cargo test --lib 2>&1 | tail -3
cargo test -p xdna-archspec --lib 2>&1 | tail -3
./scripts/emu-bridge-test.sh --no-hw -v add_one 2>&1 | tail -5
```

Expected current values:
- Library tests: `2797 passed; 0 failed; 5 ignored`
- Archspec tests: `138 passed; 1 failed` (`test_full_parse_all_devices`, pre-existing)
- Bridge smoke: Chess 10/10 PASS, Peano 9/9 PASS

Record in `docs/arch/subsys6-audit.md` (created in Task 1).

---

# Part A -- Infrastructure Relocation

**Tag at end:** `phase1-subsys-isa-decode-partA`

---

### Task 1: Audit

**Files:**
- Create: `docs/arch/subsys6-audit.md`

- [ ] **Step 1: Seed the audit doc**

```bash
mkdir -p docs/arch
touch docs/arch/subsys6-audit.md
```

- [ ] **Step 2: Capture baseline test numbers**

Run and record the output of each:

```bash
cargo test --lib 2>&1 | tail -3
cargo test -p xdna-archspec --lib 2>&1 | tail -3
./scripts/emu-bridge-test.sh --no-hw -v add_one 2>&1 | tail -5
```

Append to `docs/arch/subsys6-audit.md`:

```markdown
# Subsystem 6 -- ISA Decode Audit

## Baseline (pre-subsystem, at phase1-subsys-regs-mem tag)

- `cargo test --lib`: <paste output>
- `cargo test -p xdna-archspec --lib`: <paste output>
- Bridge `--no-hw -v add_one`: <paste output>

Failures to carry through: `test_full_parse_all_devices` (archspec, pre-existing, device count 13 vs expected 12 -- unrelated).
```

- [ ] **Step 3: Enumerate `crate::arch::*` consumers**

```bash
rg -l 'crate::arch' src/ examples/ tests/ xrt-plugin/ 2>&1 | sort > /tmp/claude-1000/subsys6-arch-consumers.txt
wc -l /tmp/claude-1000/subsys6-arch-consumers.txt
```

Append to audit under `## crate::arch Consumers`. Expected: 36 files under `src/`.

- [ ] **Step 4: Enumerate `crate::tablegen::*` consumers**

```bash
rg -l 'use crate::tablegen' src/ examples/ tests/ xrt-plugin/ 2>&1 | sort > /tmp/claude-1000/subsys6-tablegen-consumers.txt
wc -l /tmp/claude-1000/subsys6-tablegen-consumers.txt
```

Append to audit under `## crate::tablegen Consumers`. Expected: ~25-38 files.

- [ ] **Step 5: Enumerate current `xdna-emu/build.rs` codegen surface**

```bash
rg -n '^fn (gen_|extract_|compile_|run_)' build.rs
wc -l build.rs
```

Append to audit under `## xdna-emu/build.rs Surface`. Expected:
- `extract_aiert` + `gen_aiert_dma`/`locks`/`ports` (~4 functions)
- `compile_llvm_decoder_ffi` + `run_llvm_config` (2 functions)
- `gen_header`
- `main()` orchestrates all of the above plus the XRT plugin install

Line count expected: ~995.

- [ ] **Step 6: Enumerate `decoder_ffi.rs` split-line location**

```bash
rg -n '^use crate::interpreter' src/tablegen/decoder_ffi.rs
```

Append to audit under `## decoder_ffi.rs Split Line`. Expected: line 346 (`use crate::interpreter::bundle::slot::Operand;`). Everything above is pure FFI; everything from 346 onward is interpreter-coupled.

- [ ] **Step 7: Commit the audit**

```bash
git add docs/arch/subsys6-audit.md
git commit -m "$(cat <<'EOF'
docs: subsys6 ISA decode audit

Baseline test numbers, crate::arch / crate::tablegen consumer
enumeration, current build.rs codegen surface, and decoder_ffi.rs
split-line location. Guides Part A's file-by-file relocation and
Part B's atomic consumer rewrite.

Generated using Claude Code.
EOF
)"
```

---

### Task 2: Create target module skeleton in archspec

**Goal:** Set up the empty `xdna_archspec::aie2::isa::` module tree so subsequent relocation tasks can move files into a ready destination. Nothing changes compilation-wise yet.

**Files:**
- Create: `crates/xdna-archspec/src/aie2/isa/mod.rs`
- Modify: `crates/xdna-archspec/src/aie2/mod.rs`

- [ ] **Step 1: Create the empty `isa` module**

Write `crates/xdna-archspec/src/aie2/isa/mod.rs`:

```rust
//! AIE2 instruction set architecture: decoder tables, runtime model,
//! and LLVM MCDisassembler FFI.
//!
//! All content here is build-time extracted from llvm-aie (TableGen
//! sources + LLVM libraries). The runtime consumes generated constants
//! via `load_from_generated()`.
//!
//! Submodules populate across Subsystem 6's Part A relocation:
//! - `types` (arch-agnostic instruction/register/operand types)
//! - `resolver` (operand classification + semantic inference)
//! - `decoder_bytecode` (bytecode walker for instruction decode)
//! - `decoder_ffi` (LLVM MCDisassembler FFI, raw side only)
//! - `element_type_logic` (shared build+runtime element-type inference)
//!
//! The interpreter-aware half of the old `decoder_ffi.rs` (MappedOperand,
//! RegisterMap, classify_reg_name) lives in xdna-emu's
//! `interpreter::decode::register_map`, not here.
```

- [ ] **Step 2: Declare `isa` as a submodule of `aie2`**

Edit `crates/xdna-archspec/src/aie2/mod.rs`. Add at the end of the `pub mod` declarations (after `pub mod port_type { ... }` block):

```rust
/// Instruction set architecture: decoder tables, runtime model, and
/// LLVM MCDisassembler FFI. Populated during Subsystem 6 Part A.
pub mod isa;
```

- [ ] **Step 3: Verify build**

```bash
cargo build 2>&1 | tail -5
cargo test -p xdna-archspec --lib 2>&1 | tail -3
```

Expected: clean build, archspec tests unchanged (138 pass / 1 pre-existing fail).

- [ ] **Step 4: Commit**

```bash
git add crates/xdna-archspec/src/aie2/mod.rs crates/xdna-archspec/src/aie2/isa/mod.rs
git commit -m "$(cat <<'EOF'
refactor: scaffold xdna_archspec::aie2::isa module

Empty skeleton for Subsystem 6's ISA decode relocation. Subsequent
tasks populate types, resolver, decoder_bytecode, decoder_ffi (raw
half), and element_type_logic.

Generated using Claude Code.
EOF
)"
```

---

### Task 3: Move `src/tablegen/types.rs` to archspec

**Goal:** Move the pure-data TableGen runtime types into archspec. Keep xdna-emu's `src/tablegen/mod.rs` re-exporting them via forwarder so consumers compile.

**Files:**
- Move: `src/tablegen/types.rs` → `crates/xdna-archspec/src/aie2/isa/types.rs`
- Modify: `src/tablegen/mod.rs`
- Modify: `crates/xdna-archspec/src/aie2/isa/mod.rs`

- [ ] **Step 1: Copy `types.rs` to archspec**

```bash
cp src/tablegen/types.rs crates/xdna-archspec/src/aie2/isa/types.rs
```

- [ ] **Step 2: Declare it in `isa/mod.rs`**

Edit `crates/xdna-archspec/src/aie2/isa/mod.rs`. Add after the docstring:

```rust
pub mod types;
pub use types::*;
```

- [ ] **Step 3: Verify archspec builds**

```bash
cargo build -p xdna-archspec 2>&1 | tail -5
cargo test -p xdna-archspec --lib 2>&1 | tail -3
```

Expected: clean; archspec tests still 138/1.

- [ ] **Step 4: Replace xdna-emu's `types.rs` with a forwarder**

Overwrite `src/tablegen/types.rs` with:

```rust
//! Compatibility forwarder. The canonical definitions now live in
//! `xdna_archspec::aie2::isa::types`. This forwarder dissolves in
//! Subsystem 6 Part B when consumers migrate directly.

pub use xdna_archspec::aie2::isa::types::*;
```

- [ ] **Step 5: Verify xdna-emu builds and tests pass**

```bash
cargo build 2>&1 | tail -5
cargo test --lib 2>&1 | tail -3
```

Expected: clean build, 2797 passed / 0 failed / 5 ignored.

- [ ] **Step 6: Commit**

```bash
git add crates/xdna-archspec/src/aie2/isa/mod.rs crates/xdna-archspec/src/aie2/isa/types.rs src/tablegen/types.rs
git commit -m "$(cat <<'EOF'
refactor: move tablegen types to xdna_archspec::aie2::isa::types

Pure-data types (SlotDef, EncodingPart, FormatClass, InstrDef, etc.)
move wholesale -- no interpreter coupling. xdna-emu's
src/tablegen/types.rs becomes a one-line pub-use forwarder.

Generated using Claude Code.
EOF
)"
```

---

### Task 4: Move `src/tablegen/resolver/` to archspec

**Goal:** Move the entire resolver subtree. Zero interpreter coupling (verified in the spec audit).

**Files:**
- Move: `src/tablegen/resolver/mod.rs` → `crates/xdna-archspec/src/aie2/isa/resolver/mod.rs`
- Move: `src/tablegen/resolver/operand_classification.rs` → `crates/xdna-archspec/src/aie2/isa/resolver/operand_classification.rs`
- Move: `src/tablegen/resolver/semantic_inference.rs` → `crates/xdna-archspec/src/aie2/isa/resolver/semantic_inference.rs`
- Modify: `src/tablegen/mod.rs`
- Modify: `crates/xdna-archspec/src/aie2/isa/mod.rs`

- [ ] **Step 1: Copy the resolver/ subtree to archspec**

```bash
mkdir -p crates/xdna-archspec/src/aie2/isa/resolver
cp src/tablegen/resolver/mod.rs crates/xdna-archspec/src/aie2/isa/resolver/mod.rs
cp src/tablegen/resolver/operand_classification.rs crates/xdna-archspec/src/aie2/isa/resolver/operand_classification.rs
cp src/tablegen/resolver/semantic_inference.rs crates/xdna-archspec/src/aie2/isa/resolver/semantic_inference.rs
```

- [ ] **Step 2: Declare it in `isa/mod.rs`**

Edit `crates/xdna-archspec/src/aie2/isa/mod.rs`. Add after the `pub mod types;` line:

```rust
pub mod resolver;
pub use resolver::{
    build_decoder_tables, AddressingMode, CompositeEncoder, DecoderIndex, InstrEncoding,
    InstrMemWidth, OperandField, OperandType, RegisterKind, ResolveError, Resolver, SlotIndex,
    classify_operand_type, detect_addressing_mode, detect_mem_width,
    infer_branch_condition, infer_dual_element_types, infer_element_type, infer_select_variant,
    refine_branch_semantic,
};
```

- [ ] **Step 3: Verify archspec builds**

```bash
cargo build -p xdna-archspec 2>&1 | tail -5
cargo test -p xdna-archspec --lib 2>&1 | tail -3
```

Expected: clean; archspec tests still 138/1.

- [ ] **Step 4: Replace xdna-emu's resolver with forwarder directory**

First delete old files:

```bash
rm -rf src/tablegen/resolver
mkdir -p src/tablegen/resolver
```

Write `src/tablegen/resolver/mod.rs`:

```rust
//! Compatibility forwarder. The canonical definitions now live in
//! `xdna_archspec::aie2::isa::resolver`. This forwarder dissolves in
//! Subsystem 6 Part B when consumers migrate directly.

pub use xdna_archspec::aie2::isa::resolver::*;
```

- [ ] **Step 5: Verify xdna-emu builds and tests pass**

```bash
cargo build 2>&1 | tail -5
cargo test --lib 2>&1 | tail -3
```

Expected: clean build, 2797 passed.

- [ ] **Step 6: Commit**

```bash
git add crates/xdna-archspec/src/aie2/isa/resolver crates/xdna-archspec/src/aie2/isa/mod.rs src/tablegen/resolver
git commit -m "$(cat <<'EOF'
refactor: move tablegen resolver to xdna_archspec::aie2::isa::resolver

Operand classification (AddressingMode, OperandType, RegisterKind,
InstrMemWidth, InstrEncoding) and semantic inference (infer_*,
refine_*) move wholesale. Zero interpreter coupling. xdna-emu's
src/tablegen/resolver/mod.rs becomes a one-line pub-use forwarder.

Generated using Claude Code.
EOF
)"
```

---

### Task 5: Move `src/tablegen/decoder_bytecode.rs` to archspec

**Goal:** Move the decoder bytecode walker. Uses `super::types` only; zero interpreter coupling.

**Files:**
- Move: `src/tablegen/decoder_bytecode.rs` → `crates/xdna-archspec/src/aie2/isa/decoder_bytecode.rs`
- Modify: `src/tablegen/mod.rs`
- Modify: `crates/xdna-archspec/src/aie2/isa/mod.rs`

- [ ] **Step 1: Copy to archspec**

```bash
cp src/tablegen/decoder_bytecode.rs crates/xdna-archspec/src/aie2/isa/decoder_bytecode.rs
```

- [ ] **Step 2: Declare it in `isa/mod.rs`**

Edit `crates/xdna-archspec/src/aie2/isa/mod.rs`. Add:

```rust
pub mod decoder_bytecode;
```

- [ ] **Step 3: Verify archspec builds**

```bash
cargo build -p xdna-archspec 2>&1 | tail -5
cargo test -p xdna-archspec --lib 2>&1 | tail -3
```

Expected: clean; archspec tests still 138/1.

- [ ] **Step 4: Replace xdna-emu's file with forwarder**

Overwrite `src/tablegen/decoder_bytecode.rs`:

```rust
//! Compatibility forwarder. The canonical definitions now live in
//! `xdna_archspec::aie2::isa::decoder_bytecode`. This forwarder
//! dissolves in Subsystem 6 Part B.

pub use xdna_archspec::aie2::isa::decoder_bytecode::*;
```

- [ ] **Step 5: Verify**

```bash
cargo build 2>&1 | tail -5
cargo test --lib 2>&1 | tail -3
```

Expected: clean, 2797 passed.

- [ ] **Step 6: Commit**

```bash
git add crates/xdna-archspec/src/aie2/isa/decoder_bytecode.rs crates/xdna-archspec/src/aie2/isa/mod.rs src/tablegen/decoder_bytecode.rs
git commit -m "$(cat <<'EOF'
refactor: move decoder_bytecode to xdna_archspec::aie2::isa

Bytecode walker for instruction decoding. Uses super::types only; zero
interpreter coupling. xdna-emu's copy becomes a one-line forwarder.

Generated using Claude Code.
EOF
)"
```

---

### Task 6: Move `build_helpers/` to archspec, move `element_type_logic`

**Goal:** Move the entire `build_helpers/` directory (build-time TableGen extraction) to archspec. In the same task, move `element_type_logic.rs` to its Subsystem 6 home at `src/aie2/isa/element_type_logic.rs` and reference it from build_helpers via `#[path]`. Update codegen string paths to match the new module depth.

**Files:**
- Move: `build_helpers/*` → `crates/xdna-archspec/build_helpers/*`
- Create: `crates/xdna-archspec/src/aie2/isa/element_type_logic.rs` (copied from build_helpers/element_type_logic.rs for runtime resolver's use)
- Modify: `crates/xdna-archspec/build_helpers/codegen.rs` (path strings)
- Modify: `crates/xdna-archspec/src/aie2/isa/mod.rs`
- Modify: `src/tablegen/mod.rs` (remove `#[path = "../../build_helpers/element_type_logic.rs"]` include)

- [ ] **Step 1: Move build_helpers directory**

```bash
git mv build_helpers crates/xdna-archspec/build_helpers
```

- [ ] **Step 2: Copy element_type_logic.rs into isa/ for runtime use**

```bash
cp crates/xdna-archspec/build_helpers/element_type_logic.rs crates/xdna-archspec/src/aie2/isa/element_type_logic.rs
```

- [ ] **Step 3: Declare `element_type_logic` in `isa/mod.rs`**

Edit `crates/xdna-archspec/src/aie2/isa/mod.rs`. Add:

```rust
pub mod element_type_logic;
```

- [ ] **Step 4: Leave codegen string paths alone**

The emitted `use super::super::types::*;` and `use super::super::resolver::*;`
strings in `crates/xdna-archspec/build_helpers/codegen.rs` must stay
unchanged at this task. They still resolve correctly because `gen_tablegen.rs`
is still `include!()`-ed inside xdna-emu's `src/tablegen/mod.rs::generated`
module, which is two levels deep under xdna-emu's `src/tablegen/{types,resolver}`
(the forwarders from Tasks 3 and 4).

Task 7 moves the `include!()` site into archspec's `src/aie2/isa/mod.rs`
and simultaneously updates these strings to `super::types::*` /
`super::resolver::*` (one level less). Updating them here would break
xdna-emu's build until Task 7 lands.

- [ ] **Step 5: Remove `#[path]`-include of element_type_logic from xdna-emu's tablegen**

Edit `src/tablegen/mod.rs`. Find:

```rust
// Shared element type inference logic (canonical source: build_helpers/).
// Included here so the runtime resolver can delegate to the same logic
// that the build-time codegen uses. See build_helpers/element_type_logic.rs.
#[path = "../../build_helpers/element_type_logic.rs"]
mod element_type_logic;
```

Delete that block (4 lines including comments).

- [ ] **Step 6: Verify archspec's element_type_logic wires into resolver**

Archspec's `resolver/semantic_inference.rs` uses `use super::super::element_type_logic::*;` (or similar). Check:

```bash
rg -n 'element_type_logic' crates/xdna-archspec/src/aie2/isa/
```

If any import path is wrong post-move, fix it. The path from `resolver/semantic_inference.rs` up to `isa/element_type_logic.rs` is `super::super::element_type_logic`.

- [ ] **Step 7: Do not yet wire build_helpers into archspec's build.rs**

Archspec's build.rs still has no TableGen block. Task 7 adds the wiring + the `include!(gen_tablegen.rs)` line. At this step, xdna-emu's build.rs still owns the TableGen block via its own `#[path]` include. The directory move itself must update xdna-emu's build.rs path.

Edit `build.rs`. Find:

```rust
#[path = "build_helpers/mod.rs"]
mod build_helpers;
```

Replace with:

```rust
#[path = "crates/xdna-archspec/build_helpers/mod.rs"]
mod build_helpers;
```

- [ ] **Step 8: Verify build**

```bash
cargo clean -p xdna-emu 2>&1 | tail -3
cargo clean -p xdna-archspec 2>&1 | tail -3
cargo build 2>&1 | tail -10
cargo test --lib 2>&1 | tail -3
cargo test -p xdna-archspec --lib 2>&1 | tail -3
```

Expected: clean build, all tests pass. At this task the generator still
emits `super::super::types::*` / `super::super::resolver::*` (Step 4 kept
them intact). `gen_tablegen.rs` is still `include!()`-ed from xdna-emu's
`src/tablegen/mod.rs::generated` module; the two-levels-up paths resolve
via xdna-emu's `src/tablegen/{types,resolver}.rs` forwarders, which
re-export from archspec. Task 7 flips both the include site and the
emitter strings atomically.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: move build_helpers/ to xdna-archspec

Build-time TableGen extraction directory moves wholesale.
element_type_logic.rs also copied into src/aie2/isa/ as the
canonical runtime source; the build-time copy in build_helpers/
remains for the build script's use. xdna-emu's build.rs #[path]
now points into the crate.

Generated using Claude Code.
EOF
)"
```

---

### Task 7: Wire TableGen block into archspec's build.rs

**Goal:** Move `generate_tablegen_file` + the `tblgen` dependency + the `#[path]` include of `build_helpers` from xdna-emu's build.rs to archspec's build.rs. The generated `gen_tablegen.rs` now lives in archspec's `OUT_DIR` and gets included by archspec's `src/aie2/isa/mod.rs`. xdna-emu's `src/tablegen/mod.rs` stops including it.

**Files:**
- Modify: `build.rs` (xdna-emu): remove `#[path = "crates/xdna-archspec/build_helpers/mod.rs"] mod build_helpers;`, remove the `generate_tablegen_file` call
- Modify: `crates/xdna-archspec/build.rs`: add `#[path = "build_helpers/mod.rs"] mod build_helpers;`, add `generate_tablegen_file` call
- Modify: `crates/xdna-archspec/Cargo.toml`: add `tblgen` to `[build-dependencies]`
- Modify: `src/tablegen/mod.rs`: remove the `generated` module block
- Modify: `crates/xdna-archspec/src/aie2/isa/mod.rs`: add the `generated` module and `load_from_generated` entry

- [ ] **Step 1: Add `tblgen` to archspec's Cargo.toml**

Edit `crates/xdna-archspec/Cargo.toml`. Add to `[build-dependencies]`:

```toml
# Used by build_helpers/ (TableGen extraction -> gen_tablegen.rs).
tblgen = { git = "https://github.com/FIM43-Redeye/tblgen-rs.git", branch = "feat/varbit-init", default-features = false, features = ["llvm21-0"] }
```

- [ ] **Step 2: Add the `#[path]` include and generator call to archspec's build.rs**

Edit `crates/xdna-archspec/build.rs`. At the top of the file (with other `#[path]` blocks):

```rust
#[path = "build_helpers/mod.rs"]
mod build_helpers;
```

Resolve LLVM_AIE_PATH inside `main()` (copy the resolution logic from xdna-emu/build.rs). After `build_arch_model` runs, add:

```rust
// TableGen extraction -> gen_tablegen.rs (and per-slot gen_tblgen_slot_*.rs)
let llvm_aie_path = resolve_llvm_aie_path(workspace_root);
let extracted = build_helpers::extract::extract_all(&llvm_aie_path)
    .unwrap_or_else(|e| panic!("TableGen extraction failed: {}", e));
build_helpers::codegen::generate_tablegen_file(&extracted, &out_dir);
println!("cargo:rerun-if-changed={}", llvm_aie_path.display());
println!("cargo:rerun-if-env-changed=LLVM_AIE_PATH");
```

Copy `resolve_llvm_aie_path` (or its inline equivalent) from xdna-emu/build.rs.

- [ ] **Step 3: Update codegen string paths and move the include site (atomic)**

These two changes must land together: the emitted paths in
`gen_tablegen.rs` and the module that `include!()`s it must match.

First, update the emitted path strings.

Edit `crates/xdna-archspec/build_helpers/codegen.rs`. Find:

```rust
writeln!(code, "use super::super::types::*;").unwrap();
writeln!(code, "use super::super::resolver::*;").unwrap();
```

Replace with:

```rust
writeln!(code, "use super::types::*;").unwrap();
writeln!(code, "use super::resolver::*;").unwrap();
```

The other `super::types::*` / `super::resolver::*` / `super::decoder_bytecode::*`
strings in the same file are already correct for the archspec-side module
depth. Leave them.

Second, move the `include!()` site.

Edit `src/tablegen/mod.rs`. Remove:

```rust
// Build-time generated instruction tables (per-slot files for parallel compilation)
mod generated {
    include!(concat!(env!("OUT_DIR"), "/gen_tablegen.rs"));
}

// ...

/// Load the complete TableGen model from build-time generated constants.
pub fn load_from_generated() -> types::TblgenOutput {
    generated::load_from_generated()
}
```

Edit `crates/xdna-archspec/src/aie2/isa/mod.rs`. Add:

```rust
mod generated {
    include!(concat!(env!("OUT_DIR"), "/gen_tablegen.rs"));
}

/// Load the complete TableGen model from build-time generated constants.
pub fn load_from_generated() -> types::TblgenOutput {
    generated::load_from_generated()
}
```

Why atomic: the old strings (`super::super::types::*`) resolve correctly
only when `gen_tablegen.rs` is included inside xdna-emu's
`src/tablegen/mod.rs::generated` (two module levels deep over
`src/tablegen/types.rs`). The new strings (`super::types::*`) resolve
correctly only when it's included inside archspec's
`src/aie2/isa/mod.rs::generated` (one level deep over `src/aie2/isa/types.rs`).
Either change without the other breaks the build.

- [ ] **Step 4: Forward `load_from_generated` from xdna-emu**

Edit `src/tablegen/mod.rs`. Replace the removed `pub fn load_from_generated` with:

```rust
pub use xdna_archspec::aie2::isa::load_from_generated;
```

- [ ] **Step 5: Remove `tblgen` from xdna-emu's Cargo.toml**

Edit `Cargo.toml`. In `[build-dependencies]`, remove the `tblgen = { ... }` line.

- [ ] **Step 6: Remove `#[path]` include of build_helpers from xdna-emu's build.rs**

Edit `build.rs`. Remove:

```rust
#[path = "crates/xdna-archspec/build_helpers/mod.rs"]
mod build_helpers;
```

Also remove wherever `build_helpers::extract::extract_all(...)` and `build_helpers::codegen::generate_tablegen_file(...)` are called in `main()`, plus the `LLVM_AIE_PATH` resolution if it's still present (it's still needed for `compile_llvm_decoder_ffi`; keep only that part).

- [ ] **Step 7: Clean build and verify**

```bash
cargo clean 2>&1 | tail -3
cargo build 2>&1 | tail -10
cargo test --lib 2>&1 | tail -3
cargo test -p xdna-archspec --lib 2>&1 | tail -3
```

Expected: clean build, 2797 passed / 0 failed on xdna-emu, 138/1 on archspec. Generator strings emit `super::types::*` / `super::resolver::*` which resolve against archspec's `src/aie2/isa/{types.rs, resolver/mod.rs}`.

- [ ] **Step 8: Sharp verification -- bit-identical gen_tablegen.rs**

Compare against baseline. Before Task 6 started, find the current gen_tablegen.rs path and save it:

```bash
find target/debug/build -name gen_tablegen.rs -path '*xdna-archspec*' | head -1
```

After this task, run the same find; the file should exist under archspec's OUT_DIR, not xdna-emu's. Diff against git's `phase1-subsys-regs-mem` tag's gen_tablegen.rs if one was saved, or just verify it parses + `cargo test --lib` passes.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: wire TableGen extraction into xdna-archspec's build.rs

gen_tablegen.rs now emits from archspec's build.rs into archspec's
OUT_DIR, consumed by src/aie2/isa/mod.rs. tblgen build-dep moves
crates. xdna-emu's build.rs loses the #[path] include of
build_helpers/ and the generate_tablegen_file call. xdna-emu's
load_from_generated becomes a pub-use forwarder.

Generated using Claude Code.
EOF
)"
```

---

### Task 8: Move `decoder_ffi/` directory to archspec

**Goal:** Move the C++ source (aie2_decoder.cpp + .h + .inc files) to archspec. The `compile_llvm_decoder_ffi` call stays in xdna-emu/build.rs for now; it'll move in Task 9.

**Files:**
- Move: `decoder_ffi/` → `crates/xdna-archspec/decoder_ffi/`
- Modify: `build.rs` (xdna-emu): update path to point into crate

- [ ] **Step 1: Move the directory**

```bash
git mv decoder_ffi crates/xdna-archspec/decoder_ffi
```

- [ ] **Step 2: Update xdna-emu's build.rs to reference the new location**

Edit `build.rs`. Find every reference to `"decoder_ffi/..."` path strings (inside `compile_llvm_decoder_ffi` and `run_llvm_config`). Replace `"decoder_ffi/"` with `"crates/xdna-archspec/decoder_ffi/"`.

There should be two or three such references. A grep finds them:

```bash
rg -n 'decoder_ffi' build.rs
```

- [ ] **Step 3: Verify build**

```bash
cargo clean -p xdna-emu 2>&1 | tail -3
cargo build 2>&1 | tail -10
cargo test --lib 2>&1 | tail -3
```

Expected: clean build, 2797 passed. The C++ compilation still happens from xdna-emu's build.rs but reads sources from the new location.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: move decoder_ffi/ (C++ source) into xdna-archspec

aie2_decoder.cpp + .h + LLVM-generated .inc files move to
crates/xdna-archspec/decoder_ffi/. xdna-emu's build.rs still owns
the compile step (moves in Task 9); path strings updated.

Generated using Claude Code.
EOF
)"
```

---

### Task 9: Move `compile_llvm_decoder_ffi` to archspec's build.rs

**Goal:** Move the C++ compile step, `run_llvm_config`, `cc` build-dep, and LLVM_AIE_PATH resolution out of xdna-emu's build.rs and into archspec's.

**Files:**
- Modify: `build.rs` (xdna-emu): remove compile_llvm_decoder_ffi, run_llvm_config, cc and llvm_aie_path resolution
- Modify: `crates/xdna-archspec/build.rs`: add the same functions
- Modify: `Cargo.toml` (xdna-emu): remove `cc` from `[build-dependencies]`
- Modify: `crates/xdna-archspec/Cargo.toml`: add `cc` to `[build-dependencies]`

- [ ] **Step 1: Add `cc` to archspec's Cargo.toml**

Edit `crates/xdna-archspec/Cargo.toml`. Add to `[build-dependencies]`:

```toml
# Used by compile_llvm_decoder_ffi (compiles decoder_ffi/aie2_decoder.cpp).
cc = "1"
```

- [ ] **Step 2: Copy `compile_llvm_decoder_ffi` + `run_llvm_config` to archspec's build.rs**

Identify these functions in `build.rs` (xdna-emu). They are self-contained; they only take paths as arguments. Copy them verbatim to `crates/xdna-archspec/build.rs`.

Update the `main()` in archspec's build.rs to call `compile_llvm_decoder_ffi(&llvm_aie_path)` (reusing the `llvm_aie_path` already resolved during Task 7 for TableGen extraction). The source path inside the function needs updating: the C++ source lives at `<manifest_dir>/decoder_ffi/` now that the directory is inside archspec.

Check the function body for path strings. Anywhere it references the manifest or project root, make sure it refers to archspec's manifest, not xdna-emu's. The `CARGO_MANIFEST_DIR` env in the function will now point to `crates/xdna-archspec/`, so relative paths `decoder_ffi/aie2_decoder.cpp` still work as long as the function reads them relative to `CARGO_MANIFEST_DIR`.

- [ ] **Step 3: Remove the functions and call sites from xdna-emu's build.rs**

Edit `build.rs` (xdna-emu). Remove:
- `fn compile_llvm_decoder_ffi(...)` definition
- `fn run_llvm_config(...)` definition (if it's separate)
- The `compile_llvm_decoder_ffi(&llvm_aie_path)` call in `main()`
- The `llvm_aie_path` resolution in `main()` (if it's no longer needed for anything else -- after Task 7 and this step, it shouldn't be)

- [ ] **Step 4: Remove `cc` from xdna-emu's Cargo.toml**

Edit `Cargo.toml`. In `[build-dependencies]`, remove:

```toml
cc = "1"
```

- [ ] **Step 5: Clean build and verify**

```bash
cargo clean 2>&1 | tail -3
cargo build 2>&1 | tail -15
cargo test --lib 2>&1 | tail -3
cargo test -p xdna-archspec --lib 2>&1 | tail -3
```

Expected: clean build. The FFI library is now emitted by archspec's build.rs into archspec's OUT_DIR. Cargo propagates linker flags from archspec to xdna-emu automatically (the `cargo:rustc-link-*` directives in the function attach to the crate whose build.rs emitted them).

If the link step fails with unresolved symbols, double-check that the `cargo:rustc-link-lib=...` and `cargo:rustc-link-search=...` directives are still being emitted. They should be, since the function body is unchanged.

- [ ] **Step 6: Verify FFI still works**

```bash
./scripts/emu-bridge-test.sh --no-hw -v add_one 2>&1 | tail -5
```

Expected: Chess 10/10 PASS, Peano 9/9 PASS. Bridge smoke specifically exercises the decoder.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: move compile_llvm_decoder_ffi into xdna-archspec

LLVM MCDisassembler FFI compilation moves from xdna-emu/build.rs
to crates/xdna-archspec/build.rs. cc build-dep moves with it.
archspec now owns the full decoder_ffi build pipeline; xdna-emu
sees the compiled library via cargo's cargo:rustc-link-* plumbing.

Generated using Claude Code.
EOF
)"
```

---

### Task 10: Split `decoder_ffi.rs` at the interpreter boundary

**Goal:** Move the pure-FFI top half (lines 1-345) of `src/tablegen/decoder_ffi.rs` to `crates/xdna-archspec/src/aie2/isa/decoder_ffi.rs`. Move the interpreter-coupled bottom half (lines 346-1185) to a new `src/tablegen/register_map.rs`. xdna-emu's `src/tablegen/decoder_ffi.rs` becomes a forwarder that re-exports both halves.

**Files:**
- Create: `crates/xdna-archspec/src/aie2/isa/decoder_ffi.rs` (top half)
- Create: `src/tablegen/register_map.rs` (bottom half)
- Modify: `src/tablegen/decoder_ffi.rs` (replace with forwarder)
- Modify: `src/tablegen/mod.rs` (declare register_map; re-export from decoder_ffi forwarder)
- Modify: `crates/xdna-archspec/src/aie2/isa/mod.rs` (declare decoder_ffi)

- [ ] **Step 1: Extract top half (lines 1-345) to archspec**

Read `src/tablegen/decoder_ffi.rs` lines 1-345. Copy those lines verbatim into `crates/xdna-archspec/src/aie2/isa/decoder_ffi.rs`. Confirm the split line is `use crate::interpreter::bundle::slot::Operand;` -- everything above stays; everything from that line downward moves to `register_map.rs`.

The top-half content: imports, `OpKind`, `RawOperand`, `RawDecodeResult`, `Slot`, `RawInstrInfo`, `extern "C"` block, `INIT` + `INIT_OK`, `DecodedOperand`, `DecodeResult`, `init()`, `aie2_decode_slot`, `aie2_opcode_name`, `mcid` module, `InstrInfo`, `query_all_instr_info`, `AccumWidth`.

Adjust the top-half file: remove any `use crate::interpreter::...` imports (there should be none above line 346).

- [ ] **Step 2: Declare decoder_ffi in `isa/mod.rs`**

Edit `crates/xdna-archspec/src/aie2/isa/mod.rs`. Add:

```rust
pub mod decoder_ffi;
```

- [ ] **Step 3: Create `src/tablegen/register_map.rs` with the bottom half**

Read `src/tablegen/decoder_ffi.rs` lines 346-1185. Copy those lines to the new file `src/tablegen/register_map.rs`.

At the top of the new file, add imports (the file was previously relying on being inside the same file as the top half):

```rust
//! LLVM register name -> emulator Operand mapping.
//!
//! Adapter layer between the FFI in `xdna_archspec::aie2::isa::decoder_ffi`
//! and the interpreter's `Operand` enum. Moved out of the combined
//! `decoder_ffi.rs` in Subsystem 6 because the FFI layer itself is
//! arch data and belongs in archspec, while this classifier is emulator
//! execution model.
//!
//! In Subsystem 6 Part B this file relocates to
//! `src/interpreter/decode/register_map.rs`. The path here is a
//! temporary home to keep the Part A diff mechanical.

use crate::interpreter::bundle::slot::Operand;
use crate::interpreter::state::{
    LR_REG_INDEX, LS_REG_INDEX, LE_REG_INDEX, LC_REG_INDEX,
    DP_REG_INDEX, CORE_ID_REG_INDEX, SP_PTR_INDEX,
    MOD_BASE_M, MOD_BASE_DN, MOD_BASE_DJ, MOD_BASE_DC,
};
use xdna_archspec::aie2::isa::decoder_ffi::{
    AccumWidth, DecodedOperand, DecodeResult, OpKind, Slot, InstrInfo,
    init, aie2_decode_slot, aie2_opcode_name, query_all_instr_info,
    aie2_get_num_regs, aie2_get_reg_name,
};
```

(The exact imports depend on what the bottom half uses. Read each `use` statement in the original file below line 346 and carry it over unchanged. For names the bottom half referenced implicitly because they were in the same file, import them from `xdna_archspec::aie2::isa::decoder_ffi` now.)

- [ ] **Step 4: Replace `src/tablegen/decoder_ffi.rs` with a forwarder**

Overwrite `src/tablegen/decoder_ffi.rs`:

```rust
//! Compatibility forwarder. The FFI layer now lives in
//! `xdna_archspec::aie2::isa::decoder_ffi`. The RegisterMap /
//! MappedOperand / classify_reg_name classifier lives in
//! `crate::tablegen::register_map` (moves to
//! `crate::interpreter::decode::register_map` in Part B).

pub use xdna_archspec::aie2::isa::decoder_ffi::*;
pub use super::register_map::*;
```

- [ ] **Step 5: Declare `register_map` in `src/tablegen/mod.rs`**

Edit `src/tablegen/mod.rs`. After the `pub mod decoder_ffi;` line (or wherever submodules are declared), add:

```rust
pub mod register_map;
```

- [ ] **Step 6: Verify build and tests**

```bash
cargo build 2>&1 | tail -10
cargo test --lib 2>&1 | tail -3
cargo test -p xdna-archspec --lib 2>&1 | tail -3
```

Expected: clean build, 2797 passed. All `use crate::tablegen::decoder_ffi::*` imports in the interpreter still resolve via the forwarder.

- [ ] **Step 7: Verify FFI still works**

```bash
./scripts/emu-bridge-test.sh --no-hw -v add_one 2>&1 | tail -5
```

Expected: Chess 10/10 PASS, Peano 9/9 PASS.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: split decoder_ffi.rs at interpreter boundary

The pure-FFI half (OpKind, RawOperand, Slot, DecodeResult, InstrInfo,
mcid, AccumWidth, init, query_all_instr_info, aie2_decode_slot) moves
to xdna_archspec::aie2::isa::decoder_ffi. The interpreter-coupled half
(MappedOperand, classify_reg_name, RegisterMap) moves to a new
src/tablegen/register_map.rs (relocates to interpreter/decode/ in
Part B). xdna-emu's decoder_ffi.rs becomes a forwarder.

Generated using Claude Code.
EOF
)"
```

---

### Task 11: Move `extract_aiert` + `gen_aiert_*` to archspec

**Goal:** Move the aie-rt extraction that feeds `gen_aiert_dma.rs`/`gen_aiert_locks.rs`/`gen_aiert_ports.rs` from xdna-emu's build.rs to archspec's. Wrap each generated file in a `pub mod` under `xdna_archspec::aie2::aiert::{dma,locks,ports}`. xdna-emu's `src/device/aiert_validation.rs` then imports from the archspec re-export module instead of `include!()`-ing from OUT_DIR.

**Files:**
- Modify: `build.rs` (xdna-emu): remove extract_aiert, gen_aiert_*, parsing helpers
- Modify: `crates/xdna-archspec/build.rs`: add file-writing side of extract_aiert + gen_aiert_*
- Create: `crates/xdna-archspec/src/aie2/aiert/mod.rs`
- Modify: `crates/xdna-archspec/src/aie2/mod.rs`: declare `pub mod aiert;`
- Modify: `src/device/aiert_validation.rs`: import from `xdna_archspec::aie2::aiert::*` instead of `include!()`

- [ ] **Step 1: Identify extract_aiert + friends in xdna-emu's build.rs**

```bash
rg -n '^fn (extract_aiert|gen_aiert_|parse_dma_|parse_lock_|parse_ports_)' build.rs
```

Expected: ~6-10 functions. Note: archspec's build.rs ALREADY has a partial copy of `extract_aiert` for cross-validation (stripped of the file-writing calls). We'll merge the file-writing calls back into archspec's copy, then delete xdna-emu's copy entirely.

- [ ] **Step 2: Add `gen_aiert_dma/locks/ports` to archspec's build.rs**

In `crates/xdna-archspec/build.rs`, find its `extract_aiert` function. After it parses the aie-rt modules (feeding `ArchModel`'s cross-validation), add the file-writing calls:

```rust
gen_aiert_dma(&dma_modules, &out_dir);
gen_aiert_locks(&lock_modules, &out_dir);
gen_aiert_ports(&port_maps, &out_dir);
```

Copy the function definitions for `gen_aiert_dma`, `gen_aiert_locks`, `gen_aiert_ports` from xdna-emu's build.rs into archspec's build.rs (these three functions are self-contained file writers; they take parsed data and emit Rust).

- [ ] **Step 3: Delete extract_aiert + gen_aiert_* from xdna-emu's build.rs**

Edit `build.rs` (xdna-emu). Remove:
- `extract_aiert()` function
- `gen_aiert_dma()`, `gen_aiert_locks()`, `gen_aiert_ports()` functions
- All parsing helpers used only by `extract_aiert` (`parse_dma_*`, `parse_lock_*`, `parse_ports_*`, `parse_desc_*`, etc.)
- The call site in `main()` that invokes `extract_aiert(...)`

This should drop xdna-emu's build.rs by several hundred lines. The remaining content at this point: the XRT plugin install block, `gen_header`, and minimal path/env setup.

- [ ] **Step 4: Create `crates/xdna-archspec/src/aie2/aiert/mod.rs`**

Write:

```rust
//! aie-rt cross-validated constants.
//!
//! Each submodule wraps a build-time-generated file (gen_aiert_*.rs)
//! containing data extracted from aie-rt headers via C preprocessor.
//! The data is also used for ArchModel cross-validation in archspec's
//! build.rs; this module exposes the same data for xdna-emu's
//! aiert_validation runtime tests.

pub mod dma {
    include!(concat!(env!("OUT_DIR"), "/gen_aiert_dma.rs"));
}

pub mod locks {
    include!(concat!(env!("OUT_DIR"), "/gen_aiert_locks.rs"));
}

pub mod ports {
    include!(concat!(env!("OUT_DIR"), "/gen_aiert_ports.rs"));
}
```

- [ ] **Step 5: Declare `aiert` in `aie2/mod.rs`**

Edit `crates/xdna-archspec/src/aie2/mod.rs`. Add:

```rust
/// aie-rt cross-validation data. Each submodule wraps a
/// build-time-generated file extracted from aie-rt headers.
pub mod aiert;
```

- [ ] **Step 6: Rewrite `src/device/aiert_validation.rs` to import from archspec**

Edit `src/device/aiert_validation.rs`. Find the three `include!()` blocks:

```rust
include!(concat!(env!("OUT_DIR"), "/gen_aiert_dma.rs"));
// ... later:
include!(concat!(env!("OUT_DIR"), "/gen_aiert_locks.rs"));
// ... later:
include!(concat!(env!("OUT_DIR"), "/gen_aiert_ports.rs"));
```

Replace each with an import from archspec:

```rust
use xdna_archspec::aie2::aiert::dma::*;
// ...
use xdna_archspec::aie2::aiert::locks::*;
// ...
use xdna_archspec::aie2::aiert::ports::*;
```

Depending on the file's structure, these imports may need to live at the top of the file (with their consumers using `dma::AIERT_DMA_MODULES` or similar unqualified names). Match the existing call-site style; the goal is zero behavioral change.

- [ ] **Step 7: Remove unused build-deps from xdna-emu's Cargo.toml**

Edit `Cargo.toml`. In `[build-dependencies]`, remove:

```toml
serde_json = "1"
```

(kept only for extract_aiert, which just moved). If `xdna-archspec` is still listed there, keep it for now -- Task 17 decides whether to drop it entirely.

- [ ] **Step 8: Clean build and verify**

```bash
cargo clean 2>&1 | tail -3
cargo build 2>&1 | tail -15
cargo test --lib 2>&1 | tail -3
cargo test -p xdna-archspec --lib 2>&1 | tail -3
```

Expected: clean build, 2797 passed on xdna-emu, 138/1 on archspec. The archspec build's cross-validation still runs; the xdna-emu side now consumes the generated files via archspec.

- [ ] **Step 9: Verify aiert_validation tests still pass**

```bash
cargo test --lib aiert_validation 2>&1 | tail -5
```

Expected: all aiert_validation tests pass.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: move extract_aiert + gen_aiert_* to xdna-archspec

The aie-rt extraction that feeds gen_aiert_{dma,locks,ports}.rs now
lives entirely in archspec's build.rs (archspec already had the
cross-validation side; gains the file-writing side). Each generated
file is wrapped in a pub mod under xdna_archspec::aie2::aiert::*.
xdna-emu's src/device/aiert_validation.rs imports from archspec
instead of include!()'ing from OUT_DIR. xdna-emu build.rs drops
serde_json build-dep.

Generated using Claude Code.
EOF
)"
```

---

### Task 12: Part A verification gate + tag

**Goal:** Run the full HW bridge suite + ISA test suite against the end-of-Part-A state. Tag `phase1-subsys-isa-decode-partA`. Document the Part A completion section in the audit.

**Files:**
- Modify: `docs/arch/subsys6-audit.md`

- [ ] **Step 1: Fast verification smoke**

```bash
cargo build --release 2>&1 | tail -5
cargo test --lib 2>&1 | tail -3
cargo test -p xdna-archspec --lib 2>&1 | tail -3
./scripts/emu-bridge-test.sh --no-hw -v add_one 2>&1 | tail -5
```

Expected:
- release build clean
- `2797 passed; 0 failed; 5 ignored`
- `138 passed; 1 failed`
- Chess 10/10 PASS, Peano 9/9 PASS

- [ ] **Step 2: Full HW bridge run**

```bash
nice -n 19 ./scripts/emu-bridge-test.sh 2>&1 | tee /tmp/claude-1000/subsys6-partA-bridge.log
```

Expected duration: ~20-30 minutes. Expected pass/fail matrix: matches `phase1-subsys-regs-mem` baseline. Known pre-existing failure `bd_chain_repeat_on_memtile` remains.

- [ ] **Step 3: ISA test suite**

```bash
nice -n 19 ./scripts/isa-test.sh 2>&1 | tee /tmp/claude-1000/subsys6-partA-isa.log
```

Expected duration: ~10 minutes. Expected: `FAIL: 0 / 4815`.

- [ ] **Step 4: Append Part A completion to the audit**

Edit `docs/arch/subsys6-audit.md`. Append:

```markdown
## Part A Completion

Landed 2026-MM-DD. Tag: `phase1-subsys-isa-decode-partA`.

### Commits (from Task 1 through tag)

<output of `git log --oneline phase1-subsys-regs-mem..HEAD`>

### Verification (at tag)

- `cargo test --lib`: 2797 passed; 0 failed; 5 ignored.
- `cargo test -p xdna-archspec --lib`: 138 passed; 1 failed (pre-existing).
- `cargo build --release`: clean.
- Bridge `--no-hw -v add_one`: Chess 10/10, Peano 9/9.
- Full HW bridge: matches phase1-subsys-regs-mem baseline; bd_chain_repeat_on_memtile still fails (real-HW; pre-existing).
- ISA test suite: FAIL: 0 / 4815.

### Part A deliverables

- [x] build_helpers/ moved to xdna-archspec.
- [x] decoder_ffi/ moved to xdna-archspec.
- [x] src/tablegen/types.rs, resolver/, decoder_bytecode.rs moved (with forwarders).
- [x] decoder_ffi.rs split at line 346; top half to archspec; bottom half to src/tablegen/register_map.rs temporarily.
- [x] extract_aiert + gen_aiert_* moved to archspec's build.rs.
- [x] compile_llvm_decoder_ffi moved to archspec's build.rs.
- [x] tblgen, cc, serde_json build-deps moved from xdna-emu to archspec.
- [x] xdna_archspec::aie2::{isa, aiert} modules populated.
- [x] Forwarders (`pub mod arch`, `pub mod tablegen`) still live; consumer rewrites deferred to Part B.
```

- [ ] **Step 5: Commit audit update**

```bash
git add docs/arch/subsys6-audit.md
git commit -m "$(cat <<'EOF'
docs: subsys6 Part A completion log

Part A landed: all ISA decode + aie-rt infrastructure relocated to
xdna-archspec. Consumer rewrites and forwarder dissolution are
Part B's scope.

Generated using Claude Code.
EOF
)"
```

- [ ] **Step 6: Tag**

```bash
git tag phase1-subsys-isa-decode-partA -m "Phase 1b Subsystem 6 Part A: ISA decode infrastructure relocation"
```

---

# Part B -- Consumer Cleanup and Forwarder Dissolution

**Tag at end:** `phase1-subsys-isa-decode`

---

### Task 13: Move `register_map.rs` to its final home

**Goal:** Relocate `src/tablegen/register_map.rs` to `src/interpreter/decode/register_map.rs` and expose as a submodule of `interpreter::decode`. xdna-emu's `src/tablegen/mod.rs` stops re-exporting it.

**Files:**
- Move: `src/tablegen/register_map.rs` → `src/interpreter/decode/register_map.rs`
- Modify: `src/interpreter/decode/mod.rs` (declare the submodule)
- Modify: `src/tablegen/mod.rs` (remove `pub mod register_map;`)
- Modify: `src/tablegen/decoder_ffi.rs` (forwarder): remove `pub use super::register_map::*;`

- [ ] **Step 1: Move the file**

```bash
git mv src/tablegen/register_map.rs src/interpreter/decode/register_map.rs
```

- [ ] **Step 2: Declare it in interpreter::decode**

Edit `src/interpreter/decode/mod.rs`. Add (at the module-declarations section):

```rust
pub mod register_map;
```

- [ ] **Step 3: Update the forwarder in src/tablegen/mod.rs**

Edit `src/tablegen/mod.rs`. Remove:

```rust
pub mod register_map;
```

- [ ] **Step 4: Update the forwarder in src/tablegen/decoder_ffi.rs**

Edit `src/tablegen/decoder_ffi.rs`. Remove:

```rust
pub use super::register_map::*;
```

- [ ] **Step 5: Consumers of MappedOperand/RegisterMap update**

Any file that imports `MappedOperand` or `RegisterMap` from `crate::tablegen::decoder_ffi::` must update to `crate::interpreter::decode::register_map::`. Find them:

```bash
rg -l 'MappedOperand|RegisterMap|classify_reg_name' src/ --type rust
```

Edit each to replace `use crate::tablegen::decoder_ffi::{MappedOperand, ...}` with `use crate::interpreter::decode::register_map::{MappedOperand, ...}`. Preserve the specific items imported.

- [ ] **Step 6: Verify**

```bash
cargo build 2>&1 | tail -10
cargo test --lib 2>&1 | tail -3
```

Expected: clean build, 2797 passed.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: relocate register_map to interpreter::decode

MappedOperand / RegisterMap / classify_reg_name are emulator execution
model, not arch data. They now live at their correct ownership home
in src/interpreter/decode/register_map.rs. Callers that used
crate::tablegen::decoder_ffi::{MappedOperand,...} now import from
crate::interpreter::decode::register_map::*.

Generated using Claude Code.
EOF
)"
```

---

### Task 14: Atomic rewrite of `crate::tablegen::*` consumers

**Goal:** In a single commit, rewrite all `use crate::tablegen::*` imports to `xdna_archspec::aie2::isa::*` (for types, resolver, decoder_bytecode, decoder_ffi entries) or to `crate::interpreter::decode::register_map::*` (for MappedOperand/RegisterMap/classify_reg_name).

**Files:**
- Modify: 25-38 files containing `use crate::tablegen::...`

Reference the "Consumer-Import Rewrite Reference" table at the top of this plan for the exact substitutions.

- [ ] **Step 1: Enumerate the consumer files**

```bash
rg -l 'use crate::tablegen' src/ | sort > /tmp/claude-1000/subsys6-tablegen-consumers-pre-rewrite.txt
wc -l /tmp/claude-1000/subsys6-tablegen-consumers-pre-rewrite.txt
```

- [ ] **Step 2: Mechanical rewrite using sed**

Apply the substitution rules from the Consumer-Import Rewrite Reference table. For each file:

```bash
# Rewrite types + resolver -> isa root (re-exported)
sed -i 's|crate::tablegen::types::|xdna_archspec::aie2::isa::|g' <file>
sed -i 's|crate::tablegen::resolver::|xdna_archspec::aie2::isa::|g' <file>

# Rewrite top-level re-exports (SemanticOp etc.)
sed -i 's|crate::tablegen::|xdna_archspec::aie2::isa::|g' <file>

# Rewrite decoder_bytecode
sed -i 's|xdna_archspec::aie2::isa::decoder_bytecode::|xdna_archspec::aie2::isa::decoder_bytecode::|g' <file>
# (no-op after the generic replace above; listed for clarity)

# MappedOperand/RegisterMap/classify_reg_name go to interpreter::decode::register_map
sed -i 's|xdna_archspec::aie2::isa::decoder_ffi::MappedOperand|crate::interpreter::decode::register_map::MappedOperand|g' <file>
sed -i 's|xdna_archspec::aie2::isa::decoder_ffi::RegisterMap|crate::interpreter::decode::register_map::RegisterMap|g' <file>
sed -i 's|xdna_archspec::aie2::isa::decoder_ffi::classify_reg_name|crate::interpreter::decode::register_map::classify_reg_name|g' <file>
sed -i 's|xdna_archspec::aie2::isa::decoder_ffi::parse_reg_name|crate::interpreter::decode::register_map::parse_reg_name|g' <file>
```

**Do this as a script with a list of files** rather than invoking sed per-file manually. Run on the enumerated consumers in one pass.

A helper script:

```bash
# /tmp/claude-1000/subsys6-rewrite-tablegen.sh
#!/bin/bash
for f in $(cat /tmp/claude-1000/subsys6-tablegen-consumers-pre-rewrite.txt); do
  sed -i 's|crate::tablegen::types::|xdna_archspec::aie2::isa::|g' "$f"
  sed -i 's|crate::tablegen::resolver::|xdna_archspec::aie2::isa::|g' "$f"
  sed -i 's|crate::tablegen::decoder_bytecode::|xdna_archspec::aie2::isa::decoder_bytecode::|g' "$f"
  sed -i 's|crate::tablegen::decoder_ffi::|xdna_archspec::aie2::isa::decoder_ffi::|g' "$f"
  sed -i 's|crate::tablegen::|xdna_archspec::aie2::isa::|g' "$f"
  # Fix up MappedOperand/RegisterMap/classify_reg_name after the generic replace
  sed -i 's|xdna_archspec::aie2::isa::decoder_ffi::MappedOperand|crate::interpreter::decode::register_map::MappedOperand|g' "$f"
  sed -i 's|xdna_archspec::aie2::isa::decoder_ffi::RegisterMap|crate::interpreter::decode::register_map::RegisterMap|g' "$f"
  sed -i 's|xdna_archspec::aie2::isa::decoder_ffi::classify_reg_name|crate::interpreter::decode::register_map::classify_reg_name|g' "$f"
  sed -i 's|xdna_archspec::aie2::isa::decoder_ffi::parse_reg_name|crate::interpreter::decode::register_map::parse_reg_name|g' "$f"
done
```

- [ ] **Step 3: Verify zero remaining `crate::tablegen::*` references in consumers**

```bash
rg -l 'use crate::tablegen' src/
```

Expected: only `src/tablegen/mod.rs` itself (self-reference inside the forwarder, pending Task 16 deletion), plus `src/tablegen/register_map.rs` moved in Task 13 (not counted; already gone).

Actually even `src/tablegen/mod.rs` shouldn't have `use crate::tablegen` (it's the module itself). Expected count: 0 files outside `src/tablegen/` (which gets deleted in Task 16).

Confirm:

```bash
rg -l 'crate::tablegen' src/ | grep -v 'src/tablegen/'
```

Expected: empty.

- [ ] **Step 4: Build and test**

```bash
cargo build 2>&1 | tail -15
cargo test --lib 2>&1 | tail -3
```

Expected: clean build, 2797 passed. If any import fails to resolve, the sed pass missed a path -- fix manually.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: rewrite crate::tablegen::* consumers to xdna_archspec

38 consumer files switch from crate::tablegen::* to their new canonical
paths: xdna_archspec::aie2::isa::* for types / resolver / decoder
bytecode / decoder FFI, and crate::interpreter::decode::register_map::*
for MappedOperand / RegisterMap / classify_reg_name.

Generated using Claude Code.
EOF
)"
```

---

### Task 15: Atomic rewrite of `crate::arch::*` consumers

**Goal:** Rewrite all `crate::arch::*` imports to `xdna_archspec::aie2::*`. Reference the Consumer-Import Rewrite Reference table.

**Files:**
- Modify: 36 files containing `crate::arch::...`

- [ ] **Step 1: Enumerate**

```bash
rg -l 'crate::arch' src/ | sort > /tmp/claude-1000/subsys6-arch-consumers-pre-rewrite.txt
wc -l /tmp/claude-1000/subsys6-arch-consumers-pre-rewrite.txt
```

- [ ] **Step 2: Mechanical rewrite**

Helper script:

```bash
# /tmp/claude-1000/subsys6-rewrite-arch.sh
#!/bin/bash
for f in $(cat /tmp/claude-1000/subsys6-arch-consumers-pre-rewrite.txt); do
  # subsystem -> subsystems (singular -> plural; special case because the
  # old forwarder had a compat shim)
  sed -i 's|crate::arch::subsystem::|xdna_archspec::aie2::subsystems::|g' "$f"
  # Everything else is a straight path rewrite
  sed -i 's|crate::arch::|xdna_archspec::aie2::|g' "$f"
done
```

- [ ] **Step 3: Verify zero remaining `crate::arch::*` references**

```bash
rg -l 'crate::arch' src/ | grep -v 'src/lib.rs'
```

Expected: empty (only `src/lib.rs` still has the forwarder block, which Task 16 deletes).

- [ ] **Step 4: Build and test**

```bash
cargo build 2>&1 | tail -10
cargo test --lib 2>&1 | tail -3
```

Expected: clean, 2797 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: rewrite crate::arch::* consumers to xdna_archspec::aie2

36 consumer files switch from crate::arch::* to xdna_archspec::aie2::*,
using subsystems:: (plural) in place of the old subsystem:: compat
shim. The mod arch forwarder in src/lib.rs dissolves in the next task.

Generated using Claude Code.
EOF
)"
```

---

### Task 16: Delete `src/tablegen/`, `mod arch`, `mod tablegen` forwarders

**Goal:** Once consumers no longer reference the forwarders, delete them and the whole `src/tablegen/` directory.

**Files:**
- Delete: `src/tablegen/` (entire directory)
- Modify: `src/lib.rs` (remove `pub mod arch { ... }` and `pub mod tablegen;`)

- [ ] **Step 1: Verify no consumers remain**

```bash
rg -l 'crate::arch|crate::tablegen|use crate::tablegen|use crate::arch' src/
```

Expected: only `src/lib.rs` (forwarder block) and files inside `src/tablegen/`. If any other file matches, Tasks 14/15 missed something.

- [ ] **Step 2: Delete `src/tablegen/` directory**

```bash
git rm -rf src/tablegen
```

- [ ] **Step 3: Remove the forwarder blocks from src/lib.rs**

Edit `src/lib.rs`. Remove the entire `pub mod arch { ... }` block (including the doc comment and the `subsystem` sub-mod). Remove the `pub mod tablegen;` line (if present).

- [ ] **Step 4: Build and test**

```bash
cargo build 2>&1 | tail -10
cargo test --lib 2>&1 | tail -3
```

Expected: clean, 2797 passed. If builds fail, some consumer still uses `crate::arch::` or `crate::tablegen::`. Grep, fix, retry.

- [ ] **Step 5: Verify FFI still works**

```bash
./scripts/emu-bridge-test.sh --no-hw -v add_one 2>&1 | tail -5
```

Expected: Chess 10/10 PASS, Peano 9/9 PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: dissolve mod arch / mod tablegen forwarders

src/tablegen/ directory deleted; pub mod arch and pub mod tablegen
blocks removed from src/lib.rs. Consumers now import directly from
xdna_archspec::aie2::* and crate::interpreter::decode::register_map::*.

Generated using Claude Code.
EOF
)"
```

---

### Task 17: Shrink `xdna-emu/build.rs` + `Cargo.toml` cleanup

**Goal:** Reduce `xdna-emu/build.rs` to the XRT plugin install block + `gen_header`. Remove `xdna-archspec` from xdna-emu's `[build-dependencies]` (the plugin install doesn't need it).

**Files:**
- Modify: `build.rs` (xdna-emu)
- Modify: `Cargo.toml` (xdna-emu)

- [ ] **Step 1: Audit xdna-emu/build.rs current contents**

```bash
rg -n '^fn ' build.rs
wc -l build.rs
```

Expected after Tasks 6, 7, 9, 11: ~100-200 lines, with `main()`, `gen_header()`, and the XRT plugin install block.

- [ ] **Step 2: Remove any now-unused functions**

Anything that's not called from `main()` or by `gen_header` or by the plugin install block is dead. Remove. Helper imports (`use std::collections::HashMap;` etc.) that are now unused get removed too (the Rust compiler warns on unused imports; remove until clean).

- [ ] **Step 3: Update the header doc**

Replace the current header doc in `build.rs` (which explained the Subsystem 1 hybrid state) with the new post-Subsystem-6 state:

```rust
//! Build script for xdna-emu.
//!
//! After Subsystem 6 (ISA Decode), all arch-data codegen and the
//! LLVM MCDisassembler FFI compile live in
//! `crates/xdna-archspec/build.rs`. What remains here:
//!
//! - XRT plugin install (xrt-plugin/ build artifact copy).
//!
//! Generated files (written to `$OUT_DIR/`):
//!   None. (xdna-emu does not generate any Rust source anymore.)
```

- [ ] **Step 4: Remove `xdna-archspec` from `[build-dependencies]` if unused**

Edit `Cargo.toml`. If `use xdna_archspec::...` no longer appears anywhere in `build.rs`, remove the line:

```toml
xdna-archspec = { path = "crates/xdna-archspec" }
```

from `[build-dependencies]`. `xdna-archspec` stays in `[dependencies]` (runtime dep).

After this, `[build-dependencies]` should be empty or absent. If empty, remove the `[build-dependencies]` section entirely.

- [ ] **Step 5: Verify build**

```bash
cargo clean 2>&1 | tail -3
cargo build 2>&1 | tail -15
cargo build --release 2>&1 | tail -5
cargo test --lib 2>&1 | tail -3
```

Expected: clean builds, 2797 passed.

- [ ] **Step 6: Verify build.rs line count**

```bash
wc -l build.rs
```

Expected: ~80 lines (target from the spec). Anything significantly over suggests leftover dead code.

- [ ] **Step 7: Commit**

```bash
git add build.rs Cargo.toml
git commit -m "$(cat <<'EOF'
refactor: shrink xdna-emu/build.rs to plugin-install only

All codegen and FFI compilation relocated to xdna-archspec in prior
tasks. xdna-emu/build.rs reduces to the XRT plugin install block and
the gen_header helper. [build-dependencies] empties out.

Generated using Claude Code.
EOF
)"
```

---

### Task 18: Write the ISA-decode design note

**Goal:** Per the parent refactor spec's "mandatory per-seam design note" requirement, write `docs/arch/isa-decode.md` explaining why Subsystem 6 adds no trait seam and what the "what would AIE1 look like?" answer is.

**Files:**
- Create: `docs/arch/isa-decode.md`

- [ ] **Step 1: Write the design note**

Create `docs/arch/isa-decode.md`:

```markdown
# ISA Decode -- Design Note

**Subsystem:** 6 (Phase 1b)
**Tag:** `phase1-subsys-isa-decode`
**Spec:** [../superpowers/specs/2026-04-17-subsys6-isa-decode-design.md](../superpowers/specs/2026-04-17-subsys6-isa-decode-design.md)

This document is the mandatory per-seam design note required by the
parent device-family refactor. Subsystem 6 adds no trait seam; this
note explains *why not* for ISA decode specifically, and how AIE1
support fits.

---

## What lives where

All entries below are in `xdna_archspec::aie2::isa` as of the
`phase1-subsys-isa-decode` tag.

| Data/code | Module | Source |
|-----------|--------|--------|
| TableGen runtime types (SlotDef, EncodingPart, FormatClass, InstrDef, TemplateParam, etc.) | `xdna_archspec::aie2::isa::types` | llvm-aie TableGen extraction |
| Operand classification + semantic inference | `xdna_archspec::aie2::isa::resolver` | Heuristics over TableGen data |
| Decoder bytecode walker | `xdna_archspec::aie2::isa::decoder_bytecode` | Data-driven |
| LLVM MCDisassembler FFI (raw) | `xdna_archspec::aie2::isa::decoder_ffi` | llvm-aie LLVM libraries |
| SemanticOp, ImplicitReg, BranchCondition, ElementType, SelectVariant, etc. | `xdna_archspec::aie2::isa` (via `pub use types::*`) | TableGen-extracted + heuristic |
| Generated instruction tables (`gen_tablegen.rs`) | `xdna_archspec::aie2::isa::generated` | build.rs via `build_helpers/` |
| Decoded-operand -> emulator Operand classifier (MappedOperand, RegisterMap, classify_reg_name) | `xdna_emu::interpreter::decode::register_map` | emulator convention |
| Operand enum (execution state) | `xdna_emu::interpreter::bundle::slot` | emulator convention |
| Interpreter register-file indexing constants (LR_REG_INDEX, SP_PTR_INDEX, MOD_BASE_*) | `xdna_emu::interpreter::state` | emulator convention |

---

## The const-first principle, applied to ISA decode

Subsystem 1 established: lift per-arch differences behind traits only when
they are *shape* differences, not *values* differences.

For ISA decode, the delta between AIE2 and hypothetical AIE1 is:

- Different `SlotDef` values (3 slots for AIE1 vs. 6 for AIE2).
- Different `CompositeFormatDef` entries (AIE1 bundles pack 3 slots into a
  different byte layout).
- Different instruction encodings (AIE1 has fewer opcodes; different operand
  formats).
- Different `ProcessorModel` and itineraries.
- Different register classes (AIE1 has different vector-register widths).

Every one of these is a data change. The decoder bytecode walker
(`decoder_bytecode.rs`), the operand classifier (`resolver::operand_classification`),
and the semantic inference logic (`resolver::semantic_inference`) all work
off these data tables without hardcoding slot counts or specific bundle
shapes.

A trait would be justified if an AIE1 instruction required a
fundamentally different decode discipline -- e.g., non-TableGen-sourced
data, or per-operand extraction that doesn't fit the fragment model. It
does not. The shape is invariant; only the values differ.

## What would AIE1 look like?

- `xdna_archspec::aie1::isa::` module, mirroring `aie2::isa::` structure.
- Its own `build_helpers/` fed from AIE1's TableGen sources (a different
  llvm-aie directory).
- Its own `decoder_ffi/` linked against an AIE1-configured LLVM build.
- The same `types`, `resolver`, `decoder_bytecode` algorithmic code -- but
  the current archspec types genuinely do work for both. If the types
  themselves need to change (e.g., to fit a 3-slot `SlotIndex` variant that
  doesn't exist for AIE2), we'd extend the enum, not copy the code.

## Where a trait could enter

The hot-path `IsaDecoder` trait candidate:

```rust
pub trait IsaDecoder {
    fn decode_slot(&self, slot: Slot, bits: u64) -> DecodeResult;
    fn bundle_layout(&self) -> &[CompositeFormatDef];
}
```

Not introduced in Subsystem 6 because no second arch is being populated.
If AIE1 decoder data lands and the `decode_slot` algorithm diverges, this
is where the trait enters. Until then, the const-first choice holds.

---

## What about behavior seams?

- Instruction *execution* semantics (Subsystem 7) almost certainly warrant
  an `IsaExecutor` trait: vector rounding, saturation, configuration-word
  interpretation genuinely varies between arch families.
- Bundle *parsing* (which slot-category gets the next N bits) is data, per
  `CompositeFormatDef`. No trait.
- Register *file layout* (which name maps to which `Operand` variant) is
  emulator convention; stays in xdna-emu.

The trait boundary rides the "behavior differs in shape" criterion. ISA
decode is values. ISA execute is shapes.
```

- [ ] **Step 2: Commit**

```bash
git add docs/arch/isa-decode.md
git commit -m "$(cat <<'EOF'
docs: ISA decode design note

Mandatory per-seam design note. Explains the const-first rationale
(Subsystem 6 adds no trait), what would change for AIE1 support, and
where a future IsaDecoder trait would enter.

Generated using Claude Code.
EOF
)"
```

---

### Task 19: Part B verification gate, tag, and NEXT-STEPS update

**Goal:** Run the full HW + ISA suites against Part B state. Update the audit with Part B completion. Update `NEXT-STEPS.md` to point at Subsystem 2. Tag `phase1-subsys-isa-decode`.

**Files:**
- Modify: `docs/arch/subsys6-audit.md`
- Modify: `NEXT-STEPS.md`

- [ ] **Step 1: Full verification**

```bash
cargo build --release 2>&1 | tail -5
cargo test --lib 2>&1 | tail -3
cargo test -p xdna-archspec --lib 2>&1 | tail -3
./scripts/emu-bridge-test.sh --no-hw -v add_one 2>&1 | tail -5
```

Expected:
- release build clean
- `2797 passed; 0 failed; 5 ignored`
- `138 passed; 1 failed`
- Chess 10/10, Peano 9/9

- [ ] **Step 2: Full HW bridge run**

```bash
nice -n 19 ./scripts/emu-bridge-test.sh 2>&1 | tee /tmp/claude-1000/subsys6-partB-bridge.log
```

Expected: matches `phase1-subsys-isa-decode-partA` baseline.

- [ ] **Step 3: ISA test suite**

```bash
nice -n 19 ./scripts/isa-test.sh 2>&1 | tee /tmp/claude-1000/subsys6-partB-isa.log
```

Expected: `FAIL: 0 / 4815`.

- [ ] **Step 4: Verify success criteria**

Check each against the spec's Section "Success criteria":

```bash
# 1. Library tests
cargo test --lib 2>&1 | tail -3

# 2. Archspec tests
cargo test -p xdna-archspec --lib 2>&1 | tail -3

# 3. Release build
cargo build --release 2>&1 | tail -5

# 6. Zero crate::arch imports
rg -l 'crate::arch' src/

# 7. Zero crate::tablegen imports
rg -l 'use crate::tablegen' src/

# 8. build.rs line count
wc -l build.rs
```

Expected:
1. 2797 passed
2. 138 passed / 1 pre-existing fail
3. clean
6. empty
7. empty
8. ~80 lines +/- 10

- [ ] **Step 5: Append Part B completion to the audit**

Edit `docs/arch/subsys6-audit.md`. Append:

```markdown
## Part B Completion

Landed 2026-MM-DD. Tag: `phase1-subsys-isa-decode`.

### Commits (from Part A tag through final tag)

<output of `git log --oneline phase1-subsys-isa-decode-partA..HEAD`>

### Verification (at tag)

- `cargo test --lib`: 2797 passed; 0 failed; 5 ignored.
- `cargo test -p xdna-archspec --lib`: 138 passed; 1 failed (pre-existing).
- `cargo build --release`: clean.
- Bridge `--no-hw -v add_one`: Chess 10/10, Peano 9/9.
- Full HW bridge: matches phase1-subsys-isa-decode-partA baseline.
- ISA test suite: FAIL: 0 / 4815.

### Success criteria sweep

- `crate::arch::*` imports in src/: 0.
- `crate::tablegen::*` imports in src/: 0.
- `xdna-emu/build.rs` line count: <n> (target ~80).
- `src/tablegen/` directory: deleted.
- `pub mod arch` + `pub mod tablegen` forwarders: deleted.
- `xdna_archspec::aie2::isa::` populated: types, resolver, decoder_bytecode, decoder_ffi (raw), element_type_logic, generated.
- `xdna_archspec::aie2::aiert::` populated: dma, locks, ports.
- `src/interpreter/decode/register_map.rs` exists with MappedOperand / RegisterMap / classify_reg_name.
- `docs/arch/isa-decode.md` design note exists.
- `ArchConfig` trait surface unchanged (no new trait added in Subsystem 6).

### Follow-ups flagged for subsequent subsystems

None that block Subsystem 2. Candidates for Phase 2 hygiene:
- `src/interpreter/decode/register_map.rs` has ~840 lines. Could split into
  `classify.rs` (parse_reg_name + classify_reg_name) and `map.rs` (RegisterMap
  struct + impl). Not urgent.
- If AIE1 population begins, revisit whether IsaDecoder trait is warranted.
```

- [ ] **Step 6: Rewrite NEXT-STEPS.md**

Edit `NEXT-STEPS.md`. Update the header:

```markdown
# Next Steps -- Device-Family Refactor

**Last updated:** 2026-MM-DD (after Phase 1b Subsystem 6 landed)
**Current branch:** `dev` (no master merges until the refactor is done)
**Latest tag:** `phase1-subsys-isa-decode` at <commit>
```

Update the status table -- mark Subsystem 6 as Done, Subsystem 2 as Up next.

Rewrite the "How to Pick Up Subsystem 2" section:

```markdown
## How to Pick Up Subsystem 2 (Tile Topology)

This is the concrete next action. Start here in a fresh session.

1. **Read the key artifacts:**
   - `docs/superpowers/specs/2026-04-16-device-family-refactor-design.md` (parent)
   - `docs/superpowers/plans/2026-04-16-device-family-refactor-plan.md` (parent plan)
   - `docs/arch/phase1a-audit.md` (Subsystem 2 follow-ups flagged there)
   - `docs/arch/subsys6-audit.md` (most recent state)

2. **Verify current state:**
   ```bash
   git log --oneline phase1-subsys-isa-decode..HEAD
   cargo test --lib 2>&1 | tail -5
   ```
   Expect 2797 passed.

3. **Invoke brainstorming:**
   Topic: "Phase 1b Subsystem 2: Tile Topology, plus TileType -> TileKind
   deep rename".

   Key questions for the spec to shape:
   - Is the runtime-vs-archspec TileType/TileKind split sustainable, or is
     it time to collapse them?
   - How many `row == 0` and `row >= N` checks exist? (Tile classification
     should drive these from ArchModel.)
   - Shim/ShimNoc merge -- does any code actually produce a ShimPl? If not,
     Shim-as-one-variant is fine.
   - MemTile -> Mem rename: is "Mem" sufficiently unambiguous given that
     `Memory` is common?

4. **Invoke writing-plans -> subagent-driven-development** as in prior
   subsystems.

5. **At end of Subsystem 2:** tag `phase1-subsys-tile-topo`, update audit
   + NEXT-STEPS.md.
```

- [ ] **Step 7: Commit the audit + NEXT-STEPS updates**

```bash
git add docs/arch/subsys6-audit.md NEXT-STEPS.md
git commit -m "$(cat <<'EOF'
docs: subsys6 Part B completion log + NEXT-STEPS pickup guide

Part B landed: consumer rewrites, forwarder dissolution, build.rs
shrinkage complete. Success criteria sweep recorded. NEXT-STEPS now
points at Subsystem 2 (Tile Topology) with key questions pre-shaped.

Generated using Claude Code.
EOF
)"
```

- [ ] **Step 8: Tag**

```bash
git tag phase1-subsys-isa-decode -m "Phase 1b Subsystem 6: ISA decode relocation and consumer cleanup"
```

---

## Appendix A: Rollback procedure (per-task)

If any Part A task breaks compilation or tests in a way that can't be
fixed in-place within 15 minutes:

```bash
# Identify the last good commit
git log --oneline

# Revert to the last good commit (soft reset keeps your work on disk)
git reset --soft HEAD~1

# Or hard reset if the changes are fully known-bad
git reset --hard <last-good-sha>
```

Never fix Part A breakage by moving forward. Part A's "forwarders keep
everything compiling" property is the safety rail; if it breaks,
something is wrong with the forwarder, not with downstream consumers.
Fix the forwarder first.

---

## Appendix B: Codegen string-path verification

The decoder table generator emits `use super::types::*` / `use super::resolver::*`
strings (after Task 6's edits). These strings must resolve inside archspec's
module tree.

To verify at any point:

```bash
cat $(find target/debug/build -name gen_tablegen.rs -path '*xdna-archspec*' | head -1) | head -20
```

Expected top lines (after Task 7):

```rust
// Auto-generated by build.rs -- do not edit.
// Source: llvm-aie TableGen extraction at build time.

mod slot_alu { include!(concat!(env!("OUT_DIR"), "/gen_tblgen_slot_alu.rs")); }
... (etc.)

/// Load instruction decoder data from build-time extracted constants.
pub(crate) fn load_from_generated() -> super::types::TblgenOutput {
    use super::types::*;
    use super::resolver::*;
    use super::decoder_bytecode::DecoderTable;
    use std::collections::HashMap;
    ...
}
```

If paths are wrong the build breaks at `cargo build`. Fix by editing
`crates/xdna-archspec/build_helpers/codegen.rs` string literals, not by
adding forwarders.

---

## Appendix C: Known risks

1. **LLVM library path resolution** (Tasks 7, 9). The `llvm-config` invocation
   needs to find the same LLVM build both crates used. If
   `LLVM_AIE_PATH` was set for xdna-emu, archspec's build.rs must also see
   it. Both build scripts should call `println!("cargo:rerun-if-env-changed=LLVM_AIE_PATH");`.

2. **FFI symbol collision** (Task 9). When `compile_llvm_decoder_ffi` runs
   from archspec instead of xdna-emu, the `aie2_decode_slot` / `aie2_opcode_name`
   symbols are still emitted into a static lib. Cargo handles cross-crate
   linking for `cargo:rustc-link-lib=static=aie2_decoder`, but double-check
   by looking at the binary output with `nm target/debug/xdna-emu | grep aie2_`
   after Task 9.

3. **Sed over-matching** (Tasks 14, 15). The sed patterns are written to
   match `crate::tablegen::` and `crate::arch::` as literal prefixes. Risks:
   matches inside string literals or comments. Grep after rewrite to
   confirm no `"crate::tablegen"` strings remain accidentally:
   ```bash
   rg '"crate::(tablegen|arch)' src/
   ```
   If any match, manually revert those specific lines.

4. **Forwarder self-reference** (Tasks 3-11). Each forwarder file
   (`src/tablegen/types.rs` etc.) uses `pub use xdna_archspec::...`. If
   archspec's build fails for any reason (Task 7's TableGen pipeline
   breaking), xdna-emu won't compile even though its own source didn't
   change. Recovery: fix archspec first, then retry xdna-emu build.

5. **Bridge test register-name mapping** (Task 10). After the split,
   `DecodedOperand::Reg { id, name }` comes from archspec's FFI and feeds
   xdna-emu's `classify_reg_name`. If the `name` string differs (e.g.,
   due to an LLVM library version mismatch), the classifier will miss.
   `./scripts/emu-bridge-test.sh --no-hw -v add_one` catches this
   regression within 30s.
