# Subsystem 1 -- Registers & Memory Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move all AIE2 arch-data codegen, build_helpers, and the LLVM MCDisassembler FFI out of `xdna-emu/build.rs` and into the `xdna-archspec` crate under the `xdna_archspec::aie2::*` namespace, then tighten the register/memory-map slice specifically by moving derived consts from `src/device/registers_spec.rs` into `xdna_archspec::aie2::memory_map`.

**Architecture:** Two-part subsystem. Part A is a pure relocation (bit-identical codegen output, import rewrites across ~37 files); Part B is narrow semantic tightening for registers and memory map. The `xdna-archspec` crate gains its own `build.rs` that uses `#[path]` includes of its own source modules to avoid a self-dependency. The `ArchConfig` trait surface does not grow in this subsystem; the const-first principle is preserved.

**Tech Stack:** Rust 2021 workspace, `xdna-archspec` workspace crate, `#[path]` build-script module includes, `tblgen` (LLVM TableGen via Rust bindings), `cc` crate for C++ compile, LLVM 21 MCDisassembler via FFI.

**Spec:** [docs/superpowers/specs/2026-04-17-subsys1-regs-mem-design.md](../specs/2026-04-17-subsys1-regs-mem-design.md)

**Parent refactor:** [docs/superpowers/specs/2026-04-16-device-family-refactor-design.md](../specs/2026-04-16-device-family-refactor-design.md)

---

> **Sweep-as-of 2026-05-01:** Subsystem 1 completed -- tag `phase1-subsys-regs-mem`. Audit + completion log in docs commits leading up to the tag. Steps below were executed organically rather than ticked one-by-one; this sweep flips the checkboxes to match the verified completion state.


## Scope Note

Part A (Tasks 1-14) is a relocation of all of `xdna-emu`'s build-time arch-data machinery into `xdna-archspec`. Bit-identical outputs; failure in Part A is always a missed import or a path mismatch, never a semantic regression.

Part B (Tasks 15-19) is the Subsystem 1 *proper* semantic work: derived consts for the register/memory-map slice migrate into `xdna_archspec::aie2::memory_map`, and `src/device/registers_spec.rs` dissolves.

Branch: `dev`. Per-subsystem tag at end of Part A: `phase1-subsys-regs-mem-partA`. Per-subsystem tag at end of Part B: `phase1-subsys-regs-mem`.

---

## Global Invariants (every task, every commit)

- `cargo test --lib` green.
- `cargo test -p xdna-archspec --lib` green.
- `cargo build` green (release build required before each tag, not every commit).
- `./scripts/emu-bridge-test.sh --no-hw -v add_one` green (fast smoke, ~30 s).
- Full bridge HW run + `./scripts/isa-test.sh` green at each tag with no regressions vs. pre-Phase-1a baseline.
- No commit introduces `TODO`/`FIXME`/`unimplemented!()` without an open-issue reference.
- Commit messages: lowercase type prefix (`refactor:`, `docs:`, `test:`, `build:`); no emoji; ends with "Generated using Claude Code.".
- All work on `dev`. No merges to `master` during this plan.

---

## Baseline to Preserve

Before Task 1, capture the current numbers so later regression checks have a target:

```bash
cargo test --lib 2>&1 | tail -3
cargo test -p xdna-archspec --lib 2>&1 | tail -3
./scripts/emu-bridge-test.sh --no-hw -v add_one 2>&1 | tail -5
```

Record in `docs/arch/subsys1-audit.md` (created in Task 1) under `## Baseline`. Expected current values:
- Library tests: `2798 passed; 0 failed; 5 ignored`
- Archspec tests: `138 passed; 1 failed` (`test_full_parse_all_devices`, pre-existing)
- Bridge smoke: Chess 10/10 PASS, Peano 9/9 PASS

---

## File Structure

**Current layout:**

```
xdna-emu/
├── build.rs                    # ~2240 lines; all codegen + FFI compile + plugin install
├── build_helpers/              # AIE2 TableGen extractor (8 files, ~120 KB)
├── decoder_ffi/                # aie2_decoder.cpp + .h + .inc files
├── src/
│   ├── lib.rs                  # has `pub mod arch { include!(...) } ` block at lines 60-99
│   └── device/
│       └── registers_spec.rs   # derived memory-map consts + include!()s from gen_core_module / gen_memory_lock / gen_memtile_lock
├── Cargo.toml                  # build-deps: serde_json, xdna-archspec, tblgen, cc
└── crates/xdna-archspec/
    ├── Cargo.toml              # no build.rs
    └── src/
        ├── lib.rs              # build_arch_model + confirm_subsystem_ranges + populate_manual_constants inline (~500 lines)
        ├── types.rs            # arch-agnostic; no crate-local imports
        ├── device_model.rs     # uses `use crate::types::*`
        ├── regdb.rs            # no crate-local imports
        ├── regdb_extractor.rs  # uses `use crate::{regdb, types::*}`
        ├── tablegen.rs         # only `use std::path::Path`
        └── runtime.rs          # ArchConfig + ModelConfig (stays lib-only)
```

**Target layout after Part A:**

```
xdna-emu/
├── build.rs                    # XRT plugin install only (~80 lines)
├── (build_helpers/ deleted)
├── (decoder_ffi/ deleted)
├── src/
│   ├── lib.rs                  # no `pub mod arch` block
│   └── device/
│       └── registers_spec.rs   # unchanged from Part A's perspective; Part B dissolves it
├── Cargo.toml                  # build-deps: (none); xdna-archspec stays as regular dep
└── crates/xdna-archspec/
    ├── build.rs                # drives all codegen + C++ FFI compile
    ├── build_helpers/          # moved from xdna-emu
    ├── decoder_ffi/            # moved from xdna-emu
    ├── Cargo.toml              # build-deps: serde_json, tblgen, cc; build = "build.rs"
    └── src/
        ├── lib.rs              # re-exports only (model_builder factored out)
        ├── model_builder.rs    # (new) build_arch_model + confirm_subsystem_ranges + populate_manual_constants
        ├── types.rs            # unchanged
        ├── device_model.rs     # unchanged
        ├── regdb.rs            # unchanged
        ├── regdb_extractor.rs  # unchanged
        ├── tablegen.rs         # unchanged
        ├── runtime.rs          # unchanged
        └── aie2/
            ├── mod.rs          # include!() for gen_arch.rs + gen_stream_ports.rs; pub mod declarations; port_type hand-written consts
            ├── registers.rs    # (new) include!()s for gen_core_module.rs + gen_memory_lock.rs + gen_memtile_lock.rs
            ├── subsystems.rs   # (new) include!() for gen_subsystems.rs
            ├── stream_switch.rs # (new) include!() for gen_stream_ranges.rs
            ├── trace_events.rs # (new) include!() for gen_trace_events.rs
            ├── isa/
            │   └── decoder_tables.rs  # (new) include!() for gen_tablegen.rs
            └── decoder_ffi.rs  # (new) extern "C" declarations for aie2_decoder.cpp
```

**After Part B:**

- `aie2/memory_map.rs` added, hand-written, owns derived memory-map consts (AIE_DATA_MEMORY_BASE, PROGRAM_MEMORY_BASE, DATA_MEMORY_BASE, COMPUTE_DATA_MEMORY_END, MEM_TILE_DATA_MEMORY_END, PROGRAM_MEMORY_END).
- `src/device/registers_spec.rs` deleted; `sign_extend_7bit` inlined at call sites or moved to `src/device/bit_utils.rs`.
- `docs/arch/registers-memory-map.md` created.

---

## Build-Script Self-Reference Strategy

When `build.rs` moves into `xdna-archspec`, it cannot declare `xdna-archspec` as its own build-dep. The workaround: `#[path = "src/foo.rs"] mod foo;` includes the module source directly in the build-script compilation. This works for any module whose cross-module imports resolve against `crate::` (which, in build.rs context, means the build-script binary's own module tree).

The archspec modules satisfy this constraint: `regdb_extractor` uses `use crate::regdb` and `use crate::types::*`; when build.rs declares `#[path = "src/regdb.rs"] mod regdb;` and `#[path = "src/types.rs"] mod types;` before the include of `regdb_extractor.rs`, those `crate::` paths resolve to the same modules.

The only blocker is that `lib.rs`'s `build_arch_model`, `confirm_subsystem_ranges`, and `populate_manual_constants` are inline in `lib.rs` — a build.rs include of `lib.rs` would drag in `pub mod runtime;` and its `Arc`/`LazyLock` dependencies. **Task 2 factors them into `model_builder.rs`** so they become includable without `runtime.rs`.

---

## Part A -- Infrastructure Relocation

**Tag at end:** `phase1-subsys-regs-mem-partA`

---

### Task 1: Audit

**Files:**
- Create: `docs/arch/subsys1-audit.md`

- [x] **Step 1: Create directory if missing and seed audit doc**

```bash
mkdir -p docs/arch
touch docs/arch/subsys1-audit.md
```

- [x] **Step 2: Capture baseline test numbers**

Run and record the output of each:

```bash
cargo test --lib 2>&1 | tail -3
cargo test -p xdna-archspec --lib 2>&1 | tail -3
./scripts/emu-bridge-test.sh --no-hw -v add_one 2>&1 | tail -5
```

Append to `docs/arch/subsys1-audit.md`:

```markdown
# Subsystem 1 -- Registers & Memory Map Audit

## Baseline (pre-subsystem)

- `cargo test --lib`: <paste output>
- `cargo test -p xdna-archspec --lib`: <paste output>
- Bridge `--no-hw -v add_one`: <paste output>

Failures to carry through: `test_full_parse_all_devices` (archspec, pre-existing,
device count 13 vs expected 12 -- unrelated).
```

- [x] **Step 3: Enumerate `crate::arch::*` consumers**

```bash
rg -l 'crate::arch' src/ examples/ tests/ xrt-plugin/ 2>&1 | sort > /tmp/claude-1000/subsys1-arch-consumers.txt
wc -l /tmp/claude-1000/subsys1-arch-consumers.txt
```

Append to the audit under `## crate::arch Consumers`. Expected count: ~37 files under `src/`, plus whatever shows up outside `src/`. Any file outside `src/` is a hidden consumer the spec's risk section calls out.

- [x] **Step 4: Enumerate codegen include sites**

```bash
rg -n 'include!\(concat!\(env!\("OUT_DIR"\)' src/ build.rs
```

Append to audit under `## Codegen Include Sites`. Expected locations:
- `src/lib.rs` -- `mod arch { include!(gen_arch.rs); pub mod subsystem { include!(gen_subsystems.rs); } include!(gen_stream_ports.rs); pub mod stream_switch { include!(gen_stream_ranges.rs); } }`
- `src/device/registers_spec.rs` -- three `include!()` sites inside `memory_module`, `core_module`, `mem_tile_module` submodules
- `src/interpreter/decode/` (likely) -- for `gen_tablegen.rs`
- `src/device/trace/` or similar -- for `gen_trace_events.rs`

Record the exact file:line for each.

- [x] **Step 5: Enumerate build.rs codegen functions and call sites**

```bash
rg -n '^fn gen_' build.rs
rg -n '^fn extract_aiert' build.rs
rg -n 'compile_llvm_decoder_ffi' build.rs
```

Append to audit under `## build.rs Codegen Functions`. Each function with its line range.

- [x] **Step 6: Enumerate `sign_extend_7bit` call sites (for Part B)**

```bash
rg -n 'sign_extend_7bit' src/
```

Append to audit under `## sign_extend_7bit Call Sites`. Expected: a small number; informs Part B Task 17's inline-vs-move decision.

- [x] **Step 7: Enumerate `registers_spec` consumers (for Part B)**

```bash
rg -n 'device::registers_spec|registers_spec::' src/
```

Append to audit under `## registers_spec.rs Consumers`.

- [x] **Step 8: Commit the audit**

```bash
git add docs/arch/subsys1-audit.md
git commit -m "$(cat <<'EOF'
docs: subsys1 regs & memory-map audit

Baseline test numbers, crate::arch consumer enumeration, codegen
include-site inventory, build.rs function inventory, and
registers_spec / sign_extend_7bit call-site data. Guides Part A's
file-by-file relocation and Part B's tighter semantic work.

Generated using Claude Code.
EOF
)"
```

---

### Task 2: Factor `model_builder` out of `lib.rs`

**Goal:** Make `build_arch_model`, `confirm_subsystem_ranges`, and `populate_*_manual_constants` live in a module that can be `#[path]`-included by `xdna-archspec`'s future `build.rs` without pulling in `runtime::`.

**Files:**
- Create: `crates/xdna-archspec/src/model_builder.rs`
- Modify: `crates/xdna-archspec/src/lib.rs`

- [x] **Step 1: Create `model_builder.rs` with the moved content**

Cut from `crates/xdna-archspec/src/lib.rs`: the functions `build_arch_model`, `confirm_subsystem_ranges`, `populate_manual_constants`, `populate_aie2_manual_constants`, and the `#[cfg(test)] mod tests { ... }` block. Paste into `crates/xdna-archspec/src/model_builder.rs`. Top of the new file:

```rust
//! ArchModel construction and cross-validation.
//!
//! This module is `#[path]`-included by this crate's `build.rs` so
//! code generation at build time can call the same model-building
//! functions the library exposes at runtime. Because build.rs
//! cannot declare its own crate as a dependency, the module must
//! compile standalone (relying only on `crate::{types, regdb,
//! regdb_extractor, device_model}` and std/serde).
//!
//! Do not add dependencies here on `crate::runtime` or any other
//! module that uses `Arc`, `LazyLock`, or other lib-only items.

use crate::{device_model, regdb_extractor, types};
use std::path::Path;

// <paste build_arch_model + confirm_subsystem_ranges + populate_* here,
//  with `use` paths updated to `use crate::types::*` etc. as needed>

// <paste #[cfg(test)] mod tests { ... } here>
```

- [x] **Step 2: Slim `lib.rs` to re-exports**

`crates/xdna-archspec/src/lib.rs` becomes:

```rust
//! NPU Architecture Specification -- validated hardware model.
//!
//! Extracts hardware architecture from the open-source NPU toolchain
//! (aie-rt, AM025 JSON, device model) into a single typed Rust model.
//! Multi-architecture: each `ArchModel` represents one architecture
//! (AIE, AIE2, AIE2P) with all its tile types, registers, and
//! relationships.
//!
//! This crate is a workspace member of xdna-emu. Its own `build.rs`
//! performs all AIE2 code generation (under `src/aie2/`) and LLVM
//! MCDisassembler FFI compilation. Runtime users import from `runtime`
//! for `ArchConfig`/`ModelConfig` or from `aie2` for generated
//! const data.

pub mod device_model;
pub mod model_builder;
pub mod regdb;
pub mod regdb_extractor;
pub mod runtime;
pub mod tablegen;
pub mod types;

pub use model_builder::{build_arch_model, confirm_subsystem_ranges};
```

- [x] **Step 3: Build and run tests**

```bash
cargo build
cargo test --lib
cargo test -p xdna-archspec --lib
```

Expected: all previously-passing tests still pass (2798 lib + 137 passing archspec tests; `test_full_parse_all_devices` remains the single pre-existing failure).

- [x] **Step 4: Commit**

```bash
git add crates/xdna-archspec/src/lib.rs crates/xdna-archspec/src/model_builder.rs
git commit -m "$(cat <<'EOF'
refactor: factor model_builder out of xdna-archspec lib.rs

Moves build_arch_model, confirm_subsystem_ranges, and
populate_manual_constants out of lib.rs into a new model_builder.rs
module. lib.rs re-exports the public API so existing consumers are
unaffected.

Prep for Task 3: xdna-archspec's future build.rs #[path]-includes
model_builder without pulling in runtime.rs (which depends on Arc
and LazyLock).

Generated using Claude Code.
EOF
)"
```

---

### Task 3: Scaffold `xdna-archspec/build.rs`

**Files:**
- Create: `crates/xdna-archspec/build.rs`
- Modify: `crates/xdna-archspec/Cargo.toml`

- [x] **Step 1: Add `build.rs` field and build-deps to `Cargo.toml`**

Edit `crates/xdna-archspec/Cargo.toml`. The file after edit:

```toml
[package]
name = "xdna-archspec"
version = "0.1.0"
edition = "2021"
description = "NPU architecture specification -- validated hardware model extracted from the open-source toolchain"
license = "MIT"
build = "build.rs"

[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[build-dependencies]
serde_json = "1"

[dev-dependencies]
tempfile = "3"
```

Note: `tblgen` and `cc` will be added in Tasks 10 and 11 respectively; they are not needed until TableGen extraction and the C++ FFI compile move.

- [x] **Step 2: Create empty `build.rs`**

```rust
//! Build script for xdna-archspec.
//!
//! Drives all AIE2 arch-data code generation from the validated
//! ArchModel (device-model + AM025 JSON, cross-validated via
//! Confirmed<T>). Output files land in $OUT_DIR and are included
//! by modules under `src/aie2/`.
//!
//! Because this script lives inside the crate it is generating for,
//! it cannot declare `xdna-archspec` as a build-dep. Module source
//! files are `#[path]`-included so the same types and parsers used
//! at runtime are available at build time.

#[path = "src/types.rs"]
mod types;
#[path = "src/regdb.rs"]
mod regdb;
#[path = "src/device_model.rs"]
mod device_model;
#[path = "src/regdb_extractor.rs"]
mod regdb_extractor;
#[path = "src/tablegen.rs"]
mod tablegen;
#[path = "src/model_builder.rs"]
mod model_builder;

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let _out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Workspace root = crate's parent's parent (crates/xdna-archspec -> crates -> root).
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("xdna-archspec manifest has no grandparent (expected <workspace-root>/crates/xdna-archspec)");

    // Resolve AM025 JSON: MLIR_AIE_PATH env var or sibling dir.
    let mlir_aie = env::var("MLIR_AIE_PATH").unwrap_or_else(|_| {
        workspace_root
            .parent()
            .expect("workspace root has no parent")
            .join("mlir-aie")
            .to_string_lossy()
            .to_string()
    });
    let am025_path = Path::new(&mlir_aie).join("lib/Dialect/AIE/Util/aie_registers_aie2.json");

    // Device model is in the workspace root.
    let device_model_path = workspace_root.join("tools/aie-device-models.json");

    // Rebuild triggers.
    println!("cargo:rerun-if-changed={}", am025_path.display());
    println!("cargo:rerun-if-changed={}", device_model_path.display());
    println!("cargo:rerun-if-env-changed=MLIR_AIE_PATH");
    println!("cargo:rerun-if-env-changed=LLVM_AIE_PATH");
    println!("cargo:rerun-if-changed=build.rs");

    // Tasks 4-11 add actual codegen calls here.

    // Silence unused warnings until the codegen steps land.
    let _ = (am025_path, device_model_path);
}
```

- [x] **Step 3: Verify the crate builds**

```bash
cargo build -p xdna-archspec
```

Expected: clean build. If `#[path]`-included modules fail to compile, the fix is almost always resolving a `crate::` path that should be relative, or a module that implicitly relies on `runtime.rs` types.

- [x] **Step 4: Verify workspace build still works**

```bash
cargo build
cargo test --lib
```

Expected: clean. `xdna-archspec`'s `build.rs` runs during the workspace build but does nothing yet; outputs are unchanged.

- [x] **Step 5: Commit**

```bash
git add crates/xdna-archspec/Cargo.toml crates/xdna-archspec/build.rs
git commit -m "$(cat <<'EOF'
build: scaffold xdna-archspec/build.rs

Empty build script with #[path] includes of types, regdb,
regdb_extractor, device_model, tablegen, and model_builder so the
script shares source with the lib (avoiding a self-dep). Rebuild
triggers wired for AM025 JSON, device-model JSON, and MLIR_AIE_PATH /
LLVM_AIE_PATH env vars.

Tasks 4-11 will add actual codegen into this file.

Generated using Claude Code.
EOF
)"
```

---

### Task 4: Move `gen_arch.rs` generation

This is the largest single codegen move -- `gen_arch.rs` produces the topology, memory, cardinal, timing, packet, ctrl_packet, fot, and processor submodules that appear today under `crate::arch::*`.

**Files:**
- Modify: `crates/xdna-archspec/build.rs`
- Create: `crates/xdna-archspec/src/aie2/mod.rs`
- Modify: `crates/xdna-archspec/src/lib.rs` (add `pub mod aie2;`)
- Modify: `build.rs` (xdna-emu, delete `gen_arch` and related logic)
- Modify: `src/lib.rs` (xdna-emu, start dismantling `mod arch`)
- Modify: numerous consumer files (import rewrites)

- [x] **Step 1: Copy `gen_arch` and `extract_aiert` from xdna-emu/build.rs into archspec/build.rs**

Open `build.rs` (xdna-emu). Locate `fn gen_arch` (~line 297) and `fn extract_aiert` (search with `rg -n '^fn extract_aiert'` -- likely above or below gen_arch). Copy the bodies into `crates/xdna-archspec/build.rs`, replacing `xdna_archspec::` prefixes with the local `crate::` equivalents (e.g., `xdna_archspec::types::ArchModel` -> `crate::types::ArchModel`). Also copy the `gen_header` helper.

Call site in archspec's `build.rs::main()`:

```rust
let regdb = regdb::RegisterDb::from_file(&am025_path).unwrap_or_else(|e| {
    panic!("Cannot load AM025 register database at {}:\n  {}", am025_path.display(), e)
});
let mut arch_model = model_builder::build_arch_model(&device_model_path, &regdb, "npu1")
    .unwrap_or_else(|e| panic!("Failed to build ArchModel: {}", e));

extract_aiert(&workspace_root, &_out_dir, &mut arch_model);
// (LLVM slot-width confirmation moves here in Task 10.)
gen_arch(&arch_model, &_out_dir);
```

(Remove the leading underscore from `_out_dir` once it's used.)

- [x] **Step 2: Create `src/aie2/mod.rs` with `include!()`**

```rust
//! AIE2 (NPU1 Phoenix / NPU2+ Strix) architecture constants.
//!
//! Every `pub const` that reflects AIE2 hardware data -- register
//! offsets, memory sizes, per-tile-type resource counts, stream switch
//! port layouts, ISA encodings, timing constants, FoT values, event
//! IDs -- lives under this module. Multi-arch support is planned as
//! sibling modules (e.g., `xdna_archspec::aie1`, `xdna_archspec::aie2p`)
//! that would mirror this namespace with different values.
//!
//! This module's submodules are either `include!()` of
//! build.rs-generated files in $OUT_DIR or hand-written constants that
//! the toolchain does not yet emit machine-readably.

// Generated by build.rs from gen_arch(). Provides:
//   - top-level consts: COLUMNS, ROWS, NUM_MEM_TILE_ROWS, MAX_LOCK_VALUE,
//     MIN_LOCK_VALUE, TILE_COL_SHIFT, TILE_ROW_SHIFT, TILE_OFFSET_MASK,
//     TILE_ROW_BITS, TILE_COL_BITS, SHIM_ROW, COMPUTE_ROW_START,
//     DATA_MEM_HOST_OFFSET.
//   - pub mod compute, memtile, shim: per-tile-type constants.
//   - pub mod cardinal: direction indices (4-7) for core data memory addressing.
//   - pub mod timing, packet, ctrl_packet, fot, processor.
include!(concat!(env!("OUT_DIR"), "/gen_arch.rs"));

/// Stream switch port type identifier (xdna-emu convention, not hardware).
///
/// Each port index in the port arrays maps to one of these types. This
/// is a hand-written set of constants because it is an encoding
/// convention, not an AM025 register value.
pub mod port_type {
    pub const CORE: u8 = 0;
    pub const FIFO: u8 = 1;
    pub const TRACE: u8 = 2;
    /// Tile control port (configuration/debug).
    /// Per AM025: compute=port 3, memtile=port 6, shim=port 0.
    pub const CTRL: u8 = 3;
    pub const NORTH_BASE: u8 = 10;
    pub const SOUTH_BASE: u8 = 20;
    pub const EAST_BASE: u8 = 30;
    pub const WEST_BASE: u8 = 40;
    pub const DMA_BASE: u8 = 50;

    pub const fn north(n: u8) -> u8 { NORTH_BASE + n }
    pub const fn south(n: u8) -> u8 { SOUTH_BASE + n }
    pub const fn east(n: u8) -> u8 { EAST_BASE + n }
    pub const fn west(n: u8) -> u8 { WEST_BASE + n }
    pub const fn dma(n: u8) -> u8 { DMA_BASE + n }
}
```

- [x] **Step 3: Add `pub mod aie2;` to `crates/xdna-archspec/src/lib.rs`**

After `pub mod types;`:

```rust
pub mod aie2;
```

- [x] **Step 4: Build the archspec crate in isolation to verify gen_arch produces parseable output**

```bash
cargo build -p xdna-archspec
```

Expected: clean. If `gen_arch.rs` output fails to parse, the codegen function has a bug (unlikely since it's the same code that worked in xdna-emu/build.rs).

- [x] **Step 5: Rewrite consumers of `crate::arch::*` to `xdna_archspec::aie2::*`**

Use the audit's consumer list from Task 1 Step 3 as the work queue. For each file, replace `crate::arch::` with `xdna_archspec::aie2::`. The rewrites are mechanical; `sed` is appropriate here:

```bash
while read -r f; do
    [ -f "$f" ] || continue
    sed -i 's|crate::arch::|xdna_archspec::aie2::|g' "$f"
done < /tmp/claude-1000/subsys1-arch-consumers.txt
```

(The consumer list should include `src/lib.rs` -- that file gets a separate hand-edit in Step 6. For now `sed` over it changes its internal `mod arch` content too, which we are about to delete, so the collateral damage is fine.)

- [x] **Step 6: Remove the `mod arch` block in `xdna-emu/src/lib.rs`**

Edit `src/lib.rs`. Delete lines 49-99 (the doc comment + `pub mod arch { ... }` block). Keep the `pub mod` declarations above and the conditional feature modules below.

- [x] **Step 7: Delete `gen_arch` / `extract_aiert` / `gen_header` from `xdna-emu/build.rs`**

Locate and delete: `fn gen_arch` (the full body), `fn extract_aiert` (the full body), `fn gen_header` (if it's still only used by the removed functions -- keep it if other `gen_*` still call it; it'll be removed incrementally through Tasks 5-8), and the call sites of those functions in `main()`. Also delete the `xdna_archspec::build_arch_model` call that feeds them, since the model is now built in archspec's build.rs.

After Task 4, xdna-emu's build.rs no longer constructs an `ArchModel`; subsequent Tasks 5-9 delete the other `gen_*` functions and their dependencies one-by-one. Step 7 here deletes only what this task's move renders dead.

- [x] **Step 8: Build workspace**

```bash
cargo build
```

Expected: clean. If a consumer file was missed, the compiler will flag `unresolved import crate::arch::X`. For each such error, rewrite the import to `xdna_archspec::aie2::X` and rebuild.

- [x] **Step 9: Run tests**

```bash
cargo test --lib
cargo test -p xdna-archspec --lib
```

Expected: same pass counts as baseline.

- [x] **Step 10: Bridge smoke**

```bash
./scripts/emu-bridge-test.sh --no-hw -v add_one
```

Expected: Chess 10/10 PASS, Peano 9/9 PASS.

- [x] **Step 11: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: move gen_arch + extract_aiert into xdna-archspec

Relocates the biggest codegen function (topology, memory, cardinal,
timing, packet, ctrl_packet, fot, processor submodules) and the
aie-rt extractor out of xdna-emu/build.rs into
xdna-archspec/build.rs. Outputs land in the archspec crate's
$OUT_DIR and are include!()d by src/aie2/mod.rs.

All consumers of crate::arch::* rewritten to xdna_archspec::aie2::*.
The `mod arch { ... }` block in xdna-emu/src/lib.rs deleted for the
portion sourced from gen_arch; remaining include!()s (subsystems,
stream_switch) stay until their own task moves them.

Hand-written `pub mod port_type` constants moved into
xdna_archspec::aie2::port_type (convention-level, not AM025-sourced).

Values returned are bit-identical to pre-refactor.

Generated using Claude Code.
EOF
)"
```

---

### Task 5: Move `gen_subsystems.rs` generation

**Files:**
- Modify: `crates/xdna-archspec/build.rs`
- Create: `crates/xdna-archspec/src/aie2/subsystems.rs`
- Modify: `crates/xdna-archspec/src/aie2/mod.rs`
- Modify: `build.rs` (xdna-emu)

- [x] **Step 1: Copy `gen_subsystems` + `subsystem_mod_name` helper into archspec/build.rs**

From xdna-emu/build.rs (~line 583 `subsystem_mod_name` and line 619 `gen_subsystems`). Paste into archspec/build.rs, updating `xdna_archspec::types::...` to `crate::types::...`.

Add the call in archspec/build.rs's `main()` after the `gen_arch` call:

```rust
gen_subsystems(&arch_model, &_out_dir);
```

- [x] **Step 2: Create `src/aie2/subsystems.rs`**

```rust
//! Per-tile-type subsystem address ranges (from ArchModel).
//!
//! Submodules: compute, memtile, shim. Each contains `pub mod <subsystem>`
//! with `OFFSET_START` and `OFFSET_END` consts. When a subsystem kind
//! appears in multiple modules within the same tile (e.g., compute has
//! `performance` in both Core and Memory), the module name is prefixed:
//! `core_performance`, `memory_performance`.

include!(concat!(env!("OUT_DIR"), "/gen_subsystems.rs"));
```

- [x] **Step 3: Declare the submodule in `aie2/mod.rs`**

Add after the existing `port_type` module:

```rust
pub mod subsystems;
```

- [x] **Step 4: Rewrite consumer imports**

The previous namespace was `crate::arch::subsystem::*` (singular, nested under `pub mod arch`). After Task 4 those became `xdna_archspec::aie2::subsystem::*`, but the target spec uses the plural `subsystems`. Rewrite:

```bash
rg -l 'xdna_archspec::aie2::subsystem\b' src/ examples/ tests/ xrt-plugin/ 2>/dev/null \
  | xargs -I{} sed -i 's|xdna_archspec::aie2::subsystem\b|xdna_archspec::aie2::subsystems|g' {}
rg -l 'use xdna_archspec::aie2::subsystem;' src/ examples/ tests/ xrt-plugin/ 2>/dev/null \
  | xargs -I{} sed -i 's|use xdna_archspec::aie2::subsystem;|use xdna_archspec::aie2::subsystems;|g' {}
```

- [x] **Step 5: Delete `gen_subsystems` and `subsystem_mod_name` from `xdna-emu/build.rs`**

Also delete their call site in `main()`. Delete the `pub mod subsystem { include!("...gen_subsystems.rs"); }` line if it still exists in `src/lib.rs` (it shouldn't after Task 4, but verify).

- [x] **Step 6: Build + test + bridge smoke**

```bash
cargo build
cargo test --lib
cargo test -p xdna-archspec --lib
./scripts/emu-bridge-test.sh --no-hw -v add_one
```

Expected: baseline values.

- [x] **Step 7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: move gen_subsystems into xdna-archspec

Per-tile-type subsystem address ranges now generated by
xdna-archspec/build.rs and exposed as xdna_archspec::aie2::subsystems
(renamed from the singular `subsystem` to match the spec's
convention).

Generated using Claude Code.
EOF
)"
```

---

### Task 6: Move `gen_core_module.rs` + lock-request generators

**Files:**
- Modify: `crates/xdna-archspec/build.rs`
- Create: `crates/xdna-archspec/src/aie2/registers.rs`
- Modify: `crates/xdna-archspec/src/aie2/mod.rs`
- Modify: `build.rs` (xdna-emu)
- Modify: `src/device/registers_spec.rs`

- [x] **Step 1: Copy `gen_core_module`, `gen_lock_request`, and the three parser helpers**

From xdna-emu/build.rs: `fn gen_core_module` (~line 710), `fn gen_lock_request` (~line 781), `fn parse_lock_end_address` (~line 870), `fn parse_desc_range` (~line 894), `fn parse_desc_single_bit` (~line 913). Paste into archspec/build.rs.

Add to archspec/build.rs's `main()` after `gen_subsystems`:

```rust
gen_core_module(&regdb, &_out_dir);
gen_lock_request(&regdb, &_out_dir, "memory", "gen_memory_lock.rs");
gen_lock_request(&regdb, &_out_dir, "memory_tile", "gen_memtile_lock.rs");
```

- [x] **Step 2: Create `src/aie2/registers.rs`**

```rust
//! AIE2 register offset constants from the AM025 JSON.
//!
//! Submodules:
//!   - (top-level): core module register offsets (CORE_CONTROL, etc.)
//!   - `memory`: memory-module Lock_Request bitfield constants
//!   - `mem_tile`: mem-tile-module Lock_Request bitfield constants

include!(concat!(env!("OUT_DIR"), "/gen_core_module.rs"));

/// Memory-module register constants for compute tiles.
pub mod memory {
    //! AM025 memory_module Lock_Request bit layout.
    include!(concat!(env!("OUT_DIR"), "/gen_memory_lock.rs"));
}

/// Mem-tile-module register constants.
pub mod mem_tile {
    //! AM025 memory_tile_module Lock_Request bit layout.
    include!(concat!(env!("OUT_DIR"), "/gen_memtile_lock.rs"));
}
```

- [x] **Step 3: Declare the module in `aie2/mod.rs`**

```rust
pub mod registers;
```

- [x] **Step 4: Replace `include!()` sites in `src/device/registers_spec.rs` with `pub use` re-exports**

`registers_spec.rs` currently includes the three generated files inside local `pub mod memory_module`, `core_module`, `mem_tile_module` blocks. Replace each include with a re-export from the archspec crate.

Before:
```rust
pub mod memory_module {
    include!(concat!(env!("OUT_DIR"), "/gen_memory_lock.rs"));
    // ... other hand-written content ...
}
pub mod core_module {
    include!(concat!(env!("OUT_DIR"), "/gen_core_module.rs"));
}
pub mod mem_tile_module {
    include!(concat!(env!("OUT_DIR"), "/gen_memtile_lock.rs"));
}
```

After:
```rust
pub mod memory_module {
    pub use xdna_archspec::aie2::registers::memory::*;
    // ... other hand-written content preserved ...
}
pub mod core_module {
    pub use xdna_archspec::aie2::registers::*;
}
pub mod mem_tile_module {
    pub use xdna_archspec::aie2::registers::mem_tile::*;
}
```

(Part B Task 16 migrates consumers *off* `registers_spec::*` entirely, deleting these re-export shells. Part A keeps them as forwarders.)

- [x] **Step 5: Delete the generator functions and their main() calls from `xdna-emu/build.rs`**

Functions to delete: `gen_core_module`, `gen_lock_request`, `parse_lock_end_address`, `parse_desc_range`, `parse_desc_single_bit`. Also delete their main() call sites.

- [x] **Step 6: Build + test + bridge smoke**

```bash
cargo build
cargo test --lib
cargo test -p xdna-archspec --lib
./scripts/emu-bridge-test.sh --no-hw -v add_one
```

Expected: baseline values.

- [x] **Step 7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: move gen_core_module + lock-request generators into xdna-archspec

AM025 core-module register offsets and memory/mem_tile Lock_Request
bit layouts now generated by xdna-archspec/build.rs, exposed as
xdna_archspec::aie2::registers{, ::memory, ::mem_tile}.

src/device/registers_spec.rs retains its submodule structure for now
but forwards to the archspec crate via pub use. Part B dissolves the
file entirely.

Generated using Claude Code.
EOF
)"
```

---

### Task 7: Move `gen_stream_ports.rs` and `gen_stream_ranges.rs` generation

**Files:**
- Modify: `crates/xdna-archspec/build.rs`
- Modify: `crates/xdna-archspec/src/aie2/mod.rs`
- Create: `crates/xdna-archspec/src/aie2/stream_switch.rs`
- Modify: `build.rs` (xdna-emu)

- [x] **Step 1: Copy the stream-switch codegen functions into archspec/build.rs**

From xdna-emu/build.rs: `fn gen_stream_ports` (~line 946), `fn collect_port_array` (~line 1020), `fn suffix_to_port_type` (~line 1053), `fn write_port_array` (~line 1096), `fn gen_stream_ranges` (~line 1127), `fn write_direction_ranges`, `fn write_bundle_ranges`, `fn find_port_range_flex`, `fn find_port_range`, `fn find_master_enable_bit`. Also the `PortArrayData` and `PortEntry` structs and the `PT_*` constants at the top of xdna-emu/build.rs (lines 38-46). Paste into archspec/build.rs.

Add to archspec/build.rs's `main()`:

```rust
let port_data = gen_stream_ports(&regdb, &_out_dir);
gen_stream_ranges(&regdb, &port_data, &_out_dir);
```

- [x] **Step 2: Include `gen_stream_ports.rs` at the `aie2` root**

`gen_stream_ports.rs` defines top-level consts (`COMPUTE_MASTER_PORTS`, etc.) that callers expect at `xdna_archspec::aie2::COMPUTE_MASTER_PORTS`. Add after `port_type`:

```rust
// Port type arrays generated from AM025 Stream_Switch_*_Config registers.
// Defines COMPUTE_MASTER_PORTS, COMPUTE_SLAVE_PORTS, MEMTILE_MASTER_PORTS,
// MEMTILE_SLAVE_PORTS, SHIM_MASTER_PORTS, SHIM_SLAVE_PORTS.
include!(concat!(env!("OUT_DIR"), "/gen_stream_ports.rs"));

pub mod stream_switch;
```

- [x] **Step 3: Create `src/aie2/stream_switch.rs`**

```rust
//! Stream switch port ranges and configuration bits (from AM025).
//!
//! Submodules `compute`, `mem_tile`, `shim` each contain
//! `NORTH_MASTER_START/END`, `SOUTH_MASTER_START/END`, etc.
//! `ENABLE_BIT` and `SLAVE_SELECT_MASK` live at the module root.

include!(concat!(env!("OUT_DIR"), "/gen_stream_ranges.rs"));
```

- [x] **Step 4: Rewrite consumers that reference `xdna_archspec::aie2::stream_switch::*`**

After Task 4's `sed`, consumers already use `xdna_archspec::aie2::stream_switch::...`. Verify no stragglers:

```bash
rg 'crate::arch::stream_switch|crate::arch::COMPUTE_MASTER_PORTS|crate::arch::MEMTILE_MASTER_PORTS|crate::arch::SHIM_MASTER_PORTS|crate::arch::COMPUTE_SLAVE_PORTS|crate::arch::MEMTILE_SLAVE_PORTS|crate::arch::SHIM_SLAVE_PORTS|crate::arch::ENABLE_BIT|crate::arch::SLAVE_SELECT_MASK' src/
```

Expected: no matches. If any appear, hand-edit those files to use the `xdna_archspec::aie2::...` path.

- [x] **Step 5: Delete the generators and their `PT_*` constants from `xdna-emu/build.rs`**

Delete the listed functions, structs, and the top-level `PT_CORE`/`PT_FIFO`/etc. constants.

- [x] **Step 6: Build + test + bridge smoke**

```bash
cargo build
cargo test --lib
cargo test -p xdna-archspec --lib
./scripts/emu-bridge-test.sh --no-hw -v add_one
```

- [x] **Step 7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: move gen_stream_ports + gen_stream_ranges into xdna-archspec

Stream switch port type arrays (COMPUTE_MASTER_PORTS, etc.) and port
ranges (shim/mem_tile/compute submodules with NORTH/SOUTH/EAST/WEST
MASTER/SLAVE START/END + DMA/TRACE bundle ranges) now generated by
xdna-archspec/build.rs. Exposed at xdna_archspec::aie2 (port arrays,
ENABLE_BIT, SLAVE_SELECT_MASK) and xdna_archspec::aie2::stream_switch
(submodules for per-tile-type ranges).

This task only relocates; stream switch internal semantics tighten in
Subsystem 5.

Generated using Claude Code.
EOF
)"
```

---

### Task 8: Move `gen_trace_events.rs` generation

**Files:**
- Modify: `crates/xdna-archspec/build.rs`
- Create: `crates/xdna-archspec/src/aie2/trace_events.rs`
- Modify: `crates/xdna-archspec/src/aie2/mod.rs`
- Modify: `build.rs` (xdna-emu)

- [x] **Step 1: Copy `gen_trace_events` and `write_trace_event_stub` into archspec/build.rs**

From xdna-emu/build.rs: `fn gen_trace_events` (~line 1382) and any helpers it calls (`write_trace_event_stub`, Python-invocation logic). The bridge path resolves relative to the workspace root; the xdna-emu version uses `manifest_dir.parent()` -- archspec's version uses `workspace_root` directly.

Updated call in archspec/build.rs's `main()`:

```rust
let bridge_path = workspace_root.join("tools/mlir-aie-bridge.py");
println!("cargo:rerun-if-changed={}", bridge_path.display());
gen_trace_events(&bridge_path, &_out_dir);
```

- [x] **Step 2: Create `src/aie2/trace_events.rs`**

```rust
//! Trace event codes extracted from mlir-aie.
//!
//! The mlir-aie Python bridge script emits a list of event ID -> name
//! mappings for each tile module, which build.rs parses into a const
//! lookup table.

include!(concat!(env!("OUT_DIR"), "/gen_trace_events.rs"));
```

- [x] **Step 3: Declare the module in `aie2/mod.rs`**

```rust
pub mod trace_events;
```

- [x] **Step 4: Rewrite consumer imports**

The previous path was `crate::arch::trace_events` or similar. Verify what Task 4's `sed` produced:

```bash
rg 'xdna_archspec::aie2::trace_events' src/
```

If consumers were previously referring to a different path (e.g., `crate::arch::trace_event` or a top-level inclusion), hand-edit them to use `xdna_archspec::aie2::trace_events`.

- [x] **Step 5: Delete `gen_trace_events` + `write_trace_event_stub` from xdna-emu/build.rs**

Also delete the `bridge_path` rebuild trigger and the call site from `main()`.

- [x] **Step 6: Build + test + bridge smoke**

```bash
cargo build
cargo test --lib
cargo test -p xdna-archspec --lib
./scripts/emu-bridge-test.sh --no-hw -v add_one
```

- [x] **Step 7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: move gen_trace_events into xdna-archspec

Event IDs from the mlir-aie Python bridge script now generated by
xdna-archspec/build.rs; exposed as xdna_archspec::aie2::trace_events.

The bridge-script invocation path resolves relative to the workspace
root rather than manifest_dir.parent(), which was correct for
xdna-emu but would be wrong for the crates/xdna-archspec location.

Generated using Claude Code.
EOF
)"
```

---

### Task 9: Move `build_helpers/` into `xdna-archspec`

> **Deferred.** This task is not executed in Subsystem 1. The generated
> `gen_tablegen.rs` references `super::super::types::*` and
> `super::super::resolver::*` which resolve to xdna-emu's
> `src/tablegen/{types.rs, resolver/}`. Moving `gen_tablegen` to archspec
> requires moving those types + resolver too -- a restructuring that
> belongs to Subsystem 6 (ISA Decode). Task 9 is deferred there. Tasks 10
> and 11 were also affected -- see their deferral / reduced-scope notes below.

**Files:**
- Move: `build_helpers/` directory
- Modify: `crates/xdna-archspec/Cargo.toml` (add `tblgen` build-dep)
- Modify: `xdna-emu/Cargo.toml` (remove `tblgen` build-dep)
- Modify: `crates/xdna-archspec/build.rs`
- Create: `crates/xdna-archspec/src/aie2/isa/mod.rs`
- Create: `crates/xdna-archspec/src/aie2/isa/decoder_tables.rs`
- Modify: `crates/xdna-archspec/src/aie2/mod.rs`
- Modify: `build.rs` (xdna-emu)

- [x] **Step 1: Move the directory**

```bash
git mv build_helpers crates/xdna-archspec/build_helpers
```

- [x] **Step 2: Add `tblgen` as a build-dep on `xdna-archspec`**

Edit `crates/xdna-archspec/Cargo.toml`'s `[build-dependencies]`:

```toml
[build-dependencies]
serde_json = "1"
tblgen = { git = "https://github.com/FIM43-Redeye/tblgen-rs.git", branch = "feat/varbit-init", default-features = false, features = ["llvm21-0"] }
```

- [x] **Step 3: Remove `tblgen` from xdna-emu's build-deps**

Edit `Cargo.toml` (top-level)'s `[build-dependencies]`:

```toml
[build-dependencies]
serde_json = "1"
xdna-archspec = { path = "crates/xdna-archspec" }
# cc is kept for now; removed in Task 10.
cc = "1"
```

(The `tblgen` line is removed. `xdna-archspec` stays because xdna-emu's shrinking `build.rs` still calls it indirectly through the xrt-plugin install logic's environment setup, though not directly -- double-check post-Task 11.)

- [x] **Step 4: Add `mod build_helpers;` + `gen_tablegen` call to archspec/build.rs**

Copy the TableGen block from xdna-emu/build.rs (the section starting with `if aie2_td.exists()` around line 164 and the `build_helpers::extract::extract_all(...)` + `build_helpers::codegen::generate_tablegen_file(...)` calls). Paste into archspec/build.rs's `main()`. Also copy the `llvm_aie_path` resolution.

At the top of archspec/build.rs, add:

```rust
#[path = "build_helpers/mod.rs"]
mod build_helpers;
```

(Next to the other `#[path]` includes.)

- [x] **Step 5: Create `src/aie2/isa/mod.rs`**

```rust
//! AIE2 ISA decoder tables extracted from llvm-aie TableGen.

pub mod decoder_tables;
```

- [x] **Step 6: Create `src/aie2/isa/decoder_tables.rs`**

```rust
//! Complete instruction decoder tables extracted at build time from
//! llvm-aie's AIE2 TableGen records. Every decodable AIE2 instruction
//! is represented with its slot, format, encoding, and operand layout.

include!(concat!(env!("OUT_DIR"), "/gen_tablegen.rs"));
```

- [x] **Step 7: Declare the module in `aie2/mod.rs`**

```rust
pub mod isa;
```

- [x] **Step 8: Rewrite interpreter-decoder imports to point at the new path**

The interpreter currently imports decoder tables from `crate::tablegen` (xdna-emu's own module that includes `gen_tablegen.rs`). Find the consumer:

```bash
rg 'concat!\(env!\("OUT_DIR"\), "/gen_tablegen.rs"\)' src/
rg 'use crate::tablegen' src/
rg 'crate::tablegen::' src/
```

Whatever file currently does `include!(concat!(env!("OUT_DIR"), "/gen_tablegen.rs"))` switches to `pub use xdna_archspec::aie2::isa::decoder_tables::*;`. Interpreter imports of `crate::tablegen::*` rewrite similarly if they were pulling from the generated table.

Note: xdna-emu's `src/tablegen.rs` file may still exist and serve other purposes. Check whether it has hand-written content. If yes, keep the file but delete the `include!()` line.

- [x] **Step 9: Delete the TableGen block from xdna-emu/build.rs**

Delete the `extract_all` / `generate_tablegen_file` section, the `llvm_aie_path` resolution (if not still needed for the decoder-FFI compile in Task 10), and the `#[path = "build_helpers/mod.rs"] mod build_helpers;` line (xdna-emu no longer has a build_helpers dir).

- [x] **Step 10: Build + test + bridge smoke**

```bash
cargo build
cargo test --lib
cargo test -p xdna-archspec --lib
./scripts/emu-bridge-test.sh --no-hw -v add_one
```

Expected: baseline. `cargo build -p xdna-archspec` rebuilds the TableGen extraction from scratch on first run (LLVM-aware; takes ~30-60s).

- [x] **Step 11: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: move build_helpers + gen_tablegen into xdna-archspec

TableGen extractor (build_helpers/) and generated ISA decoder tables
(gen_tablegen.rs) now live under crates/xdna-archspec. Exposed as
xdna_archspec::aie2::isa::decoder_tables.

Cargo: tblgen build-dep moves from xdna-emu to xdna-archspec.

Interpreter decoder tables consumed through the new path; the Rust
decoder logic itself (bundle assembly, TRY_DECODE handling, semantic
operation mapping) stays in xdna-emu::interpreter::decode.

Generated using Claude Code.
EOF
)"
```

---

### Task 10: Move `decoder_ffi/` (aie2_decoder.cpp + LLVM link) into `xdna-archspec`

> **Deferred.** The decoder_ffi `extern "C"` block lives in
> `src/tablegen/decoder_ffi.rs` (1,185 lines), enmeshed with
> `interpreter::bundle::slot::Operand` via `MappedOperand` /
> `RegisterMap` / `classify_reg_name`. A clean move requires
> moving or abstracting those interpreter types, which belongs to
> Subsystem 6 (ISA Decode). Deferred there.

**Files:**
- Move: `decoder_ffi/` directory
- Modify: `crates/xdna-archspec/Cargo.toml` (add `cc` build-dep)
- Modify: `xdna-emu/Cargo.toml` (remove `cc` build-dep)
- Modify: `crates/xdna-archspec/build.rs`
- Create: `crates/xdna-archspec/src/aie2/decoder_ffi.rs`
- Modify: `crates/xdna-archspec/src/aie2/mod.rs`
- Modify: `build.rs` (xdna-emu)
- Modify: `src/interpreter/decode/` files that import the FFI

- [x] **Step 1: Move the directory**

```bash
git mv decoder_ffi crates/xdna-archspec/decoder_ffi
```

- [x] **Step 2: Add `cc` as a build-dep on `xdna-archspec`**

`crates/xdna-archspec/Cargo.toml`'s `[build-dependencies]`:

```toml
[build-dependencies]
serde_json = "1"
tblgen = { git = "https://github.com/FIM43-Redeye/tblgen-rs.git", branch = "feat/varbit-init", default-features = false, features = ["llvm21-0"] }
cc = "1"
```

- [x] **Step 3: Remove `cc` from xdna-emu's build-deps**

Top-level `Cargo.toml`'s `[build-dependencies]`:

```toml
[build-dependencies]
serde_json = "1"
xdna-archspec = { path = "crates/xdna-archspec" }
```

(If `xdna-archspec` build-dep is no longer used by anything in xdna-emu/build.rs post-Task 11, it gets removed in Task 11. For now it stays.)

- [x] **Step 4: Copy `compile_llvm_decoder_ffi` and its helpers into archspec/build.rs**

From xdna-emu/build.rs: locate `fn compile_llvm_decoder_ffi` and any helpers it calls (LLVM probe, library path resolution). Paste into archspec/build.rs. Update any `manifest_dir` references to `manifest_dir` (archspec's), and paths pointing to `decoder_ffi/...` still work because the directory moved alongside the script.

Call site in archspec/build.rs's `main()`:

```rust
compile_llvm_decoder_ffi(&manifest_dir, llvm_aie_path);
```

(The exact signature depends on the original; adjust to match.)

- [x] **Step 5: Create `src/aie2/decoder_ffi.rs`**

Copy any `extern "C"` declarations for `aie2_decoder.cpp` from xdna-emu's interpreter module. They are usually in `src/interpreter/decode/llvm_decoder.rs` or similar. Find:

```bash
rg 'extern "C"' src/interpreter/
```

Move the declarations into `crates/xdna-archspec/src/aie2/decoder_ffi.rs`. Example structure:

```rust
//! LLVM MCDisassembler FFI bindings for AIE2.
//!
//! Raw `extern "C"` declarations linking against the C++ wrapper in
//! decoder_ffi/aie2_decoder.cpp. Consumers in
//! xdna-emu::interpreter::decode build Rust-side semantic meaning on
//! top of these bindings.

use std::ffi::c_void;

extern "C" {
    // Example -- match the actual declarations:
    pub fn aie2_decoder_create() -> *mut c_void;
    pub fn aie2_decoder_destroy(ctx: *mut c_void);
    pub fn aie2_decode_instruction(
        ctx: *mut c_void,
        bytes: *const u8,
        len: usize,
        out_mnemonic: *mut u8,
        out_mnemonic_len: usize,
        // ... additional out-params ...
    ) -> i32;
}
```

(Copy the actual declarations from the source; this is illustrative.)

- [x] **Step 6: Declare the module in `aie2/mod.rs`**

```rust
pub mod decoder_ffi;
```

- [x] **Step 7: Rewrite interpreter callers to import from the new path**

In xdna-emu interpreter files (likely `src/interpreter/decode/llvm_decoder.rs` or equivalent), replace:

```rust
extern "C" {
    fn aie2_decoder_create() -> ...;
    // ...
}
```

with:

```rust
use xdna_archspec::aie2::decoder_ffi::{aie2_decoder_create, aie2_decode_instruction, /* ... */};
```

Delete the local `extern "C"` block.

- [x] **Step 8: Delete `compile_llvm_decoder_ffi` and its helpers from xdna-emu/build.rs**

Also delete the `llvm_aie_path` resolution if it's no longer used (xdna-emu/build.rs should have nothing referring to LLVM after this task).

- [x] **Step 9: Build + test + bridge smoke**

```bash
cargo build
cargo test --lib
cargo test -p xdna-archspec --lib
./scripts/emu-bridge-test.sh --no-hw -v add_one
```

Expected: baseline. This build is the first time the archspec crate performs the C++ link step; if it fails, re-check the paths in `compile_llvm_decoder_ffi` (they must resolve relative to archspec's `manifest_dir`, not xdna-emu's).

- [x] **Step 10: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: move decoder_ffi (aie2_decoder.cpp + LLVM link) into xdna-archspec

LLVM MCDisassembler FFI compilation and the raw extern "C" bindings
now live in xdna-archspec. Exposed as xdna_archspec::aie2::decoder_ffi
(extern "C" only). The Rust-side decoder logic that builds semantic
operations on top of these bindings stays in
xdna-emu::interpreter::decode.

Cargo: cc build-dep moves from xdna-emu to xdna-archspec.

Generated using Claude Code.
EOF
)"
```

---

### Task 11: Document hybrid build.rs state (reduced scope)

> **Reduced scope.** Tasks 9 and 10 were deferred to Subsystem 6 (see their
> deferral notes above). Their deferral blocks the original goal of reducing
> `xdna-emu/build.rs` to plugin-install only. Instead, Task 11 updates
> documentation to accurately reflect the hybrid state: what remains in
> `xdna-emu/build.rs`, why each piece is still there, and that Subsystem 6
> is the trigger for the remaining cleanup.
>
> The full build.rs shrinkage (Steps 1-2 of the original plan) will execute
> as the last step of Subsystem 6, once `extract_aiert` + `gen_aiert_*`,
> the TableGen block, and the FFI compile have all moved together with their
> coupled interpreter types.
>
> The verification steps below still apply -- with adjusted expected values
> (non-zero `crate::arch` consumers, non-empty `build.rs`, `build_helpers/`
> and `decoder_ffi/` still present at their original paths).

**Files modified in reduced scope:**
- `build.rs` (xdna-emu) -- header doc updated to describe hybrid state
- `xdna-emu/Cargo.toml` -- `[build-dependencies]` has per-line comments explaining why each dep is still needed
- `docs/superpowers/plans/2026-04-17-subsys1-regs-mem.md` -- Task 10 deferral note + this reduced-scope note
- `docs/arch/subsys1-audit.md` -- Task 10 Deferral + Task 11 Reduced Scope sections

- [x] **Step 1: Update `xdna-emu/build.rs` header doc**

Replace the existing header doc with text that names what still lives
in `xdna-emu/build.rs` and why each item awaits Subsystem 6.

- [x] **Step 2: Update `xdna-emu/Cargo.toml` build-deps**

Add per-line comments to each build-dep entry explaining why it is still
pulled in and which subsystem removes it.

- [x] **Step 3: Verify `mod arch` is the simplified forwarder**

```bash
rg -A 10 'pub mod arch' src/lib.rs
```

Expected: `pub use xdna_archspec::aie2::*;` + `pub mod subsystem { pub use xdna_archspec::aie2::subsystems::*; }`. If more complex, flag as a regression.

- [x] **Step 4: Count `crate::arch` consumers**

```bash
rg -l 'crate::arch\b' src/ examples/ tests/ xrt-plugin/ | wc -l
```

Expected: ~37 (pre-Task-4 baseline; cleanup is deferred alongside Tasks 9/10).

- [x] **Step 5: Verify `build_helpers/` and `decoder_ffi/` still present**

```bash
ls build_helpers decoder_ffi
```

Expected: both exist (deferred Tasks 9 and 10 did not move them).

- [x] **Step 6: Build + test + bridge smoke**

```bash
cargo build
TMPDIR=/tmp/claude-1000 cargo test --lib
cargo test -p xdna-archspec --lib
./scripts/emu-bridge-test.sh --no-hw -v add_one
```

Expected: baseline unchanged -- only doc comments and Cargo.toml comments modified.

- [x] **Step 7: Commit**

Commit with message describing: Tasks 9+10 deferral rationale, Task 11
reduced scope, what was updated (build.rs header, Cargo.toml comments,
plan and audit docs).

---

### Task 12: Part A verification gate

**Files:** none modified.

- [x] **Step 1: Full library tests**

```bash
cargo test --lib 2>&1 | tail -3
```

Expected: `2798 passed; 0 failed; 5 ignored` (baseline).

- [x] **Step 2: Archspec tests**

```bash
cargo test -p xdna-archspec --lib 2>&1 | tail -3
```

Expected: same as baseline (137 passing + 1 pre-existing failure).

- [x] **Step 3: Release build**

```bash
cargo build --release
```

Expected: clean. Release-only compile errors (if any) surface here.

- [x] **Step 4: Bridge smoke**

```bash
./scripts/emu-bridge-test.sh --no-hw -v add_one
```

Expected: Chess 10/10, Peano 9/9.

- [x] **Step 5: Full bridge HW run**

Long-running; ~15-30 minutes. Can be scheduled rather than run interactively.

```bash
./scripts/emu-bridge-test.sh 2>&1 | tee /tmp/claude-1000/subsys1-partA-bridge.log
```

Expected: no regressions vs. pre-Phase-1a baseline (Chess 64/64 PASS; Peano compiles with two pre-existing EMU timeouts + 1 XFAIL).

- [x] **Step 6: ISA test suite**

Long-running; ~5-10 minutes.

```bash
./scripts/isa-test.sh 2>&1 | tee /tmp/claude-1000/subsys1-partA-isa.log
```

Expected: no regressions vs. baseline pass count captured in Task 1.

- [x] **Step 7: Tag**

```bash
git tag phase1-subsys-regs-mem-partA
```

- [x] **Step 8: Append Part A completion log to the audit**

Edit `docs/arch/subsys1-audit.md`. Append:

```markdown
## Part A Completion

Landed <DATE>. Tag: `phase1-subsys-regs-mem-partA`.

### Commits
<paste `git log --oneline <prev-tag>..phase1-subsys-regs-mem-partA`>

### Verification
- `cargo test --lib`: <result>
- `cargo test -p xdna-archspec --lib`: <result>
- `cargo build --release`: <result>
- Bridge --no-hw smoke: <result>
- Full HW bridge run: <result>
- ISA test suite: <result>

### Surprises / deviations
<enumerate any, e.g., consumer files missed by the audit, import
resolution friction in #[path]-included modules>
```

Commit:

```bash
git add docs/arch/subsys1-audit.md
git commit -m "docs: subsys1 part A completion log

Generated using Claude Code."
```

---

## Part B -- Register & Memory Map Semantic Tightening

**Tag at end:** `phase1-subsys-regs-mem`

---

### Task 13: Create `xdna_archspec::aie2::memory_map`

**Files:**
- Create: `crates/xdna-archspec/src/aie2/memory_map.rs`
- Modify: `crates/xdna-archspec/src/aie2/mod.rs`

- [x] **Step 1: Create the module with the derived consts**

```rust
//! Derived memory-map constants for AIE2.
//!
//! These consts are computed from values in other `aie2` modules:
//! `compute::MEMORY_SIZE`, `compute::PROGRAM_MEM_HOST_OFFSET`,
//! `DATA_MEM_HOST_OFFSET`, `memtile::MEMORY_SIZE`, and `cardinal::EAST`.
//! They live here rather than in consumer crates so all AIE2 memory-map
//! derivations have a single home.

use super::{cardinal, compute, memtile, DATA_MEM_HOST_OFFSET};

/// AIE data memory base in the core's data address space.
///
/// This is the East cardinal direction (local memory for AIE2) base
/// address: `cardinal::EAST * MEMORY_SIZE = 7 * 0x10000 = 0x70000`.
///
/// ELF binaries place data at this address because it IS the hardware
/// address for the core's own data memory. The linker respects the
/// hardware memory map; this is NOT merely a linker convention.
///
/// Source: aie-rt `_XAie_GetTargetTileLoc()` --
/// `CardDir = Addr / DataMemSize`, where CardDir 7 = East = local tile
/// (for AIE2 with IsCheckerBoard=0).
pub const AIE_DATA_MEMORY_BASE: u32 =
    cardinal::EAST as u32 * compute::MEMORY_SIZE as u32;

/// Program memory base offset in host/CDO address space.
///
/// Source: AM025 CORE_MODULE_PROGRAM_MEMORY + aie-rt
/// XAIEMLGBL_CORE_MODULE_PROGRAM_MEMORY.
pub const PROGRAM_MEMORY_BASE: u32 = compute::PROGRAM_MEM_HOST_OFFSET;

/// Program memory end offset.
///
/// Note: only 16 KB is implemented (`compute::PROGRAM_MEMORY_SIZE`),
/// but the address window spans a full 64 KB region in the tile's host
/// address space.
pub const PROGRAM_MEMORY_END: u32 = PROGRAM_MEMORY_BASE + 0xFFFF;

/// Data memory base offset in host/CDO address space (always 0).
pub const DATA_MEMORY_BASE: u32 = DATA_MEM_HOST_OFFSET;

/// Data memory end offset for compute tile.
pub const COMPUTE_DATA_MEMORY_END: u32 = compute::MEMORY_SIZE as u32 - 1;

/// Data memory end offset for memory tile.
pub const MEM_TILE_DATA_MEMORY_END: u32 = memtile::MEMORY_SIZE as u32 - 1;
```

- [x] **Step 2: Declare the module in `aie2/mod.rs`**

```rust
pub mod memory_map;
```

- [x] **Step 3: Build and test the crate in isolation**

```bash
cargo build -p xdna-archspec
cargo test -p xdna-archspec --lib
```

Expected: clean, 138 passing + 1 pre-existing.

- [x] **Step 4: Commit**

```bash
git add crates/xdna-archspec/src/aie2/memory_map.rs crates/xdna-archspec/src/aie2/mod.rs
git commit -m "$(cat <<'EOF'
refactor: add xdna_archspec::aie2::memory_map with derived consts

New module aggregates memory-map constants derived from other aie2
submodules (AIE_DATA_MEMORY_BASE, PROGRAM_MEMORY_BASE, PROGRAM_MEMORY_END,
DATA_MEMORY_BASE, COMPUTE_DATA_MEMORY_END, MEM_TILE_DATA_MEMORY_END).

Previously defined in xdna-emu/src/device/registers_spec.rs. Task 14
migrates consumers; Task 15 dissolves registers_spec.rs.

Generated using Claude Code.
EOF
)"
```

---

### Task 14: Migrate consumers to `xdna_archspec::aie2::memory_map`

**Files:**
- Modify: every file in the Task 1 Step 7 audit list that imports from `device::registers_spec` and uses one of the derived consts.

- [x] **Step 1: Identify which `registers_spec` imports are for derived consts vs. register offsets**

```bash
rg -n 'AIE_DATA_MEMORY_BASE|PROGRAM_MEMORY_BASE|PROGRAM_MEMORY_END|DATA_MEMORY_BASE|COMPUTE_DATA_MEMORY_END|MEM_TILE_DATA_MEMORY_END' src/
```

Each match is a consumer that switches to `xdna_archspec::aie2::memory_map`.

- [x] **Step 2: Rewrite consumers**

For each file the grep found, edit the import. Patterns:

Before:
```rust
use crate::device::registers_spec::{AIE_DATA_MEMORY_BASE, PROGRAM_MEMORY_BASE};
```

After:
```rust
use xdna_archspec::aie2::memory_map::{AIE_DATA_MEMORY_BASE, PROGRAM_MEMORY_BASE};
```

Mixed imports (derived + register-offset submodules):

Before:
```rust
use crate::device::registers_spec::{DATA_MEMORY_BASE, memory_module};
```

After:
```rust
use xdna_archspec::aie2::memory_map::DATA_MEMORY_BASE;
use crate::device::registers_spec::memory_module;  // still needed for now
```

- [x] **Step 3: Build and test**

```bash
cargo build
cargo test --lib
cargo test -p xdna-archspec --lib
```

Expected: baseline.

- [x] **Step 4: Bridge smoke**

```bash
./scripts/emu-bridge-test.sh --no-hw -v add_one
```

- [x] **Step 5: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: migrate memory_map consumers to xdna_archspec::aie2

Every consumer of AIE_DATA_MEMORY_BASE, PROGRAM_MEMORY_BASE,
DATA_MEMORY_BASE, COMPUTE_DATA_MEMORY_END, MEM_TILE_DATA_MEMORY_END,
and PROGRAM_MEMORY_END now imports from xdna_archspec::aie2::memory_map.
registers_spec.rs's register-offset submodules (memory_module,
core_module, mem_tile_module) still forward from Task 6 and are
dissolved in Task 15.

Generated using Claude Code.
EOF
)"
```

---

### Task 15: Dissolve `src/device/registers_spec.rs`

**Files:**
- Delete: `src/device/registers_spec.rs`
- Create (if the audit recommends it): `src/device/bit_utils.rs`
- Modify: `src/device/mod.rs` (drop `registers_spec` decl; add `bit_utils` if created)
- Modify: consumers of `registers_spec::{memory_module, core_module, mem_tile_module}`

- [x] **Step 1: Check what's still in `registers_spec.rs`**

```bash
wc -l src/device/registers_spec.rs
rg '^pub|^fn|^use' src/device/registers_spec.rs
```

Expected content: `sign_extend_7bit` helper + its test + three `pub mod` shells that `pub use xdna_archspec::aie2::registers{, ::memory, ::mem_tile}::*`.

- [x] **Step 2: Find `sign_extend_7bit` call sites**

```bash
rg -n 'sign_extend_7bit' src/
```

Record file:line for each.

- [x] **Step 3: Move `sign_extend_7bit`**

If the audit (Task 1 Step 6) shows one or two call sites, inline it. Otherwise (3+ sites) move to a new `src/device/bit_utils.rs`:

```rust
//! Small bit-twiddling utilities used across the device module.

/// Sign-extend a 7-bit value to i8.
///
/// Used for lock acquire/release values which are 7-bit signed in the
/// BD register encoding.
#[inline]
pub const fn sign_extend_7bit(val: u32) -> i8 {
    if val & 0x40 != 0 {
        (val | 0x80) as u8 as i8
    } else {
        val as i8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_extend_7bit() {
        assert_eq!(sign_extend_7bit(0), 0);
        assert_eq!(sign_extend_7bit(1), 1);
        assert_eq!(sign_extend_7bit(63), 63);
        assert_eq!(sign_extend_7bit(0x7F), -1);
        assert_eq!(sign_extend_7bit(0x40), -64);
        assert_eq!(sign_extend_7bit(0x41), -63);
    }
}
```

Add to `src/device/mod.rs`:

```rust
pub mod bit_utils;
```

- [x] **Step 4: Rewrite `sign_extend_7bit` callers**

If moved: callers switch from `use crate::device::registers_spec::sign_extend_7bit;` to `use crate::device::bit_utils::sign_extend_7bit;`.

If inlined: delete the import and inline the two-line body at each call site.

- [x] **Step 5: Rewrite callers that use the `memory_module`/`core_module`/`mem_tile_module` shells**

```bash
rg -n 'registers_spec::memory_module|registers_spec::core_module|registers_spec::mem_tile_module' src/
```

For each, rewrite:

| Before | After |
|--------|-------|
| `use crate::device::registers_spec::memory_module::LOCK_REQUEST_BASE;` | `use xdna_archspec::aie2::registers::memory::LOCK_REQUEST_BASE;` |
| `use crate::device::registers_spec::core_module::CORE_PC;` | `use xdna_archspec::aie2::registers::CORE_PC;` |
| `use crate::device::registers_spec::mem_tile_module::LOCK_REQUEST_BASE;` | `use xdna_archspec::aie2::registers::mem_tile::LOCK_REQUEST_BASE;` |

- [x] **Step 6: Delete `src/device/registers_spec.rs`**

```bash
git rm src/device/registers_spec.rs
```

Edit `src/device/mod.rs`: delete the `pub mod registers_spec;` line.

- [x] **Step 7: Build and test**

```bash
cargo build
cargo test --lib
cargo test -p xdna-archspec --lib
```

Expected: baseline.

- [x] **Step 8: Bridge smoke**

```bash
./scripts/emu-bridge-test.sh --no-hw -v add_one
```

- [x] **Step 9: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: dissolve src/device/registers_spec.rs

Every remaining consumer of registers_spec migrated to
xdna_archspec::aie2::{registers, memory_map}. sign_extend_7bit <inlined
at its N call sites | moved to src/device/bit_utils.rs>.
registers_spec.rs deleted.

Generated using Claude Code.
EOF
)"
```

---

### Task 16: Write the registers & memory map design note

**Files:**
- Create: `docs/arch/registers-memory-map.md`

- [x] **Step 1: Write the design note**

Create `docs/arch/registers-memory-map.md`:

```markdown
# Registers & Memory Map -- Design Note

**Subsystem:** 1 (Phase 1b)
**Tag:** `phase1-subsys-regs-mem`
**Spec:** [2026-04-17-subsys1-regs-mem-design.md](../superpowers/specs/2026-04-17-subsys1-regs-mem-design.md)

This document is the mandatory per-seam design note required by the
parent device-family refactor. Subsystem 1 adds no trait seam, so this
note explains *why not* and sets the principle for subsequent
subsystems that face the same trait-vs-const decision.

## What lives where

| Data | Module | Source |
|------|--------|--------|
| Array topology (COLUMNS, ROWS, NUM_MEM_TILE_ROWS) | `xdna_archspec::aie2` | mlir-aie device model (tools/aie-device-models.json) |
| Tile address encoding (TILE_COL_SHIFT, TILE_ROW_SHIFT, TILE_OFFSET_MASK) | `xdna_archspec::aie2` | ArchModel array_topology |
| Row classification (SHIM_ROW, COMPUTE_ROW_START) | `xdna_archspec::aie2` | Derived from NUM_MEM_TILE_ROWS |
| Per-tile memory sizes (compute/memtile/shim MEMORY_SIZE) | `xdna_archspec::aie2::{compute,memtile,shim}` | mlir-aie device model |
| Physical banking (PHYSICAL_BANKS, PHYSICAL_BANK_SIZE, PHYSICAL_BANK_WIDTH_BITS) | `xdna_archspec::aie2::{compute,memtile}` | AM020 Ch 2 (hand-populated in model_builder) |
| Lock / BD / channel counts (NUM_LOCKS, NUM_BDS, NUM_DMA_CHANNELS) | `xdna_archspec::aie2::{compute,memtile,shim}` | ArchModel instance counts (cross-validated) |
| Cardinal directions (EAST, WEST, NORTH, SOUTH) | `xdna_archspec::aie2::cardinal` | aie-rt _XAie_GetTargetTileLoc |
| Derived memory-map base/end (AIE_DATA_MEMORY_BASE, etc.) | `xdna_archspec::aie2::memory_map` | Derived from cardinal + compute/memtile MEMORY_SIZE |
| Core register offsets (CORE_CONTROL, CORE_PC, etc.) | `xdna_archspec::aie2::registers` | AM025 JSON |
| Lock request bit layout (LOCK_REQUEST_BASE, _ID_SHIFT, _ID_MASK, _ACQ_REL_BIT, _VALUE_SHIFT, _VALUE_MASK) | `xdna_archspec::aie2::registers::{memory, mem_tile}` | AM025 JSON |
| Subsystem address ranges (per-tile-type OFFSET_START/END) | `xdna_archspec::aie2::subsystems` | ArchModel subsystems (cross-validated AM025 + aie-rt) |

All of the above are `pub const`. None of them are methods on
`ArchConfig`.

## The const-first principle

Build-time extraction bakes arch data into Rust as `pub const`. The
`ArchConfig` trait exists for top-level runtime polymorphism ("what
arch am I emulating?") -- not as the primary data interface.

Consequences:
- Code in const contexts (other `const` definitions, array sizes,
  pattern match constants) reaches for `xdna_archspec::aie2::X`.
- Code in runtime contexts that already holds an `Arc<dyn ArchConfig>`
  uses the existing trait methods where they exist
  (`data_memory_size`, `lock_count`, etc.). No new trait methods are
  added for data already reachable via a const.
- Hot paths avoid `dyn` dispatch by monomorphizing (generic `Arch`
  type parameter with associated consts is a future direction; see
  the parent refactor spec).

## What would AIE1 look like?

AIE1 (Versal AIE2 / xcvc1902) would have its own sibling module:
`xdna_archspec::aie1::{registers, memory_map, subsystems, ...}` -- same
module shapes, different values. Specifically:
- Register offsets differ (different AM025-equivalent JSON).
- Memory sizes differ (AIE1 compute has 32 KB data memory vs AIE2's 64 KB).
- Cardinal direction conventions differ (AIE1 is checkerboarded;
  AIE2's `IS_CHECKERBOARD=false` becomes `true`, and the East-is-local
  rule becomes row-dependent).
- Physical banking differs (AIE1 compute: 16 banks of 4 KB; AIE2: 8 banks of 8 KB).

Every one of these is a data change, not a behavior change. The same
consuming code (offset arithmetic, row checks, bank-conflict detection)
works for both arch families when given the correct const module.
Adding AIE1 is "populate the arch data in a sibling module and wire
it in at the top-level arch-selection point" -- not "implement a new
`RegisterMap` trait."

## Forward pointers

Other slices of `xdna_archspec::aie2::*` that **moved in Part A but
will tighten in their owning subsystem**:

- Stream switch (`stream_switch.rs`, top-level port arrays,
  `port_type`) -- Subsystem 5.
- Packet + control-packet format (`packet.rs`, `ctrl_packet.rs`) --
  Subsystem 5.
- Processor constants (`processor::*`) -- Subsystem 6.
- FoT mode values (`fot::*`) -- Subsystem 3.
- ISA decoder tables (`isa::decoder_tables.rs`) -- Subsystem 6.
- Timing constants (`timing::*`) -- interleave across Subsystems 3, 5, 7.
- Trace events (`trace_events.rs`) -- stays data; no subsystem owns it
  behaviorally.

Each of those subsystems will write its own design note at
`docs/arch/<subsystem>.md` when it runs.

## What about behavioral seams?

Subsystem 1 deliberately adds no trait. The trait-vs-const decision
gets re-asked in every subsequent subsystem:

- **Subsystem 3 (DMA).** BD parse/encode + channel stepping likely
  *do* differ between arch families. Expected seam: `DmaModel`.
- **Subsystem 4 (Locks).** Probably thin; may or may not justify
  `LockModel`.
- **Subsystem 5 (Stream Switch).** Topology is data (already in
  archspec). Routing legality rules likely warrant
  `StreamSwitchModel`.
- **Subsystem 6 (ISA Decode).** AIE1 (3-slot VLIW) vs AIE2 (6-slot)
  is the biggest arch cliff. Expected seam: `IsaDecoder`.
- **Subsystem 7 (ISA Execute).** Semantic ops, intrinsic handlers.
  Expected seam: `IsaExecutor`.
- **Subsystem 8 (Parser).** Container-format variance. Expected
  seam: `BinaryLoader`.

The principle is the same every time: if the per-arch difference is
shape, lift it behind a trait. If it is only values, keep it in a
const module.
```

- [x] **Step 2: Commit**

```bash
git add docs/arch/registers-memory-map.md
git commit -m "$(cat <<'EOF'
docs: registers & memory map design note

Mandatory per-seam design note for Subsystem 1. Captures what lives
in xdna_archspec::aie2::{registers, memory_map, subsystems}, states
the const-first principle, answers "what would AIE1 look like?" (data
change, not behavior change -> no trait), and forward-points to which
slices tighten in which subsystem.

Generated using Claude Code.
EOF
)"
```

---

### Task 17: Part B verification gate + tag

**Files:** none modified.

- [x] **Step 1: Full library tests**

```bash
cargo test --lib 2>&1 | tail -3
```

Expected: baseline.

- [x] **Step 2: Archspec tests**

```bash
cargo test -p xdna-archspec --lib 2>&1 | tail -3
```

Expected: baseline.

- [x] **Step 3: Release build**

```bash
cargo build --release
```

Expected: clean.

- [x] **Step 4: Bridge smoke**

```bash
./scripts/emu-bridge-test.sh --no-hw -v add_one
```

Expected: Chess 10/10, Peano 9/9.

- [x] **Step 5: Full bridge HW run**

```bash
./scripts/emu-bridge-test.sh 2>&1 | tee /tmp/claude-1000/subsys1-partB-bridge.log
```

Expected: no regressions vs. pre-Phase-1a baseline.

- [x] **Step 6: ISA test suite**

```bash
./scripts/isa-test.sh 2>&1 | tee /tmp/claude-1000/subsys1-partB-isa.log
```

Expected: no regressions.

- [x] **Step 7: Verify success criteria from the spec**

Each success criterion from the spec `## Success criteria` section:

```bash
# Zero crate::arch references in live source
rg 'crate::arch\b' src/ examples/ tests/ xrt-plugin/ 2>&1 | grep -v '^\.git' || echo "OK: no crate::arch refs"

# xdna-emu/build.rs is plugin-install only
wc -l build.rs     # expected: ~60-80 lines
rg '^fn gen_|^fn extract_aiert|^fn compile_llvm_decoder_ffi' build.rs   # expected: no matches

# build_helpers/ and decoder_ffi/ no longer exist in xdna-emu
ls build_helpers decoder_ffi 2>&1 | grep -q 'No such file' && echo OK

# memory_map owns the derived consts
rg 'AIE_DATA_MEMORY_BASE|PROGRAM_MEMORY_BASE|DATA_MEMORY_BASE|COMPUTE_DATA_MEMORY_END|MEM_TILE_DATA_MEMORY_END' crates/xdna-archspec/src/aie2/memory_map.rs

# Design note exists
test -f docs/arch/registers-memory-map.md && echo OK

# ArchConfig trait surface unchanged
rg '^\s*fn ' crates/xdna-archspec/src/runtime.rs | grep -E 'fn (columns|rows|tile_kind|is_valid_tile|is_shim_tile|is_mem_tile|is_compute_tile|data_memory_size|program_memory_size|lock_count|max_lock_value|dma_s2mm_channels|dma_mm2s_channels|dma_total_channels|dma_bd_count|name)' | wc -l
# Expected: 15 (same method count as pre-Phase-1a).
```

Append a "Success Criteria" section with the above output to `docs/arch/subsys1-audit.md`.

- [x] **Step 8: Tag**

```bash
git tag phase1-subsys-regs-mem
```

- [x] **Step 9: Append Part B completion log to the audit**

Edit `docs/arch/subsys1-audit.md`. Append:

```markdown
## Part B Completion

Landed <DATE>. Tag: `phase1-subsys-regs-mem`.

### Commits
<paste `git log --oneline phase1-subsys-regs-mem-partA..phase1-subsys-regs-mem`>

### Verification
- `cargo test --lib`: <result>
- `cargo test -p xdna-archspec --lib`: <result>
- `cargo build --release`: <result>
- Bridge --no-hw smoke: <result>
- Full HW bridge run: <result>
- ISA test suite: <result>
- Success criteria: <pasted output>

### Surprises / deviations
<enumerate any>

### Follow-ups flagged for later subsystems
<anything to carry forward, e.g., "timing constants duplicate in
stream_switch and dma modules -- consolidate when Subsys 3 runs">
```

Commit:

```bash
git add docs/arch/subsys1-audit.md
git commit -m "docs: subsys1 part B completion log

Generated using Claude Code."
```

- [x] **Step 10: Update `NEXT-STEPS.md`**

Edit `NEXT-STEPS.md` to reflect Subsystem 1 completion and point to Subsystem 2 (Tile Topology) as the next pickup. Record:
- Tag shipped: `phase1-subsys-regs-mem`.
- Follow-ups Subsys 2 should consume (from Part B completion log).
- New baselines where different.

Commit:

```bash
git add NEXT-STEPS.md
git commit -m "docs: subsys1 completion; point NEXT-STEPS at subsys2

Generated using Claude Code."
```

---

## Self-Review Notes

**Spec coverage:**
- Spec Part A Task 1 (Audit) -> Plan Task 1.
- Spec Part A Task 2 (Scaffold archspec build.rs) -> Plan Tasks 2 (factor model_builder) + 3 (scaffold).
- Spec Part A Task 3 (gen_arch) -> Plan Task 4.
- Spec Part A Task 4 (gen_subsystems) -> Plan Task 5.
- Spec Part A Task 5 (gen_core_module + locks) -> Plan Task 6.
- Spec Part A Task 6 (gen_stream_ports + ranges) -> Plan Task 7.
- Spec Part A Task 7 (gen_trace_events) -> Plan Task 8.
- Spec Part A Task 8 (build_helpers + gen_tablegen) -> Plan Task 9.
- Spec Part A Task 9 (decoder_ffi) -> Plan Task 10.
- Spec Part A Task 10 (final cleanup) -> Plan Task 11.
- Spec Part A Task 11 (verification gate + tag) -> Plan Task 12.
- Spec Part B Task 1 (derived consts into aie2::memory_map) -> Plan Task 13.
- Spec Part B Task 2 (migrate consumers) -> Plan Task 14.
- Spec Part B Task 3 (dissolve registers_spec.rs) -> Plan Task 15.
- Spec Part B Task 4 (design note) -> Plan Task 16.
- Spec Part B Task 5 (verification gate + tag) -> Plan Task 17.

All spec tasks covered. One addition not explicitly in the spec: Plan Task 2 (factor `model_builder` out of `lib.rs`). Spec Part A Task 2 mentioned "scaffold xdna-archspec/build.rs"; Plan Task 2 is a prerequisite that lets Task 3 compile. Justified: without this split, `#[path]`-including `lib.rs` into build.rs pulls in `runtime.rs` and its `Arc`/`LazyLock` deps, breaking the self-contained build-script pattern.

**Placeholder scan:**
- Task 10 Step 5 says `// Example -- match the actual declarations:` in a code block. This is intentional -- the actual `extern "C"` declarations depend on what's currently in xdna-emu's decoder wrapper, which I haven't read yet; the task says "Copy the actual declarations from the source; this is illustrative." That's explicit enough.
- Task 15 Step 3 has an either/or: "If the audit shows one or two call sites, inline it. Otherwise (3+ sites) move to a new `src/device/bit_utils.rs`". Both branches are concrete. OK.
- No TBDs, TODOs, or "fill in later."

**Type consistency:**
- `xdna_archspec::aie2::*` used consistently throughout.
- `ModelConfig`, `ArchConfig`, `ArchModel` match their crate origins.
- `xdna_archspec::aie2::memory_map`, `::registers`, `::subsystems`, `::stream_switch`, `::trace_events`, `::isa::decoder_tables`, `::decoder_ffi`, `::port_type` all defined in Plan Tasks 4-10 and referenced consistently in Tasks 13-15.

**Cadence of verification:**
- Every relocation task (4-11, 13-15) ends with `cargo build + cargo test --lib + cargo test -p xdna-archspec --lib + bridge --no-hw smoke`.
- Part A gate (Task 12) adds release build + full HW bridge + ISA tests.
- Part B gate (Task 17) adds the same + success-criteria sweep.

---

## Execution Hand-Off

Ready for subagent-driven-development or inline executing-plans execution.
