# Subsystem 1 -- Registers & Memory Map: Design

**Date**: 2026-04-17
**Status**: Approved, pending implementation plan
**Parent refactor**: [Device-Family Refactor](2026-04-16-device-family-refactor-design.md)
**Subsystem tag at end**: `phase1-subsys-regs-mem`

## Problem

After Phase 1a consolidated three parallel arch abstractions into the
`xdna-archspec` crate, a fourth parallel abstraction remains: the
build.rs-generated `crate::arch::*` module tree in the main `xdna-emu` crate.
NEXT-STEPS.md anticipated that Phase 1a had already confined this module to
`src/device/port_layout.rs`. An audit for Subsystem 1 showed otherwise: 37
source files under `src/` consume `crate::arch::*`, across at least 14
distinct categories (register offsets, memory sizes, tile address encoding,
row classification, cardinal directions, subsystem ranges, banking, stream
switch port arrays, stream packet layouts, control packet layouts,
ISA/processor constants, timing constants, FoT values, trace events).

The module is NOT a duplicate or parallel abstraction in the sense Phase 1a
addressed. It is a build-time projection of the same `ArchModel` data that
`xdna-archspec` owns, emitted as `pub const` values so consumers can use
them in `const` contexts where a trait method does not fit. The problem is
that generation lives in `xdna-emu/build.rs` (not in the archspec crate)
and that the module is AIE2-only (build.rs hardcodes `"npu1"`). Adding AIE2P
today means growing `xdna-emu/build.rs` rather than touching the crate that
exists for arch data -- the opposite of the device-family refactor's goal.

A secondary issue: several xdna-emu files derive their own `pub const`
values from `crate::arch::*` (for example, `src/device/registers_spec.rs`
computes `AIE_DATA_MEMORY_BASE`, `PROGRAM_MEMORY_BASE`, `DATA_MEMORY_BASE`,
and memory-range ends). These derived constants are effectively
arch-specific data living outside the arch crate. The same pattern appears
wherever a module builds its own `const` cache on top of `crate::arch::*`.

## Goals

1. **`xdna-archspec` becomes the single home for AIE2 arch data and its
   build-time extractors.** Every `pub const` that reflects AM025 registers,
   mlir-aie device-model data, aie-rt source values, TableGen extraction
   output, or arch-derived constants lives in the crate. The crate also
   owns the build.rs that produces them, the TableGen extractor
   (`build_helpers/`), the LLVM MCDisassembler FFI glue
   (`decoder_ffi/aie2_decoder.cpp` + the C++ build), and the event-ID
   bridge output.
2. **Per-arch module naming established now.** All AIE2 data lives under
   `xdna_archspec::aie2::*`. There is no stub `aie2p/` -- YAGNI -- but the
   namespace shape makes AIE2P a mechanical add.
3. **Build-time first, runtime-dispatch sparingly.** Values stay `pub
   const` whenever the current code reaches them that way. The
   `ArchConfig` trait is not grown to mirror every const. `ArchConfig` is
   the top-level "what arch am I emulating" boundary, not the primary data
   surface.
4. **Registers & memory map specifically are tightened.** Derived
   arch-specific consts in `src/device/registers_spec.rs` move into
   `xdna_archspec::aie2::memory_map`; the Subsystem 1 slice of
   `xdna_archspec::aie2::*` (registers, memory_map, subsystems) is
   organised as its permanent home. Other slices (stream switch, packet,
   control packet, processor, timing, FoT, trace events, ISA decoder
   tables) are **relocated** into `xdna_archspec::aie2::*` but their
   internal shape is not re-examined -- that is for Subsystems 3-6 to
   handle when they claim those slices.
5. **Preserve correctness.** `cargo test --lib` stays green every commit.
   Bridge tests pass on AIE2 at the end of Part A and Part B.

## Non-goals

- **No AIE2P data.** A future session adds `xdna_archspec::aie2p::*`; no
  empty scaffolding in this subsystem.
- **No new ArchConfig methods for anything already reachable as `const`.**
  No `tile_address_encoding()`, no `subsystem_range()`, no `cardinal_offset()`,
  no banking accessors. The design note will explain why.
- **No semantic tightening outside the register/memory-map slice.** Stream
  switch port data, ISA decoder tables, etc. move across in Part A with
  their current layout; renaming/splitting their internal modules is their
  own subsystem's job.
- **No new trait seam.** Register offsets and memory-map values are data,
  not behavior; there is no `RegisterMap` or `MemoryMap` trait. The "what
  would AIE1 look like?" answer is "the same shape with different
  numbers," so a const submodule is the right tool, not a trait.
- **No master merges.** All work is on `dev` with a single
  `phase1-subsys-regs-mem` tag at the end.

## Design

### Crate boundary

**`xdna-archspec` owns:** all arch data (registers, memory map, tile
topology, stream switch tables, packet layouts, ISA encodings, timing
constants, FoT values, event IDs), build-time extractors (AM025,
device-model, aie-rt, TableGen, mlir-aie-bridge.py), and LLVM MCDisassembler
FFI bindings (raw `extern "C"` only).

**`xdna-emu` owns:** all behavior -- instruction decode/execute, DMA
stepping, locks, stream switch routing, parsers, GUI. It consumes
`xdna-archspec` constants and calls the FFI bindings for disassembly. After
Subsystem 1, `xdna-emu/build.rs` is reduced to the XRT plugin install logic
only. The Rust-side decoder (bundle assembly, TRY_DECODE handling, semantic
operation mapping) remains in `xdna-emu::interpreter::decode`; only the
FFI declarations and the C++ link step move.

### Target directory layout

```
crates/xdna-archspec/
├── build.rs                    # assembles ArchModel + drives codegen + compiles C++ FFI
├── build_helpers/              # moved from xdna-emu (TableGen extractor)
├── decoder_ffi/                # moved from xdna-emu (aie2_decoder.cpp + headers)
└── src/
    ├── lib.rs                  # namespace + re-exports
    ├── types.rs                # (existing)
    ├── device_model.rs         # (existing)
    ├── regdb.rs                # (existing)
    ├── regdb_extractor.rs      # (existing)
    ├── tablegen.rs             # (existing; content is AIE2-only today -- Subsys 6 may split)
    ├── runtime.rs              # (existing) ArchConfig + ModelConfig
    └── aie2/
        ├── mod.rs              # flat top-level consts (COLUMNS, ROWS, SHIM_ROW,
        │                       #   NUM_MEM_TILE_ROWS, MAX_LOCK_VALUE,
        │                       #   TILE_{COL,ROW}_SHIFT, TILE_{OFFSET,PTR}_MASK,
        │                       #   DATA_MEM_HOST_OFFSET) via include!() from
        │                       #   gen_arch.rs, plus `pub mod` declarations and
        │                       #   any re-exports for back-compat
        ├── registers.rs        # core/memory/memtile module offsets (AM025)
        ├── memory_map.rs       # sizes, host offsets, cardinal, banking, derived consts
        ├── subsystems.rs       # per-tile-type subsystem address ranges
        ├── stream_switch.rs    # port arrays, ranges, bit layout (relocated; tightened in Subsys 5)
        ├── packet.rs           # stream packet format (relocated; tightened in Subsys 5)
        ├── ctrl_packet.rs      # control packet format (relocated; tightened in Subsys 5 or 8)
        ├── processor.rs        # slot widths, vector width (relocated; tightened in Subsys 6)
        ├── timing.rs           # latencies (relocated)
        ├── fot.rs              # FoT mode values (relocated; tightened in Subsys 3)
        ├── trace_events.rs     # event IDs from mlir-aie-bridge.py
        ├── isa/
        │   └── decoder_tables.rs   # TableGen decoder tables (relocated; tightened in Subsys 6)
        └── decoder_ffi.rs      # raw extern "C" bindings to aie2_decoder.cpp
```

`xdna-emu`'s `src/lib.rs` drops the `mod arch { include!(...) }` block.
Every `use crate::arch::*` becomes `use xdna_archspec::aie2::*` (or a
submodule path). `src/device/registers_spec.rs` either dissolves into the
relevant `xdna_archspec::aie2::*` modules or slims to the `sign_extend_7bit`
helper and its test.

### Consumer import rewrites (sample)

| Before | After |
|--------|-------|
| `crate::arch::compute::MEMORY_SIZE` | `xdna_archspec::aie2::compute::MEMORY_SIZE` |
| `crate::arch::memtile::NUM_DMA_CHANNELS` | `xdna_archspec::aie2::memtile::NUM_DMA_CHANNELS` |
| `crate::arch::SHIM_ROW` | `xdna_archspec::aie2::SHIM_ROW` |
| `crate::arch::{TILE_COL_SHIFT, TILE_ROW_SHIFT, TILE_OFFSET_MASK}` | `xdna_archspec::aie2::{TILE_COL_SHIFT, TILE_ROW_SHIFT, TILE_OFFSET_MASK}` |
| `crate::arch::cardinal::EAST` | `xdna_archspec::aie2::cardinal::EAST` |
| `crate::arch::subsystem::SHIM_LOCK_RANGE` | `xdna_archspec::aie2::subsystems::SHIM_LOCK_RANGE` |
| `crate::arch::stream_switch::compute::NORTH_MASTER_START` | `xdna_archspec::aie2::stream_switch::compute::NORTH_MASTER_START` |
| `crate::arch::timing::ROUTE_PER_HOP` | `xdna_archspec::aie2::timing::ROUTE_PER_HOP` |
| `crate::arch::processor::LDA_WIDTH` | `xdna_archspec::aie2::processor::LDA_WIDTH` |
| `crate::arch::ctrl_packet::OP_WRITE` | `xdna_archspec::aie2::ctrl_packet::OP_WRITE` |
| `crate::arch::packet::STREAM_ID_MASK` | `xdna_archspec::aie2::packet::STREAM_ID_MASK` |
| `crate::device::registers_spec::memory_module::...` | `xdna_archspec::aie2::registers::memory_module::...` |

`aie2/mod.rs` uses `pub use` to keep flat paths (`xdna_archspec::aie2::COLUMNS`,
`xdna_archspec::aie2::MAX_LOCK_VALUE`, etc.) stable even when an internal
module is later reshaped.

### Part A -- Infrastructure relocation

Part A is a pure move: generation site changes, call sites change, the
values returned are bit-identical. Any test failure in Part A is a missed
import or a path mismatch, never a semantic regression.

1. **Audit.** Enumerate every `include!(concat!(env!("OUT_DIR"), ...))`,
   every `crate::arch::*` reference in `src/`, every call site of
   `build_helpers::extract::*`, and the surface of the LLVM FFI (`extern
   "C"` declarations and link flags). Record in
   `docs/arch/subsys1-audit.md`. Commit the audit.
2. **Scaffold `xdna-archspec/build.rs`.** Empty body; sets up
   `cargo:rerun-if-changed` tracking for the AM025 JSON, the device-model
   JSON, `build.rs` itself, and `build_helpers/` (once moved). Commit.
3. **Move `gen_arch.rs` generation.** Generator function moves into
   `xdna-archspec/build.rs`. Output lands in `$OUT_DIR/gen_arch.rs`
   (archspec crate's OUT_DIR). Add `src/aie2/mod.rs` with `pub mod registers;`,
   `pub mod memory_map;`, etc. (placeholders filled in later steps); do the
   `include!()` in the right sub-module. Consumers rewrite
   `crate::arch::X` -> `xdna_archspec::aie2::X`. Main-crate's generation
   of `gen_arch.rs` is deleted in the same commit.
4. **Move `gen_subsystems.rs` generation.** As above, into
   `xdna_archspec::aie2::subsystems`. Consumers rewrite.
5. **Move `gen_core_module.rs`, `gen_memory_lock.rs`, `gen_memtile_lock.rs`
   generation.** Into `xdna_archspec::aie2::registers`. The `include!()`
   sites in `src/device/registers_spec.rs` rewrite to re-exports from the
   archspec crate.
6. **Move `gen_stream_ports.rs` and `gen_stream_ranges.rs` generation.**
   Into `xdna_archspec::aie2::stream_switch`. Port-layout consumers in
   `src/device/port_layout.rs` and `src/device/stream_switch/mod.rs`
   rewrite. This does not preempt Subsystem 5 -- only the location moves.
7. **Move `gen_trace_events.rs` generation.** Into
   `xdna_archspec::aie2::trace_events`. Trace-event consumers rewrite.
   The `mlir-aie-bridge.py` invocation moves into archspec's build.rs.
8. **Move `build_helpers/` (TableGen extractor).** Directory moves under
   `crates/xdna-archspec/build_helpers/`. `gen_tablegen.rs` generation
   moves to archspec's build.rs. Interpreter decoder rewrites to
   `xdna_archspec::aie2::isa::decoder_tables::*`.
9. **Move `decoder_ffi/aie2_decoder.cpp` and the C++ build logic.**
   Directory moves under `crates/xdna-archspec/decoder_ffi/`.
   `compile_llvm_decoder_ffi()` moves to archspec's build.rs. The Rust
   `extern "C"` declarations move to `xdna_archspec::aie2::decoder_ffi`.
   `xdna-emu::interpreter::decode` imports the bindings from the crate;
   the Rust-side decoder logic stays put.
10. **Final cleanup.** Delete `xdna-emu/src/lib.rs`'s `mod arch { ... }`
    block. Grep for `crate::arch` across `xdna-emu/src/`, `examples/`,
    `tests/`, and `xrt-plugin/`; confirm zero hits in live source (doc
    comments that merely *mention* `crate::arch` in a historical-narrative
    sense do not count -- but any `use`, qualified path, or active
    reference must be gone). Delete `xdna-emu/build_helpers/` and
    `xdna-emu/decoder_ffi/` directory stubs if any remain.
    `xdna-emu/build.rs` is now XRT-plugin-install only.
11. **Verification gate.** `cargo test --lib`; `cargo test -p
    xdna-archspec --lib`; `cargo build --release`;
    `./scripts/emu-bridge-test.sh --no-hw -v add_one`; full bridge HW run;
    ISA test suite (`./scripts/isa-test.sh`). Full bridge and ISA runs
    can be scheduled by the user rather than blocking the session. No
    regressions vs. pre-Phase-1a baseline.

Tag the end of Part A with `phase1-subsys-regs-mem-partA` for bisect
navigation. Do not tag `phase1-subsys-regs-mem` yet -- that is Part B's
terminal state.

### Part B -- Semantic tightening for registers & memory map

Once Part A is green, tighten the register/memory-map slice specifically.
Every other slice already moved in Part A stays unchanged until its own
subsystem takes it.

1. **Move derived consts from `registers_spec.rs` into
   `xdna_archspec::aie2::memory_map`.** Specifically `AIE_DATA_MEMORY_BASE`,
   `PROGRAM_MEMORY_BASE`, `PROGRAM_MEMORY_END`, `DATA_MEMORY_BASE`,
   `COMPUTE_DATA_MEMORY_END`, `MEM_TILE_DATA_MEMORY_END`. These are
   already defined in terms of values from the codegen; they belong next
   to those values, not in the consumer crate. Document each with its
   AM025 / aie-rt derivation.
2. **Migrate `registers_spec.rs` consumers to the archspec paths.** Every
   module that currently does `use crate::device::registers_spec::...`
   (and gets either a raw AM025 offset via the `include!()` or a derived
   const) switches to `use xdna_archspec::aie2::{registers, memory_map}::...`.
3. **Slim or dissolve `registers_spec.rs`.** Remaining content after
   step 2 is the `sign_extend_7bit` helper (plus its test). It is a
   pure-function utility and does not belong in a register-address file.
   Options: move to `xdna-emu::device::bit_utils` (or similar), or
   inline at its call sites. The audit (Part A step 1) enumerates those
   call sites; pick whichever is cleaner and delete the file.
4. **Write `docs/arch/registers-memory-map.md`.** The required per-seam
   design note, even though this subsystem adds no trait. Content:
    - What data lives in `xdna_archspec::aie2::{registers, memory_map,
      subsystems}` and where its authoritative source is (AM025 JSON,
      ArchModel, aie-rt).
    - The const-first principle and why no `ArchConfig` method additions
      happen here.
    - The "what would AIE1 look like?" answer: a sibling
      `xdna_archspec::aie1::{registers, memory_map, subsystems}` with
      the same shapes and different values. Register offsets differ;
      memory sizes differ (AIE1 compute has 32 KB data mem vs AIE2's 64
      KB); cardinal direction conventions may differ (AIE1 is
      checkerboarded; AIE2 is not). All of these are data changes, not
      behavior changes. The decision: no `RegisterMap` / `MemoryMap`
      trait. Multi-arch selection will be via per-arch const modules
      already established.
    - Forward-pointers: which other slices of `xdna_archspec::aie2::*`
      are expected to tighten in Subsystems 3, 5, 6, and which probably
      stay as pure data.
5. **Verification gate.** Same commands as Part A's gate. Tag
   `phase1-subsys-regs-mem`.

### What we are deliberately leaving

- **`ArchConfig` surface unchanged.** No new methods. Callers who already
  use `arch.data_memory_size(TileKind::Compute)` keep doing so; callers
  who already use `crate::arch::compute::MEMORY_SIZE` switch to
  `xdna_archspec::aie2::compute::MEMORY_SIZE` with no other change.
- **`crate::arch::*` const paths stay `pub const`.** We are not forcing
  them through runtime ArchConfig methods.
- **Stream switch, packet, control packet, processor, timing, FoT, ISA
  decoder tables.** These relocate in Part A but their internal module
  layout and naming are the problem of Subsystems 3 / 5 / 6 when those
  subsystems take them on.
- **Pre-existing deep renames** (`TileType::MemTile` -> `TileKind::Mem`,
  Shim / ShimNoc merge) remain flagged for Subsystem 2.

### Trait-seam design note (inline, expanded in docs/arch/)

Registers and memory map are **data**, not behavior. Two independent
architectures (AIE1, AIE2, AIE2P, hypothetically more) differ in their
register offsets and memory sizes, but every architecture has the *same
shape* of question: "given tile X of this kind, what is the subsystem Y
address range?" That question is answered with a table, and the table
differs by arch -- not the logic that consults it.

The spec's "what would AIE1 look like in ~100 words" test applied to a
putative `RegisterMap` trait yields "basically identical to AIE2," which
per the same spec means the trait is over-engineered. The correct tool is
a per-arch const module: `xdna_archspec::aie1::registers`,
`xdna_archspec::aie2::registers`, and so on. Code that needs to pick the
right table at compile time imports from the arch it targets; code that
needs runtime polymorphism goes through the existing `ArchConfig` trait
at the top-level "what am I emulating" boundary and caches or otherwise
avoids hot-path dispatch.

This subsystem therefore adds no trait seam. Subsystems 3 (DMA), 5
(Stream Switch), 6 (ISA Decode), and 7 (ISA Execute) will each face the
trait-vs-const decision again for their slice. The principle is the
same: ask whether the difference is shape or only values.

### Testing strategy

**Global invariants (every task, every commit):**

- `cargo test --lib` green (inherited from the parent refactor).
- `cargo test -p xdna-archspec --lib` green. New tests accompany each
  module moved into `xdna_archspec::aie2::*` confirming expected AIE2
  values (smoke-test against regression).
- Bridge test `--no-hw` smoke
  (`./scripts/emu-bridge-test.sh --no-hw -v add_one`) green. This is
  fast (~30 s) and catches most runtime regressions introduced by
  import rewrites or codegen-path changes.

**Gate checks (end of Part A, end of Part B, before tagging):**

- `cargo build --release` green (release-only compile and link errors
  surface here; matters especially when the C++ FFI compile moves).
- **Full HW bridge run** (`./scripts/emu-bridge-test.sh`) green with no
  regressions vs. the pre-refactor baseline. Dual-compiler: Chess is
  ground truth, Peano informational.
- **ISA test suite** (`./scripts/isa-test.sh`) green with no regressions
  vs. the pre-refactor baseline. ~5-10 minutes; catches anything the
  bridge smoke missed by exercising every ISA category against real
  hardware.

The baseline to match coming into the subsystem is:
- `cargo test --lib`: 2798 passed, 0 failed, 5 ignored
- `cargo test -p xdna-archspec --lib`: 138 passed, 1 pre-existing
  unrelated failure (`test_full_parse_all_devices`)
- Bridge `--no-hw -v add_one`: Chess 10/10, Peano 9/9
- Full bridge HW run: pre-Phase-1a result minus the two pre-existing
  Peano-EMU timeouts (`dma_task_large_linear`,
  `objectfifo_repeat/init_values_repeat`) and 1 XFAIL
- ISA test suite: capture current pass count at start of subsystem;
  any new failure is a regression

## Deliverables

1. **Subsys-1 audit** in `docs/arch/subsys1-audit.md`: enumerates every
   `crate::arch::*` consumer, every codegen-include site, every
   build_helpers / decoder_ffi reference. Guides Part A's file-by-file
   steps.
2. **Part A commits** (relocation): one commit per codegen file
   (`gen_arch`, `gen_subsystems`, `gen_core_module` + lock request
   generators, `gen_stream_ports` + `gen_stream_ranges`, `gen_trace_events`,
   `gen_tablegen` + build_helpers move, decoder_ffi move), plus the
   final cleanup commit. Tag `phase1-subsys-regs-mem-partA` at end.
3. **Part B commits** (semantic tightening for registers & memory map):
   derived-consts move, `registers_spec.rs` slim/dissolve, design note
   at `docs/arch/registers-memory-map.md`. Tag `phase1-subsys-regs-mem`
   at end.

## Risks and mitigations

- **Risk:** an import-rewrite sweep misses a file, revealed only at Part
  A's final gate.
  **Mitigation:** step 10 (final cleanup) is a single `rg crate::arch`
  sweep across `src/`; anything matched is a rewrite bug to fix before
  the gate.
- **Risk:** archspec crate's build.rs needs the device-model JSON at a
  path that worked for `xdna-emu/build.rs`. Relative paths break.
  **Mitigation:** Phase 1a's ARCHSPEC_MODELS cache already resolves the
  JSON via `CARGO_MANIFEST_DIR`; archspec's build.rs does the same.
  Both `MLIR_AIE_PATH` and `LLVM_AIE_PATH` override env vars continue
  to work.
- **Risk:** moving the C++ FFI compile changes link order and introduces
  a latent link failure in release mode.
  **Mitigation:** Part A's verification gate includes `cargo build
  --release` specifically because release-only compile and link errors
  can hide in a debug-only flow. The HW bridge run is run against both
  debug and release shared libraries.
- **Risk:** `trace_events.rs` depends on a Python bridge script that's
  awkward to invoke from the archspec crate.
  **Mitigation:** the invocation uses an absolute path already; moving
  the caller from `xdna-emu/build.rs` to `xdna-archspec/build.rs` only
  changes the cwd. Symlink or `CARGO_MANIFEST_DIR`-anchored path is a
  one-line change.
- **Risk:** hidden downstream consumer of `crate::arch::*` outside `src/`
  (examples, integration tests, the XRT plugin shim).
  **Mitigation:** the audit step sweeps `examples/`, `tests/`, and
  `xrt-plugin/` for `crate::arch::*` references as well, not just
  `src/`.

## Success criteria

- Zero references to `crate::arch` anywhere under `xdna-emu/src/`,
  `examples/`, `tests/`, or `xrt-plugin/` after the tag.
- `xdna-emu/build.rs` consists only of XRT plugin install logic.
- `xdna-emu/build_helpers/` and `xdna-emu/decoder_ffi/` directories no
  longer exist.
- `xdna_archspec::aie2` is the sole home of AIE2 `pub const` data.
- `xdna_archspec::aie2::memory_map` owns every derived const that used to
  live in `xdna-emu/src/device/registers_spec.rs`.
- `docs/arch/registers-memory-map.md` exists and states the const-first
  design principle.
- `ArchConfig` trait surface is unchanged in method count.
- `cargo test --lib`, `cargo test -p xdna-archspec`, bridge tests
  (full HW run), and the ISA test suite all pass at the
  `phase1-subsys-regs-mem` tag, with no regressions vs. the pre-Phase-1a
  baseline.
- Release build is clean.

## Out of scope (for follow-on work)

- AIE2P data or any other second-arch population.
- Reshape of `xdna_archspec::aie2::{stream_switch, packet, ctrl_packet,
  processor, timing, fot, isa}` -- these are the claims of Subsystems 3,
  5, 6, 7 respectively.
- Any trait seam (`DmaModel`, `LockModel`, `StreamSwitchModel`,
  `IsaDecoder`, `IsaExecutor`, `BinaryLoader`).
- Deep `TileType` / `TileKind` rename -- Subsystem 2.
- Large-file splits (`vmac_routing.rs` etc.) -- Phase 2 hygiene.
- Runtime-dispatch reduction on hot paths (monomorphization via an
  `Arch` type parameter with associated consts) -- follow-on to Phase 1
  once all seams are in.
