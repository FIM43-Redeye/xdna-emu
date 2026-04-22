# Subsystem 7 -- ISA Execute Audit

**Subsystem:** 7 of 8 (Phase 1b of the device-family refactor)
**Spec:** [../superpowers/specs/2026-04-21-subsys7-isa-execute-design.md](../superpowers/specs/2026-04-21-subsys7-isa-execute-design.md)
**Plan:** [../superpowers/plans/2026-04-21-subsys7-isa-execute.md](../superpowers/plans/2026-04-21-subsys7-isa-execute.md)

## Baseline (pre-subsystem, at phase1-subsys-stream-switch tag / HEAD)

- `cargo test --lib`: 2684 passed; 0 failed; 5 ignored
- `cargo test -p xdna-archspec --lib`: 297 passed; 0 failed; 2 ignored
- `cargo build --release`: clean
- Bridge smoke (`--no-hw -v add_one_cpp_aiecc`): green

Known pre-existing failures (carry through):
- `bd_chain_repeat_on_memtile` EMU deadlock (bridge suite; see NEXT-STEPS.md).

## Audit methodology

Per the spec, this audit is a per-file deep dive (Question 3 option
B) over the 20 files in `src/interpreter/execute/` plus the 9-file
`interpreter/timing/` submodule, grouped by functional area rather
than alphabetically.

Per-file subsection template:

- **Size + responsibility.** One sentence.
- **AIE2 hardcode count.** Grep count of literal `"AIE2"`,
  `AIE_ML_*`, `aie2`/`Aie2` identifiers, and arch-branded constants.
- **Divergence risks vs AIE1/AIE2P.** Evidence from file comments,
  llvm-aie TableGen, aie-rt per-arch headers.
- **Prescribed migration verb.** `move-to-archspec` /
  `read-archspec-via-accessor` / `wrap-in-trait` / `leave-alone`.
- **Estimated LOC impact.** Lines changing xdna-emu-side + lines
  added archspec-side.

Two files get ~2 pages each: `vmac_routing.rs` and `memory/mod.rs`.

---

## 1. Dispatcher / orchestration

Files: `execute/mod.rs`, `semantic.rs`, `cycle_accurate.rs`,
`vector_dispatch.rs`.

### `mod.rs`

- **Size + responsibility.** 160 lines; declares all execute submodules and re-exports the six top-level units (`CycleAccurateExecutor`, `VectorAlu`, `MemoryUnit`, `ControlUnit`, `StreamOps`, `CascadeOps`), plus the module-level dispatch-chain diagram in its doc comment.
- **AIE2 hardcode count.** 0 direct AIE2 identifiers; the doc comment describes the 384-bit cascade link and the dispatch chain, but contains no arch-branded constants. All references to hardware specifics are in prose comments only.
- **Divergence risks vs AIE1/AIE2P.** The dispatch chain order (`semantic -> VectorAlu -> MemoryUnit -> CascadeOps -> StreamOps -> ControlUnit`) is architectural: AIE1 has no cascade link, so `CascadeOps` would be a no-op. AIE2P is a superset and the order is expected to be identical. Risk is low -- the module is pure plumbing.
- **Prescribed migration verb.** `leave-alone`. The dispatch chain is arch-generic; nothing here is a hardcoded value.
- **Estimated LOC impact.** 0 xdna-emu changes; 0 archspec additions.

### `semantic.rs`

- **Size + responsibility.** 1678 lines; TableGen-driven scalar dispatch for ~40 `SemanticOp` variants covering arithmetic, bitwise, shift, comparison, pointer ops, sign/zero extension, control register writes (crSat, crRnd, crSRSSign, mask registers), and the cycle counter read.
- **AIE2 hardcode count.** 27 matches. Key instances: (1) `// The counter is 32-bit on AIE2 (wraps at 2^32)` (line 154); (2) control register IDs `9` (crSat), `6` (crRnd), `8` (crSRSSign), `16-19` (q0-q3), `28-31` (ql0-ql3), `32-35` (qh0-qh3) embedded as magic integers in the `write_dest` match arm (lines 280-318); (3) `// AIE2 has a unique flag architecture` in the module doc (line 30); (4) per-op comments like `// AIE2: ADD sets the Carry flag` citing TableGen Defs (lines 390, 400, 446, etc.); (5) `/// Per AIE2InstrPatterns.td section 4.1.11` (lines 607, 624). The carry flag comments and TableGen cross-references are documentation only; the control-register IDs at lines 280-318 are load-bearing hardcodes.
- **Divergence risks vs AIE1/AIE2P.** The control-register ID assignments (crSat=9, crRnd=6, crSRSSign=8, q0-q3=16-19, ql0-ql3=28-31, qh0-qh3=32-35) are derived from `AIE2GenRegisterInfo.td`. AIE1 has a different special-register layout (different IDs, no qh/ql split). The `SemanticOp::ReadCycleCounter` handler wraps the counter at 32 bits (line 154), which matches AIE2; AIE1 documentation does not specify a different width but should be verified. The carry flag semantics (ADD/SUB set carry, MUL/AND/OR do not) appear identical between AIE1 and AIE2 based on llvm-aie `AIEBaseInstrInfo.td`. Scalar arithmetic algorithms are arch-generic.
- **Prescribed migration verb.** `read-archspec-via-accessor`. The control-register IDs should move to archspec (extend `xdna_archspec::aie2::registers` or a new `aie2::ctrl_regs` submodule). The scalar algorithms and flag semantics stay in xdna-emu; they read arch-specific IDs via accessor at runtime.
- **Estimated LOC impact.** ~30 lines changing in xdna-emu (replace 7 literal match arms with archspec constants); ~40 lines added to archspec (new `ctrl_register_ids` module or extension of `registers`).

### `cycle_accurate.rs`

- **Size + responsibility.** 958 lines; the `CycleAccurateExecutor` struct and its `execute_internal` implementation covering: VLIW bundle dispatch (slots executed in `LoadA -> LoadB -> Store -> Scalar0 -> Scalar1 -> Vector -> Accumulator -> Control` order), pre-flight stall detection for cascade and stream ops, VLIW snapshot semantics, deferred load/store commit, hazard recording, branch delay slot bookkeeping, and timing-context event emission.
- **AIE2 hardcode count.** 9 matches. Key instances: `LatencyTable::aie2()` (line 87, constructs from LLVM FFI); `// AIE2 uses a write-back pipeline WITHOUT a hardware scoreboard` (line 302); `// AIE2 uses pure VLIW semantics` (line 427); `// AIE2 uses pure VLIW semantics... AIE2Schedule.td` (line 429); `// AIE2 has 5 delay slots` (line 536); `// per AIE2 quadrant mapping` (line 256). The most load-bearing is `LatencyTable::aie2()` at line 87 -- this is an explicit arch-dispatch call.
- **Divergence risks vs AIE1/AIE2P.** The 5-delay-slot rule is AIE2-specific (per `AIE2Schedule.td`; AIE1 has 6 delay slots per `AIE1InstrInfo.td`). The no-scoreboard write-back pipeline rule is AIE2-specific (AIE1 uses a scoreboard per `aie-rt` scheduling docs). The VLIW slot order is tied to the 8-slot AIE2 VLIW format; AIE1 has fewer slots. These are all _shape_ divergences. However, no timing values are hardcoded in this file -- they all delegate to `LatencyTable` and `HazardDetector`. The delay-slot count (5) is the clearest AIE2-specific constant in the execute path (line 536 comment references hardware but the 5-count is implicit in `tick_delay_slots`).
- **Prescribed migration verb.** `read-archspec-via-accessor`. The file delegates timing to `LatencyTable` (already archspec-driven) and hazard detection to the timing submodule. The 5-delay-slot behavior and scoreboard-vs-no-scoreboard rules are behavioral differences that would eventually need a trait method or config accessor if AIE1 is added. For Subsystem 7, no change is needed -- this file is already mostly arch-generic. The `LatencyTable::aie2()` call is the seam; it should eventually become `LatencyTable::for_arch(arch_handle::processor_model())`, but that is a naming refactor, not a behavioral migration.
- **Estimated LOC impact.** 0-5 xdna-emu changes (if delay-slot count is moved to archspec); 0 archspec additions (delay slots already in `processor::BRANCH_DELAY_SLOTS`).

### `vector_dispatch.rs`

- **Size + responsibility.** 111 lines; single `VectorAlu::execute()` entry point that pattern-matches on `SemanticOp` variants and dispatches to the individual `vector_*.rs` implementations.
- **AIE2 hardcode count.** 0 direct arch-branded constants. Imports `xdna_archspec::aie2::isa::SemanticOp` (explicit arch crate, but this is the same pattern as all other execute files -- they all target AIE2).
- **Divergence risks vs AIE1/AIE2P.** The dispatch table itself is `SemanticOp`-driven and arch-generic. However, two arms warrant note: (1) the SRS/UPS/Pack/Unpack/Convert fused-memory guard (lines 31-37) checks `op.slot.is_memory()` to avoid double-handling -- this is AIE2 slot semantics; (2) the MAC dispatch arm (lines 102-108) routes to `execute_matmul`, which is currently AIE2-only. If AIE1 MAC semantics differ, this arm would need a trait-method dispatch. For now, `leave-alone`.
- **Prescribed migration verb.** `leave-alone`.
- **Estimated LOC impact.** 0 xdna-emu changes; 0 archspec additions.

**Area summary.** The dispatcher/orchestration area is almost entirely arch-generic. The only genuine arch-specific content is the control-register IDs hardcoded in `semantic.rs` (8 magic integers mapping to crSat, crRnd, crSRSSign, and the q-register banks) and the implicit 5-delay-slot assumption in `cycle_accurate.rs` (where `processor::BRANCH_DELAY_SLOTS` is already in archspec but not yet consumed). Neither finding warrants an `IsaExecutor` trait method: both are *values* (register IDs, slot counts) not *shapes* (different algorithms). Migration verbs: `semantic.rs` → `read-archspec-via-accessor` for control register IDs; `cycle_accurate.rs` → wiring `LatencyTable` to processor model already in archspec (no new archspec additions needed). The remaining two files are `leave-alone`.

## 2. Scalar / control / stream / cascade

Files: `control.rs`, `stream.rs`, `cascade.rs`.

### `control.rs`

- **Size + responsibility.** 1088 lines; the `ControlUnit` struct handling branches (conditional/unconditional, five branch-condition variants), calls/returns, lock acquire/release, DMA start/wait, halt, and the `NeighborLocks` routing table for cross-tile lock access via the quadrant mapping.
- **AIE2 hardcode count.** 14 matches. Load-bearing instances: (1) lock ID quadrant boundaries 0-15 (South), 16-31 (West), 32-47 (North), 48-63 (East=Internal) embedded as literal integers in `route_lock()` (lines 490-530); (2) `// AIE2 lock quadrant mapping (per mlir-aie getLockLocalBaseIndex)` (lines 100-104, 201-204, 478-484); (3) `// AIE2 uses semaphore locks with 6-bit unsigned values (0-63)` (line 39); (4) `// AIE2 has two ACQ instruction variants` (line 382). The quadrant ID boundaries (0/16/32/48) are the primary hardcode; the 6-bit lock value range (0-63, which equals the per-tile 16 lock slots x 4 quadrants) is a structural fact.
- **Divergence risks vs AIE1/AIE2P.** Lock quadrant mapping (IDs 0-15 = South, etc.) is AIE2-specific, derived from `mlir-aie AIETargetModel::getLockLocalBaseIndex()`. AIE1 uses a different lock topology (no South neighbor locks for compute tiles; different quadrant boundaries). The quadrant offsets (16, 32, 48) are AIE2-specific constants that are NOT yet in archspec. The 6-bit lock value range is also AIE2-specific. Branch delay-slot handling delegates to `ExecutionContext::tick_delay_slots` (not in this file). Lock acquire/release semantics (delta-based semaphore: ACQ waits until value >= expected then subtracts delta, REL adds delta) appear structurally identical between AIE1 and AIE2 per aie-rt `xaie_locks_aieml.h` vs `xaie_locks_aie.h`.
- **Prescribed migration verb.** `read-archspec-via-accessor`. The quadrant ID boundaries (0, 16, 32, 48) and lock count (64 total) should move to archspec (extend `xdna_archspec::aie2::locks` or add a `lock_quadrant_offsets()` accessor). The lock acquire/release algorithms stay in xdna-emu.
- **Estimated LOC impact.** ~20 lines changing in xdna-emu (replace 4 magic boundaries); ~20 lines added to archspec (quadrant layout constants or existing locks module extension).

### `stream.rs`

- **Size + responsibility.** 459 lines; `StreamOps` handling stream read/write operations (`StreamReadScalar`, `StreamWriteScalar`, `StreamWritePacketHeader`) by routing through the `Tile`'s stream buffers, with blocking-read stall detection.
- **AIE2 hardcode count.** 5 matches, all in comments/docs: `// AIE2 uses streams for tile-to-tile communication` (line 4); `// Each AIE2 tile has stream switch ports` (line 21); `// AIE2 stream ops typically encode the port as an immediate operand` (line 83); one import of `xdna_archspec::aie2::isa::SemanticOp`. No load-bearing arch-specific constants in the implementation.
- **Divergence risks vs AIE1/AIE2P.** Stream port enumeration (which port index maps to which bundle type) is already archspec-resident via the stream-switch model from Subsystem 5. The stream read/write algorithm (push to FIFO, stall if empty on read) is arch-generic. AIE2P is expected to be identical at the instruction level. AIE1 has streams but the port numbering differs -- this is already handled by the stream-switch model's port-type assignments, not by this file.
- **Prescribed migration verb.** `leave-alone`.
- **Estimated LOC impact.** 0 xdna-emu changes; 0 archspec additions.

### `cascade.rs`

- **Size + responsibility.** 379 lines; `CascadeOps` implementing the 384-bit point-to-point cascade link -- reads from SCD (input FIFO), writes to MCD (output FIFO), with VLIW-safe pre-flight stall detection (`would_stall()`), and data-packing functions for vector (256-bit → 384-bit) and accumulator (512-bit → 384-bit) types.
- **AIE2 hardcode count.** 9 matches. All are structural: `// 384-bit data width = 6 x u64 = 48 bytes` (line 9); `// Depth-1 FIFO` (line 10); cascade direction register at `0x36060` (line 11); accumulator pack functions that explicitly use `[u64; 6]` (384 bits) and `[u64; 8]` (512 bits) layouts (lines 225-245). The 384-bit width is the canonical cascade width for AIE2.
- **Divergence risks vs AIE1/AIE2P.** AIE1 does **not** have a cascade link -- this subsystem would be entirely absent for AIE1. AIE2P is expected to retain a cascade link of the same width (no evidence of change in aie-rt or llvm-aie, but not verified). The 384-bit width (6 x u64) is intrinsic to how data is packed/unpacked; an AIE2P with a different cascade width would need different pack/unpack functions. This is the clearest example in this area of a genuinely arch-specific shape: cascade exists only on AIE2 (and presumably AIE2P), not on AIE1.
- **Prescribed migration verb.** `wrap-in-trait`. The cascade width (384 bits) and the FIFO depth are archspec data. However, the existence vs. non-existence of the cascade link is a shape difference, not a values difference. For Subsystem 7's purposes: the cascade data (width, register address) should move to archspec; for AIE1 support, a trait method returning `Option<&'static dyn CascadeModel>` would be cleanest, but this may be overkill if AIE1 is never targeted. **Pre-audit assessment was correct that cascade is arch-specific; the question is whether the shape-vs-values test warrants a trait method.** Given AIE1 simply has no cascade (not a different cascade), the simpler approach is an archspec feature flag (`has_cascade_link: bool`) readable via `arch_handle::processor_model()`, and the execute path gates on it. This avoids a trait method.
- **Estimated LOC impact.** ~15 lines changing in xdna-emu (guard cascade dispatch on `arch_handle::processor_model().has_cascade_link`); ~10 lines added to archspec (extend `processor` module with `has_cascade_link: bool`).

**Area summary.** No `IsaExecutor` trait methods warranted for this area. The control-register mappings in `control.rs` (lock quadrant IDs 0/16/32/48) are the primary data-migration target; they should extend the existing locks module in archspec. The 384-bit cascade width and register address in `cascade.rs` should migrate to archspec processor data, gated by a `has_cascade_link` feature flag -- not a trait method. `stream.rs` is clean (`leave-alone`). `control.rs` is `read-archspec-via-accessor`. `cascade.rs` is `move-to-archspec` for the constants.

## 3. Memory

Files: `memory/mod.rs`, `memory/neighbor.rs`. (Deep dive for `mod.rs`.)

### `memory/mod.rs` (deep dive)

- **Size + responsibility.** 3049 lines; the `MemoryUnit` struct handling the full load/store surface: scalar loads/stores, vector load-A/load-B/4x-gather, fused load+compute (vlda.ups, vlda.conv), fused compute+store (vst.srs, vst.pack, vst.conv), vector store, partial-word stores (deferred RMW model for st.s8/st.u8/st.s16/st.u16), bank-conflict recording, and the processor-bus window (read/write to tile config registers via 0x80000). The file also contains `read_memory()` / `write_memory()` as arch-generic primitives.

- **AIE2 hardcode count.** ~40 matches. Key load-bearing instances:
  - `const OFFSET_MASK: u32 = xdna_archspec::aie2::compute::MEMORY_SIZE as u32 - 1` (line 43) -- already archspec-sourced.
  - `// AIE2 cores use a 20-bit data address space... CardDir 7=East(local on AIE2)` (lines 47-51).
  - `// On AIE2 (IsCheckerBoard=0), East is always the local tile's own memory` (line 20-22) -- architectural statement about the checkerboard flag; the flag itself is already in archspec as `compute::IS_CHECKERBOARD`.
  - `// AIE2 partial-word stores... II_STHB operand latency = 7 in AIE2Schedule.td` (line 314-316).
  - `// On AIE2, all data memory accesses... have the same pipeline depth (7 cycles)` (line 65).
  - `// On AIE2, core loads have the same pipeline latency for ALL data...` (line 2445).
  - `// On AIE2, core loads from neighbor memory have the SAME pipeline` (line 2994).
  - `const PROC_BUS_BASE: u32 = 0x80000` (line 1386, 1491) -- the processor bus base address for accessing tile config registers; not yet in archspec.
  - `const PROC_BUS_END: u32 = 0xC0000` (line 1387) -- processor bus window size (256KB); not yet in archspec.

- **Structural breakdown.** The 3049 lines decompose into recognizable subsystems:
  - **Entry point / dispatch** (lines 83-181): `MemoryUnit::execute()` dispatches on `op.semantic` × `op.is_vector` × `op.slot` to one of ~13 handler functions.
  - **Scalar load/store** (lines 183-291): `execute_load()` / `execute_store()` -- bank conflict tracking, address decode, deferred write for partial-width stores.
  - **Vector loads** (lines 343-573): `execute_vector_load_a()`, `execute_vector_load_b()`, `execute_vector_load_4x()`, `execute_vector_load_unpack()` -- each reads 256-bit aligned blocks from memory, with element-width-driven decode.
  - **Fused load+compute** (lines 765-865): `execute_fused_load_ups()`, `execute_fused_load_convert()` -- load from memory, then apply SRS/UPS pipeline in-register.
  - **Fused compute+store** (lines 866-1090): `execute_fused_store_srs()`, `execute_fused_store_pack()`, `execute_fused_store_convert()` -- apply pipeline operation, then store.
  - **Vector store** (lines 1092-1348): `execute_vector_store()` with deferred partial-word mechanics.
  - **Memory primitives** (lines 1370-1700): `read_memory()`, `write_memory()`, `read_vector_from_memory()`, `write_vector_to_memory()` -- the arch-generic read/write core using `MemoryQuadrant` routing, which is already archspec-driven.
  - **Test suite** (lines 1731+): 1300+ lines of unit tests covering scalar/vector loads and stores across all cardinal directions.

- **Arch-generic vs arch-specific split.** The vast majority of the code is arch-generic in algorithm:
  - The element-width dispatch tables (8-bit, 16-bit, 32-bit loads) are data-driven by `ElementType` and `MemWidth`, not by AIE2-specific enumerations.
  - The CardDir routing (`decode_data_address`, `MemoryQuadrant`) already reads from archspec constants (`compute::MEMORY_SIZE`, `cardinal::EAST`, `compute::IS_CHECKERBOARD`).
  - The bank-conflict tracking delegates to `tile.num_banks()` which reads from the tile's archspec-derived configuration.
  - The deferred load latency (`LATENCY_MEMORY = 7`) is the primary remaining hardcode; it lives in `timing/latency.rs` (not here) but is consumed via `load_latency_for_address()` (line 74-76), which currently always returns 7.
  - **Arch-specific fragments:** The processor bus addresses (`PROC_BUS_BASE = 0x80000`, `PROC_BUS_END = 0xC0000`) are hardcoded raw constants not yet in archspec. These are addresses into the tile's configuration register space and differ from the data memory address space.

- **Key question: does this justify a `memory_load`/`memory_store` trait method?** No. The memory dispatch algorithm (element-type dispatch, CardDir routing, bank conflict tracking) is uniform across AIE1/AIE2/AIE2P. The only divergence is:
  1. Which CardDir maps to local (East=local on AIE2, alternates on AIE1/checkerboard) -- already handled via `compute::IS_CHECKERBOARD` in archspec.
  2. The processor bus base address -- a constant, not an algorithm.
  3. Load latency -- a constant consumed from `timing/latency.rs`.
  All are *values*, not *shapes*. Memory load/store stays as arch-generic code reading archspec constants via `arch_handle::*` accessors.

- **Data migration candidates.** 
  - `PROC_BUS_BASE = 0x80000` and `PROC_BUS_END = 0xC0000`: move to archspec as `compute::PROC_BUS_BASE` and `compute::PROC_BUS_WINDOW_SIZE` (or derive from existing memory map constants). ~2 constants.
  - `load_latency_for_address()` (line 74-76): currently returns `LATENCY_MEMORY` regardless of address. Should consume `xdna_archspec::aie2::timing::DATA_MEMORY_LATENCY` directly (already exists in archspec). The indirection via `timing/latency.rs::LATENCY_MEMORY` is the only remaining duplication.

- **Prescribed migration verb.** `read-archspec-via-accessor` (for the 2 processor bus constants and the latency indirection).
- **Estimated LOC impact.** ~10 lines changing in xdna-emu; ~10 lines added to archspec (processor bus constants extension).

### `memory/neighbor.rs`

- **Size + responsibility.** 166 lines; `NeighborMemory` struct providing lazy copy-on-access snapshots of adjacent tiles' data memory (indexed by `MemoryQuadrant`), with a write-buffer for deferred cross-tile stores applied after the core step.
- **AIE2 hardcode count.** 1: `use xdna_archspec::aie2::SHIM_ROW` (line 8) -- already archspec-sourced.
- **Divergence risks vs AIE1/AIE2P.** The AIE1 checkerboard topology (East is a real neighbor on some rows, local on others) is explicitly noted in comments (line 29-32); the code already handles this correctly because `MemoryQuadrant::from_address()` maps CardDir to the appropriate quadrant based on `IS_CHECKERBOARD`. No algorithmic changes are needed for AIE1. AIE2P is expected to be identical.
- **Prescribed migration verb.** `leave-alone`.
- **Estimated LOC impact.** 0.

**Deep-dive summary.** `memory/mod.rs` is predominantly arch-generic. The address routing is already data-driven from archspec constants (`MEMORY_SIZE`, `IS_CHECKERBOARD`, `cardinal::*`). The two remaining hardcodes are the processor bus base address (0x80000/0xC0000, ~2 constants to migrate) and the load latency indirection (which already exists in archspec as `timing::DATA_MEMORY_LATENCY`, just not yet consumed directly). No `IsaExecutor` trait method is warranted for the memory subsystem: all divergence is reducible to data constants. `neighbor.rs` is clean. The AIE1 checkerboard-local ambiguity is already handled architecturally.

## 4. Vector ALU

Files: `vector_arith.rs`, `vector_compare.rs`, `vector_misc.rs`,
`vector_pack.rs`, `vector_ups.rs`, `vector_srs.rs`, `vector_helpers.rs`,
`vector_semantic.rs`, `vector_permute.rs`, `vector_float.rs`,
`vector_config.rs`, `vector_convert.rs`, `vector_validate.rs`.

### `vector_arith.rs`

- **Size + responsibility.** 2468 lines; arithmetic, accumulate, and accumulator operations for the vector ALU -- `vector_addsub`, `execute_add`, `execute_binary_elementwise`, `execute_accumulate`, `execute_accum_wide` (for 512-bit bm and 1024-bit cm accumulator paths), and `execute_neg` on accumulators.
- **AIE2 hardcode count.** 14. Key instances: `// AIE2 NaN semantics: when either operand is NaN, always return s1` (lines 254, 358); `aie2_acc_fp32_add` calls for bf16 accumulator add (lines 1613, 1626, 1627, 1658, 1668, 1669); `// Wide (cm, 1024-bit) vs narrow (bm, 512-bit) accumulator` (line 1586). The NaN semantics and fp32 accumulator paths are AIE2-specific behavior derived from hardware observation. The 512/1024-bit register widths are expressed via `VECTOR_REGISTER_BITS` from archspec (indirectly), but the hard branches on width in `execute_accum_wide` use literal `256/et.bits()` (line 1228).
- **Divergence risks vs AIE1/AIE2P.** AIE2 NaN behavior (always produce AIE2 canonical NaN regardless of input sign) is hardware-specific and differs from IEEE 754. AIE1 may have different NaN canonicalization (not verified; `vector_float.rs` comment at line 308 notes the real hardware differs from the aietools Python model). Accumulator width (512/1024 bit) is AIE2-specific; AIE1 uses narrower accumulators.
- **Prescribed migration verb.** `leave-alone` for the arithmetic algorithms; the `aie2_acc_fp32_add` function in `vector_float.rs` encapsulates AIE2-specific float behavior and is already factored out. The NaN behavior is AIE2 hardware ground truth; no constant to migrate.
- **Estimated LOC impact.** 0.

### `vector_compare.rs`

- **Size + responsibility.** 920 lines; vector comparisons: `SetGe`, `SetLt`, `SetEq`, `MaxLt`, `MinGe`, `Cmp`, `VectorSelect`.
- **AIE2 hardcode count.** 1 import of `xdna_archspec::aie2::isa::SemanticOp`; no load-bearing arch-specific constants in compare logic.
- **Prescribed migration verb.** `leave-alone`.
- **Estimated LOC impact.** 0.

### `vector_misc.rs`

- **Size + responsibility.** 931 lines; miscellaneous vector ops: absolute value variants (`AbsGtz`, `NegGtz`, `NegLtz`), conditional sub (`SubLt`, `SubGe`), and `MaxDiffLt`.
- **AIE2 hardcode count.** 2 (import + one comment). No load-bearing arch constants.
- **Prescribed migration verb.** `leave-alone`.
- **Estimated LOC impact.** 0.

### `vector_pack.rs`

- **Size + responsibility.** 1035 lines; pack/unpack operations (`Pack`, `Unpack`) -- narrowing wide elements into packed byte lanes and widening.
- **AIE2 hardcode count.** 1 (import). No load-bearing arch constants; the element-width dispatch is `ElementType`-driven.
- **Prescribed migration verb.** `leave-alone`.
- **Estimated LOC impact.** 0.

### `vector_ups.rs`

- **Size + responsibility.** 942 lines; UPS (upshift / type-widening) operations -- promotes narrow integer lanes to wider accumulator lanes with optional left shift.
- **AIE2 hardcode count.** 4 (3 imports + `// Hardware behavior (derived from AIE2 architecture)` in doc). The UPS mode table (`ups_mode()` function at lines 63-70) is a 4-entry table encoding the valid lane/width combinations (8→32, 16→32, 16→64, 32→64). This table is logically archspec material; it describes what type-promotion paths the AIE2 multiply array supports.
- **Divergence risks vs AIE1/AIE2P.** AIE1's narrower 128-bit vector unit would support fewer UPS modes (e.g., no 32-lane paths since that requires a 256-bit register). The mode table would change. This is a values-migration candidate.
- **Prescribed migration verb.** `move-to-archspec`. The `ups_mode()` table (4 entries) should move to `xdna_archspec::aie2::isa` or processor module; the algorithm (sign extension + shift + mask) stays in xdna-emu.
- **Estimated LOC impact.** ~10 lines changing in xdna-emu; ~15 lines added to archspec (UPS mode table with accessor).

### `vector_srs.rs`

- **Size + responsibility.** 1483 lines; SRS (Shift-Round-Saturate) rounding mode definitions (`RoundingMode` enum with 10 modes), BIAS constant (`SRS_SHIFT_BIAS` from archspec at line 32), rounding decision logic (`srs_round()`), saturation logic (`srs_saturate()`), and the `execute_srs` / `execute_srs_narrow` / `execute_srs_wide` dispatch.
- **AIE2 hardcode count.** 7 (mostly doc references; `const BIAS: u32 = xdna_archspec::aie2::processor::SRS_SHIFT_BIAS as u32` at line 32 is already archspec-sourced).
- **Divergence risks vs AIE1/AIE2P.** The 10-mode rounding set (Floor, Ceil, SymFloor, SymCeil, NegInf, PosInf, SymZero, SymInf, ConvEven, ConvOdd -- modes 0-3 and 8-13) is AIE2-specific. The rounding algorithms (based on `sgn`, `lsb`, `grd`, `stk` bits) are the same rounding-mode algorithms that IEEE 754 defines; only the index encoding and the set of supported modes differ per arch. AIE1's SRS supports a smaller set of rounding modes (not all 10; exact subset unknown without deeper AIE1 research). **This is the primary candidate for an `IsaExecutor` trait method** (`apply_srs`) -- but the pre-audit hypothesis needs refining. The rounding *algorithms* are arch-generic (the sgn/lsb/grd/stk computation is standard); only the valid *mode indices* differ. This means the divergence is in the `RoundingMode` enumeration and mode table, not in the algorithm itself. A simpler resolution: the rounding-mode enum (10 modes, with gaps at 4-7 and 14-15) is AIE2-specific data that could move to archspec, while the algorithm that executes a given `RoundingMode` stays in xdna-emu. An AIE1 port would populate a smaller rounding-mode table in archspec; the algorithm in xdna-emu would receive whichever mode archspec reports as valid and execute it. **This resolves the SRS question without a trait method.**
- **Prescribed migration verb.** `move-to-archspec` for the `RoundingMode` enum and valid-index table; `leave-alone` for the rounding and saturation algorithms.
- **Estimated LOC impact.** ~50 lines changing in xdna-emu (move `RoundingMode` enum to archspec, update imports); ~60 lines added to archspec.

### `vector_helpers.rs`

- **Size + responsibility.** 1383 lines; shared vector utilities consumed by `vector_*.rs` implementations -- accumulator read/write adapters (`get_acc_source`, `write_acc_dest`), vector register read/write helpers (`read_vector_source`, `write_vector_dest`), wide (512-bit/1024-bit) helpers, `get_shift_amount`, `execute_binary_typeless`, etc.
- **AIE2 hardcode count.** 0. All width constants (256 bits, 512 bits) are derived from `VECTOR_REGISTER_BITS`/`VECTOR_PAIR_BITS` from archspec (lines 406-450 area).
- **Prescribed migration verb.** `leave-alone`.
- **Estimated LOC impact.** 0.

### `vector_semantic.rs`

- **Size + responsibility.** 1421 lines; `execute_shuffle`, `execute_vector_broadcast`, `execute_vector_extract`, `execute_vector_insert`, `execute_vector_push`, `execute_align`, `execute_copy`, `execute_vector_clear` -- vector data-movement operations.
- **AIE2 hardcode count.** 2 (imports). No load-bearing arch constants.
- **Prescribed migration verb.** `leave-alone`.
- **Estimated LOC impact.** 0.

### `vector_permute.rs`

- **Size + responsibility.** 1786 lines; shuffle unit (48 modes via `SHUFFLE_ROUTING` byte-level table at line 1545) and MAC permutation engine (26 modes via `MacPermuteMode` enum). The `SHUFFLE_ROUTING` table (48 × 64 bytes) is hardware-probed from real AIE2 silicon (comment at line 1542-1544). The `VEC_BYTES` and `PAIR_BYTES` constants are already archspec-sourced (lines 60-63).
- **AIE2 hardcode count.** 7. Key: `SHUFFLE_ROUTING` table (probed from AIE2 silicon, 3072 bytes); `MacPermuteMode` enum (26 variants, AIE2-specific); `// Hardware constraint: the AIE2 has acc_num accumulators (32 for...` (line 1036); `// The AIE2 has 512 multipliers feeding acc_num accumulators` (line 1170).
- **Divergence risks vs AIE1/AIE2P.** Both the shuffle routing table and the MAC permutation modes are entirely AIE2-specific. AIE1 has a different shuffle unit with different mode numbers and different byte routing. AIE2P may share the same tables or have extensions. These are the largest data-migration candidates in the vector ALU area: both the `SHUFFLE_ROUTING` table and the `MacPermuteMode` enum are pure archspec material.
- **Prescribed migration verb.** `move-to-archspec`. The `SHUFFLE_ROUTING` table (3072 bytes of hardware-probed data) and the `MacPermuteMode` enum (26 variants with their `rc2i` geometry) should move to `xdna_archspec::aie2::vmac` or a new `aie2::permute` module. The algorithm that applies the table (byte-level transpose logic) stays in xdna-emu.
- **Estimated LOC impact.** ~3200 lines moving from xdna-emu to archspec; the execute code in xdna-emu shrinks to ~300 lines of algorithm plus archspec references.

### `vector_float.rs`

- **Size + responsibility.** 1325 lines; all AIE2-specific BF16 and FP32 semantics: flush-to-zero (FTZ) for denormals, canonical NaN bit pattern (mantissa=1, positive, regardless of input; hardware-verified against NPU1 at line 308), bf16/fp32 conversion with rounding, the duplicate `RoundingMode` enum for float conversions, `aie2_fp32_add`/`aie2_acc_fp32_add` with FTZ and NaN propagation.
- **AIE2 hardcode count.** 50 (the highest of any vector file). This file is almost entirely AIE2-specific behavior. Key hardcodes: canonical NaN pattern `0x01` mantissa (line 316); FTZ rule; the 10-mode `RoundingMode` enum (duplication with `vector_srs.rs`!); `sgn_mag=True` convention for bf16 conversion rounding.
- **Divergence risks vs AIE1/AIE2P.** The canonical NaN pattern (mantissa=1, always positive) is hardware-verified AIE2 behavior that differs from IEEE 754 and from the aietools Python model (which uses mantissa=0x7F). AIE1's float handling is different (narrower pipeline, different NaN conventions). This file is entirely AIE2-specific floating-point microarchitecture. **Notable:** `vector_float.rs` defines `RoundingMode` independently from `vector_srs.rs`'s `RoundingMode` -- the two enums are identical. This duplication should be resolved by a single `RoundingMode` in archspec.
- **Prescribed migration verb.** `move-to-archspec` for the `RoundingMode` enum (consolidating the duplicate); `leave-alone` for the FTZ/NaN algorithms (they are AIE2 behavioral facts, but they belong in xdna-emu as execute-side implementations, not data). The canonical NaN constant (`0x01` mantissa, positive sign) could become an archspec constant in `aie2::processor`.
- **Estimated LOC impact.** ~15 lines migrated (RoundingMode enum consolidation + canonical NaN constant); 0 for the algorithms.

### `vector_config.rs`

- **Size + responsibility.** 1067 lines; matrix multiply configuration word parser -- `MatMulConfig` struct, `AccWidth` enum, `DENSE_GEOMETRY_TABLE` (8 entries) and `SPARSE_GEOMETRY_TABLE` (5 entries) encoding the valid (bits_x, bits_y, rows, inner, cols, acc_cmb, bfloat, sparse) tuples for AIE2's multiply array.
- **AIE2 hardcode count.** 4 (doc comments). Key: `DENSE_GEOMETRY_TABLE` and `SPARSE_GEOMETRY_TABLE` (13 entries total) are pure AIE2 archspec data -- they encode what matrix geometries the hardware multiplier array supports.
- **Divergence risks vs AIE1/AIE2P.** The geometry tables are entirely AIE2-specific. AIE1 supports a smaller set of matrix geometries (narrower multiplier array). These tables are the clearest data-migration target in the vector-ALU area.
- **Prescribed migration verb.** `move-to-archspec`. Both geometry tables should move to `xdna_archspec::aie2::vmac` (same module as the VMAC routing tables). The `MatMulConfig` struct and its parsing logic stay in xdna-emu.
- **Estimated LOC impact.** ~30 lines moving to archspec; ~15 lines changing in xdna-emu (update imports and accessor calls).

### `vector_convert.rs`

- **Size + responsibility.** 507 lines; type conversions between vector element types -- FP32↔BF16, INT↔FLOAT, with AIE2 FTZ applied.
- **AIE2 hardcode count.** 4 (doc references to `AIE2 FTZ` at lines 251, 259, 318). All are behavioral facts implemented via `vector_float.rs` helpers; no raw constants.
- **Prescribed migration verb.** `leave-alone`.
- **Estimated LOC impact.** 0.

### `vector_validate.rs`

- **Size + responsibility.** 284 lines; test module only (gated by `#[cfg(test)]`), exercising vector ALU operations.
- **AIE2 hardcode count.** 1 (import). No production constants.
- **Prescribed migration verb.** `leave-alone`.
- **Estimated LOC impact.** 0.

**Vector ALU area summary.** No `IsaExecutor` trait methods warranted. The pre-audit hypothesis that SRS rounding would require a trait method is resolved: the rounding *algorithms* (sgn/lsb/grd/stk logic) are arch-generic; only the rounding-mode enum (10 modes with specific gaps) and the UPS mode table (4 valid type-pair entries) are arch-specific data. Both move to archspec without trait methods. The largest single migration in this area is `vector_permute.rs`'s `SHUFFLE_ROUTING` table (~3072 bytes of hardware-probed data + the `MacPermuteMode` 26-variant enum) and `vector_config.rs`'s geometry tables. Notable finding: `RoundingMode` is duplicated between `vector_srs.rs` and `vector_float.rs`; consolidation into a single archspec definition is the right fix. `vector_float.rs` (50 AIE2 references) is the most AIE2-entangled file in this area but its entanglement is behavioral facts, not data constants -- it stays in xdna-emu. Files that are `leave-alone`: `vector_arith`, `vector_compare`, `vector_misc`, `vector_pack`, `vector_helpers`, `vector_semantic`, `vector_convert`, `vector_validate`.

## 5. VMAC / matmul

Files: `vmac_routing.rs` (deep dive), `vmac_hw.rs`, `vector_matmul/`.

### `vmac_routing.rs` (deep dive)

- **Size + responsibility.** 2862 lines; pure generated static data -- no algorithmic logic. Contains exactly 5 data structures plus the two evaluation functions `eval_prmx` and `eval_prmy`:
  1. `PRMX_MBIT_IDX: [i16; 789]` -- maps each of the 789 active X-crossbar m-bits to a route-table index (lines 9-50).
  2. `PRMX_OVR_START: [u32; 790]` -- start offsets into the output-routing array (lines 52-...).
  3. `PRMX_OVR_OUT: [u16; 15808]` -- output byte positions for each route entry.
  4. `PRMX_OVR_IN: [u16; 15808]` -- input byte positions for each route entry.
  5. `PRMY_ROUTE_0..25: [[i16; 512]; 26]` -- 26 Y-permute routing tables (one per pmode bit).
  The evaluation functions (`eval_prmx`, `eval_prmy` at lines 1891 and 2822) are ~15 lines each of pure index arithmetic consuming the above tables.

- **Data shape.** 789 active m-bits, 15808 route entries (PRMX tables), and 26 × 512 = 13312 Y-route entries (PRMY tables). Together these are ~120KB of static data. This is the AIE2 X-crossbar and Y-permute wiring, generated by probing the AMD C++ ISS (comment line 1-3). No algorithmic content; the file's only job is to be `include!`'d into `vmac_hw.rs`'s `routing` submodule (line 15 of `vmac_hw.rs`).

- **Consumer interface.** `vmac_hw.rs` re-exports `eval_prmx` and `eval_prmy` (line 19). These two functions are called at lines 1053 and 1062 of `vmac_hw.rs` only. No other file in the codebase calls them directly -- confirmed by the lack of any other import. The interface is minimal: two functions, each taking a byte slice and returning a byte slice.

- **Should it move wholesale to archspec?** Yes. The data is entirely AIE2-specific, has zero algorithmic content, and is already semantically archspec material (it's the AIE2 VMAC crossbar wiring). The file is `include!`'d rather than `use`'d, which makes the move straightforward: the tables become a module in `xdna_archspec::aie2::vmac::routing`, and `vmac_hw.rs` updates its `include!` path or switches to `use` imports. Alternatively, the `eval_prmx`/`eval_prmy` functions can be re-exported from archspec directly, keeping the call sites in `vmac_hw.rs` unchanged modulo module path.

- **Prescribed migration verb.** `move-to-archspec` (wholesale). The entire file moves; `vmac_hw.rs` updates two imports.
- **Estimated LOC impact.** ~2862 lines moving from xdna-emu to archspec; ~3 lines changing in `vmac_hw.rs` (update the `include!` path or `use` path).

### `vmac_hw.rs` (deep dive)

- **Size + responsibility.** 1824 lines; the VMAC hardware pipeline model with:
  - `decode_mask()` and `mask2sel()`: sparse mask decoding (lines 32-686).
  - `sgex_mask()` / `sgey_mask()`: sign extension control masks (lines 252-299).
  - `mpyl_hw_lane()`: low-path 16-lane multiplier (lines 305-450).
  - `psal_hw_lane()`: post-multiply PSA accumulator adder (lines 451-680+).
  - `vec_control()`: parse config word into one-hot mmode + pmode encoding (lines 688-800).
  - `vec_control_negate()`: negate-lane mask for subtract modes (lines ~800-900).
  - `build_prmx_control()`: build the 13-word PRMX control word from smode + pmode (lines ~900-1010).
  - `sparse_vmac()`: full sparse pipeline entry point (lines 1011-1200).
  - Large test suite (lines 1200+).

- **Arch-generic vs arch-specific split:**
  - **(a) Arch-generic (algorithm):** The multiply/accumulate algorithm (`mpyl_hw_lane`, `psal_hw_lane`, `sgex_mask`, `sgey_mask`) is mathematically general -- multiply, post-add tree, sign extension. The same algorithm shape applies to any 512-multiplier array regardless of element widths. ~400 lines.
  - **(b) Arch-specific constants:** The mmode/pmode encoding in `vec_control()` (lines 703-800) maps `(amode, bmode, variant)` tuples to specific one-hot patterns. These 26 pmode bits correspond exactly to the 26 MAC permutation modes in `vector_permute.rs::MacPermuteMode`. They are AIE2-specific. Similarly, the sign extension patterns in `sgex_mask`/`sgey_mask` encode AIE2-specific groupings. ~150 lines of const data embedded in match arms.
  - **(c) Direct consumer of `vmac_routing.rs` tables:** `eval_prmx` (line 1053) and `eval_prmy` (line 1062) -- two call sites. After `vmac_routing.rs` moves to archspec, these two calls update their import path, not their call signature.

- **Trait candidate analysis.** The question is: does `vmac_route(mbit, pmode)` need a trait method? The answer is no: the routing is pure data lookup (`eval_prmx(a_dense, &prmx_m)`) with no arch-specific algorithm -- the algorithm is the same index-table walk regardless of arch. What changes per arch is which tables exist. This is a data migration, not a trait method: `arch_handle::vmac_routing_table()` returning the right static tables for the current arch, with `eval_prmx`/`eval_prmy` staying in xdna-emu (or archspec as helpers alongside the data). The spec's candidate `fn vmac_route(&self, mbit: u16, pmode: u8) -> VmacRoute` is rejected: routing is data, not shape.

- **Prescribed migration verb.** `read-archspec-via-accessor` (after `vmac_routing.rs` moves to archspec, the call sites in `vmac_hw.rs` read via the archspec module path); `extract-constants` for the mmode/pmode encoding table in `vec_control()`.
- **Estimated LOC impact.** ~150 lines changing in xdna-emu (update import paths + extract pmode encoding); ~20 lines added to archspec (pmode encoding table or enum, if extracted).

### `vector_matmul/`

- **Size + responsibility.** 2770 lines total:
  - `mod.rs` (1801 lines): main entry point `execute_matmul()` which reads the config register, branches on sparse/dense mode, packs inputs into byte arrays, calls `sparse_vmac()` from `vmac_hw.rs` or the density path, and writes back to accumulators.
  - `bf16_pipeline.rs` (486 lines): `bf16_mac_hw_lane()` for the BF16 MAC path.
  - `helpers.rs` (483 lines): byte packing/unpacking utilities (`vec512_to_bytes`, `extract_element_bytes`, etc.) and `matmul_dense`/`matmul_sub` legacy API.

- **Relationship to `vmac_hw.rs`.** `vector_matmul/mod.rs` directly calls `super::vmac_hw::sparse_vmac()` (line 256). The `matmul_dense` and `matmul_sub` functions in `helpers.rs` implement a simpler element-wise accumulate path (not the full crossbar routing). The two paths are:
  - **Dense path** (`execute_matmul()` via helpers): config-driven element-wise multiply for the non-sparse modes, delegating to `matmul_dense`.
  - **Sparse path** (via `vmac_hw::sparse_vmac`): full crossbar routing + PSA tree for sparse modes.

- **AIE-specific content.** The dense path in `mod.rs` uses `MatMulConfig` (from `vector_config.rs`, AIE2-specific geometry tables) to determine tile dimensions. The bf16 pipeline reads `aie2_acc_fp32_add` from `vector_float.rs`. No standalone hardcoded constants; all arch-specific content is delegated to `vmac_hw.rs` and `vector_config.rs`.
- **Prescribed migration verb.** `leave-alone` for `mod.rs` and `helpers.rs`; `leave-alone` for `bf16_pipeline.rs`. The AIE2-specific content enters through `vmac_hw.rs` and `vector_config.rs` which are already targeted for migration.

### AIE1 crossbar projection

AIE1 has VMAC instructions (`vmac.80` for 80-bit accumulators, `vmac.48` for 48-bit) per `AIE1InstrInfo.td` lines 1110-1139. The register types are `ACC768` (80-bit × accumulator bank) and `VEC1024` for the extended input buffer -- structurally different from AIE2's `[u64; 8]` (512-bit) and `[u64; 16]` (1024-bit). AIE1's multiplier array is 48-bit/80-bit output, indicating different hardware width. The crossbar table would be a different size with different m-bit counts. Evidence: AIE1 defines 5 VMAC variants vs AIE2's 26+ pmode combinations, suggesting a smaller but structurally similar crossbar. An AIE1 port would provide its own routing tables in `xdna_archspec::aie1::vmac::routing`; the `eval_prmx`/`eval_prmy` functions (the algorithm) might be reusable if the data shape is structurally compatible, or would need AIE1-specific variants if the table format differs (e.g., if AIE1's m-bit encoding is different in the ISS probing format).

**VMAC area summary.** `vmac_routing.rs` moves wholesale to archspec -- zero algorithmic content, pure AIE2 probed data. `vmac_hw.rs` updates its import path; the mmode/pmode encoding may also be extracted to archspec. The spec's candidate `vmac_route()` trait method is rejected: routing is data, not shape divergence. `vector_matmul/` is `leave-alone`; it delegates all AIE2-specific content through files already targeted for migration.

## 6. Timing

Files: `interpreter/timing/{arbitration, barrier, deadlock, hazards,
latency, memory, mod, slots, sync}.rs`, plus `execute/cycle_accurate.rs`
latency tables.

### `timing/latency.rs`

- **Size + responsibility.** 753 lines; instruction latency constants (scalar, vector, memory, branch, lock), the `LatencyTable` struct (two-tier: LLVM itinerary primary + `SemanticOp` fallback), `validated_aie2()` cross-validation against `ProcessorModel`, and the `itinerary_to_timing()` name→timing mapper.
- **AIE2 hardcode count.** 30 matches. Key load-bearing instances:
  - `LATENCY_MEMORY: u8 = 7` (line 103) -- raw literal not yet reading from `timing::DATA_MEMORY_LATENCY` (which does exist in archspec at `xdna_archspec::aie2::timing::DATA_MEMORY_LATENCY`).
  - `LATENCY_SCALAR_MUL: u8 = 2` (line 79), `LATENCY_SCALAR_DIV: u8 = 6` (line 83), `LATENCY_VECTOR_MUL: u8 = 5` (line 142), `LATENCY_VECTOR_MAC: u8 = 5` (line 146), `LATENCY_VECTOR_SIMPLE: u8 = 2` (line 139), `LATENCY_BRANCH_TAKEN: u8 = 3` (line 121) -- all raw literals.
  - The `validated_aie2()` function at line 363-388 already validates `LATENCY_MEMORY` against `ProcessorModel.load_latency + 2` and `LATENCY_BRANCH_TAKEN` against `ProcessorModel.mispredict_penalty - 1`. This validation infrastructure exists but the primary source of truth is still the raw literal, not the archspec constant.
  - `LatencyTable::aie2()` (line 196) explicitly names the architecture in the constructor -- the seam for future `LatencyTable::for_arch(arch)`.
- **What's already in archspec.** `xdna_archspec::aie2::timing` contains: `DATA_MEMORY_LATENCY`, `LOCK_ACQUIRE_LATENCY`, `LOCK_RELEASE_LATENCY`, `BRANCH_PENALTY`, `ROUTE_LOCAL_TO_EXTERNAL`, and more. The `timing/memory.rs` file already reads `DATA_MEMORY_LATENCY` from archspec (line 179 of `memory.rs`). The duplication is in `latency.rs` itself, which re-declares `LATENCY_MEMORY = 7` independently.
- **Divergence risks vs AIE1/AIE2P.** Every latency value in this file is AIE2-specific. AIE1 has different pipeline depths and different memory latency (per AM020 for AIE1). The `SemanticOp`-based fallback table is arch-agnostic in structure but arch-specific in values. The `validated_aie2()` function creates a one-way dependency on `ProcessorModel` format; AIE1 would need `validated_aie1()` with different target values.
- **Prescribed migration verb.** `read-archspec-via-accessor`. The raw literals (`LATENCY_MEMORY=7`, `LATENCY_SCALAR_MUL=2`, etc.) should be replaced with reads from `xdna_archspec::aie2::timing::*` or `xdna_archspec::aie2::processor::*`. `LATENCY_MEMORY` duplicates `xdna_archspec::aie2::timing::DATA_MEMORY_LATENCY` exactly.
- **Estimated LOC impact.** ~20 lines changing in xdna-emu (replace literals with archspec references); 0 additions to archspec (all target constants already exist).

### `timing/memory.rs`

- **Size + responsibility.** 921 lines; memory bank conflict detection, `MemoryQuadrant` routing enum, cross-tile latency, bank size constants, alignment checking.
- **AIE2 hardcode count.** 20. Key instances: all primary constants are already archspec-sourced:
  - `NUM_BANKS = xdna_archspec::aie2::compute::PHYSICAL_BANKS` (line 43).
  - `QUADRANT_SIZE = xdna_archspec::aie2::compute::MEMORY_SIZE` (line 68).
  - `CROSS_TILE_LATENCY = xdna_archspec::aie2::timing::ROUTE_LOCAL_TO_EXTERNAL` (line 72).
  - `BASE_LATENCY = xdna_archspec::aie2::timing::DATA_MEMORY_LATENCY` (line 179).
  - `BANK_SIZE = xdna_archspec::aie2::compute::PHYSICAL_BANK_SIZE` (line 173).
  - `BANK_WIDTH_BYTES = xdna_archspec::aie2::compute::PHYSICAL_BANK_WIDTH_BITS / 8` (line 176).
  - `compute::IS_CHECKERBOARD` read for checkerboard-aware routing (line 133).
  The AIE2 references in this file are all documentation (`// AIE2 (IsCheckerBoard=0)`) or already-archspec references.
- **Prescribed migration verb.** `leave-alone`. This file is exemplary -- it is the model for how the rest of the timing area should be migrated.
- **Estimated LOC impact.** 0.

### `timing/hazards.rs`

- **Size + responsibility.** 724 lines; register hazard detection (RAW, WAW), `HazardDetector` struct, per-register `ready_cycle` tracking, `HazardStats`.
- **AIE2 hardcode count.** 2 (doc comment `// AIE2 has an 8-stage maximum pipeline` and import of `xdna_archspec::aie2::isa::SemanticOp`). No load-bearing arch-specific constants -- the hazard detection algorithm (advance cycle, check ready_cycle, insert stall) is arch-generic; the latencies feeding it come from `LatencyTable`.
- **Prescribed migration verb.** `leave-alone`.
- **Estimated LOC impact.** 0.

### `timing/slots.rs`

- **Size + responsibility.** 298 lines; VLIW structural hazard detection (`check_bundle_conflicts`), `ExecutionResource` enum mapping slot types to functional units.
- **AIE2 hardcode count.** 3. Key: `// AIE2 VLIW bundles can contain up to 7 slots that execute in parallel` (line 7). The `ExecutionResource` variants (LoadUnit, StoreUnit, VectorUnit, etc.) mirror the AIE2 VLIW slot structure. These are architectural constants that would differ for AIE1 (fewer slots, different structural hazard rules).
- **Divergence risks.** The structural hazard rules (`// two loads and one store can run in parallel`) are AIE2-specific. AIE1 has fewer slots and different conflict rules. However, these rules are encoded in a match expression driven by `SemanticOp`, not by hardcoded slot counts -- so they are flexible in the sense that removing a SemanticOp case would disable the conflicting unit. For Subsystem 7, this is `leave-alone`.
- **Prescribed migration verb.** `leave-alone`.
- **Estimated LOC impact.** 0.

### `timing/arbitration.rs`, `barrier.rs`, `deadlock.rs`, `sync.rs`

- **Sizes + responsibilities.** Arbitration (356 lines): memory-tile round-robin contention model (scaffolded, not wired). Barrier (778 lines): multi-core barrier timing (scaffolded, not wired). Deadlock (621 lines): lock circular-wait detection with depth limit. Sync (361 lines): lock timing state tracking.
- **AIE2 hardcode counts.** Arbitration: 1 doc reference. Barrier: 2 doc references. Deadlock: 4, including `max_cycle_length: 16` (line 393, a heuristic for AIE2 arrays). Sync: 3 (imports + doc).
- **Divergence risks.** All are arch-generic algorithms consuming arch-specific inputs. The deadlock cycle-length heuristic (16) is an operational parameter, not a hardware constant. None of these files have load-bearing AIE2 constants in the algorithm paths.
- **Prescribed migration verb.** `leave-alone` for all four.
- **Estimated LOC impact.** 0 for all four.

### `timing/mod.rs`

- **Size + responsibility.** 52 lines; re-exports only.
- **AIE2 hardcode count.** 2 (doc comment + `LatencyTable::aie2()` usage note). No constants.
- **Prescribed migration verb.** `leave-alone`.

### `execute/cycle_accurate.rs` latency tables

Already audited in Section 1. The only timing-relevant finding is `LatencyTable::aie2()` at construction (line 87), which should become `LatencyTable::for_arch(arch_handle::processor_model())`. Currently this is a naming issue rather than a behavioral divergence since `aie2()` calls the archspec LLVM FFI internally. The `BRANCH_DELAY_SLOTS = 5` in archspec (`processor::BRANCH_DELAY_SLOTS`) is not yet consumed in `cycle_accurate.rs` -- the 5-slot behavior is implicit in `ExecutionContext::tick_delay_slots`.

**Timing area summary.** The spec's "data migration only, no trait seam" stance is confirmed. `timing/memory.rs` is the exemplary file -- already fully archspec-driven. `timing/latency.rs` is the primary migration target: its 8 raw literal constants (`LATENCY_MEMORY`, `LATENCY_SCALAR_MUL`, etc.) duplicate values already in `xdna_archspec::aie2::timing`. `validated_aie2()` already validates the relationship; the next step is to replace the literals with the archspec reads, making the validation unnecessary. `hazards.rs`, `slots.rs`, `arbitration.rs`, `barrier.rs`, `deadlock.rs`, `sync.rs`, and `mod.rs` are all `leave-alone`.

---

## Closing summary

(Filled in by Task 1 Step 9.)

### Tentative trait method list

### Data migration list

### AIE1 projection

---

## Completion

(Filled in at the end of Subsystem 7, in the Part B final task.)
