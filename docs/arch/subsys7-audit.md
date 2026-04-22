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

(Filled in by Task 1 Step 5.)

## 4. Vector ALU

Files: `vector_arith.rs`, `vector_compare.rs`, `vector_misc.rs`,
`vector_pack.rs`, `vector_ups.rs`, `vector_srs.rs`, `vector_helpers.rs`,
`vector_semantic.rs`, `vector_permute.rs`, `vector_float.rs`,
`vector_config.rs`, `vector_convert.rs`, `vector_validate.rs`.

(Filled in by Task 1 Step 6.)

## 5. VMAC / matmul

Files: `vmac_routing.rs` (deep dive), `vmac_hw.rs`, `vector_matmul/`.

(Filled in by Task 1 Step 7.)

## 6. Timing

Files: `interpreter/timing/{arbitration, barrier, deadlock, hazards,
latency, memory, mod, slots, sync}.rs`, plus `execute/cycle_accurate.rs`
latency tables.

(Filled in by Task 1 Step 8.)

---

## Closing summary

(Filled in by Task 1 Step 9.)

### Tentative trait method list

### Data migration list

### AIE1 projection

---

## Completion

(Filled in at the end of Subsystem 7, in the Part B final task.)
