# Device-Family Refactor -- Design

**Date**: 2026-04-16
**Status**: Approved, pending implementation plan

## Problem

The xdna-emu codebase has reached ~100% ISA accuracy for AIE2 (NPU1 /
Phoenix), with passing bridge tests against real hardware. The structure,
however, reflects its evolution rather than its current understanding:

- Three parallel architecture abstractions coexist: `crates/xdna-archspec/`,
  `src/archspec/mod.rs`, and `src/device/arch_config.rs`. Only 9 files in
  `src/` import from the archspec crate; most code hardcodes AIE2 constants.
- The archspec crate itself is substantial and well-designed
  (`Confirmed<T>` cross-validation, multi-arch `ArchModel`, build-time
  codegen wiring) but is under-plumbed.
- Several subsystems (`interpreter/execute/*`, `device/dma/*`,
  `device/stream_switch/*`) assume AIE2-specific VLIW width, BD layout,
  and port topology in their signatures, making a second architecture
  impractical to add without a fork-and-patch.
- File sizes reflect accumulation: `interpreter/execute/vmac_routing.rs`
  is 239KB, `interpreter/execute/memory/mod.rs` is 124KB as a `mod.rs`,
  `interpreter/execute/vector_arith.rs` is 102KB. Several others are
  50-100KB.

The project's next architectural goal is supporting additional AIE
families -- AIE2P (Strix), then AIE1/Versal -- and eventually layering in
a logic fuzzer and Peano-in-the-loop compilation. Neither is practical
against the current shape.

## Goals

1. **One arch abstraction.** Collapse the three parallel arch sources into
   `xdna-archspec` and plumb it authoritatively through `src/device/`,
   `src/interpreter/`, `src/parser/`.
2. **Seams, not second implementations.** Lift AIE2-specific behavior
   behind traits keyed by architecture (`IsaDecoder`, `DmaModel`,
   `LockModel`, `StreamSwitchModel`, `BinaryLoader`, etc.) with AIE2 as
   the sole implementation at refactor end. Adding AIE2P or AIE1 becomes
   "implement the trait."
3. **Code hygiene.** Split oversized files, rationalize module
   boundaries, tighten naming, and consider crate-level reorganization
   after the arch seams are in place.
4. **Preserve correctness.** `cargo test --lib` stays green every commit.
   Bridge tests stay green on AIE2 after each subsystem.

## Non-goals

- **No second-arch implementation during the refactor.** AIE2P and
  AIE1/Versal are follow-on work. Designing the seams against both
  (by reading aie-rt and mlir-aie for reference) is in scope; writing
  the second implementation is not.
- **No performance optimization.** Correctness before performance is the
  standing rule; this refactor is structural, not perf.
- **No over-speccing Versal-as-a-platform.** PS-PL integration, Versal
  PDI container variants, and FPGA-fabric coupling are deferred until
  Versal is actually being ported. YAGNI applies.
- **No master merges.** All work happens on `dev` as reviewable commits,
  with per-subsystem tags for bisect navigation. `master` advances only
  at major milestones at the user's discretion.

## Design

### Phase 1 -- Architectural seams

Three subphases; 1b and 1c interleave per subsystem rather than running
as separate passes.

**Phase 1a -- Consolidate.** Collapse `src/archspec/mod.rs` and
`src/device/arch_config.rs` into `xdna-archspec`. Runtime code imports
only from the crate. Parallel shims are deleted (not archived -- they
duplicate the crate, git history preserves them). Target: single
`ArchModel` source-of-truth, no parallel abstractions.

**Phase 1b + 1c -- Per-subsystem plumb + seam (interleaved).** For each
subsystem:

1. Audit AIE2 assumptions. Separate data (sizes, offsets, counts) from
   behavior (decode rules, stepping semantics, routing legality).
2. Route data queries through archspec. Delete local hardcoded constants.
3. If behavior varies meaningfully across archs (validated by reading
   aie-rt's AIE1 paths and mlir-aie's AIE1 device model), lift behind
   a trait. AIE2 is the sole implementation.
4. Verify: `cargo test --lib` green; bridge test no-hw smoke green; tag
   the subsystem commit for bisect.

**Subsystem order** (lowest coupling first, compounding wins):

1. **Registers & memory map** -- foundational. Mostly plumbing.
2. **Tile topology** -- array dims, tile roles, per-tile sizes. Data.
3. **DMA engine & BD format** -- first behavioral seam. `DmaModel`.
4. **Locks** -- small exercise. `LockModel`.
5. **Stream switch** -- topology (data) + routing legality (behavior).
   `StreamSwitchModel`.
6. **ISA decode** -- bundle/slot layout, decoder tables. `IsaDecoder`.
7. **ISA execute** -- semantic ops, intrinsic handlers. `IsaExecutor`.
   Biggest seam; also the home of the largest files in the codebase.
8. **Parser (XCLBIN / PDI / ELF)** -- container format variance.
   `BinaryLoader`.

### Trait seam design principles

- **Traits decode / step / rule-check; they do not hold mutable emulator
  state.** State lives in plain structs. The trait is the interpreter of
  that state for a given architecture.
- **Monomorphize where hot.** Use generics or an `Arch` type parameter
  on hot paths (interpreter inner loops). Reach for `dyn ArchModel` only
  at the top-level "what am I emulating" boundary.
- **Coarse first.** Each trait captures the minimal delta between archs.
  If we can't articulate what AIE1's implementation would look like in
  ~100 words, the trait is probably wrong-shaped -- either over-spec'd
  or under-spec'd.
- **Design-safety note per seam.** Before each trait commits, a short
  design note answers: "If we had to implement this for AIE1, what
  would the AIE1 version look like?" Answers guide scope:
  - "Basically identical" -> trait is over-engineered; stay data-driven.
  - "Completely different" -> trait is right; proceed.
  - "Mostly the same with N carve-outs" -> the carve-outs belong in the
    trait, the rest is shared code.

Sketched (non-final) trait shapes:

```rust
trait IsaDecoder {
    fn bundle_width_bytes(&self) -> usize;
    fn decode_bundle(&self, bytes: &[u8]) -> DecodedBundle;
    fn slot_layout(&self) -> &'static SlotLayout;
}

trait DmaModel {
    fn parse_bd(&self, raw: &[u32]) -> BufferDescriptor;
    fn step_channel(&self, ch: &mut DmaChannel, ctx: &mut DmaCtx);
    fn poll_done(&self, ch: &DmaChannel) -> bool;
}

trait StreamSwitchModel {
    fn port_count(&self, tile: TileKind, bundle: Bundle) -> u8;
    fn is_legal_route(&self, src: PortId, dst: PortId) -> bool;
}
```

### Phase 2 -- Hygiene

Driven by what Phase 1 exposes. Visible targets today:

- **Large file splits.** `interpreter/execute/vmac_routing.rs` (239KB),
  `interpreter/execute/memory/mod.rs` (124KB), `vector_arith.rs` (102KB),
  `fuzzer/trace_sweep.rs` (98KB), `trace/compare.rs` (94KB),
  `interpreter/decode/decoder.rs` (87KB). Split by operation family or
  concern, not arbitrary line count.
- **Module boundaries.** Top-level `src/` has 20 directories/files
  including one-offs (`config.rs`, `build_progress.rs`) that likely
  belong inside existing subsystems or a new `cli/`. Collapse where it
  makes sense.
- **Naming audit.** Catch other duplicated-intent modules, obsolete
  names, inconsistent conventions.
- **Possible crate lifts.** If Phase 1 leaves `interpreter/` or
  `parser/` with clean seams, promote them to workspace crates. Better
  compile times, clearer dependency graph, more portfolio-friendly.

### Testing strategy

- `cargo test --lib` green at every commit. Non-negotiable.
- Bridge tests (`./scripts/emu-bridge-test.sh --no-hw`) green after each
  subsystem commits. Full HW bridge run at end of each subsystem before
  tagging.
- Each trait gets a small "contract" test: instantiate the AIE2 impl,
  exercise the contract, catch trait-signature drift.
- Refactor should not grow the existing test suite; bridge + ISA suites
  already define correctness. Re-home, don't regrow.
- `Confirmed<T>` cross-validation in archspec stays and extends wherever
  1b plumbing reveals new cross-source invariants.

## Deliverables

1. **Phase 1a** -- one commit: archspec consolidated, parallel shims
   deleted, all runtime code imports from the crate. Tag:
   `phase1a-consolidate`.
2. **Phase 1b/1c** -- one tagged commit set per subsystem (8 total),
   in the order listed above. Each leaves AIE2 behavior unchanged and
   adds the trait seam.
3. **Phase 2** -- hygiene commits grouped by intent (file splits,
   module reshape, naming, crate lifts). Structure decided after Phase 1
   exposes the real shape.
4. **Design notes per seam** -- short `docs/arch/<subsystem>.md`
   entries capturing the trait intent and the "what would AIE1 look
   like" sanity check.

## Branching and commit cadence

- All work on `dev`. No `master` merges until the user chooses to
  advance it.
- Per-subsystem tag (e.g., `phase1-subsys-dma`) at each milestone for
  bisect navigation and reviewability.
- Commits within a subsystem are small and reviewable; subsystems do
  not cross-contaminate.

## Risks and mitigations

- **Risk:** trait designed against one implementation becomes
  "AIE2-shaped with AIE1 holes poked through."
  **Mitigation:** per-seam design note cross-checked against aie-rt /
  mlir-aie AIE1 paths before commit.
- **Risk:** Phase 2 reveals Phase 1 seams were wrong-shaped.
  **Mitigation:** we have the correctness definition (test suite); bold
  structural moves in Phase 2 are safe. Re-shape without fear.
- **Risk:** refactor momentum stalls mid-subsystem.
  **Mitigation:** per-subsystem scope is bounded; each is independently
  shippable. Stopping between subsystems is always a valid state.
- **Risk:** trait indirection costs performance on hot paths.
  **Mitigation:** monomorphize via generics or `Arch` type parameter on
  hot paths; reserve `dyn` for top-level boundaries only.

## Success criteria

- One authoritative arch abstraction (`xdna-archspec`), consumed
  throughout `src/`.
- Each of the 8 subsystems plumbs data through archspec and, where
  behavior varies, defines a trait with AIE2 as the implementation.
- `cargo test --lib` and bridge tests green at every tagged commit.
- No file over ~40KB after Phase 2 (soft target; some may remain if
  they're genuinely cohesive).
- Adding AIE2P is "implement the traits"; adding Versal is "implement
  the traits + extend for platform differences." Neither requires
  re-plumbing.

## Out of scope

- Any AIE2P or AIE1/Versal implementation.
- Logic fuzzer, differential testing, Peano-in-the-loop. These are
  subsequent projects, each with their own brainstorm + spec.
- Performance tuning.
- GUI / visual debugger work.
