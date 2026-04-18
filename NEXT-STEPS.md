# Next Steps -- Device-Family Refactor

Recovery document for picking up this refactor in a future session. Read
this first, then dive into the authoritative artifacts below.

**Last updated:** 2026-04-18 (Phase 1b Subsystem 6 landed; Subsystem 2 up next)
**Current branch:** `dev` (no master merges until the refactor is done)
**Latest tag:** `phase1-subsys-isa-decode` (Part B commits land on top of `phase1-subsys-isa-decode-partA`)

---

## Where We Are

Phase 1a landed at `phase1a-consolidate` (`6534e28`). Phase 1b Subsystem 1
(Registers & Memory Map) landed at `phase1-subsys-regs-mem` (`8c0f217`)
**with reduced scope**: Tasks 9 (gen_tablegen + build_helpers) and Task 10
(decoder_ffi) were deferred to Subsystem 6 (ISA Decode) because both are
coupled to xdna-emu's interpreter types (`Operand`, `MappedOperand`,
`RegisterMap`, the tablegen `types.rs` / `resolver/` modules). Moving the
ISA infrastructure into archspec without untangling those types would
leave dangling cross-crate boundaries that the type system cannot enforce.
See `docs/arch/subsys1-audit.md` (`## Task 9 Deferral`, `## Task 10 Deferral`,
`## Task 11 Reduced Scope`) for the full reasoning.

**Consequence:** the original Phase 1b/1c order no longer makes sense.
Subsystem 6 (ISA Decode) ran second, out of numerical order, because
Subsystem 1's deferred Tasks 9/10 were blocking progress on the remaining
consumers. Subsystem 6 is now **done** at `phase1-subsys-isa-decode`
(Part A + Part B both landed): the `crate::tablegen::*` / `crate::arch::*`
consumers all migrated, the `mod arch` forwarder in `src/lib.rs`
dissolved, the `extract_aiert` / `compile_llvm_decoder_ffi` duplication
in xdna-emu/build.rs moved to archspec's build.rs.  The remaining order
(2 -> 3 -> 4 -> 5 -> 7 -> 8) resumes as originally planned.

Phase 1a (from the earlier pass): the three parallel arch abstractions
(`xdna-archspec` crate + `src/archspec/mod.rs` + `src/device/arch_config.rs`)
collapsed into one -- the `xdna-archspec` workspace crate is the single
source-of-truth, and every consumer in `src/` imports directly from
`xdna_archspec::types` or `xdna_archspec::runtime`. The port-layout
methods in `src/device/port_layout.rs` that were once a runtime-side
remnant have since been consolidated through the Subsystem 1 work into
`xdna_archspec::aie2::{port_type, stream_switch}`.

**Verification that held (at `phase1-subsys-isa-decode`):**
- `cargo test --lib`: 2712 passed, 0 failed, 5 ignored. Down from 2730
  at the Part A tag; 21 `#[cfg(test)]` tests inside the deleted
  `src/tablegen/mod.rs` + `decoder_ffi.rs` migrated to their rightful
  homes in archspec (18 tests) and `interpreter::decode::register_map`
  (3 tests). Net live test count preserved.
- `cargo test -p xdna-archspec --lib`: 220 pass / 1 pre-existing failure
  (`device_model::test_full_parse_all_devices`, unrelated to refactor).
  Up from 202 at Part A (the 18 rescued archspec tests).
- `cargo build --release`: clean.  `build.rs` shrunk from 201 -> 97
  lines; xdna-emu has no build-dependencies anymore (all codegen in
  `crates/xdna-archspec/build.rs`).
- Bridge `--no-hw -v add_one_cpp_aiecc`: Chess + Peano PASS.
- Full HW bridge + ISA suite gate at the tag: see
  `/tmp/claude-1000/subsys6-partB-{bridge,isa}.log`.

**Build-environment caveat (carried from Part A, still applies):** every
fresh `cargo build` needs
`export PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH` so tblgen's
build-script finds llvm-config 21.x (not mlir-aie's 23.x). Hygiene fix
deferred: archspec's build.rs should set `LLVM_SYS_210_PREFIX`
explicitly.

**Bridge harness fixes (incidental, applied during Subsystem 6 Part B):**
- `scripts/emu-bridge-test.sh` now substitutes lit's `%s` macro, so
  `FileCheck %s` in RUN lines works (was silently broken for
  `bd_chain_repeat_on_memtile`).
- Added `XFAIL:` parsing that the shell bridge harness lacked from the
  start (the Rust-native `src/testing/` harness had it; the XRT bridge
  didn't). `XFAIL: *` and `XFAIL: peano|chess` feature lists now
  produce `XFAIL` / `XPASS` results instead of `FAIL`.

**Pre-existing EMU regression flagged during Subsystem 6 gate (not our
fault):**
- `bd_chain_repeat_on_memtile` EMU side gets stuck in a DMA
  `check_acquire_granted granted=false` polling loop and never emits
  `PASS!`. Traced to Subsystem 1 (or earlier); passed on 20260414,
  failed by 20260416 at `phase1-subsys-regs-mem`. Independent
  workstream; open as an investigation after Subsystem 2 lands.

---

## Authoritative Artifacts

Read these for the full picture:

1. **Spec:** [`docs/superpowers/specs/2026-04-16-device-family-refactor-design.md`](docs/superpowers/specs/2026-04-16-device-family-refactor-design.md)
   The design doc. Explains the problem, the three-phase structure (Phase
   1a = consolidate, Phase 1b/1c = per-subsystem plumb+seam, Phase 2 =
   hygiene), and the trait-seam design principles ("coarse first",
   "monomorphize where hot", "what would AIE1 look like").

2. **Plan:** [`docs/superpowers/plans/2026-04-16-device-family-refactor-plan.md`](docs/superpowers/plans/2026-04-16-device-family-refactor-plan.md)
   The implementation plan. Phase 1a in full bite-sized detail; Phase
   1b/1c subsystems sketched at section level with the note that each
   gets its own detailed plan when we start it.

3. **Phase 1a audit:** [`docs/arch/phase1a-audit.md`](docs/arch/phase1a-audit.md)
   The concrete data that drove Phase 1a (ArchConfig trait surface,
   consumers, type boundary resolution, port-data dependency). Ends with
   a `## Completion` section listing all Phase 1a commits and flagging
   follow-ups.

4. **Commit series:** `git log --oneline 67ec2c5..phase1a-consolidate`
   13 commits spanning the 6 Phase 1a tasks.

---

## What Phase 1a Established (Decisions That Affect Later Subsystems)

These aren't random implementation details; they're the decisions
subsequent subsystem passes inherit.

### Architecture contract

- **`xdna-archspec` is the single source-of-truth.** It owns the
  `ArchConfig` trait, `ModelConfig`, `default_arch()`, `ARCHSPEC_MODELS`,
  all the register DB, all the `Confirmed<T>` cross-validation, and the
  device model extraction.
- **`xdna-archspec` uses `TileKind`, not `TileType`.** Runtime code still
  uses `TileType` (runtime-native, 3 variants); the `From` bridge in
  `src/device/tile/core_state.rs` (commit `43fc807`) converts both ways.
  `ShimNoc` and `ShimPl` both map to `TileType::Shim` (lossy but the
  emulator never produces `ShimPl`). Deep rename of `MemTile` -> `Mem`
  and merging of the two `Shim` variants is deferred to Subsystem 2
  (Tile Topology).
- **Port-layout methods stay runtime-side.** The six stream-switch
  port-layout methods (`master_ports`, `slave_ports`,
  `north_master_range`, `south_master_range`, `north_slave_range`,
  `south_slave_range`) live on the runtime-side `PortLayout` extension
  trait in `src/device/port_layout.rs`, not on `ArchConfig` in the
  crate. Their data comes from `crate::arch::*` (build.rs-generated from
  AM025 JSON); moving that generation into the workspace crate is a
  bigger build-system change, deferred. When Subsystem 5 (Stream Switch)
  runs, revisit this: if AIE2P diverges on port layout, it may be the
  right moment to move port-data generation into the crate.
- **`npu1()` loads from the archspec JSON at runtime.** Before Phase 1a,
  `npu1()` used build.rs-generated constants and `npu2()`/`xcve2802()`
  loaded from JSON. Phase 1a unified all three through JSON, eliminating
  the special case. Cost: one-time runtime parse via `LazyLock`. Benefit:
  no drift risk between the two paths.

### Method renames

- `ArchConfig::tile_type()` -> `ArchConfig::tile_kind()` (return type
  changed from `TileType` to `TileKind`).

### Follow-ups flagged (addressable in later subsystems or Phase 2 hygiene)

- **Deep rename `TileType::MemTile` -> `Mem`, merge `Shim`/`ShimNoc`:**
  Subsystem 2 (Tile Topology).
- **DMA channel fallback `(2, 2)` in `from_arch_model()` is silent:**
  replace with `.expect()` or `log::warn!()`. Subsystem 3 (DMA).
- **Shim DMA config not tested; NPU2 tile params not tested:** Phase 2
  hygiene or relevant subsystem pass.
- **Inner-scope `TileKind` imports repeated in `model.rs` test fns:**
  consolidate. Phase 2 hygiene.
- **Import ordering in `src/device/array/mod.rs` diverges from
  convention** (workspace crate first, std last): Phase 2 hygiene.
- **`columns: topo.columns + 1` has no overflow guard:** add a
  `checked_add().expect(...)`. Phase 2 hygiene.
- **`src/archspec/` stale references in archived plan/spec docs:** these
  are intentional history; do not touch.

---

## Phase 1b/1c -- Per-Subsystem Pass Order

Each subsystem below is a self-contained deliverable: audit -> plumb data
through archspec -> add trait seam if behavior varies -> verify.
**Each gets its own `brainstorming` -> `writing-plans` -> execution cycle**
when we start it -- don't try to pre-write all 8 subsystem plans.

**Per-subsystem tag at end:** `phase1-subsys-<name>`.

| # | Subsystem | Tag | Status | Description |
|---|-----------|-----|--------|-------------|
| 1 | Registers & Memory Map | `phase1-subsys-regs-mem` | **Done (reduced scope)** | Plumb hardcoded register offsets, memory sizes, per-tile-type counts through `ArchModel`. Mostly data; no new trait. Tasks 9+10 deferred to Subsystem 6; see audit. |
| 6 | ISA Decode | `phase1-subsys-isa-decode` | **Done** | Decoder tables, bytecode walker, MCDisassembler FFI and the resolver/types modules moved to `xdna_archspec::aie2::isa`; `MappedOperand` / `RegisterMap` / `AccumWidth` (interpreter-coupled) moved to `xdna_emu::interpreter::decode::register_map`. Subsystem 1's deferred Tasks 9/10 absorbed. No trait seam -- see `docs/arch/isa-decode.md`. |
| 2 | Tile Topology | `phase1-subsys-tile-topo` | **Up next** | Replace `row == 0` / `row >= N` checks with `ArchModel`-backed classification. **Includes the deferred `TileType` -> `TileKind` deep rename**: merge `Shim`/`ShimNoc`, rename `MemTile` -> `Mem`. |
| 3 | DMA Engine & BD Format | `phase1-subsys-dma` | Pending | First behavioral seam. Audit AIE2 vs AIE1 BD layout via aie-rt source. Lift BD parse/encode + channel stepping behind `DmaModel` trait. |
| 4 | Locks | `phase1-subsys-locks` | Pending | Small seam exercise. `LockModel` trait if acquire/release/value semantics genuinely differ (likely around lock value width). |
| 5 | Stream Switch | `phase1-subsys-stream-switch` | Pending | Topology (data, already via archspec) + routing legality (behavior, trait: `StreamSwitchModel`). |
| 7 | ISA Execute | `phase1-subsys-isa-execute` | Pending | Semantic ops, intrinsic handlers. Biggest; largest files live here (`vmac_routing.rs` 239KB, `memory/mod.rs` 124KB). `IsaExecutor` trait. |
| 8 | Parser (XCLBIN / PDI / ELF) | `phase1-subsys-parser` | Container format variance. `BinaryLoader` trait. |

**End state after Phase 1:** adding AIE2P is "implement the traits for a
second arch"; adding AIE1/Versal is "implement the traits + extend for
platform differences." Neither requires re-plumbing.

---

## How to Pick Up Subsystem 2 (Tile Topology)

This is the concrete next action. Start here in a fresh session.

1. **Read the key artifacts:**
   - `docs/superpowers/specs/2026-04-16-device-family-refactor-design.md` (parent)
   - `docs/superpowers/plans/2026-04-16-device-family-refactor-plan.md` (parent plan)
   - `docs/arch/phase1a-audit.md` -- `## Follow-ups flagged` lists the
     deferred `TileType` -> `TileKind` rename and the `Shim`/`ShimNoc`
     merge, both of which belong in Subsystem 2.
   - `docs/arch/subsys1-audit.md` -- any tile-topology hardcodes that
     Subsystem 1 left behind.
   - `docs/arch/isa-decode.md` -- the Subsystem 6 seam design note
     (for the const-first pattern template).

2. **Verify the current state hasn't drifted:**
   ```bash
   git log --oneline phase1-subsys-isa-decode..HEAD
   ```
   If nothing has landed since the tag, you're picking up exactly where
   Subsystem 6 left off.

   ```bash
   PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
   PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
   ```
   Expect xdna-emu `2712 passed; 0 failed; 5 ignored` and archspec
   `220 passed; 1 failed; 2 ignored` (the one failure is the
   pre-existing `device_model::test_full_parse_all_devices`).

3. **Invoke brainstorming** to shape Subsystem 2's spec:
   ```
   /brainstorming
   ```
   Topic: "Phase 1b Subsystem 2: Tile Topology, including the deferred
   `TileType` -> `TileKind` deep rename."

   **Shape the spec around these questions:**
   - Which tile-classification predicates in `src/device/` and
     `src/interpreter/` are still doing `row == 0` or `row >= N`
     instead of asking `ArchModel`?
   - Does `TileKind` in archspec need new variants for AIE1 / AIE2P
     cases we don't currently distinguish, or is the `Shim/ShimNoc/Mem/
     Compute` set already sufficient?
   - Is the `From<TileType>` / `Into<TileKind>` bridge at
     `src/device/tile/core_state.rs` ready to dissolve, or does the
     runtime path still need the `TileType` alias?
   - Is there a behavioral seam here (different row-indexing per arch
     family, different shim-mux placement) that wants a `TileTopology`
     trait, or is this pure plumbing?

4. **Invoke writing-plans** to produce a plan at
   `docs/superpowers/plans/YYYY-MM-DD-subsys2-tile-topology.md`.

5. **Invoke subagent-driven-development** to execute.  Template:
   implementer subagent per task; spec-compliance + code-quality review
   gates; same-session orchestration.

6. **At end of Subsystem 2:** tag `phase1-subsys-tile-topo`, append a
   completion section to its audit, update this `NEXT-STEPS.md` to
   move Subsystem 3 (DMA Engine) to "up next."

---

## Useful Commands

```bash
# See Phase 1a commits
git log --oneline 67ec2c5..phase1a-consolidate

# Run library tests (Global Invariant; green at every commit)
cargo test --lib

# Run the archspec crate tests
cargo test -p xdna-archspec --lib

# Fast bridge smoke (catches 90% of regressions, ~30s)
./scripts/emu-bridge-test.sh --no-hw -v add_one

# Full bridge run (15-30 min; requires NPU to be idle)
./scripts/emu-bridge-test.sh 2>&1 | tee /tmp/claude-1000/bridge-phase1-subsys<N>.log

# What's currently in xdna-archspec
ls crates/xdna-archspec/src/

# What's currently in src/device (after Phase 1a)
ls src/device/
```

---

## Ground Rules (Carried Over from the Spec)

- **No master merges during the refactor.** Everything lands on `dev`.
  Master advances only when the user chooses.
- **`cargo test --lib` green at every commit.** Non-negotiable.
- **Bridge test smoke green at every subsystem tag.** Full HW run before
  tag.
- **Per-subsystem tag:** `phase1-subsys-<name>`. Bisect-friendly.
- **One authoritative source per concept.** Don't re-introduce parallel
  abstractions.
- **Traits decode/step/check; they do not hold mutable state.** State
  lives in plain structs.
- **Coarse first.** For each trait seam, if we can't articulate what
  AIE1's version would look like in ~100 words, the trait is
  wrong-shaped.
- **"What would AIE1 look like?" design note per seam.** Written into
  `docs/arch/<subsystem>.md` before the trait commits.
- **No second-arch implementation during the refactor.** Phase 1 is
  seams only; filling AIE2P or AIE1 is follow-on work.

---

## Pre-existing Issues (Not Phase-1a Regressions; Don't Panic)

- `cargo test -p xdna-archspec --lib`: `test_full_parse_all_devices`
  fails with device count 13 vs expected 12. Unrelated to this refactor;
  present before Phase 1a started.
- Peano bridge EMU timeouts on `dma_task_large_linear` and
  `objectfifo_repeat/init_values_repeat`. Present in yesterday's
  pre-Phase-1a run; not caused by this refactor.
- `examples/run_add_test.rs` has a stale API compile error. Unrelated to
  this refactor; pre-existing.
- Generated file warnings (unused constants in `gen_aiert_*.rs`).
  Pre-existing.

---

## Contact Points

If you're a future session with no memory of this work, **don't spawn a
bunch of Explore agents to re-derive context** -- the artifacts above
have already done that work. Start from the spec, then the audit's
`## Completion` section, then `git log --oneline phase1a-consolidate`
to see what shipped. Only then dive into code.
