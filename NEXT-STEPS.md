# Next Steps -- Device-Family Refactor

Recovery document for picking up this refactor in a future session. Read
this first, then dive into the authoritative artifacts below.

**Last updated:** 2026-04-22 (Phase 1b Subsystem 7 landed; Subsystem 8 up next)
**Current branch:** `dev` (no master merges until the refactor is done)
**Latest tag:** `phase1-subsys-isa-execute` (Subsystem 7 completion)

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
methods that were once a runtime-side remnant in
`src/device/port_layout.rs` have since been consolidated through the
Subsystem 1 work into `xdna_archspec::aie2::{port_type, stream_switch}`;
the `PortLayout` extension trait itself was deleted in Subsystem 5,
and the data is now reached via `arch_handle::stream_switch_topology()`
/ `xdna_archspec::stream_switch::StreamSwitchTopology`.

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
- Bridge `--no-hw` spot-check at HEAD with a fresh
  `cargo build -p xdna-emu-ffi` (2026-04-18): every targeted EMU
  test passes except the pre-existing `bd_chain_repeat_on_memtile`
  deadlock documented below.  (An older `subsys6-partB-bridge.log`
  exists but is unreliable -- it was captured with a stale `.so`
  and reports five phantom EMU timeouts that do not reproduce once
  the FFI is rebuilt.  Always rebuild the FFI immediately before
  capturing a gate log; see "Useful Commands".)

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

**Known pre-existing failure (independent of Phase 1 -- do not block
on it):**
- `bd_chain_repeat_on_memtile` EMU side spins in a DMA
  `check_acquire_granted granted=false` polling loop and never emits
  `PASS!` on either Chess or Peano.  Verified to reproduce at
  `53e9337` and `0af8548` (both 2026-04-16, pre-refactor).  The
  20260416 bridge result files show PASS for this test, but the
  94k-line bridge log clearly shows the same deadlock in its tail --
  the "PASS" was a harness anomaly from the pre-`ba86b37` script not
  substituting lit's `%s` macro, which made `FileCheck %s` error in a
  way the bridge script misread as pass.  `ba86b37` (substitute `%s`
  and honor `XFAIL:`) correctly surfaces this long-latent deadlock.
  Independent workstream; investigate after the Phase 1 refactor
  if worthwhile, not during.

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
  `south_slave_range`) lived on the runtime-side `PortLayout` extension
  trait in `src/device/port_layout.rs` (deleted in Subsystem 5), not on
  `ArchConfig` in the crate. Their data source is now
  `xdna_archspec::aie2::{port_type, stream_switch}` (moved from
  `crate::arch::*` during Subsystem 1/6), and the aggregate is reached
  via `xdna_archspec::stream_switch::StreamSwitchTopology` +
  `ArchConfig::stream_switch_model().topology()`. When AIE2P diverges
  on port layout, add an AIE2P `StreamSwitchModel` impl alongside
  `Aie2StreamSwitchModel`.
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
| 2 | Tile Topology | `phase1-subsys-tile-topo` | **Done** | Trait seam (TileTopology) + AIE2 impl. TileType->TileKind deep rename. 2 bare row==0 hardcodes + 4 row>0 neighbor guards routed through archspec constants. See docs/arch/tile-topology.md. |
| 3 | DMA Engine & BD Format | `phase1-subsys-dma` | **Done** | First behavioral seam. `DmaModel` trait (9 methods: 7 feature flags + `max_tensor_dims` + `timing_config`) + `Aie2DmaModel` impl. `DeviceRegLayout` family migrated to archspec; xdna-emu `Deref` wrapper retains lock-value-width fields pending Subsystem 4. 5 AIE2-only call sites gated on `supports_*()` feature flags. `(2,2)` silent fallback fixed. 7 hygiene items. Bonus: `test_full_parse_all_devices` archspec failure fixed, giving a clean baseline. See `docs/arch/dma-model.md`. |
| 4 | Locks | `phase1-subsys-locks` | **Done** | Small seam: 3-method LockModel trait (supports_acquire_eq, supports_dynamic_value_ops, value_layout) + LockValueLayout carrier + Aie2LockModel. Migrated `sign_extend_lock_value` + lock_value_* fields from xdna-emu's DeviceRegLayout wrapper to archspec; wrapper collapsed. See docs/arch/lock-model.md. |
| 5 | Stream Switch | `phase1-subsys-stream-switch` | **Done** | Two-method `StreamSwitchModel` trait (`supports_deterministic_merge` + `topology`) + `StreamSwitchTopology` carrier (3 `TileStreamPorts` sub-structs with `for_tile(TileKind)` accessor) + `Aie2StreamSwitchModel`. Dead-code `PortLayout` extension trait (231 LOC) deleted; 3 tests migrated to archspec. 6 tile-construction sites in `stream_switch/mod.rs` migrated through `arch_handle::stream_switch_topology()`. Direct archspec-constant consumers (`routing.rs`, `stream_io.rs`, `state/`) stay on direct access as AIE1-landing follow-ups. See `docs/arch/stream-switch-model.md`. |
| 7 | ISA Execute | `phase1-subsys-isa-execute` | **Done** | Audit concluded Approach A (zero trait methods warranted): all divergence is data-expressible. `IsaExecutor` trait ships empty as a stable anchor. 11 data migrations landed across 5 tasks -- `vmac_routing.rs` (234K) + `vector_permute.rs` tables + `RoundingMode` dedup + matmul/UPS/cascade data + 5 accessor bundles. `has_cascade_link: bool` feature flag added to `ProcessorModel`. AIE1/AIE2P ports now "populate archspec; no execute/ edits." See `docs/arch/isa-execute-model.md` + `docs/arch/subsys7-audit.md`. |
| 8 | Parser (XCLBIN / PDI / ELF) | `phase1-subsys-parser` | **Up next** | Container format variance. `BinaryLoader` trait. |

**End state after Phase 1:** adding AIE2P is "implement the traits for a
second arch"; adding AIE1/Versal is "implement the traits + extend for
platform differences." Neither requires re-plumbing.

---

## How to Pick Up Subsystem 8 (Parser)

This is the concrete next action. Start here in a fresh session.

1. **Read the key artifacts:**
   - `docs/superpowers/specs/2026-04-16-device-family-refactor-design.md` (parent)
   - `docs/superpowers/plans/2026-04-16-device-family-refactor-plan.md` (parent plan)
   - `docs/arch/subsys7-audit.md` -- Subsystem 7 completion; confirms
     `IsaExecutor` ships as an empty trait anchor, all divergence
     migrated as data to `xdna_archspec::aie2::{rounding, permute,
     vmac, matmul, ups, instruction_latency}` + `ProcessorModel`
     extensions. No loose ends.
   - `docs/arch/isa-execute-model.md` -- design note showing the
     "data in archspec, algorithms in xdna-emu" split and the reasons
     each candidate trait method was rejected.
   - `docs/arch/isa-decode.md` -- Subsystem 6 (ISA Decode) design note;
     precedent for landing without a trait seam.
   - `src/parser/` -- the current xdna-emu parser path. XCLBIN / PDI /
     ELF container handling. Also check `src/parser/cdo/` for CDO
     config parsing.

2. **Verify the current state hasn't drifted:**
   ```bash
   git log --oneline phase1-subsys-isa-execute..HEAD
   ```
   If nothing has landed since the tag, you're picking up exactly where
   Subsystem 7 left off.

   ```bash
   PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
   PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
   ```
   Expect xdna-emu `2684 passed; 0 failed; 5 ignored` and archspec
   `320 passed; 0 failed; 2 ignored`.

3. **Invoke brainstorming** to shape Subsystem 8's spec:
   ```
   /brainstorming
   ```
   Topic: "Phase 1b Subsystem 8: Parser (XCLBIN / PDI / ELF)."

   **Shape the spec around these questions:**
   - **Container format variance.** Do XCLBIN, PDI, and ELF formats
     themselves differ across AIE1/AIE2/AIE2P, or just the content
     they carry? XCLBIN is a Xilinx-wide container; the AIE-specific
     sections inside are what varies. The parser should distinguish
     "I can read the container" (arch-generic) from "I can interpret
     this AIE section" (arch-specific).
   - **The `BinaryLoader` trait question.** Mirrors the `IsaExecutor`
     question from Subsystem 7: does container parsing have
     algorithmic *shape* divergence between arches, or just *values*?
     Spec's prior is a trait with 2-4 methods around section dispatch;
     audit may disprove.
   - **CDO interpretation scope.** The CDO section inside XCLBIN
     carries DMA configurations, routing, lock setup -- all of which
     are arch-specific data that Subsystems 3-5 already migrated to
     archspec. The CDO parser should read those configs through the
     existing archspec data without arch-dispatch.
   - **AIE1/Versal breadth.** Versal FPGAs (out-of-scope today) pair
     AIE1 with PL fabric; their container layouts differ from XDNA's
     XCLBIN. Does the parser abstraction need to accommodate this,
     or is "Versal is out of scope" a fine excuse to skip?

4. **Invoke writing-plans** to produce a plan at
   `docs/superpowers/plans/YYYY-MM-DD-subsys8-parser.md`.

5. **Invoke subagent-driven-development** to execute.

6. **At end of Subsystem 8:** tag `phase1-subsys-parser`, append a
   completion section to its audit, update this `NEXT-STEPS.md` to
   mark Phase 1b complete (all eight subsystems done) and transition
   to Phase 2 hygiene or a milestone tag (`phase1-complete`).

---

## Useful Commands

```bash
# EVERY cargo build/test needs this PATH prepend (tblgen needs
# llvm-config 21.x, not mlir-aie's 23.x).  Hygiene fix deferred:
# archspec's build.rs should set LLVM_SYS_210_PREFIX explicitly.
export PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH

# Commit history at the current refactor frontier
git log --oneline phase1-subsys-isa-execute..HEAD
git log --oneline phase1-subsys-stream-switch..phase1-subsys-isa-execute

# Run library tests (Global Invariant; green at every commit)
cargo test --lib                         # expect 2684 pass at the tag
cargo test -p xdna-archspec --lib        # expect 320 pass

# Fast bridge smoke (catches 90% of regressions, ~30s)
./scripts/emu-bridge-test.sh --no-hw -v add_one_cpp_aiecc

# Full bridge run (~30 min) + ISA (~10 min), sequential
./scripts/emu-bridge-test.sh 2>&1 | tee /tmp/claude-1000/bridge-subsys<N>.log
./scripts/isa-test.sh         2>&1 | tee /tmp/claude-1000/isa-subsys<N>.log

# FFI cdylib rebuild (needed after Rust changes for bridge to pick
# them up -- `cargo build` alone does NOT update libxdna_emu.so).
# ALWAYS rebuild the FFI immediately before running the bridge
# gate at a subsystem tag.  A stale .so left over from a prior
# refactor has produced at least one phantom multi-test "regression"
# narrative (retracted 2026-04-18) -- rebuild first, draw
# conclusions second.  If the .so timestamp in target/debug looks
# older than the latest source commit, `cargo clean -p xdna-emu-ffi`
# before the rebuild.
cargo build -p xdna-emu-ffi

# What's currently in xdna-archspec
ls crates/xdna-archspec/src/
ls crates/xdna-archspec/src/aie2/

# What's currently in src/ (after Subsystem 6)
ls src/                                  # no src/tablegen/ anymore
ls src/interpreter/decode/               # register_map.rs lives here
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

## Known Pre-Existing Failures

These predate Phase 1 and are orthogonal to the refactor -- do not
block on them, do not investigate them mid-subsystem:

- `bd_chain_repeat_on_memtile` bridge EMU (Chess and Peano): the
  latent DMA deadlock documented in "Where We Are" above.  Verified
  pre-refactor.  An earlier iteration of this document posted a
  multi-test deadlock narrative based on the stale
  `subsys6-partB-bridge.log`; that narrative was wrong (the other
  tests pass once the FFI is rebuilt cleanly) and has been retracted.
- `cargo test -p xdna-archspec --lib`: `test_full_parse_all_devices`
  fails with device count 13 vs expected 12.  Pre-dates Phase 1a.
- Peano bridge EMU timeouts on `dma_task_large_linear` and
  `objectfifo_repeat/init_values_repeat`.  Also pre-existing.
- Generated file warnings (unused constants in `gen_aiert_*.rs`).
  Pre-existing.

---

## Deferred Workstreams (named, not yet scheduled)

### Cycle-budget bridge test timeouts (replace wall-clock)

**The issue.** `scripts/emu-bridge-test.sh` and the underlying
test-runner harness currently decide "this test timed out" based on
wall-clock seconds. That's fragile: any concurrent heavy CPU load
(ROCm compile, other `cargo build --release`, anything that spikes
the host) slows the emulator's per-cycle wall-time and makes tests
hit the wall-clock timeout even though they're making correct
progress. Surfaced during the Subsystem 7 gate: a concurrent ROCm
build made 20 Chess/Peano bridge tests flip PASS -> TIMEOUT, all
reproducible as environment contention rather than code regressions.

**The fix.** Replace wall-clock with cycle budget. The emulator
already tracks per-tile cycle counts; propagate a "max cycles" knob
through the FFI (env var or new FFI entry point), and have the
bridge harness declare timeout based on `emu_cycle_count > N` rather
than `elapsed_seconds > M`. Tests that genuinely deadlock (infinite
loop) still terminate; tests that are just slow under contention
still pass.

**Work scope.** ~half-day. Touches:
- `scripts/emu-bridge-test.sh` (timeout decision)
- `src/ffi/` (new max-cycles knob)
- Emulator main loop (check-and-abort on cycle threshold)
- Possibly test helpers in `mlir-aie/test/npu-xrt/` (if any tests
  assume wall-clock behavior rather than simulated-time semantics)

**When.** Its own brainstorm -> plan -> implementation cycle. Could
land as Phase 2 hygiene, or as a standalone piece of work after
Subsystem 8 (Parser). Not a Subsystem 7 concern despite being
surfaced by its gate.

---

## Contact Points

If you're a future session with no memory of this work, **don't spawn a
bunch of Explore agents to re-derive context** -- the artifacts above
have already done that work. Start from the spec, then the audit's
`## Completion` section, then `git log --oneline phase1a-consolidate`
to see what shipped. Only then dive into code.
