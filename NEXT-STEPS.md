# Next Steps -- Device-Family Refactor

Recovery document for picking up this refactor in a future session. Read
this first, then dive into the authoritative artifacts below.

**Last updated:** 2026-04-17 (after Phase 1a landed)
**Current branch:** `dev` (no master merges until the refactor is done)
**Latest tag:** `phase1a-consolidate` at `6534e28`

---

## Where We Are

Phase 1a of the device-family refactor is complete. The three parallel
arch abstractions (`xdna-archspec` crate + `src/archspec/mod.rs` +
`src/device/arch_config.rs`) have collapsed into one: the `xdna-archspec`
workspace crate is the single source-of-truth, and every consumer in
`src/` imports directly from `xdna_archspec::types` or
`xdna_archspec::runtime`. The only runtime-side remnant is
`src/device/port_layout.rs` -- a narrow `PortLayout` extension trait for
the six stream-switch methods whose data comes from build.rs-generated
`crate::arch::*` and can't live in the workspace crate yet.

**Verification that held:**
- `cargo test --lib`: 2798 passed, 0 failed, 5 ignored.
- `cargo build --release`: clean.
- `cargo test -p xdna-archspec --lib`: 14 `runtime::tests` pass; 1
  pre-existing `test_full_parse_all_devices` failure unrelated to the
  refactor.
- Bridge test `--no-hw` smoke (add_one): Chess 10/10 PASS, Peano 9/9 PASS.
- Bridge test full (HW + EMU, dual-compiler, 119 tests): Chess 64/64
  PASS, Peano 55/55 compiled with 2 pre-existing Peano-EMU timeouts
  (`dma_task_large_linear`, `objectfifo_repeat/init_values_repeat`) and
  1 XFAIL. **No regressions from Phase 1a.**

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

| # | Subsystem | Tag | Description |
|---|-----------|-----|-------------|
| 1 | Registers & Memory Map | `phase1-subsys-regs-mem` | Plumb hardcoded register offsets, memory sizes, per-tile-type counts through `ArchModel`. Mostly data; likely no new trait. |
| 2 | Tile Topology | `phase1-subsys-tile-topo` | Replace `row == 0` / `row >= N` checks with `ArchModel`-backed classification. **Includes the deferred `TileType` -> `TileKind` deep rename**: merge `Shim`/`ShimNoc`, rename `MemTile` -> `Mem`. |
| 3 | DMA Engine & BD Format | `phase1-subsys-dma` | First behavioral seam. Audit AIE2 vs AIE1 BD layout via aie-rt source. Lift BD parse/encode + channel stepping behind `DmaModel` trait. |
| 4 | Locks | `phase1-subsys-locks` | Small seam exercise. `LockModel` trait if acquire/release/value semantics genuinely differ (likely around lock value width). |
| 5 | Stream Switch | `phase1-subsys-stream-switch` | Topology (data, already via archspec) + routing legality (behavior, trait: `StreamSwitchModel`). Possibly the right moment to move `crate::arch::*` port generation into the archspec crate if AIE2P forces it. |
| 6 | ISA Decode | `phase1-subsys-isa-decode` | Bundle/slot layout, decoder tables. 3-slot (AIE1) vs 6-slot (AIE2) is the biggest arch cliff. `IsaDecoder` trait. |
| 7 | ISA Execute | `phase1-subsys-isa-execute` | Semantic ops, intrinsic handlers. Biggest; largest files live here (`vmac_routing.rs` 239KB, `memory/mod.rs` 124KB). `IsaExecutor` trait. |
| 8 | Parser (XCLBIN / PDI / ELF) | `phase1-subsys-parser` | Container format variance. `BinaryLoader` trait. |

**End state after Phase 1:** adding AIE2P is "implement the traits for a
second arch"; adding AIE1/Versal is "implement the traits + extend for
platform differences." Neither requires re-plumbing.

---

## How to Pick Up Subsystem 1 (Registers & Memory Map)

This is the concrete next action. Start here in a fresh session.

1. **Read the key artifacts** (spec, plan, audit -- links above).

2. **Verify the current state hasn't drifted:**
   ```bash
   git log --oneline phase1a-consolidate..HEAD
   ```
   If nothing has landed since the tag, you're picking up exactly where
   Phase 1a left off.

   ```bash
   cargo test --lib 2>&1 | tail -5
   ```
   Expect: `2798 passed; 0 failed; 5 ignored`. If different, investigate
   before proceeding.

3. **Invoke brainstorming** to shape Subsystem 1's spec:
   ```
   /brainstorming
   ```
   Topic: "Phase 1b Subsystem 1: Registers & Memory Map." The
   brainstorming should produce a spec at
   `docs/superpowers/specs/YYYY-MM-DD-subsys1-regs-mem-design.md`.

   **Shape the spec around these questions:**
   - Where are register offsets currently hardcoded in `src/device/`?
   - Where are per-tile-type memory sizes hardcoded (not already through
     `ArchConfig::data_memory_size` / `program_memory_size`)?
   - Does anything bypass `ArchModel` and read from `crate::arch::*`
     directly outside `port_layout.rs`?
   - Are there any register-offset constants in
     `src/interpreter/` that should move to `xdna-archspec`?

4. **Invoke writing-plans** next to produce a bite-sized plan at
   `docs/superpowers/plans/YYYY-MM-DD-subsys1-regs-mem.md`.

5. **Invoke subagent-driven-development** to execute:
   ```
   /subagent-driven-development
   ```
   Same skill flow as Phase 1a: implementer subagent per task, two-stage
   review (spec compliance then code quality), same-session orchestration.

6. **At end of Subsystem 1:** tag `phase1-subsys-regs-mem`, append a
   completion section to the subsystem's audit, update this
   `NEXT-STEPS.md` to move Subsystem 2 to "up next."

### What to expect

Based on the Phase 1a audit's ArchConfig trait surface, the low-hanging
fruit for Subsystem 1:
- `crate::arch::*` references outside `port_layout.rs` (there are none
  currently; Phase 1a cleaned them up -- verify).
- Any `const` register offsets defined in `src/device/**` that duplicate
  AM025 register database content.
- Per-tile-type memory size assumptions hardcoded in
  `src/interpreter/` or `src/device/state/`.
- `src/device/regdb/` already imports from `xdna_archspec::regdb::*`;
  confirm nothing there hardcodes offsets.

This subsystem is expected to be mostly plumbing without a new trait
seam. If the audit surprises us with actual behavioral variance (e.g.,
some register layouts differ across AIE2/AIE2P in ways we thought were
just offset differences), a `RegisterMap` trait may be justified.

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
