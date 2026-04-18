# Next Steps -- Device-Family Refactor

Recovery document for picking up this refactor in a future session. Read
this first, then dive into the authoritative artifacts below.

**Last updated:** 2026-04-17 (Phase 1b Subsystem 6 Part A landed; Part B pending)
**Current branch:** `dev` (no master merges until the refactor is done)
**Latest tag:** `phase1-subsys-isa-decode-partA` at `71050cb`

---

## Session Checkpoint (2026-04-17, mid-Subsystem-6)

**State at this checkpoint:**
- Subsystem 6 Part A landed at `phase1-subsys-isa-decode-partA` (`71050cb`).
- 11 commits cover the relocation: mega-move (`11f1275`) + build_helpers
  directory (`0b68c14`) + TableGen wiring (`24e755c`) + decoder_ffi C++
  (`749cec7`) + compile_llvm_decoder_ffi (`55ff796`) + extract_aiert /
  gen_aiert_* (`25c88be`) + supporting docs. Audit log at
  `docs/arch/subsys6-audit.md` details all deviations.
- Rust-side invariants green: release build clean, 2730 xdna-emu lib
  tests pass, 202/1 archspec lib tests (1 pre-existing fail), bridge
  `--no-hw -v add_one` green.
- **Full HW bridge + ISA suite running in background at time of
  checkpoint.** Bridge log at `/tmp/claude-1000/subsys6-partA-bridge.log`
  followed by ISA at `/tmp/claude-1000/subsys6-partA-isa.log`. Sequential
  (never parallel -- both target the NPU). Expect ~30 min bridge +
  ~10 min ISA.
- **Build-environment caveat:** every fresh `cargo build` now requires
  `export PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH` so
  tblgen's build-script finds llvm-config 21.x (not mlir-aie's 23.x).
  Prior cached artifacts hid this. Hygiene fix: archspec's build.rs
  should set `LLVM_SYS_210_PREFIX` explicitly. Flagged, not done.

**What remains in Subsystem 6 (Part B, 7 tasks):**
- Task 13: move `src/tablegen/register_map.rs` -> `src/interpreter/decode/register_map.rs`.
- Task 14: atomic rewrite of ~27 `crate::tablegen::*` consumers.
- Task 15: atomic rewrite of 36 `crate::arch::*` consumers.
- Task 16: delete `src/tablegen/` directory; delete `pub mod arch` + `pub mod tablegen` forwarder blocks in `src/lib.rs`.
- Task 17: shrink `xdna-emu/build.rs` to ~80 lines (plugin install only); empty `[build-dependencies]`.
- Task 18: write `docs/arch/isa-decode.md` design note.
- Task 19: full HW + ISA gate; tag `phase1-subsys-isa-decode`; update this NEXT-STEPS.md.

**To resume:**
1. Check HW bridge + ISA logs at `/tmp/claude-1000/subsys6-partA-{bridge,isa}.log` -- confirm no new regressions vs the `phase1-subsys-regs-mem` baseline.
2. Re-open the plan at `docs/superpowers/plans/2026-04-17-subsys6-isa-decode.md` starting at Task 13.
3. Invoke `superpowers:subagent-driven-development` to continue execution.

---

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
Subsystem 6 (ISA Decode) is now the immediate next step -- it has to
complete before Subsystems 2-5 / 7-8 proceed cleanly, because a lot of the
pending cleanup for Subsystem 1 (the 36 remaining `crate::arch::*`
consumers, the `mod arch` forwarder in `src/lib.rs`, the `extract_aiert`
duplication in xdna-emu/build.rs) all unblock when Subsystem 6 lands.

Phase 1a (from the earlier pass): the three parallel arch abstractions
(`xdna-archspec` crate + `src/archspec/mod.rs` + `src/device/arch_config.rs`)
collapsed into one -- the `xdna-archspec` workspace crate is the single
source-of-truth, and every consumer in `src/` imports directly from
`xdna_archspec::types` or `xdna_archspec::runtime`. The port-layout
methods in `src/device/port_layout.rs` that were once a runtime-side
remnant have since been consolidated through the Subsystem 1 work into
`xdna_archspec::aie2::{port_type, stream_switch}`.

**Verification that held (at `phase1-subsys-regs-mem`):**
- `cargo test --lib`: 2797 passed, 0 failed, 5 ignored. (Was 2798 before
  Task 15; the `sign_extend_7bit` unit test living inside the deleted
  `registers_spec.rs` correctly dissolved with the file. That function had
  zero external consumers.)
- `cargo build --release`: clean.
- `cargo test -p xdna-archspec --lib`: 138 pass / 1 pre-existing failure
  (`test_full_parse_all_devices`, device-count mismatch unrelated to the
  refactor).
- Bridge `--no-hw -v add_one`: Chess 10/10 PASS, Peano 9/9 PASS.
- Full HW bridge + ISA suite passed at the Part A tag with baseline
  regressions only (`bd_chain_repeat_on_memtile` fails on real NPU too, so
  it's not an emulator regression; Peano EMU timeouts stable against
  baseline). Part B was doc + file-dissolution only, no semantic change,
  so the 40-minute HW run was not re-executed at the Part B tag.

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
| 6 | ISA Decode | `phase1-subsys-isa-decode` | **Up next** | Bundle/slot layout, decoder tables, decoder FFI, and the tablegen types + resolver migration that unblocks Subsystem 1's deferred Tasks 9 and 10. 3-slot (AIE1) vs 6-slot (AIE2) is the biggest arch cliff. `IsaDecoder` trait. |
| 2 | Tile Topology | `phase1-subsys-tile-topo` | Pending | Replace `row == 0` / `row >= N` checks with `ArchModel`-backed classification. **Includes the deferred `TileType` -> `TileKind` deep rename**: merge `Shim`/`ShimNoc`, rename `MemTile` -> `Mem`. |
| 3 | DMA Engine & BD Format | `phase1-subsys-dma` | Pending | First behavioral seam. Audit AIE2 vs AIE1 BD layout via aie-rt source. Lift BD parse/encode + channel stepping behind `DmaModel` trait. |
| 4 | Locks | `phase1-subsys-locks` | Pending | Small seam exercise. `LockModel` trait if acquire/release/value semantics genuinely differ (likely around lock value width). |
| 5 | Stream Switch | `phase1-subsys-stream-switch` | Pending | Topology (data, already via archspec) + routing legality (behavior, trait: `StreamSwitchModel`). |
| 7 | ISA Execute | `phase1-subsys-isa-execute` | Pending | Semantic ops, intrinsic handlers. Biggest; largest files live here (`vmac_routing.rs` 239KB, `memory/mod.rs` 124KB). `IsaExecutor` trait. |
| 8 | Parser (XCLBIN / PDI / ELF) | `phase1-subsys-parser` | Container format variance. `BinaryLoader` trait. |

**End state after Phase 1:** adding AIE2P is "implement the traits for a
second arch"; adding AIE1/Versal is "implement the traits + extend for
platform differences." Neither requires re-plumbing.

---

## How to Pick Up Subsystem 6 (ISA Decode)

This is the concrete next action. Start here in a fresh session.

1. **Read the key artifacts:**
   - `docs/superpowers/specs/2026-04-16-device-family-refactor-design.md` (parent)
   - `docs/superpowers/plans/2026-04-16-device-family-refactor-plan.md` (parent plan)
   - `docs/arch/subsys1-audit.md` (includes `## Task 9 Deferral`, `## Task 10 Deferral`, and the `## Follow-ups flagged for Subsystem 6` list at the bottom)
   - `docs/arch/registers-memory-map.md` (the Subsystem 1 design note)
   - `docs/superpowers/plans/2026-04-17-subsys1-regs-mem.md` (Subsystem 1's plan, including its blocked Tasks 9 and 10 for reference)

2. **Verify the current state hasn't drifted:**
   ```bash
   git log --oneline phase1-subsys-regs-mem..HEAD
   ```
   If nothing has landed since the tag, you're picking up exactly where
   Subsystem 1 left off.

   ```bash
   cargo test --lib 2>&1 | tail -5
   ```
   Expect: `2797 passed; 0 failed; 5 ignored`. If different, investigate
   before proceeding.

3. **Invoke brainstorming** to shape Subsystem 6's spec:
   ```
   /brainstorming
   ```
   Topic: "Phase 1b Subsystem 6: ISA Decode, plus Subsystem 1 leftovers."
   The brainstorming should produce a spec at
   `docs/superpowers/specs/YYYY-MM-DD-subsys6-isa-decode-design.md`.

   **Shape the spec around these questions (the Subsystem 1 follow-up
   list in `docs/arch/subsys1-audit.md` enumerates them):**
   - Can `src/tablegen/types.rs` and `src/tablegen/resolver/` move wholesale
     to `xdna_archspec::aie2::isa::`, or do they need to decompose into
     arch-agnostic types (archspec) and xdna-emu-specific wrappers?
   - Where does `interpreter::bundle::slot::Operand` actually belong? If it
     describes a slot-typed operand abstractly, archspec. If it's the
     emulator's interpretation of an operand during execution, xdna-emu.
   - Does the `RegisterMap` / `MappedOperand` / `classify_reg_name` layer
     want a `RegisterMapper` trait in archspec or a direct move of
     `Operand` + constants (the cleaner but bigger move)?
   - After the types migrate, can `build_helpers/` move + `gen_tablegen`
     expose as `xdna_archspec::aie2::isa::decoder_tables::*`?
   - After the types migrate, can the `extern "C"` block move + FFI Rust
     shim expose as `xdna_archspec::aie2::decoder_ffi::*`? The
     register-name mapping layer might stay in xdna-emu if it's
     fundamentally an emulator concern.
   - Is 3-slot (AIE1) vs 6-slot (AIE2) the right place to introduce an
     `IsaDecoder` trait, or is the VLIW bundle structure itself still data
     (archspec) with only the decoding behavior (bundle disassembly)
     behind a trait?

4. **Invoke writing-plans** next to produce a plan at
   `docs/superpowers/plans/YYYY-MM-DD-subsys6-isa-decode.md`. Mention
   explicitly that this plan consumes the deferred Subsystem 1 Tasks 9
   and 10, plus the `mod arch` forwarder dissolution and the 36 remaining
   `crate::arch::*` consumer rewrites.

5. **Invoke subagent-driven-development** to execute:
   ```
   /subagent-driven-development
   ```
   Same flow as prior subsystems: implementer subagent per task, two-stage
   review (spec compliance then code quality), same-session orchestration.

6. **At end of Subsystem 6:** tag `phase1-subsys-isa-decode`, append a
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
