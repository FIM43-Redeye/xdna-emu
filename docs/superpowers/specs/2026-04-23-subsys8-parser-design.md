# Subsystem 8: Parser (XCLBIN / PDI / CDO / ELF) -- Design

**Phase:** 1b, final subsystem
**Parent:** [2026-04-16-device-family-refactor-design.md](2026-04-16-device-family-refactor-design.md)
**Precedents:** Subsystems 1--7 (see `docs/arch/subsys*-audit.md`,
`docs/arch/*-model.md`)
**Status:** Design; implementation pending plan (via `writing-plans`)

---

## Purpose

Subsystem 8 is the last Phase 1b pass and has two objectives at equal
weight:

1. **Arch-seam completion.** Audit the parser surface (XCLBIN, AIE
   Partition, CDO, ELF) for AIE2 assumptions; migrate arch-specific
   *data* to `xdna-archspec`; decide whether a `BinaryLoader` trait
   seam is warranted. Same methodology every prior Phase 1b subsystem
   used.
2. **Parser-clarity payoff.** Fix the seven concrete pain points
   accumulated in the parser stack (enumerated below). The refactor's
   goal is "make future fixes easier," and the parser module has been
   a recurring drag on debugging and extension. This subsystem treats
   that as equal-weight work, not a stretch goal.

The "deep and global" framing of the parent refactor applies: we
don't stop at the arch-audit boundary just because Subsystem 7 did.
The parser pain points are within intended scope.

## Design philosophy

Two framing principles the user set explicitly during brainstorming,
elevated here so downstream plan authors and implementers inherit
them:

- **Elegant design over pristine purity.** The previous subsystems'
  "archspec doesn't touch runtime" and "parser is arch-blind"
  invariants were useful when they came for free. We don't cling to
  them when they block a better shape. The `semantics.rs` sublayer
  introduced below reads archspec deliberately; that's a departure
  from Subsystem 6's posture and it is correct for this subsystem.
- **Code goes big when the problem is big.** We don't under-scope to
  protect a small-file aesthetic. The subsystem's boundary is "what
  needs to exist for the parser to be good"; code size follows from
  that, not the other way around. What we guard against is
  disorganized largeness -- the code that lands should be tight,
  well-named, well-tested, and legible.

## Scope

### In scope

- Arch audit of `src/parser/*` and its direct consumers
  (`src/device/state/`, `src/interpreter/test_runner.rs`,
  `coordinator.rs`, `crossref.rs`, `decoder.rs`,
  `src/integration/elfanalyzer.rs`).
- Data migrations surfaced by the audit (CDO opcode identity, ELF
  machine-type constants, XCLBIN section-kind AIE classification,
  and whatever else surfaces).
- `BinaryLoader` trait decision (populated / empty anchor / no
  trait). Empty anchor is the prior per the parent spec's "reserve
  the dispatch pathway" argument from Subsystem 7; the audit can
  push to either pole. **Note:** the three landings describe the
  *top-level* `BinaryLoader` trait (which governs XCLBIN-container
  arch dispatch if any exists). The CDO `semantics::lower` function
  is always a plain free function taking `&ArchHandle` -- not a
  trait method -- regardless of the `BinaryLoader` decision. Per-arch
  `semantics::lower` variance, if audit surfaces any, lives as
  dispatch *inside* the plain function (e.g., match on
  `arch.family()` for AIE1 vs AIE2 shape differences), not as a
  trait method. This keeps the hot apply-loop monomorphic.
- **CDO two-layer split** (`framing.rs` + `syntax.rs` + `semantics.rs`)
  and the `DeviceOp` vocabulary introduced between parser and
  `device/state`.
- **ELF loading deduplication** into a single canonical `AieElf`
  API.
- **Diagnostics upgrade** -- `ParseError` enum carrying offset /
  expected / got / context across every fallible boundary.
- **Test fixture infrastructure** -- minimal-XCLBIN / CDO /
  ELF builder helpers under a test-support module so parser tests
  can exercise edge cases without depending on full real-world
  binaries.
- **Control-packet parser alignment** -- audit `src/device/
  control_packets/parser.rs` against `src/parser/*`, decide on shared
  primitives.
- "What would AIE1 look like?" design note at
  `docs/arch/binary-loader-model.md`.

### Explicitly out of scope

- **Versal / AIE1 implementation.** Matches parent spec's non-goals.
  We design seams that *could* accommodate Versal PDI / AIE1 ELFs,
  but populate none of them. The design note documents the
  hypothetical AIE1 answer; no AIE1 code lands.
- **Performance optimization of the interpreter inner loop.** That's
  Phase 2 territory. Subsystem 8 only ensures its new vocabulary
  (`DeviceOp`) doesn't *introduce* hot-path allocations or dynamic
  dispatch; it doesn't optimize pre-existing hot paths.
- **Control-packet runtime behavior rework.** 8c's control-packet
  task is audit + align-primitives-if-sensible, not a rewrite of
  control-packet processing semantics.
- **Master-branch merges.** All work lands on `dev`. Per-stage tags
  (see below) are the navigable checkpoints.

## Three-stage rollout

Subsystem 8 decomposes into three tagged stages. Each stage is
bisect-safe: `cargo test --lib` green at every commit, bridge
smoke + full pass green before the stage tag.

### Stage 8a: Arch audit + data migrations

**Tag:** `phase1-subsys-parser-arch`

Classic Phase 1b audit pass. Deliverables:

- `docs/arch/subsys8-audit.md` with nine sections (listed in "Audit
  methodology" below).
- All arch-specific data surfaced by the audit migrated to
  `xdna-archspec`. Likely candidates: CDO opcode identity map
  (whether that's discriminants of today's `CdoCommand`, a new
  bytes-to-variant table, or both -- audit decides), ELF machine
  type constants, XCLBIN `SectionKind` AIE-membership table. Audit
  confirms or expands this list.
- `BinaryLoader` trait decision documented in the audit and
  implemented. Three possible landings:
  - **Populated trait** (1--3 methods) if `semantics.rs` lowering
    shows real algorithmic variance across arches.
  - **Empty anchor** (matches `IsaExecutor`) if variance is
    data-expressible.
  - **No trait** (matches `IsaDecoder`) if even the anchor doesn't
    buy anything.
- `docs/arch/binary-loader-model.md` design note answering "what
  would AIE1 look like?" for the parser layer.
- **Refined `DeviceOp` enum proposal.** The starting hypothesis in
  §"`DeviceOp` vocabulary" below is pre-audit. Stage 8a's CDO-semantics
  audit (section 4 of the audit doc) counts which `CdoRaw` variants
  actually appear in the XCLBINs on disk, groups them by device
  effect, and produces a refined variant list. That refined list is
  a gated deliverable (see "Gate between 8a and 8b" below).

This stage is data + documentation + maybe one small trait. No
structural changes to parser module layout yet.

### Gate between 8a and 8b

**8b does not start until the user has reviewed 8a's refined
`DeviceOp` proposal and confirmed it.** The gate exists because:

- The pre-audit `DeviceOp` shape in §"`DeviceOp` vocabulary" is a
  hypothesis, not a commitment. Real CDO command populations may
  surface variants not listed (e.g., counter / perfmon setup,
  debug-only commands, platform events), or show that some proposed
  structured variants aren't worth the type (e.g., `StreamRoute`
  might be better expressed as a sequence of `RegWrite`s).
- The user has explicitly flagged that `DeviceOp` shape should be
  evaluated based on audit findings, with room for deeper thinking
  before implementation locks in.
- Reshaping a vocabulary is dramatically cheaper pre-implementation
  than mid-implementation -- every variant touches `semantics::lower`,
  `device/state::apply`, and downstream `DeviceOp`-consuming tests.

The gate is explicit so it can't be sleepwalked past. 8a's closing
commit updates the spec's "`DeviceOp` vocabulary" section to reflect
the refined proposal; user reviews that delta; 8b begins only after
confirmation. If the user wants a brainstorming re-entry to shape
the refined list, that's fine and expected.

### Stage 8b: Coupling cleanups

**Tag:** `phase1-subsys-parser-coupling`

The structural-knot pass, structured in two halves with an
intermediate bisect-safe checkpoint between them.

#### Half 1 (internal split, no behavior change)

1. **Rename `CdoCommand` -> `CdoRaw`** throughout the tree. Today's
   `CdoCommand` consumers (`src/device/state/mod.rs`, `src/main.rs`,
   `interpreter/test_runner.rs`) migrate to the new name. No
   behavioral change.
2. **Three-way CDO module split.** `src/parser/cdo.rs` (835 LOC)
   becomes:
   - `src/parser/cdo/mod.rs` -- public API, re-exports.
   - `src/parser/cdo/framing.rs` -- header parse, version detection,
     command-length framing. Byte-level, independent of opcode
     meaning.
   - `src/parser/cdo/syntax.rs` -- emits `CdoRaw::{Write32,
     MaskWrite32, Block*, DmaBdConfig, ...}`. May read archspec if
     the audit finds per-arch byte-format differences; not required
     to.
   - `src/parser/cdo/semantics.rs` -- stub in this half. Accepts
     `CdoRaw`, returns `CdoRaw` (pass-through). No `DeviceOp` yet.

At the close of Half 1 there's a bisect-safe commit: tree compiles,
`cargo test --lib` + `archspec` + bridge-smoke all green, internal
structure is in place, but the `device/state` boundary still
consumes `CdoRaw` directly (via the pass-through `semantics`). This
is the **rollback anchor** for Half 2: if `DeviceOp` shape proves
wrong mid-flight, the stage lands here and `DeviceOp` is deferred
to a follow-up.

#### Half 2 (boundary move)

3. **Introduce `DeviceOp`.** New module `src/device/ops.rs` defines
   the vocabulary enum. `semantics.rs` stops being a pass-through;
   its `lower` function now emits `DeviceOp`.
4. **Move the `device/state` boundary.** `device/state/mod.rs` stops
   consuming `CdoRaw`; consumes `DeviceOp` and pattern-matches on
   that. Parser's public surface to `device/state` becomes
   `impl Iterator<Item = DeviceOp>`.

Stage closes with the tag `phase1-subsys-parser-coupling`.

Note: **ELF dedup moved to 8c** (from an earlier draft) because it
has no coupling/structural-knot justification -- it's pure
deduplication hygiene. 8b stays focused on the CDO boundary.

### Stage 8c: Ergonomics

**Tag:** `phase1-subsys-parser-ergonomics`

The make-it-pleasant pass. Four sub-deliverables, each independent
(rollback of any one leaves the others intact).

1. **Diagnostics.** `ParseError` enum replacing most `bail!` and
   `anyhow!` sites. Each variant carries `{offset, expected, got,
   hex_context}`. Error-path messages become actionable (point at
   the offending byte with a small hex dump). `anyhow::Context`
   chains where frame-level context adds value.

2. **Test fixtures.** Minimal-XCLBIN / CDO / ELF builder helpers
   under `src/parser/testing/` (so they're visible to unit tests
   inside `src/parser/cdo/*.rs` and `src/parser/elf.rs` without
   needing a separate integration-test crate). Builders expose a
   fluent API:
   `XclbinBuilder::new().with_partition(...).with_cdo(...).build()`.
   Enables parser unit tests to cover edge cases (truncated
   sections, malformed CDO headers, unknown ELF machines, mixed
   command streams) without real-binary dependencies.

3. **ELF loading deduplication.** One canonical
   `AieElf::load_into(&mut CoreMemory)` + helper methods for symbol
   lookup, section iteration. Five current consumers (`test_runner`,
   `coordinator`, `crossref`, `decoder`, `integration/elfanalyzer`)
   migrate. Pure dedup hygiene; moved here from 8b for flavor
   coherence (see 8b's closing note).

4. **Control-packet parser alignment.** Based on 8a's audit of
   `src/device/control_packets/parser.rs`:
   - If they share framing primitives, extract a common
     `src/parser/framing.rs` (or similar) and reuse.
   - If they're unrelated, leave as is with a comment explaining
     the non-overlap.
   - If they substantially overlap (unlikely but possible), fold
     control-packets into `src/parser/` proper.

## `DeviceOp` vocabulary

### Shape

The refined proposal below replaces the starting hypothesis (see Subsystem 8 audit §Closing Summary for the reasoning). User-gated: Stage 8b Half 2 does not start until this enum is confirmed.

```rust
// src/device/ops.rs -- new module (Stage 8b Half 2)
//
// DeviceOp: arch-generic, device-facing operation vocabulary.
// Produced by cdo::semantics::lower(); consumed by device::state::apply().
//
// Design rules (from spec §"`DeviceOp` vocabulary"):
// 1. Device-facing, not CDO-facing (two CDO opcodes can produce the same op).
// 2. Arch-generic (TileAddr, BdFields, StreamRouteSpec via archspec).
// 3. Mixed granularity (RegWrite is the escape hatch).
// 4. Value-typed (Copy where possible; SmallVec for burst data only).
// 5. Audit-refined variant list (this block supersedes the starting hypothesis).

use xdna_archspec::types::TileAddr;
use xdna_archspec::aie2::dma::BdFields;
use xdna_archspec::aie2::stream_switch::StreamRouteSpec;
use xdna_archspec::aie2::dma::{DmaChannelId, DmaDir};
use smallvec::SmallVec;

#[derive(Debug, Clone)]
pub enum DeviceOp {
    // --- Register-level writes (dominant CDO outcomes, ~75% of commands) ---

    /// Single 32-bit register write.
    /// Produced by: CdoRaw::Write, CdoRaw::Write64 (after 64->32 truncation).
    RegWrite { tile: TileAddr, offset: u32, value: u32 },

    /// Masked register write: *reg = (*reg & !mask) | (value & mask).
    /// Produced by: CdoRaw::MaskWrite, CdoRaw::MaskWrite64.
    RegMask { tile: TileAddr, offset: u32, mask: u32, value: u32 },

    /// Bulk write: consecutive words starting at offset.
    /// Produced by: CdoRaw::DmaWrite (program/data memory loads).
    /// Uses SmallVec to avoid heap allocation for small payloads (<= 8 words).
    RegBurst { tile: TileAddr, offset: u32, words: SmallVec<[u32; 8]> },

    // --- Structured writes (archspec already names these) ---

    /// Configure a DMA Buffer Descriptor.
    /// Produced by: CdoRaw::Write to DMA BD register range (semantics::lower
    /// recognizes the offset range and calls arch.dma_model().parse_bd_words()).
    BdConfigure { tile: TileAddr, bd_id: u8, fields: BdFields },

    /// Initialize a lock to a specific value.
    /// Produced by: CdoRaw::Write to lock register range.
    LockInit { tile: TileAddr, lock_id: u8, value: i32 },

    /// Configure a stream switch connection.
    /// Produced by: CdoRaw::Write to stream switch register range.
    StreamRoute { tile: TileAddr, route: StreamRouteSpec },

    // --- Coarse control ---

    /// Enable or disable a compute core.
    /// Produced by: CdoRaw::Write or CdoRaw::MaskWrite to Core_Control register.
    CoreEnable { tile: TileAddr, enabled: bool },

    /// Start a DMA channel.
    /// Produced by: CdoRaw::Write to DMA channel start register.
    DmaStart { tile: TileAddr, channel: DmaChannelId, dir: DmaDir },

    // --- Synchronization / timing (audit-discovered; not in starting hypothesis) ---

    /// Poll a register until (value & mask) == expected.
    /// On real hardware: blocks until the condition is met (DMA completion, etc.).
    /// In the emulator: currently a logged no-op (writes are synchronous).
    /// Retaining as a variant preserves the information for future cycle-accurate work.
    /// Produced by: CdoRaw::MaskPoll, CdoRaw::MaskPoll64.
    /// Copy-able: all fields are primitive.
    MaskPoll { tile: TileAddr, offset: u32, mask: u32, expected: u32 },

    /// Timing delay for N cycles.
    /// On real hardware: inserts a wait. In the emulator: no-op.
    /// Produced by: CdoRaw::Delay.
    /// Copy-able.
    Delay { cycles: u32 },

    /// Debug sequence marker (value is an opaque tag).
    /// Always a no-op in device-state; useful for trace and test tooling.
    /// Produced by: CdoRaw::Marker.
    /// Copy-able.
    Marker { value: u32 },
}
```

### Design rules

1. **Device-facing, not CDO-facing.** Two CDO variants that produce
   the same device effect lower to the same `DeviceOp`. The vocabulary
   describes what happens, not how it was encoded.
2. **Arch-generic.** `TileAddr`, `BdFields`, `StreamRouteSpec`
   resolve through archspec. New variants appear only if the device
   does something new, not because the CDO encoded it differently.
3. **Mixed granularity, intentionally.** `RegWrite` is the escape
   hatch for CDO commands whose semantics we don't want to promote
   to a structured op. Structured variants exist where archspec
   already knows the semantics. This avoids a "can't represent the
   thing that came in" scenario.
4. **Value-typed.** `RegWrite` / `RegMask` / `CoreEnable` / etc. are
   `Copy`. `RegBurst` uses `SmallVec<[u32; 8]>` -- inline up to 8
   words (fits the common case), spills to heap beyond. No `Arc`,
   `Rc`, or `Box` in the vocabulary.
5. **Audit refines the variant list.** The list above is the
   starting hypothesis. 8a's CDO-semantics audit counts which
   `CdoRaw` variants appear in the XCLBINs on disk, groups them by
   device effect, and proposes the concrete `DeviceOp` enum. If the
   audit finds 95% of CDO commands are plain writes with a handful
   of structured outliers, the enum stays thin. If the audit finds
   a zoo, the enum grows.

### Lowering function

```rust
// src/parser/cdo/semantics.rs
pub fn lower(raw: &CdoRaw, arch: &ArchHandle)
    -> SmallVec<[DeviceOp; 4]>
{
    match raw {
        CdoRaw::Write32 { addr, value } => {
            let (tile, offset) = arch.memory_map().decode_global(*addr);
            smallvec![DeviceOp::RegWrite { tile, offset, value: *value }]
        }
        CdoRaw::DmaBdConfig { tile, bd_id, words } => {
            let fields = arch.dma_model().parse_bd_words(words);
            smallvec![DeviceOp::BdConfigure {
                tile: *tile, bd_id: *bd_id, fields
            }]
        }
        // ... remaining variants
    }
}
```

Plain free function, not a trait method. Inline-friendly. All arch
dispatch happens through the existing archspec accessors
(`memory_map`, `dma_model`, etc.); `lower` adds no new arch-dispatch
surface.

## End-to-end data flow

```
XCLBIN bytes
    |
    +-> xclbin::parse ----> structured sections
    |       |
    |       +-> aie_partition::parse ----> PDI bytes
    |             |
    |             +-> cdo::framing::split ---> Iterator<RawFrame>
    |                   |                    (byte slices, one per command)
    |                   |
    |                   +-> cdo::syntax::decode --> Iterator<CdoRaw>
    |                         |              (byte-level, typed)
    |                         |
    |                         +-> cdo::semantics::lower(arch)
    |                                                 --> Iterator<DeviceOp>
    |                                                      (arch-generic,
    |                                                       device-facing)
    |                                                 |
    |                                                 +-> device::state::apply
    |                                                      (&mut state, op)
    |
    +-> elf sections
          |
          +-> AieElf::load_into(&mut core_memory)
                (no DeviceOp stream by default; ELFs write core
                 memory directly. 8a's ELF consumer audit confirms
                 or refutes this: if one of the five consumers
                 needs more than raw-bytes-to-memory -- e.g.,
                 elfanalyzer wants structured section events --
                 an `ElfOp` sibling vocabulary is added and 8c's
                 ELF dedup flows through it.)
```

## Performance stance

Parser runs once at XCLBIN load -- not per-cycle. But XCLBIN load
time is user-visible every test run, and what the parser emits feeds
`device/state::apply`, which is hot during initial device setup
(thousands of CDO commands per XCLBIN). Disciplines that flow from
this:

1. **Monomorphize through the hot part.** If a `BinaryLoader` trait
   materializes, it dispatches at the top-level "which arch" boundary.
   Below that, static dispatch: `semantics::lower` is a plain
   function, `device/state::apply` is a plain match, no `dyn` in the
   apply loop.
2. **`device::state::apply(&mut state, op)` is allocation-free and
   inlinable.** Each `DeviceOp` variant produces a handful of
   register writes and state updates. No per-op `Box::new` or
   `Vec::new`.
3. **Stream, don't collect.** `cdo::semantics::lower` returns an
   iterator / small per-command SmallVec. The full
   `Vec<DeviceOp>` of a whole XCLBIN never materializes.
4. **`DeviceOp` variants are value-typed (per design rule 4 above).**
5. **`AieElf::load_into` writes directly to `CoreMemory`.** No
   intermediate buffers, matches current code's strategy.

The parent spec's "Monomorphize where hot" rule applies; the
emphasis here is that we thread it through the initial-setup loop,
not just the runtime inner loop.

## Audit methodology (8a)

Nine audit sections targeting `docs/arch/subsys8-audit.md`:

1. **XCLBIN section-kind classification.** Table: for each
   `SectionKind` variant, what's its AIE relevance (arch-agnostic
   / AIE-wide / AIE2-specific / unused) and which code path
   consumes it?
2. **AIE Partition wrapper audit.** Thin, probably one-page.
   Arch-variance enumeration.
3. **CDO syntax audit.** For each `CdoCommand` variant: byte format
   across arches; per-arch frequency counts in XCLBINs on disk;
   whether it belongs in `CdoRaw` as-is.
4. **CDO semantics audit.** For each `CdoCommand`: device effect;
   archspec representation (existing or needed); proposed
   `DeviceOp` variant it lowers into.
5. **Device-state consumer audit.** Walk every branch in
   `device/state/mod.rs` that matches on `CdoCommand`. Classify:
   moves to `semantics::lower` / stays in `apply` as a `DeviceOp`
   consumer / is dead code.
6. **ELF consumer audit.** Five call sites of `AieElf`. Each
   consumer's needs (program / data / symbol / section-iter)
   tabulated. Design the one canonical API as the superset.
7. **Control-packet parser overlap audit.** Compare
   `src/device/control_packets/parser.rs` byte-for-byte patterns
   with `src/parser/*`. Deliverable: field-level comparison table
   (framing primitives, byte-ordering conventions, error handling,
   type-reuse opportunities). Decision threshold: if >=3 framing
   primitives are effectively duplicated, extract shared module;
   if 1--2 are similar but not identical, leave as is with a
   comment; if overlap is coincidental only, explicitly document
   non-overlap. Output feeds 8c item 4 directly -- 8c does not
   relitigate the decision.
8. **Design note.** `docs/arch/binary-loader-model.md` -- the
   "what would AIE1 look like?" answer for the parser layer.
   Likely short; most variance lives in archspec constants and
   `CdoRaw` variant set and `semantics::lower` table entries.
9. **Trait-or-no-trait decision.** Based on sections 1--8.
   Conclusion documented with reasoning, matching the format of
   `docs/arch/isa-execute-model.md`'s four-candidate rejection
   table.

## Verification gates

**Per-commit:**
- `cargo test --lib` green (2686+ pass, 0 fail, 5 ignored).
- `cargo test -p xdna-archspec --lib` green (320+ pass) when the
  commit touches `crates/xdna-archspec/`.
- Bridge smoke green: `./scripts/emu-bridge-test.sh --no-hw -v
  add_one_cpp_aiecc` (Chess + Peano). Catches 90% of regressions
  in ~30 s.
- `cargo build -p xdna-emu-ffi` **before any bridge smoke** for
  commits in 8b (the stage that touches types crossing the FFI
  boundary). Stale-`.so` pitfall documented in NEXT-STEPS; 8b
  commits are the highest-risk surface for it.

**Per-stage-tag (`phase1-subsys-parser-arch`, `-coupling`,
`-ergonomics`) -- additional gates beyond per-commit:**
- Full bridge run green: `./scripts/emu-bridge-test.sh 2>&1 | tee
  /tmp/claude-1000/subsys8-<stage>-bridge.log`. Compared to
  pre-subsystem baseline captured during 8a kickoff.
- ISA test run green: `./scripts/isa-test.sh 2>&1 | tee
  /tmp/claude-1000/subsys8-<stage>-isa.log`. Compared to
  pre-subsystem baseline.

The pre-subsystem baseline is captured as the first action of 8a,
before any code changes, so each stage tag has something to
compare against. The per-commit smoke strictly subsets the per-tag
full bridge; both listings exist because they serve different
cadences (fast feedback vs thorough).

**8b Half-1 intermediate checkpoint:** not a tag (would inflate
tag count), but the closing commit of Half 1 must pass the full
per-stage-tag gates (not just per-commit). It's the bisect anchor
for Half 2's rollback posture; if Half 2 stumbles, `git checkout`
to that commit yields a shippable internal-split state.

## Rollback posture

**8b rollback is load-bearing** because `DeviceOp` is a genuine
design hypothesis that might prove wrong. The Half-1 intermediate
checkpoint (see §Stage 8b) makes rollback concrete:
- If `DeviceOp` shape proves wrong mid-Half-2, `git checkout` to
  the Half-1 closing commit yields a shippable state: internal CDO
  split is done (framing/syntax/semantics modules exist, `CdoRaw`
  is the type, `semantics::lower` is a pass-through), but
  `device/state` still consumes `CdoRaw` directly.
- The stage then tags at that state (or reopens for a reshape of
  `DeviceOp`, at the user's discretion).
- 8b's coupling cleanup is still a net win without `DeviceOp`; the
  boundary-naming payoff remains even if the boundary-move payoff
  is delayed.

For 8a: if the audit concludes no `BinaryLoader` trait
is warranted, 8a lands data migrations + audit doc + design note
only. The trait question is "decide and document," not
"implement regardless."

And for 8c: each of the three sub-deliverables (diagnostics,
fixtures, control-packet alignment) is independent. If one of
them is more work than expected, it slips to a Phase 2 hygiene
item and the stage still tags with the other two.

## Alternatives rejected

### Approach A -- neutral parser + internal split only

Parser stays syntactic. `device/state` keeps interpreting. 8b is
just a cosmetic file split + ELF dedup. Pain point (ii) -- fuzzy
CDO/device-state boundary -- stays fuzzy.

**Rejected** because the user's "deep and global" directive and
the seven enumerated pain points (especially ii) push past this.
Approach A is where we'd land if the Subsystem 6/7 "no trait, data
in archspec" pattern were the whole playbook; the user explicitly
opened scope to structural parser work beyond arch-seam-completion.

### Approach B -- parser-owns-translation (no syntax/semantics split)

Parser consults archspec during parsing and emits `DeviceOp` directly
with no intermediate `CdoRaw` layer. One layer, not two.

**Rejected** for testability. Without the `CdoRaw` intermediate,
testing CDO byte-level parse correctness requires either (a) a full
`ArchHandle` mock, or (b) asserting on `DeviceOp` output (which
conflates parse bugs and semantics bugs). The two-layer split gives
us two small, focused test surfaces for two small, focused concerns.
The cost -- one extra enum (`CdoRaw`) -- is low; the testability
payoff is high.

Note the user's reframing: "the parser doesn't need to be arch-blind"
means arch-awareness in `semantics.rs` is fine, not that we should
collapse the layering. We're keeping the split for *design quality*,
not arch-purity.

### Five-way decomposition (8a...8e)

One stage per pain point, maximum bisect granularity.

**Rejected.** Each pain point is ~hundreds of LOC; five tags over
work this cohesive is ceremony. Three stages (audit / structural /
ergonomic) group the work by nature-of-change, which matches the
mental model a bisector would use.

### Land as a single tag (Subsystem 7 pattern)

No intermediate tags; one `phase1-subsys-parser` at the end.

**Rejected.** Subsystem 7 had one large migration (vmac_routing.rs
move) plus small data migrations -- cohesive enough. Subsystem 8
has three genuinely independent workstreams (arch / coupling /
ergonomic). Intermediate tags pay off for bisect and for "we
shipped this part, it's done, stop touching it."

## Precedent links

- Parent: [2026-04-16-device-family-refactor-design.md](2026-04-16-device-family-refactor-design.md)
- Most recent subsystem completion: [../../arch/subsys7-audit.md](../../arch/subsys7-audit.md)
- ISA Execute design note (format reference): [../../arch/isa-execute-model.md](../../arch/isa-execute-model.md)
- ISA Decode design note (no-trait precedent): [../../arch/isa-decode.md](../../arch/isa-decode.md)
- Phase 1a foundation: [../../arch/phase1a-audit.md](../../arch/phase1a-audit.md)
- Previous subsystem audits: [../../arch/subsys{1..7}-audit.md](../../arch/)

## After approval

Next step after user approves this spec: invoke
`superpowers:writing-plans` to produce
`docs/superpowers/plans/2026-04-23-subsys8-parser-plan.md` with
per-task breakdown for each stage.

The plan's 8a section ends at the "Gate between 8a and 8b" described
above -- the plan for 8b presents `DeviceOp` variants conditionally on
audit output and treats the user's gate confirmation as a hard
prerequisite for starting 8b work.
