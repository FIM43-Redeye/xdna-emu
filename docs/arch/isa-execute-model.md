# ISA Execute Model -- Design Note

**Subsystem:** 7 (Phase 1b)
**Tag:** `phase1-subsys-isa-execute`
**Spec:** [../archive/specs/2026-04-21-subsys7-isa-execute-design.md](../archive/specs/2026-04-21-subsys7-isa-execute-design.md)
**Audit:** [subsys7-audit.md](../archive/audits/subsys7-audit.md) (archived; refactor shipped)

This document is the mandatory per-seam design note required by the
parent device-family refactor. It explains why the `IsaExecutor`
trait ships empty, what data lives where after Subsystem 7's
migrations, and what an AIE1 port would look like.

---

## What lives where

After Subsystem 7 closes, the ISA execute layer is split cleanly
between archspec (all arch-specific *data*) and xdna-emu (all
execution *algorithms*). The six per-area audit sections map into
these destinations:

| Data / code | Module | Source of truth |
|-------------|--------|-----------------|
| `IsaExecutor` trait (empty; reserved for future seams) | `xdna_archspec::isa_execute` | Anchor for `ArchConfig::isa_executor()` dispatch |
| `Aie2IsaExecutor` ZST + `AIE2_ISA_EXECUTOR` singleton | `xdna_archspec::aie2::isa_execute_model` | Concrete trait impl (empty body) |
| `RoundingMode` enum + `from_raw` conversion | `xdna_archspec::aie2::rounding` | Dedup'd from `vector_srs.rs` + `vector_float.rs` |
| `ShuffleMode` enum + `SHUFFLE_ROUTING` table (48 x 64 bytes, hardware-probed) | `xdna_archspec::aie2::permute` | Moved wholesale from `vector_permute.rs` |
| `MacPermuteMode` enum + `MacPermuteConfig` + `mac_permute_config()` lookup | `xdna_archspec::aie2::permute` | Moved wholesale from `vector_permute.rs` |
| VMAC crossbar data + `eval_prmx`/`eval_prmy` entry points | `xdna_archspec::aie2::vmac` | Moved wholesale from `vmac_routing.rs` (234K) |
| Matmul geometry tables (`DENSE_GEOMETRY_TABLE`, `SPARSE_GEOMETRY_TABLE`, `CONFIG_GEOMETRY_TABLE`) + `GeometryEntry` | `xdna_archspec::aie2::matmul` | Moved from `vector_config.rs` |
| UPS mode lookup table + `UpsScale`/`UpsAccMode`/`UpsMode` types | `xdna_archspec::aie2::ups` | Moved from `vector_ups.rs::ups_mode()` |
| Cascade data (`CASCADE_WORDS = 6`) + `has_cascade_link: bool` feature flag | `xdna_archspec::aie2::{mod,processor}` + `ProcessorModel` field | New field; cascade gate added to `cascade.rs` |
| Pipeline latency constants (7 items: `SCALAR_MUL`, `SCALAR_DIV`, `VECTOR_MUL`, `VECTOR_MAC`, `VECTOR_SIMPLE`, `VECTOR_SHUFFLE`, `VECTOR_PACK`) | `xdna_archspec::aie2::instruction_latency` | Moved from `timing/latency.rs` raw literals |
| Control-register IDs (`crSat`, `crRnd`, `crSRSSign`, q-regs) | `xdna_archspec::aie2::{aiert,processor}` | Read via accessor in `semantic.rs` |
| Lock quadrant boundaries (South/West/North/East + internal) | `xdna_archspec::aie2::locks::quadrants` | Read via accessor in `control.rs` |
| `PROC_BUS_BASE`, `PROC_BUS_END` | `xdna_archspec::aie2::compute` (derived from `MEMORY_SIZE`) | Read via accessor in `memory/mod.rs` |
| `BRANCH_DELAY_SLOTS`, `LatencyTable::aie2()` cached instance | `xdna_archspec::aie2::processor` + runtime cache in `arch_handle::latency_table()` | Consumed by `cycle_accurate.rs` + `timing_context.rs` |
| **Execute algorithms** (`execute_semantic`, `VectorAlu::execute`, `MemoryUnit::execute`, `ControlUnit::execute`, `StreamOps::execute`, `CascadeOps::execute`, `shuffle_vectors`, `rc2i`, `sparse_vmac`, etc.) | `src/interpreter/execute/*.rs` | Stay in xdna-emu; arch-generic or parameterized by archspec data |

The split follows one rule: data + data-descriptors live in archspec;
algorithms that consume them live in xdna-emu. No execute algorithm
became per-arch; no archspec module holds mutable state.

## Trait surface

`IsaExecutor` is empty. Full definition from
`crates/xdna-archspec/src/isa_execute/mod.rs`:

```rust
pub trait IsaExecutor: Send + Sync + core::fmt::Debug {
    // Intentionally empty per audit. See module docs.
}
```

The trait is *anchored* (not deleted) to give `ArchConfig::isa_executor()`
a stable return type. Future seams -- if a second-arch landing surfaces
algorithmic divergence that data alone cannot express -- attach methods
here without touching every consumer of `arch_handle::isa_executor()`.

The anchor costs one vtable per arch implementation (AIE2 today; AIE1
and AIE2P later) and zero dispatch overhead for the current empty
trait. Removing the anchor would save ~40 LOC and make re-introducing
the trait later a cross-subsystem plumbing change; we judged the cost
too high given the refactor's "no second-arch implementation during
the refactor" ground rule and the long-term likelihood that *some*
execute-layer seam eventually appears.

## The shape-vs-values rule, applied to ISA execute

Subsystem 6 (ISA Decode) established the rule:

> A type belongs in archspec iff it is derivable from the toolchain
> (TableGen + LLVM + aie-rt) without reference to emulator execution
> state. Types that name `Operand`, register-file indices, or emulator
> convention stay in xdna-emu.

Subsystem 7 sharpened this with a second axis: among arch-specific
content, *data* (tables, enums, constants, feature flags) lives in
archspec while *algorithms* stay in xdna-emu. The audit examined four
candidate trait methods and rejected each by this rule:

- **`vmac_route(mbit, pmode)`**: the hardware-probed routing tables
  are data (789 m-bits, 15808 routes); the `eval_prmx`/`eval_prmy`
  algorithm is shift-and-mask logic that works against whatever
  table the archspec provides. AIE1 would have its own table, same
  algorithm. Move the table, keep the algorithm.
- **`srs_round(mode, val)`**: rounding algorithms (sgn/lsb/grd/stk
  signals per the `RoundingMode` enum) are standard logic at the
  bit level; only the `RoundingMode` enum variant set differs per
  arch. Move the enum, keep the algorithm.
- **`cascade_width()`**: AIE1 has no cascade at all -- that's a
  *feature flag* (`has_cascade_link: bool`), not a shape. The
  arch-generic cascade algorithms live behind the flag. AIE2 sets
  it true and gets cascade; AIE1 sets it false and cascade
  operations become no-ops with no algorithm changes.
- **`memory_quadrant(addr)`**: address decoding is identical; only
  the `MEMORY_SIZE` constant differs (already archspec-resident for
  the memory-map subsystem). Move the constant; no trait needed.

All four candidates failed the shape-vs-values test with data.
`IsaExecutor` therefore ships empty. This is Approach A from the
spec.

## What would AIE1 look like?

An AIE1 port of the execute subsystem requires only archspec additions
-- no changes to `src/interpreter/execute/*.rs`, `src/interpreter/
timing/*.rs`, or `src/device/arch_handle.rs`.

**New archspec data:**
- `xdna_archspec::aie1::permute::{ShuffleMode, SHUFFLE_ROUTING}` with
  AIE1's narrower vector shuffle table (AIE1's vector unit is 128-bit
  vs AIE2's 256/512-bit; the table shape changes correspondingly)
- `xdna_archspec::aie1::permute::{MacPermuteMode, mac_permute_config}`
  for AIE1's different MAC permute mode set (fewer modes)
- `xdna_archspec::aie1::vmac` OR a new AIE1-specific matmul module:
  AIE1 uses a fundamentally different multiply pipeline (no X-crossbar
  in the AIE2 sense), so the AIE1 VMAC path is a new code path, not
  a variant of AIE2's. This is the only "new code" the port requires
  -- and it's new arch-specific data + a new arch-specific algorithm
  file, not a rewrite of existing AIE2 code.
- `xdna_archspec::aie1::rounding::RoundingMode` if AIE1's rounding
  mode set differs (audit suggests it does; specific deltas not
  investigated in this refactor's scope)
- `xdna_archspec::aie1::matmul`, `aie1::ups`, `aie1::processor`
  (for `has_cascade_link: false`, smaller accumulator widths, etc.)

**Trait anchor:**
- `Aie1IsaExecutor` ZST + `AIE1_ISA_EXECUTOR` singleton in
  `xdna_archspec::aie1::isa_execute_model`, mirroring AIE2's empty
  impl. `ArchConfig::isa_executor()` gets an `Aie =>
  &AIE1_ISA_EXECUTOR` arm replacing the current `unimplemented!`.

**No changes in xdna-emu:** `CycleAccurateExecutor::execute()`,
`semantic::execute_semantic()`, `VectorAlu::execute()`,
`MemoryUnit::execute()`, `CascadeOps::execute()`, `StreamOps::execute()`,
`ControlUnit::execute()`, and every vector_*.rs pipeline file work
unchanged. They read per-arch data via `arch_handle::*` accessors;
the accessors route based on `default_arch()`'s architecture variant.

The only xdna-emu change needed for AIE1 support is in
`src/device/arch_handle.rs` -- its accessors currently hardcode the
AIE2 model (e.g., `LATENCY_TABLE.get_or_init(LatencyTable::aie2)`).
That hardcoding becomes arch-dispatched (`match default_arch() {
Aie => LatencyTable::aie1(), Aie2 | Aie2p => LatencyTable::aie2() }`).
This is a single-file change; it isn't specific to ISA execute, so
the refactor leaves it for when AIE1 is actually populated.

## Alternatives rejected

### Approach A (landed, in Subsystem 7) -- data in archspec, no trait

Data-migrate everything arch-specific to archspec. Execute algorithms
stay in xdna-emu. `IsaExecutor` ships as an empty anchor trait.

**Accepted.** The audit's four-candidate rejection table showed no
trait method warranted. Mirrors the ISA Decode landing (Subsystem 6
also shipped with no trait seam).

Subsystem 7 chose to keep the `IsaExecutor` trait as an empty anchor
(per the trait-surface section above) rather than delete it, to
preserve the dispatch pathway for future seams without a cross-
subsystem plumbing change.

### Approach C -- full IsaExecutor trait covering the dispatcher layer

Move the dispatcher layer (`execute_semantic`, `VectorAlu::execute`,
`MemoryUnit::execute`, etc.) behind trait methods. Per-arch submodule
owns the entire execute pipeline.

**Rejected.** Three reasons:

1. Most dispatch is already `SemanticOp`-driven (arch-generic via
   archspec `isa` data from Subsystem 6). A trait layer on top of
   already-arch-generic code adds ceremony without real divergence.
2. Duplication risk: the AIE2 impl would re-export large amounts of
   current xdna-emu execute code; the AIE1 impl (when eventually
   built) would share most of it; the trait would be a pass-through
   for most methods.
3. Prior subsystems (DMA, Locks, StreamSwitch) all landed at "tight
   trait for the seams that actually vary" -- and we verified via
   audit that *no* seams actually vary here at the algorithm level.

### Pre-audit trait commitment

Commit upfront to a specific trait method list (`apply_srs`,
`vmac_route`, `accumulator_promotion_rule`), populate AIE2, migrate
sites, ship. Skip the audit artifact.

**Rejected.** The spec explicitly ruled this out, citing Subsystem 5
where the presumed-necessary `PortLayout` extension trait turned out
to be 231 LOC of dead code. Audit-first is cheap insurance for the
largest of the eight subsystems. The audit's conclusion (Approach A,
zero methods) validates the insurance -- pre-audit commitment would
have shipped a trait that did nothing useful.

### Sub-subsystem decomposition (7a/7b/7c...)

Split Subsystem 7 into 7a = scalar/control, 7b = vector ALU, 7c =
VMAC, etc. Each gets its own tag.

**Rejected.** The `IsaExecutor` trait was hypothesized to be
cohesive; splitting would have forced 7a to predict what 7b-7e
needed from the trait. Landing as one subsystem with multiple tasks
handled the scale without the sub-sub coordination cost. In
retrospect, with the audit landing Approach A (zero methods), the
decomposition argument is even weaker -- there's no shared trait to
coordinate across sub-subsystems anyway.
