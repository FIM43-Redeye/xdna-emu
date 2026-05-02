# Subsystem 5 -- Stream Switch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a `StreamSwitchModel` trait seam in `xdna-archspec` with an `Aie2StreamSwitchModel` concrete impl, add a `StreamSwitchTopology` data carrier (three `TileStreamPorts` sub-structs) populated from the existing archspec-generated port arrays and range constants, thread `&'static dyn StreamSwitchModel` through `ArchConfig`, migrate the 6 tile-construction call sites in `src/device/stream_switch/mod.rs` through a new `arch_handle::stream_switch_topology()` accessor, delete the dead-code `PortLayout` extension trait (231 LOC), migrate its 3 tests to archspec, and land a drift-detection test locking `AIE2_STREAM_SWITCH_TOPOLOGY` to the generated per-field constants.

**Architecture:** Pure seam + dead-code collapse, no hot-path changes. Trait + carrier + AIE2 impl land first (Task 2), then the accessor on `ArchConfig` (Task 3), then the 6 tile-construction call-site rewrites + new `arch_handle` accessor (Task 4), then the `PortLayout` collapse + test migration + stale doc-comment cleanup (Task 5), then drift-detection + shape tests (Task 6), then the gate + tag (Task 7). Consumers reach topology data via `arch_handle::stream_switch_topology().for_tile(kind).<field>`.

**Tech Stack:** Rust 2021 workspace, `xdna-archspec` workspace crate, `&'static dyn StreamSwitchModel` trait-object dispatch, AM025 register DB JSON loaded from `mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json`, aie-rt evidence from `xaiegbl_reginit.c` and `xaiemlgbl_reginit.c`.

**Spec:** [docs/superpowers/specs/2026-04-21-subsys5-stream-switch-design.md](../specs/2026-04-21-subsys5-stream-switch-design.md)

**Parent refactor:** [docs/superpowers/specs/2026-04-16-device-family-refactor-design.md](../specs/2026-04-16-device-family-refactor-design.md)

**Prior subsystem:** `phase1-subsys-locks` (Subsystem 4, 2026-04-21).

---

> **Sweep-as-of 2026-05-01:** Subsystem 5 completed -- tag `phase1-subsys-stream-switch`. Stream switch port topology + routing tables consolidated. Steps below were executed organically rather than ticked one-by-one; this sweep flips the checkboxes to match the verified completion state.


## Scope Note

Single-part subsystem with one tag (`phase1-subsys-stream-switch`) at the end. Scope is intentionally small because `PortLayout` has zero external consumers (verified during spec self-review) -- migration concentrates on the one real consumer path (tile construction in `stream_switch/mod.rs`), with other direct archspec-constant consumers staying on direct access as AIE1-landing follow-ups.

- **New in archspec:** `StreamSwitchModel` trait + `StreamSwitchTopology` carrier + `TileStreamPorts` sub-struct + `Aie2StreamSwitchModel` impl + `AIE2_STREAM_SWITCH_MODEL` + `AIE2_STREAM_SWITCH_TOPOLOGY` + `ArchConfig::stream_switch_model()` accessor + ~60 LOC of tests (drift-detection + shape tests + 3 migrated tests). Total: ~180 LOC.
- **Deleted in xdna-emu:** entire `src/device/port_layout.rs` (~231 LOC including tests).
- **Modified in xdna-emu:** 6 call sites in `src/device/stream_switch/mod.rs` (3 tile-construction functions x 2 port directions), new ~15-LOC `arch_handle::stream_switch_topology()` accessor, 1 `pub mod port_layout;` line removed from `src/device/mod.rs`, 3 stale doc-comment references updated in `crates/xdna-archspec/src/runtime.rs`.
- Estimated file-count: ~8 files touched, ~7 commits.
- **No Part A / Part B split** expected. If file count exceeds 12 or commit count exceeds 10, pause and flag.

Branch: `dev`. Tag at end: `phase1-subsys-stream-switch`.

---

## Global Invariants (every task, every commit)

- `cargo test --lib` green. Baseline at `phase1-subsys-locks`: `2687 passed; 0 failed; 5 ignored`.
- `cargo test -p xdna-archspec --lib` green. Baseline at `phase1-subsys-locks`: `282 passed; 0 failed; 2 ignored`.
- `cargo build` green. `cargo build --release` clean is required before the tag, not every commit.
- `./scripts/emu-bridge-test.sh --no-hw -v add_one_cpp_aiecc` green after rebuilding the FFI cdylib (`cargo build -p xdna-emu-ffi`).
- No commit introduces `TODO` / `FIXME` / `unimplemented!()` without an open-issue reference (the `unimplemented!("AIE1 StreamSwitchModel ...")` in the accessor is the one sanctioned exception, mirroring `dma_model()` and `lock_model()`).
- Commit messages: lowercase type prefix (`refactor:`, `docs:`, `test:`, `build:`, `refactor(archspec):`); no emoji; ends with `Generated using Claude Code.`.
- All work on `dev`. No merges to `master` during this plan.
- **Every `cargo` call** must have `PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH` prepended (tblgen needs llvm-config 21.x, not mlir-aie's 23.x).

---

## File Structure

**Current layout (post-Subsystem 4, at `phase1-subsys-locks`):**

```
xdna-emu/
├── src/
│   ├── device/
│   │   ├── port_layout.rs          # 231 LOC: PortLayout extension trait on
│   │   │                           #          ModelConfig (6 methods) + impl +
│   │   │                           #          3 tests. ZERO external consumers.
│   │   ├── arch_handle.rs          # lock_value_layout() accessor
│   │   ├── stream_switch/
│   │   │   └── mod.rs              # 834 LOC: StreamSwitch per-tile state.
│   │   │                           #          Lines 131-133, 163-165, 193-195
│   │   │                           #          = 6 build_ports_from_spec calls
│   │   │                           #          reading xdna_archspec::aie2::*.
│   │   └── mod.rs                  # line 52: pub mod port_layout;
│   └── ...
└── crates/xdna-archspec/
    └── src/
        ├── lib.rs                  # (will gain `pub mod stream_switch;`)
        ├── aie2/
        │   ├── mod.rs              # already has pub mod stream_switch (gen ranges)
        │   │                       # will gain pub mod stream_switch_model
        │   └── stream_switch.rs    # 7 LOC: include!(gen_stream_ranges.rs)
        ├── runtime.rs              # ArchConfig trait + ModelConfig impl
        │                           #   dma_model(), lock_model() already here
        │                           #   stream_switch_model() joins
        └── ...
```

**Target layout (post-Subsystem 5):**

```
xdna-emu/
├── src/
│   ├── device/
│   │   ├── port_layout.rs          # DELETED
│   │   ├── arch_handle.rs          # + stream_switch_topology() accessor
│   │   ├── stream_switch/
│   │   │   └── mod.rs              # 6 call sites migrated to
│   │   │                           # arch_handle::stream_switch_topology()
│   │   └── mod.rs                  # line 52: `pub mod port_layout;` removed
│   └── ...
└── crates/xdna-archspec/
    └── src/
        ├── lib.rs                  # + pub mod stream_switch;
        ├── stream_switch/
        │   └── mod.rs              # NEW: StreamSwitchModel trait +
        │                           #      StreamSwitchTopology + TileStreamPorts
        │                           #      + tests (migrated from port_layout.rs)
        ├── aie2/
        │   ├── mod.rs              # + pub mod stream_switch_model;
        │   └── stream_switch_model.rs  # NEW: Aie2StreamSwitchModel +
        │                               #      AIE2_STREAM_SWITCH_MODEL +
        │                               #      AIE2_STREAM_SWITCH_TOPOLOGY +
        │                               #      drift + shape tests
        └── runtime.rs              # + stream_switch_model() method
                                    # + 3 stale PortLayout doc-comment fixes
```

Archspec new LOC: ~180 (trait + carrier + AIE2 impl + singletons + tests). Archspec migrated LOC: ~20 (3 port-layout tests migrate). xdna-emu shrinks by ~231 LOC (entire `port_layout.rs` deletes). Net workspace LOC: ~-50 (small net decrease; the new trait/tests are lighter than the deleted `port_layout.rs` because tests were a chunk of it).

---

## Baseline to Preserve

Before Task 1, capture current numbers so later regression checks have a target:

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
# Expected: test result: ok. 2687 passed; 0 failed; 5 ignored; 0 measured

PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
# Expected: test result: ok. 282 passed; 0 failed; 2 ignored; 0 measured
```

Known pre-existing failure (carries through, documented in `NEXT-STEPS.md`):
- `bd_chain_repeat_on_memtile` EMU deadlock in bridge. Not blocking.

---

## Task 1: Audit scaffold and design note

**Files:**
- Create: `docs/arch/subsys5-audit.md`
- Create: `docs/arch/stream-switch-model.md`

**Goal:** Capture the current-state audit and stub the design note so later tasks can fill them in as evidence accumulates. Mirrors `docs/arch/subsys4-audit.md` and `docs/arch/lock-model.md` from Subsystem 4.

- [x] **Step 1: Create `docs/arch/subsys5-audit.md` scaffold**

```markdown
# Subsystem 5 -- Stream Switch Audit

## Baseline (pre-subsystem, at phase1-subsys-locks tag)

- `cargo test --lib`: test result: ok. 2687 passed; 0 failed; 5 ignored; 0 measured; 0 filtered out; finished in ~4s
- `cargo test -p xdna-archspec --lib`: test result: ok. 282 passed; 0 failed; 2 ignored; 0 measured; 0 filtered out; finished in ~1s

Known pre-existing failures (carry through):
- `bd_chain_repeat_on_memtile` EMU deadlock (bridge suite; see NEXT-STEPS.md).

## Audit facts

### Dead-code `PortLayout` in xdna-emu

From `src/device/port_layout.rs`:
- `PortLayout` extension trait on `ModelConfig` (6 methods: `master_ports`,
  `slave_ports`, `north_master_range`, `south_master_range`,
  `north_slave_range`, `south_slave_range`).
- 3 tests (`test_npu1_port_layouts`, `test_npu1_port_ranges`,
  `test_shim_pl_same_as_shim_noc`).
- Zero external consumers (verified: `grep PortLayout /home/triple/npu-work/xdna-emu/src/`
  shows only the file itself).

### Call-site inventory (xdna-emu)

**In scope for Subsystem 5 migration (6 sites):**
- `src/device/stream_switch/mod.rs:132` -- `build_ports_from_spec(xdna_archspec::aie2::COMPUTE_MASTER_PORTS, ...)`
- `src/device/stream_switch/mod.rs:133` -- `build_ports_from_spec(xdna_archspec::aie2::COMPUTE_SLAVE_PORTS, ...)`
- `src/device/stream_switch/mod.rs:164` -- `build_ports_from_spec(xdna_archspec::aie2::MEMTILE_MASTER_PORTS, ...)`
- `src/device/stream_switch/mod.rs:165` -- `build_ports_from_spec(xdna_archspec::aie2::MEMTILE_SLAVE_PORTS, ...)`
- `src/device/stream_switch/mod.rs:194` -- `build_ports_from_spec(xdna_archspec::aie2::SHIM_MASTER_PORTS, ...)`
- `src/device/stream_switch/mod.rs:195` -- `build_ports_from_spec(xdna_archspec::aie2::SHIM_SLAVE_PORTS, ...)`

**Out of scope (flagged follow-ups for AIE1-landing pass):**
- `src/device/dma/stream_io.rs` (~6 sites): `pub const` declarations.
- `src/device/array/routing.rs` (~15 sites): range-math sites.
- `src/device/state/{compute,memtile}.rs` (2 sites each): ENABLE_BIT / SLAVE_SELECT_MASK.
- `src/interpreter/test_runner.rs`: test-only imports.

### Stale doc-comments in archspec runtime.rs

Three references to the `PortLayout` extension trait that describe its
pre-Subsystem-1 rationale (data must stay runtime-side). Update in Task 5:
- `crates/xdna-archspec/src/runtime.rs:13`
- `crates/xdna-archspec/src/runtime.rs:63`
- `crates/xdna-archspec/src/runtime.rs:221`

### aie-rt AIE1 vs AIE2 stream-switch divergence (evidence base for trait)

Confirmed via `../aie-rt/driver/src/stream_switch/`:

| # | Behavior | AIE1 | AIE2 | Source |
|---|---|---|---|---|
| 1 | Deterministic merge | **Unavailable** (`DetMergeFeature = XAIE_FEATURE_UNAVAILABLE`) | Available on all tile types (2 arbiters, 4 positions each) | `xaiegbl_reginit.c` vs `xaiemlgbl_reginit.c` |
| 2 | Packet routing | Full support (slots/arbiter/msel) | Full support, identical mechanisms | `xaie_ss.c` (shared, no arch-dispatch) |
| 3 | Port-count deltas | Compute: 2 CORE / 2 DMA / 1 CTRL / 2 FIFO | Compute: 1 CORE / 2 DMA / 1 CTRL / 1 FIFO | `AieTileStrmMstr/Slv` vs `AieMlTileStrmMstr/Slv` |
| 4 | MemTile existence | Absent | Present | Arch-level (TileKind gating), not SS-specific |
| 5 | Port validity | Full crossbar | Restrictive per tile-type | `_XAieMl_*_StrmSwCheckPortValidity`. Not emulated. |

Only row 1 (deterministic merge) is a trait-level behavioral flag today.
Rows 2 (invariant), 3 (topology data per arch), 4 (TileKind-level), 5
(not emulated) are not direct trait concerns for Subsystem 5.

## Completion

(Filled in at end of Subsystem 5.)
```

- [x] **Step 2: Create `docs/arch/stream-switch-model.md` scaffold**

```markdown
# Stream Switch Model -- Design Note

**Subsystem:** 5 (Phase 1b)
**Tag:** `phase1-subsys-stream-switch`
**Spec:** [../superpowers/specs/2026-04-21-subsys5-stream-switch-design.md](../superpowers/specs/2026-04-21-subsys5-stream-switch-design.md)

This document is the mandatory per-seam design note required by the
parent device-family refactor. It explains the shape difference that
justifies the `StreamSwitchModel` trait and what AIE1 / AIE2P impls
will look like.

---

## What lives where

(Filled in after Task 2 lands.)

---

## The shape-vs-values principle, applied to stream switch

Subsystem 3 (DMA) introduced a 9-method trait because DMA has many
feature differences. Subsystem 4 (Locks) introduced a 3-method
trait because only lock value width and two AIE2-only features
diverge. Subsystem 5 (Stream Switch) introduces a **2-method**
trait because the aie-rt audit surfaced only one genuine behavioral
divergence (deterministic merge), and all topology data groups
into a single carrier struct. The narrower trait surface reflects
the narrower variance.

Concretely:
- **AIE2 (NPU1/NPU4/NPU5/NPU6):** deterministic merge available on
  all tile types; compute = 23 master / 25 slave ports; memtile =
  17 / 18; shim = 22 / 23.
- **AIE1 (Versal, e.g., xcvc1902):** deterministic merge unavailable;
  different compute / shim port counts per `xaiegbl_reginit.c`; no
  MemTile (handled at TileKind level).
- **AIE2P:** expected to match AIE2 1:1.

---

## The trait surface

(Filled in after Task 2 lands.)

---

## What would AIE1 look like?

- `xdna_archspec::aie1::stream_switch_model::Aie1StreamSwitchModel`
  ZST + `AIE1_STREAM_SWITCH_MODEL` static.
- `supports_deterministic_merge()` returns `false` (aie-rt:
  `DetMergeFeature = XAIE_FEATURE_UNAVAILABLE`).
- `AIE1_STREAM_SWITCH_TOPOLOGY` populated from AIE1 port-count data
  (different compute / shim counts per `xaiegbl_reginit.c`
  `AieTileStrmMstr/Slv`).
- `memtile` field on the carrier is the AIE1 ambiguity: either
  (a) `Option<TileStreamPorts>` (all memtile call sites gate), or
  (b) sentinel (empty arrays + zero ranges) so `for_tile(Mem)`
  returns benign data that AIE1 call sites never hit since
  `TileKind::Mem` is never produced. **Decision deferred.**
- `ArchConfig::stream_switch_model()` adds a `Architecture::Aie`
  arm returning `&AIE1_STREAM_SWITCH_MODEL`.

---

## Why not a flat 8-method trait (rejected Approach B in spec)

A flat trait with `master_ports(TileKind) -> &'static [u8]` +
`north_master_range(TileKind) -> (u8, u8)` etc. would put all six
port-layout methods directly on the trait, no carrier. Rejected
because the carrier gives three properties the flat trait doesn't:
(1) `&'static StreamSwitchTopology` is a narrower type than
`&dyn StreamSwitchModel` -- functions that only need port data
can take the narrower type and avoid exposing behavioral flags;
(2) shared topology across arches (AIE2 vs AIE2P) reuses the same
static; (3) `arch_handle::stream_switch_topology()` caches pure data
with no vtable. Monomorphization is equivalent between both
approaches.

## Why not a behavioral-only trait with PortLayout as extension (rejected Approach C in spec)

Rejected because `PortLayout`'s doc-comment claim ("data must stay
runtime-side because archspec doesn't run build.rs") has been
stale since Subsystem 1 moved codegen into archspec's own
build.rs. Two entry points for one concept (topology on the
extension trait, behavior on the model) violates the "one
authoritative source per concept" principle.

## Why not preemptive `supports_packet_routing`

aie-rt audit confirmed packet routing is invariant across AIE1/AIE2
(same slot/arbiter/msel mechanisms in shared `xaie_ss.c` with no
runtime arch-dispatch). Including a flag whose only valid value is
`true` is ceremony.

---

## Completion

(Filled in at end of Subsystem 5.)
```

- [x] **Step 3: Commit scaffolds**

```bash
git add docs/arch/subsys5-audit.md docs/arch/stream-switch-model.md
git commit -m "$(cat <<'EOF'
docs: Subsystem 5 audit + stream-switch-model design note scaffolds

Audit captures baseline test counts, call-site inventory (6 in-scope
migration sites in stream_switch/mod.rs, dead-code PortLayout with
zero external consumers, flagged out-of-scope direct-constant consumers
for AIE1-landing pass), and the aie-rt AIE1-vs-AIE2 divergence matrix.

Design note scaffolds the shape-vs-values rationale, the trait-surface
section (filled after Task 2), and the rejected-approach summaries.

Generated using Claude Code.
EOF
)"
```

Expected: clean commit, `cargo test --lib` and `cargo test -p xdna-archspec --lib` still green (no code change).

---

## Task 2: StreamSwitchModel trait + StreamSwitchTopology carrier + Aie2StreamSwitchModel

**Files:**
- Create: `crates/xdna-archspec/src/stream_switch/mod.rs`
- Create: `crates/xdna-archspec/src/aie2/stream_switch_model.rs`
- Modify: `crates/xdna-archspec/src/lib.rs` (+1 line: `pub mod stream_switch;`)
- Modify: `crates/xdna-archspec/src/aie2/mod.rs` (+1 line: `pub mod stream_switch_model;`)

**Goal:** Establish the trait, carrier, AIE2 impl, and singleton statics. This is the heart of the seam; Task 3 wires it to `ArchConfig`, Task 4 routes call sites through it.

- [x] **Step 1: Create `crates/xdna-archspec/src/stream_switch/mod.rs`**

```rust
//! Stream switch model trait and data carrier.
//!
//! The AIE2 family (NPU1/NPU4/NPU5/NPU6) shares a single
//! stream-switch feature set. AIE1 (Versal) diverges on exactly one
//! flag (deterministic merge: absent on AIE1, present on AIE2) and
//! on topology data (port counts per tile kind). This module
//! exposes:
//!
//! - `StreamSwitchModel` (2-method trait): the one behavioral flag
//!   + one topology accessor.
//! - `StreamSwitchTopology` (data carrier): three `TileStreamPorts`
//!   sub-structs (compute, memtile, shim) + a `for_tile` accessor.
//! - `TileStreamPorts`: per-tile-kind master/slave port arrays and
//!   N/S range data.
//!
//! See `docs/arch/stream-switch-model.md` for the per-seam design
//! note explaining the shape-vs-values principle and alternatives.

use crate::types::TileKind;

/// Per-tile-kind stream-switch port layout.
///
/// Ports are u8 ordinals encoded per
/// `xdna_archspec::aie2::port_type::{CORE, FIFO, CTRL, north(n),
/// south(n), east(n), west(n), dma(n)}`. The underlying port arrays
/// are generated by build.rs from AM025 `Stream_Switch_*_Config`
/// register names.
///
/// Range fields are `(start, end)` inclusive. Shim tiles have
/// `(0, 0)` sentinels for `south_master` and `south_slave` because
/// shim's south-side ports are the external-NoC interface rather
/// than intra-array neighbors -- the `(0, 0)` sentinel models
/// "no intra-array south-facing ports," preserving the semantic
/// carried by the (now-deleted) runtime-side `PortLayout`
/// extension trait.
#[derive(Debug, Clone, Copy)]
pub struct TileStreamPorts {
    /// Ordered port-type ordinals for stream-switch master ports.
    pub master_ports: &'static [u8],
    /// Ordered port-type ordinals for stream-switch slave ports.
    pub slave_ports: &'static [u8],
    /// North-facing master port range `(start, end)` inclusive.
    pub north_master: (u8, u8),
    /// South-facing master port range `(start, end)` inclusive.
    /// `(0, 0)` sentinel if tile kind has no intra-array south.
    pub south_master: (u8, u8),
    /// North-facing slave port range `(start, end)` inclusive.
    pub north_slave: (u8, u8),
    /// South-facing slave port range `(start, end)` inclusive.
    /// `(0, 0)` sentinel if tile kind has no intra-array south.
    pub south_slave: (u8, u8),
}

/// Stream-switch topology data for a given architecture.
///
/// Aggregates per-tile-kind `TileStreamPorts` values. Use the
/// `for_tile` accessor to dispatch to the right sub-struct; the
/// public fields are also directly accessible for compile-time
/// destructuring (e.g., `topo.compute.master_ports`).
#[derive(Debug, Clone, Copy)]
pub struct StreamSwitchTopology {
    pub compute: TileStreamPorts,
    pub memtile: TileStreamPorts,
    pub shim: TileStreamPorts,
}

impl StreamSwitchTopology {
    /// Select the per-tile-kind port layout.
    ///
    /// `ShimNoc` and `ShimPl` both route to the shim fields (both
    /// are physical shim tiles; `PL` vs `NoC` is an interface
    /// variant, not a stream-switch variant).
    pub fn for_tile(&self, tile: TileKind) -> &TileStreamPorts {
        match tile {
            TileKind::Compute => &self.compute,
            TileKind::Mem => &self.memtile,
            TileKind::ShimNoc | TileKind::ShimPl => &self.shim,
        }
    }
}

/// Stream-switch model trait.
///
/// Two methods:
/// - `supports_deterministic_merge`: AIE1 = false, AIE2/AIE2P = true.
///   Per aie-rt `xaiegbl_reginit.c` / `xaiemlgbl_reginit.c`
///   `DetMergeFeature`.
/// - `topology`: returns the architecture's
///   `StreamSwitchTopology` as a `&'static` reference suitable for
///   caching via `OnceLock` on the runtime side.
///
/// Not on the trait:
/// - `supports_packet_routing`: invariant `true` across all AIE
///   generations per aie-rt shared `xaie_ss.c` (no arch-dispatch).
/// - Port validity checking (per-tile-kind which-slave-can-reach-which-master
///   rules on AIE2): programming-time concern, not emulated today.
/// - Packet slot count / arbiter count: invariant across arches
///   (4 / 8 respectively). Constants live on the carrier only if
///   they later diverge.
pub trait StreamSwitchModel: Send + Sync + core::fmt::Debug {
    fn supports_deterministic_merge(&self) -> bool;
    fn topology(&self) -> &'static StreamSwitchTopology;
}

#[cfg(test)]
mod tests {
    use super::*;

    // A minimal synthetic topology for testing the for_tile accessor.
    // Uses the by-value data-struct property: no trait impl needed to
    // exercise carrier semantics.
    static TEST_MASTER_PORTS: &[u8] = &[1, 2, 3];
    static TEST_SLAVE_PORTS: &[u8] = &[4, 5, 6, 7];

    static TEST_TOPO: StreamSwitchTopology = StreamSwitchTopology {
        compute: TileStreamPorts {
            master_ports: TEST_MASTER_PORTS,
            slave_ports: TEST_SLAVE_PORTS,
            north_master: (1, 2),
            south_master: (3, 4),
            north_slave: (5, 6),
            south_slave: (7, 8),
        },
        memtile: TileStreamPorts {
            master_ports: TEST_MASTER_PORTS,
            slave_ports: TEST_SLAVE_PORTS,
            north_master: (11, 12),
            south_master: (13, 14),
            north_slave: (15, 16),
            south_slave: (17, 18),
        },
        shim: TileStreamPorts {
            master_ports: TEST_MASTER_PORTS,
            slave_ports: TEST_SLAVE_PORTS,
            north_master: (21, 22),
            south_master: (0, 0),
            north_slave: (23, 24),
            south_slave: (0, 0),
        },
    };

    #[test]
    fn for_tile_compute_dispatches_correctly() {
        let ports = TEST_TOPO.for_tile(TileKind::Compute);
        assert_eq!(ports.north_master, (1, 2));
        assert_eq!(ports.south_slave, (7, 8));
    }

    #[test]
    fn for_tile_memtile_dispatches_correctly() {
        let ports = TEST_TOPO.for_tile(TileKind::Mem);
        assert_eq!(ports.north_master, (11, 12));
    }

    #[test]
    fn for_tile_shim_noc_and_pl_return_same() {
        let noc = TEST_TOPO.for_tile(TileKind::ShimNoc);
        let pl = TEST_TOPO.for_tile(TileKind::ShimPl);
        assert_eq!(noc.north_master, pl.north_master);
        assert_eq!(noc.south_master, (0, 0));
        assert_eq!(noc.south_slave, (0, 0));
    }
}
```

- [x] **Step 2: Create `crates/xdna-archspec/src/aie2/stream_switch_model.rs`**

```rust
//! AIE2 stream switch model implementation.
//!
//! Covers NPU1 (Phoenix), NPU4 / NPU5 / NPU6 (Strix / Strix Halo /
//! Krackan). All AIE2-family devices share the same stream-switch
//! feature set:
//!
//! - Deterministic merge: available on all tile types (compute,
//!   memtile, shim).
//! - Packet routing: invariant across arches (same slot/arbiter/msel
//!   mechanisms per aie-rt shared xaie_ss.c).
//! - Port counts: compute = 23 master / 25 slave; memtile = 17 / 18;
//!   shim = 22 / 23. Sourced from
//!   `mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json`
//!   `Stream_Switch_*_Config` registers.
//!
//! A drift-detection test in this module asserts the hand-written
//! `AIE2_STREAM_SWITCH_TOPOLOGY` aggregate still agrees with the
//! build.rs-generated per-field constants.

use crate::aie2::stream_switch::{compute, mem_tile, shim};
use crate::aie2::{
    COMPUTE_MASTER_PORTS, COMPUTE_SLAVE_PORTS,
    MEMTILE_MASTER_PORTS, MEMTILE_SLAVE_PORTS,
    SHIM_MASTER_PORTS, SHIM_SLAVE_PORTS,
};
use crate::stream_switch::{StreamSwitchModel, StreamSwitchTopology, TileStreamPorts};

/// The AIE2 stream switch topology.
///
/// Static so hot-path consumers can cache `&'static StreamSwitchTopology`
/// at construction time. Drift-detection test below asserts this
/// aggregate still agrees with the build.rs-generated constants
/// it aggregates.
pub static AIE2_STREAM_SWITCH_TOPOLOGY: StreamSwitchTopology = StreamSwitchTopology {
    compute: TileStreamPorts {
        master_ports: COMPUTE_MASTER_PORTS,
        slave_ports: COMPUTE_SLAVE_PORTS,
        north_master: (compute::NORTH_MASTER_START, compute::NORTH_MASTER_END),
        south_master: (compute::SOUTH_MASTER_START, compute::SOUTH_MASTER_END),
        north_slave: (compute::NORTH_SLAVE_START, compute::NORTH_SLAVE_END),
        south_slave: (compute::SOUTH_SLAVE_START, compute::SOUTH_SLAVE_END),
    },
    memtile: TileStreamPorts {
        master_ports: MEMTILE_MASTER_PORTS,
        slave_ports: MEMTILE_SLAVE_PORTS,
        north_master: (mem_tile::NORTH_MASTER_START, mem_tile::NORTH_MASTER_END),
        south_master: (mem_tile::SOUTH_MASTER_START, mem_tile::SOUTH_MASTER_END),
        north_slave: (mem_tile::NORTH_SLAVE_START, mem_tile::NORTH_SLAVE_END),
        south_slave: (mem_tile::SOUTH_SLAVE_START, mem_tile::SOUTH_SLAVE_END),
    },
    shim: TileStreamPorts {
        master_ports: SHIM_MASTER_PORTS,
        slave_ports: SHIM_SLAVE_PORTS,
        north_master: (shim::NORTH_MASTER_START, shim::NORTH_MASTER_END),
        // Shim south-facing ports are the external NoC interface;
        // (0, 0) sentinel models "no intra-array south."
        south_master: (0, 0),
        north_slave: (shim::NORTH_SLAVE_START, shim::NORTH_SLAVE_END),
        south_slave: (0, 0),
    },
};

/// AIE2 stream switch model.
///
/// Zero-sized: a single `AIE2_STREAM_SWITCH_MODEL` static serves
/// every tile in every AIE2-family NPU. `ArchConfig::stream_switch_model()`
/// returns a `&'static dyn StreamSwitchModel` pointing at this singleton.
#[derive(Debug, Clone, Copy)]
pub struct Aie2StreamSwitchModel;

/// The single `Aie2StreamSwitchModel` instance used across every
/// AIE2-family consumer. Reference via `ArchConfig::stream_switch_model()`.
pub static AIE2_STREAM_SWITCH_MODEL: Aie2StreamSwitchModel = Aie2StreamSwitchModel;

impl StreamSwitchModel for Aie2StreamSwitchModel {
    fn supports_deterministic_merge(&self) -> bool {
        true
    }

    fn topology(&self) -> &'static StreamSwitchTopology {
        &AIE2_STREAM_SWITCH_TOPOLOGY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TileKind;

    #[test]
    fn aie2_stream_switch_model_feature_flag() {
        assert!(
            AIE2_STREAM_SWITCH_MODEL.supports_deterministic_merge(),
            "AIE2 has deterministic-merge registers on all tile types"
        );
    }

    #[test]
    fn aie2_stream_switch_model_topology_returns_static() {
        // Two calls return identical &'static references.
        let topo1 = AIE2_STREAM_SWITCH_MODEL.topology() as *const _;
        let topo2 = AIE2_STREAM_SWITCH_MODEL.topology() as *const _;
        assert_eq!(topo1, topo2, "topology() must return the same static");
    }

    #[test]
    fn aie2_topology_port_counts() {
        let topo = &AIE2_STREAM_SWITCH_TOPOLOGY;
        // Per AM025: compute = 23 master / 25 slave;
        // memtile = 17 / 18; shim = 22 / 23.
        assert_eq!(topo.compute.master_ports.len(), 23,
                   "compute master port count");
        assert_eq!(topo.compute.slave_ports.len(), 25,
                   "compute slave port count");
        assert_eq!(topo.memtile.master_ports.len(), 17,
                   "memtile master port count");
        assert_eq!(topo.memtile.slave_ports.len(), 18,
                   "memtile slave port count");
        assert_eq!(topo.shim.master_ports.len(), 22,
                   "shim master port count");
        assert_eq!(topo.shim.slave_ports.len(), 23,
                   "shim slave port count");
    }

    #[test]
    fn aie2_topology_compute_ranges() {
        let ports = AIE2_STREAM_SWITCH_TOPOLOGY.for_tile(TileKind::Compute);
        // Per AM025 compute ranges.
        assert_eq!(ports.south_master, (5, 8));
        assert_eq!(ports.south_slave, (5, 10));
        assert_eq!(ports.north_master, (13, 18));
        assert_eq!(ports.north_slave, (15, 18));
    }

    #[test]
    fn aie2_topology_memtile_ranges() {
        let ports = AIE2_STREAM_SWITCH_TOPOLOGY.for_tile(TileKind::Mem);
        assert_eq!(ports.south_master, (7, 10));
        assert_eq!(ports.south_slave, (7, 12));
        assert_eq!(ports.north_master, (11, 16));
        assert_eq!(ports.north_slave, (13, 16));
    }

    #[test]
    fn aie2_topology_shim_ranges() {
        let ports = AIE2_STREAM_SWITCH_TOPOLOGY.for_tile(TileKind::ShimNoc);
        assert_eq!(ports.north_master, (12, 17));
        assert_eq!(ports.north_slave, (14, 17));
        // Sentinels: shim has no intra-array south.
        assert_eq!(ports.south_master, (0, 0));
        assert_eq!(ports.south_slave, (0, 0));
    }

    #[test]
    fn aie2_topology_shim_pl_equals_shim_noc() {
        let noc = AIE2_STREAM_SWITCH_TOPOLOGY.for_tile(TileKind::ShimNoc);
        let pl = AIE2_STREAM_SWITCH_TOPOLOGY.for_tile(TileKind::ShimPl);
        assert_eq!(noc.master_ports, pl.master_ports);
        assert_eq!(noc.slave_ports, pl.slave_ports);
        assert_eq!(noc.north_master, pl.north_master);
    }

    /// Drift-detection: if the build.rs-generated per-field constants
    /// ever change without `AIE2_STREAM_SWITCH_TOPOLOGY` being updated,
    /// this test fires.
    #[test]
    fn aie2_topology_matches_generated_constants() {
        use crate::aie2::stream_switch::{compute, mem_tile, shim};

        // Port arrays
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.compute.master_ports,
                   crate::aie2::COMPUTE_MASTER_PORTS,
                   "compute master ports drifted");
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.compute.slave_ports,
                   crate::aie2::COMPUTE_SLAVE_PORTS,
                   "compute slave ports drifted");
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.memtile.master_ports,
                   crate::aie2::MEMTILE_MASTER_PORTS,
                   "memtile master ports drifted");
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.memtile.slave_ports,
                   crate::aie2::MEMTILE_SLAVE_PORTS,
                   "memtile slave ports drifted");
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.shim.master_ports,
                   crate::aie2::SHIM_MASTER_PORTS,
                   "shim master ports drifted");
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.shim.slave_ports,
                   crate::aie2::SHIM_SLAVE_PORTS,
                   "shim slave ports drifted");

        // Compute ranges
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.compute.north_master,
                   (compute::NORTH_MASTER_START, compute::NORTH_MASTER_END));
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.compute.south_master,
                   (compute::SOUTH_MASTER_START, compute::SOUTH_MASTER_END));
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.compute.north_slave,
                   (compute::NORTH_SLAVE_START, compute::NORTH_SLAVE_END));
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.compute.south_slave,
                   (compute::SOUTH_SLAVE_START, compute::SOUTH_SLAVE_END));

        // MemTile ranges
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.memtile.north_master,
                   (mem_tile::NORTH_MASTER_START, mem_tile::NORTH_MASTER_END));
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.memtile.south_master,
                   (mem_tile::SOUTH_MASTER_START, mem_tile::SOUTH_MASTER_END));
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.memtile.north_slave,
                   (mem_tile::NORTH_SLAVE_START, mem_tile::NORTH_SLAVE_END));
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.memtile.south_slave,
                   (mem_tile::SOUTH_SLAVE_START, mem_tile::SOUTH_SLAVE_END));

        // Shim ranges (north only; south is (0, 0) sentinel)
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.shim.north_master,
                   (shim::NORTH_MASTER_START, shim::NORTH_MASTER_END));
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.shim.north_slave,
                   (shim::NORTH_SLAVE_START, shim::NORTH_SLAVE_END));
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.shim.south_master, (0, 0),
                   "shim south-master sentinel preserved");
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.shim.south_slave, (0, 0),
                   "shim south-slave sentinel preserved");
    }
}
```

- [x] **Step 3: Export the new modules in `lib.rs`**

Open `crates/xdna-archspec/src/lib.rs` and add the new top-level module export. Find the line that currently reads `pub mod locks;` and add `pub mod stream_switch;` immediately after (maintaining alphabetical grouping of seam modules):

```rust
// ... existing pub mod declarations ...
pub mod locks;
pub mod stream_switch;
// ...
```

- [x] **Step 4: Export the AIE2 impl module in `aie2/mod.rs`**

Open `crates/xdna-archspec/src/aie2/mod.rs` and add `pub mod stream_switch_model;` alongside the existing `pub mod locks;` / `pub mod dma;` declarations (the ones introduced by Subsystems 3 and 4, around lines 99-106):

```rust
/// AIE2-family DMA model (`Aie2DmaModel` + `AIE2_DMA_MODEL` static). Subsystem 3 seam.
pub mod dma;

/// AIE2-family lock model (`Aie2LockModel` + `AIE2_LOCK_MODEL` static). Subsystem 4 seam.
pub mod locks;

/// AIE2-family stream switch model (`Aie2StreamSwitchModel` +
/// `AIE2_STREAM_SWITCH_MODEL` static). Subsystem 5 seam.
pub mod stream_switch_model;
```

- [x] **Step 5: Run archspec tests to verify the trait compiles and statics initialize**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -5
```

Expected: `test result: ok. 293 passed; 0 failed; 2 ignored` (baseline 282 + 3 trait tests + 8 impl tests in stream_switch_model).

If any test fails, fix before committing. Pay particular attention to
the drift-detection test -- it is the main safety net for this
subsystem. If it fires, the hand-written aggregate and the generated
constants have genuinely drifted; fix the aggregate to match.

- [x] **Step 6: Run xdna-emu tests (no regression expected; archspec changes only)**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
```

Expected: `test result: ok. 2687 passed; 0 failed; 5 ignored`.

- [x] **Step 7: Commit Task 2**

```bash
git add crates/xdna-archspec/src/stream_switch/mod.rs \
        crates/xdna-archspec/src/aie2/stream_switch_model.rs \
        crates/xdna-archspec/src/lib.rs \
        crates/xdna-archspec/src/aie2/mod.rs

git commit -m "$(cat <<'EOF'
feat(archspec): StreamSwitchModel trait + StreamSwitchTopology + Aie2StreamSwitchModel

Two-method trait: supports_deterministic_merge (AIE1 false, AIE2 true
per aie-rt xaiegbl_reginit.c / xaiemlgbl_reginit.c) + topology()
returning &'static StreamSwitchTopology. The carrier groups
per-tile-kind TileStreamPorts sub-structs (master/slave port arrays +
N/S ranges) with a for_tile(TileKind) accessor.

AIE2 concrete impl: Aie2StreamSwitchModel (ZST) + AIE2_STREAM_SWITCH_MODEL
+ AIE2_STREAM_SWITCH_TOPOLOGY statics. Topology aggregates existing
build.rs-generated port arrays (COMPUTE/MEMTILE/SHIM_MASTER/SLAVE_PORTS)
and range constants (compute::, mem_tile::, shim:: modules). Shim
south_master/south_slave use (0, 0) sentinel (no intra-array south),
preserving the semantic the deleted PortLayout extension trait carried.

Drift-detection test asserts the hand-written aggregate still agrees
with all six port arrays and all four ranges per tile kind. Shape
tests cover the two trait methods and the for_tile accessor for all
four TileKind variants.

No trait-wiring yet -- ArchConfig::stream_switch_model() accessor
lands in Task 3; xdna-emu call-site migration lands in Task 4.

Generated using Claude Code.
EOF
)"
```

---

## Task 3: ArchConfig::stream_switch_model() accessor

**Files:**
- Modify: `crates/xdna-archspec/src/runtime.rs` (add trait method + impl)

**Goal:** Wire the new trait through `ArchConfig`, mirroring the existing `dma_model()` and `lock_model()` accessors. After this task, any code path holding `&dyn ArchConfig` or `Arc<dyn ArchConfig>` can call `.stream_switch_model()` to get the architecture's model.

- [x] **Step 1: Add the trait method declaration**

Open `crates/xdna-archspec/src/runtime.rs`. Locate the existing trait method declarations (search for `fn lock_model(&self)` -- that is the neighbor to mimic). Add the new method declaration immediately after:

```rust
    /// Returns the stream-switch model for this architecture.
    ///
    /// The returned reference is `&'static` so consumers may cache
    /// it for the lifetime of the process without concern about
    /// dangling pointers. All AIE2-family devices share a single
    /// `AIE2_STREAM_SWITCH_MODEL` singleton.
    ///
    /// Calls on `Architecture::Aie` (AIE1 / Versal) panic via
    /// `unimplemented!()` until an AIE1 model is populated.
    fn stream_switch_model(&self) -> &'static dyn crate::stream_switch::StreamSwitchModel;
```

- [x] **Step 2: Add the `ModelConfig` impl**

Find the matching `impl ArchConfig for ModelConfig` block -- specifically the `fn lock_model` implementation -- and add the new method immediately after:

```rust
    fn stream_switch_model(&self) -> &'static dyn crate::stream_switch::StreamSwitchModel {
        match self.architecture {
            Architecture::Aie2 | Architecture::Aie2p => {
                &crate::aie2::stream_switch_model::AIE2_STREAM_SWITCH_MODEL
            }
            Architecture::Aie => unimplemented!(
                "AIE1 StreamSwitchModel not populated; add \
                 xdna_archspec::aie1::stream_switch_model::AIE1_STREAM_SWITCH_MODEL"
            ),
        }
    }
```

- [x] **Step 3: Add a trait-dispatch test**

Still in `runtime.rs`, scroll to the existing `#[cfg(test)] mod tests` block and add a test that verifies `stream_switch_model()` returns the AIE2 singleton for each AIE2-family device:

```rust
    #[test]
    fn stream_switch_model_dispatches_to_aie2_for_aie2_family() {
        use crate::stream_switch::StreamSwitchModel;
        for name in &["npu1", "npu2", "npu4", "npu5", "npu6"] {
            if let Some(model) = ARCHSPEC_MODELS.get(*name) {
                let cfg = ModelConfig::from_arch_model(model, name);
                assert!(
                    cfg.stream_switch_model().supports_deterministic_merge(),
                    "{}: AIE2-family must report supports_deterministic_merge = true",
                    name
                );
            }
        }
    }
```

(Only devices present in the loaded archspec models will be exercised;
the `if let Some(model)` pattern keeps the test resilient to which set
of devices is configured in the model JSON.)

- [x] **Step 4: Run archspec tests**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -5
```

Expected: archspec test count grows by 1 (the new dispatch test). Total should be 294 passed. No failures.

- [x] **Step 5: Run xdna-emu tests (regression check for `ArchConfig` supertrait change)**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
```

Expected: `2687 passed; 0 failed; 5 ignored` (no regression). Adding a
new trait method does not break existing implementations because the
trait object supports the new method via the impl added in Step 2.

- [x] **Step 6: Commit Task 3**

```bash
git add crates/xdna-archspec/src/runtime.rs
git commit -m "$(cat <<'EOF'
feat(archspec): ArchConfig::stream_switch_model() accessor

Dispatches on Architecture enum, mirroring dma_model() and lock_model():
Aie2 | Aie2p -> &AIE2_STREAM_SWITCH_MODEL, Aie -> unimplemented!() with
the standard "populate me when AIE1 support starts" message.

Test asserts the accessor returns the AIE2 singleton (feature flag
true) for every AIE2-family device present in ARCHSPEC_MODELS. Test
stays resilient to which device set is loaded via an `if let Some`
gate on the model lookup.

Generated using Claude Code.
EOF
)"
```

---

## Task 4: Runtime-side accessor + tile-construction call-site migration

**Files:**
- Modify: `src/device/arch_handle.rs` (add `stream_switch_topology()` accessor)
- Modify: `src/device/stream_switch/mod.rs` (6 call-site rewrites + use-statement)

**Goal:** Add the process-global `OnceLock` accessor on the xdna-emu side, then route the three `StreamSwitch::new_*` constructors through it instead of reading the archspec constants directly. Mirrors the `arch_handle::lock_value_layout()` pattern from Subsystem 4.

- [x] **Step 1: Add `stream_switch_topology()` accessor in `arch_handle.rs`**

Open `src/device/arch_handle.rs`. Locate the existing `lock_value_layout()` function (added in Subsystem 4). Add the new accessor immediately after it, mirroring the same `OnceLock<&'static ...>` pattern:

```rust
/// Process-global cached `&'static StreamSwitchTopology` for the
/// default architecture.
///
/// Seeded on first access from `default_arch().stream_switch_model().topology()`.
/// Bridge pattern: when GUI runtime arch-switch lands, consumers pick
/// up the live arch's topology by calling this function per-access
/// rather than stashing the reference at construction time.
///
/// Mirrors `lock_value_layout()` (Subsystem 4) and should be unified
/// with it by Subsystem 7/8 if more process-global handles accumulate.
pub fn stream_switch_topology() -> &'static xdna_archspec::stream_switch::StreamSwitchTopology {
    static CACHED: std::sync::OnceLock<&'static xdna_archspec::stream_switch::StreamSwitchTopology>
        = std::sync::OnceLock::new();
    CACHED.get_or_init(|| {
        default_arch().stream_switch_model().topology()
    })
}
```

(The `default_arch()` function already exists in this module from
Subsystem 4; if the exact name differs, match whatever the existing
`lock_value_layout()` uses -- they share the same seed helper.)

- [x] **Step 2: Migrate the 6 call sites in `stream_switch/mod.rs`**

Open `src/device/stream_switch/mod.rs`. At lines 131-133, 163-165, and 193-195 (three pairs of `build_ports_from_spec` calls, one per `new_*_tile` constructor), rewrite each call to read through the new accessor.

Before (at `new_compute_tile`, around line 131):

```rust
    pub fn new_compute_tile(col: u8, row: u8) -> Self {
        let masters = Self::build_ports_from_spec(xdna_archspec::aie2::COMPUTE_MASTER_PORTS, PortDirection::Master);
        let slaves = Self::build_ports_from_spec(xdna_archspec::aie2::COMPUTE_SLAVE_PORTS, PortDirection::Slave);
```

After:

```rust
    pub fn new_compute_tile(col: u8, row: u8) -> Self {
        let ports = crate::device::arch_handle::stream_switch_topology()
            .for_tile(xdna_archspec::types::TileKind::Compute);
        let masters = Self::build_ports_from_spec(ports.master_ports, PortDirection::Master);
        let slaves = Self::build_ports_from_spec(ports.slave_ports, PortDirection::Slave);
```

Before (at `new_mem_tile`, around line 163):

```rust
    pub fn new_mem_tile(col: u8, row: u8) -> Self {
        let masters = Self::build_ports_from_spec(xdna_archspec::aie2::MEMTILE_MASTER_PORTS, PortDirection::Master);
        let slaves = Self::build_ports_from_spec(xdna_archspec::aie2::MEMTILE_SLAVE_PORTS, PortDirection::Slave);
```

After:

```rust
    pub fn new_mem_tile(col: u8, row: u8) -> Self {
        let ports = crate::device::arch_handle::stream_switch_topology()
            .for_tile(xdna_archspec::types::TileKind::Mem);
        let masters = Self::build_ports_from_spec(ports.master_ports, PortDirection::Master);
        let slaves = Self::build_ports_from_spec(ports.slave_ports, PortDirection::Slave);
```

Before (at `new_shim_tile`, around line 193):

```rust
    pub fn new_shim_tile(col: u8) -> Self {
        let masters = Self::build_ports_from_spec(xdna_archspec::aie2::SHIM_MASTER_PORTS, PortDirection::Master);
        let slaves = Self::build_ports_from_spec(xdna_archspec::aie2::SHIM_SLAVE_PORTS, PortDirection::Slave);
```

After:

```rust
    pub fn new_shim_tile(col: u8) -> Self {
        let ports = crate::device::arch_handle::stream_switch_topology()
            .for_tile(xdna_archspec::types::TileKind::ShimNoc);
        let masters = Self::build_ports_from_spec(ports.master_ports, PortDirection::Master);
        let slaves = Self::build_ports_from_spec(ports.slave_ports, PortDirection::Slave);
```

- [x] **Step 3: Run xdna-emu tests**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
```

Expected: `2687 passed; 0 failed; 5 ignored`. The migrated call sites
read the same data via a different path; there should be zero
behavioral change.

If any test fails, inspect the diff at the three migration points and
verify the `TileKind` variant selected matches the original constant
(e.g., `new_compute_tile` -> `TileKind::Compute`).

- [x] **Step 4: Run archspec tests**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
```

Expected: no change from Task 3 (no archspec code touched this task).

- [x] **Step 5: Fast bridge smoke**

```bash
cargo build -p xdna-emu-ffi 2>&1 | tail -5
./scripts/emu-bridge-test.sh --no-hw -v add_one_cpp_aiecc 2>&1 | tail -30
```

Expected: both Chess and Peano PASS. If the bridge test fails here, it
indicates the tile-construction path is producing different port
layouts than before -- pause and inspect.

- [x] **Step 6: Commit Task 4**

```bash
git add src/device/arch_handle.rs src/device/stream_switch/mod.rs
git commit -m "$(cat <<'EOF'
refactor: migrate tile-construction call sites to StreamSwitchTopology

arch_handle::stream_switch_topology() accessor added, mirroring
lock_value_layout() from Subsystem 4 (OnceLock<&'static ...> seeded
from default_arch().stream_switch_model().topology()).

Three tile-construction functions in src/device/stream_switch/mod.rs
(new_compute_tile, new_mem_tile, new_shim_tile) each previously read
two archspec port-array constants directly via
xdna_archspec::aie2::{COMPUTE,MEMTILE,SHIM}_{MASTER,SLAVE}_PORTS. Each
now reads through the seam:

    let ports = arch_handle::stream_switch_topology()
        .for_tile(TileKind::<kind>);
    let masters = build_ports_from_spec(ports.master_ports, ...);
    let slaves = build_ports_from_spec(ports.slave_ports, ...);

Data read is identical (drift-detection test locks the aggregate to
the underlying constants); dispatch path is now arch-dispatched, ready
for AIE1 plug-in.

Generated using Claude Code.
EOF
)"
```

---

## Task 5: Delete PortLayout + migrate tests + fix stale doc-comments

**Files:**
- Delete: `src/device/port_layout.rs`
- Modify: `src/device/mod.rs` (remove `pub mod port_layout;`)
- Modify: `crates/xdna-archspec/src/stream_switch/mod.rs` (append 3 migrated tests)
- Modify: `crates/xdna-archspec/src/runtime.rs` (3 stale doc-comment updates)

**Goal:** Delete the dead-code `PortLayout` extension trait, migrate its 3 tests to archspec with assertions restated against the new carrier, and clean up the 3 stale doc-comment references that still describe the old runtime-side-only rationale.

- [x] **Step 1: Migrate the 3 tests into archspec's stream_switch/mod.rs**

Open `crates/xdna-archspec/src/stream_switch/mod.rs`. In the existing `#[cfg(test)] mod tests` block (added in Task 2), append 3 more tests that restate the assertions from `src/device/port_layout.rs:138-229` against `AIE2_STREAM_SWITCH_TOPOLOGY`:

```rust
    // === Migrated from src/device/port_layout.rs ===
    //
    // These tests originally asserted properties of the runtime-side
    // PortLayout extension trait. With PortLayout deleted and its data
    // source (the archspec-generated port arrays) reachable via
    // AIE2_STREAM_SWITCH_TOPOLOGY, the assertions are restated against
    // the carrier.

    #[test]
    fn test_npu1_port_layouts_migrated() {
        use crate::aie2::stream_switch_model::AIE2_STREAM_SWITCH_TOPOLOGY;

        // Verify port counts match AM025 spec (restated from the original
        // `test_npu1_port_layouts` in src/device/port_layout.rs).
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.shim.master_ports.len(), 22);
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.shim.slave_ports.len(), 23);

        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.memtile.master_ports.len(), 17);
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.memtile.slave_ports.len(), 18);

        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.compute.master_ports.len(), 23);
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.compute.slave_ports.len(), 25);
    }

    #[test]
    fn test_npu1_port_ranges_migrated() {
        use crate::aie2::stream_switch::{compute, mem_tile, shim};
        use crate::aie2::stream_switch_model::AIE2_STREAM_SWITCH_TOPOLOGY;

        // Shim north masters: 12-17 (6 ports)
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.shim.north_master, (12, 17));
        let (start, end) = AIE2_STREAM_SWITCH_TOPOLOGY.shim.north_master;
        assert_eq!(end - start + 1, 6);

        // MemTile south masters: 7-10 (4 ports)
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.memtile.south_master, (7, 10));

        // MemTile north masters: 11-16 (6 ports)
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.memtile.north_master, (11, 16));

        // Shim has no intra-array south (sentinel).
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.shim.south_master, (0, 0));
        assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.shim.south_slave, (0, 0));

        // E/W generated constants (not in carrier fields today, but the
        // drift-detection test above locks them. These direct-access
        // assertions are preserved here because they were in the
        // original test.)
        assert_eq!(compute::EAST_MASTER_START, 19);
        assert_eq!(compute::EAST_MASTER_END, 22);
        assert_eq!(compute::WEST_MASTER_START, 9);
        assert_eq!(compute::WEST_MASTER_END, 12);
        assert_eq!(shim::EAST_MASTER_START, 18);
        assert_eq!(shim::EAST_MASTER_END, 21);
        assert_eq!(shim::WEST_MASTER_START, 8);
        assert_eq!(shim::WEST_MASTER_END, 11);
        assert_eq!(shim::SOUTH_MASTER_START, 2);
        assert_eq!(shim::SOUTH_MASTER_END, 7);

        // MemTile has no E/W ports (structural check: the symbols must
        // still exist on compute/shim but not mem_tile).
        assert_eq!(mem_tile::SOUTH_MASTER_START, 7);
        assert_eq!(mem_tile::NORTH_MASTER_START, 11);
    }

    #[test]
    fn test_shim_pl_same_as_shim_noc_migrated() {
        use crate::aie2::stream_switch_model::AIE2_STREAM_SWITCH_TOPOLOGY;

        let noc = AIE2_STREAM_SWITCH_TOPOLOGY.for_tile(TileKind::ShimNoc);
        let pl = AIE2_STREAM_SWITCH_TOPOLOGY.for_tile(TileKind::ShimPl);
        assert_eq!(noc.master_ports, pl.master_ports);
        assert_eq!(noc.slave_ports, pl.slave_ports);
        assert_eq!(noc.north_master, pl.north_master);
    }
```

- [x] **Step 2: Run archspec tests to verify migrated tests pass**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib stream_switch 2>&1 | tail -10
```

Expected: all three migrated tests pass (plus the Task 2 tests).

- [x] **Step 3: Delete `src/device/port_layout.rs`**

```bash
rm /home/triple/npu-work/xdna-emu/src/device/port_layout.rs
```

Verify it's gone:

```bash
ls /home/triple/npu-work/xdna-emu/src/device/port_layout.rs 2>&1
```

Expected: `ls: cannot access ... No such file or directory`.

- [x] **Step 4: Remove `pub mod port_layout;` from `src/device/mod.rs`**

Open `src/device/mod.rs`. Find line 52 (`pub mod port_layout;`) and delete it. Also scan the rest of the file for any `use port_layout::PortLayout;` re-exports and delete those too. Expected edit:

```rust
// Line 52 previously read:
// pub mod port_layout;
// Now deleted.
```

- [x] **Step 5: Run `cargo check` to verify nothing else referenced PortLayout**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo check --lib 2>&1 | tail -20
```

Expected: clean compile. If anything complains about an unresolved
`PortLayout` import or an unknown `use crate::device::port_layout`,
that's a latent external consumer -- note it, migrate it to the seam,
and retry.

- [x] **Step 6: Update 3 stale doc-comment references in archspec runtime.rs**

Open `crates/xdna-archspec/src/runtime.rs`. At each of the three line numbers noted in the audit (13, 63, 221), there are doc-comments referring to `PortLayout` or its runtime-side rationale. Replace each with language describing the new seam.

At **line 13** (module-level doc):

Find and replace the paragraph mentioning `PortLayout` with a reference to
`StreamSwitchModel`:

```rust
//! Stream-switch port layout is reachable via
//! `ArchConfig::stream_switch_model().topology()` on this trait.
//! See also `xdna_archspec::stream_switch::StreamSwitchModel` for
//! the trait definition and `xdna_archspec::aie2::stream_switch_model`
//! for the AIE2 concrete implementation.
```

At **line 63** (doc-comment above a stream-switch-related `ArchConfig` method
or const):

Update the existing sentence that reads "...exposed via the runtime-side
`PortLayout` extension trait because their data comes from build.rs-generated..."
to:

```rust
/// ...exposed via `ArchConfig::stream_switch_model().topology()`
/// (Subsystem 5). The underlying build.rs-generated port arrays
/// and range constants are aggregated into
/// `StreamSwitchTopology` in `xdna_archspec::aie2::stream_switch_model`.
```

At **line 221** (doc-comment, similar):

Update any remaining `PortLayout`-related phrasing to describe the
new seam.

Exact wording can vary as long as:
- No references to `PortLayout` remain.
- Stream-switch port-layout access is attributed to `ArchConfig::stream_switch_model().topology()`.
- The build.rs codegen story (port arrays + range modules in archspec) is
  still accurately described.

After edits, verify no stale references remain:

```bash
grep -rn "PortLayout" /home/triple/npu-work/xdna-emu/crates/xdna-archspec/src/ \
                       /home/triple/npu-work/xdna-emu/src/ 2>&1
```

Expected: no hits.

- [x] **Step 7: Run full tests**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
```

Expected:
- xdna-emu: `2684 passed; 0 failed; 5 ignored` (baseline 2687 minus the 3 tests that migrated to archspec).
- archspec: `297 passed; 0 failed; 2 ignored` (Task 3 brought it to 294; 3 migrated tests land here).

- [x] **Step 8: Commit Task 5**

```bash
git add -u src/device/port_layout.rs src/device/mod.rs \
            crates/xdna-archspec/src/runtime.rs \
            crates/xdna-archspec/src/stream_switch/mod.rs

git commit -m "$(cat <<'EOF'
refactor: collapse dead-code PortLayout extension trait

src/device/port_layout.rs deletes entirely (231 LOC: trait + impl +
3 tests). Verified zero external consumers pre-delete via `grep
PortLayout` -- the trait was runtime-side only because a stale
pre-Subsystem-1 doc-comment claimed archspec couldn't access the
data, which stopped being true when Subsystem 1 moved the codegen
into archspec's own build.rs.

Three tests migrate to archspec's stream_switch/mod.rs with assertions
restated against AIE2_STREAM_SWITCH_TOPOLOGY (port counts, ranges,
ShimPl/ShimNoc equivalence). No assertion values change.

Three stale PortLayout doc-comment references in
crates/xdna-archspec/src/runtime.rs updated to describe the new
StreamSwitchModel seam.

pub mod port_layout line removed from src/device/mod.rs.

Net test count: -3 in xdna-emu (2687 -> 2684), +3 in archspec (294 ->
297). No behavioral change.

Generated using Claude Code.
EOF
)"
```

---

## Task 6: Hygiene + audit/docs completion sweep

**Files:**
- Modify: `docs/arch/subsys5-audit.md` (fill in Completion section)
- Modify: `docs/arch/stream-switch-model.md` (fill in What lives where + trait surface + Completion)
- Possibly modify: `NEXT-STEPS.md` (will be updated in Task 7 at tag time, not here)

**Goal:** Before the tag, fill in the docs that were scaffolded in Task 1 with the real commit history, test counts, and file-layout snapshot. This is a "docs catch up with reality" pass.

- [x] **Step 1: Gather the task-by-task commit history**

```bash
git log --oneline phase1-subsys-locks..HEAD
```

Expected output (roughly):

```
<sha>  docs: Subsystem 5 audit + stream-switch-model design note scaffolds
<sha>  feat(archspec): StreamSwitchModel trait + StreamSwitchTopology + Aie2StreamSwitchModel
<sha>  feat(archspec): ArchConfig::stream_switch_model() accessor
<sha>  refactor: migrate tile-construction call sites to StreamSwitchTopology
<sha>  refactor: collapse dead-code PortLayout extension trait
```

Copy the SHAs for the audit's Completion section.

- [x] **Step 2: Fill in `docs/arch/subsys5-audit.md` Completion section**

Replace the `## Completion\n\n(Filled in at end of Subsystem 5.)` stub
with a populated section matching the shape of
`docs/arch/subsys4-audit.md`:

```markdown
## Completion

Landed YYYY-MM-DD. Tag: `phase1-subsys-stream-switch` (set in Task 7).

### Commits (Task 1 through tag)

```
<sha>  docs: Subsystem 5 audit + stream-switch-model design note scaffolds
<sha>  feat(archspec): StreamSwitchModel trait + StreamSwitchTopology + Aie2StreamSwitchModel
<sha>  feat(archspec): ArchConfig::stream_switch_model() accessor
<sha>  refactor: migrate tile-construction call sites to StreamSwitchTopology
<sha>  refactor: collapse dead-code PortLayout extension trait
<sha>  docs: Subsystem 5 completion log + NEXT-STEPS points at Subsystem 7
(The final "docs: Subsystem 5 completion log" commit is the one this
list is being written into -- see `git log --oneline`.)
```

### Verification (at tag)

- `cargo test --lib`: 2684 passed; 0 failed; 5 ignored.
- `cargo test -p xdna-archspec --lib`: 297 passed; 0 failed; 2 ignored.
- `cargo build --release`: clean.
- FFI cdylib rebuild (`cargo build -p xdna-emu-ffi`): clean.
- Bridge `--no-hw -v add_one_cpp_aiecc`: Chess and Peano PASS.
- Full HW bridge: matches phase1-subsys-locks character (pre-existing
  `bd_chain_repeat_on_memtile` EMU deadlock is the only HW failure).
- ISA test suite: 4815/4815 PASS (100.0%); FAIL: 0.

### Success criteria sweep

- `StreamSwitchModel` trait in `xdna_archspec::stream_switch` (2 methods):
  populated.
- `StreamSwitchTopology` carrier + `TileStreamPorts` + `for_tile`
  accessor: populated.
- `Aie2StreamSwitchModel` + `AIE2_STREAM_SWITCH_MODEL` +
  `AIE2_STREAM_SWITCH_TOPOLOGY` statics: populated.
- `ArchConfig::stream_switch_model()` accessor: populated.
- `src/device/port_layout.rs`: **deleted** (all 231 LOC, all 3 tests).
- `arch_handle::stream_switch_topology()` accessor: populated.
- 6 xdna-emu tile-construction call sites migrated: done.
- 3 port-layout tests migrated to archspec: done.
- 3 stale `PortLayout` doc-comments in archspec runtime.rs updated: done.
- Drift-detection test in `aie2/stream_switch_model.rs` locks all 6
  port arrays and all 4 ranges per tile kind: added.
- `docs/arch/stream-switch-model.md` design note: written.

### Net code delta

- New in archspec: ~180 LOC (StreamSwitchModel + StreamSwitchTopology +
  TileStreamPorts + Aie2StreamSwitchModel + statics + tests + drift
  test).
- Deleted in xdna-emu: ~231 LOC (entire port_layout.rs including tests).
- Modified in xdna-emu: 6 call-site rewrites (~20 LOC touched), new
  `arch_handle::stream_switch_topology()` accessor (~15 LOC), one
  `pub mod port_layout;` line removed.
- Modified in archspec: ~3 doc-comment updates in runtime.rs.
- Net workspace LOC change: ~-50 LOC (the port_layout.rs deletion
  outweighs the new archspec code because port_layout.rs included
  3 tests + a lot of doc-comments).

### Follow-ups flagged

Follow-ups that fit naturally in later work, NOT blocking:

- **AIE1 plug-in:** `Aie1StreamSwitchModel` +
  `AIE1_STREAM_SWITCH_TOPOLOGY` fill in when AIE1 support starts. The
  `memtile` field's `Option<_>` vs sentinel decision lands at that
  point.
- **Direct archspec-constant consumers migrate to the seam** at AIE1
  landing: `src/device/dma/stream_io.rs` const declarations,
  `src/device/array/routing.rs` range-math sites, `src/device/state/
  {compute,memtile}.rs` ENABLE_BIT / SLAVE_SELECT_MASK uses. On AIE2
  they work correctly via direct constant access; on AIE1 they would
  silently read AIE2 data and produce wrong routing decisions.
- **Carrier expansion:** E/W ranges, TRACE_SLAVE, DMA_MASTER/SLAVE
  ranges would join `TileStreamPorts` when `routing.rs` migrates
  through the seam. Not done today because adding fields no one
  reads is ceremony.
- **Generic-type-parameter monomorphization:** post-seam-pass
  optimization direction. Hot types reaching `&'static dyn
  StreamSwitchModel` switch to `<S: StreamSwitchModel>`; the
  `StreamSwitchTopology` carrier stays as-is.
- **`arch_handle` module generalization:** now exposes
  `lock_value_layout()` + `stream_switch_topology()`. If Subsystem 7/8
  needs similar handles, extend the module; if enough accumulate,
  split into submodules.
- **Phase 2 hygiene carried through:**
  - `OnceLock<&'static StreamSwitchTopology>` in `arch_handle` could
    simplify to `OnceLock<StreamSwitchTopology>` by value (mirrors the
    Subsystem 4 Phase 2 note for `lock_value_layout()`).
  - Pre-existing dead-code warnings and Subsystem-6-era rot (not
    Subsystem 5's scope).
- **Subsystem 7 (ISA Execute):** see `NEXT-STEPS.md` pickup guide
  (written in Task 7 at tag time).
```

- [x] **Step 3: Fill in `docs/arch/stream-switch-model.md`**

Replace the stubs in the `## What lives where` and `## The trait surface` sections with real content now that the code exists:

```markdown
## What lives where

All entries below reflect state as of the `phase1-subsys-stream-switch` tag.

| Data/code | Module | Source |
|---|---|---|
| `StreamSwitchModel` trait (2 methods) + `StreamSwitchTopology` + `TileStreamPorts` carrier | `xdna_archspec::stream_switch` | Emulator design |
| `Aie2StreamSwitchModel` concrete impl + `AIE2_STREAM_SWITCH_MODEL` + `AIE2_STREAM_SWITCH_TOPOLOGY` statics | `xdna_archspec::aie2::stream_switch_model` | aie-rt xaiemlgbl_reginit.c + AM025 register DB JSON |
| `ArchConfig::stream_switch_model()` accessor | `xdna_archspec::runtime` | Dispatches on `Architecture` |
| `arch_handle::stream_switch_topology()` | `xdna_emu::device::arch_handle` | Process-global `OnceLock` cache |
| `StreamSwitch` struct + routing state | `xdna_emu::device::stream_switch` | Unchanged (traits describe, don't hold state) |
```

And replace the `## The trait surface` stub with:

```markdown
## The trait surface

```rust
pub trait StreamSwitchModel: Send + Sync + core::fmt::Debug {
    fn supports_deterministic_merge(&self) -> bool;
    fn topology(&self) -> &'static StreamSwitchTopology;
}
```

Two methods, "coarse first":

- One feature flag covers the one genuine behavioral divergence
  surfaced by the aie-rt audit (AIE1 lacks deterministic merge; AIE2
  has it on all tile types).
- `topology` returns a carrier struct with three `TileStreamPorts`
  sub-structs (one per tile kind) and a `for_tile(TileKind)`
  accessor. Carrier is `Copy` and `Debug`, trivially composable
  across arch boundaries.

Not on the trait:

- `supports_packet_routing`: invariant `true` across AIE1/AIE2
  per aie-rt shared `xaie_ss.c`. Ceremony.
- Port-validity check (`validate_route`): programming-time concern
  in aie-rt, not emulated today.
- Packet slot count / arbiter count: invariant (4 / 8 across arches).
- `StreamSwitch` runtime state (slots, arbiter locks, priority
  pointers): holds state, not a trait concern (traits describe,
  don't hold state).
```

Then replace the final `## Completion\n\n(Filled in at end of Subsystem 5.)` stub with a populated section:

```markdown
## Completion (YYYY-MM-DD)

Landed at `phase1-subsys-stream-switch`. Net effect:

- `xdna_archspec::stream_switch::StreamSwitchModel` trait (2 methods:
  `supports_deterministic_merge`, `topology`) + `StreamSwitchTopology`
  carrier + `TileStreamPorts` + `for_tile(TileKind)` accessor live at
  the crate root.
- `xdna_archspec::aie2::stream_switch_model::Aie2StreamSwitchModel`
  concrete impl + `AIE2_STREAM_SWITCH_MODEL` + `AIE2_STREAM_SWITCH_TOPOLOGY`
  statics for AIE2-family devices (NPU1/NPU4/NPU5/NPU6).
- `xdna_archspec::runtime::ArchConfig::stream_switch_model()` accessor
  returns `&'static dyn StreamSwitchModel`, dispatching on
  `ModelConfig::architecture` exactly like `dma_model()` and
  `lock_model()`.
- xdna-emu's `src/device/port_layout.rs` (231 LOC, dead-code extension
  trait with 3 tests) fully deleted; tests migrated to
  `xdna_archspec::stream_switch` with assertions restated against
  `AIE2_STREAM_SWITCH_TOPOLOGY`.
- Six tile-construction call sites (`StreamSwitch::new_compute_tile` /
  `new_mem_tile` / `new_shim_tile`, each x masters + slaves) migrate
  to `arch_handle::stream_switch_topology().for_tile(kind).master_ports`
  (and `slave_ports`).
- New `src/device/arch_handle.rs` accessor `stream_switch_topology()`
  provides a process-global `OnceLock` cache of
  `&'static StreamSwitchTopology`, seeded from
  `default_arch().stream_switch_model().topology()`. Mirrors the
  Subsystem 4 `lock_value_layout()` bridge pattern until GUI runtime
  arch-switch lands.
- Drift-detection test in `crates/xdna-archspec/src/aie2/stream_switch_model.rs`
  asserts `AIE2_STREAM_SWITCH_TOPOLOGY` agrees with all six
  `*_MASTER_PORTS` / `*_SLAVE_PORTS` arrays and all four
  N/S range constants per tile kind.

Verification: `cargo test --lib` = 2684 passed / 0 failed / 5 ignored;
archspec = 297 passed / 0 failed / 2 ignored; full HW bridge matches
phase1-subsys-locks character; ISA 4815/4815 PASS.
```

- [x] **Step 4: Fill in actual commit SHAs from `git log --oneline`**

Replace the `<sha>` placeholders in both docs with the real SHAs from
`git log --oneline phase1-subsys-locks..HEAD`. Replace `YYYY-MM-DD`
with today's date.

- [x] **Step 5: Fill in the Task 7 placeholder**

The audit + design note both reference "Task 7" for the tag and
NEXT-STEPS update. Task 7 will commit both of these doc updates in
the same commit as the NEXT-STEPS update, so no separate commit here
-- just leave the docs in a staged-but-uncommitted state for Task 7.

- [x] **Step 6: Run tests once more as a regression check**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
```

Expected: unchanged from Task 5 (2684 / 297). Docs don't affect tests.

No commit this step; the doc updates are committed in Task 7 alongside NEXT-STEPS updates.

---

## Task 7: Final gate + tag

**Files:**
- Modify: `NEXT-STEPS.md` (update "Last updated", "Latest tag", the
  status table, and the "How to Pick Up Subsystem 7" section)
- Modify: `docs/arch/subsys5-audit.md` + `docs/arch/stream-switch-model.md`
  (if still unstaged from Task 6, stage them)

**Goal:** Run the full gate suite, validate everything is green, tag
the subsystem, and update `NEXT-STEPS.md` to point at Subsystem 7
(ISA Execute). This is the one task that exercises HW-touching
suites (full bridge + ISA) -- budget ~45-60 min for the runs.

- [x] **Step 1: Clean release build + FFI rebuild**

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build --release 2>&1 | tail -5
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build -p xdna-emu-ffi 2>&1 | tail -5
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo build -p xdna-emu-ffi --release 2>&1 | tail -5
```

Expected: all three clean (no warnings, no errors). If any warning
appears that wasn't present at the Subsystem 4 baseline, pause and
investigate -- it likely indicates an import or visibility
misalignment from the port_layout deletion.

- [x] **Step 2: Fast bridge smoke**

```bash
./scripts/emu-bridge-test.sh --no-hw -v add_one_cpp_aiecc 2>&1 | tail -30
```

Expected: Chess PASS, Peano PASS.

- [x] **Step 3: Run full bridge (HW, ~30 min)**

```bash
./scripts/emu-bridge-test.sh 2>&1 | tee /tmp/claude-1000/bridge-subsys5.log
```

Expected at tag (matches Subsystem 4 / Subsystem 3 character):
- Chess: 62/62 compiled, HW 63 pass / 1 fail (the pre-existing
  `bd_chain_repeat_on_memtile` deadlock; documented in NEXT-STEPS).
- Peano: 55/55 compiled, HW 53 pass / 1 fail / 1 XFAIL (same
  pre-existing deadlock).

If any new failure appears (i.e., not the documented pre-existing one),
pause. Do NOT tag until you have explained the new failure -- likely
candidates are:
- A tile-construction call site in `stream_switch/mod.rs` selecting
  the wrong `TileKind` variant.
- An archspec `stream_switch_model()` dispatch mismatch.
- A missed `PortLayout` reference still compiled into a release binary.

- [x] **Step 4: Run ISA test suite (~10 min)**

```bash
./scripts/isa-test.sh 2>&1 | tee /tmp/claude-1000/isa-subsys5.log
```

Expected: `4815/4815 PASS (100.0%); FAIL: 0`.

- [x] **Step 5: Update `NEXT-STEPS.md`**

Open `NEXT-STEPS.md`. Update the following fields:

- Top-of-file metadata:
  - `**Last updated:**` → `2026-04-21 (Phase 1b Subsystem 5 landed; Subsystem 7 up next)`
    (substitute today's actual date)
  - `**Latest tag:**` → `phase1-subsys-stream-switch`

- In the status table, change Subsystem 5's status from `**Up next**`
  to `**Done**` with the summary:

  ```
  | 5 | Stream Switch | `phase1-subsys-stream-switch` | **Done** | Two-method StreamSwitchModel trait (supports_deterministic_merge + topology) + StreamSwitchTopology carrier (3 TileStreamPorts sub-structs, for_tile(TileKind) accessor) + Aie2StreamSwitchModel. Dead-code PortLayout extension trait (231 LOC) deleted; 3 tests migrated to archspec. 6 tile-construction sites in stream_switch/mod.rs migrated through arch_handle::stream_switch_topology(). Direct archspec-constant consumers (routing.rs, stream_io.rs, state/) stay on direct access as AIE1-landing follow-ups. See docs/arch/stream-switch-model.md. |
  ```

  And change Subsystem 7's status line from `Pending` to `**Up next**`.

- Replace the "How to Pick Up Subsystem 5 (Stream Switch)" section
  with a "How to Pick Up Subsystem 7 (ISA Execute)" section. Mirror
  the shape of the existing section but point at:
  - New artifacts: `docs/arch/subsys5-audit.md` (completion reference),
    `docs/arch/stream-switch-model.md` (behavioral-seam template,
    alongside dma-model.md and lock-model.md).
  - Shaping questions specific to ISA Execute:
    - Semantic op divergence between AIE1/AIE2 (if any).
    - Intrinsic-handler seam vs per-op dispatch.
    - How much of the `vmac_routing.rs` (239KB) and `memory/mod.rs`
      (124KB) falls on each side of the seam.
    - What the `IsaExecutor` trait minimum surface looks like.

- [x] **Step 6: Commit all doc updates in a single "completion log" commit**

The docs updates from Task 6 (audit + design note) are still unstaged.
Stage them along with NEXT-STEPS.md:

```bash
git add docs/arch/subsys5-audit.md docs/arch/stream-switch-model.md NEXT-STEPS.md
git commit -m "$(cat <<'EOF'
docs: Subsystem 5 completion log + NEXT-STEPS points at Subsystem 7

Completion sections filled in docs/arch/subsys5-audit.md and
docs/arch/stream-switch-model.md: task-by-task commit history,
final test counts (xdna-emu 2684/0/5, archspec 297/0/2, bridge
matches Subsystem-4 character with the pre-existing
bd_chain_repeat_on_memtile HW failure, ISA 4815/4815), net code
delta (+180 archspec, -231 xdna-emu, ~-50 net).

NEXT-STEPS.md: "Last updated" and "Latest tag" bumped to
phase1-subsys-stream-switch. Subsystem 5 row in the status table
marked Done; Subsystem 7 (ISA Execute) becomes Up next. Pickup
guide rewritten to shape ISA Execute's brainstorming phase around
semantic-op divergence, intrinsic-handler seam shape, and the
large-file split concerns in vmac_routing.rs / memory/mod.rs.

Follow-ups flagged for AIE1-landing pass: direct archspec-constant
consumers (routing.rs 15 sites, stream_io.rs 6 const decls,
state/{compute,memtile}.rs ENABLE_BIT uses) migrate to the seam
then, not now. Carrier grows E/W / TRACE / DMA range fields
when the first consumer needs them.

Generated using Claude Code.
EOF
)"
```

- [x] **Step 7: Tag the subsystem**

```bash
git tag -a phase1-subsys-stream-switch -m "$(cat <<'EOF'
Phase 1b Subsystem 5 -- Stream Switch

Two-method StreamSwitchModel trait (supports_deterministic_merge +
topology) + StreamSwitchTopology carrier in xdna-archspec.
Dead-code PortLayout extension trait deleted from xdna-emu.
6 tile-construction call sites migrated through
arch_handle::stream_switch_topology().

See docs/arch/subsys5-audit.md and docs/arch/stream-switch-model.md
for full details.

Verification at tag:
  cargo test --lib: 2684 passed; 0 failed; 5 ignored
  cargo test -p xdna-archspec --lib: 297 passed; 0 failed; 2 ignored
  Bridge: matches phase1-subsys-locks character.
  ISA: 4815/4815 PASS (100.0%).

Generated using Claude Code.
EOF
)"
```

- [x] **Step 8: Verify the tag landed**

```bash
git tag -l 'phase1-subsys-*'
git show --stat phase1-subsys-stream-switch 2>&1 | head -30
```

Expected: new tag `phase1-subsys-stream-switch` in the list; `git show`
displays the commit message and stats.

- [x] **Step 9: Final post-tag cleanup sweep**

Run the global test suites one more time from a clean state to confirm
nothing regressed between the gate and the tag:

```bash
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test --lib 2>&1 | tail -3
PATH=/home/triple/npu-work/llvm-aie/build/bin:$PATH cargo test -p xdna-archspec --lib 2>&1 | tail -3
```

Expected: 2684 / 297 unchanged.

---

## Rollback plan

If anything goes sideways after the tag:

```bash
# Before pushing: reset the branch and delete the tag.
git reset --hard phase1-subsys-locks
git tag -d phase1-subsys-stream-switch

# After pushing: don't force-push; create a follow-up revert commit
# that re-adds port_layout.rs and removes the new seam, then re-tag
# a new `-retry` tag.
```

Rollback criteria (any of these):
- A bridge HW test that was passing at `phase1-subsys-locks` is newly
  failing at the tag candidate.
- `cargo test --lib` or `cargo test -p xdna-archspec --lib` regresses.
- A release-build warning indicates a visibility or import misalignment.

Not rollback criteria:
- The pre-existing `bd_chain_repeat_on_memtile` deadlock (unchanged).
- Bridge-side timeouts on `dma_task_large_linear` or
  `objectfifo_repeat/init_values_repeat` (pre-existing, see
  NEXT-STEPS.md).

---

## Plan self-review checklist

Before handing this plan off, verify:

- [x] All 7 tasks have concrete code blocks, not placeholders.
- [x] Type signatures for `StreamSwitchModel`, `StreamSwitchTopology`,
  `TileStreamPorts`, `for_tile` match across all tasks.
- [x] The `for_tile` accessor takes `&self, TileKind` in every
  reference (no drift to `&TileKind` or similar).
- [x] `&'static [u8]` (not `&'static [(AieRtPortType, u8)]`) for
  port-array fields -- matches the generated `gen_stream_ports.rs`.
- [x] Commit messages follow the convention (`refactor:`,
  `feat(archspec):`, `test(archspec):`, `docs:` prefixes; no emoji;
  ends with `Generated using Claude Code.`).
- [x] Test-count invariants: 2687 -> 2684 xdna-emu, 282 -> 297 archspec
  (baseline + 11 new archspec tests + 3 migrated + 1 dispatch test).
- [x] Drift-detection test is added (locked to generated constants).
- [x] No task introduces new runtime state in xdna-emu (seam is
  construction-time only).
