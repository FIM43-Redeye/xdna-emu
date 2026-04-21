# Subsystem 5 -- Stream Switch -- Design

**Subsystem:** 5 of 8 (Phase 1b of the device-family refactor)
**Date:** 2026-04-21
**Parent refactor:** [docs/superpowers/specs/2026-04-16-device-family-refactor-design.md](2026-04-16-device-family-refactor-design.md)
**Prior subsystem:** [docs/superpowers/specs/2026-04-21-subsys4-locks-design.md](2026-04-21-subsys4-locks-design.md)
**Planned tag:** `phase1-subsys-stream-switch`

---

## Goal

Introduce a `StreamSwitchModel` trait seam in `xdna-archspec`, collapse
the stale runtime-side `PortLayout` extension trait
(`src/device/port_layout.rs`), and expose per-tile-kind stream-switch
port topology as a data carrier (`StreamSwitchTopology`) that lives in
archspec. The trait carries exactly two items: one feature flag
(`supports_deterministic_merge`, the one real arch divergence surfaced
by the aie-rt audit) and one topology accessor returning
`&'static StreamSwitchTopology`. After this subsystem, stream-switch
port layout has a single entry point (`arch_handle::stream_switch_topology()`
in xdna-emu, or directly through `ArchConfig::stream_switch_model().topology()`),
the `PortLayout` extension trait deletes, and its three tests migrate
to archspec.

## Non-goals

- **No AIE1 or AIE2P `StreamSwitchModel` implementation.** Subsystem 5
  ships the trait, `StreamSwitchTopology` carrier, `TileStreamPorts`
  per-tile-kind struct, `Aie2StreamSwitchModel` impl, and the PortLayout
  collapse. Populating AIE1 (no deterministic merge; different
  compute/shim port counts per `xaiegbl_reginit.c`; no MemTile) and
  AIE2P (expected to match AIE2 1:1) is orthogonal future work.
- **No port-validity enforcement.** aie-rt's per-tile-kind
  `_XAieMl_*_StrmSwCheckPortValidity` functions (AIE2) enforce which
  slave ports can route to which master ports at CDO generation time.
  The emulator trusts the CDO it loads -- the toolchain already
  validated route legality. Adding validity enforcement is a separate
  future feature, not a device-family refactor concern.
- **No packet-routing or circuit-routing behavioral change.** The
  aie-rt audit confirmed packet routing mechanisms (slot count per
  slave port, arbiter count, msel selection, header format) are
  invariant across AIE1 and AIE2. Both use the same shared
  `xaie_ss.c` path with no runtime arch-dispatch. `supports_packet_routing`
  is deliberately **not** on the trait today -- including a flag whose
  only valid value is `true` is ceremony. If a future arch narrows
  packet-routing support, the flag appears then.
- **No change to xdna-emu's per-tile `StreamSwitch` struct.** That
  struct in `src/device/stream_switch/mod.rs` holds runtime state
  (slots, arbiter locks, priority pointers, packet-routing
  state). It stays. Only how it sources its port-layout data
  changes.
- **No `build.rs` change.** The generated port arrays
  (`COMPUTE_MASTER_PORTS` etc.) and stream-switch range constants
  (`compute::NORTH_MASTER_START` etc.) stay exactly where they are.
  The `StreamSwitchTopology` static aggregates them by hand, with a
  drift-detection test catching any divergence. Same strategy used
  for `AIE2_LOCK_VALUE_LAYOUT` in Subsystem 4.
- **No hot-path dispatch change.** `stream_switch_topology()` returns
  `&'static StreamSwitchTopology`, cached in `arch_handle::stream_switch_topology()`
  via `OnceLock`. Per-cycle call sites get a pure data-struct
  reference with no vtable hops.
- **No runtime behavior change for AIE2.** The trait is a dispatch
  seam; current AIE2 call sites should produce byte-identical
  decisions before and after. The test suite catches any drift.
- **No second-arch implementation during the refactor.** Phase 1
  ground rule.
- **No `AieRtPortType` API change.** The carrier preserves the enum
  type (`&'static [(AieRtPortType, u8)]`) across the seam; `PortLayout`'s
  `&'static [u8]` return type is a narrower convenience that goes away
  with the extension trait.

---

## Context

Subsystem 4 (Locks) landed at `phase1-subsys-locks` (`3296bdd`).
Baselines at that tag: `cargo test --lib` = 2687 pass, 0 fail, 5
ignored; `cargo test -p xdna-archspec --lib` = 282 pass, 0 fail, 2
ignored. `cargo build --release` clean. Bridge smoke green. ISA
4815/4815. Full HW bridge matches Subsystem 3's character (one
pre-existing `bd_chain_repeat_on_memtile` failure on each compiler).

Stream switch is the fifth of eight Phase 1b subsystems, and the
second behavioral seam (DMA was the first; Locks had a narrow
behavioral surface). Per the parent refactor's pass order (2 -> 3 ->
4 -> 5 -> 7 -> 8, with 6 having run out-of-order to unblock 1), this
subsystem is the last before ISA Execute (the largest remaining).

### What the Explore audit found

A single Explore-agent sweep against the aie-rt stream-switch
implementation
(`driver/src/stream_switch/xaie_ss.c` shared,
`xaie_ss_aie.c` for AIE1, `xaie_ss_aieml.c` for AIE2,
`xaiegbl_reginit.c` for AIE1 register tables,
`xaiemlgbl_reginit.c` for AIE2 register tables) surfaced seven
divergence facts:

| # | Feature | AIE1 | AIE2 | Source |
|---|---|---|---|---|
| 1 | Packet routing | Full support (slots/arbiter/msel) | Full support, identical mechanisms | Both use `XAie_StrmPktSw*` APIs in shared `xaie_ss.c`; no runtime arch-dispatch |
| 2 | Deterministic merge | **Unavailable** (`DetMergeFeature = XAIE_FEATURE_UNAVAILABLE`) | Available on all tile types (2 arbiters, 4 positions each) | `xaiegbl_reginit.c` vs `xaiemlgbl_reginit.c` |
| 3 | Port-count deltas | Compute: 2 CORE / 2 DMA / 1 CTRL / 2 FIFO; Shim: 0 CORE / 0 DMA / 1 CTRL / 2 FIFO | Compute: 1 CORE / 2 DMA / 1 CTRL / 1 FIFO; Shim: 0 CORE / 0 DMA / 1 CTRL / 1 FIFO | `AieTileStrmMstr/Slv` vs `AieMlTileStrmMstr/Slv` in `xaie_ss.c` |
| 4 | MemTile existence | Absent (AIE1 = Versal; no MemTile) | Present: DMA x6, CTRL x1, South x6, North x6 | MemTile only in AIE2 |
| 5 | Port-validity rules | Full crossbar (any slave -> any master) | Restrictive per tile-type (e.g., Compute TRACE -> FIFO/SOUTH/DMA0 only; MemTile TRACE -> SOUTH/DMA5 only) | `_XAie_StrmSwCheckPortValidity` (AIE1) vs `_XAieMl_*_StrmSwCheckPortValidity` (AIE2). Programming-time concern, not emulated today. |
| 6 | Shared-code branches | None (all divergence baked at init via per-arch `XAie_StrmMod` structs) | None | `xaie_ss.c` has no runtime `DevInst->DevProp.DevGen` checks for stream switch |
| 7 | South-slave delta | Compute: 6 ports; Shim: 8 ports (2-port delta) | Same 6 / 8, orderings preserved | `xaiegbl_reginit.c:250, 337` / `xaiemlgbl_reginit.c` (analogue) |

Of these seven, only row 2 (deterministic merge) is genuine behavioral
divergence that belongs on the trait today. Rows 3/7 (port-count /
south-slave-count deltas) are topology data already present per-arch
in the archspec build.rs-generated arrays -- they populate the carrier
struct per arch, no trait method required. Row 4 (MemTile absence)
is handled architecture-level: AIE1's `ArchConfig` never produces
`TileKind::Mem`. Rows 1/5/6 are non-divergences for the emulator's
current scope. Row 5 would become a trait concern if port-validity
enforcement lands.

### Pinned-to-xdna-emu items inherited from Subsystem 1

Subsystem 1 (Registers & Memory Map) partially migrated the
stream-switch data to archspec: `xdna_archspec::aie2::
{COMPUTE,MEMTILE,SHIM}_{MASTER,SLAVE}_PORTS` (the port-type arrays,
generated from AM025 register names) and
`xdna_archspec::aie2::stream_switch::{compute,mem_tile,shim}::{NORTH,SOUTH,EAST,WEST}_{MASTER,SLAVE}_{START,END}`
(the port-range constants) are archspec-resident today. The
runtime-side `PortLayout` extension trait on
`src/device/port_layout.rs` reads through to those generated consts;
its module-level doc-comment claims the data must stay runtime-side
because "the archspec workspace crate does not run the main crate's
build.rs," but that claim has been stale since Subsystem 1 moved the
codegen out of xdna-emu's `build.rs` into archspec's own `build.rs`.
The data is already in archspec; only the API-shape migration is
outstanding.

The six `PortLayout` methods to migrate: `master_ports`,
`slave_ports`, `north_master_range`, `south_master_range`,
`north_slave_range`, `south_slave_range`.

### Call-site inventory (xdna-emu)

Verified by `grep PortLayout` and `grep 'xdna_archspec::aie2::(stream_switch|COMPUTE_MASTER_PORTS|...)'`
at spec-writing time:

**`PortLayout` extension-trait consumers: zero.** No file in xdna-emu
outside `src/device/port_layout.rs` itself imports or calls any
method of `PortLayout`. The extension trait is dead code. Deleting
it is a self-contained change: remove the file + the `pub mod
port_layout;` line in `src/device/mod.rs` + three stale doc-comment
references in `crates/xdna-archspec/src/runtime.rs`.

**Direct archspec-constant consumers (7 files, ~30 sites):** these
bypass `PortLayout` and read `xdna_archspec::aie2::*` constants
directly. In scope for Subsystem 5 migration:

- `src/device/stream_switch/mod.rs` (6 sites): tile construction
  in `StreamSwitch::new_compute` / `new_memtile` / `new_shim`,
  via `build_ports_from_spec(xdna_archspec::aie2::*_PORTS, ...)`.
  These are the natural consumers of `StreamSwitchModel::topology()`
  and migrate through the seam.

Out of scope for Subsystem 5 (stay on direct archspec access; flagged
as follow-ups for AIE1-landing pass):

- `src/device/dma/stream_io.rs` (~6 sites): `pub const MASTER_PORT_COUNT
  = COMPUTE_MASTER_PORTS.len();` and similar const declarations. These
  are build-time expressions; migrating to the runtime seam would lose
  const-eval. AIE2-specific by design.
- `src/device/array/routing.rs` (~15 sites): direct use of
  `compute::NORTH_MASTER_START as usize` etc. for inter-tile pipeline
  index math. Migration is mechanical (`topo.for_tile(kind).north_master.0`)
  but the 15-site churn in tight numerical code is defer-able -- the
  direct constants still work on AIE2.
- `src/device/state/{compute,memtile}.rs` (2 sites each): `use
  xdna_archspec::aie2::stream_switch::{ENABLE_BIT, SLAVE_SELECT_MASK};`.
  These are stream-switch config bits (register-layout constants,
  not port-layout), orthogonal to the port-data seam.
- `src/interpreter/test_runner.rs`, other test-only imports: test
  code, not product migration concern.

The Subsystem 5 seam is ready for AIE1's port-count variance; the
direct consumers can migrate to it lazily as AIE1-landing work.
This is a more conservative scope than Locks (which migrated all 6
of its xdna-emu call sites) but tracks the real need: the dead
code deletes, the one real AIE2/AIE1-diverging API (tile
construction) routes through the seam, and widely-spread
AIE2-only register-layout constant use stays put until necessary.

Tests currently in `src/device/port_layout.rs` (3 tests, all migrate
to archspec):
- `test_npu1_port_layouts` -- port-count assertions per tile kind.
- `test_npu1_port_ranges` -- range-constant assertions per tile kind.
  Note: assertions checking `cfg.south_master_range(ShimNoc) == (0, 0)`
  preserve the sentinel after migration (`AIE2_STREAM_SWITCH_TOPOLOGY.shim.south_master == (0, 0)`).
  See "Carrier semantics for shim south" below.
- `test_shim_pl_same_as_shim_noc` -- `ShimPl`/`ShimNoc` both route to
  the shim port arrays, now verified via `topo.for_tile(ShimPl)` ==
  `topo.for_tile(ShimNoc)`.

### Carrier semantics for shim south

The archspec range constants expose shim south-facing port ranges
(`shim::SOUTH_MASTER_START = 2`, `SOUTH_MASTER_END = 7` in the
register layout, because shim ports 2-7 are the master-side
south-facing interface that connects to the external NoC). The
dead `PortLayout::south_master_range(ShimNoc)` returned `(0, 0)`
instead, modeling "no intra-array south neighbor" rather than the
register-layout range.

With `PortLayout` deleted and no consumers depending on either
semantic, the carrier preserves the `(0, 0)` sentinel for
`shim.south_master` / `shim.south_slave`. Rationale: (a) the three
migrated tests keep their assertion values, minimizing churn;
(b) the register-layout range is still reachable via the generated
`xdna_archspec::aie2::stream_switch::shim::SOUTH_MASTER_START`
constant for code that needs it (and existing callers in
`routing.rs` use that path already); (c) the tile-topology question
"does this tile have an intra-array south neighbor?" is better
answered by `ArchConfig::tile_kind(col, row - 1)` on the topology
seam (Subsystem 2), not by poking at port ranges. If a future
consumer needs the register-layout shim-south range through the
carrier, the carrier grows a second field (`shim.south_master_registers:
(u8, u8)`) without breaking current behavior.

---

## Design

### Principle: shape vs values, applied to stream switch

Subsystem 3 (DMA) introduced a 9-method trait because DMA has many
feature differences (task queue, interleave, compression, BD
iteration). Subsystem 4 (Locks) introduced a 3-method trait because
lock *value width* is the only data that varies and two AIE2-only
features are boolean flags. Subsystem 5 (Stream Switch) introduces
a **2-method trait** because only one behavioral flag diverges across
arches (deterministic merge), and the topology data naturally groups
into a single carrier (16 effective fields grouped into three
`TileStreamPorts` sub-structs).

Even narrower than Locks. The narrower trait surface reflects the
narrower variance.

Concretely:

- **AIE2 (NPU1/NPU4/NPU5/NPU6):** Deterministic merge available on
  all three tile types (compute, memtile, shim). Restrictive
  per-tile-kind port-validity rules (not emulated). Port counts:
  compute = 23 master / 25 slave; memtile = 17 master / 18 slave;
  shim = 22 master / 23 slave.
- **AIE1 (Versal, e.g., xcvc1902):** No deterministic merge. Full
  crossbar validity. No MemTile. Different compute port counts (per
  aie-rt `AieTileStrmMstr/Slv` tables).
- **AIE2P:** expected to match AIE2's trait values 1:1.

### The trait surface

```rust
pub trait StreamSwitchModel: Send + Sync + core::fmt::Debug {
    fn supports_deterministic_merge(&self) -> bool;
    fn topology(&self) -> &'static StreamSwitchTopology;
}
```

Two methods, "coarse first":

- One feature flag covers the one behavioral divergence the emulator
  can observe today (AIE1 lacks deterministic-merge registers).
- `topology()` returns a single data carrier bundling all per-tile-kind
  port layout.

Not on the trait:

- **Port-validity check** (`fn validate_route(&self, kind, slave, master) -> bool`):
  defer until emulated. CDO is trusted today.
- **Packet-slot count / arbiter count**: invariant across arches
  (4 slots/slave, 8 arbiters/switch). Constants live on the carrier
  only if they later diverge.
- **`supports_packet_routing`**: invariant `true` across arches;
  ceremony.

### The topology carrier

```rust
#[derive(Debug, Clone, Copy)]
pub struct TileStreamPorts {
    pub master_ports: &'static [(AieRtPortType, u8)],
    pub slave_ports: &'static [(AieRtPortType, u8)],
    pub north_master: (u8, u8),    // (start, end) inclusive
    pub south_master: (u8, u8),    // (0, 0) sentinel if N/A (shim)
    pub north_slave:  (u8, u8),
    pub south_slave:  (u8, u8),    // (0, 0) sentinel if N/A (shim)
}

#[derive(Debug, Clone, Copy)]
pub struct StreamSwitchTopology {
    pub compute: TileStreamPorts,
    pub memtile: TileStreamPorts,
    pub shim:    TileStreamPorts,
}

impl StreamSwitchTopology {
    pub fn for_tile(&self, tile: TileKind) -> &TileStreamPorts {
        match tile {
            TileKind::Compute => &self.compute,
            TileKind::Mem => &self.memtile,
            TileKind::ShimNoc | TileKind::ShimPl => &self.shim,
        }
    }
}
```

Three named sub-structs of six fields each, one accessor. No forwarding
per-field accessors on the parent: call sites go through
`topo.for_tile(tile).master_ports` (explicit, readable). The `(0, 0)`
sentinel for shim south fields preserves the existing `PortLayout`
convention (current `south_master_range(ShimNoc) == (0, 0)`, asserted
in `test_npu1_port_ranges`).

### AIE2 topology static

```rust
pub static AIE2_STREAM_SWITCH_TOPOLOGY: StreamSwitchTopology = StreamSwitchTopology {
    compute: TileStreamPorts {
        master_ports: crate::aie2::COMPUTE_MASTER_PORTS,
        slave_ports:  crate::aie2::COMPUTE_SLAVE_PORTS,
        north_master: (compute::NORTH_MASTER_START, compute::NORTH_MASTER_END),
        south_master: (compute::SOUTH_MASTER_START, compute::SOUTH_MASTER_END),
        north_slave:  (compute::NORTH_SLAVE_START,  compute::NORTH_SLAVE_END),
        south_slave:  (compute::SOUTH_SLAVE_START,  compute::SOUTH_SLAVE_END),
    },
    memtile: TileStreamPorts {
        master_ports: crate::aie2::MEMTILE_MASTER_PORTS,
        slave_ports:  crate::aie2::MEMTILE_SLAVE_PORTS,
        north_master: (mem_tile::NORTH_MASTER_START, mem_tile::NORTH_MASTER_END),
        south_master: (mem_tile::SOUTH_MASTER_START, mem_tile::SOUTH_MASTER_END),
        north_slave:  (mem_tile::NORTH_SLAVE_START,  mem_tile::NORTH_SLAVE_END),
        south_slave:  (mem_tile::SOUTH_SLAVE_START,  mem_tile::SOUTH_SLAVE_END),
    },
    shim: TileStreamPorts {
        master_ports: crate::aie2::SHIM_MASTER_PORTS,
        slave_ports:  crate::aie2::SHIM_SLAVE_PORTS,
        north_master: (shim::NORTH_MASTER_START, shim::NORTH_MASTER_END),
        south_master: (0, 0),    // shim has no south masters
        north_slave:  (shim::NORTH_SLAVE_START,  shim::NORTH_SLAVE_END),
        south_slave:  (0, 0),    // shim has no south slaves
    },
};

#[derive(Debug)]
pub struct Aie2StreamSwitchModel;

impl StreamSwitchModel for Aie2StreamSwitchModel {
    fn supports_deterministic_merge(&self) -> bool { true }
    fn topology(&self) -> &'static StreamSwitchTopology {
        &AIE2_STREAM_SWITCH_TOPOLOGY
    }
}

pub static AIE2_STREAM_SWITCH_MODEL: Aie2StreamSwitchModel = Aie2StreamSwitchModel;
```

### Runtime-side bridge

```rust
// crates/xdna-archspec/src/runtime.rs (addition)
pub trait ArchConfig: /* ... existing supertraits ... */ {
    // ... existing methods ...

    fn stream_switch_model(&self) -> &'static dyn StreamSwitchModel;
}

impl ArchConfig for ModelConfig {
    fn stream_switch_model(&self) -> &'static dyn StreamSwitchModel {
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
}
```

```rust
// src/device/arch_handle.rs (addition parallel to lock_value_layout)
pub fn stream_switch_topology() -> &'static StreamSwitchTopology {
    static CACHED: OnceLock<&'static StreamSwitchTopology> = OnceLock::new();
    CACHED.get_or_init(|| default_arch().stream_switch_model().topology())
}
```

### Module layout

**Archspec additions:**
- `crates/xdna-archspec/src/stream_switch/mod.rs` (new): `StreamSwitchModel`
  trait, `TileStreamPorts`, `StreamSwitchTopology` + `for_tile` accessor.
- `crates/xdna-archspec/src/aie2/stream_switch_model.rs` (new):
  `Aie2StreamSwitchModel`, `AIE2_STREAM_SWITCH_MODEL`,
  `AIE2_STREAM_SWITCH_TOPOLOGY`, drift-detection test.
- `crates/xdna-archspec/src/aie2/mod.rs`: re-export new module
  (`pub mod stream_switch_model;`).
- `crates/xdna-archspec/src/lib.rs`: re-export new top-level module
  (`pub mod stream_switch;`).
- `crates/xdna-archspec/src/runtime.rs`: add `fn stream_switch_model`
  to `ArchConfig` trait + impl on `ModelConfig`.

**xdna-emu changes:**
- `src/device/port_layout.rs`: **deleted** (~231 LOC).
- `src/device/mod.rs`: remove `pub mod port_layout`; remove any
  `use port_layout::PortLayout;` re-export.
- `src/device/arch_handle.rs`: add `stream_switch_topology()`
  accessor, mirroring `lock_value_layout()`.
- Call sites that previously wrote `cfg.master_ports(tile)` etc.
  migrate to `arch_handle::stream_switch_topology().for_tile(tile).master_ports`.

---

## Alternatives considered

### Approach B: flat 8-method trait, no carrier

```rust
pub trait StreamSwitchModel: Send + Sync + core::fmt::Debug {
    fn supports_deterministic_merge(&self) -> bool;
    fn master_ports(&self, tile: TileKind) -> &'static [(AieRtPortType, u8)];
    fn slave_ports(&self, tile: TileKind) -> &'static [(AieRtPortType, u8)];
    fn north_master_range(&self, tile: TileKind) -> (u8, u8);
    fn south_master_range(&self, tile: TileKind) -> (u8, u8);
    fn north_slave_range(&self, tile: TileKind) -> (u8, u8);
    fn south_slave_range(&self, tile: TileKind) -> (u8, u8);
}
```

**Rejected** because the carrier struct gives three properties the flat
trait doesn't:

1. **Topology is a usable type by itself.** Functions can take
   `&'static StreamSwitchTopology` (narrower interface, no
   behavioral-flag exposure). With the flat trait, the equivalent
   is `&dyn StreamSwitchModel` -- wider than needed.
2. **Shared data across arches.** If we later discover AIE2 and
   AIE2P have identical port layouts but differ on one flag, the
   carrier lets both point at one `AIE2_STREAM_SWITCH_TOPOLOGY`
   static and vary only the model impl. The flat trait forces
   per-arch redeclaration of all six layout methods.
3. **Caching is cheaper.** `arch_handle::stream_switch_topology()`
   caches `&'static StreamSwitchTopology` -- pure data, no vtable.
   With the flat trait, caching would be `&dyn StreamSwitchModel`,
   and every port lookup is one vtable hop through the cached
   reference.

Monomorphization is equivalent between the two approaches: both
allow `<M: StreamSwitchModel>` generics with per-concrete-type
inlining. The carrier approach additionally admits pure-data
function signatures where no trait is needed, which is a strictly
smaller codegen footprint.

The cost of the carrier is one extra dot at call sites
(`arch.stream_switch_model().topology().for_tile(tile)` vs
`arch.stream_switch_model().for_tile_master_ports(tile)`). In
practice call sites go through `arch_handle::stream_switch_topology()`
which is a single call -- same syntactic length as today's
`cfg.master_ports(tile)`.

### Approach C: behavioral-only trait, leave `PortLayout` runtime-side

Rejected because the `PortLayout` module's "data must stay
runtime-side" doc-comment has been stale since Subsystem 1 moved the
generated port arrays into archspec. Keeping `PortLayout` as a
separate extension trait on `ModelConfig` would create two public
entry points for one concept (stream-switch port data on the
extension trait; stream-switch behavior flags on the model), with
no real technical reason for the split. The whole refactor principle
is "one authoritative source per concept."

### Preemptive `supports_packet_routing` flag

Rejected because both AIE1 and AIE2 fully support packet routing
with identical slot/arbiter/msel mechanisms (aie-rt audit, shared
`xaie_ss.c` with no arch-dispatch). Adding a flag whose only valid
value is `true` is ceremony. If a future arch narrows the semantics,
the flag appears then.

### Build.rs codegen of the carrier

Rejected because build.rs already emits the per-field constants
(`COMPUTE_MASTER_PORTS`, `compute::NORTH_MASTER_START`, etc.).
Adding a second codegen path just duplicates the drift-risk surface.
A hand-written static aggregating the existing constants + a
drift-detection test is the same strategy used for
`AIE2_LOCK_VALUE_LAYOUT` in Subsystem 4 and works cleanly.

### Sub-trait split: `StreamSwitchTopology` vs `StreamSwitchBehavior`

Rejected for the same reason a similar split was rejected in
Subsystem 4: every real device has both axes linked -- no arch is
expected to have AIE2's topology with AIE1's feature flags. Two
traits for a 2-method surface is ceremony. If a later subsystem
introduces more independent axes of variance, a split becomes
justified then.

---

## What would AIE1 look like?

- `xdna_archspec::aie1::stream_switch_model::Aie1StreamSwitchModel`
  zero-sized struct + `AIE1_STREAM_SWITCH_MODEL` static.
- `supports_deterministic_merge()` returns `false` (aie-rt:
  `DetMergeFeature = XAIE_FEATURE_UNAVAILABLE`).
- `AIE1_STREAM_SWITCH_TOPOLOGY` populated from AIE1 port-count data:
  compute has 2 CORE / 2 FIFO (vs AIE2's 1/1), different shim counts
  per `xaiegbl_reginit.c` `AieTileStrmMstr/Slv` / `AieShimStrmMstr/Slv`
  tables. The numeric range constants differ correspondingly.
- `memtile` field on `AIE1_STREAM_SWITCH_TOPOLOGY` is the one
  ambiguity: AIE1 has no MemTile. Two resolutions possible at
  AIE1-landing time: (a) make `StreamSwitchTopology.memtile` an
  `Option<TileStreamPorts>` (forces all memtile call sites to
  `.unwrap()` or arch-gate; ~5-10 sites of churn); (b) sentinel --
  empty `master_ports: &[]` + zero ranges -- so `for_tile(TileKind::Mem)`
  returns benign-looking data that callers never reach on AIE1
  (since `TileKind::Mem` is never produced on an AIE1 array;
  memtile rows don't exist). **Decision deferred to AIE1 landing**,
  same pattern as `Lock::MIN_VALUE`/`MAX_VALUE` per-arch question
  in Subsystem 4.
- `ArchConfig::stream_switch_model()` adds a `Architecture::Aie` arm
  returning `&AIE1_STREAM_SWITCH_MODEL`.
- Drift-detection test added analogous to AIE2.

Call sites in xdna-emu require no changes: the N sites that read
`topology().for_tile(kind)` in this subsystem work the same on both
arches (they read the carrier fields as data, not as behavior).

---

## Testing

### Drift-detection (archspec-resident)

In `crates/xdna-archspec/src/aie2/stream_switch_model.rs`:

```rust
#[test]
fn aie2_topology_matches_generated_constants() {
    use crate::aie2::stream_switch::{compute, mem_tile, shim};

    assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.compute.master_ports,
               crate::aie2::COMPUTE_MASTER_PORTS);
    assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.compute.slave_ports,
               crate::aie2::COMPUTE_SLAVE_PORTS);
    assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.compute.north_master,
               (compute::NORTH_MASTER_START, compute::NORTH_MASTER_END));
    // ... analogous assertions for all three tile kinds and all four ranges ...
    assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.shim.south_master, (0, 0));
    assert_eq!(AIE2_STREAM_SWITCH_TOPOLOGY.shim.south_slave,  (0, 0));
}
```

Catches any drift between the hand-written carrier aggregation and
the build.rs-generated per-field constants.

### Trait-shape test (archspec-resident)

```rust
#[test]
fn aie2_stream_switch_model_accessors() {
    assert!(AIE2_STREAM_SWITCH_MODEL.supports_deterministic_merge());
    let topo = AIE2_STREAM_SWITCH_MODEL.topology();
    assert_eq!(topo.for_tile(TileKind::Compute).master_ports.len(), 23);
    assert_eq!(topo.for_tile(TileKind::Mem).master_ports.len(), 17);
    assert_eq!(topo.for_tile(TileKind::ShimNoc).master_ports.len(), 22);
}
```

### Migrated tests

The three tests currently in `src/device/port_layout.rs` move to
archspec (`stream_switch/mod.rs` tests or `aie2/stream_switch_model.rs`
tests). Their assertions transform from
`cfg.master_ports(TileKind::ShimNoc).len()` to
`AIE2_STREAM_SWITCH_TOPOLOGY.shim.master_ports.len()` etc. No
assertion value changes.

### Runtime behavior

No new unit tests required on the xdna-emu side; the migrated call
sites should produce identical data reads. The bridge and ISA gates
catch any runtime-behavior drift.

### Global gates (at tag)

- `cargo test --lib` green at every commit (xdna-emu).
- `cargo test -p xdna-archspec --lib` green at every commit.
- `cargo build --release` clean at tag.
- `cargo build -p xdna-emu-ffi` (bridge FFI cdylib) clean at tag.
- Bridge smoke: `./scripts/emu-bridge-test.sh --no-hw -v add_one_cpp_aiecc`
  PASS on both Chess and Peano.
- Full bridge: matches phase1-subsys-dma / phase1-subsys-locks
  character (pre-existing `bd_chain_repeat_on_memtile` failure is
  the only HW failure).
- ISA test suite: 4815/4815 PASS.

Baseline at `phase1-subsys-locks`:
- xdna-emu: 2687 pass / 0 fail / 5 ignored
- archspec: 282 pass / 0 fail / 2 ignored

Expected at `phase1-subsys-stream-switch`:
- xdna-emu: ~2684 pass (3 migrated tests leave)
- archspec: ~287 pass (3 migrated tests arrive + 2 new drift/shape tests)
- Net test count preserved; maybe +2 for new drift/shape coverage.

---

## Future direction

Post-Phase-1, when a third arch (AIE1) lands:

- **AIE1 plug-in:** `Aie1StreamSwitchModel` + `AIE1_STREAM_SWITCH_TOPOLOGY`
  fill in. Decision on `memtile: Option<_>` vs sentinel happens
  at this time.
- **Port-validity enforcement:** if the emulator grows to reject
  invalid routes (rather than trusting CDO), `fn validate_route`
  joins the trait. AIE1's `false` for `supports_deterministic_merge`
  plus per-tile-kind validity tables form a consistent story.
- **Generic-type-parameter monomorphization:** hot call sites (if
  any emerge in stream switch; most are construction-time today)
  switch from `&'static dyn StreamSwitchModel` to
  `<S: StreamSwitchModel>`. The carrier stays the same.
- **`arch_handle` generalization:** the module currently exposes
  `lock_value_layout()` and (post-Subsystem 5) `stream_switch_topology()`.
  If Subsystem 7 (ISA Execute) or Subsystem 8 (Parser) need similar
  process-global handles, extend the module; if one subsystem
  accumulates enough handles to justify a sub-module, split then.

---

## Follow-ups flagged (not blocking)

- **Direct archspec-constant consumers migrate to the seam at AIE1
  landing.** Specifically: `dma/stream_io.rs` const declarations,
  `array/routing.rs` range-math sites, `state/compute.rs` +
  `state/memtile.rs` ENABLE_BIT / SLAVE_SELECT_MASK uses. On AIE2
  they work correctly via direct constant access; on AIE1 they
  would silently read AIE2 data and produce wrong routing decisions.
  The AIE1-landing pass either (a) migrates them through
  `stream_switch_topology()` (runtime-dispatched), or (b) grows
  per-arch `aie1::stream_switch` modules for const-expression
  consumers and has them select by `cfg!(arch)` or build-script
  detection. Deferring this to AIE1 landing keeps Subsystem 5
  scope tight.
- Extending the carrier with E/W master/slave ranges, TRACE_SLAVE,
  and DMA_MASTER/SLAVE ranges for full coverage of the
  `routing.rs` usage pattern. Not done today because
  `routing.rs` is not migrating in this subsystem; adding fields
  no one reads is ceremony. Grow the carrier when the first
  consumer needs them.
- `OnceLock<&'static StreamSwitchTopology>` double-indirection in
  `arch_handle.rs` (parallel to the Subsystem 4 follow-up for
  `lock_value_layout()`) could simplify to `OnceLock<StreamSwitchTopology>`
  by value, since `StreamSwitchTopology: Copy`. Phase 2 hygiene.
- Pre-existing Subsystem 6-era rot in `tests/arch_constants.rs`,
  `examples/run_add_test.rs`, `examples/bdd_validate.rs`, and
  generated-file warnings. Not Subsystem 5's scope.
- `src/device/port_layout.rs`'s stale module-level doc-comment about
  build.rs access disappears with the file; no standalone follow-up.

---

## Success criteria

- `StreamSwitchModel` trait in `xdna_archspec::stream_switch`
  (2 methods: `supports_deterministic_merge`, `topology`):
  populated.
- `StreamSwitchTopology` data carrier + `TileStreamPorts` sub-struct
  with `for_tile(TileKind) -> &TileStreamPorts` accessor: populated.
- `Aie2StreamSwitchModel` concrete impl + `AIE2_STREAM_SWITCH_MODEL`
  static + `AIE2_STREAM_SWITCH_TOPOLOGY` static: populated.
- `ArchConfig::stream_switch_model()` accessor: populated,
  dispatches on `Architecture` (`Aie2` / `Aie2p` both return the
  AIE2 model static; `Aie` `unimplemented!()`s with the populate-me
  message).
- xdna-emu `src/device/port_layout.rs`: **deleted** entirely (all
  231 LOC), no source still imports `PortLayout`, no doc-comment
  referring to the extension trait.
- xdna-emu `src/device/stream_switch/mod.rs` tile-construction
  call sites (6: `new_compute`, `new_memtile`, `new_shim`
  `build_ports_from_spec` calls for masters + slaves) migrate from
  `xdna_archspec::aie2::*_PORTS` direct reads to
  `arch_handle::stream_switch_topology().for_tile(tile).master_ports`
  (and `slave_ports`).
- Other direct archspec-constant consumers (dma/stream_io.rs const
  expressions, array/routing.rs range-math sites, state/compute.rs
  & state/memtile.rs ENABLE_BIT/SLAVE_SELECT_MASK uses) **stay on
  direct constant access** per the call-site inventory scope
  decision. Flagged as follow-ups for AIE1-landing pass.
- `arch_handle::stream_switch_topology()` accessor in
  `src/device/arch_handle.rs` following the `lock_value_layout()`
  OnceLock pattern.
- Drift-detection test in `crates/xdna-archspec/src/aie2/stream_switch_model.rs`
  asserts `AIE2_STREAM_SWITCH_TOPOLOGY` agrees with the
  `COMPUTE_MASTER_PORTS`, `compute::NORTH_MASTER_START`, etc. generated
  constants for all three tile kinds and all four ranges.
- Trait-shape test asserts `supports_deterministic_merge` and
  `topology().for_tile()` per-tile-kind port-count values.
- Three tests migrated from `src/device/port_layout.rs` to archspec
  with equivalent assertions.
- `docs/arch/stream-switch-model.md` design note written (per-seam
  convention): trait surface, AIE1 projection, rejected approaches.
- All global gates pass at `phase1-subsys-stream-switch`:
  `cargo test --lib` green, `cargo build --release` clean, bridge
  smoke green, full bridge matches prior character, ISA 4815/4815.

---

## Net code delta (estimated)

- New in archspec: ~180 LOC (`stream_switch/` module with trait +
  carrier + accessors + tests, `aie2/stream_switch_model.rs` with
  impl + statics + drift/shape tests).
- Deleted in xdna-emu: ~231 LOC (entire `port_layout.rs`, including
  the three tests and doc-comments).
- Modified in xdna-emu:
  - `src/device/stream_switch/mod.rs`: 6 call-site rewrites (three
    `new_compute` / `new_memtile` / `new_shim` functions, each with a
    masters + slaves `build_ports_from_spec` call).
  - `src/device/arch_handle.rs`: gains `stream_switch_topology()`
    accessor (~10 LOC, parallel to existing `lock_value_layout()`).
  - `src/device/mod.rs`: drops `pub mod port_layout;` line.
  - `crates/xdna-archspec/src/runtime.rs`: three stale doc-comment
    references to `PortLayout` updated to describe the seam
    (lines 13, 63, 221 per current state).
- Net workspace LOC change: roughly neutral, maybe -30 to -50 LOC.
  Smaller scope than initially planned because `PortLayout` had no
  external consumers; migration concentrates on the one real consumer
  path (tile construction). Direct archspec-constant consumers that
  don't go through `PortLayout` stay in place; they're AIE2-specific
  by their current const-expression shape and migrate lazily at
  AIE1 landing.
