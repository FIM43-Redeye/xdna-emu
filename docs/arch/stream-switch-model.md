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

All entries below reflect state as of the `phase1-subsys-stream-switch` tag.

| Data/code | Module | Source |
|---|---|---|
| `StreamSwitchModel` trait (2 methods) + `StreamSwitchTopology` + `TileStreamPorts` carrier | `xdna_archspec::stream_switch` | Emulator design |
| `Aie2StreamSwitchModel` concrete impl + `AIE2_STREAM_SWITCH_MODEL` + `AIE2_STREAM_SWITCH_TOPOLOGY` statics | `xdna_archspec::aie2::stream_switch_model` | aie-rt xaiemlgbl_reginit.c + AM025 register DB JSON |
| `ArchConfig::stream_switch_model()` accessor | `xdna_archspec::runtime` | Dispatches on `Architecture` |
| `arch_handle::stream_switch_topology()` | `xdna_emu::device::arch_handle` | Process-global `OnceLock` cache |
| `StreamSwitch` struct + routing state | `xdna_emu::device::stream_switch` | Unchanged (traits describe, don't hold state) |

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

## Completion (2026-04-22)

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
  N/S range constants per tile kind. A follow-up fix-up commit
  (`6dab467`) restored 10 slave-side E/W/SOUTH range assertions and
  5 arithmetic count checks that the initial test migration dropped;
  coverage for those generated constants now lives in
  `test_npu1_port_ranges_migrated`.

Verification: `cargo test --lib` = 2684 passed / 0 failed / 5 ignored;
archspec = 297 passed / 0 failed / 2 ignored; full HW bridge matches
phase1-subsys-locks character (Task 7); ISA 4815/4815 PASS (Task 7).
