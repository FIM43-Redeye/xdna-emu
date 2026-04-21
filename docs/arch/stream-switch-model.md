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
