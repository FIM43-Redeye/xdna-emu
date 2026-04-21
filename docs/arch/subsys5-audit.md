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
