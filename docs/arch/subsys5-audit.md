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

Landed 2026-04-22. Tag: `phase1-subsys-stream-switch` (set in Task 7).

### Commits (Task 1 through tag)

```
cfdca21  docs: Subsystem 5 audit + stream-switch-model design note scaffolds
48fb17d  feat(archspec): StreamSwitchModel trait + StreamSwitchTopology + Aie2StreamSwitchModel
70f87cc  feat(archspec): ArchConfig::stream_switch_model() accessor
18fc1c5  refactor: migrate tile-construction call sites to StreamSwitchTopology
d776dd0  refactor: collapse dead-code PortLayout extension trait
6dab467  test(archspec): restore slave-range assertions dropped during port_layout migration
```

(The final "docs: Subsystem 5 completion log" commit added in Task 7 will
land this audit fill-in along with the NEXT-STEPS overhaul.)

### Verification (at tag)

- `cargo test --lib`: 2684 passed; 0 failed; 5 ignored.
- `cargo test -p xdna-archspec --lib`: 297 passed; 0 failed; 2 ignored.
- `cargo build --release`: clean (verified in Task 7).
- FFI cdylib rebuild (`cargo build -p xdna-emu-ffi`): clean (verified in Task 7).
- Bridge `--no-hw -v add_one_cpp_aiecc`: Chess and Peano PASS (verified in Task 4 + Task 7).
- Full HW bridge: matches phase1-subsys-locks character (pre-existing
  `bd_chain_repeat_on_memtile` EMU deadlock is the only HW failure;
  verified in Task 7).
- ISA test suite: 4815/4815 PASS (100.0%); FAIL: 0 (verified in Task 7).

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
  - Task 2 quality-review minor flags from Subsystem 5 itself:
    (a) `for_tile_memtile_dispatches_correctly` in `stream_switch/mod.rs`
    asserts only one field (compute and shim tests each assert two);
    (b) `aie2_stream_switch_model_topology_returns_static` is a
    pointer-equality sanity check that can never fail without also
    failing to compile. Both are cosmetic; fix if a Phase 2 hygiene
    pass touches the file.
- **Subsystem 7 (ISA Execute):** see `NEXT-STEPS.md` pickup guide
  (written in Task 7 at tag time).
