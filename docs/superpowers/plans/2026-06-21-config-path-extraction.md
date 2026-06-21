# Config-Path Extraction (Inference Engine Plan 2 of 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Plan 1's hand-authored structural ledger with one **auto-generated from the loaded binary** — a faithful Rust route-graph reconstruction + physical-config dump, a Python translator that orients trace-event pairs by graph reachability, and full automated Axis-2 HW validation that reproduces `add_one_using_dma`'s 5 stochastic roots from the *generated* ledger.

**Architecture:** Four tiers, bottom-up.
1. **Rust route-graph resolver** — factor the emulator's *static* routing resolution out of its per-word runtime stepping into a reusable `DeviceState::resolve_route_graph()`, validated against the dynamic enactment (the static graph must be a superset of the hops the simulation actually takes). This is the faithful home for route reconstruction: it reuses the same adjacency logic (`propagate_inter_tile`), the same `local_routes`/packet-slot config, and the same build-derived topology constants (`xdna_archspec::aie2::stream_switch`) the simulation uses — one source of truth, no Python re-implementation to drift.
2. **Rust config dump** — `examples/dump_config_json.rs` serializes the resolved route graph + per-tile port metadata + event-port-selection bindings + BD chains + lock pairings to JSON. A pure *quote of the loaded binary* (the spec's definition of structural support).
3. **Python config_path generator** — consumes the dump, computes reachability over the route graph, maps physical resources to trace event keys (via the existing `trace_capture.load_event_ids` vocabulary + the dumped event-port bindings), and emits the same ledger schema Plan 1 consumed. **Plan 2 wires the stream-route translation only**; BD-chain and lock-pairing *translation* are sequenced to a later plan (their physical dump lands now so that later plan is pure Python).
4. **Full Axis-2 HW validation** — the generated ledger replaces the hand-authored fixture in the HW smoke; the engine must re-derive the same 5 roots / 5 derives, closing Plan 1's "ledger is hand-authored" caveat.

**Tech Stack:** Rust (emulator + serde/serde_json, already in `Cargo.toml`; example binaries under `examples/`), Python 3.13 (pytest, flat colocated `tools/test_<module>.py`, `tools/conftest.py` on path; no new third-party deps).

## Global Constraints

These bind every task. Exact values, copied verbatim from the design spec
(`docs/superpowers/specs/2026-06-21-trace-inference-engine-design.md`) and Plan 1.

- **Orientation is reachability, never a naming heuristic.** A `config_path(parent, child)` edge may be emitted **only** when `child`'s resource is reachable from `parent`'s resource over the *directed route graph*, whose edges run in **true dataflow direction** (source → sink, exactly as the emulator enacts routing). Never orient by a hardcoded assumption like "MM2S is upstream of S2MM" or "input DMA precedes output DMA" — that is the kind of hardcoded shortcut CLAUDE.md forbids, and it produces backwards orientations on kernels with multiple DMA pairs. The dump and the graph share physical port identity, so reachability is correct regardless of the AIE master/slave DMA-naming convention.
- **Derive from the toolchain.** The route graph reuses the emulator's existing routing logic and the build-derived `xdna_archspec::aie2::stream_switch` topology constants. No hardcoded NSEW adjacency, no hardcoded port indices, no copied register layouts. The Rust resolver mirrors `propagate_inter_tile()` / `resolve_packet_route()` — it does not invent a parallel adjacency.
- **The dump is a pure quote of the binary.** `dump_config_json.rs` serializes only what is present in the loaded `DeviceState` after `apply_cdo`. No interpretation, no event-semantics — that lives in Python.
- **The generator is the new trust anchor.** With hand-authoring gone, every `config_path` leaf rests on the generator's correctness. The **generator-audit / spot-check** (Task C4) is a first-class deliverable, not a footnote (spec line 144: "the audit target is the generator + spot-check").
- **event_key = `col|row|pkt|name`**, pipe-delimited, in that order. `col` is **absolute** (decoder space, col 1 on `add_one_using_dma`). Build keys with the `trace_join._key(col, row, pkt, name)` pattern.
- **pkt_type semantics:** `{0: core, 1: memmod, 2: shim, 3: memtile}` (`trace_capture.PKT_TO_TILE_TYPE`). DMA events live in pkt 1 (memmod) and pkt 2 (shim); PORT_RUNNING events live in pkt 0 (core) and pkt 3 (memtile).
- **Event-name authority** is `trace_capture.load_event_ids(tile_type)` (parses the aie-rt events header). Never hardcode an event-name table.
- **Ledger schema (byte-compatible with Plan 1):** `{"entries": [{"cite": str, "a": event_key, "b": event_key, "kind": "route"|"bd"|"lock"|"identity"}]}`. Plan 2 emits only `kind: "route"`. `a` is the child (downstream), `b` is the parent (upstream) — verify against `inference/ledger.py` (`route` ⇒ `config_path(a, b, cite)`, meaning "a's producer is upstream of b's consumer"; read the loader to confirm the arg order before emitting).
- **anchor = `1|2|0|PERF_CNT_2`, eps = 2.0** (unchanged from Plan 1; this plan does not touch the verifier or eps).
- **Tests:** Rust unit tests inline (`#[cfg(test)]`) or under the module; Rust integration tests that need an xclbin go in `tests/` or as `#[test]` in the example's module guarded to skip when the fixture xclbin is absent. Python tests are flat colocated: `tools/test_config_extract_<module>.py`, importing from `config_extract.<module>` with `tools/conftest.py` already putting `tools/` on the path.
- **HW invocations** use `env -u XDNA_EMU -u XDNA_EMU_RUNTIME` (chess = ground truth). HW-gated tests skip unless `XDNA_HW_SMOKE=1`. Never run two HW suites concurrently.
- **After any Rust change, `cargo test --lib`** (and `cargo build` for examples). The FFI `.so` is **not** needed for this plan — the dump example and the offline Python translator do not go through the XRT plugin.
- **Commit messages** end with the two-line trailer (`Generated using Claude Code.` / `Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh`); no emoji.

## Scope amendment (2026-06-21, mid-execution — Maya-approved)

Execution of B1 surfaced, and a hands-on investigation **confirmed with decoded
bytes**, that per-run trace configuration — specifically the memtile
`Stream_Switch_Event_Port_Selection` registers — is written by the **runtime
instruction stream (`insts.bin`)**, not the xclbin CDO. (Decoded
`add_one/chess/insts.bin`: `word[66]=0x001b0f00` writes memtile EvtPortSel_0,
etc.; the trace patcher only overrides `Trace_Event0/1`.) A CDO-only dump
therefore shows `event_port_selection` all-null on every tile, which is why the
PORT_RUNNING→physical-port mapping was missing.

**Resolution (approved):** "quote of the loaded binary" now means **CDO + applied
instruction stream** — faithful, because that is literally what executes on the
NPU. Reuse the emulator's existing `ControlPacketProcessor` (the same machinery a
real run uses); do not hand-roll instruction parsing.

**Revised B/C-tier task list:**
- **B1 — DONE** (CDO dump scaffold: structs, loader, route_graph + ports +
  event_port_selection serialization). Code accepted.
- **B1b — NEW:** apply `insts.bin` after the CDO so `event_port_selection` /
  trace config populate; **verify the emulator models the `0xB0F00` control-packet
  write** into `tile.event_port_selection` (if it doesn't, that's a real emulator
  fidelity gap to surface, not paper over). Regenerate the fixture.
- **B2 — extended:** also dump the shim mux arrays (`shim_mux_mm2s_slaves` /
  `shim_mux_s2mm_masters`, already CDO-parsed) alongside BD chains / DMA channels /
  locks. For add_one: `MM2S_0→south-slave-2`, `S2MM_0→south-master-2`.
- **Tier C (C2)** resolves DMA events via the mux arrays (shim DMA is mux-routed,
  not a `PortType::Dma` SS port) and PORT_RUNNING via the now-populated
  `event_port_selection`. The route-graph reachability architecture is unchanged.

## Design decisions (resolving spec open questions)

The design spec (2026-06-21) deferred these to implementation planning; resolved here:

- **"The `config_path` derivation rule"** (spec line 338): a `config_path(child, parent)` edge is emitted iff (1) both events map to physical route-graph nodes, and (2) `parent`'s node reaches `child`'s node over the directed graph. The "rule" is **graph reachability**, computed in Python over the Rust-resolved graph. Its "verifier" (spec phrasing) is the generator-audit (C4) + the route-graph's own static-vs-dynamic validation (A5).
- **"The structural-ledger format and generator-audit workflow"** (spec line 345): format is the Plan-1 ledger schema (above). The generator-audit is C4 — every generated `cite` resolves to a concrete config location (a route-graph edge path), and the generated ledger loads through `inference/ledger.py` with `provenance_ok` holding.
- **Where route reconstruction lives:** the **emulator (Rust)**, as `DeviceState::resolve_route_graph()`. Confirmed with Maya — reconstructing in Python would duplicate the emulator's authoritative routing logic and risk silent drift; the Rust resolver is also a reusable capability (a future routing visualizer consumes the same graph).
- **Translation split:** Python owns physical→event-key translation; Rust stays a pure physical dump. Confirmed with Maya.
- **Packet vs circuit:** circuit routes are fully static (`local_routes`). Packet routes are static over *configured slots* (a superset of header-dependent edges). Task A3 builds both; Task A4's first execution step **confirms whether `add_one_using_dma`'s data path is circuit- or packet-switched** and records it. If circuit-only, the packet path gets synthetic unit tests in this plan and is HW-exercised when a packet-routed kernel enters the corpus (consistent with Maya's "do the dump now, sequence the rest").

---

## File Structure

**Rust (new):**
- `src/device/stream_switch/route_graph.rs` — `PortRef`, `RouteEdge`, `StreamRouteGraph` types (serde) + the static resolver functions. Declared from `src/device/stream_switch/mod.rs`.
- `examples/dump_config_json.rs` — the dump binary.

**Rust (modified):**
- `src/device/stream_switch/mod.rs` — `mod route_graph; pub use route_graph::*;` and the `StreamSwitch`-level helpers the resolver needs.
- `src/device/state/mod.rs` (or wherever `DeviceState` is defined) — `impl DeviceState { pub fn resolve_route_graph(&self) -> StreamRouteGraph }`.
- `src/device/array/routing.rs` — a gated enacted-hop recorder for the A5 validation (off by default).

**Python (new), package `tools/config_extract/`:**
- `tools/config_extract/__init__.py`
- `tools/config_extract/dump_model.py` — typed loader for the JSON dump (dataclasses mirroring the schema).
- `tools/config_extract/reachability.py` — directed graph + transitive reachability.
- `tools/config_extract/event_map.py` — physical resource → trace event_key resolution.
- `tools/config_extract/generator.py` — the config_path generator (emits the ledger) + `main()` CLI.
- Tests: `tools/test_config_extract_reachability.py`, `tools/test_config_extract_event_map.py`, `tools/test_config_extract_generator.py`.

**Python (modified):**
- `tools/test_inference_hw_smoke.py` — point the HW smoke at the generated ledger (Task D1).

**Docs/fixtures:**
- `tools/config_extract/fixtures/add_one_using_dma.config.json` — a captured dump used by offline Python tests (so C-tier tests don't require building/running the Rust example).
- `docs/superpowers/findings/2026-06-21-config-path-extraction-axis2.md` — the close-out finding (Task D2).

---

## Tier A — Rust route-graph resolver

### Task A1: Route-graph types

**Files:**
- Create: `src/device/stream_switch/route_graph.rs`
- Modify: `src/device/stream_switch/mod.rs` (add `mod route_graph; pub use route_graph::*;`)

**Interfaces:**
- Produces: `PortRef { col: u8, row: u8, port: u8, dir: PortDir, kind: String }`, `PortDir` (`Master`/`Slave`, serde lowercase), `RouteEdge { src: PortRef, dst: PortRef, kind: EdgeKind }`, `EdgeKind` (`InterTile`/`Circuit`/`Packet`, serde snake_case), `StreamRouteGraph { edges: Vec<RouteEdge> }` with `#[derive(Debug, Clone, Serialize, Deserialize)]`. `kind` on `PortRef` is the stringified `PortType` (e.g. `"north"`, `"south"`, `"dma"`, `"core"`, `"trace"`) so the dump is self-describing.
- Consumed by: A2, A3, A4 (construct edges), B1 (serialize), and the Python `dump_model.py` (deserialize the same JSON).

- [ ] **Step 1: Write the failing test** (in `route_graph.rs` under `#[cfg(test)]`)

```rust
#[test]
fn route_graph_serializes_round_trip() {
    let g = StreamRouteGraph {
        edges: vec![RouteEdge {
            src: PortRef { col: 1, row: 0, port: 12, dir: PortDir::Master, kind: "north".into() },
            dst: PortRef { col: 1, row: 1, port: 7, dir: PortDir::Slave, kind: "south".into() },
            kind: EdgeKind::InterTile,
        }],
    };
    let json = serde_json::to_string(&g).unwrap();
    let back: StreamRouteGraph = serde_json::from_str(&json).unwrap();
    assert_eq!(back.edges.len(), 1);
    assert_eq!(back.edges[0].src.port, 12);
    assert_eq!(back.edges[0].kind, EdgeKind::InterTile);
}
```

- [ ] **Step 2: Run it, verify it fails** — `cargo test -p xdna-emu route_graph_serializes_round_trip` → FAIL (types undefined).
- [ ] **Step 3: Implement the types** with the derives above. `PortDir` and `EdgeKind` get `#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]` and `#[serde(rename_all = "snake_case")]`. Add a helper `PortType::as_kind_str(&self) -> &'static str` (or a `From<PortType> for String`) returning the snake_case name, on whatever module owns `PortType` — derive it from the existing enum variants, do not hardcode a parallel list that can drift.
- [ ] **Step 4: Run it, verify it passes.**
- [ ] **Step 5: Commit** — `feat(#140): route-graph types for static route reconstruction`.

### Task A2: Static inter-tile adjacency

**Files:**
- Modify: `src/device/stream_switch/route_graph.rs`

**Interfaces:**
- Consumes: `xdna_archspec::aie2::stream_switch` constants (per-tile-kind `{NORTH,SOUTH,EAST,WEST}_MASTER_START/END` etc.), `TileKind`, array dims.
- Produces: `pub fn inter_tile_dest(src_kind: TileKind, src_col: u8, src_row: u8, master_port: u8, cols: u8, rows: u8) -> Option<PortRef>` — given a *master* port on a tile, returns the neighbor tile's *slave* `PortRef` it physically wires to, or `None` if that master index has no inter-tile destination (e.g. a DMA/core master, or an edge of the array).

This is the static distillation of `propagate_inter_tile()` (`src/device/array/routing.rs`, the per-direction blocks). **Mirror that function's index arithmetic exactly** — same constants, same 1:1 index mapping, same neighbor selection. Do not re-derive the mapping from first principles.

- [ ] **Step 1: Write the failing test** — pin the mappings the runtime uses. Read `propagate_inter_tile()` and `xdna_archspec::aie2::stream_switch` first to fill the exact expected indices; the cases below are the *shape* (replace the `?` with the real constants discovered from the source):

```rust
#[test]
fn inter_tile_shim_north_master_reaches_memtile_south_slave() {
    // Shim (1,0) north master K -> MemTile (1,1) south slave K' per archspec constants.
    let d = inter_tile_dest(TileKind::Shim, 1, 0, /*shim NORTH_MASTER_START*/ ?, 5, 6).unwrap();
    assert_eq!((d.col, d.row), (1, 1));
    assert_eq!(d.dir, PortDir::Slave);
    // d.port == memtile SOUTH_SLAVE_START (verify against archspec)
}

#[test]
fn inter_tile_array_edge_has_no_dest() {
    // A north master on the top compute row points off-array -> None.
    assert!(inter_tile_dest(TileKind::Compute, 1, 5, /*compute NORTH_MASTER_START*/ ?, 5, 6).is_none());
}
```

- [ ] **Step 2: Run, verify fails.**
- [ ] **Step 3: Implement `inter_tile_dest`** by porting the (tile_kind, direction) match arms from `propagate_inter_tile()`. For each direction range, map `master_port` in `[DIR_MASTER_START, DIR_MASTER_END]` to the neighbor's corresponding `DIR'_SLAVE_START + (master_port - DIR_MASTER_START)`, with the neighbor coordinates from the direction (North → row+1, South → row-1, East → col+1, West → col-1). Return `None` for out-of-range ports or off-array neighbors.
- [ ] **Step 4: Run, verify passes.**
- [ ] **Step 5: Commit** — `feat(#140): static inter-tile adjacency mirroring propagate_inter_tile`.

### Task A3: Intra-tile crossbar edges (circuit + configured packet)

**Files:**
- Modify: `src/device/stream_switch/route_graph.rs`, and add a read-only helper to `src/device/stream_switch/mod.rs` if `slave_slots`/`master_packet_config` need a public accessor.

**Interfaces:**
- Consumes: a `&Tile` (its `stream_switch.local_routes`, `slave_slots`, `master_packet_config`, `masters`, `slaves`).
- Produces: `pub fn intra_tile_edges(tile: &Tile) -> Vec<RouteEdge>` — every enabled slave→master crossbar connection on the tile, `EdgeKind::Circuit` for `local_routes`, `EdgeKind::Packet` for configured packet slots. Each edge's `src` is the *slave* port (`PortDir::Slave`), `dst` is the *master* port (`PortDir::Master`) — data flows slave(in)→master(out) through the crossbar.

For packet edges, reuse the resolution in `resolve_packet_route()` (`src/device/stream_switch/mod.rs`): for each enabled `PacketSlot` on a slave, find the masters whose `MasterPacketConfig` accepts `(slot.arbiter, slot.msel)` and emit an edge per (slave, master). Enumerate **configured slots only** (not all 32 packet IDs) — the static graph is a superset over what is configured. If `resolve_packet_route` is private, factor a pure `fn packet_targets(slave_slots: &[PacketSlot;4], masters: &[MasterPacketConfig]) -> Vec<u8>`-style helper out of it so both the runtime and the resolver call one implementation.

- [ ] **Step 1: Write failing tests** with a synthetic tile.

```rust
#[test]
fn intra_tile_circuit_edge_from_local_route() {
    let mut tile = Tile::new_compute(1, 2);      // use the real constructor
    tile.stream_switch.local_routes.push(LocalRoute { slave_idx: 3, master_idx: 7, enabled: true, latency: 3 });
    let edges = intra_tile_edges(&tile);
    let e = edges.iter().find(|e| e.kind == EdgeKind::Circuit).expect("circuit edge");
    assert_eq!(e.src.dir, PortDir::Slave);
    assert_eq!(e.dst.dir, PortDir::Master);
    assert_eq!((e.src.port, e.dst.port), (3, 7));
    assert_eq!((e.src.col, e.src.row), (1, 2));
}

#[test]
fn intra_tile_disabled_route_is_skipped() {
    let mut tile = Tile::new_compute(1, 2);
    tile.stream_switch.local_routes.push(LocalRoute { slave_idx: 3, master_idx: 7, enabled: false, latency: 3 });
    assert!(intra_tile_edges(&tile).iter().all(|e| e.kind != EdgeKind::Circuit));
}
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement `intra_tile_edges`** — iterate `local_routes` (enabled only) → circuit edges; iterate `slave_slots`/`master_packet_config` via the shared packet helper → packet edges. Resolve each `port` index's `kind` string from the corresponding `StreamPort.port_type`.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `feat(#140): intra-tile crossbar edges (circuit + configured packet)`.

### Task A4: `resolve_route_graph` + integration on add_one

**Files:**
- Modify: `src/device/state/mod.rs` (or `DeviceState`'s home) — `impl DeviceState { pub fn resolve_route_graph(&self) -> StreamRouteGraph }`.
- Test: a `#[test]` that loads `add_one_using_dma`'s xclbin (skip if absent).

**Interfaces:**
- Consumes: A2 (`inter_tile_dest`), A3 (`intra_tile_edges`), `self.array` iteration.
- Produces: `StreamRouteGraph` = (every tile's `intra_tile_edges`) ∪ (for every enabled master port on every tile, the `inter_tile_dest` edge if `Some`). Each inter-tile edge `kind = EdgeKind::InterTile`.

- [ ] **Step 1: Confirm circuit vs packet for add_one (discovery step).** Build and run the existing inspector to see the data-path routing mode:
  `cargo run --example dump_ss_slots -- /home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin > /tmp/add_one_ss.txt` then `Read /tmp/add_one_ss.txt`. Record in the task report whether the shim↔memtile↔compute data path is circuit (`local_routes`) or packet (slots). This sets which edge kind the A4 integration test asserts on.
- [ ] **Step 2: Write the failing integration test** — assert the resolved graph contains the connectivity that orients add_one's known event pairs. Assert at the **physical-reachability** level, not exact indices, so it is robust:

```rust
#[test]
fn resolve_route_graph_orients_add_one_dataflow() {
    let path = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin";
    let Ok(state) = load_state_from_xclbin(path) else { eprintln!("fixture absent; skipping"); return; };
    let g = state.resolve_route_graph();
    assert!(!g.edges.is_empty(), "graph must have edges");
    // The shim DMA source port must reach the shim DMA sink port through the fabric
    // (MM2S data -> compute -> S2MM). Resolve the DMA ports by type, then BFS.
    let src = dma_port_ref(&state, 1, 0, /*channel*/ 0, /*source side*/);   // the port with outgoing fabric edges
    let sink = dma_port_ref(&state, 1, 0, /*channel*/ 0, /*sink side*/);
    assert!(reachable(&g, &src, &sink), "shim DMA source must reach shim DMA sink");
    // The memtile lead stream port must reach its co-firing downstream ports.
    // (exact ports come from event_port_selection; this asserts non-empty memtile fan-out)
    assert!(g.edges.iter().any(|e| e.src.row == 1 && e.dst.row == 1 && e.kind != EdgeKind::InterTile),
            "memtile intra-tile fan-out present");
}
```

Provide `reachable(&StreamRouteGraph, &PortRef, &PortRef) -> bool` as a small BFS test helper (or a `pub fn` on the graph — the Python side reimplements reachability anyway, so a Rust BFS here is just for the test). `dma_port_ref` / `load_state_from_xclbin` use existing parser/device APIs (`Xclbin::from_file`, `find_cdo_offset`/`Cdo::parse`, `DeviceState::new_npu1`, `apply_cdo`).
- [ ] **Step 3: Implement `resolve_route_graph`** composing A2+A3 over `self.array.iter()`.
- [ ] **Step 4: Run** — `cargo test -p xdna-emu resolve_route_graph_orients_add_one_dataflow`. If the fixture xclbin is present it must PASS; if absent it prints "skipping" and returns.
- [ ] **Step 5: Commit** — `feat(#140): DeviceState::resolve_route_graph + add_one dataflow test`.

### Task A5: Validate static graph ⊇ dynamic enactment

**Files:**
- Modify: `src/device/array/routing.rs` (gated enacted-hop recorder), `src/device/stream_switch/route_graph.rs` (the validation test).

**Interfaces:**
- Produces: an opt-in recorder that, when enabled, appends `(src: PortRef, dst: PortRef)` for each inter-tile word **insertion** (in `propagate_inter_tile`, where both src master port and dst slave port are known — *not* at delivery, where `InFlightWord` no longer carries the source). Gate it behind a cheap runtime flag (an `Option<Vec<(PortRef, PortRef)>>` field on `TileArray` set by a `pub fn enable_hop_recording(&mut self)`, or an env check consistent with the existing `XDNA_EMU_WATCH` style). Default: disabled, zero overhead.

This is the soundness gate Maya called out: the static reconstruction must *be* the emulator's routing, not a parallel guess.

- [ ] **Step 1: Write the failing test** — run add_one to completion with hop recording on, assert every enacted inter-tile hop is an `InterTile` edge in the static graph:

```rust
#[test]
fn static_graph_is_superset_of_enacted_inter_tile_hops() {
    let path = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin";
    let Ok(mut state) = load_state_from_xclbin(path) else { eprintln!("fixture absent; skipping"); return; };
    let graph = state.resolve_route_graph();
    state.array.enable_hop_recording();
    run_to_completion(&mut state);                 // existing interpreter entry
    let enacted = state.array.take_recorded_hops();
    assert!(!enacted.is_empty(), "add_one must take inter-tile hops");
    let edgeset: std::collections::HashSet<_> =
        graph.edges.iter().filter(|e| e.kind == EdgeKind::InterTile)
            .map(|e| (e.src.clone(), e.dst.clone())).collect();
    for (s, d) in &enacted {
        assert!(edgeset.contains(&(s.clone(), d.clone())),
                "enacted hop {s:?}->{d:?} missing from static graph");
    }
}
```

(`PortRef` needs `Hash, Eq` — add the derives in A1 if not already there. Reuse the interpreter's existing run entry for `run_to_completion`.)
- [ ] **Step 2: Run, verify fails** (recorder not yet present).
- [ ] **Step 3: Implement the recorder** at the `propagate_inter_tile` insertion site; implement `enable_hop_recording` / `take_recorded_hops`.
- [ ] **Step 4: Run** — PASS with fixture present (circuit routing is deterministic, so the enacted set is a clean subset). Also run `cargo test --lib` to confirm no regression from the recorder field.
- [ ] **Step 5: Commit** — `test(#140): validate static route graph superset of enacted hops`.

---

## Tier B — Rust config dump

### Task B1: `dump_config_json.rs` — route graph + tile/port/event bindings

**Files:**
- Create: `examples/dump_config_json.rs`
- Create (fixture, committed): `tools/config_extract/fixtures/add_one_using_dma.config.json` (generated by running the example; committed so Python tests run offline).

**Interfaces:**
- Produces the JSON dump (schema below). Consumed by Python `dump_model.py` (C1).

**JSON schema (schema-first — define before serializing):**

```json
{
  "device": "npu1",
  "route_graph": { "edges": [ { "src": {"col":1,"row":0,"port":12,"dir":"master","kind":"north"},
                                "dst": {"col":1,"row":1,"port":7,"dir":"slave","kind":"south"},
                                "kind": "inter_tile" } ] },
  "tiles": [
    { "col":1, "row":0, "kind":"shim",
      "ports": [ {"index":12,"dir":"master","kind":"north","packet":false},
                 {"index":3,"dir":"slave","kind":"dma","dma_channel":0,"packet":false} ],
      "event_port_selection": [ {"slot":0,"port":7,"is_master":false}, null, null, null, null, null, null, null ]
    }
  ]
}
```

- `route_graph` is `state.resolve_route_graph()` serialized verbatim (A1 types).
- `tiles[].ports` lists every `StreamPort` with `index`, `dir`, `kind` (port_type string), `packet` (`packet_enable`), and `dma_channel` when `port_type` is `Dma(ch)`. This is how Python finds the physical port for `DMA_MM2S_0` / `DMA_S2MM_0`.
- `tiles[].event_port_selection` is the parsed `tile.event_port_selection: [Option<(u8,bool)>;8]` — `null` for unconfigured slots, else `{slot, port, is_master}`. This is how Python maps `PORT_RUNNING_N` → physical port.

- [ ] **Step 1: Write a failing smoke test** (as `#[test]` in the example, fixture-guarded):

```rust
#[test]
fn dump_produces_route_graph_and_event_bindings_for_add_one() {
    let path = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin";
    let Ok(state) = load_state_from_xclbin(path) else { return; };
    let dump = build_dump(&state);                 // the fn the binary's main() calls
    let json = serde_json::to_value(&dump).unwrap();
    assert!(json["route_graph"]["edges"].as_array().unwrap().len() > 0);
    // memtile (row 1) must have at least one configured event_port_selection slot
    let memtile = json["tiles"].as_array().unwrap().iter()
        .find(|t| t["row"] == 1 && t["kind"] == "memtile").unwrap();
    assert!(memtile["event_port_selection"].as_array().unwrap().iter().any(|s| !s.is_null()));
    // shim (row 0) must expose a dma port with a dma_channel
    let shim = json["tiles"].as_array().unwrap().iter().find(|t| t["row"] == 0).unwrap();
    assert!(shim["ports"].as_array().unwrap().iter().any(|p| p.get("dma_channel").is_some()));
}
```

- [ ] **Step 2: Run, verify fails.**
- [ ] **Step 3: Implement** the `#[derive(Serialize)]` dump structs (`ConfigDump`, `TileDump`, `PortDump`, `EventPortSel`), `build_dump(&DeviceState) -> ConfigDump`, and `main()` (`args`: xclbin path → stdout JSON, pretty-printed). Reuse the load path from A4.
- [ ] **Step 4: Run** the test (PASS with fixture). Then generate the committed fixture:
  `cargo run --example dump_config_json -- /home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin > tools/config_extract/fixtures/add_one_using_dma.config.json`
- [ ] **Step 5: Commit** — `feat(#140): dump_config_json example (route graph + event bindings) + add_one fixture`.

### Task B2: Dump BD chains + DMA channels + lock pairings

**Files:**
- Modify: `examples/dump_config_json.rs`
- Modify: `tools/config_extract/fixtures/add_one_using_dma.config.json` (regenerate)

**Interfaces:**
- Extends the schema with the remaining two structural sources (full dump now; *translation* deferred to a later plan):

```json
"tiles": [ { "...": "...",
  "dma_channels": [ {"index":0,"dir":"mm2s","start_bd":0}, {"index":0,"dir":"s2mm","start_bd":4} ],
  "bds": [ {"id":0,"valid":true,"use_next_bd":true,"next_bd":1,
            "lock_acq_id":0,"lock_acq_value":-1,"lock_rel_id":1,"lock_rel_value":1} ],
  "locks": [ {"id":0,"value":0}, {"id":1,"value":0} ] } ]
```

- `bds` from `tile.dma_bds` (`BufferDescriptor`): `id`, `valid`, `use_next_bd`, `next_bd`, and the lock fields `lock_acq_id/lock_acq_value/lock_rel_id/lock_rel_value` (the lock-pairing evidence — dumped now, translated later).
- `dma_channels` from `tile.dma_channels`: `index`, `dir` (`"mm2s"`/`"s2mm"` per the channel's role), `start_bd` (the initial BD; read from `start_queue`/`current_bd` per the channel struct).
- `locks` from `tile.locks`: `id`, `value`.

- [ ] **Step 1: Write the failing test** — extend B1's smoke:

```rust
#[test]
fn dump_includes_bd_chains_and_locks_for_add_one() {
    let path = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin";
    let Ok(state) = load_state_from_xclbin(path) else { return; };
    let json = serde_json::to_value(&build_dump(&state)).unwrap();
    let any_bd = json["tiles"].as_array().unwrap().iter()
        .flat_map(|t| t["bds"].as_array().cloned().unwrap_or_default())
        .any(|bd| bd["valid"] == true);
    assert!(any_bd, "add_one must configure at least one valid BD");
    let any_dma = json["tiles"].as_array().unwrap().iter()
        .flat_map(|t| t["dma_channels"].as_array().cloned().unwrap_or_default())
        .any(|c| c["dir"] == "mm2s" || c["dir"] == "s2mm");
    assert!(any_dma, "add_one must configure DMA channels");
}
```

- [ ] **Step 2: Run, verify fails.**
- [ ] **Step 3: Implement** the extra serialize structs and extend `build_dump`. Only serialize valid BDs (or include `valid` and let Python filter — prefer including all configured BD slots with `valid` so the dump is a faithful quote).
- [ ] **Step 4: Run** (PASS), regenerate the committed fixture (same command as B1 Step 4).
- [ ] **Step 5: Commit** — `feat(#140): dump BD chains, DMA channels, lock pairings (full physical quote)`.

---

## Tier C — Python config_path generator

### Task C1: Dump loader + reachability

**Files:**
- Create: `tools/config_extract/__init__.py` (empty), `tools/config_extract/dump_model.py`, `tools/config_extract/reachability.py`
- Test: `tools/test_config_extract_reachability.py`

**Interfaces:**
- `dump_model.load_dump(path) -> ConfigDump` with frozen dataclasses `ConfigDump(device, route_graph, tiles)`, `RouteGraph(edges)`, `RouteEdge(src, dst, kind)`, `PortRef(col, row, port, dir, kind)`, `TileDump(col, row, kind, ports, event_port_selection, dma_channels, bds, locks)` etc. mirroring the B1/B2 schema. `PortRef` is frozen + hashable.
- `reachability.Reachability(edges)` with `reachable(src: PortRef, dst: PortRef) -> bool` (BFS/transitive closure over directed edges) and `reaches_any(src) -> set[PortRef]`.

- [ ] **Step 1: Write failing tests:**

```python
from config_extract.dump_model import PortRef
from config_extract.reachability import Reachability

def _p(col, row, port, d="master"):
    return PortRef(col=col, row=row, port=port, dir=d, kind="x")

def test_reachable_direct_edge():
    r = Reachability([(_p(1,0,12), _p(1,1,7))])
    assert r.reachable(_p(1,0,12), _p(1,1,7))

def test_reachable_transitive_two_hops():
    r = Reachability([(_p(1,0,12), _p(1,1,7,"slave")),
                      (_p(1,1,7,"slave"), _p(1,1,11))])
    assert r.reachable(_p(1,0,12), _p(1,1,11))

def test_not_reachable_wrong_direction():
    r = Reachability([(_p(1,0,12), _p(1,1,7))])
    assert not r.reachable(_p(1,1,7), _p(1,0,12))

def test_self_not_reachable_without_self_loop():
    r = Reachability([(_p(1,0,12), _p(1,1,7))])
    assert not r.reachable(_p(1,0,12), _p(1,0,12))
```

(`Reachability` accepts either `RouteEdge`s or `(src, dst)` tuples — provide a small normalizer so tests can pass tuples.)
- [ ] **Step 2: Run** — `cd tools && python -m pytest test_config_extract_reachability.py -v` → FAIL (modules absent).
- [ ] **Step 3: Implement** `dump_model` dataclasses + `load_dump` (json.load → dataclasses; tolerate `null` event-port slots), and `Reachability` (adjacency dict keyed by `PortRef`, BFS).
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `feat(#140): config dump loader + route-graph reachability`.

### Task C2: Event-key resolution

**Files:**
- Create: `tools/config_extract/event_map.py`
- Test: `tools/test_config_extract_event_map.py`

**Interfaces:**
- `event_map.resolve_event_port(tile: TileDump, event_name: str) -> PortRef | None` — maps a trace event name to the physical route-graph node it observes:
  - `DMA_MM2S_{ch}_*` / `DMA_S2MM_{ch}_*` → the tile's `ports` entry with `kind == "dma"` and `dma_channel == ch`, picking the `dir` that matches the DMA engine side (resolve by which port carries fabric edges, **not** by assuming MM2S=master; see below). Return its `PortRef`.
  - `PORT_RUNNING_{n}` → `tile.event_port_selection[n]` → `PortRef(col, row, port=sel.port, dir = "master" if sel.is_master else "slave", kind=<from ports>)`.
  - Other names (`LOCK_STALL`, `PERF_CNT_2`, `CONFLICT_*`, …) → `None` (not route-orientable in Plan 2).
- `event_map.event_key(col, row, pkt, name) -> str` — the `f"{col}|{row}|{pkt}|{name}"` helper (import or mirror `trace_join._key`).
- The DMA side selection: a DMA event is a **source** if its port has outgoing route edges, a **sink** if it has incoming edges. Resolve via the graph, not the name. `resolve_event_port` takes the `Reachability`/edge set (or an out-degree map) so it can pick the port consistent with the dataflow. This is the Global-Constraint "orientation is reachability, never naming" applied to port selection itself.

- [ ] **Step 1: Write failing tests** against the committed fixture dump (so the test is real, not synthetic):

```python
from config_extract.dump_model import load_dump
from config_extract.event_map import resolve_event_port, event_key
from pathlib import Path

FIX = Path(__file__).resolve().parent / "config_extract" / "fixtures" / "add_one_using_dma.config.json"

def _tile(dump, col, row):
    return next(t for t in dump.tiles if t.col == col and t.row == row)

def test_port_running_maps_to_event_port_selection():
    dump = load_dump(FIX)
    memtile = _tile(dump, 1, 1)
    pr = resolve_event_port(memtile, "PORT_RUNNING_0", dump)
    assert pr is not None and (pr.col, pr.row) == (1, 1)

def test_dma_event_maps_to_dma_port():
    dump = load_dump(FIX)
    shim = _tile(dump, 1, 0)
    pr = resolve_event_port(shim, "DMA_MM2S_0_START_TASK", dump)
    assert pr is not None and pr.kind == "dma"

def test_non_route_event_returns_none():
    dump = load_dump(FIX)
    core = _tile(dump, 1, 2)
    assert resolve_event_port(core, "LOCK_STALL", dump) is None

def test_event_key_format():
    assert event_key(1, 0, 2, "DMA_MM2S_0_START_TASK") == "1|0|2|DMA_MM2S_0_START_TASK"
```

- [ ] **Step 2: Run, verify fails.**
- [ ] **Step 3: Implement** `event_map`. Parse the channel index out of `DMA_(MM2S|S2MM)_(\d+)_` with a regex; parse the slot out of `PORT_RUNNING_(\d+)`. Use the dump's `ports`/`event_port_selection`. For the DMA source/sink side, compute per-port out-degree from the route graph and pick the port whose direction matches the event family's role (a `*_START_TASK`/`*_FINISHED_TASK` on an MM2S engine is a source; if both a master and a slave dma port exist for the channel, choose by graph degree).
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `feat(#140): trace-event -> route-graph node resolution`.

### Task C3: config_path generator

**Files:**
- Create: `tools/config_extract/generator.py`
- Test: `tools/test_config_extract_generator.py`

**Interfaces:**
- `generator.generate_ledger(dump: ConfigDump, fired_event_keys: list[str]) -> dict` — for every ordered pair of fired events `(parent, child)` whose resources both resolve to graph nodes and where `parent`'s node **reaches** `child`'s node, emit a ledger entry `{"cite": <path-derived cite>, "a": child, "b": parent, "kind": "route"}`. (Confirm `a`=child / `b`=parent against `inference/ledger.py` before finalizing arg order.) Returns `{"_comment": "...generated...", "entries": [...]}`.
- The candidate event set is the **fired** events — pass in the decoded event keys (the generator orients only events that actually fire, matching how the engine consumes candidate pairs). `fired_event_keys` come from a captured run's `trace.events.json` (a helper `fired_keys_from_run(run_dir)` may reuse `inference.loader`/`trace_join`).
- `cite` is path-derived and human-auditable, e.g. `route:1|0|2:DMA_MM2S_0_START_TASK->1|0|2:DMA_S2MM_0_START_TASK@<n>hops` — it names the edge path so the audit (C4) can resolve it back to the graph.
- `generator.main()` CLI: `python -m config_extract.generator <config.json> <run_dir> -o <ledger.json>`.

- [ ] **Step 1: Write the failing test** — the keystone: from the add_one fixture dump + the 11 fired keys, the generator reproduces exactly the 5 hand-authored route edges and nothing spurious:

```python
from config_extract.dump_model import load_dump
from config_extract.generator import generate_ledger
from pathlib import Path

FIX = Path(__file__).resolve().parent / "config_extract" / "fixtures" / "add_one_using_dma.config.json"
FIRED = [
    "1|0|2|DMA_MM2S_0_FINISHED_TASK", "1|0|2|DMA_MM2S_0_START_TASK",
    "1|0|2|DMA_S2MM_0_FINISHED_TASK", "1|0|2|DMA_S2MM_0_START_TASK",
    "1|0|2|DMA_S2MM_0_STREAM_STARVATION",
    "1|1|3|PORT_RUNNING_0", "1|1|3|PORT_RUNNING_1",
    "1|1|3|PORT_RUNNING_4", "1|1|3|PORT_RUNNING_5",
    "1|2|0|LOCK_STALL", "1|2|0|PERF_CNT_2",
]
EXPECTED_EDGES = {
    ("1|0|2|DMA_S2MM_0_START_TASK", "1|0|2|DMA_MM2S_0_START_TASK"),
    ("1|0|2|DMA_S2MM_0_STREAM_STARVATION", "1|0|2|DMA_MM2S_0_START_TASK"),
    ("1|1|3|PORT_RUNNING_1", "1|1|3|PORT_RUNNING_0"),
    ("1|1|3|PORT_RUNNING_4", "1|1|3|PORT_RUNNING_0"),
    ("1|1|3|PORT_RUNNING_5", "1|1|3|PORT_RUNNING_0"),
}

def test_generator_reproduces_hand_authored_route_edges():
    dump = load_dump(FIX)
    led = generate_ledger(dump, FIRED)
    got = {(e["a"], e["b"]) for e in led["entries"] if e["kind"] == "route"}
    assert got == EXPECTED_EDGES
    for e in led["entries"]:
        assert e["cite"].startswith("route:")
```

(If A4's discovery shows the memtile fan-out resolves more PORT_RUNNING pairs than the hand-authored 3 — e.g. PORT_RUNNING_0 also reaches a fired-but-unledgered port — that is a *real* structural edge the hand authoring omitted, not a bug. In that case update `EXPECTED_EDGES` to the graph-justified set and note the discrepancy in the task report for the final review to adjudicate; do **not** suppress a structurally-valid edge to match the hand authoring. This is a plan-vs-reality reconciliation point — surface it, do not silently override.)
- [ ] **Step 2: Run, verify fails.**
- [ ] **Step 3: Implement** `generate_ledger` over `Reachability` + `resolve_event_port`, plus `fired_keys_from_run` and `main()`.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `feat(#140): config_path generator (route reachability -> ledger)`.

### Task C4: Generator-audit / spot-check + ledger compatibility

**Files:**
- Modify: `tools/config_extract/generator.py` (add `audit_ledger`)
- Test: `tools/test_config_extract_generator.py` (extend)

**Interfaces:**
- `generator.audit_ledger(led: dict, dump: ConfigDump) -> list[str]` — returns a list of audit failures (empty = clean). For each entry: (1) `a` and `b` resolve to graph nodes via `event_map`; (2) the cited path actually exists (`parent` reaches `child`); (3) `kind == "route"`; (4) the `cite` is well-formed. This is the spec's generator-audit (the trust anchor now that hand-authoring is gone).
- The generated ledger must load through the **existing** `inference/ledger.py` and yield `provenance_ok` when the engine runs (the real proof of byte-compatibility).

- [ ] **Step 1: Write failing tests:**

```python
from config_extract.dump_model import load_dump
from config_extract.generator import generate_ledger, audit_ledger
import sys; sys.path.insert(0, ".")  # tools/ on path via conftest
from inference.ledger import load_ledger, ledger_facts
from inference.facts import KB, provenance_ok

def test_audit_clean_on_generated_ledger():
    dump = load_dump(FIX)
    led = generate_ledger(dump, FIRED)
    assert audit_ledger(led, dump) == []

def test_generated_ledger_loads_through_inference_ledger(tmp_path):
    import json
    dump = load_dump(FIX)
    led = generate_ledger(dump, FIRED)
    p = tmp_path / "gen.ledger.json"; p.write_text(json.dumps(led))
    parsed = load_ledger(str(p))          # must not raise
    facts = ledger_facts(parsed)
    assert all(f.predicate == "config_path" for f in facts)
    assert len(facts) == 5
```

- [ ] **Step 2: Run, verify fails.**
- [ ] **Step 3: Implement** `audit_ledger`. Read `inference/ledger.py` first to match its expected schema exactly (key names, arg order). Fix any schema mismatch in `generate_ledger` so the generated JSON is byte-compatible.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `feat(#140): generator-audit + inference-ledger compatibility`.

---

## Tier D — Full Axis-2 HW validation + close-out

### Task D1: Automated Axis-2 — generated ledger reproduces the 5 roots on silicon

**Files:**
- Modify: `tools/test_inference_hw_smoke.py`

**Interfaces:**
- The HW smoke now **generates** the ledger from a freshly-built config dump of `add_one_using_dma` instead of reading the hand-authored fixture, then runs the engine on the captured HW runs and asserts the same outcome as Plan 1 (5 stochastic roots, 5 derives, `provenance_ok`, replication clean, STREAM_STARVATION `derived`).

- [ ] **Step 1: Generate the config dump for the smoke kernel** (one-time, committed as the fixture used by D1):
  `env -u XDNA_EMU -u XDNA_EMU_RUNTIME cargo run --example dump_config_json -- /home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin > tools/config_extract/fixtures/add_one_using_dma.config.json` (already produced in B1/B2; confirm it is current).
- [ ] **Step 2: Add the failing HW-gated test** to `test_inference_hw_smoke.py`:

```python
def test_engine_with_generated_ledger_matches_hand_authored():
    """Plan 2: the GENERATED ledger reproduces the validation, closing the
    'ledger is hand-authored' caveat. Same 5 roots / 5 derives as Plan 1."""
    from config_extract.dump_model import load_dump
    from config_extract.generator import generate_ledger
    import json, tempfile
    from inference.engine import run_engine
    dump = load_dump(_CONFIG_FIX)        # the committed config dump
    fired = _fired_keys_across_runs(_run_dirs())     # union of fired keys
    led = generate_ledger(dump, fired)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(led, f); gen_path = f.name
    rep = run_engine(_run_dirs(), gen_path, _CANDIDATE_PAIRS)
    assert set(rep["stochastic_roots"]) == _EXPECTED_ROOTS
    assert rep["provenance_ok"] is True
    assert rep["replication_violations"] == []
    assert {d[0] for d in rep["derives"]} == {c for c, _ in _CANDIDATE_PAIRS}
    assert rep["classification"]["1|0|2|DMA_S2MM_0_STREAM_STARVATION"] == "derived"
```

(`_CONFIG_FIX` points at the committed dump; `_fired_keys_across_runs` reuses the C3 helper. If the generator yields more route edges than the 5 hand-authored candidate pairs — see C3's reconciliation note — extend `_CANDIDATE_PAIRS`/`_EXPECTED_ROOTS` to the graph-justified set and document it; the engine's conclusion, not the hand-authored count, is ground truth.)
- [ ] **Step 3: Run** the offline pieces first (`pytest test_config_extract_*.py`), then the HW-gated smoke against the existing capture:
  `cd tools && XDNA_HW_SMOKE=1 XDNA_SMOKE_RUNS=../build/experiments/infer-smoke python -m pytest test_inference_hw_smoke.py -v`
  (Reuses the Plan-1 capture under `build/experiments/infer-smoke`; recapture only if absent, via `env -u XDNA_EMU -u XDNA_EMU_RUNTIME python tools/capture_infer_smoke.py build/experiments/infer-smoke 6`.)
- [ ] **Step 4: Verify** all smoke tests pass (4 Plan-1 + the new generated-ledger test).
- [ ] **Step 5: Commit** — `test(#140): automated Axis-2 — generated ledger reproduces 5 roots on NPU1`.

### Task D2: Close-out — finding + parity note + memory

**Files:**
- Create: `docs/superpowers/findings/2026-06-21-config-path-extraction-axis2.md`
- Modify: the inference-engine memory file + `MEMORY.md` index.

- [ ] **Step 1: Write the finding** — document: the route-graph reconstruction now lives in the emulator (`resolve_route_graph`, validated superset-of-enacted), the full physical dump, the route-translation generator, and the **closed caveat**: the 5 roots are now reproduced from an auto-generated ledger, not hand authoring. State the two remaining honest gaps: (a) BD-chain and lock-pairing *translation* still deferred (dump present, no kernel exercises them); (b) packet-route translation synthetic-only until a packet-routed kernel enters the corpus (note whether add_one was circuit per A4 Step 1).
- [ ] **Step 2: Update memory** — `project_inference_engine_brainstorm_inflight.md` description + body to "Plan 2 of 3 COMPLETE (route-graph reconstruction in emulator + auto-generated ledger; hand-authored caveat CLOSED); next Plan 3 (groups + Z3)"; update the `MEMORY.md` index line.
- [ ] **Step 3: Commit** — `docs(#140): close-out config-path extraction (Plan 2) — caveat closed`.

---

## Self-Review

- **Spec coverage:** resolves spec open-questions on the config_path derivation rule (C1–C3 reachability) and the structural-ledger format + generator-audit (C4). Axis-2 (D1) matches spec Section 4's "independently re-derive the 5 roots." The `same_source`/identity full-gate (spec Axis-2 second clause) is **not** in scope — no kernel here aliases an event across trace units; that stays deferred (consistent with Plan 1's finding) and is noted, not silently dropped.
- **Placeholder scan:** Rust steps that port existing logic cite the exact source functions (`propagate_inter_tile`, `resolve_packet_route`) and give concrete test contracts; the `?` placeholders in A2's test are explicitly flagged "fill from the source" discovery, not hidden. Python steps carry complete code.
- **Type consistency:** `PortRef`/`RouteEdge`/`EdgeKind`/`PortDir` are defined in A1 and consumed unchanged by B1 (serialize) and C1 (deserialize, mirrored). The ledger schema `{cite,a,b,kind}` and `a`=child/`b`=parent are pinned to `inference/ledger.py` (C3/C4 read it to confirm). event_key order `col|row|pkt|name` is constant throughout.
- **Reconciliation risk flagged:** if the route graph justifies more PORT_RUNNING edges than the hand-authored 3, that is surfaced (C3 note, D1 note) for human adjudication, not auto-suppressed — the structural truth governs, per the Global Constraints.
