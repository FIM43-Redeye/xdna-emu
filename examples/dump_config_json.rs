//! Dump static configuration (route graph + per-tile port/event metadata) to JSON.
//!
//! Loads an xclbin, applies its CDO, then serialises the static routing
//! configuration to stdout as pretty-printed JSON.  The output is consumed by
//! `dump_model.py` (Tier C) and committed as a fixture for offline Python tests.
//!
//! # Usage
//!
//! ```text
//! cargo run --example dump_config_json -- path/to/aie.xclbin
//! ```
//!
//! # JSON schema
//!
//! ```json
//! {
//!   "device": "npu1",
//!   "route_graph": { "edges": [ ... ] },
//!   "tiles": [
//!     { "col": 1, "row": 0, "kind": "shim",
//!       "ports": [ {"index": 12, "dir": "master", "kind": "north", "packet": false},
//!                  {"index": 3, "dir": "slave", "kind": "dma", "dma_channel": 0, "packet": false} ],
//!       "event_port_selection": [
//!         {"slot": 0, "port": 7, "is_master": false},
//!         null, null, null, null, null, null, null
//!       ]
//!     }
//!   ]
//! }
//! ```
//!
//! - `route_graph` is `DeviceState::resolve_route_graph()` serialised verbatim.
//! - `tiles[].ports` lists every master and slave `StreamPort` (all ports,
//!   not only enabled ones), with `dma_channel` included when the port type is
//!   `Dma(ch)`.
//! - `tiles[].event_port_selection` is the 8-slot `[Option<(u8, bool)>; 8]`
//!   from `Tile::event_port_selection`.  `null` for unconfigured slots.

use serde::Serialize;

use xdna_emu::device::stream_switch::StreamRouteGraph;
use xdna_emu::device::{DeviceState, PortType};
use xdna_emu::device::tile::TileKind;

// ---------------------------------------------------------------------------
// Dump structs (cross-language JSON contract with dump_model.py)
// ---------------------------------------------------------------------------

/// Top-level configuration dump.
#[derive(Debug, Serialize)]
pub struct ConfigDump {
    /// Device name (always "npu1" for the current emulator target).
    pub device: String,
    /// Full static stream-switch route graph.
    pub route_graph: StreamRouteGraph,
    /// Per-tile metadata: ports and event-port selection.
    pub tiles: Vec<TileDump>,
}

/// Metadata for a single tile.
#[derive(Debug, Serialize)]
pub struct TileDump {
    /// Column index.
    pub col: u8,
    /// Row index.
    pub row: u8,
    /// Tile kind: "shim", "memtile", or "compute".
    pub kind: String,
    /// All stream-switch ports on this tile (masters and slaves).
    pub ports: Vec<PortDump>,
    /// 8-slot event-port selection; `null` for unconfigured slots.
    pub event_port_selection: [Option<EventPortSel>; 8],
}

/// A single stream-switch port.
#[derive(Debug, Serialize)]
pub struct PortDump {
    /// Port index within the tile's master or slave list.
    pub index: u8,
    /// Direction: "master" or "slave".
    pub dir: String,
    /// Port type kind string (e.g. "north", "south", "dma", "core", "trace").
    pub kind: String,
    /// True if this port is in packet-switched mode.
    pub packet: bool,
    /// DMA channel index — present only when `kind == "dma"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dma_channel: Option<u8>,
}

/// One event-port selection slot (non-null entry in `event_port_selection`).
#[derive(Debug, Serialize)]
pub struct EventPortSel {
    /// Slot index (0-7).
    pub slot: u8,
    /// Physical stream-switch port index (master or slave list, per `is_master`).
    pub port: u8,
    /// True if the selected port is a master port; false for slave.
    pub is_master: bool,
}

// ---------------------------------------------------------------------------
// kind string for TileKind
// ---------------------------------------------------------------------------

/// Simplified tile-kind label used in the JSON dump.
///
/// The brief schema uses flat three-way strings: "shim" (both ShimNoc and
/// ShimPl), "memtile", "compute".  This collapses the hardware distinction
/// between ShimNoc and ShimPl for Python consumers, which only need to
/// distinguish the three routing domains.
fn tile_kind_str(kind: TileKind) -> &'static str {
    match kind {
        TileKind::ShimNoc | TileKind::ShimPl => "shim",
        TileKind::Mem => "memtile",
        TileKind::Compute => "compute",
    }
}

// ---------------------------------------------------------------------------
// build_dump
// ---------------------------------------------------------------------------

/// Build a `ConfigDump` from a fully-configured `DeviceState`.
///
/// Walks `state.array` to collect per-tile port and event-port data, then
/// appends the route graph resolved by `state.resolve_route_graph()`.
pub fn build_dump(state: &DeviceState) -> ConfigDump {
    let route_graph = state.resolve_route_graph();

    let tiles = state
        .array
        .iter()
        .map(|tile| {
            // --- ports: masters ---
            let masters: Vec<PortDump> = tile
                .stream_switch
                .masters
                .iter()
                .map(|p| {
                    let dma_channel = if let PortType::Dma(ch) = p.port_type {
                        Some(ch)
                    } else {
                        None
                    };
                    PortDump {
                        index: p.index,
                        dir: "master".to_owned(),
                        kind: p.port_type.as_kind_str().to_owned(),
                        packet: p.packet_enable,
                        dma_channel,
                    }
                })
                .collect();

            // --- ports: slaves ---
            let slaves: Vec<PortDump> = tile
                .stream_switch
                .slaves
                .iter()
                .map(|p| {
                    let dma_channel = if let PortType::Dma(ch) = p.port_type {
                        Some(ch)
                    } else {
                        None
                    };
                    PortDump {
                        index: p.index,
                        dir: "slave".to_owned(),
                        kind: p.port_type.as_kind_str().to_owned(),
                        packet: p.packet_enable,
                        dma_channel,
                    }
                })
                .collect();

            // Combine: masters first, then slaves.
            let mut ports = masters;
            ports.extend(slaves);

            // --- event_port_selection ---
            let event_port_selection: [Option<EventPortSel>; 8] = std::array::from_fn(|i| {
                tile.event_port_selection[i].map(|(port, is_master)| EventPortSel {
                    slot: i as u8,
                    port,
                    is_master,
                })
            });

            TileDump {
                col: tile.col,
                row: tile.row,
                kind: tile_kind_str(tile.tile_kind).to_owned(),
                ports,
                event_port_selection,
            }
        })
        .collect();

    ConfigDump { device: "npu1".to_owned(), route_graph, tiles }
}

// ---------------------------------------------------------------------------
// Loader (mirrors load_npu1_state in route_graph.rs test module)
// ---------------------------------------------------------------------------

/// Load a fully-configured `DeviceState` from an xclbin file.
///
/// Returns `Err` if the file cannot be opened, parsed, or the CDO applied.
/// The caller (smoke test or `main`) decides how to handle the absence of the
/// fixture — the test guard uses `let Ok(state) = ... else { return; }`.
pub fn load_state_from_xclbin(path: &str) -> Result<DeviceState, String> {
    use xdna_emu::parser::xclbin::SectionKind;
    use xdna_emu::parser::{AiePartition, Xclbin};
    use xdna_emu::parser::cdo::{find_cdo_offset, Cdo};

    let xclbin = Xclbin::from_file(path).map_err(|e| format!("open xclbin: {e}"))?;
    let section = xclbin
        .find_section(SectionKind::AiePartition)
        .ok_or_else(|| "AiePartition section not found".to_owned())?;
    let partition = AiePartition::parse(section.data()).map_err(|e| format!("parse partition: {e}"))?;
    let pdi = partition.primary_pdi().ok_or_else(|| "primary PDI not found".to_owned())?;
    let cdo_offset = find_cdo_offset(pdi.pdi_image).ok_or_else(|| "CDO magic not found".to_owned())?;
    let cdo = Cdo::parse(&pdi.pdi_image[cdo_offset..]).map_err(|e| format!("parse CDO: {e}"))?;

    let mut state = DeviceState::new_npu1();
    state.apply_cdo(&cdo).map_err(|e| format!("apply CDO: {e}"))?;
    Ok(state)
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: dump_config_json <path/to/aie.xclbin>");
        std::process::exit(1);
    });

    let state = load_state_from_xclbin(&path).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });

    let dump = build_dump(&state);

    let json = serde_json::to_string_pretty(&dump).unwrap_or_else(|e| {
        eprintln!("serialisation error: {e}");
        std::process::exit(1);
    });

    println!("{json}");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Integration smoke test: dump from add_one_using_dma xclbin.
    ///
    /// Checks:
    /// 1. route_graph has edges after CDO load.
    /// 2. Every tile has an event_port_selection array of exactly 8 elements.
    ///    (all null for this non-trace fixture — schema correctness regardless)
    /// 3. The memtile at row 1 has kind "memtile" and exposes DMA ports with
    ///    `dma_channel` (memtile DMA lives in the stream switch, unlike the shim
    ///    where DMA is behind a separate mux and has no SS ports).
    ///
    /// Note: add_one_using_dma does not configure event port selection (no tracing).
    /// The structural assertion on event_port_selection (8 nulls) is correct schema
    /// verification even when the fixture is all-unconfigured.  The synthetic unit
    /// test `event_port_selection_serializes_correctly` below verifies the non-null
    /// path using a directly-constructed DeviceState.
    #[test]
    fn dump_produces_route_graph_and_event_bindings_for_add_one() {
        let path = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin";
        let Ok(state) = load_state_from_xclbin(path) else {
            return;
        };
        let dump = build_dump(&state);
        let json = serde_json::to_value(&dump).unwrap();

        // route_graph must be non-empty after CDO load
        assert!(
            json["route_graph"]["edges"].as_array().unwrap().len() > 0,
            "route_graph must have edges after CDO load"
        );

        // Every tile must have an event_port_selection array of exactly 8 elements
        for tile in json["tiles"].as_array().unwrap() {
            let sel = tile["event_port_selection"].as_array().unwrap();
            assert_eq!(
                sel.len(),
                8,
                "event_port_selection must be 8 elements for tile ({},{})",
                tile["col"],
                tile["row"]
            );
        }

        // memtile (row 1) must be present with kind "memtile" and expose DMA ports.
        // The memtile's DMA channels appear directly in the stream switch port list
        // (unlike the shim, where DMA bypasses the SS via a mux).
        let memtile = json["tiles"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["row"] == 1 && t["kind"] == "memtile")
            .expect("memtile (row 1) must exist in dump with kind='memtile'");
        assert_eq!(
            memtile["event_port_selection"].as_array().unwrap().len(),
            8,
            "memtile event_port_selection must have 8 slots"
        );
        assert!(
            memtile["ports"]
                .as_array()
                .unwrap()
                .iter()
                .any(|p| p.get("dma_channel").is_some()),
            "memtile must expose at least one dma port with dma_channel"
        );
    }

    /// Unit test: event_port_selection serializes correctly for non-null slots.
    ///
    /// add_one_using_dma doesn't configure event port selection (no tracing),
    /// so we verify the non-null path by directly mutating a DeviceState tile.
    /// This ensures `build_dump` round-trips configured slots correctly.
    #[test]
    fn event_port_selection_serializes_correctly() {
        let mut state = DeviceState::new_npu1();

        // Directly configure event port selection on the first memtile (col 0, row 1).
        // Slot 0: physical port 7, slave (is_master=false)
        // Slot 3: physical port 2, master (is_master=true)
        // Slots 1,2,4-7: None
        let tile = state.array.get_mut(0, 1).expect("memtile at (0,1) must exist");
        tile.event_port_selection[0] = Some((7, false));
        tile.event_port_selection[3] = Some((2, true));

        let dump = build_dump(&state);
        let json = serde_json::to_value(&dump).unwrap();

        let memtile = json["tiles"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["col"] == 0 && t["row"] == 1)
            .expect("tile (0,1) must be in dump");

        let sel = memtile["event_port_selection"].as_array().unwrap();
        assert_eq!(sel.len(), 8);

        // Slot 0: non-null, port=7, is_master=false
        assert!(!sel[0].is_null(), "slot 0 must be non-null");
        assert_eq!(sel[0]["slot"], 0);
        assert_eq!(sel[0]["port"], 7);
        assert_eq!(sel[0]["is_master"], false);

        // Slot 3: non-null, port=2, is_master=true
        assert!(!sel[3].is_null(), "slot 3 must be non-null");
        assert_eq!(sel[3]["slot"], 3);
        assert_eq!(sel[3]["port"], 2);
        assert_eq!(sel[3]["is_master"], true);

        // Slots 1, 2, 4-7: null
        for &i in &[1usize, 2, 4, 5, 6, 7] {
            assert!(sel[i].is_null(), "slot {i} must be null");
        }
    }
}
