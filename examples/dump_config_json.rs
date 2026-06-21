//! Dump static configuration (route graph + per-tile port/event metadata) to JSON.
//!
//! Loads an xclbin, applies its CDO, then optionally applies the runtime
//! instruction stream (`insts.bin`) to capture per-run register configuration
//! (e.g. `Stream_Switch_Event_Port_Selection`).  Serialises the resulting
//! device state to stdout as pretty-printed JSON.  The output is consumed by
//! `dump_model.py` (Tier C) and committed as a fixture for offline Python tests.
//!
//! # Usage
//!
//! ```text
//! cargo run --example dump_config_json -- path/to/aie.xclbin path/to/insts.bin
//! ```
//!
//! The `insts.bin` argument is optional.  When present, the register-write
//! instructions (Write32, BlockWrite, MaskWrite) are applied to the device
//! state after the CDO; Sync (TCT), DdrPatch, and MaskPoll instructions are
//! skipped because they require host-memory context or live DMA state not
//! available during a static dump.  This populates fields such as
//! `event_port_selection` that are configured at runtime rather than in the CDO.
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

/// Apply the config-only register writes from a runtime instruction stream.
///
/// Parses `insts_bytes` using the existing `NpuInstructionStream` parser, then
/// iterates the instructions and applies only the register-write variants
/// (Write32, BlockWrite, MaskWrite) to `state`.
///
/// The following instruction types are **skipped** because they require
/// host-memory context or live DMA state unavailable during a static dump:
/// - `Sync` (TCT): would block waiting for a DMA channel to complete.
/// - `DdrPatch`: rewrites BD address fields with host-buffer addresses;
///   meaningful only during a live run.
/// - `MaskPoll`: polls a register until a condition is met; only relevant with
///   a running engine.
///
/// This is the "config-vs-execution" clean split: the register-write ops carry
/// the per-run static configuration (e.g. `Stream_Switch_Event_Port_Selection`
/// at 0xB0F00/0xB0F04 on the memtile); the skipped ops carry execution
/// triggers.  Since neither Sync nor DdrPatch are applied, the function never
/// blocks and requires no host memory.
///
/// Reuses `xdna_emu::npu::NpuInstructionStream` (the same parser the
/// `XclbinSuite` / `NpuExecutor` use during live runs) to guarantee parity
/// with the instruction format the emulator's runtime path understands.
pub fn apply_config_writes_from_insts(state: &mut DeviceState, insts_bytes: &[u8]) -> Result<usize, String> {
    use xdna_emu::npu::{NpuInstruction, NpuInstructionStream};
    use xdna_archspec::aie2::{TILE_COL_SHIFT, TILE_ROW_SHIFT, TILE_OFFSET_MASK};

    let stream = NpuInstructionStream::parse(insts_bytes).map_err(|e| format!("parse insts: {e}"))?;

    let start_col = state.start_col;
    let mut applied = 0usize;

    // Decode a 32-bit NPU register address into (physical_col, row, offset).
    // The address encodes a logical column; shift by start_col to get physical.
    let decode = |addr: u32| -> (u8, u8, u32) {
        let logical_col = ((addr >> TILE_COL_SHIFT) & 0x7F) as u8;
        let row = ((addr >> TILE_ROW_SHIFT) & 0x1F) as u8;
        let offset = addr & TILE_OFFSET_MASK;
        let physical_col = logical_col.saturating_add(start_col);
        (physical_col, row, offset)
    };

    for instr in stream.instructions() {
        match instr {
            NpuInstruction::Write32 { reg_off, value } => {
                let (col, row, offset) = decode(*reg_off);
                state.write_tile_register(col, row, offset, *value);
                applied += 1;
            }
            NpuInstruction::BlockWrite { reg_off, values } => {
                let (col, row, base_offset) = decode(*reg_off);
                for (i, &value) in values.iter().enumerate() {
                    let offset = base_offset + (i as u32) * 4;
                    state.write_tile_register(col, row, offset, value);
                }
                applied += values.len();
            }
            NpuInstruction::MaskWrite { reg_off, value, mask } => {
                let (col, row, offset) = decode(*reg_off);
                // Read-modify-write: preserve bits not covered by mask.
                // Read the current value WITHOUT side effects via registers_ref()
                // (matching the canonical RMW path in
                // DeviceState::mask_write_register).  Tile::read_register takes
                // &mut self and performs a real lock acquire/release on reads in
                // the LOCK_REQUEST range -- a static config dump must be
                // side-effect-free / idempotent, so a lock-touching MaskWrite in
                // insts.bin must not spuriously fire a lock op here.
                let current = state
                    .tile_mut(col as usize, row as usize)
                    .map(|t| *t.registers_ref().get(&offset).unwrap_or(&0))
                    .unwrap_or(0);
                let new_value = (current & !mask) | (value & mask);
                state.write_tile_register(col, row, offset, new_value);
                applied += 1;
            }
            // Sync/TCT: would block on DMA completion. Skip.
            NpuInstruction::Sync { .. } => {}
            // DdrPatch: patches BD address with host-buffer address. Skip.
            NpuInstruction::DdrPatch { .. } => {}
            // MaskPoll: polls until condition met. Skip.
            NpuInstruction::MaskPoll { .. } => {}
            // Unknown: opaque payload, cannot safely apply. Skip.
            NpuInstruction::Unknown { .. } => {}
        }
    }

    Ok(applied)
}

/// Load a fully-configured `DeviceState` from an xclbin file and optionally
/// an `insts.bin` runtime instruction stream.
///
/// Applies the CDO unconditionally.  If `insts_path` is `Some`, also applies
/// the config-register writes from the instruction stream (see
/// `apply_config_writes_from_insts`).
///
/// Returns `Err` if the file cannot be opened, parsed, or the CDO applied.
/// The caller (smoke test or `main`) decides how to handle the absence of the
/// fixture — the test guard uses `let Ok(state) = ... else { return; }`.
pub fn load_state_from_xclbin(xclbin_path: &str, insts_path: Option<&str>) -> Result<DeviceState, String> {
    use xdna_emu::parser::xclbin::SectionKind;
    use xdna_emu::parser::{AiePartition, Xclbin};
    use xdna_emu::parser::cdo::{find_cdo_offset, Cdo};

    let xclbin = Xclbin::from_file(xclbin_path).map_err(|e| format!("open xclbin: {e}"))?;
    let section = xclbin
        .find_section(SectionKind::AiePartition)
        .ok_or_else(|| "AiePartition section not found".to_owned())?;
    let partition = AiePartition::parse(section.data()).map_err(|e| format!("parse partition: {e}"))?;
    let pdi = partition.primary_pdi().ok_or_else(|| "primary PDI not found".to_owned())?;
    let cdo_offset = find_cdo_offset(pdi.pdi_image).ok_or_else(|| "CDO magic not found".to_owned())?;
    let cdo = Cdo::parse(&pdi.pdi_image[cdo_offset..]).map_err(|e| format!("parse CDO: {e}"))?;

    let mut state = DeviceState::new_npu1();
    state.apply_cdo(&cdo).map_err(|e| format!("apply CDO: {e}"))?;

    if let Some(path) = insts_path {
        let insts_bytes = std::fs::read(path).map_err(|e| format!("read insts.bin: {e}"))?;
        apply_config_writes_from_insts(&mut state, &insts_bytes)
            .map_err(|e| format!("apply insts config writes: {e}"))?;
    }

    Ok(state)
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let xclbin_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: dump_config_json <path/to/aie.xclbin> [path/to/insts.bin]");
        std::process::exit(1);
    });

    let insts_path = std::env::args().nth(2);

    let state = load_state_from_xclbin(&xclbin_path, insts_path.as_deref()).unwrap_or_else(|e| {
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

    /// Integration smoke test: CDO-only dump from add_one_using_dma xclbin.
    ///
    /// Checks:
    /// 1. route_graph has edges after CDO load.
    /// 2. Every tile has an event_port_selection array of exactly 8 elements
    ///    (CDO-only load: all null, since event port selection is written by insts.bin).
    /// 3. The memtile at row 1 has kind "memtile" and exposes DMA ports with
    ///    `dma_channel` (memtile DMA lives in the stream switch, unlike the shim
    ///    where DMA is behind a separate mux and has no SS ports).
    #[test]
    fn dump_produces_route_graph_and_event_bindings_for_add_one() {
        let xclbin_path =
            "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin";
        let Ok(state) = load_state_from_xclbin(xclbin_path, None) else {
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

    /// Step-3 decisive verification: applying `insts.bin` populates
    /// `event_port_selection` on the memtile (row 1).
    ///
    /// The `Stream_Switch_Event_Port_Selection` registers (0xB0F00/0xB0F04) on
    /// the memtile are written by the runtime instruction stream, NOT the xclbin
    /// CDO.  Without insts.bin the dump shows all-null for every tile.  After
    /// applying insts.bin the memtile (col 0, row 1) must have at least one
    /// non-null slot.
    ///
    /// Decoded from add_one_using_dma/chess/insts.bin:
    ///   word[66]=0x001B0F00 value=0x23222120  -> slots 0-3: ports 0x20,0x21,0x22,0x23
    ///   word[72]=0x001B0F04 value=0x03020100  -> slots 4-7: ports 0x00,0x01,0x02,0x03
    ///
    /// Each byte encodes: bits[4:0] = port_idx, bit[5] = is_master.
    /// Slot 0: byte 0x20 = port_idx=0, is_master=true  (0x20 & 0x1F = 0, 0x20 & 0x20 = set)
    /// Slot 4: byte 0x00 = port_idx=0, is_master=false (0x00 & 0x1F = 0, 0x00 & 0x20 = 0)
    ///
    /// If this test passes: the emulator models the 0xB0F00 control-packet write.
    /// If this test FAILS (all-null after insts): the emulator drops the write —
    /// a real fidelity gap that must be fixed before proceeding.
    #[test]
    fn applying_insts_populates_memtile_event_port_selection() {
        let xclbin_path =
            "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin";
        let insts_path =
            "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/insts.bin";

        // Skip if fixtures are absent (CI / pre-build environment).
        if !std::path::Path::new(xclbin_path).exists() || !std::path::Path::new(insts_path).exists() {
            eprintln!("SKIP applying_insts_populates_memtile_event_port_selection: fixtures not found");
            return;
        }

        let state =
            load_state_from_xclbin(xclbin_path, Some(insts_path)).expect("load xclbin + insts must succeed");

        // The memtile targeted by insts.bin is at (col 0, row 1) in logical
        // space (logical col 0 = physical col 0 with start_col=0).
        let memtile = state.array.get(0, 1).expect("memtile (0,1) must exist");
        let sel = &memtile.event_port_selection;

        // DECISIVE CHECK: at least one slot must be non-null after applying insts.bin.
        // If all slots are still None, the emulator does not model the 0xB0F00 write.
        assert!(
            sel.iter().any(|s| s.is_some()),
            "FIDELITY GAP: memtile (0,1) event_port_selection is all-null after applying insts.bin \
             -- the emulator does not model the 0xB0F00 Stream_Switch_Event_Port_Selection write"
        );

        // Verify the decoded values match the expected insts.bin encoding.
        // word[66]=0x001B0F00 value=0x23222120:
        //   byte0=0x20 -> port=0,  is_master=true   (slot 0)
        //   byte1=0x21 -> port=1,  is_master=true   (slot 1)
        //   byte2=0x22 -> port=2,  is_master=true   (slot 2)
        //   byte3=0x23 -> port=3,  is_master=true   (slot 3)
        // word[72]=0x001B0F04 value=0x03020100:
        //   byte0=0x00 -> port=0,  is_master=false  (slot 4)
        //   byte1=0x01 -> port=1,  is_master=false  (slot 5)
        //   byte2=0x02 -> port=2,  is_master=false  (slot 6)
        //   byte3=0x03 -> port=3,  is_master=false  (slot 7)
        let expected: [Option<(u8, bool)>; 8] = [
            Some((0, true)),
            Some((1, true)),
            Some((2, true)),
            Some((3, true)),
            Some((0, false)),
            Some((1, false)),
            Some((2, false)),
            Some((3, false)),
        ];
        assert_eq!(
            sel, &expected,
            "memtile (0,1) event_port_selection does not match expected insts.bin decoded values"
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
