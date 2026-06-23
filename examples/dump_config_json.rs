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
//! cargo run --example dump_config_json -- path/to/aie.xclbin [path/to/insts.bin]
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
//!       ],
//!       "bds": [
//!         {"id": 0, "valid": true, "use_next_bd": false, "next_bd": 0,
//!          "lock_acq_id": 1, "lock_acq_value": -1, "lock_rel_id": 1, "lock_rel_value": 1}
//!       ],
//!       "dma_channels": [
//!         {"index": 0, "dir": "s2mm", "start_bd": 4},
//!         {"index": 0, "dir": "mm2s", "start_bd": 0}
//!       ],
//!       "locks": [{"id": 0, "value": 0}],
//!       "shim_mux": {"mm2s_slaves": [2, null], "s2mm_masters": [2, null]}
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
//! - `tiles[].bds` is all BD slots; only valid BDs have meaningful data, but
//!   all slots are included so the index is the BD id.
//! - `tiles[].dma_channels` lists all DMA channels; `dir` is `"s2mm"` or
//!   `"mm2s"`, `index` is the per-direction channel index.
//! - `tiles[].locks` lists all lock slots with their current value.
//! - `tiles[].shim_mux` is only present on shim tiles; maps DMA channel
//!   indices to stream-switch port indices (null if unmapped).

use serde::Serialize;

use xdna_emu::device::stream_switch::StreamRouteGraph;
use xdna_emu::device::{DeviceState, PortType};
use xdna_emu::device::dma::ChannelType;
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
    /// Per-tile metadata: ports, event-port selection, BDs, channels, locks.
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
    /// All DMA buffer descriptor slots (index == BD id).
    pub bds: Vec<BdDump>,
    /// All DMA channels (s2mm channels first, then mm2s channels).
    pub dma_channels: Vec<DmaChannelDump>,
    /// All lock slots.
    pub locks: Vec<LockDump>,
    /// Shim DMA mux mapping (shim tiles only; absent on other tile types).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shim_mux: Option<ShimMuxDump>,
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

/// Parsed Buffer Descriptor (all fields from `BufferDescriptor`).
///
/// All BD slots are included so that the BD `id` == array index.
/// `valid == false` means the slot is unused; downstream consumers can filter.
#[derive(Debug, Serialize)]
pub struct BdDump {
    /// BD slot index.
    pub id: u8,
    /// BD is valid (enabled) per the Valid_BD field.
    pub valid: bool,
    /// Use-next-BD chaining flag.
    pub use_next_bd: bool,
    /// Next BD id to chain to (meaningful only when `use_next_bd` is true).
    pub next_bd: u8,
    /// Lock id to acquire before transfer starts.
    pub lock_acq_id: u8,
    /// Lock acquire value (signed semaphore threshold).
    pub lock_acq_value: i8,
    /// Lock id to release after transfer completes.
    pub lock_rel_id: u8,
    /// Lock release delta (signed).
    pub lock_rel_value: i8,
    /// BD base address in bytes (tile-local for compute/memtile, host/DDR for shim).
    #[serde(default)]
    pub base_addr: u64,
    /// Transfer length in bytes.
    #[serde(default)]
    pub length: u32,
}

/// A single DMA channel (quoting the runtime-loaded channel state).
#[derive(Debug, Serialize)]
pub struct DmaChannelDump {
    /// Per-direction channel index (0-based within the direction).
    pub index: u8,
    /// Direction: "s2mm" (stream-to-memory) or "mm2s" (memory-to-stream).
    pub dir: String,
    /// Start BD id: the BD this channel was programmed to start from.
    ///
    /// Read from `DmaChannel::current_bd`, which the register parser sets to
    /// `start_bd_id` from the START_QUEUE register.  Zero when the channel
    /// has not been programmed.
    pub start_bd: u8,
}

/// A single lock slot.
#[derive(Debug, Serialize)]
pub struct LockDump {
    /// Lock slot index.
    pub id: u8,
    /// Current semaphore value (range -64..=63 per aie-rt).
    pub value: i8,
}

/// Shim DMA mux mapping (shim tiles only).
///
/// The shim tile has no DMA stream-switch ports; instead a dedicated mux
/// connects DMA MM2S outputs to SS South slave ports and SS South master
/// ports to DMA S2MM inputs.  This mapping is parsed from the Mux_Config /
/// Demux_Config registers by the CDO loader and stored on the tile.
#[derive(Debug, Serialize)]
pub struct ShimMuxDump {
    /// DMA MM2S channel → SS South slave port index.  `null` = unmapped.
    /// (index = MM2S channel; value = the SS South slave port it feeds.)
    pub mm2s_slaves: Vec<Option<usize>>,
    /// SS South master port → DMA S2MM channel input.  `null` = unmapped.
    /// (index = S2MM channel; value = the SS South master port feeding it.)
    pub s2mm_masters: Vec<Option<usize>>,
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
// BD reconstruction helper
// ---------------------------------------------------------------------------

/// Build a `BdDump` for one BD slot, reading from the authoritative source for
/// each tile type.
///
/// - **Compute and MemTile**: BD registers live in the DMA *subsystem* address
///   space (compute base 0x1D000, memtile base 0xA0000), which lies **past the
///   end** of the tile's `data_memory()` slice (compute 0x10000, memtile
///   0x80000).  Reading via `BufferDescriptor::from_memory()` therefore reads
///   out of bounds and returns an all-zero ("invalid") BD.  The correctly
///   decoded BD lives in `DmaEngine.bd_configs[]`, populated by the CDO loader
///   (`dma_write_*_bd_data -> configure_bd`) and valid right after `apply_cdo()`.
///   We read it via `state.array.dma_engine(col,row).get_bd(bd_id)`.
///
/// - **Shim tiles**: have no data memory and no `bd_configs` entries with lock
///   fields populated by the CDO (their BD words are split between the legacy
///   `dma_bds[i]` struct (words 0-5) and `tile.registers` (words 6-7)).  We
///   reconstruct the word array then call `BufferDescriptor::from_registers`.
///   This path is unchanged.
///
/// Validity is keyed off the source's own `valid` flag (`BdConfig.valid` for
/// compute/memtile, the decoded `BufferDescriptor.valid` for shim), so a real
/// lock-0 acquire is not confused with "no lock".  `acquire_lock`/`release_lock`
/// are `Option<u8>` in `BdConfig`; we surface the inner id (defaulting to 0 when
/// `None`, matching the shim path's zero-default for absent locks).
fn read_bd_for_tile(state: &DeviceState, tile: &xdna_emu::device::tile::Tile, bd_id: u8) -> BdDump {
    use xdna_emu::device::dma::{BufferDescriptor, bd_base_address, bd_register_count, BD_SPACING};
    use xdna_emu::device::tile::TileKind;

    match tile.tile_kind {
        // Compute and MemTile: read the decoded BdConfig from the DMA engine.
        TileKind::Compute | TileKind::Mem => {
            let engine = state.array.dma_engine(tile.col, tile.row);
            let bd = engine.and_then(|d| d.get_bd(bd_id));

            // MemTile BD lock IDs are stored RAW (8-bit cross-tile address space:
            // West 0-63, Own 64-127, East 128-191) -- never masked at parse time
            // (see state/memtile.rs).  Resolve to the local lock index using the
            // engine's authoritative `resolve_lock_id` (derived from
            // mlir-aie getLockLocalBaseIndex / aie-rt), so the dump reports the
            // tile-local id (e.g. raw 64 -> local 0).  Compute passes through
            // unchanged.  West/East neighbour locks also resolve to that
            // neighbour's local index; we surface that local id (the dump has a
            // single u8 field, and add_one's BDs all use Own locks).
            let resolve = |raw: u8| -> u8 {
                use xdna_emu::device::dma::LockTarget;
                match engine.and_then(|d| d.resolve_lock_id(raw)) {
                    Some(LockTarget::Own(id) | LockTarget::West(id) | LockTarget::East(id)) => id,
                    None => raw,
                }
            };

            match bd {
                Some(cfg) => BdDump {
                    id: bd_id,
                    valid: cfg.valid,
                    use_next_bd: cfg.next_bd.is_some(),
                    next_bd: cfg.next_bd.unwrap_or(0),
                    lock_acq_id: cfg.acquire_lock.map(resolve).unwrap_or(0),
                    lock_acq_value: cfg.acquire_value,
                    lock_rel_id: cfg.release_lock.map(resolve).unwrap_or(0),
                    lock_rel_value: cfg.release_value,
                    base_addr: cfg.base_addr,
                    length: cfg.length,
                },
                None => BdDump {
                    id: bd_id,
                    valid: false,
                    use_next_bd: false,
                    next_bd: 0,
                    lock_acq_id: 0,
                    lock_acq_value: 0,
                    lock_rel_id: 0,
                    lock_rel_value: 0,
                    base_addr: 0,
                    length: 0,
                },
            }
        }
        // Shim: no data memory; reconstruct from legacy struct + register store.
        TileKind::ShimNoc | TileKind::ShimPl => {
            let reg_count = bd_register_count(tile.tile_kind);
            let bd_base = bd_base_address(tile.tile_kind) as u32;
            let bd_stride = BD_SPACING as u32;

            let bd = if (bd_id as usize) >= tile.dma_bds.len() {
                BufferDescriptor::default()
            } else {
                let legacy = &tile.dma_bds[bd_id as usize];
                let mut words = vec![0u32; reg_count];

                // Words 0-5 come from the legacy struct fields.
                if reg_count > 0 {
                    words[0] = legacy.addr_low;
                }
                if reg_count > 1 {
                    words[1] = legacy.addr_high;
                }
                if reg_count > 2 {
                    words[2] = legacy.length;
                }
                if reg_count > 3 {
                    words[3] = legacy.control;
                }
                if reg_count > 4 {
                    words[4] = legacy.d0;
                }
                if reg_count > 5 {
                    words[5] = legacy.d1;
                }

                // Words 6-7 (shim BDs have 8 words) are in the register HashMap.
                for w in 6..reg_count {
                    let offset = bd_base + (bd_id as u32) * bd_stride + (w as u32) * 4;
                    words[w] = tile.registers().get(&offset).copied().unwrap_or(0);
                }

                BufferDescriptor::from_registers(&words, tile.tile_kind)
            };

            BdDump {
                id: bd_id,
                valid: bd.valid,
                use_next_bd: bd.use_next_bd,
                next_bd: bd.next_bd,
                lock_acq_id: bd.lock_acq_id,
                lock_acq_value: bd.lock_acq_value,
                lock_rel_id: bd.lock_rel_id,
                lock_rel_value: bd.lock_rel_value,
                base_addr: bd.base_addr_bytes(),
                length: bd.length_bytes() as u32,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// build_dump
// ---------------------------------------------------------------------------

/// Build a `ConfigDump` from a fully-configured `DeviceState`.
///
/// Walks `state.array` to collect per-tile port, event-port, BD, DMA channel,
/// lock, and shim-mux data, then appends the route graph resolved by
/// `state.resolve_route_graph()`.
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

            // --- bds: parse all BD slots ---
            let bds: Vec<BdDump> = (0..tile.dma_bds.len())
                .map(|bd_id| read_bd_for_tile(state, tile, bd_id as u8))
                .collect();

            // --- dma_channels: s2mm channels first, then mm2s ---
            //
            // `DmaChannel` carries no explicit direction field -- it stores raw
            // register state (control, start_queue, current_bd) and the
            // s2mm/mm2s distinction is purely positional.  This is the
            // emulator's canonical convention, shared by the DMA engine
            // (`ChannelType::from_channel_index`), the lock arbiter, and the
            // register-write dispatch: the flat `dma_channels` array holds the
            // S2MM channels first (indices 0..s2mm_count) then the MM2S
            // channels.  We derive `dir` from that SAME authoritative helper
            // rather than re-implementing the split, so this dump cannot drift
            // from the engine's interpretation.
            //
            // `s2mm_count` comes from `shim_mux_s2mm_masters.len()`, which is
            // sized to `params.dma_s2mm_channels` for EVERY tile type (see
            // `Tile::new`), not just shim tiles.
            let s2mm_count = tile.shim_mux_s2mm_masters.len();

            // Pin the layout invariant we depend on: the flat array length must
            // equal s2mm_count + mm2s_count, with s2mm channels occupying the
            // first `s2mm_count` slots.  If a future refactor reorders the
            // channel array (e.g. interleaves directions), this fires loudly
            // instead of silently mislabeling every channel's `dir`.
            debug_assert_eq!(
                tile.dma_channels.len(),
                s2mm_count + tile.shim_mux_mm2s_slaves.len(),
                "DMA channel layout invariant violated for tile ({},{}): expected \
                 {} s2mm + {} mm2s = {} flat channels, found {} -- the position-based \
                 dir split is no longer valid",
                tile.col,
                tile.row,
                s2mm_count,
                tile.shim_mux_mm2s_slaves.len(),
                s2mm_count + tile.shim_mux_mm2s_slaves.len(),
                tile.dma_channels.len(),
            );

            let dma_channels: Vec<DmaChannelDump> = tile
                .dma_channels
                .iter()
                .enumerate()
                .map(|(ch_idx, ch)| {
                    // Use the engine's authoritative direction helper.
                    let dir = match ChannelType::from_channel_index(ch_idx, s2mm_count) {
                        ChannelType::S2MM => "s2mm".to_owned(),
                        ChannelType::MM2S => "mm2s".to_owned(),
                    };
                    // Per-direction index: 0-based within the channel's direction.
                    let index = if ch_idx < s2mm_count {
                        ch_idx as u8
                    } else {
                        (ch_idx - s2mm_count) as u8
                    };
                    DmaChannelDump { index, dir, start_bd: ch.current_bd }
                })
                .collect();

            // --- locks ---
            let locks: Vec<LockDump> = tile
                .locks
                .iter()
                .enumerate()
                .map(|(lock_id, lock)| LockDump { id: lock_id as u8, value: lock.value })
                .collect();

            // --- shim_mux (shim tiles only) ---
            let shim_mux = if tile.tile_kind.is_shim() {
                Some(ShimMuxDump {
                    mm2s_slaves: tile.shim_mux_mm2s_slaves.clone(),
                    s2mm_masters: tile.shim_mux_s2mm_masters.clone(),
                })
            } else {
                None
            };

            TileDump {
                col: tile.col,
                row: tile.row,
                kind: tile_kind_str(tile.tile_kind).to_owned(),
                ports,
                event_port_selection,
                bds,
                dma_channels,
                locks,
                shim_mux,
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
/// After CDO + insts are applied, loads compute-core ELFs from the `.prj`
/// directory adjacent to the xclbin (if present).  This populates compute-tile
/// program memory so that `resolve_route_graph` (called by `build_dump`) can
/// emit `CoreLockRelay` edges for the config dump.
///
/// ELF load strategy: scan the xclbin's parent directory for any `*.prj`
/// subdirectory and load every `core_*.elf` found there.  The `.prj` naming
/// varies by MLIR source filename (e.g. `aie_arch.mlir.prj`), so we probe
/// by readdir rather than hard-coding the stem.  If no `.prj` directory is
/// found the state is returned as-is (CDO config only; no `CoreLockRelay`
/// edges will be emitted by `resolve_route_graph`).
///
/// Source choice rationale: the xclbin parser does not expose per-core ELF
/// bytes (the `AiePartition` / PDI layer only provides the raw CDO image).
/// The `.prj` directory is the canonical Chess-compiler artifact location
/// and the same source used by the P0.5 `load_state_with_core_elfs` test
/// helper in `route_graph.rs`.
///
/// Returns `Err` if the file cannot be opened, parsed, or the CDO applied.
/// The caller (smoke test or `main`) decides how to handle the absence of the
/// fixture — the test guard uses `let Ok(state) = ... else { return; }`.
pub fn load_state_from_xclbin(xclbin_path: &str, insts_path: Option<&str>) -> Result<DeviceState, String> {
    use xdna_emu::parser::xclbin::SectionKind;
    use xdna_emu::parser::{AieElf, AiePartition, Xclbin};
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

    // --- Load compute-core ELFs from the .prj directory (if present) ---
    //
    // The xclbin parser does not expose per-core ELF bytes; they live in the
    // Chess `.prj` directory adjacent to the xclbin.  Probe for a `*.prj`
    // subdirectory, then load every `core_*.elf` into the matching compute tile.
    // This populates program memory so resolve_route_graph emits CoreLockRelay
    // edges in the config dump.  If no .prj dir is present (e.g. Peano builds
    // without an adjacent .prj dir, or CDO-only callers), this block is a no-op.
    if let Some(xclbin_parent) = std::path::Path::new(xclbin_path).parent() {
        // Probe for a *.prj subdirectory (name varies by MLIR source filename).
        let prj_dir = std::fs::read_dir(xclbin_parent)
            .ok()
            .and_then(|rd| {
                rd.flatten().find(|e| {
                    let p = e.path();
                    p.is_dir()
                        && p.file_name()
                            .and_then(|n| n.to_str())
                            .map(|n| n.ends_with(".prj"))
                            .unwrap_or(false)
                })
            })
            .map(|e| e.path());

        if let Some(prj_dir) = prj_dir {
            // Collect all (path, col, row) tuples first so we can read each
            // ELF file's bytes into a Vec before parsing — AieElf borrows the
            // data slice so the Vec must outlive the parse+load_into call.
            let core_elfs: Vec<(std::path::PathBuf, u8, u8)> = std::fs::read_dir(&prj_dir)
                .map_err(|e| format!("read .prj dir {:?}: {e}", prj_dir))?
                .flatten()
                .filter_map(|entry| {
                    let path = entry.path();
                    let name = path.file_name()?.to_string_lossy().to_string();
                    if !name.ends_with(".elf") || !name.contains("core_") {
                        return None;
                    }
                    // Parse "core_COL_ROW.elf" — find "core_" then split the
                    // remainder on '_' to get col and row.
                    let core_idx = name.find("core_")?;
                    let after_core = &name[core_idx + 5..];
                    let mut parts = after_core.splitn(3, '_');
                    let col: u8 = parts.next()?.parse().ok()?;
                    let row_str = parts.next()?;
                    let row: u8 = row_str.trim_end_matches(".elf").parse().ok()?;
                    Some((path, col, row))
                })
                .collect();

            for (path, col, row) in core_elfs {
                let data = std::fs::read(&path).map_err(|e| format!("read core ELF {:?}: {e}", path))?;
                let elf = AieElf::parse(&data).map_err(|e| format!("parse core ELF {:?}: {e}", path))?;
                elf.load_into(state.array.tile_mut(col, row));
                eprintln!("dump_config_json: loaded core ELF ({col},{row}) from {:?}", path);
            }
        }
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

    /// B2 smoke test: BD chains, DMA channels, locks, and shim mux.
    ///
    /// Asserts the new structural sources are populated for add_one_using_dma
    /// after applying CDO + insts.bin (full physical quote of the binary).
    #[test]
    fn dump_includes_bd_chains_dma_channels_locks_shim_mux_for_add_one() {
        let xclbin_path =
            "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin";
        let insts_path =
            "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/insts.bin";

        // Skip if fixtures are absent.
        if !std::path::Path::new(xclbin_path).exists() || !std::path::Path::new(insts_path).exists() {
            eprintln!(
                "SKIP dump_includes_bd_chains_dma_channels_locks_shim_mux_for_add_one: fixtures not found"
            );
            return;
        }

        let state =
            load_state_from_xclbin(xclbin_path, Some(insts_path)).expect("load xclbin + insts must succeed");
        let json = serde_json::to_value(&build_dump(&state)).unwrap();

        // 1. At least one valid BD must exist across all tiles.
        let any_bd = json["tiles"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|t| t["bds"].as_array().cloned().unwrap_or_default())
            .any(|bd| bd["valid"] == true);
        assert!(any_bd, "add_one_using_dma must configure at least one valid BD");

        // 2. DMA channels with mm2s or s2mm direction must be present.
        let any_dma = json["tiles"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|t| t["dma_channels"].as_array().cloned().unwrap_or_default())
            .any(|c| c["dir"] == "mm2s" || c["dir"] == "s2mm");
        assert!(any_dma, "add_one_using_dma must configure DMA channels");

        // 3. The shim tile at (0,0) must have a shim_mux with mm2s_slaves[0]
        //    and s2mm_masters[0] non-null (the mux is configured for add_one).
        let shim_tile = json["tiles"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["col"] == 0 && t["row"] == 0 && t["kind"] == "shim")
            .expect("shim tile (0,0) must exist");
        let shim_mux = &shim_tile["shim_mux"];
        assert!(!shim_mux.is_null(), "shim tile (0,0) must have shim_mux");
        assert!(
            !shim_mux["mm2s_slaves"][0].is_null(),
            "shim_mux.mm2s_slaves[0] must be non-null (MM2S ch0 is mapped for add_one)"
        );
        assert!(
            !shim_mux["s2mm_masters"][0].is_null(),
            "shim_mux.s2mm_masters[0] must be non-null (S2MM ch0 is mapped for add_one)"
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

    /// Tier-E regression: memtile (0,1) BD lock fields must be valid in the dump.
    ///
    /// Memtile BD registers live at offset 0xA0000, but the memtile's
    /// `data_memory()` slice is only 0x80000 bytes -- the BD registers are in the
    /// DMA subsystem address space, not data memory.  So reading BDs via
    /// `BufferDescriptor::from_memory()` reads past the slice end and returns
    /// all-zero ("invalid") BDs.  The correctly-decoded BD lives in
    /// `DmaEngine.bd_configs[]`, populated by the CDO loader.  This test pins the
    /// fix: the dump must read memtile BDs from `bd_configs` (via `get_bd`).
    ///
    /// Ground truth (add_one_using_dma memtile (0,1), S2MM0 start_bd 0):
    ///   acquire lock 0, release lock 1, valid == true, base_addr != 0.
    #[test]
    fn memtile_bd_lock_fields_are_valid_in_dump() {
        let xclbin_path =
            "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin";
        let insts_path =
            "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/insts.bin";

        // Skip if fixtures are absent (CI / pre-build environment).
        if !std::path::Path::new(xclbin_path).exists() || !std::path::Path::new(insts_path).exists() {
            eprintln!("SKIP memtile_bd_lock_fields_are_valid_in_dump: fixtures not found");
            return;
        }

        let state =
            load_state_from_xclbin(xclbin_path, Some(insts_path)).expect("load xclbin + insts must succeed");
        let json = serde_json::to_value(&build_dump(&state)).unwrap();

        // Find the memtile at (col 0, row 1).
        let memtile = json["tiles"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["col"] == 0 && t["row"] == 1 && t["kind"] == "memtile")
            .expect("memtile (0,1) must exist in dump");

        // S2MM0 active channel starts at BD 0 (per the committed fixture's
        // dma_channels).  Its BD must decode with valid==true and the kernel's
        // lock ids: acquire lock 0, release lock 1.
        let bd0 = memtile["bds"]
            .as_array()
            .unwrap()
            .iter()
            .find(|b| b["id"] == 0)
            .expect("memtile must have BD id 0");

        assert_eq!(
            bd0["valid"], true,
            "memtile BD 0 must be valid (read from bd_configs, not out-of-bounds data_memory)"
        );
        assert_eq!(bd0["lock_acq_id"], 0, "memtile BD 0 (S2MM0) acquire lock id must be 0");
        assert_eq!(bd0["lock_rel_id"], 1, "memtile BD 0 (S2MM0) release lock id must be 1");
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
