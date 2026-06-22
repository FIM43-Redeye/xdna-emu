//! Static stream-switch route-graph types.
//!
//! Serializable graph representation of the stream-switch routing configuration
//! extracted from a loaded binary. Used to generate a structural `config_path`
//! ledger (Plan 2, #140) and serialized to JSON for the Python dump side.
//!
//! # Cross-language contract
//!
//! The serde field names and variant strings are a stable cross-language
//! contract consumed by `dump_model.py` and later Python tooling:
//! - `PortDir`: serializes as lowercase `"master"` / `"slave"`.
//! - `EdgeKind`: serializes as snake_case `"inter_tile"` / `"circuit"` /
//!   `"packet"` / `"dma_buffer_relay"`.
//! - `PortRef.kind`: a free-form string derived from `PortType::as_kind_str()`,
//!   e.g. `"north"`, `"south"`, `"dma"`, `"core"`, `"trace"`.

use serde::{Deserialize, Serialize};
use xdna_archspec::types::TileKind;
use xdna_archspec::aie2::stream_switch::{compute, mem_tile, shim};

/// Direction of a stream port from the perspective of the local stream switch.
///
/// Hardware convention: a Master port sends data (the switch drives it);
/// a Slave port receives data (an upstream source drives it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortDir {
    /// Master port — the switch sends data out on this port.
    Master,
    /// Slave port — the switch receives data in on this port.
    Slave,
}

/// Classification of a route edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    /// Edge crosses a tile boundary (north/south/east/west wire).
    InterTile,
    /// Circuit-switched route within a single tile's stream switch.
    Circuit,
    /// Packet-switched route within a single tile's stream switch.
    Packet,
    /// Intra-tile DMA buffer relay: an S2MM channel writes a memory buffer that
    /// an MM2S channel later reads (overlapping byte range). This is a memory
    /// hop through the DMA engine, NOT a stream-switch crossbar route, so the
    /// switch graph alone cannot model it. The edge is oriented
    /// `src = S2MM master DMA port (writer) -> dst = MM2S slave DMA port (reader)`,
    /// matching the routing convention in `src/device/array/routing.rs`
    /// (master DMA port = S2MM, slave DMA port = MM2S). The reverse direction
    /// is back-pressure, not dataflow, and is never emitted.
    DmaBufferRelay,
}

/// A reference to a specific port on a specific tile in the array.
///
/// `kind` is the human-readable port-type string derived from `PortType::as_kind_str()`
/// (e.g. `"north"`, `"south"`, `"dma"`, `"core"`, `"trace"`, `"tile_ctrl"`,
/// `"cascade"`, `"fifo"`, `"east"`, `"west"`). Kept as `String` rather than
/// an enum so the graph remains self-describing in JSON without an extra layer
/// of indirection, and so it naturally absorbs future port types.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PortRef {
    /// Column index of the tile.
    pub col: u8,
    /// Row index of the tile.
    pub row: u8,
    /// Port index within the tile (master or slave list, depending on `dir`).
    pub port: u8,
    /// Port direction.
    pub dir: PortDir,
    /// Port type name (derived from `PortType`; see `PortType::as_kind_str()`).
    pub kind: String,
}

/// A directed data-flow edge between two ports in the stream-switch graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteEdge {
    /// Source port (data flows from here).
    pub src: PortRef,
    /// Destination port (data flows to here).
    pub dst: PortRef,
    /// Edge classification.
    pub kind: EdgeKind,
}

/// The complete static stream-switch route graph for an array configuration.
///
/// Constructed by walking the CDO-configured stream-switch state after binary
/// load. Serializes to JSON for the Python `dump_model.py` side and for
/// regression comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamRouteGraph {
    /// All directed edges in the graph.
    pub edges: Vec<RouteEdge>,
}

impl StreamRouteGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self { edges: Vec::new() }
    }

    /// Add an edge to the graph.
    pub fn add_edge(&mut self, edge: RouteEdge) {
        self.edges.push(edge);
    }
}

impl Default for StreamRouteGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Static distillation of `propagate_inter_tile()` for route-graph construction.
///
/// Given a *master* port on a source tile, returns the neighbor tile's *slave*
/// `PortRef` it physically wires to, or `None` if:
/// - the master port index falls outside every directional range for that tile kind
///   (i.e., it is a DMA, Core, TileCtrl, or other non-directional port), or
/// - the implied neighbor is off the edge of the array (e.g., north from the top row),
///   in which case `kind_at` returns `None`, or
/// - the `(src_kind, dest_kind)` pair has no wiring arm in `propagate_inter_tile()`
///   (e.g., a compute tile directly above a shim, which never happens on real NPU
///   layouts but is rejected here exactly as the runtime would `continue` past it).
///
/// The mapping mirrors `propagate_inter_tile()` in `src/device/array/routing.rs`
/// exactly: same `xdna_archspec::aie2::stream_switch::{compute, mem_tile, shim}`
/// constants, same 1:1 index arithmetic (`slave = SLAVE_START + (master - MASTER_START)`),
/// same neighbor selection (North → row+1, South → row-1, East → col+1, West → col-1),
/// and crucially the **same `(src_kind, dest_kind)` tuple matching** — the
/// destination slave constant depends on BOTH tile kinds. The runtime keys each
/// directional block on `match (tile_kind, above_type/below_type/...)`; this
/// function does the same, looking up the neighbor's kind via `kind_at`.
///
/// `kind_at(col, row)` returns the tile kind at the given coordinate, or `None`
/// if the coordinate is off-array. It is the static analogue of the runtime's
/// `self.tiles[neighbor_idx].tile_kind`. (`cols`/`rows` are intentionally NOT
/// parameters: bound-checking is delegated to `kind_at` returning `None`, which
/// avoids hardcoding any row/column layout here.)
///
/// # Why dest_kind matters (the Compute-south fork)
///
/// A compute tile's South master can route to two different slave constants
/// depending on what is below it:
/// - below a MemTile → `mem_tile::NORTH_SLAVE_START` (compute row 2 → memtile row 1)
/// - below another Compute → `compute::NORTH_SLAVE_START` (compute rows 3+ → compute below)
///
/// These are distinct ports (13 vs 15). Collapsing them silently misroutes every
/// Compute→Compute south edge. The `(src_kind, dest_kind)` match prevents that.
///
/// # Port kind strings
///
/// The returned `PortRef.kind` uses `PortType::as_kind_str()` conventions:
/// - North-facing slave → `"south"` (the slave receives from the south, i.e., the port
///   faces south on the *destination* tile)
/// - South-facing slave → `"north"`
/// - East-facing slave → `"west"`
/// - West-facing slave → `"east"`
///
/// These match the slave-side port type at the destination, which is the mirror
/// direction of the master-side port type at the source.
pub fn inter_tile_dest(
    src_kind: TileKind,
    src_col: u8,
    src_row: u8,
    master_port: u8,
    kind_at: impl Fn(u8, u8) -> Option<TileKind>,
) -> Option<PortRef> {
    // Each arm: check whether master_port is in [DIR_MASTER_START, DIR_MASTER_END],
    // then compute the offset and return the matching slave on the neighbor tile.
    // Order mirrors propagate_inter_tile: North, South, East, West. The slave
    // constant is selected by matching (src_kind, dest_kind) exactly as the
    // runtime matches (tile_kind, neighbor_type).

    // --- North masters: data flows to tile above (row + 1) ---
    let north_row = src_row.checked_add(1)?; // None on u8 overflow (unreachable in practice)
    if let Some(dest_kind) = kind_at(src_col, north_row) {
        // Runtime arms: (Shim,Mem), (Mem,Compute), (Compute,Compute).
        let mapping = match (src_kind, dest_kind) {
            (TileKind::ShimNoc | TileKind::ShimPl, TileKind::Mem) => {
                Some((shim::NORTH_MASTER_START, shim::NORTH_MASTER_END, mem_tile::SOUTH_SLAVE_START))
            }
            (TileKind::Mem, TileKind::Compute) => {
                Some((mem_tile::NORTH_MASTER_START, mem_tile::NORTH_MASTER_END, compute::SOUTH_SLAVE_START))
            }
            (TileKind::Compute, TileKind::Compute) => {
                Some((compute::NORTH_MASTER_START, compute::NORTH_MASTER_END, compute::SOUTH_SLAVE_START))
            }
            _ => None,
        };
        if let Some((ms, me, ss)) = mapping {
            if let Some(p) = range_to_slave(master_port, ms, me, ss, src_col, north_row, "south") {
                return Some(p);
            }
        }
    }

    // --- South masters: data flows to tile below (row - 1) ---
    if let Some(south_row) = src_row.checked_sub(1) {
        if let Some(dest_kind) = kind_at(src_col, south_row) {
            // Runtime arms: (Mem,Shim), (Compute,Mem), (Compute,Compute).
            // Compute-south forks on dest_kind -- this is the bug-fix arm.
            let mapping = match (src_kind, dest_kind) {
                (TileKind::Mem, TileKind::ShimNoc | TileKind::ShimPl) => {
                    Some((mem_tile::SOUTH_MASTER_START, mem_tile::SOUTH_MASTER_END, shim::NORTH_SLAVE_START))
                }
                (TileKind::Compute, TileKind::Mem) => Some((
                    compute::SOUTH_MASTER_START,
                    compute::SOUTH_MASTER_END,
                    mem_tile::NORTH_SLAVE_START,
                )),
                (TileKind::Compute, TileKind::Compute) => {
                    Some((compute::SOUTH_MASTER_START, compute::SOUTH_MASTER_END, compute::NORTH_SLAVE_START))
                }
                _ => None,
            };
            if let Some((ms, me, ss)) = mapping {
                if let Some(p) = range_to_slave(master_port, ms, me, ss, src_col, south_row, "north") {
                    return Some(p);
                }
            }
        }
    }

    // --- East masters: data flows to tile at (col + 1) ---
    let east_col = src_col.checked_add(1)?;
    if let Some(dest_kind) = kind_at(east_col, src_row) {
        // Runtime arms: (Compute,Compute), (Shim,Shim). Same-type adjacency only.
        let mapping = match (src_kind, dest_kind) {
            (TileKind::Compute, TileKind::Compute) => {
                Some((compute::EAST_MASTER_START, compute::EAST_MASTER_END, compute::WEST_SLAVE_START))
            }
            (TileKind::ShimNoc | TileKind::ShimPl, TileKind::ShimNoc | TileKind::ShimPl) => {
                Some((shim::EAST_MASTER_START, shim::EAST_MASTER_END, shim::WEST_SLAVE_START))
            }
            _ => None,
        };
        if let Some((ms, me, ss)) = mapping {
            if let Some(p) = range_to_slave(master_port, ms, me, ss, east_col, src_row, "west") {
                return Some(p);
            }
        }
    }

    // --- West masters: data flows to tile at (col - 1) ---
    if let Some(west_col) = src_col.checked_sub(1) {
        if let Some(dest_kind) = kind_at(west_col, src_row) {
            // Runtime arms: (Compute,Compute), (Shim,Shim).
            let mapping = match (src_kind, dest_kind) {
                (TileKind::Compute, TileKind::Compute) => {
                    Some((compute::WEST_MASTER_START, compute::WEST_MASTER_END, compute::EAST_SLAVE_START))
                }
                (TileKind::ShimNoc | TileKind::ShimPl, TileKind::ShimNoc | TileKind::ShimPl) => {
                    Some((shim::WEST_MASTER_START, shim::WEST_MASTER_END, shim::EAST_SLAVE_START))
                }
                _ => None,
            };
            if let Some((ms, me, ss)) = mapping {
                if let Some(p) = range_to_slave(master_port, ms, me, ss, west_col, src_row, "east") {
                    return Some(p);
                }
            }
        }
    }

    // Port is not in any directional master range for this (src_kind, dest_kind).
    None
}

/// Map a master port in `[master_start, master_end]` to the destination slave
/// port using 1:1 index arithmetic: `slave = slave_start + (master_port - master_start)`.
/// Returns `None` if `master_port` is outside the range.
#[inline]
fn range_to_slave(
    master_port: u8,
    master_start: u8,
    master_end: u8,
    slave_start: u8,
    dst_col: u8,
    dst_row: u8,
    kind: &'static str,
) -> Option<PortRef> {
    if master_port < master_start || master_port > master_end {
        return None;
    }
    let offset = master_port - master_start;
    Some(PortRef {
        col: dst_col,
        row: dst_row,
        port: slave_start + offset,
        dir: PortDir::Slave,
        kind: kind.to_owned(),
    })
}

/// Emit every enabled slave→master crossbar connection for a single tile.
///
/// - `EdgeKind::Circuit` for each enabled `LocalRoute` (slave_idx → master_idx).
/// - `EdgeKind::Packet`  for each configured `PacketSlot` on a slave that
///   resolves to at least one master via the shared `packet_targets` helper.
///
/// This is the static route graph: it enumerates all connections that *could*
/// carry data given the current CDO configuration, regardless of whether data
/// is flowing right now. Packet edges are a superset over what's actually active
/// (any configured slot is included; arbiter locks are a runtime concept).
///
/// Port `kind` strings are derived from `PortType::as_kind_str()` on the
/// corresponding `StreamPort`.
pub fn intra_tile_edges(tile: &crate::device::tile::Tile) -> Vec<RouteEdge> {
    let ss = &tile.stream_switch;
    let col = tile.col;
    let row = tile.row;
    let mut edges: Vec<RouteEdge> = Vec::new();

    // --- Circuit edges ---
    for route in &ss.local_routes {
        if !route.enabled {
            continue;
        }
        let si = route.slave_idx as usize;
        let mi = route.master_idx as usize;
        let slave_kind = ss
            .slaves
            .get(si)
            .map(|p| p.port_type.as_kind_str())
            .unwrap_or("unknown")
            .to_owned();
        let master_kind = ss
            .masters
            .get(mi)
            .map(|p| p.port_type.as_kind_str())
            .unwrap_or("unknown")
            .to_owned();
        edges.push(RouteEdge {
            src: PortRef { col, row, port: route.slave_idx, dir: PortDir::Slave, kind: slave_kind },
            dst: PortRef { col, row, port: route.master_idx, dir: PortDir::Master, kind: master_kind },
            kind: EdgeKind::Circuit,
        });
    }

    // --- Packet edges ---
    // For each slave, scan its configured slots. For each enabled slot, call
    // the shared `packet_targets` helper (same logic as `resolve_packet_route`
    // in the runtime, factored out so both share one implementation).
    let slave_slots = ss.slave_slots();
    let master_cfgs = ss.master_packet_configs();
    for (slave_idx, slots) in slave_slots.iter().enumerate() {
        let slave_kind = ss
            .slaves
            .get(slave_idx)
            .map(|p| p.port_type.as_kind_str())
            .unwrap_or("unknown")
            .to_owned();
        for slot in slots.iter() {
            if !slot.enable {
                continue;
            }
            let targets = crate::device::stream_switch::packet_targets(master_cfgs, slot.arbiter, slot.msel);
            for master_idx in targets {
                let master_kind = ss
                    .masters
                    .get(master_idx as usize)
                    .map(|p| p.port_type.as_kind_str())
                    .unwrap_or("unknown")
                    .to_owned();
                edges.push(RouteEdge {
                    src: PortRef {
                        col,
                        row,
                        port: slave_idx as u8,
                        dir: PortDir::Slave,
                        kind: slave_kind.clone(),
                    },
                    dst: PortRef { col, row, port: master_idx, dir: PortDir::Master, kind: master_kind },
                    kind: EdgeKind::Packet,
                });
            }
        }
    }

    edges
}

/// Emit intra-tile DMA buffer-relay edges for a single tile.
///
/// A DMA buffer relay is a memory hop: an **S2MM** channel writes a memory
/// buffer that an **MM2S** channel later reads. When their BD byte ranges
/// overlap, data produced by the S2MM channel reaches the consumer through the
/// MM2S channel — but via local memory, not the stream-switch crossbar. The
/// static switch graph cannot see this hop, so we surface it explicitly here.
///
/// # Direction convention (authoritative per `src/device/array/routing.rs`)
///
/// - **StreamSwitch master port → DMA S2MM**: a DMA *master* port is an S2MM
///   channel (the buffer **writer** — the switch drives data into the DMA, the
///   DMA writes it to memory). Looked up via `tile.stream_switch.dma_master(ch)`.
/// - **DMA MM2S → StreamSwitch slave port**: a DMA *slave* port is an MM2S
///   channel (the buffer **reader** — the DMA reads memory and drives it out as
///   a slave source). Looked up via `tile.stream_switch.dma_slave(ch)`.
///
/// The relay edge is therefore oriented `src = S2MM master DMA port (writer)`
/// → `dst = MM2S slave DMA port (reader)`. The reverse (MM2S → S2MM) is
/// back-pressure, not dataflow, and is never emitted.
///
/// # Channel / BD derivation (derive, never hardcode)
///
/// - The S2MM / MM2S split for a flat channel index is decided by the canonical
///   `ChannelType::from_channel_index(idx, s2mm_count)` helper, with
///   `s2mm_count` taken from the DMA engine itself (`DmaEngine::s2mm_count`).
/// - Each channel's **configured start BD** is decoded from the channel's
///   `start_queue` register using the AM025-derived `start_bd_id` bit field
///   (memtile vs compute/shim layout selected by tile kind), NOT a hardcoded
///   mask. `start_queue` is the configured source of truth; this is a static,
///   pre-run method (`resolve_route_graph` runs on an un-stepped state), so the
///   configured start is the right anchor (the runtime `current_bd` pointer has
///   not advanced yet).
/// - From the start BD we follow the `next_bd` chain to gather every BD the
///   channel touches, guarding against cycles (add_one's chains are 2-BD
///   ping-pong loops, BD0↔BD1).
///
/// # Overlap
///
/// Buffer ranges are half-open `[base_addr, base_addr + length)`; two channels
/// relay iff any of their BD ranges intersect. One edge is emitted per
/// (S2MM master port, MM2S slave port) pair (deduplicated).
pub fn dma_buffer_relay_edges(
    tile: &crate::device::tile::Tile,
    dma: &crate::device::dma::DmaEngine,
    s2mm_count: usize,
) -> Vec<RouteEdge> {
    use crate::device::dma::ChannelType;

    let col = tile.col;
    let row = tile.row;

    // The configured-start-BD bit field: same AM025-derived layout the CDO
    // loader uses, picked by tile kind (memtile = 6-bit, compute/shim = 4-bit).
    let layout = crate::device::regdb::device_reg_layout();
    let start_bd_field = if tile.tile_kind.is_mem() {
        &layout.memtile_channel.start_bd_id
    } else {
        &layout.memory_channel.start_bd_id
    };

    // Gather the half-open byte ranges a channel's BD chain covers, starting at
    // its configured start BD and following `next_bd` (cycle-guarded).
    //
    // A channel only relays if it was actually *configured* (a Start_Queue write
    // occurred -- on a static post-load state that leaves a task queued / the
    // channel active). Un-configured channels leave `start_queue == 0`, which
    // would otherwise decode to BD 0 and fabricate spurious overlaps against
    // whatever genuinely uses BD 0. `channel_has_pending_work` is the canonical
    // engine query for "this channel has work" and is the right gate on the
    // un-stepped state `resolve_route_graph` sees.
    let channel_ranges = |flat_ch: usize| -> Vec<(u64, u64)> {
        if !dma.channel_has_pending_work(flat_ch as u8) {
            return Vec::new();
        }
        let Some(channel) = tile.dma_channels.get(flat_ch) else {
            return Vec::new();
        };
        let start_bd = start_bd_field.extract(channel.start_queue) as u8;

        let mut ranges: Vec<(u64, u64)> = Vec::new();
        let mut seen: std::collections::HashSet<u8> = std::collections::HashSet::new();
        let mut cur = Some(start_bd);
        while let Some(bd_id) = cur {
            if !seen.insert(bd_id) {
                break; // cycle guard (ping-pong chains revisit the start)
            }
            let Some(bd) = dma.get_bd(bd_id) else { break };
            if !bd.valid {
                break;
            }
            if bd.length > 0 {
                ranges.push((bd.base_addr, bd.base_addr + bd.length as u64));
            }
            cur = bd.next_bd;
        }
        ranges
    };

    // Half-open interval intersection.
    let overlaps = |a: &[(u64, u64)], b: &[(u64, u64)]| -> bool {
        a.iter().any(|&(a0, a1)| b.iter().any(|&(b0, b1)| a0 < b1 && b0 < a1))
    };

    let num_channels = dma.num_channels();
    let mut edges: Vec<RouteEdge> = Vec::new();
    let mut seen_pairs: std::collections::HashSet<(u8, u8)> = std::collections::HashSet::new();

    for s2mm_flat in 0..num_channels {
        if ChannelType::from_channel_index(s2mm_flat, s2mm_count) != ChannelType::S2MM {
            continue;
        }
        let s2mm_ranges = channel_ranges(s2mm_flat);
        if s2mm_ranges.is_empty() {
            continue;
        }
        // Per-direction channel index for the stream-switch port lookup.
        let s2mm_ch = s2mm_flat as u8;
        let Some(src_port) = tile.stream_switch.dma_master(s2mm_ch) else {
            continue;
        };

        for mm2s_flat in 0..num_channels {
            if ChannelType::from_channel_index(mm2s_flat, s2mm_count) != ChannelType::MM2S {
                continue;
            }
            let mm2s_ranges = channel_ranges(mm2s_flat);
            if mm2s_ranges.is_empty() {
                continue;
            }
            if !overlaps(&s2mm_ranges, &mm2s_ranges) {
                continue;
            }
            let mm2s_ch = (mm2s_flat - s2mm_count) as u8;
            let Some(dst_port) = tile.stream_switch.dma_slave(mm2s_ch) else {
                continue;
            };

            if !seen_pairs.insert((src_port.index, dst_port.index)) {
                continue;
            }
            edges.push(RouteEdge {
                src: PortRef {
                    col,
                    row,
                    port: src_port.index,
                    dir: PortDir::Master,
                    kind: src_port.port_type.as_kind_str().to_owned(),
                },
                dst: PortRef {
                    col,
                    row,
                    port: dst_port.index,
                    dir: PortDir::Slave,
                    kind: dst_port.port_type.as_kind_str().to_owned(),
                },
                kind: EdgeKind::DmaBufferRelay,
            });
        }
    }

    edges
}

/// Compose every tile's intra-tile edges (A3) with every enabled master's
/// inter-tile edge (A2) into a single `StreamRouteGraph` for the full array.
///
/// # Algorithm
///
/// 1. Build a `kind_at(col, row) -> Option<TileKind>` closure from the live
///    array.  This delegates off-array detection to `TileArray::get`, which
///    returns `None` for out-of-bounds coordinates — exactly the semantics
///    `inter_tile_dest` expects.  The live array is used (not a hardcoded
///    row→kind table) so the dest-kind is always faithful to the loaded device.
///
/// 2. For each tile, add its intra-tile edges (circuit + packet crossbar).
///
/// 3. For each tile, iterate its master ports.  For each *enabled* master,
///    call `inter_tile_dest` to find the neighbor slave.  If `Some`, add an
///    `EdgeKind::InterTile` edge from (tile's master) → (neighbor's slave).
///
///    The source `PortRef` carries the master's port index and kind string
///    (from `StreamPort::port_type.as_kind_str()`), consistent with how
///    `intra_tile_edges` labels the same ports.
///
/// # Edge semantics
///
/// - `EdgeKind::Circuit` / `EdgeKind::Packet` — intra-tile crossbar
///   (from A3, unmodified).
/// - `EdgeKind::InterTile` — physical wire between adjacent tiles
///   (master on source tile → slave on neighbor tile).
impl crate::device::state::DeviceState {
    /// Resolve the full stream-switch route graph for the configured array.
    ///
    /// See module-level documentation and the `resolve_route_graph` doc-comment
    /// above for the algorithm.
    pub fn resolve_route_graph(&self) -> StreamRouteGraph {
        let mut g = StreamRouteGraph::new();

        // kind_at closure: delegates to the live array so off-array returns None.
        // Captures `self.array` by reference; the closure is short-lived (used
        // only in the loop below) so the borrow is trivially satisfied.
        let kind_at = |c: u8, r: u8| self.array.get(c, r).map(|t| t.tile_kind);

        for tile in self.array.iter() {
            // --- Intra-tile edges (A3) ---
            let intra = intra_tile_edges(tile);
            for edge in intra {
                g.add_edge(edge);
            }

            // --- Intra-tile DMA buffer-relay edges (E2) ---
            // S2MM (writer) -> MM2S (reader) memory hops within the tile. The
            // `s2mm_count` is the per-tile S2MM channel count, taken from the
            // same authoritative source the dump uses (`shim_mux_s2mm_masters`
            // is sized to `params.dma_s2mm_channels` for every tile kind).
            if let Some(dma) = self.array.dma_engine(tile.col, tile.row) {
                let s2mm_count = tile.shim_mux_s2mm_masters.len();
                for edge in dma_buffer_relay_edges(tile, dma, s2mm_count) {
                    g.add_edge(edge);
                }
            }

            // --- Inter-tile edges (A2) ---
            // Walk every master port.  Only enabled masters can physically
            // drive the inter-tile wire; disabled masters are not connected.
            for (master_idx, master_port) in tile.stream_switch.masters.iter().enumerate() {
                if !master_port.enabled {
                    continue;
                }
                let Some(dst) =
                    inter_tile_dest(tile.tile_kind, tile.col, tile.row, master_idx as u8, &kind_at)
                else {
                    continue;
                };
                g.add_edge(RouteEdge {
                    src: PortRef {
                        col: tile.col,
                        row: tile.row,
                        port: master_idx as u8,
                        dir: PortDir::Master,
                        kind: master_port.port_type.as_kind_str().to_owned(),
                    },
                    dst,
                    kind: EdgeKind::InterTile,
                });
            }
        }

        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        // Wire-format assertions: the JSON tokens are a cross-language contract
        // with the Python loader, which expects lowercase/snake_case. A bare
        // round-trip would pass even if these regressed to PascalCase (Rust
        // would serialize and deserialize "Master" self-consistently), so we
        // assert the literal tokens here.
        assert!(json.contains("\"master\""), "PortDir must serialize lowercase: {json}");
        assert!(json.contains("\"slave\""), "PortDir must serialize lowercase: {json}");
        assert!(json.contains("\"inter_tile\""), "EdgeKind must serialize snake_case: {json}");
        assert!(json.contains("\"north\""), "PortType kind string must be present: {json}");
        assert!(json.contains("\"south\""), "PortType kind string must be present: {json}");

        let back: StreamRouteGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(back.edges.len(), 1);
        assert_eq!(back.edges[0].src.port, 12);
        assert_eq!(back.edges[0].kind, EdgeKind::InterTile);
        // Exercise the `dir` field post-round-trip in both directions.
        assert_eq!(back.edges[0].src.dir, PortDir::Master);
        assert_eq!(back.edges[0].dst.dir, PortDir::Slave);
    }

    // NPU1 array: 5 cols × 6 rows (rows 0-5; row 0 = shim, row 1 = memtile, rows 2-5 = compute)
    const COLS: u8 = 5;
    const ROWS: u8 = 6;

    /// Test-local tile-kind lookup mirroring the NPU1 column layout:
    /// row 0 = shim, row 1 = memtile, rows 2+ = compute. Returns `None` for any
    /// coordinate outside the 5×6 array, which is exactly how the real array's
    /// lookup (A4's `kind_at`) reports off-array neighbors. This is the static
    /// analogue of `self.tiles[idx].tile_kind` -- the function under test must
    /// NOT bake in this layout itself.
    fn npu1_kind_at(col: u8, row: u8) -> Option<TileKind> {
        if col >= COLS || row >= ROWS {
            return None;
        }
        Some(match row {
            0 => TileKind::ShimNoc,
            1 => TileKind::Mem,
            _ => TileKind::Compute,
        })
    }

    // --- inter_tile_dest tests ---
    // All expected values are traced from xdna_archspec::aie2::stream_switch constants
    // (gen_stream_ranges.rs), mirroring propagate_inter_tile() exactly.

    /// Shim (1,0) north master at shim::NORTH_MASTER_START (12)
    /// → MemTile (1,1) south slave at mem_tile::SOUTH_SLAVE_START (7).
    /// Verifies the shim→memtile upward wire using the first port of each range.
    #[test]
    fn inter_tile_shim_north_master_reaches_memtile_south_slave() {
        // shim::NORTH_MASTER_START = 12; mem_tile::SOUTH_SLAVE_START = 7
        let d = inter_tile_dest(TileKind::ShimNoc, 1, 0, shim::NORTH_MASTER_START, npu1_kind_at)
            .expect("shim north master must have a destination");
        assert_eq!((d.col, d.row), (1, 1), "destination must be the memtile directly above");
        assert_eq!(d.dir, PortDir::Slave);
        assert_eq!(d.port, mem_tile::SOUTH_SLAVE_START, "must land on memtile SOUTH_SLAVE_START");
        assert_eq!(d.kind, "south", "slave-side port kind is 'south'");
    }

    /// Shim (1,0) north master at the last port of the range, shim::NORTH_MASTER_END (17)
    /// → MemTile (1,1) south slave at mem_tile::SOUTH_SLAVE_START + (17-12) = 12.
    /// Verifies the 1:1 index offset arithmetic at the far end of the range.
    #[test]
    fn inter_tile_shim_north_master_end_reaches_memtile_south_slave_end() {
        // shim::NORTH_MASTER_END = 17; mem_tile::SOUTH_SLAVE_END = 12
        // offset = 17 - 12 = 5; slave = 7 + 5 = 12
        let d = inter_tile_dest(TileKind::ShimNoc, 1, 0, shim::NORTH_MASTER_END, npu1_kind_at)
            .expect("shim north master END must have a destination");
        assert_eq!((d.col, d.row), (1, 1));
        assert_eq!(d.port, mem_tile::SOUTH_SLAVE_END);
    }

    /// MemTile (1,1) north master at mem_tile::NORTH_MASTER_START (11)
    /// → Compute (1,2) south slave at compute::SOUTH_SLAVE_START (5).
    #[test]
    fn inter_tile_memtile_north_master_reaches_compute_south_slave() {
        // mem_tile::NORTH_MASTER_START = 11; compute::SOUTH_SLAVE_START = 5
        let d = inter_tile_dest(TileKind::Mem, 1, 1, mem_tile::NORTH_MASTER_START, npu1_kind_at)
            .expect("memtile north master must have a destination");
        assert_eq!((d.col, d.row), (1, 2));
        assert_eq!(d.dir, PortDir::Slave);
        assert_eq!(d.port, compute::SOUTH_SLAVE_START);
        assert_eq!(d.kind, "south");
    }

    /// Compute (1,3) north master at compute::NORTH_MASTER_START (13)
    /// → Compute (1,4) south slave at compute::SOUTH_SLAVE_START (5).
    #[test]
    fn inter_tile_compute_north_master_reaches_compute_south_slave() {
        // compute::NORTH_MASTER_START = 13; compute::SOUTH_SLAVE_START = 5
        let d = inter_tile_dest(TileKind::Compute, 1, 3, compute::NORTH_MASTER_START, npu1_kind_at)
            .expect("compute north master must have a destination");
        assert_eq!((d.col, d.row), (1, 4));
        assert_eq!(d.dir, PortDir::Slave);
        assert_eq!(d.port, compute::SOUTH_SLAVE_START);
        assert_eq!(d.kind, "south");
    }

    /// A north master on the top compute row (row 5 in a 6-row array) points
    /// off-array → None.  This is the array-edge bound check (kind_at returns None).
    #[test]
    fn inter_tile_array_edge_has_no_dest() {
        // compute::NORTH_MASTER_START = 13; row 5 is the top row (ROWS-1 = 5), no row 6
        assert!(
            inter_tile_dest(TileKind::Compute, 1, 5, compute::NORTH_MASTER_START, npu1_kind_at).is_none(),
            "north master at top row must return None"
        );
    }

    /// MemTile (1,1) south master at mem_tile::SOUTH_MASTER_START (7)
    /// → Shim (1,0) north slave at shim::NORTH_SLAVE_START (14).
    #[test]
    fn inter_tile_memtile_south_master_reaches_shim_north_slave() {
        // mem_tile::SOUTH_MASTER_START = 7; shim::NORTH_SLAVE_START = 14
        let d = inter_tile_dest(TileKind::Mem, 1, 1, mem_tile::SOUTH_MASTER_START, npu1_kind_at)
            .expect("memtile south master must reach shim");
        assert_eq!((d.col, d.row), (1, 0));
        assert_eq!(d.dir, PortDir::Slave);
        assert_eq!(d.port, shim::NORTH_SLAVE_START);
        assert_eq!(d.kind, "north");
    }

    /// Compute (1,2) south master at compute::SOUTH_MASTER_START (5), with a
    /// MemTile below (the compute-row-2 boundary) → MemTile (1,1) north slave at
    /// mem_tile::NORTH_SLAVE_START (13). This pins the (Compute, Mem) south arm.
    #[test]
    fn inter_tile_compute_south_master_to_memtile_uses_memtile_north_slave() {
        // compute::SOUTH_MASTER_START = 5; mem_tile::NORTH_SLAVE_START = 13
        let d = inter_tile_dest(TileKind::Compute, 1, 2, compute::SOUTH_MASTER_START, npu1_kind_at)
            .expect("compute south master must reach memtile");
        assert_eq!((d.col, d.row), (1, 1));
        assert_eq!(d.dir, PortDir::Slave);
        assert_eq!(d.port, mem_tile::NORTH_SLAVE_START, "below a MemTile -> mem_tile::NORTH_SLAVE_START");
        assert_eq!(d.kind, "north");
    }

    /// Compute (1,3) south master at compute::SOUTH_MASTER_START (5), with a
    /// Compute below (compute rows 3+ boundary) → Compute (1,2) north slave at
    /// compute::NORTH_SLAVE_START (15). This pins the (Compute, Compute) south arm
    /// and is the regression guard for the dest_kind fork: it MUST differ from the
    /// (Compute, Mem) case above (15, not 13).
    #[test]
    fn inter_tile_compute_south_master_to_compute_uses_compute_north_slave() {
        // compute::SOUTH_MASTER_START = 5; compute::NORTH_SLAVE_START = 15
        let d = inter_tile_dest(TileKind::Compute, 1, 3, compute::SOUTH_MASTER_START, npu1_kind_at)
            .expect("compute south master must reach the compute tile below");
        assert_eq!((d.col, d.row), (1, 2));
        assert_eq!(d.dir, PortDir::Slave);
        assert_eq!(d.port, compute::NORTH_SLAVE_START, "below a Compute -> compute::NORTH_SLAVE_START");
        assert_eq!(d.kind, "north");
        // Explicit cross-check: the two south arms resolve to different ports.
        assert_ne!(
            compute::NORTH_SLAVE_START,
            mem_tile::NORTH_SLAVE_START,
            "the dest_kind fork is only meaningful if these constants differ"
        );
    }

    /// Compute (1,2) east master at compute::EAST_MASTER_START (19)
    /// → Compute (2,2) west slave at compute::WEST_SLAVE_START (11).
    #[test]
    fn inter_tile_compute_east_master_reaches_compute_west_slave() {
        // compute::EAST_MASTER_START = 19; compute::WEST_SLAVE_START = 11
        let d = inter_tile_dest(TileKind::Compute, 1, 2, compute::EAST_MASTER_START, npu1_kind_at)
            .expect("compute east master must reach east neighbor");
        assert_eq!((d.col, d.row), (2, 2));
        assert_eq!(d.dir, PortDir::Slave);
        assert_eq!(d.port, compute::WEST_SLAVE_START);
        assert_eq!(d.kind, "west");
    }

    /// Compute (2,2) west master at compute::WEST_MASTER_START (9)
    /// → Compute (1,2) east slave at compute::EAST_SLAVE_START (19).
    #[test]
    fn inter_tile_compute_west_master_reaches_compute_east_slave() {
        // compute::WEST_MASTER_START = 9; compute::EAST_SLAVE_START = 19
        let d = inter_tile_dest(TileKind::Compute, 2, 2, compute::WEST_MASTER_START, npu1_kind_at)
            .expect("compute west master must reach west neighbor");
        assert_eq!((d.col, d.row), (1, 2));
        assert_eq!(d.dir, PortDir::Slave);
        assert_eq!(d.port, compute::EAST_SLAVE_START);
        assert_eq!(d.kind, "east");
    }

    /// Shim (1,0) east master at shim::EAST_MASTER_START (18)
    /// → Shim (2,0) west slave at shim::WEST_SLAVE_START (10).
    #[test]
    fn inter_tile_shim_east_master_reaches_shim_west_slave() {
        // shim::EAST_MASTER_START = 18; shim::WEST_SLAVE_START = 10
        let d = inter_tile_dest(TileKind::ShimNoc, 1, 0, shim::EAST_MASTER_START, npu1_kind_at)
            .expect("shim east master must reach east neighbor");
        assert_eq!((d.col, d.row), (2, 0));
        assert_eq!(d.port, shim::WEST_SLAVE_START);
        assert_eq!(d.kind, "west");
    }

    /// A DMA master port on a compute tile (port index 1 = compute::DMA_MASTER_START)
    /// has no inter-tile destination — it feeds the on-tile DMA, not a neighbor.
    #[test]
    fn inter_tile_dma_master_has_no_dest() {
        // compute::DMA_MASTER_START = 1; not in any directional range
        assert!(
            inter_tile_dest(TileKind::Compute, 1, 2, compute::DMA_MASTER_START, npu1_kind_at).is_none(),
            "DMA master must not have an inter-tile destination"
        );
    }

    /// East master on the rightmost column has no east neighbor → None.
    #[test]
    fn inter_tile_array_east_edge_has_no_dest() {
        // col 4 is the last column (COLS-1 = 4) in a 5-col array
        assert!(
            inter_tile_dest(TileKind::Compute, 4, 2, compute::EAST_MASTER_START, npu1_kind_at).is_none(),
            "east master at rightmost column must return None"
        );
    }

    /// West master on the leftmost column has no west neighbor → None.
    #[test]
    fn inter_tile_array_west_edge_has_no_dest() {
        assert!(
            inter_tile_dest(TileKind::Compute, 0, 2, compute::WEST_MASTER_START, npu1_kind_at).is_none(),
            "west master at leftmost column must return None"
        );
    }

    // ========================================================================
    // A4: resolve_route_graph integration test
    // ========================================================================

    /// BFS reachability on a `StreamRouteGraph`.
    ///
    /// Returns `true` if there is a directed path from `src` to `dst` following
    /// the edges in the graph. Used to assert data-flow connectivity without
    /// requiring exact intermediate routes.
    fn reachable(g: &StreamRouteGraph, src: &PortRef, dst: &PortRef) -> bool {
        use std::collections::{HashSet, VecDeque};

        // Nodes are unified by PHYSICAL identity (col, row, port, dir), NOT by the
        // `kind` string. This is essential at the inter/intra seam: the same physical
        // slave port is produced as an inter_tile_dest `dst` with kind="south"
        // (the static mirror-direction label) and as an intra_tile_edges `src` with
        // kind from `port_type.as_kind_str()`. Those labels can differ, but they are
        // the SAME physical port and must traverse as one node. Keying on physical
        // identity makes the graph compose across the seam regardless of label.
        type PhysKey = (u8, u8, u8, bool); // (col, row, port, is_master)
        let key = |p: &PortRef| -> PhysKey { (p.col, p.row, p.port, p.dir == PortDir::Master) };

        let dst_key = key(dst);
        let src_key = key(src);
        if src_key == dst_key {
            return true;
        }

        let mut visited: HashSet<PhysKey> = HashSet::new();
        let mut queue: VecDeque<PhysKey> = VecDeque::new();
        queue.push_back(src_key);
        visited.insert(src_key);

        while let Some(cur) = queue.pop_front() {
            for edge in g.edges.iter().filter(|e| key(&e.src) == cur) {
                let dk = key(&edge.dst);
                if dk == dst_key {
                    return true;
                }
                if visited.insert(dk) {
                    queue.push_back(dk);
                }
            }
        }
        false
    }

    /// Load an xclbin from disk and return a fully-configured `DeviceState`,
    /// or `None` if the fixture is absent.
    fn load_npu1_state(path: &str) -> Option<crate::device::DeviceState> {
        use crate::parser::xclbin::SectionKind;
        use crate::parser::{Xclbin, AiePartition};
        use crate::parser::cdo::{find_cdo_offset, Cdo};

        if !std::path::Path::new(path).exists() {
            return None;
        }
        let xclbin = Xclbin::from_file(path).ok()?;
        let section = xclbin.find_section(SectionKind::AiePartition)?;
        let partition = AiePartition::parse(section.data()).ok()?;
        let pdi = partition.primary_pdi()?;
        let cdo_offset = find_cdo_offset(pdi.pdi_image)?;
        let cdo = Cdo::parse(&pdi.pdi_image[cdo_offset..]).ok()?;

        let mut state = crate::device::DeviceState::new_npu1();
        state.apply_cdo(&cdo).ok()?;
        Some(state)
    }

    /// Full integration test: resolve_route_graph on add_one_using_dma.
    ///
    /// Asserts:
    /// 1. The graph is non-empty.
    /// 2. Shim tile(0,0) has at least one inter-tile edge (data crosses the
    ///    shim→memtile boundary).
    /// 3. Memtile (row 1) has at least one intra-tile edge (non-InterTile),
    ///    asserting that A3 is wired in for the memtile.
    /// 4. The single-hop 1:1 index arithmetic holds for every CDO-enabled shim
    ///    north master edge landing on shim(0,0) (regression guard; the same
    ///    property A2 tests in isolation).
    /// 5. **Composition BFS 1 (shim→memtile boundary):** `reachable` finds a
    ///    path from a shim(0,0) north master through an inter-tile edge to a
    ///    memtile south slave, then through the memtile's intra-tile crossbar to a
    ///    memtile DMA master.  Proves the inter/intra SEAM composes: the InterTile
    ///    dst slave and the Circuit src slave unify by physical identity.
    /// 6. **Composition BFS 2 (memtile→compute boundary):** `reachable` finds a
    ///    path from a memtile DMA slave through its intra-tile crossbar to a
    ///    memtile north master, then through a second inter-tile edge to a compute
    ///    tile south slave.  Together with BFS 1 this covers both tile boundaries
    ///    in the add_one data path.  The DMA engine bridges the two BFS segments
    ///    (it is a memory relay, not a stream-switch crossbar route).
    #[test]
    fn resolve_route_graph_add_one() {
        let path = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin";
        let Some(state) = load_npu1_state(path) else {
            println!("SKIP: fixture absent at {}", path);
            return;
        };

        let g = state.resolve_route_graph();
        assert!(!g.edges.is_empty(), "graph must have edges after CDO load");

        // Shim tile (0,0) must have at least one inter-tile edge leaving it
        // (shim north master → memtile south slave).
        assert!(
            g.edges
                .iter()
                .any(|e| e.src.col == 0 && e.src.row == 0 && e.kind == EdgeKind::InterTile),
            "shim (0,0) must have at least one InterTile edge"
        );

        // Memtile row=1 must have at least one intra-tile edge (circuit or packet).
        assert!(
            g.edges
                .iter()
                .any(|e| e.src.row == 1 && e.dst.row == 1 && e.kind != EdgeKind::InterTile),
            "memtile (row 1) must have at least one intra-tile edge"
        );

        // --- Single-hop arithmetic (regression guard, kept alongside BFS) ---
        // Every inter-tile edge from the shim (row=0) must land on the memtile
        // (row=1) with the correct slave port via 1:1 index arithmetic, regardless
        // of which specific master port(s) the CDO activates.
        let shim_inter_tile_edges: Vec<_> = g
            .edges
            .iter()
            .filter(|e| e.src.col == 0 && e.src.row == 0 && e.kind == EdgeKind::InterTile)
            .collect();
        for edge in &shim_inter_tile_edges {
            assert_eq!(edge.dst.row, 1, "shim inter-tile edge must land on memtile (row 1)");
            assert_eq!(edge.dst.dir, PortDir::Slave, "destination must be a slave port");
            assert_eq!(edge.dst.kind, "south", "memtile south slave kind must be 'south'");
            let master_offset = edge.src.port - xdna_archspec::aie2::stream_switch::shim::NORTH_MASTER_START;
            let expected_slave =
                xdna_archspec::aie2::stream_switch::mem_tile::SOUTH_SLAVE_START + master_offset;
            assert_eq!(
                edge.dst.port, expected_slave,
                "shim north master port {} should reach memtile south slave port {}",
                edge.src.port, expected_slave
            );
        }

        // --- Composition via multi-hop BFS (the A4 core property) ---
        //
        // TOPOLOGY FINDING (add_one_using_dma): the memtile is a DMA *relay*, not a
        // stream-switch pass-through. The real data path is:
        //   (0,0) M16  --InterTile--> (0,1) S11(south)
        //   (0,1) S11  --Circuit-->   (0,1) M0(dma)       [into memtile S2MM buffer]
        //   (0,1) S0(dma) --Circuit-> (0,1) M12(north)    [out of memtile MM2S buffer]
        //   (0,1) M12  --InterTile--> (0,2) S6(south)     [into compute]
        // The static stream-switch graph deliberately does NOT bridge the DMA buffer
        // (M0_dma -> S0_dma is a memory hop through the DMA engine, not a crossbar
        // route), so there is NO pure-switch two-tile-boundary path in this kernel.
        // We therefore assert composition across the shim->memtile boundary PLUS the
        // memtile's intra-tile crossbar hop (inter-tile edge dst -> intra Circuit edge
        // src -> crossbar master). This is exactly the inter/intra SEAM that A2 and A3
        // never exercise individually: BFS must unify the InterTile dst slave with the
        // intra Circuit src slave by physical identity to traverse it.
        let src = shim_inter_tile_edges
            .first()
            .map(|e| e.src.clone())
            .expect("shim(0,0) must have an inter-tile source port");

        // Resolve the sink DYNAMICALLY (CDO-agnostic): take the memtile slave the
        // shim's inter-tile edge lands on, then follow the memtile's intra-tile
        // crossbar to whichever master it feeds. That downstream master is the sink.
        let memtile_slave = shim_inter_tile_edges
            .first()
            .map(|e| e.dst.clone())
            .expect("shim inter-tile edge must have a memtile destination");
        let crossbar_master = g
            .edges
            .iter()
            .find(|e| {
                e.kind != EdgeKind::InterTile
                    && e.src.col == memtile_slave.col
                    && e.src.row == memtile_slave.row
                    && e.src.port == memtile_slave.port
                    && e.src.dir == PortDir::Slave
            })
            .map(|e| e.dst.clone())
            .expect("memtile must route its south slave through an intra-tile crossbar edge");

        // The sink is on a DIFFERENT tile from the source (boundary crossed) and is a
        // master reached only via an intra-tile crossbar edge (composition required).
        assert_eq!(src.row, 0, "source is on the shim");
        assert_eq!(crossbar_master.row, 1, "sink is inside the memtile (one boundary up)");
        assert_eq!(crossbar_master.dir, PortDir::Master, "sink is a crossbar master");

        assert!(
            reachable(&g, &src, &crossbar_master),
            "shim source {:?} must reach memtile crossbar master {:?} via composed \
             inter-tile + intra-tile edges (the seam)",
            src,
            crossbar_master
        );

        // --- Second boundary: memtile → compute (BFS 2 of 2) ---
        //
        // The DMA engine bridges the two BFS segments (M0_dma → S0_dma is
        // memory, not a switch route). BFS 2 starts at the memtile MM2S slave
        // (the DMA output port) and proves it reaches a compute tile slave via
        // the memtile's intra crossbar + a second inter-tile edge. Together
        // BFS 1 (shim→memtile boundary) and BFS 2 (memtile→compute boundary)
        // cover both tile boundaries in the add_one data path.
        //
        // Resolve the MM2S slave DYNAMICALLY: find a memtile intra-tile edge
        // that feeds a north master (which then crosses into compute). From
        // that intra edge's src (a memtile slave) run BFS 2.
        let memtile_to_compute_inter = g
            .edges
            .iter()
            .find(|e| e.kind == EdgeKind::InterTile && e.src.row == 1 && e.dst.row >= 2)
            .expect("memtile→compute inter-tile edge must exist in add_one");
        let memtile_north_master = &memtile_to_compute_inter.src;
        let compute_slave = memtile_to_compute_inter.dst.clone();

        // Find the intra-tile edge inside the memtile that feeds this north master.
        let memtile_mm2s_slave = g
            .edges
            .iter()
            .find(|e| {
                e.kind != EdgeKind::InterTile
                    && e.dst.col == memtile_north_master.col
                    && e.dst.row == memtile_north_master.row
                    && e.dst.port == memtile_north_master.port
                    && e.dst.dir == PortDir::Master
            })
            .map(|e| e.src.clone())
            .expect("memtile north master must be fed by an intra-tile crossbar edge");

        assert_eq!(memtile_mm2s_slave.row, 1, "BFS 2 source is on the memtile");
        assert!(compute_slave.row >= 2, "BFS 2 sink is on a compute tile");

        assert!(
            reachable(&g, &memtile_mm2s_slave, &compute_slave),
            "memtile slave {:?} must reach compute slave {:?} via composed \
             intra-tile + inter-tile edges (memtile→compute boundary)",
            memtile_mm2s_slave,
            compute_slave
        );
    }

    // ========================================================================
    // E2: DmaBufferRelay edges (intra-tile memory hop)
    // ========================================================================

    /// The memtile in add_one_using_dma relays each buffer through its DMA: an
    /// S2MM channel writes a buffer that an MM2S channel later reads. The static
    /// graph must surface this as a directed `DmaBufferRelay` edge from the S2MM
    /// master DMA port (the buffer writer) to the MM2S slave DMA port (the
    /// reader). This is the memory hop the stream-switch crossbar does not model,
    /// and it is what orients the memtile DMA trace events for the config_path
    /// generator.
    ///
    /// Convention (authoritative per `src/device/array/routing.rs`):
    ///   - master DMA port = S2MM channel (buffer writer)
    ///   - slave  DMA port = MM2S channel (buffer reader)
    /// so the relay edge is `src = S2MM master port -> dst = MM2S slave port`.
    /// The reverse (MM2S slave -> S2MM master) is back-pressure, never dataflow,
    /// and must NOT appear.
    ///
    /// add_one memtile (0,1) buffers (post-E1):
    ///   - in0  (0x80000/0x80040): S2MM0 BD0/1 writes, MM2S0 BD2/3 reads -> edge M0:dma -> S0:dma
    ///   - out0 (0x80080/0x800c0): S2MM1 BD26/27 writes, MM2S1 BD24/25 reads -> edge M1:dma -> S1:dma
    /// in0 and out0 ranges are disjoint, so no false cross-overlap exists.
    #[test]
    fn resolve_route_graph_dma_buffer_relay_add_one() {
        let path = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin";
        let Some(state) = load_npu1_state(path) else {
            println!("SKIP: fixture absent at {}", path);
            return;
        };

        let g = state.resolve_route_graph();

        // Helper: does an edge of the given kind exist whose src/dst land on the
        // given (col,row,port,dir,kind)?
        let has_relay = |src_port: u8, src_dir: PortDir, dst_port: u8, dst_dir: PortDir| -> bool {
            g.edges.iter().any(|e| {
                e.kind == EdgeKind::DmaBufferRelay
                    && e.src.col == 0
                    && e.src.row == 1
                    && e.src.port == src_port
                    && e.src.dir == src_dir
                    && e.src.kind == "dma"
                    && e.dst.col == 0
                    && e.dst.row == 1
                    && e.dst.port == dst_port
                    && e.dst.dir == dst_dir
                    && e.dst.kind == "dma"
            })
        };

        // (a) channel-0 buffer relay: S2MM0 master port 0 -> MM2S0 slave port 0.
        assert!(
            has_relay(0, PortDir::Master, 0, PortDir::Slave),
            "expected DmaBufferRelay (0,1) M0:dma -> S0:dma (in0 buffer); edges: {:#?}",
            g.edges
                .iter()
                .filter(|e| e.kind == EdgeKind::DmaBufferRelay)
                .collect::<Vec<_>>()
        );

        // (b) channel-1 buffer relay: S2MM1 master port 1 -> MM2S1 slave port 1.
        assert!(
            has_relay(1, PortDir::Master, 1, PortDir::Slave),
            "expected DmaBufferRelay (0,1) M1:dma -> S1:dma (out0 buffer); edges: {:#?}",
            g.edges
                .iter()
                .filter(|e| e.kind == EdgeKind::DmaBufferRelay)
                .collect::<Vec<_>>()
        );

        // (c) the REVERSE edges (MM2S slave -> S2MM master) must NOT exist; that
        //     would be back-pressure, not dataflow.
        assert!(
            !has_relay(0, PortDir::Slave, 0, PortDir::Master),
            "reverse DmaBufferRelay S0:dma -> M0:dma must NOT exist (back-pressure)"
        );
        assert!(
            !has_relay(1, PortDir::Slave, 1, PortDir::Master),
            "reverse DmaBufferRelay S1:dma -> M1:dma must NOT exist (back-pressure)"
        );

        // (d) no false cross-buffer relay: in0 (ch0) and out0 (ch1) are disjoint,
        //     so there is no relay between channel 0's writer and channel 1's
        //     reader (or vice versa).
        assert!(
            !has_relay(0, PortDir::Master, 1, PortDir::Slave),
            "no relay should exist between disjoint in0 (S2MM0) and out0 (MM2S1)"
        );
        assert!(
            !has_relay(1, PortDir::Master, 0, PortDir::Slave),
            "no relay should exist between disjoint out0 (S2MM1) and in0 (MM2S0)"
        );

        // (e) pre-existing edge classes are unchanged: the inter-tile,
        //     circuit, and packet edges must still be present (regression guard).
        assert!(
            g.edges.iter().any(|e| e.kind == EdgeKind::InterTile),
            "InterTile edges must still exist alongside DmaBufferRelay"
        );
        assert!(
            g.edges.iter().any(|e| e.kind == EdgeKind::Circuit),
            "Circuit edges must still exist alongside DmaBufferRelay"
        );
    }

    // ========================================================================
    // A5: static graph ⊇ dynamic enactment
    // ========================================================================

    /// Soundness gate: every inter-tile hop the emulator *actually enacts* at
    /// runtime for add_one_using_dma must be present as an `EdgeKind::InterTile`
    /// edge in the static route graph.
    ///
    /// If any enacted hop is missing from the static graph the static
    /// reconstruction is unsound (it is a parallel guess, not a faithful
    /// mirror of the runtime).  A failure here is a real graph bug — report it
    /// loudly rather than masking it.
    ///
    /// Node identity is purely physical: `(col, row, port, dir)`.  The `kind`
    /// string may differ between the static edge (derived from `PortType`) and
    /// the recorded hop (also from `PortType` but on the live tile) — both
    /// paths use the same source, so in practice they agree, but the comparison
    /// intentionally ignores `kind` to stay robust to future label changes.
    #[test]
    fn static_graph_is_superset_of_enacted_inter_tile_hops() {
        use crate::interpreter::engine::{InterpreterEngine, EngineStatus};
        use crate::npu::{NpuInstructionStream, NpuExecutor, HostBuffer};

        let xclbin_path =
            "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin";
        let insts_path =
            "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/insts.bin";

        if !std::path::Path::new(xclbin_path).exists() {
            eprintln!("SKIP: fixture absent at {}", xclbin_path);
            return;
        }

        // --- Load the xclbin and build a configured DeviceState ---
        use crate::parser::xclbin::SectionKind;
        use crate::parser::{Xclbin, AiePartition};
        use crate::parser::cdo::{find_cdo_offset, Cdo};

        let xclbin = Xclbin::from_file(xclbin_path).expect("parse xclbin");
        let section = xclbin.find_section(SectionKind::AiePartition).expect("AiePartition section");
        let partition = AiePartition::parse(section.data()).expect("parse partition");
        let pdi = partition.primary_pdi().expect("primary PDI");
        let cdo_offset = find_cdo_offset(pdi.pdi_image).expect("CDO offset");
        let cdo = Cdo::parse(&pdi.pdi_image[cdo_offset..]).expect("parse CDO");

        // --- Build static route graph BEFORE running ---
        // Apply CDO to a DeviceState just for graph extraction.
        let mut state_for_graph = crate::device::DeviceState::new_npu1();
        state_for_graph.apply_cdo(&cdo).expect("apply CDO for graph");
        let graph = state_for_graph.resolve_route_graph();

        // Key each InterTile edge by physical (col,row,port,dir) of src AND dst.
        // The `kind` string is intentionally excluded from the key — the static
        // and runtime paths both derive it from `PortType::as_kind_str()` on the
        // same port, so they agree in practice, but the comparison is more
        // robust when keyed on physical identity alone.
        type PhysKey = (u8, u8, u8, bool); // (col, row, port, is_master)
        let phys = |p: &PortRef| -> PhysKey { (p.col, p.row, p.port, p.dir == PortDir::Master) };

        let static_edge_set: std::collections::HashSet<(PhysKey, PhysKey)> = graph
            .edges
            .iter()
            .filter(|e| e.kind == EdgeKind::InterTile)
            .map(|e| (phys(&e.src), phys(&e.dst)))
            .collect();

        // --- Set up the engine and enable hop recording ---
        let mut engine = InterpreterEngine::new_npu1();
        engine.device_mut().assign_partition_columns(0, partition.column_width() as u8);
        engine.device_mut().apply_cdo(&cdo).expect("apply CDO to engine");

        // Enable hop recording on the live array.
        engine.device_mut().array.enable_hop_recording();

        // --- Load NPU instructions (insts.bin) to drive shim DMA ---
        let mut npu_executor: Option<NpuExecutor> = None;
        if std::path::Path::new(insts_path).exists() {
            let insts_data = std::fs::read(insts_path).expect("read insts.bin");
            if !insts_data.is_empty() {
                let stream = NpuInstructionStream::parse(&insts_data).expect("parse insts");
                let mut executor = NpuExecutor::new();
                // Default host-buffer layout matching xclbin_suite defaults.
                executor.set_host_buffers(vec![
                    HostBuffer { address: 0x0000, size: 4096 },
                    HostBuffer { address: 0x1000, size: 256 },
                    HostBuffer { address: 0x2000, size: 4096 },
                ]);
                executor.load(&stream);
                npu_executor = Some(executor);
            }
        }

        // Load ELF files from the chess project directory.
        let prj_dir =
            "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.mlir.prj";
        if std::path::Path::new(prj_dir).exists() {
            let entries = std::fs::read_dir(prj_dir).expect("read prj dir");
            for entry in entries.flatten() {
                let path = entry.path();
                let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                if name.ends_with(".elf") && name.contains("core_") {
                    // Parse col/row from "main_core_C_R.elf" or "core_C_R.elf"
                    if let Some(coords) = parse_core_coords_a5(&name) {
                        let data = std::fs::read(&path).expect("read ELF");
                        engine
                            .load_elf_bytes(coords.0 as usize, coords.1 as usize, &data)
                            .expect("load ELF");
                    }
                }
            }
        }
        engine.sync_cores_from_device();

        // --- Run to completion (mirrors xclbin_suite::run_engine logic) ---
        const MAX_CYCLES: u64 = 500_000;
        let mut cycles = 0u64;
        while cycles < MAX_CYCLES {
            if let Some(executor) = npu_executor.as_mut() {
                let (device, host_mem) = engine.device_and_host_memory();
                if let crate::npu::AdvanceResult::Error(msg) = executor.try_advance(device, host_mem) {
                    panic!("NPU executor fatal at cycle {}: {}", cycles, msg);
                }
            }
            engine.step();
            cycles = engine.total_cycles();

            match engine.status() {
                EngineStatus::Halted => {
                    let pending = npu_executor
                        .as_mut()
                        .map(|ex| !ex.is_done() || !ex.syncs_satisfied(engine.device()))
                        .unwrap_or(false);
                    if pending {
                        engine.force_running();
                    } else {
                        break; // done
                    }
                }
                EngineStatus::Error => {
                    panic!("engine error at cycle {}", cycles);
                }
                EngineStatus::Running
                | EngineStatus::Ready
                | EngineStatus::Paused
                | EngineStatus::Stalled => {}
            }

            if let Some(executor) = npu_executor.as_mut() {
                if executor.is_done() && executor.syncs_satisfied(engine.device()) {
                    break; // DMA sync conditions satisfied
                }
            }
        }

        // --- Drain recorded hops ---
        let enacted: Vec<_> = engine.device_mut().array.take_recorded_hops();

        // The kernel must have taken at least one inter-tile hop.
        assert!(
            !enacted.is_empty(),
            "add_one must enact at least one inter-tile hop (recorded {} hops after {} cycles)",
            enacted.len(),
            cycles
        );

        println!("A5: {} inter-tile hops enacted over {} cycles", enacted.len(), cycles);

        // --- Superset check: every enacted hop must be in the static graph ---
        let mut missing: Vec<String> = Vec::new();
        for (s, d) in &enacted {
            let key = (phys(s), phys(d));
            if !static_edge_set.contains(&key) {
                missing.push(format!(
                    "  ENACTED HOP NOT IN STATIC GRAPH: ({},{}) master[{}] kind={:?} \
                     -> ({},{}) slave[{}] kind={:?}",
                    s.col, s.row, s.port, s.kind, d.col, d.row, d.port, d.kind,
                ));
            }
        }

        if !missing.is_empty() {
            panic!(
                "STATIC GRAPH SOUNDNESS FAILURE — {} enacted hops missing from static graph:\n{}",
                missing.len(),
                missing.join("\n")
            );
        }

        println!(
            "A5 PASS: all {} enacted hops are present in the static graph ({} InterTile edges)",
            enacted.len(),
            static_edge_set.len()
        );
    }

    /// Parse core coordinates from a filename like "main_core_0_2.elf".
    fn parse_core_coords_a5(name: &str) -> Option<(u8, u8)> {
        let core_idx = name.find("core_")?;
        let after_core = &name[core_idx + 5..];
        let parts: Vec<&str> = after_core.split('_').take(2).collect();
        if parts.len() >= 2 {
            let col: u8 = parts[0].parse().ok()?;
            let row_str = parts[1].trim_end_matches(".elf");
            let row: u8 = row_str.parse().ok()?;
            return Some((col, row));
        }
        None
    }

    // ========================================================================
    // A3: intra_tile_edges tests
    // ========================================================================

    use crate::device::tile::Tile;
    use crate::device::stream_switch::packet_switch::LocalRoute;

    /// An enabled LocalRoute produces a Circuit edge with the right ports and coords.
    #[test]
    fn intra_tile_circuit_edge_from_local_route() {
        let mut tile = Tile::compute(1, 2);
        tile.stream_switch.local_routes.push(LocalRoute {
            slave_idx: 3,
            master_idx: 7,
            enabled: true,
            latency: 3,
        });
        let edges = intra_tile_edges(&tile);
        let e = edges.iter().find(|e| e.kind == EdgeKind::Circuit).expect("circuit edge");
        assert_eq!(e.src.dir, PortDir::Slave);
        assert_eq!(e.dst.dir, PortDir::Master);
        assert_eq!((e.src.port, e.dst.port), (3, 7));
        assert_eq!((e.src.col, e.src.row), (1, 2));
        assert_eq!((e.dst.col, e.dst.row), (1, 2));
    }

    /// A disabled LocalRoute must not appear in the edge list.
    #[test]
    fn intra_tile_disabled_route_is_skipped() {
        let mut tile = Tile::compute(1, 2);
        tile.stream_switch.local_routes.push(LocalRoute {
            slave_idx: 3,
            master_idx: 7,
            enabled: false,
            latency: 3,
        });
        assert!(intra_tile_edges(&tile).iter().all(|e| e.kind != EdgeKind::Circuit));
    }

    /// A configured packet slot produces Packet edges to every accepting master.
    #[test]
    fn intra_tile_packet_edge_from_configured_slot() {
        let mut tile = Tile::compute(1, 2);
        let ss = &mut tile.stream_switch;

        // Slot: arbiter=0, msel=0, enabled — on slave port 23 (trace)
        ss.configure_slave_slot(23, 0, make_slot_reg_a3(0, 0x1F, 0, 0));
        // Master 7: packet_enable, arbiter=0, msel_enable=0b0001 (accepts msel=0)
        ss.configure_master_packet(7, make_master_pkt_reg_a3(0, 0b0001, false));

        let edges = intra_tile_edges(&tile);
        let pkt_edges: Vec<_> = edges.iter().filter(|e| e.kind == EdgeKind::Packet).collect();
        assert!(!pkt_edges.is_empty(), "should have at least one packet edge");

        let e = pkt_edges
            .iter()
            .find(|e| e.src.port == 23 && e.dst.port == 7)
            .expect("expected packet edge from slave 23 to master 7");
        assert_eq!(e.src.dir, PortDir::Slave);
        assert_eq!(e.dst.dir, PortDir::Master);
        assert_eq!((e.src.col, e.src.row), (1, 2));
    }

    /// A disabled slot (enable=false) must not produce packet edges.
    #[test]
    fn intra_tile_disabled_packet_slot_skipped() {
        let mut tile = Tile::compute(1, 2);
        let ss = &mut tile.stream_switch;

        // Slot with enable bit = 0 (no `1 << 8` in the register value)
        let disabled_slot_reg: u32 = (0u32 << 24) | (0x1Fu32 << 16) | (0 << 4) | 0; // enable=0
        ss.configure_slave_slot(23, 0, disabled_slot_reg);
        ss.configure_master_packet(7, make_master_pkt_reg_a3(0, 0b0001, false));

        let edges = intra_tile_edges(&tile);
        assert!(
            edges.iter().all(|e| e.kind != EdgeKind::Packet),
            "disabled slot must not produce packet edges"
        );
    }

    /// Circuit edge kind string is derived from PortType (not hardcoded).
    #[test]
    fn intra_tile_circuit_edge_kind_str_from_port_type() {
        // Slave 0 on a compute tile is Core; master 0 is Core.
        let mut tile = Tile::compute(1, 2);
        tile.stream_switch.configure_local_route(0, 0);
        let edges = intra_tile_edges(&tile);
        let e = edges.iter().find(|e| e.kind == EdgeKind::Circuit).expect("circuit edge");
        assert_eq!(e.src.kind, "core", "slave 0 on compute tile is Core");
        assert_eq!(e.dst.kind, "core", "master 0 on compute tile is Core");
    }

    // Register builder helpers local to A3 tests (mirrors tests::mod helpers
    // in tests.rs but scoped here to avoid cross-module dep).
    fn make_slot_reg_a3(pkt_id: u8, mask: u8, msel: u8, arbiter: u8) -> u32 {
        ((pkt_id as u32) << 24) | ((mask as u32) << 16) | (1 << 8) | ((msel as u32) << 4) | (arbiter as u32)
    }

    fn make_master_pkt_reg_a3(arbiter: u8, msel_enable: u8, drop_header: bool) -> u32 {
        (1 << 31)
            | (1 << 30)
            | if drop_header { 1 << 7 } else { 0 }
            | ((msel_enable as u32) << 3)
            | (arbiter as u32)
    }
}
