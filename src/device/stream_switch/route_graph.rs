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
//! - `EdgeKind`: serializes as snake_case `"inter_tile"` / `"circuit"` / `"packet"`.
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
