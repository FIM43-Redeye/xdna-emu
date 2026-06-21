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
        let back: StreamRouteGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(back.edges.len(), 1);
        assert_eq!(back.edges[0].src.port, 12);
        assert_eq!(back.edges[0].kind, EdgeKind::InterTile);
    }
}
