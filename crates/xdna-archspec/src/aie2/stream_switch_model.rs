//! AIE2 stream switch model implementation.
//!
//! Covers NPU1 (Phoenix), NPU4 / NPU5 / NPU6 (Strix / Strix Halo /
//! Krackan). All AIE2-family devices share the same stream-switch
//! feature set:
//!
//! - Deterministic merge: available on all tile types (compute,
//!   memtile, shim).
//! - Packet routing: invariant across arches (same slot/arbiter/msel
//!   mechanisms per aie-rt shared xaie_ss.c).
//! - Port counts: compute = 23 master / 25 slave; memtile = 17 / 18;
//!   shim = 22 / 23. Sourced from
//!   `mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json`
//!   `Stream_Switch_*_Config` registers.
//!
//! A drift-detection test in this module asserts the hand-written
//! `AIE2_STREAM_SWITCH_TOPOLOGY` aggregate still agrees with the
//! build.rs-generated per-field constants.

use crate::aie2::stream_switch::{compute, mem_tile, shim};
use crate::aie2::{
    COMPUTE_MASTER_PORTS, COMPUTE_SLAVE_PORTS, MEMTILE_MASTER_PORTS, MEMTILE_SLAVE_PORTS, SHIM_MASTER_PORTS,
    SHIM_SLAVE_PORTS,
};
use crate::stream_switch::{StreamSwitchModel, StreamSwitchTopology, TileStreamPorts};

/// The AIE2 stream switch topology.
///
/// Static so hot-path consumers can cache `&'static StreamSwitchTopology`
/// at construction time. Drift-detection test below asserts this
/// aggregate still agrees with the build.rs-generated constants
/// it aggregates.
pub static AIE2_STREAM_SWITCH_TOPOLOGY: StreamSwitchTopology = StreamSwitchTopology {
    compute: TileStreamPorts {
        master_ports: COMPUTE_MASTER_PORTS,
        slave_ports: COMPUTE_SLAVE_PORTS,
        north_master: (compute::NORTH_MASTER_START, compute::NORTH_MASTER_END),
        south_master: (compute::SOUTH_MASTER_START, compute::SOUTH_MASTER_END),
        north_slave: (compute::NORTH_SLAVE_START, compute::NORTH_SLAVE_END),
        south_slave: (compute::SOUTH_SLAVE_START, compute::SOUTH_SLAVE_END),
    },
    memtile: TileStreamPorts {
        master_ports: MEMTILE_MASTER_PORTS,
        slave_ports: MEMTILE_SLAVE_PORTS,
        north_master: (mem_tile::NORTH_MASTER_START, mem_tile::NORTH_MASTER_END),
        south_master: (mem_tile::SOUTH_MASTER_START, mem_tile::SOUTH_MASTER_END),
        north_slave: (mem_tile::NORTH_SLAVE_START, mem_tile::NORTH_SLAVE_END),
        south_slave: (mem_tile::SOUTH_SLAVE_START, mem_tile::SOUTH_SLAVE_END),
    },
    shim: TileStreamPorts {
        master_ports: SHIM_MASTER_PORTS,
        slave_ports: SHIM_SLAVE_PORTS,
        north_master: (shim::NORTH_MASTER_START, shim::NORTH_MASTER_END),
        // Shim south-facing ports are the external NoC interface;
        // (0, 0) sentinel models "no intra-array south."
        south_master: (0, 0),
        north_slave: (shim::NORTH_SLAVE_START, shim::NORTH_SLAVE_END),
        south_slave: (0, 0),
    },
};

/// AIE2 stream switch model.
///
/// Zero-sized: a single `AIE2_STREAM_SWITCH_MODEL` static serves
/// every tile in every AIE2-family NPU. `ArchConfig::stream_switch_model()`
/// returns a `&'static dyn StreamSwitchModel` pointing at this singleton.
#[derive(Debug, Clone, Copy)]
pub struct Aie2StreamSwitchModel;

/// The single `Aie2StreamSwitchModel` instance used across every
/// AIE2-family consumer. Reference via `ArchConfig::stream_switch_model()`.
pub static AIE2_STREAM_SWITCH_MODEL: Aie2StreamSwitchModel = Aie2StreamSwitchModel;

impl StreamSwitchModel for Aie2StreamSwitchModel {
    fn supports_deterministic_merge(&self) -> bool {
        true
    }

    fn topology(&self) -> &'static StreamSwitchTopology {
        &AIE2_STREAM_SWITCH_TOPOLOGY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TileKind;

    #[test]
    fn aie2_stream_switch_model_feature_flag() {
        assert!(
            AIE2_STREAM_SWITCH_MODEL.supports_deterministic_merge(),
            "AIE2 has deterministic-merge registers on all tile types"
        );
    }

    #[test]
    fn aie2_stream_switch_model_topology_returns_static() {
        // Two calls return identical &'static references.
        let topo1 = AIE2_STREAM_SWITCH_MODEL.topology() as *const _;
        let topo2 = AIE2_STREAM_SWITCH_MODEL.topology() as *const _;
        assert_eq!(topo1, topo2, "topology() must return the same static");
    }

    #[test]
    fn aie2_topology_port_counts() {
        let topo = &AIE2_STREAM_SWITCH_TOPOLOGY;
        // Per AM025: compute = 23 master / 25 slave;
        // memtile = 17 / 18; shim = 22 / 23.
        assert_eq!(topo.compute.master_ports.len(), 23, "compute master port count");
        assert_eq!(topo.compute.slave_ports.len(), 25, "compute slave port count");
        assert_eq!(topo.memtile.master_ports.len(), 17, "memtile master port count");
        assert_eq!(topo.memtile.slave_ports.len(), 18, "memtile slave port count");
        assert_eq!(topo.shim.master_ports.len(), 22, "shim master port count");
        assert_eq!(topo.shim.slave_ports.len(), 23, "shim slave port count");
    }

    #[test]
    fn aie2_topology_compute_ranges() {
        let ports = AIE2_STREAM_SWITCH_TOPOLOGY.for_tile(TileKind::Compute);
        // Per AM025 compute ranges.
        assert_eq!(ports.south_master, (5, 8));
        assert_eq!(ports.south_slave, (5, 10));
        assert_eq!(ports.north_master, (13, 18));
        assert_eq!(ports.north_slave, (15, 18));
    }

    #[test]
    fn aie2_topology_memtile_ranges() {
        let ports = AIE2_STREAM_SWITCH_TOPOLOGY.for_tile(TileKind::Mem);
        assert_eq!(ports.south_master, (7, 10));
        assert_eq!(ports.south_slave, (7, 12));
        assert_eq!(ports.north_master, (11, 16));
        assert_eq!(ports.north_slave, (13, 16));
    }

    #[test]
    fn aie2_topology_shim_ranges() {
        let ports = AIE2_STREAM_SWITCH_TOPOLOGY.for_tile(TileKind::ShimNoc);
        assert_eq!(ports.north_master, (12, 17));
        assert_eq!(ports.north_slave, (14, 17));
        // Sentinels: shim has no intra-array south.
        assert_eq!(ports.south_master, (0, 0));
        assert_eq!(ports.south_slave, (0, 0));
    }

    #[test]
    fn aie2_topology_shim_pl_equals_shim_noc() {
        let noc = AIE2_STREAM_SWITCH_TOPOLOGY.for_tile(TileKind::ShimNoc);
        let pl = AIE2_STREAM_SWITCH_TOPOLOGY.for_tile(TileKind::ShimPl);
        assert_eq!(noc.master_ports, pl.master_ports);
        assert_eq!(noc.slave_ports, pl.slave_ports);
        assert_eq!(noc.north_master, pl.north_master);
    }

    /// Drift-detection: if the build.rs-generated per-field constants
    /// ever change without `AIE2_STREAM_SWITCH_TOPOLOGY` being updated,
    /// this test fires.
    #[test]
    fn aie2_topology_matches_generated_constants() {
        use crate::aie2::stream_switch::{compute, mem_tile, shim};

        // Port arrays
        assert_eq!(
            AIE2_STREAM_SWITCH_TOPOLOGY.compute.master_ports,
            crate::aie2::COMPUTE_MASTER_PORTS,
            "compute master ports drifted"
        );
        assert_eq!(
            AIE2_STREAM_SWITCH_TOPOLOGY.compute.slave_ports,
            crate::aie2::COMPUTE_SLAVE_PORTS,
            "compute slave ports drifted"
        );
        assert_eq!(
            AIE2_STREAM_SWITCH_TOPOLOGY.memtile.master_ports,
            crate::aie2::MEMTILE_MASTER_PORTS,
            "memtile master ports drifted"
        );
        assert_eq!(
            AIE2_STREAM_SWITCH_TOPOLOGY.memtile.slave_ports,
            crate::aie2::MEMTILE_SLAVE_PORTS,
            "memtile slave ports drifted"
        );
        assert_eq!(
            AIE2_STREAM_SWITCH_TOPOLOGY.shim.master_ports,
            crate::aie2::SHIM_MASTER_PORTS,
            "shim master ports drifted"
        );
        assert_eq!(
            AIE2_STREAM_SWITCH_TOPOLOGY.shim.slave_ports,
            crate::aie2::SHIM_SLAVE_PORTS,
            "shim slave ports drifted"
        );

        // Compute ranges
        assert_eq!(
            AIE2_STREAM_SWITCH_TOPOLOGY.compute.north_master,
            (compute::NORTH_MASTER_START, compute::NORTH_MASTER_END)
        );
        assert_eq!(
            AIE2_STREAM_SWITCH_TOPOLOGY.compute.south_master,
            (compute::SOUTH_MASTER_START, compute::SOUTH_MASTER_END)
        );
        assert_eq!(
            AIE2_STREAM_SWITCH_TOPOLOGY.compute.north_slave,
            (compute::NORTH_SLAVE_START, compute::NORTH_SLAVE_END)
        );
        assert_eq!(
            AIE2_STREAM_SWITCH_TOPOLOGY.compute.south_slave,
            (compute::SOUTH_SLAVE_START, compute::SOUTH_SLAVE_END)
        );

        // MemTile ranges
        assert_eq!(
            AIE2_STREAM_SWITCH_TOPOLOGY.memtile.north_master,
            (mem_tile::NORTH_MASTER_START, mem_tile::NORTH_MASTER_END)
        );
        assert_eq!(
            AIE2_STREAM_SWITCH_TOPOLOGY.memtile.south_master,
            (mem_tile::SOUTH_MASTER_START, mem_tile::SOUTH_MASTER_END)
        );
        assert_eq!(
            AIE2_STREAM_SWITCH_TOPOLOGY.memtile.north_slave,
            (mem_tile::NORTH_SLAVE_START, mem_tile::NORTH_SLAVE_END)
        );
        assert_eq!(
            AIE2_STREAM_SWITCH_TOPOLOGY.memtile.south_slave,
            (mem_tile::SOUTH_SLAVE_START, mem_tile::SOUTH_SLAVE_END)
        );

        // Shim ranges (north only; south is (0, 0) sentinel)
        assert_eq!(
            AIE2_STREAM_SWITCH_TOPOLOGY.shim.north_master,
            (shim::NORTH_MASTER_START, shim::NORTH_MASTER_END)
        );
        assert_eq!(
            AIE2_STREAM_SWITCH_TOPOLOGY.shim.north_slave,
            (shim::NORTH_SLAVE_START, shim::NORTH_SLAVE_END)
        );
        assert_eq!(
            AIE2_STREAM_SWITCH_TOPOLOGY.shim.south_master,
            (0, 0),
            "shim south-master sentinel preserved"
        );
        assert_eq!(
            AIE2_STREAM_SWITCH_TOPOLOGY.shim.south_slave,
            (0, 0),
            "shim south-slave sentinel preserved"
        );
    }
}
