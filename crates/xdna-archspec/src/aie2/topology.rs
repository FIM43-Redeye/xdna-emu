//! AIE2 tile topology implementation.
//!
//! Covers NPU1 (Phoenix), NPU4 / NPU5 / NPU6 (Strix / Strix Halo /
//! Krackan). AIE2 and AIE2P share this layout discipline:
//!
//! - row 0 is the shim row (every column is `ShimNoc`; AIE2 has no
//!   software-accessible `ShimPl`).
//! - rows 1 through `num_mem_tile_rows` are memtile rows (`Mem`).
//! - rows above that are compute (`Compute`).
//! - memory adjacency is uniform (no row-parity dependence).
//!
//! Extents (columns, rows, memtile row count) come from
//! `ArchModel::array_topology` and are plumbed into the struct at
//! construction.

use crate::topology::{Direction, TileTopology};
use crate::types::{ArchModel, TileKind};

/// AIE2 tile topology.
///
/// Built via `Aie2Topology::from_model(&arch_model)` from an `ArchModel`
/// whose `architecture` is `Aie2` or `Aie2p`.
#[derive(Debug, Clone, Copy)]
pub struct Aie2Topology {
    pub columns: u8,
    pub rows: u8,
    pub num_mem_tile_rows: u8,
}

impl Aie2Topology {
    /// Build an `Aie2Topology` from an `ArchModel`.
    ///
    /// Caller must ensure the model's architecture is AIE2-family
    /// (`Aie2` or `Aie2p`). `ArchModel::topology()` dispatches this.
    pub fn from_model(model: &ArchModel) -> Self {
        let topo = model
            .array_topology
            .as_ref()
            .expect("Aie2Topology::from_model: ArchModel has no array_topology");
        Self {
            columns: topo.columns,
            rows: topo.rows,
            num_mem_tile_rows: topo.num_mem_tile_rows,
        }
    }

    /// First compute row index (immediately above the last memtile row).
    #[inline]
    const fn compute_row_start(&self) -> u8 {
        1 + self.num_mem_tile_rows
    }
}

impl TileTopology for Aie2Topology {
    fn classify(&self, _col: u8, row: u8) -> TileKind {
        if row == 0 {
            TileKind::ShimNoc
        } else if row < self.compute_row_start() {
            TileKind::Mem
        } else {
            TileKind::Compute
        }
    }

    fn neighbor(&self, col: u8, row: u8, dir: Direction) -> Option<(u8, u8)> {
        match dir {
            Direction::South => {
                if row == 0 { None } else { Some((col, row - 1)) }
            }
            Direction::North => {
                if row + 1 >= self.rows { None } else { Some((col, row + 1)) }
            }
            Direction::East => {
                if col + 1 >= self.columns { None } else { Some((col + 1, row)) }
            }
            Direction::West => {
                if col == 0 { None } else { Some((col - 1, row)) }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Canonical NPU1 (Phoenix) topology: 5 columns x 6 rows, 1 memtile row.
    fn npu1() -> Aie2Topology {
        Aie2Topology { columns: 5, rows: 6, num_mem_tile_rows: 1 }
    }

    #[test]
    fn classify_shim_row() {
        let topo = npu1();
        for col in 0..5 {
            assert_eq!(topo.classify(col, 0), TileKind::ShimNoc);
        }
    }

    #[test]
    fn classify_memtile_row() {
        let topo = npu1();
        for col in 0..5 {
            assert_eq!(topo.classify(col, 1), TileKind::Mem);
        }
    }

    #[test]
    fn classify_compute_rows() {
        let topo = npu1();
        for col in 0..5 {
            for row in 2..6 {
                assert_eq!(topo.classify(col, row), TileKind::Compute);
            }
        }
    }

    #[test]
    fn neighbor_south_from_shim_is_none() {
        let topo = npu1();
        for col in 0..5 {
            assert_eq!(topo.neighbor(col, 0, Direction::South), None);
        }
    }

    #[test]
    fn neighbor_south_from_compute_returns_memtile() {
        let topo = npu1();
        assert_eq!(topo.neighbor(2, 2, Direction::South), Some((2, 1)));
    }

    #[test]
    fn neighbor_north_from_top_row_is_none() {
        let topo = npu1();
        for col in 0..5 {
            assert_eq!(topo.neighbor(col, 5, Direction::North), None);
        }
    }

    #[test]
    fn neighbor_east_from_rightmost_column_is_none() {
        let topo = npu1();
        for row in 0..6 {
            assert_eq!(topo.neighbor(4, row, Direction::East), None);
        }
    }

    #[test]
    fn neighbor_west_from_leftmost_column_is_none() {
        let topo = npu1();
        for row in 0..6 {
            assert_eq!(topo.neighbor(0, row, Direction::West), None);
        }
    }

    #[test]
    fn neighbor_round_trip_north_south() {
        let topo = npu1();
        for col in 0..5 {
            for row in 0..5 {
                let north = topo.neighbor(col, row, Direction::North).unwrap();
                let back = topo.neighbor(north.0, north.1, Direction::South).unwrap();
                assert_eq!(back, (col, row));
            }
        }
    }

    #[test]
    fn neighbor_round_trip_east_west() {
        let topo = npu1();
        for col in 0..4 {
            for row in 0..6 {
                let east = topo.neighbor(col, row, Direction::East).unwrap();
                let back = topo.neighbor(east.0, east.1, Direction::West).unwrap();
                assert_eq!(back, (col, row));
            }
        }
    }

    #[test]
    fn compute_row_start_matches_memtile_count() {
        let t1 = Aie2Topology { columns: 5, rows: 6, num_mem_tile_rows: 1 };
        assert_eq!(t1.compute_row_start(), 2);

        // Hypothetical AIE2-family device with 0 memtile rows:
        let t0 = Aie2Topology { columns: 5, rows: 5, num_mem_tile_rows: 0 };
        assert_eq!(t0.compute_row_start(), 1);
        assert_eq!(t0.classify(0, 1), TileKind::Compute);
    }
}
