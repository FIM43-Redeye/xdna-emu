//! Validation tests comparing aie-rt extracted constants against regdb values.
//!
//! These tests ensure the aie-rt preprocessing pipeline in build.rs produces
//! constants that match what the AM025 JSON register database provides. This
//! cross-validation catches drift between sources.

#[cfg(test)]
mod aiert_dma {
    // Re-export all DMA submodules from archspec so test code uses the same
    // names (compute_dma::BD_BASE, memtile_dma::BD_BASE, etc.) as before.
    pub use xdna_archspec::aie2::aiert::dma::*;

    #[test]
    fn memtile_dma_bd_base_matches_regdb() {
        let layout = crate::device::regdb::device_reg_layout();
        assert_eq!(memtile_dma::BD_BASE, layout.memtile_bd_base, "MemTile BD base: aie-rt vs regdb mismatch");
    }

    #[test]
    fn memtile_dma_bd_stride_matches_regdb() {
        let layout = crate::device::regdb::device_reg_layout();
        assert_eq!(
            memtile_dma::BD_STRIDE,
            layout.memtile_bd_stride,
            "MemTile BD stride: aie-rt vs regdb mismatch"
        );
    }

    #[test]
    fn compute_dma_bd_base_matches_regdb() {
        let layout = crate::device::regdb::device_reg_layout();
        assert_eq!(compute_dma::BD_BASE, layout.memory_bd_base, "Compute BD base: aie-rt vs regdb mismatch");
    }

    #[test]
    fn compute_dma_bd_stride_matches_regdb() {
        let layout = crate::device::regdb::device_reg_layout();
        assert_eq!(
            compute_dma::BD_STRIDE,
            layout.memory_bd_stride,
            "Compute BD stride: aie-rt vs regdb mismatch"
        );
    }

    #[test]
    fn shim_dma_bd_base_matches_regdb() {
        let layout = crate::device::regdb::device_reg_layout();
        assert_eq!(shim_dma::BD_BASE, layout.shim_bd_base, "Shim BD base: aie-rt vs regdb mismatch");
    }

    #[test]
    fn shim_dma_bd_stride_matches_regdb() {
        let layout = crate::device::regdb::device_reg_layout();
        assert_eq!(shim_dma::BD_STRIDE, layout.shim_bd_stride, "Shim BD stride: aie-rt vs regdb mismatch");
    }

    #[test]
    fn compute_dma_channel_base_matches_regdb() {
        let layout = crate::device::regdb::device_reg_layout();
        assert_eq!(
            compute_dma::CH_CTRL_BASE,
            layout.memory_channel_base,
            "Compute channel base: aie-rt vs regdb mismatch"
        );
    }

    #[test]
    fn shim_dma_channel_base_matches_regdb() {
        let layout = crate::device::regdb::device_reg_layout();
        assert_eq!(
            shim_dma::CH_CTRL_BASE,
            layout.shim_channel_base,
            "Shim channel base: aie-rt vs regdb mismatch"
        );
    }

    #[test]
    fn compute_dma_status_base_matches_regdb() {
        let layout = crate::device::regdb::device_reg_layout();
        assert_eq!(
            compute_dma::CH_STATUS_BASE,
            layout.memory_status_base,
            "Compute status base: aie-rt vs regdb mismatch"
        );
    }
}

#[cfg(test)]
mod aiert_locks {
    // Re-export all lock submodules from archspec so test code uses the same
    // names (compute_locks::SET_VAL_BASE, etc.) as before.
    pub use xdna_archspec::aie2::aiert::locks::*;

    #[test]
    fn compute_lock_set_val_base_matches_regdb() {
        let layout = crate::device::regdb::device_reg_layout();
        assert_eq!(
            compute_locks::SET_VAL_BASE,
            layout.memory_lock_base,
            "Compute lock set-val base: aie-rt vs regdb mismatch"
        );
    }

    #[test]
    fn compute_lock_set_val_stride_matches_regdb() {
        let layout = crate::device::regdb::device_reg_layout();
        assert_eq!(
            compute_locks::SET_VAL_STRIDE,
            layout.memory_lock_stride,
            "Compute lock set-val stride: aie-rt vs regdb mismatch"
        );
    }

    #[test]
    fn memtile_lock_set_val_base_matches_regdb() {
        let layout = crate::device::regdb::device_reg_layout();
        assert_eq!(
            memtile_locks::SET_VAL_BASE,
            layout.memtile_lock_base,
            "MemTile lock set-val base: aie-rt vs regdb mismatch"
        );
    }

    #[test]
    fn memtile_lock_set_val_stride_matches_regdb() {
        let layout = crate::device::regdb::device_reg_layout();
        assert_eq!(
            memtile_locks::SET_VAL_STRIDE,
            layout.memtile_lock_stride,
            "MemTile lock set-val stride: aie-rt vs regdb mismatch"
        );
    }

    #[test]
    fn compute_lock_count() {
        assert_eq!(compute_locks::NUM_LOCKS, 16);
    }

    #[test]
    fn memtile_lock_count() {
        assert_eq!(memtile_locks::NUM_LOCKS, 64);
    }

    #[test]
    fn shim_lock_count() {
        assert_eq!(shim_locks::NUM_LOCKS, 16);
    }
}

#[cfg(test)]
mod aiert_ports {
    // Re-export all port constants and the AieRtPortType enum from archspec
    // so test code uses the same names as before.
    pub use xdna_archspec::aie2::aiert::ports::*;

    #[test]
    fn compute_master_port_count() {
        // AIE2 compute tile has 23 master ports per aie-rt
        assert_eq!(COMPUTE_MASTER_PORTS.len(), 23);
    }

    #[test]
    fn compute_slave_port_count() {
        // AIE2 compute tile has 25 slave ports per aie-rt
        assert_eq!(COMPUTE_SLAVE_PORTS.len(), 25);
    }

    #[test]
    fn compute_master_first_is_core() {
        assert_eq!(COMPUTE_MASTER_PORTS[0], (AieRtPortType::Core, 0));
    }

    #[test]
    fn compute_master_dma_ports() {
        assert_eq!(COMPUTE_MASTER_PORTS[1], (AieRtPortType::Dma, 0));
        assert_eq!(COMPUTE_MASTER_PORTS[2], (AieRtPortType::Dma, 1));
    }

    #[test]
    fn memtile_master_port_count() {
        // AIE2 memtile has 17 master ports per aie-rt
        assert_eq!(MEMTILE_MASTER_PORTS.len(), 17);
    }

    #[test]
    fn shim_master_port_count() {
        // AIE2 shim has 22 master ports per aie-rt
        assert_eq!(SHIM_MASTER_PORTS.len(), 22);
    }

    #[test]
    fn memtile_slave_port_count() {
        // AIE2 memtile has 18 slave ports per aie-rt
        assert_eq!(MEMTILE_SLAVE_PORTS.len(), 18);
    }

    #[test]
    fn shim_slave_port_count() {
        // AIE2 shim has 23 slave ports per aie-rt
        assert_eq!(SHIM_SLAVE_PORTS.len(), 23);
    }

    #[test]
    fn port_type_enum_values() {
        assert_eq!(AieRtPortType::Core as u8, 0);
        assert_eq!(AieRtPortType::Dma as u8, 1);
        assert_eq!(AieRtPortType::Ctrl as u8, 2);
        assert_eq!(AieRtPortType::Fifo as u8, 3);
        assert_eq!(AieRtPortType::South as u8, 4);
        assert_eq!(AieRtPortType::West as u8, 5);
        assert_eq!(AieRtPortType::North as u8, 6);
        assert_eq!(AieRtPortType::East as u8, 7);
        assert_eq!(AieRtPortType::Trace as u8, 8);
    }
}
