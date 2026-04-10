//! NPU Architecture Specification -- validated hardware model.
//!
//! Extracts hardware architecture from the open-source NPU toolchain
//! (aie-rt, AM025 JSON, device model) into a single typed Rust model.
//! Multi-architecture: each `ArchModel` represents one architecture
//! (AIE, AIE2, AIE2P) with all its tile types, registers, and
//! relationships.
//!
//! This crate is a workspace member of xdna-emu, usable as both a
//! library dependency (for runtime queries) and a build dependency
//! (for compile-time code generation from the validated spec).

pub mod device_model;
pub mod regdb;
pub mod regdb_extractor;
pub mod tablegen;
pub mod types;

use std::path::Path;

/// Build a fully-populated ArchModel for a named device.
///
/// This is the primary entry point for build.rs. It:
/// 1. Extracts device topology from the device model JSON
/// 2. Enriches with register data from the AM025 register database
/// 3. Cross-validates via Confirmed<T> (panics on conflicts)
///
/// The result is the validated architecture spec, ready for code generation.
pub fn build_arch_model(
    device_model_path: &Path,
    regdb: &regdb::RegisterDb,
    device_name: &str,
) -> Result<types::ArchModel, String> {
    let mut model = device_model::extract_device_model(device_model_path, device_name)
        .map_err(|e| format!("Device model extraction failed: {}", e))?;
    regdb_extractor::populate_tile_modules(&mut model, regdb);
    populate_manual_constants(&mut model);
    Ok(model)
}

/// Cross-validate subsystem address ranges against aie-rt extracted constants.
///
/// Called after subsystems are populated from AM025. The aie-rt values come
/// from the gcc -E pipeline in build.rs (DMA BaseAddr, Lock LockSetValBase).
///
/// Each entry is (TileKind, ModuleKind, SubsystemKind, aiert_base_offset, source).
/// The aiert_base_offset is confirmed against the subsystem's offset_start.
/// If AM025 and aie-rt disagree, `Confirmed::confirm()` panics with a
/// detailed conflict message.
pub fn confirm_subsystem_ranges(
    model: &mut types::ArchModel,
    confirmations: &[(
        types::TileKind,
        types::ModuleKind,
        types::SubsystemKind,
        u32,
        types::SourceAttribution,
    )],
) {
    for (tile_kind, mod_kind, sub_kind, aiert_base, source) in confirmations {
        let tile = model.tile_types.iter_mut().find(|t| t.kind == *tile_kind);
        let tile = match tile {
            Some(t) => t,
            None => continue, // Tile type not in model, skip
        };
        let module = tile.modules.iter_mut().find(|m| m.kind == *mod_kind);
        let module = match module {
            Some(m) => m,
            None => continue, // Module not in model, skip
        };
        let subsystem = module.subsystems.iter_mut().find(|s| s.kind == *sub_kind);
        if let Some(sub) = subsystem {
            sub.offset_start.confirm(*aiert_base, source.clone());
        }
    }
}

/// Populate architecture constants that have no machine-readable source.
///
/// These values come from AM020 prose, AM025 register descriptions, and
/// hardware observation. They are hand-written here so that ArchModel is
/// the single home for all architecture data regardless of provenance.
///
/// When a data-driven extractor is added for any of these values, it
/// should call `Confirmed::confirm()` to cross-validate against the
/// hand-written baseline.
fn populate_manual_constants(model: &mut types::ArchModel) {
    match model.arch {
        types::Architecture::Aie2 => populate_aie2_manual_constants(model),
        // Other architectures: add population functions as needed.
        _ => {}
    }
}

fn populate_aie2_manual_constants(model: &mut types::ArchModel) {
    use types::*;

    let src = SourceAttribution {
        origin: Source::Am020,
        file: "AM020 AIE-ML Architecture Manual".into(),
        detail: "hand-written constants (Ch2, Ch4, Table 2, Table 3)".into(),
    };

    // -- Physical banking --
    // Device model provides logical banking (from mlir-aie). Physical
    // banking comes from AM020 Ch2: "8 memory banks" for compute tiles,
    // 16 for memtiles, each 128 bits wide.
    let phys_src = SourceAttribution {
        origin: Source::Am020,
        file: "AM020 AIE-ML Architecture Manual".into(),
        detail: "Ch2: physical SRAM bank organization".into(),
    };
    for tile_type in &mut model.tile_types {
        if let Some(ref mut mem) = tile_type.memory {
            let (num_banks, bank_size) = match tile_type.kind {
                TileKind::Compute => (8u8, 8 * 1024u64),   // 64KB / 8 banks
                TileKind::Mem => (16u8, 32 * 1024u64),     // 512KB / 16 banks
                _ => continue,
            };
            if mem.physical.is_none() {
                mem.physical = Some(BankingModel {
                    num_banks,
                    bank_size,
                    bank_width_bits: 128,
                    source: phys_src.clone(),
                });
            }
        }
    }

    // -- Core address map --
    // From aie-rt AieMlCoreMod (xaiemlgbl_reginit.c:2318-2326) and
    // XAIEMLGBL_CORE_MODULE_PROGRAM_MEMORY (xaiemlgbl_params.h:32).
    let addr_src = SourceAttribution {
        origin: Source::AieRt,
        file: "aie-rt/driver/src/global/xaiemlgbl_reginit.c".into(),
        detail: "AieMlCoreMod: DataMemAddr, DataMemShift, IsCheckerBoard, ProgMemHostOffset".into(),
    };
    for tile_type in &mut model.tile_types {
        if tile_type.kind == TileKind::Compute {
            tile_type.core_address_map = Some(CoreAddressMap {
                data_mem_addr: 0x40000,   // AieMlCoreMod.DataMemAddr
                data_mem_shift: 16,       // AieMlCoreMod.DataMemShift
                is_checkerboard: false,   // AieMlCoreMod.IsCheckerBoard = 0
                program_mem_host_offset: 0x20000,  // XAIEMLGBL_CORE_MODULE_PROGRAM_MEMORY
                source: addr_src.clone(),
            });
        }
    }

    // -- Timing model --
    model.timing = Some(TimingModel {
        lock: LockTiming {
            acquire_latency: 1,
            release_latency: 1,
            retry_interval: 1,
        },
        dma: DmaTiming {
            bd_setup_cycles: 4,
            channel_start_cycles: 2,
            // AIE2 data memory bus is 128 bits = 16 bytes = 4 words per cycle.
            // Source: xaiemlgbl_params.h DATAMEMORY_WIDTH = 128.
            // (AIE2P is 256 bits = 8 words/cycle per xaie2psgbl_params.h.)
            words_per_cycle: 4,
            memory_latency_cycles: 5,
            lock_acquire_cycles: 1,
            lock_release_cycles: 1,
            bd_chain_cycles: 2,
            host_memory_latency_cycles: 100,
        },
        stream_switch: StreamSwitchTiming {
            local_slave_fifo_depth: 4,
            local_master_fifo_depth: 2,
            local_to_local_latency: 3,
            local_to_external_latency: 4,
            external_to_external_latency: 4,
            external_to_local_latency: 3,
            packet_arbitration_overhead: 1,
        },
        instruction: InstructionTiming {
            data_memory_latency: 5,
            branch_penalty: 3,
        },
        source: src.clone(),
    });

    // -- Packet and protocol model --
    model.packet = Some(PacketModel {
        stream: StreamPacketFormat {
            stream_id_mask: 0x1F,
            packet_type_shift: 12,
            packet_type_mask: 0x7,
            src_row_shift: 16,
            src_row_mask: 0x1F,
            src_col_shift: 21,
            src_col_mask: 0x7F,
            parity_shift: 31,
        },
        control: ControlPacketFormat {
            address_mask: 0x000F_FFFF,
            length_shift: 20,
            length_mask: 0x3,
            operation_shift: 22,
            operation_mask: 0x3,
            response_id_shift: 24,
            response_id_mask: 0x7F,
            parity_bit: 31,
            op_write: 0,
            op_read: 1,
            op_write_incr: 2,
            op_block_write: 3,
        },
        fot: FotConfig {
            disabled: 0,
            no_counts: 1,
            counts_with_tokens: 2,
            counts_from_register: 3,
        },
        source: src.clone(),
    });

    // -- Processor model --
    // VLIW slot widths from llvm-aie AIE2Slots.td, register sizes from
    // AIE2GenRegisterInfo.td, pipeline constants from AM020 Ch4.
    let proc_src = SourceAttribution {
        origin: Source::Am020,
        file: "llvm-aie AIE2Slots.td + AIE2GenRegisterInfo.td + AM020 Ch4".into(),
        detail: "VLIW slot widths, register sizes, pipeline constants".into(),
    };
    model.processor = Some(ProcessorModel {
        slot_widths: vec![
            ("lda".into(), 21),
            ("ldb".into(), 16),
            ("alu".into(), 20),
            ("mv".into(), 22),
            ("st".into(), 21),
            ("vec".into(), 26),
            ("lng".into(), 42),
        ],
        vector_register_bits: 512,
        vector_pair_bits: 1024,
        accumulator_bits: 512,
        branch_delay_slots: 5,
        partial_store_data_latency: 6,
        srs_shift_bias: 4,
        source: proc_src,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::*;

    fn test_source(origin: Source, detail: &str) -> SourceAttribution {
        SourceAttribution {
            origin,
            file: "test".into(),
            detail: detail.into(),
        }
    }

    /// Build a minimal ArchModel with one tile type, one module, and one
    /// subsystem to test confirm_subsystem_ranges() in isolation.
    fn minimal_model_with_subsystem(
        tile_kind: TileKind,
        mod_kind: ModuleKind,
        sub_kind: SubsystemKind,
        offset_start: u32,
    ) -> ArchModel {
        let src = test_source(Source::Am025Json, "test register grouping");
        let subsystem = SubsystemModel {
            kind: sub_kind,
            offset_start: Confirmed::new(offset_start, src.clone()),
            offset_end: Confirmed::new(offset_start + 0x1000, src.clone()),
            registers: vec![],
        };
        let module = ModuleModel {
            kind: mod_kind,
            subsystems: vec![subsystem],
            source: src.clone(),
        };
        let tile = TileTypeModel {
            kind: tile_kind,
            name: format!("{:?}", tile_kind),
            representative: None,
            instances: InstanceCount {
                locks: Confirmed::new(16, src.clone()),
                bds: Confirmed::new(16, src.clone()),
                channels: Confirmed::new(2, src.clone()),
            },
            memory: None,
            dma_capabilities: None,
            core_address_map: None,
            switchbox_ports: vec![],
            shim_mux_ports: vec![],
            modules: vec![module],
            bd_schema: None,
            channel_schema: None,
            source: test_source(Source::Am025Json, "test tile"),
        };
        ArchModel {
            arch: Architecture::Aie2,
            generation: Some(DeviceGeneration::Aie2Ipu),
            device_id: None,
            is_npu: true,
            tile_types: vec![tile],
            relationships: vec![],
            device_constants: None,
            array_topology: None,
            timing: None,
            packet: None,
            processor: None,
        }
    }

    #[test]
    fn confirm_subsystem_ranges_matching_value_adds_source() {
        let mut model = minimal_model_with_subsystem(
            TileKind::Compute,
            ModuleKind::Memory,
            SubsystemKind::Dma,
            0x1D000,
        );

        // Before confirmation: 1 source (AM025)
        let dma = &model.tile_types[0].modules[0].subsystems[0];
        assert_eq!(dma.offset_start.sources().len(), 1);

        // Confirm with matching aie-rt value
        let confirmations = vec![(
            TileKind::Compute,
            ModuleKind::Memory,
            SubsystemKind::Dma,
            0x1D000u32,
            test_source(Source::AieRt, "AieMlTileDmaMod.BaseAddr"),
        )];
        confirm_subsystem_ranges(&mut model, &confirmations);

        // After confirmation: 2 sources
        let dma = &model.tile_types[0].modules[0].subsystems[0];
        assert_eq!(dma.offset_start.sources().len(), 2);
        assert_eq!(dma.offset_start.sources()[0].origin, Source::Am025Json);
        assert_eq!(dma.offset_start.sources()[1].origin, Source::AieRt);
    }

    #[test]
    #[should_panic(expected = "GRAPH CONFLICT")]
    fn confirm_subsystem_ranges_mismatched_value_panics() {
        let mut model = minimal_model_with_subsystem(
            TileKind::Compute,
            ModuleKind::Memory,
            SubsystemKind::Dma,
            0x1D000,
        );

        // Confirm with WRONG aie-rt value -- should panic
        let confirmations = vec![(
            TileKind::Compute,
            ModuleKind::Memory,
            SubsystemKind::Dma,
            0xDEAD,
            test_source(Source::AieRt, "wrong value"),
        )];
        confirm_subsystem_ranges(&mut model, &confirmations);
    }

    #[test]
    fn confirm_subsystem_ranges_missing_tile_skips_gracefully() {
        let mut model = minimal_model_with_subsystem(
            TileKind::Compute,
            ModuleKind::Memory,
            SubsystemKind::Dma,
            0x1D000,
        );

        // Try to confirm a tile type that doesn't exist -- should skip
        let confirmations = vec![(
            TileKind::Mem,
            ModuleKind::MemTile,
            SubsystemKind::Dma,
            0xA0000,
            test_source(Source::AieRt, "nonexistent tile"),
        )];
        confirm_subsystem_ranges(&mut model, &confirmations);

        // Original subsystem unchanged (still 1 source)
        let dma = &model.tile_types[0].modules[0].subsystems[0];
        assert_eq!(dma.offset_start.sources().len(), 1);
    }

    #[test]
    fn confirm_subsystem_ranges_missing_subsystem_skips_gracefully() {
        let mut model = minimal_model_with_subsystem(
            TileKind::Compute,
            ModuleKind::Memory,
            SubsystemKind::Dma,
            0x1D000,
        );

        // Try to confirm a subsystem that doesn't exist in this module
        let confirmations = vec![(
            TileKind::Compute,
            ModuleKind::Memory,
            SubsystemKind::Lock,
            0x1F000,
            test_source(Source::AieRt, "lock in module without lock subsystem"),
        )];
        confirm_subsystem_ranges(&mut model, &confirmations);

        // Original DMA subsystem unchanged
        let dma = &model.tile_types[0].modules[0].subsystems[0];
        assert_eq!(dma.offset_start.sources().len(), 1);
    }

    #[test]
    fn confirm_subsystem_ranges_multiple_confirmations() {
        // Build a model with both DMA and Lock subsystems
        let src = test_source(Source::Am025Json, "register grouping");
        let dma_sub = SubsystemModel {
            kind: SubsystemKind::Dma,
            offset_start: Confirmed::new(0x1D000, src.clone()),
            offset_end: Confirmed::new(0x1E000, src.clone()),
            registers: vec![],
        };
        let lock_sub = SubsystemModel {
            kind: SubsystemKind::Lock,
            offset_start: Confirmed::new(0x1F000, src.clone()),
            offset_end: Confirmed::new(0x1F100, src.clone()),
            registers: vec![],
        };
        let module = ModuleModel {
            kind: ModuleKind::Memory,
            subsystems: vec![dma_sub, lock_sub],
            source: src.clone(),
        };
        let tile = TileTypeModel {
            kind: TileKind::Compute,
            name: "compute".into(),
            representative: None,
            instances: InstanceCount {
                locks: Confirmed::new(16, src.clone()),
                bds: Confirmed::new(16, src.clone()),
                channels: Confirmed::new(2, src.clone()),
            },
            memory: None,
            dma_capabilities: None,
            core_address_map: None,
            switchbox_ports: vec![],
            shim_mux_ports: vec![],
            modules: vec![module],
            bd_schema: None,
            channel_schema: None,
            source: test_source(Source::Am025Json, "test tile"),
        };
        let mut model = ArchModel {
            arch: Architecture::Aie2,
            generation: Some(DeviceGeneration::Aie2Ipu),
            device_id: None,
            is_npu: true,
            tile_types: vec![tile],
            relationships: vec![],
            device_constants: None,
            array_topology: None,
            timing: None,
            packet: None,
            processor: None,
        };

        // Confirm both DMA and Lock in one call
        let confirmations = vec![
            (
                TileKind::Compute,
                ModuleKind::Memory,
                SubsystemKind::Dma,
                0x1D000,
                test_source(Source::AieRt, "AieMlTileDmaMod.BaseAddr"),
            ),
            (
                TileKind::Compute,
                ModuleKind::Memory,
                SubsystemKind::Lock,
                0x1F000,
                test_source(Source::AieRt, "AieMlTileLockMod.LockSetValBase"),
            ),
        ];
        confirm_subsystem_ranges(&mut model, &confirmations);

        // Both should now have 2 sources
        let subs = &model.tile_types[0].modules[0].subsystems;
        assert_eq!(subs[0].kind, SubsystemKind::Dma);
        assert_eq!(subs[0].offset_start.sources().len(), 2);
        assert_eq!(subs[1].kind, SubsystemKind::Lock);
        assert_eq!(subs[1].offset_start.sources().len(), 2);
    }
}
