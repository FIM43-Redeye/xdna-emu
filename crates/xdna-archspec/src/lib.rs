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
            words_per_cycle: 1,
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
        source: src,
    });
}
