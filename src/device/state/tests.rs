use super::*;
use crate::parser::xclbin::SectionKind;
use crate::parser::{Xclbin, AiePartition};
use crate::parser::cdo::find_cdo_offset;

#[test]
fn test_device_state_creation() {
    let state = DeviceState::new_npu1();
    assert_eq!(state.array.cols(), 5);
    assert_eq!(state.array.rows(), 6);
}

#[test]
fn test_apply_real_cdo() {
    use crate::config::Config;

    let Some(test_xclbin) = Config::get().add_one_xclbin() else {
        eprintln!("Skipping real CDO test: file not found (set MLIR_AIE_PATH)");
        return;
    };

    // Load xclbin
    let xclbin = Xclbin::from_file(&test_xclbin).unwrap();
    let section = xclbin.find_section(SectionKind::AiePartition).unwrap();
    let partition = AiePartition::parse(section.data()).unwrap();
    let pdi = partition.primary_pdi().unwrap();
    let cdo_offset = find_cdo_offset(pdi.pdi_image).unwrap();
    let cdo = Cdo::parse(&pdi.pdi_image[cdo_offset..]).unwrap();

    // Apply to device
    let mut state = DeviceState::new_npu1();
    state.apply_cdo(&cdo).unwrap();

    // Verify something was configured
    assert!(state.stats.commands > 0, "Expected commands to be processed");
    assert!(state.stats.dma_writes > 0, "Expected DMA writes (shim control packets)");
    assert!(state.stats.writes > 0 || state.stats.mask_writes > 0,
        "Expected register writes");

    // Note: For this xclbin, DMA_WRITE goes to shim tile for control packets,
    // not to compute tiles for code/data. That's expected - core code is loaded
    // via XRT at runtime, not embedded in CDO.
}

#[test]
fn test_lock_write() {
    let mut state = DeviceState::new_npu1();

    // Write to lock 5 in tile(1,2)
    // AIE-ML lock registers are 16 bytes apart (0x10 spacing per lock)
    let addr = TileAddress::encode(1, 2, 0x1F050); // Lock 5 = 0x1F000 + 5*0x10

    // Lock_value field is 6-bit signed (bits [5:0], per AM025 regdb).
    // Value 5 fits in unsigned 6-bit range -> positive.
    state.write_register(addr, 5).unwrap();
    assert_eq!(state.array.tile(1, 2).locks[5].value, 5);

    // Value 0x3F = 63 = all 6 bits set -> sign-extends to -1.
    state.write_register(addr, 0x3F).unwrap();
    assert_eq!(state.array.tile(1, 2).locks[5].value, -1);

    // Value 0x20 = bit 5 set -> sign-extends to -32.
    state.write_register(addr, 0x20).unwrap();
    assert_eq!(state.array.tile(1, 2).locks[5].value, -32);

    // Value 31 = max positive in 6-bit signed.
    state.write_register(addr, 31).unwrap();
    assert_eq!(state.array.tile(1, 2).locks[5].value, 31);
}

#[test]
fn test_shim_lock_write() {
    let mut state = DeviceState::new_npu1();

    // Shim tile lock registers start at 0x14000, stride 0x10 (per AM025).
    // Row 0 is the shim row.
    let col: u8 = 0;
    let row: u8 = 0;

    // Write to lock 3 in shim tile(0,0)
    let addr = TileAddress::encode(col, row, 0x14030); // Lock 3 = 0x14000 + 3*0x10
    state.write_register(addr, 7).unwrap();
    assert_eq!(state.array.tile(col, row).locks[3].value, 7);

    // Negative value: 0x3F = -1 in 6-bit signed.
    state.write_register(addr, 0x3F).unwrap();
    assert_eq!(state.array.tile(col, row).locks[3].value, -1);

    // Write to lock 0 (base address)
    let addr0 = TileAddress::encode(col, row, 0x14000);
    state.write_register(addr0, 10).unwrap();
    assert_eq!(state.array.tile(col, row).locks[0].value, 10);

    // Write to lock 15 (last lock)
    let addr15 = TileAddress::encode(col, row, 0x140F0); // Lock 15 = 0x14000 + 15*0x10
    state.write_register(addr15, 1).unwrap();
    assert_eq!(state.array.tile(col, row).locks[15].value, 1);

    // Verify lock 3 was not disturbed by other writes.
    assert_eq!(state.array.tile(col, row).locks[3].value, -1);
}

#[test]
fn test_shim_lock_mask_write() {
    let mut state = DeviceState::new_npu1();

    let col: u8 = 1;
    let row: u8 = 0; // Shim row

    // Set lock 5 to value 10 via direct write first.
    let addr = TileAddress::encode(col, row, 0x14050); // Lock 5
    state.write_register(addr, 10).unwrap();
    assert_eq!(state.array.tile(col, row).locks[5].value, 10);

    // Mask write: clear bit 1 (mask=0x02, value=0x00) -> 10 & !2 = 8
    state.mask_write_register(addr, 0x02, 0x00).unwrap();
    assert_eq!(state.array.tile(col, row).locks[5].value, 8);

    // Mask write: set bit 0 (mask=0x01, value=0x01) -> 8 | 1 = 9
    state.mask_write_register(addr, 0x01, 0x01).unwrap();
    assert_eq!(state.array.tile(col, row).locks[5].value, 9);
}

#[test]
fn test_dma_channel_write() {
    let mut state = DeviceState::new_npu1();

    // Write to DMA_MM2S_0_CTRL in tile(1,2)
    let addr = TileAddress::encode(1, 2, 0x1DE10);
    state.write_register(addr, 0x01).unwrap(); // Enable

    let tile = state.array.tile(1, 2);
    assert!(tile.dma_channels[2].is_enabled()); // Channel 2 is MM2S_0
}

#[test]
fn test_core_control_mask_write() {
    let mut state = DeviceState::new_npu1();

    // Mask write to enable core
    let addr = TileAddress::encode(1, 2, 0x32000);
    state.mask_write_register(addr, 0x1, 0x1).unwrap();

    let tile = state.array.tile(1, 2);
    assert!(tile.core.enabled);
}

#[test]
fn test_cdo_writes_tile_init_data() {
    // Verify that CDO DmaWrite correctly loads in2_mem_buff_0 into
    // tile(0,2) data memory at offset 0x400.
    // This is a regression test for the vec_vec_add_tile_init failure.
    let xclbin_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../mlir-aie/build/test/npu-xrt/vec_vec_add_tile_init/aie.xclbin");
    if !xclbin_path.exists() {
        eprintln!("Skipping tile init test: {:?} not found", xclbin_path);
        return;
    }

    let xclbin = Xclbin::from_file(&xclbin_path).unwrap();
    let section = xclbin.find_section(SectionKind::AiePartition).unwrap();
    let partition = AiePartition::parse(section.data()).unwrap();
    let pdi = partition.primary_pdi().unwrap();
    let cdo_offset = find_cdo_offset(pdi.pdi_image).unwrap();
    let cdo = Cdo::parse(&pdi.pdi_image[cdo_offset..]).unwrap();

    let mut state = DeviceState::new_npu1();
    state.apply_cdo(&cdo).unwrap();

    // Check tile(0,2) data memory at offset 0x400
    // Expected: in2_mem_buff_0 = [0, 1, 2, ..., 255] as i32
    let tile = state.array.tile(0, 2);
    let dm = tile.data_memory();

    // Read all 256 words from offset 0x400 and verify CDO init is correct.
    let mut values = Vec::new();
    for i in 0..256 {
        let off = 0x400 + i * 4;
        let word = u32::from_le_bytes([dm[off], dm[off+1], dm[off+2], dm[off+3]]);
        values.push(word);
    }

    for i in 0..256 {
        assert_eq!(values[i], i as u32,
            "in2_mem_buff_0[{}] should be {} but was {}", i, i, values[i]);
    }

    // Also verify data_bytes was counted
    assert!(state.stats.data_bytes >= 1024,
        "Expected at least 1024 data bytes written, got {}", state.stats.data_bytes);
}

/// Test that single-word BD writes (as from control packets) defer parsing
/// until the DMA channel start queue is written.
#[test]
fn test_lazy_bd_parsing_single_word_writes() {
    use crate::device::registers::TileAddress;
    use crate::device::regdb::device_reg_layout;

    let mut state = DeviceState::new_npu1();
    let col: u8 = 1;
    let row: u8 = 2; // Compute tile

    let reg_layout = device_reg_layout();
    let bd_base = reg_layout.memory_bd_base;   // 0x1D000
    let bd_stride = reg_layout.memory_bd_stride; // 0x20
    let bd_idx: usize = 3;

    // Build known BD words for a compute tile (6 words).
    // Use the real parser to construct expected values, then write
    // those words one at a time.
    let base_addr_words: u32 = 0x100; // 256 words = 1024 bytes
    let length_words: u32 = 64;       // 64 words = 256 bytes

    // Word 0: Base_Address | Buffer_Length (from regdb field layout)
    let lay = &reg_layout.memory_bd;
    let w0 = lay.base_address.insert(0, base_addr_words)
               | lay.buffer_length.insert(0, length_words);
    // Word 1-4: leave as zero (no packet, no strides, no iteration)
    let w1: u32 = 0;
    let w2: u32 = 0;
    let w3: u32 = 0;
    let w4: u32 = 0;
    // Word 5: Valid_BD = 1 (and no locks, no chaining)
    let w5 = lay.valid_bd.insert(0, 1);

    let words = [w0, w1, w2, w3, w4, w5];

    // Write each word individually (simulating control packet path)
    for (i, &word) in words.iter().enumerate() {
        let offset = bd_base + (bd_idx as u32) * bd_stride + (i as u32) * 4;
        let addr = TileAddress::encode(col, row, offset);
        state.write_register(addr, word).unwrap();
    }

    // Verify the BD is marked dirty in the DMA engine
    let dma = state.array.dma_engine(col, row).unwrap();
    assert!(dma.is_bd_dirty(bd_idx as u8),
        "BD should be dirty after single-word writes");

    // Verify the BD config has NOT been updated yet (should be default)
    let bd_before = dma.get_bd(bd_idx as u8).unwrap();
    assert_eq!(bd_before.base_addr, 0,
        "BD config should not be updated until channel start");
    assert_eq!(bd_before.length, 0,
        "BD length should be 0 before channel start");

    // Now write the channel start queue register to trigger re-parse.
    // MM2S channel 0 = channel index 2 (after S2MM_0, S2MM_1).
    // Start queue offset = channel_base + ch_idx * stride + 4
    let ch_idx: usize = 2; // MM2S_0
    let start_queue_offset = reg_layout.memory_channel_base
        + (ch_idx as u32) * reg_layout.memory_channel_stride + 4;

    // Start_BD_ID field value = bd_idx, repeat_count = 0
    let queue_val = reg_layout.memory_channel.start_bd_id.insert(0, bd_idx as u32);

    let addr = TileAddress::encode(col, row, start_queue_offset);
    state.write_register(addr, queue_val).unwrap();

    // Now the BD should have been re-parsed and configured
    let dma = state.array.dma_engine(col, row).unwrap();
    assert!(!dma.is_bd_dirty(bd_idx as u8),
        "BD should no longer be dirty after channel start");

    let bd_after = dma.get_bd(bd_idx as u8).unwrap();
    assert_eq!(bd_after.base_addr, (base_addr_words * 4) as u64,
        "BD base_addr should be {} bytes", base_addr_words * 4);
    assert_eq!(bd_after.length, length_words * 4,
        "BD length should be {} bytes", length_words * 4);
    assert!(bd_after.valid,
        "BD should be valid after re-parse");
}

// ---------------------------------------------------------------------------
// Regression tests for the MemTile DMA Start_BD_ID field width.
//
// The compute/shim Start_Queue Start_BD_ID is 4-bit (16 BDs); the MemTile
// Start_Queue Start_BD_ID is 6-bit (48 BDs).  Using the compute layout to
// extract a MemTile BD silently truncates bits [5:4], turning BD 24 -> 8
// and BD 26 -> 10.  The DMA engine's BD-channel-validity check then
// rejects the start (BD 8 is even, channel 1 is odd -> "invalid"), and
// the channel deadlocks.  These tests pin the fix at both DMA channel
// entry points that may run for MemTile.
// ---------------------------------------------------------------------------

#[test]
fn start_dma_channel_memtile_preserves_bd_high_bits() {
    use crate::device::regdb::device_reg_layout;
    use xdna_archspec::types::DmaDirection;

    let mut state = DeviceState::new_npu1();
    let col: u8 = 0;
    let row: u8 = 1; // MemTile row in NPU1.

    // Build a Start_Queue value that pushes BD 26 (>= 16, exercises the
    // 6-bit field).  Insert via the MemTile layout so the bits land at
    // [5:0]; using a literal would make the test depend on a specific
    // bit position.
    let lay = &device_reg_layout().memtile_channel;
    let bd_id_raw = lay.start_bd_id.insert(0, 26);

    state.start_dma_channel(col, row, /*channel*/ 0, DmaDirection::S2mm, bd_id_raw);

    let tile = state.array.tile(col, row);
    assert_eq!(
        tile.dma_channels[0].current_bd, 26,
        "MemTile BD>=16 must be preserved by the 6-bit memtile_channel layout; \
         got {} (4-bit truncation would yield 10)",
        tile.dma_channels[0].current_bd,
    );
}

#[test]
fn start_dma_channel_compute_uses_4bit_field() {
    // Compute Start_BD_ID is 4-bit; values >= 16 are intentionally
    // truncated by hardware (compute tiles only have 16 BDs).  This
    // pins the negative side of the helper -- compute tiles must keep
    // the legacy 4-bit semantics.
    use crate::device::regdb::device_reg_layout;
    use xdna_archspec::types::DmaDirection;

    let mut state = DeviceState::new_npu1();
    let col: u8 = 0;
    let row: u8 = 2; // Compute row.

    // Stuff BD 7 (a normal compute BD) into the compute layout.
    let lay = &device_reg_layout().memory_channel;
    let bd_id_raw = lay.start_bd_id.insert(0, 7);

    state.start_dma_channel(col, row, /*channel*/ 0, DmaDirection::S2mm, bd_id_raw);

    let tile = state.array.tile(col, row);
    assert_eq!(tile.dma_channels[0].current_bd, 7);
}

#[test]
fn channel_field_layout_helper_picks_memtile_for_memtile_kind() {
    // Direct test of the helper that both `start_dma_channel` and
    // `mask_write_dma_channel` (and `write_dma_channel`) now consult
    // before extracting Start_BD_ID / Repeat_Count / Enable_Token_Issue.
    //
    // Note: dispatch routes MemTile DMA mask-writes to
    // `mask_write_memtile_dma_channel` (a separate handler), so the
    // legacy `mask_write_dma_channel` site is reached only on Compute
    // today.  The helper still routes by tile_kind there as
    // defense-in-depth: if a future FFI / control-packet path lands a
    // MaskWrite to a MemTile Start_Queue, the BD field will be extracted
    // with the right width even though the offset arithmetic in that
    // function is still compute-tile-shaped (and the bounds check will
    // refuse the address before the truncation matters).
    use crate::device::regdb::device_reg_layout;
    use crate::device::state::compute::channel_field_layout;
    use xdna_archspec::types::TileKind;

    let reg_layout = device_reg_layout();

    let lay_mem = channel_field_layout(reg_layout, TileKind::Mem);
    assert_eq!(
        lay_mem.start_bd_id.width, 6,
        "MemTile Start_BD_ID is 6-bit per aie-rt xaiemlgbl_params.h"
    );

    let lay_compute = channel_field_layout(reg_layout, TileKind::Compute);
    assert_eq!(
        lay_compute.start_bd_id.width, 4,
        "Compute Start_BD_ID is 4-bit per aie-rt xaiemlgbl_params.h"
    );

    let lay_shim = channel_field_layout(reg_layout, TileKind::ShimNoc);
    assert_eq!(
        lay_shim.start_bd_id.width, 4,
        "Shim Start_BD_ID is 4-bit (same as compute)"
    );
}
