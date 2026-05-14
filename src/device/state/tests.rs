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
    assert!(state.stats.writes > 0 || state.stats.mask_writes > 0, "Expected register writes");

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
        let word = u32::from_le_bytes([dm[off], dm[off + 1], dm[off + 2], dm[off + 3]]);
        values.push(word);
    }

    for i in 0..256 {
        assert_eq!(values[i], i as u32, "in2_mem_buff_0[{}] should be {} but was {}", i, i, values[i]);
    }

    // Also verify data_bytes was counted
    assert!(
        state.stats.data_bytes >= 1024,
        "Expected at least 1024 data bytes written, got {}",
        state.stats.data_bytes
    );
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
    let bd_base = reg_layout.memory_bd_base; // 0x1D000
    let bd_stride = reg_layout.memory_bd_stride; // 0x20
    let bd_idx: usize = 3;

    // Build known BD words for a compute tile (6 words).
    // Use the real parser to construct expected values, then write
    // those words one at a time.
    let base_addr_words: u32 = 0x100; // 256 words = 1024 bytes
    let length_words: u32 = 64; // 64 words = 256 bytes

    // Word 0: Base_Address | Buffer_Length (from regdb field layout)
    let lay = &reg_layout.memory_bd;
    let w0 = lay.base_address.insert(0, base_addr_words) | lay.buffer_length.insert(0, length_words);
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
    assert!(dma.is_bd_dirty(bd_idx as u8), "BD should be dirty after single-word writes");

    // Verify the BD config has NOT been updated yet (should be default)
    let bd_before = dma.get_bd(bd_idx as u8).unwrap();
    assert_eq!(bd_before.base_addr, 0, "BD config should not be updated until channel start");
    assert_eq!(bd_before.length, 0, "BD length should be 0 before channel start");

    // Now write the channel start queue register to trigger re-parse.
    // MM2S channel 0 = channel index 2 (after S2MM_0, S2MM_1).
    // Start queue offset = channel_base + ch_idx * stride + 4
    let ch_idx: usize = 2; // MM2S_0
    let start_queue_offset =
        reg_layout.memory_channel_base + (ch_idx as u32) * reg_layout.memory_channel_stride + 4;

    // Start_BD_ID field value = bd_idx, repeat_count = 0
    let queue_val = reg_layout.memory_channel.start_bd_id.insert(0, bd_idx as u32);

    let addr = TileAddress::encode(col, row, start_queue_offset);
    state.write_register(addr, queue_val).unwrap();

    // Now the BD should have been re-parsed and configured
    let dma = state.array.dma_engine(col, row).unwrap();
    assert!(!dma.is_bd_dirty(bd_idx as u8), "BD should no longer be dirty after channel start");

    let bd_after = dma.get_bd(bd_idx as u8).unwrap();
    assert_eq!(
        bd_after.base_addr,
        (base_addr_words * 4) as u64,
        "BD base_addr should be {} bytes",
        base_addr_words * 4
    );
    assert_eq!(bd_after.length, length_words * 4, "BD length should be {} bytes", length_words * 4);
    assert!(bd_after.valid, "BD should be valid after re-parse");
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
fn dma_start_queue_memtile_preserves_bd_high_bits() {
    use crate::device::regdb::device_reg_layout;
    use xdna_archspec::aie2::registers::dma::MEMTILE_DMA_S2MM_0_START_QUEUE;

    let mut state = DeviceState::new_npu1();
    let col: u8 = 0;
    let row: u8 = 1; // MemTile row in NPU1.

    // Build a Start_Queue value that pushes BD 26 (>= 16, exercises the
    // 6-bit field).  Insert via the MemTile layout so the bits land at
    // [5:0]; using a literal would make the test depend on a specific
    // bit position.
    let lay = &device_reg_layout().memtile_channel;
    let bd_id_raw = lay.start_bd_id.insert(0, 26);

    // Drive the universal register bus on the MemTile S2MM ch0 Start_Queue
    // offset; this routes through `write_memtile_dma_channel`, the single
    // path both CDO and non-CDO writes traverse.
    let addr = TileAddress::encode(col, row, MEMTILE_DMA_S2MM_0_START_QUEUE);
    state.write_register(addr, bd_id_raw).unwrap();

    let tile = state.array.tile(col, row);
    assert_eq!(
        tile.dma_channels[0].current_bd, 26,
        "MemTile BD>=16 must be preserved by the 6-bit memtile_channel layout; \
         got {} (4-bit truncation would yield 10)",
        tile.dma_channels[0].current_bd,
    );
}

#[test]
fn dma_start_queue_compute_uses_4bit_field() {
    // Compute Start_BD_ID is 4-bit; values >= 16 are intentionally
    // truncated by hardware (compute tiles only have 16 BDs).  This
    // pins the negative side of the helper -- compute tiles must keep
    // the legacy 4-bit semantics.
    use crate::device::regdb::device_reg_layout;
    use xdna_archspec::aie2::registers::dma::COMPUTE_DMA_S2MM_0_START_QUEUE;

    let mut state = DeviceState::new_npu1();
    let col: u8 = 0;
    let row: u8 = 2; // Compute row.

    // Stuff BD 7 (a normal compute BD) into the compute layout.
    let lay = &device_reg_layout().memory_channel;
    let bd_id_raw = lay.start_bd_id.insert(0, 7);

    // Drive the universal register bus on the Compute S2MM ch0 Start_Queue
    // offset; this routes through `write_dma_channel`, the single path both
    // CDO and non-CDO writes traverse.
    let addr = TileAddress::encode(col, row, COMPUTE_DMA_S2MM_0_START_QUEUE);
    state.write_register(addr, bd_id_raw).unwrap();

    let tile = state.array.tile(col, row);
    assert_eq!(tile.dma_channels[0].current_bd, 7);
}

#[test]
fn channel_field_layout_helper_picks_memtile_for_memtile_kind() {
    // Direct test of the helper that `write_dma_channel` and
    // `mask_write_dma_channel` consult before extracting
    // Start_BD_ID / Repeat_Count / Enable_Token_Issue.
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
    assert_eq!(lay_mem.start_bd_id.width, 6, "MemTile Start_BD_ID is 6-bit per aie-rt xaiemlgbl_params.h");

    let lay_compute = channel_field_layout(reg_layout, TileKind::Compute);
    assert_eq!(
        lay_compute.start_bd_id.width, 4,
        "Compute Start_BD_ID is 4-bit per aie-rt xaiemlgbl_params.h"
    );

    let lay_shim = channel_field_layout(reg_layout, TileKind::ShimNoc);
    assert_eq!(lay_shim.start_bd_id.width, 4, "Shim Start_BD_ID is 4-bit (same as compute)");
}

/// Regression test: a CDO write to CORE_CONTROL must be visible via
/// the register-bus read path (`tile.read_register_pure(offset)`).
///
/// The bug being pinned: a typed `apply_core_enable` handler used to
/// update only `tile.core.control` and skip `tile.registers`, so a
/// subsequent `read_register_pure(CORE_CONTROL)` returned 0 instead
/// of the written value. The fix routes `apply_device_op::CoreEnable`
/// through `write_register`, which stores the raw word in
/// `tile.registers` alongside running the offset-dispatch branch.
#[test]
fn core_control_cdo_write_is_readable_via_register_bus() {
    use crate::parser::cdo::semantics::lower;
    use crate::parser::cdo::CdoRaw;

    let mut state = DeviceState::new_npu1();
    let col: u8 = 1;
    let row: u8 = 2; // compute row
    let cc_offset = xdna_archspec::aie2::registers::CORE_CONTROL;
    let address = TileAddress::encode(col, row, cc_offset);

    let cmd = CdoRaw::Write { address, value: 0x1 };
    for op in lower(&cmd) {
        state.apply_device_op(&op).unwrap();
    }

    let tile = state.array.tile(col, row);
    assert_eq!(
        tile.read_register_pure(cc_offset),
        0x1,
        "CDO write to CORE_CONTROL must be visible via register-bus reads"
    );
}

/// Regression test: a CDO MaskWrite to CORE_CONTROL with a partial
/// mask must only modify the bits covered by the mask, leaving other
/// bits unchanged.
///
/// The bug being pinned: `lower_mask_write` used to promote
/// `MaskWrite` to a typed `CoreEnable` op that dropped the mask, then
/// the apply path overwrote the full word — corrupting bits 31..1.
/// The fix removes the promotion entirely; MaskWrite rides through
/// `mask_write_register` -> `mask_write_core_register`, which
/// mask-blends correctly.
#[test]
fn core_control_cdo_mask_write_preserves_unmasked_bits() {
    use crate::parser::cdo::semantics::lower;
    use crate::parser::cdo::CdoRaw;

    let mut state = DeviceState::new_npu1();
    let col: u8 = 1;
    let row: u8 = 2; // compute row
    let cc_offset = xdna_archspec::aie2::registers::CORE_CONTROL;
    let address = TileAddress::encode(col, row, cc_offset);

    // Pre-set tile.core.control directly, bypassing the register bus.
    // The test is proving a bug in the write path itself, so we cannot
    // use the write path to set up the precondition.
    {
        let tile = state.array.get_mut(col, row).unwrap();
        tile.core.control = 0xABCD_0001;
        tile.core.enabled = true;
    }

    // Clear bit 0 (disable) but leave bits 31..1 alone.
    let cmd = CdoRaw::MaskWrite { address, mask: 0x1, value: 0x0 };
    for op in lower(&cmd) {
        state.apply_device_op(&op).unwrap();
    }

    let tile = state.array.tile(col, row);
    assert_eq!(
        tile.core.control, 0xABCD_0000,
        "MaskWrite with mask=0x1 must only clear bit 0, preserving bits 31..1"
    );
    assert!(!tile.core.enabled, "Bit 0 cleared -> core disabled");
}

/// Invariant proof for D.3 commit 2: writing CORE_CONTROL via the
/// register bus does not match any branch of
/// `apply_tile_local_effects` (cascade, shim mux, lock overflow/
/// underflow clear, perf counters, trace registers).
///
/// The non-CDO path already exercises this code path; this test
/// pins the invariant so that when D.3 commit 2 routes the CDO
/// promotion path through `write_register` (and thus through
/// `apply_tile_local_effects`), no surprise side effect fires.
///
/// GREEN today, GREEN after commit 2.
#[test]
fn register_bus_write_to_core_control_does_not_trigger_unrelated_tile_effects() {
    let mut state = DeviceState::new_npu1();
    let col: u8 = 1;
    let row: u8 = 2; // compute row
    let cc_offset = xdna_archspec::aie2::registers::CORE_CONTROL;
    let cc_addr = TileAddress::encode(col, row, cc_offset);

    // Snapshot fields apply_tile_local_effects can touch.
    let tile = state.array.tile(col, row);
    let cascade_in_pre = tile.cascade_input_dir;
    let cascade_out_pre = tile.cascade_output_dir;
    let lock_over_pre: Vec<bool> = tile.locks.iter().map(|l| l.overflow).collect();
    let lock_under_pre: Vec<bool> = tile.locks.iter().map(|l| l.underflow).collect();
    let mut reg_pre: Vec<(u32, u32)> = tile
        .registers_ref()
        .iter()
        .filter(|(&offset, _)| offset != cc_offset)
        .map(|(&offset, &value)| (offset, value))
        .collect();
    reg_pre.sort_by_key(|(o, _)| *o);

    // The register-map snapshot above is the catch-all: shim-mux config,
    // perf-counter latches, and trace-register state all live in
    // `tile.registers` as raw u32 entries (apply_tile_local_effects writes
    // them via `tile.registers.insert(...)`), so any side effect on those
    // subsystems would show up as a new (offset, value) entry in `reg_post`.
    // No separate accessor checks are needed.
    state.write_register(cc_addr, 0x1).unwrap();

    let tile = state.array.tile(col, row);
    assert_eq!(tile.cascade_input_dir, cascade_in_pre, "cascade_input_dir must not change");
    assert_eq!(tile.cascade_output_dir, cascade_out_pre, "cascade_output_dir must not change");

    let lock_over_post: Vec<bool> = tile.locks.iter().map(|l| l.overflow).collect();
    let lock_under_post: Vec<bool> = tile.locks.iter().map(|l| l.underflow).collect();
    assert_eq!(lock_over_pre, lock_over_post, "lock overflow bits must not change");
    assert_eq!(lock_under_pre, lock_under_post, "lock underflow bits must not change");

    let mut reg_post: Vec<(u32, u32)> = tile
        .registers_ref()
        .iter()
        .filter(|(&offset, _)| offset != cc_offset)
        .map(|(&offset, &value)| (offset, value))
        .collect();
    reg_post.sort_by_key(|(o, _)| *o);
    assert_eq!(
        reg_pre, reg_post,
        "Only CORE_CONTROL should change in the register map; other entries must be untouched"
    );

    // Sanity: the targeted effects DID happen.
    assert_eq!(tile.core.control, 0x1);
    assert_eq!(*tile.registers_ref().get(&cc_offset).unwrap(), 0x1);
}

/// Same invariant as the CORE_CONTROL test, but for a compute
/// Start_Queue offset. Verifies that writing Start_Queue via the
/// register bus does not trigger any apply_tile_local_effects branch.
#[test]
fn register_bus_write_to_compute_start_queue_does_not_trigger_unrelated_tile_effects() {
    use xdna_archspec::aie2::registers::dma::COMPUTE_DMA_S2MM_0_START_QUEUE;

    let mut state = DeviceState::new_npu1();
    let col: u8 = 1;
    let row: u8 = 2;
    let sq_offset = COMPUTE_DMA_S2MM_0_START_QUEUE;
    let sq_addr = TileAddress::encode(col, row, sq_offset);

    let tile = state.array.tile(col, row);
    let cascade_in_pre = tile.cascade_input_dir;
    let cascade_out_pre = tile.cascade_output_dir;
    let lock_over_pre: Vec<bool> = tile.locks.iter().map(|l| l.overflow).collect();
    let lock_under_pre: Vec<bool> = tile.locks.iter().map(|l| l.underflow).collect();

    // Write 0 (BD 0, no repeat) -- innocuous payload, just need the
    // write to land on the offset.
    // This test is intentionally shallower than the CORE_CONTROL twin:
    // Start_Queue writes legitimately mutate DMA-channel state (BD enqueue),
    // so a full `tile.registers` snapshot would be noisy. Cascade and lock
    // overflow/underflow are the only fields apply_tile_local_effects could
    // touch that are safely orthogonal to a Start_Queue write.
    state.write_register(sq_addr, 0x0).unwrap();

    let tile = state.array.tile(col, row);
    assert_eq!(tile.cascade_input_dir, cascade_in_pre);
    assert_eq!(tile.cascade_output_dir, cascade_out_pre);

    let lock_over_post: Vec<bool> = tile.locks.iter().map(|l| l.overflow).collect();
    let lock_under_post: Vec<bool> = tile.locks.iter().map(|l| l.underflow).collect();
    assert_eq!(lock_over_pre, lock_over_post);
    assert_eq!(lock_under_pre, lock_under_post);
}

/// `NeighborView::tile(own_col, own_row)` must return `None` -- the executing
/// tile is held by `&mut Tile` from the same `split_tile_mut` call, so it's
/// deliberately a hole in the view to keep the borrow safe.
#[test]
fn neighbor_view_holes_out_own_tile() {
    let mut device = DeviceState::new_npu1();
    let (own_col, own_row) = (1, 3);
    let (_own, view) = device.split_tile_mut(own_col, own_row).expect("split valid");

    assert!(view.tile(own_col, own_row).is_none(), "own tile must be a hole in the view");
    // Sanity: an existing neighbor IS visible through the view.
    assert!(view.tile(own_col, own_row - 1).is_some(), "south neighbor should be visible");
    assert!(view.tile(own_col - 1, own_row).is_some(), "west neighbor should be visible");
}

/// Out-of-bounds coordinates must return `None` rather than panic. This keeps
/// callers from having to bounds-check before calling.
#[test]
fn split_tile_mut_returns_none_for_oob() {
    let mut device = DeviceState::new_npu1();
    let cols = device.cols();
    let rows = device.rows();

    assert!(device.split_tile_mut(cols, 0).is_none(), "col >= cols");
    assert!(device.split_tile_mut(0, rows).is_none(), "row >= rows");
    assert!(device.split_tile_mut(cols + 5, rows + 5).is_none(), "both OOB");
}

// ----------------------------------------------------------------------
// Tile_Control isolation register write effect
//
// Compute Tile_Control is at 0x36030, memtile at 0x96030. Low 4 bits
// are S/W/N/E isolation per AM025; we mirror them onto tile.isolation
// so the cross-tile gates can read a single byte. Higher bits
// (clock-gating, etc.) flow through the generic register store
// untouched. See `apply_tile_local_effects` in state/effects.rs.
// ----------------------------------------------------------------------

#[test]
fn tile_control_write_updates_isolation_byte_compute() {
    use crate::device::tile::isolation as iso;
    let mut state = DeviceState::new_npu1();
    let col: u8 = 1;
    let row: u8 = 2;
    let addr = TileAddress::encode(col, row, 0x36030);

    // ISOLATE_FROM_SOUTH | ISOLATE_FROM_EAST = 0b1001 = 0x9.
    state.write_register(addr, 0x9).unwrap();
    let tile = state.array.tile(col, row);
    assert_eq!(tile.isolation, iso::SOUTH | iso::EAST);
    // Register store keeps the raw write so subsequent reads return it.
    assert_eq!(*tile.registers_ref().get(&0x36030).unwrap(), 0x9);
}

#[test]
fn tile_control_write_updates_isolation_byte_memtile() {
    use crate::device::tile::isolation as iso;
    let mut state = DeviceState::new_npu1();
    let col: u8 = 1;
    let row: u8 = 1;
    let addr = TileAddress::encode(col, row, 0x96030);

    state.write_register(addr, iso::ALL_DIRECTIONS as u32).unwrap();
    let tile = state.array.tile(col, row);
    assert_eq!(tile.isolation, iso::ALL_DIRECTIONS);
}

/// High bits of Tile_Control (clock-gating etc.) must NOT bleed into
/// the isolation byte. Only the low 4 bits matter to the gate.
#[test]
fn tile_control_write_masks_isolation_to_low_4_bits() {
    use crate::device::tile::isolation as iso;
    let mut state = DeviceState::new_npu1();
    let col: u8 = 1;
    let row: u8 = 2;
    let addr = TileAddress::encode(col, row, 0x36030);

    // Bits above [3:0] should be ignored by the isolation snapshot.
    state.write_register(addr, 0xFFFF_FFFA).unwrap();
    let tile = state.array.tile(col, row);
    assert_eq!(tile.isolation, 0xA & iso::ALL_DIRECTIONS);
}

/// Writing a different offset must NOT touch isolation, even if the
/// value would map to a non-zero direction byte. Pins the dispatch
/// guard: the effect only fires for the matching register.
#[test]
fn unrelated_register_write_does_not_change_isolation() {
    let mut state = DeviceState::new_npu1();
    let col: u8 = 1;
    let row: u8 = 2;

    // Pre-set isolation via Tile_Control to a known value.
    let tc_addr = TileAddress::encode(col, row, 0x36030);
    state.write_register(tc_addr, 0x5).unwrap();
    assert_eq!(state.array.tile(col, row).isolation, 0x5);

    // Write a totally unrelated offset; isolation byte must survive.
    let other_addr = TileAddress::encode(col, row, 0x36034);
    state.write_register(other_addr, 0xFFFF_FFFF).unwrap();
    assert_eq!(state.array.tile(col, row).isolation, 0x5);
}
