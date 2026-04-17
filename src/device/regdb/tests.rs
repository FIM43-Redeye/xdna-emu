//! Tests for the register database module.

use super::*;
use xdna_archspec::regdb::{AccessMode, BitField};

/// Helper to load the real database (skips if not available).
fn load_test_db() -> Option<RegisterDb> {
    super::load_for_device("aie2").ok()
}

#[test]
fn test_parse_register_database() {
    let Some(db) = load_test_db() else {
        eprintln!("Skipping: register database JSON not found (set MLIR_AIE_PATH)");
        return;
    };

    // Verify version string is present
    assert!(!db.version.is_empty(), "Version should not be empty");

    // Verify all expected modules exist
    assert!(db.module("core").is_some(), "core module missing");
    assert!(db.module("memory").is_some(), "memory module missing");
    assert!(db.module("memory_tile").is_some(), "memory_tile module missing");
    assert!(db.module("shim").is_some(), "shim module missing");

    // Verify register counts are substantial
    let mem = db.module("memory").unwrap();
    assert!(mem.registers.len() > 100,
        "Expected >100 registers in memory module, got {}", mem.registers.len());
}

#[test]
fn test_bitfield_extract() {
    // Buffer_Length: bits 13:0 (14 bits)
    let bf = BitField::from_range("test".to_string(), 0, 13);
    assert_eq!(bf.mask, 0x3FFF);
    assert_eq!(bf.shift, 0);
    assert_eq!(bf.extract(0x0000FFFF), 0x3FFF);
    assert_eq!(bf.extract(0xFFFF0000), 0x0000);

    // Base_Address: bits 27:14 (14 bits)
    let bf2 = BitField::from_range("test".to_string(), 14, 27);
    assert_eq!(bf2.mask, 0x3FFF);
    assert_eq!(bf2.shift, 14);
    assert_eq!(bf2.extract(0x0FFFC000), 0x3FFF);
    assert_eq!(bf2.extract(0x00003FFF), 0x0000);

    // Single bit: bit 31
    let bf3 = BitField::from_range("test".to_string(), 31, 31);
    assert_eq!(bf3.mask, 1);
    assert!(bf3.extract_bool(0x80000000));
    assert!(!bf3.extract_bool(0x7FFFFFFF));
}

#[test]
fn test_bitfield_insert() {
    // Base_Address: bits 27:14 (14 bits)
    let bf = BitField::from_range("test".to_string(), 14, 27);

    // Insert 0x100 into an empty word
    let word = bf.insert(0, 0x100);
    assert_eq!(word, 0x100 << 14);
    assert_eq!(bf.extract(word), 0x100);

    // Insert into a word with other fields set (should not disturb them)
    let word = bf.insert(0xFFFF, 0x200);
    // Low 14 bits should be preserved (0x3FFF from original 0xFFFF)
    assert_eq!(word & 0x3FFF, 0x3FFF);
    assert_eq!(bf.extract(word), 0x200);

    // Mask truncates values exceeding field width
    let word = bf.insert(0, 0xFFFF); // 16 bits into 14-bit field
    assert_eq!(bf.extract(word), 0x3FFF);
}

#[test]
fn test_bitfield_set_bit() {
    // Single-bit field at bit 19
    let bf = BitField::from_range("test".to_string(), 19, 19);
    let word = bf.set_bit(0);
    assert_eq!(word, 1 << 19);
    assert!(bf.extract_bool(word));

    // set_bit on already-set word is idempotent
    let word2 = bf.set_bit(word);
    assert_eq!(word, word2);

    // set_bit preserves other bits
    let word = bf.set_bit(0xDEAD_0000);
    assert_eq!(word, 0xDEAD_0000 | (1 << 19));
}

#[test]
fn test_bitfield_insert_roundtrip() {
    // Verify insert -> extract roundtrip for various field positions
    let fields = [
        BitField::from_range("low".to_string(), 0, 3),     // bits 3:0
        BitField::from_range("mid".to_string(), 8, 15),    // bits 15:8
        BitField::from_range("high".to_string(), 24, 31),  // bits 31:24
        BitField::from_range("single".to_string(), 19, 19), // single bit
    ];

    for bf in &fields {
        let max_val = bf.mask;
        let word = bf.insert(0, max_val);
        assert_eq!(bf.extract(word), max_val,
            "Roundtrip failed for field '{}' [{},{}]", bf.name, bf.lsb, bf.msb);
    }
}

#[test]
fn test_bd_field_layout_from_regdb() {
    let Some(db) = load_test_db() else {
        eprintln!("Skipping: register database JSON not found");
        return;
    };

    let layout = BdFieldLayout::from_regdb(&db, "memory")
        .expect("Failed to build BdFieldLayout");

    // Verify Buffer_Length
    assert_eq!(layout.buffer_length.lsb, 0);
    assert_eq!(layout.buffer_length.msb, 13);
    assert_eq!(layout.buffer_length.mask, 0x3FFF);

    // Verify Base_Address
    assert_eq!(layout.base_address.lsb, 14);
    assert_eq!(layout.base_address.msb, 27);

    // Verify a Word 5 field
    assert_eq!(layout.valid_bd.lsb, 25);
    assert_eq!(layout.valid_bd.msb, 25);
    assert_eq!(layout.valid_bd.mask, 1);
}

#[test]
fn test_channel_field_layout_from_regdb() {
    let Some(db) = load_test_db() else {
        eprintln!("Skipping: register database JSON not found");
        return;
    };

    let layout = ChannelFieldLayout::from_regdb(&db, "memory")
        .expect("Failed to build ChannelFieldLayout");

    // Verify FoT_Mode: bits 17:16
    assert_eq!(layout.fot_mode.lsb, 16);
    assert_eq!(layout.fot_mode.msb, 17);
    assert_eq!(layout.fot_mode.mask, 0x3);

    // Verify Start_BD_ID: bits 3:0
    assert_eq!(layout.start_bd_id.lsb, 0);
    assert_eq!(layout.start_bd_id.msb, 3);
    assert_eq!(layout.start_bd_id.mask, 0xF);
}

#[test]
fn test_device_reg_layout_from_regdb() {
    let Some(db) = load_test_db() else {
        eprintln!("Skipping: register database JSON not found");
        return;
    };

    let layout = DeviceRegLayout::from_regdb(db)
        .expect("Failed to build DeviceRegLayout");

    // Verify lock layout (AM025)
    assert_eq!(layout.memory_lock_base, 0x1F000);
    assert_eq!(layout.memory_lock_stride, 0x10);
    assert_eq!(layout.memtile_lock_base, 0xC0000);
    assert_eq!(layout.memtile_lock_stride, 0x10);
    assert_eq!(layout.shim_lock_base, 0x14000, "Shim Lock0 offset");
    assert_eq!(layout.shim_lock_stride, 0x10, "Shim lock stride");
    assert_eq!(layout.shim_locks_overflow, 0x14120, "Shim Locks_Overflow offset");
    assert_eq!(layout.shim_locks_underflow, 0x14128, "Shim Locks_Underflow offset");

    // Verify Lock_value field (data-driven from regdb)
    assert_eq!(layout.lock_value_width, 6, "Lock_value field width");
    assert_eq!(layout.lock_value_mask, 0x3F, "Lock_value field mask");
    assert_eq!(layout.lock_value_sign_bit, 5, "Lock_value sign bit");

    // Verify sign_extend_lock_value with known values
    assert_eq!(layout.sign_extend_lock_value(0), 0);
    assert_eq!(layout.sign_extend_lock_value(31), 31);    // max positive
    assert_eq!(layout.sign_extend_lock_value(0x20), -32); // min negative
    assert_eq!(layout.sign_extend_lock_value(0x3F), -1);  // all bits set
    assert_eq!(layout.sign_extend_lock_value(0xFF), -1);  // extra bits masked

    // Verify compute tile BD layout (AM025)
    assert_eq!(layout.memory_bd_base, 0x1D000, "Compute BD base");
    assert_eq!(layout.memory_bd_stride, 0x20, "Compute BD stride");
    assert_eq!(layout.memory_bd_words, 6, "Compute BD words");

    // Verify compute tile channel layout (AM025)
    assert_eq!(layout.memory_channel_base, 0x1DE00, "Compute channel base");
    assert_eq!(layout.memory_channel_stride, 0x08, "Compute channel stride");
    assert_eq!(layout.memory_status_base, 0x1DF00, "Compute status base");

    // Verify memtile BD layout (AM025)
    assert_eq!(layout.memtile_bd_base, 0xA0000, "MemTile BD base");
    assert_eq!(layout.memtile_bd_stride, 0x20, "MemTile BD stride");
    assert_eq!(layout.memtile_bd_words, 8, "MemTile BD words");

    // Verify memtile channel layout (AM025)
    assert_eq!(layout.memtile_channel_s2mm_base, 0xA0600, "MemTile S2MM base");
    assert_eq!(layout.memtile_channel_mm2s_base, 0xA0630, "MemTile MM2S base");
    assert_eq!(layout.memtile_channel_stride, 0x08, "MemTile channel stride");

    // Verify shim BD layout (AM025)
    assert_eq!(layout.shim_bd_base, 0x1D000, "Shim BD base");
    assert_eq!(layout.shim_bd_stride, 0x20, "Shim BD stride");
    assert_eq!(layout.shim_bd_words, 8, "Shim BD words");

    // Verify shim channel layout (AM025)
    assert_eq!(layout.shim_channel_base, 0x1D200, "Shim channel base");
    assert_eq!(layout.shim_channel_stride, 0x08, "Shim channel stride");

    // Verify stream switch slave slot layout (AM025)
    assert_eq!(layout.memory_stream_switch.slave_slot_port_stride, 0x10,
        "Compute slave slot port stride");
    assert_eq!(layout.memory_stream_switch.slave_slot_count, 4,
        "Compute slave slots per port");
    assert_eq!(layout.memtile_stream_switch.slave_slot_port_stride, 0x10,
        "MemTile slave slot port stride");
    assert_eq!(layout.memtile_stream_switch.slave_slot_count, 4,
        "MemTile slave slots per port");

    // Verify shim BD field layout was populated
    assert_eq!(layout.shim_bd.buffer_length.width, 32, "Shim buffer_length is 32-bit");
    assert_eq!(layout.shim_bd.base_address_low.lsb, 2, "Shim addr_low starts at bit 2");

    // Verify event/trace register layout (from register database)
    assert_eq!(layout.core_events.trace_control_base, 0x340D0, "Core Trace_Control0");
    assert_eq!(layout.core_events.trace_control_end, 0x340E4, "Core Trace_Event1");
    assert_eq!(layout.core_events.event_generate, 0x34008, "Core Event_Generate");
    assert_eq!(layout.core_events.event_broadcast_base, 0x34010, "Core Event_Broadcast0");
    assert_eq!(layout.core_events.edge_detection, 0x34408, "Core Edge_Detection");
    assert_eq!(layout.core_events.event_port_select, Some([0x3FF00, 0x3FF04]), "Core port select");

    assert_eq!(layout.memory_events.trace_control_base, 0x140D0, "Memory Trace_Control0");
    assert_eq!(layout.memory_events.event_generate, 0x14008, "Memory Event_Generate");
    assert_eq!(layout.memory_events.edge_detection, 0x14408, "Memory Edge_Detection");
    assert_eq!(layout.memory_events.event_port_select, None, "Memory has no port select");

    assert_eq!(layout.memtile_events.trace_control_base, 0x940D0, "MemTile Trace_Control0");
    assert_eq!(layout.memtile_events.event_generate, 0x94008, "MemTile Event_Generate");
    assert_eq!(layout.memtile_events.edge_detection, 0x94408, "MemTile Edge_Detection");
    assert_eq!(layout.memtile_events.event_port_select, Some([0xB0F00, 0xB0F04]), "MemTile port select");

    assert_eq!(layout.cascade_config_offset, 0x36060, "Accumulator_Control (cascade config)");
}

#[test]
fn validate_status_field_layout() {
    let Some(db) = load_test_db() else {
        eprintln!("Skipping: register database JSON not found");
        return;
    };

    let layout = DeviceRegLayout::from_regdb(db)
        .expect("Failed to build DeviceRegLayout");

    // Compute tile status fields (DMA_S2MM_Status_0)
    let cs = &layout.memory_status;
    assert_eq!(cs.status.lsb, 0, "Status[1:0]");
    assert_eq!(cs.status.msb, 1);
    assert_eq!(cs.stalled_lock_acq.lsb, 2, "Stalled_Lock_Acq[2]");
    assert_eq!(cs.stalled_lock_rel.lsb, 3, "Stalled_Lock_Rel[3]");
    assert_eq!(cs.stalled_stream.lsb, 4, "Stalled_Stream[4]");
    assert_eq!(cs.stalled_tct.lsb, 5, "Stalled_TCT[5]");
    assert_eq!(cs.error_bd_unavailable.lsb, 10, "Error_BD_Unavailable[10]");
    assert_eq!(cs.error_bd_invalid.lsb, 11, "Error_BD_Invalid[11]");
    assert_eq!(cs.task_queue_overflow.lsb, 18, "Task_Queue_Overflow[18]");
    assert_eq!(cs.channel_running.lsb, 19, "Channel_Running[19]");
    assert_eq!(cs.task_queue_size.lsb, 20, "Task_Queue_Size[22:20]");
    assert_eq!(cs.task_queue_size.msb, 22);
    assert_eq!(cs.cur_bd.lsb, 24, "Cur_BD[27:24] compute");
    assert_eq!(cs.cur_bd.msb, 27);
    assert_eq!(cs.cur_bd.width, 4, "Compute Cur_BD is 4 bits");

    // MemTile status fields -- key difference is Cur_BD width
    let ms = &layout.memtile_status;
    assert_eq!(ms.cur_bd.lsb, 24, "Cur_BD[29:24] memtile");
    assert_eq!(ms.cur_bd.msb, 29);
    assert_eq!(ms.cur_bd.width, 6, "MemTile Cur_BD is 6 bits (48 BDs)");
    // Other fields are the same
    assert_eq!(ms.channel_running.lsb, 19);
    assert_eq!(ms.task_queue_size.lsb, 20);
}

#[test]
fn validate_shim_mux_layout() {
    let Some(db) = load_test_db() else {
        eprintln!("Skipping: register database JSON not found");
        return;
    };

    let layout = DeviceRegLayout::from_regdb(db)
        .expect("Failed to build DeviceRegLayout");

    let mux = &layout.shim_mux;

    // Mux_Config at 0x1F000
    assert_eq!(mux.mux_offset, 0x1F000, "Mux_Config offset");

    // Mux fields: South2[9:8]->slave[4], South3[11:10]->slave[5],
    //             South6[13:12]->slave[8], South7[15:14]->slave[9]
    assert_eq!(mux.mux_fields.len(), 4, "Mux has 4 South port fields");
    // Sorted by port_index
    assert_eq!(mux.mux_fields[0].port_index, 4);
    assert_eq!(mux.mux_fields[0].field.lsb, 8);
    assert_eq!(mux.mux_fields[1].port_index, 5);
    assert_eq!(mux.mux_fields[1].field.lsb, 10);
    assert_eq!(mux.mux_fields[2].port_index, 8);
    assert_eq!(mux.mux_fields[2].field.lsb, 12);
    assert_eq!(mux.mux_fields[3].port_index, 9);
    assert_eq!(mux.mux_fields[3].field.lsb, 14);

    // Demux_Config at 0x1F004
    assert_eq!(mux.demux_offset, 0x1F004, "Demux_Config offset");

    // Demux fields: South2[5:4]->master[4], South3[7:6]->master[5],
    //               South4[9:8]->master[6], South5[11:10]->master[7]
    assert_eq!(mux.demux_fields.len(), 4, "Demux has 4 South port fields");
    assert_eq!(mux.demux_fields[0].port_index, 4);
    assert_eq!(mux.demux_fields[0].field.lsb, 4);
    assert_eq!(mux.demux_fields[1].port_index, 5);
    assert_eq!(mux.demux_fields[1].field.lsb, 6);
    assert_eq!(mux.demux_fields[2].port_index, 6);
    assert_eq!(mux.demux_fields[2].field.lsb, 8);
    assert_eq!(mux.demux_fields[3].port_index, 7);
    assert_eq!(mux.demux_fields[3].field.lsb, 10);
}

// ====================================================================
// Spot-check: verify JSON field layouts match AM025 expected values.
// These tests use inline expected values (from AM025) rather than
// hardcoded references, since the JSON is now the single authoritative
// source for bit field definitions.
// ====================================================================

#[test]
fn validate_memory_module_bd_fields() {
    let Some(db) = load_test_db() else {
        eprintln!("Skipping: register database JSON not found");
        return;
    };

    let mem = db.module("memory").unwrap();

    // BD base address (AM025: DMA_BD0_0 @ 0x1D000)
    let bd0_0 = mem.register("DMA_BD0_0").unwrap();
    assert_eq!(bd0_0.offset, 0x1D000, "DMA_BD0_0 offset");

    // BD stride: 0x20 (AM025: DMA_BD1_0 @ 0x1D020)
    let bd1_0 = mem.register("DMA_BD1_0").unwrap();
    assert_eq!(bd1_0.offset - bd0_0.offset, 0x20, "BD stride");

    // Word 0: Buffer_Length[13:0], Base_Address[27:14]
    let buf_len = bd0_0.field("Buffer_Length").unwrap();
    assert_eq!((buf_len.lsb, buf_len.msb), (0, 13), "Buffer_Length bits");
    assert_eq!(buf_len.mask, 0x3FFF);

    let base_addr = bd0_0.field("Base_Address").unwrap();
    assert_eq!((base_addr.lsb, base_addr.msb), (14, 27), "Base_Address bits");
    assert_eq!(base_addr.mask, 0x3FFF);

    // Word 1: packet control fields
    let bd0_1 = mem.register("DMA_BD0_1").unwrap();
    assert_eq!(bd0_1.field("Enable_Compression").unwrap().lsb, 31);
    assert_eq!(bd0_1.field("Enable_Packet").unwrap().lsb, 30);
    assert_eq!(bd0_1.field("Out_Of_Order_BD_ID").unwrap().lsb, 24);
    assert_eq!(bd0_1.field("Packet_ID").unwrap().lsb, 19);
    assert_eq!(bd0_1.field("Packet_Type").unwrap().lsb, 16);

    // Word 5: lock and chaining fields
    let bd0_5 = mem.register("DMA_BD0_5").unwrap();
    assert_eq!(bd0_5.field("TLAST_Suppress").unwrap().lsb, 31);
    assert_eq!(bd0_5.field("Next_BD").unwrap().lsb, 27);
    assert_eq!(bd0_5.field("Use_Next_BD").unwrap().lsb, 26);
    assert_eq!(bd0_5.field("Valid_BD").unwrap().lsb, 25);
    assert_eq!(bd0_5.field("Lock_Rel_Value").unwrap().lsb, 18);
    assert_eq!(bd0_5.field("Lock_Acq_Enable").unwrap().lsb, 12);
    assert_eq!(bd0_5.field("Lock_Acq_Value").unwrap().lsb, 5);
    assert_eq!(bd0_5.field("Lock_Acq_ID").unwrap().mask, 0xF);
}

#[test]
fn validate_lock_registers() {
    let Some(db) = load_test_db() else {
        eprintln!("Skipping: register database JSON not found");
        return;
    };

    // Memory module locks (AM025: Lock0_value @ 0x1F000, stride 0x10)
    let mem = db.module("memory").unwrap();
    let lock0 = mem.register("Lock0_value").unwrap();
    assert_eq!(lock0.offset, 0x1F000, "Lock0 offset");

    let lock1 = mem.register("Lock1_value").unwrap();
    assert_eq!(lock1.offset - lock0.offset, 0x10, "Lock stride");

    let lock_field = lock0.field("Lock_value").unwrap();
    assert_eq!((lock_field.lsb, lock_field.msb), (0, 5), "Lock_value bits");

    // Memory tile locks (AM025: Lock0_value @ 0xC0000, stride 0x10)
    let mt_mod = db.module("memory_tile").unwrap();
    let mt_lock0 = mt_mod.register("Lock0_value").unwrap();
    assert_eq!(mt_lock0.offset, 0xC0000, "MemTile Lock0 offset");

    let mt_lock1 = mt_mod.register("Lock1_value").unwrap();
    assert_eq!(mt_lock1.offset - mt_lock0.offset, 0x10, "MemTile Lock stride");

    // Shim tile locks (AM025: Lock0_value @ 0x14000, stride 0x10)
    let shim_mod = db.module("shim").unwrap();
    let shim_lock0 = shim_mod.register("Lock0_value").unwrap();
    assert_eq!(shim_lock0.offset, 0x14000, "Shim Lock0 offset");

    let shim_lock1 = shim_mod.register("Lock1_value").unwrap();
    assert_eq!(shim_lock1.offset - shim_lock0.offset, 0x10, "Shim Lock stride");

    let shim_lock_field = shim_lock0.field("Lock_value").unwrap();
    assert_eq!((shim_lock_field.lsb, shim_lock_field.msb), (0, 5), "Shim Lock_value bits");

    // Shim lock overflow/underflow status registers
    let shim_overflow = shim_mod.register("Locks_Overflow").unwrap();
    assert_eq!(shim_overflow.offset, 0x14120, "Shim Locks_Overflow offset");
    let shim_underflow = shim_mod.register("Locks_Underflow").unwrap();
    assert_eq!(shim_underflow.offset, 0x14128, "Shim Locks_Underflow offset");
}

#[test]
fn validate_dma_channel_registers() {
    let Some(db) = load_test_db() else {
        eprintln!("Skipping: register database JSON not found");
        return;
    };

    let mem = db.module("memory").unwrap();

    // S2MM channel control (AM025: 0x1DE00)
    let s2mm_ctrl = mem.register("DMA_S2MM_0_Ctrl").unwrap();
    assert_eq!(s2mm_ctrl.offset, 0x1DE00, "S2MM_0_Ctrl offset");

    // Start queue at +4
    let start_q = mem.register("DMA_S2MM_0_Start_Queue").unwrap();
    assert_eq!(start_q.offset, 0x1DE04, "S2MM_0_Start_Queue offset");

    // Channel stride: 0x08
    let s2mm1_ctrl = mem.register("DMA_S2MM_1_Ctrl").unwrap();
    assert_eq!(s2mm1_ctrl.offset - s2mm_ctrl.offset, 0x08, "Channel stride");

    // Channel control fields (AM025)
    assert_eq!(s2mm_ctrl.field("FoT_Mode").unwrap().lsb, 16);
    assert_eq!(s2mm_ctrl.field("FoT_Mode").unwrap().mask, 0x3);
    assert_eq!(s2mm_ctrl.field("Controller_ID").unwrap().lsb, 8);
    assert_eq!(s2mm_ctrl.field("Controller_ID").unwrap().mask, 0xFF);
    assert_eq!(s2mm_ctrl.field("Decompression_Enable").unwrap().lsb, 4);
    assert_eq!(s2mm_ctrl.field("Enable_Out_of_Order").unwrap().lsb, 3);
    assert_eq!(s2mm_ctrl.field("Reset").unwrap().lsb, 1);

    // Start queue fields
    assert_eq!(start_q.field("Enable_Token_Issue").unwrap().lsb, 31);
    assert_eq!(start_q.field("Repeat_Count").unwrap().lsb, 16);
    assert_eq!(start_q.field("Repeat_Count").unwrap().mask, 0xFF);
    assert_eq!(start_q.field("Start_BD_ID").unwrap().mask, 0xF);
}

#[test]
fn validate_memtile_bd_fields() {
    let Some(db) = load_test_db() else {
        eprintln!("Skipping: register database JSON not found");
        return;
    };

    let mt_mod = db.module("memory_tile").unwrap();

    // BD base (AM025: DMA_BD0_0 @ 0xA0000)
    let bd0_0 = mt_mod.register("DMA_BD0_0").unwrap();
    assert_eq!(bd0_0.offset, 0xA0000, "MemTile DMA_BD0_0 offset");

    // BD stride: 0x20
    let bd1_0 = mt_mod.register("DMA_BD1_0").unwrap();
    assert_eq!(bd1_0.offset - bd0_0.offset, 0x20, "MemTile BD stride");

    // Word 0: Buffer_Length[16:0] (17 bits for MemTile)
    let buf_len = bd0_0.field("Buffer_Length").unwrap();
    assert_eq!(buf_len.mask, 0x1FFFF, "MemTile Buffer_Length mask");

    // Word 1: Base_Address[18:0], Use_Next_BD[19], Next_BD[25:20]
    let bd0_1 = mt_mod.register("DMA_BD0_1").unwrap();
    assert_eq!(bd0_1.field("Base_Address").unwrap().mask, 0x7FFFF);
    assert_eq!(bd0_1.field("Use_Next_BD").unwrap().lsb, 19);
    assert_eq!(bd0_1.field("Next_BD").unwrap().lsb, 20);
    assert_eq!(bd0_1.field("Next_BD").unwrap().mask, 0x3F);

    // Word 7: Lock and valid fields
    let bd0_7 = mt_mod.register("DMA_BD0_7").unwrap();
    assert_eq!(bd0_7.field("Valid_BD").unwrap().lsb, 31);
    assert_eq!(bd0_7.field("Lock_Rel_Value").unwrap().lsb, 24);
    assert_eq!(bd0_7.field("Lock_Rel_Value").unwrap().mask, 0x7F);
    assert_eq!(bd0_7.field("Lock_Rel_ID").unwrap().lsb, 16);
    assert_eq!(bd0_7.field("Lock_Rel_ID").unwrap().mask, 0xFF);
    assert_eq!(bd0_7.field("Lock_Acq_Enable").unwrap().lsb, 15);
    assert_eq!(bd0_7.field("Lock_Acq_Value").unwrap().lsb, 8);
    assert_eq!(bd0_7.field("Lock_Acq_Value").unwrap().mask, 0x7F);
    assert_eq!(bd0_7.field("Lock_Acq_ID").unwrap().mask, 0xFF);
}

#[test]
fn validate_memtile_channel_registers() {
    let Some(db) = load_test_db() else {
        eprintln!("Skipping: register database JSON not found");
        return;
    };

    let mt_mod = db.module("memory_tile").unwrap();

    // S2MM channel base (AM025: 0xA0600)
    let s2mm_ctrl = mt_mod.register("DMA_S2MM_0_Ctrl").unwrap();
    assert_eq!(s2mm_ctrl.offset, 0xA0600, "MemTile S2MM_0_Ctrl offset");

    // MM2S channel base (AM025: 0xA0630)
    let mm2s_ctrl = mt_mod.register("DMA_MM2S_0_Ctrl").unwrap();
    assert_eq!(mm2s_ctrl.offset, 0xA0630, "MemTile MM2S_0_Ctrl offset");

    // Channel stride: 0x08
    let s2mm1_ctrl = mt_mod.register("DMA_S2MM_1_Ctrl").unwrap();
    assert_eq!(s2mm1_ctrl.offset - s2mm_ctrl.offset, 0x08, "MemTile channel stride");
}

#[test]
fn validate_shim_bd_fields() {
    let Some(db) = load_test_db() else {
        eprintln!("Skipping: register database JSON not found");
        return;
    };

    let shim = db.module("shim").unwrap();

    // BD base (AM025: DMA_BD0_0 @ 0x1D000)
    let bd0_0 = shim.register("DMA_BD0_0").unwrap();
    assert_eq!(bd0_0.offset, 0x1D000, "Shim DMA_BD0_0 offset");

    // BD stride: 0x20
    let bd1_0 = shim.register("DMA_BD1_0").unwrap();
    assert_eq!(bd1_0.offset - bd0_0.offset, 0x20, "Shim BD stride");

    // Word 0: Buffer_Length[31:0] (full 32 bits for DDR)
    let buf_len = bd0_0.field("Buffer_Length").unwrap();
    assert_eq!((buf_len.lsb, buf_len.msb), (0, 31), "Shim Buffer_Length bits");

    // Word 1: Base_Address_Low[31:2]
    let bd0_1 = shim.register("DMA_BD0_1").unwrap();
    let addr_low = bd0_1.field("Base_Address_Low").unwrap();
    assert_eq!((addr_low.lsb, addr_low.msb), (2, 31), "Shim Base_Address_Low bits");

    // Word 2: Base_Address_High[15:0], packet fields
    let bd0_2 = shim.register("DMA_BD0_2").unwrap();
    assert_eq!(bd0_2.field("Base_Address_High").unwrap().msb, 15);
    assert_eq!(bd0_2.field("Enable_Packet").unwrap().lsb, 30);
    assert_eq!(bd0_2.field("Out_Of_Order_BD_ID").unwrap().lsb, 24);

    // Word 3: D0_Stepsize[19:0] (20-bit for DDR range)
    let bd0_3 = shim.register("DMA_BD0_3").unwrap();
    assert_eq!(bd0_3.field("D0_Stepsize").unwrap().msb, 19, "Shim D0_Stepsize 20-bit");
    assert_eq!(bd0_3.field("Secure_Access").unwrap().lsb, 30);

    // Word 4: Burst_Length[31:30]
    let bd0_4 = shim.register("DMA_BD0_4").unwrap();
    assert_eq!(bd0_4.field("Burst_Length").unwrap().lsb, 30);

    // Word 5: SMID[31:28], AxCache[27:24], AxQoS[23:20]
    let bd0_5 = shim.register("DMA_BD0_5").unwrap();
    assert_eq!(bd0_5.field("SMID").unwrap().lsb, 28);
    assert_eq!(bd0_5.field("AxCache").unwrap().lsb, 24);
    assert_eq!(bd0_5.field("AxQoS").unwrap().lsb, 20);

    // Word 7: locks and chaining (same layout as compute BD word 5)
    let bd0_7 = shim.register("DMA_BD0_7").unwrap();
    assert_eq!(bd0_7.field("Valid_BD").unwrap().lsb, 25);
    assert_eq!(bd0_7.field("Lock_Acq_ID").unwrap().mask, 0xF);

    // Channel base (AM025: 0x1D200)
    let s2mm_ctrl = shim.register("DMA_S2MM_0_Ctrl").unwrap();
    assert_eq!(s2mm_ctrl.offset, 0x1D200, "Shim S2MM_0_Ctrl offset");

    // Channel stride: 0x08
    let s2mm1_ctrl = shim.register("DMA_S2MM_1_Ctrl").unwrap();
    assert_eq!(s2mm1_ctrl.offset - s2mm_ctrl.offset, 0x08, "Shim channel stride");
}

#[test]
fn validate_core_module_registers() {
    use xdna_archspec::aie2::registers as cm;

    let Some(db) = load_test_db() else {
        eprintln!("Skipping: register database JSON not found");
        return;
    };

    let core = db.module("core").unwrap();

    // Cross-validate all hardcoded core_module constants against AM025 JSON.
    // These remain hardcoded for hot-path match arms, but this test catches
    // drift if the toolchain or JSON evolves.
    //
    // Note: JSON register names omit the "Core_" prefix used in our
    // constants for some registers. E.g. "Core_Enable_Events" in our code
    // corresponds to "Enable_Events" in the JSON.
    assert_eq!(core.register("Core_Control").unwrap().offset, cm::CORE_CONTROL);
    assert_eq!(core.register("Core_Status").unwrap().offset, cm::CORE_STATUS);
    assert_eq!(core.register("Enable_Events").unwrap().offset, cm::CORE_ENABLE_EVENTS);
    assert_eq!(core.register("Reset_Event").unwrap().offset, cm::CORE_RESET_EVENT);
    assert_eq!(core.register("Core_PC").unwrap().offset, cm::CORE_PC);
    assert_eq!(core.register("Core_SP").unwrap().offset, cm::CORE_SP);
    assert_eq!(core.register("Core_LR").unwrap().offset, cm::CORE_LR);
    assert_eq!(core.register("Debug_Control0").unwrap().offset, cm::CORE_DEBUG_CONTROL0);
    assert_eq!(core.register("Tile_Control").unwrap().offset, cm::TILE_CONTROL);
    assert_eq!(core.register("Memory_Control").unwrap().offset, cm::MEMORY_CONTROL);
}

/// Spot check: extract a known BD configuration from raw words using
/// the JSON-loaded layout, verify field extraction correctness.
#[test]
fn spot_check_bd_extraction() {
    let Some(db) = load_test_db() else {
        eprintln!("Skipping: register database JSON not found");
        return;
    };

    let layout = DeviceRegLayout::from_regdb(db)
        .expect("Failed to build DeviceRegLayout");

    // Construct a realistic BD word 0:
    // Buffer_Length = 1024 words (0x400), Base_Address = 0x100
    let word0: u32 = (0x100 << 14) | 0x400;
    assert_eq!(layout.memory_bd.buffer_length.extract(word0), 0x400);
    assert_eq!(layout.memory_bd.base_address.extract(word0), 0x100);

    // Construct a realistic BD word 5:
    // Valid=1, Use_Next=1, Next_BD=3, Lock_Acq_ID=5, Lock_Acq_Value=1
    let word5: u32 = (1 << 25) | (1 << 26) | (3 << 27) | (1 << 5) | 5;
    assert!(layout.memory_bd.valid_bd.extract_bool(word5));
    assert!(layout.memory_bd.use_next_bd.extract_bool(word5));
    assert_eq!(layout.memory_bd.next_bd.extract(word5), 3);
    assert_eq!(layout.memory_bd.lock_acq_id.extract(word5), 5);
}

// ====================================================================
// Tier 3: Register metadata (reset values, access modes, widths)
// ====================================================================

#[test]
fn test_register_width_and_access_parsed() {
    let Some(db) = load_test_db() else {
        eprintln!("Skipping: register database JSON not found");
        return;
    };

    let core = db.module("core").unwrap();

    // Most core registers are 32-bit, but some (e.g. Program_Memory) are wider
    let wide_regs: Vec<&str> = core.registers.iter()
        .filter(|r| r.width != 32)
        .map(|r| r.name.as_str())
        .collect();
    // Program_Memory is 128-bit (VLIW bundle interface)
    assert!(wide_regs.contains(&"Program_Memory"),
        "Program_Memory should be wider than 32 bits");
    let pm = core.register("Program_Memory").unwrap();
    assert_eq!(pm.width, 128, "Program_Memory should be 128-bit");

    // Core_Status is read-only (hardware reports core state)
    let status = core.register("Core_Status").unwrap();
    assert_eq!(status.access, AccessMode::ReadOnly,
        "Core_Status should be read-only");

    // Core_Control is mixed (some bits are w1tc, others rw)
    let ctrl = core.register("Core_Control").unwrap();
    assert!(
        ctrl.access == AccessMode::Mixed || ctrl.access == AccessMode::ReadWrite,
        "Core_Control should be mixed or rw, got {:?}", ctrl.access
    );
}

#[test]
fn test_known_nonzero_reset_values() {
    let Some(db) = load_test_db() else {
        eprintln!("Skipping: register database JSON not found");
        return;
    };

    let core = db.module("core").unwrap();

    // Core_LE (Loop End) has a non-zero reset value (0x000FFFFF per AM025)
    let core_le = core.register("Core_LE").unwrap();
    assert_ne!(core_le.reset_value, 0,
        "Core_LE should have non-zero reset value");

    // Core_Control has reset 0x00000002 (bit 1 = Reset set on power-on)
    let core_ctrl = core.register("Core_Control").unwrap();
    assert_eq!(core_ctrl.reset_value, 0x00000002,
        "Core_Control reset should be 0x02 (Reset bit set)");
}

#[test]
fn test_non_zero_reset_values_iterator() {
    let Some(db) = load_test_db() else {
        eprintln!("Skipping: register database JSON not found");
        return;
    };

    let core = db.module("core").unwrap();
    let non_zero: Vec<(u32, u32)> = core.non_zero_reset_values().collect();

    // There should be some non-zero reset values in the core module
    assert!(!non_zero.is_empty(),
        "Core module should have at least one non-zero reset value");

    // Core_Control @ 0x32000 should be in the list with reset=0x02
    assert!(non_zero.iter().any(|&(off, val)| off == 0x32000 && val == 0x02),
        "Core_Control (0x32000) with reset 0x02 should be in non-zero list");
}

#[test]
fn test_access_mode_distribution() {
    let Some(db) = load_test_db() else {
        eprintln!("Skipping: register database JSON not found");
        return;
    };

    // Count access modes across all modules
    let mut rw_count = 0usize;
    let mut ro_count = 0usize;
    let mut wo_count = 0usize;
    let mut wtc_count = 0usize;
    let mut mixed_count = 0usize;

    for module in db.modules.values() {
        for reg in &module.registers {
            match reg.access {
                AccessMode::ReadWrite => rw_count += 1,
                AccessMode::ReadOnly => ro_count += 1,
                AccessMode::WriteOnly => wo_count += 1,
                AccessMode::WriteToClear => wtc_count += 1,
                AccessMode::Mixed => mixed_count += 1,
            }
        }
    }

    // Based on earlier exploration: rw~1592, wo~82, ro~70, wtc~36, mixed~26
    assert!(rw_count > 1000, "Expected >1000 rw registers, got {}", rw_count);
    assert!(ro_count > 30, "Expected >30 ro registers, got {}", ro_count);
    assert!(wo_count > 30, "Expected >30 wo registers, got {}", wo_count);
    assert!(wtc_count > 10, "Expected >10 wtc registers, got {}", wtc_count);
    assert!(mixed_count > 10, "Expected >10 mixed registers, got {}", mixed_count);
}

#[test]
fn validate_stream_switch_layout() {
    let Some(db) = load_test_db() else {
        eprintln!("Skipping: register database JSON not found");
        return;
    };

    // Core module (compute tile) stream switch
    // AM025 JSON classifies these under "core", not "memory"
    let core_ss = StreamSwitchLayout::from_regdb(&db, "core")
        .expect("core stream switch layout");
    // Base addresses must match exactly (first register in each group)
    assert_eq!(core_ss.master_base, 0x3F000,
        "core master_base should be 0x3F000");
    assert_eq!(core_ss.slave_base, 0x3F100,
        "core slave_base should be 0x3F100");
    assert_eq!(core_ss.slave_slot_base, 0x3F200,
        "core slave_slot_base should be 0x3F200");
    // End addresses are last_register + 4. The old hardcoded values were
    // padded round numbers (0x3F058, 0x3F180, 0x3F390). The JSON-derived
    // values are tighter: exact end of the defined register space. Verify
    // they are above base and within the same address block.
    assert!(core_ss.master_end > core_ss.master_base
        && core_ss.master_end <= 0x3F100,
        "core master_end {:#X} should be in (0x3F000, 0x3F100]",
        core_ss.master_end);
    assert!(core_ss.slave_end > core_ss.slave_base
        && core_ss.slave_end <= 0x3F200,
        "core slave_end {:#X} should be in (0x3F100, 0x3F200]",
        core_ss.slave_end);
    assert!(core_ss.slave_slot_end > core_ss.slave_slot_base
        && core_ss.slave_slot_end <= 0x3F400,
        "core slave_slot_end {:#X} should be in (0x3F200, 0x3F400]",
        core_ss.slave_slot_end);

    // Memory tile stream switch
    let mt_ss = StreamSwitchLayout::from_regdb(&db, "memory_tile")
        .expect("memtile stream switch layout");
    assert_eq!(mt_ss.master_base, 0xB0000,
        "memtile master_base should be 0xB0000");
    assert_eq!(mt_ss.slave_base, 0xB0100,
        "memtile slave_base should be 0xB0100");
    assert_eq!(mt_ss.slave_slot_base, 0xB0200,
        "memtile slave_slot_base should be 0xB0200");
    assert!(mt_ss.master_end > mt_ss.master_base
        && mt_ss.master_end <= 0xB0100,
        "memtile master_end {:#X} should be in (0xB0000, 0xB0100]",
        mt_ss.master_end);
    assert!(mt_ss.slave_end > mt_ss.slave_base
        && mt_ss.slave_end <= 0xB0200,
        "memtile slave_end {:#X} should be in (0xB0100, 0xB0200]",
        mt_ss.slave_end);
    assert!(mt_ss.slave_slot_end > mt_ss.slave_slot_base
        && mt_ss.slave_slot_end <= 0xB0400,
        "memtile slave_slot_end {:#X} should be in (0xB0200, 0xB0400]",
        mt_ss.slave_slot_end);
}

#[test]
fn test_registers_with_access_filter() {
    let Some(db) = load_test_db() else {
        eprintln!("Skipping: register database JSON not found");
        return;
    };

    let core = db.module("core").unwrap();

    // Core_Status should be in the read-only set
    let ro_regs: Vec<&str> = core.registers_with_access(AccessMode::ReadOnly)
        .map(|r| r.name.as_str())
        .collect();
    assert!(ro_regs.contains(&"Core_Status"),
        "Core_Status should be in read-only registers");

    // DMA status registers are typically read-only
    let mem = db.module("memory").unwrap();
    let mem_ro: Vec<&str> = mem.registers_with_access(AccessMode::ReadOnly)
        .map(|r| r.name.as_str())
        .collect();
    assert!(mem_ro.iter().any(|n| n.contains("Status")),
        "Memory module should have read-only status registers");
}
