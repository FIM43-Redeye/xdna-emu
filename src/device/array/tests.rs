//! Tests for the tile array module.

use super::*;

#[test]
fn test_array_creation_npu1() {
    let array = TileArray::npu1();
    assert_eq!(array.cols(), 5);
    assert_eq!(array.rows(), 6);
    assert_eq!(array.tiles.len(), 30);

    let (shim, mem, compute) = array.count_by_type();
    assert_eq!(shim, 5); // Row 0, all columns
    assert_eq!(mem, 5); // Row 1, all columns
    assert_eq!(compute, 20); // Rows 2-5, all columns
}

#[test]
fn test_tile_access() {
    let mut array = TileArray::npu1();

    // Access by coordinates
    let tile = array.get(1, 2).unwrap();
    assert_eq!(tile.col, 1);
    assert_eq!(tile.row, 2);
    assert!(tile.is_compute());

    // Modify tile
    let tile = array.get_mut(1, 2).unwrap();
    tile.core.pc = 0x100;
    assert_eq!(array.tile(1, 2).core.pc, 0x100);
}

#[test]
fn test_out_of_bounds() {
    let array = TileArray::npu1();
    assert!(array.get(10, 0).is_none());
    assert!(array.get(0, 10).is_none());
}

/// Column reset (AIE_Tile_Column_Reset assert edge) clears the state of
/// every non-shim tile in the column -- core, locks, DMA -- but preserves
/// tile memory and leaves the shim row and other columns untouched.
#[test]
fn reset_column_clears_nonshim_state_preserves_memory_and_isolates_columns() {
    let mut array = TileArray::npu1();

    // Stamp recognizable state into column 2: a compute tile (row 2), the
    // mem tile (row 1), plus a memory marker that must survive the reset.
    {
        let t = array.get_mut(2, 2).unwrap();
        t.core.pc = 0x200;
        t.locks[0].value = 7;
        assert!(t.write_data_u32(0x40, 0xDEAD_BEEF));
    }
    array.get_mut(2, 1).unwrap().locks[0].value = 4;

    // A different column must be entirely unaffected.
    {
        let t = array.get_mut(1, 2).unwrap();
        t.core.pc = 0x111;
        t.locks[0].value = 5;
    }

    array.reset_column(2);

    // Column 2 non-shim tiles: state cleared, memory preserved.
    let t = array.get(2, 2).unwrap();
    assert_eq!(t.core.pc, 0, "compute core PC reset to boot state");
    assert_eq!(t.locks[0].value, 0, "compute lock reset to boot state");
    assert_eq!(t.read_data_u32(0x40), Some(0xDEAD_BEEF), "data memory preserved across reset");
    assert_eq!(array.get(2, 1).unwrap().locks[0].value, 0, "mem-tile lock reset");

    // Other column untouched.
    assert_eq!(array.get(1, 2).unwrap().core.pc, 0x111, "other column core untouched");
    assert_eq!(array.get(1, 2).unwrap().locks[0].value, 5, "other column lock untouched");
}

#[test]
fn test_tile_types() {
    let array = TileArray::npu1();

    // Shim tiles at row 0
    assert!(array.tile(0, 0).is_shim());
    assert!(array.tile(4, 0).is_shim());

    // Mem tiles at row 1
    assert!(array.tile(0, 1).is_mem());
    assert!(array.tile(4, 1).is_mem());

    // Compute tiles at rows 2-5
    assert!(array.tile(0, 2).is_compute());
    assert!(array.tile(4, 5).is_compute());
}

#[test]
fn test_compute_tile_iteration() {
    let array = TileArray::npu1();
    let compute_count = array.compute_tiles().count();
    assert_eq!(compute_count, 20); // 5 cols x 4 rows (2-5)
}

#[test]
fn test_reset() {
    let mut array = TileArray::npu1();

    // Modify some state
    array.tile_mut(1, 2).core.pc = 0x1000;
    array.tile_mut(1, 2).locks[0].value = 5;

    // Reset
    array.reset();

    // Verify reset
    assert_eq!(array.tile(1, 2).core.pc, 0);
    assert_eq!(array.tile(1, 2).locks[0].value, 0);
}

// === DMA Engine Integration Tests ===

#[test]
fn test_dma_engine_creation() {
    let array = TileArray::npu1();

    // Each tile should have a DMA engine
    assert_eq!(array.dma_engines.len(), 30);

    // Compute tile (row 2+) should have 4 channels
    let engine = array.dma_engine(1, 2).unwrap();
    assert_eq!(engine.num_channels(), 4);

    // Memory tile (row 1) should have 12 channels
    let engine = array.dma_engine(1, 1).unwrap();
    assert_eq!(engine.num_channels(), 12);
}

#[test]
fn test_dma_engine_access() {
    let mut array = TileArray::npu1();

    // Get mutable engine and configure it
    let engine = array.dma_engine_mut(2, 3).unwrap();
    assert_eq!(engine.col, 2);
    assert_eq!(engine.row, 3);
}

#[test]
fn test_tile_and_dma() {
    let mut array = TileArray::npu1();

    // Get both tile and DMA engine
    let (tile, engine) = array.tile_and_dma(3, 4).unwrap();
    assert_eq!(tile.col, 3);
    assert_eq!(tile.row, 4);
    assert_eq!(engine.col, 3);
    assert_eq!(engine.row, 4);
}

#[test]
fn test_no_active_dma_initially() {
    let array = TileArray::npu1();
    assert!(!array.any_dma_active());
}

#[test]
fn test_any_dma_active_includes_queued_tasks() {
    // A channel with FSM=Idle but a queued task must report active --
    // otherwise TDR's quiescent / natural-completion path can fire while
    // work is still pending (sibling fix to bb3aad5's Phase-5 tick).
    use crate::device::dma::BdConfig;

    let mut array = TileArray::npu1();
    assert!(!array.any_dma_active(), "fresh array reports no DMA");

    let engine = array.dma_engine_mut(1, 2).unwrap();
    engine.configure_bd(0, BdConfig::simple_1d(0x100, 32)).unwrap();
    engine.enqueue_task(2, 0, 0, false);
    // FSM has NOT been stepped yet -- channel is Idle with a queued task.

    assert!(array.any_dma_active(), "queued task on Idle FSM must register as active");
}

#[test]
fn test_dma_reset() {
    use crate::device::dma::BdConfig;

    let mut array = TileArray::npu1();

    // Configure and start a DMA transfer
    let engine = array.dma_engine_mut(1, 2).unwrap();
    engine.configure_bd(0, BdConfig::simple_1d(0x100, 32)).unwrap();
    engine.start_channel(0, 0).unwrap();
    assert!(engine.channel_active(0));

    // Reset should clear it
    array.reset();
    let engine = array.dma_engine(1, 2).unwrap();
    assert!(!engine.any_channel_active());
}

// === Clock Control Gate Tests ===

/// When the DMA module is clock-gated (column active, but MCC bit 1 clear on a
/// compute tile), `step_dma` must return `None` rather than running the engine.
#[test]
fn module_gate_skips_dma_when_dma_module_gated() {
    use crate::device::clock_control::ModuleKind;
    let mut array = TileArray::npu1();

    // Ungate column 2 (write to Column_Clock_Control at shim row 0).
    array.clock_mut().write_register(2, 0, 0x000FFF20, 0x1);

    // Write MCC for compute tile (col=2, row=3): Core only (bit 2), no DMA (bit 1).
    array.clock_mut().write_register(2, 3, 0x00060000, 1 << 2);

    assert!(array.clock().is_module_active(2, 3, ModuleKind::Core), "Core must be active");
    assert!(!array.clock().is_module_active(2, 3, ModuleKind::Dma), "Dma must be gated");

    // With no BDs configured, an active-DMA tile returns Some(DmaResult) and a
    // gated tile must return None.
    let mut host = crate::device::host_memory::HostMemory::new();
    let result = array.step_dma(2, 3, &mut host);
    assert!(result.is_none(), "step_dma must return None when DMA module is clock-gated, got {:?}", result);
}

/// When the StreamSwitch module is clock-gated, `step_tile_switches` must not
/// forward any words for that tile.  We seed a word on a slave port and verify
/// it is NOT forwarded to the master when SS is gated.
#[test]
fn module_gate_skips_ss_when_ss_module_gated() {
    use crate::device::clock_control::ModuleKind;

    let mut array = TileArray::npu1();

    // Ungate column 1.
    array.clock_mut().write_register(1, 0, 0x000FFF20, 0x1);

    // Gate the stream switch on compute tile (col=1, row=2): Core+DMA active, SS gated.
    // MCC bit 2 = Core, bit 1 = Memory/DMA, bit 0 = SS.  Set bits 2+1, clear bit 0.
    array.clock_mut().write_register(1, 2, 0x00060000, (1 << 2) | (1 << 1));

    assert!(!array.clock().is_module_active(1, 2, ModuleKind::StreamSwitch), "SS must be gated");

    // Seed a word on the core slave port (port 0) of that tile.
    let idx = array.tile_index(1, 2);
    array.tiles[idx]
        .stream_switch
        .slaves
        .first_mut()
        .expect("compute tile must have at least one slave port")
        .push(0xDEAD_BEEF);

    // Configure a local route from slave[0] -> master[0] so words would normally forward.
    array.tiles[idx].stream_switch.configure_local_route(0, 0);

    // Step tile switches: the SS-gated tile must not forward the seeded word.
    let forwarded = array.step_tile_switches();
    assert_eq!(forwarded, 0, "SS-gated tile must not forward any words");

    // The word must still be in the slave FIFO.
    assert!(
        !array.tiles[idx].stream_switch.slaves[0].fifo.is_empty(),
        "word must remain in slave FIFO when SS is gated"
    );
}

// === Cascade Routing Tests ===

#[test]
fn test_cascade_route_south() {
    // NPU1: 5 cols x 6 rows. Compute tiles at rows 2-5.
    let mut array = TileArray::npu1();
    array.clock_mut().ungate_all();

    // Configure tile (1,3): output direction = South (0)
    // Configure tile (1,2): input direction = North (0)
    // cascade_input_dir: bit 0 (0=North, 1=West)
    // cascade_output_dir: bit 1 (0=South, 1=East)
    array.tile_mut(1, 3).cascade_input_dir = 0; // North
    array.tile_mut(1, 3).cascade_output_dir = 0; // South
    array.tile_mut(1, 2).cascade_input_dir = 0; // North
    array.tile_mut(1, 2).cascade_output_dir = 0; // South

    // Push cascade data to tile (1,3) output
    let data = [0xAAu64, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11, 0x22];
    array.tile_mut(1, 3).push_cascade_output(data);

    // Route cascade
    array.route_cascade();

    // Data should arrive at tile (1,2) input
    assert!(array.tile(1, 2).has_cascade_input());
    assert!(!array.tile(1, 3).has_cascade_output());

    let received = array.tile_mut(1, 2).pop_cascade_input().unwrap();
    assert_eq!(received, data);
}

#[test]
fn test_cascade_route_east() {
    let mut array = TileArray::npu1();
    array.clock_mut().ungate_all();

    // Configure tile (1,3): output direction = East (1)
    array.tile_mut(1, 3).cascade_input_dir = 0; // North
    array.tile_mut(1, 3).cascade_output_dir = 1; // East
                                                 // Configure tile (2,3): input direction = West (1)
    array.tile_mut(2, 3).cascade_input_dir = 1; // West
    array.tile_mut(2, 3).cascade_output_dir = 0; // South

    let data = [1u64, 2, 3, 4, 5, 6, 7, 8];
    array.tile_mut(1, 3).push_cascade_output(data);

    array.route_cascade();

    assert!(array.tile(2, 3).has_cascade_input());
    assert!(!array.tile(1, 3).has_cascade_output());

    let received = array.tile_mut(2, 3).pop_cascade_input().unwrap();
    assert_eq!(received, data);
}

#[test]
fn test_cascade_backpressure() {
    let mut array = TileArray::npu1();

    // Configure South cascade path: (1,3) -> (1,2)
    array.tile_mut(1, 3).cascade_input_dir = 0;
    array.tile_mut(1, 3).cascade_output_dir = 0;
    array.tile_mut(1, 2).cascade_input_dir = 0;
    array.tile_mut(1, 2).cascade_output_dir = 0;

    // Fill destination's input FIFO (depth 4)
    for _ in 0..4 {
        array.tile_mut(1, 2).push_cascade_input([0; xdna_archspec::aie2::CASCADE_WORDS]);
    }

    // Push data to source
    let data = [42u64; xdna_archspec::aie2::CASCADE_WORDS];
    array.tile_mut(1, 3).push_cascade_output(data);

    // Route should NOT transfer (backpressure)
    array.route_cascade();

    // Source still has data, destination still has old data
    assert!(array.tile(1, 3).has_cascade_output());
    assert!(array.tile(1, 2).has_cascade_input());
}

#[test]
fn test_cascade_direction_mismatch_no_route() {
    let mut array = TileArray::npu1();

    // Source outputs South, but destination expects West (mismatch)
    array.tile_mut(1, 3).cascade_input_dir = 0;
    array.tile_mut(1, 3).cascade_output_dir = 0; // South
    array.tile_mut(1, 2).cascade_input_dir = 1; // West (wrong!)
    array.tile_mut(1, 2).cascade_output_dir = 0;

    let data = [99u64; xdna_archspec::aie2::CASCADE_WORDS];
    array.tile_mut(1, 3).push_cascade_output(data);

    array.route_cascade();

    // No transfer because directions don't match
    assert!(array.tile(1, 3).has_cascade_output());
    assert!(!array.tile(1, 2).has_cascade_input());
}

/// OP_READ response injection: handle_read_registers queues response
/// words, drain_ctrl_responses pushes them into the TileCtrl slave
/// port's FIFO across multiple drain cycles.
#[test]
fn test_handle_read_registers_injects_response() {
    use crate::device::stream_switch::{PacketHeader, PortType};

    let mut array = TileArray::npu1();

    // Use compute tile at (2,3)
    let col: u8 = 2;
    let row: u8 = 3;

    // Pre-populate 4 consecutive data memory locations with known values.
    // Uses write_data_u32 to match how cores actually write data memory.
    let base_offset: u32 = 0x440;
    let values = [0xAAAA_0001u32, 0xBBBB_0002, 0xCCCC_0003, 0xDDDD_0004];
    for (i, &val) in values.iter().enumerate() {
        array
            .tile_mut(col, row)
            .write_data_u32((base_offset + (i as u32) * 4) as usize, val);
    }

    // Verify the TileCtrl slave port exists (port 3 on compute tiles)
    let ctrl_slave_idx = array
        .tile(col, row)
        .stream_switch
        .tile_ctrl_slave_port()
        .expect("compute tile should have TileCtrl slave port");
    assert!(
        matches!(
            array.tile(col, row).stream_switch.slave(ctrl_slave_idx).unwrap().port_type,
            PortType::TileCtrl,
        ),
        "slave port at ctrl index should be TileCtrl"
    );

    // Call handle_read_registers with response_id=5, count=4
    let response_id: u8 = 5;
    let count: u8 = 4;
    let ok = array.handle_read_registers(col, row, base_offset, count, response_id);
    assert!(ok, "handle_read_registers should succeed");

    // Response words are queued, not yet in the FIFO
    assert_eq!(
        array.tile(col, row).pending_ctrl_response.len(),
        5,
        "pending buffer should have header + 4 data words"
    );

    // Drain into FIFO. With AM020-spec FIFO depth (4 x 32-bit words on
    // local slave ports), only 4 of the 5 response words fit in one
    // drain pass; the 5th stays pending until the FIFO drains.
    let injected1 = array.drain_ctrl_responses();
    assert_eq!(injected1, 4, "first drain fills the 4-deep slave FIFO");
    assert_eq!(
        array.tile(col, row).pending_ctrl_response.len(),
        1,
        "one word left pending after first drain"
    );

    // Pop and verify the stream header
    let slave = array.tile_mut(col, row).stream_switch.slave_mut(ctrl_slave_idx).unwrap();
    let (header_word, header_tlast) = slave.pop_with_tlast().unwrap();
    assert!(!header_tlast, "header should not have TLAST");

    let (decoded, parity_ok) = PacketHeader::decode(header_word);
    assert!(parity_ok, "packet header parity should be valid");
    assert_eq!(decoded.stream_id, response_id, "stream_id should equal response_id");
    assert_eq!(decoded.src_col, col, "src_col should match tile column");
    assert_eq!(decoded.src_row, row, "src_row should match tile row");

    // Pop the first three data words (no TLAST yet -- the 4th data word
    // is still in pending_ctrl_response, waiting on FIFO space).
    for (i, &expected) in values.iter().take(3).enumerate() {
        let slave = array.tile_mut(col, row).stream_switch.slave_mut(ctrl_slave_idx).unwrap();
        let (data, tlast) = slave.pop_with_tlast().unwrap();
        assert_eq!(data, expected, "data word {} should match register value", i);
        assert!(!tlast, "TLAST should not appear before the final data word");
    }

    // FIFO is empty; pending still has the last word. Re-drain to inject it.
    let injected2 = array.drain_ctrl_responses();
    assert_eq!(injected2, 1, "second drain injects the final data word");
    assert!(
        array.tile(col, row).pending_ctrl_response.is_empty(),
        "pending buffer should be empty after second drain"
    );

    // Pop the final data word -- carries TLAST.
    let slave = array.tile_mut(col, row).stream_switch.slave_mut(ctrl_slave_idx).unwrap();
    let (data, tlast) = slave.pop_with_tlast().unwrap();
    assert_eq!(data, values[3], "final data word should match register value");
    assert!(tlast, "TLAST must accompany the final data word");

    // FIFO should be empty now
    let slave = array.tile(col, row).stream_switch.slave(ctrl_slave_idx).unwrap();
    assert_eq!(slave.fifo_level(), 0, "FIFO should be empty after draining");
}

/// OP_READ with count=1: queue + drain produces header + 1 data word,
/// where the data word has TLAST set.
#[test]
fn test_handle_read_registers_single_word() {
    let mut array = TileArray::npu1();

    let col: u8 = 1;
    let row: u8 = 2;
    let offset: u32 = 0x500;
    let expected: u32 = 0x1234_5678;
    array.tile_mut(col, row).write_data_u32(offset as usize, expected);

    let ok = array.handle_read_registers(col, row, offset, 1, 0);
    assert!(ok);

    // Drain response into FIFO
    let injected = array.drain_ctrl_responses();
    assert_eq!(injected, 2, "header + 1 data word should fit in one drain");

    let ctrl_idx = array.tile(col, row).stream_switch.tile_ctrl_slave_port().unwrap();
    let slave = array.tile(col, row).stream_switch.slave(ctrl_idx).unwrap();
    assert_eq!(slave.fifo_level(), 2, "header + 1 data word");

    // Pop header (no TLAST)
    let slave = array.tile_mut(col, row).stream_switch.slave_mut(ctrl_idx).unwrap();
    let (_, h_tlast) = slave.pop_with_tlast().unwrap();
    assert!(!h_tlast, "header should not have TLAST when count > 0");

    // Pop data word (with TLAST)
    let (data, d_tlast) = slave.pop_with_tlast().unwrap();
    assert_eq!(data, expected);
    assert!(d_tlast, "single data word should have TLAST");
}

/// OP_READ for a non-existent tile should fail gracefully.
#[test]
fn test_handle_read_registers_bad_tile() {
    let mut array = TileArray::npu1();
    let ok = array.handle_read_registers(99, 99, 0x440, 4, 0);
    assert!(!ok, "should fail for out-of-bounds tile");
}

/// Parity handler errors (Second_Header_Parity) must latch the sticky
/// `pkt_handler_status` bit and continue -- NOT push `CtrlPacketAction::Error`
/// or populate `fatal_errors`. This exercises the full routing path through
/// `route_tile_switches_to_ctrl` to prove the fix, not a hand-set-bit.
///
/// Hardware ref: aie-rt xaiegbl_params.h:7761, AM025
/// `Tile_Control_Packet_Handler_Status` -- poll-only sticky, no interrupt/abort.
#[test]
fn parity_handler_error_is_sticky_continue_not_fatal() {
    use crate::device::control_packets::status::PktHandlerError;
    use crate::device::stream_switch::PortType;
    use crate::device::tile::CtrlPacketAction;
    use crate::device::host_memory::HostMemory;

    let mut array = TileArray::npu1();
    let col: u8 = 1;
    let row: u8 = 2; // compute tile

    // Build a header word with EVEN popcount so odd_parity_ok() returns false.
    // 0x00000101 = bits[8] + bits[0] = 2 set bits (even) -> SecondHeaderParity.
    // This matches build_test_header(0x101, 0, 0, 0) used in reassembler unit tests.
    let bad_parity_word: u32 = 0x0000_0101;
    assert_eq!(bad_parity_word.count_ones() % 2, 0, "word must have even popcount for parity failure");

    // Find the TileCtrl master port index (same search the routing fn uses).
    let tile_idx = array.tile_index(col, row);
    let ctrl_master_idx = array.tiles[tile_idx]
        .stream_switch
        .masters
        .iter()
        .position(|p| matches!(p.port_type, PortType::TileCtrl))
        .expect("compute tile must have a TileCtrl master port");

    // Push the parity-bad word directly onto the TileCtrl master port
    // with TLAST=false (a standalone header, no data beats expected).
    let pushed =
        array.tiles[tile_idx].stream_switch.masters[ctrl_master_idx].push_with_tlast(bad_parity_word, false);
    assert!(pushed, "test setup: TileCtrl master port must accept the word");

    // Run step_data_movement, which calls route_tile_switches_to_ctrl
    // internally and feeds the word to the tile's ctrl reassembler.
    let mut host_memory = HostMemory::new();
    array.step_data_movement(&mut host_memory);

    // Assert 1: SecondHeaderParity bit (0x2) latched in pkt_handler_status.
    let handler_status = array.tiles[tile_idx].pkt_handler_status;
    assert_ne!(
        handler_status & PktHandlerError::SecondHeaderParity.bit(),
        0,
        "SecondHeaderParity bit (0x2) must be latched in pkt_handler_status after a bad-parity header"
    );

    // Assert 2: No CtrlPacketAction::Error produced for the handler error.
    // (The structural-rejection Error arm is NOT triggered here -- this is a
    // parity error, not an opcode/length error, and must be sticky-continue.)
    let actions = array.drain_ctrl_packet_actions();
    let error_actions: Vec<_> = actions.iter().filter(|a| matches!(a, CtrlPacketAction::Error(_))).collect();
    assert!(
        error_actions.is_empty(),
        "HandlerError must NOT produce CtrlPacketAction::Error (sticky-continue); got: {:?}",
        error_actions
    );

    // Assert 3: fatal_errors is empty -- the engine would NOT abort.
    assert!(
        array.fatal_errors.is_empty(),
        "fatal_errors must be empty after a handler error (sticky-continue); got: {:?}",
        array.fatal_errors
    );
}

#[test]
fn has_pending_control_packet_is_control_specific() {
    // The hazard detector must fire ONLY on control-register-write packets being
    // delivered (TileCtrl master / reassembler / pending action), NOT on other
    // packet-switched traffic. Trace packets are also packet-switched (trace
    // unit -> shim DDR) but never touch the control delivery path; a broad
    // packet signal fired ~41x per traced tenant-4 run purely on trace traffic.
    let mut array = TileArray::npu1();
    assert!(!array.has_pending_control_packet(), "fresh array: no control packet pending");

    let col = 1u8;
    let row = 2u8;
    let tile_idx = array.tile_index(col, row);

    // A packet-switched word on a slave INPUT port (not the control delivery
    // path, which is the TileCtrl MASTER + reassembler) must NOT trip the
    // control-specific detector -- this stands in for trace traffic.
    {
        let ss = &mut array.tiles[tile_idx].stream_switch;
        let slave = ss.tile_ctrl_slave_port().expect("ctrl slave exists");
        ss.slave_mut(slave).unwrap().packet_enable = true;
        ss.configure_slave_slot(slave, 0, 0x0000_0100);
        ss.slave_mut(slave).unwrap().push(0x8000_0000);
    }
    assert!(
        !array.has_pending_control_packet(),
        "a packet not on the control delivery path (trace-like) must not trip the detector"
    );

    // A control word waiting at the TileCtrl MASTER port IS a pending control
    // packet -- this is the destination delivery point the reassembler drains.
    {
        let ss = &mut array.tiles[tile_idx].stream_switch;
        let ctrl_master = ss
            .masters
            .iter()
            .position(|p| matches!(p.port_type, crate::device::stream_switch::PortType::TileCtrl))
            .expect("compute tile must have a TileCtrl master port");
        ss.masters[ctrl_master].push(0x8000_0000);
    }
    assert!(
        array.has_pending_control_packet(),
        "a control word at the TileCtrl master port must register as a pending control packet"
    );
}

#[test]
fn tile_array_exposes_clock_controller_with_silicon_accurate_default() {
    // TileArray::npu1() is the existing test helper.
    let array = TileArray::npu1();
    // Default state: silicon-accurate, all columns gated.
    let cols = array.cols();
    for col in 0..cols {
        assert!(
            !array.clock().is_column_active(col),
            "col {} should be gated at TileArray construction",
            col
        );
    }
}

#[test]
fn step_data_movement_skips_gated_columns_no_progress() {
    use crate::device::host_memory::HostMemory;
    let mut array = TileArray::npu1();
    let mut host = HostMemory::new();
    // All columns gated by default (silicon-accurate).
    // step_data_movement should report no DMA active and no streams active.
    let (dma_active, streams_active, words_routed) = array.step_data_movement(&mut host);
    assert!(!dma_active, "no DMA activity when all columns gated");
    assert!(!streams_active, "no stream activity when all columns gated");
    assert_eq!(words_routed, 0, "no words routed when all columns gated");
}

#[test]
fn step_data_movement_runs_ungated_columns() {
    use crate::device::host_memory::HostMemory;
    let mut array = TileArray::npu1();
    let mut host = HostMemory::new();
    // Ungate column 2. With no BDs programmed, still no DMA activity, but
    // the function must traverse col 2 without panicking.
    array.clock_mut().write_register(2, 0, 0x000F_FF20, 0x1);
    assert!(array.clock().is_column_active(2), "col 2 should be active after write");
    // Must not panic; exact return values don't matter with empty BDs.
    let _ = array.step_data_movement(&mut host);
}
