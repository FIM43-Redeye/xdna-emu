//! Tests for the DMA transfer module.

use super::*;
use crate::device::dma::addressing::{AddressGenerator, ZeroPadConfig};
use crate::device::dma::{BdConfig, DmaError};
use xdna_archspec::types::TileKind;

fn simple_bd() -> BdConfig {
    BdConfig { base_addr: 0x1000, length: 256, valid: true, ..Default::default() }
}

#[test]
fn test_transfer_creation() {
    let bd = simple_bd();
    let transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileKind::Compute).unwrap();

    assert_eq!(transfer.bd_index, 0);
    assert_eq!(transfer.channel, 0);
    assert_eq!(transfer.total_bytes, 256);
    assert_eq!(transfer.remaining_bytes(), 256);
}

#[test]
fn test_transfer_invalid_bd() {
    let mut bd = simple_bd();
    bd.valid = false;

    let result = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileKind::Compute);
    assert!(matches!(result, Err(DmaError::BdNotValid(0))));
}

#[test]
fn test_transfer_with_acquire_lock() {
    let mut bd = simple_bd();
    bd.acquire_lock = Some(5);
    bd.acquire_value = 1;

    let transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileKind::Compute).unwrap();
    assert_eq!(transfer.acquire_lock, Some(5));
    assert_eq!(transfer.acquire_value, 1);
    assert_eq!(transfer.acquire_mode(), Some(LockAcquireMode::Equal(1)));
}

#[test]
fn test_transfer_advance_to_completion() {
    let mut bd = simple_bd();
    bd.acquire_lock = Some(5);
    bd.release_lock = Some(5);

    let mut transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileKind::Compute).unwrap();

    // Lock config is stored on the transfer
    assert_eq!(transfer.acquire_lock, Some(5));
    assert_eq!(transfer.release_lock, Some(5));

    // Advance all data
    transfer.advance(256);
    assert_eq!(transfer.remaining_bytes(), 0);
    assert_eq!(transfer.bytes_transferred, 256);
}

#[test]
fn test_transfer_progress() {
    let bd = simple_bd();
    let mut transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileKind::Compute).unwrap();

    assert_eq!(transfer.progress(), 0.0);

    transfer.advance(128);
    assert_eq!(transfer.progress(), 0.5);

    transfer.advance(128);
    assert_eq!(transfer.progress(), 1.0);
}

#[test]
fn test_transfer_remaining() {
    let bd = simple_bd();
    let mut transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileKind::Compute).unwrap();

    assert_eq!(transfer.remaining_bytes(), 256);

    transfer.advance(100);
    assert_eq!(transfer.remaining_bytes(), 156);
}

#[test]
fn test_transfer_direction_s2mm() {
    let bd = simple_bd();
    let transfer = Transfer::new(&bd, 0, 0, TransferDirection::S2MM, 1, 2, TileKind::Compute).unwrap();

    assert!(matches!(transfer.source, TransferEndpoint::Stream { .. }));
    assert!(matches!(transfer.dest, TransferEndpoint::TileMemory { .. }));
}

#[test]
fn test_transfer_direction_mm2s() {
    let bd = simple_bd();
    let transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileKind::Compute).unwrap();

    assert!(matches!(transfer.source, TransferEndpoint::TileMemory { .. }));
    assert!(matches!(transfer.dest, TransferEndpoint::Stream { .. }));
}

#[test]
fn test_host_to_tile_transfer() {
    let transfer = Transfer::new_host_to_tile(1, 2, 0x8000_0000, 0x1000, 512);

    assert!(matches!(transfer.source, TransferEndpoint::HostMemory));
    assert!(matches!(transfer.dest, TransferEndpoint::TileMemory { col: 1, row: 2 }));
    assert_eq!(transfer.total_bytes, 512);
}

#[test]
fn test_tile_to_host_transfer() {
    let transfer = Transfer::new_tile_to_host(1, 2, 0x1000, 0x8000_0000, 512);

    assert!(matches!(transfer.source, TransferEndpoint::TileMemory { col: 1, row: 2 }));
    assert!(matches!(transfer.dest, TransferEndpoint::HostMemory));
    assert_eq!(transfer.total_bytes, 512);
}

#[test]
fn test_transfer_error() {
    let bd = simple_bd();
    let mut transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileKind::Compute).unwrap();

    transfer.set_error(DmaError::AddressOutOfBounds { address: 0xFFFF, limit: 0x1000 });

    assert!(transfer.error.is_some());
    assert!(matches!(transfer.error, Some(DmaError::AddressOutOfBounds { .. })));
}

#[test]
fn test_next_bd_chaining() {
    let bd = BdConfig::simple_1d(0x1000, 256).with_next(3);
    let transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileKind::Compute).unwrap();

    assert_eq!(transfer.next_bd, Some(3));
}

#[test]
fn test_packet_header_disabled_by_default() {
    let bd = simple_bd();
    let transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileKind::Compute).unwrap();

    assert!(!transfer.enable_packet);
    assert!(!transfer.needs_packet_header());
    assert!(transfer.generate_packet_header().is_none());
}

#[test]
fn test_packet_header_generation() {
    let mut bd = simple_bd();
    bd.enable_packet = true;
    bd.packet_id = 0x1F; // 5-bit value, max
    bd.packet_type = 0x3; // 3-bit value (Trace)

    // Create transfer at tile (3, 5)
    let transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 3, 5, TileKind::Compute).unwrap();

    assert!(transfer.needs_packet_header());

    let header = transfer.generate_packet_header().unwrap();

    // Verify header format per AM020 Ch2, Table 2:
    // | 31    | 30-28 | 27-21      | 20-16     | 15  | 14-12       | 11-5    | 4-0       |
    // | Parity| Rsvd  | Src Column | Src Row   | Rsvd| Packet Type | Rsvd    | Stream ID |

    // Check source column (bits 27:21, 7-bit field)
    let col = (header >> 21) & 0x7F;
    assert_eq!(col, 3, "Source column should be 3");

    // Check source row (bits 20:16, 5-bit field)
    let row = (header >> 16) & 0x1F;
    assert_eq!(row, 5, "Source row should be 5");

    // Check packet type (bits 14:12, 3-bit field)
    let pkt_type = (header >> 12) & 0x7;
    assert_eq!(pkt_type, 3, "Packet type should be 3 (Trace)");

    // Check stream ID (bits 4:0, 5-bit field)
    let stream_id = header & 0x1F;
    assert_eq!(stream_id, 0x1F, "Stream ID should be 0x1F");

    // Verify odd parity
    let ones = header.count_ones();
    assert_eq!(ones % 2, 1, "Parity should be odd, got {} ones", ones);

    // Verify round-trip via PacketHeader::decode
    let (decoded, parity_ok) = crate::device::stream_switch::PacketHeader::decode(header);
    assert!(parity_ok, "Parity check should pass");
    assert_eq!(decoded.stream_id, 0x1F);
    assert_eq!(decoded.src_col, 3);
    assert_eq!(decoded.src_row, 5);

    // Verify parse helper functions
    assert_eq!(parse_stream_id_from_header(header), 0x1F);
    assert_eq!(parse_packet_type_from_header(header), 3);
    assert_eq!(parse_source_tile_from_header(header), (3, 5));
}

#[test]
fn test_packet_header_sent_tracking() {
    let mut bd = simple_bd();
    bd.enable_packet = true;
    bd.packet_id = 0x10;

    let mut transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileKind::Compute).unwrap();

    // Initially needs header
    assert!(transfer.needs_packet_header());
    assert!(!transfer.packet_header_sent);

    // Mark as sent
    transfer.mark_packet_header_sent();

    // No longer needs header
    assert!(!transfer.needs_packet_header());
    assert!(transfer.packet_header_sent);
}

#[test]
fn test_lock_acquire_mode_conversion() {
    // Positive values -> Equal mode
    assert_eq!(LockAcquireMode::from_bd_value(1), LockAcquireMode::Equal(1));
    assert_eq!(LockAcquireMode::from_bd_value(5), LockAcquireMode::Equal(5));

    // Negative values -> GreaterEqual mode
    assert_eq!(LockAcquireMode::from_bd_value(-1), LockAcquireMode::GreaterEqual(1));
    assert_eq!(LockAcquireMode::from_bd_value(-5), LockAcquireMode::GreaterEqual(5));

    // Zero -> Simple mode
    assert_eq!(LockAcquireMode::from_bd_value(0), LockAcquireMode::Simple);

    // Round-trip conversion
    assert_eq!(LockAcquireMode::Equal(3).to_bd_value(), 3);
    assert_eq!(LockAcquireMode::GreaterEqual(3).to_bd_value(), -3);
    assert_eq!(LockAcquireMode::Simple.to_bd_value(), 0);

    // Threshold values
    assert_eq!(LockAcquireMode::Equal(5).threshold(), 5);
    assert_eq!(LockAcquireMode::GreaterEqual(7).threshold(), 7);
    assert_eq!(LockAcquireMode::Simple.threshold(), 1);
}

#[test]
fn test_transfer_acquire_mode() {
    // Transfer without lock
    let bd = simple_bd();
    let transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileKind::Compute).unwrap();
    assert!(transfer.acquire_mode().is_none());

    // Transfer with lock in Equal mode
    let mut bd_locked = simple_bd();
    bd_locked.acquire_lock = Some(5);
    bd_locked.acquire_value = 1;
    let transfer_locked =
        Transfer::new(&bd_locked, 0, 0, TransferDirection::MM2S, 1, 2, TileKind::Compute).unwrap();
    assert_eq!(transfer_locked.acquire_mode(), Some(LockAcquireMode::Equal(1)));

    // Transfer with lock in GE mode
    let mut bd_ge = simple_bd();
    bd_ge.acquire_lock = Some(5);
    bd_ge.acquire_value = -2;
    let transfer_ge = Transfer::new(&bd_ge, 0, 0, TransferDirection::MM2S, 1, 2, TileKind::Compute).unwrap();
    assert_eq!(transfer_ge.acquire_mode(), Some(LockAcquireMode::GreaterEqual(2)));
}

// --- ZeroPadState tests ---

#[test]
fn test_pad_state_no_padding() {
    // No padding configured: all output should be data
    let config = ZeroPadConfig::default();
    let mut state = ZeroPadState::new(config, 4, 1, 1);

    // With no padding, total output = data words only
    assert_eq!(state.total_output_words(), 4);

    let gen = AddressGenerator::new_1d(0x1000, 4, 4);
    for _ in 0..4 {
        assert!(matches!(state.current_action(&gen), PadAction::Data(_)));
        state.advance();
    }
    assert!(state.is_finished());
}

#[test]
fn test_pad_state_d0_only() {
    // D0 before=2, after=1, d0_size=3, d1=1, d2=1
    // Expected output: [0,0, D,D,D, 0] = 6 words total
    let config = ZeroPadConfig { d0_before: 2, d0_after: 1, ..Default::default() };
    let mut state = ZeroPadState::new(config, 3, 1, 1);
    let gen = AddressGenerator::new_1d(0x1000, 3, 4);

    assert_eq!(state.total_output_words(), 6);

    // Collect output sequence
    let mut actions = Vec::new();
    while !state.is_finished() {
        actions.push(state.current_action(&gen));
        state.advance();
    }

    assert_eq!(actions.len(), 6);
    assert_eq!(actions[0], PadAction::Zero); // d0_before
    assert_eq!(actions[1], PadAction::Zero); // d0_before
    assert!(matches!(actions[2], PadAction::Data(_))); // data
    assert!(matches!(actions[3], PadAction::Data(_))); // data
    assert!(matches!(actions[4], PadAction::Data(_))); // data
    assert_eq!(actions[5], PadAction::Zero); // d0_after
}

#[test]
fn test_pad_state_d0_with_d1_iterations() {
    // D0 before=1, after=1, d0_size=2, d1_size=3, d2=1
    // Per D1 iteration: [0, D,D, 0] = 4 words
    // Total: 4 * 3 = 12 words
    let config = ZeroPadConfig { d0_before: 1, d0_after: 1, ..Default::default() };
    let mut state = ZeroPadState::new(config, 2, 3, 1);
    let gen = AddressGenerator::new_2d(0x1000, 2, 4, 3, 8);

    assert_eq!(state.total_output_words(), 12);

    let mut actions = Vec::new();
    while !state.is_finished() {
        actions.push(state.current_action(&gen));
        state.advance();
    }

    // Verify structure: 3 repetitions of [Zero, Data, Data, Zero]
    for iter in 0..3 {
        let base = iter * 4;
        assert_eq!(actions[base], PadAction::Zero, "d0_before at D1 iter {}", iter);
        assert!(matches!(actions[base + 1], PadAction::Data(_)), "data at D1 iter {}", iter);
        assert!(matches!(actions[base + 2], PadAction::Data(_)), "data at D1 iter {}", iter);
        assert_eq!(actions[base + 3], PadAction::Zero, "d0_after at D1 iter {}", iter);
    }
}

#[test]
fn test_pad_state_all_dimensions() {
    // D0 before=1, d0_size=2, D1 before=1, after=1, d1=2, D2 before=1, after=1, d2=1
    //
    // Per AM025:
    // - D0 padding: individual words
    // - D1 padding: "wraps of dim0" (complete D0 rows of zeros)
    // - D2 padding: "wraps of dim0dim1" (complete D1 blocks of zeros)
    //
    // d0_wrap = 1+2+0 = 3 words per D0 output row
    // d1_total = 1+2+1 = 4 D0 wraps per D1 block (including D1 padding)
    //
    // D2 before = 1 D1 block = 4 * 3 = 12 zero words
    // D1 iteration 0:
    //   D1 before = 1 D0 wrap = 3 zero words
    //   D0 data row: [0, D, D] = 3 words (1 d0_before zero + 2 data)
    //   D1 after = 1 D0 wrap = 3 zero words
    // D1 iteration 1: same as iter 0
    // D2 after = 1 D1 block = 12 zero words
    //
    // Total = 12 + (3+3+3)*2 + 12 = 12 + 18 + 12 = 42 words
    let config =
        ZeroPadConfig { d0_before: 1, d0_after: 0, d1_before: 1, d1_after: 1, d2_before: 1, d2_after: 1 };
    let mut state = ZeroPadState::new(config, 2, 2, 1);
    let gen = AddressGenerator::new_2d(0x1000, 2, 4, 2, 8);

    // total = (1+1+1)*(1+2+1)*(1+2+0) = 3*4*3 = 36; data = 2*2*1 = 4; pad = 36-4 = 32
    // Hmm, total output should be d2_total * d1_total * d0_wrap = 3 * 4 * 3 = 36
    assert_eq!(state.total_output_words(), 36);

    let mut sequence = Vec::new();
    while !state.is_finished() {
        let is_data = matches!(state.current_action(&gen), PadAction::Data(_));
        sequence.push(if is_data { 'D' } else { '0' });
        state.advance();
    }

    let pattern: String = sequence.into_iter().collect();
    // D2 before: 12 zeros
    // D1 iter 0: d1_before(3 zeros) + d0_before(1)+data(2) + d1_after(3 zeros) = 9 words
    // D1 iter 1: same = 9 words
    // D2 after: 12 zeros
    // Total: 12 + 9 + 9 + 12 = 42... but total_output says 36?
    //
    // Wait: total_output = d2_total * d1_total * d0_wrap = 3 * 4 * 3 = 36
    // D2 before: 1 D1 block = d1_total * d0_wrap = 4 * 3 = 12 zeros
    // Data D1 iterations: 2 * (d1_before_wraps + d0_row + d1_after_wraps)
    //   = 2 * (1*3 + 3 + 1*3) = 2 * 9 = 18 words (but only 4 are data)
    // D2 after: 12 zeros
    // Total: 12 + 18 + 12 = 42 -- but formula says 36!
    //
    // The discrepancy is because d1_total already includes d1_before+d1_after,
    // so the data D1 iterations should NOT add their own d1_before/d1_after again.
    // Actually d1_total = d1_before + d1_size + d1_after = 1+2+1 = 4
    // And d2_total = d2_before + d2_size + d2_after = 1+1+1 = 3
    // total = 3 * 4 * 3 = 36.
    // This means: 3 D2 blocks, each containing 4 D0 wraps, each of 3 words.
    // D2 block 0 (d2_before): all zeros = 4*3 = 12 zeros
    // D2 block 1 (data):
    //   D1 wrap 0 (d1_before): 3 zeros
    //   D1 wrap 1 (data iter 0): 1 zero + 2 data = 0DD
    //   D1 wrap 2 (data iter 1): 1 zero + 2 data = 0DD
    //   D1 wrap 3 (d1_after): 3 zeros
    // D2 block 2 (d2_after): all zeros = 12 zeros
    // Total: 12 + 3 + 3 + 3 + 3 + 12 = 36
    assert_eq!(
        pattern,
        "000000000000" // D2 before (12 zeros)
        .to_owned()
        + "000" // D1 before wrap (3 zeros)
        + "0DD" // data D1 iter 0
        + "0DD" // data D1 iter 1
        + "000" // D1 after wrap (3 zeros)
        + "000000000000"
    ); // D2 after (12 zeros)
}

#[test]
fn test_pad_state_advance_returns_is_data() {
    // Verify advance() returns true for data words, false for padding
    let config = ZeroPadConfig { d0_before: 1, d0_after: 1, ..Default::default() };
    let mut state = ZeroPadState::new(config, 2, 1, 1);

    // Expected: [Zero, Data, Data, Zero]
    assert!(!state.advance()); // d0_before zero
    assert!(state.advance()); // data
    assert!(state.advance()); // data
    assert!(!state.advance()); // d0_after zero
    assert!(state.is_finished());
}

#[test]
fn test_transfer_with_padding_total_bytes() {
    // Verify total_bytes for padded transfers.
    //
    // Per CDO lowering: Buffer_Length includes both data and padding words.
    // The padding state machine handles data/padding interleaving, so
    // total_bytes = Buffer_Length (no additional padding added on top).
    use crate::device::dma::DimensionConfig;

    // Simulate CDO-style BD: d0=5 data words, d0_pad 1+1, d1=2 rows
    // D0 wrap = 1+5+1 = 7 words per row
    // Total output = 2 * 7 = 14 words = 56 bytes
    // CDO sets Buffer_Length = 14 words (total output including padding)
    let bd = BdConfig {
        base_addr: 0x80000,
        length: 56, // 14 words * 4 bytes (CDO: data + padding)
        d0: DimensionConfig::new(5, 4),
        d1: DimensionConfig::new(2, 20),
        valid: true,
        zero_padding: ZeroPadConfig { d0_before: 1, d0_after: 1, ..Default::default() },
        ..Default::default()
    };

    // MemTile MM2S: padding active, total_bytes = Buffer_Length
    let transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 1, TileKind::Mem).unwrap();
    assert_eq!(transfer.total_bytes, 56);
    assert!(transfer.has_zero_padding());

    // Same BD on compute tile -- no padding applied (compute doesn't pad)
    let transfer_compute =
        Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileKind::Compute).unwrap();
    assert_eq!(transfer_compute.total_bytes, 56);
    assert!(!transfer_compute.has_zero_padding());

    // MemTile S2MM -- no padding (S2MM ignores padding config)
    let transfer_s2mm = Transfer::new(&bd, 0, 0, TransferDirection::S2MM, 1, 1, TileKind::Mem).unwrap();
    assert!(!transfer_s2mm.has_zero_padding());
}

/// Validates the CDO-path i8 padding scenario from add_21_i8 test.
///
/// In the CDO path, mlir-aie converts D0 padding from element counts to
/// 32-bit word counts before writing to BD registers. For i8 data with
/// pad_before=4 elements and pad_after=4 elements:
/// - d0_size = 8 elements * (8/32) = 2 words
/// - d0_zero_before = 4 elements * (8/32) = 1 word
/// - d0_zero_after = 4 elements * (8/32) = 1 word
/// - buffer_length = 16 elements * (8/32) = 4 words
///
/// Total output should be 1 + 2 + 1 = 4 words = 16 bytes.
#[test]
fn test_pad_state_i8_cdo_path() {
    // Simulates the BD register values from CDO-compiled
    // add_21_i8_using_dma_op_with_padding test:
    // memref<16xi8>, size=8, stride=1, pad_before=4, pad_after=4
    let config = ZeroPadConfig {
        d0_before: 1, // 4 i8 elements -> 1 word (CDO converts)
        d0_after: 1,  // 4 i8 elements -> 1 word (CDO converts)
        ..Default::default()
    };
    // d0_size = 2 words (8 i8 elements)
    let mut state = ZeroPadState::new(config, 2, 1, 1);
    let gen = AddressGenerator::new_1d(0x0, 2, 4);

    // Total: 1 pad + 2 data + 1 pad = 4 words
    assert_eq!(state.total_output_words(), 4);

    let mut actions = Vec::new();
    while !state.is_finished() {
        actions.push(state.current_action(&gen));
        state.advance();
    }

    assert_eq!(actions.len(), 4);
    assert_eq!(actions[0], PadAction::Zero); // 1 word of zeros (4 i8 zeros)
    assert!(matches!(actions[1], PadAction::Data(_)));
    assert!(matches!(actions[2], PadAction::Data(_)));
    assert_eq!(actions[3], PadAction::Zero); // 1 word of zeros (4 i8 zeros)
}

/// Validates transfer total_bytes for the add_378_i32 padding scenario.
///
/// BD config from CDO: 13 data words + 2 pad_before + 1 pad_after = 16 total.
/// Buffer_Length = 16 words = 64 bytes. The transfer must produce exactly
/// 16 words, not 19 (which was the bug: padding double-counted).
#[test]
fn test_transfer_padding_no_double_count() {
    use crate::device::dma::DimensionConfig;

    let bd = BdConfig {
        base_addr: 0x80000,
        length: 64,                      // 16 words * 4 bytes (CDO: total including padding)
        d0: DimensionConfig::new(13, 4), // 13 data words
        valid: true,
        zero_padding: ZeroPadConfig { d0_before: 2, d0_after: 1, ..Default::default() },
        ..Default::default()
    };

    let transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 1, TileKind::Mem).unwrap();

    // total_bytes must equal Buffer_Length (64), not Buffer_Length + pad (76)
    assert_eq!(transfer.total_bytes, 64);
    assert!(transfer.has_zero_padding());

    // ZeroPadState's total must match: 2 + 13 + 1 = 16 words
    let pad = transfer.zero_pad_state.as_ref().unwrap();
    assert_eq!(pad.total_output_words(), 16);
}
