//! Semantics: CdoRaw -> DeviceOp lowering.
//!
//! Stage 8b Half 2 of Subsystem 8: `lower` takes a parser-level
//! `CdoRaw` command and emits one or more arch-generic `DeviceOp`s that
//! `device::state::apply` consumes. This is the parser's last word on
//! device behavior -- after lowering, state sees only a typed op stream
//! that names tiles by `TileAddr` and register effects by kind.
//!
//! Most variants map 1:1 to a register-level op (`RegWrite`, `RegMask`,
//! `RegBurst`, `MaskPoll`, `Delay`, `Marker`). Two promotions recognize
//! well-known control registers and emit typed ops instead:
//!
//!   - A write to `CORE_CONTROL` on a compute tile becomes
//!     `CoreEnable` (direction carried by bit 0 of the value).
//!   - A write to a compute-tile DMA channel **Start_Queue** register
//!     becomes `DmaStart { channel, dir }`. Writes to the channel
//!     **Ctrl** register (Start_Queue - 4) are configuration only and
//!     pass through as `RegWrite` -- real hardware treats them the
//!     same way: Ctrl holds channel-level state, Start_Queue is the
//!     trigger that pushes a BD ID and enqueues the transfer.
//!
//! `CoreEnable` is gated to compute tiles (row >= 2 on AIE2) since shim
//! and memtile have no core. `DmaStart` promotion applies to all three
//! tile kinds: compute uses `Start_Queue` (base 0x1DE00, 2 channels per
//! direction), memtile uses `Start_Queue` (base 0xA0600, 6 channels per
//! direction), shim uses `Task_Queue` (base 0x1D200, 2 channels per
//! direction). All are at Ctrl + 4 within their respective channel
//! slots.
//!
//! Address decoding is inlined rather than routed through an
//! `ArchHandle`: AIE2 and AIE2P share the same 5-bit column / 5-bit row
//! split, and the parser currently has only one architecture target.
//! When additional arches come online the helper will move behind an
//! arch trait per the original Stage 8b design note.
//!
//! See docs/superpowers/specs/2026-04-23-subsys8-parser-design.md
//! §Stage 8b for the two-halves rationale.

use smallvec::SmallVec;

use xdna_archspec::aie2::registers::dma::{
    COMPUTE_DMA_MM2S_0_START_QUEUE, COMPUTE_DMA_MM2S_1_START_QUEUE,
    COMPUTE_DMA_S2MM_0_START_QUEUE, COMPUTE_DMA_S2MM_1_START_QUEUE,
    MEMTILE_CHANNELS_PER_DIR, MEMTILE_CHANNEL_STRIDE,
    MEMTILE_DMA_MM2S_0_START_QUEUE, MEMTILE_DMA_S2MM_0_START_QUEUE,
    SHIM_CHANNELS_PER_DIR, SHIM_CHANNEL_STRIDE,
    SHIM_DMA_MM2S_0_TASK_QUEUE, SHIM_DMA_S2MM_0_TASK_QUEUE,
};
use xdna_archspec::aie2::registers::CORE_CONTROL;
use xdna_archspec::aie2::{TILE_COL_SHIFT, TILE_OFFSET_MASK, TILE_ROW_SHIFT};
use xdna_archspec::types::{DmaDirection, TileAddr};

use crate::device::ops::DeviceOp;

use super::syntax::CdoRaw;

/// Decode an AIE2 host-view address into a tile coordinate and a 20-bit
/// tile-local offset.
///
/// AIE2/AIE2P layout: bits[29:25] = column, bits[24:20] = row,
/// bits[19:0] = tile-local offset. Same shape as `TILE_COL_SHIFT` /
/// `TILE_ROW_SHIFT` / `TILE_OFFSET_MASK` in `xdna_archspec::aie2`.
#[inline]
fn decode_aie2_address(addr: u32) -> (TileAddr, u32) {
    let col = ((addr >> TILE_COL_SHIFT) & 0x1F) as u8;
    let row = ((addr >> TILE_ROW_SHIFT) & 0x1F) as u8;
    let offset = addr & TILE_OFFSET_MASK;
    (TileAddr { col, row }, offset)
}

/// Is this tile a compute tile on AIE2?
///
/// Row 0 is shim, row 1 is memtile, rows >= 2 are compute. This is the
/// layout for all current AIE2 parts (NPU1/Phoenix). AIE2P keeps the
/// same convention. Used to gate `CoreEnable` (shim and memtile have
/// no core).
#[inline]
fn is_compute_row(row: u8) -> bool {
    row >= 2
}

/// Is this tile a memtile on AIE2 (row 1)?
#[inline]
fn is_mem_row(row: u8) -> bool {
    row == 1
}

/// Is this tile the shim row on AIE2 (row 0)?
#[inline]
fn is_shim_row(row: u8) -> bool {
    row == 0
}

/// Match a tile-local DMA Start_Queue (or shim Task_Queue) offset for the
/// given tile row, returning `(channel, direction)` if it names one. A
/// write to that register is the hardware trigger for a DMA transfer;
/// Ctrl writes (offset - 4) are configuration and pass through as
/// `RegWrite`.
///
/// Compute tiles have 2 channels per direction (offsets 0x1DE04 / 0x1DE0C
/// for S2MM_0/1, 0x1DE14 / 0x1DE1C for MM2S_0/1). Memtiles have 6
/// channels per direction starting at 0xA0604 (S2MM) and 0xA0634 (MM2S),
/// stride 8. Shim has 2 channels per direction starting at 0x1D204
/// (S2MM) and 0x1D214 (MM2S), stride 8 (the register is named
/// `Task_Queue` on shim but is the same trigger semantically).
#[inline]
fn match_dma_start_queue(row: u8, offset: u32) -> Option<(u8, DmaDirection)> {
    if is_compute_row(row) {
        return match offset {
            o if o == COMPUTE_DMA_S2MM_0_START_QUEUE => Some((0, DmaDirection::S2mm)),
            o if o == COMPUTE_DMA_S2MM_1_START_QUEUE => Some((1, DmaDirection::S2mm)),
            o if o == COMPUTE_DMA_MM2S_0_START_QUEUE => Some((0, DmaDirection::Mm2s)),
            o if o == COMPUTE_DMA_MM2S_1_START_QUEUE => Some((1, DmaDirection::Mm2s)),
            _ => None,
        };
    }
    if is_mem_row(row) {
        return match_strided_start_queue(
            offset,
            MEMTILE_DMA_S2MM_0_START_QUEUE,
            MEMTILE_DMA_MM2S_0_START_QUEUE,
            MEMTILE_CHANNELS_PER_DIR,
            MEMTILE_CHANNEL_STRIDE,
        );
    }
    if is_shim_row(row) {
        return match_strided_start_queue(
            offset,
            SHIM_DMA_S2MM_0_TASK_QUEUE,
            SHIM_DMA_MM2S_0_TASK_QUEUE,
            SHIM_CHANNELS_PER_DIR,
            SHIM_CHANNEL_STRIDE,
        );
    }
    None
}

/// Helper: match `offset` against a contiguous `[base, base + n*stride)`
/// channel-trigger range for one direction. Falls through to the next
/// direction when the first doesn't match.
#[inline]
fn match_strided_start_queue(
    offset: u32,
    s2mm_base: u32,
    mm2s_base: u32,
    channels_per_dir: u8,
    stride: u32,
) -> Option<(u8, DmaDirection)> {
    let n = channels_per_dir as u32;
    if offset >= s2mm_base && offset < s2mm_base + n * stride {
        let delta = offset - s2mm_base;
        if delta % stride == 0 {
            return Some(((delta / stride) as u8, DmaDirection::S2mm));
        }
    }
    if offset >= mm2s_base && offset < mm2s_base + n * stride {
        let delta = offset - mm2s_base;
        if delta % stride == 0 {
            return Some(((delta / stride) as u8, DmaDirection::Mm2s));
        }
    }
    None
}

/// Lower a `CdoRaw` command into zero or more `DeviceOp`s.
///
/// Returns a `SmallVec` with inline capacity 4 -- every current variant
/// produces either zero or one `DeviceOp`, so inline storage is never
/// spilled in practice. The 4-slot budget leaves headroom for future
/// lowerings that fan out (e.g., a single `DmaWrite` that becomes a
/// `RegBurst` + a trailing `DmaStart`).
pub fn lower(raw: &CdoRaw) -> SmallVec<[DeviceOp; 4]> {
    let mut out: SmallVec<[DeviceOp; 4]> = SmallVec::new();

    match raw {
        CdoRaw::Write { address, value } => {
            out.push(lower_write(*address, *value));
        }
        CdoRaw::MaskWrite { address, mask, value } => {
            out.push(lower_mask_write(*address, *mask, *value));
        }
        CdoRaw::Write64 { address, value } => {
            // High 32 of the address is always zero for AIE tile
            // addresses; the CDO "64" width is on address, not value.
            out.push(lower_write(*address as u32, *value));
        }
        CdoRaw::MaskWrite64 { address, mask, value } => {
            out.push(lower_mask_write(*address as u32, *mask, *value));
        }
        CdoRaw::DmaWrite { address, data } => {
            let (tile, offset) = decode_aie2_address(*address);
            let words = bytes_to_words_le(data);
            out.push(DeviceOp::RegBurst { tile, offset, words });
        }
        CdoRaw::MaskPoll { address, mask, expected } => {
            let (tile, offset) = decode_aie2_address(*address);
            out.push(DeviceOp::MaskPoll {
                tile,
                offset,
                mask: *mask,
                expected: *expected,
            });
        }
        CdoRaw::MaskPoll64 { address, mask, expected } => {
            let (tile, offset) = decode_aie2_address(*address as u32);
            out.push(DeviceOp::MaskPoll {
                tile,
                offset,
                mask: *mask,
                expected: *expected,
            });
        }
        CdoRaw::Delay { cycles } => {
            out.push(DeviceOp::Delay { cycles: *cycles });
        }
        CdoRaw::Marker { value } => {
            out.push(DeviceOp::Marker { value: *value });
        }
        // Nop is parser padding; EndMark is a stream boundary. Neither
        // produces a DeviceOp -- state has nothing to do for them.
        CdoRaw::Nop { .. } => {}
        CdoRaw::EndMark => {}
        // Unknown commands log at warn like the pre-refactor code did,
        // and produce no DeviceOp so state sees a silent drop.
        CdoRaw::Unknown { opcode, payload } => {
            log::warn!(
                "CDO lower: unknown opcode 0x{:03X} with {} payload words dropped",
                opcode,
                payload.len(),
            );
        }
    }

    out
}

/// Lower a `CdoRaw::Write` to either a promoted typed op or a
/// `RegWrite`. See the module-level doc comment for the promotion rules.
fn lower_write(address: u32, value: u32) -> DeviceOp {
    let (tile, offset) = decode_aie2_address(address);

    // Core enable/disable on a compute tile. The raw `value` rides along
    // so `device::state` can store it verbatim into `tile.core.control`
    // for readback (hardware stores the full 32-bit word, not just bit 0).
    if offset == CORE_CONTROL && is_compute_row(tile.row) {
        return DeviceOp::CoreEnable {
            tile,
            enabled: (value & 1) != 0,
            value,
        };
    }

    // DMA channel start on any tile (compute, memtile, or shim): any
    // write to the channel's Start_Queue / Task_Queue register pushes a
    // BD ID and starts a transfer. Ctrl writes are channel-level config
    // and are NOT promoted here -- they pass through as RegWrite exactly
    // like a write to any other config register.
    //
    // The raw `value` is carried as `bd_id`: `device::state` extracts
    // start_bd_id / repeat_count / enable_token_issue from it the same
    // way `write_dma_channel`'s Start_Queue branch did pre-refactor.
    if let Some((channel, dir)) = match_dma_start_queue(tile.row, offset) {
        return DeviceOp::DmaStart { tile, channel, dir, bd_id: value };
    }

    DeviceOp::RegWrite { tile, offset, value }
}

/// Lower a `CdoRaw::MaskWrite` to a `RegMask`.
fn lower_mask_write(address: u32, mask: u32, value: u32) -> DeviceOp {
    let (tile, offset) = decode_aie2_address(address);

    // MaskWrites are NOT promoted to typed ops. Mask-blending lives
    // on the apply side (`mask_write_register` -> module-specific
    // mask handler), so the typed handler would either duplicate
    // it or drop the mask. Letting MaskWrite ride as `RegMask`
    // keeps mask-blend correctness in one place. If we later want
    // mask-aware promotion, add `mask: Option<u32>` to
    // `DeviceOp::CoreEnable` (see D.3 spec, "room preserved for
    // option (a)").
    DeviceOp::RegMask { tile, offset, mask, value }
}

/// Convert a little-endian byte buffer into a word vector.
///
/// CDO `DmaWrite` payloads are always a whole number of 32-bit words
/// (they originate as `Vec<u32>` in the parser and are flattened via
/// `.to_le_bytes()`). If a trailing odd-byte tail is ever encountered
/// we log a warning and pad with zeros rather than silently truncate,
/// matching the defensive pattern the existing state layer uses.
fn bytes_to_words_le(data: &[u8]) -> SmallVec<[u32; 64]> {
    let full_words = data.len() / 4;
    let tail = data.len() % 4;
    let mut words: SmallVec<[u32; 64]> = SmallVec::with_capacity(full_words + (tail > 0) as usize);
    for chunk in data.chunks_exact(4) {
        words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    if tail != 0 {
        log::warn!(
            "CDO lower: DmaWrite payload not 4-byte aligned ({} bytes, {} extra); zero-padding",
            data.len(),
            tail,
        );
        let mut pad = [0u8; 4];
        let start = full_words * 4;
        pad[..tail].copy_from_slice(&data[start..]);
        words.push(u32::from_le_bytes(pad));
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    // The Ctrl consts aren't used by the production code (promotion
    // matches Start_Queue instead), but tests reference them to verify
    // Ctrl writes fall through to RegWrite.
    use xdna_archspec::aie2::registers::dma::COMPUTE_DMA_S2MM_0_CTRL;

    /// Build an AIE2 host-view address for a tile-local offset.
    fn aie_addr(col: u8, row: u8, offset: u32) -> u32 {
        ((col as u32) << TILE_COL_SHIFT) | ((row as u32) << TILE_ROW_SHIFT) | offset
    }

    fn compute_tile_addr() -> (u8, u8) {
        (1, 2)
    }

    // ------------------------------------------------------------------
    // Pass-through variants (1:1 with DeviceOp register-level variants)
    // ------------------------------------------------------------------

    #[test]
    fn write_non_special_offset_lowers_to_regwrite() {
        let (col, row) = compute_tile_addr();
        let addr = aie_addr(col, row, 0x1D000); // BD0 word 0, not a promotable offset
        let raw = CdoRaw::Write { address: addr, value: 0xDEAD_BEEF };
        let ops = lower(&raw);
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            DeviceOp::RegWrite { tile, offset, value } => {
                assert_eq!(*tile, TileAddr::new(col, row));
                assert_eq!(*offset, 0x1D000);
                assert_eq!(*value, 0xDEAD_BEEF);
            }
            _ => panic!("expected RegWrite, got {:?}", ops[0]),
        }
    }

    #[test]
    fn mask_write_non_special_offset_lowers_to_regmask() {
        let (col, row) = compute_tile_addr();
        let addr = aie_addr(col, row, 0x1D000);
        let raw = CdoRaw::MaskWrite {
            address: addr,
            mask: 0x0000_FFFF,
            value: 0x0000_1234,
        };
        let ops = lower(&raw);
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            DeviceOp::RegMask { tile, offset, mask, value } => {
                assert_eq!(*tile, TileAddr::new(col, row));
                assert_eq!(*offset, 0x1D000);
                assert_eq!(*mask, 0x0000_FFFF);
                assert_eq!(*value, 0x0000_1234);
            }
            _ => panic!("expected RegMask, got {:?}", ops[0]),
        }
    }

    #[test]
    fn write64_truncates_address_and_lowers_to_regwrite() {
        let (col, row) = compute_tile_addr();
        let lo = aie_addr(col, row, 0x1D000);
        let addr64: u64 = lo as u64; // high 32 bits are zero for AIE
        let raw = CdoRaw::Write64 { address: addr64, value: 0x1111_2222 };
        let ops = lower(&raw);
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            DeviceOp::RegWrite { tile, offset, value } => {
                assert_eq!(*tile, TileAddr::new(col, row));
                assert_eq!(*offset, 0x1D000);
                assert_eq!(*value, 0x1111_2222);
            }
            _ => panic!("expected RegWrite, got {:?}", ops[0]),
        }
    }

    #[test]
    fn mask_write64_truncates_address_and_lowers_to_regmask() {
        let (col, row) = compute_tile_addr();
        let lo = aie_addr(col, row, 0x1D000);
        let raw = CdoRaw::MaskWrite64 {
            address: lo as u64,
            mask: 0xFF,
            value: 0x55,
        };
        let ops = lower(&raw);
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            DeviceOp::RegMask { tile, offset, mask, value } => {
                assert_eq!(*tile, TileAddr::new(col, row));
                assert_eq!(*offset, 0x1D000);
                assert_eq!(*mask, 0xFF);
                assert_eq!(*value, 0x55);
            }
            _ => panic!("expected RegMask, got {:?}", ops[0]),
        }
    }

    #[test]
    fn mask_poll_lowers_with_tile_and_offset() {
        let (col, row) = compute_tile_addr();
        let addr = aie_addr(col, row, 0x1DF00);
        let raw = CdoRaw::MaskPoll { address: addr, mask: 0x3, expected: 0x2 };
        let ops = lower(&raw);
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            DeviceOp::MaskPoll { tile, offset, mask, expected } => {
                assert_eq!(*tile, TileAddr::new(col, row));
                assert_eq!(*offset, 0x1DF00);
                assert_eq!(*mask, 0x3);
                assert_eq!(*expected, 0x2);
            }
            _ => panic!("expected MaskPoll, got {:?}", ops[0]),
        }
    }

    #[test]
    fn mask_poll64_truncates_address() {
        let (col, row) = compute_tile_addr();
        let addr = aie_addr(col, row, 0x1DF00);
        let raw = CdoRaw::MaskPoll64 {
            address: addr as u64,
            mask: 0x3,
            expected: 0x2,
        };
        let ops = lower(&raw);
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            DeviceOp::MaskPoll { tile, offset, mask, expected } => {
                assert_eq!(*tile, TileAddr::new(col, row));
                assert_eq!(*offset, 0x1DF00);
                assert_eq!(*mask, 0x3);
                assert_eq!(*expected, 0x2);
            }
            _ => panic!("expected MaskPoll, got {:?}", ops[0]),
        }
    }

    #[test]
    fn delay_passes_through() {
        let ops = lower(&CdoRaw::Delay { cycles: 123 });
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], DeviceOp::Delay { cycles: 123 }));
    }

    #[test]
    fn marker_passes_through() {
        let ops = lower(&CdoRaw::Marker { value: 0xDEAD_BEEF });
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], DeviceOp::Marker { value: 0xDEAD_BEEF }));
    }

    // ------------------------------------------------------------------
    // Empty lowerings
    // ------------------------------------------------------------------

    #[test]
    fn nop_produces_empty() {
        let ops = lower(&CdoRaw::Nop { words: 0 });
        assert!(ops.is_empty(), "Nop should lower to no DeviceOps");
    }

    #[test]
    fn end_mark_produces_empty() {
        let ops = lower(&CdoRaw::EndMark);
        assert!(ops.is_empty(), "EndMark should lower to no DeviceOps");
    }

    #[test]
    fn unknown_produces_empty_and_warns() {
        // The log::warn! side effect isn't asserted here (we don't
        // instrument a test logger); the empty result is the contract.
        let ops = lower(&CdoRaw::Unknown {
            opcode: 0xFFF,
            payload: vec![1, 2, 3],
        });
        assert!(ops.is_empty(), "Unknown should lower to no DeviceOps");
    }

    // ------------------------------------------------------------------
    // DmaWrite -> RegBurst
    // ------------------------------------------------------------------

    #[test]
    fn dma_write_bytes_reshape_to_words_little_endian() {
        let (col, row) = compute_tile_addr();
        let addr = aie_addr(col, row, 0x0);
        // 3 little-endian words: 0x0000_0001, 0xDEAD_BEEF, 0xCAFE_BABE
        let data: Vec<u8> = vec![
            0x01, 0x00, 0x00, 0x00,
            0xEF, 0xBE, 0xAD, 0xDE,
            0xBE, 0xBA, 0xFE, 0xCA,
        ];
        let raw = CdoRaw::DmaWrite { address: addr, data };
        let ops = lower(&raw);
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            DeviceOp::RegBurst { tile, offset, words } => {
                assert_eq!(*tile, TileAddr::new(col, row));
                assert_eq!(*offset, 0x0);
                assert_eq!(words.as_slice(), &[0x0000_0001, 0xDEAD_BEEF, 0xCAFE_BABE]);
                assert!(!words.spilled(), "3 words should stay inline");
            }
            _ => panic!("expected RegBurst, got {:?}", ops[0]),
        }
    }

    #[test]
    fn dma_write_odd_tail_pads_with_zeros() {
        // Pathological: 5 bytes, not a whole word multiple. Real CDOs
        // always align, but we still want defensive behavior: pad to a
        // whole word, don't truncate.
        let raw = CdoRaw::DmaWrite {
            address: aie_addr(1, 2, 0),
            data: vec![0x11, 0x22, 0x33, 0x44, 0xAA],
        };
        let ops = lower(&raw);
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            DeviceOp::RegBurst { words, .. } => {
                assert_eq!(words.len(), 2);
                assert_eq!(words[0], 0x4433_2211);
                // Trailing word: 0xAA padded with three zero bytes.
                assert_eq!(words[1], 0x0000_00AA);
            }
            _ => panic!("expected RegBurst, got {:?}", ops[0]),
        }
    }

    // ------------------------------------------------------------------
    // CoreEnable promotion (CORE_CONTROL on compute tile)
    // ------------------------------------------------------------------

    #[test]
    fn core_control_write_enable_one_lowers_to_core_enable_true() {
        let (col, row) = compute_tile_addr();
        let addr = aie_addr(col, row, CORE_CONTROL);
        let ops = lower(&CdoRaw::Write { address: addr, value: 0x1 });
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            DeviceOp::CoreEnable { tile, enabled, value } => {
                assert_eq!(*tile, TileAddr::new(col, row));
                assert!(*enabled, "bit 0 set -> enabled=true");
                assert_eq!(*value, 0x1, "raw value must be carried for readback");
            }
            _ => panic!("expected CoreEnable, got {:?}", ops[0]),
        }
    }

    #[test]
    fn core_control_write_enable_zero_lowers_to_core_enable_false() {
        let (col, row) = compute_tile_addr();
        let addr = aie_addr(col, row, CORE_CONTROL);
        // Value with other bits set but bit 0 = 0 -> disable.
        let ops = lower(&CdoRaw::Write { address: addr, value: 0x2 });
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            DeviceOp::CoreEnable { enabled, value, .. } => {
                assert!(!*enabled, "bit 0 clear -> enabled=false");
                assert_eq!(*value, 0x2, "raw value preserved even when disabling");
            }
            _ => panic!("expected CoreEnable, got {:?}", ops[0]),
        }
    }

    #[test]
    fn core_control_write_on_shim_tile_is_regwrite_not_core_enable() {
        // Row 0 = shim. A write to offset CORE_CONTROL on a shim tile
        // is not a core enable (shims don't have compute cores). Must
        // fall through to RegWrite.
        let addr = aie_addr(0, 0, CORE_CONTROL);
        let ops = lower(&CdoRaw::Write { address: addr, value: 0x1 });
        assert_eq!(ops.len(), 1);
        assert!(
            matches!(ops[0], DeviceOp::RegWrite { .. }),
            "shim write to CORE_CONTROL offset must not promote to CoreEnable, got {:?}",
            ops[0]
        );
    }

    #[test]
    fn core_control_write_on_memtile_is_regwrite_not_core_enable() {
        // Row 1 = memtile. No core either; must fall through.
        let addr = aie_addr(0, 1, CORE_CONTROL);
        let ops = lower(&CdoRaw::Write { address: addr, value: 0x1 });
        assert_eq!(ops.len(), 1);
        assert!(
            matches!(ops[0], DeviceOp::RegWrite { .. }),
            "memtile write to CORE_CONTROL offset must not promote to CoreEnable, got {:?}",
            ops[0]
        );
    }

    #[test]
    fn core_control_mask_write_touching_bit0_stays_reg_mask() {
        // Post-D.3: MaskWrites are never promoted to CoreEnable, even
        // when the mask touches bit 0. Mask-blend correctness lives
        // on the apply side (mask_write_register -> mask_write_core_register).
        let (col, row) = compute_tile_addr();
        let addr = aie_addr(col, row, CORE_CONTROL);
        let op = lower_mask_write(addr, 0x1, 0x0);
        match op {
            DeviceOp::RegMask { tile, offset, mask, value } => {
                assert_eq!(tile.col, col);
                assert_eq!(tile.row, row);
                assert_eq!(offset, CORE_CONTROL);
                assert_eq!(mask, 0x1);
                assert_eq!(value, 0x0);
            }
            other => panic!(
                "MaskWrite to CORE_CONTROL must lower to RegMask post-D.3, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn core_control_mask_write_not_touching_bit0_stays_reg_mask() {
        let (col, row) = compute_tile_addr();
        let addr = aie_addr(col, row, CORE_CONTROL);
        // Mask touches bit 1 only; enable bit is untouched.
        let ops = lower(&CdoRaw::MaskWrite {
            address: addr,
            mask: 0x2,
            value: 0x2,
        });
        assert_eq!(ops.len(), 1);
        assert!(
            matches!(ops[0], DeviceOp::RegMask { .. }),
            "MaskWrite to CORE_CONTROL that misses bit 0 must stay RegMask, got {:?}",
            ops[0]
        );
    }

    // ------------------------------------------------------------------
    // DmaStart promotion (DMA Start_Queue offsets on a compute tile).
    //
    // Start_Queue is the trigger register that pushes a BD ID and
    // enqueues a DMA transfer. The paired Ctrl register (Start_Queue
    // - 4) is channel configuration only and must pass through as
    // RegWrite. This mirrors real hardware's separation of channel
    // state (Ctrl) from transfer trigger (Start_Queue).
    // ------------------------------------------------------------------

    #[test]
    fn dma_s2mm_0_start_queue_on_compute_lowers_to_dma_start() {
        let (col, row) = compute_tile_addr();
        let addr = aie_addr(col, row, COMPUTE_DMA_S2MM_0_START_QUEUE);
        let ops = lower(&CdoRaw::Write { address: addr, value: 0x7 }); // BD ID 7
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            DeviceOp::DmaStart { tile, channel, dir, bd_id } => {
                assert_eq!(*tile, TileAddr::new(col, row));
                assert_eq!(*channel, 0);
                assert_eq!(*dir, DmaDirection::S2mm);
                assert_eq!(*bd_id, 0x7, "raw Start_Queue value must be preserved");
            }
            _ => panic!("expected DmaStart(S2mm, 0), got {:?}", ops[0]),
        }
    }

    #[test]
    fn dma_s2mm_1_start_queue_on_compute_lowers_to_dma_start() {
        let (col, row) = compute_tile_addr();
        let addr = aie_addr(col, row, COMPUTE_DMA_S2MM_1_START_QUEUE);
        let ops = lower(&CdoRaw::Write { address: addr, value: 0x3 });
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            DeviceOp::DmaStart { channel, dir, bd_id, .. } => {
                assert_eq!(*channel, 1);
                assert_eq!(*dir, DmaDirection::S2mm);
                assert_eq!(*bd_id, 0x3);
            }
            _ => panic!("expected DmaStart(S2mm, 1), got {:?}", ops[0]),
        }
    }

    #[test]
    fn dma_mm2s_0_start_queue_on_compute_lowers_to_dma_start() {
        let (col, row) = compute_tile_addr();
        let addr = aie_addr(col, row, COMPUTE_DMA_MM2S_0_START_QUEUE);
        let ops = lower(&CdoRaw::Write { address: addr, value: 0x1 });
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            DeviceOp::DmaStart { channel, dir, bd_id, .. } => {
                assert_eq!(*channel, 0);
                assert_eq!(*dir, DmaDirection::Mm2s);
                assert_eq!(*bd_id, 0x1);
            }
            _ => panic!("expected DmaStart(Mm2s, 0), got {:?}", ops[0]),
        }
    }

    #[test]
    fn dma_mm2s_1_start_queue_on_compute_lowers_to_dma_start() {
        let (col, row) = compute_tile_addr();
        let addr = aie_addr(col, row, COMPUTE_DMA_MM2S_1_START_QUEUE);
        let ops = lower(&CdoRaw::Write { address: addr, value: 0x5 });
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            DeviceOp::DmaStart { channel, dir, bd_id, .. } => {
                assert_eq!(*channel, 1);
                assert_eq!(*dir, DmaDirection::Mm2s);
                assert_eq!(*bd_id, 0x5);
            }
            _ => panic!("expected DmaStart(Mm2s, 1), got {:?}", ops[0]),
        }
    }

    #[test]
    fn dma_ctrl_write_on_compute_stays_regwrite() {
        // Ctrl (Start_Queue - 4) is configuration only; writing it
        // never starts a transfer. Must pass through as RegWrite even
        // with bit 0 set (channel-armed configuration, not a trigger).
        let (col, row) = compute_tile_addr();
        let addr = aie_addr(col, row, COMPUTE_DMA_S2MM_0_CTRL);
        let ops = lower(&CdoRaw::Write { address: addr, value: 0x1 });
        assert_eq!(ops.len(), 1);
        assert!(
            matches!(ops[0], DeviceOp::RegWrite { .. }),
            "DMA Ctrl write must stay RegWrite regardless of value, got {:?}",
            ops[0]
        );
    }

    #[test]
    fn dma_start_queue_write_on_shim_tile_is_regwrite_not_dma_start() {
        // Shim has DMA channels too but at a different base address
        // (0x1D200 Ctrl / 0x1D204 Start_Queue). A write at compute's
        // Start_Queue offset on a shim tile is either a different
        // register or out-of-range; it must not promote.
        let addr = aie_addr(0, 0, COMPUTE_DMA_S2MM_0_START_QUEUE);
        let ops = lower(&CdoRaw::Write { address: addr, value: 0x1 });
        assert_eq!(ops.len(), 1);
        assert!(
            matches!(ops[0], DeviceOp::RegWrite { .. }),
            "shim write at compute Start_Queue offset must stay RegWrite, got {:?}",
            ops[0]
        );
    }

    #[test]
    fn dma_start_queue_write_on_memtile_is_regwrite_not_dma_start() {
        let addr = aie_addr(0, 1, COMPUTE_DMA_S2MM_0_START_QUEUE);
        let ops = lower(&CdoRaw::Write { address: addr, value: 0x1 });
        assert_eq!(ops.len(), 1);
        assert!(
            matches!(ops[0], DeviceOp::RegWrite { .. }),
            "memtile write at compute Start_Queue offset must stay RegWrite, got {:?}",
            ops[0]
        );
    }
}
