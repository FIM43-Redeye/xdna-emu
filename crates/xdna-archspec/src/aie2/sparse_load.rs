//! The AIE2 sparse loads `vldb.sparse.{reset,fill,peek,pop}.{4,8,16}`: per-form facts and the
//! load rule.
//!
//! A sparse load unit turns a compressed stream in memory into half a `qx` register (the
//! stored 2:4 data of [`super::matmul::SparseLayout`] plus its mask) that the sparse MAC reads.
//! The forms (names, `Defs`, `Uses`) come from llvm-aie TableGen and are cross-checked against it
//! by the tests below; the rule is read off NPU1 silicon (`vec_sparseload_*`, six captures, see
//! `docs/fidelity-gaps/vector-compute.md`):
//!
//! - The unit has a ring of [`RING_LINES`] 32-byte lines. `reset [p]` and `fill [p]` write the
//!   line at `p` into it and advance `p` by [`LINE_BYTES`]; `reset` also starts the read position
//!   (`DP = p`). `pop [p]` reads, then writes the line at `p` and advances `p` the same way;
//!   `peek [p]` reads and writes the line at `p` too, but leaves `p`. The line of a `pop` or
//!   `peek` is dropped while the ring already holds [`FETCH_AHEAD_LINES`] lines beyond the read
//!   position ([`fetch_has_room`]); `reset` and `fill` always write.
//! - A read takes an 8-byte mask at the read position and expands the data bytes that follow it
//!   (see [`expand_half`]); a `pop` then moves `DP` by [`dp_advance`].
//! - `srSparse_of` is set by a read of a mask with a group of more than two selected units;
//!   `srCompr_uf` by a read whose remaining bytes fall in [`compr_uf_window`].
//!
//! Widths 4 and 8 behave identically (the stream is byte-granular); width 16 selects 16-bit
//! elements with two mask bits each.

use super::matmul::{SPARSE_NO_SLOT, SPARSE_SLOT_TABLE};

/// Bytes the unit fetches per `reset`, `fill` and `pop`.
pub const LINE_BYTES: usize = 32;
/// Lines the ring holds. Reads past what has been written return what the ring still holds.
pub const RING_LINES: usize = 10;
pub const RING_BYTES: usize = LINE_BYTES * RING_LINES;
/// Bytes of mask a `pop` or `peek` reads (64 bits, one half of a `qx` mask).
pub const MASK_BYTES: usize = 8;
/// Bytes of expanded data a `pop` or `peek` produces (256 bits, one half of a `qx` data).
pub const HALF_DATA_BYTES: usize = 32;

/// The four operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseLoadOp {
    /// Forget the buffered stream and start it at the pointer.
    Reset,
    /// Fetch one more line.
    Fill,
    /// Read half a register without consuming it.
    Peek,
    /// Read half a register and consume it.
    Pop,
}

/// One `vldb.sparse.*` mnemonic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseLoadForm {
    pub mnemonic: &'static str,
    pub op: SparseLoadOp,
    /// Element width the mnemonic names (4, 8 or 16 bits).
    pub element_bits: u8,
    /// Stream bytes one selectable unit owns: the byte for 4- and 8-bit data, the 16-bit element
    /// (with two mask bits, one per byte) for 16-bit data.
    pub unit_bytes: u8,
    /// TableGen `Defs`/`Uses` (checked against the records).
    pub defs_dp: bool,
    pub uses_dp: bool,
    pub defs_sparse_of: bool,
    pub defs_compr_uf: bool,
    /// Writes a `qwl`/`qwh` destination (`pop`, `peek`).
    pub writes_destination: bool,
    /// The pointer operand leaves the instruction advanced by [`LINE_BYTES`].
    pub advances_pointer: bool,
}

include!(concat!(env!("OUT_DIR"), "/gen_sparse_load_forms.rs"));

/// Lines the ring may hold beyond the read position before a `pop` or `peek` stops fetching.
pub const FETCH_AHEAD_LINES: usize = 9;

/// Whether a `pop` or `peek` writes its line, given the bytes the ring holds beyond the read
/// position (`LINE_BYTES * lines written since reset - (DP - start)`; negative past the end).
pub fn fetch_has_room(ahead_bytes: i64) -> bool {
    ahead_bytes < (FETCH_AHEAD_LINES * LINE_BYTES) as i64
}

/// Bytes `DP` moves for a `pop` of `mask`: the mask itself plus the mask's population count
/// rounded up to a whole word, modulo 64 (a mask with every bit set moves only the mask).
pub fn dp_advance(mask: &[u8; MASK_BYTES]) -> u32 {
    let selected: u32 = mask.iter().map(|b| b.count_ones()).sum();
    MASK_BYTES as u32 + (selected.div_ceil(4) * 4) % 64
}

/// Whether a read sets `srCompr_uf`, from `remaining`: the bytes the ring has had written beyond
/// the read position (negative once the read is past them). Set while at most two lines remain,
/// and for a read up to one line past them; further out it is not set.
pub fn compr_uf_window(remaining: i64) -> bool {
    -(LINE_BYTES as i64) < remaining && remaining <= 2 * LINE_BYTES as i64
}

/// A group of four units selected by the pattern `p` (bit `i` for unit `i`) as the stored slot
/// of each selected unit, in unit order; `None` for the patterns of more than two units.
fn group_slots(pattern: usize) -> Option<([usize; 2], usize)> {
    let mut slots = [0usize; 2];
    let mut n = 0;
    for position in 0..4 {
        if pattern >> position & 1 == 1 {
            let slot = SPARSE_SLOT_TABLE[pattern][position];
            if slot == SPARSE_NO_SLOT || n == 2 {
                return None;
            }
            slots[n] = slot as usize;
            n += 1;
        }
    }
    Some((slots, n))
}

/// Expand one popped or peeked mask and its data bytes into the 32 data bytes of half a `qx`
/// register. `next` yields the stream bytes after the mask, one per set mask bit of a group that
/// is placed. Returns the data and whether any group had more than two selected units
/// (`srSparse_of`).
///
/// Width 4 and 8 (`unit_bytes == 1`): 16 groups of four mask bits, one bit per stream byte; group
/// `g` owns output bytes `2g` (slot 0) and `2g + 1` (slot 1). Width 16 (`unit_bytes == 2`): 8
/// groups of four 16-bit elements (one mask byte, two bits per element: bit `2e` selects the low
/// byte, bit `2e + 1` the high byte, either selects the element); slot `s` of group `j` owns
/// output bytes `4j + 2s` and `4j + 2s + 1`. A group with no, three or four selected units is
/// zero and consumes nothing.
pub fn expand_half(
    unit_bytes: u8,
    mask: &[u8; MASK_BYTES],
    mut next: impl FnMut() -> u8,
) -> ([u8; HALF_DATA_BYTES], bool) {
    let mut out = [0u8; HALF_DATA_BYTES];
    let mut overflow = false;
    let wide = unit_bytes == 2;
    let groups = if wide { MASK_BYTES } else { MASK_BYTES * 2 };
    for g in 0..groups {
        // Selection of the group's four units, and the mask bits of each.
        let bits_of_unit = |unit: usize| -> u8 {
            if wide {
                (mask[g] >> (2 * unit)) & 3
            } else {
                let nibble = (mask[g / 2] >> (4 * (g % 2))) & 0xF;
                (nibble >> unit) & 1
            }
        };
        let pattern = (0..4).fold(0usize, |p, u| p | usize::from(bits_of_unit(u) != 0) << u);
        let Some((slots, n)) = group_slots(pattern) else {
            overflow |= pattern.count_ones() > 2;
            continue;
        };
        let units: Vec<usize> = (0..4).filter(|&u| pattern >> u & 1 == 1).collect();
        for (&unit, &slot) in units.iter().zip(&slots[..n]) {
            let bits = bits_of_unit(unit);
            if wide {
                for byte in 0..2 {
                    if bits >> byte & 1 == 1 {
                        out[4 * g + 2 * slot + byte] = next();
                    }
                }
            } else {
                out[2 * g + slot] = next();
            }
        }
    }
    (out, overflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expand(unit_bytes: u8, mask: [u8; 8], data: &[u8]) -> ([u8; 32], bool) {
        let mut it = data.iter().copied();
        expand_half(unit_bytes, &mask, || it.next().expect("data exhausted"))
    }

    #[test]
    fn one_bit_in_the_upper_half_of_a_group_takes_slot_one() {
        // Group 0 (mask bits 0..3): bit 2 only.
        let (out, of) = expand(1, [0x04, 0, 0, 0, 0, 0, 0, 0], &[0xAB]);
        assert_eq!(out[1], 0xAB);
        assert_eq!(out.iter().filter(|&&b| b != 0).count(), 1);
        assert!(!of);
    }

    #[test]
    fn two_bits_take_both_slots_in_order_and_groups_are_two_bytes_apart() {
        // Group 0: bits 0 and 1; group 1 (the high nibble of byte 0): bit 0 only.
        let (out, _) = expand(1, [0x13, 0, 0, 0, 0, 0, 0, 0], &[1, 2, 3]);
        assert_eq!(&out[..4], &[1, 2, 3, 0]);
    }

    #[test]
    fn three_or_four_bits_zero_the_group_set_overflow_and_consume_nothing() {
        let (out, of) = expand(1, [0x07, 0, 0, 0, 0, 0, 0, 0], &[]);
        assert_eq!(out, [0; 32]);
        assert!(of);
        let (_, of) = expand(1, [0x0F, 0, 0, 0, 0, 0, 0, 0], &[]);
        assert!(of);
    }

    #[test]
    fn sixteen_bit_elements_take_a_stream_byte_per_set_bit() {
        // Mask byte 0 = 0x03: element 0, both bytes. Mask byte 1 = 0x05: elements 0 and 1, low bytes.
        let (out, of) = expand(2, [0x03, 0x05, 0, 0, 0, 0, 0, 0], &[0xA, 0xB, 0xC, 0xD]);
        assert_eq!(&out[..2], &[0xA, 0xB]);
        assert_eq!(out[4], 0xC);
        assert_eq!(out[6], 0xD);
        assert!(!of);
    }

    #[test]
    fn a_full_sixteen_bit_group_is_four_elements_and_so_zero() {
        // 0x55: bits 0, 2, 4, 6 are one bit of each of four elements.
        let (out, of) = expand(2, [0x55; 8], &[]);
        assert_eq!(out, [0; 32]);
        assert!(of);
    }

    #[test]
    fn dp_advances_by_the_mask_and_the_rounded_popcount() {
        assert_eq!(dp_advance(&[0; 8]), 8);
        assert_eq!(dp_advance(&[0x03, 0, 0, 0, 0, 0, 0, 0]), 8 + 4);
        assert_eq!(dp_advance(&[0xFF; 8]), 8, "64 set bits wrap to no data");
        assert_eq!(dp_advance(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F]), 8 + 60);
        assert_eq!(dp_advance(&[0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x07]), 8 + 32);
    }

    #[test]
    fn a_fetch_needs_room_for_the_ring_to_stay_below_nine_lines_ahead() {
        assert!(fetch_has_room(8 * 32 + 31));
        assert!(!fetch_has_room(9 * 32));
        assert!(fetch_has_room(-100));
    }

    #[test]
    fn underflow_window_edges() {
        assert!(compr_uf_window(64) && !compr_uf_window(68));
        assert!(compr_uf_window(-28) && !compr_uf_window(-32));
        assert!(compr_uf_window(0));
    }

    /// Every TableGen `VLDB_SPARSE_*` record is a form of the table (the build already refuses a
    /// table that disagrees with the records), the decoder tables know exactly those mnemonics,
    /// and peek and pop are the forms with a destination besides the pointer.
    #[test]
    fn every_vldb_sparse_record_is_a_form_and_the_decoder_tables_agree() {
        let isa = crate::aie2::isa::load_from_generated();
        let mut seen = std::collections::BTreeSet::new();
        for e in isa.encodings_by_slot.values().flatten() {
            if !e.name.starts_with("VLDB_SPARSE_") {
                continue;
            }
            let f = sparse_load_form(&e.mnemonic).unwrap_or_else(|| panic!("{}: no form", e.mnemonic));
            assert!(e.may_load, "{}: mayLoad", e.mnemonic);
            assert_eq!(
                e.output_order.len(),
                if f.writes_destination { 2 } else { 1 },
                "{}: outputs {:?}",
                e.mnemonic,
                e.output_order
            );
            seen.insert(e.mnemonic.clone());
        }
        let table: std::collections::BTreeSet<String> =
            SPARSE_LOAD_FORMS.iter().map(|f| f.mnemonic.to_string()).collect();
        assert_eq!(seen, table);
        assert_eq!(table.len(), 12);
    }

    #[test]
    fn defs_and_uses_come_from_the_records() {
        let flags = |m: &str| {
            let f = sparse_load_form(m).unwrap();
            (f.defs_dp, f.uses_dp, f.defs_sparse_of, f.defs_compr_uf)
        };
        // Every form uses DP (a read needs the position, a fetch is ordered after it); reset and
        // pop define it.
        assert_eq!(flags("vldb.sparse.reset.8"), (true, true, true, false));
        assert_eq!(flags("vldb.sparse.fill.8"), (false, true, true, false));
        assert_eq!(flags("vldb.sparse.peek.16"), (false, true, true, true));
        assert_eq!(flags("vldb.sparse.pop.4"), (true, true, true, true));
    }
}
