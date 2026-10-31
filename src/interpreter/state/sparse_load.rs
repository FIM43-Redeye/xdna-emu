//! The sparse-load unit behind `vldb.sparse.{reset,fill,peek,pop}`.
//!
//! The rule is `xdna_archspec::aie2::sparse_load` (read off NPU1 silicon). The unit owns a
//! ring of 32-byte lines the loads write; a read takes its 8-byte mask and the data bytes behind
//! it from the ring at the position `DP` names. `DP` and the two status flags are not here: they
//! are scalar-file slots (`DP_REG_INDEX`, `SPARSE_OF_REG_INDEX`, `COMPR_UF_REG_INDEX`) so a move
//! reads and writes them like any other register; the handler keeps them in step.

use xdna_archspec::aie2::sparse_load::{
    compr_uf_window, dp_advance, expand_half, fetch_has_room, HALF_DATA_BYTES, LINE_BYTES, MASK_BYTES,
    RING_BYTES,
};

/// What one `pop` or `peek` read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseRead {
    pub mask: [u8; MASK_BYTES],
    pub data: [u8; HALF_DATA_BYTES],
    /// A group had more than two selected units (`srSparse_of`).
    pub overflow: bool,
    /// The ring was low or just over (`srCompr_uf`).
    pub underflow: bool,
    /// Bytes `DP` moves if this is a `pop`.
    pub dp_advance: u32,
}

#[derive(Clone)]
pub struct SparseLoadUnit {
    ring: [u8; RING_BYTES],
    /// Where the next line is written.
    write: usize,
    /// Ring position of the stream start (the write position at the last `reset`).
    origin: usize,
    /// Lines written since the last `reset`.
    lines: i64,
    /// `DP` at the last `reset`: the address of the stream start.
    dp_base: u32,
}

impl Default for SparseLoadUnit {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseLoadUnit {
    pub fn new() -> Self {
        Self { ring: [0; RING_BYTES], write: 0, origin: 0, lines: 0, dp_base: 0 }
    }

    /// `reset [p]`: the stream starts at address `pointer`; its first line is written by [`fetch`].
    pub fn start(&mut self, pointer: u32) {
        self.origin = self.write;
        self.lines = 0;
        self.dp_base = pointer;
    }

    /// Write a line into the ring.
    pub fn fetch(&mut self, line: &[u8; LINE_BYTES]) {
        self.ring[self.write..self.write + LINE_BYTES].copy_from_slice(line);
        self.write = (self.write + LINE_BYTES) % RING_BYTES;
        self.lines += 1;
    }

    /// Bytes the ring holds beyond the read position `dp` (negative once past them).
    fn ahead(&self, dp: u32) -> i64 {
        LINE_BYTES as i64 * self.lines - i64::from(dp.wrapping_sub(self.dp_base) as i32)
    }

    /// Whether a `pop` or `peek` at `dp` writes its line.
    pub fn has_room(&self, dp: u32) -> bool {
        fetch_has_room(self.ahead(dp))
    }

    fn at(&self, offset: i64) -> u8 {
        self.ring[(self.origin as i64 + offset).rem_euclid(RING_BYTES as i64) as usize]
    }

    /// Read half a register at `dp`: the mask, the expanded data, the flags it sets and the
    /// `DP` advance a `pop` makes. `unit_bytes` is 1 for the 4- and 8-bit forms, 2 for 16-bit.
    pub fn read(&self, dp: u32, unit_bytes: u8) -> SparseRead {
        let offset = i64::from(dp.wrapping_sub(self.dp_base) as i32);
        let mut mask = [0u8; MASK_BYTES];
        for (i, m) in mask.iter_mut().enumerate() {
            *m = self.at(offset + i as i64);
        }
        let mut cursor = offset + MASK_BYTES as i64;
        let (data, overflow) = expand_half(unit_bytes, &mask, || {
            let b = self.at(cursor);
            cursor += 1;
            b
        });
        SparseRead {
            mask,
            data,
            overflow,
            underflow: compr_uf_window(self.ahead(dp)),
            dp_advance: dp_advance(&mask),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(fill: u8) -> [u8; LINE_BYTES] {
        [fill; LINE_BYTES]
    }

    #[test]
    fn a_read_takes_the_mask_then_the_data_it_selects() {
        let mut u = SparseLoadUnit::new();
        u.start(0x1000);
        let mut first = [0u8; LINE_BYTES];
        first[0] = 0x03; // group 0: bits 0 and 1
        first[8] = 0xAA;
        first[9] = 0xBB;
        u.fetch(&first);
        let r = u.read(0x1000, 1);
        assert_eq!(&r.data[..2], &[0xAA, 0xBB]);
        assert_eq!(r.dp_advance, 8 + 4);
        assert!(!r.overflow);
    }

    #[test]
    fn reading_past_what_was_written_returns_what_the_ring_holds() {
        let mut u = SparseLoadUnit::new();
        u.start(0);
        for i in 0..10u8 {
            u.fetch(&line(i + 1));
        }
        // The ring is full; the next line replaces the oldest, and a read at the start sees it.
        u.fetch(&line(0x77));
        assert_eq!(u.read(0, 1).mask, [0x77; 8]);
        // One full turn past the start reads the same slot again.
        assert_eq!(u.read(320, 1).mask, [0x77; 8]);
    }

    #[test]
    fn a_reset_continues_the_ring_where_the_last_stream_left_off() {
        let mut u = SparseLoadUnit::new();
        u.start(0);
        u.fetch(&line(1));
        u.fetch(&line(2));
        u.start(0x40); // the new stream starts at write position 64
        u.fetch(&line(3));
        assert_eq!(u.read(0x40, 1).mask, [3; 8]);
        // The old stream's second line is still there, one line before the new start.
        assert_eq!(u.read(0x40 - 32, 1).mask, [2; 8]);
    }

    #[test]
    fn the_underflow_flag_follows_the_bytes_left_beyond_the_read() {
        let mut u = SparseLoadUnit::new();
        u.start(0);
        for _ in 0..4 {
            u.fetch(&line(0));
        }
        assert!(!u.read(0, 1).underflow, "four lines ahead");
        assert!(u.read(64, 1).underflow, "two lines ahead");
        assert!(u.read(128 + 16, 1).underflow, "past the end by less than a line");
        assert!(!u.read(128 + 40, 1).underflow, "far past the end");
    }

    #[test]
    fn room_is_lost_at_nine_lines_ahead_and_regained_by_consuming() {
        let mut u = SparseLoadUnit::new();
        u.start(0);
        for _ in 0..9 {
            u.fetch(&line(0));
        }
        assert!(!u.has_room(0));
        assert!(u.has_room(32));
    }
}
