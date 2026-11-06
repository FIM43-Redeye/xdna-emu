//! The data rings behind the core's two load ports.
//!
//! Each load port (A: `vlda`, `lda` in all widths, `lda q`, `vlda.ups`, `vlda.conv`; B: `vldb`,
//! `vldb.128`, `vldb.unpack`, `vldb.sparse`) keeps the 16-byte memory rows it fetched in a ring
//! of [`RING_DEPTH`] entries per 16-byte half of its 256-bit path: one ring for the rows at
//! address bit 4 = 0 and one for bit 4 = 1, each with its own pointer. Read off NPU1 silicon
//! with `tools/gen_port_latch_probe.py` (suite `build/experiments/harness/port-latch/suite`, every
//! case r1 == r2):
//!
//! - A load with memory behind it writes every row it fetches (the whole row, also for a scalar
//!   or 128-bit load) into the slot at its half's pointer and advances that pointer. A 256-bit
//!   load writes and advances both halves, a narrower one only the half its address selects.
//! - A load into a direction with no core memory behind it (south of a row-2 core: the memory
//!   tiles; north of the top row) writes nothing: it returns the slot at the pointer and advances
//!   the pointer just the same. That is the "stale port data" such a read returns.
//! - Stores never touch a ring, and the ports never touch each other's rings.
//!
//! A `vldb.4x*` gather is one access on each port: lanes 0-1 (or 4-5) on A, 2-3 (or 6-7) on B,
//! each lane through its half (even lanes have bit 4 clear, odd lanes set).
//!
//! The rings persist across kernel launches on silicon; an emulator core starts with zeros.

/// Entries per half-ring. Silicon: a sweep of south `vldb` probes cycles with period six whatever
/// the address (`sweep` ph8/ph9), and after a fill of six lines returns them oldest first
/// (`ring_a` b0, `ring_d` b0).
pub const RING_DEPTH: usize = 6;
/// Bytes of one ring entry: one 128-bit memory bank row.
pub const ROW_BYTES: usize = 16;

/// A core load port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadPort {
    A,
    B,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct HalfRing {
    rows: [[u8; ROW_BYTES]; RING_DEPTH],
    ptr: usize,
}

impl HalfRing {
    fn pass(&mut self, fetched: Option<[u8; ROW_BYTES]>) -> [u8; ROW_BYTES] {
        let out = match fetched {
            Some(row) => {
                self.rows[self.ptr] = row;
                row
            }
            None => self.rows[self.ptr],
        };
        self.ptr = (self.ptr + 1) % RING_DEPTH;
        out
    }
}

/// Both ports' rings, two halves each.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LoadPortRings {
    rings: [[HalfRing; 2]; 2],
}

impl LoadPortRings {
    pub fn new() -> Self {
        Self::default()
    }

    /// One row through `port`'s ring for `half` (address bit 4): `Some(row)` is a row fetched
    /// from memory (written and returned), `None` an access with no memory behind it (the slot
    /// is returned). Either way the pointer advances.
    pub fn pass(&mut self, port: LoadPort, half: usize, fetched: Option<[u8; ROW_BYTES]>) -> [u8; ROW_BYTES] {
        self.rings[port as usize][half & 1].pass(fetched)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(tag: u8) -> [u8; ROW_BYTES] {
        [tag; ROW_BYTES]
    }

    #[test]
    fn a_read_with_no_memory_returns_the_row_written_six_accesses_ago() {
        let mut r = LoadPortRings::new();
        for t in 1..=6 {
            r.pass(LoadPort::B, 0, Some(row(t)));
        }
        let got: Vec<u8> = (0..7).map(|_| r.pass(LoadPort::B, 0, None)[0]).collect();
        assert_eq!(got, vec![1, 2, 3, 4, 5, 6, 1]);
    }

    #[test]
    fn a_fetched_row_is_returned_and_overwrites_its_slot() {
        let mut r = LoadPortRings::new();
        for t in 1..=6 {
            r.pass(LoadPort::A, 1, Some(row(t)));
        }
        assert_eq!(r.pass(LoadPort::A, 1, Some(row(9))), row(9));
        let got: Vec<u8> = (0..6).map(|_| r.pass(LoadPort::A, 1, None)[0]).collect();
        assert_eq!(got, vec![2, 3, 4, 5, 6, 9]);
    }

    #[test]
    fn halves_and_ports_are_independent() {
        let mut r = LoadPortRings::new();
        r.pass(LoadPort::B, 0, Some(row(1)));
        r.pass(LoadPort::B, 1, Some(row(2)));
        r.pass(LoadPort::A, 0, Some(row(3)));
        // B lo has advanced once, B hi once, A lo once, A hi never
        for _ in 0..5 {
            r.pass(LoadPort::B, 0, None);
        }
        assert_eq!(r.pass(LoadPort::B, 0, None), row(1));
        assert_eq!(r.pass(LoadPort::A, 1, None), [0; ROW_BYTES]);
        for _ in 0..4 {
            r.pass(LoadPort::A, 0, None);
        }
        assert_eq!(r.pass(LoadPort::A, 0, None), [0; ROW_BYTES]);
        assert_eq!(r.pass(LoadPort::A, 0, None), row(3));
    }
}
