//! Same-cycle collisions of register-file writes (pure resolver).
//!
//! Compilers never schedule two writes onto one register-file write port in one cycle: the
//! TableGen resource model forbids it. Hand-written (probe) schedules can, and NPU1 silicon is
//! deterministic about what lands. The rules below were read off the collision probes
//! (`build/experiments/collisions/probe*`, silicon runs r1 and r2 byte-identical) and replayed
//! against them point by point; the evidence counts are in
//! `docs/fidelity-gaps/vector-compute.md`.
//!
//! # Model
//!
//! A register file is a set of 256-bit **banks**. The accumulator file has four: a `cm` register
//! is four quarters, and quarter `q` of half `h` (`bml` = 0, `bmh` = 1) of *any* `cm` goes to bank
//! `2h + q`. The vector (W) file has two: `wl` (even register) and `wh` (odd). The bank, not the
//! register, is the contended resource.
//!
//! A write is a list of **chunks**, one 256-bit piece per bank it targets, numbered by
//! **ordinal** in the order the instruction produces them (a `bm` half has ordinals 0 and 1, a
//! `cm` 0..3, an `x` register 0 and 1, an `am` quarter or a `w` register only 0).
//!
//! 1. **Same port, same landing cycle.** The port has one 256-bit data lane per ordinal. Lane
//!    `k` carries the data of ordinal `k` of the *latest-issued* writer that has one. Bank `p`
//!    takes its data from lane `p` when some writer drives it, else from lane 0, and goes to the
//!    register of the latest-issued writer that targets bank `p`; the chunks of earlier writers
//!    on that bank are dropped. Banks nobody targets are untouched. A port with a single writer
//!    passes through unchanged.
//! 2. **Different ports, same landing cycle.** Only chunks on the same register and the same
//!    bank conflict: the lower-indexed port wins, with its own data. Chunks on different
//!    registers both land, even on the same bank.
//!
//! "Latest-issued" is the issue sequence number of the writing instruction (bundle, then slot
//! order within a bundle).

/// Identity of the instruction behind a queued register write, as the register files record it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteTag {
    /// The write port ([`Chunk::port`]).
    pub port: u16,
    /// Issue sequence number ([`Chunk::writer`]).
    pub seq: u64,
    /// Absolute issued-bundle count at which the result lands (issue + result operand cycle).
    /// Writes collide only when this is equal.
    pub land: u64,
}

/// One 256-bit piece of a register write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chunk {
    /// Issue sequence number of the writing instruction (larger = issued later).
    pub writer: u64,
    /// Index of the write port, in functional-unit order (smaller = higher priority).
    pub port: u16,
    /// Position of this chunk within its instruction's write.
    pub ordinal: u8,
    /// Bank the chunk targets.
    pub bank: u8,
    /// Register the chunk is written to (opaque; compared for equality).
    pub reg: u16,
    /// The 256 bits the instruction itself computed for this chunk.
    pub data: [u32; 8],
}

/// What happens to one input chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Landing {
    /// Whether the chunk is written to its register at all.
    pub lands: bool,
    /// The 256 bits written when it does.
    pub data: [u32; 8],
}

/// Resolve the chunks that land in one cycle in one register file. `out[i]` describes
/// `chunks[i]`.
pub fn resolve(chunks: &[Chunk]) -> Vec<Landing> {
    let mut out: Vec<Landing> = chunks.iter().map(|c| Landing { lands: true, data: c.data }).collect();
    let mut ports: Vec<u16> = chunks.iter().map(|c| c.port).collect();
    ports.sort_unstable();
    ports.dedup();
    for &port in &ports {
        resolve_port(chunks, port, &mut out);
    }
    resolve_cross_port(chunks, &mut out);
    out
}

fn writers_of(chunks: &[Chunk], port: u16) -> Vec<u64> {
    let mut w: Vec<u64> = chunks.iter().filter(|c| c.port == port).map(|c| c.writer).collect();
    w.sort_unstable();
    w.dedup();
    w
}

fn resolve_port(chunks: &[Chunk], port: u16, out: &mut [Landing]) {
    if writers_of(chunks, port).len() < 2 {
        return;
    }
    let on_port = || chunks.iter().enumerate().filter(|(_, c)| c.port == port);
    // lane k: the data of ordinal k of the latest-issued writer that has one
    let lane = |k: u8| -> Option<[u32; 8]> {
        on_port()
            .filter(|(_, c)| c.ordinal == k)
            .max_by_key(|(_, c)| c.writer)
            .map(|(_, c)| c.data)
    };
    for (i, c) in on_port() {
        let latest_on_bank = on_port().filter(|(_, o)| o.bank == c.bank).map(|(_, o)| o.writer).max();
        if latest_on_bank != Some(c.writer) {
            out[i].lands = false;
            continue;
        }
        out[i].data = lane(c.bank).or_else(|| lane(0)).unwrap_or(c.data);
    }
}

fn resolve_cross_port(chunks: &[Chunk], out: &mut [Landing]) {
    for i in 0..chunks.len() {
        if !out[i].lands {
            continue;
        }
        let beaten = (0..chunks.len()).any(|j| {
            out[j].lands
                && chunks[j].port < chunks[i].port
                && chunks[j].reg == chunks[i].reg
                && chunks[j].bank == chunks[i].bank
        });
        if beaten {
            out[i].lands = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A chunk whose 256 bits are recognisable: every word is `tag`.
    fn chunk(writer: u64, port: u16, ordinal: u8, bank: u8, reg: u16, tag: u32) -> Chunk {
        Chunk { writer, port, ordinal, bank, reg, data: [tag; 8] }
    }

    fn lands(l: &Landing) -> Option<u32> {
        l.lands.then(|| l.data[0])
    }

    #[test]
    fn a_lone_writer_passes_through_even_off_bank_zero() {
        // a bmh write: banks 2 and 3, ordinals 0 and 1
        let r = resolve(&[chunk(1, 0, 0, 2, 7, 0xA0), chunk(1, 0, 1, 3, 7, 0xA1)]);
        assert_eq!(r.iter().map(lands).collect::<Vec<_>>(), [Some(0xA0), Some(0xA1)]);
    }

    #[test]
    fn same_port_later_writer_takes_the_banks_it_targets() {
        // A: x register (banks 0,1), B: wl register (bank 0), one port, one landing cycle.
        // Bank 0 goes to B's register with B's data; A keeps bank 1 and its own data there.
        let a = [chunk(1, 0, 0, 0, 8, 0xA0), chunk(1, 0, 1, 1, 9, 0xA1)];
        let b = [chunk(2, 0, 0, 0, 4, 0xB0)];
        let all = [a[0], a[1], b[0]];
        let r = resolve(&all);
        assert_eq!(r.iter().map(lands).collect::<Vec<_>>(), [None, Some(0xA1), Some(0xB0)]);
    }

    #[test]
    fn same_port_a_narrow_later_writer_on_the_high_bank_takes_the_other_writers_lane() {
        // B writes only wh (bank 1, ordinal 0); A is a 512-bit writer. Bank 1 goes to B's
        // register but carries lane 1, which only A drives; bank 0 stays A's register but
        // carries lane 0, whose latest driver is B (ordinal 0), so A's low half gets B's data.
        let all = [chunk(1, 0, 0, 0, 8, 0xA0), chunk(1, 0, 1, 1, 9, 0xA1), chunk(2, 0, 0, 1, 5, 0xB0)];
        let r = resolve(&all);
        assert_eq!(r.iter().map(lands).collect::<Vec<_>>(), [Some(0xB0), None, Some(0xA1)]);
    }

    #[test]
    fn same_port_disjoint_banks_still_share_lane_zero() {
        // A writes bank 0 (lane 0). B writes bank 3 with ordinal 0 only. Lane 3 is nobody's,
        // so bank 3 takes lane 0, which is the later-issued B's own data (B drives lane 0).
        let all = [chunk(1, 0, 0, 0, 8, 0xA0), chunk(2, 0, 0, 3, 5, 0xB0)];
        let r = resolve(&all);
        // bank 0 (A only): lane 0 is B's ordinal-0 data
        assert_eq!(r.iter().map(lands).collect::<Vec<_>>(), [Some(0xB0), Some(0xB0)]);
    }

    #[test]
    fn different_registers_on_the_same_port_still_contend() {
        let all = [chunk(1, 0, 0, 0, 8, 0xA0), chunk(2, 0, 0, 0, 4, 0xB0)];
        let r = resolve(&all);
        assert_eq!(r.iter().map(lands).collect::<Vec<_>>(), [None, Some(0xB0)]);
    }

    #[test]
    fn cross_port_same_register_and_bank_the_lower_port_wins_with_its_own_data() {
        // issue order does not matter: the later writer (port 0) wins, then the earlier does
        for (wa, wb) in [(1u64, 2u64), (2, 1)] {
            let all = [chunk(wa, 0, 0, 0, 8, 0xA0), chunk(wb, 1, 0, 0, 8, 0xB0)];
            let r = resolve(&all);
            assert_eq!(r.iter().map(lands).collect::<Vec<_>>(), [Some(0xA0), None]);
        }
    }

    #[test]
    fn cross_port_different_registers_both_land() {
        let all = [chunk(1, 0, 0, 0, 8, 0xA0), chunk(2, 1, 0, 0, 4, 0xB0)];
        let r = resolve(&all);
        assert_eq!(r.iter().map(lands).collect::<Vec<_>>(), [Some(0xA0), Some(0xB0)]);
    }

    #[test]
    fn cross_port_same_register_other_bank_both_land() {
        let all = [chunk(1, 0, 0, 0, 8, 0xA0), chunk(2, 1, 0, 1, 8, 0xB0)];
        let r = resolve(&all);
        assert_eq!(r.iter().map(lands).collect::<Vec<_>>(), [Some(0xA0), Some(0xB0)]);
    }
}
