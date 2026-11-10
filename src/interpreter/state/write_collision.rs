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
//!    takes its data from lane `p` when some writer drives it, else from the lane of the chunk's
//!    own ordinal, else from lane 0, and goes to the
//!    register of the latest-issued writer that targets bank `p`; the chunks of earlier writers
//!    on that bank are dropped. Banks nobody targets are untouched. A port with a single writer
//!    passes through unchanged. In the accumulator file (and beside a cascade read in either
//!    file) a narrow writer on an odd bank drives lane 1 as well as lane 0.
//! 2. **Different ports, same landing cycle.** Only chunks on the same register and the same
//!    bank conflict: the lower-indexed port wins, with its own data. Chunks on different
//!    registers both land, even on the same bank.
//! 3. **The B load channel.** `vldb` and its forms hold no write port in the tables. In silicon
//!    their result shares the bus of the vector file's last write port without taking part in
//!    rule 1: landing with a write through that port on the same register and bank, the two are
//!    ORed (both chunks carry the OR); with writes to other registers or banks nothing happens;
//!    against a lower-indexed port rule 2 applies (it loses).
//!    `vunpack`, whose tables write port is a borrowed read port, writes over the same channel
//!    (probe13; derived in `decode::ports`).
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
    /// The write is a `vmov x/bm, SCD` cascade read. Silicon (probe3,
    /// `build/experiments/cascade-collisions/probe3`): beside a cascade read, a narrow
    /// (single-chunk) writer on an odd bank also drives lane 1 of the port.
    pub cascade_read: bool,
    /// The write comes from the B load channel (rule 3 of the module docs); `port` is then the
    /// write port whose bus it shares.
    pub b_channel: bool,
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
    /// The chunk belongs to a cascade read ([`WriteTag::cascade_read`]).
    pub cascade: bool,
    /// The chunk is written to the accumulator file, where a narrow (single-chunk) writer on an
    /// odd bank also drives lane 1 of its port. Silicon (`gen_acc_timing_probe.py --waw`,
    /// `build/experiments/harness/cfloat/ldtiming*`): a later-issued `vlda amlh0` or
    /// `vmov amlh0, q1` landing with a `vups` / `vlda.ups` of the whole `bml0` puts its data in
    /// both banks of `bml0`; on bank 0 (`amll0`) it leaves bank 1 alone. In the vector file a
    /// narrow `wh` writer does not (collision probe6).
    pub acc_file: bool,
    /// The chunk comes from the B load channel ([`WriteTag::b_channel`]).
    pub b_channel: bool,
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
    let mut w: Vec<u64> = chunks
        .iter()
        .filter(|c| c.port == port && !c.b_channel)
        .map(|c| c.writer)
        .collect();
    w.sort_unstable();
    w.dedup();
    w
}

fn resolve_port(chunks: &[Chunk], port: u16, out: &mut [Landing]) {
    resolve_port_writers(chunks, port, out);
    // rule 3: a B-channel chunk and a landing chunk of the port on its register and bank are ORed
    for (i, b) in chunks.iter().enumerate().filter(|(_, c)| c.port == port && c.b_channel) {
        for j in 0..chunks.len() {
            let c = &chunks[j];
            if c.port == port && !c.b_channel && out[j].lands && c.reg == b.reg && c.bank == b.bank {
                let ored: [u32; 8] = std::array::from_fn(|k| out[i].data[k] | out[j].data[k]);
                out[i].data = ored;
                out[j].data = ored;
            }
        }
    }
}

fn resolve_port_writers(chunks: &[Chunk], port: u16, out: &mut [Landing]) {
    if writers_of(chunks, port).len() < 2 {
        return;
    }
    let on_port = || chunks.iter().enumerate().filter(|(_, c)| c.port == port && !c.b_channel);
    // In the accumulator file, and in either file beside a cascade read, a narrow writer on an
    // odd bank also drives lane 1 (see `Chunk::acc_file`, `WriteTag::cascade_read`); otherwise
    // lane k belongs to ordinal k alone.
    let cascade_present = on_port().any(|(_, c)| c.cascade);
    let chunks_of = |w: u64| on_port().filter(|(_, c)| c.writer == w).count();
    let drives = |c: &Chunk, k: u8| {
        c.ordinal == k
            || ((cascade_present || c.acc_file)
                && k == 1
                && !c.cascade
                && c.ordinal == 0
                && c.bank & 1 == 1
                && chunks_of(c.writer) == 1)
    };
    // lane k: the data of the latest-issued writer that drives it
    let lane = |k: u8| -> Option<[u32; 8]> {
        on_port()
            .filter(|(_, c)| drives(c, k))
            .max_by_key(|(_, c)| c.writer)
            .map(|(_, c)| c.data)
    };
    for (i, c) in on_port() {
        let latest_on_bank = on_port().filter(|(_, o)| o.bank == c.bank).map(|(_, o)| o.writer).max();
        if latest_on_bank != Some(c.writer) {
            out[i].lands = false;
            continue;
        }
        out[i].data = lane(c.bank).or_else(|| lane(c.ordinal)).or_else(|| lane(0)).unwrap_or(c.data);
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
        Chunk {
            writer,
            port,
            ordinal,
            bank,
            reg,
            data: [tag; 8],
            cascade: false,
            acc_file: false,
            b_channel: false,
        }
    }

    // Silicon (`tools/gen_vec_load_timing_probe.py tie` and `--waw`, harness/ldvec/tie1, vt2):
    // a `vldb` landing with a move through the last port: ORed on the same register and bank,
    // untouched elsewhere; the first port beats it.
    #[test]
    fn a_b_channel_write_is_ored_with_the_last_port_on_its_register_and_bank() {
        const WA: u16 = 1;
        const WM: u16 = 2;
        let b = |bank, reg, tag| Chunk { b_channel: true, ..chunk(1, WM, 0, bank, reg, tag) };
        let r = resolve(&[b(0, 8, 0x0100), chunk(2, WM, 0, 0, 8, 0x0003)]);
        assert_eq!(r.iter().map(lands).collect::<Vec<_>>(), [Some(0x0103), Some(0x0103)]);
        // another register on the same bank, the same register on another bank: both intact
        let r = resolve(&[b(0, 8, 0x0100), chunk(2, WM, 0, 0, 16, 0x0003)]);
        assert_eq!(r.iter().map(lands).collect::<Vec<_>>(), [Some(0x0100), Some(0x0003)]);
        let r = resolve(&[b(0, 8, 0x0100), chunk(2, WM, 0, 1, 9, 0x0003)]);
        assert_eq!(r.iter().map(lands).collect::<Vec<_>>(), [Some(0x0100), Some(0x0003)]);
        // the first port wins the register bank outright
        let r = resolve(&[b(0, 8, 0x0100), chunk(2, WA, 0, 0, 8, 0x0003)]);
        assert_eq!(r.iter().map(lands).collect::<Vec<_>>(), [None, Some(0x0003)]);
    }

    fn lands(l: &Landing) -> Option<u32> {
        l.lands.then(|| l.data[0])
    }

    // Silicon (cascade-collisions probe3, `vmov bmh, SCD` against `vmov bm, bm`): a chunk on a
    // bank no writer drives a lane for takes the lane of its own ordinal before lane 0.
    #[test]
    fn an_undriven_bank_takes_the_lane_of_its_own_ordinal() {
        // A (earlier): bmh, banks 2,3, ordinals 0,1. B (later): bm, banks 0,1, ordinals 0,1.
        let all = [
            chunk(1, 0, 0, 2, 7, 0xA0),
            chunk(1, 0, 1, 3, 7, 0xA1),
            chunk(2, 0, 0, 0, 5, 0xB0),
            chunk(2, 0, 1, 1, 5, 0xB1),
        ];
        let r = resolve(&all);
        assert_eq!(r.iter().map(lands).collect::<Vec<_>>(), [Some(0xB0), Some(0xB1), Some(0xB0), Some(0xB1)]);
    }

    // Silicon (probe3, `vmov bm, SCD` against `vmov am.., q`): beside a cascade read a narrow
    // writer on an odd bank also drives lane 1; on an even bank it does not.
    #[test]
    fn beside_a_cascade_read_a_narrow_odd_bank_writer_drives_lane_one() {
        let scd = |ordinal, bank, tag| Chunk { cascade: true, ..chunk(1, 0, ordinal, bank, 4, tag) };
        // B (later) on bank 3: the read's second chunk (bank 1) takes B's data, B keeps its own
        let all = [scd(0, 0, 0xA0), scd(1, 1, 0xA1), chunk(2, 0, 0, 3, 11, 0xB0)];
        let r = resolve(&all);
        assert_eq!(r.iter().map(lands).collect::<Vec<_>>(), [Some(0xB0), Some(0xB0), Some(0xB0)]);
        // B on bank 2: lane 1 stays the read's, its second chunk keeps its own data
        let all = [scd(0, 0, 0xA0), scd(1, 1, 0xA1), chunk(2, 0, 0, 2, 11, 0xB0)];
        let r = resolve(&all);
        assert_eq!(r.iter().map(lands).collect::<Vec<_>>(), [Some(0xB0), Some(0xA1), Some(0xB0)]);
        // without a cascade read the same pair is the plain rule: bank 3 is no lane of A's
        let plain = [chunk(1, 0, 0, 0, 4, 0xA0), chunk(1, 0, 1, 1, 4, 0xA1), chunk(2, 0, 0, 3, 11, 0xB0)];
        let r = resolve(&plain);
        assert_eq!(r.iter().map(lands).collect::<Vec<_>>(), [Some(0xB0), Some(0xA1), Some(0xB0)]);
    }

    // Silicon (`gen_acc_timing_probe.py --waw`, cfloat/ldtiming and ldtiming2: `vlda amlh0` or
    // `vmov amlh0, q1` issued after a `vlda.ups` / `vups` of `bml0`, landing in the same cycle on
    // CM_WM_PORT): in the accumulator file a narrow writer on an odd bank also drives lane 1, so
    // both banks of `bml0` take its data; on bank 0 it leaves lane 1 to the earlier writer.
    #[test]
    fn in_the_accumulator_file_a_narrow_odd_bank_writer_drives_lane_one() {
        let acc =
            |writer, ordinal, bank, tag| Chunk { acc_file: true, ..chunk(writer, 0, ordinal, bank, 0, tag) };
        let ups = [acc(1, 0, 0, 0xA0), acc(1, 1, 1, 0xA1)];
        let r = resolve(&[ups[0], ups[1], acc(2, 0, 1, 0xB0)]);
        assert_eq!(r.iter().map(lands).collect::<Vec<_>>(), [Some(0xB0), None, Some(0xB0)]);
        let r = resolve(&[ups[0], ups[1], acc(2, 0, 0, 0xB0)]);
        assert_eq!(r.iter().map(lands).collect::<Vec<_>>(), [None, Some(0xA1), Some(0xB0)]);
        // the same chunks in the vector file: lane 1 stays the earlier writer's
        let r =
            resolve(&[chunk(1, 0, 0, 0, 0, 0xA0), chunk(1, 0, 1, 1, 0, 0xA1), chunk(2, 0, 0, 1, 0, 0xB0)]);
        assert_eq!(r.iter().map(lands).collect::<Vec<_>>(), [Some(0xB0), None, Some(0xA1)]);
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
