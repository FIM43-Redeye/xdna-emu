//! Lock request and grant timing of DMA BDs, observed the way the trace sees it.
//!
//! The rig steps one DMA engine and its tile through the same per-cycle order the array
//! uses (lock requests submitted, the tile's lock arbiter resolved at the cycle, the engine
//! stepped) and collects what the memory-module trace would carry: the lock acquire and
//! release stamps of the arbiter, the channel's STALLED_LOCK level edges and its
//! FINISHED_BD pulses.

use super::*;
use crate::interpreter::state::EventType;

/// What a run shows on the memory-module trace, by cycle.
#[derive(Debug, Default)]
struct Trace {
    /// (cycle, lock id) of every granted acquire (LOCK_n_ACQ_GE).
    acquires: Vec<(u64, u8)>,
    /// (cycle, lock id) of every release (LOCK_n_REL).
    releases: Vec<(u64, u8)>,
    /// STALLED_LOCK intervals of `channel` as [rise, last stalled cycle].
    stalls: Vec<(u64, u64)>,
    /// FINISHED_BD pulses of `channel`.
    finishes: Vec<u64>,
}

/// One DMA engine and its tile, stepped cycle by cycle in the array's order.
struct Rig {
    eng: DmaEngine,
    tile: Tile,
    host: HostMemory,
    channel: u8,
    cycle: u64,
    stall_from: Option<u64>,
    trace: Trace,
}

impl Rig {
    fn new(eng: DmaEngine, tile: Tile, channel: u8) -> Self {
        Self {
            eng,
            tile,
            host: HostMemory::new(),
            channel,
            cycle: 0,
            stall_from: None,
            trace: Trace::default(),
        }
    }

    /// One cycle. `before_resolve` runs after the channels submitted their lock requests and
    /// before the tile's arbiter resolves (a core release lands here); `after_step` runs after
    /// the engine stepped (an inline DMA release of another channel lands here).
    fn step_with(&mut self, before_resolve: impl FnOnce(&mut Tile), after_step: impl FnOnce(&mut Tile)) {
        self.cycle += 1;
        let cycle = self.cycle;
        self.eng.set_current_cycle(cycle);
        self.eng.reset_cycle_drain_counters();
        self.eng.submit_lock_requests(&mut self.tile, &mut NeighborTiles::empty());
        before_resolve(&mut self.tile);
        self.tile.resolve_lock_requests(cycle);
        self.eng.step(&mut self.tile, &mut NeighborTiles::empty(), &mut self.host);
        after_step(&mut self.tile);
        while self.eng.pop_stream_out_for_channel(self.channel).is_some() {}
        for (c, e) in self.tile.mem_trace_pending.drain(..) {
            match e {
                EventType::LockAcquire { lock_id } => self.trace.acquires.push((c, lock_id)),
                EventType::LockRelease { lock_id } => self.trace.releases.push((c, lock_id)),
                _ => {}
            }
        }
        for (c, e) in self.eng.drain_trace_events() {
            match e {
                EventType::DmaStalledLock { channel, active } if channel == self.channel => {
                    if active {
                        self.stall_from = Some(c);
                    } else if let Some(from) = self.stall_from.take() {
                        self.trace.stalls.push((from, c - 1));
                    }
                }
                EventType::DmaFinishedBd { channel } if channel == self.channel => {
                    self.trace.finishes.push(c)
                }
                _ => {}
            }
        }
    }

    fn step(&mut self) {
        self.step_with(|_| {}, |_| {});
    }

    fn run_until(&mut self, cycle: u64) {
        while self.cycle < cycle {
            self.step();
        }
    }
}

// --------------------------------------------------------------------------------------------
// Tiles whose lock timing has not been measured: pins of the emulator's own behaviour.
// --------------------------------------------------------------------------------------------

/// A BD on `channel` that acquires `lock` (>= 1) and is released by an arbiter-path release
/// at cycle 60 (or finds the lock free from the start). Returns (the acquire's stamp, the
/// channel's STALLED_LOCK intervals, its FINISHED_BD cycles).
fn pinned_grant(
    mut rig: Rig,
    lock: u8,
    free: bool,
    words_in: Option<u8>,
) -> (u64, Vec<(u64, u64)>, Vec<u64>) {
    if free {
        rig.tile.locks[lock as usize].value = 1;
    }
    for _ in 0..400 {
        let release_now = !free && rig.cycle + 1 == 60;
        rig.step_with(
            |t| {
                if release_now {
                    t.defer_core_lock_release(lock as usize, 1);
                }
            },
            |_| {},
        );
        if let Some(ch) = words_in {
            rig.eng.push_stream_in(StreamData { data: 7, tlast: false, channel: ch });
        }
    }
    let acq = rig.trace.acquires.iter().find(|&&(_, l)| l == lock).expect("granted").0;
    (acq, rig.trace.stalls, rig.trace.finishes)
}

/// Memtile lock timing is not measured: the compute-tile request and grant law does not
/// apply to it. These are the emulator's own cycles before that law was modelled, pinned so
/// that a change to the compute path cannot move them.
#[test]
fn memtile_mm2s_lock_grant_is_unchanged_by_the_compute_lock_law() {
    for (free, expected) in [(false, (60, vec![(1, 59)], vec![84])), (true, (1, vec![], vec![25]))] {
        let mut eng = DmaEngine::new_mem_tile(1, 1);
        let own_lock = 64 + 3; // the memtile's own lock 3 in the BD lock address space
        eng.configure_bd(0, BdConfig::simple_1d(0x400, 64).with_acquire(own_lock, -1))
            .unwrap();
        eng.start_channel(6, 0).unwrap();
        let got = pinned_grant(Rig::new(eng, Tile::mem_tile(1, 1), 6), 3, free, None);
        assert_eq!(got, expected, "memtile MM2S, lock free = {free}");
    }
}

/// Shim lock timing is not measured: pinned as for the memtile (the acquire and the stall;
/// this rig gives the shim MM2S no host path to finish on).
#[test]
fn shim_mm2s_lock_grant_is_unchanged_by_the_compute_lock_law() {
    for (free, expected) in [(false, (60, vec![(1, 59)], vec![])), (true, (1, vec![], vec![]))] {
        let mut eng = DmaEngine::new_shim_tile(1, 0);
        eng.configure_bd(0, BdConfig::simple_1d(0x8000_0000, 64).with_acquire(3, -1))
            .unwrap();
        eng.start_channel(2, 0).unwrap();
        let mut rig = Rig::new(eng, Tile::shim(1, 0), 2);
        rig.host.allocate_region("in".to_string(), 0x8000_0000, 64).unwrap();
        let got = pinned_grant(rig, 3, free, None);
        assert_eq!(got, expected, "shim MM2S, lock free = {free}");
    }
}

/// A compute-tile S2MM's acquire is not measured (the law was measured on MM2S BDs): pinned.
#[test]
fn compute_s2mm_lock_grant_is_unchanged_by_the_compute_lock_law() {
    for (free, expected) in [(false, (60, vec![(1, 59)], vec![72])), (true, (1, vec![], vec![17]))] {
        let mut eng = DmaEngine::new_compute_tile(1, 2);
        eng.configure_bd(0, BdConfig::simple_1d(0x400, 64).with_acquire(3, -1)).unwrap();
        eng.start_channel(0, 0).unwrap();
        let got = pinned_grant(Rig::new(eng, Tile::compute(1, 2), 0), 3, free, Some(0));
        assert_eq!(got, expected, "compute S2MM, lock free = {free}");
    }
}
// --------------------------------------------------------------------------------------------
// Compute-tile MM2S BD chains: the probe designs of `tools/gen_dma_lock_probe.py` (sync and
// chain families) on the emulator's own channel. Every expected number is a silicon
// measurement (NPU1, 2026-11), in the frame of the head BD's grant g0 (the cycle its
// STALLED_LOCK falls).
// --------------------------------------------------------------------------------------------

/// Lock ids of the rig: the head waits on START, the waiting (last) BD on A, a core-released
/// mid BD on B, free-lock mids on FREE + i.
const LOCK_A: u8 = 0;
const LOCK_START: u8 = 1;
const LOCK_B: u8 = 2;
const LOCK_FREE: u8 = 3;
/// The head's lock is set this long after the channel starts: the channel is long parked.
const HEAD_RELEASE: u64 = 100;

/// A BD between the head and the waiting BD.
#[derive(Debug, Clone, Copy)]
enum Mid {
    /// No lock.
    Plain(u32),
    /// Acquires B, which the core releases at the row's RB.
    Core(u32),
    /// Acquires its own lock, free from the start.
    Free(u32),
}

/// Parse the probe tool's mid notation: "4c 16" = a 4-word BD on a core-released lock, then a
/// lock-free 16-word BD; "4f" = a lock that is free.
fn mids(spec: &str) -> Vec<Mid> {
    spec.split_whitespace()
        .map(|m| match m.strip_suffix('c') {
            Some(n) => Mid::Core(n.parse().unwrap()),
            None => match m.strip_suffix('f') {
                Some(n) => Mid::Free(n.parse().unwrap()),
                None => Mid::Plain(m.parse().unwrap()),
            },
        })
        .collect()
}

/// Who makes a lock available, and how the rig applies it.
#[derive(Debug, Clone, Copy)]
enum Releaser {
    /// The core: a release through the tile's lock arbiter in cycle g0 + r (LOCK_0_REL at g0 + r).
    Core(i64),
    /// An S2MM BD's release in cycle g0 + r: applied after that cycle's arbitration, as the
    /// DMA's inline release is.
    S2mm(i64),
    /// Free from the start.
    Free,
}

impl Releaser {
    /// Apply the release to `lock` if it happens in cycle `cycle` (g0-relative): before the
    /// arbitration when `before_resolve`, after the engine's step otherwise.
    fn apply(self, tile: &mut Tile, lock: u8, cycle: i64, before_resolve: bool) {
        match self {
            Releaser::Core(r) if before_resolve && r == cycle => {
                tile.defer_core_lock_release(lock as usize, 1)
            }
            Releaser::S2mm(r) if !before_resolve && r == cycle => {
                let _ = tile.locks[lock as usize].release_with_value(1);
            }
            _ => {}
        }
    }
}

/// What the waiting BD's channel showed, g0-relative.
#[derive(Debug, PartialEq, Eq)]
struct ChainObs {
    /// The waiting BD's grant: its LOCK_0_ACQ_GE + 1.
    grant: i64,
    /// The cycle from which the channel holds the waiting BD's lock (silicon: STALLED_LOCK
    /// falls, or ACQ_GE + 1 when it never rose).
    held: i64,
    /// Every STALLED_LOCK interval after the head's.
    stalls: Vec<(i64, i64)>,
    /// Every FINISHED_BD, the head's included.
    finishes: Vec<i64>,
}

/// Run head (`head` words on START) -> `mids` -> the waiting BD (`big` words on A), with B
/// made available by `b` (`Mid::Core`), on the emulator's compute MM2S 0.
fn run_chain(head: u32, mids: &[Mid], big: u32, b: Releaser, a: Releaser) -> ChainObs {
    let mut eng = DmaEngine::new_compute_tile(1, 2);
    let mut tile = Tile::compute(1, 2);
    let mut addr = 0x400u64;
    let mut bds = vec![BdConfig::simple_1d(addr, head * 4).with_acquire(LOCK_START, -1)];
    addr += 0x400;
    for (i, m) in mids.iter().enumerate() {
        let bd = match *m {
            Mid::Plain(n) => BdConfig::simple_1d(addr, n * 4),
            Mid::Core(n) => BdConfig::simple_1d(addr, n * 4).with_acquire(LOCK_B, -1),
            Mid::Free(n) => {
                let lock = LOCK_FREE + i as u8;
                tile.locks[lock as usize].value = 1;
                BdConfig::simple_1d(addr, n * 4).with_acquire(lock, -1)
            }
        };
        bds.push(bd);
        addr += 0x400;
    }
    bds.push(BdConfig::simple_1d(addr, big * 4).with_acquire(LOCK_A, -1));
    for (releaser, lock) in [(a, LOCK_A), (b, LOCK_B)] {
        if matches!(releaser, Releaser::Free) {
            tile.locks[lock as usize].value = 1;
        }
    }
    let last = bds.len() - 1;
    for (i, bd) in bds.into_iter().enumerate() {
        let bd = if i < last { bd.with_next(i as u8 + 1) } else { bd };
        eng.configure_bd(i as u8, bd).unwrap();
    }
    eng.start_channel(2, 0).unwrap();
    let mut rig = Rig::new(eng, tile, 2);
    rig.run_until(HEAD_RELEASE - 1);
    rig.tile.locks[LOCK_START as usize].value = 1;
    // The head's grant: its STALLED_LOCK falls.
    while rig.trace.stalls.is_empty() {
        rig.step();
        assert!(rig.cycle < 1000, "the head was never granted");
    }
    let g0 = rig.trace.stalls[0].1 as i64 + 1;
    let mut held = None;
    for _ in 0..400 {
        let next = (rig.cycle + 1) as i64 - g0;
        rig.step_with(
            |t| {
                a.apply(t, LOCK_A, next, true);
                b.apply(t, LOCK_B, next, true);
            },
            |t| {
                a.apply(t, LOCK_A, next, false);
                b.apply(t, LOCK_B, next, false);
            },
        );
        let wc = &rig.eng.channels[2].word_clock;
        if let Some(g) = wc
            .bds()
            .find(|bd| bd.spec.acquire.is_some_and(|(l, _)| l == LOCK_A))
            .and_then(|bd| bd.granted_at)
        {
            held.get_or_insert(g as i64 - g0);
        }
    }
    let rel = |c: u64| c as i64 - g0;
    let grant = rig
        .trace
        .acquires
        .iter()
        .find(|&&(_, l)| l == LOCK_A)
        .map(|&(c, _)| rel(c) + 1)
        .expect("A granted");
    ChainObs {
        grant,
        held: held.expect("A held"),
        stalls: rig.trace.stalls[1..].iter().map(|&(a, b)| (rel(a), rel(b))).collect(),
        finishes: rig.trace.finishes.iter().map(|&c| rel(c)).collect(),
    }
}

/// The lock request law (`LockPort`): a BD's lock is requested at Q = max(g + 3, T + 3), g the
/// grant of the channel's previous lock BD, T the cycle the BD before it has its third-to-last
/// word sequenced -- while the channel is still on the BDs before. Silicon (NPU1, 2026-11,
/// `tools/gen_dma_lock_probe.py` chain family b1 / b3 / b4 and the sync family; g0 frame) with
/// the waiting BD's lock free on request: it is granted at Q, and every FINISHED_BD follows.
/// The core releases the mid lock B at g0 + 14 (silicon: B granted at g1 = g0 + 16). Rows: a 4-, 2- or 16-word
/// lock BD, then a lock-free BD of N words (b1n13: the grant at g1 + 3 comes 5 cycles before
/// the 13-word BD finishes; N = 15: g1 + 6; N = 20: g1 + 11; N = 32: g1 + 23).
#[test]
fn a_free_lock_is_granted_at_its_request_while_the_channel_is_on_the_bds_before() {
    #[rustfmt::skip]
    let rows: &[(&str, i64, &[i64])] = &[
        // mids, grant of A (silicon), FINISHED_BD (silicon)
        ("4c", 19, &[3, 19, 72]),               // b1n00 q00_00 p2
        ("4c 13", 19, &[3, 19, 24, 86]),        // b1n13 q00_00 p1
        ("4c 15", 22, &[3, 19, 24, 87]),        // b1n15 q00_00 p1
        ("4c 20", 27, &[3, 19, 28, 92]),        // b1n20 q00_01 p1
        ("4c 32", 39, &[3, 19, 40, 104]),       // b1n32 q00_00 p1
        ("2c 11", 19, &[3, 19, 23, 84]),        // b4p02n11 q00_00 p1
        ("2c 14", 19, &[3, 19, 24, 86]),        // b4p02n14 q00_00 p2
        ("16c 8", 27, &[3, 22, 28, 92]),        // b3p16n08 q00_00 p2
    ];
    for &(spec, grant, finishes) in rows {
        let obs = run_chain(4, &mids(spec), 64, Releaser::Core(14), Releaser::Free);
        assert_eq!((obs.held, obs.finishes.as_slice()), (grant, finishes), "{spec}");
    }
    // Sync family, the lock free long before: lsfree is granted at g0 + 3 (its one-cycle
    // STALLED_LOCK at g0 + 2); ltfree (N = 32) shows no stall, its grant is not observed.
    let obs = run_chain(4, &[], 64, Releaser::Free, Releaser::Free);
    assert_eq!((obs.held, obs.finishes.as_slice()), (3, &[3i64, 56][..]), "lsfree");
    let obs = run_chain(4, &[Mid::Plain(32)], 64, Releaser::Free, Releaser::Free);
    assert_eq!(obs.finishes, [3, 24, 88], "ltfree");
}

/// The lock grant law (`LockPort`) on the sync design (silicon, NPU1 2026-11,
/// `tools/gen_dma_lock_probe.py` sync family, g0 frame; the core releases A at g0 + R, the
/// LOCK_0_REL stamp): the grant G is the first cycle >= max(Q, R + 2) with (G - Q) mod 3 != 2
/// and G != DEAD, Q the request; LOCK_0_ACQ_GE fires at G - 1 (the waiting BD's grant column
/// is ACQ_GE + 1 when the trace shows it, the STALLED_LOCK fall otherwise); FINISHED_BD
/// follows. T1..T5 of the measured table are among these rows (pre2 32: R 25 -> 27, 26 -> 29,
/// 27 -> 29, 30 -> 32; pre2 16: 14 -> 17; pre2 5..8: 15 -> 18).
#[test]
fn the_sync_designs_grant_lands_on_silicons_cycle() {
    #[rustfmt::skip]
    let rows: &[(u32, i64, Option<i64>, &[(i64, i64)], &[i64])] = &[
        // N, R, grant, STALLED_LOCK after the head, FINISHED_BD (all silicon)
        (0, 14, Some(16), &[(2, 15)], &[3, 68]), // ls00 s1.hw
        (0, 23, Some(25), &[(2, 24)], &[3, 77]), // ls08 s1.hw
        (0, 32, Some(34), &[(2, 33)], &[3, 86]), // ls17 s1.hw
        (0, 40, Some(42), &[(2, 41)], &[3, 94]), // ls26 s1.hw
        (0, 54, Some(57), &[(2, 56)], &[3, 109]), // ls40 s1.hw
        (5, 14, Some(16), &[(4, 15)], &[3, 6, 68]), // lt00 p2.hw
        (5, 18, Some(21), &[(4, 20)], &[3, 6, 73]), // lt03 p3.hw
        (5, 22, Some(24), &[(4, 23)], &[3, 6, 76]), // lt07 p1.hw
        (5, 25, Some(27), &[(4, 26)], &[3, 6, 79]), // lt10 p1.hw
        (5, 34, Some(36), &[(4, 35)], &[3, 6, 88]), // lt20 k1.hw
        (8, 14, Some(16), &[(4, 15)], &[3, 6, 68]), // lt00 p1.hw
        (8, 18, Some(21), &[(4, 20)], &[3, 6, 73]), // lt03 p2.hw
        (8, 21, Some(24), &[(4, 23)], &[3, 6, 76]), // lt06 p2.hw
        (8, 24, Some(27), &[(4, 26)], &[3, 6, 79]), // lt09 p1.hw
        (8, 34, Some(36), &[(4, 35)], &[3, 6, 88]), // lt20 k1.hw
        (13, 14, Some(16), &[(7, 15)], &[3, 8, 70]), // lt00 p1.hw
        (13, 17, Some(19), &[(7, 18)], &[3, 8, 71]), // lt03 p2.hw
        (13, 20, Some(22), &[(7, 21)], &[3, 8, 74]), // lt05 p1.hw
        (13, 24, Some(27), &[(7, 26)], &[3, 8, 79]), // lt09 p2.hw
        (13, 28, Some(30), &[(7, 29)], &[3, 8, 82]), // lt13 p1.hw
        (14, 14, Some(16), &[(7, 15)], &[3, 8, 70]), // lt00 p1.hw
        (14, 16, Some(18), &[(7, 17)], &[3, 8, 70]), // lt01 p1.hw
        (14, 20, Some(22), &[(7, 21)], &[3, 8, 74]), // lt05 p1.hw
        (14, 24, Some(27), &[(7, 26)], &[3, 8, 79]), // lt09 p1.hw
        (14, 27, Some(30), &[(7, 29)], &[3, 8, 82]), // lt12 p1.hw
        (16, 14, Some(17), &[(9, 16)], &[3, 8, 72]), // lt00 p1.hw
        (16, 19, Some(21), &[(9, 20)], &[3, 8, 73]), // lt04 p1.hw
        (16, 22, Some(24), &[(9, 23)], &[3, 8, 76]), // lt08 p1.hw
        (16, 25, Some(27), &[(9, 26)], &[3, 8, 79]), // lt10 p1.hw
        (16, 35, Some(38), &[(9, 37)], &[3, 8, 90]), // lt20 k1.hw
        (17, 14, Some(17), &[(10, 16)], &[3, 9, 73]), // lt00 p1.hw
        (17, 17, Some(20), &[(10, 19)], &[3, 9, 73]), // lt03 p1.hw
        (17, 20, Some(23), &[(10, 22)], &[3, 9, 75]), // lt05 p1.hw
        (17, 24, Some(26), &[(10, 25)], &[3, 9, 78]), // lt09 p2.hw
        (17, 28, Some(30), &[(10, 29)], &[3, 9, 82]), // lt13 p1.hw
        (20, 14, Some(17), &[(13, 16)], &[3, 12, 76]), // lt00 p1.hw
        (20, 18, Some(20), &[(13, 19)], &[3, 12, 76]), // lt03 p1.hw
        (20, 22, Some(24), &[(13, 23)], &[3, 12, 76]), // lt07 p1.hw
        (20, 25, Some(27), &[(13, 26)], &[3, 12, 79]), // lt11 p1.hw
        (20, 34, Some(36), &[(13, 35)], &[3, 12, 88]), // lt20 k1.hw
        (32, 14, None, &[], &[3, 24, 88]), // lt00 s2.hw
        (32, 23, Some(26), &[(25, 25)], &[3, 24, 88]), // lt09 s1.hw
        (32, 32, Some(35), &[(25, 34)], &[3, 24, 88]), // lt17 s2.hw
        (32, 43, Some(45), &[(25, 44)], &[3, 24, 97]), // lt28 s1.hw
        (32, 54, Some(56), &[(25, 55)], &[3, 24, 108]), // lt40 s1.hw
    ];
    for &(n, r, grant, stalls, finishes) in rows {
        let mids = if n == 0 { vec![] } else { vec![Mid::Plain(n)] };
        let obs = run_chain(4, &mids, 64, Releaser::Free, Releaser::Core(r));
        assert_eq!(obs.finishes, finishes, "N = {n}, R = {r}: finishes");
        assert_eq!(obs.stalls, stalls, "N = {n}, R = {r}: STALLED_LOCK");
        if let Some(g) = grant {
            assert_eq!((obs.held, obs.grant), (g, g), "N = {n}, R = {r}: grant (held, ACQ_GE + 1)");
        }
    }
}

/// The grant law on the chain family (silicon, NPU1 2026-11, `tools/gen_dma_lock_probe.py`
/// chain family, g0 frame): a mid BD on lock B that the core releases (b1: 4 words, b3: 8 /
/// 16, b4: 2 -- a partial granule), on a free lock (b2), a longer head (h), a 16-word waiting
/// BD (w16), and A released by an S2MM BD (s1, T10: R 38 -> G 41, 40 blocked). The core's
/// release of B is not traced: RB is taken as the core's event 0 + its nops + 5, the one
/// fitted input of these rows. Covers T6..T10 and T12 (b1n16: a free lock granted at g1 + 8,
/// the request moved off the previous BD's release slot g1 + 7).
#[test]
fn the_chain_designs_grants_land_on_silicons_cycle() {
    #[rustfmt::skip]
    let rows: &[(u32, &str, u32, i64, i64, bool, i64, &[(i64, i64)], &[i64])] = &[
        // head, mids, waiting BD, RB, RA, A released by an S2MM, grant, stalls, finishes (silicon)
        (4, "4c", 64, 14, 16, false, 19, &[(2, 15), (18, 18)], &[3, 19, 72]), // b1n00 q00_00 p2.hw
        (4, "4c", 64, 15, 25, false, 27, &[(2, 17), (20, 26)], &[3, 21, 79]), // b1n00 q00_07 p1.hw
        (4, "4c", 64, 15, 35, false, 37, &[(2, 17), (20, 36)], &[3, 21, 89]), // b1n00 q00_17 p2.hw
        (4, "4c 8", 64, 14, 16, false, 19, &[(2, 15)], &[3, 19, 22, 80]), // b1n08 q00_00 p1.hw
        (4, "4c 8", 64, 14, 26, false, 28, &[(2, 15), (20, 27)], &[3, 19, 22, 80]), // b1n08 q00_09 p1.hw
        (4, "4c 8", 64, 14, 34, false, 37, &[(2, 15), (20, 36)], &[3, 19, 22, 89]), // b1n08 q00_17 p2.hw
        (4, "4c 13", 64, 14, 16, false, 19, &[(2, 15)], &[3, 19, 24, 86]), // b1n13 q00_00 p1.hw
        (4, "4c 13", 64, 15, 25, false, 27, &[(2, 17), (25, 26)], &[3, 21, 26, 88]), // b1n13 q00_07 p2.hw
        (4, "4c 13", 64, 15, 34, false, 36, &[(2, 17), (25, 35)], &[3, 21, 26, 88]), // b1n13 q00_16 p2.hw
        (4, "4c 15", 64, 14, 16, false, 22, &[(2, 15)], &[3, 19, 24, 87]), // b1n15 q00_00 p1.hw
        (4, "4c 15", 64, 14, 25, false, 28, &[(2, 15), (24, 27)], &[3, 19, 24, 87]), // b1n15 q00_08 p1.hw
        (4, "4c 15", 64, 14, 34, false, 37, &[(2, 15), (24, 36)], &[3, 19, 24, 89]), // b1n15 q00_17 p1.hw
        (4, "4c 16", 64, 14, 16, false, 24, &[(2, 15)], &[3, 19, 24, 88]), // b1n16 q00_00 p1.hw
        (4, "4c 16", 64, 14, 26, false, 28, &[(2, 15), (25, 27)], &[3, 19, 24, 88]), // b1n16 q00_09 p1.hw
        (4, "4c 16", 64, 15, 35, false, 38, &[(2, 17), (27, 37)], &[3, 21, 26, 90]), // b1n16 q00_17 p2.hw
        (4, "4c 20", 64, 14, 17, false, 27, &[(2, 15)], &[3, 19, 28, 92]), // b1n20 q00_01 p1.hw
        (4, "4c 20", 64, 14, 27, false, 30, &[(2, 15), (29, 29)], &[3, 19, 28, 92]), // b1n20 q00_10 p1.hw
        (4, "4c 20", 64, 15, 34, false, 36, &[(2, 17), (31, 35)], &[3, 21, 30, 94]), // b1n20 q00_16 p2.hw
        (4, "4c 32", 64, 14, 16, false, 39, &[(2, 15)], &[3, 19, 40, 104]), // b1n32 q00_00 p1.hw
        (4, "4c 32", 64, 14, 27, false, 39, &[(2, 15)], &[3, 19, 40, 104]), // b1n32 q00_10 p2.hw
        (4, "4c 32", 64, 15, 34, false, 41, &[(2, 17)], &[3, 21, 42, 106]), // b1n32 q00_16 p2.hw
        (4, "4f 8", 64, 14, 15, false, 18, &[(2, 2), (7, 17)], &[3, 6, 9, 70]), // b2n08 q00_00 p1.hw
        (4, "4f 8", 64, 14, 21, false, 24, &[(2, 2), (7, 23)], &[3, 6, 9, 76]), // b2n08 q00_06 p1.hw
        (4, "4f 8", 64, 15, 29, false, 31, &[(2, 2), (7, 30)], &[3, 6, 9, 83]), // b2n08 q00_13 p2.hw
        (4, "4f 16", 64, 14, 15, false, 17, &[(2, 2), (13, 16)], &[3, 6, 12, 76]), // b2n16 q00_00 p1.hw
        (4, "4f 16", 64, 15, 21, false, 23, &[(2, 2), (13, 22)], &[3, 6, 12, 76]), // b2n16 q00_05 p2.hw
        (4, "4f 16", 64, 15, 28, false, 30, &[(2, 2), (13, 29)], &[3, 6, 12, 82]), // b2n16 q00_12 p1.hw
        (4, "4f 20", 64, 14, 15, false, 18, &[(2, 2), (17, 17)], &[3, 6, 16, 80]), // b2n20 q00_00 p1.hw
        (4, "4f 20", 64, 14, 21, false, 24, &[(2, 2), (17, 23)], &[3, 6, 16, 80]), // b2n20 q00_06 p1.hw
        (4, "4f 20", 64, 15, 28, false, 30, &[(2, 2), (17, 29)], &[3, 6, 16, 82]), // b2n20 q00_12 p1.hw
        (4, "8c", 64, 14, 16, false, 19, &[(2, 15), (18, 18)], &[3, 20, 76]), // b3p08 q00_00 p1.hw
        (4, "8c", 64, 14, 25, false, 28, &[(2, 15), (18, 27)], &[3, 20, 80]), // b3p08 q00_08 p1.hw
        (4, "8c", 64, 14, 34, false, 37, &[(2, 15), (18, 36)], &[3, 20, 89]), // b3p08 q00_17 p2.hw
        (4, "16c", 64, 14, 16, false, 19, &[(2, 15)], &[3, 22, 84]), // b3p16 q00_00 p1.hw
        (4, "16c", 64, 15, 25, false, 27, &[(2, 17), (23, 26)], &[3, 24, 86]), // b3p16 q00_07 p1.hw
        (4, "16c", 64, 15, 34, false, 36, &[(2, 17), (23, 35)], &[3, 24, 88]), // b3p16 q00_16 p1.hw
        (4, "16c 8", 64, 14, 16, false, 27, &[(2, 15)], &[3, 22, 28, 92]), // b3p16n08 q00_00 p2.hw
        (4, "16c 8", 64, 14, 25, false, 27, &[(2, 15)], &[3, 22, 28, 92]), // b3p16n08 q00_08 p1.hw
        (4, "16c 8", 64, 14, 34, false, 36, &[(2, 15), (29, 35)], &[3, 22, 28, 92]), // b3p16n08 q00_17 p1.hw
        (4, "2c 11", 64, 14, 16, false, 19, &[(2, 15)], &[3, 19, 23, 84]), // b4p02n11 q00_00 p1.hw
        (4, "2c 11", 64, 14, 24, false, 26, &[(2, 15), (21, 25)], &[3, 19, 23, 84]), // b4p02n11 q00_07 p1.hw
        (4, "2c 11", 64, 15, 35, false, 37, &[(2, 17), (23, 36)], &[3, 21, 25, 89]), // b4p02n11 q00_17 p1.hw
        (4, "2c 13", 64, 14, 16, false, 19, &[(2, 15)], &[3, 19, 24, 86]), // b4p02n13 q00_00 p1.hw
        (4, "2c 13", 64, 15, 26, false, 28, &[(2, 17), (25, 27)], &[3, 21, 26, 88]), // b4p02n13 q00_08 p1.hw
        (4, "2c 13", 64, 15, 34, false, 36, &[(2, 17), (25, 35)], &[3, 21, 26, 88]), // b4p02n13 q00_16 p1.hw
        (4, "2c 14", 64, 14, 16, false, 19, &[(2, 15)], &[3, 19, 24, 86]), // b4p02n14 q00_00 p2.hw
        (4, "2c 14", 64, 14, 26, false, 28, &[(2, 15), (23, 27)], &[3, 19, 24, 86]), // b4p02n14 q00_09 p1.hw
        (4, "2c 14", 64, 14, 34, false, 37, &[(2, 15), (23, 36)], &[3, 19, 24, 89]), // b4p02n14 q00_17 p1.hw
        (4, "2c 15", 64, 14, 16, false, 22, &[(2, 15)], &[3, 19, 24, 87]), // b4p02n15 q00_00 p2.hw
        (4, "2c 15", 64, 14, 26, false, 28, &[(2, 15), (24, 27)], &[3, 19, 24, 87]), // b4p02n15 q00_09 p1.hw
        (4, "2c 15", 64, 14, 34, false, 37, &[(2, 15), (24, 36)], &[3, 19, 24, 89]), // b4p02n15 q00_17 p1.hw
        (4, "2c 16", 64, 14, 16, false, 24, &[(2, 15)], &[3, 19, 24, 88]), // b4p02n16 q00_00 p1.hw
        (4, "2c 16", 64, 14, 27, false, 30, &[(2, 15), (25, 29)], &[3, 19, 24, 88]), // b4p02n16 q00_10 p1.hw
        (4, "2c 16", 64, 14, 34, false, 36, &[(2, 15), (25, 35)], &[3, 19, 24, 88]), // b4p02n16 q00_17 p1.hw
        (4, "2c 17", 64, 14, 16, false, 24, &[(2, 15)], &[3, 19, 25, 89]), // b4p02n17 q00_00 p1.hw
        (4, "2c 17", 64, 14, 26, false, 28, &[(2, 15), (26, 27)], &[3, 19, 25, 89]), // b4p02n17 q00_09 p1.hw
        (4, "2c 17", 64, 15, 34, false, 36, &[(2, 17), (28, 35)], &[3, 21, 27, 91]), // b4p02n17 q00_16 p1.hw
        (8, "16", 64, 15, 16, false, 18, &[(13, 17)], &[4, 12, 76]), // h08n16 q00_00 p1.hw
        (8, "16", 64, 16, 22, false, 24, &[(13, 23)], &[4, 12, 76]), // h08n16 q00_05 p2.hw
        (8, "16", 64, 15, 29, false, 32, &[(13, 31)], &[4, 12, 84]), // h08n16 q00_13 p1.hw
        (16, "8", 64, 17, 18, false, 20, &[(13, 19)], &[6, 12, 76]), // h16n08 q00_00 p1.hw
        (16, "8", 64, 17, 23, false, 26, &[(13, 25)], &[6, 12, 78]), // h16n08 q00_05 p1.hw
        (16, "8", 64, 18, 31, false, 33, &[(13, 32)], &[6, 12, 85]), // h16n08 q00_12 p1.hw
        (16, "16", 64, 17, 18, false, 20, &[], &[6, 20, 84]), // h16n16 q00_00 p1.hw
        (16, "16", 64, 17, 26, false, 28, &[(21, 27)], &[6, 20, 84]), // h16n16 q00_08 p1.hw
        (16, "16", 64, 18, 31, false, 34, &[(21, 33)], &[6, 20, 86]), // h16n16 q00_12 p2.hw
        (4, "16", 16, 14, 15, false, 17, &[(9, 16)], &[3, 8, 24]), // w16n16 q00_00 p2.hw
        (4, "16", 16, 14, 21, false, 23, &[(9, 22)], &[3, 8, 29]), // w16n16 q00_06 p1.hw
        (4, "16", 16, 15, 28, false, 30, &[(9, 29)], &[3, 8, 36]), // w16n16 q00_12 p1.hw
        (4, "4c 16", 16, 15, 17, false, 26, &[(2, 17)], &[3, 21, 26, 42]), // w16b1n16 q00_00 p1.hw
        (4, "4c 16", 16, 14, 26, false, 28, &[(2, 15), (25, 27)], &[3, 19, 24, 40]), // w16b1n16 q00_09 p1.hw
        (4, "4c 16", 16, 15, 35, false, 38, &[(2, 17), (27, 37)], &[3, 21, 26, 44]), // w16b1n16 q00_17 p1.hw
        (4, "13", 64, 14, 35, true, 37, &[(7, 36)], &[3, 8, 89]), // s1n13 qs t1.hw
        (4, "13", 64, 15, 35, true, 37, &[(7, 36)], &[3, 8, 89]), // s1n13 qs t3.hw
        (4, "16", 64, 14, 38, true, 41, &[(9, 40)], &[3, 8, 93]), // s1n16 qs t1.hw
        (4, "20", 64, 14, 42, true, 44, &[(13, 43)], &[3, 12, 96]), // s1n20 qs t1.hw
        (4, "24", 64, 15, 46, true, 48, &[(17, 47)], &[3, 16, 100]), // s1n24 qs t1.hw
        (4, "32", 64, 14, 54, true, 56, &[(25, 55)], &[3, 24, 108]), // s1n32 qs t2.hw
        (4, "32", 64, 15, 54, true, 56, &[(25, 55)], &[3, 24, 108]), // s1n32 qs t1.hw
    ];
    for &(head, spec, big, rb, ra, s2mm, grant, stalls, finishes) in rows {
        let a = if s2mm {
            Releaser::S2mm(ra)
        } else {
            Releaser::Core(ra)
        };
        let obs = run_chain(head, &mids(spec), big, Releaser::Core(rb), a);
        assert_eq!(
            (obs.held, obs.grant, obs.stalls.as_slice(), obs.finishes.as_slice()),
            (grant, grant, stalls, finishes),
            "head {head}, {spec}, waiting {big}, RB {rb}, RA {ra}"
        );
    }
}

/// STALLED_LOCK of a compute MM2S lock BD (`LockPort::stall_level`): shown on [P, G - 1] with
/// P = max(D, Q, the previous BD's last access) - 1 (D its first granule fully sequenced), so it can rise before the channel
/// leaves the BD before, and a lock free on request shows one cycle. LOCK_SEL0_ACQ_GE fires
/// once per acquire at G - 1, a free lock's included. Silicon (NPU1, 2026-11,
/// `tools/gen_dma_lock_probe.py` sync and chain families, g0 frame): T5 (lsfree: one cycle at
/// g0 + 2, grant g0 + 3), T1 (pre2 32, A released at g0 + 23: one cycle at g0 + 25, grant
/// g0 + 26; at g0 + 22: no level), T12 (pre1 on a free lock: one cycle at g0 + 2).
#[test]
fn stalled_lock_rises_at_p_and_acq_ge_fires_the_cycle_before_the_grant() {
    let obs = run_chain(4, &[], 64, Releaser::Free, Releaser::Free);
    assert_eq!((obs.stalls.as_slice(), obs.grant, obs.held), (&[(2, 2)][..], 3, 3), "lsfree");
    let obs = run_chain(4, &[Mid::Plain(32)], 64, Releaser::Free, Releaser::Core(23));
    assert_eq!((obs.stalls.as_slice(), obs.grant, obs.held), (&[(25, 25)][..], 26, 26), "pre2 32, R 23");
    let obs = run_chain(4, &[Mid::Plain(32)], 64, Releaser::Free, Releaser::Core(22));
    assert_eq!(obs.stalls, [], "pre2 32, R 22");
    let obs = run_chain(4, &mids("4f 16"), 64, Releaser::Free, Releaser::Core(15));
    assert_eq!((obs.stalls.as_slice(), obs.grant), (&[(2, 2), (13, 16)][..], 17), "b2n16, R 15");
}

/// The rows of the measured table (T1..T4, T6, T8..T10, T12; silicon, NPU1 2026-11,
/// `tools/gen_dma_lock_probe.py` sync and chain families, g0 frame), each a measured row:
/// T1 pre2 32, R 25 -> 27, 26 -> 29, 27 -> 29, 30 -> 32, 33 -> 35; T2 pre2 16, R 14 -> 17,
/// 15 -> 17, 17 -> 20 (P 9, FINISHED 8 and 72); T3 pre2 5..8, R 15 / 16 -> 18 (FINISHED 6);
/// T4 pre2 13 (P 7, FINISHED 8, 70 when granted by 18); T6 a 4-word mid on B: R g1 + 2 ->
/// g1 + 4, g1 + 4 -> g1 + 6, g1 + 5 -> g1 + 9 (g1 + 7 its release slot, g1 + 8 the blocked
/// phase), g1 + 8 -> g1 + 10, free -> g1 + 3; T8 pre2 20: g1 + 11, 12, 14, pre2 16 free:
/// g1 + 8; T9 an 8-word mid: R g1 + 5 -> g1 + 7; T10 an S2MM releaser, R 38 -> 41; T12 a
/// free-lock mid: R 15 -> 17, 16 -> 18, 17 -> 20, 20 -> 23, 23 -> 26, 26 -> 29, 28 -> 30.
#[test]
fn the_measured_table_rows_land_on_silicons_cycles() {
    #[rustfmt::skip]
    let sync: &[(&str, u32, i64, Option<i64>, &[(i64, i64)], &[i64])] = &[
        // silicon row, N, R, grant, STALLED_LOCK after the head, FINISHED_BD
        ("lt10 s2.hw", 32, 25, Some(27), &[(25, 26)], &[3, 24, 88]),
        ("lt11 s3.hw", 32, 26, Some(29), &[(25, 28)], &[3, 24, 88]),
        ("lt12 s2.hw", 32, 27, Some(29), &[(25, 28)], &[3, 24, 88]),
        ("lt16 s2.hw", 32, 30, Some(32), &[(25, 31)], &[3, 24, 88]),
        ("lt18 s1.hw", 32, 33, Some(35), &[(25, 34)], &[3, 24, 88]),
        ("lt00 p1.hw", 16, 14, Some(17), &[(9, 16)], &[3, 8, 72]),
        ("lt01 p1.hw", 16, 15, Some(17), &[(9, 16)], &[3, 8, 72]),
        ("lt02 p1.hw", 16, 17, Some(20), &[(9, 19)], &[3, 8, 72]),
        ("lt00 p1.hw", 5, 15, Some(18), &[(4, 17)], &[3, 6, 70]),
        ("lt02 p1.hw", 5, 16, Some(18), &[(4, 17)], &[3, 6, 70]),
        ("lt00 p1.hw", 6, 15, Some(18), &[(4, 17)], &[3, 6, 70]),
        ("lt01 p2.hw", 6, 16, Some(18), &[(4, 17)], &[3, 6, 70]),
        ("lt01 p1.hw", 7, 15, Some(18), &[(4, 17)], &[3, 6, 70]),
        ("lt01 p2.hw", 7, 16, Some(18), &[(4, 17)], &[3, 6, 70]),
        ("lt00 p2.hw", 8, 15, Some(18), &[(4, 17)], &[3, 6, 70]),
        ("lt01 p1.hw", 8, 16, Some(18), &[(4, 17)], &[3, 6, 70]),
        ("lt00 p1.hw", 13, 14, Some(16), &[(7, 15)], &[3, 8, 70]),
        ("lt01 p1.hw", 13, 15, Some(18), &[(7, 17)], &[3, 8, 70]),
        ("lt02 p1.hw", 13, 16, Some(18), &[(7, 17)], &[3, 8, 70]),
    ];
    for &(row, n, r, grant, stalls, finishes) in sync {
        let mids = if n == 0 { vec![] } else { vec![Mid::Plain(n)] };
        let obs = run_chain(4, &mids, 64, Releaser::Free, Releaser::Core(r));
        assert_eq!((obs.stalls.as_slice(), obs.finishes.as_slice()), (stalls, finishes), "{row}");
        if let Some(g) = grant {
            assert_eq!((obs.held, obs.grant), (g, g), "{row}: grant");
        }
    }
    #[rustfmt::skip]
    let chain: &[(&str, u32, &str, u32, i64, i64, i64, &[(i64, i64)], &[i64])] = &[
        // row, head, mids, waiting BD, RB, RA, grant, stalls, finishes (silicon; RB the fitted proxy)
        ("T6 R g1+2: b1n00 q00_02 p2.hw", 4, "4c", 64, 15, 20, 22, &[(2, 17), (20, 21)], &[3, 21, 74]),
        ("T6 R g1+4: b1n00 q00_03 p1.hw", 4, "4c", 64, 14, 20, 22, &[(2, 15), (18, 21)], &[3, 19, 74]),
        ("T6 R g1+5: b1n00 q00_04 p1.hw", 4, "4c", 64, 14, 21, 25, &[(2, 15), (18, 24)], &[3, 19, 77]),
        ("T6 R g1+8: b1n00 q00_07 p2.hw", 4, "4c", 64, 14, 24, 26, &[(2, 15), (18, 25)], &[3, 19, 78]),
        ("T6 free: b1n00 q00_00 p1.hw", 4, "4c", 64, 15, 17, 21, &[(2, 17), (20, 20)], &[3, 21, 74]),
        ("T8 G g1+11: b1n20 q00_00 p1.hw", 4, "4c 20", 64, 15, 17, 29, &[(2, 17)], &[3, 21, 30, 94]),
        ("T8 G g1+12: b1n20 q00_09 p1.hw", 4, "4c 20", 64, 14, 26, 28, &[(2, 15)], &[3, 19, 28, 92]),
        ("T8 G g1+14: b1n20 q00_10 p1.hw", 4, "4c 20", 64, 14, 27, 30, &[(2, 15), (29, 29)], &[3, 19, 28, 92]),
        ("T8 free g1+8: b1n16 q00_00 p1.hw", 4, "4c 16", 64, 14, 16, 24, &[(2, 15)], &[3, 19, 24, 88]),
        ("T9 R g1+5: b3p08 q00_04 p2.hw", 4, "8c", 64, 14, 21, 23, &[(2, 15), (18, 22)], &[3, 20, 76]),
        ("T12 R 15: b2n16 q00_00 p1.hw", 4, "4f 16", 64, 14, 15, 17, &[(2, 2), (13, 16)], &[3, 6, 12, 76]),
        ("T12 R 16: b2n16 q00_01 p1.hw", 4, "4f 16", 64, 14, 16, 18, &[(2, 2), (13, 17)], &[3, 6, 12, 76]),
        ("T12 R 17: b2n16 q00_01 p2.hw", 4, "4f 16", 64, 15, 17, 20, &[(2, 2), (13, 19)], &[3, 6, 12, 76]),
        ("T12 R 20: b2n16 q00_04 p1.hw", 4, "4f 16", 64, 15, 20, 23, &[(2, 2), (13, 22)], &[3, 6, 12, 76]),
        ("T12 R 23: b2n16 q00_07 p2.hw", 4, "4f 16", 64, 15, 23, 26, &[(2, 2), (13, 25)], &[3, 6, 12, 78]),
        ("T12 R 26: b2n16 q00_11 p1.hw", 4, "4f 16", 64, 14, 26, 29, &[(2, 2), (13, 28)], &[3, 6, 12, 81]),
        ("T12 R 28: b2n16 q00_12 p1.hw", 4, "4f 16", 64, 15, 28, 30, &[(2, 2), (13, 29)], &[3, 6, 12, 82]),
        ("T10: s1n16 qs t1.hw", 4, "16", 64, 14, 38, 41, &[(9, 40)], &[3, 8, 93]),
    ];
    for &(row, head, spec, big, rb, ra, grant, stalls, finishes) in chain {
        let a = if row.starts_with("T10") {
            Releaser::S2mm(ra)
        } else {
            Releaser::Core(ra)
        };
        let obs = run_chain(head, &mids(spec), big, Releaser::Core(rb), a);
        assert_eq!(
            (obs.held, obs.grant, obs.stalls.as_slice(), obs.finishes.as_slice()),
            (grant, grant, stalls, finishes),
            "{row}"
        );
    }
}

/// T11, the mid BDs of the short-BD cadence chains (silicon, NPU1 2026-11, chain family c2n /
/// c2f): six lock-free 2-word BDs behind the head finish every 2 cycles (g0 + 3, 5, ..., 15);
/// six 2-word BDs each on its own free lock every 3 (3, 6, ..., 21), each with a one-cycle
/// STALLED_LOCK at its grant - 1 (2, 5, ..., 17). Only the mids are asserted: the waiting BD
/// behind six lock-free mids is outside the law (its STALLED_LOCK rises at g0 + 20 on silicon;
/// open, pinned in `lock_free_chains_outside_the_law_keep_the_emulators_pinned_cycles`); behind
/// six free-lock ones it rises at 21 (`a_chain_of_short_lock_bds_holds_three_lock_bds_in_flight`).
#[test]
fn a_chain_of_short_bds_keeps_silicons_cadence_up_to_the_waiting_bd() {
    let obs = run_chain(4, &mids("2 2 2 2 2 2"), 64, Releaser::Free, Releaser::Core(45));
    assert_eq!(obs.finishes[..7], [3, 5, 7, 9, 11, 13, 15], "c2n");
    let obs = run_chain(4, &mids("2f 2f 2f 2f 2f 2f"), 64, Releaser::Free, Releaser::Core(45));
    assert_eq!(obs.finishes[..7], [3, 6, 9, 12, 15, 18, 21], "c2f");
    assert_eq!(obs.stalls[..6], [(2, 2), (5, 5), (8, 8), (11, 11), (14, 14), (17, 17)], "c2f blips");
}

/// Chains of short lock BDs before the waiting BD (silicon, NPU1 2026-11,
/// `tools/gen_dma_lock_probe.py` chain families k4f, c1f..c4f; head frame, the core releases A
/// at g0 + R, the LOCK_0_REL stamp). Three terms beyond the request and grant law:
/// (a) a lock BD starts sequencing no earlier than the lock release (FINISHED_BD + 4) of the
///     third lock BD before it: MEASURED, the sequencing starts the rises and finishes imply
///     (k4f3: the third mid shows no STALLED_LOCK blip and finishes at 12, so its first word is
///     sequenced at exactly 7; the 64-word BD of k4f2 / k4f3 / k4f5 from 7 / 11 / 19, its clock
///     floor 72 / 76 / 84); INFERRED, that these are the releases of the third lock BD before;
/// (b) a grant never falls on the release slot of any earlier lock BD of the run (k4f5: never
///     at 20, the fourth mid's FINISHED_BD 16 + 4);
/// (c) STALLED_LOCK rises at max(the first granule fully sequenced, Q, the previous BD's last
///     access) - 1 (c3f: the third 3-word mid, first word at 7, blips at 8).
/// Rows of the fitted designs and of the held-out ones (k4f4, c2f, c4f: predicted, then scored
/// once against silicon).
#[test]
fn a_chain_of_short_lock_bds_holds_three_lock_bds_in_flight() {
    #[rustfmt::skip]
    let rows: &[(&str, i64, i64, &[(i64, i64)], &[i64])] = &[
        // mids, RA, grant of A, STALLED_LOCK after the head, FINISHED_BD (silicon)
        ("4f", 15, 18, &[(2, 2), (5, 17)], &[3, 6, 70]), // k4f1 q00_00 p1.hw
        ("4f 4f", 15, 18, &[(2, 2), (5, 5), (9, 17)], &[3, 6, 9, 72]), // k4f2 q00_00 p1.hw
        ("4f 4f", 17, 19, &[(2, 2), (5, 5), (9, 18)], &[3, 6, 9, 72]), // k4f2 q00_02 p1.hw
        ("4f 4f 4f", 15, 18, &[(2, 2), (5, 5), (13, 17)], &[3, 6, 9, 12, 76]), // k4f3 q00_00 p1.hw
        ("4f 4f 4f", 17, 19, &[(2, 2), (5, 5), (13, 18)], &[3, 6, 9, 12, 76]), // k4f3 q00_02 p1.hw
        ("4f 4f 4f 4f", 15, 18, &[(2, 2), (5, 5), (17, 17)], &[3, 6, 9, 12, 16, 80]), // k4f4 q00_00 p1.hw (held out)
        ("4f 4f 4f 4f", 18, 21, &[(2, 2), (5, 5), (17, 20)], &[3, 6, 9, 12, 16, 80]), // k4f4 q00_02 p1.hw (held out)
        ("4f 4f 4f 4f 4f", 16, 19, &[(2, 2), (5, 5)], &[3, 6, 9, 12, 16, 20, 84]), // k4f5 q00_00 p1.hw
        ("4f 4f 4f 4f 4f", 18, 22, &[(2, 2), (5, 5), (21, 21)], &[3, 6, 9, 12, 16, 20, 84]), // k4f5 q00_02 p1.hw
        ("1f 1f 1f 1f 1f 1f", 46, 48, &[(2, 2), (5, 5), (8, 8), (11, 11), (14, 14), (17, 17), (21, 47)], &[3, 6, 9, 12, 15, 18, 21, 100]), // c1f q00_30 t1.hw
        ("2f 2f 2f 2f 2f 2f", 45, 48, &[(2, 2), (5, 5), (8, 8), (11, 11), (14, 14), (17, 17), (21, 47)], &[3, 6, 9, 12, 15, 18, 21, 100]), // c2f q00_30 t2.hw (held out)
        ("3f 3f 3f 3f 3f 3f", 45, 48, &[(2, 2), (5, 5), (8, 8), (11, 11), (14, 14), (17, 17), (21, 47)], &[3, 6, 9, 12, 15, 18, 21, 100]), // c3f q00_30 t1.hw
        ("4f 4f 4f 4f 4f 4f", 45, 47, &[(2, 2), (5, 5), (25, 46)], &[3, 6, 9, 12, 16, 20, 24, 99]), // c4f q00_30 t1.hw (held out)
        ("4f 4f 4f 4f 4f 4f", 46, 48, &[(2, 2), (5, 5), (25, 47)], &[3, 6, 9, 12, 16, 20, 24, 100]), // c4f q00_30 t2.hw (held out)
    ];
    for &(spec, ra, grant, stalls, finishes) in rows {
        let obs = run_chain(4, &mids(spec), 64, Releaser::Free, Releaser::Core(ra));
        assert_eq!(
            (obs.held, obs.grant, obs.stalls.as_slice(), obs.finishes.as_slice()),
            (grant, grant, stalls, finishes),
            "{spec}, RA {ra}"
        );
    }
}

/// Chains of K <= 5 lock-free short BDs before the waiting BD (silicon, NPU1 2026-11, chain
/// family kn, head frame): the request and grant law holds unchanged (the waiting BD's
/// STALLED_LOCK rises at the previous BD's last access - 1 = 2K + 1). kn1x2 / kn1x5 / kn4x4
/// were held out of the short-chain fit and scored once.
#[test]
fn a_chain_of_up_to_five_lock_free_short_bds_keeps_the_grant_law() {
    #[rustfmt::skip]
    let rows: &[(&str, i64, i64, &[(i64, i64)], &[i64])] = &[
        ("1 1", 45, 48, &[(5, 47)], &[3, 5, 7, 100]), // kn1x2 q00_30 t1.hw (held out)
        ("1 1 1 1 1", 45, 47, &[(11, 46)], &[3, 5, 7, 9, 11, 13, 99]), // kn1x5 q00_30 t1.hw (held out)
        ("4 4 4", 46, 48, &[(7, 47)], &[3, 5, 7, 9, 100]), // kn4x3 q00_30 t2.hw
        ("4 4 4 4", 45, 47, &[(9, 46)], &[3, 5, 7, 9, 11, 99]), // kn4x4 q00_30 t1.hw (held out)
        ("4 4 4 4", 46, 48, &[(9, 47)], &[3, 5, 7, 9, 11, 100]), // kn4x4 q00_30 t2.hw (held out)
        ("4 4 4 4 4", 45, 47, &[(13, 46)], &[3, 5, 7, 9, 11, 13, 99]), // kn4x5 q00_30 t1.hw
    ];
    for &(spec, ra, grant, stalls, finishes) in rows {
        let obs = run_chain(4, &mids(spec), 64, Releaser::Free, Releaser::Core(ra));
        assert_eq!(
            (obs.held, obs.grant, obs.stalls.as_slice(), obs.finishes.as_slice()),
            (grant, grant, stalls, finishes),
            "{spec}, RA {ra}"
        );
    }
}

/// PIN, NOT SILICON: lock-free chains where the request and grant law misses silicon (open, see
/// the DMA gap doc). These are the emulator's own cycles, pinned so that a later model of these
/// rows shows up as a deliberate change. Silicon (NPU1 2026-11, chain families c / kn, head
/// frame) differs as follows:
/// - six lock-free mids (the waiting BD is the chain's eighth BD and bd_id 7): STALLED_LOCK
///   rises at 20 on silicon for 1- to 4-word mids (the emulator: the previous BD's last access
///   - 1 = 13, or 14 / 17 with 3- / 4-word mids), and the grant phase moves with the mid
///   length (c1n RA 45 -> 48 on silicon, 47 here; c2n RA 45 -> 47, 46 -> 49 as here);
/// - five lock-free 4-word mids, RA 46 (kn4x5 q00_30 t2.hw): silicon grants at 49 and rises at
///   13, FINISHED_BD 101; the emulator grants at 48.
#[test]
fn lock_free_chains_outside_the_law_keep_the_emulators_pinned_cycles() {
    #[rustfmt::skip]
    let rows: &[(&str, i64, i64, &[(i64, i64)], &[i64])] = &[
        // mids, RA, grant, stalls, finishes (EMULATOR; silicon in the comment)
        ("1 1 1 1 1 1", 45, 47, &[(13, 46)], &[3, 5, 7, 9, 11, 13, 15, 99]), // c1n t1: silicon 48, [(20, 47)], .. 100
        ("1 1 1 1 1 1", 46, 48, &[(13, 47)], &[3, 5, 7, 9, 11, 13, 15, 100]), // c1n t2: silicon 48, [(20, 47)]
        ("2 2 2 2 2 2", 45, 47, &[(13, 46)], &[3, 5, 7, 9, 11, 13, 15, 99]), // c2n t1: silicon 47, [(20, 46)]
        ("2 2 2 2 2 2", 46, 49, &[(13, 48)], &[3, 5, 7, 9, 11, 13, 15, 101]), // c2n t3: silicon 49, [(20, 48)]
        ("4 4 4 4 4 4", 45, 48, &[(17, 47)], &[3, 5, 7, 9, 11, 13, 16, 100]), // c4n t2: silicon 48, [(20, 47)]
        ("4 4 4 4 4", 46, 48, &[(13, 47)], &[3, 5, 7, 9, 11, 13, 100]), // kn4x5 t2: silicon 49, [(13, 48)], .. 101
    ];
    for &(spec, ra, grant, stalls, finishes) in rows {
        let obs = run_chain(4, &mids(spec), 64, Releaser::Free, Releaser::Core(ra));
        assert_eq!(
            (obs.held, obs.grant, obs.stalls.as_slice(), obs.finishes.as_slice()),
            (grant, grant, stalls, finishes),
            "{spec}, RA {ra}"
        );
    }
}
