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
    /// Set by a register write in cycle g0 + g, before that cycle's arbitration.
    WriteAt(i64),
}

impl Releaser {
    /// Apply the release to `lock` if it happens in cycle `cycle` (g0-relative): before the
    /// arbitration when `before_resolve`, after the engine's step otherwise.
    fn apply(self, tile: &mut Tile, lock: u8, cycle: i64, before_resolve: bool) {
        match self {
            Releaser::Core(r) if before_resolve && r == cycle => {
                tile.defer_core_lock_release(lock as usize, 1)
            }
            Releaser::WriteAt(r) if before_resolve && r == cycle => tile.locks[lock as usize].value = 1,
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
        stalls: rig.trace.stalls[1..].iter().map(|&(a, b)| (rel(a), rel(b))).collect(),
        finishes: rig.trace.finishes.iter().map(|&c| rel(c)).collect(),
    }
}

/// The compute MM2S word clock (`WordClock`) on the sync design: a 4-word head parked on its
/// lock, a lock-free BD of N words, a 64-word BD on lock A. With the waiting BD's grant G
/// placed where silicon measured it (the lock written in that cycle), every FINISHED_BD lands
/// on silicon's cycle (NPU1, 2026-11, `tools/gen_dma_lock_probe.py` sync family; g0 frame):
/// the 4-granule park whatever the granule sizes (N = 13: 64-word finish g0 + 70, a 16-word
/// lead gives + 69), the BD-switch access gap (N = 5..8: the N-word BD finishes 3 after the
/// head), and max(clock, G + 52) for the 64-word BD.
#[test]
fn the_word_clock_puts_every_sync_design_finish_on_silicons_cycle() {
    #[rustfmt::skip]
    let rows: &[(u32, i64, &[i64])] = &[
        // N, grant (silicon), FINISHED_BD of head / N-word BD / 64-word BD (silicon)
        (0, 16, &[3, 68]), (0, 25, &[3, 77]), (0, 34, &[3, 86]), (0, 42, &[3, 94]), (0, 57, &[3, 109]),
        (5, 16, &[3, 6, 68]), (5, 21, &[3, 6, 73]), (5, 24, &[3, 6, 76]), (5, 27, &[3, 6, 79]),
        (8, 16, &[3, 6, 68]), (8, 21, &[3, 6, 73]), (8, 27, &[3, 6, 79]), (8, 36, &[3, 6, 88]),
        (13, 16, &[3, 8, 70]), (13, 19, &[3, 8, 71]), (13, 22, &[3, 8, 74]), (13, 27, &[3, 8, 79]),
        (14, 16, &[3, 8, 70]), (14, 18, &[3, 8, 70]), (14, 22, &[3, 8, 74]), (14, 30, &[3, 8, 82]),
        (16, 17, &[3, 8, 72]), (16, 21, &[3, 8, 73]), (16, 24, &[3, 8, 76]), (16, 38, &[3, 8, 90]),
        (17, 17, &[3, 9, 73]), (17, 20, &[3, 9, 73]), (17, 23, &[3, 9, 75]), (17, 30, &[3, 9, 82]),
        (20, 17, &[3, 12, 76]), (20, 20, &[3, 12, 76]), (20, 24, &[3, 12, 76]), (20, 36, &[3, 12, 88]),
        (32, 35, &[3, 24, 88]), (32, 45, &[3, 24, 97]), (32, 56, &[3, 24, 108]),
    ];
    for &(n, grant, finishes) in rows {
        let mids = if n == 0 { vec![] } else { vec![Mid::Plain(n)] };
        let obs = run_chain(4, &mids, 64, Releaser::Free, Releaser::WriteAt(grant));
        assert_eq!(obs.finishes, finishes, "N = {n}, grant g0 + {grant}");
    }
}

/// The same with a longer head or a short waiting BD (silicon, chain family h08n16, h16n08,
/// w16n16; g0 frame): an 8- or 16-word head parks 2 or 4 of its granules.
#[test]
fn the_word_clock_puts_long_head_and_short_tail_finishes_on_silicons_cycle() {
    #[rustfmt::skip]
    let rows: &[(u32, u32, u32, i64, &[i64])] = &[
        // head, mid, waiting BD words, grant, finishes (silicon)
        (8, 16, 64, 18, &[4, 12, 76]), (8, 16, 64, 23, &[4, 12, 76]), (8, 16, 64, 27, &[4, 12, 79]),
        (8, 16, 64, 32, &[4, 12, 84]),
        (16, 8, 64, 20, &[6, 12, 76]), (16, 8, 64, 24, &[6, 12, 76]), (16, 8, 64, 29, &[6, 12, 81]),
        (16, 8, 64, 33, &[6, 12, 85]),
        (4, 16, 16, 17, &[3, 8, 24]), (4, 16, 16, 23, &[3, 8, 29]), (4, 16, 16, 26, &[3, 8, 32]),
        (4, 16, 16, 30, &[3, 8, 36]),
    ];
    for &(head, mid, big, grant, finishes) in rows {
        let obs = run_chain(head, &[Mid::Plain(mid)], big, Releaser::Free, Releaser::WriteAt(grant));
        assert_eq!(obs.finishes, finishes, "head {head}, mid {mid}, waiting {big}, grant g0 + {grant}");
    }
}
