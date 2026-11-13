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
