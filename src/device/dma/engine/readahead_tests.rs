//! Tests for the shim MM2S read-ahead window.
//!
//! A shim MM2S channel reading host memory keeps reads outstanding up to a
//! window of data not yet accepted by its stream port, each read returning a
//! fixed latency after it issues; FINISHED_BD / FINISHED_TASK fire when a BD's
//! last read RETURNS, while the window still drains into the stream. NPU1
//! silicon (shim-inflight probe, 2026-11-07): a back-pressured shim MM2S
//! reports FINISHED with ~3.4 KiB still between it and the consumer's memory.

use super::*;

const IN_A: u64 = 0x4000_0000;
const IN_B: u64 = 0x4010_0000;

/// Host memory holding `words` distinct words at `base` (value = tag + index).
fn host_with(regions: &[(u64, usize, u32)]) -> HostMemory {
    let mut host = HostMemory::new();
    for (i, &(base, words, tag)) in regions.iter().enumerate() {
        host.allocate_region(format!("in{i}"), base, words * 4).unwrap();
        for w in 0..words {
            host.write_u32(base + 4 * w as u64, tag + w as u32);
        }
    }
    host
}

/// The shim's first MM2S channel (flat index).
fn mm2s0(engine: &DmaEngine) -> u8 {
    (0..engine.num_channels() as u8)
        .find(|&c| matches!(engine.channel_type(c), ChannelType::MM2S))
        .expect("shim has an MM2S channel")
}

fn window_words(engine: &DmaEngine) -> usize {
    engine.timing_config.shim_mm2s_read_window_bytes as usize / 4
}

fn latency(engine: &DmaEngine) -> u64 {
    engine.timing_config.shim_mm2s_read_latency_cycles as u64
}

/// One cycle: step the engine at `cycle`, then let the consumer take at most
/// one word from the stream port if `take` (the stream is one word per cycle).
struct Rig {
    engine: DmaEngine,
    tile: Tile,
    host: HostMemory,
    ch: u8,
    cycle: u64,
    received: Vec<u32>,
    events: Vec<(u64, EventType)>,
}

impl Rig {
    fn new(host: HostMemory) -> Self {
        let engine = DmaEngine::new_shim_tile(1, 0);
        let ch = mm2s0(&engine);
        Rig {
            engine,
            tile: Tile::shim(1, 0),
            host,
            ch,
            cycle: 0,
            received: Vec::new(),
            events: Vec::new(),
        }
    }

    fn step(&mut self, take: bool) {
        self.engine.set_current_cycle(self.cycle);
        self.engine.step(&mut self.tile, &mut NeighborTiles::empty(), &mut self.host);
        if take {
            if let Some(d) = self.engine.pop_stream_out_for_channel(self.ch) {
                self.received.push(d.data);
            }
        }
        self.events.extend(self.engine.drain_trace_events());
        self.cycle += 1;
    }

    fn first(&self, want: impl Fn(&EventType) -> bool) -> Option<u64> {
        self.events.iter().find(|(_, e)| want(e)).map(|(c, _)| *c)
    }

    fn finished_bd(&self) -> Option<u64> {
        self.first(|e| matches!(e, EventType::DmaFinishedBd { .. }))
    }
}

/// The two calibrated constants and their evidence: L = 325 is the 2026-05-27
/// per-task MM2S overhead re-read as the read latency (inside the probe's HW
/// rel->FINISHED spread 308-722); the window is pinned between the 512 B-tile
/// and 128 B-tile points of the shim-inflight sweep.
#[test]
fn read_ahead_constants_carry_their_calibration() {
    let cfg = DmaTimingConfig::default();
    assert_eq!(cfg.shim_mm2s_read_latency_cycles, 325);
    assert_eq!(cfg.shim_mm2s_read_window_bytes, 3328);
    // The latency is no longer charged as a pre-transfer per-task overhead.
    assert_eq!(cfg.shim_per_task_overhead_mm2s_cycles, 0);
}

/// (a) A back-pressured multi-KiB BD reports FINISHED once its last read has
/// returned, with the whole window still undelivered.
#[test]
fn backpressured_shim_mm2s_finishes_with_the_window_undelivered() {
    let words = 2048; // 8 KiB
    let mut rig = Rig::new(host_with(&[(IN_A, words, 0)]));
    rig.engine.configure_bd(0, BdConfig::simple_1d(IN_A, words as u32 * 4)).unwrap();
    assert!(rig.engine.enqueue_task(rig.ch, 0, 0, false));
    let w = window_words(&rig.engine);

    // The consumer takes exactly the words that must leave before the final
    // read can issue, then stops taking.
    for _ in 0..20_000 {
        let take = rig.received.len() < words - w;
        rig.step(take);
        if rig.finished_bd().is_some() {
            break;
        }
    }
    let fin = rig.finished_bd().expect("the BD finishes while back-pressured");
    assert!(rig.first(|e| matches!(e, EventType::DmaFinishedTask { .. })) == Some(fin));
    assert_eq!(rig.received.len(), words - w, "the window is still undelivered at FINISHED");
    assert_eq!(rig.engine.read_ahead_words(rig.ch), w, "read buffer + output register hold the window");

    // The held data then drains, in order, after FINISHED.
    for _ in 0..(w as u64 + 10) {
        rig.step(true);
    }
    assert_eq!(rig.received, (0..words as u32).collect::<Vec<_>>());
    assert_eq!(rig.engine.read_ahead_words(rig.ch), 0);
    assert!(!rig.engine.channel_has_pending_work(rig.ch));
}

/// With a stream that never back-pressures, data enters the stream one read
/// latency after the first read issues and FINISHED lands one read latency
/// after the last read issues.
#[test]
fn free_stream_finishes_one_read_latency_after_the_last_issue() {
    let words = 256;
    let mut rig = Rig::new(host_with(&[(IN_A, words, 0)]));
    rig.engine.configure_bd(0, BdConfig::simple_1d(IN_A, words as u32 * 4)).unwrap();
    assert!(rig.engine.enqueue_task(rig.ch, 0, 0, false));

    let mut first_issue = None;
    let mut last_issue = None;
    let mut first_rx = None;
    for _ in 0..5_000 {
        let issued_before = rig.engine.channel_stats(rig.ch).unwrap().bytes_transferred;
        let at = rig.cycle;
        rig.step(true);
        let issued_after = rig.engine.channel_stats(rig.ch).unwrap().bytes_transferred;
        if issued_after > issued_before {
            first_issue.get_or_insert(at);
            last_issue = Some(at);
        }
        if first_rx.is_none() && !rig.received.is_empty() {
            first_rx = Some(at);
        }
        if rig.finished_bd().is_some() && rig.received.len() == words {
            break;
        }
    }
    let l = latency(&rig.engine);
    assert_eq!(
        first_rx.unwrap() - first_issue.unwrap(),
        l,
        "first data one read latency after the first issue"
    );
    assert_eq!(
        rig.finished_bd().unwrap() - last_issue.unwrap(),
        l,
        "FINISHED one read latency after the last issue"
    );
    assert_eq!(
        last_issue.unwrap() - first_issue.unwrap(),
        words as u64 - 1,
        "reads issue one word per cycle"
    );
    assert_eq!(rig.received, (0..words as u32).collect::<Vec<_>>());
}

/// (c) A following task starts as soon as the previous task's last read has
/// issued -- before that read returns -- and reads while the previous task's
/// data is still draining; its data stays ordered behind it and FINISHED_TASK
/// keeps task order.
#[test]
fn next_task_issues_behind_the_previous_tasks_draining_data() {
    let words = 512;
    let mut rig = Rig::new(host_with(&[(IN_A, words, 0), (IN_B, words, 10_000)]));
    rig.engine.configure_bd(0, BdConfig::simple_1d(IN_A, words as u32 * 4)).unwrap();
    rig.engine.configure_bd(1, BdConfig::simple_1d(IN_B, words as u32 * 4)).unwrap();
    assert!(rig.engine.enqueue_task(rig.ch, 0, 0, false));
    assert!(rig.engine.enqueue_task(rig.ch, 1, 0, false));

    // Slow consumer: one word every 4 cycles keeps the window full.
    let mut second_task_started_at = None;
    let mut second_task_issuing_at = None;
    let mut first_task_drained_at = None;
    for _ in 0..40_000 {
        let take = rig.cycle % 4 == 0;
        rig.step(take);
        let on_bd1 = rig.engine.current_bd(rig.ch) == Some(1);
        if second_task_started_at.is_none() && on_bd1 && rig.engine.channel_phase(rig.ch) != "Idle" {
            second_task_started_at = Some(rig.cycle - 1);
        }
        if second_task_issuing_at.is_none()
            && on_bd1
            && rig.engine.get_transfer(rig.ch).is_some_and(|t| t.bytes_transferred > 0)
        {
            second_task_issuing_at = Some(rig.cycle - 1);
        }
        if first_task_drained_at.is_none() && rig.received.len() == words {
            first_task_drained_at = Some(rig.cycle - 1);
        }
        if rig.received.len() == 2 * words {
            break;
        }
    }
    let finished: Vec<u64> = rig
        .events
        .iter()
        .filter(|(_, e)| matches!(e, EventType::DmaFinishedTask { .. }))
        .map(|(c, _)| *c)
        .collect();
    assert_eq!(finished.len(), 2);
    assert!(finished[0] < finished[1]);
    let started = second_task_started_at.expect("the second task starts");
    assert_eq!(finished[0] - started, latency(&rig.engine), "task 1 starts when task 0's last read issues");
    let issuing = second_task_issuing_at.expect("the second task issues reads");
    let drained = first_task_drained_at.unwrap();
    assert!(issuing < drained, "task 1 reads ({issuing}) while task 0's data still drains (until {drained})");
    let mut expect: Vec<u32> = (0..words as u32).collect();
    expect.extend(10_000..10_000 + words as u32);
    assert_eq!(rig.received, expect, "task 1's data is ordered behind task 0's");
}

/// A chained BD inside one task issues straight after its predecessor's last
/// read (no read latency per BD: the cold start is per task), and each BD's
/// FINISHED_BD fires when that BD's own last read returns.
#[test]
fn chained_bd_issues_without_waiting_for_the_previous_bds_return() {
    let words = 128;
    let mut rig = Rig::new(host_with(&[(IN_A, words, 0), (IN_B, words, 500)]));
    rig.engine
        .configure_bd(0, BdConfig::simple_1d(IN_A, words as u32 * 4).with_next(1))
        .unwrap();
    rig.engine.configure_bd(1, BdConfig::simple_1d(IN_B, words as u32 * 4)).unwrap();
    assert!(rig.engine.enqueue_task(rig.ch, 0, 0, false));

    let mut issue_cycles = Vec::new();
    for _ in 0..5_000 {
        let before = rig.engine.channel_stats(rig.ch).unwrap().bytes_transferred;
        let at = rig.cycle;
        rig.step(true);
        if rig.engine.channel_stats(rig.ch).unwrap().bytes_transferred > before {
            issue_cycles.push(at);
        }
        if rig.received.len() == 2 * words {
            break;
        }
    }
    assert_eq!(issue_cycles.len(), 2 * words);
    let bd0_last = issue_cycles[words - 1];
    let bd1_first = issue_cycles[words];
    assert!(bd1_first - bd0_last <= 2, "BD1 issues {} cycles after BD0's last read", bd1_first - bd0_last);
    let fins: Vec<u64> = rig
        .events
        .iter()
        .filter(|(_, e)| matches!(e, EventType::DmaFinishedBd { .. }))
        .map(|(c, _)| *c)
        .collect();
    let l = latency(&rig.engine);
    assert_eq!(fins, vec![bd0_last + l, issue_cycles[2 * words - 1] + l]);
    let mut expect: Vec<u32> = (0..words as u32).collect();
    expect.extend(500..500 + words as u32);
    assert_eq!(rig.received, expect);
}

/// The window counts data not yet accepted by the stream port: a fully
/// back-pressured channel issues exactly the window and stalls.
#[test]
fn blocked_stream_issues_exactly_the_window() {
    let words = 4096;
    let mut rig = Rig::new(host_with(&[(IN_A, words, 0)]));
    rig.engine.configure_bd(0, BdConfig::simple_1d(IN_A, words as u32 * 4)).unwrap();
    assert!(rig.engine.enqueue_task(rig.ch, 0, 0, false));
    for _ in 0..10_000 {
        rig.step(false);
    }
    let w = window_words(&rig.engine);
    assert_eq!(rig.engine.channel_stats(rig.ch).unwrap().bytes_transferred, 4 * w as u64);
    assert_eq!(rig.engine.read_ahead_words(rig.ch), w);
    assert!(rig.finished_bd().is_none());
    // Stream-side backpressure is asserted (a returned word waits on the port).
    assert!(rig
        .events
        .iter()
        .any(|(_, e)| matches!(e, EventType::DmaStreamStarvation { active: true, .. })));
}

/// STREAM_BACKPRESSURE is a stream-side level: it keeps tracing after FINISHED
/// while the window drains into a back-pressured port (HW k8-t1s: spans run
/// to 53.7k after FINISHED at 29.6k), and closes once the data has gone.
#[test]
fn stream_backpressure_outlives_finished() {
    let words = 1024;
    let mut rig = Rig::new(host_with(&[(IN_A, words, 0)]));
    rig.engine.configure_bd(0, BdConfig::simple_1d(IN_A, words as u32 * 4)).unwrap();
    assert!(rig.engine.enqueue_task(rig.ch, 0, 0, false));
    // A port that takes one word in 8.
    for _ in 0..20_000 {
        let take = rig.cycle % 8 == 0;
        rig.step(take);
        if rig.received.len() == words {
            break;
        }
    }
    let fin = rig.finished_bd().unwrap();
    let bp_after_fin = rig
        .events
        .iter()
        .filter(|(c, e)| *c > fin && matches!(e, EventType::DmaStreamStarvation { active: true, .. }));
    assert!(bp_after_fin.count() > 0, "backpressure keeps tracing after FINISHED");
    let last_level = rig
        .events
        .iter()
        .rev()
        .find_map(|(_, e)| match e {
            EventType::DmaStreamStarvation { active, .. } => Some(*active),
            _ => None,
        })
        .unwrap();
    assert!(!last_level, "the level closes once the window has drained");
    assert_eq!(rig.received, (0..words as u32).collect::<Vec<_>>());
}

/// (d) A compute-tile MM2S (banked tile memory, egress staging) is untouched:
/// it holds its 25-word staging and does not finish until its last word has
/// entered the stream.
#[test]
fn compute_tile_mm2s_still_finishes_only_when_its_data_is_in_the_stream() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = Tile::compute(1, 2);
    let mut host = HostMemory::new();
    let ch = mm2s0(&engine);
    engine.configure_bd(0, BdConfig::simple_1d(0x400, 256)).unwrap(); // 64 words
    assert!(engine.enqueue_task(ch, 0, 0, false));
    let mut events = Vec::new();
    for cycle in 0..2_000u64 {
        engine.set_current_cycle(cycle);
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host);
        events.extend(engine.drain_trace_events());
    }
    let staged = engine.staged_words(ch);
    assert!(staged > 0 && staged <= engine.egress_staging_capacity(), "staging holds {staged}");
    assert_eq!(engine.read_ahead_words(ch), 0, "tile-memory MM2S has no read-ahead window");
    assert!(!events.iter().any(|(_, e)| matches!(e, EventType::DmaFinishedBd { .. })));
    let mut got = 0;
    for cycle in 2_000..4_000u64 {
        engine.set_current_cycle(cycle);
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host);
        if engine.pop_stream_out_for_channel(ch).is_some() {
            got += 1;
        }
        events.extend(engine.drain_trace_events());
    }
    assert_eq!(got, 64);
    assert!(events.iter().any(|(_, e)| matches!(e, EventType::DmaFinishedBd { .. })));
}

/// A channel reset drops the read buffer and the pending completions.
#[test]
fn channel_reset_drops_outstanding_reads() {
    let words = 2048;
    let mut rig = Rig::new(host_with(&[(IN_A, words, 0)]));
    rig.engine.configure_bd(0, BdConfig::simple_1d(IN_A, words as u32 * 4)).unwrap();
    assert!(rig.engine.enqueue_task(rig.ch, 0, 0, false));
    for _ in 0..2_000 {
        rig.step(false);
    }
    assert!(rig.engine.read_ahead_words(rig.ch) > 0);
    rig.engine.reset_channel(rig.ch);
    assert_eq!(rig.engine.read_ahead_words(rig.ch), 0);
    assert!(!rig.engine.channel_has_pending_work(rig.ch));
}

/// Silicon (lock-leftovers shim_mm2s_rel_c1..c7): a shim BD whose release meets a flagged lock
/// does not hold the channel -- the release is parked and the chain runs to its end (every
/// later BD's gate is taken), later releases queue behind the parked one. The channel reads as
/// idle with up to two releases parked, and as running with Stalled_Lock_Rel from three on.
/// After the clear every release is performed.
#[test]
fn a_shim_release_on_a_frozen_lock_is_parked_and_the_chain_runs_on() {
    let mut rig = Rig::new(host_with(&[(IN_A, 16, 0), (IN_B, 16, 100)]));
    rig.tile.locks[5].set(0);
    rig.tile.locks[5].overflow = true;
    for gate in [6, 8, 10] {
        rig.tile.locks[gate].set(1);
    }
    let bds = [
        BdConfig::simple_1d(IN_A, 64).with_release(5, 1).with_next(1),
        BdConfig::simple_1d(IN_B, 64)
            .with_acquire(6, -1)
            .with_release(7, 1)
            .with_next(2),
        BdConfig::simple_1d(IN_A, 64)
            .with_acquire(8, -1)
            .with_release(9, 1)
            .with_next(3),
        BdConfig::simple_1d(IN_B, 64).with_acquire(10, -1).with_release(11, 1),
    ];
    for (i, bd) in bds.into_iter().enumerate() {
        rig.engine.configure_bd(i as u8, bd).unwrap();
    }
    assert!(rig.engine.enqueue_task(rig.ch, 0, 0, false));
    let run = |rig: &mut Rig, cycles: u32| {
        for _ in 0..cycles {
            rig.engine.submit_lock_requests(&mut rig.tile, &mut NeighborTiles::empty());
            rig.tile.resolve_lock_requests(rig.cycle);
            rig.step(true);
        }
    };
    run(&mut rig, 4000);
    let gates: Vec<u8> = [6, 8, 10].iter().map(|&l| rig.tile.locks[l].value).collect();
    assert_eq!(gates, [0, 0, 0], "the chain ran to its end with the first release parked");
    let releases: Vec<u8> = [5, 7, 9, 11].iter().map(|&l| rig.tile.locks[l].value).collect();
    assert_eq!(releases, [0, 0, 0, 0], "every release waits behind the flagged one");
    let status = rig.engine.get_channel_status(rig.ch);
    assert_ne!(status & (1 << 19), 0, "three or more parked releases keep the channel running");
    assert_ne!(status & (1 << 3), 0, "and Stalled_Lock_Rel is set");
    assert_eq!(rig.received.len(), 64, "all four BDs' data went out");
    rig.tile.locks[5].clear_flags();
    run(&mut rig, 50);
    let releases: Vec<u8> = [5, 7, 9, 11].iter().map(|&l| rig.tile.locks[l].value).collect();
    assert_eq!(releases, [1, 1, 1, 1], "the clear lets all four through");
    assert_eq!(rig.engine.get_channel_status(rig.ch) & (1 << 3), 0);
}

/// Silicon (lock-leftovers mem_s2mm_rel): a shim MM2S whose BDs are all issued but whose sink
/// takes nothing still reads Channel_Running with Stalled_Stream_Backpressure -- the window is
/// undelivered -- though its BDs have FINISHED.
#[test]
fn a_shim_mm2s_with_an_undelivered_window_reads_running_and_back_pressured() {
    let words = 2048;
    let mut rig = Rig::new(host_with(&[(IN_A, words, 0)]));
    rig.engine.configure_bd(0, BdConfig::simple_1d(IN_A, words as u32 * 4)).unwrap();
    assert!(rig.engine.enqueue_task(rig.ch, 0, 0, false));
    let w = window_words(&rig.engine);
    for _ in 0..20_000 {
        let take = rig.received.len() < words - w;
        rig.step(take);
        if rig.finished_bd().is_some() {
            break;
        }
    }
    assert!(rig.finished_bd().is_some());
    for _ in 0..10 {
        rig.step(false);
    }
    let status = rig.engine.get_channel_status(rig.ch);
    assert_ne!(status & (1 << 19), 0, "Channel_Running while the window is undelivered");
    assert_ne!(status & (1 << 4), 0, "Stalled_Stream_Backpressure");
    for _ in 0..(w as u64 + 10) {
        rig.step(true);
    }
    assert_eq!(rig.engine.get_channel_status(rig.ch) & ((1 << 19) | (1 << 4)), 0, "idle once drained");
}
