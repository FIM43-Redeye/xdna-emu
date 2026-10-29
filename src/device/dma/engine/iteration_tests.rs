//! Tests for BD iteration (Iteration_Current / _Wrap / _Stepsize).
//!
//! Iteration is per BD execution: each load of the BD moves Buffer_Length
//! bytes at base + current * stepsize, then Iteration_Current advances,
//! wrapping after the last slot (AM025: "incremented after BD loaded",
//! "offset each execution of this BD"). NPU1 silicon: mlir-aie
//! test/npu-xrt/bd_iteration_dma_task built for npu1 passes (2026-10-26).

use super::*;
use crate::device::dma::addressing::IterationConfig;

/// A 4-word BD over four 4-word slots, starting at slot `current`.
fn iterating_bd(base: u64, current: u8) -> BdConfig {
    let mut bd = BdConfig::simple_1d(base, 16);
    bd.iteration = IterationConfig { current, wrap: 3, stepsize: 3 };
    bd
}

fn read_words(tile: &Tile, addr: usize, n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| {
            let at = addr + 4 * i;
            u32::from_le_bytes(tile.data_memory()[at..at + 4].try_into().unwrap())
        })
        .collect()
}

#[test]
fn s2mm_repeats_place_each_execution_in_the_next_slot() {
    // The bd_iteration_dma_task shape: four executions from current = 2
    // land chunks 0..3 in slots 2, 3, 0, 1.
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = Tile::compute(1, 2);
    let mut host_mem = HostMemory::new();
    engine.configure_bd(0, iterating_bd(0x100, 2)).unwrap();
    engine.enqueue_task(0, 0, 3, false); // S2MM ch0, four executions

    let mut feed: std::collections::VecDeque<u32> = (0..16).map(|i| 100 + i).collect();
    for _ in 0..400 {
        if let Some(&data) = feed.front() {
            if engine.can_accept_stream_in_for_channel(0)
                && engine.push_stream_in(StreamData { data, tlast: false, channel: 0 })
            {
                feed.pop_front();
            } else {
                engine.consume_bd_switch_accept_block(0);
            }
        }
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
    }

    assert!(!engine.channel_has_pending_work(0), "four 4-word executions take exactly 16 words");
    let chunk = |k: u32| (0..4).map(|j| 100 + 4 * k + j).collect::<Vec<_>>();
    assert_eq!(read_words(&tile, 0x100 + 2 * 16, 4), chunk(0));
    assert_eq!(read_words(&tile, 0x100 + 3 * 16, 4), chunk(1));
    assert_eq!(read_words(&tile, 0x100, 4), chunk(2));
    assert_eq!(read_words(&tile, 0x100 + 16, 4), chunk(3));
}

#[test]
fn iteration_current_advances_per_load_and_wraps() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile_with_pattern();
    let mut host_mem = HostMemory::new();
    engine.configure_bd(0, iterating_bd(0x100, 3)).unwrap();

    // One MM2S execution from slot 3, then the next from slot 0.
    let mut sent = Vec::new();
    for _ in 0..2 {
        engine.enqueue_task(2, 0, 0, false);
        for _ in 0..200 {
            engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
            while let Some(word) = engine.pop_stream_out() {
                sent.push(word.data);
            }
        }
    }
    assert_eq!(sent, [12, 13, 14, 15, 0, 1, 2, 3], "slot 3, then wrap to slot 0");
    assert_eq!(engine.get_bd(0).unwrap().iteration.current, 1);
}

#[test]
fn a_bd_without_iteration_is_unchanged() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile_with_pattern();
    let mut host_mem = HostMemory::new();
    engine.configure_bd(0, BdConfig::simple_1d(0x100, 16)).unwrap();
    engine.enqueue_task(2, 0, 1, false); // two executions, same words

    let mut sent = Vec::new();
    for _ in 0..300 {
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        while let Some(word) = engine.pop_stream_out() {
            sent.push(word.data);
        }
    }
    assert_eq!(sent, [0, 1, 2, 3, 0, 1, 2, 3]);
}

/// Words 0..16 at 0x100 (four 4-word slots holding 0..3, 4..7, ...).
fn make_tile_with_pattern() -> Tile {
    let mut tile = Tile::compute(1, 2);
    for i in 0..16u32 {
        let at = 0x100 + 4 * i as usize;
        tile.data_memory_mut()[at..at + 4].copy_from_slice(&i.to_le_bytes());
    }
    tile
}
