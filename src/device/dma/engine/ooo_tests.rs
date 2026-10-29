//! Tests for out-of-order S2MM (Ctrl Enable_Out_of_Order).
//!
//! An out-of-order channel places each incoming packet into the BD whose id
//! the packet header carries (bits 10-5, stamped by the sender's
//! Out_Of_Order_BD_ID), not the BD the task queue names. The header is
//! consumed, Start_BD_ID and the BD chain are ignored, and a task completes
//! after Repeat_Count + 1 packets. Silicon reference: mlir-aie
//! test/npu-xrt/dma_s2mm_ooo_shim_recv passes on NPU1 (2026-10-26).

use std::collections::VecDeque;

use super::*;
use crate::device::stream_switch::PacketHeader;

const S2MM0: u8 = 0;

fn ooo_engine() -> DmaEngine {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    engine.set_channel_compression_config(S2MM0, false, false, true);
    engine
}

/// One packet: the header naming `bd`, then `payload`, TLAST on the last word.
fn packet(bd: u8, payload: &[u32]) -> Vec<StreamData> {
    let header = PacketHeader::new(0, 1, 3).with_ooo_bd_id(bd).encode();
    let mut words = vec![StreamData { data: header, tlast: false, channel: S2MM0 }];
    for (i, &data) in payload.iter().enumerate() {
        words.push(StreamData { data, tlast: i + 1 == payload.len(), channel: S2MM0 });
    }
    words
}

/// Step the engine for `cycles`, offering one stream word per cycle whenever
/// the channel accepts it (the 32-bit stream beat rate).
fn run(engine: &mut DmaEngine, tile: &mut Tile, feed: &mut VecDeque<StreamData>, cycles: usize) {
    let mut host_mem = HostMemory::new();
    for _ in 0..cycles {
        if let Some(&word) = feed.front() {
            if engine.can_accept_stream_in_for_channel(S2MM0) && engine.push_stream_in(word) {
                feed.pop_front();
            } else {
                engine.consume_bd_switch_accept_block(S2MM0 as usize);
            }
        }
        engine.submit_lock_requests(tile, &mut NeighborTiles::empty());
        tile.resolve_lock_requests(0);
        engine.step(tile, &mut NeighborTiles::empty(), &mut host_mem);
    }
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
fn places_each_packet_in_the_bd_its_header_names() {
    let mut engine = ooo_engine();
    let mut tile = Tile::compute(1, 2);
    engine.configure_bd(5, BdConfig::simple_1d(0x100, 16)).unwrap();
    engine.configure_bd(9, BdConfig::simple_1d(0x200, 16)).unwrap();
    // Two packets; Start_BD_ID 0 is what aie-rt writes in this mode.
    engine.enqueue_task(S2MM0, 0, 1, false);

    let mut feed: VecDeque<_> = [packet(9, &[1, 2, 3, 4]), packet(5, &[101, 102, 103, 104])].concat().into();
    run(&mut engine, &mut tile, &mut feed, 300);

    assert!(feed.is_empty(), "every word must be accepted");
    assert_eq!(
        read_words(&tile, 0x200, 5),
        [1, 2, 3, 4, 0],
        "BD 9 gets the first packet, header not written"
    );
    assert_eq!(read_words(&tile, 0x100, 5), [101, 102, 103, 104, 0], "BD 5 gets the second packet");
    assert!(!engine.channel_has_pending_work(S2MM0), "the task ends after Repeat_Count + 1 packets");
    assert_eq!(engine.channel_stats(S2MM0).unwrap().transfers_completed, 2);
}

#[test]
fn ignores_start_bd_and_the_bd_chain() {
    let mut engine = ooo_engine();
    let mut tile = Tile::compute(1, 2);
    let mut chained = BdConfig::simple_1d(0x100, 16);
    chained.next_bd = Some(9);
    engine.configure_bd(5, chained).unwrap();
    engine.configure_bd(9, BdConfig::simple_1d(0x200, 16)).unwrap();
    // Start_BD_ID 9 must not be used; the header says BD 5.
    engine.enqueue_task(S2MM0, 9, 0, false);

    let mut feed: VecDeque<_> = packet(5, &[7, 8, 9, 10]).into();
    run(&mut engine, &mut tile, &mut feed, 300);

    assert_eq!(read_words(&tile, 0x100, 4), [7, 8, 9, 10]);
    assert_eq!(read_words(&tile, 0x200, 4), [0, 0, 0, 0], "neither Start_BD_ID nor next_bd is followed");
    assert!(!engine.channel_has_pending_work(S2MM0));
    assert_eq!(engine.channel_stats(S2MM0).unwrap().transfers_completed, 1);
}

#[test]
fn task_waits_for_its_remaining_packets() {
    let mut engine = ooo_engine();
    let mut tile = Tile::compute(1, 2);
    engine.configure_bd(5, BdConfig::simple_1d(0x100, 16)).unwrap();
    engine.configure_bd(9, BdConfig::simple_1d(0x200, 16)).unwrap();
    engine.enqueue_task(S2MM0, 0, 2, false); // three packets

    let mut feed: VecDeque<_> = [packet(9, &[1, 2, 3, 4]), packet(5, &[5, 6, 7, 8])].concat().into();
    run(&mut engine, &mut tile, &mut feed, 300);
    assert!(engine.channel_active(S2MM0), "two of three packets: the task is still waiting for a header");

    feed.extend(packet(9, &[9, 10, 11, 12]));
    run(&mut engine, &mut tile, &mut feed, 300);
    assert_eq!(read_words(&tile, 0x200, 4), [9, 10, 11, 12], "a BD can be reused by a later packet");
    assert!(!engine.channel_has_pending_work(S2MM0));
}

#[test]
fn queued_tasks_each_take_their_own_packet_count() {
    let mut engine = ooo_engine();
    let mut tile = Tile::compute(1, 2);
    engine.configure_bd(5, BdConfig::simple_1d(0x100, 16)).unwrap();
    engine.configure_bd(9, BdConfig::simple_1d(0x200, 16)).unwrap();
    // The dma_s2mm_ooo_shim_recv shape: two single-packet tasks on one channel.
    engine.enqueue_task(S2MM0, 0, 0, false);
    engine.enqueue_task(S2MM0, 0, 0, false);

    let mut feed: VecDeque<_> = [packet(9, &[1, 2, 3, 4]), packet(5, &[5, 6, 7, 8])].concat().into();
    run(&mut engine, &mut tile, &mut feed, 400);

    assert_eq!(read_words(&tile, 0x200, 4), [1, 2, 3, 4]);
    assert_eq!(read_words(&tile, 0x100, 4), [5, 6, 7, 8]);
    assert!(!engine.channel_has_pending_work(S2MM0));
}

#[test]
fn a_packet_waiting_on_its_bd_lock_blocks_the_channel() {
    // Head-of-line: the channel is one FIFO. A packet whose BD cannot acquire
    // its lock stalls every packet behind it (mlir-aie dma_s2mm_ooo README).
    let mut engine = ooo_engine();
    let mut tile = Tile::compute(1, 2);
    engine
        .configure_bd(9, BdConfig::simple_1d(0x200, 16).with_acquire(3, 1))
        .unwrap();
    engine.configure_bd(5, BdConfig::simple_1d(0x100, 16)).unwrap();
    engine.enqueue_task(S2MM0, 0, 1, false);

    let mut feed: VecDeque<_> = [packet(9, &[1, 2, 3, 4]), packet(5, &[5, 6, 7, 8])].concat().into();
    run(&mut engine, &mut tile, &mut feed, 300);
    assert_eq!(read_words(&tile, 0x100, 4), [0, 0, 0, 0], "BD 5's packet is stuck behind BD 9's");

    tile.locks[3].set(1);
    run(&mut engine, &mut tile, &mut feed, 300);
    assert_eq!(read_words(&tile, 0x200, 4), [1, 2, 3, 4]);
    assert_eq!(read_words(&tile, 0x100, 4), [5, 6, 7, 8]);
    assert!(!engine.channel_has_pending_work(S2MM0));
}

#[test]
fn a_header_naming_no_bd_raises_bd_unavailable() {
    let mut engine = ooo_engine();
    let mut tile = Tile::compute(1, 2);
    engine.configure_bd(5, BdConfig::simple_1d(0x100, 16)).unwrap();
    engine.enqueue_task(S2MM0, 0, 0, false);

    // A compute tile has 16 BDs; id 40 names none of them.
    let mut feed: VecDeque<_> = packet(40, &[1, 2, 3, 4]).into();
    run(&mut engine, &mut tile, &mut feed, 100);

    assert!(engine.error_bd_unavailable(S2MM0));
    assert_eq!(read_words(&tile, 0x100, 4), [0, 0, 0, 0]);
}

#[test]
fn in_order_channel_still_writes_the_header_as_data() {
    // NPU1 silicon (hdr-probe): an in-order S2MM with keep_pkt_header stores
    // the header as the first word. Out-of-order handling must not leak in.
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = Tile::compute(1, 2);
    engine.configure_bd(0, BdConfig::simple_1d(0x100, 20)).unwrap();
    engine.enqueue_task(S2MM0, 0, 0, false);

    let words = packet(9, &[1, 2, 3, 4]);
    let header = words[0].data;
    let mut feed: VecDeque<_> = words.into();
    run(&mut engine, &mut tile, &mut feed, 300);

    assert_eq!(read_words(&tile, 0x100, 5), [header, 1, 2, 3, 4]);
}
