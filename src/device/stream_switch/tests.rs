use super::*;

#[test]
fn test_stream_port_fifo() {
    let mut port = StreamPort::new(0, PortDirection::Master, PortType::Dma(0));

    assert!(!port.has_data());
    assert!(port.can_accept());

    port.push(0xDEADBEEF);
    assert!(port.has_data());
    assert_eq!(port.peek(), Some(0xDEADBEEF));

    let data = port.pop();
    assert_eq!(data, Some(0xDEADBEEF));
    assert!(!port.has_data());
}

#[test]
fn test_stream_port_backpressure() {
    let mut port = StreamPort::new(0, PortDirection::Master, PortType::Dma(0));

    // Fill the FIFO
    while port.can_accept() {
        port.push(0x12345678);
    }

    assert!(port.is_full());
    assert!(!port.push(0xFFFFFFFF)); // Should fail
}

/// PORT_RUNNING (`cycle_beat`) watches a stream-switch port's ONE external
/// AXI4-Stream interface. A SLAVE port's interface is its input (the push from
/// upstream); the pop that drains it into the internal crossbar is NOT an
/// external handshake and must not beat. Otherwise a word buffered for >=1
/// cycle (push on cycle N, crossbar-pop on cycle M>N) double-counts, breaking
/// the HW law `sum(PORT_RUNNING) == words`. Derived from the master/slave port
/// selection in `Stream_Switch_Event_Port_Selection` (AM025) -- see #140.
#[test]
fn slave_port_beats_on_external_push_not_internal_pop() {
    let mut port = StreamPort::new(0, PortDirection::Slave, PortType::South);

    // External input handshake (upstream -> slave): beats.
    port.cycle_beat = false;
    port.push(0xAA);
    assert!(port.cycle_beat, "slave beats on its external input (push)");

    // Next cycle the crossbar drains it (begin_routing_cycle clears cycle_beat).
    port.cycle_beat = false;
    assert_eq!(port.pop(), Some(0xAA));
    assert!(!port.cycle_beat, "slave must NOT beat on the internal crossbar pop (not an external handshake)");
}

/// A MASTER port's external interface is its output (the pop driving
/// downstream). The crossbar push that fills it is internal and must not beat.
#[test]
fn master_port_beats_on_external_pop_not_internal_push() {
    let mut port = StreamPort::new(0, PortDirection::Master, PortType::South);

    // Internal crossbar fill (crossbar -> master): must NOT beat.
    port.cycle_beat = false;
    port.push(0xBB);
    assert!(
        !port.cycle_beat,
        "master must NOT beat on the internal crossbar push (not an external handshake)"
    );

    // Next cycle the master drives the word downstream: external pop beats.
    port.cycle_beat = false;
    assert_eq!(port.pop(), Some(0xBB));
    assert!(port.cycle_beat, "master beats on its external output (pop)");
}

#[test]
fn test_stream_switch_compute() {
    let ss = StreamSwitch::new_compute_tile(1, 2);

    // Per AM025 AIE_TILE_MODULE: Compute tile has 2 DMA channels (0-1).
    // S2MM (slaves) and MM2S (masters) are at the same channel indices.
    assert!(ss.dma_slave(0).is_some(), "Should have DMA S2MM channel 0");
    assert!(ss.dma_slave(1).is_some(), "Should have DMA S2MM channel 1");
    assert!(ss.dma_master(0).is_some(), "Should have DMA MM2S channel 0");
    assert!(ss.dma_master(1).is_some(), "Should have DMA MM2S channel 1");

    // Verify port counts per AM025 CORE_MODULE/STREAM_SWITCH (matches aie-rt AieMlTileStrmMstr/Slv):
    // Masters: 0=Core, 1-2=DMA(2), 3=Tile_Ctrl, 4=FIFO0, 5-8=South(4), 9-12=West(4),
    //          13-18=North(6), 19-22=East(4) = 23 total
    // Slaves:  0=Core, 1-2=DMA(2), 3=Tile_Ctrl, 4=FIFO0, 5-10=South(6), 11-14=West(4),
    //          15-18=North(4), 19-22=East(4), 23-24=Trace(2) = 25 total
    assert_eq!(ss.masters.len(), 23);
    assert_eq!(ss.slaves.len(), 25);
}

#[test]
fn test_stream_switch_mem_tile() {
    let ss = StreamSwitch::new_mem_tile(0, 1);

    // Per AM025 MEMORY_TILE_MODULE: MemTile has 6 DMA channels (0-5).
    for i in 0..6 {
        assert!(ss.dma_slave(i).is_some(), "Should have DMA S2MM channel {}", i);
        assert!(ss.dma_master(i).is_some(), "Should have DMA MM2S channel {}", i);
    }

    // Verify port counts per AM025:
    // Masters: 0-5=DMA, 6=Tile_Ctrl, 7-10=South(4), 11-16=North(6) = 17 total
    // Slaves: 0-5=DMA, 6=Tile_Ctrl, 7-12=South(6), 13-16=North(4), 17=Trace = 18 total
    assert_eq!(ss.masters.len(), 17);
    assert_eq!(ss.slaves.len(), 18);

    // Verify asymmetric N/S connectivity (key architectural feature)
    let south_masters = ss.masters.iter().filter(|p| matches!(p.port_type, PortType::South)).count();
    let south_slaves = ss.slaves.iter().filter(|p| matches!(p.port_type, PortType::South)).count();
    let north_masters = ss.masters.iter().filter(|p| matches!(p.port_type, PortType::North)).count();
    let north_slaves = ss.slaves.iter().filter(|p| matches!(p.port_type, PortType::North)).count();

    assert_eq!(south_masters, 4, "MemTile should have 4 South masters");
    assert_eq!(south_slaves, 6, "MemTile should have 6 South slaves");
    assert_eq!(north_masters, 6, "MemTile should have 6 North masters");
    assert_eq!(north_slaves, 4, "MemTile should have 4 North slaves");
}

#[test]
fn test_stream_switch_shim() {
    let ss = StreamSwitch::new_shim_tile(0);

    // Shim should be at row 0
    assert_eq!(ss.row, 0);

    // Verify port counts per AM025 PL_MODULE:
    // Masters: 0=Ctrl, 1=FIFO, 2-7=South(6), 8-11=West(4), 12-17=North(6), 18-21=East(4) = 22 total
    // Slaves: 0=Ctrl, 1=FIFO, 2-9=South(8), 10-13=West(4), 14-17=North(4), 18-21=East(4), 22=Trace = 23 total
    assert_eq!(ss.masters.len(), 22, "Shim should have 22 master ports");
    assert_eq!(ss.slaves.len(), 23, "Shim should have 23 slave ports");

    // Verify 6 North masters for connecting to MemTile
    let north_masters = ss.masters.iter().filter(|p| matches!(p.port_type, PortType::North)).count();
    assert_eq!(north_masters, 6, "Shim should have 6 North masters");
}

#[test]
fn test_route_configuration() {
    let mut port = StreamPort::new(0, PortDirection::Master, PortType::Dma(0));

    assert!(!port.enabled);

    port.set_route(1, 2, 3);
    assert!(port.enabled);
    assert_eq!(port.route_to, Some((1, 2, 3)));

    port.clear_route();
    assert!(!port.enabled);
    assert!(port.route_to.is_none());
}

#[test]
fn test_stream_packet() {
    let pkt = StreamPacket::new(0xCAFEBABE, 0, 1, 2, 1, 2, 3);

    assert_eq!(pkt.data, 0xCAFEBABE);
    assert_eq!(pkt.src_col, 0);
    assert_eq!(pkt.dest_col, 1);
    assert!(!pkt.is_last);

    let pkt_last = pkt.with_last();
    assert!(pkt_last.is_last);
}

#[test]
fn test_local_route() {
    let route = LocalRoute::new(0, 1);
    assert_eq!(route.slave_idx, 0);
    assert_eq!(route.master_idx, 1);
    assert!(route.enabled);
}

#[test]
fn test_configure_local_route() {
    let mut ss = StreamSwitch::new_compute_tile(0, 2);

    assert_eq!(ss.local_route_count(), 0);

    // Configure route from slave 0 to master 0
    ss.configure_local_route(0, 0);
    assert_eq!(ss.local_route_count(), 1);

    // Duplicate route should not be added
    ss.configure_local_route(0, 0);
    assert_eq!(ss.local_route_count(), 1);

    // Different route should be added
    ss.configure_local_route(1, 1);
    assert_eq!(ss.local_route_count(), 2);
}

#[test]
fn test_switch_step_basic() {
    let mut ss = StreamSwitch::new_compute_tile(0, 2);

    // Configure route from slave 0 (Core) to master 0 (Core) -- local->local = 3 cycles
    ss.configure_local_route(0, 0);
    let latency = ss.local_routes[0].latency;
    assert_eq!(latency, 3, "Core->Core should be local->local latency");

    // Put data in slave port
    ss.slaves[0].push(0xDEADBEEF);
    assert!(ss.slaves[0].has_data());
    assert!(!ss.masters[0].has_data());

    // First step: data enters pipeline (popped from slave)
    let forwarded = ss.step();
    assert_eq!(forwarded, 0, "Data in pipeline, not yet delivered");
    assert!(!ss.slaves[0].has_data(), "Slave popped");
    assert!(!ss.masters[0].has_data(), "Master not yet received");

    // Step through pipeline: needs 'latency' more steps to deliver
    // (countdown: latency -> latency-1 -> ... -> 1 -> 0=deliver)
    for _ in 0..latency {
        ss.step();
    }

    // After latency+1 total steps, data should be in master
    assert!(ss.masters[0].has_data(), "Delivered after {} pipeline cycles", latency);
    assert_eq!(ss.masters[0].peek(), Some(0xDEADBEEF));
}

#[test]
fn test_switch_step_multiple_routes() {
    let mut ss = StreamSwitch::new_compute_tile(0, 2);

    // Configure two routes (both local->local)
    ss.configure_local_route(0, 0);
    ss.configure_local_route(1, 1);
    let latency = ss.local_routes[0].latency;

    // Put data in both slave ports
    ss.slaves[0].push(0x11111111);
    ss.slaves[1].push(0x22222222);

    // Step through pipeline: 1 accept + latency delivery
    for _ in 0..=latency {
        ss.step();
    }

    assert_eq!(ss.masters[0].pop(), Some(0x11111111));
    assert_eq!(ss.masters[1].pop(), Some(0x22222222));
}

#[test]
fn test_switch_step_backpressure() {
    let mut ss = StreamSwitch::new_compute_tile(0, 2);
    ss.configure_local_route(0, 0);
    let latency = ss.local_routes[0].latency;

    // Fill the master port's FIFO
    while ss.masters[0].can_accept() {
        ss.masters[0].push(0x99999999);
    }

    // Put data in slave
    ss.slaves[0].push(0xDEADBEEF);

    // Step through pipeline -- data should NOT be delivered (master full)
    for _ in 0..latency + 2 {
        ss.step();
    }
    assert!(!ss.slaves[0].has_data(), "Slave was popped into pipeline");
    // Data is stuck in pipeline (backpressure at master)
    assert!(!ss.switch_pipeline.is_empty(), "Data stuck in pipeline due to backpressure");

    // Make room in master
    ss.masters[0].pop();

    // Next step should deliver
    ss.step();
    assert!(ss.switch_pipeline.is_empty(), "Delivered after backpressure cleared");
}

#[test]
fn test_switch_step_no_data() {
    let mut ss = StreamSwitch::new_compute_tile(0, 2);
    ss.configure_local_route(0, 0);

    // No data in slave - step does nothing
    let forwarded = ss.step();
    assert_eq!(forwarded, 0);
}

#[test]
fn test_has_pending_local() {
    let mut ss = StreamSwitch::new_compute_tile(0, 2);
    ss.configure_local_route(0, 0);

    assert!(!ss.has_pending_local());

    ss.slaves[0].push(0x12345678);
    assert!(ss.has_pending_local());

    // After 1 step, data is in pipeline (still pending)
    ss.step();
    assert!(ss.has_pending_local(), "Pipeline has in-flight data");

    // Step through pipeline delivery
    let latency = ss.local_routes[0].latency;
    for _ in 0..latency {
        ss.step();
    }
    // Data delivered to master -- master has data so has_pending_data is true
    // but has_pending_local checks slaves and pipeline, not masters
    // Pipeline should be empty now, and slave is empty
    assert!(
        !ss.switch_pipeline.is_empty() || ss.masters[0].has_data(),
        "Data should be delivered or in pipeline"
    );
    // Clear pipeline check
    ss.masters[0].pop();
    assert!(!ss.has_pending_local(), "Pipeline empty after delivery and drain");
}

#[test]
fn test_remove_local_route() {
    let mut ss = StreamSwitch::new_compute_tile(0, 2);

    ss.configure_local_route(0, 0);
    ss.configure_local_route(1, 1);
    assert_eq!(ss.local_route_count(), 2);

    ss.remove_local_route(0, 0);
    assert_eq!(ss.local_route_count(), 1);

    ss.clear_local_routes();
    assert_eq!(ss.local_route_count(), 0);
}

// ========================================================================
// Circuit-mode Multicast Tests
// ========================================================================

#[test]
fn test_circuit_multicast_one_slave_two_masters() {
    // One slave port routes to two master ports (multicast).
    // Both masters should receive the same word.
    let mut ss = StreamSwitch::new_compute_tile(0, 2);
    ss.configure_local_route(0, 0); // slave 0 -> master 0
    ss.configure_local_route(0, 1); // slave 0 -> master 1 (multicast)

    // Push one word to the slave
    ss.slaves[0].push_with_tlast(0xCAFEBABE, false);

    // Determine pipeline latency (should be same for both routes)
    let latency = ss.local_routes[0].latency;

    // Step: should pop once, enter pipeline for both masters
    ss.step();
    assert!(!ss.slaves[0].has_data(), "Slave data should be consumed");
    // Pipeline should have 2 entries (one per destination)
    assert_eq!(ss.switch_pipeline.len(), 2, "Both multicast destinations should be in pipeline");

    // Step through pipeline delivery
    for _ in 0..latency {
        ss.step();
    }

    // Both masters should have received the word
    assert_eq!(ss.masters[0].pop(), Some(0xCAFEBABE), "Master 0 should get multicast data");
    assert_eq!(ss.masters[1].pop(), Some(0xCAFEBABE), "Master 1 should get multicast data");
}

#[test]
fn test_circuit_multicast_preserves_tlast() {
    let mut ss = StreamSwitch::new_compute_tile(0, 2);
    ss.configure_local_route(0, 0);
    ss.configure_local_route(0, 1);

    // Push word with TLAST
    ss.slaves[0].push_with_tlast(0xDEAD0001, true);

    let latency = ss.local_routes[0].latency;
    for _ in 0..=latency {
        ss.step();
    }

    // Both should get data with TLAST preserved
    let (d0, t0) = ss.masters[0].pop_with_tlast().unwrap();
    let (d1, t1) = ss.masters[1].pop_with_tlast().unwrap();
    assert_eq!(d0, 0xDEAD0001);
    assert_eq!(d1, 0xDEAD0001);
    assert!(t0, "Master 0 should see TLAST");
    assert!(t1, "Master 1 should see TLAST");
}

#[test]
fn test_circuit_multicast_multiple_words() {
    // Stream 4 words through a multicast route, verify all arrive at both destinations.
    let mut ss = StreamSwitch::new_compute_tile(0, 2);
    ss.configure_local_route(0, 0);
    ss.configure_local_route(0, 1);

    let words = [0x11111111u32, 0x22222222, 0x33333333, 0x44444444];
    for &w in &words {
        ss.slaves[0].push(w);
    }

    // Step and drain masters as words arrive. Master FIFO depth is only 2,
    // so we must consume delivered words to make room for the next batch.
    let mut received_0 = Vec::new();
    let mut received_1 = Vec::new();
    for _ in 0..40 {
        ss.step();
        while let Some(w) = ss.masters[0].pop() {
            received_0.push(w);
        }
        while let Some(w) = ss.masters[1].pop() {
            received_1.push(w);
        }
    }

    // Verify all words arrived at both masters in order
    assert_eq!(received_0, words.to_vec(), "Master 0 should get all multicast words in order");
    assert_eq!(received_1, words.to_vec(), "Master 1 should get all multicast words in order");
}

// ========================================================================
// Per-route backpressure (slave-pop gated on pipeline + master FIFO budget)
// ========================================================================

#[test]
fn test_per_route_backpressure_caps_pipeline_at_latency_plus_master_fifo() {
    // Regression test for the memtile_dmas/blockwrite_using_locks wedge.
    //
    // step() used to drain slave FIFOs unconditionally into an unbounded
    // switch_pipeline Vec, so a producer (DMA MM2S) feeding a slave whose
    // route led to a fully-backpressured master never saw its slave fill --
    // every cycle, the slave was popped and the word buried in pipeline.  A
    // self-chained MM2S BD would then run forever without ever asserting
    // Stalled_Stream_Backpressure, and the run loop never reached natural
    // completion.
    //
    // After the fix the slave-pop is gated by the per-route in-flight budget
    // (`latency + master.fifo_capacity`).  Once that budget is reached the
    // slave retains data and backpressures upstream.
    let mut ss = StreamSwitch::new_compute_tile(0, 2);
    ss.configure_local_route(0, 0);
    let latency = ss.local_routes[0].latency as usize;
    let master_cap = ss.masters[0].fifo_capacity;
    let budget = latency + master_cap;

    // Fill master so deliveries can't drain in-flight words; pipeline is
    // the only place words can pool.
    while ss.masters[0].can_accept() {
        ss.masters[0].push(0x0);
    }

    // Fill the slave to capacity.  With master full and slave full, the
    // pipeline can absorb at most `latency` more words before the budget
    // is reached and the slave starts to back up.
    let slave_cap = ss.slaves[0].fifo_capacity;
    assert!(
        slave_cap > latency,
        "test setup needs slave capacity > latency ({} vs {}); the leak vs fixed comparison \
         only makes sense if some slave words must be retained when the budget is reached",
        slave_cap,
        latency,
    );
    for i in 0..slave_cap {
        assert!(ss.slaves[0].push(i as u32), "slave push must succeed up to capacity");
    }
    let pushed_to_slave = ss.slaves[0].fifo.len();

    // Run more cycles than the budget so a leaky step() would have moved
    // every slave word into the pipeline by now.
    for _ in 0..(budget * 4) {
        ss.step();
    }

    // Master stayed full (we never popped it), so no entries drained from
    // the pipeline.  Pipeline + master FIFO must be exactly at budget; any
    // further pops would exceed it.
    let in_flight = ss.switch_pipeline.len() + ss.masters[0].fifo.len();
    assert_eq!(in_flight, budget, "in-flight must equal budget at steady state");

    // The slave retains words -- this is the backpressure signal that
    // lets an upstream DMA MM2S stall in Transferring.  Master starts
    // full (cap), so only `latency` slave words could enter the pipeline.
    let expected_remaining = pushed_to_slave - latency;
    assert_eq!(
        ss.slaves[0].fifo.len(),
        expected_remaining,
        "slave must retain {} words once budget is reached; leaky pop would have emptied it",
        expected_remaining,
    );

    // Drain one master slot.  The pipeline should advance one word to
    // master, which opens budget for one more slave-pop.
    ss.masters[0].pop();
    ss.step();
    let in_flight_after = ss.switch_pipeline.len() + ss.masters[0].fifo.len();
    assert_eq!(in_flight_after, budget, "drain + step restores budget exactly");
    assert_eq!(
        ss.slaves[0].fifo.len(),
        expected_remaining - 1,
        "draining the master must let exactly one slave word into the pipeline",
    );
}

// ========================================================================
// Packet Header Tests
// ========================================================================

#[test]
fn test_packet_header_new() {
    let header = PacketHeader::new(5, 2, 3);
    assert_eq!(header.stream_id, 5);
    assert_eq!(header.src_col, 2);
    assert_eq!(header.src_row, 3);
    assert_eq!(header.packet_type, PacketType::Data);
}

#[test]
fn test_packet_header_with_type() {
    let header = PacketHeader::new(10, 1, 2).with_type(PacketType::Control);
    assert_eq!(header.stream_id, 10);
    assert_eq!(header.packet_type, PacketType::Control);
}

#[test]
fn test_packet_header_encode_decode() {
    let original = PacketHeader::new(7, 3, 4);
    let encoded = original.encode();
    let (decoded, parity_ok) = PacketHeader::decode(encoded);

    assert!(parity_ok, "Parity check should pass");
    assert_eq!(decoded.stream_id, original.stream_id);
    assert_eq!(decoded.src_col, original.src_col);
    assert_eq!(decoded.src_row, original.src_row);
    assert_eq!(decoded.packet_type, original.packet_type);
}

#[test]
fn test_packet_header_encode_decode_all_types() {
    for ptype in [PacketType::Data, PacketType::Control, PacketType::Config, PacketType::Trace] {
        let original = PacketHeader::new(15, 5, 6).with_type(ptype);
        let encoded = original.encode();
        let (decoded, parity_ok) = PacketHeader::decode(encoded);

        assert!(parity_ok);
        assert_eq!(decoded.packet_type, ptype);
    }
}

#[test]
fn test_packet_header_field_masks() {
    // Test with maximum values for each field
    let header = PacketHeader {
        stream_id: 0x1F, // 5 bits max
        packet_type: PacketType::Reserved,
        src_row: 0x1F, // 5 bits max
        src_col: 0x7F, // 7 bits max
    };

    let encoded = header.encode();
    let (decoded, _) = PacketHeader::decode(encoded);

    assert_eq!(decoded.stream_id, 0x1F);
    assert_eq!(decoded.src_row, 0x1F);
    assert_eq!(decoded.src_col, 0x7F);
}

#[test]
fn test_packet_header_parity_error() {
    let header = PacketHeader::new(5, 2, 3);
    let mut encoded = header.encode();

    // Flip a bit to corrupt parity
    encoded ^= 0x100;

    let (_, parity_ok) = PacketHeader::decode(encoded);
    assert!(!parity_ok, "Parity should fail after bit flip");
}

// ========================================================================
// Packet Switch Tests
// ========================================================================

#[test]
fn test_packet_switch_new() {
    let ps = PacketSwitch::new();
    assert_eq!(ps.route_count(), 0);
    assert!(!ps.in_packet());
}

#[test]
fn test_packet_switch_add_route() {
    let mut ps = PacketSwitch::new();

    ps.add_route(5, 0);
    assert_eq!(ps.route_count(), 1);

    // Adding same stream ID adds to existing route
    ps.add_route(5, 1);
    assert_eq!(ps.route_count(), 1);

    // Lookup should return both ports
    let dests = ps.lookup(5).unwrap();
    assert_eq!(dests.len(), 2);
    assert!(dests.contains(&0));
    assert!(dests.contains(&1));
}

#[test]
fn test_packet_switch_multicast_route() {
    let mut ps = PacketSwitch::new();

    ps.add_multicast_route(10, vec![0, 1, 2]);
    assert_eq!(ps.route_count(), 1);

    let dests = ps.lookup(10).unwrap();
    assert_eq!(dests, &[0, 1, 2]);
}

#[test]
fn test_packet_switch_lookup_not_found() {
    let ps = PacketSwitch::new();
    assert!(ps.lookup(5).is_none());
}

#[test]
fn test_packet_switch_remove_route() {
    let mut ps = PacketSwitch::new();

    ps.add_route(5, 0);
    ps.add_route(10, 1);
    assert_eq!(ps.route_count(), 2);

    ps.remove_route(5);
    assert_eq!(ps.route_count(), 1);
    assert!(ps.lookup(5).is_none());
    assert!(ps.lookup(10).is_some());
}

#[test]
fn test_packet_switch_process_header() {
    let mut ps = PacketSwitch::new();
    ps.add_route(5, 2);

    let header = PacketHeader::new(5, 1, 3);
    let encoded = header.encode();

    let result = ps.process_header(encoded);
    assert!(result.is_some());

    let (decoded, dests) = result.unwrap();
    assert_eq!(decoded.stream_id, 5);
    assert_eq!(dests, vec![2]);

    // Should be in packet now
    assert!(ps.in_packet());
    assert!(ps.has_arb_delay());
}

#[test]
fn test_packet_switch_process_header_no_route() {
    let mut ps = PacketSwitch::new();
    // No routes configured

    let header = PacketHeader::new(5, 1, 3);
    let encoded = header.encode();

    let result = ps.process_header(encoded);
    assert!(result.is_none(), "Should return None for unknown stream ID");
}

#[test]
fn test_packet_switch_arb_delay() {
    let mut ps = PacketSwitch::new();
    ps.add_route(5, 0);

    let header = PacketHeader::new(5, 1, 2);
    ps.process_header(header.encode());

    // With 1-cycle overhead, first tick should complete
    assert!(ps.has_arb_delay());

    // Tick until complete (may be immediate if overhead is 1)
    while ps.has_arb_delay() {
        ps.tick_arb_delay();
    }
    assert!(!ps.has_arb_delay());
}

#[test]
fn test_packet_switch_complete_packet() {
    let mut ps = PacketSwitch::new();
    ps.add_route(5, 0);

    let header = PacketHeader::new(5, 1, 2);
    ps.process_header(header.encode());

    // Count some data words
    ps.count_data_word();
    ps.count_data_word();
    ps.count_data_word();

    // Complete packet
    let result = ps.complete_packet();
    assert!(result.is_some());

    let (completed_header, word_count) = result.unwrap();
    assert_eq!(completed_header.stream_id, 5);
    assert_eq!(word_count, 3);

    // No longer in packet
    assert!(!ps.in_packet());
}

#[test]
fn test_packet_type_from_u8() {
    assert_eq!(PacketType::from_u8(0), PacketType::Data);
    assert_eq!(PacketType::from_u8(1), PacketType::Control);
    assert_eq!(PacketType::from_u8(2), PacketType::Config);
    assert_eq!(PacketType::from_u8(3), PacketType::Trace);
    assert_eq!(PacketType::from_u8(4), PacketType::Reserved);
    assert_eq!(PacketType::from_u8(7), PacketType::Reserved);
}

// ========================================================================
// Arbiter Locking Tests
// ========================================================================

/// Helper: build a slave slot register value.
/// Layout: id[28:24], mask[20:16], enable[8], msel[5:4], arbiter[2:0]
fn make_slot_reg(pkt_id: u8, mask: u8, msel: u8, arbiter: u8) -> u32 {
    ((pkt_id as u32) << 24) | ((mask as u32) << 16) | (1 << 8) | ((msel as u32) << 4) | (arbiter as u32)
}

/// Helper: build a master packet config register value.
/// Layout: enable[31], packet_enable[30], drop_header[7], msel_enable[6:3], arbiter[2:0]
fn make_master_pkt_reg(arbiter: u8, msel_enable: u8, drop_header: bool) -> u32 {
    (1 << 31)
        | (1 << 30)
        | if drop_header { 1 << 7 } else { 0 }
        | ((msel_enable as u32) << 3)
        | (arbiter as u32)
}

#[test]
fn test_arbiter_lock_prevents_interleave() {
    // Two slaves route through the SAME arbiter to the same master.
    // With locking, one completes its full packet before the other starts.
    let mut ss = StreamSwitch::new_compute_tile(0, 2);

    // Enable packet mode on trace slave ports
    ss.slaves[23].packet_enable = true;
    ss.slaves[24].packet_enable = true;
    // Slave 23 (core trace): pkt_id=1, arbiter=0, msel=0
    ss.configure_slave_slot(23, 0, make_slot_reg(1, 0x1F, 0, 0));
    // Slave 24 (mem trace): pkt_id=2, arbiter=0, msel=0
    ss.configure_slave_slot(24, 0, make_slot_reg(2, 0x1F, 0, 0));

    // Master 7: packet mode, arbiter=0, msel_enable=0b0001 (accepts msel=0)
    ss.configure_master_packet(7, make_master_pkt_reg(0, 0b0001, false));

    // Build two 4-word packets (header + 3 data + TLAST on last).
    // Slave FIFO is 4 deep, so these fit. Master FIFO is only 2 deep,
    // so we drain between steps to simulate downstream consumption.
    let hdr1 = PacketHeader::new(1, 0, 2).with_type(PacketType::Trace);
    let hdr2 = PacketHeader::new(2, 0, 2).with_type(PacketType::Trace);

    // Packet 1 on slave 23
    ss.slaves[23].push_with_tlast(hdr1.encode(), false);
    ss.slaves[23].push_with_tlast(0xAAAA_0001, false);
    ss.slaves[23].push_with_tlast(0xAAAA_0002, false);
    ss.slaves[23].push_with_tlast(0xAAAA_0003, true);

    // Packet 2 on slave 24
    ss.slaves[24].push_with_tlast(hdr2.encode(), false);
    ss.slaves[24].push_with_tlast(0xBBBB_0001, false);
    ss.slaves[24].push_with_tlast(0xBBBB_0002, false);
    ss.slaves[24].push_with_tlast(0xBBBB_0003, true);

    // Step and drain the master each time (simulates downstream consumer)
    let mut output: Vec<u32> = Vec::new();
    for _ in 0..20 {
        ss.step();
        while let Some((word, _)) = ss.masters[7].pop_with_tlast() {
            output.push(word);
        }
    }

    // Should have all 8 words (2 x 4-word packets)
    assert_eq!(output.len(), 8, "expected 8 words, got {}: {:08X?}", output.len(), output);

    // First 4 words must ALL be from the same packet (no interleaving).
    // Slave 23 has lower index, so it gets priority.
    let first_pkt_id = output[0] & 0x1F;
    assert_eq!(first_pkt_id, 1, "slave 23 (lower index) should go first");

    // Words 1-3 of first packet: 0xAAAA_xxxx
    for i in 1..4 {
        assert_eq!(output[i] >> 16, 0xAAAA, "word {} should be from packet 1, got 0x{:08X}", i, output[i]);
    }

    // Second packet: words 4-7
    let second_pkt_id = output[4] & 0x1F;
    assert_eq!(second_pkt_id, 2, "packet 2 should follow");
    for i in 5..8 {
        assert_eq!(output[i] >> 16, 0xBBBB, "word {} should be from packet 2, got 0x{:08X}", i, output[i]);
    }
}

#[test]
fn test_different_arbiters_no_contention() {
    // Two slaves use DIFFERENT arbiters -- both proceed simultaneously.
    let mut ss = StreamSwitch::new_compute_tile(0, 2);

    // Enable packet mode on trace slave ports
    ss.slaves[23].packet_enable = true;
    ss.slaves[24].packet_enable = true;
    // Slave 23: pkt_id=1, arbiter=0, msel=0
    ss.configure_slave_slot(23, 0, make_slot_reg(1, 0x1F, 0, 0));
    // Slave 24: pkt_id=2, arbiter=1, msel=0
    ss.configure_slave_slot(24, 0, make_slot_reg(2, 0x1F, 0, 1));

    // Master 7: arbiter=0, master 8: arbiter=1
    ss.configure_master_packet(7, make_master_pkt_reg(0, 0b0001, false));
    ss.configure_master_packet(8, make_master_pkt_reg(1, 0b0001, false));

    let hdr1 = PacketHeader::new(1, 0, 2).with_type(PacketType::Trace);
    let hdr2 = PacketHeader::new(2, 0, 2).with_type(PacketType::Trace);

    // 2-word packets (header + data+TLAST) fit within 2-deep master FIFO
    ss.slaves[23].push_with_tlast(hdr1.encode(), false);
    ss.slaves[23].push_with_tlast(0xAAAA_0001, true);

    ss.slaves[24].push_with_tlast(hdr2.encode(), false);
    ss.slaves[24].push_with_tlast(0xBBBB_0001, true);

    // Step 1: both headers forward in parallel (different arbiters)
    let words = ss.step();
    assert_eq!(words, 2, "both slaves should forward in parallel");

    // Step 2: both TLAST data words forward in parallel
    let words2 = ss.step();
    assert_eq!(words2, 2, "both data words should forward in parallel");

    // Verify master 7 got packet 1, master 8 got packet 2
    let mut m7: Vec<u32> = Vec::new();
    let mut m8: Vec<u32> = Vec::new();
    while let Some(w) = ss.masters[7].pop() {
        m7.push(w);
    }
    while let Some(w) = ss.masters[8].pop() {
        m8.push(w);
    }

    assert_eq!(m7.len(), 2, "master 7 should have 2 words");
    assert_eq!(m8.len(), 2, "master 8 should have 2 words");
    assert_eq!(m7[0] & 0x1F, 1, "master 7 = packet 1");
    assert_eq!(m8[0] & 0x1F, 2, "master 8 = packet 2");
}

#[test]
fn test_arbiter_lock_released_on_tlast() {
    // Verify the arbiter is freed after TLAST so the next packet can proceed.
    let mut ss = StreamSwitch::new_compute_tile(0, 2);

    // Enable packet mode on trace slave ports
    ss.slaves[23].packet_enable = true;
    ss.slaves[24].packet_enable = true;
    // Slave 23: arbiter=0
    ss.configure_slave_slot(23, 0, make_slot_reg(1, 0x1F, 0, 0));
    // Slave 24: arbiter=0 (same!)
    ss.configure_slave_slot(24, 0, make_slot_reg(2, 0x1F, 0, 0));
    // Master 7: arbiter=0
    ss.configure_master_packet(7, make_master_pkt_reg(0, 0b0001, false));

    // First packet: 2 words (header + data+TLAST) from slave 23
    let hdr1 = PacketHeader::new(1, 0, 2).with_type(PacketType::Trace);
    ss.slaves[23].push_with_tlast(hdr1.encode(), false);
    ss.slaves[23].push_with_tlast(0xAAAA_0001, true);

    // Step and drain: process first packet completely
    ss.step(); // header forwarded
    ss.masters[7].pop(); // drain header (make room)
    ss.step(); // data+TLAST forwarded, arbiter released
    ss.masters[7].pop(); // drain data

    // Arbiter should be free now
    assert!(ss.arbiter_locks[0].is_none(), "arbiter 0 should be free after TLAST");

    // Now slave 24 should be able to send through the same arbiter
    let hdr2 = PacketHeader::new(2, 0, 2).with_type(PacketType::Trace);
    ss.slaves[24].push_with_tlast(hdr2.encode(), false);
    ss.slaves[24].push_with_tlast(0xBBBB_0001, true);

    let words = ss.step(); // header from slave 24
    assert_eq!(words, 1, "slave 24 should now use arbiter 0");

    // Verify master 7 got packet 2's header
    let (w, _) = ss.masters[7].pop_with_tlast().unwrap();
    assert_eq!(w & 0x1F, 2, "should be packet 2 header");
}

#[test]
fn test_cycle_active_tracks_port_activity() {
    // cycle_active captures whether a port had data at any point during
    // the routing cycle. This is needed for PORT_RUNNING trace events
    // because data enters and exits FIFOs within a single route_streams()
    // call, making between-step has_data() checks always see empty ports.
    let mut port = StreamPort::new(0, PortDirection::Slave, PortType::Dma(0));

    // Initially not active
    assert!(!port.cycle_active);
    assert!(!port.has_data());

    // Push marks active
    port.push_with_tlast(0xAAAA, false);
    assert!(port.cycle_active);

    // Pop drains data, but cycle_active persists
    port.pop();
    assert!(!port.has_data(), "FIFO should be empty after pop");
    assert!(port.cycle_active, "cycle_active should persist after pop");
}

#[test]
fn test_cycle_beat_tracks_actual_beat_crossings() {
    // cycle_beat is the HW-faithful PORT_RUNNING signal: it is set only when a
    // beat crosses the port's ONE external AXI interface this cycle, and --
    // unlike cycle_active -- it is NOT seeded from has_data() at
    // begin_routing_cycle. This distinguishes "running" (external beat) from
    // "idle-with-buffered-data" (a port holding residual FIFO data). A master
    // port (here the S2MM-feeding side) watches its OUTPUT, so its external beat
    // is the pop; the crossbar push that fills it is internal. (The slave-side
    // push semantics are covered by
    // slave_port_beats_on_external_push_not_internal_pop.)
    let mut port = StreamPort::new(0, PortDirection::Master, PortType::Dma(0));
    assert!(!port.cycle_beat);

    // The crossbar fills a master internally -- not an external handshake.
    port.push_with_tlast(0x1111, false);
    assert!(!port.cycle_beat, "master push (internal crossbar fill) must NOT set cycle_beat");

    // The master drives its downstream consumer -- external output handshake.
    port.cycle_beat = false;
    port.pop();
    assert!(port.cycle_beat, "master pop (external output handshake) sets cycle_beat");
}

#[test]
fn test_begin_routing_cycle_does_not_seed_beat_from_fifo() {
    // The crux of the receive-port fix: a port that merely HOLDS buffered data
    // (no beat this cycle) must NOT be marked running. begin_routing_cycle
    // seeds cycle_active from has_data() (for clock gating) but leaves
    // cycle_beat false, so PORT_RUNNING only asserts on real beat crossings.
    let mut ss = StreamSwitch::new_compute_tile(0, 2);
    ss.slaves[1].push_with_tlast(0x1234, false);

    ss.begin_routing_cycle();

    assert!(ss.slaves[1].cycle_active, "buffered-data port stays active (clock gating)");
    assert!(!ss.slaves[1].cycle_beat, "buffered-data port without a beat this cycle must NOT be running");
}

#[test]
fn test_begin_routing_cycle_seeds_from_fifo() {
    let mut ss = StreamSwitch::new_compute_tile(0, 2);

    // Put data in slave[1] (simulating backpressure holdover)
    ss.slaves[1].push_with_tlast(0x1234, false);

    // begin_routing_cycle seeds cycle_active from existing FIFO state
    ss.begin_routing_cycle();

    assert!(ss.slaves[1].cycle_active, "slave with existing data should be active");
    assert!(!ss.slaves[0].cycle_active, "empty slave should not be active");
    assert!(!ss.masters[0].cycle_active, "empty master should not be active");
}

#[test]
fn test_begin_routing_cycle_clears_previous() {
    let mut ss = StreamSwitch::new_compute_tile(0, 2);

    // Push data to mark a port active
    ss.slaves[0].push_with_tlast(0xAAAA, false);
    assert!(ss.slaves[0].cycle_active);

    // Drain it
    ss.slaves[0].pop();

    // begin_routing_cycle clears the stale active flag
    ss.begin_routing_cycle();
    assert!(!ss.slaves[0].cycle_active, "empty port should not be active after begin_routing_cycle");
}

#[test]
fn test_cycle_tlast_tracks_tlast_on_push() {
    let mut port = StreamPort::new(0, PortDirection::Master, PortType::Dma(0));

    // Push without TLAST: cycle_tlast stays false
    port.push_with_tlast(0x1111, false);
    assert!(!port.cycle_tlast);

    // Push with TLAST: cycle_tlast becomes true
    port.push_with_tlast(0x2222, true);
    assert!(port.cycle_tlast);
}

#[test]
fn test_cycle_tlast_tracks_tlast_on_pop() {
    let mut port = StreamPort::new(0, PortDirection::Slave, PortType::Dma(0));

    // Push a word with TLAST (cycle_tlast set on push)
    port.push_with_tlast(0xAAAA, true);
    assert!(port.cycle_tlast);

    // Reset to test pop path
    port.cycle_tlast = false;
    let (_, tlast) = port.pop_with_tlast().unwrap();
    assert!(tlast);
    assert!(port.cycle_tlast, "pop_with_tlast should set cycle_tlast");
}

#[test]
fn test_begin_routing_cycle_clears_stalled_and_tlast() {
    let mut ss = StreamSwitch::new_compute_tile(0, 2);

    // Manually set flags to verify they get cleared
    ss.slaves[0].cycle_stalled = true;
    ss.slaves[0].cycle_tlast = true;
    ss.masters[0].cycle_stalled = true;
    ss.masters[0].cycle_tlast = true;

    ss.begin_routing_cycle();

    assert!(!ss.slaves[0].cycle_stalled);
    assert!(!ss.slaves[0].cycle_tlast);
    assert!(!ss.masters[0].cycle_stalled);
    assert!(!ss.masters[0].cycle_tlast);
}

// ========================================================================
// Packet routing integration tests (MemTile-focused)
// ========================================================================

#[test]
fn test_packet_routing_basic_memtile() {
    // MemTile DMA slave[0] routes pkt_id=0 to North master[12].
    // Verifies the fundamental packet path: header match -> arbiter -> master.
    let mut ss = StreamSwitch::new_mem_tile(0, 1);

    // Enable packet mode on slave[0] (bit 31=enable, bit 30=packet_enable)
    ss.slaves[0].packet_enable = true;
    // Slave[0] (DMA:0): pkt_id=0, mask=0x1F, arbiter=0, msel=0
    ss.configure_slave_slot(0, 0, make_slot_reg(0, 0x1F, 0, 0));
    // Master[12] (North:1): packet mode, arbiter=0, msel_enable=0b0001
    ss.configure_master_packet(12, make_master_pkt_reg(0, 0b0001, false));

    // Build a 4-word packet: header + 2 data + data+TLAST
    let hdr = PacketHeader::new(0, 0, 1).encode();
    ss.slaves[0].push_with_tlast(hdr, false);
    ss.slaves[0].push_with_tlast(0xDA7A_0001, false);
    ss.slaves[0].push_with_tlast(0xDA7A_0002, false);
    ss.slaves[0].push_with_tlast(0xDA7A_0003, true);

    // Step and drain master each time (FIFO capacity is 2)
    let mut output = Vec::new();
    for _ in 0..8 {
        ss.step();
        while let Some((w, _)) = ss.masters[12].pop_with_tlast() {
            output.push(w);
        }
    }

    assert_eq!(output.len(), 4, "master[12] should have 4 words: {:08X?}", output);
    assert_eq!(output[0], hdr, "first word should be the header");
    assert_eq!(output[1], 0xDA7A_0001);
    assert_eq!(output[2], 0xDA7A_0002);
    assert_eq!(output[3], 0xDA7A_0003);

    // No other master should have data (spot-check DMA and South masters)
    for m in [0, 1, 7, 8, 9, 10, 11, 13] {
        assert!(!ss.masters[m].has_data(), "master[{}] should be empty but has data", m);
    }
}

#[test]
fn test_packet_routing_multi_slave_memtile() {
    // Two MemTile slaves with different pkt_ids route to different masters.
    // slave[0] (DMA:0) pkt_id=0 -> master[9] (South:2)
    // slave[13] (North:0) pkt_id=1 -> master[0] (DMA:0)
    let mut ss = StreamSwitch::new_mem_tile(0, 1);

    // Enable packet mode on both slaves
    ss.slaves[0].packet_enable = true;
    ss.slaves[13].packet_enable = true;
    // Slave[0]: pkt_id=0, arbiter=0, msel=0
    ss.configure_slave_slot(0, 0, make_slot_reg(0, 0x1F, 0, 0));
    // Slave[13]: pkt_id=1, arbiter=1, msel=0
    ss.configure_slave_slot(13, 0, make_slot_reg(1, 0x1F, 0, 1));

    // Master[9] (South:2): arbiter=0, msel_enable=0b0001
    ss.configure_master_packet(9, make_master_pkt_reg(0, 0b0001, false));
    // Master[0] (DMA:0): arbiter=1, msel_enable=0b0001
    ss.configure_master_packet(0, make_master_pkt_reg(1, 0b0001, false));

    // Packet from DMA slave (pkt_id=0): 2 words
    let hdr0 = PacketHeader::new(0, 0, 1).encode();
    ss.slaves[0].push_with_tlast(hdr0, false);
    ss.slaves[0].push_with_tlast(0xAAAA_0001, true);

    // Packet from North slave (pkt_id=1): 2 words
    let hdr1 = PacketHeader::new(1, 0, 2).encode();
    ss.slaves[13].push_with_tlast(hdr1, false);
    ss.slaves[13].push_with_tlast(0xBBBB_0001, true);

    // Step enough times
    for _ in 0..4 {
        ss.step();
    }

    // Master[9] should have pkt_id=0 data
    let (w, _) = ss.masters[9].pop_with_tlast().unwrap();
    assert_eq!(w & 0x1F, 0, "master[9] should have pkt_id=0");

    // Master[0] should have pkt_id=1 data
    let (w, _) = ss.masters[0].pop_with_tlast().unwrap();
    assert_eq!(w & 0x1F, 1, "master[0] should have pkt_id=1");
}

#[test]
fn test_packet_routing_drop_header() {
    // When drop_header=true, the header word is consumed from the slave
    // but NOT forwarded to the master. Only data words appear in output.
    let mut ss = StreamSwitch::new_mem_tile(0, 1);

    // Enable packet mode on slave[0]
    ss.slaves[0].packet_enable = true;
    // Slave[0]: pkt_id=0, arbiter=0, msel=0
    ss.configure_slave_slot(0, 0, make_slot_reg(0, 0x1F, 0, 0));
    // Master[0]: arbiter=0, msel_enable=0b0001, drop_header=TRUE
    ss.configure_master_packet(0, make_master_pkt_reg(0, 0b0001, true));

    // 3-word packet: header + data + data+TLAST
    let hdr = PacketHeader::new(0, 0, 1).encode();
    ss.slaves[0].push_with_tlast(hdr, false);
    ss.slaves[0].push_with_tlast(0xDA7A_0001, false);
    ss.slaves[0].push_with_tlast(0xDA7A_0002, true);

    for _ in 0..4 {
        ss.step();
    }

    // Master should have only the 2 data words (header dropped)
    let mut output = Vec::new();
    while let Some((w, _)) = ss.masters[0].pop_with_tlast() {
        output.push(w);
    }
    assert_eq!(output.len(), 2, "drop_header should remove header, leaving 2 data words: {:08X?}", output);
    assert_eq!(output[0], 0xDA7A_0001);
    assert_eq!(output[1], 0xDA7A_0002);

    // Slave should be empty (header was consumed, not left behind)
    assert!(!ss.slaves[0].has_data(), "slave should be drained");
}

#[test]
fn test_packet_routing_no_route_is_fatal() {
    // A packet with no matching route should produce a fatal error.
    let mut ss = StreamSwitch::new_mem_tile(0, 1);

    // Enable packet mode on slave[0]
    ss.slaves[0].packet_enable = true;
    // Configure slave[0] for pkt_id=0 only
    ss.configure_slave_slot(0, 0, make_slot_reg(0, 0x1F, 0, 0));
    ss.configure_master_packet(9, make_master_pkt_reg(0, 0b0001, false));

    // Push a packet with pkt_id=1 (unmatched)
    let hdr = PacketHeader::new(1, 0, 1).encode();
    ss.slaves[0].push_with_tlast(hdr, false);
    ss.slaves[0].push_with_tlast(0xDA7A_0001, true);

    ss.step();

    // Should produce a fatal error
    assert!(!ss.fatal_errors.is_empty(), "unroutable packet should produce fatal error");
    assert!(
        ss.fatal_errors[0].contains("no packet route"),
        "error should mention 'no packet route': {}",
        ss.fatal_errors[0]
    );
}

/// Build an N-word multicast test packet: header + (N-2) data + TLAST data word.
fn build_multicast_pkt(stream_id: u8, n: usize) -> Vec<(u32, bool)> {
    let hdr = PacketHeader::new(stream_id, 0, 2).with_type(PacketType::Trace);
    let mut words = Vec::with_capacity(n);
    words.push((hdr.encode(), false));
    for i in 0..(n - 2) {
        words.push((0xAAAA_0000 | (i as u32), false));
    }
    words.push((0xAAAA_FFFF, true));
    words
}

#[test]
fn test_multicast_slow_path_does_not_block_fast_path() {
    // One slave multicasts a packet to two masters on the same arbiter+msel.
    // The downstream of one master (M_B = port 8) is never drained during
    // phase 1; M_A (port 7) is drained every step. Under the old all-or-
    // nothing rule, the slave would stall once M_B's FIFO filled and M_A
    // would also stop receiving. Under per-target backpressure, the slave
    // keeps pushing into M_A's pending queue independently.
    //
    // Unit-level analog of the packet_flow_fanout multicast deadlock in
    // findings/2026-05-11-emu-trace-widened-distributed-routing.md.
    let mut ss = StreamSwitch::new_compute_tile(0, 2);

    ss.slaves[23].packet_enable = true;
    ss.configure_slave_slot(23, 0, make_slot_reg(1, 0x1F, 0, 0));
    ss.configure_master_packet(7, make_master_pkt_reg(0, 0b0001, false));
    ss.configure_master_packet(8, make_master_pkt_reg(0, 0b0001, false));

    // 12-word packet -- longer than the slave FIFO (depth 4), so we refill
    // from a virtual upstream each cycle.
    let mut feed = build_multicast_pkt(1, 12).into_iter().peekable();

    let mut a_drained: Vec<u32> = Vec::new();
    for _ in 0..60 {
        while let Some(&(d, t)) = feed.peek() {
            if !ss.slaves[23].push_with_tlast(d, t) {
                break;
            }
            feed.next();
        }
        ss.step();
        while let Some((w, _)) = ss.masters[7].pop_with_tlast() {
            a_drained.push(w);
        }
    }

    assert_eq!(
        a_drained.len(),
        12,
        "fast multicast target should receive full packet despite blocked sibling: got {}: {:08X?}",
        a_drained.len(),
        a_drained,
    );
    assert!(ss.arbiter_locks[0].is_some(), "arbiter should still be held while M_B's pending is undrained");

    // Phase 2: drain M_B and watch the packet complete.
    let mut b_drained: Vec<u32> = Vec::new();
    for _ in 0..60 {
        ss.step();
        while let Some((w, _)) = ss.masters[8].pop_with_tlast() {
            b_drained.push(w);
        }
    }

    assert_eq!(b_drained.len(), 12, "slow target should eventually receive full packet");
    assert!(ss.arbiter_locks[0].is_none(), "arbiter should release once both targets have drained TLAST");
}

#[test]
fn test_multicast_reconverging_arbiter_no_deadlock() {
    // Same multicast topology, but drain M_A every cycle and M_B every
    // fourth cycle -- simulating M_B losing arbitration on a downstream
    // tile (the packet_flow_fanout pattern). Both masters must eventually
    // receive every word.
    let mut ss = StreamSwitch::new_compute_tile(0, 2);

    ss.slaves[23].packet_enable = true;
    ss.configure_slave_slot(23, 0, make_slot_reg(1, 0x1F, 0, 0));
    ss.configure_master_packet(7, make_master_pkt_reg(0, 0b0001, false));
    ss.configure_master_packet(8, make_master_pkt_reg(0, 0b0001, false));

    let mut feed = build_multicast_pkt(1, 10).into_iter().peekable();
    let mut a_out: Vec<u32> = Vec::new();
    let mut b_out: Vec<u32> = Vec::new();

    for cyc in 0..120 {
        while let Some(&(d, t)) = feed.peek() {
            if !ss.slaves[23].push_with_tlast(d, t) {
                break;
            }
            feed.next();
        }
        ss.step();
        while let Some((w, _)) = ss.masters[7].pop_with_tlast() {
            a_out.push(w);
        }
        if cyc % 4 == 3 {
            while let Some((w, _)) = ss.masters[8].pop_with_tlast() {
                b_out.push(w);
            }
        }
    }
    for _ in 0..40 {
        ss.step();
        while let Some((w, _)) = ss.masters[8].pop_with_tlast() {
            b_out.push(w);
        }
    }

    assert_eq!(a_out.len(), 10, "M_A received {} words: {:08X?}", a_out.len(), a_out);
    assert_eq!(b_out.len(), 10, "M_B received {} words: {:08X?}", b_out.len(), b_out);
    assert!(ss.arbiter_locks[0].is_none(), "arbiter released after packet completes");
}
