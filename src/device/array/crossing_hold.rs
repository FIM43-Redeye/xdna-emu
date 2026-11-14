//! A quiet switch's hold of a stream's first word, and its way back to the stream's source.
//!
//! Silicon (NPU1, 2026-11-13: crossing-wake probes v1 / v2 / v3, the slack twins, depth probes
//! 9 / 10 / 11, sw, mr): every compute or memtile stream switch whose adaptive gate is engaged
//! (no word crossed any of its ports, in either direction, for its abort period) holds the first
//! word that would cross one of its ports for one cycle, and wakes. Where the word is:
//!
//! - offered by the tile's own MM2S at its DMA slave: the port cannot send it that cycle (the
//!   source switch's hold, k = 0; `route_dma_to_tile_switches`);
//! - arriving from a neighbour: it waits one cycle inside the switch (the adaptive gate's
//!   first-word charge, `step_tile_switches_impl`), and the hold travels back toward the
//!   stream's source as a [`HoldPulse`];
//! - parked in the switch, leaving toward its own S2MM or a neighbour: it leaves one cycle
//!   later (`route_tile_switches_to_dma`, `propagate_inter_tile`).
//!
//! The pulse passes back one switch at a time, launched the cycle after the hold and 3 cycles a
//! hop; it reaches the source port no later than the port's first word + 8k for the switch k
//! hops out. There it takes the stream's slot 8k: one port cycle, counted from the first word,
//! in which the port moves nothing, and which every cycle the port has already stalled since the
//! first word pushes one later. A slot in which the port has nothing to offer (the stream's idle
//! cycle at a BD boundary) absorbs it. Silicon: + 8 / + 16 / + 24 for k = 1 / 2 / 3 with the
//! switches between quiet (slack) and with the first crossing kept awake (crossing arm b, v3's
//! split heads); the third hold of slack's 24-word head behind two stalls lands on the head's idle
//! cycle and costs nothing (h24n3w3q), as slots, not cycles, give. The pulse moves no word in the
//! switches it passes (unobservable in the measured traces; the port's stall is what the MM2S's
//! finishes show). Sustained backpressure is untouched: it is the slave and master FIFOs'
//! occupancy, 8 words a hop.
//!
//! The packet stage (silicon, the same streams, both compilers, three samples identical): every switch
//! that routes a stream through packet rules, a shim switch included, holds each packet one cycle at its
//! packet stage, whatever its quiet state, and the hold reaches the source port at the packet's slot
//! 8k + 4 (counted from the packet's header) for the switch k hops out: half a hop past the switch's
//! slave-side hold, as a hold at pipeline stage L stalls the port at slot 2L and a switch is four
//! stages. It needs no quiet state, so the port takes it straight from the packet's route when the
//! header leaves (`note_packet_word`, `packet_stage_hops`): one hold for every packet-routed switch the
//! route passes, slots as for the slave-side hold (stalls since push it later, a slot with nothing to
//! send absorbs it). A switch that carries the stream on a circuit route has none. In the switch, the
//! packet rules take a header one cycle after it won its arbiter (`StreamSwitch::packet_stage_held`), so
//! the stream leaves each packet-routed switch one cycle later and, the port's stalls being the matching
//! back-pressure, contiguous (silicon: the memtile S2MM ports take packet streams without a gap while
//! their sources stall). The switch's hold acts on every packet; the port's stall only at the MM2S ports
//! of every tile kind: the trace units, the core's stream output and the control-packet responses push
//! into their slave ports without a one-word-per-cycle port and take none.
//!
//! A packet-routed hop holds and passes the pulse on like a circuit one: the pulse follows the
//! packet rules back to the slave whose packet they last routed to the master (silicon: memtile
//! packet_flow / fanin / fanout and the compute packet-gap probe, a quiet switch one or two hops
//! out stalls the packet stream at slot 8k). Shim switches neither hold (the measured exemptions)
//! nor pass a pulse on.

use super::TileArray;
use crate::device::stream_switch::{inter_tile_dest, PacketHeader, PortType};
use crate::device::tile::TileKind;

/// Cycles a hold takes to pass back over one hop. With the forward front (4 cycles a hop and one
/// more past a quiet switch) it brings the hold of the switch k hops out to the port by its
/// first word + 8k, before or at the slot it takes.
const BACK_CYCLES_PER_HOP: u64 = 3;
/// Slots of the stream between the source port and a switch one hop further out.
const SLOTS_PER_HOP: u64 = 8;
/// Slots between a switch's slave side and its packet stage (two of its four pipeline stages).
const PACKET_STAGE_SLOT: u64 = 4;
/// A port idle this long ends its stream: the next word is a new stream's first.
const STREAM_GAP: u64 = 64;

/// A hold on its way back: it reaches `tile`'s switch at `due`, at the master that carried the
/// stream's first word on toward the switch that held it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HoldPulse {
    pub tile: usize,
    pub master: usize,
    pub due: u64,
    /// Hops from the switch that held to `tile`'s switch.
    pub hops: u64,
}

/// A hold that reached its source port: the port stalls at the slot `slot` of its stream `stream` (the
/// stream's first word's cycle).
#[derive(Debug, Clone, Copy)]
pub(crate) struct PortHold {
    pub tile: usize,
    pub slave: usize,
    pub slot: u64,
    pub stream: u64,
}

/// An MM2S port's current stream: its first word's cycle, its stalled cycles since, its last word, and
/// whether the port's next word opens a packet (its first, or the one after a TLAST).
#[derive(Debug, Clone, Copy)]
pub(crate) struct PortStream {
    pub first: u64,
    pub stalls: u64,
    pub last: u64,
    pub head_next: bool,
}

impl TileArray {
    /// Whether a switch takes part in the hold: compute and memtile switches only.
    pub(super) fn holds_first_words(&self, idx: usize) -> bool {
        matches!(self.tiles[idx].tile_kind, TileKind::Compute | TileKind::Mem)
    }

    /// Whether `idx`'s quiet switch would hold a word crossing one of its ports this cycle.
    pub(super) fn switch_is_quiet(&self, idx: usize) -> bool {
        let t = &self.tiles[idx];
        self.holds_first_words(idx) && self.clock.is_adaptive_ss_engaged(t.col, t.row)
    }

    /// Record a hold by `idx`'s switch: it wakes (its idle detector sees activity this cycle).
    pub(super) fn note_hold(&mut self, idx: usize) {
        self.ss_woken_by_hold[idx] = true;
    }

    /// The word that crossed from `src`'s master `master` is held by the quiet switch it entered:
    /// launch the hold back toward its source.
    pub(super) fn launch_hold_pulse(&mut self, src: usize, master: usize) {
        if !self.holds_first_words(src) {
            return;
        }
        let due = self.hold_clock + 1 + BACK_CYCLES_PER_HOP;
        self.hold_pulses.push(HoldPulse { tile: src, master, due, hops: 1 });
    }

    /// Advance the pulses due this cycle: a pulse at a switch whose master is fed by the tile's own
    /// MM2S becomes that port's hold; one fed from a neighbour passes back to it.
    pub(super) fn step_hold_pulses(&mut self) {
        if self.hold_pulses.is_empty() {
            return;
        }
        let now = self.hold_clock;
        let mut keep = Vec::with_capacity(self.hold_pulses.len());
        for p in std::mem::take(&mut self.hold_pulses) {
            if p.due > now {
                keep.push(p);
                continue;
            }
            let switch = &self.tiles[p.tile].stream_switch;
            // The slave that fed the master: by its circuit route, or, on a packet-routed hop, the slave whose
            // packet the packet rules last routed to it.
            let Some(slave) = switch
                .local_routes
                .iter()
                .find(|r| r.enabled && r.master_idx as usize == p.master)
                .map(|r| r.slave_idx as usize)
                .or_else(|| switch.packet_feeders.get(p.master).copied().flatten().map(usize::from))
            else {
                continue; // unrouted: the pulse ends here
            };
            match switch.slaves[slave].port_type {
                PortType::Dma(_) => {
                    if let Some(stream) = self.port_streams.get(&(p.tile, slave)).map(|s| s.first) {
                        self.port_holds.push(PortHold {
                            tile: p.tile,
                            slave,
                            slot: SLOTS_PER_HOP * p.hops,
                            stream,
                        });
                    }
                }
                _ => {
                    if let Some((up, up_master)) = self.crossing_feeder(p.tile, slave) {
                        if self.holds_first_words(up) {
                            keep.push(HoldPulse {
                                tile: up,
                                master: up_master,
                                due: now + BACK_CYCLES_PER_HOP,
                                hops: p.hops + 1,
                            });
                        }
                    }
                }
            }
        }
        self.hold_pulses = keep;
    }

    /// Whether the port at (`idx`, DMA slave `slave`) stalls this cycle for a hold whose slot has
    /// come: consumes the hold, whether the port has a word to offer (`offered`) or not (absorbed).
    pub(super) fn port_hold_due(&mut self, idx: usize, slave: usize, offered: bool) -> bool {
        let now = self.hold_clock;
        let Some(stream) = self.port_streams.get(&(idx, slave)).copied() else {
            return false;
        };
        if now.saturating_sub(stream.last) > STREAM_GAP {
            return false;
        }
        let slot = now - stream.first - stream.stalls;
        let Some(k) = self
            .port_holds
            .iter()
            .position(|h| h.tile == idx && h.slave == slave && h.stream == stream.first && h.slot <= slot)
        else {
            return false;
        };
        self.port_holds.swap_remove(k);
        offered
    }

    /// Account the port's cycle: a word sent (`sent`), or one offered and not sent.
    pub(super) fn note_port_cycle(&mut self, idx: usize, slave: usize, sent: bool, offered: bool) {
        let now = self.hold_clock;
        let e = self.port_streams.entry((idx, slave)).or_insert(PortStream {
            first: now,
            stalls: 0,
            last: 0,
            head_next: true,
        });
        if sent {
            if e.last == 0 || now.saturating_sub(e.last) > STREAM_GAP {
                *e = PortStream { first: now, stalls: 0, last: now, head_next: e.head_next };
            }
            e.last = now;
        } else if offered && now.saturating_sub(e.last) <= STREAM_GAP && e.last != 0 {
            e.stalls += 1;
            e.last = now;
        }
    }

    /// The word `data` (with `tlast`) left the port at (`idx`, slave `slave`) this cycle, after
    /// `note_port_cycle` accounted it. A word that opens a packet takes the packet stage's hold of every
    /// packet-routed switch on its route, at the packet's slot 8k + 4 for the switch k hops out.
    pub(super) fn note_packet_word(&mut self, idx: usize, slave: usize, data: u32, tlast: bool) {
        let now = self.hold_clock;
        let Some(e) = self.port_streams.get_mut(&(idx, slave)) else {
            return;
        };
        let opens = std::mem::replace(&mut e.head_next, tlast);
        if !opens {
            return;
        }
        let (stream, head_slot) = (e.first, now - e.first - e.stalls);
        for k in self.packet_stage_hops(idx, slave, data) {
            self.port_holds.push(PortHold {
                tile: idx,
                slave,
                slot: head_slot + SLOTS_PER_HOP * k + PACKET_STAGE_SLOT,
                stream,
            });
        }
    }

    /// Hops from the source switch (0 = its own) of every switch that routes a packet with header `header`,
    /// entering at (`idx`, slave `slave`), through packet rules: the packet's route as configured, over
    /// circuit and packet hops alike, every branch of a multicast.
    pub(super) fn packet_stage_hops(&self, idx: usize, slave: usize, header: u32) -> Vec<u64> {
        let id = PacketHeader::decode(header).0.stream_id;
        let kind_at = |c: u8, r: u8| {
            self.arch
                .is_valid_tile(c, r)
                .then(|| self.tiles[self.tile_index(c, r)].tile_kind)
        };
        let mut hops = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut todo = vec![(idx, slave, 0u64)];
        while let Some((t, s, k)) = todo.pop() {
            if !seen.insert((t, s)) || !self.tile_present[t] {
                continue;
            }
            let tile = &self.tiles[t];
            let switch = &tile.stream_switch;
            let masters: Vec<usize> = match switch.packet_rule_targets(s, id) {
                Some(targets) => {
                    hops.push(k);
                    targets.into_iter().map(usize::from).collect()
                }
                None => switch
                    .local_routes
                    .iter()
                    .filter(|r| r.enabled && r.slave_idx as usize == s)
                    .map(|r| r.master_idx as usize)
                    .collect(),
            };
            for m in masters {
                if !switch.masters.get(m).is_some_and(|p| p.port_type.is_external()) {
                    continue;
                }
                if let Some(d) = inter_tile_dest(tile.tile_kind, tile.col, tile.row, m as u8, kind_at) {
                    todo.push((self.tile_index(d.col, d.row), d.port as usize, k + 1));
                }
            }
        }
        hops.sort_unstable();
        hops.dedup();
        hops
    }

    /// Holds whose stream has ended (an idle gap, or a new stream at the port) before their slot came are
    /// dropped.
    pub(super) fn expire_port_holds(&mut self) {
        let now = self.hold_clock;
        let streams = &self.port_streams;
        self.port_holds.retain(|h| {
            streams
                .get(&(h.tile, h.slave))
                .is_some_and(|s| s.first == h.stream && now.saturating_sub(s.last) <= STREAM_GAP)
        });
    }

    fn crossing_feeder(&self, idx: usize, slave: usize) -> Option<(usize, usize)> {
        self.crossing_feeders.get(idx)?.get(slave).copied().flatten()
    }
}
