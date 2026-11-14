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
//! Shim switches neither hold (the measured exemptions) nor pass a pulse on; a packet-switched
//! hop (no circuit route to follow) ends the pulse. Both unmeasured for DMA streams.

use super::TileArray;
use crate::device::stream_switch::PortType;
use crate::device::tile::TileKind;

/// Cycles a hold takes to pass back over one hop. With the forward front (4 cycles a hop and one
/// more past a quiet switch) it brings the hold of the switch k hops out to the port by its
/// first word + 8k, before or at the slot it takes.
const BACK_CYCLES_PER_HOP: u64 = 3;
/// Slots of the stream between the source port and a switch one hop further out.
const SLOTS_PER_HOP: u64 = 8;
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

/// A hold that reached its source port: the port stalls at the stream's slot `slot`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PortHold {
    pub tile: usize,
    pub slave: usize,
    pub slot: u64,
    pub arrived: u64,
}

/// An MM2S port's current stream: its first word's cycle, its stalled cycles since, its last word.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PortStream {
    pub first: u64,
    pub stalls: u64,
    pub last: u64,
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
            let Some(slave) = switch
                .local_routes
                .iter()
                .find(|r| r.enabled && r.master_idx as usize == p.master)
                .map(|r| r.slave_idx as usize)
            else {
                continue; // packet-switched or unrouted: the pulse ends here
            };
            match switch.slaves[slave].port_type {
                PortType::Dma(_) => self.port_holds.push(PortHold {
                    tile: p.tile,
                    slave,
                    slot: SLOTS_PER_HOP * p.hops,
                    arrived: now,
                }),
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
            .position(|h| h.tile == idx && h.slave == slave && h.slot <= slot)
        else {
            return false;
        };
        self.port_holds.swap_remove(k);
        offered
    }

    /// Account the port's cycle: a word sent (`sent`), or one offered and not sent.
    pub(super) fn note_port_cycle(&mut self, idx: usize, slave: usize, sent: bool, offered: bool) {
        let now = self.hold_clock;
        let e =
            self.port_streams
                .entry((idx, slave))
                .or_insert(PortStream { first: now, stalls: 0, last: 0 });
        if sent {
            if e.last == 0 || now.saturating_sub(e.last) > STREAM_GAP {
                *e = PortStream { first: now, stalls: 0, last: now };
            }
            e.last = now;
        } else if offered && now.saturating_sub(e.last) <= STREAM_GAP && e.last != 0 {
            e.stalls += 1;
            e.last = now;
        }
    }

    /// Holds that never found their slot (the stream ended first) are dropped.
    pub(super) fn expire_port_holds(&mut self) {
        let now = self.hold_clock;
        self.port_holds.retain(|h| now.saturating_sub(h.arrived) <= 4 * STREAM_GAP);
    }

    fn crossing_feeder(&self, idx: usize, slave: usize) -> Option<(usize, usize)> {
        self.crossing_feeders.get(idx)?.get(slave).copied().flatten()
    }
}
