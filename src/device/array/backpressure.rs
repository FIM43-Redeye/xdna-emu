//! Combinational cold-terminal backpressure reachability (#140 SP-4a campaign).
//!
//! HW propagates a cold terminal's TREADY-low **combinationally end-to-end**:
//! when a runtime-dispatched shim S2MM is still un-armed, every upstream stage of
//! the coupled objectfifo chain backs up in ~the same cycle, so the producing
//! cores stall producer-first within ~2cy. The EMU's per-hop FIFO-fill routing
//! instead propagates that backpressure one hop per cycle, so the drain-nearest
//! core stalls 60-100cy before the source (consumer-first). This module
//! precomputes, from the static route graph, the set of upstream DMA channels
//! that transitively feed each terminal S2MM, so the routing phases can hold them
//! all in the SAME cycle the terminal is found cold -- reproducing the
//! combinational propagation without a per-cycle graph walk.
//!
//! The forward data graph has two edge classes:
//!   * **fabric**: MM2S source --(circuit route across tiles)--> terminal S2MM
//!     (from `build_mm2s_terminal_map`, the 1:1 circuit walk).
//!   * **relay**: S2MM writer --(shared memtile buffer / lock handoff /
//!     through-core)--> MM2S reader, intra-tile (from the route-graph E2/E3/P2
//!     detectors: `dma_buffer_relay_edges`, `dma_lock_pair_edges`,
//!     `core_lock_relay_edges`).
//! Reversing every edge yields the backpressure graph; a backward BFS from a
//! terminal S2MM enumerates every source that must be held when it is cold.
//!
//! Engaged ONLY when a terminal is un-armed (`!channel_is_started`), the same
//! guard the fix-(a) source gate uses, so steady state and every kernel without a
//! pre-dispatch terminal are a provable no-op.
//!
//! # Gate modes (`XDNA_EMU_SP4A_BP_MODE`, read once at build)
//! - `0` (default): campaign gates OFF -- only the pre-existing fix-(a) direct
//!   terminal-source drain gate runs (baseline behavior; byte-identical).
//! - `1`: + accept-gate every reachable interior S2MM (the principled
//!   combinational hold -- the fabric backs up to the source in one cycle
//!   instead of filling stage-by-stage).
//! - `2`: mode 1 + also drain-gate every reachable MM2S source.

use std::collections::{HashMap, HashSet, VecDeque};

use super::TileArray;
use crate::device::dma::DmaEngine;
use crate::device::tile::Tile;

/// Per-root backward-reachable upstream channels, split by gate site.
#[derive(Debug, Clone, Default)]
pub(crate) struct Reachable {
    /// MM2S sources to gate at the drain (`route_dma_to_tile_switches`).
    /// `(col, row, mm2s_ch)`.
    pub(crate) mm2s: Vec<(u8, u8, u8)>,
    /// Interior S2MM stages to gate at the accept (`route_tile_switches_to_dma`).
    /// `(col, row, s2mm_ch)`. Excludes the root itself (self-handled by the
    /// existing `!channel_is_started` accept gate).
    pub(crate) s2mm: Vec<(u8, u8, u8)>,
}

/// Static combinational-backpressure reachability for the configured array.
#[derive(Debug, Clone, Default)]
pub(crate) struct BackpressureReach {
    /// Candidate cold roots -> their backward-reachable upstream channels. A root
    /// is a terminal S2MM `(col, row, s2mm_ch)`; per cycle it fires when un-armed.
    pub(crate) per_root: HashMap<(u8, u8, u8), Reachable>,
    /// Gate mode from `XDNA_EMU_SP4A_BP_MODE`, read once at build time (never in
    /// the per-cycle hot loop). `0` = campaign gates off.
    pub(crate) mode: u8,
}

/// A graph node: `(col, row, is_s2mm, ch)`. `is_s2mm` disambiguates the S2MM and
/// MM2S channel-number spaces (both start at 0).
type Node = (u8, u8, bool, u8);

impl TileArray {
    /// Build the static combinational-backpressure reachability. Reuses the
    /// already-resolved `mm2s_terminal_s2mm` fabric map (caller ensures it is
    /// populated first) and the route-graph intra-tile relay detectors.
    pub(crate) fn build_backpressure_reach(&self) -> BackpressureReach {
        let mode = std::env::var("XDNA_EMU_SP4A_BP_MODE")
            .ok()
            .and_then(|s| s.trim().parse::<u8>().ok())
            .unwrap_or(0);

        // Forward adjacency: node -> successors. Two edge classes:
        //   fabric: (MM2S) -> (terminal S2MM)   relay: (S2MM) -> (MM2S)
        let mut fwd: HashMap<Node, Vec<Node>> = HashMap::new();
        let mut terminals: HashSet<Node> = HashSet::new();

        // Fabric edges from the resolved 1:1 circuit terminal map.
        if let Some(term_map) = self.mm2s_terminal_s2mm.as_ref() {
            for (&(sc, sr, sch), &(tc, tr, tch)) in term_map.iter() {
                let src: Node = (sc, sr, false, sch);
                let dst: Node = (tc, tr, true, tch);
                fwd.entry(src).or_default().push(dst);
                terminals.insert(dst);
            }
        }

        // Relay edges (intra-tile S2MM writer -> MM2S reader).
        for i in 0..self.tiles.len() {
            let tile = &self.tiles[i];
            let dma = &self.dma_engines[i];
            for (s2mm_ch, mm2s_ch) in self.tile_relay_pairs(tile, dma) {
                let src: Node = (tile.col, tile.row, true, s2mm_ch);
                let dst: Node = (tile.col, tile.row, false, mm2s_ch);
                fwd.entry(src).or_default().push(dst);
            }
        }

        // Reverse adjacency = backpressure graph.
        let mut rev: HashMap<Node, Vec<Node>> = HashMap::new();
        for (&u, vs) in fwd.iter() {
            for &v in vs {
                rev.entry(v).or_default().push(u);
            }
        }

        // Backward BFS from each terminal S2MM root.
        let mut per_root: HashMap<(u8, u8, u8), Reachable> = HashMap::new();
        for &root in terminals.iter() {
            let mut reach = Reachable::default();
            let mut seen: HashSet<Node> = HashSet::new();
            seen.insert(root);
            let mut q: VecDeque<Node> = VecDeque::new();
            q.push_back(root);
            while let Some(n) = q.pop_front() {
                let Some(preds) = rev.get(&n) else { continue };
                for &p in preds {
                    if seen.insert(p) {
                        if p.2 {
                            reach.s2mm.push((p.0, p.1, p.3));
                        } else {
                            reach.mm2s.push((p.0, p.1, p.3));
                        }
                        q.push_back(p);
                    }
                }
            }
            let (rc, rr, _, rch) = root;
            per_root.insert((rc, rr, rch), reach);
        }

        BackpressureReach { per_root, mode }
    }

    /// Intra-tile relay pairs `(s2mm_ch, mm2s_ch)` for one tile, from the
    /// route-graph E2/E3/P2 detectors, translated port-space -> channel-space.
    fn tile_relay_pairs(&self, tile: &Tile, dma: &DmaEngine) -> Vec<(u8, u8)> {
        use crate::device::stream_switch::{dma_buffer_relay_edges, dma_lock_pair_edges};

        // `s2mm_count` per the same authoritative source `resolve_route_graph`
        // uses (`shim_mux_s2mm_masters` is sized to `params.dma_s2mm_channels`).
        let s2mm_count = tile.shim_mux_s2mm_masters.len();

        // Inverse port maps: master port index -> S2MM ch, slave port -> MM2S ch.
        let mut master_to_s2mm: HashMap<u8, u8> = HashMap::new();
        for ch in 0..dma.s2mm_channel_count() as u8 {
            if let Some(p) = tile.stream_switch.dma_master(ch) {
                master_to_s2mm.insert(p.index, ch);
            }
        }
        let mut slave_to_mm2s: HashMap<u8, u8> = HashMap::new();
        for ch in 0..dma.mm2s_channel_count() as u8 {
            if let Some(p) = tile.stream_switch.dma_slave(ch) {
                slave_to_mm2s.insert(p.index, ch);
            }
        }

        let mut edges = dma_buffer_relay_edges(tile, dma, s2mm_count);
        edges.extend(dma_lock_pair_edges(tile, dma, s2mm_count));
        // Through-core relay (compute tiles with a loaded program only). Guard on
        // a non-zero program to avoid decoding ~16KB of zeros on CDO-only tiles,
        // mirroring `resolve_route_graph`'s guard.
        if tile.is_compute() {
            if let Some(prog) = tile.program_memory() {
                if prog.iter().any(|&b| b != 0) {
                    let dec = crate::interpreter::InstructionDecoder::load_cached();
                    let usage =
                        crate::device::stream_switch::core_relay::analyze_core_program(&prog[..], 0, &dec);
                    edges.extend(crate::device::stream_switch::core_relay::core_lock_relay_edges(
                        tile, dma, s2mm_count, &usage,
                    ));
                }
            }
        }

        let mut pairs: Vec<(u8, u8)> = Vec::new();
        let mut seen: HashSet<(u8, u8)> = HashSet::new();
        for e in edges {
            if let (Some(&s), Some(&m)) = (master_to_s2mm.get(&e.src.port), slave_to_mm2s.get(&e.dst.port)) {
                if seen.insert((s, m)) {
                    pairs.push((s, m));
                }
            }
        }
        pairs
    }

    /// Per-cycle combinational-backpressure gate sets, honoring the build-time
    /// gate mode. Returns `(drain_gate, accept_gate)` each as
    /// `{(tile_index, channel)}`:
    /// - `drain_gate`: MM2S channels to skip in `route_dma_to_tile_switches`
    ///   (mode >= 2).
    /// - `accept_gate`: S2MM channels to skip in `route_tile_switches_to_dma`
    ///   (mode >= 1).
    ///
    /// Only roots that are currently un-armed (`!channel_is_started`) contribute,
    /// so the gates empty the moment the runtime dispatches the terminal drain.
    pub(crate) fn sp4a_campaign_gates(&self) -> (HashSet<(usize, u8)>, HashSet<(usize, u8)>) {
        let mut drain_gate: HashSet<(usize, u8)> = HashSet::new();
        let mut accept_gate: HashSet<(usize, u8)> = HashSet::new();
        let Some(reach) = self.backpressure_reach.as_ref() else {
            return (drain_gate, accept_gate);
        };
        if reach.mode == 0 {
            return (drain_gate, accept_gate);
        }
        for (&(rc, rr, rch), r) in reach.per_root.iter() {
            let tidx = self.tile_index(rc, rr);
            if self.dma_engines[tidx].channel_is_started(rch) {
                continue; // terminal armed -> no backpressure from this root
            }
            // mode >= 1: accept-gate reachable interior S2MMs.
            for &(c, ro, ch) in &r.s2mm {
                accept_gate.insert((self.tile_index(c, ro), ch));
            }
            // mode >= 2: also drain-gate reachable MM2S sources.
            if reach.mode >= 2 {
                for &(c, ro, ch) in &r.mm2s {
                    drain_gate.insert((self.tile_index(c, ro), ch));
                }
            }
        }
        (drain_gate, accept_gate)
    }
}
