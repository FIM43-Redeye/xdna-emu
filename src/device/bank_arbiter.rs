//! Per-physical-bank memory arbiter of a compute tile: a served-access count
//! decides core versus DMA.
//!
//! Per AM020 ch.2: each memory bank has its own arbiter. A bank accepts one
//! request per cycle; when several requesters hit the same bank in one cycle,
//! one wins and the rest stall for a cycle and retry.
//!
//! Arbitration is over PHYSICAL banks (single-port SRAMs), not the four
//! programmer-visible logical banks -- the hardware exposes eight
//! CONFLICT_DM_BANK events, and an HW capture confirmed a single logical bank
//! splitting its conflicts across two independently-counted arbiters.
//!
//! # Core versus DMA: the served-access count (winner selection)
//!
//! Each physical bank counts the accesses it has SERVED since reset
//! (`BankAccessCounts`): core loads and stores, the accesses of all four DMA
//! channels, and configuration writes (once per initialised 16-byte line). A
//! refused request adds nothing. In a cycle where the core and at least one
//! DMA channel want the same bank, the DMA is served when that count is at
//! `BANK_ARBITER_DMA_RESIDUE` modulo `BANK_ARBITER_PERIOD`, and the core is
//! served otherwise. The loser retries the next cycle, so a core that wants a
//! bank every cycle lets a waiting DMA in once in every five served cycles,
//! never after more than four refusals. Measured on Phoenix (NPU1) with the
//! lock-fetch arbitration and ring-order probes (`tools/gen_dma_lock_probe.py`
//! arb / ring cases, 222 silicon rows, every one as the rule predicts): MM2S 0
//! and 1 and S2MM 0 and 1 all win on the same single residue of the same
//! count, and serving one channel advances the count the others see. Whether
//! the DMA is about to run dry plays no part (the producer probe's 56 DMA wins
//! came at the same staging as its 1557 losses).
//!
//! Not measured, and modelled by keeping the arbiter's existing rotor: which
//! DMA channel is served when two or more want the bank in the same cycle
//! (with or without the core), and which core port is served when two of the
//! core's own ports want one bank. Among the requesters of the winning class
//! (DMA or core) the rotor picks, by stable requester identity
//! (`Requester::ordinal`), and it advances by one every contended cycle.
//! Neighbouring cores accessing this tile's memory, vector-width accesses
//! spanning two banks and memory-tile banks (never arbitrated here) are
//! unmeasured too.
//!
//! # The retry contract (load-bearing invariant)
//!
//! AM020's two sentences are ONE design: round-robin priority is the
//! anti-starvation mechanism only because "the other requesters are stalled
//! for one cycle and the hardware retries the memory request in the next
//! cycle." This arbiter's starvation-freedom therefore DEPENDS ON its caller
//! honouring that contract:
//!
//! * **Every loser re-presents its UNSERVED demand on the very next cycle.** A
//!   requester that loses is stalled -- it does not withdraw, skip ahead, or
//!   come back later. It holds its request until granted.
//! * **A winning access COMPLETED, per BANK.** It latched and must NOT
//!   re-request. Because each bank arbitrates independently, this is
//!   necessarily per-bank: a requester whose access spans two banks (any
//!   32-byte vector access -- `banks_for_access`) can win one and lose the
//!   other, and the half that won is done. `Arbitration::bank_grants` reports
//!   exactly which banks each requester won, and the retry re-presents only the
//!   rest.
//!
//! Under that contract a pending requester re-asks at every bank it still needs
//! while that bank's rotor sweeps, so within one class each bank grants it
//! within at most `NUM_REQUESTERS` contended cycles. Across classes the count
//! is the guarantee: a DMA waiting against an ever-present core is served
//! within `BANK_ARBITER_PERIOD` cycles (the count advances once per served
//! cycle), and the core is served in every contended cycle but one in five.
//! Several DMA channels share the one DMA turn, so a DMA channel kept waiting
//! by another DMA channel waits for the next turn (silicon: an MM2S starved for
//! 21 cycles behind its sibling while the core camped).
//!
//! # What is proven, and what is not
//!
//! `no_requester_starves_under_the_retry_contract` sweeps demand masks the
//! model really generates (1-, 2- and 3-bank), every demand period up to
//! 2*`NUM_REQUESTERS` and phase, 5 requester mixes, two rival cadences, and
//! every rotor skew -- with per-BANK sticky retry. No starvation; worst wait 6
//! cycles against the bound of 7. All 5 mixes stay WITHIN one class
//! (core-vs-core or DMA-vs-DMA); `a_camping_core_lets_a_waiting_dma_in_within_one_period`
//! pins the cross-class bound.
//!
//! That bound is proven ONLY for the per-bank retry discipline. An
//! ALL-OR-NOTHING retry -- re-present the whole multi-bank mask, discarding the
//! bank you won -- genuinely starves: two rotors can hold a relative phase in
//! which a two-bank requester never wins both banks in the same cycle while
//! each single-bank rival keeps winning its own.
//! `multi_bank_requester_starves_without_per_bank_stickiness` pins that
//! counter-example. The core honours the per-bank contract
//! (`CoreInterpreter::bank_served_banks`).
//!
//! A denied DMA channel does NOT: it re-presents its whole mask, because
//! `DmaEngine::step_with_denied` skips its FSM step entirely and it therefore
//! has no state saying which bank it won. That is safe only because a DMA
//! DEMAND -- what `peek_bank_demand` presents to this arbiter -- is SINGLE-BANK
//! by construction: the DMA's memory side is one 128-bit port and takes at most
//! one 16-byte granule per cycle, in either direction
//! (`DmaEngine::granule_capped_words`, measured on a compute tile at 16.0 B /
//! 4.00 stream beats: `xdna-emu-priv: research/findings/2026-07-14-dma-bank-access-width.md`).
//! Single-bank is a property of the DEMAND, not of every bank the commit path
//! can touch: the decompression S2MM peek under-claims (a known, recorded gap --
//! `docs/fidelity-gaps/dma-stream-resources.md`). An under-claim can only miss a
//! conflict, never fabricate a denial, so it does not threaten this bound.
//! A one-bank mask cannot be partially served, so whole-mask and per-bank retry
//! are the same thing for it. `dma_whole_mask_retry_starves_a_multi_bank_channel`
//! pins what that construction buys: give a DMA channel a two-bank demand
//! against two OTHER DMA channels and the whole-mask retry starves it outright
//! at several rotor skews -- so if a multi-bank DMA demand is ever
//! reintroduced, the single-bank invariant it depends on must be reintroduced
//! with it, or the retry must go per-bank.

use smallvec::SmallVec;

use super::banking::{BankLayout, COMPUTE_PHYSICAL_BANKS};

/// Which of the compute core's three independent memory ports issued a
/// request. AM020 ch.4:69: the core has two load ports and one store port,
/// each a genuinely independent requester at the bank arbiters -- a physical
/// bank is single-port, so the core's OWN load and store contend with each
/// other exactly like any other pair of requesters when they target the same
/// bank (see `Arbitration::core_lost` for the bundle-granularity stall this
/// still produces).
///
/// These variants are kept in lockstep with `interpreter::bundle::SlotIndex`,
/// which is the toolchain-derived source for this port count (the 128-bit
/// VLIW bundle encoding has independent LDA/LDB bit fields, plus one ST
/// field) -- callers map `SlotIndex::LoadA/LoadB/Store` directly onto these
/// three variants rather than re-deriving the port count from scratch.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum CorePort {
    /// Primary load port (LDA).
    LoadA,
    /// Secondary load port (LDB).
    LoadB,
    /// Store port.
    Store,
}

impl CorePort {
    /// All three ports. Exists so the port count is counted once (`ALL.len()`)
    /// instead of hand-copied into a second constant that could drift.
    pub const ALL: [CorePort; 3] = [CorePort::LoadA, CorePort::LoadB, CorePort::Store];
}

/// An agent that can request a data-memory bank in a given cycle.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Requester {
    /// One of the compute core's memory ports (see `CorePort`).
    Core(CorePort),
    /// Stream-to-memory DMA channel.
    S2mm(u8),
    /// Memory-to-stream DMA channel.
    Mm2s(u8),
}

/// Number of the core's independent memory ports (LoadA, LoadB, Store --
/// AM020 ch.4:69).
const NUM_CORE_PORTS: usize = CorePort::ALL.len();

/// Number of DMA channels per direction on a compute tile (cross-validated,
/// derived from the architecture spec -- this arbiter only ever sees compute
/// tile requesters, matching `COMPUTE_PHYSICAL_BANKS` below).
///
/// `pub(crate)` so callers that need to size their OWN per-channel
/// bookkeeping (e.g. the coordinator's DMA bank-pressure held-level state)
/// derive it from here instead of re-deriving it from `xdna_archspec`
/// directly -- one canonical path, not two that happen to agree.
pub(crate) const NUM_DMA_CHANNELS: usize = xdna_archspec::aie2::compute::NUM_DMA_CHANNELS as usize;

/// Total number of distinct requester IDENTITIES the arbiter's round-robin
/// state must track: one slot per core port, one slot per S2mm channel, one
/// slot per Mm2s channel. Derived so it can never silently drift from the
/// real channel count (Finding 2: the old `[bool; 8]` collided with the
/// unrelated bank count and any requester past index 7 was silently never
/// reported lost).
const NUM_REQUESTERS: usize = NUM_CORE_PORTS + 2 * NUM_DMA_CHANNELS;

impl Requester {
    /// Stable ordinal identity used to key round-robin priority.
    ///
    /// This MUST depend only on which requester this fundamentally is (a
    /// specific core port, or a specific DMA channel+direction) -- never on
    /// position within a given cycle's demand list. That list is rebuilt
    /// every cycle and its shape changes constantly (DMA channels
    /// appear/disappear as bursts start and finish; a core port is only
    /// present when the bundle actually issues a memory op on it that
    /// cycle), so indexing the rotor by list position makes "round robin"
    /// degenerate: index N names a different agent every cycle, and the
    /// rotor's fairness guarantee stops meaning anything.
    ///
    /// Panics on an out-of-range channel id instead of silently mis-tracking
    /// it -- an out-of-range requester must be impossible, not quietly
    /// dropped from arbitration.
    fn ordinal(self) -> usize {
        match self {
            Requester::Core(CorePort::LoadA) => 0,
            Requester::Core(CorePort::LoadB) => 1,
            Requester::Core(CorePort::Store) => 2,
            Requester::S2mm(ch) => {
                assert!(
                    (ch as usize) < NUM_DMA_CHANNELS,
                    "S2mm channel {ch} exceeds NUM_DMA_CHANNELS ({NUM_DMA_CHANNELS})"
                );
                NUM_CORE_PORTS + ch as usize
            }
            Requester::Mm2s(ch) => {
                assert!(
                    (ch as usize) < NUM_DMA_CHANNELS,
                    "Mm2s channel {ch} exceeds NUM_DMA_CHANNELS ({NUM_DMA_CHANNELS})"
                );
                NUM_CORE_PORTS + NUM_DMA_CHANNELS + ch as usize
            }
        }
    }
}

/// Outcome of one cycle of arbitration.
#[derive(Clone, Debug, Default)]
pub struct Arbitration {
    /// Requesters that got EVERY bank they asked for. Their access completed
    /// and latched this cycle: per the retry contract (module docs) they must
    /// NOT re-request while the rest of their bundle/burst is stalled.
    pub granted: Vec<Requester>,
    /// Requesters that were denied at least one bank they asked for. They must
    /// hold their request and retry it next cycle -- but only for the banks
    /// they did NOT win (see `bank_grants`).
    pub lost: Vec<Requester>,
    /// Per requester, the mask of banks it actually WON this cycle. A requester
    /// that asked for two banks and won one appears here with the one it won,
    /// and in `lost` (its access is not complete). Each bank is an independent
    /// single-port SRAM with its own arbiter, so the half of a multi-bank
    /// access that won its bank COMPLETED and latched -- the retry re-requests
    /// only the unserved half.
    pub bank_grants: Vec<(Requester, u16)>,
    /// Bitmask of banks that had more than one requester this cycle. Drives the
    /// CONFLICT_DM_BANK_n trace events.
    pub contended_banks: u16,
}

impl Arbitration {
    /// Did the core lose arbitration on ANY of its ports this cycle?
    ///
    /// AM020 ch.4:69: a bank conflict on any port stalls the WHOLE datapath,
    /// so this is the bundle-granularity answer: the core cannot retire this
    /// bundle this cycle.
    ///
    /// **It is not sufficient on its own.** The stalled bundle must be
    /// re-presented next cycle WITHOUT the ports that already won -- their
    /// accesses completed and latched (see `granted`). Re-presenting the
    /// identical demand every cycle deterministically livelocks any bundle
    /// whose own ports collide (a single-port bank grants exactly one core
    /// port per cycle, so `core_lost()` would be true forever). The
    /// coordinator must accumulate `granted` core ports across the stall and
    /// feed them back as the "already served" set (see
    /// `CycleAccurateExecutor::peek_bank_demand`'s `served` argument).
    pub fn core_lost(&self) -> bool {
        self.lost.iter().any(|r| matches!(r, Requester::Core(_)))
    }

    /// Banks each core port won this cycle -- accumulate these into the served
    /// set while a bundle is stalled. Per BANK, not per port: a port that won
    /// bank 0 and lost bank 1 latched bank 0 and must re-request only bank 1.
    pub fn granted_core_banks(&self) -> impl Iterator<Item = (CorePort, u16)> + '_ {
        self.bank_grants.iter().filter_map(|(r, m)| match r {
            Requester::Core(p) if *m != 0 => Some((*p, *m)),
            _ => None,
        })
    }
}

/// Per physical bank of a compute tile's data memory: the accesses it has served since
/// reset. The core-versus-DMA decision reads it (`BankArbiter::arbitrate`).
///
/// Lives on the tile (`Tile::bank_access_counts`), not on the arbiter, because the
/// configuration path writes the initial image into data memory before any core or DMA
/// runs, and silicon counts those writes: one access per initialised 16-byte line, per bank
/// (`record_config_write`). Reset with the tile (a partition or column reset): separately
/// loaded runs of one design show the same phase on silicon.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BankAccessCounts {
    served: [u64; COMPUTE_PHYSICAL_BANKS as usize],
}

impl BankAccessCounts {
    /// Accesses physical bank `bank` has served since reset.
    pub fn served(&self, bank: usize) -> u64 {
        self.served[bank]
    }

    /// One access served by `bank` this cycle.
    pub(crate) fn serve(&mut self, bank: usize) {
        self.served[bank] += 1;
    }

    /// A configuration-path write of `len` bytes at tile-local `offset` (the program image's
    /// data, written before the core and DMA run, or any other host or control write into
    /// data memory): each 16-byte line it touches is one access of that line's bank.
    ///
    /// The unit is measured for the image load (silicon, ring probe with extra initialised
    /// words in bank 0: 1 to 4 words in one line move the count by 1, 5 words over two lines
    /// and 12 words over lines b0, b1, b0 by 2). That a single-word register write into data
    /// memory (control packets, a host `write32`) counts as one access too is the same rule
    /// applied to a path that is not measured.
    pub fn record_config_write(&mut self, offset: usize, len: usize) {
        if len == 0 {
            return;
        }
        let line = BankLayout::Compute.access_granule_bytes() as usize;
        let mem = xdna_archspec::aie2::compute::MEMORY_SIZE as usize;
        let first = offset / line;
        let last = (offset + len - 1) / line;
        for l in first..=last {
            let addr = (l * line) % mem;
            self.serve(BankLayout::Compute.physical_bank(addr as u32) as usize);
        }
    }

    /// Whether a contended `bank` serves the DMA this cycle: its served count is at the DMA
    /// residue (module docs).
    fn dma_turn(&self, bank: usize) -> bool {
        use xdna_archspec::aie2::timing::{BANK_ARBITER_DMA_RESIDUE, BANK_ARBITER_PERIOD};
        self.served[bank] % BANK_ARBITER_PERIOD as u64 == BANK_ARBITER_DMA_RESIDUE as u64
    }
}

/// One arbiter per physical bank: the served-access count (`BankAccessCounts`) decides core
/// versus DMA, a round-robin rotor decides within the winning class.
#[derive(Clone, Debug)]
pub struct BankArbiter {
    /// Per bank: the requester ORDINAL (a stable identity, never a
    /// demand-list position) that has priority next WITHIN a class (DMA
    /// channels among themselves, core ports among themselves). Advances by
    /// exactly one position every time this bank is contended, regardless of
    /// who wins. Which DMA channel silicon serves when several wait is not
    /// measured; this is the emulator's existing order, kept as such.
    rotor: [u8; COMPUTE_PHYSICAL_BANKS as usize],
}

impl Default for BankArbiter {
    fn default() -> Self {
        Self::new()
    }
}

impl BankArbiter {
    pub fn new() -> Self {
        Self { rotor: [0; COMPUTE_PHYSICAL_BANKS as usize] }
    }

    /// Arbitrate one cycle. `demands` is each requester and the bitmask of
    /// physical banks it needs this cycle; `counts` is the tile's
    /// served-access count, which this advances by one for every bank that
    /// serves an access this cycle (contended or not).
    ///
    /// Grants at most one requester per bank. On a bank wanted by the core and
    /// by a DMA channel the count picks the class (module docs); within the
    /// class, and on a bank wanted only by DMA channels or only by core ports,
    /// the rotor picks by stable requester identity (`Requester::ordinal`). A
    /// requester denied ANY bank it needs is reported in `lost` -- it must
    /// stall and retry next cycle. A requester that got every bank it needed
    /// is reported in `granted` -- its access completed; it must not
    /// re-request (see the module-level retry contract).
    pub fn arbitrate(&mut self, demands: &[(Requester, u16)], counts: &mut BankAccessCounts) -> Arbitration {
        debug_assert!(
            {
                let mut seen = [false; NUM_REQUESTERS];
                demands
                    .iter()
                    .all(|(who, _)| !std::mem::replace(&mut seen[who.ordinal()], true))
            },
            "a requester may appear at most once in a cycle's demands (a duplicate would \
             self-contend and silently corrupt arbitration): {demands:?}"
        );

        let mut out = Arbitration::default();
        let mut denied = [0u16; NUM_REQUESTERS]; // ordinal -> mask of banks lost

        for bank in 0..COMPUTE_PHYSICAL_BANKS as usize {
            let bit = 1u16 << bank;
            // Who wants this bank, by stable ordinal (not demand-list index).
            let wanters: SmallVec<[usize; NUM_REQUESTERS]> = demands
                .iter()
                .filter(|(_, mask)| mask & bit != 0)
                .map(|(who, _)| who.ordinal())
                .collect();

            if wanters.is_empty() {
                continue;
            }
            if wanters.len() > 1 {
                out.contended_banks |= bit;

                // Class: core ordinals are always < NUM_CORE_PORTS; anything
                // else is a DMA channel. Only a core-versus-DMA bank consults
                // the count.
                let core: SmallVec<[usize; NUM_REQUESTERS]> =
                    wanters.iter().copied().filter(|&ord| ord < NUM_CORE_PORTS).collect();
                let dma: SmallVec<[usize; NUM_REQUESTERS]> =
                    wanters.iter().copied().filter(|&ord| ord >= NUM_CORE_PORTS).collect();
                let winner_class = if core.is_empty() || dma.is_empty() {
                    &wanters
                } else if counts.dma_turn(bank) {
                    &dma
                } else {
                    &core
                };

                // Round-robin WITHIN the winning class: the wanter whose ordinal
                // is first at-or-after the rotor wins, wrapping around the fixed
                // requester space.
                let start = self.rotor[bank] as usize;
                let winner_ordinal = *winner_class
                    .iter()
                    .min_by_key(|&&ord| (ord + NUM_REQUESTERS - start) % NUM_REQUESTERS)
                    .expect("winner_class is non-empty (derived from wanters, which has >= 2 elements)");

                for &ord in &wanters {
                    if ord != winner_ordinal {
                        denied[ord] |= bit;
                    }
                }
                // Advance one position every contended cycle, regardless of who
                // won: jumping to the winner's position would let a pair of
                // continuously-present rivals trap the pointer between
                // themselves, phase-locking out an intermittent requester.
                self.rotor[bank] = ((start + 1) % NUM_REQUESTERS) as u8;
            }
            // The bank served exactly one access this cycle, whoever made it.
            counts.serve(bank);
        }

        for (who, mask) in demands.iter() {
            let lost_banks = denied[who.ordinal()];
            out.bank_grants.push((*who, mask & !lost_banks));
            if lost_banks != 0 {
                out.lost.push(*who);
            } else {
                out.granted.push(*who);
            }
        }
        out
    }
}

/// Test fixtures shared with the DMA engine's arbitration tests.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::device::banking::banks_for_access;

    pub(crate) fn lone_core_access(
        arb: &mut BankArbiter,
        counts: &mut BankAccessCounts,
        port: CorePort,
        local: u32,
    ) {
        let mask = banks_for_access(local, 4, BankLayout::Compute);
        let a = arb.arbitrate(&[(Requester::Core(port), mask)], counts);
        assert!(a.lost.is_empty() && a.contended_banks == 0);
    }

    /// The lock-fetch / arb / ring probe design's data-memory accesses before its window, in
    /// the order they happen: the configuration image's initialised buffers (src: 64 words at
    /// 0x400; ldbuf: 16 words at 0x800), the crt0 stores of p7 and lr (tile-local 0x4, 0x0),
    /// the kernel's frame stores of r16..r31 (0x20..0x5C), then `settle` extra loads at one
    /// address (the arb probe's `m` loads before the release). Counted, not offset: the result
    /// is whatever the accesses add up to.
    pub(crate) fn lock_fetch_history(settle: Option<(u32, usize)>) -> (BankArbiter, BankAccessCounts) {
        let mut arb = BankArbiter::new();
        let mut counts = BankAccessCounts::default();
        counts.record_config_write(0x400, 64 * 4);
        counts.record_config_write(0x800, 16 * 4);
        for addr in [0x4, 0x0] {
            lone_core_access(&mut arb, &mut counts, CorePort::Store, addr);
        }
        for addr in (0x20..0x60).step_by(4) {
            lone_core_access(&mut arb, &mut counts, CorePort::Store, addr);
        }
        if let Some((addr, m)) = settle {
            for _ in 0..m {
                lone_core_access(&mut arb, &mut counts, CorePort::LoadA, addr);
            }
        }
        (arb, counts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::banking::banks_for_access;

    // -- the served-access count (silicon: lock-fetch arb and ring probes, NPU1 2026-11) ------
    //
    // Cycles below are relative to the MM2S 0 lock grant g (Part-1 rows) or to the core's camp
    // start (ring rows), as the probe tables give them. Every expected number is a silicon
    // measurement; the harness inputs are the DMA's request cycles (the probe's MM2S access map,
    // or the first cycle of each measured contact run) and the core's access cycles.

    /// MM2S 0 bank accesses of the probe's 64-word lock-started BD, grant-relative (silicon map).
    const LOCK_FETCH_MM2S_BANK0: &[u64] = &[2, 4, 7, 15, 23, 31, 39, 47];
    const LOCK_FETCH_MM2S_BANK1: &[u64] = &[3, 5, 11, 19, 27, 35, 43, 51];

    use test_support::{lock_fetch_history, lone_core_access};

    /// What one bank saw over a window: the cycles it was contended (CONFLICT_DM_BANK), the
    /// cycles the core lost (each one MEMORY_STALL), and when each DMA request was served.
    #[derive(Debug, Default, PartialEq, Eq)]
    struct Window {
        contended: Vec<u64>,
        core_lost: Vec<u64>,
        dma_served: Vec<(Requester, u64)>,
    }

    /// Drive one physical bank cycle by cycle from cycle 0 to `until`. The core makes `n`
    /// consecutive accesses on `port` from `start`, one per cycle; a lost access repeats the
    /// next cycle. Each DMA requester asks at each of its arrival cycles and holds the request
    /// until served (a denied DMA retries the next cycle); a request whose arrival passed while
    /// an earlier one was waiting is presented the cycle after that one is served.
    fn run_bank(
        arb: &mut BankArbiter,
        counts: &mut BankAccessCounts,
        bank: usize,
        core: Option<(CorePort, u64, u32)>,
        dma: &[(Requester, &[u64])],
        until: u64,
    ) -> Window {
        let bit = 1u16 << bank;
        let mut next = vec![0usize; dma.len()];
        let mut core_left = core.map_or(0, |(_, _, n)| n);
        let mut w = Window::default();
        for t in 0..until {
            let mut demands: Vec<(Requester, u16)> = Vec::new();
            if let Some((port, start, _)) = core {
                if t >= start && core_left > 0 {
                    demands.push((Requester::Core(port), bit));
                }
            }
            for (i, (who, arrivals)) in dma.iter().enumerate() {
                if next[i] < arrivals.len() && arrivals[next[i]] <= t {
                    demands.push((*who, bit));
                }
            }
            if demands.is_empty() {
                continue;
            }
            let a = arb.arbitrate(&demands, counts);
            if a.contended_banks & bit != 0 {
                w.contended.push(t);
            }
            if a.core_lost() {
                w.core_lost.push(t);
            } else if demands.iter().any(|(r, _)| matches!(r, Requester::Core(_))) {
                core_left -= 1;
            }
            for (i, (who, _)) in dma.iter().enumerate() {
                if a.granted.contains(who) {
                    next[i] += 1;
                    w.dma_served.push((*who, t));
                }
            }
        }
        w
    }

    /// Part-1 probe row: `n` same-bank core accesses from grant + `start` against MM2S 0's
    /// lock-started BD, with `m` settle loads on the bank before the release.
    fn arb_row(bank: usize, port: CorePort, n: u32, start: u64, m: usize) -> Window {
        let settle_addr = if bank == 0 { 0x800 } else { 0x810 };
        let (mut arb, mut counts) = lock_fetch_history(Some((settle_addr, m)));
        let map = if bank == 0 {
            LOCK_FETCH_MM2S_BANK0
        } else {
            LOCK_FETCH_MM2S_BANK1
        };
        run_bank(&mut arb, &mut counts, bank, Some((port, start, n)), &[(Requester::Mm2s(0), map)], 60)
    }

    /// The count from reset, counted (the probe analysis's accounting, now the emulator's):
    /// bank 0 has served 8 + 2 image lines, 2 crt0 stores and 8 frame stores before the
    /// window, bank 1 8 + 2 lines and 8 frame stores.
    #[test]
    fn the_lock_fetch_design_reaches_its_window_with_20_and_18_served_accesses() {
        let (_, counts) = lock_fetch_history(None);
        assert_eq!((counts.served(0), counts.served(1)), (20, 18));
        for bank in 2..COMPUTE_PHYSICAL_BANKS as usize {
            assert_eq!(counts.served(bank), 0, "bank {bank}");
        }
    }

    /// T1 (calibrates the residue): a single bank-0 load at g + 7 loses (5 of 5 contacts);
    /// one cycle earlier (grant phase + 3) it meets nothing.
    #[test]
    fn t1_a_single_bank0_load_at_g7_loses_the_bank() {
        for port in [CorePort::LoadA, CorePort::Store] {
            let w = arb_row(0, port, 1, 7, 0);
            assert_eq!((w.contended, w.core_lost), (vec![7], vec![7]), "{port:?}");
        }
        let w = arb_row(0, CorePort::LoadA, 1, 6, 0);
        assert_eq!((w.contended, w.core_lost), (vec![], vec![]));
    }

    /// T2 (the count from reset, no per-bank constant): bank 1's single load loses at g + 27
    /// and wins at every other MM2S access; at g + 51 the denied DMA access is served at g + 52
    /// (FINISHED_BD + 53 on silicon) and that lone retry is not a conflict.
    #[test]
    fn t2_a_single_bank1_load_loses_only_at_g27() {
        let w = arb_row(1, CorePort::LoadA, 1, 27, 0);
        assert_eq!((w.contended, w.core_lost), (vec![27], vec![27]));
        for cell in [3u64, 5, 11, 19, 35, 43, 51] {
            let w = arb_row(1, CorePort::LoadA, 1, cell, 0);
            assert_eq!((w.contended, w.core_lost), (vec![cell], vec![]), "g + {cell}");
        }
        let w = arb_row(1, CorePort::LoadA, 1, 51, 0);
        assert_eq!(w.dma_served.last(), Some(&(Requester::Mm2s(0), 52)));
    }

    /// T3: bank 0's single load also loses at g + 47 (5 of 5, predicted from the count before
    /// it was run) and wins at g + 2, 4, 15, 23, 31 and 39.
    #[test]
    fn t3_a_single_bank0_load_loses_at_g47_and_wins_elsewhere() {
        let w = arb_row(0, CorePort::LoadA, 1, 47, 0);
        assert_eq!((w.contended, w.core_lost), (vec![47], vec![47]));
        for cell in [2u64, 4, 15, 23, 31, 39] {
            let w = arb_row(0, CorePort::LoadA, 1, cell, 0);
            assert_eq!((w.contended, w.core_lost), (vec![cell], vec![]), "g + {cell}");
        }
    }

    /// T4, history: the core's own uncontended loads long before the DMA runs advance the
    /// count. Four settle loads turn the g + 15 contact into a loss (2 of 2), one to three keep
    /// it a win; one to four turn the g + 7 loss into a win (9 of 9).
    #[test]
    fn t4_settle_loads_before_the_release_move_the_losing_access() {
        let w = arb_row(0, CorePort::LoadA, 1, 15, 4);
        assert_eq!((w.contended, w.core_lost), (vec![15], vec![15]));
        for m in 1..=3 {
            let w = arb_row(0, CorePort::LoadA, 1, 15, m);
            assert_eq!((w.contended, w.core_lost), (vec![15], vec![]), "m = {m}");
        }
        for m in 1..=4 {
            let w = arb_row(0, CorePort::LoadA, 1, 7, m);
            assert_eq!((w.contended, w.core_lost), (vec![7], vec![]), "m = {m}");
        }
    }

    /// T5, bursts (table rows; a start one cycle early is the + 3 grant phase): the DMA is
    /// refused while the count is off its residue and served at the first contended cycle on
    /// it, at most the 5th; the conflict event marks every contended cycle including the one
    /// the DMA won, and not the core's lone retry after it. Stores behave as loads.
    #[test]
    fn t5_bursts_let_the_dma_in_on_the_residue_and_mark_every_contended_cycle() {
        #[rustfmt::skip]
        let rows: &[(usize, u32, u64, &[u64], &[u64])] = &[
            // bank, n, start, contended, core lost
            (0, 5, 15, &[15, 16, 17, 18, 19], &[19]),   // ar0l5k16m0
            (0, 6, 15, &[15, 16, 17, 18, 19], &[19]),   // ar0l6k16m0
            (0, 5, 14, &[15, 16, 17, 18], &[18]),       // ar0l5k16m0 (+3)
            (0, 4, 15, &[15, 16, 17, 18], &[]),         // ar0l4k16m0
            (0, 4, 14, &[15, 16, 17], &[]),             // ar0l4k16m0 (+3)
            (0, 2, 14, &[15], &[]),                     // ar0l2k16m0 (+3)
            (0, 3, 23, &[23, 24, 25], &[]),             // ar0l3k24m0
            (0, 4, 23, &[23, 24, 25, 26], &[26]),       // ar0l4k24m0
            (0, 4, 22, &[23, 24, 25], &[25]),           // ar0l4k24m0 (+3)
            (0, 2, 31, &[31, 32], &[]),                 // ar0l2k32m0
            (0, 3, 31, &[31, 32, 33], &[33]),           // ar0l3k32m0
            (0, 3, 30, &[31, 32], &[32]),               // ar0l3k32m0 (+3)
            (0, 6, 31, &[31, 32, 33], &[33]),           // ar0l6k32m0
            (1, 2, 19, &[19, 20], &[20]),               // ar1l2k20m0
            (1, 3, 18, &[19], &[19]),                   // ar1l3k20m0 (+3)
            (1, 4, 35, &[35, 36, 37, 38], &[]),         // ar1l4k36m0
            (1, 4, 34, &[35, 36, 37], &[]),             // ar1l4k36m0 (+3)
            (1, 5, 35, &[35, 36, 37, 38, 39], &[39]),   // ar1l5k36m0
            (1, 5, 34, &[35, 36, 37, 38], &[38]),       // ar1l5k36m0 (+3)
            (1, 6, 35, &[35, 36, 37, 38, 39], &[39]),   // ar1l6k36m0
        ];
        for &(bank, n, start, contended, lost) in rows {
            let w = arb_row(bank, CorePort::LoadA, n, start, 0);
            assert_eq!(
                (w.contended.as_slice(), w.core_lost.as_slice()),
                (contended, lost),
                "bank {bank} n {n} g + {start}"
            );
            if bank == 0 {
                let s = arb_row(bank, CorePort::Store, n, start, 0);
                assert_eq!(
                    (s.contended, s.core_lost),
                    (w.contended, w.core_lost),
                    "stores, n {n} g + {start}"
                );
            }
        }
    }

    /// The image unit (ring probe, extra initialised words in bank 0 at 0x0C00 / 0x0C20): one
    /// count per initialised 16-byte line of the bank, not per word and not per buffer.
    #[test]
    fn a_configuration_write_counts_once_per_16_byte_line_of_each_bank() {
        let bank0 = |writes: &[(usize, usize)]| {
            let mut c = BankAccessCounts::default();
            for &(addr, words) in writes {
                c.record_config_write(addr, 4 * words);
            }
            (c.served(0), c.served(1))
        };
        for words in 1..=4 {
            assert_eq!(bank0(&[(0xC00, words)]), (1, 0), "{words} words in one line");
        }
        assert_eq!(bank0(&[(0xC00, 4), (0xC20, 1)]), (2, 0), "5 words, two buffers, two lines");
        assert_eq!(bank0(&[(0xC00, 12)]), (2, 1), "12 words in one buffer: lines b0, b1, b0");
    }

    /// Ring camp, `variant` against a core that loads bank 0 every cycle from camp + 0
    /// (silicon: 90 consecutive loads), with `extra` (address, words) initialised in bank 0.
    fn ring_camp(dma: Requester, arrivals: &[u64], extra: &[(usize, usize)]) -> Window {
        let (mut arb, mut counts) = lock_fetch_history(None);
        for &(addr, words) in extra {
            counts.record_config_write(addr, 4 * words);
        }
        run_bank(&mut arb, &mut counts, 0, Some((CorePort::LoadA, 0, 90)), &[(dma, arrivals)], 120)
    }

    /// T4 (ring order): each of the four DMA channels against a camping core is let in at one
    /// residue of the count, the same one (camp cycle mod 5 = 2 from the common count), after
    /// 0..4 refusals, and the core loses exactly those cycles. Arrivals are the first cycles of
    /// the measured contact runs (one run per DMA request); the run ends and the losses are
    /// the rule's.
    #[test]
    fn ring_each_dma_channel_against_a_camping_core_wins_on_the_same_single_residue() {
        #[rustfmt::skip]
        let rows: &[(Requester, &[u64], &[u64])] = &[
            // rgm0 r1 (grant + 2)
            (Requester::Mm2s(0), &[3, 9, 14, 19, 24, 32, 40, 48], &[7, 12, 17, 22, 27, 32, 42, 52]),
            // rgm1 r1 (grant + 3)
            (Requester::Mm2s(1), &[4, 9, 14, 19, 25, 33, 41, 49], &[7, 12, 17, 22, 27, 37, 42, 52]),
            // rgs0 r2
            (Requester::S2mm(0), &[18, 26, 34, 42, 50, 58, 66, 74], &[22, 27, 37, 42, 52, 62, 67, 77]),
            // rgs1 r1
            (Requester::S2mm(1), &[27, 35, 43, 51, 59, 67, 75, 83], &[27, 37, 47, 52, 62, 67, 77, 87]),
        ];
        for &(dma, arrivals, lost) in rows {
            let w = ring_camp(dma, arrivals, &[]);
            assert_eq!(w.core_lost, lost, "{dma:?}");
            let served: Vec<u64> = w.dma_served.iter().map(|&(_, t)| t).collect();
            assert_eq!(served, lost, "{dma:?}: served exactly where the core lost");
            for &t in lost {
                assert_eq!(t % 5, 2, "{dma:?}: the one residue");
            }
            // every refused cycle and the won one are conflicts: the runs arrival..=served
            let runs: Vec<u64> = arrivals.iter().zip(lost).flat_map(|(&a, &l)| a..=l).collect();
            assert_eq!(w.contended, runs, "{dma:?}");
        }
    }

    /// T5 (ring pads): one extra initialised line in bank 0 moves the residue from 2 to 1, two
    /// lines (in two buffers, or one 12-word buffer over lines b0, b1, b0) to 0.
    #[test]
    fn ring_pad_lines_shift_the_dma_turn() {
        #[rustfmt::skip]
        let rows: &[(&[(usize, usize)], &[u64], &[u64])] = &[
            (&[(0xC00, 1)], &[3, 8, 13, 18, 24, 32, 40, 48], &[6, 11, 16, 21, 26, 36, 41, 51]),          // p1 r1
            (&[(0xC00, 4)], &[3, 8, 13, 18, 24, 32, 40, 48], &[6, 11, 16, 21, 26, 36, 41, 51]),          // p4 r2
            (&[(0xC00, 4), (0xC20, 1)], &[3, 7, 12, 17, 24, 32, 40, 48], &[5, 10, 15, 20, 25, 35, 40, 50]), // p5 r2
            (&[(0xC00, 12)], &[3, 7, 12, 17, 24, 32, 40, 48], &[5, 10, 15, 20, 25, 35, 40, 50]),         // p12o r1
        ];
        for &(extra, arrivals, lost) in rows {
            let w = ring_camp(Requester::Mm2s(0), arrivals, extra);
            assert_eq!(w.core_lost, lost, "pads {extra:?}");
        }
    }

    /// T6: two DMA channels with the core camping share the ONE DMA turn: the core loses
    /// exactly every 5th cycle at the residue while either waits (rgmm n120 / rgms n090: 16
    /// losses, camp + 7, 12, ..., 82, never two in a row), and a served access to one channel
    /// advances the count the other sees. Which channel takes a turn is the rotor's (unmeasured).
    #[test]
    fn two_dma_channels_with_a_camping_core_share_one_dma_turn() {
        let (mut arb, mut counts) = lock_fetch_history(None);
        let m0: &[u64] = &[3, 9, 14, 19, 24, 32, 40, 48];
        let m1: &[u64] = &[3, 9, 14, 19, 24, 32, 40, 48];
        let w = run_bank(
            &mut arb,
            &mut counts,
            0,
            Some((CorePort::LoadA, 0, 120)),
            &[(Requester::Mm2s(0), m0), (Requester::Mm2s(1), m1)],
            140,
        );
        let expected: Vec<u64> = (0..16).map(|i| 7 + 5 * i).collect();
        assert_eq!(w.core_lost, expected);
        assert_eq!(w.dma_served.len(), 16);
    }

    /// Two DMA channels meeting at a bank with the core silent raise the conflict event (ring
    /// mx: one CONFLICT_DM_BANK_0 per run, 6 of 6). Which one is served is not measured.
    #[test]
    fn two_dma_channels_meeting_at_a_bank_are_a_conflict() {
        let mut arb = BankArbiter::new();
        let mut counts = BankAccessCounts::default();
        let a = arb.arbitrate(&[(Requester::Mm2s(0), 1), (Requester::Mm2s(1), 1)], &mut counts);
        assert_eq!(a.contended_banks, 1);
        assert_eq!((a.granted.len(), a.lost.len()), (1, 1));
        assert_eq!(counts.served(0), 1);
    }

    /// Served accesses advance the count, contended or not, one per bank per cycle; refused
    /// requests add nothing.
    #[test]
    fn only_served_accesses_advance_the_count() {
        let mut arb = BankArbiter::new();
        let mut counts = BankAccessCounts::default();
        arb.arbitrate(&[(Requester::Core(CorePort::LoadA), 0b01), (Requester::S2mm(0), 0b10)], &mut counts);
        assert_eq!((counts.served(0), counts.served(1)), (1, 1));
        arb.arbitrate(&[(Requester::Core(CorePort::LoadA), 0b01), (Requester::S2mm(0), 0b01)], &mut counts);
        assert_eq!(counts.served(0), 2, "a contended bank serves one access, the loser adds nothing");
    }

    /// The cross-class bound (silicon sticky captures: 113 five-cycle runs, all ending with the
    /// DMA served, none longer): whatever the count's phase, a DMA waiting against a core that
    /// wants the bank every cycle is served within 5 contended cycles, and then the core again.
    #[test]
    fn a_camping_core_lets_a_waiting_dma_in_within_one_period() {
        for phase in 0..5u64 {
            let mut arb = BankArbiter::new();
            let mut counts = BankAccessCounts::default();
            for _ in 0..phase {
                lone_core_access(&mut arb, &mut counts, CorePort::LoadA, 0);
            }
            let w = run_bank(
                &mut arb,
                &mut counts,
                0,
                Some((CorePort::LoadA, 0, 40)),
                &[(Requester::S2mm(0), &[0, 0, 0, 0])],
                50,
            );
            let served: Vec<u64> = w.dma_served.iter().map(|&(_, t)| t).collect();
            assert!(served[0] < 5, "phase {phase}: first DMA turn within 5 cycles ({served:?})");
            assert!(
                served.windows(2).all(|p| p[1] - p[0] == 5),
                "phase {phase}: then every 5th ({served:?})"
            );
            assert_eq!(w.core_lost, served);
        }
    }

    #[test]
    fn two_core_ports_still_rotate_within_the_core_class() {
        // Core-vs-core is unchanged: exactly one core port wins, and across two
        // contended cycles the winner alternates (within-class round-robin).
        let mut arb = BankArbiter::new();
        let d =
            [(Requester::Core(CorePort::LoadA), 1u16 << 0), (Requester::Core(CorePort::Store), 1u16 << 0)];
        let first = arb.arbitrate(&d, &mut BankAccessCounts::default()).lost;
        let second = arb.arbitrate(&d, &mut BankAccessCounts::default()).lost;
        assert_eq!(first.len(), 1);
        assert_ne!(first, second, "within-core rotation still alternates the loser");
    }

    #[test]
    fn no_contention_everyone_wins() {
        let mut arb = BankArbiter::new();
        let a = arb.arbitrate(
            &[(Requester::Core(CorePort::LoadA), 1 << 0), (Requester::S2mm(0), 1 << 1)],
            &mut BankAccessCounts::default(),
        );
        assert!(a.lost.is_empty());
        assert_eq!(a.contended_banks, 0);
    }

    #[test]
    fn same_bank_collision_grants_exactly_one() {
        let mut arb = BankArbiter::new();
        let a = arb.arbitrate(
            &[(Requester::Core(CorePort::LoadA), 1 << 0), (Requester::S2mm(0), 1 << 0)],
            &mut BankAccessCounts::default(),
        );
        assert_eq!(a.lost.len(), 1, "exactly one requester loses a single bank");
        assert_eq!(a.contended_banks, 1 << 0);
    }

    #[test]
    fn a_requester_losing_any_needed_bank_is_reported_lost() {
        let mut arb = BankArbiter::new();
        // Core needs banks 0 and 1; DMA contends only on bank 1.
        let a = arb.arbitrate(
            &[(Requester::Core(CorePort::LoadA), (1 << 0) | (1 << 1)), (Requester::S2mm(0), 1 << 1)],
            &mut BankAccessCounts::default(),
        );
        // Whoever loses bank 1 is reported; contention is only on bank 1.
        assert_eq!(a.contended_banks, 1 << 1);
        assert_eq!(a.lost.len(), 1);
    }

    /// One requester in the starvation sweep: who it is, the bank mask it
    /// demands, how often it issues a fresh request, and whether its retry
    /// honours the sticky grant per BANK (the core's discipline: keep the banks
    /// you won, re-request only the rest) or re-presents its whole mask (the
    /// DMA's discipline: a denied channel's step is skipped entirely, so its
    /// demand next cycle is byte-identical).
    #[derive(Copy, Clone, Debug)]
    struct SweepReq {
        who: Requester,
        mask: u16,
        period: usize,
        phase: usize,
        bank_sticky: bool,
    }

    /// Drives the arbiter over `CYCLES` cycles with a FAITHFUL retry model
    /// (AM020 ch.2:166): every requester holds a pending request until every
    /// bank of it is granted -- a loser is stalled, it does not withdraw and
    /// come back on its next natural period. Each requester issues a fresh
    /// request on cycles where `t % period == phase` (coalesced if one is still
    /// pending). `skew` pre-contends bank 0 that many times before the sweep
    /// starts, putting the per-bank rotors into a fixed relative PHASE -- which
    /// is the whole game for a multi-bank requester: it only completes on a
    /// cycle where every rotor it needs happens to favour it at once.
    ///
    /// Returns (grants, worst wait in cycles from request-issue to full grant,
    /// every bank that was ever CONTENDED). The last one is what lets a caller
    /// prove its cell was not vacuous: a scenario whose requesters never collide
    /// passes any starvation assertion for free, and a test that cannot fail is
    /// not a test.
    fn simulate_retry_contract(reqs: &[SweepReq], skew: usize) -> (Vec<u32>, u64, u16) {
        // 100 cycles, not 400: the round-robin bound is NUM_REQUESTERS (7)
        // contended cycles and the worst wait observed anywhere in the sweep is
        // 6, so a requester with ZERO grants in 100 cycles is unambiguously
        // starving -- >14x the bound, with room for any rotor warmup transient.
        // The proof lives in the CELLS (every mix x mask x rival shape x period x
        // phase x cadence x rotor skew), not in the cycle count per cell; 400 only
        // bought wall-clock. If 100 ever fails to show a grant that 400 would
        // have, the bound itself is wrong and that is a finding, not a knob.
        const CYCLES: u64 = 100;

        let mut arb = BankArbiter::new();
        for _ in 0..skew {
            // Contend bank 0 only: bank 0's rotor advances, the others do not.
            arb.arbitrate(
                &[(Requester::Core(CorePort::Store), 1 << 0), (Requester::Mm2s(1), 1 << 0)],
                &mut BankAccessCounts::default(),
            );
        }

        // Per requester: (cycle the request issued, banks still unserved).
        let mut pending: Vec<Option<(u64, u16)>> = vec![None; reqs.len()];
        let mut grants = vec![0u32; reqs.len()];
        let mut worst_wait = 0u64;
        let mut contended = 0u16;

        for t in 0..CYCLES {
            for (i, r) in reqs.iter().enumerate() {
                if pending[i].is_none() && (t as usize) % r.period == r.phase {
                    pending[i] = Some((t, r.mask));
                }
            }
            let demands: Vec<(Requester, u16)> = reqs
                .iter()
                .zip(pending.iter())
                .filter_map(|(r, p)| p.map(|(_, unserved)| (r.who, unserved)))
                .collect();
            if demands.is_empty() {
                continue;
            }

            let a = arb.arbitrate(&demands, &mut BankAccessCounts::default());
            contended |= a.contended_banks;
            for (i, r) in reqs.iter().enumerate() {
                let Some((issued, unserved)) = pending[i] else {
                    continue;
                };
                let won = a
                    .bank_grants
                    .iter()
                    .find(|(who, _)| *who == r.who)
                    .map_or(0, |(_, banks)| *banks);
                let left = if r.bank_sticky {
                    unserved & !won // keep the half that latched (silicon)
                } else if won == unserved {
                    0 // all-or-nothing: only a whole grant completes it
                } else {
                    unserved // denied any bank -> the whole request re-presents
                };
                if left == 0 {
                    worst_wait = worst_wait.max(t - issued);
                    grants[i] += 1;
                    pending[i] = None; // access completed; next natural request may issue
                } else {
                    pending[i] = Some((issued, left));
                }
            }
        }
        (grants, worst_wait, contended)
    }

    /// The bank mask a rival carries, always overlapping the victim so the cell
    /// cannot be vacuous: `scalar` picks one of the victim's OWN banks (a 4-byte
    /// access), `vector` the aligned 16-byte-interleaved bank PAIR containing it
    /// -- which is exactly what `banks_for_access` returns for a 32-byte vector
    /// load/store, the demand a real VLIW bundle's LoadA/LoadB/Store each carry.
    fn rival_mask(victim_mask: u16, i: usize, vector: bool) -> u16 {
        let banks: Vec<u32> = (0..COMPUTE_PHYSICAL_BANKS).filter(|b| (victim_mask >> b) & 1 == 1).collect();
        let bank = banks[i % banks.len()];
        if vector {
            let pair = bank & !1; // physical banks interleave in pairs (BankLayout::Compute)
            (1 << pair) | (1 << (pair + 1))
        } else {
            1 << bank
        }
    }

    #[test]
    fn multi_bank_requester_starves_without_per_bank_stickiness() {
        // The counter-example that killed the old single-bank-only proof, kept
        // as a live regression: a requester demanding TWO banks, whose retry
        // re-presents the WHOLE mask instead of keeping the bank it won, can be
        // starved forever by two single-bank rivals -- the two rotors hold a
        // relative phase in which it never wins both banks in the same cycle,
        // while each rival keeps winning its own. Nothing is wrong with the
        // arbiter (AM020's independent per-bank round-robin is intact); the
        // all-or-nothing RETRY is what starves. This test pins that fact so
        // nobody re-introduces the all-or-nothing retry.
        //
        // All three parties are DMA, not the core: a core-versus-DMA bank is
        // decided by the served-access count (module docs), which would make
        // this about the count instead of the within-class rotor phase-lock the
        // test exists to pin. A bank with no core wanter falls through to the
        // plain rotor.
        let victim = Requester::S2mm(1);
        let reqs = |sticky| {
            vec![
                SweepReq { who: victim, mask: 0b11, period: 1, phase: 0, bank_sticky: sticky },
                SweepReq { who: Requester::S2mm(0), mask: 1 << 0, period: 1, phase: 0, bank_sticky: true },
                SweepReq { who: Requester::Mm2s(0), mask: 1 << 1, period: 1, phase: 0, bank_sticky: true },
            ]
        };

        // WHICH rotor skew is adversarial depends on the ordinals' relative
        // positions, so this sweeps rather than hardcoding one (see the
        // identical reasoning in `dma_whole_mask_retry_starves_a_multi_bank_channel`).
        let starving: Vec<usize> = (0..NUM_REQUESTERS)
            .filter(|&skew| simulate_retry_contract(&reqs(false), skew).0[0] == 0)
            .collect();
        assert!(
            !starving.is_empty(),
            "the all-or-nothing retry is expected to starve the 2-bank requester at some rotor \
             skew -- if this no longer starves, the arbiter changed and this test's premise is stale"
        );

        for skew in &starving {
            let (per_bank, worst, _) = simulate_retry_contract(&reqs(true), *skew);
            assert!(
                per_bank[0] > 0,
                "skew {skew}: per-BANK sticky grants must un-starve the 2-bank requester"
            );
            assert!(
                worst <= NUM_REQUESTERS as u64,
                "skew {skew}: worst wait {worst} exceeds the round-robin bound"
            );
        }
    }

    #[test]
    fn dma_whole_mask_retry_starves_a_multi_bank_channel() {
        // The DMA's retry discipline is all-or-nothing by construction: a denied
        // channel's FSM step is skipped ENTIRELY (`DmaEngine::step_with_denied`),
        // so next cycle it re-presents a byte-identical mask -- it cannot keep a
        // bank it won, because it has no state saying it won one. Under that
        // discipline a DMA channel demanding two banks starves exactly like the
        // core did before `66b42a6a`: the two rotors hold a relative phase in
        // which it never wins both banks in the same cycle while each single-bank
        // rival keeps winning its own.
        //
        // The fix is NOT to give the DMA a per-bank sticky retry. It is that a DMA
        // channel's memory side is ONE 128-bit port and can only ever demand ONE
        // granule -- hence one bank -- per cycle
        // (xdna-emu-priv: research/findings/2026-07-14-dma-bank-access-width.md;
        // `strided_s2mm_takes_one_granule_per_cycle`). A one-bank mask cannot be
        // partially served, so the whole-mask retry is trivially correct and this
        // starvation shape is structurally unreachable. This test pins WHY that
        // matters: if a multi-bank DMA demand ever comes back, so does this.
        // WHICH rivals and WHICH rotor skew starve a given victim depends on the
        // ordinals' relative positions in the rotor space, so this sweeps the
        // adversarial phases rather than hardcoding one -- the claim is that a
        // starving phase EXISTS for a real DMA channel against real rivals (two
        // other DMA channels, each hammering one of its banks), not that any
        // particular skew is magic. Both rivals are DMA, not a core port: a
        // core rival's bank would be decided by the served-access count (see the
        // module docs), not by the within-class rotor this test exists to pin. Starvation is checked as zero grants over
        // CYCLES cycles of continuous demand.
        let victim = Requester::Mm2s(0);
        let rivals = [(Requester::S2mm(0), 1u16 << 0), (Requester::S2mm(1), 1u16 << 1)];
        let reqs: Vec<SweepReq> = std::iter::once(SweepReq {
            who: victim,
            mask: 0b11,
            period: 1,
            phase: 0,
            bank_sticky: false, // the DMA's actual retry: re-present the WHOLE mask
        })
        .chain(rivals.iter().map(|(who, mask)| SweepReq {
            who: *who,
            mask: *mask,
            period: 1,
            phase: 0,
            bank_sticky: true,
        }))
        .collect();

        let starving: Vec<usize> = (0..NUM_REQUESTERS)
            .filter(|&skew| simulate_retry_contract(&reqs, skew).0[0] == 0)
            .collect();
        assert!(
            !starving.is_empty(),
            "a 2-bank DMA demand under the whole-mask retry must be starvable -- an emulator \
             livelock, not a theoretical one. If this no longer starves, the arbiter changed and \
             this test's premise is stale."
        );

        eprintln!("starving rotor skews for a 2-bank DMA demand: {starving:?}");

        // And the same demand with a per-bank sticky retry does not starve --
        // which is the proof that the RETRY, not the arbiter, is the starving
        // ingredient (as on the core, `66b42a6a`).
        let sticky: Vec<SweepReq> = reqs.iter().map(|r| SweepReq { bank_sticky: true, ..*r }).collect();
        for skew in &starving {
            let (grants, worst, _) = simulate_retry_contract(&sticky, *skew);
            assert!(grants[0] > 0, "skew {skew}: per-bank stickiness must un-starve the same demand");
            assert!(worst <= NUM_REQUESTERS as u64, "skew {skew}: wait {worst} exceeds the bound");
        }
    }

    #[test]
    fn no_requester_starves_under_the_retry_contract() {
        // Settles the "rotor aliasing" starvation claim EMPIRICALLY, over the
        // demands the model ACTUALLY GENERATES -- multi-bank included. (The
        // previous version of this sweep hardcoded a single-bank demand for
        // every requester and therefore proved nothing about the two-bank masks
        // `banks_for_access` returns for any 32-byte vector access; a two-bank
        // requester really could starve. See
        // `multi_bank_requester_starves_without_per_bank_stickiness`.)
        //
        // What is proven here: under AM020 ch.2:166's retry contract -- every
        // loser re-presents its UNSERVED banks on the very next cycle, and a
        // bank it won stays won (per-bank sticky grant) -- a pending requester
        // re-asks at every bank it still needs while that bank's rotor sweeps,
        // so each bank grants it within NUM_REQUESTERS contended cycles and the
        // whole access completes within that same bound. Round-robin (AM020's
        // word) PLUS retry (AM020's next sentence) IS the anti-starvation
        // mechanism; they are one design, and the retry has to be at BANK
        // granularity for it to hold.
        //
        // Sweep: 5 requester mixes x every victim bank mask (single-bank, the
        // 2-bank vector-access masks, and a 3-bank straddle) x SCALAR AND VECTOR
        // rivals (so the rivals carry multi-bank masks too, not just the victim)
        // x every demand period 1..=2*NUM_REQUESTERS (covers the alleged period-7
        // aliasing hole and its harmonics) x every phase x two rival cadences x
        // every rotor skew 0..NUM_REQUESTERS (the adversarial relative phases).
        //
        // The rival shape matters and was the LAST hole: an earlier version of
        // this sweep gave every rival a single-bank mask, so the geometry the
        // model really produces -- a VLIW bundle whose LoadA, LoadB and Store
        // EACH carry a 2-bank vector-access mask, contending alongside DMA -- was
        // never exercised. The structural argument says the bound survives it
        // (the rotor advances +1 on every contended cycle, so it reaches any
        // ordinal within NUM_REQUESTERS contended cycles whatever the rivals
        // look like, and a bank once won stays won). That argument is why the
        // sweep is EXPECTED to pass, not a substitute for running it: this arc
        // has twice been burned by a sweep that could not exercise what the model
        // generates, and a bound no test can falsify is how that happens a third
        // time. Every cell also asserts it actually created contention, so a
        // vacuous cell (rivals that never collide with the victim) fails loudly
        // instead of passing for free.
        //
        // All 5 mixes are WITHIN one class (core-vs-core or DMA-vs-DMA): a
        // core-versus-DMA bank is decided by the served-access count, not the
        // rotor, and has its own bound (one turn per BANK_ARBITER_PERIOD served
        // cycles, `a_camping_core_lets_a_waiting_dma_in_within_one_period`).
        // This sweep proves the round-robin retry bound WITHIN a class, for
        // every mask/period/phase/skew the model generates.
        let victim_mixes: [(Requester, Vec<Requester>); 5] = [
            // core self-collision vs 1 other port
            (Requester::Core(CorePort::LoadA), vec![Requester::Core(CorePort::LoadB)]),
            // core self-collision vs BOTH other ports -- the real VLIW
            // geometry, LoadA/LoadB/Store all firing in the same bundle
            (
                Requester::Core(CorePort::LoadA),
                vec![Requester::Core(CorePort::LoadB), Requester::Core(CorePort::Store)],
            ),
            // DMA vs 1 other DMA channel
            (Requester::S2mm(1), vec![Requester::Mm2s(0)]),
            // DMA vs 2 other DMA channels, each pinned to one of the victim's
            // two banks -- the reviewer's original starvation geometry, now
            // within one class
            (Requester::S2mm(1), vec![Requester::Mm2s(0), Requester::Mm2s(1)]),
            // DMA vs every OTHER DMA channel on the tile
            (Requester::S2mm(0), vec![Requester::S2mm(1), Requester::Mm2s(0), Requester::Mm2s(1)]),
        ];
        // Masks the model really generates: a scalar access (one bank), a
        // 32-byte vector access (two adjacent banks -- `banks_for_access`), and
        // an unaligned/strided access spanning three.
        let victim_masks: [u16; 4] = [0b0001, 0b0011, 0b0110, 0b0111];

        let mut worst_overall = 0u64;
        for (victim, rivals) in &victim_mixes {
            for victim_mask in victim_masks {
                for rival_vector in [false, true] {
                    for period in 1..=(2 * NUM_REQUESTERS) {
                        for phase in 0..period {
                            for rival_period in [1usize, NUM_REQUESTERS] {
                                for skew in 0..NUM_REQUESTERS {
                                    let mut reqs = vec![SweepReq {
                                        who: *victim,
                                        mask: victim_mask,
                                        period,
                                        phase,
                                        bank_sticky: true,
                                    }];
                                    // Rivals are spread across the VICTIM'S OWN
                                    // banks -- scalar (one of them) or vector (the
                                    // aligned pair containing one) -- so every cell
                                    // genuinely contends and every bank of a
                                    // multi-bank demand is fought over.
                                    reqs.extend(rivals.iter().enumerate().map(|(i, r)| SweepReq {
                                        who: *r,
                                        mask: rival_mask(victim_mask, i, rival_vector),
                                        period: rival_period,
                                        // Issue in step with the victim, not at a
                                        // fixed phase 0: a rival that shares the
                                        // victim's bank but never its CYCLE never
                                        // collides with it (e.g. victim period 7
                                        // phase 1 vs rival period 7 phase 0), and
                                        // that cell proves nothing. Same phase
                                        // guarantees a collision on every issue
                                        // (modulo the rival's own period -- a
                                        // phase >= period never fires at all).
                                        phase: phase % rival_period,
                                        bank_sticky: true,
                                    }));

                                    let (grants, worst_wait, contended) =
                                        simulate_retry_contract(&reqs, skew);
                                    let cell = format!(
                                        "victim {victim:?} mask {victim_mask:#06b} \
                                         rival_vector={rival_vector} period={period} phase={phase} \
                                         rival_period={rival_period} skew={skew}"
                                    );

                                    assert!(
                                        contended & victim_mask != 0,
                                        "VACUOUS CELL: no bank the victim wants was ever contended, \
                                         so a starvation assertion here passes for free -- {cell}"
                                    );
                                    for (i, r) in reqs.iter().enumerate() {
                                        assert!(grants[i] > 0, "{:?} STARVED (0 grants): {cell}", r.who);
                                    }
                                    assert!(
                                        worst_wait <= NUM_REQUESTERS as u64,
                                        "wait {worst_wait} exceeds the round-robin bound of \
                                         NUM_REQUESTERS ({NUM_REQUESTERS}) contended cycles: {cell}"
                                    );
                                    worst_overall = worst_overall.max(worst_wait);
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(worst_overall > 0, "the sweep must actually produce contention");
        eprintln!("worst observed wait across sweep: {worst_overall} cycles (bound {NUM_REQUESTERS})");
    }

    #[test]
    fn sticky_grants_converge_a_two_way_core_self_collision_in_one_stall_cycle() {
        // CRITICAL: a bank is single-port, so when two of the CORE'S OWN ports
        // hit one bank exactly one is granted per cycle and `core_lost()` is
        // true forever if the retry re-presents BOTH ports -- a deterministic
        // livelock. Hardware does not livelock: the granted port's access
        // COMPLETED and latched, so only the unserved port re-requests. Cost:
        // exactly ONE stall cycle. This drives the arbiter across cycles the
        // way the coordinator (Task 6) must.
        let mut arb = BankArbiter::new();
        let full =
            [(Requester::Core(CorePort::LoadA), 1u16 << 0), (Requester::Core(CorePort::Store), 1u16 << 0)];

        // Cycle 0: both ports present, one wins, the bundle stalls.
        let c0 = arb.arbitrate(&full, &mut BankAccessCounts::default());
        assert_eq!(c0.contended_banks, 1 << 0);
        assert_eq!(c0.granted.len(), 1, "a single-port bank grants exactly one port");
        assert_eq!(c0.lost.len(), 1);
        assert!(c0.core_lost(), "bundle stalls (AM020 ch.4:69: any port conflict stalls the datapath)");

        // Cycle 1: the caller honours the sticky grant -- only the UNSERVED
        // (port, bank) re-requests.
        let served: Vec<(CorePort, u16)> = c0.granted_core_banks().collect();
        assert_eq!(served.len(), 1);
        let retry: Vec<(Requester, u16)> = full
            .iter()
            .copied()
            .map(|(r, mask)| match r {
                Requester::Core(p) => {
                    let won = served.iter().find(|(sp, _)| *sp == p).map_or(0, |(_, m)| *m);
                    (r, mask & !won)
                }
                _ => (r, mask),
            })
            .filter(|(_, mask)| *mask != 0)
            .collect();
        assert_eq!(retry.len(), 1, "only the unserved port re-presents");

        let c1 = arb.arbitrate(&retry, &mut BankAccessCounts::default());
        assert_eq!(c1.contended_banks, 0, "nothing left to contend with");
        assert!(c1.lost.is_empty());
        assert!(!c1.core_lost(), "the bundle retires after exactly ONE stall cycle");
    }

    #[test]
    fn per_bank_arbiters_are_independent() {
        // Each bank has its OWN round-robin pointer (AM020 ch.2:166).
        //
        // Contend bank 0 an ODD number of times (one) before ever touching
        // bank 3, not an even number: an even count returns a shared/global
        // rotor to its starting value by symmetry, so a buggy implementation
        // with ONE rotor for all banks would pass this test by coincidence.
        // Comparing exact `lost` identity (not just cardinality) matters too
        // -- both a correct and a shared-rotor-buggy arbiter deny exactly one
        // requester per contended bank, so only checking `lost.len()` never
        // distinguishes them.
        let mut arb = BankArbiter::new();
        // Two DMA channels: a core-versus-DMA bank is decided by the count, not
        // the rotor, so only a same-class pair exercises the rotor.
        let d0 = [(Requester::Mm2s(0), 1 << 0), (Requester::S2mm(0), 1 << 0)];
        let first = arb.arbitrate(&d0, &mut BankAccessCounts::default());

        // Bank 3 has never been touched. If its rotor is truly independent,
        // its very first contention must resolve exactly like bank 0's very
        // first contention did (both start from a fresh rotor). A shared
        // rotor would instead carry over bank 0's post-arbitration state and
        // flip the winner.
        let d3 = [(Requester::Mm2s(0), 1 << 3), (Requester::S2mm(0), 1 << 3)];
        let a = arb.arbitrate(&d3, &mut BankAccessCounts::default());
        assert_eq!(a.contended_banks, 1 << 3);
        assert_eq!(a.lost, first.lost, "bank 3's rotor must be independent of bank 0's");
    }

    #[test]
    fn distinct_core_ports_contend_on_the_same_physical_bank() {
        // AM020 ch.4:69 + design spec: a physical bank is genuinely
        // single-port, so the core's OWN load and store ports contend when
        // they target the same bank -- there is nothing special about "both
        // requesters are the core". This is the core self-collision case.
        let mut arb = BankArbiter::new();
        let a = arb.arbitrate(
            &[(Requester::Core(CorePort::LoadA), 1 << 0), (Requester::Core(CorePort::Store), 1 << 0)],
            &mut BankAccessCounts::default(),
        );
        assert_eq!(a.contended_banks, 1 << 0, "same-bank load+store must contend");
        assert_eq!(a.lost.len(), 1, "exactly one of the two ports loses");
        assert!(
            a.core_lost(),
            "the core must be reported lost overall (bundle-granularity stall, AM020 ch.4:69)"
        );
    }

    #[test]
    fn distinct_core_ports_on_different_physical_banks_do_not_contend() {
        // The Peano heuristic's common case: a load and a store far enough
        // apart that the paired 16-byte interleave puts them on different
        // physical banks. No contention, no core loss.
        let mut arb = BankArbiter::new();
        let a = arb.arbitrate(
            &[(Requester::Core(CorePort::LoadA), 1 << 0), (Requester::Core(CorePort::Store), 1 << 1)],
            &mut BankAccessCounts::default(),
        );
        assert_eq!(a.contended_banks, 0, "different physical banks must not contend");
        assert!(a.lost.is_empty());
        assert!(!a.core_lost());
    }

    #[test]
    fn three_core_ports_can_all_be_distinct_requesters_at_once() {
        // LoadA, LoadB, and Store are three independent ordinals; a bundle
        // that fires all three on the same bank must still grant exactly
        // one and deny the other two (not silently merge any pair).
        let mut arb = BankArbiter::new();
        let a = arb.arbitrate(
            &[
                (Requester::Core(CorePort::LoadA), 1 << 0),
                (Requester::Core(CorePort::LoadB), 1 << 0),
                (Requester::Core(CorePort::Store), 1 << 0),
            ],
            &mut BankAccessCounts::default(),
        );
        assert_eq!(a.contended_banks, 1 << 0);
        assert_eq!(a.lost.len(), 2, "two of the three ports must lose a single-port bank");
        assert!(a.core_lost());
    }
}
