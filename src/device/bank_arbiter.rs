//! Per-physical-bank memory arbiter: core-class priority with a DMA urgency
//! override.
//!
//! AM020 ch.2:166: "Each memory bank has its own arbitrator to arbitrate
//! between all requesters. The memory bank arbitration is round-robin to avoid
//! starving any requester. It handles a new request every clock cycle. When
//! there are multiple requests in the same cycle to the same memory bank, only
//! one request per cycle is allowed to access the memory. The other requesters
//! are stalled for one cycle and the hardware retries the memory request in the
//! next cycle."
//!
//! Arbitration is over PHYSICAL banks (single-port SRAMs), not the four
//! programmer-visible logical banks -- the hardware exposes eight
//! CONFLICT_DM_BANK events, and an HW capture confirmed a single logical bank
//! splitting its conflicts across two independently-counted arbiters.
//!
//! # Class priority and urgency (winner selection)
//!
//! AM020's "round-robin" describes the retry mechanism (below), not a promise
//! that every requester is treated identically: the compute core outranks the
//! DMA on a contended bank, EXCEPT an urgent DMA -- its egress FIFO near
//! underflow, `DmaEngine::urgent_channels` -- outranks the core right back, so
//! the stream never actually underflows waiting its turn. `arbitrate`'s
//! `winner_class` picks the class -- first non-empty of [urgent DMA, core,
//! other DMA] -- and round-robin applies ONLY WITHIN that class: the rotor
//! still advances by one every contended cycle regardless of who wins (no
//! identity is ever skipped over indefinitely), but it adjudicates a tie
//! inside the winning class only, never across classes. `two_core_ports_still_rotate_within_the_core_class`
//! and `two_urgent_dma_channels_rotate_and_the_core_still_wins_when_neither_is_urgent`
//! pin that within-class rotation for both classes; `core_beats_nonurgent_dma_on_a_contended_bank`,
//! `urgent_dma_beats_the_core`, `core_beats_the_same_dma_every_cycle_without_urgency`,
//! and `dma_is_not_starved_because_urgency_escalates_before_underflow` pin the
//! cross-class priority and its urgency override.
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
//! while that bank's rotor sweeps, so each bank grants it within at most
//! `NUM_REQUESTERS` contended cycles and the whole access completes within that
//! same bound -- PROVIDED it stays inside the winning class every cycle it
//! contends. A non-urgent DMA pitted against an ever-present core is, by
//! design, never inside the winning class: that is not starvation in the
//! AM020 sense, it is why urgency escalation exists (see above) -- the real
//! system asserts `urgent` well before the egress FIFO underflows.
//!
//! # What is proven, and what is not
//!
//! `no_requester_starves_under_the_retry_contract` sweeps demand masks the
//! model really generates (1-, 2- and 3-bank), every demand period up to
//! 2*`NUM_REQUESTERS` and phase, 5 requester mixes, two rival cadences, and
//! every rotor skew -- with per-BANK sticky retry. No starvation; worst wait 6
//! cycles against the bound of 7. All 5 mixes stay WITHIN one priority class
//! (core-vs-core or DMA-vs-DMA): a plain round-robin sweep can no longer make
//! a cross-class "nobody starves" claim, since class priority makes that false
//! by design for a non-urgent DMA against an ever-present core. Cross-class
//! fairness is the urgency guarantee proven separately (above).
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
//! 4.00 stream beats: `docs/superpowers/findings/2026-07-14-dma-bank-access-width.md`).
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

use super::banking::COMPUTE_PHYSICAL_BANKS;

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

/// One round-robin arbiter per physical bank.
#[derive(Clone, Debug)]
pub struct BankArbiter {
    /// Per bank: the requester ORDINAL (a stable identity, never a
    /// demand-list position) that has priority next. Advances by exactly one
    /// position every time this bank is actually contended, regardless of
    /// who wins -- a steady rotation through the fixed set of possible
    /// requesters, so a requester that only shows up intermittently still
    /// gets its scheduled turn instead of being repeatedly skipped by
    /// whichever requester happened to be present most recently (AM020
    /// ch.2:166 anti-starvation guarantee).
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
    /// physical banks it needs this cycle. `urgent` is the set of DMA
    /// channels whose egress FIFO is near underflow this cycle -- they must
    /// be served or the stream starves.
    ///
    /// Grants at most one requester per contended bank. The winner is chosen
    /// by CLASS, not by a single flat rotor: hardware gives the compute core
    /// priority over the DMA, except an urgent DMA (FIFO near underflow)
    /// overrides the core so the stream does not starve. Within whichever
    /// class wins, priority still rotates by stable requester identity
    /// (`Requester::ordinal`), not by position in this cycle's `demands`
    /// slice. A requester denied ANY bank it needs is reported in `lost` --
    /// it must stall and retry the whole request next cycle. A requester
    /// that got every bank it needed is reported in `granted` -- its access
    /// completed; it must not re-request (see the module-level retry
    /// contract).
    pub fn arbitrate(&mut self, demands: &[(Requester, u16)], urgent: &[Requester]) -> Arbitration {
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

        // Urgency keyed by ordinal, computed once per call rather than
        // rescanning `urgent` for every contended bank.
        let mut is_urgent = [false; NUM_REQUESTERS];
        for r in urgent {
            is_urgent[r.ordinal()] = true;
        }

        let mut out = Arbitration::default();
        let mut denied = [0u16; NUM_REQUESTERS]; // ordinal -> mask of banks lost

        for bank in 0..COMPUTE_PHYSICAL_BANKS as usize {
            let bit = 1u16 << bank;
            // Who wants this bank, by stable ordinal (not demand-list index).
            let wanters: Vec<usize> = demands
                .iter()
                .filter(|(_, mask)| mask & bit != 0)
                .map(|(who, _)| who.ordinal())
                .collect();

            if wanters.len() < 2 {
                continue; // free, or uncontended -- granted, nothing to do
            }
            out.contended_banks |= bit;

            // Class selection: an urgent DMA (FIFO near underflow) outranks
            // everything, else the core outranks a non-urgent DMA. Core
            // ordinals are always < NUM_CORE_PORTS; anything else is a DMA
            // channel.
            let urgent_dma: Vec<usize> = wanters.iter().copied().filter(|&ord| is_urgent[ord]).collect();
            let core: Vec<usize> = wanters.iter().copied().filter(|&ord| ord < NUM_CORE_PORTS).collect();
            let winner_class = if !urgent_dma.is_empty() {
                &urgent_dma
            } else if !core.is_empty() {
                &core
            } else {
                &wanters
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
            // won -- see the `rotor` field doc for why this must not simply
            // jump to (winner + 1): jumping to the winner's position lets a
            // pair of continuously-present rivals trap the pointer between
            // themselves, phase-locking out an intermittent requester
            // forever. A steady tick guarantees every ordinal's turn comes
            // up on schedule.
            self.rotor[bank] = ((start + 1) % NUM_REQUESTERS) as u8;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_beats_nonurgent_dma_on_a_contended_bank() {
        let mut arb = BankArbiter::new();
        let a = arb.arbitrate(
            &[(Requester::Core(CorePort::LoadA), 1 << 0), (Requester::S2mm(0), 1 << 0)],
            &[], // no urgent DMA
        );
        assert_eq!(a.contended_banks, 1 << 0);
        assert!(a.lost.contains(&Requester::S2mm(0)), "the DMA loses to the core by class priority");
        assert!(!a.core_lost(), "the core wins");
    }

    #[test]
    fn urgent_dma_beats_the_core() {
        let mut arb = BankArbiter::new();
        let a = arb.arbitrate(
            &[(Requester::Core(CorePort::LoadA), 1 << 0), (Requester::Mm2s(0), 1 << 0)],
            &[Requester::Mm2s(0)], // MM2S is near FIFO underflow
        );
        assert!(a.lost.contains(&Requester::Core(CorePort::LoadA)), "urgent DMA forces the grant");
        assert!(a.core_lost());
    }

    #[test]
    fn two_urgent_dma_channels_rotate_and_the_core_still_wins_when_neither_is_urgent() {
        // Two DMA channels can be near FIFO underflow AT ONCE (e.g. both
        // denied long enough by the core to escalate). Both then outrank the
        // core, and within that urgent-DMA class the existing rotor rule
        // still applies -- neither is starved by the other, bounded by the
        // same NUM_REQUESTERS contended-cycle bound as any other class.
        // Interleaved with a cycle where NEITHER is urgent, the core resumes
        // winning immediately: urgency is a per-cycle override, not a
        // lasting demotion of the core.
        let mut arb = BankArbiter::new();
        let demands = [
            (Requester::Core(CorePort::LoadA), 1u16 << 0),
            (Requester::Mm2s(0), 1u16 << 0),
            (Requester::Mm2s(1), 1u16 << 0),
        ];
        let both_urgent = [Requester::Mm2s(0), Requester::Mm2s(1)];

        let mut since_win = [0u32; 2]; // [Mm2s(0), Mm2s(1)]
        for cycle in 0..(4 * NUM_REQUESTERS as u32) {
            let a = arb.arbitrate(&demands, &both_urgent);
            assert!(a.core_lost(), "cycle {cycle}: two urgent DMA channels must both outrank the core");

            for (i, who) in [Requester::Mm2s(0), Requester::Mm2s(1)].into_iter().enumerate() {
                if a.lost.contains(&who) {
                    since_win[i] += 1;
                } else {
                    since_win[i] = 0;
                }
                assert!(
                    since_win[i] <= NUM_REQUESTERS as u32,
                    "cycle {cycle}: {who:?} has gone {} contended cycles without winning against \
                     its urgent peer -- exceeds the within-class round-robin bound",
                    since_win[i]
                );
            }
        }

        // No DMA urgent this cycle: the core is not permanently demoted by
        // having lost to urgency -- it must win the very next non-urgent
        // cycle.
        let a = arb.arbitrate(&demands, &[]);
        assert!(!a.core_lost(), "with no DMA urgent, the core must win -- it is not starved across cycles");
    }

    #[test]
    fn two_core_ports_still_rotate_within_the_core_class() {
        // Core-vs-core is unchanged: exactly one core port wins, and across two
        // contended cycles the winner alternates (within-class round-robin).
        let mut arb = BankArbiter::new();
        let d =
            [(Requester::Core(CorePort::LoadA), 1u16 << 0), (Requester::Core(CorePort::Store), 1u16 << 0)];
        let first = arb.arbitrate(&d, &[]).lost;
        let second = arb.arbitrate(&d, &[]).lost;
        assert_eq!(first.len(), 1);
        assert_ne!(first, second, "within-core rotation still alternates the loser");
    }

    #[test]
    fn no_contention_everyone_wins() {
        let mut arb = BankArbiter::new();
        let a =
            arb.arbitrate(&[(Requester::Core(CorePort::LoadA), 1 << 0), (Requester::S2mm(0), 1 << 1)], &[]);
        assert!(a.lost.is_empty());
        assert_eq!(a.contended_banks, 0);
    }

    #[test]
    fn same_bank_collision_grants_exactly_one() {
        let mut arb = BankArbiter::new();
        let a =
            arb.arbitrate(&[(Requester::Core(CorePort::LoadA), 1 << 0), (Requester::S2mm(0), 1 << 0)], &[]);
        assert_eq!(a.lost.len(), 1, "exactly one requester loses a single bank");
        assert_eq!(a.contended_banks, 1 << 0);
    }

    #[test]
    fn core_beats_the_same_dma_every_cycle_without_urgency() {
        // Class priority replaces the old symmetric round-robin expectation:
        // the core does not merely tend to win, it wins EVERY contended
        // cycle against a non-urgent DMA, with no alternation at all. AM020's
        // "round-robin to avoid starving any requester" is still the anti-
        // starvation mechanism, but it now operates WITHIN a priority class
        // (see `two_core_ports_still_rotate_within_the_core_class`) plus
        // urgency escalation across classes (see
        // `dma_is_not_starved_because_urgency_escalates_before_underflow`),
        // not as a single flat rotor over every requester.
        let mut arb = BankArbiter::new();
        let demands = [(Requester::Core(CorePort::LoadA), 1 << 0), (Requester::S2mm(0), 1 << 0)];
        for cycle in 0..10 {
            let a = arb.arbitrate(&demands, &[]);
            assert!(!a.core_lost(), "cycle {cycle}: the core must win every contended cycle");
            assert!(a.lost.contains(&Requester::S2mm(0)), "cycle {cycle}: the DMA must lose every cycle");
        }
    }

    #[test]
    fn a_requester_losing_any_needed_bank_is_reported_lost() {
        let mut arb = BankArbiter::new();
        // Core needs banks 0 and 1; DMA contends only on bank 1.
        let a = arb.arbitrate(
            &[(Requester::Core(CorePort::LoadA), (1 << 0) | (1 << 1)), (Requester::S2mm(0), 1 << 1)],
            &[],
        );
        // Whoever loses bank 1 is reported; contention is only on bank 1.
        assert_eq!(a.contended_banks, 1 << 1);
        assert_eq!(a.lost.len(), 1);
    }

    #[test]
    fn dma_is_not_starved_because_urgency_escalates_before_underflow() {
        // AM020 ch.2:166's anti-starvation guarantee didn't disappear, it
        // MOVED: a non-urgent DMA losing every single cycle to the core
        // (see `core_beats_the_same_dma_every_cycle_without_urgency`) is not
        // "starvation" in the AM020 sense, because the real system asserts
        // `urgent` well before the egress FIFO actually underflows
        // (`DmaEngine::urgent_channels` / `transfer_is_urgent`) -- and an
        // urgent DMA overrides the core outright (`urgent_dma_beats_the_core`).
        // Model that lifecycle: the DMA loses while it can afford to wait,
        // then wins the instant urgency is asserted, then the core resumes
        // winning once the pressure is off -- urgency is a per-cycle
        // override, not a lasting demotion of the core.
        let mut arb = BankArbiter::new();
        let demands = [(Requester::Core(CorePort::LoadA), 1u16 << 0), (Requester::S2mm(0), 1u16 << 0)];

        for cycle in 0..5 {
            let a = arb.arbitrate(&demands, &[]);
            assert!(a.lost.contains(&Requester::S2mm(0)), "cycle {cycle}: DMA loses while it can wait");
        }

        let a = arb.arbitrate(&demands, &[Requester::S2mm(0)]);
        assert!(!a.lost.contains(&Requester::S2mm(0)), "urgency wins the grant before FIFO underflow");
        assert!(a.core_lost());

        let a = arb.arbitrate(&demands, &[]);
        assert!(!a.core_lost(), "the core resumes winning immediately once no DMA is urgent");
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
            arb.arbitrate(&[(Requester::Core(CorePort::Store), 1 << 0), (Requester::Mm2s(1), 1 << 0)], &[]);
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

            let a = arb.arbitrate(&demands, &[]);
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
        // All three parties are DMA, not the core: under class priority the
        // core would win any bank it contends UNCONDITIONALLY (see the module
        // docs), which would make this about cross-class priority instead of
        // the within-class rotor phase-lock the test exists to pin. The
        // within-class math is unaffected by class priority (a bank with no
        // core wanter and no urgent DMA falls through to the plain flat rotor,
        // same as before this arbiter had classes at all).
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
        // core did before `31cf1511`: the two rotors hold a relative phase in
        // which it never wins both banks in the same cycle while each single-bank
        // rival keeps winning its own.
        //
        // The fix is NOT to give the DMA a per-bank sticky retry. It is that a DMA
        // channel's memory side is ONE 128-bit port and can only ever demand ONE
        // granule -- hence one bank -- per cycle
        // (docs/superpowers/findings/2026-07-14-dma-bank-access-width.md;
        // `strided_s2mm_takes_one_granule_per_cycle`). A one-bank mask cannot be
        // partially served, so the whole-mask retry is trivially correct and this
        // starvation shape is structurally unreachable. This test pins WHY that
        // matters: if a multi-bank DMA demand ever comes back, so does this.
        // WHICH rivals and WHICH rotor skew starve a given victim depends on the
        // ordinals' relative positions in the rotor space, so this sweeps the
        // adversarial phases rather than hardcoding one -- the claim is that a
        // starving phase EXISTS for a real DMA channel against real rivals (two
        // other DMA channels, each hammering one of its banks), not that any
        // particular skew is magic. Both rivals are DMA, not a core port:
        // under class priority a core rival would win its bank UNCONDITIONALLY
        // (see the module docs), which would make the victim's non-starvation
        // hinge on cross-class urgency instead of on the within-class rotor
        // this test exists to pin. Starvation is checked as zero grants over
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
        // ingredient (as on the core, `31cf1511`).
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
        // All 5 mixes are now WITHIN one priority class (core-vs-core or
        // DMA-vs-DMA), never core-vs-DMA: class priority (module docs) makes a
        // continuously-present, non-urgent DMA lose every single cycle to the
        // core by DESIGN, so a plain "nobody starves" sweep can no longer
        // include that pairing without hard-coding a false claim. This sweep
        // proves what's still true unconditionally -- the round-robin retry
        // bound holds WITHIN a class, for every mask/period/phase/skew the
        // model generates. Cross-class fairness is a SEPARATE, urgency-gated
        // guarantee, proven by `core_beats_the_same_dma_every_cycle_without_urgency`,
        // `urgent_dma_beats_the_core`, `dma_is_not_starved_because_urgency_escalates_before_underflow`,
        // and `two_urgent_dma_channels_rotate_and_the_core_still_wins_when_neither_is_urgent`.
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
        let c0 = arb.arbitrate(&full, &[]);
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

        let c1 = arb.arbitrate(&retry, &[]);
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
        let d0 = [(Requester::Core(CorePort::LoadA), 1 << 0), (Requester::S2mm(0), 1 << 0)];
        let first = arb.arbitrate(&d0, &[]);

        // Bank 3 has never been touched. If its rotor is truly independent,
        // its very first contention must resolve exactly like bank 0's very
        // first contention did (both start from a fresh rotor). A shared
        // rotor would instead carry over bank 0's post-arbitration state and
        // flip the winner.
        let d3 = [(Requester::Core(CorePort::LoadA), 1 << 3), (Requester::S2mm(0), 1 << 3)];
        let a = arb.arbitrate(&d3, &[]);
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
            &[],
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
            &[],
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
            &[],
        );
        assert_eq!(a.contended_banks, 1 << 0);
        assert_eq!(a.lost.len(), 2, "two of the three ports must lose a single-port bank");
        assert!(a.core_lost());
    }
}
