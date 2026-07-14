//! Per-physical-bank round-robin memory arbiter.
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
//! # The retry contract (load-bearing invariant)
//!
//! AM020's two sentences are ONE design: round-robin priority is the
//! anti-starvation mechanism only because "the other requesters are stalled
//! for one cycle and the hardware retries the memory request in the next
//! cycle." This arbiter's starvation-freedom therefore DEPENDS ON its caller
//! honouring that contract:
//!
//! * **Every loser re-presents the SAME demand on the very next cycle.** A
//!   requester that loses is stalled -- it does not withdraw, skip ahead, or
//!   come back later. It holds its request until granted.
//! * **A winner's access COMPLETED.** It latched its result and must NOT
//!   re-request while the rest of its bundle/burst is still stalled (sticky
//!   grants -- see `Arbitration::granted`).
//!
//! Under that contract a pending requester re-asks on every cycle while the
//! rotor sweeps, so it is granted within at most `NUM_REQUESTERS` contended
//! cycles -- no aliasing hole, no starvation, proven by
//! `no_requester_starves_under_the_retry_contract` over every demand period
//! and phase. Break the contract (let a loser withdraw and return on a period
//! that aliases the rotor) and that guarantee is void.

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
    /// hold their request and retry it next cycle.
    pub lost: Vec<Requester>,
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

    /// Core ports granted this cycle -- accumulate these into the served set
    /// while a bundle is stalled.
    pub fn granted_core_ports(&self) -> impl Iterator<Item = CorePort> + '_ {
        self.granted.iter().filter_map(|r| match r {
            Requester::Core(p) => Some(*p),
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
    /// physical banks it needs this cycle.
    ///
    /// Grants at most one requester per contended bank, rotating priority by
    /// stable requester identity (`Requester::ordinal`), not by position in
    /// this cycle's `demands` slice. A requester denied ANY bank it needs is
    /// reported in `lost` -- it must stall and retry the whole request next
    /// cycle. A requester that got every bank it needed is reported in
    /// `granted` -- its access completed; it must not re-request (see the
    /// module-level retry contract).
    pub fn arbitrate(&mut self, demands: &[(Requester, u16)]) -> Arbitration {
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
        let mut denied = [false; NUM_REQUESTERS]; // indexed by requester ordinal

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

            // Round-robin: the wanter whose ordinal is first at-or-after the
            // rotor wins, wrapping around the fixed requester space.
            let start = self.rotor[bank] as usize;
            let winner_ordinal = *wanters
                .iter()
                .min_by_key(|&&ord| (ord + NUM_REQUESTERS - start) % NUM_REQUESTERS)
                .expect("wanters has at least 2 elements (checked above)");

            for &ord in &wanters {
                if ord != winner_ordinal {
                    denied[ord] = true;
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

        for (who, _) in demands.iter() {
            if denied[who.ordinal()] {
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
    fn no_contention_everyone_wins() {
        let mut arb = BankArbiter::new();
        let a = arb.arbitrate(&[(Requester::Core(CorePort::LoadA), 1 << 0), (Requester::S2mm(0), 1 << 1)]);
        assert!(a.lost.is_empty());
        assert_eq!(a.contended_banks, 0);
    }

    #[test]
    fn same_bank_collision_grants_exactly_one() {
        let mut arb = BankArbiter::new();
        let a = arb.arbitrate(&[(Requester::Core(CorePort::LoadA), 1 << 0), (Requester::S2mm(0), 1 << 0)]);
        assert_eq!(a.lost.len(), 1, "exactly one requester loses a single bank");
        assert_eq!(a.contended_banks, 1 << 0);
    }

    #[test]
    fn round_robin_alternates_the_winner() {
        // AM020: "round-robin to avoid starving any requester"
        let mut arb = BankArbiter::new();
        let demands = [(Requester::Core(CorePort::LoadA), 1 << 0), (Requester::S2mm(0), 1 << 0)];
        let first = arb.arbitrate(&demands).lost;
        let second = arb.arbitrate(&demands).lost;
        assert_ne!(first, second, "the same requester must not lose twice in a row");
    }

    #[test]
    fn a_requester_losing_any_needed_bank_is_reported_lost() {
        let mut arb = BankArbiter::new();
        // Core needs banks 0 and 1; DMA contends only on bank 1.
        let a = arb.arbitrate(&[
            (Requester::Core(CorePort::LoadA), (1 << 0) | (1 << 1)),
            (Requester::S2mm(0), 1 << 1),
        ]);
        // Whoever loses bank 1 is reported; contention is only on bank 1.
        assert_eq!(a.contended_banks, 1 << 1);
        assert_eq!(a.lost.len(), 1);
    }

    #[test]
    fn core_is_not_starved_by_continuous_dma_contention() {
        // AM020 ch.2:166 anti-starvation guarantee: "round-robin to avoid
        // starving any requester." Reproduces the reviewer's exact scenario:
        // two DMA channels hammer bank 0 every cycle (steady burst traffic)
        // while the Core only issues a memory op every OTHER cycle
        // (realistic -- not every cycle is a load/store). A rotor keyed by
        // demand-list INDEX instead of stable requester IDENTITY makes the
        // Core lose every single contention it participates in; this must
        // not happen.
        let mut arb = BankArbiter::new();
        let mut core_contentions = 0u32;
        let mut core_wins = 0u32;

        for cycle in 0..20u32 {
            let mut demands = vec![(Requester::S2mm(0), 1u16 << 0), (Requester::Mm2s(0), 1u16 << 0)];
            let core_present = cycle % 2 == 0;
            if core_present {
                demands.push((Requester::Core(CorePort::LoadA), 1u16 << 0));
            }

            let a = arb.arbitrate(&demands);

            if core_present {
                core_contentions += 1;
                if !a.lost.contains(&Requester::Core(CorePort::LoadA)) {
                    core_wins += 1;
                }
            }
        }

        assert_eq!(core_contentions, 10);
        assert!(core_wins > 0, "Core must not be totally starved (AM020 anti-starvation guarantee)");
        assert!(
            core_wins * 4 >= core_contentions,
            "Core should win at least a quarter of its contended cycles under fair round-robin; got {core_wins}/{core_contentions}"
        );
    }

    /// Drives the arbiter over `CYCLES` cycles with a FAITHFUL retry model
    /// (AM020 ch.2:166): every requester holds a pending request until it is
    /// GRANTED -- a loser is stalled, it does not withdraw and come back on
    /// its next natural period. Each requester issues a fresh request on
    /// cycles where `t % period == phase` (coalesced if one is still pending).
    /// All requesters target one bank -- the worst case.
    ///
    /// Returns (grants, worst wait in cycles from request-issue to grant).
    fn simulate_retry_contract(reqs: &[(Requester, usize, usize)]) -> (Vec<u32>, u64) {
        const CYCLES: u64 = 400;
        const BANK: u16 = 1 << 0;

        let mut arb = BankArbiter::new();
        let mut pending: Vec<Option<u64>> = vec![None; reqs.len()];
        let mut grants = vec![0u32; reqs.len()];
        let mut worst_wait = 0u64;

        for t in 0..CYCLES {
            for (i, (_, period, phase)) in reqs.iter().enumerate() {
                if pending[i].is_none() && (t as usize) % period == *phase {
                    pending[i] = Some(t);
                }
            }
            let demands: Vec<(Requester, u16)> = reqs
                .iter()
                .enumerate()
                .filter(|(i, _)| pending[*i].is_some())
                .map(|(_, (who, _, _))| (*who, BANK))
                .collect();
            if demands.is_empty() {
                continue;
            }

            let a = arb.arbitrate(&demands);
            for who in &a.granted {
                let i = reqs.iter().position(|(r, _, _)| r == who).unwrap();
                let issued = pending[i].expect("granted requester must have been pending");
                worst_wait = worst_wait.max(t - issued);
                grants[i] += 1;
                pending[i] = None; // access completed; next natural request may issue
            }
            // Losers keep their pending request: they re-present it next cycle.
        }
        (grants, worst_wait)
    }

    #[test]
    fn no_requester_starves_under_the_retry_contract() {
        // Settles the disputed "rotor aliasing" starvation claim EMPIRICALLY.
        //
        // The claim: with NUM_REQUESTERS == 7 and a +1-tick rotor, a requester
        // whose demand period is a multiple of 7 always observes the same
        // rotor value and can lose 100% of its contentions. That simulation
        // modelled a requester that WITHDRAWS after losing -- which contradicts
        // AM020 ch.2:166 ("the other requesters are stalled for one cycle and
        // the hardware retries the memory request in the next cycle").
        //
        // Under the real contract -- every loser re-presents the same request
        // on the very next cycle -- a pending requester re-asks while the rotor
        // sweeps, so the rotor must reach its ordinal within NUM_REQUESTERS
        // contended cycles. Round-robin (AM020's word) PLUS retry (AM020's next
        // sentence) IS the anti-starvation mechanism; they are one design.
        //
        // Sweep: every demand period 1..=2*NUM_REQUESTERS (covers the alleged
        // period-7 aliasing hole and its harmonics) x every phase offset x
        // four requester mixes x two rival cadences (continuous, and rivals
        // *also* aliased to the rotor modulus).
        let victim_mixes: [(Requester, Vec<Requester>); 4] = [
            // core vs 1 DMA
            (Requester::Core(CorePort::LoadA), vec![Requester::S2mm(0)]),
            // core vs 2 DMA
            (Requester::Core(CorePort::LoadA), vec![Requester::S2mm(0), Requester::Mm2s(0)]),
            // core vs 4 DMA (every DMA channel on the tile)
            (
                Requester::Core(CorePort::LoadA),
                vec![Requester::S2mm(0), Requester::S2mm(1), Requester::Mm2s(0), Requester::Mm2s(1)],
            ),
            // DMA vs DMA (the core is not special)
            (Requester::S2mm(1), vec![Requester::Mm2s(0), Requester::Mm2s(1)]),
        ];

        let mut worst_overall = 0u64;
        for (victim, rivals) in &victim_mixes {
            for period in 1..=(2 * NUM_REQUESTERS) {
                for phase in 0..period {
                    for rival_period in [1usize, NUM_REQUESTERS] {
                        let mut reqs = vec![(*victim, period, phase)];
                        reqs.extend(rivals.iter().map(|r| (*r, rival_period, 0)));

                        let (grants, worst_wait) = simulate_retry_contract(&reqs);

                        for (i, (who, _, _)) in reqs.iter().enumerate() {
                            assert!(
                                grants[i] > 0,
                                "{who:?} STARVED (0 grants) with victim {victim:?} \
                                 period={period} phase={phase} rival_period={rival_period}"
                            );
                        }
                        assert!(
                            worst_wait <= NUM_REQUESTERS as u64,
                            "wait {worst_wait} exceeds the round-robin bound of \
                             NUM_REQUESTERS ({NUM_REQUESTERS}) contended cycles: victim \
                             {victim:?} period={period} phase={phase} rival_period={rival_period}"
                        );
                        worst_overall = worst_overall.max(worst_wait);
                    }
                }
            }
        }
        // Measured worst case across the whole sweep: strictly bounded, and
        // well under the NUM_REQUESTERS ceiling (only 5 of 7 ordinals are ever
        // live in these mixes).
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
        let c0 = arb.arbitrate(&full);
        assert_eq!(c0.contended_banks, 1 << 0);
        assert_eq!(c0.granted.len(), 1, "a single-port bank grants exactly one port");
        assert_eq!(c0.lost.len(), 1);
        assert!(c0.core_lost(), "bundle stalls (AM020 ch.4:69: any port conflict stalls the datapath)");

        // Cycle 1: the caller honours the sticky grant -- only the UNSERVED
        // port re-requests.
        let served: Vec<CorePort> = c0.granted_core_ports().collect();
        assert_eq!(served.len(), 1);
        let retry: Vec<(Requester, u16)> = full
            .iter()
            .copied()
            .filter(|(r, _)| !matches!(r, Requester::Core(p) if served.contains(p)))
            .collect();
        assert_eq!(retry.len(), 1, "only the unserved port re-presents");

        let c1 = arb.arbitrate(&retry);
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
        let first = arb.arbitrate(&d0);

        // Bank 3 has never been touched. If its rotor is truly independent,
        // its very first contention must resolve exactly like bank 0's very
        // first contention did (both start from a fresh rotor). A shared
        // rotor would instead carry over bank 0's post-arbitration state and
        // flip the winner.
        let d3 = [(Requester::Core(CorePort::LoadA), 1 << 3), (Requester::S2mm(0), 1 << 3)];
        let a = arb.arbitrate(&d3);
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
        let a = arb.arbitrate(&[
            (Requester::Core(CorePort::LoadA), 1 << 0),
            (Requester::Core(CorePort::Store), 1 << 0),
        ]);
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
        let a = arb.arbitrate(&[
            (Requester::Core(CorePort::LoadA), 1 << 0),
            (Requester::Core(CorePort::Store), 1 << 1),
        ]);
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
        let a = arb.arbitrate(&[
            (Requester::Core(CorePort::LoadA), 1 << 0),
            (Requester::Core(CorePort::LoadB), 1 << 0),
            (Requester::Core(CorePort::Store), 1 << 0),
        ]);
        assert_eq!(a.contended_banks, 1 << 0);
        assert_eq!(a.lost.len(), 2, "two of the three ports must lose a single-port bank");
        assert!(a.core_lost());
    }
}
