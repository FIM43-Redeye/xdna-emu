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
const NUM_DMA_CHANNELS: usize = xdna_archspec::aie2::compute::NUM_DMA_CHANNELS as usize;

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
    /// Requesters that were denied at least one bank they asked for. They must
    /// hold their request and retry next cycle.
    pub lost: Vec<Requester>,
    /// Bitmask of banks that had more than one requester this cycle. Drives the
    /// CONFLICT_DM_BANK_n trace events.
    pub contended_banks: u16,
}

impl Arbitration {
    /// Did the core lose arbitration on ANY of its ports this cycle?
    ///
    /// AM020 ch.4:69: a bank conflict on any port stalls the WHOLE datapath,
    /// so even though the three core ports arbitrate individually (each can
    /// collide with a different rival, or with each other), the caller only
    /// ever needs the bundle-granularity answer: did the core, as a whole,
    /// fail to issue this cycle? Callers (the per-cycle coordinator) must use
    /// this rather than re-deriving it by scanning `lost` for `Requester::Core`
    /// variants themselves.
    pub fn core_lost(&self) -> bool {
        self.lost.iter().any(|r| matches!(r, Requester::Core(_)))
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
    /// cycle.
    pub fn arbitrate(&mut self, demands: &[(Requester, u16)]) -> Arbitration {
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
