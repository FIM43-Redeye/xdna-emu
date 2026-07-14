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

/// An agent that can request a data-memory bank in a given cycle.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Requester {
    /// The compute core's load/store ports (bundle granularity: a conflict on
    /// any port stalls the whole datapath, AM020 ch.4:69).
    Core,
    /// Stream-to-memory DMA channel.
    S2mm(u8),
    /// Memory-to-stream DMA channel.
    Mm2s(u8),
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

/// One round-robin arbiter per physical bank.
#[derive(Clone, Debug)]
pub struct BankArbiter {
    /// Per bank: index into the current cycle's demand list that has priority.
    /// Advances past the winner on every grant, so no requester starves.
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
    /// Grants at most one requester per contended bank, rotating priority. A
    /// requester denied ANY bank it needs is reported in `lost` -- it must stall
    /// and retry the whole request next cycle.
    pub fn arbitrate(&mut self, demands: &[(Requester, u16)]) -> Arbitration {
        let mut out = Arbitration::default();
        let mut denied = [false; 8]; // index into `demands`

        for bank in 0..COMPUTE_PHYSICAL_BANKS as usize {
            let bit = 1u16 << bank;
            // Who wants this bank?
            let wanters: Vec<usize> = demands
                .iter()
                .enumerate()
                .filter(|(_, (_, mask))| mask & bit != 0)
                .map(|(i, _)| i)
                .collect();

            if wanters.len() < 2 {
                continue; // free, or uncontended -- granted, nothing to do
            }
            out.contended_banks |= bit;

            // Round-robin: the first wanter at or after the rotor wins.
            let start = self.rotor[bank] as usize;
            let winner = *wanters.iter().find(|&&i| i >= start).unwrap_or(&wanters[0]);

            for &i in &wanters {
                if i != winner && i < denied.len() {
                    denied[i] = true;
                }
            }
            // Advance past the winner so a different requester leads next time.
            self.rotor[bank] = ((winner + 1) % demands.len().max(1)) as u8;
        }

        for (i, (who, _)) in demands.iter().enumerate() {
            if i < denied.len() && denied[i] {
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
        let a = arb.arbitrate(&[(Requester::Core, 1 << 0), (Requester::S2mm(0), 1 << 1)]);
        assert!(a.lost.is_empty());
        assert_eq!(a.contended_banks, 0);
    }

    #[test]
    fn same_bank_collision_grants_exactly_one() {
        let mut arb = BankArbiter::new();
        let a = arb.arbitrate(&[(Requester::Core, 1 << 0), (Requester::S2mm(0), 1 << 0)]);
        assert_eq!(a.lost.len(), 1, "exactly one requester loses a single bank");
        assert_eq!(a.contended_banks, 1 << 0);
    }

    #[test]
    fn round_robin_alternates_the_winner() {
        // AM020: "round-robin to avoid starving any requester"
        let mut arb = BankArbiter::new();
        let demands = [(Requester::Core, 1 << 0), (Requester::S2mm(0), 1 << 0)];
        let first = arb.arbitrate(&demands).lost;
        let second = arb.arbitrate(&demands).lost;
        assert_ne!(first, second, "the same requester must not lose twice in a row");
    }

    #[test]
    fn a_requester_losing_any_needed_bank_is_reported_lost() {
        let mut arb = BankArbiter::new();
        // Core needs banks 0 and 1; DMA contends only on bank 1.
        let a = arb.arbitrate(&[(Requester::Core, (1 << 0) | (1 << 1)), (Requester::S2mm(0), 1 << 1)]);
        // Whoever loses bank 1 is reported; contention is only on bank 1.
        assert_eq!(a.contended_banks, 1 << 1);
        assert_eq!(a.lost.len(), 1);
    }

    #[test]
    fn per_bank_arbiters_are_independent() {
        // Each bank has its OWN round-robin pointer (AM020 ch.2:166).
        let mut arb = BankArbiter::new();
        // Contend only bank 0 twice; bank 3's pointer must be untouched.
        let d0 = [(Requester::Core, 1 << 0), (Requester::S2mm(0), 1 << 0)];
        arb.arbitrate(&d0);
        arb.arbitrate(&d0);
        let d3 = [(Requester::Core, 1 << 3), (Requester::S2mm(0), 1 << 3)];
        let a = arb.arbitrate(&d3);
        assert_eq!(a.contended_banks, 1 << 3);
        assert_eq!(a.lost.len(), 1);
    }
}
