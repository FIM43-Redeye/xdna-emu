//! Xtensa MMU-v3 model: TLBs, config registers, and the autorefill page-table
//! walk. Derived from QEMU `target/xtensa/mmu_helper.c` (the authoritative
//! specification for the hardware behavior); this is our original
//! implementation of that mechanism. See
//! `docs/superpowers/specs/2026-07-04-m2b-mmu-mechanism-design.md`.

/// ITLB way count (MMU-v3: 7 ways). `overlay_tool.h` `ITLB()`.
pub const ITLB_NWAYS: usize = 7;
/// DTLB way count (MMU-v3: 10 ways). `overlay_tool.h` `DTLB()`.
pub const DTLB_NWAYS: usize = 10;
/// Max entries per way (`cpu.h:229` MAX_TLB_WAY_SIZE). Only `way_size[wi]`
/// entries of each way are actually addressable.
pub const MAX_TLB_WAY_SIZE: usize = 8;

/// One TLB entry (`cpu.h:313-320` xtensa_tlb_entry). `vaddr` is the stored VPN
/// (page-aligned virtual tag); `asid` is the concrete ASID byte (not the ring
/// number); `attr` is the 4-bit PTE attribute nibble; `variable=false` marks a
/// hard-wired entry that software (`witlb`/autorefill) may not overwrite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TlbEntry {
    pub vaddr: u32,
    pub paddr: u32,
    pub asid: u8,
    pub attr: u8,
    pub variable: bool,
}

/// The Xtensa MMU-v3 state: separate I/D TLBs, the config SRs, and the
/// round-robin autorefill way index.
pub struct Mmu {
    pub itlb: [[TlbEntry; MAX_TLB_WAY_SIZE]; ITLB_NWAYS],
    pub dtlb: [[TlbEntry; MAX_TLB_WAY_SIZE]; DTLB_NWAYS],
    pub ptevaddr: u32,
    pub rasid: u32,
    pub itlbcfg: u32,
    pub dtlbcfg: u32,
    pub autorefill_idx: u32,
}

impl Mmu {
    /// Reset state per `mmu_helper.c:413-441`: RASID=0x04030201, cfg=0, every
    /// entry invalid+variable, then ways 5/6 loaded with the fixed region
    /// entries (varway56=false).
    pub fn new() -> Self {
        // `variable=true`, everything else zero == an empty, software-writable
        // entry (asid==0 is the invalid marker used by `lookup`).
        let empty = TlbEntry { variable: true, ..TlbEntry::default() };
        let mut mmu = Self {
            itlb: [[empty; MAX_TLB_WAY_SIZE]; ITLB_NWAYS],
            dtlb: [[empty; MAX_TLB_WAY_SIZE]; DTLB_NWAYS],
            ptevaddr: 0,
            rasid: 0x04030201,
            itlbcfg: 0,
            dtlbcfg: 0,
            autorefill_idx: 0,
        };
        Self::load_fixed_ways56(&mut mmu.itlb);
        Self::load_fixed_ways56(&mut mmu.dtlb);
        mmu
    }

    /// Install the four hard-wired ways-5/6 entries (varway56=false path of
    /// `reset_tlb_mmu_ways56`, `mmu_helper.c:351-397`). Same for I and D TLBs.
    fn load_fixed_ways56(tlb: &mut [[TlbEntry; MAX_TLB_WAY_SIZE]]) {
        let fixed = |vaddr, paddr, attr| TlbEntry { vaddr, paddr, asid: 1, attr, variable: false };
        tlb[5][0] = fixed(0xd0000000, 0, 7);
        tlb[5][1] = fixed(0xd8000000, 0, 3);
        tlb[6][0] = fixed(0xe0000000, 0xf0000000, 7);
        tlb[6][1] = fixed(0xf0000000, 0xf0000000, 3);
    }
}

impl Default for Mmu {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_state_matches_mmu_v3_defaults() {
        // Derived from mmu_helper.c:413-441 (reset). RASID default 0x04030201,
        // cfg regs 0, autorefill_idx 0, all autorefill/way-4 entries invalid
        // (asid==0, variable==true).
        let mmu = Mmu::new();
        assert_eq!(mmu.rasid, 0x04030201);
        assert_eq!(mmu.itlbcfg, 0);
        assert_eq!(mmu.dtlbcfg, 0);
        assert_eq!(mmu.autorefill_idx, 0);
        // Autorefill ways start empty and software-writable.
        assert_eq!(mmu.itlb[0][0].asid, 0);
        assert!(mmu.itlb[0][0].variable);
        assert_eq!(mmu.dtlb[0][0].asid, 0);
    }

    #[test]
    fn ways_5_and_6_hold_fixed_region_entries() {
        // mmu_helper.c:351-397 reset_tlb_mmu_ways56 (varway56=false): ways 5/6
        // get two hard-wired variable=false entries each, for I and D TLBs.
        let mmu = Mmu::new();
        for tlb in [&mmu.itlb[..], &mmu.dtlb[..]] {
            assert_eq!(tlb[5][0].vaddr, 0xd0000000);
            assert_eq!(tlb[5][0].paddr, 0);
            assert_eq!(tlb[5][0].asid, 1);
            assert_eq!(tlb[5][0].attr, 7);
            assert!(!tlb[5][0].variable);

            assert_eq!(tlb[5][1].vaddr, 0xd8000000);
            assert_eq!(tlb[5][1].attr, 3);
            assert!(!tlb[5][1].variable);

            assert_eq!(tlb[6][0].vaddr, 0xe0000000);
            assert_eq!(tlb[6][0].paddr, 0xf0000000);
            assert_eq!(tlb[6][0].attr, 7);
            assert!(!tlb[6][0].variable);

            assert_eq!(tlb[6][1].vaddr, 0xf0000000);
            assert_eq!(tlb[6][1].paddr, 0xf0000000);
            assert_eq!(tlb[6][1].attr, 3);
            assert!(!tlb[6][1].variable);
        }
    }
}
