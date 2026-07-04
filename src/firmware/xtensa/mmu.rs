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

/// Autorefill way size (entries per autorefill way 0-3) and its ×4
/// `nrefillentries`. MMU-v3 default (`overlay_tool.h` `TLB_TEMPLATE` with
/// `XCHAL_*TLB_ARF_ENTRIES_LOG2`). Confirmed against the QEMU core configs:
/// every MMU-enabled core (`core-dc232b`, `core-dc233c`, `core-fsf`,
/// `core-test_mmuhifi_c3`, `core-test_kc705_be`, `core-de233_fpu`) defines
/// `XCHAL_ITLB_ARF_ENTRIES_LOG2` / `XCHAL_DTLB_ARF_ENTRIES_LOG2` == 2, i.e.
/// way size 4, nrefillentries 16, `is32=false` -- no config disagrees.
pub const AUTOREFILL_WAY_SIZE: usize = 4;
const NREFILLENTRIES: u32 = 16;

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

    /// Entries actually addressable in way `wi` (`overlay_tool.h` `way_size[]`,
    /// varway56=false): ways 0-3 = AUTOREFILL_WAY_SIZE, way 4 = 4, ways 5/6 = 2,
    /// ways 7-9 (DTLB) = 1.
    fn way_size(&self, dtlb: bool, wi: usize) -> usize {
        let nways = if dtlb { DTLB_NWAYS } else { ITLB_NWAYS };
        debug_assert!(wi < nways);
        match wi {
            0..=3 => AUTOREFILL_WAY_SIZE,
            4 => 4,
            5 | 6 => 2,
            _ => 1, // ways 7-9, DTLB only
        }
    }

    /// Page-size selector for variable ways 4/5/6 from ITLBCFG/DTLBCFG
    /// (`mmu_helper.c:86-104`). Ways 0-3 and 7-9 return 0 (fixed).
    fn get_page_size(&self, dtlb: bool, wi: usize) -> u32 {
        let cfg = if dtlb { self.dtlbcfg } else { self.itlbcfg };
        match wi {
            4 => (cfg >> 16) & 0x3,
            5 => (cfg >> 20) & 0x1,
            6 => (cfg >> 24) & 0x1,
            _ => 0,
        }
    }

    /// VPN/page mask for way `wi` (`xtensa_tlb_get_addr_mask`,
    /// `mmu_helper.c:109-141`, varway56=false else-arms).
    fn addr_mask(&self, dtlb: bool, wi: usize) -> u32 {
        match wi {
            4 => 0xfff00000u32 << (self.get_page_size(dtlb, wi) * 2),
            5 => 0xf8000000,
            6 => 0xf0000000,
            _ => 0xfffff000, // ways 0-3, 7-9
        }
    }

    /// Split a vaddr into (VPN, entry-index) for way `wi`
    /// (`split_tlb_entry_spec_way`, `mmu_helper.c:176-226`, varway56=false).
    fn split_entry(&self, vaddr: u32, dtlb: bool, wi: usize) -> (u32, usize) {
        let ei = if wi < 4 {
            let is32 = NREFILLENTRIES == 32;
            ((vaddr >> 12) & if is32 { 0x7 } else { 0x3 }) as usize
        } else {
            match wi {
                4 => {
                    let eibase = 20 + self.get_page_size(dtlb, wi) * 2;
                    ((vaddr >> eibase) & 0x3) as usize
                }
                5 => ((vaddr >> 27) & 0x1) as usize,
                6 => ((vaddr >> 28) & 0x1) as usize,
                _ => 0,
            }
        };
        let vpn = vaddr & self.addr_mask(dtlb, wi);
        (vpn, ei)
    }

    /// Ring (0-3) whose RASID byte lane equals `asid`, else 0xff
    /// (`get_ring`, `mmu_helper.c:443-452`).
    fn get_ring(&self, asid: u8) -> u32 {
        for i in 0..4 {
            if ((self.rasid >> (i * 8)) & 0xff) as u8 == asid {
                return i;
            }
        }
        0xff
    }

    /// Per-way direct-mapped TLB lookup (`xtensa_tlb_lookup`,
    /// `mmu_helper.c:463-495`). A slot hits iff its stored VPN equals the
    /// vaddr's VPN for that way, `asid != 0` (populated), and `get_ring(asid)
    /// < 4` (the ASID is currently live in RASID). More than one hit is a
    /// multi-hit fault; zero hits is a miss.
    pub fn lookup(&self, vaddr: u32, dtlb: bool) -> Result<TlbHit, u32> {
        let nways = if dtlb { DTLB_NWAYS } else { ITLB_NWAYS };
        let mut hit: Option<TlbHit> = None;
        for wi in 0..nways {
            let (vpn, ei) = self.split_entry(vaddr, dtlb, wi);
            let entry = &(if dtlb { &self.dtlb[..] } else { &self.itlb[..] })[wi][ei];
            if entry.vaddr == vpn && entry.asid != 0 {
                let ring = self.get_ring(entry.asid);
                if ring < 4 {
                    if hit.is_some() {
                        return Err(if dtlb { 25 } else { 17 }); // *_TLB_MULTI_HIT
                    }
                    hit = Some(TlbHit { wi, ei, ring });
                }
            }
        }
        hit.ok_or(if dtlb { 24 } else { 16 }) // *_TLB_MISS
    }

    /// Install a TLB entry (`HELPER(wtlb)` + `split_tlb_entry_spec`,
    /// `mmu_helper.c:562-570, 232-249`). `as_` (the AS operand) encodes the way
    /// index in its low bits (ITLB &7 / DTLB &0xf) and the VPN in the rest;
    /// `at` (the AT operand) is the PTE-format value paddr|attr with the ring
    /// in bits [5:4]. Refuses to overwrite a fixed (`variable=false`) entry
    /// (`xtensa_tlb_set_entry`, `mmu_helper.c:290-317`).
    pub fn write_tlb(&mut self, dtlb: bool, at: u32, as_: u32) {
        let nways = if dtlb { DTLB_NWAYS } else { ITLB_NWAYS };
        let wi = (as_ & if dtlb { 0xf } else { 0x7 }) as usize;
        if wi >= nways {
            return; // invalid way index -> no-op (split_tlb_entry_spec else)
        }
        let (vpn, ei) = self.split_entry(as_, dtlb, wi);
        debug_assert!(ei < self.way_size(dtlb, wi), "TLB entry index out of way bounds");
        let ring = (at >> 4) & 0x3;
        let asid = ((self.rasid >> (ring * 8)) & 0xff) as u8;
        let entry = TlbEntry {
            vaddr: vpn,
            paddr: at & self.addr_mask(dtlb, wi),
            asid,
            attr: (at & 0xf) as u8,
            variable: true,
        };
        let slot = &mut (if dtlb {
            &mut self.dtlb[..]
        } else {
            &mut self.itlb[..]
        })[wi][ei];
        if !slot.variable {
            log::debug!("firmware mmu: witlb/wdtlb to fixed way {} entry {} ignored", wi, ei);
            return;
        }
        *slot = entry;
    }

    /// Invalidate a TLB entry (`HELPER(itlb)`, `mmu_helper.c:524-534`): decode
    /// the AS operand the same way, and if the target entry is variable and
    /// populated, mark it invalid by zeroing its ASID.
    pub fn invalidate_tlb(&mut self, dtlb: bool, as_: u32) {
        let nways = if dtlb { DTLB_NWAYS } else { ITLB_NWAYS };
        let wi = (as_ & if dtlb { 0xf } else { 0x7 }) as usize;
        if wi >= nways {
            return;
        }
        let (_vpn, ei) = self.split_entry(as_, dtlb, wi);
        debug_assert!(ei < self.way_size(dtlb, wi), "TLB entry index out of way bounds");
        let slot = &mut (if dtlb {
            &mut self.dtlb[..]
        } else {
            &mut self.itlb[..]
        })[wi][ei];
        if slot.variable && slot.asid != 0 {
            slot.asid = 0;
        }
    }

    /// Write RASID, forcing the ring-0 ASID byte to 1 (`wsr_rasid`,
    /// `mmu_helper.c:77-84`).
    pub fn write_rasid(&mut self, v: u32) {
        self.rasid = (v & 0xffffff00) | 0x1;
    }
}

/// Access permission bits (subset of QEMU's PAGE_* we model -- cache-policy
/// bits are behaviorally inert in this interpreter).
pub const PAGE_READ: u32 = 1;
pub const PAGE_WRITE: u32 = 2;
pub const PAGE_EXEC: u32 = 4;

/// Decode the 4-bit page/PTE attribute nibble into R/W/X permissions
/// (`mmu_attr_to_access`, `mmu_helper.c:576-606`). attr<12: READ always,
/// +EXEC if bit0, +WRITE if bit1 (cache policy bits[3:2] ignored here);
/// attr==13: RW isolate; 12/14/15: no access.
pub fn attr_to_access(attr: u8) -> u32 {
    if attr < 12 {
        let mut a = PAGE_READ;
        if attr & 0x1 != 0 {
            a |= PAGE_EXEC;
        }
        if attr & 0x2 != 0 {
            a |= PAGE_WRITE;
        }
        a
    } else if attr == 13 {
        PAGE_READ | PAGE_WRITE
    } else {
        0
    }
}

/// True if `access` grants the operation (`is_access_granted`,
/// `mmu_helper.c:852-861`). is_write: 0=load(READ), 1=store(WRITE), 2=fetch(EXEC).
pub fn access_granted(access: u32, is_write: u8) -> bool {
    let need = match is_write {
        1 => PAGE_WRITE,
        2 => PAGE_EXEC,
        _ => PAGE_READ,
    };
    access & need != 0
}

/// A resolved TLB hit: which way/entry, and the ring (0-3) the page belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TlbHit {
    pub wi: usize,
    pub ei: usize,
    pub ring: u32,
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

    #[test]
    fn autorefill_ways_are_fixed_4k() {
        // Ways 0-3: mask 0xfffff000 (4 KB), ei = bits[13:12] or [14:12] of vaddr
        // depending on nrefillentries (is32). split_tlb_entry_spec_way default arm.
        let mmu = Mmu::new();
        for wi in 0..4 {
            assert_eq!(mmu.addr_mask(false, wi), 0xfffff000);
            let (vpn, _ei) = mmu.split_entry(0x2000_0340, false, wi);
            assert_eq!(vpn, 0x2000_0000, "VPN is 4KB-aligned for autorefill ways");
        }
        // Entry index selects the set from vaddr bits above the 4KB page.
        let (_v, ei) = mmu.split_entry(0x2000_3340, false, 0);
        assert_eq!(ei, (0x2000_3340u32 >> 12) as usize & (AUTOREFILL_WAY_SIZE - 1));
    }

    #[test]
    fn way4_page_size_follows_cfg() {
        // get_page_size way 4 = ITLBCFG/DTLBCFG bits[17:16]; addr_mask way4 =
        // 0xfff00000 << (page_size*2). mmu_helper.c:118, 90.
        let mut mmu = Mmu::new();
        assert_eq!(mmu.get_page_size(false, 4), 0);
        assert_eq!(mmu.addr_mask(false, 4), 0xfff00000);
        mmu.itlbcfg = 0x2 << 16; // page_size selector = 2
        assert_eq!(mmu.get_page_size(false, 4), 2);
        assert_eq!(mmu.addr_mask(false, 4), 0xfff00000u32 << 4);
        // DTLB uses DTLBCFG independently.
        mmu.dtlbcfg = 0x1 << 16;
        assert_eq!(mmu.get_page_size(true, 4), 1);
    }

    #[test]
    fn ways_5_6_masks_varway56_false() {
        // varway56=false: way5 mask 0xf8000000, way6 mask 0xf0000000
        // (mmu_helper.c:124-127 else-arms). ei from vaddr bits [27]/[28].
        let mmu = Mmu::new();
        assert_eq!(mmu.addr_mask(false, 5), 0xf8000000);
        assert_eq!(mmu.addr_mask(false, 6), 0xf0000000);
        let (_v, ei5) = mmu.split_entry(0xd8000000, false, 5);
        assert_eq!(ei5, 1); // bit 27 of 0xd8000000 is set
        let (_v, ei6) = mmu.split_entry(0xf0000000, false, 6);
        assert_eq!(ei6, 1); // bit 28 of 0xf0000000 is set
    }

    #[test]
    fn get_ring_maps_asid_through_rasid_lanes() {
        // RASID default 0x04030201: ring0->1, ring1->2, ring2->3, ring3->4.
        let mmu = Mmu::new();
        assert_eq!(mmu.get_ring(1), 0);
        assert_eq!(mmu.get_ring(4), 3);
        assert_eq!(mmu.get_ring(9), 0xff); // not present in any lane
    }

    #[test]
    fn lookup_hits_installed_entry() {
        let mut mmu = Mmu::new();
        // Install into DTLB way 0, the entry index split_entry picks for this vaddr.
        let (vpn, ei) = mmu.split_entry(0x1000_5000, true, 0);
        mmu.dtlb[0][ei] = TlbEntry { vaddr: vpn, paddr: 0x0008_0000, asid: 1, attr: 3, variable: true };
        let hit = mmu.lookup(0x1000_5000, true).expect("should hit");
        assert_eq!(hit.wi, 0);
        assert_eq!(hit.ei, ei);
        assert_eq!(hit.ring, 0);
    }

    #[test]
    fn lookup_misses_uncovered_vaddr() {
        let mmu = Mmu::new();
        // Nothing maps 0x2000_0340; fixed ways cover only 0xd..-0xf.. .
        assert_eq!(mmu.lookup(0x2000_0340, false), Err(16)); // INST_TLB_MISS
        assert_eq!(mmu.lookup(0x2000_0340, true), Err(24)); // LOAD_STORE_TLB_MISS
    }

    #[test]
    fn lookup_reports_multi_hit() {
        let mut mmu = Mmu::new();
        // Force two different ways to both cover one vaddr (ways 0 and 1, same
        // autorefill addressing) -> multi-hit.
        let (vpn, ei) = mmu.split_entry(0x3000_0000, true, 0);
        let e = TlbEntry { vaddr: vpn, paddr: 0, asid: 1, attr: 3, variable: true };
        mmu.dtlb[0][ei] = e;
        mmu.dtlb[1][ei] = e;
        assert_eq!(mmu.lookup(0x3000_0000, true), Err(25)); // LOAD_STORE_TLB_MULTI_HIT
    }

    #[test]
    fn rewriting_rasid_invalidates_entries_by_context() {
        // An entry whose asid no longer appears in any RASID lane becomes
        // unreachable (get_ring -> 0xff) without being cleared (mmu_helper.c
        // get_ring semantics). Use asid=4 (ring3 by default), then blank ring3.
        let mut mmu = Mmu::new();
        let (vpn, ei) = mmu.split_entry(0x1000_0000, true, 0);
        mmu.dtlb[0][ei] = TlbEntry { vaddr: vpn, paddr: 0, asid: 4, attr: 3, variable: true };
        assert!(mmu.lookup(0x1000_0000, true).is_ok());
        // Overwrite ring3's lane (byte 3) so asid 4 is no longer present.
        mmu.rasid = 0x00030201;
        assert_eq!(mmu.lookup(0x1000_0000, true), Err(24)); // now a miss
    }

    // -- M2b Task 4: witlb/wdtlb/iitlb/idtlb + RASID write ----------------

    #[test]
    fn write_tlb_installs_and_reads_back() {
        let mut mmu = Mmu::new();
        // AS: way index in low bits (DTLB &0xf) + VPN in the high bits.
        // Install DTLB way 0 mapping VPN 0x40001000 -> paddr 0x00090000, attr 3.
        let as_ = 0x4000_1000 | 0; // way 0
        let at = 0x0009_0000 | 0x3; // paddr|attr, ring bits [5:4]=0
        mmu.write_tlb(true, at, as_);
        let hit = mmu.lookup(0x4000_1abc, true).expect("installed");
        assert_eq!(hit.wi, 0);
        let e = mmu.dtlb[hit.wi][hit.ei];
        assert_eq!(e.paddr, 0x0009_0000);
        assert_eq!(e.attr, 3);
        assert_eq!(e.asid, 1); // RASID lane for ring 0
    }

    #[test]
    fn write_tlb_refuses_fixed_entry() {
        let mut mmu = Mmu::new();
        // Way 6 entry 0 is the fixed 0xe0000000 mapping (variable=false).
        let before = mmu.itlb[6][0];
        // AS targets way 6 with a VPN colliding the fixed entry index.
        let as_ = 0xe000_0000 | 6;
        mmu.write_tlb(false, 0x1234_0007, as_);
        assert_eq!(mmu.itlb[6][0], before, "fixed entry must not be overwritten");
    }

    #[test]
    fn invalidate_tlb_clears_installed_entry() {
        let mut mmu = Mmu::new();
        let as_ = 0x4000_1000 | 0;
        mmu.write_tlb(true, 0x0009_0000 | 0x3, as_);
        assert!(mmu.lookup(0x4000_1abc, true).is_ok());
        mmu.invalidate_tlb(true, as_);
        assert_eq!(mmu.lookup(0x4000_1abc, true), Err(24));
    }

    #[test]
    fn write_rasid_forces_ring0_asid_to_one() {
        let mut mmu = Mmu::new();
        mmu.write_rasid(0x08070605);
        assert_eq!(mmu.rasid, 0x08070601, "ring-0 byte forced to 1 (mmu_helper.c:77)");
    }

    // -- M2b Task 5: attribute-nibble permission decode -------------------

    #[test]
    fn attr_decode_matches_isa_table() {
        // mmu_helper.c:576-606. attr<12: READ always; +EXEC if bit0; +WRITE if bit1.
        assert_eq!(attr_to_access(0), PAGE_READ);
        assert_eq!(attr_to_access(1), PAGE_READ | PAGE_EXEC);
        assert_eq!(attr_to_access(2), PAGE_READ | PAGE_WRITE);
        assert_eq!(attr_to_access(3), PAGE_READ | PAGE_WRITE | PAGE_EXEC);
        assert_eq!(attr_to_access(7), PAGE_READ | PAGE_WRITE | PAGE_EXEC); // cached RWX
        assert_eq!(attr_to_access(13), PAGE_READ | PAGE_WRITE); // isolate: RW, no exec
        assert_eq!(attr_to_access(12), 0); // reserved -> no access
        assert_eq!(attr_to_access(14), 0);
        assert_eq!(attr_to_access(15), 0);
    }

    #[test]
    fn access_granted_by_operation() {
        let rwx = PAGE_READ | PAGE_WRITE | PAGE_EXEC;
        assert!(access_granted(rwx, 0)); // load needs READ
        assert!(access_granted(rwx, 1)); // store needs WRITE
        assert!(access_granted(rwx, 2)); // fetch needs EXEC
        assert!(!access_granted(PAGE_READ, 1)); // store on read-only -> denied
        assert!(!access_granted(PAGE_READ | PAGE_WRITE, 2)); // fetch on no-exec -> denied
    }
}
