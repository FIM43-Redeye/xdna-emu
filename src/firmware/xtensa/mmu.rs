//! Xtensa MMU Option: instruction and data TLBs, hardware autorefill, the
//! TLB-management instructions and the MMU special registers.
//!
//! Written from the *Xtensa Instruction Set Architecture (ISA) Reference
//! Manual* (RC-2010.1, PD-09-0801-10-01). Citations read `[RM section,
//! p.N]` with N the printed page number. Where the manual is silent,
//! self-contradictory, or the target core is known to differ, the choice is
//! called out as a *platform choice* or *unspecified* and, where it is a
//! knob, lives in [`MmuConfig`].
//!
//! # Model
//!
//! The ITLB has 7 ways and the DTLB 10 [RM 4.6.5, p.159; 4.6.5.3,
//! p.163-164]. Ways 0-3 are the autorefill ways (4 KB pages); way 4 has a
//! page size chosen by xTLBCFG; ways 5 and 6 are the *variable* ways of
//! RC-2009.0 and later (software-written, size by xTLBCFG; way 6 is
//! pre-loaded at reset to identity-map all of memory); DTLB ways 7-9 hold a
//! single 4 KB entry each. The *fixed* form of ways 5/6 and the 8-entry
//! refill-way variant are not modelled: no supported core uses them.
//!
//! An entry has no valid bit; ASID 0 means invalid [RM 4.6.1.1, p.141].
//! The ring of an entry is not stored: it is the position of the entry's
//! ASID within RASID [RM 4.6.1.2, p.143].
//!
//! The MMU never writes EXCVADDR or EXCCAUSE. A failed translation returns a
//! [`TranslationFault`] and the CPU, which owns those registers, raises the
//! exception. The privilege check for the TLB instructions (PrivilegedCause
//! at CRING != 0) is likewise the CPU's job.

use super::bus::XtensaBus;

/// Which TLB [RM 4.6.2.1, p.147].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tlb {
    Instruction,
    Data,
}

/// Kind of memory access being translated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessKind {
    Fetch,
    Load,
    Store,
}

impl AccessKind {
    /// Fetches go through the ITLB, loads and stores through the DTLB.
    fn tlb(self) -> Tlb {
        match self {
            AccessKind::Fetch => Tlb::Instruction,
            AccessKind::Load | AccessKind::Store => Tlb::Data,
        }
    }
}

/// EXCCAUSE codes the MMU can produce [RM Table 4-64, p.89-91; Table 4-106,
/// p.159-160]. Codes 19, 21-23 and 27 are reserved.
pub mod cause {
    pub const INST_TLB_MISS: u32 = 16;
    pub const INST_TLB_MULTI_HIT: u32 = 17;
    pub const INST_FETCH_PRIVILEGE: u32 = 18;
    pub const INST_FETCH_PROHIBITED: u32 = 20;
    pub const LOAD_STORE_TLB_MISS: u32 = 24;
    pub const LOAD_STORE_TLB_MULTI_HIT: u32 = 25;
    pub const LOAD_STORE_PRIVILEGE: u32 = 26;
    pub const LOAD_PROHIBITED: u32 = 28;
    pub const STORE_PROHIBITED: u32 = 29;
}

/// How overlapping matches that involve way 6 are resolved.
///
/// The manual calls overlapping matches illegal; implementations "may"
/// detect them [RM 4.6.1.1, p.139] and the TLB-lookup description says the
/// condition "currently raises" a MultiHit exception [RM 4.6.2.2, p.147].
/// It gives no way a lower priority than another, so the yielding policy is a
/// platform choice, not a manual rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverlapPolicy {
    /// Any two matching entries raise the MultiHit cause [RM 4.6.2.2, p.147].
    Strict,
    /// A match in way 6 is used only when no other way matches; two or more
    /// matches among the other ways still raise MultiHit. Applies to normal
    /// lookups, the PTE-address lookup during refill, and probes.
    Way6Yields,
}

/// The per-core knobs of the MMU Option.
///
/// Everything else the manual lets a core vary (refill-way entry count,
/// fixed ways 5/6) is unsupported; see the module documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MmuConfig {
    /// Attribute written into every way-6 entry at reset, in both TLBs.
    ///
    /// The manual says 4'h2 ("Bypass") [RM 4.6.5.3, p.164], but by its own
    /// rights table (Table 4-109, p.178) attribute 2 forbids instruction
    /// fetch, which would make the first fetch after reset fault. Attribute
    /// 3 (the same bypass with fetch allowed) is the self-consistent reading.
    pub way6_reset_attribute: u8,
    /// Multi-hit resolution policy.
    pub overlap: OverlapPolicy,
}

impl MmuConfig {
    /// Manual behaviour wherever the manual is clear.
    pub const GENERIC: MmuConfig = MmuConfig { way6_reset_attribute: 3, overlap: OverlapPolicy::Strict };
}

impl Default for MmuConfig {
    fn default() -> Self {
        Self::GENERIC
    }
}

/// A successful translation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Translation {
    pub paddr: u32,
    /// Page size of the matching entry in bytes. A bulk operation uses it to
    /// stay inside one page.
    pub page_size: u32,
    /// Lookup ring: the RASID position of the matching entry's ASID.
    pub ring: u8,
    pub attribute: u8,
    pub way: u8,
    /// Entry index within `way`.
    pub slot: u8,
}

/// A translation fault for the CPU to raise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TranslationFault {
    /// EXCCAUSE value.
    pub cause: u32,
    /// Value for EXCVADDR: the virtual address being translated, never the
    /// address of the PTE that a failed refill tried to read.
    pub vaddr: u32,
}

/// One TLB entry, exposed read-only through [`Mmu::entry`] for diagnostics.
/// `asid == 0` is the invalid marker [RM 4.6.1.1, p.141].
///
/// `vaddr`/`paddr` hold the page bases at the page size in force when the
/// entry was written; lookups re-derive index and tag bits from the *current*
/// xTLBCFG size, so an entry left over from before a size change is matched
/// as-is (unspecified by the manual, which only says the way "should be
/// invalidated" when its size changes [RM 4.6.5.2, p.162]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TlbEntry {
    pub vaddr: u32,
    pub paddr: u32,
    pub asid: u8,
    pub attr: u8,
}

impl TlbEntry {
    const INVALID: TlbEntry = TlbEntry { vaddr: 0, paddr: 0, asid: 0, attr: 0 };
}

const ITLB_WAYS: usize = 7;
const DTLB_WAYS: usize = 10;
/// Most entries any way holds (way 6).
pub const MAX_WAY_ENTRIES: usize = 8;
/// Autorefill ways are 0..=3 [RM 4.6.5.3, p.163].
const REFILL_WAYS: u8 = 4;
const REFILL_PAGE_BITS: u32 = 12;
/// ASID the manual gives ring 0 and the way-6 reset entries [RM 4.6.1.2,
/// p.143; 4.6.5.3, p.164].
const RING0_ASID: u8 = 1;
const RASID_RESET: u32 = 0x0403_0201;

/// Where an access landed in the TLB: the matched way and entry slot.
#[derive(Clone, Copy)]
struct Hit {
    way: u8,
    slot: usize,
    entry: TlbEntry,
    /// RASID position of the entry's ASID.
    ring: u8,
}

/// The MMU: both TLBs, the MMU special registers, and the refill state.
#[derive(Clone, Debug)]
pub struct Mmu {
    config: MmuConfig,
    itlb: [[TlbEntry; MAX_WAY_ENTRIES]; ITLB_WAYS],
    dtlb: [[TlbEntry; MAX_WAY_ENTRIES]; DTLB_WAYS],
    /// Upper 10 bits of PTEVADDR [RM 4.6.5.2, Fig. 4-29, p.161].
    ptebase: u32,
    rasid: u32,
    itlbcfg: u32,
    dtlbcfg: u32,
    /// Next refill way to evict per TLB when all of 0..=3 are valid at the
    /// target slot. The manual does not say which way refill replaces
    /// [RM 4.6.5.9, p.174]; this policy (lowest invalid way first, then
    /// round-robin) is deterministic.
    refill_victim: [u8; 2],
}

impl Mmu {
    /// Reset state for `config` [RM 4.6.5.3, p.163-164; 4.6.5.2, p.161-163].
    pub fn new(config: MmuConfig) -> Self {
        let mut way6 = [TlbEntry::INVALID; MAX_WAY_ENTRIES];
        // 512 MB pages at xTLBCFG == 0: entry n maps [n * 512 MB, +512 MB)
        // onto itself in ring 0.
        for (n, e) in way6.iter_mut().enumerate() {
            let base = (n as u32) << 29;
            *e = TlbEntry {
                vaddr: base,
                paddr: base,
                asid: RING0_ASID,
                attr: config.way6_reset_attribute & 0xF,
            };
        }
        let mut itlb = [[TlbEntry::INVALID; MAX_WAY_ENTRIES]; ITLB_WAYS];
        let mut dtlb = [[TlbEntry::INVALID; MAX_WAY_ENTRIES]; DTLB_WAYS];
        itlb[6] = way6;
        dtlb[6] = way6;
        Mmu {
            config,
            itlb,
            dtlb,
            ptebase: 0,
            rasid: RASID_RESET,
            itlbcfg: 0,
            dtlbcfg: 0,
            refill_victim: [0; 2],
        }
    }

    // ------------------------------------------------------------------
    // Geometry
    // ------------------------------------------------------------------

    fn way_count(tlb: Tlb) -> usize {
        match tlb {
            Tlb::Instruction => ITLB_WAYS,
            Tlb::Data => DTLB_WAYS,
        }
    }

    /// Entries in `way` [RM 4.6.5.3, p.163-164].
    fn entry_count(way: u8) -> usize {
        match way {
            0..=5 => 4,
            6 => 8,
            _ => 1,
        }
    }

    /// Page-offset bit count of `way` under the current xTLBCFG
    /// [RM 4.6.5.2, p.162]. Note way 6's size bit is the reverse of way 5's.
    fn page_bits(&self, tlb: Tlb, way: u8) -> u32 {
        let cfg = self.tlbcfg(tlb);
        match way {
            4 => [20, 22, 24, 26][((cfg >> 16) & 3) as usize],
            5 => {
                if cfg & (1 << 20) != 0 {
                    28
                } else {
                    27
                }
            }
            6 => {
                if cfg & (1 << 24) != 0 {
                    28
                } else {
                    29
                }
            }
            _ => REFILL_PAGE_BITS,
        }
    }

    /// Slot selected by `vaddr`: `vaddr[P+I-1 .. P]` [RM 4.6.1.1, p.141].
    fn slot_of(&self, tlb: Tlb, way: u8, vaddr: u32) -> usize {
        let n = Self::entry_count(way);
        ((vaddr >> self.page_bits(tlb, way)) as usize) & (n - 1)
    }

    /// Mask of the "VPN without index" bits, `[31 : P+I]`, which an entry
    /// must match [RM 4.6.1.1, p.141].
    fn tag_mask(&self, tlb: Tlb, way: u8) -> u32 {
        let index_bits = Self::entry_count(way).trailing_zeros();
        let low = self.page_bits(tlb, way) + index_bits;
        if low >= 32 {
            0
        } else {
            u32::MAX << low
        }
    }

    fn page_mask(&self, tlb: Tlb, way: u8) -> u32 {
        !((1u32 << self.page_bits(tlb, way)) - 1)
    }

    fn ways(&self, tlb: Tlb) -> &[[TlbEntry; MAX_WAY_ENTRIES]] {
        match tlb {
            Tlb::Instruction => &self.itlb,
            Tlb::Data => &self.dtlb,
        }
    }

    fn ways_mut(&mut self, tlb: Tlb) -> &mut [[TlbEntry; MAX_WAY_ENTRIES]] {
        match tlb {
            Tlb::Instruction => &mut self.itlb,
            Tlb::Data => &mut self.dtlb,
        }
    }

    /// Decode the way from `as` of a TLB instruction: bits 2:0 for the ITLB
    /// (7 ways), bits 3:0 for the DTLB (10 ways) [RM 4.6.5.11, p.179]. `None`
    /// for the undefined way values [RM 4.6.5.5, p.166].
    fn decode_way(tlb: Tlb, as_: u32) -> Option<u8> {
        let way = match tlb {
            Tlb::Instruction => (as_ & 7) as u8,
            Tlb::Data => (as_ & 0xF) as u8,
        };
        if (way as usize) < Self::way_count(tlb) {
            Some(way)
        } else {
            log::debug!("mmu: {:?} TLB instruction on undefined way {} ignored", tlb, way);
            None
        }
    }

    // ------------------------------------------------------------------
    // RASID helpers
    // ------------------------------------------------------------------

    fn rasid_asid(&self, ring: u8) -> u8 {
        (self.rasid >> (8 * u32::from(ring & 3))) as u8
    }

    /// Ring whose RASID byte equals `asid`, or `None` if the ASID is invalid
    /// or in no ring. The lowest ring wins if software duplicated an ASID,
    /// which the manual leaves undefined [RM 4.6.5.2, p.161].
    fn ring_of_asid(&self, asid: u8) -> Option<u8> {
        if asid == 0 {
            return None;
        }
        (0..4u8).find(|&r| self.rasid_asid(r) == asid)
    }

    /// Read-only view of one entry for diagnostics and tests; out-of-range
    /// `way`/`slot` read as an invalid entry.
    pub fn entry(&self, tlb: Tlb, way: usize, slot: usize) -> TlbEntry {
        if way >= Self::way_count(tlb) || slot >= Self::entry_count(way as u8) {
            return TlbEntry::INVALID;
        }
        self.ways(tlb)[way][slot]
    }

    /// Every way of `tlb`, each padded to [`MAX_WAY_ENTRIES`] with invalid
    /// entries (a snapshot source for diagnostics).
    pub fn entries(&self, tlb: Tlb) -> &[[TlbEntry; MAX_WAY_ENTRIES]] {
        self.ways(tlb)
    }

    // ------------------------------------------------------------------
    // Lookup
    // ------------------------------------------------------------------

    /// Every entry matching `vaddr` [RM 4.6.2.2, p.147]: a VPN match at the
    /// indexed slot of some way, with an ASID present in RASID. Returns the
    /// matches in way order and how many there are.
    fn matches(&self, tlb: Tlb, vaddr: u32) -> ([Option<Hit>; DTLB_WAYS], usize) {
        let mut out = [None; DTLB_WAYS];
        let mut n = 0;
        for way in 0..Self::way_count(tlb) as u8 {
            let slot = self.slot_of(tlb, way, vaddr);
            let entry = self.ways(tlb)[way as usize][slot];
            let mask = self.tag_mask(tlb, way);
            if (entry.vaddr & mask) != (vaddr & mask) {
                continue;
            }
            if let Some(ring) = self.ring_of_asid(entry.asid) {
                out[n] = Some(Hit { way, slot, entry, ring });
                n += 1;
            }
        }
        (out, n)
    }

    /// Resolve overlapping matches under the configured policy. `Ok(None)` is
    /// a miss; `Err(())` is a multi-hit.
    fn resolve(&self, tlb: Tlb, vaddr: u32) -> Result<Option<Hit>, ()> {
        let (hits, mut n) = self.matches(tlb, vaddr);
        let mut hits: Vec<Hit> = hits[..n].iter().flatten().copied().collect();
        if self.config.overlap == OverlapPolicy::Way6Yields && n > 1 {
            hits.retain(|h| h.way != 6);
            n = hits.len();
        }
        match n {
            0 => Ok(None),
            1 => Ok(Some(hits[0])),
            _ => Err(()),
        }
    }

    fn miss_cause(tlb: Tlb) -> u32 {
        match tlb {
            Tlb::Instruction => cause::INST_TLB_MISS,
            Tlb::Data => cause::LOAD_STORE_TLB_MISS,
        }
    }

    fn multi_hit_cause(tlb: Tlb) -> u32 {
        match tlb {
            Tlb::Instruction => cause::INST_TLB_MULTI_HIT,
            Tlb::Data => cause::LOAD_STORE_TLB_MULTI_HIT,
        }
    }

    /// Rights column of the attribute table [RM Table 4-109, p.178]. Local
    /// memory ignores the cache columns, so only these matter here.
    pub fn attribute_allows(attr: u8, kind: AccessKind) -> bool {
        match attr {
            // Attributes 0..=11: load always; store iff bit 1; fetch iff bit 0.
            0..=11 => match kind {
                AccessKind::Load => true,
                AccessKind::Store => attr & 2 != 0,
                AccessKind::Fetch => attr & 1 != 0,
            },
            // 13: cache-isolate (test only), loads and stores.
            13 => kind != AccessKind::Fetch,
            // 12 and 14 illegal, 15 reserved.
            _ => false,
        }
    }

    /// Privilege and attribute checks on a resolved hit, in the manual's
    /// order: ring first, then attribute [RM 4.6.2.3, p.148; 4.4.1.11,
    /// p.97-98].
    fn check(
        &self,
        hit: Hit,
        vaddr: u32,
        kind: AccessKind,
        ring: u8,
    ) -> Result<Translation, TranslationFault> {
        let tlb = kind.tlb();
        let fault = |cause| TranslationFault { cause, vaddr };
        if hit.ring < ring {
            return Err(fault(match kind {
                AccessKind::Fetch => cause::INST_FETCH_PRIVILEGE,
                _ => cause::LOAD_STORE_PRIVILEGE,
            }));
        }
        if !Self::attribute_allows(hit.entry.attr, kind) {
            return Err(fault(match kind {
                AccessKind::Fetch => cause::INST_FETCH_PROHIBITED,
                AccessKind::Load => cause::LOAD_PROHIBITED,
                AccessKind::Store => cause::STORE_PROHIBITED,
            }));
        }
        let page_mask = self.page_mask(tlb, hit.way);
        Ok(Translation {
            paddr: (hit.entry.paddr & page_mask) | (vaddr & !page_mask),
            page_size: (!page_mask).wrapping_add(1),
            ring: hit.ring,
            attribute: hit.entry.attr,
            way: hit.way,
            slot: hit.slot as u8,
        })
    }

    /// Lookup without refill and without side effects: a miss returns the
    /// TLB-miss cause. Answers "does this vaddr hit, and if not why".
    pub fn lookup(&self, vaddr: u32, kind: AccessKind, ring: u8) -> Result<Translation, TranslationFault> {
        let tlb = kind.tlb();
        match self.resolve(tlb, vaddr) {
            Ok(Some(hit)) => self.check(hit, vaddr, kind, ring),
            Ok(None) => Err(TranslationFault { cause: Self::miss_cause(tlb), vaddr }),
            Err(()) => Err(TranslationFault { cause: Self::multi_hit_cause(tlb), vaddr }),
        }
    }

    /// Full translation with autorefill [RM 4.6.2, p.145-148]. `ring` is the
    /// access ring (CRING, or PS.RING for L32E/S32E).
    pub fn translate<B: XtensaBus + ?Sized>(
        &mut self,
        bus: &mut B,
        vaddr: u32,
        kind: AccessKind,
        ring: u8,
    ) -> Result<Translation, TranslationFault> {
        let tlb = kind.tlb();
        match self.resolve(tlb, vaddr) {
            Ok(Some(hit)) => return self.check(hit, vaddr, kind, ring),
            Err(()) => return Err(TranslationFault { cause: Self::multi_hit_cause(tlb), vaddr }),
            Ok(None) => {}
        }
        // Miss: hardware autorefill, then retry the access.
        if !self.autorefill(bus, tlb, vaddr) {
            return Err(TranslationFault { cause: Self::miss_cause(tlb), vaddr });
        }
        self.lookup(vaddr, kind, ring)
    }

    // ------------------------------------------------------------------
    // Autorefill
    // ------------------------------------------------------------------

    /// Address of the PTE for `vaddr`: `PTEBase || vaddr[31:12] || 2'b00`
    /// [RM 4.6.5.9, p.174].
    fn pte_address(&self, vaddr: u32) -> u32 {
        self.ptebase | ((vaddr >> 12) << 2)
    }

    /// Load the PTE for `vaddr` and write it into an autorefill way of `tlb`.
    /// Returns false on "an error of any sort during the automatic refill
    /// process" [RM 4.6.5.9, p.174], which the caller reports as the
    /// original access's TLB-miss cause.
    ///
    /// The PTE load is translated by the DTLB at ring 0 whichever TLB is
    /// being refilled [RM 4.6.2.1, p.147; 4.6.5.9, p.174]. A miss there is an
    /// error: the manual describes no nested refill and its examples map the
    /// page table through always-resident ways [RM 4.6.5.9, p.175].
    fn autorefill<B: XtensaBus + ?Sized>(&mut self, bus: &mut B, tlb: Tlb, vaddr: u32) -> bool {
        let pte_vaddr = self.pte_address(vaddr);
        let Ok(pte_loc) = self.lookup(pte_vaddr, AccessKind::Load, 0) else {
            return false;
        };
        let pte = bus.pte_load32(pte_loc.paddr);

        // PTE: PPN[31:12], Sw[11:6] (not stored), Ring[5:4], Attribute[3:0]
        // [RM 4.6.5.9, Fig. 4-42, p.174]. The ring indexes RASID to get the
        // ASID stored with the entry.
        let entry = TlbEntry {
            vaddr: vaddr & 0xFFFF_F000,
            paddr: pte & 0xFFFF_F000,
            asid: self.rasid_asid(((pte >> 4) & 3) as u8),
            attr: (pte & 0xF) as u8,
        };
        // The entry is written even if its attribute forbids the access: the
        // retry then takes the Prohibited exception, which is how software
        // uses attributes as page-valid and dirty bits [RM 4.6.1.3, p.145].
        let slot = self.slot_of(tlb, 0, vaddr);
        let way = self.pick_refill_way(tlb, slot);
        self.ways_mut(tlb)[way as usize][slot] = entry;
        true
    }

    /// The manual does not say which of ways 0-3 refill replaces. Prefer the
    /// lowest way whose slot is invalid; otherwise rotate.
    fn pick_refill_way(&mut self, tlb: Tlb, slot: usize) -> u8 {
        let free = (0..REFILL_WAYS).find(|&w| self.ways(tlb)[w as usize][slot].asid == 0);
        if let Some(w) = free {
            return w;
        }
        let victim = &mut self.refill_victim[tlb as usize];
        let w = *victim;
        *victim = (w + 1) % REFILL_WAYS;
        w
    }

    // ------------------------------------------------------------------
    // TLB instructions
    // ------------------------------------------------------------------

    /// WITLB / WDTLB `at, as` [RM 4.6.5.5, p.165-168; p.557, p.560].
    ///
    /// `as_` carries the way in its low bits and the virtual address in bits
    /// 31:12 (bits the way does not use are ignored, so a plain virtual
    /// address works). `at` carries the PPN in 31:12, ring in 5:4 and
    /// attribute in 3:0. The entry's ASID becomes the current RASID byte for
    /// that ring; if RASID holds 0 there the entry is invalid. Takes effect
    /// immediately (ISYNC/DSYNC timing is implementation-specific).
    pub fn write_entry(&mut self, tlb: Tlb, as_: u32, at: u32) {
        let Some(way) = Self::decode_way(tlb, as_) else {
            return;
        };
        let slot = self.slot_of(tlb, way, as_);
        let page_mask = self.page_mask(tlb, way);
        let entry = TlbEntry {
            vaddr: as_ & page_mask,
            paddr: at & page_mask,
            asid: self.rasid_asid(((at >> 4) & 3) as u8),
            attr: (at & 0xF) as u8,
        };
        self.ways_mut(tlb)[way as usize][slot] = entry;
    }

    /// IITLB / IDTLB `as` [RM 4.6.5.8, p.172-174; p.348, p.355]: sets the
    /// addressed entry's ASID to zero.
    pub fn invalidate_entry(&mut self, tlb: Tlb, as_: u32) {
        let Some(way) = Self::decode_way(tlb, as_) else {
            return;
        };
        let slot = self.slot_of(tlb, way, as_);
        self.ways_mut(tlb)[way as usize][slot].asid = 0;
    }

    /// RITLB0 / RDTLB0 `as` [RM 4.6.5.6, p.168-171]: the entry's ASID in
    /// bits 7:0 and its "VPN without index" in 31:12 (index bits and bits
    /// above the way's tag read 0; bits 11:8 are undefined and read 0).
    pub fn read_entry_virtual(&self, tlb: Tlb, as_: u32) -> u32 {
        let Some(way) = Self::decode_way(tlb, as_) else {
            return 0;
        };
        let e = self.ways(tlb)[way as usize][self.slot_of(tlb, way, as_)];
        (e.vaddr & self.tag_mask(tlb, way)) | u32::from(e.asid)
    }

    /// RITLB1 / RDTLB1 `as` [RM 4.6.5.6, Fig. 4-37, p.171]: the physical page
    /// base in 31:12 and the attribute in 3:0 (bits 11:4 undefined, read 0).
    pub fn read_entry_translation(&self, tlb: Tlb, as_: u32) -> u32 {
        let Some(way) = Self::decode_way(tlb, as_) else {
            return 0;
        };
        let e = self.ways(tlb)[way as usize][self.slot_of(tlb, way, as_)];
        (e.paddr & self.page_mask(tlb, way)) | u32::from(e.attr)
    }

    /// PITLB / PDTLB `as` [RM 4.6.5.7, p.171-172; p.469-470]: which way
    /// would translate the address at ring `ps_ring` (PS.RING, not CRING),
    /// without any refill. A hit returns `(as & 0xFFFF_F000) | hit_bit | way`
    /// (hit bit 3 for the ITLB, bit 4 for the DTLB), usable as the `as` of
    /// the other TLB instructions; no hit returns 0. A multi-hit is a fault
    /// whose EXCVADDR is `as`. An entry whose ring is below `ps_ring` does
    /// not hit; overlap resolution happens before that ring filter, since
    /// multi-hit detection ignores rings [RM 4.6.1.2, p.142].
    pub fn probe(&self, tlb: Tlb, as_: u32, ps_ring: u8) -> Result<u32, TranslationFault> {
        match self.resolve(tlb, as_) {
            Err(()) => Err(TranslationFault { cause: Self::multi_hit_cause(tlb), vaddr: as_ }),
            Ok(Some(hit)) if hit.ring >= ps_ring => {
                let hit_bit = match tlb {
                    Tlb::Instruction => 0x8,
                    Tlb::Data => 0x10,
                };
                Ok((as_ & 0xFFFF_F000) | hit_bit | u32::from(hit.way))
            }
            Ok(_) => Ok(0),
        }
    }

    // ------------------------------------------------------------------
    // Special registers
    // ------------------------------------------------------------------

    /// WSR.PTEVADDR: only PTEBase (bits 31:22) is writable [RM 4.6.5.2,
    /// p.161].
    pub fn write_ptevaddr(&mut self, value: u32) {
        self.ptebase = value & 0xFFC0_0000;
    }

    /// RSR.PTEVADDR: `PTEBase || EXCVADDR[31:12] || 2'b00` [RM 4.6.5.2,
    /// Fig. 4-29, p.161].
    pub fn read_ptevaddr(&self, excvaddr: u32) -> u32 {
        self.pte_address(excvaddr)
    }

    /// WSR.RASID: ring 0's ASID is hardwired to 1 [RM 4.6.5.2, p.161;
    /// Table 5-150, p.222]. Other bytes are stored as written; duplicates
    /// and zeros are undefined in the manual and treated as described at
    /// [`Mmu::ring_of_asid`].
    pub fn write_rasid(&mut self, value: u32) {
        self.rasid = (value & !0xFF) | u32::from(RING0_ASID);
    }

    pub fn rasid(&self) -> u32 {
        self.rasid
    }

    /// WSR.ITLBCFG / WSR.DTLBCFG: the whole word is stored and read back
    /// [RM Tables 5-151/5-152, p.223]; only bits 17:16 (way 4), 20 (way 5)
    /// and 24 (way 6) are decoded. The way is not invalidated on a size
    /// change; the manual says software should do so [RM 4.6.5.2, p.162].
    pub fn write_tlbcfg(&mut self, tlb: Tlb, value: u32) {
        match tlb {
            Tlb::Instruction => self.itlbcfg = value,
            Tlb::Data => self.dtlbcfg = value,
        }
    }

    pub fn tlbcfg(&self, tlb: Tlb) -> u32 {
        match tlb {
            Tlb::Instruction => self.itlbcfg,
            Tlb::Data => self.dtlbcfg,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;

    use Tlb::{Data as D, Instruction as I};

    /// Bus whose only meaningful method is `pte_load32`; it records the
    /// addresses read.
    #[derive(Default)]
    struct PteBus {
        ptes: HashMap<u32, u32>,
        reads: RefCell<Vec<u32>>,
    }

    impl XtensaBus for PteBus {
        fn fetch8(&mut self, _: u32, _: u32) -> u8 {
            0
        }
        fn inst_load8(&mut self, _: u32) -> u8 {
            0
        }
        fn literal_load32(&mut self, _: u32, _: u32) -> u32 {
            0
        }
        fn data_load32(&mut self, _: u32, _: u32) -> u32 {
            panic!("MMU must read PTEs with pte_load32 only")
        }
        fn data_load8(&mut self, _: u32) -> u8 {
            panic!("MMU must read PTEs with pte_load32 only")
        }
        fn data_store32(&mut self, _: u32, _: u32) {
            panic!("MMU must not store")
        }
        fn data_store8(&mut self, _: u32, _: u32) {
            panic!("MMU must not store")
        }
        fn pte_load32(&mut self, paddr: u32) -> u32 {
            self.reads.borrow_mut().push(paddr);
            self.ptes.get(&paddr).copied().unwrap_or(0)
        }
    }

    fn way6_yields() -> Mmu {
        Mmu::new(MmuConfig { way6_reset_attribute: 3, overlap: OverlapPolicy::Way6Yields })
    }

    fn generic() -> Mmu {
        Mmu::new(MmuConfig::GENERIC)
    }

    /// Invalidate the eight reset way-6 entries of both TLBs (valid while
    /// xTLBCFG bit 24 is 0).
    fn clear_way6(m: &mut Mmu) {
        for n in 0..8u32 {
            m.invalidate_entry(I, (n << 29) | 6);
            m.invalidate_entry(D, (n << 29) | 6);
        }
    }

    fn fault(cause: u32, vaddr: u32) -> TranslationFault {
        TranslationFault { cause, vaddr }
    }

    // -- 10.1 Reset ----------------------------------------------------

    #[test]
    fn reset_registers() {
        let m = generic();
        assert_eq!(m.rasid(), 0x0403_0201);
        assert_eq!(m.tlbcfg(I), 0);
        assert_eq!(m.tlbcfg(D), 0);
        assert_eq!(m.read_ptevaddr(0), 0);
    }

    #[test]
    fn reset_way6_identity_maps_everything() {
        for m in [generic(), way6_yields()] {
            for kind in [AccessKind::Fetch, AccessKind::Load, AccessKind::Store] {
                for va in [0x1234_5678, 0xE000_0010, 0xFFFF_FFFC, 0] {
                    let t = m.lookup(va, kind, 0).unwrap();
                    assert_eq!(t.paddr, va);
                    assert_eq!(t.page_size, 0x2000_0000);
                    assert_eq!((t.ring, t.way, t.attribute), (0, 6, 3));
                }
            }
        }
    }

    #[test]
    fn reset_way6_attribute_is_configurable() {
        let m = Mmu::new(MmuConfig { way6_reset_attribute: 2, overlap: OverlapPolicy::Strict });
        // The manual's attribute 2: load/store but no fetch.
        assert!(m.lookup(0x1000, AccessKind::Load, 0).is_ok());
        assert_eq!(m.lookup(0x1000, AccessKind::Fetch, 0), Err(fault(cause::INST_FETCH_PROHIBITED, 0x1000)));
    }

    #[test]
    fn reset_ring1_access_is_a_privilege_fault() {
        let m = generic();
        assert_eq!(m.lookup(0x1000, AccessKind::Fetch, 1), Err(fault(18, 0x1000)));
        assert_eq!(m.lookup(0x1000, AccessKind::Load, 1), Err(fault(26, 0x1000)));
        assert_eq!(m.lookup(0x1000, AccessKind::Store, 1), Err(fault(26, 0x1000)));
    }

    #[test]
    fn reset_way6_reads() {
        let m = generic();
        assert_eq!(m.read_entry_virtual(I, 0x4000_0006), 0x0000_0001);
        assert_eq!(m.read_entry_translation(I, 0x4000_0006), 0x4000_0003);
        assert_eq!(m.read_entry_virtual(D, 0xE000_0006), 0x0000_0001);
        assert_eq!(m.read_entry_translation(D, 0xE000_0006), 0xE000_0003);
    }

    // -- 10.2 Special registers ----------------------------------------

    #[test]
    fn rasid_ring0_is_hardwired() {
        let mut m = generic();
        m.write_rasid(0x0807_0600);
        assert_eq!(m.rasid(), 0x0807_0601);
    }

    #[test]
    fn ptevaddr_write_keeps_base_and_read_merges_excvaddr() {
        let mut m = generic();
        m.write_ptevaddr(0x1234_5678);
        assert_eq!(m.read_ptevaddr(0xFFFF_FFFF), 0x123F_FFFC);
        assert_eq!(m.read_ptevaddr(0x0040_3123), 0x1200_100C);
    }

    #[test]
    fn tlbcfg_round_trips_whole_word_per_tlb() {
        let mut m = generic();
        m.write_tlbcfg(I, 0x0101_0000);
        assert_eq!(m.tlbcfg(I), 0x0101_0000);
        assert_eq!(m.tlbcfg(D), 0);
        m.write_tlbcfg(D, 0xFFFF_FFFF);
        assert_eq!(m.tlbcfg(D), 0xFFFF_FFFF);
    }

    // -- 10.3 Entry writes and lookups ---------------------------------

    #[test]
    fn dtlb_way7_installs_and_translates() {
        let mut m = way6_yields();
        m.write_entry(D, 0x4000_1000 | 7, 0x0009_0000 | 3);
        let t = m.lookup(0x4000_1ABC, AccessKind::Load, 0).unwrap();
        assert_eq!((t.paddr, t.page_size, t.way), (0x0009_0ABC, 0x1000, 7));
    }

    #[test]
    fn strict_policy_multihits_with_way6_until_it_is_cleared() {
        let mut m = generic();
        m.write_entry(D, 0x4000_1000 | 7, 0x0009_0000 | 3);
        assert_eq!(m.lookup(0x4000_1ABC, AccessKind::Load, 0), Err(fault(25, 0x4000_1ABC)));
        clear_way6(&mut m);
        assert_eq!(m.lookup(0x4000_1ABC, AccessKind::Load, 0).unwrap().paddr, 0x0009_0ABC);
    }

    #[test]
    fn way6_yields_does_not_hide_two_other_matches() {
        let mut m = way6_yields();
        m.write_entry(D, 0x4000_1000 | 7, 0x0009_0000 | 3);
        m.write_entry(D, 0x4000_1000 | 8, 0x0009_0000 | 3);
        assert_eq!(m.lookup(0x4000_1ABC, AccessKind::Load, 0), Err(fault(25, 0x4000_1ABC)));
    }

    #[test]
    fn way4_4mb_itlb() {
        let mut m = way6_yields();
        m.write_tlbcfg(I, 0x0001_0000);
        m.write_entry(I, 0x12C0_0000 | 4, 0x2000_0000 | 3);
        let t = m.lookup(0x12C1_2345, AccessKind::Fetch, 0).unwrap();
        assert_eq!((t.paddr, t.page_size, t.way), (0x2001_2345, 0x40_0000, 4));
        // Index 2 of the same 4 MB way holds nothing: falls through to way 6.
        assert_eq!(m.lookup(0x1281_2345, AccessKind::Fetch, 0).unwrap().way, 6);
        assert_eq!(m.read_entry_virtual(I, 0x12C0_0004), 0x1200_0001);
        assert_eq!(m.read_entry_translation(I, 0x12C0_0004), 0x2000_0003);
    }

    #[test]
    fn way4_page_sizes() {
        for (code, bits) in [(0u32, 20u32), (1, 22), (2, 24), (3, 26)] {
            let mut m = way6_yields();
            m.write_tlbcfg(D, code << 16);
            let size = 1u32 << bits;
            m.write_entry(D, 0x8000_0000 | 4, 0x1000_0000 | 3);
            let t = m.lookup(0x8000_0000 + size - 4, AccessKind::Load, 0).unwrap();
            assert_eq!((t.way, t.page_size, t.paddr), (4, size, 0x1000_0000 + size - 4));
        }
    }

    #[test]
    fn itlb_way_field_ignores_bit3() {
        let mut m = way6_yields();
        m.write_tlbcfg(I, 0x0001_0000);
        m.write_entry(I, 0x12C0_0000 | 0xC, 0x2000_0000 | 3);
        let t = m.lookup(0x12C1_2345, AccessKind::Fetch, 0).unwrap();
        assert_eq!((t.paddr, t.way), (0x2001_2345, 4));
    }

    #[test]
    fn way5_128mb_dtlb_and_256mb() {
        let mut m = way6_yields();
        m.write_entry(D, 0xD800_0005, 0x0800_0003);
        let t = m.lookup(0xDFFF_FFFC, AccessKind::Store, 0).unwrap();
        assert_eq!((t.paddr, t.page_size, t.way), (0x0FFF_FFFC, 0x800_0000, 5));

        m.write_tlbcfg(D, 1 << 20);
        m.write_entry(D, 0xC000_0005, 0x1000_0000 | 3);
        let t = m.lookup(0xCFFF_FFF0, AccessKind::Load, 0).unwrap();
        assert_eq!((t.paddr, t.page_size, t.way), (0x1FFF_FFF0, 0x1000_0000, 5));
    }

    #[test]
    fn way6_256mb_itlb() {
        let mut m = generic();
        m.write_tlbcfg(I, 0x0100_0000);
        for n in 0..8u32 {
            m.invalidate_entry(I, (n << 28) | 6);
        }
        m.write_entry(I, 0xA000_0006, 0x3000_0003);
        assert_eq!(m.lookup(0xA123_4567, AccessKind::Fetch, 0).unwrap().paddr, 0x3123_4567);
        assert_eq!(m.lookup(0xA123_4567, AccessKind::Fetch, 0).unwrap().page_size, 0x1000_0000);
        // Same index (2) but the top VPN bit differs.
        assert_eq!(m.lookup(0x2123_4567, AccessKind::Fetch, 0), Err(fault(16, 0x2123_4567)));
    }

    #[test]
    fn way6_512mb_write_uses_three_index_bits() {
        let mut m = generic();
        clear_way6(&mut m);
        m.write_entry(D, 0x6000_0006, 0x4000_0000 | 3);
        let t = m.lookup(0x7FFF_FFFC, AccessKind::Load, 0).unwrap();
        assert_eq!((t.paddr, t.way), (0x5FFF_FFFC, 6));
        assert!(m.lookup(0x5FFF_FFFC, AccessKind::Load, 0).is_err());
    }

    #[test]
    fn refill_ways_use_4kb_pages_and_two_index_bits() {
        let mut m = generic();
        clear_way6(&mut m);
        m.write_entry(D, 0x0040_3000 | 2, 0x0A00_0000 | 3);
        let t = m.lookup(0x0040_3FFC, AccessKind::Load, 0).unwrap();
        assert_eq!((t.paddr, t.page_size, t.way), (0x0A00_0FFC, 0x1000, 2));
        // 0x0044_3000 has the same index bits [13:12] but a different tag.
        assert!(m.lookup(0x0044_3000, AccessKind::Load, 0).is_err());
        // 0x0040_4000 has index 0.
        assert!(m.lookup(0x0040_4000, AccessKind::Load, 0).is_err());
    }

    #[test]
    fn ring_in_at_selects_the_asid() {
        let mut m = generic();
        clear_way6(&mut m);
        m.write_entry(D, 0x5000_0000 | 7, 0x0001_0000 | (2 << 4) | 3);
        assert_eq!(m.read_entry_virtual(D, 7), 0x5000_0003);
        let t = m.lookup(0x5000_0010, AccessKind::Load, 2).unwrap();
        assert_eq!(t.ring, 2);
        assert_eq!(m.lookup(0x5000_0010, AccessKind::Load, 3), Err(fault(26, 0x5000_0010)));
        // Ring 0 may reach every ring's pages.
        assert!(m.lookup(0x5000_0010, AccessKind::Load, 0).is_ok());
    }

    #[test]
    fn entry_stops_matching_when_its_asid_leaves_rasid() {
        let mut m = generic();
        clear_way6(&mut m);
        m.write_entry(D, 0x5000_0000 | 7, 0x0001_0000 | (2 << 4) | 3);
        m.write_rasid(0x0405_0201);
        assert_eq!(m.lookup(0x5000_0010, AccessKind::Load, 0), Err(fault(24, 0x5000_0010)));
    }

    #[test]
    fn writing_through_a_zero_rasid_byte_stores_an_invalid_entry() {
        let mut m = generic();
        clear_way6(&mut m);
        m.write_rasid(0x0403_0001); // ring 1 ASID 0
        m.write_entry(D, 0x5000_0000 | 7, 0x0001_0000 | (1 << 4) | 3);
        assert_eq!(m.lookup(0x5000_0010, AccessKind::Load, 0), Err(fault(24, 0x5000_0010)));
        assert_eq!(m.read_entry_virtual(D, 7) & 0xFF, 0);
    }

    #[test]
    fn duplicate_rasid_asids_resolve_to_the_lowest_ring() {
        let mut m = generic();
        clear_way6(&mut m);
        m.write_entry(D, 0x5000_0000 | 7, 0x0001_0000 | (2 << 4) | 3); // ASID 3
        m.write_rasid(0x0303_0201); // rings 2 and 3 both ASID 3
        assert_eq!(m.lookup(0x5000_0010, AccessKind::Load, 0).unwrap().ring, 2);
    }

    // -- 10.4 Rights ---------------------------------------------------

    #[test]
    fn attribute_rights_table() {
        // (load, store, fetch) per attribute 0..15, RM Table 4-109.
        let table: [(bool, bool, bool); 16] = [
            (true, false, false),
            (true, false, true),
            (true, true, false),
            (true, true, true),
            (true, false, false),
            (true, false, true),
            (true, true, false),
            (true, true, true),
            (true, false, false),
            (true, false, true),
            (true, true, false),
            (true, true, true),
            (false, false, false),
            (true, true, false),
            (false, false, false),
            (false, false, false),
        ];
        for (attr, &(load, store, fetch)) in table.iter().enumerate() {
            let mut m = generic();
            clear_way6(&mut m);
            m.write_entry(D, 0x5000_0000 | 7, 0x0001_0000 | attr as u32);
            m.write_tlbcfg(I, 0x0001_0000);
            m.write_entry(I, 0x4000_0000 | 4, 0x0100_0000 | attr as u32);
            let want = |ok, cause| if ok { None } else { Some(cause) };
            let got = |r: Result<Translation, TranslationFault>| r.err().map(|f| f.cause);
            assert_eq!(got(m.lookup(0x5000_0000, AccessKind::Load, 0)), want(load, 28), "attr {attr} load");
            assert_eq!(
                got(m.lookup(0x5000_0000, AccessKind::Store, 0)),
                want(store, 29),
                "attr {attr} store"
            );
            assert_eq!(
                got(m.lookup(0x4000_0000, AccessKind::Fetch, 0)),
                want(fetch, 20),
                "attr {attr} fetch"
            );
        }
    }

    #[test]
    fn privilege_is_checked_before_attribute() {
        let mut m = generic();
        clear_way6(&mut m);
        m.write_entry(D, 0x5000_0000 | 7, 0x0001_0000 | (1 << 4) | 12);
        assert_eq!(m.lookup(0x5000_0000, AccessKind::Load, 2), Err(fault(26, 0x5000_0000)));
        assert_eq!(m.lookup(0x5000_0000, AccessKind::Load, 1), Err(fault(28, 0x5000_0000)));
    }

    // -- 10.5 Invalidate -----------------------------------------------

    #[test]
    fn invalidate_falls_back_to_way6_or_misses() {
        let mut m = way6_yields();
        m.write_entry(D, 0x4000_1000 | 7, 0x0009_0000 | 3);
        m.invalidate_entry(D, 0x4000_1000 | 7);
        assert_eq!(m.lookup(0x4000_1ABC, AccessKind::Load, 0).unwrap().way, 6);

        let mut g = generic();
        clear_way6(&mut g);
        g.write_entry(D, 0x4000_1000 | 7, 0x0009_0000 | 3);
        g.invalidate_entry(D, 0x4000_1000 | 7);
        assert_eq!(g.lookup(0x4000_1ABC, AccessKind::Load, 0), Err(fault(24, 0x4000_1ABC)));
    }

    #[test]
    fn invalidate_clears_only_the_indexed_slot() {
        let mut m = generic();
        clear_way6(&mut m);
        for slot in 0..4u32 {
            m.write_entry(D, (0x0040_0000 + slot * 0x1000) | 0, 0x0100_0000 | 3);
        }
        m.invalidate_entry(D, 0x0040_3000);
        for slot in 0..4u32 {
            let ok = m.lookup(0x0040_0000 + slot * 0x1000, AccessKind::Load, 0).is_ok();
            assert_eq!(ok, slot != 3, "slot {slot}");
        }
    }

    #[test]
    fn undefined_ways_are_ignored() {
        let mut m = generic();
        clear_way6(&mut m);
        m.write_entry(I, 0x5000_0000 | 7, 3); // ITLB way 7 does not exist
        m.write_entry(D, 0x5000_0000 | 10, 3); // nor DTLB way 10
        m.invalidate_entry(I, 7);
        assert_eq!(m.read_entry_virtual(I, 7), 0);
        assert_eq!(m.read_entry_translation(D, 15), 0);
        assert!(m.lookup(0x5000_0000, AccessKind::Fetch, 0).is_err());
        assert!(m.lookup(0x5000_0000, AccessKind::Load, 0).is_err());
    }

    #[test]
    fn dtlb_has_ways_7_to_9_but_itlb_does_not() {
        let mut m = generic();
        clear_way6(&mut m);
        for way in 7..=9u32 {
            m.write_entry(D, (0x6000_0000 + way * 0x1000) | way, 0x0200_0000 | 3);
            assert_eq!(m.lookup(0x6000_0000 + way * 0x1000, AccessKind::Load, 0).unwrap().way as u32, way);
        }
    }

    #[test]
    fn size_change_without_invalidation_uses_current_geometry() {
        // Unspecified by the manual: stale entries are matched as-is against
        // the new size's index and tag bits.
        let mut m = generic();
        clear_way6(&mut m);
        m.write_entry(D, 0x0000_0000 | 4, 0x1000_0000 | 3); // 1 MB page at 0
        m.write_tlbcfg(D, 0x0001_0000); // way 4 becomes 4 MB
        let t = m.lookup(0x0000_0010, AccessKind::Load, 0).unwrap();
        assert_eq!((t.page_size, t.paddr), (0x40_0000, 0x1000_0010));
    }

    // -- 10.6 Autorefill ------------------------------------------------

    /// Generic MMU, way 6 cleared, PTE page at 0x8000_1000 mapped by DTLB
    /// way 8 onto paddr 0x0010_0000, PTE for page 0x0040_3000 present.
    fn refill_setup(pte: u32) -> (Mmu, PteBus) {
        let mut m = generic();
        clear_way6(&mut m);
        m.write_ptevaddr(0x8000_0000);
        m.write_entry(D, 0x8000_1000 | 8, 0x0010_0000 | 3);
        let mut bus = PteBus::default();
        bus.ptes.insert(0x0010_000C, pte);
        (m, bus)
    }

    #[test]
    fn refill_loads_the_pte_once_and_caches_it() {
        let (mut m, mut bus) = refill_setup(0x0ABC_D000 | (1 << 4) | 3);
        let t = m.translate(&mut bus, 0x0040_3123, AccessKind::Load, 0).unwrap();
        assert_eq!((t.paddr, t.page_size, t.ring), (0x0ABC_D123, 0x1000, 1));
        assert_eq!(*bus.reads.borrow(), vec![0x0010_000C]);
        m.translate(&mut bus, 0x0040_3FFC, AccessKind::Load, 0).unwrap();
        assert_eq!(bus.reads.borrow().len(), 1);
        let way = t.way as u32;
        assert!(way < 4);
        assert_eq!(m.read_entry_virtual(D, 0x0040_3000 | way), 0x0040_0002);
        assert_eq!(m.read_entry_translation(D, 0x0040_3000 | way), 0x0ABC_D003);
    }

    #[test]
    fn refill_entry_ring_comes_from_the_pte_ring_field() {
        let (mut m, mut bus) = refill_setup(0x0ABC_D000 | (1 << 4) | 3);
        let f = m.translate(&mut bus, 0x0040_3123, AccessKind::Load, 2).unwrap_err();
        assert_eq!(f, fault(26, 0x0040_3123));
    }

    #[test]
    fn itlb_refill_translates_the_pte_address_through_the_dtlb() {
        let (mut m, mut bus) = refill_setup(0x0ABC_D000 | 1);
        let t = m.translate(&mut bus, 0x0040_3000, AccessKind::Fetch, 0).unwrap();
        assert!(t.way < 4);
        assert_eq!(t.paddr, 0x0ABC_D000);
        assert_eq!(*bus.reads.borrow(), vec![0x0010_000C]);
        // Only the ITLB gained an entry for the page.
        assert!(m.lookup(0x0040_3000, AccessKind::Load, 0).is_err());
    }

    #[test]
    fn unmapped_pte_page_is_a_miss_with_the_original_vaddr_and_no_recursion() {
        let (mut m, mut bus) = refill_setup(0x0ABC_D000 | 3);
        m.invalidate_entry(D, 0x8000_1000 | 8);
        let f = m.translate(&mut bus, 0x0040_3123, AccessKind::Load, 0).unwrap_err();
        assert_eq!(f, fault(24, 0x0040_3123));
        let f = m.translate(&mut bus, 0x0040_3123, AccessKind::Fetch, 0).unwrap_err();
        assert_eq!(f, fault(16, 0x0040_3123));
        assert!(bus.reads.borrow().is_empty());
        // The would-be nested refill left no entry behind.
        assert!(m.lookup(0x8000_100C, AccessKind::Load, 0).is_err());
    }

    #[test]
    fn pte_page_with_illegal_attribute_is_a_miss() {
        let (mut m, mut bus) = refill_setup(0x0ABC_D000 | 3);
        m.write_entry(D, 0x8000_1000 | 8, 0x0010_0000 | 12);
        let f = m.translate(&mut bus, 0x0040_3123, AccessKind::Load, 0).unwrap_err();
        assert_eq!(f, fault(24, 0x0040_3123));
        let f = m.translate(&mut bus, 0x0040_3123, AccessKind::Fetch, 0).unwrap_err();
        assert_eq!(f, fault(16, 0x0040_3123));
    }

    #[test]
    fn forbidding_pte_is_cached_and_faults_on_retry() {
        let (mut m, mut bus) = refill_setup(0x0ABC_D000 | 12);
        assert_eq!(m.translate(&mut bus, 0x0040_3123, AccessKind::Load, 0), Err(fault(28, 0x0040_3123)));
        assert_eq!(m.translate(&mut bus, 0x0040_3123, AccessKind::Store, 0), Err(fault(29, 0x0040_3123)));
        assert_eq!(m.translate(&mut bus, 0x0040_3123, AccessKind::Fetch, 0), Err(fault(20, 0x0040_3123)));
    }

    #[test]
    fn read_only_pte_allows_load_and_refuses_store() {
        let (mut m, mut bus) = refill_setup(0x0ABC_D000 | 1);
        assert!(m.translate(&mut bus, 0x0040_3123, AccessKind::Load, 0).is_ok());
        assert_eq!(m.translate(&mut bus, 0x0040_3123, AccessKind::Store, 0), Err(fault(29, 0x0040_3123)));
    }

    #[test]
    fn refill_fills_ways_in_order_then_rotates() {
        let (mut m, mut bus) = refill_setup(0);
        // Five pages sharing index 3; their PTEs all sit in the mapped PTE
        // page 0x8000_1000 (paddr 0x0010_0000 + offset).
        let pages = [0x0040_3000u32, 0x0041_3000, 0x0042_3000, 0x0043_3000, 0x0044_3000];
        for (i, &va) in pages.iter().enumerate() {
            let pte_paddr = 0x0010_0000 + (((va >> 12) << 2) & 0xFFF);
            bus.ptes.insert(pte_paddr, (0x0100_0000 + (i as u32) * 0x1000) | 3);
            let t = m.translate(&mut bus, va, AccessKind::Load, 0).unwrap();
            assert_eq!(t.way as usize, i % 4, "page {i}");
        }
        // The fifth page replaced way 0, so the first page misses in the TLB
        // (no refill in `lookup`) while the others still hit.
        assert!(m.lookup(0x0040_3000, AccessKind::Load, 0).is_err());
        assert!(m.lookup(0x0041_3000, AccessKind::Load, 0).is_ok());
    }

    #[test]
    fn ptevaddr_read_after_refill_fault_names_the_pte() {
        let (m, _) = refill_setup(0);
        assert_eq!(m.read_ptevaddr(0x0040_3123), 0x8000_100C);
    }

    // -- 10.7 Way-6 policy ---------------------------------------------

    #[test]
    fn way6_yields_to_a_more_specific_entry_even_if_it_is_stricter() {
        let mut m = way6_yields();
        m.write_entry(D, 0x4000_1000 | 7, 0x0009_0000); // attribute 0: read-only
        assert_eq!(m.lookup(0x4000_1000, AccessKind::Store, 0), Err(fault(29, 0x4000_1000)));
        let t = m.lookup(0x4000_2000, AccessKind::Store, 0).unwrap();
        assert_eq!(t.way, 6);
    }

    #[test]
    fn no_refill_with_way6_intact() {
        let mut m = way6_yields();
        let mut bus = PteBus::default();
        for va in [0u32, 0x1234_5678, 0xFFFF_FFFC, 0x8000_0000] {
            for kind in [AccessKind::Fetch, AccessKind::Load, AccessKind::Store] {
                m.translate(&mut bus, va, kind, 0).unwrap();
            }
        }
        assert!(bus.reads.borrow().is_empty());
    }

    #[test]
    fn refill_pte_lookup_follows_the_overlap_policy() {
        // Way6Yields: the PTE page is covered by way 6 alone, so the walk succeeds
        // using it. Generic would multi-hit if another way also matched.
        let mut m = way6_yields();
        for n in 0..8u32 {
            if n != 0 {
                m.invalidate_entry(D, (n << 29) | 6);
            }
            m.invalidate_entry(I, (n << 29) | 6);
        }
        m.write_ptevaddr(0x0040_0000);
        let mut bus = PteBus::default();
        bus.ptes.insert(0x0040_0000 + ((0x0050_0000u32 >> 12) << 2), 0x0777_7000 | 3);
        let t = m.translate(&mut bus, 0x0050_0010, AccessKind::Fetch, 0).unwrap();
        assert_eq!(t.paddr, 0x0777_7010);
    }

    // -- 10.8 Probe ----------------------------------------------------

    #[test]
    fn probe_reports_hit_and_way() {
        let mut m = way6_yields();
        m.write_entry(D, 0x4000_1000 | 7, 0x0009_0000 | 3);
        assert_eq!(m.probe(D, 0x4000_1ABC, 0), Ok(0x4000_1000 | 0x10 | 7));
    }

    #[test]
    fn probe_result_is_a_valid_instruction_operand() {
        let mut m = way6_yields();
        m.write_tlbcfg(I, 0x0001_0000);
        m.write_entry(I, 0x12C0_0000 | 4, 0x2000_0000 | 3);
        let p = m.probe(I, 0x12C1_2345, 0).unwrap();
        assert_eq!(p, 0x12C1_2000 | 0x8 | 4);
        m.invalidate_entry(I, p);
        assert_eq!(m.lookup(0x12C1_2345, AccessKind::Fetch, 0).unwrap().way, 6);
    }

    #[test]
    fn probe_filters_by_ps_ring() {
        let mut m = generic();
        clear_way6(&mut m);
        m.write_entry(D, 0x5000_0000 | 7, 0x0001_0000 | (2 << 4) | 3);
        assert_eq!(m.probe(D, 0x5000_0010, 2).unwrap() & 0x10, 0x10);
        assert_eq!(m.probe(D, 0x5000_0010, 3), Ok(0));
    }

    #[test]
    fn probe_multihit_is_a_fault_with_the_probed_address() {
        let mut m = generic();
        m.write_entry(D, 0x4000_1000 | 7, 0x0009_0000 | 3);
        assert_eq!(m.probe(D, 0x4000_1ABC, 0), Err(fault(25, 0x4000_1ABC)));
        assert_eq!(m.probe(D, 0x9000_0000, 0).map(|v| v & 0x10), Ok(0x10)); // way 6 only
    }

    #[test]
    fn probe_does_not_refill() {
        let (m, bus) = refill_setup(0x0ABC_D000 | 3);
        assert_eq!(m.probe(D, 0x0040_3123, 0), Ok(0));
        assert!(bus.reads.borrow().is_empty());
    }
}
