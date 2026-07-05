//! Synthesized PSP autorefill page table (M2c). The x86 PSP builds the code
//! region's virtual->physical page table in management-processor RAM before
//! starting the firmware; that table is absent from every artifact we hold, so
//! we reconstruct its OBSERVED EFFECT by coherence: a linear map of the code
//! region (virtual 0x20000000+) onto the firmware image, matching the firmware's
//! own way-5 region install (`virtual 0x20000000 -> phys 0`, `mmu_helper.c`
//! varway56 path) extended to per-page autorefill entries. See
//! `docs/superpowers/specs/2026-07-04-m2c-mapping-boot-to-idle-design.md`.

use crate::firmware::mmio::{MAILBOX_BASE, MAILBOX_END, PAGE_TABLE_BASE};
use crate::firmware::xtensa::mmu::Mmu;
use crate::firmware::Bus;

/// Virtual base of the firmware code region (the `jx 0x20000340` target's page).
pub const CODE_REGION_BASE: u32 = 0x2000_0000;

/// Install the synthesized page table: a region entry making the PTEVADDR window
/// fetchable by the autorefill walk, plus one PTE per code page.
pub fn install(mmu: &mut Mmu, bus: &mut Bus, _load_offset: u32, image_len: u32) {
    let ptevaddr = mmu.ptevaddr;

    // Region entry so get_pte can translate pt_vaddr (0x3c000000 window) without
    // autorefill: way-4 D-TLB, virtual PTEVADDR -> phys PTEVADDR (identity into
    // the PageTable aperture), attr 3 (R/W). The PSP establishes this window on
    // real hardware; we model its effect. Way 4 is untouched by the firmware's
    // own way-5/6 TLB ops.
    debug_assert_eq!(ptevaddr, PAGE_TABLE_BASE, "synth PT assumes PTEVADDR == page-table aperture base");
    mmu.write_tlb(true, ptevaddr | 0x3, ptevaddr | 4);

    // One PTE per 4 KB code page across the image. phys = virtual - CODE_REGION_BASE
    // (the way-5 region map extended per page); attr 7 = cached RWX, ring 0.
    let npages = image_len / 0x1000 + 1;
    for i in 0..npages {
        let v = CODE_REGION_BASE + i * 0x1000;
        let phys = v - CODE_REGION_BASE;
        let pte = phys | 0x7;
        let pt_phys = (ptevaddr | (v >> 10)) & !3;
        bus.write_page_table_word(pt_phys, pte);
    }

    // Mailbox / peripheral aperture (0x27000000..0x28000000): identity PTEs, attr 2
    // (RW, non-executable device memory). The firmware's prologue invalidates the
    // varway56 region entry covering this range, so its mailbox stores fall to the
    // autorefill walk and need a writable PTE here or they fault STORE_PROHIBITED.
    // The PSP maps this aperture 1:1 on real hardware; we model that effect. The
    // mailbox *contents* remain a Bus RAM stub -- this only grants access.
    let mut v = MAILBOX_BASE;
    while v < MAILBOX_END {
        let pte = v | 0x2; // phys == virtual (identity), attr 2 = RW device
        let pt_phys = (ptevaddr | (v >> 10)) & !3;
        bus.write_page_table_word(pt_phys, pte);
        v += 0x1000;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::firmware::Bus;
    use crate::firmware::xtensa::mmu::Mmu;

    #[test]
    fn install_makes_code_region_autorefill_translate() {
        let mut mmu = Mmu::new_with_varway56(true);
        let mut bus = Bus::new_with_load_offset(vec![0u8; 0x40000], 0x5c);
        mmu.ptevaddr = 0x3c00_0000;
        mmu.dtlbcfg = 0x0003_0000; // as the prologue sets it
        install(&mut mmu, &mut bus, 0x5c, 0x40000);

        // varway56 reset leaves way-6 entry 1 as an identity region covering
        // 0x20000000..0x3fffffff (which contains BOTH the code region AND the
        // 0x3c000000 PTEVADDR window). The real prologue invalidates it on both
        // the I- and D-side (iitlb+idtlb 0x20000006); mimic both here:
        //   - the I-side so the code fetch lookup MISSES (-> autorefill), and
        //   - the D-side so the autorefill walk's own get_pte lookup of the PTE
        //     address (0x3c08xxxx, inside the same identity region) does not
        //     MULTI-HIT against our way-4 PT region entry.
        mmu.invalidate_tlb(false, 0x2000_0006); // ITLB way-6 entry 1
        mmu.invalidate_tlb(true, 0x2000_0006); // DTLB way-6 entry 1

        // A fetch of virtual 0x20000340 must now autorefill from the synth PT to
        // phys 0x340 (page base 0 | offset 0x340).
        let t = mmu
            .translate(&mut bus, 0x2000_0340, 2 /*fetch*/, 0)
            .expect("autorefill from synth PT");
        assert_eq!(t.paddr, 0x0000_0340);
        // And a page further in maps linearly.
        let t2 = mmu.translate(&mut bus, 0x2000_1abc, 2, 0).expect("second page");
        assert_eq!(t2.paddr, 0x0000_1abc);
    }

    #[test]
    fn pte_word_encodes_phys_and_attr() {
        // The PTE for virtual page 0x20003000 is phys 0x3000 | attr 7.
        let mut mmu = Mmu::new_with_varway56(true);
        let mut bus = Bus::new_with_load_offset(vec![0u8; 0x40000], 0x5c);
        mmu.ptevaddr = 0x3c00_0000;
        install(&mut mmu, &mut bus, 0x5c, 0x40000);
        let pt_phys = (0x3c00_0000u32 | (0x2000_3000u32 >> 10)) & !3;
        assert_eq!(bus.load32(pt_phys), 0x0000_3000 | 0x7);
    }

    #[test]
    fn install_maps_mailbox_region_rw() {
        use crate::firmware::mmio::{MAILBOX_BASE, MAILBOX_END};
        let mut mmu = Mmu::new_with_varway56(true);
        let mut bus = Bus::new_with_load_offset(vec![0u8; 0x40000], 0x5c);
        mmu.ptevaddr = 0x3c00_0000;
        mmu.dtlbcfg = 0x0003_0000;
        install(&mut mmu, &mut bus, 0x5c, 0x40000);

        // Every mailbox page has an identity PTE with attr 2 (RW, non-exec).
        for v in (MAILBOX_BASE..MAILBOX_END).step_by(0x1000) {
            let pt_phys = (0x3c00_0000u32 | (v >> 10)) & !3;
            assert_eq!(
                bus.load32(pt_phys),
                v | 0x2,
                "mailbox page {v:#x} must map identity with attr 2 (RW device)"
            );
        }

        // And a store into the mailbox now translates+writes without faulting.
        // Invalidate way-6 entry 1 (0x20000000..0x3fffffff region) as the prologue
        // does, so the store falls to the synth-PT autorefill rather than the region.
        mmu.invalidate_tlb(true, 0x2000_0006);
        let t = mmu
            .translate(&mut bus, MAILBOX_BASE + 0x40, 1 /*store*/, 0)
            .expect("mailbox store must translate via synth PT (attr 2 grants write)");
        assert_eq!(t.paddr, MAILBOX_BASE + 0x40, "mailbox maps identity");
    }
}
