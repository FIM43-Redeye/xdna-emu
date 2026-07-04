# M2b Xtensa MMU-v3 Mechanism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a faithful Xtensa MMU-v3 (region-protection + hardware-autorefill paging) model in the firmware interpreter so instruction fetch and data load/store translate virtual addresses through TLBs and a page-table walk exactly as silicon does.

**Architecture:** A new `Mmu` struct (`src/firmware/xtensa/mmu.rs`), owned as a field on `Cpu`, holds the ITLB (7 ways) / DTLB (10 ways), the config registers, and the autorefill index. All translation flows through one `Cpu::translate(bus, vaddr, access)` chokepoint wired into fetch and every `mem.rs` load/store; `Bus` stays strictly physical (it only ever receives already-translated addresses). TLB faults raise through the existing `raise_general_exception` path, extended with the real MMU cause codes and a double-fault branch.

**Tech Stack:** Rust; the firmware interpreter under `src/firmware/xtensa/`. Test framework: `cargo test --lib`.

**Derivation oracle:** every formula/constant/cause code below is derived from QEMU 10.2.1 `target/xtensa/mmu_helper.c` (+ `cpu.h`, `exc_helper.c`, `overlay_tool.h`), at `scratchpad/qemu-src/qemu-10.2.1+ds/target/xtensa/`. The design spec is `docs/superpowers/specs/2026-07-04-m2b-mmu-mechanism-design.md`. Read it once for the full mechanism narrative; this plan is the task-by-task build.

## Global Constraints

- DERIVE FROM THE TOOLCHAIN. Every MMU formula/constant/cause code must match the QEMU `mmu_helper.c`/`cpu.h`/`exc_helper.c` source; cite the mechanism (not the tool) in comments, per the repo's source-derivation policy. Never guess a bit position.
- `Bus` (`src/firmware/mmio.rs`) and its `region()` classifier are UNCHANGED. After M2b, `Bus` receives ONLY physical addresses. Do not add translation logic inside `Bus`.
- `RegFile` (`regfile.rs`) is UNCHANGED (it is scoped to windowed-ABI register mechanics). All MMU state lives in the new `Mmu` struct on `Cpu`.
- Autorefill-only ways are 0-3 (fixed 4 KB). Way 4 and ways 5/6 are variable-page-size, software-installed. Ways 7-9 (DTLB only) are single-entry.
- MMU config constants (`ITLB_NWAYS=7`, `DTLB_NWAYS=10`, `MAX_TLB_WAY_SIZE=8`, `varway56=false`, autorefill way size / `nrefillentries`) are the standard MMU-v3 defaults, named so they become the AMD-Xtensa-config swap point later. The exact autorefill way size (`nrefillentries` 16-vs-32 / `is32`) is confirmed from source in Task 2.
- No emoji anywhere.
- TDD: failing test first, run to confirm it fails, minimal implementation, run to pass, commit. Run `cargo test --lib` (bare, never piped) before each commit.
- New EXCCAUSE values (`cpu.h:266-294`): InstTLBMiss=16, InstTLBMultiHit=17, InstFetchPrivilege=18, InstFetchProhibited=20, LoadStoreTLBMiss=24, LoadStoreTLBMultiHit=25, LoadStorePrivilege=26, LoadProhibited=28, StoreProhibited=29.
- Firmware runs kernel-only: current CPU ring (`mmu_idx`) is always 0; `PS.RING`/`PS.UM` are not modeled. Privilege checks are implemented faithfully but trivially pass at ring 0.

---

## File Structure

- **Create** `src/firmware/xtensa/mmu.rs` — `TlbEntry`, `Mmu`, `Access`, config constants, addressing helpers (`get_page_size`/`addr_mask`/`split_entry`), `lookup`, `get_ring`, `attr_to_access`, `install`/`invalidate`, the autorefill `translate`. All synthetic MMU unit tests. Wired via `mod mmu;` in `xtensa/mod.rs`.
- **Modify** `src/firmware/xtensa/mod.rs` — add `pub mod mmu;`.
- **Modify** `src/firmware/xtensa/interp/mod.rs` — add `mmu: Mmu` + `excvaddr: u32` to `Cpu`; add `Access`, `Cpu::translate`; new EXCCAUSE constants + SR constants; SR routing arms for PTEVADDR/RASID/ITLBCFG/DTLBCFG/EXCVADDR; double-fault + EXCVADDR in the raise path; wire fetch through `translate`.
- **Modify** `src/firmware/xtensa/interp/mem.rs` — route each load/store effective address (and `l32r` target) through `cpu.translate` before the bus call.
- **Modify** `src/firmware/xtensa/interp/system.rs` — replace `Witlb`/`Wdtlb`/`Iitlb`/`Idtlb` log-no-op bodies with real `Mmu` calls; route the `Wsr`/`Rsr` MMU-config SRs.
- **Modify** `src/firmware/mod.rs` — characterization logging hook for Task 10 (only if needed).

**Task dependency chain:** 1 → 2 → 3 → (4,5) → 6 → 7 → (8,9) → 10. Each task ends with an independently testable deliverable and a commit.

---

### Task 1: `Mmu` struct, `TlbEntry`, reset state

**Files:**
- Create: `src/firmware/xtensa/mmu.rs`
- Modify: `src/firmware/xtensa/mod.rs` (add `pub mod mmu;`)

**Interfaces:**
- Produces: `struct TlbEntry { vaddr: u32, paddr: u32, asid: u8, attr: u8, variable: bool }`; `struct Mmu` with fields `itlb: [[TlbEntry; 8]; 7]`, `dtlb: [[TlbEntry; 8]; 10]`, `ptevaddr: u32`, `rasid: u32`, `itlbcfg: u32`, `dtlbcfg: u32`, `autorefill_idx: u32`; `impl Mmu { pub fn new() -> Self }`. Constants `ITLB_NWAYS: usize = 7`, `DTLB_NWAYS: usize = 10`, `MAX_TLB_WAY_SIZE: usize = 8`.

- [ ] **Step 1: Write the failing test** (append the `mod tests` at the bottom of the new file)

```rust
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
        for tlb in [&mmu.itlb, &mmu.dtlb] {
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib firmware::xtensa::mmu 2>&1 | tail -20`
Expected: compile error (module/type not defined).

- [ ] **Step 3: Write minimal implementation** (top of `mmu.rs`)

```rust
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
```

Add to `src/firmware/xtensa/mod.rs` (next to the other `pub mod` lines):

```rust
pub mod mmu;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib firmware::xtensa::mmu 2>&1 | tail -20`
Expected: PASS (2 tests). Then `cargo test --lib 2>&1 | tail -5` — no regression.

- [ ] **Step 5: Commit**

```bash
git add src/firmware/xtensa/mmu.rs src/firmware/xtensa/mod.rs
git commit -m "feat(#140): M2b Task 1 -- Mmu struct + TlbEntry + MMU-v3 reset state"
```

---

### Task 2: addressing helpers — `way_size`, `get_page_size`, `addr_mask`, `split_entry`

**Files:**
- Modify: `src/firmware/xtensa/mmu.rs`

**Interfaces:**
- Consumes: `Mmu`, `ITLB_NWAYS`/`DTLB_NWAYS` (Task 1).
- Produces (methods on `Mmu`):
  - `fn way_size(&self, dtlb: bool, wi: usize) -> usize`
  - `fn get_page_size(&self, dtlb: bool, wi: usize) -> u32`
  - `fn addr_mask(&self, dtlb: bool, wi: usize) -> u32`
  - `fn split_entry(&self, vaddr: u32, dtlb: bool, wi: usize) -> (u32 /*vpn*/, usize /*ei*/)`
  - const `AUTOREFILL_WAY_SIZE: usize` (with `NREFILLENTRIES`), documented as the config-derived autorefill way size.

**Derivation:** `xtensa_tlb_get_addr_mask` (`mmu_helper.c:109-141`), `get_page_size` (`mmu_helper.c:86-104`), `split_tlb_entry_spec_way` (`mmu_helper.c:176-226`). `varway56=false` throughout, so the `varway56` branches collapse to the else-arms.

- [ ] **Step 1: Write the failing tests** (add to `mod tests`)

```rust
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib firmware::xtensa::mmu 2>&1 | tail -20`
Expected: FAIL (methods undefined).

- [ ] **Step 3: Write the implementation** (add to `impl Mmu`)

```rust
/// Autorefill way size (entries per autorefill way 0-3) and its ×4
/// `nrefillentries`. MMU-v3 default (`overlay_tool.h` `TLB_TEMPLATE` with
/// `XCHAL_*TLB_ARF_ENTRIES_LOG2`). CONFIRM against the QEMU core configs at
/// implementation time: grep `scratchpad/qemu-src/.../target/xtensa/core-*/`
/// for `XCHAL_ITLB_ARF_ENTRIES_LOG2` / `XCHAL_DTLB_ARF_ENTRIES_LOG2`. The
/// `is32` split in `split_entry` (below) depends only on whether this is 32.
/// LOG2=2 -> way size 4, nrefillentries 16, is32=false (the value used here).
pub const AUTOREFILL_WAY_SIZE: usize = 4;
const NREFILLENTRIES: u32 = 16;

/// Entries actually addressable in way `wi` (`overlay_tool.h` `way_size[]`,
/// varway56=false): ways 0-3 = AUTOREFILL_WAY_SIZE, way 4 = 4, ways 5/6 = 2,
/// ways 7-9 (DTLB) = 1.
fn way_size(&self, dtlb: bool, wi: usize) -> usize {
    let nways = if dtlb { DTLB_NWAYS } else { ITLB_NWAYS };
    debug_assert!(wi < nways);
    match wi {
        0..=3 => Self::AUTOREFILL_WAY_SIZE,
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
        let is32 = Self::NREFILLENTRIES == 32;
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --lib firmware::xtensa::mmu 2>&1 | tail -20`
Expected: PASS. Then confirm `AUTOREFILL_WAY_SIZE`/`NREFILLENTRIES` against the QEMU core configs and adjust if the AMD-relevant config differs (record the grep result in the commit message). Then `cargo test --lib 2>&1 | tail -5`.

- [ ] **Step 5: Commit**

```bash
git add src/firmware/xtensa/mmu.rs
git commit -m "feat(#140): M2b Task 2 -- TLB addressing helpers (way_size/page_size/mask/split)"
```

---

### Task 3: TLB lookup, `get_ring`, multi-hit

**Files:**
- Modify: `src/firmware/xtensa/mmu.rs`

**Interfaces:**
- Consumes: `split_entry`, `way_size` (Task 2), `rasid` (Task 1).
- Produces:
  - `fn get_ring(&self, asid: u8) -> u32` (0..4, or 0xff if no RASID lane matches)
  - `pub fn lookup(&self, vaddr: u32, dtlb: bool) -> Result<TlbHit, u32>` where `pub struct TlbHit { pub wi: usize, pub ei: usize, pub ring: u32 }` and the `Err(u32)` is the cause code (`INST_TLB_MISS`=16 / `LOAD_STORE_TLB_MISS`=24, or the multi-hit causes 17/25).

**Derivation:** `xtensa_tlb_lookup` (`mmu_helper.c:463-495`), `get_ring` (`mmu_helper.c:443-452`).

- [ ] **Step 1: Write the failing tests**

```rust
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib firmware::xtensa::mmu 2>&1 | tail -20`
Expected: FAIL (`lookup`/`get_ring`/`TlbHit` undefined).

- [ ] **Step 3: Write the implementation**

```rust
/// A resolved TLB hit: which way/entry, and the ring (0-3) the page belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TlbHit {
    pub wi: usize,
    pub ei: usize,
    pub ring: u32,
}

impl Mmu {
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
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --lib firmware::xtensa::mmu 2>&1 | tail -20`
Expected: PASS. Then `cargo test --lib 2>&1 | tail -5`.

- [ ] **Step 5: Commit**

```bash
git add src/firmware/xtensa/mmu.rs
git commit -m "feat(#140): M2b Task 3 -- TLB lookup + get_ring + multi-hit detection"
```

---

### Task 4: `witlb`/`wdtlb`/`iitlb`/`idtlb` install/invalidate + config-SR routing

**Files:**
- Modify: `src/firmware/xtensa/mmu.rs`
- Modify: `src/firmware/xtensa/interp/mod.rs` (SR constants + `write_sr`/`read_sr` arms)
- Modify: `src/firmware/xtensa/interp/system.rs` (replace the four log-no-op arms)

**Interfaces:**
- Consumes: `split_entry` (Task 2), `Mmu` fields (Task 1). `Cpu` has `pub mmu: Mmu` — ADD this field now (default `Mmu::new()` in `Cpu::new`).
- Produces (methods on `Mmu`):
  - `pub fn write_tlb(&mut self, dtlb: bool, at: u32 /*paddr|attr*/, as_: u32 /*way+VPN*/)`
  - `pub fn invalidate_tlb(&mut self, dtlb: bool, as_: u32)`
  - `pub fn write_rasid(&mut self, v: u32)` (forces ring-0 byte to 1)

**Derivation:** `HELPER(wtlb)` + `split_tlb_entry_spec` (`mmu_helper.c:562-570, 232-249`), `HELPER(itlb)` (`mmu_helper.c:524-534`), `xtensa_tlb_set_entry` variable-guard (`mmu_helper.c:290-317`), `wsr_rasid` (`mmu_helper.c:77-84`). PTE decode `xtensa_tlb_set_entry_mmu` (`mmu_helper.c:279-288`): `paddr = at & addr_mask`, `attr = at & 0xf`, `ring = (at>>4)&3`, `asid = RASID_byte[ring]`.

- [ ] **Step 1: Add the `mmu` field to `Cpu`**

In `interp/mod.rs`, add to `struct Cpu` and `Cpu::new`:

```rust
// in `pub struct Cpu { ... }` add:
    /// Xtensa MMU-v3 state (TLBs + config regs). Translation flows through
    /// `Cpu::translate`; `Bus` only ever sees physical addresses.
    pub mmu: super::mmu::Mmu,

// in `Cpu::new`, extend the struct literal:
    Self { pc: entry, regs: RegFile::new(), vecbase: 0, epc1: 0, mmu: super::mmu::Mmu::new() }
```

- [ ] **Step 2: Write the failing tests** (in `mmu.rs` `mod tests`, plus one in `system.rs`)

```rust
// mmu.rs
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
```

```rust
// system.rs mod tests -- prove the Op arms mutate real TLB state now. Drive
// system::exec directly with a constructed Op (no ROM/decode-vector needed --
// this test is about the exec arm, not decode).
#[test]
fn witlb_installs_into_the_mmu() {
    use super::exec;
    use crate::firmware::mmio::Bus;
    use crate::firmware::xtensa::decode::Op;
    let mut cpu = Cpu::new(0);
    let mut bus = Bus::new(vec![0u8; 16]);
    // witlb a3,a4: AS = AR[4] (way 0 + VPN), AT = AR[3] (paddr|attr).
    cpu.regs.write_ar(4, 0x4000_1000 | 0);
    cpu.regs.write_ar(3, 0x0009_0000 | 0x3);
    let step = exec(&mut cpu, &mut bus, &Op::Witlb { t: 3, s: 4 }, 0, 3);
    assert!(matches!(step, Some(Step::Ran)));
    // The entry is now installed and resolvable.
    assert!(cpu.mmu.lookup(0x4000_1abc, false).is_ok());
}
```

Note to implementer: this drives `system::exec` directly with `Op::Witlb { t: 3, s: 4 }`, matching how the existing `system.rs` exec tests exercise arms without a full decode round-trip. `exec` returns `Option<Step>`; import it via `use super::exec`. The binding assertion is the post-install `lookup(..).is_ok()`. Add a parallel `idtlb_invalidates` test with `Op::Idtlb { s: 4 }` after installing.

- [ ] **Step 3: Run to verify it fails**

Run: `cargo test --lib firmware::xtensa 2>&1 | tail -20`
Expected: FAIL (`write_tlb`/`invalidate_tlb`/`write_rasid` undefined; system arms still no-op).

- [ ] **Step 4: Write the implementation**

In `mmu.rs`:

```rust
impl Mmu {
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
        let ring = (at >> 4) & 0x3;
        let asid = ((self.rasid >> (ring * 8)) & 0xff) as u8;
        let entry = TlbEntry {
            vaddr: vpn,
            paddr: at & self.addr_mask(dtlb, wi),
            asid,
            attr: (at & 0xf) as u8,
            variable: true,
        };
        let slot = &mut (if dtlb { &mut self.dtlb[..] } else { &mut self.itlb[..] })[wi][ei];
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
        let slot = &mut (if dtlb { &mut self.dtlb[..] } else { &mut self.itlb[..] })[wi][ei];
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
```

In `interp/mod.rs`, add SR constants (near the existing SR consts) and route them. SR numbers: PTEVADDR=0x53, RASID=0x5A, ITLBCFG=0x5B, DTLBCFG=0x5C (QEMU sregs indices 83/90/91/92; confirm ITLBCFG/DTLBCFG/PTEVADDR against `decode/system.rs`'s existing decode tests, which already assert them):

```rust
/// MMU config special registers (QEMU `cpu.h` sregs indices; PTEVADDR/ITLBCFG/
/// DTLBCFG cross-checked against decode/system.rs's decode tests).
const SR_PTEVADDR: u8 = 0x53;
const SR_RASID: u8 = 0x5A;
const SR_ITLBCFG: u8 = 0x5B;
const SR_DTLBCFG: u8 = 0x5C;
```

Add arms to `write_sr` (before the `_ =>` catch-all):

```rust
            SR_PTEVADDR => self.mmu.ptevaddr = value,
            SR_RASID => self.mmu.write_rasid(value),
            SR_ITLBCFG => self.mmu.itlbcfg = value,
            SR_DTLBCFG => self.mmu.dtlbcfg = value,
```

Add arms to `read_sr` (before the `_ =>` catch-all):

```rust
            SR_PTEVADDR => self.mmu.ptevaddr,
            SR_RASID => self.mmu.rasid,
            SR_ITLBCFG => self.mmu.itlbcfg,
            SR_DTLBCFG => self.mmu.dtlbcfg,
```

In `interp/system.rs`, replace the four log-no-op arm bodies:

```rust
        Op::Witlb { t, s } => {
            let as_ = cpu.regs.read_ar(*s);
            let at = cpu.regs.read_ar(*t);
            cpu.mmu.write_tlb(false, at, as_);
        }
        Op::Wdtlb { t, s } => {
            let as_ = cpu.regs.read_ar(*s);
            let at = cpu.regs.read_ar(*t);
            cpu.mmu.write_tlb(true, at, as_);
        }
        Op::Iitlb { s } => {
            let as_ = cpu.regs.read_ar(*s);
            cpu.mmu.invalidate_tlb(false, as_);
        }
        Op::Idtlb { s } => {
            let as_ = cpu.regs.read_ar(*s);
            cpu.mmu.invalidate_tlb(true, as_);
        }
```

- [ ] **Step 5: Run to verify it passes**

Run: `cargo test --lib firmware::xtensa 2>&1 | tail -20`
Expected: PASS. The existing `witlb_and_isync_are_logged_no_ops` / `wdtlb_iitlb_idtlb_dsync_are_logged_no_ops` tests in `system.rs` will now FAIL (behavior changed from no-op to real install) — UPDATE them to assert the new real TLB mutation (that is the intended behavior change; rename them, e.g. `witlb_installs_a_tlb_entry`). Then `cargo test --lib 2>&1 | tail -5`.

- [ ] **Step 6: Commit**

```bash
git add src/firmware/xtensa/mmu.rs src/firmware/xtensa/interp/mod.rs src/firmware/xtensa/interp/system.rs
git commit -m "feat(#140): M2b Task 4 -- witlb/wdtlb/iitlb/idtlb install + config-SR routing"
```

---

### Task 5: attribute-nibble permission decode

**Files:**
- Modify: `src/firmware/xtensa/mmu.rs`

**Interfaces:**
- Produces: `bitflags`-free access bits as `const PAGE_READ/PAGE_WRITE/PAGE_EXEC: u32` and `fn attr_to_access(attr: u8) -> u32`; `fn access_granted(access: u32, is_write: u8) -> bool`. (`is_write`: 0=load, 1=store, 2=fetch, matching QEMU.)

**Derivation:** `mmu_attr_to_access` (`mmu_helper.c:576-606`), `is_access_granted` (search `mmu_helper.c` for the PAGE_* checks in `get_physical_addr_mmu`, `mmu_helper.c:852-861`). We model only READ/WRITE/EXEC (cache-policy bits are behaviorally irrelevant to this single-memory interpreter).

- [ ] **Step 1: Write the failing tests**

```rust
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
```

- [ ] **Step 2: Run to verify it fails.** `cargo test --lib firmware::xtensa::mmu 2>&1 | tail -20` — FAIL.

- [ ] **Step 3: Write the implementation**

```rust
/// Access permission bits (subset of QEMU's PAGE_* we model — cache-policy
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
```

- [ ] **Step 4: Run to verify it passes.** `cargo test --lib firmware::xtensa::mmu 2>&1 | tail -20` — PASS. `cargo test --lib 2>&1 | tail -5`.

- [ ] **Step 5: Commit**

```bash
git add src/firmware/xtensa/mmu.rs
git commit -m "feat(#140): M2b Task 5 -- attribute-nibble permission decode"
```

---

### Task 6: autorefill page-table walk + full MMU translate

**Files:**
- Modify: `src/firmware/xtensa/mmu.rs`

**Interfaces:**
- Consumes: `lookup` (Task 3), `attr_to_access`/`access_granted` (Task 5), `split_entry`/`addr_mask` (Task 2). Needs `&mut Bus` for the PTE load — import `use crate::firmware::Bus;`.
- Produces: `pub struct Translation { pub paddr: u32, pub page_size: u32 }` and
  `pub fn translate(&mut self, bus: &mut Bus, vaddr: u32, is_write: u8, mmu_idx: u32) -> Result<Translation, MmuFault>` where `pub struct MmuFault { pub cause: u32, pub vaddr: u32 }` (vaddr is the faulting address for EXCVADDR).

**Derivation:** `get_physical_addr_mmu` (`mmu_helper.c:807-868`), `get_pte` (`mmu_helper.c:870-904`). `dtlb = is_write != 2`. PTE address `(ptevaddr | (vaddr>>10)) & !3`. PTE decode: `ring=(pte>>4)&3`, `attr=pte&0xf`, `paddr=pte & addr_mask(way0)`, `asid=RASID_byte[ring]`. Refill way `(++autorefill_idx)&3`. Recursion guard: the PTE's own translation runs with `may_lookup_pt=false`.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn autorefill_walks_synthetic_page_table() {
    use crate::firmware::mmio::Bus;
    // Lay a page table in RAM (RAM aperture 0x08b00000..). PTEVADDR points at a
    // physical base that our fixed/installed TLB can translate to that RAM.
    // Simplest faithful setup: install a static DTLB region entry mapping the
    // PTEVADDR region identity into RAM, then place the PTE for our target VPN.
    let mut mmu = Mmu::new();
    let mut bus = Bus::new(vec![0u8; 16]); // tiny ROM; RAM grows on write

    // Target: fetch vaddr 0x20000340. PTE addr = (PTEVADDR | (vaddr>>10)) & ~3.
    let ptevaddr = 0x08c0_0000; // a base inside the RAM aperture
    mmu.ptevaddr = ptevaddr;
    let pte_addr = (ptevaddr | (0x2000_0340u32 >> 10)) & !3;
    // PTE: paddr 0x08b0_5000, attr 1 (R+X), ring 0 (bits[5:4]=0).
    let pte = 0x08b0_5000u32 | 0x1;
    bus.store32(pte_addr, pte);

    // The PTE's own address must be translatable WITHOUT autorefill: install a
    // static DTLB entry (way 4, large page) covering the PTEVADDR region
    // identity-mapped. Use write_tlb with a way-4 AS.
    // way4 page_size 0 -> mask 0xfff00000 (1MB). Map 0x08c00000 -> 0x08c00000.
    mmu.dtlbcfg = 0; // way4 page size selector 0
    mmu.write_tlb(true, 0x08c0_0000 | 0x3 /*RW*/, 0x08c0_0000 | 4 /*way 4*/);

    // Now an ITLB miss on 0x20000340 should autorefill from the PTE.
    let t = mmu.translate(&mut bus, 0x2000_0340, 2 /*fetch*/, 0).expect("autorefill");
    assert_eq!(t.paddr, 0x08b0_5000 | 0x340); // page base | offset
    // The refilled entry now lives in an autorefill way (0-3).
    let hit = mmu.lookup(0x2000_0340, false).expect("now resident");
    assert!(hit.wi < 4, "refilled into an autorefill way");
}

#[test]
fn autorefill_round_robins_ways() {
    use crate::firmware::mmio::Bus;
    let mut mmu = Mmu::new();
    let mut bus = Bus::new(vec![0u8; 16]);
    let ptevaddr = 0x08c0_0000;
    mmu.ptevaddr = ptevaddr;
    mmu.write_tlb(true, 0x08c0_0000 | 0x3, 0x08c0_0000 | 4);
    // Two different VPNs -> two refills -> autorefill_idx advances 1 then 2.
    for (vaddr, ppage) in [(0x2000_0000u32, 0x08b0_1000u32), (0x2100_0000u32, 0x08b0_2000u32)] {
        let pte_addr = (ptevaddr | (vaddr >> 10)) & !3;
        bus.store32(pte_addr, ppage | 0x1);
        mmu.translate(&mut bus, vaddr, 2, 0).expect("refill");
    }
    assert_eq!(mmu.autorefill_idx, 2);
}

#[test]
fn autorefill_miss_on_pt_yields_original_cause() {
    use crate::firmware::mmio::Bus;
    // If the PTE's own address can't be translated (no covering entry), the
    // original TLB-miss cause stands, NOT a nested fault (get_pte returns
    // false -> outer ret unchanged; mmu_helper.c:815-833).
    let mut mmu = Mmu::new();
    let mut bus = Bus::new(vec![0u8; 16]);
    mmu.ptevaddr = 0x2000_0000; // region NOT covered by any TLB entry
    let fault = mmu.translate(&mut bus, 0x2000_0340, 2, 0).unwrap_err();
    assert_eq!(fault.cause, 16); // INST_TLB_MISS, not a nested cause
    assert_eq!(fault.vaddr, 0x2000_0340);
}

#[test]
fn prohibited_when_permission_absent() {
    use crate::firmware::mmio::Bus;
    let mut mmu = Mmu::new();
    let mut bus = Bus::new(vec![0u8; 16]);
    // Install a read-only page directly (attr 0 = R only), then attempt a store.
    mmu.write_tlb(true, 0x0009_0000 | 0x0 /*attr0 = R*/, 0x4000_1000 | 0 /*way0*/);
    let fault = mmu.translate(&mut bus, 0x4000_1abc, 1 /*store*/, 0).unwrap_err();
    assert_eq!(fault.cause, 29); // STORE_PROHIBITED
}
```

- [ ] **Step 2: Run to verify it fails.** `cargo test --lib firmware::xtensa::mmu 2>&1 | tail -20` — FAIL.

- [ ] **Step 3: Write the implementation**

```rust
use crate::firmware::Bus;

/// A successful translation: physical address + the page size (bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Translation {
    pub paddr: u32,
    pub page_size: u32,
}

/// A translation fault: architectural EXCCAUSE code + the faulting vaddr
/// (for EXCVADDR).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MmuFault {
    pub cause: u32,
    pub vaddr: u32,
}

impl Mmu {
    /// Full-MMU translation with hardware autorefill (`get_physical_addr_mmu`,
    /// `mmu_helper.c:807-868`). `is_write`: 0=load, 1=store, 2=fetch.
    /// `mmu_idx` = current CPU ring (0 for this kernel firmware).
    pub fn translate(
        &mut self,
        bus: &mut Bus,
        vaddr: u32,
        is_write: u8,
        mmu_idx: u32,
    ) -> Result<Translation, MmuFault> {
        self.translate_inner(bus, vaddr, is_write, mmu_idx, true)
    }

    fn translate_inner(
        &mut self,
        bus: &mut Bus,
        vaddr: u32,
        is_write: u8,
        mmu_idx: u32,
        may_lookup_pt: bool,
    ) -> Result<Translation, MmuFault> {
        let dtlb = is_write != 2;
        let fault = |cause| MmuFault { cause, vaddr };

        let hit = match self.lookup(vaddr, dtlb) {
            Ok(hit) => hit,
            Err(cause) => {
                let is_miss = cause == 16 || cause == 24;
                if is_miss && may_lookup_pt {
                    if let Some(pte) = self.get_pte(bus, vaddr) {
                        self.refill(vaddr, dtlb, pte)
                    } else {
                        return Err(fault(cause)); // original miss stands
                    }
                } else {
                    return Err(fault(cause)); // multi-hit, or a PT-walk miss
                }
            }
        };

        let (paddr_base, attr, ring, wi) = {
            let e = &(if dtlb { &self.dtlb[..] } else { &self.itlb[..] })[hit.wi][hit.ei];
            (e.paddr, e.attr, hit.ring, hit.wi)
        };

        // Privilege: a page more privileged than the current ring faults.
        if ring < mmu_idx {
            return Err(fault(if dtlb { 26 } else { 18 })); // *_PRIVILEGE
        }
        // Permission: attr decodes to R/W/X; fetch needs X, store needs W, load R.
        if !access_granted(attr_to_access(attr), is_write) {
            let cause = match (dtlb, is_write) {
                (false, _) => 20, // INST_FETCH_PROHIBITED
                (true, 1) => 29,  // STORE_PROHIBITED
                (true, _) => 28,  // LOAD_PROHIBITED
            };
            return Err(fault(cause));
        }
        let mask = self.addr_mask(dtlb, wi);
        Ok(Translation {
            paddr: paddr_base | (vaddr & !mask),
            page_size: !mask + 1,
        })
    }

    /// Fetch the PTE for `vaddr` via the page-table walk (`get_pte`,
    /// `mmu_helper.c:870-904`). The PTE address `(ptevaddr | (vaddr>>10)) & ~3`
    /// is itself translated, but with `may_lookup_pt=false` so it cannot
    /// trigger a nested walk (recursion guard). Returns None if the PTE's own
    /// translation misses or the load can't be satisfied — the caller then
    /// keeps the original miss cause.
    fn get_pte(&mut self, bus: &mut Bus, vaddr: u32) -> Option<u32> {
        let pt_vaddr = (self.ptevaddr | (vaddr >> 10)) & !3;
        let t = self.translate_inner(bus, pt_vaddr, 0 /*load*/, 0 /*ring0*/, false).ok()?;
        Some(bus.load32(t.paddr))
    }

    /// Install an autorefilled PTE into the round-robin autorefill way
    /// (`get_physical_addr_mmu` refill block, `mmu_helper.c:824-833`). Returns
    /// the resulting TlbHit.
    fn refill(&mut self, vaddr: u32, dtlb: bool, pte: u32) -> TlbHit {
        let ring = (pte >> 4) & 0x3;
        let asid = ((self.rasid >> (ring * 8)) & 0xff) as u8;
        self.autorefill_idx = self.autorefill_idx.wrapping_add(1);
        let wi = (self.autorefill_idx & 0x3) as usize;
        let (vpn, ei) = self.split_entry(vaddr, dtlb, wi);
        let entry = TlbEntry {
            vaddr: vpn,
            paddr: pte & self.addr_mask(dtlb, wi),
            asid,
            attr: (pte & 0xf) as u8,
            variable: true,
        };
        if dtlb {
            self.dtlb[wi][ei] = entry;
        } else {
            self.itlb[wi][ei] = entry;
        }
        TlbHit { wi, ei, ring }
    }
}
```

Note to implementer: `refill` returns a `TlbHit` whose `ring` is the PTE's ring; the outer `translate_inner` re-reads the freshly installed entry via `hit.wi/hit.ei`, so the borrow ends before the permission checks. Confirm the borrow checker is satisfied (the `match` returns `hit` by value). If `get_pte`'s recursion trips a borrow conflict, restructure `refill` to take the entry fields rather than returning a borrow — it already returns by value, so this should compile as written.

- [ ] **Step 4: Run to verify it passes.** `cargo test --lib firmware::xtensa::mmu 2>&1 | tail -20` — PASS (4 tests). `cargo test --lib 2>&1 | tail -5`.

- [ ] **Step 5: Commit**

```bash
git add src/firmware/xtensa/mmu.rs
git commit -m "feat(#140): M2b Task 6 -- autorefill page-table walk + full MMU translate"
```

---

### Task 7: `Cpu::translate` seam, EXCCAUSE/EXCVADDR, fault raising + double-fault

**Files:**
- Modify: `src/firmware/xtensa/interp/mod.rs`

**Interfaces:**
- Consumes: `Mmu::translate` + `MmuFault` (Task 6), the `mmu` field (Task 4).
- Produces:
  - `pub enum Access { Fetch, Load, Store }`
  - `pub fn translate(&mut self, bus: &mut Bus, vaddr: u32, access: Access) -> Result<u32, Step>` on `Cpu`
  - new EXCCAUSE constants; `excvaddr: u32` field on `Cpu` + `SR_EXCVADDR` routing
  - a double-fault branch + EXCVADDR set in `raise_general_exception`

**Derivation:** `exc_helper.c:48-76` (double-fault: if PS.EXCM already set, vector to EXC_DOUBLE = VECBASE+0x3C0; set EXCVADDR). `EXC_DOUBLE` offset 0x3C0 (cross-checked across QEMU core configs, same technique as `KERNEL_EXCEPTION_VECTOR_OFFSET`).

- [ ] **Step 1: Write the failing tests** (in `interp/mod.rs` `mod tests`)

```rust
#[test]
fn translate_returns_paddr_on_hit() {
    use crate::firmware::mmio::Bus;
    let mut cpu = Cpu::new(0);
    let mut bus = Bus::new(vec![0u8; 16]);
    // Install a DTLB mapping and translate a load through it.
    cpu.mmu.write_tlb(true, 0x0009_0000 | 0x3, 0x4000_1000 | 0);
    let paddr = cpu.translate(&mut bus, 0x4000_1abc, Access::Load).expect("hit");
    assert_eq!(paddr, 0x0009_0abc);
}

#[test]
fn translate_raises_itlb_miss_as_exception() {
    use crate::firmware::mmio::Bus;
    let mut cpu = Cpu::new(0);
    cpu.vecbase = 0x4000_0000;
    let mut bus = Bus::new(vec![0u8; 16]);
    // No mapping, no page table -> ITLB miss -> Step::Exception at kernel vector.
    let err = cpu.translate(&mut bus, 0x2000_0340, Access::Fetch).unwrap_err();
    match err {
        Step::Exception { cause, pc } => {
            assert_eq!(cause, 16); // INST_TLB_MISS
            assert_eq!(pc, 0x4000_0000 + 0x300); // kernel vector
        }
        other => panic!("expected Exception, got {:?}", other),
    }
    assert_eq!(cpu.regs.exccause, 16);
    assert_eq!(cpu.excvaddr, 0x2000_0340);
    assert_eq!(cpu.epc1, 0x2000_0340);
}

#[test]
fn double_fault_vectors_to_0x3c0() {
    use crate::firmware::mmio::Bus;
    let mut cpu = Cpu::new(0);
    cpu.vecbase = 0x4000_0000;
    cpu.regs.set_excm(); // already in exception mode
    let mut bus = Bus::new(vec![0u8; 16]);
    let err = cpu.translate(&mut bus, 0x2000_0340, Access::Fetch).unwrap_err();
    match err {
        Step::Exception { pc, .. } => assert_eq!(pc, 0x4000_0000 + 0x3C0),
        other => panic!("expected Exception, got {:?}", other),
    }
}
```

- [ ] **Step 2: Run to verify it fails.** `cargo test --lib firmware::xtensa::interp 2>&1 | tail -20` — FAIL.

- [ ] **Step 3: Write the implementation**

Add the EXCCAUSE constants near the existing ones:

```rust
/// MMU-fault EXCCAUSE values (`cpu.h:266-294`). Derived from QEMU; these are
/// the architectural cause codes a TLB miss/multi-hit/privilege/prohibited
/// fault reports through the same EXCCAUSE channel as syscall/divide-by-zero.
pub const EXCCAUSE_INST_TLB_MISS: u32 = 16;
pub const EXCCAUSE_INST_TLB_MULTI_HIT: u32 = 17;
pub const EXCCAUSE_INST_FETCH_PRIVILEGE: u32 = 18;
pub const EXCCAUSE_INST_FETCH_PROHIBITED: u32 = 20;
pub const EXCCAUSE_LOAD_STORE_TLB_MISS: u32 = 24;
pub const EXCCAUSE_LOAD_STORE_TLB_MULTI_HIT: u32 = 25;
pub const EXCCAUSE_LOAD_STORE_PRIVILEGE: u32 = 26;
pub const EXCCAUSE_LOAD_PROHIBITED: u32 = 28;
pub const EXCCAUSE_STORE_PROHIBITED: u32 = 29;

/// VECBASE-relative offset of the DoubleExceptionVector (`EXC_DOUBLE`): a
/// fault raised while PS.EXCM is already set vectors here instead of the
/// kernel/user vector (`exc_helper.c:56-58`). 0x3C0 cross-checked across the
/// QEMU core configs (same technique as KERNEL_EXCEPTION_VECTOR_OFFSET).
const DOUBLE_EXCEPTION_VECTOR_OFFSET: u32 = 0x3C0;

/// EXCVADDR special register (`cpu.h` sregs index 238 = 0xEE).
const SR_EXCVADDR: u8 = 0xEE;
```

Add `excvaddr` to `Cpu` and `Cpu::new`:

```rust
    /// Faulting virtual address of the most recent load/store/fetch fault
    /// (Xtensa EXCVADDR); set by `translate` before raising (`exc_helper.c:73`).
    pub excvaddr: u32,
// Cpu::new literal gains: excvaddr: 0,
```

Add `Access` and `translate`, and extend the raise path. Replace `raise_general_exception` with the double-fault-aware version and add an EXCVADDR setter path:

```rust
/// Which access class a translation is for — selects ITLB vs DTLB and the
/// permission subset checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Fetch,
    Load,
    Store,
}

impl Cpu {
    /// Translate a virtual address to physical for `access`, or return a
    /// ready-to-propagate `Step::Exception` on a TLB/permission fault. The one
    /// chokepoint for all address translation; `Bus` downstream sees only
    /// physical addresses. mmu_idx is 0 (kernel-only firmware).
    pub fn translate(&mut self, bus: &mut Bus, vaddr: u32, access: Access) -> Result<u32, Step> {
        let is_write = match access {
            Access::Fetch => 2u8,
            Access::Load => 0,
            Access::Store => 1,
        };
        match self.mmu.translate(bus, vaddr, is_write, 0) {
            Ok(t) => Ok(t.paddr),
            Err(fault) => {
                self.excvaddr = fault.vaddr;
                Err(self.raise_general_exception(self.pc, fault.cause))
            }
        }
    }
}
```

Update `raise_general_exception` to branch on PS.EXCM (double-fault):

```rust
    fn raise_general_exception(&mut self, faulting_pc: u32, cause: u32) -> Step {
        self.regs.exccause = cause;
        self.epc1 = faulting_pc;
        let offset = if self.regs.excm() {
            // Fault while already in exception mode -> DoubleExceptionVector
            // (exc_helper.c:56-58). Closes M2a carry-forward 9a.
            DOUBLE_EXCEPTION_VECTOR_OFFSET
        } else {
            KERNEL_EXCEPTION_VECTOR_OFFSET
        };
        self.regs.set_excm();
        let vector = self.vecbase.wrapping_add(offset);
        self.pc = vector;
        Step::Exception { cause, pc: vector }
    }
```

Add `SR_EXCVADDR` arms to `write_sr`/`read_sr`:

```rust
// write_sr: SR_EXCVADDR => self.excvaddr = value,
// read_sr:  SR_EXCVADDR => self.excvaddr,
```

- [ ] **Step 4: Run to verify it passes.** `cargo test --lib firmware::xtensa::interp 2>&1 | tail -20` — PASS. Then check whether any existing `raise_general_exception` test (syscall/divide-by-zero) asserted vectoring while EXCM was preset — the double-fault change only triggers when EXCM is ALREADY set at raise time, which those tests do not do, so they must still pass. `cargo test --lib 2>&1 | tail -5`.

- [ ] **Step 5: Commit**

```bash
git add src/firmware/xtensa/interp/mod.rs
git commit -m "feat(#140): M2b Task 7 -- Cpu::translate seam + MMU EXCCAUSE + double-fault (9a)"
```

---

### Task 8: wire fetch through translation (page-safe)

**Files:**
- Modify: `src/firmware/xtensa/interp/mod.rs` (`Cpu::step`)

**Interfaces:**
- Consumes: `Cpu::translate` + `Access::Fetch` (Task 7).

**Derivation:** instruction length from `op0 = b0 & 0xF` (`decode/mod.rs:1046-1051`): narrow (2 bytes) if `op0 ∈ 0x8..=0xD`, 1 byte if `op0 ∈ {0xE,0xF}` (Unknown), else 3 bytes. Fetch only the bytes the instruction occupies — never fault on a speculative byte past the real length.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn fetch_translates_through_itlb() {
    use crate::firmware::mmio::Bus;
    // Map virtual code page 0x20000000 -> physical ROM 0, execute a nop.n there.
    // nop.n = `3d f0` (VERIFIED vector from op-vectors; op0=0xD narrow, 2 bytes,
    // pure pc-advance -- ideal for a fetch-path test).
    let mut rom = vec![0u8; 16];
    rom[0] = 0x3d;
    rom[1] = 0xf0;
    let mut bus = Bus::new(rom);
    let mut cpu = Cpu::new(0x2000_0000);
    // ITLB entry: VPN 0x20000000 -> paddr 0, attr 1 (R+X), way 0.
    cpu.mmu.write_tlb(false, 0x0000_0000 | 0x1, 0x2000_0000 | 0);
    assert!(matches!(cpu.step(&mut bus), Step::Ran));
    assert_eq!(cpu.pc, 0x2000_0002); // advanced in VIRTUAL space
}

#[test]
fn fetch_miss_raises_exception_not_unknown() {
    use crate::firmware::mmio::Bus;
    let mut bus = Bus::new(vec![0u8; 16]);
    let mut cpu = Cpu::new(0x2000_0340);
    cpu.vecbase = 0x4000_0000;
    // No ITLB mapping, no page table -> ITLB miss, NOT Step::Unknown.
    match cpu.step(&mut bus) {
        Step::Exception { cause, .. } => assert_eq!(cause, 16),
        other => panic!("expected ITLB-miss Exception, got {:?}", other),
    }
}

#[test]
fn narrow_op_at_page_end_does_not_fault_on_third_byte() {
    use crate::firmware::mmio::Bus;
    // A 2-byte op whose bytes are the last two of a 4KB page: the (unmapped)
    // next page must NOT be touched. Physical page 0 holds the op at 0xFFE.
    let mut rom = vec![0u8; 0x1000];
    rom[0xFFE] = 0x3d; // nop.n (VERIFIED vector)
    rom[0xFFF] = 0xf0;
    let mut bus = Bus::new(rom);
    let mut cpu = Cpu::new(0x2000_0FFE);
    // Map only virtual page 0x20000000 -> paddr 0 (R+X). The next virtual
    // page 0x20001000 is deliberately left unmapped.
    cpu.mmu.write_tlb(false, 0x0000_0000 | 0x1, 0x2000_0000 | 0);
    assert!(matches!(cpu.step(&mut bus), Step::Ran), "must not fault on the 3rd byte");
    assert_eq!(cpu.pc, 0x2000_1000);
}
```

- [ ] **Step 2: Run to verify it fails.** `cargo test --lib firmware::xtensa::interp 2>&1 | tail -20` — FAIL.

- [ ] **Step 3: Write the implementation** — replace the fetch block at the top of `Cpu::step`:

```rust
    pub fn step(&mut self, bus: &mut Bus) -> Step {
        let pc = self.pc;
        // Translate + fetch the first byte; its op0 nibble fixes the length,
        // so we translate/fetch ONLY the bytes the instruction occupies (never
        // faulting on a speculative byte in a possibly-unmapped next page).
        let phys0 = match self.translate(bus, pc, Access::Fetch) {
            Ok(p) => p,
            Err(step) => return step,
        };
        let b0 = bus.load8(phys0);
        let op0 = b0 & 0xF;
        let need: usize = if op0 == 0xE || op0 == 0xF {
            1
        } else if (0x8..=0xD).contains(&op0) {
            2
        } else {
            3
        };
        let mut buf = [b0, 0u8, 0u8];
        for i in 1..need {
            let phys_i = match self.translate(bus, pc.wrapping_add(i as u32), Access::Fetch) {
                Ok(p) => p,
                Err(step) => return step,
            };
            buf[i] = bus.load8(phys_i);
        }
        let decoded = decode::decode(&buf[..need], pc);
        if let Op::Unknown { word } = decoded.op {
            return Step::Unknown { pc, word };
        }
        let len = decoded.len;
        // ... unchanged dispatch chain + loop-back tail ...
```

Note to implementer: keep the rest of `step()` (the `mem::exec ... branch::exec` chain and the loop-back tail) exactly as it was. Only the fetch preamble changes. The existing `unknown_step_leaves_pc_unchanged_for_reinspection` and `loop_back_*` tests use `Cpu::new(0)` with no ITLB mapping — with translation now in the path, `translate(0, Fetch)` will MISS. Two options, pick the faithful one: (a) update those tests to install an identity ITLB mapping for their code page first (preferred — it exercises the real path), or (b) if a test's intent is purely decode/loop mechanics, install the mapping in a shared test helper. Do NOT special-case "no MMU" in `step` — the firmware always runs with the MMU on. Update every `interp`/`mem`/`arith`/`branch`/`control`/`system` test that calls `step()` to install an identity mapping for its ROM page; a small test helper `fn mapped_cpu(entry: u32) -> Cpu` that installs `write_tlb(false, attr R+X, entry_page | way0)` and `write_tlb(true, attr RW, ...)` for the data pages it touches keeps this DRY. This is the largest mechanical part of M2b — budget for touching many existing tests.

- [ ] **Step 4: Run to verify it passes.** Run the full firmware suite: `cargo test --lib firmware 2>&1 | tail -30`. Fix each `step()`-based test to map its pages. Expected: all green. Then `cargo test --lib 2>&1 | tail -5`.

- [ ] **Step 5: Commit**

```bash
git add src/firmware/xtensa/interp/mod.rs src/firmware/xtensa/interp/*.rs
git commit -m "feat(#140): M2b Task 8 -- fetch through MMU translation (page-safe length-aware)"
```

---

### Task 9: wire load/store through translation

**Files:**
- Modify: `src/firmware/xtensa/interp/mem.rs`

**Interfaces:**
- Consumes: `Cpu::translate` + `Access::{Load,Store}` (Task 7).

**Derivation:** every load/store effective address (and the `l32r` literal target) is a VIRTUAL address that must be translated to physical before the `Bus` call. Data faults return `Some(Step::Exception)` without advancing `pc`.

- [ ] **Step 1: Write the failing tests** (in `mem.rs` `mod tests`)

```rust
#[test]
fn load_translates_through_dtlb() {
    use crate::firmware::mmio::Bus;
    // l32i.n a4,a5,16 = `48 45` (VERIFIED vector from mem.rs's existing tests).
    // Reads from AR[5]+16 (virtual), which we map to physical RAM.
    let rom = vec![0x48, 0x45];
    let mut bus = Bus::new(rom);
    bus.store32(0x08b0_0010, 0xfeed_face); // physical backing at base+16
    let mut cpu = Cpu::new(0);
    // Map code page 0 (R+X) so the fetch works; map virtual data page
    // 0x40000000 -> physical RAM 0x08b00000 (RW).
    cpu.mmu.write_tlb(false, 0x0000_0000 | 0x1, 0x0000_0000 | 0);
    cpu.mmu.write_tlb(true, 0x08b0_0000 | 0x3, 0x4000_0000 | 0);
    cpu.regs.write_ar(5, 0x4000_0000); // virtual base; +16 -> 0x40000010
    assert!(matches!(cpu.step(&mut bus), Step::Ran));
    assert_eq!(cpu.regs.read_ar(4), 0xfeed_face);
}

#[test]
fn store_fault_raises_without_advancing_pc() {
    use crate::firmware::mmio::Bus;
    // s32i.n a6,a7,0x30 = `69 c7` (VERIFIED vector). AR[7]+0x30 is the store
    // target; point it at an unmapped page so the store faults.
    let rom = vec![0x69, 0xc7];
    let mut bus = Bus::new(rom);
    let mut cpu = Cpu::new(0);
    cpu.vecbase = 0x4000_0000;
    cpu.mmu.write_tlb(false, 0x0000_0000 | 0x1, 0x0000_0000 | 0); // code page
    cpu.regs.write_ar(7, 0x5000_0000); // unmapped data page -> DTLB miss on store
    match cpu.step(&mut bus) {
        Step::Exception { cause, .. } => assert_eq!(cause, 24), // LOAD_STORE_TLB_MISS
        other => panic!("expected store fault, got {:?}", other),
    }
    assert_eq!(cpu.pc, 0); // faulting store did not advance
    assert_eq!(cpu.epc1, 0);
}
```

- [ ] **Step 2: Run to verify it fails.** `cargo test --lib firmware::xtensa::interp::mem 2>&1 | tail -20` — FAIL.

- [ ] **Step 3: Write the implementation** — route each site through `translate`. Pattern (apply to all arms; example for the 32-bit load/store and `l32r`):

```rust
pub(super) fn exec(cpu: &mut Cpu, bus: &mut Bus, op: &Op, pc: u32, len: u8) -> Option<Step> {
    // Helper: translate a data address or return the fault Step early.
    // (Implement as a local closure is awkward with the &mut borrows; inline
    //  the match at each site, or add a private `translate_data` on Cpu.)
    match op {
        Op::L32iN { t, s, imm } | Op::L32i { t, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let paddr = match cpu.translate(bus, vaddr, super::Access::Load) {
                Ok(p) => p,
                Err(step) => return Some(step),
            };
            let v = bus.load32(paddr);
            cpu.regs.write_ar(*t, v);
        }
        Op::L32r { t, target } => {
            let paddr = match cpu.translate(bus, *target, super::Access::Load) {
                Ok(p) => p,
                Err(step) => return Some(step),
            };
            let v = bus.load32(paddr);
            cpu.regs.write_ar(*t, v);
        }
        // ... l8ui/l16ui/l16si/s8i/s16i/s32i* : identical pattern, using
        // Access::Load for loads and Access::Store for stores. For the 16-bit
        // composite load16/store16 helpers, translate the base address for
        // Access, then pass the resulting paddr to load16/store16 (which do
        // the two byte accesses on the PHYSICAL address). A 16-bit access that
        // straddles a page boundary is not expected in this firmware; translate
        // the two byte addresses independently if you want full faithfulness
        // (recommended — mirrors the fetch page-safety).
        _ => return None,
    }
    cpu.pc = pc.wrapping_add(len as u32);
    Some(Step::Ran)
}
```

Note to implementer: `load16`/`store16` currently take `bus` + a single `addr` and do two `load8`/`store8`. Change them to take the already-translated PHYSICAL base, OR (preferred for faithfulness) translate each of the two byte addresses via `cpu.translate` before composing — matching the fetch page-safety. Keep the signatures internal; whichever you choose, the store/load must go to physical addresses. Because the borrow of `cpu` (for `translate`) and `bus` overlap, translate FIRST (releasing the `cpu.mmu` borrow) then do the bus access — the pattern above already does this.

- [ ] **Step 4: Run to verify it passes.** `cargo test --lib firmware 2>&1 | tail -30` — update any remaining `mem.rs` tests that used raw physical addresses in `read_ar` to instead map a virtual page (or map identity so their existing addresses still work: `write_tlb(true, phys|0x3, phys|0)` mapping the page identity). Expected all green. `cargo test --lib 2>&1 | tail -5`.

- [ ] **Step 5: Commit**

```bash
git add src/firmware/xtensa/interp/mem.rs
git commit -m "feat(#140): M2b Task 9 -- load/store + l32r through MMU translation"
```

---

### Task 10: real-firmware characterization handoff

**Files:**
- Create: `docs/superpowers/findings/2026-07-04-m2b-autorefill-characterization.md`
- Modify: `src/firmware/xtensa/mmu.rs` or `src/firmware/mod.rs` (a test-only or `log`-gated observation path)

**Interfaces:**
- Consumes: the full M2b stack (Tasks 1-9), the firmware-gated boot driver (`coverage_scan.rs`/`boot_to_idle` pattern).

**Purpose:** This is an OBSERVATION run, not a pass/fail correctness test. Boot the real firmware with the live MMU and record what the autorefill computes, as the empirical starting point for M2c. The firmware binary is not in the repo, so this is firmware-gated (skips cleanly when absent, like the existing boot/coverage tests).

- [ ] **Step 1: Write the firmware-gated observation test** (in `mmu.rs` or alongside `coverage_scan.rs`)

```rust
#[test]
fn characterize_real_firmware_autorefill() {
    // Firmware-gated: skip cleanly if the real image isn't present (same gate
    // as coverage_scan's boot tests). This is an OBSERVATION, not an assertion
    // of boot success -- M2b is not expected to boot past the wall.
    let Some(image) = crate::firmware::test_support::load_real_firmware() else {
        eprintln!("skipping: real firmware image not present");
        return;
    };
    // Boot with the live MMU; step until the first fetch fault or a step cap.
    // Record: PTEVADDR after the prologue, the first autorefill pt_vaddr, the
    // PTE value read, the physical address the faulting fetch resolves to, and
    // which ways/entries the boot witlb/wdtlb installed.
    // (Implementer: reuse the boot harness in coverage_scan.rs / firmware::mod.
    //  Emit the findings via eprintln!/log so they land in the test output; the
    //  human transcribes them into the findings doc, OR write them to a file
    //  under build/experiments/ if the harness supports it.)
    // The ONLY assertions here are sanity: the boot reached the wall via the
    // MMU (e.g. the last Step was an ITLB-miss Exception or an Unknown at a
    // translated physical address), and PTEVADDR == 0x3c000000 (the value the
    // boot prologue programs).
}
```

Note to implementer: match the exact firmware-gating and boot-harness helpers that `coverage_scan.rs` and `firmware::mod`'s `boot_to_idle` already use (the summary and prior tasks reference `test_support`/firmware-gated skips — use whatever the real helper is named; do not invent a new gating mechanism). If the cleanest place to instrument the autorefill is a `log::debug!` inside `Mmu::get_pte`/`refill`, add it there (guarded, low-noise) and run the boot with `RUST_LOG=debug` capturing to a file.

- [ ] **Step 2: Run the observation.** `cargo test --lib firmware::xtensa::mmu::tests::characterize_real_firmware_autorefill -- --nocapture 2>&1 | tee /tmp/m2b-char.txt` then Read `/tmp/m2b-char.txt`.

- [ ] **Step 3: Write the findings document** — `docs/superpowers/findings/2026-07-04-m2b-autorefill-characterization.md` recording, from the observation run:
  - The value of `PTEVADDR` after the boot prologue.
  - The first autorefill `pt_vaddr` computed (expected `(0x3c000000 | 0x80000) = 0x3c080000`).
  - The PTE value read from that address (expected garbage/zero — outside the 248 KB image).
  - The physical address the faulting fetch at `0x20000340` resolves to (expected `0x340`).
  - Which ways/entries the boot `witlb`/`wdtlb` installed, with their VPN/paddr/attr — the firmware's own region map, which is the concrete anchor M2c builds the reconstruction from.
  - An explicit statement: M2b confirms the mechanism is live; M2c must supply the page-table data at `pt_vaddr` (or the correct static region entry) so `0x20000340` resolves to real code.

- [ ] **Step 4: Confirm no regression.** `cargo test --lib 2>&1 | tail -5` — all green (the observation test passes/skips; it does not assert boot success).

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/findings/2026-07-04-m2b-autorefill-characterization.md src/firmware/xtensa/mmu.rs
git commit -m "feat(#140): M2b Task 10 -- real-firmware autorefill characterization (M2c handoff)"
```

---

## Self-Review notes (author)

- **Spec coverage:** every in-scope item (1-10) maps to a task: TLB struct+reset (T1), addressing (T2), lookup/ring/multi-hit (T3), witlb/wdtlb/iitlb/idtlb + config SRs (T4), attr decode (T5), autorefill walk (T6), translate seam + EXCCAUSE/EXCVADDR + double-fault 9a (T7), fetch wiring + page-safety (T8), load/store wiring (T9), characterization handoff (T10).
- **Type consistency:** `Mmu::translate` returns `Result<Translation, MmuFault>`; `Cpu::translate` returns `Result<u32, Step>`; `lookup` returns `Result<TlbHit, u32>`; `Access` (Fetch/Load/Store) maps to `is_write` (2/0/1). `write_tlb(dtlb, at, as_)` and `invalidate_tlb(dtlb, as_)` consistent across T4/T8/T9.
- **Borrow-checker risk** (flagged in T6/T9): translate mutably borrows `cpu.mmu` and needs `&mut Bus`; always translate FIRST (ending the mmu borrow) then do the bus access. `refill` returns `TlbHit` by value so no borrow escapes.
- **Biggest mechanical cost** (flagged in T8): every existing `step()`-driven test in `firmware::xtensa::{interp,mem,arith,branch,control,system}` now needs an identity ITLB/DTLB mapping for its pages. A shared `mapped_cpu` test helper keeps it DRY. Reviewers: expect a large but mechanical test-diff in T8.
- **Confirm-at-implementation items** (derive, don't guess): `AUTOREFILL_WAY_SIZE`/`NREFILLENTRIES` (T2, from QEMU core configs); the RASID SR number 0x5A and the `witlb`/`iitlb` decode byte vectors (T4, from `decode/system.rs`); the firmware-gating helper name (T10, from `coverage_scan.rs`).
