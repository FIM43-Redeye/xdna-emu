# M2c: Mapping Reconstruction & Boot-to-Idle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Boot the real XDNA management firmware from its reset entry past the
MMU wall to the steady-state command loop (`_XAie_ExecuteCmd`) waiting on the
host mailbox, with `reached_idle = true`.

**Architecture:** One PSP load-offset `L` (`phys = file - L`), `varway56=true`
(honor the firmware's own region/data TLB installs), and a synthesized PSP
autorefill page table backing the code region, carry boot from the reset entry
through the C-runtime into the command loop. Phase 1 pins the map and reaches
the C entry (deterministic mechanism + one empirically-pinned constant); Phase 2
is an instrument-first walk-and-stub loop to idle.

**Tech Stack:** Rust; the in-tree Xtensa interpreter (`src/firmware/`); QEMU
`target/xtensa/mmu_helper.c` as the MMU-behavior oracle.

## Global Constraints

- **Derive from the toolchain.** MMU behavior comes from QEMU
  `target/xtensa/mmu_helper.c` (cached under the session scratchpad; the file
  the M2b tests already cite). Every `varway56` formula in this plan is copied
  from that source -- do not invent bit math.
- **No emoji anywhere.** Comments and commit messages are plain text.
- **`cargo test --lib` green after every task.** Never pipe it through
  `tail`/`head`/`grep` -- run it bare, redirect to a file if long.
- **Firmware binary is not in the repo** (downloaded). Boot/coherence tests are
  firmware-gated: they locate the image via `firmware_path()` and `return`
  cleanly (printing `skip:`) when it is absent, exactly like the existing
  `boot_tests`.
- **Physical values (`L`, checkpoint PCs) are pinned empirically, not guessed.**
  Where this plan gives a candidate (e.g. `L = 0x5c`), the accompanying test is
  what confirms it; a task that "passes" only because a constant was force-fit
  without the firmware present is not done.
- **Comment the source of hardware knowledge as a hardware fact** (e.g. "way-5
  region install per the firmware's own prologue, `mmu_helper.c` varway56 path"),
  never as tool internals.
- The `varway56` PTE/attr/ring decode, the autorefill walk, `Cpu::translate`,
  and `Bus` physical-only routing already exist from M2b -- this milestone
  parameterizes and feeds them, it does not rebuild them.

## Background the implementer needs

Read these before starting; they are the derivation this plan rests on:
- Spec: `docs/superpowers/specs/2026-07-04-m2c-mapping-boot-to-idle-design.md`
  (the authoritative design; this plan implements it).
- Finding: `docs/superpowers/findings/2026-07-04-m2b-autorefill-characterization.md`
  (the firmware's own TLB operands and the wall).
- `src/firmware/xtensa/mmu.rs` (the MMU this milestone parameterizes),
  `src/firmware/mmio.rs` (the `Bus`), `src/firmware/mod.rs`
  (`FirmwareProcessor::load` and the `boot_tests`).

Key facts (from the characterization, all confirmed against the image):
- Reset vector at file `0x200` (`j 0x214`); reset head `0x214-0x28e` sets
  `INTENABLE=0`, `VECBASE=0x800`, `ATOMCTL=0x15`, cache-init, `j 0x320`.
- Prologue `0x320-0x399` sets `ITLBCFG=0`, `DTLBCFG=0x00030000`,
  `PTEVADDR=0x3c000000`; installs way-5 code region (`VPN 0x20000000 -> PPN 0`,
  attr 7); invalidates way-6 regions `0x20000000..0xe0000000`; `jx 0x20000340`.
- The coherent continuation for virtual `0x20000340` is at file `0x39c` -- it
  tears down way 5, installs the way-6 data windows, runs data-copy loops, then
  `call0 0xe080` (C entry). Reaching file `0x39c` from virtual `0x20000340`
  under 4 KB paging forces `phys = file - L` with `L != 0` (candidate `0x5c`).
- Steady-state command loop: `_XAie_ExecuteCmd @0x32b34` (its rodata: "Invalid
  transaction opcode").
- The firmware installs a code mapping exactly once (prologue way 5); after it
  invalidates that entry, `.text` fetches rely on the autorefill page table the
  PSP built at `PTEVADDR` -- which is absent from the image and is what the
  synthesized PT supplies.

## File Structure

| File | Responsibility | New/Modified |
|------|----------------|--------------|
| `src/firmware/xtensa/mmu.rs` | Xtensa MMU. Gains a `varway56` flag toggling ways 5/6 between fixed region entries and software-writable variable ways (masks, entry-index, way sizes, reset population), per `mmu_helper.c`. | Modified |
| `src/firmware/mmio.rs` (`Bus`) | Add the PSP load-offset (`phys = file - L`) on ROM access, and a page-table aperture serving the synthesized PT as physical memory. | Modified |
| `src/firmware/psp_map.rs` | New. Builds the synthesized PSP autorefill page table (code-region PTEs) and the region-entry spec that makes the PTEVADDR window fetchable. One responsibility, unit-tested without a boot. | New |
| `src/firmware/mod.rs` (`FirmwareProcessor::load`) | Assemble: reset entry, load-offset, `varway56=true`, install the PT region entry, populate the PT. Retire the provisional way-4 map. | Modified |
| `src/firmware/sysstub.rs` | (Phase 2) Extend the system-aperture stub for the off-array MMIO the boot path touches. | Modified |
| `docs/fidelity-gaps/firmware-mmu.md` | Update the `varway56` entry with the Task 1 outcome / coherence confirmation. | Modified |

---

## Phase 1 -- pin the map, reach the C entry

### Task 1: Xtensa config extraction (bounded investigation)

**Goal:** authoritatively confirm-or-correct `varway56` (and, opportunistically,
the autorefill way count, TLB way sizes, `ndepc`, reset vector) from a real
artifact before relying on coherence. Non-blocking: Task 2 proceeds on
`varway56=true` regardless of this outcome.

**Files:**
- Create: `docs/superpowers/findings/2026-07-04-m2c-xtensa-config.md`
- Modify: `docs/fidelity-gaps/firmware-mmu.md` (the `varway56` row)

**This is a research task, not a TDD task.** A fixed pass over the four candidate
sources from the spec, then a written verdict. Do not open-endedly hunt beyond
these.

- [ ] **Step 1: Search the firmware image.** Grep `npu.dev.sbin` (and the
  decoded `build/experiments/firmware-re/{listing.txt,symbols.txt,INFODUMP.md}`)
  for a Tensilica config signature, core-name string, or `core-isa`-style table.
  Commands: `strings -t x ../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin | grep -iE 'xtensa|tensilica|core-isa|dc[0-9]{3}|LX[0-9]'`.
- [ ] **Step 2: Search xdna-driver / RyzenAI-SW** for headers or comments naming
  the management core: `grep -rniE 'xtensa|tensilica|varway|core-isa' /home/triple/npu-work/xdna-driver /home/triple/npu-work/RyzenAI-SW 2>/dev/null | head -50`.
- [ ] **Step 3: Decode the `$PS1` container fields** not yet decoded (the header
  dump: offsets `0x00-0x60`). Note any that look like a config ID or core
  descriptor. (The header is: 16-byte hash, `$PS1`@0x10, size@0x14=`0x3c910`,
  a second hash@0x38, size@0x50, `0x81011052`@0x58, `0x1ff`@0x60.)
- [ ] **Step 4: Cross-reference QEMU core configs.** If a string/behavior
  identifies the core model, check whether QEMU's `target/xtensa/core-*.c`
  carries it and read its `varway56` / `nrefillentries`. Otherwise note that the
  six MMU-enabled QEMU cores M2b surveyed all have `varway56=false` (so
  `varway56=true` would be an AMD-specific config choice, confirmable only by
  coherence).
- [ ] **Step 5: Write the verdict.** In the findings note, record for each
  source what was searched and found. Conclude with one of: (a) config found ->
  cite it, list the confirmed values; (b) not found -> state that `varway56=true`
  rests on the Phase 1 coherence gate. Update the `varway56` row in
  `docs/fidelity-gaps/firmware-mmu.md` accordingly (real citation, or
  "coherence-inferred, confirmed by the M2c Phase 1 coherence gate").
- [ ] **Step 6: Commit.**

```bash
git add docs/superpowers/findings/2026-07-04-m2c-xtensa-config.md docs/fidelity-gaps/firmware-mmu.md
git commit -m "doc(#140): M2c Task 1 -- Xtensa config extraction verdict"
```

---

### Task 2: `varway56` parameterization in the MMU

**Files:**
- Modify: `src/firmware/xtensa/mmu.rs`
- Test: `src/firmware/xtensa/mmu.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `Mmu::new_with_varway56(varway56: bool) -> Mmu` (and `Mmu::new()`
  keeps `varway56=false`); a public field `pub varway56: bool`. Every
  way-5/6-dependent method (`way_size`, `get_page_size` callers `addr_mask`,
  `split_entry`, and the reset population) honors it.

The QEMU semantics (from `mmu_helper.c`, verbatim):
- `xtensa_tlb_get_addr_mask` (way 5): `varway56 ? 0xf8000000 << page_size : 0xf8000000`.
- `xtensa_tlb_get_addr_mask` (way 6): `varway56 ? 0xf0000000 << (1 - page_size) : 0xf0000000`.
- `split_tlb_entry_spec_way` ei (way 5): `varway56 ? (v >> (27 + page_size)) & 0x3 : (v >> 27) & 0x1`.
- `split_tlb_entry_spec_way` ei (way 6): `varway56 ? (v >> (29 - page_size)) & 0x7 : (v >> 28) & 0x1`.
- `way_size` (entries addressable): way 5 = `varway56 ? 4 : 2`, way 6 =
  `varway56 ? 8 : 2` (`0x3` and `0x7` masks). `MAX_TLB_WAY_SIZE` is already 8, so
  way 6's 8 entries fit.
- `reset_tlb_mmu_ways56` (`varway56=true` branch): ways 5/6 are NOT loaded with
  the fixed entries; way 6 gets 8 identity region entries
  (`entry[6][ei] = { vaddr: ei<<29, paddr: ei<<29, asid: 1, attr: 3 }`,
  `variable` left `true` from the base reset), and way 5 is left empty+variable.

- [ ] **Step 1: Write the failing tests.** Add to `mod tests`:

```rust
#[test]
fn varway56_true_masks_and_indices() {
    // mmu_helper.c varway56=true arms. page_size 0 (cfg 0) for both ways here.
    let mmu = Mmu::new_with_varway56(true);
    // way 5 mask: 0xf8000000 << page_size(=0) = 0xf8000000; way 6: 0xf0000000 << (1-0) = 0xe0000000.
    assert_eq!(mmu.addr_mask(false, 5), 0xf800_0000);
    assert_eq!(mmu.addr_mask(false, 6), 0xe000_0000);
    // ei way5: (v >> (27+0)) & 0x3 ; way6: (v >> (29-0)) & 0x7
    let (_v5, ei5) = mmu.split_entry(0x2800_0000, false, 5); // bits[28:27]=0b01 -> 1
    assert_eq!(ei5, 1);
    let (_v6, ei6) = mmu.split_entry(0x6000_0000, false, 6); // bits[31:29]=0b011 -> 3
    assert_eq!(ei6, 3);
}

#[test]
fn varway56_true_way_sizes() {
    let mmu = Mmu::new_with_varway56(true);
    assert_eq!(mmu.way_size(false, 5), 4);
    assert_eq!(mmu.way_size(false, 6), 8);
    // varway56=false is unchanged (regression).
    let fixed = Mmu::new_with_varway56(false);
    assert_eq!(fixed.way_size(false, 5), 2);
    assert_eq!(fixed.way_size(false, 6), 2);
}

#[test]
fn varway56_true_reset_populates_way6_identity_regions() {
    // mmu_helper.c reset_tlb_mmu_ways56 varway56=true: way6[ei] = identity region
    // ei<<29, asid 1, attr 3, variable (software-writable). way5 empty+variable.
    let mmu = Mmu::new_with_varway56(true);
    for ei in 0..8usize {
        assert_eq!(mmu.itlb[6][ei].vaddr, (ei as u32) << 29);
        assert_eq!(mmu.itlb[6][ei].paddr, (ei as u32) << 29);
        assert_eq!(mmu.itlb[6][ei].asid, 1);
        assert_eq!(mmu.itlb[6][ei].attr, 3);
        assert!(mmu.itlb[6][ei].variable, "software-writable so the firmware can invalidate/reinstall");
    }
    assert_eq!(mmu.itlb[5][0].asid, 0, "way5 empty under varway56=true");
    assert!(mmu.itlb[5][0].variable);
}

#[test]
fn varway56_true_witlb_to_way5_installs() {
    // Under varway56=true a witlb to way 5 must install (vs the varway56=false
    // no-op M2b tested). The firmware's own prologue does exactly this.
    let mut mmu = Mmu::new_with_varway56(true);
    // AS: way 5 in low 3 bits, VPN in the rest. AT: paddr|attr, ring 0.
    // Firmware operands: AS=0x20000005, AT=7 (VPN 0x20000000 -> PPN 0, attr 7).
    mmu.write_tlb(false, 7, 0x2000_0005);
    let hit = mmu.lookup(0x2000_0340, false).expect("way5 install now covers the code region");
    assert_eq!(mmu.itlb[hit.wi][hit.ei].attr, 7);
}
```

- [ ] **Step 2: Run to verify they fail.**

Run: `cargo test --lib firmware::xtensa::mmu 2>/tmp/t2.txt; tail -30 /tmp/t2.txt`
Expected: FAIL -- `Mmu::new_with_varway56` does not exist.

- [ ] **Step 3: Implement.** In `src/firmware/xtensa/mmu.rs`:
  - Add `pub varway56: bool` to the `Mmu` struct.
  - Replace `pub fn new()` body to call `new_with_varway56(false)`; add:

```rust
pub fn new_with_varway56(varway56: bool) -> Self {
    let empty = TlbEntry { variable: true, ..TlbEntry::default() };
    let mut mmu = Self {
        itlb: [[empty; MAX_TLB_WAY_SIZE]; ITLB_NWAYS],
        dtlb: [[empty; MAX_TLB_WAY_SIZE]; DTLB_NWAYS],
        ptevaddr: 0,
        rasid: 0x04030201,
        itlbcfg: 0,
        dtlbcfg: 0,
        autorefill_idx: 0,
        varway56,
    };
    Self::reset_ways56(&mut mmu.itlb, varway56);
    Self::reset_ways56(&mut mmu.dtlb, varway56);
    mmu
}
```

  - Rename `load_fixed_ways56` to `reset_ways56(tlb, varway56)` and branch:

```rust
fn reset_ways56(tlb: &mut [[TlbEntry; MAX_TLB_WAY_SIZE]], varway56: bool) {
    if !varway56 {
        // mmu_helper.c reset_tlb_mmu_ways56 varway56=false: four fixed entries.
        let fixed = |vaddr, paddr, attr| TlbEntry { vaddr, paddr, asid: 1, attr, variable: false };
        tlb[5][0] = fixed(0xd0000000, 0, 7);
        tlb[5][1] = fixed(0xd8000000, 0, 3);
        tlb[6][0] = fixed(0xe0000000, 0xf0000000, 7);
        tlb[6][1] = fixed(0xf0000000, 0xf0000000, 3);
    } else {
        // varway56=true: way 6 gets 8 identity region entries (software-writable
        // so the firmware's own iitlb/idtlb/wdtlb can invalidate and reinstall
        // them); way 5 is left empty+variable.
        for ei in 0..8usize {
            let v = (ei as u32) << 29;
            tlb[6][ei] = TlbEntry { vaddr: v, paddr: v, asid: 1, attr: 3, variable: true };
        }
    }
}
```

  - In `way_size`, replace the `5 | 6 => 2` arm:

```rust
5 => if self.varway56 { 4 } else { 2 },
6 => if self.varway56 { 8 } else { 2 },
```

  - In `addr_mask`, replace the way-5 and way-6 arms:

```rust
5 => if self.varway56 { 0xf8000000u32 << self.get_page_size(dtlb, wi) } else { 0xf8000000 },
6 => if self.varway56 { 0xf0000000u32 << (1 - self.get_page_size(dtlb, wi)) } else { 0xf0000000 },
```

  - In `split_entry`, replace the way-5 and way-6 arms:

```rust
5 => if self.varway56 {
    ((vaddr >> (27 + self.get_page_size(dtlb, wi))) & 0x3) as usize
} else {
    ((vaddr >> 27) & 0x1) as usize
},
6 => if self.varway56 {
    ((vaddr >> (29 - self.get_page_size(dtlb, wi))) & 0x7) as usize
} else {
    ((vaddr >> 28) & 0x1) as usize
},
```

  - Update the doc comments on `way_size`/`addr_mask`/`split_entry`/the renamed
    `reset_ways56` to note the varway56 branch and cite `mmu_helper.c`.

- [ ] **Step 4: Run to verify pass** (including the M2b regression tests, which
  must still pass with `varway56=false`).

Run: `cargo test --lib firmware::xtensa::mmu 2>/tmp/t2.txt; tail -30 /tmp/t2.txt`
Expected: PASS -- all mmu tests, old and new.

- [ ] **Step 5: Commit.**

```bash
git add src/firmware/xtensa/mmu.rs
git commit -m "feat(#140): M2c Task 2 -- varway56 parameterization (ways 5/6 variable)"
```

---

### Task 3: PSP load-offset in the Bus

**Files:**
- Modify: `src/firmware/mmio.rs`
- Test: `src/firmware/mmio.rs` (`mod tests`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `Bus::new_with_load_offset(rom: Vec<u8>, load_offset: u32) -> Bus`
  (and `Bus::new` keeps offset 0). ROM-region access at physical `P` reads image
  byte `P + load_offset` (`phys = file - load_offset`, i.e. `file = phys +
  load_offset`). Only the ROM region is offset; RAM/mailbox/array/system are
  unchanged.

- [ ] **Step 1: Write the failing test.**

```rust
#[test]
fn rom_access_honors_psp_load_offset() {
    // phys = file - L. With L = 4, physical address 0 reads image byte 4.
    let mut bus = Bus::new_with_load_offset(vec![0, 0, 0, 0, 0x78, 0x56, 0x34, 0x12], 4);
    assert_eq!(bus.load32(0), 0x12345678); // phys 0 -> file 4
    assert_eq!(bus.load8(1), 0x56);        // phys 1 -> file 5
    // Bus::new keeps offset 0 (regression).
    let mut z = Bus::new(vec![0x78, 0x56, 0x34, 0x12]);
    assert_eq!(z.load32(0), 0x12345678);
}
```

- [ ] **Step 2: Run to verify it fails.**

Run: `cargo test --lib firmware::mmio 2>/tmp/t3.txt; tail -20 /tmp/t3.txt`
Expected: FAIL -- `Bus::new_with_load_offset` does not exist.

- [ ] **Step 3: Implement.** In `src/firmware/mmio.rs`:
  - Add `load_offset: u32` to the `Bus` struct.
  - `pub fn new(rom: Vec<u8>) -> Self { Self::new_with_load_offset(rom, 0) }` and:

```rust
/// Create a bus whose ROM aperture applies the PSP load-offset: a physical
/// address `P` in the ROM region reads image byte `P + load_offset`
/// (`phys = file - load_offset`). The x86 PSP loads the firmware body at a
/// physical base below its file offset; this models that placement so the
/// code region's virtual->physical map lands on real image bytes (M2c). RAM,
/// mailbox, array, and system apertures are unaffected.
pub fn new_with_load_offset(rom: Vec<u8>, load_offset: u32) -> Self {
    Self { rom, ram: Vec::new(), mailbox: Vec::new(), sysstub: SysStub::new(), load_offset }
}
```

  - In `load32`, `load8`, and `peek8`, the `Region::Rom` arm indexes with the
    offset. Replace each `Region::Rom => ...&self.rom, addr...` with the offset
    applied, e.g. in `load32`:

```rust
Region::Rom => read_le32(&self.rom, addr.wrapping_add(self.load_offset)),
```

  and correspondingly `byte_at(&self.rom, addr.wrapping_add(self.load_offset))`
  in `load8` and `peek8`. (ROM stores stay logged-and-ignored; no offset needed.)

- [ ] **Step 4: Run to verify pass** (including the existing mmio regression
  tests, which use `Bus::new` at offset 0).

Run: `cargo test --lib firmware::mmio 2>/tmp/t3.txt; tail -20 /tmp/t3.txt`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/firmware/mmio.rs
git commit -m "feat(#140): M2c Task 3 -- PSP load-offset on the ROM aperture"
```

---

### Task 4: Page-table aperture in the Bus

The synthesized page table must be readable by the autorefill walk as physical
memory. `get_pte` translates `pt_vaddr = (PTEVADDR | (vaddr>>10)) & ~3` (which a
region entry maps to a physical address) then `bus.load32(paddr)`. We place the
PT at physical base `0x3c000000` (identity with `PTEVADDR`) and serve that window
from a dedicated backing store.

**Files:**
- Modify: `src/firmware/mmio.rs`
- Test: `src/firmware/mmio.rs` (`mod tests`)

**Interfaces:**
- Consumes: nothing new.
- Produces: a `Region::PageTable` aperture at `[0x3c000000, 0x3c000000 + 0x100000)`
  backed by a lazily-grown `page_table: Vec<u8>`, offset-keyed from
  `PAGE_TABLE_BASE`; `pub const PAGE_TABLE_BASE: u32 = 0x3c00_0000`. Reads/writes
  route to it like RAM. `pub fn write_page_table_word(&mut self, phys: u32, v: u32)`
  for the builder (Task 5) to populate it.

- [ ] **Step 1: Write the failing test.**

```rust
#[test]
fn page_table_aperture_round_trips() {
    let mut bus = Bus::new(vec![]);
    assert_eq!(Bus::region(0x3c08_0000), Region::PageTable);
    bus.write_page_table_word(0x3c08_0000, 0x08b0_5001);
    assert_eq!(bus.load32(0x3c08_0000), 0x08b0_5001);
    // Below and above the aperture is still System (regression).
    assert_eq!(Bus::region(0x3c10_0000), Region::System);
}
```

- [ ] **Step 2: Run to verify it fails.**

Run: `cargo test --lib firmware::mmio 2>/tmp/t4.txt; tail -20 /tmp/t4.txt`
Expected: FAIL -- `Region::PageTable` / `write_page_table_word` do not exist.

- [ ] **Step 3: Implement.** In `src/firmware/mmio.rs`:
  - Add `PageTable` to `enum Region` (doc: "Synthesized PSP autorefill page table
    at `0x3c000000` (M2c); real physical memory the autorefill walk reads.").
  - Add `pub const PAGE_TABLE_BASE: u32 = 0x3c00_0000;` and a private
    `const PAGE_TABLE_END: u32 = 0x3c10_0000;` (1 MB window; the code-region PTEs
    occupy `0x3c080000..` and fit comfortably).
  - Add `page_table: Vec<u8>` to the struct; initialize `Vec::new()` in
    `new_with_load_offset`.
  - In `region`, add the branch BEFORE the final `System` else:
    `else if (PAGE_TABLE_BASE..PAGE_TABLE_END).contains(&addr) { Region::PageTable }`.
  - Route `Region::PageTable` in `load32`/`load8`/`peek8`/`store32`/`store8` to
    `self.page_table` offset-keyed from `PAGE_TABLE_BASE` (same shape as `Ram`).
  - Add:

```rust
/// Populate a word of the synthesized page table (M2c `psp_map`). Physical
/// address must fall in the PageTable aperture.
pub fn write_page_table_word(&mut self, phys: u32, v: u32) {
    debug_assert_eq!(Self::region(phys), Region::PageTable);
    write_le32(&mut self.page_table, phys - PAGE_TABLE_BASE, v);
}
```

- [ ] **Step 4: Run to verify pass.**

Run: `cargo test --lib firmware::mmio 2>/tmp/t4.txt; tail -20 /tmp/t4.txt`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/firmware/mmio.rs
git commit -m "feat(#140): M2c Task 4 -- page-table aperture serving the synth PT"
```

---

### Task 5: Synthesized PSP page-table builder

**Files:**
- Create: `src/firmware/psp_map.rs`
- Modify: `src/firmware/mod.rs` (add `mod psp_map;`)
- Test: `src/firmware/psp_map.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `Bus::write_page_table_word` (Task 4), `Bus::PAGE_TABLE_BASE`,
  `Mmu::write_tlb` (M2b), `PTEVADDR`.
- Produces:
  - `pub const CODE_REGION_BASE: u32 = 0x2000_0000;`
  - `pub fn install(mmu: &mut Mmu, bus: &mut Bus, load_offset: u32, image_len: u32)`
    -- installs the PTEVADDR region entry into the MMU and writes the code-region
    PTEs into the Bus page-table aperture.

The mechanism, exactly:
- **Region entry for the PTEVADDR window.** `get_pte` must translate `pt_vaddr`
  (in `0x3c000000..`) WITHOUT autorefill (recursion guard). Install a way-4 D-TLB
  region entry mapping virtual `0x3c000000 -> phys 0x3c000000` (identity, the
  PageTable aperture), attr 3 (R/W). Way 4 (variable, outside the autorefill
  round-robin, untouched by the firmware's way-5/6 ops). AS = `0x3c000000 | 4`,
  AT = `0x3c000000 | 3`. With `DTLBCFG` way-4 page size 0 the mask is 1 MB, which
  covers the whole PT window.
- **Code-region PTEs.** For each 4 KB virtual page `v = CODE_REGION_BASE + i*0x1000`
  spanning the image (`i` in `0 ..= image_len/0x1000`), the physical page is
  `phys = v - CODE_REGION_BASE` (the way-5 region map's `virtual 0x20000000 ->
  phys 0`, extended per-page), and the PTE word is `phys | attr` with attr 7
  (cached RWX), ring 0 (bits[5:4]=0). It is stored at physical
  `pt_phys = (PTEVADDR | (v >> 10)) & !3` (identity-mapped into the PageTable
  aperture, so `pt_phys` is where `get_pte` will read). `PTEVADDR` is
  `0x3c000000` (the value the prologue programs; pass it or hardcode-with-cite).

Note the physical page here is `v - CODE_REGION_BASE`, and the Bus ROM offset
(`load_offset`, Task 3) then maps that physical page to image bytes -- the two
compose: virtual code page -> phys (this PTE) -> image byte (Bus load-offset).

- [ ] **Step 1: Write the failing test.**

```rust
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
        // 0x20000000; the real prologue invalidates it (iitlb 0x20000006) before
        // relying on autorefill. Mimic that here so the lookup MISSES (rather
        // than hitting the identity region) and the autorefill walk runs.
        mmu.invalidate_tlb(false, 0x2000_0006);

        // A fetch of virtual 0x20000340 must now autorefill from the synth PT to
        // phys 0x340 (page base 0 | offset 0x340).
        let t = mmu.translate(&mut bus, 0x2000_0340, 2 /*fetch*/, 0).expect("autorefill from synth PT");
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
}
```

- [ ] **Step 2: Run to verify it fails.**

Run: `cargo test --lib firmware::psp_map 2>/tmp/t5.txt; tail -20 /tmp/t5.txt`
Expected: FAIL -- module `psp_map` does not exist.

- [ ] **Step 3: Implement.** Create `src/firmware/psp_map.rs`:

```rust
//! Synthesized PSP autorefill page table (M2c). The x86 PSP builds the code
//! region's virtual->physical page table in management-processor RAM before
//! starting the firmware; that table is absent from every artifact we hold, so
//! we reconstruct its OBSERVED EFFECT by coherence: a linear map of the code
//! region (virtual 0x20000000+) onto the firmware image, matching the firmware's
//! own way-5 region install (`virtual 0x20000000 -> phys 0`, `mmu_helper.c`
//! varway56 path) extended to per-page autorefill entries. See
//! `docs/superpowers/specs/2026-07-04-m2c-mapping-boot-to-idle-design.md`.

use crate::firmware::mmio::PAGE_TABLE_BASE;
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
}
```

  Add `mod psp_map;` to `src/firmware/mod.rs` (near the other `mod` lines). If
  `Mmu` / `Bus` fields used by the test are not `pub` at the needed path, expose
  them minimally (the test uses `mmu.ptevaddr`, `mmu.dtlbcfg`, already `pub`).

- [ ] **Step 4: Run to verify pass.**

Run: `cargo test --lib firmware::psp_map 2>/tmp/t5.txt; tail -20 /tmp/t5.txt`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/firmware/psp_map.rs src/firmware/mod.rs
git commit -m "feat(#140): M2c Task 5 -- synthesized PSP page-table builder"
```

---

### Task 6: Assemble in `load()` and pin `L` with the Phase 1 coherence gate

This is the empirical task: it wires the mechanism together, introduces the
load-offset constant `L`, and pins `L` by requiring the real firmware to reach
the C-entry coherence checkpoints. Candidate `L = 0x5c`; the gate is what
confirms it.

**Files:**
- Modify: `src/firmware/mod.rs` (`FirmwareProcessor::load`; add a `RESET_ENTRY`
  const and a load path that starts there; add the coherence test)
- Test: `src/firmware/mod.rs` (`boot_tests`)

**Interfaces:**
- Consumes: `Mmu::new_with_varway56`, `Bus::new_with_load_offset`,
  `psp_map::install`, `psp_map::CODE_REGION_BASE`.
- Produces: an M2c load path. Keep the existing `FirmwareProcessor::load(image,
  entry)` signature working for the M2b tests (they pass `BOOT_ENTRY=0x320`); add
  the M2c wiring behind it or as a sibling `load_m2c(image)` -- see Step 3.

- [ ] **Step 1: Write the failing coherence gate test.** In `boot_tests`:

```rust
/// M2c Phase 1 coherence gate: with the load-offset, varway56, and the synth
/// PT in place, the real firmware boots from the reset entry past the MMU wall,
/// through the way-5 teardown and data-copy, to the C entry (`call0 0xe080`).
/// This test PINS the load-offset L: it passes iff L makes the continuation
/// coherent. If it fails, `last_pc` / `funcs_entered` localize the correct L.
#[test]
fn m2c_boot_reaches_c_entry() {
    let Some(path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let raw = std::fs::read(&path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse");
    let mut proc = FirmwareProcessor::load_m2c(img);

    // Record whether the boot reaches the C-entry call (file 0xe080). The C
    // entry is reached via the continuation after the way-5 teardown, so
    // reaching it proves the whole code-region map is coherent.
    let reached = proc.reaches_pc(0xe080, 200_000);
    eprintln!("m2c boot: reached C entry (0xe080) = {reached}, last_pc = {:#x}", proc.cpu.pc);
    assert!(reached, "boot did not reach the C entry; last_pc={:#x} -- L or the map is wrong", proc.cpu.pc);
}
```

  Add a small helper `reaches_pc` to `FirmwareProcessor` (steps until the CPU's
  PC equals `target` in the code-region-mapped physical sense, or `max` is hit):

```rust
/// Step until the CPU reaches the code at file offset `file_target` (returns
/// true), or `max` instructions pass / the run stops (returns false). Phase-1
/// coherence probe. The mapping has a load-offset: pre-paging the PC runs at low
/// physical `file - L`; post-paging it runs in the code region's virtual space,
/// `virtual = CODE_REGION_BASE + (file - L)`. Match both (the C entry is
/// post-paging, so `virt_alias` is the one that fires there).
pub fn reaches_pc(&mut self, file_target: u32, max: u64) -> bool {
    let phys_target = file_target.wrapping_sub(PSP_LOAD_OFFSET);
    let virt_alias = crate::firmware::psp_map::CODE_REGION_BASE.wrapping_add(phys_target);
    for _ in 0..max {
        if self.cpu.pc == phys_target || self.cpu.pc == virt_alias {
            return true;
        }
        match self.cpu.step(&mut self.bus) {
            Step::Ran | Step::Exception { .. } => {}
            Step::Wait(_) | Step::Unknown { .. } => return false,
        }
    }
    false
}
```

- [ ] **Step 2: Run to verify it fails.**

Run: `cargo test --lib firmware::boot_tests::m2c_boot_reaches_c_entry 2>/tmp/t6.txt; tail -30 /tmp/t6.txt`
Expected: FAIL -- `FirmwareProcessor::load_m2c` / `reaches_pc` do not exist (or,
once they do, the boot walls before the C entry until `L` is right).

- [ ] **Step 3: Implement `load_m2c` and the load-offset constant.** In
  `src/firmware/mod.rs`:

```rust
/// The PSP load-offset: physical `P` in the ROM aperture reads image byte
/// `P + PSP_LOAD_OFFSET` (`phys = file - PSP_LOAD_OFFSET`). Pinned by the M2c
/// Phase 1 coherence gate (`m2c_boot_reaches_c_entry`): the value that makes the
/// `jx 0x20000340` target land on the coherent continuation at file 0x39c.
/// Candidate 0x5c (0x39c - 0x340); confirmed by the gate. Hardware fact: the
/// x86 PSP loads the firmware body at this physical base before start.
const PSP_LOAD_OFFSET: u32 = 0x5c;

/// The physical reset entry: the reset vector at file 0x200 sits at physical
/// `0x200 - PSP_LOAD_OFFSET`. Boot begins here and the reset head `j`-es to the
/// MMU prologue.
const RESET_ENTRY: u32 = 0x200 - PSP_LOAD_OFFSET;
```

  Add the M2c constructor (leaving `load` for M2b's `BOOT_ENTRY` tests intact):

```rust
/// Load `image` for the M2c boot-to-idle path: PSP load-offset, varway56=true,
/// synthesized code-region page table, starting at the physical reset entry.
pub fn load_m2c(image: FirmwareImage) -> Self {
    let image_len = image.bytes().len() as u32;
    let mut bus = Bus::new_with_load_offset(image.bytes().to_vec(), PSP_LOAD_OFFSET);
    let mut cpu = Cpu::new(RESET_ENTRY);
    cpu.mmu = xtensa::mmu::Mmu::new_with_varway56(true);

    // NO provisional low-region map here (unlike the M2b `load` path). With
    // varway56=true the reset populates way-6 entry 0 as an identity region
    // 0..0x1fffffff, attr 3 (RWX), which already covers the reset head and
    // prologue's low physical addresses -- and the firmware's own prologue
    // leaves way-6 entry 0 alone (it invalidates entries 1..7 only). Adding a
    // separate provisional entry would MULTI-HIT against way-6 entry 0 and fault
    // (cause 17) on the very first fetch. The way-6 reset identity IS the
    // low-region map the PSP established; we do not re-invent it.

    // The prologue programs PTEVADDR/DTLBCFG itself (to these exact values), but
    // the synth PT install below needs them now to place the region entry (its
    // way-4 page size is read from DTLBCFG) and the PTEs. Setting them early is
    // consistent: the prologue re-writes the identical values.
    cpu.mmu.ptevaddr = 0x3c00_0000;
    cpu.mmu.dtlbcfg = 0x0003_0000;
    psp_map::install(&mut cpu.mmu, &mut bus, PSP_LOAD_OFFSET, image_len);

    let symbols = load_symbols();
    Self { cpu, bus, entry: RESET_ENTRY, symbols }
}
```

  (`Cpu` must expose `mmu` as assignable -- it already holds `pub mmu`. If
  `Cpu::new` does not let `mmu` be replaced, set fields instead; adjust to the
  actual `Cpu` API.)

- [ ] **Step 4: Run and pin `L`.** Run the gate:

Run: `XDNA_FIRMWARE=../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin cargo test --lib firmware::boot_tests::m2c_boot_reaches_c_entry -- --nocapture 2>/tmp/t6.txt; tail -30 /tmp/t6.txt`

Expected: PASS (boot reaches the C entry). **If it FAILS:** the printed
`last_pc` localizes the problem. If `last_pc` is a low ROM address near the
prologue, the reset-head/prologue path desynced (check `RESET_ENTRY`). If it
walls right after the `jx` (`last_pc` in the code region near `0x2000_034x`),
`L` is wrong: the continuation is at file `0x39c`, so `L = 0x39c - 0x340 = 0x5c`
should be exact; if the image's real continuation differs, set `L` to
`(continuation_file_offset) - 0x340` and re-run. Do NOT force the test green by
loosening the assertion -- the reached-C-entry checkpoint is the pin.

- [ ] **Step 5: Retire the provisional way-4 map comment drift and commit.**
  Ensure the old `FirmwareProcessor::load` provisional-map comment still
  accurately describes only the M2b path; the M2c path documents its own map.

```bash
git add src/firmware/mod.rs
git commit -m "feat(#140): M2c Task 6 -- assemble load_m2c, pin PSP load-offset via coherence gate"
```

- [ ] **Step 6: Run the full suite.**

Run: `cargo test --lib 2>/tmp/t6full.txt; tail -5 /tmp/t6full.txt`
Expected: PASS, 0 failed (new tests included; firmware-gated ones run since the
image is present on this devbox).

---

## Phase 2 -- boot to idle (instrument-first walk-and-stub)

**This phase is NOT a fixed task list.** The number and nature of the off-array
MMIO walls between the C entry and the command-loop idle are unknown until the
boot is walked -- that is by design (the spec's instrument-first stance). Phase 2
is an iterated procedure. Each iteration is one commit and one reviewer gate,
exactly like a task; the executor authors iterations until the termination
condition holds.

**Termination:** `FirmwareProcessor::load_m2c(img).boot_to_idle(N)` returns
`reached_idle == true` with `wait_reason` at the command loop, and
`funcs_entered` contains `_XAie_ExecuteCmd` (`0x32b34`). At that point Phase 2 --
and M2c -- is complete; proceed to the final whole-branch review.

### The iteration procedure

- [ ] **Step A: Run the boot and find the current wall.** Run a firmware-gated
  observation (model it on the existing `characterize_real_firmware_autorefill`
  test, or extend the Phase 1 `boot_to_idle` call to print the stop reason):
  `boot_to_idle` stops on one of `unresolved_spin` (a tight poll on an unmodeled
  system-aperture address -- `SysStub::spinning`), `unknown_op`, or a fault. Note
  the stopping address / PC and the function context (via `funcs_entered` and the
  `symbols` map / `build/experiments/firmware-re/listing.txt`).
- [ ] **Step B: Identify what the firmware expects there.** Cross-reference the
  address against `build/experiments/firmware-re/peripheral-map.txt`, `INFODUMP.md`
  (the mailbox/command protocol), and the driver headers. Determine the minimal
  faithful behavior: a status bit the firmware polls until set, a readback
  register, a doorbell. Prefer modeling the real semantic (what the hardware
  would return) over a blind constant; a blind constant that merely unblocks the
  poll is acceptable only when the real semantic is genuinely out of scope for
  reaching idle, and must be commented as such.
- [ ] **Step C: Write a failing test for the stub.** A `sysstub` (or `mmio`) unit
  test asserting the new stub returns/accepts what the firmware needs at that
  address. Firmware-independent where possible.
- [ ] **Step D: Implement the stub** in `src/firmware/sysstub.rs` (or the
  relevant aperture), commented with the firmware access it services and the
  source (peripheral-map / INFODUMP / driver header).
- [ ] **Step E: Re-run the boot.** Confirm it advances PAST the previous wall
  (the stop reason's address changes, or it reaches idle). Record the new
  furthest PC / `funcs_entered` in the commit message so progress is legible.
- [ ] **Step F: Commit** (`feat(#140): M2c Phase 2 -- stub <addr> (<what>), boot now reaches <pc>`),
  then return to Step A. If `reached_idle`, stop.

### Guardrails for Phase 2

- **No silent scope creep, but stubs are expected.** A growing stub list is the
  planned shape here, not scope creep -- each stub is one wall the real firmware
  hits. What IS a scope signal to surface to the human: a wall that needs real
  `DeviceState`/array behavior the emulator already models elsewhere (route it,
  don't re-stub), or a wall implying a whole subsystem (a full mailbox ring
  protocol) rather than a single register. Surface those; do not silently build
  a subsystem inside a Phase 2 iteration.
- **Watch for the double-fault gap.** If the boot double-faults (an exception
  while `PS.EXCM` is set), the M2b DEPC gap (`docs/fidelity-gaps/firmware-mmu.md`)
  becomes live -- fix it (implement DEPC per `mmu_helper.c`) as a prerequisite,
  it is not out of scope.
- **Cap the walk.** If an iteration cannot identify a wall's expected behavior
  from the available artifacts, stop and surface it to the human rather than
  guessing -- an unexplained wall is a finding, not a stub.
- **Record H2.** When idle is reached, note `window_exceptions` in the final
  commit (the H2 observation: whether window overflow fired in real windowed
  code). It is an observation, not a gate.

### Phase 2 completion

- [ ] The `boot_to_idle` firmware-gated test asserts `reached_idle == true`,
  `funcs_entered` contains `_XAie_ExecuteCmd`, and prints `window_exceptions`.
- [ ] `cargo test --lib` green.
- [ ] Update `docs/fidelity-gaps/firmware-mmu.md` and the M2c findings note with
  the final map (`L`, `varway56` verdict) and the H2 observation.

---

## Final whole-branch review

After Phase 2 completes, dispatch the final whole-branch code review
(superpowers:requesting-code-review) on the most capable model, pointed at the
review package for the branch, then use superpowers:finishing-a-development-branch.

## Self-review notes (plan author)

- **Spec coverage:** load-offset `L` (Task 3, 6), reset-entry start (Task 6),
  `varway56=true` (Task 2), synthesized autorefill PT (Tasks 4, 5), system-aperture
  stubs (Phase 2), config extraction (Task 1), coherence gate to C entry (Task 6),
  idle gate (Phase 2 completion), all testing modes -- each maps to a task.
- **Empirical constant:** `L` is the one value not known at plan-write time; Task 6
  gives the exact procedure to pin it (candidate `0x5c`, confirmed by the gate,
  with a localization recipe if it differs). This is a derivation procedure, not a
  placeholder.
- **Phase 2 shape:** deliberately a procedure, not fabricated TDD tasks, because
  the walls are undiscovered; writing concrete stub code for unknown addresses
  would be the fabrication the No-Placeholders rule forbids.
