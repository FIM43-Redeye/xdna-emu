# M2b: Xtensa MMU-v3 Mechanism -- Design Spec

**Milestone:** M2b (second of the M2 three-way split: M2a base-ISA completion [DONE,
merged `7151ce74`], M2b MMU mechanism [this spec], M2c mapping reconstruction).

**Issue:** #140 (firmware emulation).

**Goal:** Implement a faithful Xtensa MMU-v3 (region-protection + hardware-autorefill
paging) model in the firmware interpreter, so instruction fetch and data load/store
translate virtual addresses through TLBs and a page-table walk exactly as silicon does.
The mechanism is validated on synthetic TLB/page-table fixtures where we control the PTE
data; it is NOT expected to boot the real firmware past the MMU wall (that needs the
PSP-defined page-table data M2c reconstructs).

**Overriding rule:** DERIVE FROM THE TOOLCHAIN. The authoritative specification for every
formula, constant, register layout, and cause code in this document is QEMU 10.2.1
`target/xtensa/mmu_helper.c` (+ `cpu.h`, `exc_helper.c`, `helper.c`, `overlay_tool.h`),
read at `scratchpad/qemu-src/qemu-10.2.1+ds/target/xtensa/`. Semantics oracle for
instruction behavior is the same tree's `translate.c`. Nothing here is invented; every
number below has a `mmu_helper.c` line anchor in the derivation notes.

---

## Why M2b, and why it will not (yet) boot past the wall

M2a completed the base ISA: the firmware now boots through its 42-instruction MMU-init
prologue (which programs `ITLBCFG`/`DTLBCFG`/`PTEVADDR`, installs region TLB entries via
`witlb`/`wdtlb`, then `jx` to virtual `0x20000340`) and stops at the first fetch from
`0x20000340` -- currently an unbacked read that decodes to `Op::Unknown`. Today all the
MMU-config instructions are logged no-ops; there is no TLB, no page-table walker, no
config-register storage.

M2b makes that machinery real. But a faithful mechanism does not resolve the wall on the
real firmware, and this is a derived fact, not a limitation to fix here:

- Boot jumps to virtual `0x20000340`. A faithful ITLB lookup misses (no installed entry
  covers it), triggering hardware autorefill.
- The walk computes the PTE address `pt_vaddr = (PTEVADDR | (vaddr >> 10)) & ~3 =
  (0x3c000000 | 0x80000) & ~3 = 0x3c080000`.
- `0x3c080000` is ~62 MB past the end of our 248 KB firmware image. The PTE load returns
  zero/garbage -> decodes to `paddr=0, attr=0` -> the original fetch resolves to physical
  `0x340` = ROM `isync` garbage (matching the prior "virt 0x20000340 -> phys 0x340 =
  garbage" finding).

So M2b's honest outcome on the real firmware is: the wall persists, but is now produced by
the *real mechanism* walking an *absent page table*, rather than by an unimplemented stub.
The missing PTE data is the PSP-defined virt->phys map -- M2c's deliverable. M2b's success
is measured on synthetic fixtures (below), plus a characterization of exactly what the live
autorefill computes on the real firmware, handed to M2c.

---

## Scope

### In scope
1. `Mmu` state: ITLB (7 ways), DTLB (10 ways), config registers, autorefill index.
2. Full-MMU address translation: per-way direct-mapped lookup, ASID/ring resolution,
   multi-hit detection.
3. Hardware autorefill page-table walk (the `get_pte` mechanism), including the recursion
   guard on the PTE's own translation.
4. `witlb`/`wdtlb` (install) and `iitlb`/`idtlb` (invalidate) as real TLB mutations.
5. MMU config registers `PTEVADDR`/`RASID`/`ITLBCFG`/`DTLBCFG` as real SR state, with
   `get_page_size` decode of the cfg registers and the `wsr.rasid` ring-0-forced-to-1 quirk.
6. Translation seam: `Cpu::translate(bus, vaddr, access)` chokepoint wired into fetch and
   all `mem.rs` load/store sites; `Bus` stays strictly physical.
7. TLB-fault exception raising through the existing `raise_general_exception` path, with the
   real Xtensa MMU cause codes and `EXCVADDR`, reusing the kernel vector.
8. **Double-exception handling (folds in carry-forward finding 9a):** a fault while
   `PS.EXCM` is already set vectors to `DoubleExceptionVector` (`vecbase + 0x3C0`) and saves
   PC to the double path, per `exc_helper.c`.
9. Per-fetch-byte-span translation so a page-straddling 2-3 byte instruction translates
   faithfully -- and, critically, fetch must NOT fault on speculative bytes beyond the
   instruction's real length. Xtensa encodes length in the first byte (the op0 nibble), so
   translate/fetch byte 0, determine the 2-vs-3-byte length, then translate only the bytes
   the instruction actually occupies. A 2-byte instruction ending exactly at a page boundary
   must never fault on a non-existent 3rd byte in the next (possibly unmapped) page.
10. **Real-firmware characterization handoff (final task):** run the real firmware with the
    live MMU and record what the autorefill computes (`PTEVADDR`, `pt_vaddr`, the PTE value
    read, the physical address the fetch lands on, and which ways the boot `witlb`/`wdtlb`
    installs target) into a handoff document for M2c.

### Out of scope (deferred)
- Reconstructing the real PSP page table / making the firmware boot past the wall (M2c).
- The region-protection-only and MPU-only translation paths (`get_physical_addr_region`,
  `get_physical_addr_mpu`) -- our firmware uses the full-MMU option; the others are modeled
  only insofar as the reset ways-5/6 fixed entries exist. (If characterization shows the
  firmware needs region-translation semantics distinct from MMU paging, that is a spec
  amendment, not silent scope creep.)
- `rtlb0`/`rtlb1` (read-TLB) and `ptlb`/`pptlb` (probe-TLB) instructions, UNLESS the boot
  prologue or command loop uses them (decode already recognizes the family; check during
  implementation and add if the firmware issues them -- otherwise YAGNI).
- QEMU-softTLB coherency machinery (`tlb_flush_page`) -- we have no host-side shadow TLB;
  our `translate()` is called fresh per access.

---

## Architecture

### Module & ownership
New `src/firmware/xtensa/mmu.rs` defines `Mmu`, owned as `pub mmu: Mmu` on `Cpu`
(`interp/mod.rs`), parallel to the existing `vecbase`/`epc1` fields. `RegFile` is untouched
-- it stays scoped to the windowed-ABI register mechanics. `Mmu` holds no reference to
`Bus`; the page-table walk receives `&mut Bus` as a parameter from the `Cpu::translate`
call, keeping ownership acyclic.

### The translation seam (Bus stays physical)
Two `Cpu` methods are the only places translation happens:

```
enum Access { Fetch, Load, Store }

// Returns the physical address, or a ready-to-propagate exception Step.
fn translate(&mut self, bus: &mut Bus, vaddr: u32, access: Access) -> Result<u32, Step>
```

- **Fetch** (`Cpu::step`, `interp/mod.rs:377`): translate `pc` (and each subsequent
  instruction byte's address, to handle page crossing) with `Access::Fetch` BEFORE
  `decode::decode`. An ITLB miss/fault returns `Step::Exception` immediately -- fetch never
  fabricates an `Op::Unknown` from an untranslated address again.
- **Load/store** (`interp/mem.rs`, ~9 sites): each computes its effective virtual address as
  today, calls `cpu.translate(bus, vaddr, Load|Store)?`, then hands the resulting **physical
  address** to the unchanged `bus.loadN/storeN`. The 16-bit composite loads/stores translate
  the (up to two) byte addresses they touch.

`Bus::region` and all its `loadN/storeN/peek8` methods are unchanged -- they now only ever
receive physical addresses. The `coverage_scan.rs` boot driver's `peek8` path (which walks
physical ROM offsets directly, not via `pc`) is unaffected: it is a physical scan by
construction.

### `Mmu` internal structure (all constants derived, MMU-v3 defaults)

```
struct TlbEntry { vaddr: u32, paddr: u32, asid: u8, attr: u8, variable: bool }

struct Mmu {
    itlb: [[TlbEntry; 8]; 7],     // MAX_TLB_WAY_SIZE = 8
    dtlb: [[TlbEntry; 8]; 10],
    ptevaddr: u32, rasid: u32, itlbcfg: u32, dtlbcfg: u32,
    autorefill_idx: u32,
}
```

Way layout (from `overlay_tool.h` `TLB_TEMPLATE`/`ITLB`/`DTLB` macros, MMU-v3):
- ITLB `nways=7`, DTLB `nways=10`.
- `way_size = { R, R, R, R, 4, (varway56?4:2), (varway56?8:2), 1, 1, 1 }` where `R =
  nrefillentries/4` (autorefill way size). We set the standard config: `varway56=false`,
  `nrefillentries` per the MMU-v3 default (the `is32` selector -- confirm the exact default
  from `overlay_tool.h` at implementation time; the addressing code branches on
  `nrefillentries==32`).
- Ways 0-3: autorefill, fixed 4 KB pages (`addr_mask=0xfffff000`), hardware-walk filled.
- Way 4: 4 entries, variable page size via `ITLBCFG/DTLBCFG` bits `[17:16]`.
- Ways 5/6: 2 entries each (varway56=false), variable page size via cfg bits `[20]`/`[24]`;
  hold the reset fixed-region entries.
- Ways 7-9 (DTLB only): single entry, `0xfffff000` mask.

Config knobs (`nways`, `way_size[]`, `varway56`, `nrefillentries`) are named constants in
`mmu.rs`, documented as the AMD-Xtensa-config swap point for the day we obtain the real
`XCHAL_*` values.

### Reset state (`mmu_helper.c:413-441`, `reset_tlb_mmu_*`)
- `rasid = 0x04030201` (ring0->ASID1, ring1->2, ring2->3, ring3->4).
- `itlbcfg = dtlbcfg = 0`, `autorefill_idx = 0`.
- All entries start `asid=0` (invalid), `variable=true`.
- Then, `varway56=false` only, ways 5/6 get four hard-wired `variable=false` entries:
  - i/d way5[0] = {vaddr `0xd0000000`, paddr 0, asid 1, attr 7}
  - i/d way5[1] = {vaddr `0xd8000000`, paddr 0, asid 1, attr 3}
  - i/d way6[0] = {vaddr `0xe0000000`, paddr `0xf0000000`, asid 1, attr 7}
  - i/d way6[1] = {vaddr `0xf0000000`, paddr `0xf0000000`, asid 1, attr 3}

### Translation algorithm (full-MMU path, `get_physical_addr_mmu` + `xtensa_tlb_lookup`)
Given `(vaddr, access)` with `dtlb = access != Fetch`:
1. **Lookup** across `nways`: for each way `wi`, `split_tlb_entry_spec_way` computes `(vpn,
   ei)` deterministically from `vaddr` and the way's page size; the slot hits iff
   `entry.vaddr == vpn && entry.asid != 0 && get_ring(entry.asid) < 4`. More than one hit ->
   `INST_TLB_MULTI_HIT_CAUSE`(17) / `LOAD_STORE_TLB_MULTI_HIT_CAUSE`(25). No hit -> miss.
2. **On miss, autorefill** (if not itself a PT-walk): `get_pte` computes `pt_vaddr =
   (ptevaddr | (vaddr>>10)) & ~3`, translates it with `Access::Load`, `mmu_idx=0`, and the
   **recursion guard `may_lookup_pt=false`** (the PTE's own translation may only be satisfied
   by resident entries; it can never trigger a nested walk). Load the 32-bit PTE from the
   resulting physical address via `bus.load32`. Decode: `ring=(pte>>4)&3`, `attr=pte&0xf`,
   `paddr=pte & 0xfffff000`, `asid = RASID_byte[ring]`. Install into way `(++autorefill_idx)
   & 3`, entry index `ei` from way-0 addressing; set `EXCVADDR=vaddr`. If `get_pte` fails
   (PT's own translation misses, or the PTE bus load fails), the original miss cause stands
   -- no distinct nested cause.
3. **Permission check:** `ring < mmu_idx` -> privilege cause (18 fetch / 26 load-store).
   `mmu_idx` = current CPU ring = 0 for this kernel-only firmware (`PS.RING` never set), so
   these trivially pass, but the check is implemented faithfully.
4. **Access check:** `access_bits = mmu_attr_to_access(attr)` masked (`~PAGE_EXEC` for data,
   `~(PAGE_READ|PAGE_WRITE)` for fetch). If the required permission is absent -> prohibited
   cause (20 fetch / 28 load / 29 store).
5. **Success:** `paddr = entry.paddr | (vaddr & ~addr_mask(wi))`.

`get_ring` (`mmu_helper.c:443-452`): return the ring `i` in 0..4 whose `RASID` byte equals
`asid`, else `0xff`. `mmu_attr_to_access` (`mmu_helper.c:576-606`): attr<12 -> READ, +EXEC
if bit0, +WRITE if bit1, cache policy from bits[3:2]; attr==13 -> RW+isolate; 12/14/15 -> no
access.

### `witlb`/`wdtlb`/`iitlb`/`idtlb` (`mmu_helper.c` `wtlb`/`itlb` helpers)
- Install (`witlb`/`wdtlb`): `AS` operand's low bits are the way index (`&7` ITLB / `&0xf`
  DTLB); `split_tlb_entry_spec_way` derives `ei`+`vpn` from the rest; `AT` operand is the
  PTE-format value (paddr|attr, same layout as an autorefill PTE). Refuse (log, no-op) if the
  target entry is `variable=false`.
- Invalidate (`iitlb`/`idtlb`): decode the same `AS` addressing; if the entry is
  `variable && asid != 0`, set `asid = 0`.

These replace the current `log::debug!` no-op bodies of the existing `Op::Witlb/Wdtlb/
Iitlb/Idtlb` arms in `interp/system.rs` (decode already extracts the `t`/`s` operands).

### Config registers
Add SR constants `PTEVADDR=0x53`, `RASID=0x5A`, `ITLBCFG=0x5B`, `DTLBCFG=0x5C` (SR numbers
from the decode tests in `decode/system.rs`; confirm `RASID`'s number against the firmware/
xtensa-modules.c during implementation -- it is not in the current boot vectors). Route
`wsr`/`rsr` for these into `Mmu` fields via new arms in `Cpu::write_sr`/`read_sr` (replacing
the current catch-all drop). `wsr.rasid` forces the ring-0 byte to `0x1` (`mmu_helper.c:77`).
`get_page_size(dtlb, way)` reads cfg bits: way4 `[17:16]`, way5 `[20]`, way6 `[24]`.

### Exception raising (reuse `raise_general_exception`, add double-fault)
TLB-fault causes flow through the existing `Cpu::raise_general_exception(faulting_pc,
cause)`, which already sets `EXCCAUSE`/`EPC1`/`PS.EXCM` and vectors to `vecbase+0x300`. Two
additions:
- Set `EXCVADDR` (new `Cpu`/`Mmu` field + SR 238) to the faulting vaddr before raising, per
  `exc_helper.c:71-76`.
- **Double-fault:** if `PS.EXCM` is already set at raise time, vector to
  `vecbase + 0x3C0` (`EXC_DOUBLE`) and save PC to the double path instead of the kernel
  vector (`exc_helper.c:48-69`). This closes carry-forward 9a.

New EXCCAUSE constants (`cpu.h:266-294`): `INST_TLB_MISS=16`, `INST_TLB_MULTI_HIT=17`,
`INST_FETCH_PRIVILEGE=18`, `INST_FETCH_PROHIBITED=20`, `LOAD_STORE_TLB_MISS=24`,
`LOAD_STORE_TLB_MULTI_HIT=25`, `LOAD_STORE_PRIVILEGE=26`, `LOAD_PROHIBITED=28`,
`STORE_PROHIBITED=29`.

---

## Testing strategy

All correctness tests are **hermetic and synthetic** -- they build an `Mmu`, install TLB
entries and/or a page table in a scratch RAM region we control, and assert translation
outcomes. This is what makes M2b verifiable independent of the unknown PSP map.

Coverage, family by family:
1. **Reset state:** RASID/cfg defaults; ways 5/6 fixed entries present with correct
   vaddr/paddr/asid/attr and `variable=false`; all other entries invalid.
2. **Lookup hit/miss/multi-hit:** install an entry, assert hit -> paddr with correct offset
   splice; assert miss on an uncovered vaddr; force two ways to cover one vaddr, assert
   multi-hit cause.
3. **ASID/ring:** entry with asid matching a RASID lane hits; after rewriting RASID so the
   lane no longer contains that asid, the same entry misses (context-switch invalidation);
   `wsr.rasid` ring-0-forced-to-1.
4. **witlb/wdtlb round-trip:** install via the instruction operands, read back a translation;
   assert way-index extraction from AS; assert `variable=false` entries refuse overwrite.
5. **iitlb/idtlb:** install then invalidate, assert subsequent miss; assert fixed entries
   can't be invalidated.
6. **Autorefill walk:** lay a synthetic page table in scratch RAM, point `PTEVADDR` at it,
   miss on a mapped vaddr, assert (a) the correct `pt_vaddr` formula, (b) the PTE decodes to
   the right paddr/attr/asid/ring, (c) the entry lands in the round-robin autorefill way, (d)
   `EXCVADDR` set. Assert the recursion guard: a PT-vaddr that would itself miss yields the
   original TLB-miss cause, not a nested one.
7. **Config page sizes:** program `ITLBCFG`/`DTLBCFG`, assert way-4/5/6 `get_page_size` and
   the resulting VPN mask / entry-index bit positions.
8. **Permissions:** attr-nibble decode table (exec/write/read/cache, attr 13 isolate, 12/14/
   15 no-access); privilege cause when `ring < mmu_idx` (construct with a nonzero mmu_idx
   fixture even though the firmware runs ring 0); prohibited causes on missing R/W/X.
9. **Seam integration:** a fetch through a mapped ITLB entry executes the instruction at the
   translated physical address; a fetch miss with no page table raises the ITLB-miss
   exception (not `Op::Unknown`); a page-straddling instruction translates both spans.
10. **Double-fault:** raise a TLB fault with `PS.EXCM` preset, assert `vecbase+0x3C0` vector.

The **real-firmware characterization** (final task) is NOT a pass/fail correctness test --
it is an observation run (firmware-gated, like the boot/coverage tests): boot the real
firmware with the live MMU, log `PTEVADDR`, the first autorefill `pt_vaddr`, the PTE value
read, the landed physical address, and the ways/entries the boot prologue's `witlb`/`wdtlb`
installed. Output is a handoff doc under `docs/superpowers/findings/` for M2c.

---

## File structure

- **Create** `src/firmware/xtensa/mmu.rs` -- `Mmu`, `TlbEntry`, `Access`, the lookup/refill/
  install/invalidate logic, config decode, all synthetic unit tests. Wire `mod mmu;` in
  `xtensa/mod.rs`.
- **Modify** `src/firmware/xtensa/interp/mod.rs` -- add `mmu: Mmu` and `excvaddr` to `Cpu`;
  add `Access` + `translate()`; wire fetch through `translate`; add SR arms for
  PTEVADDR/RASID/ITLBCFG/DTLBCFG/EXCVADDR; add the double-fault branch to
  `raise_general_exception`; add the new EXCCAUSE constants.
- **Modify** `src/firmware/xtensa/interp/mem.rs` -- route each load/store effective address
  through `cpu.translate(...)` before the bus call.
- **Modify** `src/firmware/xtensa/interp/system.rs` -- replace the `Witlb/Wdtlb/Iitlb/Idtlb`
  log-no-op bodies with real `Mmu` calls.
- **Modify** `src/firmware/mod.rs` (`boot_to_idle`) if needed for the characterization run's
  logging hook.

Bus (`mmio.rs`) and `regfile.rs` are unchanged.

---

## Success criteria
1. All synthetic MMU unit tests pass; `cargo test --lib` green with no regression.
2. Every formula/constant/cause code matches the `mmu_helper.c`/`cpu.h`/`exc_helper.c`
   derivation (the plan's oracle for each is the QEMU line, cited in tests).
3. Fetch and load/store are routed through `translate`; `Bus` receives only physical
   addresses; no untranslated-address `Op::Unknown` path remains for in-bounds virtual code.
4. Double-fault (9a) handled and tested.
5. Real-firmware characterization handoff document produced for M2c.
6. On the real firmware, boot reaching the wall now does so via the live autorefill
   mechanism (observable), not the old stub -- confirmed by the characterization run.
