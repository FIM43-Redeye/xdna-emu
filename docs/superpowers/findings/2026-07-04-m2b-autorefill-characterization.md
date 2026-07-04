# M2b Task 10: real-firmware autorefill characterization (M2c handoff)

**Date:** 2026-07-04  **Issue:** #140 (firmware emulation, M2b Xtensa MMU-v3 mechanism)
**Status:** OBSERVATION run, not a pass/fail correctness test. The full M2b MMU
(Tasks 1-9) is live; this records what it actually computes booting the real
firmware, as the empirical starting point for M2c (mapping reconstruction).
**Test:** `firmware::boot_tests::characterize_real_firmware_autorefill`
(`src/firmware/mod.rs`), firmware-gated (skips cleanly if the image is absent).
The real image (`xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin`) is
present on this devbox, so the test ran (not skipped). Full run: `instructions
executed = 43`, `stop reason = Exception cause=16 vector_pc=0x300` (details below).

## 1. PTEVADDR after the prologue

`0x3c00_0000` -- exactly the value assumed since M2a (the design spec's own
derivation) and confirmed here empirically: the boot prologue's `wsr.ptevaddr`
programs it before the `jx` that leaves ROM.

## 2. The firmware's own MMU-setup operands

Every `witlb`/`wdtlb`/`iitlb`/`idtlb` the 42-instruction prologue issues,
recorded with its PC and operands (AS = way index + VPN, AT = paddr|attr for
the two install ops; the invalidate ops take no AT):

| PC | op | way | AS | AT |
|---|---|---|---|---|
| `0x33c` | witlb | 5 | `0x20000005` | `7` |
| `0x342` | wdtlb | 5 | `0x20000005` | `7` |
| `0x357` | iitlb | 6 | `0x20000006` | -- |
| `0x35a` | idtlb | 6 | `0x20000006` | -- |
| `0x360` | iitlb | 6 | `0x40000006` | -- |
| `0x363` | idtlb | 6 | `0x40000006` | -- |
| `0x369` | iitlb | 6 | `0x60000006` | -- |
| `0x36c` | idtlb | 6 | `0x60000006` | -- |
| `0x372` | iitlb | 6 | `0x80000006` | -- |
| `0x375` | idtlb | 6 | `0x80000006` | -- |
| `0x37b` | iitlb | 6 | `0xa0000006` | -- |
| `0x37e` | idtlb | 6 | `0xa0000006` | -- |
| `0x384` | iitlb | 6 | `0xc0000006` | -- |
| `0x387` | idtlb | 6 | `0xc0000006` | -- |
| `0x38d` | iitlb | 6 | `0xe0000006` | -- |
| `0x390` | idtlb | 6 | `0xe0000006` | -- |

Confirms and sharpens the Task 8 finding (which had the AS/AT values but not
the PCs or the full pairing): every one of these 16 calls targets way 5 or 6
(`AS & 0x7` for the I-side, `AS & 0xf` for the D-side, both give 5 or 6),
which this MMU model hardwires (`Mmu::load_fixed_ways56`, `variable=false`,
`varway56=false` default) -- so **every one is currently a no-op**
(`write_tlb`/`invalidate_tlb` both refuse fixed slots; `pt_lookup =
proc.cpu.mmu.lookup(pt_vaddr, true)` after the run confirms `Err(24)`, nothing
in the DTLB covers the page-table region).

Decoded intent (masking AT/AS through `Mmu::addr_mask`/`decode_pte` the way
`write_tlb` would, were the slot writable):

- **`witlb`/`wdtlb` (way 5, one call each, same operands):** VPN = `0x20000005
  & 0xf8000000 = 0x20000000` (way 5's 128 MB granularity), paddr = `7 &
  0xf8000000 = 0`, attr = `7` (cached RWX). **If way 5 were writable, this
  installs virtual `0x20000000..0x27ffffff -> physical 0..0x07ffffff`, RWX** --
  i.e. the entire high region truncated straight onto low physical memory by
  masking off the top 3 bits, no page-table walk at all. That would resolve
  the `jx` target `0x20000340` to physical `0x340`, which is exactly the
  pre-M2b "virt `0x20000340` -> phys `0x340` = garbage" observation the design
  spec already named -- i.e. this specific witlb is a broad region-attribute
  mapping, not the real code-region map (phys `0x340` is ROM header/literal
  bytes, not the firmware's actual command-loop code).
- **`iitlb`/`idtlb` (way 6, seven pairs, invalidate-only):** AS cycles through
  seven VPNs (`0x20000000, 0x40000000, ..., 0xe0000000`, i.e. the top nibble
  stepping by `0x20000000` each time) invalidating whatever entry occupies
  that AS's computed way-6 slot. On **this** model, way 6's entry index is
  `(vaddr >> 28) & 0x1` (`varway56=false`), and every one of those seven VPNs
  has bit 28 clear -- so all seven calls collapse onto the **same** slot
  (way 6, entry 0), which is also fixed and so a no-op regardless. A firmware
  routine invalidating the same single slot seven times in a row with a
  changing tag is a strange thing to write deliberately; far more likely is
  that the real core's way 6 has more addressable entries than this model's
  two (i.e. `varway56=true` gives way 6 a wider `ei` derivation, so those seven
  calls hit seven distinct slots -- a "clear the whole way before repopulating
  it" idiom, which is a completely ordinary MMU-setup pattern). This is
  additional evidence for the M2c hypothesis below, not just the witlb/wdtlb
  pair.

## 3. The autorefill outcome at the jx target (0x20000340)

After the (no-op'd) prologue, the `jx` lands on virtual `0x20000340`:

- **ITLB lookup misses** (nothing covers it -- fixed ways only cover
  `0xd0000000+`/`0xe0000000+`/`0xf0000000+`; the provisional low-region way-4
  map only covers the boot ROM page).
- **Autorefill computes** `pt_vaddr = (PTEVADDR | (0x20000340 >> 10)) & ~3 =
  (0x3c000000 | 0x80000) & ~3 = 0x3c080000` -- exactly the design spec's
  pre-derived value, now confirmed by the live mechanism rather than by hand
  calculation.
- **The PTE's own address is itself unmapped**: `pt_vaddr` (`0x3c080000`) is
  probed directly against the DTLB (`Mmu::lookup(pt_vaddr, dtlb=true)`, a
  read-only check, no bus access) and returns `Err(24)`
  (`LOAD_STORE_TLB_MISS`) -- the page-table walk's own translation misses
  before ever reading a PTE byte. Per `get_pte`'s recursion guard
  (`may_lookup_pt=false`), a miss here is NOT itself autorefilled; it just
  makes `get_pte` return `None`.
- **Final fault**: the ORIGINAL miss cause stands -- `EXCCAUSE_INST_TLB_MISS`
  (16) at `vaddr=0x20000340` -- vectoring to the KernelExceptionVector
  (`vecbase=0 + 0x300 = 0x300`). Boot stops there (43 instructions executed:
  42-instruction prologue + the one exception-raising step).

This exactly matches the M2a-era design-spec prediction (`docs/superpowers/
specs/2026-07-04-m2b-mmu-mechanism-design.md`), now empirically walked by the
real mechanism rather than assumed. `firmware::boot_tests::
boots_real_firmware_from_pinned_entry` (the M1.7 milestone test, still
unmodified since Task 8) independently confirms the very next step past this
point: the handler at `0x300` runs one bogus-but-decodable instruction and
hits a real `Op::Unknown` at `0x303` -- a different, later wall (no installed
exception handler / no modeled MMIO for it), out of scope for M2b/M2c.

## 4. Synthesis for M2c

- **(a) The mechanism is live and correct.** Full-MMU translation, per-way
  lookup, ring/ASID resolution, and the hardware autorefill walk (including
  its recursion guard) all run exactly as `mmu_helper.c` specifies, confirmed
  end-to-end against the real firmware image, not just synthetic fixtures.
- **(b) With `varway56=false`, the firmware can install *neither* its
  high-region map (`witlb`/`wdtlb`, way 5) *nor* clear/populate its
  page-table-adjacent way (`iitlb`/`idtlb`, way 6) as more than a single
  collapsed slot** -- so autorefill has nothing to walk once it reaches
  `pt_vaddr`, and the wall is produced by a faithful mechanism hitting a truly
  absent page table, not a missing feature.
- **(c) The `varway56` hypothesis is the central M2c question, and section 2's
  operand table is now the concrete evidence for it, not just a suspicion.**
  If the real AMD/Ryzen-AI Xtensa core has `varway56=true`:
  - Way 5 becomes software-writable with the operands already captured here:
    the firmware would install `0x20000000..0x27ffffff -> 0..0x07ffffff`
    (RWX) via its own `witlb`/`wdtlb` -- a real, observed mapping, not one
    M2c has to invent. But per section 2's analysis, this specific mapping
    resolves the `jx` target to physical `0x340` (ROM header bytes), which is
    the ALREADY-KNOWN "garbage" result from before M2b existed -- so this
    mapping alone is necessary but not sufficient; it establishes a *default*
    broad region attribute, not the specific code page.
  - Way 6's seven invalidate calls, with a wider `ei` derivation, would each
    target a distinct slot -- consistent with "clear the way before
    installing the real entries," which implies there are follow-on `witlb`/
    `wdtlb` calls to way 6 (or elsewhere) this scan didn't need to reach,
    because the prologue as captured ends at the `jx` before any such
    installs would occur (if they exist at all in this firmware -- worth
    checking if M2c re-walks further; this run does not by itself prove more
    installs exist, only that the invalidate pattern is shaped like prep for
    them).
  - **Reshapes the prior framing.** The M2b Task 8 report and the M2b design
    spec both described the code-region map as "PSP-defined and absent from
    every artifact we hold." That still holds for the actual page-table
    contents at `pt_vaddr` (`0x3c080000` and beyond -- nothing in this 248 KB
    image plausibly backs a table that far past the image end, `varway56` or
    not). But the *high-region region-attribute* map (way 5) is NOT absent
    once `varway56=true` is assumed -- it is sitting in this firmware's own
    prologue, in registers we can already read. M2c's job narrows from
    "reconstruct an entirely absent map from nothing" to "confirm
    `varway56=true` on the real core, apply the now-known way-5/6 operands,
    and separately account for why they resolve to ROM garbage rather than
    the real command-loop code" -- plausibly because the REAL page-table walk
    (once `pt_vaddr`'s region is genuinely backed, which requires further
    config this scan didn't reach) supersedes the coarse way-5 region
    mapping for the specific `jx` target, the same way autorefill entries
    (ways 0-3) take priority in real hardware once installed. That
    displacement mechanism -- not just the raw operands -- is what M2c should
    resolve next.

## Sanity assertions in the test (not boot-success gates)

- `PTEVADDR == 0x3c00_0000` after the run.
- At least one `witlb`/`wdtlb`/`iitlb`/`idtlb` was observed (an empty list
  would mean the boot desynced before reaching them).
- Every observed op targets way 5 or 6 (a different way would overturn the
  varway56 framing above).
- The computed `pt_vaddr` is confirmed unmapped via a direct, read-only DTLB
  probe (`Mmu::lookup`).
- The run stops at a recognizable MMU-wall outcome (an `Exception` or
  `Unknown`), not a step-cap timeout or a `Wait`.

All five hold on the current tree. `cargo test --lib`: 3894 passed, 0 failed,
30 ignored (full crate green, +1 over Task 9's 3893 for this new test).
