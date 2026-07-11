# Firmware VMA->file map is not statically recoverable from the image

> **SUPERSEDED PREMISE (2026-07-11, same day).** This finding's premise -- that
> we need a `VMA->file` map for the line-0 service path -- is wrong. That path
> (into `FUN_00008c68` at 0x8c6c) is **never executed in the natural boot**; it
> is an artifact of injecting a line-0 interrupt, and the real code at 0x8c98+ is
> the AT publish helper. There is no map to recover. The static-analysis facts
> below (reset coherence, PSP flat-load, `$PS1` has no scatter table) remain
> valid and useful; the "blocked at a fundamental limit" conclusion does not.
> See `2026-07-11-goalive-reached-isr-collision-was-injection-artifact.md`.

**Date:** 2026-07-11
**Status:** SUPERSEDED PREMISE (see banner). Static-analysis facts retained.
**Branch:** `feat/m2c-mapping-boot-to-idle` (unmerged).

## Executive verdict

The `+0x5c`/`+0x100` vaddr-keyed overlay heuristic that carried the firmware
interpreter from reset to the go-alive `waiti` (0x5645) has hit a **provable
fundamental limit**: no static `VMA -> file_delta` interval table can serve the
whole boot, because two genuinely-executed paths require *different bytes at the
same virtual address*.

- Publish path at VMA `0x8cb4` needs file `0x8db4` (`MoviN a11,-1`).
- Line-0 interrupt-service path at the same VMA `0x8cb4` needs file `0x8d10`
  (`Addmi a5,a5,0x1000`).
- The same split appears in L32R pools: literal `0x354c` needs file `0x364c`
  (`0x000fff28`) for publish vs file `0x35a8` (`0x0000f9a0`) for service.

One virtual address cannot hold two byte streams. Ordinary GNU `AT()` placement
(VMA != LMA) alone cannot explain it. Either (1) some canonical VMAs in one call
graph remain shifted by `0xa4` (= `0x100 - 0x5c`) -- i.e. a still-misframed root,
where the ambiguity propagates up every call chain -- or (2) Phoenix has a
runtime **executable bank/view selector** (Xtensa overlay-manager style) that
changes the backing of a VMA between task and interrupt execution. **The image
alone cannot distinguish these two.**

## The mechanism (verified)

- **Reset is coherent under a uniform `+0x5c` physical view.** Runtime VMA
  `0x1a4` = file `0x200` (reset entry). Startup programs `ITLBCFG=0`,
  `DTLBCFG=0x30000`, `PTEVADDR=0x3c000000`, `WITLB/WDTLB` mapping virtual
  `0x20000000 -> phys 0`, seven paired ITLB/DTLB invalidations, then
  `JX 0x20000340` -> phys `0x340` (file `0x39c`).
- **No code scatter-copy in startup.** The continuation does DTLB setup + cache
  prefetch (`DPFL`/`IPFL`); no `RITLB`/`RDTLB`, no relocation loop, zero stores
  to any `+0x100` VMA across a full boot.
- **The `0xa4` is NOT an MMU effect.** Xtensa translation always preserves the
  page offset (`paddr_base | (vaddr & !mask)`; `src/firmware/xtensa/mmu.rs`), so
  no TLB page or region can add a sub-page `0xa4` within a page. The overlay is
  purely a fetch/L32R file-offset selector keyed on VMA
  (`src/firmware/mmio.rs:220`).
- **The host driver hands the PSP an opaque blob.** `PSP_VALIDATE` +
  `PSP_START_COPY_FW` receive only address+size, no section list
  (`xdna-driver/src/driver/amdxdna/aie_psp.c`, `aie2_psp.c`). The scatter
  placement (file -> IRAM) lives entirely in the on-platform PSP loader.
- **The `$PS1` container carries no map.** Pure signed-blob wrapper: nonce,
  magic at 0x10, size at 0x14/0x50, GUIDs, hashes at 0xd0. `0x58` (a plausible
  load-address slot) is zero across all five firmware versions. No scatter
  table, no ELF/program headers (stripped).

## What IS recoverable (cross-version + execution)

Five version labels reduce to three layouts (`1502_00`; `17f0_10`==`17f0_11`
except file `0x2cc8..0x2cca`; `17f1_10`==`17f2_10`). Defensible `+0x100`
boundaries (high confidence, cross-version + RFE/entry anchors):

| VMA range | File extent | Evidence |
|---|---|---|
| `0x0800..0x0980` | `0x0900..0x0a80` | six `0x40` window-vector slots, conserved |
| `0x2630..0x2b51` | `0x2730..0x2c51` | executable bound; final RFE at `0x2b4e` |
| `0x3500..0x3520` | `0x3600..0x3620` | object extent (delta needs live L32R too) |
| `0xdf98..0xe0b1` | `0xe098..0xe1b1` | 281-byte extent matches `17f0` |
| `0xe1fc..0xe334` | `0xe2fc..0xe434` | cross-version blocks + RFE/next-entry |

**One correction landed this session:** `CTXSW_CALLEE_HI` was `0x2bf5`, exactly
`0xa4` above the real executable bound. The function's final RFE is at `0x2b4e`
(bytes `00 30 00`), so the section ends at `0x2b51`. Shrinking the overlay to
`0x2b51` keeps every test green (including the ctxsw `+0x100` guard and the
boot-to-waiti guard) -- the `[0x2b51,0x2bf5)` tail was an over-claim, the same
bug class as the `0x8c98` collision but caught before it bit. Fixed in
`load_m2c`.

The remaining go-alive ranges **cannot be formally bounded** from cross-version
diffs -- absolute pointers establish section *roots*, not *ends*, and every
self-consistent subgraph is `0xa4`-ambiguous (both framings decode coherently, so
a clean decode is necessary but not sufficient). `coherence_mapper.rs` is an
execution-calibrated oracle, not an authoritative linker map.

## What would settle it (external ground truth; none on disk)

1. **Unstripped Phoenix MPNPU firmware ELF + linker `.map`** for release
   `1.5.5.391` / package `1502_00` (`p_offset`/`p_vaddr`/`p_paddr`, linker
   script, any Xtensa overlay-manager tables). Cleanest; likely does not exist
   publicly.
2. **The Phoenix NPU PSP `PSP_START_COPY_FW` loader** (on-platform PSP/AGESA
   component behind the scratch-register protocol; not shipped by `amdxdna` or
   `linux-firmware`).
3. **A one-shot JTAG/OCD or internal-trace capture** on real Phoenix (never BAR0
   polling) recording instruction-fetch physical address + returned bytes + any
   bank selector at VMA `0x8cb4`, during natural publish AND inside the line-0
   service callback. A pre-start PSP DMA trace comparing file `0x900 -> IRAM
   0x800` and file `0x4b0c -> IRAM 0x4a0c` would independently settle scatter
   placement.

## Where the arc stands

Everything upstream of this limit is solid and banked: the firmware boots its
own code to a valid `mgmt_mbox_chann_info` and rests at the pre-publish
`waiti 0x5645`; the go-alive gate wakes on interrupt line 0; the `0x27200800`
block is a ZDMA channel whose interrupt registers the firmware configures (see
`2026-07-11-firmware-goalive-async-completion-characterization.md`). The
remaining step -- letting a delivered line-0 interrupt run the service path to
publish + `FW_ALIVE_OFF` -- is blocked only by the VMA/file selector we cannot
recover from the image. Resolving it requires one of the three artifacts above.

Reproduction instruments (all `XDNA_FW_PROBE`-gated, in
`src/firmware/boot_tests/`): `m2c_probe_waiti_wake_condition` (+`XDNA_FW_WAKE=1`),
`m2c_probe_decode_c68_framings`, `m2c_probe_framing_collision_overlap`,
`m2c_verify_0xa4_shift`, `m2c_verify_codex_field_packer_and_fp_load`,
`completion_grounding::re_probe_goalive_completion_values`.
