# Phoenix 0x26d4 cache/page-root timeline

Date: 2026-07-12

Target: Phoenix/NPU1 `1502_00/npu.dev.sbin`

Firmware SHA-256: `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`

Branch: `feat/m2c-mapping-boot-to-idle`

Base commit: `dcc6009e`

## Verdict

**VERIFIED (reconstructed execution): the management firmware does not execute
an instruction-side cache operation, page-root/config write, ITLB operation, or
ITLB entry change between the early AT instruction that spans VMA `0x26d4` and
the later BASE `Entry` at VMA `0x26d4`.** The observed firmware-side MMU and
cache activity is exclusively D-side and does not address the low instruction
window.

Within the discriminator posed by the brief, the evidence therefore supports
an **external/HW instruction-bank or below-CPU agent**, not a firmware-active
cache/page-root/pinned-TLB view switch.

This is a bounded negative, not a claim that every AMD-private selector has
been enumerated. It rules out the architected Xtensa mechanisms visible to the
probe: all 19 decoded cache-op classes, `PTEVADDR`, `RASID`, `ITLBCFG`,
`DTLBCFG`, explicit ITLB/DTLB writes and invalidations, and actual ITLB/DTLB
entry mutations. A vendor MMIO bank selector below those interfaces remains a
separate possibility.

## Method and reconstruction boundary

The env-gated test
`m2c_probe_26d4_cache_pageroot_timeline` starts when the early AT context enters
`0x2630`, detects the instruction whose byte span includes `0x26d4`, and then
records an ordered `n / pc / op / detail` timeline through the service sink at
`0x7fec`. It records:

- every decoded I-side or D-side cache operation and its effective address;
- writes to `PTEVADDR`, `RASID`, `ITLBCFG`, and `DTLBCFG`;
- explicit `WITLB`, `WDTLB`, `IITLB`, and `IDTLB` operations;
- every actual ITLB and DTLB entry change, including automatic refills; and
- call, entry, and return boundaries.

**Reconstruction limitation:** no static instruction view currently carries
the reconstructed boot through both epochs. To traverse the already-proven
path, this test rides the existing runtime-view discriminator's two
counterfactual selections. They are emitted as `HARNESS_VIEW` events, are
explicitly marked “not firmware,” and are excluded from all selector counts.
The probe does not change registers, firmware data, MMU state, cache state, or
production behavior. The transport is not evidence for the mechanism and is
not proposed as a fix; it only exposes the firmware instructions on both sides
of the missing view transition.

## Ordered timeline

The complete reproducible log is written by the command under “Reproduction.”
The mechanism-bearing extract is:

```text
n=47672 pc=0x0026d3 op=VIEW_EPOCH
  early AT Movi { t: 4, imm: 255 }, len=3, spans byte 0x26d4
  PA=0x000026d4 ITLB[6][0]={vaddr=0,paddr=0,asid=1,attr=3}
  PTEVADDR=0x3c000000 RASID=0x04030201
  ITLBCFG=0x00000000 DTLBCFG=0x00030000

n=49473 pc=0x002ad6 op=Wdtlb { t: 7, s: 10 }
  AS=0x08a80009 AT=0x0001300a
n=49473 pc=0x002ad6 op=TLB_CHANGE
  DTLB[9][0] non-autorefill:
  {vaddr=0,paddr=0,asid=0,attr=0} ->
  {vaddr=0x08a80000,paddr=0x00013000,asid=1,attr=0xa}

n=52057 pc=0x004a33 op=TLB_CHANGE
  DTLB[1][0] autorefill: 0x27010000 identity -> 0x27210000 identity
n=52206 pc=0x00563e op=TLB_CHANGE
  DTLB[2][0] autorefill: 0x27200000 identity -> 0x27010000 identity
n=52366 pc=0x0029fb op=TLB_CHANGE
  DTLB[3][0] autorefill: 0x27270000 identity -> 0x27200000 identity

n=53209..53225 pc=0x08b0e720 op=Dhwbi
  nine D-side operations, EA=0x0000fae0..0x0000fb60 in 0x10 steps

n=53567 pc=0x00283b op=Callx8 { s: 5 }
n=53578 pc=0x00878a op=Call8 { target: 0x00c530 }
n=53579 pc=0x00c530 op=Entry { s: 1, imm: 48 }
n=53596 pc=0x00c55c op=Callx8 { s: 8 }
n=53603..53619 pc=0x08b0e720 op=Dhwbi
  nine D-side operations, EA=0x0000fae0..0x0000fb60 in 0x10 steps
n=53630 pc=0x00c56e op=Call8 { target: 0x007fc4 }
n=53631 pc=0x007fc4 op=Entry { s: 1, imm: 32 }
n=53639 pc=0x007fe1 op=Call8 { target: 0x008c6c }
n=53640 pc=0x008c6c op=HARNESS_VIEW
  select established BASE service overlap; not firmware
n=53640 pc=0x008c6c op=Entry { s: 1, imm: 32 }
n=53672 pc=0x008cba op=RetwN
n=53673 pc=0x007fe4 op=Call8 { target: 0x00d7f0 }
n=53783 pc=0x007fe7 op=Call8 { target: 0x0026d4 }

n=53784 pc=0x0026d4 op=VIEW_EPOCH
  later BASE Entry
  PA=0x000026d4 ITLB[6][0]={vaddr=0,paddr=0,asid=1,attr=3}
  PTEVADDR=0x3c000000 RASID=0x04030201
  ITLBCFG=0x00000000 DTLBCFG=0x00030000
n=53784 pc=0x0026d4 op=HARNESS_VIEW
  select established BASE 0x26d4 view; not firmware
n=53784 pc=0x0026d4 op=Entry { s: 1, imm: 80 }
n=53813 pc=0x002734 op=Call8 { target: 0x00c530 }
n=53874 pc=0x007fec op=MARK
  service sink a7=6 EXCCAUSE=1 EPC1=0x08b0e713
```

The probe's guarded interval summary is:

```text
selector_i_cache_ops=0
selector_d_cache_ops=18
selector_root_writes=0
selector_root_changes=0
selector_tlb_ops=1
selector_itlb_ops=0
selector_dtlb_ops=1
selector_tlb_changes=4
selector_itlb_changes=0
selector_dtlb_changes=4
selector_non_autorefill_changes=1
```

### What the non-zero events mean

**VERIFIED:** the single explicit TLB operation is the `WDTLB` at `n=49473`,
`pc=0x2ad6`. It installs DTLB way 9 mapping virtual `0x08a80000` to physical
`0x00013000` with attribute `0xa`. It neither changes the ITLB nor addresses
the low `0x26d4` instruction page.

**VERIFIED:** the other three TLB changes are ordinary DTLB autorefill-way
replacements for identity-mapped `0x2701xxxx`, `0x2720xxxx`, and `0x2721xxxx`
device pages. No ITLB entry changes during fetch or instruction retirement.

**VERIFIED:** the 18 cache operations in the selector interval are two
identical nine-line `Dhwbi` walks at `pc=0x08b0e720`, over local data
`0xfae0..0xfb60`. There are no I-side cache operations. A third identical
D-side walk occurs after the later BASE entry (`n=53844..53860`) and is logged
but deliberately excluded from the between-epoch count.

**VERIFIED:** both view-epoch samples translate `0x26d4` to PA `0x26d4` through
the same `ITLB[6][0]`, with identical `PTEVADDR`, `RASID`, `ITLBCFG`, and
`DTLBCFG` values. No write to any of those four registers and no actual root
value change occurs in the interval.

## Why this selects the external/HW class

The later view does not follow an architected firmware-side I-cache flush,
invalidate, or prefetch. It does not follow a page-root/ASID change. It does
not follow an ITLB write, invalidation, refill, or pinned-entry mutation. The
only explicit non-autorefill TLB operation is D-side and maps a different
address space; the only cache maintenance is D-side and touches local mailbox
data.

Consequently, none of the firmware-visible selectors enumerated in the brief
accounts for AT bytes at the early identity-mapped fetch and BASE bytes at the
later identity-mapped fetch. The smallest surviving class is a selector below
the standard CPU MMU/cache instruction interface: an instruction-memory bank,
an external agent changing the presented bank, or an AMD-private below-CPU
alias mechanism.

## Fidelity holes

1. **VERIFIED emulator gap:** all 19 decoded cache-op classes remain no-ops in
   `src/firmware/xtensa/interp/system.rs`. The repeated `Dhwbi` plus `Dsync`
   helper proves that the firmware deliberately maintains data-cache
   coherence. This trace does not show a resulting stale-data failure, so it
   does not establish the exact missing data-cache behavior; it does establish
   that treating the operations as semantically irrelevant is not a faithful
   architectural model.
2. **VERIFIED inventory result:** page-root/config writes and ITLB mutations are
   absent in this bounded transition. The one pinned/non-autorefill mutation is
   DTLB way 9 and is recorded with its exact before/after fields.
3. **CLAIMED remaining mechanism:** an AMD-private MMIO bank selector is not
   decoded by an Xtensa-special-register/cache-op inventory. Its existence is
   not demonstrated by this run.
4. **OBSERVED reconstruction limitation:** reaching both views still requires
   the test-only counterfactual transport described above. The transport makes
   the firmware instruction interval observable; it cannot prove what supplies
   the view on silicon.

No production cache semantics were changed in this pass.

## Single next observation

The highest-value next observation is an **MMIO-write timeline over the same
bounded interval**, reduced to writes whose effective address is outside normal
local data/stack and annotated with the caller chain. Correlating any such
write with the AMD-private `sram_alias`/`mpnpu/mmu` seam identified in
`2026-07-12-memory-architecture-model.md` would either name a below-standard-MMU
bank selector or earn its absence. This remains entirely distinct from the
closed PSP-loader and CPU-self-modifying-code investigations.

## Reproduction

```bash
mkdir -p build/experiments/firmware-re
XDNA_FW_PROBE=1 cargo test --lib \
  m2c_probe_26d4_cache_pageroot_timeline -- --nocapture \
  > build/experiments/firmware-re/0x26d4-cache-pageroot-timeline.log 2>&1
```

The targeted run reached `0x7fec` at `n=53874` and passed. Final verification
ran `cargo test --lib`: **4091 passed, 0 failed, 30 ignored** in 44.84 seconds.
The additional passing test over the 4090 baseline is this env-gated probe.
