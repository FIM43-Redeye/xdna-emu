# Fork A: the MMU is not the 0x8cae code-view selector

Date: 2026-07-11
Target: Phoenix/NPU1 firmware `1502_00/npu.dev.sbin`
SHA-256: `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`
Branch: `feat/m2c-mapping-boot-to-idle`

## Question

The alive publisher (rooted `0x8c98`, AT-framed) and the syscall service path
(rooted `0x8c6c`, BASE-framed) collide at VMA `0x8cae`: they require different
file bytes at the same virtual address (finding
`2026-07-11-alive-sram-overlay-collision.md`). A page remap could supply two
physical banks selected by the ITLB -- IF the firmware reprograms the ITLB for
that page between the two executions. Fork A tests exactly that, read-only.

## Result: identical ITLB, no remap. The MMU is not the selector.

`m2c_probe_itlb_code_view_selector` logs every executed Witlb/Iitlb and samples
the ITLB translation of `0x8cae` at the publisher and the service:

```
-- all executed ITLB modifications (9) --   (all at n=993..1026, reset-time)
n=993  pc=0x2e0 WITLB va=0x20000005 data=0x7
n=1002..1020    IITLB va=0x20000006 / 0x40.. / .. / 0xe0000006   (region invalidations)
n=1026 pc=0x343 WITLB va=0x20000005 data=0x7
-- ITLB modifications touching the 0x8000..0x9000 code page (0) --
-- ITLB view of collision cell 0x8cae at publisher vs service --
PUBLISHER (x5, pc 0x8c98/0x8cac): PA=0x8cae via itlb[6][0] {vaddr:0,paddr:0,asid:1,attr:3}
SERVICE   (pc 0x7fe1,0x8c6c,0x8cae,0x8cb1): PA=0x8cae via itlb[6][0] {vaddr:0,paddr:0,asid:1,attr:3}
```

VMA `0x8cae` translates to PA `0x8cae` (identity) via the reset-default way-6
region entry, unchanged for the whole boot. No ITLB op touches the code page.
Publisher and service fetch the same PA via the same entry.

## Consequence

Since `0x8cae` is identity-mapped for both paths, real silicon has exactly ONE
physical byte at PA `0x8cae`, and both paths fetch it. The paths therefore
cannot genuinely need different bytes there on hardware. Two possibilities
remain:

1. A still-misframed upstream root spuriously routes one path (likely the
   syscall service entry `0x7fe1 -> 0x8c6c`) through `0x8cxx`; the collision is a
   reconstruction artifact masked by the 0xa4 decode ambiguity.
2. An exotic sub-MMU instruction-fetch bank configured by the PSP (non-standard
   for Xtensa).

Fork A eliminated MMU-remap; the earlier finding
(`2026-07-11-firmware-vma-file-map-not-statically-recoverable.md`) eliminated
memory-copy overlays (zero stores to any `+0x100` VMA). Both standard Xtensa
overlay mechanisms are now ruled out. The tie-breaker for both remaining
possibilities is the same artifact the flat `$PS1` image lacks: the PSP loader's
scatter/placement map (what physical bytes land at each PA). That is the PSP-RE
escalation, now justified by elimination rather than assumed.

## Reproduction

```
XDNA_FW_PROBE=1 XDNA_FW_MAX=100000 cargo test --lib \
  m2c_probe_itlb_code_view_selector -- --nocapture
```

Read-only: logs ITLB ops and samples the fetch translation of `0x8cae`. No
production decode/MMU/scheduler/overlay change.
