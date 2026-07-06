# M2c iter16: the low `.text` has a piecewise VMA/LMA layout (+0x100 block)

**Date:** 2026-07-06
**Branch:** `feat/m2c-mapping-boot-to-idle`
**Commit:** `1f35935b`
**Status:** RESOLVED -- 0x588c wall cleared; boot 47761 -> 48144.

## The wall

Boot walled at `Unknown pc=0x588c word=0x000238e4`. The reaching trace:

```
0xd83b  L32r  a3, [0x3d30]     ; a3 = 0x1186c   (struct base, low-window literal)
0xd840  L32iN a3, a3, 0x24     ; a3 = mem[0x11890] = 0x588c
0xd842  Callx8 a3              ; call 0x588c  -> Unknown (mid-instruction)
```

The iter16 handoff hypothesis was "init wrote a corrupted pointer 0x588c
(should be 0x5880); the `& ~0xf == 0x5880` clue." **That hypothesis is wrong.**

## Root cause

`0x588c` is a CORRECT, compiled-in function pointer. `init` builds a 3-entry
dispatch table `{0x581c, 0x5858, 0x588c}` (+ metadata `8, 7`) by loading three
consecutive literals from the code-region literal pool and registering them via
a call:

```
0x200046b4  l32r a10, [0x200032e8]  ; = 0x581c   (file 0x3344, base +0x5c)
0x200046b7  l32r a11, [0x200032ec]  ; = 0x5858
0x200046ba  l32r a12, [0x200032f0]  ; = 0x588c
...          call8 (registrar)      ; stores them at [0x1186c + 0x1c/0x20/0x24]
```

The literal reads are correct -- the raw image at file 0x3344 holds exactly
`1c 58 00 00 58 58 00 00 8c 58 00 00`. The bug is where we FETCH the
pointed-to code.

**The firmware `.text` is not a single uniform file offset.** Most code is at
`file = vaddr + 0x5c` (the base PSP load offset, proven by the running code and
the `m2c_boot_reaches_c_entry` gate). But a block of small dispatch functions
around vaddr `0x581c`-`0x59xx` is stored at **`file = vaddr + 0x100`** -- a
localized **+0xa4** file shift. Classic explicitly-placed-section behavior: the
section is linked at a fixed low VMA but its LMA (file position) follows other
code.

Proof (xtdis oracle, AMD binutils FLIX config):
- At `+0x100`: file 0x591c / 0x5958 / 0x598c each decode as a clean
  `entry a1,a1,0x20` prologue with a coherent function body.
- At `+0x5c` (our model): those vaddrs land mid-instruction (e.g. 0x588c ->
  file 0x58e8 -> inside a `7ce4 movi.n`), which is the `Unknown 0x588c` wall.

This **contradicts** `image-structure-verdict.md`'s "single contiguous payload"
conclusion. The payload is one signed blob, but it is NOT a linear VMA image:
at least one section has VMA != LMA-linear. (The recon was about a *dropped high
segment*, which is still correct -- this is a different, internal, low-VMA
placement it did not probe for.)

## Why a phys-keyed overlay fails (the collision)

First attempt keyed the file-offset override on the PHYSICAL address. It
regressed boot (47761 -> 42243, into a bogus window overflow). Reason: the code
region (`0x2000_0000+`) maps to the SAME low physical range as the low window
(`phys = vaddr - 0x2000_0000`). The normal `+0x5c` `.text` runs via code-region
vaddrs `0x2000_55xx` -> phys `0x55xx`, which collides with the block's phys
range. A phys-keyed overlay corrupted that running code.

This is the exact collision `Bus::is_local_data` documents: "the local/image
split cannot be made on the physical address, because the code region and the
low window collide there." The override must be a **vaddr predicate**.

## Fix

A vaddr-keyed ROM fetch overlay (`Bus::fetch8`, wired into `Cpu::step`'s two
byte reads). A fetch whose *virtual* address is in `[LOW_TEXT_BLOCK_LO, HI)`
reads `vaddr + LOW_TEXT_BLOCK_FILE_OFFSET (0x100)`; every other fetch --
including code-region aliases of the same phys -- uses the normal path. Data
loads are untouched (low-window data already routes to `local_data`; the block
is code, never read as data on this path).

Bounds (`LOW_TEXT_BLOCK_LO = 0x581c`, `HI = 0x5d30`) are **empirically
determined** (walk-and-stub): the seam is code-to-code with no padding marker,
and the `$PS1` container has no segment table to derive exact extents from.
`LO = 0x581c` is the first confirmed +0x100 func; a lower `LO = 0x57bc` was
tried but is unnecessary and risks shadowing `+0x5c` low-window code, so it was
left at the confirmed edge. `HI` covers the Ghidra-delimited dense small-func
run.

## Deferred

- **FIXME(iter16):** reconstruct the firmware's full piecewise VMA/LMA layout
  (every seam) instead of this single hand-bounded block. Would eliminate the
  whole class of these walls but needs real section-extent RE (no segment table
  in the file).
- The trace-to-wall probe's *display* uses phys `peek8`, so it shows garbage
  disasm for block addresses (execution via `fetch8` is correct). Cosmetic.

## Next frontier (iter17)

Boot now reaches instr 48144 and walls at `pc=0x880` = `VECBASE(0x800)+0x80` =
the **WindowOverflow8** vector, which reads as `word=0` in our model. Newly
unlocked code (`entry a1,a1,0x30` at 0xc530) triggers a call8 window overflow
whose 8-register-window vector we do not populate. `window_exceptions=1`.
