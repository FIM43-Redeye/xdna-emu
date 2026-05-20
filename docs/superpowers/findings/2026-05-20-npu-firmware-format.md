# NPU firmware format characterization (Phoenix, FW 1.5.5.391)

**Status:** Initial format characterization complete (2026-05-20).  Body
confirmed plaintext, code identified as ARM Thumb-2, function entry
points enumerated.  Full reverse engineering not attempted; this doc
captures the prerequisites that make it feasible.

**Related:**
[`2026-05-20-amdxdna-tdr-recovery-incomplete-on-phoenix.md`](2026-05-20-amdxdna-tdr-recovery-incomplete-on-phoenix.md)
(the investigation that led to wanting to look at FW internals).

## File layout

The firmware blob lives at `/lib/firmware/amdnpu/1502_00/`:

| File | Size | Notes |
|------|------|-------|
| `npu.dev.sbin` | 248,592 B | Uncompressed copy ("dev" is misleading, see below) |
| `npu.sbin.1.5.5.391.zst` | 75,293 B | zstd-compressed; decompresses to **byte-identical** `npu.dev.sbin` |
| `npu.sbin.1.5.2.380.zst` | 73,534 B | Older version, similar shape |
| `npu_7.sbin.zst` | symlink | → 1.5.5.391 (used by current driver per dmesg `Request fw amdnpu/1502_00/npu.dev.sbin`) |

**The "dev" suffix is misleading**: `npu.dev.sbin` is byte-identical to
the production `npu.sbin.1.5.5.391.zst` after decompression.  Same 413
strings, same entropy profile, same SHA-256 in the metadata header.
There's no separate "dev" build; it's just an uncompressed convenience
copy of the prod firmware.

## Format: AMD PSP `$PS1`-signed binary

Header format from
[`psptool`](https://github.com/PSPReverse/psptool)'s
`HeaderFile` parser (`psptool/header_file.py:42-52`):

| Offset | Field | Our value | Meaning |
|--------|-------|-----------|---------|
| `0x00` | hash/signature prefix | `4e 81 cd c2 6c 02 f0 f6 a7 26 a8 26 7d 34 78 d9` | First 16 bytes of signature blob |
| `0x10` | magic | `$PS1` (`24 50 53 31`) | AMD PSP signed binary v1 |
| `0x14` | (body_size in some variants) | `0x000309C9` = 199,625 | Implementation-specific |
| `0x18` | encrypted flag | `0` | **NOT encrypted** |
| `0x34` | signature_type | `0` | Signature is 256 bytes (RSA-2048-class) |
| `0x38` | signature_fingerprint | `12 e2 74 db 36 9e 47 39 ad 5c b0 17 5e 48 21 ff` | Identifies signing cert |
| `0x48` | compressed flag | `0` | **NOT compressed** |
| `0x50` | size_uncompressed | `0x000309C9` = 199,625 | Matches body_size since not packed |
| `0x6C` | packed_size | `0x0003CB10` = 248,592 | Matches total file size |

Header is 256 bytes (0x100).  Body starts at file offset 0x100.
Signature is 256 bytes, located at file end.  Body covers
`[0x100, file_end - 0x100)` = 248,080 bytes of plaintext code+data.

## Section structure

Entropy mapping at 4KB granularity reveals two dense payload regions
separated by a 120KB zero gap:

```
off=    0 (  0 KB):  25% non-null  (header + metadata)
off= 4096 (  4 KB):   0%           (padding)
off=16384 ( 16 KB):  95% non-null  ###  <-- Section 1 starts
off=20480 ( 20 KB):  94%           ###
... (dense through 56KB) ...
off=61440 ( 60 KB):  18%           (tail of Section 1)
off=65536 ( 64 KB):   0%           (gap)
... (zero through ~180KB) ...
off=184320 (180 KB):  89% non-null  ###  <-- Section 2 starts
off=200704 (196 KB):  91%           ###
... (dense through 240KB) ...
off=245760 (240 KB):  66%           (tail of Section 2)
```

- **Section 1**: file offset 0x4000–0xF000 (16K–60K), ~44KB
- **Zero gap**: file offset 0xF000–0x2D000 (60K–180K), ~120KB
- **Section 2**: file offset 0x2D000–0x3CA10 (180K–244K), ~68KB

The 120KB gap is significant: it suggests the two sections load at
addresses separated by ~120KB in the NPU's address space, with the
file padding the LMA difference.  This is typical of an ELF-derived
binary stripped to raw payload (TEXT segment + RODATA segment at
distinct virtual addresses).

## Code architecture: ARM Thumb-2

Section 1 disassembles cleanly as ARM Thumb-2:

- 14 valid `push {...lr}` function prologues (0xB5xx pattern) in
  section 1 alone -- typical of a code section with ~14 functions
  totalling ~44KB.
- First prologue at file offset 0x4A18 is `b5 69` =
  `push {r0, r3, r5, r6, lr}`, followed by coherent Thumb-2
  instructions including the canonical `add r0, pc, #N` PC-relative
  addressing idiom (used for loading string/constant addresses).

Example disasm of first function found:

```
00000000 <.data>:
   0: b569       push {r0, r3, r5, r6, lr}
   2: 2258       movs r2, #88     @ 0x58
   4: 2462       movs r4, #98     @ 0x62
   6: d05f       beq.n 0xc8
   8: a03e       add  r0, pc, #248  (adr r0, 0x104)    <-- string ref
   a: 6562       str  r2, [r4, #84]
   ...
```

This is ordinary ARM Thumb-2 / Cortex-A or Cortex-M code.  Specific
core variant unknown without more analysis (vector table inspection
would tell us Cortex-M vs Cortex-A).

## Plaintext rodata: 413 strings

The body contains 413 plaintext strings totalling several KB.
Notable categories:

- **aie-rt internal API symbols**: `XAie_Read32`, `XAie_BlockWrite32`,
  `XAie_CmdWrite`, `XAie_DmaChannelReset`, `XAie_CoreReset`,
  `XAie_CoreUnreset`, `XAie_CoreGetDebugHaltStatus`, `XAie_Txn_*`,
  etc.  The FW links against the same aie-rt library we have at
  `/home/triple/npu-work/aie-rt/`.
- **Test framework strings**: `aie2_core_module_access_test`,
  `aie2_mem_tile_access_test`, `msix_interrupt_test`, `tmr_test`,
  `app_fatal_error_test`, `aie_error_async_msg_sanity` -- these
  suggest a self-test harness compiled into production FW.
- **Error message strings**: `Failed to flush cmd buffer`,
  `Cmd Write operation is not supported when auto flush is disabled`,
  `Invalid DMA channel reset value`, etc.
- **Metadata**: SHA-256 hex digest at offset 0x130, followed by
  the literal "Release 1.5.5.391" version string.

The presence of unstripped symbol names is unusual for production
firmware and is a significant aid to reverse engineering: cross-
referencing strings to their callers immediately identifies function
purpose.

## What this enables

Full decompilation is feasible.  The prerequisites are:

1. **Header parsing**: psptool's `HeaderFile` parser handles `$PS1`
   blobs.  It currently fails on this file (`Blob` parser expects a
   full ROM, minimum ~8MB) -- needs a small wrapper to use the
   `HeaderFile` class directly on the extracted blob, or just extract
   the body manually (`dd if=npu.dev.sbin of=body.bin bs=1 skip=256
   count=248080`).
2. **Architecture**: ARM Thumb-2.  Specific variant (Cortex-A55 vs
   Cortex-M vs custom) needs the vector table to confirm.
3. **Section layout**: two sections (16K-60K, 180K-244K body offsets)
   with a 120KB virtual-address gap.  Specific load addresses are
   not yet recovered -- need to look for jump tables or PC-relative
   accesses near section boundaries to infer.
4. **Tools**: Ghidra or radare2 with an ARM Cortex profile.  Specify
   the two sections at their respective load addresses; use the 413
   plaintext strings as anchors for identifying function purposes.

## What this enables for our specific goals

The most valuable functions to identify by reverse engineering, given
the recovery investigation:

- **SMU command dispatch** -- understand the POWER_OFF state-machine
  blocker.  Confirm or refute the "needs FW handshake for clean
  shutdown" hypothesis from the prior finding.
- **Mailbox dispatcher** -- map opcodes 0x18, 0x10a, 0x11, 0x108,
  0x101-0x106, 0x3, etc to handler functions.  Discover whether
  there's an opcode for "FW soft-reset" that we could use.
- **Init / shutdown sequences** -- understand what the FW does at
  startup (after PSP_START) and shutdown.  Useful for understanding
  why driver reload fails.

## Operational notes for future work

`psptool` is cloned at `/home/triple/npu-work/psptool/` (a sibling of
xdna-emu / mlir-aie / aie-rt).  Install location:
`/home/triple/npu-work/mlir-aie/ironenv/bin/psptool` (got pip-installed
into the active mlir-aie venv).  The `Blob` parser can't read this
firmware directly (size check); future scripts should instantiate
`HeaderFile` directly with the file bytes.

## Source policy reminder

This is static analysis of AMD's NPU firmware to understand hardware
behavior, used as a reading reference per the xdna-emu source policy
(see top-level CLAUDE.md "Licensing and Relationship to AMD").  No
code is copied from the firmware into the emulator.  Knowledge about
the FW's interface, mailbox protocol, and state machines is used to
inform original emulator implementations.
