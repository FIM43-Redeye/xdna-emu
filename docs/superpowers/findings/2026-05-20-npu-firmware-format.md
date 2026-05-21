# NPU firmware format characterization (Phoenix, FW 1.5.5.391)

**Status:** Format and architecture characterized (2026-05-20). Body
confirmed plaintext; processor identified as **Xtensa LX7 little-endian**;
load base recovered; 590 functions disassembled with a connected call
graph. Functional reverse engineering (mailbox dispatcher, power-down
handshake) is the follow-on task.

> **Correction (2026-05-20):** an earlier revision of this document
> claimed the firmware was ARM Thumb-2. That was wrong. It rested on a
> single 16-bit pattern match (`0xB5xx`, read as a Thumb `push {…,lr}`
> prologue) at body offset 0x4918 -- which, once actually disassembled,
> is data, not code (a region of ASCII hex digits from the SHA-256
> block). The architecture below is verified: see "Evidence" and
> "Verification".

**Related:**
[`2026-05-20-amdxdna-tdr-recovery-incomplete-on-phoenix.md`](2026-05-20-amdxdna-tdr-recovery-incomplete-on-phoenix.md)
(the investigation that motivated looking at FW internals).

## File layout

The firmware blob lives at `/lib/firmware/amdnpu/1502_00/`:

| File | Size | Notes |
|------|------|-------|
| `npu.dev.sbin` | 248,592 B | Uncompressed copy ("dev" is misleading, see below) |
| `npu.sbin.1.5.5.391.zst` | 75,293 B | zstd-compressed; decompresses to **byte-identical** `npu.dev.sbin` |
| `npu.sbin.1.5.2.380.zst` | 73,534 B | Older version, similar shape |
| `npu_7.sbin.zst` | symlink | -> 1.5.5.391 (used by current driver per dmesg `Request fw amdnpu/1502_00/npu.dev.sbin`) |

The `npu.dev.sbin` "dev" suffix is misleading: it is byte-identical to
the production `npu.sbin.1.5.5.391.zst` after decompression. Same
strings, same entropy profile. There is no separate "dev" build; it is
just an uncompressed convenience copy.

The driver picks the firmware path in
`xdna-driver/src/driver/amdxdna/npu1_regs.c:84`:
`.fw_path = "amdnpu/1502_00/npu.dev.sbin"`.

## Container: AMD PSP `$PS1`-signed binary

| File offset | Field | Value | Meaning |
|-------------|-------|-------|---------|
| `0x000` | signature prefix | 16 bytes | First bytes of the signature blob |
| `0x010` | magic | `$PS1` | AMD PSP signed binary v1 |
| `0x014` | body size | `0x0003C910` = 248,080 | Plaintext payload size |
| `0x018` | encrypted flag | `0` | **NOT encrypted** |
| `0x030` | (constant) | `1` | -- |
| `0x038` | signing fingerprint | 16 bytes | Identifies the signing cert |
| `0x048` | compressed flag | `0` | **NOT compressed** |
| `0x050` | uncompressed size | `0x0003C910` = 248,080 | Equals body size -> not packed |
| `0x06C` | packed size | `0x0003CB10` = 248,592 | Equals total file size |
| `0x0D0` | digest | 32 bytes | Image hash |

Header is 256 bytes (`0x100`). Body is `[0x100, 0x3CA10)` = 248,080
bytes of plaintext code+data. Signature is the trailing 256 bytes.

(The earlier psptool-derived field table was a generic `$PS1` guess and
mislabelled several offsets; the table above is read directly from the
file. psptool's `Blob` parser cannot read this file at all -- it expects
a multi-MB ROM -- so the header was parsed by hand.)

## Not encrypted, not compressed

A sliding-window (2 KB) Shannon-entropy map of the body never exceeds
~7.1 bits/byte. Whole-image encryption or compression would read ~7.9+.
The dense code regions sit at ~7.0 -- high for fixed-width RISC, but
normal for the variable-length Xtensa encoding (see below). The body is
plaintext.

## Processor architecture: Xtensa LX7 (little-endian)

The firmware runs on a **Cadence/Tensilica Xtensa LX7** core -- the NPU's
embedded management microcontroller.

### Evidence (from the open-source driver -- authoritative)

- `xdna-driver/src/driver/amdxdna/aie4_debugfs.c:392` --
  `{"echo msg between host and lx7 firmware", test_msg_echo}`. The
  firmware that exchanges mailbox messages with the host is explicitly
  the **"lx7 firmware"**.
- `xdna-driver/src/include/uapi/drm_local/amdxdna_accel.h:691-692` --
  fatal-error reporting carries `fatal_error_exception_type`
  ("LX7 exception type") and `fatal_error_exception_pc`
  ("LX7 program counter"). The driver models the NPU controller's faults
  in LX7 terms.
- `xdna-driver/.../amdxdna_ctx.h:99-100` -- same LX7 exception fields.

The `uc_index /* microblaze controller index */` comments in
`amdxdna_ctx.h:89` / `ve2_hwctx.c` are a **red herring** for Phoenix:
they describe the Versal/VE2 path, not the Ryzen AI NPU. (aie-rt's build
system also has a MicroBlaze cross-compile target, `Makefile.rsc`, but
that is a generic aie-rt build option, not what this firmware uses.)

### Verification

Imported into Ghidra 12.1 as `Xtensa:LE:32:default` (Ghidra ships an
Xtensa processor module). The result is unambiguous:

| | as ARM Thumb (wrong) | as Xtensa LE (correct) |
|---|---|---|
| Functions recovered | 40 | **590** |
| With a real caller/callee | 1 | **~500** |
| Disassembly | incoherent | coherent; branch targets resolve |

Sample (function at `0x08ad5630`):

```
08ad5630  entry a1,0x50            ; Xtensa windowed-ABI prologue
08ad5633  rsil  a2,0x2             ; read+set interrupt level (privileged)
08ad5639  l32r  a2,0x08ad554c      ; PC-relative literal load
08ad5641  memw                    ; memory barrier
08ad5644  l32r  a5,0x08ad5550      ; -> 0xDEADBEEF
08ad5656  beq   a7,a5,0x08ad566e   ; branch target inside the function
```

`entry`, `rsil`, `memw`, `l32r`, `add.n`/`l32i.n` (16-bit "narrow"
code-density instructions), and `call8` windowed calls are all
characteristic Xtensa. The mixed 16/24-bit instruction lengths are why
the body's entropy reads ~7.0 and why every fixed-width disassembler
(ARM, Thumb, MicroBlaze, AArch64, RISC-V, PPC, SH) produced garbage.

## Load base address: 0x08ad3000

Xtensa code reaches absolute targets (string literals, function
pointers) through PC-relative `l32r` literal pools, where each literal
holds `load_base + offset`. Correlating the firmware's aligned 32-bit
words against the 590 Ghidra function offsets *and* the string offsets
(`tools/fw-find-base.py`) yields a single base that satisfies **both**
anchor types: `0x08ad3000`.

Cross-check: the literal at body `0x2d08c` holds `0x08b006c4`;
`0x08b006c4 - 0x08ad3000 = 0x2d6c4`, exactly the body offset of the
string `"XAie_Read32"`. With the image based at `0x08ad3000`, 125+
string xrefs and the function-pointer tables resolve cleanly.

Function address range (rebased): `0x08ad5630` - `0x08b0f624`.

## Body structure

| Body offset | Content |
|-------------|---------|
| `0x00000`-`0x00100` | Metadata: version fields (build 391 = `0x187`), SHA-256 hex, `"Release 1.5.5.391"` |
| `0x00100`-`0x05000` | Low-density data + pointer tables (function-pointer arrays around `0x03000`-`0x03c00`) |
| `0x05630`-`0x0f000` | Code region 1 (Xtensa) |
| `0x0e800`-`0x10000` | Tail strings -- FW self-test names (`aie_ipu_mgmt_test`, etc.) |
| `0x10000`-`0x2cf00` | Zero gap (~116 KB; LMA padding between the two loaded sections) |
| `0x2cf00`-`0x32000` | String tables + `l32r` literal-pointer pools (aie-rt symbol/error strings) |
| `0x32000`-`0x3c624` | Code region 2 (Xtensa) |

The bulk of the readable strings (181 `XAie_*` symbols) belong to the
**vendored aie-rt library** linked into the firmware -- we already have
its source at `../aie-rt/`, so that portion needs no RE. The
firmware-specific logic (mailbox dispatch, management loop, power
handling) is the code that *calls* aie-rt.

## Toolchain pipeline (CLI-driven, reproducible)

Ghidra's GUI is not scriptable for this workflow, so analysis runs
headless:

- `tools/ghidra-npu-fw.sh` -- one-shot driver: extracts the body, runs
  `analyzeHeadless` with the Xtensa language + correct load base, dumps
  text artifacts.
- `tools/ghidra-scripts/SetImageBase.java` -- preScript; rebases the
  image to `0x08ad3000` before analysis so `l32r` literals resolve.
- `tools/ghidra-scripts/DumpNpuFw.java` -- postScript; writes
  `functions.tsv`, `strings.tsv`, `disasm.txt`.
- `tools/fw-find-base.py` -- recovers the load base from literal pools.
- `tools/fw-arch-probe.py` -- multi-architecture decode scorer (kept for
  reference / future firmware blobs).

Outputs land in `ghidra-projects/npu-fw/analysis-xtensa/`.

## What this enables for our specific goals

The suspend / `waitmode` path has since been mapped on top of this
characterization -- see
[`2026-05-20-npu-fw-suspend-waitmode-path.md`](2026-05-20-npu-fw-suspend-waitmode-path.md).

With a coherent disassembly, the recovery-investigation targets are now
reachable:

- **Mailbox dispatcher** -- map host message opcodes to handler
  functions; look for a "prepare for power-down / suspend" opcode.
- **Power-down handshake** -- understand what the LX7 firmware must do
  before the SMU can complete `POWER_OFF`. The prior finding's
  hypothesis (POWER_OFF needs FW cooperation that a hung FW cannot
  provide) is now testable against the actual shutdown path.
- **Init / fatal-error path** -- the FW has an LX7 exception handler
  that feeds `fatal_error_exception_pc` back to the driver; mapping it
  explains what the driver sees when the NPU wedges.

## Source policy reminder

This is static analysis of AMD's NPU firmware to understand hardware
behavior, used as a reading reference per the xdna-emu source policy
(top-level CLAUDE.md, "Licensing and Relationship to AMD"). No firmware
code is copied into the emulator. Knowledge of the FW's mailbox
protocol and state machines informs original emulator implementations.
