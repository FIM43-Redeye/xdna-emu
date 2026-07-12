# `$PS1` header has no third NPU load base

Date: 2026-07-11  
Target: Phoenix/NPU1 firmware `1502_00/npu.dev.sbin`  
SHA-256: `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`  
Branch: `feat/m2c-mapping-boot-to-idle`

## Verdict: WIN B, by a stronger elimination than the brief expected

The signed header does **not** contain a third placement base. The apparent
`0x5c = 0x81011052` field was an offset-reading error: bytes `52 10 01 81`
begin at file offset `0x58`, while the actual dword at `0x5c` is zero. Public
parsers identify `0x58` as firmware/checksum metadata, `0x60` as the four-byte
firmware version, and `0x6c` as total packed size. The older PSPTool layout
names `0x68` `load_addr`; current coreboot reserves `0x64..0x6b`. It is zero in
this image and in all four other local NPU firmware versions.

Consequently there is no untested header-derived nonzero base. The two
defensible zero-base counterfactuals were executed:

1. strip the 256-byte header and copy the body at address zero
   (`file = VMA + 0x100`);
2. copy the whole signed file at address zero (`file = VMA`).

Neither produces a coherent boot. The body-at-zero case executes 47,869
instructions, live-loads shifted roots `0x588c` and `0x5948`, then fetches zero
at the real syscall vector `0x0ae0`. Preserving the independently
behavior-derived Segment-B preload makes no difference. The whole-file case
fails at `0x20000340` after 1,025 instructions. Neither builds `_NPU`.

The old publisher and service anchors therefore do collapse under uniform
`+0x100`, but into a boot that fails before either call graph. Under the only
mapping that executes both natural paths far enough to observe them, the
publisher remains AT-framed and the registered service remains BASE-framed,
with the genuine byte collision at identity-mapped `0x8cae`. No production
`load_m2c` correction is justified.

The remaining placement policy must be supplied by `PSP_START_COPY_FW`
implementation logic, not by an in-band destination/scatter table. That logic
may use a fixed NPU memory layout, firmware ID `0x1052`, or another
command-specific rule; the on-disk header does not say which. PSP-ROM/loader RE
is therefore pinned by elimination.

## 1. Public-format cross-check

Three public sources bound the interpretation:

- [PSPTool `HeaderFile`](https://github.com/PSPReverse/psptool/blob/edae097857e94b35d5203e644c177a3e1a3d444b/psptool/header_file.py)
  parses a 256-byte header, `size_signed` at `0x14`, metadata/checksum flags at
  `0x58`, version at `0x60`, legacy `load_addr` at `0x68`, and `rom_size` at
  `0x6c`. It places a type-0 signature in the last `0x100` bytes.
- [coreboot `struct amd_fw_header`](https://github.com/coreboot/coreboot/blob/main/util/amdfwtool/amdfwtool.h)
  agrees on signed/uncompressed/compressed sizes and version, identifies a
  firmware ID beginning at `0x58`, treats `0x64..0x6b` as reserved, and names
  `0x6c` `size_total`.
- [coreboot's PSP integration guide](https://doc.coreboot.org/soc/amd/psp_integration.html#firmware-version-of-binaries)
  independently documents the 256-byte header and version bytes at `0x60`.
  Its PSP-directory entry has source and size but no destination; the separate
  BIOS-directory format explicitly adds a destination only for entries that
  need one.

The public sources disagree on whether `0x68` in older generations deserves
the name `load_addr` or is reserved. That disagreement does not affect this
image: the value is zero. More importantly, none puts a load address at `0x58`
or `0x5c`.

## 2. Byte-level decode of `1502_00`

| Offset | Raw bytes | Public-format field | Image value / validation |
|---|---|---|---|
| `0x10` | `24 50 53 31` | magic | `$PS1` |
| `0x14` | `10 c9 03 00` | `fw_size_signed` / `size_signed` | `0x3c910` |
| `0x18` | `00 00 00 00` | encrypted | false |
| `0x30` | `01 00 00 00` | signed | true |
| `0x34` | `00 00 00 00` | signature type | type 0, `0x100`-byte signature |
| `0x48` | `00 00 00 00` | compressed | false |
| `0x50` | `10 c9 03 00` | uncompressed size | `0x3c910` |
| `0x54` | `00 00 00 00` | compressed size | zero |
| `0x58` | `52 10 01 81` | firmware/checksum metadata | LE rendering `0x81011052`; firmware ID `0x1052`; PSPTool BE flags `0x52100181` select SHA-256 |
| `0x5c` | `00 00 00 00` | reserved metadata | zero; **not** `0x81011052` |
| `0x60` | `ff 01 00 00` | version bytes | LE word `0x000001ff`; byte version `00.00.01.ff` |
| `0x64` | `00 00 00 00` | reserved | zero |
| `0x68` | `00 00 00 00` | legacy `load_addr` / current reserved | zero |
| `0x6c` | `10 cb 03 00` | `rom_size` / `size_total` | `0x3cb10`, exactly the file length |

The extents close exactly:

```text
header [0x00000,0x00100) = 0x100
body   [0x00100,0x3ca10) = 0x3c910
sig    [0x3ca10,0x3cb10) = 0x100
total                         0x3cb10
```

SHA-256 of the body is
`d319d9dab94a93d2673e97ae90f55823a1e5a21bd5811086ec8741190c85145b`,
byte-identical to header bytes `0xd0..0xef`. This validates PSPTool's
interpretation of the `0x58` metadata as selecting SHA-256; it is not merely an
address-looking coincidence.

The four other local NPU images independently reinforce the classification:

| Image | firmware ID | metadata flags | Body hash | `0x68` |
|---|---:|---:|---|---:|
| `1502_00` | `0x1052` | `0x52100181` | SHA-256 matches | `0` |
| `17f0_10` | `0x1052` | `0x52100282` | SHA-384 matches | `0` |
| `17f0_11` | `0x1052` | `0x52100282` | SHA-384 matches | `0` |
| `17f1_10` | `0x1052` | `0x52100282` | SHA-384 matches | `0` |
| `17f2_10` | `0x1052` | `0x52100282` | SHA-384 matches | `0` |

There is one `$PS1`, no nested container, no ELF header, and no region table.
The manifest at body offset zero (`file 0x100..0x1ff`) is content, not placement
metadata.

## 3. Mapping implications

`PSP_LOAD_OFFSET=0x5c` in `src/firmware/mod.rs` is an execution-derived file
delta (`0x39c - 0x340`), not a parsed header field. The numerical equality with
the mistakenly reported field offset supplied no placement evidence.

If the legacy `0x68 load_addr=0` label is taken literally and the PSP strips
the signed header, the only implied uniform low mapping is:

```text
destination = 0
source      = file[0x100..0x3ca10)
file        = VMA + 0x100
reset code  = file 0x200 -> VMA 0x100
```

Because current coreboot calls `0x68` reserved, a second conservative trial
copied the whole file at zero:

```text
destination = 0
source      = file[0..0x3cb10)
file        = VMA
reset code  = file 0x200 -> VMA 0x200
```

No third delta follows from any decoded header field.

## 4. Execution discriminator

The existing collision search first reproduced the four mixed assignments:

```text
code=BASE literal=BASE: publisher Unknown pc=0x8c32, magic=0
code=BASE literal=AT:   publisher Unknown pc=0x8c32, magic=0
code=AT   literal=BASE: publisher waiti pc=0x5645, _NPU=0x55504e5f;
                        service Unknown pc=0x8cb1
code=AT   literal=AT:   publisher waiti pc=0x5645, _NPU=0x55504e5f;
                        service Unknown pc=0x8cb1
free section variables: []
```

The extended uniform trials produced:

| Candidate | Live roots | Terminal |
|---|---|---|
| body at zero, no Segment-B preload | `n=41907 pc=0x4616 L32r [0x324c] -> 0x588c`; `n=41953 pc=0x5a5b L32r [0x32f4] -> 0x5948` | `n=47869 Unknown pc=0x0ae0 word=0`; magic `0` |
| body at zero, behavior-derived Segment-B retained | identical live roots | identical `n=47869 pc=0x0ae0`; magic `0` |
| whole file at zero, Segment-B retained | static `[0x324c]=0`, `[0x32f4]=0x04000005`; neither old root executes | `n=1025 Unknown pc=0x20000340 word=0x700020`; magic `0` |

The body-at-zero run proves the shifted values are live, not just plausible
static words. It also proves Segment B is not the cause of this candidate's
failure: with and without that preload, instruction count, roots, stop PC, and
word are identical.

The candidate fails at `0x0ae0` because uniform `+0x100` serves file `0x0be0`,
which is zero. The working mixed reconstruction serves the real syscall-vector
stub from BASE file `0x0b3c`. Thus a single body-at-zero copy cannot even supply
the architectural exception vector required by the executed boot, before the
publisher/service collision is reached.

## 5. Why PSP loader RE is now the bounded next step

The driver's contract leaves placement entirely on the other side of the PSP
mailbox:

- `psp_alloc_fw_buf` aligns host memory and copies the raw firmware bytes;
- `PSP_VALIDATE` receives only host physical address and size;
- `PSP_START(PSP_START_COPY_FW)` receives no destination or region list.

This does **not** imply the destination must be encoded inside the signed
image. A fixed command-specific destination or firmware-ID-specific placement
policy in PSP code authenticates the same bytes without putting that policy in
the envelope. The decoded header and uniform execution tests show that such
out-of-band placement behavior is required here.

The next RE target is therefore the implementation behind
`PSP_START_COPY_FW`: recover how NPU firmware ID `0x1052` is placed into the low
instruction/data windows and `0x08b00000` RAM, including whether the PSP
duplicates or relocates selected ranges. No MMU, scheduler, interrupt, struct,
or firmware state change is justified by this pass.

## Reproduction

```text
XDNA_FW_PROBE=1 cargo test --lib \
  m2c_probe_execution_guided_framing_search -- --nocapture
```

The extension is additive and `XDNA_FW_PROBE`-gated. Production `load_m2c` is
unchanged.
