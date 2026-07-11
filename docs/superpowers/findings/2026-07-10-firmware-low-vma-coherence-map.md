# Firmware low-VMA structure and coherence mapper

**Date:** 2026-07-10  
**Image:** `../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin`  
**SHA-256:** `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`  
**Tool:** `src/firmware/boot_tests/coherence_mapper.rs`

## Executive verdict

The two low-image deltas are real VMA/LMA placement classes, not two coarse
segments and not an inline `0xa4` section-header format.

For the scattered `+0x100` class, the emulator's already-proven PSP load rule
gives:

```text
file = vma + 0x100
phys = file - 0x5c
therefore phys = vma + 0xa4
```

Thus `0xa4` has a confirmed operational meaning: it is the displacement from a
scattered low section's linked VMA to the physical address where the common
`file-0x5c` PSP aperture would expose its bytes. The stripped blob does not say
why the original linker chose that displacement. It is not evidence for a
`0xa4`-byte per-section header.

No payload-resident relocation/overlay directory was found. Exhaustive scans
found no standard Xtensa-style `[start,end,load-address]` triples, no records
enumerating the 43 known ranges, no shared start marker, and no usable alignment
or stride rule. The open-source host driver also passes the PSP only an opaque
blob address and size; it supplies no scatter map.

There is a fundamental ambiguity in blind byte-only recovery: bytes coherent at
`file=vma+0x100` are the same bytes at the shifted alias
`vma'=vma+0xa4`, `file=vma'+0x5c`. PC-relative control flow and L32R reach the
same physical bytes in both descriptions. Canonical VMA roots from absolute
pointers or coherent execution are required to break that symmetry. This is
why the new mapper can classify supplied canonical boundaries perfectly, while
its image-only boundary discovery remains deliberately reported as uncertain.
The root-assisted calibration pins the two established `+0x5c` execution roots
from the task (`RESET_ENTRY` and PC `0x4525`) rather than pretending the bytes
can identify them unaided.

## Concrete evidence for `0xa4`

At file `0x2bc` the payload contains:

```text
file 0x2bc: 9c 02 00 00 40 03 00 00
             0x0000029c  0x00000340
```

The words differ by exactly `0xa4`, and both name file `0x39c` under the two
placement rules:

```text
0x29c + 0x100 = 0x39c
0x340 + 0x05c = 0x39c
```

The reset code uses both values. Raw decode around file `0x348` loads `0x29c`
into `a14` and `0x340` into `a3`; the prologue ORs the code-region base into
`a3` and the executed `jx a3` at file `0x399` reaches virtual `0x20000340`.
There is also a later `jx a14` at file `0x4bd`; static reachability of that
alternate path was not established.

The same eight bytes occur at file `0x2bc` in Phoenix releases 1.5.2.380,
1.5.5.391, and 1.5.6.399. That rules out a 1.5.5 signing/header accident.

The window vectors are the cleanest independent example of separately placed
Xtensa output-section content:

```text
VMA 0x800, file 0x900 (+0x100):
00 c5 49   s32e a0,a5,-16
10 d5 49   s32e a1,a5,-12
20 e5 49   s32e a2,a5,-8
30 f5 49   s32e a3,a5,-4
00 34 00   rfwo

VMA 0x800, file 0x85c (+0x5c): all zero
```

This is consistent with linker output sections whose VMA and load address
differ, such as the generic Xtensa linker `AT(...)` idiom. Original ELF program
headers are stripped, so the exact original `p_offset-p_vaddr` records cannot
be recovered.

## Searches that ruled out a table or framing format

The following scans covered the entire 248,592-byte file, not just the signed
header:

- Canonical aligned triples `[start,end,LMA]` with `0 <= start < end <= 0x10000`
  and `LMA-start` equal to `0x5c`, `0xa4`, or `0x100`: **zero hits for all
  three deltas**.
- Exact adjacent `[lo,hi]`, `[lo,length]`, `[lo,lo+0x100]`, or
  `[lo+0x100,lo]` records for all 43 oracle ranges: **zero hits**.
- ELF and section-name strings (`\x7fELF`, `.text`, `.rodata`, `.symtab`,
  `.shstrtab`, `.xt.prop`): **zero hits**.
- `0xa4` as an ELF32 header/program-header total does not fit:
  `(0xa4 - sizeof(Elf32_Ehdr=0x34)) % sizeof(Elf32_Phdr=0x20) = 0x10`.

The known boundaries also reject a marker/alignment rule:

```text
43 ranges, 8,162 bytes total
start mod 4:  42 at 0, 1 at 3
start mod 16: 14 at 0, 14 at 0xc, 8 at 8, 6 at 4, 1 at 7
10 ends are not word-aligned
```

Code boundaries use ordinary ABI instructions shared by both placement
classes. For example, base `+0x5c` code at VMA `0x4514` / file `0x4570` starts
`36 81 00` (`entry ... 0x40`), while overlay code at VMA `0x4a0c` / file
`0x4b0c` starts `36 41 00` (`entry ... 0x20`). `entry` is a boundary candidate,
not a delta marker.

At `0x4a0c`, the correct view executes a complete 16-instruction function:

```text
0x4a0c entry
0x4a0f/0x4a12 l32r
...
0x4a33 s32i.n
0x4a35 retw.n
```

The `+0x5c` bytes start with an unknown word but then densely decode as a long
sequence of plausible `l32i`/`s32i.n` operations. This directly falsifies
decode density as a classifier.

The host-side load path contains no hidden map:

- `../xdna-driver/src/driver/amdxdna/aie_psp.c:27-51` aligns and copies the
  opaque firmware bytes.
- `../xdna-driver/src/driver/amdxdna/aie2_psp.c:87-101` sends only physical
  address and aligned size to `PSP_VALIDATE`, then issues
  `PSP_START_COPY_FW` with no section arguments.
- `../xdna-driver/src/driver/amdxdna/aie2_pci.c:690-694` provides only
  `fw->size` and `fw->data`.

If the real PSP owns exact scatter boundaries, they may be PSP-private fixed
policy or build metadata not found in the representations scanned here or in
the open-source driver. An unrecognized proprietary encoding remains possible.

## Low-file picture (`0x200..0x10000`)

This is a page-scale picture of raw nonzero bytes and the file bytes named by
the 43 known `+0x100` VMA intervals. “Other nonzero” is intentionally not
called base code: it mixes `+0x5c` code, data, literal pools, and unclassified
candidate sections.

| file range | nonzero | known `+0x100` bytes | other nonzero | zero |
|---|---:|---:|---:|---:|
| `00200-01200` | 760 | 384 | 376 | 3336 |
| `01200-02200` | 0 | 0 | 0 | 4096 |
| `02200-03200` | 1517 | 1509 | 8 | 2579 |
| `03200-04200` | 3479 | 216 | 3263 | 617 |
| `04200-05200` | 3893 | 401 | 3492 | 203 |
| `05200-06200` | 3856 | 1991 | 1865 | 240 |
| `06200-07200` | 3856 | 0 | 3856 | 240 |
| `07200-08200` | 3804 | 158 | 3646 | 292 |
| `08200-09200` | 3819 | 659 | 3160 | 277 |
| `09200-0a200` | 3910 | 546 | 3364 | 186 |
| `0a200-0b200` | 3963 | 0 | 3963 | 133 |
| `0b200-0c200` | 4050 | 0 | 4050 | 46 |
| `0c200-0d200` | 3779 | 328 | 3451 | 317 |
| `0d200-0e200` | 3758 | 1658 | 2100 | 338 |
| `0e200-0f200` | 1710 | 312 | 1398 | 2386 |
| `0f200-10000` | 675 | 0 | 675 | 2909 |

Maximal zero runs of at least `0x100` bytes are:

```text
file 0x004bf..0x00901  length 0x442
file 0x00bb5..0x02328  length 0x1773
file 0x02329..0x0242a  length 0x101
file 0x02c50..0x032a0  length 0x650
file 0x0e1ea..0x0e2fc  length 0x112
file 0x0f411..0x0fae0  length 0x6cf
```

These holes coexist with fine-grained code/data interleaving. They are not
coarse segment boundaries. Segment B still begins independently at file
`0x2d100` and maps to physical `0x08b00000`, as established in
`build/experiments/firmware-re/image-structure-verdict.md`.

## Mapper design

Run all mapper reports with:

```bash
XDNA_FW_PROBE=1 cargo test --lib m2c_probe_coherence_mapper -- --nocapture
XDNA_FW_PROBE=1 cargo test --lib m2c_probe_blind_map_boot_frontier -- --nocapture
XDNA_FW_PROBE=1 cargo test --lib m2c_probe_calibrated_map_boots_alive -- --nocapture
XDNA_FW_PROBE=1 cargo test --lib m2c_probe_overlay_store_conflicts -- --nocapture
```

The tool uses the repository decoder directly (`decode::decode`). For each
canonical entry candidate and each delta it:

1. Requires a real `entry` prologue.
2. Recursively walks conditional branches and local jumps, continuing across
   calls and stopping only on returns or tail transfers.
3. Rejects unknown instructions, runaway control flow, and paths without a
   return/tail terminator.
4. Collects direct-call targets and L32R targets.
5. Scores literal words by known image/address domains and mask shapes, with
   junk values penalized.
6. Recognizes the window-vector block using semantic `s32e/l32e` plus
   `rfwo/rfwu` signatures, not decode density.
7. Uses values actually loaded from coherent L32R pools as canonical indirect
   function seeds. Raw low-looking words are not seeds.
8. Treats the established coherent base PCs `RESET_ENTRY` and `0x4525` as
   negative/root anchors. This is the minimum external execution evidence that
   prevents their `0xa4`-shifted aliases from being relabeled `+0x100`.

The module has a synthetic red/green unit test pinning selection of a complete
`entry -> retw.n` function under `+0x100` and rejection under `+0x5c`.

### Blind image-only result

The blind run emitted 59 ranges / 8,302 bytes. Against the deliberately
closed-world 43-range oracle:

```text
true-positive bytes = 2,165
predicted bytes     = 8,302
oracle bytes        = 8,162
precision           = 0.2608
recall              = 0.2653
```

That run does **not** boot alive:

```text
ranges=59
instrs=47,474
unknown=(pc 0x2000e035, word 0x41f0)
magic@local_data[0x14820]=0
```

The low closed-world precision does not prove all non-oracle candidates are
wrong because the oracle is explicitly only the boot-executed subset. It does
prove blind output is not safe to install. Examples outside the oracle include
`0x4628-0x4806`, `0x7ea0-0x81e6`, `0x847c-0x851d`, `0x9e5c-0x9f40`, and
`0xa660-0xa6b3`. Each is coherent under local metrics, but none is promoted to
a new confirmed section: the shifted-alias ambiguity remains unresolved and
the blind boot rejects the set as a whole.

### Calibration-oracle result

The calibration candidate set contains the 43 known `+0x100` intervals plus two
known `+0x5c` negative controls (`RESET_ENTRY`'s range and the `0x4525` base
function range). With the two established base roots pinned, the coherence
engine selects all 43 positives and rejects both negatives:

```text
range precision = 43/43 = 1.0000
range recall    = 43/43 = 1.0000
negative controls rejected = 2/2
byte precision  = 8,162/8,162 = 1.0000
byte recall     = 8,162/8,162 = 1.0000
```

This is root-assisted boundary classification, not blind boundary discovery.
The distinction is material and is why the tool prints both results.

A fresh loader constructed without calling `load_m2c`'s production overlay
installation, then given only the calibrated emitted intervals, reaches:

```text
idle=true
wait=Waiti
unknown=None
instructions=52,391
pc=0x5645
local_data[0x14820]=0x55504e5f  ("_NPU")
```

This validates the emitted calibration map end to end without editing or
reusing the production overlay installation.

## Store-conflict correction

An interval-wide “zero stores” veto is invalid for the current coarse oracle
bounds. A full successful boot recorded 45 store instructions whose effective
low VMA lies inside an oracle interval. The largest group is exactly the known
base `+0x5c` loop at PC `0x20004525` writing `0x2b70..0x2bf4`; four more stores
target `0x58c0..0x58cc`.

These stores go to the Harvard local-data backing and do not mutate fetched ROM,
so boot still reaches `_NPU` and `waiti 0x5645`. The result means a mapper may
use “no store targets this exact instruction/literal byte” as supporting
evidence, but may not reject a hand-bounded overlay interval merely because it
contains writable VMA gaps. This also confirms that the oracle ranges are
fetch/literal service intervals, not clean linker-section extents.

## What is resolved and what is not

Resolved:

- `0xa4` is the scattered class's VMA-to-physical displacement after the
  `0x5c` PSP load rule.
- It is not an inline header size, signing artifact, repeating stride, or a
  standard retained ELF table.
- No discoverable payload or host-driver table enumerates the overlay ranges.
- Coherence classifies canonical boundaries; decode density does not.
- The calibrated map boots alive from a fresh non-production-overlay load.

Unresolved:

- The exact original linker/PSP mechanism that chose and delivered every
  scattered low output section.
- A unique blind partition of every low byte into canonical `+0x5c`, canonical
  `+0x100`, data, or zero. The bytes alone admit shifted coherent aliases.
- Any new `+0x100` ranges beyond the boot oracle. The tool emits candidates,
  but none has an independent canonical-VMA root and successful-map validation
  strong enough to call confirmed.

The proper next discriminator is new coherent execution or an external symbol/
load map, not more decode-density tuning.
