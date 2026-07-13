# Upstream cone framing: bounded static negative, dynamic mechanism unresolved

Date: 2026-07-12

## Verdict

No candidate in the requested upstream search reaches the exact device-SRAM
alive publication.  The corrected search evaluated 74,128 candidates across
seven upstream regions.  Every candidate independently selects:

- `delta_lo` and `delta_hi` from `{0, +0x5c, +0x100, +0x244}`;
- every byte split in the selected region, canonicalized to remove equivalent
  uniform maps;
- the BASE or AT view for the whole collision region `[0x8c98,0x8d52)`; and
- the BASE or AT view for the live service literal at `[0x354c,0x3550)`.

The acceptance oracle samples the backing device SRAM independently of decoded
CPU stores.  All 74,128 rows have `service_pass=false` and `alive_store=false`.
Of them, 13,216 are inconclusive because execution reaches an unknown opcode,
or the instruction budget (12,166 `unknown`, 1,050 `budget`).  Therefore this
is a complete enumeration with zero observed solutions before stop, not an
exhaustion proof for the matrix and not a proof that every possible static
reconstruction is impossible.  In particular, the search changes one upstream
code region at a time and does not jointly remap every literal pool associated
with each upstream function.

The search does dissolve the original `0x8cae` shared-byte contradiction for
valid-boundary candidates.  Those candidates carry the service through the
BASE instructions at `0x8cae..0x8cba`, then expose another time-dependent view
at `0x26d4`: the earlier execution uses the AT stream through that VMA, while a
later `Call8 0x26d4` requires the BASE `Entry`.  The test-only discriminator
that switches both views still reaches only the service sink at `0x7fec` with
`a7=6`; it does not publish.

The dynamic audit falsifies a plain reload of either exact low-code view from
the Segment-B RAM image, but it does **not** prove that every DMA/context
overlay mechanism is absent.  No live MMIO write visibly supplies a
Segment-B/code source together with a low-code destination.  However, the
model has immutable instruction backing, the audit cannot observe an
unmodelled IRAM-side transfer, and a source in Segment A or a computed/encoded
descriptor remains possible.  Hardware instruction-fetch banking is the
other live mechanism class.  No faithful production fix is justified until a
selector or transfer engine is identified.

## Search definition and corrections

The generalized candidate is:

```text
(region, delta_lo, delta_hi, split, literal_delta, local_delta)
```

The two acceptance predicates remain:

1. publisher: `local[0x14820] == 0x55504e5f` and execution reaches `0x5645`;
2. service: after the live `0x283b -> 0x8770` callback edge, all sixteen words
   at `0x030bb000..0x030bb03f` equal the expected descriptor and
   `0x030bf000 == 0x030bb000`.

The rerun includes four correctness fixes to the exploratory harness:

- boundary sets are re-derived for every candidate;
- any executed instruction crossing the candidate region start, internal
  split, region end, or collision-view seam is rejected as
  `split-instruction`;
- `0x2000xxxx` high aliases are not treated as if low-VMA overlays changed
  their bytes; and
- service acceptance samples backing SRAM every 64 instructions and at stop,
  so a non-CPU transfer could satisfy the predicate.

The TSVs are under `build/experiments/firmware-re/` with names of the form
`delta-split-search-<region>-d0-5c-100-244-l5c-100-m100000.tsv`.

## Results

| Region | Range | Production delta | Candidates | Publisher pass | Publisher + service entry | Inconclusive | Joint stop `0x26d4` | Joint stop `0x8cb1` | Exact publish |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `p55f8` | `[0x55f8,0x581c)` | `+0x100` | 26,272 | 2,850 | 2,564 | 2,800 | 0 | 2,564 | 0 |
| `p50d4` | `[0x501c,0x518f)` | `+0x100` | 17,776 | 232 | 220 | 5,244 | 46 | 160 | 0 |
| `p8f44` | `[0x8f44,0x9065)` | `+0x100` | 13,840 | 248 | 248 | 1,286 | 126 | 122 | 0 |
| `s8770` | `[0x8770,0x87eb)` | `+0x5c` | 5,872 | 2,936 | 2,936 | 2,658 | 0 | 568 | 0 |
| `sc530` | `[0xc530,0xc583)` | `+0x5c` | 3,952 | 1,976 | 1,976 | 668 | 0 | 120 | 0 |
| `s7fc4` | `[0x7fc4,0x801f)` | `+0x5c` | 4,336 | 2,168 | 2,168 | 438 | 0 | 356 | 0 |
| `s8c6c` | `[0x8c6c,0x8c98)` | `+0x5c` | 2,080 | 1,040 | 1,040 | 122 | 18 | 62 | 0 |
| **Total** | | | **74,128** | **11,450** | **11,152** | **13,216** | **190** | **3,952** | **0** |

`Exact publish` is zero for every region.  The backing-state oracle and the
diagnostic store tracker also agree that no row writes `FW_ALIVE_OFF`.

## A valid-boundary candidate that dissolves the first collision

The representative row is:

```text
region=p8f44 delta_lo=+0x100 delta_hi=0 split=0x8f47
literal_delta=+0x5c local_delta=+0x5c
publisher_pass=true service_entered=true service_pass=false
stop=Unknown pc=0x26d4 word=0x39ffa0
```

`0x8f44..0x8f46` is the complete AT `Entry`; `0x8f47` is the next instruction
boundary.  This candidate builds `_NPU`, reaches `0x5645`, and does not execute
the publisher's old approach through the collision region
(`publisher_boundaries={}`).  The service later derives this BASE boundary
stream from the candidate bytes:

```text
0x8c8b [37 69 1f] Bbci bit=3,target=0x8cae  files 0x8ce7..0x8ce9
0x8cae [82 c8 60] Addi a8,a8,0x60          files 0x8d0a..0x8d0c
0x8cb1 [42 d4 10] Addmi a4,a4,0x1000       files 0x8d0d..0x8d0f
0x8cb4 [52 d5 10] Addmi a5,a5,0x1000       files 0x8d10..0x8d12
0x8cb7 [30 e6 13] Wsr                      files 0x8d13..0x8d15
0x8cba [1d f0]    RetwN                    files 0x8d16..0x8d17
```

Thus the asserted publisher `Bgeu @0x8cac` and service `Addi @0x8cae` are not
universal boundaries.  Moving the publisher framing at `p8f44` removes that
specific contradiction, but the candidate remains inconclusive at `0x26d4`
and performs neither a descriptor-target nor `FW_ALIVE_OFF` write.

Both `literal_delta=+0x5c` and `+0x100` give the same outcome class for this
representative framing.  Across `p8f44`, the joint rows split as follows:

```text
local_delta=+0x5c, literal_delta=+0x5c: 63 stop at 0x26d4
local_delta=+0x5c, literal_delta=+0x100: 63 stop at 0x26d4
local_delta=+0x100, literal_delta=+0x5c: 61 stop at 0x8cb1
local_delta=+0x100, literal_delta=+0x100: 61 stop at 0x8cb1
```

The live service literal is therefore not the missing discriminator.

## The second time-dependent view

The runtime discriminator is observation-only and uses test-only view switches;
it is not a proposed fix.  Its fresh trace is:

```text
n=53640 pc=0x8c6c select BASE service view
n=53783 pc=0x7fe7 Call8 target=0x26d4
n=53784 pc=0x26d4 select BASE context view
n=53784 pc=0x26d4 Entry a1,0x50
n=53813 pc=0x2734 Call8 target=0xc530
n=53831 pc=0xc55c Callx8 target=0x08b0e710
n=53843 pc=0x08b0e71d Loopnez
n=53844..53861 pc=0x08b0e720/0x08b0e723 Dhwbi/Addi
n=53862 pc=0x08b0e726 Dsync
n=53863 pc=0x08b0e729 RetwN
n=53871 pc=0xc56e Call8 target=0x7fc4
n=53873 pc=0x7fc7 Bgeui a7,6,target=0x7fec
n=53874 pc=0x7fec service sink, a7=6
```

Earlier in the same boot, the AT context path crosses `0x26d4` as part of the
stream beginning at `0x2630`.  Later, BASE begins a new `Entry` at that same
VMA.  A static view of `0x26d4` cannot encode both executions.  The second
switch advances the service but still does not satisfy the publish predicate.

## DMA/block-copy/context-overlay audit

Firmware SHA-256:

```text
d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e
```

Segment B is file `[0x2d100,0x3cb10)`, loaded at
`[0x08b00000,0x08b0fa10)`.  A direct scan gives:

```text
service Addi bytes   82 c8 60: whole image 1, Segment B 0
publisher bytes      87 ba 02: whole image 1, Segment B 0
service-root 16 bytes:          whole image 1, Segment B 0
publisher-root 16 bytes:        whole image 1, Segment B 0
bytewise u32 values in [0x8c00,0x8e00): 0
aligned Segment-B pointers: 379
aligned 0x8000 page words: 8
static RAM-pointer/page pairs within 128 bytes: 0
```

This rules out a literal copy of either exact view from Segment B and finds no
plain static `{Segment-B source, low-page destination}` descriptor there.
The counts and exact bytes are reproduced by
`m2c_probe_reload_source_scan` and retained in
`build/experiments/firmware-re/upstream-reload-source-scan.log`.

The continuous natural-boot audit reaches the production wall with the
publisher intact and reports:

```text
n=53660 stop=Unknown pc=0x8cb1 word=0x61a800
MMIO accesses=6002, MMIO writes=360
low/source-looking MMIO writes=11, CPU stores=144
```

The 11 MMIO writes contain no visible pair that names both a code source and a
low-code destination.  The potentially misleading low-looking writes are:

```text
pc=0x200089ad addr=0x272003b4 value=0x00008000
pc=0x200089d4 addr=0x27200304 value=0x0000e000
pc=0x20008964 addr=0x27200318 value=0x0000f000
pc=0x200089ad addr=0x272003b8 value=0x00008000
pc=0x200089d4 addr=0x27200308 value=0x00008000
pc=0x200089ad addr=0x272003bc value=0x00008000
pc=0x200089d4 addr=0x2720030c value=0x0000ffff
pc=0x200089ad addr=0x272003bc value=0x20000000
```

The values are written into adjacent controller registers by the known bit-mask
walk at `0x893c..0x89d4`; they are not accompanied by a source, length, or kick
write.  The only Segment-B-looking MMIO value is
`pc=0x2000d4f7, addr=0x27200190, value=0x08b041bc`, the mailbox/high pointer
already used repeatedly as data in local structures.

Two writes at `pc=0x8db0` target `0x27200908` with values `0xc400` then
`0xe400`.  A fresh decode of `0x8d88..0x8db2` shows `L32r` of literal
`0x27200904`, then `Srli/Extui/Addx4/Slli/Ssl/Sll/And/Xor/Or`, one `S32iN` at
`0x8db0`, and `RetwN` at `0x8db2`.  That is a packed field update, not a
source/destination/length descriptor.

The nearby-word heuristic is not an absence proof after execution: final local
memory contains 7 and Segment-B RAM contains 65 source/destination-looking
pairs within 128 bytes, and the CPU makes 144 source/low-looking stores.  The
samples include repeated mailbox-pointer propagation and values structurally
compatible with callback/function tables.  Regardless of classification, their
existence means address-shape scanning alone cannot close DMA.

The actually traced context path provides one narrower negative.  From BASE
`0x26d4`, execution reaches `Call8 0xc530`; `0xc530` builds a local IPC record,
then `Callx8 0x08b0e710`.  The Segment-B helper executes `Loopnez`, repeated
`Dhwbi` plus pointer `Addi`, `Dsync`, and `RetwN`, then returns through
`0xc56e -> 0x7fc4`.  That observed helper is a cache maintenance walk, not a
copy into low code.

### Earned absence and what remains

The evidence earns only these negative statements:

- neither exact low-code view is present as a Segment-B byte source;
- no plain static Segment-B-pointer/low-page descriptor pair was found;
- the observed `0x26d4 -> 0xc530 -> 0x08b0e710` transition performs cache
  maintenance, not a CPU copy; and
- the natural-boot MMIO trace exposes no obvious source/destination programming
  pair for the low code window.

It does **not** exclude a DMA/overlay source in Segment A, computed or encoded
descriptor fields, or an instruction-side reload that the present bus model
cannot observe.  It also does not select between such a reload and hardware
fetch banking/mask-ROM views.  Those are the remaining mechanism classes.

## Consequence for the emulator

There is no production `load_m2c` mapping diff to apply, and no faithful dynamic
model can yet be written.  A per-path byte swap would merely encode the
discriminator and is forbidden.  The next decisive evidence must identify
either a real transfer/overlay programming sequence or the instruction-fetch
bank selector.

## Verification

```text
cargo test --lib split_candidates_are_canonical_and_piecewise -- --nocapture
  1 passed; 0 failed

XDNA_FW_PROBE=1 XDNA_FW_MAX=100000 \
  cargo test --lib m2c_probe_runtime_view_discriminator -- --nocapture
  1 passed; reached 0x7fec with a7=6 after both test-only switches

XDNA_FW_PROBE=1 cargo test --lib \
  m2c_probe_reload_programming_audit -- --nocapture
  1 passed; reached the natural 0x8cb1 wall with _NPU intact

XDNA_FW_PROBE=1 cargo test --lib \
  m2c_probe_reload_source_scan -- --nocapture
  1 passed; exact view counts 1/0, low words 0, pointer/page pairs 0

XDNA_FW_PROBE=1 XDNA_FW_MAX=100000 \
  cargo test --lib m2c_probe_alive_device_sram_struct -- --nocapture
  expected negative: n=53659, Unknown 0x8cb1 word=0x61a800,
  "firmware emitted no device-SRAM descriptor stores"

cargo test --lib
  4090 passed; 0 failed; 30 ignored
```

No production mapping was changed and no commit was made.
