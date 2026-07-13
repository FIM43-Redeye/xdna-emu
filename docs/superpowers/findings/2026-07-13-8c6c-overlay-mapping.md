# `0x8c6c` is a base function cut by the `0x8c98` overlay

Date: 2026-07-13  
Repository base: `59d7ab32`  
Firmware: Phoenix `1502_00/npu.dev.sbin`  
SHA-256: `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`

## Verdict

For the requested half-open VMA range `[0x8c6c, 0x8c98)`, the correct source is
the default mapping:

```text
file = VMA + 0x5c
0x8c6c -> file 0x8cc8
```

It is not a missing `+0x100` overlay. The base bytes form the entry and first
half of one coherent function, whose complete body is `[0x8c6c, 0x8cbc)` and
whose `RetwN` is at `0x8cba`. The `+0x100` bytes at VMA `0x8c6c` are the tail of
one unrelated function, padding, and then another function rooted at `0x8c78`.
They cannot be the target of the executed windowed `Call8 0x8c6c`.

The `0x8cb1` dead-end has a different, fully local explanation: the already
installed `+0x100` publisher overlay begins at VMA `0x8c98` and cuts across the
middle of the coherent base function. Thus the natural run executes base bytes
through `0x8c95`, branches to `0x8cae`, and fetches publisher bytes there. It
lands first in the last byte of the publisher's `Bgeu`, then at `0x8cb1` in the
last byte of its `J` followed by its next narrow instruction. The three fetched
bytes are `00 a8 61`, hence the observed little-endian word `0x61a800`.

This also corrects one factual premise in the task brief: the byte streams do
not re-converge at `0x8c98` in this pinned image.

```text
VMA 0x8c98, default: file 0x8cf4 = c0 20 00 = Memw
VMA 0x8c98, +0x100:  file 0x8d98 = 36 81 00 = Entry a1, 0x40
```

The `+0x100` function rooted at `0x8c98` is independently coherent and is an
executed publisher helper. Therefore `[0x8c98, 0x8cbc)` has two semantically
anchored overlapping code views. The current address-only ROM-overlay model
cannot serve both. Moving the boundary backward would break the base callee;
moving it forward would break the publisher. No static range edit is derived.

## 1. Ground-truth candidate disassembly

### Tools and configuration limits

The requested independent tools were run directly on raw extracts:

```text
GNU xtensa-lx106-elf-objdump 2.45.50.20251209
Ubuntu LLVM 20.1.8 llvm-mc
```

Commands:

```sh
xtensa-lx106-elf-objdump -D -b binary -m xtensa \
  --adjust-vma=0x8c6c candidate.bin

llvm-mc-20 --disassemble --triple=xtensa --mattr=+density
```

Both public disassemblers have a configuration limitation relevant to this
management core. The LX106 configuration lacks the windowed-register option,
so it prints windowed opcodes such as `Entry`, `Call8`, `Retw`, and `RetwN` as
`excw`. LLVM's generic Xtensa target exposes only the `density` feature; it
rejects windowed instructions, `Loop`, and `Rsil`. They nevertheless agree on
every instruction that both implement, including byte widths and all scalar
branch targets. The repository's config-exact decoder supplies the names of the
unsupported windowed opcodes; its full output is recorded in
`build/experiments/firmware-re/8c6c-ours-plus5c.log` and
`build/experiments/firmware-re/8c6c-ours-plus100.log`.

### Candidate A: default `+0x5c`, file `0x8cc8`

Raw `[0x8cc8, 0x8cf4)` bytes:

```text
36 41 00 30 62 00 81 36 ea 41 3e ea 51 3a ea 0c
39 62 a0 f7 0c 17 82 c8 40 76 89 2e 92 08 00 37
69 1f 98 28 27 99 1a 98 05 07 69 0c
```

Exact `xtensa-lx106-elf-objdump` output:

```text
00008c6c <.data>:
    8c6c: 364100          excw
    8c6f: 306200          rsil    a3, 2
    8c72: 8136ea          l32r    a8, 0x354c
    8c75: 413eea          l32r    a4, 0x3570
    8c78: 513aea          l32r    a5, 0x3560
    8c7b: 0c39            movi.n  a9, 3
    8c7d: 62a0f7          movi    a6, 247
    8c80: 0c17            movi.n  a7, 1
    8c82: 82c840          addi    a8, a8, 64
    8c85: 76892e          excw
    8c88: 920800          l8ui    a9, a8, 0
    8c8b: 37691f          bbci    a9, 3, 0x8cae
    8c8e: 9828            l32i.n  a9, a8, 8
    8c90: 27991a          bne     a9, a2, 0x8cae
    8c93: 9805            l32i.n  a9, a5, 0
    8c95: 07690c          bbci    a9, 0, 0x8ca5
```

`llvm-mc-20 --mattr=+density` output, associated with each supplied byte line
(address annotations added; `invalid` is LLVM's warning):

```text
8c6c  36 41 00  invalid                         # Entry, unsupported windowed op
8c6f  30 62 00  invalid                         # Rsil unsupported by LLVM target
8c72  81 36 ea  l32r a8, . -22312
8c75  41 3e ea  l32r a4, . -22280
8c78  51 3a ea  l32r a5, . -22296
8c7b  0c 39     movi.n a9, 3
8c7d  62 a0 f7  movi a6, 247
8c80  0c 17     movi.n a7, 1
8c82  82 c8 40  addi a8, a8, 64
8c85  76 89 2e  invalid                         # Loop unsupported by LLVM target
8c88  92 08 00  l8ui a9, a8, 0
8c8b  37 69 1f  bbci a9, 3, . +35               # absolute 0x8cae
8c8e  98 28     l32i.n a9, a8, 8
8c90  27 99 1a  bne a9, a2, . +30               # absolute 0x8cae
8c93  98 05     l32i.n a9, a5, 0
8c95  07 69 0c  bbci a9, 0, . +16               # absolute 0x8ca5
```

The config-exact decode of the three unsupported structural opcodes is:

```text
0x8c6c  Entry a1, 0x20
0x8c6f  Rsil a3, 2
0x8c85  Loop a9, 0x8cb7
```

The window in the question ends before this function does. Extending the same
default byte stream establishes its complete control flow:

```text
0x8c98  Memw
0x8c9b  S32iN  a7, [a4]
0x8c9d  Memw
0x8ca0  L32iN  a9, [a5]
0x8ca2  Bbci   a9, 1, 0x8ca0
0x8ca5  L8ui   a9, [a8]
0x8ca8  And    a9, a9, a6
0x8cab  S8i    a9, [a8]
0x8cae  Addi   a8, a8, 96
0x8cb1  Addmi  a4, a4, 0x1000
0x8cb4  Addmi  a5, a5, 0x1000
0x8cb7  Wsr    a3, PS
0x8cba  RetwN
```

Every internal target is an instruction boundary: loop end `0x8cb7`, forward
branches `0x8cae` and `0x8ca5`, and backward poll `0x8ca0`. There is no default
stream dead-end at `0x8cb1`; it is a valid `Addmi`.

### Candidate B: `+0x100`, file `0x8d6c`

Raw `[0x8d6c, 0x8d98)` bytes:

```text
22 2a 00 20 28 74 90 00 00 00 00 00 36 41 00 9c
42 41 17 ea 0b 22 20 20 74 58 34 50 52 a0 39 05
48 44 40 22 a0 39 02 1d f0 00 00 00
```

Exact `xtensa-lx106-elf-objdump` output:

```text
00008c6c <.data>:
    8c6c: 222a00          l32i    a2, a10, 0
    8c6f: 202874          extui   a2, a2, 8, 8
    8c72: 900000          excw
    8c75: 000000          ill
    8c78: 364100          excw
    8c7b: 9c42            beqz.n  a2, 0x8c93
    8c7d: 4117ea          l32r    a4, 0x34dc
    8c80: 0b22            addi.n  a2, a2, -1
    8c82: 202074          extui   a2, a2, 0, 8
    8c85: 5834            l32i.n  a5, a4, 12
    8c87: 5052a0          addx4   a5, a2, a5
    8c8a: 3905            s32i.n  a3, a5, 0
    8c8c: 4844            l32i.n  a4, a4, 16
    8c8e: 4022a0          addx4   a2, a2, a4
    8c91: 3902            s32i.n  a3, a2, 0
    8c93: 1df0            excw
    8c95: 000000          ill
```

`llvm-mc-20 --mattr=+density` output:

```text
8c6c  22 2a 00  l32i a2, a10, 0
8c6f  20 28 74  extui a2, a2, 8, 8
8c72  90 00 00  invalid                         # Retw, unsupported windowed op
8c75  00 00 00  invalid                         # padding / ill
8c78  36 41 00  invalid                         # Entry, unsupported windowed op
8c7b  9c 42     beqz.n a2, . +24                # absolute 0x8c93
8c7d  41 17 ea  l32r a4, . -22436
8c80  0b 22     addi.n a2, a2, -1
8c82  20 20 74  extui a2, a2, 0, 8
8c85  58 34     l32i.n a5, a4, 12
8c87  50 52 a0  addx4 a5, a2, a5
8c8a  39 05     s32i.n a3, a5, 0
8c8c  48 44     l32i.n a4, a4, 16
8c8e  40 22 a0  addx4 a2, a2, a4
8c91  39 02     s32i.n a3, a2, 0
8c93  1d f0     invalid                         # RetwN, unsupported windowed op
8c95  00 00 00  invalid                         # padding / ill
```

The config-exact structural decode is `Retw` at `0x8c72`, `Entry a1,0x20` at
`0x8c78`, and `RetwN` at `0x8c93`. Therefore this candidate contains:

1. the last three instructions of a function rooted before `0x8c6c`;
2. zero padding at `0x8c75`;
3. a separate complete function `[0x8c78, 0x8c95)`; and
4. zero padding at `0x8c95`.

It is coherent image content, but not one function rooted at `0x8c6c`.

## 2. Call and entry semantics

The upstream landing is not a decode-size artifact. Under the base stream, the
caller is a coherent windowed function rooted at `0x7fc4`. Its decisive bytes
are:

```text
0x7fde  70 a7 20  Or     a10, a7, a7
0x7fe1  a5 c8 00  Call8  0x8c6c
0x7fe4  a5 80 05  Call8  0xd7f0
0x7fe7  e5 6e fa  Call8  0x26d4
0x7fea  1d f0     RetwN
```

Fresh natural provenance is:

```text
n=53567  pc=0x283b  Callx8 -> 0x8770
n=53578  pc=0x878a  Call8  -> 0xc530
n=53579  pc=0xc530  Entry
n=53630  pc=0xc56e  Call8  -> 0x7fc4
n=53631  pc=0x7fc4  Entry
n=53639  pc=0x7fe1  Call8  -> 0x8c6c
n=53640  pc=0x8c6c  Entry
n=53657  pc=0x8c8b  Bbci   -> 0x8cae
n=53659  pc=0x8cb1  Unknown word=0x61a800
```

The call target is explicit in the instruction encoding, and the target begins
with the required `Entry` only under `+0x5c`. Under `+0x100`, it begins with
`L32i`, reaches `Retw` without establishing the callee frame, and then reaches
padding. This decides criterion 2 independently of boot advancement.

## 3. Image structure and overlay-pattern consistency

No image-derived rule selects `+0x100` for `[0x8c6c,0x8c98)`:

- `FirmwareImage::parse` validates only `$PS1` at file `0x10` and the size word
  at `0x14`; it exposes the file as a base-0 byte image.
- A fresh string scan found only `$PS1` at file `0x10`, not ELF magic or
  `.text`, `.rodata`, or `.symtab` names.
- At this revision, `psp_load_map` is in `src/firmware/mod.rs`. It contains only
  the base `file_start=0x5c, phys_base=0` segment and the high
  `file_start=0x2d100, phys_base=0x08b00000` segment. `psp_map.rs` installs the
  runtime autorefill page table; it carries no low-VMA file-section map.
- The loader comment explicitly says the scattered overlay bounds are
  empirically determined by walk-and-stub, the seams have no padding marker,
  and the `$PS1` container has no segment table from which to derive extents.
- The existing starts have no common alignment or stride. An earlier exhaustive
  scan found no retained standard `[VMA,end,LMA]` table or repeatable section
  header in this artifact.

The existing `+0x100` range `[0x8c98,0x8d52)` is nevertheless independently
anchored: the publisher calls VMA `0x8c98`, whose `+0x100` bytes are a coherent
`Entry a1,0x40 ... RetwN` function ending at `0x8d50`. Around the collision:

```text
+0x100 bytes at VMA 0x8cac:
  87 ba 02       Bgeu ..., 0x8cb2
  46 27 00       J 0x8d50
  a8 61          L32iN ...             # at 0x8cb2

base service branch target 0x8cae enters byte 02 above;
the next fetch at 0x8cb1 is 00 a8 61 -> word 0x61a800.
```

Thus the evidence proves two valid overlapping sections/aliases; it does not
yield a wider or narrower static boundary. This is exactly the case where a
hand-added interval cannot be made correct by adjusting its endpoints.

## 4. Proposed loader change and downstream verification

### Production diff

None. In particular:

```text
$ git diff -- src/firmware/mod.rs
<no output>
```

Adding `[0x8c6c,0x8c98)` as `+0x100` would violate both the executed `Call8`
ABI and the two independent disassemblies. Shrinking the publisher overlay to
start at `0x8cbc` would make the service coherent but would corrupt the
independently executed publisher rooted at `0x8c98`. Neither is a derived fix.

### Natural and verification-only frontiers

With no fitted production edit, the plain boot remains honestly at:

```text
natural boot: n=53659 stop=Unknown pc=0x8cb1 word=0x61a800
firmware emitted no device-SRAM descriptor stores
```

The existing observational runtime-view discriminator selects the base bytes
only after the real call reaches `0x8c6c`. It reports:

```text
n=53640 pc=0x8c6c: selected BASE for code [0x8cae,0x8cbc)
runtime-view discriminator: returned=true
tail n=53783 pc=0x7fe7 Call8 0x26d4
```

That consequence verifies that the derived base tail passes `0x8cb1`, executes
the `RetwN`, and returns to the caller. The probe then performs a separate,
pre-existing test-only selection at `0x26d4`; it is not evidence of a natural
copy-out. No device-SRAM copy-out or `FW_ALIVE_OFF` publication was reached in
this verification.

## 5. Same-signature scan

An additive, environment-gated probe now scans every candidate entry in
`[0x1a4,0x10000)`. Its recursive checker requires an `Entry`, rejects any
reachable `Unknown`, follows internal branches/loops, and requires a return or
tail exit. It reports two inventories because the task brief's assumed
signature and the actual `0x8c6c` signature are different.

Run:

```sh
XDNA_FW_PROBE=1 cargo test --lib m2c_probe_overlay_gap_scan -- --nocapture
```

Full output is in
`build/experiments/firmware-re/8c6c-overlay-gap-scan.log`.

### 5.1 Requested nominal signature

These 34 merged ranges have at least one root where:

1. the installed first eight bytes are the default `+0x5c` bytes;
2. `+0x5c` and `+0x100` differ;
3. recursive analysis rejects the `+0x5c` root; and
4. recursive analysis accepts the `+0x100` root.

`Roots` is the count before merging; `example` gives one accepted `+0x100`
entry and its heuristic literal/coherence score.

| Candidate VMA range | Roots | Example evidence |
|---|---:|---|
| `0x2568-0x262f` | 1 | `0x2568/score=30` |
| `0x3cf8-0x4a0a` | 24 | `0x3cf8/score=62` |
| `0x4a38-0x4a59` | 1 | `0x4a38/score=50` |
| `0x4ae0-0x4b1b` | 1 | `0x4ae0/score=55` |
| `0x4b78-0x501b` | 16 | `0x4b78/score=62` |
| `0x5190-0x55f5` | 6 | `0x5190/score=58` |
| `0x5d30-0x7bce` | 49 | `0x5d30/score=62` |
| `0x7c20-0x7c59` | 1 | `0x7c20/score=54` |
| `0x7d40-0x7d4b` | 1 | `0x7d40/score=37` |
| `0x7e28-0x851d` | 14 | `0x7e28/score=61` |
| `0x85c8-0x86f8` | 2 | `0x85c8/score=64` |
| `0x8720-0x896b` | 7 | `0x8720/score=53` |
| `0x89d4-0x8c95` | 10 | `0x89d4/score=54` |
| `0x8d54-0x8d85` | 1 | `0x8d54/score=55` |
| `0x8db4-0x8f43` | 6 | `0x8db4/score=47` |
| `0x9068-0x9291` | 10 | `0x9068/score=41` |
| `0x92bc-0x936d` | 4 | `0x92bc/score=38` |
| `0x9390-0x93e9` | 2 | `0x9390/score=44` |
| `0x9470-0x9492` | 1 | `0x9470/score=32` |
| `0x94b8-0x95eb` | 8 | `0x94b8/score=30` |
| `0x9790-0xb101` | 35 | `0x9790/score=44` |
| `0xb2c8-0xb4d4` | 3 | `0xb2c8/score=82` |
| `0xc0e8-0xc489` | 9 | `0xc0e8/score=57` |
| `0xc4f4-0xc525` | 1 | `0xc4f4/score=57` |
| `0xc538-0xc602` | 5 | `0xc538/score=39` |
| `0xc7e0-0xc828` | 1 | `0xc7e0/score=66` |
| `0xc850-0xc882` | 2 | `0xc850/score=42` |
| `0xc894-0xcc19` | 11 | `0xc894/score=62` |
| `0xccc4-0xcd93` | 4 | `0xccc4/score=54` |
| `0xce44-0xd864` | 29 | `0xce44/score=41` |
| `0xde04-0xdf7c` | 10 | `0xde04/score=62` |
| `0xe0b8-0xe0db` | 1 | `0xe0b8/score=35` |
| `0xe344-0xe441` | 7 | `0xe344/score=39` |
| `0xe477-0xe731` | 15 | `0xe477/score=38` |

These are suspected overlay gaps, not derived loader ranges. The same stored
bytes also form coherent code at an address shifted by `0xa4`, so static
coherence alone cannot say which VMA is canonical. Each root needs an
independently anchored call, pointer, vector slot, or literal consumer before it
can justify a loader edit.

### 5.2 Actual `0x8c6c` signature

The corrected scan also asks whether a coherent default function starts in the
installed base view but encounters different installed bytes before its own
return. It found 14 static collision clusters:

| Coherent default range | Base root(s) | First installed-overlay intrusion |
|---|---|---|
| `0x260c-0x26d3` | `0x260c` | `0x2630` |
| `0x4a00-0x4aae` | `0x4a00` | `0x4a0c` |
| `0x4f50-0x5076` | `0x4f50` | `0x501c` |
| `0x5524-0x5699` | `0x5524` | `0x55f8` |
| `0x7b94-0x7bdc` | `0x7b94` | `0x7bd0` |
| `0x85c4-0x8790` | `0x85c4`, `0x866c` | `0x86f8` |
| `0x88e0-0x89d8` | `0x88e0` | `0x8970` |
| `0x8c6c-0x8cbc` | `0x8c6c` | `0x8c98` |
| `0x8e80-0x8f45` | `0x8e80`, `0x8f16` | `0x8f44` |
| `0x95d0-0x95ee` | `0x95d0` | `0x95ec` |
| `0xc39c-0xc4a4` | `0xc39c` | `0xc48c` |
| `0xc624-0xc683` | `0xc624` | `0xc648` |
| `0xd84c-0xd8a7` | `0xd84c` | `0xd864` |
| `0xdf84-0xdfa3` | `0xdf84` | `0xdf98` |

Only the `0x8c6c` root is independently anchored by the natural execution
trace in this investigation. The other 13 may simply be coherent shifted
aliases cut by already-correct overlays; they are not confirmed runtime
collisions and must not be used to change production mapping without
provenance.

### Probe diff

The only source change is this observational test:

```diff
diff --git a/src/firmware/boot_tests/coherence_mapper.rs b/src/firmware/boot_tests/coherence_mapper.rs
index ecc8cefc..4101a78b 100644
--- a/src/firmware/boot_tests/coherence_mapper.rs
+++ b/src/firmware/boot_tests/coherence_mapper.rs
@@ -1251,6 +1251,78 @@ fn m2c_probe_coherence_mapper() {
     assert!(negatives.iter().all(|range| !calibrated.contains(range)));
 }
 
+/// List low-VMA entries whose installed view is the default `+0x5c`, then find
+/// both `+0x100` candidates and coherent default functions cut by an installed
+/// overlay. This is a candidate inventory, not a placement oracle: shifted
+/// aliases need an independently anchored call/pointer before they can become
+/// loader ranges or confirmed collisions.
+#[test]
+fn m2c_probe_overlay_gap_scan() {
+    if std::env::var("XDNA_FW_PROBE").is_err() {
+        eprintln!("skip: set XDNA_FW_PROBE=1 to scan for unmapped +0x100 candidates");
+        return;
+    }
+    let Some(path) = firmware_path() else { return };
+    let raw = std::fs::read(path).expect("read firmware");
+    assert_calibration_image(&raw);
+    let mut proc = FirmwareProcessor::load_m2c(FirmwareImage::parse(&raw).expect("parse firmware"));
+    let mut gaps = Vec::new();
+    let mut collisions = Vec::new();
+
+    for entry in 0x1a4..0x1_0000 {
+        let base_analysis = analyze_entry(&raw, entry, BASE_DELTA);
+        let overlay_analysis = analyze_entry(&raw, entry, OVERLAY_DELTA);
+        let Some(base) = image_bytes(&raw, entry, BASE_DELTA) else {
+            continue;
+        };
+        let Some(shifted) = image_bytes(&raw, entry, OVERLAY_DELTA) else {
+            continue;
+        };
+        let installed: [u8; 8] = std::array::from_fn(|i| proc.bus.fetch8(entry + i as u32, entry + i as u32));
+        if installed.as_slice() != base {
+            continue;
+        }
+
+        if let (None, Some(overlay)) = (&base_analysis, &overlay_analysis) {
+            if base != shifted {
+                gaps.push((entry, overlay.range, overlay.score));
+            }
+        }
+
+        if let Some(base_fn) = base_analysis {
+            let first_intrusion = (base_fn.range.lo..base_fn.range.hi).find(|&addr| {
+                raw.get((addr + BASE_DELTA) as usize)
+                    .is_some_and(|&expected| proc.bus.fetch8(addr, addr) != expected)
+            });
+            if let Some(first_intrusion) = first_intrusion {
+                collisions.push((entry, base_fn.range, first_intrusion));
+            }
+        }
+    }
+
+    let ranges = merge_ranges(gaps.iter().map(|(_, range, _)| *range).collect());
+    eprintln!("=== default-incoherent / +0x100-coherent candidates ({}) ===", ranges.len());
+    for range in ranges {
+        let roots: Vec<_> = gaps
+            .iter()
+            .filter(|(_, evidence, _)| evidence.lo < range.hi && range.lo < evidence.hi)
+            .map(|(entry, _, score)| format!("{entry:#x}/score={score}"))
+            .collect();
+        eprintln!("{:#06x}-{:#06x} roots={}", range.lo, range.hi, roots.join(","));
+    }
+
+    let ranges = merge_ranges(collisions.iter().map(|(_, range, _)| *range).collect());
+    eprintln!("=== coherent default functions cut by installed overlays ({}) ===", ranges.len());
+    for range in ranges {
+        let roots: Vec<_> = collisions
+            .iter()
+            .filter(|(_, evidence, _)| evidence.lo < range.hi && range.lo < evidence.hi)
+            .map(|(entry, _, first)| format!("{entry:#x}/first={first:#x}"))
+            .collect();
+        eprintln!("{:#06x}-{:#06x} roots={}", range.lo, range.hi, roots.join(","));
+    }
+}
+
 #[test]
 fn m2c_probe_calibrated_map_boots_alive() {
     if std::env::var("XDNA_FW_PROBE").is_err() {
```

## 6. Verification

Targeted scan:

```text
test firmware::boot_tests::coherence_mapper::m2c_probe_overlay_gap_scan ... ok
test result: ok. 1 passed; 0 failed; 4122 filtered out; finished in 0.44s
```

Mandatory full library suite after the final probe revision:

```text
test result: ok. 4093 passed; 0 failed; 30 ignored; 0 measured;
0 filtered out; finished in 49.13s
```

Also green:

```text
cargo fmt --all -- --check
git diff --check
```

No changes were committed.

## 7. Ranked derive-only next steps

1. **Obtain a canonical placement artifact for this exact `1502_00` image.** A
   linker map, pre-signing ELF/program headers, or toolchain-emitted load
   manifest that names both overlapping low-VMA sections is the decisive way to
   derive the missing mapping dimension. Do not infer it from which emulated
   path advances farther.
2. **Provenance-check the other 13 static boundary collisions.** For each base
   root, require a compiled-in direct call, registered absolute pointer, vector
   slot, or natural dynamic execution before calling it real. A second case
   where both overlapping roots are independently anchored would prove that the
   address-only interval representation is generally insufficient.
3. **Triage the 34 nominal gap clusters by anchored roots, not score.** Intersect
   them with direct-call and absolute-pointer consumers, then repeat the two-tool
   disassembly at each anchored target. Add no range merely because `+0x100`
   scores better.
4. **Change `load_m2c` only after the selector is derived.** Encode the actual
   artifact rule or mapping key, with tests for both the `0x8c6c` service and
   `0x8c98` publisher. A special case, forced branch, or endpoint tweak would
   conceal the proven overlap rather than model it.
