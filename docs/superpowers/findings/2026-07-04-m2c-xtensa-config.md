# M2c Task 1: Xtensa config extraction -- verdict

**Date:** 2026-07-04 **Issue:** #140 (firmware emulation, M2c mapping/boot-to-idle)
**Status:** Bounded investigation complete. `varway56` is **not** confirmed by a
literal AMD configuration artifact. It **is** materially better-supported than a
pure coherence-only inference: a QEMU reference core built on the *same*
hardware generation as AMD's management core (LX7) sets it true, and the task
brief's own stated fallback premise (all six MMU-enabled QEMU cores agree on
`false`) turns out to be wrong once actually checked -- the real split is 3-3,
and the LX7 core sits in the `true` half. M2c Task 2 proceeds on
`varway56=true`, now with this corroboration in addition to the Phase 1
coherence gate.

## Step 1: the firmware image itself

```
strings -t x ../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin \
  | grep -iE 'xtensa|tensilica|core-isa|dc[0-9]{3}|LX[0-9]'
```

One hit: a `dc[0-9]{3}`-shaped substring inside a 128-character run of hex
digits immediately preceding the `"Release 1.5.5.391"` string at file offset
`~0x82`. Confirmed by widening the strings window (`-n 4`, `-n 8`) and reading
the surrounding bytes: this is a SHA-256 hex digest (the same digest field
`docs/superpowers/findings/2026-05-20-npu-firmware-format.md` already
documents at this location), and the `dc8` match is coincidental hex noise,
not a Tensilica core name (`dc232b`/`dc233c` etc.). No `xtensa`, `tensilica`,
`core-isa`, or `LX[0-9]` string appears anywhere in the 248,592-byte image.
**Not found.**

## Step 2: xdna-driver / RyzenAI-SW

```
grep -rniE 'xtensa|tensilica|varway|core-isa' \
  /home/triple/npu-work/xdna-driver /home/triple/npu-work/RyzenAI-SW
```

90 hits. Filtering `varway` and `core-isa` alone (the config-specific terms)
yields **zero** hits in both trees -- no source file names or comments on a
config option or a `core-isa`-style table. The 90 `xtensa`/`tensilica` hits
resolve into two buckets:

1. **Generic ELF machine-type boilerplate** (the large majority): `EM_XTENSA
   = 94` and the string `"Tensilica Xtensa Architecture"` appear in every
   vendored ELFIO copy across `xdna-driver`'s XRT tree and RyzenAI-SW's Chess
   toolchain packages (`elf_types.hpp`, `elfio_dump.hpp`). This is boilerplate
   ELF-spec machine-ID enumeration (every EM_* constant from the ELF ABI is
   listed), present because ELFIO supports arbitrary ELF machine types -- not
   evidence AMD's tooling specifically targets or configures Xtensa.
2. **A genuine Xtensa GCC toolchain artifact** -- see below, the one hit worth
   following up.

### `xtensa-config.h` in RyzenAI-SW's bundled GCC

Found at `RyzenAI-SW/venv/lib/python3.12/site-packages/tps/lnx64/gcc/include/xtensa-config.h`,
a real Tensilica-HAL-format header (`XCHAL_HAVE_*` macros, same shape as the
canonical `core-isa.h` files bundled with QEMU -- see Step 4). This looked
promising enough to chase down properly rather than dismiss on sight. Result:
**it is not AMD's mgmt-core config, and it does not resolve `varway56`.**

- The toolchain it ships inside (`tps/lnx64/gcc/bin/gcc -v`) reports `Target:
  x86_64-pc-linux-gnu`, built from `/proj/rdi/staff/rajshree/nobkup/gcc8.3/gcc-8.3.0/`
  (an internal Xilinx/AMD build path). There is no `xtensa-*`-prefixed `as`,
  `ld`, or `gcc` anywhere alongside it -- this is a plain x86_64 host
  compiler, not an Xtensa cross-compiler. `xtensa-config.h` is very likely
  vendored boilerplate carried in GCC 8.3.0's own source tree (GCC ships a
  per-target config header for every architecture it supports, regardless of
  which one a given build targets) rather than something written for this
  specific chip.
- The header only defines a small subset of the full `core-isa.h` macro set --
  exactly the ones GCC's own code generator consults (ABI, cache line size,
  `XCHAL_HAVE_MMU`, `XCHAL_MMU_MIN_PTE_PAGE_SIZE`). It does **not** define
  `XCHAL_HAVE_SPANNING_WAY` (the macro that is actually `varway56` -- see Step
  4), `XCHAL_HAVE_PTP_MMU`, or the ITLB/DTLB autorefill-way macros. Even if
  this header were authoritative, it has nothing to say about `varway56`.
- It asserts `XCHAL_HAVE_BE = 1` (big-endian). This directly contradicts the
  independently-verified fact (Ghidra 10-architecture auto-ID sweep + real
  `xtensa-lx106-elf-objdump` cross-validation, both in
  `docs/superpowers/findings/2026-05-20-npu-firmware-format.md`) that the
  actual firmware is **little-endian**. Cross-checking QEMU's bundled
  `core-fsf/core-isa.h` (Tensilica's generic/"Free Software Foundation"
  sample reference core) shows the identical `XCHAL_HAVE_BE = 1` -- i.e. this
  header's endianness matches the *generic sample default*, not any real
  production core. Every real production-style core in the QEMU survey
  (`dc232b`, `dc233c`, `de212`, `de233_fpu`, `dsp3400`, `lx106`,
  `sample_controller`, `test_mmuhifi_c3`) is little-endian.

Verdict for this artifact: a real header, genuinely interesting, but the
endianness contradiction alone disqualifies it as AMD's actual mgmt-core
config, and its truncated field set means it couldn't answer the `varway56`
question even if it were. **Not found** (via this route).

## Step 3: the `$PS1` container header (offsets `0x00`-`0x60`)

Read directly from `npu.dev.sbin` (`xxd -l 256`), decoding every field in the
range the brief flagged as undecoded:

| Offset | Bytes (LE) | Decoded | Note |
|--------|-----------|---------|------|
| `0x00`-`0x0F` | (16 bytes) | signature-blob prefix | Already known |
| `0x10` | `24 50 53 31` | `"$PS1"` magic | Already known |
| `0x14` | `10 c9 03 00` | `0x0003C910` = body size | Already known |
| `0x18` | `00 00 00 00` | encrypted flag = 0 | Already known |
| `0x30` | `01 00 00 00` | constant = 1 | Already known |
| `0x38`-`0x47` | (16 bytes) | `12e274db 369e4739 ad5cb017 5e4821ff` | signing fingerprint (already known field, value confirmed) |
| `0x48` | `00 00 00 00` | compressed flag = 0 | Already known |
| `0x50` | `10 c9 03 00` | `0x0003C910` = uncompressed size | Already known, equals body size |
| **`0x58`** | `52 10 01 81` | **`0x81011052`** | newly decoded |
| **`0x60`** | `ff 01 00 00` | **`0x000001FF`** | newly decoded |
| `0x6C` | `10 cb 03 00` | `0x0003CB10` = packed size | Already known, equals file size |
| `0xD0`-`0xEF` | (32 bytes) | `d319d9da b94a93d2 673e97ae 90f55823 a1e5a21b d5811086 ec874119 0c85145b` | image digest |

The two newly-decoded fields at `0x58` and `0x60` do not resemble a Tensilica
config descriptor. Real Tensilica `ConfigID`s are a **paired** 64-bit value
split across two adjacent 32-bit words (`XCHAL_HW_CONFIGID0`/`CONFIGID1`,
e.g. `de233_fpu`'s `0xC1039286`/`0x28C872E0` -- see Step 4); `0x58` is a single
dword with a zeroed neighbor at `0x5C`, not a hi/lo pair, and doesn't match
either LX7 reference core's ConfigID. `0x60`'s value (`0x1FF` = 511, all 9 low
bits set) reads as a small bitmask or version/feature-flag field, again not
ConfigID-shaped. Both are far more plausibly generic AMD PSP-directory
metadata (a version/cookie field and a compatibility/algorithm bitmask -- this
container format is shared across many different AMD chip firmware types, per
the PSP-signed-binary format, not something Xtensa-specific). **Not found.**

## Step 4: QEMU core cross-reference

The brief's fallback text (in `docs/superpowers/plans/2026-07-04-m2c-mapping-boot-to-idle.md`)
asserted "the six MMU-enabled QEMU cores M2b surveyed all have `varway56=false`."
**This is incorrect** -- it does not appear to have been checked against
`target/xtensa/core-*/core-isa.h` directly. Doing that check properly:

QEMU's `xtensa_tlb.varway56` field is not an independent QEMU invention -- it
is set verbatim from a real, documented Tensilica HAL macro,
`XCHAL_HAVE_SPANNING_WAY` ("one way maps I+D 4GB vaddr"), via
`overlay_tool.h`'s `TLB_SECTION`/`ITLB`/`DTLB` macros (`.varway56 = (way56)`
in `TLB_TEMPLATE`, invoked as `ITLB(XCHAL_HAVE_SPANNING_WAY)`). So the six
"MMU-enabled" cores are the six with `XCHAL_HAVE_PTP_MMU=1` (full paged MMU,
the mode `varway56` actually governs), read directly from each core's
`core-isa.h`:

| Core | `XCHAL_HW_VERSION_NAME` | `XCHAL_HAVE_SPANNING_WAY` (= `varway56`) |
|------|------------------------|---------------------------|
| `dc232b` | LX2.1.1 | **0** (false) |
| `fsf` | LX2.0.0 (generic sample) | **0** (false) |
| `test_mmuhifi_c3` | LX3.0.0 (QEMU MMU test core) | **0** (false) |
| `dc233c` | LX4.0.1 | **1** (true) |
| `test_kc705_be` | LX6.0.2 | **1** (true) |
| **`de233_fpu`** | **LX7.1.3** | **1 (true)** |

Six cores, confirmed -- but a clean 3-3 split, not unanimous `false`. The
pattern tracks hardware generation: `varway56=false` only appears on the
oldest surveyed generation (LX2.x) plus one QEMU-internal MMU test config
(`test_mmuhifi_c3`, plausibly chosen specifically *because* it's the less
common configuration, to exercise that code path in QEMU's own test suite);
every newer generation surveyed (LX4, LX6, LX7) has `varway56=true`.

**The most relevant single data point:** `de233_fpu` is built on hardware
release **LX7.1.3** -- the exact same "LX7" generation independently confirmed
for AMD's NPU management core (`docs/superpowers/findings/2026-05-20-npu-firmware-format.md`:
the driver's own debugfs and UAPI headers name it explicitly, `"lx7 firmware"`,
`fatal_error_exception_type` documented as "LX7 exception type"). `de233_fpu`
has `varway56=true`.

This is **not** AMD's own `core-isa.h` -- `de233_fpu` is a different,
QEMU-bundled reference/test configuration, and Tensilica customers routinely
customize TLB options per design. It does not amount to "config found, cite
it." But it is real, checkable, and directly relevant corroboration, in the
opposite direction from what the brief assumed going in.

### Opportunistic values (autorefill ways, `ndepc`, reset vector)

- **Autorefill entries**: every full-MMU core surveyed, `de233_fpu` included,
  uses `XCHAL_ITLB_ARF_ENTRIES_LOG2 = XCHAL_DTLB_ARF_ENTRIES_LOG2 = 2` (4
  entries/way) with no variance across generations. Low-risk if AMD's core
  follows the same universal pattern (informational only; M2b's model already
  assumes 4).
- **`ndepc`**: derived as `(XCHAL_XEA_VERSION >= 2)`, not an independent
  config bit. Every core surveyed, including `de233_fpu`, has
  `XCHAL_XEA_VERSION = 2`, so `ndepc` would read `true` on all of them. If
  AMD's LX7 core follows the same universal pattern, real hardware likely
  does use DEPC/double-fault semantics that M2b's "deliberate, inert"
  simplification (`firmware-mmu.md`'s double-fault-EPC1 row) currently omits
  -- still non-blocking today (the firmware boot path doesn't reach a double
  fault), but a slightly-elevated-confidence signal for whenever that gap
  needs closing.
- **Reset vector**: `core-isa.h`'s `XCHAL_VECBASE_RESET_VADDR` /
  `XCHAL_RESET_VECTOR*_VADDR` are core-fixed straps (e.g. `de233_fpu`:
  `VECBASE=0x2000`, reset vectors `0xFE000000`/`0x1000`). These don't transfer
  usefully to AMD's SoC-integrated core -- boot-vector placement is an SoC
  integration choice, not a core-isa constant, and the real values are
  already independently pinned by the M1.7/M2a boot trace (`PTEVADDR =
  0x3c000000`, `jx` target `0x20000340`). Not a useful cross-reference point;
  noted and not pursued further (per the bounded-investigation scope).

## Verdict

**(b) Not found** -- no artifact in any of the four candidate sources states
AMD's Xtensa LX7 management-core configuration directly. `varway56=true`
still formally rests on the M2c Phase 1 coherence gate (per
`docs/superpowers/specs/2026-07-04-m2c-mapping-boot-to-idle-design.md`), not a
literal config citation.

That said, this is a stronger outcome than a plain "not found." Step 4's
proper QEMU survey (a) corrects a wrong assumption already written into the
M2c plan (the six-MMU-cores-all-false claim), and (b) shows the one core
sharing AMD's confirmed hardware generation (LX7) has `varway56=true`,
consistent with the newer-generation pattern across the whole surveyed set.
Combined with the M2b autorefill-characterization evidence (the firmware's
own boot prologue issues 16 `witlb`/`wdtlb`/`iitlb`/`idtlb` calls at ways 5/6
that are meaningless no-ops under `varway56=false`, including seven
invalidate calls with a changing tag that only make sense against distinct
slots -- i.e. only make sense if `varway56=true`), M2c Task 2 proceeds on
`varway56=true` with two independent, mutually-reinforcing lines of evidence
(coherence + generation cross-reference), not one.

`docs/fidelity-gaps/firmware-mmu.md`'s `varway56` row updated accordingly.
