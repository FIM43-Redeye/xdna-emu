# Phoenix management-firmware memory architecture: model and holes

Date: 2026-07-12

Target: Phoenix/NPU1 `1502_00/npu.dev.sbin`

Firmware SHA-256: `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`

Branch: `feat/m2c-mapping-boot-to-idle`

## Confidence convention

This finding uses the repository convention from `ROADMAP.md:5-19`:

- **VERIFIED**: guarded by a test or reproducible on demand;
- **OBSERVED**: captured in a particular run but not continuously guarded; and
- **CLAIMED**: inherited, coherence-inferred, or otherwise not systematically
  verified.

An emulator fact and a silicon fact are marked separately where the distinction
matters.  In particular, a reproducible trace through our reconstructed bus is
not silently promoted into a hardware-memory observation.

## Executive model

The current evidence does **not** describe one flat physical byte array with two
virtual aliases.  The smallest model that fits it is:

1. **VERIFIED (emulator):** low virtual addresses translate to the same numeric
   low physical addresses through way-6 entry 0, but instruction fetch, general
   data access, and `L32r` do not share one backing.  General low data uses
   `local_data`; fetch and `L32r` use instruction-side backing
   (`src/firmware/mmio.rs:58-65,92-121,596-669` and
   `src/firmware/xtensa/interp/mem.rs:171-185`).
2. **VERIFIED (emulator):** low instruction fetch is itself view-sensitive.
   BASE fetches use file `PA + 0x5c`; registered AT intervals use file
   `VMA + 0x100` before the normal physical path.  The selection is explicitly
   keyed by virtual address, not by translated PA
   (`src/firmware/mmio.rs:200-231,647-660`).
3. **VERIFIED (emulator):** the high `0x2000xxxx` execution region is a virtual
   alias onto low numeric PA.  The persistent mapping is supplied by a
   synthesized PSP page table whose PTEs encode
   `PA = VMA - 0x20000000`; high fetches do not match low-VMA AT overlays and
   therefore see the BASE file view (`src/firmware/psp_map.rs:1-8,30-39`).
4. **VERIFIED (emulator):** Segment B is a distinct writable RAM window at
   `0x08b00000`.  `load_m2c` preloads file `[0x2d100,0x3cb10)` into
   `[0x08b00000,0x08b0fa10)` and code executes from it
   (`src/firmware/mod.rs:118-130,402-410,536-550`).
5. **VERIFIED (reconstruction-level):** coherent execution now requires
   different instruction views at the same low VMA at different times.  The
   surviving concrete instance is `0x26d4`: the earlier AT stream crosses it,
   while the later `0x7fe7 Call8` requires the BASE `Entry`.  This is a proven
   limitation of a static `load_m2c` view; it does not yet identify whether
   silicon implements an IRAM reload, an instruction bank, an instruction-cache
   effect, or another selector below the standard ITLB.

The strongest concrete implementation lead found in this pass is not stock
Zephyr and not mask ROM: a sibling AMD MPNPU firmware contains private-tree
source paths for `mpnpu/mmu.c`, `dtlb.h`, and a driver literally named
`drivers/sram_alias/sram_alias.c`.  **OBSERVED:** the code is not available, so
the name is a lead rather than mechanism proof, but it moves AMD's custom
MMU/SRAM-alias layer to the front of the remaining hypotheses.

The central correction is therefore:

> **VERIFIED correction:** identity translation does not prove one persistent
> instruction byte.  It proves only the address produced by the modeled ITLB.
> The earlier sentence “same PA, therefore silicon has exactly one byte” folded
> the instruction port, backing/bank selector, and time into the PA number.

The narrow ITLB result remains valid.  The “single physical byte” inference does
not.

## Consolidated map

### Static source image versus runtime objects

**VERIFIED:** `$PS1` is one signed, uncompressed byte container with no in-band
region table (`src/firmware/image.rs:1-6,21-36` and
`2026-07-11-ps1-header-has-no-third-load-base.md:67-113`).  That says what bytes
the host gives the PSP; it does **not** say the PSP presents those bytes as one
runtime segment.  The current loader already needs two placements plus
piecewise low instruction views.

| Runtime object | Address range | Current classification | File relationship | Confidence |
|---|---:|---|---|---|
| Low I-side window | Executed evidence below `0x10000`; model routes all VMA `<0x04000000` this way | Local instruction aperture with overlay/bank semantics; not adequately modeled as immutable ROM | BASE: `file=PA+0x5c`; AT intervals: `file=VMA+0x100` | **VERIFIED** for emulator routing and executed roots; **CLAIMED** for the physical silicon implementation |
| Low D-side window | VMA `<0x04000000` | Separate writable local DRAM backing | Initially mirrors the BASE image, then diverges under stores/memset | **VERIFIED** emulator behavior; **OBSERVED** coherence requirement from boot |
| Segment B | `0x08b00000..0x08b0fa10` | Separate physical writable RAM carrying data, rodata, and executable tail code | `file=0x2d100+(addr-0x08b00000)` | **VERIFIED** emulator placement and execution; PSP preload on silicon is **CLAIMED** by coherence, not a captured transfer |
| High AT/code region | `0x20000000+` | Virtual alias to low numeric PA, not a separate physical ROM | Normal high fetch: `file=(VMA-0x20000000)+0x5c` | **VERIFIED** emulator PTEs; actual PSP PTE bytes are **CLAIMED**/external |
| Signed `$PS1` file | `file[0..0x3cb10)` | Immutable host source artifact, not a runtime ROM classification | Contains every candidate byte stream | **VERIFIED** |
| Host-visible management SRAM | device `0x03080000+` | Separate shared SRAM behind PCI BAR2 | Not an alias of Xtensa low IRAM | **VERIFIED** from the driver map |

The useful mapping equations are:

```text
low BASE fetch at VMA x:       ITLB PA=x; fetch file[x+0x5c]
low AT fetch at VMA x:         ITLB PA=x; fetch file[x+0x100]
high fetch at 0x20000000+x:    ITLB PA=x; fetch file[x+0x5c]
Segment-B fetch at 0x08b00000+y:
                               RAM byte preloaded from file[0x2d100+y]
low general data at VMA x:     separate local_data[x]
low L32r at VMA x:             instruction-side view, including AT overrides
```

### Low window

**VERIFIED (current-session trace):**
`m2c_probe_itlb_code_view_selector` again reported all nine executed ITLB
modifications at `n=993..1026`, none touching the `0x8000..0x9000` page.  At
publisher and service samples, VMA `0x8cae` translated to PA `0x8cae` via the
same `itlb[6][0] = {vaddr:0,paddr:0,asid:1,attr:3}`.  The run ended at the known
static-view wall `n=53660, pc=0x8cb1`.

That trace establishes the modeled standard-ITLB state.  It does not inspect
an IRAM bank-select register, cache tag, local-memory port, or the real silicon
storage cell.  The current Bus demonstrates the distinction directly:
`fetch8(vaddr, phys)` can return `rom[vaddr+overlay_delta]` without consulting
the PA-backed path (`src/firmware/mmio.rs:221-231`).

**VERIFIED (Harvard requirement):** the boot's low-window memset writes numeric
addresses also used by live low code and literal pools, yet execution continues.
The emulator consequently separates low I-side and D-side backing, and routes
`L32r` to the I-side.  This is guarded by
`low_window_l32r_reads_image_not_clobbered_local_data`
(`src/firmware/xtensa/interp/mem.rs:452-480`).

**Classification:** low code is neither proven mask ROM nor ordinary writable
RAM.  It is an instruction-side local-memory aperture whose byte source is
view-sensitive in the reconstruction.  Calling the emulator vector `rom` a
silicon ROM was an abstraction leak.

### `0x2000xxxx` high region

**VERIFIED (firmware decode):** the reset prologue transiently installs way 5
with `AS=0x20000005`, `AT=7`, which denotes a cached RWX mapping from
`0x20000000..0x27ffffff` to low PA.  It then deliberately invalidates that
entry; the trace records `dtlb[5][0].asid 0->1` at `n=995` and `1->0` at
`n=1027` (`2026-07-08-boot-wake-breach-journey.md:3097-3117`).

**VERIFIED (emulator), CLAIMED (hardware contents):** after the transient map is
removed, `psp_map::install` supplies external page-table state.  Its code-region
PTEs use `PA = VMA - 0x20000000`, but the module explicitly says the real PSP
table is absent and this is a reconstruction of its observed effect
(`src/firmware/psp_map.rs:1-8,30-39`).

Thus high and low are **numeric PA aliases in the current MMU model**.  They are
not proven to be the same silicon storage object, and they are already not the
same byte view in the emulator: low AT overlays are VMA-keyed and high aliases
retain BASE bytes.  For example:

```text
high 0x2000324c -> modeled PA 0x324c -> file 0x32a8 -> 0x000055f8
```

That live word supplies the go-alive publisher root.  It does not prove that
high and low fetch ports terminate at one persistent IRAM cell.

### Segment B at `0x08b00000`

**VERIFIED (emulator):** `load_m2c` copies the exact file tail into a writable
RAM object.  `m2c_load_map_places_segment_b` guards executable anchors at
`0x08b041f0` and `0x08b0e290`
(`src/firmware/boot_tests/guards.rs:722-739`).  The instruction bus executes
Segment-B helpers by absolute `0x08b0xxxx` addresses.

**OBSERVED (current-session trace):** after the test-only BASE selection at
`0x26d4`, the service path executes:

```text
n=53783  0x7fe7       Call8 0x26d4
n=53784  0x26d4       Entry a1,0x50
n=53813  0x2734       Call8 0xc530
n=53831  0xc55c       Callx8 0x08b0e710
n=53843  0x08b0e71d  Loopnez
n=53844..53861        Dhwbi / Addi
n=53862  0x08b0e726  Dsync
n=53863  0x08b0e729  RetwN
```

This helper is a data-cache writeback/invalidate walk, not a byte copy into low
code.  The trace from `0x7fe7` through the service sink contains no instruction-
cache maintenance instruction.  That is a bounded observation, not an absence
proof for the whole firmware.

**CLAIMED (silicon preload):** the PSP performs a corresponding Segment-B
preload.  We have strong execution coherence and zero CPU writes that could
construct the segment, but no captured PSP transaction.  Segment B is therefore
not evidence that the two low views are sourced from this RAM; the prior exact
byte scan found neither collision view in Segment B.

## Resolving identity map versus two views

There are three different propositions that prior findings compressed into
one:

1. `ITLB(VMA 0x8cae) = numeric PA 0x8cae` in the reconstructed CPU;
2. real silicon uses the same identity translation; and
3. numeric PA `0x8cae` names exactly one persistent instruction byte.

**VERIFIED:** proposition 1 is true in the current model.  **CLAIMED:**
proposition 2 is coherence-inferred because `varway56=true` and reset way-6
contents are not backed by an AMD core-config artifact
(`docs/fidelity-gaps/firmware-mmu.md:4-5,20-21`).  Proposition 3 does not follow
from either and is contradicted by the architecture needed to run the image.

The physical identity of a local-memory access is at least:

```text
(I-side or D-side, numeric PA, instruction backing/bank, cache/view state, time)
```

The ITLB probe observed only the numeric-PA component.  Therefore the crack is
primarily **“single physical byte,” not identity translation**.  A sub-MMU IRAM
bank or reload can preserve identity translation while changing the returned
instruction bytes.  The emulator's own VMA-keyed overlay is a mechanism-free
way to reproduce that separation; it is not evidence that hardware uses VMA as
the selector.

There is also an important refinement to the collision history:

- **VERIFIED:** `0x8cae` is no longer itself a required two-byte cell.  Correct
  upstream framing removes the publisher's asserted boundary there while the
  BASE service still executes `Addi a8,a8,0x60` from file `0x8d0a`.  The old AT
  bytes `87 ba 02` occur at file `0x8dae`; the contradiction came from imposing
  the wrong publisher boundary.
- **VERIFIED (reconstruction-level):** `0x26d4` remains a time-dependent view.
  AT execution beginning at VMA `0x2630` uses file `0x2730`; later BASE execution
  begins at VMA `0x26d4` from that same file `0x2730` (`36 a1 00`,
  `Entry a1,0x50`).  At VMA `0x26d4`, the AT file location is instead `0x27d4`
  (`39 a1 f9...`).  The current-session discriminator selected BASE at
  `n=53784` and advanced coherently to the service sink at `0x7fec`.
- **CLAIMED:** the exact silicon mechanism supplying those temporal views is
  still unknown.  The reproducible need for two views is not itself a hardware
  register or memory-content capture.

## Zephyr 3.7.1 and MERT mechanism audit

### Code/data relocation

**VERIFIED (Zephyr v3.7.1 source, tag commit
`9f824289b28d7aea2eee74f62787c385a5005453`):** Xtensa supports
`CONFIG_CODE_DATA_RELOCATION` ([`arch/Kconfig:907-915`](https://github.com/zephyrproject-rtos/zephyr/blob/v3.7.1/arch/Kconfig#L907-L915)).
The build system groups selected object sections into generated linker regions;
`gen_relocate_app.py` emits fixed `reloc_start`, `rom_start`, and size symbols
plus direct `z_early_memcpy` calls
([`gen_relocate_app.py:182-205`](https://github.com/zephyrproject-rtos/zephyr/blob/v3.7.1/scripts/build/gen_relocate_app.py#L182-L205)
and [`419-430`](https://github.com/zephyrproject-rtos/zephyr/blob/v3.7.1/scripts/build/gen_relocate_app.py#L419-L430)).
`z_data_copy()` invokes the generated relocation copy once during startup
([`kernel/xip.c:26-50`](https://github.com/zephyrproject-rtos/zephyr/blob/v3.7.1/kernel/xip.c#L26-L50)).

`__ramfunc` is the same class: a static `.ramfunc` placement under XIP
([`gcc.h:194-210`](https://github.com/zephyrproject-rtos/zephyr/blob/v3.7.1/include/zephyr/toolchain/gcc.h#L194-L210))
and one startup copy (`kernel/xip.c:30-33`).  `NOCOPY` suppresses that copy so
text executes directly in the selected XIP region
([`code-relocation.rst:116-131`](https://github.com/zephyrproject-rtos/zephyr/blob/v3.7.1/doc/kernel/code-relocation.rst#L116-L131)).
The documentation also warns that some early kernel/architecture code executes
before relocation ([`code-relocation.rst:147-151`](https://github.com/zephyrproject-rtos/zephyr/blob/v3.7.1/doc/kernel/code-relocation.rst#L147-L151)).

**Verdict:** this mechanism can explain a non-uniform load-memory/link-memory
layout such as BASE versus AT sections.  It does **not** explain switching the
same VMA on each later dispatch: the generated copy is startup initialization,
not a scheduler overlay manager.  `NOCOPY` XIP is even more static.

### Demand paging and Xtensa

**VERIFIED (Zephyr 3.7.1 source):** `CONFIG_DEMAND_PAGING` depends on the hidden
architecture capability `ARCH_HAS_DEMAND_PAGING`
([`kernel/Kconfig.vm:117-133`](https://github.com/zephyrproject-rtos/zephyr/blob/v3.7.1/kernel/Kconfig.vm#L117-L133)).
In this tag, 32-bit x86 selects that capability
([`arch/Kconfig:78-92`](https://github.com/zephyrproject-rtos/zephyr/blob/v3.7.1/arch/Kconfig#L78-L92));
Xtensa selects code/data relocation but not demand paging
([`arch/Kconfig:127-137`](https://github.com/zephyrproject-rtos/zephyr/blob/v3.7.1/arch/Kconfig#L127-L137)).
Its TLB-miss path touches the PTE page for hardware autorefill and returns; it
does not call `k_mem_page_fault`
([`xtensa_asm2_util.S:395-436`](https://github.com/zephyrproject-rtos/zephyr/blob/v3.7.1/arch/xtensa/core/xtensa_asm2_util.S#L395-L436)).
There is no Xtensa page-in/page-out/backing-store backend.

**VERIFIED (firmware trace):** no executed ITLB operation touches the low code
page after reset initialization.  Stock Zephyr demand paging would require an
architecture page-fault/page-table path and a changed translation or resident
page state; neither exists in the 3.7.1 Xtensa port or in the observed standard
ITLB stream.

**Verdict:** Zephyr demand paging is not the Phoenix mechanism.  The general
concept of an external agent replacing an IRAM bank remains possible, but it is
not `CONFIG_DEMAND_PAGING`.

### Stock Xtensa page-root switching

**VERIFIED (Zephyr source):** Xtensa can change `PTEVADDR`, `RASID`, and pinned
DTLB entries on an actual userspace context restore
([`mmu.c:66-104`](https://github.com/zephyrproject-rtos/zephyr/blob/v3.7.1/arch/xtensa/core/mmu.c#L66-L104)
and [`xtensa_asm2_util.S:280-296`](https://github.com/zephyrproject-rtos/zephyr/blob/v3.7.1/arch/xtensa/core/xtensa_asm2_util.S#L280-L296)).
This is an important correction to the coarse phrase “no MMU selector”: a page
root can change without an executed `WITLB` for every page.

It still does not supply the observed stock-Zephyr mechanism.  Zephyr maps
kernel `.text` identity and shared
([`ptables.c:144-150`](https://github.com/zephyrproject-rtos/zephyr/blob/v3.7.1/arch/xtensa/core/ptables.c#L144-L150)
and [`216-247`](https://github.com/zephyrproject-rtos/zephyr/blob/v3.7.1/arch/xtensa/core/ptables.c#L216-L247)),
clones those mappings into new domains
([`749-789`](https://github.com/zephyrproject-rtos/zephyr/blob/v3.7.1/arch/xtensa/core/ptables.c#L749-L789)),
and propagates one global VA-to-PA mapping to all domains
([`435-466`](https://github.com/zephyrproject-rtos/zephyr/blob/v3.7.1/arch/xtensa/core/ptables.c#L435-L466)).
Stock page-root switching therefore preserves the code view.  AMD's private
MMU layer could change that policy, but its source is unavailable.

### MERT

**CLAIMED:** MERT is the AMD scheduler/runtime layer present in this binary, but
no corresponding open-source MERT source, linker map, `.config`, or relocation
manifest was found in this workspace or the open-source driver/aie-rt trees.

**OBSERVED (current-session sibling-image strings):** the less-stripped
`17f1_10/npu.dev.sbin` (SHA-256
`2ccea76a935be05c81d1a625f5881ca4cec7ea76d9b3d497aaed795a77994e9e`)
contains the Zephyr 3.7.1 banner at file `0x3d7f1` and the private CI/source
paths:

```text
0x3aa52  .../mpnpu-core/soc/xtensa/amd_common/mpnpu/mmu.c
0x3aba9  .../mpnpu-core/soc/xtensa/amd_common/mpnpu/include/soc/common/dtlb.h
0x3e460  .../mpnpu-core/drivers/sram_alias/sram_alias.c
```

Adjacent `sram_alias.c` assertion strings constrain a buffer to power-of-two,
aligned sizes from 4 bytes through 4 KiB and an index below 64.  **CLAIMED:**
those shapes are compatible with an alias-slot/page facility, but without the
source or a call trace they do not prove that it serves Phoenix low IRAM or
`0x26d4`.

The `0x7fe7 -> 0x26d4` timing makes a context transition a plausible selector
point, but the actually executed downstream helper is only the Segment-B
`Dhwbi`/`Dsync` data-cache walk shown above.  Nothing in the trace identifies a
stock Zephyr overlay API or the private alias driver's register programming.

**Combined verdict:** open-source Zephyr explains how statically relocated
BASE/AT sections could be produced at build/startup.  It does not explain the
temporal `0x26d4` re-view.  The private AMD `mmu.c` / `dtlb.h` /
`sram_alias.c` layer is now the best named software mechanism class; an
instruction-side hardware bank/cache or external transfer remains possible.

## Host observability of management Xtensa IRAM

### Answer

**VERIFIED from the open-source interfaces: no documented host path exposes
Phoenix management-Xtensa IRAM or the low code window.**  There is no safe BAR
offset or one-shot sequence to provide.  Guessing a BAR0 offset would violate
the derive-from-toolchain rule and risks the management aperture.

The NPU1 map in
`/home/triple/npu-work/xdna-driver/drivers/accel/amdxdna/npu1_regs.c:16-47,105-156`
is:

| Host resource | Device aperture | Exposed object |
|---|---:|---|
| BAR0 / `resource0` | `0x03000000` | selected MP-NPU PSP/SMU/register aperture |
| BAR2 / `resource2` | `0x03080000` | shared management SRAM |
| BAR4 / `resource4` | `0x030c0000` | mailbox registers |

`DEFINE_BAR_OFFSET` subtracts the relevant aperture base
(`/home/triple/npu-work/xdna-driver/drivers/accel/amdxdna/aie.h:60-67`).
Consequently the existing SRAM oracle is well-defined:

```text
descriptor device 0x030bb000 -> BAR2 offset 0x0003b000
FW_ALIVE  device 0x030bf000 -> BAR2 offset 0x0003f000
```

There is no configured BAR base for Xtensa VMA/PA `0x00000000..` or virtual
`0x2000xxxx`.  BAR2 SRAM is a distinct shared object, not a window onto IRAM.

The apparent alternatives do not cross that ceiling:

- **VERIFIED:** PSP commands expose validate, start/copy-firmware, release-TMR,
  and certificate validation, with no read-memory/halt/snapshot command
  (`/home/triple/npu-work/xdna-driver/drivers/accel/amdxdna/aie_psp.c:18-25,115-205`).
- **VERIFIED, corrected 2026-08-05:** `MSG_OP_AIE_RW_ACCESS` accesses the AIE
  array, not the management Xtensa address space. The current driver encodes a
  context ID plus relative row/column, but Phoenix 1.5.5.391 decodes bytes 4
  and 5 as physical row/column and has no context field. The route must remain
  disabled on NPU1; see
  `2026-08-05-phoenix-aie-rw-access-wire-layout-mismatch.md`.
- **VERIFIED:** aie-rt core debug halt accepts only AIE compute tiles, and
  `XAie_DataMemBlockRead` accepts only AIE compute/memory tiles
  (`/home/triple/npu-work/aie-rt/driver/src/core/xaie_core.c:362-405` and
  `driver/src/memory/xaie_mem.c:474-515`).
- **VERIFIED:** aie-rt's program-memory host aperture belongs to the different
  AIE2PS shim microcontroller (`driver/src/core/xaie_uc.c:52-78`), not the
  Phoenix MP-NPU management Xtensa.

**Ceiling:** the host can read shared SRAM, selected BAR0 PSP/SMU registers,
firmware-provided logs/telemetry, and AIE tile memory/registers.  It cannot halt,
address, or snapshot management-Xtensa IRAM through the documented NPU1 stack.
An authoritative OCD/JTAG interface or a firmware-supported snapshot/readback
command would be required.  No live MMIO or hardware read was attempted in this
pass.

## Assumption audit and holes

| Hole | Current status | Evidence that would close it |
|---|---|---|
| What selects the temporal low instruction view? | **CLAIMED:** AMD's private `sram_alias`/MMU layer is the strongest named lead; IRAM bank/reload, instruction-cache state, or external transfer remain live classes. Standard ITLB selection is **VERIFIED negative** for `0x8cae`; the emulator has no cache model (`src/firmware/xtensa/interp/system.rs:1-7,103-168`). | Source or disassembly-backed call graph for `sram_alias.c`/`mpnpu/mmu.c`, an authoritative register description, or a halted IRAM/cache-tag snapshot at both `0x26d4` epochs. |
| Is `0x26d4` a silicon overlay or the last reconstruction error? | **VERIFIED:** current coherent reconstruction needs AT early and BASE later. **CLAIMED:** mechanism on silicon. | Unstripped ELF/linker map and MERT relocation manifest; alternatively a hardware instruction trace carrying fetched bytes or an IRAM snapshot. |
| Are low and high truly the same silicon storage? | **VERIFIED:** numeric aliases in the emulator. **CLAIMED:** real PSP PTE contents and backing topology. | Authoritative PSP-created PTE dump or management-core debug state. `psp_map.rs` explicitly reconstructs, rather than reads, this state. |
| Is `varway56=true` the exact AMD core configuration? | **CLAIMED:** strongly coherence-inferred and corroborated by an LX7 reference, but no AMD `core-isa.h`/ConfigID exists in the artifacts searched. | AMD core configuration artifact, readable ConfigID/OCD state, or equivalent hardware register capture. |
| Are `+0x5c` and `+0x100` complete placement classes? | **VERIFIED** for the executed anchors that gate boot; **CLAIMED** as a global section map. Hand-bounded ranges and dynamic conflict show the global model is incomplete. | Original ELF program headers, linker map, generated Zephyr `linker.cmd`, or relocation manifest. |
| Does the flat `$PS1` imply a flat hardware load? | **VERIFIED negative:** the header has no usable destination/scatter table; that says nothing about out-of-band placement. | A pre-start memory image or authoritative placement manifest. PSP-loader RE remains closed and is not proposed here. |
| Is Segment B physically preloaded by PSP exactly as modeled? | **VERIFIED** emulator behavior; **CLAIMED** hardware transaction. | A pre-release one-shot read of known Segment-B anchors or an authoritative load manifest. This is independent of the unavailable IRAM read. |
| Does a page-root/context switch participate? | **VERIFIED:** stock Zephyr Xtensa can rewrite `PTEVADDR`/`RASID` on userspace restore, while sharing the same text mapping across domains. The existing “ITLB op” probe does not by itself inventory every page-root write. | Trace `WSR.PTEVADDR`, `WSR.RASID`, and pinned-TLB state at both `0x26d4` epochs; then compare with stock Zephyr policy and any recovered AMD `mmu.c` behavior. |
| Do cache operations participate? | **VERIFIED:** the interpreter decodes 19 cache operations but models all as no-ops. **OBSERVED:** the immediate `0x26d4` downstream trace performs only D-side `Dhwbi`/`Dsync`. | A full executed cache-op timeline annotated with effective addresses, followed by toolchain/hardware semantics for any I-side operation adjacent to a view change. |
| Can host BAR access settle IRAM contents? | **VERIFIED negative** for documented driver/XRT/aie-rt interfaces. | A source-derived NPU1 OCD/debug aperture or a firmware dump command. Do not probe guessed BAR0 offsets. |

## Ranked next evidence

1. **Follow the named AMD alias seam.** In the sibling image, identify callers
   and register effects of the code attributed to `sram_alias.c`, `mpnpu/mmu.c`,
   and `dtlb.h`; then test whether the Phoenix image contains the same routines
   by structure.  This is a bounded mechanism-oriented RE target, not a new
   framing sweep and not PSP-loader RE.
2. **Read-only cache/control timeline in the emulator.** Record every executed
   I-side cache op, `PTEVADDR`/`RASID` write, pinned-TLB change, and call boundary
   between the early AT crossing of `0x26d4` and the later BASE entry.  This is
   the cheapest dynamic discriminator and addresses real fidelity holes: cache
   operations currently decode but have no state, and the existing ITLB probe
   does not summarize all page-root writes.
3. **Recover build artifacts, not more byte sweeps.** A Zephyr/MERT `.config`,
   linker map, generated `linker.cmd`, or code-relocation manifest would settle
   the static section topology and tell us whether a vendor overlay facility was
   linked.
4. **Seek an authoritative management-core debug path.** Only an OCD/JTAG or
   firmware-supported halted snapshot can directly distinguish changed IRAM
   contents from a bank/cache view.  The current PCI BAR interfaces cannot.

No production mapping, probe code, firmware state, or test code was changed.
No commit was made.  The two targeted existing probes passed; a full
`cargo test --lib` run was not required because this pass changes documentation
only.
