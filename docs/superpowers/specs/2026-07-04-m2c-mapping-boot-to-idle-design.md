# M2c: Mapping Reconstruction & Boot-to-Idle -- Design

**Date:** 2026-07-04  **Issue:** #140 (firmware emulation)
**Milestone:** M2c -- the third and final step of M2 ("boot past the MMU wall to
command-loop idle"). Depends on M2a (base ISA) and M2b (Xtensa MMU-v3 mechanism),
both merged. Predecessor context: `2026-07-04-m2b-mmu-mechanism-design.md`,
finding `findings/2026-07-04-m2b-autorefill-characterization.md`, fidelity-gaps
`fidelity-gaps/firmware-mmu.md`.

## Goal

Boot the real XDNA management firmware (`1502_00/npu.dev.sbin`, Phoenix/NPU1)
from its reset entry all the way to the steady-state command loop
(`_XAie_ExecuteCmd @0x32b34`) waiting on the host mailbox, with `reached_idle =
true`. This is the endgame of the firmware-emulation dream: once the firmware
reaches its idle command loop under its own power, the currently-hardcoded
firmware timings (core reset-deassert, mailbox latency, dispatch pacing) become
things the real firmware *does*, not magic numbers we invent.

## Background: what M2b established

M2b built a faithful Xtensa MMU-v3 mechanism (7-way ITLB / 10-way DTLB, config
registers, `witlb`/`wdtlb`/`iitlb`/`idtlb`, the hardware autorefill page-table
walk, TLB/double-fault exception raise) and wired instruction fetch and data
load/store through a single `Cpu::translate` chokepoint. It boots the firmware's
42-instruction prologue through a **provisional low-region way-4 identity map**
to a wall: the prologue's final `jx` targets virtual `0x20000340`, which the
model cannot translate, so it faults (`EXCCAUSE_INST_TLB_MISS`).

M2b left `varway56=false` (MMU ways 5/6 are hardwired fixed region-protection
entries) and left the code-region virtual->physical map unreconstructed. Those
two are exactly what M2c resolves.

## Characterization: the code-region map is recoverable, not absent

Reverse-engineering the boot path (the reset head and prologue are below the
Ghidra listing's `0x2730` floor, so decoded from raw image bytes) established
the following facts. These are the ground the design stands on.

### Prologue config values (exact, from the image)

The reset head and prologue program, in order:

| Register | Value | Source |
|----------|-------|--------|
| `INTENABLE` | `0x00000000` | `movi.n a0,0; wsr.intenable` @0x214 |
| `VECBASE` | `0x00000800` | `l32r [0x208]; wsr.vecbase` @0x229 |
| `ATOMCTL` | `0x15` | `movi a3,21; wsr.atomctl` @0x22f |
| `ITLBCFG` | `0x00000000` | `movi.n a2,0; wsr.itlbcfg` @0x320 |
| `DTLBCFG` | `0x00030000` | `l32r [0x2b0]; wsr.dtlbcfg` @0x328 |
| `PTEVADDR` | `0x3c000000` | `l32r [0x2b4]; wsr.ptevaddr` @0x331 |

`ITLBCFG=0` / `DTLBCFG=0x00030000` set the variable-way page sizes (relevant
once `varway56=true` makes ways 5/6 variable): the D-side bits `[17:16]=3`
select way-4 page size, and the way-5/6 sizes derive from the same field per
`get_page_size` in `mmu_helper.c`.

### The firmware's own TLB operations (full image inventory)

- **One `witlb` (I-TLB install) in all 248 KB**, @0x33c: way 5, VPN `0x20000000`,
  PPN 0, attr 7 (cached RWX) -- a coarse 128 MB region map of the code region.
  Mirrored by one `wdtlb` @0x342 (same operands, D-side).
- **Seven `iitlb`/`idtlb` pairs** @0x357-0x390: way 6, invalidating VPNs
  `0x20000000, 0x40000000, ..., 0xe0000000` (top nibble stepping by
  `0x20000000`) -- a "clear the way before repopulating" idiom that only makes
  sense if way 6 has more than the two entries the `varway56=false` model gives
  it, i.e. additional evidence for `varway56=true`.
- **Post-`jx` continuation** (file `0x39c` onward): `iitlb`/`idtlb` invalidating
  the way-5 code map the prologue just installed, then `wdtlb` **data**-TLB
  installs for the `0x18000000` window, the `0x40000000-0xe0000000` peripheral
  windows, and the `0x08b00000` rodata/data window; then three data-copy loops;
  then `j 0x291 -> call0 0xe080` (the C entry / `_start`).
- **Runtime data-TLB routine** `FUN_00008620` issues further `wdtlb`/`idtlb` --
  data-side only. Beyond the single prologue way-5 install, the firmware **never
  installs a fine-grained code page for its own `.text`**; it relies entirely on
  the autorefill page table (`PTEVADDR=0x3c000000`) the x86 PSP built before
  firmware start.

### The coherent continuation and the alignment constraint

The `jx` target virtual `0x20000340` has a coherent physical landing site at
file `0x39c` -- unmistakable real boot code (way-5 teardown, data-TLB installs,
data copy, C entry). The steady-state command dispatch is `_XAie_ExecuteCmd
@0x32b34` (its rodata: "Invalid transaction opcode", "Custom OP Transaction
handler hook point"), reached from the C entry.

The autorefill page table works in 4 KB pages, so virtual page `0x20000000` must
map to a **page-aligned** physical page. Virtual `0x20000340` (page
`0x20000000`, offset `0x340`) reaches file `0x39c` only if `virtual 0x20000000 ->
phys 0` **and** `file 0x39c == phys 0x340`. Both hold simultaneously only if the
image is loaded at a **non-zero offset** `L` such that `phys = file - L` (leading
candidate `L = 0x5c`). Under the emulator's current `phys == file`, no real TLB
mechanism -- region or 4 KB page -- maps `0x20000340` to file `0x39c`; you get
file `0x340`, the observed garbage self-loop.

**Conclusion:** there is a single PSP load-offset `L`. Once applied, the
firmware's own way-5 region install (`virtual 0x20000000 -> phys 0`) is the
correct coarse code map, and the synthesized autorefill page table uses the same
`L`. One constant drives reset entry, way-5 coherence, PT contents, and data
windows. `L`'s exact value is pinned empirically in Phase 1 by coherence; the
alignment constraint already proves `L != 0`.

## Architecture

One load-offset `L`, `varway56=true`, a synthesized PSP autorefill page table,
and a system-aperture stub layer -- carrying boot from the real reset entry to
the command-loop wait.

1. **PSP load-offset `L`.** A single constant: emulator physical memory =
   image with `phys = file - L`. Retires the `phys == file` assumption and the
   provisional way-4 identity map. Pinned in Phase 1 by coherence.

2. **Start from the real reset entry** (`~0x214`), not the prologue, so
   `VECBASE=0x800`, `ATOMCTL`, cache-init, and `INTENABLE` are set as the
   firmware expects before the prologue runs. The reset head flows into the
   prologue via its own `j 0x320`.

3. **`varway56=true`.** Honor the firmware's own region/data installs: the way-5
   code bootstrap and (load-bearing for reaching idle) the way-6 data-window
   installs. Without this the C-runtime faults on its own data (`0x08b00000`)
   before idle. This is the M2b-deferred parameterization of ways 5/6, derived
   from `mmu_helper.c`'s `varway56` branches (`get_page_size`,
   `split_tlb_entry_spec_way`).

4. **Synthesize the PSP autorefill page table.** Back the `PTEVADDR` window
   (`0x3c000000+`) with emulator physical memory holding PTEs that map the code
   region `virtual 0x20000000+ -> code phys` (attr 7), contents derived from
   `L`, plus a region entry so `pt_vaddr` is itself fetchable. This backs
   `.text` after the firmware invalidates its way-5 bootstrap; the hardware
   autorefill walk built in M2b reads it exactly as silicon. Contents are
   reconstructed by coherence (the PSP's real table is absent from every
   artifact), commented as such.

5. **System-aperture stub layer.** Extend the existing `SysStub` for the
   off-array MMIO the C-runtime and command loop touch (peripheral windows,
   mailbox registers `0x27010dxx`) so boot traverses rather than walls. The more
   open, RE-driven part -- instrument-first (Phase 2).

Success: boot reaches `_XAie_ExecuteCmd`'s mailbox poll/WAITI with
`reached_idle=true`.

## Phasing (one spec, two internal phases)

### Phase 1 -- pin the map, reach the C entry

Load-offset `L` + reset-head start + `varway56=true` + synthesized code PT.
Coherence gate: boot flows past the wall, through the way-5 teardown -> data-TLB
installs -> data-copy loops -> `call0 0xe080`. Deterministic once `L` is pinned;
de-risks the mapping before any speculative stubbing. This phase resolves the
`varway56` open question from `fidelity-gaps/firmware-mmu.md` empirically: if
boot flows coherently with `varway56=true`, the hypothesis is confirmed by
coherence.

### Phase 2 -- boot to idle

Walk the C-runtime and command loop, stubbing each off-array MMIO wall as it
appears, until `reached_idle=true` at `_XAie_ExecuteCmd`'s mailbox wait. Settles
H2 (whether window overflow/underflow fires in real windowed code -- M2b's
`window_exceptions` counter is already in place to observe it). The stub count
is unknown until the boot is walked; that is why it is a separate, instrument-
first phase rather than a pre-enumerated task list.

## Components

| File | Change | Phase |
|------|--------|-------|
| `src/firmware/mod.rs` (`FirmwareProcessor::load`) | Start from reset entry; apply `L`; retire the provisional way-4 map; install synthesized PT + region backing; set `varway56=true`. Orchestration. | 1 |
| `src/firmware/xtensa/mmu.rs` | Add a `varway56` flag making ways 5/6 software-writable (variable page size from ITLBCFG/DTLBCFG, wider `ei` derivation) per `mmu_helper.c`. M2b hardwired `false`; this is its deferred parameterization. | 1 |
| `src/firmware/psp_map.rs` (new) | Synthesized-PT builder: given `L` and the code extent, emit PTE bytes (`paddr \| ring<<4 \| attr`) into the `PTEVADDR` window + the region entry making `pt_vaddr` fetchable. One responsibility, unit-testable without a boot. | 1 |
| `src/firmware/image.rs` / `Bus` | Thread the load-offset `L` so `phys = file - L` (today: raw bytes at offset 0). Single constant, one place. | 1 |
| `src/firmware/sysstub.rs` | Extend the system-aperture stub layer for the peripheral/mailbox MMIO the C-runtime and command loop touch. Each stub commented with the firmware access it services. | 2 |

## Data flow (boot sequence, post-fix)

1. `load` places the image at `phys = file - L`, points the CPU at the reset
   entry, installs the synthesized autorefill PT + region backing, sets
   `varway56=true`.
2. Reset head runs (identity/pre-paging): sets `VECBASE`, `ATOMCTL`, cache-init,
   `INTENABLE`; `j 0x320`.
3. Prologue runs: sets `ITLBCFG`/`DTLBCFG`/`PTEVADDR`; installs the way-5 code
   region map (now effective, `varway56=true`); invalidates way 6; `jx
   0x20000340`.
4. `jx` lands on the continuation via the way-5 region map (coherent under `L`).
5. Continuation invalidates way 5; subsequent `.text` fetches miss the TLB ->
   autorefill walk -> synthesized PT -> coherent code phys. Data-TLB installs
   (way 6, `varway56=true`) back the `0x08b00000` / peripheral windows.
6. Data-copy loops, `call0 0xe080` -> C runtime.
7. C runtime + command loop run, touching off-array MMIO -> served by SysStub
   (Phase 2) -> `_XAie_ExecuteCmd` mailbox wait -> `reached_idle=true`.

## Testing

- **Phase 1 coherence gate** (firmware-gated, skips cleanly without the image):
  after `load` + boot, assert the run reaches the coherence checkpoints in order
  -- way-5 teardown at the mapped continuation, the data-copy loops, `call0
  0xe080` (C entry) -- via `funcs_entered` / PC observation. This *is* the
  empirical pin of `L`: the test encodes "these PCs must be reached," correct iff
  they are.
- **`L`-derivation / `psp_map` unit test**: the builder produces PTEs that
  translate `virtual 0x20000340 -> the continuation phys`; assert against the
  coherence-derived value without a full boot.
- **`varway56` mechanism tests** (synthetic fixtures in `mmu.rs`): a `witlb` to
  way 5/6 with `varway56=true` installs and translates (vs the M2b no-op with
  `false`); page size honors ITLBCFG/DTLBCFG. Independent of the image.
- **Phase 2 idle gate** (existing `boot_to_idle` harness): `reached_idle=true`,
  `wait_reason` at the command-loop mailbox poll/WAITI, `funcs_entered` includes
  `_XAie_ExecuteCmd`; record `window_exceptions` (H2 observation, not an
  assertion).
- **Full suite** `cargo test --lib` green throughout.

## Known risks / open items

- **`L`'s exact value** -- Phase 1 pins it by coherence; `0x5c` is the leading
  candidate, the alignment constraint proves `L != 0`. If `0x5c` fails the
  coherence gate, the gate's PC-checkpoint mismatch localizes the correct `L`.
- **`varway56=true` is inferred by coherence**, not from an AMD Xtensa config
  artifact (none is open-source). Documented as such in
  `fidelity-gaps/firmware-mmu.md`; Phase 1's coherent boot is the confirmation.
- **Phase 2 stub count is unknown** until the boot is walked -- instrument-first
  by design. If a wall turns out to need real device-state behavior rather than
  a stub (e.g. a peripheral the emulator already models array-side), that is a
  scope signal to surface, not to silently stub over.
- **M2b DEPC / double-fault gap** (`fidelity-gaps/firmware-mmu.md`) becomes live
  if a handler double-faults during boot. Watch for it; if hit, it is a
  prerequisite fix, not out of scope.
- **Reset-head side effects** beyond the registers captured above (the three
  cache-init loops) are modeled as no-ops -- the emulator has no cache; verify
  nothing in the continuation depends on cache state (expected: it does not).

## Non-goals

- Executing host commands / dispatch (M3+). M2c stops at idle.
- Timing calibration / retiring the hardcoded seams (`DEFAULT_MAILBOX_CYCLES`,
  `DispatchGate`, `release_core_resets`). That is the payoff M2c *enables*, in a
  later milestone that observes the now-running firmware.
- Strix/other-device firmware. Phoenix/NPU1 (`1502_00`) only.
