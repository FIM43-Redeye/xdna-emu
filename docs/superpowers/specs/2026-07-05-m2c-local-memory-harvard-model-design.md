# M2c: Firmware local-memory Harvard model (design)

**Date:** 2026-07-05
**Context:** Firmware-emulation dream, M2c Phase 2, branch
`feat/m2c-mapping-boot-to-idle`. Resolves the code/DRAM aliasing collision found
in iter10 (the boot's own 128 MiB region-zeroing memset erases `.text` once the
low region is made writable). Supersedes the parked writable-stack fix
(`build/experiments/firmware-re/parked-writable-stack-fix.patch`) and the
briefly-considered self-relocation hypothesis.

## Problem

The boot walls at `break 0x1,0xf` @ `0x2000e035` (a crt0 "main returned" trap).
Root cause (iter10): the one `syscall` on the boot path stores an arg struct on
the stack (`s32i.n a3,a1,0x10` at `a1 ~ 0x121xx`) and passes a pointer via
`threadptr`; the kernel dereferences it and reads `0`, not the stored `1`,
because the store was **dropped** -- the low region is read-only `Region::Rom`.

Making the low region writable (the parked fix) exposes a deeper defect: the
boot's region-zeroing routine issues `memset(0x1000, 0, 0x08000000)` -- a real
128 MiB byte-fill -- which, once writes land, **erases the firmware's own code**.
The boot re-walls at `0x20004155` (the memset caller's `retw.n`, now zeroed).

### Why the collision is fundamental to the current model

Our bus is **physical-address-keyed**: the CPU translates vaddr -> paddr, then
calls `bus.load/store(paddr)`, and `Bus::region(paddr)` routes. Three virtual
regions collapse onto the same low physical bytes:

| virtual | -> phys (current) | backing (current) |
|---|---|---|
| code region `0x2000xxxx` | `0x0..` (witlb way-5, attr 7) | `rom` image |
| low fetch (vectors/dispatcher) `0x1a4..0x291c` | identity (attr 3) | `rom` image |
| low data (stack, memset scratch) `0x1000..` | identity (attr 3) | `rom` image |

`virtual 0x20004155` (code) and `virtual 0x4155` (a memset byte) both resolve to
`paddr 0x4155` in the shared `rom` backing. Read-only `Rom` masked this (all low
writes dropped); a writable low region exposes it (the 128 MiB zero erases code).

### What the hardware actually does (evidence)

Two probes over the full 47,515-instruction boot settled the mechanism:

1. **PC-region histogram.** Execution splits: 32,421 fetches from segment B
   (`0x08b0xxxx`), 14,032 from the code region (`0x2000xxxx`), and **1,063 from
   the low window (`0x1a4..0x291c`)** -- the reset head, exception vectors, and
   dispatcher. The low-window code is fetched until `instr 47510`, five before
   the wall. Its `max_pc` `0x291c` sits **inside** the `0x1000..0x08001000`
   memset range. So low-window code is live to the end yet nominally inside a
   128 MiB data-zero: instruction and data memory at low addresses **cannot be
   the same physical memory**.

2. **Full-boot TLB-write log.** Exactly 16 TLB writes, all in the prologue
   (`instr 993..1021`). Every instruction-TLB op (`witlb`/`iitlb`) is paired
   with a byte-identical data-TLB op (`wdtlb`/`idtlb`) -- **I ≡ D always**. They
   map only the system code region (`0x20000000 -> paddr 0`, attr 7) and
   invalidate the stale high identity regions (`0x20000000..0xe0000000`).
   **Zero ops touch the low region** (`< 0x08000000`).

Conclusion: the firmware never programs a divergent I/D mapping and never remaps
the low window. The Harvard split of the low window is a **static core-config
fact** -- the Xtensa's separate local instruction (IRAM) and data (DRAM)
memories, which are MMU-independent (hence no low-region TLB ops). The strongest
proof: the memset runs *from* code at `0x20004144` and writes `paddr 0x1000..`;
if that were the same physical memory as the executing code, the CPU would fetch
zeroed bytes the instant the memset returned and die mid-function. On silicon it
does not.

## Goal

Model the low virtual window as Harvard local memory: **instruction fetches read
the image (local IRAM); data loads/stores use a separate writable backing (local
DRAM).** This removes the shared-write aliasing -- low-data writes no longer
touch the image that fetches read -- so the code region, the low vectors, and the
memset scratch occupy distinct backings. The immediate payoff: the syscall's
stack store persists, `main` stops unwinding, and the boot advances past
`0x2000e035`.

Non-goal: changing the MMU, autorefill, PTE-synthesis, or system-region routing.
No self-relocation modeling (there is none). No change to fetches or to any
access at vaddr >= the window.

## Design

### The split (vaddr-keyed, access-typed)

A load/store is a **local-data access** iff its virtual address is below the
window boundary. Local-data accesses go to the new `local_data` backing,
bypassing MMU translation. All other accesses -- every instruction fetch, and
every data access at or above the window -- are unchanged.

```
LOCAL_DATA_END = 0x04000000
is_local_data(vaddr) = vaddr < LOCAL_DATA_END      // a VADDR predicate
```

The boundary coincides numerically with the existing `ROM_END` but is now a
*virtual*-address predicate applied before translation. It is justified by
coherence: nothing but the low code/data lives below `0x04000000` (the array
aperture starts there); the code region is `>= 0x20000000` and segment B is
`>= 0x08000000`, so neither is ever a local-data access. The 128 MiB memset
(`0x1000..0x08001000`) splits cleanly at the boundary: `0x1000..0x04000000` ->
local DRAM, `0x04000000..0x08000000` -> array stub, `0x08000000..0x08001000` ->
system stub (unchanged drops).

Rationale for keying on vaddr, not paddr: code-region rodata (`0x2000e740` ->
`paddr 0xe740`) and low stack (`0x121xx` -> `paddr 0x121xx`) both land in the
low `Region::Rom` paddr range, so a paddr predicate cannot separate "read the
image" from "read/write DRAM". Only the original vaddr distinguishes them, and
the interp holds it before translating.

Rationale for bypassing the MMU on local data: Xtensa local memories are
MMU-independent, and the firmware provably never remaps or invalidates the low
window (16 TLB writes, none below `0x08000000`) and never faults it (varway56
reset identity, attr 3 = RWX). So `paddr == vaddr` there and translation is a
no-op we can skip. Fetches keep translating (they must, and it already yields the
image); only local *data* bypasses.

### local_data backing

A new writable byte buffer, **offset-keyed from 0** (the local vaddr is the
offset), **blank zero-init**, grown lazily on write -- identical mechanics to the
existing `ram`/`mailbox` backings. Models uninitialized local SRAM/DRAM.

**Zero-init is a correctness bet:** it is correct iff the firmware always writes
a low-data location before reading it (true for stack frames and memset scratch).
If some low-data *read* expects an initialized image constant before any write,
it will read 0 and the boot will wall on that read. That is empirically visible
(a wall at a low-data load returning 0). **Fallback if it occurs:** preload
`local_data` from the image bytes for the affected range (a writable copy),
turning the bet into image-preloaded DRAM. Not expected; recorded here so the
plan's implementer recognizes the signature.

### Bus surface (mmio.rs)

Add, parallel to the paddr-keyed region methods:

- field `local_data: Vec<u8>`
- `const LOCAL_DATA_END: u32 = 0x0400_0000`
- `Bus::is_local_data(vaddr: u32) -> bool` (associated fn, like `Bus::region`)
- `load_local32(off) -> u32`, `load_local8(off) -> u8`
- `store_local32(off, v)`, `store_local8(off, v)`
- `fill_local(off, pattern, byte_len)` (bulk fill, for the fast-path)

(No `peek_local8`: the only side-effect-free reads in the codebase are
instruction peeks, which are fetch-intent and stay on `peek8`/the image. Add one
only if a data-intent local peek consumer appears.)

The local accessors take the local **offset** (== vaddr). They never consult
`Bus::region` and never touch `rom`.

### Interp data path (mem.rs)

Each data load/store helper (`l32i`/`l32i.n`/`l8ui`/`l16*`, `s32i`/`s32i.n`/
`s8i`/`s16i`, and the unaligned 16-bit split helpers) gains a front branch:

```
if Bus::is_local_data(vaddr) {
    // local DRAM, MMU-bypassed
    bus.store_local32(vaddr, v)   // or load_local*, store_local8, ...
} else {
    let paddr = cpu.translate(bus, vaddr, Access::Load|Store)?;
    bus.store32(paddr, v)         // unchanged
}
```

Fetches (`Access::Fetch`, in `mod.rs`/`fastpath.rs` decode) are **not** branched:
they keep `translate -> Region::Rom -> rom` (the image = local IRAM).

### Fill-loop fast-path (fastpath.rs)

`try_fill_loop` must route the local-window portion of a fill to `fill_local`,
so it stays byte-identical to grinding (grinding now routes low-data stores to
`local_data` via mem.rs; the fast-path must match). The per-chunk loop:

- If the current fill `vaddr < LOCAL_DATA_END`: fill `local_data` directly via
  `fill_local`, chunking up to `LOCAL_DATA_END` (no MMU translate -- local data
  never faults). Advance.
- Else: the existing path -- `cpu.mmu.translate` for page size + fault, then
  `bus.fill_pattern(paddr, ...)`.

A fill spanning the boundary fills local below and drops (array/system) above,
exactly as the per-store grind would. The fault-replication arm is unaffected
(local data does not fault; faults only arise past the boundary, unchanged).

## Components / files

| File | Change |
|---|---|
| `src/firmware/mmio.rs` | `local_data` field, `LOCAL_DATA_END`, `is_local_data`, `*_local` accessors, `fill_local`; blank-init + lazy-grow like `ram` |
| `src/firmware/xtensa/interp/mem.rs` | data load/store helpers route `is_local_data(vaddr)` to `*_local`; fetches unchanged |
| `src/firmware/xtensa/interp/fastpath.rs` | `try_fill_loop` routes local-window fill portion to `fill_local` |
| `src/firmware/mod.rs` | remove temp probes `m2c_probe_pc_regions`, `m2c_probe_tlb_writes` |

The MMU (`mmu.rs`), translate, autorefill, PTE synthesis (`psp_map.rs`), and all
system-region routing are untouched.

## Testing

**Unit (mmio.rs):**
- `is_local_data` boundary: `< 0x04000000` true, `>= 0x04000000` false.
- `local_data` round-trips and starts blank (unwritten offset reads 0).
- Anti-aliasing invariant: a `store_local*` at offset X leaves `rom` byte X (and
  `load32` of the code-region vaddr mapping to `paddr X`) unchanged.

**Unit (mem.rs):**
- A store to a low-window vaddr (e.g. `0x1000`) lands in `local_data` and is
  read back by a load from the same vaddr; the image is untouched.
- A store to a code-region vaddr whose paddr overlaps the low range still routes
  through translate to the image path (regression: code-region data is not
  captured by the local branch).

**Fast-path (fastpath.rs):** extend the grind-vs-fast tests with a local-window
destination (e.g. `DEST` in `0x1000..`):
- byte / half / word fills into `local_data` are byte-identical fast vs grind;
- a fill spanning `LOCAL_DATA_END` fills local below the boundary and drops
  above, fast == grind.

**Integration gate (mod.rs boot harness):**
- The boot advances **past** `0x2000e035` (the iter10 wall). The new wall's PC is
  recorded in the finding; `reached_idle` is not required by this task.
- `cargo test --lib` green (full suite), temp probes removed.

## Success criteria

1. The syscall stack store at `0x121xx` persists in `local_data`; the kernel
   reads `1` via `threadptr`; `main` no longer unwinds.
2. The 128 MiB memset fills `local_data`, not the image; low vectors/dispatcher
   and the code region survive (no zeroed-code wall).
3. Boot advances past `0x2000e035`.
4. Fast-path fills into the local window are byte-identical to grinding.
5. Full suite green; temp probes removed.

## Risks

- **Zero-init bet** (see above): fallback is image-preloaded `local_data`.
- **Boundary width:** if a later boot phase makes a legitimate *data* access to
  the image at a vaddr `< 0x04000000` (rodata co-located with the low vectors),
  it would wrongly read blank DRAM. Not observed in the boot to date (the low
  code's `l32r` literals resolve to high/relocated addresses, e.g. the
  dispatcher's `0xfffe3094`). Same signature and same fallback as the zero-init
  bet (preload the affected range).
