# M2c: Harvard local-memory model -- boot clears the 0x2000e035 wall

**Date:** 2026-07-05
**Context:** Firmware-emulation dream, M2c Phase 2, branch
`feat/m2c-mapping-boot-to-idle`. Resolves the iter10 code/DRAM aliasing collision
via a full brainstorm -> spec -> plan -> subagent-driven execution cycle. Spec:
`docs/superpowers/specs/2026-07-05-m2c-local-memory-harvard-model-design.md`;
plan: `docs/superpowers/plans/2026-07-05-m2c-local-memory-harvard-model.md`.

## Result

The boot now **clears the iter10 wall at `0x2000e035`** (the crt0 "main returned"
break). With the syscall stack store persisting, `main` no longer unwinds. The
boot runs ~47,437 instructions and stops at a **new, deeper wall: a fetch of
PC=0** (an all-zero word -- a call/branch to address 0). `funcs_entered` comes
back empty despite the high instruction count, which is a lead for the next
iteration. Full suite 3953/0.

## The problem (from iter10)

The bus is physical-address-keyed. Three virtual regions collapsed onto the same
low physical bytes in one `rom` backing: the code region (`0x2000xxxx` -> paddr
0 via witlb), the low fetch window (vectors/dispatcher `0x1a4..0x291c`, identity),
and low data (stack, memset scratch, identity). Read-only ROM masked it; the
firmware's own 128 MiB region-zero memset, once low writes landed, erased `.text`.

## What the hardware does (two full-boot probes)

1. **PC-region histogram:** execution splits 32,421 fetches from segment B
   (`0x08b0xxxx`), 14,032 from the code region (`0x2000xxxx`), 1,063 from the low
   window (`0x1a4..0x291c`). The low-window code (exception vectors + dispatcher)
   is fetched until 5 instructions before the wall, yet `max_pc 0x291c` sits
   INSIDE the memset's `0x1000..0x08001000` range -- so low instruction and data
   memory cannot be the same physical memory.
2. **Full-boot TLB-write log:** exactly 16 TLB writes, all in the prologue, every
   instruction-TLB op paired with a byte-identical data-TLB op (I == D), NONE
   below `0x08000000`. The firmware never programs a divergent I/D mapping and
   never remaps the low window.

Conclusion: the Harvard split of the low window is a **static core-config fact**
(the Xtensa's separate local instruction/data memories, MMU-independent), not
firmware-driven relocation or a divergent I/D TLB. The strongest proof: the
memset runs FROM code at `0x20004144` and writes `paddr 0x1000..`; if that were
the same physical memory as the executing code, the CPU would fetch zeroed bytes
the instant the memset returned. On silicon it does not.

## The model

Low-window (`vaddr < LOCAL_DATA_END = 0x0400_0000`) DATA loads/stores route to a
new writable `local_data` backing, MMU-bypassed; instruction fetches and all
accesses at vaddr `>= LOCAL_DATA_END` are unchanged. Keyed on the VIRTUAL
address (the code region and the low window collide in physical space, so only
the vaddr distinguishes rodata-at-`0x2000e740` from stack-at-`0x121xx`). The
split lives in the interp (`mem.rs` per-instruction, `fastpath.rs` for the
memset), before translation.

**`local_data` is an image-backed overlay, not blank.** The initial blank
zero-init bet -- that the firmware always writes low data before reading it --
was tested and FAILED in execution: the reset prologue loads its setup constants
(VECBASE, PTEVADDR, TLB values, the `jx 0x20000340` target) via `l32r` from its
literal pool at low image addresses (`0x1a8..`), and blank returned 0, killing
the boot at PC=0 within 85 instructions. The overlay (eager preload of the low
image: `local_data[i] = rom[i + load_offset]`) serves those literals from the
image, routes stack/scratch writes and the memset to `local_data`, and keeps
`rom` pristine (anti-aliasing). This was the spec's documented fallback, promoted
to the model.

## Files

- `src/firmware/mmio.rs`: `local_data` overlay backing (preloaded in
  `new_with_load_offset`), `LOCAL_DATA_END`, `is_local_data`, `load_local*`/
  `store_local*`, `fill_local` (zero-fill allocation cap).
- `src/firmware/xtensa/interp/mem.rs`: four `data_*` helpers route low-window
  data to the overlay; `assert_low_window_identity` (side-effect-free via
  `Mmu::lookup`, lenient) catches a future non-identity low-region remap loudly.
- `src/firmware/xtensa/interp/fastpath.rs`: `try_fill_loop` routes the
  low-window portion of a fill to `fill_local`, shared-`off` invariant preserved.
- `src/firmware/mod.rs`: integration gate (boot must not wall at `0x2000e035`);
  three investigation probes removed.

Commits: `6f98de0a` (backing), `079e20e0` (overlay amendment), `159fea54`
(routing), `d1f13ec9` (side-effect-free assert), `3852fad8` (fast-path fill),
`b5034478` (gate + cleanup). Final Opus whole-branch review: APPROVE.

## Non-blocking follow-ups (from review)

- The `assert_low_window_identity` helper reaches into `cpu.mmu.dtlb[wi][ei]`
  directly (`TlbHit` has no `.paddr`); a `Mmu::lookup_is_identity`/`lookup_paddr`
  accessor would encapsulate the coupling.
- The fast-path local branch lacks the `assert_low_window_identity` tripwire that
  the grind path has (behavior-neutral asymmetry, not a correctness gap).
- No dedicated test for "fill starts local, crosses `LOCAL_DATA_END`, then faults
  above" (provably correct by the shared-`off` invariant).

## Next iteration

Chase the PC=0 wall: a call/branch to address 0 after ~47,437 instructions, with
`funcs_entered` empty. Likely a null function pointer or an unhandled path once
`main` proceeds correctly past where it used to unwind.
