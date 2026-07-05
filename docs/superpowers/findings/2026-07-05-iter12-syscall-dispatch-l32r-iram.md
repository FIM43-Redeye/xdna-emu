# M2c iter12: l32r reads the IRAM literal pool -- boot clears the PC=0 syscall wall

**Date:** 2026-07-05
**Context:** Firmware-emulation dream, M2c Phase 2, branch
`feat/m2c-mapping-boot-to-idle`. Resolves the iter12 PC=0 wall that the Harvard
local-memory model (`2026-07-05-harvard-local-memory-model.md`) left as the next
lead. Commit `9c72d047`.

## Result

The boot now dispatches the firmware's first `SYSCALL` correctly and advances
past it: 47,437 -> 47,474 instructions. `main` returns normally to the crt0
post-return site, and the new wall is an **undecoded op `0x41f0` at
`0x2000e035`** (iter13's target). Full suite 3954/0 (+1 pinning test).

## The wall (what PC=0 actually was)

Not a null function pointer. The trace-to-wall probe showed the whole dispatch:

```
47433  pc=0x8b043e1  Syscall                      <- firmware's first SYSCALL
47434  pc=0xae0      wsr.excsave1 a3              <- HW vectors to kernel exc vector (vecbase 0x800 + 0x2e0)
47435  pc=0xae3      l32r a3,=0xadc               <- load dispatcher addr from the vector's literal pool
47436  pc=0xae6      jx a3        (a3 = 0x0)      <- ... but a3 loaded 0x0
47437  pc=0x0        Unknown                       <- jumps to null
```

The kernel exception vector's stub loads the exception-dispatcher address from
its literal pool at `0xadc` and jumps there. The iter7 static probe
(`m2c_probe_vector_table`) confirmed that literal holds **`0x28b4`** in the
image. But the live `l32r` returned **`0x0`**.

## Root cause (a fidelity bug in the Harvard overlay)

The vector literal at `0xadc` sits in the low window, so the Harvard model
routed its `l32r` read through the mutable `local_data` overlay. A write-watch
on `0xadc` found the clobber:

```
[FW_WATCH] fill off=0x4 len=0xfec val=0x0 covers 0xadc   (x6)
```

The firmware zeroes DRAM `[0x4, 0xff0)` -- the sub-4 KiB block that overlaps the
vector table's literal pool -- as scratch early in boot. That zeroed our overlay
copy of the literal, so `l32r` loaded 0.

On silicon this cannot happen: the vector code **and its literal pool live in
instruction memory (IRAM)**; a data-side DRAM memset touches neither. `l32r`
reads its literal from IRAM. Our model had collapsed IRAM and DRAM into one
low-window backing.

## The fix

Route `l32r` literal loads to the pristine image, not `local_data`: a dedicated
`l32r_load` that translates the target and reads the paddr backing, even for a
low-window target. Everything else low-window (general `l32i`/`s32i`/`l8ui`/
memset -- DRAM scratch) still uses `local_data`.

Grounded in Xtensa L32R semantics: **L32R is THE instruction-stream literal
load**; its pool is placed with the code (IRAM). The low window is the varway56
way-6 identity across the whole boot (`assert_low_window_identity`), so
translating a low target is the identity and never faults -- `l32r_load` reads
IRAM where the MMU-bypass read the DRAM overlay.

This is fix **(a)** of a two-way fork. Fix **(b)** -- a full IRAM/DRAM
address-range split -- was deferred: it needs the actual IRAM/DRAM base/size
boundaries, which the open-source toolchain does not expose. Revisit (b) only if
an access appears that (a) cannot classify (e.g. a low-window `l32i` that must
read image rodata, or an `l32r` of a genuinely DRAM-resident word). None seen so
far; literal pools are compile-time constants and are never rewritten.

## Files

- `src/firmware/xtensa/interp/mem.rs`: `l32r_load` helper (translate + paddr
  read, image-backed); `Op::L32r` routes through it; pinning test
  `low_window_l32r_reads_image_not_clobbered_local_data` (clobbers the overlay
  at the literal's vaddr, asserts `l32r` still reads the image).
- `src/firmware/mod.rs`: integration gate `m2c_boot_advances_into_c_runtime`
  reworked -- it pinned "never wall at `0x2000e035`" (a misread: reaching that
  address is now the correct path, `main` returning normally), now pins "never
  wall at PC=0".

## Note on the Harvard finding's "cleared 0x2000e035" claim

The prior finding said the Harvard model "clears the wall at `0x2000e035`." More
precisely: it diverted the boot away from `0x2000e035` to the deeper PC=0 wall
(the syscall dispatch). `0x2000e035` is the crt0 site reached *after `main`
returns* -- undecoded op `0x41f0`. iter12 restores the boot to it, this time via
`main` returning after a correctly-serviced syscall.

**The integration gate cannot prove the servicing is correct** (empirically
established during review): the pre-Harvard iter10 state *also* reached
`0x2000e035` at ~47.5k instructions with the same `0x41f0` wall, despite its
stack-store-drop bug -- because `window_exceptions=0` keeps the crt0->`main`
return chain in the physical register-window file, immune to lost stack data. So
the same-address wall is consistent with either a correctly- or
incorrectly-serviced syscall, and neither the `>20k` floor nor `unknown_op.pc`
distinguishes them. The gate therefore pins only the coarse iter12 regression
(no PC=0 wall); the precise guard for the l32r-reads-IRAM behavior is the unit
test `low_window_l32r_reads_image_not_clobbered_local_data`.

## Non-blocking follow-up

`l32r_load` goes through `Cpu::translate`, which on a *miss* would mutate TLB
state (autorefill). Today a low-window target is always a clean varway56 way-6
identity hit (the full-boot TLB-write log shows zero writes below `0x08000000`),
so this never fires -- but unlike `data_load32`'s low branch, `l32r_load` has no
`assert_low_window_identity` tripwire to flag a future paged/invalidated
low-window mapping loudly. Behavior-neutral asymmetry; worth a one-line note if
this path is revisited. (Sibling of the Harvard model's M3 follow-up.)

## Next iteration (iter13)

Decode/stub the instruction at `0x2000e035` (word `0x41f0`) -- the crt0
continuation after `main` returns. Walk-and-stub as usual.
