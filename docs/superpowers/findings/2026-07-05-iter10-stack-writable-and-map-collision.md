# M2c iter10: the `break` wall is a stack-memory bug, which uncovers a physical-map collision

**Date:** 2026-07-05
**Context:** Firmware-emulation dream, M2c Phase 2, branch
`feat/m2c-mapping-boot-to-idle`. After iter9 (FLIX decode) the boot walled at
`break 0x1,0xf` @ `0x2000e035`. This is the investigation of that wall. It found
a real bug (the firmware stack is not writable), and fixing it uncovered a
deeper physical-memory-map defect. **The stack fix is PARKED** (correct but
entangled with the map fix); the map reconstruction is the next task.

## The `break` is a red herring (twice over)

`0x2000e035` is a crt0 "main returned" trap:

```
2000e024: movi.n a0,0 ; l32r a1,<sp> ; wsr.ps a3 ; rsync
2000e032: call4  main            ; -> 0x20003e00 (a thin wrapper: entry; call8 inner; movi a2,0; retw.n)
2000e035: break  0x1, 0xf        ; "main must never return"
2000e038: j      .               ; infinite-loop backstop
```

`main` is `noreturn` by contract; the `break`+`j .` is the standard "can't
happen" backstop. We only reach it because the firmware **unwound `main`**.

## Why main unwound: a dropped stack store

The single `syscall` on the boot path (`0x8b043e1`) is a structured kernel
service call. Its wrapper stores an arg struct on the stack and passes a pointer
via `threadptr`:

```
08b043cc: entry a1,0x60
08b043d2: movi.n a3,1 ; s32i.n a3,a1,0x10   ; argstruct[0] = 1  (service selector)
08b043d9: addi   a3,a1,0x10                  ; a3 = &argstruct
08b043de: wur.threadptr a3                    ; threadptr = &argstruct
08b043e1: syscall
08b043e4: retw.n                              ; (expects to resume here)
```

The kernel dereferences `threadptr` -> reads `0` (not `1`): **the stack store was
dropped**. Worse, the exception dispatcher (`0x28b4`) reads its saved context
from `a1+0x1c/0x24/0x28/0x38` -- all stack memory -- so it operates on garbage
(raw image bytes), walks off, and `retw.n`'s up the stack until `main` returns
-> `break`.

### Root cause: the low region is read-only, but the stack lives there

The firmware's stack (`a1 ~ 0x121xx`) is at low physical addresses, which
`Bus::region` classified as read-only `Rom` (anything `< ROM_END = 0x04000000`);
`store*` to `Rom` were silently ignored. The Xtensa **register-window ABI**
keeps locals/args/return-addresses in rotating register windows, so for 47,515
instructions no stack *memory* was ever read back -- the drop stayed invisible.
The syscall arg-passing is the first cross-boundary stack **memory** read, and it
gets `0`. Proof: `threadptr=0x12130` -> `phys 0x12130` (identity), and both
`peek8` and `load32` read `[0,0,0,0]`, matching the image bytes at file
`0x1218c` (= `0x12130 + 0x5c`); the MMU maps this RWX (varway56 reset identity,
attr 3), so the backing being read-only is the bug.

## The parked fix (correct, necessary, not sufficient)

Make the low region writable, image-preloaded RAM (renamed `Region::Rom` ->
`Region::Image`; stores land in place at `P + load_offset`, growing the
backing). Saved as `build/experiments/firmware-re/parked-writable-stack-fix.patch`
(reapply with `git apply`). It root-causes the `break`, but it cannot land alone
-- see below.

## What the fix uncovered: the code aliases the low scratch/stack

With the low region writable, the boot's own **128 MiB region-zeroing memset**
(`memset(virtual 0x1000, 0, 0x08000000)`, a genuine `s8i;addi.n` loop, from the
region-descriptor table `@0xe740`) now actually writes -- and it **erases the
code**. The boot re-walls at `0x20004155`, whose real bytes (`1d f0`, `retw.n`)
have been overwritten with `00`.

The map aliases three things that must be physically distinct:

| virtual | -> phys | backing |
|---|---|---|
| code region `0x20000000` | `0x0..` (`psp_map`: `phys = v - 0x20000000`) | image |
| stack `0x121xx` | `0x121xx` (varway56 way-6 identity) | **same image** |
| memset scratch `0x1000..0x8001000` | `0x1000..` (identity) | **same image** |

So `virtual 0x20001000` (code) and `virtual 0x1000` (scratch) are the **same
physical byte**. Read-only Image masked it (all low writes dropped); writable
Image exposes it (the 128 MiB zero erases `.text`). On real hardware the firmware
would never zero its own code -- so mapping the code region linearly onto
`phys 0x0` is wrong: the C-runtime code's true physical base is **not** `0x0`, it
is a region that does not overlap the low scratch/stack RAM the firmware zeroes.
The firmware does **not** remap `virtual 0x1000` itself (its `witlb`/`wdtlb`
target the `0x20000000` region), so the low identity is the firmware's intent --
the code placement is the free variable.

## Resolution direction (next task)

Reconstruct the code region's true physical base by coherence (it is PSP-defined
and absent from every artifact, exactly like the original M2c code-map work):
find a base for `virtual 0x20000000` such that the firmware's zeroed regions
(`0x1000` size 128 MiB, `0x8000` size 64 MiB, ... from the `@0xe740` table) do
NOT overlap the code, so code, stack, and scratch occupy distinct physical
memory. Then the parked writable-stack fix reapplies cleanly and the memset
zeros scratch RAM, not `.text`.

Open sub-questions for the examination:
- Full decode of the `@0xe740` region table (all `{base,size}` entries) -- it is
  the firmware declaring its memory layout.
- Where do the large regions (128/64 MiB) physically live -- separate DDR
  aperture(s), or device memory we should stub rather than back with RAM?
- Does the code belong in segment B's physical range (`0x08b00000`), or another
  base entirely? (Some code already runs from segment B, e.g. `0x8b04xxxx`.)

## State

Fix parked (reverted from the tree; patch saved). Suite green on the old masked
path. Diagnostic probes added: `m2c_probe_syscall_service` (logs each syscall's
threadptr arg struct). `xtdis` gained `XTDIS_RAW=1` earlier.
