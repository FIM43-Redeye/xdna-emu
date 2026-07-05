# M2c iter7: the kernel/general exception vector offset is 0x2e0, not 0x300

**Date:** 2026-07-05
**Context:** Firmware-emulation dream, M2c Phase 2 (walk-and-stub boot-to-idle).
The boot walled with `Unknown pc=0x40b88 word=0x0` (a `call4` into unmapped
space past the image). This finding is the root cause.

## TL;DR

The first exception the boot ever raises is a `syscall` (EXCCAUSE_SYSCALL, not in
exception mode). Our interpreter vectored it to `VECBASE + 0x300`. On this AMD NPU
management core `0x300` is **not** the kernel/general-exception vector -- it lands
misaligned inside the DoubleException region, and the resulting garbage decode ends
in a bogus `call4 0x40b88` (past the image). The real kernel/general-exception
vector is at **`VECBASE + 0x2e0`**. `KERNEL_EXCEPTION_VECTOR_OFFSET` is corrected
from `0x300` to `0x2e0`.

## Why the old 0x300 was wrong

`0x300` was derived (M2a) by cross-checking `XCHAL_KERNEL_VECOFS` across five
**standard** Tensilica reference configs (dc233c, de233_fpu, sample_controller,
de212, test_mmuhifi_c3), which all agree on `0x300`. But this firmware runs on a
**custom AMD Xtensa config** (INFODUMP.md: "Xtensa is a CONFIGURABLE ISA ... AMD's
Xtensa config overlay"), and its vector-table layout deviates from the reference
configs. The correctness rule (derive from THIS core, not a generic prior) wins.

## Derivation (all from the firmware image `npu.dev.sbin`)

Low ROM is served by the ROM aperture at `phys = file - 0x5c` (the pinned M2c PSP
load-offset `L=0x5c`). The image is a flat `$PS1` blob (no segment table), so the
vector table was reconstructed by decoding it with our own windowed-capable decoder
(`m2c_probe_vector_table`) plus `xtensa-lx106-elf-objdump` for the non-windowed
parts.

1. **VECBASE = 0x800, confirmed (not assumed).** The prologue's `wsr.vecbase` is at
   file `0x22c` (runtime phys `0x1d0`); it loads a2 via `l32r` from the literal at
   runtime `0x1ac` (file `0x208`), whose value is exactly `0x00000800`.

2. **Kernel/general-exception vector = VECBASE + 0x2e0 (= 0xae0).** At runtime
   `0xae0` our decoder gives the textbook exception-vector stub:
   ```
   0xae0: wsr.excsave1 a3           ; save a3 (SR 209)
   0xae3: l32r a3, [0xadc] = 0x28b4 ; load the dispatcher address
   0xae6: jx a3                     ; jump to the real dispatcher
   ```
   The dispatcher at runtime `0x28b4` (file `0x2910`) is coherent exception-entry
   code (`rsil`, `wsr.ps`, register save/restore); its `rsr.exccause` is at file
   `0x29c3` = `0x2910 + 0xb3`, consistent with `L=0x5c`. `window_exceptions = 0`
   for the whole boot, so the syscall is the first and only exception -- the window
   vectors never got a chance to (mis)calibrate the offset.

3. **DoubleException handler = 0xb1c (VECBASE + 0x31c), inline.** Decoded coherently:
   `wsr.excsave1/2/5/6; movi.n a2,2; rsr.exccause a3; bne a2,a3,0xb54;
   rsr.depc/wsr.depc (skip faulting instr); ...; rfde`. The old `0x300` -> `0xb00`
   is not this handler's entry: `0xb00` holds `bnez.n a1, 0xb42` followed by
   padding zeros, and a1 (the stack pointer) is always nonzero, so the boot branches
   to `0xb42` -- an address that is NOT an instruction boundary in the real
   `0xb1c`-aligned double handler (its boundaries run ...0xb41, 0xb44...). Decoding
   out of phase from `0xb42` produces the fictitious `call4 0x40b88`.

## What is NOT changed here (deliberate scope)

- **`DOUBLE_EXCEPTION_VECTOR_OFFSET` stays 0x3C0.** The real double handler is at
  `0xb1c` (offset `0x31c`), so `0x3C0` is almost certainly also wrong for this core
  -- but no double fault occurs in the boot yet, and the non-aligned `0x31c` entry
  needs independent confirmation. Flagged in the constant's doc-comment; left for the
  dispatcher-direction work rather than changed on an unexercised path.

## The next wall (out of scope, for the regroup)

With the offset corrected, the boot re-vectors `0xae0 -> jx 0x28b4` into the real
dispatcher and walls at its **first instruction**: `ff f8 81`, op0=`0xf`. This is an
**AMD Xtensa config-overlay opcode** (INFODUMP flags these; lx106/base-Ghidra can't
decode them). Our decoder deliberately rejects op0 `0xE`/`0xF` (`decode/mod.rs`),
so it surfaces as `Unknown`. Identifying/implementing it (behavioral RE, no config
overlay available) -- or modeling the syscall dispatcher at a higher level -- is the
open fork to decide at the regroup.

## Reusable diagnostic

`m2c_probe_vector_table` (in `src/firmware/mod.rs`, XDNA_FW_PROBE-gated) statically
disassembles the vector entries with our decoder and resolves `l32r` literals. It is
the tool that pinned the layout above; extend its entry list to characterize more of
the table.
