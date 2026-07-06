# M2c iter13 finding: the "main returns" wall is missing USER-mode exception-vector routing

**Branch:** `feat/m2c-mapping-boot-to-idle` (UNMERGED, kept). **Date:** 2026-07-05.
**Status:** RESOLVED 2026-07-06 (fix landed). This doc's original body below
captured the ROOT-CAUSE hypothesis; the RESOLUTION section corrects two of its
claims after the follow-up investigation.

---

## RESOLUTION (2026-07-06): the fingerprint was wrong; there is ONE unified handler

Two claims in the original body did not survive investigation:

1. **"Locate the User vector by finding the handler that reads THREADPTR
   (`rur`, UR 0xE7)" is DEAD.** The whole image has **25 `wur.threadptr` writes
   and ZERO `rur.threadptr` reads** -- nothing reads threadptr. That fingerprint
   cannot locate anything. (Confirmed with the correct swizzled `rur` encoding
   `?0 0e e3`, verified via the xtdis oracle.)

2. **There is no separate static USER vector to route to.** The real
   general-exception handler is at **phys 0x2958** -- found by scanning for
   `rsr.exccause` sites (only 3 in the image) and reading the one with the
   `bnei a3,1` (EXCCAUSE==1==SYSCALL) dispatch that advances EPC1+3. It is a
   textbook RTOS save-prologue (saves excsave2-6, EPC1/EXCCAUSE/EXCVADDR, full
   a0-a15 + SAR/PS/loop/FPU). Routing the boot's user-mode syscall there is the
   ONLY target that clears the wall (dynamically proven). But **nothing reaches
   0x2958 statically**: no literal equals it (low `0x2958` OR high `0x20002958`),
   no `j` reaches it, and an exhaustive 1024-offset VECBASE sweep finds nothing.
   It is reached at runtime via a dispatch pointer `init` installs in RAM (the
   `0xe1fc` slot, zero in our boot).

**iter7 was mislabeled.** `0xae0 -> 0x28b4` is NOT the kernel/general vector:
`0x28b4` uses the *interrupted* `a1`/`a4` as a live call-frame + dispatch table
(`a4=0x200` at syscall time is just `init`'s leftover) -- architecturally
impossible for an exception handler. And `0xb1c` (VECBASE+0x31c) is the real
**DoubleException** handler, confirmed by its terminating `rfde` + `depc`
manipulation (so `DOUBLE_EXCEPTION_VECTOR_OFFSET` is now `0x31c`, was `0x3C0`).

**The fix that landed (option A, Maya's call -- "working now, faithful later"):**
`raise_general_exception` routes non-double general exceptions directly to the
unified handler `GENERAL_EXCEPTION_HANDLER = 0x2958` (absolute image address,
NOT VECBASE-relative); double faults route to VECBASE+0x31c. No PS.UM branching
(user and kernel general exceptions converge at 0x2958 in our model). Boot now
clears 0x2000e035 (47474 -> 47501 instrs), walling at the NEXT frontier:
`Unknown 0x2a09` = **`rur.fcr`** (the handler saving FPU state) -- our decoder
has `Wur` but no `Rur`. That is iter14.

**FIXME(iter13-B), deferred:** model `init`'s runtime handler-registration so
kernel/user route through their true per-mode VECBASE vectors + the RAM dispatch
pointer, instead of the direct-to-0x2958 shortcut. Needs RE of `init`'s
dispatch-table setup, entangled with the Harvard IRAM/DRAM split. Tracked in the
`GENERAL_EXCEPTION_HANDLER` doc comment (`interp/mod.rs`).

---

## Original body (root-cause hypothesis, 2026-07-05 -- see RESOLUTION for corrections)

## TL;DR (resume here)

The boot walls because `main` returns (crt0 traps that with `break 0x1,0xf; j .`,
which our decoder surfaces as "Unknown op 0x41f0 at 0x2000e035"). `main` returns
because the one boot **SYSCALL** (the MERT hand-off at the end of `init`) is never
serviced -- and it is never serviced because **we route it to the wrong exception
vector**.

**Decisive fact: the syscall fires in USER mode (`PS = 0x60022`, `PS.UM = 1`).**
Real Xtensa routes a general exception three ways: `EXCM=1` -> Double vector,
**`PS.UM=1` -> User vector**, else -> Kernel vector. Our `raise_general_exception`
(`src/firmware/xtensa/interp/mod.rs:369`) models only Kernel + Double -- there is
**no User-vector branch**. So the user-mode syscall is misrouted to a kernel
handler and silently fails.

**NEXT STEP (do this first next session):** locate the **User exception vector** in
the firmware's vector table (VECBASE = 0x800). Its fingerprint: the handler that
reads `THREADPTR` (`rur.threadptr`, UR 0xE7) -- because the syscall passes its
arguments via THREADPTR (see below). Then implement `PS.UM`-based routing
(`EXCM`->double, `UM`->user, else->kernel) so the syscall reaches that handler.
This also corrects iter7's vector-offset assignments, which we now know were wrong.

## How we got here (the chain of evidence)

### 1. `0xfffe3094` / the a15 read is a RED HERRING (closed)
The exception dispatcher's entry `l32r a15` wraps (PC-relative from low PC 0x28b4)
to `0xfffe3094`, stubbed to 0. We swept forced a15 values {0, 1, 0xffffffff,
0x08b095f8}: **every value produced the identical outcome** -- same dispatcher
return target (`0x20003DFC`), same wall, same instruction count (+-1). a15 does not
affect the outcome. Diagnostic: `m2c_probe_a15_loadbearing` (XDNA_FW_FORCE_A15).
Provenance study: `build/experiments/firmware-re/exception-dispatch-pc-verdict.md`
(gitignored) -- no linear rebase or 256KB-wrap lands the literal in-image; the
256KB-wrap target (phys 0x23094) is in the image's zero gap.

### 2. The external-read surface is tiny (enumeration)
Full boot = 47,474 instrs. Of 141 stub sites / 5,886 accesses, only **5 are
System-region (external) reads**, collapsing to **3 distinct addresses, ZERO
confirmed PSP/HW secrets**: `0xfffe3094` (the a15 red herring), `0x8a80228` (our
RAM-aperture floor is set too high -- own data misrouted), and `0xb0027xxx` (a ~3.6
KB memcpy control-block, runtime-computed pointer, the one bounded genuine unknown
-- NOT yet chased). The firmware image is a **single segment** ($PS1 container, no
segment table; high addresses provably not in the file). Details:
`build/experiments/firmware-re/{external-reads-enumeration.md,image-structure-verdict.md}`
(gitignored).

### 3. The syscall calling convention: args via THREADPTR
Every syscall stub (e.g. `0x8b043cc`, whose `syscall` is at `0x8b043e1`) does:
```
entry a1,a1,0x60
movi.n a3,0x1;  s32i.n a3,a1,0x10     ; arg0 (op selector) = 1 at [sp+0x10]
s32i.n a2,a1,0x18                      ; arg = a2 at [sp+0x18]
addi a3,a1,0x10;  wur.threadptr a3     ; THREADPTR = &args
syscall
retw.n a0
```
So **THREADPTR points at a stack arg-block**; the real handler must read THREADPTR.
(Retroactively explains iter6's "threadptr = stack ptr then syscall" observation.)

### 4. The current handler (`0x2e0`->`0x28b4`) is wrong; `0xb1c` is the KERNEL handler
- `0x28b4` (what iter7 wired the kernel vector to) reads **no** EXCCAUSE/THREADPTR/
  EXCSAVE. It indexes a handler table off `a4`, but `a4` = `0x200` (leftover junk),
  so it computes `[0x200 + (0x2450&0xff)*4 + 0x38] = [0x378]` (inside the reset
  code), reads 0, finds no handler, and just returns. Trace-proven (dispatcher runs
  ~14 instrs, never `callx8`s a handler).
- `0xb1c` (VECBASE+0x31c; iter7 mislabeled it "DoubleException") is the real
  **kernel** cause-dispatcher: `wsr.excsave1/2/5/6; rsr a3,EXCCAUSE; movi a2,2;
  bne a2,a3 -> 0xb54`. Cause 2 handled inline (ends `rfe`); **all other causes ->
  `jx [0xb00] = 0xe1fc`**. Experiment (temporarily route kernel vector to 0x31c):
  the syscall correctly reads EXCCAUSE=1 and dispatches, but jumps to `0xe1fc` which
  is **empty** (image AND local_data both zero at 0xe1fc -- verified via
  `m2c_probe_low_window_code`, so NOT a Harvard fetch/data-split issue and NOT a
  runtime-installed handler). `0xe1fc` is the kernel handler's "unexpected cause"
  dead-end -- which is exactly what a **user** syscall wrongly landing in the kernel
  path looks like.

### 5. The syscall is USER mode -> we need the User vector
`m2c_probe_low_window_code` captures `PS = 0x60022` at the syscall: `UM=1`,
`EXCM=0`, `INTLEVEL=2`, `RING=0`, `WOE=1`. User mode. Both `0x28b4` and `0xb1c` are
kernel-path handlers; neither reads THREADPTR. The syscall needs the **User
exception vector**, which we neither route to nor have located yet.

## The fix (two parts, NOT yet done)

1. **Locate the User exception vector.** Disassemble the full vector table around
   VECBASE=0x800 with the xtdis oracle (`build/experiments/firmware-re/xtdis/`,
   FLIX-aware; Ghidra's `listing.txt` uses FILE offsets and mis-decodes FLIX --
   trust xtdis and remember phys = file - 0x5c). The real user handler is the one
   that reads `THREADPTR` (rur, UR 0xE7) and services the syscall. Known vector
   entries so far: `0xae0` (+0x2e0) stub->0x28b4; `0xb1c` (+0x31c) kernel
   cause-dispatcher. The User vector is a third entry TBD.
2. **Implement PS.UM-based routing** in `raise_general_exception`
   (`src/firmware/xtensa/interp/mod.rs:369-385`): `if EXCM -> DOUBLE (0x3C0, itself
   suspect); else if PS.UM -> USER (new const); else -> KERNEL`. Add a
   `USER_EXCEPTION_VECTOR_OFFSET`. Revisit iter7's `KERNEL_EXCEPTION_VECTOR_OFFSET =
   0x2e0` -- it is likely wrong (the real kernel dispatcher is at 0x31c/0xb1c; 0x2e0
   ->0x28b4 may be a narrower/interrupt handler). Do this as a proper SDD task with
   `PS.UM` unit tests; existing tests at `system.rs:271`, `arith.rs:902+` assert the
   0x2e0 routing and will need updating to match the corrected model.

**Coherence gate for the fix:** boot should get PAST the `0x2000e035` main-return
wall once the user syscall is serviced (the handler should NOT return into `init`;
it hands off toward the MERT command/idle loop `FUN_0000c928` @ `0x2000c8cc`).

## Seam note (Maya's framing)
Everything here is firmware-INTERNAL Xtensa exception architecture -- derivable, not
the host/SMU external surface. The external seam (mailbox 0x27010dxx / SMU
handshake) is still ahead, in the command loop we're trying to reach. The one
genuinely-external bounded unknown found so far is the `0xb0027xxx` ~3.6KB memcpy
(item 2), still unchased.

## Diagnostic probes added this session (committed, XDNA_FW_PROBE-gated, inert)
- `m2c_probe_a15_loadbearing` (`XDNA_FW_FORCE_A15=<hex>`): forces a15 after the
  dispatcher's entry l32r; proves load-bearing vs red-herring.
- `m2c_probe_low_window_code` (`XDNA_FW_DUMP_ADDR=<hex>`): dumps image vs local_data
  at a low-window address (fetch/data-split check) AND prints `PS` at the syscall.
- `m2c_probe_peripheral_reads` MAX raised 40k -> 200k (runs the full natural boot).
