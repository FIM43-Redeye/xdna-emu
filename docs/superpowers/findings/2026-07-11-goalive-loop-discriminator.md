# Go-alive loop discriminator

**Date:** 2026-07-11  
**Target:** Phoenix/NPU1 `1502_00/npu.dev.sbin`  
**Image SHA-256:** `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`

## Verdict: a premise is wrong

The repeated `0x55f8` dispatch is a **mapping artifact**, not leg 1, leg 2, or
leg 3. The installed AT queue-pop overlay ends at `0xccb4`, but the executed
empty-queue branch targets `0xccb3` and its coherent AT tail continues through
`0xccc0`. The current model therefore executes an instruction assembled from
one AT byte and two BASE bytes. That mixed instruction fails to clear the
already-consumed output descriptor's valid bit, so the MERT worker calls the
stale `0x55f8` descriptor again.

A test-local overlay of only `[0xccb4,0xccc1)` executes the coherent empty tail,
clears the valid bit, and reduces `0x55f8` from repeated entries to exactly one.
No production mapping or scheduler code was changed.

A second premise is also wrong: firmware **does execute its alive publisher**
on the first `0x55f8` dispatch. It byte-stores `0x030bb000` to firmware VA
`0..3`. The emulator maps that destination to local PA `0..3`; real hardware
exposes the identical value in `FW_ALIVE_OFF` (`0x030bf000`). The missing piece
is PSP/load-time destination mapping, not a branch that withholds publication.

## A. Queue ownership

The fixed-pool queue is real and it retires the item:

- `n=47362 pc=0xd6e6`: `[0x2320] <- 0x55f8`; the persistent record is
  `0x2320..0x2334`.
- `n=47383 pc=0xd785`: pool count `[0x24c4]` changes `0 -> 1`; cursor
  `[0x24c5]` is `0`.
- `n=49686 pc=0xcc25`: the pop reads count `1`.
- `n=49701..49708 pc=0xcc50..0xcc60`: it copies record words
  `0x55f8, 0xff, 0, 0` from `0x2320..0x232f` to output descriptor
  `0x15fc0..0x15fcf`.
- `n=49721 pc=0xcc85`: it sets output valid byte `[0x15fcb] <- 1`.
- `n=49734 pc=0xcca9`: it consumes count `1`; `n=49736 pc=0xccae` stores
  `[0x24c4] <- 0`.

Thus leg 1's claimed non-retirement is false. The pool is empty before the
first run-function entry.

The three entry snapshots are byte-identical:

| Entry | `n` | predecessor | current task `[0x2278]` | count/cursor/aux | output descriptor |
|---:|---:|---:|---:|---:|---|
| 1 | `49925` | `0x08b041dc` | `0x10dfc` | `00/00/00` | `0x55f8,0xff,0x01040000,0,0` |
| 2 | `53336` | `0x08b041dc` | `0x10dfc` | `00/00/00` | same |
| 3 | `56694` | `0x08b041dc` | `0x10dfc` | `00/00/00` | same |

Current-task ownership changes only twice: `0 -> 0x10f10` at
`n=41463 pc=0x46ac`, then `0x10f10 -> 0x10dfc` at
`n=47985 pc=0x285d`; it remains `0x10dfc` across all three entries. This is the
MERT worker, whose function pointer is `[0x10dfc+0x20] = 0x08b041bc`.

The actual re-presentation happens after the queue is already empty:

- `n=53141 pc=0xcc25`: count load returns `0`.
- `n=53144 pc=0xcc2e`: `beqz a6,0xccb3` is taken.
- The installed overlay is `[0xcc1c,0xccb4)`. At `0xccb3`, live fetch combines
  AT byte `0x32` with BASE bytes `0x00,0x00`, producing the executed
  `l8ui a3,[a0]` at `n=53145`, not the coherent AT
  `l8ui a3,[a2+11]` (`32 02 0b`). It then executes BASE
  `movi.n a2,9; wsr.ps a3; retw.n` through `n=53148`.
- Because `[0x15fcb]` remains `1`, `n=53149 pc=0xc660` reloads `1`, and the
  `bbsi a6,0,0xc672` edge reaccepts the stale descriptor. At
  `n=53333 pc=0x08b041d7`, the worker reloads `[0x15fc0] = 0x55f8` and dispatch
  2 follows.

The local falsifier completes only `[0xccb4,0xccc1)` as AT. On the first empty
poll, `n=53148 pc=0xccbc` executes `s8i a3,[a2+11]`, storing
`[0x15fcb] 1 -> 0`. By `n=53659`, count is `0`, valid is `0`, and the total
`0x55f8` entry count is **one**. The arm then reaches the separately known
`0x8cb1` framing collision; that later wall does not weaken the stale-item
discriminator.

## B. Loop inputs

The probe prints every executed load between entries 1 and 3 as
`n/pc/EA/value/class/op`, plus every conditional edge and its consumed register
values. The only fixed external word in the `0x563e/0x5640` cluster is:

```text
n=52206 pc=0x563e L32iN EA=0x27010ac0 value=0
n=52207 pc=0x5640 BeqzN a2=0 -> taken to 0x5645
n=55622 pc=0x563e L32iN EA=0x27010ac0 value=0
n=55623 pc=0x5640 BeqzN a2=0 -> taken to 0x5645
```

`0x27010ac0` is in the modeled `0x27000000..0x28000000` mailbox/MMIO
aperture. A nonzero value would fall through to `waiti` at `0x5642`; zero skips
it. This is a real external-status gate, but it does **not** cause the repeated
dispatch:

1. the alive publisher has already executed before this load;
2. the queue-tail falsifier leaves this MMIO value unchanged at zero yet
   eliminates every later `0x55f8` entry.

The value that directly gates stale redispatch is local `[0x15fcb] = 1`, loaded
at `pc=0xc660` and tested by `pc=0xc666`. It should be cleared by the correctly
framed empty-queue tail. Therefore leg 3 is not the loop's cause, and no host,
DMA, or doorbell stimulus is the next fix for this recurrence.

## C. Alive publication

`0x55f8` is a coherent go-alive dispatch, not a lumped trace label. On the first
entry, the executed chain is:

```text
n=51765 pc=0x560d  Call8 0x5044
n=51766 pc=0x5044  Entry
n=52115 pc=0x50ba  L32r a2,[0x31bc] -> a2=0
n=52119 pc=0x50c6  S8i EA=0 value=0x00
n=52120 pc=0x50c9  S8i EA=3 value=0x03
n=52121 pc=0x50cc  S8i EA=2 value=0x0b
n=52122 pc=0x50cf  S8i EA=1 value=0xb0
```

The four stores compose little-endian `0x030bb000`. `0x560d -> 0x5044` is
unconditional, and the executed publisher precedes the later
`0x27010ac0`/`0x5640` status branch. The tracked symbol `0x50e8` is not an
instruction boundary in the coherent AT view; the executed publisher entry is
`0x5044`.

The driver contract still places `FW_ALIVE_OFF` at device address
`0x030bf000` (BAR2 offset `0x3f000`). A safe kernel-side hardware capture saw
that slot change to exactly `0x030bb000` at 72.8 ms. The raw image contains no
literal `0x030bf000` or `0x030bb000`; the destination literal at VMA `0x31bc`
is zero in both raw views. The evidence therefore proves an incomplete
PSP/load-time destination model, but not its exact silicon mechanism. The two
remaining possibilities are a PSP-provided DTLB mapping for VA zero or a
load-time relocation/patch of the destination literal.

`local_data[0x14820] == 0x55504e5f` remains valid evidence that the channel
descriptor was built locally. It is not the host's alive observation by
itself; the executed VA-zero publication plus the missing PSP mapping explains
why the emulator's explicit read of `0x030bf000` remains zero.

## Reproduction

```text
XDNA_FW_PROBE=1 XDNA_FW_MAX=100000 cargo test --lib \
  m2c_probe_goalive_loop_discriminator -- --nocapture
```

The probe is in `src/firmware/boot_tests/coherence_mapper.rs`. It leaves the
baseline arm untouched and applies `[0xccb4,0xccc1)` only to its local
falsifier arm. Production `load_m2c`, scheduler state, interrupts, and firmware
memory are not modified.
