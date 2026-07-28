# The DMA memory-pressure events are FIFO-occupancy signals, not arbitration signals

Date: 2026-07-14
Branch: `feat/core-memory-stall-model`
Hardware: real Phoenix NPU1 (AIE2), 3 runs, all clean
Kernel: the **unchanged reference** `of_q0_rich` (`spike_q0_trace/aie.mlir`) -- the
same design that produced `build/experiments/memory-stall-bankcap/`, so every
number below is directly comparable to the 756 / 853 already on record.
Capture: `build/experiments/s2mm-backpressure-semantics/`

## The question

The emulator asserts `S2MM_0_MEMORY_BACKPRESSURE` and `MM2S_0_MEMORY_STARVATION`
on **bank-arbitration loss**. The prior finding
([2026-07-14-dma-bank-access-width.md](2026-07-14-dma-bank-access-width.md)) proved
that is wrong on silicon: the DMA lost 112 bank arbitrations and `MM2S_0_MEMORY_STARVATION`
stayed at **0**. Meanwhile HW asserts `S2MM_0_MEMORY_BACKPRESSURE` for 756 / 853
cycles on the consumers of `of_q0_rich`, while only ~232 / 255 cycles had *any* bank
contention at all. The wiring is being removed; we need to know what to replace it
with before we re-wire.

## Answer

**Both events are S2MM/MM2S FIFO-occupancy signals. On this kernel
`S2MM_0_MEMORY_BACKPRESSURE` is driven entirely by the LOCK, not by the bank.**

The hypothesis was:

> `S2MM_MEMORY_BACKPRESSURE` = the S2MM ingress FIFO is full and cannot drain to
> memory -- and on the reference kernel the thing that stops it draining for
> hundreds of cycles is the consumer core not having released the buffer.

**Confirmed, and to the cycle.**

## Evidence

Interval area (cycles asserted), median over 3 runs. Tiles are HW-shifted (col0 -> col1).

| tile | S2MM_0_MEMORY_BACKPRESSURE | S2MM_0_STALLED_LOCK | contended (BANK_0+1) | S2MM_0_STREAM_STARVATION |
|---|---:|---:|---:|---:|
| ConsA (1,3) | **757** | 2007 | 205 | 7372 |
| ConsB (2,3) | **853** | 2038 | 213 | 7318 |

`MEMORY_STALL` is 220 / 244 -- identical to the reference capture, confirming this
is the same kernel. Backpressure reproduces at 757 / 853 against the recorded
756 / 853.

### Cycle-level overlap

| | ConsA | ConsB |
|---|---:|---:|
| BACKPRESSURE cycles **inside** a lock-stall window | **99.1%** (750/757) | **99.1%** (845/853) |
| BACKPRESSURE cycles outside any lock stall | **7** | **8** |
| BACKPRESSURE ∩ STREAM_STARVATION | **0** | **0** |
| BACKPRESSURE explained by bank contention | 23% (coincident, not causal) | 23% |

Backpressure is a near-perfect **subset** of the lock stall (it never asserts
without one), but only 37-41% of it -- so the lock stall is *necessary but not
sufficient*. The missing ingredient is the FIFO filling up, and it is visible
directly.

### The smoking gun: per-lock-window structure

Every steady-state lock-stall window on both consumers has the identical shape:

```
lock window (536524, 537698) len 1174  BP 125 cycles, starts +1049   <- cold start
lock window (537719, 537838) len  119  BP 104 cycles, starts   +15
lock window (537859, 537978) len  119  BP 104 cycles, starts   +15
lock window (537999, 538118) len  119  BP 104 cycles, starts   +15
...
lock window (538560, 538678) len  118  BP   0 cycles            <- stream drained, FIFO never fills
```

Across **39 steady-state windows** (2 tiles x 3 runs), the delay from lock-stall
start to backpressure assert is **15 cycles, with zero variance** -- every window,
every run. And backpressure deasserts **exactly 1 cycle after** the lock stall ends,
also with zero variance.

```
BACKPRESSURE = [ lock_stall_start + 15 , lock_stall_end + 1 )
```

That is precisely the FIFO story: the BD cannot write to memory (lock not yet
released), the ingress FIFO accepts stream beats for **15 cycles until it is full**,
backpressure asserts and stays asserted until the lock frees the BD to drain.

The offset scan is *saturated* rather than peaked (shifting backpressure earlier by
any k <= -1 leaves overlap flat at 757, and it degrades monotonically for k > 0),
which is the expected signature of a strict subset whose *end* is pinned to the lock
window's end. This is a containment relationship, not the -1 causal-edge
relationship that MEMORY_STALL/CONFLICT showed; the offset scan is the wrong
instrument here and the per-window structure above is the right one.

### The cross-check passes exactly

`BACKPRESSURE ∩ STREAM_STARVATION = 0`, **exactly zero**, on both tiles in all
three runs. An empty FIFO cannot backpressure, and a starving stream means an empty
FIFO. The two events are perfectly mutually exclusive -- which is only true if both
are reading the same FIFO's occupancy. That is strong independent confirmation that
we are reading the right signal.

(`STREAM_STARVATION` is huge -- 7372 / 7318 cycles -- because the upstream memtile
only feeds these consumers in bursts; the channel spends most of its life with an
empty ingress FIFO. `MM2S_0_MEMORY_STARVATION` is **0**, as predicted: with 3 spare
bank cycles in 4, the MM2S output FIFO never runs dry.)

### Arbitration is not the cause

Total bank-contended cycles are 205 / 213, against 757 / 853 cycles of backpressure
-- backpressure exceeds *all* contention by 3.7x, so arbitration cannot be its
cause even in principle. The 23% coincidence is exactly that: conflicts happen to
fall inside backpressure windows because both are common, not because one drives the
other. Removing the emulator's arbitration wiring loses nothing real.

## Recommendation for the emulator

**`S2MM_n_MEMORY_BACKPRESSURE`** -- assert when the channel's **ingress FIFO is
full**: stream beats are arriving but the BD cannot write them to memory. Do not
assert it on bank-arbitration loss. The observed silicon behaviour to reproduce:

- The dominant (here, sole) reason the BD cannot write is a **lock wait**. Whenever
  the S2MM is lock-stalled *and* the upstream stream is delivering, the FIFO fills in
  **15 cycles** and backpressure asserts for the remainder of the stall, deasserting
  1 cycle after the lock is acquired.
- Model it as real FIFO occupancy rather than special-casing the lock, and the
  15-cycle fill and the cold-start / stream-drained windows (where it correctly does
  *not* assert) fall out for free. **The measured ingress FIFO depth is ~15-16
  beats** at 1 beat/cycle.
- Bank-arbitration loss *can* in principle contribute, but empirically does not:
  with the 4-beats-per-bank-access slack the DMA absorbs denials without the FIFO
  ever backing up. A faithful FIFO model will reproduce that automatically.

**`MM2S_n_MEMORY_STARVATION`** -- assert when the **output FIFO is empty while the
stream wants a beat**. On silicon this is **0** on every workload we have measured,
including one where the MM2S lost 112 bank arbitrations. Do not assert it on
arbitration loss. A faithful FIFO model will emit 0 here, which is the correct
answer.

The two events are duals reading the same FIFO from opposite ends, and the
`BACKPRESSURE ∩ STREAM_STARVATION = 0` result says the hardware treats them that
way. Model the FIFO once and both events fall out.

## Provenance

Follows directly from [2026-07-14-dma-bank-access-width.md](2026-07-14-dma-bank-access-width.md),
which established the memory-side slack that makes the arbitration explanation
impossible. Gap registry: `docs/fidelity-gaps/core-compute-timing.md`.
