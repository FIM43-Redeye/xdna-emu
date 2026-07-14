# The AIE2 tile DMA reads memory 128 bits at a time -- one bank access per four stream beats

Date: 2026-07-14
Branch: `feat/core-memory-stall-model`
Hardware: real Phoenix NPU1 (AIE2), 12 runs (4 variants x 3 repeats), all clean
Capture: `build/experiments/bankwidth/{idle,apart,collide,collide2}/`
Generator: `tools/experiments/bankdisc.py`; analysis
`tools/experiments/bankdisc_{measure,analyze}.py`

## The question

The emulator's DMA declares a memory-bank demand **every cycle** while streaming
(`words_per_cycle_for`, `src/device/dma/engine/stepping.rs:1357` -- 1 word = 4 B
per cycle). Two candidate models for what silicon does:

- **Model A (buffered / 128-bit):** the DMA gathers 4 x 32-bit stream beats and
  performs one 128-bit bank access -- it occupies a physical bank slot **1 cycle
  in 4**.
- **Model B (per-beat / 32-bit):** each beat takes a full bank arbitration slot
  -- the DMA occupies a bank slot **every cycle**.

AM020 does not state the compute-tile DMA's memory-side port width (ch.5:105
states it only for the *memtile*). This was the leading hypothesis for why the
bank-arbitration model charged the producer ~100x the stall cycles silicon does,
and (as it turned out wrongly -- see below) for an apparent consumer
over-production that was really a trace-coverage artifact
([2026-07-13-memory-stall-bank-arbitration.md](2026-07-13-memory-stall-bank-arbitration.md)).
It was measured, not guessed and not fitted.

## Answer

**Model A. The tile DMA performs one 128-bit (16-byte) bank access per four
32-bit stream beats.** The measured width is **16.0 bytes / 4.00 stream beats**,
against Model B's 4 bytes. Two independent observables agree, and the
pre-registered validity gate passes.

## Method

A compute tile (0,2) whose MM2S DMA drains a fixed 256-word (1 KB) buffer to the
shim while the core hammers a separate 8 KB scratch buffer. **The only thing that
changes between variants is which logical bank each buffer lives in.** Physical
banks interleave every 16 B inside a logical bank, so a multi-word buffer
necessarily straddles both physical banks of its pair; the experiment therefore
works at logical-bank granularity (`logical = (addr >> 14) & 3`,
`physical = 2*logical + ((addr >> 4) & 1)`).

| variant | hammer_buf | dma_buf | core hammers? | overlap |
|---|---|---|---|---|
| `idle` | 0x0400 (logical 0) | 0x2400 (logical 0) | no | uncontended floor |
| `apart` | 0x0400 (logical 0) | 0x8000 (logical **2**) | yes | none |
| `collide` | 0x0400 (logical 0) | 0x2400 (logical 0) | yes | **same bank** |
| `collide2` | 0x8000 (logical **2**) | 0xA000 (logical **2**) | yes | same bank, relocated |

Addresses were pinned with `aie.buffer {address = ...}` and **confirmed in the
final `input_with_addresses.mlir` of every build before any hardware run**. (The
bank-aware allocator independently annotated `mem_bank = 0/2` matching the
derivation. Buffers must be declared in ascending address order or its per-bank
cursor rejects them.)

The core body is **byte-for-byte identical** across `apart`/`collide`/`collide2`;
`idle` differs only by omitting the hammer loop. Single-buffered by design: the
lock forbids the core from touching `dma_buf` while the DMA drains it, so during
every measured transfer the core's only memory traffic is the hammer.

**Bracket.** `DMA_MM2S_0_START_TASK` / `FINISHED_TASK` are **unusable here**: a
self-looping `aie.mem` BD chain is a *single* task, enqueued at CDO time before
the trace window opens, so both fire zero times. Used instead:

- `DMA_MM2S_0_FINISHED_BD` (event 25) -- fires per BD, i.e. **16 times, once per
  transfer**. Discrete; its `ts` is the transfer end.
- `DMA_MM2S_0_STALLED_LOCK` (event 33) -- level, asserted while the MM2S waits on
  `lk_full`. Its **falling edge is the transfer start**.

Level events are integrated as **interval area** from the mode-0 B/E rebuild,
never as record counts; timestamps are `ts`, never `soc` (both traps per the
prior finding).

## Results

Median over 3 HW runs x 15 bracketed transfers each. Run-to-run variation was nil
(every repeat produced identical medians and conflict areas).

| variant | T (median) | T (mean) | conflict cycles | core MEMORY_STALL | MM2S starvation |
|---|---:|---:|---:|---:|---:|
| `idle` | **244** | 244.00 | 7 | 1 | 0 |
| `apart` | **244** | 244.00 | **0** | 0 | 0 |
| `collide` | **244** | 244.33 | 144 | 32 | 0 |
| `collide2` | **244** | 244.40 | 149 | 34 | 0 |

- **Validity gate: PASS.** `T_APART / T_IDLE = 1.0000`. The core cannot slow a DMA
  it shares no bank with, so nothing outside bank contention couples them.
- **`T_COLLIDE / T_APART = 1.0000`.** Model A predicted 1.0-1.3; Model B predicted
  2-4. `T_COLLIDE2 / T_APART = 1.0000` replicates it in a different bank pair.

### Every conflict is core-vs-DMA, and they all land inside the transfer

`apart` records **zero** bank conflicts on every traced bank -- the identical
hammer loop, running alone, self-conflicts not at all (its two loads are 16 B
apart, i.e. sibling physical banks, and load/store use separate ports). So all 144
conflict cycles in `collide` require the DMA. And **100% of conflict cycles fall
inside the measured transfer windows** (144/144, 149/149).

The one confound that could have faked this -- a stream-limited DMA with memory
slack for the wrong reason -- is dead: `DMA_MM2S_0_STREAM_BACKPRESSURE` **inside**
the transfer windows is 30 / 28 / 30 / 30 cycles out of ~3660, *identical in all
four variants*. The transfer is not stream-limited, and whatever backpressure
exists is a constant that cannot explain a difference.

### The decisive fact: the DMA loses arbitrations and does not slow down

| variant | contended cycles | core denied | **DMA denied** | DMA extra cycles, all 16 transfers |
|---|---:|---:|---:|---:|
| `collide` | 144 | 32 | **112** | **5.0** |
| `collide2` | 149 | 34 | **115** | **6.0** |

The DMA lost **112 bank arbitrations** and paid **5 cycles** for them. This also
rules out the alternative explanation that the DMA simply *wins* arbitration by
priority (in which case duration would be flat under both models): it demonstrably
loses, 78% of the time, and absorbs it. A requester that can eat ~7 lost bank
cycles per 244-cycle transfer at zero throughput cost has memory-side slack --
it is not claiming a bank every cycle.

### Inverting the conflict area for the width

The conflict area is the observable that is *directly* sensitive to how often the
DMA claims a bank. Under round-robin single-port banks, for uncorrelated request
phases:

```
conflict_cycles = SUM_b  P(core wants b) * P(DMA wants b) * T
```

`f_core` is measured, not assumed. From `llvm-objdump` of the core ELF, the
hammer is a zero-overhead loop `ZLS_Fcore_0_2_304` (PC 0x210..0x270, `lc = 0x600`
= 1536 iterations) of **10 bundles with 3 memory ops** (`lda [p0,#16]`,
`lda [p0],#4`, `st [p1,#0]`), giving 1.5 accesses per physical bank per iteration
-> **f_core = 1.5/10 = 0.15**. That 10-cycle figure is confirmed independently by
the measured rep period: predicted 256 (fill, 1 bundle/iter) + 1536x10 (hammer) +
~40 (lock/setup) = 15656; **measured 15661-15664**.

Solving for the DMA's duty cycle with the measured `conflict = 9.6` cycles per
transfer and `T = 244`:

```
d_dma (both banks) = 0.262
bank accesses per transfer = 64.0
bytes per bank access = 1024 / 64.0 = 16.0 B  =  4.00 stream beats
```

`collide2` gives 15.5 B / 3.87 beats. Forward-predicting conflict cycles per
transfer from each model instead:

| | conflict cycles / transfer |
|---|---:|
| Model A (128-bit, 16 B) predicts | 8.8 |
| Model B (32-bit, 4 B) predicts | 35.3 |
| **OBSERVED `collide`** | **9.6** |
| **OBSERVED `collide2`** | **9.9** |

Model A is right to within 9-13%. Model B is off by **3.7x**.

## Two further findings, both new

### 1. The physical bank mapping is confirmed -- banks 4/5 traced for the first time

`physical = 2*logical + ((addr >> 4) & 1)` was Sol's inference and had never been
tested; the prior capture only ever traced banks 0-3, so "banks 4-7 = 0" was
*unmeasured*, not observed.

| variant | dma_buf logical bank | BANK_0 | BANK_1 | BANK_4 | BANK_5 |
|---|---|---:|---:|---:|---:|
| `collide` | 0 | 81 | 63 | **0** | **0** |
| `collide2` | **2** | **0** | **0** | 84 | 65 |

Moving the buffers from logical bank 0 to logical bank 2 moves the conflicts from
physical banks 0/1 to physical banks **4/5**, exactly and exclusively. The
derivation is correct.

### 2. Silicon's DMA memory-pressure events do NOT fire on bank-arbitration loss

`DMA_MM2S_0_MEMORY_STARVATION` is **0 in every variant** -- including `collide`,
where the MM2S lost **112 bank arbitrations**. On silicon a lost arbitration
produces *zero* starvation.

The emulator emits `MM2S_0_MEMORY_STARVATION` precisely on arbitration loss (40 /
53 / 128 cycles in the Task 8 table, against HW's 0). That is now explained: the
emission is semantically wrong, not merely miscalibrated. It is the same defect
the prior finding flagged from the other side -- HW's `S2MM_0_MEMORY_BACKPRESSURE`
asserts for 756 cycles when only 232 cycles had *any* bank contention, so it too
cannot be an arbitration signal. Both events track the DMA's *data-path* inability
to move data, of which arbitration loss is not a cause (it has slack to absorb it).

**Not answered here:** what `S2MM_0_MEMORY_BACKPRESSURE` *does* track. This design
has no S2MM channel on the compute tile, so the event is trivially 0 and the
capture says nothing about it. Open follow-up.

## What this means for the emulator (decision belongs to Maya -- nothing changed)

The fix is now specified by measurement rather than by guess: model the tile DMA's
bank access as **one 16-byte burst per 4 stream beats**, i.e. a bank demand on 1
cycle in 4 rather than every cycle. Two consequences follow from the data above
and should be expected:

- The core's DMA-caused stall count should fall by roughly the same ~4x factor in
  the DMA's collision density.
- `MM2S_0_MEMORY_STARVATION` should stop being emitted on arbitration loss
  entirely; that is a semantic change, independent of the cadence change.

This is a DMA-timing change and it moves DMA timing everywhere else in the
emulator, so it stays a separate scoped decision. Gap registry row:
`docs/fidelity-gaps/core-compute-timing.md`.

## Outcome: the fix landed, and it moved the producer, not the consumers

Stage 1 landed the granule model (`BankLayout::access_granule_bytes`,
`word_opens_granule` -- the DMA claims a bank only on the word that opens a new
16-byte granule, and the disproven `denied_dma -> MEMORY_STARVATION/BACKPRESSURE`
wiring is gone). Re-running `of_q0_rich_bank_arbitration_vs_hw`:

| Tile | MEMORY_STALL before | after | HW |
|---|---:|---:|---:|
| Producer | 102 | **4** | 1 |
| ConsA | 425 | **391** | 220 |
| ConsB | 440 | **394** | 245 |

The producer -- whose stalls were almost entirely DMA-caused -- collapsed by 25x
and now sits within 3 cycles of silicon. The consumers barely moved, and the
census says exactly why: of ConsA's 406 contended cycles, only **29** involve a
bank a DMA channel demanded at all. **377 are the core conflicting with ITSELF**
-- two of its own memory ports landing in one physical bank in the same cycle.
The DMA's collision density was never what the consumers' stall count was made
of, so no DMA-side change could have closed it. The prediction that the 4x
cadence change would drag 425 toward 220 was wrong, and it was wrong for an
informative reason.

**There is no residual.** The apparent 1.8x that this section originally called "a
CORE-port question" was a **trace-coverage artifact, not a model error**, and the
claim is retracted: the `of_q0_rich` HW capture holds ~9 of the kernel's 16
pipeline iterations (Q=0, so the cores free-run from CDO time before the host arms
the trace unit), while the emulator's census counts all 16. `16/9 = 1.78 = 391/220`,
and ConsB confirms it at `16/10 = 1.60 = 394/245`. Per rep the two sides are
identical -- 24 stall cycles in a 47-cycle burst, gap 2, banks 0/1. The core's
load-vs-store self-conflict is the real mechanism and it is HW-confirmed. Proof:
[`.superpowers/sdd/coreports-report.md`](../../../.superpowers/sdd/coreports-report.md).
The stall SHAPE also matches silicon exactly: 381/386 singleton runs at a dominant
gap of 2.

## Trap for anyone re-running this

`DMA_MM2S_0_START_TASK` / `FINISHED_TASK` fire **zero** times for any
objectfifo-style or hand-written `aie.mem` self-looping BD chain: the chain is one
task, enqueued at configuration time, and it never terminates. Use `FINISHED_BD` +
`STALLED_LOCK`. The brief for this experiment specified the task events; they do
not work.
