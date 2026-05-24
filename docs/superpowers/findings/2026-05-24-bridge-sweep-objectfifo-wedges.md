# Bridge Sweep: objectfifo / dynamic_object_fifo Wedge Analysis

**Date**: 2026-05-24
**Sweep**: `build/bridge-test-results/20260524`
**Scope**: Three failing bridge tests — two from `dynamic_object_fifo/` and one from
`objectfifo_repeat/` — investigated to characterize root cause.

---

## Baseline

`TMPDIR=/tmp/claude-1000 cargo test --lib` after commits c4a2aab + cccb5c8
(clock-control work): **3190 passed / 0 failed / 5 ignored**. No regression.

---

## Bridge Sweep Snapshot

Results from `build/bridge-test-results/20260524`. Tests analyzed:

| Test | Compiler | Status |
|------|----------|--------|
| `dynamic_object_fifo/nested_loops` | chess | wedge (Tier C TDR) |
| `dynamic_object_fifo/sliding_window` | chess | wedge (Tier C TDR) |
| `objectfifo_repeat/compute_repeat` | peano | wedge (Tier C TDR, false positive) |

---

## Per-Test Characterization

### dynamic_object_fifo/nested_loops and sliding_window

Both tests use the same two-tile, double-buffered objectfifo topology: shim MM2S
(col 1, row 0) streams data into compute tile (1,2) S2MM ch0, and compute tile
MM2S ch2 streams results back out. The core runs an outer×5/inner×5 loop (25
total kernel invocations) with a ping-pong lock pair on each direction.

Lock configuration (CDO init): `lock0=2` (in_cons_prod), `lock2=2` (out_prod).
Expected: 25 MM2S output transfers (one per inner iteration), 5 S2MM input
transfers (one per outer iteration).

**Terminal state** (cycle ~104,000):
```
DMA check_acquire_granted tile(1,2) ch2 bd_lock=3 target=Own(3)
  local_lock=3 lock_value=0 granted=false

Tier C wedge: DMA (1,2)ch0 Transferring(0/40),
              DMA (1,2)ch2 AcquiringLock(3)
```

The log shows:
- DMA ch2 (MM2S output): 24 `AcquiringLock→Transferring` + 25
  `Transferring→AcquiringLock` transitions. It delivered 24 transfers and is
  blocked on the 25th acquire with `lock_value=0`.
- DMA ch0 (S2MM input): stuck in `Transferring(0/40)` — a zombie 6th iteration.
  The shim-side MM2S went Idle at cycle 4204 after delivering all 50 words (5
  outer × 10 i32). The compute tile S2MM loops unconditionally, acquired a
  fresh `lock0` credit, and started a 6th 40-byte transfer — but no stream data
  remains.

**Root cause (Bucket 2)**: The compute tile S2MM over-runs the shim's data
supply by one iteration. In the final outer loop pass, the zombie S2MM acquire
and the MM2S ch2 final-iteration acquire race for lock arbiter time in the same
tile. The zombie S2MM acquires `lock0`, consuming a credit from the
`in_cons_prod_lock`, which shifts the lock-release sequencing and leaves `lock3`
(`out_cons_lock`) at 0 when the 25th MM2S transfer needs it. One `lock3` credit
is effectively absorbed by the lock arbiter's round-robin scheduling of the
zombie acquire in the overlapping cycle window.

### objectfifo_repeat/compute_repeat

Single-tile test (compute tile at 1,3), `repeat_count=4`. The `of_out`
objectfifo is configured with `depth=1`, `repeat_count=4`, producing a 4-BD
chain (bd1→bd2→bd3→bd4→bd1) with `lock2` (`out_prod_lock`) initialized to 4.
The core holds the full 4 credits at start, does one acquire-by-4 to grab them
all, then processes 4096 elements in an inner copy loop before releasing.

Lock timeline:
- Cycle 6464: S2MM releases `lock1` (in_cons_cons_lock)
- Cycle 6465: core successfully acquires `lock1` (AcquireGe 1)
- Cycle 106465: Tier C TDR fires, core is `Ready`

**Terminal state**:
```
Tier C wedge: core(1,3) Ready; DMA (1,3)ch0 AcquiringLock(0);
              DMA (1,3)ch2 AcquiringLock(3)
error at index[0]: expected 1 got 0
```

The 4096-element inner copy loop takes approximately 4096 × ~24 cycles ≈ 98,304
cycles. The stall detector threshold is 100,000 cycles. The core is executing
instructions continuously — no DMA, no lock activity — and the stall detector
fires 100,000 cycles after the last progress event (lock1 acquire at cycle 6465).

**Root cause (Bucket 1)**: Stall detector false positive. `StallDetector::check`
in `src/device/tdr/detector.rs` tracks progress via `total_dma_bytes_transferred`
and `total_lock_releases` only. Core instruction execution is invisible. A
100,000-cycle pure-compute inner loop is legitimate compute progress, but the
stall detector cannot distinguish it from a genuine stall.

---

## Hypothesis

**Bucket 1 — false positive** (compute_repeat): The stall detector's progress
metric is too narrow. It must be extended to also track core instruction
execution (total instructions retired, or per-tile PC change). With that change,
compute_repeat should pass — the core completes ~98,304 cycles of inner loop
work well within any reasonable threshold.

**Bucket 2 — zombie DMA over-run** (nested_loops, sliding_window): The compute
tile's S2MM DMA channel is not stopped when the shim-side stream is exhausted.
It runs one extra iteration, competing with the still-active MM2S channel for
the lock arbiter in the final outer loop. The off-by-one in lock credit
consumption wedges the MM2S output path.

---

## Cross-Reference with Upstream

**Bucket 1**: The mlir-aie lowering of `repeat_count=4` (`of_out`) sets
`lock2` init=4 and the core acquires/releases by 4 (confirmed in
`input_with_addresses.mlir`). This is correct; the test is architecturally
valid. The stall detector in `detector.rs` is the sole point of failure.

**Bucket 2**: The zombie S2MM iteration is consistent with how aie-rt programs
compute-tile DMA: BD chains are circular and loop until externally stopped
(`XAIE_ENABLE` → `XAIE_DISABLE` sequence in `xaie_dma_aieml.c`). The mlir-aie
lowering does not generate an explicit DMA-stop after the final outer iteration;
the shim-side stream termination is implicitly expected to block the tile-side
DMA via stream credits. The emulator does not yet implement stream-credit
back-pressure that would stall the S2MM acquire, so the zombie acquire reaches
the lock arbiter.

---

## Next-Step Recommendations

1. **Bucket 1 (immediate)**: Add `total_instructions_executed` (or equivalent
   PC-change tracking per active core) to the stall detector's progress test.
   A single additional counter in the array-level stats suffices; the check
   in `StallDetector::check` gains one more `||` clause.

2. **Bucket 2 (follow-on)**: Determine whether the correct fix is (a) implement
   stream-credit back-pressure so S2MM stalls on empty stream rather than
   acquiring lock, or (b) detect the zombie iteration earlier and suppress
   the lock acquire. Option (a) is the hardware-accurate path (streams have
   finite credit windows); option (b) is a workaround. Check aie-rt
   `xaie_dma_aieml.c` for how `_XAieMl_DmaWaitForDone` detects a stalled
   S2MM to confirm which behavior the silicon exhibits.
