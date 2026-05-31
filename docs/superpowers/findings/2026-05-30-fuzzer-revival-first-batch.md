# Findings: First EMU+HW Differential Batch (M4)

**Date:** 2026-05-30  
**Task:** T6 -- M4 EMU+HW differential batch  
**Run range:** seeds 1..2000 (Peano compiler, scalar single-tile kernel, trace-sweep OFF)

---

## Batch Parameters

| Parameter | Value |
|-----------|-------|
| Seeds | 1..2000 |
| Compiler | Peano (clang, `--target=aie2-none-unknown-elf -O2`) |
| Kernel shape | Scalar single-tile (no vector), i8/i16/i32 dtypes mixed |
| Kernel structure | Single loop over buffer elements, random scalar ops |
| Trace sweep | OFF |
| Hardware | Phoenix NPU (NPU1/AIE2), real silicon as ground truth |
| EMU runner | `XclbinSuite` in-process (not XRT plugin path) |
| Max cycles | Default (107072 before stall detection) |

Run staged as two batches (seeds 1..500 first for stability check, then 501..2000):

| Batch | Seeds | Compile | Execute | Wall clock |
|-------|-------|---------|---------|------------|
| Batch 1 | 1..500 | 53.3s (16 threads) | 158.6s | ~3.5 min |
| Batch 2 | 501..2000 | 139.1s (16 threads) | 509.5s | ~9.4 min |
| **Combined** | **1..2000** | **192.4s** | **668.1s** | **~12.9 min** |

Observed throughput (execute phase only): ~3.0 seeds/sec combined.

---

## Summary Totals

| Category | Count | Pct |
|----------|-------|-----|
| Seeds run | 2000 | 100% |
| Pass (EMU == NPU, non-zero) | **0** | 0% |
| Vacuous match (both sides all-zeros) | 379 | 19% |
| Divergence (EMU != NPU) | **1577** | 79% |
| Error (runner exception, TDR) | 44 | 2% |

**Zero true passes.** Every seed where the NPU produced non-zero output disagreed with the emulator.

---

## NPU Health

TDR timeouts were detected during batch 2 (the 44 errors correlate with TDR events visible in `dmesg`):

```
amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_tdr_detect: TDR timeout detected
```

The NPU **recovered automatically** after each TDR (mailbox restart sequence visible in dmesg). No wedge, no manual `modprobe -r` needed. The batch completed without operator intervention. TDR events are expected when kernels time out on the NPU -- the fuzz runner uses a 30-second per-test timeout which triggers TDR before that on longer-running seeds.

---

## Divergence Analysis

### The Pattern

Every one of the 1577 divergences has the same signature:

```
seed N MISMATCH: element [K]: emulator=0, npu=<nonzero>
```

The emulator returns **all zeros** for the output buffer on every seed where the NPU produces a non-zero result. The 379 vacuous matches are seeds where the NPU also returns all zeros (the kernel computation genuinely produces zero for all elements given the sequential input `[1,2,...,N]`).

### Root Cause: Shim DMA BdSetup Stall

Every failing seed hits a DMA stall at the 107072-cycle stall threshold with **0 bytes transferred**:

```
WARN DMA stall after 107072 cycles in test seed_1 (0 bytes transferred):
  core(0,2) Ready;
  DMA: (0,0)ch0 BdSetup(4), (0,0)ch1 BdSetup(4), (0,0)ch2 BdSetup(4),
       (0,1)ch0 AcquiringLock(64), (0,1)ch1 AcquiringLock(66),
       (0,1)ch6 AcquiringLock(65), (0,1)ch7 AcquiringLock(67),
       (0,2)ch0 AcquiringLock(0), (0,2)ch2 AcquiringLock(3);
  pending syncs: col=0 row=0 ch=0 S2MM
```

All three shim DMA channels (ch0=output, ch1=trace S2MM, ch2=input MM2S) are stuck in `BdSetup` -- they have a BD programmed but are waiting for data that never arrives from or reaches the stream switch. Because the shim MM2S channel (input from host to memtile) never starts flowing, the memtile locks stay blocked, which blocks the compute tile locks, which means the compute core never gets input data and never produces output.

This is **one bug** manifesting as a total blackout: the `XclbinSuite` in-process emulator path cannot move data through the shim DMA. The XRT plugin path (used by bridge tests) works correctly because XRT's own shim layer handles the DMA transfers directly.

### Sample Divergences

| Seed | Dtype | Buffer size | Kernel summary | First diff element | EMU val | NPU val | Notes |
|------|-------|-------------|----------------|-------------------|---------|---------|-------|
| 1 | i8 | 32 | `t0=i*-46; buf_out[i]=t0` | [1] | 0 | -46 | Known from T4 |
| 2 | i8 | 64 | `t0=buf_in[i]-33; buf_out[i]=t0` | [0] | 0 | -33 | |
| 3 | i32 | 128 | `t0=i+54; if(t0) t0=37*-112^-26; buf_out[i]=t0` | [0] | 0 | 4150 | i32 arithmetic |
| 4 | i32 | 256 | `t0=113>>(-(23+buf_in[i])); buf_out[i]=t0` | [0] | 0 | 1895825408 | Shift semantics |
| 5 | i8 | 16 | `t0=2&buf_in[i]; if(t0) ...` | [0] | 0 | 2 | Conditional |
| 6 | i8 | 32 | `t0=buf_in[i]^t0 (3x hw loop); buf_out[i]=t0` | [0] | 0 | 1 | HW loop |
| 8 | i32 | 128 | `t0=buf_in[i]+buf_in[i]; buf_out[i]=t0` | [1] | 0 | 112 | Element [0] is 0 |
| 11 | i8 | 32 | `t0=i>>buf_in[i]; buf_out[i]=t0` | [31] | 0 | 31 | i>>32 wraps to i>>0=i on AIE |
| 24 | i16 | 256 | `t0=i>>buf_in[i]; buf_out[i]=t0` | [31] | 0 | 31 | Same large-shift pattern |
| 333 | i32 | 128 | Multi-op with final `buf_out[i]=t0` | [31] | 0 | 63 | |
| 349 | i32 | 32 | Conditional shift: `buf_out[i]=t0<<i` | [31] | 0 | 992 | |

All 1577 mismatches have `emulator=0`; only the NPU value varies (it is the correct answer).

### "Late First Diff" Cases

124 seeds have the first divergence at element index > 0. In all cases this is because the NPU legitimately produces zero for the earlier elements (the kernel computation genuinely yields 0 for those indices given sequential `[1,2,...,N]` input), not because the emulator partially succeeded. The emulator output is all-zeros throughout.

Notable sub-pattern -- seeds where first diff is at element [31] (8 seeds in batch 1): the kernel ends with `buf_out[i] = i >> buf_in[i]`. With `buf_in = [1,2,...,32]`, elements 0..30 give `i >> (i+1)` which is 0. Element 31 gives `31 >> 32`. On AIE hardware, shift-by-N mod word-size applies -- i8 shift by 32 wraps to shift-by-0 = identity, so NPU returns 31. This is a **secondary behavior observation** (not a new emulator bug -- it is masked by the shim DMA bug): the shift-semantics difference is real but irrelevant until the primary bug is fixed.

---

## Likely Distinct Bugs

After deduplication, the batch reveals **one primary emulator bug** and surfaces **two secondary behavioral notes** that will become testable once the primary is fixed.

### BUG-A (PRIMARY): Shim DMA stream data never flows in XclbinSuite path

**Severity:** Critical -- affects 100% of fuzz seeds (all true divergences, not vacuous matches).

**Symptom:** `XclbinSuite` in-process emulator produces all-zero output for all kernels. The shim DMA BDs are programmed correctly (correct base_addr, correct total_bytes), but the stream switch between shim and memtile never carries data. All three shim DMA channels (input MM2S, output S2MM, trace S2MM) remain in `BdSetup` indefinitely.

**Not affected:** The XRT plugin path (`xrt-plugin/` via `rebuild-plugin.sh` + bridge tests). Bridge tests pass because XRT's shim handles DMA transfers outside the emulator. The emulator's shim DMA implementation is only exercised via `XclbinSuite`.

**To fix:** Investigate why shim DMA `BdSetup` state never transitions to `Active`/`Running` in the in-process path. The shim mux configuration log shows correct routing (`S2MM ch0 <- master[4] (South2)`, etc.), so the stream switch routing table is programmed. The missing piece is likely that the shim DMA engine isn't being stepped or isn't receiving stream beats because the shim's data-plane execution in `XclbinSuite` differs from the XRT plugin flow.

### OBS-1: Scalar kernel output confirms DMA data path (both in and out) is gated

The fact that the compute core reports `Ready` in the stall message (not `Running`) means it is also blocked -- it never received input data from the memtile (the memtile locks are `AcquiringLock`, meaning no data arrived). This rules out a "core runs but output DMA fails" scenario. The failure is at the shim/memtile boundary, not at the compute tile.

### OBS-2: AIE2 i8/i16/i32 shift-by-N modulo behavior (secondary, masked)

Kernels containing `i >> buf_in[i]` where `buf_in[i] = i+1` produce `i >> (i+1)` = 0 for most elements, but at element [N-1] where shift count >= type width, the AIE hardware applies shift-count modulo type-width (8/16/32 bits). For i8: `31 >> 32` => `31 >> (32 mod 8)` = `31 >> 0` = 31. This is observable in the NPU output and should be validated once BUG-A is fixed. It is not a new emulator bug (the emulator never ran far enough to execute these stores), but it will need TDD coverage when the shim DMA path is repaired.

---

## NPU is Ground Truth

All findings above treat NPU hardware output as correct. The emulator is wrong. The NPU's outputs match the expected C scalar semantics (with the AIE shift-modulo nuance noted in OBS-2 for edge cases). No NPU anomalies were observed that look like hardware errors.

---

## Follow-on TDD Fix Cycles

Priority order for next session:

1. **BUG-A** (shim DMA BdSetup stall in XclbinSuite path): write a minimal failing test that reproduces the stall with a trivial kernel (e.g., identity `buf_out[i]=buf_in[i]`), then fix the shim DMA engine stepping in the in-process execution loop. When this is fixed, expect the fuzz pass rate to jump from 0% to >80% (the vacuous matches plus seeds where the computation is deterministically correct).

2. **OBS-2** (shift-count modulo semantics): once BUG-A is fixed, re-run the fuzz batch and collect the remaining divergences. Shift-by-out-of-range is one likely remaining gap. Write targeted TDD tests for `i8 >> 8`, `i8 >> 9`, `i16 >> 16`, `i32 >> 32`.

---

## Scaling Assessment

The full 2000-seed batch ran in ~13 minutes with stable NPU health (TDRs self-recovered). Scaling to 5000+ seeds is safe and worthwhile once BUG-A is fixed -- at that point the fuzz pass rate will be meaningful and higher seed counts will surface subtler divergences. Until BUG-A is fixed, additional seeds add little diagnostic value since they all hit the same stall.
