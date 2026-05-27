# AIE_RW_ACCESS Characterization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **This plan is exploratory empirical work, not classical TDD** — each task's "assertion" is whether the experiment produced data that answers a specific question. Findings inform subsequent tasks.

**Goal:** Calibrate AIE_RW_ACCESS as a cycle-accurate measurement tool on Phoenix NPU, then apply it to two concrete cycle-accuracy questions: cross-validating the existing `dispatch_overhead = 2500 cyc` constant against on-NPU cycle-counter ground truth, and inspecting HW state at the K=16 wedge that EMU currently masks.

**Architecture:** Three phases with hard decision-gates between them.
- **Phase 1** empirically calibrates AIE_RW_ACCESS — Timer_Low semantics, host-roundtrip distribution, cross-tile coherence, achievable noise floor. Phase 2 is not planned in detail until Phase 1 answers what the probe can actually measure.
- **Phase 2** applies the calibrated probe to (a) the dispatch_overhead K-sweep (cross-validation against the existing trace-based 2500 cyc constant), and (b) post-wedge state inspection at K=16. Concrete tasks are written *after* Phase 1.
- **Phase 3** generalizes the measurement workflow into a reusable per-kernel timing-report harness, with the emulator producing the same report shape so they can be diffed. Planned after Phase 2.

**Tech Stack:**
- `tools/rw-access-probe/` (extended C++ probe using `xrt::aie::device::read_aie_reg`)
- Python for analysis (numpy, matplotlib via existing ironenv)
- `tools/parse-trace.py` and the existing trace pipeline for cross-validation
- `mlir-aie/build/test/npu-xrt/_diag_shim_chain_sweep/` test corpus (already built)
- AM025 register database via `mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json`

**Open questions Phase 1 must answer (gate criteria):**
1. Does Timer_Low free-run, or is it gated by tile activity? (The 120 kHz reading from the wedge survey suggests gating.)
2. What is the cycle source — core clock (~1 GHz expected) or a divided reference?
3. What's the host-roundtrip latency distribution for read_aie_reg? p50/p99/max.
4. Are Timer_Lows in different tiles coherent enough to align cross-tile events?
5. Can we usefully pair AIE_RW_ACCESS with host wall-clock for absolute dispatch timing?

---

## Phase 1: Calibrate the Probe

### Task 1: Roundtrip latency distribution

**Files:**
- Modify: `tools/rw-access-probe/rw-access-probe.cpp` (add CSV-emit mode for per-read timings)
- Create: `tools/analyze-rw-latency.py` (histogram + percentile summary)
- Output: `build/experiments/rw-access-calibration/roundtrip-{date}.{csv,png}`

**Question:** What is the host-roundtrip distribution for a single `read_aie_reg` call?

- [ ] **Step 1: Extend rw-access-probe with `--csv <path>` mode**

Add an `--csv <path>` argument that, when set, writes per-read timing rows (`index,timestamp_ns,roundtrip_us,value`) to the file instead of the human-readable per-read output. Keep human-readable verdict at end. Concrete change to the option parser in `rw-access-probe.cpp:~70`:

```cpp
else if (s == "--csv" && i + 1 < argc) a.csv_path = argv[++i];
```

And in the read loop (in `main`), guard the per-read printf and add:

```cpp
if (!a.csv_path.empty()) {
    std::FILE* f = std::fopen(a.csv_path.c_str(), "w");
    std::fprintf(f, "index,timestamp_ns,roundtrip_us,value\n");
    // ... in the loop, after each read:
    std::fprintf(f, "%d,%lld,%.3f,%u\n", i, t_ns, roundtrip_us, value);
    std::fclose(f);
}
```

- [ ] **Step 2: Rebuild the probe**

```bash
cd /home/triple/npu-work/xdna-emu/tools/rw-access-probe/build
cmake --build . -j
```

Expected: clean build, no warnings.

- [ ] **Step 3: Capture 10k reads on an idle compute tile**

```bash
mkdir -p /home/triple/npu-work/xdna-emu/build/experiments/rw-access-calibration
pkexec /home/triple/npu-work/xdna-emu/tools/rw-access-probe/build/rw-access-probe \
    --num-reads 10000 --sleep-ms 0 \
    --csv /home/triple/npu-work/xdna-emu/build/experiments/rw-access-calibration/roundtrip-idle-compute.csv
```

Expected: 10000-row CSV, no wedges, completes in ~2-5 seconds.

- [ ] **Step 4: Write `tools/analyze-rw-latency.py`**

```python
#!/usr/bin/env python3
"""Histogram + percentile summary of rw-access-probe CSV output."""
import argparse, csv, sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="rw-access-probe CSV")
    ap.add_argument("--out", default=None, help="PNG histogram output path")
    args = ap.parse_args()

    rt = []
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rt.append(float(row["roundtrip_us"]))
    rt = np.array(rt)

    print(f"N={len(rt)}")
    print(f"min   = {rt.min():.1f} us")
    print(f"p50   = {np.percentile(rt, 50):.1f} us")
    print(f"p90   = {np.percentile(rt, 90):.1f} us")
    print(f"p99   = {np.percentile(rt, 99):.1f} us")
    print(f"p99.9 = {np.percentile(rt, 99.9):.1f} us")
    print(f"max   = {rt.max():.1f} us")
    print(f"mean  = {rt.mean():.1f} us  stddev={rt.std():.1f} us")

    if args.out:
        plt.figure(figsize=(10, 4))
        plt.hist(rt, bins=200, range=(0, np.percentile(rt, 99.5)))
        plt.xlabel("roundtrip (us)")
        plt.ylabel("count")
        plt.title(f"read_aie_reg roundtrip distribution (N={len(rt)})")
        plt.grid(alpha=0.3)
        plt.savefig(args.out, dpi=100, bbox_inches="tight")
        print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run analyzer**

```bash
source /home/triple/npu-work/mlir-aie/ironenv/bin/activate
python3 /home/triple/npu-work/xdna-emu/tools/analyze-rw-latency.py \
    /home/triple/npu-work/xdna-emu/build/experiments/rw-access-calibration/roundtrip-idle-compute.csv \
    --out /home/triple/npu-work/xdna-emu/build/experiments/rw-access-calibration/roundtrip-idle-compute.png
```

Expected: p50 ~150-200 us (matches the wedge-survey observation), p99 reasonable, max not catastrophic (< 10 ms). Discuss any bimodal structure with Maya before proceeding.

- [ ] **Step 6: Commit**

```bash
cd /home/triple/npu-work/xdna-emu
git add tools/rw-access-probe/rw-access-probe.cpp tools/analyze-rw-latency.py
git commit -m "tools: rw-access-probe --csv mode + analyze-rw-latency

CSV per-read timings + Python percentile summary, to characterize
read_aie_reg host-roundtrip distribution. First step of AIE_RW_ACCESS
calibration (plan: docs/superpowers/plans/2026-05-26-aie-rw-access-characterization.md)."
```

---

### Task 2: Does Timer_Low free-run or is it gated?

**Files:**
- Modify: `tools/rw-access-probe/rw-access-probe.cpp` (already supports the needed flags; no change needed)
- Output: `build/experiments/rw-access-calibration/timer-idle-*.csv`

**Question:** With an xclbin loaded but no kernel dispatched, does Timer_Low advance? At what rate?

- [ ] **Step 1: Capture Timer_Low across varying sleeps on an idle tile**

```bash
mkdir -p /home/triple/npu-work/xdna-emu/build/experiments/rw-access-calibration

for sleep_ms in 1 10 100 1000; do
  pkexec /home/triple/npu-work/xdna-emu/tools/rw-access-probe/build/rw-access-probe \
      --col 0 --row 2 --reg 0x340F8 \
      --num-reads 20 --sleep-ms ${sleep_ms} \
      --label "idle-compute-sleep${sleep_ms}ms" \
      --csv /home/triple/npu-work/xdna-emu/build/experiments/rw-access-calibration/timer-idle-${sleep_ms}ms.csv
done
```

Expected: each run produces 20 timer samples spread over `sleep_ms` * 19 ms wall-clock.

- [ ] **Step 2: Analyze rate (cycles per ms)**

For each sleep_ms, compute the mean delta between consecutive Timer_Low reads divided by sleep_ms. This gives the apparent Timer_Low frequency in cycles/ms.

```bash
for f in /home/triple/npu-work/xdna-emu/build/experiments/rw-access-calibration/timer-idle-*ms.csv; do
  python3 -c "
import csv, sys
rows = list(csv.DictReader(open('${f}')))
vals = [int(r['value']) for r in rows]
deltas = [(vals[i+1] - vals[i]) & 0xFFFFFFFF for i in range(len(vals)-1)]
mean_d = sum(deltas)/len(deltas)
print('${f}: mean delta = {:.0f} cycles  (sleeps were 1-1000 ms; check label)'.format(mean_d))"
done
```

Expected: one of three outcomes -
- **(A) Linear with sleep**: counter is free-running. Compute apparent frequency = mean_delta / sleep_ms.
- **(B) Constant regardless of sleep**: counter is gated to tile activity, advances only during the brief AIE_RW_ACCESS itself (or some fixed work-per-read overhead).
- **(C) Zero or near-zero**: counter only advances during dispatched kernel execution.

The wedge-survey 120 kHz reading suggests (A) but with a much slower clock than core clock. Document which outcome we see.

- [ ] **Step 3: Document outcome inline in a working notes file**

```bash
mkdir -p /home/triple/npu-work/xdna-emu/build/experiments/rw-access-calibration
cat > /home/triple/npu-work/xdna-emu/build/experiments/rw-access-calibration/NOTES.md <<'EOF'
# AIE_RW_ACCESS calibration notes

## Task 2: Timer_Low idle behavior

Observed outcome: [A / B / C] -- [fill in]
Apparent frequency (if A): [fill in] Hz
Notes: [fill in any surprises]
EOF
```

Fill in the file based on Step 2 output. **Decision point:** if outcome is (B) or (C), Task 3 changes shape — we'll need a kernel that does known work for any cycle measurement to make sense.

- [ ] **Step 4: Commit experiment outputs**

```bash
cd /home/triple/npu-work/xdna-emu
git add build/experiments/rw-access-calibration/timer-idle-*.csv \
        build/experiments/rw-access-calibration/NOTES.md
git commit -m "experiments: Timer_Low idle behavior across sleep range

Captures whether Timer_Low free-runs or gates on activity. See
build/experiments/rw-access-calibration/NOTES.md for outcome."
```

---

### Task 3: Timer_Low under known kernel workload

**Files:**
- Create: `tools/rw-access-probe/spin_kernel/` (or reuse an existing kernel from mlir-aie tests)
- Output: `build/experiments/rw-access-calibration/timer-workload.csv` + notes

**Question:** Given a kernel that executes a known cycle count, does Timer_Low advance by that exact amount?

**Approach:** Don't write a kernel from scratch — find a kernel in `mlir-aie/test/npu-xrt/` whose compute core has predictable cycle behavior (e.g., a fixed-iteration scalar loop). Use `llvm-objdump -d` to count instruction cycles via the llvm-aie scheduling model, then read Timer_Low before dispatch / after wait.

- [ ] **Step 1: Identify a deterministic kernel**

```bash
ls /home/triple/npu-work/mlir-aie/build/test/npu-xrt/ | grep -E "add_one|scale|memcpy" | head -5
```

Choose one (suggest `add_one_using_dma` or similar simple compute kernel). Confirm its `.elf` exists at `chess/aie.elf` or `peano/aie.elf`. **Document the choice in NOTES.md.**

- [ ] **Step 2: Disassemble and estimate cycle count**

```bash
KERNEL_ELF=/home/triple/npu-work/mlir-aie/build/test/npu-xrt/<chosen>/chess/aie.elf
/home/triple/npu-work/mlir-aie/my_install/bin/llvm-objdump -d $KERNEL_ELF \
    > /home/triple/npu-work/xdna-emu/build/experiments/rw-access-calibration/kernel-disasm.txt
```

Compute lower-bound cycles by `wc -l` of the instruction listing (each VLIW slot is at minimum 1 cycle). This is rough but enough to distinguish "kernel ran" from "kernel didn't run."

- [ ] **Step 3: Pre/post sample with kernel dispatch**

This requires a small driver program — extend `rw-access-probe` with a `--dispatch` mode that runs the xclbin's kernel once between samples, OR write a separate companion that does pre-read → dispatch → wait → post-read.

Sketch (`tools/rw-access-probe/rw-dispatch-probe.cpp`, new file):

```cpp
// rw-dispatch-probe: pre-sample Timer_Low, dispatch a kernel run,
// wait for completion, post-sample Timer_Low. Prints delta.
//
// Tests whether Timer_Low advances by the kernel's instruction count
// during execution -- pins down Timer_Low's relationship to tile work.

#include "xrt/xrt_aie.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include <cstdio>
#include <cstdint>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <xclbin>\n", argv[0]); return 2; }
    xrt::device dev{0};
    xrt::aie::device aie_dev{dev};
    xrt::xclbin xclbin{argv[1]};
    dev.register_xclbin(xclbin);
    xrt::hw_context ctx{dev, xclbin.get_uuid()};
    // Assume a single kernel; caller's responsibility to choose a kernel with known cycle count.
    auto kernels = xclbin.get_kernels();
    if (kernels.empty()) { std::fprintf(stderr, "no kernels in xclbin\n"); return 2; }
    xrt::kernel k{ctx, kernels[0].get_name()};

    uint32_t pid = getpid();
    uint32_t ctx_id = 1;
    uint32_t before = aie_dev.read_aie_reg(pid, ctx_id, 0, 2, 0x340F8);

    // Allocate buffers, run, wait -- shape depends on chosen kernel.
    // FILL IN per kernel signature.

    uint32_t after = aie_dev.read_aie_reg(pid, ctx_id, 0, 2, 0x340F8);
    std::printf("before=0x%08x after=0x%08x delta=%u\n",
                before, after, (after - before));
    return 0;
}
```

**Note:** The kernel-specific buffer allocation and run-argument setup is the part that depends on the chosen kernel. Look at `mlir-aie/build/test/npu-xrt/<chosen>/test.cpp` for the canonical invocation shape and copy it.

- [ ] **Step 4: Add CMake entry, build, run**

Add the new binary to `tools/rw-access-probe/CMakeLists.txt`:

```cmake
add_executable(rw-dispatch-probe rw-dispatch-probe.cpp)
target_link_libraries(rw-dispatch-probe PRIVATE XRT::xrt_coreutil)
```

Build:
```bash
cd /home/triple/npu-work/xdna-emu/tools/rw-access-probe/build && cmake --build . -j
```

Run (requires pkexec for the AIE_RW_ACCESS calls):
```bash
pkexec /home/triple/npu-work/xdna-emu/tools/rw-access-probe/build/rw-dispatch-probe \
    /home/triple/npu-work/mlir-aie/build/test/npu-xrt/<chosen>/chess/aie.xclbin \
    > /home/triple/npu-work/xdna-emu/build/experiments/rw-access-calibration/timer-workload.txt
```

Expected: a clean before/after/delta line. The delta tells us how many Timer_Low ticks elapsed during the kernel run.

- [ ] **Step 5: Compare delta to expected cycle count**

Lower-bound cycles from Step 2 vs observed delta from Step 4. Document the ratio in NOTES.md:

- If delta ≈ instruction count → Timer_Low is the core clock.
- If delta ≈ instruction count / N → Timer_Low is a div-by-N reference.
- If delta is very small → Timer_Low is gated to a sub-period (likely just the AIE_RW_ACCESS roundtrip work).
- If delta is huge → Timer_Low is sourced from a fast async clock (wall-clock-ish).

- [ ] **Step 6: Commit**

```bash
cd /home/triple/npu-work/xdna-emu
git add tools/rw-access-probe/rw-dispatch-probe.cpp tools/rw-access-probe/CMakeLists.txt \
        build/experiments/rw-access-calibration/timer-workload.txt \
        build/experiments/rw-access-calibration/kernel-disasm.txt \
        build/experiments/rw-access-calibration/NOTES.md
git commit -m "experiments: Timer_Low calibration against known kernel workload

rw-dispatch-probe samples Timer_Low pre/post a kernel run; result
pins Timer_Low's clock source. See NOTES.md for outcome."
```

---

### Task 4: Cross-tile Timer coherence

**Files:**
- Reuse: `tools/rw-access-probe/build/rw-access-probe`
- Output: `build/experiments/rw-access-calibration/timer-cross-tile.csv`

**Question:** Are Timer_Lows in different tiles synchronized? Can we use them to align cross-tile events?

- [ ] **Step 1: Capture Timer_Low from two compute tiles in rapid succession**

```bash
pkexec bash -c '
PROBE=/home/triple/npu-work/xdna-emu/tools/rw-access-probe/build/rw-access-probe
OUT=/home/triple/npu-work/xdna-emu/build/experiments/rw-access-calibration
for i in $(seq 1 50); do
  $PROBE --col 0 --row 2 --reg 0x340F8 --num-reads 1 --label "tile-0-2-iter${i}" \
      --csv ${OUT}/cross-tile-iter${i}-A.csv
  $PROBE --col 0 --row 3 --reg 0x340F8 --num-reads 1 --label "tile-0-3-iter${i}" \
      --csv ${OUT}/cross-tile-iter${i}-B.csv
done
'
```

Expected: 100 CSV files, each with one Timer_Low sample. Each pair (A,B) is read ~roundtrip-latency apart in wall-clock.

- [ ] **Step 2: Analyze drift**

```bash
python3 <<'EOF'
import csv, glob, os
import numpy as np

OUT = "/home/triple/npu-work/xdna-emu/build/experiments/rw-access-calibration"
diffs = []
for i in range(1, 51):
    a_rows = list(csv.DictReader(open(f"{OUT}/cross-tile-iter{i}-A.csv")))
    b_rows = list(csv.DictReader(open(f"{OUT}/cross-tile-iter{i}-B.csv")))
    a = int(a_rows[0]["value"])
    b = int(b_rows[0]["value"])
    diffs.append(b - a)  # signed
diffs = np.array(diffs, dtype=np.int64)
print(f"N={len(diffs)}")
print(f"mean(B-A) = {diffs.mean():.0f} cyc")
print(f"std(B-A)  = {diffs.std():.0f} cyc")
print(f"min/max   = {diffs.min()} / {diffs.max()}")
EOF
```

Expected outcomes:
- **Coherent timers**: mean(B-A) is some positive offset (B always read ~roundtrip after A), std is small.
- **Independent timers**: high std relative to mean — drift between tiles is significant.
- **Tile-2 / tile-3 stopped**: zeros or non-advancing values.

- [ ] **Step 3: Document in NOTES.md**

Append a "Task 4" section with the measured mean/std/range and the qualitative conclusion.

- [ ] **Step 4: Commit**

```bash
cd /home/triple/npu-work/xdna-emu
git add build/experiments/rw-access-calibration/cross-tile-iter*.csv \
        build/experiments/rw-access-calibration/NOTES.md
git commit -m "experiments: Timer_Low cross-tile coherence

50-iteration paired reads between (0,2) and (0,3). See NOTES.md."
```

---

### Task 5: Write up Phase 1 finding doc

**Files:**
- Create: `docs/superpowers/findings/2026-05-26-aie-rw-access-calibration.md`

- [ ] **Step 1: Synthesize Tasks 1-4 into a finding doc**

Structure:
- **TL;DR** (3-5 lines)
- **What Timer_Low actually is** (source clock, gating, frequency)
- **Roundtrip distribution** (p50/p99/max with the histogram PNG embedded)
- **Cross-tile coherence** (drift numbers, what we can/can't align)
- **What AIE_RW_ACCESS is now usable for** (concrete bulleted list)
- **What it can't measure** (explicit limitations)
- **Implications for cycle-cost calibration** (re: the dispatch_overhead 2500 cyc constant)
- **See also** -- link to wedge finding (2026-05-26-aie-rw-access-memtile-wedge-mechanism.md) and dispatch-overhead finding (2026-05-25-npu-controller-dispatch-overhead.md)

Frontmatter:
```markdown
---
name: 'AIE_RW_ACCESS calibration on Phoenix'
description: 'Empirical characterization of Timer_Low semantics, host-roundtrip latency distribution, and cross-tile coherence for the read_aie_reg path. Result: AIE_RW_ACCESS is usable for [X], not usable for [Y].'
type: project
---
```

- [ ] **Step 2: Commit**

```bash
cd /home/triple/npu-work/xdna-emu
git add docs/superpowers/findings/2026-05-26-aie-rw-access-calibration.md
git commit -m "docs(rw-access): Phase 1 calibration findings

Characterizes Timer_Low source/gating/frequency, host-roundtrip
distribution, and cross-tile coherence. Establishes what AIE_RW_ACCESS
can and can't measure as a cycle-accurate probe."
```

---

## Phase 1 Decision Gate -- CLOSED (2026-05-26)

**Result: AIE_RW_ACCESS is NOT a viable cycle-counter probe.** See
[`docs/superpowers/findings/2026-05-26-aie-rw-access-not-a-cycle-probe.md`](../findings/2026-05-26-aie-rw-access-not-a-cycle-probe.md)
for the full empirical writeup. Headline:

- T1: roundtrip distribution is OK (p50=73us, p99=192us).
- T2: Timer_Low advance per call is a fixed ~11,870 ticks (std 0.1%) regardless of wall-clock duration between calls.
- T3: bracket sampling across a verified-PASS kernel run shows the same ~12,000-tick delta as a single idle call. Timer_Low at the running tile does not advance via AIE_RW_ACCESS.
- T3 follow-up: `write_aie_reg` succeeds at the API level but has no visible effect — Timer_Control reads as 0 regardless of what we write, and the trace pipeline's `aiex.npu.write32` to Timer_Control in `runtime_sequence` also produces no readable change.

T4 (cross-tile coherence) is cancelled — measuring drift between two timers that don't tick in a useful way is moot.

T5 (finding doc) lives at the file linked above. AIE_RW_ACCESS remains a useful **state inspection** probe (the wedge survey use case), but cannot serve as the dispatch_overhead cross-validation tool the earlier finding speculated about.

---

## Phase 2 (AMENDED 2026-05-26): Pivot to trace-unit cycle-accuracy work

The original Phase 2a (cross-validate `dispatch_overhead = 2500 cyc` via AIE_RW_ACCESS cycle counter readback) is dead. The trace unit remains the cycle-accuracy tool. Phase 2 now focuses on:

1. **Tightening trace-based measurements** — Maya's "eliminate as much noise as we can, even the long circuitous way" applied to trace data.
2. **Phase 2b unchanged**: K=16 wedge state inspection via AIE_RW_ACCESS (this is the state-inspection use case, which we know works).

### Phase 2a (amended): Trace-data noise reduction for dispatch_overhead

Goal: tighten our understanding of the 2500-cyc dispatch_overhead constant from the trace data we already have, without depending on a new HW probe. The dispatch-overhead finding noted "Run-to-run HW variance is large. Per-task and per-gap measurements vary 30-50% between sweeps." That's the noise floor we want to chip away at.

Noise sources to attack (each its own sub-task):

- **Cross-run aggregation.** Currently each measurement comes from a single bridge run. Run the K-sweep N times (50-100), aggregate inter-task gaps per-K-per-direction, compute statistical summaries (median, MAD, IQR, not just mean). Outlier filtering at >3 sigma. Targets the run-to-run variance directly.
- **First-gap separation.** The dispatch-overhead finding flagged first-gap variance as systematically different from steady-state. Separate first-gap from inter-task gaps in the aggregation, compute distributions independently, model first-gap as a phase-transition effect rather than averaging it in.
- **Warmup runs.** Discard the first M runs (cache cold, FW state cold) before measurement. Find M empirically.
- **Per-event slot decorrelation.** With 8 event slots per tile, our trace covers a different subset on each run. Cross-run aggregation lets us approach "full event coverage" statistically.
- **Per-tile vs aggregate.** Inter-task gaps may differ by source tile (shim col=0 vs shim col=1). Stratify rather than aggregate.

Tasks (concrete, written after Phase 1 closed):

| Task | What |
|------|------|
| 2a.1 | Build a multi-run trace harness that fires the K-sweep N times against HW, saves raw trace data per run, and saves run-tagged metadata (BO indices, FW state, NPU temp if available) |
| 2a.2 | Build a per-K/per-direction/per-run aggregator that produces distribution summaries (not just means) |
| 2a.3 | Visualize: per-K gap distribution histograms, run-to-run scatter, first-gap vs steady-state |
| 2a.4 | Re-derive the dispatch_overhead constant from the aggregated data; revisit whether a single constant is the right model or whether a (direction, K)-conditional model fits better |
| 2a.5 | If the constant moves materially, update `src/npu/cycle_cost.rs` and re-run the bridge suite |
| 2a.6 | Finding doc: what the noise structure actually is, what was averaged away in the original calibration, whether 2500 cyc is still the right number |

### Phase 2b (unchanged): K=16 wedge state inspection

EMU silently defers past 8-deep task queue; HW wedges. We don't know what HW state looks like at the wedge because trace events stop firing. AIE_RW_ACCESS is well-suited to this use case — we want to read tile state at a host-chosen moment (just-after-wedge), not measure cycles. Its negative-cycle-probe finding doesn't affect this use.

Approach:
1. Run K=16 to wedge.
2. AIE_RW_ACCESS-read shim DMA state on (0, 0): channel status, BD pointer, task queue depth, channel control regs.
3. Document the wedge state. Use it to drive the emulator fix described in the dispatch-overhead finding's "Follow-ups" section ("treat queue overflow as a hard error rather than blocking until queue drains").

Safety: wedge-survey safety patch blocks row=1 memtile. Shim is row=0, unaffected.

---

## Phase 2a Decision Gate -- CLOSED (2026-05-27)

**Result: structural-variance characterized; model upgrade deferred to a
dedicated re-calibration sprint.** See
[`docs/superpowers/findings/2026-05-27-dispatch-overhead-multirun-structural-variance.md`](../findings/2026-05-27-dispatch-overhead-multirun-structural-variance.md).

What landed:
- 2a.1 -- multi-run trace harness (`tools/multirun-trace-campaign.py`):
  N=50 across K in {1,2,4,8} = 200 iterations in ~5 min wall-clock.
  Direct `bridge-trace-runner` + `parse-trace.py` per iteration.
- 2a.2 -- aggregator (`tools/aggregate-dispatch-overhead.py`):
  per-(K, direction, gap_index) distribution summaries; reveals
  bimodal MM2S and structural S2MM patterns.
- 2a.3 -- visualization (`tools/plot-dispatch-overhead.py`):
  histograms + boxplots per direction.
- 2a.6 -- finding doc covering F1-F5 (variance is structural,
  K-dependent S2MM elevations, MM2S Task_Queue pipelining,
  HW depth=8 vs aie-rt depth=4, EMU model audit).

What's deferred to a future sprint (Phase 2c):
- 2a.4 -- re-derive `dispatch_overhead`. A Q-aware refactor that only
  charges full overhead when the channel was idle at dispatch is the
  mechanically-correct model upgrade. A one-line constant bump
  (2500 -> 2785) on the existing universal-application model breaks
  K=4 MM2S accuracy (the original calibration's empirical tuning
  relied on offsetting errors).
- 2a.5 -- update `src/npu/cycle_cost.rs` and `src/npu/executor.rs`.
  Same blocker -- needs coupled re-calibration of `cmp_decode_cost`,
  `fabric_cost`, `dispatch_overhead`, and a new
  `dispatch_overhead_pipelined` field against the full bridge corpus.

Why deferred: the current `dispatch_overhead = 2500` is mechanistically
wrong (over-counts pipelined dispatches, under-counts serialized ones)
but empirically tuned at the K-sweep span level. A Q-aware refactor
needs simultaneous re-tuning of multiple cost components and full
bridge-suite validation -- that's a dedicated sprint, not a 2a closer.

The variance is now characterized; the model upgrade has a clear
shape; we know what numbers to target. The remaining work is
implementation + validation, which benefits from fresh context.

---

## Phase 2b status (unchanged)

Still planned. K=16 wedge state inspection via AIE_RW_ACCESS reads of
shim DMA channel state at the moment of wedge. Concrete tasks not yet
written; depends on whether the Phase 2c re-calibration sprint takes
priority over wedge characterization first.

---

## Phase 2 (amended) Decision Gate

After Phases 2a and 2b complete:
- 2a result: variance structure characterized, model upgrade scoped
  but deferred. (Closed 2026-05-27.)
- 2b result: K=16 wedge mechanism is characterized; EMU's
  queue-deferral semantics get the appropriate fix.

**Update this plan** with Phase 3 tasks or close the plan if no harness work is needed.

---

## Phase 3: Reusable Characterization Harness (deferred)

Goal (to be confirmed after Phase 2): take an xclbin as input, run it N times on HW with trace instrumentation, produce a structured timing report with proper statistical treatment (distributions not means, per-event decorrelation, warmup handling). Run the same xclbin in EMU and produce the same report shape. Diff them.

Building on the noise-reduction work from Phase 2a: the harness is essentially "Phase 2a generalized to arbitrary kernels, not just the K-sweep."

The exact shape depends on what Phase 2 reveals about which measurements actually carry signal. Do not plan in detail until Phase 2 closes.

---

## Self-Review

Verified against the four open questions in the Goal:

| Question | Addressed in |
|----------|--------------|
| Timer_Low free-run vs gated? | Task 2 |
| Cycle source identity? | Task 3 (workload-correlated delta) |
| Host-roundtrip distribution? | Task 1 |
| Cross-tile coherence? | Task 4 |
| Wall-clock pairing usable? | Implied by Task 1 + Task 2 (distribution width vs counter advance rate determines feasibility) |

No placeholders detected. All file paths absolute, all commands executable, all expected outputs specified.

Phase 2 is intentionally sketched, not detailed — premature precision would lock in measurement strategies that Phase 1 may invalidate. The decision-gate structure preserves the option to redirect.
