---
name: Validate read_aie_reg path before building calibration on top of it
description: Throwaway C++ harness that proves the XRT read_aie_reg path returns real, wall-time-correlated, round-trip-honest data on Phoenix NPU1, plus probes the bridge-runner pre-launch lifecycle bug
type: spec
---

# Spec: validate-readback harness

## Why

`read_aie_reg` (XRT C++ API, opcode `MSG_OP_AIE_RW_ACCESS`) was unblocked
on NPU1 yesterday by a driver op-table fix. Before we plumb it into the
calibration / bridge-test loop as ground truth, we need to confirm:

1. The XRT path returns *real* data, not stale, cached, zero-filled, or
   synthetic values.
2. Tile addressing (`col`, `row`) maps to the tile we think it maps to.
3. Read and write round-trip honestly.
4. Run-to-run the same kernel produces consistent counter values.
5. The bridge-runner pre-launch lifecycle bug (`hwctx->num_col == 0`
   until first `run.start()`) actually behaves as diagnosed.

Without this layer of assurance, any calibration built on top of
`read_aie_reg` could be trusting a phantom signal.

## Scope

Throwaway harness. Single binary, ~250 LOC, lives at
`tools/validate-readback/`. Default xclbin: `add_one_using_dma`.
Probably deleted after we have confidence; may stay as a regression
check.

## Architecture

```
tools/validate-readback/
├── CMakeLists.txt          links libxrt_coreutil, mirrors bridge-runner pattern
├── validate-readback.cpp   single-file binary, ~250 LOC
└── README.md               short, what each test means and how to read output
```

CLI:

```
validate-readback --xclbin PATH [--col N] [--row N] [-v]
```

- `--xclbin`: path to a compiled xclbin whose kernel runs on (col, 2).
  Default: hardcoded to a known build of `add_one_using_dma`.
- `--col`, `--row`: physical column / row to target. Defaults to the
  start_col reported by the hwctx after the dummy run.
- `-v`: verbose, prints raw register values and intermediate state.

Build path:

```
cmake -S tools/validate-readback -B tools/validate-readback/build
cmake --build tools/validate-readback/build
./tools/validate-readback/build/validate-readback --xclbin <path>
```

## Test sequence

Each test prints one line: `[ID] PASS|FAIL|SKIP|INFO  description`.
Final line: `VALIDATION: N/M PASS`. Exit code = number of failures.

### L0 — Pre-launch read fails as diagnosed (lifecycle probe)

Right after `xrt::hw_context` construction, before any `run.start()`:
attempt `read_aie_reg(col, 2, TIMER_LOW)`. Catch any exception.

- **PASS**: read threw (lifecycle bug confirmed; we proceed knowing
  pre-launch programming is unsupported until partition allocates).
- **INFO** (no FAIL): read succeeded. The lifecycle diagnosis was
  incomplete; we record the value and continue. This is unexpected but
  good news.

### L1 — Same-hwctx warmup unblocks pre-launch (lifecycle workaround)

After the first dummy `run.start() + run.wait()` completes (which
allocates the partition and sets `hwctx->num_col` for THIS hwctx),
attempt a `read_aie_reg(col, 2, TIMER_LOW)` on the same hwctx, before
calling `run.start()` again. This is what option A in `NEXT-STEPS.md`
proposes: warmup run first, then pre-launch programming on subsequent
runs.

- **PASS**: read succeeds. Confirms option A is viable for the bridge
  harness — a warmup run unblocks all subsequent pre-launch
  programming on the same hwctx.
- **FAIL**: read still throws. Bug is per-run-scoped, not
  per-hwctx-scoped; option A doesn't work as stated and we'd need a
  different workaround.

Stretch test — does not block remaining tests if it fails.

(Optional second arm — only if we have time: also test a *fresh*
hwctx after the first hwctx has run, to see if partition state is
hwctx-local or system-wide. Skip if it complicates the binary.)

### V0 — TIMER_LOW monotonic across wall time

After dummy kernel run completes, read `TIMER_LOW` (compute tile,
col=col, row=2, addr=0x340F8). Sleep 1 ms via `usleep(1000)`. Read
again. Compute delta.

- **PASS**: delta in [200_000, 800_000] (factor-of-2 tolerance around
  the expected ~400_000 cycles at 400 MHz MP-NPU clock).
- **FAIL**: delta out of range, including delta == 0 (counter not
  advancing) or delta > 800k (clock estimate wrong, or read latency
  dominates).

This is the strongest single test: synthetic data is unlikely to
correlate with wall time at the right rate.

### V1 — Write-and-read-back PERF_COUNTER0

Write `0xDEADBEEF` to `PERF_COUNTER0` (col=col, row=2, addr=0x31520).
Immediately read it back.

- **PASS**: read value == 0xDEADBEEF.
- **FAIL**: any other value (zero, garbage, or evolved-by-counting).

If the counter's `start_event` is not zero and is currently firing, the
value may have advanced by a few cycles between write and read; allow
small tolerance (delta ≤ 100). The test should warn but PASS in this
case; we'll know we need to either disable the counter first or accept
the noise.

### V2 — Cross-tile distinctness

Configure PERF_CTRL0 on (col, 2) for `EVENT_ACTIVE_CORE` (0x1C). Zero
PERF_COUNTER0 on both (col, 2) and (col, 3). Run a real kernel that
executes on (col, 2). Read PERF_COUNTER0 on both (col, 2) and (col, 3).

- **PASS**: (col, 2) > 0 AND (col, 3) == 0.
- **FAIL**: (col, 3) > 0 (col indexing maps to wrong tile, or both
  counters were programmed) or (col, 2) == 0 (counter not counting).

### V3 — Sanity-check magnitude

From V2, take the (col, 2) PERF_COUNTER0 value. Compare to
`kernel_us * 400`, where `kernel_us` is the host-side wall-time delta
between `run.start()` and `run.wait()` return (`std::chrono::steady_clock`),
giving an expected cycle count at 400 MHz.

- **PASS**: ratio in [0.1x, 10x] of expected.
- **FAIL**: ratio outside that range. (Tight bound: if the counter is
  100x off, something is wrong with our event configuration or with
  the count itself.)

### V4 — Run-to-run reproducibility

Re-zero PERF_COUNTER0 on (col, 2). Run the same kernel a second time.
Read PERF_COUNTER0.

- **PASS**: `|run2 - run1| / run1 < 0.5` (50% tolerance — generous,
  because run-to-run variance under arbiter / DRAM contention is real).
- **FAIL**: outside that range.

## Data flow

```
   xrt::xclbin (load)
       |
   xrt::hw_context (construct)
       |
   [L0]   <-- pre-launch read attempt
       |
   xrt::run (construct, fill kernargs)
       |
   run.start() / run.wait()    <-- dummy run, allocates partition
       |
   [L1]   <-- second hwctx, pre-launch attempt (stretch)
       |
   [V0]   <-- TIMER_LOW * 2 with sleep
       |
   [V1]   <-- write PERF_COUNTER0, read back
       |
   configure PERF_CTRL0 for ACTIVE_CORE
   zero PERF_COUNTER0 on tiles A and B
       |
   run.start() / run.wait()    <-- real run for V2
       |
   [V2]   <-- read PERF_COUNTER0 on A and B
       |
   [V3]   <-- compare counter to kernel_us * 400
       |
   zero PERF_COUNTER0
   run.start() / run.wait()    <-- second real run
       |
   [V4]   <-- compare to V2 counter
       |
   summary, exit code
```

## Error handling

- All XRT calls are wrapped in `try { ... } catch (const std::exception& e)`.
- Exceptions during a test produce `FAIL` with the exception message.
- Exceptions during setup (xclbin load, hwctx construction, kernel
  resolve) abort the harness with a clear error and exit code 2.
- Test infrastructure failures are distinguished from test FAILs:
  setup errors print `SETUP ERROR: ...` not `[Vx] FAIL`.

## Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| xclbin's kernel doesn't run on (col, 2) | Default to `add_one_using_dma` known-good; CLI override |
| HW partition mapping shifts kernel to physical col != 0 | Query `hwctx` for start_col, use as the default `col` |
| Counter from a prior run leaks into V1 | Disable counter (PERF_CTRL0 = 0) before V1 write |
| Test suite leaves the counter programmed | Final cleanup: zero PERF_CTRL0 |
| Partition state contaminates V0 if dummy kernel is mid-flight | All reads happen post-`run.wait()` |

## Out of scope

- Multi-counter measurement (cycles + stalls + lock-acquires)
- Memtile / shim perf counters
- Calibration plumbing (per-test JSON output, dashboard, etc.)
- Cross-architecture (NPU4 / Strix) — that's task #357
- Fixing the lifecycle bug (only diagnosing it)

## Success criteria

- L0 confirms or refutes the lifecycle-bug diagnosis.
- L1 produces a clear signal on whether option-A workaround is viable.
- V0 + V1 together prove the readback path is honest (wall-time
  correlated and round-trip exact).
- V2 + V3 + V4 prove the path is honest under real kernel runs.
- All six tests run to completion in under 30 seconds.

If V0+V1 PASS, we have enough confidence to move on to plumbing
calibration measurements; V2-V4 are corroborative.

## Effort estimate

2-3 hours: 30 min skeleton + CMake; 1 hr L0/L1/V0/V1 (cheap, no kernel
runs); 1 hr V2/V3/V4 (need real kernel run); 30 min cleanup and shake-out.

## After this lands

- If validation PASSES: spec a follow-up that integrates per-test
  cycle measurement into the bridge harness and uses the resulting
  diff as the calibration signal for #355a.
- If any V test FAILS: diagnose the failure mode before building
  anything on top. Probably a follow-up finding doc.

## See also

- `NEXT-STEPS.md` lines 31-110 — original lifecycle-bug diagnosis
- `docs/archive/findings/2026-05-05-aie-rw-access-firmware-actually-supported.md` — driver fix breakthrough
- `bridge-runner/bridge-trace-runner.cpp:1700-1750` — existing
  pre/post-launch read-perf-counter scaffolding (the lifecycle bug
  lives here)
- `xdna-driver/src/driver/amdxdna/aie2_message.c:1772` — `aie2_rw_aie_reg`
  table check, now satisfied for NPU1
- task #355a — what this validation enables
- task #356 — completed: op-table entry verified
