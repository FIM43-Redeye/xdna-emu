# Session notes -- 2026-05-06

Scratch pad for resuming after context compaction. Captures everything
material from today's session in enough detail that a fresh session can
pick up cleanly.

Today: wrapped #355b validation, fixed column-offset for good (option C
placement), took a measured first step on #355a, kept the trace pipeline
honest with a bridge-test fix. **Then in the second half of the session:
specced + built + landed validate-readback (proves read_aie_reg returns
real ground truth), patched the xdna-driver to allow non-root AIE
register writes, rebased the driver branch onto upstream main, and got
DKMS to plug-and-play the result.**

## What landed today (xdna-emu, commits in order)

| SHA       | Subject                                                              |
|-----------|----------------------------------------------------------------------|
| `2d84e55` | interpreter: emit periodic LOCK_STALL during held WaitingLock state (#355b) |
| `d747c24` | bridge-test: pass --remap-columns to trace.log compare path          |
| `0fcf234` | trace: declarative per-side placement origin (option C)              |
| `bcf862b` | archspec: bump host_memory_latency_cycles 100 -> 500 (#355a step 1)  |
| `3c035cb` | spec: validate read_aie_reg path before building calibration on top  |
| `9bbbbca` | plan: validate-readback implementation plan                          |
| `d0c6d13` | validate-readback: scaffold, load xclbin and hwctx                   |
| `2188893` | validate-readback: add dummy kernel run helper                       |
| `6ee1acc` | validate-readback: L0 pre-launch read probe                          |
| `6b040a3` | validate-readback: L1 post-warmup pre-launch unblock probe           |
| `cd25504` | validate-readback: V0 TIMER_LOW monotonic probe                      |
| `12214bb` | validate-readback: V1 write-and-read-back probe                      |
| `5704a6f` | validate-readback: summary, cleanup, README                          |

## What landed today (xdna-driver, branch `xdna-emu-cycle-budget`)

Backup tag: `pre-rebase-2026-05-06` at the pre-rebase HEAD `6aa4dc7`.

| SHA       | Subject                                                              |
|-----------|----------------------------------------------------------------------|
| `25b3f51` | accel/amdxdna: add raw-mailbox debugfs test_case04 (was `e67177a` pre-rebase, yesterday) |
| `289c207` | accel/amdxdna: enable MSG_OP_AIE_RW_ACCESS on NPU1 (was `33a7fe7` pre-rebase, yesterday) |
| `0ec3706` | amdxdna: dev-tree opt-in to allow non-root SET_STATE IOCTL (was `6aa4dc7` pre-rebase, today) |

These three sit on top of upstream `e6df57f` (latest origin/main as of
2026-05-06). Six upstream commits absorbed in the rebase, all on
`drivers/accel/amdxdna/*` (in-tree submission path) -- zero file-level
overlap with our patches in `src/driver/amdxdna/*`. Rebase was clean.

(The last xdna-emu commit before today was `10d30cf` "docs: diagnose #355
cycle-divergence as two independent problems".)

## Driver state (end-of-day, 2026-05-06, post second reboot)

**No action needed -- DKMS plug-and-play across kernel updates.**

- Kernel: 7.0.3-custom (same as yesterday).
- Loaded `amdxdna.ko` is at `/lib/modules/7.0.3-custom/updates/dkms/amdxdna.ko`,
  signed with our MOK key, includes ALL THREE of our patches:
  - `test_case04` debugfs (yesterday)
  - npu1 op-table entry for `MSG_OP_AIE_RW_ACCESS` (yesterday)
  - `AMDXDNA_DEV_UNSAFE_USER_WRITES` opt-in dropping `DRM_ROOT_ONLY`
    on `SET_STATE` (today)
- DKMS source at `/usr/src/xrt-amdxdna-2.23.0/` is auto-synced from
  `/home/triple/npu-work/xdna-driver/src/` by `./build.sh -release`.
  On future kernel upgrades DKMS auto-rebuilds + auto-signs from there.
- After a source change, manual rebuild sequence (until we wrap it in a
  script):
  ```bash
  cd /home/triple/npu-work/xdna-driver/build && ./build.sh -release
  pkexec bash -c '
    dkms remove xrt-amdxdna/2.23.0 -k $(uname -r) && \
    dkms install xrt-amdxdna/2.23.0 -k $(uname -r) && \
    modprobe -r amdxdna && modprobe amdxdna'
  ```
  The `dkms install` step now auto-signs with our MOK key (no manual
  `sign-file` call needed).
- The earlier "manual sign + insmod" sequence in this file's comment
  history is no longer the recommended path -- use the DKMS sequence
  above instead.
- DKMS auto-rebuilt and installed it on first boot. Verified: `accel0`
  device live, `xrt::hw_context::read_aie_reg` path is open.
- The path of the previous build artifact: `/home/triple/npu-work/xdna-driver/build/Release/driver/amdxdna.ko`
  (release build of our patched tree).
- If the kernel changes again later, sequence is:
  ```bash
  cd /home/triple/npu-work/xdna-driver/build && ./build.sh -release
  pkexec bash -c '
    /lib/modules/$(uname -r)/build/scripts/sign-file sha256 \
      /var/lib/shim-signed/mok/MOK.priv \
      /var/lib/shim-signed/mok/MOK.der \
      /home/triple/npu-work/xdna-driver/build/Release/driver/amdxdna.ko
    modprobe -r amdxdna && insmod /home/triple/npu-work/xdna-driver/build/Release/driver/amdxdna.ko'
  ```

## #355b validation (commit 2d84e55)

Code path: in `src/interpreter/core/interpreter.rs`, the `WaitingLock`
arm of `try_resume_stall` accumulates a `lock_stall_periodic` counter
per cycle of held stall and emits a `LockStall` trace event every
`LOCK_STALL_TRACE_PERIOD = 1024` cycles. Counter resets on lock acquire
and on `reset()`.

Lib tests pass (2863). Bridge test on `add_one_using_dma` passes too.

**However**: on `add_one_using_dma`, EMU's actual core-side lock wait is
only ~253 cyc (under the 1024 threshold), so the periodic emission
doesn't fire on this test. The full validation needs a test where the
core spends >1024 cyc held in `WaitingLock`. That depends on #355a
making EMU's pipeline fill take long enough that the core sees a real
held wait. Right now EMU's DMA fills mostly in parallel with kernel
setup, leaving only a brief held wait at the end.

Closed tasks: #353 (LOCK_STALL emission), #360 (#355b code change).
Still open: #354 (PERF_CNT_0 anchor pulses) -- same emission pattern,
slot 0 instead of slot 6, deferred.

## Column-offset fix (commits d747c24 + 0fcf234)

Two-stage fix. First stage was a one-line bandaid (passing
`--remap-columns` to the trace.log compare path that was missing it).
Second stage was option C from the brainstorm: declarative per-side
placement.

**Schema**: `tools/trace_config_schema.json` gained an optional
top-level `placement: {origin_col, origin_row}` field. Documents the
MLIR-intended kernel placement.

**Producers**:
- `tools/mlir-trace-inject.py`: writes placement = min(col, row) across
  `tiles_traced` into `trace_config.json` at trace-prep time.
- `tools/parse-trace.py`: emits per-side observed `placement` into
  events.json (smallest col/row that produced events). This reflects
  HW's runtime-resolved start_col, not the MLIR intent.

**Consumer**: `src/trace/compare.rs::load_events_json` returns the
placement; `compare_batch_with_opts` uniformly translates each side's
tile keys by its own origin. Falls back to dense-remap
(`--remap-columns`) for events.json files that predate placement.

**Why "uniform shift" matters**: dense-remap maps cols `{1, 3}` to
`{0, 1}`, collapsing gaps. Placement-shift maps to `{0, 2}`, preserving
gaps. For contiguous kernels both produce the same result; for
non-contiguous kernels (HW touching scattered cols) only placement is
correct. We don't have such tests today, but the design is now ready
for them.

**Verified**: HW placement `{1, 2}`, EMU placement `{0, 2}` on
add_one_using_dma; both shift to `(0, 0)`; comparator produces a real
per-tile breakdown.

Bridge script's `--remap-columns` flag is still wired up, now as a
no-op-when-placement-present backstop. New test added in
`tools/tests/test_parse_trace.py` to assert placement is emitted.

Closed task: #317 (compare HW vs EMU bridge test outputs).

## #355a status (commit bcf862b + finding doc)

**Took one defensible calibration step**: bumped
`host_memory_latency_cycles` from 100 to 500 in `model_builder.rs:165`.

**Measured response**: linear, slope 1.0. EMU pipeline fill on
`add_one_using_dma` (measured as count of `tile(0,2) ch2.*granted=false`
log lines between core's first `LockAcquire WAIT` and corresponding
`SUCCESS`):

| host_memory_latency | EMU pipeline fill | EMU/HW ratio  |
|--------------------:|------------------:|--------------:|
| 100                 | 2319 cyc          | 2.59x         |
| 500                 | 2719 cyc          | 2.21x         |

**Why the gap can't close on this constant alone**: hitting HW's
~6000 cyc target would need host_lat ~ 3700, well above any plausible
PCIe + NoC + DDR fill estimate (~250-500 cyc realistic on Phoenix).
500 is defensible as a baseline; the residual gap lives in missing
structural latency.

**Three identified structural gaps** (priority order):

1. **Stream-switch fabric per-hop latency** -- constants exist in
   `StreamSwitchTiming` (`local_to_local_latency: 3`, etc.) but the
   data-path FSM doesn't consume them. Each tile-to-tile transfer
   should add ~3-4 cyc/hop. Probable +6-12 cyc/transfer addition.
2. **Direct HW ground truth via `read_aie_reg`** -- now works on
   Phoenix (op-table fix landed yesterday in `33a7fe7` of xdna-driver).
   Would let us calibrate against actual `Performance_Counter0`
   readback instead of inferred trace metrics. Bigger infra change but
   removes the guesswork.
3. **NoC bandwidth/contention** -- heavier; matters for
   matmul-cascade-style tests with multiple concurrent shim transfers.

The full analysis is at
`docs/superpowers/findings/2026-05-06-355a-host-latency-response.md`.

Lib tests still pass (2863). #355a stays `in_progress`; this is one
step, not closure.

## Decomposed problem geometry (for reference)

The original 12.7x trace-span ratio decomposed into three roughly
independent sub-problems. Today's commits address (b) and (c) directly,
plus position us for (a):

| Sub-problem                                | Status        | Owner |
|-------------------------------------------|---------------|-------|
| (a) DMA pipeline fill 2.6x too fast       | partial fix   | #355a |
| (b) EMU trace-controller starts post-fill | n/a (we now emit periodic LockStall during held wait) | #355b |
| (c) Trace span misleading due to (a)+(b)  | resolved when (a) is properly calibrated | -- |

Two derivative problems live alongside:

- **EMU core boots ~2.9x too fast** (~2085 cyc to first lock vs HW
  ~6000). Likely instruction-execution timing too aggressive on
  certain bundles. Out of scope for #355a; would be a separate
  cycle-cost-framework calibration.
- **EMU PERF_CNT_0 not emitted** (#354). Same fix shape as #355b but
  for slot 0; deferred until we have a kernel that actually exercises
  the held-wait path long enough to need it.

## Open tasks (relevant, still pending)

- **#321** -- EMU trace-stop timing for late-kernel events. Still
  open; orthogonal to #355a/#355b.
- **#342** -- Differential validation of calibrated `CycleCostModel`.
  Subsumes parts of #355a once full calibration is done.
- **#347** -- Measure broadcast event propagation latency on real HW.
  Becomes much easier if we wire up `read_aie_reg`-based timing.
- **#354** -- EMU PERF_CNT_0 anchor pulses. Same fix as #355b for a
  different slot; will land when needed.
- **#357** -- On NPU4 retry `xrt::hw_context::read_aie_reg`. Strix
  hardware required.
- **#359 (#355a)** -- Calibrate DMA pipeline-fill cycles against HW.
  In progress; one step taken today.

## What's newly unblocked as of end-of-day

The validate-readback work changes the calibration plan in a real way:

- **`xrt::hw_context::read_aie_reg` is confirmed honest ground truth**
  on Phoenix NPU1. Reads work for any user; writes work for any user
  too with our DKMS-installed driver patch. So we can program perf
  counters and read them back per kernel run, without root, without
  trace-pipeline indirection.
- The lifecycle bug is real but the workaround is simple:
  **warmup-run-then-measure** within a single hwctx. L1 confirmed this
  works: after the first `run.wait()` returns, all subsequent
  pre-launch register access on the same hwctx succeeds.
- One unexpected subtlety: `add_one_using_dma`'s compute core hits
  `aie.end` after 4 iterations and a second `run.start()` on the same
  hwctx hangs (`ERT_CMD_STATE_NORESPONSE`). Bridge-runner does many
  runs without this hang -- probably by recreating hwctx per batch,
  or by using a kernel with a non-terminating loop. This is THE thing
  to figure out before plumbing read_aie_reg measurements into a
  per-test sweep.

## Recommended next-session moves (rough priority order, updated)

1. **Crack the multi-run-on-same-hwctx pattern** -- before any
   calibration plumbing, figure out how bridge-runner avoids the
   second-run hang. Two probable causes to investigate:
   (a) `add_one_using_dma`'s core terminates at `aie.end`, and either
   bridge-runner uses a different kernel that loops forever, or it
   recreates the hwctx between runs;
   (b) something about runtime_sequence cycling that I'm missing.
   Once we know the pattern, V2/V3/V4 in validate-readback unblock
   trivially, and the per-test cycle-measurement plumbing becomes
   straightforward.
2. **Wire `read_aie_reg`-based HW cycle measurement into the bridge
   harness.** Per kernel run, program PERF_CTRL0/PERF_COUNTER0 for
   ACTIVE_CORE on the compute tile, run, read counter back, emit in
   per-test JSON. Once #1 is solved, this is mostly bridge-runner
   plumbing. Calibration sweeps then have direct ground truth instead
   of trace-derived metrics.
3. **Sweep host_memory_latency over a small grid** (e.g., 250, 500,
   1000) on multiple test patterns (single-input, multi-input,
   matmul) to validate the linear response holds across patterns
   AGAINST DIRECT MEASUREMENT, not inferred metrics. Now that #2
   gives us ground truth, this becomes well-conditioned.
4. **Re-run the full bridge sweep** on the current commits to make
   sure the host_lat=500 change didn't regress anything else. We only
   tested add_one_using_dma EMU-only today.
5. **Optional: wrap the post-source-edit DKMS rebuild in a script.**
   `tools/refresh-driver.sh` or similar that does
   `./build.sh -release && pkexec dkms remove + install + modprobe`.
   Saves typing whenever we re-edit the driver.
6. **Optional: address #354 (PERF_CNT_0 emission)** -- same code
   shape as #355b, would naturally fall out alongside #355a once we
   have a long-wait test scenario.

## Things to remember (subtle)

- The "EMU pipeline fill = 2319/2719 cyc" measurement is the count of
  `tile(0,2) ch2.*granted=false` log lines on the COMPUTE OUTPUT
  channel during the COMPUTE CORE's first `LockAcquire raw=49 WAIT`
  window. It measures how long ch2 polled before the kernel released
  its first output lock -- a proxy for "kernel time to first output",
  not "DMA pipeline fill" in the strict sense. This is the metric the
  finding doc has used since 2026-05-05 and is what we should keep
  using for cross-session continuity.
- The trace-event-derived "core wait window" (HW 2839 cyc, EMU 253
  cyc) is a different metric -- time the core was held in
  `WaitingLock` between first `LOCK_STALL` event and first
  `INSTR_LOCK_ACQUIRE_REQ`. Both metrics are valid; they answer
  different questions.
- DKMS happily rebuilds our patched module after kernel upgrades. No
  manual signing/insmod usually needed; just verify with
  `modinfo amdxdna | grep signer`.
- Sandbox: tests need `TMPDIR=/tmp/claude-1000` for `cargo test --lib`
  to avoid `/tmp` issues. Already noted in `xdna-emu/CLAUDE.md`.

## Validation evidence (today)

### xdna-emu

- `cargo test --lib`: **2863 passed, 0 failed** (with all of today's
  changes applied).
- `tools/tests/test_parse_trace.py`: 11 passed.
- `tools/test_trace_prepare.py`: 11 passed.
- Bridge test on `add_one_using_dma` (chess + peano):
  - Yesterday w/ HW: PASS x4.
  - Today EMU-only after host_lat bump: PASS x2.
- Schema validation: `python3 tools/trace_config.py validate` passes
  with and without the new `placement` field (correctly optional).

### validate-readback (final state, run against rebased + DKMS module)

```
[L0] PASS pre-launch read threw as expected: ... err=-22: Invalid argument
[L1] PASS post-warmup pre-launch read OK, TIMER_LOW=0x...
[V0] PASS timer_lo: ... (delta=~11_850 over ~1ms, ~11.8 MHz effective)
[V1] PASS wrote 0xdeadbeef, read 0xdeadbeef
[V2] SKIP second-run-on-same-hwctx hangs (ERT_CMD_STATE_NORESPONSE); ...
[V3] SKIP blocked on V2
[V4] SKIP blocked on V2
VALIDATION: 4/7 PASS (3 skipped, 0 failed)
```

V0's effective rate (~12 MHz) is itself a finding: the tile clock is
gated when no kernel is running, vs the ~400 MHz core clock when
active. TIMER_LOW deltas across kernel runs reflect *active* core
time, which is what we want for calibration.

### xdna-driver

- Branch `xdna-emu-cycle-budget` rebased onto upstream
  `e6df57f Update npu3 fw + umq_version`. Conflict-free.
- DKMS install rebuilt from `/usr/src/xrt-amdxdna-2.23.0/`,
  auto-signed with our MOK key. Verified with `modinfo` showing the
  new `srcversion: 60F700EDCD38AC6592975B5`.
- After reboot, `modprobe amdxdna` brings up the patched module
  automatically; no manual signing or insmod needed.
