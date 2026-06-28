# validate-readback

Throwaway harness that probes whether `xrt::hw_context::read_aie_reg`
returns real, wall-time-correlated, round-trip-honest data on Phoenix
NPU1 -- a sanity check before building calibration on top of it.

Spec: `docs/superpowers/specs/2026-05-06-validate-read-aie-reg-design.md`
Plan: `docs/archive/plans/2026-05-06-validate-read-aie-reg.md`

## Build

From the xdna-emu root:

```bash
cmake -S tools/validate-readback -B tools/validate-readback/build
cmake --build tools/validate-readback/build
```

## Run

```bash
./tools/validate-readback/build/validate-readback [-v]
```

By default, uses the peano build of `add_one_using_dma`. Override with
`--xclbin <path> --insts <path>`.

The harness writes AIE registers (PERF_CTRL0 / PERF_COUNTER0). Stock
xdna-driver gates that on `CAP_SYS_ADMIN`; in this dev tree the
`AMDXDNA_DEV_UNSAFE_USER_WRITES` opt-in (in
`xdna-driver/src/driver/amdxdna/amdxdna_drm.c`) drops the gate, so any
user can run this binary without `pkexec`. With a stock driver, run as
root.

## Tests

| ID | What it proves | Status |
|----|----------------|--------|
| L0 | Pre-launch `read_aie_reg` fails (lifecycle bug confirmed)             | PASS |
| L1 | Post-warmup pre-launch read works (option A workaround viable)        | PASS |
| V0 | TIMER_LOW advances at a sane wall-time-correlated rate                | PASS |
| V1 | Write-and-read-back is honest (round-trip exact)                      | PASS |
| V2 | col/row indexing reaches the addressed tile (cross-tile distinct)     | SKIP |
| V3 | Counter magnitude is plausible vs kernel_us * 400 MHz                 | SKIP |
| V4 | Same kernel run twice yields close counter values                     | SKIP |

V2/V3/V4 require a second kernel run on the same hwctx with the perf
counter freshly programmed; that second run hangs with
`ERT_CMD_STATE_NORESPONSE` because the `add_one_using_dma` compute
core hits `aie.end` after its 4 iterations and is halted, and a
subsequent `run.start()` does not reset the core. Working around this
(probably by recreating the hwctx per measurement) is out of scope for
this validation pass; the path-honesty question was answered by L0-V1.

Exit code = number of FAIL verdicts. SKIP does not contribute.

## Findings

Captured by running this harness against Phoenix NPU1 on 2026-05-06:

- **`read_aie_reg` returns real wall-time-correlated data** through the
  XRT C++ API, end-to-end. Reads do NOT require root.
- **`write_aie_reg` requires either root or our driver opt-in**
  (`AMDXDNA_DEV_UNSAFE_USER_WRITES = 1` in `amdxdna_drm.c`). With the
  opt-in, any user can write AIE registers.
- **Lifecycle bug confirmed**: pre-launch `read_aie_reg` /
  `write_aie_reg` fail with `EINVAL` until the first `run.start()` has
  allocated the partition. Warmup-run workaround (option A) is viable.
- **Tile clock is gated when idle**: `TIMER_LOW` advances at ~12 MHz
  effective when no kernel is running, vs ~400 MHz core clock.
  Informational; means TIMER_LOW deltas across kernel runs reflect
  active core time, which is what we want.
