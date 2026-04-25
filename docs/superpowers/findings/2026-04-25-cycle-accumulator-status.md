# A.5 Findings: cycle accumulator + EMU_SECONDS_PER_CYCLE + HW clock (2026-04-25)

## Summary

All three A.5 sub-items investigated. Net result: **no code change
needed**, plus one piece of follow-up reframed.

## A.5a: EMU per-instruction cycle accumulator

**Already TableGen-driven and deterministic.**

`src/interpreter/execute/cycle_accurate.rs:86-106` documents that
the executor uses `arch_handle::latency_table()`, which is populated
once at process start from llvm-aie's TableGen scheduling model
(278+ itinerary classes). `operation_cycles(op)` looks up each slot
op's latency from this table. Hazard detection (`HazardTracker`)
applies the latencies for pipeline stall accounting.

Unit tests confirm latencies match the AIE2 ISA:
- Scalar add: 1 cycle (`cycle_accurate.rs:686`)
- Scalar mul: 2 cycles (`cycle_accurate.rs:707`)
- Memory load: 5 cycles (`cycle_accurate.rs:739`)

Empirical determinism check (2026-04-25): two back-to-back
`emu-bridge-test.sh --no-hw -v cascade_flows --chess-only` runs both
produced `XDNA_EMU_STATUS: halt_reason=completed cycles=2256` --
identical to the cycle. The accumulator is bit-deterministic for this
test, which it should be for any non-multithreaded run.

**Action:** none. Already in the desired state.

## A.5b: EMU_SECONDS_PER_CYCLE retirement

**Reframed: it's a wallclock-timeout parameter, not in the cycle-diff
path. Already correctly scoped.**

`scripts/emu-bridge-test.sh:99-105`:

```bash
# EMU_SECONDS_PER_CYCLE: wall-clock seconds per simulated cycle. Emulator's
# reported simulation rate is ~800 MHz (pessimistic) to ~1 GHz; 2e-9 s/cycle
# = 500 M sim-cycles/sec is a conservative starting value...
EMU_SECONDS_PER_CYCLE=${EMU_SECONDS_PER_CYCLE:-2e-9}
```

Used at line 1828 only:

```bash
_timeout_s="$(awk -v c="$_hw_cycles" -v m="$EMU_CYCLE_BUDGET_MULTIPLIER" -v s="$EMU_SECONDS_PER_CYCLE" \
    'BEGIN{ t=c*m*s; if (t<600) t=600; printf "%d", t + 0.5 }')"
```

Pure wallclock timeout calculation: `t = HW_cycles * MULTIPLIER *
sim_seconds_per_cycle`, with a 600s floor. Doesn't influence cycle
counts, ratios, or the MATCH/DRIFT classifier.

The Phase E validation doc (`2026-04-23-phase-e-validation.md`) said
the constant was `1e-3` (un-calibrated). The current value is `2e-9`
(matches a 500 MHz sim rate, which is plausible for Rust release
builds on this machine).

**Action:** none -- the constant is already scoped correctly. Update
the Phase E validation doc's "Tuning opportunities" section to drop
the `EMU_SECONDS_PER_CYCLE` calibration item, since the current value
is reasonable and the 600s floor dominates anyway.

## A.5c: XRT HW clock query

**Available via `DRM_AMDXDNA_QUERY_CLOCK_METADATA`; not blocking.**

Driver ioctl path (`xdna-driver/include/uapi/drm/amdxdna_accel.h`):

```c
struct amdxdna_drm_query_clock_metadata {
    struct amdxdna_drm_query_clock mp_npu_clock;
    struct amdxdna_drm_query_clock h_clock;
};
struct amdxdna_drm_query_clock {
    char     name[16];
    __u32    freq_mhz;
    ...
};
```

Shim wrapper at `xdna-driver/src/shim/device.cpp:1378-1400` exposes
this as a `clocks` vector with `m_freq_Mhz` per clock. XRT C++ API
accesses it via `xrt::device::get_info` queries.

But: the cycle-diff classifier compares HW cycles to EMU cycles
*directly*, both dimensionless. We don't need HW clock for ratio
classification. Where it would matter:
- Translating HW cycles to wallclock-microseconds in human-readable
  output ("kernel ran for 41.18 us").
- Sanity-checking expected cycle ranges against a configured
  frequency.

**Action:** record as future work for any UI/display layer that wants
wallclock-time. Not blocking the validation pipeline.

## What this means for thread A

A.5 is closed without code changes. The two future-work bits:

1. **Phase E validation doc cleanup**: drop the
   `EMU_SECONDS_PER_CYCLE` calibration TODO; the constant is fine.
2. **HW clock display**: defer to whenever we build a
   wallclock-rendering UI on top of cycle data.

The determinism finding (cycles=2256 exact across two runs) supports
the broader picture: when EMU and HW disagree, it's not because the
EMU is non-deterministic. Any DRIFT in the cycle-diff is genuine
divergence between the two cycle accountings, worth investigating in
A.2 or follow-up tickets.
