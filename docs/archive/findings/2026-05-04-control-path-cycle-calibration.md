---
name: control-path cycle calibration provisional results
description: Empirical fast-mode integer values for AIE_CONTROL_PATH_LATENCY schema, the artifacts we couldn't characterise, and why on-NPU timing failed. Posterity record so the next person doesn't re-derive everything.
type: project
---

# Control-path cycle calibration -- provisional results

> **2026-05-05 update**: the "read_aie_reg is firmware-blocked on
> Phoenix" conclusion at the bottom of this document is **wrong**.
> Phoenix firmware DOES implement `MSG_OP_AIE_RW_ACCESS`; the only
> obstacle was a missing entry in `npu1_regs.c`'s op-table. See
> `2026-05-05-aie-rw-access-firmware-actually-supported.md` for the
> correct picture and the one-line driver fix. The calibration data
> itself (write32 / blockwrite / maskwrite per-packet costs, the two
> artifact classes) is unaffected and remains valid.

## TL;DR

Calibration sweeps against real NPU1 hardware nailed down clean
**fast-mode integer per-packet costs** for write32 / blockwrite /
maskwrite. They also revealed two artifact classes (one-time event
+90-130 cyc, stochastic slow-mode +2780 cyc) that we cannot definitively
attribute to either the trace controller or real CMP behaviour without
on-NPU timing -- which our toolchain doesn't currently support.

The clean values are loaded into `CycleCostModel::provisional_npu1()`
with appropriate caveats. The artifact classes are documented but **not**
modelled, on the working hypothesis that they are measurement-side.

## Calibration tools

Built under `tools/calibration/`:

- `gen_kernel.py` -- parameterised calibration kernel. Generates an
  MLIR runtime sequence with N control packets between two trace anchors
  (`USER_EVENT_0` / `USER_EVENT_1`). Knobs: `--kind {write32, blockwrite,
  maskwrite}`, `--count N`, `--target-col/row`, `--anchor-col/row`,
  `--ticker-period` (compute core ticker rate), `--no-trace` (strip
  trace controller for external timing).
- `run_sweep.py` -- compiles + runs a sweep on real HW, parses traces,
  records `hw_cycles` per measurement.
- `run_sweep_notrace.py` -- same idea but uses `--no-trace` kernels and
  reads bridge-runner's wall-clock `kernel_us` instead. **Wall-clock
  resolution is insufficient** to see NPU-cycle-level effects; kept as
  scaffolding in case we ever get a finer external timer.
- `sweep_gen.py` -- programmatic sweep config generator.
- `analyze_sweep.py` -- least-squares per-(kind, target, anchor, payload)
  slope fits.
- `analyze_dense.py` -- BIC-scored modular-period detection for
  contiguous count sweeps. Reveals integer-valued period-2 structure in
  write32 deltas that geometric-count slopes hide.
- `analyze_bimodal.py` -- gap-clustering for high-rep sweeps; isolates
  fast/slow cycle distributions.

Every sweep config used is checked into `tools/calibration/sweeps/`.

## Headline empirical model (NPU1, fast mode only)

| Packet kind             | Per-packet cost | Notes |
|-------------------------|-----------------|-------|
| `write32`               | **100.5 cyc**   | period-2: 87 cyc (even N), 114 cyc (odd N). Per-pair cost = 201 cyc, integer-exact. |
| `blockwrite (payload=8)`| **203 cyc**     | period-1, no alternation. |
| `maskwrite`             | **~210 cyc**    | high intrinsic per-rep noise (~50 cyc spread); period structure not resolved. |

Derived per-word cost for blockwrite: `(203 - 100.5) / 8 ≈ 13 cyc/word`.
Conservative lower bound; the per-word constant from the
`AIE_CONTROL_PATH_LATENCY` schema is what AMD uses internally.

Distance from anchor tile is **free**: 24-tile sweep across the npu1 4x6
array showed slopes within 1.34 cyc spread, per-hop fit -0.05 cyc with
R^2=0.03 (noise). Per-tile-type write32 costs were within 0.5 cyc across
shim/mem/compute targets.

## The two artifact classes (NOT modelled)

### Event 1: one-time +91 cyc step at write32 N=23

Cycle count jumps by ~+91 between N=22 and N=24 (split: +56 at N=23,
+37 at N=24), then period-2 resumes with the offset baked in. **100%
deterministic** across 20 reps. Same shape phenomenon at blockwrite
N=12 (+34 cyc) and approximately the same place for maskwrite (hidden
in noise).

### Event 2: one-time +127 cyc step at write32 N=108-109

Visible only inside slow mode (since fast mode is gone by N=108 -- see
below). +60 at N=108, +67 at N=109, then stable.

### Slow-mode artifact: stochastic +2780 cyc

Bimodal cycle distribution from N~35 (write32) onwards. The "slow"
cluster is exactly +2780 ± 30 cyc above the "fast" cluster. Probability
ramps from 4% at N=35 to 100% at N=80. Once a run is in slow mode it
stays there; reverts ARE observed at higher N but do not become more
common.

### Cycle-threshold pattern

The smoking gun for "this is measurement-side, not real CMP":

| Kind         | Event 1 trigger | Event 1 cycles | Slow-mode first | cycles |
|--------------|-----------------|----------------|-----------------|--------|
| write32      | N=23            | 2469           | N=35 (4%)       | 3505   |
| blockwrite   | N=12            | 2552           | N=21            | 4263   |
| maskwrite    | N=~12           | ~2400          | N=17            | 3570   |

**Events fire around 2400-2600 NPU cycles, slow mode first appears
around 3500-4300 cycles, regardless of packet kind.** That's the
signature of a *concurrent, cycle-clock-driven subsystem* (trace
controller flush timer, DMA tick, similar) -- not the signature of
CMP-architectural state.

### Negative results from Option 2 diagnostic tests

- **Trace BO size 4x change** (65536 -> 262144): threshold barely moved
  (N=38 -> N=41). Rules out simple buffer overflow.
- **Compute core ticker rate 4x range** (64 / 256 / 1024 inner-loop iters
  per `INSTR_EVENT_0` fire, i.e. 16x range of trace event volume): event
  N and first-slow N are *identical* across all three. Rules out
  event-volume-driven trace flushing.

Remaining viable explanations for the artifact:
1. Trace controller has internal cycle-driven timer that fires regardless
   of event volume.
2. Real CMP architectural cycle-driven behaviour.
3. Something else entirely.

We cannot distinguish (1) from (2) from this side of the trace path.

## Why Option 1 (on-NPU timing without trace) didn't ship

Two attempts:

### Attempt A: bridge-runner reads Performance_Counter0 via patched XRT

mlir-aie's `runtime_lib/test_lib/cycle_counter.h` documents the recipe:
program `PERF_CTRL0` (offset 0x31500) on each compute tile to count
`ACTIVE_CORE` (event id 0x1C), zero `PERF_COUNTER0` (offset 0x31520),
read it back after `run.wait()`. Uses the `xrt::hw_context::read_aie_reg`
/ `write_aie_reg` methods that the patched XRT exposes.

Wired up in `bridge-trace-runner.cpp` behind `--read-perf-counter`. The
register-reads fail with `DRM_IOCTL_AMDXDNA_GET_ARRAY` returning EINVAL
(setup) and EOPNOTSUPP (post-wait readback).

**Correction (2026-05-05):** the original "no open-source definition"
diagnosis below was wrong about the userspace path but right about the
endpoint being unreachable on NPU1 -- just for a different reason.

What actually happens (verified 2026-05-05 with `strace -e ioctl` +
verbose driver dmesg):

The XRT shim and kernel paths are fully open. `xrt::hw_context::read_aie_reg`
in the patched XRT routes to `xrt_core::query::aie_read` (open shim
code in `xdna-driver/src/shim/device.cpp:1523`), which builds a
`DRM_AMDXDNA_AIE_TILE_READ` (param=9) `amdxdna_drm_get_array` ioctl
and submits it. The kernel dispatcher routes through
`aie2_get_array` -> `aie2_aie_tile_read` (open, in `aie2_pci.c:1564`)
-> `aie2_rw_aie_reg` (open, in `aie2_message.c:1765`).

`aie2_rw_aie_reg` calls
`aie2_is_supported_msg(ndev, MSG_OP_AIE_RW_ACCESS)`, which iterates
the per-device firmware-feature table:

- `npu1_regs.c` (Phoenix): table contains only `MSG_OP_CHAIN_EXEC_NPU`
  at FW 5.8. **No `MSG_OP_AIE_RW_ACCESS` entry** -> returns false ->
  `aie2_rw_aie_reg` returns EOPNOTSUPP at the mailbox layer.
- `npu4_regs.c` (Strix): contains
  `{ AIE2_FW_VERSION(6, 24), MSG_OP_AIE_RW_ACCESS }` -> Strix firmware
  >= 6.24 supports it.

So this is a **firmware-level limitation on Phoenix**, not closed
source and not a userspace bug. AMD never shipped Phoenix firmware
that implements opcode 0x203 (`MSG_OP_AIE_RW_ACCESS`). On NPU4 the
path will work end-to-end.

We additionally hit a setup-time EINVAL on the *first* call because
the hwctx's partition (`hwctx->num_col`) is allocated lazily on first
kernel run, so calling `read_aie_reg` before `run.start()` reaches the
"Column %u outside partition range [0, 0)" check at `aie2_pci.c:1635`.
The post-wait call gets past that (partition allocated) and is the one
that hits the firmware-support EOPNOTSUPP. Both errors are real, but
the firmware one is the showstopper for NPU1.

The original (wrong) text follows for posterity:

> The XRT shim correctly dispatches `DRM_AMDXDNA_AIE_TILE_READ` (the
> un-guarded path), and the loaded `amdxdna.ko` does export
> `aie2_rw_aie_reg`, but the wrapper function `aie2_read_aie_reg` (called
> by the dispatch case) has **no open-source definition** in our
> `xdna-driver` checkout.

That wrapper belongs to the *legacy* gated path
(`DRM_AMDXDNA_READ_AIE_REG`). The new path uses
`aie2_aie_tile_read` (open). The actual block is the firmware feature
table, not anything closed.

### Attempt B: lock-synchronised compute-core timing kernel

Designed but not built. Compute core polls a "start" memory write,
reads its own `TIMER_LOW` register (`0x340F8`), waits for "stop" memory
write, reads timer again, writes the delta to a buffer that an
`objectFifo` pushes back to host. Real engineering work (~3-6 hr) and
introduces enough new on-NPU code that its own measurement overhead
becomes a confound. Deferred unless we get a downstream reason to
build it.

## Decision for the cost model

Use the **fast-mode integer values** from our calibration as the
provisional CycleCostModel for NPU1, integer-rounded. Do NOT model the
event jumps or slow-mode penalty -- on the working assumption that they
are trace-controller artifacts that wouldn't apply when no trace is
configured. If that assumption is wrong, the kernels affected are
all small (<100 control packets) and the model under-counts by at most
a few thousand cycles total.

This is logged in `cycle_cost.rs::provisional_npu1()` with caveats
strongly worded enough that a future reader will know what they're
looking at.

## What it would take to ship a *good* model

Either:

1. AMD ships the `AIE_CONTROL_PATH_LATENCY.{AIE_TILE_BD, MEM_TILE_BD,
   SHIM_TILE_BD, WRITE_32, ...}` calibration JSON that
   `libaie2_cluster_msm_v1_0_0.osci.so` reads -- the schema is already in
   place, only the numeric values are gated by their internal config.
2. Someone wires up Option 1 (Attempt B) properly: synchronised compute
   core timing kernel, run it across all kinds and tile types, derive
   integer values from clean measurements.

Either path produces ground truth. Without one of them, our values are
the best we can do given the toolchain.

## See also

- `2026-05-01-aietools-control-path-latency-schema.md` -- earlier
  string-extraction-derived schema discovery.
- `tools/calibration/` -- the harness, all sweep configs, all analysers.
- `tools/calibration/sweeps/` -- every sweep config we've actually run
  is checked in for reproducibility.
