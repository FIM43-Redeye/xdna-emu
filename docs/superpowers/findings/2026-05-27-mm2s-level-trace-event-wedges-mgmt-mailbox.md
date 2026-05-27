---
name: 'MM2S-side level trace events wedge the amdxdna mgmt mailbox on Phoenix'
description: 'Adding any MM2S-direction level event (STREAM_BACKPRESSURE, STALLED_LOCK tested) to the shim trace event slot triggers the amdxdna mgmt mailbox to stop after ~30s of sustained kernel dispatch. Old default config (S2MM_1_STREAM_STARVATION in slot 7) and S2MM-direction level events (STREAM_STARVATION_0) run indefinitely without wedging. Implication: trace-event-based stall decomposition cannot use MM2S-side level events on Phoenix without losing campaign reproducibility, blocking that as a calibration source for the dispatch-overhead warm-up-transient question.'
type: project
---

# MM2S-side level trace events wedge the amdxdna mgmt mailbox on Phoenix

## TL;DR

While trying to decompose MM2S transfer-duration variation by adding
`DMA_MM2S_0_STREAM_BACKPRESSURE` to the shim trace event slot, the
amdxdna mgmt mailbox started stopping ~30s into sustained K-sweep
campaigns. Replacing the same slot with `DMA_MM2S_0_STALLED_LOCK`
(different event family, still MM2S-direction, still level) reproduced
the wedge with identical signature. Reverting to the upstream default
event list (S2MM_1_STREAM_STARVATION in the slot) restored stable
operation: 80/80 OK in 67.5s.

The driver-visible failure is
`DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-22)` with dmesg
showing `aie_send_mgmt_msg_wait: command opcode 0x2 failed,
status 0x2000003` and then `Mailbox channel stopped, irq: 145`.
Single-shot `bridge-trace-runner` invocations after the wedge still
succeed (device recovers without driver reload), so the failure
mode is per-campaign-burst rather than persistent.

Bottom line: **MM2S-direction level events cannot be added to the
shim trace event slots on Phoenix without forfeiting campaign
reproducibility.** S2MM-direction level events are safe; the
upstream default has STREAM_STARVATION_0 and _1 in two slots and
ran 200/200 OK in the Phase 2c calibration campaign.

## Reproduction

Tooling:

- Trace event swap via the new `XDNA_TRACE_SHIM_EVENTS` env var
  added to `scripts/emu-bridge-test.sh` in `fc31c3f`.
- Rebuild: `./scripts/emu-bridge-test.sh --compile --no-hw --no-emu
  '_diag_shim_chain_sweep/k[1248]$'`.
- Campaign: `python3 tools/multirun-trace-campaign.py --n-runs N
  --session SESSION` (multirun runs `bridge-trace-runner` once per
  K-sweep iteration; each invocation does its own `CREATE_HWCTX`).

Driver and firmware (consistent across all campaigns in this
finding):

- amdxdna `drivers/accel` tree, `0620bb6ae29c` (out-of-tree DKMS
  module `xrt-amdxdna/2.23.0`)
- Phoenix FW 1.5.5.391
- Kernel 7.0.9-custom

## Differential evidence

### Control: upstream default config (no slot swap)

```
shim slot 0: DMA_S2MM_0_START_TASK
shim slot 1: DMA_S2MM_1_START_TASK
shim slot 2: DMA_MM2S_0_START_TASK
shim slot 3: DMA_S2MM_0_FINISHED_TASK
shim slot 4: DMA_S2MM_1_FINISHED_TASK
shim slot 5: DMA_MM2S_0_FINISHED_TASK
shim slot 6: DMA_S2MM_0_STREAM_STARVATION   (level, S2MM-direction)
shim slot 7: DMA_S2MM_1_STREAM_STARVATION   (level, S2MM-direction)
```

Campaigns observed under this config:

| Session | N | Runs | OK | Fail | Wall |
|---|---:|---:|---:|---:|---:|
| `2026-05-27T04-19-33` | 50 | 200 | 200 | 0 | 5:02 |
| `2d-wedge-ctrl`      | 20 |  80 |  80 | 0 | 1:07 |

### Test 1: swap slot 7 -> DMA_MM2S_0_STREAM_BACKPRESSURE

```
shim slot 7: DMA_MM2S_0_STREAM_BACKPRESSURE  (level, MM2S-direction)
```

| Session | N | Runs attempted | OK before first fail | Time to first fail |
|---|---:|---:|---:|---:|
| `2d-mm2s-bp`      | 50 | 63 | 31 | 27.7s |
| `2d-wedge-repro`  | 20 | 59 | 29 | 25.5s |

Both campaigns aborted on the multirun script's "more failures than
successes" guard (`n_fail >= 5 and n_fail > n_ok`). Failure mode is
all-or-nothing: every run after the first failure also fails. dmesg
during failure:

```
amdxdna 0000:c6:00.1: [drm] *ERROR* aie_send_mgmt_msg_wait:
  command opcode 0x2 failed, status 0x2000003
amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_xrs_load:
  create context failed, ret -22
amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_alloc_resource:
  Allocate AIE resource failed, ret -22
amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_hwctx_init:
  Alloc hw resource failed, ret -22
amdxdna 0000:c6:00.1: [drm] *ERROR* amdxdna_drm_create_hwctx_ioctl:
  Init hwctx failed, ret -22
...
amdxdna 0000:c6:00.1: xdna_mailbox.145: Mailbox channel stopped, irq: 145
```

opcode `0x2` is the mgmt-mailbox CREATE_HWCTX request to firmware;
the FW returns status `0x2000003` (high bit set, error code 3),
which the driver collapses to `-EINVAL` and the kernel exposes as
EINVAL to userspace. After ~13s of repeated failures the mailbox
channel itself stops, indicating the FW gave up servicing the
channel rather than returning piecemeal errors.

### Test 2: swap slot 7 -> DMA_MM2S_0_STALLED_LOCK

```
shim slot 7: DMA_MM2S_0_STALLED_LOCK   (level, MM2S-direction, different family)
```

| Session | N | Runs attempted | OK before first fail | Time to first fail |
|---|---:|---:|---:|---:|
| `2d-wedge-lock` | 20 | 67 | 33 | ~30s |

Same failure mode, same dmesg signature, similar timing. Different
event family from the BACKPRESSURE case (STALLED_LOCK is a
lock-acquire-stall level signal, not a stream-backpressure level
signal), so the wedge is not specific to STREAM_* events.

### Cross-test pattern

|                                | OK before fail | Time to first fail | Outcome |
|--------------------------------|---:|---:|---|
| S2MM_1_STREAM_STARVATION (level, S2MM) | --  | n/a | clean 200/200 |
| MM2S_0_STREAM_BACKPRESSURE (level, MM2S, x2 runs) | 29-31 | 25.5-27.7s | wedge |
| MM2S_0_STALLED_LOCK (level, MM2S)      | 33 | ~30s | wedge |

The constant factor across wedging configs is **MM2S-direction
level event** in a shim trace slot. The non-wedging configs all
have S2MM-direction level events (or no level events). Edge events
on MM2S (the existing `DMA_MM2S_0_START_TASK` and
`DMA_MM2S_0_FINISHED_TASK` slots) coexist fine.

## Recovery characteristics

After a campaign wedge:

- The mailbox shows "stopped, irq: 145" in dmesg. No further
  failures are logged unless another campaign starts.
- Single-shot `bridge-trace-runner` invocations succeed within
  seconds of the campaign abort: the device returns to a working
  state on its own without driver reload, SBR, suspend, or any
  recovery operation from the host.
- A subsequent campaign with the same wedging trace config
  reproduces the wedge at similar timing (~30 OK, ~26s).

So the failure is **per-campaign-burst** and recovers without
intervention, but reliably re-triggers under sustained dispatch
with an MM2S-side level event in the trace slot.

## Hypothesis (untested)

The wedge looks like a FW-internal resource leak or counter
overflow tied to MM2S-side level event broadcasting. Possible
mechanisms (not investigated):

1. Each `CREATE_HWCTX` call programs the shim trace pipeline,
   including the event broadcast network routing for the configured
   slot events. If MM2S-direction level broadcast lines use a
   resource that isn't freed on context destroy (e.g., a hardware
   counter, a routing entry), repeated CREATE/DESTROY cycles
   accumulate state until the resource is exhausted and the FW
   refuses new contexts.
2. MM2S-direction level events may share a broadcast line or
   internal FW signal with something the runtime PM / clock-gating
   path also touches, and the trace routing collides with that
   path after enough cycles.
3. Status `0x2000003` is the same shape as the SMU error responses
   documented in `CLAUDE.md` (high bit + low-byte error code). If
   the mgmt mailbox shares the same response convention, `0x3`
   might decode to a specific FW error -- possibly "resource
   busy" or "no free slots".

The "per-campaign-burst" recovery (no driver reload needed)
suggests it's transient FW state, not persistent damage. A kernel
timer or PM idle event probably resets whatever counter is wedged.

## Implications

### For Phase 2d / stall decomposition

MM2S-side level events are off the table as a calibration source
on Phoenix. The MM2S transfer-duration variation (`K=8`: 1739->370
cyc over 8 tasks) cannot be decomposed via trace-event sampling
on MM2S; the only signals available there are edge events
(START_TASK / FINISHED_TASK / FINISHED_BD), and those don't
expose stall mechanics.

S2MM-side level events (STARVATION_*, MEMORY_BACKPRESSURE_*,
STALLED_LOCK_*, PORT_RUNNING_*) remain usable.

### For future trace experiments

`XDNA_TRACE_SHIM_EVENTS` env var (committed in `fc31c3f`) supports
arbitrary slot lists. **Don't include MM2S-direction level events
in any list intended for sustained-campaign use.** Edge events
(`*_START_TASK`, `*_FINISHED_TASK`, `*_FINISHED_BD`) on MM2S are
safe.

### For the calibration mission

This forces the dispatch-overhead warm-up-transient question
either to Phase 2e (FW_TRACE / DPT framework, which would peek
inside the controller directly) or to empirical modeling against
the existing per-task duration curves we already have from the
Phase 2c HW campaign.

## What we didn't test

- Other MM2S-direction level events: `MEMORY_STARVATION`,
  `MEMORY_BACKPRESSURE` variants. Predicted to wedge based on the
  pattern but unconfirmed.
- MM2S level events in slots other than slot 7 (might be slot-
  specific, though no evidence supports that).
- MM2S level events on channels other than `_0` (e.g., `_1`).
- Whether the wedge happens on memtile/core MM2S level events
  too, or is shim-specific.
- The mgmt-mailbox response status `0x2000003` decoded against
  any FW response code table -- the code was not located in the
  amdxdna driver source.

These would refine the model but are not required to act on the
"don't use MM2S level events in trace slots" finding.

## Artifacts

- Campaign manifests:
  `build/experiments/dispatch-overhead-multirun/{2d-mm2s-bp,
  2d-wedge-repro, 2d-wedge-lock, 2d-wedge-ctrl}/manifest.json`
  (each gitignored; reproducible from this finding).
- Trace event swap mechanism: `scripts/emu-bridge-test.sh`
  (env var `XDNA_TRACE_SHIM_EVENTS`, commit `fc31c3f`).
- Driving question: dispatch-overhead Phase 2d in
  `docs/coverage/aie2/implementation-gaps.md` (calibration
  question that motivated trying MM2S backpressure).
