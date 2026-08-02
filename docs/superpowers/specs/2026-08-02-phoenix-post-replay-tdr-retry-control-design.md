# Phoenix Post-Replay TDR Retry Control

**Date:** 2026-08-02
**Status:** **IMPLEMENTED AND OBSERVED; CONTROL PASSED**

## Goal

Characterize the same-handle execute contract after the qualified Phoenix
driver has completed its TDR recovery replay. Run the frozen finite A fixture
once, let its second execute time out, establish that CREATE/MAP/CONFIG replay
has returned, and submit A3 through the original XRT workload and hardware
context.

This control follows the immediate-retry observation, where A3 was consumed by
the restarted host scheduler before MAP/CONFIG and never reached firmware. It
must distinguish that driver race from post-replay firmware and array behavior
without using an arbitrary delay.

## Replay-Complete Barrier

Use the open driver uAPI's read-only
`DRM_AMDXDNA_QUERY_FIRMWARE_VERSION` request as the barrier. After A2's
`run.wait()` returns its non-completed state, the producer opens the explicitly
supplied accel device node, issues `DRM_IOCTL_AMDXDNA_GET_INFO`, records the
returned version, and closes the descriptor before submitting A3.

This is an ordering primitive on the pinned primary driver, not a timing
heuristic:

- `aie2_sched_job_timedout()` holds the device's `dev_lock` from before TDR
  handling until after `aie2_hwctx_restart()` returns;
- `aie2_hwctx_restart()` performs CREATE, MAP, and CONFIG in that order;
- opening a new DRM client takes `dev_lock` before publishing the client; and
- `amdxdna_drm_get_info_ioctl()` also takes `dev_lock` around the read-only
  firmware-version query.

The open and query therefore cannot finish until recovery has released the
same lock. On the pinned D0 tuple, the query copies the cached firmware-version
structure and does not create a hardware context or send a firmware mailbox
command. Opening the descriptor only after A2 returns keeps the A1/A2 path
identical to the prior same-context controls.

The barrier is valid only when `DEVICE` belongs to the same PCI function that
XRT selects as device 0. Preflight must resolve the accel class device through
sysfs and require PCI BDF `0000:c6:00.1`; a mismatch stops before module
replacement or physical traffic.

This reasoning is pinned to primary-driver commit
`216cefececd74effcd7a88350c71b99f5ef9a215`. The producer must compile against
that sibling tree's `include/uapi/drm/amdxdna_accel.h`, and the campaign must
record the header hash.

## Producer Contract

Extend `tools/phoenix-vfio-user/context-repartition.cpp` with one mode while
preserving the existing positional, `--same-context-repeat`, and
`--immediate-post-tdr-retry` modes and markers:

```text
--post-replay-tdr-retry DEVICE A.xclbin A.insts
```

The new mode performs this sequence:

1. Construct one A `Workload` from the frozen Chess `add_one_using_dma`
   artifacts.
2. Run A1 and require ordered output `2..=65`.
3. Refill the same buffers, submit A2, and require a non-completed state.
4. Open `DEVICE`, issue the firmware-version GET_INFO request, close it, and
   print the returned `major.minor.patch.build` as a barrier marker.
5. Refill the same buffers and submit A3 through the unchanged kernel and
   hardware-context objects.
6. If A3 completes, verify `2..=65`; otherwise record its returned state.
7. Destroy A once after the observation.

The mode has distinct markers for A1, A2, barrier success, A3 completion or
noncompletion, clean destruction, and infrastructure failure. A2 completing
unexpectedly stops before the barrier. Barrier failure stops before A3.

No sleep, polling loop, trace-derived producer gate, second hardware context,
driver patch, synthetic completion, or automatic retry is permitted.

## One Physical Transaction

Perform exactly one invocation on the same pinned NPU1 tuple as the immediate
retry control: qualified response-trace module, signed `1502_00` firmware, XRT
2.26 packages, frozen Chess A artifacts, `tdr_timeout_ms=2000`,
`force_cmdlist=Y`, and D0 power pin.

A trapped `pkexec` wrapper must:

- reject active clients and pin the source, binary, uAPI header, module,
  firmware, XRT, and fixture hashes before mutation;
- prove that the supplied accel node resolves to PCI BDF `0000:c6:00.1`, the
  function selected by XRT device 0 for this pinned single-NPU host;
- save the normal module identity and power policy;
- load `amd_pmf` and the qualified response-trace module;
- capture the complete mailbox/job trace, stdout, stderr, and dmesg delta;
- run the producer as `triple` with emulator environment variables removed;
- perform no retry; and
- restore and independently verify the normal module, device node, no holders,
  and original power policy on every exit path.

A2 intentionally causes one normal TDR. If A3 reaches firmware but does not
complete, the same producer invocation may cause a second normal TDR. That is
part of the observation and does not authorize another invocation.

## Evidence and Interpretation

The qualified trace is the final ordering oracle. A licensed post-replay
result requires recovery CONFIG's response and head advance to precede A3's
job publication and execute tail.

- **A3 completes with correct output:** the same public XRT handle executes
  successfully after the driver's full recovery replay on the pinned tuple.
- **A3 has an execute tail but no response:** the first remaining boundary is
  after firmware-visible publication, not the previously observed host
  scheduler race.
- **A3 is accepted but has no execute tail:** the driver remains the boundary;
  the GET_INFO barrier or a later host lifecycle requires correction before
  drawing a firmware conclusion.
- **A2 completes, the barrier fails, trace ordering contradicts the claimed
  gate, evidence is lost, or restoration fails:** the run licenses no
  post-replay conclusion and is not retried.

The trace cannot directly establish the AIE core's internal state. Any causal
claim about PDI reload, core reset, DMA state, locks, or firmware dispatch must
be derived separately from the signed-firmware/model path after this external
boundary is known.

## Observed Result

The campaign was invoked exactly once:

```text
build/experiments/npu1-firmware-evidence/physical-post-replay-tdr-retry-20260802-01/
```

A1 completed, A2 returned TDR state 8, the barrier returned firmware version
`1.5.5.391`, and A3 completed with correct output through the original XRT
workload and hardware context. The 72-of-72-entry private trace orders recovery
CREATE, CONFIG, and MAP responses and head advances before A3's job publication.
CONFIG's head advanced at `57752.154732`, MAP's at `57752.154828`, A3 was
received at `57752.154863`, and its execute tail was published at
`57752.154956`. A3's firmware response followed at `57752.155195`.

This licenses the narrow positive result: on the pinned Phoenix tuple, the same
public XRT handle executes successfully after the primary driver's complete TDR
replay. It does not identify which internal reset, PDI, DMA, lock, or core-state
effects make the finite fixture runnable again. Those remain the next
signed-firmware/model boundary.

`post-run-attestation.txt` and `SHA256SUMS` seal the complete receipts. Both the
wrapper and an independent live check verified restoration of the normal
module, exact BDF, device node, zero holders, and original power policy.

## Verification

Before physical traffic:

- prove the old producer rejects the new mode;
- add the mode through a focused RED/GREEN CLI check;
- compile warning-clean against pinned XRT 2.26 and the exact sibling uAPI;
- prove all three existing modes still select their original paths; and
- pin the producer source, binary, and uAPI header hashes in preflight.

After the run:

- mechanically reduce A1/A2/A3 job and mailbox counts;
- require the recovery CONFIG response/head to precede A3's execute tail;
- seal every campaign file with SHA-256;
- verify restoration live;
- update the context finding and fidelity ledger; and
- run `nice -n 19 cargo test --lib`.

No emulator or driver implementation change belongs in this slice. The result
determines the next firmware-model boundary.
