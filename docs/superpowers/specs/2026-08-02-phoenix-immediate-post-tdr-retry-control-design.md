# Phoenix Immediate Post-TDR Retry Control

**Date:** 2026-08-02
**Status:** Approved boundary, pending written review

## Goal

Characterize the driver-reachable retry contract after the pinned finite A
fixture times out. Submit A1, A2, and A3 through one live `xrt::hw_context`,
without constructing B, replacing the context handle, or adding an artificial
recovery delay.

The control observes whether an application can resubmit as soon as A2's
`run.wait()` returns a non-completed state. It does not assume that the driver's
DESTROY/CREATE/MAP/CONFIG replay has finished before userspace wakes.

## Producer

Extend `tools/phoenix-vfio-user/context-repartition.cpp` with one explicit
`--immediate-post-tdr-retry` mode while preserving both existing modes and
their output strings.

1. Construct one A `Workload` from the frozen Chess `add_one_using_dma`
   artifacts.
2. Run A1 and require its ordered `2..=65` output.
3. Refill the same buffers and submit A2. Require a non-completed state and
   print that state without throwing away the `Workload`.
4. Immediately refill the same buffers and submit A3 through the same kernel
   and hardware-context objects.
5. If A3 completes, verify `2..=65`; otherwise record its returned state.
6. Destroy A once after the observation.

A2 completing unexpectedly stops before A3 because the intended TDR boundary
was not reached. A3 completion and noncompletion are both valid semantic
outcomes. Distinct markers must make infrastructure failure, unexpected A2
completion, A3 success, and A3 noncompletion mechanically separable.

No sleep, trace-derived userspace gate, debugfs polling, retry, second context,
or private driver operation is permitted. The qualified trace determines
whether A3 queues during recovery or publishes after firmware replay.

## One Physical Transaction

Perform exactly one run on the same pinned NPU1 tuple as the same-context A1/A2
control. A single trapped `pkexec` transaction must:

- reject active NPU clients and pin every input hash before mutation;
- save the normal module identity and current PCI power policy;
- load `amd_pmf` and the qualified response-trace `amdxdna` module with
  `tdr_timeout_ms=2000` and `force_cmdlist=Y`;
- create a private tracefs instance and capture stdout, stderr, the complete
  mailbox/job trace, and the dmesg delta;
- run the producer as the unprivileged `triple` account;
- perform no retry; and
- restore and independently verify the normal module, device node, absence of
  holders, and original power policy on every exit path.

A2 intentionally causes one normal TDR. If A3 also fails to complete, the same
single producer invocation may cause a second normal TDR. That is part of the
approved observation, not permission for another invocation.

## Interpretation

- **A3 completes with correct output:** the public-XRT immediate-retry path
  regains an executable context after the timeout/replay lifecycle.
- **A3 is submitted but does not complete:** the trace pins the first
  host-visible boundary at which immediate retry remains unavailable.
- **A3 never reaches a device execute request:** the driver-side recovery and
  queue lifecycle, not signed-firmware execution, is the next boundary.
- **A2 completes, infrastructure fails, evidence is lost, or restoration
  fails:** no post-TDR retry conclusion is licensed and the run is not retried.

The trace may establish command order and external responses. It cannot by
itself establish the AIE core's internal state.

## Unattended Follow-Through

After the privileged transaction is complete and restoration is verified, all
remaining work is unprivileged. Reproduce the observed command ordering first
against the signed-firmware in-process seam and then through the pinned KVM
driver path. Use Halo only for a genuinely heavy build.

If a first divergence is forced by the open driver, signed firmware, aie-rt,
mlir-aie, or llvm-aie, add one focused red test and correct the shared hardware
seam. Do not add driver policy, a response shim, an arbitrary delay, or a
synthetic completion. If the required behavior is not derivable, stop with a
finding rather than guess. No later `pkexec`, module change, or privileged host
mutation is permitted while Maya is asleep.

## Verification

Before physical traffic:

- demonstrate that the old producer rejects the new CLI mode;
- compile the changed producer warning-clean against pinned XRT 2.26;
- prove the two existing CLI modes still select their original paths; and
- pin the producer source and binary hashes into the campaign preflight.

After the run, mechanically reduce the A1/A2/A3 command and response counts,
seal all evidence with SHA-256, verify restoration live, and update the context
finding and fidelity ledger. Any model correction additionally requires its
focused test and `nice -n 19 cargo test --lib`.
