# Phoenix Immediate Post-TDR Retry Control

**Date:** 2026-08-02
**Status:** Implemented; one physical observation sealed

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

## Result

The one approved invocation is sealed under
`build/experiments/npu1-firmware-evidence/physical-immediate-post-tdr-retry-20260802-01/`.
A1 completed and A2 returned state 8 after TDR. A3 was then received, queued,
run, and freed by the driver after recovery CREATE responded but before the
MAP and CONFIG replay. It has no `sent to device` tracepoint and no execute
mailbox tail. XRT reported an unexpected command state and the producer exited
1. The complete trace retained 65 of 65 entries, and the normal module and
power policy were restored and verified.

The primary-driver source at exact commit
`216cefececd74effcd7a88350c71b99f5ef9a215` explains the boundary:
`aie2_hwctx_stop()` restarts the scheduler after destroying the context, while
the timeout callback performs CREATE/MAP/CONFIG later through
`aie2_hwctx_restart()`. A scheduler job returns before `sent to device` while
the destroyed context's mailbox channel remains null. The exact branch is a
source-constrained inference because the tracepoint is after the early return.

## Follow-Through Decision

The planned signed-firmware and KVM reproduction does not apply to the observed
outcome: A3 never crossed the driver/firmware seam. Inventing an execute request
would test a different contract. No emulator change is licensed.

Same-handle retry after CREATE/MAP/CONFIG replay remains a separate unobserved
boundary. Characterizing it requires either a deliberate post-replay gate or a
driver recovery correction, not an immediate-retry rerun.

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
