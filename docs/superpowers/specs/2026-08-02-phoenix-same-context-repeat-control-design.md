# Phoenix Same-Context Repeat Control

**Date:** 2026-08-02
**Status:** Implemented and physically observed

## Goal

Run the smallest physical discriminator missing from the Phoenix `A1 -> B ->
A2` observation: submit the identical finite A fixture twice through one live
`xrt::hw_context`, without ever constructing B.

This determines whether A2 already stalls in the control condition. It does
not attempt to explain the result or change emulator behavior.

## Producer

Extend the existing
`tools/phoenix-vfio-user/context-repartition.cpp` producer with one explicit
`--same-context-repeat` mode:

1. create one A `Workload` from the pinned Chess `add_one_using_dma` artifacts;
2. run and verify A1;
3. refill the existing buffers and submit A2 through the same kernel and
   hardware context;
4. destroy A once after A2 returns or throws; and
5. emit repeat-specific markers distinct from the historical A-B-A markers.

The existing A-B-A mode and outputs remain unchanged. No second producer,
generic experiment framework, retry, or emulator path is added.

## Physical Transaction

Perform exactly one run using the same pinned firmware, XRT packages, fixture,
2,000 ms TDR setting, `force_cmdlist=Y`, qualified response-trace module, and
normal power state as the 2026-08-01 physical observation.

The privileged transaction must:

- reject active NPU clients before changing the module;
- preserve or load the qualified module's `amd_pmf` dependency;
- capture preflight hashes, stdout, stderr, trace, and the dmesg delta;
- accept either producer completion or bounded TDR as an observed result;
- perform no retry; and
- restore and verify the normal distro-resolved `amdxdna` module and power
  state even when the producer fails or times out.

## Interpretation

- **A2 stalls:** the finite/relaunch contract is sufficient to explain the
  physical A-B-A nonresponse; B is not evidence of interference.
- **A2 completes:** B introduced a real multi-context interaction. The next
  probe must distinguish array-state disturbance from firmware-context state.
- **Infrastructure or restoration failure:** no semantic conclusion is
  licensed.

The evidence updates
[`../findings/2026-08-01-phoenix-context-repartition-proof-correction.md`](../findings/2026-08-01-phoenix-context-repartition-proof-correction.md)
and the host/firmware fidelity ledger. It does not directly justify an
emulator change.

The one physical control took the first branch: A1 completed, A2 stalled after
mailbox publication, and TDR recovered the sole context. B is not required for
the earlier nonresponse. The physical trace does not expose the core's internal
state, so this result closes the B confound without promoting the modeled
finite-program explanation to a silicon observation.

## Verification

Before physical traffic, watch the new CLI mode fail against the old producer,
then compile the changed producer with the existing warning-clean XRT command.
After the run, verify evidence hashes, module restoration, device presence,
and the exact A1/A2 lifecycle before recording a conclusion.
