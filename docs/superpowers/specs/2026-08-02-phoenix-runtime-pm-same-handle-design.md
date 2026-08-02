# Phoenix Runtime-PM Same-Handle Proof

**Date:** 2026-08-02  
**Status:** Executed; passed

## Goal

Prove one driver-reachable Phoenix/NPU1 lifecycle:

```text
A1 completes
-> PCI runtime PM suspends while the public XRT handle survives
-> the driver resumes and replays the firmware context
-> A2 completes through that same public handle
```

The proof uses the Chess `nd_memcpy_linear_repeat` fixture already observed to
complete twice through one live context. It does not test system sleep.

## Derived Driver Contract

At pinned driver commit `216cefececd74effcd7a88350c71b99f5ef9a215`, command
submission acquires a runtime-PM reference and job cleanup releases it. A live
userspace device, hardware context, kernel, and BO set therefore do not prevent
runtime autosuspend after a completed job.

The runtime suspend callback walks every live client and destroys its firmware
contexts before stopping the hardware. Resume starts the hardware and replays
each surviving host context as `CREATE_CONTEXT -> MAP_HOST_BUFFER ->` optional
additional heaps `-> CONFIG_CU`. The next command submission is the native wake
trigger.

## Experiment

Use an evidence-local derivative of the already captured persistent-repeat
producer. Its only behavioral change is a cooperative stdin gate after A1 has
completed and its output has been verified. Run it under `timeout` so losing
the controller cannot leave the public handle alive indefinitely.

1. Record the loaded module, firmware, XRT, fixture, instruction, and power
   tuple; reject unexpected device clients.
2. Start the existing amdxdna tracepoint capture.
3. Run A1 and stop at the cooperative gate with the original XRT objects live.
4. Install a bounded automatic fallback that restores PCI `power/control=on`,
   then temporarily write `auto`.
5. Require `runtime_status=suspended` and an increase in
   `runtime_suspended_time`; preserve the sysfs observations.
6. Release the gate. A2 must submit through the same userspace objects, causing
   runtime resume and driver replay.
7. Capture the trace and dmesg delta, restore `power/control=on`, and verify the
   module, PCI binding, device node, absence of clients, and active power state.
8. Do not retry. Infrastructure, execution, and restoration failures remain
   distinct outcomes.

## Success Boundary

Success requires all of the following:

- A1 and A2 both complete with byte-exact fixture output;
- one genuine `active -> suspended -> active` runtime-PM transition occurs
  between them;
- the trace orders firmware-context teardown before suspension and
  `CREATE_CONTEXT -> MAP_HOST_BUFFER -> CONFIG_CU` replay before A2 execution;
- the producer retains the same public XRT object lifetime throughout; and
- cleanup restores `power/control=on` and the original healthy device state.

Sysfs state without successful A2 is not enough. Successful A2 without an
observed suspension is the ordinary repeat control, not this proof.

## Observed Result

The single successful campaign is sealed under
`build/experiments/firmware-runtime-pm-repeat-20260802-03/`. A1 and A2 both
passed byte-exact checks through one surviving public XRT object lifetime. Its
lossless 159/159-entry private trace orders context destruction before genuine
runtime suspension and replayed CREATE, MAP, and CONFIG before A2 execute. The
full interpretation and pinned tuple are recorded in
[`../findings/2026-08-02-phoenix-context-closure-audit.md`](../findings/2026-08-02-phoenix-context-closure-audit.md).

## Explicit Exclusions

- **No system suspend or hibernation.** On this machine, system suspend triggers
  firmware defects and can leave the CPU capped near 0.95 GHz.
- No module unload, replacement, reset, TDR injection, or direct management
  `SUSPEND` command.
- No fixed sleep as the transition oracle.
- No generalized PM harness, capture campaign, or emulator change.
