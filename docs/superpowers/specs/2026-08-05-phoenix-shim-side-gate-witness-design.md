# Phoenix Shim-Side Column-Gate Witness

**Status:** Approved for implementation.

**Target:** Phoenix/NPU1 with the pinned `edge-compute-mm2s` PM-address-fault
fixture and unmodified firmware `1.5.5.391`.

## Context

The existing event-65-started core counter is qualified as a periodic source:
threshold 64 produces its first `PERF_CNT_3` event at fault +64 and subsequent
events every 65 active core cycles. Its final missing pulse is not yet a column
gate witness because all existing observations stop with the compute column.

This slice adds an independently live shim observation. It proves only that
periodic core-originated traffic ceases while the shim trace path remains live.
It does not locate the gate to one cycle or authorize an emulator scheduler
change.

## Decision

Reuse the fixture's existing four-slot shim trace stream and the existing
periodic core counter. Carry each core `PERF_CNT_3` event to the shim on one
otherwise-unused event-broadcast channel, and trace that arrival beside an
independent shim-local periodic counter event.

Both decisive events are then timestamped by the same shim trace unit:

- `BROADCAST_A_13`: arrival of the compute-originated heartbeat; and
- `PERF_CNT_0`: proof that the shim observation path remains clocked.

Channel 13 is a fixture configuration choice, not a hardware constant. The
instrument must reject the fixture unless channel 13 is unused by its compiled
event-broadcast configuration. Channels 14 and 15 remain the compiled trace
stop/start channels, and aie-rt's interrupt-reserved channels 0 through 2 are
not candidates.

## Toolchain-Derived Instrumentation

Resolve event IDs with the existing `trace_capture.load_event_ids()` parser over
the generated aie-rt event header. Resolve every register offset and field from
`aie_registers_aie2.json` through the existing named-register patcher.

Starting from the already-qualified periodic fault instruction stream:

1. Map core `PERF_CNT_3` to `Event_Broadcast13` on logical tile `(0,2)`.
2. Patch logical shim `(0,0)` trace slots to retain
   `DMA_S2MM_0_START_TASK` and `DMA_S2MM_0_FINISHED_TASK`, then add
   `BROADCAST_A_13` and `PERF_CNT_0`.
3. Set the shim trace stop event to `NONE`, matching the already-qualified core
   trace treatment, so the host's ordinary broadcast-14 stop does not end the
   witness before firmware gates the compute column.
4. Configure shim counter 0 with start `BROADCAST_A_15`, stop `NONE`, reset
   `PERF_CNT_0`, and an initially adjustable threshold of 64. Starting on the
   existing trace-start flood keeps the heartbeat phase inside the capture.
5. Insert all counter and broadcast configuration before the existing
   broadcast-15 trace start. Do not alter DMA, routing, firmware traffic, the
   faulting ELF, or application output.

Reset-default event-broadcast directions are already unblocked for the pinned
fixture. Do not add speculative block-mask writes. If hardware does not deliver
channel 13 to the shim, stop and inspect the compiled/toolchain routing rather
than adding guessed masks.

## Evidence Arms

Run four bounded arms at the already-qualified low-QoS clock pair:

1. **Core baseline:** existing periodic fault instrument only.
2. **Shim-only fault:** add the shim heartbeat and extended shim trace, but no
   core broadcast mapping.
3. **Full witness fault:** add both shim instrumentation and core broadcast.
4. **No-fault control:** use the shim instrumentation with the control XCLBIN;
   it must contain shim heartbeats and no channel-13 arrivals.

Every arm requires exact application output, successful trace decoding, and
identical before/after power mode and reported clock pair.

## Classification Contract

For the full witness arm, let `c` be the ordered core `PERF_CNT_3` series, `b`
the ordered shim `BROADCAST_A_13` series, and `h` the ordered shim
`PERF_CNT_0` series.

The witness is qualified only when all of the following hold:

- `c` retains the qualified first offset and exact 65-cycle cadence;
- `b` and `c` have equal nonzero counts, proving complete transport before
  cessation;
- at least three `b` events establish one exact positive shim-domain arrival
  cadence `delta_b`;
- `h` has one exact positive cadence and is present both before and after the
  final `b` event; and
- at least one `h` event occurs strictly after `last(b) + delta_b`, proving the
  shim trace was alive after an expected core heartbeat failed to arrive.

This licenses the classification **core-originated periodic traffic ceased
while the shim remained live**. It does not identify the exact gate cycle; the
gate lies only somewhere after the final delivered heartbeat and before the
first missing one.

## Non-Perturbation Gate

The core baseline, shim-only fault, and full witness fault must preserve the
same exact core-side signature relative to `PM_ADDRESS_OUT_OF_RANGE`: ordered
event names and offsets, `PERF_CNT_3` count/cadence/final offset, and final core
trace offset. Exact output and clocks are part of the signature admission.

If the three signatures differ, preserve the captures and report the transport
result, but do not call the witness non-perturbing or qualified. Fresh-context
variance must not be averaged into equivalence.

## Implementation Boundary

Keep persistent code in the existing Phoenix clock-characterization module:

- one pure instrumentation helper;
- one relabeling helper; and
- one pure classifier returning a structured verdict and reason.

Drive the one-off hardware qualification from its timestamped experiment
directory using the existing runner and parser sessions. Do not add a second
campaign framework or dependency. Promote orchestration into `tools/` only if
the witness becomes a repeated campaign.

Tests come first and cover derived patch contents, occupied-channel rejection,
positive classification, missing liveness, dropped broadcast, irregular
cadence, spurious no-fault arrival, and core-signature mismatch.

Required software verification:

```bash
nice -n 19 python3 -m pytest tools/test_phoenix_pm_clock_characterize.py
nice -n 19 cargo test --lib
```

The physical four-arm capture is the acceptance test for the witness claim.

## Artifact Contract

Preserve the run under a new timestamped directory in
`build/experiments/phoenix-pm-clock-characterization/`. Record fixture,
firmware, driver, kernel, XRT, instruction-stream, register-database, and event
header identities; clock metadata; every arm's raw trace/events/output; exact
core signatures; shim series and cadences; classification and stop reason; and
an explicit statement that production emulator scheduling was unchanged.

## Non-Goals

- no one-cycle gate bracket;
- no clock-ratio fit or firmware CPI inference;
- no cross-domain timestamp comparison;
- no DMA or stream-token heartbeat;
- no production emulator timing change; and
- no claim beyond this pinned Phoenix lifecycle.
