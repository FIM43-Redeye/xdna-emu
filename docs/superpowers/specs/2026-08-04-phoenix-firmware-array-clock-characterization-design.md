# Phoenix Firmware/Array Clock Characterization

**Status:** The original comparator qualified, but QoS discovery exposed only
two reported ratios. A narrower signed-firmware timeline has now produced an
exact low-QoS CDO-NOOP law and falsified scalar interpreter-step timing. No
production scheduler change is authorized. See
[`2026-08-04-phoenix-qos-clock-ratio-collapse.md`](../findings/2026-08-04-phoenix-qos-clock-ratio-collapse.md)
and
[`2026-08-08-phoenix-firmware-clock-timeline.md`](../findings/2026-08-08-phoenix-firmware-clock-timeline.md).

**2026-08-05 correction:** direct Phoenix `AIE_RW_ACCESS` observations are
invalid and unsafe under the current driver ABI. The failed column-clock read
targeted an aliased row-5 compute tile, not the shim clock register. See
[`2026-08-05-phoenix-aie-rw-access-wire-layout-mismatch.md`](../findings/2026-08-05-phoenix-aie-rw-access-wire-layout-mismatch.md).

**2026-08-08 qualitative scheduler correction:** the protected physical gate
experiment proved that column gating freezes and later resumes the same core
state, but it did not establish a firmware-to-array clock ratio. Reconciliation
then found that the engine skipped a gated enabled core while leaving its
`all_halted` reduction true. The engine now reports `WaitingForClock` for this
externally resumable state, and the firmware pump reports
`ArrayClockGatedFirmwareWaiting` when firmware is also idle. Neither state is
completion. This removes the false-halt diagnosis without changing scheduling
cadence; the signed-firmware PM-fault guard remains the quantitative RED.
The fully provisioned guard reproduces that boundary as
`ArrayClockGatedFirmwareWaiting` with report fields
`firmware_instructions=1` and `aie_cycles=1`, with core `(1,2)` enabled and
unfinished at PC `0x484` while both its column and core-module clocks are
disabled. The firmware count still includes the already-halted `WAITI` revisit
described below and is not timing evidence. The guard's required
`ResponseCompleted` assertion therefore still exits nonzero.

**2026-08-08 quantitative discriminator:** a separate post-TCT timeline
bracketed authentic CDO `NOOP` blocks with toolchain-derived shim events. At
the reported `400/800 MHz` low-QoS identity, two complete physical runs were
digit-for-digit identical. Settled nonzero blocks obeyed `80 + 33*N` nominal
MP-NPU cycles, while the marker-only path cost 42. The same unmodified signed
firmware consumed `19 + 14*N` interpreter attempts. Since `33/14 != 42/19`, a
scalar CPI or immediate scheduler multiplier is disproven. Per the authorization
table below, the next boundary is missing Xtensa timing-class characterization,
not scheduler implementation.

**Target:** Phoenix/NPU1 with pinned unmodified firmware
`amdnpu/1502_00/npu.dev.sbin` version `1.5.5.391`.

**Predecessor:**
[`2026-08-04-phoenix-pm-fault-array-ordering-design.md`](2026-08-04-phoenix-pm-fault-array-ordering-design.md)
closed the proposed array-internal timing correction and isolated the remaining
failure to firmware/array scheduling. Its signed-firmware RED guard publishes a
successful response and gates the column before the modeled core reaches its
already-correct native PM-address fault.

## Decision

Characterize the real post-fault firmware/array clock relationship before
changing production scheduling. Use an on-tile performance-counter threshold
as a non-perturbing comparator, search for the clock-gate boundary at three
Phoenix DPM points, and reserve one point as a falsifier for the candidate
two-clock model.

No fixed delay, average CPI, regression tolerance, or scheduler change is
authorized merely because it improves the current RED guard. A production
scheduler is considered only if the hardware result and corrected emulator
work accounting identify the same model.

Failed or ambiguous probes are expected outcomes. They refine the next probe;
they do not license a best-fit approximation.

## Current Defect

`pump_runtime` is a functional boundary scheduler:

1. run firmware to a wait, unresolved poll, unknown instruction, or budget;
2. advance exactly one AIE engine cycle;
3. publish modeled TCT and L2-error work;
4. repeat.

That policy can let firmware consume and answer an error before the array has
advanced far enough to produce it. It also makes no clock claim: the firmware
may execute a large instruction burst while zero AIE cycles elapse.

The existing signed-firmware guard remains the RED case. The hardware
characterization does not weaken or replace it.

## Rejected Timer-Read Witness

Do not infer the gate cycle from post-run `AIE_RW_ACCESS` reads of
`Timer_Low`. The prior Phoenix calibration
[`2026-05-26-aie-rw-access-not-a-cycle-probe.md`](../findings/2026-05-26-aie-rw-access-not-a-cycle-probe.md)
sampled the wrong physical tiles because Phoenix decodes the current driver's
context ID and relative row as physical row and column. Those values cannot
witness when the intended domain stopped.

This explicitly supersedes the initially proposed "read twice and require an
equal value" check. `AIE_RW_ACCESS` is not a valid ordinary state-inspection
path on Phoenix under the current ABI either.

## Pinned Fixture and Causal Anchor

Reuse the control/fault pair preserved under
`build/experiments/phoenix-pm-fault-array-ordering/20260804T194245Z/edge-compute-mm2s/`.
The fault core ELF has SHA-256
`54aa8187261a592a048ad8f19752802ae331fbb16026a19d0fb8657296383239`.
The control and fault ELFs are byte-identical until the terminal control-flow
site; the fault arm replaces the former terminal `done` with the six-byte
out-of-range jump.

The native core event `PM_ADDRESS_OUT_OF_RANGE` (event 65) is the causal start.
It is in the same core module whose clock is later gated, avoids host launch
jitter, and requires no injected timing marker. Existing trace anchors and
output checks remain liveness and integrity controls.

The control PDI is not used to estimate a gate interval. It proves that the
event-65-started comparator does not fire when its start event never occurs.

## Non-Perturbing Gate Comparator

Use spare core performance counter 3, configured through the existing
toolchain-derived trace register path:

- start event: `PM_ADDRESS_OUT_OF_RANGE`;
- stop event: none;
- reset event: none;
- event threshold: candidate integer `N`;
- emitted event: `PERF_CNT_3`;
- trace stop event: none.

The AM025 register database and aie-rt performance-counter API define this
start/stop/reset/threshold contract. No numeric bit encoding is hand-written;
configuration continues to use named registers and events lowered by the
existing trace tooling.

If the core module remains clocked for `N` cycles after event 65, counter 3
reaches the threshold and emits `PERF_CNT_3` into ordinary trace slot 4. If
firmware gates the column first, the counter and trace stop progressing and the
comparator event never occurs. Periodic counter-2 events provide both liveness
and packet flush after a firing comparator.

Qualification corrected the original trace-stop proposal. Phoenix consumes a
configured stop event before recording that event in a trace slot; the emulator
trace unit has the same stop-before-slot ordering. Making `PERF_CNT_3` both the
witness and trace stop therefore made a typed witness structurally impossible.
Trace stop is left disabled so the counter event itself remains observable.

The ordinary end-of-runtime trace stop is disabled for this characterization
fixture so it cannot truncate the post-fault interval. Existing periodic
counter-2 events remain a liveness and packet-flush canary. If the first
negative control shows insufficient complete trace data, adjust only that
existing period; do not introduce another capture channel.

### Comparator qualification

Qualify the instrument before any DPM search:

1. A small threshold already bounded by the prior trace (initially 64 cycles)
   must produce a typed comparator event after event 65.
2. A very large threshold must leave a valid trace containing event 65 and the
   periodic liveness markers but no comparator stop.
3. The unmodified control PDI must contain no event 65 and no comparator stop.
4. Output bytes and the pre-fault causal anchors must still match the pinned
   fixture.

Failure of any qualification arm stops the campaign. Diagnose the counter,
trace-stop, or packet-flush mechanism and try another bounded instrument; do
not interpret an unqualified absence as clock gating.

## Boundary Search

For one admitted clock pair, classify threshold `N` as:

- **fires:** event 65 is present and the typed `PERF_CNT_3` event is present;
- **gates first:** event 65 and the liveness prefix are valid, but no comparator
  stop is present;
- **invalid:** any required fixture, output, clock, or trace check fails.

Exponential search first establishes one firing and one non-firing threshold.
Integer binary search then closes them to adjacent values. The result is the
one-cycle bracket:

```text
[largest threshold that fires, smallest threshold at which gating wins]
```

The bracket deliberately leaves same-cycle event/gate priority unspecified.
No point estimate is manufactured between its endpoints.

Run the complete search twice in independent capture passes. The two brackets
must agree exactly. Any disagreement is an unclassified nondeterminism or
instrument defect and blocks clock-model fitting; it is never averaged away.

## QoS-selected DPM Matrix and Restoration

The loaded mainline driver rejects `powersaver` and `balanced` power-mode
requests with `-EOPNOTSUPP`. Do not backport part of the newer driver's power
mode implementation merely to run this experiment. Select DPM through the
ordinary context QoS path already implemented by the loaded driver instead.

Pass positive `gops` and `fps` values through
`xrt::hw_context::cfg_param_type`. The driver resource solver then chooses the
lowest DPM level satisfying the request using the XCLBIN's declared
`operations_per_cycle` and its own Phoenix H-clock table. For the pinned
fixture, the declared value is 2048. Candidate QoS tuples are probe inputs,
not assumed clock identities; the SMU-returned MP-NPU/H pair is the identity
used by analysis.

The bridge runner keeps its synchronous context alive after each completed
batch command. Prime each new QoS session with one recorded, non-analysis run,
then query `DRM_AMDXDNA_QUERY_CLOCK_METADATA`, execute the measured run through
a freshly recreated context with the same QoS, and query again. A measured run
is valid only when the before/after pair is identical. Keep asynchronous and
cross-run context reuse disabled so one context is active at a time.

Admit only distinct observed clock pairs. The campaign requires at least three
distinct H/MP-NPU ratios; choose the two with the widest ratio separation for
calibration and reserve every other pair as a falsifier. If the normal solver
cannot expose enough distinct ratios, stop as insufficient rather than swap
drivers or reinterpret duplicate points.

Record the original power mode and clock pair without changing the mode. In a
`finally` path, create one ordinary no-QoS context, run the same cheap fixture,
destroy it, and require the reported mode and clock pair to match the original.
No privileged mode transition is part of this campaign.

## Exact Clock-Model Test

For admitted clock pair `m`, let:

- `G_m` be the measured one-cycle gate bracket in core-module cycles;
- `H_m` and `P_m` be the run's reported H and MP-NPU clocks;
- `A` be mode-invariant array/error-publication work in core-module cycles;
- `F` be mode-invariant firmware work in MP-NPU cycles.

The candidate model is:

```text
G_m = A + quantize(F * H_m / P_m)
```

`quantize` denotes only the adjacent integer outcomes allowed by rational clock
phase. It is not a fitted error bar.

Use the two pairs with the widest observed `H/P` separation to enumerate
integer `(A, F)` candidates consistent with their brackets. Keep all other
pairs untouched as falsifiers. The model passes only if the calibration leaves
an identifiable firmware-cycle term and that same term predicts every held-out
bracket exactly under the allowed integer phase outcomes.

If reported clocks differ from the driver's nominal table, the reported values win. If
no candidate survives, materially different candidates survive, or the held-out
point fails, the result is **insufficient or falsified**. Do not select the
closest regression, widen a tolerance, or average the three modes.

## Emulator Work Accounting Gate

The existing `IdleReport::instrs_executed` is not suitable for the comparison.
`Step::Wait` currently represents both:

- execution and retirement of the initial `WAITI`; and
- a later scheduler call while the CPU is already halted, when no instruction
  executes.

`boot_to_idle_on` increments both. That produces phantom firmware work whenever
the runtime revisits a sleeping CPU.

Only after the hardware model survives every held-out DPM point, correct this
semantic under TDD. The execution result or accounting seam must distinguish:

- an instruction or faulting attempt that consumed CPU work;
- the initial `WAITI`, which consumes work once and then halts;
- an already-halted yield, which consumes zero CPU work; and
- interrupt/exception entry work, labeled explicitly rather than called a
  retired instruction.

The correction is shared CPU semantics, not a counter heuristic local to the
PM-fault guard. Existing callers must continue to stop on the same wait reason;
only truthful work accounting is added.

Count the pinned emulator path from modeled event-65/L2 publication through the
firmware-owned column-clock write. A one-MP-cycle-per-work-step hypothesis is
accepted only if that corrected count equals the hardware-derived `F` under the
same rational phase model.

If it does not equal `F`, do not use `F / work_steps` as a scalar CPI. The next
probe must localize the missing cost -- initially interrupt entry, MMIO, or
instruction-class timing -- and repeat the same falsifiable process.

## Authorization Outcomes

| Result | Authorized next action |
|---|---|
| Comparator qualification fails | Repair or replace only the instrument. |
| Independent brackets disagree | Classify the nondeterminism; no clock fit. |
| DPM model is ambiguous or falsified | Design a narrower clock-domain probe. |
| Hardware `F` differs from corrected emulator work | Characterize the missing timing class; no scalar CPI or delay. |
| Hardware `F`, corrected work, and held-out mode all agree | Design the minimal integer/rational firmware-array scheduler and make the existing signed-firmware guard its RED case. |

Even the final row licenses only an observed scheduler correction for this
pinned lifecycle. It does not establish general firmware cycle accuracy. That
claim requires independent firmware paths and instruction mixes after the
starvation defect is closed.

## Artifact Contract

Preserve the campaign under a new timestamped directory in
`build/experiments/`. Its receipt records:

- firmware, driver, kernel, XRT, xclbin, instruction-stream, and fault-ELF
  identities;
- original/restored power mode and clock pair;
- requested QoS tuple and observed clock pair for every admitted regime;
- before/after MP-NPU and H clock queries for every run;
- threshold classification and raw trace/output paths;
- the two exact brackets per admitted clock pair;
- calibration candidates, held-out verdict, and any stop reason;
- an explicit statement that no production scheduler change was made.

Raw captures remain experiment evidence and are not committed wholesale. The
small receipt, analysis code, and any generally useful fixture transform may be
committed after review.

## Explicit Non-Goals

- no production scheduler in this characterization slice;
- no fixed firmware delay or average CPI;
- no post-gate timer-read inference;
- no shim/NoC cross-domain timestamp as a causal gate oracle;
- no general trace framework or new background service;
- no claim about older firmware or AIE2P timing;
- no change to DMA, routing, locks, TCT transport, L2 error semantics, or
  firmware-owned clock writes.

## TDD and Execution Order

1. Add the smallest parser/analysis RED checks for typed comparator positive,
   negative, invalid-prefix, and adjacent-bracket behavior.
2. Reuse the existing trace injector/runner path to configure counter 3 and its
   trace slot; add no generalized API unless the current named-register seam
   genuinely cannot express it.
3. Run the three comparator qualification arms on Phoenix.
4. Qualify distinct QoS-selected clock pairs, then preserve two complete
   searches at each admitted pair and restore the original clock state in all
   exits.
5. Perform the exact two-point calibration and held-out test.
6. Stop and review the evidence.
7. Only if authorized, begin a separate TDD correction for firmware work
   accounting and then design the production rational scheduler.
