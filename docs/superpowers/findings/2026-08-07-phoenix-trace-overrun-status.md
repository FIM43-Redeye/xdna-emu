# Phoenix trace overrun is state 3; its trigger remains unmodeled

## Scope

This finding is limited to the pinned Phoenix NPU1 tuple used by the protected
column-gate experiment: firmware 1.5.5.391, management protocol 5.8, and the
driver/toolchain artifacts recorded by the experiment request. It does not yet
claim a general overrun trigger or capacity for AIE2 trace units.

## Evidence

The physical treatment restored the column clock and generated the configured
source-local trace stop event. The first subsequent core `Trace_Status` read was
`0x00000300`; every later read remained `0x00000300` until the old one-second
idle poll timed out. The shim trace status was `0x00000000`. The raw evidence is
under:

`build/experiments/phoenix-pm-clock-characterization/20260806T202217Z-protected-column-gate/host/treatment-20260808T021613Z-1805321/`

The authoritative AM025 register database describes `Trace_Status.State`
(bits 9:8) as `00=idle, 01=running, 11=overrun`, so the observed core state is
unambiguously overrun. The generated aie-rt register header agrees on the field
mask (`0x00000300`). aie-rt's `XAie_TraceState` C enum instead assigns its third
member the sequential value 2, while `XAie_TraceGetState` returns the raw field
without translating it. That enum value is therefore not the AIE2 register
encoding; the register database plus silicon are the authority here.

## Emulator disposition

The emulator previously exposed only Idle and Running architecturally and
returned Idle for its internal stopped/drain latch. It now represents Overrun
as the architectural value 3 and treats it as terminal for tail draining.

The transition into Overrun is deliberately still absent. The emulator's trace
packet queues are presently unbounded, and the physical run did not establish
whether the overrun arose before gating, during the gated interval, during
restore, or while the stop wave propagated. Guessing a capacity or trigger
would turn one observed state into an invented mechanism.

The protected probe now captures core and shim trace status at three causal
seams: immediately before gating, immediately after restoration, and after the
stop event. The next paired control/treatment run can therefore localize the
transition. Only that localized result, followed by a narrower capacity or
backpressure probe if needed, licenses an emulator transition rule.
