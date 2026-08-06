# Phoenix Real Column-Gate Freeze/Resume Witness

**Status:** Approved for implementation.

**Target:** Phoenix/NPU1, pinned firmware `1.5.5.391`, and the qualified
`edge-compute-mm2s` full shim-witness fixture.

**Predecessors:**

- [`2026-08-04-phoenix-firmware-array-clock-characterization-design.md`](2026-08-04-phoenix-firmware-array-clock-characterization-design.md)
  isolated the firmware/array scheduling question.
- [`2026-08-05-phoenix-shim-side-gate-witness-design.md`](2026-08-05-phoenix-shim-side-gate-witness-design.md)
  qualified the periodic core heartbeat, its broadcast-13 transport, and the
  independent shim heartbeat.
- [`2026-08-05-phoenix-aie-rw-access-wire-layout-mismatch.md`](../findings/2026-08-05-phoenix-aie-rw-access-wire-layout-mismatch.md)
  disqualified direct Phoenix `AIE_RW_ACCESS` as an ordinary register-inspection
  path under the current driver ABI.

## Decision

Prove freeze and resume with the real Phoenix column clock gate. Put the gate,
a bounded firmware-only dwell, and the restore in one NPU transaction command
while the already-qualified shim trace remains live.

Run a paired control and treatment made from the same instruction stream:

- the control writes clock-enabled at both transitions;
- the treatment writes clock-disabled, dwells, then writes clock-enabled; and
- the two binaries differ in exactly that first gate-value word.

The witness passes only if the control remains periodic, while the treatment's
shim-observed core heartbeat disappears during the physical gate and returns
after restore. Exact output, command completion, stable reported clocks, and a
fresh-context canary are mandatory. No emulator scheduler change follows until
the hardware evidence is interpreted separately.

## Why This Is the Real Gate

The AM025 register database defines shim `Column_Clock_Control` at offset
`0xFFF20`; field `Clock_Buffer_Enable` is bit 0. Its description says that it
gates the tile and memory-tile clocks above the shim while leaving the shim
itself unaffected. aie-rt's internal
`_XAieMl_PmSetColumnClockBuffer()` performs a mask write of that field only.

Do not call or reproduce the broader `_XAieMl_SetColumnClk()` operation. It
also changes shim module-clock state through `_XAieMl_PmSetShimClk()`, which
would destroy the independent observer this experiment requires.

The shim-local `PERF_CNT_0` heartbeat is therefore the independent stopwatch.
The core's own counter and trace timestamp stop with the core clock; they can
prove clean pre- and post-restore activity, but cannot measure elapsed gated
time by themselves.

## Pinned Starting Evidence

Use the qualified full-witness instruction stream at
`build/experiments/phoenix-pm-clock-characterization/20260805T232931Z-shim-witness/full-witness-fault.insts.bin`
(SHA-256
`f6329e498d8d254e6522eb0a960c3b8305991f758344e3575f42bc11596f5af1`).
It carries core `PERF_CNT_3` to shim `BROADCAST_A_13` and traces it beside
shim-local `PERF_CNT_0`.

The follow-up run at
`build/experiments/phoenix-pm-clock-characterization/20260806T003952Z-post-tct-noops/`
qualified opcode-5 NOOPs as four-byte records and showed that 256 post-TCT
NOOPs add substantial live witness time without changing output or clocks.
That validated NOOP encoding is the only dwell primitive used here.

## Address Trust Boundary

This experiment intentionally uses the signed firmware's raw transaction
executor to reach NPI registers. That is a pinned research mechanism, not a
production API. Generation and submission are fail-closed.

### Transaction base

Signed firmware `1.5.5.391` does not add transaction offsets to the physical
array base. Its syscall-`0x67` partition record returns the task's transaction
virtual base, and `FUN_08b0f624` adds each operation's low 32-bit register
offset to that base before access. The corresponding signed-firmware setup path
derives the base as:

```text
transaction_base = (physical_start_col + 6) << 25
```

The task's MMU relocates ordinary logical array addresses into the live
partition through the signed transaction interpreter's `0x84000000` array
view. Thus the target clock operation keeps the ordinary logical shim offset
`0x000FFF20`; it is not encoded as a physical-array escape.

NPI is outside that relocated array window. Its operation offset is derived as
`npi_absolute_address - transaction_base`, with checked 32-bit arithmetic. For
the pinned context this reaches the observed Phoenix NPI aperture at
`0xAC000000`. No NPI offset is emitted until the signed-firmware structural
preflight below reproduces the expected effective addresses.

### Live placement guard

The runner must query `DRM_AMDXDNA_HW_CONTEXT_ALL` after creating the exact
context and before allocating or submitting the instruction BO. It requires
exactly one entry for its PID and checks the expected physical `start_col` and
`num_col`. This campaign runs with context reuse and the asynchronous context
pipeline disabled, so the queried entry is unambiguous.

The experiment requires a one-column partition. A placement mismatch aborts
before submission; it is never repaired by changing offsets after the fact.
The small runner check reuses the already-working UAPI query pattern in
`tools/txn-poll-probe/txn-poll-probe.cpp`.

For this pinned fixture, the admitted placement is `start_col=1,num_col=1`.
That makes the transaction base `0x0E000000`, the two NPI operation offsets
`0x9E00000C` and `0x9E000200`, and the expected signed-firmware clock target
`0x860FFF20` (`0x84000000 + (1 << 25) + 0xFFF20`). This is the transaction
array view of architectural tile `(1,0)` register `0xFFF20`, not the separate
`0x9C000000` management alias. These are required manifest results, not
fallback constants. A different live placement stops the run.

### Manifest and allowlist

The generator emits a manifest beside each binary. It records the pinned input
hash, firmware hash, expected live placement, derived transaction base, every
inserted operation, its encoded offset, its pre-MMU effective address, and its
expected signed-firmware target.

Only these targets are allowed:

1. Phoenix NPI PCSR lock at NPI offset `0x0C`;
2. Phoenix NPI protected-register control at NPI offset `0x200`; and
3. logical shim `Column_Clock_Control` at array offset `0xFFF20` for the live
   one-column partition.

All register-offset high words must be zero. Address subtraction, addition,
alignment, and range checks must succeed without wraparound. The control and
treatment inserted-operation lists must be identical apart from the first
clock-field value, and a byte comparison must locate exactly one differing
32-bit word.

The generator resolves the normal aie-rt tree and parses the named AIEML NPI
offset, unlock-code, and protected-field macros from its source. It resolves
the clock register and field through the existing named AM025 register-database
path. The numeric values in this spec are review anchors, not a second set of
source constants. The Phoenix NPI base and transaction-base formula are instead
pinned-firmware facts and are admitted only under the recorded firmware hash
and successful structural preflight.

## aie-rt-Derived Privilege Sequence

Every clock transition gets its own complete protection envelope. Protection
is closed again before the gated dwell and is opened only when the restore is
ready to execute.

For physical column `C`, derive the protected-register value from aie-rt's
AIEML fields:

```text
protected(enable, C) = enable | (C << 1) | (C << 8)
```

The last-column field equals the first-column field because the admitted
partition is one column wide. The disable value retains the same column range
with bit 0 clear.

Each transition is encoded in this exact order:

1. write NPI lock with aie-rt's unlock code `0xF9E8D7C6`;
2. mask-poll the lock with mask/value zero, as `_XAie_NpiSetLock()` does;
3. write NPI protected-register control with the derived enable value;
4. mask-poll that register with mask/value zero;
5. write NPI lock with zero and perform the same zero-mask poll;
6. mask-write only `Clock_Buffer_Enable` at logical shim offset `0xFFF20`;
7. repeat steps 1 through 5 with the derived protected-register disable value.

This reproduces the ordering in `_XAie_NpiSetProtectedRegEnable()` while
narrowing the admitted protected range to the one live column. The instruction
stream never leaves protected access enabled across a NOOP dwell.

The current public path does not expose a trustworthy post-command readback of
the NPI lock/protection state. Therefore close/relock is established
structurally by the exact signed-firmware-executed sequence and behaviorally by
post-restore activity plus the canary; it is not described as direct register
attestation.

## Command Construction

Insert the experiment after the last TCT boundary and before the fixture's two
ordinary final trace-stop writes:

```text
gate transition
256 validated four-byte NOOPs
restore transition
256 validated four-byte NOOPs
existing final trace-stop writes
```

The second 256-NOOP tail is sized to provide the required three clean
post-restore heartbeats before normal trace termination. It uses the already
qualified dwell length rather than introducing another timing parameter.

The treatment's first clock mask-write uses value zero. The control uses value
one. Both restore with value one. No DMA, routing, trace-event selection,
counter threshold, ELF, XCLBIN, TCT, application buffer, or final stop command
differs between the arms.

## Structural Preflight

Before real hardware, execute both exact generated streams through the pinned
signed firmware in the emulator. This preflight checks only structure:

- `FUN_08b0f624` receives the derived transaction base for the admitted
  context;
- every inserted access resolves to the manifest's allowlisted address;
- operation order and values exactly match the two protection envelopes;
- protection is disabled before each dwell and after restore;
- the two streams reach the ordinary final trace-stop writes; and
- no operation added by this experiment touches an unexpected system or array
  target.

The preflight is not evidence that the real clock gate works. It protects the
real NPU from a malformed stream and proves that the signed firmware, rather
than a synthetic executor, interprets the bytes as intended.

## Hardware Sequence

Run KVM/VFIO first with the unmodified pinned firmware and driver. This keeps a
guest failure away from the host desktop, but it does not make the physical NPU
disposable: the same silicon can still wedge. One arm runs at a time, and any
failure stops the campaign before another experimental submission.

For each arm:

1. require the live placement guard;
2. record before-run firmware, driver, kernel, XRT, and clock identities;
3. submit the full command and require ordinary completion;
4. decode both core and shim traces and verify exact application output;
5. require the same reported clock pair before and after;
6. require the post-restore heartbeat contract below; and
7. destroy the experimental context, create a fresh ordinary context, and run
   the pinned unmodified control as a canary with exact output.

Run the control before the treatment. A KVM treatment passes only when all
checks and the canary pass. Host execution is a separate later confirmation,
performed only if KVM passes and only after review of the KVM receipt.

## Classification Contract

Let:

- `c` be core-local `PERF_CNT_3` timestamps;
- `b` be shim timestamps for core-originated `BROADCAST_A_13`; and
- `h` be shim-local `PERF_CNT_0` timestamps.

The already-qualified active cadence is 65 cycles. Classification remains
exact; captures are never averaged into a tolerance.

### Control

The control passes only if:

- `h` has exact cadence 65 and at least seven events;
- `b` has exact cadence 65, at least seven events, and no anomalous gap;
- `c` has exact active-core cadence 65 and the same event count as `b`; and
- completion, output, clocks, and canary all pass.

### Treatment

The treatment passes only if:

- `h` retains exact cadence 65 across the complete command, with at least three
  events strictly inside the unique `b` gap;
- `b` has at least three exact-cadence arrivals before the gate;
- `b` has exactly one gap at least four times its normal cadence, followed by
  at least three exact-cadence arrivals;
- every other adjacent `b` interval is exactly 65;
- `c` has exact active-core cadence 65, the same event count as `b`, and at
  least three corresponding events on each side of the gap;
- post-gap `b` and `c` prove that the core clock actually resumed; and
- completion, output, clocks, and canary all pass.

No wall-time claim is made from `c`: its clock freezes. The decisive causal
evidence is the paired result: the one-word control remains periodic, while
the treatment alone creates a core-traffic gap under a continuously periodic
shim observer and then resumes cleanly.

If a packet is dropped, cadence is irregular, more than one gap appears, the
trace ends during the gate, or the core never returns, classify the arm as
invalid. Do not reinterpret a malformed trace as clock silence.

## Failure and Recovery Rules

Stop before submission on any identity, placement, address, allowlist,
high-word, ordering, or one-word-diff failure.

Stop after an arm on any timeout, trace decode failure, output mismatch, clock
change, missing relock sequence, missing post-restore activity, or failed
canary. Preserve the receipt and diagnostics. Do not attempt the host run.

If the NPU wedges, use the existing recovery escalation in
[`docs/operations.md`](../../operations.md); the experiment does not invent a
new reset path. The generator cannot guarantee restoration after a firmware or
device failure, which is why KVM-first execution, one arm at a time, and the
post-run canary are mandatory.

## Minimal Implementation Boundary

Reuse the existing machinery:

- `tools/trace-patch-events.py` for transaction walking and TCT-boundary
  insertion;
- `tools/phoenix-pm-clock-characterize.py` for fixture construction,
  manifests, and classification;
- `tools/test_phoenix_pm_clock_characterize.py` for pure TDD coverage; and
- `bridge-runner/bridge-trace-runner.cpp` for one live-placement assertion
  option, `--expect-placement <start_col>:<num_col>`, using the existing
  `txn-poll-probe` UAPI pattern.

Add no dependency, campaign framework, background service, synthetic marker,
new trace event, or general clock-control API. A small fixture-specific builder
and classifier are enough. Generalize only after another NPU or experiment
needs the same seam.

## Tests and Verification

Tests come first. The focused Python checks cover:

- exact control/treatment construction and one-word binary difference;
- base, relative-offset, protection-field, and manifest derivation;
- overflow, nonzero-high-word, allowlist, and placement rejection;
- exact two-envelope ordering and protection closure around both dwells;
- positive control and treatment classification; and
- missing shim liveness, insufficient pre/post samples, short or multiple
  gaps, irregular cadence, absent resume, output mismatch, and canary failure.

Do not add a C++ test framework for the runner's small guard. Its executable
checks are one KVM mismatch invocation that must fail before submission and one
exact-placement invocation that must reach the control run. Missing or
ambiguous same-PID entries use the same fail-closed path.

Required software verification:

```bash
nice -n 19 python3 -m pytest tools/test_phoenix_pm_clock_characterize.py
nice -n 19 cmake --build bridge-runner/build
nice -n 19 cargo test --lib
```

Then run the focused signed-firmware structural preflight, followed by the
KVM control/treatment/canary sequence. Full bridge or ISA suites are not useful
for this fixture-specific evidence slice and are deferred until a production
emulator change exists.

## Artifact Contract

Preserve the run under a new timestamped directory in
`build/experiments/phoenix-pm-clock-characterization/`. Its receipt records:

- all pinned software and fixture hashes;
- control and treatment binary hashes plus the exact one-word diff;
- live context placement and derived transaction/NPI/clock addresses;
- the complete allowlist manifest and structural-preflight result;
- raw instructions, trace, decoded events, output, and clock metadata;
- `c`, `b`, and `h` series, their exact cadences, the treatment gap, and the
  recovery samples;
- command completion, fresh-context canary, and KVM/host boundary; and
- the classification or exact stop reason.

Raw captures remain experiment evidence and are not committed wholesale. The
small receipt and generally useful source/test changes may be committed after
review.

## Non-Goals

- no `AIE_RW_ACCESS` register readback;
- no direct driver MMIO seam or shim clock gating;
- no public or reusable NPI clock API;
- no synthetic cycle counter, event suppression, or lifecycle proxy;
- no production emulator scheduler or timing change;
- no claim of one-cycle gate location or firmware CPI;
- no claim beyond the pinned Phoenix firmware/context/fixture; and
- no older-firmware or AIE2P conclusion.

## Authorization Outcome

A passing KVM pair authorizes review for one host confirmation. A passing host
pair establishes the bounded fact that the real Phoenix column gate freezes
and resumes the pinned core workload while the shim remains live. Only then do
we design the smallest emulator scheduling correction needed to reproduce the
observed causal behavior.
