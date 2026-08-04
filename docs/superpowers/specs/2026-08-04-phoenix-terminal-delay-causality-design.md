# Phoenix Terminal Application-Delay Causality Probe

**Status:** Discussion approved; implementation waits for review of this
written specification.

**Target:** Phoenix/NPU1 with pinned unmodified firmware
`amdnpu/1502_00/npu.dev.sbin` version `1.5.5.391`.

**Predecessors:**

- [`2026-08-04-phoenix-firmware-array-clock-characterization-design.md`](2026-08-04-phoenix-firmware-array-clock-characterization-design.md)
  defines the event-65-started performance-counter comparator and the original
  two-clock hypothesis.
- [`2026-08-04-phoenix-qos-clock-ratio-collapse.md`](../findings/2026-08-04-phoenix-qos-clock-ratio-collapse.md)
  records that ordinary context QoS exposes only two distinct reported
  H/MP-NPU ratios, leaving that hypothesis underidentified.
- The pinned fault fixture under
  `build/experiments/phoenix-pm-fault-array-ordering/20260804T194245Z/`
  establishes a deterministic 106-cycle final-release-to-event-65 interval on
  six Phoenix runs.

## Decision

Perturb only terminal application timing and ask whether the firmware-owned
column gate follows event 65 or an earlier lifecycle edge.

Insert one toolchain-derived counted loop into padding immediately before the
existing terminal PM-address fault. Compare the original and delayed fault
fixtures at both physically observable clock ratios with the already-qualified
gate comparator.

This probe tests the causal origin of the measured gate interval. It does not
create a third clock ratio, identify the array-work and firmware-work terms, or
authorize a production scheduler change.

## Scope

The probe may change only application execution time after all observable
kernel work and before the terminal fault. It must not change:

- signed firmware or firmware command traffic;
- non-program CDO configuration, DMA descriptors, locks, or routing;
- application output or any instruction before the terminal padding;
- the existing `j #0x4000` fault instruction;
- the comparator configuration except for its searched threshold; or
- emulator production scheduling.

Firmware-work perturbations remain a possible later experiment only if this
application-only probe is insufficient.

## ELF Mutation Contract

Extend `tools/patch-aie2-pm-address-fault.py` with an optional
`--delay-iterations N`. Omitting the option preserves the current behavior and
must reproduce the pinned fault ELF byte-for-byte.

The patcher continues to derive all instruction bytes with the selected Peano
toolchain. Its existing no-delay path keeps the current terminal-`done`
contract. When a delay is requested, it additionally must:

1. locate the unique terminal `done` from Peano disassembly;
2. verify the existing six two-byte NOPs after it;
3. verify an exact 28-byte all-NOP region immediately before it;
4. link the delay fragment at that discovered program address;
5. reject unresolved relocations or any unexpected linked disassembly; and
6. replace only the verified 28-byte pre-fault pad.

The linked fragment is:

```asm
movxm r0, #N
movxm p0, #loop
loop:
jnzd r0, r0, p0
nop                    // delay slot 5
nop                    // delay slot 4
nop                    // delay slot 3
nop                    // delay slot 2
nop                    // delay slot 1
nop                    // one final fall-through cycle
```

Peano emits 28 bytes for those instructions at the pinned site: two six-byte
moves, one four-byte `jnzd`, and six two-byte NOPs. Section-alignment padding
emitted after the fragment is not part of the fragment and must never be copied
over the following fault.

Registers `r0` and `p0` are terminally dead at this site: all output and lock
traffic has completed, and the next application action is the existing fatal
out-of-range jump. The hardware output and pre-delay trace controls below are
still required; terminal deadness is not accepted from static reasoning alone.

Compared with the current fault ELF, the delayed ELF may differ only in those
28 bytes. The six-byte `j #0x4000` and its five architectural delay-slot NOPs
must remain byte-identical.

The first campaign uses `N=15`, selected before observing a gate boundary. The
integer is a perturbation control, not a cycle claim. Its actual effect is
measured from the hardware trace.

## Fixture Construction and Preflight

Build the delayed fixture from the same pinned control ELF and the same
pre-baked packaging path as the base fault fixture (`--no-xchesscc` and
`--no-xbridge`). Record the control, base-fault, delayed-fault, XCLBIN,
instruction-stream, firmware, driver, kernel, XRT, and Peano identities.

Before the boundary campaign, require:

- the no-delay patch to reproduce fault ELF SHA-256
  `54aa8187261a592a048ad8f19752802ae331fbb16026a19d0fb8657296383239`;
- a decoded program-memory comparison showing that only the 28-byte terminal
  pad differs between base and delayed variants;
- identical expected output bytes from base, delayed, and the pinned hardware
  output;
- event 65 after the final `INSTR_LOCK_RELEASE_REQ` in both variants; and
- preservation of the existing pre-delay core and compute-DMA anchors without
  treating cross-domain shim timestamps as ordering evidence.

An emulator run is an integrity check for parsing, execution, and output. It is
not the timing oracle.

If the delayed fixture does not reach event 65, stop and preserve the capture.
Do not silently reduce `N`, infer that an earlier gate preempted the loop, or
continue to threshold search without a new reviewed probe.

## Hardware Matrix

Use the two distinct clock regimes already exposed through ordinary context
QoS:

| Regime | Reported MP-NPU/H |
|---|---:|
| Low | `400/800 MHz` |
| High | `600/1028 MHz` |

For each regime, run both the base and delayed fault variants. Every cell uses
fresh synchronous contexts, with asynchronous execution and cross-run context
reuse disabled.

For each cell:

1. prime the QoS session with one non-analysis dispatch;
2. query clocks before the measured work;
3. run the exponential and integer-binary comparator search;
4. query clocks after every dispatch;
5. repeat the complete search in an independent pass; and
6. require the two adjacent boundary brackets to agree exactly.

A run is valid only when output matches, required trace anchors are present,
and before/after power mode and clock metadata equal the admitted regime.

After all cells, run one ordinary no-QoS context and require exact restoration
of the original power mode and clock pair.

## Measurements

For variant `v` and clock regime `m`, define:

- `D(v,m)` as the same-core decoded `soc` delta from the final preceding
  `INSTR_LOCK_RELEASE_REQ` to the first `PM_ADDRESS_OUT_OF_RANGE`; and
- `G(v,m) = [L(v,m), U(v,m)]` as the adjacent comparator bracket from event 65
  to column gating, where `L` fires and `U=L+1` gates first.

Every valid threshold capture in one cell must report the same `D`. The delayed
increment

```text
delta_D(m) = D(delayed,m) - D(base,m)
```

must be positive and identical at both clock regimes. Otherwise the
application perturbation is not controlled and the campaign stops.

The corresponding release-to-gate interval is derived without choosing a
point inside the bracket:

```text
R(v,m) = [D(v,m) + L(v,m), D(v,m) + U(v,m)]
```

Use decoded `soc`, never the event-position-inflated `ts`, for `D`. Both events
come from the same core module, so their timer origin cancels; `G` is already
measured in that module's active cycles. All arithmetic uses exact integers.
There is no fitted tolerance, averaging, or comparison of unrelated trace
clock domains.

## Verdict Contract

Only exact, repeated invariants receive a causal classification.

| Observation | Verdict |
|---|---|
| `G(base,m) == G(delayed,m)` at both regimes | The gate interval is fault-relative for this pinned lifecycle. |
| `R(base,m) == R(delayed,m)` at both regimes | The gate follows an earlier release-relative lifecycle edge, not event 65. |
| Different invariants at the two regimes | Insufficient; the proposed causal model is incomplete. |
| Neither exact invariant holds | Insufficient; characterize phase or another missing edge. |
| Repeated brackets or `D` values disagree | Unclassified nondeterminism or instrument failure. |
| Delayed event 65 is absent | Invalid for this comparison; preserve and redesign. |

Near equality is not promoted to an invariant. A one-cycle difference may be a
real clock-phase effect, but this experiment does not assume that explanation.

Even the fault-relative verdict does not independently falsify the candidate
two-clock equation

```text
G_m = A + quantize(F * H_m / P_m)
```

because the delayed run uses the same `H/P` ratio. The original `A/F`
underidentification therefore remains. The result only tells the next design
which causal anchor it is legitimate to model.

## TDD and Implementation Boundary

Implementation proceeds in this order:

1. Add failing patcher tests for no-delay byte identity, the exact delayed
   instruction layout and byte boundary, invalid counts, unexpected terminal
   layouts, and unresolved-link rejection.
2. Implement the smallest patcher extension that passes them.
3. Add failing synthetic tests for `D`, `G`, `R`, and every verdict-table row.
4. Extend the existing Phoenix clock campaign with a paired base/delayed mode;
   preserve its default three-ratio model gate unchanged.
5. Build and preflight the delayed fixture locally.
6. Run the physical matrix and write a finding with the exact result.

Do not create a second patcher, add dependencies, generalize the experiment to
other devices, or alter production emulator timing in this slice.

Required software verification after changes:

```bash
nice -n 19 python3 -m pytest \
  tools/test_patch_aie2_pm_address_fault.py \
  tools/test_phoenix_pm_clock_characterize.py
nice -n 19 cargo test --lib
```

The hardware campaign is the acceptance test for the causal claim.

## Artifact Contract

Preserve the campaign under a new timestamped directory in
`build/experiments/phoenix-pm-clock-characterization/`. Its state and receipt
must record:

- every pinned software, firmware, ELF, XCLBIN, and instruction-stream hash;
- the exact Peano-derived delayed disassembly and byte-diff proof;
- `N`, every measured `D`, and both `delta_D` values;
- requested QoS and stable before/after clock metadata for every dispatch;
- all threshold classifications and raw output/trace/event paths;
- both independent brackets for every matrix cell;
- exact `G` and `R` comparisons and the resulting verdict;
- original and restored no-QoS state; and
- an explicit statement that no production scheduler change was made.

The campaign stops on the first violated admission rule, preserves everything
already captured, restores no-QoS state in a `finally` path, and reports the
failure rather than weakening the proof boundary.
