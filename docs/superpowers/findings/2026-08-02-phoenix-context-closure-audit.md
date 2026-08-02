# Phoenix context-semantics closure delta

- Date: 2026-08-02
- Target: Phoenix/NPU1, firmware 1.5.5.391
- Firmware SHA-256: `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`
- Pinned driver object: `216cefececd74effcd7a88350c71b99f5ef9a215`

## Scope

This is a delta to the complete driver-reachable lifecycle census in
[`2026-07-27-phoenix-pinned-command-lifecycle-audit.md`](2026-07-27-phoenix-pinned-command-lifecycle-audit.md),
not a replacement opcode audit. It reconciles the remaining context-semantics
claims after persistent same-context execution and post-TDR replay were proven.
No physical-device or power-management transition was run for this delta.

## Exact `DESTROY_CONTEXT -> BUSY` predicate

The signed firmware fixes the previously open predicate:

1. `FUN_08ad9344` validates a four-byte context ID below six and calls
   `FUN_08ad70b8(id, 0)` for ordinary destruction.
2. The context table starts at virtual `0x0000e760`, has `0x1b8`-byte entries,
   and stores its state byte at `+2`. An entry without state bit `0x08` returns
   `MGMT_ERT_INVALID_PARAM`.
3. `FUN_08ad70b8` selects lane `state & 7` and calls `FUN_08adc3b0`. That helper
   loads the lane pointer from `0x000117e0 + lane * 0x14 + 0x20`. A null pointer
   is treated as drained.
4. For a non-null pointer, the helper compares the producer at
   `pointer + 0x24000204` with the consumer at
   `pointer + 0x24000200`.
5. Ordinary destruction returns `MGMT_ERT_BUSY` (`0x02000006`) exactly when
   those counters differ and state bit `0x10` is clear. Internal forced
   teardown passes a nonzero force argument and bypasses this BUSY branch.

The ring has 128 `u32` entries and both counters advance modulo `0x80`.
`FUN_08adc3f0` enqueues at the producer, `FUN_08adc4f8` reports occupancy, and
`FUN_08adc5b4` advances the consumer. The main supervisor sets state bit `0x10`
after a signalled partition-event group is mapped back to contexts. Its exact
public meaning is not established, so this audit names it only the
**partition-event escape flag**.

An execution trace through the unmodified image mapped the first allocated
context (`id 5`) to pointer `0x0002c014`; the compared effective and physical
addresses are consumer `0x2402c214` and producer `0x2402c218`. The permanent
guard `m2c_destroy_rejects_nonempty_completion_ring_until_drained` proves:

- equal counters permit normal destruction;
- one queued entry makes the same request return `MGMT_ERT_BUSY`; and
- restoring equality permits destruction and deterministic reuse of slot 5.

This corrects the old direct-causality story. A blocked APP-ERT task or missing
final TCT may explain why the physical system reached a nonempty completion
ring, but neither is the condition tested by `DESTROY_CONTEXT`. The historical
physical sequence remains valid; the unobserved causal antecedent that left its
ring nonempty remains open.

## Closure matrix

| Context surface | Disposition |
|---|---|
| Allocation order, partition ownership, six-slot exhaustion | Closed for the pinned NPU1 tuple |
| Firmware identity, CQ authority, host heap, CU configuration | Closed for the pinned NPU1 tuple |
| Clean destroy, stale direct-mailbox behavior, deterministic reuse | Closed |
| Repeated submissions through one live firmware context | Closed by the persistent same-context guard |
| Driver timeout destroy/recreate/MAP/CONFIG replay | Closed for the pinned external sequence and in-process signed-firmware sequence |
| Busy destruction | Signed-firmware predicate closed; physical cause of the queued completion in the historical wedge remains open |
| Runtime suspend/resume with a surviving public host handle | Open |
| System suspend/resume with a surviving public host handle | Open; same callbacks, distinct and riskier trigger |
| Driver DPM failure after firmware CREATE | Driver cleanup bug and valid input trace; not internal firmware behavior to invent |
| Invalid local ioctl handles | Driver-only rejection; no firmware request exists |

Normal driver power management destroys firmware contexts before management
`SUSPEND`, preserves host handles, and recreates `CREATE -> MAP -> optional
CONFIG` after hardware resume. Therefore the missing proof is not management
`SUSPEND` with a live firmware context. It is one public-handle sequence:

```text
A1 completes -> runtime PM tears down firmware context -> resume replay completes
-> A2 through the same public host handle completes
```

System suspend should be tested separately only after the runtime-PM path is
understood. The driver ignores the result of its two-second last-fence wait and
does not roll back partial replay, so those failure edges remain part of the
eventual contract.

## Gate disposition

The context-semantics gate is substantially closed but not complete. The next
bounded step is a read-only runtime-PM feasibility audit, followed by the
single same-handle A1/resume/A2 proof above. Broad capture campaigns, direct
management `SUSPEND` with live firmware slots, and system-suspend experiments
are not justified at this boundary.
