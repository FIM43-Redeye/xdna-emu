# Phoenix BlockWrite Phase and History Interaction

## Verdict

The complete word-aligned Phoenix signed-firmware `BlockWrite(1)` crossover is
deterministic but does not produce one phase-only timing map. Both physical
orders repeated digit-for-digit, yet six shared phases had different costs:

| Phase | Forward predecessor | Forward leading NOOPs | Forward cycles | Reverse predecessor | Reverse leading NOOPs | Reverse cycles |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | none | 2 | 266 | 4 | 14 | 266 |
| 4 | 0 | 0 | 216 | 8 | 14 | 216 |
| 8 | 4 | 0 | 246 | 12 | 14 | 246 |
| 12 | 8 | 0 | 232 | 16 | 14 | 248 |
| 16 | 12 | 0 | 230 | 20 | 14 | 266 |
| 20 | 16 | 0 | 216 | 24 | 14 | 216 |
| 24 | 20 | 0 | 246 | 28 | 14 | 246 |
| 28 | 24 | 0 | 232 | 32 | 14 | 248 |
| 32 | 28 | 0 | 230 | 36 | 14 | 266 |
| 36 | 32 | 0 | 216 | 40 | 14 | 216 |
| 40 | 36 | 0 | 246 | 44 | 14 | 246 |
| 44 | 40 | 0 | 232 | 48 | 14 | 248 |
| 48 | 44 | 0 | 230 | 52 | 14 | 266 |
| 52 | 48 | 0 | 216 | 56 | 14 | 216 |
| 56 | 52 | 0 | 246 | 60 | 14 | 246 |
| 60 | 56 | 0 | 232 | none | 1 | 232 |

The existing fail-closed classifier therefore returned qualified
`mixed_or_history_dependent`, not `phase_following`. Transaction byte phase is
still an admitted causal input from the earlier crossover, but phase alone is
not a complete state description for this path. No phase-only cost table,
payload-length sweep, or scheduler change is licensed.

## Pinned tuple and artifacts

- Phoenix/NPU1 firmware `1.5.5.391`, payload SHA-256
  `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`;
- kernel `7.1.7-custom+`;
- loaded `amdxdna.ko` SHA-256
  `21a46896fc9db2d7c3a41e7376fddeacdb70e6cbfc3c38d872462db49abecf35`;
- worktree commit `32447b1fdf961605d94a25e28b79a7fbce60de31`;
- source transaction SHA-256
  `25f020f9845a761576205e98db046dd26743d7423595f3c9fe4102e2bb8084ec`;
- XCLBIN SHA-256
  `d25ab5b8b45a0119c7a62efbe291599020adf86e27609fdc01a6346637ab51b3`;
- expected output SHA-256
  `64ed86b909d6d0502b64b28db0ea1272ffb358e20e9b1d88b63ccb07fa900cf5`;
- ascending candidate SHA-256
  `445305c5adb67f02ceda4383ae880f82c6b8d62b97b277f0edc0aff609018b1c`;
- descending candidate SHA-256
  `fc1b0b9ab4b676f70537f6696e16bc75a54d74a7d3e0d15bbf041cf693eb49de`.

The harness, qualification receipt, exact candidates, provenance, raw traces,
decoded events, outputs, logs, classification, and byte-derived layout are
preserved under:

```text
build/experiments/phoenix-pm-clock-characterization/
  20260808T215806Z-firmware-blockwrite-sixteen-phase-map/
```

The thin campaign harness pins the already-qualified four-phase harness by
SHA-256 rather than duplicating it. Its dry-run candidate regeneration and all
provenance pins passed before hardware access.

## Signed-firmware qualification

Both exact candidates passed the pinned signed-firmware `CHAIN_EXEC_NPU` guard.
Each exposed the exact emulator access sequence consisting of one leading
source `USER_EVENT_1` followed by sixteen `USER_EVENT_1` / `USER_EVENT_0`
measurement pairs. Every window consumed exactly 47 attempted firmware
instructions and executed the same 47-PC sequence within and across both
candidates. That sequence also matches the earlier four-phase crossover; its
packed-little-endian-u32 SHA-256 is
`4f9c9139aaa9f3c9ac150e5108f41cd3371f34a76f973a35a7f8633b612a2ef6`.

The temporary evidence assertions were removed after both runs. An initial
emulator preflight failed before execution because `XDNA_FIRMWARE` was absent
and the worktree-relative fallback did not reach the sibling `xdna-driver`
tree. The failure log is retained; rerunning with the exact hash-pinned sibling
firmware path qualified both candidates. No hardware was touched by that
preflight failure.

## Physical crossover and restoration

The low-QoS physical order was ascending, descending, ascending, descending.
Each run reproduced the expected output, retained the reported `400/800 MHz`
identity before and after dispatch, included both shim DMA lifecycle witnesses,
and exposed exactly sixteen alternating measurement pairs. Both repetitions of
each candidate produced identical costs.

The `finally` restore ran the ordinary source transaction without QoS,
reproduced the expected output, and returned to the reported default
`600/1028 MHz` identity. The device was unowned afterward and the recent kernel
warning/error query was empty. A separate post-campaign read confirmed the
same clock identity, no owner, and no warning/error output.

## What the candidate bytes add

The exact transaction streams explain why the reversal is not a pure phase
permutation. After its first window, the ascending candidate naturally advances
to the next phase by four bytes and needs zero CDO `NOOP` records. The descending
candidate needs fourteen four-byte `NOOP` records before every subsequent start
marker to move backward by four modulo 64. Predecessor phase changes too, so
this campaign does not isolate which history variable matters.

Phase 44 provides the strongest cross-campaign discriminator:

| Campaign | Arm | Predecessor phase | Leading NOOPs | Cycles |
|---|---|---:|---:|---:|
| four-phase | forward | 20 | 5 | 248 |
| four-phase | reverse | 56 | 12 | 248 |
| sixteen-phase | forward | 40 | 0 | 232 |
| sixteen-phase | reverse | 48 | 14 | 248 |

Three different predecessors with 5, 12, or 14 leading `NOOP` records all cost
248 cycles, while the zero-`NOOP` observation costs 232. Recent CDO `NOOP`
history is therefore the leading hypothesis, but it is not proven because the
predecessor was not held fixed.

## Licensed conclusions and next boundary

This evidence licenses only the following:

1. The complete crossover is deterministic within each exact transaction
   stream.
2. Transaction phase alone is insufficient to predict `BlockWrite(1)` timing.
3. At least one pre-window history variable changes the cost at phases
   `12,16,28,32,44,48` in this matrix.
4. Leading CDO `NOOP` count is the best current discriminator, not an
   established mechanism. The evidence does not identify a cache line, MMIO,
   NoC, firmware pipeline, or live clock phase.
5. Do not average the two maps, fit a phase-only table, begin payload scaling,
   or alter the scheduler.

The next minimal experiment should hold target phase 44, predecessor phase 40,
measurement ordinal, payload, and measured firmware path fixed. Its treatment
adds exactly sixteen four-byte CDO `NOOP` records immediately before the target
start marker, preserving phase modulo 64 while changing only recent `NOOP`
history. Run control, treatment, control, treatment and stop if either arm does
not repeat exactly.

## Subsequent closure

That balanced relocation crossover is now complete. Moving the same sixteen
`NOOP` records from before the phase-40 window to immediately before phase 44
held the phase-40 witness at 246 cycles and changed the phase-44 target from 232
to 248 cycles in both A/B repetitions. Relative record history is therefore a
confirmed causal input, although the physical state carrier and any marginal
law remain unproven. See
[`2026-08-08-phoenix-blockwrite-noop-relocation.md`](2026-08-08-phoenix-blockwrite-noop-relocation.md).
