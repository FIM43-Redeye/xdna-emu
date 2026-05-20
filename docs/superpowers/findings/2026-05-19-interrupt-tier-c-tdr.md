# Interrupt Tier C — TDR / Context-Restart (shipped)

**Status:** Shipped on `dev`.
Spec: [`../specs/2026-05-19-interrupt-tier-c-tdr-design.md`](../specs/2026-05-19-interrupt-tier-c-tdr-design.md).
Plan: [`../plans/2026-05-19-interrupt-tier-c-tdr.md`](../plans/2026-05-19-interrupt-tier-c-tdr.md).

## What landed

Per-context state model + device-side TdrDetector that classifies engine
run state per cycle. xdna_emu_run consumes the classifier; on Wedged
verdict the context transitions to Failed and the run returns
`XdnaEmuHaltReason::WedgeRecovered`. Plugin observes via a new
`last_run_wedged()` accessor and the diagnostic `wedge_recovered` log
line; caller calls `xdna_emu_reset_context` (now required-resolved,
takes a context_id) before the next submission.

`platform_emu.cpp` translates the wedge to `ERT_CMD_STATE_ABORT` (spec
§4.5 / §5.2): the per-sub-command execute loop breaks on
`last_run_wedged()` (otherwise the next sub-cmd's FFI entry guard would
throw `ExecutionError` and abandon the rest of `submit_cmd`), and both
the outer ert_packet and each chain sub-command's state field get
`ABORT` so `run.wait()` consumers observe failure instead of treating
the hang as a normal completion.

QuiescenceDetector and StallDetector lifted out of `src/testing/` into
`src/device/tdr/` where they belong. Single in-tree consumer
(`xclbin_suite.rs`) updated; behavior unchanged.

## Plumbed for multi-context

All APIs take `ContextId` even though `Vec<Context>` has length 1 today.
The expansion path is storage + engine scheduling, not API reshape.

## Out of scope (tracked)

- Multi-context engine scheduling + lifecycle ioctls
- `Disconnected` context state and firmware-reload semantics
- Real-clock TDR timeout cadence
- Wedging xclbin fixture (hardware-painful to construct without
  genuinely deadlocking the NPU). The wedge -> ABORT path is plumbed
  through the FFI and translated in `platform_emu.cpp`, but no
  automated test in our corpus end-to-end exercises it -- the Task 15
  device-state-layer test
  (`classify_into_wedged_then_mark_failed_then_reset_recovers`) is
  the only proof point. Both the bridge harness wedge fixture and the
  FFI-end-to-end wedge fixture are gated on having a deterministic
  wedging input.
- Auto-reset-on-wedge behavioral knob (currently plugin-explicit)
