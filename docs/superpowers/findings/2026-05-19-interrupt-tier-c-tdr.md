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
- Bridge test for wedge -> EIO behavior (needs deadlock-kernel fixture)
- Auto-reset-on-wedge behavioral knob (currently plugin-explicit)
- Map `last_run_wedged()` to `ERT_CMD_STATE_ABORT` in `platform_emu.cpp`
  (deferred per Task 14's investigation -- no clean injection point
  without structural change; revisit when bridge harness needs it)
