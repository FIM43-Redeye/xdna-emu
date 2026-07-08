# TCT Completion Model -- Design Record

**Branch:** `feat/array-tct-completion` (off `master`).
**Goal:** replace the flat `DEFAULT_MAILBOX_CYCLES = 8000` fudge with a real
Task-Completion-Token (TCT) driven completion path in the emulator's NPU-executor
run flow. This is a **fidelity/structure** change (completion becomes token-driven
like the hardware; the emitted-but-dead token goes live; it becomes the real
"seam A" that firmware wiring plugs into later) -- NOT a timing-derivation change.
The array->accumulator latency stays a calibration knob (`XDNA_EMU_MAILBOX_LATENCY`);
the design record `firmware-array-plugin-wiring.md` always said seam A's latency is
"documented nowhere -> a calibration knob."

## Grounded findings (scout + reads, 2026-07-08)

- **The 8000 fudge:** `src/npu/executor.rs:664` (`DEFAULT_MAILBOX_CYCLES`, env
  `XDNA_EMU_MAILBOX_LATENCY`). Charged ONCE per run in the `BlockedOnSync` arm
  (`executor.rs:636-676`): first satisfied sync pays `8000 + STREAM_FLUSH_CYCLES(4)`
  via a `mailbox_charged` bool (`:198`), later syncs pay only 4. The charge is a
  transition into `FlushingStreams { remaining }` (`:701-714`) that burns `remaining`
  cycles; the run loop (`xclbin_suite.rs:1200,1222-1236`) turns busy iterations into
  `engine.total_cycles` (`coordinator.rs:1721`) -- the number trace/bridge tests compare.
- **Completion today = `Channel_Running` polling**, not tokens: `is_sync_satisfied`
  (`executor.rs:333-372`) reads the channel status reg and the `channel_running` bit,
  satisfied on running->idle (`sync.started` latch) or fast-completion fallback.
- **The token is already emitted but never consumed.** `maybe_emit_task_token`
  (`stepping.rs:2482`) issues on task completion iff per-task Enable_Token_Issue
  (bit 31, `token.rs:72`) -- matches aie-rt (`xaie_dma.c:1792-1799`). It issues with
  `ch_idx as u8` (the ABSOLUTE channel index) into the per-engine `TokenState`
  (`token.rs:342`). `pop_task_token`/`has_task_token` (`task_queue_ops.rs:209-216`)
  are dead outside tests.
- **DATA MODEL ALREADY SUFFICIENT (Phase 1 collapses).** `Token{channel_id,
  controller_id}` (`token.rs:319`) carries `channel_id == abs_channel`, and lives in
  the engine at `(col,row)`. A `PendingSync{column,row,channel,direction}`
  (`executor.rs:157`) maps to `abs_channel` via `sync_abs_channel` (`executor.rs:305`).
  So a sync matches its token by: engine = `(col,row)`, token `channel_id ==
  abs_channel`. No `Token` enrichment with col/row/actor_id is needed -- the tile is
  implicit in which engine's `TokenState` you read.
- **Toolchain TCT semantics (authoritative):** emission gated by per-task
  Enable_Token_Issue bit 31 (already modeled); each `WAIT_TCTS` waits for **N=1**
  matching token; on-stream header is `col<<21 | row<<16 | actor_id` where actor_id is
  a `(tile-kind, direction, channel)` lookup (`AIENpuToCert.cpp:143-183`). abs_channel
  is the emulator's equivalent of actor_id for within-engine matching.

## Design (approved: token-driven, keep Channel_Running as migration cross-check)

`is_sync_satisfied` becomes token-driven, with the old `channel_running` result kept
in parallel as a `debug_assert!` guard during bring-in, removed once the sweep is clean:

```
is_sync_satisfied(sync, device):        // device now &mut (for consume-once)
    abs = sync_abs_channel(sync, device)
    running_ok = <existing channel_running logic>     // old path, side-effect-free
    if sync.satisfied { return true }                 // latched: token already consumed
    token_ok = engine(col,row).has_task_token_for_channel(abs)
    debug_assert!(token_ok == running_ok, "TCT vs Channel_Running disagree ...")
    if token_ok {
        engine(col,row).pop_task_token_for_channel(abs)   // consume EXACTLY once
        sync.satisfied = true
    }
    token_ok
```

- **Consume-once via a `satisfied` latch** on `PendingSync` (new field), so a token is
  popped exactly at the running->satisfied transition and never re-checked. This gives
  correct multi-sync semantics (each `WAIT_TCTS` consumes its own token) that the
  non-consuming `channel_running` path lacks -- the guard will surface any kernel where
  the two disagree.
- **New engine API (channel-filtered):** `has_task_token_for_channel(abs) -> bool` and
  `pop_task_token_for_channel(abs) -> Option<Token>` in `task_queue_ops.rs` (the current
  `pop_task_token` is unfiltered). Filter `TokenState` by `channel_id == abs`.
- **`&mut DeviceState` threading:** `is_sync_satisfied` / `syncs_satisfied` need `&mut`
  for consumption. Trace the run-loop caller (`xclbin_suite.rs:~1283`); the executor
  already holds the device mutably in `try_advance`, so the borrow should thread.
- **Latency (Phase 3):** replace the flat `FlushingStreams(8000)` transition with a
  token-transport latency reusing `XDNA_EMU_MAILBOX_LATENCY`, PRESERVING the pipelining
  property (concurrent in-flight tokens must NOT serialize into N x latency -- the
  current per-run/`mailbox_charged` insight must survive in token form, e.g. charge
  transport once per "notice burst", not per token).

## TDD phases

1. **Filtered consume API** -- `has/pop_task_token_for_channel`. *Test:* an engine with
   tokens on channels {0,2} answers per-channel correctly and pops only the matching one.
2. **Token-driven satisfaction + guard** -- `is_sync_satisfied` consumes a matching token,
   `satisfied` latch, `debug_assert` cross-check. *Test:* a sync on `(col,row,dir,ch)` is
   unsatisfied with no token, satisfied once a matching token exists; a second sync on the
   same channel needs its own token (consume-once).
3. **Latency** -- token-transport charge replacing the flat 8000, pipelining preserved.
   *Test:* single sync pays the transport latency; an N-sync kernel does NOT pay N x.
4. **Validation** -- `cargo test --lib`, then bridge/trace sweep vs the 8000-calibrated
   baseline; re-calibrate the knob against HW if the structure shifts the numbers.

## Open decisions / risks
- Re-validation: the 8000 was calibrated; a token model may shift cycle counts. Keep the
  `debug_assert` guard until a full sweep is clean, then remove `channel_running` from the
  sync path.
- Pipelining preservation in Phase 3 is the subtle part -- must not regress multi-sync
  kernels into N x latency.
- Fast-completion path (`is_sync_satisfied` case 2): a token may already be present before
  the first poll; the latch + filtered pop handle it, but test it explicitly.
