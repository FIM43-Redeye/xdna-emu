# TCT Completion Model -- Design Record

**Branch:** `feat/array-tct-completion` (off `master`).
**Goal:** replace the flat `DEFAULT_MAILBOX_CYCLES = 8000` fudge with a real
Task-Completion-Token (TCT) driven completion path in the emulator's NPU-executor
run flow. Phases 1-2 are a **fidelity/structure** change (completion becomes token-driven like the
hardware; the emitted-but-dead token goes live; it becomes the real "seam A" that firmware
wiring plugs into later). Phase 3 then goes after the timing itself: the END GOAL is to
DISSOLVE the `XDNA_EMU_MAILBOX_LATENCY` knob, not re-tune it -- the ~8000 cycles should
EMERGE from modeling the per-cycle work (token transiting the fabric, firmware processing
the mailbox) rather than being a hardcoded constant. That is the whole reason firmware
emulation was on the table. (The wiring record `firmware-array-plugin-wiring.md` earlier
called seam A's latency "a calibration knob" as an interim stance; Phase 3's ambition is
to retire that stance, with HW as the check on the emergent number.)

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

## Design (AS LANDED: token-primary + logged Channel_Running fallback)

The originally-approved plan kept `channel_running` as a `debug_assert(token == running)`
migration guard. That was **superseded mid-implementation** (Maya approved the change):
a token only issues when a task sets Enable_Token_Issue (bit 31); `Channel_Running` goes
idle on **any** completion. So the two encode DIFFERENT events and legitimately disagree
on no-token tasks -- asserting equality would panic on the legacy path (e.g. the existing
`start_channel`-driven test). Landed instead as **token-primary with a logged fallback**:

```
sync_signal(sync, &device) -> {Token | Fallback | None}:   // NON-consuming peek
    abs = sync_abs_channel(sync, device)
    if engine(col,row).has_task_token_for_channel(abs): return Token   // faithful HW path
    <existing channel_running logic; sets sync.started>                // running->idle
    return Fallback if (running->idle | fast-completion) else None

is_sync_satisfied(sync: &mut, device: &mut) -> bool:       // CONSUMING transition
    if sync.satisfied: return true                          // consume-once latch
    match sync_signal(sync, device):
        Token    => pop_task_token_for_channel(abs); sync.satisfied = true; true
        Fallback => log::trace!("... fallback (no token)"); sync.satisfied = true; true
        None     => false
```

- **Two entry points, one signal.** `sync_signal` is the shared non-consuming peek.
  `is_sync_satisfied` (the CONSUMING transition) is driven ONLY from `try_advance`'s
  `BlockedOnSync` arm, which already holds `&mut DeviceState` -- so the read-only
  `syncs_satisfied` termination check stays on `&DeviceState` and its 8 call sites
  (route_graph.rs, xclbin_suite.rs) are UNTOUCHED. That works because those run only once
  `is_done()`, by which point every sync has already transitioned and latched.
- **Consume-once via a `satisfied` latch** on `PendingSync` (new field): the token is
  popped exactly at the transition, never re-checked. Correct multi-`WAIT_TCTS`-on-one-
  channel semantics (each sync needs its own token) that the non-consuming
  `channel_running` path lacked.
- **Fallback is legitimate, not a bug.** The trace log gives migration visibility (grep a
  sweep to see which kernels rely on the non-token path) without panicking on it.
- **New engine API:** `has/pop_task_token_for_channel(abs)` (channel-filtered, Phase 1)
  and `DmaEngine::issue_task_token(channel, controller_id)` -- the external token-injection
  point (the sync tests today; the seam-A firmware/array completion wiring later).

## TDD phases

1. **DONE `7f8cb62f`** -- Filtered consume API (`has/pop_task_token_for_channel`).
2. **DONE `67eeafeb`** -- Token-primary satisfaction + logged fallback + consume-once latch
   + `sync_signal` split + `issue_task_token`. Test `test_sync_consumes_matching_token_once`
   (two WAIT_TCTS on one channel need two tokens); existing no-token tests stay green via
   the fallback. `cargo test --lib` 3898 pass.
3. **IN PROGRESS (HW in the loop) -- thin slice: emergent token transport.** Scope chosen
   with Maya 2026-07-08 after grounding proved the ~8000 is a ONE-TIME firmware cold-start
   cost (the `mailbox_charged` latch), and that fully dissolving it needs firmware actually
   running (the array-blocked dream). The reachable slice NOW is the token's stream-fabric
   transit. See "Phase 3 design" below. The mailbox `8000` first-sync charge stays, now
   explicitly labeled the **firmware cold-start residual** -- the piece that cannot emerge
   until the firmware loop closes. MUST preserve pipelining (per-run `mailbox_charged`).
4. **Validation** -- `cargo test --lib` (row-2 compute syncs unchanged -> most tests green),
   then bridge/trace sweep on shim/mem-waiting kernels where the transit shift is visible;
   compare emergent cycle counts against HW (the oracle), NOT against the 8000 baseline.

## Phase 3 design -- emergent token transport (thin slice)

**Grounded finding (Explore, 2026-07-08):** `STREAM_FLUSH_CYCLES = 4` (charged every sync in
the `BlockedOnSync`->satisfied arm, `executor.rs:735`) is already the token transit distance
for a compute tile, FROZEN as a constant: a row-2 tile's TCT routes 2 hops down the column to
the firmware at the shim (row 0), and the per-hop stream latency is
`xdna_archspec::aie2::timing::INTER_TILE_HOP_LATENCY = 2` cy (derived in
`crates/xdna-archspec/src/model_builder.rs:266`). 2 hops x 2 cy = 4. TCTs route to the shim's
South port as column-control packets (`AIEGenerateColumnControlOverlay.cpp:168-187`); header
`col<<21 | row<<16 | actor_id` (`AIENpuToCert.cpp:143-147`). The cycle clock is live at the
charge point via `device.array.current_cycle` (<- `coordinator.total_cycles`).

**The change:** replace the flat `STREAM_FLUSH_CYCLES = 4` per-sync charge with
`transit = sync.row * INTER_TILE_HOP_LATENCY` -- hops-to-shim (the waited tile's row; WAIT_TCTS
matches (tile, actor) so the waited tile IS the producer) times the toolchain per-hop constant.
This REMOVES a hardcode in favour of a derived value (on the "derive from the toolchain"
mission). Emergent, not relabelled: the cost derives from a physical property (row = hop count)
x a toolchain constant, and `FlushingStreams` burns those cycles one-per-cycle running the
router. Behaviour shift: shim-DMA waits (row 0) 4->0, mem-tile (row 1) 4->2, compute (row 2)
stays 4. Approved to land (Maya, 2026-07-08) and validate against HW in Phase 4.

**Scope boundary (deferred, NOT this slice):**
- No `Token` struct change / no `ready_at` cycle-stamp / no stepping change. The change is
  localized to the executor charge point. The fully-faithful "token literally in-flight, gated
  at consumption by `ready_at <= current_cycle`" version earns its machinery only when firmware
  actually READS tokens at the shim (seam A / the dream).
- Column distance (horizontal hops for cross-column waits) deferred; row is the dominant,
  well-defined term.
- The firmware cold-start `8000` stays a residual. Possible future bridge: is the first-sync
  cold-start the SAME physical event as the SP-4a firmware-launch transient (CORE_CONTROL reset
  deassert)? If so, one physical model could generate both -- a real dissolution path, gated on
  the dream. Recorded, not pursued in this slice.

## Open decisions / risks
- Transit shift is the validation crux: is 0 for a shim-local token more correct than the old 4,
  or did the 4 absorb something real? Phase 4 HW check settles it (Maya chose "land it", no floor).
- Pipelining preservation must not regress multi-sync kernels into N x latency.
- Fast-completion path: a token may already be present before the first poll; the latch +
  filtered pop handle it (covered by the Phase 2 test).
