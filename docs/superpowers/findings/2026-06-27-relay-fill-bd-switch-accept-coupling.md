# #140 relay-fill: the recv accept cursor (per-BD pacing), and why "the encoder merges gaps" was wrong

**Date:** 2026-06-27
**Status:** RESOLVED to the static-feature ceiling. The recv **accept cursor**
is implemented, TDD'd, green, and **verified before/after on a freshly-rebuilt
`.so`** to re-pace the memtile S2MM-0 recv port from a front-loaded burst to
per-BD pacing. The encoder and BOTH trace decoders are correct; there is **no
encoder fix to make**. The single residual divergence from HW is a `prod_lock`
consumer-drain micro-timing stall (see "Remaining gap").

> **Correction to earlier revisions of this doc.** Prior revisions claimed the
> trace-unit *encoder* merged regularly-spaced 1-cycle PORT_RUNNING gaps,
> collapsing `[16,16,16,16]` into `[34,16,14]`, and that byte-identical output
> was "blocked on a separate encoder layer." **That was a stale-`.so` phantom.**
> The `[34,16,14]` is not an encode/decode artifact at all -- it is the genuine
> *pre-fix* front-loaded **signal**. See "The decisive before/after" below. The
> lesson is the one CLAUDE.md repeats: rebuild the FFI `.so` before trusting any
> bridge/trace result, and decode the actual emitted bytes before blaming the
> encoder.

## The decisive before/after (freshly-rebuilt `.so`, both decoders agree)

Memtile (col1,row1) S2MM-SEL0 = the recv-from-shim relay-fill port (slot0).
`add_one_using_dma.chess`, EMU-only, `XDNA_EMU_XFORM_PROBE=1` (XEDGE logs the
exact PORT_RUNNING/PORT_STALLED level edges fed to the encoder):

| build | ep0 RUNNING signal (XEDGE) | decoded slot0 (ours **and** upstream `parse.py`) |
|-------|----------------------------|--------------------------------------------------|
| **pre-fix** (accept cursor stashed) | `[34,16,14]` | `[34,16,14]` |
| **post-fix** (accept cursor) | `[16,16,2,14,16]` | `[16,16,2,14,16]` |

Two independent decoders (our `tools/trace_decoder/` and upstream
`aie.utils.trace.parse_trace`) decode the **actual `trace_raw.bin` bytes**
identically, and both match the raw XEDGE signal exactly, in both builds. So:

- The encoder emits correct bytes. Neither decoder merges anything.
- `[34,16,14]` is the **front-loaded recv signal**: 34 = the 2-deep `prod_lock`
  double-buffer (32 words) + the 2-deep input FIFO. Without per-BD accept
  gating, the slave FIFO absorbs the BD-switch bubble and the pop front-loads.
- `[16,16,2,14,16]` is the **per-BD-paced** signal the accept cursor produces.

HW reference (decisive capture, prior session): `[16,16,16,16]` -- a 1-cycle
stall at each of the three 16-word BD boundaries, no front-load, no multi-cycle
backpressure (HW's consumer keeps pace).

## What the accept cursor does (the real, correct model)

HW gates the memtile S2MM recv stream port (TREADY) **per-BD**: it accepts
exactly the BD's length (16 words), deasserts for the 1-cycle BD-switch
reconfiguration, then repeats -- independent of, and running ahead of, the
memory-write completion (the recv pop front-loads through the FIFO +
double-buffer). So the gate must be driven by **accepted-word count**, not
memory-write completion. An earlier "arm at memory completion" attempt
(`enter_chained_bd`) was insufficient for exactly this reason.

The accept cursor walks the BD chain by accepted-word count, distinct from
`current_bd` (the memory-write cursor):

- `ChannelContext` (`channel.rs`): `accept_bd: Option<u8>` +
  `accept_words_remaining: u32` (the accept-side cursor) and
  `bd_switch_accept_block: u16` (the 1-cycle deassert window).
- `init_accept_cursor` (`stream_io.rs`): called from `start_channel_with_repeat`
  for S2MM channels -- cursor starts at the first BD, `words = length/4`.
- `advance_accept_cursor` (`stream_io.rs`): called from `push_stream_in` on each
  accepted word -- decrements `accept_words_remaining`; at 0 arms
  `bd_switch_accept_block = bd_switch_bubble_cycles` and walks `accept_bd` to
  `bd_configs[accept_bd].next_bd` (next length).
- `can_accept_stream_in_for_channel` returns false while
  `bd_switch_accept_block > 0` (the BD-switch deassert), else FIFO-full.
- `route_tile_switches_to_dma` (`routing.rs`): when the channel refuses the
  stream, `consume_bd_switch_accept_block` elapses one cycle of the gap (no-op
  if the refusal was FIFO-full). All gap logic is in the route phase -- avoids
  the Phase-3-step / Phase-4-route ordering trap.
- Test: `s2mm_recv_deasserts_accept_at_each_bd_boundary` (`tests.rs`) -- locked
  2-BD ring, asserts no accept-run exceeds one BD's word count (RED was a
  ~32-word front-load). Full `--lib` green (3544/0); off1/#26 untouched.

`input_fifo_capacity()` is the 2-deep `STREAM_LOCAL_MASTER_FIFO_DEPTH` (not 4).

## Remaining gap (separate subsystem, not trace encoding)

Post-fix EMU slot0 is `[16,16,2,14,16]` vs HW `[16,16,16,16]`. Same 64 words,
but EMU's **BD3 accepts 2 words, stalls 7 cycles, then accepts 14** -- a
`prod_lock` double-buffer backpressure stall. HW's consumer (memtile MM2S ->
compute) drains fast enough that the recv port never hard-stalls; EMU's falls
behind by ~7 cycles on that one BD. This is the **consumer-drain / micro-timing
axis** -- a DMA/compute-timing fidelity gap, NOT trace encode/decode. CLAUDE.md
lists micro-timing among the areas that may need non-open-source cross-reference.
This is the open follow-up.

## Separability from off1/#26 (held)

off1/#26 (`chained_bd_inserts_port_running_bubble_at_each_boundary`) guards the
MM2S **send** bubble via `route_dma_to_tile_switches` + the
`enter_transfer_after_lock_grant` collapse. The accept cursor is on the S2MM
**recv** path (`route_tile_switches_to_dma` / `can_accept_stream_in_for_channel`)
-- a different function. `--lib` confirms off1/#26 stays green.

## Per-stage HW calibration (in progress — the prod_lock drain gap)

Goal: close EMU `[16,16,2,14,16]` -> HW `[16,16,16,16]` by calibrating the
pipeline-latency stages against HW. Progress so far:

**Mechanism (from `aie.mlir` memtile_dma + XFORM probe).** in0 is a 2-buffer ring
(`buff_0`/`buff_1`), `prod_lock` init=2, `cons_lock` init=0. recv S2MM-0 (bb1/bb2):
acquire prod>=1, write 16w, release cons. MM2S-0 (bb4/bb5): acquire cons>=1, drain
16w, **release prod at BD end**. So: recv BD1->buff_0 (prod 2->1), BD2->buff_1
(prod 1->0); **recv BD3 reuses buff_0 and blocks on prod_lock until MM2S-0 drains
buff_0 and releases prod.** HW frees it with a 1-cycle gap; EMU lags 7 cycles ->
the BD3 split + `s0st=1` stall (cyc 7947-7953).

**HW vs EMU port capture** (`XDNA_TRACE_MEMTILE_EVENTS="PORT_RUNNING_0,PORT_RUNNING_4"`,
both decoders agree), relative to each recv BD1 start:
- HW: recv `[16,16,16,16]` clean; the second traced port first-beats at rel-33.
- EMU: recv `[16,16,2,14,16]`; second traced port first-beats at rel-26 (EARLIER).

**Prime suspect REFUTED (provisionally).** EMU's MM2S-side port starts *earlier*
than HW's, yet EMU stalls -- so MM2S-*start* latency is not the cause. The gap is
downstream: the MM2S drain -> **prod_lock-release** timing (when a drained buffer
becomes reusable). Caveat to close first: confirm `PORT_RUNNING_4` is actually
MM2S-0-to-compute (the memtile has S2MM-0/1 + MM2S-0/1; the port<->DMA-channel
map must be pinned before trusting the rel-26/rel-33 numbers).

**Structural constraint for cross-tile stages.** Per-stage latencies that span
timer domains (memtile->compute) are NOT directly HW-measurable without
timer-sync (the PARKED BROADCAST_15 work) -- HW events in different
(col,row,pkt_type) domains have unsynchronized timers. The within-domain stages
(memtile S2MM-done -> MM2S, both memtile timer; compute S2MM -> core, both
compute timer) ARE measurable. The prod_lock-release stage above is
within-memtile-domain, so it is reachable now.

**RESUME (calibration):**
1. Pin the memtile port<->DMA-channel map (which PORT_RUNNING_N is MM2S-0); the
   XFORM `s4`/`ch6` was the drain in EMU but the traced PORT index needs
   confirming against the route config.
2. Measure HW vs EMU MM2S-BD-drain -> prod_lock-release timing within the memtile
   domain. Derive the correct release model from aie-rt (when the memtile MM2S
   releases the producer lock relative to its BD drain).
3. Cross-domain stages (memtile->compute, core->release) need timer-sync first;
   defer to the BROADCAST_15 arc or measure within-domain proxies.
Oracle: EMU memtile slot0 decodes to `[16,16,16,16]` AND off1/#26 + `--lib` green.

## Reproduction recipe

- `cargo build -p xdna-emu-ffi` (ALWAYS -- a stale `.so` is what produced the
  phantom this doc corrects).
- `XDNA_EMU_XFORM_PROBE=1 ./scripts/emu-bridge-test.sh --chess-only --no-hw
  --trace -v add_one_using_dma` -> XEDGE PORT_RUNNING/PORT_STALLED edges in the
  `.chess.bridge.log`; decoded output under
  `build/bridge-test-results/<date>/add_one_using_dma.chess.emu/`.
- Decode the actual bytes two ways (both must agree with the XEDGE signal):
  - ours: `tools/trace_decoder` `decode_words` + `rebuild_perfetto_mode0`.
  - upstream: `aie.utils.trace.utils.convert_to_commands` + the `parse.py`
    `convert_commands_to_json` walk (mind the MLIR-version skew in the
    `parse-trace.py --decoder` slot-name lookup; decoding bytes directly avoids
    it).
- Before/after: `git stash push <the 6 accept-cursor files>`, rebuild `.so`,
  re-run -> pre-fix `[34,16,14]`; `git stash pop`, rebuild -> `[16,16,2,14,16]`.
