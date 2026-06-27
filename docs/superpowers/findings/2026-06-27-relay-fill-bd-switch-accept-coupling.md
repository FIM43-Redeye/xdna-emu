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

## Remaining gap: S2MM ingress buffering depth (NOT lock timing)

Post-fix EMU slot0 is `[16,16,2,14,16]` vs HW `[16,16,16,16]`. Same 64 words,
but EMU's **BD3 accepts 2 words, stalls, then accepts 14**. Earlier revisions of
this doc called that a `prod_lock` consumer-drain / MM2S-timing gap. **A clean
absolute-cycle HW+EMU co-capture refuted that** (see "Root cause" below). The
real gap is the **S2MM ingress buffering depth**: HW's recv port absorbs a full
lock-stalled BD into a deep DMA ingress FIFO and keeps PORT_RUNNING clean, while
EMU's 2-deep stream-switch master-port FIFO backpressures after exactly 2 words.

## Separability from off1/#26 (held)

off1/#26 (`chained_bd_inserts_port_running_bubble_at_each_boundary`) guards the
MM2S **send** bubble via `route_dma_to_tile_switches` + the
`enter_transfer_after_lock_grant` collapse. The accept cursor is on the S2MM
**recv** path (`route_tile_switches_to_dma` / `can_accept_stream_in_for_channel`)
-- a different function. `--lib` confirms off1/#26 stays green.

## Root cause: clean absolute-cycle co-capture (2026-06-27)

**Port map PINNED (verified two ways).** `_memtile_default_port_config`
(`tools/mlir-trace-inject.py:253`) returns `(slot % 4, slot < 4)` where the bool
is `master`, mirroring mlir-aie `setup.py::_get_default_events_for_tile`. The EMU
runtime confirms it (`coordinator.rs:1096-1102`: `event_port_selection[ep]` ->
`(port_idx, is_master)` -> `masters[port_idx]` for S2MM / `slaves[port_idx]` for
MM2S). So for add_one_using_dma: **`PORT_RUNNING_0` (ep0/slot0) = memtile S2MM-0
= recv-from-shim; `PORT_RUNNING_4` (ep4/slot4) = memtile MM2S-0 = drain-to-compute.**

**Compiled config (`chess/.../input_with_addresses.mlir:105-136`).** in0 is a
2-BD S2MM ring (bd0->bd1->bd0...), `prod_lock` init=2, buffers buff_0@0 / buff_1@64
(16xi32 each). S2MM bd0/bd1: `AcquireGE(prod,1)`, write 16w, `Release(cons,1)`.
MM2S bd2/bd3: `AcquireGE(cons,1)`, drain 16w, `Release(prod,1)`. So recv's 3rd
fill = bd0 again (buff_0) and must `AcquireGE(prod,1)` -- prod hits 0 after the
first two fills, so the recv-BD3 reuse genuinely depends on MM2S draining buff_0.

**Absolute-cycle co-capture** (one matched bridge run,
`XDNA_TRACE_MEMTILE_EVENTS="PORT_RUNNING_0,PORT_RUNNING_4"`; both decoders agree;
decode_abs.py on each side's `trace_raw.bin`):

| side | slot0 recv (B-E per BD) | slot1 MM2S-0 (first drains) |
|------|--------------------------|------------------------------|
| HW   | 2274-2290, 2291-2307, **2308-2324**, 2325-2341  (clean `[16,16,16,16]`) | **2309**-2317, 2318-2326, ... |
| EMU  | 4413-4429, 4430-4446, **4447-4449 (2) +stall+ 4456-4470 (14)**, 4471-4487 | **4439**-4455, 4456-4472, ... |

**Both prior suspects REFUTED with this data:**
- *MM2S start latency*: HW MM2S first-beats at **2309** (19cy after recv BD1
  cons-release at 2290); EMU at **4439** (10cy after 4429). EMU's MM2S starts
  *earlier*, not later -- the old "rel-33 HW / rel-26 EMU" comparison had the
  HW number wrong (it predated the pinned map). MM2S timing is not the cause.
- *prod_lock-release timing*: the EMU model is already faithful -- recv BD3
  resumes at 4456, exactly 1cy after MM2S finishes draining buff_0 (4455) and
  releases prod inline (`apply_lock_release_direct`). aie-rt confirms release
  fires after the full BD drain, serialized, no early/pipelined token, and gives
  no cycle timing. Nothing to fix here.

**The decisive observation.** HW recv BD3 (2308-2324) starts **one cycle before**
MM2S even begins draining buff_0 (2309) and runs to completion *overlapping* that
drain. The recv port cannot be writing buff_0 yet (prod still 0) -- so HW is
**accepting the full 16-word BD into a deep S2MM ingress FIFO ahead of the
lock-gated buffer write**, and PORT_RUNNING reflects those port beats. EMU
accepts exactly **2** words (= `input_fifo_capacity()` =
`STREAM_LOCAL_MASTER_FIFO_DEPTH`, `stream_io.rs:30`) then backpressures until prod
frees buff_0. The "2" is the smoking gun: the split is the FIFO depth, not a lock
or drain-timing lag.

**Design tension (why this is not a one-line bump).** The 2-deep value is
AM020-faithful for the stream-switch *local master port* FIFO, and was made
shallow deliberately: a prior hardcoded 256-word ingress buffer caused an
over-long double-buffer warmup transient (root-caused 2026-06-13; see the
stream_io.rs comment). EMU folded the DMA's ingress staging into that same
2-deep port FIFO ("rather than adding a deep decoupling buffer"). The HW evidence
says the memtile **S2MM DMA channel** has its OWN ingress FIFO, deeper than the
master port (>=16w for this kernel), distinct from the (correctly 2-deep) port.
The fix is to model that second FIFO at its real depth -- WITHOUT reintroducing
the warmup transient the 256-word buffer caused.

**RESUME (root cause now known):**
1. Derive the memtile S2MM DMA-channel ingress FIFO depth from AM025 / aie-rt
   (distinct from `STREAM_LOCAL_MASTER_FIFO_DEPTH`). Do NOT guess a number.
2. Model it as a separate buffer downstream of the 2-deep master port: the recv
   accept cursor should gate on `min(port-FIFO, DMA-ingress-FIFO)` space, so a
   lock-stalled BD can stage in the DMA FIFO while PORT_RUNNING stays asserted.
3. Re-run the dma_passthrough / double-buffer warmup sweep that motivated the
   2026-06-13 shallowing -- confirm no warmup-transient regression.
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
