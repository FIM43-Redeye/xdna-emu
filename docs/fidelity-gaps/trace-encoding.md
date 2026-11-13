---
class: trace-encoding
subsystem: trace unit frame encoding (held-level spans, cycles field, Repeat / Sync tokens)
posture: encoding-artifact -- the underlying mechanism (cycle_beat, level rising/falling) is faithful; the disagreement is in how spans are *encoded* into trace frames
status: encoding layer CLOSED (2026-06-27); one residual falling-edge case documented-and-bounded; packet hand-off timing, end of stream, first frame, inter-packet gap, trace stop instant and lock / DMA event selection OPEN (2026-11-12)
---

# Trace Held-Level Encoding Gaps

After the held-level re-architecture (`LOCK_STALL` / `PORT_*` / DMA
stall-starvation / core-stall family emit as B..E spans, decoded by upstream
`parse_trace`), the span *structure* is faithful. These are **timing/encoding
artifacts, not mechanism bugs**, and are stable Phoenix->Strix (DMA delivery
timing, not silicon-version-specific). Measured on
`vec_mul_trace_distribute_lateral` and `add_one_using_dma` (peano, interpreter
backend) vs real NPU1.

| Gap | Model vs hardware | Where | Status / rationale | Obs. |
|-----|-------------------|-------|--------------------|------|
| Held-level falling edge, close-to-empty | a held span whose deassert lands on a cycle with no *other* concurrent frame cannot encode its falling edge in mode-0 (no empty frame); the close defers to the next frame / end-of-segment. HW closes the span at the deassert cycle. | `src/device/trace_unit/mod.rs` (`set_event_level` / `commit_pending_frame`, the `active==0` defer); finding: 2026-06-08 falling-edge depends on concurrent levels (`xdna-emu-priv: research/findings/2026-06-08-lock-stall-falling-edge-depends-on-concurrent-levels.md`) | **DOCUMENTED.** Matches how HW closes a level span (no synthetic empty frame); the residual is only the *lone-level* case where no concurrent frame carries the close. With the per-cycle TRUE metronome the deferral is bounded to ~1 cycle. | confirmed |
| Held-level count under-emission (DDR-fill / bursty delivery) | our stream/DMA delivery was *smoother* than HW's bursty ~1024-cycle bursts, so level events asserted fewer times (e.g. memtile `PORT_RUNNING` ports decoded 71/187/90/156 cycles instead of the hardware law `sum(PORT_RUNNING) == words == 64` per port). | `src/device/trace_unit/mod.rs` (`commit_pending_frame`, skip-token held-level encoding); spec `xdna-emu-priv: research/specs/2026-06-08-skip-token-held-level-encoding.md` | **CLOSED (2026-06-27).** Root cause was **not** DDR jitter (that narrative was a metric artifact -- the cadence tool counted trace *frame-records*, not *spans*; corrected to span-based `tools/port-span-cadence.py`). The real defect was the encoder: pairs of RUNNING spans merged across the deassert->reassert gap. Four encoder/decoder/model fixes (`e0ce540f` decoder two-phase deactivate/activate; `ba1b4a59`† `Repeat(D-2)` for gap-opened lone holds; `2f85d1c9` `cycle_beat` set once per external interface, not on both push+pop; `5311905c`†+`90ce9fa8`† defer same-cycle level commits into the one-frame-per-cycle path) plus the two recv-path model fixes (see [`dma-stream-resources.md`](dma-stream-resources.md)) bring all four memtile ports to exactly 64. **Regression oracle: `sum(PORT_RUNNING) == words` per port.** All fixes are trace-only (data / cycle-counts / RUNNING-STALLED exclusivity unchanged). Full chain: finding `xdna-emu-priv: research/findings/2026-06-16-port-cadence-metric-was-frame-records.md`. | confirmed |

† Commit pruned from public history in the 2026-10 provenance cleanup (its
file also held removed trace code); the change is folded into the next
version of that file. The original commit is in xdna-emu-priv's
`archive/public-pre-round3/master`.

## Core-stall family conversion (coverage note, not a confirmed disagreement)

The core stall family conversion (MEMORY_STALL / STREAM_STALL / CASCADE_STALL as
held levels) is **not exercised by `distribute_lateral`** -- its trace config does
not include events 23/24/25 -- so those three have unit coverage but **partial
HW-trace validation**. MEMORY_STALL is now HW-validated and divergent (a
core-timing gap, tracked in [`core-compute-timing.md`](core-compute-timing.md));
STREAM_STALL / CASCADE_STALL still await a cascade/stream-stall kernel capture.

Two related modeling notes (not HW-confirmed disagreements, hence not rows
above): mode-1 (EventPc) PC anchoring is dropped for stall held levels (the level
path carries no PC, matching the LOCK_STALL precedent), and the executor-level
*intra-bundle* structural bank conflict stays a bounded MEMORY_STALL pulse (it
resolves within one executor step, so a same-step rising+falling would encode as
a zero-width span).

## Frame model and the REQ stamp (2026-11-09, silicon vs emulator, fixed)

`build/experiments/trace-frame-pc`, probes `tools/gen_lock_pulse_probe.py` (`cases()`, `quiet_p*`)
and `tools/gen_fetch_pc_probe.py`; silicon corpus statistics from 817 mode-0 tile streams of the
bridge corpus plus the probe captures (`corpus_scan.py`).

What silicon does, and the encoder now follows (`commit_pending_frame`, `push_frame_run`,
`push_syncs`, `push_op`):

- **A frame per active cycle.** Every cycle with anything asserted writes a frame (held levels
  included); its cycles field is the *idle* cycles since the previous frame, the frame's own cycle
  being the decoder's implicit +1 (markers 2..1500 bundles apart decode `ts` 2..1502 apart, a
  frame in the next cycle has 0). The start cycle counts as the previous frame. `ts` is therefore
  the true cycle on silicon, and since the emulator writes the same, on both; `soc` (= `ts` minus
  frames) drifts one cycle per frame on both.
- **Identical consecutive frames are one frame plus Repeats** (0 of 44k silicon frames repeat
  without one; Repeat0 up to 15, Repeat1 16..1023, 1023-chunks beyond, each Repeat1(1023) written in the cycle of the next identical frame (one after its 1023rd repeat), not when the run ends: every shim packet in a quiet stretch of `objectfifo_repeat_init_values_repeat` ends in 11 of them; a run-ending Repeat is written in the cycle of the next different frame, an event frame's token one cycle after its own cycle, and a completed packet's first word reaches the shim S2MM 10 cycles after its last byte is written, 8 of them the trace unit's output stage, `stream_switch.trace_packet_emit_cycles`). A held level is such a
  run: its open frame carries the idle cycles before it, the rest are `(mask, 0)` frames, a pulse
  inside it is its own frame and the cycle after it a held-only frame again. This replaces the
  skip-token / skew-debt bookkeeping, which approximated it.
- **No multi-byte opcode straddles a 32-bit payload word**: the rest of the word is Skip (0xFE)
  (12.8k fills in the corpus, none where the opcode would have fit).
- **Field overflow is Sync tokens.** The field is 18 bits; idle cycles beyond it are one Sync
  (0xFF, then a Repeat for more) per wrap, 2^18 cycles each, before the frame, which carries
  `idle mod 2^18`. A wrap landing in the frame's own cycle is lost: idle = 2^18 is no Sync and a
  field of 0, idle = 2 * 2^18 one Sync and 0. A Sync ends a run of identical frames. Silicon:
  `quiet_p262142..1000000`, PERF_CNT_2 period 262143..1000001. The in-tree decoder took a Sync as
  0x3FFFF and a Repeat after it as a repeat of the previous event (fixed; upstream mlir-aie's
  `parse.py` already has 1 << 18). Each Sync is written when its wrap comes, in `L + k * 2^18 + 2`
  after a frame of cycle `L`, whether a frame follows or not; later Syncs of the stretch are a run
  of the first (no-data trace probe: a packet whose 28th byte is the Sync reaches the shim S2MM at
  `L + 2^18 + 14`, about 3000 cycles before the next frame, and the same with no frame before the
  stop).
- **End token** (opcode `110111xx`, byte `0xDC`; the name is inferred): 4 bytes, 4-byte aligned, `0xDC b1 b2 b3`, an
  18-bit count (low 2 bits of `b1`, then `b2`, `b3`) of idle cycles from the last frame to the stop; Syncs before it
  carry whole 2^18 wraps; it is the last non-Skip token of a stream. Written by the emulator at the stop, after the
  Syncs not yet due (see the end of stream row below).
- **A self-resetting performance counter fires every threshold + 1 cycles** (PERF_CNT_2 at 1024:
  frames 1025 apart in 206 of 206 clean core captures; the emulator fired every 1024). The cycle
  after a reset counts nothing (`PerfCounterBank::reset_hold`). Only the self-reset case is
  measured; the same cycle is lost for a reset by any other event, unmeasured.
- **INSTR_LOCK_ACQUIRE_REQ / _RELEASE_REQ at c+1**, a request in the bundle at cycle c, granted or
  refused, so a core held across the trace start has no REQ for its acquire; the emulator stamped
  completion (c+4, or the end of a hold). Raised by the next step that issues a bundle, so a
  request issued at c+3 behind another one is at c+5, after the window's LOCK_STALL pulse (c+4).
- **Mode-1 PC is the fetch PC**, in straight-line code `FETCH_LEAD_BUNDLES` = 3 bundles after the
  bundle executing in the cycle the event is stamped: INSTR_EVENT_0/1 (40 sites), INSTR_CALL,
  INSTR_RETURN, INSTR_VECTOR at +3; INSTR_LOCK_*_REQ, INSTR_LOAD, INSTR_STORE stamped one cycle after
  issue, at +4. The emulator carried the issue PC (`trace::trace_pc`; load / store / vector keep it,
  see below).

Probe: the 54 lock-pulse cases, window frames (INSTR_EVENT_0 to INSTR_EVENT_1) of the core
trace silicon against emulator: 50 identical, the other four differ only in the length of a held
acquire (hold duration is the DMA / producer timing gap) or the phase of the PERF_CNT_2 pulse.

| Gap | Silicon | Emulator | Status |
|-----|---------|----------|--------|
| Cycles field of a frame | idle cycles between frames | **FIXED** (above) | closed |
| Repeat compression of identical frames, word alignment of opcodes | as above | **FIXED** | closed |
| Field overflow | Sync tokens | **FIXED** (leading Syncs; Syncs written as their wraps pass, trailing ones included, 2026-11-12). | closed |
| INSTR_LOCK_ACQUIRE_REQ / _RELEASE_REQ stamp | c+1 | **FIXED** | closed |
| Mode-1 PC of INSTR_EVENT / CALL / RETURN / LOCK REQ | fetch PC, +3 / +4 bundles | **FIXED** in straight-line code | closed there |
| Mode-1 PC near taken branches and in delay slots | **not a fixed lead.** `br_j` (event `event #0` at every bundle from 5 before to 5 after a `j`; 12 skipped bundles before the target): lead 3, 7, ?, 7, 10 bundles for the five bundles before the branch, 5 and 8 in the first two delay slots (the PC is then in the skipped fall-through code, fetch running on sequentially), 3 in execution order in delay slots 3..5 (the PC is at the target). The same sites cost 0..7 extra cycles between consecutive events, cycling with the site's alignment (gaps 45..52 for 45 bundles): (the "0..7 extra cycles" are not a stall: the taken branch resumes at its line start and re-runs the bundles before the label; counter reads show no branch penalty, see `core-compute-timing.md` "Taken branches cost no extra cycles"). The PC is also not a line-granular fetch pointer: its lead in straight code is 8 / ~14 / ~36 bytes for 2 / 6 / 16-byte bundles. | sequential walk of 3 / 4 bundles | **OPEN.** Needs a decode-stage PC model whose lead depends on bundle sizes; `br_j`, `br_jl` in `tools/gen_fetch_pc_probe.py`, data `build/experiments/trace-frame-pc/f1`, `f0` |
| Mode-1 PC and stamp of INSTR_LOAD / INSTR_STORE / INSTR_VECTOR | stamped at c+1 (load, store) and c (vector) in mode 0, PC +4 / +3 | issue PC; the emulator's mode-0 stamps of load / store differ (`straight_mem`: HW 25-cycle spacing between the two, emulator 24) and `vadd.32` raised no INSTR_VECTOR | **OPEN** |
| Hardware zero-overhead loop in the probe | the hand-written loops ran **once** on silicon: the loop registers are read at fetch, four lines ahead of issue, so LE must be a line start and LE written at least five lines before the loop end (rule table in `core-compute-timing.md` "Zero-overhead loop set-up") | iterates only when the set-up is in time (`src/interpreter/state/fetch.rs`) | **FIXED 2026-11-09**; loop PCs (mode 1) are still the sequential walk, see the branch row |
| LOCK_STALL pulse of a request window | one 1-cycle pulse at c+4 | FIXED 2026-11-09 | closed |
| Held level at the start of the trace | first frame, one cycle after the start event | FIXED 2026-11-09 (`level_signals`) | closed |

## Packet hand-off, end of stream, first frame (silicon vs emulator, 2026-11-12)

The token stream of a tile (frames, Repeats, Skip padding) now matches silicon packet for packet in a quiet
stretch of `objectfifo_repeat_init_values_repeat` (28 bytes: `Repeat0(10)`, a frame, `Repeat0(7)`, `Skip`, a
multiple-cycle frame and eleven saturated `Repeat1(1023)`). What differs is when the bytes are written, when
the packet leaves the trace unit, and how a stream begins and ends.

Method: packets of the shim that carries the trace (shim S2MM channel 1), six tests (`repeat_init_values_repeat`,
`add_one_objFifo_elf`, `add_one_two_txn`, `add_one_ctrl_packet_4_cores`, `objectfifo_channel_turns`,
`objectfifo_turns_walkers`), three silicon captures, one emulator run. The k-th packet in buffer order is
matched to the k-th 8-cycle non-starved slot of that channel's `STREAM_STARVATION` level (this assumes buffer
order is arrival order); delay = arrival cycle minus the last cycle the packet's own tokens cover, on the
shim's timer. Only the shim's own packets are tabulated, because tile timers are not synchronised.

| Last token of the packet | Silicon (cycles, 3 captures pooled) | Emulator |
|---|---|---|
| Single-cycle frame | 11 in 48 of 48 | 2 in 17 of 17 |
| Multiple-cycle frame | 11 in 25 of 26 (one 17) | 2 in 5 of 5 |
| Saturated `Repeat1(1023)` | 11 in 149 of 149 | 3 in 51 of 51 |
| `Repeat0` or unsaturated `Repeat1`, a frame in the next cycle | 11 in 47 of 48 (one 18) | `Repeat0` 3 in 6 of 6; `Repeat1` 9 in 8 of 9 (one 3) |
| `Repeat0` or unsaturated `Repeat1`, idle after it | 8 cases: gap + 10 in 7 (gap 5: 15; gap 9: 19, six cases), one case at gap 5 is 20 | 3 cases, gap 9: 11 (gap + 2) |
| Single-cycle frame, idle after it | 1 case, gap 9: 11 | none |
| `Skip` padding (end of trace) | 12 packets, 11 to 1033 | 8 packets, 3 to 633 |

"Gap" is the distance in cycles from the last covered cycle to the tile's next frame.

| Gap | Silicon | Emulator | Status | Obs. |
|-----|---------|----------|--------|------|
| Packet hand-off timing | A packet's first word reaches the shim S2MM 10 cycles after the byte that completes it is written. Event frames therefore reach it 11 cycles after their own cycle, so the frame token is written one cycle after the frame's cycle (inferred from the arrival time). A Repeat that ends a run is written when the next frame arrives (gap + 10). A saturated `Repeat1(1023)` is written in the cycle of the next identical frame (11). A packet finished by the Skip fill in front of a token that does not fit leaves one cycle sooner (10 after the cycle that token covers up to). | Modelled 2026-11-12: Repeat / frame write cycles, the 8-cycle output stage (`stream_switch.trace_packet_emit_cycles`), the fill one cycle before its token; the shim's own packets after long idle no longer pay a gate wake. Over three captures of every traced bridge test the shim's own packets match the silicon delay table in every class (Single, Multiple, Repeat0, saturated and unsaturated Repeat1, fill-finished). | **FIXED** on the shim. The 8-cycle stage of the core, memory-module and memtile trace units is inferred (their timers are not synchronised with the shim's). | confirmed (shim), inferred elsewhere |
| End of stream | Each tile's mode-0 stream ends with a 4-byte End token: `0xDC` then an 18-bit big-endian count (low 2 bits of byte 1, bytes 2 and 3), i.e. the opcode `110111xx` with the same 18-bit field width as the cycles field of a frame. Measured over three captures (about 780 streams each): byte 0 is `0xDC` in every token; the token is the last non-Skip token of its stream in every stream that has exactly one (4-byte aligned in the payload, the rest of the packet is Skip); the count is the idle cycles from the last written frame (including its Repeats) to the stop, identical for every tile of a test once Syncs are added back (spread 0 in 2211 of 2211 streams; longer idle is written as Syncs, 2^18 cycles each, before the token, and the count holds the remainder); held-level tiles end with count 0 (shim 77%, core 30%, never memtile or memory module). A quiet tile is `Start`, optional Syncs, token. A stream traced in several start/stop rounds carries one token per round. The token is sometimes absent when it would need a packet of its own (about a third of those cases, mostly core streams; cause open). Padding packets at the end of the trace arrive 11 to 1033 cycles after the last covered cycle (12 packets). The name End is inferred; the +0 / +1 convention of the count against the stop cycle is inferred (every tile shares one stop instant, so no capture pins it). | The emulator writes the token at the stop (mode 0), after the Syncs not yet due; Syncs whose wrap came earlier went in as it passed. The decoder and the stream sweep read the count. A token that would need a packet of its own is omitted. | **OPEN** for the rest: when silicon keeps or drops a token that needs a packet of its own. Whether the count is the stop minus the last frame cycle or one less is not pinned. What the late padding arrivals depend on is not pinned. | confirmed |
| First frame of a stream | The shim's first frame is the starved level at once: `Single(mask 128, cycles 0)` on `repeat_init_values_repeat` (3 captures) and on most shim streams of the corpus (shim S2MM channel 1 `STREAM_STARVATION` is high from one cycle after the trace Start, so the held run begins at Start + 1; 40 of 204 shim streams have a different first frame, for example another channel's mask at 700 to 3000 cycles). Inferred: the channel was started (BD pushed, no stream data yet) before the trace Start event. | The same frame carries a cycles field of 218: the starved level begins 218 idle cycles after the start, and once raised continues exactly as silicon's does. Inferred: the emulator's latency from the task-queue write to the starved state (channel start), which silicon does not show in front of the trace start. | **OPEN.** The gap between the task-queue write and the start event on silicon is not measured, and the origin of the constant 218 is not traced. | confirmed |
| Trace carrier residuals (shim S2MM channel 1 `STREAM_STARVATION` interval counts) | `add_one_ctrl_packet_4_cores` shim (2,0): Chess 50 / 52 / 52, Peano 52 / 52 / 52. `dynamic_object_fifo_two_core_sliding_window` shim (0,0): 12 / 11 / 12. | After the saturated-Repeat fix (`f3813fc`): 60 (Chess), 59 (Peano); 14. Other carriers are below silicon: `add_one_objFifo_elf` 8 against 11 / 13 / 11, `add_one_two_txn` 8 against 9 / 10 / 11, `objectfifo_channel_turns` 30 against 34 / 33 / 32. | **OPEN.** Probably downstream of the hand-off timing above, since each arriving packet ends a starved stretch (untested); rescore once that row is modelled. | confirmed |
| Inter-packet gap at a shared packet-switched port | At the carrier shim's S2MM channel 1, consecutive packets from different sources that were both waiting (gap of 2 cycles or less between the last word of one and the first word of the next) are separated by 0, 1 or 2 starved cycles: 18 / 19 / 36 pairs over three captures (25% / 26% / 49% of the 73 queue-limited pairs); consecutive packets of one source never queue this way (0 of 231). A gap of 2 is almost always in front of a packet from the carrier shim's own trace slave (31 of 43 such pairs); the split of 0 against 1 for other seconds (14 against 15 of 41) follows no attribute in the data (source kind, column, hops, start-cycle phase). Inferred candidate: whether the second packet's header was already waiting in its slave when the first one's TLAST passed. No documented per-packet arbitration figure exists. | The arbiter is released in the step the slave pops TLAST and the target's pending has drained; a waiting slave is granted in the same step if it is visited later in the slave loop, else one step later: 0 or 1 steps depending on slave index order, no per-packet cost. The gap of 2 in front of the carrier's own packets is the local slave pipeline. In one emulator run the same pairs give 0 in 16, 1 in 1, 2 in 14: gap 2 and 0 are reproduced, gap 1 almost never. | `src/device/stream_switch/mod.rs` (`StreamSwitch::step_packet_routes`) | **OPEN, needs a probe.** The six-test corpus cannot separate 0 from 1 because the second packet's readiness is not observable; a probe with two controlled sources into one shim is needed. Consequence in the carriers: the emulator leaves some packets back to back (spacing 8) where silicon leaves one or two starved cycles. | confirmable: two-source probe into one shim |
| Trace stop instant | Silicon's trace stops at the same instant for every tile of a test, stable between captures (core streams run 36399 / 36367 / 36482 cycles from Start to the stop on `add_one_ctrl_packet_4_cores.chess`). | The emulator keeps tracing about 7.4k to 8.0k cycles (about 7.7 periods of the 1025-cycle `PERF_CNT_2` tick) past that point: its last frame is 44057 cycles after Start (43938 Peano) on the same test. That accounts for the extra periodic tokens (about 8 per core stream) and the +10 packets of that carrier (74 against 64), and with them the residual event-count difference there. Neither Start-only streams nor the missing End token contributes. | trace stop / runtime-sequence timing, not the trace unit's encoding | **OPEN.** Attributed to host / runtime-sequence timing of the stop, not pinned: the stop is the last two register writes of the sequence on silicon, and why the emulator's lands later is not checked. | confirmed |

## Event selection registers (silicon vs emulator, 2026-11-12)

| Gap | Silicon | Emulator | Where | Status | Obs. |
|-----|---------|----------|-------|--------|------|
| Lock event selection | Lock events read `Locks_Event_Selection_n`: memtile `0xC0400 + 4n` (n = 0..7, `Lock_Select` bits 21:16, 64 locks), compute memory module `0x1F100 + 4n` (n = 0..7, bits 19:16), shim `0x14100 + 4n` (n = 0..5, bits 19:16; the events are named `LOCK_n_*` but follow the same scheme). `Lock_Value` (bits 5:0, reset 0x3F) is the trigger value of `LOCK_SELn_EQUAL_TO_VALUE`. Reset selects lock 0 for every n. Layout from the register database, not silicon-confirmed. | The registers are never read; `LOCK_SELn` is hardwired to lock n (acquire-GE and release events only, locks 0..7 on compute and memtile, 0..5 on shim). `ACQ_EQ` and `EQUAL_TO_VALUE` lock events are not emitted. A trace that points a selector at another lock, or traces a memtile lock above 7, differs from silicon. | `src/trace/mod.rs` (`mem_event_to_hw_id`, `memtile_event_to_hw_ids`, `shim_event_to_hw_id`) | **OPEN.** Needs the selector registers read at trace configuration, the lock id resolved through them, and the two missing event kinds. | confirmable: a probe that programs a selector and traces the lock |
| Memtile DMA selector events | `DMA_{S2MM,MM2S}_SEL{0,1}_*` follow `DMA_Event_Channel_Selection` (`0xA06A0`, channel fields at bits 2:0, 10:8, 18:16, 26:24); the family includes `MEMORY_STARVATION` and `MEMORY_BACKPRESSURE` (events 41 to 44). | The memtile mapping covers start task, finished BD / task, stalled lock and stream starvation / backpressure per channel; the memtile `MEMORY_STARVATION` / `MEMORY_BACKPRESSURE` events are not emitted. | `src/trace/mod.rs` (`memtile_event_to_hw_ids`) | **OPEN.** | confirmable |
