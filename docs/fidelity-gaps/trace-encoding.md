---
class: trace-encoding
subsystem: trace unit frame encoding (held-level spans, cycles field, Repeat / Sync tokens)
posture: encoding-artifact -- the underlying mechanism (cycle_beat, level rising/falling) is faithful; the disagreement is in how spans are *encoded* into trace frames
status: encoding layer CLOSED (2026-06-27); one residual falling-edge case documented-and-bounded
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
  without one; Repeat0 up to 15, Repeat1 16..1023, 1023-chunks beyond). A held level is such a
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
  `parse.py` already has 1 << 18).
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
| Field overflow | Sync tokens | **FIXED** (leading Syncs). Trailing idle (a quiet tile at the end of the trace) also writes Syncs in real time on silicon (110 corpus tile streams are `Start Sync ...`); the emulator writes none at flush. | leading closed, trailing OPEN: the cycle a trailing Sync lands in is not pinned |
| INSTR_LOCK_ACQUIRE_REQ / _RELEASE_REQ stamp | c+1 | **FIXED** | closed |
| Mode-1 PC of INSTR_EVENT / CALL / RETURN / LOCK REQ | fetch PC, +3 / +4 bundles | **FIXED** in straight-line code | closed there |
| Mode-1 PC near taken branches and in delay slots | **not a fixed lead.** `br_j` (event `event #0` at every bundle from 5 before to 5 after a `j`; 12 skipped bundles before the target): lead 3, 7, ?, 7, 10 bundles for the five bundles before the branch, 5 and 8 in the first two delay slots (the PC is then in the skipped fall-through code, fetch running on sequentially), 3 in execution order in delay slots 3..5 (the PC is at the target). The same sites cost 0..7 extra cycles between consecutive events, cycling with the site's alignment (gaps 45..52 for 45 bundles): the front end stalls on a taken branch by an alignment-dependent amount and the fetch PC runs ahead meanwhile. | sequential walk of 3 / 4 bundles | **OPEN.** Needs the fetch / branch stall model (alignment of the target against the 16-byte fetch line); `br_j`, `br_jl` in `tools/gen_fetch_pc_probe.py`, data `build/experiments/trace-frame-pc/f1`, `f0` |
| Mode-1 PC and stamp of INSTR_LOAD / INSTR_STORE / INSTR_VECTOR | stamped at c+1 (load, store) and c (vector) in mode 0, PC +4 / +3 | issue PC; the emulator's mode-0 stamps of load / store differ (`straight_mem`: HW 25-cycle spacing between the two, emulator 24) and `vadd.32` raised no INSTR_VECTOR | **OPEN** |
| Hardware zero-overhead loop in the probe | the hand-written loops (`add.nc lc, r4, #0`, `movxm ls`, `movxm le`, body 12 or 40 bundles, aligned or not) ran **once** on silicon (4 iterations in the emulator); the compiler's loops work, so the set-up differs from the compiler's in something not found | iterates | **OPEN**: loop PCs unprobed |
| LOCK_STALL pulse of a request window | one 1-cycle pulse at c+4 | FIXED 2026-11-09 | closed |
| Held level at the start of the trace | first frame, one cycle after the start event | FIXED 2026-11-09 (`level_signals`) | closed |
