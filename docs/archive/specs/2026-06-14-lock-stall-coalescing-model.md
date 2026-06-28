# LOCK_STALL trace-count gap: root cause and fix (task #136)

**Date:** 2026-06-14
**Status:** RESOLVED. The long-standing LOCK_STALL "2 vs 1 interval" gap was a
**measurement category error**, not an emitter limitation: the comparator was
grouping program counters as if they were cycles. Fixed with a mode-aware
comparator (option A). The cycle-cost half (the real tenant-4 timing win) was
shipped earlier as `737f5505` and is untouched.

**Supersedes** this document's earlier conclusion that the gap was "not
independently fixable" because mode-0 has no empty-frame close. That conclusion
was wrong -- it reasoned about the *encoder* while the artifact lived in the
*measurement pipeline*. The two reverted emitter attempts (per-transaction
pulse; per-cycle level "Option A-emitter") were chasing a phantom.

---

## TL;DR

- The compute-core trace (`LOCK_STALL`, `MEMORY_STALL`, `INSTR_VECTOR`, all core
  events) is captured in **mode 1 / EVENT_PC**. In that mode the per-event
  scalar is the **program counter** of the firing instruction, *not* a cycle.
- The comparator's `events_to_intervals` grouped those PCs into "cycle
  intervals." HW fires `LOCK_STALL` at **two PCs** -- 832 (the acquire
  instruction) and 864 (the release, 32 bytes later); EMU fires at one PC, 848.
  Grouping PCs as cycles produced the phantom "2 intervals vs 1."
- **Fix:** route comparison by trace mode. EVENT_TIME tiles (shim/memtile DMA,
  real cycles) drive the cycle-drift verdict; EVENT_PC tiles (core) are compared
  on the PC-anchored axis and excluded from the cycle-drift count. The core tile
  now contributes nothing spurious to the verdict.

## How the gap presented (the data)

`build/bridge-test-results/baseline-cycle-only/add_one_using_dma.chess.{hw,emu}/`
(`trace_raw.bin`, `events.json`, `trace_config.json`) -- decoded with
`tools/parse-trace.py --decoder ours`:

- `trace_config.json`: `"tracing": { "mode": "event_pc" }` -- the core is mode 1.
- Per-`pkt_type` `ts` ranges in HW `events.json`:

  | stream | pkt_type | mode | `ts` range | meaning |
  |--------|----------|------|-----------|---------|
  | core | 0 | EVENT_PC | 204-864 | **PCs** |
  | shim/PL | 2 | EVENT_TIME | 332174-338346 | real cycles |
  | memtile | 3 | EVENT_TIME | 335040-335691 | real cycles |

- HW `LOCK_STALL`: 1906 records at **two** PCs -- 832 (x1890, via Repeat) and
  864 (x16). EMU: 16 records at **one** PC, 848. `events_to_intervals` collapses
  same-PC records, yielding HW=2 "intervals" / EMU=1 -> the phantom count gap.
  (The "1890 records at one timestamp" Maya flagged is the decoder expanding a
  mode-1 `Repeat` at a fixed PC -- one record per held sample, all at PC 832.)

### The raw command streams confirm it

HW core (mode 1): `Start timer_value=333222` (a real absolute cycle, in the same
window as the shim/memtile cycles), then
`Single1(LOCK_STALL, PC=832) + Repeat1(919)` -- a held level lasting ~919
cycles, twice (~1872 cycles of DDR-fill acquire wait), then
`Single1(LOCK_STALL, PC=864)` (the release stall).

EMU core (mode 1): `Start timer_value=3499`, then 40 discrete `Single1` frames,
**zero Repeats** -- one frame per occurrence, no held durations.

Two distinct facts fall out:
1. EMU's trace timer starts near 0 (3499) vs HW's 333222 (HW includes the
   ~330k-cycle config-replay preamble). This is the known origin offset the
   per-anchor comparator already normalizes (sim power-on vs trace enable).
2. **The cycle is recoverable in mode 1** -- `cycle = Start.timer_value +
   frame_ordinal` (one trace sample per cycle). mode-1 carries *both* the PC
   (explicit) and the cycle (implicit). `rebuild_timeline_mode1` currently emits
   only the PC; the cycle is available if a future duration comparison wants it.

## Why the earlier "coalescing" framing was wrong

The "two spans 32 cycles apart" were never two time intervals -- they are two
PCs (acquire instruction, release instruction). There was no temporal-coalescing
problem to solve. Both reverted emitter attempts fought a comparator metric that
mis-reads PCs as cycles:

- **Per-transaction pulse** (reverted): over-counted where HW appeared to hold.
- **Per-cycle level "Option A-emitter"** (reverted): regressed 2/2 -> 2/3 on
  some kernels; still couldn't separate same-event stalls across an "unencodable
  gap" -- because the gap was a PC delta, not a cycle gap.

"Impossible to coalesce" was simultaneously true and irrelevant.

## The fix (option A: route by trace mode)

The comparator holds two physical quantities and must compare each in its own
domain:

- **Cycle domain** (EVENT_TIME: shim/memtile DMA events + the total-cycle
  ratio): drives the MATCH/DRIFT verdict, with the existing drift tolerance.
- **PC domain** (EVENT_PC: core stalls / instr events): compared as PC-set /
  multiset agreement on a separate axis (`compare_pc_anchored_for_tile`, the
  `--pc-anchored` report), and **excluded from the cycle-drift verdict**.

### Implementation

Producer -- per-event `mode` now travels through `events.json`:
- `tools/trace_decoder/frame.py`: `Event` gains `mode: int = 0`.
- `tools/trace_decoder/decode.py`: `_emit_event` takes `mode`; mode-1 rebuild
  emits `mode=TraceMode.EVENT_PC`, mode-0 defaults to EVENT_TIME.
- `tools/parse-trace.py`: flat event dict includes `"mode": e.mode`.
- Verified: core->1, shim/memtile->0 on both HW and EMU.

Consumer -- `src/trace/compare.rs`:
- `TRACE_MODE_EVENT_TIME`/`TRACE_MODE_EVENT_PC` constants; `EventRecord` gains
  `mode: Option<u8>`; `load_events_json` returns a per-tile `TileModes` map.
- `compare_tile_events` early-returns an empty `TileResult` for non-EVENT_TIME
  tiles -- so PC-domain tiles never feed the cycle-drift level/edge summary.
- The PC-anchored pass is now gated on `mode == EVENT_PC` (was `pkt_type == 0`).
- `tile_mode()` legacy fallback for events.json predating the field: cores
  default EVENT_PC, shim/memtile EVENT_TIME (every bridge trace setup configures
  cores as event_pc). Fresh data carries the real mode and is authoritative;
  the fallback only governs old snapshots, so nothing extractable is hardcoded.
- Callers updated: `compare_batch_with_opts`, `src/visual/data.rs`.

### Tests

- `mode_aware_dispatch_excludes_event_pc_tile_from_cycle_summary`: EVENT_TIME
  yields the phantom 2-vs-1 count mismatch; EVENT_PC yields empty.
- `tile_mode_defaults_core_to_event_pc_when_absent`: legacy fallback.
- Full suite green: 3502 Rust lib tests; 43 Python decoder/parse tests.
- End-to-end on the baseline: the Core tile contributes nothing to the
  Level/Edge summary; fresh and legacy events.json produce identical output.

## What this fixes and what it does not

- **Fixes:** the LOCK_STALL PC-as-cycle phantom across all core stall/instr
  events. The core tile no longer pollutes the cycle-drift verdict. Kernels
  whose only divergence was the LOCK_STALL phantom flip clean.
- **Does not touch (separate matters):**
  - **Real mode-0 divergences.** add_one still shows `EDGE_DETECTION_EVENT_0/1`
    (17/20, 30/29) on shim and `PORT_RUNNING_0/4` (6/2, 8/10) on memtile -- real
    cycle-domain comparisons, unrelated to the PC category error.
  - **Stall-duration fidelity (row 50).** HW holds LOCK_STALL ~1872 cycles
    (DDR-fill acquire) vs EMU ~16 -- the known DMA-delivery-optimism gap. It is
    captured where it physically lives (total-cycle ratio + mode-0 DMA anchors),
    not via a phantom count.

## Cycle-accuracy: scope clarification

The core's *internal per-instruction* cycle timing is not directly witnessed by
the core trace (it is PC-anchored). Cycle accuracy is validated through a
different, genuine channel: **mode-0 DMA anchors in real cycles** (shim/memtile,
`trace-anchors.py` -> `timing-three-way.py`) plus the **total-cycle ratio**.
The core's timing is bounded end-to-end by those, not witnessed instruction-by-
instruction. Not "no idea how cycle-accurate" -- "bounded end-to-end, not
witnessed per-instruction."

## Follow-ups (separate threads, not this fix)

1. **PR mode-1/mode-2 decode to mlir-aie.** Upstream `parse_trace` is mode-0
   only (this tree, HEAD 2026-05-19): an EventPC frame (`0xC4`) collides with
   the `Multiple0` mask and is parsed as garbage -- which is why it hung on the
   core trace. Our mode-1 (EVENT_PC) + mode-2 (INST_EXEC) decoders are
   PR-worthy. Issue-first (contributor stance). Re-check the latest upstream
   `main` first; the local tree is ~1 month stale.
2. **Core PC-sampling offset.** EMU reports LOCK_STALL at PC 848 while HW reports
   832/864, and INSTR_LOCK_ACQUIRE_REQ at 826 (EMU) vs 828 (HW) -- a small
   pipeline-stage / PC-sampling offset worth its own investigation on the
   PC-anchored axis.
3. **EMU encoder emits no Repeat/held-levels** (40 discrete Single1 vs HW's
   Single+Repeat). Whether to match HW's held-level encoding depends on whether
   we compare reconstructed durations (decoder normalizes) or raw frame shape.

## Regression gate (run 2026-06-14, RESULT)

Full bridge run on the fresh corpus (regenerated events.json with `mode`):

- **LOCK_STALL fix proven corpus-wide:** 0 of 140 kernels show any core
  stall/instr comparison line (was the phantom on ~80). The core/mode-1 tile is
  universally excluded from the cycle-drift comparison. All 148 (chess) + 78
  (peano) bridge functional tests pass, no regression.

### Two findings the run surfaced

1. **The bridge trace verdict was vacuous.** Phase 5 did `grep -q "CLEAN"` on
   the trace-compare report, and `format_report` always prints the line "Edge
   timing (CLEAN pairs only...)" -- so the verdict was CLEAN for every kernel
   that compared at all, regardless of divergence. It had been silently green on
   the entire corpus. FIXED: `format_report` now emits an authoritative
   `TRACE_VERDICT: CLEAN|DIVERGE` token (computed from the edge/level diverged +
   count-mismatch totals it already aggregates; mode-1 tiles contribute nothing);
   `emu-bridge-test.sh` Phase 5 parses that token. Unit test:
   `test_trace_verdict_token_clean_and_diverge`.

2. **The vacuous verdict was masking a pervasive real divergence.** With the
   honest verdict the corpus is **15 CLEAN / 125 DIVERGE**. All 125 are
   shim/memtile **mode-0** (genuine-cycle) divergences, concentrated in DMA/
   stream delivery-timing events:

   | event | mismatches (corpus) | status |
   |-------|--------------------:|--------|
   | EDGE_DETECTION_EVENT | 213 | shim; not yet root-caused |
   | CONFLICT_DM_BANK | 84 | memtile; not yet root-caused |
   | PORT_RUNNING | 70 | **already root-caused** -- bursty-DDR-delivery ("EMU delivers DMA too smoothly"), 2026-06-07 finding, row-50 family |
   | DMA_*_START_TASK | 12 | minor edge-count |

   These are NOT the PC-as-cycle error (they are mode-0, real cycles) and the
   LOCK_STALL fix did not cause them -- it *unmasked* them. This is the real
   trace-fidelity frontier: shim/memtile DMA-delivery timing. Next work.

Tree state at fix: `737f5505` (cycle-cost) + the mode-aware comparator + the
TRACE_VERDICT fix. Release `trace-compare` current. The 125-DIVERGE wall is the
honest baseline going into the DMA-timing fidelity work.
