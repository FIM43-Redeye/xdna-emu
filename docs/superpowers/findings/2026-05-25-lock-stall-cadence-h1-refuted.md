---
name: 'HW LOCK_STALL emission is edge-every-cycle, matching EMU -- H1 refuted'
description: HW LOCK_STALL trace emission fires every stalled cycle, NOT on perf-counter rollovers. The Phase C "44 events at ~1024-cyc spacing" claim was a measurement error. Hypothesis H1 (perf-counter-driven cadence) is refuted by decoded HW captures showing 2233-2766 LOCK_STALL events per kernel run, matching EMU's current LOCK_STALL_TRACE_PERIOD=1 behavior. The actual EMU/HW asymmetry is wire-format compression -- HW uses skip-tokens for repeated-event runs in the trace buffer, EMU emits one packet per cycle -- which is what produced the appearance of EMU "over-emitting." Item #9 retargets from emission-cadence modeling to trace-unit compression.
type: project
---

# HW LOCK_STALL emission cadence: H1 refuted, real gap is trace compression

## TL;DR

The hypothesis that HW emits LOCK_STALL on perf-counter rollover (H1 in
`docs/coverage/cycle-accuracy-mission.md` item #9, motivated by Phase C's
"44 events at ~1024-cyc spacing" claim) is **refuted** by direct
measurement.

| Source | Decoded LOCK_STALL events | Bin content words | Distinct soc values |
|---|---:|---:|---:|
| HW (2026-05-24, chess) | 2233 | 160 | 2 |
| HW (2026-05-20, chess) | 2766 | 160 | 2 |
| EMU (2026-05-25, chess) | ~2700 | 3032 | per-event |

Both HW and EMU emit one LOCK_STALL marker per stalled cycle. The Phase
C "44 events" cannot be reproduced from any extant HW capture on this
test; it was a measurement error of unknown provenance.

The MLIR-configured PERF_CNT_2 threshold of 1024 IS real, but it
governs the **PERF_CNT_2 trace event** (which appears 12 times in HW),
not the LOCK_STALL cadence. The two are independent streams.

## Setup

Test: `_diag_phase_b_add_one_instrumented.chess` (Phase C / cycle-
accuracy reference test).

HW trace controller config from
`build/test/npu-xrt/_diag_phase_b_add_one_instrumented/traced/aie_traced.mlir`:

```
aie.trace.config @perf_core_0_2(%tile_0_2) {
  aie.trace.reg register = "Performance_Control1" value = 28
  aie.trace.reg register = "Performance_Control2" value = 458752
  aie.trace.reg register = "Performance_Counter2_Event_Value" value = 1024
  ...
}
```

So PERF_CNT_2 rolls over every 1024 cycles. That would set the PERF_CNT_2
trace event cadence to ~1024 cyc -- which it does (12 events spanning the
kernel run, consistent with ~12 × 1024 ≈ 12k cyc of activity).

But LOCK_STALL is a separate trace slot (slot 6 on core tile, mapped to
the LOCK_STALL trace event), independent of PERF_CNT_2.

## Method

1. Located preserved HW trace.bin files at
   `build/bridge-test-results/20260520/` and `.../20260524/`.
2. Decoded with `tools/parse-trace.py --decoder ours --trace-mode auto
   --out-events <out>.json`.
3. Counted LOCK_STALL events per (col, row), inspected the soc
   distribution.
4. Cross-checked EMU's events.json for the same test (chess compiler,
   today's bridge run).
5. Inspected raw trace_raw.bin to see how many content words actually
   carry data on each side.

## Observations

### Decoded event counts

HW captures contain 2233 (2026-05-24) and 2766 (2026-05-20) LOCK_STALL
events on tile (1, 2). EMU produces ~2700 events on the same tile after
its `LOCK_STALL_TRACE_PERIOD=1` change of 2026-05-11. The HW/EMU
difference (~20%) is smaller than the HW capture-to-capture variance
(2766 vs 2233 between days), suggesting both sides are tracking the
same underlying stall duration with minor run-to-run noise.

### Wire format asymmetry

HW trace_raw.bin: 160 content words (of 262144 word buffer).
EMU trace_raw.bin: 3032 content words.

Both decode to roughly the same event count. The 19× size difference
is explained by HW using **skip-tokens for repeated-event runs**, while
EMU emits one packet per cycle uncompressed. The post-decode event
list is semantically the same on both sides.

Sample of EMU bin around the LOCK_STALL run:

```
[5] 0xC5000330  [6] 0xC5000330  [7] 0xC5000330  [8] 0x00220001  [9] 0xC5000330
[10] 0xC5000330  [11] 0xC5000330  ...
```

`0xC5000330` is the LOCK_STALL marker pattern, repeated word-for-word.

Sample of HW bin in the same region:

```
[3] 0xC5000340  [4] 0xD8C6FEFE  [5] 0xC5080340  [6] 0xC5000340  [7] 0xDA91FEFE
```

Same `0xC5000340` marker, but interleaved with `0xD8C6FEFE` /
`0xDA91FEFE`-style words that are the trace controller's skip /
event-sync tokens. The decoder unpacks them into per-cycle records.

### soc clustering

HW: all 2233 LOCK_STALL events at exactly 2 distinct soc values
(832 and 864). EMU: per-event soc values, monotonically increasing.

Cause: the HW decoder assigns the chunk's bookkeeping soc to every
event unpacked from a single trace packet block, because the packet
header timestamp covers the whole block. EMU emits one packet per
event so soc is per-event.

This is what makes the ts/soc gotcha asymmetric and was the proximate
cause of today's misdiagnosis (ts inflated by ~event_count in EMU
because each LOCK_STALL packet is its own event-in-stream slot).

## Implications

### Item #9 retargets

Old framing (now refuted): "EMU over-emits LOCK_STALL on long-stall
tests (2701 events vs HW's ~44); model perf-counter-driven emission."

Actual situation: both sides emit one LOCK_STALL marker per stalled
cycle. The behavior matches.

New framing: "EMU's trace unit emits one packet per event with no
compression; HW uses skip-tokens for repeated-event runs. Implement
skip-token wire-format compression in EMU's trace unit so trace_raw.bin
sizes and per-event ts values track HW."

This is a **trace encoder change** in `src/device/trace_unit/`, not an
interpreter / cycle_accurate model change. The current
`LOCK_STALL_TRACE_PERIOD = 1` in `interpreter.rs:101` is correct and
stays. The 2026-05-11 framework change that motivated it is also
correct, though for different reasons than originally claimed.

### Phase C "44 events" annotation

The Phase C archive cites 44 LOCK_STALL events on
`_diag_phase_b_add_one_instrumented` and the ts/soc gotcha finding from
earlier today (`2026-05-25-trace-ts-vs-soc-measurement-gotcha.md`)
repeats that as the more-credible count. Neither matches the extant
HW captures from 2026-05-20, 2026-05-21, or 2026-05-24, all of which
show ~2233-2766 events.

Possible explanations for the original 44-event claim:

- Different test variant (e.g., a smaller iteration count that did
  produce fewer stall cycles)
- Different decoder mode (e.g., counting trace _packets_ rather than
  unpacked events; HW emits ~12 PERF_CNT_2 + small number of
  edge-event packets, plus the LOCK_STALL packet block(s))
- The original measurement looked at a different tile or a different
  event entirely (e.g., PERF_CNT_2 events, which DO have ~1024-cyc
  spacing and would fit "~44 events ~1024-cyc spaced" if the kernel
  ran longer than the diag's actual ~13k cyc)
- Pre-trim trace bin counted differently

I'm not annotating the Phase C archive itself; it's historical. But
the ts/soc gotcha finding gets a follow-up note pointing here, since
that's the active doc.

### What the campaign actually needs from item #9 now

Nothing on the modeling side -- the model is correct. The closeout work
for item #9 is:

1. Implement skip-token compression in EMU's trace unit (`src/device/
   trace_unit/`). HW emits skip tokens like `0xFExxxxxx` (visible in
   HW bin samples above as `0xFE` byte patterns) when N consecutive
   cycles have no event change. EMU should match.
2. Cross-validate: post-implementation EMU trace_raw.bin content words
   should drop from ~3000 to ~150-300 on the diag test, and per-event
   ts values should track HW's chunked timestamps.
3. The ts/soc gotcha then becomes irrelevant for cross-tool comparison
   because both sides produce comparable ts streams; soc continues to
   be the safer choice but the inflation gap closes.

PERF_CTRL config extraction (G2, both Python + Rust paths landed
today) stays useful as ground-truth-readout for the perf-counter
trace events themselves -- they're real, they're at the threshold the
MLIR specifies. Item #9 doesn't need it for cadence modeling though.

## See also

- `docs/coverage/cycle-accuracy-mission.md` item #9 -- to retarget
- `docs/coverage/hw-measurement-campaign.md` -- update G2 framing
- `docs/superpowers/findings/2026-05-25-trace-ts-vs-soc-measurement-gotcha.md` --
  the gotcha is real, but propagated the "44 events" claim that doesn't
  hold up; needs annotation
- `src/device/trace_unit/` -- where compression work lands
- `src/interpreter/core/interpreter.rs:97-101` -- LOCK_STALL_TRACE_PERIOD
  stays at 1 (correct)
- `src/interpreter/execute/cycle_accurate.rs:818-822` -- initial-entry
  emit stays
- `tools/parse-trace.py` -- decoder produces post-decode events.json
  identical on both sides
- HW captures: `build/bridge-test-results/{20260520,20260521,20260524}/
  _diag_phase_b_add_one_instrumented.{chess,peano}.hw/trace_raw.bin`
