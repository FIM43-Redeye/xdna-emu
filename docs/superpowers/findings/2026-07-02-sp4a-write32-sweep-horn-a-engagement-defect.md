# SP-4a cold-start fill-state: the write32 sweep proves a REAL EMU dataflow-engagement defect (Horn A)

**Date:** 2026-07-02  **Issue:** #140 (timer-sync arc follow-on; the SP-4a
cold-start fill-state gap)
**Status:** DISPOSITIVE. The gap is a real emulator defect, NOT a trace
artifact. The prior session-3 lean-toward "retire the byte-match premise" is
refuted. Fix locus identified; fix itself OPEN.
**Evidence on disk:** `build/experiments/sp4a-write32-sweep/` (gitignored):
`of_q0_lean_sweep.py` (variant generator), `build-variant.sh`, `run-variant.sh`,
`analyze.py`, and `N{0,8,64,256}/{build,hw,emu}/`.

## What was tested

The SP-4a oracle: in the lean Q=0 objectfifo kernel (ProdCore(0,2) ->
MemTile(0,1) -> ConsA(0,3), pure lock/DMA, no compute; `of_q0_lean.py`), the
prod->consA first-`LOCK_STALL` cross-domain offset is **+2 on HW** (range 0) but
**-52 on EMU**, and the shim S2MM drain STREAM_STARVATION lands at **t+13 on HW**
(pipeline EMPTY at drain-start) vs **t+1683 on EMU** (pipeline FULL). Three prior
sessions left this ambiguous; session 3 (2026-06-30) validated the control-path
*components* (write32 ~100cy/pkt; the 3050cy S2MM dispatch gate, HW-measured) and
leaned toward "the offset is a trace-perturbed cold-start sample, retire the
byte-match."

The knot session 3 never resolved: if the control-path constants are faithful,
HW runs the *same* traced insts.bin through the *same* ~6672cy pre-drain window
with the *same* CDO-enabled free-running cores -- so HW should fill to the same
deadlock EMU reaches. But HW is empty at the baseline (starve t+13) and EMU is
full (t+1683). A trace confound *identical in both worlds* cannot produce a
*between-world* fill divergence. So either the windows differ (control-path
composition defect) or HW simply doesn't fill during the window
(dataflow-engagement defect) -- both are real EMU defects. This experiment
distinguishes them.

## The instrument: dummy-write32 sweep

Inject N benign `npu_write32` ops (to a ProdCore DM scratch word, `0x78000`, that
nothing reads) into the runtime sequence *before* the of_out drain dispatch,
lengthening the pre-drain window by a known amount. Cores are CDO-enabled and
free-run independent of runtime-sequence length, so a longer pre-drain window =
more free-run time before the drain engages. Sweep N in {0, 8, 64, 256} and read
the shim S2MM STREAM_STARVATION offset (gross empty/full readout, robust to the
trace perturbation because the perturbation is constant across N):

- **Stays flat** => extending the window does NOT change fill => the divergence is
  intrinsic to dataflow engagement (Horn A).
- **Grows with N** => HW fills given time => the divergence is window-length
  (Horn B, control-path composition).

Build faithfulness: N=0's insts.bin is BYTE-IDENTICAL to the surviving baseline
build (sha `32a20a07...`), so the harness reproduces the original exactly.

## Result (dispositive): Horn A

| N | HW starv offset (first 6) | EMU starv offset | HW drain-START base | EMU base |
|---|---|---|---|---|
| 0 | `[13,69,133,197,261,325]` | `[1683,1683,1747,...]` | 423094 | 6779 |
| 8 | `[13,69,133,197,261,325]` | `[1683,1683,1747,...]` | 429893 | 6779 |
| 64 | `[69,133,197,261,325,389]` | `[1683,1683,1747,...]` | 434139 | 11835 |
| 256 | `[13,69,133,197,261,325]` | `[1683,1683,1747,...]` | 440483 | 33339 |

- **HW pipeline is EMPTY at drain-start for every window length**, including the
  +28000cy (N=256) extension. The starvation cadence is byte-identical across all
  N: `[13,69,133,197,...]` every 64cy. The N=64 "first=69" is the same cadence
  with a one-slot phase slip (the 13-slot not recorded first), not creeping fill
  -- every value is <<1683.
- **EMU pipeline is deadlock-FULL for every window length** (offset pinned at
  1683), saturated already at N=0.
- **The manipulation provably worked:** the drain START_TASK base shifts later
  with N in both worlds (EMU deterministically 6779->11835->33339; HW
  monotonically 423094->440483), and insts.bin grew +24 bytes/write32. The
  pre-drain window genuinely extended -- and neither world's fill-state moved.

**=> Horn B is falsified.** HW's emptiness is NOT because its window is too short;
extending it 5x leaves it empty. The fill-state divergence is intrinsic to how
each world engages inter-tile dataflow during the pre-drain window. **This is a
real EMU dataflow-engagement defect, not a trace-measurement artifact.**

Benignness: out.bin byte-identical across all 8 runs; on the deterministic EMU,
N=0 vs N=8 event structure AND counts identical (1257=1257) -- write32s change
only absolute timing. (HW event-count jitter 1225-vs-1217 is ordinary HW trace
nondeterminism; the (row,name) key set is identical, no new event types.)

## The mechanism, coherent end-to-end

- **EMU** engages all cores + tile DMAs from cy=0 (CDO config) and fills the chain
  to deadlock during the pre-drain window: of_out cannot drain (shim S2MM not
  started until the drain dispatch), so backpressure fills every stage ~2-deep.
  At drain-start the drain chews the backlog for 1683cy; the cores wake by
  *backward* drain-cascade (memtile -> consA -> prod), so the producer stalls
  LAST -> **-52**.
- **HW** keeps the pipeline EMPTY until the drain engages -- the producer's data
  does not propagate downstream during the pre-drain window. Cores engage from
  empty, the producer fills and stalls FIRST -> **prod-first, +2**; both cores'
  first LOCK_STALLs land ~together at +43/+45 past the drain base.

So the EMU is over-eager: it ties inter-tile dataflow engagement to CDO time
(cy=0); HW ties it to ~the runtime-sequence/drain. This reconciles session 2
("real cold-start over-fill, localized to the drain path") with the falsification
of session 3's "trace artifact" conclusion: the over-fill is real, and the
control-path components being individually faithful does NOT make the composition
faithful -- the EMU fills the pipeline during a window HW keeps empty.

## Fix locus (OPEN) and the remaining unknown

The defect is the EMU's dataflow-engagement gating, NOT the control-path constants
(session 3 validated those) and NOT the trace unit. The EMU should not let the
inter-tile chain fill to deadlock during the pre-drain control window when HW
keeps it empty.

### The fill has TWO sources -- a warm-up-only fix is falsified

The EMU advances inter-tile dataflow during the ENTIRE pre-drain window, from two
distinct sites, BOTH of which must be addressed:
- **Warm-up** (`crates/xdna-emu-ffi/src/backend.rs:240-250`): steps the whole
  engine until `all_cores_blocked` before processing any runtime instruction.
- **Main-loop pre-drain stepping** (`backend.rs:308` `engine.step()`, called every
  iteration alongside `executor.try_advance()` at `:271`): while the executor
  grinds one-instruction-at-a-time through the ~6672cy of trace-setup write32s
  before reaching the of_out `dma_memcpy_nd` dispatch, `engine.step()` advances
  the array every cycle -- refilling the pipeline.

This is why the prior `XDNA_EMU_WARMUP_CAP=0` experiment (2026-06-29) was
BYTE-IDENTICAL to default: removing the warm-up leaves `backend.rs:308` to refill
during the main-loop pre-drain window. So the `emu-engagement-map` agent's
"root cause = warm-up" is INCOMPLETE -- gating only the warm-up cannot work
(already empirically falsified). The fix must keep the pipeline empty across
both windows.

### The mechanism is A2, on NON-confounded evidence

WHY HW's pipeline stays empty despite the toolchain enabling cores + DMA channels
at CDO time (`mlir-aie AIERT.cpp` addCoreEnable + pushToBdQueueAndEnable):
- **A1:** cores/DMAs free-run from CDO but data does not propagate pre-drain.
- **A2:** cores/DMAs effectively do not engage their dataflow loops until ~the
  runtime-sequence/drain, so nothing fills.

Two POST-baseline trace observables (recorded by the trace, so NOT confounded by
the pre-baseline recording gap) both point to A2:
- **of_out is empty at drain-start** (HW starves t+13, immediately) -- nothing
  reached of_out before the drain; the producer did not fill it pre-drain.
- **the producer's first LOCK_STALL is at +43, AFTER the drain base** -- had the
  producer run its produce loop pre-drain, backpressure (drain not started) would
  have stalled it EARLIER (before the base). It stalled after => it was not
  filling pre-drain.
So on HW the cores do not engage their dataflow loops until ~the drain; the EMU
engages them at CDO/warm-up time and fills. This is stronger than the sweep's
gross result and is not confounded.

The residual for the FIX (not for establishing the defect -- the sweep settles
that) is the CALIBRATION TARGET: how HW ties core dataflow-engagement to the
runtime sequence, and at what offset. The trace gives a soft +43; the clean,
trace-free measurement is the in-kernel-timer oracle (Path A), gated on a cheap
`cntr`/`Timer_Control` hardware-identity check.

## Why this matters beyond SP-4a

This engagement gap is the largest single Delta_wall error in the cross-column
grounding model and the blocker on vertical-anisotropy resolution (the R1 d_v
split is swamped by exactly this cold-start Delta_wall infidelity; see
`2026-07-02-sp5c-phase3-anisotropy-blocked-on-fill-gap.md`). Closing it is the
primary cycle-accuracy campaign post-SP-5c.
