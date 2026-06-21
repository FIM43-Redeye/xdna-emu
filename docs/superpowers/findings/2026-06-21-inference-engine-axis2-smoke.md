# Trace Inference Engine — Axis-2 HW Smoke (#140, Plan 1 Task 14)

**Date:** 2026-06-21
**Kernel:** `add_one_using_dma` (chess), real NPU1 (Phoenix), 6 runs.
**Result:** PASS — on real silicon, the sound inference engine independently
re-derives the capture-engine validation's 5 stochastic roots and the
deterministic backbone, with provenance grounded and replication clean.

## What ran

1. **Capture (HW).** `tools/capture_infer_smoke.py` drove
   `trace_capture.run_loop('add_one_using_dma', SEED_ACTIVE_PLAN, n_runs=6)` on
   the real NPU1 under `env -u XDNA_EMU -u XDNA_EMU_RUNTIME` (chess = ground
   truth), writing `run_NN/batch_00/hw/trace.events.json` under
   `build/experiments/infer-smoke/`. 6/6 runs clean, 11 slots fired each.
2. **Infer (offline).** `inference.engine.run_engine` read the captured run dirs
   + a hand-authored structural ledger
   (`tools/inference/fixtures/add_one_using_dma.ledger.json`) + five
   config-oriented candidate `(child, parent)` pairs, and produced the placement
   report.

## Outcome — the engine reproduces the validation, soundly

Of the 11 fired events, **10 are stochastic** (std across runs ~1000+ cycles)
and only the anchor `1|2|0|PERF_CNT_2` is deterministic (std 0). The sound
engine reduced the 10 to **exactly the validation's 5 stochastic roots** by
deriving the other 5:

| Re-derived (5) | placed via `derives` at offset | from root |
|---|---|---|
| `1\|0\|2\|DMA_S2MM_0_START_TASK` | 934 | `DMA_MM2S_0_START_TASK` |
| `1\|0\|2\|DMA_S2MM_0_STREAM_STARVATION` | 939 | `DMA_MM2S_0_START_TASK` |
| `1\|1\|3\|PORT_RUNNING_1` | 109 | `PORT_RUNNING_0` |
| `1\|1\|3\|PORT_RUNNING_4` | 30 | `PORT_RUNNING_0` |
| `1\|1\|3\|PORT_RUNNING_5` | 197 | `PORT_RUNNING_0` |

**5 stochastic roots** (the DMA-delivery degrees of freedom, identical to the
2026-06-17 capture-engine validation): `DMA_MM2S_0_FINISHED_TASK`,
`DMA_MM2S_0_START_TASK`, `DMA_S2MM_0_FINISHED_TASK`, `PORT_RUNNING_0`,
`LOCK_STALL`.

Gates, all green on silicon:
- **`provenance_ok: true`** — the keystone holds end-to-end: every `derives`
  bottoms out in measured `fired` leaves + a ledgered `config_path` structural
  leaf. No hanging nodes.
- **`replication_violations: []`** — measured leaves replicate across batches.
- Each `derives` was admitted **only because config_path oriented it AND
  `correlates` independently confirmed the stable offset** on the measured HW
  data (the offsets above match the reference greedy graph to the cycle). The
  orientation (config) and the co-variation (measurement) are independent
  checks; their agreement on real silicon is the validation.

## Placement-not-causation, demonstrated on hardware

`DMA_S2MM_0_STREAM_STARVATION` is classified **`derived`** (placed at offset 939
from the upstream `DMA_MM2S_0_START_TASK`). Starvation fires from *downstream*
backpressure — its causal arrow runs opposite to the dataflow arrow — yet the
engine **places** it correctly and **never labels it causal** (the rule name is
`derives_rule_placement`). This is the spec's exact backpressure example,
confirmed on silicon.

## Honest limitations of this smoke

- **The structural ledger is hand-authored, not auto-extracted.** The five
  `config_path` cites encode `add_one_using_dma`'s known dataflow (shim input
  DMA upstream of output DMA + its starvation; memtile lead port upstream of the
  trailing co-firing ports). This proves the engine *consumes* a structural
  ledger and reconstructs placement soundly — **not** that the ledger was
  independently derived. Automated extraction from the binary's routes/BDs/locks
  is **Plan 2** (a Rust `examples/dump_config_json.rs`). The engine's
  `correlates` admission gate means a mis-asserted orientation still cannot
  manufacture a derive without a real measured stable offset.
- **No `same_source` instance on this kernel.** `degeneracy: []` — the engine
  found no coincident root pairs, so the structural-degeneracy gate (emit
  falsifiable non-separation prediction → confirm) had **no candidate** to
  exercise here. The 5 roots are causally-independent distinct physical events;
  `add_one_using_dma`'s single-column trace aliases no one physical event across
  two trace units. The gate machinery is unit-tested (Axis-1, Task 10); a HW
  exercise of it is deferred to a kernel that genuinely aliases an event across
  trace units (e.g. a shim event re-observed at a memtile via a shared route).

## Reproduce

```bash
# capture (HW)
cd /home/triple/npu-work/xdna-emu
env -u XDNA_EMU -u XDNA_EMU_RUNTIME \
  python tools/capture_infer_smoke.py build/experiments/infer-smoke 6
# infer + assert (offline)
cd tools && XDNA_HW_SMOKE=1 XDNA_SMOKE_RUNS=../build/experiments/infer-smoke \
  python -m pytest test_inference_hw_smoke.py -v
```

The captured `engine-report.json` is saved under `build/experiments/infer-smoke/`.
