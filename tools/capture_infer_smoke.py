#!/usr/bin/env python3
"""Drive trace_capture.run_loop for the inference-engine Axis-2 HW smoke (#140).

trace_capture.py has no CLI; this is the 10-line driver the plan calls for. It
captures add_one_using_dma's seed active plan across N runs on the real NPU1
(chess = ground truth), writing run_NN/batch_MM/hw/trace.events.json under the
output dir, which the inference engine then reads offline.

Run under `env -u XDNA_EMU -u XDNA_EMU_RUNTIME` so the bridge targets real HW.

    python tools/capture_infer_smoke.py [out_dir] [n_runs]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trace_capture as tc

out = sys.argv[1] if len(sys.argv) > 1 else "build/experiments/infer-smoke"
n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 6

print(f"[infer-smoke] capturing add_one_using_dma -> {out} ({n_runs} runs, chess)")
summary = tc.run_loop("add_one_using_dma", tc.SEED_ACTIVE_PLAN, n_runs=n_runs, out=out)
print(f"[infer-smoke] DONE. summary={summary}")
