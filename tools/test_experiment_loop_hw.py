# tools/test_experiment_loop_hw.py
"""Phoenix-gated: the active loop closes on real NPU1 and converges.

    cd tools && XDNA_HW_SMOKE=1 env -u XDNA_EMU \\
      python -m pytest test_experiment_loop_hw.py -v -k add_one

Requires a built kernel under mlir-aie/build/test/npu-xrt/<test>/chess/.
"""
import os
from pathlib import Path
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("XDNA_HW_SMOKE") != "1",
    reason="HW loop test requires a real NPU; set XDNA_HW_SMOKE=1")

_FIX = (Path(__file__).resolve().parent
        / "config_extract" / "fixtures" / "add_one_using_dma.config.json")


def _cfg(tmp_path):
    from inference.run_experiment import KernelConfig
    return KernelConfig(test="add_one_using_dma", compiler="chess",
                        dump_path=str(_FIX), start_col=1,
                        anchor_tile_abs="1|2|0", anchor_event="PERF_CNT_2",
                        traced_col=1, n_runs=6,
                        out_root=str(tmp_path / "add_one"))


def test_loop_converges_on_add_one_hw(tmp_path):
    from inference.run_experiment import run_experiment
    rep = run_experiment(_cfg(tmp_path))
    assert rep["terminal_state"] in ("placed", "halted_falsifiable")
    # Every recorded constraint is falsifiable (carries its provenance batch).
    assert all(c["provenance_batch"] for c in rep["constraints"])


def test_loop_places_a_through_core_event_hw(tmp_path):
    # The through-core (program_path) pair S2MM_0_START <- MM2S_0_START is
    # orientable ONLY via the core_lock_relay edge -- config alone cannot place
    # it. Its presence in derives proves the loop used HW timing evidence.
    from inference.run_experiment import run_experiment
    rep = run_experiment(_cfg(tmp_path))
    derived_children = {d[0] for d in rep["derives"]}
    assert "1|0|2|DMA_S2MM_0_START_TASK" in derived_children, rep["derives"]


def test_forced_wrong_batch_changes_outcome_hw(tmp_path):
    # Falsifiability: if we corrupt the measured offset (shuffle one event's
    # timestamps across runs so std>>eps), the engine must REJECT that derive.
    # We run normally, then re-run the engine on a perturbed copy of the run
    # dirs and assert the perturbed pair is no longer derived.
    import json, shutil
    from inference.run_experiment import run_experiment, KernelConfig
    from inference.engine import run_engine

    cfg = _cfg(tmp_path)
    rep = run_experiment(cfg)
    base_children = {d[0] for d in rep["derives"]}
    assert base_children, "nothing derived; cannot test falsifiability"

    # Perturb: copy run dirs, scramble PORT_RUNNING_4's ts by a DIFFERENT amount
    # per run so the PR4<-PR0 offset becomes unstable across runs (a uniform
    # shift would leave the cross-run std unchanged and stay correlated -- it is
    # the variance, not the absolute offset, that the verifier rejects on).
    target = "PORT_RUNNING_4"
    pert = Path(cfg.out_root) / "perturbed"
    run_dirs = []
    for idx, rd in enumerate(sorted(p for p in Path(cfg.out_root).glob("capture_*/run_*"))):
        dst = pert / rd.relative_to(cfg.out_root)
        shutil.copytree(rd, dst)
        bump = idx * 9999      # per-run-varying -> offset std explodes
        for ev_path in dst.glob("batch_*/hw/trace.events.json"):
            doc = json.loads(ev_path.read_text())
            for e in doc["events"]:
                if e["name"] == target:
                    e["ts"] += bump; e["soc"] += bump
            ev_path.write_text(json.dumps(doc))
        run_dirs.append(str(dst))

    led = Path(cfg.out_root) / "ledger.json"   # written by run_experiment
    perturbed = run_engine(run_dirs, str(led),
                           [("1|1|3|PORT_RUNNING_4", "1|1|3|PORT_RUNNING_0")])
    perturbed_children = {d[0] for d in perturbed["derives"]}
    assert "1|1|3|PORT_RUNNING_4" not in perturbed_children
