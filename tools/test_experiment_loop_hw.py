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
    assert rep["engine_ok"] is True
    assert rep["terminal_state"] in ("placed", "halted_falsifiable")
    # Every recorded constraint is falsifiable (carries its provenance batch).
    assert all(c["provenance_batch"] for c in rep["constraints"])


def test_through_core_event_is_placed_as_gap_hw(tmp_path):
    # The through-core (program_path) pair S2MM_0_START <- MM2S_0_START is
    # orientable ONLY via the core_lock_relay edge. It spans shim -> core -> shim
    # (cross timer-domain), so under explicit grounding it is a NAMED GAP:
    # existence + orientation, deterministically derivable every run -- NO retry.
    from inference.run_experiment import run_experiment
    target = "1|0|2|DMA_S2MM_0_START_TASK"
    rep = run_experiment(_cfg(tmp_path))
    assert rep["engine_ok"] is True
    children = {d[0] for d in rep["derives"]}
    assert target in children, f"through-core {target} not placed; derives={rep['derives']}"
    gap_children = {g[0] for g in rep["gaps"]}
    assert target in gap_children, f"{target} should be a gap (cross-domain), not a segment"


def test_core_lock_segment_grounds_exact_hw(tmp_path):
    # The core compute segment INSTR_LOCK_ACQUIRE_REQ -> INSTR_LOCK_RELEASE_REQ
    # is within ONE timer domain (core module) and exact across the run set ->
    # a SEGMENT with a cycle-accurate offset. This is the cycle-exact deliverable
    # the instruction-event layer made producible.
    from inference.run_experiment import run_experiment
    child = "1|2|0|INSTR_LOCK_RELEASE_REQ"
    parent = "1|2|0|INSTR_LOCK_ACQUIRE_REQ"
    rep = run_experiment(_cfg(tmp_path))
    seg = next((s for s in rep["segments"] if s[0] == child and s[1] == parent), None)
    assert seg is not None, f"core lock segment not grounded; segments={rep['segments']}"
    # Exact, positive offset (release after acquire). Value is kernel-specific
    # (~22 on add_one); assert it is a concrete int that agreed across the runs.
    assert isinstance(seg[2], int) and seg[2] > 0


def test_perturbed_segment_downgrades_to_gap_hw(tmp_path):
    # Falsifiability: corrupt the EXACT core-lock segment's offset per-run so the
    # cross-run range != 0. The engine must DOWNGRADE it from a segment to a gap
    # (it can no longer claim a cycle-exact offset) -- existence/orientation
    # survive, the cycle count does not.
    import json, shutil
    from pathlib import Path
    from inference.run_experiment import run_experiment, KernelConfig
    from inference.engine import run_engine
    from inference.selfmodel import (enumerate_configured_events,
                                     candidate_pairs_from_dump)
    from config_extract.dump_model import load_dump

    cfg = _cfg(tmp_path)
    rep = run_experiment(cfg)
    child, parent = "1|2|0|INSTR_LOCK_RELEASE_REQ", "1|2|0|INSTR_LOCK_ACQUIRE_REQ"
    assert any(s[0] == child and s[1] == parent for s in rep["segments"]), \
        "baseline must ground the core-lock segment to perturb it"

    # Perturb RELEASE's ts by a per-run-varying amount so the offset range != 0.
    pert = Path(cfg.out_root) / "perturbed"
    run_dirs = []
    for idx, rd in enumerate(sorted(p for p in Path(cfg.out_root).glob("capture_*/run_*"))):
        dst = pert / rd.relative_to(cfg.out_root)
        shutil.copytree(rd, dst)
        bump = (idx + 1) * 7  # per-run-varying -> offset range explodes
        for ev_path in dst.glob("batch_*/hw/trace.events.json"):
            doc = json.loads(ev_path.read_text())
            for e in doc["events"]:
                if e["name"] == "INSTR_LOCK_RELEASE_REQ" and e["col"] == 1 and e["row"] == 2:
                    e["ts"] += bump; e["soc"] += bump
            ev_path.write_text(json.dumps(doc))
        run_dirs.append(str(dst))

    dump = load_dump(cfg.dump_path)
    configured = enumerate_configured_events(dump, cfg.start_col)
    pairs = candidate_pairs_from_dump(dump, configured, cfg.start_col)
    led = Path(cfg.out_root) / "ledger.json"   # written by run_experiment
    perturbed = run_engine(run_dirs, str(led), pairs)
    seg_children = {s[0] for s in perturbed["segments"]}
    gap_children = {g[0] for g in perturbed["gaps"]}
    assert child not in seg_children, "perturbed segment must no longer be exact"
    assert child in gap_children, "perturbed edge must survive as a gap (placed)"


# ---------------------------------------------------------------------------
# Suite convergence: add_one_objFifo + vector_scalar_using_dma
#
# Both kernels share the same single-column layout as add_one_using_dma:
#   shim (0,0) -> memtile (0,1) -> core (0,2), placed at absolute col 1.
# A defined terminal state is "placed" (full placement) or
# "halted_falsifiable" (honest halt, all constraints provenance-complete).
# ---------------------------------------------------------------------------

_SUITE = {
    "add_one_objFifo":         dict(start_col=1, anchor_tile_abs="1|2|0", traced_col=1),
    "vector_scalar_using_dma": dict(start_col=1, anchor_tile_abs="1|2|0", traced_col=1),
}


@pytest.mark.parametrize("kernel", sorted(_SUITE))
def test_suite_reaches_terminal_state_hw(kernel, tmp_path):
    from inference.run_experiment import KernelConfig, run_experiment
    p = _SUITE[kernel]
    fix = (Path(__file__).resolve().parent / "config_extract" / "fixtures"
           / f"{kernel}.config.json")
    cfg = KernelConfig(test=kernel, compiler="chess", dump_path=str(fix),
                       start_col=p["start_col"],
                       anchor_tile_abs=p["anchor_tile_abs"],
                       anchor_event="PERF_CNT_2", traced_col=p["traced_col"],
                       n_runs=6, out_root=str(tmp_path / kernel))
    rep = run_experiment(cfg)
    assert rep["engine_ok"] is True
    # A defined terminal state (placed, or an honest falsifiable halt) -- never
    # the unexplained-halt bug signal.
    assert rep["terminal_state"] in ("placed", "halted_falsifiable"), rep
    # Report is provenance-complete: every constraint cites the batch that set it.
    assert all(c["provenance_batch"] for c in rep["constraints"])
