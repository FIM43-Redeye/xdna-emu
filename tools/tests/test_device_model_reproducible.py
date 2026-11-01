"""The checked-in device model must be exactly what its generator emits.

`tools/aie-device-models.json` is a toolchain-derived artifact: Rust
(xdna-archspec) and the topology harness read it as the array's facts. It
drifted once without anyone noticing (the generator silently skipped xcve2802,
and fields added to the generator never reached the file), so this test
regenerates it and compares. Skipped when the mlir-aie bindings are not
importable; when mlir-aie itself moves, a failure here means: regenerate,
review the diff, commit.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BRIDGE = REPO / "tools" / "mlir-aie-bridge.py"
MODELS = REPO / "tools" / "aie-device-models.json"


def test_committed_device_model_matches_generator():
    run = subprocess.run(
        [sys.executable, str(BRIDGE), "device-model"],
        capture_output=True, text=True, timeout=120,
    )
    if run.returncode != 0 and "could not import mlir-aie" in run.stderr:
        pytest.skip("mlir-aie python bindings not importable")
    assert run.returncode == 0, run.stderr
    # Any per-device failure is only a warning on stderr; treat it as drift.
    assert "Warning" not in run.stderr, run.stderr

    generated = json.loads(run.stdout)
    committed = json.loads(MODELS.read_text())
    assert generated == committed, (
        "tools/aie-device-models.json is stale: regenerate with "
        "`python3 tools/mlir-aie-bridge.py device-model > "
        "tools/aie-device-models.json` and review the diff"
    )
