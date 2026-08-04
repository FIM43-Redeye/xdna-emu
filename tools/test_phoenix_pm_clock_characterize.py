import importlib.util
import json
import struct
from pathlib import Path

import pytest


_TOOL = Path(__file__).parent / "phoenix-pm-clock-characterize.py"
_SPEC = importlib.util.spec_from_file_location("phoenix_pm_clock", _TOOL)
pm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pm)


def write32(address, value):
    return struct.pack("<IIQII", 0, 0, address, value, 24)


def address(col, row, offset):
    return (col << 25) | (row << 20) | offset


def fixture_insts():
    records = [
        write32(address(0, 2, 0x340D0), 0x797A0000),
        write32(address(0, 2, 0x340E0), 0),
        write32(address(0, 2, 0x340E4), 0),
        write32(address(0, 2, 0x31504), 28),
        write32(address(0, 2, 0x31508), 7 << 16),
        struct.pack("<IIII", 0x80, 16, 0x100, 0x10100),
    ]
    payload = b"".join(records)
    return struct.pack("<IIII", 0x06030100, 0, len(records), 16 + len(payload)) + payload


def register_db(tmp_path):
    path = tmp_path / "registers.json"
    path.write_text(json.dumps({
        "modules": {"core": {"registers": [
            {
                "name": "Performance_Control1",
                "offset": "0x31504",
                "bit_fields": [
                    {"name": "Cnt3_Stop_Event", "bit_range": [24, 30]},
                    {"name": "Cnt3_Start_Event", "bit_range": [16, 22]},
                    {"name": "Cnt2_Stop_Event", "bit_range": [8, 14]},
                    {"name": "Cnt2_Start_Event", "bit_range": [0, 6]},
                ],
            },
            {
                "name": "Performance_Control2",
                "offset": "0x31508",
                "bit_fields": [
                    {"name": "Cnt3_Reset_Event", "bit_range": [24, 30]},
                    {"name": "Cnt2_Reset_Event", "bit_range": [16, 22]},
                ],
            },
            {
                "name": "Performance_Counter3_Event_Value",
                "offset": "0x3158C",
                "bit_fields": [
                    {"name": "Counter_Event_Value", "bit_range": [0, 31]},
                ],
            },
        ]}},
    }))
    return path


EVENT_IDS = {
    "PERF_CNT_2": 7,
    "PERF_CNT_3": 8,
    "INSTR_EVENT_0": 33,
    "INSTR_EVENT_1": 34,
    "PM_ADDRESS_OUT_OF_RANGE": 65,
    "INSTR_LOCK_RELEASE_REQ": 44,
    "INSTR_LOCK_ACQUIRE_REQ": 45,
    "LOCK_STALL": 26,
}


def test_instrument_comparator_preserves_counter2(tmp_path):
    patched = pm.instrument_comparator(
        fixture_insts(), threshold=64,
        register_db=register_db(tmp_path), event_ids=EVENT_IDS,
    )
    writes = {addr & 0xFFFFF: value for _, addr, value in pm.patcher._walk_write32(patched)}

    assert writes[0x340D0] == 0x007A0000
    assert writes[0x340E0] == 0x41222107
    assert writes[0x340E4] == 0x1A2D2C08
    assert writes[0x31504] == 28 | (65 << 16)
    assert writes[0x31508] == 7 << 16
    assert writes[0x3158C] == 64
    assert struct.unpack_from("<I", patched, 8)[0] == 7
    threshold_offset = next(
        offset for offset, addr, _ in pm.patcher._walk_write32(patched)
        if addr & 0xFFFFF == 0x3158C
    )
    offset = 16
    while patched[offset] != 0x80:
        offset += pm.patcher._instruction_length(patched, offset)
    assert threshold_offset < offset


def event(name, ts, pkt_type=0):
    return {"name": name, "ts": ts, "col": 1, "row": 2,
            "pkt_type": pkt_type}


def valid_prefix():
    return [
        event("PERF_CNT_2", 10),
        event("DMA_MM2S_0_FINISHED_BD", 15, pkt_type=1),
        event("INSTR_LOCK_RELEASE_REQ", 20),
    ]


def test_capture_classifier_distinguishes_fire_and_gate():
    prefix = valid_prefix() + [event("PM_ADDRESS_OUT_OF_RANGE", 30)]
    assert pm.classify_capture(
        prefix + [event("PERF_CNT_3", 31)], b"ok", b"ok",
    ) == "fires"
    assert pm.classify_capture(
        prefix + [event("PERF_CNT_2", 40)], b"ok", b"ok",
    ) == "gates_first"


def test_capture_classifier_rejects_ambiguous_absence():
    prefix = valid_prefix() + [event("PM_ADDRESS_OUT_OF_RANGE", 30)]
    assert pm.classify_capture(prefix, b"ok", b"ok") == "invalid"
    assert pm.classify_capture(
        valid_prefix() + [event("PERF_CNT_3", 29), event("PM_ADDRESS_OUT_OF_RANGE", 30)],
        b"ok", b"ok",
    ) == "invalid"
    assert pm.classify_capture(
        prefix + [event("PERF_CNT_3", 31)], b"wrong", b"ok",
    ) == "invalid"


def test_probe_classifier_requires_requested_stable_clock():
    events = valid_prefix() + [
        event("PM_ADDRESS_OUT_OF_RANGE", 30),
        event("PERF_CNT_3", 31),
    ]
    stable = {
        "power_mode": "balanced",
        "power_mode_id": 2,
        "mp_npu_mhz": 600,
        "h_mhz": 1028,
    }

    assert pm.classify_probe(
        events, b"ok", b"ok", stable, stable, "balanced",
    ) == "fires"
    assert pm.classify_probe(
        events, b"ok", b"ok", stable,
        {**stable, "h_mhz": 1024}, "balanced",
    ) == "invalid"
    assert pm.classify_probe(
        events, b"ok", b"ok", stable, stable, "performance",
    ) == "invalid"
    assert pm.classify_probe(
        events, b"ok", b"ok", stable, stable, "balanced",
        expected_clock={**stable, "h_mhz": 800},
    ) == "invalid"


def test_control_classifier_requires_clean_prefix_without_comparator():
    assert pm.classify_control(valid_prefix(), b"ok", b"ok") == "control"
    assert pm.classify_control(
        valid_prefix() + [event("PM_ADDRESS_OUT_OF_RANGE", 30)], b"ok", b"ok",
    ) == "invalid"
    assert pm.classify_control(
        valid_prefix() + [event("PERF_CNT_3", 30)], b"ok", b"ok",
    ) == "invalid"


def test_relabels_only_core_comparator_slot():
    document = {
        "events": [
            {**event("GROUP_ERRORS_0", 30), "slot": 4},
            {**event("DMA_MM2S_0_STALLED_LOCK", 30, pkt_type=1), "slot": 4},
        ],
        "slot_names": {
            "core": ["a", "b", "c", "d", "GROUP_ERRORS_0", "f", "g", "h"],
            "mem": ["a", "b", "c", "d", "DMA_MM2S_0_STALLED_LOCK"],
        },
    }

    pm.relabel_comparator_events(document)

    assert document["events"][0]["name"] == "PERF_CNT_3"
    assert document["events"][1]["name"] == "DMA_MM2S_0_STALLED_LOCK"
    assert document["slot_names"]["core"][4] == "PERF_CNT_3"


def test_search_boundary_returns_adjacent_thresholds():
    assert pm.search_boundary(
        lambda threshold: "fires" if threshold <= 333 else "gates_first",
        initial=64,
    ) == (333, 334)


def test_search_boundary_stops_on_invalid_probe():
    with pytest.raises(RuntimeError, match="invalid threshold 128"):
        pm.search_boundary(
            lambda threshold: "fires" if threshold == 64 else "invalid",
            initial=64,
        )


def qos_candidate(label, before, after=None, *, ok=True, output_ok=True):
    return {
        "label": label,
        "qos": {"gops": 1, "fps": 1000},
        "runner_ok": ok,
        "output_ok": output_ok,
        "clock_before": before,
        "clock_after": before if after is None else after,
    }


def clock(mp_npu, h, mode="default"):
    return {
        "power_mode": mode,
        "power_mode_id": 0,
        "mp_npu_mhz": mp_npu,
        "h_mhz": h,
    }


def test_select_clock_regimes_deduplicates_observed_pairs():
    records = [
        qos_candidate("low", clock(400, 800)),
        qos_candidate("medium", clock(600, 1028)),
        qos_candidate("medium-duplicate", clock(600, 1028)),
        qos_candidate("upper", clock(720, 1309)),
        qos_candidate("high", clock(847, 1600)),
    ]

    selected = pm.select_clock_regimes(records, expected_mode="default")

    assert [record["label"] for record in selected] == [
        "low", "medium", "upper", "high",
    ]


def test_select_clock_regimes_requires_three_distinct_ratios():
    records = [
        qos_candidate("low", clock(400, 800)),
        qos_candidate("medium", clock(600, 1028)),
        qos_candidate("collapsed-high", clock(600, 1028)),
    ]

    with pytest.raises(RuntimeError, match="only 2 distinct clock ratios"):
        pm.select_clock_regimes(records, expected_mode="default")


@pytest.mark.parametrize("record", [
    qos_candidate("failed", clock(400, 800), ok=False),
    qos_candidate("bad-output", clock(400, 800), output_ok=False),
    qos_candidate("unstable", clock(400, 800), after=clock(600, 1028)),
    qos_candidate("changed-mode", clock(400, 800, mode="performance")),
])
def test_select_clock_regimes_rejects_invalid_candidate(record):
    with pytest.raises(RuntimeError, match="invalid QoS clock candidate"):
        pm.select_clock_regimes([record], expected_mode="default")


def test_probe_clock_regime_primes_context_before_measured_pair(
    tmp_path, monkeypatch,
):
    observed = iter([clock(400, 800), clock(400, 800)])
    monkeypatch.setattr(pm, "_query_clock", lambda _path: next(observed))

    class Session:
        def __init__(self):
            self.calls = 0

        def run_one(self, *, trace_out, outputs, **_kwargs):
            self.calls += 1
            trace_out.write_bytes(b"trace")
            outputs[0].write_bytes(b"expected")
            return {"ok": True, "run_idx": self.calls - 1}

    session = Session()
    template = tmp_path / "template.insts.bin"
    template.write_bytes(b"insts")

    record = pm.probe_clock_regime(
        session=session,
        template=template,
        clock_query=tmp_path / "clock-query",
        expected_output=b"expected",
        run_dir=tmp_path / "probe",
        label="fps-1000",
        qos={"gops": 1, "fps": 1000},
    )

    assert session.calls == 2
    assert record["clock_before"] == clock(400, 800)
    assert record["clock_after"] == clock(400, 800)
    assert record["runner_ok"] is True
    assert record["output_ok"] is True
