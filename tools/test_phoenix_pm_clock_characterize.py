import importlib.util
import hashlib
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


def witness_fixture_insts(occupied_module=None):
    records = [
        write32(address(0, 2, 0x340D0), 0x797A0000),
        write32(address(0, 2, 0x340E0), 0),
        write32(address(0, 2, 0x340E4), 0),
        write32(address(0, 2, 0x31504), 28),
        write32(address(0, 2, 0x31508), 7 << 16),
        write32(address(0, 0, 0x340D0), 0x7E7F0000),
        write32(address(0, 0, 0x340E0), 0x0000160E),
        write32(address(0, 0, 0x3404C), 127),
        write32(address(0, 0, 0x34008), 127),
        write32(address(0, 0, 0x34048), 126),
        write32(address(0, 0, 0x34008), 126),
    ]
    if occupied_module in ("core", "core_conflict"):
        records.append(write32(address(0, 2, 0x34044), 8))
        if occupied_module == "core_conflict":
            records.append(write32(address(0, 2, 0x34048), 33))
    elif occupied_module == "memmod":
        records.append(write32(address(0, 2, 0x14044), 123))
    elif occupied_module == "memtile":
        records.append(write32(address(0, 1, 0x94044), 155))
    elif occupied_module == "shim":
        records.append(write32(address(0, 0, 0x34044), 123))
    records.append(struct.pack("<IIII", 0x80, 16, 0x100, 0x10100))
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
            {"name": "Event_Broadcast13", "offset": "0x34044"},
            {"name": "Event_Broadcast14", "offset": "0x34048"},
        ]}, "shim": {"registers": [
            {"name": "Event_Generate", "offset": "0x34008"},
            {"name": "Trace_Event0", "offset": "0x340E0"},
            {"name": "Event_Broadcast13_A", "offset": "0x34044"},
            {
                "name": "Column_Clock_Control",
                "offset": "0xFFF20",
                "bit_fields": [
                    {"name": "Clock_Buffer_Enable", "bit_range": [0, 0]},
                ],
            },
            {
                "name": "Performance_Ctrl0",
                "offset": "0x31000",
                "bit_fields": [
                    {"name": "Cnt0_Start_Event", "bit_range": [0, 6]},
                    {"name": "Cnt0_Stop_Event", "bit_range": [8, 14]},
                ],
            },
            {
                "name": "Performance_Ctrl1",
                "offset": "0x31008",
                "bit_fields": [
                    {"name": "Cnt0_Reset_Event", "bit_range": [0, 6]},
                ],
            },
            {
                "name": "Performance_Counter0_Event_Value",
                "offset": "0x31080",
                "bit_fields": [
                    {"name": "Counter_Event_Value", "bit_range": [0, 31]},
                ],
            },
        ]}, "memory": {"registers": [
            {"name": "Event_Broadcast13", "offset": "0x14044"},
        ]}, "memory_tile": {"registers": [
            {"name": "Event_Broadcast13", "offset": "0x94044"},
        ]}},
    }))
    return path


def aieml_npi_source(tmp_path):
    path = tmp_path / "xaie_npi_aieml.c"
    path.write_text("""
#define XAIEML_NPI_PCSR_UNLOCK_CODE 0xF9E8D7C6U
#define XAIEML_NPI_PCSR_LOCK 0X0000000CU
#define XAIEML_NPI_PROT_REG_CNTR 0x00000200U
#define XAIEML_NPI_PROT_REG_CNTR_EN_MSK 0x00000001U
#define XAIEML_NPI_PROT_REG_CNTR_EN_LSB 0U
#define XAIEML_NPI_PROT_REG_CNTR_FIRSTCOL_MSK 0x000000FEU
#define XAIEML_NPI_PROT_REG_CNTR_FIRSTCOL_LSB 1U
#define XAIEML_NPI_PROT_REG_CNTR_LASTCOL_MSK 0x00007F00U
#define XAIEML_NPI_PROT_REG_CNTR_LASTCOL_LSB 8U
""")
    return path


def aieml_events_source(tmp_path):
    path = tmp_path / "xaie_events_aieml.h"
    path.write_text("""
#define XAIEML_EVENTS_CORE_BROADCAST_14 121U
#define XAIEML_EVENTS_CORE_USER_EVENT_0 124U
#define XAIEML_EVENTS_CORE_USER_EVENT_1 125U
#define XAIEML_EVENTS_PL_BROADCAST_A_13 123U
#define XAIEML_EVENTS_PL_BROADCAST_A_14 124U
#define XAIEML_EVENTS_PL_USER_EVENT_0 126U
""")
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

SHIM_EVENT_IDS = {
    "NONE": 0,
    "PERF_CNT_0": 5,
    "DMA_S2MM_0_START_TASK": 14,
    "DMA_S2MM_0_FINISHED_TASK": 22,
    "BROADCAST_A_13": 123,
    "USER_EVENT_0": 126,
    "USER_EVENT_1": 127,
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


def test_post_tct_noops_preserve_the_trailing_writes():
    data = bytearray(witness_fixture_insts())
    tct_end = len(data)
    trailing = (
        write32(address(0, 0, 0x34008), 0)
        + write32(address(0, 0, 0x3404C), 0)
    )
    data.extend(trailing)
    struct.pack_into("<I", data, 8, struct.unpack_from("<I", data, 8)[0] + 2)
    struct.pack_into("<I", data, 12, len(data))

    patched = pm.instrument_post_tct_noops(bytes(data), 3)

    expected = data[:tct_end] + b"\x05\x00\x00\x00" * 3 + data[tct_end:]
    struct.pack_into("<I", expected, 8, struct.unpack_from("<I", data, 8)[0] + 3)
    struct.pack_into("<I", expected, 12, len(expected))
    assert patched == bytes(expected)


def firmware_timeline_fixture_insts():
    records = [
        write32(address(0, 0, 0x340D0), 0x7E7F0000),
        write32(address(0, 0, 0x340E0), 0x0000160E),
        write32(address(0, 0, 0x3404C), SHIM_EVENT_IDS["USER_EVENT_1"]),
        write32(address(0, 0, 0x34008), SHIM_EVENT_IDS["USER_EVENT_1"]),
        struct.pack("<IIII", 0x80, 16, 0x100, 0x10100),
        write32(address(0, 0, 0x34048), SHIM_EVENT_IDS["USER_EVENT_0"]),
        write32(address(0, 0, 0x34008), SHIM_EVENT_IDS["USER_EVENT_0"]),
    ]
    payload = b"".join(records)
    return struct.pack(
        "<IIII", 0x06030100, 0, len(records), 16 + len(payload),
    ) + payload


def test_firmware_clock_timeline_brackets_each_noop_block(tmp_path):
    blocks = (0, 1, 4, 1, 0)
    patched = pm.instrument_firmware_clock_timeline(
        firmware_timeline_fixture_insts(), register_db(tmp_path),
        SHIM_EVENT_IDS, blocks,
    )

    marker_address = address(0, 0, 0x34008)
    marker = write32(marker_address, SHIM_EVENT_IDS["USER_EVENT_0"])
    expected_records = marker + b"".join(
        b"\x05\x00\x00\x00" * count + marker for count in blocks
    )
    tct_end = pm.patcher._last_tct_boundary(patched)
    assert patched[tct_end:tct_end + len(expected_records)] == expected_records

    writes = list(pm.patcher._walk_write32(patched))
    marker_writes = [
        (offset, value) for offset, target, value in writes
        if target == marker_address and value == SHIM_EVENT_IDS["USER_EVENT_0"]
    ]
    flush_writes = [
        (offset, value) for offset, target, value in writes
        if target == marker_address and value == SHIM_EVENT_IDS["USER_EVENT_1"]
    ]
    assert len(marker_writes) == len(blocks) + 1
    assert len(flush_writes) == 2
    assert flush_writes[-1][0] > marker_writes[-1][0]
    assert struct.unpack_from("<I", patched, 8)[0] == 13 + sum(blocks)
    assert struct.unpack_from("<I", patched, 12)[0] == len(patched)

    trace_control = next(
        value for _, target, value in writes
        if target == address(0, 0, 0x340D0)
    )
    trace_events = next(
        value for _, target, value in writes
        if target == address(0, 0, 0x340E0)
    )
    assert trace_control == 0x007F0000
    assert trace_events == 0x7F7E160E


@pytest.mark.parametrize("blocks", [(), (1, -1), (True,), (0x40000000,)])
def test_firmware_clock_timeline_rejects_invalid_blocks(tmp_path, blocks):
    with pytest.raises(ValueError):
        pm.instrument_firmware_clock_timeline(
            firmware_timeline_fixture_insts(), register_db(tmp_path),
            SHIM_EVENT_IDS, blocks,
        )


def test_firmware_clock_timeline_requires_one_existing_start_and_stop(tmp_path):
    data = bytearray(firmware_timeline_fixture_insts())
    stop = write32(address(0, 0, 0x34008), SHIM_EVENT_IDS["USER_EVENT_0"])
    data.extend(stop)
    struct.pack_into("<I", data, 8, struct.unpack_from("<I", data, 8)[0] + 1)
    struct.pack_into("<I", data, 12, len(data))

    with pytest.raises(ValueError, match="exactly one standard stop trigger"):
        pm.instrument_firmware_clock_timeline(
            bytes(data), register_db(tmp_path), SHIM_EVENT_IDS, (0, 1),
        )


def test_classifies_exact_firmware_clock_timeline():
    blocks = (0, 1, 4, 1, 0)
    timestamps = (100, 111, 129, 182, 200, 211)
    events = [
        {"pkt_type": 2, "col": 1, "row": 0, "slot": 0,
         "name": "DMA_S2MM_0_START_TASK", "ts": 90},
        *[
            {"pkt_type": 2, "col": 1, "row": 0, "slot": 2,
             "name": "USER_EVENT_0", "ts": ts}
            for ts in timestamps
        ],
        {"pkt_type": 2, "col": 1, "row": 0, "slot": 1,
         "name": "DMA_S2MM_0_FINISHED_TASK", "ts": 220},
    ]
    clock = {
        "power_mode": "default", "power_mode_id": 0,
        "mp_npu_mhz": 600, "h_mhz": 1028,
    }

    result = pm.classify_firmware_clock_timeline(
        events, blocks, b"output", b"output", clock, clock,
    )

    assert result["qualified"] is True
    assert result["reason"] == "captured"
    assert result["intervals"] == [
        {"noop_count": 0, "array_cycles": 11},
        {"noop_count": 1, "array_cycles": 18},
        {"noop_count": 4, "array_cycles": 53},
        {"noop_count": 1, "array_cycles": 18},
        {"noop_count": 0, "array_cycles": 11},
    ]
    assert result["repeat_exact"] == {"0": True, "1": True}
    assert result["deterministic"] is True
    assert result["zero_marker_cycles"] == 11
    assert result["above_zero_cycles"] == {"1": 7, "4": 42}


def test_firmware_clock_timeline_preserves_nonexact_repeats_as_evidence():
    blocks = (0, 1, 0)
    events = [
        {"pkt_type": 2, "col": 1, "row": 0, "slot": slot,
         "name": name, "ts": ts}
        for slot, name, ts in (
            (0, "DMA_S2MM_0_START_TASK", 90),
            (2, "USER_EVENT_0", 100),
            (2, "USER_EVENT_0", 111),
            (2, "USER_EVENT_0", 129),
            (2, "USER_EVENT_0", 141),
            (1, "DMA_S2MM_0_FINISHED_TASK", 150),
        )
    ]
    clock = {"mp_npu_mhz": 400, "h_mhz": 800}

    result = pm.classify_firmware_clock_timeline(
        events, blocks, b"same", b"same", clock, clock,
    )

    assert result["qualified"] is True
    assert result["deterministic"] is False
    assert result["repeat_exact"] == {"0": False}
    assert "zero_marker_cycles" not in result
    assert "above_zero_cycles" not in result


def test_firmware_clock_timeline_rejects_missing_marker():
    blocks = (0, 1)
    events = [
        {"pkt_type": 2, "col": 1, "row": 0, "slot": 0,
         "name": "DMA_S2MM_0_START_TASK", "ts": 90},
        {"pkt_type": 2, "col": 1, "row": 0, "slot": 2,
         "name": "USER_EVENT_0", "ts": 100},
        {"pkt_type": 2, "col": 1, "row": 0, "slot": 2,
         "name": "USER_EVENT_0", "ts": 110},
        {"pkt_type": 2, "col": 1, "row": 0, "slot": 1,
         "name": "DMA_S2MM_0_FINISHED_TASK", "ts": 120},
    ]
    clock = {"mp_npu_mhz": 400, "h_mhz": 800}

    result = pm.classify_firmware_clock_timeline(
        events, blocks, b"same", b"same", clock, clock,
    )

    assert result == {
        "qualified": False,
        "reason": "marker_count_mismatch",
        "marker_count": 2,
        "expected_marker_count": 3,
    }


def test_relabels_firmware_clock_timeline_marker_slot():
    document = {
        "slot_names": {
            "shim": ["DMA_START", "DMA_FINISH", "NONE", "NONE"],
        },
        "events": [
            {"pkt_type": 2, "row": 0, "slot": 2, "name": "NONE"},
            {"pkt_type": 2, "row": 0, "slot": 3, "name": "NONE"},
            {"pkt_type": 2, "row": 1, "slot": 2, "name": "OTHER"},
            {"pkt_type": 0, "row": 0, "slot": 2, "name": "CORE"},
        ],
    }

    pm.relabel_firmware_clock_timeline_events(document)

    assert document["slot_names"]["shim"][2] == "USER_EVENT_0"
    assert document["slot_names"]["shim"][3] == "USER_EVENT_1"
    assert document["events"][0]["name"] == "USER_EVENT_0"
    assert document["events"][1]["name"] == "USER_EVENT_1"
    assert document["events"][2]["name"] == "OTHER"
    assert document["events"][3]["name"] == "CORE"


def test_prepares_real_gate_trace_as_producer_originated_shutdown_wave(tmp_path):
    db = register_db(tmp_path)
    data = pm.instrument_comparator(
        witness_fixture_insts(), threshold=64,
        register_db=db, event_ids=EVENT_IDS,
    )
    data = pm.instrument_shim_witness(
        data, register_db=db, core_event_ids=EVENT_IDS,
        shim_event_ids=SHIM_EVENT_IDS, threshold=64,
    )
    prepared = pm.prepare_real_column_gate_trace(
        data, db, aieml_events_source(tmp_path),
    )

    writes = list(pm.patcher._walk_write32(prepared))
    values = {
        ((target >> 20) & 0x1F, target & 0xFFFFF): value
        for _, target, value in writes
    }
    assert values[(2, 0x31504)] == (124 << 24) | (125 << 16) | 28
    assert values[(2, 0x340D0)] == 0x7C7A0000
    assert values[(2, 0x34044)] == 8
    assert values[(2, 0x34048)] == 124
    assert values[(0, 0x340D0)] == 0x7C7F0000
    assert values[(0, 0x34048)] == 126
    assert values[(0, 0x31000)] == 123
    assert [
        value for _, target, value in writes
        if target == address(0, 0, 0x34008)
    ] == [127]
    offset = 16
    noops = 0
    while offset < len(prepared):
        noops += struct.unpack_from("<I", prepared, offset)[0] == 5
        offset += pm.patcher._instruction_length(prepared, offset)
    assert noops == 1
    assert struct.unpack_from("<I", prepared, 8)[0] == struct.unpack_from("<I", data, 8)[0] + 1
    assert struct.unpack_from("<I", prepared, 12)[0] == len(prepared)


def test_real_gate_trace_rejects_existing_core_shutdown_source(tmp_path):
    data, _ = pm.patcher.patch_trace_control(
        witness_fixture_insts(occupied_module="core_conflict"),
        0, 2, "core", stop_event=0,
    )
    data, _ = pm.patcher.patch_trace_control(
        data, 0, 0, "shim", stop_event=0,
    )

    with pytest.raises(ValueError, match="core Event_Broadcast14 is already configured"):
        pm.prepare_real_column_gate_trace(
            data, register_db(tmp_path), aieml_events_source(tmp_path),
        )


def test_inserts_mixed_records_after_tct_before_trailing_writes():
    data = bytearray(witness_fixture_insts())
    tct_end = len(data)
    trailing = (
        write32(address(0, 0, 0x34008), 0)
        + write32(address(0, 0, 0x3404C), 0)
    )
    data.extend(trailing)
    struct.pack_into("<I", data, 8, struct.unpack_from("<I", data, 8)[0] + 2)
    struct.pack_into("<I", data, 12, len(data))
    inserted = (
        struct.pack("<IIQIII", 3, 0, 0xFFF20, 1, 1, 28)
        + b"\x05\x00\x00\x00"
    )

    patched = pm.patcher.insert_records_after_last_tct(bytes(data), inserted)

    expected = data[:tct_end] + inserted + data[tct_end:]
    struct.pack_into("<I", expected, 8, struct.unpack_from("<I", data, 8)[0] + 2)
    struct.pack_into("<I", expected, 12, len(expected))
    assert patched == bytes(expected)


def test_builds_exact_real_column_gate_pair_from_named_sources(tmp_path):
    data, _ = pm.patcher.patch_trace_control(
        witness_fixture_insts(), 0, 2, "core", stop_event=0,
    )
    data, _ = pm.patcher.patch_trace_control(
        data, 0, 0, "shim", stop_event=0,
    )
    data = bytearray(data)
    tct_end = len(data)
    trailing = (
        write32(address(0, 0, 0x34008), 0)
        + write32(address(0, 0, 0x3404C), 0)
    )
    data.extend(trailing)
    struct.pack_into("<I", data, 8, struct.unpack_from("<I", data, 8)[0] + 2)
    struct.pack_into("<I", data, 12, len(data))
    data = bytes(data)
    pair = pm.build_real_column_gate_pair(
        data,
        register_db(tmp_path),
        aieml_npi_source(tmp_path),
        expected_input_sha256=hashlib.sha256(data).hexdigest(),
        firmware_sha256=(
            "d13ff9fb95c6cea40213fa69e5a346552"
            "9f00bb67c0984d62343c6e31808fb9e"
        ),
        physical_start_col=1,
        num_col=1,
    )

    manifest = pair["manifest"]
    assert manifest["placement"] == {"start_col": 1, "num_col": 1}
    assert manifest["transaction_base"] == "0x0e000000"
    assert manifest["transaction_array_base"] == "0x84000000"
    assert manifest["targets"] == {
        "npi_lock": "0xac00000c",
        "npi_protection": "0xac000200",
        "column_clock": "0x860fff20",
    }
    gate = manifest["arms"]["treatment"]["operations"][:13]
    assert [(op["opcode"], op.get("value")) for op in gate] == [
        ("write32", "0xf9e8d7c6"),
        ("mask_poll", "0x00000000"),
        ("write32", "0x00000103"),
        ("mask_poll", "0x00000000"),
        ("write32", "0x00000000"),
        ("mask_poll", "0x00000000"),
        ("mask_write", "0x00000000"),
        ("write32", "0xf9e8d7c6"),
        ("mask_poll", "0x00000000"),
        ("write32", "0x00000102"),
        ("mask_poll", "0x00000000"),
        ("write32", "0x00000000"),
        ("mask_poll", "0x00000000"),
    ]
    treatment_phases = [
        op["phase"] for op in manifest["arms"]["treatment"]["operations"]
    ]
    assert treatment_phases.count("gate_dwell") == 256
    assert treatment_phases.count("restore_dwell") == 256
    restore = manifest["arms"]["treatment"]["operations"][269:282]
    assert restore[6]["opcode"] == "mask_write"
    assert restore[6]["value"] == "0x00000001"
    assert gate[9]["value"] == restore[9]["value"] == "0x00000102"
    assert gate[11]["value"] == restore[11]["value"] == "0x00000000"
    register_ops = [
        op for op in manifest["arms"]["treatment"]["operations"]
        if op["opcode"] != "noop"
    ]
    assert {op["expected_firmware_target"] for op in register_ops} == set(
        manifest["targets"].values()
    )
    assert {op["reg_offset_high"] for op in register_ops} == {"0x00000000"}
    assert pair["control"][-len(trailing):] == trailing
    assert pair["treatment"][-len(trailing):] == trailing
    differing_words = [
        offset for offset in range(0, len(pair["control"]), 4)
        if pair["control"][offset:offset + 4]
        != pair["treatment"][offset:offset + 4]
    ]
    assert differing_words == [manifest["one_word_diff"]["byte_offset"]]
    assert struct.unpack_from("<I", pair["control"], differing_words[0])[0] == 1
    assert struct.unpack_from("<I", pair["treatment"], differing_words[0])[0] == 0


def test_superseded_raw_gate_pair_preserves_open_ended_trace_controls(tmp_path):
    data, _ = pm.patcher.patch_trace_control(
        witness_fixture_insts(), 0, 2, "core", stop_event=0,
    )
    data, _ = pm.patcher.patch_trace_control(
        data, 0, 0, "shim", stop_event=0,
    )

    pair = pm.build_real_column_gate_pair(
        data,
        register_db(tmp_path),
        aieml_npi_source(tmp_path),
        expected_input_sha256=hashlib.sha256(data).hexdigest(),
        firmware_sha256=(
            "d13ff9fb95c6cea40213fa69e5a346552"
            "9f00bb67c0984d62343c6e31808fb9e"
        ),
        physical_start_col=1,
        num_col=1,
    )

    writes = {
        ((addr >> 20) & 0x1F, addr & 0xFFFFF): value
        for _, addr, value in pm.patcher._walk_write32(pair["control"])
    }
    assert writes[(2, 0x340D0)] == 0x007A0000
    assert writes[(0, 0x340D0)] == 0x007F0000


def test_real_column_gate_rejects_unpinned_identity_or_placement(tmp_path):
    data = witness_fixture_insts()
    kwargs = {
        "expected_input_sha256": hashlib.sha256(data).hexdigest(),
        "firmware_sha256": (
            "d13ff9fb95c6cea40213fa69e5a346552"
            "9f00bb67c0984d62343c6e31808fb9e"
        ),
        "physical_start_col": 1,
        "num_col": 1,
    }

    with pytest.raises(ValueError, match="input instruction hash"):
        pm.build_real_column_gate_pair(
            data, register_db(tmp_path), aieml_npi_source(tmp_path),
            **{**kwargs, "expected_input_sha256": "0" * 64},
        )
    with pytest.raises(ValueError, match="firmware hash"):
        pm.build_real_column_gate_pair(
            data, register_db(tmp_path), aieml_npi_source(tmp_path),
            **{**kwargs, "firmware_sha256": "0" * 64},
        )
    with pytest.raises(ValueError, match="physical placement 1:1"):
        pm.build_real_column_gate_pair(
            data, register_db(tmp_path), aieml_npi_source(tmp_path),
            **{**kwargs, "physical_start_col": 2},
        )


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "#define XAIEML_NPI_PCSR_LOCK 0X0000000CU\n",
            "",
            "missing aie-rt NPI macro",
        ),
        (
            "#define XAIEML_NPI_PCSR_LOCK 0X0000000CU",
            "#define XAIEML_NPI_PCSR_LOCK 0XFFFFFFFFU",
            "allowlisted target does not fit",
        ),
        (
            "#define XAIEML_NPI_PCSR_LOCK 0X0000000CU",
            "#define XAIEML_NPI_PCSR_LOCK 0X0000000DU",
            "NPI transaction offset is invalid",
        ),
        (
            "#define XAIEML_NPI_PROT_REG_CNTR_FIRSTCOL_MSK 0x000000FEU",
            "#define XAIEML_NPI_PROT_REG_CNTR_FIRSTCOL_MSK 0x00000001U",
            "protection fields overlap",
        ),
        (
            "#define XAIEML_NPI_PROT_REG_CNTR 0x00000200U",
            "#define XAIEML_NPI_PROT_REG_CNTR 0x0000000CU",
            "three distinct allowlisted targets",
        ),
    ],
)
def test_real_column_gate_rejects_unsafe_aiert_derivation(
    tmp_path, old, new, message,
):
    data = witness_fixture_insts()
    source = aieml_npi_source(tmp_path)
    source.write_text(source.read_text().replace(old, new))

    with pytest.raises(ValueError, match=message):
        pm.build_real_column_gate_pair(
            data,
            register_db(tmp_path),
            source,
            expected_input_sha256=hashlib.sha256(data).hexdigest(),
            firmware_sha256=(
                "d13ff9fb95c6cea40213fa69e5a346552"
                "9f00bb67c0984d62343c6e31808fb9e"
            ),
            physical_start_col=1,
            num_col=1,
        )


def test_real_column_gate_rejects_clock_field_outside_register(tmp_path):
    data = witness_fixture_insts()
    db = register_db(tmp_path)
    document = json.loads(db.read_text())
    clock = next(
        register
        for register in document["modules"]["shim"]["registers"]
        if register["name"] == "Column_Clock_Control"
    )
    clock["bit_fields"][0]["bit_range"] = [32, 32]
    db.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="Clock_Buffer_Enable"):
        pm.build_real_column_gate_pair(
            data,
            db,
            aieml_npi_source(tmp_path),
            expected_input_sha256=hashlib.sha256(data).hexdigest(),
            firmware_sha256=(
                "d13ff9fb95c6cea40213fa69e5a346552"
                "9f00bb67c0984d62343c6e31808fb9e"
            ),
            physical_start_col=1,
            num_col=1,
        )


def test_real_column_gate_requires_single_clock_enable_bit(tmp_path):
    data = witness_fixture_insts()
    db = register_db(tmp_path)
    document = json.loads(db.read_text())
    clock = next(
        register
        for register in document["modules"]["shim"]["registers"]
        if register["name"] == "Column_Clock_Control"
    )
    clock["bit_fields"][0]["bit_range"] = [0, 1]
    db.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="single bit"):
        pm.build_real_column_gate_pair(
            data,
            db,
            aieml_npi_source(tmp_path),
            expected_input_sha256=hashlib.sha256(data).hexdigest(),
            firmware_sha256=(
                "d13ff9fb95c6cea40213fa69e5a346552"
                "9f00bb67c0984d62343c6e31808fb9e"
            ),
            physical_start_col=1,
            num_col=1,
        )


def test_instrument_shim_witness_derives_complete_configuration(tmp_path):
    db = register_db(tmp_path)
    periodic = pm.instrument_comparator(
        witness_fixture_insts(), threshold=64,
        register_db=db, event_ids=EVENT_IDS,
    )

    patched = pm.instrument_shim_witness(
        periodic, register_db=db, core_event_ids=EVENT_IDS,
        shim_event_ids=SHIM_EVENT_IDS, threshold=64,
    )
    writes = {
        ((addr >> 20) & 0x1F, addr & 0xFFFFF): value
        for _, addr, value in pm.patcher._walk_write32(patched)
    }

    assert writes[(0, 0x340D0)] == 0x007F0000
    assert writes[(0, 0x340E0)] == 0x057B160E
    assert writes[(0, 0x31000)] == 127
    assert writes[(0, 0x31008)] == 5
    assert writes[(0, 0x31080)] == 64
    assert writes[(2, 0x34044)] == 8
    assert struct.unpack_from("<I", patched, 8)[0] == 17

    offsets = {
        ((addr >> 20) & 0x1F, addr & 0xFFFFF): offset
        for offset, addr, _ in pm.patcher._walk_write32(patched)
    }
    start = next(
        offset for offset, addr, _ in pm.patcher._walk_write32(patched)
        if ((addr >> 20) & 0x1F, addr & 0xFFFFF) == (0, 0x3404C)
    )
    assert offsets[(0, 0x31000)] < start
    assert offsets[(0, 0x31008)] < start
    assert offsets[(0, 0x31080)] < start
    assert offsets[(2, 0x34044)] < start


@pytest.mark.parametrize(
    "occupied_module", ["core", "memmod", "memtile", "shim"],
)
def test_instrument_shim_witness_rejects_occupied_channel(
    tmp_path, occupied_module,
):
    db = register_db(tmp_path)
    periodic = pm.instrument_comparator(
        witness_fixture_insts(occupied_module), threshold=64,
        register_db=db, event_ids=EVENT_IDS,
    )

    with pytest.raises(ValueError, match="broadcast channel 13 is already configured"):
        pm.instrument_shim_witness(
            periodic, register_db=db, core_event_ids=EVENT_IDS,
            shim_event_ids=SHIM_EVENT_IDS, threshold=64,
        )


def test_instrument_shim_only_omits_core_broadcast(tmp_path):
    db = register_db(tmp_path)
    periodic = pm.instrument_comparator(
        witness_fixture_insts(), threshold=64,
        register_db=db, event_ids=EVENT_IDS,
    )

    patched = pm.instrument_shim_witness(
        periodic, register_db=db, core_event_ids=EVENT_IDS,
        shim_event_ids=SHIM_EVENT_IDS, threshold=64,
        enable_transport=False,
    )

    assert not any(
        ((addr >> 20) & 0x1F, addr & 0xFFFFF) == (2, 0x34044)
        for _, addr, _ in pm.patcher._walk_write32(patched)
    )
    assert any(
        ((addr >> 20) & 0x1F, addr & 0xFFFFF) == (0, 0x31080)
        for _, addr, _ in pm.patcher._walk_write32(patched)
    )


def event(name, ts, pkt_type=0, row=2):
    return {"name": name, "ts": ts, "col": 1, "row": row,
            "pkt_type": pkt_type}


def valid_prefix():
    return [
        event("PERF_CNT_2", 10),
        event("DMA_MM2S_0_FINISHED_BD", 15, pkt_type=1),
        event("INSTR_LOCK_RELEASE_REQ", 20),
    ]


def witness_events():
    return [
        event("DMA_MM2S_0_FINISHED_BD", 90, pkt_type=1),
        event("PM_ADDRESS_OUT_OF_RANGE", 100),
        event("PERF_CNT_3", 164),
        event("PERF_CNT_3", 229),
        event("PERF_CNT_3", 294),
        event("PERF_CNT_0", 90, pkt_type=2, row=0),
        event("PERF_CNT_0", 155, pkt_type=2, row=0),
        event("PERF_CNT_0", 220, pkt_type=2, row=0),
        event("PERF_CNT_0", 285, pkt_type=2, row=0),
        event("PERF_CNT_0", 350, pkt_type=2, row=0),
        event("PERF_CNT_0", 415, pkt_type=2, row=0),
        event("PERF_CNT_0", 480, pkt_type=2, row=0),
        event("BROADCAST_A_13", 200, pkt_type=2, row=0),
        event("BROADCAST_A_13", 265, pkt_type=2, row=0),
        event("BROADCAST_A_13", 330, pkt_type=2, row=0),
    ]


def real_gate_events(core, broadcasts, heartbeats):
    return (
        [event("PM_ADDRESS_OUT_OF_RANGE", 35)]
        + [event("PERF_CNT_3", ts) for ts in core]
        + [event("BROADCAST_A_13", ts, pkt_type=2, row=0) for ts in broadcasts]
        + [event("PERF_CNT_0", ts, pkt_type=2, row=0) for ts in heartbeats]
    )


def real_gate_case(arm):
    if arm == "control":
        series = {
            "core": [100, 165, 230, 295, 360, 425, 490],
            "broadcasts": [110, 175, 240, 305, 370, 435, 500],
            "heartbeats": [0, 65, 130, 195, 260, 325, 390, 455],
        }
    else:
        series = {
            "core": [100, 165, 230, 295, 360, 425],
            "broadcasts": [10, 75, 140, 465, 530, 595],
            "heartbeats": [0, 65, 130, 195, 260, 325, 390, 455, 520, 585, 650],
        }
    clock = {"power_mode": "default", "mp_npu_mhz": 600, "h_mhz": 1028}
    return {
        "arm": arm,
        "events": real_gate_events(**series),
        "output": b"ok",
        "expected_output": b"ok",
        "clock_before": clock,
        "clock_after": dict(clock),
        "command_ok": True,
        "canary_ok": True,
    }


def replace_real_gate_series(case, **changes):
    series = {
        "core": [
            item["ts"] for item in case["events"]
            if item["name"] == "PERF_CNT_3"
        ],
        "broadcasts": [
            item["ts"] for item in case["events"]
            if item["name"] == "BROADCAST_A_13"
        ],
        "heartbeats": [
            item["ts"] for item in case["events"]
            if item["name"] == "PERF_CNT_0"
        ],
    }
    series.update(changes)
    return {**case, "events": real_gate_events(**series)}


def register_witness_log(arm, *, freezes=False, resumes=True):
    core_restored = 1_800 if freezes else 2_200
    core_stopped = core_restored + (300 if resumes else 100)
    trace_restored = 0x300 if freezes else 0x100
    trace_stopped = 0x300 if freezes else 0
    values = {
        "arm": arm,
        "period": "65",
        "core_before": "0x000003e8",
        "shim_before": "0x0000044c",
        "core_restored": f"0x{core_restored:08x}",
        "shim_restored": "0x00000834",
        "core_stopped": f"0x{core_stopped:08x}",
        "shim_stopped": "0x00000960",
        "trace_before_core": "0x00000100",
        "trace_before_shim": "0x00000100",
        "trace_restored_core": f"0x{trace_restored:08x}",
        "trace_restored_shim": "0x00000100",
        "trace_stopped_core": f"0x{trace_stopped:08x}",
        "trace_stopped_shim": "0x00000000",
    }
    return "kernel: PHOENIX_COLUMN_GATE_WITNESS " + " ".join(
        f"{key}={value}" for key, value in values.items()
    ) + "\n"


def test_real_column_gate_register_witness_is_strict_and_structured():
    log = register_witness_log("treatment", freezes=True)

    witness = pm.parse_real_column_gate_register_witness(log)

    assert witness == {
        "arm": "treatment",
        "period": 65,
        "timers": {
            "before_gate": {"core": 1_000, "shim": 1_100},
            "after_restore": {"core": 1_800, "shim": 2_100},
            "after_stop": {"core": 2_100, "shim": 2_400},
        },
        "trace_status": {
            "before_gate": {"core": 0x100, "shim": 0x100},
            "after_restore": {"core": 0x300, "shim": 0x100},
            "after_stop": {"core": 0x300, "shim": 0},
        },
    }
    with pytest.raises(ValueError, match="exactly one"):
        pm.parse_real_column_gate_register_witness(log + log)
    with pytest.raises(ValueError, match="malformed"):
        pm.parse_real_column_gate_register_witness(
            log.replace("core_stopped=0x00000834", "core_stopped=2100"),
        )


def test_real_column_gate_register_witness_proves_nested_timer_inversion():
    control = pm.parse_real_column_gate_register_witness(
        register_witness_log("control"),
    )
    treatment = pm.parse_real_column_gate_register_witness(
        register_witness_log("treatment", freezes=True),
    )

    control_verdict = pm.classify_real_column_gate_register_witness(
        "control", control,
    )
    treatment_verdict = pm.classify_real_column_gate_register_witness(
        "treatment", treatment, control,
    )

    assert control_verdict["qualified"] is True
    assert control_verdict["reason"] == "control"
    assert control_verdict["timer_deltas"] == {
        "gate_core": 1_200,
        "gate_shim": 1_000,
        "resume_core": 300,
        "resume_shim": 300,
    }
    assert treatment_verdict["qualified"] is True
    assert treatment_verdict["reason"] == "freeze_resume"
    assert treatment_verdict["timer_deltas"]["gate_core"] == 800
    assert treatment_verdict["timer_deltas"]["gate_shim"] == 1_000
    assert treatment_verdict["trace_status"]["after_stop"]["core"] == 0x300


def test_real_column_gate_register_witness_rejects_missing_causal_edges():
    control = pm.parse_real_column_gate_register_witness(
        register_witness_log("control"),
    )
    no_freeze = pm.parse_real_column_gate_register_witness(
        register_witness_log("treatment"),
    )
    no_resume = pm.parse_real_column_gate_register_witness(
        register_witness_log("treatment", freezes=True, resumes=False),
    )

    assert pm.classify_real_column_gate_register_witness(
        "treatment", no_freeze, control,
    )["reason"] == "core_timer_did_not_freeze"
    assert pm.classify_real_column_gate_register_witness(
        "treatment", no_resume, control,
    )["reason"] == "core_timer_did_not_resume"
    assert pm.classify_real_column_gate_register_witness(
        "treatment", no_freeze,
    )["reason"] == "missing_control_witness"


def test_physical_register_witness_replaces_only_the_physical_oracle():
    trace_result = {
        "qualified": False,
        "classification": {
            "qualified": False,
            "reason": "missing_or_multiple_gate_gaps",
        },
    }
    control = pm.apply_physical_real_column_gate_witness(
        "control", trace_result, register_witness_log("control"),
    )
    treatment = pm.apply_physical_real_column_gate_witness(
        "treatment", trace_result,
        register_witness_log("treatment", freezes=True),
        control,
    )

    assert control["classification"]["reason"] == "control"
    assert treatment["classification"]["reason"] == "freeze_resume"
    assert treatment["qualified"] is True
    assert treatment["trace_classification"] == trace_result["classification"]
    assert treatment["register_witness"]["trace_status"][
        "after_restore"
    ]["core"] == 0x300


def test_real_column_gate_classifier_accepts_control_and_freeze_resume():
    control = pm.classify_real_column_gate(**real_gate_case("control"))
    treatment = pm.classify_real_column_gate(**real_gate_case("treatment"))

    assert control["qualified"] is True
    assert control["reason"] == "control"
    assert control["cadence"] == 65
    assert treatment["qualified"] is True
    assert treatment["reason"] == "freeze_resume"
    assert treatment["broadcast_gap"] == {
        "left": 140,
        "right": 465,
        "cycles": 325,
        "shim_heartbeats_inside": 5,
    }


def test_real_column_gate_classifier_requires_native_pm_fault():
    case = real_gate_case("control")
    case["events"] = [
        item for item in case["events"]
        if item["name"] != "PM_ADDRESS_OUT_OF_RANGE"
    ]

    verdict = pm.classify_real_column_gate(**case)

    assert verdict["qualified"] is False
    assert verdict["reason"] == "missing_pm_fault"


def test_real_column_gate_kvm_disposition_admits_only_exact_scheduler_red():
    qualified = pm.classify_real_column_gate(**real_gate_case("control"))
    red_case = replace_real_gate_series(
        real_gate_case("control"), core=[], broadcasts=[],
    )
    red_case["events"] = [
        item for item in red_case["events"]
        if item["name"] != "PM_ADDRESS_OUT_OF_RANGE"
    ]
    scheduler_red = pm.classify_real_column_gate(**red_case)

    assert pm.classify_real_column_gate_kvm_disposition(qualified) == {
        "admitted": True,
        "reason": "behavioral_witness",
    }
    assert scheduler_red["reason"] == "missing_pm_fault"
    assert scheduler_red["series"] == {
        "core": [],
        "broadcasts": [],
        "heartbeats": [0, 65, 130, 195, 260, 325, 390, 455],
    }
    assert pm.classify_real_column_gate_kvm_disposition(scheduler_red) == {
        "admitted": True,
        "reason": "known_scheduler_red",
    }

    for series in (
        {"core": [100], "broadcasts": []},
        {"core": [], "broadcasts": [110]},
        {"core": [], "broadcasts": [],
         "heartbeats": [0, 65, 131, 196, 261, 326, 391]},
    ):
        rejected_case = replace_real_gate_series(red_case, **series)
        rejected_case["events"] = [
            item for item in rejected_case["events"]
            if item["name"] != "PM_ADDRESS_OUT_OF_RANGE"
        ]
        rejected = pm.classify_real_column_gate(**rejected_case)
        assert pm.classify_real_column_gate_kvm_disposition(rejected) == {
            "admitted": False,
            "reason": "behavioral_failure",
        }


def test_real_column_gate_artifact_classifier_records_exact_evidence(tmp_path):
    case = real_gate_case("control")
    events = tmp_path / "events.json"
    output = tmp_path / "output.bin"
    expected = tmp_path / "expected.bin"
    canary = tmp_path / "canary.bin"
    before = tmp_path / "clock-before.json"
    after = tmp_path / "clock-after.json"
    kernel_log = tmp_path / "dmesg.log"
    events.write_text(json.dumps({"slot_names": {}, "events": case["events"]}))
    output.write_bytes(case["output"])
    expected.write_bytes(case["expected_output"])
    canary.write_bytes(case["expected_output"])
    before.write_text(json.dumps(case["clock_before"]))
    after.write_text(json.dumps(case["clock_after"]))
    kernel_log.write_text(register_witness_log("control"))

    result = pm.classify_real_column_gate_artifacts(
        "control", events, output, expected, before, after, canary, kernel_log,
    )

    assert result["qualified"] is True
    assert result["classification"]["reason"] == "control"
    assert result["kvm_disposition"] == {
        "admitted": True,
        "reason": "behavioral_witness",
    }
    assert result["output"]["matches"] is True
    assert result["canary"]["matches"] is True
    assert result["clock_before"] == case["clock_before"]
    assert result["clock_after"] == case["clock_after"]
    assert result["register_witness"]["arm"] == "control"


@pytest.mark.parametrize(("mutate", "reason"), [
    (
        lambda case: replace_real_gate_series(case, heartbeats=[]),
        "irregular_shim_heartbeat",
    ),
    (
        lambda case: replace_real_gate_series(
            case, heartbeats=[0, 65, 131, 196, 261, 326, 391],
        ),
        "irregular_shim_heartbeat",
    ),
    (
        lambda case: replace_real_gate_series(
            case, broadcasts=[10, 75, 400, 465, 530, 595],
        ),
        "insufficient_pre_gate_samples",
    ),
    (
        lambda case: replace_real_gate_series(
            case, broadcasts=[10, 75, 140, 205, 530, 595],
        ),
        "insufficient_post_restore_samples",
    ),
    (
        lambda case: replace_real_gate_series(
            case, broadcasts=[10, 75, 140, 335, 400, 465],
        ),
        "missing_or_multiple_gate_gaps",
    ),
    (
        lambda case: replace_real_gate_series(
            case, broadcasts=[10, 75, 400, 465, 790, 855],
        ),
        "missing_or_multiple_gate_gaps",
    ),
    (
        lambda case: replace_real_gate_series(
            case, broadcasts=[10, 75, 140, 465, 531, 596],
        ),
        "irregular_broadcast_cadence",
    ),
    (
        lambda case: replace_real_gate_series(
            case,
            core=[100, 165, 230],
            broadcasts=[10, 75, 140],
        ),
        "insufficient_broadcast_samples",
    ),
    (
        lambda case: replace_real_gate_series(
            case, core=[100, 165, 231, 296, 361, 426],
        ),
        "irregular_core_heartbeat",
    ),
    (
        lambda case: replace_real_gate_series(
            case, core=[100, 165, 230, 295, 360],
        ),
        "core_to_shim_count_mismatch",
    ),
    (
        lambda case: replace_real_gate_series(case, heartbeats=[195, 260]),
        "shim_not_live_inside_gate",
    ),
    (
        lambda case: replace_real_gate_series(case, heartbeats=[195, 260, 325]),
        "shim_heartbeat_does_not_span_gate",
    ),
    (lambda case: {**case, "command_ok": False}, "command_failed"),
    (lambda case: {**case, "output": b"bad"}, "output_mismatch"),
    (
        lambda case: {
            **case,
            "clock_after": {**case["clock_after"], "h_mhz": 1024},
        },
        "clocks_changed",
    ),
    (
        lambda case: {
            **case,
            "clock_before": {**case["clock_before"], "h_mhz": True},
            "clock_after": {**case["clock_after"], "h_mhz": True},
        },
        "missing_clock_identity",
    ),
    (lambda case: {**case, "canary_ok": False}, "canary_failed"),
])
def test_real_column_gate_classifier_rejects_malformed_evidence(mutate, reason):
    verdict = pm.classify_real_column_gate(**mutate(real_gate_case("treatment")))

    assert verdict["qualified"] is False
    assert verdict["reason"] == reason


def test_real_column_gate_control_requires_full_periodic_witness():
    case = real_gate_case("control")
    too_short = replace_real_gate_series(
        case,
        core=[100, 165, 230, 295, 360, 425],
        broadcasts=[110, 175, 240, 305, 370, 435],
        heartbeats=[0, 65, 130, 195, 260, 325],
    )
    irregular = replace_real_gate_series(
        case,
        broadcasts=[110, 175, 240, 305, 371, 436, 501],
    )

    assert pm.classify_real_column_gate(**too_short)["reason"] == (
        "insufficient_shim_heartbeats"
    )
    assert pm.classify_real_column_gate(**irregular)["reason"] == (
        "irregular_control_broadcast"
    )


def test_shim_witness_classifier_proves_same_domain_liveness():
    verdict = pm.classify_shim_witness(witness_events(), b"ok", b"ok")

    assert verdict["qualified"] is True
    assert verdict["reason"] == "qualified"
    assert verdict["broadcast_cadence"] == 65
    assert verdict["heartbeat_cadence"] == 65


@pytest.mark.parametrize(("mutate", "reason"), [
    (
        lambda events: [
            event for event in events
            if not (event["name"] == "BROADCAST_A_13" and event["ts"] == 330)
        ],
        "core_to_shim_count_mismatch",
    ),
    (
        lambda events: [
            {**event, "ts": 266}
            if event["name"] == "BROADCAST_A_13" and event["ts"] == 265
            else event for event in events
        ],
        "irregular_broadcast_cadence",
    ),
    (
        lambda events: [
            {**event, "ts": 221}
            if event["name"] == "PERF_CNT_0" and event["ts"] == 220
            else event for event in events
        ],
        "irregular_shim_heartbeat",
    ),
    (
        lambda events: [
            {**event, "ts": 90 + index * 66}
            if event["name"] == "PERF_CNT_0" else event
            for index, event in enumerate(events)
        ],
        "unexpected_shim_heartbeat_cadence",
    ),
    (
        lambda events: [
            event for event in events
            if event["name"] != "PERF_CNT_0" or event["ts"] <= 350
        ],
        "shim_not_live_after_missing_core_heartbeat",
    ),
])
def test_shim_witness_classifier_rejects_incomplete_evidence(mutate, reason):
    verdict = pm.classify_shim_witness(mutate(witness_events()), b"ok", b"ok")

    assert verdict["qualified"] is False
    assert verdict["reason"] == reason


def test_shim_witness_classifier_rejects_core_signature_change():
    baseline = pm.classify_shim_witness(witness_events(), b"ok", b"ok")

    verdict = pm.classify_shim_witness(
        witness_events(), b"ok", b"ok",
        expected_core_signature=[("different", 0)],
    )

    assert baseline["qualified"] is True
    assert verdict["qualified"] is False
    assert verdict["reason"] == "core_signature_mismatch"


def test_shim_witness_no_fault_control_rejects_spurious_transport():
    heartbeats = [
        event("PERF_CNT_0", ts, pkt_type=2, row=0)
        for ts in (90, 155, 220, 285)
    ]
    control = pm.classify_shim_witness(
        heartbeats, b"ok", b"ok", no_fault_control=True,
    )
    spurious = pm.classify_shim_witness(
        heartbeats + [event("BROADCAST_A_13", 200, pkt_type=2, row=0)],
        b"ok", b"ok", no_fault_control=True,
    )

    assert control["qualified"] is True
    assert control["reason"] == "control"
    assert spurious["qualified"] is False
    assert spurious["reason"] == "spurious_core_broadcast"


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


def test_relabels_only_shim_witness_slots():
    document = {
        "events": [
            {**event("NONE", 30, pkt_type=2, row=0), "slot": 2},
            {**event("NONE", 31, pkt_type=2, row=0), "slot": 3},
            {**event("INSTR_EVENT_1", 32), "slot": 2},
        ],
        "slot_names": {
            "core": ["a", "b", "INSTR_EVENT_1"],
            "shim": ["start", "finish", "NONE", "NONE"],
        },
    }

    pm.relabel_shim_witness_events(document)

    assert document["events"][0]["name"] == "BROADCAST_A_13"
    assert document["events"][1]["name"] == "PERF_CNT_0"
    assert document["events"][2]["name"] == "INSTR_EVENT_1"
    assert document["slot_names"]["shim"][2:] == [
        "BROADCAST_A_13", "PERF_CNT_0",
    ]


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
