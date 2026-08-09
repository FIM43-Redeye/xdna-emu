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


def register_db(tmp_path, bd_words=8):
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
            {"name": "DMA_BD13_0", "offset": "0x1c1a0"},
            {
                "name": "DMA_BD13_7",
                "offset": "0x1c1bc",
                "bit_fields": [
                    {"name": "Next_BD", "bit_range": [27, 30]},
                    {"name": "Use_Next_BD", "bit_range": [26, 26]},
                ],
            },
            {"name": "DMA_BD14_0", "offset": "0x1c1c0"},
            {
                "name": "DMA_BD14_7",
                "offset": "0x1c1dc",
                "bit_fields": [
                    {"name": "Next_BD", "bit_range": [27, 30]},
                    {"name": "Use_Next_BD", "bit_range": [26, 26]},
                ],
            },
            {
                "name": "DMA_BD15_0",
                "offset": f"{0x1C1C0 + 4 * bd_words:#x}",
            },
            {
                "name": "DMA_BD15_7",
                "offset": f"{0x1C1C0 + 4 * bd_words + 28:#x}",
                "bit_fields": [
                    {"name": "Next_BD", "bit_range": [27, 30]},
                    {"name": "Use_Next_BD", "bit_range": [26, 26]},
                ],
            },
            {
                "name": "DMA_S2MM_0_Task_Queue",
                "offset": "0x1c204",
                "bit_fields": [
                    {"name": "Start_BD_ID", "bit_range": [0, 3]},
                ],
            },
            {
                "name": "DMA_S2MM_1_Task_Queue",
                "offset": "0x1c20c",
                "bit_fields": [
                    {"name": "Start_BD_ID", "bit_range": [0, 3]},
                ],
            },
            {
                "name": "DMA_MM2S_0_Task_Queue",
                "offset": "0x1c214",
                "bit_fields": [
                    {"name": "Start_BD_ID", "bit_range": [0, 3]},
                ],
            },
            {
                "name": "DMA_MM2S_1_Task_Queue",
                "offset": "0x1c21c",
                "bit_fields": [
                    {"name": "Start_BD_ID", "bit_range": [0, 3]},
                ],
            },
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


def mlir_aie_preempt_sources(tmp_path, *, shift=8, level_zero="Noop."):
    root = tmp_path / "mlir-aie"
    encoding = root / "include/aie/Runtime/TxnEncoding.h"
    dialect = root / "include/aie/Dialect/AIEX/IR/AIEX.td"
    encoding.parent.mkdir(parents=True)
    dialect.parent.mkdir(parents=True)
    encoding.write_text("""
enum TxnOpcode : uint32_t {
  TXN_OPC_PREEMPT = 6,
};
inline void txn_append_preempt(std::vector<uint32_t> &txn, uint32_t level) {
  txn.push_back(TXN_OPC_PREEMPT | (level << SHIFT));
}
""".replace("SHIFT", str(shift)))
    dialect.write_text(f"""
def AIE_NpuPreemptOp {{
  let description = [{{
    Levels:
    0: {level_zero}
  }}];
}}
""")
    return root


def mlir_aie_write32_source(tmp_path, *, opcode=0, words=6, extra_bits=0):
    root = tmp_path / "mlir-aie-write32"
    encoding = root / "include/aie/Runtime/TxnEncoding.h"
    encoding.parent.mkdir(parents=True, exist_ok=True)
    encoding.write_text(f"""
enum TxnOpcode : uint32_t {{
  TXN_OPC_WRITE = {opcode},
}};
inline void txn_append_write32(std::vector<uint32_t> &txn, uint32_t addr,
                               uint32_t val) {{
  size_t pos = txn.size();
  txn.resize(pos + {words}, 0);
  txn[pos + 0] = TXN_OPC_WRITE;
  txn[pos + 2] = addr;
  txn[pos + 3] = {extra_bits};
  txn[pos + 4] = val;
  txn[pos + 5] = {words} * sizeof(uint32_t);
}}
""")
    return root


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


def test_derives_level_zero_preempt_record_from_mlir_aie_sources(tmp_path):
    assert pm.derive_level_zero_preempt_record(
        mlir_aie_preempt_sources(tmp_path),
    ) == b"\x06\x00\x00\x00"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shift": 16},
        {"level_zero": "Mem tile."},
    ],
)
def test_preempt_record_derivation_rejects_changed_contract(tmp_path, kwargs):
    with pytest.raises(ValueError, match=r"PREEMPT\(0\).*changed"):
        pm.derive_level_zero_preempt_record(
            mlir_aie_preempt_sources(tmp_path, **kwargs),
        )


def test_derives_write32_record_from_mlir_aie_source(tmp_path):
    assert pm.derive_write32_record(
        mlir_aie_write32_source(tmp_path), 0x1C1C0, 0,
    ) == bytes.fromhex(
        "00000000 00000000 c0c10100 00000000 00000000 18000000"
    )
    assert pm.derive_write32_record(
        mlir_aie_write32_source(tmp_path, opcode=9), 0x1C1C0, 0,
    )[:4] == b"\x09\x00\x00\x00"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"words": 7},
        {"extra_bits": 1},
    ],
)
def test_write32_record_derivation_rejects_changed_layout(tmp_path, kwargs):
    with pytest.raises(ValueError, match="WRITE32 encoding changed"):
        pm.derive_write32_record(
            mlir_aie_write32_source(tmp_path, **kwargs), 0x1C1C0, 0,
        )


@pytest.mark.parametrize("address,value", [(-1, 0), (1 << 32, 0), (0, -1), (0, 1 << 32)])
def test_write32_record_derivation_rejects_non_u32_operands(
    tmp_path, address, value,
):
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        pm.derive_write32_record(
            mlir_aie_write32_source(tmp_path), address, value,
        )


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


def firmware_timeline_fixture_insts(extra_before_tct=()):
    records = [
        write32(address(0, 0, 0x340D0), 0x7E7F0000),
        write32(address(0, 0, 0x340E0), 0x0000160E),
        write32(address(0, 0, 0x3404C), SHIM_EVENT_IDS["USER_EVENT_1"]),
        write32(address(0, 0, 0x34008), SHIM_EVENT_IDS["USER_EVENT_1"]),
        *extra_before_tct,
        struct.pack("<IIII", 0x80, 16, 0x100, 0x10100),
        write32(address(0, 0, 0x34048), SHIM_EVENT_IDS["USER_EVENT_0"]),
        write32(address(0, 0, 0x34008), SHIM_EVENT_IDS["USER_EVENT_0"]),
    ]
    payload = b"".join(records)
    return struct.pack(
        "<IIII", 0x06030100, 0, len(records), 16 + len(payload),
    ) + payload


def firmware_crossover_records(data):
    result = []
    offset = pm.patcher._last_tct_boundary(data)
    while offset < len(data) and sum(kind == "stop" for kind, _ in result) < 2:
        length = pm.patcher._instruction_length(data, offset)
        opcode = data[offset]
        if opcode == 5:
            kind = "noop"
        elif opcode == 6:
            kind = "preempt"
        elif opcode == 1:
            kind = "clear" if length == 48 else "block"
        elif opcode == 0:
            value = struct.unpack_from("<I", data, offset + 16)[0]
            kind = {
                SHIM_EVENT_IDS["USER_EVENT_1"]: "start",
                SHIM_EVENT_IDS["USER_EVENT_0"]: "stop",
            }.get(value, "write32")
        else:
            kind = f"opcode-{opcode}"
        result.append((kind, offset))
        offset += length
    return result


def test_firmware_clock_timeline_brackets_each_noop_block(tmp_path):
    blocks = (0, 1, 4, 1)
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
        if target == marker_address and value == SHIM_EVENT_IDS["PERF_CNT_0"]
    ]
    start_writes = [
        (offset, value) for offset, target, value in writes
        if target == marker_address and value == SHIM_EVENT_IDS["USER_EVENT_1"]
    ]
    assert len(marker_writes) == len(blocks) + 1
    assert len(flush_writes) == 1
    assert len(start_writes) == 1
    assert flush_writes[-1][0] > marker_writes[-1][0]
    assert struct.unpack_from("<I", patched, 8)[0] == 8 + len(blocks) + sum(blocks)
    assert struct.unpack_from("<I", patched, 12)[0] == len(patched)

    trace_control = next(
        value for _, target, value in writes
        if target == address(0, 0, 0x340D0)
    )
    trace_events = next(
        value for _, target, value in writes
        if target == address(0, 0, 0x340E0)
    )
    assert trace_control == 0x057F0000
    assert trace_events == 0x007E160E


def test_firmware_blockwrite_timeline_uses_cleared_unused_bd14(tmp_path):
    blocks = (1, 2, 4, 8, 1)
    patched = pm.instrument_firmware_blockwrite_timeline(
        firmware_timeline_fixture_insts(), register_db(tmp_path),
        SHIM_EVENT_IDS, blocks,
    )

    marker = write32(
        address(0, 0, 0x34008), SHIM_EVENT_IDS["USER_EVENT_0"],
    )
    bd14 = address(0, 0, 0x1C1C0)

    def blockwrite(words):
        return (
            struct.pack("<IIII", 1, 0, bd14, 16 + 4 * words)
            + bytes(4 * words)
        )

    expected_records = blockwrite(8) + marker + b"".join(
        blockwrite(words) + marker for words in blocks
    )
    tct_end = pm.patcher._last_tct_boundary(patched)
    assert patched[tct_end:tct_end + len(expected_records)] == expected_records
    assert struct.unpack_from("<I", patched, 8)[0] == 9 + 2 * len(blocks)
    assert struct.unpack_from("<I", patched, 12)[0] == len(patched)


def test_firmware_blockwrite_layout_derives_requested_unused_bd(tmp_path):
    assert pm._firmware_blockwrite_layout(
        firmware_timeline_fixture_insts(), register_db(tmp_path), bd_id=13,
    ) == (address(0, 0, 0x1C1A0), 8)


def test_firmware_blockwrite_layout_rejects_enabled_next_bd_link(tmp_path):
    bd15 = address(0, 0, 0x1C1E0)
    links_to_bd13 = (1 << 26) | (13 << 27)
    descriptor = struct.pack(
        "<IIII8I", 1, 0, bd15, 48, *([0] * 7), links_to_bd13,
    )
    queue = write32(address(0, 0, 0x1C20C), 0x8000000F)

    with pytest.raises(ValueError, match="BD13.*Next_BD"):
        pm._firmware_blockwrite_layout(
            firmware_timeline_fixture_insts((descriptor, queue)),
            register_db(tmp_path),
            bd_id=13,
        )


def test_firmware_blockwrite_layout_rejects_unknown_queued_bd_link(tmp_path):
    queue = address(0, 0, 0x1C20C)
    masked_queue = struct.pack(
        "<IIQIII", 3, 0, queue, 15, 0xF, 28,
    )

    with pytest.raises(ValueError, match="cannot prove.*BD15.*Next_BD"):
        pm._firmware_blockwrite_layout(
            firmware_timeline_fixture_insts((masked_queue,)),
            register_db(tmp_path),
            bd_id=13,
        )


def test_firmware_blockwrite_layout_rejects_source_bd_read(tmp_path):
    bd13 = address(0, 0, 0x1C1A0)
    poll = struct.pack("<IIQIII", 4, 0, bd13, 0, 0xFFFFFFFF, 28)

    with pytest.raises(ValueError, match="BD13 is already used"):
        pm._firmware_blockwrite_layout(
            firmware_timeline_fixture_insts((poll,)),
            register_db(tmp_path),
            bd_id=13,
        )


@pytest.mark.parametrize(
    "phases",
    [
        (0, 20, 44, 56),
        (56, 44, 20, 0),
        (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60),
    ],
)
def test_firmware_blockwrite_phase_crossover_places_distinct_fixed_windows(
    tmp_path, phases,
):
    patched = pm.instrument_firmware_blockwrite_phase_crossover(
        firmware_timeline_fixture_insts(), register_db(tmp_path),
        SHIM_EVENT_IDS, phases,
    )

    generate = address(0, 0, 0x34008)
    start = write32(generate, SHIM_EVENT_IDS["USER_EVENT_1"])
    stop = write32(generate, SHIM_EVENT_IDS["USER_EVENT_0"])
    bd14 = address(0, 0, 0x1C1C0)
    clear = struct.pack("<IIII8I", 1, 0, bd14, 48, *([0] * 8))
    one_word = struct.pack("<IIIII", 1, 0, bd14, 20, 0)

    offset = pm.patcher._last_tct_boundary(patched)
    assert patched[offset:offset + len(clear)] == clear
    offset += len(clear)
    for phase in phases:
        while patched[offset:offset + 4] == b"\x05\x00\x00\x00":
            offset += 4
        assert patched[offset:offset + len(start)] == start
        offset += len(start)
        assert offset % 64 == phase
        assert patched[offset:offset + len(one_word)] == one_word
        offset += len(one_word)
        assert patched[offset:offset + len(stop)] == stop
        offset += len(stop)

    writes = list(pm.patcher._walk_write32(patched))
    trace_events = next(
        value for _, target, value in writes
        if target == address(0, 0, 0x340E0)
    )
    assert trace_events == 0x7F7E160E


@pytest.mark.parametrize(
    "phases", [(), (0,), (0, 0), (0, 1), (0, 64), (0, -4), (0, True)],
)
def test_firmware_blockwrite_phase_crossover_rejects_invalid_phases(
    tmp_path, phases,
):
    with pytest.raises(ValueError, match="phases"):
        pm.instrument_firmware_blockwrite_phase_crossover(
            firmware_timeline_fixture_insts(), register_db(tmp_path),
            SHIM_EVENT_IDS, phases,
        )


def test_firmware_blockwrite_phase_crossover_relocates_balanced_full_turn(
    tmp_path,
):
    source = firmware_timeline_fixture_insts()
    db = register_db(tmp_path)
    control = pm.instrument_firmware_blockwrite_phase_crossover(
        source, db, SHIM_EVENT_IDS, (40, 44),
        leading_full_turns=(1, 0),
    )
    treatment = pm.instrument_firmware_blockwrite_phase_crossover(
        source, db, SHIM_EVENT_IDS, (40, 44),
        leading_full_turns=(0, 1),
    )

    control_records = firmware_crossover_records(control)
    treatment_records = firmware_crossover_records(treatment)
    assert [kind for kind, _ in control_records] == (
        ["clear"] + ["noop"] * 24
        + ["start", "block", "stop", "start", "block", "stop"]
    )
    assert [kind for kind, _ in treatment_records] == (
        ["clear"] + ["noop"] * 8
        + ["start", "block", "stop"] + ["noop"] * 16
        + ["start", "block", "stop"]
    )
    assert len(control) == len(treatment)
    assert struct.unpack_from("<II", control, 8) == struct.unpack_from("<II", treatment, 8)

    control_blocks = [
        (ordinal, offset) for ordinal, (kind, offset) in enumerate(control_records)
        if kind == "block"
    ]
    treatment_blocks = [
        (ordinal, offset) for ordinal, (kind, offset) in enumerate(treatment_records)
        if kind == "block"
    ]
    assert [offset % 64 for _, offset in control_blocks] == [40, 44]
    assert [offset % 64 for _, offset in treatment_blocks] == [40, 44]
    assert control_blocks[1] == treatment_blocks[1]


def test_firmware_blockwrite_phase_crossover_balances_three_depth_arms(tmp_path):
    source = firmware_timeline_fixture_insts()
    db = register_db(tmp_path)
    arms = [
        pm.instrument_firmware_blockwrite_phase_crossover(
            source, db, SHIM_EVENT_IDS, (40, 44), leading_full_turns=turns,
        )
        for turns in ((2, 0), (1, 1), (0, 2))
    ]
    records = [firmware_crossover_records(arm) for arm in arms]
    blocks = [
        [(ordinal, offset) for ordinal, (kind, offset) in enumerate(arm_records)
         if kind == "block"]
        for arm_records in records
    ]

    assert len({len(arm) for arm in arms}) == 1
    assert len({struct.unpack_from("<II", arm, 8) for arm in arms}) == 1
    assert [sum(kind == "noop" for kind, _ in arm) for arm in records] == [40] * 3
    assert [[offset % 64 for _, offset in arm] for arm in blocks] == [[40, 44]] * 3
    assert len({arm[1] for arm in blocks}) == 1
    assert [
        sum(kind == "noop" for kind, _ in arm_records[:arm_blocks[0][0]])
        for arm_records, arm_blocks in zip(records, blocks)
    ] == [40, 24, 8]


def test_noop_preempt_path_candidates_preserve_balanced_layout(tmp_path):
    candidates = pm.instrument_firmware_blockwrite_noop_preempt_path_candidates(
        firmware_timeline_fixture_insts(),
        register_db(tmp_path),
        SHIM_EVENT_IDS,
        mlir_aie_preempt_sources(tmp_path),
    )
    assert tuple(candidates) == ("A", "B", "C")

    records = {
        label: firmware_crossover_records(candidate)
        for label, candidate in candidates.items()
    }
    blocks = {
        label: [
            (ordinal, offset)
            for ordinal, (kind, offset) in enumerate(arm_records)
            if kind == "block"
        ]
        for label, arm_records in records.items()
    }

    assert len({len(candidate) for candidate in candidates.values()}) == 1
    assert len({
        struct.unpack_from("<II", candidate, 8)
        for candidate in candidates.values()
    }) == 1
    assert [
        [offset % 64 for _, offset in blocks[label]]
        for label in candidates
    ] == [[40, 44]] * 3
    assert len({blocks[label][1] for label in candidates}) == 1
    assert blocks["B"][0] == blocks["C"][0]

    assert [sum(kind == "noop" for kind, _ in records[label]) for label in candidates] == [24, 24, 8]
    assert [sum(kind == "preempt" for kind, _ in records[label]) for label in candidates] == [0, 0, 16]

    b = candidates["B"]
    c = candidates["C"]
    changed = [offset for offset, (left, right) in enumerate(zip(b, c)) if left != right]
    preempt_offsets = [offset for kind, offset in records["C"] if kind == "preempt"]
    assert changed == preempt_offsets
    assert all(b[offset] == 5 and c[offset] == 6 for offset in changed)
    assert all(c[offset + 1:offset + 4] == b"\x00\x00\x00" for offset in changed)


def test_blockwrite_reprime_order_candidates_swap_only_the_inter_window_order(
    tmp_path,
):
    candidates = pm.instrument_firmware_blockwrite_reprime_order_candidates(
        firmware_timeline_fixture_insts(),
        register_db(tmp_path),
        SHIM_EVENT_IDS,
    )
    assert tuple(candidates) == ("A", "B")

    records = {
        label: firmware_crossover_records(candidate)
        for label, candidate in candidates.items()
    }
    blocks = {
        label: [
            (ordinal, offset)
            for ordinal, (kind, offset) in enumerate(arm_records)
            if kind == "block"
        ]
        for label, arm_records in records.items()
    }

    assert len({len(candidate) for candidate in candidates.values()}) == 1
    assert len({
        struct.unpack_from("<II", candidate, 8)
        for candidate in candidates.values()
    }) == 1
    assert len(blocks["A"]) == len(blocks["B"]) == 3
    assert blocks["A"][0] == blocks["B"][0]
    assert blocks["A"][2] == blocks["B"][2]
    assert [offset % 64 for _, offset in blocks["A"]] == [40, 0, 44]
    assert [offset % 64 for _, offset in blocks["B"]] == [40, 0, 44]

    reprime_a = blocks["A"][1][1]
    reprime_b = blocks["B"][1][1]
    target = blocks["A"][2][1]
    assert reprime_b - reprime_a == 64
    assert target - reprime_a == 108
    assert target - reprime_b == 44

    bd14 = address(0, 0, 0x1C1C0)
    reprime = struct.pack("<IIII", 1, 0, bd14, 20) + bytes(4)
    noops = b"\x05\x00\x00\x00" * 16
    assert candidates["A"][reprime_a:reprime_a + 84] == reprime + noops
    assert candidates["B"][reprime_a:reprime_a + 84] == noops + reprime


def test_write32_recency_candidates_swap_only_the_inter_window_order(tmp_path):
    candidates = pm.instrument_firmware_write32_recency_candidates(
        firmware_timeline_fixture_insts(),
        register_db(tmp_path),
        SHIM_EVENT_IDS,
        mlir_aie_write32_source(tmp_path),
    )
    assert tuple(candidates) == ("A", "B")

    records = {
        label: firmware_crossover_records(candidate)
        for label, candidate in candidates.items()
    }
    blocks = {
        label: [
            (ordinal, offset)
            for ordinal, (kind, offset) in enumerate(arm_records)
            if kind == "block"
        ]
        for label, arm_records in records.items()
    }
    bd14 = address(0, 0, 0x1C1C0)
    treatments = {
        label: [
            (offset, value)
            for offset, target, value in pm.patcher._walk_write32(candidate)
            if target == bd14
        ]
        for label, candidate in candidates.items()
    }

    assert len({len(candidate) for candidate in candidates.values()}) == 1
    assert len({
        struct.unpack_from("<II", candidate, 8)
        for candidate in candidates.values()
    }) == 1
    assert blocks["A"] == blocks["B"]
    assert [offset % 64 for _, offset in blocks["A"]] == [40, 44]
    assert all(len(treatments[label]) == 1 for label in candidates)

    write_a = treatments["A"][0][0]
    write_b = treatments["B"][0][0]
    target = blocks["A"][1][1]
    assert treatments["A"][0][1] == treatments["B"][0][1] == 0
    assert write_b - write_a == 64
    assert write_a % 64 == write_b % 64
    assert target - write_a == 112
    assert target - write_b == 48

    treatment = bytes.fromhex(
        "00000000 00000000 c0c10100 00000000 00000000 18000000"
    )
    noops = b"\x05\x00\x00\x00" * 16
    assert candidates["A"][write_a:write_a + 88] == treatment + noops
    assert candidates["B"][write_a:write_a + 88] == noops + treatment
    assert candidates["A"][:write_a] == candidates["B"][:write_a]
    assert candidates["A"][write_a + 88:] == candidates["B"][write_a + 88:]


def test_write32_recency_candidates_reject_source_bd14_use(tmp_path):
    bd14 = address(0, 0, 0x1C1C0)
    occupied = write32(bd14, 1)

    with pytest.raises(ValueError, match="BD14 is already used"):
        pm.instrument_firmware_write32_recency_candidates(
            firmware_timeline_fixture_insts((occupied,)),
            register_db(tmp_path),
            SHIM_EVENT_IDS,
            mlir_aie_write32_source(tmp_path),
        )


@pytest.mark.parametrize(
    "turns", [(0,), (0, -1), (0, True), (0, 1.5)],
)
def test_firmware_blockwrite_phase_crossover_rejects_invalid_full_turns(
    tmp_path, turns,
):
    with pytest.raises(ValueError, match="turns"):
        pm.instrument_firmware_blockwrite_phase_crossover(
            firmware_timeline_fixture_insts(), register_db(tmp_path),
            SHIM_EVENT_IDS, (40, 44), leading_full_turns=turns,
        )


def test_classifies_firmware_blockwrite_phase_run_from_distinct_marker_pairs():
    phases = (0, 20)
    events = [
        {"pkt_type": 2, "col": 1, "row": 0,
         "name": "DMA_S2MM_0_START_TASK", "ts": 90},
        {"pkt_type": 2, "col": 1, "row": 0,
         "name": "USER_EVENT_1", "ts": 100},
        {"pkt_type": 2, "col": 1, "row": 0,
         "name": "USER_EVENT_0", "ts": 111},
        {"pkt_type": 2, "col": 1, "row": 0,
         "name": "USER_EVENT_1", "ts": 120},
        {"pkt_type": 2, "col": 1, "row": 0,
         "name": "USER_EVENT_0", "ts": 145},
        {"pkt_type": 2, "col": 1, "row": 0,
         "name": "DMA_S2MM_0_FINISHED_TASK", "ts": 160},
    ]
    clock = {
        "power_mode": "default", "power_mode_id": 0,
        "mp_npu_mhz": 400, "h_mhz": 800,
    }

    result = pm.classify_firmware_blockwrite_phase_run(
        events, phases, b"same", b"same", clock, clock,
    )

    assert result == {
        "qualified": True,
        "reason": "captured",
        "marker_timestamps": [100, 111, 120, 145],
        "intervals": [
            {"phase": 0, "array_cycles": 11},
            {"phase": 20, "array_cycles": 25},
        ],
    }


def test_firmware_blockwrite_phase_run_requires_license_for_leading_start():
    phases = (0, 20)
    events = [
        {"pkt_type": 2, "col": 1, "row": 0,
         "name": "DMA_S2MM_0_START_TASK", "ts": 80},
        {"pkt_type": 2, "col": 1, "row": 0,
         "name": "USER_EVENT_1", "ts": 90},
        {"pkt_type": 2, "col": 1, "row": 0,
         "name": "USER_EVENT_1", "ts": 100},
        {"pkt_type": 2, "col": 1, "row": 0,
         "name": "USER_EVENT_0", "ts": 111},
        {"pkt_type": 2, "col": 1, "row": 0,
         "name": "USER_EVENT_1", "ts": 120},
        {"pkt_type": 2, "col": 1, "row": 0,
         "name": "USER_EVENT_0", "ts": 145},
        {"pkt_type": 2, "col": 1, "row": 0,
         "name": "DMA_S2MM_0_FINISHED_TASK", "ts": 160},
    ]
    clock = {"mp_npu_mhz": 400, "h_mhz": 800}

    unlicensed = pm.classify_firmware_blockwrite_phase_run(
        events, phases, b"same", b"same", clock, clock,
    )
    licensed = pm.classify_firmware_blockwrite_phase_run(
        events, phases, b"same", b"same", clock, clock,
        source_start_recorded=True,
    )

    assert unlicensed["reason"] == "marker_sequence_mismatch"
    assert licensed["qualified"] is True
    assert licensed["marker_timestamps"] == [100, 111, 120, 145]


@pytest.mark.parametrize(
    "events,clock,reason",
    [
        (None, {"mp_npu_mhz": 400, "h_mhz": 800}, "malformed_trace"),
        ([None], {"mp_npu_mhz": 400, "h_mhz": 800}, "malformed_trace"),
        ([], None, "invalid_clock"),
    ],
)
def test_firmware_blockwrite_phase_run_fails_closed_on_malformed_trace(
    events, clock, reason,
):
    result = pm.classify_firmware_blockwrite_phase_run(
        events, (0, 20), b"same", b"same", clock, clock,
    )

    assert result == {"qualified": False, "reason": reason}


@pytest.mark.parametrize(
    "forward_costs,reverse_costs,reason,qualified",
    [
        ([10, 20, 30, 40], [40, 30, 20, 10], "phase_following", True),
        ([10, 20, 30, 40], [10, 20, 30, 40], "ordinal_following", True),
        ([10, 10, 10, 10], [10, 10, 10, 10], "uniform", True),
        ([10, 20, 20, 10], [10, 20, 20, 10], "ambiguous", False),
        ([10, 20, 30, 40], [11, 21, 31, 41], "mixed_or_history_dependent", True),
    ],
)
def test_classifies_firmware_blockwrite_phase_crossover_decisions(
    forward_costs, reverse_costs, reason, qualified,
):
    forward_phases = (0, 20, 44, 56)
    reverse_phases = tuple(reversed(forward_phases))

    def run(phases, costs):
        return {
            "qualified": True,
            "intervals": [
                {"phase": phase, "array_cycles": cycles}
                for phase, cycles in zip(phases, costs)
            ],
        }

    result = pm.classify_firmware_blockwrite_phase_crossover(
        [run(forward_phases, forward_costs)] * 2,
        [run(reverse_phases, reverse_costs)] * 2,
    )

    assert result["qualified"] is qualified
    assert result["reason"] == reason


def test_firmware_blockwrite_phase_crossover_rejects_nonrepeat_or_invalid_runs():
    forward_phases = (0, 20, 44, 56)
    reverse_phases = tuple(reversed(forward_phases))

    def run(phases, costs, qualified=True):
        return {
            "qualified": qualified,
            "intervals": [
                {"phase": phase, "array_cycles": cycles}
                for phase, cycles in zip(phases, costs)
            ],
        }

    nondeterministic = pm.classify_firmware_blockwrite_phase_crossover(
        [run(forward_phases, [10, 20, 30, 40]),
         run(forward_phases, [11, 20, 30, 40])],
        [run(reverse_phases, [40, 30, 20, 10])] * 2,
    )
    invalid = pm.classify_firmware_blockwrite_phase_crossover(
        [run(forward_phases, [10, 20, 30, 40], qualified=False)] * 2,
        [run(reverse_phases, [40, 30, 20, 10])] * 2,
    )

    assert nondeterministic == {"qualified": False, "reason": "nondeterministic"}
    assert invalid == {"qualified": False, "reason": "invalid_run"}


@pytest.mark.parametrize(
    "forward_runs",
    [
        None,
        [{"qualified": True}] * 2,
        [{"qualified": True, "intervals": [{"phase": 0, "array_cycles": 10}]}] * 2,
        [{
            "qualified": True,
            "intervals": [
                {"phase": 0, "array_cycles": 10},
                {"phase": 0, "array_cycles": 20},
            ],
        }] * 2,
        [{
            "qualified": True,
            "intervals": [
                {"phase": 0, "array_cycles": 0},
                {"phase": 20, "array_cycles": 20},
            ],
        }] * 2,
    ],
)
def test_firmware_blockwrite_phase_crossover_fails_closed_on_malformed_runs(
    forward_runs,
):
    reverse = {
        "qualified": True,
        "intervals": [
            {"phase": 20, "array_cycles": 20},
            {"phase": 0, "array_cycles": 10},
        ],
    }

    result = pm.classify_firmware_blockwrite_phase_crossover(
        forward_runs, [reverse, reverse],
    )

    assert result == {"qualified": False, "reason": "invalid_run"}


@pytest.mark.parametrize(
    "treatment_target,reason,delta",
    [(248, "history_sensitive", 16), (232, "history_invariant", 0)],
)
def test_classifies_firmware_blockwrite_history_crossover(
    treatment_target, reason, delta,
):
    def run(predecessor, target):
        return {
            "qualified": True,
            "intervals": [
                {"phase": 40, "array_cycles": predecessor},
                {"phase": 44, "array_cycles": target},
            ],
        }

    result = pm.classify_firmware_blockwrite_history_crossover(
        [run(246, 232)] * 2,
        [run(246, treatment_target)] * 2,
        target_phase=44,
    )

    assert result == {
        "qualified": True,
        "reason": reason,
        "target_phase": 44,
        "control_intervals": [
            {"phase": 40, "array_cycles": 246},
            {"phase": 44, "array_cycles": 232},
        ],
        "treatment_intervals": [
            {"phase": 40, "array_cycles": 246},
            {"phase": 44, "array_cycles": treatment_target},
        ],
        "control_target_cycles": 232,
        "treatment_target_cycles": treatment_target,
        "target_delta_cycles": delta,
    }


def test_firmware_blockwrite_history_crossover_rejects_contaminated_control():
    def run(predecessor, target):
        return {
            "qualified": True,
            "intervals": [
                {"phase": 40, "array_cycles": predecessor},
                {"phase": 44, "array_cycles": target},
            ],
        }

    result = pm.classify_firmware_blockwrite_history_crossover(
        [run(246, 232)] * 2,
        [run(247, 248)] * 2,
        target_phase=44,
    )

    assert result == {
        "qualified": False,
        "reason": "control_window_changed",
        "mismatches": [
            {"phase": 40, "control_cycles": 246, "treatment_cycles": 247},
        ],
    }


def test_firmware_blockwrite_history_crossover_fails_closed():
    control = {
        "qualified": True,
        "intervals": [
            {"phase": 40, "array_cycles": 246},
            {"phase": 44, "array_cycles": 232},
        ],
    }
    changed = {
        "qualified": True,
        "intervals": [
            {"phase": 40, "array_cycles": 246},
            {"phase": 44, "array_cycles": 233},
        ],
    }
    wrong_phases = {
        "qualified": True,
        "intervals": [
            {"phase": 40, "array_cycles": 246},
            {"phase": 48, "array_cycles": 248},
        ],
    }

    nondeterministic = pm.classify_firmware_blockwrite_history_crossover(
        [control, changed], [control, control], target_phase=44,
    )
    invalid = pm.classify_firmware_blockwrite_history_crossover(
        [control, control], [wrong_phases, wrong_phases], target_phase=44,
    )

    assert nondeterministic == {"qualified": False, "reason": "nondeterministic"}
    assert invalid == {"qualified": False, "reason": "invalid_run"}


@pytest.mark.parametrize(
    "targets,reason,qualified",
    [
        ((232, 248, 248), "saturating_hot_cold", True),
        ((232, 248, 232), "periodic_temporal", True),
        ((232, 248, 264), "accumulating_linear", True),
        ((232, 248, 249), "unclassified", False),
    ],
)
def test_classifies_firmware_blockwrite_history_depth(targets, reason, qualified):
    def run(target):
        return {
            "qualified": True,
            "intervals": [
                {"phase": 40, "array_cycles": 246},
                {"phase": 44, "array_cycles": target},
            ],
        }

    result = pm.classify_firmware_blockwrite_history_depth(
        *([run(target)] * 2 for target in targets), target_phase=44,
    )

    assert result["qualified"] is qualified
    assert result["reason"] == reason
    assert result["target_cycles"] == {
        "A": targets[0], "B": targets[1], "C": targets[2],
    }


def test_firmware_blockwrite_history_depth_rejects_changed_predecessor():
    def run(predecessor, target):
        return {
            "qualified": True,
            "intervals": [
                {"phase": 40, "array_cycles": predecessor},
                {"phase": 44, "array_cycles": target},
            ],
        }

    result = pm.classify_firmware_blockwrite_history_depth(
        [run(246, 232)] * 2,
        [run(247, 248)] * 2,
        [run(246, 248)] * 2,
        target_phase=44,
    )

    assert result == {
        "qualified": False,
        "reason": "control_window_changed",
        "mismatches": [
            {"phase": 40, "cycles": {"A": 246, "B": 247, "C": 246}},
        ],
    }


def test_firmware_blockwrite_history_depth_fails_closed_on_nonrepeat():
    def run(target):
        return {
            "qualified": True,
            "intervals": [
                {"phase": 40, "array_cycles": 246},
                {"phase": 44, "array_cycles": target},
            ],
        }

    result = pm.classify_firmware_blockwrite_history_depth(
        [run(232), run(233)], [run(248)] * 2, [run(248)] * 2,
        target_phase=44,
    )

    assert result == {"qualified": False, "reason": "nondeterministic"}


@pytest.mark.parametrize(
    "path_target,reason,qualified",
    [
        (232, "noop_opcode_path_sensitive", True),
        (248, "record_path_invariant", True),
        (249, "unclassified", False),
    ],
)
def test_classifies_firmware_blockwrite_noop_preempt_path(
    path_target, reason, qualified,
):
    def run(target):
        return {
            "qualified": True,
            "intervals": [
                {"phase": 40, "array_cycles": 246},
                {"phase": 44, "array_cycles": target},
            ],
        }

    result = pm.classify_firmware_blockwrite_noop_preempt_path(
        [run(232)] * 2,
        [run(248)] * 2,
        [run(path_target)] * 2,
        target_phase=44,
    )

    assert result["qualified"] is qualified
    assert result["reason"] == reason
    assert result["target_cycles"] == {
        "A": 232, "B": 248, "C": path_target,
    }


@pytest.mark.parametrize(
    "predecessor,a_target,b_target,target_phase",
    [
        (245, 232, 248, 44),
        (246, 233, 248, 44),
        (246, 232, 247, 44),
        (246, 232, 248, 48),
    ],
)
def test_noop_preempt_path_classifier_requires_exact_controls(
    predecessor, a_target, b_target, target_phase,
):
    def run(target):
        return {
            "qualified": True,
            "intervals": [
                {"phase": 40, "array_cycles": predecessor},
                {"phase": target_phase, "array_cycles": target},
            ],
        }

    result = pm.classify_firmware_blockwrite_noop_preempt_path(
        [run(a_target)] * 2,
        [run(b_target)] * 2,
        [run(232)] * 2,
        target_phase=target_phase,
    )

    assert result["qualified"] is False
    assert result["reason"] == "control_mismatch"


@pytest.mark.parametrize(
    "treatment_target,reason",
    [
        (248, "blockwrite_reprime_invariant"),
        (232, "blockwrite_reprime_matches_cold"),
        (249, "blockwrite_reprime_changes_target"),
    ],
)
def test_classifies_firmware_blockwrite_reprime_order(
    treatment_target, reason,
):
    def run(target):
        return {
            "qualified": True,
            "intervals": [
                {"phase": 40, "array_cycles": 246},
                {"phase": 44, "array_cycles": target},
            ],
        }

    result = pm.classify_firmware_blockwrite_reprime_order(
        [run(248)] * 2,
        [run(treatment_target)] * 2,
        target_phase=44,
    )

    assert result["qualified"] is True
    assert result["reason"] == reason
    assert result["target_cycles"] == {
        "A": 248,
        "B": treatment_target,
    }


@pytest.mark.parametrize(
    "a_predecessor,b_predecessor,a_target,target_phase",
    [
        (245, 246, 248, 44),
        (246, 247, 248, 44),
        (246, 246, 247, 44),
        (246, 246, 248, 48),
    ],
)
def test_blockwrite_reprime_order_classifier_requires_exact_controls(
    a_predecessor, b_predecessor, a_target, target_phase,
):
    def run(predecessor, target):
        return {
            "qualified": True,
            "intervals": [
                {"phase": 40, "array_cycles": predecessor},
                {"phase": target_phase, "array_cycles": target},
            ],
        }

    result = pm.classify_firmware_blockwrite_reprime_order(
        [run(a_predecessor, a_target)] * 2,
        [run(b_predecessor, 248)] * 2,
        target_phase=target_phase,
    )

    assert result["qualified"] is False
    assert result["reason"] == "control_mismatch"


def test_blockwrite_reprime_order_classifier_fails_closed_on_nonrepeat():
    def run(target):
        return {
            "qualified": True,
            "intervals": [
                {"phase": 40, "array_cycles": 246},
                {"phase": 44, "array_cycles": target},
            ],
        }

    result = pm.classify_firmware_blockwrite_reprime_order(
        [run(248)] * 2,
        [run(232), run(233)],
        target_phase=44,
    )

    assert result == {"qualified": False, "reason": "nondeterministic"}


@pytest.mark.parametrize(
    "a_target,b_target,reason",
    [
        (248, 232, "write32_recency_matches_blockwrite"),
        (248, 248, "write32_recency_invariant"),
        (249, 231, "write32_recency_other"),
    ],
)
def test_classifies_firmware_write32_recency(a_target, b_target, reason):
    def run(target):
        return {
            "qualified": True,
            "intervals": [
                {"phase": 40, "array_cycles": 246},
                {"phase": 44, "array_cycles": target},
            ],
        }

    result = pm.classify_firmware_write32_recency(
        [run(a_target)] * 2,
        [run(b_target)] * 2,
        target_phase=44,
    )

    assert result["qualified"] is True
    assert result["reason"] == reason
    assert result["target_cycles"] == {"A": a_target, "B": b_target}


@pytest.mark.parametrize(
    "a_predecessor,b_predecessor,target_phase",
    [
        (245, 246, 44),
        (246, 247, 44),
        (246, 246, 48),
    ],
)
def test_write32_recency_classifier_requires_exact_controls(
    a_predecessor, b_predecessor, target_phase,
):
    def run(predecessor, target):
        return {
            "qualified": True,
            "intervals": [
                {"phase": 40, "array_cycles": predecessor},
                {"phase": target_phase, "array_cycles": target},
            ],
        }

    result = pm.classify_firmware_write32_recency(
        [run(a_predecessor, 248)] * 2,
        [run(b_predecessor, 232)] * 2,
        target_phase=target_phase,
    )

    assert result["qualified"] is False
    assert result["reason"] == "control_mismatch"


def test_write32_recency_classifier_fails_closed_on_nonrepeat():
    def run(target):
        return {
            "qualified": True,
            "intervals": [
                {"phase": 40, "array_cycles": 246},
                {"phase": 44, "array_cycles": target},
            ],
        }

    result = pm.classify_firmware_write32_recency(
        [run(248)] * 2,
        [run(232), run(233)],
        target_phase=44,
    )

    assert result == {"qualified": False, "reason": "nondeterministic"}


def test_firmware_blockwrite_timeline_rejects_source_bd14_use(tmp_path):
    bd14 = address(0, 0, 0x1C1C0)
    occupied = struct.pack("<IIII8I", 1, 0, bd14, 48, *([0] * 8))

    with pytest.raises(ValueError, match="BD14 is already used"):
        pm.instrument_firmware_blockwrite_timeline(
            firmware_timeline_fixture_insts((occupied,)), register_db(tmp_path),
            SHIM_EVENT_IDS, (1, 2, 4, 8),
        )


def test_firmware_blockwrite_phase_crossover_rejects_source_bd14_use(tmp_path):
    bd14 = address(0, 0, 0x1C1C0)
    occupied = struct.pack("<IIII8I", 1, 0, bd14, 48, *([0] * 8))

    with pytest.raises(ValueError, match="BD14 is already used"):
        pm.instrument_firmware_blockwrite_phase_crossover(
            firmware_timeline_fixture_insts((occupied,)), register_db(tmp_path),
            SHIM_EVENT_IDS, (0, 20, 44, 56),
        )


@pytest.mark.parametrize("opcode", [0, 3, 0x81])
def test_firmware_blockwrite_timeline_rejects_other_source_bd14_writes(
    tmp_path, opcode,
):
    bd14 = address(0, 0, 0x1C1C0)
    occupied = {
        0: write32(bd14, 0),
        3: struct.pack("<IIQIII", 3, 0, bd14, 0, 0xFFFFFFFF, 28),
        0x81: struct.pack("<12I", 0x81, 48, 0, 0, 0, 0, bd14, 0, 0, 0, 0, 0),
    }[opcode]

    with pytest.raises(ValueError, match="BD14 is already used"):
        pm.instrument_firmware_blockwrite_timeline(
            firmware_timeline_fixture_insts((occupied,)), register_db(tmp_path),
            SHIM_EVENT_IDS, (1, 2, 4, 8),
        )


def test_firmware_blockwrite_timeline_rejects_queued_bd14(tmp_path):
    queue = write32(address(0, 0, 0x1C204), 0x8000000E)

    with pytest.raises(ValueError, match="BD14 is already queued"):
        pm.instrument_firmware_blockwrite_timeline(
            firmware_timeline_fixture_insts((queue,)), register_db(tmp_path),
            SHIM_EVENT_IDS, (1, 2, 4, 8),
        )


@pytest.mark.parametrize("blocks", [(), (0,), (9,), (True,), (1, 0)])
def test_firmware_blockwrite_timeline_rejects_invalid_word_counts(tmp_path, blocks):
    with pytest.raises(ValueError):
        pm.instrument_firmware_blockwrite_timeline(
            firmware_timeline_fixture_insts(), register_db(tmp_path),
            SHIM_EVENT_IDS, blocks,
        )


def test_firmware_blockwrite_timeline_derives_bd_word_capacity(tmp_path):
    with pytest.raises(ValueError, match="1 through 4"):
        pm.instrument_firmware_blockwrite_timeline(
            firmware_timeline_fixture_insts(), register_db(tmp_path, bd_words=4),
            SHIM_EVENT_IDS, (5,),
        )


@pytest.mark.parametrize(
    "blocks", [(), (1, -1), (True,), (1, 0), (0x40000000,)],
)
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


def test_firmware_clock_timeline_rejects_existing_shim_counter_use(tmp_path):
    data = bytearray(firmware_timeline_fixture_insts())
    data.extend(write32(address(0, 0, 0x31000), 1))
    struct.pack_into("<I", data, 8, struct.unpack_from("<I", data, 8)[0] + 1)
    struct.pack_into("<I", data, 12, len(data))

    with pytest.raises(ValueError, match="shim performance counter"):
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
        "slot_names": {"shim": ["DMA_START", "DMA_FINISH", "NONE"]},
        "events": [
            {"pkt_type": 2, "row": 0, "slot": 2, "name": "NONE"},
            {"pkt_type": 2, "row": 1, "slot": 2, "name": "OTHER"},
            {"pkt_type": 0, "row": 0, "slot": 2, "name": "CORE"},
        ],
    }

    pm.relabel_firmware_clock_timeline_events(document)

    assert document["slot_names"]["shim"][2] == "USER_EVENT_0"
    assert document["events"][0]["name"] == "USER_EVENT_0"
    assert document["events"][1]["name"] == "OTHER"
    assert document["events"][2]["name"] == "CORE"


def test_relabels_firmware_blockwrite_phase_marker_slots():
    document = {
        "slot_names": {
            "shim": ["DMA_START", "DMA_FINISH", "NONE", "NONE"],
        },
        "events": [
            {"pkt_type": 2, "row": 0, "slot": 2, "name": "NONE"},
            {"pkt_type": 2, "row": 0, "slot": 3, "name": "NONE"},
            {"pkt_type": 2, "row": 1, "slot": 3, "name": "OTHER"},
            {"pkt_type": 0, "row": 0, "slot": 3, "name": "CORE"},
        ],
    }

    pm.relabel_firmware_blockwrite_phase_events(document)

    assert document["slot_names"]["shim"][2:] == [
        "USER_EVENT_0", "USER_EVENT_1",
    ]
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
