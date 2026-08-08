#!/usr/bin/env python3
"""Phoenix PM-fault clock comparator helpers.

The hardware campaign is intentionally built from the existing transaction
patcher and trace runner. This module owns only the fixture-specific
instrumentation and falsifiable threshold classification.
"""

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys


_PATCHER = Path(__file__).with_name("trace-patch-events.py")
_SPEC = importlib.util.spec_from_file_location("trace_patch_events", _PATCHER)
patcher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(patcher)

_CORE_EVENTS = [
    "PERF_CNT_2",
    "INSTR_EVENT_0",
    "INSTR_EVENT_1",
    "PM_ADDRESS_OUT_OF_RANGE",
    "PERF_CNT_3",
    "INSTR_LOCK_RELEASE_REQ",
    "INSTR_LOCK_ACQUIRE_REQ",
    "LOCK_STALL",
]
_DEFAULT_QOS_FPS = (1000, 1800, 2300, 3000)
_PINNED_PHOENIX_FIRMWARE_SHA256 = (
    "d13ff9fb95c6cea40213fa69e5a346552"
    "9f00bb67c0984d62343c6e31808fb9e"
)
_PHOENIX_NPI_BASE = 0xAC000000
_PHOENIX_TRANSACTION_ARRAY_BASE = 0x84000000

_AIEML_NPI_MACROS = (
    "XAIEML_NPI_PCSR_UNLOCK_CODE",
    "XAIEML_NPI_PCSR_LOCK",
    "XAIEML_NPI_PROT_REG_CNTR",
    "XAIEML_NPI_PROT_REG_CNTR_EN_MSK",
    "XAIEML_NPI_PROT_REG_CNTR_EN_LSB",
    "XAIEML_NPI_PROT_REG_CNTR_FIRSTCOL_MSK",
    "XAIEML_NPI_PROT_REG_CNTR_FIRSTCOL_LSB",
    "XAIEML_NPI_PROT_REG_CNTR_LASTCOL_MSK",
    "XAIEML_NPI_PROT_REG_CNTR_LASTCOL_LSB",
)

_AIEML_TRACE_EVENT_MACROS = (
    "XAIEML_EVENTS_CORE_USER_EVENT_0",
    "XAIEML_EVENTS_CORE_USER_EVENT_1",
    "XAIEML_EVENTS_PL_BROADCAST_A_13",
    "XAIEML_EVENTS_PL_BROADCAST_A_14",
    "XAIEML_EVENTS_PL_USER_EVENT_0",
)


def instrument_post_tct_noops(data: bytes, count: int) -> bytes:
    """Keep the firmware command open with finite management-only work."""
    return patcher.insert_noops_after_last_tct(data, count)


def _derive_aieml_macros(
    source: Path, names: tuple[str, ...], kind: str,
) -> dict[str, int]:
    try:
        text = source.read_text()
    except OSError as error:
        raise ValueError(
            f"cannot read aie-rt {kind} source {source}: {error}"
        ) from error
    values = {}
    for name, literal in re.findall(
        r"^\s*#define\s+([A-Z0-9_]+)\s+"
        r"(0[xX][0-9A-Fa-f]+|[0-9]+)(?:[uUlL]+)?(?:\s|$)",
        text,
        re.MULTILINE,
    ):
        if name in names:
            if name in values:
                raise ValueError(f"duplicate aie-rt {kind} macro {name}")
            values[name] = int(literal, 0)
    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError(
            f"missing aie-rt {kind} macro(s): " + ", ".join(missing)
        )
    return values


def _derive_aieml_npi(source: Path) -> dict[str, int]:
    """Read the named AIE2 NPI fields used by aie-rt's protection path."""
    values = _derive_aieml_macros(source, _AIEML_NPI_MACROS, "NPI")
    if any(value > 0xFFFFFFFF for value in values.values()):
        raise ValueError("aie-rt NPI macro does not fit in 32 bits")
    return values


def _derive_aieml_trace_events(source: Path) -> dict[str, int]:
    """Read the named AIE2 events that define the causal trace shutdown."""
    values = _derive_aieml_macros(
        source, _AIEML_TRACE_EVENT_MACROS, "event",
    )
    if any(value >= 1 << 7 for value in values.values()):
        raise ValueError("aie-rt trace-stop event exceeds its 7-bit field")
    return values


def prepare_real_column_gate_trace(
    data: bytes,
    register_db: Path,
    aieml_events_source: Path,
    *,
    col: int = 0,
    core_row: int = 2,
    shim_row: int = 0,
) -> bytes:
    """Bound the periodic witness to the protected lifecycle at its source."""
    events = _derive_aieml_trace_events(aieml_events_source)
    core_start = events["XAIEML_EVENTS_CORE_USER_EVENT_1"]
    core_stop = events["XAIEML_EVENTS_CORE_USER_EVENT_0"]
    shim_start = events["XAIEML_EVENTS_PL_BROADCAST_A_13"]
    shim_stop = events["XAIEML_EVENTS_PL_BROADCAST_A_14"]
    old_shim_trigger = events["XAIEML_EVENTS_PL_USER_EVENT_0"]

    controls = {
        (core_row, "core"): patcher._TRACE_CONTROL0_REGS["core"],
        (shim_row, "shim"): patcher._TRACE_CONTROL0_REGS["shim"],
    }
    for (row, tile_type), register_offset in controls.items():
        target = patcher._npu_address(col, row, register_offset)
        matches = [
            value for _, address, value in patcher._walk_write32(data)
            if address == target
        ]
        if len(matches) != 1 or (matches[0] >> 24) & 0x7F:
            raise ValueError(
                f"real-gate {tile_type} trace is not the pinned open-ended session"
            )

    data, _ = patcher.patch_register_fields(
        data, col, core_row, "core", "Performance_Control1",
        {
            "Cnt3_Start_Event": core_start,
            "Cnt3_Stop_Event": core_stop,
        },
        register_db,
    )
    data, _ = patcher.patch_trace_control(
        data, col, core_row, "core", stop_event=core_stop,
    )
    data, _ = patcher.patch_trace_control(
        data, col, shim_row, "shim", stop_event=shim_stop,
    )

    broadcast14_offset = patcher._register_offset(
        register_db, "core", "Event_Broadcast14",
    )
    broadcast14_address = patcher._npu_address(
        col, core_row, broadcast14_offset,
    )
    if any(
        address == broadcast14_address
        for _, address, _ in patcher._walk_write32(data)
    ):
        raise ValueError("core Event_Broadcast14 is already configured")

    generate_offset = patcher._register_offset(
        register_db, "shim", "Event_Generate",
    )
    generate_address = patcher._npu_address(col, shim_row, generate_offset)
    matches = [
        offset for offset, address, value in patcher._walk_write32(data)
        if address == generate_address and value == old_shim_trigger
    ]
    if len(matches) != 1:
        raise ValueError(
            "real-gate stream does not contain exactly one standard stop trigger"
        )
    data, _ = patcher.patch_register_fields(
        data, col, shim_row, "shim", "Performance_Ctrl0",
        {"Cnt0_Start_Event": shim_start}, register_db,
    )
    offset = matches[0]
    if patcher._instruction_length(data, offset) != 24:
        raise ValueError("real-gate stop trigger is not a Write32 record")

    result = bytearray(data)
    result[offset:offset + 24] = b"\x05\x00\x00\x00"
    struct.pack_into("<I", result, 12, len(result))
    return patcher.insert_register_write_after(
        bytes(result), col, core_row, "core", "Event_Broadcast14",
        core_stop, "Event_Broadcast13", register_db,
    )


def _derived_field(value: int, lsb: int, mask: int, name: str) -> int:
    shifted_mask = mask >> lsb if 0 <= lsb < 32 else 0
    if (
        mask > 0xFFFFFFFF
        or shifted_mask == 0
        or shifted_mask & (shifted_mask + 1)
        or mask != shifted_mask << lsb
        or value < 0
        or value > shifted_mask
    ):
        raise ValueError(f"invalid aie-rt field {name}")
    return (value << lsb) & mask


def _hex32(value: int) -> str:
    if not 0 <= value <= 0xFFFFFFFF:
        raise ValueError(f"value {value:#x} does not fit in 32 bits")
    return f"0x{value:08x}"


def build_real_column_gate_pair(
    data: bytes,
    register_db: Path,
    aie_rt_source: Path,
    *,
    expected_input_sha256: str,
    firmware_sha256: str,
    physical_start_col: int,
    num_col: int,
) -> dict:
    """Build the pinned one-word Phoenix column-gate control/treatment pair."""
    input_sha256 = hashlib.sha256(data).hexdigest()
    if input_sha256 != expected_input_sha256:
        raise ValueError("input instruction hash does not match its pin")
    if firmware_sha256 != _PINNED_PHOENIX_FIRMWARE_SHA256:
        raise ValueError("firmware hash is not pinned Phoenix 1.5.5.391")
    if (physical_start_col, num_col) != (1, 1):
        raise ValueError("real column-gate witness requires physical placement 1:1")

    npi = _derive_aieml_npi(aie_rt_source)
    transaction_base = (physical_start_col + 6) << 25
    if transaction_base > 0xFFFFFFFF:
        raise ValueError("transaction base does not fit in 32 bits")

    clock_register = patcher._register_definition(
        register_db, "shim", "Column_Clock_Control",
    )
    try:
        clock_offset = int(clock_register["offset"], 0)
        clock_field = next(
            field for field in clock_register["bit_fields"]
            if field["name"] == "Clock_Buffer_Enable"
        )
        clock_low, clock_high = clock_field["bit_range"]
    except (KeyError, TypeError, ValueError, StopIteration) as error:
        raise ValueError(
            f"cannot derive Column_Clock_Control.Clock_Buffer_Enable: {error}"
        ) from error
    if not (
        isinstance(clock_low, int)
        and isinstance(clock_high, int)
        and 0 <= clock_low <= clock_high < 32
    ):
        raise ValueError(
            "Column_Clock_Control.Clock_Buffer_Enable is outside its 32-bit register"
        )
    if clock_low != clock_high:
        raise ValueError(
            "Column_Clock_Control.Clock_Buffer_Enable must remain a single bit"
        )
    clock_mask = ((1 << (clock_high - clock_low + 1)) - 1) << clock_low
    clock_enabled = patcher._set_register_fields(
        clock_register, 0, {"Clock_Buffer_Enable": 1},
    )
    clock_disabled = patcher._set_register_fields(
        clock_register, 0, {"Clock_Buffer_Enable": 0},
    )

    lock_absolute = _PHOENIX_NPI_BASE + npi["XAIEML_NPI_PCSR_LOCK"]
    protection_absolute = (
        _PHOENIX_NPI_BASE + npi["XAIEML_NPI_PROT_REG_CNTR"]
    )
    clock_firmware_target = (
        _PHOENIX_TRANSACTION_ARRAY_BASE
        + (physical_start_col << 25)
        + clock_offset
    )
    if any(target > 0xFFFFFFFF for target in (
        lock_absolute, protection_absolute, clock_firmware_target,
    )):
        raise ValueError("allowlisted target does not fit in 32 bits")
    if len({lock_absolute, protection_absolute, clock_firmware_target}) != 3:
        raise ValueError("real-gate sequence requires three distinct allowlisted targets")

    def relative_npi(absolute: int) -> int:
        if absolute < transaction_base:
            raise ValueError("NPI address precedes the transaction base")
        relative = absolute - transaction_base
        if relative > 0xFFFFFFFF or relative & 3:
            raise ValueError("NPI transaction offset is invalid")
        return relative

    lock_offset = relative_npi(lock_absolute)
    protection_offset = relative_npi(protection_absolute)
    if clock_offset > 0xFFFFFFFF or clock_offset & 3:
        raise ValueError("column-clock transaction offset is invalid")
    if transaction_base + clock_offset > 0xFFFFFFFF:
        raise ValueError("column-clock pre-MMU address wraps")
    allowed_offsets = {
        lock_offset: lock_absolute,
        protection_offset: protection_absolute,
        clock_offset: clock_firmware_target,
    }
    if len(allowed_offsets) != 3:
        raise ValueError("real-gate sequence requires three distinct encoded offsets")

    enable_mask = npi["XAIEML_NPI_PROT_REG_CNTR_EN_MSK"]
    first_mask = npi["XAIEML_NPI_PROT_REG_CNTR_FIRSTCOL_MSK"]
    last_mask = npi["XAIEML_NPI_PROT_REG_CNTR_LASTCOL_MSK"]
    if enable_mask & first_mask or enable_mask & last_mask or first_mask & last_mask:
        raise ValueError("aie-rt protection fields overlap")

    def protection_value(enabled: int) -> int:
        last_col = physical_start_col + num_col - 1
        return (
            _derived_field(
                enabled,
                npi["XAIEML_NPI_PROT_REG_CNTR_EN_LSB"],
                enable_mask,
                "enable",
            )
            | _derived_field(
                physical_start_col,
                npi["XAIEML_NPI_PROT_REG_CNTR_FIRSTCOL_LSB"],
                first_mask,
                "first column",
            )
            | _derived_field(
                last_col,
                npi["XAIEML_NPI_PROT_REG_CNTR_LASTCOL_LSB"],
                last_mask,
                "last column",
            )
        )

    protected = protection_value(1)
    unprotected = protection_value(0)
    unlock = npi["XAIEML_NPI_PCSR_UNLOCK_CODE"]
    allowlist = {lock_absolute, protection_absolute, clock_firmware_target}

    def build_arm(first_clock_value: int) -> tuple[bytes, list[dict]]:
        records = []
        operations = []

        def emit(
            phase: str,
            opcode: str,
            encoded_offset: int | None = None,
            value: int | None = None,
            mask: int | None = None,
            expected_firmware_target: int | None = None,
        ) -> None:
            if opcode == "noop":
                record = b"\x05\x00\x00\x00"
                operation = {"phase": phase, "opcode": opcode}
            else:
                if encoded_offset is None or value is None or expected_firmware_target is None:
                    raise AssertionError("register operation is incomplete")
                if encoded_offset > 0xFFFFFFFF or encoded_offset & 3:
                    raise ValueError("register offset is not an aligned 32-bit value")
                if (
                    expected_firmware_target not in allowlist
                    or allowed_offsets.get(encoded_offset) != expected_firmware_target
                ):
                    raise ValueError("operation escaped the real-gate allowlist")
                pre_mmu = transaction_base + encoded_offset
                if pre_mmu > 0xFFFFFFFF:
                    raise ValueError("operation pre-MMU address wraps")
                opcode_number = {"write32": 0, "mask_write": 3, "mask_poll": 4}[opcode]
                if mask is None:
                    record = struct.pack("<IIQII", opcode_number, 0, encoded_offset, value, 24)
                else:
                    record = struct.pack(
                        "<IIQIII", opcode_number, 0, encoded_offset, value, mask, 28,
                    )
                operation = {
                    "phase": phase,
                    "opcode": opcode,
                    "encoded_offset": _hex32(encoded_offset),
                    "reg_offset_high": "0x00000000",
                    "value": _hex32(value),
                    "pre_mmu_effective": _hex32(pre_mmu),
                    "expected_firmware_target": _hex32(expected_firmware_target),
                }
                if mask is not None:
                    operation["mask"] = _hex32(mask)
            operation["index"] = len(operations)
            records.append(record)
            operations.append(operation)

        def transition(phase: str, clock_value: int) -> None:
            def write(offset, value, target):
                emit(
                    phase, "write32", offset, value,
                    expected_firmware_target=target,
                )

            def poll(offset, target):
                emit(phase, "mask_poll", offset, 0, 0, target)

            write(lock_offset, unlock, lock_absolute)
            poll(lock_offset, lock_absolute)
            write(protection_offset, protected, protection_absolute)
            poll(protection_offset, protection_absolute)
            write(lock_offset, 0, lock_absolute)
            poll(lock_offset, lock_absolute)
            emit(
                phase, "mask_write", clock_offset, clock_value,
                clock_mask, clock_firmware_target,
            )
            write(lock_offset, unlock, lock_absolute)
            poll(lock_offset, lock_absolute)
            write(protection_offset, unprotected, protection_absolute)
            poll(protection_offset, protection_absolute)
            write(lock_offset, 0, lock_absolute)
            poll(lock_offset, lock_absolute)

        transition("gate_transition", first_clock_value)
        for _ in range(256):
            emit("gate_dwell", "noop")
        transition("restore_transition", clock_enabled)
        for _ in range(256):
            emit("restore_dwell", "noop")
        return patcher.insert_records_after_last_tct(data, b"".join(records)), operations

    control, control_ops = build_arm(clock_enabled)
    treatment, treatment_ops = build_arm(clock_disabled)
    differing_words = [
        offset for offset in range(0, len(control), 4)
        if control[offset:offset + 4] != treatment[offset:offset + 4]
    ]
    if len(differing_words) != 1:
        raise ValueError(
            f"control/treatment differ in {len(differing_words)} words, expected one"
        )
    diff_offset = differing_words[0]
    control_word = struct.unpack_from("<I", control, diff_offset)[0]
    treatment_word = struct.unpack_from("<I", treatment, diff_offset)[0]
    if (control_word, treatment_word) != (clock_enabled, clock_disabled):
        raise ValueError("one-word difference is not the first clock transition")

    manifest = {
        "schema_version": 1,
        "target": "phoenix_npu1",
        "firmware": {
            "version": "1.5.5.391",
            "sha256": firmware_sha256,
        },
        "input": {
            "sha256": input_sha256,
            "expected_sha256": expected_input_sha256,
        },
        "placement": {"start_col": physical_start_col, "num_col": num_col},
        "transaction_base": _hex32(transaction_base),
        "npi_base": _hex32(_PHOENIX_NPI_BASE),
        "transaction_array_base": _hex32(_PHOENIX_TRANSACTION_ARRAY_BASE),
        "targets": {
            "npi_lock": _hex32(lock_absolute),
            "npi_protection": _hex32(protection_absolute),
            "column_clock": _hex32(clock_firmware_target),
        },
        "sources": {
            "aie_rt": {
                "path": str(aie_rt_source.resolve()),
                "sha256": hashlib.sha256(aie_rt_source.read_bytes()).hexdigest(),
                "macros": {name: _hex32(npi[name]) for name in _AIEML_NPI_MACROS},
            },
            "am025": {
                "path": str(register_db.resolve()),
                "sha256": hashlib.sha256(register_db.read_bytes()).hexdigest(),
                "register": "Column_Clock_Control",
                "field": "Clock_Buffer_Enable",
                "offset": _hex32(clock_offset),
                "mask": _hex32(clock_mask),
            },
        },
        "arms": {
            "control": {
                "sha256": hashlib.sha256(control).hexdigest(),
                "operations": control_ops,
            },
            "treatment": {
                "sha256": hashlib.sha256(treatment).hexdigest(),
                "operations": treatment_ops,
            },
        },
        "one_word_diff": {
            "byte_offset": diff_offset,
            "control": _hex32(control_word),
            "treatment": _hex32(treatment_word),
        },
    }
    return {"control": control, "treatment": treatment, "manifest": manifest}


def instrument_comparator(
    data: bytes,
    threshold: int,
    register_db: Path,
    event_ids: dict[str, int],
    col: int = 0,
    row: int = 2,
) -> bytes:
    """Configure counter 3 as an event-65-started gate comparator."""
    if not 0 < threshold <= 0xFFFFFFFF:
        raise ValueError("threshold must fit in a nonzero 32-bit counter")
    events = [event_ids[name] for name in _CORE_EVENTS]
    data, _ = patcher.patch_events(data, col, row, "core", events)
    data, _ = patcher.patch_trace_control(
        data, col, row, "core", stop_event=0,
    )
    data, _ = patcher.patch_register_fields(
        data, col, row, "core", "Performance_Control1",
        {
            "Cnt3_Start_Event": event_ids["PM_ADDRESS_OUT_OF_RANGE"],
            "Cnt3_Stop_Event": 0,
        },
        register_db,
    )
    data, _ = patcher.patch_register_fields(
        data, col, row, "core", "Performance_Control2",
        {"Cnt3_Reset_Event": 0},
        register_db,
    )
    return patcher.insert_register_write_after(
        data, col, row, "core", "Performance_Counter3_Event_Value",
        threshold, "Performance_Control2", register_db,
    )


def instrument_shim_witness(
    data: bytes,
    register_db: Path,
    core_event_ids: dict[str, int],
    shim_event_ids: dict[str, int],
    threshold: int = 64,
    channel: int = 13,
    enable_transport: bool = True,
    col: int = 0,
    core_row: int = 2,
    shim_row: int = 0,
) -> bytes:
    """Add the qualified same-shim-domain liveness witness."""
    if not 0 < threshold <= 0xFFFFFFFF:
        raise ValueError("threshold must fit in a nonzero 32-bit counter")
    if channel in {0, 1, 2, 14, 15} or not 0 <= channel < 16:
        raise ValueError(f"broadcast channel {channel} is reserved or invalid")

    core_broadcast = f"Event_Broadcast{channel}"
    shim_broadcast = f"Event_Broadcast{channel}_A"
    offsets = {
        "core": patcher._register_offset(
            register_db, "core", core_broadcast,
        ),
        "memmod": patcher._register_offset(
            register_db, "memmod", core_broadcast,
        ),
        "memtile": patcher._register_offset(
            register_db, "memtile", core_broadcast,
        ),
        "shim": patcher._register_offset(
            register_db, "shim", shim_broadcast,
        ),
    }
    for _, address, _ in patcher._walk_write32(data):
        row = (address >> 20) & 0x1F
        offset = address & 0xFFFFF
        occupied = (
            (row == shim_row and offset == offsets["shim"])
            or (row == 1 and offset == offsets["memtile"])
            or (row >= core_row and offset in {offsets["core"], offsets["memmod"]})
        )
        if occupied:
            raise ValueError(
                f"broadcast channel {channel} is already configured"
            )

    shim_events = [
        shim_event_ids["DMA_S2MM_0_START_TASK"],
        shim_event_ids["DMA_S2MM_0_FINISHED_TASK"],
        shim_event_ids[f"BROADCAST_A_{channel}"],
        shim_event_ids["PERF_CNT_0"],
    ]
    data, _ = patcher.patch_events(
        data, col, shim_row, "shim", shim_events,
    )
    data, _ = patcher.patch_trace_control(
        data, col, shim_row, "shim", stop_event=shim_event_ids["NONE"],
    )

    def encoded(register_name, fields):
        register = patcher._register_definition(
            register_db, "shim", register_name,
        )
        return patcher._set_register_fields(register, 0, fields)

    counter_control = encoded("Performance_Ctrl0", {
        "Cnt0_Start_Event": shim_event_ids["USER_EVENT_1"],
        "Cnt0_Stop_Event": shim_event_ids["NONE"],
    })
    counter_reset = encoded("Performance_Ctrl1", {
        "Cnt0_Reset_Event": shim_event_ids["PERF_CNT_0"],
    })
    counter_value = encoded("Performance_Counter0_Event_Value", {
        "Counter_Event_Value": threshold,
    })
    data = patcher.insert_register_write_after(
        data, col, shim_row, "shim", "Performance_Ctrl0",
        counter_control, "Trace_Event0", register_db,
    )
    data = patcher.insert_register_write_after(
        data, col, shim_row, "shim", "Performance_Ctrl1",
        counter_reset, "Performance_Ctrl0", register_db,
    )
    data = patcher.insert_register_write_after(
        data, col, shim_row, "shim", "Performance_Counter0_Event_Value",
        counter_value, "Performance_Ctrl1", register_db,
    )
    if enable_transport:
        data = patcher.insert_register_write_after(
            data, col, core_row, "core", core_broadcast,
            core_event_ids["PERF_CNT_3"],
            "Performance_Counter3_Event_Value", register_db,
        )
    return data


def _constant_cadence(series: list[int]) -> int | None:
    deltas = [right - left for left, right in zip(series, series[1:])]
    if not deltas or deltas[0] <= 0 or any(delta != deltas[0] for delta in deltas):
        return None
    return deltas[0]


def core_fault_signature(
    events: list[dict], col: int = 1, row: int = 2,
) -> list[tuple[str, int]] | None:
    """Return the exact ordered post-fault core event signature."""
    core = [
        event for event in events
        if event.get("pkt_type") == 0
        and event.get("col") == col
        and event.get("row") == row
    ]
    faults = [event["ts"] for event in core
              if event.get("name") == "PM_ADDRESS_OUT_OF_RANGE"]
    if not faults:
        return None
    fault = min(faults)
    return [
        (event["name"], event["ts"] - fault)
        for event in sorted(
            (event for event in core if event["ts"] >= fault),
            key=lambda event: (event["ts"], event.get("slot", -1), event["name"]),
        )
    ]


_REAL_COLUMN_GATE_WITNESS_FIELDS = (
    "arm", "period",
    "core_before", "shim_before",
    "core_restored", "shim_restored",
    "core_stopped", "shim_stopped",
    "trace_before_core", "trace_before_shim",
    "trace_restored_core", "trace_restored_shim",
    "trace_stopped_core", "trace_stopped_shim",
)


def parse_real_column_gate_register_witness(log: str) -> dict:
    """Parse the one fixed register witness published by the driver hook."""
    marker = "PHOENIX_COLUMN_GATE_WITNESS "
    payloads = [
        line.split(marker, 1)[1].strip()
        for line in log.splitlines()
        if marker in line
    ]
    if len(payloads) != 1:
        raise ValueError("expected exactly one Phoenix column-gate witness")

    values = {}
    for token in payloads[0].split():
        key, separator, value = token.partition("=")
        if not separator or not key or key in values:
            raise ValueError("Phoenix column-gate witness is malformed")
        values[key] = value
    if tuple(values) != _REAL_COLUMN_GATE_WITNESS_FIELDS:
        raise ValueError("Phoenix column-gate witness is malformed")
    if values["arm"] not in {"control", "treatment"}:
        raise ValueError("Phoenix column-gate witness is malformed")
    if not values["period"].isdecimal() or int(values["period"]) <= 0:
        raise ValueError("Phoenix column-gate witness is malformed")
    for key in _REAL_COLUMN_GATE_WITNESS_FIELDS[2:]:
        if re.fullmatch(r"0x[0-9a-f]{8}", values[key]) is None:
            raise ValueError("Phoenix column-gate witness is malformed")

    return {
        "arm": values["arm"],
        "period": int(values["period"]),
        "timers": {
            "before_gate": {
                "core": int(values["core_before"], 16),
                "shim": int(values["shim_before"], 16),
            },
            "after_restore": {
                "core": int(values["core_restored"], 16),
                "shim": int(values["shim_restored"], 16),
            },
            "after_stop": {
                "core": int(values["core_stopped"], 16),
                "shim": int(values["shim_stopped"], 16),
            },
        },
        "trace_status": {
            "before_gate": {
                "core": int(values["trace_before_core"], 16),
                "shim": int(values["trace_before_shim"], 16),
            },
            "after_restore": {
                "core": int(values["trace_restored_core"], 16),
                "shim": int(values["trace_restored_shim"], 16),
            },
            "after_stop": {
                "core": int(values["trace_stopped_core"], 16),
                "shim": int(values["trace_stopped_shim"], 16),
            },
        },
    }


def classify_real_column_gate_register_witness(
    arm: str,
    witness: dict,
    control_witness: dict | None = None,
) -> dict:
    """Prove freeze/resume from nested core/shim timer intervals."""
    verdict = {"qualified": False, "arm": arm}

    def stop(reason):
        verdict["reason"] = reason
        return verdict

    if arm not in {"control", "treatment"} or witness.get("arm") != arm:
        return stop("witness_arm_mismatch")
    period = witness.get("period")
    timers = witness.get("timers")
    trace_status = witness.get("trace_status")
    if (
        not isinstance(period, int) or isinstance(period, bool) or period <= 0
        or not isinstance(timers, dict)
        or not isinstance(trace_status, dict)
    ):
        return stop("malformed_register_witness")
    try:
        before = timers["before_gate"]
        restored = timers["after_restore"]
        stopped = timers["after_stop"]
        raw_values = [
            point[module]
            for point in (before, restored, stopped)
            for module in ("core", "shim")
        ] + [
            point[module]
            for point in (
                trace_status["before_gate"],
                trace_status["after_restore"],
                trace_status["after_stop"],
            )
            for module in ("core", "shim")
        ]
    except (KeyError, TypeError):
        return stop("malformed_register_witness")
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        or not 0 <= value <= 0xFFFF_FFFF
        for value in raw_values
    ):
        return stop("malformed_register_witness")

    deltas = {
        "gate_core": (restored["core"] - before["core"]) & 0xFFFF_FFFF,
        "gate_shim": (restored["shim"] - before["shim"]) & 0xFFFF_FFFF,
        "resume_core": (stopped["core"] - restored["core"]) & 0xFFFF_FFFF,
        "resume_shim": (stopped["shim"] - restored["shim"]) & 0xFFFF_FFFF,
    }
    verdict.update(
        period=period,
        timer_deltas=deltas,
        trace_status=trace_status,
    )
    if any(delta == 0 or delta >= 0x8000_0000 for delta in deltas.values()):
        return stop("invalid_timer_window")
    if deltas["resume_core"] < 3 * period:
        return stop("core_timer_did_not_resume")

    if arm == "control":
        if deltas["gate_core"] < deltas["gate_shim"]:
            return stop("control_timer_order_inverted")
        verdict.update(qualified=True, reason="control")
        return verdict

    if control_witness is None:
        return stop("missing_control_witness")
    control = classify_real_column_gate_register_witness(
        "control", control_witness,
    )
    verdict["control"] = control
    if control.get("qualified") is not True:
        return stop("control_witness_failed")
    if control.get("period") != period:
        return stop("witness_period_mismatch")
    if deltas["gate_core"] >= deltas["gate_shim"]:
        return stop("core_timer_did_not_freeze")
    verdict.update(qualified=True, reason="freeze_resume")
    return verdict


def apply_physical_real_column_gate_witness(
    arm: str,
    result: dict,
    kernel_log: str,
    control_result: dict | None = None,
) -> dict:
    """Replace the physical oracle while preserving trace corroboration."""
    witness = parse_real_column_gate_register_witness(kernel_log)
    control_witness = (
        control_result.get("register_witness")
        if isinstance(control_result, dict)
        else None
    )
    classification = classify_real_column_gate_register_witness(
        arm, witness, control_witness,
    )
    return {
        **result,
        "qualified": classification["qualified"],
        "trace_classification": result.get("classification", {}),
        "register_witness": witness,
        "classification": classification,
    }


def classify_real_column_gate(
    arm: str,
    events: list[dict],
    output: bytes,
    expected_output: bytes,
    clock_before: dict,
    clock_after: dict,
    command_ok: bool,
    canary_ok: bool,
    col: int = 1,
    core_row: int = 2,
    shim_row: int = 0,
    channel: int = 13,
) -> dict:
    """Apply the exact paired freeze/resume trace contract."""
    cadence = 65
    verdict = {"qualified": False, "arm": arm, "cadence": cadence}

    def stop(reason):
        verdict["reason"] = reason
        return verdict

    if arm not in {"control", "treatment"}:
        return stop("unknown_arm")
    if command_ok is not True:
        return stop("command_failed")
    if output != expected_output:
        return stop("output_mismatch")
    if not isinstance(clock_before, dict) or not isinstance(clock_after, dict):
        return stop("missing_clock_identity")
    for name in ("mp_npu_mhz", "h_mhz"):
        if (
            not isinstance(clock_before.get(name), int)
            or isinstance(clock_before[name], bool)
            or clock_before[name] <= 0
        ):
            return stop("missing_clock_identity")
    if clock_before != clock_after:
        return stop("clocks_changed")
    if canary_ok is not True:
        return stop("canary_failed")

    def timestamps(name, pkt_type, row):
        values = [
            event.get("ts") for event in events
            if event.get("name") == name
            and event.get("pkt_type") == pkt_type
            and event.get("col") == col
            and event.get("row") == row
        ]
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            return None
        return sorted(values)

    faults = timestamps("PM_ADDRESS_OUT_OF_RANGE", 0, core_row)
    core = timestamps("PERF_CNT_3", 0, core_row)
    broadcasts = timestamps(f"BROADCAST_A_{channel}", 2, shim_row)
    heartbeats = timestamps("PERF_CNT_0", 2, shim_row)
    if faults is None or core is None or broadcasts is None or heartbeats is None:
        return stop("invalid_timestamp")
    verdict["series"] = {
        "core": core,
        "broadcasts": broadcasts,
        "heartbeats": heartbeats,
    }

    heartbeat_cadence = _constant_cadence(heartbeats)
    if heartbeat_cadence != cadence:
        return stop("irregular_shim_heartbeat")
    if not faults:
        if len(heartbeats) < 7:
            return stop("insufficient_shim_heartbeats")
        return stop("missing_pm_fault")
    core_cadence = _constant_cadence(core)
    if core_cadence != cadence:
        return stop("irregular_core_heartbeat")
    if len(core) != len(broadcasts):
        return stop("core_to_shim_count_mismatch")

    if arm == "control":
        if len(heartbeats) < 7:
            return stop("insufficient_shim_heartbeats")
        if len(broadcasts) < 7 or _constant_cadence(broadcasts) != cadence:
            return stop("irregular_control_broadcast")
        verdict.update(qualified=True, reason="control")
        return verdict

    if len(broadcasts) < 6:
        return stop("insufficient_broadcast_samples")
    deltas = [right - left for left, right in zip(broadcasts, broadcasts[1:])]
    gaps = [index for index, delta in enumerate(deltas) if delta >= 4 * cadence]
    if len(gaps) != 1:
        return stop("missing_or_multiple_gate_gaps")
    gap_index = gaps[0]
    if any(delta != cadence for index, delta in enumerate(deltas)
           if index != gap_index):
        return stop("irregular_broadcast_cadence")
    before_count = gap_index + 1
    after_count = len(broadcasts) - before_count
    if before_count < 3:
        return stop("insufficient_pre_gate_samples")
    if after_count < 3:
        return stop("insufficient_post_restore_samples")
    left = broadcasts[gap_index]
    right = broadcasts[gap_index + 1]
    inside = [heartbeat for heartbeat in heartbeats if left < heartbeat < right]
    if len(inside) < 3:
        return stop("shim_not_live_inside_gate")
    if not any(heartbeat < left for heartbeat in heartbeats) or not any(
        heartbeat > right for heartbeat in heartbeats
    ):
        return stop("shim_heartbeat_does_not_span_gate")

    verdict["broadcast_gap"] = {
        "left": left,
        "right": right,
        "cycles": right - left,
        "shim_heartbeats_inside": len(inside),
    }
    verdict.update(qualified=True, reason="freeze_resume")
    return verdict


def classify_real_column_gate_kvm_disposition(classification: dict) -> dict:
    """Admit behavioral proof or the exact known KVM scheduler RED."""
    if classification.get("qualified") is True:
        return {"admitted": True, "reason": "behavioral_witness"}
    series = classification.get("series")
    heartbeats = series.get("heartbeats") if isinstance(series, dict) else None
    if (
        classification.get("reason") == "missing_pm_fault"
        and classification.get("cadence") == 65
        and isinstance(series, dict)
        and series.get("core") == []
        and series.get("broadcasts") == []
        and isinstance(heartbeats, list)
        and len(heartbeats) >= 7
        and _constant_cadence(heartbeats) == 65
    ):
        return {"admitted": True, "reason": "known_scheduler_red"}
    return {"admitted": False, "reason": "behavioral_failure"}


def classify_real_column_gate_artifacts(
    arm: str,
    events_path: Path,
    output_path: Path,
    expected_output_path: Path,
    clock_before_path: Path,
    clock_after_path: Path,
    canary_output_path: Path,
    kernel_log_path: Path | None = None,
) -> dict:
    """Classify one completed KVM arm from its preserved raw artifacts."""
    document = json.loads(events_path.read_text())
    relabel_comparator_events(document)
    relabel_shim_witness_events(document)
    events_path.write_text(json.dumps(document, indent=2) + "\n")

    output = output_path.read_bytes()
    expected_output = expected_output_path.read_bytes()
    canary_output = canary_output_path.read_bytes()
    clock_before = json.loads(clock_before_path.read_text())
    clock_after = json.loads(clock_after_path.read_text())
    classification = classify_real_column_gate(
        arm, document.get("events", []), output, expected_output,
        clock_before, clock_after, command_ok=True,
        canary_ok=canary_output == expected_output,
    )
    result = {
        "schema_version": 1,
        "arm": arm,
        "qualified": classification["qualified"],
        "command_ok": True,
        "output": {
            "path": str(output_path),
            "sha256": hashlib.sha256(output).hexdigest(),
            "expected_sha256": hashlib.sha256(expected_output).hexdigest(),
            "matches": output == expected_output,
        },
        "canary": {
            "path": str(canary_output_path),
            "sha256": hashlib.sha256(canary_output).hexdigest(),
            "matches": canary_output == expected_output,
        },
        "clock_before": clock_before,
        "clock_after": clock_after,
        "classification": classification,
        "kvm_disposition": classify_real_column_gate_kvm_disposition(
            classification,
        ),
    }
    if kernel_log_path is not None:
        result["register_witness"] = parse_real_column_gate_register_witness(
            kernel_log_path.read_text(),
        )
    return result


def classify_shim_witness(
    events: list[dict],
    output: bytes,
    expected_output: bytes,
    expected_core_signature: list[tuple[str, int]] | None = None,
    no_fault_control: bool = False,
    core_threshold: int = 64,
    channel: int = 13,
    col: int = 1,
    core_row: int = 2,
    shim_row: int = 0,
) -> dict:
    """Classify the same-shim-domain heartbeat evidence."""
    verdict = {"qualified": False}

    def stop(reason):
        verdict["reason"] = reason
        return verdict

    if output != expected_output:
        return stop("output_mismatch")

    shim = [
        event for event in events
        if event.get("pkt_type") == 2
        and event.get("col") == col
        and event.get("row") == shim_row
    ]
    broadcasts = sorted(
        event["ts"] for event in shim
        if event.get("name") == f"BROADCAST_A_{channel}"
    )
    heartbeats = sorted(
        event["ts"] for event in shim
        if event.get("name") == "PERF_CNT_0"
    )
    heartbeat_cadence = _constant_cadence(heartbeats)
    verdict.update(
        broadcast_count=len(broadcasts),
        heartbeat_count=len(heartbeats),
        heartbeat_cadence=heartbeat_cadence,
    )
    if heartbeat_cadence is None:
        return stop("irregular_shim_heartbeat")
    if heartbeat_cadence != core_threshold + 1:
        return stop("unexpected_shim_heartbeat_cadence")

    if no_fault_control:
        if broadcasts:
            return stop("spurious_core_broadcast")
        verdict.update(qualified=True, reason="control")
        return verdict

    signature = core_fault_signature(events, col, core_row)
    verdict["core_signature"] = signature
    if signature is None:
        return stop("missing_pm_fault")
    if expected_core_signature is not None and signature != expected_core_signature:
        return stop("core_signature_mismatch")

    core_heartbeats = sorted(
        offset for name, offset in signature if name == "PERF_CNT_3"
    )
    verdict["core_heartbeat_count"] = len(core_heartbeats)
    if len(core_heartbeats) < 3 or core_heartbeats[0] != core_threshold:
        return stop("irregular_core_heartbeat")
    if _constant_cadence(core_heartbeats) != core_threshold + 1:
        return stop("irregular_core_heartbeat")

    if len(broadcasts) != len(core_heartbeats):
        return stop("core_to_shim_count_mismatch")
    broadcast_cadence = _constant_cadence(broadcasts)
    verdict["broadcast_cadence"] = broadcast_cadence
    if len(broadcasts) < 3 or broadcast_cadence is None:
        return stop("irregular_broadcast_cadence")
    if not any(heartbeat < broadcasts[-1] for heartbeat in heartbeats):
        return stop("shim_heartbeat_not_concurrent")
    if not any(
        heartbeat > broadcasts[-1] + broadcast_cadence
        for heartbeat in heartbeats
    ):
        return stop("shim_not_live_after_missing_core_heartbeat")

    verdict.update(qualified=True, reason="qualified")
    return verdict


def classify_capture(
    events: list[dict],
    output: bytes,
    expected_output: bytes,
    col: int = 1,
    row: int = 2,
) -> str:
    """Return ``fires``, ``gates_first``, or ``invalid`` for one fault run."""
    if output != expected_output:
        return "invalid"
    if not any(event.get("name") == "DMA_MM2S_0_FINISHED_BD" for event in events):
        return "invalid"

    core = [
        event for event in events
        if event.get("pkt_type") == 0
        and event.get("col") == col
        and event.get("row") == row
    ]
    faults = [event["ts"] for event in core
              if event.get("name") == "PM_ADDRESS_OUT_OF_RANGE"]
    if not faults:
        return "invalid"
    fault = min(faults)
    if not any(event.get("name") == "INSTR_LOCK_RELEASE_REQ"
               and event["ts"] < fault for event in core):
        return "invalid"
    if not any(event.get("name") == "PERF_CNT_2"
               and event["ts"] < fault for event in core):
        return "invalid"

    comparator = [event["ts"] for event in core
                  if event.get("name") == "PERF_CNT_3"]
    if comparator:
        return "fires" if min(comparator) > fault else "invalid"
    if any(event.get("name") == "PERF_CNT_2" and event["ts"] > fault
           for event in core):
        return "gates_first"
    return "invalid"


def classify_probe(
    events: list[dict],
    output: bytes,
    expected_output: bytes,
    clock_before: dict,
    clock_after: dict,
    expected_mode: str,
    expected_clock: dict | None = None,
) -> str:
    """Classify one threshold run only when its reported clocks are stable."""
    if clock_before != clock_after:
        return "invalid"
    if clock_before.get("power_mode") != expected_mode:
        return "invalid"
    if expected_clock is not None and clock_before != expected_clock:
        return "invalid"
    return classify_capture(events, output, expected_output)


def classify_control(
    events: list[dict],
    output: bytes,
    expected_output: bytes,
    col: int = 1,
    row: int = 2,
) -> str:
    """Validate the no-fault control arm of comparator qualification."""
    if output != expected_output:
        return "invalid"
    if not any(event.get("name") == "DMA_MM2S_0_FINISHED_BD" for event in events):
        return "invalid"
    core = [
        event for event in events
        if event.get("pkt_type") == 0
        and event.get("col") == col
        and event.get("row") == row
    ]
    names = {event.get("name") for event in core}
    if not {"PERF_CNT_2", "INSTR_LOCK_RELEASE_REQ"} <= names:
        return "invalid"
    if {"PM_ADDRESS_OUT_OF_RANGE", "PERF_CNT_3"} & names:
        return "invalid"
    return "control"


def relabel_comparator_events(document: dict, row: int = 2) -> None:
    """Correct decoder metadata for slot 4 patched from GROUP_ERRORS to counter 3."""
    core_names = document.get("slot_names", {}).get("core", [])
    if len(core_names) > 4:
        core_names[4] = "PERF_CNT_3"
    for event in document.get("events", []):
        if (event.get("pkt_type") == 0 and event.get("row") == row
                and event.get("slot") == 4):
            event["name"] = "PERF_CNT_3"


def relabel_shim_witness_events(
    document: dict, row: int = 0, channel: int = 13,
) -> None:
    """Correct decoder metadata for the two patched shim witness slots."""
    names = document.get("slot_names", {}).get("shim", [])
    replacements = {2: f"BROADCAST_A_{channel}", 3: "PERF_CNT_0"}
    for slot, name in replacements.items():
        if len(names) > slot:
            names[slot] = name
    for event in document.get("events", []):
        if event.get("pkt_type") == 2 and event.get("row") == row:
            name = replacements.get(event.get("slot"))
            if name is not None:
                event["name"] = name


def search_boundary(probe, initial: int = 64) -> tuple[int, int]:
    """Exponential then integer-binary search for adjacent fire/gate bounds."""
    if probe(initial) != "fires":
        raise RuntimeError(f"initial threshold {initial} did not fire")
    low = initial
    while True:
        high = min(low * 2, 0xFFFFFFFF)
        result = probe(high)
        if result == "gates_first":
            break
        if result == "invalid":
            raise RuntimeError(f"invalid threshold {high}")
        if high == 0xFFFFFFFF:
            raise RuntimeError("comparator still fires at maximum threshold")
        low = high

    while high - low > 1:
        threshold = (low + high) // 2
        result = probe(threshold)
        if result == "fires":
            low = threshold
        elif result == "gates_first":
            high = threshold
        else:
            raise RuntimeError(f"invalid threshold {threshold}")
    return low, high


def select_clock_regimes(records: list[dict], expected_mode: str) -> list[dict]:
    """Validate QoS probes, deduplicate clock pairs, and require three ratios."""
    selected = []
    pairs = set()
    ratios = set()
    for record in records:
        before = record.get("clock_before", {})
        after = record.get("clock_after", {})
        valid = (
            record.get("runner_ok") is True
            and record.get("output_ok") is True
            and before == after
            and before.get("power_mode") == expected_mode
            and isinstance(before.get("mp_npu_mhz"), int)
            and before["mp_npu_mhz"] > 0
            and isinstance(before.get("h_mhz"), int)
            and before["h_mhz"] > 0
        )
        if not valid:
            raise RuntimeError(
                f"invalid QoS clock candidate {record.get('label', '<unnamed>')}"
            )
        pair = (before["mp_npu_mhz"], before["h_mhz"])
        if pair in pairs:
            continue
        pairs.add(pair)
        ratios.add(Fraction(pair[1], pair[0]))
        selected.append(record)
    if len(ratios) < 3:
        raise RuntimeError(
            f"only {len(ratios)} distinct clock ratios observed; need at least 3"
        )
    return selected


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _query_clock(executable: Path) -> dict:
    return json.loads(subprocess.check_output([str(executable)], text=True))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe_clock_regime(
    session,
    template: Path,
    clock_query: Path,
    expected_output: bytes,
    run_dir: Path,
    label: str,
    qos: dict,
) -> dict:
    """Prime one QoS session, then measure clocks across a fresh context."""
    run_dir.mkdir(parents=True)

    def dispatch(name):
        output = run_dir / f"{name}.out.bin"
        result = session.run_one(
            instr=template,
            trace_out=run_dir / f"{name}.trace.bin",
            outputs=[output],
        )
        return result, output

    prime, prime_output = dispatch("prime")
    clock_before = _query_clock(clock_query)
    measured, measured_output = dispatch("measured")
    clock_after = _query_clock(clock_query)
    record = {
        "label": label,
        "qos": qos,
        "prime": prime,
        "measured": measured,
        "runner_ok": prime.get("ok") is True and measured.get("ok") is True,
        "output_ok": (
            prime_output.is_file()
            and measured_output.is_file()
            and prime_output.read_bytes() == expected_output
            and measured_output.read_bytes() == expected_output
        ),
        "clock_before": clock_before,
        "clock_after": clock_after,
    }
    _write_json(run_dir / "result.json", record)
    return record


def _run_probe(
    session,
    parser_session,
    template: Path,
    threshold: int,
    register_db: Path,
    clock_query: Path,
    expected_mode: str,
    expected_clock: dict,
    xclbin_mlir: Path,
    expected_output: bytes,
    run_dir: Path,
) -> dict:
    from trace_runner import _parse_trace_bin

    run_dir.mkdir()
    paths = {
        "insts": run_dir / "insts.bin",
        "trace": run_dir / "trace.bin",
        "events": run_dir / "events.json",
        "cycles": run_dir / "cycles.txt",
        "parse_log": run_dir / "parse.log",
        "output": run_dir / "out.bin",
        "result": run_dir / "result.json",
    }
    patched, _ = patcher.patch_register_fields(
        template.read_bytes(), 0, 2, "core",
        "Performance_Counter3_Event_Value",
        {"Counter_Event_Value": threshold}, register_db,
    )
    paths["insts"].write_bytes(patched)
    result = {
        "threshold": threshold,
        "insts_sha256": _sha256(paths["insts"]),
        "clock_before": _query_clock(clock_query),
        "paths": {name: str(path) for name, path in paths.items()
                  if name != "result"},
    }
    try:
        result["runner"] = session.run_one(
            instr=paths["insts"], trace_out=paths["trace"],
            outputs=[paths["output"]],
        )
    finally:
        result["clock_after"] = _query_clock(clock_query)
        _write_json(paths["result"], result)
    if not result["runner"].get("ok"):
        result["classification"] = "invalid"
        _write_json(paths["result"], result)
        return result

    ok, error, cycles, event_count = _parse_trace_bin(
        paths["trace"], xclbin_mlir, paths["events"], paths["cycles"],
        paths["parse_log"], os.environ.copy(), parser_session=parser_session,
    )
    result.update(parse_ok=ok, parse_error=error, parse_cycles=cycles,
                  parser_event_count=event_count)
    if not ok:
        result["classification"] = "invalid"
        _write_json(paths["result"], result)
        return result

    document = json.loads(paths["events"].read_text())
    relabel_comparator_events(document)
    paths["events"].write_text(json.dumps(document, indent=2) + "\n")
    result["actual_event_count"] = len(document.get("events", []))
    result["output_sha256"] = _sha256(paths["output"])
    result["classification"] = classify_probe(
        document.get("events", []), paths["output"].read_bytes(),
        expected_output, result["clock_before"], result["clock_after"],
        expected_mode, expected_clock,
    )
    _write_json(paths["result"], result)
    return result


def run_campaign(args) -> dict:
    from trace_runner import ParseSession, RunnerSession

    fixture = args.fixture.resolve()
    output = args.output.resolve()
    if "XDNA_EMU" in os.environ:
        raise RuntimeError("XDNA_EMU must be unset for a physical clock campaign")
    if output.exists():
        raise FileExistsError(f"campaign output already exists: {output}")
    required = [
        args.template, args.register_db, args.clock_query,
        fixture / "fault-package/aie.xclbin",
        fixture / "fault-package/work/input_with_addresses.mlir",
        fixture / "hw.out.bin",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing campaign input(s): " + ", ".join(missing))
    output.mkdir(parents=True)
    state_path = output / "campaign.json"
    original = _query_clock(args.clock_query)
    expected_output = (fixture / "hw.out.bin").read_bytes()
    runner_env = {
        "BRIDGE_RUNNER_ASYNC_CTX": "0",
        "BRIDGE_RUNNER_REUSE_CONTEXT": "0",
    }
    state = {
        "schema_version": 2,
        "fixture": str(fixture),
        "fixture_xclbin_sha256": _sha256(fixture / "fault-package/aie.xclbin"),
        "template": str(args.template.resolve()),
        "template_sha256": _sha256(args.template),
        "original_clock": original,
        "clock_candidates": [],
        "regimes": [],
        "status": "running",
    }
    _write_json(state_path, state)
    failure = None
    try:
        for fps in args.qos_fps:
            qos = {"gops": args.qos_gops, "fps": fps}
            label = f"gops-{args.qos_gops}-fps-{fps}"
            candidate_dir = output / "clock-probes" / label
            candidate_dir.mkdir(parents=True)
            with RunnerSession(
                xclbin=fixture / "fault-package/aie.xclbin",
                runner_env=runner_env,
                side="HW",
                stderr_log=candidate_dir / "runner.log",
                qos_gops=args.qos_gops,
                qos_fps=fps,
            ) as runner:
                record = probe_clock_regime(
                    runner, args.template, args.clock_query, expected_output,
                    candidate_dir / "dispatches", label, qos,
                )
            state["clock_candidates"].append(record)
            _write_json(state_path, state)

        selected = select_clock_regimes(
            state["clock_candidates"], original["power_mode"],
        )
        state["regimes"] = [
            {
                "label": record["label"],
                "qos": record["qos"],
                "clock": record["clock_before"],
                "passes": [],
            }
            for record in selected
        ]
        _write_json(state_path, state)

        for regime in state["regimes"]:
            label = regime["label"]
            qos = regime["qos"]
            for pass_number in (1, 2):
                pass_dir = output / "searches" / label / f"pass-{pass_number}"
                pass_dir.mkdir(parents=True)
                pass_state = {"pass": pass_number, "runs": []}
                regime["passes"].append(pass_state)
                _write_json(state_path, state)
                with RunnerSession(
                    xclbin=fixture / "fault-package/aie.xclbin",
                    runner_env=runner_env, side="HW",
                    stderr_log=pass_dir / "runner.log",
                    qos_gops=qos["gops"], qos_fps=qos["fps"],
                ) as runner, ParseSession(
                    side="HW", stderr_log=pass_dir / "parser.log",
                ) as parser:
                    primed = probe_clock_regime(
                        runner, args.template, args.clock_query, expected_output,
                        pass_dir / "prime", label, qos,
                    )
                    if not (
                        primed["runner_ok"] and primed["output_ok"]
                        and primed["clock_before"] == regime["clock"]
                        and primed["clock_after"] == regime["clock"]
                    ):
                        raise RuntimeError(
                            f"{label} did not reproduce its admitted clock pair"
                        )

                    def probe(threshold):
                        run_dir = pass_dir / (
                            f"{len(pass_state['runs']):02d}-threshold-{threshold}"
                        )
                        result = _run_probe(
                            runner, parser, args.template, threshold,
                            args.register_db, args.clock_query,
                            original["power_mode"], regime["clock"],
                            fixture / "fault-package/work/input_with_addresses.mlir",
                            expected_output, run_dir,
                        )
                        pass_state["runs"].append({
                            "threshold": threshold,
                            "classification": result["classification"],
                            "result": str(run_dir / "result.json"),
                        })
                        _write_json(state_path, state)
                        return result["classification"]

                    pass_state["bracket"] = list(search_boundary(probe))
                    _write_json(state_path, state)

            brackets = [entry["bracket"] for entry in regime["passes"]]
            if brackets[0] != brackets[1]:
                raise RuntimeError(f"{label} pass brackets disagree: {brackets}")
            regime["bracket"] = brackets[0]
            _write_json(state_path, state)
    except BaseException as error:
        failure = error
        state["status"] = "stopped"
        state["stop_reason"] = f"{type(error).__name__}: {error}"
        _write_json(state_path, state)
    finally:
        try:
            restore_dir = output / "restore-no-qos"
            restore_dir.mkdir(parents=True)
            with RunnerSession(
                xclbin=fixture / "fault-package/aie.xclbin",
                runner_env=runner_env,
                side="HW",
                stderr_log=restore_dir / "runner.log",
            ) as runner:
                state["restore_probe"] = probe_clock_regime(
                    runner, args.template, args.clock_query, expected_output,
                    restore_dir / "dispatches", "no-qos", None,
                )
            state["restored_clock"] = _query_clock(args.clock_query)
            state["restored"] = state["restored_clock"] == original
            if not state["restored"]:
                raise RuntimeError(
                    "no-QoS restore did not reproduce the original clock state"
                )
        except BaseException as error:
            state["restored"] = False
            state["restore_error"] = f"{type(error).__name__}: {error}"
            if failure is None:
                failure = error
        if failure is None:
            state["status"] = "captured"
        _write_json(state_path, state)
    if failure is not None:
        raise failure
    return state


def _parse_args(argv=None):
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run the pinned Phoenix PM-fault clock-boundary campaign",
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True,
                        help="qualified comparator insts.bin")
    parser.add_argument(
        "--register-db", type=Path,
        default=repo.parent / "mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json",
    )
    parser.add_argument("--clock-query", type=Path,
                        default=repo / "build/tools/xdna-clock-query")
    parser.add_argument("--qos-gops", type=int, default=1)
    parser.add_argument("--qos-fps", type=int, action="append")
    args = parser.parse_args(argv)
    if args.qos_fps is None:
        args.qos_fps = list(_DEFAULT_QOS_FPS)
    return args


def _parse_real_column_gate_args(argv):
    parser = argparse.ArgumentParser(
        description="Classify one preserved Phoenix real column-gate arm",
    )
    parser.add_argument("--arm", choices=("control", "treatment"), required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-output", type=Path, required=True)
    parser.add_argument("--clock-before", type=Path, required=True)
    parser.add_argument("--clock-after", type=Path, required=True)
    parser.add_argument("--canary-output", type=Path, required=True)
    parser.add_argument("--kernel-log", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args(argv)


def _parse_prepare_real_column_gate_trace_args(argv):
    parser = argparse.ArgumentParser(
        description="Prepare the pinned open trace for protected-hook closure",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--register-db", type=Path, required=True)
    parser.add_argument("--events-header", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        if sys.argv[1:2] == ["prepare-real-column-gate-trace"]:
            args = _parse_prepare_real_column_gate_trace_args(sys.argv[2:])
            prepared = prepare_real_column_gate_trace(
                args.input.read_bytes(), args.register_db, args.events_header,
            )
            args.output.write_bytes(prepared)
            print(hashlib.sha256(prepared).hexdigest())
            raise SystemExit(0)
        if sys.argv[1:2] == ["classify-real-column-gate"]:
            args = _parse_real_column_gate_args(sys.argv[2:])
            result = classify_real_column_gate_artifacts(
                args.arm, args.events, args.output, args.expected_output,
                args.clock_before, args.clock_after, args.canary_output,
                args.kernel_log,
            )
            _write_json(args.result, result)
            print(json.dumps(result, indent=2))
            raise SystemExit(0 if result["qualified"] else 1)
        campaign = run_campaign(_parse_args())
        print(json.dumps(campaign, indent=2))
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
