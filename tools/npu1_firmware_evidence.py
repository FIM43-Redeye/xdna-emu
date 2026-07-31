#!/usr/bin/env python3
"""Frozen NPU1 firmware-evidence campaign and safe transaction model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import random
import re
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


class Outcome(str, Enum):
    SUCCESS = "success"
    INTENTIONAL_REJECTION = "intentional_rejection"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    PROVENANCE_FAILURE = "provenance_failure"
    SEMANTIC_MISMATCH = "semantic_mismatch"
    DEVICE_FAULT_OR_WEDGE = "device_fault_or_wedge"


@dataclass(frozen=True)
class Arm:
    name: str
    force_cmdlist: str
    execute_opcode: int


TREATMENT = Arm("treatment", "Y", 0x18)
CONTROL = Arm("control", "N", 0x10)


@dataclass(frozen=True)
class ScheduleEntry:
    ordinal: int
    repetition: int
    arm: Arm

    @property
    def run_id(self) -> str:
        return f"run.{self.ordinal:03d}.{self.arm.name}.{self.repetition:03d}"


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(frozen=True)
class RawCaptureIndex:
    run_id: str
    lifecycle_text: str
    tdr_before: int
    tdr_after: int
    iommu_faults_before: int
    iommu_faults_after: int
    teardown_ok: bool
    restoration_ok: bool
    kernel_evidence_ok: bool = True
    kernel_evidence_reason: str = ""
    execute_status: int | None = 0


@dataclass(frozen=True)
class LifecycleRecord:
    kind: str
    value: str


@dataclass(frozen=True)
class LifecycleResult:
    ok: bool
    records: tuple[LifecycleRecord, ...]
    execute_opcode: int | None
    reason: str


@dataclass(frozen=True)
class KernelEvidenceResult:
    ok: bool
    execute_status: int | None
    reason: str


@dataclass(frozen=True)
class OutputOracleResult:
    ok: bool
    values: tuple[int, ...]
    pass_marker: bool
    reason: str


@dataclass(frozen=True)
class RunClassification:
    run_id: str
    outcome: Outcome
    reasons: tuple[str, ...]
    execute_opcode: int | None
    output_values: tuple[int, ...]
    unknown_success_words: tuple[str, ...]


@dataclass(frozen=True)
class CampaignClassification:
    outcome: Outcome
    completed_run_ids: tuple[str, ...]
    failed_run_id: str | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FixtureInput:
    input_id: str
    semantic_kind: str
    logical_name: str
    sha256: str
    fixture_bundle_id: str
    artifact_path: str
    source_path: str


@dataclass(frozen=True)
class VerticalSpec:
    pci_id: str
    firmware_logical_path: str
    firmware_sha256: str
    driver_protocol_revision: str
    xclbin_sha256: str
    instructions_sha256: str
    executable_sha256: str
    tdr_timeout_ms: int

    @property
    def file_hashes(self) -> Mapping[str, str]:
        return {
            "firmware": self.firmware_sha256,
            "xclbin": self.xclbin_sha256,
            "instructions": self.instructions_sha256,
            "executable": self.executable_sha256,
        }


VERTICAL_SPEC = VerticalSpec(
    pci_id="1022:1502",
    firmware_logical_path="amdnpu/1502_00/npu.dev.sbin",
    firmware_sha256="d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e",
    driver_protocol_revision="216cefececd74effcd7a88350c71b99f5ef9a215",
    xclbin_sha256="c46198460a07ff2aa03a12b125851a223eeb1e8c315132d60aec18d831453bf6",
    instructions_sha256="ee49b0a66c53d3952604460fe83fab879f38f1dad6cb70a994fc4422aa285896",
    executable_sha256="511d40e38eecf70def29322b5af8ce261bb79dfb793dc0ca45abc8a8f99b8806",
    tdr_timeout_ms=2000,
)

RUN_TIMEOUT_SECONDS = 20


def vertical_schedule(seed: int) -> tuple[ScheduleEntry, ...]:
    return repetition_schedule(1, 1, seed)


def repetition_schedule(
    treatment_count: int, control_count: int, seed: int
) -> tuple[ScheduleEntry, ...]:
    if treatment_count < 0 or control_count < 0:
        raise ValueError("schedule counts must be nonnegative")
    entries = [
        *(
            ScheduleEntry(0, repetition, TREATMENT)
            for repetition in range(treatment_count)
        ),
        *(ScheduleEntry(0, repetition, CONTROL) for repetition in range(control_count)),
    ]
    random.Random(seed).shuffle(entries)
    return tuple(
        ScheduleEntry(ordinal, entry.repetition, entry.arm)
        for ordinal, entry in enumerate(entries)
    )


_OUTPUT = re.compile(r"^Correct output ([0-9]+) == ([0-9]+)$")


def parse_output(stdout: str) -> OutputOracleResult:
    lines = [line.strip() for line in stdout.splitlines()]
    pairs = [match.groups() for line in lines if (match := _OUTPUT.fullmatch(line))]
    values = tuple(int(actual) for actual, _ in pairs)
    references = tuple(int(reference) for _, reference in pairs)
    expected = tuple(range(2, 66))
    pass_count = sum(line == "PASS!" for line in lines)
    if values != expected or references != expected:
        return OutputOracleResult(
            False, values, pass_count == 1, "ordered output is not exactly 2 through 65"
        )
    if pass_count != 1:
        return OutputOracleResult(
            False, values, False, "expected exactly one PASS! marker"
        )
    if any(line.startswith("Error in output ") for line in lines):
        return OutputOracleResult(False, values, True, "host reported an output error")
    return OutputOracleResult(True, values, True, "")


_TRACE_EVENT = re.compile(
    r":\s+(tracing_mark_write|mbox_set_tail|mbox_set_head|mbox_irq_handle|"
    r"mbox_rx_worker|xdna_job):\s+(.*)$"
)
_MBOX_MESSAGE = re.compile(
    r"^xdna_mailbox\.([0-9]+) id (0x[0-9a-fA-F]+) opcode (0x[0-9a-fA-F]+)$"
)
_MBOX_CHANNEL = re.compile(r"^xdna_mailbox\.([0-9]+)$")
_FENCE_EVENT = re.compile(
    r"fence=\(context:([0-9]+), seqno:([0-9]+)\), .* "
    r"(job run|signaling fence), op=[0-9]+$"
)
_KERNEL_MESSAGE = re.compile(
    r"opcode 0x([0-9a-fA-F]+) size ([0-9]+) id 0x([0-9a-fA-F]+)"
)


def _scoped_lines(text: str, run_id: str) -> tuple[list[str], str | None]:
    lines = [line.strip() for line in text.splitlines()]
    begin_marker = re.compile(rf"(?:^|:\s)NPU1_FW_BEGIN {re.escape(run_id)}$")
    end_marker = re.compile(rf"(?:^|:\s)NPU1_FW_END {re.escape(run_id)}$")
    begin = [index for index, line in enumerate(lines) if begin_marker.search(line)]
    end = [index for index, line in enumerate(lines) if end_marker.search(line)]
    if len(begin) != 1 or len(end) != 1 or begin[0] >= end[0]:
        return [], "run boundary markers are missing, duplicated, or out of order"
    return lines[begin[0] + 1 : end[0]], None


def parse_lifecycle(text: str, entry: ScheduleEntry) -> LifecycleResult:
    lines, boundary_error = _scoped_lines(text, entry.run_id)
    if boundary_error:
        return LifecycleResult(False, (), None, boundary_error)
    events = [
        (index, match.group(1), match.group(2))
        for index, line in enumerate(lines)
        if (match := _TRACE_EVENT.search(line))
    ]
    messages: dict[str, list[tuple[int, int, int, int]]] = {
        "mbox_set_tail": [],
        "mbox_set_head": [],
    }
    for index, kind, payload in events:
        if kind in messages and (match := _MBOX_MESSAGE.fullmatch(payload)):
            messages[kind].append(
                (
                    index,
                    int(match.group(1)),
                    int(match.group(2), 16),
                    int(match.group(3), 16),
                )
            )

    expected_opcodes = (0x02, 0x106, 0x11, entry.arm.execute_opcode, 0x03)
    tails = messages["mbox_set_tail"]
    records = [LifecycleRecord("request", f"0x{opcode:x}") for _, _, _, opcode in tails]
    observed_execute = [
        opcode
        for _, _, _, opcode in tails
        if opcode in (TREATMENT.execute_opcode, CONTROL.execute_opcode)
    ]
    execute_opcode = observed_execute[0] if len(observed_execute) == 1 else None
    if tuple(opcode for _, _, _, opcode in tails) != expected_opcodes:
        return LifecycleResult(
            False,
            tuple(records),
            execute_opcode,
            "firmware request sequence is missing, duplicated, unexpected, or out of order",
        )

    heads = messages["mbox_set_head"]
    for tail in tails:
        matches = [head for head in heads if head[1:] == tail[1:] and head[0] > tail[0]]
        if len(matches) != 1:
            return LifecycleResult(
                False,
                tuple(records),
                execute_opcode,
                "firmware response is missing, duplicated, or mismatched",
            )
        records.append(LifecycleRecord("response", f"0x{tail[3]:x}"))
    if len(heads) != len(tails):
        return LifecycleResult(
            False,
            tuple(records),
            execute_opcode,
            "unexpected firmware response is present",
        )

    execute_tail = tails[3]
    execute_head = next(head for head in heads if head[1:] == execute_tail[1:])
    job_runs = [
        (index, match.group(1), match.group(2))
        for index, kind, payload in events
        if kind == "xdna_job"
        and (match := _FENCE_EVENT.search(payload))
        and match.group(3) == "job run"
    ]
    fence_signals = [
        (index, match.group(1), match.group(2))
        for index, kind, payload in events
        if kind == "xdna_job"
        and (match := _FENCE_EVENT.search(payload))
        and match.group(3) == "signaling fence"
    ]
    if len(job_runs) != 1 or len(fence_signals) != 1:
        return LifecycleResult(
            False,
            tuple(records),
            execute_opcode,
            "scheduler or fence evidence is missing or duplicated",
        )
    job_run = job_runs[0]
    fence = fence_signals[0]
    if job_run[1:] != fence[1:] or job_run[0] >= execute_tail[0]:
        return LifecycleResult(
            False,
            tuple(records),
            execute_opcode,
            "scheduler and fence evidence do not identify one causal job",
        )

    def channel_events(kind: str) -> list[int]:
        return [
            index
            for index, event_kind, payload in events
            if event_kind == kind
            and (match := _MBOX_CHANNEL.fullmatch(payload))
            and int(match.group(1)) == execute_tail[1]
            and execute_tail[0] < index < execute_head[0]
        ]

    interrupts = channel_events("mbox_irq_handle")
    workers = channel_events("mbox_rx_worker")
    if (
        not interrupts
        or not workers
        or not (interrupts[0] < workers[0] < fence[0] < execute_head[0])
        or tails[4][0] <= fence[0]
    ):
        return LifecycleResult(
            False,
            tuple(records),
            execute_opcode,
            "execute interrupt, response worker, fence, queue head, or teardown is not causal",
        )
    records.extend(
        LifecycleRecord("event", event)
        for event in (
            "scheduler",
            "interrupt",
            "mailbox_response",
            "fence",
            "queue_head",
        )
    )
    return LifecycleResult(True, tuple(records), entry.arm.execute_opcode, "")


def parse_kernel_evidence(text: str, entry: ScheduleEntry) -> KernelEvidenceResult:
    lines, boundary_error = _scoped_lines(text, entry.run_id)
    if boundary_error:
        return KernelEvidenceResult(False, None, boundary_error)
    headers = [
        (index, int(match.group(1), 16), int(match.group(3), 16))
        for index, line in enumerate(lines)
        if (match := _KERNEL_MESSAGE.search(line))
    ]
    expected_opcodes = (0x02, 0x02, 0x106, 0x106, 0x11, 0x11)
    expected_opcodes += (entry.arm.execute_opcode, entry.arm.execute_opcode, 0x03, 0x03)
    if tuple(opcode for _, opcode, _ in headers) != expected_opcodes:
        return KernelEvidenceResult(
            False,
            None,
            "kernel request/response headers are missing, duplicated, or unexpected",
        )
    for offset in range(0, len(headers), 2):
        request, response = headers[offset : offset + 2]
        if request[1:] != response[1:]:
            return KernelEvidenceResult(
                False, None, "kernel request/response IDs do not match"
            )
        request_lines = lines[request[0] + 1 : response[0]]
        if not any(
            re.search(rf"\b{request[2]:08x}\s+{request[1]:08x}\b", line, re.IGNORECASE)
            and "req data:" in line
            for line in request_lines
        ):
            return KernelEvidenceResult(False, None, "kernel request bytes are missing")
        if request[1] not in (TREATMENT.execute_opcode, CONTROL.execute_opcode):
            next_header = (
                headers[offset + 2][0] if offset + 2 < len(headers) else len(lines)
            )
            if not any(
                re.search(r"resp data: .*:\s+00000000\b", line, re.IGNORECASE)
                for line in lines[response[0] + 1 : next_header]
            ):
                return KernelEvidenceResult(
                    False, None, "zero-status response bytes are missing"
                )

    status_pattern = (
        re.compile(r"\bStatus 0x([0-9a-fA-F]+)$")
        if entry.arm == TREATMENT
        else re.compile(r"\bResp status 0x([0-9a-fA-F]+)$")
    )
    statuses = [
        int(match.group(1), 16)
        for line in lines
        if (match := status_pattern.search(line))
    ]
    if len(statuses) != 1:
        return KernelEvidenceResult(
            False, None, "execute status is missing or duplicated"
        )
    if statuses[0] != 0:
        return KernelEvidenceResult(False, statuses[0], "execute status is nonzero")
    return KernelEvidenceResult(True, statuses[0], "")


_TDR = re.compile(
    r"\baie2_tdr(?:_[a-z_]+)?\b|\bTDR (?:timeout|timed out|detected)\b", re.IGNORECASE
)
_IOMMU_FAULT = re.compile(r"\bIO_PAGE_FAULT\b")


def derive_capture_index(
    entry: ScheduleEntry,
    trace_text: str,
    dmesg_before: str,
    dmesg_after: str,
    *,
    teardown_ok: bool = True,
    restoration_ok: bool = True,
) -> RawCaptureIndex:
    kernel = parse_kernel_evidence(dmesg_after, entry)
    return RawCaptureIndex(
        run_id=entry.run_id,
        lifecycle_text=trace_text,
        tdr_before=len(_TDR.findall(dmesg_before)),
        tdr_after=len(_TDR.findall(dmesg_after)),
        iommu_faults_before=len(_IOMMU_FAULT.findall(dmesg_before)),
        iommu_faults_after=len(_IOMMU_FAULT.findall(dmesg_after)),
        teardown_ok=teardown_ok,
        restoration_ok=restoration_ok,
        kernel_evidence_ok=kernel.ok,
        kernel_evidence_reason=kernel.reason,
        execute_status=kernel.execute_status,
    )


_DYNAMIC_SELECTOR = re.compile(r"^file ([A-Za-z0-9_.-]+) line ([0-9]+) \+p$")
_DYNAMIC_CONTROL = re.compile(r"^(\S+):([0-9]+) \[amdxdna\]\S+ (\S+) ")


def dynamic_debug_print_states(
    control_text: str,
    selectors: Sequence[str],
) -> dict[str, bool]:
    callsites = []
    for line in control_text.splitlines():
        if match := _DYNAMIC_CONTROL.match(line):
            callsites.append(
                (Path(match.group(1)).name, int(match.group(2)), match.group(3))
            )
    states: dict[str, bool] = {}
    for selector in selectors:
        match = _DYNAMIC_SELECTOR.fullmatch(selector)
        if not match:
            raise ValueError(f"invalid dynamic-debug selector: {selector}")
        found = [
            flags
            for filename, line, flags in callsites
            if filename == match.group(1) and line == int(match.group(2))
        ]
        if len(found) != 1:
            raise ValueError(
                f"dynamic-debug selector matched {len(found)} callsites: {selector}"
            )
        states[selector] = "p" in found[0]
    return states


def dynamic_debug_restore_commands(states: Mapping[str, bool]) -> tuple[str, ...]:
    return tuple(
        selector.removesuffix("+p") + "-p"
        for selector, was_enabled in states.items()
        if not was_enabled
    )


def classify_run(
    entry: ScheduleEntry,
    command: CommandResult,
    capture: RawCaptureIndex,
) -> RunClassification:
    reasons: list[str] = []
    if capture.run_id != entry.run_id:
        return RunClassification(
            entry.run_id,
            Outcome.PROVENANCE_FAILURE,
            ("capture run ID does not match the schedule",),
            None,
            (),
            (),
        )
    if (
        capture.tdr_after < capture.tdr_before
        or capture.iommu_faults_after < capture.iommu_faults_before
    ):
        return RunClassification(
            entry.run_id,
            Outcome.PROVENANCE_FAILURE,
            ("kernel log counters regressed",),
            None,
            (),
            (),
        )
    if capture.tdr_after > capture.tdr_before:
        reasons.append("new TDR observed")
    if capture.iommu_faults_after > capture.iommu_faults_before:
        reasons.append("new IOMMU fault observed")
    if reasons:
        return RunClassification(
            entry.run_id, Outcome.DEVICE_FAULT_OR_WEDGE, tuple(reasons), None, (), ()
        )
    if command.timed_out:
        return RunClassification(
            entry.run_id,
            Outcome.INFRASTRUCTURE_FAILURE,
            ("host process timed out",),
            None,
            (),
            (),
        )
    if command.returncode != 0:
        return RunClassification(
            entry.run_id,
            Outcome.INFRASTRUCTURE_FAILURE,
            (f"host process exited {command.returncode}",),
            None,
            (),
            (),
        )
    if not capture.kernel_evidence_ok:
        outcome = (
            Outcome.SEMANTIC_MISMATCH
            if capture.execute_status not in (None, 0)
            else Outcome.PROVENANCE_FAILURE
        )
        return RunClassification(
            entry.run_id,
            outcome,
            (capture.kernel_evidence_reason,),
            None,
            (),
            (),
        )
    if not capture.teardown_ok or not capture.restoration_ok:
        return RunClassification(
            entry.run_id,
            Outcome.INFRASTRUCTURE_FAILURE,
            ("capture teardown or host-state restoration failed",),
            None,
            (),
            (),
        )
    lifecycle = parse_lifecycle(capture.lifecycle_text, entry)
    if not lifecycle.ok:
        outcome = (
            Outcome.PROVENANCE_FAILURE
            if not lifecycle.records
            else Outcome.SEMANTIC_MISMATCH
        )
        return RunClassification(
            entry.run_id, outcome, (lifecycle.reason,), lifecycle.execute_opcode, (), ()
        )
    output = parse_output(command.stdout)
    if not output.ok:
        return RunClassification(
            entry.run_id,
            Outcome.SEMANTIC_MISMATCH,
            (output.reason,),
            lifecycle.execute_opcode,
            output.values,
            (),
        )
    return RunClassification(
        entry.run_id,
        Outcome.SUCCESS,
        (),
        lifecycle.execute_opcode,
        output.values,
        (("fail_cmd_idx", "fail_cmd_status") if entry.arm == TREATMENT else ()),
    )


def classify_campaign(
    schedule: Sequence[ScheduleEntry],
    results: Sequence[RunClassification],
) -> CampaignClassification:
    completed: list[str] = []
    for index, entry in enumerate(schedule):
        if index >= len(results):
            return CampaignClassification(
                Outcome.INFRASTRUCTURE_FAILURE,
                tuple(completed),
                entry.run_id,
                ("scheduled run has no sealed result",),
            )
        result = results[index]
        completed.append(result.run_id)
        if result.run_id != entry.run_id:
            return CampaignClassification(
                Outcome.PROVENANCE_FAILURE,
                tuple(completed),
                entry.run_id,
                ("run results do not follow the frozen schedule",),
            )
        if result.outcome != Outcome.SUCCESS:
            return CampaignClassification(
                result.outcome, tuple(completed), result.run_id, result.reasons
            )
    return CampaignClassification(Outcome.SUCCESS, tuple(completed), None, ())


def _known(value: Any) -> dict[str, Any]:
    return {"state": "known", "value": value}


def render_observation_plan(
    campaign_id: str,
    schedule: Sequence[ScheduleEntry],
    fixture_inputs: Sequence[FixtureInput],
) -> bytes:
    inputs = sorted(fixture_inputs, key=lambda item: item.input_id)
    dependencies = {
        (item.fixture_bundle_id, item.artifact_path): {
            "fixture_bundle_id": item.fixture_bundle_id,
            "artifact_path": item.artifact_path,
            "artifact_sha256": item.sha256,
            "semantic_kind": item.semantic_kind,
            "source_path": item.source_path,
        }
        for item in inputs
    }
    plan = {
        "schema_version": 2,
        "role": "observation",
        "body": {
            "campaign": {
                "id": campaign_id,
                "tuple_ids": ["tuple.npu1.phoenix.fw-1_5_5_391"],
                "inventory_ids": ["inventory.npu1.firmware.command-list-execution"],
                "fact_ids": ["fact.npu1.firmware.physical-execution-envelope-pair"],
                "evidence_ids": ["evidence.npu1.firmware.physical-vertical-pair"],
                "provenance": "current",
                "risk_class": "non_destructive",
                "outcome": "success",
                "platform": _synthetic_platform(),
                "stimulus": {
                    "command": {
                        "argv": [
                            "./test.exe",
                            "-x",
                            "aie.xclbin",
                            "-k",
                            "MLIR_AIE",
                            "-i",
                            "insts.bin",
                        ],
                        "environment": {},
                    },
                    "source_revisions": [
                        {
                            "repository": "amdxdna-driver",
                            "commit": VERTICAL_SPEC.driver_protocol_revision,
                        }
                    ],
                    "build_recipe": _known(
                        {"logical_name": "vertical-campaign.json", "sha256": "f" * 64}
                    ),
                    "inputs": [
                        {
                            "id": item.input_id,
                            "semantic_kind": item.semantic_kind,
                            "content": {
                                "logical_name": item.logical_name,
                                "sha256": item.sha256,
                            },
                        }
                        for item in inputs
                    ],
                    "initial_state": ["normal_tdr", "serialized_hardware_access"],
                    "external_events": [],
                },
                "runs": [
                    {
                        "id": entry.run_id,
                        "ordinal": entry.ordinal,
                        "repetition": entry.repetition,
                        "completion": "complete",
                        "output_artifact_paths": [],
                        "observations": [],
                        "timing": [],
                        "errors": [],
                        "recovery_actions": [],
                        "teardown": "clean",
                        "control_run_ids": [
                            other.run_id
                            for other in schedule
                            if other.arm == CONTROL and entry.arm == TREATMENT
                        ],
                    }
                    for entry in schedule
                ],
            },
            "input_references": [
                {
                    "input_id": item.input_id,
                    "fixture_bundle_id": item.fixture_bundle_id,
                    "artifact_path": item.artifact_path,
                }
                for item in inputs
            ],
        },
        "dependencies": sorted(
            dependencies.values(),
            key=lambda item: (item["fixture_bundle_id"], item["artifact_path"]),
        ),
        "artifacts": [],
    }
    return (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode()


def _synthetic_platform() -> dict[str, Any]:
    def component(name: str) -> dict[str, Any]:
        return {
            "name": name,
            "revision": "synthetic",
            "sha256": _known("e" * 64),
        }

    return {
        "architecture": "aie2",
        "device_model_key": _known("npu1"),
        "driver_platform_id": _known("npu1"),
        "pci": _known(
            {
                "vendor_id": "1022",
                "device_id": "1502",
                "subsystem_vendor_id": "1022",
                "subsystem_device_id": "1502",
                "revision_id": "00",
            }
        ),
        "board_identity": _known("synthetic"),
        "firmware": _known(
            {
                "logical_name": VERTICAL_SPEC.firmware_logical_path,
                "sha256": VERTICAL_SPEC.firmware_sha256,
            }
        ),
        "host_kernel": _known(component("linux")),
        "kernel_modules": [component("amdxdna")],
        "driver": _known(
            {
                "repository": "executing-amdxdna",
                "commit": "synthetic",
            }
        ),
        "xrt_components": [component("xrt")],
        "toolchain_components": [component("mlir-aie")],
        "compiler_mode": _known("chess"),
        "execution_mode": _known("physical"),
        "reset_state": _known("fresh_module_epoch"),
        "power_state": _known("d0"),
        "clock_state": _known("default"),
        "iommu_state": _known("enabled"),
        "address_state": _known("captured"),
    }


@dataclass(frozen=True)
class PreflightSnapshot:
    file_hashes: Mapping[str, str]
    environment: Mapping[str, str]
    pci_id: str
    active_clients: int
    tdr_parameter_present: bool
    kernel_release: str = ""
    loaded_module_path: str = ""
    loaded_module_sha256: str = ""
    loaded_module_srcversion: str = ""
    original_module_srcversion: str = ""
    xrt_coreutil_path: str = ""
    device_node_present: bool = False
    power_control: str = ""


def preflight_errors(
    spec: VerticalSpec, snapshot: PreflightSnapshot
) -> tuple[str, ...]:
    errors = [
        f"{name} SHA-256 drift"
        for name, expected in spec.file_hashes.items()
        if snapshot.file_hashes.get(name) != expected
    ]
    errors.extend(
        f"{name} must be absent"
        for name in ("XDNA_EMU", "XDNA_EMU_RUNTIME")
        if name in snapshot.environment
    )
    if snapshot.pci_id != spec.pci_id:
        errors.append(f"physical PCI identity mismatch: expected {spec.pci_id}")
    if snapshot.active_clients:
        errors.append("active NPU clients are present")
    if not snapshot.tdr_parameter_present:
        errors.append("candidate module does not provide tdr_timeout_ms")
    return tuple(errors)


def module_preflight_errors(
    module: QualifiedModuleManifest,
    snapshot: PreflightSnapshot,
) -> tuple[str, ...]:
    errors: list[str] = []
    if module.source_revision != VERTICAL_SPEC.driver_protocol_revision:
        errors.append(
            "candidate module source revision is not the pinned protocol surface"
        )
    if module.kernel_release != snapshot.kernel_release:
        errors.append(
            "candidate module kernel release does not match the running kernel"
        )
    if (
        not snapshot.loaded_module_path
        or Path(snapshot.loaded_module_path).resolve() != module.original_path.resolve()
        or snapshot.loaded_module_sha256 != module.original_sha256
    ):
        errors.append(
            "loaded module does not match the recorded original system module"
        )
    if (
        not snapshot.loaded_module_srcversion
        or snapshot.loaded_module_srcversion != snapshot.original_module_srcversion
    ):
        errors.append(
            "loaded module srcversion does not match the original module file"
        )
    if not snapshot.xrt_coreutil_path.startswith("/opt/xilinx/xrt/lib/"):
        errors.append("XRT core library did not resolve beneath /opt/xilinx/xrt/lib")
    if not snapshot.device_node_present:
        errors.append("physical NPU device node is missing")
    if snapshot.power_control != "on":
        errors.append("NPU PCI power/control is not pinned on")
    return tuple(errors)


REQUIRED_TRACE_EVENTS = (
    "xdna_job",
    "mbox_set_tail",
    "mbox_set_head",
    "mbox_irq_handle",
    "mbox_rx_worker",
    "mbox_poll_handle",
    "uc_irq_handle",
    "uc_wakeup",
)

REQUIRED_DYNAMIC_DEBUG_SELECTORS = (
    "file aie2_message.c line 1076 +p",
    "file amdxdna_mailbox.c line 191 +p",
    "file amdxdna_mailbox.c line 235 +p",
    "file amdxdna_mailbox.c line 270 +p",
    "file amdxdna_mailbox.c line 460 +p",
    "file amdxdna_mailbox_helper.c line 48 +p",
    "file aie2_ctx.c line 300 +p",
    "file aie2_ctx.c line 356 +p",
)


@dataclass(frozen=True)
class QualifiedModuleManifest:
    candidate_path: Path
    candidate_sha256: str
    original_path: Path
    original_sha256: str
    source_repository: str = "synthetic"
    source_revision: str = "synthetic"
    build_recipe_sha256: str = "0" * 64
    kernel_release: str = "synthetic"
    tdr_parameter_present: bool = True
    trace_events: tuple[str, ...] = REQUIRED_TRACE_EVENTS
    dynamic_debug_selectors: tuple[str, ...] = REQUIRED_DYNAMIC_DEBUG_SELECTORS


@dataclass(frozen=True)
class CaptureRequest:
    campaign_id: str
    seed: int
    campaign_dir: Path
    status_path: Path
    module: QualifiedModuleManifest
    schedule: tuple[ScheduleEntry, ...]
    firmware: Path
    executable: Path
    xclbin: Path
    instructions: Path
    location_plan: Path | None = None

    @classmethod
    def synthetic(
        cls, root: Path, schedule: Sequence[ScheduleEntry]
    ) -> "CaptureRequest":
        return cls(
            campaign_id="campaign.synthetic",
            seed=0,
            campaign_dir=root,
            status_path=root / "status.json",
            module=QualifiedModuleManifest(
                root / "candidate.ko",
                "a" * 64,
                root / "original.ko",
                "b" * 64,
                trace_events=REQUIRED_TRACE_EVENTS,
                dynamic_debug_selectors=REQUIRED_DYNAMIC_DEBUG_SELECTORS,
            ),
            schedule=tuple(schedule),
            firmware=root / "firmware.bin",
            executable=root / "test.exe",
            xclbin=root / "aie.xclbin",
            instructions=root / "insts.bin",
        )


_MODULE_MANIFEST_FIELDS = {
    "candidate_path",
    "candidate_sha256",
    "original_path",
    "original_sha256",
    "source_repository",
    "source_revision",
    "build_recipe_sha256",
    "kernel_release",
    "tdr_parameter_present",
    "trace_events",
    "dynamic_debug_selectors",
}


def _qualified_module_from_data(data: Mapping[str, Any]) -> QualifiedModuleManifest:
    if set(data) != _MODULE_MANIFEST_FIELDS:
        raise ValueError("qualified-module manifest fields are missing or unexpected")
    module = QualifiedModuleManifest(
        candidate_path=Path(data["candidate_path"]),
        candidate_sha256=data["candidate_sha256"],
        original_path=Path(data["original_path"]),
        original_sha256=data["original_sha256"],
        source_repository=data["source_repository"],
        source_revision=data["source_revision"],
        build_recipe_sha256=data["build_recipe_sha256"],
        kernel_release=data["kernel_release"],
        tdr_parameter_present=data["tdr_parameter_present"],
        trace_events=tuple(data["trace_events"]),
        dynamic_debug_selectors=tuple(data["dynamic_debug_selectors"]),
    )
    for label, value in (
        ("candidate", module.candidate_sha256),
        ("original", module.original_sha256),
        ("build recipe", module.build_recipe_sha256),
    ):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"{label} SHA-256 is invalid")
    if module.trace_events != REQUIRED_TRACE_EVENTS:
        raise ValueError("qualified module trace-event surface is incomplete")
    if module.dynamic_debug_selectors != REQUIRED_DYNAMIC_DEBUG_SELECTORS:
        raise ValueError("qualified module dynamic-debug surface is incomplete")
    if (
        type(module.tdr_parameter_present) is not bool
        or not module.tdr_parameter_present
    ):
        raise ValueError("qualified module lacks normal-TDR capability")
    return module


def load_qualified_module_manifest(path: Path) -> QualifiedModuleManifest:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("qualified-module manifest must be an object")
    return _qualified_module_from_data(data)


def validate_location_plan(data: Mapping[str, Any]) -> tuple[str, ...]:
    if set(data) != {"roots"} or not isinstance(data.get("roots"), list):
        return ("location plan must contain only a roots array",)
    roots = data["roots"]
    errors: list[str] = []
    if len(roots) != 3:
        errors.append("location plan requires one primary and two replica roots")
    aliases: list[str] = []
    paths: list[str] = []
    domains: list[str] = []
    for index, root in enumerate(roots):
        if not isinstance(root, dict) or set(root) != {
            "alias",
            "path",
            "failure_domain_id",
            "bundles",
        }:
            errors.append(f"location root {index} fields are missing or unexpected")
            continue
        for field, target in (
            ("alias", aliases),
            ("path", paths),
            ("failure_domain_id", domains),
        ):
            value = root[field]
            if not isinstance(value, str) or not value.strip():
                errors.append(f"location root {index} {field} must be nonblank")
            else:
                target.append(value)
        if (
            isinstance(root["path"], str)
            and root["path"]
            and not Path(root["path"]).is_absolute()
        ):
            errors.append(f"location root {index} path must be absolute")
        if not isinstance(root["bundles"], list):
            errors.append(f"location root {index} bundles must be an array")
    for label, values in (
        ("alias", aliases),
        ("path", paths),
        ("failure domain", domains),
    ):
        if len(values) != len(set(values)):
            errors.append(f"location root {label}s must be distinct")
    return tuple(errors)


def load_capture_request(path: Path) -> CaptureRequest:
    data = json.loads(path.read_text())
    expected = {
        "campaign_id",
        "seed",
        "campaign_dir",
        "status_path",
        "module",
        "schedule",
        "firmware",
        "executable",
        "xclbin",
        "instructions",
        "location_plan",
    }
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError("capture request fields are missing or unexpected")
    arms = {TREATMENT.name: TREATMENT, CONTROL.name: CONTROL}
    schedule: list[ScheduleEntry] = []
    for item in data["schedule"]:
        if not isinstance(item, dict) or set(item) != {"ordinal", "repetition", "arm"}:
            raise ValueError("capture schedule entry fields are missing or unexpected")
        if item["arm"] not in arms:
            raise ValueError(f"unknown capture arm: {item['arm']}")
        schedule.append(
            ScheduleEntry(item["ordinal"], item["repetition"], arms[item["arm"]])
        )
    if tuple(entry.ordinal for entry in schedule) != tuple(range(len(schedule))):
        raise ValueError("capture schedule ordinals are not contiguous")
    location_plan = data["location_plan"]
    return CaptureRequest(
        campaign_id=data["campaign_id"],
        seed=data["seed"],
        campaign_dir=Path(data["campaign_dir"]),
        status_path=Path(data["status_path"]),
        module=_qualified_module_from_data(data["module"]),
        schedule=tuple(schedule),
        firmware=Path(data["firmware"]),
        executable=Path(data["executable"]),
        xclbin=Path(data["xclbin"]),
        instructions=Path(data["instructions"]),
        location_plan=Path(location_plan) if location_plan is not None else None,
    )


def validate_privileged_request(
    request: CaptureRequest,
    pkexec_uid: str | None,
    campaign_owner_uid: int,
) -> tuple[str, ...]:
    errors: list[str] = []
    if pkexec_uid is None or not pkexec_uid.isdecimal():
        errors.append("PKEXEC_UID is missing or invalid")
        return tuple(errors)
    uid = int(pkexec_uid)
    if uid != campaign_owner_uid:
        errors.append("PKEXEC_UID does not own the campaign directory")
    root = request.campaign_dir.resolve()
    for label, path in [("status", request.status_path)]:
        if not path.resolve().is_relative_to(root):
            errors.append(f"{label} path escapes the campaign directory")
    return tuple(errors)


@dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...]
    environment: tuple[tuple[str, str], ...] = ()
    force_cmdlist: str | None = None


@dataclass(frozen=True)
class TransactionPlan:
    setup: tuple[CommandSpec, ...]
    trace_actions: tuple[str, ...]
    runs: tuple[CommandSpec, ...]
    cleanup: tuple[CommandSpec, ...]
    rollback: tuple[CommandSpec, ...]


@dataclass(frozen=True)
class TransactionExecution:
    campaign: CampaignClassification
    runs: tuple[RunClassification, ...]
    cleanup_ok: bool
    rollback_attempted: bool


@dataclass(frozen=True)
class PreparedCapture:
    request: CaptureRequest
    request_path: Path
    pkexec_argv: tuple[str, ...]


def build_transaction_plan(
    request: CaptureRequest, uid: int, *, submitted: bool
) -> TransactionPlan:
    trace_instance = f"npu1-fw-{request.campaign_id}"
    account = pwd.getpwuid(uid)
    setup = (
        CommandSpec(("modprobe", "-r", "amdxdna")),
        CommandSpec(
            (
                "insmod",
                str(request.module.candidate_path),
                f"tdr_timeout_ms={VERTICAL_SPEC.tdr_timeout_ms}",
            )
        ),
    )
    trace_actions = (
        f"create tracefs instance {trace_instance}",
        *(f"enable amdxdna event {event}" for event in request.module.trace_events),
        *(
            f"enable dynamic-debug selector {selector}"
            for selector in request.module.dynamic_debug_selectors
        ),
        "restore exact dynamic-debug selectors",
    )
    runs = tuple(
        CommandSpec(
            (
                "runuser",
                "-u",
                account.pw_name,
                "--",
                "env",
                "-i",
                f"HOME={account.pw_dir}",
                f"USER={account.pw_name}",
                f"LOGNAME={account.pw_name}",
                "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "XILINX_XRT=/opt/xilinx/xrt",
                "timeout",
                "--signal=TERM",
                "--kill-after=2s",
                f"{RUN_TIMEOUT_SECONDS}s",
                str(request.executable),
                "-x",
                str(request.xclbin),
                "-k",
                "MLIR_AIE",
                "-i",
                str(request.instructions),
            ),
            force_cmdlist=entry.arm.force_cmdlist,
        )
        for entry in request.schedule
    )
    cleanup = (
        CommandSpec(("rmdir", f"/sys/kernel/tracing/instances/{trace_instance}")),
    )
    rollback = (
        (
            CommandSpec(("modprobe", "-r", "amdxdna")),
            CommandSpec(("insmod", str(request.module.original_path))),
        )
        if not submitted
        else ()
    )
    return TransactionPlan(setup, trace_actions, runs, cleanup, rollback)


def execute_capture_transaction(
    request: CaptureRequest,
    uid: int,
    runner,
    capture_reader,
    set_force_cmdlist,
    trace_action,
) -> TransactionExecution:
    plan = build_transaction_plan(request, uid, submitted=False)
    results: list[RunClassification] = []
    submitted = False
    failed_before_traffic: str | None = None

    for command_spec in plan.setup:
        result = runner(command_spec)
        if result.timed_out or result.returncode != 0:
            failed_before_traffic = "privileged setup command failed"
            break
    if failed_before_traffic is None:
        for action in plan.trace_actions[:-1]:
            if not trace_action(action):
                failed_before_traffic = f"trace setup failed: {action}"
                break

    if failed_before_traffic is None:
        for entry, command_spec in zip(request.schedule, plan.runs, strict=True):
            readback = set_force_cmdlist(entry.arm.force_cmdlist)
            if readback != entry.arm.force_cmdlist:
                results.append(
                    RunClassification(
                        entry.run_id,
                        Outcome.PROVENANCE_FAILURE,
                        ("force_cmdlist readback mismatch",),
                        None,
                        (),
                        (),
                    )
                )
                break
            submitted = True
            command_result = runner(command_spec)
            result = classify_run(entry, command_result, capture_reader(entry))
            results.append(result)
            if result.outcome != Outcome.SUCCESS:
                break

    cleanup_ok = trace_action(plan.trace_actions[-1])
    for command_spec in plan.cleanup:
        result = runner(command_spec)
        cleanup_ok &= not result.timed_out and result.returncode == 0

    rollback_attempted = not submitted and (
        failed_before_traffic is not None
        or any(result.outcome != Outcome.SUCCESS for result in results)
    )
    if rollback_attempted:
        for command_spec in plan.rollback:
            result = runner(command_spec)
            cleanup_ok &= not result.timed_out and result.returncode == 0

    if failed_before_traffic is not None:
        campaign = CampaignClassification(
            Outcome.INFRASTRUCTURE_FAILURE,
            (),
            request.schedule[0].run_id if request.schedule else None,
            (failed_before_traffic,),
        )
    else:
        campaign = classify_campaign(request.schedule, results)
    if not cleanup_ok:
        campaign = CampaignClassification(
            Outcome.INFRASTRUCTURE_FAILURE
            if campaign.outcome == Outcome.SUCCESS
            else campaign.outcome,
            campaign.completed_run_ids,
            campaign.failed_run_id,
            (*campaign.reasons, "capture cleanup or restoration failed"),
        )
    return TransactionExecution(
        campaign, tuple(results), cleanup_ok, rollback_attempted
    )


_CAMPAIGN_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def capture_preflight_errors(
    location_plan: Path,
    module: QualifiedModuleManifest,
    files: Mapping[str, Path],
    snapshot: PreflightSnapshot,
    spec: VerticalSpec = VERTICAL_SPEC,
) -> tuple[str, ...]:
    errors = list(preflight_errors(spec, snapshot))
    for name, expected in spec.file_hashes.items():
        path = files.get(name)
        if path is None or not path.is_file():
            errors.append(f"{name} file is missing")
        elif sha256_file(path) != expected:
            errors.append(f"{name} file bytes do not match the frozen pin")
    for label, path, expected in [
        ("candidate module", module.candidate_path, module.candidate_sha256),
        ("original module", module.original_path, module.original_sha256),
    ]:
        if not path.is_file() or sha256_file(path) != expected:
            errors.append(f"{label} bytes do not match the qualified manifest")
    if not module.source_repository.strip() or not module.source_revision.strip():
        errors.append("candidate module source provenance is incomplete")
    if not re.fullmatch(r"[0-9a-f]{64}", module.build_recipe_sha256):
        errors.append("candidate module build recipe SHA-256 is invalid")
    if snapshot.kernel_release:
        errors.extend(module_preflight_errors(module, snapshot))
    if (
        not module.tdr_parameter_present
        or module.trace_events != REQUIRED_TRACE_EVENTS
        or module.dynamic_debug_selectors != REQUIRED_DYNAMIC_DEBUG_SELECTORS
    ):
        errors.append(
            "candidate module debug or normal-TDR capabilities are incomplete"
        )
    if not location_plan.is_file():
        errors.append("location plan is missing")
    else:
        try:
            location_errors = validate_location_plan(
                json.loads(location_plan.read_text())
            )
        except (json.JSONDecodeError, OSError) as error:
            location_errors = (f"location plan is unreadable: {error}",)
        errors.extend(location_errors)
    return tuple(sorted(set(errors)))


def prepare_capture(
    repository_root: Path,
    campaign_id: str,
    seed: int,
    batch: bool,
    location_plan: Path,
    module: QualifiedModuleManifest,
    files: Mapping[str, Path],
    snapshot: PreflightSnapshot,
    spec: VerticalSpec = VERTICAL_SPEC,
) -> PreparedCapture:
    if not _CAMPAIGN_ID.fullmatch(campaign_id):
        raise ValueError("campaign ID must be a stable lowercase path component")
    errors = capture_preflight_errors(location_plan, module, files, snapshot, spec)
    if errors:
        raise ValueError("; ".join(errors))

    campaign_dir = (
        repository_root
        / "build"
        / "experiments"
        / "npu1-firmware-evidence"
        / campaign_id
    )
    if campaign_dir.exists():
        if not campaign_dir.is_dir() or any(campaign_dir.iterdir()):
            raise FileExistsError(f"campaign directory is not empty: {campaign_dir}")
    else:
        campaign_dir.mkdir(parents=True)
    schedule = repetition_schedule(50, 50, seed) if batch else vertical_schedule(seed)
    request = CaptureRequest(
        campaign_id=campaign_id,
        seed=seed,
        campaign_dir=campaign_dir,
        status_path=campaign_dir / "status.json",
        module=module,
        schedule=schedule,
        firmware=files["firmware"].resolve(),
        executable=files["executable"].resolve(),
        xclbin=files["xclbin"].resolve(),
        instructions=files["instructions"].resolve(),
        location_plan=location_plan.resolve(),
    )
    request_path = campaign_dir / "capture-request.json"
    _write_capture_request(request_path, request)
    request_sha256 = sha256_file(request_path)
    return PreparedCapture(
        request,
        request_path,
        (
            "pkexec",
            sys.executable,
            str(Path(__file__).resolve()),
            "_privileged",
            str(request_path),
            request_sha256,
        ),
    )


def _write_capture_request(path: Path, request: CaptureRequest) -> None:
    data = {
        "campaign_id": request.campaign_id,
        "seed": request.seed,
        "campaign_dir": str(request.campaign_dir),
        "status_path": str(request.status_path),
        "module": {
            "candidate_path": str(request.module.candidate_path),
            "candidate_sha256": request.module.candidate_sha256,
            "original_path": str(request.module.original_path),
            "original_sha256": request.module.original_sha256,
            "source_repository": request.module.source_repository,
            "source_revision": request.module.source_revision,
            "build_recipe_sha256": request.module.build_recipe_sha256,
            "kernel_release": request.module.kernel_release,
            "tdr_parameter_present": request.module.tdr_parameter_present,
            "trace_events": list(request.module.trace_events),
            "dynamic_debug_selectors": list(request.module.dynamic_debug_selectors),
        },
        "schedule": [
            {
                "ordinal": entry.ordinal,
                "repetition": entry.repetition,
                "arm": entry.arm.name,
            }
            for entry in request.schedule
        ],
        "firmware": str(request.firmware),
        "executable": str(request.executable),
        "xclbin": str(request.xclbin),
        "instructions": str(request.instructions),
        "location_plan": str(request.location_plan) if request.location_plan else None,
    }
    write_terminal_status(path, data)


def write_terminal_status(path: Path, status: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(status, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def transient_service_argv(
    script: Path,
    campaign_id: str,
    seed: int,
    location_plan: Path,
    module_manifest: Path,
) -> tuple[str, ...]:
    return (
        "systemd-run",
        "--user",
        "--collect",
        f"--unit=npu1-firmware-evidence-{campaign_id}",
        sys.executable,
        str(script),
        "batch",
        "--campaign-id",
        campaign_id,
        "--seed",
        str(seed),
        "--location-plan",
        str(location_plan),
        "--module-manifest",
        str(module_manifest),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_command(argv: Sequence[str], timeout: int = 10) -> CommandResult:
    command = tuple(str(part) for part in argv)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=False,
            timeout=timeout,
            check=False,
        )
        return CommandResult(
            command,
            result.returncode,
            result.stdout,
            result.stderr,
            result.returncode == 124 and "timeout" in command,
        )
    except subprocess.TimeoutExpired as error:
        stdout = (
            error.stdout.decode()
            if isinstance(error.stdout, bytes)
            else error.stdout or ""
        )
        stderr = (
            error.stderr.decode()
            if isinstance(error.stderr, bytes)
            else error.stderr or ""
        )
        return CommandResult(command, 124, stdout, stderr, True)


def _command_stdout(argv: Sequence[str], timeout: int = 10) -> str:
    result = _run_command(argv, timeout)
    if result.returncode or result.timed_out:
        raise RuntimeError(
            f"{' '.join(result.argv)} failed: {result.stderr.strip() or result.returncode}"
        )
    return result.stdout.strip()


def _workspace_root() -> Path:
    return next(
        parent
        for parent in Path(__file__).resolve().parents
        if parent.name == "npu-work"
    )


def frozen_input_paths() -> dict[str, Path]:
    workspace = _workspace_root()
    workload = workspace / "mlir-aie/build/test/npu-xrt/add_one_using_dma"
    return {
        "firmware": Path("/usr/lib/firmware") / VERTICAL_SPEC.firmware_logical_path,
        "xclbin": workload / "chess/aie.xclbin",
        "instructions": workload / "chess/insts.bin",
        "executable": workload / "test.exe",
    }


def _physical_npu() -> tuple[str, Path]:
    matches = []
    for device in Path("/sys/bus/pci/devices").iterdir():
        try:
            vendor = (device / "vendor").read_text().strip().removeprefix("0x")
            product = (device / "device").read_text().strip().removeprefix("0x")
        except OSError:
            continue
        if f"{vendor}:{product}".lower() == VERTICAL_SPEC.pci_id:
            matches.append(device)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one physical {VERTICAL_SPEC.pci_id}, found {len(matches)}"
        )
    return matches[0].name, matches[0]


def _device_node_for_bdf(bdf: str) -> Path:
    for accel in Path("/sys/class/accel").iterdir():
        if bdf in str(accel.resolve()):
            return Path("/dev/accel") / accel.name
    raise RuntimeError(f"no accel device class entry belongs to {bdf}")


def _active_npu_clients(device_node: Path) -> int:
    result = _run_command(("fuser", "-s", str(device_node)))
    if result.returncode == 0:
        return 1
    if result.returncode == 1:
        return 0
    raise RuntimeError(f"fuser failed for {device_node}: {result.stderr.strip()}")


def _modinfo(field: str, module: str | Path) -> str:
    return _command_stdout(("modinfo", "-F", field, str(module)))


def _resolved_xrt_coreutil(executable: Path) -> str:
    output = _command_stdout(("ldd", str(executable)))
    matches = re.findall(r"^\s*libxrt_coreutil\.so\.2 => (\S+)", output, re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError(
            "host executable did not resolve exactly one libxrt_coreutil.so.2"
        )
    return str(Path(matches[0]).resolve())


def collect_preflight_snapshot(
    module: QualifiedModuleManifest,
    files: Mapping[str, Path],
) -> PreflightSnapshot:
    bdf, pci = _physical_npu()
    device_node = _device_node_for_bdf(bdf)
    loaded_path = Path(_modinfo("filename", "amdxdna")).resolve()
    return PreflightSnapshot(
        file_hashes={
            name: sha256_file(path)
            for name, path in files.items()
            if name in VERTICAL_SPEC.file_hashes and path.is_file()
        },
        environment={
            name: os.environ[name]
            for name in ("XDNA_EMU", "XDNA_EMU_RUNTIME")
            if name in os.environ
        },
        pci_id=(
            f"{(pci / 'vendor').read_text().strip().removeprefix('0x')}:"
            f"{(pci / 'device').read_text().strip().removeprefix('0x')}"
        ).lower(),
        active_clients=_active_npu_clients(device_node),
        tdr_parameter_present=module.tdr_parameter_present,
        kernel_release=os.uname().release,
        loaded_module_path=str(loaded_path),
        loaded_module_sha256=sha256_file(loaded_path),
        loaded_module_srcversion=Path("/sys/module/amdxdna/srcversion")
        .read_text()
        .strip(),
        original_module_srcversion=_modinfo("srcversion", module.original_path),
        xrt_coreutil_path=_resolved_xrt_coreutil(files["executable"]),
        device_node_present=device_node.is_char_device(),
        power_control=(pci / "power/control").read_text().strip(),
    )


def _schedule_json(schedule: Sequence[ScheduleEntry]) -> list[dict[str, Any]]:
    return [
        {
            "run_id": entry.run_id,
            "ordinal": entry.ordinal,
            "repetition": entry.repetition,
            "arm": entry.arm.name,
            "force_cmdlist": entry.arm.force_cmdlist,
            "execute_opcode": f"0x{entry.arm.execute_opcode:x}",
        }
        for entry in schedule
    ]


def run_coordinator(
    command: str,
    campaign_id: str,
    seed: int,
    location_plan: Path,
    module_manifest: Path,
) -> int:
    repository = Path(__file__).resolve().parents[1]
    module = load_qualified_module_manifest(module_manifest)
    files = frozen_input_paths()
    snapshot = collect_preflight_snapshot(module, files)
    prepared = prepare_capture(
        repository,
        campaign_id,
        seed,
        command == "batch",
        location_plan,
        module,
        files,
        snapshot,
    )
    preflight = {
        "campaign_id": campaign_id,
        "seed": seed,
        "schedule": _schedule_json(prepared.request.schedule),
        "request_path": str(prepared.request_path),
        "request_sha256": prepared.pkexec_argv[-1],
        "qualified_module_sha256": module.candidate_sha256,
        "original_module_sha256": module.original_sha256,
        "file_hashes": dict(snapshot.file_hashes),
        "snapshot": {**snapshot.__dict__, "file_hashes": dict(snapshot.file_hashes)},
        "pkexec_argv": list(prepared.pkexec_argv),
    }
    write_terminal_status(prepared.request.campaign_dir / "preflight.json", preflight)
    write_terminal_status(
        prepared.request.status_path,
        {
            "state": "prepared",
            "campaign_id": campaign_id,
            "request_sha256": prepared.pkexec_argv[-1],
        },
    )
    json.dump(preflight, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    sys.stdout.flush()
    result = subprocess.run(prepared.pkexec_argv, shell=False, check=False)
    if (
        result.returncode
        and json.loads(prepared.request.status_path.read_text()).get("state")
        == "prepared"
    ):
        write_terminal_status(
            prepared.request.status_path,
            {
                "state": "authorization_or_privileged_failure",
                "campaign_id": campaign_id,
                "returncode": result.returncode,
            },
        )
    return result.returncode


def _elf_build_id(module: Path) -> str:
    output = _command_stdout(("readelf", "-n", str(module)))
    matches = re.findall(r"Build ID: ([0-9a-f]+)", output)
    if len(matches) != 1:
        raise RuntimeError(f"could not resolve one build ID for {module}")
    return matches[0]


def _loaded_build_id() -> str:
    data = Path("/sys/module/amdxdna/notes/.note.gnu.build-id").read_bytes()
    if len(data) < 16:
        raise RuntimeError("loaded amdxdna build-ID note is truncated")
    namesz, descsz, note_type = struct.unpack_from("=III", data)
    description = 12 + ((namesz + 3) & ~3)
    if note_type != 3 or data[12 : 12 + namesz].rstrip(b"\0") != b"GNU":
        raise RuntimeError("loaded amdxdna build-ID note is malformed")
    return data[description : description + descsz].hex()


def _loaded_srcversion() -> str | None:
    try:
        return Path("/sys/module/amdxdna/srcversion").read_text().strip()
    except OSError:
        return None


def _write_owned_text(path: Path, text: str, uid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        output.write(text)
        output.flush()
        os.fsync(output.fileno())
    os.chown(path, uid, -1)


def _write_owned_json(path: Path, data: Mapping[str, Any], uid: int) -> None:
    write_terminal_status(path, data)
    os.chown(path, uid, -1)


def _command_result_json(result: CommandResult) -> dict[str, Any]:
    return {
        "argv": list(result.argv),
        "returncode": result.returncode,
        "timed_out": result.timed_out,
    }


def _classification_json(result: RunClassification) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "outcome": result.outcome.value,
        "reasons": list(result.reasons),
        "execute_opcode": (
            f"0x{result.execute_opcode:x}"
            if result.execute_opcode is not None
            else None
        ),
        "output_values": list(result.output_values),
        "unknown_success_words": list(result.unknown_success_words),
    }


def run_privileged_capture(request_path: Path, request_sha256: str) -> int:
    if os.geteuid() != 0:
        print("privileged capture must run as root", file=sys.stderr)
        return 2
    if (
        not re.fullmatch(r"[0-9a-f]{64}", request_sha256)
        or sha256_file(request_path) != request_sha256
    ):
        print("capture request SHA-256 mismatch", file=sys.stderr)
        return 2

    request = load_capture_request(request_path)
    owner_uid = request.campaign_dir.stat().st_uid
    errors = list(
        validate_privileged_request(request, os.environ.get("PKEXEC_UID"), owner_uid)
    )
    if (
        request_path.stat().st_uid != owner_uid
        or not request_path.resolve().is_relative_to(request.campaign_dir.resolve())
    ):
        errors.append(
            "capture request is not owned and confined to the campaign directory"
        )
    expected_schedule = (
        vertical_schedule(request.seed)
        if len(request.schedule) == 2
        else repetition_schedule(50, 50, request.seed)
    )
    if request.schedule != expected_schedule:
        errors.append(
            "capture schedule does not match its recorded seed and campaign size"
        )
    if request.location_plan is None:
        errors.append("capture request has no location plan")

    files = {
        "firmware": request.firmware,
        "xclbin": request.xclbin,
        "instructions": request.instructions,
        "executable": request.executable,
    }
    try:
        snapshot = collect_preflight_snapshot(request.module, files)
        if request.location_plan is not None:
            errors.extend(
                capture_preflight_errors(
                    request.location_plan, request.module, files, snapshot
                )
            )
    except (OSError, RuntimeError, ValueError) as error:
        errors.append(str(error))
        snapshot = None
    if errors:
        _write_owned_json(
            request.status_path,
            {
                "state": "pretraffic_preflight_failed",
                "campaign_id": request.campaign_id,
                "errors": sorted(set(errors)),
            },
            owner_uid,
        )
        return 2

    try:
        bdf, pci = _physical_npu()
        device_node = _device_node_for_bdf(bdf)
        trace_root = Path("/sys/kernel/tracing")
        trace_instance = trace_root / "instances" / f"npu1-fw-{request.campaign_id}"
        dynamic_control = Path("/proc/dynamic_debug/control")
        force_parameter = Path("/sys/module/amdxdna/parameters/force_cmdlist")
        candidate_srcversion = _modinfo("srcversion", request.module.candidate_path)
        candidate_build_id = _elf_build_id(request.module.candidate_path)
        original_srcversion = _modinfo("srcversion", request.module.original_path)
    except (OSError, RuntimeError, ValueError) as error:
        _write_owned_json(
            request.status_path,
            {
                "state": "pretraffic_preflight_failed",
                "campaign_id": request.campaign_id,
                "errors": [str(error)],
            },
            owner_uid,
        )
        return 2

    _write_owned_json(
        request.status_path,
        {
            "state": "running",
            "campaign_id": request.campaign_id,
            "seed": request.seed,
            "schedule": _schedule_json(request.schedule),
        },
        owner_uid,
    )
    debug_states: dict[str, bool] = {}
    force_initial: str | None = None
    trace_subsystem: str | None = None
    run_index = 0
    unload_calls = 0
    captures: dict[str, RawCaptureIndex] = {}

    def marker(value: str) -> None:
        with (trace_instance / "trace_marker").open("w") as output:
            output.write(value + "\n")
        with Path("/dev/kmsg").open("w") as output:
            output.write(value + "\n")

    def dmesg() -> str:
        return _command_stdout(("dmesg", "--raw"))

    def run_one(spec: CommandSpec) -> CommandResult:
        nonlocal run_index
        entry = request.schedule[run_index]
        run_index += 1
        run_dir = request.campaign_dir / "raw" / entry.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        os.chown(run_dir, owner_uid, -1)
        before = dmesg()
        (trace_instance / "tracing_on").write_text("0\n")
        (trace_instance / "trace").write_text("")
        (trace_instance / "tracing_on").write_text("1\n")
        begin = f"NPU1_FW_BEGIN {entry.run_id}"
        end = f"NPU1_FW_END {entry.run_id}"
        marker_error = ""
        start_ns = time.time_ns()
        try:
            marker(begin)
            result = _run_command(spec.argv, RUN_TIMEOUT_SECONDS + 10)
        except (OSError, RuntimeError) as error:
            marker_error = str(error)
            result = CommandResult(spec.argv, 125, "", marker_error)
        finally:
            try:
                marker(end)
            except OSError as error:
                marker_error = f"{marker_error}; {error}".strip("; ")
            (trace_instance / "tracing_on").write_text("0\n")
        end_ns = time.time_ns()
        trace_text = (trace_instance / "trace").read_text()
        after = dmesg()
        if marker_error:
            result = CommandResult(
                result.argv,
                result.returncode or 125,
                result.stdout,
                f"{result.stderr}\n{marker_error}".strip(),
                result.timed_out,
            )
        active_clients = _active_npu_clients(device_node)
        _write_owned_text(run_dir / "stdout.log", result.stdout, owner_uid)
        _write_owned_text(run_dir / "stderr.log", result.stderr, owner_uid)
        _write_owned_text(run_dir / "trace.log", trace_text, owner_uid)
        _write_owned_text(run_dir / "dmesg-before.log", before, owner_uid)
        _write_owned_text(run_dir / "dmesg-after.log", after, owner_uid)
        _write_owned_text(
            run_dir / "exit-code.txt", f"{result.returncode}\n", owner_uid
        )
        _write_owned_json(
            run_dir / "command.json",
            {
                **_command_result_json(result),
                "start_time_ns": start_ns,
                "end_time_ns": end_ns,
                "force_cmdlist": entry.arm.force_cmdlist,
                "environment": {
                    key: value
                    for part in spec.argv
                    if "=" in part
                    and part.split("=", 1)[0]
                    in {"HOME", "USER", "LOGNAME", "PATH", "XILINX_XRT"}
                    for key, value in (part.split("=", 1),)
                },
            },
            owner_uid,
        )
        capture = derive_capture_index(
            entry,
            (run_dir / "trace.log").read_text(),
            (run_dir / "dmesg-before.log").read_text(),
            (run_dir / "dmesg-after.log").read_text(),
            teardown_ok=active_clients == 0,
        )
        captures[entry.run_id] = capture
        _write_owned_json(
            request.campaign_dir / "derived" / f"{entry.run_id}.capture-index.json",
            {
                **capture.__dict__,
                "lifecycle_text": "raw/trace.log",
            },
            owner_uid,
        )
        return result

    def runner(spec: CommandSpec) -> CommandResult:
        nonlocal unload_calls
        if spec.argv[:3] == ("modprobe", "-r", "amdxdna"):
            unload_calls += 1
            if unload_calls > 1 and _loaded_srcversion() in (None, original_srcversion):
                return CommandResult(spec.argv, 0, "", "")
            return _run_command(spec.argv, 30)
        if spec.argv and spec.argv[0] == "insmod":
            module_path = Path(spec.argv[1])
            expected_srcversion = (
                candidate_srcversion
                if module_path == request.module.candidate_path
                else original_srcversion
            )
            if _loaded_srcversion() == expected_srcversion:
                return CommandResult(spec.argv, 0, "", "")
            result = _run_command(spec.argv, 30)
            if result.returncode == 0:
                settled = _run_command(("udevadm", "settle", "--timeout=5"), 10)
                if settled.returncode:
                    return CommandResult(
                        spec.argv, settled.returncode, result.stdout, settled.stderr
                    )
            return result
        if spec.argv and spec.argv[0] == "runuser":
            try:
                return run_one(spec)
            except Exception as error:
                try:
                    if trace_instance.exists():
                        (trace_instance / "tracing_on").write_text("0\n")
                except OSError:
                    pass
                return CommandResult(
                    spec.argv, 125, "", f"capture adapter failed: {error}"
                )
        if spec.argv and spec.argv[0] == "rmdir":
            try:
                path = Path(spec.argv[1])
                if path.exists():
                    path.rmdir()
                return CommandResult(spec.argv, 0, "", "")
            except OSError as error:
                return CommandResult(spec.argv, 1, "", str(error))
        return _run_command(spec.argv)

    def trace_action(action: str) -> bool:
        nonlocal debug_states, force_initial, trace_subsystem
        try:
            if action.startswith("create tracefs instance "):
                if (
                    sha256_file(request.module.candidate_path)
                    != request.module.candidate_sha256
                    or _loaded_srcversion() != candidate_srcversion
                    or _loaded_build_id() != candidate_build_id
                    or Path("/sys/module/amdxdna/parameters/tdr_timeout_ms")
                    .read_text()
                    .strip()
                    != str(VERTICAL_SPEC.tdr_timeout_ms)
                    or not force_parameter.is_file()
                    or (pci / "power/control").read_text().strip() != "on"
                    or not device_node.is_char_device()
                ):
                    return False
                force_initial = force_parameter.read_text().strip()
                debug_states = dynamic_debug_print_states(
                    dynamic_control.read_text(), request.module.dynamic_debug_selectors
                )
                trace_instance.mkdir()
                (trace_instance / "tracing_on").write_text("0\n")
                (trace_instance / "trace").write_text("")
                (trace_instance / "buffer_size_kb").write_text("4096\n")
                subsystems = [
                    name
                    for name in ("amdxdna", "amdxdna_trace")
                    if (trace_instance / "events" / name).is_dir()
                ]
                if len(subsystems) != 1:
                    return False
                trace_subsystem = subsystems[0]
                _write_owned_json(
                    request.campaign_dir / "runtime-preflight.json",
                    {
                        "candidate_sha256": request.module.candidate_sha256,
                        "candidate_srcversion": candidate_srcversion,
                        "candidate_build_id": candidate_build_id,
                        "tdr_timeout_ms": VERTICAL_SPEC.tdr_timeout_ms,
                        "force_cmdlist_initial": force_initial,
                        "dynamic_debug_print_states": debug_states,
                        "trace_subsystem": trace_subsystem,
                        "pci_bdf": bdf,
                        "power_control": "on",
                    },
                    owner_uid,
                )
                return True
            if action.startswith("enable amdxdna event "):
                event = action.rsplit(" ", 1)[1]
                if event not in request.module.trace_events or trace_subsystem is None:
                    return False
                enable = trace_instance / "events" / trace_subsystem / event / "enable"
                enable.write_text("1\n")
                return enable.read_text().strip() == "1"
            if action.startswith("enable dynamic-debug selector "):
                selector = action.removeprefix("enable dynamic-debug selector ")
                if selector not in request.module.dynamic_debug_selectors:
                    return False
                dynamic_control.write_text(selector + "\n")
                return dynamic_debug_print_states(
                    dynamic_control.read_text(), (selector,)
                )[selector]
            if action == "restore exact dynamic-debug selectors":
                ok = True
                try:
                    if trace_instance.exists():
                        (trace_instance / "tracing_on").write_text("0\n")
                except OSError:
                    ok = False
                for restore in dynamic_debug_restore_commands(debug_states):
                    try:
                        dynamic_control.write_text(restore + "\n")
                    except OSError:
                        ok = False
                try:
                    if (
                        debug_states
                        and dynamic_debug_print_states(
                            dynamic_control.read_text(), tuple(debug_states)
                        )
                        != debug_states
                    ):
                        ok = False
                    if force_initial is not None and force_parameter.exists():
                        force_parameter.write_text(force_initial + "\n")
                        ok &= force_parameter.read_text().strip() == force_initial
                except (OSError, ValueError):
                    ok = False
                try:
                    _write_owned_json(
                        request.campaign_dir / "cleanup.json",
                        {
                            "dynamic_debug_restored": (
                                not debug_states
                                or dynamic_debug_print_states(
                                    dynamic_control.read_text(), tuple(debug_states)
                                )
                                == debug_states
                            ),
                            "force_cmdlist_initial": force_initial,
                            "force_cmdlist_after": (
                                force_parameter.read_text().strip()
                                if force_parameter.exists()
                                else None
                            ),
                        },
                        owner_uid,
                    )
                except (OSError, ValueError):
                    ok = False
                return ok
        except (OSError, RuntimeError, ValueError):
            return False
        return False

    def set_force_cmdlist(value: str) -> str:
        try:
            if _active_npu_clients(device_node):
                return ""
            entry = request.schedule[run_index]
            run_dir = request.campaign_dir / "raw" / entry.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            os.chown(run_dir, owner_uid, -1)
            before = force_parameter.read_text().strip()
            force_parameter.write_text(value + "\n")
            readback = force_parameter.read_text().strip()
            _write_owned_json(
                run_dir / "force-cmdlist.json",
                {"before": before, "requested": value, "readback": readback},
                owner_uid,
            )
            return readback
        except (OSError, RuntimeError):
            return ""

    execution = execute_capture_transaction(
        request,
        owner_uid,
        runner,
        lambda entry: captures.get(
            entry.run_id,
            RawCaptureIndex(
                entry.run_id,
                "",
                0,
                0,
                0,
                0,
                False,
                False,
                False,
                "raw capture missing",
                None,
            ),
        ),
        set_force_cmdlist,
        trace_action,
    )
    terminal = {
        "state": "complete"
        if execution.campaign.outcome == Outcome.SUCCESS
        else "failed",
        "campaign_id": request.campaign_id,
        "outcome": execution.campaign.outcome.value,
        "completed_run_ids": list(execution.campaign.completed_run_ids),
        "failed_run_id": execution.campaign.failed_run_id,
        "reasons": list(execution.campaign.reasons),
        "cleanup_ok": execution.cleanup_ok,
        "pretraffic_rollback_attempted": execution.rollback_attempted,
        "loaded_srcversion_after": _loaded_srcversion(),
        "candidate_left_loaded": _loaded_srcversion() == candidate_srcversion,
        "runs": [_classification_json(result) for result in execution.runs],
    }
    _write_owned_json(request.status_path, terminal, owner_uid)
    return 0 if execution.campaign.outcome == Outcome.SUCCESS else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("vertical-schedule", "batch-schedule"):
        schedule_parser = subparsers.add_parser(command)
        schedule_parser.add_argument("--seed", type=int, required=True)
    for command in ("vertical", "batch"):
        capture_parser = subparsers.add_parser(command)
        capture_parser.add_argument("--campaign-id", required=True)
        capture_parser.add_argument("--seed", type=int, required=True)
        capture_parser.add_argument("--location-plan", type=Path, required=True)
        capture_parser.add_argument("--module-manifest", type=Path, required=True)
    privileged = subparsers.add_parser("_privileged")
    privileged.add_argument("request", type=Path)
    privileged.add_argument("request_sha256")
    args = parser.parse_args(argv)
    if args.command in ("vertical", "batch"):
        return run_coordinator(
            args.command,
            args.campaign_id,
            args.seed,
            args.location_plan,
            args.module_manifest,
        )
    if args.command == "_privileged":
        return run_privileged_capture(args.request, args.request_sha256)
    schedule = (
        vertical_schedule(args.seed)
        if args.command == "vertical-schedule"
        else repetition_schedule(50, 50, args.seed)
    )
    json.dump(
        _schedule_json(schedule),
        sys.stdout,
        indent=2,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
