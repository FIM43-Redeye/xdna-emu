#!/usr/bin/env python3
"""Frozen NPU1 firmware-evidence campaign and safe transaction model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import tempfile
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


def vertical_schedule(seed: int) -> tuple[ScheduleEntry, ...]:
    return repetition_schedule(1, 1, seed)


def repetition_schedule(treatment_count: int, control_count: int, seed: int) -> tuple[ScheduleEntry, ...]:
    if treatment_count < 0 or control_count < 0:
        raise ValueError("schedule counts must be nonnegative")
    entries = [
        *(ScheduleEntry(0, repetition, TREATMENT) for repetition in range(treatment_count)),
        *(ScheduleEntry(0, repetition, CONTROL) for repetition in range(control_count)),
    ]
    random.Random(seed).shuffle(entries)
    return tuple(ScheduleEntry(ordinal, entry.repetition, entry.arm) for ordinal, entry in enumerate(entries))


_OUTPUT = re.compile(r"^Correct output ([0-9]+) == ([0-9]+)$")


def parse_output(stdout: str) -> OutputOracleResult:
    lines = [line.strip() for line in stdout.splitlines()]
    pairs = [match.groups() for line in lines if (match := _OUTPUT.fullmatch(line))]
    values = tuple(int(actual) for actual, _ in pairs)
    references = tuple(int(reference) for _, reference in pairs)
    expected = tuple(range(2, 66))
    pass_count = sum(line == "PASS!" for line in lines)
    if values != expected or references != expected:
        return OutputOracleResult(False, values, pass_count == 1, "ordered output is not exactly 2 through 65")
    if pass_count != 1:
        return OutputOracleResult(False, values, False, "expected exactly one PASS! marker")
    if any(line.startswith("Error in output ") for line in lines):
        return OutputOracleResult(False, values, True, "host reported an output error")
    return OutputOracleResult(True, values, True, "")


_REQUEST = re.compile(r"^FW_REQUEST opcode=(0x[0-9a-fA-F]+)$")
_EVENT = re.compile(r"^LIFECYCLE event=([a-z_]+)$")
_EXPECTED_EVENTS = ("interrupt", "mailbox_response", "queue_head", "scheduler", "fence")


def _scoped_lines(text: str, run_id: str) -> tuple[list[str], str | None]:
    lines = [line.strip() for line in text.splitlines()]
    begin = [index for index, line in enumerate(lines) if line == f"NPU1_FW_BEGIN {run_id}"]
    end = [index for index, line in enumerate(lines) if line == f"NPU1_FW_END {run_id}"]
    if len(begin) != 1 or len(end) != 1 or begin[0] >= end[0]:
        return [], "run boundary markers are missing, duplicated, or out of order"
    return lines[begin[0] + 1 : end[0]], None


def parse_lifecycle(text: str, entry: ScheduleEntry) -> LifecycleResult:
    lines, boundary_error = _scoped_lines(text, entry.run_id)
    if boundary_error:
        return LifecycleResult(False, (), None, boundary_error)
    records: list[LifecycleRecord] = []
    tokens: list[tuple[str, str | int]] = []
    for line in lines:
        if match := _REQUEST.fullmatch(line):
            opcode = int(match.group(1), 16)
            tokens.append(("request", opcode))
            records.append(LifecycleRecord("request", f"0x{opcode:x}"))
        elif match := _EVENT.fullmatch(line):
            event = match.group(1)
            tokens.append(("event", event))
            records.append(LifecycleRecord("event", event))
    expected: list[tuple[str, str | int]] = [
        ("request", 0x02),
        ("request", 0x106),
        ("request", 0x11),
        ("request", entry.arm.execute_opcode),
        *(("event", event) for event in _EXPECTED_EVENTS),
        ("request", 0x03),
    ]
    execute = [
        value
        for kind, value in tokens
        if kind == "request" and value in (TREATMENT.execute_opcode, CONTROL.execute_opcode)
    ]
    if tokens != expected:
        return LifecycleResult(
            False,
            tuple(records),
            execute[0] if len(execute) == 1 else None,
            "firmware lifecycle is missing, duplicated, unexpected, or out of order",
        )
    return LifecycleResult(True, tuple(records), entry.arm.execute_opcode, "")


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
    if capture.tdr_after > capture.tdr_before:
        reasons.append("new TDR observed")
    if capture.iommu_faults_after > capture.iommu_faults_before:
        reasons.append("new IOMMU fault observed")
    if reasons:
        return RunClassification(entry.run_id, Outcome.DEVICE_FAULT_OR_WEDGE, tuple(reasons), None, (), ())
    if command.timed_out:
        return RunClassification(entry.run_id, Outcome.INFRASTRUCTURE_FAILURE, ("host process timed out",), None, (), ())
    if command.returncode != 0:
        return RunClassification(
            entry.run_id,
            Outcome.INFRASTRUCTURE_FAILURE,
            (f"host process exited {command.returncode}",),
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
        outcome = Outcome.PROVENANCE_FAILURE if not lifecycle.records else Outcome.SEMANTIC_MISMATCH
        return RunClassification(entry.run_id, outcome, (lifecycle.reason,), lifecycle.execute_opcode, (), ())
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
        ("fail_cmd_idx", "fail_cmd_status"),
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
            return CampaignClassification(result.outcome, tuple(completed), result.run_id, result.reasons)
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
    component = lambda name: {
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


def preflight_errors(spec: VerticalSpec, snapshot: PreflightSnapshot) -> tuple[str, ...]:
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
    trace_events: tuple[str, ...] = (
        "xdna_job",
        "mbox_set_tail",
        "mbox_set_head",
        "mbox_irq_handle",
        "mbox_rx_worker",
        "mbox_poll_handle",
        "uc_irq_handle",
        "uc_wakeup",
    )
    dynamic_debug_selectors: tuple[str, ...] = ("amdxdna firmware request",)


@dataclass(frozen=True)
class CaptureRequest:
    campaign_id: str
    campaign_dir: Path
    status_path: Path
    module: QualifiedModuleManifest
    schedule: tuple[ScheduleEntry, ...]
    executable: Path
    xclbin: Path
    instructions: Path
    location_plan: Path | None = None

    @classmethod
    def synthetic(cls, root: Path, schedule: Sequence[ScheduleEntry]) -> "CaptureRequest":
        return cls(
            campaign_id="campaign.synthetic",
            campaign_dir=root,
            status_path=root / "status.json",
            module=QualifiedModuleManifest(
                root / "candidate.ko",
                "a" * 64,
                root / "original.ko",
                "b" * 64,
            ),
            schedule=tuple(schedule),
            executable=root / "test.exe",
            xclbin=root / "aie.xclbin",
            instructions=root / "insts.bin",
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


def build_transaction_plan(request: CaptureRequest, uid: int, *, submitted: bool) -> TransactionPlan:
    trace_instance = f"npu1-fw-{request.campaign_id}"
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
                str(uid),
                "--",
                "env",
                "-u",
                "XDNA_EMU",
                "-u",
                "XDNA_EMU_RUNTIME",
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
    cleanup = (CommandSpec(("rmdir", f"/sys/kernel/tracing/instances/{trace_instance}")),)
    rollback = (
        CommandSpec(("modprobe", "-r", "amdxdna")),
        CommandSpec(("insmod", str(request.module.original_path))),
    ) if not submitted else ()
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
    return TransactionExecution(campaign, tuple(results), cleanup_ok, rollback_attempted)


_CAMPAIGN_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


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
    if snapshot.kernel_release and module.kernel_release != snapshot.kernel_release:
        errors.append("candidate module kernel release does not match the running kernel")
    if not module.tdr_parameter_present or not module.trace_events or not module.dynamic_debug_selectors:
        errors.append("candidate module debug or normal-TDR capabilities are incomplete")
    if not location_plan.is_file():
        errors.append("location plan is missing")
    if errors:
        raise ValueError("; ".join(sorted(set(errors))))

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
        campaign_dir=campaign_dir,
        status_path=campaign_dir / "status.json",
        module=module,
        schedule=schedule,
        executable=files["executable"].resolve(),
        xclbin=files["xclbin"].resolve(),
        instructions=files["instructions"].resolve(),
        location_plan=location_plan.resolve(),
    )
    request_path = campaign_dir / "capture-request.json"
    _write_capture_request(request_path, request)
    return PreparedCapture(
        request,
        request_path,
        (
            "pkexec",
            sys.executable,
            str(Path(__file__).resolve()),
            "_privileged",
            str(request_path),
        ),
    )


def _write_capture_request(path: Path, request: CaptureRequest) -> None:
    data = {
        "campaign_id": request.campaign_id,
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
        "executable": str(request.executable),
        "xclbin": str(request.xclbin),
        "instructions": str(request.instructions),
        "location_plan": str(request.location_plan) if request.location_plan else None,
    }
    write_terminal_status(path, data)


def write_terminal_status(path: Path, status: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(status, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("vertical-schedule", "batch-schedule"))
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    schedule = (
        vertical_schedule(args.seed)
        if args.command == "vertical-schedule"
        else repetition_schedule(50, 50, args.seed)
    )
    json.dump(
        [
            {
                "run_id": entry.run_id,
                "ordinal": entry.ordinal,
                "repetition": entry.repetition,
                "arm": entry.arm.name,
                "force_cmdlist": entry.arm.force_cmdlist,
                "execute_opcode": f"0x{entry.arm.execute_opcode:x}",
            }
            for entry in schedule
        ],
        sys.stdout,
        indent=2,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
