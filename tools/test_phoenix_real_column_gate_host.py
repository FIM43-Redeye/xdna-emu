import hashlib
import importlib.util
import json
import struct
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts/phoenix-real-column-gate-host.py"


def load_host():
    assert _SCRIPT.is_file(), "physical host wrapper is missing"
    spec = importlib.util.spec_from_file_location("phoenix_gate_host", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_pair(tmp_path):
    pair = tmp_path / "pair"
    control = struct.pack("<III", 9, 1, 7)
    treatment = struct.pack("<III", 9, 0, 7)
    pair.mkdir()
    (pair / "control.insts.bin").write_bytes(control)
    (pair / "treatment.insts.bin").write_bytes(treatment)
    (pair / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "target": "phoenix_npu1",
        "firmware": {"version": "1.5.5.391", "sha256": "a" * 64},
        "placement": {"start_col": 1, "num_col": 1},
        "arms": {
            "control": {"sha256": hashlib.sha256(control).hexdigest()},
            "treatment": {"sha256": hashlib.sha256(treatment).hexdigest()},
        },
        "one_word_diff": {
            "byte_offset": 4,
            "control": "0x00000001",
            "treatment": "0x00000000",
        },
    }))
    run = pair / "kvm" / "control-run"
    module = run / "driver-source/drivers/accel/amdxdna/amdxdna.ko"
    module.parent.mkdir(parents=True)
    module.write_bytes(b"module")
    (run / "tuple.txt").write_text(
        "driver_commit=abc\n"
        "guest_kernel_version=test-kernel\n"
        f"{hashlib.sha256(module.read_bytes()).hexdigest()}  {module}\n"
    )
    (run / "result.json").write_text(json.dumps({
        "kvm_disposition": {
            "admitted": True,
            "reason": "known_scheduler_red",
        },
    }))
    (pair / "kvm/control-safety-qualified").write_text(
        f"run={run}\ndisposition=known_scheduler_red\n"
    )
    return pair, run, module


def add_host_artifacts(tmp_path, pair, run):
    paths = {
        "runner": tmp_path / "worktree/bridge-runner/build/bridge-trace-runner",
        "classifier": tmp_path / "worktree/target/debug/libxdna_emu.so",
        "clock_query": run / "xdna-clock-query",
        "xclbin": tmp_path / "fixture/fault-package/aie.xclbin",
        "mlir": tmp_path / "fixture/fault-package/work/input_with_addresses.mlir",
        "expected_output": tmp_path / "fixture/hw.out.bin",
        "canary_instructions": tmp_path / "full-witness-fault.insts.bin",
        "firmware": tmp_path / "usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin",
        "register_db": (
            tmp_path
            / "work/mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json"
        ),
        "xrt_coreutil": tmp_path / "opt/xilinx/xrt/lib/libxrt_coreutil.so.2.26.0",
        "xrt_core": tmp_path / "opt/xilinx/xrt/lib/libxrt_core.so.2.26.0",
        "xrt_driver": tmp_path / "opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2.26.0",
    }
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"firmware" if name == "firmware" else name.encode())
    manifest = json.loads((pair / "manifest.json").read_text())
    manifest["firmware"]["sha256"] = hashlib.sha256(
        paths["firmware"].read_bytes()
    ).hexdigest()
    (pair / "manifest.json").write_text(json.dumps(manifest))
    with (run / "tuple.txt").open("a") as output:
        for path in paths.values():
            output.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n")
    return paths


@pytest.mark.parametrize(("arm", "classification", "expected"), [
    ("control", {"qualified": True, "reason": "control"}, True),
    ("treatment", {"qualified": True, "reason": "freeze_resume"}, True),
    ("control", {"qualified": False, "reason": "missing_pm_fault"}, False),
    ("treatment", {"qualified": True, "reason": "control"}, False),
])
def test_host_disposition_requires_exact_physical_behavior(
    arm, classification, expected,
):
    assert load_host().host_behavioral_pass(arm, classification) is expected


def test_pair_validation_requires_manifest_hashes_and_one_word_diff(tmp_path):
    pair, _, _ = write_pair(tmp_path)

    validated = load_host().validate_pair(pair)

    assert validated["one_word_diff"]["byte_offset"] == 4


def test_pair_validation_rejects_changed_arm(tmp_path):
    pair, _, _ = write_pair(tmp_path)
    (pair / "treatment.insts.bin").write_bytes(b"changed")

    with pytest.raises(ValueError, match="treatment hash"):
        load_host().validate_pair(pair)


def test_kvm_control_resolution_is_confined_and_safety_qualified(tmp_path):
    pair, run, module = write_pair(tmp_path)

    resolved = load_host().resolve_kvm_control(pair)

    assert resolved["run"] == run.resolve()
    assert resolved["module"] == module.resolve()
    assert resolved["disposition"] == "known_scheduler_red"


def test_kvm_control_resolution_rejects_marker_escape(tmp_path):
    pair, _, _ = write_pair(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (pair / "kvm/control-safety-qualified").write_text(
        f"run={outside}\ndisposition=known_scheduler_red\n"
    )

    with pytest.raises(ValueError, match="outside pair KVM evidence"):
        load_host().resolve_kvm_control(pair)


def test_tuple_artifact_resolution_requires_unique_matching_bytes(tmp_path):
    artifact = tmp_path / "fault-package/aie.xclbin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"xclbin")
    tuple_path = tmp_path / "tuple.txt"
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    tuple_path.write_text(
        "driver_commit=abc\n"
        f"{digest}  {artifact}\n"
        f"{digest}  {artifact}\n"
    )

    parsed = load_host().parse_tuple(tuple_path)

    assert parsed["values"]["driver_commit"] == "abc"
    assert load_host().tuple_artifact(parsed, "fault-package/aie.xclbin") == (
        artifact.resolve()
    )


def test_tuple_artifact_resolution_rejects_changed_bytes(tmp_path):
    artifact = tmp_path / "hw.out.bin"
    artifact.write_bytes(b"changed")
    parsed = {
        "values": {},
        "files": {artifact.resolve(): "0" * 64},
    }

    with pytest.raises(ValueError, match="tuple hash"):
        load_host().tuple_artifact(parsed, "hw.out.bin")


def test_tuple_parser_rejects_conflicting_duplicate_attestation(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"artifact")
    tuple_path = tmp_path / "tuple.txt"
    tuple_path.write_text(
        f"{'0' * 64}  {artifact}\n"
        f"{'1' * 64}  {artifact}\n"
    )

    with pytest.raises(ValueError, match="conflicting tuple file"):
        load_host().parse_tuple(tuple_path)


def test_module_transaction_uses_physical_tdr_and_restores_installed_module(
    tmp_path,
):
    module = tmp_path / "amdxdna.ko"

    commands = load_host().module_transaction_commands(module)

    assert commands == {
        "setup": (
            ("rmmod", "amdxdna"),
            (
                "insmod", str(module), "tdr_timeout_ms=2000",
                "force_cmdlist=Y", "dyndbg=+p",
            ),
        ),
        "restore": (
            ("rmmod", "amdxdna"),
            ("modprobe", "amdxdna"),
        ),
    }


def test_runner_command_is_bounded_and_requires_exact_placement(tmp_path):
    paths = {
        "runner": tmp_path / "bridge-trace-runner",
        "xclbin": tmp_path / "aie.xclbin",
        "instructions": tmp_path / "arm.insts.bin",
        "trace": tmp_path / "arm.trace.bin",
        "output": tmp_path / "arm.out.bin",
    }

    assert load_host().runner_argv(paths, (1, 1)) == (
        "timeout", "-k", "5", "650",
        str(paths["runner"]),
        "--xclbin", str(paths["xclbin"]),
        "--instr", str(paths["instructions"]),
        "--trace-out", str(paths["trace"]),
        "--output", str(paths["output"]),
        "--qos-gops", "1", "--qos-fps", "1000",
        "--expect-placement", "1:1", "-v",
    )


def test_lifecycle_requires_destroy_between_two_contexts():
    log = (
        "xdna_mailbox.0: opcode 0x18 size 24 id 1\n"
        "xdna_mailbox.0: opcode 0x18 size 12 id 1\n"
        "xdna_mailbox.0: opcode 0x3 size 4 id 1\n"
        "xdna_mailbox.0: opcode 0x18 size 24 id 2\n"
        "xdna_mailbox.0: opcode 0x18 size 12 id 2"
    )

    assert load_host().lifecycle_ok(log) is True
    assert load_host().lifecycle_ok(log.replace("opcode 0x3", "opcode 0x4")) is False


def test_kernel_log_is_confined_by_exact_run_markers():
    log = (
        "old message\n"
        "PHOENIX_REAL_COLUMN_GATE_HOST_BEGIN abc\n"
        "inside one\n"
        "inside two\n"
        "PHOENIX_REAL_COLUMN_GATE_HOST_END abc\n"
        "new message"
    )

    assert load_host().kernel_log_between(log, "abc") == "inside one\ninside two\n"


def test_host_run_pass_requires_behavior_lifecycle_and_restoration():
    status = {"state": "complete", "restored": True, "lifecycle_ok": True}
    result = {"classification": {"qualified": True, "reason": "control"}}

    assert load_host().host_run_pass("control", status, result) is True
    for field in ("restored", "lifecycle_ok"):
        changed = dict(status, **{field: False})
        assert load_host().host_run_pass("control", changed, result) is False
    assert load_host().host_run_pass(
        "control", dict(status, state="failed"), result,
    ) is False


def test_host_artifacts_are_derived_from_qualified_kvm_tuple(tmp_path):
    pair, run, module = write_pair(tmp_path)
    expected = add_host_artifacts(tmp_path, pair, run)

    artifacts = load_host().resolve_host_artifacts(
        pair, load_host().resolve_kvm_control(pair),
    )

    assert artifacts["module"] == module.resolve()
    for name, path in expected.items():
        assert artifacts[name] == path.resolve()
    assert artifacts["control_instructions"] == (
        pair / "control.insts.bin"
    ).resolve()
    assert artifacts["treatment_instructions"] == (
        pair / "treatment.insts.bin"
    ).resolve()


def test_treatment_requires_restored_behavioral_host_control(tmp_path):
    pair, _, _ = write_pair(tmp_path)
    run = pair / "host/control-run"
    run.mkdir(parents=True)
    (run / "result.json").write_text(json.dumps({
        "classification": {"qualified": True, "reason": "control"},
    }))
    (run / "status.json").write_text(json.dumps({
        "state": "complete",
        "restored": True,
        "lifecycle_ok": True,
    }))
    (pair / "host/control-behavior-qualified").write_text(f"run={run}\n")

    assert load_host().resolve_host_control(pair) == run.resolve()

    (run / "status.json").write_text(json.dumps({
        "state": "failed",
        "restored": False,
    }))
    with pytest.raises(ValueError, match="not restored"):
        load_host().resolve_host_control(pair)


def test_host_request_is_confined_and_pins_the_kvm_artifacts(tmp_path):
    pair, run, module = write_pair(tmp_path)
    paths = add_host_artifacts(tmp_path, pair, run)
    repository = tmp_path / "worktree"
    run_dir = pair / "host/control-request"

    request = load_host().build_host_request(
        "control", pair, repository, run_dir, {}, "test-kernel",
    )

    assert request["arm"] == "control"
    assert Path(request["run_dir"]) == run_dir.resolve()
    assert Path(request["artifacts"]["module"]["path"]) == module.resolve()
    assert request["artifacts"]["module"]["sha256"] == hashlib.sha256(
        module.read_bytes()
    ).hexdigest()
    assert Path(request["artifacts"]["runner"]["path"]) == paths[
        "runner"
    ].resolve()


def test_host_request_rejects_emulator_environment(tmp_path):
    pair, run, _ = write_pair(tmp_path)
    add_host_artifacts(tmp_path, pair, run)

    with pytest.raises(ValueError, match="XDNA_EMU"):
        load_host().build_host_request(
            "control", pair, tmp_path / "worktree",
            pair / "host/control-request", {"XDNA_EMU": "1"},
            "test-kernel",
        )


def test_privileged_request_revalidation_rejects_tampering(tmp_path):
    pair, run, _ = write_pair(tmp_path)
    add_host_artifacts(tmp_path, pair, run)
    request = load_host().build_host_request(
        "control", pair, tmp_path / "worktree",
        pair / "host/control-request", {}, "test-kernel",
    )
    request["artifacts"]["module"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="request does not match"):
        load_host().validate_host_request(request)


def test_worker_environment_is_physical_and_points_at_pinned_classifier(tmp_path):
    request = {
        "repository": str(tmp_path / "repository"),
        "artifacts": {
            "classifier": {"path": str(tmp_path / "target/debug/libxdna_emu.so")},
        },
    }

    environment = load_host().worker_environment(
        request, "maya", "/home/maya",
    )

    assert "XDNA_EMU" not in environment
    assert environment["XDNA_EMU_DIR"] == str(tmp_path / "target/debug")
    assert environment["XILINX_XRT"] == "/opt/xilinx/xrt"
    assert environment["BRIDGE_RUNNER_ASYNC_CTX"] == "0"
    assert environment["BRIDGE_RUNNER_REUSE_CONTEXT"] == "0"
