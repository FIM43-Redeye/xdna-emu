import hashlib
import importlib.util
import json
import struct
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts/phoenix-real-column-gate-host.py"
_PROTECTED_GATE_PATCH = (
    Path(__file__).parents[1]
    / "docs/patches/0005-LOCAL-phoenix-protected-column-gate.patch"
)
_KVM_SCRIPT = Path(__file__).parents[1] / "scripts/phoenix-vfio-user-qemu.sh"
_GUEST_INIT = (
    Path(__file__).parents[1]
    / "tools/phoenix-vfio-user/guest-driver-probe-init.sh"
)


def load_host():
    assert _SCRIPT.is_file(), "physical host wrapper is missing"
    spec = importlib.util.spec_from_file_location("phoenix_gate_host", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protected_gate_patch_has_one_fixed_restoring_operation():
    text = _PROTECTED_GATE_PATCH.read_text()

    assert "phoenix_column_gate" in text
    assert 'sysfs_streq(kbuf, "control")' in text
    assert 'sysfs_streq(kbuf, "treatment")' in text
    assert "aie2_set_runtime_cfg(ndev, 1, 0)" in text
    assert "aie2_set_runtime_cfg(ndev, 1, 1)" in text
    assert "0x1000000c" in text
    assert "0x10000200" in text
    assert "0x000fff20" in text
    assert "usleep_range(1000, 1100)" in text


def test_protected_gate_finalizes_trace_after_restore_from_aiert_definitions():
    text = _PROTECTED_GATE_PATCH.read_text()

    assert '#include "xaie_events_aieml.h"' in text
    assert '#include "xaiemlgbl_params.h"' in text
    assert "phoenix_finalize_column_gate_trace" in text
    assert "XAIEML_EVENTS_PL_USER_EVENT_0" in text
    assert "XAIEMLGBL_PL_MODULE_EVENT_GENERATE" in text
    assert "XAIEMLGBL_CORE_MODULE_TRACE_STATUS" in text
    assert "XAIEMLGBL_PL_MODULE_TRACE_STATUS" in text
    finalize_body = text[
        text.index("static int phoenix_finalize_column_gate_trace"):
        text.index("static int phoenix_validate_column_gate_context")
    ]
    assert "TRACE_CONTROL0" not in finalize_body

    gate = text[text.index("int aie2_phoenix_column_gate"):]
    restore = gate.index("restore_ret = phoenix_set_column_clock")
    finalize = gate.index("trace_ret = phoenix_finalize_column_gate_trace")
    handback = gate.index("policy_ret = aie2_set_runtime_cfg", finalize)
    assert restore < finalize < handback


def test_protected_gate_brackets_transition_with_witness_dwells():
    gate = _PROTECTED_GATE_PATCH.read_text()
    gate = gate[gate.index("int aie2_phoenix_column_gate"):]

    transition = gate.index("ret = phoenix_set_column_clock")
    restore = gate.index("restore_ret = phoenix_set_column_clock")
    finalize = gate.index("trace_ret = phoenix_finalize_column_gate_trace")
    dwell = "usleep_range(1000, 1100)"
    pre = gate.index(dwell)
    gated = gate.index(dwell, pre + 1)
    post = gate.index(dwell, gated + 1)

    assert gate.count(dwell) == 3
    assert pre < transition < gated < restore < post < finalize


def test_kvm_gate_uses_fixed_hook_and_prepared_trace_witness():
    kvm = _KVM_SCRIPT.read_text()
    guest = _GUEST_INIT.read_text()

    assert (
        'readonly PROTECTED_GATE_PATCH="$ROOT/docs/patches/'
        '0005-LOCAL-phoenix-protected-column-gate.patch"'
    ) in kvm
    assert 'readonly GATE_INSTS="$RUN_DIR/active-gate.insts.bin"' in kvm
    assert "prepare-real-column-gate-trace" in kvm
    apply = 'git -C / apply --directory="${DRIVER_SOURCE#/}"'
    assert kvm.count(apply) == 3
    assert f'{apply} "$NPI_READ_PATCH"' in kvm
    assert f'{apply} "$PROTECTED_GATE_PATCH"' in kvm
    assert '--phoenix-column-gate "$real_column_gate_arm"' in guest


def test_kvm_treatment_requires_control_from_exact_source_commit():
    kvm = _KVM_SCRIPT.read_text()

    assert 'readonly XDNA_EMU_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"' in kvm
    assert 'marker_commit="$(sed -n \'s/^xdna_emu_commit=//p\' "$GATE_CONTROL_MARKER")"' in kvm
    assert '[[ "$marker_commit" == "$XDNA_EMU_COMMIT" ]]' in kvm
    assert "xdna_emu_commit=%s\\n'" in kvm


def test_kvm_gate_stages_and_attests_official_aiert_headers():
    kvm = _KVM_SCRIPT.read_text()

    assert 'AIE_RT_SOURCE="${AIE_RT_PATH-$NPU_WORK/aie-rt/driver/src}"' in kvm
    assert "readonly AIE_RT_SOURCE" in kvm
    assert 'readonly AIEML_EVENTS="$AIE_RT_SOURCE/events/xaie_events_aieml.h"' in kvm
    assert 'readonly AIEML_PARAMS="$AIE_RT_SOURCE/global/xaiemlgbl_params.h"' in kvm
    assert 'install -m 0644 "$AIEML_EVENTS" "$AIEML_PARAMS"' in kvm
    assert 'sha256sum "$AIEML_EVENTS" "$AIEML_PARAMS"' in kvm


def test_kvm_guest_modules_resolve_for_the_pinned_kernel():
    kvm = _KVM_SCRIPT.read_text()

    assert 'modprobe --show-depends' not in kvm
    assert kvm.count(
        'modprobe --set-version "$GUEST_KERNEL_VERSION" --show-depends'
    ) == 2
    assert (
        'modinfo -k "$GUEST_KERNEL_VERSION" -F "$signature_field" amdxdna'
        in kvm
    )


def test_kvm_guest_boots_the_raw_newc_archive():
    kvm = _KVM_SCRIPT.read_text()
    start = kvm.index("    guest_qemu=(")
    end = kvm.index('    if [[ "$guest_result" -ne 0 ]]', start)
    launch = kvm[start:end]

    assert 'readonly INITRAMFS="$RUN_DIR/initramfs.cpio"' in kvm
    assert 'gzip -n -9 -c "$initramfs_cpio" >"$INITRAMFS"' not in kvm
    assert '-initrd "$INITRAMFS"' in launch
    assert '"execute":"qmp_capabilities"' not in launch


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
        f"xdna_emu_commit={'a' * 40}\n"
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
        f"xdna_emu_commit={'a' * 40}\n"
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


def write_npi_kvm_run(tmp_path):
    repository = tmp_path / "worktree"
    run = repository / "build/experiments/phoenix-vfio-user/npi-run"
    module = run / "driver-source/drivers/accel/amdxdna/amdxdna.ko"
    patch = repository / "docs/patches/0004-LOCAL-phoenix-read-only-npi-lock-probe.patch"
    firmware = tmp_path / "usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin"
    for path, data in (
        (module, b"module"),
        (patch, b"patch"),
        (firmware, b"firmware"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (run / "tuple.txt").write_text(
        "driver_commit=216cefececd74effcd7a88350c71b99f5ef9a215\n"
        f"xdna_emu_commit={'a' * 40}\n"
        "guest_kernel_version=test-kernel\n"
        "research_probe=phoenix-read-only-npi-lock\n"
        "expected_management_request=opcode:0x203,size:24,type:2,row:0,col:0,offset:0x1000000c\n"
        "expected_system_read=0xac00000c\n"
        + "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n"
            for path in (module, patch, firmware)
        )
    )
    (run / "guest.log").write_text(
        "PHOENIX_NPI_READ_BEGIN\n"
        "PHOENIX_NPI_READ value=0x00000000\n"
        "PHOENIX_NPI_READ_PASS\n"
        "PHOENIX_DRIVER_PROBE_PASS\n"
    )
    (run / "dmesg.log").write_text(
        "xdna_mailbox.1: opcode 0x203 size 24 id 0x1d00000e\n"
        "xdna_mailbox.1: opcode 0x203 size 8 id 0x1d00000e\n"
    )
    return repository, run, module, firmware


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
        f"xdna_emu_commit={'a' * 40}\n"
    )

    with pytest.raises(ValueError, match="outside pair KVM evidence"):
        load_host().resolve_kvm_control(pair)


def test_kvm_control_resolution_rejects_marker_tuple_commit_mismatch(tmp_path):
    pair, run, _ = write_pair(tmp_path)
    (pair / "kvm/control-safety-qualified").write_text(
        f"run={run}\ndisposition=known_scheduler_red\n"
        f"xdna_emu_commit={'b' * 40}\n"
    )

    with pytest.raises(ValueError, match="source commit"):
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


def test_original_module_parameters_preserve_absence_and_values(tmp_path):
    parameters = tmp_path / "parameters"
    parameters.mkdir()
    force = parameters / "force_cmdlist"
    force.write_text("Y\n")
    host = load_host()

    snapshot = host.module_parameters(parameters)

    assert snapshot == {"tdr_timeout_ms": None, "force_cmdlist": "Y"}
    force.write_text("N\n")
    host.restore_module_parameters(parameters, snapshot)
    assert force.read_text() == "Y\n"
    assert not (parameters / "tdr_timeout_ms").exists()

    (parameters / "tdr_timeout_ms").write_text("2000\n")
    with pytest.raises(RuntimeError, match="unexpectedly exists"):
        host.restore_module_parameters(parameters, snapshot)


def test_candidate_signature_must_match_installed_module_trust():
    host = load_host()
    original = {
        "signer": "enrolled key",
        "sig_id": "PKCS#7",
        "sig_key": "AA:BB",
        "sig_hashalgo": "sha512",
    }

    assert host.module_signature_matches(original, dict(original)) is True
    assert host.module_signature_matches(
        {**original, "sig_id": ""}, original,
    ) is False
    assert host.module_signature_matches(
        {**original, "sig_key": "CC:DD"}, original,
    ) is False


def test_absent_module_skips_device_lookup_during_restore():
    class NoModule:
        def _loaded_srcversion(self):
            return None

        def _device_node_for_bdf(self, _bdf):
            raise AssertionError("device lookup must be skipped")

    assert load_host().active_clients_before_restore(NoModule(), "bdf") == 0


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


def test_npi_lifecycle_requires_one_ordered_matching_request_and_response():
    log = (
        "xdna_mailbox.1: opcode 0x203 size 24 id 0x1d00000e\n"
        "xdna_mailbox.1: opcode 0x203 size 8 id 0x1d00000e\n"
    )
    host = load_host()

    assert host.npi_lifecycle_ok(log) is True
    assert host.npi_lifecycle_ok(log.replace("0x1d00000e\n", "0x1d00000f\n", 1)) is False
    assert host.npi_lifecycle_ok("\n".join(reversed(log.splitlines()))) is False
    assert host.npi_lifecycle_ok(log.replace("xdna_mailbox.1", "xdna_mailbox.2", 1)) is False
    assert host.npi_lifecycle_ok(log + log) is False


def test_npi_lifecycle_cli_reuses_the_ordered_validator(tmp_path):
    log = tmp_path / "dmesg.log"
    log.write_text(
        "xdna_mailbox.1: opcode 0x203 size 24 id 0x1d00000e\n"
        "xdna_mailbox.1: opcode 0x203 size 8 id 0x1d00000e\n"
    )
    host = load_host()

    assert host.main(["_validate_npi_lifecycle", str(log)]) == 0
    log.write_text("\n".join(reversed(log.read_text().splitlines())))
    assert host.main(["_validate_npi_lifecycle", str(log)]) == 1


def test_cli_rejects_every_superseded_raw_gate_entry_point(tmp_path, monkeypatch):
    host = load_host()
    calls = []

    def record_call(*args):
        calls.append(args)
        return 0

    monkeypatch.setattr(host, "_coordinator", record_call)
    monkeypatch.setattr(host, "_run_privileged", record_call)
    monkeypatch.setattr(host, "_run_worker", record_call)
    request = tmp_path / "request.json"
    digest = "0" * 64

    for argv in (
        ["control", str(tmp_path / "pair")],
        ["treatment", str(tmp_path / "pair")],
        ["_privileged", str(request), digest],
        ["_worker", str(request), digest],
    ):
        assert host.main(argv) == 1

    assert calls == []


def test_npi_kvm_run_resolution_rederives_exact_qualification(tmp_path):
    repository, run, module, firmware = write_npi_kvm_run(tmp_path)
    host = load_host()
    host.NPI_FIRMWARE_SHA256 = hashlib.sha256(firmware.read_bytes()).hexdigest()

    resolved = host.resolve_npi_kvm_run(run, repository)

    assert resolved["module"] == module.resolve()
    assert resolved["tuple_values"]["expected_system_read"] == "0xac00000c"

    (run / "dmesg.log").write_text(
        "xdna_mailbox.1: opcode 0x203 size 24 id 1\n"
        "xdna_mailbox.1: opcode 0x203 size 8 id 2\n"
    )
    with pytest.raises(ValueError, match="mailbox lifecycle"):
        host.resolve_npi_kvm_run(run, repository)


def test_npi_kvm_run_resolution_rejects_incomplete_guest(tmp_path):
    repository, run, _, firmware = write_npi_kvm_run(tmp_path)
    host = load_host()
    host.NPI_FIRMWARE_SHA256 = hashlib.sha256(firmware.read_bytes()).hexdigest()
    guest = run / "guest.log"
    guest.write_text(guest.read_text().replace("PHOENIX_DRIVER_PROBE_PASS\n", ""))

    with pytest.raises(ValueError, match="guest qualification"):
        host.resolve_npi_kvm_run(run, repository)


def test_npi_kvm_run_resolution_rejects_out_of_order_guest_success(tmp_path):
    repository, run, _, firmware = write_npi_kvm_run(tmp_path)
    host = load_host()
    host.NPI_FIRMWARE_SHA256 = hashlib.sha256(firmware.read_bytes()).hexdigest()
    guest = run / "guest.log"
    lines = guest.read_text().splitlines()
    guest.write_text("\n".join([lines[-1], *lines[:-1]]) + "\n")

    with pytest.raises(ValueError, match="guest qualification"):
        host.resolve_npi_kvm_run(run, repository)


def test_npi_kvm_run_resolution_rejects_guest_failure_marker(tmp_path):
    repository, run, _, firmware = write_npi_kvm_run(tmp_path)
    host = load_host()
    host.NPI_FIRMWARE_SHA256 = hashlib.sha256(firmware.read_bytes()).hexdigest()
    guest = run / "guest.log"
    guest.write_text(guest.read_text() + "PHOENIX_DRIVER_PROBE_FAIL: late failure\n")

    with pytest.raises(ValueError, match="guest qualification"):
        host.resolve_npi_kvm_run(run, repository)


def test_npi_kvm_run_resolution_rejects_unpinned_firmware(tmp_path):
    repository, run, _, _ = write_npi_kvm_run(tmp_path)

    with pytest.raises(ValueError, match="firmware"):
        load_host().resolve_npi_kvm_run(run, repository)


def test_npi_host_pass_requires_read_canary_lifecycle_and_restoration():
    status = {
        "state": "complete",
        "restored": True,
        "lifecycle_ok": True,
        "canary_ok": True,
    }
    result = {"qualified": True, "value": "0x00000000"}
    host = load_host()

    assert host.npi_host_run_pass(status, result) is True
    for field in ("restored", "lifecycle_ok", "canary_ok"):
        assert host.npi_host_run_pass({**status, field: False}, result) is False


def test_npi_canary_is_bounded_and_uses_the_physical_runtime(tmp_path):
    xrt_smi = tmp_path / "xrt-smi"

    assert load_host().npi_canary_argv(xrt_smi, "maya", "/home/maya") == (
        "timeout", "-k", "5", "120", "runuser", "-u", "maya", "--",
        "env", "-i", "HOME=/home/maya", "USER=maya", "LOGNAME=maya",
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "XILINX_XRT=/opt/xilinx/xrt",
        "LD_LIBRARY_PATH=/opt/xilinx/xrt/lib",
        str(xrt_smi), "validate",
    )

def test_npi_canary_home_stages_the_installed_archive_by_xrt_version(tmp_path):
    run = tmp_path / "read-run"
    coreutil = tmp_path / "libxrt_coreutil.so.2.26.0"
    archive = tmp_path / "xrt_smi_phx.a"
    coreutil.write_bytes(b"coreutil")
    archive.write_bytes(b"archive")

    home = load_host().stage_npi_canary_home(run, coreutil, archive)

    staged = home / ".local/share/xrt/2.26.0/amdxdna/bins/xrt_smi_phx.a"
    assert staged.is_symlink()
    assert staged.resolve() == archive.resolve()


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
