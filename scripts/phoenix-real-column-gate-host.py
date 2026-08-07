#!/usr/bin/env python3
"""Run pinned Phoenix physical gate and read-only NPI probes."""

import argparse
import hashlib
import importlib.util
import json
import os
import pwd
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

NPI_DRIVER_PIN = "216cefececd74effcd7a88350c71b99f5ef9a215"
NPI_FIRMWARE_SHA256 = "d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e"
NPI_REQUEST = "opcode:0x203,size:24,type:2,row:0,col:0,offset:0x1000000c"
NPI_SYSTEM_READ = "0xac00000c"
XRT_VALIDATION_ARCHIVE = Path(
    "/opt/xilinx/xrt/share/amdxdna/bins/xrt_smi_phx.a"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_kvm_control(pair: Path) -> dict:
    pair = pair.resolve(strict=True)
    marker = pair / "kvm/control-safety-qualified"
    values = {}
    for line in marker.read_text().splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise ValueError("KVM control marker is malformed")
        values[key] = value
    if set(values) != {"run", "disposition", "xdna_emu_commit"}:
        raise ValueError("KVM control marker fields are incomplete")
    if not re.fullmatch(r"[0-9a-f]{40}", values["xdna_emu_commit"]):
        raise ValueError("KVM control marker source commit is invalid")
    run = Path(values["run"]).resolve(strict=True)
    if not run.is_relative_to((pair / "kvm").resolve()) or not run.name.startswith(
        "control-"
    ):
        raise ValueError("KVM control marker points outside pair KVM evidence")
    result = json.loads((run / "result.json").read_text())
    disposition = result.get("kvm_disposition", {})
    if (
        disposition.get("admitted") is not True
        or disposition.get("reason") != values["disposition"]
        or disposition.get("reason")
        not in {"behavioral_witness", "known_scheduler_red"}
    ):
        raise ValueError("KVM control is not safety qualified")
    tuple_path = run / "tuple.txt"
    if not tuple_path.is_file():
        raise ValueError("KVM control module or tuple is missing")
    parsed = parse_tuple(tuple_path)
    expected = {
        "driver_commit": NPI_DRIVER_PIN,
        "real_column_gate_arm": "control",
        "xrt_execution": "signed-firmware-chain-exec-npu",
        "research_probe": "phoenix-protected-column-gate",
        "expected_live_placement": "1:1",
        "bridge_runner_async_context": "0",
        "bridge_runner_reuse_context": "0",
    }
    if any(parsed["values"].get(key) != value for key, value in expected.items()):
        raise ValueError("KVM control tuple differs from the protected operation")
    module = tuple_artifact(
        parsed, "driver-source/drivers/accel/amdxdna/amdxdna.ko",
    )
    if module != (
        run / "driver-source/drivers/accel/amdxdna/amdxdna.ko"
    ).resolve():
        raise ValueError("KVM control module is outside its evidence run")
    if parsed["values"]["xdna_emu_commit"] != values["xdna_emu_commit"]:
        raise ValueError("KVM control marker source commit differs from tuple")
    return {
        "run": run,
        "module": module.resolve(),
        "tuple": tuple_path.resolve(),
        "disposition": disposition["reason"],
    }


def parse_tuple(path: Path) -> dict:
    values = {}
    files = {}
    for line in path.read_text().splitlines():
        file_match = re.fullmatch(r"([0-9a-f]{64})  (/.+)", line)
        if file_match:
            file_path = Path(file_match.group(2)).resolve()
            digest = file_match.group(1)
            if file_path in files and files[file_path] != digest:
                raise ValueError(f"conflicting tuple file {file_path}")
            files[file_path] = digest
            continue
        value_match = re.fullmatch(r"([a-z0-9_]+)=(.*)", line)
        if value_match:
            key = value_match.group(1)
            if key in values:
                raise ValueError(f"duplicate tuple value {key}")
            values[key] = value_match.group(2)
    return {"values": values, "files": files}


def tuple_artifact(parsed: dict, suffix: str) -> Path:
    matches = [
        (path, digest) for path, digest in parsed["files"].items()
        if str(path).endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"tuple does not pin exactly one {suffix}")
    path, digest = matches[0]
    if not path.is_file() or _sha256(path) != digest:
        raise ValueError(f"tuple hash does not match {path}")
    return path


def npi_lifecycle_ok(log: str) -> bool:
    events = re.findall(
        r"xdna_mailbox\.(\d+): opcode 0x203 size (24|8) id (\S+)$",
        log,
        re.MULTILINE,
    )
    return (
        len(events) == 2
        and events[0][1] == "24"
        and events[1][1] == "8"
        and events[0][0] == events[1][0]
        and events[0][2] == events[1][2]
        and "command opcode 0x203 failed" not in log
    )


def resolve_npi_kvm_run(run: Path, repository: Path) -> dict:
    repository = repository.resolve(strict=True)
    run = run.resolve(strict=True)
    root = (repository / "build/experiments/phoenix-vfio-user").resolve()
    if not run.is_relative_to(root):
        raise ValueError("NPI KVM run is outside repository evidence")

    tuple_path = run / "tuple.txt"
    guest_path = run / "guest.log"
    dmesg_path = run / "dmesg.log"
    if not all(path.is_file() for path in (tuple_path, guest_path, dmesg_path)):
        raise ValueError("NPI KVM evidence is incomplete")
    parsed = parse_tuple(tuple_path)
    values = parsed["values"]
    expected = {
        "driver_commit": NPI_DRIVER_PIN,
        "research_probe": "phoenix-read-only-npi-lock",
        "expected_management_request": NPI_REQUEST,
        "expected_system_read": NPI_SYSTEM_READ,
    }
    if any(values.get(key) != value for key, value in expected.items()):
        raise ValueError("NPI KVM tuple differs from the exact qualified operation")
    if not re.fullmatch(r"[0-9a-f]{40}", values.get("xdna_emu_commit", "")):
        raise ValueError("NPI KVM tuple has no exact emulator commit")
    if not values.get("guest_kernel_version"):
        raise ValueError("NPI KVM tuple has no guest kernel")

    module = tuple_artifact(
        parsed, "driver-source/drivers/accel/amdxdna/amdxdna.ko",
    )
    patch = tuple_artifact(
        parsed, "docs/patches/0004-LOCAL-phoenix-read-only-npi-lock-probe.patch",
    )
    firmware = tuple_artifact(
        parsed, "/usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin",
    )
    if parsed["files"][firmware] != NPI_FIRMWARE_SHA256:
        raise ValueError("NPI KVM firmware differs from Phoenix 1.5.5.391")
    if module != (run / "driver-source/drivers/accel/amdxdna/amdxdna.ko").resolve():
        raise ValueError("NPI KVM module is outside its evidence run")
    if patch != (
        repository / "docs/patches/0004-LOCAL-phoenix-read-only-npi-lock-probe.patch"
    ).resolve():
        raise ValueError("NPI KVM tuple does not pin the repository patch")

    guest = guest_path.read_text()
    markers = (
        "PHOENIX_NPI_READ_BEGIN",
        "PHOENIX_NPI_READ value=0x00000000",
        "PHOENIX_NPI_READ_PASS",
        "PHOENIX_DRIVER_PROBE_PASS",
    )
    lines = guest.splitlines()
    positions = [
        lines.index(marker) for marker in markers if lines.count(marker) == 1
    ]
    if (
        len(positions) != len(markers)
        or positions != sorted(positions)
        or any(line.startswith("PHOENIX_DRIVER_PROBE_FAIL:") for line in lines)
    ):
        raise ValueError("NPI KVM guest qualification markers differ")
    if not npi_lifecycle_ok(dmesg_path.read_text()):
        raise ValueError("NPI KVM mailbox lifecycle differs")
    return {
        "run": run,
        "module": module,
        "patch": patch,
        "firmware": firmware,
        "tuple": tuple_path.resolve(),
        "guest_log": guest_path.resolve(),
        "dmesg_log": dmesg_path.resolve(),
        "tuple_values": values,
    }


def resolve_host_artifacts(pair: Path, kvm_control: dict) -> dict:
    pair.resolve(strict=True)
    parsed = parse_tuple(kvm_control["tuple"])
    artifacts = {
        "module": tuple_artifact(
            parsed, "driver-source/drivers/accel/amdxdna/amdxdna.ko",
        ),
        "runner": tuple_artifact(
            parsed, "bridge-runner/build/bridge-trace-runner",
        ),
        "classifier": tuple_artifact(parsed, "target/debug/libxdna_emu.so"),
        "clock_query": tuple_artifact(parsed, "/xdna-clock-query"),
        "xclbin": tuple_artifact(parsed, "fault-package/aie.xclbin"),
        "mlir": tuple_artifact(
            parsed, "fault-package/work/input_with_addresses.mlir",
        ),
        "expected_output": tuple_artifact(parsed, "hw.out.bin"),
        "gate_instructions": tuple_artifact(parsed, "/active-gate.insts.bin"),
        "canary_instructions": tuple_artifact(
            parsed, "full-witness-fault.insts.bin",
        ),
        "firmware": tuple_artifact(
            parsed, "/usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin",
        ),
        "register_db": tuple_artifact(
            parsed, "/mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json",
        ),
        "npi_patch": tuple_artifact(
            parsed, "docs/patches/0004-LOCAL-phoenix-read-only-npi-lock-probe.patch",
        ),
        "protected_gate_patch": tuple_artifact(
            parsed, "docs/patches/0005-LOCAL-phoenix-protected-column-gate.patch",
        ),
        "xrt_coreutil": tuple_artifact(
            parsed, "/opt/xilinx/xrt/lib/libxrt_coreutil.so.2.26.0",
        ),
        "xrt_core": tuple_artifact(
            parsed, "/opt/xilinx/xrt/lib/libxrt_core.so.2.26.0",
        ),
        "xrt_driver": tuple_artifact(
            parsed, "/opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2.26.0",
        ),
    }
    if artifacts["module"] != kvm_control["module"]:
        raise ValueError("tuple module is not the safety-qualified KVM module")
    if _sha256(artifacts["firmware"]) != NPI_FIRMWARE_SHA256:
        raise ValueError("firmware differs from pinned Phoenix 1.5.5.391")
    mlir_aie = artifacts["register_db"].parents[4]
    if mlir_aie.name != "mlir-aie":
        raise ValueError("cannot derive mlir-aie root from pinned register DB")
    parser_python = mlir_aie / "ironenv/bin/python3"
    if not parser_python.is_file():
        raise ValueError("mlir-aie parser Python is missing")
    artifacts["parser_python"] = parser_python
    artifacts["tuple_values"] = parsed["values"]
    return artifacts


def resolve_host_control(pair: Path) -> Path:
    pair = pair.resolve(strict=True)
    marker = pair / "host/control-behavior-qualified"
    key, separator, value = marker.read_text().strip().partition("=")
    if key != "run" or not separator or not value:
        raise ValueError("host control marker is malformed")
    run = Path(value).resolve(strict=True)
    if not run.is_relative_to((pair / "host").resolve()) or not run.name.startswith(
        "control-"
    ):
        raise ValueError("host control marker points outside pair host evidence")
    status = json.loads((run / "status.json").read_text())
    result = json.loads((run / "result.json").read_text())
    if not host_run_pass("control", status, result):
        raise ValueError("host control is not restored and behaviorally qualified")
    return run


def build_host_request(
    arm: str,
    pair: Path,
    repository: Path,
    run_dir: Path,
    environment: dict,
    kernel_release: str,
) -> dict:
    if arm not in {"control", "treatment"}:
        raise ValueError("host arm must be control or treatment")
    if "XDNA_EMU" in environment:
        raise ValueError("XDNA_EMU must be unset for physical execution")
    pair = pair.resolve(strict=True)
    repository = repository.resolve(strict=True)
    run_dir = run_dir.resolve()
    host_root = (pair / "host").resolve()
    if not run_dir.is_relative_to(host_root) or not run_dir.name.startswith(
        f"{arm}-"
    ):
        raise ValueError("host run directory escapes its arm evidence root")
    kvm_control = resolve_kvm_control(pair)
    artifacts = resolve_host_artifacts(pair, kvm_control)
    expected_kernel = artifacts["tuple_values"].get("guest_kernel_version")
    if kernel_release != expected_kernel:
        raise ValueError(
            f"host kernel {kernel_release} does not match KVM {expected_kernel}"
        )
    host_control = resolve_host_control(pair) if arm == "treatment" else None
    pinned = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in artifacts.items()
        if name != "tuple_values"
    }
    return {
        "schema_version": 1,
        "arm": arm,
        "pair": str(pair),
        "repository": str(repository),
        "run_dir": str(run_dir),
        "kernel_release": kernel_release,
        "environment": {
            key: environment[key]
            for key in ("XDNA_EMU", "XDNA_EMU_RUNTIME")
            if key in environment
        },
        "kvm_control_run": str(kvm_control["run"]),
        "kvm_disposition": kvm_control["disposition"],
        "host_control_run": str(host_control) if host_control else None,
        "tuple_values": artifacts["tuple_values"],
        "artifacts": pinned,
        "harness": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }


def build_npi_host_request(
    kvm_run: Path,
    repository: Path,
    run_dir: Path,
    environment: dict,
    kernel_release: str,
) -> dict:
    if "XDNA_EMU" in environment:
        raise ValueError("XDNA_EMU must be unset for physical execution")
    repository = repository.resolve(strict=True)
    run_dir = run_dir.resolve()
    host_root = (repository / "build/experiments/phoenix-npi-read-host").resolve()
    if not run_dir.is_relative_to(host_root) or not run_dir.name.startswith("read-"):
        raise ValueError("NPI host run directory escapes its evidence root")
    qualified = resolve_npi_kvm_run(kvm_run, repository)
    if kernel_release != qualified["tuple_values"]["guest_kernel_version"]:
        raise ValueError(
            f"host kernel {kernel_release} does not match KVM "
            f"{qualified['tuple_values']['guest_kernel_version']}"
        )
    paths = {
        name: qualified[name]
        for name in ("module", "patch", "firmware", "tuple", "guest_log", "dmesg_log")
    }
    paths.update({
        "xrt_coreutil": Path(
            "/opt/xilinx/xrt/lib/libxrt_coreutil.so.2"
        ).resolve(strict=True),
        "xrt_core": Path("/opt/xilinx/xrt/lib/libxrt_core.so.2").resolve(
            strict=True,
        ),
        "xrt_driver": Path(
            "/opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2"
        ).resolve(strict=True),
        "xrt_smi": Path("/opt/xilinx/xrt/bin/xrt-smi").resolve(strict=True),
        "xrt_validation_archive": XRT_VALIDATION_ARCHIVE.resolve(strict=True),
    })
    return {
        "schema_version": 1,
        "operation": "phoenix_npi_read",
        "repository": str(repository),
        "run_dir": str(run_dir),
        "kernel_release": kernel_release,
        "environment": {
            key: environment[key]
            for key in ("XDNA_EMU", "XDNA_EMU_RUNTIME")
            if key in environment
        },
        "kvm_run": str(qualified["run"]),
        "tuple_values": qualified["tuple_values"],
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "harness": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }


def validate_host_request(request: dict) -> None:
    rebuilt = build_host_request(
        request.get("arm"),
        Path(request.get("pair", "")),
        Path(request.get("repository", "")),
        Path(request.get("run_dir", "")),
        request.get("environment", {}),
        request.get("kernel_release", ""),
    )
    if request != rebuilt:
        raise ValueError("host request does not match current pinned evidence")


def validate_npi_host_request(request: dict) -> None:
    rebuilt = build_npi_host_request(
        Path(request.get("kvm_run", "")),
        Path(request.get("repository", "")),
        Path(request.get("run_dir", "")),
        request.get("environment", {}),
        request.get("kernel_release", ""),
    )
    if request != rebuilt:
        raise ValueError("NPI host request does not match current pinned evidence")


def module_transaction_commands(module: Path) -> dict:
    return {
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


def module_parameters(root: Path) -> dict[str, str | None]:
    return {
        name: (path.read_text().strip() if path.is_file() else None)
        for name in ("tdr_timeout_ms", "force_cmdlist")
        for path in (root / name,)
    }


def restore_module_parameters(root: Path, expected: dict[str, str | None]) -> None:
    for name, value in expected.items():
        path = root / name
        if value is None:
            if path.exists():
                raise RuntimeError(f"amdxdna parameter {name} unexpectedly exists")
            continue
        if not path.is_file():
            raise RuntimeError(f"amdxdna parameter {name} disappeared")
        if path.read_text().strip() != value:
            path.write_text(value + "\n")
        if path.read_text().strip() != value:
            raise RuntimeError(f"failed to restore amdxdna {name}")


def runner_argv(
    paths: dict,
    placement: tuple[int, int],
    gate_arm: str | None = None,
) -> tuple[str, ...]:
    command = (
        "timeout", "-k", "5", "650", str(paths["runner"]),
        "--xclbin", str(paths["xclbin"]),
        "--instr", str(paths["instructions"]),
        "--trace-out", str(paths["trace"]),
        "--output", str(paths["output"]),
        "--qos-gops", "1", "--qos-fps", "1000",
        "--expect-placement", f"{placement[0]}:{placement[1]}",
    )
    if gate_arm is not None:
        command += ("--phoenix-column-gate", gate_arm)
    return (*command, "-v")


def worker_environment(request: dict, user: str, home: str) -> dict:
    classifier = Path(request["artifacts"]["classifier"]["path"])
    return {
        "HOME": home,
        "USER": user,
        "LOGNAME": user,
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "XILINX_XRT": "/opt/xilinx/xrt",
        "LD_LIBRARY_PATH": "/opt/xilinx/xrt/lib",
        "XDNA_EMU_DIR": str(classifier.parent),
        "BRIDGE_RUNNER_ASYNC_CTX": "0",
        "BRIDGE_RUNNER_REUSE_CONTEXT": "0",
    }


def gate_worker_argv(
    request: dict,
    request_path: Path,
    request_sha256: str,
    environment: dict,
) -> tuple[str, ...]:
    return (
        "timeout", "-k", "10", "1400", "env", "-i",
        *(f"{key}={value}" for key, value in environment.items()),
        "nice", "-n", "19",
        request["artifacts"]["parser_python"]["path"],
        request["harness"]["path"], "_worker",
        str(request_path), request_sha256,
    )


def validate_parser_python(executable: Path) -> None:
    try:
        result = subprocess.run(
            (str(executable), "-I", "-c", "import numpy"),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(
            f"pinned parser Python cannot import NumPy: {error}"
        ) from error
    if result.returncode:
        detail = result.stderr.strip() or f"status {result.returncode}"
        raise RuntimeError(f"pinned parser Python cannot import NumPy: {detail}")


def stage_npi_canary_home(
    run_dir: Path,
    xrt_coreutil: Path,
    archive: Path,
) -> Path:
    match = re.fullmatch(r"libxrt_coreutil\.so\.(\d+\.\d+\.\d+)", xrt_coreutil.name)
    if match is None:
        raise ValueError("cannot derive XRT version from libxrt_coreutil")
    archive = archive.resolve(strict=True)
    home = run_dir / "xrt-canary-home"
    staged = (
        home / ".local/share/xrt" / match.group(1)
        / "amdxdna/bins" / archive.name
    )
    staged.parent.mkdir(parents=True, exist_ok=True)
    if staged.is_symlink():
        if staged.resolve(strict=True) != archive:
            raise ValueError("staged XRT validation archive points elsewhere")
    elif staged.exists():
        raise ValueError("staged XRT validation archive is not a symlink")
    else:
        staged.symlink_to(archive)
    return home


def npi_canary_argv(
    xrt_smi: Path,
    user: str,
    home: str,
) -> tuple[str, ...]:
    return (
        "timeout", "-k", "5", "120", "runuser", "-u", user, "--",
        "env", "-i", f"HOME={home}", f"USER={user}", f"LOGNAME={user}",
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "XILINX_XRT=/opt/xilinx/xrt",
        "LD_LIBRARY_PATH=/opt/xilinx/xrt/lib",
        str(xrt_smi), "validate",
    )


def lifecycle_ok(log: str) -> bool:
    requests = 0
    responses = 0
    destroyed_between = False
    bad_order = False
    for line in log.splitlines():
        if re.search(r"xdna_mailbox\.\d+: opcode 0x18 size 24 id ", line):
            requests += 1
            if requests == 2 and not destroyed_between:
                bad_order = True
        if re.search(r"xdna_mailbox\.\d+: opcode 0x18 size 12 id ", line):
            responses += 1
        if (
            requests == 1
            and re.search(r"xdna_mailbox\.\d+: opcode 0x3 size 4 id ", line)
        ):
            destroyed_between = True
    return (
        requests == 2 and responses == 2 and destroyed_between and not bad_order
    )


def host_behavioral_pass(arm: str, classification: dict) -> bool:
    expected = {"control": "control", "treatment": "freeze_resume"}
    return (
        classification.get("qualified") is True
        and classification.get("reason") == expected.get(arm)
    )


def kernel_log_between(
    log: str,
    token: str,
    prefix: str = "PHOENIX_REAL_COLUMN_GATE_HOST",
) -> str:
    begin = f"{prefix}_BEGIN {token}"
    end = f"{prefix}_END {token}"
    lines = log.splitlines()
    begins = [index for index, line in enumerate(lines) if begin in line]
    ends = [index for index, line in enumerate(lines) if end in line]
    if len(begins) != 1 or len(ends) != 1 or begins[0] >= ends[0]:
        raise ValueError("physical run kernel-log markers are missing or ambiguous")
    return "\n".join(lines[begins[0] + 1:ends[0]]) + "\n"


def host_run_pass(arm: str, status: dict, result: dict) -> bool:
    return (
        status.get("state") == "complete"
        and status.get("restored") is True
        and status.get("lifecycle_ok") is True
        and host_behavioral_pass(arm, result.get("classification", {}))
    )


def npi_host_run_pass(status: dict, result: dict) -> bool:
    return (
        status.get("state") == "complete"
        and status.get("restored") is True
        and status.get("lifecycle_ok") is True
        and status.get("canary_ok") is True
        and result.get("qualified") is True
        and re.fullmatch(r"0x[0-9a-f]{8}", result.get("value", "")) is not None
    )


def _write_json(path: Path, value: dict, uid: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    if uid is not None:
        os.chown(path, uid, -1)


def _write_text(path: Path, value: str, uid: int | None = None) -> None:
    path.write_text(value)
    if uid is not None:
        os.chown(path, uid, -1)


def _load_source(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_evidence(repository: Path):
    tools = str(repository / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import npu1_firmware_evidence
    return npu1_firmware_evidence


def _command_data(result) -> dict:
    return {
        "argv": list(result.argv),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "timed_out": result.timed_out,
    }


def _module_identity(evidence, path: Path) -> dict:
    path = path.resolve(strict=True)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "srcversion": evidence._modinfo("srcversion", path),
        "vermagic": evidence._modinfo("vermagic", path),
        "build_id": evidence._elf_build_id(path),
        **{
            field: evidence._modinfo(field, path)
            for field in ("signer", "sig_id", "sig_key", "sig_hashalgo")
        },
    }


def module_signature_matches(candidate: dict, original: dict) -> bool:
    fields = ("signer", "sig_id", "sig_key", "sig_hashalgo")
    return (
        candidate.get("sig_id") == "PKCS#7"
        and all(candidate.get(field) == original.get(field) for field in fields)
    )


def active_clients_before_restore(evidence, bdf: str) -> int:
    if evidence._loaded_srcversion() is None:
        return 0
    return evidence._active_npu_clients(evidence._device_node_for_bdf(bdf))


def _assert_repository_inputs(request: dict) -> None:
    repository = Path(request["repository"])
    status = subprocess.run(
        ("git", "-C", str(repository), "status", "--porcelain", "--untracked-files=all"),
        capture_output=True, text=True, check=False, timeout=10,
    )
    if status.returncode or status.stdout:
        raise RuntimeError("physical execution requires a clean source worktree")
    pin = request["tuple_values"].get("xdna_emu_commit")
    if not re.fullmatch(r"[0-9a-f]{40}", pin or ""):
        raise RuntimeError("KVM tuple has no exact xdna-emu commit")
    helpers = (
        "tools/phoenix-pm-clock-characterize.py",
        "tools/trace_runner.py",
        "tools/parse-trace.py",
        "tools/trace_decoder",
        "tools/npu1_firmware_evidence.py",
    )
    unchanged = subprocess.run(
        ("git", "-C", str(repository), "diff", "--quiet", pin, "--", *helpers),
        check=False, timeout=10,
    )
    if unchanged.returncode != 0:
        raise RuntimeError("host helpers differ from the safety-qualified KVM commit")


def _host_snapshot(request: dict, privileged: bool) -> dict:
    if request.get("arm") in {"control", "treatment"}:
        validate_parser_python(
            Path(request["artifacts"]["parser_python"]["path"]),
        )
    repository = Path(request["repository"])
    evidence = _load_evidence(repository)
    bdf, pci = evidence._physical_npu()
    device = evidence._device_node_for_bdf(bdf)
    if device != Path("/dev/accel/accel0") or not device.is_char_device():
        raise RuntimeError("the pinned Phoenix device is not /dev/accel/accel0")
    if evidence._active_npu_clients(device):
        raise RuntimeError("/dev/accel/accel0 has an active client")

    original_path = Path(evidence._modinfo("filename", "amdxdna"))
    original = _module_identity(evidence, original_path)
    candidate = _module_identity(
        evidence, Path(request["artifacts"]["module"]["path"]),
    )
    if request["kernel_release"] != os.uname().release:
        raise RuntimeError("host kernel changed after request construction")
    if (
        candidate["vermagic"] != original["vermagic"]
        or not candidate["vermagic"].startswith(request["kernel_release"] + " ")
    ):
        raise RuntimeError("candidate and installed module vermagic differ")
    if not module_signature_matches(candidate, original):
        raise RuntimeError("candidate signature does not match installed module trust")
    if "amdnpu/1502_00/npu.dev.sbin" not in evidence._modinfo(
        "firmware", candidate["path"],
    ).splitlines():
        raise RuntimeError("candidate module does not request pinned Phoenix firmware")

    loaded = {
        "srcversion": evidence._loaded_srcversion(),
        "build_id": evidence._loaded_build_id(),
    }
    if loaded != {
        "srcversion": original["srcversion"],
        "build_id": original["build_id"],
    }:
        raise RuntimeError("loaded module is not the installed module")

    aliases = {
        "xrt_coreutil": "/opt/xilinx/xrt/lib/libxrt_coreutil.so.2",
        "xrt_core": "/opt/xilinx/xrt/lib/libxrt_core.so.2",
        "xrt_driver": "/opt/xilinx/xrt/lib/libxrt_driver_xdna.so.2",
    }
    for name, alias in aliases.items():
        if Path(alias).resolve(strict=True) != Path(
            request["artifacts"][name]["path"],
        ):
            raise RuntimeError(f"runtime {name} alias does not select its pinned XRT file")

    snapshot = {
        "bdf": bdf,
        "device": str(device),
        "original_module": original,
        "candidate_module": candidate,
        "loaded_module": loaded,
        "power_control": (pci / "power/control").read_text().strip(),
    }
    if privileged:
        snapshot["parameters"] = module_parameters(
            Path("/sys/module/amdxdna/parameters"),
        )
    return snapshot


def _run_logged(argv: tuple[str, ...], stdout_path: Path, stderr_path: Path) -> int:
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        return subprocess.run(
            argv, stdout=stdout, stderr=stderr, check=False,
        ).returncode


def _query_clock(executable: Path, prefix: Path) -> dict:
    rc = _run_logged(
        ("timeout", "-k", "2", "10", str(executable)),
        prefix.with_suffix(".json"), prefix.with_suffix(".stderr"),
    )
    if rc:
        raise RuntimeError(f"clock query failed with status {rc}")
    return json.loads(prefix.with_suffix(".json").read_text())


def _run_worker(request_path: Path, request_sha256: str) -> int:
    if os.geteuid() != 0:
        print(
            "physical gate worker must run in the privileged transaction",
            file=sys.stderr,
        )
        return 2
    if _sha256(request_path) != request_sha256:
        print("physical worker request SHA-256 mismatch", file=sys.stderr)
        return 2
    request = json.loads(request_path.read_text())
    validate_host_request(request)
    if "XDNA_EMU" in os.environ:
        raise RuntimeError("XDNA_EMU leaked into physical worker")

    repository = Path(request["repository"])
    run_dir = Path(request["run_dir"])
    artifacts = {
        name: Path(value["path"])
        for name, value in request["artifacts"].items()
    }
    result_path = run_dir / "result.json"
    arm = request["arm"]
    try:
        mismatch = {
            "runner": artifacts["runner"],
            "xclbin": artifacts["xclbin"],
            "instructions": artifacts["gate_instructions"],
            "trace": run_dir / "mismatch.trace.bin",
            "output": run_dir / "mismatch.out.bin",
        }
        mismatch_rc = _run_logged(
            runner_argv(mismatch, (2, 1)), run_dir / "mismatch.stdout",
            run_dir / "mismatch.stderr",
        )
        mismatch_error = mismatch["trace"].exists() or mismatch["output"].exists()
        expected_mismatch = (
            "live hardware context placement mismatch: expected 2:1, got 1:1"
        )
        if (
            mismatch_rc == 0
            or mismatch_error
            or expected_mismatch not in (run_dir / "mismatch.stderr").read_text()
        ):
            raise RuntimeError("live-placement mismatch guard did not stop submission")

        clock_before = _query_clock(
            artifacts["clock_query"], run_dir / "clock-before",
        )
        arm_paths = {
            "runner": artifacts["runner"],
            "xclbin": artifacts["xclbin"],
            "instructions": artifacts["gate_instructions"],
            "trace": run_dir / "arm.trace.bin",
            "output": run_dir / "arm.out.bin",
        }
        arm_rc = _run_logged(
            runner_argv(arm_paths, (1, 1), arm), run_dir / "arm.stdout",
            run_dir / "arm.stderr",
        )
        clock_after = None
        clock_error = None
        try:
            clock_after = _query_clock(
                artifacts["clock_query"], run_dir / "clock-after",
            )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            clock_error = error

        canary = {
            "runner": artifacts["runner"],
            "xclbin": artifacts["xclbin"],
            "instructions": artifacts["canary_instructions"],
            "trace": run_dir / "canary.trace.bin",
            "output": run_dir / "canary.out.bin",
        }
        canary_rc = _run_logged(
            runner_argv(canary, (1, 1)), run_dir / "canary.stdout",
            run_dir / "canary.stderr",
        )
        if clock_error is not None:
            raise RuntimeError(f"after-run clock query failed: {clock_error}")
        required = (
            "verified live hw_context placement 1:1",
            f"classifier loaded from {artifacts['classifier']}",
            "classifier roles: arg0=data_mm2s arg2=data_s2mm arg3=data_s2mm",
        )
        for name, rc in (("arm", arm_rc), ("canary", canary_rc)):
            stderr = (run_dir / f"{name}.stderr").read_text()
            if rc or any(text not in stderr for text in required):
                raise RuntimeError(f"{name} command or provenance check failed ({rc})")

        tools = repository / "tools"
        if str(tools) not in sys.path:
            sys.path.insert(0, str(tools))
        import trace_runner
        mlir_aie = artifacts["register_db"].parents[4]
        if mlir_aie.name != "mlir-aie":
            raise RuntimeError("cannot derive mlir-aie root from pinned register DB")
        trace_runner.MLIR_AIE_ROOT = mlir_aie
        trace_runner.PARSE_TOOL = tools / "parse-trace.py"
        parse_ok, parse_error, cycles, event_count = trace_runner._parse_trace_bin(
            arm_paths["trace"], artifacts["mlir"], run_dir / "arm.events.json",
            run_dir / "arm.cycles.txt", run_dir / "parser.log",
            os.environ.copy(),
        )
        if not parse_ok:
            raise RuntimeError(parse_error or "trace parse failed")

        pm = _load_source(
            "phoenix_pm_clock_characterize_host",
            tools / "phoenix-pm-clock-characterize.py",
        )
        result = pm.classify_real_column_gate_artifacts(
            arm, run_dir / "arm.events.json", arm_paths["output"],
            artifacts["expected_output"], run_dir / "clock-before.json",
            run_dir / "clock-after.json", canary["output"],
        )
        result["host_checks"] = {
            "mismatch_guard": True,
            "arm_returncode": arm_rc,
            "canary_returncode": canary_rc,
            "parse_cycles": cycles,
            "parse_event_count": event_count,
            "clock_before": clock_before,
            "clock_after": clock_after,
        }
        _write_json(result_path, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if host_behavioral_pass(arm, result["classification"]) else 1
    except Exception as error:  # noqa: BLE001 - preserve any worker failure
        if not result_path.exists():
            _write_json(result_path, {
                "schema_version": 1,
                "arm": arm,
                "qualified": False,
                "classification": {"qualified": False, "reason": "worker_failure"},
                "error": f"{type(error).__name__}: {error}",
            })
        print(f"physical worker failed: {error}", file=sys.stderr)
        return 1


def _write_kmsg(marker: str) -> None:
    with Path("/dev/kmsg").open("w") as output:
        output.write(marker + "\n")


def _run_privileged(request_path: Path, request_sha256: str) -> int:
    if os.geteuid() != 0:
        print("physical module transaction must run as root", file=sys.stderr)
        return 2
    if (
        not re.fullmatch(r"[0-9a-f]{64}", request_sha256)
        or _sha256(request_path) != request_sha256
    ):
        print("physical request SHA-256 mismatch", file=sys.stderr)
        return 2
    request = json.loads(request_path.read_text())
    run_dir = Path(request["run_dir"])
    owner_text = os.environ.get("PKEXEC_UID", "")
    if not owner_text.isdecimal():
        print("PKEXEC_UID is missing", file=sys.stderr)
        return 2
    owner_uid = int(owner_text)
    if (
        run_dir.stat().st_uid != owner_uid
        or request_path.stat().st_uid != owner_uid
        or not request_path.resolve().is_relative_to(run_dir.resolve())
    ):
        print("physical request ownership or confinement failed", file=sys.stderr)
        return 2

    status_path = run_dir / "status.json"
    repository = Path(request["repository"])
    evidence = _load_evidence(repository)
    try:
        validate_host_request(request)
        _assert_repository_inputs(request)
        initial = _host_snapshot(request, privileged=True)
    except Exception as error:  # noqa: BLE001 - fail closed before mutation
        _write_json(status_path, {
            "state": "pretraffic_preflight_failed",
            "restored": True,
            "errors": [f"{type(error).__name__}: {error}"],
        }, owner_uid)
        return 2

    _write_json(status_path, {
        "state": "running",
        "restored": False,
        "initial": initial,
    }, owner_uid)
    commands = []
    errors = []
    worker_rc = None
    lifecycle = False
    restored = False
    swap_started = False
    marker_started = False
    token = request_sha256

    def checked(argv: tuple[str, ...], timeout: int = 30):
        result = evidence._run_command(argv, timeout)
        commands.append(_command_data(result))
        if result.returncode or result.timed_out:
            detail = result.stderr.strip() or str(result.returncode)
            raise RuntimeError(f"{' '.join(argv)} failed: {detail}")
        return result

    try:
        setup = module_transaction_commands(
            Path(request["artifacts"]["module"]["path"]),
        )["setup"]
        checked(setup[0])
        swap_started = True
        checked(setup[1])
        checked(("udevadm", "settle", "--timeout=5"), 10)

        candidate = initial["candidate_module"]
        bdf = initial["bdf"]
        device = evidence._device_node_for_bdf(bdf)
        runtime = {
            "loaded_srcversion": evidence._loaded_srcversion(),
            "loaded_build_id": evidence._loaded_build_id(),
            "device": str(device),
            "device_present": device.is_char_device(),
            "active_clients": evidence._active_npu_clients(device),
            "tdr_timeout_ms": Path(
                "/sys/module/amdxdna/parameters/tdr_timeout_ms"
            ).read_text().strip(),
            "force_cmdlist": Path(
                "/sys/module/amdxdna/parameters/force_cmdlist"
            ).read_text().strip(),
            "force_iova": Path(
                "/sys/module/amdxdna/parameters/force_iova"
            ).read_text().strip(),
        }
        _write_json(run_dir / "runtime-preflight.json", runtime, owner_uid)
        if runtime != {
            "loaded_srcversion": candidate["srcversion"],
            "loaded_build_id": candidate["build_id"],
            "device": "/dev/accel/accel0",
            "device_present": True,
            "active_clients": 0,
            "tdr_timeout_ms": "2000",
            "force_cmdlist": "Y",
            "force_iova": "N",
        }:
            raise RuntimeError("experimental module runtime preflight differed")

        _write_kmsg(f"PHOENIX_REAL_COLUMN_GATE_HOST_BEGIN {token}")
        marker_started = True
        owner = pwd.getpwuid(owner_uid)
        environment = worker_environment(request, owner.pw_name, owner.pw_dir)
        worker = gate_worker_argv(
            request, request_path, request_sha256, environment,
        )
        result = evidence._run_command(worker, 1420)
        worker_rc = result.returncode
        _write_text(run_dir / "worker.stdout", result.stdout, owner_uid)
        _write_text(run_dir / "worker.stderr", result.stderr, owner_uid)
        commands.append(_command_data(result))
    except Exception as error:  # noqa: BLE001 - restoration belongs in finally
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        if marker_started:
            try:
                _write_kmsg(f"PHOENIX_REAL_COLUMN_GATE_HOST_END {token}")
                dmesg = evidence._command_stdout(("dmesg", "--raw"), 15)
                kernel_log = kernel_log_between(dmesg, token)
                _write_text(run_dir / "dmesg.log", kernel_log, owner_uid)
                lifecycle = lifecycle_ok(kernel_log)
                if not lifecycle:
                    errors.append("submission/canary lifecycle differed")
            except Exception as error:  # noqa: BLE001 - preserve kernel evidence
                errors.append(f"kernel evidence failed: {error}")

        if swap_started:
            try:
                if active_clients_before_restore(evidence, initial["bdf"]):
                    raise RuntimeError("active NPU client blocks safe module restoration")
                restore = module_transaction_commands(
                    Path(request["artifacts"]["module"]["path"]),
                )["restore"]
                if evidence._loaded_srcversion() is not None:
                    checked(restore[0])
                checked(restore[1])
                checked(("udevadm", "settle", "--timeout=5"), 10)
                restore_module_parameters(
                    Path("/sys/module/amdxdna/parameters"), initial["parameters"],
                )
                bdf, pci = evidence._physical_npu()
                if (pci / "power/control").read_text().strip() != initial["power_control"]:
                    (pci / "power/control").write_text(initial["power_control"] + "\n")
                device = evidence._device_node_for_bdf(bdf)
                original = initial["original_module"]
                restored = (
                    evidence._loaded_srcversion() == original["srcversion"]
                    and evidence._loaded_build_id() == original["build_id"]
                    and device == Path("/dev/accel/accel0")
                    and device.is_char_device()
                    and (pci / "power/control").read_text().strip()
                    == initial["power_control"]
                )
                if not restored:
                    raise RuntimeError("installed amdxdna identity did not restore")
            except Exception as error:  # noqa: BLE001 - report; never auto-reset
                errors.append(f"module restoration failed: {error}")
        else:
            restored = True

    state = (
        "complete"
        if worker_rc == 0 and lifecycle and restored and not errors
        else "failed"
    )
    _write_json(status_path, {
        "state": state,
        "restored": restored,
        "lifecycle_ok": lifecycle,
        "worker_returncode": worker_rc,
        "initial": initial,
        "commands": commands,
        "errors": errors,
    }, owner_uid)
    return 0 if state == "complete" else 1


def _run_npi_privileged(request_path: Path, request_sha256: str) -> int:
    if os.geteuid() != 0:
        print("physical NPI transaction must run as root", file=sys.stderr)
        return 2
    if (
        not re.fullmatch(r"[0-9a-f]{64}", request_sha256)
        or _sha256(request_path) != request_sha256
    ):
        print("physical NPI request SHA-256 mismatch", file=sys.stderr)
        return 2
    request = json.loads(request_path.read_text())
    run_dir = Path(request["run_dir"])
    owner_text = os.environ.get("PKEXEC_UID", "")
    if not owner_text.isdecimal():
        print("PKEXEC_UID is missing", file=sys.stderr)
        return 2
    owner_uid = int(owner_text)
    if (
        run_dir.stat().st_uid != owner_uid
        or request_path.stat().st_uid != owner_uid
        or not request_path.resolve().is_relative_to(run_dir.resolve())
    ):
        print("physical NPI request ownership or confinement failed", file=sys.stderr)
        return 2

    status_path = run_dir / "status.json"
    repository = Path(request["repository"])
    evidence = _load_evidence(repository)
    try:
        validate_npi_host_request(request)
        _assert_repository_inputs(request)
        initial = _host_snapshot(request, privileged=True)
    except Exception as error:  # noqa: BLE001 - fail closed before mutation
        _write_json(status_path, {
            "state": "pretraffic_preflight_failed",
            "restored": True,
            "lifecycle_ok": False,
            "canary_ok": False,
            "errors": [f"{type(error).__name__}: {error}"],
        }, owner_uid)
        return 2

    _write_json(status_path, {
        "state": "running",
        "restored": False,
        "lifecycle_ok": False,
        "canary_ok": False,
        "initial": initial,
    }, owner_uid)
    commands = []
    errors = []
    value = None
    lifecycle = False
    canary_ok = False
    restored = False
    swap_started = False
    marker_started = False
    token = request_sha256

    def checked(argv: tuple[str, ...], timeout: int = 30):
        result = evidence._run_command(argv, timeout)
        commands.append(_command_data(result))
        if result.returncode or result.timed_out:
            detail = result.stderr.strip() or str(result.returncode)
            raise RuntimeError(f"{' '.join(argv)} failed: {detail}")
        return result

    def close_marker() -> None:
        nonlocal lifecycle, marker_started
        if not marker_started:
            return
        try:
            _write_kmsg(f"PHOENIX_NPI_READ_HOST_END {token}")
            dmesg = evidence._command_stdout(("dmesg", "--raw"), 15)
            kernel_log = kernel_log_between(
                dmesg, token, "PHOENIX_NPI_READ_HOST",
            )
            _write_text(run_dir / "dmesg.log", kernel_log, owner_uid)
            lifecycle = npi_lifecycle_ok(kernel_log)
        except Exception as error:  # noqa: BLE001 - preserve kernel evidence
            errors.append(f"kernel evidence failed: {error}")
        finally:
            marker_started = False

    try:
        setup = module_transaction_commands(
            Path(request["artifacts"]["module"]["path"]),
        )["setup"]
        checked(setup[0])
        swap_started = True
        checked(setup[1])
        checked(("udevadm", "settle", "--timeout=5"), 10)

        candidate = initial["candidate_module"]
        bdf = initial["bdf"]
        device = evidence._device_node_for_bdf(bdf)
        runtime = {
            "loaded_srcversion": evidence._loaded_srcversion(),
            "loaded_build_id": evidence._loaded_build_id(),
            "device": str(device),
            "device_present": device.is_char_device(),
            "active_clients": evidence._active_npu_clients(device),
            "tdr_timeout_ms": Path(
                "/sys/module/amdxdna/parameters/tdr_timeout_ms"
            ).read_text().strip(),
            "force_cmdlist": Path(
                "/sys/module/amdxdna/parameters/force_cmdlist"
            ).read_text().strip(),
            "force_iova": Path(
                "/sys/module/amdxdna/parameters/force_iova"
            ).read_text().strip(),
        }
        _write_json(run_dir / "runtime-preflight.json", runtime, owner_uid)
        if runtime != {
            "loaded_srcversion": candidate["srcversion"],
            "loaded_build_id": candidate["build_id"],
            "device": "/dev/accel/accel0",
            "device_present": True,
            "active_clients": 0,
            "tdr_timeout_ms": "2000",
            "force_cmdlist": "Y",
            "force_iova": "N",
        }:
            raise RuntimeError("experimental NPI module runtime preflight differed")

        nodes = list(Path("/sys/kernel/debug/accel").glob("*/phoenix_npi_lock"))
        if len(nodes) != 1 or not nodes[0].is_file():
            raise RuntimeError(
                f"expected one Phoenix NPI debugfs node, found {len(nodes)}"
            )
        _write_kmsg(f"PHOENIX_NPI_READ_HOST_BEGIN {token}")
        marker_started = True
        try:
            read = checked(
                ("timeout", "-k", "2", "10", "cat", str(nodes[0])), 15,
            )
            value = read.stdout.strip()
            if re.fullmatch(r"0x[0-9a-f]{8}", value or "") is None:
                raise RuntimeError(f"invalid Phoenix NPI lock value {value!r}")
            _write_json(run_dir / "read.json", {
                "node": str(nodes[0]),
                "value": value,
                "operation": NPI_REQUEST,
                "system_read": NPI_SYSTEM_READ,
            }, owner_uid)
        finally:
            close_marker()
        if not lifecycle:
            raise RuntimeError("physical NPI mailbox lifecycle differed")

        owner = pwd.getpwuid(owner_uid)
        canary_home = stage_npi_canary_home(
            run_dir,
            Path(request["artifacts"]["xrt_coreutil"]["path"]),
            Path(request["artifacts"]["xrt_validation_archive"]["path"]),
        )
        canary = evidence._run_command(
            npi_canary_argv(
                Path(request["artifacts"]["xrt_smi"]["path"]),
                owner.pw_name,
                str(canary_home),
            ),
            130,
        )
        commands.append(_command_data(canary))
        _write_text(run_dir / "canary.stdout", canary.stdout, owner_uid)
        _write_text(run_dir / "canary.stderr", canary.stderr, owner_uid)
        if canary.returncode or canary.timed_out:
            raise RuntimeError(
                f"post-read xrt-smi canary failed: {canary.returncode}"
            )
        canary_ok = True
    except Exception as error:  # noqa: BLE001 - restoration belongs in finally
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        close_marker()
        if swap_started:
            try:
                if active_clients_before_restore(evidence, initial["bdf"]):
                    raise RuntimeError("active NPU client blocks safe module restoration")
                restore = module_transaction_commands(
                    Path(request["artifacts"]["module"]["path"]),
                )["restore"]
                if evidence._loaded_srcversion() is not None:
                    checked(restore[0])
                checked(restore[1])
                checked(("udevadm", "settle", "--timeout=5"), 10)
                restore_module_parameters(
                    Path("/sys/module/amdxdna/parameters"), initial["parameters"],
                )
                bdf, pci = evidence._physical_npu()
                if (pci / "power/control").read_text().strip() != initial[
                    "power_control"
                ]:
                    (pci / "power/control").write_text(
                        initial["power_control"] + "\n"
                    )
                device = evidence._device_node_for_bdf(bdf)
                original = initial["original_module"]
                restored = (
                    evidence._loaded_srcversion() == original["srcversion"]
                    and evidence._loaded_build_id() == original["build_id"]
                    and device == Path("/dev/accel/accel0")
                    and device.is_char_device()
                    and (pci / "power/control").read_text().strip()
                    == initial["power_control"]
                )
                if not restored:
                    raise RuntimeError("installed amdxdna identity did not restore")
            except Exception as error:  # noqa: BLE001 - report; never auto-reset
                errors.append(f"module restoration failed: {error}")
        else:
            restored = True

    state = (
        "complete"
        if value is not None and lifecycle and canary_ok and restored and not errors
        else "failed"
    )
    result = {
        "schema_version": 1,
        "operation": "phoenix_npi_read",
        "qualified": state == "complete",
        "value": value,
        "request": NPI_REQUEST,
        "system_read": NPI_SYSTEM_READ,
    }
    _write_json(run_dir / "result.json", result, owner_uid)
    _write_json(status_path, {
        "state": state,
        "restored": restored,
        "lifecycle_ok": lifecycle,
        "canary_ok": canary_ok,
        "initial": initial,
        "commands": commands,
        "errors": errors,
    }, owner_uid)
    return 0 if state == "complete" else 1


def _write_receipt(request: dict, status: dict, result: dict) -> None:
    run_dir = Path(request["run_dir"])
    arm = request["arm"]
    classification = result.get("classification", {})
    passed = host_run_pass(arm, status, result)
    initial = status.get("initial", {})
    candidate = initial.get("candidate_module", {})
    original = initial.get("original_module", {})
    output = result.get("output", {})
    canary = result.get("canary", {})
    conclusion = (
        "established" if arm == "treatment" and passed
        else "not established (control only)" if passed
        else "not established"
    )
    lines = [
        f"# Phoenix real column-gate physical {arm} receipt",
        "",
        (
            f"- Physical behavioral result: **{'PASS' if passed else 'STOP'}** "
            f"({classification.get('reason', 'no classification')})."
        ),
        f"- Physical freeze/resume conclusion: {conclusion}.",
        "- Live placement: exact 1:1; deliberate 2:1 mismatch required to stop before submission.",
        f"- Submission/canary lifecycle: {'pass' if status.get('lifecycle_ok') else 'fail'}.",
        f"- Output exact: {str(output.get('matches') is True).lower()}.",
        f"- Fresh-context canary exact: {str(canary.get('matches') is True).lower()}.",
        f"- Clock before: {json.dumps(result.get('clock_before'))}.",
        f"- Clock after: {json.dumps(result.get('clock_after'))}.",
        f"- Installed module restored: {str(status.get('restored') is True).lower()}.",
        f"- Experimental module SHA-256: {candidate.get('sha256', 'unavailable')}.",
        f"- Experimental module srcversion: {candidate.get('srcversion', 'unavailable')}.",
        f"- Experimental module build ID: {candidate.get('build_id', 'unavailable')}.",
        f"- Restored module SHA-256: {original.get('sha256', 'unavailable')}.",
        f"- Restored module srcversion: {original.get('srcversion', 'unavailable')}.",
        f"- Restored module build ID: {original.get('build_id', 'unavailable')}.",
        f"- Raw evidence: {run_dir}.",
        "- Full software pins: request.json; module transition: status.json; behavioral result: result.json.",
    ]
    if status.get("errors"):
        lines.append(f"- Stop reasons: {'; '.join(status['errors'])}.")
    _write_text(run_dir / "receipt.md", "\n".join(lines) + "\n")


def _write_npi_receipt(request: dict, status: dict, result: dict) -> None:
    run_dir = Path(request["run_dir"])
    passed = npi_host_run_pass(status, result)
    initial = status.get("initial", {})
    candidate = initial.get("candidate_module", {})
    original = initial.get("original_module", {})
    lines = [
        "# Phoenix read-only NPI management probe receipt",
        "",
        f"- Physical result: **{'PASS' if passed else 'STOP'}**.",
        f"- Observed NPI lock value: {result.get('value', 'unavailable')}.",
        f"- Exact mailbox lifecycle: {'pass' if status.get('lifecycle_ok') else 'fail'}.",
        f"- Post-read XRT canary: {'pass' if status.get('canary_ok') else 'fail'}.",
        f"- Installed module restored: {str(status.get('restored') is True).lower()}.",
        f"- Experimental module SHA-256: {candidate.get('sha256', 'unavailable')}.",
        f"- Experimental module srcversion: {candidate.get('srcversion', 'unavailable')}.",
        f"- Restored module SHA-256: {original.get('sha256', 'unavailable')}.",
        f"- Restored module srcversion: {original.get('srcversion', 'unavailable')}.",
        f"- KVM qualification: {request.get('kvm_run')}.",
        f"- Raw evidence: {run_dir}.",
        "- This research hook exposed and issued no NPI write or clock-transition operation.",
    ]
    if status.get("errors"):
        lines.append(f"- Stop reasons: {'; '.join(status['errors'])}.")
    _write_text(run_dir / "receipt.md", "\n".join(lines) + "\n")


def _npi_coordinator(kvm_run: Path, preflight_only: bool) -> int:
    if os.geteuid() == 0:
        raise RuntimeError("run the physical NPI coordinator as the desktop user")
    repository = Path(__file__).resolve().parent.parent
    host_root = repository / "build/experiments/phoenix-npi-read-host"
    marker = host_root / "read-qualified"
    if marker.exists() and not preflight_only:
        raise RuntimeError("a physical read-only NPI probe is already qualified")
    run_name = (
        "read-preflight"
        if preflight_only
        else f"read-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{os.getpid()}"
    )
    run_dir = host_root / run_name
    request = build_npi_host_request(
        kvm_run, repository, run_dir, os.environ, os.uname().release,
    )
    _assert_repository_inputs(request)
    snapshot = _host_snapshot(request, privileged=False)
    preflight = {
        "operation": request["operation"],
        "run_dir": str(run_dir.resolve()),
        "kvm_run": request["kvm_run"],
        "snapshot": snapshot,
        "pinned_hashes": {
            name: value["sha256"] for name, value in request["artifacts"].items()
        },
    }
    print(json.dumps(preflight, indent=2, sort_keys=True))
    if preflight_only:
        return 0

    run_dir.mkdir(parents=True)
    stage_npi_canary_home(
        run_dir,
        Path(request["artifacts"]["xrt_coreutil"]["path"]),
        Path(request["artifacts"]["xrt_validation_archive"]["path"]),
    )
    request_path = run_dir / "request.json"
    _write_json(request_path, request)
    request_sha256 = _sha256(request_path)
    _write_json(run_dir / "preflight.json", preflight)
    _write_json(run_dir / "status.json", {
        "state": "prepared",
        "restored": True,
        "lifecycle_ok": False,
        "canary_ok": False,
        "request_sha256": request_sha256,
    })
    command = (
        "pkexec", sys.executable, str(Path(__file__).resolve()),
        "_npi_privileged", str(request_path), request_sha256,
    )
    privileged = subprocess.run(command, check=False)
    status = json.loads((run_dir / "status.json").read_text())
    if privileged.returncode and status.get("state") == "prepared":
        status = {
            "state": "authorization_or_privileged_failure",
            "restored": True,
            "lifecycle_ok": False,
            "canary_ok": False,
            "errors": [
                f"pkexec returned {privileged.returncode} before transaction"
            ],
        }
        _write_json(run_dir / "status.json", status)
    result = (
        json.loads((run_dir / "result.json").read_text())
        if (run_dir / "result.json").is_file()
        else {}
    )
    _write_npi_receipt(request, status, result)
    passed = npi_host_run_pass(status, result)
    if passed:
        _write_text(marker, f"run={run_dir.resolve()}\n")
    print(
        f"phoenix physical read-only NPI probe: "
        f"{'PASS' if passed else 'STOP'}\nevidence: {run_dir}"
    )
    return 0 if passed else 1


def _coordinator(arm: str, pair: Path, preflight_only: bool) -> int:
    if os.geteuid() == 0:
        raise RuntimeError("run the physical coordinator as the desktop user")
    repository = Path(__file__).resolve().parent.parent
    host_root = pair.resolve(strict=True) / "host"
    if arm == "control" and not preflight_only and (
        host_root / "control-behavior-qualified"
    ).exists():
        raise RuntimeError("this pair already has a behaviorally qualified host control")
    run_name = (
        f"{arm}-preflight"
        if preflight_only
        else f"{arm}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{os.getpid()}"
    )
    run_dir = host_root / run_name
    request = build_host_request(
        arm, pair, repository, run_dir, os.environ, os.uname().release,
    )
    _assert_repository_inputs(request)
    snapshot = _host_snapshot(request, privileged=False)
    preflight = {
        "arm": arm,
        "pair": str(pair.resolve()),
        "run_dir": str(run_dir),
        "kvm_control_run": request["kvm_control_run"],
        "kvm_disposition": request["kvm_disposition"],
        "host_control_run": request["host_control_run"],
        "snapshot": snapshot,
        "pinned_hashes": {
            name: value["sha256"] for name, value in request["artifacts"].items()
        },
    }
    print(json.dumps(preflight, indent=2, sort_keys=True))
    if preflight_only:
        return 0

    run_dir.mkdir(parents=True)
    request_path = run_dir / "request.json"
    _write_json(request_path, request)
    request_sha256 = _sha256(request_path)
    _write_json(run_dir / "preflight.json", preflight)
    _write_json(run_dir / "status.json", {
        "state": "prepared",
        "restored": True,
        "request_sha256": request_sha256,
    })
    result = subprocess.run(
        (
            "pkexec", sys.executable, str(Path(__file__).resolve()),
            "_privileged", str(request_path), request_sha256,
        ),
        check=False,
    )
    status = json.loads((run_dir / "status.json").read_text())
    if result.returncode and status.get("state") == "prepared":
        status = {
            "state": "authorization_or_privileged_failure",
            "restored": True,
            "lifecycle_ok": False,
            "errors": [f"pkexec returned {result.returncode} before transaction"],
        }
        _write_json(run_dir / "status.json", status)
    result_data = (
        json.loads((run_dir / "result.json").read_text())
        if (run_dir / "result.json").is_file()
        else {}
    )
    _write_receipt(request, status, result_data)
    passed = host_run_pass(arm, status, result_data)
    if arm == "control" and passed:
        _write_text(
            host_root / "control-behavior-qualified", f"run={run_dir.resolve()}\n",
        )
    print(
        f"phoenix real column-gate physical {arm}: "
        f"{'PASS' if passed else 'STOP'}\nevidence: {run_dir}"
    )
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] in (["_privileged"], ["_worker"]):
        if len(argv) != 3:
            return 2
        function = _run_privileged if argv[0] == "_privileged" else _run_worker
        return function(Path(argv[1]), argv[2])
    if argv[:1] == ["_validate_npi_lifecycle"]:
        if len(argv) != 2:
            return 2
        try:
            log = Path(argv[1]).read_text()
        except OSError:
            return 2
        return 0 if npi_lifecycle_ok(log) else 1
    if argv[:1] == ["_npi_privileged"]:
        if len(argv) != 3:
            return 2
        return _run_npi_privileged(Path(argv[1]), argv[2])
    if argv[:1] == ["npi-read"]:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--preflight", action="store_true")
        parser.add_argument("kvm_run", type=Path)
        args = parser.parse_args(argv[1:])
        try:
            return _npi_coordinator(args.kvm_run, args.preflight)
        except Exception as error:  # noqa: BLE001 - CLI must leave a clear stop
            print(f"error: {error}", file=sys.stderr)
            return 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("arm", choices=("control", "treatment"))
    parser.add_argument("pair", type=Path)
    args = parser.parse_args(argv)
    try:
        return _coordinator(args.arm, args.pair, args.preflight)
    except Exception as error:  # noqa: BLE001 - CLI must leave a clear stop
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
