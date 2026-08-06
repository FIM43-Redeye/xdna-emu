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


def instrument_post_tct_noops(data: bytes, count: int) -> bytes:
    """Keep the firmware command open with finite management-only work."""
    return patcher.insert_noops_after_last_tct(data, count)


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


if __name__ == "__main__":
    try:
        campaign = run_campaign(_parse_args())
        print(json.dumps(campaign, indent=2))
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
