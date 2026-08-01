import json
import pwd
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest import mock

from tools import npu1_firmware_evidence as fw

PROTOCOL = fw.ProtocolOpcodes(
    create_context=0x2,
    destroy_context=0x3,
    execute_buffer_cf=0xC,
    config_cu=0x11,
    chain_exec_npu=0x18,
    map_host_buffer=0x106,
)
TREATMENT, CONTROL = fw.campaign_arms(PROTOCOL)


def successful_stdout() -> str:
    return "\n".join(
        [*(f"Correct output {value} == {value}" for value in range(2, 66)), "PASS!"]
    )


def response_event(
    channel: int, message_id: str, opcode: int, words: tuple[int, ...]
) -> str:
    data = b"".join(word.to_bytes(4, "little") for word in words)
    return (
        "worker-2 [002] ..... 1.000002: mbox_response: "
        f"xdna_mailbox.{channel} id {message_id} opcode 0x{opcode:x} "
        f"size {len(data)} data {data.hex(' ')}"
    )


def lifecycle(
    run_id: str, execute_opcode: int, protocol: fw.ProtocolOpcodes = PROTOCOL
) -> str:
    execute_id = "0x1d000004"
    execute_words = (0, 0, 0) if execute_opcode == protocol.chain_exec_npu else (0,)
    lines = [
        f"capture-1 [000] ..... 1.000000: tracing_mark_write: NPU1_FW_BEGIN {run_id}",
        f"worker-1 [000] ..... 1.000001: mbox_set_tail: xdna_mailbox.145 id 0x1d000001 opcode 0x{protocol.create_context:x}",
        response_event(145, "0x1d000001", protocol.create_context, (0,) * 19),
        f"worker-1 [000] ..... 1.000002: mbox_set_head: xdna_mailbox.145 id 0x1d000001 opcode 0x{protocol.create_context:x}",
        f"worker-1 [000] ..... 1.000003: mbox_set_tail: xdna_mailbox.145 id 0x1d000002 opcode 0x{protocol.map_host_buffer:x}",
        response_event(145, "0x1d000002", protocol.map_host_buffer, (0,)),
        f"worker-1 [000] ..... 1.000004: mbox_set_head: xdna_mailbox.145 id 0x1d000002 opcode 0x{protocol.map_host_buffer:x}",
        f"worker-1 [000] ..... 1.000005: mbox_set_tail: xdna_mailbox.136 id 0x1d000003 opcode 0x{protocol.config_cu:x}",
        "worker-1 [000] ..... 1.000006: xdna_job: fence=(context:7, seqno:1), ctx.42.1 seq#:0 job run, op=0",
        response_event(136, "0x1d000003", protocol.config_cu, (0,)),
        f"worker-1 [000] ..... 1.000007: mbox_set_head: xdna_mailbox.136 id 0x1d000003 opcode 0x{protocol.config_cu:x}",
        f"worker-1 [000] ..... 1.000008: mbox_set_tail: xdna_mailbox.136 id {execute_id} opcode 0x{execute_opcode:x}",
        "idle-0 [001] d.h1. 1.000009: mbox_irq_handle: xdna_mailbox.136",
        "worker-2 [002] ..... 1.000010: mbox_rx_worker: xdna_mailbox.136",
        "worker-2 [002] ..... 1.000011: xdna_job: fence=(context:7, seqno:1), ctx.42.1 seq#:0 signaling fence, op=0",
        response_event(136, execute_id, execute_opcode, execute_words),
        f"worker-2 [002] ..... 1.000012: mbox_set_head: xdna_mailbox.136 id {execute_id} opcode 0x{execute_opcode:x}",
        f"test.exe-42 [003] ..... 1.000013: mbox_set_tail: xdna_mailbox.145 id 0x1d000005 opcode 0x{protocol.destroy_context:x}",
        response_event(145, "0x1d000005", protocol.destroy_context, (0,)),
        f"worker-2 [002] ..... 1.000014: mbox_set_head: xdna_mailbox.145 id 0x1d000005 opcode 0x{protocol.destroy_context:x}",
        f"capture-1 [000] ..... 1.000015: tracing_mark_write: NPU1_FW_END {run_id}",
    ]
    return "\n".join(lines)


def kernel_log(
    run_id: str,
    execute_opcode: int,
    status: int = 0,
    *,
    protocol: fw.ProtocolOpcodes = PROTOCOL,
    include_async_responses: bool = False,
) -> str:
    messages = (
        ("1d000001", protocol.create_context, 28, 76),
        ("1d000002", protocol.map_host_buffer, 20, 4),
        ("1d000003", protocol.config_cu, 132, 4),
        (
            "1d000004",
            execute_opcode,
            24 if execute_opcode == protocol.chain_exec_npu else 80,
            12 if execute_opcode == protocol.chain_exec_npu else 4,
        ),
        ("1d000005", protocol.destroy_context, 4, 4),
    )
    lines = [f"<12>[ 1.000000] [ T42] NPU1_FW_BEGIN {run_id}"]
    for message_id, opcode, request_size, response_size in messages:
        lines.extend(
            [
                f"amdxdna: opcode 0x{opcode:x} size {request_size} id 0x{message_id}",
                f"req data: 00000000: {request_size:08x} 00010000 {message_id} {opcode:08x}",
                f"amdxdna: opcode 0x{opcode:x} size {response_size} id 0x{message_id}",
            ]
        )
        if opcode in (protocol.config_cu, execute_opcode):
            if include_async_responses:
                suffix = (
                    " deadbeef cafebabe" if opcode == protocol.chain_exec_npu else ""
                )
                response_status = status if opcode == execute_opcode else 0
                lines.append(
                    f"resp data: 00000000: {response_status:08x}{suffix}  ...."
                )
        elif opcode == protocol.create_context:
            lines.extend(
                [
                    "resp data: 00000000: 00000000 00000005 00010005 030da004  ................",
                    "resp data: 00000010: 030da000 030aa000 00000400 030db004  ................",
                ]
            )
        else:
            lines.append("resp data: 00000000: 00000000  ....")
    lines.append(f"<12>[ 1.000100] [ T42] NPU1_FW_END {run_id}")
    return "\n".join(lines)


def capture(entry: fw.ScheduleEntry) -> fw.RawCaptureIndex:
    return fw.RawCaptureIndex(
        run_id=entry.run_id,
        lifecycle_text=lifecycle(entry.run_id, entry.arm.execute_opcode),
        tdr_before=0,
        tdr_after=0,
        iommu_faults_before=0,
        iommu_faults_after=0,
        teardown_ok=True,
        restoration_ok=True,
    )


def command(stdout: str | None = None, *, returncode: int = 0, timed_out: bool = False):
    return fw.CommandResult(
        argv=("./test.exe", "-x", "aie.xclbin", "-k", "MLIR_AIE", "-i", "insts.bin"),
        returncode=returncode,
        stdout=successful_stdout() if stdout is None else stdout,
        stderr="",
        timed_out=timed_out,
    )


class CampaignModelTests(unittest.TestCase):
    def test_protocol_opcodes_are_derived_from_named_driver_enums(self):
        header = """
        enum aie2_msg_opcode {
            MSG_OP_CREATE_CONTEXT = 0x2,
            MSG_OP_DESTROY_CONTEXT = 0x3,
            MSG_OP_EXECUTE_BUFFER_CF = 0xC,
            MSG_OP_EXEC_DPU = 0x10,
            MSG_OP_CONFIG_CU = 0x11,
            MSG_OP_CHAIN_EXEC_NPU = 0x18,
            MSG_OP_MAP_HOST_BUFFER = 0x106,
        };
        """

        protocol = fw.parse_protocol_opcodes(header)
        treatment, control = fw.campaign_arms(protocol)

        self.assertEqual(
            protocol,
            fw.ProtocolOpcodes(
                create_context=0x2,
                destroy_context=0x3,
                execute_buffer_cf=0xC,
                config_cu=0x11,
                chain_exec_npu=0x18,
                map_host_buffer=0x106,
            ),
        )
        self.assertEqual(treatment.execute_opcode, 0x18)
        self.assertEqual(control.execute_opcode, 0xC)

    def test_protocol_opcode_derivation_rejects_ambiguous_sources(self):
        header = "\n".join(
            f"{name} = {index},"
            for index, name in enumerate(
                [
                    "MSG_OP_CREATE_CONTEXT",
                    "MSG_OP_DESTROY_CONTEXT",
                    "MSG_OP_EXECUTE_BUFFER_CF",
                    "MSG_OP_CONFIG_CU",
                    "MSG_OP_CHAIN_EXEC_NPU",
                    "MSG_OP_MAP_HOST_BUFFER",
                ],
                start=1,
            )
        )

        with self.assertRaises(ValueError):
            fw.parse_protocol_opcodes(header.replace("MSG_OP_CONFIG_CU", "MISSING"))
        with self.assertRaises(ValueError):
            fw.parse_protocol_opcodes(header + "\nMSG_OP_CONFIG_CU = 99,\n")

    def test_schedules_are_seeded_deterministic_and_balanced(self):
        self.assertEqual(
            fw.vertical_schedule(17, PROTOCOL), fw.vertical_schedule(17, PROTOCOL)
        )
        self.assertNotEqual(
            fw.vertical_schedule(17, PROTOCOL), fw.vertical_schedule(18, PROTOCOL)
        )
        schedule = fw.repetition_schedule(50, 50, 1234, PROTOCOL)
        self.assertEqual(sum(entry.arm == TREATMENT for entry in schedule), 50)
        self.assertEqual(sum(entry.arm == CONTROL for entry in schedule), 50)
        self.assertEqual(schedule, fw.repetition_schedule(50, 50, 1234, PROTOCOL))

    def test_output_oracle_requires_exact_values_and_pass_marker(self):
        result = fw.parse_output(successful_stdout())
        self.assertTrue(result.ok)
        self.assertEqual(result.values, tuple(range(2, 66)))
        for text in [
            "PASS!",
            successful_stdout().replace(
                "Correct output 12 == 12", "Correct output 99 == 12"
            ),
            successful_stdout().replace("Correct output 12 == 12\n", ""),
            successful_stdout().replace(
                "Correct output 12 == 12\nCorrect output 13 == 13",
                "Correct output 13 == 13\nCorrect output 12 == 12",
            ),
        ]:
            self.assertFalse(fw.parse_output(text).ok)

    def test_lifecycle_requires_ordered_unique_mode_specific_opcodes(self):
        for arm in (TREATMENT, CONTROL):
            entry = fw.ScheduleEntry(0, 0, arm)
            parsed = fw.parse_lifecycle(
                lifecycle(entry.run_id, arm.execute_opcode), entry
            )
            self.assertTrue(parsed.ok, parsed.reason)
            self.assertEqual(parsed.execute_opcode, arm.execute_opcode)

        entry = fw.ScheduleEntry(0, 0, TREATMENT)
        valid = lifecycle(entry.run_id, TREATMENT.execute_opcode)
        irq = "idle-0 [001] d.h1. 1.000009: mbox_irq_handle: xdna_mailbox.136"
        worker = "worker-2 [002] ..... 1.000010: mbox_rx_worker: xdna_mailbox.136"
        mutations = [
            valid.replace(
                "worker-1 [000] ..... 1.000008: mbox_set_tail: xdna_mailbox.136 id 0x1d000004 opcode 0x18\n",
                "",
            ),
            valid.replace("opcode 0x18", "opcode 0x10"),
            valid.replace(
                "worker-1 [000] ..... 1.000008: mbox_set_tail: xdna_mailbox.136 id 0x1d000004 opcode 0x18",
                "worker-1 [000] ..... 1.000008: mbox_set_tail: xdna_mailbox.136 id 0x1d000004 opcode 0x18\n"
                "worker-1 [000] ..... 1.000008: mbox_set_tail: xdna_mailbox.136 id 0x1d000004 opcode 0x18",
            ),
            valid.replace(f"{irq}\n{worker}", f"{worker}\n{irq}"),
            valid.replace(
                "mbox_set_head: xdna_mailbox.136 id 0x1d000004 opcode 0x18",
                "mbox_set_head: xdna_mailbox.136 id 0x1d000099 opcode 0x18",
            ),
        ]
        for text in mutations:
            self.assertFalse(fw.parse_lifecycle(text, entry).ok)

    def test_lifecycle_requires_one_response_body_per_request(self):
        entry = fw.ScheduleEntry(0, 0, TREATMENT)
        response = response_event(136, "0x1d000003", PROTOCOL.config_cu, (0,))
        text = lifecycle(entry.run_id, entry.arm.execute_opcode).replace(
            response + "\n", ""
        )

        parsed = fw.parse_lifecycle(text, entry)

        self.assertFalse(parsed.ok)
        self.assertEqual(
            parsed.reason, "mailbox response body is missing or duplicated"
        )

    def test_lifecycle_rejects_uncorrelated_response_body(self):
        entry = fw.ScheduleEntry(0, 0, TREATMENT)
        text = lifecycle(entry.run_id, entry.arm.execute_opcode).replace(
            f"capture-1 [000] ..... 1.000015: tracing_mark_write: NPU1_FW_END {entry.run_id}",
            response_event(136, "0x1d000099", PROTOCOL.config_cu, (0,))
            + "\n"
            + f"capture-1 [000] ..... 1.000015: tracing_mark_write: NPU1_FW_END {entry.run_id}",
        )

        parsed = fw.parse_lifecycle(text, entry)

        self.assertFalse(parsed.ok)
        self.assertEqual(parsed.reason, "unexpected mailbox response body is present")

    def test_lifecycle_rejects_response_size_mismatch(self):
        entry = fw.ScheduleEntry(0, 0, TREATMENT)
        response = response_event(136, "0x1d000003", PROTOCOL.config_cu, (0,))
        text = lifecycle(entry.run_id, entry.arm.execute_opcode).replace(
            response, response.replace("size 4", "size 8")
        )

        parsed = fw.parse_lifecycle(text, entry)

        self.assertFalse(parsed.ok)
        self.assertEqual(
            parsed.reason, "mailbox response body size does not match payload"
        )

    def test_lifecycle_preserves_config_and_execute_response_words(self):
        for arm, execute_words in ((TREATMENT, (0, 0, 0)), (CONTROL, (0,))):
            entry = fw.ScheduleEntry(0, 0, arm)

            parsed = fw.parse_lifecycle(
                lifecycle(entry.run_id, entry.arm.execute_opcode), entry
            )

            self.assertEqual(getattr(parsed, "config_cu_response", None), (0,))
            self.assertEqual(getattr(parsed, "execute_response", None), execute_words)

    def test_lifecycle_requires_firmware_derived_success_response_words(self):
        cases = (
            (TREATMENT, PROTOCOL.config_cu, "0x1d000003", (0,), (1,)),
            (TREATMENT, PROTOCOL.chain_exec_npu, "0x1d000004", (0, 0, 0), (1, 0, 0)),
            (TREATMENT, PROTOCOL.chain_exec_npu, "0x1d000004", (0, 0, 0), (0, 1, 0)),
            (TREATMENT, PROTOCOL.chain_exec_npu, "0x1d000004", (0, 0, 0), (0, 0, 1)),
            (CONTROL, PROTOCOL.execute_buffer_cf, "0x1d000004", (0,), (1,)),
        )
        for arm, opcode, message_id, good_words, bad_words in cases:
            with self.subTest(arm=arm.name, opcode=opcode, bad_words=bad_words):
                entry = fw.ScheduleEntry(0, 0, arm)
                channel = 136
                text = lifecycle(entry.run_id, entry.arm.execute_opcode).replace(
                    response_event(channel, message_id, opcode, good_words),
                    response_event(channel, message_id, opcode, bad_words),
                )

                parsed = fw.parse_lifecycle(text, entry)

                self.assertFalse(parsed.ok)
                self.assertEqual(
                    parsed.reason,
                    "successful firmware response words do not match the derived contract",
                )

    def test_lifecycle_uses_protocol_derived_common_opcodes(self):
        protocol = fw.ProtocolOpcodes(0x22, 0x23, 0x2C, 0x31, 0x38, 0x206)
        _, control = fw.campaign_arms(protocol)
        entry = fw.ScheduleEntry(0, 0, control)

        parsed = fw.parse_lifecycle(
            lifecycle(entry.run_id, entry.arm.execute_opcode, protocol), entry
        )

        self.assertTrue(parsed.ok, parsed.reason)

    def test_kernel_evidence_requires_request_bytes_responses_and_zero_status(self):
        for arm in (TREATMENT, CONTROL):
            entry = fw.ScheduleEntry(0, 0, arm)
            parsed = fw.parse_kernel_evidence(
                kernel_log(entry.run_id, arm.execute_opcode), entry
            )
            self.assertTrue(parsed.ok, parsed.reason)
            self.assertIsNone(parsed.execute_status)

        entry = fw.ScheduleEntry(0, 0, TREATMENT)
        valid = kernel_log(entry.run_id, entry.arm.execute_opcode)
        for text in (
            valid.replace("req data: 00000000:", "missing request:", 1),
            valid.replace("resp data: 00000000:", "missing response:", 1),
            kernel_log(
                entry.run_id,
                entry.arm.execute_opcode,
                status=3,
                include_async_responses=True,
            ),
        ):
            self.assertFalse(fw.parse_kernel_evidence(text, entry).ok)

    def test_run_classification_detects_faults_process_failures_and_cleanup(self):
        entry = fw.ScheduleEntry(0, 0, TREATMENT)
        treatment = fw.classify_run(entry, command(), capture(entry))
        self.assertEqual(treatment.outcome, fw.Outcome.SUCCESS)
        self.assertEqual(treatment.execute_status, 0)
        self.assertEqual(treatment.execute_status_source, "host_ert_completed")
        self.assertEqual(treatment.unknown_success_words, ())
        self.assertEqual(getattr(treatment, "config_cu_response", None), (0,))
        self.assertEqual(getattr(treatment, "execute_response", None), (0, 0, 0))
        control_entry = fw.ScheduleEntry(0, 0, CONTROL)
        control = fw.classify_run(control_entry, command(), capture(control_entry))
        self.assertEqual(control.outcome, fw.Outcome.SUCCESS)
        self.assertEqual(control.execute_status, 0)
        self.assertEqual(control.execute_status_source, "host_ert_completed")
        self.assertEqual(control.unknown_success_words, ())
        self.assertEqual(getattr(control, "config_cu_response", None), (0,))
        self.assertEqual(getattr(control, "execute_response", None), (0,))
        timeout = fw.classify_run(entry, command(timed_out=True), capture(entry))
        self.assertEqual(timeout.outcome, fw.Outcome.INFRASTRUCTURE_FAILURE)
        nonzero = fw.classify_run(entry, command(returncode=1), capture(entry))
        self.assertEqual(nonzero.outcome, fw.Outcome.INFRASTRUCTURE_FAILURE)

        fault = capture(entry)
        fault = fw.RawCaptureIndex(**{**fault.__dict__, "tdr_after": 1})
        self.assertEqual(
            fw.classify_run(entry, command(), fault).outcome,
            fw.Outcome.DEVICE_FAULT_OR_WEDGE,
        )
        bad_cleanup = capture(entry)
        bad_cleanup = fw.RawCaptureIndex(
            **{**bad_cleanup.__dict__, "restoration_ok": False}
        )
        self.assertEqual(
            fw.classify_run(entry, command(), bad_cleanup).outcome,
            fw.Outcome.INFRASTRUCTURE_FAILURE,
        )

    def test_classification_json_preserves_observed_response_words(self):
        entry = fw.ScheduleEntry(0, 0, TREATMENT)
        result = fw.classify_run(entry, command(), capture(entry))

        data = fw._classification_json(result)

        self.assertEqual(data.get("config_cu_response_words"), [0])
        self.assertEqual(data.get("execute_response_words"), [0, 0, 0])

    def test_campaign_stops_at_first_failure_without_retry(self):
        schedule = fw.repetition_schedule(3, 3, 7, PROTOCOL)
        results = [
            fw.classify_run(schedule[0], command(), capture(schedule[0])),
            fw.classify_run(schedule[1], command(stdout="PASS!"), capture(schedule[1])),
            fw.classify_run(schedule[2], command(), capture(schedule[2])),
        ]
        campaign = fw.classify_campaign(schedule, results)
        self.assertEqual(campaign.outcome, fw.Outcome.SEMANTIC_MISMATCH)
        self.assertEqual(
            campaign.completed_run_ids, (schedule[0].run_id, schedule[1].run_id)
        )
        self.assertEqual(campaign.failed_run_id, schedule[1].run_id)

    def test_emission_plan_is_byte_stable_and_reuses_fixture_requirements(self):
        schedule = fw.vertical_schedule(17, PROTOCOL)
        inputs = tuple(
            fw.FixtureInput(
                input_id=f"input.{index}",
                semantic_kind=f"fixture.kind.{index}",
                logical_name=f"fixture-{index}.bin",
                sha256=f"{index + 1:064x}",
                fixture_bundle_id=f"bundle.sha256.{index + 10:064x}",
                artifact_path=f"raw/fixture-{index}.bin",
                source_path=f"/reserve/fixture-{index}",
            )
            for index in range(6)
        )
        first = fw.render_observation_plan("campaign.synthetic", schedule, inputs)
        second = fw.render_observation_plan(
            "campaign.synthetic", schedule, tuple(reversed(inputs))
        )
        self.assertEqual(first, second)
        parsed = json.loads(first)
        self.assertEqual(len(parsed["dependencies"]), 6)
        self.assertEqual(len(parsed["body"]["input_references"]), 6)


class SafeTransactionTests(unittest.TestCase):
    def test_preflight_rejects_non_executable_host_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {
                name: root / filename
                for name, filename in {
                    "firmware": "firmware.bin",
                    "xclbin": "aie.xclbin",
                    "instructions": "insts.bin",
                    "executable": "test.exe",
                }.items()
            }
            for name, path in files.items():
                path.write_bytes(name.encode())
            spec = fw.VerticalSpec(
                pci_id="1022:1502",
                firmware_logical_path="firmware.bin",
                firmware_sha256=sha256(b"firmware").hexdigest(),
                driver_protocol_revision="revision.synthetic",
                xclbin_sha256=sha256(b"xclbin").hexdigest(),
                instructions_sha256=sha256(b"instructions").hexdigest(),
                executable_sha256=sha256(b"executable").hexdigest(),
                tdr_timeout_ms=2000,
            )
            candidate = root / "candidate.ko"
            original = root / "original.ko"
            candidate.write_bytes(b"candidate")
            original.write_bytes(b"original")
            module = fw.QualifiedModuleManifest(
                candidate,
                sha256(b"candidate").hexdigest(),
                original,
                sha256(b"original").hexdigest(),
                source_revision=spec.driver_protocol_revision,
            )
            snapshot = fw.PreflightSnapshot(
                file_hashes={
                    name: fw.sha256_file(path) for name, path in files.items()
                },
                environment={},
                pci_id=spec.pci_id,
                active_clients=0,
                tdr_parameter_present=True,
            )

            self.assertEqual(
                fw.capture_preflight_errors(None, module, files, snapshot, spec),
                ("executable file is not executable",),
            )

    def test_frozen_workload_paths_use_canonical_fixture_bytes(self):
        paths = fw.frozen_input_paths()
        reserve = fw._workspace_root() / "npu1-research-reserve"

        for name in ("xclbin", "instructions", "executable"):
            self.assertTrue(paths[name].is_relative_to(reserve))
            self.assertEqual(
                fw.sha256_file(paths[name]), fw.VERTICAL_SPEC.file_hashes[name]
            )

    def test_cli_dispatches_coordinator_and_privileged_modes_without_shell(self):
        with mock.patch.object(
            fw, "run_coordinator", create=True, return_value=7
        ) as coordinator:
            self.assertEqual(
                fw.main(
                    [
                        "vertical",
                        "--campaign-id",
                        "campaign.test",
                        "--seed",
                        "17",
                        "--location-plan",
                        "/reserve/locations.json",
                        "--module-manifest",
                        "/reserve/module.json",
                    ]
                ),
                7,
            )
            coordinator.assert_called_once()
        with mock.patch.object(
            fw, "run_privileged_capture", create=True, return_value=9
        ) as privileged:
            self.assertEqual(
                fw.main(["_privileged", "/capture/request.json", "a" * 64]),
                9,
            )
            privileged.assert_called_once_with(Path("/capture/request.json"), "a" * 64)

    def test_raw_capture_derivation_preserves_kernel_and_fault_evidence(self):
        entry = fw.ScheduleEntry(0, 0, TREATMENT)
        before = (
            kernel_log(entry.run_id, entry.arm.execute_opcode) + "\nold kernel state\n"
        )
        current = kernel_log(entry.run_id, entry.arm.execute_opcode)
        derived = fw.derive_capture_index(
            entry,
            lifecycle(entry.run_id, entry.arm.execute_opcode),
            before,
            before + current,
        )
        self.assertTrue(derived.kernel_evidence_ok)
        self.assertIsNone(derived.execute_status)
        self.assertEqual(
            fw.classify_run(entry, command(), derived).outcome,
            fw.Outcome.SUCCESS,
        )

        faulted = fw.derive_capture_index(
            entry,
            lifecycle(entry.run_id, entry.arm.execute_opcode),
            before,
            before + current + "\namd_iommu: IO_PAGE_FAULT device=00:00.0\n",
        )
        self.assertEqual(
            fw.classify_run(entry, command(), faulted).outcome,
            fw.Outcome.DEVICE_FAULT_OR_WEDGE,
        )
        incomplete = fw.derive_capture_index(
            entry,
            lifecycle(entry.run_id, entry.arm.execute_opcode),
            before,
            before
            + current.replace("resp data: 00000000: 00000000", "missing response", 1),
        )
        self.assertEqual(
            fw.classify_run(entry, command(), incomplete).outcome,
            fw.Outcome.PROVENANCE_FAILURE,
        )

        regressed = fw.derive_capture_index(
            entry,
            lifecycle(entry.run_id, entry.arm.execute_opcode),
            "before state\n",
            current,
        )
        self.assertFalse(regressed.kernel_evidence_ok)
        self.assertEqual(
            regressed.kernel_evidence_reason,
            "kernel log snapshots are not append-only",
        )

    def test_dynamic_debug_restore_changes_only_print_flags_enabled_by_capture(self):
        selectors = (
            "file aie2_ctx.c line 300 +p",
            "file aie2_ctx.c line 356 +p",
        )
        control = "\n".join(
            [
                'drivers/accel/amdxdna/aie2_ctx.c:300 [amdxdna]handler =_ "Resp status 0x%x\\n"',
                'drivers/accel/amdxdna/aie2_ctx.c:356 [amdxdna]handler =p "Status 0x%x\\n"',
            ]
        )
        states = fw.dynamic_debug_print_states(control, selectors)
        self.assertEqual(states, {selectors[0]: False, selectors[1]: True})
        self.assertEqual(
            fw.dynamic_debug_restore_commands(states),
            ("file aie2_ctx.c line 300 -p",),
        )
        with self.assertRaises(ValueError):
            fw.dynamic_debug_print_states(control.replace(":356", ":999"), selectors)

    def test_module_manifest_requires_complete_debug_surface(self):
        required_selectors = [
            "file aie2_message.c line 1077 +p",
            "file amdxdna_mailbox.c line 192 +p",
            "file amdxdna_mailbox.c line 236 +p",
            "file amdxdna_mailbox.c line 271 +p",
            "file amdxdna_mailbox.c line 464 +p",
            "file amdxdna_mailbox_helper.c line 49 +p",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qualified-module.json"
            manifest = {
                "candidate_path": "/qualified/amdxdna.ko",
                "candidate_sha256": "a" * 64,
                "original_path": "/system/amdxdna.ko",
                "original_sha256": "b" * 64,
                "source_repository": "https://example.invalid/xdna-driver.git",
                "source_revision": fw.VERTICAL_SPEC.driver_protocol_revision,
                "build_recipe_sha256": "c" * 64,
                "kernel_release": "7.1.5-custom+",
                "tdr_parameter_present": True,
                "trace_events": [
                    "xdna_job",
                    "mbox_set_tail",
                    "mbox_set_head",
                    "mbox_response",
                    "mbox_irq_handle",
                    "mbox_rx_worker",
                    "mbox_poll_handle",
                    "uc_irq_handle",
                    "uc_wakeup",
                ],
                "dynamic_debug_selectors": required_selectors,
            }
            path.write_text(json.dumps(manifest))
            try:
                loaded = fw.load_qualified_module_manifest(path)
            except ValueError as error:
                self.fail(str(error))
            self.assertEqual(loaded.dynamic_debug_selectors, tuple(required_selectors))
            manifest["trace_events"].remove("mbox_response")
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                fw.load_qualified_module_manifest(path)
            manifest["trace_events"].append("mbox_response")
            manifest["dynamic_debug_selectors"].remove(
                "file amdxdna_mailbox_helper.c line 49 +p"
            )
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                fw.load_qualified_module_manifest(path)

    def test_location_plan_accepts_one_or_more_declared_roots(self):
        valid = {
            "roots": [
                {
                    "alias": f"root-{index}",
                    "path": f"/reserve/{index}",
                    "failure_domain_id": f"operator-domain-{index}",
                    "bundles": [],
                }
                for index in range(1)
            ]
        }
        self.assertEqual(fw.validate_location_plan(valid), ())
        self.assertTrue(fw.validate_location_plan({"roots": []}))
        duplicate = json.loads(json.dumps(valid))
        duplicate["roots"].append(
            {
                **duplicate["roots"][0],
                "alias": "root-1",
                "path": "/reserve/1",
            }
        )
        self.assertTrue(fw.validate_location_plan(duplicate))

    def test_capture_request_round_trips_and_rejects_unknown_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture-request.json"
            request = fw.CaptureRequest.synthetic(
                Path(directory), fw.vertical_schedule(17, PROTOCOL)
            )
            fw._write_capture_request(path, request)
            loaded = fw.load_capture_request(path)
            self.assertEqual(loaded, request)
            self.assertEqual(loaded.firmware, request.firmware)
            self.assertEqual(loaded.seed, request.seed)
            payload = json.loads(path.read_text())
            payload["schedule"][0]["arm"] = "unknown"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                fw.load_capture_request(path)

    def test_live_module_preflight_rejects_loaded_identity_and_runtime_drift(self):
        module = fw.QualifiedModuleManifest(
            Path("/qualified/amdxdna.ko"),
            "a" * 64,
            Path("/system/amdxdna.ko"),
            "b" * 64,
            source_revision=fw.VERTICAL_SPEC.driver_protocol_revision,
            kernel_release="7.1.5-custom+",
        )
        snapshot = fw.PreflightSnapshot(
            file_hashes=fw.VERTICAL_SPEC.file_hashes,
            environment={},
            pci_id=fw.VERTICAL_SPEC.pci_id,
            active_clients=0,
            tdr_parameter_present=True,
            kernel_release=module.kernel_release,
            loaded_module_path=str(module.original_path),
            loaded_module_sha256=module.original_sha256,
            loaded_module_srcversion="ORIGINAL",
            original_module_srcversion="ORIGINAL",
            xrt_coreutil_path="/opt/xilinx/xrt/lib/libxrt_coreutil.so.2",
            device_node_present=True,
            power_control="on",
        )
        self.assertEqual(fw.module_preflight_errors(module, snapshot), ())
        auto = fw.PreflightSnapshot(**{**snapshot.__dict__, "power_control": "auto"})
        self.assertEqual(fw.module_preflight_errors(module, auto), ())
        drifted = fw.PreflightSnapshot(
            **{
                **snapshot.__dict__,
                "loaded_module_sha256": "0" * 64,
                "xrt_coreutil_path": "",
                "power_control": "invalid",
            }
        )
        errors = fw.module_preflight_errors(module, drifted)
        self.assertTrue(any("loaded module" in error for error in errors))
        self.assertTrue(any("XRT" in error for error in errors))
        self.assertTrue(any("power/control" in error for error in errors))

    def test_preflight_rejects_pin_drift_emulator_environment_and_missing_tdr(self):
        spec = fw.VERTICAL_SPEC
        snapshot = fw.PreflightSnapshot(
            file_hashes={
                "firmware": spec.firmware_sha256,
                "xclbin": spec.xclbin_sha256,
                "instructions": spec.instructions_sha256,
                "executable": "0" * 64,
            },
            environment={"XDNA_EMU": "1"},
            pci_id=spec.pci_id,
            active_clients=0,
            tdr_parameter_present=False,
        )
        errors = fw.preflight_errors(spec, snapshot)
        self.assertTrue(any("executable" in error for error in errors))
        self.assertTrue(any("XDNA_EMU" in error for error in errors))
        self.assertTrue(any("tdr_timeout_ms" in error for error in errors))

    def test_privileged_request_is_owned_and_confined(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = fw.CaptureRequest.synthetic(
                root, fw.vertical_schedule(17, PROTOCOL)
            )
            self.assertEqual(fw.validate_privileged_request(request, "1000", 1000), ())
            self.assertTrue(fw.validate_privileged_request(request, None, 1000))
            self.assertTrue(fw.validate_privileged_request(request, "1001", 1000))
            escaped = fw.CaptureRequest(
                **{**request.__dict__, "status_path": root.parent / "escaped.json"}
            )
            self.assertTrue(fw.validate_privileged_request(escaped, "1000", 1000))

    def test_transaction_plan_is_serialized_and_never_contains_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            request = fw.CaptureRequest.synthetic(
                Path(directory), fw.vertical_schedule(17, PROTOCOL)
            )
            plan = fw.build_transaction_plan(request, 1000, submitted=False)
            self.assertEqual(len(plan.runs), 2)
            self.assertTrue(
                all(
                    run.argv[:3] == ("runuser", "-u", pwd.getpwuid(1000).pw_name)
                    for run in plan.runs
                )
            )
            self.assertIn("timeout", plan.runs[0].argv)
            self.assertEqual(plan.setup[0].argv, ("rmmod", "amdxdna"))
            self.assertEqual(
                tuple(command.argv for command in plan.rollback),
                (("rmmod", "amdxdna"), ("modprobe", "amdxdna")),
            )
            self.assertEqual(plan.trace_actions[0], "pin NPU PCI power/control on")
            post_traffic = fw.build_transaction_plan(request, 1000, submitted=True)
            self.assertEqual(post_traffic.rollback, ())
            flattened = " ".join(
                part
                for command_spec in (
                    *plan.setup,
                    *plan.runs,
                    *plan.cleanup,
                    *plan.rollback,
                )
                for part in command_spec.argv
            )
            for forbidden in ("xrt-smi", "suspend", "reboot", "reset", "pm-cycle"):
                self.assertNotIn(forbidden, flattened)
            self.assertNotIn("modprobe -r", flattened)

    def test_terminal_status_and_transient_service_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = root / "status.json"
            fw.write_terminal_status(status, {"state": "complete", "runs": 2})
            self.assertEqual(
                json.loads(status.read_text()), {"runs": 2, "state": "complete"}
            )
            command_result = fw._command_result_json(
                fw.CommandResult(("insmod", "candidate.ko"), 1, "", "missing dep")
            )
            self.assertEqual(command_result["stdout"], "")
            self.assertEqual(command_result["stderr"], "missing dep")
            power_control = root / "power-control"
            power_control.write_text("auto\n")
            self.assertEqual(fw._set_power_control(power_control, "on"), ("auto", "on"))
            self.assertEqual(
                fw._set_power_control(power_control, "auto"), ("on", "auto")
            )
            argv = fw.transient_service_argv(
                Path("/repo/tools/npu1_firmware_evidence.py"),
                "campaign.test",
                17,
                Path("/reserve/locations.json"),
                Path("/reserve/module.json"),
            )
            self.assertEqual(argv[:3], ("systemd-run", "--user", "--collect"))
            self.assertIn("campaign.test", argv)

    def test_coordinator_preparation_pins_inputs_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            workload = repository / "workload"
            workload.mkdir()
            files = {
                "firmware": workload / "firmware.bin",
                "xclbin": workload / "aie.xclbin",
                "instructions": workload / "insts.bin",
                "executable": workload / "test.exe",
            }
            for name, path in files.items():
                path.write_bytes(name.encode())
            files["executable"].chmod(0o700)
            spec = fw.VerticalSpec(
                pci_id="1022:1502",
                firmware_logical_path="firmware.bin",
                firmware_sha256=sha256(b"firmware").hexdigest(),
                driver_protocol_revision="revision.synthetic",
                xclbin_sha256=sha256(b"xclbin").hexdigest(),
                instructions_sha256=sha256(b"instructions").hexdigest(),
                executable_sha256=sha256(b"executable").hexdigest(),
                tdr_timeout_ms=2000,
            )
            candidate = repository / "candidate.ko"
            original = repository / "original.ko"
            candidate.write_bytes(b"candidate")
            original.write_bytes(b"original")
            module = fw.QualifiedModuleManifest(
                candidate,
                sha256(b"candidate").hexdigest(),
                original,
                sha256(b"original").hexdigest(),
            )
            snapshot = fw.PreflightSnapshot(
                file_hashes={
                    name: fw.sha256_file(path) for name, path in files.items()
                },
                environment={},
                pci_id=spec.pci_id,
                active_clients=0,
                tdr_parameter_present=True,
            )

            prepared = fw.prepare_capture(
                repository,
                "campaign.test",
                17,
                False,
                None,
                module,
                files,
                snapshot,
                protocol=PROTOCOL,
                spec=spec,
            )
            self.assertTrue(prepared.request_path.is_file())
            self.assertIsNone(prepared.request.location_plan)
            self.assertEqual(prepared.pkexec_argv[0], "pkexec")
            self.assertEqual(
                prepared.pkexec_argv[-1],
                sha256(prepared.request_path.read_bytes()).hexdigest(),
            )
            self.assertTrue(
                prepared.request.campaign_dir.is_relative_to(
                    repository / "build/experiments/npu1-firmware-evidence"
                )
            )
            with self.assertRaises(FileExistsError):
                fw.prepare_capture(
                    repository,
                    "campaign.test",
                    17,
                    False,
                    None,
                    module,
                    files,
                    snapshot,
                    protocol=PROTOCOL,
                    spec=spec,
                )

    def test_transaction_execution_stops_at_first_failure_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            schedule = fw.repetition_schedule(3, 3, 9, PROTOCOL)
            request = fw.CaptureRequest.synthetic(Path(directory), schedule)
            run_index = 0
            commands: list[fw.CommandSpec] = []
            trace_actions: list[str] = []

            def runner(spec: fw.CommandSpec) -> fw.CommandResult:
                nonlocal run_index
                commands.append(spec)
                if spec.argv and spec.argv[0] == "runuser":
                    result = command(
                        stdout=successful_stdout() if run_index == 0 else "PASS!"
                    )
                    run_index += 1
                    return result
                return fw.CommandResult(spec.argv, 0, "", "")

            def read_capture(entry: fw.ScheduleEntry) -> fw.RawCaptureIndex:
                return capture(entry)

            def set_force_cmdlist(value: str) -> str:
                return value

            execution = fw.execute_capture_transaction(
                request,
                1000,
                runner,
                read_capture,
                set_force_cmdlist,
                lambda action: trace_actions.append(action) or True,
            )
            self.assertEqual(execution.campaign.outcome, fw.Outcome.SEMANTIC_MISMATCH)
            self.assertEqual(run_index, 2)
            self.assertFalse(execution.rollback_attempted)
            self.assertTrue(execution.cleanup_ok)
            self.assertTrue(trace_actions)
            self.assertEqual(sum(spec.argv[0] == "runuser" for spec in commands), 2)


if __name__ == "__main__":
    unittest.main()
