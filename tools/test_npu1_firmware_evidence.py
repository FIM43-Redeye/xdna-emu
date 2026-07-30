import json
import os
import tempfile
import unittest
from pathlib import Path

from tools import npu1_firmware_evidence as fw


def successful_stdout() -> str:
    return "\n".join(
        [*(f"Correct output {value} == {value}" for value in range(2, 66)), "PASS!"]
    )


def lifecycle(run_id: str, execute_opcode: int) -> str:
    lines = [
        f"NPU1_FW_BEGIN {run_id}",
        "FW_REQUEST opcode=0x02",
        "FW_REQUEST opcode=0x106",
        "FW_REQUEST opcode=0x11",
        f"FW_REQUEST opcode=0x{execute_opcode:x}",
        "LIFECYCLE event=interrupt",
        "LIFECYCLE event=mailbox_response",
        "LIFECYCLE event=queue_head",
        "LIFECYCLE event=scheduler",
        "LIFECYCLE event=fence",
        "FW_REQUEST opcode=0x03",
        f"NPU1_FW_END {run_id}",
    ]
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
    def test_schedules_are_seeded_deterministic_and_balanced(self):
        self.assertEqual(fw.vertical_schedule(17), fw.vertical_schedule(17))
        self.assertNotEqual(fw.vertical_schedule(17), fw.vertical_schedule(18))
        schedule = fw.repetition_schedule(50, 50, 1234)
        self.assertEqual(sum(entry.arm == fw.TREATMENT for entry in schedule), 50)
        self.assertEqual(sum(entry.arm == fw.CONTROL for entry in schedule), 50)
        self.assertEqual(schedule, fw.repetition_schedule(50, 50, 1234))

    def test_output_oracle_requires_exact_values_and_pass_marker(self):
        result = fw.parse_output(successful_stdout())
        self.assertTrue(result.ok)
        self.assertEqual(result.values, tuple(range(2, 66)))
        for text in [
            "PASS!",
            successful_stdout().replace("Correct output 12 == 12", "Correct output 99 == 12"),
            successful_stdout().replace("Correct output 12 == 12\n", ""),
            successful_stdout().replace(
                "Correct output 12 == 12\nCorrect output 13 == 13",
                "Correct output 13 == 13\nCorrect output 12 == 12",
            ),
        ]:
            self.assertFalse(fw.parse_output(text).ok)

    def test_lifecycle_requires_ordered_unique_mode_specific_opcodes(self):
        for arm in (fw.TREATMENT, fw.CONTROL):
            entry = fw.ScheduleEntry(0, 0, arm)
            parsed = fw.parse_lifecycle(lifecycle(entry.run_id, arm.execute_opcode), entry)
            self.assertTrue(parsed.ok, parsed.reason)
            self.assertEqual(parsed.execute_opcode, arm.execute_opcode)

        entry = fw.ScheduleEntry(0, 0, fw.TREATMENT)
        valid = lifecycle(entry.run_id, fw.TREATMENT.execute_opcode)
        mutations = [
            valid.replace("FW_REQUEST opcode=0x18", ""),
            valid.replace("FW_REQUEST opcode=0x18", "FW_REQUEST opcode=0x10"),
            valid.replace(
                "FW_REQUEST opcode=0x18",
                "FW_REQUEST opcode=0x18\nFW_REQUEST opcode=0x18",
            ),
            valid.replace(
                "LIFECYCLE event=interrupt\nLIFECYCLE event=mailbox_response",
                "LIFECYCLE event=mailbox_response\nLIFECYCLE event=interrupt",
            ),
        ]
        for text in mutations:
            self.assertFalse(fw.parse_lifecycle(text, entry).ok)

    def test_run_classification_detects_faults_process_failures_and_cleanup(self):
        entry = fw.ScheduleEntry(0, 0, fw.TREATMENT)
        self.assertEqual(
            fw.classify_run(entry, command(), capture(entry)).outcome,
            fw.Outcome.SUCCESS,
        )
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

    def test_campaign_stops_at_first_failure_without_retry(self):
        schedule = fw.repetition_schedule(3, 3, 7)
        results = [
            fw.classify_run(schedule[0], command(), capture(schedule[0])),
            fw.classify_run(schedule[1], command(stdout="PASS!"), capture(schedule[1])),
            fw.classify_run(schedule[2], command(), capture(schedule[2])),
        ]
        campaign = fw.classify_campaign(schedule, results)
        self.assertEqual(campaign.outcome, fw.Outcome.SEMANTIC_MISMATCH)
        self.assertEqual(campaign.completed_run_ids, (schedule[0].run_id, schedule[1].run_id))
        self.assertEqual(campaign.failed_run_id, schedule[1].run_id)

    def test_emission_plan_is_byte_stable_and_reuses_fixture_requirements(self):
        schedule = fw.vertical_schedule(17)
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
            request = fw.CaptureRequest.synthetic(root, fw.vertical_schedule(17))
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
                Path(directory), fw.vertical_schedule(17)
            )
            plan = fw.build_transaction_plan(request, 1000, submitted=False)
            self.assertEqual(len(plan.runs), 2)
            self.assertTrue(all(run.argv[:3] == ("runuser", "-u", "1000") for run in plan.runs))
            self.assertTrue(plan.rollback)
            post_traffic = fw.build_transaction_plan(request, 1000, submitted=True)
            self.assertEqual(post_traffic.rollback, ())
            flattened = " ".join(
                part
                for command_spec in (*plan.setup, *plan.runs, *plan.cleanup, *plan.rollback)
                for part in command_spec.argv
            )
            for forbidden in ("xrt-smi", "suspend", "reboot", "reset", "pm-cycle"):
                self.assertNotIn(forbidden, flattened)

    def test_terminal_status_and_transient_service_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = root / "status.json"
            fw.write_terminal_status(status, {"state": "complete", "runs": 2})
            self.assertEqual(json.loads(status.read_text()), {"runs": 2, "state": "complete"})
            argv = fw.transient_service_argv(
                Path("/repo/tools/npu1_firmware_evidence.py"),
                "campaign.test",
                17,
                Path("/reserve/locations.json"),
                Path("/reserve/module.json"),
            )
            self.assertEqual(argv[:3], ("systemd-run", "--user", "--collect"))
            self.assertIn("campaign.test", argv)


if __name__ == "__main__":
    unittest.main()
