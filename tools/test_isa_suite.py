"""Tests for isa_suite: the ISA generator's manifest rendered as a harness suite."""

import importlib
import json
import os

import pytest

import isa_suite
from harness import suite as S
from harness.device_model import load_device_model
from harness.plan import plan_channels
from harness.run import prng_bytes
from harness.topology import Cascade, CoreStream
from harness.validate import validate

SEED = 42


def _single(index=3, in_size=64, out_size=48, source_type="assembly", filename=None):
    ext = "ll" if source_type == "llvm_ir" else "s"
    return {
        "batch_index": index, "filename": filename or f"batch_{index:03d}.{ext}",
        "source_type": source_type, "test_count": 2, "in_size": in_size, "out_size": out_size,
        "tests": [
            {"instruction": "ADD", "slot": "alu", "combo_index": 0, "operands": {"r0": "r1"},
             "in_offset": 0, "in_size": 32, "out_offset": 0, "out_size": 16},
            {"instruction": "SUB", "slot": "alu", "combo_index": 1, "operands": {},
             "in_offset": 32, "in_size": 32, "out_offset": 32, "out_size": 16},
        ]}


def _cascade(index=7):
    return {
        "batch_index": index, "filename": f"batch_{index:03d}_consumer.s",
        "source_type": "cascade_pair",
        "producer_filename": f"batch_{index:03d}_producer.s",
        "consumer_filename": f"batch_{index:03d}_consumer.s",
        "producer_in_size": 64, "producer_out_size": 8,
        "consumer_in_size": 0, "consumer_out_size": 12,
        "instruction": "VMOV_HI", "slot": "mv", "test_count": 1, "in_size": 64, "out_size": 20,
        "tests": [{"instruction": "VMOV_HI", "slot": "mv", "combo_index": 0, "operands": {},
                   "in_offset": 0, "in_size": 64, "out_offset": 0, "out_size": 20}]}


def _stream(index=9):
    b = _cascade(index)
    b.update(source_type="stream_pair", producer_in_size=0, in_size=0, consumer_out_size=4,
             out_size=12, instruction="MOV_D1")
    b["tests"] = [{**b["tests"][0], "instruction": "MOV_D1", "in_size": 0, "out_size": 12}]
    return b


def _manifest(*batches):
    return {"testable_instructions": 5, "skipped_instructions": 2, "skip_reasons": {"x": 2},
            "total_test_points": sum(len(b["tests"]) for b in batches),
            "total_batches": len(batches), "batches": list(batches)}


def _sources(*batches):
    out = {}
    for b in batches:
        for key in ("filename", "producer_filename", "consumer_filename"):
            if key in b:
                out[b[key]] = f"// {b[key]}\n"
    return out


class TestBuildCase:
    def test_assembly_batch_is_one_core_at_row_2(self):
        b = _single()
        case = isa_suite.build_case(b, _sources(b), SEED)
        assert case.name == "batch_003"
        (core,) = case.topology.cores
        assert (core.name, core.tile.col, core.tile.row) == ("core", 0, 2)
        assert str(core.source) == "kernel.s" and case.files["kernel.s"] == "// batch_003.s\n"
        inp, out = core.args
        assert (inp.name, inp.direction, inp.size_bytes) == ("in", "in", 64)
        assert (out.name, out.direction, out.size_bytes) == ("out", "out", 48)
        assert case.topology.links == ()

    def test_input_is_the_host_prng_stream_of_the_seed(self):
        b = _single()
        case = isa_suite.build_case(b, _sources(b), SEED)
        assert case.topology.cores[0].args[0].data == prng_bytes(64, SEED)
        other = isa_suite.build_case(b, _sources(b), 7)
        assert other.topology.cores[0].args[0].data == prng_bytes(64, 7)

    def test_llvm_ir_batch_keeps_its_suffix(self):
        b = _single(index=4, source_type="llvm_ir")
        case = isa_suite.build_case(b, _sources(b), SEED)
        assert str(case.topology.cores[0].source) == "kernel.ll"
        assert "kernel.ll" in case.files

    def test_test_points_slice_the_one_output_buffer(self):
        b = _single()
        meta = isa_suite.build_case(b, _sources(b), SEED).meta
        assert meta["source_type"] == "assembly" and meta["batch_index"] == 3
        assert [t["instruction"] for t in meta["tests"]] == ["ADD", "SUB"]
        assert meta["tests"][1]["slices"] == [
            {"core": "core", "buffer": "out", "offset": 32, "size": 16}]
        assert meta["tests"][1]["combo_index"] == 1

    def test_cascade_pair_is_producer_north_of_consumer(self):
        b = _cascade()
        case = isa_suite.build_case(b, _sources(b), SEED)
        prod, cons = case.topology.cores
        assert (prod.name, prod.tile.col, prod.tile.row) == ("producer", 0, 3)
        assert (cons.name, cons.tile.col, cons.tile.row) == ("consumer", 0, 2)
        assert case.topology.links == (Cascade("producer", "consumer"),)
        assert [(a.name, a.direction, a.size_bytes) for a in prod.args] == [
            ("in", "in", 64), ("out", "out", 8)]
        # The consumer kernel takes only its output pointer (p0).
        assert [(a.name, a.direction, a.size_bytes) for a in cons.args] == [("out", "out", 12)]
        assert set(case.files) == {"producer.s", "consumer.s"}

    def test_stream_pair_is_a_core_stream_with_output_only_kernels(self):
        b = _stream()
        case = isa_suite.build_case(b, _sources(b), SEED)
        prod, cons = case.topology.cores
        assert case.topology.links == (CoreStream("producer", "consumer", 0, 0),)
        assert [a.direction for a in prod.args] == ["out"]
        assert [a.direction for a in cons.args] == ["out"]

    def test_pair_test_point_slices_both_cores_outputs(self):
        b = _cascade()
        (t,) = isa_suite.build_case(b, _sources(b), SEED).meta["tests"]
        assert t["slices"] == [
            {"core": "producer", "buffer": "out", "offset": 0, "size": 8},
            {"core": "consumer", "buffer": "out", "offset": 0, "size": 12}]

    def test_a_pair_consumer_with_input_is_refused(self):
        b = _cascade()
        b["consumer_in_size"] = 8
        with pytest.raises(isa_suite.IsaSuiteError):
            isa_suite.build_case(b, _sources(b), SEED)

    def test_unknown_source_type_is_refused(self):
        b = _single(source_type="mystery")
        with pytest.raises(isa_suite.IsaSuiteError):
            isa_suite.build_case(b, _sources(b), SEED)

    @pytest.mark.parametrize("make", [_single, _cascade, _stream])
    def test_every_shape_validates_and_plans_on_npu1(self, make):
        b = make()
        topo = isa_suite.build_case(b, _sources(b), SEED).topology
        dev = load_device_model("npu1")
        assert validate(topo, dev) == []
        assert len(plan_channels(topo, dev).phases) == 1


class TestWriteSuite:
    def _write(self, tmp_path, batches, seed=SEED, src=None):
        src = src or tmp_path / "src"
        src.mkdir(exist_ok=True)
        for name, text in _sources(*batches).items():
            (src / name).write_text(text)
        suite_dir = tmp_path / "suite"
        isa_suite.write_isa_suite(_manifest(*batches), src, suite_dir, seed)
        return S.load_suite(suite_dir)

    def test_suite_lists_cases_in_manifest_order(self, tmp_path):
        s = self._write(tmp_path, [_single(0), _single(1, source_type="llvm_ir"), _cascade(2), _stream(3)])
        assert s.cases == ("batch_000", "batch_001", "batch_002", "batch_003")
        assert s.generator == "isa-test-gen.py"
        assert s.meta["seed"] == SEED and s.meta["total_test_points"] == 6
        assert s.meta["skip_reasons"] == {"x": 2}
        assert "batches" not in s.meta

    def test_case_json_and_topology_round_trip(self, tmp_path):
        s = self._write(tmp_path, [_single(0), _cascade(1)])
        assert s.case_meta("batch_001")["instruction"] == "VMOV_HI"
        assert s.case_meta("batch_000")["tests"][0]["instruction"] == "ADD"
        assert s.topology("batch_001").links == (Cascade("producer", "consumer"),)

    def test_regenerating_drops_cases_that_no_longer_exist(self, tmp_path):
        self._write(tmp_path, [_single(0), _single(1)])
        s = self._write(tmp_path, [_single(0)])
        assert s.cases == ("batch_000",)
        assert not (tmp_path / "suite" / "cases" / "batch_001").exists()

    def test_missing_source_file_is_an_error(self, tmp_path):
        (tmp_path / "src").mkdir()
        with pytest.raises(isa_suite.IsaSuiteError):
            isa_suite.write_isa_suite(_manifest(_single(0)), tmp_path / "src", tmp_path / "s", SEED)


class TestSelect:
    def _suite(self, tmp_path):
        batches = [_single(0), _cascade(1), _stream(2)]
        src = tmp_path / "src"
        src.mkdir()
        for name, text in _sources(*batches).items():
            (src / name).write_text(text)
        isa_suite.write_isa_suite(_manifest(*batches), src, tmp_path / "suite", SEED)
        return S.load_suite(tmp_path / "suite")

    def test_filter_matches_case_name_source_type_or_instruction(self, tmp_path):
        s = self._suite(tmp_path)
        assert isa_suite.filter_cases(s, "batch_000") == ["batch_000"]
        assert isa_suite.filter_cases(s, "cascade_pair") == ["batch_001"]
        assert isa_suite.filter_cases(s, "MOV_D1") == ["batch_002"]
        assert isa_suite.filter_cases(s, "ADD|VMOV") == ["batch_000", "batch_001"]
        assert isa_suite.filter_cases(s, "nomatch") == []


ANALYZER = importlib.import_module("isa-test-analyze")


def _bin(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class TestAnalyzer:
    """The analyzer reads two tags of a suite and compares each test point's slices."""

    def _run(self, tmp_path, hw, emu):
        """hw/emu: {case: {(core, buffer): bytes}}"""
        batches = [_single(0), _cascade(1)]
        src = tmp_path / "src"
        src.mkdir(parents=True)
        for name, text in _sources(*batches).items():
            (src / name).write_text(text)
        suite_dir = tmp_path / "suite"
        isa_suite.write_isa_suite(_manifest(*batches), src, suite_dir, SEED)
        for tag, data in (("hw-t", hw), ("emu-t", emu)):
            for case, bufs in data.items():
                for (core, buf), content in bufs.items():
                    _bin(S.output_path(suite_dir, tag, case, core, buf), content)
        return S.load_suite(suite_dir)

    def _good(self):
        return {"batch_000": {("core", "out"): bytes(range(48))},
                "batch_001": {("producer", "out"): b"\1" * 8, ("consumer", "out"): b"\2" * 12}}

    def test_identical_runs_pass_every_test_point(self, tmp_path, capsys):
        s = self._run(tmp_path, self._good(), self._good())
        res = ANALYZER.analyze(s, "hw-t", "emu-t")
        assert res == ANALYZER.Totals(passed=3, failed=0, skipped_cases=0)

    def test_a_difference_inside_a_slice_fails_only_that_test_point(self, tmp_path):
        emu = self._good()
        emu["batch_000"][("core", "out")] = bytes(range(47)) + b"\xff"      # inside SUB's slice
        res = ANALYZER.analyze(self._run(tmp_path, self._good(), emu), "hw-t", "emu-t")
        assert (res.passed, res.failed) == (2, 1)

    def test_a_difference_outside_every_slice_is_not_a_failure(self, tmp_path):
        emu = self._good()
        emu["batch_000"][("core", "out")] = bytes(range(16)) + b"\xee" * 16 + bytes(range(32, 48))
        res = ANALYZER.analyze(self._run(tmp_path, self._good(), emu), "hw-t", "emu-t")
        assert (res.passed, res.failed) == (3, 0)      # bytes 16..32 belong to no test point

    def test_a_pair_fails_when_either_core_differs(self, tmp_path):
        for core in ("producer", "consumer"):
            emu = self._good()
            emu["batch_001"][(core, "out")] = b"\x09" * len(emu["batch_001"][(core, "out")])
            res = ANALYZER.analyze(self._run(tmp_path / core, self._good(), emu), "hw-t", "emu-t")
            assert (res.passed, res.failed) == (2, 1), core

    def test_a_case_missing_from_a_tag_is_skipped_and_counted(self, tmp_path):
        emu = self._good()
        del emu["batch_001"]
        res = ANALYZER.analyze(self._run(tmp_path, self._good(), emu), "hw-t", "emu-t")
        assert (res.passed, res.failed, res.skipped_cases) == (2, 0, 1)


class TestRealGeneration:
    """The real ISA database through the generator: every batch becomes a runnable case."""

    @pytest.fixture(scope="class")
    def suite(self, tmp_path_factory):
        isa_json = os.path.join(os.path.dirname(__file__), "aie2-isa.json")
        if not os.path.exists(isa_json):
            pytest.skip("aie2-isa.json not found")
        gen = importlib.import_module("isa-test-gen")
        d = tmp_path_factory.mktemp("isa") / "suite"
        manifest = gen.generate_suite(isa_json, str(d), SEED)
        return manifest, S.load_suite(d)

    def test_one_case_per_batch_and_all_test_points_kept(self, suite):
        manifest, s = suite
        assert len(s.cases) == manifest["total_batches"]
        n = sum(len(s.case_meta(c)["tests"]) for c in s.cases)
        assert n == manifest["total_test_points"]

    def test_every_topology_validates_and_plans(self, suite):
        _, s = suite
        dev = load_device_model("npu1")
        for c in s.cases:
            topo = s.topology(c)
            assert validate(topo, dev) == [], c
            assert len(plan_channels(topo, dev).phases) == 1, c

    def test_pairs_have_two_cores_and_one_link(self, suite):
        _, s = suite
        pairs = [c for c in s.cases if s.case_meta(c)["source_type"] in isa_suite.PAIR_TYPES]
        assert len(pairs) == 17
        assert all(len(s.topology(c).cores) == 2 and len(s.topology(c).links) == 1 for c in pairs)


class TestHollowGuard:
    """A test point that checked nothing fails, even where both backends agree."""

    def _suite(self, tmp_path, batches, hw, emu=None):
        src = tmp_path / "src"
        src.mkdir(parents=True)
        for name, text in _sources(*batches).items():
            (src / name).write_text(text)
        suite_dir = tmp_path / "suite"
        isa_suite.write_isa_suite(_manifest(*batches), src, suite_dir, SEED)
        for tag, data in (("hw-t", hw), ("emu-t", emu or hw)):
            for case, bufs in data.items():
                for (core, buf), content in bufs.items():
                    _bin(S.output_path(suite_dir, tag, case, core, buf), content)
        return S.load_suite(suite_dir)

    def _pair(self, payload, variants=None):
        b = _cascade(0)
        b["tests"][0]["payload"] = payload
        if variants:
            b["tests"][0]["variants"] = variants
        return b

    CONSUMER = [{"core": "consumer", "buffer": "out", "offset": 0, "size": 12}]

    def test_an_all_zero_payload_is_hollow(self, tmp_path):
        bufs = {"batch_000": {("producer", "out"): b"\xaa" * 8, ("consumer", "out"): bytes(12)}}
        res = ANALYZER.analyze(self._suite(tmp_path, [self._pair(self.CONSUMER)], bufs), "hw-t", "emu-t")
        assert (res.passed, res.failed, res.hollow) == (0, 1, 1)

    def test_markers_outside_the_declared_payload_do_not_count(self, tmp_path):
        """The producer's marker words are non-zero, the consumer saw nothing."""
        bufs = {"batch_000": {("producer", "out"): b"\xaa\xcc" * 4, ("consumer", "out"): bytes(12)}}
        res = ANALYZER.analyze(self._suite(tmp_path, [self._pair(self.CONSUMER)], bufs), "hw-t", "emu-t")
        assert res.hollow == 1

    def test_without_a_declared_payload_the_whole_output_counts(self, tmp_path):
        b = _cascade(0)
        bufs = {"batch_000": {("producer", "out"): b"\xaa" * 8, ("consumer", "out"): bytes(12)}}
        res = ANALYZER.analyze(self._suite(tmp_path, [b], bufs), "hw-t", "emu-t")
        assert (res.passed, res.hollow) == (1, 0)

    def test_variants_that_are_all_identical_are_hollow(self, tmp_path):
        variants = [[{"core": "consumer", "buffer": "out", "offset": 4 * i, "size": 4}] for i in range(3)]
        same = {"batch_000": {("producer", "out"): bytes(8), ("consumer", "out"): b"\x07\x00\x00\x00" * 3}}
        res = ANALYZER.analyze(self._suite(tmp_path, [self._pair(self.CONSUMER, variants)], same), "hw-t", "emu-t")
        assert res.hollow == 1
        varied = {"batch_000": {("producer", "out"): bytes(8), ("consumer", "out"): bytes([1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0])}}
        res = ANALYZER.analyze(self._suite(tmp_path / "v", [self._pair(self.CONSUMER, variants)], varied), "hw-t", "emu-t")
        assert (res.passed, res.hollow) == (1, 0)

    def test_a_zero_among_other_results_of_an_instruction_is_not_hollow(self, tmp_path):
        """An undeclared test point is one combination of its instruction: false is an answer."""
        data = {"batch_000": {("core", "out"): bytes(16) + bytes(range(1, 33))}}
        a = _single(0)
        a["tests"] = [{**t, "instruction": "EQ"} for t in a["tests"]]
        res = ANALYZER.analyze(self._suite(tmp_path, [a], data), "hw-t", "emu-t")
        assert (res.passed, res.hollow) == (2, 0)

    def test_an_instruction_whose_every_result_is_zero_is_hollow(self, tmp_path):
        data = {"batch_000": {("core", "out"): bytes(48)}}
        a = _single(0)
        a["tests"] = [{**t, "instruction": "EQZ"} for t in a["tests"]]
        res = ANALYZER.analyze(self._suite(tmp_path, [a], data), "hw-t", "emu-t")
        assert (res.passed, res.hollow) == (0, 2)

    def test_a_zero_expected_test_point_is_not_hollow(self, tmp_path):
        """vclr on a loaded register: zero is the right answer."""
        data = {"batch_000": {("core", "out"): bytes(48)}}
        a = _single(0)
        a["tests"] = [{**t, "instruction": "VCLR_vclr", "zero_expected": True} for t in a["tests"]]
        res = ANALYZER.analyze(self._suite(tmp_path, [a], data), "hw-t", "emu-t")
        assert (res.passed, res.hollow) == (2, 0)

    def test_every_test_point_of_an_instruction_identical_is_hollow(self, tmp_path):
        same = {"batch_000": {("core", "out"): b"\x05" * 48}, "batch_001": {("core", "out"): b"\x05" * 48}}
        a, b = _single(0), _single(1)
        b["tests"] = [{**t, "instruction": "ADD"} for t in b["tests"]]
        a["tests"] = [{**t, "instruction": "ADD"} for t in a["tests"]]
        res = ANALYZER.analyze(self._suite(tmp_path, [a, b], same), "hw-t", "emu-t")
        assert res.hollow == 4 and res.passed == 0

    def test_distinct_points_of_an_instruction_pass(self, tmp_path):
        data = {"batch_000": {("core", "out"): bytes(range(1, 49))}}
        a = _single(0)
        a["tests"] = [{**t, "instruction": "ADD"} for t in a["tests"]]
        res = ANALYZER.analyze(self._suite(tmp_path, [a], data), "hw-t", "emu-t")
        assert (res.passed, res.hollow) == (2, 0)

    def test_hollow_reason_unit(self):
        assert ANALYZER.hollow_reason(b"\0\0", [], [b"\0\0"]).startswith("payload is all zero (")
        assert ANALYZER.hollow_reason(b"\0\0", [], [b"\0\0", b"\0\0"], declared=False).startswith("payload is all zero for all 2")
        assert ANALYZER.hollow_reason(b"\0\0", [], [b"\0\0", b"\1\0"], declared=False) is None
        assert ANALYZER.hollow_reason(b"\1", [b"a", b"a"], []).startswith("payload identical across its 2")
        assert ANALYZER.hollow_reason(b"\1", [], [b"a", b"a", b"a"]).startswith("payload identical across the instruction's 3")
        assert ANALYZER.hollow_reason(b"\1", [b"a", b"b"], [b"a", b"b"]) is None
