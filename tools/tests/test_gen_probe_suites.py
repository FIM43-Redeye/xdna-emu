"""The acc-timing and collision generators write suites that the runner and analyzers read."""
import gen_acc_timing_probe as acc
import gen_collision_probe as col
from harness import suite as S


def test_acc_timing_generator_writes_a_suite(tmp_path):
    pairs = acc.valid_pairs()[:4]
    m = acc.generate(str(tmp_path), pairs, 2)
    s = S.load_suite(tmp_path)
    assert s.cases == ("k000", "k001") and s.meta == {"in_size": acc.IN_SIZE, "record_bytes": acc.RECORD_BYTES}
    assert [e["name"] for e in s.case_entries()] == [k["name"] for k in m["kernels"]]
    core = s.topology("k000").cores[0]
    assert "test_kernel" in core.source.read_text()
    assert len(core.args[0].data) == acc.IN_SIZE
    assert core.args[1].size_bytes == m["kernels"][0]["out_size"]


def test_collision_suite_writer_shapes_cases_and_meta(tmp_path):
    entry = {"name": "k000", "dom": "i", "points": [{"id": 0}], "in_size": col.IN_SIZE,
             "out_size": col.RECORD_BYTES}
    m = col.write_probe_suite(str(tmp_path), "demo", [(entry, "; asm\n")], {"catalogue": {"a": "b"}})
    s = S.load_suite(tmp_path)
    assert s.name == "demo" and s.meta["catalogue"] == {"a": "b"} and s.meta["record_bytes"] == col.RECORD_BYTES
    assert s.case_entries() == [entry] and m["kernels"] == [entry]
