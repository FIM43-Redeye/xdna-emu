"""Tests for config_extract.dump_model — load_dump and ConfigDump schema."""
import pytest


def test_load_dump_reads_start_col(tmp_path):
    from config_extract.dump_model import load_dump

    p = tmp_path / "d.json"
    p.write_text('{"device":"npu1","start_col":1,"route_graph":{"edges":[]},"tiles":[]}')
    assert load_dump(p).start_col == 1


def test_load_dump_start_col_absent_is_none(tmp_path):
    from config_extract.dump_model import load_dump

    p = tmp_path / "d.json"
    p.write_text('{"device":"npu1","route_graph":{"edges":[]},"tiles":[]}')
    assert load_dump(p).start_col is None
