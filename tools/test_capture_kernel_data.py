"""Data functions of the single-purpose capture generators (acc_addsub, vshift_wide).

Each generator's kernel_data() must agree with the design and with the committed
silicon golden: the staged buffers are what the host copies into the BOs and
compares against.
"""

import json

import pytest

import capture_kernel_render as render
import gen_acc_addsub_f32 as acc
import gen_vshift_wide as vsh
import vv_data

MODULES = [acc, vsh]


@pytest.mark.parametrize("mod", MODULES, ids=lambda m: m.KERNEL)
def test_buffers_are_the_packed_design_and_the_golden_output(mod):
    golden = json.load(open(mod.GOLDEN))
    data = mod.kernel_data()
    assert len(data.inputs) == 1
    assert vv_data.unpack(data.inputs[0], "uint32_t") == mod.pack_input(mod.design())
    assert vv_data.unpack(data.expected, "uint32_t") == golden["silicon"]


@pytest.mark.parametrize("mod", MODULES, ids=lambda m: m.KERNEL)
def test_stale_golden_blocks_staging(mod, tmp_path, monkeypatch):
    golden = json.load(open(mod.GOLDEN))
    golden["inputs"][0][0] ^= 1
    stale = tmp_path / "g.json"
    stale.write_text(json.dumps(golden))
    monkeypatch.setattr(mod, "GOLDEN", str(stale))
    with pytest.raises(SystemExit):
        mod.kernel_data()


@pytest.mark.parametrize("mod", MODULES, ids=lambda m: m.KERNEL)
def test_no_golden_means_no_expected(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "GOLDEN", str(tmp_path / "absent.json"))
    data = mod.kernel_data()
    assert data.expected is None and len(data.inputs[0]) > 0


@pytest.mark.parametrize("mod", MODULES, ids=lambda m: m.KERNEL)
def test_generate_emits_run_lit_with_knobs_and_no_host(mod, tmp_path):
    files = mod.generate(str(tmp_path))
    assert "test.cpp" not in files
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(files)
    golden = json.load(open(mod.GOLDEN))
    assert f"-DVV_NIN=1 -DVV_OUT_T=uint32_t -DVV_NOUT={golden['n']}" in files["run.lit"]


def test_render_kernel_data_packs_the_given_input_type():
    data = render.kernel_data([1, 2], [3], in_ctype="uint16_t")
    assert data.inputs == [b"\x01\x00\x02\x00"] and data.expected == b"\x03\x00\x00\x00"
