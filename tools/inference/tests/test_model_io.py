import pytest
from inference.model_io import MODULE_PKT_TYPE, to_domain_key, SidecarError


def test_module_pkt_type_mapping_matches_decoder():
    # Confirmed from build/experiments/sp3-spike-trace/spike.events.json
    # (row0=shim=2, row1=memtile=3, compute=core=0) + dma-fill-measure.py (mem=1).
    assert MODULE_PKT_TYPE == {"core": 0, "mem": 1, "shim": 2, "memtile": 3}


def test_to_domain_key_translates_module_kind_to_pkt_type():
    assert to_domain_key(1, 2, "core") == "1|2|0"
    assert to_domain_key(1, 2, "mem") == "1|2|1"
    assert to_domain_key(0, 0, "shim") == "0|0|2"
    assert to_domain_key(1, 1, "memtile") == "1|1|3"


def test_to_domain_key_rejects_unknown_module_kind():
    with pytest.raises(SidecarError):
        to_domain_key(1, 2, "bogus")
