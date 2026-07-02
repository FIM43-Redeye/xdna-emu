from calibration.skew.schema import empty_constants, write_constants, read_constants


def test_empty_constants_has_provenance_fields():
    c = empty_constants()
    for k in ("d_h", "d_v", "intra", "fit_residual", "source_route", "provenance",
              "per_channel", "b_vector_range", "jitter_range", "assumptions", "d_h_path"):
        assert k in c, k
    assert c["assumptions"] == {}


def test_write_read_roundtrip_with_provenance(tmp_path):
    p = tmp_path / "skew_constants.json"
    write_constants(
        p, d_h=4.0, d_v=2.0, intra_core=0.0, intra_mem=0.0, fit_residual=1e-9,
        source_route="shim_row", provenance="phase2 N=20 spaced",
        per_channel={"ch14": 4.0, "ch13": 4.0}, b_vector_range=0.0,
        jitter_range=0.0, assumptions={"horizontal_direction_isotropy": "assumed"},
        d_h_path="shim_row",
    )
    got = read_constants(p)
    assert got["d_h_path"] == "shim_row"
    assert got["per_channel"]["ch14"] == 4.0
    assert got["assumptions"]["horizontal_direction_isotropy"] == "assumed"
    assert got["b_vector_range"] == 0.0
    assert got["jitter_range"] == 0.0


def test_write_constants_defaults_assumptions_and_ranges(tmp_path):
    p = tmp_path / "c.json"
    write_constants(p, d_h=4.0, d_v=2.0)  # assumptions omitted -> {}
    got = read_constants(p)
    assert got["assumptions"] == {}
    assert got["per_channel"] is None
    assert got["b_vector_range"] is None
    assert got["jitter_range"] is None
