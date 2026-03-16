"""Unit tests for chess_type_stubs.py (Chess type stub generator)."""

import pytest
from chess_type_stubs import parse_chess_traits, generate_stub_header


class TestChessTypeStubs:
    def test_parse_vector_type(self):
        text = """template <> struct chessTraitsOf<v16int32> {
    static const unsigned bits = 512;
    static const unsigned elems = 16;
};"""
        traits = parse_chess_traits(text)
        assert traits["v16int32"] == 512

    def test_parse_primitive_type_with_sizeof(self):
        text = """template <> struct chessTraitsOf<int> {
    static const unsigned bits = sizeof(int) * __CHAR_BIT__; // pertinent to host, may differ from target 32;
};"""
        traits = parse_chess_traits(text)
        assert traits["int"] == 32

    def test_skip_commented_out(self):
        text = """//!template <> struct chessTraitsOf<void *> {
//!    static const unsigned bits = 64;
//!};
template <> struct chessTraitsOf<v8acc64> {
    static const unsigned bits = 512;
};"""
        traits = parse_chess_traits(text)
        assert "void *" not in traits
        assert traits["v8acc64"] == 512

    def test_generate_stub_header(self):
        traits = {"v16int32": 512, "bfloat16": 16, "mask64": 64}
        header = generate_stub_header(traits)
        assert "struct v16int32 { char _data[64]; };" in header
        assert "struct bfloat16 { char _data[2]; };" in header
        assert "struct mask64 { char _data[8]; };" in header

    def test_skip_builtin_c_types(self):
        traits = {"int": 32, "float": 32, "v16int32": 512}
        header = generate_stub_header(traits)
        assert "struct int " not in header
        assert "struct float " not in header
        assert "struct v16int32" in header

    def test_parse_accumulator_type(self):
        text = """template <> struct chessTraitsOf<v1acc32> {
    static const unsigned bits = 32;
};"""
        traits = parse_chess_traits(text)
        assert traits["v1acc32"] == 32

    def test_parse_real_file(self):
        from pathlib import Path
        types_path = Path(__file__).parent.parent.parent / "aietools/data/aie_ml/lib/isg/me_chess_types.h"
        if not types_path.exists():
            pytest.skip("aietools not available")
        text = types_path.read_text()
        traits = parse_chess_traits(text)
        assert traits["v16int32"] == 512
        assert traits["v8acc64"] == 512
        # bfloat16 is commented out in the real file (//! prefix), so it must
        # not appear -- verify the skip-commented logic works on real content.
        assert "bfloat16" not in traits
        # A type known to be 16 bits and not commented out.
        assert traits["v4int4"] == 16
        assert traits["v64int8"] == 512
        assert len(traits) >= 100
