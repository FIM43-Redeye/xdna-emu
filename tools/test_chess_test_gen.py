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


from chess_preprocess import preprocess_chess_header, ChessAnnotation


class TestChessPreprocess:
    def test_strip_chess_property(self):
        text = 'mod_t undef_mod() chess_property(dont_care);\n'
        clean, annotations = preprocess_chess_header(text)
        assert "chess_property" not in clean
        assert "mod_t undef_mod();" in clean
        assert "undef_mod" in annotations
        assert "dont_care" in annotations["undef_mod"].properties

    def test_strip_chess_property_multiword(self):
        text = 'void acquire_guarded(unsigned, unsigned) chess_property(guarded_memory_fence volatile output_stage_offset_7);\n'
        clean, annotations = preprocess_chess_header(text)
        assert "chess_property" not in clean
        ann = annotations["acquire_guarded"]
        assert "guarded_memory_fence" in ann.properties
        assert "volatile" in ann.properties

    def test_strip_chess_storage_in_param(self):
        text = 'void acquire_equal_inner(const void *a, char chess_storage(TM) *mem);\n'
        clean, annotations = preprocess_chess_header(text)
        assert "chess_storage" not in clean
        ann = annotations["acquire_equal_inner"]
        assert "TM" in ann.storage_params

    def test_strip_if0_block(self):
        text = '''some_func();
#if 0//!
namespace me_primitive {
    void hidden_func();
} //namespace me_primitive
#endif//!
another_func();
'''
        clean, _ = preprocess_chess_header(text)
        assert "hidden_func" not in clean
        assert "some_func" in clean
        assert "another_func" in clean

    def test_strip_bang_comment_lines(self):
        text = '//!v256uint4_sparse sparse_pop_aux(...);\nreal_func();\n'
        clean, _ = preprocess_chess_header(text)
        assert "sparse_pop_aux" not in clean
        assert "real_func" in clean

    def test_strip_chess_protect_access(self):
        text = 'extern chess_protect_access v16acc32 chess_storage(SCD) scd;\n'
        clean, _ = preprocess_chess_header(text)
        assert "chess_protect_access" not in clean

    def test_replace_vbitzconstexpr(self):
        text = 'VBITzCONSTEXPR inline cint32(int, int) chess_property(do_generate);\n'
        clean, _ = preprocess_chess_header(text)
        assert "VBITzCONSTEXPR" not in clean
        assert "inline" in clean

    def test_strip_chess_manifest(self):
        text = 'if (chess_manifest(idx < 0 || idx > 3)) ;\n'
        clean, _ = preprocess_chess_header(text)
        assert "chess_manifest" not in clean

    def test_strip_chess_dont_warn_dead(self):
        text = 'chess_dont_warn_dead(cmp);\n'
        clean, _ = preprocess_chess_header(text)
        assert "chess_dont_warn_dead" not in clean

    def test_strip_chess_memory_fence(self):
        text = 'chess_memory_fence();\n'
        clean, _ = preprocess_chess_header(text)
        assert "chess_memory_fence" not in clean

    def test_strip_chess_separator_scheduler(self):
        text = 'chess_separator_scheduler();\n'
        clean, _ = preprocess_chess_header(text)
        assert "chess_separator_scheduler" not in clean

    def test_strip_chess_unroll_loop(self):
        text = 'for (int n = 0; n < 3; n++) chess_unroll_loop(*)\n'
        clean, _ = preprocess_chess_header(text)
        assert "chess_unroll_loop" not in clean

    def test_property_word_splitting(self):
        text = 'void f() chess_property(functional loop_free);\n'
        _, annotations = preprocess_chess_header(text)
        ann = annotations["f"]
        assert "functional" in ann.properties
        assert "loop_free" in ann.properties

    def test_ifdef_chess_error_stripped(self):
        text = '''#ifdef __chess__
#error "generated native file not intended for compilation by chess"
#endif
void real_func();
'''
        clean, _ = preprocess_chess_header(text)
        assert '#error' not in clean
        assert "real_func" in clean

    def test_multiple_annotations_same_name(self):
        text = '''int f(int) chess_property(volatile);
int f(int, int) chess_property(functional);
'''
        _, annotations = preprocess_chess_header(text)
        ann = annotations["f"]
        assert "volatile" in ann.properties
        assert "functional" in ann.properties

    def test_integration_real_file(self):
        from pathlib import Path
        opns_path = Path(__file__).parent.parent.parent / "aietools/data/aie_ml/lib/isg/me_chess_opns.h"
        if not opns_path.exists():
            pytest.skip("aietools not available")
        text = opns_path.read_text()
        clean, annotations = preprocess_chess_header(text)
        assert "chess_property(" not in clean
        assert "chess_storage(" not in clean
        assert "chess_manifest(" not in clean
        assert "chess_memory_fence()" not in clean
        assert "VBITzCONSTEXPR" not in clean
        assert len(annotations) > 50
        dont_care_funcs = [
            name for name, a in annotations.items()
            if "dont_care" in a.properties
        ]
        assert len(dont_care_funcs) >= 10
