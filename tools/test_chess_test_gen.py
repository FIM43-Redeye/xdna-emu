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

    def test_strip_chess_output_qualifier(self):
        text = 'void load_lut(const void *lut, chess_output v16int32 &v1);\n'
        clean, _ = preprocess_chess_header(text)
        assert "chess_output" not in clean
        assert "v16int32 &v1" in clean

    def test_strip_chess_error(self):
        text = '  chess_error("idx must be in range [0,3]");\n'
        clean, _ = preprocess_chess_header(text)
        assert "chess_error" not in clean
        # Statement is replaced by a no-op expression.
        assert "((void)0)" in clean

    def test_strip_chess_dont_care(self):
        text = '  v8cint32 r = (v8cint32)chess_dont_care(v16int32);\n'
        clean, _ = preprocess_chess_header(text)
        assert "chess_dont_care" not in clean
        # Replaced with value-initialization of the named type.
        assert "v16int32{}" in clean

    def test_strip_chess_dont_care_direct(self):
        text = '  cint32_w64 w = chess_dont_care(cint32_w64);\n'
        clean, _ = preprocess_chess_header(text)
        assert "chess_dont_care" not in clean
        assert "cint32_w64{}" in clean

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
        assert "chess_output" not in clean
        assert "chess_error(" not in clean
        assert "chess_dont_care(" not in clean
        assert len(annotations) > 50
        dont_care_funcs = [
            name for name, a in annotations.items()
            if "dont_care" in a.properties
        ]
        assert len(dont_care_funcs) >= 10


import tempfile  # noqa: E402 -- used by TestClangParsing


class TestClangParsing:
    def test_parse_simple_stub_and_declaration(self):
        """clang.cindex can parse a type stub + function declaration."""
        import clang.cindex
        source = """
struct v16int32 { char _data[64]; };
struct v64int8 { char _data[64]; };
v16int32 broadcast_to_v16int32(int);
v64int8 some_vector_op(v16int32, v16int32, int);
"""
        index = clang.cindex.Index.create()
        tu = index.parse("test.cpp", unsaved_files=[("test.cpp", source)],
                         args=["-std=c++17", "-fsyntax-only"])
        errors = [d for d in tu.diagnostics if d.severity >= clang.cindex.Diagnostic.Error]
        assert len(errors) == 0, f"Parse errors: {[d.spelling for d in errors]}"
        funcs = []
        for cursor in tu.cursor.get_children():
            if cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                funcs.append(cursor.spelling)
        assert "broadcast_to_v16int32" in funcs
        assert "some_vector_op" in funcs

    def test_parse_namespace_declarations(self):
        """clang.cindex handles me_primitive namespace blocks."""
        import clang.cindex
        source = """
struct v512w8 { char _data[64]; };
struct v32w32 { char _data[128]; };
struct pmode_t { char _data[4]; };
struct smode_t { char _data[4]; };
namespace me_primitive {
v512w8 prmx_hw_prom(v32w32, pmode_t, smode_t);
} //namespace me_primitive
namespace me_primitive {
int some_other_prim(int, int);
} //namespace me_primitive
"""
        index = clang.cindex.Index.create()
        tu = index.parse("test.cpp", unsaved_files=[("test.cpp", source)],
                         args=["-std=c++17", "-fsyntax-only"])
        errors = [d for d in tu.diagnostics if d.severity >= clang.cindex.Diagnostic.Error]
        assert len(errors) == 0, f"Parse errors: {[d.spelling for d in errors]}"

    def test_parse_overloaded_functions(self):
        """clang.cindex distinguishes overloaded function signatures."""
        import clang.cindex
        source = """
struct v16int32 { char _data[64]; };
struct v16acc64 { char _data[128]; };
struct v32int16 { char _data[64]; };
struct v64uint8 { char _data[64]; };
v16acc64 mul_2x8_8x8(v32int16 a, v64uint8 b);
v16acc64 mul_2x8_8x8(v32int16 a, int sgn_x, v64uint8 b, int sgn_y);
"""
        index = clang.cindex.Index.create()
        tu = index.parse("test.cpp", unsaved_files=[("test.cpp", source)],
                         args=["-std=c++17", "-fsyntax-only"])
        funcs = []
        for cursor in tu.cursor.get_children():
            if cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                params = [c.type.spelling for c in cursor.get_children()
                          if c.kind == clang.cindex.CursorKind.PARM_DECL]
                funcs.append((cursor.spelling, params))
        mul_overloads = [f for f in funcs if f[0] == "mul_2x8_8x8"]
        assert len(mul_overloads) == 2
        param_counts = sorted(len(f[1]) for f in mul_overloads)
        assert param_counts == [2, 4]

    def test_parse_preprocessed_real_header(self):
        """Integration: pre-process + stub + parse the real me_chess_opns.h."""
        from pathlib import Path
        from chess_preprocess import preprocess_chess_header
        from chess_type_stubs import parse_chess_traits, generate_stub_header
        import clang.cindex

        opns_path = Path(__file__).parent.parent.parent / "aietools/data/aie_ml/lib/isg/me_chess_opns.h"
        types_path = Path(__file__).parent.parent.parent / "aietools/data/aie_ml/lib/isg/me_chess_types.h"
        if not opns_path.exists() or not types_path.exists():
            pytest.skip("aietools not available")

        traits = parse_chess_traits(types_path.read_text())
        stubs = generate_stub_header(traits)
        clean, annotations = preprocess_chess_header(opns_path.read_text())
        combined = stubs + "\n" + clean

        index = clang.cindex.Index.create()
        tu = index.parse("me_chess_opns_clean.cpp",
                         unsaved_files=[("me_chess_opns_clean.cpp", combined)],
                         args=["-std=c++17", "-fsyntax-only"])

        errors = [d for d in tu.diagnostics if d.severity >= clang.cindex.Diagnostic.Error]

        func_count = 0
        def walk(cursor):
            nonlocal func_count
            if cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                func_count += 1
            for child in cursor.get_children():
                walk(child)
        walk(tu.cursor)

        # Should find hundreds of functions even with some parse errors
        assert func_count >= 500, (
            f"Only found {func_count} functions, expected 500+. "
            f"Errors: {len(errors)}"
        )
        print(f"Parsed {func_count} functions, {len(errors)} errors, "
              f"{len(annotations)} annotations")


import importlib  # noqa: E402 -- used by chess-test-gen tests

# The module name contains a hyphen, so standard import cannot be used.
chess_test_gen = importlib.import_module("chess-test-gen")


class TestAnnotationJoining:
    """End-to-end tests: chess_property -> preprocess -> walk -> filter."""

    def test_annotation_reaches_filter(self):
        """chess_property(dont_care) -> preprocess -> walk -> filter = skipped."""
        from chess_preprocess import preprocess_chess_header
        original = """
struct v16int32 { char _data[64]; };
v16int32 undef_v16int32() chess_property(dont_care);
v16int32 broadcast_to_v16int32(int);
"""
        clean, annotations = preprocess_chess_header(original)
        intrinsics = chess_test_gen.walk_ast(clean, annotations)
        assert len(intrinsics) == 2
        for i in intrinsics:
            status, reason = chess_test_gen.classify_chess_intrinsic(i)
            if i.name == "undef_v16int32":
                assert status == "skipped"
                assert "dont_care" in reason
            elif i.name == "broadcast_to_v16int32":
                assert status == "generated"

    def test_volatile_annotation_reaches_filter(self):
        """chess_property(volatile) -> preprocess -> walk -> filter = skipped."""
        from chess_preprocess import preprocess_chess_header
        original = "struct dummy { char _data[4]; };\ndummy acquire_guarded(unsigned, unsigned) chess_property(volatile);\n"
        clean, annotations = preprocess_chess_header(original)
        intrinsics = chess_test_gen.walk_ast(clean, annotations)
        assert len(intrinsics) == 1
        status, _ = chess_test_gen.classify_chess_intrinsic(intrinsics[0])
        assert status == "skipped"


class TestChessASTWalker:
    """Unit tests for walk_ast -- verify correct extraction from clang.cindex."""

    def test_walk_simple_function(self):
        """Extract name, return_type, return_size, params, namespace for a plain function."""
        source = """
struct v16int32 { char _data[64]; };
v16int32 broadcast_to_v16int32(int x);
"""
        intrinsics = chess_test_gen.walk_ast(source, {})
        assert len(intrinsics) == 1
        i = intrinsics[0]
        assert i.name == "broadcast_to_v16int32"
        assert i.return_type == "v16int32"
        assert i.return_size == 64
        assert i.namespace == ""
        assert len(i.params) == 1
        assert i.params[0][0] == "int"
        assert i.params[0][1] == 4

    def test_walk_namespace(self):
        """Functions inside me_primitive namespace have namespace='me_primitive'."""
        source = """
struct v16int32 { char _data[64]; };
namespace me_primitive {
v16int32 load_vec(int);
} // namespace me_primitive
"""
        intrinsics = chess_test_gen.walk_ast(source, {})
        assert len(intrinsics) == 1
        assert intrinsics[0].namespace == "me_primitive"
        assert intrinsics[0].name == "load_vec"

    def test_walk_overloads(self):
        """Overloaded functions produce separate entries with different overload_index."""
        source = """
struct v16acc64 { char _data[128]; };
struct v32int16 { char _data[64]; };
struct v64uint8 { char _data[64]; };
v16acc64 mul_2x8_8x8(v32int16 a, v64uint8 b);
v16acc64 mul_2x8_8x8(v32int16 a, int sgn_x, v64uint8 b, int sgn_y);
"""
        intrinsics = chess_test_gen.walk_ast(source, {})
        overloads = [i for i in intrinsics if i.name == "mul_2x8_8x8"]
        assert len(overloads) == 2
        indices = sorted(i.overload_index for i in overloads)
        assert indices[0] != indices[1]

    def test_walk_void_return(self):
        """void functions are extracted but with return_size=0 (filtered later)."""
        source = "void noop_func(int x);\n"
        intrinsics = chess_test_gen.walk_ast(source, {})
        assert len(intrinsics) == 1
        assert intrinsics[0].return_type == "void"
        assert intrinsics[0].return_size == 0


class TestChessFilter:
    """Unit tests for classify_chess_intrinsic -- verify skip criteria."""

    def _make(self, name="f", namespace="", return_type="v16int32",
              return_size=64, params=None, is_inline=False,
              properties=None, storage_params=None, overload_index=0,
              source_line=1):
        """Build a minimal ChessIntrinsic for filter testing."""
        return chess_test_gen.ChessIntrinsic(
            name=name,
            namespace=namespace,
            return_type=return_type,
            return_size=return_size,
            params=params or [("int", 4)],
            is_inline=is_inline,
            properties=properties or [],
            storage_params=storage_params or [],
            overload_index=overload_index,
            source_line=source_line,
        )

    def test_pass_pure_function(self):
        """A clean function with no skip-triggers passes the filter."""
        i = self._make()
        status, _ = chess_test_gen.classify_chess_intrinsic(i)
        assert status == "generated"

    def test_filter_dont_care(self):
        i = self._make(properties=["dont_care"])
        status, reason = chess_test_gen.classify_chess_intrinsic(i)
        assert status == "skipped"
        assert "dont_care" in reason

    def test_filter_volatile(self):
        i = self._make(properties=["volatile"])
        status, reason = chess_test_gen.classify_chess_intrinsic(i)
        assert status == "skipped"
        assert "volatile" in reason

    def test_filter_void_return(self):
        i = self._make(return_type="void", return_size=0)
        status, reason = chess_test_gen.classify_chess_intrinsic(i)
        assert status == "skipped"
        assert "void" in reason.lower() or "return" in reason.lower()

    def test_filter_operator(self):
        i = self._make(name="operator+")
        status, reason = chess_test_gen.classify_chess_intrinsic(i)
        assert status == "skipped"
        assert "operator" in reason

    def test_filter_diagnostic(self):
        """Functions like chess_report / chess_assert / chess_error are skipped."""
        for prefix in ("chess_report_", "chess_assert_something",
                       "chess_error_handler", "chess_warning_",
                       "chess_exit", "chess_stop"):
            i = self._make(name=prefix)
            status, _ = chess_test_gen.classify_chess_intrinsic(i)
            assert status == "skipped", f"Expected skip for {prefix}"

    def test_filter_storage_params(self):
        i = self._make(storage_params=["TM"])
        status, reason = chess_test_gen.classify_chess_intrinsic(i)
        assert status == "skipped"
        assert "storage" in reason.lower()

    def test_filter_unsized_return(self):
        """return_size <= 0 triggers skip even if return_type is non-void."""
        i = self._make(return_type="unknown_t", return_size=0)
        status, reason = chess_test_gen.classify_chess_intrinsic(i)
        assert status == "skipped"

    def test_filter_non_functional(self):
        i = self._make(properties=["non_functional"])
        status, _ = chess_test_gen.classify_chess_intrinsic(i)
        assert status == "skipped"

    def test_filter_keep_with_operand(self):
        i = self._make(properties=["keep_with_operand"])
        status, _ = chess_test_gen.classify_chess_intrinsic(i)
        assert status == "skipped"

    def test_filter_arg_mem_only(self):
        i = self._make(properties=["arg_mem_only"])
        status, _ = chess_test_gen.classify_chess_intrinsic(i)
        assert status == "skipped"


class TestChessDirectoryName:
    """Unit tests for dir_name -- verify stable directory name generation."""

    def test_simple_function(self):
        """Single-arg function: name__param_type."""
        i = chess_test_gen.ChessIntrinsic(
            name="broadcast_to_v16int32", namespace="", return_type="v16int32",
            return_size=64, params=[("int", 4)], is_inline=False,
            properties=[], storage_params=[], overload_index=0, source_line=1,
        )
        assert chess_test_gen.dir_name(i) == "broadcast_to_v16int32__int"

    def test_multi_arg_function(self):
        """Multi-arg function: name__type1_type2."""
        i = chess_test_gen.ChessIntrinsic(
            name="mul_2x8_8x8", namespace="", return_type="v16acc64",
            return_size=128, params=[("v32int16", 64), ("v64uint8", 64)],
            is_inline=False, properties=[], storage_params=[],
            overload_index=0, source_line=1,
        )
        assert chess_test_gen.dir_name(i) == "mul_2x8_8x8__v32int16_v64uint8"

    def test_no_args(self):
        """Zero-arg function: just the name."""
        i = chess_test_gen.ChessIntrinsic(
            name="get_something", namespace="", return_type="int",
            return_size=4, params=[], is_inline=False,
            properties=[], storage_params=[], overload_index=0, source_line=1,
        )
        assert chess_test_gen.dir_name(i) == "get_something"
