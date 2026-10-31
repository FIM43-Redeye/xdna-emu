"""Tests for gen_pack.py: the designed inputs, the layout, and the staged data contract."""

import json
import os

import gen_pack as gen

# Core data memory is 64 KiB; keep the double-buffer-free working set well inside it.
DATA_MEMORY_BUDGET_BYTES = 48 * 1024


def int16_lanes(block):
    return [block[i] | block[i + 1] << 8 for i in range(0, len(block), 2)]


class TestProbeDesign:
    def test_every_byte_value_reaches_every_byte_position(self):
        seen = {p: set() for p in range(gen.BLOCK_BYTES)}
        for blk in gen.probe_blocks():
            for p, v in enumerate(blk):
                seen[p].add(v)
        for p, values in seen.items():
            assert values == set(range(256)), f"byte position {p} misses {set(range(256)) - values}"

    def test_every_clamp_value_reaches_every_int16_lane(self):
        seen = {l: set() for l in range(32)}
        for blk in gen.probe_blocks():
            for l, v in enumerate(int16_lanes(blk)):
                seen[l].add(v)
        want = set(gen.clamp_values_16())
        for l, values in seen.items():
            assert want <= values, f"int16 lane {l} misses {sorted(want - values)}"

    def test_clamp_values_include_every_bound_neighbour(self):
        signed = {v - 65536 if v >= 32768 else v for v in gen.clamp_values_16()}
        # Signed [-128, 127], symmetric [-127, 127] and unsigned [0, 255], inside / on / outside.
        for v in (126, 127, 128, 129, -127, -128, -129, -130, 254, 255, 256, 257, 32767, -32768):
            assert v in signed, v
        assert {0x8000, 0xFFFF, 0xFF80} <= set(gen.clamp_values_16())

    def test_one_hot_blocks_name_their_position(self):
        for l, blk in enumerate(gen.one_hot_int16()):
            assert [i for i, v in enumerate(int16_lanes(blk)) if v] == [l]
        for p, blk in enumerate(gen.one_hot_bytes()):
            assert [i for i, v in enumerate(blk) if v] == [p]

    def test_dense_run_covers_both_bounds_on_both_sides(self):
        vals = {v for blk in gen.dense_int16() for v in int16_lanes(blk)}
        for v in range(-320, 320):
            assert v & 0xFFFF in vals

    def test_blocks_fill_whole_chunks(self):
        for build in (gen.probe_blocks, gen.rand_blocks, gen.addr_blocks):
            blocks = build()
            assert len(blocks) % gen.BLOCKS_PER_CHUNK == 0
            assert all(len(b) == gen.BLOCK_BYTES for b in blocks)

    def test_held_out_set_is_not_the_probe(self):
        assert not set(map(tuple, gen.rand_blocks())) & set(map(tuple, gen.probe_blocks()))


class TestLayout:
    def test_pass_order_is_crsat_major(self):
        assert gen.PASSES == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]

    def test_header_carries_the_run_time_controls(self):
        hdr = gen.header_words()
        assert len(hdr) == gen.HEADER_WORDS
        assert hdr[0:2] == [0, 1] and hdr[4:8] == [0, 1, 2, 3]
        assert [(v + 2**31) % 2**32 - 2**31 for v in hdr[8:12]] == list(gen.IDX_OFFSETS)
        assert (hdr[3] + 2**31) % 2**32 - 2**31 == gen.STEP_BYTES
        assert hdr[16:24] == [0] * 8   # the zero vector the standalone kernel adds

    def test_vector_table_matches_the_output_shape(self):
        for kernel in (gen.LANE_KERNELS and next(iter(gen.LANE_KERNELS)), gen.ADDR_KERNEL):
            table = gen.vector_table(kernel)
            per_block = gen.words_per_block(kernel)
            used = len(table) * gen.WORDS_PER_VECTOR
            # The addr kernel ends every pass with one pointer record.
            assert used <= per_block
            assert {t["form"] for t in table} == {f[0] for f in gen.FORMS}

    def test_addr_postmod_forms_run_backwards(self):
        table = [t for t in gen.vector_table(gen.ADDR_KERNEL) if t["group"] == "fused_postmod"][:4]
        assert [t["form"] for t in table] == [f[0] for f in gen.FORMS[::-1]]

    def test_vectors_are_32_byte_aligned(self):
        assert gen.WORDS_PER_VECTOR * 4 % 32 == 0
        assert gen.WORDS_PER_PASS % 8 == 0 and gen.ADDR_WORDS_PER_PASS % 8 == 0
        assert gen.HEADER_WORDS % 8 == 0 and gen.BLOCK_WORDS % 8 == 0
        assert gen.CRT_WORDS_PER_BLOCK % 8 == 0


class TestKernelSource:
    def test_d_forms_take_the_run_time_sign_and_s_forms_do_not(self):
        src = gen.render_lane_kernel()
        assert "pack(a, sgn)" in src and "pack(b, sgn)" in src
        assert "pack(a);" in src and "pack(b);" in src

    def test_crsat_is_a_run_time_value(self):
        for src in (gen.render_lane_kernel(), gen.render_addr_kernel(), gen.render_crt_kernel()):
            assert "set_saturation(static_cast<aie::saturation_mode>(" in src
            assert "saturation_mode::" not in src.replace("aie::saturation_mode>(", "")

    def test_each_register_write_is_in_the_function_that_uses_it(self):
        src = gen.render_lane_kernel()
        for fn in ("fused(", "standalone("):
            body = src.split(f"static void {fn}")[1].split("}\n")[0]
            assert "set_saturation" in body


class TestProof:
    def test_addressing_classes(self):
        cls = gen.addressing_class
        assert cls("vst.pack.d8.d16\tx0, [p1, #0x40]") == "idx_imm"
        assert cls("vst.pack.d8.d16\tx0, [p1, dj0]") == "idx"
        assert cls("vst.pack.s8.s16\tx0, [p1], #0x20") == "pstm_imm"
        assert cls("vst.pack.s4.s8\tx0, [p2], m0") == "pstm"
        assert cls("vpack.d4.d8\twl1, x1") == "reg"

    def test_mnemonic_filter(self):
        for m in ("vpack.s8.s16", "vst.pack.d4.d8", "vst.2d.pack.s4.s8", "vst.3d.pack.d8.d16"):
            assert gen.PACK_MNEMONIC.match(m)
        for m in ("vst.srs.s16.s32", "vunpack.s16.s8", "vldb.unpack.s16.s8", "vst.conv.bf16.fp32"):
            assert not gen.PACK_MNEMONIC.match(m)


class TestStagedData:
    def test_every_kernel_fits_data_memory_and_the_shim_repeat_limit(self):
        for name in gen.kernel_names():
            chunks = gen.n_chunks(name)
            assert chunks <= 64, name
            per_chunk_in = len(gen.input_words(name)) // chunks
            per_chunk_out = gen.n_out(name) // chunks
            assert (per_chunk_in + per_chunk_out) * 4 <= DATA_MEMORY_BUDGET_BYTES, name

    def test_inputs_match_the_committed_golden(self):
        for name in gen.kernel_names():
            path = gen.golden_path(name)
            assert os.path.exists(path), name
            with open(path) as f:
                golden = json.load(f)
            assert golden["inputs"] == [gen.input_words(name)], name
            assert len(golden["silicon"]) == gen.n_out(name), name

    def test_generated_kernels_are_current(self):
        for name in gen.kernel_names():
            with open(os.path.join(gen.VERIFY, name, "aie.mlir")) as f:
                assert f"{gen.n_chunks(name)} chunks" in f.read(), name
