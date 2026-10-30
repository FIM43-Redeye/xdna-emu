"""Tests for gen_agu_2d3d.py: the designed inputs, the memory budget and the staged data contract."""

import json
import os

import gen_agu_2d3d as gen

# Core data memory is 64 KiB. One chunk's input and output buffers, the walkers' scratch
# regions and the stack must fit with room to spare.
DATA_MEMORY_BUDGET_BYTES = 56 * 1024
# Core program memory is 16 KiB; each chunk function is about 800 bytes. Leave room for the
# dispatcher and the runtime start-up code.
MAX_CHUNKS_PER_KERNEL = 10


class TestDesign:
    def test_every_walk_stays_inside_its_region(self):
        # chunk_sets asserts the extent; this makes the pools part of the contract.
        for dim in (2, 3):
            for scalar in (False, True):
                assert len(gen.chunk_sets(dim, scalar)) == gen.GROUPS * gen.WALKERS

    def test_pools_cover_counts_below_at_and_above_their_sizes_and_negative_strides(self):
        for dim, pool in ((2, gen.POOL_2D_VECTOR), (2, gen.POOL_2D_SCALAR),
                          (3, gen.POOL_3D_VECTOR), (3, gen.POOL_3D_SCALAR)):
            below = at = above = negative = 0
            for s in pool:
                if dim == 2:
                    m, size, stride, count = s
                    counts, sizes, strides = [count], [size], [stride, m]
                else:
                    m, size1, stride1, size2, stride2, count1, count2 = s
                    counts, sizes, strides = [count1, count2], [size1, size2], [stride1, stride2, m]
                for c, z in zip(counts, sizes):
                    below += c < z
                    at += c == z
                    above += c > z
                negative += any(v < 0 for v in strides)
            assert below and above and negative, (dim, below, at, above, negative)

    def test_scalar_pools_move_in_word_units_and_vector_pools_in_vector_units(self):
        for pool in (gen.POOL_2D_SCALAR, gen.POOL_3D_SCALAR):
            for s in pool:
                strides = [s[0], s[2]] + ([s[4]] if len(s) > 4 else [])
                assert all(v % 4 == 0 for v in strides), s
        for pool in (gen.POOL_2D_VECTOR, gen.POOL_3D_VECTOR):
            for s in pool:
                strides = [s[0], s[2]] + ([s[4]] if len(s) > 4 else [])
                assert all(v % 32 == 0 for v in strides), s

    def test_records_keep_vector_stores_32_byte_aligned(self):
        assert gen.REC_WORDS * 4 % 32 == 0
        assert gen.REGION_WORD % 8 == 0

    def test_source_regions_keep_their_promised_ranges(self):
        assert all(b < 8 for b in gen.region_nibble())
        small = gen.region_small()
        assert all(small[2 * i] < 100 and small[2 * i + 1] == 0 for i in range(len(small) // 2))
        b = gen.region_bf16()
        for i in range(0, len(b), 2):
            exp = ((b[i + 1] & 0x7F) << 1) | (b[i] >> 7)
            assert 110 <= exp <= 140

    def test_every_chunk_names_a_distinct_function_and_a_mnemonic(self):
        for kernel in gen.kernel_names():
            chunks = gen.chunks_of(kernel)
            assert len(chunks) <= MAX_CHUNKS_PER_KERNEL, kernel
            assert len({(f.name, d) for f, d in chunks}) == len(chunks)
            assert all(f.mnemonic for f, _ in chunks)

    def test_all_kernels_together_reach_each_family_in_both_dimensions_but_the_unreachable(self):
        pairs = [(f.name, d) for k in gen.kernel_names() for f, d in gen.chunks_of(k)]
        assert len(pairs) == len(set(pairs))
        assert {d for _, d in pairs} == {2, 3}
        # st.3d with a q register is the one pair Chess never emits (it picks vst.3d.128).
        missing = {(f.name, d) for k in gen.kernel_names() for f in gen.KERNELS[k][1] for d in (2, 3)} - set(pairs)
        assert missing == {("stq", 3)}


class TestStagedData:
    def test_every_kernel_fits_data_memory_and_the_shim_repeat_limit(self):
        for name in gen.kernel_names():
            chunks = gen.n_chunks(name)
            assert chunks <= 64, name
            scratch = gen.WALKERS * gen.REGION_BYTES
            total = (gen.NIN + gen.NOUT) * 4 + scratch
            assert total <= DATA_MEMORY_BUDGET_BYTES, (name, total)
            assert len(gen.input_words(name)) == chunks * gen.NIN
            assert gen.n_out(name) == chunks * gen.NOUT

    def test_inputs_match_the_committed_golden(self):
        for name in gen.kernel_names():
            path = gen.golden_path(name)
            assert os.path.exists(path), name
            with open(path) as f:
                golden = json.load(f)
            assert golden["inputs"] == [gen.input_words(name)], name
            assert len(golden["silicon"]) == gen.n_out(name), name

    def test_every_chunk_carries_its_own_number_and_its_tables(self):
        for name in gen.kernel_names():
            words = gen.input_words(name)
            for i in range(gen.n_chunks(name)):
                chunk = words[i * gen.NIN:(i + 1) * gen.NIN]
                assert chunk[0] == i
                fam, dim = gen.chunks_of(name)[i]
                table = []
                for s in gen.chunk_sets(dim, fam.scalar):
                    table += [v & gen.M32 for v in s]
                assert chunk[gen.HDR_WORDS:gen.HDR_WORDS + len(table)] == table

    def test_generated_sources_are_current(self):
        for name in gen.kernel_names():
            symbol, _ = gen.KERNELS[name]
            path = os.path.join(gen.VERIFY, name, f"{symbol}.cc")
            with open(path) as f:
                assert f.read() == gen.render_kernel_source(name), f"{name}: rerun gen_agu_2d3d.py generate"


class TestProof:
    def test_proof_reads_the_mnemonic_and_the_descriptor_operand(self, monkeypatch):
        listing = "\n".join([
            "       c: 59 b0 18 00  \tlda.2d\tr24, [p0], d0",
            "      4e: 99 4b 18 08  \tvst.2d.srs.s16.s32\tbmh2, s0, [p0], d3",
            "      60: 03 20 8b 13 02 80 12 80  \tvlda.3d.ups.s32.s16\tbml1, s0, [p3], d2_3d;\t\tmov\tdj0, r14",
            "      70: 19 08 00 00  \tvlda\twl0, [p0, #0x20]",
        ])
        monkeypatch.setattr(gen, "disassemble", lambda elf: (listing, []))
        found = gen.proof("unused.elf")
        assert set(found) == {
            ("lda.2d", "d0", "r"),
            ("vst.2d.srs.s16.s32", "d3", "bmh"),
            ("vlda.3d.ups.s32.s16", "d2_3d", "bml"),
        }
