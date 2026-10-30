"""Tests for gen_vldb_unpack.py: the designed inputs and the staged data contract."""

import json
import os

import gen_vldb_unpack as gen

# Core data memory is 64 KiB; keep the double-buffer-free working set well inside it.
DATA_MEMORY_BUDGET_BYTES = 48 * 1024


def probe():
    return gen.probe_blocks()


class TestProbeDesign:
    def test_every_byte_value_reaches_every_byte_position(self):
        seen = {p: set() for p in range(gen.BLOCK_BYTES)}
        for blk in probe():
            for p, v in enumerate(blk):
                seen[p].add(v)
        for p, values in seen.items():
            assert values == set(range(256)), f"byte position {p} misses {set(range(256)) - values}"

    def test_every_nibble_value_reaches_every_nibble_position(self):
        seen = {q: set() for q in range(2 * gen.BLOCK_BYTES)}
        for blk in probe():
            for p, v in enumerate(blk):
                seen[2 * p].add(v & 15)
                seen[2 * p + 1].add(v >> 4)
        for q, values in seen.items():
            assert values == set(range(16)), f"nibble position {q} misses {set(range(16)) - values}"

    def test_one_hot_blocks_name_their_position(self):
        blocks = gen.nibble_one_hot()
        assert len(blocks) == 64
        for q, blk in enumerate(blocks):
            nibbles = [n for v in blk for n in (v & 15, v >> 4)]
            assert [i for i, n in enumerate(nibbles) if n] == [q]
        for p, blk in enumerate(gen.byte_one_hot()):
            assert [i for i, v in enumerate(blk) if v] == [p]

    def test_blocks_fill_whole_chunks(self):
        assert len(probe()) % gen.BLOCKS_PER_CHUNK == 0
        assert len(gen.rand_blocks()) % gen.BLOCKS_PER_CHUNK == 0

    def test_held_out_set_is_not_the_probe(self):
        assert not set(map(tuple, gen.rand_blocks())) & set(map(tuple, probe()))


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

    def test_addr_records_keep_vector_stores_32_byte_aligned(self):
        # An 80-byte record misaligned every other 64-byte vector store on silicon.
        assert gen.REC_WORDS * 4 % 32 == 0

    def test_lane_vectors_are_64_byte_aligned(self):
        assert gen.WORDS_PER_VECTOR * 4 % 32 == 0
        assert gen.WORDS_PER_BLOCK % 8 == 0
        assert gen.CRT_WORDS_PER_BLOCK % 8 == 0
