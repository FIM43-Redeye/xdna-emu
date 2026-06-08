# Shuffle/permute: enum verified vs me_enums.h; per-byte routing is HW-gated

**Date:** 2026-06-08. **Status:** resolved (scope finding for the vector-compute audit, #103).

## Question

The vector-compute differential audit drives the **genuine aietools python
model** as oracle. Shuffle/permute (`SemanticOp::Shuffle`, the 48-mode
`shuffle_vectors` seam) has no function in that model (no `shuffle.py` in
`python_model/model/`). Can any aietools artifact serve as a Shuffle oracle?

## What exists in aietools

| Source | Contains | Usable as oracle? |
|--------|----------|-------------------|
| `python_model/model/*.py` | SRS/UPS/pack/mulmac/bf16 | ❌ no shuffle at all |
| `data/aie_ml/lib/me_enums.h` | 48 shuffle mode `#define`s (name + index 0..47), `eShuffleMode` enum, INTLV/DINTLV aliases | ✅ **mode enumeration only** -- no per-byte routing |
| `include/aie_api/.../aie2/shuffle_mode.hpp` | mode-selection semantics (zip/unzip) | ⚠️ semantics, not routing |

The **per-byte permutation** (`SHUFFLE_ROUTING[mode][byte]`, in
`crates/xdna-archspec/src/aie2/permute.rs`) is **hardware-probed** -- generated
by running identity patterns through VSHUFFLE on real AIE2 silicon
(`tests/shuffle-sweep/`). No aietools header or model encodes that routing, so
there is no no-HW oracle for the routing *values*. Silicon is the only ground
truth, which we already used.

## What was verified (no HW)

The one checkable thing the header provides is the **mode set + indexing**. A
point-in-time cross-check of all 48 `#define`s in `me_enums.h` against the
`ShuffleMode` enum in `permute.rs` (normalizing `_lo`/`_hi` vs `Lo`/`Hi`):

```
AMD modes parsed: 48, ours parsed: 48
ALL 48 SHUFFLE MODES MATCH (name + index)
```

So our shuffle mode enumeration is confirmed against AMD's authoritative header
-- no mis-indexed or mis-named mode. The enum already encodes the indices as
explicit `#[repr(u8)]` discriminants, so this freezes a verified mapping.

## Disposition

- **Mode enumeration**: VERIFIED vs me_enums.h (this check). Done.
- **Per-byte routing values**: no aietools oracle exists; remains
  **hardware-probed / silicon-gated** -- belongs to Half-B (the silicon
  `Verified{evidence}` pass), batched with the Phoenix-survival capture. The
  existing `tests/shuffle-sweep/` HW probe is the routing's ground truth.

Net: Shuffle is **not** a python-model differential class. Its enumeration is
verified statically; its routing fidelity is a HW-gated item, not an open
no-HW gap.
