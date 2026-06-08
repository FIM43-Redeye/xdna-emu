# Vector-compute golden data

`vector_ops.json` is the committed golden corpus for the AIE2 vector-compute
differential audit. Rust tests (`src/interpreter/execute/vector_validate.rs`)
load it at `cargo test` time and assert the emulator's vector semantics
bit-exact against it. **There is no aietools dependency at test time** -- the
JSON is the artifact.

## Provenance: the genuine aietools model is the oracle

`vector_ops.json` is produced by `tools/gen_vector_golden.py`, which drives the
**genuine aietools Python reference model** (`srs.srs_lane`, `ups.ups_lane`,
`helpers.trnc`) as the oracle. It is NOT a hand-port. An earlier version
re-implemented those lane functions in Python; that was circular (a misread of
the reference would corrupt the emulator and the golden identically, hiding the
bug). Driving the real model removes that blind spot -- the golden is
provenanced to the silicon reference.

Element-wise integer ops (`vadd`/`vsub`/`vmul`/`vmin`/`vmax`) use plain
wrap-around arithmetic, which *is* their spec (no rounding/saturation subtlety),
so those are computed directly, not from the model.

## Regenerating the golden (requires aietools)

Per the licensing policy (xdna-emu/CLAUDE.md), **aietools code stays
out-of-repo** -- only the derived golden JSON is committed, matching the aiesim
oracle posture. Regeneration therefore requires standing up the model
out-of-repo as a transient oracle artifact:

1. Copy the pure-Python model (no `.so` deps) out-of-repo, e.g. to
   `~/npu-work/experiments/vector-oracle/model/`:

   ```
   cp -r <aietools>/data/aie_ml/lib/python_model/model \
         ~/npu-work/experiments/vector-oracle/
   ```

2. Port it Python 2 -> Python 3 (the model ships as py2). `lib2to3` was removed
   in py3.13, so apply these mechanical, semantics-preserving fixes in place:

   - **Print statements** -> `print()` calls (the files only need to *parse*;
     no executed compute path prints). Handle line-start, post-`:`/`;` inline,
     and `print >>f, x` -> `print(x, file=f)` redirects.
   - **Integer division** `/` -> `//` via an AST transform on `BinOp` *and*
     `AugAssign` nodes. Every division site in the model is integer-context, so
     this preserves py2 int/int floor semantics under py3.
   - **`2L`/`0L` long literals** -> strip the `L` suffix.
   - **`itertools.izip`** -> `zip`; wrap `map(...)` in `list(...)` where the
     result is indexed or `len()`-ed (py3 `map` returns a lazy iterator).
   - **`long`**: shimmed at import time by the generator
     (`builtins.long = int`), not edited in the model source.

   Driver scripts that automate these fixes for this machine live alongside the
   working copy (`fix_py2_prints*.py`, `ast_intdiv.py`). They are not committed
   (out-of-repo, aietools-adjacent tooling).

3. Regenerate, pointing the generator at the ported model:

   ```
   VECTOR_ORACLE_MODEL=~/npu-work/experiments/vector-oracle/model \
       python3 tools/gen_vector_golden.py
   ```

   The generator fails loud if the oracle dir is absent.

## De-circularization checkpoint

The first regeneration through the genuine model reproduced the prior
hand-port golden **byte-for-byte** (same SHA-256) across all 32400 SRS + 2840
UPS cases. That simultaneously (a) proved the genuine-model oracle path works
end-to-end and (b) confirmed the retired hand-port was faithful for SRS/UPS --
so the de-circularization changed provenance without changing data.
