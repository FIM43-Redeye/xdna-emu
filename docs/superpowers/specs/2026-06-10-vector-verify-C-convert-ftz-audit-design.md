# Phase C: Convert FTZ-Path Audit on Silicon

**Campaign:** Vector-compute verification depth (A -> B -> C -> D), task #111.
**Goal:** Resolve empirically whether AIE2 (NPU1 Phoenix) silicon flushes bf16/f32
denormals to zero in the **standalone** convert datapaths the emulator currently
flushes -- the paths phase B never exercised.

This is the smallest-C cut (Maya's call): the convert FTZ audit only; op breadth
(shuffle/vsel/vcmp/vshift/min-max/reductions) folds into the phase-D fuzzer rather
than being hand-authored.

## Background: why this is still open after B

Phase B silicon-verified that f32->bf16 narrowing does NOT flush denormals -- but
it did so through the **fused store-from-accumulator** path
(`accum<accfloat>(v).to_vector<bfloat16>()` -> `memory/mod.rs:1110`), which calls
`f32_to_bf16` directly and never touches `fp32_flush_to_zero`. B cleared a path
that was never flushing.

The paths that DO flush all route through `VectorAlu::vector_convert`
(`src/interpreter/execute/vector_convert.rs`) and were blanket-FTZ'd in `ef77756`
as an unvalidated accuracy-era generalization. Reachability and observability:

| Path | Site | FTZ observable? | Discriminator | Reached by |
|------|------|-----------------|---------------|-----------|
| bf16 -> f32 expand | `:273` | **Yes** | f32 denormal bits (`0x0000_00xx`) vs `+/-0` | standalone VCONV expand |
| bf16 -> int32 vfloor | `:343` | **Yes, cleanly** | negative bf16 denormal: floor = **-1** vs **0** | standalone VFLOOR |
| f32 -> bf16 standalone | `:281` | Yes | f32 denormal -> tiny bf16 vs `+/-0` | standalone VCONV (B only cleared the fused-from-accum variant) |
| f32 -> int32 / uint32 | `:307,:315` | **No** | denormal truncates to 0 either way | n/a (analytic) |

## Approach

### Two silicon kernels (the observable, uncharacterized paths)

1. **bf16 -> f32 expand** -- bf16 denormal inputs; discriminator is the output f32
   bit pattern (a denormal `0x0000_00xx` under no-FTZ vs `0x0000_0000` under FTZ).
2. **bf16 -> int32 vfloor** -- bf16 denormal inputs including negatives;
   discriminator is `-1` (no-FTZ, `floor` of a tiny negative) vs `0` (FTZ to
   signed zero then floor). This is the cleanest brand-new datapoint.

### Inputs: directly generated, no new model corpus class

These paths need bf16 denormal **inputs**, which no existing golden class produces
(corpus classes are int-SRS/UPS/Pack, f32->bf16, and matmul). Rather than build new
model corpus classes (hand-authoring pain, and pointless here), we use phase B's
**silicon-golden tier directly**:

- Generate the denormal bf16 input set straight into the kernel spec (a dense
  exp=0 sweep over both signs, plus a handful of normal/zero anchors so a
  divergence is visibly localized to the denormal lanes).
- Bootstrap-run on HW to freeze silicon output as `EXP`
  (`tools/golden/silicon_edge/<kernel>.json`, via `capture_silicon_edge.py`).
- Verify emulator == silicon.

Silicon **is** the oracle, so no aietools model value is required. The capture
record's model-vs-silicon divergence column is optional for these kernels (it can
hold the emulator's pre-fix output for the A/B record, or be omitted); the binding
comparison is emulator-output vs frozen-silicon-`EXP`.

### Analytic dispositions (no HW run)

- **f32 -> int32 / uint32** (`:307,:315`): an f32 denormal lies in `(-1, 1)`, so
  truncation yields `0` whether or not it is first flushed. FTZ here is a provable
  no-op. Disposition: code comment + a unit test asserting denormal-in -> 0-out
  with and without the flush; no silicon kernel.
- **standalone f32 -> bf16 VCONV** (`:281`): attempt a third kernel ONLY if Chess
  emits the standalone op. If the compiler keeps fusing into a load/store (the A/B
  compile lottery), disposition by inference from B's store-path finding (silicon
  rounds f32->bf16 denormals; the same narrowing hardware backs both) rather than
  fighting codegen. Record which outcome occurred.

### The fix (if silicon says no-FTZ -- the likely outcome)

Remove `fp32_flush_to_zero` from the confirmed `vector_convert` branches, derived
from silicon and mirroring the already-clean store path. One regression test per
fixed branch (denormal-in -> rounded/expanded-out, not flushed). If instead
silicon DOES flush in a standalone path, keep the flush and document the
store-path-vs-standalone-path asymmetry as a verified fact.

## Verification discipline (carried from A/B)

- **Disassemble each kernel's `.lst`** and confirm the intended VCONV/VFLOOR opcode
  is actually emitted before trusting the capture. The compile lottery (Chess
  emitting a fused or alternate form) bit us in phase A (`VST.SRS` fused store) and
  shaped phase B. A kernel that doesn't emit the target instruction proves nothing
  about that instruction's FTZ behavior.
- **Deterministic HW capture**: two passes, identical output, per the phase-B
  protocol. NPU recovery only via `modprobe -r/modprobe amdxdna` if wedged.
- After any emulator change: `cargo build -p xdna-emu-ffi` before the HW run
  (stale `.so` -> phantom bugs), and `cargo test --lib` green.

## Files

- `tools/vector_kernel_specs.py` -- two new `KernelSpec`s (expand, vfloor) with
  directly-generated bf16 denormal inputs and `silicon=True`/`silicon_golden`.
- `tools/gen_vector_kernel.py` -- extend only if the direct-input path needs a new
  input-source mode (vs the corpus-slice path); prefer reusing the silicon
  bootstrap baking from B.
- `tools/golden/silicon_edge/<kernel>.json` -- two new HW-captured goldens.
- `src/interpreter/execute/vector_convert.rs` -- the fix (drop FTZ where silicon
  proves no-flush) + per-branch regression tests.
- `tools/capture_silicon_edge.py` -- reuse; extend only if a no-model-column record
  needs support.

## Acceptance

- Both audit kernels emit the intended opcode (`.lst`-verified), capture clean on
  silicon (2 passes identical), and end EMU == silicon.
- Each FTZ branch in `vector_convert.rs` is either silicon-confirmed-correct (FTZ
  kept, documented) or fixed (FTZ removed, regression test added).
- f32->int analytic no-op documented + unit-tested.
- `cargo test --lib` green; full vector regression green.
- Plan doc section C and known-fidelity-gaps updated; the migrated-from-B open
  question is closed.

## Out of scope (-> phase D)

Op breadth: shuffle routing, vsel, vcmp, vshift, vmin/vmax, reductions. These go to
the vector differential fuzzer (#112), not hand-authored kernels.
