# Trace BO empty: address_patch arg_idx was XRT slot, not BO index

**Date:** 2026-05-05
**Status:** fixed and verified on real HW (add_one_using_dma, both compilers)

## Symptom

Across the full bridge test matrix (~100 tests, both compilers, both HW and
EMU), every `trace_raw.bin` came back as 1MB of zeros. Tests passed (data
correct), but trace BOs were never populated. Same on real silicon -- so it
was not an emulator bug.

## Root cause

The `arg_idx` field in `aiex.npu.address_patch` and `aie.trace.host_config`
is **not** the XRT kernel slot. It is the **BO argument index**: a 0-based
counter over only the BO entries in the kernel regmap, starting after
`opcode` / `instr_BO` / `ninstr` (offset 0x14).

Firmware path on patch:

```
patched_addr = regmap[0x14 + arg_idx * 8]
```

So with N existing memref args at BO indices 0..N-1 (XRT slots 3..3+N-1),
the **next free BO arg index** for a trace BO is N -- which corresponds to
**XRT slot 3 + N** on the host side.

`tools/mlir-trace-inject.py` was computing:

```python
chosen_arg_idx = 3 + max_existing_memref_args   # = 6 for a 3-memref test
```

and using that as both the XRT slot **and** the address_patch arg_idx. The
host-side slot was correct (`kernel.group_id(6)` binds bo_trace at XRT
slot 6), but the firmware then read `regmap[0x14 + 6 * 8] = regmap[0x44]` --
past the end of the populated regmap (only 4 BOs at offsets 0x14, 0x1C,
0x24, 0x2C). It got garbage, patched BD15 with that garbage, and the shim
DMA wrote trace data to nowhere useful. Real silicon and the emulator both
silently dropped the writes.

## Upstream pattern (vec_mul_event_trace, the working reference)

The default `aie.trace.host_config buffer_size = 8192` uses `arg_idx = 4`.
The host runtime in `mlir-aie/python/utils/hostruntime/hostruntime.py`
pads the args list with fillers so the trace BO always lands at **BO arg
index 4** (XRT slot 7):

```python
# args = [in1, in2, out]  (3 memrefs)
pad_until = trace_config.DEFAULT_TRACE_BUFFER_INDEX   # 4
while len(args) < pad_until: args.append(filler)      # → [in1, in2, out, filler]
args.append(trace_buff)                               # → [..., filler, trace_buff]
# trace_buff is now at index 4 in the args list = BO arg_idx 4 = XRT slot 7
```

So for upstream, `arg_idx = 4` always lines up because of the host-side
padding. Our injector skips the padding and places the trace BO directly
after the existing memrefs, so we need `arg_idx = N` (where N = number of
memref args), not `4`.

## Fix

`tools/mlir-trace-inject.py`:

```python
chosen_arg_idx = max_existing_memref_args
```

The trace_config schema records:

- `buffer.kernel_arg_slot = 3 + chosen_arg_idx`  (XRT slot for host code,
  consumed by `cpp_trace_patch.py` → `kernel.group_id(N)`)
- `diagnostics.expected_address_patch_arg_idx = chosen_arg_idx` (BO
  arg index for sanity checks against the lowered MLIR)

Verified with a direct invocation:

```
$ python3 tools/trace-prepare.py mlir-aie/test/npu-xrt/add_one_using_dma -o /tmp/...
...
kernel_arg_slot: 6
expected_address_patch_arg_idx: 3
$ grep address_patch /tmp/.../aie_traced.mlir
aiex.npu.address_patch {addr = 119268, arg_idx = 3, arg_plus = 0}
```

## Cache invalidation

The bridge test caches `build_dir/aie_arch.mlir` per test-and-compiler.
Cache invalidates only on `src_mlir -nt aie_arch.mlir` -- changing the
injector alone does not bump it. Force a rebuild with `--compile`, by
deleting cached `aie_arch.mlir` files, or by `touch`ing the source aie.mlir
files in `mlir-aie/test/npu-xrt/`.

## HW verification (2026-05-05)

Re-ran the bridge test on `add_one_using_dma` with `--trace --compile`:

```
$ ./scripts/emu-bridge-test.sh --trace --compile add_one_using_dma
...
TRACE PREP add_one_using_dma: OK
COMPILE add_one_using_dma (peano): OK
COMPILE add_one_using_dma (chess): OK
HW add_one_using_dma (chess): PASS  @461s
HW add_one_using_dma (peano): PASS  @461s
BRIDGE add_one_using_dma (chess): PASS
BRIDGE add_one_using_dma (peano): PASS
```

`trace_raw.bin` populated for all four compiler/side pairs (previously all
zero):

| Variant | nonzero words |
|---|---|
| chess.hw | 32 |
| chess.emu | 16 |
| peano.hw | 32 |
| peano.emu | 16 |

The trace decoder reports valid events and the comparator runs end-to-end.
Remaining HW/EMU divergences (EMU has fewer events; column-offset
mismatches in the comparator output) are unrelated to this bug -- they are
the existing `#321` EMU broadcast/trace-stop timing issue and the known
cosmetic col=0 vs col=start_col reporting offset.

**Gotcha:** `--no-trace` is the bridge-test default. Without `--trace` the
script never even invokes `tools/trace-prepare.py`, and the test "passes"
while leaving the cached (buggy) `aie_traced.mlir` in place. First run of
this verification used the default and produced misleading "still zero"
output. Don't rely on the absence of an explicit `--no-trace` flag.

## Takeaway

The MLIR `arg_idx` naming is misleading: it conflates "argument index"
with "kernel slot" but really means "0-based index into the BO portion of
the regmap." Both the upstream `aie.trace.host_config` op definition
("XRT argument index for trace buffer") and the various host-side helpers
that pass `arg_idx=4` reinforce the wrong intuition. Anyone touching this
plumbing should sanity-check that the address_patch arg_idx, the host BO
binding, and the kernel signature line up under the BO-index reading, not
the XRT-slot reading.
