# Mode-2 Capture Fixtures

Phase 0 of the A.2b mode-2 trace work: capture real-NPU mode-2 (Execution)
trace bytes from kernels with known ZOL structure, decode them, and use the
decoded LC frames to validate (or refute) the placeholder hypothesis that
was baked into `src/device/trace_unit/mod.rs::compute_lc_flag`.

> **Hypothesis under test**: the LC frame's bit-28 flag is set (=1) on the
> LC frame whose final atom corresponds to the iteration where the loop
> count register reaches 0 - i.e. the last iteration of the ZOL.

**Status (2026-04-30): hypothesis refuted.** Across 700 LC frames captured
from these fixtures (single-loop and software-outer/ZOL-inner nested
configurations), bit-28 is always 0. The LC frame is also emitted **once
per ZOL invocation** (not per iteration), with `count = trip count`.
See [`docs/superpowers/findings/2026-04-30-mode2-lc-flag-semantics.md`](../../docs/superpowers/findings/2026-04-30-mode2-lc-flag-semantics.md)
for the full empirical record and follow-on doc-search results.

The methodology below is preserved as the template for future Phase-0-style
hypothesis tests on undocumented HW behavior.

> **Important build note**: Peano-traced builds produce empty trace BOs
> (the trace shim DMA setup misbehaves under `--no-xchesscc --no-xbridge`).
> The working path captured by Phase 0 was **Chess-built kernel.o + chess
> aiecc with `--xchesscc --xbridge`**. The build script below uses Peano
> by default; for trace captures, build with Chess (compile kernel.cc via
> `xchesscc_wrapper aie2`, drop the `--no-` flags from aiecc).

## Why Peano kernels (not iron API)

The original Phase 0 plan called for hand-written kernels using mlir-aie's
iron Python API to get controlled ZOL structure. A calibration probe
(`tools/mode2_loop_probes/probes.cc`) confirmed that Peano emits ZOL
straightforwardly for ordinary C loops as long as the trip count cannot be
folded to a compile-time constant. So all fixtures here are short C kernels
that read their trip count(s) from the input buffer; Peano cannot inline-fold
them, and `llvm-objdump -d` lets us verify the emitted loop shape before
running on hardware.

## Fixture matrix

| Fixture | Trip count source | Expected ZOL shape | What it tests |
|---------|-------------------|--------------------|---------------|
| `runtime_loop/`   | `in[0]` (host-set N) | `add.nc lc, r1, #0x0` (LC = N) | bit-28 firing pattern across multiple N values; LC=0 edge case via N=1 |
| `fixed_loop_64/`  | hardcoded 64 | `add.nc lc, r0, #-0x1` with r0=0x10 (LC = 15, vectorized to 16 iters) | deterministic baseline; trace pattern for compile-time-known counts |
| `nested_loop/`    | `in[0]` (outer), `in[1]` (inner) | inner: `add.nc lc, r2, #0x0` (LC = inner); outer: `add r0,#-1; jnz` software | bit-28 firing once per ZOL completion vs. once per LC=0 boundary |

### Off-by-one caveat (to be resolved by HW capture)

Peano initializes LC differently for runtime vs fixed-known counts:
- **fixed_loop_64**: `add.nc lc, r0, #-0x1` -> LC = N-1. Per our existing
  hardware-loop semantics in `Context::check_hardware_loop`, body executes
  LC+1 = N times. This matches the C-source iteration count.
- **runtime_loop / nested_loop inner**: `add.nc lc, r1, #0x0` -> LC = N.
  Same hardware semantics imply body executes N+1 times. That looks like
  one too many, but: we may be misreading the disassembly's immediate
  encoding, or Peano may rely on the loop body's induction variable to
  cap iterations independently of LC.

The HW capture will resolve this directly: count atoms in the LC frames
of a single runtime_loop run with N=4 and compare against 4 vs 5.

## Building

```bash
source /home/triple/npu-work/toolchain-build/activate-npu-env.sh
./build_fixture.sh runtime_loop
./build_fixture.sh fixed_loop_64
./build_fixture.sh nested_loop
```

Each invocation produces:

```
<fixture>/build/
  kernel.o                    # disassemble with llvm-objdump -d to verify ZOL
  baseline/
    aie.xclbin                # untraced kernel (sanity-running only)
    insts.bin
  traced/
    aie.xclbin                # mode-2 trace-instrumented kernel
    insts.bin
```

The `traced/` xclbin is what you hand to `bridge-trace-runner` for HW
capture. The `baseline/` pair is for sanity-running the kernel without
trace overhead if the host-side data path is in doubt.

## HW capture procedure

Build the bridge runner once:

```bash
cd /home/triple/npu-work/xdna-emu/bridge-runner/build && cmake .. && make
```

For each fixture, generate one or more input `.bin` files (raw 256 bytes =
64 i32). The first one or two i32s are the trip counts (per the table
below); the rest is dummy data. Suggested layouts:

| Fixture | Inputs to capture | First i32(s) |
|---------|-------------------|--------------|
| runtime_loop  | N=1 / N=2 / N=4 / N=8 | `[N, ...]` |
| fixed_loop_64 | one capture (any data) | `[anything, ...]` |
| nested_loop   | (outer=4, inner=8) | `[4, 8, ...]` |

A small Python helper (`gen_inputs.py`, see "Next steps" below) is the
easiest way to produce these `.bin` files; for a single one-shot you can
just write 256 bytes by hand with `python3 -c 'import struct, sys;
sys.stdout.buffer.write(struct.pack("<64i", 4, *range(63)))'`.

Then run the trace:

```bash
./bridge-runner/build/bridge-trace-runner \
  --xclbin   tools/mode2_capture_fixtures/runtime_loop/build/traced/aie.xclbin \
  --instr    tools/mode2_capture_fixtures/runtime_loop/build/traced/insts.bin \
  --input    /tmp/n4.bin \
  --output   /tmp/out.bin \
  --trace-out /tmp/trace_runtime_n4.bin \
  --trace-size 65536
```

`trace_runtime_n4.bin` is the raw mode-2 byte stream from the trace unit
on tile (0,2), padded with sentinel bytes if the buffer is larger than the
captured stream.

## Decoding

```bash
python3 tools/parse-trace.py \
  --trace-mode inst_exec \
  --trace-bin /tmp/trace_runtime_n4.bin \
  --tile 0,2 \
  --out-cmd-stream /tmp/trace_runtime_n4_decoded.json
```

The decoded JSON is a per-tile list of frame records (PascalCase types:
`E_atom`, `N_atom`, `New_PC`, `LC`, `Repeat0`, `Repeat1`, `Start`, `Stop`,
`Filler0`, `Filler1`, `Sync`).

## Validating the hypothesis

For each captured trace, look at the LC frames in order:

1. Find every `LC` record and note its `count` and `flag` fields.
2. For each contiguous run of LC frames belonging to a single loop
   activation, check whether `flag == 1` only on the *last* LC frame of
   that activation (as the placeholder rule predicts), or whether the
   pattern is something else.

Hypothesis predictions per fixture:

- **runtime_loop, N=4**: one loop activation. Expect roughly 4-5 LC
  frames (one per atom batch; depends on RLE compression). Flag=1 on the
  last one only.
- **runtime_loop, N=1**: one loop activation, single iteration. Expect
  one LC frame with flag=1.
- **fixed_loop_64**: one loop activation with 16 vectorized iterations.
  Expect a small number of LC frames, last one flagged.
- **nested_loop, outer=4 inner=8**: four inner-loop activations, each
  with its own LC frame run. Expect flag=1 on the last LC of each of
  the four runs - i.e. 4 flagged frames total, all others unflagged.

If the captured pattern matches all four predictions, the placeholder rule
in `compute_lc_flag` is correct (and we can ship the kernel comment to that
effect). If anything diverges, the divergence shape tells us what the real
rule is. Either way, write up findings in
`docs/superpowers/findings/2026-04-29-mode2-lc-flag-semantics.md` and
update `compute_lc_flag` accordingly.

## Next steps after HW capture

1. Capture mode-2 traces for all the inputs above.
2. Decode each and inspect LC frames.
3. Document findings in the findings doc.
4. Update `src/device/trace_unit/mod.rs::compute_lc_flag` if the
   placeholder rule is wrong.
5. (Optional) Add a `gen_inputs.py` helper if more fixture matrices are
   needed.
6. (Phase 7) Open the upstream mlir-aie issue to ask whether anyone there
   knows the bit-28 semantics; share whatever findings we documented.
