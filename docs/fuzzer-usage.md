# Differential Fuzzer

`cargo run --release -- fuzz [OPTIONS]` generates random single-tile scalar
kernels, compiles them with Peano, runs each on the emulator and (with `--hw`)
the real NPU, and diffs the outputs. The NPU is ground truth; a divergence
indicates an emulator bug.

## Status (2026-06-28)

Revived and runs end-to-end. The differential loop (generate -> Peano-compile ->
EMU + NPU -> byte-diff) works and the in-sandbox HW path is proven.

**Primary EMU-vs-silicon correctness gate: the XRT bridge path** (`emu-bridge-test.sh`).
This is the full hardware-equivalent flow (`test.exe -> XRT -> plugin -> emulator`)
and is the canonical validation target per CLAUDE.md. The fuzzer's `--hw` flag
exercises this path and is where real divergence findings emerge.

**Secondary path: in-process `XclbinSuite`** (no XRT). This path has a known
unresolved issue (BUG-A): the in-process EMU path stalls shim DMA at `BdSetup`,
causing the emulator to return all-zeros for every non-trivial kernel. This makes
the in-process path unreliable as a differential oracle until BUG-A is resolved
(tracked in `docs/superpowers/findings/2026-05-30-fuzzer-revival-first-batch.md`;
`BdSetup` state is still present in `src/device/dma/channel.rs`). **BUG-A is not
yet fixed.**

## Options

| Flag | Meaning | Default |
|------|---------|---------|
| `--iterations N` | number of seeds to run | 0 |
| `--seed S` | base seed (deterministic; seeds S..S+N-1) | wall clock |
| `--hw` | also run on the real NPU and diff | off (EMU-only) |
| `--jobs J` | parallel compile/EMU jobs | nproc |
| `--max-cycles C` | per-case EMU runaway guard | 10_000_000 |
| `--trace-sweep` | multi-group trace capture + compare (SLOW) | off |
| `--trace-sweep-reps R` | NPU reps for trace determinism | 5 |
| `--verbose` | extra logging | off |

## Examples

    cargo run --release -- fuzz --iterations 100              # EMU-only batch
    cargo run --release -- fuzz --iterations 2000 --hw        # EMU+HW differential

A divergence's reproducer is its seed: `... fuzz --iterations 1 --seed <SEED> --hw`.

## Scope

Single-tile scalar ops only (arith/bitwise/shift/branch/hw-loop), Peano-compiled.
Vector ops and Chess are planned next phases. Test-case shrinking is out of scope
(the deterministic seed is the reproducer). Artifacts land under `build/fuzz/`.

## Throughput

Observed on the Phoenix devbox: ~3 seeds/sec with `--hw`, ~5 seeds/sec EMU-only.
These numbers will shift as the kernel generator grows in scope.
