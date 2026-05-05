---
name: aiecc pooled chess compilation — design draft
description: Design draft for an aiecc batching mode that pools xchesscc invocations across multiple MLIR inputs so the ~21 s per-process Synopsys startup is amortized
status: draft (internal review)
issue_url: (not posted)
---

# Pooled chess compilation in aiecc — design draft

**Status**: internal draft, 2026-05-05. Not yet posted upstream. Pending review of dynamics (current upstream noise budget, whether to package this as an issue first or jump straight to a PR proposal).

## Problem

Each `xchesscc_wrapper aie2 -c ... -f input.ll -o out.o` invocation runs the full Synopsys chess pipeline:

```
chess-clang → noodle → chess-backend → bridge → darts
```

Five sequential proprietary binaries. On an empty `.cc`/`.ll` input the pipeline still takes **~21 s wallclock and ~22 s user CPU**, with peak resident set ~5.7 GB (chess-backend's "tale" architecture model dominates).

Today aiecc invokes xchesscc once per AIE core. Per-test wallclock is therefore roughly `Ncores × ~30 s` for chess-built tests. For a test suite of N kernels with average K cores, total chess wallclock is `N × K × ~30 s`. Even with `aiecc -j 0` (parallel core compilation) and `xargs -P` orchestrating multiple aiecc invocations, each xchesscc subprocess still pays its own startup; parallelism just trades wall-time for memory and CPU pressure.

For the xdna-emu bridge test set (~70 kernels, mostly 1–2 cores, with a long tail at 4 cores), a single dual-compiler bridge run currently spends roughly **30 minutes in the chess compile phase**. The non-chess phases (peano, MLIR lowering, xclbin assembly, host compile) are negligible by comparison.

## Empirical measurements

Single `xchesscc_wrapper aie2 -c -f tiny.ll -o tiny.o` on a near-empty LLVM IR module:

| Setup | Wall | User | Peak RSS |
|---|---|---|---|
| 1 file | 37 s | 31 s | ~5.7 GB |
| 2 files, no `+P` | 40 s | 33 s | ~5.7 GB |
| 4 files, `+P 4` | **38 s** | 121 s | ~5.7 GB |

Three takeaways:

1. **`+P N` parallelizes the entire pipeline across input files within one xchesscc process.** Wallclock for 4 files matches wallclock for 1.
2. **No-`+P` multi-file invocation is essentially serial** (40 s ≈ 2 × startup-amortized work). xchesscc loops the pipeline per file.
3. **The startup cost is paid once per process, not once per file.** 4 files in one process = 1 × startup; 4 files in 4 processes = 4 × startup (in parallel: bounded by memory, not CPU).

The implication: if all `chesslinked.ll` files across an entire compile job were submitted to a single xchesscc invocation with `+P $(nproc)`, total chess wallclock would drop from `N × K × 30 s` to roughly `ceil(N×K / nproc) × 30 s`. For our test set: 30 minutes → under 1 minute.

## What already exists

`aiecc --unified` (default off) compiles all cores in a single AIE device into one merged LLVM IR module, then runs xchesscc once on that. This solves the per-test multi-core case (e.g. cascade matmul) and is well-defined. We have not yet validated it on our test set; that is independent of this proposal.

`aiecc -j N` thread-parallelizes core compilation but each thread still spawns its own xchesscc subprocess. It improves wallclock for multi-core tests under a single aiecc invocation but does not amortize startup across cores or across tests.

Neither flag addresses the **cross-test** startup tax, which is the dominant cost for a typical test suite where most kernels are one or two cores.

## Proposed design

The minimal-surface change is to expose the chess phase as a separable step, then let any consumer build a pool driver on top.

Two new flags on aiecc:

### `aiecc --frontend-only <in.mlir> -o <work_dir>`

Runs all phases up to and including chess-llvm-link, then exits. Output: under `<work_dir>`,

- `chesslinked/<deviceName>_core_<col>_<row>.chesslinked.ll` — one per core
- `manifest.json` — describes per-core (col, row, chesslinked.ll path, expected .o output name) and per-device (deviceName, aieTarget, xclbin output name)

Skips the per-core xchesscc invocation, the per-core link/BCF, the xclbin assembly, the host build.

### `aiecc --use-objects=<dir> <in.mlir> -o <out.xclbin>`

Skips the chess pipeline entirely. Expects pre-compiled `.o` files at `<dir>/<deviceName>_core_<col>_<row>.o` matching the names that the chess phase would have produced. Continues with linking, control-packet generation, NPU instruction generation, xclbin assembly.

### Pool orchestrator (out of tree, in any consumer)

```
# Phase 1 (parallel, N processes): emit chesslinked.ll for every kernel
xargs -P$NPROC -I{} aiecc --frontend-only {} -o ${WORKDIRS_BY_INPUT[{}]}

# Phase 2 (single xchesscc): one pooled invocation
xchesscc_wrapper aie2 +w pool/work +P $NPROC -c -d +Wclang,-xir \
    -f a/chesslinked/foo_core_0_2.chesslinked.ll \
    -f b/chesslinked/bar_core_0_2.chesslinked.ll \
    -f b/chesslinked/bar_core_0_3.chesslinked.ll \
    ... \
    +o pool/objects/

# Phase 3 (parallel, N processes): finish each xclbin from the pooled .o files
xargs -P$NPROC -I{} aiecc --use-objects=pool/objects/ {} -o ${OUT_BY_INPUT[{}]}
```

### Why split rather than `aiecc --batch in1 in2 in3`

- **Smaller surface area**: one new boundary (the `chesslinked.ll` ↔ `.o` interface), not a new top-level driver.
- **Reusable**: lit, IRON, our bridge script, and ad-hoc users can each write their own orchestrator. Some may want different pool sizes, batch sizes, scheduling.
- **Easier to land**: adds two clearly-scoped flags rather than a parallel control-flow path through aiecc.cpp.
- **Composable with `--unified`**: a user can choose unified-per-test in the frontend phase if they want, and still pool across tests in the chess phase.

### Naming collisions

The pool gets `chesslinked.ll` files from many tests. Object filenames must be unique across the pool. Two viable approaches:

1. **Prefix-by-test**: orchestrator copies/symlinks `chesslinked.ll` into pool/ with a `<test_id>__<core_name>.chesslinked.ll` prefix; corresponding `.o` is `<test_id>__<core_name>.o`; orchestrator's `--use-objects` resolution understands the prefix.
2. **Pool-side per-test subdirs**: `pool/<test_id>/chesslinked/...` and `pool/<test_id>/objects/...`. Cleaner but requires `+o` to write to a different dir per file, which xchesscc may or may not support without per-input control. Needs a quick test.

Option 1 is implementable today with stock xchesscc behavior.

### IR symbol collisions

xchesscc compiles each `.ll` independently — they live in separate compilation units. Symbol collisions between them are not a concern as long as outputs go to separate `.o` files. (This is standard LLVM behavior; the only risk is if the chess pipeline somehow flattens all inputs into one context, which `+P N` empirically does not.)

## Implementation phasing

1. **Land `--frontend-only`**: small refactor to factor `compileAIEModule` such that everything from "Phase A frontend" through "chess-llvm-link" exits cleanly with a manifest written to disk.
2. **Land `--use-objects`**: companion path that picks up after the chess step and runs Phase B/C with on-disk `.o` files instead of invoking xchesscc.
3. **Reference orchestrator**: a small Python script (`tools/aiecc-pool.py` or similar) committed alongside as a reference implementation. Validates the design end-to-end and gives downstream users a starting template.
4. **Performance numbers**: measure on the mlir-aie test suite end-to-end. Show before/after wallclock for `ninja check-aie` or equivalent.

Each step is independently reviewable. Step 1 alone is a useful intermediate (lets tools inspect what aiecc would have compiled without paying chess cost).

## Risks and open questions

1. **`--unified` interaction**: should `--frontend-only --unified` emit one merged `chesslinked.ll` per device, or one per core? Probably one merged file is the right answer (matches `--unified` semantics) — the orchestrator can still pool merged files across tests. Needs a decision.
2. **`+o <dir>` filename mapping**: need to verify that xchesscc, given multiple `-f input.ll` flags and `+o outdir/`, produces predictable output filenames (probably `<input-stem>.o`). If not, we need a per-input rename step.
3. **Peano path**: this whole design targets the chess pipeline. Peano builds are already fast (no per-process startup tax). `--frontend-only` should still emit a peano-frontend artifact (the pre-llc IR) so the pool driver can also batch peano work if it wants, but the win is much smaller and may not be worth designing for in the first PR.
4. **Aiesim path**: `--aiesim` and `--unified` interaction is already documented as constrained. `--frontend-only` should refuse `--aiesim` (or run aiesim in `--use-objects` so it sees the final ELFs), TBD.
5. **Error reporting**: a single pooled xchesscc invocation that fails on one input will report a single failure for the whole batch. The orchestrator needs to be able to attribute failures back to specific tests. xchesscc's stderr should name the failing input file; needs verification.

## Posting strategy (open)

Two routes:

- **Issue first**: file an upstream issue describing the cost measurements + proposed split, get a thumbs-up on the API direction before implementing. Lowers risk of wasted work but requires another back-and-forth.
- **PR first**: implement steps 1+2 + reference orchestrator + benchmark numbers in a draft PR. Heavier upfront but unambiguous about feasibility.

Given current upstream noise level (mode-2 findings issue posted 2026-05-05 as #3047), waiting a beat before opening a second thread is probably the right call. Decision deferred until after we validate `--unified` on our test set and have concrete benchmark numbers from a local prototype.

## Cross-references

- mlir-aie #3047 — mode-2 findings issue (separate topic, but consumes the same upstream budget).
- `aiecc.cpp:387` (`unified` flag), `aiecc.cpp:2807` (`compileCoresUnified`), `aiecc.cpp:2640` (`compileCores` per-core path), `aiecc.cpp:5268` (dispatch).
- This emulator project's bridge test results timing (build/bridge-test-results/) — provides the local baseline we'd benchmark against.
