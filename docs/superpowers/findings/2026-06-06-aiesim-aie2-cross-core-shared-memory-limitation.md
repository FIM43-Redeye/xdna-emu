# aiesimulator cannot model AIE2 compute-to-compute shared memory

**Date:** 2026-06-06
**Status:** CONFIRMED (three independent legs of evidence, including a live native
AMD aiesimulator repro)
**Scope:** AMD `aiesimulator` (AIE2/AIE-ML cluster model), not our device or bridge
**Supersedes:** the "matmul/buffer aiesim hang = make_npu1 shim/connections
rebuild" hypothesis (task #82), which was DISPROVEN here.

## TL;DR

AMD's `aiesimulator` for AIE2/AIE-ML **does not simulate compute-to-compute
neighbour shared-memory / lock handoff**. A producer core's lock release is never
made visible to the adjacent consumer core's neighbour-lock view, so any kernel
that hands data core-to-core through shared memory + locks deadlocks. This is a
limitation of AMD's simulator, **documented by AMD's own test suite as `XFAIL`**
and **reproduced live in AMD's native `--aiesim` flow** with no involvement from
our device file or bridge.

Practical consequence: aiesim **cannot be an oracle** for compute-to-compute
neighbour-objectfifo kernels on AIE2. The `matrix_multiplication_using_cascade`
`buffer` variant (the corpus's only such kernel) must be excluded from
aiesim-oracle validation. Real silicon runs it fine; AMD's simulator cannot.

## How we got here

The `matrix_multiplication_using_cascade` `buffer` variant was the last corpus
kernel that hung through the aiesim backend (`XDNA_BACKEND=aiesim`). The `cascade`
and `plain` variants of the same kernel pass; `add_256` and the rest of the
corpus pass. The three variants differ only in the inter-core handoff:

- `plain` / `cascade` — partial sums handed off via **cascade ports** (a dedicated
  core-to-core FIFO), or recomputed locally.
- `buffer` — partial sums handed off via **depth-1 objectfifos between adjacent
  compute tiles**, i.e. the consumer reads the producer's *west-neighbour* data
  memory under lock synchronisation. This is the only compute-to-compute
  neighbour-shared-memory kernel in the npu-xrt corpus.

## The wedge mechanism (our path)

Multi-tile probe (`XDNA_AIESIM_PROBE_TILE=all`) at the post-advance wedge, NPU1
device, `matmul/buffer`:

```
[probe timeout] core(0,2) status=0x00000201 pc=0x0057e stall[E=1] | L4=0 L5=1
[probe timeout] core(1,2) status=0x00000081 pc=0x006ce stall[W=1] | L4=1 L5=0
[probe timeout] core(2,2) status=0x00000081 pc=0x006ce stall[W=1] | L4=1 L5=0
[probe timeout] core(3,2) status=0x00000081 pc=0x0057e stall[W=1] |
```

(Lock-stall bit decode for AIE-ML core status: [7]=W [9]=E; `isMemEast`=own/internal,
`isMemWest`=col-1 neighbour. See `AIE2TargetModel::getMemWest`,
mlir-aie `lib/Dialect/AIE/IR/AIETargetModel.cpp`.)

Read it as a chain:

- **core0 (producer of of0)** finished its first production and **released
  of0_cons** — tile0 lock L5 reads **1** in the external probe. It now E-stalls
  (own lock) waiting for core1 to hand back of0_prod (L4=0).
- **core1 (consumer of of0)** must `AcquireGreaterEqual 1` on of0_cons, which lives
  in tile0 — its **west** neighbour. L5 is already 1, so the acquire should succeed
  instantly. Instead core1 **W-stalls forever**: through its west-neighbour lock
  window it never sees core0's release.
- core2/core3 back up identically. The final result is never produced; the shim
  output S2MM takes its initial burst (txns=8) then freezes (idle climbing across
  0->115us sim).

So: a value that is correct in tile0's own lock register is **stale/disconnected**
in core1's west-neighbour view. Same flavour as the cont.27 control-read aliasing
bug — correct in one view of the model, wrong in another.

## Three independent legs of proof

### 1. Empirical, our path — and the device is exonerated

- Flows are **byte-identical** between `buffer` and the passing `cascade` (every
  `aie.flow`: shim->memtile, memtile->each core, output back). Memtile structure
  identical. Objectfifo lock inits replay correctly.
- **Geometry-invariant.** Forcing the kernel one column over
  (`XDNA_AIESIM_FORCE_START_COL=0` vs `1`, flipping every core's column parity)
  produces a **byte-identical wedge**. No placement/parity dependence — consistent
  with mlir-aie's `AIE2TargetModel` which removed AIE1's checkerboard rule
  (west is always `(col-1,row)`).
- **Hangs on the unmodified vanilla VC2802 too.** Running `buffer` on
  `VC2802.plaintext.json` (the pristine Versal device, no make_npu1) in the
  bridge's Versal-overlay row-remap mode (compute rows 2->3) still deadlocks. The
  derived NPU1 device file is therefore not the cause.
- The bridge's `cdo_replay` passes **all** config writes through (no filtering),
  so any tile-isolation / lock-enable config in the CDO is replayed, not dropped.

### 2. Documentary — AMD's own test suite marks it XFAIL

`mlir-aie` ships matched compute-to-compute shared-memory unit tests:

| test | AIE1 (`chess_compiler_tests`) | AIE2 (`chess_compiler_tests_aie2`) |
|------|------|------|
| `04_shared_memory` (cross-core shared mem) | **passes** | **`XFAIL: *`** |
| `08_tile_locks` (same-tile locks) | passes | passes |
| `09_memtile_locks` | -- | passes |

The AIE2 `04_shared_memory` test runs through the native aiesim flow
(`aiecc.py --aiesim --xchesscc --xbridge` then `aie.mlir.prj/aiesim.sh`,
`aie.device(xcve2802)`) and is marked expected-to-fail on *all* configurations.
Same-tile and memtile locks pass on AIE2; only **cross-core** shared memory is
XFAIL. The XFAIL was added in the very first AIE2-sim bringup commit
(`a25e849463`, "Bringup AIE simulation tests for AIE and AIE2 using new flow",
Neuendorffer, 2023-05-21) — i.e. it never passed.

### 3. Live native repro — AMD's aiesimulator fails standalone

We compiled `chess_compiler_tests_aie2/04_shared_memory` (cores at tile(1,3) and
tile(1,4); core14 reads buf in tile13 across the neighbour boundary under a
cross-tile lock) through AMD's own native flow and ran AMD's `aiesimulator`
(`aie2simmsm`, AIE2 ISS r1p8) directly — **no bridge, no make_npu1, no remap**:

```
test start.
Acquire input buffer lock first.
Core [1, 3] AIE2 locks are: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Core [1, 4] AIE2 locks are: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Release input buffer lock.
Waiting to acquire output lock for read ...
ERROR: timeout hit!
ERROR After: mlir_aie_read_buffer_c(_xaie, 5) expected 175, but was 0!
Fail!
[INFO] : Simulation Finished, Sim result: 0  Total Simulation time 5399745 ps,
        Wall clock time 26.0868 s
```

Identical mechanism to the matmul wedge: the consumer waits on a lock the producer
"released", the release never becomes visible (both cores' lock arrays read
all-zero), the timeout hits, the cross-core output buffer is never written.

Full log: `build/experiments/aiesim-c2c-limitation/native-aiesim-04_shared_memory.log`.
Kernel: `build/experiments/aiesim-c2c-limitation/04_shared_memory-xcve2802.mlir`.

## Why it is the way it is (best read of the model)

The native run logs `ISS disables unused tiles` / `Running AIE2 MTMODEL Simulation`.
The AIE2 ISS models each core tile as an isolated unit; the neighbour memory/lock
**bridge between adjacent core tiles is not wired** in this path. Same-tile and
memtile locks work because no cross-tile state sharing is needed; cross-core
shared memory fails because the producer's lock state is never propagated into the
consumer tile's neighbour view. This is a missing-functionality gap, not a
localised glitch.

## Patchability (vs cont.27)

- **Same interposable target.** The cluster lib `libaie2_cluster_msm_v1_0_0.osci.so`
  is the same `.so` we LD_PRELOAD-interposed for the cont.27 ss_probe clone, so the
  source being locked does not make a patch impossible in principle.
- **But likely much harder than cont.27.** cont.27 was a localised aliasing bug
  fixed by cloning beats at a push site. This is missing cross-tile lock
  propagation — a patch would have to *synthesise* the neighbour-lock bridge (hook
  lock read/release, route neighbour lock state between tile models), i.e. a
  mini-implementation against an opaque binary, not a clone. A bounded feasibility
  probe (is there a clean lock-eval symbol to interpose, or is it buried in the ISS
  core model?) should precede any commitment.

## Recommendation

1. **Exclude compute-to-compute neighbour-objectfifo kernels from the aiesim
   oracle.** Mark `matmul/buffer` as a known aiesim-fidelity gap (XFAIL on AMD's
   own simulator). It is not a bug in our stack.
2. **Document-and-defer the patch** unless a feasibility probe finds a cont.27-shaped
   seam. We would be building functionality AMD left out, for a single corpus
   kernel class.
3. **CONFIRMED: our own Rust emulator runs `matmul/buffer` correctly.** Through the
   XRT-plugin path (`XDNA_EMU=1 ./test.exe -x aie2_buffer.xclbin -i insts2_buffer.txt`)
   the kernel completes (`halt_reason=completed cycles=155137`) and the host's
   matmul reference VERIFY passes (`PASS!`, Error count 0). So for this kernel class
   **xdna-emu is strictly more faithful than AMD's proprietary aiesimulator**:

   | | matmul/buffer (compute-to-compute objectfifo) |
   |---|---|
   | Real Phoenix silicon | passes |
   | xdna-emu (our interpreter) | **PASS** (completed + verified) |
   | AMD aiesimulator | **deadlock / FAIL** (XFAIL, native repro above) |

   Worth surfacing in the roadmap as a concrete accuracy advantage.

## Reproduction

Native AMD aiesim repro (sandbox OFF; links aietools + needs the AIE license):

```bash
cd <tmp>/aiesim-c2c
cp mlir-aie/test/unit_tests/chess_compiler_tests_aie2/04_shared_memory/{aie.mlir,test.cpp} .
TL=mlir-aie/install/runtime_lib/x86_64/test_lib
aiecc.py --aiesim --xchesscc --xbridge aie.mlir \
  -I"$TL/include" -L"$TL/lib" -ltest_lib -ltest_utils test.cpp
./aie.mlir.prj/aiesim.sh        # -> "ERROR: timeout hit!" ... "Fail!"
```

Note: the lit `%test_lib_flags` resolves to `-ltest_lib` (defines
`mlir_aie_init_device`); `-ltest_utils` alone leaves `ps.so` with an undefined
symbol and the sim aborts at host init before reaching the kernel.
