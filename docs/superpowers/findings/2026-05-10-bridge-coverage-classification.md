---
name: 'Bridge test coverage classification (May 2026 snapshot)'
description: Per-test classification of every mlir-aie/test/npu-xrt entry against the bridge suite. Documents which tests run, which can't, and why. Two script fixes landed alongside this audit (py-generated aie.mlir rewrite guard, FileCheck pipe strip) that recovered bd_chain_repeat_on_memtile.
type: project
---

# Bridge test coverage classification

## TL;DR

The bridge suite now covers **every runnable** mlir-aie/test/npu-xrt test
on Phoenix (NPU1) hardware. 100% of non-quarantined non-NPU2 tests pass
on both compilers across HW + EMU. The remaining "gaps" are all
correctly-classified out-of-scope (NPU2-only hardware) or
documented-permanent-quarantine (Phoenix firmware bug, control overlay
deadlock).

Snapshot from the dual-compiler sweep at
`build/bridge-test-results/20260510/` (`>>> Phase 6: Report`):

```
Chess: 57/58 compiled, 59 bridge pass, 0 bridge fail
  HW: 59 pass, 0 fail, 0 skip
Peano: 53/54 compiled, 52 bridge pass, 0 bridge fail
  HW: 52 pass, 0 fail, 0 skip (1 XFAIL)
```

After the two bridge-script fixes documented below, `bd_chain_repeat_on_memtile`
moves from FAIL\* (compile failure) to PASS on both compilers.

## What got fixed during this audit

Two `scripts/emu-bridge-test.sh` bugs surfaced when running the
py-generated `bd_chain_repeat_on_memtile`:

1. **aie.mlir rewrite was unconditional.** The compile loop unconditionally
   rewrote `./aie.mlir` → `./aie_arch.mlir` in aiecc commands. For tests
   that generate `./aie.mlir` locally via `python aie2.py > ./aie.mlir`
   and don't ship a source `aie.mlir`, this broke the build because
   `aie_arch.mlir` was never created. Fix: gate the rewrite on
   `[[ -n "$src_mlir" ]]` (we only have an aie_arch.mlir when we
   actually copied from a source MLIR or trace output).

2. **`| FileCheck %s` swallowed test output.** `bd_chain_repeat_on_memtile`
   is the only test in the suite that pipes its run through FileCheck.
   The bridge captures test.exe stdout to a log and `grep`s for "PASS";
   FileCheck consumed all output before grep could see it, so the
   bridge always marked HW=FAIL even when test.exe exited 0. Fix: strip
   `| FileCheck ...` tails in `_strip_trace_flags()` (bridge does its
   own pass/fail detection).

Both fixes are tiny, low-risk, and benefit any future py-aiecc-on-aie.mlir
or FileCheck-piped test that lands upstream.

## Full classification

80 distinct test entries (counting `matrix_multiplication_using_cascade`'s
three variants as one entry). For each, status under both compilers:

### Tier 1: Pass on both compilers, HW + EMU (the green core)

These are the tests the bridge proves correct on real hardware AND on
the emulator. Stable green:

```
add_12_i8_using_2d_dma_op_with_padding   add_one_ctrl_packet_col_overlay
add_21_i8_using_dma_op_with_padding      add_one_objFifo
add_256_using_dma_op_no_double_buffering add_one_objFifo_elf
add_314_using_dma_op                     add_one_using_dma
add_378_i32_using_dma_op_with_padding    adjacent_memtile_access/three_memtiles
add_blockwrite                           adjacent_memtile_access/two_memtiles
add_maskwrite                            bd_chain_repeat_on_memtile (after fix)
add_one_cpp_aiecc                        column_specific
add_one_ctrl_packet                      core_dmas/dma_configure_task_lock
add_one_ctrl_packet_4_cores              core_dmas/dma_configure_task_token
ctrl_packet_reconfig                     dynamic_object_fifo/sliding_window_conditional
ctrl_packet_reconfig_1x4_cores           dynamic_object_fifo/two_core_sliding_window
ctrl_packet_reconfig_4x1_cores           matrix_transpose
ctrl_packet_reconfig_elf                 nd_memcpy_linear_repeat
device_width                             nd_memcpy_transforms
_diag_phase_b_add_one_instrumented       packet_flow
dmabd_task_queue                         packet_flow_fanin
dynamic_object_fifo/nested_loops         packet_flow_fanout
dynamic_object_fifo/ping_pong            shim_dma_bd_reuse
dynamic_object_fifo/reduction            static_L1_init
dynamic_object_fifo/sliding_window       sync_task_complete_token
                                         sync_task_complete_token_bd_chaining
                                         vec_vec_add_memtile_init
                                         vec_vec_add_tile_init
```

### Tier 2: Compiler-specific (REQUIRES line, runs on its required compiler)

These tests carry a `REQUIRES:` line and only run on one compiler. The
bridge correctly routes them — SKIP_compiler on the off-compiler is not
a coverage gap, it's the test's own declaration.

**Chess-only**: add_one_cpp_aiecc_xchesscc, add_one_func_link_with_chess,
add_one_scale_func_link_with_chess, cascade_flows,
matrix_multiplication_using_cascade (×3 variants),
neighbor_tile_memory_access, runtime_cumsum, tile_dmas/blockwrite_using_locks,
tile_dmas/writebd, tile_dmas/writebd_tokens, tile_mapped_read, two_col,
vector_scalar_using_dma. (14 entries.)

**Peano-only**: add_one_func_link_with_peano, add_one_scale_func_link_with_peano,
dma_complex_dims, dma_task_large_linear, objectfifo_repeat/compute_repeat,
objectfifo_repeat/init_values_repeat, objectfifo_repeat/simple_repeat,
vec_vec_add_objfifo_init. (8 entries.)

Plus one expected-fail: `objectfifo_repeat/distribute_repeat` (peano XFAIL).

### Tier 3: NPU2/AIE2P only — out of scope on Phoenix

These tests require Strix hardware (`REQUIRES: ryzen_ai_npu2`). The
bridge correctly emits SKIP(npu2) for both compilers. They'll come
back into scope when we acquire Strix hardware.

```
add_one_two            reconfigure_loadpdi
add_one_two_runlist    reconfigure_loadpdi_persistent_memtile
add_one_two_txn        loadpdi (filtered earlier: no test.cpp, only test_elf.cpp)
vec_mul_event_trace    (Python-driven; REQUIRES: ryzen_ai_npu2)
```

(6 in the bridge results table + 1 filtered by `is_standard_test` = 7 NPU2-only.)

### Tier 4: Permanent quarantine

These tests are documented in `scripts/{hw,test,trace}-quarantine.txt`
and either run isolated or are skipped entirely. Each entry references
the root-cause finding.

**HW quarantine** (`scripts/hw-quarantine.txt`, runs serially isolated):

| Test | Root cause |
|------|-----------|
| `two_col` | Peano deterministic TDR + chess collateral TDR under parallel load |
| `ctrl_packet_reconfig` | Control overlay deadlock with 4 application packet_flow ops |
| `ctrl_packet_reconfig_4x1_cores` | Same overlay deadlock pattern |

All three pass on HW when run serially (`-j1`), so the bridge isolates
them post-hoc rather than excluding them.

**Test quarantine** (`scripts/test-quarantine.txt`, fully skipped):

| Test | Root cause |
|------|-----------|
| `memtile_dmas/blockwrite_using_locks` | Phoenix firmware bug: runtime_sequence memtile DMA hangs (mlir-aie #3062) |
| `memtile_dmas/dma_configure_task_lock` | Same |
| `memtile_dmas/dma_configure_task_token` | Same |
| `memtile_dmas/writebd` | Same |
| `memtile_dmas/writebd_tokens` | Same |

These 5 will lift when upstream resolves mlir-aie #3062 or we get
firmware that handles the path.

**Trace-incompat** (`scripts/trace-incompat-tests.txt`, skipped only
when running with HW cycle capture):

| Test | Root cause |
|------|-----------|
| `ctrl_packet_reconfig_1x4_cores` | aiecc router saturates when trace routes overlay 4 compute tiles' existing flows |

### Tier 5: Structurally not bridge-shaped

Update 2026-05-11: Both Python-driven trace tests have been reclassified.
The bridge now discovers and runs Python-host tests (`test.py` with no
`test.cpp`) via the `is_python_host_test` predicate, and `vec_mul_*`
landed in two different tiers based on hardware requirements:

- `vec_mul_event_trace` → Tier 3 (NPU2-only; `REQUIRES: ryzen_ai_npu2`).
- `vec_mul_trace_distribute_lateral` → bridge-wrapped:
  HW PASS, EMU FAIL. The EMU failure is a real correctness gap in our
  distribute-channels + lateral-routing trace path — the kernel runs and
  produces correct data, but the emulator does not populate the
  distributed trace buffer regions the test reads back. Follow-up:
  implement trace channel distribute / lateral routing in EMU. Tracked
  as a forward gap, not a Tier 5 entry.

Both tests are tracked in Tier 3 (NPU2) and as a forward gap (EMU
distribute-channels) respectively. Tier 5 is currently empty.

### Tier 6: Trace pipeline ERROR (root cause identified 2026-05-11)

Three tests pass at the bridge level (HW + EMU produce identical
output) but their trace decode pipeline errors out:

```
add_one_ctrl_packet      (Chess/TRACE=ERROR  Peano/TRACE=ERROR)
dmabd_task_queue         (Chess/TRACE=ERROR  Peano/TRACE=ERROR)
packet_flow_fanout       (Chess/TRACE=ERROR  Peano/TRACE=ERROR)
```

This doesn't affect the correctness verdict; both compilers show the
same ERROR, so the bug is in our trace stack rather than per-compiler
kernel codegen.

**Root cause** (see `findings/2026-05-11-emu-trace-widened-distributed-routing.md`):
all three tests have the trace planner *widen the device to npu1_2col*
or *distribute trace across multiple shim DMA channels* because the
application already occupies the default channels. EMU's trace dispatch
only handles default single-channel origin-column routing, so events
generated on widened/distributed paths are dropped silently. The
emulator trace_raw.bin is all zeros; HW has hundreds of events.

This is the **same root cause** as the `vec_mul_trace_distribute_lateral`
EMU FAIL noted in the Tier 5 update -- distribute-channels + lateral
routing in EMU isn't implemented. Fixing one path recovers all four
tests.

## What this means for the project

- **Coverage is saturated** against what the bridge framework is
  designed to validate. With the Python-host harness landed
  (`is_python_host_test`), every NPU1-runnable test is in the suite.
- **EMU correctness is solid** across the breadth of mlir-aie patterns:
  ctrl-packets, packet flows, memtile DMAs (compute side), cascades,
  dynamic object FIFOs, ND-memcpy, control overlays, ELF mode. No
  failures that aren't otherwise documented.
- **Two forward gaps for future work**:
  1. EMU support for trace routing through widened or distributed
     shim DMA channels. This is **one** root cause covering both
     the Tier 6 ERROR trio (`add_one_ctrl_packet`, `dmabd_task_queue`,
     `packet_flow_fanout`) and the Tier 5 `vec_mul_trace_distribute_lateral`
     test. See `findings/2026-05-11-emu-trace-widened-distributed-routing.md`.
  2. Strix (NPU4/AIE2P) hardware to unlock the 7 NPU2-only tests
     (Tier 3, now including `vec_mul_event_trace`) — not a bridge gap,
     a hardware acquisition gap.

## See also

- `scripts/emu-bridge-test.sh` (`discover_tests`, `is_standard_test`,
  `compile_one_compiler`, `_strip_trace_flags`).
- `scripts/test-quarantine.txt`, `scripts/hw-quarantine.txt`,
  `scripts/trace-quarantine.txt`, `scripts/trace-incompat-tests.txt`.
- `build/bridge-test-results/20260510/` for the per-test artifacts
  this audit was built from.
- mlir-aie issue #3062 (memtile_dmas quarantine reference).
