---
name: 'bridge cache uses mtime-only check; stale trace-injected aie_arch.mlir survives in build_dir after trace-quarantine flip, causes IOMMU faults on next run'
description: ctrl_packet_reconfig_4x1_cores peano IOMMU faults at 0x0, 0x20, ..., 0xc0 (originally attributed to a peano backend bug emitting null-relative addresses) are actually caused by a stale build artifact. The bridge runner's compile cache checked `src_mlir -nt build_dir/aie_arch.mlir` to decide whether to refresh the canonical input MLIR. When a test gets trace-injected, build_dir/aie_arch.mlir gets a newer mtime than the original source (which rarely changes). If the test later moves into trace-quarantine.txt (or an in-session quarantine flip), `is_trace_quarantined` correctly skips trace prep, but the cache still sees build_dir/aie_arch.mlir as "fresher than source" and skips refresh. aiecc then compiles the previously-trace-injected MLIR, emitting a shim DMA BD targeting %arg2 (the trace buffer) which test.cpp doesn't bind. XRT substitutes IOVA=0; the BD's S2MM writes hit unmapped pages at 0x0..0xc0. Fix; replace mtime-only invalidation with content compare (cp+sed into tempfile, cmp against cached aie_arch.mlir, mv-on-diff). Supersedes 2026-05-13-ctrl-packet-reconfig-4x1-peano-iommu-fault.md.
type: finding
---

# Bridge cache mtime-only check leaks stale trace injection -- 2026-05-13

**Supersedes** [2026-05-13-ctrl-packet-reconfig-4x1-peano-iommu-fault.md](../../archive/findings/2026-05-13-ctrl-packet-reconfig-4x1-peano-iommu-fault.md). That doc's title and hypothesis ("peano emitted null-relative addresses") were wrong. The IOMMU faults at 0x0, 0x20, ..., 0xc0 turned out not to be a peano codegen bug at all -- they were successive trace DMA bursts to an unbound buffer, caused by a stale build artifact.

## TL;DR

`compile_one_compiler` in `scripts/emu-bridge-test.sh` used `[[ $src_mlir -nt $build_dir/aie_arch.mlir ]]` to decide when to refresh the canonical input MLIR copied into the per-compiler build dir. This breaks when:

1. An earlier run compiled the test with trace injection on (so `aie_traced.mlir` was the src_mlir, mtime "now"). `aie_arch.mlir` in build_dir gets the injected content.
2. The test later moves into `trace-quarantine.txt` (or quarantine logic changes). `is_trace_quarantined` correctly returns true, `TRACE_OK=false`, src_mlir falls back to `$src_dir/aie.mlir` (the original, mtime months old).
3. Cache check: `original_aie.mlir -nt build_dir/aie_arch.mlir`? No -- the build artifact is the recent one. `need_copy=false`. aiecc compiles the stale traced MLIR.

For `ctrl_packet_reconfig_4x1_cores` on peano this manifested as:

- `peano/aie_arch.mlir` (built 2026-05-10 23:45) carries `aie.trace.host_config buffer_size = 8192 arg_idx = 2` plus 8 trace.config / 38 npu.write32 / 1 address_patch ops. Runtime sequence has an extra `%arg2: memref<8192xi8>` arg.
- `chess/aie_arch.mlir` (built 2026-05-09 02:50) is clean; runtime sequence has only `%arg0, %arg1`.
- Source `test/npu-xrt/ctrl_packet_reconfig_4x1_cores/aie.mlir` (2025-12-22) is clean -- so the chess version matches it, the peano version doesn't.
- `is_trace_quarantined ctrl_packet_reconfig_4x1_cores` returns true today (the test is in the quarantine list), so today's run did skip trace prep -- but the stale peano build dir survived.

At runtime, aiecc emits a shim DMA BD for trace data (bd_id=15, buffer_length=2048, address_patch arg_idx=2). test.cpp only calls set_arg(0..5) with no buffer for arg2 -- XRT substitutes IOVA=0. The shim BD's S2MM writes proceed in 32-byte chunks at offsets 0, 0x20, 0x40, ..., 0xc0 of the unbound trace buffer, page-faulting at every chunk until the firmware stops.

So the 7 IOMMU faults aren't 7 buffer descriptors with zeroed bases; they're 7 successive trace bursts to an unmapped IOVA.

## Why chess passed

Chess's build dir was last refreshed before any trace injection ran on this test (or after a `--compile` flush). The artifact already matches the clean source, mtime cache happens to be coherent. Pure luck of timing.

## Why other variants passed

`ctrl_packet_reconfig`, `_1x4_cores`, `_elf` either weren't trace-injected during the window when this test's peano build dir caught a stale copy, or their stale copies happened to still match a then-current pipeline.

## The fix

Drop the mtime-only comparison. Generate the canonical expected content into a tempfile (cp + NPUDEVICE substitution), then `cmp -s` against the cached `aie_arch.mlir`. Refresh if FORCE_COMPILE, file missing, or content differs. Same pattern for the secondary `aie_arch_orig.mlir` copy path. Cost: one cp+sed+cmp per compile cycle -- microseconds vs aiecc's seconds.

```diff
- elif [[ "$src_mlir" -nt "$build_dir/aie_arch.mlir" ]]; then need_copy=true
+ # cp + sed source into tempfile, cmp against cached, refresh on diff
+ # (mtime-only invalidation missed stale traced artifacts; see this finding)
```

Verified: refreshing peano build dir for `ctrl_packet_reconfig_4x1_cores` via the fix's content-aware path produces a clean `aie_arch.mlir` (matches chess), the xclbin recompiles, HW PASSes, zero IOMMU faults.

After the fix, runtime sequence is `(%arg0, %arg1)`, external_buffers.json reports ctrlpkt at `xrt_id: 2` (matching chess; was `xrt_id: 3` when trace-injected with the extra %arg2 in slot 2).

## What this doesn't tell us

- Why the original quarantine for `ctrl_packet_reconfig_4x1_cores` was put in place (control-packet-reconfig-fabric deadlock, March 2026). The test passes today even with peano; the original deadlock may already have been resolved upstream. Worth running with the trace-quarantine entry removed to see if the test still passes when traced.
- Whether other tests have similar stale build_dir state. The fix is self-healing on next compile, so this clears up over time, but a `--compile` flush of all tests would catch any silent bit-rot now.

## Cross-references

- `scripts/emu-bridge-test.sh` -- the changed `compile_one_compiler` function.
- `../../archive/findings/2026-05-13-ctrl-packet-reconfig-4x1-peano-iommu-fault.md` -- the superseded "peano emits null-relative" finding.
- `scripts/trace-quarantine.txt` -- list of tests where trace prep is skipped; `ctrl_packet_reconfig_4x1_cores` was on it the whole time, so the cache leak was the only thing that mattered.
