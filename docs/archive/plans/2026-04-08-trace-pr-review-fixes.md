# Trace PR #3001 Review Feedback Implementation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address reviewer feedback on PR #3001 (lateral routing + channel distribution) -- fix argIdx bounds bug, add shim channel conflict detection with lateral fallback, and add multi-column >2 traces test coverage.

**Architecture:** Three independent fixes to `AIEInsertTraceFlows.cpp` on the `trace-routing-v3` branch. Each adds a failing test first, then the minimal implementation. Channel conflict detection scans existing `aie.flow`/`aie.packet_flow`/`ShimDMAAllocationOp` for S2MM claims, then falls back to lateral routing or single-channel mode. The argIdx fix validates bounds before using `primaryArgIdx+1` for the secondary channel.

**Tech Stack:** C++ (MLIR pass), MLIR FileCheck tests, tablegen (AIEPasses.td)

**Working branch:** `trace-routing-v3` (PR head)

---

### Task 1: Fix argIdx+1 bounds safety (Copilot review item #3)

The secondary channel in `buildChannelDescriptors` uses `primaryArgIdx + 1` unconditionally. When `host_config arg_idx=-1`, the primary resolves to `args.size()-1`, making `+1 == args.size()` (out of bounds). The fix: check bounds and fall back to single-channel when the second arg doesn't exist.

**Files:**
- Modify: `test/dialect/AIE/trace/test_insert_trace_flows_distribute.mlir` (add test case)
- Modify: `lib/Dialect/AIE/Transforms/AIEInsertTraceFlows.cpp` (fix `buildChannelDescriptors`)

- [ ] **Step 1: Write the failing test**

Add a new test module to `test_insert_trace_flows_distribute.mlir` that uses `arg_idx = -1` with distribute enabled. The pass should fall back to single-channel (both traces on same DMA channel) rather than generating an out-of-bounds arg_idx.

```mlir
// -----

// Test: distribute with arg_idx=-1 falls back to single channel
// (arg_idx=-1 resolves to last arg; +1 would be out of bounds)
// CHECK-LABEL: module @distribute_auto_argidx_fallback
module @distribute_auto_argidx_fallback {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)
    %tile00 = aie.tile(0, 0)

    aie.trace @trace_a(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.trace @trace_b(%tile03) {
      aie.trace.packet id=2 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 8192 arg_idx = -1
      aie.trace.start_config @trace_a
      aie.trace.start_config @trace_b
    }

    // With arg_idx=-1, distribute falls back to single channel
    // Both traces use the same DMA channel (no distribute)
    // CHECK: aie.packet_dest<%{{.*}}, DMA : 1>
    // CHECK: aie.packet_dest<%{{.*}}, DMA : 1>
    // Only one writebd and one address_patch (single channel)
    // CHECK-COUNT-1: aiex.npu.writebd
    // CHECK-COUNT-1: aiex.npu.address_patch
  }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `nice -n 19 cmake --build build --target check-aie-mlir 2>&1 | tail -30`

Or more targeted: `nice -n 19 build/bin/aie-opt test/dialect/AIE/trace/test_insert_trace_flows_distribute.mlir --split-input-file -aie-insert-trace-flows="distribute-channels=true" 2>&1`

Expected: The test either crashes (OOB access) or produces `arg_idx = 1` (out of bounds for a 1-arg runtime_sequence), failing the CHECK pattern.

- [ ] **Step 3: Implement the bounds check**

In `AIEInsertTraceFlows.cpp`, modify `buildChannelDescriptors` to accept the number of runtime_sequence arguments and validate before creating the secondary channel:

```cpp
  /// Build channel descriptors. Always includes the primary channel.
  /// Adds a secondary channel when distribute-channels is enabled, there
  /// are multiple traces, and a second arg_idx is available.
  std::vector<ChannelDescriptor>
  buildChannelDescriptors(size_t numTraces, int primaryChannel, int primaryBdId,
                          int primaryArgIdx, int numRuntimeArgs) {
    std::vector<ChannelDescriptor> chans;
    chans.push_back({primaryChannel, primaryBdId, primaryArgIdx});
    if (clDistributeChannels && numTraces > 1 && primaryBdId > 0 &&
        primaryArgIdx + 1 < numRuntimeArgs) {
      int ch2 = (primaryChannel == 1) ? 0 : 1;
      chans.push_back({ch2, primaryBdId - 1, primaryArgIdx + 1});
    }
    return chans;
  }
```

Update the call site (around line 327) to pass the arg count:

```cpp
    int numRuntimeArgs = runtimeSeq.getBody().getArguments().size();

    for (auto &[col, shimInfo] : shimInfos) {
      shimInfo.channels = buildChannelDescriptors(
          shimInfo.traceSources.size(), shimInfo.channel, shimInfo.bdId,
          shimInfo.argIdx, numRuntimeArgs);
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `nice -n 19 build/bin/aie-opt test/dialect/AIE/trace/test_insert_trace_flows_distribute.mlir --split-input-file -aie-insert-trace-flows="distribute-channels=true" 2>&1`

Expected: All CHECK patterns pass. The `arg_idx=-1` test case shows single-channel fallback.

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `nice -n 19 cmake --build build --target check-aie-mlir 2>&1 | tail -20`

Expected: All existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add test/dialect/AIE/trace/test_insert_trace_flows_distribute.mlir \
        lib/Dialect/AIE/Transforms/AIEInsertTraceFlows.cpp
git commit -m "fix(trace): bounds-check argIdx before distribute secondary channel

When host_config uses arg_idx=-1 (auto-resolve to last runtime_sequence
arg), the secondary distribute channel would use argIdx+1 which is out
of bounds. Fall back to single-channel mode when the second arg index
is not available.

Addresses Copilot review feedback on PR #3001."
```

---

### Task 2: Shim S2MM channel conflict detection with lateral fallback (Jack items #2 + #4)

Before building channel descriptors, scan the IR for existing S2MM channel claims on each target shim tile. If channels are partially or fully occupied, adjust: collapse to single-channel if 1 free, redirect to lateral if 0 free (or error if lateral is unavailable). This also covers Jack's item #4 (forced column with partial use).

**Files:**
- Create: `test/dialect/AIE/trace/test_insert_trace_flows_conflicts.mlir` (channel conflict tests)
- Modify: `test/dialect/AIE/trace/test_insert_trace_flows_lateral.mlir` (lateral fallback test)
- Modify: `test/dialect/AIE/trace/test_insert_trace_flows_verify.mlir` (error case)
- Modify: `lib/Dialect/AIE/Transforms/AIEInsertTraceFlows.cpp` (conflict detection + fallback)

- [ ] **Step 1: Create the conflict test file with both scenarios**

Create `test/dialect/AIE/trace/test_insert_trace_flows_conflicts.mlir`. Two RUN lines, same CHECK prefix -- output is identical with and without distribute for these scenarios, so sharing the prefix implicitly verifies both modes.

```mlir
//===- test_insert_trace_flows_conflicts.mlir -----------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Both RUN lines share one CHECK prefix -- conflict detection behavior is
// identical with and without distribute-channels, so this implicitly verifies
// both modes produce the same result.
// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows | FileCheck %s
// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows="distribute-channels=true" | FileCheck %s

// -----

// Test: Default trace channel (DMA:1) is already claimed by an existing flow.
// The pass must detect the conflict and switch the primary to DMA:0.
// This is the critical path -- it tests that the PRIMARY channel gets
// reassigned, not just that distribute collapses.
// CHECK-LABEL: module @default_channel_conflict
module @default_channel_conflict {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)

    // Existing flow claims S2MM channel 1 (the trace default) on shim
    aie.flow(%tile02, DMA : 0, %tile00, DMA : 1)

    aie.trace @trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 8192
      aie.trace.start_config @trace
    }

    // Trace switches to DMA:0 (channel 1 is occupied)
    // CHECK: aie.packet_dest<%{{.*}}, DMA : 0>
  }
}

// -----

// Test: S2MM channel 0 is claimed. With distribute, the secondary channel
// (DMA:0) is unavailable, so distribute collapses to single-channel on
// DMA:1. Without distribute, DMA:1 is already the default -- same output.
// CHECK-LABEL: module @secondary_channel_used
module @secondary_channel_used {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)
    %tile00 = aie.tile(0, 0)

    // Existing flow claims S2MM channel 0 (distribute secondary) on shim
    aie.flow(%tile02, DMA : 0, %tile00, DMA : 0)

    aie.trace @trace_a(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.trace @trace_b(%tile03) {
      aie.trace.packet id=2 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 8192
      aie.trace.start_config @trace_a
      aie.trace.start_config @trace_b
    }

    // Both traces use channel 1 (channel 0 occupied, no distribute possible)
    // CHECK: aie.packet_dest<%{{.*}}, DMA : 1>
    // CHECK: aie.packet_dest<%{{.*}}, DMA : 1>
    // Single channel means single BD
    // CHECK-COUNT-1: aiex.npu.writebd
  }
}
```

- [ ] **Step 3: Write the test for "both channels used, lateral fallback"**

Add to existing `test_insert_trace_flows_lateral.mlir`. Both S2MM channels on column 0's shim are claimed. With lateral routing enabled, the pass should redirect to the spare column.

```mlir
// -----

// Test: Both S2MM channels on target shim are used -- falls back to lateral.
// CHECK-LABEL: module @lateral_fallback_full_shim
module @lateral_fallback_full_shim {
  aie.device(npu1_2col) {
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)

    %core = aie.core(%tile02) { aie.end }

    // Both S2MM channels claimed on column 0 shim
    aie.flow(%tile02, DMA : 0, %tile00, DMA : 0)
    aie.flow(%tile02, DMA : 1, %tile00, DMA : 1)

    aie.trace @trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 8192
      aie.trace.start_config @trace
    }

    // Trace redirects to column 1 (spare, both channels free)
    // CHECK: aiex.npu.writebd {{{.*}}column = 1
    // CHECK: aie.packet_dest<%{{.*}}1_0{{.*}}, DMA : 1>
  }
}
```

- [ ] **Step 4: Write the test for "both channels used, no lateral, error"**

Add to existing `test_insert_trace_flows_verify.mlir`. Both channels used, lateral routing is off (default), no spare available. Should emit a diagnostic.

```mlir
// -----

// Test: Both S2MM channels used, no lateral routing -- error
module @shim_full_no_lateral {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)

    aie.flow(%tile02, DMA : 0, %tile00, DMA : 0)
    aie.flow(%tile02, DMA : 1, %tile00, DMA : 1)

    aie.trace @trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence(%arg0: memref<16xi32>) {
      // expected-error@below {{no S2MM channels available on shim tile at column 0}}
      aie.trace.host_config buffer_size = 8192
      aie.trace.start_config @trace
    }
  }
}
```

- [ ] **Step 5: Run the tests to verify they fail**

Run each new/modified test file against the current (unmodified) pass to confirm failures:

```bash
nice -n 19 build/bin/aie-opt test/dialect/AIE/trace/test_insert_trace_flows_conflicts.mlir \
  --split-input-file -aie-insert-trace-flows 2>&1
nice -n 19 build/bin/aie-opt test/dialect/AIE/trace/test_insert_trace_flows_lateral.mlir \
  --split-input-file -aie-insert-trace-flows="lateral-routing=true" 2>&1
```

Expected: The new test cases fail (no conflict detection exists yet).

- [ ] **Step 6: Implement S2MM channel usage scanning**

In `AIEInsertTraceFlows.cpp`, add a private method to scan existing channel claims. Place it with the other private methods (after `findNearestSpareColumn`):

```cpp
  /// Scan the device for existing S2MM channel claims on shim tiles.
  /// Checks aie.flow destinations, aie.packet_flow destinations, and
  /// ShimDMAAllocationOp declarations.
  std::map<int, std::set<int>>
  scanUsedS2MMChannels(DeviceOp device) {
    std::map<int, std::set<int>> used; // shimCol -> set of used S2MM channels

    device.walk([&](FlowOp flow) {
      auto destTile = cast<TileOp>(flow.getDest().getDefiningOp());
      if (destTile.isShimTile() && flow.getDestBundle() == WireBundle::DMA) {
        used[destTile.getCol()].insert(flow.getDestChannel());
      }
    });

    device.walk([&](PacketFlowOp pktFlow) {
      for (auto &op : pktFlow.getPorts().front()) {
        if (auto dest = dyn_cast<PacketDestOp>(op)) {
          auto destTile = cast<TileOp>(dest.getDest().getDefiningOp());
          if (destTile.isShimTile() && dest.getBundle() == WireBundle::DMA) {
            used[destTile.getCol()].insert(dest.getChannel());
          }
        }
      }
    });

    device.walk([&](ShimDMAAllocationOp alloc) {
      if (alloc.getChannelDir() == DMAChannelDir::S2MM) {
        auto tile = alloc.getTileOp();
        used[tile.getCol()].insert(alloc.getChannelIndex());
      }
    });

    return used;
  }
```

- [ ] **Step 7: Implement conflict resolution logic**

Insert the conflict resolution between lateral routing (Phase 2b-lateral) and channel descriptor building. This goes right after the lateral routing block and before the `buildChannelDescriptors` loop (around line 320):

```cpp
    // Phase 2b-conflict: Check S2MM channel availability on target shims.
    // If channels are occupied, adjust: use free channel, redirect lateral,
    // or error.
    auto usedChannels = scanUsedS2MMChannels(device);

    for (auto &[col, shimInfo] : shimInfos) {
      int shimCol = shimInfo.shimTile.getCol();
      auto usedIt = usedChannels.find(shimCol);
      if (usedIt == usedChannels.end())
        continue; // No conflicts on this shim

      const auto &usedSet = usedIt->second;
      int freeCount = 2 - static_cast<int>(usedSet.size());

      if (freeCount <= 0) {
        // No channels free -- try lateral redirect
        if (clLateralRouting) {
          std::set<int> activeColumns;
          device.walk([&](CoreOp core) {
            auto coreTile = cast<TileOp>(core.getTile().getDefiningOp());
            activeColumns.insert(coreTile.getCol());
          });
          // Also treat columns with full shims as "active" for spare search
          for (auto &[fullCol, channels] : usedChannels) {
            if (static_cast<int>(channels.size()) >= 2)
              activeColumns.insert(fullCol);
          }
          int spare =
              findNearestSpareColumn(shimCol, activeColumns, targetModel);
          if (spare >= 0) {
            shimInfo.shimTile = getOrCreateShim(device, builder, spare);
            // Reset channel to default since spare shim is clean
            shimInfo.channel = clShimChannel;
            continue;
          }
        }
        // No lateral available -- emit error
        device.emitError()
            << "no S2MM channels available on shim tile at column " << shimCol
            << " (both channels in use by existing flows); enable "
               "lateral-routing to redirect to a spare column";
        return signalPassFailure();
      }

      if (freeCount == 1) {
        // One channel free -- use it, disable distribute for this shim
        int freeChannel = -1;
        for (int ch = 0; ch < 2; ch++) {
          if (usedSet.count(ch) == 0) {
            freeChannel = ch;
            break;
          }
        }
        shimInfo.channel = freeChannel;
        // Note: buildChannelDescriptors will only create 1 descriptor
        // when the primary channel matches the only free channel,
        // because attempting to use the other channel would conflict.
      }
    }
```

For the single-free-channel case, also update `buildChannelDescriptors` to accept a set of available channels:

```cpp
  std::vector<ChannelDescriptor>
  buildChannelDescriptors(size_t numTraces, int primaryChannel, int primaryBdId,
                          int primaryArgIdx, int numRuntimeArgs,
                          const std::set<int> &availableChannels) {
    std::vector<ChannelDescriptor> chans;
    chans.push_back({primaryChannel, primaryBdId, primaryArgIdx});
    if (clDistributeChannels && numTraces > 1 && primaryBdId > 0 &&
        primaryArgIdx + 1 < numRuntimeArgs) {
      int ch2 = (primaryChannel == 1) ? 0 : 1;
      // Only add secondary channel if it's available
      if (availableChannels.count(ch2)) {
        chans.push_back({ch2, primaryBdId - 1, primaryArgIdx + 1});
      }
    }
    return chans;
  }
```

Update the call site to pass available channels:

```cpp
    int numRuntimeArgs = runtimeSeq.getBody().getArguments().size();

    for (auto &[col, shimInfo] : shimInfos) {
      int shimCol = shimInfo.shimTile.getCol();
      std::set<int> available = {0, 1};
      auto usedIt = usedChannels.find(shimCol);
      if (usedIt != usedChannels.end()) {
        for (int ch : usedIt->second)
          available.erase(ch);
      }
      shimInfo.channels = buildChannelDescriptors(
          shimInfo.traceSources.size(), shimInfo.channel, shimInfo.bdId,
          shimInfo.argIdx, numRuntimeArgs, available);
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `nice -n 19 build/bin/aie-opt test/dialect/AIE/trace/test_insert_trace_flows_conflicts.mlir --split-input-file -aie-insert-trace-flows 2>&1`

Run: `nice -n 19 build/bin/aie-opt test/dialect/AIE/trace/test_insert_trace_flows_lateral.mlir --split-input-file -aie-insert-trace-flows="lateral-routing=true" 2>&1`

Run: `nice -n 19 build/bin/aie-opt test/dialect/AIE/trace/test_insert_trace_flows_verify.mlir -split-input-file -verify-diagnostics -aie-insert-trace-flows 2>&1`

Expected: All new test cases pass.

- [ ] **Step 9: Run the full test suite**

Run: `nice -n 19 cmake --build build --target check-aie-mlir 2>&1 | tail -20`

Expected: All tests pass.

- [ ] **Step 10: Commit**

```bash
git add lib/Dialect/AIE/Transforms/AIEInsertTraceFlows.cpp \
        test/dialect/AIE/trace/test_insert_trace_flows_conflicts.mlir \
        test/dialect/AIE/trace/test_insert_trace_flows_lateral.mlir \
        test/dialect/AIE/trace/test_insert_trace_flows_verify.mlir
git commit -m "feat(trace): detect shim S2MM channel conflicts and fall back

Scan existing aie.flow, aie.packet_flow, and ShimDMAAllocationOp for
S2MM channel claims before allocating trace channels. When the target
shim has:
- 1 channel free: use that channel, skip distribute
- 0 channels free + lateral enabled: redirect to spare column
- 0 channels free + no lateral: emit diagnostic

Addresses review feedback from jackl-xilinx on PR #3001 (items 2+4)."
```

---

### Task 3: Multi-column >2 traces test (Jack item #1)

Add a test with 4 traces across 3 columns on `npu1_3col`. Verifies round-robin distribution works correctly when traces outnumber channels (4 traces, 2 channels).

**Files:**
- Modify: `test/dialect/AIE/trace/test_insert_trace_flows_distribute.mlir` (add test case)

- [ ] **Step 1: Write the multi-column distribute test**

```mlir
// -----

// Test: 4 traces across 3 columns, distributed across 2 channels.
// Round-robin: traces 0,2 -> channel 1 (primary), traces 1,3 -> channel 0.
// CHECK-LABEL: module @distribute_four_traces_three_cols
module @distribute_four_traces_three_cols {
  aie.device(npu1_3col) {
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)
    %tile12 = aie.tile(1, 2)
    %tile22 = aie.tile(2, 2)
    %tile00 = aie.tile(0, 0)

    aie.trace @trace_0_2(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.trace @trace_0_3(%tile03) {
      aie.trace.packet id=2 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.trace @trace_1_2(%tile12) {
      aie.trace.packet id=3 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.trace @trace_2_2(%tile22) {
      aie.trace.packet id=4 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
      aie.trace.host_config buffer_size = 8192
      aie.trace.start_config @trace_0_2
      aie.trace.start_config @trace_0_3
      aie.trace.start_config @trace_1_2
      aie.trace.start_config @trace_2_2
    }

    // All 4 traces route to column 0 shim (Single routing strategy)
    // Distributed across 2 channels: traces alternate between DMA 1 and DMA 0
    // CHECK-DAG: aie.packet_dest<%{{.*}}0_0{{.*}}, DMA : 1>
    // CHECK-DAG: aie.packet_dest<%{{.*}}0_0{{.*}}, DMA : 0>
    // Two BDs configured (one per channel)
    // CHECK-DAG: aiex.npu.writebd {bd_id = 15
    // CHECK-DAG: aiex.npu.writebd {bd_id = 14
    // Two address patches
    // CHECK-DAG: aiex.npu.address_patch {{{.*}}arg_idx = 4
    // CHECK-DAG: aiex.npu.address_patch {{{.*}}arg_idx = 5

    // Without distribute, all 4 traces use same channel
    // NODIST: aie.packet_dest<%{{.*}}, DMA : 1>
    // NODIST: aie.packet_dest<%{{.*}}, DMA : 1>
    // NODIST: aie.packet_dest<%{{.*}}, DMA : 1>
    // NODIST: aie.packet_dest<%{{.*}}, DMA : 1>
  }
}
```

- [ ] **Step 2: Run the test**

Run: `nice -n 19 build/bin/aie-opt test/dialect/AIE/trace/test_insert_trace_flows_distribute.mlir --split-input-file -aie-insert-trace-flows="distribute-channels=true" 2>&1`

Expected: PASS (round-robin logic already handles >2 traces).

If it fails, investigate the routing strategy -- the `Single` strategy routes all traces to column 0's shim. Four traces from 3 columns should all end up on column 0 with round-robin channel assignment.

- [ ] **Step 3: Run the full test suite**

Run: `nice -n 19 cmake --build build --target check-aie-mlir 2>&1 | tail -20`

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add test/dialect/AIE/trace/test_insert_trace_flows_distribute.mlir
git commit -m "test(trace): add multi-column 4-trace distribute test

Verifies round-robin channel distribution works correctly with more
traces (4) than channels (2) across multiple columns (3). Traces
alternate between S2MM channels 0 and 1.

Addresses review feedback from jackl-xilinx on PR #3001 (item 1)."
```

---

### Task 4: Draft and post PR reply

After all tests pass, draft a reply to Jack's comment addressing all 5 items. Show what's done, ask about the end-to-end test infrastructure.

**Files:** None (GitHub comment only)

- [ ] **Step 1: Draft the reply**

The reply should cover:
1. Item 1 (>2 traces): "Added a 4-trace-across-3-columns test. Round-robin assignment handles N traces across 2 channels."
2. Item 2 (shim full): "Added S2MM conflict detection. 1 channel free -> single-channel fallback. 0 free -> lateral redirect (or error if lateral off)."
3. Item 3 (auto column): "Already implemented -- `lateral-routing=true` without `lateral-target-col` auto-selects nearest spare shim NOC column."
4. Item 4 (forced + partial): "Conflict detection now applies to forced targets too. If forced column has 1 free channel, uses it single-channel."
5. Item 5 (e2e): Ask Jack for guidance on test infrastructure and what specific hardware interactions to validate.

- [ ] **Step 2: Review the draft with the user before posting**

Show the draft comment text. Wait for approval or edits.

- [ ] **Step 3: Post the reply**

Use `gh api` to post the reply as a top-level PR comment.
