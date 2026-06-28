# Trace Lateral Routing & Channel Distribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port lateral routing, S2MM channel distribution, and FileCheck tests onto the merged `origin/main` (post-#2962), producing a clean branch ready for PR.

**Architecture:** The merged `AIEInsertTraceFlows` pass uses `TraceHostConfigOp` for buffer/routing config and only supports `TraceShimRouting::Single`. We add two orthogonal CLI pass options (`--lateral-routing`, `--distribute-channels`) that compose with the existing routing strategy. Lateral routing redirects trace flows to spare columns (no active cores) to avoid vertical stream switch contention. Channel distribution round-robins traces across both S2MM DMA channels per shim. Both default to off, preserving existing behavior.

**Tech Stack:** MLIR/C++ (tablegen, pass infrastructure), FileCheck/lit tests, ninja build system

**Key design decision -- CLI flags vs routing enum:** Lateral routing and channel distribution are kept as CLI pass options rather than extending `TraceShimRouting`. This is intentional: @yenjames is actively designing the `host_config` op and routing enum for Python/IRON integration (#2988). CLI flags work now, and when he's ready to wire them into IRON, adding enum values like `routing = "lateral"` to `TraceShimRoutingAttr` is a small follow-up. We keep a clean seam for that future integration.

**Key design decision -- composability with future per-column routing:** @yenjames removed per-column routing from #2962 to reintroduce it in a separate PR with IRON verification. Our lateral routing is structured as a **post-processing step on shim selection**, not baked into the `Single` routing path. After the base routing strategy picks target shim columns, lateral routing optionally redirects those targets to spare columns. This means when `TraceShimRouting::PerColumn` returns, lateral routing will compose with it automatically (each column redirects to its nearest spare instead of itself). Channel distribution is also independent of routing strategy -- it operates on the final `shimInfos` map regardless of how shims were selected.

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `include/aie/Dialect/AIE/Transforms/AIEPasses.td` | Add 3 new pass options |
| Modify | `lib/Dialect/AIE/Transforms/AIEInsertTraceFlows.cpp` | Core feature implementation |
| Create | `test/dialect/AIE/trace/test_insert_trace_flows_lateral.mlir` | Lateral routing tests |
| Create | `test/dialect/AIE/trace/test_insert_trace_flows_distribute.mlir` | Channel distribution tests |

Tests that exercise non-routing features (auto packet ID, core+mem, memtile, etc.) are NOT included -- the merged #2962 already covers those in `test_insert_trace_flows.mlir`. We only add tests for new behavior.

---

### Task 1: Create branch and verify baseline

**Files:** None modified

- [ ] **Step 1: Create fresh branch from origin/main**

```bash
git checkout -b trace-routing-v3 origin/main
```

- [ ] **Step 2: Rebuild to verify baseline**

```bash
cd /home/triple/npu-work/mlir-aie/build
nice -n 19 ninja aie-opt 2>&1 | tail -5
```

Expected: builds successfully, no errors.

- [ ] **Step 3: Run existing trace tests to confirm green baseline**

```bash
cd /home/triple/npu-work/mlir-aie/build
nice -n 19 ninja check-aie 2>&1 | grep -E "(PASS|FAIL|test_insert_trace)"
```

Or run just the trace tests:

```bash
/home/triple/npu-work/mlir-aie/build/bin/llvm-lit -v test/dialect/AIE/trace/
```

Expected: all existing trace tests pass.

- [ ] **Step 4: Commit (empty, branch marker)**

Not needed -- clean branch from origin/main.

---

### Task 2: Add pass options to AIEPasses.td

**Files:**
- Modify: `include/aie/Dialect/AIE/Transforms/AIEPasses.td`

- [ ] **Step 1: Write the lateral routing test file (TDD -- test before implementation)**

Create `test/dialect/AIE/trace/test_insert_trace_flows_lateral.mlir`:

```mlir
//===- test_insert_trace_flows_lateral.mlir -------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows="lateral-routing=true" | FileCheck %s
// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows="lateral-routing=true lateral-target-col=3" | FileCheck %s --check-prefix=FORCED

// -----

// Test: Lateral routing sends trace to spare column (no active core).
// Column 0 has a core (active), column 1 has no core (spare).
// CHECK-LABEL: module @lateral_basic
module @lateral_basic {
  aie.device(npu1_2col) {
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)

    %core = aie.core(%tile02) {
      aie.end
    }

    aie.trace @core_trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 8192
      aie.trace.start_config @core_trace
    }

    // Trace routes to column 1 (spare), not column 0 (active)
    // CHECK: aie.packet_flow(1)
    // CHECK:   aie.packet_source<%{{.*}}, Trace : 0>
    // CHECK:   aie.packet_dest<%{{.*}}1_0{{.*}}, DMA : 1>
    // CHECK:   keep_pkt_header = true
  }
}

// -----

// Test: No spare column available -- falls back to column 0 shim.
// Both columns have cores.
// CHECK-LABEL: module @lateral_no_spare
module @lateral_no_spare {
  aie.device(npu1_2col) {
    %tile02 = aie.tile(0, 2)
    %tile12 = aie.tile(1, 2)
    %tile00 = aie.tile(0, 0)

    %core0 = aie.core(%tile02) { aie.end }
    %core1 = aie.core(%tile12) { aie.end }

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

    // No spare column, falls back to column 0
    // CHECK: aie.packet_dest<%{{.*}}0_0{{.*}}, DMA : 1>
  }
}

// -----

// Test: Forced lateral target column via CLI option.
// FORCED-LABEL: module @lateral_basic
// (reuses first test module -- FORCED prefix checks forced column)
// Trace should go to the forced column 3 (even though column 1 is nearer spare)
// This test is only valid with npu1_4col but we use 2col so col 3 doesn't exist.
// Instead, test forced column with 4-column device in a dedicated module below.

// FORCED-LABEL: module @lateral_no_spare
// With forced col=3 on a 2-col device, column 3 doesn't exist. Falls back.

// -----

// Test: lateral-routing=false (default) keeps traces on column 0 shim even with spare
// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows | FileCheck %s --check-prefix=NOLATRL

// NOLATRL-LABEL: module @lateral_basic
// Without lateral routing, trace stays on column 0
// NOLATRL: aie.packet_dest<%{{.*}}0_0{{.*}}, DMA : 1>
```

- [ ] **Step 2: Run the test to verify it fails (pass options don't exist yet)**

```bash
/home/triple/npu-work/mlir-aie/build/bin/aie-opt test/dialect/AIE/trace/test_insert_trace_flows_lateral.mlir -aie-insert-trace-flows="lateral-routing=true" 2>&1 | head -5
```

Expected: error about unknown option `lateral-routing`.

- [ ] **Step 3: Add the three new options to AIEPasses.td**

In `include/aie/Dialect/AIE/Transforms/AIEPasses.td`, inside the `AIEInsertTraceFlows` options list, after the `clTraceBurstLength` option and before `];`:

```tablegen
    Option<"clDistributeChannels", "distribute-channels", "bool", "false",
           "Distribute traces across multiple S2MM channels per shim tile">,
    Option<"clLateralRouting", "lateral-routing", "bool", "false",
           "Route traces to spare columns to minimize data path perturbation">,
    Option<"clLateralTargetCol", "lateral-target-col", "int", "-1",
           "Force lateral routing target column (-1 = auto-detect nearest spare)">
```

- [ ] **Step 4: Rebuild to pick up tablegen changes**

```bash
cd /home/triple/npu-work/mlir-aie/build
nice -n 19 ninja aie-opt 2>&1 | tail -5
```

Expected: builds. The options now exist but the pass ignores them.

- [ ] **Step 5: Verify the option is recognized (pass runs, test still fails on CHECK)**

```bash
/home/triple/npu-work/mlir-aie/build/bin/aie-opt test/dialect/AIE/trace/test_insert_trace_flows_lateral.mlir --split-input-file -aie-insert-trace-flows="lateral-routing=true" 2>&1 | head -20
```

Expected: pass runs without "unknown option" error. Output still routes to column 0 (feature not implemented yet).

- [ ] **Step 6: Commit**

```bash
git add include/aie/Dialect/AIE/Transforms/AIEPasses.td test/dialect/AIE/trace/test_insert_trace_flows_lateral.mlir
git commit -m "feat(trace): add lateral-routing and distribute-channels pass options

Add three new CLI options to -aie-insert-trace-flows:
  --lateral-routing: route traces to spare columns (default: off)
  --distribute-channels: round-robin across S2MM channels (default: off)
  --lateral-target-col: force a specific lateral target column

Options registered but not yet implemented in the pass logic."
```

---

### Task 3: Implement lateral routing in AIEInsertTraceFlows.cpp

**Files:**
- Modify: `lib/Dialect/AIE/Transforms/AIEInsertTraceFlows.cpp`

The implementation adds lateral routing as a **post-processing step after Phase 2b** shim selection. This keeps it composable with any future routing strategy (e.g., when @yenjames reintroduces per-column routing). When `--lateral-routing=true`, the pass identifies columns with active `aie.core` ops and redirects already-selected shim targets to the nearest NOC shim column without active cores. When no spare exists, the original target is kept.

- [ ] **Step 1: Add `#include <climits>` and helper methods**

At the top of the file, after the existing includes, add:

```cpp
#include <climits>
```

In the `private:` section of `AIEInsertTraceFlowsPass`, add these two helpers after the existing `computeTimerCtrlAddress`:

```cpp
  /// Find or create a shim tile at the given column.
  TileOp getOrCreateShim(DeviceOp device, OpBuilder &builder, int col) {
    for (auto tile : device.getOps<TileOp>()) {
      if (tile.getCol() == col && tile.getRow() == 0)
        return tile;
    }
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&device.getRegion().front());
    return TileOp::create(builder, device.getLoc(), col, 0);
  }

  /// Find the nearest NOC shim column without active cores.
  /// Returns -1 if no spare column exists.
  int findNearestSpareColumn(int sourceCol, const std::set<int> &activeColumns,
                             const AIETargetModel &targetModel) {
    if (clLateralTargetCol >= 0)
      return clLateralTargetCol;
    int bestCol = -1;
    int bestDist = INT_MAX;
    int numCols = targetModel.columns();
    for (int c = 0; c < numCols; c++) {
      if (activeColumns.count(c) == 0 && targetModel.isShimNOCTile(c, 0)) {
        int dist = std::abs(c - sourceCol);
        if (dist > 0 && dist < bestDist) {
          bestDist = dist;
          bestCol = c;
        }
      }
    }
    return bestCol;
  }
```

- [ ] **Step 2: Add lateral routing as a post-processing step after Phase 2b**

The existing Phase 2b code (`if (routing == TraceShimRouting::Single) { ... }`) is left **untouched**. After it completes (and after the closing brace), add a new section:

```cpp
    // Phase 2b-lateral: Optionally redirect shim targets to spare columns.
    // This is a post-processing step that works with ANY routing strategy.
    // When per-column routing returns, lateral routing will compose with it
    // (each column's shim redirects to its nearest spare).
    if (clLateralRouting) {
      std::set<int> activeColumns;
      device.walk([&](CoreOp core) {
        auto coreTile = cast<TileOp>(core.getTile().getDefiningOp());
        activeColumns.insert(coreTile.getCol());
      });

      // Collect all unique shim target columns and find redirects
      std::map<int, int> redirects; // old target col -> new target col
      for (auto &[col, shimInfo] : shimInfos) {
        int curTarget = shimInfo.shimTile.getCol();
        if (redirects.count(curTarget))
          continue; // already computed
        if (activeColumns.count(curTarget) == 0)
          continue; // already spare, no redirect needed
        int spare = findNearestSpareColumn(curTarget, activeColumns,
                                           targetModel);
        if (spare >= 0)
          redirects[curTarget] = spare;
      }

      // Apply redirects: rebuild shimInfos with new target columns
      if (!redirects.empty()) {
        std::map<int, ShimInfo> newShimInfos;
        for (auto &[col, shimInfo] : shimInfos) {
          int curTarget = shimInfo.shimTile.getCol();
          auto it = redirects.find(curTarget);
          if (it != redirects.end()) {
            int newTarget = it->second;
            shimInfo.shimTile = getOrCreateShim(device, builder, newTarget);
          }
          newShimInfos[col] = shimInfo;
        }
        shimInfos = std::move(newShimInfos);
      }
    }
```

- [ ] **Step 3: Rebuild and run the lateral routing test**

```bash
cd /home/triple/npu-work/mlir-aie/build
nice -n 19 ninja aie-opt 2>&1 | tail -3
/home/triple/npu-work/mlir-aie/build/bin/llvm-lit -v ../test/dialect/AIE/trace/test_insert_trace_flows_lateral.mlir
```

Expected: lateral_basic and lateral_no_spare tests pass. FORCED and NOLATRL prefixes may need adjustment depending on exact output.

- [ ] **Step 4: Run ALL existing trace tests to verify no regressions**

```bash
/home/triple/npu-work/mlir-aie/build/bin/llvm-lit -v ../test/dialect/AIE/trace/
```

Expected: all tests pass (lateral routing defaults to off, so existing tests are unaffected).

- [ ] **Step 5: Commit**

```bash
git add lib/Dialect/AIE/Transforms/AIEInsertTraceFlows.cpp
git commit -m "feat(trace): implement lateral routing in AIEInsertTraceFlows

When --lateral-routing=true, the pass identifies columns with active
aie.core ops and redirects trace packet flows to the nearest NOC shim
column without active cores. This avoids trace traffic competing with
data flows on vertical stream switch ports.

Falls back to column 0 when no spare column exists. A specific target
column can be forced with --lateral-target-col=N."
```

---

### Task 4: Implement channel distribution

**Files:**
- Modify: `lib/Dialect/AIE/Transforms/AIEInsertTraceFlows.cpp`
- Create: `test/dialect/AIE/trace/test_insert_trace_flows_distribute.mlir`

Channel distribution spreads traces across both S2MM DMA channels on a shim tile, doubling trace bandwidth. This requires restructuring `ShimInfo` to track multiple channels with per-channel BD/argIdx.

- [ ] **Step 1: Write the channel distribution test file**

Create `test/dialect/AIE/trace/test_insert_trace_flows_distribute.mlir`:

```mlir
//===- test_insert_trace_flows_distribute.mlir ----------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows="distribute-channels=true" | FileCheck %s
// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows | FileCheck %s --check-prefix=NODIST

// -----

// Test: Two traces are distributed across DMA channels 0 and 1.
// CHECK-LABEL: module @distribute_two_traces
module @distribute_two_traces {
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
      aie.trace.host_config buffer_size = 8192
      aie.trace.start_config @trace_a
      aie.trace.start_config @trace_b
    }

    // First trace -> channel 1 (default shim-channel), second -> channel 0
    // CHECK-DAG: aie.packet_dest<%{{.*}}, DMA : 1>
    // CHECK-DAG: aie.packet_dest<%{{.*}}, DMA : 0>
  }
}

// -----

// Test: Single trace -- no distribution even when enabled (only 1 trace).
// CHECK-LABEL: module @distribute_single_trace
module @distribute_single_trace {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)

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

    // Single trace always uses default channel
    // CHECK: aie.packet_dest<%{{.*}}, DMA : 1>
  }
}

// -----

// Test: Without distribute-channels, both traces use same DMA channel.
// NODIST-LABEL: module @distribute_two_traces
// NODIST: aie.packet_dest<%{{.*}}, DMA : 1>
// NODIST: aie.packet_dest<%{{.*}}, DMA : 1>
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/home/triple/npu-work/mlir-aie/build/bin/llvm-lit -v ../test/dialect/AIE/trace/test_insert_trace_flows_distribute.mlir
```

Expected: CHECK-DAG for `DMA : 0` fails (both traces go to channel 1).

- [ ] **Step 3: Restructure ShimInfo and add channel distribution logic**

This is the most involved change. In `AIEInsertTraceFlows.cpp`:

**3a.** Add the `ChannelDescriptor` struct after `TraceInfo`:

```cpp
/// Per-channel DMA resource allocation.
struct ChannelDescriptor {
  int channel; // S2MM channel number
  int bdId;    // Buffer descriptor ID
  int argIdx;  // Runtime sequence argument index
};
```

**3b.** Restructure `ShimInfo` to use channels:

```cpp
struct ShimInfo {
  TileOp shimTile;
  int channel;      // Primary S2MM channel (used when not distributing)
  int bdId;         // Primary buffer descriptor ID
  int argIdx;       // Primary runtime sequence argument index
  int bufferOffset; // Offset in bytes (for trace_after_last_tensor)
  std::vector<ChannelDescriptor> channels; // Per-channel descriptors
  std::vector<int> traceChannelAssignment; // Per-trace index into channels
  std::vector<TraceInfo> traceSources;     // All traces routed to this shim
  std::optional<int> startBroadcast;
  std::optional<int> stopBroadcast;
};
```

**3c.** Add `buildChannelDescriptors` helper in the `private:` section:

```cpp
  /// Build channel descriptors. Always includes the primary channel.
  /// Adds a secondary channel when distribute-channels is enabled
  /// and there are multiple traces.
  std::vector<ChannelDescriptor>
  buildChannelDescriptors(size_t numTraces, int primaryChannel,
                          int primaryBdId, int primaryArgIdx) {
    std::vector<ChannelDescriptor> chans;
    chans.push_back({primaryChannel, primaryBdId, primaryArgIdx});
    if (clDistributeChannels && numTraces > 1) {
      int ch2 = (primaryChannel == 1) ? 0 : 1;
      chans.push_back({ch2, primaryBdId - 1, primaryArgIdx + 1});
    }
    return chans;
  }
```

**3d.** After Phase 2b builds `shimInfos`, add channel descriptor construction and assignment:

```cpp
    // Build channel descriptors and trace-to-channel assignments
    for (auto &[col, shimInfo] : shimInfos) {
      shimInfo.channels = buildChannelDescriptors(
          shimInfo.traceSources.size(), shimInfo.channel,
          shimInfo.bdId, shimInfo.argIdx);
      // Round-robin assignment of traces to channels
      for (size_t i = 0; i < shimInfo.traceSources.size(); i++) {
        shimInfo.traceChannelAssignment.push_back(
            i % shimInfo.channels.size());
      }
    }
```

**3e.** Update Phase 3 (packet flow insertion) to use channel assignment:

Replace the Phase 3 loop body. Instead of `shimInfo.channel` for the DMA dest, use:

```cpp
    for (auto &info : traceInfos) {
      int col = info.tile.getCol();
      ShimInfo &shimInfo = shimInfos[col];

      // Find this trace's index in shimInfo.traceSources to get channel
      int chanIdx = 0;
      for (size_t i = 0; i < shimInfo.traceSources.size(); i++) {
        if (shimInfo.traceSources[i].packetId == info.packetId) {
          chanIdx = shimInfo.traceChannelAssignment[i];
          break;
        }
      }
      auto &chanDesc = shimInfo.channels[chanIdx];

      auto packetFlowOp = PacketFlowOp::create(
          builder, device.getLoc(), builder.getI8IntegerAttr(info.packetId),
          nullptr, nullptr);

      Block *flowBody = new Block();
      packetFlowOp.getPorts().push_back(flowBody);
      OpBuilder flowBuilder = OpBuilder::atBlockEnd(flowBody);

      PacketSourceOp::create(flowBuilder, device.getLoc(),
                             Value(info.tile.getResult()), info.tracePort,
                             info.traceChannel);

      PacketDestOp::create(flowBuilder, device.getLoc(),
                           Value(shimInfo.shimTile.getResult()),
                           WireBundle::DMA, chanDesc.channel);

      EndOp::create(flowBuilder, device.getLoc());

      packetFlowOp->setAttr("keep_pkt_header", builder.getBoolAttr(true));
    }
```

**3f.** Update Phase 4c-4e to loop over channels:

In the `// 4c-4f. Insert per-shim configurations` loop, wrap the BD/address-patch/DMA-config section in a per-channel loop:

```cpp
      for (auto &chanDesc : shimInfo.channels) {
        // Convert buffer size to words
        int bufferLengthWords = bufferSizeBytes / 4;

        // 4c. Write buffer descriptor
        xilinx::AIEX::NpuWriteBdOp::create(
            builder, runtimeSeq.getLoc(),
            shimCol, chanDesc.bdId, bufferLengthWords,
            0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            clTraceBurstLength);

        // 4d. Address patch
        uint32_t bdAddress = computeBDAddress(shimCol, chanDesc.bdId,
                                              shimInfo.shimTile, targetModel);
        xilinx::AIEX::NpuAddressPatchOp::create(
            builder, runtimeSeq.getLoc(),
            bdAddress, chanDesc.argIdx, shimInfo.bufferOffset);

        // 4e. DMA channel configuration
        uint32_t ctrlAddr = computeCtrlAddress(
            DMAChannelDir::S2MM, chanDesc.channel,
            shimInfo.shimTile, targetModel);
        xilinx::AIEX::NpuMaskWrite32Op::create(
            builder, runtimeSeq.getLoc(), ctrlAddr, 3840, 7936,
            nullptr, builder.getI32IntegerAttr(shimCol),
            builder.getI32IntegerAttr(0));

        // Push BD to task queue
        uint32_t taskQueueAddr = computeTaskQueueAddress(
            DMAChannelDir::S2MM, chanDesc.channel,
            shimInfo.shimTile, targetModel);
        uint32_t bdIdWithToken = (1U << 31) | chanDesc.bdId;
        xilinx::AIEX::NpuWrite32Op::create(
            builder, runtimeSeq.getLoc(), taskQueueAddr, bdIdWithToken,
            nullptr, builder.getI32IntegerAttr(shimCol),
            builder.getI32IntegerAttr(0));
      }
```

Note: The 4f broadcast section stays OUTSIDE the per-channel loop (broadcasts are per-shim, not per-channel).

- [ ] **Step 4: Rebuild and run tests**

```bash
cd /home/triple/npu-work/mlir-aie/build
nice -n 19 ninja aie-opt 2>&1 | tail -3
/home/triple/npu-work/mlir-aie/build/bin/llvm-lit -v ../test/dialect/AIE/trace/test_insert_trace_flows_distribute.mlir
```

Expected: all distribution tests pass.

- [ ] **Step 5: Run full trace test suite**

```bash
/home/triple/npu-work/mlir-aie/build/bin/llvm-lit -v ../test/dialect/AIE/trace/
```

Expected: all tests pass including existing ones (distribute defaults to off).

- [ ] **Step 6: Commit**

```bash
git add lib/Dialect/AIE/Transforms/AIEInsertTraceFlows.cpp test/dialect/AIE/trace/test_insert_trace_flows_distribute.mlir
git commit -m "feat(trace): implement S2MM channel distribution

When --distribute-channels=true, traces are round-robin distributed
across both S2MM DMA channels per shim tile (channels 0 and 1). Each
channel gets its own buffer descriptor and runtime sequence argument
index. This doubles available trace bandwidth when multiple traces
target the same shim.

Single-trace configurations are unaffected (only one channel used).
Default behavior (distribute-channels=false) is unchanged."
```

---

### Task 5: Combined lateral + distribute test and integration verification

**Files:**
- Create: test added to `test_insert_trace_flows_lateral.mlir` (append)
- Run: full test suite

- [ ] **Step 1: Add a combined lateral+distribute test**

Append to `test/dialect/AIE/trace/test_insert_trace_flows_lateral.mlir`, adding a new RUN line and test module:

Add this RUN line at the top of the file:

```
// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows="lateral-routing=true distribute-channels=true" | FileCheck %s --check-prefix=COMBO
```

Append this test module at the end of the file:

```mlir
// -----

// Test: Lateral routing + channel distribution compose together.
// Two traces in column 0 (active) should route laterally to column 1 (spare)
// AND distribute across two DMA channels.
// COMBO-LABEL: module @lateral_and_distribute
module @lateral_and_distribute {
  aie.device(npu1_2col) {
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)
    %tile00 = aie.tile(0, 0)

    %core0 = aie.core(%tile02) { aie.end }

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

    // Both traces route to column 1 (spare) with different DMA channels
    // COMBO-DAG: aie.packet_dest<%{{.*}}1_0{{.*}}, DMA : 1>
    // COMBO-DAG: aie.packet_dest<%{{.*}}1_0{{.*}}, DMA : 0>
  }
}
```

- [ ] **Step 2: Run the combined test**

```bash
/home/triple/npu-work/mlir-aie/build/bin/llvm-lit -v ../test/dialect/AIE/trace/test_insert_trace_flows_lateral.mlir
```

Expected: all prefixes (CHECK, FORCED, NOLATRL, COMBO) pass.

- [ ] **Step 3: Run full trace test suite one final time**

```bash
/home/triple/npu-work/mlir-aie/build/bin/llvm-lit -v ../test/dialect/AIE/trace/
```

Expected: all green.

- [ ] **Step 4: Run clang-format on modified C++ files**

```bash
clang-format -i lib/Dialect/AIE/Transforms/AIEInsertTraceFlows.cpp
git diff lib/Dialect/AIE/Transforms/AIEInsertTraceFlows.cpp
```

If there are formatting changes, stage and commit them.

- [ ] **Step 5: Commit**

```bash
git add test/dialect/AIE/trace/test_insert_trace_flows_lateral.mlir
git commit -m "test(trace): add combined lateral+distribute integration test"
```

If clang-format made changes:

```bash
git add lib/Dialect/AIE/Transforms/AIEInsertTraceFlows.cpp
git commit -m "style: clang-format AIEInsertTraceFlows.cpp"
```

---

### Task 6: Local hardware validation

**Files:** None modified (testing only)

This task is manual -- verifying that a real trace-enabled design still works with the new pass on hardware.

- [ ] **Step 1: Build the full toolchain**

```bash
cd /home/triple/npu-work/mlir-aie/build
nice -n 19 ninja 2>&1 | tail -10
```

- [ ] **Step 2: Run the existing trace programming example**

```bash
cd /home/triple/npu-work/mlir-aie/programming_examples/basic/event_trace
# Build and run on hardware (if available)
make clean && make
```

Or if hardware is not immediately available, at least verify the compilation pipeline produces valid output:

```bash
/home/triple/npu-work/mlir-aie/build/bin/aie-opt \
  programming_examples/basic/event_trace/aie_trace.mlir \
  -aie-insert-trace-flows 2>&1 | head -20
```

- [ ] **Step 3: Test lateral routing on a multi-column example**

Create a quick test MLIR or use an existing multi-column example and run:

```bash
/home/triple/npu-work/mlir-aie/build/bin/aie-opt <test.mlir> \
  -aie-insert-trace-flows="lateral-routing=true" 2>&1
```

Verify the output routes to a spare column.

- [ ] **Step 4: Note results for PR description**

Record what was tested and whether it passed. This goes in the PR body.

---

## Implementation Notes for Future Python Integration

When @yenjames is ready to expose these features through IRON/Python (#2988):

1. **Lateral routing composes with any routing strategy.** Because it's a post-processing step, when `TraceShimRouting::PerColumn` returns, lateral routing automatically works with it. Each column's shim target gets redirected to the nearest spare. No changes needed to our code.

2. **Extending the routing enum (optional):** If @yenjames wants to make lateral routing a first-class routing mode instead of a CLI flag:
   ```tablegen
   def TraceShimRoutingLateral : I32EnumAttrCase<"Lateral", 1, "lateral">;
   ```
   Then emit `routing = lateral` from `configure_trace_output()`. The pass would check `routing == TraceShimRouting::Lateral` as an alternative to `clLateralRouting`. Both paths would call the same post-processing logic.

3. **Channel distribution:** Could become a `distribute_channels = true` attribute on `host_config`, or a separate pass option forwarded from Python. The implementation is independent of routing strategy.

4. **CLI flags continue to work as overrides** regardless of what the `host_config` op says, so nothing breaks during the transition.
