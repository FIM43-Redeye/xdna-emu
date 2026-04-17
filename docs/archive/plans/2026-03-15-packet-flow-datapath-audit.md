# Packet Flow Data Path Audit

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find and fix the bug causing all-zero output in packet_flow_fanout (and similar packet-routed tests).

**Architecture:** The packet_flow_fanout test uses packet-switched stream routing through 4 tiles: Shim(0,0) -> MemTile(0,1) -> Compute(0,2)+(0,3) -> MemTile(0,1) -> Shim(0,0). The forward path (shim to compute) works. The return path (compute to shim) produces zero output. Each task below isolates one component of the return path for independent verification.

**Tech Stack:** Rust, cargo test, XDNA_EMU bridge tests

**Key data flow (return path):**
```
Compute MM2S (reads tile mem, pushes stream_out)
  -> route_dma_to_tile_switches (pops stream_out, pushes to switch slave)
  -> step_tile_switches (routes slave->master via local_routes or packet_rules)
  -> propagate_inter_tile (master -> next tile's slave, 1-cycle pipeline)
  -> step_tile_switches (second pass)
  -> route_tile_switches_to_dma (pops master, pushes to S2MM stream_in)
  -> DMA S2MM (consumes stream_in, writes to memory/DDR)
```

**Test topology (packet_flow_fanout):**
| Packet ID | From | To | Purpose |
|-----------|------|-----|---------|
| 1 | Compute(0,2) MM2S 0 | MemTile(0,1) S2MM via North:0 | Return result from tile 0,2 |
| 2 | MemTile(0,1) MM2S 1 | Shim(0,0) S2MM 0 via South:2 | Forward result to host |
| 5 | Compute(0,3) MM2S 0 | MemTile(0,1) S2MM via North:3 | Return result from tile 0,3 |
| 6 | MemTile(0,1) MM2S 3 | Shim(0,0) S2MM 1 via South:3 | Forward result to host |

---

## Task 1: Verify Compute MM2S Produces Data

**Goal:** Confirm that after lock 3 is acquired, the compute DMA MM2S channel actually reads tile memory and pushes words to `stream_out`.

**Files:**
- Read: `src/device/dma/engine/stepping.rs` (transfer_mm2s function)
- Test: run packet_flow_fanout with DMA-level logging

- [ ] **Step 1: Run test with MM2S transfer logging**

```bash
cd /home/triple/npu-work/mlir-aie/build/test/npu-xrt/packet_flow_fanout/chess
RUST_LOG=xdna_emu::device::dma::engine::stepping=debug XDNA_EMU=debug timeout 300 nice -n 19 ./test.exe 2>&1 | grep -E "MM2S.*tile\(0,2\)|MM2S.*tile\(0,3\)" | grep -v "check_acquire" | head -20
```

Expected: lines like `MM2S transfer: addr=0x8000 bytes=4 words=1` from compute tiles.
If no output: the MM2S transfer never fires -- investigate channel FSM state.

- [ ] **Step 2: Check that stream_out has data**

```bash
# Same test, look for route_dma_to_tile_switches consuming stream_out
RUST_LOG=xdna_emu::device::array::routing=info XDNA_EMU=debug timeout 300 nice -n 19 ./test.exe 2>&1 | grep -E "DMA_MM2S.*tile.*\(0,2\)|DMA_MM2S.*tile.*\(0,3\)" | head -20
```

Expected: lines like `DMA_MM2S->TileSwitch: tile (0,2) slave[N] <- 0xXXXXXXXX`
If no output: stream_out is empty or route_dma_to_tile_switches can't find the right slave port.

---

## Task 2: Verify Compute Tile Stream Switch Routes

**Goal:** Confirm that the compute tile's stream switch has a local route (or packet route) from the DMA slave port to a North master port.

**Files:**
- Read: `src/device/stream_switch/mod.rs` (step function, packet routing)
- Read: `src/device/state/dispatch.rs` (CDO stream switch configuration)

- [ ] **Step 1: Dump compute tile stream switch configuration**

Add a temporary log or use existing debug logging to dump the local_routes for compute tiles (0,2) and (0,3) after CDO configuration completes. Check:
- What DMA slave ports exist (should include DMA slave for MM2S ch0)
- What master ports exist (should include North masters)
- What local_routes or packet_rules connect them

```bash
RUST_LOG=xdna_emu::device::stream_switch=debug XDNA_EMU=debug timeout 300 nice -n 19 ./test.exe 2>&1 | grep -E "tile\(0,2\).*route|tile\(0,3\).*route|configure.*\(0,2\)|configure.*\(0,3\)|packet_rule.*\(0,2\)|packet_rule.*\(0,3\)" | head -30
```

- [ ] **Step 2: Check if step_tile_switches forwards data**

```bash
RUST_LOG=xdna_emu::device::stream_switch=debug XDNA_EMU=debug timeout 300 nice -n 19 ./test.exe 2>&1 | grep -E "step.*forward|route.*slave.*master|packet.*match|no.*route" | head -30
```

Expected: data forwarded from DMA slave to North master.
If "no route" or no forwarding: the compute tile switch is misconfigured.

---

## Task 3: Verify Compute -> MemTile Inter-Tile Propagation

**Goal:** Confirm that data exits compute tile North masters and enters MemTile South slaves.

**Files:**
- Read: `src/device/array/routing.rs` (propagate_inter_tile)

- [ ] **Step 1: Check inter-tile pipeline for compute->memtile transfers**

```bash
RUST_LOG=xdna_emu::device::array::routing=debug XDNA_EMU=debug timeout 300 nice -n 19 ./test.exe 2>&1 | grep -E "InterTile.*\(0,2\).*\(0,1\)|InterTile.*\(0,3\).*\(0,1\)" | head -20
```

Expected: `InterTile: (0,2) master[N] -> pipeline -> (0,1) slave[M]`
If no output: data never leaves compute tile switch masters.

---

## Task 4: Verify MemTile Stream Switch Routes (Return Path)

**Goal:** Confirm the MemTile switch routes return-path packets from South slaves to DMA masters (for S2MM reception) and then from DMA slaves (after S2MM->MM2S ping-pong) to South masters (toward shim).

**Files:**
- Read: `src/device/stream_switch/packet_switch.rs` (packet matching)
- Read: CDO MLIR for packet_rules configuration

- [ ] **Step 1: Dump MemTile packet routing rules**

```bash
RUST_LOG=xdna_emu::device::stream_switch=debug XDNA_EMU=debug timeout 300 nice -n 19 ./test.exe 2>&1 | grep -E "tile\(0,1\).*packet|tile\(0,1\).*rule|tile\(0,1\).*slot|memtile.*route" | head -30
```

- [ ] **Step 2: Check if MemTile receives return data from South**

```bash
RUST_LOG=xdna_emu::device::array::routing=debug XDNA_EMU=debug timeout 300 nice -n 19 ./test.exe 2>&1 | grep -E "InterTile.*\(0,1\).*slave" | head -20
```

---

## Task 5: Verify MemTile -> Shim Inter-Tile Propagation

**Goal:** Confirm data exits MemTile South masters and enters Shim North slaves.

- [ ] **Step 1: Check for MemTile->Shim transfers**

```bash
RUST_LOG=xdna_emu::device::array::routing=debug XDNA_EMU=debug timeout 300 nice -n 19 ./test.exe 2>&1 | grep -E "InterTile.*\(0,1\).*\(0,0\)|InterTile.*\(0,0\)" | head -20
```

---

## Task 6: Verify Shim S2MM Receives and Writes DDR

**Goal:** Confirm shim DMA S2MM channels consume stream data and write to host DDR.

- [ ] **Step 1: Check shim DMA S2MM activity**

```bash
RUST_LOG=xdna_emu::device::dma::engine::stepping=debug XDNA_EMU=debug timeout 300 nice -n 19 ./test.exe 2>&1 | grep -E "S2MM.*tile\(0,0\)|TileSwitch->DMA.*tile\(0,0\)" | head -20
```

---

## Task 7: Verify Packet Header Handling

**Goal:** The return path uses packet-switched routing. Verify that:
1. Compute MM2S inserts packet headers (pkt_id=1 for tile 0,2, pkt_id=5 for tile 0,3)
2. Stream switches match packet headers against routing rules
3. Packet headers are stripped before delivery to S2MM

**Files:**
- Read: `src/device/dma/transfer/core.rs` (generate_packet_header)
- Read: `src/device/stream_switch/packet_switch.rs` (packet matching logic)
- Read: `src/device/stream_switch/mod.rs` (step function, header stripping)

- [ ] **Step 1: Verify packet headers are generated**

Check that compute MM2S BD1 has `enable_packet=true` and correct packet_id.
The earlier log shows: `DMA(0,2) ch2 BD1 packet header: 0x80020001 (pkt_id=1, type=Data)` -- this ONLY fires at BD start. Verify the header word is actually pushed to stream_out.

- [ ] **Step 2: Verify packet matching in stream switch**

Add targeted logging in the packet_switch step function to see if packets with ID 1 and 5 are matched against any rules in the compute tile switches. If no rules exist, the packets are dropped.

---

## Execution Strategy

Start with Task 1 (does MM2S produce data?). If it does, move to Task 2 (does the switch route it?). Follow the data path forward until you find where it stops. Each task is independently runnable and takes 2-5 minutes.

The most likely failure point based on prior analysis: **Task 2** -- the compute tile stream switch may not have packet routing rules configured for the DMA MM2S slave port, causing return-path packets to be silently dropped.
