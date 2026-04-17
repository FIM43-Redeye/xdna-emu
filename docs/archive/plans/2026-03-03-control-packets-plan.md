# Control Packet Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the emulator correctly handle control packets end-to-end: write operations through full register dispatch, read operations with response routing.

**Architecture:** Control packet writes route through DeviceState's module-dispatch path (same as CDO writes) via a closure passed from array.rs. OP_READ generates response packets pushed into the TileCtrl slave port. Verification through mock_xrt test.exe for binary-compatible testing.

**Tech Stack:** Rust, mock_xrt (C++ shared library), cmake, mlir-aie test.cpp

**Design doc:** `docs/plans/2026-03-03-control-packets-design.md`

---

### Task 1: Unit Test for Register Dispatch Gap

Demonstrate that control packet writes to MemTile DMA BD registers do NOT
update structured state. This is the RED phase -- prove the bug exists.

**Files:**
- Test: `src/device/tile.rs` (test module at bottom)

**Step 1: Write the failing test**

In the `#[cfg(test)] mod tests` section of `src/device/tile.rs`, add:

```rust
#[test]
fn test_ctrl_packet_write_updates_dma_bd() {
    // Control packet writes to DMA BD registers should update structured
    // BD state, not just the register HashMap. Currently they only call
    // tile.write_register() which doesn't handle MemTile BD offsets.
    use crate::device::aie2_spec::*;

    let params = super::TileParams::mem_tile();
    let mut tile = super::Tile::new(
        super::TileType::MemTile, 0, 1, &params,
    );

    // Write to MemTile BD0 word 0 (Buffer_Address) via write_register.
    // MemTile BD base offset comes from regdb: typically 0xA0000.
    let reg_layout = super::super::regdb::device_reg_layout();
    let bd0_offset = reg_layout.memtile_bd_base;

    // Write a known value
    tile.write_register(bd0_offset, 0xDEAD_0000);

    // Check: the structured BD should have been updated.
    // Currently this FAILS because tile.write_register() only handles
    // compute-tile BD range (0x1D000-0x1D1FF), not MemTile BD range.
    assert_eq!(
        tile.dma_bds[0].addr_low, 0xDEAD_0000,
        "write_register should update structured BD state for MemTile"
    );
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --lib test_ctrl_packet_write_updates_dma_bd`
Expected: FAIL -- `tile.dma_bds[0].addr_low` is still 0 because
`tile.write_register()` doesn't dispatch MemTile BD writes.

**Step 3: Commit**

```
git add src/device/tile.rs
git commit -m "test: RED - ctrl packet write doesn't update memtile BD state"
```

---

### Task 2: Implement ctrl_packet_write on DeviceState

Add a method that control packets can use to write registers through the full
module-dispatch path, identical to CDO writes.

**Files:**
- Modify: `src/device/state.rs` (add method near existing write_register)
- Modify: `src/device/array.rs` (expose method for tile ctrl routing)

**Step 1: Add ctrl_packet_write to DeviceState**

In `src/device/state.rs`, add a public method:

```rust
/// Write a register as if from a control packet.
///
/// Routes through the same module-dispatch path as CDO write_register,
/// ensuring MemTile DMA BDs, stream switch config, and all other
/// structured state gets updated.
pub fn ctrl_packet_write(&mut self, col: u8, row: u8, offset: u32, value: u32) {
    // Reconstruct the full tile address that write_register expects.
    // TileAddress::encode builds the 32-bit address from (col, row, offset).
    let full_addr = TileAddress::encode(col, row, offset);
    if let Err(e) = self.write_register(full_addr, value) {
        log::error!("ctrl_packet_write({},{}) offset=0x{:05X}: {}",
            col, row, offset, e);
    }
}
```

Check whether `TileAddress::encode` exists. If not, build the address manually:
```rust
// Tile address encoding: col in bits [25:23], row in bits [22:18],
// offset in bits [17:0] (for 18-bit intra-tile offset).
// Verify this matches TileAddress::decode().
let full_addr = ((col as u32) << 25) | ((row as u32) << 18) | offset;
```

Look at `TileAddress::decode()` in state.rs to determine the exact encoding.

**Step 2: Verify existing write_register test still passes**

Run: `cargo test --lib test_lock_value_write`
Expected: PASS (no regression)

**Step 3: Commit**

```
git add src/device/state.rs
git commit -m "feat: add ctrl_packet_write for full module dispatch"
```

---

### Task 3: Wire Control Packet Execution Through DeviceState

Refactor `execute_ctrl_packet` to use the DeviceState dispatch path instead
of `tile.write_register()`. The tile collects packet data (state machine in
`process_ctrl_packet_word`), then execution routes through DeviceState.

**Files:**
- Modify: `src/device/tile.rs` (change execute_ctrl_packet to return actions)
- Modify: `src/device/array.rs` (route_tile_switches_to_ctrl dispatches actions)
- Modify: `src/device/state.rs` (if needed for array access)

**Step 1: Define a CtrlPacketAction enum**

In `src/device/tile.rs`:

```rust
/// Action produced by control packet execution, to be dispatched by
/// the caller (which has DeviceState access).
#[derive(Debug)]
pub enum CtrlPacketAction {
    /// Write value to register at (col, row, offset).
    WriteRegister { offset: u32, value: u32 },
    /// Read N registers starting at offset, generate response.
    ReadRegisters { offset: u32, count: u8, response_id: u8 },
    /// Error encountered during execution.
    Error(String),
}
```

**Step 2: Refactor execute_ctrl_packet to return actions**

Change `execute_ctrl_packet` to return `Vec<CtrlPacketAction>` instead of
performing writes directly:

```rust
fn execute_ctrl_packet(
    &self,
    base_address: u32,
    operation: u8,
    response_id: u8,
    data: &[u32],
) -> Vec<CtrlPacketAction> {
    use crate::device::aie2_spec::*;
    let mut actions = Vec::new();

    match operation {
        CTRL_PKT_OP_WRITE | CTRL_PKT_OP_BLOCK_WRITE => {
            for (i, &value) in data.iter().enumerate() {
                let addr = base_address + (i as u32) * 4;
                log::info!("Tile ({},{}) ctrl_pkt WRITE: [0x{:05X}] = 0x{:08X}",
                    self.col, self.row, addr, value);
                actions.push(CtrlPacketAction::WriteRegister {
                    offset: addr,
                    value,
                });
            }
        }
        CTRL_PKT_OP_READ => {
            log::info!("Tile ({},{}) ctrl_pkt READ: addr=0x{:05X} beats={}",
                self.col, self.row, base_address, data.len().max(1));
            // beats field from header tells how many words to read.
            // data slice is empty for reads; use beats_total from caller.
            actions.push(CtrlPacketAction::ReadRegisters {
                offset: base_address,
                count: data.len() as u8, // Will be 0; caller should pass beats_total
                response_id,
            });
        }
        CTRL_PKT_OP_WRITE_INCR => {
            for (i, &value) in data.iter().enumerate() {
                let addr = base_address + (i as u32) * 4;
                log::info!("Tile ({},{}) ctrl_pkt WRITE_INCR: [0x{:05X}] = 0x{:08X}",
                    self.col, self.row, addr, value);
                actions.push(CtrlPacketAction::WriteRegister {
                    offset: addr,
                    value,
                });
            }
        }
        _ => {
            actions.push(CtrlPacketAction::Error(format!(
                "Tile ({},{}) ctrl_pkt: unknown operation {} (addr=0x{:05X})",
                self.col, self.row, operation, base_address,
            )));
        }
    }
    actions
}
```

**Step 3: Update process_ctrl_packet_word to return actions**

Change `process_ctrl_packet_word` to return `Vec<CtrlPacketAction>`:

```rust
pub fn process_ctrl_packet_word(&mut self, word: u32, tlast: bool) -> Vec<CtrlPacketAction> {
    // ... existing state machine ...
    // Where it currently calls self.execute_ctrl_packet(...), instead:
    // let actions = self.execute_ctrl_packet(...);
    // return actions;
    // All other paths return Vec::new()
}
```

Note: for OP_READ, the beats_total is known from the header but data.len()
will be 0 (reads have no data payload). Pass beats_total explicitly to
execute_ctrl_packet, or store it in the ReadRegisters action.

**Step 4: Update route_tile_switches_to_ctrl in array.rs**

```rust
fn route_tile_switches_to_ctrl(&mut self) -> usize {
    // ... existing TileCtrl master port draining ...
    while self.tiles[i].stream_switch.masters[master_idx].has_data() {
        if let Some((data, tlast)) = self.tiles[i].stream_switch.masters[master_idx].pop_with_tlast() {
            let actions = self.tiles[i].process_ctrl_packet_word(data, tlast);
            words_routed += 1;

            for action in actions {
                match action {
                    CtrlPacketAction::WriteRegister { offset, value } => {
                        // Route through full module dispatch.
                        // Need DeviceState access here -- this is in TileArray,
                        // so we need to bubble up or restructure.
                        // For now, collect actions and return them to caller.
                    }
                    CtrlPacketAction::ReadRegisters { offset, count, response_id } => {
                        // Read registers and push response into TileCtrl slave
                    }
                    CtrlPacketAction::Error(msg) => {
                        log::error!("{}", msg);
                        self.tiles[i].stream_switch.fatal_errors.push(msg);
                    }
                }
            }
        }
    }
}
```

**Architecture note**: `route_tile_switches_to_ctrl` lives on `TileArray`, not
`DeviceState`. To call `DeviceState::ctrl_packet_write`, we need to either:
(a) Move ctrl packet routing to `DeviceState::step()`, or
(b) Return pending actions from `TileArray::step()` for DeviceState to execute, or
(c) Have TileArray call a write method that goes through the same dispatch.

Option (b) is cleanest -- return actions from TileArray::step(), DeviceState
dispatches them. Check how `TileArray::step()` is called from `DeviceState`.

**Step 5: Run all tests**

Run: `cargo test --lib`
Expected: All existing tests pass. Task 1 test should now pass (if dispatch
works for MemTile BDs).

**Step 6: Commit**

```
git add src/device/tile.rs src/device/array.rs src/device/state.rs
git commit -m "feat: route ctrl packet writes through full module dispatch"
```

---

### Task 4: Make Task 1 Test Pass (GREEN Phase)

Verify that the register dispatch fix makes the Task 1 test pass. If it
doesn't, debug and fix the specific dispatch path for MemTile BDs.

**Step 1: Run the specific test**

Run: `cargo test --lib test_ctrl_packet_write_updates_dma_bd`
Expected: PASS

If it still fails, the issue is that the MemTile BD offset in
`tile.write_register()` doesn't match the range that `state.write_register()`
handles. Check that `RegisterModule::from_offset()` correctly classifies
MemTile BD offsets, and that the offset reconstruction preserves the module
bits.

**Step 2: Commit if needed**

```
git commit -m "fix: ensure ctrl packet MemTile BD dispatch works"
```

---

### Task 5: Unit Test for OP_READ Response Generation

**Files:**
- Test: `src/device/tile.rs` (test module)

**Step 1: Write the test**

```rust
#[test]
fn test_ctrl_packet_read_generates_response() {
    use crate::device::aie2_spec::*;

    let params = super::TileParams::compute();
    let mut tile = super::Tile::new(
        super::TileType::Compute, 0, 2, &params,
    );

    // Pre-populate a register value that the read will return
    tile.registers.insert(0x440, 0xCAFE_0001);
    tile.registers.insert(0x444, 0xCAFE_0002);
    tile.registers.insert(0x448, 0xCAFE_0003);
    tile.registers.insert(0x44C, 0xCAFE_0004);

    // Build a READ control packet header:
    // address=0x440, beats=3 (meaning 4 words), operation=1, response_id=2
    let address: u32 = 0x440;
    let beats: u32 = 3; // 3 means 4 words (beats + 1)
    let operation: u32 = CTRL_PKT_OP_READ as u32;
    let response_id: u32 = 2;
    let header = (response_id << CTRL_PKT_RESPONSE_ID_SHIFT)
        | (operation << CTRL_PKT_OPERATION_SHIFT)
        | (beats << CTRL_PKT_LENGTH_SHIFT)
        | address;

    // Feed the header (from Idle state)
    let actions = tile.process_ctrl_packet_word(header, true);

    // Should produce a ReadRegisters action
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        CtrlPacketAction::ReadRegisters { offset, count, response_id: rid } => {
            assert_eq!(*offset, 0x440);
            assert_eq!(*count, 4); // beats + 1
            assert_eq!(*rid, 2);
        }
        other => panic!("Expected ReadRegisters, got {:?}", other),
    }
}
```

**Step 2: Run test to verify behavior**

Run: `cargo test --lib test_ctrl_packet_read_generates_response`
Expected: PASS (if Task 3 refactoring is correct)

**Step 3: Commit**

```
git add src/device/tile.rs
git commit -m "test: ctrl packet OP_READ produces ReadRegisters action"
```

---

### Task 6: Implement OP_READ Response Packet Injection

When a ReadRegisters action is produced, read the registers and push a
response packet into the tile's TileCtrl slave port.

**Files:**
- Modify: `src/device/array.rs` (handle ReadRegisters action)
- Modify: `src/device/stream_switch.rs` (if TileCtrl slave port needs helpers)

**Step 1: Identify the TileCtrl slave port index**

In `src/device/stream_switch.rs`, the TileCtrl port is tagged during
construction. Add a helper:

```rust
/// Find the TileCtrl slave port index, if any.
pub fn tile_ctrl_slave_port(&self) -> Option<usize> {
    self.slaves.iter().position(|p| matches!(p.port_type, PortType::TileCtrl))
}
```

**Step 2: Handle ReadRegisters in route_tile_switches_to_ctrl**

```rust
CtrlPacketAction::ReadRegisters { offset, count, response_id } => {
    let col = self.tiles[i].col;
    let row = self.tiles[i].row;

    // Read registers
    let mut response_data: Vec<u32> = Vec::with_capacity(count as usize);
    for j in 0..count as u32 {
        let addr = offset + j * 4;
        let val = self.tiles[i].read_register(addr);
        response_data.push(val);
        log::info!("Tile ({},{}) ctrl_pkt READ: [0x{:05X}] = 0x{:08X}",
            col, row, addr, val);
    }

    // Build stream packet header for response
    // pkt_id = response_id, pkt_type from routing config
    let pkt_header = PacketHeader::encode(response_id, 0);

    // Push into TileCtrl slave port
    if let Some(slave_idx) = self.tiles[i].stream_switch.tile_ctrl_slave_port() {
        // Stream header first
        self.tiles[i].stream_switch.slaves[slave_idx].push(pkt_header);
        // Then data words
        for (j, &val) in response_data.iter().enumerate() {
            let is_last = j == response_data.len() - 1;
            self.tiles[i].stream_switch.slaves[slave_idx]
                .push_with_tlast(val, is_last);
        }
        log::info!("Tile ({},{}) ctrl_pkt READ response: {} words via slave[{}]",
            col, row, count, slave_idx);
    } else {
        log::error!("Tile ({},{}) ctrl_pkt READ: no TileCtrl slave port", col, row);
    }
}
```

**Note**: Check PacketHeader::encode signature. The stream packet header
format is: pkt_id in bits [4:0], pkt_type in bits [14:12], etc. The exact
encoding is in stream_switch.rs -- find `PacketHeader::encode` or equivalent.

**Step 3: Write integration test**

```rust
#[test]
fn test_ctrl_packet_read_response_reaches_switch() {
    // Set up a compute tile with TileCtrl ports
    // Feed a READ control packet
    // Verify response data appears in TileCtrl slave port
}
```

**Step 4: Run tests**

Run: `cargo test --lib`
Expected: All pass

**Step 5: Commit**

```
git add src/device/array.rs src/device/stream_switch.rs
git commit -m "feat: OP_READ generates response packets through TileCtrl slave"
```

---

### Task 7: Run Unit Tests and Fix Regressions

Full regression check after all changes.

**Step 1: Run all library tests**

Run: `cargo test --lib`
Expected: All existing tests pass, plus new tests from Tasks 1/5.

**Step 2: Run npu-test suite (quick check)**

Run: `cargo run --bin npu-test 2>&1 | tee /tmp/npu-test-ctrl-pkt.log`

Check that:
- No new timeouts or regressions
- Control packet tests may still show XFAIL (npu-test uses zeros for Opaque)
- All previously passing tests still pass

**Step 3: Commit if fixes needed**

```
git commit -m "fix: address regressions from ctrl packet dispatch refactor"
```

---

### Task 8: Assess mock_xrt Readiness

Before trying to run test.exe, verify that mock_xrt can build and that
the FFI bridge is functional. DO NOT assume it works.

**Files:**
- Read: `mock_xrt/CMakeLists.txt`
- Read: `mock_xrt/src/emulator_bridge.cpp`
- Read: `mock_xrt/scripts/run_mlir_aie_tests.sh`

**Step 1: Build the Rust FFI library**

Run: `cd /home/triple/npu-work/xdna-emu && cargo build --release`
Expected: libxdna_emu.so exists in target/release/

**Step 2: Build mock_xrt**

```bash
cd /home/triple/npu-work/xdna-emu/mock_xrt
mkdir -p build && cd build
cmake .. -DXDNA_EMU_ROOT=/home/triple/npu-work/xdna-emu
make -j$(nproc)
```

Expected: libxrt_mock.so built successfully. If cmake or build fails, note
the errors -- they indicate what mock_xrt work is needed before test.exe
can run.

**Step 3: Check if test artifacts exist**

```bash
ls /home/triple/npu-work/xdna-emu/tests/mlir-aie/peano/add_one_ctrl_packet/
ls /home/triple/npu-work/xdna-emu/tests/mlir-aie/chess/add_one_ctrl_packet/
```

We need: xclbin file, aie_run_seq.bin (or insts.txt). If these exist, the
test was pre-built by the build-mlir-aie-tests.sh script.

**Step 4: Try compiling test.exe**

If mock_xrt built successfully and test artifacts exist:

```bash
cd /home/triple/npu-work/xdna-emu/mock_xrt/build
# Check if the test runner script can compile add_one_ctrl_packet
../scripts/run_mlir_aie_tests.sh -l add_one_ctrl_packet
../scripts/run_mlir_aie_tests.sh -c add_one_ctrl_packet
```

Note what happens. If compilation fails, record the errors for a follow-up
task.

**Step 5: Document findings**

Record mock_xrt status in this session's notes. Possible outcomes:
(a) Mock_xrt builds and test.exe compiles -- proceed to Task 9
(b) Mock_xrt builds but test.exe compilation fails -- note what headers/APIs
    are missing
(c) Mock_xrt doesn't build -- note cmake/build errors for future fix

**Step 6: Commit build fixes if any**

---

### Task 9: Run add_one_ctrl_packet via mock_xrt (if Task 8 succeeded)

**Prerequisite**: Task 8 outcome (a) -- mock_xrt and test.exe both build.

**Step 1: Run the test**

```bash
cd /home/triple/npu-work/xdna-emu/mock_xrt/build
../scripts/run_mlir_aie_tests.sh -v add_one_ctrl_packet
```

Or manually:
```bash
./tests/add_one_ctrl_packet/test.exe \
    --xclbin tests/mlir-aie/chess/add_one_ctrl_packet/*.xclbin \
    --instr tests/mlir-aie/chess/add_one_ctrl_packet/aie_run_seq.bin
```

**Step 2: Analyze results**

If PASS: Control packets work end-to-end. Update test_overrides.toml to
remove XFAIL for this test.

If FAIL: Check emulator logs (RUST_LOG=xdna_emu=info) to see:
- Are control packets arriving at the TileCtrl port?
- Are lock values being set correctly?
- Is the core executing?
- Are DMA transfers completing?
- Is the OP_READ response being generated and routed?

**Step 3: Iterate on failures**

Debug and fix based on log analysis. Common issues:
- Buffer address mapping (mock_xrt device addresses vs emulator addresses)
- NPU instruction sequence parsing (sync commands, DMA descriptors)
- Packet routing configuration mismatch
- OP_READ response format wrong

**Step 4: Commit fixes and update overrides**

```
git commit -m "feat: control packets working via mock_xrt"
```

---

### Task 10: Update Documentation and XFAIL Expectations

**Files:**
- Modify: `tests/test_overrides.toml`
- Modify: memory files

**Step 1: Update test overrides based on results**

Remove XFAIL entries for tests that now pass, or update the reason string
for tests that fail for a different reason than before.

**Step 2: Update MEMORY.md**

Update the accuracy roadmap section to reflect control packet status.

**Step 3: Commit**

```
git commit -m "docs: update control packet status and test expectations"
```
