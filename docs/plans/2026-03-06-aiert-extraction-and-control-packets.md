# aie-rt Extraction Infrastructure + Control Packet Completion

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract hardware constants from aie-rt at build time (replacing hardcoded values), then fix control packet support to pass all 7 bridge tests.

**Architecture:** `build.rs` runs `gcc -E` on aie-rt's `xaiemlgbl_reginit.c` to resolve all macros, then a Rust parser extracts DMA module properties, lock module properties, and stream switch port mappings into generated Rust source files. Control packet BD writes switch from eager re-parsing to lazy storage (matching hardware behavior where BD registers are just storage until the DMA engine reads them). OP_READ response routing is verified end-to-end.

**Tech Stack:** Rust build.rs, gcc preprocessor, aie-rt C headers (read-only), existing AM025 JSON pipeline.

**Ground Truth Sources:**
- aie-rt: `../aie-rt/driver/src/` (branch xlnx_rel_v2025.2)
- Key file: `global/xaiemlgbl_reginit.c` -- DMA modules (lines 416-909), lock modules (lines 2445-2500), stream switch (lines 1700-2174)
- Key file: `global/xaiegbl.h` -- enum definitions (StrmSwPortType at line 249)
- Key file: `dma/xaie_dma_aieml.c` -- BD write sequence (lines 285-408), wait-for-done (lines 1209-1238)

---

## Phase 1: aie-rt Extraction Infrastructure

### Task 1: gcc -E preprocessor integration in build.rs

**Files:**
- Modify: `build.rs` -- add aie-rt preprocessing function
- Generated: `$OUT_DIR/gen_aiert_dma.rs` -- DMA module constants

**Context:** `build.rs` already generates Rust from AM025 JSON and device model JSON. We add a third source: aie-rt C headers, preprocessed to resolve macros.

**Preprocessing command (verified working):**
```
gcc -E -I<aie-rt>/driver/src -I<aie-rt>/driver/src/global
      -I<aie-rt>/driver/src/core -I<aie-rt>/driver/src/dma
      -I<aie-rt>/driver/src/locks -I<aie-rt>/driver/src/stream_switch
      -I<aie-rt>/driver/src/io_backend -I<aie-rt>/driver/src/interrupt
      -I<aie-rt>/driver/src/timer -I<aie-rt>/driver/src/events
      -I<aie-rt>/driver/src/pm -I<aie-rt>/driver/src/rsc
      -I<aie-rt>/driver/src/routing -I<aie-rt>/driver/src/perfcount
      <aie-rt>/driver/src/global/xaiemlgbl_reginit.c
```

All subdirectories of `aie-rt/driver/src/` must be on the include path.

**Step 1: Write test for DMA constant extraction**

Add to `build.rs` a function `extract_aiert_dma()` that:
1. Locates aie-rt via `AIE_RT_PATH` env var or `../aie-rt` relative to manifest
2. Runs `gcc -E` with all include paths
3. Parses three DMA module structs from the output
4. Generates `gen_aiert_dma.rs`

The preprocessed output contains exactly three `XAie_DmaMod` struct initializers with these fields (all resolved to numeric literals):
```
.BaseAddr = 0x000A0000,    // MemTile
.IdxOffset = 0x20,
.NumBds = 48,
.NumLocks = 192U,
.StartQueueBase = 0x000A0604,
.ChCtrlBase = 0x000A0600,
.NumChannels = 6,
.ChIdxOffset = 0x8,
.ChStatusBase = 0x000A0660,
```

The three structs appear in order: MemTile DMA, Compute DMA, Shim DMA. They are identified by their `BaseAddr` values: `0xA0000` (MemTile), `0x1D000` (Compute), `0x1D000` (Shim -- same BD base, different channel base `0x1D200`).

**Parser approach:** Line-by-line scan for `.FieldName = value,` patterns within struct initializer blocks. Track which struct we're in by counting `{`/`}` depth after seeing `XAie_DmaMod`.

**Generated output format (`gen_aiert_dma.rs`):**
```rust
// Auto-generated from aie-rt xaiemlgbl_reginit.c -- do not edit.

pub mod memtile_dma {
    pub const BD_BASE: u32 = 0x000A0000;
    pub const BD_STRIDE: u32 = 0x20;
    pub const NUM_BDS: usize = 48;
    pub const NUM_LOCKS: usize = 192;
    pub const START_QUEUE_BASE: u32 = 0x000A0604;
    pub const CH_CTRL_BASE: u32 = 0x000A0600;
    pub const NUM_CHANNELS: usize = 6;
    pub const CH_STRIDE: u32 = 0x8;
    pub const CH_STATUS_BASE: u32 = 0x000A0660;
}

pub mod compute_dma {
    pub const BD_BASE: u32 = 0x0001D000;
    pub const BD_STRIDE: u32 = 0x20;
    pub const NUM_BDS: usize = 16;
    pub const NUM_LOCKS: usize = 16;
    pub const START_QUEUE_BASE: u32 = 0x0001DE04;
    pub const CH_CTRL_BASE: u32 = 0x0001DE00;
    pub const NUM_CHANNELS: usize = 2;
    pub const CH_STRIDE: u32 = 0x8;
    pub const CH_STATUS_BASE: u32 = 0x0001DF00;
}

pub mod shim_dma {
    pub const BD_BASE: u32 = 0x0001D000;
    pub const BD_STRIDE: u32 = 0x20;
    pub const NUM_BDS: usize = 16;
    pub const NUM_LOCKS: usize = 16;
    pub const START_QUEUE_BASE: u32 = 0x0001D204;
    pub const CH_CTRL_BASE: u32 = 0x0001D200;
    pub const NUM_CHANNELS: usize = 2;
    pub const CH_STRIDE: u32 = 0x8;
    pub const CH_STATUS_BASE: u32 = 0x0001D220;
}
```

**Step 2: Write test that validates extracted constants against current hardcoded values**

In a new file `src/device/aiert_validation.rs` (or as `#[cfg(test)]` in an existing file), write tests that compare the generated constants against the values currently in `regdb.rs` / `registers_spec.rs`. Any mismatch is a bug in one or the other.

```rust
#[cfg(test)]
mod tests {
    include!(concat!(env!("OUT_DIR"), "/gen_aiert_dma.rs"));

    #[test]
    fn validate_memtile_dma_constants() {
        let lay = crate::device::regdb::device_reg_layout();
        assert_eq!(memtile_dma::BD_BASE, lay.memtile_bd_base);
        assert_eq!(memtile_dma::BD_STRIDE, lay.memtile_bd_stride);
        // ... etc
    }
}
```

**Step 3: Build and verify**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib aiert_validation`
Expected: All assertions pass (constants match).

**Step 4: Commit**

```
feat: extract DMA module constants from aie-rt at build time
```

**Graceful fallback:** If aie-rt is not found or gcc is unavailable, `build.rs` should `println!("cargo:warning=...")` and skip generation. The existing regdb-derived constants remain the runtime source. aie-rt extraction is a validation layer, not a hard dependency (yet).

---

### Task 2: Lock module extraction

**Files:**
- Modify: `build.rs` -- add lock module extraction
- Generated: `$OUT_DIR/gen_aiert_locks.rs`

**Context:** aie-rt defines three lock modules in `xaiemlgbl_reginit.c` (lines 2445-2500):
- Compute tile: BaseAddr=0x40000 (XAIEMLGBL_MEMORY_MODULE_LOCK0_VALUE), NumLocks=16, LockIdOff=0x10
- MemTile: BaseAddr=0xC0000 (XAIEMLGBL_MEM_TILE_MODULE_LOCK0_VALUE), NumLocks=64, LockIdOff=0x10
- Shim: BaseAddr=0x40000, NumLocks=16, LockIdOff=0x10

**Parsed fields:** `.BaseAddr`, `.NumLocks`, `.LockIdOff`

**Generated output (`gen_aiert_locks.rs`):**
```rust
pub mod compute_locks {
    pub const BASE: u32 = 0x00040000;
    pub const NUM_LOCKS: usize = 16;
    pub const STRIDE: u32 = 0x10;
}
pub mod memtile_locks {
    pub const BASE: u32 = 0x000C0000;
    pub const NUM_LOCKS: usize = 64;
    pub const STRIDE: u32 = 0x10;
}
pub mod shim_locks {
    pub const BASE: u32 = 0x00040000;
    pub const NUM_LOCKS: usize = 16;
    pub const STRIDE: u32 = 0x10;
}
```

**Validation test:** Compare against `reg_layout.memory_lock_base`, `memtile_lock_base`, etc.

**Step 1-4:** Same TDD pattern as Task 1.

**Step 5: Commit**

```
feat: extract lock module constants from aie-rt at build time
```

---

### Task 3: Stream switch port map extraction

**Files:**
- Modify: `build.rs` -- add port map extraction
- Generated: `$OUT_DIR/gen_aiert_ports.rs`

**Context:** aie-rt defines six port map arrays (master + slave for each of 3 tile types). Each entry is `{ .PortType = <enum>, .PortNum = <u8> }`. The PortType enum (from `xaiegbl.h` line 249): CORE=0, DMA=1, CTRL=2, FIFO=3, SOUTH=4, WEST=5, NORTH=6, EAST=7, TRACE=8.

After `gcc -E`, PortType values remain as symbolic names (they're C enums, not macros). The parser must map them to integers using the known enum ordering.

**Arrays to extract (6 total):**
- `AieMlTileStrmSwMasterPortMap[]` -- compute tile master ports
- `AieMlTileStrmSwSlavePortMap[]` -- compute tile slave ports
- `AieMlMemTileStrmSwMasterPortMap[]` -- memtile master ports
- `AieMlMemTileStrmSwSlavePortMap[]` -- memtile slave ports
- `AieMlShimStrmSwMasterPortMap[]` -- shim master ports
- `AieMlShimStrmSwSlavePortMap[]` -- shim slave ports

**Generated output (`gen_aiert_ports.rs`):**
```rust
/// Port type enum matching aie-rt XAie_StrmSwPortType.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AieRtPortType {
    Core = 0, Dma = 1, Ctrl = 2, Fifo = 3,
    South = 4, West = 5, North = 6, East = 7, Trace = 8,
}

/// (PortType, PortNum) for each physical port index.
pub const COMPUTE_MASTER_PORTS: &[(AieRtPortType, u8)] = &[
    (AieRtPortType::Core, 0),  // phys 0
    (AieRtPortType::Dma, 0),   // phys 1
    (AieRtPortType::Dma, 1),   // phys 2
    (AieRtPortType::Ctrl, 0),  // phys 3
    (AieRtPortType::Fifo, 0),  // phys 4
    (AieRtPortType::South, 0), // phys 5
    // ...
];
// ... 5 more arrays
```

**Validation test:** Compare against the port arrays currently generated by `gen_stream_ports()` in build.rs (from AM025 JSON). Any differences indicate a bug in the AM025 extraction or our interpretation.

**Step 1-4:** TDD pattern.

**Step 5: Commit**

```
feat: extract stream switch port maps from aie-rt at build time
```

---

### Task 4: Wire extracted constants into runtime (replace hardcoded values)

**Files:**
- Modify: `src/device/registers.rs` -- use aie-rt constants for module classification
- Modify: `src/device/regdb.rs` -- cross-validate at startup
- Potentially modify: `src/device/aie2_spec.rs` -- port maps

**Step 1:** Add `include!` for the generated files in appropriate modules.

**Step 2:** Add startup validation in `regdb.rs` that asserts aie-rt constants match regdb values. If they diverge, log an error. This catches drift between AM025 JSON and aie-rt.

**Step 3:** Where aie-rt provides values that regdb does NOT (e.g., `StartQueueBase`, `ChStatusBase`, `NumChannels`), use the aie-rt constants directly instead of hardcoding.

**Step 4:** Run full test suite.

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: 1540+ tests pass.

**Step 5: Commit**

```
feat: wire aie-rt extracted constants into runtime, replace hardcoded values
```

---

## Phase 2: Fix Control Packet BD Writes

### Task 5: Lazy BD parsing -- store raw words, parse on demand

**Files:**
- Modify: `src/device/state.rs:1053` -- `write_memtile_dma_bd()`
- Modify: `src/device/state.rs` -- equivalent for `write_dma_bd()` (compute tile)
- Modify: `src/device/dma/engine.rs` -- lazy BD config loading

**Root cause:** `write_memtile_dma_bd()` re-parses the entire BD (all 8 words) after EVERY single-word write. Control packets write BDs word-by-word, so the first 7 writes produce corrupt BdConfig entries.

**Ground truth (aie-rt):** `_XAieMl_MemTileDmaWriteBd()` writes all words atomically via `XAie_BlockWrite32()`. The DMA engine reads BDs when the channel starts or fetches the next BD in a chain. The valid bit (word 7 for MemTile, word 5 for compute) gates whether the BD is active.

**Fix:** Split BD handling into two paths:

1. **Single-word write** (from control packets): Store the raw value in the BD struct. Mark the BD as "dirty" but do NOT re-parse into BdConfig.

2. **Bulk write** (from CDO DmaWrite): All words arrive at once. Parse immediately (existing behavior, correct).

3. **DMA engine reads BD:** When the DMA engine needs a BD (channel start, next-BD chain), it checks if the BD is dirty and re-parses from raw words at that point.

**Implementation detail:** Add a `dirty: Vec<bool>` (one per BD) to the tile or DMA engine. `write_memtile_dma_bd()` sets `dirty[bd_idx] = true` and stores the word. `DmaEngine::start_channel()` and `DmaEngine::advance_to_next_bd()` check dirty flags and re-parse.

**Step 1: Write failing test**

```rust
#[test]
fn test_bd_single_word_write_does_not_corrupt() {
    // Write BD words one at a time (simulating control packets)
    // Verify the DMA engine sees the correct final config
    // after all 8 words are written
}
```

**Step 2: Implement lazy parsing in write_memtile_dma_bd()**

Remove the `parse_memtile_bd_from_words()` call from the single-word path. Add dirty flag. Keep bulk write path unchanged.

**Step 3: Implement dirty-check in DMA engine**

When DMA engine reads a BD, if dirty, call the parse function.

**Step 4: Run tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: All existing tests still pass (CDO path unchanged).

**Step 5: Commit**

```
fix: lazy BD parsing for single-word writes (control packet support)
```

---

### Task 6: Verify DMA channel control via control packets

**Files:**
- Modify: `src/device/state.rs:1209+` -- `write_memtile_dma_channel()`

**Context:** After control packets write BD registers, they also write channel control registers to start the DMA. Verify that `write_memtile_dma_channel()` correctly handles:
- Channel enable (start)
- StartBd field (which BD to begin with)
- Task queue push

**Ground truth:** aie-rt `DMA_S2MM_0_Ctrl` register at offset `ChCtrlBase`. The channel start sequence writes the StartBd index, then enables the channel.

**Step 1:** Add focused logging to `write_memtile_dma_channel()`.

**Step 2:** Run `ctrl_packet_reconfig` (chess) and verify channel start commands appear in logs after BD writes.

**Step 3:** Fix any issues found.

**Step 4: Commit**

```
fix: verify DMA channel start via control packets
```

---

### Task 7: Run ctrl_packet_reconfig and fix remaining issues

**Files:** Various -- depends on what breaks.

**Step 1:** Run `ctrl_packet_reconfig` (chess) through bridge test.

```bash
./scripts/emu-bridge-test.sh ctrl_packet_reconfig
```

**Step 2:** If it passes, run all 4 reconfig variants:

```bash
./scripts/emu-bridge-test.sh -v ctrl_packet_reconfig ctrl_packet_reconfig_1x4_cores ctrl_packet_reconfig_4x1_cores ctrl_packet_reconfig_elf
```

**Step 3:** Fix any remaining issues (likely timing, lock sequencing, or BD field interpretation).

**Step 4: Commit each fix separately.**

---

## Phase 3: OP_READ Response Routing

### Task 8: Verify OP_READ response path end-to-end

**Files:**
- Read: `src/device/array.rs:241-327` -- handle_read_registers, drain_ctrl_responses
- Read: `src/device/stream_switch.rs` -- TileCtrl slave port identification
- Read: `src/interpreter/engine/coordinator.rs:245-248` -- ReadRegisters dispatch

**Context:** The `add_one_ctrl_packet` tests send OP_READ control packets and expect register values back in a `ctrlOut` host buffer. The response path is:

```
OP_READ arrives at tile
  -> handle_read_registers() reads registers, queues response in pending_ctrl_response
  -> drain_ctrl_responses() pushes words to TileCtrl slave port
  -> packet switch routes response to shim
  -> shim DMA S2MM writes to host buffer
```

**Step 1:** Run `add_one_ctrl_packet` (chess) with debug logging and trace the response path.

**Step 2:** Identify where responses get stuck:
- Are responses queued? (check pending_ctrl_response)
- Are they drained to TileCtrl slave? (check drain_ctrl_responses)
- Does the packet switch route them? (check packet_flow config for response path)
- Does the shim DMA receive them? (check S2MM channel)

**Step 3:** Fix the broken link in the chain.

**Step 4: Commit**

```
fix: OP_READ response routing through packet switch to host
```

---

### Task 9: Pass all add_one_ctrl_packet variants

**Step 1:** Run all 3 variants:

```bash
./scripts/emu-bridge-test.sh -v add_one_ctrl_packet add_one_ctrl_packet_4_cores add_one_ctrl_packet_col_overlay
```

**Step 2:** Fix any variant-specific issues (4-core = multiple tiles, col_overlay = column overlay CDO format).

**Step 3: Commit each fix separately.**

---

### Task 10: Full verification -- all 7 tests, both compilers

**Step 1:** Run complete bridge suite:

```bash
./scripts/emu-bridge-test.sh
```

**Step 2:** Verify all 7 control packet tests pass on Chess. Note Peano status (Peano compile failures are upstream llvm-aie bugs, not our problem).

**Step 3:** Update test status documentation and MEMORY.md.

**Step 4: Commit**

```
docs: update test status after control packet completion
```

---

## Architecture Notes to Capture Along the Way

As we implement, note any incomplete subsystems found. Create entries in MEMORY.md for future work:

- [ ] RegisterModule::from_offset() is row-agnostic (can't distinguish MemTile vs Compute at same offset) -- works today because offset ranges don't overlap, but fragile
- [ ] Tile::write_register() duplicates some of DeviceState::write_register() logic (locks, DMA BDs) -- the tile version is a subset; consider whether tile should delegate to DeviceState
- [ ] Stream switch port maps are generated from AM025 JSON AND aie-rt -- after Task 3 validation, consider which source to use as primary
- [ ] BD struct has only 6 fields (legacy) but real BDs have 8 words -- the word 6-7 spillover to register map is fragile
