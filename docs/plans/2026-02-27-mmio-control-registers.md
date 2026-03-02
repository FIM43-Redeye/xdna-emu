# MMIO Register Access + Control Register Writes - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable instruction-driven control register configuration (crRnd, crSat, crSRSSign) and memory-mapped register access from load/store instructions.

**Architecture:** Two independent features implemented in sequence. Control registers add a new `RegisterKind::Control` to the decoder that maps `mCRm` operands to `Operand::ControlReg(u8)`, with execution dispatch updating `ctx.srs_config`. MMIO intercepts load/store addresses >= 0x10000 in `memory.rs` and routes to the existing `tile.read_register()` / `write_register()`.

**Tech Stack:** Rust, AIE2 ISA (llvm-aie TableGen), AM025 register database

---

## Feature A: Control Register Writes

### Task 1: Add RegisterKind::Control and map mCRm

**Files:**
- Modify: `src/tablegen/resolver.rs:137-156` (RegisterKind enum)
- Modify: `src/tablegen/resolver.rs:226-234` (classify_operand_type match)

**Step 1: Add Control variant to RegisterKind**

In `src/tablegen/resolver.rs`, add after `Accumulator` (line 155):

```rust
    /// mCRm: control registers (crRnd, crSat, crSRSSign, crVaddSign, etc.)
    Control,
```

**Step 2: Map mCRm in classify_operand_type**

In `src/tablegen/resolver.rs`, add to the exact-match block (after line 233):

```rust
        "mCRm" => return OperandType::Register(RegisterKind::Control),
```

**Step 3: Build to verify**

Run: `TMPDIR=/tmp/claude-1000 cargo build 2>&1 | tail -5`
Expected: Compile error about non-exhaustive match in `decoder.rs:635`

---

### Task 2: Add Operand::ControlReg and wire decoder

**Files:**
- Modify: `src/interpreter/bundle/slot.rs:853-879` (Operand enum)
- Modify: `src/interpreter/decode/decoder.rs:635-648` (register kind match)

**Step 1: Add ControlReg variant to Operand**

In `src/interpreter/bundle/slot.rs`, add after `BufferDescriptor(u8)` (line 878):

```rust
    /// Control register (crRnd=6, crSat=9, crSRSSign=8, crVaddSign=0, etc.).
    /// The u8 is the 4-bit hardware register ID from the ISA encoding.
    ControlReg(u8),
```

**Step 2: Wire RegisterKind::Control in decoder**

In `src/interpreter/decode/decoder.rs`, add to the match at line 647 (after Accumulator arm):

```rust
                        RegisterKind::Control => Operand::ControlReg(reg),
```

**Step 3: Build to verify**

Run: `TMPDIR=/tmp/claude-1000 cargo build 2>&1 | tail -5`
Expected: Clean build (warnings OK). There may be non-exhaustive matches
elsewhere for `Operand` -- fix any that arise by adding `ControlReg` arms.

---

### Task 3: Handle control register writes in execution

**Files:**
- Modify: `src/interpreter/execute/scalar.rs:368-377` (write_dest)

**Step 1: Add ControlReg handling in write_dest**

In `src/interpreter/execute/scalar.rs`, replace the `_ => {}` catch-all in
`write_dest` (line 374) with explicit control register handling:

```rust
                Operand::ControlReg(id) => {
                    // Control register write: update SRS/UPS config.
                    // Each control register selects different bits from the
                    // value. The hardware register IDs come from
                    // AIE2GenRegisterInfo.td.
                    match id {
                        9 => { // crSat
                            ctx.srs_config.saturation_mode = (value & 0x3) as u8;
                            log::trace!("crSat = {} (raw 0x{:X})", value & 0x3, value);
                        }
                        6 => { // crRnd
                            ctx.srs_config.rounding_mode = (value & 0xF) as u8;
                            log::trace!("crRnd = {} (raw 0x{:X})", value & 0xF, value);
                        }
                        8 => { // crSRSSign
                            ctx.srs_config.srs_sign = (value & 1) != 0;
                            log::trace!("crSRSSign = {} (raw 0x{:X})", value & 1, value);
                        }
                        _ => {
                            log::trace!("control register write: id={}, value=0x{:X}", id, value);
                        }
                    }
                }
                _ => {} // Other operand types not valid as scalar destinations
```

**Step 2: Build to verify**

Run: `TMPDIR=/tmp/claude-1000 cargo build 2>&1 | tail -5`
Expected: Clean build.

---

### Task 4: Change SrsConfig::default() to hardware reset

**Files:**
- Modify: `src/interpreter/state/context.rs:438-454` (Default impl)

**Step 1: Change default to hardware reset values**

In `src/interpreter/state/context.rs`, replace the Default impl:

```rust
impl Default for SrsConfig {
    /// Hardware reset defaults: all zero.
    ///
    /// Kernel preamble code configures crRnd/crSat/crSRSSign via control
    /// register write instructions before any SRS/UPS operations.
    fn default() -> Self {
        Self {
            rounding_mode: 0,   // Floor
            saturation_mode: 0, // No saturation
            srs_sign: false,    // Unsigned
        }
    }
}
```

**Step 2: Build and run tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1 | tail -10`
Expected: All tests pass. If any SRS-related tests fail because they depended
on the old defaults, update them to either:
- Explicitly set srs_config before the SRS operation, or
- Use `SrsConfig { rounding_mode: 9, saturation_mode: 1, srs_sign: true }`

---

### Task 5: Add unit tests for control register writes

**Files:**
- Modify: `src/interpreter/execute/scalar.rs` (tests module at bottom)

**Step 1: Add test for crSat write**

```rust
    #[test]
    fn test_control_reg_write_crsat() {
        let mut ctx = ExecutionContext::new();
        // Write value 3 (symmetric saturate) to crSat (id=9)
        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarMov)
            .with_source(Operand::Immediate(3))
            .with_dest(Operand::ControlReg(9));
        ScalarAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.srs_config.saturation_mode, 3);
    }
```

**Step 2: Add test for crRnd write**

```rust
    #[test]
    fn test_control_reg_write_crrnd() {
        let mut ctx = ExecutionContext::new();
        // Write rounding mode 9 (PosInf) to crRnd (id=6)
        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarMov)
            .with_source(Operand::Immediate(9))
            .with_dest(Operand::ControlReg(6));
        ScalarAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.srs_config.rounding_mode, 9);
    }
```

**Step 3: Add test for crSRSSign write**

```rust
    #[test]
    fn test_control_reg_write_srssign() {
        let mut ctx = ExecutionContext::new();
        // Write 1 (signed) to crSRSSign (id=8)
        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarMov)
            .with_source(Operand::Immediate(1))
            .with_dest(Operand::ControlReg(8));
        ScalarAlu::execute(&op, &mut ctx);
        assert!(ctx.srs_config.srs_sign);
    }
```

**Step 4: Run tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1 | tail -10`
Expected: All pass.

**Step 5: Commit**

```bash
git add -A && git commit -m "feat(decode): add control register write support (crRnd, crSat, crSRSSign)"
```

---

## Feature B: MMIO Register Access

### Task 6: Add MMIO intercept in memory.rs write path

**Files:**
- Modify: `src/interpreter/execute/memory.rs:899-914` (write_memory)

**Step 1: Add register space intercept before data memory write**

In `write_memory()`, after `decode_data_address()` at line 900, before the
quadrant != 0 check at line 902, add the MMIO intercept:

```rust
        // MMIO register space: quadrant 0, offset >= 0x10000.
        // Data memory is 64KB (0x0000-0xFFFF). Everything above is register-mapped:
        // locks (0x1F000+), DMA BDs (0x1D000+), DMA channels (0x1D200+), etc.
        if quadrant == 0 && offset >= 0x10000 {
            tile.write_register(offset as u32, value as u32);
            return;
        }
```

**Step 2: Build to verify**

Run: `TMPDIR=/tmp/claude-1000 cargo build 2>&1 | tail -5`
Expected: Clean build.

---

### Task 7: Add MMIO intercept in memory.rs read path

**Files:**
- Modify: `src/interpreter/execute/memory.rs:811-826` (read_memory)

**Step 1: Add register space intercept before data memory read**

The challenge: `read_memory` takes `&Tile` (immutable) but `tile.read_register()`
takes `&mut self` because lock reads are side-effecting. Two options:

- **Option A**: Change `read_memory` to take `&mut Tile`. This cascades through
  `execute_load()` and callers. More invasive but correct.
- **Option B**: Add a non-mutating `read_register_pure()` that returns register
  values without side effects (no lock acquire on read). Good enough for MMIO
  reads of non-lock registers; lock reads via MMIO would need Option A later.

**Use Option B** for now -- most MMIO reads are for DMA status, timer, etc.
Lock reads via MMIO are rare (kernel code uses lock instructions, not MMIO).

Add to `Tile` in `tile.rs`:

```rust
    /// Read a register value without side effects.
    ///
    /// Unlike `read_register()`, this does NOT execute lock operations.
    /// Used for MMIO loads from the memory unit where mutable tile access
    /// is not available during instruction execution.
    pub fn read_register_pure(&self, offset: u32) -> u32 {
        // DMA BD range
        if (0x1D000..0x1D200).contains(&offset) {
            let bd_offset = offset - 0x1D000;
            let bd_index = (bd_offset / 0x20) as usize;
            let reg_in_bd = (bd_offset % 0x20) as usize / 4;
            if bd_index < self.dma_bds.len() {
                let bd = &self.dma_bds[bd_index];
                return match reg_in_bd {
                    0 => bd.addr_low,
                    1 => bd.addr_high,
                    2 => bd.length,
                    3 => bd.control,
                    4 => bd.d0,
                    5 => bd.d1,
                    _ => self.registers.get(&offset).copied().unwrap_or(0),
                };
            }
        }
        // DMA channel control
        if (0x1D200..0x1D400).contains(&offset) {
            let ch_offset = offset - 0x1D200;
            let ch_index = (ch_offset / 0x8) as usize;
            if ch_index < self.dma_channels.len() {
                return if ch_offset % 0x8 == 0 {
                    self.dma_channels[ch_index].control
                } else {
                    self.dma_channels[ch_index].start_queue
                };
            }
        }
        // Lock value registers (read-only, no acquire side effect)
        let reg_layout = super::regdb::device_reg_layout();
        let lock_base = if self.tile_type == TileType::MemTile {
            reg_layout.memtile_lock_base
        } else {
            reg_layout.memory_lock_base
        };
        let lock_stride = if self.tile_type == TileType::MemTile {
            reg_layout.memtile_lock_stride
        } else {
            reg_layout.memory_lock_stride
        };
        let lock_end = lock_base + (self.locks.len() as u32) * lock_stride;
        if (lock_base..lock_end).contains(&offset) {
            let lock_id = ((offset - lock_base) / lock_stride) as usize;
            if lock_id < self.locks.len() {
                return self.locks[lock_id].value as u32 & reg_layout.lock_value_mask;
            }
        }
        // Fall back to register map
        self.registers.get(&offset).copied().unwrap_or(0)
    }
```

Then in `read_memory()` (memory.rs), add before the quadrant selection:

```rust
        // MMIO register space: quadrant 0, offset >= 0x10000.
        if quadrant == 0 && offset >= 0x10000 {
            return tile.read_register_pure(offset as u32) as u64;
        }
```

**Step 2: Build to verify**

Run: `TMPDIR=/tmp/claude-1000 cargo build 2>&1 | tail -5`
Expected: Clean build.

---

### Task 8: Add unit tests for MMIO access

**Files:**
- Modify: `src/interpreter/execute/memory.rs` (tests module) or new test block

**Step 1: Add test for MMIO write to lock register**

```rust
    #[test]
    fn test_mmio_write_lock_register() {
        let mut tile = Tile::compute(1, 2);
        // Write value 5 to Lock0 via MMIO (offset 0x1F000)
        MemoryUnit::write_memory(&mut tile, 0x1F000, 5, MemWidth::Word, None);
        // Lock should be set (via tile.write_register -> lock set)
        assert_eq!(tile.locks[0].value, 5);
    }
```

**Step 2: Add test for MMIO read from lock register**

```rust
    #[test]
    fn test_mmio_read_lock_register() {
        let mut tile = Tile::compute(1, 2);
        tile.locks[3].set(7);
        // Read Lock3 via MMIO (offset 0x1F000 + 3*0x10 = 0x1F030)
        let value = MemoryUnit::read_memory(&tile, 0x1F030, MemWidth::Word, None);
        assert_eq!(value, 7);
    }
```

**Step 3: Add test for MMIO write to DMA BD**

```rust
    #[test]
    fn test_mmio_write_dma_bd() {
        let mut tile = Tile::compute(1, 2);
        // Write to DMA_BD0_0 (addr_low) via MMIO
        MemoryUnit::write_memory(&mut tile, 0x1D000, 0xDEAD_0000, MemWidth::Word, None);
        assert_eq!(tile.dma_bds[0].addr_low, 0xDEAD_0000);
    }
```

**Step 4: Add test that data memory is not corrupted by MMIO writes**

```rust
    #[test]
    fn test_mmio_write_does_not_touch_data_memory() {
        let mut tile = Tile::compute(1, 2);
        // Data memory at offset 0 should be zero initially
        assert_eq!(tile.data_memory()[0], 0);
        // Write to register space (0x1F000) -- should NOT write to data memory
        MemoryUnit::write_memory(&mut tile, 0x1F000, 42, MemWidth::Word, None);
        // Data memory should still be zero
        assert_eq!(tile.data_memory()[0], 0);
    }
```

**Step 5: Run tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1 | tail -10`
Expected: All pass.

**Step 6: Commit**

```bash
git add -A && git commit -m "feat(memory): add MMIO register access for load/store instructions"
```

---

### Task 9: Final verification

**Step 1: Run full test suite**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1 | tail -10`
Expected: All tests pass (count should be ~1255+).

**Step 2: Verify no regressions in SRS behavior**

The SrsConfig default change (Task 4) means hardware reset values (all-zero)
are now used instead of the old PosInf/saturate/signed defaults. If any
existing unit tests relied on the old defaults, they will have been fixed in
Task 4. This step confirms the full suite is clean.

**Step 3: Update MEMORY.md**

Remove "Core MMIO register access" and "Control register writes" from Known
Open Issues. Update completeness notes.
