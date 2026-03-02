# MMIO Register Access + Control Register Writes

**Date**: 2026-02-27
**Status**: Approved

## Problem

Two independent gaps in emulator accuracy:

1. **MMIO**: Load/store instructions to register-mapped addresses (locks, DMA,
   timer, etc.) go to data memory instead of the register interface. Addresses
   >= 0x10000 in quadrant 0 are register space, not data memory.

2. **Control registers**: Instructions like `movx crRnd, r5` configure SRS
   rounding/saturation behavior. The decoder doesn't recognize `mCRm` register
   class operands, so writes are silently dropped. `SrsConfig` uses hardcoded
   defaults instead of instruction-driven values.

## Design

### Control Register Writes (ISA register operand path)

Control registers are written via ISA register operands, not MMIO. The
instruction `movx crRnd, r5` is a register-to-register move where the
destination is a control register.

**Decoder**: Add `RegisterKind::Control` to `classify_operand_type()`. Map
`mCRm` to this kind. The 4-bit encoding gives the control register ID.

**Operand**: Add `Operand::ControlReg(u8)` variant. The u8 is the raw 4-bit
hardware ID:

| Register | ID (binary) | Core_CR bits | Effect |
|----------|-------------|--------------|--------|
| crVaddSign | 0b0000 | TBD | Vector add sign |
| crRnd | 0b0110 | [5:2] | SRS rounding mode |
| crSRSSign | 0b1000 | [17] | SRS sign mode |
| crSat | 0b1001 | [1:0] | Saturation mode |

**Execution**: When MOVX dispatch sees a `ControlReg` destination, extract
relevant bits from source scalar register and update `ctx.srs_config`.

**Default change**: `SrsConfig::default()` becomes hardware reset (all-zero).
Kernel preamble code sets the real values via control register writes.

### MMIO Register Access (memory-mapped path)

When load/store targets address >= 0x10000 in quadrant 0, route to
`tile.read_register()` / `tile.write_register()` instead of data memory.

**Detection**: In `read_memory()` / `write_memory()`, check
`quadrant == 0 && offset >= 0x10000`.

**Read path**: `tile.read_register(offset)` returns register value. Already
handles locks, DMA BDs, DMA channels, cascade config.

**Write path**: `tile.write_register(offset, value)`. Side effects (lock
operations, DMA start, etc.) already happen inside.

**Scope**: Local tile (quadrant 0) only. Cross-tile register MMIO is future work.

### Files Modified

| File | Changes |
|------|---------|
| `src/tablegen/resolver.rs` | Add `RegisterKind::Control`, map `mCRm` |
| `src/interpreter/bundle/slot.rs` | Add `Operand::ControlReg(u8)` |
| `src/interpreter/execute/` | Handle control reg write in MOVX dispatch |
| `src/interpreter/state/context.rs` | `SrsConfig::default()` -> hardware reset |
| `src/interpreter/execute/memory.rs` | MMIO intercept in read/write_memory |
| `src/device/tile.rs` | Extend `read_register()` for lock/DMA reads |

## Verification

- Existing `cargo test --lib` must pass
- New unit tests for control register write and MMIO read/write
- SRS tests continue to pass (preamble sets config via instructions now)
