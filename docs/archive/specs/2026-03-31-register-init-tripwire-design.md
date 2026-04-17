# Register Initialization Tripwire

**Date**: 2026-03-31
**Status**: Approved
**Scope**: Fill all register state with a configurable sentinel pattern before
normal initialization, so any register read before proper setup produces a
deterministic, recognizable value instead of runtime-varying garbage.

## Problem

When code reads an uninitialized register, the value depends on whatever
happened to be in that storage from a prior execution. This produces:

- Non-deterministic results across runs and runtimes (HW vs aiesim vs EMU)
- Bugs that appear as "format mismatches" or "data disagreements" rather than
  the actual root cause (uninitialized read)
- Hours of debugging chasing phantom divergence

Today's sparse MAC session demonstrated this: a half-load bug left the upper
64 lanes of a 128-element sparse vector uninitialized. Real hardware and
aiesimulator diverged because their uninitialized register contents differed,
leading us to investigate data packing formats for hours before discovering
the actual bug was a missing second pop.

## Design

### Fill-Then-Init Pattern

The tripwire inserts a pattern-fill step at the very beginning of register
file construction, before any meaningful initialization. The normal init
code then overwrites the fields it's responsible for. Anything that doesn't
get initialized retains the sentinel.

```
ExecutionContext::new_for_tile(col, row):
  1. Blast every register array with sentinel pattern    <-- NEW
  2. scalar.set_core_id(col, row)                        <-- existing
  3. SP = 0x70000                                        <-- existing
  4. PC = 0                                              <-- existing
  5. ... rest of initialization unchanged ...
```

If a new register field is added later and not initialized, it automatically
contains the sentinel. No action required to "opt in" new state.

### Configuration

Environment variable `XDNA_EMU_REG_INIT` controls the fill pattern:

| Value | Behavior |
|-------|----------|
| (unset) | Default: `0xDEADBEEF` |
| `deadbeef` | Fill with `0xDEADBEEF` |
| `zero` | Fill with `0x00000000` |
| `<hex>` | Fill with arbitrary 32-bit pattern (e.g., `cafebabe`, `ff`) |

The value is read once at context creation. No caching complexity needed;
context creation is not a hot path.

### What Gets Filled

Every raw storage array in every register file:

| Register File | Storage | Fill Unit |
|---------------|---------|-----------|
| Scalar | `regs: [u32; 48]` | pattern as u32 |
| Vector | `regs: [[u32; 8]; 32]` | pattern as u32 |
| Accumulator | `regs: [[u64; 8]; 18]` | pattern doubled to u64 (`0xDEADBEEF_DEADBEEF`) |
| Pointer | `regs: [u32; 8]` | pattern as u32 |
| Modifier | `regs: [u32; 32]` | pattern as u32 |
| Mask | `regs: [[u32; 4]; 4]` | pattern as u32 |

Scalar fields in `ExecutionContext` also get filled:

| Field | Fill |
|-------|------|
| `pc` | pattern as u32 |
| `sp_value` | pattern as u32 |
| `cycles`, `instructions`, `stall_cycles` | 0 (counters, not registers) |
| `halted` | false (control state, not register) |

Counters and control booleans are exempt -- they're not hardware register
state, and filling them with garbage would break the emulator's own logic
rather than catch kernel bugs.

### What Overwrites It

Existing initialization code runs after the fill and overwrites specific
fields. This is unchanged:

- `scalar.set_core_id(col, row)` -- reg 37
- `sp_value = 0x70000` (or provided stack address)
- `pc = 0`
- `flags = Flags::default()` (zeroed)
- `srs_config = SrsConfig::default()` (zeroed)
- Timing context, pending writes, etc. (emulator internals, not HW registers)

Everything else retains the sentinel until the kernel's preamble writes it.

### Implementation

Each register file struct gets a `fill_pattern(&mut self, pattern: u32)` method:

```rust
impl ScalarRegisterFile {
    pub fn fill_pattern(&mut self, pattern: u32) {
        self.regs.fill(pattern);
    }
}

impl VectorRegisterFile {
    pub fn fill_pattern(&mut self, pattern: u32) {
        for reg in &mut self.regs {
            reg.fill(pattern);
        }
    }
}

impl AccumulatorRegisterFile {
    pub fn fill_pattern(&mut self, pattern: u32) {
        let wide = (pattern as u64) << 32 | (pattern as u64);
        for reg in &mut self.regs {
            reg.fill(wide);
        }
    }
}
// ... same pattern for Pointer, Modifier, Mask
```

`ExecutionContext` gets a `fill_all_registers(&mut self, pattern: u32)` that
calls each file's method plus fills `pc` and `sp_value`.

The env var is read in `ExecutionContext::new_for_tile()`:

```rust
fn reg_init_pattern() -> u32 {
    match std::env::var("XDNA_EMU_REG_INIT").as_deref() {
        Ok("zero") => 0x00000000,
        Ok(hex) => u32::from_str_radix(hex, 16).unwrap_or(0xDEADBEEF),
        Err(_) => 0xDEADBEEF,
    }
}
```

### Testing

One unit test:

1. Create `ExecutionContext::new_for_tile(0, 0)` (uses default deadbeef)
2. Verify `scalar.read(0) == 0xDEADBEEF` (r0 is not touched by init)
3. Verify `scalar.read(CORE_ID_REG_INDEX) != 0xDEADBEEF` (overwritten by init)
4. Verify `vector.read(0) == [0xDEADBEEF; 8]`
5. Verify `pc == 0` (overwritten by init)

### Files Changed

| File | Change |
|------|--------|
| `src/interpreter/state/registers.rs` | Add `fill_pattern()` to each register file |
| `src/interpreter/state/context.rs` | Add `fill_all_registers()`, call in `new()`/`new_for_tile()`/`with_stack()`, add `reg_init_pattern()` |
