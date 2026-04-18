# CRITICAL: Core Data Address Space is Wrong

## Status: CONFIRMED BUG (2026-03-07)

The emulator's `decode_data_address()` uses a fictional 18-bit address space
that does not match the hardware. This silently corrupts cross-tile memory
accesses and is likely a root cause of persistent routing/offset bugs.

## Hardware Reality (from aie-rt + Chess BCF)

The core uses a **20-bit data address space** with cardinal directions:

```
0x00000-0x03FFF: Program memory (16KB, core view)
0x04000-0x3FFFF: Reserved/unused
0x40000-0x4FFFF: South neighbor data memory (CardDir 4)
0x50000-0x5FFFF: West neighbor data memory  (CardDir 5)
0x60000-0x6FFFF: North neighbor data memory (CardDir 6)
0x70000-0x7FFFF: East/local data memory     (CardDir 7)
```

Routing formula (aie-rt `_XAie_GetTargetTileLoc`, xaie_elfloader.c:139):
```c
CardDir = (u8)(Addr / CoreMod->DataMemSize);  // DataMemSize = 0x10000
```

For AIE2 (`IsCheckerBoard = 0`), RowParity is forced to 1:
- CardDir 4 (South): row - 1
- CardDir 5 (West): col - 1
- CardDir 6 (North): row + 1
- CardDir 7 (East): same tile (local) -- ALWAYS local in AIE2

### Architectural asymmetry

AIE2 cores can only directly access 3 neighbors + local via load/store:
South, West, North, and East(local). There is NO direct load/store path
to the eastern neighbor. The memory module is always on the east side of
the core. The eastern neighbor's core sees our memory as its "West."
Data flows eastward via DMA/stream switches, not direct memory access.

AIE1 (checkerboard) alternates: odd rows have East=local, even rows have
West=local. Both directions are reachable across two adjacent rows.

## What the Emulator Does (WRONG)

File: `src/interpreter/execute/memory.rs`, `decode_data_address()`:

```rust
const DATA_MEMORY_ADDRESS_LIMIT: u32 = 0x3FFFF;  // 18-bit

fn decode_data_address(addr: u32) -> (u32, usize) {
    if addr > DATA_MEMORY_ADDRESS_LIMIT {
        // "Linker/system address" -- strip to local 16-bit offset
        (0, (addr & LOCAL_MEMORY_MASK) as usize)
    } else {
        let quadrant = (addr >> 16) & 0x3;
        let offset = (addr & LOCAL_MEMORY_MASK) as usize;
        (quadrant, offset)
    }
}
```

### Bugs in this function:

1. **0x40000-0x6FFFF silently becomes local**: Any South/West/North
   neighbor access via the hardware's actual address space gets masked
   to a local offset, corrupting local memory instead of routing to
   the neighbor tile.

2. **Wrong quadrant numbering**: Emulator uses 0=Local, 1=West, 2=North,
   3=East. Hardware uses 4=South, 5=West, 6=North, 7=East(local).
   South is missing entirely from the emulator.

3. **Mislabeled as "linker convention"**: The 0x70000 address is the
   hardware's East cardinal direction, not a linker convention. Comments
   throughout the file are wrong.

4. **18-bit vs 20-bit**: The comment says "AGU generates 20-bit addresses"
   but then uses an 18-bit limit (0x3FFFF). The actual AGU address space
   is 20 bits (0x00000-0x7FFFF, or at least 0x40000-0x7FFFF for data).

### Impact

- Any cross-tile load/store using hardware addresses silently writes to
  local memory instead
- Cross-tile access only works "by accident" when the emulator's 18-bit
  quadrant scheme (0x10000-0x3FFFF) happens to be used -- but compiled
  code uses 0x40000-0x7FFFF addresses
- Explains persistent cross-tile offset and routing bugs

## Affected Code

| File | What's wrong |
|------|-------------|
| `src/interpreter/execute/memory.rs` | `decode_data_address()` -- wrong address space model |
| `src/interpreter/execute/memory.rs` | `NeighborMemory` quadrant mapping comments |
| `src/interpreter/timing/memory.rs` | `MemoryQuadrant` enum -- missing South, wrong numbering |
| `src/device/registers_spec.rs` | `AIE_DATA_MEMORY_BASE` labeled "linker convention" -- it's hardware |
| `src/interpreter/test_runner.rs` | ELF loader uses `AIE_DATA_MEMORY_BASE` as subtraction base |

## Correct Implementation

```rust
/// Cardinal direction from core data address (aie-rt formula).
/// DataMemSize = 0x10000 for AIE2.
fn decode_data_address(addr: u32) -> (CardinalDir, usize) {
    let card_dir = addr / DATA_MEM_SIZE;  // 0x10000
    let offset = (addr & (DATA_MEM_SIZE - 1)) as usize;
    match card_dir {
        4 => (CardinalDir::South, offset),
        5 => (CardinalDir::West, offset),
        6 => (CardinalDir::North, offset),
        7 => (CardinalDir::East, offset),  // Local for AIE2
        _ => /* invalid or PM range */
    }
}
```

## ArchModel Constants Needed

These should be in `arch::compute::*` (generated from ArchModel):
- `DATA_MEM_ADDR` = 0x40000 (from aie-rt CoreMod.DataMemAddr)
- `DATA_MEM_SIZE` = 0x10000 (already exists as MEMORY_SIZE)
- `DATA_MEM_SHIFT` = 16 (from aie-rt CoreMod.DataMemShift)
- `IS_CHECKERBOARD` = false (from aie-rt CoreMod.IsCheckerBoard)
- `PROGRAM_MEMORY_HOST_OFFSET` = 0x20000 (from AM025 register offset)
- Cardinal direction constants: SOUTH=4, WEST=5, NORTH=6, EAST=7

## Source References

- `aie-rt/driver/src/core/xaie_elfloader.c:124-183` -- `_XAie_GetTargetTileLoc`
- `aie-rt/driver/src/global/xaiemlgbl_reginit.c:2318-2326` -- AieMlCoreMod init
- `aietools/data/aie_ml/lib/me.bcf` -- BCF memory map (0x40000-0x7FFFF)
- `aie-rt/driver/src/global/xaiegbl_regdef.h:280-309` -- XAie_CoreMod struct
