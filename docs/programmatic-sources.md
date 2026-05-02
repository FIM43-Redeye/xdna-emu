# Programmatic Sources for NPU Emulation

This document maps every programmatic source available in our local toolchain
(mlir-aie, llvm-aie, aie-rt) for driving the emulator's implementation. The
goal is to replace hand-coded constants, heuristic decoders, and documentation
transcriptions with data derived directly from the compiler and runtime
toolchain -- the same source of truth that the real hardware toolchain uses.

**Created**: 2026-02-14
**Status**: Research complete; the toolchain-derivation principle this
document advocates is now the project's working policy (see
[CLAUDE.md](../CLAUDE.md) "Correctness Principle"). The
`xdna-archspec` crate is the runtime arch source-of-truth, the
TableGen pipeline emits decoder bytecode at build time, and the BD
field parsing reads the AM025 register database JSON. The "116-variant
Operation enum" mentioned at the bottom of this document is gone:
SemanticOp now lives in `crates/xdna-archspec/src/aie2/isa/types.rs`
and has ~48 variants resolved from TableGen.

---

## Table of Contents

1. [Overview and Priority Tiers](#overview-and-priority-tiers)
2. [Source 1: JSON Register Database](#source-1-json-register-database-aie_registers_aie2json)
3. [Source 2: AIETargetModel (Device Model)](#source-2-aietargetmodel-device-model)
4. [Source 3: aie-rt Hardware Driver Library](#source-3-aie-rt-hardware-driver-library)
5. [Source 4: MCCodeEmitter (Register Encoding)](#source-4-mccodeemitter-register-encoding)
6. [Source 5: TableGen Instruction Definitions](#source-5-tablegen-instruction-definitions-already-used)
7. [Source 6: Scheduling Model](#source-6-scheduling-model)
8. [Source 7: Calling Convention and ABI](#source-7-calling-convention-and-abi)
9. [Source 8: Intrinsics and Builtins](#source-8-intrinsics-and-builtins)
10. [Current State: Data-Driven vs Hand-Coded](#current-state-data-driven-vs-hand-coded)
11. [Implementation Tiers](#implementation-tiers)

---

## Overview and Priority Tiers

Our local toolchain contains three major programmatic knowledge sources:

| Source | Location | Format | What It Defines |
|--------|----------|--------|-----------------|
| **llvm-aie** (Peano) | `../llvm-aie/` | TableGen (.td), C++ | ISA, register encoding, scheduling, ABI |
| **mlir-aie** | `../mlir-aie/` | C++, JSON, Python | Device model, routing, register database |
| **aie-rt** | `../aie-rt/` (official Xilinx) | C headers/source | Register offsets, DMA/lock implementations, tests, AIE2P |

Implementation priority (revised 2026-02-14):

| Tier | Focus | Key Sources | Impact |
|------|-------|-------------|--------|
| **Tier 2** (first) | Device model & registers | JSON regdb, AIETargetModel | Replace ~600 lines of hand-coded constants |
| **Tier 1** (second) | Data-driven decoder | MCCodeEmitter inverse | Fix operand decoding accuracy |
| **Tier 3** (third) | Cycle-accurate timing | AIE2Schedule.td | Real pipeline latencies |
| **Tier 4** (goal) | Full semantic coverage | Intrinsics, patterns | Auto-generate execution logic |

---

## Source 1: JSON Register Database (aie_registers_aie2.json)

**The single most valuable resource we have.**

### Location

```
mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json  (source)
mlir-aie/install/lib/regdb/aie_registers_aie2.json      (installed)
```

### Metadata

- **Version**: AM025-2024-11-13-1.1
- **Source**: Parsed from AMD AM025 HTML documentation
- **Parsed date**: 2025-12-04
- **Size**: 2.6 MB
- **Format**: JSON

### Contents

| Module | Registers | Bit Fields | What It Covers |
|--------|-----------|------------|----------------|
| core | 457 | 1,245 | Program memory, core control/status, events, trace, debug |
| memory | 203 | 747 | Data memory, DMA BDs, DMA channels, locks, events |
| memory_tile | 718 | 2,843 | MemTile DMA (48 BDs), locks (64), stream switch |
| shim | 428 | 1,577 | Shim DMA, NOC interface, stream switch, locks |
| **Total** | **1,806** | **6,412** | |

### Schema

```json
{
  "version": "AM025-2024-11-13-1.1",
  "modules": {
    "<module_name>": {
      "registers": [
        {
          "name": "DMA_BD0_0",
          "offset": "0x000001D000",
          "width": 32,
          "bit_fields": [
            {
              "name": "Buffer_Length",
              "bits": "13:0",
              "bit_range": [0, 13],
              "type": "rwNormal read/write"
            },
            {
              "name": "Base_Address",
              "bits": "27:14",
              "bit_range": [14, 27],
              "type": "rwNormal read/write"
            }
          ]
        }
      ]
    }
  }
}
```

### Key Register Groups (Memory Module)

**DMA Buffer Descriptors** (6 words per BD, 16 BDs, stride 0x20):

| Register | Offset | Fields |
|----------|--------|--------|
| DMA_BD0_0 | 0x1D000 | Base_Address (27:14), Buffer_Length (13:0) |
| DMA_BD0_1 | 0x1D004 | Enable_Compression (31), Enable_Packet (30), Out_Of_Order_BD_ID (29:24), Packet_ID (23:19), Packet_Type (18:16) |
| DMA_BD0_2 | 0x1D008 | D1_Stepsize (25:13), D0_Stepsize (12:0) |
| DMA_BD0_3 | 0x1D00C | D1_Wrap (28:21), D0_Wrap (20:13), D2_Stepsize (12:0) |
| DMA_BD0_4 | 0x1D010 | Iteration_Current (24:19), Iteration_Wrap (18:13), Iteration_Stepsize (12:0) |
| DMA_BD0_5 | 0x1D014 | TLAST_Suppress (31), Next_BD (30:27), Use_Next_BD (26), Valid_BD (25), Lock_Rel_Value (24:18), Lock_Rel_ID (16:13), Lock_Acq_Enable (12), Lock_Acq_Value (11:5), Lock_Acq_ID (3:0) |

**DMA Channel Control**:

| Register | Offset | Key Fields |
|----------|--------|------------|
| DMA_S2MM_0_Ctrl | 0x1DE00 | FoT_Mode (17:16), Controller_ID (15:8), Reset (1) |
| DMA_S2MM_0_Start_Queue | 0x1DE04 | Enable_Token_Issue (31), Repeat_Count (23:16), Start_BD_ID (3:0) |
| DMA_S2MM_1_Ctrl | 0x1DE08 | (same layout) |
| DMA_S2MM_1_Start_Queue | 0x1DE0C | (same layout) |
| DMA_MM2S_0_Ctrl | 0x1DE10 | (same layout) |
| DMA_MM2S_0_Start_Queue | 0x1DE14 | (same layout) |
| DMA_MM2S_1_Ctrl | 0x1DE18 | (same layout) |
| DMA_MM2S_1_Start_Queue | 0x1DE1C | (same layout) |

**Lock Registers** (16 locks, stride 0x10):

| Register | Offset | Fields |
|----------|--------|--------|
| Lock0_value | 0x1F000 | Lock_value (5:0) |
| Lock1_value | 0x1F010 | Lock_value (5:0) |
| ... | ... | ... |
| Lock15_value | 0x1F0F0 | Lock_value (5:0) |

### How to Consume

Load at build time or runtime. Deserialize into Rust structs:

```rust
struct RegisterDef {
    name: String,
    offset: u32,
    width: u8,
    fields: Vec<BitFieldDef>,
}

struct BitFieldDef {
    name: String,
    msb: u8,
    lsb: u8,
}
```

Generate field extraction functions from the definitions rather than
hand-coding masks and shifts.

### What This Replaces

- `src/device/registers_spec.rs` -- all hand-coded BD field masks/shifts
- `src/device/dma/bd.rs` -- BD word parsing logic (generated from JSON)
- `src/device/aie2_spec.rs` -- lock addresses, DMA addresses, register offsets

---

## Source 2: AIETargetModel (Device Model)

### Location

```
mlir-aie/include/aie/Dialect/AIE/IR/AIETargetModel.h   (interface, 320+ lines)
mlir-aie/lib/Dialect/AIE/IR/AIETargetModel.cpp          (implementation, 969 lines)
```

### Device Enumeration

```
TK_AIE1_VC1902          -- Versal FPGA (AIE1)
TK_AIE2_VE2302          -- Versal VE2302 (AIE2, 17 cols x 4 rows)
TK_AIE2_VE2802          -- Versal VE2802 (AIE2, 38 cols x 11 rows)
TK_AIE2_NPU1_1Col       -- Phoenix 1-column virtual
TK_AIE2_NPU1_2Col       -- Phoenix 2-column virtual
TK_AIE2_NPU1_3Col       -- Phoenix 3-column virtual
TK_AIE2_NPU1_4Col       -- Phoenix 4-column (full array)
TK_AIE2_NPU2            -- Strix Point (AIE2P, 8 cols)
TK_AIE2_NPU2_1Col..7Col -- Strix Point virtual variants
```

### Virtual Interface (Key Methods)

```cpp
// Geometry
int columns() const;
int rows() const;

// Tile classification
bool isCoreTile(int col, int row) const;
bool isMemTile(int col, int row) const;
bool isShimNOCTile(int col, int row) const;
bool isShimPLTile(int col, int row) const;

// Resources per tile
uint32_t getLocalMemorySize() const;              // 0x10000 = 64KB (AIE2)
uint32_t getAccumulatorCascadeSize() const;        // 384 bits (AIE2)
uint32_t getNumLocks(int col, int row) const;      // 16 (core), 64 (memtile)
uint32_t getMaxLockValue() const;                  // 0x3F = 63 (AIE2 semaphore)
uint32_t getNumBDs(int col, int row) const;        // 16 (core), 48 (memtile)
uint32_t getNumBanks(int col, int row) const;      // memory bank count

// Memory addressing
uint32_t getMemInternalBaseAddress(TileID src) const;
uint32_t getMemSouthBaseAddress() const;
uint32_t getMemWestBaseAddress() const;
uint32_t getMemNorthBaseAddress() const;
uint32_t getMemEastBaseAddress() const;

// Lock addressing
optional<uint32_t> getLocalLockAddress(uint32_t lockId, TileID tile) const;
optional<uint32_t> getLockLocalBaseIndex(int localCol, int localRow,
                                         int lockCol, int lockRow) const;

// DMA addressing
uint64_t getDmaBdAddress(int col, int row, uint32_t bd_id, ...) const;
uint32_t getDmaBdAddressOffset(int col, int row) const;
uint32_t getDmaControlAddress(int col, int row, int channel, Direction) const;

// Stream switch
uint32_t getNumDestSwitchboxConnections(int col, int row, WireBundle) const;
uint32_t getNumSourceSwitchboxConnections(int col, int row, WireBundle) const;
bool isLegalTileConnection(int col, int row,
                          WireBundle src, int srcChan,
                          WireBundle dst, int dstChan) const;

// Memory affinity (which cores can access which memory)
bool isLegalMemAffinity(int coreCol, int coreRow, int memCol, int memRow) const;
optional<TileID> getMemWest(TileID src) const;
optional<TileID> getMemEast(TileID src) const;
optional<TileID> getMemNorth(TileID src) const;
optional<TileID> getMemSouth(TileID src) const;
```

### Model Properties (Bitfield)

```cpp
UsesSemaphoreLocks       = 1 << 0   // AIE2 uses semaphore locks (vs binary)
IsNPU                    = 1 << 1   // NPU device (special CDO handling)
IsVirtualized            = 1 << 2   // Virtualized column subset
UsesMultiDimensionalBDs  = 1 << 3   // Multi-dim buffer descriptors
```

### NPU1 Concrete Values (Phoenix, Our Target)

```
columns = 5, rows = 6
Row 0: Shim (ShimNOC at cols 0,1; ShimPL at cols 2,3,4 -- or vice versa)
Row 1: MemTile (1 row)
Rows 2-5: Core tiles (4 rows)

getLocalMemorySize()        = 0x10000 (64 KB)
getAccumulatorCascadeSize() = 384 bits
getNumLocks(core)           = 16
getNumLocks(memtile)        = 64
getMaxLockValue()           = 63
getNumBDs(core)             = 16
getNumBDs(memtile)          = 48
getMemTileSize()            = 0x80000 (512 KB)

Lock base addresses:
  Core tile:    0x1F000
  MemTile:      0xC0000
  Shim tile:    0x14000
Lock stride:    0x10 (16 bytes)
```

### What This Replaces

- `xdna_archspec::runtime` (`ArchConfig` trait, `ModelConfig`) -- all tile classification and resource queries; `src/device/port_layout.rs` hosts the runtime-side `PortLayout` extension trait
- `src/device/aie2_spec.rs` -- memory sizes, lock counts, BD counts, port layouts

### How to Consume

Two options:
1. **Build-time extraction**: Write a C++ tool that instantiates the
   AIETargetModel, queries all methods, and outputs JSON/Rust constants.
2. **Manual transcription with validation**: Keep our Rust constants but add
   a build-time validation step that checks them against AIETargetModel queries.

Option 1 is cleaner but requires linking against mlir-aie. Option 2 is more
pragmatic -- we already have most values correct, we just need confidence.

---

## Source 3: aie-rt Hardware Driver Library

### Location

```
aie-rt/driver/src/               -- Official Xilinx (branch xlnx_rel_v2025.2)
  global/xaiemlgbl_params.h      -- All register offset #defines (~10K lines)
  global/xaiemlgbl_reginit.c     -- Structured register property tables
  global/xaie2psgbl_params.h     -- AIE2P register definitions (future target)
  global/xaie2psgbl_reginit.c    -- AIE2P property tables (future target)
  global/xaiegbl_defs.h          -- Device type constants, tile types
  global/xaiegbl_regdef.h        -- Type definitions for register properties
  dma/xaie_dma_aieml.c           -- DMA BD read/write (56K lines)
  locks/xaie_locks_aieml.c       -- Lock acquire/release implementation
  device/xaie_device_aieml.c     -- Device initialization
  core/xaie_core_aieml.c         -- Core enable/disable/status
  routing/xaie_routing.c         -- Auto-routing (not in mlir-aie's fork)
aie-rt/driver/tests/utest/       -- Unit tests (valuable reference)
  test_dma_aieml.cpp             -- DMA BD programming tests
  test_locks_aieml.cpp           -- Lock acquire/release tests
aie-rt/driver/examples/
  xaie_tile_dma_loopback.c       -- DMA loopback example
```

### Key Constants

```c
// Device generations
XAIE_DEV_GEN_AIE        = 1   // AIE1
XAIE_DEV_GEN_AIEML      = 2   // AIE2
XAIE_DEV_GEN_AIE2IPU    = 3   // AIE2/NPU1 (Phoenix)
XAIE_DEV_GEN_AIE2P      = 4   // AIE2P (Strix)

// Tile types
XAIEGBL_TILE_TYPE_AIETILE  = 0
XAIEGBL_TILE_TYPE_SHIMNOC  = 1
XAIEGBL_TILE_TYPE_SHIMPL   = 2
XAIEGBL_TILE_TYPE_MEMTILE  = 3

// BD word counts
XAIEML_TILEDMA_NUM_BD_WORDS    = 6
XAIEML_SHIMDMA_NUM_BD_WORDS    = 8
XAIEML_MEMTILEDMA_NUM_BD_WORDS = 8
```

### Register Property Tables (xaiemlgbl_reginit.c)

This file contains structured C initialization of register properties:

```c
// Example: Tile DMA BD0 properties
.AieMlMultiDimAddr.DmaDimProp[0].StepSize.Idx = 2U,  // Word 2
.AieMlMultiDimAddr.DmaDimProp[0].StepSize.Lsb = 0U,  // bit 0
.AieMlMultiDimAddr.DmaDimProp[0].StepSize.Mask = 0x00001FFFU,
```

### Relationship to JSON Register Database

The JSON register database (Source 1) is a **higher-level, more consumable**
version of the same information in these C headers. For the emulator, prefer
the JSON. The C source is useful for:
- Understanding *behavior* (DMA state machines, lock acquire semantics)
- Cross-validating the JSON (if something seems wrong, check the C)
- Understanding shim column type detection (`(col % 4) in {0,1}` vs `{2,3}`)

---

## Source 4: MCCodeEmitter (Register Encoding)

### Location

```
llvm-aie/llvm/lib/Target/AIE/MCTargetDesc/
  AIE2MCCodeEmitterRegOperandDef.h  -- COMPLETE encoder implementations (~150 lines)
  AIE2MCCodeEmitterDeclaration.h    -- Method signatures (74 lines)
  AIE2MCCodeEmitter.cpp             -- Entry point (63 lines)
  AIEBaseMCCodeEmitter.h            -- Base class (100+ lines)
```

### Composite Encoder Functions (Must Invert for Decoding)

Each function maps a register + register class to a raw bit field value.
We need the inverse: raw bit field -> register operand.

| Encoder Function | Composite Group | Discriminant Bits |
|-----------------|-----------------|-------------------|
| `getmLdaCgOpValue` | mLdaCg, mLdbCg | 2 low bits: 00=scalar, 10=modifier, 1101=pointer |
| `getmLdaSclOpValue` | mLdaScl, mSclSt, mSclMS | Same as LdaCg + lr=0b0000101 |
| `getmMvSclSrcOpValue` | mMvSclSrc, mMvSclDst, mMvSclDstCg | 2-4 low bits, special regs by HWEncoding |
| `getmAluCgOpValue` | mAluCg | 1 low bit: 0=scalar, LC=0b000001 |
| `getmMvAMWQDstOpValue` | mMvAMWQDst | Vector/accumulator composite |
| `getmMvAMWQSrcOpValue` | mMvAMWQSrc | Vector/accumulator composite |
| `getmMvBMXSrcOpValue` | mMvBMXSrc | 512-bit vector/accumulator |
| `getmMvBMXDstOpValue` | mMvBMXDst | 512-bit vector/accumulator |
| `geteRS4OpValue` | eRS4 | Scaled scalar register |
| `getmShflDstOpValue` | mShflDst | Shuffle destination |
| `getmWm_1OpValue` | mWm_1 | Vector register minus 1 |
| `getmQXHLbOpValue` | mQXHLb | Quarter/half vector select |

### Encoding Patterns (Authoritative)

**mLdaCg (load destination composite group)**:
```
LC              -> 0b0010101 (fixed encoding)
Pointer pN      -> (HWEncoding << 4) | 0b1101
Scalar rN       -> (HWEncoding << 2) | 0b00
Modifier M      -> ((HWEncoding | 0b00000) << 2) | 0b10
Modifier DN     -> ((HWEncoding | 0b01000) << 2) | 0b10
Modifier DJ     -> ((HWEncoding | 0b10000) << 2) | 0b10
Modifier DC     -> ((HWEncoding | 0b11000) << 2) | 0b10
```

**mMvSclSrc (move scalar source composite group)**:
```
Special regs    -> HWEncoding as-is (LC=87, SP=103, lr=39, LS=7, LE=71, DP=23, CORE_ID=55)
Pointer pN      -> (HWEncoding << 4) | 0b0011     (note: 0011 not 1101!)
Scalar rN       -> (HWEncoding << 2) | 0b00
Modifier M/DN/DJ/DC -> ((HWEncoding | mode) << 2) | 0b10
Status eS       -> (HWEncoding << 5) | 0b01011
Control mCRm    -> (HWEncoding << 3) | 0b001
Status mSRm     -> (HWEncoding << 3) | 0b101
```

### What This Replaces

- ~300 lines of heuristic operand decoding in `src/interpreter/decode/decoder.rs`
- The `decode_generic_operand()` function (guesses types from field names)
- Special-case blocks for mLdaScl, mSclSt, mMvSclSrc, mLdaCg, ag_* fields

### Implementation Plan

See the existing plan file: `.claude/plans/misty-hopping-clock.md`

---

## Source 5: TableGen Instruction Definitions (Already Used)

### Location

```
llvm-aie/llvm/lib/Target/AIE/
  AIE2GenInstrInfo.td       (658 lines, auto-generated instruction defs)
  AIE2GenInstrFormats.td    (1,194 lines, format class hierarchy)
  AIE2InstrInfo.td          (629 lines, main instruction mnemonics)
  AIE2GenFixupInstrInfo.td  (2,478 lines, vector/fixup instructions)
  AIE2CompositeFormats.td   (1,340 lines, VLIW composite formats)
  AIE2InstrPatterns.td      (1,139 lines, DAG selection patterns)
  AIE2Slots.td              (141 lines, VLIW slot definitions)
```

### What We Already Extract

- Instruction encodings (fixed bits, operand fields, widths)
- VLIW slot properties and composite format definitions
- Instruction attributes (mayLoad, mayStore, hasSideEffects, Defs, Uses)
- Semantic patterns (Pat<(add ...), (ADD ...)> -> SemanticOp)
- Split field encoding via FieldFragment abstraction

### What We Could Additionally Extract

- `OperandDef.reg_class` -> `OperandType` mapping (planned in Tier 1)
- Instruction itinerary class assignment (for Tier 3 timing)
- Implicit register uses/defs (partially done)
- Multi-slot pseudo instructions (vector ops that expand to real ops)

### Modules

```
src/tablegen/mod.rs              -- Crate-level interface
src/tablegen/parser.rs           -- Regex-based .td parser
src/tablegen/tblgen_records.rs   -- llvm-tblgen --print-records parser
src/tablegen/resolver.rs         -- Converts parsed data to InstrEncoding
src/tablegen/types.rs            -- Type definitions
```

---

## Source 6: Scheduling Model

### Location

```
llvm-aie/llvm/lib/Target/AIE/AIE2Schedule.td   (1,181 lines)
```

### Contents

**Functional Units** (hardware resources):
```
Structural:  UPPER_SRS, UPS_UNIT, PROC_BUS, STORE_UNIT, LOAD_UNIT_A, SEMAPHORE
Scalar ports: R_RV_PORT, RS_WM_PORT, R_WX_PORT, R_WA_PORT
Address ports: P_RM_PORT, P_WM_PORT, M_WM_PORT, DJ_WM_PORT, DN_WM_PORT, DC_WM_PORT
Vector ports: W_RS_PORT, W_WA_PORT, W_WM_PORT
Accumulator:  CM_RM_PORT, CM_WM_PORT, CM_WA_PORT
Special:      PART_WORD_STORE, DONE_UNIT, EXEC_TRACE_UNIT
```

**Instruction Itinerary Classes** (200+ classes):
```
ALU:     II_ABS, II_ADD, II_AND, II_ASHL, II_CLB, II_CLZ, II_DIVS, II_MUL, II_SUB
Branch:  II_J, II_JL, II_JNZ, II_JZ, II_RET
Load:    II_LDA, II_LDA_POST_1D, II_LDA_POST_2D, II_LDA_POST_3D
Store:   II_ST, II_ST_MS, II_ST_POST_1D, II_ST_POST_2D, II_ST_POST_3D
Move:    II_MOV, II_MOVA, II_MOVX, II_MOV_SCL, II_MOV_SS
Vector:  II_VADD, II_VADDMAC, II_VMUL, II_VACCf
Sync:    II_ACQ, II_REL, II_DONE
```

Each itinerary class defines:
- Which functional units are used (and for how many cycles)
- Pipeline stage assignments
- Read/write latencies for operands
- Bypass relationships

### What This Replaces

- Estimated timing constants in `src/device/aie2_spec.rs`
  (LATENCY_SCALAR_ADD_SUB, LATENCY_DATA_MEMORY, DMA_BD_SETUP_CYCLES, etc.)
- Hand-coded pipeline assumptions in the execution engine

### How to Consume

Parse alongside the instruction definitions we already parse from TableGen.
Each instruction's `Itinerary` field references one of these classes. Map
the itinerary to cycle counts and resource requirements.

---

## Source 7: Calling Convention and ABI

### Location

```
llvm-aie/llvm/lib/Target/AIE/AIE2CallingConv.td    (201 lines)
llvm-aie/llvm/lib/Target/AIE/AIE2FrameLowering.h   (42 lines)
llvm-aie/llvm/lib/Target/AIE/AIE2RegisterInfo.cpp   (150+ lines)
```

### Key Definitions

**Argument registers**:
```
Scalar:  r0-r7   (8 GPRs)
Pointer: p0-p5   (6 pointer registers)
Vector:  w/W/X/Y registers (by type width)
Accum:   AM/BM/CM registers (by accumulator width)
```

**Return registers**: r0-r1 (scalar), p0-p1 (pointer)

**Stack alignment**: 32 bytes

**Reserved registers**: SP, lr, LC, LS, LE, CORE_ID, mCRm, mSRm

**Callee-saved**: CSR_AIE2_SaveList (defined in AIE2RegisterInfo.cpp)

**Sticky registers** (persistent state across calls):
srCompr_uf, srSparse_of, srF2FFlags, srF2IFlags, srFPFlags, srSRS_of, srUPS_of

### What This Replaces

- Hard-coded register assumptions in `src/interpreter/` stack frame handling
- Function call/return register setup

---

## Source 8: Intrinsics and Builtins

### Location

```
llvm-aie/clang/include/clang/Basic/BuiltinsAIE.def    (AIE1, 100+ lines)
llvm-aie/clang/include/clang/Basic/BuiltinsAIE2P.def  (AIE2P, 100+ lines)
```

Note: No BuiltinsAIE2.def found -- AIE2 builtins may be in a different
location or may share with AIE1.

### Categories

- **Synchronization**: `__builtin_aie_event`, `__builtin_aie_acquire`, `__builtin_aie_release`
- **Streams**: `__builtin_aie_get_ss`, `__builtin_aie_put_ms`, `__builtin_aie_packet_header`
- **Bit ops**: `__builtin_aie_bitget`, `__builtin_aie_bitset`
- **Math**: `__builtin_aie_sqrt_flt_flt`, `__builtin_aie_inv_flt_flt`
- **Vector**: `__builtin_aie_vfpmul`, `__builtin_aie_mul4_*`, `__builtin_aie_mac16_*`
- **Pack/Unpack** (AIE2P): `__builtin_aie2p_pack_I512_I8_I16`
- **MAC** (AIE2P): `__builtin_aie2p_I1024_I1024_ACC2048_addmac_conf`

### What This Could Replace

- ~3,000 lines of hand-coded vector operation semantics in
  `src/interpreter/execute/vector.rs`
- The builtin signatures define input types, output types, and accumulator
  widths -- enough to auto-generate execution logic for many operations.

---

## Current State: Data-Driven vs Hand-Coded

| Area | Lines | Source | Data-Driven? | Confidence |
|------|-------|--------|-------------|------------|
| TableGen extraction | ~2,000 | llvm-aie .td | YES | HIGH |
| Device constants | ~200 | AM020/AM025 docs | Transcribed | HIGH |
| Register addresses | ~400 | AM025 docs | Transcribed | HIGH |
| DMA BD parsing | ~500 | AM025 docs | Transcribed | HIGH |
| Lock behavior | ~200 | AM025 + deduction | Hand-coded | MEDIUM |
| Operand decoding | ~300 | Heuristics | Hand-coded | LOW |
| Instruction semantics | ~8,000 | Hand-coded | Partial (SemanticOp) | MEDIUM |
| Instruction timing | ~100 | Estimated | Hand-coded | LOW |
| Stream switch ports | ~150 | AM025 docs | Transcribed | HIGH |

**Key insight**: Our "transcribed from docs" code is generally correct (well-
commented with AM025 section citations), but fragile. The JSON register
database contains the same information in machine-readable form, making it
possible to validate or replace the transcriptions automatically.

---

## Implementation Tiers

### Tier 2: Device Model and Registers (FIRST PRIORITY)

**Goal**: Replace hand-coded register constants with data loaded from the
JSON register database. Validate device model constants against AIETargetModel.

**Key deliverables**:
1. JSON register database loader (`src/device/regdb.rs` or similar)
2. Generated BD field extraction from register definitions
3. Validation of all existing hand-coded constants against JSON
4. Multi-module support (core, memory, memory_tile, shim)

**Files affected**:
- `src/device/registers_spec.rs` (validate or replace)
- `src/device/dma/bd.rs` (generate field extraction from JSON)
- `src/device/aie2_spec.rs` (validate or replace)
- `xdna_archspec::runtime` (`ArchConfig`/`ModelConfig` -- validate against AIETargetModel; port-layout extension in `src/device/port_layout.rs`)

**Estimated scope**: ~400 lines new code, ~600 lines replaced/validated

### Tier 1: Data-Driven Instruction Decoder (SECOND PRIORITY)

**Goal**: Replace heuristic operand decoding with inverse encoder functions
derived from MCCodeEmitterRegOperandDef.h.

**Plan**: See `.claude/plans/misty-hopping-clock.md` for full details.

**Key deliverables**:
1. OperandType enum with CompositeEncoder variants
2. classify_operand_type() from OperandDef.reg_class
3. Inverse encoder functions for all composite groups
4. Rewritten extract_operands() with zero heuristics

**Files affected**:
- `src/tablegen/resolver.rs` (add OperandType, classify function)
- `src/interpreter/decode/decoder.rs` (rewrite extract_operands)

### Tier 3: Cycle-Accurate Timing (THIRD PRIORITY)

**Goal**: Replace estimated timing constants with real pipeline latencies
from AIE2Schedule.td.

**Key deliverables**:
1. Parse instruction itinerary classes from AIE2Schedule.td
2. Map each instruction to its itinerary (functional units, latencies)
3. Replace estimated constants with parsed values
4. Add resource conflict detection (functional unit contention)

**Files affected**:
- `src/tablegen/` (extend parser for scheduling info)
- `src/device/aie2_spec.rs` (replace timing constants)
- `src/interpreter/` (use real latencies in execution)

### Tier 4: Full Semantic Coverage (ULTIMATE GOAL)

**Goal**: Auto-generate instruction execution logic from intrinsics and
TableGen semantic patterns, achieving true 100% emulation coverage.

**Key deliverables**:
1. Complete SemanticOp coverage for all 300+ instructions
2. Auto-generated vector operation handlers from builtin signatures
3. Accumulator and saturation mode handling from intrinsic types
4. Eliminate the 116-variant Operation enum fallback path

**Files affected**:
- `src/interpreter/execute/` (major rewrite/generation)
- `src/interpreter/bundle/slot.rs` (simplify Operation enum)
