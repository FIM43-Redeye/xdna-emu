# AIE-ML DMA Architecture Reference

This document provides a complete reference for implementing DMA in the xdna-emu emulator,
derived from AMD AM020/AM025 documentation, mlir-aie source code, and llvm-aie ISA definitions.

**Validation Status:** Cross-referenced against mlir-aie (AIETargetModel.cpp, AIEDmaToNpu.cpp),
llvm-aie (AIE2GenInstrInfo.td), and xdna-driver. All register layouts confirmed accurate.

## Overview

The AIE-ML DMA system transfers data between:
- **Memory ↔ Stream** within a tile
- **Tiles** via the stream switch network
- **DDR memory** via Shim tiles and NoC

Each DMA transfer is defined by a **Buffer Descriptor (BD)** containing:
- Memory address and transfer size
- Multi-dimensional addressing (up to 4D for MemTile)
- Lock acquire/release operations
- Chaining to next BD

## Tile Types and DMA Capabilities

| Feature | Compute Tile | MemTile | Shim Tile |
|---------|-------------|---------|-----------|
| Memory Size | 64 KB | 512 KB | DDR access |
| S2MM Channels | 2 | 6 | 2 |
| MM2S Channels | 2 | 6 | 2 |
| Buffer Descriptors | 16 | 48 | 16 |
| BD Registers | 6 | 8 | 8 |
| Address Bits | 14 (word) | 19 (word) | 46 (word) |
| Dimensions | 3 | 4 | 3 |
| Locks | 16 | 64 | 16 |

## Buffer Descriptor Layouts

### Compute Tile (Memory Module) - 6 Registers per BD

Base: `0x1D000`, Spacing: `0x20` (32 bytes)

```
BD_0 @ 0x1D000 + bd_id * 0x20
  [27:14] Base_Address      - Word address in local memory
  [13:0]  Buffer_Length     - Transfer size in 32-bit words (actual value)

BD_1 @ 0x1D004 + bd_id * 0x20
  [31]    Enable_Compression
  [30]    Enable_Packet     - Add packet header (MM2S only)
  [29:24] Out_Of_Order_BD_ID
  [23:19] Packet_ID
  [18:16] Packet_Type

BD_2 @ 0x1D008 + bd_id * 0x20
  [25:13] D1_Stepsize       - Dim1 step (actual - 1)
  [12:0]  D0_Stepsize       - Dim0 step (actual - 1)

BD_3 @ 0x1D00C + bd_id * 0x20
  [28:21] D1_Wrap           - Wrap count for dim1 (0 = no wrap)
  [20:13] D0_Wrap           - Wrap count for dim0 (0 = no wrap)
  [12:0]  D2_Stepsize       - Dim2 step (actual - 1)

BD_4 @ 0x1D010 + bd_id * 0x20
  [24:19] Iteration_Current - Current iteration (auto-incremented)
  [18:13] Iteration_Wrap    - Wrap iteration (actual - 1)
  [12:0]  Iteration_Stepsize - Offset per BD execution (actual - 1)

BD_5 @ 0x1D014 + bd_id * 0x20
  [31]    TLAST_Suppress    - Don't assert TLAST at end (MM2S)
  [30:27] Next_BD           - Next BD ID (0-15)
  [26]    Use_Next_BD       - Chain to next BD
  [25]    Valid_BD          - BD is valid
  [24:18] Lock_Rel_Value    - Lock release delta (signed, 0 = no release)
  [16:13] Lock_Rel_ID       - Lock ID to release (0-15)
  [12]    Lock_Acq_Enable   - Enable lock acquire
  [11:5]  Lock_Acq_Value    - Lock acquire threshold (signed)
  [3:0]   Lock_Acq_ID       - Lock ID to acquire (0-15)
```

### MemTile - 8 Registers per BD

Base: `0xA0000`, Spacing: `0x20` (32 bytes)

```
BD_0 @ 0xA0000 + bd_id * 0x20
  [31]    Enable_Packet
  [30:28] Packet_Type
  [27:23] Packet_ID
  [22:17] Out_Of_Order_BD_ID
  [16:0]  Buffer_Length     - Up to 128K words

BD_1 @ 0xA0004 + bd_id * 0x20
  [31:26] D0_Zero_Before    - Zero padding before dim0 (MM2S)
  [25:20] Next_BD           - Next BD ID (0-47, but only 0-23 valid)
  [19]    Use_Next_BD
  [18:0]  Base_Address      - 19-bit word address (512KB)

BD_2 @ 0xA0008 + bd_id * 0x20
  [31]    TLAST_Suppress
  [26:17] D0_Wrap
  [16:0]  D0_Stepsize       - 17-bit range

BD_3 @ 0xA000C + bd_id * 0x20
  [31:27] D1_Zero_Before
  [26:17] D1_Wrap
  [16:0]  D1_Stepsize

BD_4 @ 0xA0010 + bd_id * 0x20
  [31]    Enable_Compression
  [30:27] D2_Zero_Before
  [26:17] D2_Wrap
  [16:0]  D2_Stepsize

BD_5 @ 0xA0014 + bd_id * 0x20
  [31:28] D2_Zero_After
  [27:23] D1_Zero_After
  [22:17] D0_Zero_After
  [16:0]  D3_Stepsize       - 4th dimension!

BD_6 @ 0xA0018 + bd_id * 0x20
  [28:23] Iteration_Current
  [22:17] Iteration_Wrap
  [16:0]  Iteration_Stepsize

BD_7 @ 0xA001C + bd_id * 0x20
  [31]    Valid_BD
  [30:24] Lock_Rel_Value    - 7-bit signed
  [23:16] Lock_Rel_ID       - 8-bit (0-63 locks)
  [15]    Lock_Acq_Enable
  [14:8]  Lock_Acq_Value    - 7-bit signed
  [7:0]   Lock_Acq_ID       - 8-bit (0-63 locks)
```

### Shim Tile (NoC Module) - 8 Registers per BD

Base: `0x1D000`, Spacing: `0x20` (32 bytes)

```
BD_0 @ 0x1D000 + bd_id * 0x20
  [31:0]  Buffer_Length     - Full 32-bit range for DDR

BD_1 @ 0x1D004 + bd_id * 0x20
  [31:2]  Base_Address_Low  - Lower 30 bits of 46-bit word address
  [1:0]   Reserved

BD_2 @ 0x1D008 + bd_id * 0x20
  [30]    Enable_Packet
  [29:24] Out_Of_Order_BD_ID
  [23:19] Packet_ID
  [18:16] Packet_Type
  [15:0]  Base_Address_High - Upper 16 bits of 46-bit word address

BD_3 @ 0x1D00C + bd_id * 0x20
  [30]    Secure_Access
  [29:20] D0_Wrap
  [19:0]  D0_Stepsize       - 20-bit range (up to 1M)

BD_4 @ 0x1D010 + bd_id * 0x20
  [31:30] Burst_Length      - 00=64B, 01=128B, 10=256B
  [29:20] D1_Wrap
  [19:0]  D1_Stepsize

BD_5 @ 0x1D014 + bd_id * 0x20
  [31:28] SMID              - AXI stream master ID
  [27:24] AxCache           - AXI cache attributes
  [23:20] AxQoS             - AXI QoS
  [19:0]  D2_Stepsize

BD_6 @ 0x1D018 + bd_id * 0x20
  [31:26] Iteration_Current
  [25:20] Iteration_Wrap
  [19:0]  Iteration_Stepsize

BD_7 @ 0x1D01C + bd_id * 0x20
  [31]    TLAST_Suppress
  [30:27] Next_BD
  [26]    Use_Next_BD
  [25]    Valid_BD
  [24:18] Lock_Rel_Value
  [16:13] Lock_Rel_ID
  [12]    Lock_Acq_Enable
  [11:5]  Lock_Acq_Value
  [3:0]   Lock_Acq_ID
```

## DMA Channel Control Registers

### Compute Tile

```
S2MM Channel Control:
  S2MM_0_Ctrl      @ 0x1DE00
  S2MM_0_Start_Queue @ 0x1DE04
  S2MM_1_Ctrl      @ 0x1DE08
  S2MM_1_Start_Queue @ 0x1DE0C

MM2S Channel Control:
  MM2S_0_Ctrl      @ 0x1DE10
  MM2S_0_Start_Queue @ 0x1DE14
  MM2S_1_Ctrl      @ 0x1DE18
  MM2S_1_Start_Queue @ 0x1DE1C

Status Registers:
  S2MM_Status_0    @ 0x1DF00
  S2MM_Status_1    @ 0x1DF04
  MM2S_Status_0    @ 0x1DF10
  MM2S_Status_1    @ 0x1DF14
```

### Start_Queue Register Fields

```
  [31]    Enable_Token_Issue - Issue token on completion
  [23:16] Repeat_Count       - Task repetitions (actual - 1)
  [3:0]   Start_BD_ID        - First BD to execute
```

### Status Register Fields

```
  [27:24] Cur_BD                    - Current BD being processed
  [22:20] Task_Queue_Size           - Pending tasks in queue
  [19]    Channel_Running           - Channel active
  [18]    Task_Queue_Overflow       - Queue full error
  [11]    Error_BD_Invalid          - Invalid BD loaded
  [5]     Stalled_TCT               - Token backpressure
  [4]     Stalled_Stream            - Stream backpressure/starvation
  [3]     Stalled_Lock_Rel          - Waiting for lock release
  [2]     Stalled_Lock_Acq          - Waiting for lock acquire
  [1:0]   Status                    - 00=IDLE, 01=STARTING, 10=RUNNING
```

## Stream Switch Port Mappings

### Compute Tile

**Master Ports (23 total) - Data OUT from switch:**
| Port | Name | Function |
|------|------|----------|
| 0 | Core | Core stream output |
| 1 | DMA0 | S2MM channel 0 (receives from switch) |
| 2 | DMA1 | S2MM channel 1 (receives from switch) |
| 3 | Tile_Ctrl | Control port |
| 4 | FIFO0 | Local FIFO |
| 5-8 | South0-3 | To tile below (4) |
| 9-12 | West0-3 | To tile west (4) |
| 13-18 | North0-5 | To tile above (6) |
| 19-22 | East0-3 | To tile east (4) |

**Slave Ports (25 total) - Data IN to switch:**
| Port | Name | Function |
|------|------|----------|
| 0 | Core | Core stream input |
| 1 | DMA_0 | MM2S channel 0 (sends to switch) |
| 2 | DMA_1 | MM2S channel 1 (sends to switch) |
| 3 | Tile_Ctrl | Control port |
| 4 | FIFO_0 | Local FIFO |
| 5-10 | South0-5 | From tile below (6) |
| 11-14 | West0-3 | From tile west (4) |
| 15-18 | North0-3 | From tile above (4) |
| 19-22 | East0-3 | From tile east (4) |
| 23 | AIE_Trace | Core trace |
| 24 | Mem_Trace | Memory trace |

### MemTile

**Master Ports (17 total):**
| Port | Name | Function |
|------|------|----------|
| 0-5 | DMA0-5 | S2MM channels (6) |
| 6 | Tile_Ctrl | Control |
| 7-10 | South0-3 | To Shim below (4) |
| 11-16 | North0-5 | To Compute above (6) |

**Slave Ports (18 total):**
| Port | Name | Function |
|------|------|----------|
| 0-5 | DMA0-5 | MM2S channels (6) |
| 6 | Tile_Ctrl | Control |
| 7-12 | South0-5 | From Shim below (6) |
| 13-16 | North0-3 | From Compute above (4) |
| 17 | Trace | Trace output |

### Shim Tile

**Master Ports:**
| Port | Name | Function |
|------|------|----------|
| 0 | FIFO | Local FIFO |
| 1 | Tile_Ctrl | Control |
| 2-7 | South0-5 | To NoC/PL (6) |
| 8-11 | West0-3 | To west Shim (4) |
| 12-17 | North0-5 | To MemTile above (6) |
| 18-21 | East0-3 | To east Shim (4) |

**Slave Ports:**
| Port | Name | Function |
|------|------|----------|
| 0 | FIFO | Local FIFO |
| 1 | Tile_Ctrl | Control |
| 2-9 | South0-7 | From NoC/PL (8) |
| 10-13 | West0-3 | From west Shim (4) |
| 14-17 | North0-3 | From MemTile above (4) |
| 18-21 | East0-3 | From east Shim (4) |
| 22 | Trace | Trace output |

## Inter-Tile Stream Connections

Stream switches connect via directional ports. The port mappings between adjacent tiles:

```
Shim North Masters [12-17] → MemTile South Slaves [7-12]
MemTile North Masters [11-16] → Compute South Slaves [5-10]
Compute South Masters [5-8] → MemTile North Slaves [13-16]
MemTile South Masters [7-10] → Shim North Slaves [14-17]
```

## Lock Mechanism

### Lock Registers (Memory Module)

Base: `0x1F000`, Spacing: `0x10` (16 bytes)

```
Lock0_value  @ 0x1F000
Lock1_value  @ 0x1F010
...
Lock15_value @ 0x1F0F0

Each lock:
  [5:0] Lock_value - 6-bit unsigned (0-63)
```

### Lock Acquire/Release Semantics

**From BD (DMA):**
- **Acquire**: Wait until `lock_value >= acq_value` (if acq_value >= 0) or `lock_value > 0` (if acq_value < 0)
- **Release**: `lock_value += rel_value` (signed addition)

**From Core (via address space 0x40000-0x43FFC):**
Address encoding:
```
[13:10] Lock_ID      - Which lock (0-15)
[9]     Acq_Rel      - 0=acquire, 1=release
[8:2]   Change_Value - Signed delta
```
Result at bit 0: 1=success, 0=failed (retry)

### Lock Acquire Conditions

The acquire semantics depend on the sign of `Lock_Acq_Value`:

| Acq_Value | Condition | Description |
|-----------|-----------|-------------|
| >= 0 | lock_value == acq_value | Exact match (producer/consumer sync) |
| < 0 | lock_value >= |acq_value| | Greater-than-or-equal (semaphore) |

## Core ISA Instructions (from llvm-aie)

The AIE2 core interacts with DMA and locks via specific instructions.

### Lock Instructions

**Acquire (blocking until condition met):**
```
ACQ $lockId, $value      ; Acquire lock (immediate ID, 6-bit)
ACQ $mRx, $mRy           ; Acquire lock (register ID, value in mRy)
ACQ_COND $lockId, $mRy   ; Conditional acquire (checks r26)
```

**Release:**
```
REL $lockId, $value      ; Release lock (immediate ID)
REL $mRx, $mRy           ; Release lock (register ID)
REL_COND $lockId, $mRy   ; Conditional release (checks r26)
```

**Encoding (ALU slot):**
```
alu = {mLockId[6:0], -, op[1:0], mRy[4:0], 0b0010, 0b0}

op values:
  0b00 = REL
  0b01 = ACQ
  0b10 = REL_COND
  0b11 = ACQ_COND

mLockId encoding:
  Immediate: {id[5:0], 0b0}
  Register:  {mRx, -, 0b1}
```

### Stream Instructions

**Write to Master Stream (core output):**
```
MOV ms, $src             ; Blocking write to stream
MOV.NB ms, $src          ; Non-blocking write (sets srMS0 status)
MOV.TLAST ms, $src       ; Write with TLAST assertion
MOV ms, $src, $tlast     ; Write with register-controlled TLAST (r28)
```

**Read from Slave Stream (core input):**
```
MOV $mRa, SS             ; Blocking read from stream
MOV.NB $mRa, SS          ; Non-blocking read (sets srSS0 status)
```

**Packet-switched Stream:**
```
MOV.PH ms, $id, $pcktType    ; Send packet header (ID register, 3-bit type)
MOV.CPH ms, $addr, $nw, $op, $id  ; Custom packet header
```

**Stream Encoding (ST slot):**
```
st = {dst[2:0], -, -, 0b0111, 0b1000, src[6:0], 0b0}

dst[2:0] controls behavior:
  0b000 = Blocking, no TLAST
  0b001 = Blocking, with TLAST
  0b100 = Non-blocking, no TLAST
  0b101 = Non-blocking, with TLAST
```

### Memory Load/Store (DMA interaction)

The core accesses DMA-managed memory via standard load/store with address generators:

```
LDA.S8 $mRa, [$ptr, $imm]    ; Load signed 8-bit
LDA.U16 $mRa, [$ptr, $dj]    ; Load unsigned 16-bit with register offset
ST.S16 $mRv, [$ptr, $imm]    ; Store signed 16-bit

; 2D/3D addressing variants use hardware counters
LDA_2D ...                    ; 2D load with counter management
LDA_3D ...                    ; 3D load with dual counters
```

### Instruction Side Effects

| Instruction | hasSideEffects | mayLoad | mayStore | Blocking |
|-------------|----------------|---------|----------|----------|
| ACQ/REL | Yes | No | No | Yes (ACQ) |
| MOV ms | Yes | No | No | Yes |
| MOV.NB ms | Yes | No | No | No |
| MOV SS | Yes | Yes | No | Yes |
| LDA | No | Yes | No | No |
| ST | No | No | Yes | No |

## Data Flow Examples

### Simple Passthrough (DDR → Compute → DDR)

```
1. Shim MM2S: Read from DDR, output to stream
   - BD: DDR address, length, route to North

2. Stream routing: Shim → MemTile → Compute
   - Shim North master → MemTile South slave
   - MemTile routes internally
   - MemTile North master → Compute South slave

3. Compute S2MM: Receive stream, write to local memory
   - BD: Local address, length, acquire lock 0

4. Core processes data, releases lock 1

5. Compute MM2S: Read local memory, output to stream
   - BD: Local address, length, acquire lock 1

6. Stream routing: Compute → MemTile → Shim
   - Compute South master → MemTile North slave
   - MemTile routes internally
   - MemTile South master → Shim North slave

7. Shim S2MM: Receive stream, write to DDR
   - BD: DDR address, length
```

## Implementation Notes

### mlir-aie Runtime Status

As of 2026-01 (may be outdated upstream): mlir-aie runtime BD configuration **only supports MemTile and Shim tiles**.
Compute tile runtime BD configuration is marked as TODO (AIEDmaToNpu.cpp:609-611).
However, compute tiles receive BD configuration via CDO at load time, not runtime patching.

### Address Patch Offset (Critical!)

When patching DDR addresses at runtime, the offset within the BD differs by tile type:
- **Compute tile:** Offset `+0x0` (Base_Address is in BD_0 bits [27:14])
- **MemTile:** Offset `+0x4` (Base_Address is in BD_1 bits [18:0])
- **Shim tile:** Offset `+0x4` (Base_Address_Low is in BD_1 bits [31:2])

This is returned by `AIETargetModel::getDmaBdAddressOffset()`.

### Critical Details Often Missed

1. **Address units are WORDS not bytes** - All addresses and stepsizes in 32-bit word units

2. **Stepsizes are stored as (actual - 1)** - Add 1 when computing actual offset

3. **Lock spacing is 16 bytes** - Not 4 bytes like typical registers

4. **BD spacing is 32 bytes** - Even though compute tile only uses 24 bytes

5. **DMA slave ports receive MM2S output** - MM2S produces data that enters switch via slave port

6. **DMA master ports send to S2MM input** - S2MM consumes data that exits switch via master port

7. **Lock release happens AFTER data transfer completes**

8. **Lock acquire happens BEFORE data transfer starts**

9. **Iteration_Current is auto-incremented by hardware** after each BD execution

10. **MemTile has 8-bit Lock IDs** (64 locks) vs 4-bit for Compute/Shim (16 locks)

### State Machine

```
DMA Channel States:
  IDLE (00) → STARTING (01) → RUNNING (10)
                   ↓
            STALLED (on lock or backpressure)
```

Channel transitions:
1. Write to Start_Queue puts BD in task queue
2. Channel fetches BD from queue → STARTING
3. Lock acquire succeeds → RUNNING
4. Data transfer completes
5. Lock release
6. If Use_Next_BD: fetch Next_BD, goto step 3
7. If !Use_Next_BD: goto IDLE or fetch next task from queue

## Runtime DMA Control (from mlir-aie)

### Starting a DMA Task

To start a DMA transfer, write to the channel's Start_Queue register:

```
Control register addresses (from AIETargetModel.cpp):
  Shim:    0x1D200 + channel * 0x8, +0x10 for MM2S
  MemTile: 0xA0600 + channel * 0x8, +0x30 for MM2S
  Compute: 0x1DE00 + channel * 0x8, +0x10 for MM2S

Queue register = Control register + 0x4
```

**Start_Queue command format:**
```
  [31]    Enable_Token_Issue - Set to receive completion notification
  [23:16] Repeat_Count       - Execute BD chain this many times (actual - 1)
  [3:0]   Start_BD_ID        - First BD in the chain
```

**Example (Compute tile MM2S channel 0):**
```
Control: 0x1DE10
Queue:   0x1DE14
Command: (bd_id & 0xF) | ((repeat_count & 0xFF) << 16) | (issue_token ? 0x80000000 : 0)
```

### Writing Buffer Descriptors

BDs are written via block writes to BD register addresses:

```
BD base addresses:
  Shim:    0x1D000 + bd_id * 0x20
  MemTile: 0xA0000 + bd_id * 0x20
  Compute: 0x1D000 + bd_id * 0x20
```

The mlir-aie runtime patches DDR addresses at load time using `NpuAddressPatchOp`
which writes the low 32 bits of the address to BD register offset:
- Compute tile: +0x0 (Base_Address in BD_0)
- MemTile/Shim: +0x4 (Base_Address in BD_1)

### Token Synchronization

When `Enable_Token_Issue` is set, the DMA issues a Task Complete Token (TCT) when
the BD chain finishes. The host waits for this token using `NpuSyncOp`:

```
Sync parameters:
  column, row   - Tile location
  direction     - 0=S2MM, 1=MM2S
  channel       - Channel number
  row_num       - Number of rows (usually 1)
  col_num       - Number of columns (usually 1)
```

## Complete Example: Passthrough Kernel

The passthrough_kernel test does:

1. **Shim MM2S (host → device):**
   - BD at Shim tile (0,0): DDR address, length, route North
   - Start via Write32 to 0x1D220 (Shim MM2S ch0 queue)

2. **Routing:**
   - Shim North master[12] → MemTile South slave[7]
   - MemTile internal route (configured via stream switch)
   - MemTile North master[11] → Compute South slave[5]
   - Compute internal route: South slave[5] → DMA master[1] (S2MM ch0)

3. **Compute S2MM (stream → local memory):**
   - BD at (0,2): local address 0, length 1024, acquire lock 0
   - Automatically started by stream data arrival

4. **Core execution:**
   - Waits on lock 0 (input ready)
   - Processes data
   - Releases lock 1 (output ready)

5. **Compute MM2S (local memory → stream):**
   - BD at (0,2): local address 0, length 1024, acquire lock 1
   - Routes to: DMA slave[1] → South master[5]

6. **Routing back:**
   - Compute South master[5] → MemTile North slave[13]
   - MemTile internal route
   - MemTile South master[7] → Shim North slave[14]
   - Shim internal route → S2MM

7. **Shim S2MM (device → host):**
   - BD at Shim tile (0,0): DDR address, length
   - Host waits for completion token
