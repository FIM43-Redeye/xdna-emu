_Chapter 5:_ AIE-ML v2 Memory Tile Overview and Features

#### _Chapter 5_

## AIE-ML v2 Memory Tile Overview and Features

The AIE-ML v2 architecture incorporates memory tiles that provide substantial on-chip storage within the array, improving data locality and bandwidth while reducing the utilization of memory resources in the PL. The AIE-ML v2 memory tile is similar to the AIE-ML v2 tile but without the core and program memory. Instead, it features a larger (512 KB) high-density and high-bandwidth data memory, together with an integrated DMA. All DMA channels can access the local memory and a subset of them can directly access the memory in the nearest neighboring memory tiles to the east and west. A subset of DMA channels can directly access the memory in the nearest neighboring memory tiles to the east and west.

The AIE-ML v2 memory tile has the following functional blocks. They are either the same, or very similar to the equivalent blocks in AIE-ML v2 tile:

- Memory

- DMA

- Locks

- Stream switch

- AXI4 switch

- Control, debug, and trace

- Events and event broadcast

The following are the features of the AIE-ML v2 memory tile:

- Memory

- 512 KB of memory arranged in eight physical banks. Each bank is 256-bit wide and 2K words deep.

- ECC protected

- AIE-ML v2 memory tile DMA

- Six memory to stream (MM2S) DMA channels, each with:

- 64-bit stream interface

- 256-bit memory interface

- Support for 4D tensor address generation

- Support for inserting constant value into the stream data called constant padding.

- Access memory and locks in the neighboring tiles (Channels 0 – 3)

- Support task queue and task-complete-tokens with task repeat-count

- Compression of zeros in data before sending on stream

- Store incremental address offset between BD calls

- TLAST suppress feature.

- Six memory to stream (MM2S) DMA channels, each with:

- 64-bit stream interface

- 256-bit memory interface

- Support for 4D tensor address generation

- Access memory and locks in the neighboring tiles (Channels 0 – 3)

- Support task queue and task-complete-tokens with task repeat-count

- Decompression of stream data to re-insert compressed zeroes before storing to memory.

- Store incremental address offset between BD calls

- Support for out of order packet transfer

- Support for Finish on TLAST

- Buffer descriptors (BD): 48 shared buffer descriptors across all 12 DMA channels

- Stream switch:

- Shares the same design as AIE-ML v2 tile with 17 manager and 18 subordinate ports.

- The data path has 64-bit throughput which supports two 32-bit words in parallel per cycle.

- TKEEP signal added to support transfer of an odd number of 32-bit-words.

- North and south ports and DMA can run at full throughput of 64-bit per cycle.

- Trace and control ports have 32-bit interface and adapters are placed to convert them to 64-bit.

- Lock module:

- Accessible from local and neighboring AIE-ML v2 memory tile DMA channels

- There are 64 semaphore locks and each lock state is 6-bit unsigned.

- Additional control and status registers:

- Events, event actions, event broadcast, and combo events

- Task-complete-tokens logic

- Configuration/debug interconnect (AXI4)

- 1 MB address space per tile

- Stream control-packet support

- Debug and trace

- Event trace streams

- Six performance counters and a 64-bit tile timer

### **AIE-ML v2 Memory Tile Memory**

Each memory tile has 512 KB of memory formed from eight banks of 64 KB. Each bank is 256bit wide and 2K words deep. Each bank can be accessed by ten read and ten write interfaces.

The memory tile read interfaces are:

- AXI4 read (including control packets)

- One MM2S interface from the MM2S channels in the neighboring memory tile to the left

- 6x local MM2S channels [0 – 5]

- One MM2S interface from the MM2S channels in the neighboring memory tile to the right

- ECC scrubber

|Col1|MM2S 2|MM2S 3|MM2S 4 MM2S 5|Col5|

To MEM read(n-1) [[2]]

From MUXE(n-1) [[3]]

[1] The S2MM interfaces exist for the memory tile, but are not used in the memory read operation.

[2] MEM read(n-1) and MEM read(n+1) correspond to the neighboring tile memory banks on the east and west.

[3] MUXE(n-1) refers to the multiplexer of the tile on the east and MUXW(n+1) refers to the multiplexer of the tile on the west.

To MEM read(n+1) [[2]] From MUXW(n+1) [[3]]

The memory tile write interfaces are:

- AXI4 write (including control packets)

- One MM2S interface from the MM2S channels in the neighboring memory tile to the left

- 6x local S2MM channels [0 – 5]

- One MM2S interface from the MM2S channels in the neighboring memory tile to the right

- ECC scrubber/zeroization

|Figure 25: Memory Tile Memory Write Interfaces|Col2|Col3|

[1] The MM2S interfaces exist for the memory tile, but are not used in the memory write operation.

[2] MEM write -1 and MEM write +1 correspond to the neighboring tiles on the east and west.

[3] MUXE(n-1) refers to the multiplexer of the tile on the east and MUXW(n+1) refers to the multiplexer of the tile on the west.

DMA S2MM channels (0 – 3) and MM2S channels (0 – 3) can access the local memory banks and the memory banks of the Memory-tile to the east and the west. The memory in memory tile supports bank interleaving. Interleaving is done at 256-bit granularity, such that sequential 256-bit accesses map to different banks and wrap around after the eight banks (every 256B). The memory tile banks have ECC protection and ECC scrubbing.

### **AIE-ML v2 Memory Tile DMA**

The list of features in the AIE-ML v2 memory tile DMA is covered in Chapter 5: AIE-ML v2 Memory Tile Overview and Features. The memory tile DMA is similar to the AI Engine AIE-ML v2 tile DMA with the following enhancements:

- Supports 5D tensor address generation (including iteration-offset)

- Allows out-of-order buffer descriptor (BD) processing based on incoming packet header information

- Supports compression and decompression

The memory tile DMA has 12 independent channels: six S2MM and six MM2S. Each channel has an input task queue. It can load a BD, generate address, access memory over a shared interface, and read or write to and from its stream port. Each channel can also trigger the issuing of a task-complete-token upon completing a task. Each S2MM channel can be configured to work in either in-order or out-of-order mode.

DMA S2MM channels 0 – 3 and MM2S channels 0 – 3 can access the memory banks in the tile to the west and east, in addition to the local memory banks. These same channels can also access lock modules in tile to the east and west. Both MM2S and S2MM channels 4 – 5 can only access local memory banks and local lock modules. All 12 channels use the same address scheme and lock indexes, as shown in the following table.

|Table 12: Address and Lock Ranges for Memory Tile DMAs|Col2|Col3|Col4|
||**Address Ranges**|**Lock Indexes**|**Description**|
|West|0x0_0000 –0x7_FFFF|0 – 63|Channels 0 – 3 only|
|Local|0x8_0000 –0xF_FFFF|64 – 127||
|East|0x10_0000 –0x17_FFFF|128 – 191|Channels 0 – 3 only|

With this addressing scheme, it is possible to configure the hardware where the address and lock requests could be out of range for a specific DMA channel. This condition can result in the DMA channel stalling requiring a channel reset to proceed.

The memory tile MM2S channels support constant-padding insertion into the stream data. There is a 32-bit register per MM2S channel which can be set to any value, this value is inserted by hardware during padding. This feature satisfies two application requirements:

- Algorithmic padding: To recreate surrounding data on the edge of valid data.

- Granularity padding: An optimized kernel can operate on 16 channels while a layer can have 24 channels, the MM2S channel pads the channel dimension up to 32 channels.

The constant-padding insertion is linked to 4D address generation. For the lower three illustrates constant padding in three dimensions. The padding on a dimension is added on top of the wrap for that dimension.

Constant-pad in dimension 0 Constant-pad in dimension 1 Constant-pad in dimension 2

Volume set on stream

### **AIE-ML v2 Memory Tile Locks Module and** **Stream Switch**

The memory tile semaphore locks are identical to those in the AIE-ML v2 tile. The memory tile supports up to 64 semaphore locks each carrying a 6-bit unsigned lock state. All the locks are accessible from local DMA channels and from S2MM and MM2S channels 0 – 3 from both east and west neighboring memory tiles. Locks are also accessible through memory mapped AXI4.

The memory tile stream switch is similar to the tile stream switch with 17 manager ports and 18 subordinate ports. Unlike the tile stream switch, this design has no east or west stream ports.