_Chapter 5:_ AIE-ML Memory Tile Architecture

#### _Chapter 5_

## AIE-ML Memory Tile Architecture

### **AIE-ML Memory Tile Overview and Features**

The AIE-ML memory tile is introduced in the AIE-ML architecture to significantly increase the on-chip memory inside the AIE-ML array. The memory tile reduces the utilization of PL resources (LUTs, block RAMs and UltraRAMs) in ML applications. It is similar to the AIE-ML tile but without the AIE-ML processor and program memory. The AIE-ML memory tile contains high-density (512 KB) and high bandwidth memory, and an integrated DMA to access local memory and neighboring memories. The AIE-ML memory tile only has vertical streaming interfaces (no cascade or horizontal). A subset of DMA channels can directly access memory in the nearest tile architecture.

The memory tile has the following functional blocks. They are either the same or similar to the equivalent blocks in the AIE-ML tile:

- Memory

- DMA

- Locks

- AXI4-Stream switch

- Memory-mapped AXI4 switch

- Control, debug, and trace

- Events and event broadcast

The following is a list of AIE-ML memory tile features:

- Memory

- 512 KB memory arranged into 16 banks (each 128-bit wide and 2k words deep), ECC protected

- The memory banks in the AIE-ML memory tile initializes to zero at boot and reset

- Supports up to 30 GB/s read and 30 GB/s write in parallel per memory tile

- DMA

- Memory to stream DMA (MM2S) with six channels

- 6 x 32-bit stream interfaces

- 6 x 128-bit memory interfaces

- 5D tensor address generation (including iteration-offset)

- Support inserting zero padding into stream data and compression

- Access memory and locks in east/west neighboring tiles (channels 0–3)

- Support task queue and task-complete-tokens; queue depth is four tasks per channel (see Task-Completion-Tokens for more information)

- Stream to memory DMA (S2MM) with six channels

- 6x32-bit stream interfaces

- 6x128-bit memory interfaces

- 5D tensor address generation (including iteration-offset)

- Support out-of-order packet transfer, finish-on-TLAST, and decompression

- Access memory and locks in east/west neighboring tiles (channel 0-3)

- Support task queue and task-complete-tokens; queue depth is four tasks per channel (see Task-Completion-Tokens for more information)

- Buffer descriptors (BD)

- 48 shared BDs

- Each channel can access 24 BDs and each BD can be accessed by six channels

- Stream Switch

- Share the same design as AIE-ML tile. 17 master and 18 slave ports

- North and South ports but no east and west streams

- Trace and control ports

- Lock Module

- Accessible from neighboring AIE-ML memory tile DMA channels; there are 64 semaphore locks and each lock state is 6-bit unsigned

- Additional control and status registers

- Events, event actions, event broadcast, combo events

- Task-complete-tokens logic (see Task-Completion-Tokens for more information)

- Configuration/debug interconnect (memory-mapped AXI4)

- 1 MB address space per tile

- Write bandwidth improvement and stream control-packet support

- Debug and Trace

- Similar to that in AIE-ML tile

- Event trace stream; 4x performance counters and 64-bit tile timer

### **AIE-ML Memory Tile Memory**

Each AIE-ML memory tile has 512 KB of memory as 16 banks of 32 KB. Each bank is 128 bits wide and 2k words deep. Each bank allows one read or one write every cycle and can be accessed by nine read interfaces and nine write interfaces. Each interface is 128-bit. The following block diagrams show the memory tile read and write interfaces.

|Col1|MM2S 2|MM2S 3|MM2S 4 MM2S 5|Col5|

To MEM read(n-1) [[2]]

From MUXE(n-1) [[3]]

[1] The S2MM interfaces exist for the memory tile, but are not used in the memory read operation.

[2] MEM read(n-1) and MEM read(n+1) correspond to the neighboring tile memory banks on the east and west.

[3] MUXE(n-1) refers to the multiplexer of the tile on the east and MUXW(n+1) refers to the multiplexer of the tile on the west.

To MEM read(n+1) [[2]] From MUXW(n+1) [[3]]

|Figure 41: Memory Tile Memory Write Interfaces|Col2|Col3|

[1] The MM2S interfaces exist for the memory tile, but are not used in the memory write operation.

[2] MEM write -1 and MEM write +1 correspond to the neighboring tiles on the east and west.

[3] MUXE(n-1) refers to the multiplexer of the tile on the east and MUXW(n+1) refers to the multiplexer of the tile on the west.

The interfaces also have the following features:

- **Read Interfaces:** Memory-mapped AXI4 write including control packets for memory tile –1 and memory tile +1, and six MM2S channels [0-5]

- **Write Interfaces:** Memory-mapped AXI4 read including control packets for memory tile –1 and memory tile +1, and six S2MM channels [0-5]

DMA S2MM channels (0 – 3) and MM2S channels (0 – 3) can access the local memory banks and the memory banks of the memory tile to the east and the west.

The memory in memory tile supports bank interleaving. Interleaving is done at a 128-bit granularity, such that sequential 128-bit accesses map to different banks and wrap around after the 16 banks, every 256B.

The memory tile banks have ECC protection and ECC scrubbing similar to the AIE-ML data memory.

_Chapter 5:_ AIE-ML Memory Tile Architecture

### **AIE-ML Memory Tile DMA**

The list of features in the AIE-ML memory tile DMA is covered in the AIE-ML Memory Tile Overview and Features section. The memory tile DMA is similar to the AI Engine (AIE) tile DMA with a few enhancements:

- Supports 5D tensor address generation (including iteration-offset)

- Allows out-of-order buffer descriptor (BD) processing based on incoming packet header information

- Supports compression and decompression

The memory tile DMA has 12 independent channels, six S2MM and six MM2S. Each channel has an input task queue. It can load a BD, generate address, access memory over a shared interface, and read or write to and from its stream port. Each channel can also trigger the issuing of a task-complete-token upon completing a task. The AIE-ML memory tile DMA supports address generation as described in the Data Movement section. The memory tile DMA supports up to four dimensions (K=4).

Of the six S2MM and MM2S channels, DMA S2MM channels 0-3 and MM2S channels 0-3 can access the memory banks in the tile to the west and east, in addition to the local memory banks.
These same channels can also access lock modules in tile to the east and west. Both MM2S and S2MM channels 4-5 can only access local memory banks and local lock modules.

All 12 channels use the same address scheme and lock indexes, as shown in the following table.

|Table 12: Address and Lock Ranges for Memory Tile DMAs|Col2|Col3|Col4|
||**Address Ranges**|**Lock Indexes**|**Description**|
|West|0x0_0000 –0x7_FFFF|0–63|Channels 0–3 only|
|Local|0x8_0000 –0xF_FFFF|64–127||
|East|0x10_0000 –0x17_FFFF|128–191|Channels 0–3 only|

With this addressing scheme, it is possible to configure the hardware where the address and lock requests could be out of range for a specific DMA channel. This condition can result in the DMA channel stalling requiring a channel reset to proceed.

The memory tile MM2S channels support zero-padding insertion. This feature satisfies two application requirements:

- Algorithmic padding: To recreate surrounding data on the edge of valid data.

- Granularity padding: An optimized kernel can operate on 16 channels while a layer can have 24 channels, the MM2S channel pads the channel dimension up to 32 channels.

The zero-padding insertion is linked to 4D address generation. For the lower three dimensions, zero-padding in three dimensions. The padding on a dimension is added on top of the wrap for that dimension.

### **AIE-ML Memory Tile Locks Module and Stream** **Switch**

The memory tile semaphore locks are identical to those in the AIE-ML tile. The memory tile supports up to 64 semaphore locks each identified by a 6-bit unsigned lock state. All the locks are accessible from local DMA channels as well as S2MM and MM2S channels from both east and west neighboring memory tiles. Locks are also accessible through memory-mapped AXI4.

The memory tile stream switch is similar to the tile stream switch with 17 master ports and 18 slave ports. The differences from the tile stream switch are:

- No east or west stream ports

- No stream FIFO