_Appendix A:_ Key Differences between AI Engine and AIE-ML

#### _Appendix A_

## Key Differences between AI Engine and AIE-ML

AI Engines offered in some AMD Versal™ adaptive SoCs are present in different versions optimized for different markets. The initial version, AI Engine (AIE), is optimized for DSP and communication applications, while the AI Engine-Machine Learning ( AIE-ML) introduces a version optimized for machine learning. In this section, the main differences between AIE and AIE-ML are described, including:

- Increased throughput for ML/AI inference workloads.

- Optimized ML/AI application precision. For example, added bfloat16.

- Increased on-chip memory capacity and bandwidth (two times the data memory in each AIE-ML tile and the addition of AIE-ML memory tiles per column in the AIE-ML array).

- Increased multiplier performance.

- Focus on power efficiency (increase TOPs/W).

- Improved hardware for synchronization and reconfiguration.

The differences between the AIE and AIE-ML blocks include the following:

- Removed:

- Native support for INT32. Multiplication of 32-bit numbers are not directly supported but are emulated via decomposition into multiple multiplications of 16 x 16 bit. Also supports cint32 x cint16 multiplication to optimize FFT performance.

- Native FP32 (supported through emulation using bfloat16).

- Added:

- Double INT8/16 compute per tile vs AIE

- Bfloat16

- Local memory tiles

**TIP:** _To understand the features in the first version of AI Engine, refer to the Versal Adaptive SoC AI Engine_ _[Architecture Manual (AM009).](https://docs.amd.com/go/en-US/am009-versal-ai-engine)_

_Appendix A:_ Key Differences between AI Engine and AIE-ML

### **AIE-ML Array Architecture**

This section compares differences in the arrays. For more information, see Chapter 3: AIE-ML Array Interface Architecture and AI Engine Array Interface Architecture in the _Versal Adaptive SoC_ _AI Engine Architecture Manual_ [(AM009). The following provides a summary of the key features of](https://docs.amd.com/go/en-US/am009-versal-ai-engine) AIE-ML that are similar to AIE:

- Same process, voltage, frequency, clock, and power distribution

- Same array topology (one VLIW SIMD processor per AIE-ML tile)

- Each AIE-ML tile has eight integrated banks of data memory shared with three neighboring tiles.

- Each AIE-ML tile has two DMA channels in each direction

- AIE-ML tile to tile stream interconnect has same bandwidth as AIE

- Same PL and NoC interface

- Same debug/trace functionality

The following provides a summary of the key features of AIE-ML that are different or enhanced from AIE:

- At the tile level, the compute/memory is doubled. A processor bus is added to allow the AIE-ML perform direct read/write accesses to local tile memory mapped registers.

- Enhanced DMA are added to the AIE-ML tiles, AIE-ML memory tiles, and AIE-ML array interface tiles that include 3D address generation for tiles/array interface tiles and 4D address generation for memory tiles, out-of-order packets, and Finish-on-TLast in S2MM.
Supports Compression and decompression (tiles and memory tiles) are supported to better handle sparse weights and activations in CNN and RNN application. See Sparsity for more information.

- Addition of AIE-ML memory tiles (maximum of two rows) to significantly reduce programmable logic (PL) resources (LUTs and UltraRAMs) utilization. There is 512 KB of memory per memory tile with ECC and 12 DMA channels (6 MM2S and 6 S2MM).

- Increased memory capacity due to the doubling of the data memory in AIE-ML tiles and the addition of AIE-ML memory tiles.

- Increase in power efficiency (TOPs/W).

- Improved stream switch functionality including source to destination parity check and deterministic merge.

- Improved reconfiguration and synchronization support.

- Grid array architecture to support vertical (from top to bottom) and horizontal (from left to right) 512-bit cascade, versus 384-bit horizontal cascade only.

in AIE-ML. Of note, in AIE-ML the tile rows are all in the same direction. The cascade connections are only from north to south and from west to east.

_Appendix A:_ Key Differences between AI Engine and AIE-ML

### **AIE-ML Tile Architecture**

The AIE-ML tile architecture leverages the functionality and performance requirements from the AI Engine tile architecture. The following provides an overview of the changes made to the AIE-ML tile architecture:

- AIE-ML (see AIE-ML Processor for more information)

- Data memory:

- Data memory is increased from 32 KB to 64 KB organized as eight banks of 8 KB from a hardware perspective. From a programmer's perspective, every two banks are interleaved to form one bank, that is, a total of four banks of 16 KB. The AIE-ML tile has access to the four nearest memory modules in the cardinal directions: north, south, east (local data memory in the tile itself) and west.

- Added memory zero-init

- DMA:

- Features an improved address generation to support 3D addressing modes and iterationstate offset

- Adds task queues and task-complete-tokens (see Task-Completion-Tokens for more information)

- Supports S2MM Finish on TLAST and out-of-order packets

- Added decompression to two S2MM channels

- Added compression to two MM2S channels

- Memory-mapped AXI4 interface: improved read and write bandwidth

- Lock module: 16 semaphore locks and each lock state is 6-bit unsigned versus 16 locks with binary data value in the AI Engine.

### **AIE-ML Processor**

Similar to AIE, the AI Engine processor in AIE-ML consists of a scalar 32-bit data path, a SIMD vector data path, two load units, and a store unit, and is optimized for ML applications.

The following provides a list of AIE-ML processor features:

- Instruction-based VLIW SIMD processor with new instructions

- Same 16 KB program memory as in AIE

- Vector unit supports 256 (8b x 8b) and 512 (4b x 8b) MAC operations

- Vector unit supports 128 bloat16 MAC operations with FP32 accumulation

- Vector unit supports structure sparsity and FFT processing for ML inference applications, including cint32 x cint16 multiplication (data in cint32 and twiddle factor is cint16), control support for complex and conjugation, new permute mode, and shuffle mode. See Sparsity for more information.

- A new processor bus that allows the processor to access memory mapped registers in the local AIE-ML tile

- The complex circular addressing modes are dropped and replaced by a 3D addressing mode

- On-the-fly decompression during loading of sparse weights. See Sparsity for more information.

The AIE-ML processor removes some advanced DSP functionality used in the AIE processor including:

- 32-bit floating-point vector data path is not directly supported but can be emulated via decomposition into multiple multiplications of 16 x 16-bit

- Scalar non-linear functions, including sin/cos, sqrt, inverse sqrt and inverse

- Scalar floating point/integer conversions

- Complex circular addressing and FFT addressing modes. However, some level of FFT and complex support is provided; see the AIE-ML processor features.

- Limited support 128-bit load/store

- Non-aligned memory access

- Support for some complex data-types; some level of complex support is provided, see the

AIE-ML processor features

- Native support for 32 × 32 multiplication but can be emulated using 16-bit integer operands

- Removal of non-blocking 128-bit stream interfaces and stream FIFOs

- Control streams and packet header generations

### **AIE-ML Memory Tile**

Memory tiles are added in AIE-ML, not in AIE. See Chapter 2: AIE-ML Tile Architecture for more

The following is a high-level list of AIE-ML memory tile features:

- 512 KB memory arranged in 16 banks, ECC protected

- Supports up to 30 GB/s read and 30 GB/s write in parallel per memory tile

- DMA channels can directly access memory in the nearest neighbor memory tiles to the east and west

- Memory to stream DMA (MM2S) with six channels and support for 4D tensor address generation, zero-padding insertion, and compression. Accesses memory and locks in east and west neighboring tiles. Supports task queue and task-complete-tokens.

- Stream to memory DMA (S2MM) with six channels and support for 4D tensor address generation, out-of-order packet transfer, finish-on-TLAST, and decompression. Accesses memory and locks in east and west neighboring tiles. Supports task queue and task-completetokens.

- Stream switch in the AIE-ML memory tile shares the same design as the AIE-ML tile. There are 17 master ports and 18 slave ports, but no east or west streams.

- Locks module is accessible from neighboring AIE-ML memory tile DMA channels; there are 64 semaphore locks and each lock state is 6-bit unsigned.

- Additional control and status registers

- Configuration/debug interconnect with a 1 MB memory-mapped AXI4 address space per tile

- Debug and trace similar to that in the AIE-ML tile.

### **AIE-ML Array Interface**

Similar to the Array interface in AIE, the AIE-ML array interface provides the necessary functionality to interface with the rest of the device. The array interface is made up of PL and NoC interface tiles, and there is one configuration interface tile per device. The following is a list of changes from the AIE array interface. The AIE-ML array interface has:

- AIE-ML array interface DMA (read and write to external memory)

- Supports 32-bit aligned start addresses

- 3D address generation and iteration-state offset that supports, with a single buffer descriptor (BD) configuration, an incremental offset to be added to the base address with each subsequent transfer. Also supports 32-bit aligned addresses to external memory.

- Task queue and task-complete-tokens

- Support for S2MM out-of-order and Finish-on-TLAST features (enabling compressed spill and restore of intermediate results to external memory)

- Task queues and task complete tokens

- A lock design with 16 semaphore locks and 6-bit unsigned lock state

- One stream FIFO (stream switch). This is a reduction from two in the AIE array interface.

- Additional control and status registers for new features

- Memory-mapped AXI4 interface for improved read and write bandwidth

### **Software Programmability**

Versal AI Engines are software programmable. You write C/C++ functions/kernels using intrinsic code targeting the VLIW ISA processors in a software programmable environment.

**RECOMMENDED:** _Use the AMD provided AI Engine library functions where appropriate, and write_ _custom AI Engine kernels for the functions not covered by existing optimized library elements._

AIE and AIE-ML have a different set of instructions and intrinsics. Kernels written with intrinsics are not portable across the versions of AI Engine. For portability between the versions of AI Engines, users should use the AI Engine API (see _AI Engine Kernel and Graph Programming Guide_ [(UG1079)) or the AI Engine libraries (see AI Engine Documentation on the AMD website).](https://docs.amd.com/access/sources/dita/map?isLatest=true&ft:locale=en-US&url=ug1079-ai-engine-kernel-coding)

|_Table 14:_**Summary of the Key Differences between AIE and AIE-ML**|_Table 14:_**Summary of the Key Differences between AIE and AIE-ML**|_Table 14:_**Summary of the Key Differences between AIE and AIE-ML**|
||**AIE**|**AIE-ML1**|
|Array structure|Checkerboard|All lines identical|
|Cascade interface|384-bits wide<br>Horizontal direction|512-bits wide<br>Horizontal and vertical<br>directions|
|Tile stream interface|2 × 32-bit in and 2 × out 32-<br>bit out|1 × 32-bit in and 1 × out 32-<br>bit out|
|Memory load/store per cycle|512/256 bits|512/256 bits|
|Advanced DSP functionality|Yes|No|
|INT4 operations/tile|256|512|
|INT8 operations/tile|256|512|
|INT16 operations/tile|64|128|
|INT32 operations/tile|16|324|
|Bfloat16 float operations/tile|–|256|
|FP32 float operations/tile|16|423|
|Data memory/tile|32 KB|64 KB|
|Program memory/tile|16 KB|16 KB|
|Memory tiles|–|512 KB|
|Programmable logic (PL) to AIE array bandwidth|1X|1X|

|Table 14: Summary of the Key Differences between AIE and AIE-ML (cont'd)|Col2|Col3|
||**AIE**|**AIE-ML1**|
|Local memory locks|Boolean|Semaphore|