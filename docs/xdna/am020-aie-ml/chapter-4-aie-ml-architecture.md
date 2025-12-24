_Chapter 4:_ AIE-ML Architecture

#### _Chapter 4_

## AIE-ML Architecture

### **Functional Overview**

The AIE-ML is a highly-optimized processor featuring single-instruction multiple-data (SIMD) and very-long instruction word (VLIW) processor that supports both fixed-point and floating-point vector unit, two load units, one store unit, and an instruction fetch and decode unit.

The features of the AIE-ML include:

- Instruction-based VLIW SIMD processor

- 32-bit scalar RISC processor

- Scalar register files and special registers

- 32 x 32-bit multiplier (signed and unsigned)

- 32-bit add/subtract

- ALU operations like shifts, compares, and logical operations

- No floating point unit: Supported through emulation

- Three address generator units (AGU)

- Two 256-bit load and one 256-bit store units with aligned addresses

- Supports 2D/3D addressing modes for ML functionality

- On-the-fly decompression during loading of sparse weights. See Sparsity for more information.

- One AGU dedicated for the store unit

- Vector fixed-point/integer unit

- Supports FFT processing and sparsity for ML inference applications, including cint32 x cint16 multiplication (data in cint32 and twiddle factor in cint16), control support for complex and conjugation, new permute mode, and shuffle mode. See Sparsity for more information.

- Accommodate multiple precision for complex and real operand. See Table 7: Supported Precision Width of the Vector Data Path for more information.

|Table 7: Supported Precision Width of the Vector Data Path|Col2|Col3|Col4|Col5|
|**Precision 1**|**Precision 2**|**Number of**<br>**Accumulator**<br>**Lanes**|**Bits per**<br>**Accumulator**<br>**Lane**|**Number of MACs**|
|bfloat 163|bfloat 16|16|SPFP 322|128|

- The multiplier|multiplicand can be signed or unsigned. The accumulator is always signed.

- The accumulation can be performed in two operation modes, with either 32 lanes of 32 bits or 16 lanes of 64 bits.

- The total number of multipliers and the number of accumulation lanes determine the depth of the post-adding.

- In terms of component use, consider the first row in Table 7: Supported Precision Width of the Vector Data Path. Depending on whether or not sparsity is used, the multiplier inputs can be 1024 x 512 or 512 x 512 bits. The number of int8 multipliers is 256. The accumulation is on 32 lanes of 32 bits. See Sparsity for more information.

- Single-precision floating-point (SPFP) vector unit:

- Supports 128 bfloat 16 MAC operations with FP32 accumulation by reusing the integer multipliers and post adders along with additional blocks for floating point exponent compute and mantissa shifting and normalization.

- Concurrent operation of multiple vector lanes.

- Supports multiplying bfloat16 numbers (16-bit vector lanes) and accumulating in SPFP (32-bit register lanes). Only 16 accumulator lanes are used in this mode.

- The vector unit also supports integer arithmetic on 8, 16, and 32 bit operands, and bitwise AND, OR, and NEGATE.

- Balanced pipeline:

- Different pipeline on each functional unit (eight stages maximum).

- Load and store units manage the 5-cycle latency of data memory.

- Three data memory ports:

- Two load ports and one store port

- Each port operates in 256-bit/128-bit vector register mode. Scalar accesses (32-bit/16-bit/ 8-bit) are supported by only one load port and one store port. The 8-bit and 16-bit stores are implemented as read-modify-write instructions.

- Concurrent operation of all three ports

- A bank conflict on any port stalls the entire data path

- Very-long instruction word (VLIW) function:

- Concurrent issuing of operation to all functional units

- Support for multiple instruction formats and variable length instructions

- Up to six operations can be issued in parallel using one VLIW word

- Direct stream interface:

- One input stream and one output stream

- Each stream is 32-bits wide

- Vertical in addition to horizontal cascade stream in and stream out in 512 bits

- Interface to the following modules:

- Lock module

- Stall module

- Debug and trace module

- Event interface is a 16-bit wide output interface from the AIE-ML.

- Processor bus interface:

- The AIE-ML architecture is a processor that allows the AIE-ML to perform direct read/ write access to local tile memory mapped registers.

_Chapter 4:_ AIE-ML Architecture

### **Register Files**

The AIE-ML has several types of registers. Some of the registers are used in different functional units. This section describes the various types of registers.

**Scalar Registers**

Scalar registers include configuration registers. See the following table for register descriptions.

|Table 8: Scalar Registers|Col2|Col3|
|**Syntax**|**Number of Bits**|**Description**|
|r0..r31|32 bits|General-purpose registers|
|m0..m7|20 bits|Modifier registers|
|p0..p7|20 bits|Pointer registers|

|Special Registers|Col2|Col3|
|_Table 9:_**Special Registers**|_Table 9:_**Special Registers**|_Table 9:_**Special Registers**|
|**Syntax**|**Number of Bits**|**Description**|
|dn0..dn7|20 bits|AGU dimension size register|
|dj0..dj7|20 bits|AGU dimension stride (jump) register|
|dc0..dc7|20 bits|AGU dimension count register|
|s0..s3|6 bits|Shift control|
|sp|20 bits|Stack pointer|
|lr|20 bits|Link register|
|pc|20 bits|Program counter|
|fc|20 bits|Fetch counter|
||32 bits|Status register1|
||32 bits|Mode control register1|
|ls|20 bits|Loop start|
|lc|32 bits|Loop count|
|lci|32 bits|Loop count (PCU)|

**Vector Registers**

Vector registers are wide to allow SIMD instructions and to be used as operand storage. These registers are prefixed with a _W_ . There are 24 x 256-bit registers: wln and whn, n-0..11. Two W registers can be grouped to form a 512-bit register prefixed with an X. Two X registers then can be grouped to form a 1,024-bit register with the prefix Y and Y2 … Y5 are aliased for X4 … X11.

|Table 10: AIE-ML Vector Registers|Col2|Col3|
|**256-bit**|**512-bit**|**1024-bit**|

**Mask Registers**

In addition to the vector registers, there are 4 x 128-bit mask registers (Q0 to Q3) used for sparsity. See Sparsity for more information.

**Accumulator Registers**

Accumulator registers are used to store the results of the vector data path. 256 bit wide, they can be viewed as eight lanes of 32-bit data or four lanes of 64-bit data. The accumulator registers are prefixed with am. Two of them are aliased to form a 512-bit register prefixed with bm, and two bm can be aliased to form a 1024-bit register prefixed with cm.

|Table 11: Accumulator Registers|Col2|Col3|
|**256-bit**|**512-bit**|**1024-bit**|

### **Instruction Fetch and Decode Unit**

The instruction fetch and decode unit sends out the current program counter (PC) register value as an address to the program memory. The program memory returns the fetched 128-bit wide instruction value. The instruction value is then decoded, and all control signals are forwarded to the functional units of the AIE-ML. The program memory size on the AIE-ML is 16 KB, which allows storing 1024 instructions of 128-bit each.

The AIE-ML instruction size ranges from 16 to 128 bits and support multiple instruction formats and variable length instructions to reduce the program memory size. In most cases, the full 128 bits are needed when using all VLIW slots. However, for many instructions in the outer loops, main program, control code, or occasionally the pre- and post-ambles of the inner loop, the shorter format instructions are sufficient, and can be used to store the more compressed instructions with a small instruction buffer.

_Chapter 4:_ AIE-ML Architecture

### **Load and Store Unit**

The AIE-ML has two load units and one store unit for accessing data memory. Data is loaded or stored in data memory.

Each of the load or store units has an address generation unit (AGU). AGUA and AGUB are the load units and the store unit is AGUS. Each AGU has a 20-bit input from the P-register file and a 20-bit input from the M-register file (refer to the pointer registers and the modifier registers in Register Files). The AGU has a one cycle latency.

An individual data memory block is 64 KB. The AIE-ML accesses four 64 KB data memory blocks to create a 256 KB unit. These four memory blocks are located on each side of the AIE-ML and

0xFFE0 0xFFF0

In a logical representation the 256 KB memory can be viewed as one contiguous 256 KB block or four 64 KB blocks, and each block can be divided into odd and even banks. The memory can also be viewed as eight 32 KB banks (four odd and four even). The AGU generates addresses for data memory access that span from 0x0000 to 0x3FFFF (256 KB).

One of the load units supports online decompression of activations/weights.

### **Scalar Unit**

and scalar functional units.

The scalar unit contains the following functional blocks.

- Register files and special registers

- Arithmetic and logical unit (ALU)

Integer add, subtract, compare, and shift functions are one-cycle operations. The integer multiplication operation has a two-cycle latency.

##### **Arithmetic Logic Unit and Scalar Functions**

The arithmetic logic unit (ALU) in the AIE-ML manages the following operations. In all cases the issue rate is one instruction per cycle.

- Integer addition and subtraction: 32 bits. The operation has a one cycle latency.

- Bit-wise logical operation on 32-bit integer numbers (BAND, BOR, BXOR). The operation has a one cycle latency.

- Integer multiplication: 32 x 32 bit with output result of 32 bits stored in the R register file. The operation has a two cycle latency.

- Shift operation: Both left and right shift are supported. A positive shift amount is used for left shift and a negative shift amount is used for right shift. The shift amount is passed through a general purpose register. A one bit operand to the shift operation indicates whether a positive or negative shift is required. The operation has a one-cycle latency.

There is no floating point unit in the scalar unit. The floating point operations are supported through emulation. In general, it is preferred to perform add and multiply in the vector unit.

_Chapter 4:_ AIE-ML Architecture

### **Vector Unit**

##### **Fixed-Point Vector Unit**

**AIE-ML Fixed-Point Vector Unit**

The following is a block diagram of the fixed-point vector data path. The datapath is split into five pipeline stages.

|Permute PRMX Permute PRMY E2|Col2|Col3|
|**Permute PRMY**<br>E2<br>**Permute PRMX**|||

E3

The features of the units in the datapath are as follow:

- The multiplier unit is fed by the output of the permute blocks. The vector adder is in a separate functional unit together with a vector shuffle and shift datapath.

- There are two permute units PRMX and PRMY that handle a set of permutes of X vector registers.

- In addition to the permute and multiplier, there are two additional vector units: shuffle/shift and add/compare. The input comes directly from two vector registers and the results are stored back in the vector registers. The supported bit-width modes are (both signed and unsigned):

- 16 lanes of 32-bit

- 32 lanes of 16-bit

- 64 lanes of 8-bit

The unit supports lane-by-lane control whether addition or subtraction is performed.

The previous image shows that in addition to the vector adder, there is a vector shuffle and shift datapath. The vector shift unit takes one or two 512-bit vector registers as an input and produces one 512-bit output vector. It supports the following modes:

- Standard right shift with 8-bit granularity

- Shift and push in scalar value either at the left or right-hand side. An 8, 16, or 32-bit lane can be shifted into the LSB lane of a 512-bit vector register, and all existing values are shifted one lane up. The value of MSB lane is dropped.

The shuffle unit allows different modes to transform the input vectors. It supports the following features:

- Interleaving and deinterleaving of values at 8-bit, 16-bit, and 32-bit

- Extraction of upper and lower half of the transformed input.

**Fixed-Point SRS and UPS Conversions**

|Col1|Accumulator Register A|Col3|Col4|Col5|Col6|Col7|
|**FP Downshift**<br>**Integer SRS**<br>E2|**FP Downshift**<br>**Integer SRS**<br>E2|**FP Downshift**<br>**Integer SRS**<br>E2|**FP Downshift**<br>**Integer SRS**<br>E2|**FP Downshift**<br>**Integer SRS**<br>E2|**FP Downshift**<br>**Integer SRS**<br>E2|**FP Downshift**<br>**Integer SRS**<br>E2|
|**VEC**<br>E3|**VEC**<br>E3|**VEC**<br>E3|**VEC**<br>E3|**VEC**<br>E3|**VEC**<br>E3|**VEC**<br>E3|

The SRS unit reads an accumulator register, performs the conversion, and restores the result either back to the vector register or directly to memory. The UPS unit reads a vector register directly from memory or a register and stores the result into an accumulator register. The supported modes include:

- 32 lanes of 8-bit to/from 32-bit conversion

- 32 lanes of 16-bit to/from 32-bit conversion

- 16 lanes of 16-bit to/from 64-bit conversion

- 16 lanes of 32-bit to/from 64-bit conversion

A floating-point conversion mode is also supported. It converts bfloat16 to single precision or vice versa. The modes supported are:

- 16 lanes of fp32 accumulators to bfloat16 vector registers

- 16 lanes of bfloat16 vector registers to fp32 accumulators

In addition (not shown in the figure), the unit also supports floating-point to integer conversion:
16 lanes of bfloat16 vector registers to 32-bit signed registers.

##### **Floating-Point Vector Unit**

8 x 8 multipliers with half of the integer datapath along with additional blocks for floating point exponent compute and mantissa shifting and normalization.

|Vector Register X|Col2|Col3|Col4|

E1

|Col1|E3<br>ACC ACC|Col3|

To reduce the accumulation feedback loop, multiple accumulator registers are used to allow back-to-back floating-point MAC instructions.

The BF and FP mantissa shift unit shifts down each of the 128 multiplier lanes and the 2 x 16 accumulator lanes. The accumulator unit supports addition/subtract/negate of accumulator registers in a single-precision FP32 format. All floating-point additions are done in one go, by aligning all mantissas to the one with the largest exponent and with 23 bits of fractional bits.
The FP normalization unit handles the cases where the mantissa coming from the post-adder is negative and if the mantissa is outside the acceptable range.

The AIE-ML supports several vector element-wise functions for the bfloat16 format. These functions include a vector comparison, minimum, and maximum. They operate in an elementwise fashion comparing two vectors. The separate fixed-point vector add/compare unit is extended to handle the floating-point elementary function.

The floating-point unit can issue events that correspond to standard floating-point exceptions and the status registers keep track of the events. There are eight exception bits per floatingpoint functional unit. The exceptions are (from bit 0 to 7): zero, infinity, tiny (underflow), huge (overflow), inexact, huge integer, and divide-by-zero. Of the eight exceptions, tiny, huge, invalid, and divide-by-zero can be converted into an event that can be broadcast to the AIE-ML array interface and then sent to the PS/PMC as an interrupt.

Denormalized numbers are not supported by the AIE-ML floating-point data path.

### **Register Move Functionality**

The register move capabilities of the AIE-ML are covered in this section (refer to the Register Files section for a description of the naming of register types.

- Scalar to scalar:

- Move scalar values between R, M, P, and special registers.

- Move immediate values to R, M, P, and special registers.

- Move a scalar value to/from an AXI4-Stream.

- Vector to vector: Move one 128-bit V-register to an arbitrary V-register in one cycle. It also applies to the 256-bit W-register and the 512-bit X-register. However, vector sizes must be the same in all cases.

- Accumulator to accumulator: Move one 512-bit accumulator (AM) register to another AMregister in one cycle. There is also register BM to BM accumulator register move (1024 bits).

- Vector to accumulator: there are three possibilities:

- Up shift path takes 16 or 32-bit vector values and writes into an accumulator.

- Use the normal multiplication datapath and multiply each value by a constant value of 1.

- Move between BM and X registers.

- Accumulator to vector: Shift-round saturate datapath moves the accumulator to a vector register. There is also a direct register move from accumulator to vector register.

- Accumulator to cascade stream and cascade to accumulator: Cascade stream connects the AIE-MLs in the array in a chain and allows the AIE-MLs to transfer an accumulator register (512-bit) from one to the next. A small two-deep 512-bit wide FIFO on both the input and output streams allows storing up to four values in the FIFOs between the AIE-MLs.

- Scalar to vector: Moves a scalar value from an R-register to a vector register. Different from AIE where most operations were on the 128-bit granularity except for shift element operation, only operations on 512-bit registers are allowed in AIE-MLs.

- Vector to scalar: Extracts an arbitrary 8, 16, or 32-bit value from a 512-bit vector register and writes results into a scalar R-register.