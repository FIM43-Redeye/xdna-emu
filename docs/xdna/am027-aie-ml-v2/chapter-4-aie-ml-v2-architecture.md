_Chapter 4:_ AIE-ML v2 Architecture

#### _Chapter 4_

## AIE-ML v2 Architecture

### **Functional Overview**

The AIE-ML v2 is a highly-optimized processor featuring single-instruction multiple-data (SIMD) unit, a vector unit, two load units, one store unit, and an instruction fetch and decode unit.

The features of the AIE-ML v2 include:

- Instruction-based VLIW SIMD processor

- 32-bit scalar RISC processor

- Scalar register files and special registers

- 32 x 32-bit multiplier

- 32-bit add/subtract

- ALU operations like shifts, compares, and logical operations.

- Hardware acceleration for inverse, square root, and inverse square root.

- Vector Multiplication Unit

- Vector unit supporting MAC operations for multiple precisions (for example, 512x 8-bit × 8-bit and 512x 4-bit × 4-bit)

- Sparsity is supported for all integer and floating point modes except for MX block floating point (50% sparsity)

- Support for floating point multiplication (float8, bfloat16, and float16) accumulating in single precision floating point (fp32). For fp8, several formats are supported, such as E4M3 and E5M2, and with different representation capabilities for infinity, NaNs and zeros.

- Support for multiplying MX block floating point types and accumulating in floating point.
The MX9 type has eight bits per block element (sign and mantissa in twos complement), eight bits for the shared exponent, and eight bits for shared sub-tile shifts. The MX6 type has five bits per block element (sign and mantissa in twos complement), eight bits of shared exponent, and eight bits of shared sub-tile shifts. The MX4 type has three bits per block element (sign and mantissa in twos complement), eight bits of shared exponent, and eight bits of shared sub-tile shifts.

- The multiplier|multiplicand can be signed or unsigned. The accumulator is always signed.

- The accumulation can be performed in several operation modes: 64 lanes of 32 bits, or 32 lanes of 64 bits.

- The total number of multipliers and the number of accumulation lanes determine the depth of the post-adding.

|Table 7: Supported Precision Width of the Vector Data Path|Col2|Col3|Col4|Col5|
|**Precision 1**|** Precision 2**|**Number of**<br>**Accumulator Lanes**|**Bits per Accumulator**<br>**Lane**|**Number of MACs**|

- Vector Addition Unit

- Vector unit supporting 8, 16 or 32-bit addition, subtraction, comparison and min/max computation

- Support for non-linear functions: tanh, exp2 (bfloat16 with tanh and exp2, float16 with exp2)

- Support processing of two 512-bit wide vectors

- Includes comparisons and min/max computation for bfloat16/float16 vector

- Load/Store Units

- For loading/storing data and weights

- AGU handles optimized address generation for ML functionality

- Two 512-bit load and one 512-bit store units with aligned addresses

- Supports 2D/3D addressing modes for ML functionality

- Ports to Streaming interconnect switch

- 2×32-bit subordinate port

- 1×32-bit manager port

- Processor bus interface: The processor bus allows the AIE-ML v2 to perform direct read/write access to local tile memory mapped registers.

### **Register Files**

The AIE-ML v2 has several types of registers. Some of the registers are used in different functional units. This section describes the various types of registers.

##### **Scalar Registers**

Scalar registers include configuration registers. See the following table for register descriptions.

|Table 8: Scalar Registers|Col2|Col3|
|**Syntax**|**Number of bits**|**Description**|
|r0..r31|32 bits|General-purpose registers|
|m0..m7|20 bits|Modifier registers|
|p0..p7|20 bits|Pointer registers|

|Special Registers|Col2|Col3|
|_Table 9:_**Special Registers**|_Table 9:_**Special Registers**|_Table 9:_**Special Registers**|
|**Syntax**|**Number of bits**|**Description**|
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
|F|128 bits|MX6 data MSB registers|
|G|64 bits|MX6 sub-title shift registers|
|E|64 bits|MX6 exponent registers|

##### **Vector Registers**

Vector registers are wide to allow SIMD instructions and to be used as operand storage. These registers are prefixed with a _W_ . There are 24 x 256-bit registers: wln and whn, n = 0..11. Two W registers can be grouped to form a 512-bit register prefixed with an X. Two X registers then can be grouped to form a 1024-bit register with the prefix Y.

|Table 10: AIE-ML v2 Vector Registers|Col2|Col3|
|**256-bit**|**512-bit**|**1024-bit**|

|Table 10: AIE-ML v2 Vector Registers (cont'd)|Col2|Col3|
|**256-bit**|**512-bit**|**1024-bit**|

##### **Mask Registers**

In addition to the vector registers, there are 4 x 128-bit mask registers (Q0 to Q3) used for sparsity.

##### **Accumulator Registers**

Accumulator registers are used to store the results of the vector data path and are 512-bit wide, they can be viewed as sixteen lanes of 32-bit data or eight lanes of 64-bit data. The main reason to have wider accumulator registers is to have high precision multiplication and accumulate over those results without bit overflows. The accumulator registers are prefixed with bm. Two of them are aliased to form a 1024-bit register prefixed with cm, and two cm can be aliased to form a 2048-bit register prefixed with dm.

|Table 11: AIE-ML v2 Accumulator Registers|Col2|Col3|
|**512-bit**|**1024-bit**|**2048-bit**|

### **Instruction Fetch and Decode Unit**

The instruction fetch and decode unit sends out the current program counter (PC) register value as an address to the program memory. The program memory returns the fetched 128-bit wide instruction value. The instruction value is then decoded, and all control signals are forwarded to the functional units of the AIE-ML v2. The program memory size on the AIE-ML v2 is 16 KB.

The AIE-ML v2 instruction size ranges from 16 to 128 bits and supports multiple instruction formats and variable length instructions to reduce the program memory size. In most cases, the full 128 bits are needed when using all VLIW slots. However, for many instructions in the outer loops, main program, control code, or occasionally the pre and post-ambles of the inner loop, the shorter format instructions are sufficient.

### **Load and Store Unit**

The AIE-ML v2 has two load units and one store unit for accessing data memory. Data is loaded or stored in data memory.

Each of the load or store units has an address generation unit (AGU). AGUA and AGUB are the load units and the store unit is AGUS. Each AGU has a 20-bit input from the P-register file and a 20-bit input from the M-register file (refer to the pointer registers and the modifier registers in Register Files). The AGU has a one cycle latency.

An individual data memory block is 64 KB. The AIE-ML v2 accesses four 64 KB data memory blocks to create a 256 KB unit. These four memory blocks are located on each side of the

0xFFC0 0xFFE0

In a logical representation the 256 KB memory can be viewed as one contiguous 256 KB block or four 64 KB blocks, and each block can be divided into odd and even banks. The memory can also be viewed as eight 32 KB banks (four odd and four even). The AGU generates addresses for data memory access that span from 0x0000 to 0x3FFFF (256 KB).

### **Scalar Unit**

and scalar functional units. The scalar unit contains the following functional blocks:

- Register files and special registers

- Scalar arithmetic and logical unit (ALU)

Integer add, subtract, compare, and shift functions are one-cycle operations. The integer multiplication operation has a two-cycle latency.

_Chapter 4:_ AIE-ML v2 Architecture

##### **Arithmetic Logic Unit and Scalar Functions**

The arithmetic logic unit (ALU) in the AIE-ML v2 manages the following operations. In all cases the issue rate is one instruction per cycle:

- Integer addition and subtraction: 32 bits. The operation has a one cycle latency.

- Bit-wise logical operation on 32-bit integer numbers (BAND, BOR, BXOR). The operation has a one cycle latency.

- Integer multiplication: 32 x 32 bit with output result of 32 bits stored in the R register file. The operation has a two cycle latency.

- Shift operation: Both left and right shift are supported. A positive shift amount is used for left shift and a negative shift amount is used for right shift. The shift amount is passed through a general purpose register. A one bit operand to the shift operation indicates whether a positive or negative shift is required. The operation has a one-cycle latency.

- Hardware acceleration for inverse, square root, and inverse square root, with support for integer and floating-point values at both input and output.

There is no floating point arithmetic unit in the scalar unit. The floating point operations are supported through emulation. In general, it is preferred to perform add and multiply in the vector unit.

_Chapter 4:_ AIE-ML v2 Architecture

### **Vector Unit**

##### **Fixed-point Vector Unit**

This is the data path for integer vector data. The multiplication data path is split into six pipeline stages as shown in the following diagram.

||128x 16b / 64x 32b /32x 64b|||

The features of the units in the multiplication datapath are as follows:

- There are two permute units that handle a set of permutes of X vector registers.

- The multiplier unit is fed by the output of the permute blocks.

- The post-add and accumulate unit adds the result of the multipliers with up to two accumulator inputs.

-    - 16 lanes of 32-bit

- 32 lanes of 16-bit

- 64 lanes of 8-bit

In addition to the multiplier datapath, there are two additional vector units: shuffle/shift and add/compare. The input comes directly from two vector registers and the results are stored back in the vector registers.

The vector add/compare units support the following bit-width modes (both signed and unsigned):

- 16 lanes of 32-bit

- 32 lanes of 16-bit

- 64 lanes of 8-bit

The vector add/compare unit supports lane-by-lane control whether addition or subtraction is performed. This unit also has limited support for 16-bit and 8-bit floating point formats. This includes the following modes in a lane-by-lane fashion:

- x < y, x > y, x ≤ y, x ≥ y, x < 0

- min(x,y), max(x,y)

The vector shift unit takes one or two 512-bit vector registers as an input and produces one 512-bit output vector. It supports the following modes:

- Standard right shift with 8-bit granularity

- Shift and push in scalar value either at the left or right-hand side. An 8, 16, or 32-bit lane can be shifted into the LSB lane of a 512-bit vector register, and all existing values are shifted one lane up. The value of MSB lane is dropped.

The shuffle unit allows different modes to transform the input vectors. It supports the following features:

- Interleaving and de-interleaving of values at 8-bit, 16-bit, and 32-bit

- Extraction of upper and lower half of the transformed input.

- The shuffle unit shuffles the exponents and sub-tile shifts from MX block floating-point data.

###### **_Fixed-Point SRS and UPS Conversions_**

|Col1|Accumulator Register A|Col3|Col4|Col5|Col6|Col7|
|**FP Downshift**<br>**Integer SRS**<br>E2|**FP Downshift**<br>**Integer SRS**<br>E2|**FP Downshift**<br>**Integer SRS**<br>E2|**FP Downshift**<br>**Integer SRS**<br>E2|**FP Downshift**<br>**Integer SRS**<br>E2|**FP Downshift**<br>**Integer SRS**<br>E2|**FP Downshift**<br>**Integer SRS**<br>E2|
|**VEC**<br>E3|**VEC**<br>E3|**VEC**<br>E3|**VEC**<br>E3|**VEC**<br>E3|**VEC**<br>E3|**VEC**<br>E3|

The SRS unit reads an accumulator register, performs the conversion, and restores the result either back to the vector register or directly to memory. The UPS unit reads a vector register directly from memory or a register and stores the result into an accumulator register. The supported modes include:

- 8-bit to/from 32-bit conversion

- 16-bit to/from 32-bit conversion

- 16-bit to/from 64-bit conversion

- 32-bit to/from 64-bit conversion

The core supports a floating-point conversion mode which converts bfloat16/float16/bfloat8/ float8 to a single precision or the other way around. The following modes are supported:

- Floating-point to floating-point conversions:

- 16/32 lanes of fp32 accumulators to bfloat16/float16 vector registers

- 16/32 lanes of bfloat16/float16 vector registers to fp32 accumulators

- 32 lanes of fp32 accumulators to bfloat8/float8 vector registers

- Floating-point to integer conversions: 16 lanes of bfloat16 vector registers to 32-bit signed integers

- Block Floating-point to floating-point conversions:

- A block of MX9 data converts to single precision floating-point accumulators

- A block of MX6 data converts to single precision floating-point accumulators

- A block of MX4 data converts to single precision floating-point accumulators

- Floating-point to MX conversions:

- A block of single precision floating-point accumulators converts to a block of MX9 with 16 8-bit mantissas with a shared exponent and sub-tile shift values.

- A block of single precision floating-point accumulators converts to a block of MX6 with 16 5-bit mantissas with a shared exponent and sub-tile shift values.

- A block of single precision floating-point accumulators converts to a block of MX4 with 16 3-bit mantissas, a shared exponent and sub-tile shift values.

##### **Floating-point Vector Unit**

The floating-point vector data path is split in six pipeline stages. The mantissas from the MX and floating-point datablocks are pre-processed in the same fashion as the integer data path through the permute units. The MX exponents come from the E register file and are used in the alignment stage to prepare for addition and normalization using the fp32 accumulation data path. The MX sub-tile shift values come from the G register file.

The MX and floating-point shift unit shifts down the post-add output and the two accumulator lanes. The accumulator unit supports addition/subtract/negate of accumulator registers in singleprecision FP32 format. All floating-point additions are done in one go, by aligning all mantissas to the one with the largest exponent and with 23 bits of fractional bits. The floating-point normalization unit converts the accumulation result to FP32.

### **Register Move Functionality**

The register move capabilities of the AIE-ML v2 are covered in this section (refer to the Register Files section for a description of the naming of register types.

- **Scalar to scalar:**

- Move scalar values between R, M, P, and special registers.

- Move immediate values to R, M, P, and special registers.

- Move a scalar value to/from an AXI4-Stream.

- **Vector to vector:**

Move one 256-bit W-register to an arbitrary W-register in one cycle. It also applies to the 512-bit X-register and the 1024-bit Y-register. However, vector sizes must be the same in all cases.

- **Accumulator to accumulator:** Move one 512-bit accumulator (BM) register to another BMregister in one cycle. There is also register CM to CM accumulator register move (1024 bits).

- **Vector to accumulator:** There are three possibilities:

- Up shift path takes 16 or 32-bit vector values and writes into an accumulator.

- Use the normal multiplication datapath and multiply each value by a constant value of 1.

- Move between BM and X registers.

- **Accumulator to vector:** Shift-round saturate datapath moves the accumulator to a vector register. There is also a direct register move from accumulator to vector register.

- **Accumulator to cascade stream and cascade to accumulator:** Cascade stream connects the AIE-ML v2 in the array in a chain and allows the AIE-ML v2 to transfer an accumulator register (512-bit) from one to the next. A small two-deep 512-bit wide FIFO on both the input and output streams allows storing up to four values in the FIFOs between the AIE-ML v2s.

- **Scalar to vector:** Moves a scalar value from an R-register to a vector register.

- **Vector to scalar:** Extracts an arbitrary 8, 16, or 32-bit value from a 512-bit vector register and writes results into a scalar R-register.