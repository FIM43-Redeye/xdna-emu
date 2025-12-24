_Chapter 1:_ Overview

#### _Chapter 1_

## Overview

### **Introduction to Versal Adaptive SoCs**

AMD Versal™ adaptive SoCs combine programmable logic (PL), processing system (PS), and AI Engines with leading-edge memory and interfacing technologies to deliver powerful heterogeneous acceleration for any application. The hardware and software are targeted for programming and optimization by data scientists and software and hardware developers. A host of tools, software, libraries, IP, middleware, and frameworks enable Versal adaptive SoCs to support all industry-standard design flows.

The Versal portfolio is the first platform to combine software programmability and domainspecific hardware acceleration with the adaptability necessary to meet today's rapid pace of innovation. The portfolio is uniquely architected to deliver scalability and AI inference capabilities for a host of applications across different markets—from cloud—to networking—to wireless communications—to edge computing and endpoints.

The Versal architecture has a wealth of connectivity and communication capability and a programmable network on chip (NoC) to enable seamless memory-mapped access to the full height and width of the device. AI Engines are SIMD VLIW vector processors for adaptive inference and advanced signal processing compute. The PL combines configurable logic blocks, memory, and DSP Engines architected for high-compute density. The PS includes application and real-time processors from Arm [®] for intensive compute tasks.

The Versal AI Edge Series Gen 2 delivers end-to-end acceleration for AI-driven embedded systems—all in a single device built on a foundation of enhanced safety and security. Combining world-class programmable logic with a new high-performance processing system of integrated Arm CPUs and next-generation AI Engines, these devices enable all three phases of compute in embedded AI applications: preprocessing, AI inference, and postprocessing.

The Versal AI Edge Series focuses on AI performance per watt for real-time systems in automated drive, predictive factory and healthcare systems, multi-mission payloads in aerospace & defense, and a breadth of other applications. More than just AI, the Versal AI Edge Series accelerates the whole application from sensor to AI to real-time control, all with the highest levels of safety and security to meet critical standards such as ISO 26262 and IEC 61508.

The Versal AI Core Series delivers breakthrough AI inference acceleration and compute performance. This series is designed for a breadth of applications, including cloud for dynamic workloads and network for massive bandwidth, all while delivering advanced safety and security features. AI and data scientists, as well as software and hardware developers, can all take advantage of the high-compute density to accelerate the performance of any application.

The Versal Prime Series Gen 2 combines world-class programmable logic from AMD with a new high-performance processing system of integrated Arm CPUs—offering significantly greater scalar compute than existing Versal or Zynq™ adaptive SoCs. This powerful combination of flexible, real-time sensor processing and the ability to handle complex embedded computing workloads allows designers to maximize system performance while avoiding the overhead of a multi-chip solution.

The Versal Prime Series is the foundation and the mid-range of the Versal portfolio, serving the broadest range of uses across multiple markets. These applications include 100G to 200G networking equipment, network and storage acceleration in the data center, communications test equipment, broadcast, and aerospace & defense. The series integrates mainstream 58G transceivers and optimized I/O and DDR connectivity, achieving low-latency acceleration and performance across diverse workloads.

The Versal Premium Series Gen 2 offers new levels of memory and data bandwidth with CXL [®]

3.1, PCIe [®] Gen6, and DDR5/LPDDR5X interfacing capabilities, tailored to fit the application requirements of tomorrow’s data center, communications, test & measurement, and aerospace & defense data-intensive applications. As a heterogeneous compute platform, the series is engineered to help users reach high levels of acceleration for a wide range of compute-intensive workloads by providing high compute density, custom memory hierarchy, and DSP Engine resources.

The Versal Premium Series provides breakthrough heterogeneous integration, very highperformance compute, connectivity, and security in an adaptable platform with a minimized power and area footprint. The series is designed to exceed the demands of high-bandwidth, compute-intensive applications in communications, data center, test & measurement, and other applications. The Versal Premium Series includes 112G PAM4 transceivers and integrated blocks for 600G Ethernet, 600G Interlaken, PCI Express [®] Gen5, and high-speed cryptography.

The Versal HBM Series enables the convergence of fast memory, adaptable compute, and secure connectivity in a single platform. The series is architected to keep up with the higher memory needs of the most compute intensive, memory-bound applications, providing adaptable acceleration for data center, communications, test & measurement, and aerospace & defense applications. Versal HBM adaptive SoCs integrate the most advanced HBM2e DRAM, providing high memory bandwidth and capacity within a single device.

[The Versal architecture documentation suite is available at: https://www.amd.com/versal.](https://www.amd.com/versal)

_Chapter 1:_ Overview

##### **Navigating Content by Design Process**

AMD Adaptive Computing documentation is organized around a set of standard design processes to help you find relevant content for your current development task. You can access [the AMD Versal™ adaptive SoC design processes on the Design Hubs page. You can also use the](https://docs.amd.com/p/design-hubs) [Design Flow Assistant to better understand the design flows and find content that is specific to](https://docs.amd.com/p/versal-decision-tree-welcome) your intended design needs. This document covers the following design processes:

- **System and Solution Planning:** Identifying the components, performance, I/O, and data transfer requirements at a system level. Includes application mapping for the solution to PS, PL, and AI Engine. Topics in this document that apply to this design process include:

- Chapter 1: Overview provides an overview of the AIE-ML architecture and includes:

- AIE-ML Array Overview

- AIE-ML Array Hierarchy

- Performance

- Chapter 2: AIE-ML Tile Architecture describes the interaction between the memory module and the interconnect and between the AIE-ML and the memory module.

- Chapter 3: AIE-ML Array Interface Architecture is a high-level view of the AIE-ML array interface to the PL and NoC.

- Chapter 4: AIE-ML Architecture describes the processor functional unit and register files.

- Chapter 5: AIE-ML Memory Tile Architecture describes the features and functionality of the AIE-ML memory tile, which is an additional functional unit in the AIE-ML architecture.

- Chapter 6: AIE-ML Configuration and Boot describes configuring the AIE-ML v2 array from the processing system during boot and reconfiguration.

- **AI Engine Development:** Creating the AI Engine graph and kernels, library use, simulation debugging and profiling, and algorithm development. Also includes the integration of the PL and AI Engine kernels. Topics in this document that apply to this design process include:

- Chapter 2: AIE-ML Tile Architecture

- Chapter 4: AIE-ML Architecture

- Chapter 5: AIE-ML Memory Tile Architecture

_Chapter 1:_ Overview

### **Motivation to AIE-ML**

The non-linear increase in demand in machine learning and other compute intensive applications leads to the development of the AMD Versal™ adaptive SoC AI Engine-Machine Learning (AIEML). The AIE-ML, the dual-core Arm [®] Cortex [®] -A72 and Cortex-R5F processor (PS), and the next generation programmable logic (PL) are all tied together with a high-bandwidth NoC to form a new architecture in adaptive SoC. The AIE-ML and PL are intended to complement each other to handle functions that match their strengths. With the custom memory hierarchy, multi-cast stream capability on AI interconnect and AI-optimized vector instructions support, the Versal adaptive SoC AIE-MLs are optimized for various compute-intensive applications, for example machine learning inference acceleration in data center applications by enabling deterministic latency and low neural network latency with high performance per watt.

### **AIE-ML Array Features**

AMD developed multiple iterations of AI Engines. This architecture manual details the specifics of the AIE-ML.

Some Versal adaptive SoCs include the AIE-ML that consists of an array of AIE-ML tiles, AIE-ML memory tiles, and the AIE-ML array interface consisting of the network on chip (NoC) and programmable logic (PL) tiles. The following lists the features of each. A pictorial view of the array organization is shown in Figure 1: Versal Device (with AIE-ML) Top-Level Block Diagram.

**AIE-ML Tile Features**

- A separate building block, integrated into the silicon, outside the programmable logic (PL).

- One AIE-ML incorporates a high-performance very-long instruction word (VLIW) singleinstruction multiple-data (SIMD) vector processor optimized for many applications including machine learning applications.

- From a hardware perspective, data memory is 64 KB organized as eight banks of 8 KB. From a programmer's perspective, every two banks are interleaved to form one bank, that is, a total of four banks of 16 KB each.

- Streaming interconnect for deterministic throughput, high-speed data flow between AIE-ML tiles and/or the programmable logic in the Versal device.

- Direct memory access (DMA) in the AIE-ML tile moves data from incoming stream(s) to local memory and from local memory to outgoing stream(s).

- Configuration interconnect (through memory-mapped AXI4 interface) with a shared, transaction-based switched interconnect for access from external masters to internal AIE-ML tile.

- Hardware synchronization primitives provide synchronization of the AIE-ML, between the AIE-ML and the tile DMA, and between the AIE-ML and an external master (through the memory-mapped AXI4 interface).

- Debug, trace, and profile functionality.

- The AIE-ML tile has additional granularity on clock gating and reset. Clock gating and reset of the AIE-ML tile can be done via the memory-mapped AXI4 register inside the tile. In AIE-ML the memory-mapped AXI4, clock gating, and reset registers are moved into an always-on domain to give modular control to core, stream-switch and memory module in the tile. A similar arrangement also applies to the memory tile, a functional unit in the AIE-ML that is introduced in the following section.

**AIE-ML Memory Tile Features**

- A tile containing 512 KB of high-density, high-bandwidth memory to reduce the use of PL resources in machine learning (ML) applications.

- The memory tile DMA has the same channel features as the AIE-ML tile with the exception that the memory tile DMA also supports 4-D addressing modes. See Chapter 5: AIE-ML Memory Tile Architecture for a more detailed description.

- AXI4-Stream interconnect is the same as AIE-ML tile except the number of ports and connectivity are different.

- Memory-mapped AXI4 configuration is the same as the AIE-ML tile.

**AIE-ML Array Interface to NoC and PL Resources**

- Direct memory access (DMA) in the AIE-ML NoC interface tile manages incoming and outgoing memory-mapped and streams traffic into and out of the AIE-ML array. The interface tile is described in Chapter 3: AIE-ML Array Interface Architecture.

- Configuration and control interconnect functionality (through the memory-mapped AXI4 interface)

- Streaming interconnect that leverages the AIE-ML tile streaming interconnect functionality.

- AIE-ML to programmable logic (PL) interface that provides asynchronous clock-domain crossing between the AIE-ML clock and the PL clock.

- AIE-ML to NoC interface logic to the NoC master unit (NMU) and NoC slave unit (NSU)
components.

- Hardware synchronization primitives leverage features from the AIE-ML tile locks module.

- Debug, trace, and profile functionality that leverage all the features from the AIE-ML tile.

- For a list of changes from the AI Engine (AIE) array interface, see Chapter 3: AIE-ML Array Interface Architecture.

_Chapter 1:_ Overview

### **AIE-ML Array Overview**

acceleration platforms (adaptive SoCs) with an AIE-ML array in it. The device consists of the processor system (PS), programmable logic (PL), and the AIE-ML array.

The AIE-ML array is the top-level hierarchy of the AIE-ML architecture. It integrates a twodimensional array of AIE-ML tiles. Each AIE-ML tile integrates a very-long instruction word (VLIW) processor, integrated memory, and interconnects for streaming, configuration, and debug.
The AIE-ML array introduced a separate functional block, the memory tile, that is used to significantly reduce PL resources (LUTs and UltraRAMs) for ML applications. The memory tile

has 512 KB data memory, 12 DMA channels (eight can access neighboring memory tiles) and stream interfaces. Depending on the device there can be one or two rows of memory tiles. The AIE-ML array interface enables the AIE-ML array to communicate with the rest of the Versal device through the NoC or directly to the PL. The AIE-ML array also interfaces to the processing system (PS) and platform management controller (PMC) through the NoC.

AMD Versal™ adaptive SoC devices that integrate AIE-ML tiles have access to the following types of memory:

- External DDR memory (via NoC)

- On-chip PL memory resources (UltraRAM/block RAM)

- On-chip shared memory in AIE-ML memory tiles

- On-chip local data memory in AIE-ML tiles

Depending on the use case, the data and weights move through the memory hierarchy in different ways.

### **AIE-ML Array Hierarchy**

The AIE-ML array is made up of AIE-ML tiles, one or two rows of AIE-ML memory tiles, and view of the complete tile hierarchy associated with the AIE-ML array. See Chapter 2: AIE-ML Tile Architecture, Chapter 3: AIE-ML Array Interface Architecture, and Chapter 4: AIE-ML Architecture for detailed descriptions of the various tiles.

_Chapter 1:_ Overview

### **Performance**

The AIE-ML array has a single clock domain for all the tiles and array interface blocks. The performance target of the AIE-ML array for the -1L speed grade devices is 1 GHz with VCCINT at 0.70V. In addition, the AIE-ML array has clocks for interfacing to other blocks. The following table summarizes the various clocks in the AIE-ML array and their performance targets. For more information, see _Versal AI Core Series Data Sheet: DC and AC Switching Characteristics_ [(DS957) or](https://docs.amd.com/go/en-US/ds957-versal-ai-core) _Versal AI Edge Series Data Sheet: DC and AC Switching Characteristics_ [(DS958).](https://docs.amd.com/go/en-US/ds958-versal-ai-edge)

|Table 1: AIE-ML Interfacing Clock Domains|Col2|Col3|Col4|
|**Clock**|**Target for -1L**|**Source**|**Relation to AIE-ML Clock**|
|AIE-ML array clock|1 GHz|AIE-ML PLL|N/A|

|Table 1: AIE-ML Interfacing Clock Domains (cont'd)|Col2|Col3|Col4|
|**Clock**|**Target for -1L**|**Source**|**Relation to AIE-ML Clock**|
|NoC clock|960 MHz|NoC clocking|Asynchronous, clock domain crossing (CDC) within<br>the NoC|
|PL clocks|500 MHz|PL clocking|Asynchronous, CDC within AIE-ML array interface|
|NPI clock|300 MHz|NPI clocking|Asynchronous|

### **Clocking Structure**

Each AIE-ML interface tile has a column clock gate control register that controls the clock to all the memory tiles and AIE-ML tiles in the same column. The register does not affect the clock of the AIE-ML interface tile. When all the memory tiles and AIE-ML tiles in a column are unused, disabling its clock through this control register reduces the power consumption of the AIE-ML array.

### **Memory Error Handling**

Each AIE-ML has 64 KB of data memory and 16 KB of program memory. Due to the large amount of memory in the AIE-ML tiles, protection is provided against soft errors. The 128-bit word in the program memory is protected with two 8-bit ECC (one for each 64-bit). The 8-bit ECC can detect 2-bit errors and detect/correct a 1-bit error within the 64-bit word. The two 64-bit data and two 8-bit ECC fields are each interleaved within its own pair (distance of two) to create larger bit separation.

There are eight memory banks in each data memory module. The first two memory banks have 7-bit ECC protection for each of the four 32-bit fields. The 7-bit ECC can detect 2-bit errors and detect/correct a 1-bit error. The last six memory banks have even parity bit protection for each 32 bits in a 128-bit word. The four 32-bit fields are interleaved with a distance of four.

Error injection is supported for both program and data memory. Errors can be introduced into program memory over memory-mapped AXI4. Similarly, errors can be injected into data memory banks over AIE-ML DMA or memory-mapped AXI4.

When the memory-mapped AXI4 access reads or writes to AIE-ML data memory, two requests are sent to the memory module. On an ECC/parity event, the event might be counted twice in the AIE-ML performance counter. There is duplicate memory access but no impact on functionality. Refer to Chapter 2: AIE-ML Tile Architecture for more information on events and performance counters.

Internal memory errors (correctable and uncorrectable) create internal events that use the normal debug, trace, and profiling mechanism to report error conditions. They can also be used to raise an interrupt to the PMC/PS.

### **Task-Completion-Tokens**

In the context of task-completion tokens, a task refers to a unit of work that is pushed into the input task queue of an actor (such as an AIE-ML tile, AIE-ML memory, and AIE-ML array interface ) through the memory mapped interface. The AIE-ML tile, AIE-ML memory, and AIE-ML array interfaces are referred to as actors, each equipped with an input task queue. A task can be placed in the queue through the AXI4 interface, managed by a controller. The actor processes the task until it is completed before proceeding to the next one. Once all tasks in the queue are finished, the actor enters an idle state, awaiting the insertion of a new task.

Upon the completion of a task, the actor triggers the tile task completion unit to send a task completion token to the stream switch through the control-packet response port. In AIE-ML, these task completion tokens are exposed at the AXIS PLIO interface located at the perimeter of the AI Engine array and are intended to be used through PL.

|NS<br>No<br>NM|NS|U|
|**No**<br>NS<br>NM|NM|U|

The figure shows the controller implemented in soft logic in the PL, however, a valid design methodology is to have the front end of the application run on the PS and only a minimum back-end IP on the PL to service the token streams. The task-complete-token consists of a single stream word. This token is routed back to the issuing controller through the standard stream-switch network. There is an eight-bit controller ID register that determines the ownership of each actor. The lower five bits of the controller ID register are used as the stream packet ID, and the upper three bits are available for further routing in the PL.

### **Sparsity**

Many neural networks generate zero activations between layers due to the use of the ReLU function, which simply clamps the negative values to zero. So, these arbitrary zeros are introduced by the ReLU function in the input and output activations. These zeroes are not transmitted over the AXI4-Streams because of decompression/compression logic added to the AIE-ML tile and the AIE-ML Mem DMAs. The zero-valued data weights can also be compressed offline – moving from external memory in a compressed state and decompressed by the S2MM channel of the tile DMA.

Moreover, AIE-ML cores support on-the-fly decompression during data loading by inserting zero weights before performing convolutions. This feature ensures that only non-zero weights need to be stored in local tile memories.

Compression of activations in AXI4-Streams and on-the-fly decompression of weights during core loads are optional features. The compression/decompression in AIE-ML memory and AIEML tile DMA is controlled by a dedicated bit in the BD. The two primary use cases are shown in

AIE-ML

Tile Data Memory

AIE-ML

Tile Data Memory

weights and activations. This algorithm works with 8-bit data samples and employs a 32-bit mask to encode zero and non-zero bytes within a 256-bit word. Zero bytes are represented as 0 in the bit mask, while non-zero bytes are represented as 1. Zero-valued bytes are omitted from the compressed data. Guard bits are inserted as necessary to ensure 32-bit alignment for the subsequent mask. This compression process is consistently applied to all 256-bit data words.

