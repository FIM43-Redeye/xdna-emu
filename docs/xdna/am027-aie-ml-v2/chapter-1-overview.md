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

The Versal RF Series incorporates high-frequency, high sample-rate RF-sampling data converters with integrated signal processing IP for high-performance RF systems. These IP tiles provide a low power consumption and latency data interface between the DACs/ADCs and programmable logic. RF-DACs and RF-ADCs also include programmable filters and digital up and down converters (DUCs/DDCs).

The Versal HBM Series enables the convergence of fast memory, adaptable compute, and secure connectivity in a single platform. The series is architected to keep up with the higher memory needs of the most compute intensive, memory-bound applications, providing adaptable acceleration for data center, communications, test & measurement, and aerospace & defense applications. Versal HBM adaptive SoCs integrate the most advanced HBM2e DRAM, providing high memory bandwidth and capacity within a single device.

[The Versal architecture documentation suite is available at: https://www.amd.com/versal.](https://www.amd.com/versal)

### **Navigating Content by Design Process**

AMD Adaptive Computing documentation is organized around a set of standard design processes to help you find relevant content for your current development task. You can access [the AMD Versal™ adaptive SoC design processes on the Design Hubs page. You can also use the](https://docs.amd.com/p/design-hubs) [Design Flow Assistant to better understand the design flows and find content that is specific to](https://docs.amd.com/p/versal-decision-tree-welcome) your intended design needs. This document covers the following design processes:

- **System and Solution Planning:** Identifying the components, performance, I/O, and data transfer requirements at a system level. Includes application mapping for the solution to PS, PL, and AI Engine. Topics in this document that apply to this design process include:

- Chapter 1: Overview provides an overview of the AIE-ML v2 architecture and includes:

- AIE-ML v2 Array Overview

- AIE-ML v2 Array Hierarchy

- Performance

- Chapter 2: AIE-ML v2 Tile Architecture describes the interaction between the memory module and the interconnect and between the AIE-ML v2 and the memory module.

- Chapter 3: AIE-ML v2 Array Interface Architecture is a high-level view of the AIE-ML v2 array interface to the PL and NoC.

- Chapter 4: AIE-ML v2 Architecture describes the processor functional unit and register files.

- Chapter 5: AIE-ML v2 Memory Tile Overview and Features describes the features and functionality of the AIE-ML v2 memory tile, which is an additional functional unit in the AIE-ML v2 architecture.

- Chapter 6: AIE-ML v2 Configuration and Boot describes configuring the AIE-ML v2 array from the processing system during boot and reconfiguration.

- **AI Engine Development:** Creating the AI Engine graph and kernels, library use, simulation debugging and profiling, and algorithm development. Also includes the integration of the PL and AI Engine kernels. Topics in this document that apply to this design process include:

- Chapter 2: AIE-ML v2 Tile Architecture

- Chapter 4: AIE-ML v2 Architecture

- Chapter 5: AIE-ML v2 Memory Tile Overview and Features

### **Introduction to AI Engines and AIE-ML v2**

Next-generation wireless, data center, automotive, and industrial applications demand significant increases in compute acceleration without increasing board area and while remaining power efficient. With Moore's Law and Dennard Scaling no longer following their traditional trajectories, simply porting existing hardware architectures to next-generation process nodes is not sufficient to meet the system-level performance requirements of these applications – architectural innovation is critical to overcome the diminishing gains from process scaling. To address this need, AMD has developed an innovative processing technology, the AI Engine, as part of the Versal architecture.

AI Engines are very long instruction word (VLIW), single instruction multiple data (SIMD) vector processors optimized for both machine learning and advanced signal processing workloads.
AI Engines are arranged in 2D arrays within a given device, with the number of individual processors varying by device, allowing for scalability to meet the compute needs of a breadth of applications. AMD offers two primary types of AI Engines within the Versal portfolio: AI Engines (AIE), which are balanced between signal processing and machine learning workloads, and AI Engines-Machine Learning (AIE-ML), which are optimized for machine learning workloads.
Although both types of AI Engines can be used for either machine learning or signal processing functions, differences in native data type support, memory, and other capabilities impact which workloads each AI Engine type is best suited for.

AI Engines-Machine Learning enable high performance per watt with low latency for inference workloads. AIE-ML v2 is the second generation of the AI Engines-Machine Learning architecture. AIE-ML v2 offers several architectural enhancements over the first-generation AIEML architecture, including up to two times compute/tile, new native data type support, and improved energy efficiency.

This architecture manual details the specifics of the AIE-ML v2 architecture.

### **AIE-ML v2 Array Features**

Versal AI Edge Series Gen 2 adaptive SoCs feature an array of AIE-ML v2 tiles, AIE-ML v2 memory tiles, and the AIE-ML v2 array interface. The following lists the features of each. A pictorial view of the array organization is shown in AIE-ML v2 Array Overview.

_Chapter 1:_ Overview

##### **AIE-ML v2 Tile Features**

- A separate building block, integrated into the silicon, outside the programmable logic (PL).

- One AIE-ML v2 incorporates a high-performance very-long instruction word (VLIW) singleinstruction multiple-data (SIMD) vector processor optimized for many applications including machine learning applications.

- From a hardware perspective, data memory is 64 KB organized as eight banks of 8 KB (256-bit wide and 256 words deep). From a programmer's perspective, every two banks are interleaved to form one bank, that is, a total of four banks of 512-bit wide.

- Streaming interconnect for deterministic throughput, high-speed data flow between AIE-ML v2 tiles and/or the programmable logic in the Versal device.

- Direct memory access (DMA) in the AIE-ML v2 tile moves data from incoming stream(s) to local memory and from local memory to outgoing stream(s). It has two S2MM channels and two MM2S channels and supports 4-D address generation.

- Configuration interconnect (through memory-mapped AXI4 interface) with a shared, transaction-based switched interconnect for access from external masters to internal AIE-ML v2 tile.

- Hardware synchronization primitives provide synchronization of the AIE-ML v2, between the AIE-ML v2 and the tile DMA, and between the AIE-ML v2 and an external manager (through the memory-mapped AXI4 interface).

- Debug, trace, and profile functionality.

##### **AIE-ML v2 Memory Tile Features**

- A tile containing 512 KB of high-density, high-bandwidth memory to reduce the use of PL resources. Memory is arranged in eight physical banks of 256-bit wide and 2K words deep.

- The memory tile DMA has six MM2S and S2MM channels with 64-bit stream interfaces and 256-bit memory interfaces. The DMA also supports compression and decompression of zeroes in data before sending onto the stream. It also supports 4-D tensor address generation.

- The AXI4-Stream interconnect has north ports, south ports, and the DMA MM2S and S2MM channels that are 64-bit wide. It also has 32-bit trace, control response, and control that are connected using adapters.

- Memory-mapped AXI4 configuration is the same as the AIE-ML v2 tile.

##### **AIE-ML v2 Array Interface to NoC and PL Resources**

- A control module used for setting up the application, orchestrate execution and data movement during runtime for each column. This is also used to allow synchronization with external host.

- Configuration through the memory-mapped AXI4 interface of local registers and control module.

- Streaming interconnect provides routing between PL resources and array tiles.

- AIE-ML v2 to programmable logic (PL) interface that provides asynchronous clock-domain crossing between the AIE-ML v2 clock and the PL clock.

- AIE-ML v2 to NoC interface logic to the NoC manager unit (NMU) and NoC subordinate unit (NSU) components.

- Hardware synchronization primitives leverage features from the AIE-ML v2 tile locks module.

- Debug, trace, and profile functionality that leverage all the features from the AIE-ML v2 tile.

### **AIE-ML v2 Array Overview**

v2 array in it. The device consists of the processor system (PS), programmable logic (PL), and the AIE-ML v2 array.

The AIE-ML v2 array is the top-level hierarchy of the AIE-ML v2 architecture. It integrates a two-dimensional array of AIE-ML v2 tiles. Each AIE-ML v2 tile integrates a very-long instruction word (VLIW) processor, integrated memory, and interconnects for streaming, configuration, and debug. The AIE-ML v2 array introduces a separate functional block, the memory tile, that is used to significantly reduce PL resources (LUTs and UltraRAMs) for ML applications. The memory tile has 512 KB data memory, 12 DMA channels (eight can access neighboring memory tiles), and stream interfaces. Depending on the device there can be one or two rows of memory tiles. The AIE-ML v2 array interface enables the AIE-ML v2 array to communicate with the rest of the Versal device through the NoC or directly to the PL. The AIE-ML v2 array also interfaces to the processing system (PS) and platform management controller (PMC) through the NoC.

Versal adaptive SoC devices that integrate AIE-ML v2 tiles have access to the following types of memory:

- External DDR memory (via NoC)

- On-chip PL memory resources (UltraRAM/block RAM)

- On-chip shared memory in AIE-ML v2 memory tiles

- On-chip local data memory in AIE-ML v2 tiles

### **AIE-ML v2 Array Hierarchy**

The AIE-ML v2 array is made up of AIE-ML v2 tiles, one or two rows of AIE-ML v2 memory conceptual view of the complete tile hierarchy associated with the AIE-ML v2 array. See Chapter 2: AIE-ML v2 Tile Architecture, Chapter 3: AIE-ML v2 Array Interface Architecture, and Chapter 4: AIE-ML v2 Architecture for detailed descriptions of the various tiles.

### **Performance**

The AIE-ML v2 array has a single clock domain for all the tiles and array interface blocks. The performance targed of the AIE-ML v2 array for the –1 speed grade devices is 1 GHz with VCC_AIE at 0.725V. In addition, the AIE-ML v2 array has clocks for interfacing to other blocks.
The following table summarizes the various clocks in the AIE-ML v2 array and their performance targets.

|Table 1: AIE-ML v2 Interfacing Clock Domains|Col2|Col3|Col4|
|**Clock**|**Target for –1L**|**Source**|**Relation to AIE-ML v2 Clock**|
|AIE-ML v2 array clock|1 GHz|AIE-ML v2 PLL|N/A|
|NoC clock|1000 MHz|NoC clocking|Asynchronous, clock domain crossing (CDC) within<br>the NoC|
|PL clocks|500 MHz|PL clocking|Asynchronous, CDC within AIE-ML v2 array interface|
|NPI clock|300 MHz|NPI clocking|Asynchronous|

The following table shows the minimum bandwidth of the AXI4 memory-mapped interface by target block, under the following conditions:

- The AXI4 memory-mapped manager operates ideally.

- Access occurs in bulk and contiguously.

- Bandwidth measures at the AIE-ML v2 array AXI4 NoC subordinate interface (NSU).

- AIE-ML v2 array clock runs at 1 GHz.

The actual bandwidth varies depending on:

- Internal bottlenecks and other non-idealities of the AXI4 memory-mapped manager (for example, PS, DMA in PMC, custom AXI4 manager in PL).

- Contention in the NoC.

- The AXI4 memory-mapped burst size and data access pattern.

- The AIE-ML v2 array clock frequency.

|Table 2: AIE‑ML v2 Internal AXI4 Memory-mapped Bandwidth at an AIE‑ML v2 Clock<br>of 1 GHz|Col2|Col3|Col4|Col5|
||**128-bit Access**|**128-bit Access**|**32-bit Access**|**32-bit Access**|
|Block|Target Write BW<br>(GB/s)|Target Write BW<br>(GB/s)|Target Write BW<br>(GB/s)|Target Write BW<br>(GB/s)|
|Tile and memory tile data memory|3.0|1.1|0.80|0.32|
|Tile program memory|2.9|1.1|0.32|0.32|

|Table 2: AIE‑ML v2 Internal AXI4 Memory-mapped Bandwidth at an AIE‑ML v2 Clock<br>of 1 GHz (cont'd)|Col2|Col3|Col4|Col5|
||**128-bit Access**|**128-bit Access**|**32-bit Access**|**32-bit Access**|
|Tile, memory tile, and array interface tile<br>stream-switch and BD registers|3.0|2.0|1.7|0.80|

### **Clocking Structure**

Each AIE-ML v2 interface tile has a column clock gate control register that controls the clock to all the memory tiles and AIE-ML v2 tiles in the same column. The register does not affect the clock of the AIE-ML v2 interface tile. When all the memory tiles and AIE-ML v2 tiles in a column are unused, disabling its clock through this control register reduces the power consumption of the AIE-ML v2 array.

### **Memory Error Handling**

Each AIE-ML v2 tile has 64 KB of data memory and 16 KB of program memory. Due to the large amount of memory in the AIE-ML v2 tiles, protection is provided against soft errors. The AIE-ML v2 program memory 16 KB has ECC protection. The 64 KB of data memory is divided into eight memory banks each of size 8 KB. Two of these banks have ECC protection and the remaining six banks has parity protection. The memory tile data memory has ECC protection.

The program memory is protected with two 8-bit ECC. Each of the 8-bit ECC covers 64-bits. The 8-bit ECC can detect 2-bit errors and detect/correct a 1-bit error within the 64-bit word.

There are eight memory banks in data memory module. The first two memory banks have 7-bit ECC protection for each of the eight 32-bit lanes in the 256-bit word. The 7-bit ECC can detect 2-bit errors and detect/correct a 1-bit error. The last six memory banks have an odd parity for each 32-bit within 256-bit word. If parity error occurs, events are generated from the memory module.

The AIE-ML v2 memory tile has 512 KB of memory formed from eight banks of 64 KB. All memory banks have 7-bit ECC protection on each of the eight 32-bit lines in the 256-bit word.
This ECC corrects 1-bit errors within 32-bit words and detects 2-bit errors within 32-bit words.

The control module contains three SRAMs. It has 32 KB of program memory and a data memory of 16 KB with a 32-bit interface. It also has a shared memory of 32 KB with a 128-bit interface.
The ECC is generated and checked at 32-bit data granularity for the shared memory. In case there are correctable and uncorrectable errors, events are generated. The ECC is checked at 32-bit granularity for program memory and data memory.

_Chapter 1:_ Overview

##### **Privileged Register Access Control**

AIE-ML v2 has a few privileged registers. These registers exist in the AIE-ML v2 tile, memory, and array interface-tiles. The security mechanism controls write requests to these registers and has no effect on reads to these registers. Per the AXI4 protocol, an AXI4 write transaction with AWPROT[1]=0 indicates that the transaction is secure. AIE-ML v2 allows secure transactions to write to privileged registers in the array-interface tile registers. It also allows an external privileged manager to have secured write transactions.