_Chapter 2:_ AIE-ML Tile Architecture

#### _Chapter 2_

## AIE-ML Tile Architecture

The top-level block diagram of the AIE-ML tile architecture, key building blocks, and connectivity

Integrated synchronization primitives (locks)

The AIE-ML tile consists of the following high-level modules:

- Tile interconnect

- AIE-ML

- AIE-ML memory module

The tile interconnect module handles AXI4-Stream and memory mapped AXI4 input/output traffic. The memory-mapped AXI4 and AXI4-Stream interconnect is further described in the following sections. The AIE-ML memory module has 64 KB of data memory divided into eight memory banks, a memory interface, DMA, and locks. There is a DMA in both incoming and outgoing directions and there is a Locks block within each memory module. The AIE-ML can

access memory modules in all four directions as one contiguous block of memory. The memory interface maps memory accesses in the right direction based on the address generated from the AIE-ML. The AIE-ML has a scalar datapath, a vector datapath, three address generators, and 16 KB of program memory. The program and data memory of AIE-ML tile initializes to zero at boot and resets. It also has a cascade stream access for forwarding accumulator output to the next AIE-ML tile. The AIE-ML is described in more detail in Chapter 4: AIE-ML Architecture. Both the AIE-ML and the AIE-ML memory module have control, debug, and trace units. Some of these units are described later in this chapter:

- Control and status registers

- Events, event broadcast, and event actions

- Performance counters for profiling and timers

interconnect units arrayed together. Sharing data with local memory between neighboring AIEMLs is the main mechanism for data movement within the AIE-ML array. Each AIE-ML can access up to four memory modules:

- Its own

- The module on the north

- The module on the south

- The module on the west

The AIE-ML on the edges of the array have access to one or two fewer memory modules.

Together with the flexible and dedicated interconnects, the AIE-ML array provides deterministic performance, low latency, and high bandwidth. The modular and scalable architecture allows more compute power as more tiles are added to the array.

The AIE-ML has both horizontal and vertical cascade connections, directed from north to south and from west to east. The cascade start points and end points are tied off at the array edges.

### **Memory-mapped AXI4 Interconnect**

Each AIE-ML tile contains a memory-mapped AXI4 interconnect for use by external blocks to write to or read from any of the registers or memories in the AIE-ML tile. The memory-mapped AXI4 interconnect inside the AIE-ML array can be driven from outside of the AIE-ML array by any AXI4 master that can connect to the network on chip (NoC). All internal resources in an AIE-ML tile including memory, and all registers in an AIE-ML and AIE-ML memory module, are mapped onto a memory-mapped AXI4 interface.

Each AIE-ML tile has a memory-mapped AXI4 switch that accepts all memory-mapped AXI4 accesses from the south direction. If the address is for the tile, access occurs. Otherwise, the access is passed to the next tile in the north direction.

The increase in the tile address space is to accommodate the memory tile (512 KB) and also the increase in tile data memory from 32 KB to 64 KB. The lower 20 bits represent the tile address range, followed by five bits that represent the row location and seven bits that represent the column location.

|Column|Row [22:18]|AIE-ML Tile/Memory Tile/Array<br>Interface Tile|
|[31:25] (7b)|[24:20] (5b)|[19:0] (20b)|

### **AXI4-Stream Interconnect**

Each AIE-ML tile has an AXI4-Stream interconnect (alternatively called a stream switch) that is a fully programmable, 32-bit, AXI4-Stream crossbar, and is statically configured through the memory-mapped AXI4 interconnect. It handles backpressure and is capable of the full bandwidth switch. The switch has master ports (data flowing from the switch) and slave ports (data flowing to the switch). The building blocks of the AXI4-Stream interconnect are as follows.

- Port handlers

- FIFOs

- Arbiters

- Stream switch configuration registers

The following lists some of the features of the AXI4-Stream interconnect:

- AIE-ML features 1-to-1 loopback, where only ports with the same ID are connected to each other

- There are 25 slave ports and 23 master ports

- The switch has one FIFO that is 16-deep and 34 bit (32 bit + 1 bit parity + 1 bit TLAST)

In AIE-ML, the ports are divided into external and local ports. External ports are South, West, North, and East. The local ports are AIE-ML, DMA, FIFO, and trace. The features of the ports are as follows:

- External ports are 2-cycle latency and a 4-deep FIFO

- Local slave ports are 2-cycle latency and a 4-deep FIFO

- Local master ports have one register slice with 1-cycle latency and a 2-deep FIFO

Therefore, the latency and buffering crossing the switch are (excluding packet switch arbitration overhead):

- Local slave to local master: 3-cycle latency and 6-deep FIFO

- Local slave to external master: 4-cycle latency and 8-deep FIFO

- External slave to local master; 3-cycle latency and 6-deep FIFO

- External to external: 4-cycle latency and 8-deep FIFO

Each stream port can be configured for either circuit-switched or packet-switched streams (never at the same time) using a packet-switching bit in the configuration register. A circuit-switched stream is a one-to-many streams. This means that it has exactly one source port and an arbitrary number of destination ports. All data entering the stream at the source is streamed to all destinations. A packet-switched stream can share ports (and therefore, physical wires) with other logical streams. Because there is a potential for resource contention with other packet-switched streams, they do not provide deterministic latency. The latency for the word transmitted in a circuit-switched stream is deterministic; if the bandwidth is limited, the built-in backpressure causes performance degradation.

A packet-switched stream is identified by a 5-bit ID which has to be unique amongst all streams it shares ports with. The stream ID also identifies the destination of the packet. A destination can be an arbitrary number of master ports and packet-switched streams make it possible to realize all combinations of single/multiple master/slave ports in any given stream.

A packet-switched packet has:

- **Packet header:** Routing and control information for the packet

- **Data:** Actual data in the packet

- **TLAST:** Last word in the packet must have TLAST asserted to mark the end of packet

The packet header is shown here:

|Table 2: Packet Header|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|**Odd**<br>**Parity**|**3'b000**|**Source Column**|**Source Row**|**1'b0**|**Packet**<br>**Type**|**7'b0000000**|**Stream ID**|

The following table summarizes the AXI4-Stream tile interconnect bandwidth for the -1L speed grade devices.

|Table 3: AIE-ML AXI4-Stream Tile Interconnect Bandwidth|Col2|Col3|Col4|Col5|Col6|
|**Connection Type**|**Number of**<br>**Connections**|**Data**<br>**Width**<br>**(bits)**|**Clock Domain**|**Bandwidth per**<br>**Connection**<br>**(GB/s)**|**Aggregate**<br>**Bandwidth**<br>**(GB/s)**|
|To North/From South|6|32|AIE-ML (1 GHz)|4|24|
|To South/From North|4|32|AIE-ML (1 GHz)|4|16|
|To West/From East|4|32|AIE-ML (1 GHz)|4|16|
|To East/From West|4|32|AIE-ML (1 GHz)|4|16|

_Chapter 2:_ AIE-ML Tile Architecture

### **AIE-ML Tile Program Memory**

The AIE-ML has a local 16 KB of program memory that can be used to store VLIW instructions.
There are two interfaces to the program memory:

- Memory-mapped AXI4 interface

- AIE-ML interface

An external master can read or write to the program memory using the memory-mapped AXI4 interface. The AIE-ML has 128-bit wide interfaces to the program memory to fetch instructions.
The AIE-ML can read from, but not write to, the program memory. To access the program memory simultaneously from the memory-mapped AXI4 and AIE-ML, divide the memory into multiple banks and access mutually exclusive parts of the program memory. Arbitration logic is needed to avoid conflicts between accesses and to assign priority when accesses are to the same bank.

### **AIE-ML Interfaces**

The AIE-ML has multiple interfaces. The following block diagram shows the interfaces.

- **Data Memory Interface:** The AIE-ML can access data memory modules on all four directions.
They are accessed as one contiguous memory. The AIE-ML has two 256-bit wide load units and one 256-bit wide store unit. From the AIE-MLs perspective, the throughput of each of the loads (two) and store (one) is 256 bits per clock cycle.

- **Program Memory Interface:** This 128-bit wide interface is used by the AIE-ML to access the program memory. A new instruction can be fetched every clock cycle.

- **Direct AXI4-Stream Interface:** The AIE-ML has one 32-bit input AXI4-Stream interfaces and one 32-bit output AXI4-Stream interfaces. There are no 128-bit stream interfaces and no FIFO connected to either input or output of the stream.

- **Cascade Stream Interface:** The 512-bit accumulator data from one AIE-ML can be forwarded to another by using these cascade streams to form a chain. There is a small, two-deep, 512-bit wide FIFO on both the input and output streams that allow storing up to four values between AIE-MLs. In addition to the horizontal cascade, there is additional vertical cascade interface and the direction is controlled by the configuration memory-mapped AXI4 register.

- **Debug Interface:** This interface is able to read or write all AIE-ML registers over the memorymapped AXI4 interface.

- **Hardware Synchronization (Locks) Interface:** This interface allows synchronization between two AIE-MLs or between an AIE-ML and DMA. The AIE-ML can access the lock modules in all four directions. There is also added support for semaphore locks.

- **Stall Handling:** An AIE-ML can be stalled due to multiple reasons and from different sources.
Examples include: external memory-mapped AXI4 master (for example, PS), lock modules, empty or full AXI4-Stream interfaces, data memory collisions, and event actions from the event unit.

- **AIE-ML Event Interface:** This 16-bit wide EVENT interface can be used to set different events.

- **Tile Timer:** The input interface to read the 64-bit timer value inside the tile.

- **Execution Trace Interface:** A 32-bit wide interface where the AIE-ML generated packet-based execution trace can be sent over the AXI4-Stream.

### **AIE-ML Memory Module**

input streams to memory map (S2MM) DMA, two memory-map to output DMA streams (MM2S), and a hardware synchronization module (locks). For each of the four directions (south, west, north, and east), there are separate ports for even and odd ports, and three address generators, two loads, and one store.

- **Memory Banks:** The AIE-ML data memory is 64 KB, organized as eight memory banks, where each memory bank is a 512 word x 128-bit single-port memory. From a programmer's perspective, every two banks are interleaved to form one bank, that is, a total of four banks of 16 KB. Each memory bank has a write enable for each 32-bit word. Banks [0-1] have ECC protection and banks [2-7] have parity check. Bank [0] starts at address 0 of the memory module. ECC protection is a 1-bit error detector/corrector and 2-bit error detector per 32-bit word.

- **Memory Arbitration:** Each memory bank has its own arbitrator to arbitrate between all requesters. The memory bank arbitration is round-robin to avoid starving any requester. It handles a new request every clock cycle. When there are multiple requests in the same cycle to the same memory bank, only one request per cycle is allowed to access the memory. The other requesters are stalled for one cycle and the hardware retries the memory request in the next cycle.

- **Tile DMA Controller:** The tile DMA has two incoming and two outgoing streams to the stream switches in the AIE-ML tile. The tile DMA controller is divided into two separate modules, S2MM to store stream data to memory (32-bit data) and MM2S to write the contents of the memory to a stream (32-bit data). Each DMA transfer is defined by a DMA buffer descriptor and the DMA controller has access to the 16 buffer descriptors. These buffer descriptors can also be accessed using a memory-mapped AXI4 interconnect for configuration. Each buffer descriptor contains all information needed for a DMA transfer and can point to the next DMA transfer for the DMA controller to continue with after the current DMA transfer is complete.
The DMA controller also has access to the 16 locks that are the synchronization mechanism used between the AIE-ML and DMA or any external memory-mapped AXI4 master (outside of the AIE-ML array) and the DMA. Each buffer descriptor can be associated with locks. This is part of the configuration of any buffer descriptor using memory-mapped AXI4 interconnect.
The DMA controller has the support for the following features:

- Support 4D tensor address generation (including iteration-offset)

- Supports task queues and task complete tokens

- Supports S2MM finish on TLAST and out-of-order packets

- Adds decompression to the two S2MM channels

- Adds compression to the two MM2S channels

- Supports task queue and task-complete-tokens (see Task-Completion-Tokens for more information)

- **Lock Module:** The AIE-ML memory module contains a lock module to achieve synchronization amongst the AIE-MLs, tile DMA, and an external memory-mapped AXI4 interface master (for example, the processor system (PS)). The AIE-ML features 16 semaphore locks. The semaphore lock has a larger state and no acquired bit; each lock state is 6-bit unsigned. The lock module handles lock requests from the AIE-MLs in all four directions, the local DMA controller, and memory-mapped AXI4.

### **Data Movement**

This section describes examples of the data communications within the AIE-ML array and between the AIE-ML tile and the PL, using various combinations of shared memory, AXI4-Stream interconnect, and the AI Engine tile DMA.

_Chapter 2:_ AIE-ML Tile Architecture

##### **AIE-ML to AIE-ML Data Communication via Local** **Memory**

In the case where multiple kernels fit in a single AIE-ML, communications between two consecutive kernels can be established using a common buffer in the shared memory. For cases where the kernels are in separate but neighboring AIE-ML, the communication is through the shared memory module. The processing of data movement can be through a simple pipeline or can use ping and pong buffers (not shown in the figure) on separate memory banks to avoid access conflicts. The synchronization is done through locks. DMA and AXI4-Stream interconnect are not needed for this type of communication.

are a logical representation of the AIE-ML tiles and shared memory modules.

|Col1|2-dimensional Dataflow Communication Among Neighboring AIE-ML Tiles|Col3|

_Chapter 2:_ AIE-ML Tile Architecture

##### **AIE-ML Tile to AIE-ML Tile Data Communication via** **Memory and DMA**

The communication described in the previous section is inside an AIE-ML tile or between two neighboring AIE-ML tiles. For non-neighboring AIE-ML tiles, a similar communication can be established using the DMA in the memory module associated with each AIE-ML tile, as shown carried out by the locks in a similar manner to the AIE-ML to AIE-ML Data Communication via Local Memory section. In comparison to the neighboring tile communication, the main differences in this mode are increased communication latency and memory resources.

##### **AIE-ML Tile to AIE-ML Tile Data Communication via** **AXI4-Stream Interconnect**

AIE-MLs can directly communicate through the AXI4-Stream interconnect without any DMA and another through the streaming interface in a serial fashion, or the same information can be sent to an arbitrary number of AIE-ML tiles using a multicast communication approach. The streams can go in north/south and east/west directions. In all the streaming cases there are built-in hand-shake and backpressure mechanisms.

_**Note**_ **:** In a multicast communication approach, if one of the receivers is not ready, the whole broadcast stops until all receivers are ready again.

|Col1|Cascade Streaming|Col3|
||AIE-ML 0<br>AIE-ML 1<br>AIE-ML 2||

|Col1|Streaming Multicast|Col3|

_Chapter 2:_ AIE-ML Tile Architecture

##### **AIE-ML to PL Data Communication via Shared** **Memory**

In the generic case, the PL block consumes data via the stream interface. It then generates a data stream and forwards it to the array interface, where inside there is a FIFO that receives the PL stream and converts it into an AIE-ML stream. The AIE-ML stream is then routed to the AIE-ML destination function. Depending on whether the communication is block-based or stream-based, DMA, and ping-pong buffers could be involved.

the AIE-ML tile. The DMA moves the stream into a memory block neighboring the consuming AIE-ML. The first diagram represents the logical view and the second diagram represents the physical view.

|Col1|Streaming Communication|Col3|

|Interconnect|Interconnect|Interconnect|

|Col1|Streaming Communication|Col3|Col4|Col5|Col6|Col7|Col8|
||AIE<br>M<br>Interconnect|AIE<br>M<br>Interconnect|AIE<br>M<br>Interconnect|AIE<br>M<br>Interconnect|AIE<br>M<br>Interconnect|AIE<br>M<br>Interconnect||
||AIE<br>M<br>Interconnect|AIE|M|M|M|M|M|
||AIE<br>M<br>Interconnect|AIE|M|||||
||AIE<br>M<br>Interconnect|Interconnect|Interconnect|Interconnect|Interconnect|Interconnect|Interconnect|

_Chapter 2:_ AIE-ML Tile Architecture

##### **AIE-ML Data Movement with Memory Tiles**

The AIE-ML Memory tile is introduced in the AIE-ML architecture to significantly increase the on-chip memory inside the AIE-ML array. This new functional unit reduced the utilization of illustrates the general concept:

Depending on the characteristics of the ML applications and bandwidth requirement, the AIE-ML data movement architecture supports different dataflow mappings where either the activations of data alternative mappings supported by the AIE-ML architecture.

_Chapter 2:_ AIE-ML Tile Architecture

### **AIE-ML Debug**

Debugging the AIE-ML uses the memory-mapped AXI4 interface. All the major components in the AIE-ML array are memory mapped.

- Program memories

- Data memories

- AIE-ML registers

- DMA registers

- Lock module registers

- Stream switch registers

- AIE-ML break points registers

- Events and performance counters registers

These memory-mapped registers can be read and/or written from any master that can produce memory-mapped AXI4 interface requests (PS, PL, and PMC). These requests come through the shows a typical debugging setup involving a software development environment running on a host development system combined with its integrated debugger.

The debugger connects to the platform management controller (PMC) on an AIE-ML enabled Versal device either using a JTAG connection or the AMD high-speed debug port (HSDP) connection.

### **AIE-ML Trace and Profiling**

The AIE-ML tile has provisions for trace and profiling. It also has configuration registers that control the trace and profiling hardware.

##### **Trace**

There are two trace streams coming out of each AIE-ML tile. One stream from the AIE-ML and the other from the memory module. Both these streams are connected to the tile stream switch. There is a trace unit in each AIE-ML module and memory module in an AIE-ML tile, and an AIE-ML programmable logic (PL) module in an AIE-ML PL interface tile (see types of array interface tiles). The units can operate in the following modes:

- AIE-ML modes

- Event-time

- Event-PC

- Execution-trace

- AIE-ML memory module mode

- Event-time

- AIE-ML PL module mode

- Event-time

The trace is output from the unit through the AXI4-Stream as an AIE-ML packet-switched stream packet. The packet size is 8x32 bits, including one word of header and seven words of data. The information contained in the packet header is used by the array AXI4-Stream switches to route the packet to any AIE-ML destination it can be routed to, including AIE-ML local data memory through the AIE-ML tile DMA, external DDR memory through the AIE-ML array interface DMA, and block RAM or UltraRAM through the AIE-ML to PL AXI4-Stream.

The event-time mode tracks up to eight independent numbered events on a per-cycle basis. A trace frame is created to record state changes in the tracked events. The frames are collected in an output buffer into an AIE-ML packet-switched stream packet. Multiple frames can be packed into one 32-bit stream word but they cannot cross a 32-bit boundary (filler frames are used for 32-bit alignment).

In the event-PC mode, a trace frame is created each cycle where any one or more of the eight watched events are asserted. The trace frame records the current program counter (PC) value of the AIE-ML together with the current value of the eight watched events. The frames are collected in an output buffer into an AIE-ML packet-switched stream packet.

The trace unit in the AIE-ML can operate in execution-trace mode. In real time, the unit will send, via the AXI4-Stream, a minimum set of information to allow an offline debugger to reconstruct the program execution flow. This assumes the offline debugger has access to the ELF. The information includes:

- Conditional and unconditional direct branches

- All indirect branches

- Zero-overhead-loop LC

The AIE-ML generates the packet-based execution trace, which can be sent over the 32-bit wide AIE-ML tile. The two trace streams out of the tile are connected internally to the event logic, configuration registers, broadcast events, and trace buffers.

_**Note**_ **:** The different operating modes between the two modules are not shown.

To control the trace stream for an event trace, there is a 32-bit trace_control0/1 register to start and stop the trace. There are also the trace_event0/1 registers to program the internal event number to be added to the trace.

##### **Profiling (Performance Counters)**

The AIE-ML array has performance counters that can be used for profiling. The AIE-ML has four performance counters that can be configured to count any of the internal events. It will either count the occurrence of the events or the number of clock cycles between two defined events.
The memory module and the PL modules in the PL and NoC array interface tiles each have two shows a high-level logical view of the profiling hardware in the AIE-ML tile.

_Chapter 2:_ AIE-ML Tile Architecture

### **AIE-ML Events**

The AIE-ML and memory modules each have an event logic unit. Each unit has a defined set of local events. The following diagram shows the high-level logical view of events in the AIE-ML tile.
The event logic needs to be configured with a set of action registers that can be programmed over the memory-mapped AXI4 interface. There are separate sets of registers associated with event logic for the AIE-ML and memory modules. Event actions can be configured to perform a task whenever a specific event occurs. Also, there is separate broadcast logic to send event signals to neighboring modules.

North Events

An event itself does not have an action, but events can be used to create an action. Event broadcast and event trace can be configured to monitor the event. Examples of event actions include:

- Enable, disable, or reset of an AIE-ML

- Debug-halt, resume, or single-step of an AIE-ML

- Error halt of an AIE-ML

- Resynchronize timer

- Start and stop performance counters

- Start and stop trace streams

- Generate broadcast events

- Drive combo events

- ECC scrubbing event

For each of these event actions there are associated registers where a 7-bit event number is set and is used to configure the action to trigger on a given event.

**Event Broadcast**

Broadcast events are both the events and the event actions because they are triggered when a inside the AIE-ML tile. The units in the broadcast logic in the AIE-ML and memory modules receive input from and send out signals in all four directions. The broadcast logic is connected to the event logic, which generates all the events. There are configuration registers to select the event sent over, and mask registers to block any event from going out of the AIE-ML tile.

Each module has an internal register that determines the broadcast event signal to broadcast in the other directions. To avoid broadcast loops, the incoming event signals are ORed with the internal events to drive the outgoing event signals according the following list:

- Internal, east, north, south → west

- Internal, west, north, south → east

- Internal, south → north

- Internal, north → south

**TIP:** _The AIE-ML module east broadcast event interface is internally connected to the memory module_ _west broadcast event interface and does not go out of the AIE-ML tile. In the AIE-ML module, there are_ _16 broadcast events each in the north, south, and west directions. In the memory module, there are 16_ _broadcast events each in the north, south, and east directions._