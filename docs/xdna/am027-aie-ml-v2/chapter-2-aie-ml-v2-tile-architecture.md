_Chapter 2:_ AIE-ML v2 Tile Architecture

#### _Chapter 2_

## AIE-ML v2 Tile Architecture

The top-level block diagram of the AIE-ML v2 tile architecture, key building blocks, and

Integrated synchronization primitives (locks)

The AIE-ML v2 tile consists of the following high-level modules:

- Tile interconnect

- AIE-ML v2

- AIE-ML v2 memory module

The tile interconnect module manages incoming and outgoing AXI4-Stream and memory-mapped AXI4 traffic. The subsequent sections elaborate on the functioning of the memory-mapped AXI4 and AXI4-Stream interconnect. Within the AIE-ML v2 memory module, there are eight memory banks totaling 64 KB of data memory. This module includes a memory interface, DMA functionality, and locks. Both incoming and outgoing directions are equipped with DMA, while each memory module contains a locks block. The AIE-ML v2 tile can access memory modules in

all four directions as a single contiguous block of memory. The memory interface appropriately directs memory accesses based on the address generated from the AIE-ML v2. Featuring a scalar datapath, a vector datapath, three address generators, and 16 KB of program memory, the AIE-ML v2 facilitates efficient processing. Upon boot and reset, the program and data memory of the AIE-ML v2 tile initialize to zero. Additionally, the AIE-ML v2 includes a cascade stream access mechanism, allowing the forwarding of accumulator output to the subsequent AIE-ML v2 tile.

The AIE-ML v2 is described in more detail in Chapter 4: AIE-ML v2 Architecture. Both the AIE-ML v2 tile and the AIE-ML v2 memory module have control, debug, and trace units. Some of these units are described later in this document:

- Control and status registers

- Events, event broadcast, and event actions

- Performance counters for profiling and timers

Figure 4 shows the connectivity across AIE-ML v2 tiles within the AIE-ML v2 array. This structure enables neighboring AIE-ML v2 to exchange data via shared data memory.

The architecture is designed to enable each AIE-ML v2 unit to interface with up to four distinct memory modules. These modules encompass:

- Its own local memory module

- The module positioned to the north

- The module situated to the south

- The module placed towards the west

However, the AIE-ML v2 units located at the edges of the array might have access to a slightly limited memory configuration compared to those centrally positioned within the array. This difference results in edge units having access to one or two fewer memory modules than their counterparts positioned more centrally. This distinction arises due to the bordering placement of these units within the array's structure.

Together with the flexible and dedicated interconnects, the AIE-ML v2 array provides deterministic performance, low latency, and high bandwidth. The modular and scalable architecture allows more compute power as more tiles are added to the array.

The AIE-ML v2 has both horizontal and vertical cascade connections, directed from north to south and from west to east. The cascade start points and end points are tied off at the array edges.

### **Memory-mapped AXI4 Interconnect**

Each AIE-ML v2 engine tile contains a memory-mapped AXI4 interconnect for use by external blocks to write to or read from any of the registers or memories in the AIE-ML v2 tile. The memory-mapped AXI4 interconnect inside the AIE-ML v2 array can be driven from outside of the array by any memory-mapped AXI4 manager that can connect to the NoC. All internal resources in an AIE-ML v2 tile including memory, and all registers in an AIE-ML v2 tile and AIE-ML v2 memory module, are mapped onto a memory-mapped AXI4 interface.

Each AIE-ML v2 tile has a memory-mapped AXI4 switch that accepts all memory-mapped AXI4 requests from the south direction. If the address is for the tile, the request is served locally.
Otherwise, the request is passed to the next tile in the north direction. Each tile has 1 MB address space.

The following table shows the addressing scheme of memory-mapped AXI4 in the AIE-ML v2 tile. The lower 20 bits represent the tile address range of 0x00000 to 0xFFFFF, followed by five bits that represent the row location and seven bits that represent the column location.

|Table 3: AIE-ML v2 Memory-Mapped AXI4 Tile Addresses|Col2|Col3|
|**Column**|**Row [22:18]**|**AIE-ML v2 Tile/AIE-ML v2 Memory Tile/AIE-ML v2 Array**<br>**Interface-Tile**|
|[31:25] (7b)|[24:20] (5b)|[19:0] (20b)|

### **AXI4-Stream Interconnect**

Each AIE-ML v2 tile has an AXI4-Stream interconnect (alternatively called a stream switch) that is a fully programmable, a 64-bit AXI4-Stream crossbar, and is statically configured through the memory-mapped AXI4 interconnect. It handles backpressure and is capable of the full bandwidth on the AXI4-Stream. The AXI4-Stream interconnect has the following properties:

- All elements of the switch data-path are 64-bit, which supports the transport of two 32-bitwords in parallel per cycle.

- A TKEEP signal is added to support transfer of an odd number of 32-bit-words.

- Stream parity is calculated on the 32-bit word and two parity bits transported with the data.

- All external south, west, north, east, and stream-FIFO ports are capable of full throughput of 64-bit per cycle.

- Sources and destinations of DMA which can run at full throughput of 64-bit per cycle.

- Core, trace, and control-packet blocks have a 32-bit stream interface, so an adapter is placed for 32 <-> 64-bit conversion.

The tile ports are split into external ports and local ports. The external ports are south, west, north, and east. The local ports are core, DMA, FIFO, and trace. In the following description, a double-word refers to two adjacent 32-bit words on the stream, and in the case of TLAST=1 and TKEEP=0, a double-word refers to 32-bit word and the adjacent null-word.

- External and local slave ports have two cycles of latency and four double-words of buffering.

- Local master ports have one cycle of latency and two double-words of buffering.

Therefore, excluding packet switch arbitration overhead, the latency and buffering crossing the switch is external to external.

- Local slave to local master: three cycles of latency, six double-words of buffering

- Local slave to external master: four cycles of latency, eight double-words of buffering

- External slave to local master: three cycles of latency, six double-words of buffering

- External to external: four cycles of latency, eight double-words of buffering

The stream switch contains one 16-deep, 68-bit (64-bit data + 2-bit parity + 1-bit TLAST + 1-bit TKEEP) wide FIFO. Each stream port operates in either circuit-switched or packet-switched mode, selected by a packet-switching bit in the port’s configuration register. A port cannot use both modes at the same time. In packet-switched mode, multiple logical streams can share the same physical port and wires.

|Table 4: AIE-ML v2 AXI4-Stream Tile Interconnect Bandwidth|Col2|Col3|Col4|Col5|Col6|
|**Connection**<br>**Type**|**Number of**<br>**Connections**|**Data Width**<br>**(Bits)**|**Clock Domain**<br>**(1 GHz)**|**Bandwidth per**<br>**Connection**<br>**(GB/s)**|**Aggregate**<br>**Bandwidth**<br>**(GB/s)**|
|To North/From<br>South|6|64|AIE-ML v2 array<br>clock|8|48|
|To South/From<br>North|4|64|AIE-ML v2 array<br>clock|8|32|
|To West/From East|4|64|AIE-ML v2 array<br>clock|8|32|
|To East/From West|4|64|AIE-ML v2 array<br>clock|8|32|

_**Note**_ **:** The previous table assumes the slowest speed grade. Higher bandwidth can be achieve with a higher speed grade.

A circuit-switched stream has a one-to-one or one-to-many topology: a single source port feeds one or more destination ports, and all data entering the source is broadcast to all configured destinations. Latency is deterministic. When bandwidth is constrained, built-in backpressure throttles the source, which reduces overall performance.

Packet-switched streams can share ports with other packet-switched streams, which introduces potential resource contention and therefore non-deterministic latency. Each packet-switched stream uses a 5-bit stream ID that must be unique among streams sharing the same ports; this ID determines the packet’s destination(s). Packet switching supports flexible topologies, enabling any combination of single or multiple manager and subordinate ports within a stream.

A packet-switched packet has:

- **Packet header:** Routing and control information for the packet

- **Data:** Actual data in the packet

|• TLAST: Last word in the packet must have TLAST asserted to mark the end of packet|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|_Table 5:_**Packet Header**|_Table 5:_**Packet Header**|_Table 5:_**Packet Header**|_Table 5:_**Packet Header**|_Table 5:_**Packet Header**|_Table 5:_**Packet Header**|_Table 5:_**Packet Header**|_Table 5:_**Packet Header**|
|**Odd**<br>**Parity**|**3'b000**|**Source Column**|**Source Row**|**1'b0**|**Packet**<br>**Type**|**7'b0000000**|**Stream ID**|

### **AIE-ML v2 Tile Program Memory**

The AIE-ML v2 tile has a local 16 KB of program memory with a data-width of 128-bit that can be used to store VLIW instructions. There are two interfaces to the program memory:

- Memory-mapped AXI4 interface

- AIE-ML v2 interface

An external manager can read or write to the program memory using the memory-mapped AXI4 interface. The AIE-ML v2 tile has 128-bit wide interfaces to the program memory to fetch instructions. The AIE-ML v2 tile can read from, but not write to, the program memory.
To access the program memory simultaneously from the memory-mapped AXI4 and AIE-ML v2 tile, divide the memory into multiple banks and access mutually exclusive parts of the program memory. Arbitration logic is needed to avoid conflicts between accesses and to assign priority when accesses are to the same bank. The core program memory load gets priority and the other requests get a SLVERR if a concurrent access is made to the program memory.

### **AIE-ML v2 Interfaces**

The AIE-ML v2 has multiple interfaces. The following block diagram shows the interfaces.

- **Data Memory Interface:** The AIE-ML v2 core can access data memory modules on all four directions. They are accessed as one contiguous memory. The AIE-ML v2 core has two 512-bit wide load units and one 512-bit wide store unit. From the AIE-ML v2 perspective, the throughput of each of the loads (two) and store (one) is 512 bits per clock cycle.

- **Program Memory Interface:** This 128-bit wide interface is used by the AIE-ML v2 core to access the program memory. A new instruction can be fetched every clock cycle.

- **Direct AXI4-Stream Interface:** The AIE-ML v2 core has one 32-bit input AXI4-Stream interface and one 32-bit output AXI4-Stream interface. Each stream allows the AIE-ML v2 core to have a four-word (128-bit) access every four cycles, or a one-word (32-bit) access every cycle.

- **Cascade Stream Interface:** The 512-bit accumulator data from one AI Engine can be forwarded to another by using these cascade streams to form a chain. Using this stream, the core can transfer accumulator data to the next core. There is a small FIFO which accumulates up to two 512-bit wide words on both input and output streams. This allows the storing of four accumulator words in FIFO between each AIE-ML v2 core.

- **Debug Interface:** This interface is able to read or write all AIE-ML v2 core registers over the memory-mapped AXI4 interface.

- **Hardware Synchronization (Locks) Interface:** This interface allows synchronization between two AIE-ML v2s or between an AIE-ML v2 and DMA. The AIE-ML v2 can access the lock modules in all four directions. There is also added support for semaphore locks.

- **Stall Handling:** An AIE-ML v2 can be stalled due to multiple reasons and from different sources. Examples include external memory-mapped AXI4 manager (for example, PS), lock modules, empty or full AXI4-Stream interfaces, data memory collisions, and event actions from the event unit.

- **AIE-ML Event Interface:** This 16-bit wide EVENT interface can be used to set different events.
The tile provides 128 internal core module and 128 memory module events. Each event has a unique number.

- **Tile Timer:** The input interface to read the 64-bit timer value inside the tile.

- **Execution Trace Interface:** A 32-bit wide interface where the AIE-ML v2 generated packetbased execution trace can be sent over the AXI4-Stream.

### **AIE-ML v2 Memory Module**

two input streams to memory map (S2MM) DMA, two memory-map to output DMA streams (MM2S), and a hardware synchronization module (locks).

- **Memory Banks:** The AIE-ML v2 data memory is 64 KB, organized as eight memory banks, where each memory bank is a 256 word x 256-bit single-port memory. From a programmer's perspective, a pair of hardware banks are interleaved at 256-bit granularity to present four programmer banks each being 512-bit wide. Banks 0 and 1 have ECC error correction. Banks 2–7 have parity protection.

- **Memory Arbitration:** Arbitration is performed independently for each bank with parallel accesses to different banks supported. Therefore, there are eight independent arbiters per tile DM. Arbitration is between all accessors with active requests. The arbitration is hierarchical, with accessors arranged into fixed groups.

- **Tile DMA Controller:** The tile DMA connects to the AIE-ML v2 stream switches with two incoming and two outgoing streams. It consists of two modules: S2MM, which writes 64-bit stream data to memory, and MM2S, which reads 64-bit data from memory to a stream. DMA transfers are defined by buffer descriptors; the controller supports 16 descriptors, accessible and configurable from any memory-mapped AXI4 manager. Each descriptor contains all

parameters for a transfer and can optionally chain to the next descriptor for continuous operation. The controller also manages 16 locks used for synchronization between the AIEML v2 array, the DMA, and any external memory-mapped AXI4 managers. Descriptors can be configured to use these locks through the same AXI4 interface. The DMA datapath sustains a throughput of two 32-bit words per cycle.

The DMA controller supports:

- 4D tensor address generation (including iteration-offset)

- Task queues and task complete tokens

- S2MM finish on TLAST and handling of out-of-order packets

- Decompression on the two S2MM channels

- Compression to the two MM2S channels

- **Lock Module:** The AIE-ML v2 memory module contains a lock module to achieve synchronization amongst the AIE-ML v2s, tile DMA, and an external memory-mapped AXI4 interface manager (for example, the processor system (PS)). The AIE-ML v2 features 16 semaphore locks, each carrying a 6-bit unsigned as state. The lock module handles lock requests from the AIE-ML v2s in all four directions, the local DMA controller, and memorymapped AXI4.

### **AIE-ML v2 Debug**

Debugging the AIE-ML v2 uses the memory-mapped AXI4 interface. All the major components in the AIE-ML v2 array are memory-mapped:

- Program memories

- Data memories

- AIE-ML v2 registers

- DMA registers

- Lock module registers

- Stream switch registers

- AIE-ML v2 break points registers

- Events and performance counters registers

These memory-mapped registers can be read and/or written from any manager that can produce memory-mapped AXI4 interface requests (PS, PL, and PMC). These requests come through the NoC to the AIE-ML v2 array interface, and then to the target tile in the array. The following figure shows a typical debugging setup involving a software development environment running on a host development system combined with its integrated debugger.

The debugger connects to the platform management controller (PMC) on an AIE-ML v2 enabled Versal device either using a JTAG connection or the AMD high-speed debug port (HSDP) connection.

### **AIE-ML v2 Trace and Profiling**

The AIE-ML v2 tile has provisions for trace and profiling. It also has configuration registers that control the trace and profiling hardware.

##### **Trace**

The AIE-ML v2 tiles have provisions for trace streams. There are two trace units in the tile, one for the core module and another for the memory-module. Both these trace streams are connected to the tile stream switch. There is a trace unit in an AIE-ML v2 array interface tile.
Each trace unit can collect trace data and send it out via the AXI4-Stream network.

The units can operate in the following modes:

- **AIE-ML v2 Tile Core Module Modes:**

- Event-time

- Event-PC

- **AIE-ML v2 Tile Memory Module Modes:** Event-time

- **AIE-ML v2 Memory Tile Module Mode:** Event-time

- **AIE-ML v2 Array Interface Tile Mode:** Event-time

The trace is output from the unit through the AXI4-Stream as an AIE-ML v2 packet-switched stream packet. The packet size is 8x32 bits, including one word of header and seven words of data. The stream ID in the packet header is set via a configuration register. The information contained in the packet header is used by the array AXI4-Stream switches to route the packet to any AIE-ML v2 destination it can be routed to, including AIE-ML v2 local data memory through the AIE-ML v2 tile DMA, external DDR memory through the AIE-ML v2 array interface DMA, and block RAM or UltraRAM through the AIE-ML v2 to PL AXI4-Stream.

The event-time mode tracks up to eight independent numbered events on a per-cycle basis. A trace frame is created to record state changes in the tracked events. The frames are collected in an output buffer into an AIE-ML v2 packet-switched stream packet. Multiple frames can be packed into one 32-bit stream word, but they cannot cross a 32-bit boundary (filler frames are used for 32-bit alignment).

In the event-PC mode, a trace frame is created each cycle where any one or more of the eight watched events are asserted. The trace frame records the current program counter (PC) value of the AIE-ML v2 together with the current value of the eight watched events. The frames are collected in an output buffer into an AIE-ML v2 packet-switched stream packet.

trace streams out of the tile are connected internally to the event logic, configuration registers, broadcast events, and trace buffers.

_**Note**_ **:** The different operating modes between the two modules are not shown.

To control the trace stream for an event trace, there is a 32-bit trace_control0/1 register to start and stop the trace. There are also the trace_event0/1 registers to program the internal event number to be added to the trace.

##### **Profiling (Performance Counters)**

The AIE-ML v2 array incorporates performance counters tailored for profiling purposes. Within the AIE-ML v2 and its associated memory module, each has four versatile performance counters available for configuration. These counters are capable of monitoring various internal events within the system. You can set them to either tally the occurrence of specific events or calculate the number of clock cycles between two defined events.

Moreover, the AIE-ML v2 memory tile and the AIE-ML v2 array interface are equipped with six performance counters, aligning their functionalities with those of the aforementioned counters.
These counters are also customizable to execute similar monitoring tasks.

of the profiling hardware integrated within the AIE-ML v2 tile. This hardware configuration is instrumental in facilitating performance monitoring and analysis within the AIE-ML v2 array.

### **AIE-ML v2 Events**

The AIE-ML v2 and memory modules each have an event logic unit. Each unit has a defined set of local events. The following diagram shows the high-level logical view of events in the AIE-ML v2 tile. The event logic needs to be configured with a set of action registers that can be programmed over the memory-mapped AXI4 interface. There are separate sets of registers associated with event logic for the AIE-ML v2 and memory modules. Event actions can be configured to perform a task whenever a specific event occurs. Also, there is separate broadcast logic to send event signals to neighboring modules.

North Events

An event itself does not have an action, but events can be used to create an action. Event broadcast and event trace can be configured to monitor the event. Examples of event actions include:

- Enable, disable, or reset of an AIE-ML v2

- Debug-halt, resume, or single-step of an AIE-ML v2

- Error halt of an AIE-ML v2

- Resynchronize timer

- Start and stop performance counters

- Start and stop trace streams

- Generate broadcast events

- Drive combo events

- ECC scrubbing event

For each of these event actions there are associated registers where a 7-bit event number is set and is used to configure the action to trigger on a given event.

**Event Broadcast**

Broadcast events are both the events and the event actions because they are triggered when a inside the AIE-ML v2 tile. The units in the broadcast logic in the AIE-ML v2 and memory modules receive input from and send out signals in all four directions. The broadcast logic is connected to the event logic, which generates all the events. There are configuration registers to select the event sent over, and mask registers to block any event from going out of the AIE-ML v2 tile.

Each module has an internal register that determines the broadcast event signal to broadcast in the other directions. To avoid broadcast loops, the incoming event signals are ORed with the internal events to drive the outgoing event signals according the following list:

- Internal, east, north, south → west

- Internal, west, north, south → east

- Internal, south → north

- Internal, north → south

**TIP:** _The AIE-ML v2 module east broadcast event interface is internally connected to the memory module_ _west broadcast event interface and does not go out of the AIE-ML v2 tile. In the AIE-ML v2 module, there_ _are 16 broadcast events each in the north, south, and west directions. In the memory module, there are 16_ _broadcast events each in the north, south, and east directions._