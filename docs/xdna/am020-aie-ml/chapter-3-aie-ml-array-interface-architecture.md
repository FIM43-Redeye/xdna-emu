_Chapter 3:_ AIE-ML Array Interface Architecture

#### _Chapter 3_

## AIE-ML Array Interface Architecture

interface provides the necessary functionality to interface with the rest of the device. The AIE-ML array interface has three types of AIE-ML interface tiles. There is a one-to-one correspondence of interface tiles for every column of the AIE-ML array. The interface tiles form a row and move memory-mapped AXI4 and AXI4-Stream data horizontally (left and right) and also vertically up a AIE-ML tile column. The AIE-ML interface tiles are based on a modular internal hierarchy of the AIE-ML array interface in the AIE-ML array.

The types of array interface tiles and the modules within them are described in this section.

- AIE-ML PL interface tile

- PL module includes:

- AXI4-Stream switch

- Memory-mapped AXI4 switch

- AIE-ML to PL stream interface

- Control, debug, and trace unit

- AIE-ML configuration interface tile (exactly one instance per AIE-ML array)

- PLL for AIE-ML clock generation

- Power-on-reset (POR) unit

- Interrupt generation unit

- Dynamic function exchange (DFx) logic

- NoC peripheral interconnect (NPI) unit

- AIE-ML array global registers that control global features such as PLL/clock control, secure/non-secure behavior, interrupt controllers, global reset control, and DFx logic

- AIE-ML NoC interface tile

- PL module (see previous description)

- NoC module with interfaces to NMU and NSU includes:

- Bi-directional NoC streaming interface

- Array interface DMA

### **AIE-ML Array Interface**

The AIE-ML array interface consists of PL and NoC interface tiles. There is also one configuration AIE-ML array uses to communicate with other blocks in the Versal architecture. Also specified are the number of streams in the AXI4-Stream interconnect interfacing with the PL, NoC, or AIE-ML tiles, and between the AXI4-Stream switches.

**TIP:** _The exact number of PL and NoC interface tiles is device specific. The Versal Architecture and Product_ _[Data Sheet: Overview (DS950) lists the size of the AIE-ML array.](https://docs.amd.com/go/en-US/ds950-versal-overview)_

|PL Interface Tile<br>PL Clock|PL Interface Tile<br>PL Clock|PL Interface Tile<br>PL Clock|PL Interface Tile<br>PL Clock|PL Interface Tile<br>PL Clock|

_**Note**_ **:** The AIE-ML FMAX is 1 GHz for the -1L speed grade devices. The PL clock should be set at half that speed to 500 MHz. There is also a clock domain crossing at the NoC interface tile between the clocks for the AIE-ML and the NoC.

_**Note**_ **:** In both the NMU and NSU, AXI4 and NoC streams are mutually exclusive. Each of the two interfaces has a configuration register that selects between AXI4 and NoC stream. When NoC streams are enabled, they can be connected to up to four AXI4-Streams to/from the AIE4-Stream Switch. The outgoing NoC stream to the NMU is time-multiplexed across the incoming streams. Traffic from each stream is tagged with a separate TDEST set via configuration registers. Similarly, the incoming NoC stream from the NSU is demultiplexed across up to four streams to the AIE4-Stream Switch based on TDEST.

The types of interfaces to the PL and NoC are:

- Memory-mapped AXI4 interface: the communication channel is from the NSU to the AIE-ML as a slave

- AXI4-Stream interconnect has four types of interfaces:

- Connection to stream switches in other tiles

- Bi-directional connection to the PL streaming interface

- Connection to the array interface DMA that generates traffic into the NoC using a memory-mapped AXI4 interface

- Direct connection to the NoC streaming interfaces (NSU and NMU)

The AIE-ML array interface tiles manage the two high performance interfaces:

- AIE-ML to PL

- AIE-ML to NoC

The following tables summarize the bandwidth performance of the AIE-ML array interface with the PL, the NoC, and the AIE-ML tile. The bandwidth performances are specified per each AIEML column for the -1L speed grade devices. There is a reduction in the number of connections per column between the PL to AIE-ML interface and the AXI4-Stream switch to the AIE-ML tile.
This is to support the horizontally connected stream switches that provide additional horizontal routing capability. The total bandwidth for the various devices across speed grades can be found in the _Versal AI Core Series Data Sheet: DC and AC Switching Characteristics_ [(DS957) or](https://docs.amd.com/go/en-US/ds957-versal-ai-core) _Versal AI_ _Edge Series Data Sheet: DC and AC Switching Characteristics_ [(DS958).](https://docs.amd.com/go/en-US/ds958-versal-ai-edge)

|Table 4: AIE-ML Array Interface to PL Interface Bandwidth Performance|Col2|Col3|Col4|Col5|Col6|
|**Connection Type**|**Number of**<br>**Connections**|**Data**<br>**Width**<br>**(bits)**|**Clock**<br>**Domain**|**Bandwidth per**<br>**Connection**<br>**(GB/s)**|**Aggregate**<br>**Bandwidth**<br>**(GB/s)**|
|PL to AIE-ML array interface|8|641|PL<br>(500 MHz)|4|32|
|AIE-ML array interface to PL|6|64|PL<br>(500 MHz)|4|24|
|AIE-ML array interface to AXI4-<br>Stream switch|8|32|AIE-ML<br>(1 GHz)|4|32|
|AXI4-Stream switch to AIE-ML array<br>interface|6|32|AIE-ML<br>(1 GHz)|4|24|
|Horizontal interface between AXI4-<br>Stream switches2|4|32|AIE-ML<br>(1 GHz)|4|16|

|Table 5: AIE-ML Array Interface to NoC AXI4-Stream Interface Bandwidth<br>Performance|Col2|Col3|Col4|Col5|Col6|
|AIE-ML to NoC (NoC side)|1|128|NoC Interface<br>(960 MHz)1|16|16|
|AIE-ML to NoC (AIE-ML side)|4|32|AIE-ML<br>(1 GHz)|4|16|
|NoC to AIE-ML (NoC side)|1|128|NoC Interface<br>(960 MHz)1|16|16|
|NoC to AIE-ML (AIE-ML side)|4|32|AIE-ML<br>(1 GHz)|4|16|
|**Notes:**<br>1.<br>The frequency is based off of a –1 speed grade.|**Notes:**<br>1.<br>The frequency is based off of a –1 speed grade.|**Notes:**<br>1.<br>The frequency is based off of a –1 speed grade.|**Notes:**<br>1.<br>The frequency is based off of a –1 speed grade.|**Notes:**<br>1.<br>The frequency is based off of a –1 speed grade.|**Notes:**<br>1.<br>The frequency is based off of a –1 speed grade.|

|Table 6: AIE-ML Array Interface to AIE-ML Tile Bandwidth Performance|Col2|Col3|Col4|Col5|Col6|
|**Connection Type**|**Number of**<br>**Connections**|**Data**<br>**Width**<br>**(bits)**|**Clock**<br>**Domain**|**Bandwidth per**<br>**Connection**<br>**(GB/s)**|**Aggregate**<br>**Bandwidth**<br>**(GB/s)**|
|AXI4-Stream switch to AIE-ML tile|6|32|AIE-ML<br>(1 GHz)|4|24|
|AIE-ML tile to AXI4-Stream switch|4|32|AIE-ML<br>(1 GHz)|4|16|

The following sections contain additional AIE-ML array interface descriptions. The AIE-ML tiles are described in Chapter 2: AIE-ML Tile Architecture.

### **Features of the AIE-ML Array Interface**

- **Memory Mapped AXI4 Interconnect:** Provides functionality to transfer the incoming memorymapped AXI4 requests from the NoC to inside the AIE-ML array.

- **AXI4 Master: Interface-DMA:** Memory mapped access to the rest of the device via the NoC, including external memory.

- **AXI4-Stream Interconnect:** Leverages the AIE-ML tile streaming interconnect functionality.

- **AIE-ML to PL Interface:** The AIE-ML PL modules directly communicate with the PL.
Asynchronous FIFOs are provided to handle clock domain crossing.

- **AIE-ML to NoC Interface:** The AIE-ML to NoC module handles the conversion of 128-bit NoC streams into 32-bit AIE-ML streams (and vice versa). It provides the interface logic to the NoC components (NMU and NSU). Level shifting is performed because the NMU and NSU are in a different power domain from the AIE-ML.

- **Hardware Locks:** Leverages the corresponding unit in the AIE-ML tile and is accessible from the AIE-ML array interface or an external memory-mapped AXI4 master, the module is used to synchronize the array interface to DMA transfer to/from external memory. The lock module has 16 semaphore locks and the lock state is 6-bit unsigned.

- **Debug, Trace, and Profile:** Leverages all the features from the AIE-ML tile for local event debugging, tracing, and profiling.

### **Array Interface Memory-Mapped AXI4 Slave** **Interconnect**

The main task of the AIE-ML memory-mapped AXI4 interconnect is to allow external access to internal AIE-ML tile resources such as memories and registers for configuration and debug.
It is not designed to carry the bulk of the data movement to and from the AIE-ML array.
The memory-mapped AXI4 interfaces are all interconnected across the AIE-ML array interface row. This enables the memory-mapped AXI4 interconnects in the array interface tiles to move incoming memory-mapped signals to the correct column horizontally and then forward them vertically to the memory-mapped AXI4 interconnect in the bottom AIE-ML tile of that column through a switch.

Each memory-mapped AXI4 interface is a 32-bit address with 32-bit data. The maximum memory-mapped AXI4 bandwidth is designed to be 1.5 GB/s. The memory-mapped AXI4 interface supports 1 MB address space per tile.

To feed the memory-mapped AXI4 interface, the NoC module contains a memory-mapped AXI4 bridge that accepts memory-mapped AXI4 transfers from the NoC NSU interface, and acts as a memory-mapped AXI4 master to the internal memory-mapped AXI4 interface switch.

_Chapter 3:_ AIE-ML Array Interface Architecture

### **Array Interface DMA Memory-Mapped AXI4** **Master Interface**

The AIE-ML array interface DMA provides direct access to external memory. The DMA is an AXI4 master, capable of issuing read and write requests to the NoC NMU interface, and hence to any AXI4 slave on the Versal device provided the NoC configuration provides the path. The DMA supports a 32-bit aligned start address. Each DMA channel generates addresses based on the base address in the buffer descriptor that stores the incremental address offset between BD calls and avoids the need to reconfigure a BD for subsequent buffer transfers.

The DMA is composed of four independent channels, two MM2S (read from external memory), and two S2MM (write to external memory). Each channel can sustain 4 bytes per cycle (4 Gb/s at 1 GHz) throughput, giving a total of up to 8 Gb/s read and 8 Gb/s write in parallel per interface tile.

MM2S Channels (two in total) :

- 32-bit stream master interface per channel

- 128-bit AXI4 master read interface, shared between two channels

- 4D tensor address generation (including iteration-offset)

- Access shared lock module (local to interface tile)

- Support task queue and task-complete-tokens; queue depth is four tasks per channel (see

Task-Completion-Tokens for more information)

S2MM Channels (two in total)

- 32-bit stream slave interface per channel

- 128-bit AXI4-MM master write interface, shared between two channels

- 4D tensor address generation (including iteration-offset)

- Access shared lock module (local to interface tile)

- Support task queue and task-complete-tokens; queue depth is four tasks per channel (see

Task-Completion-Tokens for more information)

- Support out-of-order packet transfer, finish-on-TLAST enabling compressed spill and restore of intermediate results to external memory

Buffer descriptors (BD):

- 16 shared BDs

The interface DMA, together with tile and memory tile DMAs, and the streaming interconnect supports the following data-flows (non-exhaustive list).

- Buffer copy from external-memory to memory tile

- Buffer copy from external-memory to AIE-ML tile data memory

- Buffer copy from memory tile to external-memory

- Buffer copy from AIE-ML tile data memory to external-memory

### **Array Interface AXI4-Stream Interconnect**

The main task of the AIE-ML AXI4-Stream switch is to carry deterministic throughput and high-speed circuit or packet data-flow between AIE-MLs and the programmable logic or NoC.
Therefore, it is designed to carry the bulk of the data movement to/from the AIE-ML array. The AXI4-Stream switches in the bottom row of AIE-ML tiles interface directly to another row of AXI4-Stream interconnected switches in the AIE-ML array interface. The stream switch has one stream FIFO.

### **AIE-ML to Programmable Logic Interface**

AXI4-Stream switches in the AIE-ML to PL tiles can directly communicate with the programmable logic using the AXI4-Stream interface. There are six streams from AIE-ML to PL and eight streams from PL to each AIE-ML column. From a bandwidth perspective, each AXI4-Stream interface can support the following.

- 24 GB/s from each AIE-ML column to PL

- 32 GB/s from PL to each AIE-ML column

Each stream has a 64-bit interface but can be configured to operate as a 32-bit or 64-bit stream, or two physical streams can be configured to operate as a single 128-bit stream. Streams from the PL support a subset of the AXI4-Stream protocol:

- When TLAST = 0, TKEEP is ignored and assumed to be all ones

- When TLAST = 1, TKEEP is only supported at 32-bit granularity and valid words must be contiguous:

- 32-bit streams: TKEEP must be 0xF

- 64-bit streams: TKEEP can be 0x0F or 0xFF

- 128-bit streams: TKEEP can be 0x000F, 0x00FF, 0x0FFF, 0xFFFF

In the VC2802 device, there are 38 columns of AIE-ML tiles, AIE-ML memory tiles, and AIE-ML interface tiles. However, only 28 array interface tiles are available to the PL interface. Therefore, the aggregate bandwidth for PL interface is approximately:

- 670 GB/s from AI Engine to PL

- 900 GB/s from PL to AI Engine

All bandwidth calculations assume a nominal 1 GHz AI Engine clock for the -1L speed grade devices at VCCINT = 0.70V. The number of array interface tiles available to the PL interface and total bandwidth of the AI Engine to PL interface for other devices and across different speed grades is specified in _Versal AI Core Series Data Sheet: DC and AC Switching Characteristics_ [(DS957).](https://docs.amd.com/go/en-US/ds957-versal-ai-core)

### **AIE-ML to NoC Interface AXI-Stream**

The AIE-ML to NoC interface tile, in addition to the AXI4-Stream interface capability, also contains paths to connect to the horizontal NoC (HNoC). Looking from the AIE-ML, there are four streams from the AIE-ML to the NoC, and four streams from the NoC to the AIE-ML. From a bandwidth perspective each AIE-ML to NoC interface tile can direct traffic between the HNoC and the AXI4-Stream switch.

**TIP:** _The actual total bandwidth can be limited by the number of horizontal and vertical channels available_ _in the device and also the bandwidth limitation of the NoC._

### **Interrupt Handling**

It is possible to setup interrupts to the processor system (PS) and the platform management controller (PMC) triggered by events inside the AIE-ML array. This section gives an introduction to the types of interrupts from the AIE-ML array.

The AIE-ML array generates four interrupts that can be routed from the AIE-ML array to the PMC, application processing unit (APU), and real-time processing unit (RPU). The overall hierarchy for interrupt generation from AIE-ML array is as follows:

- Events get triggered from any of the AIE-ML tiles or AIE-ML interface tiles.

- Each column has first-level interrupt handlers that can capture the trigger/event generated and forward it to the second-level interrupt handler. Second-level interrupt handlers are only available in NoC interface tiles.

- A second-level interrupt handler can drive any one of the four interrupt lines in a AIE-ML array interface.

- These four interrupt lines are eventually connected to the AIE-ML configuration interface tile.

from the AIE-ML array to other blocks in the Versal device. The diagram does not show the actual layout/placement of the array interface tiles and the AIE-ML tiles.

In the previous figure, the four interrupts are generated from a NoC interface tile. They pass through the PL interface tile and reach a configuration interface tile. Internal errors (such as PLL lock loss) are then ORed with the four incoming interrupts and the resulting four interrupts are connected directly to the NPI interrupt signals on the NPI interface, which is a 32-bit wide memory-mapped AXI4 bus.

At the device level, the four NPI interrupts are assigned 4 to 7. There are three groups of NPI registers (IMR0…IMR3, IER0…IER3, and IDR0…IDR3). Each of the pairs (IMR, IER, and IDR) can be used to configure the four NPI interrupts. IMR registers are read only, and IER and IDR registers are write only. Only the registers corresponding to NPI interrupt 4 can be programmed.
For NPI interrupts 5, 6, and 7, the three sets of registers have no effect and the three interrupts cannot be masked by programming the NPI register.