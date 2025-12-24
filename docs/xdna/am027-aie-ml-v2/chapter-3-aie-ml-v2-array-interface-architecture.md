_Chapter 3:_ AIE-ML v2 Array Interface Architecture

#### _Chapter 3_

## AIE-ML v2 Array Interface Architecture

interface provides the necessary functionality to interface with the rest of the device. There is a one-to-one correspondence of interface tiles for every column of the AIE-ML v2 array. The interface-tiles form a row and move memory-mapped AXI4 and AXI4-Stream data horizontally (left and right) and vertically up an AIE-ML v2 tile column. The AIE-ML v2 interface-tiles are based on a modular architecture, but the final composition is device specific. Refer to the following figure for the internal hierarchy of the AIE-ML v2 array interface in the AIE-ML v2 array.

_Chapter 3:_ AIE-ML v2 Array Interface Architecture

### **AIE-ML v2 Array Interface**

The AIE-ML v2 array interface tile has NPI, POR, PLL, stream switch, CDT, event switch, control module, NoC, PL interface, two NMU interfaces, and an NSU interface. The array interface tile also supports AXI_MM isolation, which means AXI-MM transactions coming from west/east into array interface tiles are blocked.

The functionality of each of the modules in the array interface tiles are described as follows:

- **Power on Reset (POR):** The POR module is powered by the NoC voltage rail. This provides the necessary hardware reset signal to the AIE-ML v2 array.

- **AXI-MM Switch:** The AXI-MM switch provides functionality to move AXI4 horizontally within the AIE-ML v2 array interface tiles. The traffic can be directed vertically to the columns. This can also be used as a local interface for configuring local registers and has 1 MB of address space. This is a 32-bit interface.

- Array interface tile west interface: requests to lower index columns

- Array interface tile east interface: requests to higher index columns

- Array interface tile north interface: requests to all other tiles in same column (AIE-ML v2 memory and AIE-ML v2 tile)

- **AXI4-Stream PL Interface:** This interface allows streams to and from the PL through the BLI interface. Each stream has a 64-bit interface but can be configured to operate as a 32-bit or 64-bit stream, or two physical streams can be configured to operate as a single 128-bit stream.Streams from the PL support a subset of the AXI4-Stream protocol:

- When TLAST = 0, TKEEP is ignored and assumed to be all ones

- When TLAST = 1, TKEEP is only supported at 32-bit granularity and valid words must be contiguous:

- 32-bit streams: TKEEP must be 0xF

- 64-bit streams: TKEEP can be 0x0F or 0xFF

- 128-bit streams: TKEEP can be 0x000F, 0x00FF, 0x0FFF, 0xFFFF

|Table 6: AIE-ML v2 Array Interface to PL Interface Bandwidth Performance|Col2|Col3|Col4|Col5|Col6|
|PL to AIE-ML v2 array interface|8|641|PL<br>(500 MHz)|4|32|
|AIE-ML v2 array interface to PL|6|64|PL<br>(500 MHz)|4|24|
|AIE-ML v2 array interface to AXI4-<br>Stream switch|8|64|AIE-ML v2<br>(1 GHz)|8|64|
|AXI4-Stream switch to AIE-ML v2<br>array interface|6|64|AIE-ML v2<br>(1 GHz)|8|48|
|Horizontal interface between AXI4-<br>Stream switches2|4|64|AIE-ML v2<br>(1 GHz)|8|32|

_**Note**_ **:** The previous table assumes the slowest speed grade. Higher bandwidth can be achieve with a higher speed grade.

- **AXI4 Manager Interface (NMU):** This interface is used to carry AXI4 traffic from array interface tile DMA and control-interface. Array interface tile supports up to two of these interfaces and support passing traffic through neighbor NMU. This is a 128-bit interface and supports 48-bit addresses.

- **AXI4 Subordinate Interface (NSU):** This interface is used to carry AXI4 configuration, control, status and debug traffic from external controller and host debugger to the array interface tiles.

- **Event Broadcast Switch:** Broadcast events are both events and event actions because they are triggered when a configured event is asserted. The switch carries events to/from PL via BLI interface.

- **NPI and Interrupt:** The AIE-ML v2 array provides a single NPI subordinate (end-point)
interface in the AIE-ML v2 array interface tile, which provides access to the AIE-ML v2 PLL configuration registers, global reset registers for AIE-ML v2 array and global AIE-ML v2 array register settings (security setting). Four of the interrupts are driven from the array interface tiles on to NPI Interrupts lines to PMC.

AIE-ML v2 array interface tile contains hardware activity monitors that can be leveraged to provide telemetry data to allow power management schemes. This activity monitor shall capture the count of vector instructions executed by all the AIE-ML v2 tiles over a time period. The counter is of 40-bit width and can be read via two 32-bit NPI registers.

- **PLL:** A single, low-jitter dedicated PLL for the AIE-ML v2 clock generation. Depending on the device some devices require a PLL internal to the AIE-ML v2 array.

- **Stream Switch:** The main task of the AXI4-Stream switch is to carry deterministic throughput and high-speed circuit or packet data-flow between AIE-ML v2 and the programmable logic or NoC. It is designed to carry the bulk of the data movement to/from the AIE-ML v2 array.
The switch data path has a 64-bit width and supports two 32-bit words in parallel per cycle.
AXI4-Stream allows for data-flow to/from east, west, and north directions. It uses the PL or the DMA interface to/from south directions. All these paths have 64-bit data width. The switch also allows flow from/to the control module, trace, and control response. Each of these have a data width of 32-bit.

- **Control Module:** The control module performs application set-up, orchestration of execution and data movement during runtime, and synchronization with external host. This control module block has an AXI4 manager port connecting to the NoC, or read/write configuration in the array. An AXI4 subordinate port allows read/write to registers. This also includes an AXI4-Stream interface to handle task-complete-tokens and issue control-packets.

- **Control, Debug, and Trace Unit:** This unit leverages the features from the AIE-ML v2 tile for local event debugging, tracing, and profiling. The number of performance counters is six. The number of events has been increased to 256 to accommodate the additional events generated by control module. An input is added to the second level interrupt handler, driven by the control module.

- **NoC Interface:** This module handles the interface to and from the NoC in AIE-ML v2. The NoC module has the following modules:

- DMA 2xS2MM, 2xMM2S channels, and BDs

- Lock unit (16 locks)

- Interface to NMU

- Second level interrupt handler

- **DMA:** This DMA is composed of two S2MM channels, two MM2S channels, a shared pool of buffer descriptors, and a lock interface. The DMA supports a burst of up to 512B. The S2MM channels share access to the write ports of the AXI4 manager interface. The MM2S channels share the read ports of the AXI4 manager interface. This AXI4 manager interface is connected to the NoC NMU. There is a shared lock module that is equivalent to the one present in the AIE-ML v2 tiles. The S2MM channel consumes data coming from the incoming stream, creates AXI4 write transactions based on the address generation in the BD, and issues a burst on the write ports of the AXI4 manager interface. The MM2S channel requests AXI4 read transfers from the NoC based on the information in the DMA BDs and pushes data to stream. It handles the backpressure of the stream. Both channels have a task queue and can issue task-complete-tokens.

- **Lock Unit:** Inside the DMA there are 16 semaphore hardware locks. These locks are like the locks inside the AIE-ML v2 tile and are used by the DMA to handle synchronization for BD usage.

communicate with other blocks in the Versal architecture. Also specified are the number of streams in the AXI4-Stream interconnect interfacing with the PL, NoC, or AIE-ML v2 memory tiles, and between the AXI4-Stream switches.

||Interface Tile|Interface Tile|Interface Tile|Interface Tile|

_Chapter 3:_ AIE-ML v2 Array Interface Architecture

### **Interrupt Handling**

It is possible to setup interrupts to the PS and the platform management controller (PMC) triggered by events inside the AIE-ML v2 array. This section introduces the types of interrupts from the AIE-ML v2 array.

AIE-ML v2 has the capability to report hardware errors at the hypervisor level and are under privileged register control. To allow this AIE-ML v2 has additional logic to propagate these hardware errors to the array interface-tile and privileged registers for control and logging. The hardware errors are divided into three different categories.

- **Uncorrectable hardware errors:** 2-bit ECC errors and parity errors

- **Correctable hardware errors:** 1-bit ECC errors

- **AXI errors:** SLVERR/DECERR

These are hardwired and cannot be configured. All the events in a category are Ored together inside each tile and then along every column. The Array interface tiles contain logic to latch, mask, clear, and generate an interrupt from hardware/AXI errors received from the column.

The AIE-ML v2 array generates four interrupts that can be routed from the AIE-ML v2 array to the PMC, application processing unit (APU), and real-time processing unit (RPU). The overall hierarchy for interrupt generation from AIE-ML v2 array is as follows:

- Events get triggered from any of the AIE-ML v2 tiles or AIE-ML v2 array interface-tiles.

- Each column has first-level interrupt handlers that can capture the trigger/event generated and forward it to the second-level interrupt handler. Second-level interrupt handler, HW-error interrupt handler in AIE-ML v2 array interface-tile drive the interrupt.

- A second-level interrupt handler can drive any one of the four interrupt lines in an AIE-ML v2 array interface-tile.

- These four interrupt lines are eventually connected to the AIE-ML v2 array interface-tile.

from the AIE-ML v2 array to other blocks in the AMD Versal™ device. The diagram does not show the actual layout/placement of the array interface-tiles and the AIE-ML v2 tiles.

As indicated in the previous diagram, four interrupts originate from an NoC interface tile. These signals traverse through the PL interface tile and ultimately arrive at a configuration interface tile.
Upon reaching this point, internal errors such as the loss of PLL lock, are combined (ORed) with the incoming four interrupts. Subsequently, the resulting four interrupts are directly linked to the NPI interrupt signals on the NPI interface, which operates as a 32-bit wide memory-mapped AXI4 bus.

At the device level, the four NPI interrupts are designated as 4 to 7. There exist three distinct groups of NPI registers denoted as IMR0…IMR3, IER0…IER3, and IDR0…IDR3. Each set of registers (IMR, IER, and IDR) serves the purpose of configuring the behavior of the four NPI interrupts. The IMR registers are read-only, while the IER and IDR registers are write-only. Only the registers corresponding to NPI interrupt 4 can be programmed.

For NPI interrupts 5, 6, and 7, the three sets of registers—IMR, IER, and IDR—do not have any impact, rendering their programming ineffective. Consequently, these three interrupts cannot be controlled or masked using the NPI registers.