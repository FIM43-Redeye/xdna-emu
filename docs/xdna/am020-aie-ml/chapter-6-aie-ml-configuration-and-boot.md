_Chapter 6:_ AIE-ML Configuration and Boot

#### _Chapter 6_

## AIE-ML Configuration and Boot

### **AIE-ML Array Configuration**

There are two top-level scenarios in the AIE-ML array configuration: AIE-ML array configuration view of the AIE-ML array and configuration interface along with the registers to the PS and the platform management controller (PMC) through the NoC.

Any memory-mapped AXI4 master can configure any memory-mapped AXI4 register in the AIE-ML array using the NoC (for example, the PS and PMC). The global registers (including PLL configuration, global reset, and security bits) in the array configuration interface tile can be programmed using the NPI interface because the global registers are mapped onto the NPI address space.

### **AIE-ML Boot Sequence**

This section describes the steps involved in the boot process for the AIE-ML array.

- The column clock enable value is disabled by default.

- Memory zeroization hardware logic is added that applies to each of the tile program memory, tile data memory and memory tile data memory. For each of the memories there is a 1-bit memory-mapped AXI4 register that is set to 1 when zeroization starts. When the process is completed, the internal hardware sets this bit to 0.

1. Power-on and power-on-reset (POR) deassertion: Power is turned on for all modules related to the AIE-ML array, including the PLL. After power-on, the PLL runs at a default speed.
The platform management controller (PMC) and NoC need to be up and running before the AIE-ML boot sequence is initiated. After the array power is turned on, the PMC can deassert a POR signal in the AIE-ML array.

2. AIE-ML array configuration using NPI: After power-on, the PMC uses the NPI interface to program the different global registers in the AIE-ML array (for example, the PLL configuration registers). The AIE-ML configuration image that is required over the NPI for AIE-ML array initialization comes from a flash device.

3. Enable PLL: Once the PLL registers are configured (after POR), the PLL-enable bit can be enabled to turn on the PLL. The PLL then settles on the programmed frequency and asserts the **LOCK** signal. The source of the PLL input (ref_clk) is from hsm_ref_clk and is generated in the control interfaces and processing system (CIPS).

- The generation and distribution of the clock is described in the _PMC and PS Clocks_ chapter of the _Versal Adaptive SoC Technical Reference Manual_ [(AM011).](https://docs.amd.com/go/en-US/am011-versal-acap-trm)

4. Column clock and column reset assertion/de-assertion: Once the PLL is locked, all column clocks are enabled by writing a 1 to a memory-mapped AXI4 register bit. All column resets are then asserted writing a 1 to a memory-mapped AXI4 register bit. After waiting for a number of cycles, all column resets are de-asserted by writing a 0 to the same register bit.

5. The array is partitioned into one or more independent partitions, with an integer number of AI Engine (AIE) columns per partition. Isolation is enabled by default in all tiles, therefore must be disabled on the internal edges of each partition.

6. AIE-ML array programming: The AIE-ML array interface needs to be configured over the memory-mapped AXI4 from the NoC interface. This includes all program memories, AXI4 stream switches, DMAs, event, and trace configuration registers.

### **AIE-ML Array Reconfiguration**

The AIE-ML configuration process writes a programmable device image (PDI) produced by the bootgen tool into AIE-ML configuration registers. The AIE-ML configuration is done over memory-mapped AXI4 via the NoC. Any master on the NoC can configure the AIE-ML array.
For more information on generating a PDI with the bootgen tool, refer to _Bootgen User Guide_ [(UG1283).](https://docs.amd.com/access/sources/dita/map?isLatest=true&url=ug1283-bootgen-user-guide&ft:locale=en-US)

The AIE-ML array can be reconfigured at any time. The application drives the reconfiguration.
Safe reconfiguration requires:

- Ensuring that reconfiguration is not occurring during ongoing traffic.

- Disabling the AIE-ML to PL interface prior to reconfiguration.

- Draining all data in the sub-region before it is reconfigured to prevent side-effects from remnant data from a previous configuration.

Complete reconfiguration is currently used to reconfigure the AIE-ML array. The global reset is asserted for the AIE-ML array and the entire array is reconfigured by downloading a new configuration image.

The PMC and PS are responsible for initializing the AIE-ML array. The following table summarizes the reset controls available for the global AIE-ML array.

|Table 13: Categories of AIE-ML Resets|Col2|Col3|
|**Type**|**Trigger**|**Scope**|
|Internal power-on-reset|Part of boot sequence|AIE-ML array|
|System reset|NPI input|AIE-ML array|
|INITSTATE reset|PCSR bit|AIE-ML array|
|Array soft reset|Software register write over NPI|AIE-ML array|
|AIE-ML tile column reset|Memory-mapped AIE-ML register bit in<br>the array interface tile|AIE-ML tile column|
|AIE-ML array interface reset|From NPI register|AIE-ML array interface tile|

The combination of column reset and array interface tile reset (refer to AIE-ML Array Hierarchy) enables a partial reconfiguration use case where a sub-array that comprises AIE-ML tiles and array interface tiles can be reset and reprogrammed without disturbing adjacent sub-arrays. The specifics of handling the array splitting and adding isolation depend on the type of use case (multi-user/tenancy or single-user/tenancy multiple-tasks).