_Chapter 6:_ AIE-ML v2 Configuration and Boot

#### _Chapter 6_

## AIE-ML v2 Configuration and Boot

### **AIE-ML v2 Array Configuration**

There are two top-level scenarios in the AIE-ML v2 array configuration: AIE-ML v2 array shows a high-level view of the AIE-ML v2 array and configuration interface along with the registers to the PS and the platform management controller (PMC) through the NoC.

Any memory-mapped AXI4 manager can configure any memory-mapped AXI4 register in the AIE-ML v2 array using the NoC (for example, the PS and PMC). The global registers (including PLL configuration, global reset, and security bits) in the array configuration interface tile can be programmed using the NPI interface because the global registers are mapped onto the NPI address space.

### **AIE-ML v2 Boot Sequence**

This section describes the steps involved in the boot process for the AIE-ML v2 array.

- The column clock enable value is disabled by default.

- Memory zeroization hardware logic is added that applies to each of the tile program memory, tile data memory, memory tile data memory and control module memory. For each of the memories there is a 1-bit memory-mapped AXI4 register that is set to 1 when zeroization starts. When the process is completed, the internal hardware sets this bit to 0. For the control module memory it is a 3-bit memory mapped register that is set to 0’b111.

1. Power-on and power-on-reset (POR) deassertion: Power is turned on for all modules related to the AIE-ML v2 array, including the PLL. After power-on, the PLL runs at a default speed.
The platform management controller (PMC) and NoC need to be up and running before the AIE-ML v2 boot sequence is initiated. After the array power is turned on, the PMC can deassert a POR signal in the AIE-ML v2 array.

2. AIE-ML v2 array configuration using NPI: After power-on, the PMC uses the NPI interface to program the different global registers in the AIE-ML v2 array (for example, the PLL configuration registers). The AIE-ML v2 configuration image that is required over the NPI for AIE-ML v2 array initialization comes from a flash device.

3. Enable PLL: After the PLL registers are configured (after POR), the PLL-enable bit can be enabled to turn on the PLL. The PLL then settles on the programmed frequency and asserts the LOCK signal. The source of the PLL input (ref_clk) is from hsm_ref_clk and is generated in the control interfaces and processing system (CIPS).

4. Column clock and column reset assertion/de-assertion: After the PLL is locked, all column clocks are enabled by writing a 1 to a memory-mapped AXI4 register bit. All column resets are then asserted writing a 1 to a memory-mapped AXI4 register bit. After waiting for a number of cycles, all column resets are de-asserted by writing a 0 to the same register bit.

5. The control module should be setup.

6. The array is partitioned into one or more independent partitions, with an integer number of AIE-ML v2 columns per partition. Isolation is enabled by default in all tiles, therefore must be disabled on the internal edges of each partition.

7. AIE-ML v2 array programming: The AIE-ML v2 array interface needs to be configured over the memory-mapped AXI4 from the NoC interface. This includes all program memories, AXI4 stream switches, DMAs, event, and trace configuration registers.

### **AIE-ML v2 Array Reconfiguration**

The AIE-ML v2 configuration process writes a programmable device image (PDI) produced by the bootgen tool into AIE-ML v2 configuration registers. The AIE-ML v2 configuration is done over memory-mapped AXI4 via the NoC. For more information on generating a PDI with the bootgen tool, refer to _Bootgen User Guide_ [(UG1283).](https://docs.amd.com/access/sources/dita/map?isLatest=true&url=ug1283-bootgen-user-guide&ft:locale=en-US)

The following table summarizes the reset controls available for the global AIE-ML v2 array.

|Table 13: Categories of AIE-ML v2 Resets|Col2|Col3|
|**Type**|**Trigger**|**Scope**|
|Internal power-on-reset|Part of boot sequence|AIE-ML v2 array|
|System reset|NPI input|AIE-ML v2 array|
|INITSTATE reset|PCSR bit|AIE-ML v2 array|
|Array soft reset|Software register write over NPI|AIE-ML v2 array|
|AIE-ML v2 tile column reset|Memory-mapped AIE-ML register bit in<br>the array interface tile|AIE-ML v2 tile column|
|AIE-ML v2 core reset|Memory-mapped AIE-ML v2 register bit<br>in the core tile||
|AIE-ML v2 array interface reset|From NPI register|AIE-ML v2 array interface tile|

The combination of column reset and array interface tile reset (refer to AIE-ML v2 Array Hierarchy) enables a partial reconfiguration use case where a sub-array that comprises AIE-ML v2 tiles and array interface tiles can be reset and reprogrammed without disturbing adjacent sub-arrays. The specifics of handling the array splitting and adding isolation depend on the type of use case (multi-user/tenancy or single-user/tenancy multiple-tasks).