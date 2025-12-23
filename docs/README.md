# xdna-emu Documentation

Technical reference documentation for the xdna-emu project.

## Format Specifications

- [XCLBIN Format](formats/xclbin.md) - Container format for NPU binaries
- [AIE Partition Format](formats/aie_partition.md) - AIE partition section structure
- [CDO Format](formats/cdo.md) - Configuration Data Objects (tile setup commands)
- [ELF Format](formats/elf.md) - AIE core ELF executables

## Architecture References

The following documentation will be added as the emulator develops:

- **AIE2 Tile Architecture** - Core, memory, DMA, stream switch per tile
- **Register Definitions** - Memory-mapped register layout (see NEXT-STEPS.md)
- **DMA Descriptor Format** - Buffer descriptor structures
- **Stream Switch Configuration** - Routing and packet switching
- **Lock/Synchronization** - 64 locks per tile, acquire/release semantics

See AMD documentation (AM020, AM025) for authoritative hardware references.

## External Documentation

### AMD Official Docs
- [AM020 - AIE-ML Architecture Manual](https://docs.amd.com/) - Core architecture
- [AM025 - AIE-ML Register Reference](https://docs.amd.com/) - Register definitions
- [UG1304 - Versal ACAP System Software](https://docs.amd.com/r/en-US/ug1304-versal-acap-ssdg) - CDO/PDI formats

### Source Code References
- **XRT**: `/opt/xilinx/xrt/` - Runtime headers, xclbin definitions
- **mlir-aie**: `/home/triple/npu-work/mlir-aie/` - Compiler, device models, bootgen
- **llvm-aie (Peano)**: https://github.com/Xilinx/llvm-aie - ISA definitions

### Key Files in mlir-aie
```
include/aie/Dialect/AIE/IR/AIETargetModel.h  - Device parameters (memory sizes, locks, etc.)
third_party/bootgen/cdo-*.{h,c}              - CDO format implementation
lib/Targets/AIETargetCDODirect.cpp           - CDO generation from MLIR
```

### Key Files in XRT
```
include/xrt/detail/xclbin.h    - XCLBIN format definition
include/aiebu/aiebu_assembler.h - AIE binary utilities
```

## Test Data

Real xclbin files for testing are available in mlir-aie:
```
/home/triple/npu-work/mlir-aie/build/test/npu-xrt/*/aie.xclbin
/home/triple/npu-work/mlir-aie/programming_examples/*/build/*.xclbin
```
