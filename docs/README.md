# xdna-emu Documentation

Technical reference documentation for the xdna-emu project.

## Reference Documents

Living technical reference material:

- [aiesimulator.md](aiesimulator.md) -- AMD aiesimulator integration and usage
- [dma-reference.md](dma-reference.md) -- DMA engine reference (BD fields,
  sequences, polling semantics)
- [driver-diagnostics.md](driver-diagnostics.md) -- XDNA driver debugfs and
  IOCTL telemetry
- [observability-leads.md](observability-leads.md) -- Untapped debug/trace
  capabilities in aie-rt and xdna-driver, with action priorities for
  trace-sweep / sequence-skeleton work
- [programmatic-sources.md](programmatic-sources.md) -- Survey of
  programmatically-extractable sources of hardware truth

## Format Specifications

- [XCLBIN Format](formats/xclbin.md) -- Container format for NPU binaries
- [AIE Partition Format](formats/aie_partition.md) -- AIE partition section
- [CDO Format](formats/cdo.md) -- Configuration Data Objects
- [ELF Format](formats/elf.md) -- AIE core ELF executables

## Subdirectories

- [`roadmap/`](roadmap/) -- Phase-by-phase project roadmap with status
  markers (VERIFIED / OBSERVED / CLAIMED)
- [`investigations/`](investigations/) -- Debugging writeups for notable
  bugs and hardware behaviors (errata, silicon quirks)
- [`xdna/`](xdna/) -- AMD AM020/AM025 hardware reference manual extracts
- [`patches/`](patches/) -- Local patches against upstream projects
- [`plans/`](plans/) -- Active implementation plans (when present)
- [`superpowers/specs/`](superpowers/specs/) -- Active design specs from
  brainstorming sessions
- [`archive/`](archive/) -- Historical content (completed plans, dated
  audits, superseded investigations). Not actively maintained; preserved
  for reference when the git log isn't enough.

## External References

### AMD Official Documentation
- [AM020 -- AIE-ML Architecture Manual](https://docs.amd.com/)
- [AM025 -- AIE-ML Register Reference](https://docs.amd.com/)
- [UG1304 -- Versal ACAP System Software](https://docs.amd.com/r/en-US/ug1304-versal-acap-ssdg)

### Source Code References
- **XRT**: `/opt/xilinx/xrt/` -- Runtime headers, xclbin definitions
- **mlir-aie**: `../mlir-aie/` -- Compiler, device models, bootgen
- **llvm-aie (Peano)**: https://github.com/Xilinx/llvm-aie -- ISA definitions
- **aie-rt**: `../aie-rt/` -- Official Xilinx hardware abstraction layer

See [`../CLAUDE.md`](../CLAUDE.md) for the authoritative source hierarchy
and how to derive from the toolchain.
