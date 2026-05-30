# xdna-emu Documentation

Technical reference documentation for the xdna-emu project.

## Reference Documents

Living technical reference material:

- [coverage/aie2/architecture-index.md](coverage/aie2/architecture-index.md) -- Generated
  Axis-2 coverage matrix: every SemanticOp category rolled up to its
  provenance/verification verdict. Do not hand-edit; regenerate with
  `cargo run -p xdna-archspec --example gen_coverage_artifacts`.
- [coverage/aie2/subsystem-index.md](coverage/aie2/subsystem-index.md) -- Generated
  per-subsystem coverage index: each capability-spine domain with its
  authoritative source, our implementation location, own verdict,
  rolled-up category verdict, and known-gaps narrative -- the
  regenerated form of the retired hand-maintained architecture index.
  Do not hand-edit; regenerate with
  `cargo run -p xdna-archspec --example gen_coverage_artifacts`.
- [coverage/aie2/implementation-gaps.md](coverage/aie2/implementation-gaps.md) -- Generated
  queue of subsystems only partially built or stubbed
  (`Modeled{Partial|Stub}`), the third honesty queue alongside
  perishable-queue.md / comprehension-gaps.md. Do not hand-edit;
  regenerate with
  `cargo run -p xdna-archspec --example gen_coverage_artifacts`.
- [operations.md](operations.md) -- Operational runbook: build discipline,
  formatting enforcement, test-suite costs, hardware-testing rules, NPU
  recovery escalation chain, and devbox environment state. The full-text home
  for the quick-reference rules in CLAUDE.md.
- [toolchain-sources.md](toolchain-sources.md) -- Detailed per-source breakdown
  of the authoritative toolchain sources (aie-rt, AM025 regdb, llvm-aie
  TableGen, mlir-aie device model): what each provides, key files, where the
  emulator consumes it. The detail behind CLAUDE.md's Correctness Principle.
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

## Trace Tooling

- [trace/tooling.md](trace/tooling.md) -- Per-tool inventory of the six-layer
  trace pipeline (pre-build, run, decode, compare, matrix/regression, glue) plus
  deprecated tools. The detail behind CLAUDE.md's Tracing Ecosystem section.
- [trace/strategy.md](trace/strategy.md) -- Trace-driven validation strategy
  (the logic-fuzzer end goal)
- [trace/pc-anchored.md](trace/pc-anchored.md) -- PC-anchored mode-1
  trace sweep + comparison workflow

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
- [`superpowers/findings/`](superpowers/findings/) -- Investigation notes
  for in-flight issues. Closed findings are moved to `archive/findings/`.
- [`archive/`](archive/) -- Historical content (completed plans + specs +
  notes + closed findings, dated audits, superseded investigations). Not
  actively maintained; preserved for reference when the git log isn't
  enough.

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
