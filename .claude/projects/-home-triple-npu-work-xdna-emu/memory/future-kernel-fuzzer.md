# Future Work: Kernel Template Generator / Differential Fuzzer

## Vision

Generate valid AIE2 kernels that exercise specific instructions, compile
them with xchesscc, and run on both the real NPU and the emulator. Compare
outputs to find executor semantic bugs.

## Architecture

```
Kernel Template Generator
    |
    v
kernel.cc (parameterized compute function)
    |
    v
xchesscc -> ELF
    |
    v
Fixed DMA/routing harness (from mlir-aie template) -> xclbin
    |
    +---> npu-runner (real hardware) -> output buffer
    |
    +---> emulator -> output buffer
    |
    v
Diff output buffers
```

## Key Design Decisions

### What varies (the kernel compute function)
- Single intrinsic calls with known inputs/outputs
- Sourced from aie_api headers (aietools/include/aie_api/)
- Each intrinsic = one test case

### What stays fixed (the harness)
- DMA descriptors: read N bytes in, write N bytes out
- Stream switch routing: shim -> compute tile -> shim
- Lock synchronization: standard acquire/release pattern
- xclbin wrapper: reuse existing mlir-aie test structure

### Data movement (the easy part)
- DMA BD programming is well-understood (aie-rt reference)
- CDO generation for routing is already in our codebase
- The mlir-aie test suite has working templates to copy from
- This is all described in cleartext (register specs, not ISA encoding)

### Instruction coverage (the hard part)
- aie_api has ~300+ intrinsics
- Many have configuration words (rounding, saturation, lane select)
- Need to generate valid configuration for each
- Could use BDD roots to enumerate valid operand combinations

## Prerequisites
- Working test runner (currently needs cleanup)
- Chess compiler accessible (have it, xchesscc works)
- npu-runner for hardware execution (exists in tools/)
- Emulator executor coverage (the thing we're validating)

## Validation Properties
- **Functional correctness**: output bytes match between NPU and emulator
- **Side effects**: lock states, DMA completion status
- NOT cycle accuracy (that's a separate concern)

## Status: FUTURE
Not started. Prerequisite: catalog executor gaps from existing Chess tests first.

## Related
- BDD validation: tools/bdd_enum/ (decoder coverage = 100%)
- Existing test runner: examples/run_mlir_aie_tests.rs
- aie_api intrinsics: aietools/include/aie_api/
- Vector semantics reference: aietools/data/aie_ml/lib/python_model/model/
