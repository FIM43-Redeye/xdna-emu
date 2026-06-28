# Authoritative Sources -- Detailed Breakdown

The one critical rule -- **DERIVE FROM THE TOOLCHAIN** -- and the priority
order of sources live in CLAUDE.md. This file is the detailed per-source
breakdown: what each authoritative source provides, the key files, and where
the emulator already consumes it data-driven. Reach here when you need to know
*which file* in aie-rt / llvm-aie / the regdb defines a given behavior.

The **Appendix: Future Derivation Targets** at the end lists the llvm-aie sources
not yet consumed data-driven (encoding, scheduling, ABI, intrinsics). (The older
research survey that first catalogued these, `programmatic-sources.md`, is archived
under `docs/archive/`.)

## 1. aie-rt -- Hardware Abstraction Layer

**The same library that programs real silicon.** Official Xilinx repository
cloned at `../aie-rt/` (branch `xlnx_rel_v2025.2`). mlir-aie also vendors
a patched fork at `third_party/aie-rt/` for its build -- use the official
Xilinx clone as the emulator's reference source.

| What it provides | Key files |
|-----------------|-----------|
| Register offsets and bit fields (exhaustive) | `global/xaiemlgbl_params.h` |
| DMA BD programming sequences | `dma/xaie_dma_aieml.h` |
| Lock acquire/release/set/get operations | `locks/xaie_locks_aieml.h` |
| Stream switch circuit/packet configuration | `stream_switch/xaie_ss.h` |
| DMA polling and completion detection | `dma/xaie_dma_aieml.c` |
| Data structures for all hardware objects | `global/xaiegbl.h` |
| AIE2P/AIE2PS register definitions | `global/xaie2psgbl_params.h` |
| Unit tests (DMA, locks, events, etc.) | `../tests/utest/test_*.cpp` |
| DMA loopback example | `../examples/xaie_tile_dma_loopback.c` |
| Auto-routing module | `routing/xaie_routing.c` |

**Path**: `../aie-rt/driver/src/`

When implementing DMA, lock, or stream switch behavior, the aie-rt function
that does the same thing on real hardware is the reference implementation.
Example: `_XAieMl_DmaWaitForDone()` is exactly how the host polls for DMA
completion -- our `syncs_satisfied()` should match its logic.

## 2. AM025 Register Database (JSON)

Machine-readable register specification extracted from AMD's register
reference manual. Already parsed by our `regdb.rs` at startup.

| What it provides | Coverage |
|-----------------|----------|
| 1,806 register definitions | Names, offsets, widths, reset values |
| 6,412 bit field definitions | Exact positions and widths within registers |
| Per-module organization | Core, memory, DMA, locks, stream switch |

**Path**: `../mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json`

BD field parsing is already fully data-driven from this source (zero hardcoded
bit positions in `bd.rs`). Extend this pattern to other register interactions.

## 3. llvm-aie TableGen -- ISA Specification

Complete instruction set definition for the Peano compiler backend.

| What it provides | Coverage |
|-----------------|----------|
| Instruction encodings and formats | ~600+ definitions, all VLIW slots |
| Register file structure | Scalar, vector, accumulator, pointer classes |
| Semantic operation mappings | ~48 SemanticOp types (Add, Mul, Br, etc.) |
| Scheduling model / latencies | 278+ itinerary classes with cycle counts |
| Intrinsic signatures | 317 AIE2-specific intrinsics with types |
| Split field layouts | Non-contiguous operand bit positions |

**Path**: `../llvm-aie/llvm/lib/Target/AIE/`

Instruction decoding is already fully TableGen-driven. Instruction semantics
are ~33% data-driven (via SemanticOp), with the rest in legacy fallback
handlers. The goal is to close that gap.

## 4. mlir-aie Device Model -- Array Topology

Architecture-level parameters extracted from `AIETargetModel`.

| What it provides | Coverage |
|-----------------|----------|
| Array dimensions (cols, rows) | Per-device variant |
| Tile type classification | Shim, memtile, compute |
| Memory sizes, lock counts, BD counts | Per tile type |
| DMA channel counts | Including shim mux ports |
| Stream switch port counts per bundle | Switchbox and shim mux |

**Path**: `xdna-emu/tools/aie-device-models.json` (pre-extracted)
**Generator**: `xdna-emu/tools/aie-device-dump.py` (queries mlir-aie Python API)

Architecture configuration is already fully data-driven from this source.

## What Still Requires Non-Open-Source References

The open-source toolchain does not fully specify these areas. Use aietools
as a reading reference (see CLAUDE.md's Licensing section) and AM020/AM025
documentation. Read to understand the hardware, then write original code.

- **Vector operation computational semantics**: Intrinsics give function
  signatures (argument/return types, configuration word) but not what the
  operation computes. How does VMAC's configuration word affect rounding,
  saturation, and accumulator behavior? The aietools Python models at
  `amd-unified-software/aietools/data/aie_ml/lib/python_model/model/` describe these semantics
  (mulmac.py, srs_ups.py, permute.py, constants.py). Read them to understand
  the hardware behavior, then implement independently.
- **Stream switch per-port type assignments**: mlir-aie gives port counts per
  bundle, not which index maps to which bundle type. Currently hardcoded from
  AM025 in `aie2_spec.rs`.
- **Micro-timing details**: DMA pipeline depth, NoC latency, memory bank
  conflict resolution. These affect cycle-accuracy but not functional
  correctness.

**Documentation path**: `xdna-emu/docs/xdna/` (extracted text files, not PDFs)

**aietools reference paths** (read-only, never copy). aietools lives at
`amd-unified-software/aietools/` (the canonical install path; an earlier
`aietools/` symlink was removed for clarity, so always use the full path):

- Vector semantics: `amd-unified-software/aietools/data/aie_ml/lib/python_model/model/`
- Register IDs: `amd-unified-software/aietools/data/aie_ml/lib/isg/me_regid.txt` (Synopsys copyright)
- AIE API headers: `amd-unified-software/aietools/include/aie_api/` (MIT licensed)
- Event types: `amd-unified-software/aietools/data/eventanalyze/event_type_table.txt`
- Trace decoder library: `amd-unified-software/aietools/lib/lnx64.o/libxv_trace_decoder_opt.so` (Synopsys-copyrighted; readable for symbols, never linked or copied)

## Appendix: Future Derivation Targets (llvm-aie, not yet consumed)

These llvm-aie sources are authoritative for behaviour the emulator does not yet
derive data-driven -- the standing "derive from the toolchain" backlog for the
features that consume them. Paths are under `llvm-aie/llvm/lib/Target/AIE/` unless
noted. (Salvaged from the retired `programmatic-sources.md` research survey.)

- **Register encoding (MCCodeEmitter).** `MCTargetDesc/AIE2MCCodeEmitter*.h`
  (`AIE2MCCodeEmitterRegOperandDef.h` holds the complete composite encoders). Each
  `get<group>OpValue` maps a register operand to its raw bit field; inverting them
  replaces the remaining heuristic operand decoding (the guess-type-from-field-name
  path). The composite-group encoding patterns there are authoritative (e.g. `mLdaCg`,
  `mMvSclSrc`, special-register HWEncodings).

- **Scheduling / cycle model.** `AIE2Schedule.td` (~1,180 lines): functional units
  (load/store/vector/accumulator ports), 200+ instruction itinerary classes, and
  per-operand read/write latencies + bypass relationships. This is the source for
  per-instruction cycle costs (ROADMAP #322, `docs/coverage/cycle-accuracy-mission.md`
  item 1) -- each TableGen instruction's `Itinerary` field references one of these classes.

- **Calling convention / ABI.** `AIE2CallingConv.td`, `AIE2FrameLowering.h`,
  `AIE2RegisterInfo.cpp`: argument/return registers (r0-r7, p0-p5; returns r0-r1 / p0-p1),
  32-byte stack alignment, reserved + callee-saved sets (`CSR_AIE2_SaveList`), and the
  sticky status registers persisting across calls. Authoritative for stack-frame and
  call/return register handling.

- **Intrinsics / builtins.** `clang/include/clang/Basic/BuiltinsAIE.def` (AIE1) and
  `BuiltinsAIE2P.def` (AIE2P); there is no dedicated `BuiltinsAIE2.def` (AIE2 shares
  with AIE1). Builtin signatures encode input/output types and accumulator widths for
  the sync, stream, bit-op, math, and vector/MAC families -- a derivation target for
  vector-op execution semantics that are currently hand-coded.

(TableGen *instruction* definitions -- `AIE2*InstrInfo.td`, formats, patterns -- are
already consumed data-driven; see section 3 above.)
