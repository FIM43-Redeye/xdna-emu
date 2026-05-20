# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace Setup

**Start Claude from the parent `npu-work` directory**, not from xdna-emu directly:

```bash
cd /home/triple/npu-work
claude
```

This ensures:
- NPU development environment auto-activates (Python 3.13, mlir-aie, Peano compiler)
- XRT and hardware access are configured
- Helper functions like `npu-compile` and `npu-run-quiet` are available

The workspace contains:
- `xdna-emu/` - This emulator project (main focus)
- `aie-rt/` - Official Xilinx aie-rt (hardware abstraction layer, tests, FAL)
  - `driver/src/` - Register definitions, DMA/lock/stream switch implementations
  - `driver/tests/` - Unit tests for DMA, locks, events (valuable reference)
  - `fal/` - Full Abstraction Layer (profiling, tracing, resource management)
- `mlir-aie/` - MLIR-based AIE compiler, test suite, and device models
  - `third_party/aie-rt/` - Patched aie-rt fork (mlir-aie build dependency only)
  - `lib/Dialect/AIE/Util/aie_registers_aie2.json` - AM025 register database
- `llvm-aie/` - Peano compiler (LLVM with AIE backend, TableGen ISA definitions)

See `/home/triple/npu-work/CLAUDE.md` for environment details.

---

There is one absolutely critical rule to always keep to: **DERIVE FROM THE TOOLCHAIN.** The open-source toolchain (aie-rt, llvm-aie, mlir-aie) is the authoritative specification for hardware behavior. Never hardcode what can be extracted. See "Correctness Principle" below for the full source hierarchy and guidance.

## Project Vision

**xdna-emu** is an open-source, cycle-accurate emulator and visual debugger for AMD XDNA NPUs (Ryzen AI). It fills the gap left by AMD's proprietary aiesimulator (which is license-gated, CLI-only, and not cycle-accurate).

### Goals
1. **Binary-Compatible Emulation**: Load real `.xclbin` binaries, execute them faithfully
2. **Visual Debugging**: See tiles, data flow, DMA transfers, locks, routing in real-time
3. **Hardware Validation**: Run in emulator -> run on real NPU -> compare results
4. **Optional Compilation**: Invoke Peano compiler to build kernels, run immediately

### The Dream Workflow
```
Write kernel.cc
    |
xdna-emu compiles via Peano -> kernel.xclbin
    |
xdna-emu emulates (visual, step-through, breakpoints)
    |
"Run on Hardware" button
    |
Same binary executes on real NPU via XRT
    |
Compare emulated vs actual results
```

### Licensing and Relationship to AMD

This project is **MIT-licensed** and exists to help the AMD NPU ecosystem.
The emulator is orders of magnitude slower than real silicon -- it is a
development tool, not a hardware substitute. Its purpose is to lower the
barrier to entry for NPU programming, which benefits AMD by expanding the
developer community for their hardware.

**Source derivation policy:**

All emulator code is original. Hardware behavior is derived from these sources,
in order of preference:

1. **Open-source toolchain** (aie-rt, llvm-aie, mlir-aie) -- Apache 2.0 /
   MIT licensed. Primary and preferred source for all emulator behavior.
   Derive from these wherever possible.

2. **Hardware observation** -- Running binaries on the real NPU we own and
   observing results. The hardware itself is ground truth.

3. **aietools** (AMD proprietary, locally installed) -- Used strictly as a
   **reading reference** to understand hardware behavior that the open-source
   toolchain does not document (primarily vector compute semantics). Never
   copy code or data from aietools into this repository. Read, understand
   the hardware facts, then write original implementations. aiesimulator may
   be used as a debugging aid to understand where the emulator diverges from
   expected behavior, but the real NPU is always ground truth.

4. **AM020/AM025 documentation** -- AMD architecture reference manuals for
   areas not covered elsewhere.

**What this means in practice**: when implementing a feature, comment the
source of behavioral knowledge as the hardware behavior itself (e.g.,
"Rounding matches observed NPU output" or "BD field layout per AM025 register
database"), not as proprietary tool internals. The knowledge is about how
the silicon works; the implementation is ours.

## Current Status

See [ROADMAP.md](ROADMAP.md) for detailed status with confidence markers
(VERIFIED / OBSERVED / CLAIMED).

| Phase | Status | Summary |
|-------|--------|---------|
| [1. Core Accuracy](docs/roadmap/phase1-core-accuracy.md) | **100% ISA** | 4815/4815 ISA test points; bridge tests pass on ~75 mlir-aie kernels |
| [2. Toolchain Integration](docs/roadmap/phase2-toolchain-integration.md) | Partial | XRT plugin works; bridge tests run; Peano compilation not integrated |
| [3. Developer Experience](docs/roadmap/phase3-developer-experience.md) | GUI exists | GUI renders; debugging features not built |
| [4. Validation & Testing](docs/roadmap/phase4-validation-testing.md) | Active | Dual-compiler bridge tests, trace sweep, parallel HW |
| [5. Production Readiness](docs/roadmap/phase5-production-readiness.md) | Not started | |
| [6. Community & Ecosystem](docs/roadmap/phase6-community-ecosystem.md) | Not started | |

Run `cargo test --lib` to see the current test count. Do not rely on
numbers written in documentation -- they go stale within a session.

## Target Devices

These are the official device names from the xdna-driver source:

| Driver ID | Product Name | Codename | Architecture | Array Size | Status |
|-----------|--------------|----------|--------------|------------|--------|
| NPU1 | Ryzen AI | Phoenix/Hawk Point | AIE2 (XDNA) | 5 cols x 6 rows | **Primary target** |
| NPU4 | Ryzen AI 300 | Strix Point | AIE2P (XDNA2) | 5 cols x 6 rows | Planned |
| NPU5 | Ryzen AI Max | Strix Halo | AIE2P (XDNA2) | 8+ cols x 6 rows | Planned |
| NPU6 | (TBD) | Krackan | AIE2P (XDNA2) | 5 cols x 6 rows | Planned |

Array sizes include the shim tile row (row 0). Driver IDs NPU2/NPU3 are
prototypes marked for deprecation -- not consumer devices.

We are starting with **Phoenix (NPU1/AIE2)** because it is the hardware we have.
AIE2P support will be incremental once AIE2 is solid.

### mlir-aie Device Target Naming

mlir-aie uses different device names that can be confusing:

| mlir-aie target | Maps to Driver | Architecture |
|-----------------|----------------|--------------|
| `npu1`, `npu1_Xcol` | NPU1 (Phoenix) | AIE2 |
| `npu2`, `npu2_Xcol` | NPU4 (Strix) | AIE2P |
| `xcvc1902` | N/A (Versal) | AIE1 |
| `xcve2802` | N/A (Versal) | AIE2 |

**Not in scope**: Versal FPGAs (AIE1 + PL fabric) - different use case, no local hardware. However, AIE1 support may be added later since the TableGen parser handles multiple architectures.

## Correctness Principle: Derive From the Toolchain

**CRITICAL: The open-source toolchain IS the hardware specification. Derive
emulator behavior from it. Never hardcode what can be extracted.**

This project's accuracy depends on treating mlir-aie, llvm-aie, and aie-rt as
the authoritative sources of truth for hardware behavior. Every hardcoded
constant, bit position, or behavioral assumption is a potential bug. The
toolchain evolves with the hardware -- if we derive from it, we evolve too.

**The rule**: before implementing any hardware feature, check whether the
toolchain already defines it. Only fall back to AM020/AM025 documentation for
things the toolchain genuinely does not cover (primarily vector operation
computational semantics).

### Authoritative Sources (in priority order)

#### 1. aie-rt -- Hardware Abstraction Layer

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

#### 2. AM025 Register Database (JSON)

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

#### 3. llvm-aie TableGen -- ISA Specification

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

#### 4. mlir-aie Device Model -- Array Topology

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

### What Still Requires Non-Open-Source References

The open-source toolchain does not fully specify these areas. Use aietools
as a reading reference (see Licensing section above) and AM020/AM025
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

### Research Guidance

**Use Explore agents for documentation and toolchain research.** These sources
are extensive. Rather than reading files directly and burning context, spawn an
Explore agent to navigate them. Good queries:

- "How does aie-rt implement DMA channel start?" (check aie-rt)
- "What are the BD field positions for shim tiles?" (check regdb JSON)
- "What does the VMAC configuration word control?" (check AM020)
- "How many stream switch ports does a memtile have?" (check device model)

### Binary Formats
- **XCLBIN**: Container (ELF cores + PDI/CDO configuration)
- **ELF**: Per-core executables
- **CDO**: Configuration Data Objects (DMA descriptors, routing)

## What We're Emulating

### Per Tile
- **Compute Core**: VLIW processor, 256-bit vector unit
- **Local Memory**: 64KB (compute tile), 512KB (mem tile)
- **DMA Engine**: 2 channels, n-dimensional addressing
- **Locks**: 64 synchronization primitives
- **Stream Switch**: Circuit/packet routing

### Array Level
- **Shim Tiles**: DDR interface via NoC
- **Mem Tiles**: Shared memory between columns
- **Routing**: Configured stream switches

### Execution Model
- Cores run independently (no global clock)
- DMAs operate concurrently with cores
- Locks synchronize between tiles
- Stream switches route based on CDO configuration

## Technical Decisions

### Language: Rust

Chosen for:
- Memory safety (emulator = complex state machine)
- Excellent binary parsing (`goblin`, `nom`)
- Built-in profiling (`cargo flamegraph`, `cargo bench`)
- Modern tooling (`cargo` just works)

### GUI Framework: egui

Using `eframe` (egui framework) for the visual debugger:
- Pure Rust, no external dependencies
- Immediate mode - simple state management
- Cross-platform (Linux, Windows, macOS, web)

## Component Documentation

Detailed documentation for each module lives in `.claude/components/`. Read
the relevant file when working on that area of the codebase.

| Component | File | When to Read |
|-----------|------|-------------|
| Device model | [`.claude/components/device.md`](.claude/components/device.md) | Working on tiles, array, DMA, streams, locks, host memory (`src/device/`) |
| Interpreter | [`.claude/components/interpreter.md`](.claude/components/interpreter.md) | Working on instruction decode, execution, timing, multi-core (`src/interpreter/`) |
| Parser | [`.claude/components/parser.md`](.claude/components/parser.md) | Working on XCLBIN, ELF, or CDO parsing (`src/parser/`) |
| TableGen | [`.claude/components/tablegen.md`](.claude/components/tablegen.md) | Working on ISA definitions, decoder tables, llvm-aie integration (`crates/xdna-archspec/src/aie2/isa/`, with consumers in `src/interpreter/decode/`) |
| Testing | [`.claude/components/testing.md`](.claude/components/testing.md) | Working on tests, test runner, FFI, NPU instructions, config (`src/testing/`, `src/npu/`, `src/ffi/`, `tests/`) |
| Visual | [`.claude/components/visual.md`](.claude/components/visual.md) | Working on the GUI debugger (`src/visual/`) |

Top-level source files not covered by component docs:
- `src/main.rs` -- CLI and GUI entry point
- `src/lib.rs` -- crate root, module declarations
- `src/integration/mod.rs` -- external tool integration (placeholder)

## Related Resources

- **aie-rt**: `../aie-rt/driver/src/` - Official Xilinx hardware abstraction layer (branch `xlnx_rel_v2025.2`). The reference implementation for DMA, locks, and stream switch programming. Includes unit tests, examples, FAL, and AIE2P definitions. (mlir-aie vendors a patched fork at `third_party/aie-rt/` for its build -- that is NOT the emulator's reference.)
- **mlir-aie**: `../mlir-aie` - MLIR-based AIE compiler, test binaries, device models, AM025 register database JSON
- **llvm-aie**: `../llvm-aie` (local clone) - Peano compiler, ISA definitions via TableGen
- **aietools**: `../amd-unified-software/aietools` - AMD proprietary tools (Chess compiler, aiesimulator, analysis tools). Read-only reference for hardware semantics not covered by open-source toolchain. See Licensing section.
- **xdna-driver**: `/home/triple/npu-work/xdna-driver` - Linux kernel driver, device definitions
- **XRT**: https://github.com/Xilinx/XRT - runtime (installed at /opt/xilinx/xrt)

## Available Development Tools

### XRT Tools (in /opt/xilinx/xrt/bin/)
- **xclbinutil**: Essential tool for XCLBIN inspection, extraction, and manipulation
  - `xclbinutil --info --input file.xclbin` - dump sections and metadata
  - `xclbinutil --dump-section` - extract specific sections
- **xrt-smi**: Device management and diagnostics

### MLIR-AIE Tools (in mlir-aie/my_install/ or ironenv/)
- **llvm-objdump**: Disassemble AIE ELF binaries
  - `llvm-objdump -d file.elf` - disassemble to see instruction mnemonics
- **aie-translate**: MLIR to various formats
- **aie-opt**: MLIR optimization passes

### aietools (AMD proprietary, read-only reference)
- **aiesimulator**: Cycle-accurate AIE simulator (debugging aid, not oracle)
  - `aiesimulator --pkg-dir=<dir>` - run simulation
  - Variants: `aie2simmsm` (cycle-accurate), `aie2simmsm_func` (functional)
- **elfanalyzer**: Static analysis of AIE ELF binaries (code layout, memory map)
- **hwanalyze**: Hardware trace analysis from real NPU event traces
- **eventanalyze**: VCD/trace event analysis (95+ event types)
- **xchesscc**: Chess compiler (via `xchesscc_wrapper` for architecture selection)
- Note: aietools LD_LIBRARY_PATH must be appended (not prepended) to avoid
  shadowing system libstdc++ -- see activate-npu-env.sh

### RyzenAI-SW Tools
- Located in `/home/triple/npu-work/RyzenAI-SW/`
- Contains NPU driver source and examples

## Build Commands

```bash
# Build
cargo build

# Build optimized
cargo build --release

# Run
cargo run -- path/to/binary.xclbin

# Test
./scripts/run-tests.sh          # All tests (doc tests run with nice 19)
./scripts/run-tests.sh --lib    # Fast: library tests only
./scripts/run-tests.sh --doc    # Doc tests only (nice'd, limited parallelism)
cargo test --lib                # Direct: library tests without script
TMPDIR=/tmp/claude-1000 cargo test --lib  # Same, but sandbox-safe (temp dirs)

# Benchmark
cargo bench

# Profile (generates flamegraph.svg)
cargo install flamegraph
cargo flamegraph --release -- path/to/binary.xclbin

# Bridge tests (dual-compiler, requires XRT + NPU)
./scripts/emu-bridge-test.sh                    # Full run
./scripts/emu-bridge-test.sh --no-hw add_one    # Quick EMU-only

# Trace comparison binary
cargo build --release --bin trace-compare

# Build FFI crate for bridge tests (debug .so loaded by XRT plugin)
cargo build -p xdna-emu-ffi
# Build + install plugin (release .so + C++ wrapper)
./scripts/rebuild-plugin.sh
```

**Building for bridge tests**: `cargo build` builds the main binary but NOT
the FFI crate's cdylib `.so`. Bridge tests load `target/debug/libxdna_emu.so`
(or release). After changing emulator code, run `cargo build -p xdna-emu-ffi`
to update the `.so`, or use `./scripts/rebuild-plugin.sh` for the full
release build + install cycle.

**Note on doc tests**: Doc tests spawn separate processes that each load TableGen
files from llvm-aie. The test script runs them with `nice -n 19` and limited
parallelism to avoid overwhelming the system during other work.

**Long-running commands**: Never pipe a long-running command through `tail`,
`head`, or any filter that buffers -- the output will appear to hang because
the filter waits for EOF. Instead, redirect output to a file and read it
afterward, or use `run_in_background` and check the output file with Read.
The same applies to `dmesg -w` and other streaming commands: redirect to a
file, never pipe interactively.

**Sandbox mode**: The sandbox is used to enable unattended work, not to
prevent tool use. `dangerouslyDisableSandbox` still requires user
authorization, so it is safe to use -- but use it sparingly, because each
invocation pauses work until the user approves. Prefer staying in the
sandbox by default:
- Use `$TMPDIR` (or `/tmp/claude-1000/`) instead of `/tmp` for temp files
- Many tests that create temp dirs will fail in the sandbox because they
  use `std::env::temp_dir()` which resolves to `/tmp`. Set
  `TMPDIR=/tmp/claude-1000` when running `cargo test` to fix this.
- Only reach for `dangerouslyDisableSandbox` when the sandbox is genuinely
  blocking (e.g., network access, writing outside allowed paths).

## Test Infrastructure

### Primary: Bridge Test Suite (`scripts/emu-bridge-test.sh`)

The **XRT bridge path is the real validation target.** It exercises the full
hardware-equivalent flow: `test.exe -> XRT -> plugin -> emulator`.

**Dual-compiler**: Every test is compiled with BOTH compilers by default.
Chess is ground truth; Peano failures are informational. Five phases:
discover, compile (parallel), run HW (-j5), run EMU (-j nproc), report.

Flags: `--chess-only`, `--peano-only`, `--no-hw`, `--compile`,
`--serial-hw`, `--sweep`, `--trace=pc-anchored`, `-v <filter>`.

**Build dirs**: `mlir-aie/build/test/npu-xrt/$name/chess/` and `peano/`
**Results**: `build/bridge-test-results/YYYYMMDD/` (symlink at `build/bridge-test-results/latest`)
**Plugin**: `pkexec cp xrt-plugin/build/libxrt_driver_emu.so.2 /opt/xilinx/xrt/lib/`

### Backup: in-process xclbin runner

`src/testing/xclbin_suite.rs` runs xclbins in-process against the emulator
without going through XRT, useful for isolated subsystem testing. Driven
from unit tests; not a separate binary. **NEVER run a real-NPU capture
inside the Claude Code sandbox** -- license checks and filesystem
isolation will fail.

## Tracing Ecosystem

Binary trace comparison between emulator and real NPU hardware. All traces
converge to Perfetto JSON (viewable at ui.perfetto.dev).

**Division of labor.** Upstream mlir-aie owns trace *injection* (via the
declarative IRON API) and *decoding* (`aie.utils.trace.parse_trace`). Our
local layer owns *prep glue, sweep, matrix, regression-verification* on top.
The local pieces are the regression gate for the emulator refactor -- don't
drop them in favor of upstream-only solutions until upstream ships an
equivalent 8-batch sweep + matrix diff.

For deeper context on the strategy, see [`docs/trace/strategy.md`](docs/trace/strategy.md).

**Active pipeline (six layers, executed top to bottom):**

1. **Pre-build** -- one-shot per test before aiecc runs.
   | Tool | Purpose |
   |------|---------|
   | `tools/trace-prepare.py` | Compile-side prep: injects trace MLIR via mlir-trace-inject, patches `test.cpp` via cpp_trace_patch. Used by `scripts/emu-bridge-test.sh`. |
   | `tools/mlir-trace-inject.py` | Declarative MLIR injector, calls mlir-aie's IRON API. |
   | `tools/cpp_trace_patch.py` | Tree-sitter C++ transformer for `test.cpp` (trace BO alloc + set_arg). |
   | `tools/trace_config.py`, `tools/trace_config_schema.json`, `tools/trace_config_examples/` | Config layer (events per tile, mode, packet IDs). |

2. **Run** -- per event-set, swaps event slots without recompiling.
   | Tool | Purpose |
   |------|---------|
   | `bridge-runner/bridge-trace-runner` | C++ multi-batch orchestrator (HW + EMU). Supports `--batch-stdin` (RESET command for worker reuse) and `--snapshot-on-timeout <dir>` (captures CORE/DMA/lock register state on `run.wait` timeout, before driver recovery wipes it). |
   | `tools/trace-sweep.py` | Gen-2 multi-tile sweep, 8-event batches per tile. |
   | `tools/trace-patch-events.py` | Gen-2 patcher: rewrites event-slot bytes in compiled `insts.bin`. |

3. **Decode** -- raw trace buffer to events JSON / cycles / Perfetto.
   | Tool | Purpose |
   |------|---------|
   | `tools/parse-trace.py` | Single-source decoder, wraps mlir-aie's `parse_trace`. Emits any combination of flat events JSON, cycles scalar, raw Perfetto, raw command stream. |
   | `tools/trace_decoder/` | In-tree decoder backend (default; alternate to upstream parser). |

4. **Compare** -- HW vs EMU events.
   | Tool | Purpose |
   |------|---------|
   | `src/bin/trace_compare.rs` | Rust comparator (event-sequence diff with anchor alignment, configurable tolerance, per-tile MATCH/DRIFT/ORDER_MISMATCH/MISSING_EVENT verdict). Consumes events JSON from parse-trace.py. |

5. **Matrix / regression** -- drive across (test x compiler x tile).
   | Tool | Purpose |
   |------|---------|
   | `scripts/trace-sweep-all.py` | Drives trace-sweep.py across the bridge matrix. |
   | `scripts/show-sweep-matrix.py` | Renders + diffs sweep result trees. |
   | `scripts/merge-sweep-results.py` | Post-hoc merge of sweep result dirs. |
   | `scripts/trace-quarantine.txt` | Tests to skip in trace mode (deadlocks, IOMMU faults). |
   | `scripts/trace-incompat-tests.txt` | Tests structurally incompatible with trace injection. |

6. **Glue** -- end-to-end orchestration + EMU-side emission.
   | Tool | Purpose |
   |------|---------|
   | `scripts/emu-bridge-test.sh` | End-to-end bridge driver (`--sweep`, `--trace=pc-anchored`, etc.). |
   | `src/device/trace_unit/` | Emulator-side trace unit -- writes the same packet-stream format HW does. |

Build the Rust comparator: `cargo build --release --bin trace-compare`.

**Deprecated tools** (archive at `tools/deprecated/`, kept for reference;
do not add new callers). Files with a `-v1` suffix were renamed during the
gen-2 rollout so their base names stop colliding with current tools at the
top of `tools/`:

| Tool | Original purpose |
|------|------------------|
| `tools/deprecated/trace-inject.py` | Inject trace routing into MLIR (capacity planner, collision-aware IDs) |
| `tools/deprecated/trace-sweep-v1.py` | Multi-batch event sweep orchestrator (superseded by `tools/trace-sweep.py`) |
| `tools/deprecated/trace-trim.py` | Strip sentinel padding from raw trace buffers |
| `tools/deprecated/trace-merge.py` | Merge per-batch Perfetto JSON with TRUE anchor alignment |
| `tools/deprecated/trace-patch-events-v1.py` | Patch event slots in compiled insts.bin (superseded by `tools/trace-patch-events.py`) |
| `tools/deprecated/trace-bridge.sh` | End-to-end shell driver, superseded by `scripts/emu-bridge-test.sh` |
| `tools/deprecated/trace-compare.py` | Python HW/EMU comparator, superseded by `src/bin/trace_compare.rs` |

See `tools/deprecated/README.md` for the standdown rationale.

## XRT Plugin (`xrt-plugin/`)

Driver plugin replacing the real XDNA kernel driver. XRT loads the .so,
which delegates to the Rust emulator via FFI (`src/ffi/`).

**Build**: `./scripts/rebuild-plugin.sh` (debug) / `--release` (release)
**Install**: rebuild-plugin.sh symlinks the build output into `/opt/xilinx/xrt/lib/`.
**Env contract**:
- `XDNA_EMU` -- presence (any value) activates the emulator. Plugin replaces
  `xrt::device(0)` so tests target the emulator with no BDF magic. Unset =
  real HW.
- `XDNA_EMU_RUNTIME=release|debug` -- which `.so` profile to dlopen
  (default `debug`).
- For HW invocations from a poisoned shell, use `env -u XDNA_EMU XDNA_EMU_RUNTIME`.

## How To Begin

1. Read [ROADMAP.md](ROADMAP.md) for the development plan and confidence markers
2. Check the relevant [phase documentation](docs/roadmap/) for current details
3. Run `cargo test --lib` to verify everything works
4. Read the component doc (`.claude/components/`) for the module you are working on

## Feature Implementation Policy

**Finish what you start.** When implementing an isolated subsystem (control
packets, DMA padding, cascade flows, etc.), build it to 100% coverage before
moving on. A half-implemented feature is worse than an unimplemented one --
it creates false confidence and makes debugging harder because you can never
be sure whether a failure is "the feature isn't done yet" or a real bug.

This means: all test cases for the feature pass, edge cases are handled, and
the implementation matches hardware behavior across the full test matrix
(both compilers, both HW and EMU). Only then move to the next feature.

## Debugging Guidelines

**Match real hardware behavior.** When debugging emulator issues, the goal is
always to reproduce what the silicon does, not to invent workarounds or
simplified approximations. If aie-rt does something a particular way, we do
it that way too.

When investigating a failing test:
1. Start from the failing assertion and work backward through the data path.
2. Do not jump to hypotheses about unrelated subsystems (e.g., do not
   investigate stream routing if the data is wrong at source memory level).
3. If unsure about hardware semantics, ask rather than guess.

**Memory watch mechanism.** Set `XDNA_EMU_WATCH` to log every memory access
to specified address ranges. Format: comma-separated `address:bytes` pairs
(hex, 0x prefix optional, bytes defaults to 4). Requires `RUST_LOG=info`.

```bash
# Watch three addresses (40 bytes each) during a bridge test
XDNA_EMU=1 XDNA_EMU_WATCH=0xC000:40,0x428:40,0x400:40 RUST_LOG=info \
  ./test.exe 2>watch.log
grep "\[WATCH\]" watch.log
```

Output shows cycle-correlated DMA and core memory operations:
```
[WATCH] cycle=200 DMA-WR   tile=(0,2) addr=0x0C000 value=0x00000001 ch=S2MM0
[WATCH] cycle=249 DMA-RD   tile=(0,2) addr=0x00400 value=0x00000000 ch=MM2S2
```

Note: DMA watches use tile-local offsets. Core watches use the full 20-bit
address space (e.g., 0x70400 for local memory at offset 0x400).

**Correctness before performance.** Do not optimize (including multithreading)
until emulator behavior is indistinguishable from real hardware. Making wrong
answers faster helps nobody, and threading introduces its own bugs that muddy
correctness work.

## Validation

Always run `cargo test --lib` after making changes. Do not consider work
complete until tests pass. If tests were passing before your changes and are
now failing, that is a regression to fix before moving on.

**Planned: differential fuzzing.** The long-term validation strategy is a
logic fuzzer that generates valid kernels, runs them on both the emulator and
real NPU hardware, and compares results. This is future work -- do not start
building it until hand-written test coverage confirms baseline correctness.

## Operational Notes

Durable rules and operational procedures that didn't fit cleanly elsewhere.
These were previously scattered across Claude memory files; consolidating
them here keeps the project self-documenting.

### Build discipline

**Rebuild before testing.** The XRT plugin loads `libxdna_emu.so` at
runtime. `cargo test --lib` does NOT trigger a plugin rebuild, and
ISA/bridge tests load whatever `.so` is on disk. After any Rust source
change, run `cargo build` (and `cargo build --release` if release is
being exercised). Stale `.so`s have produced phantom bugs, including a
memorable "concurrency bug" that was really just a stale debug lib.

**Profile clarification.** The rule "one build at a time" means one
invocation of a given target, not one build overall. `cargo build` and
`cargo build --release` can run concurrently -- cargo handles the
locking between them. Don't run the same command twice concurrently.

**Stale cargo warnings.** If `cargo build` prints
`Plugin install failed: Read-only file system (os error 30)` but the
filesystem is writable, it's a cached warning from a prior
sandbox-blocked build. `cargo clean -p xdna-emu` (with the matching
profile) flushes it.

### Code formatting

The repo has a tuned `rustfmt.toml` at the root. Stable rustfmt only;
non-default settings preserve the project's semantic conventions
(import groups, compact struct literals, ~110-char lines, in-line method
chains).

The codebase is **fully fmt'd** as of the Path B convergence sprint
(2026-04-29). New code stays clean via two enforcement layers:

1. **`PostToolUse` hook** in `.claude/settings.json` auto-runs rustfmt
   on any `.rs` file Claude edits (via `.claude/hooks/rustfmt-edited.sh`,
   which uses stdin mode to avoid module recursion).
2. **Editor format-on-save** handles the same role for human edits.
3. **Pre-commit hook** at `scripts/git-hooks/pre-commit`. Runs
   `cargo fmt --check` on commits that touch `*.rs`. Blocks the
   commit on drift with an actionable message.
4. **CI check** (TODO -- no CI workflow yet): once added, run
   `cargo fmt --check` as a required step.

**One-time local setup per checkout (two configs):**

```bash
git config core.hooksPath scripts/git-hooks
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

The first activates the pre-commit hook. The second makes `git blame`
skip the chunked `cargo fmt` commits listed in
`.git-blame-ignore-revs` and attribute lines to the original content
author. GitHub honors the blame-revs file automatically; no per-user
setup there.

**Don't run `cargo fmt` repo-wide in one commit.** Even now that
everything's clean, future bulk reformats (e.g., from a config tweak)
should stay chunked by subtree -- each commit reviewable, each SHA
appended to `.git-blame-ignore-revs`.

### Test suite costs

- `./scripts/isa-test.sh`: ~5-10 minutes (release build + 123 HW batches)
- `./scripts/emu-bridge-test.sh`: ~15-30 minutes (dual-compiler, HW + EMU)

These are expensive; don't re-run them just to "check progress." Run
once after a batch of fixes and examine results. For targeted re-runs,
use filter arguments or single-test invocations.

**Use tee for long runs.** When backgrounding a test, pipe through tee
so output is both live-monitorable and logged:

```bash
./scripts/isa-test.sh 2>&1 | tee /tmp/claude-1000/isa-test.log
```

A bare redirect hides progress from both sides. (`/tmp` is fine here --
the log is ephemeral.)

### Hardware testing

**Never run two hardware test suites concurrently.** Bridge tests and
ISA tests both target the NPU device; running them in parallel causes
them to fight for the device and both must be killed. Run hardware
suites sequentially. Pure `cargo test --lib` unit tests are safe to run
alongside since they don't touch hardware.

### NPU recovery

When the NPU wedges, recovery escalates through:

1. **Driver reload** -- handles most TDR recoveries:
   ```bash
   pkexec sh -c 'modprobe -r amdxdna && modprobe amdxdna'
   ```
   **Caveat: not safe in a poisoned-mailbox state.** If the user-context
   mailbox has been killed by an `AIE_RW_ACCESS` memtile-read timeout
   (or an equivalent firmware-level hang), `modprobe -r` itself wedges
   uninterruptibly in `drm_dev_unplug -> synchronize_srcu` and reboot
   becomes the only recovery path.  We previously set `tdr_dump_ctx=1`
   to disable TDR recovery and avoid that poisoning, but that turned
   every firmware silent-drop into a permanent `aie2_hmm_invalidate`
   wedge requiring reboot anyway (see
   `docs/superpowers/findings/2026-05-13-chain-exec-npu-silent-drop-on-phoenix.md`).
   We removed `tdr_dump_ctx=1` on 2026-05-13: TDR now actually recovers,
   so `modprobe -r` is again attempted-first when the NPU wedges.

   **2026-05-20 refinement** (see
   `docs/superpowers/findings/2026-05-20-amdxdna-tdr-recovery-incomplete-on-phoenix.md`):
   The `synchronize_srcu` wedge risk appears to be **failure-mode-specific**
   to AIE_RW_ACCESS-style poisoning, where the user mailbox dies
   mid-submission with outstanding work holding the rq lock.  For the
   debug_halt_probe failure mode (TDR runs to "completion"
   bookkeeping-wise and releases the user mailbox cleanly, but FW and
   MGMT mailbox are dead), `modprobe -r` completed in ~10s without
   hanging.  modprobe re-probe still fails -- the dmesg `smu cmd 4
   failed, 0xff` is `SMU_RESP_CMD_FAIL` from the PPSMC convention,
   meaning the SMU itself is alive and responding but cannot complete
   POWER_OFF on an NPU whose FW-managed internal state machines are
   hung.  So reboot is still the only recovery, but at least the
   reload attempt itself is safe to try.  Also: **`fw_reload` is
   AIE4-only** (`aie4_pci.c:44`); on Phoenix the module param is
   accepted but does nothing.  Don't include it in the Phoenix
   recovery escalation chain.

   The SMU response codes follow PPSMC convention (see also
   amdgpu's `drivers/gpu/drm/amd/pm/swsmu/smu_cmn.c`): `0x01`=OK,
   `0xFC`=busy, `0xFD`=bad prereq, `0xFE`=unknown command,
   `0xFF`=generic fail, `0x00`=no response (SMU dead).  Always read
   the raw `0x%x` from the dmesg line; the kernel-return `-EINVAL`
   from `aie_smu_exec` collapses all non-OK codes to one value.

2. **Bridge PM-cycle** -- reset the upstream bridge function:
   ```bash
   pkexec modprobe -r amdxdna
   pkexec sh -c 'echo 1 > /sys/bus/pci/devices/0000:00:08.2/reset'
   sleep 5
   pkexec modprobe amdxdna
   ```
   Note: this is what the kernel calls "reset" on the bridge; per the
   bridge's `reset_method = pm`, it's a D0->D3hot->D0 cycle of the bridge
   function, NOT a true Secondary Bus Reset on bus c6.

3. **True Secondary Bus Reset** -- toggle BCR.SBR on the bridge:
   ```bash
   pkexec modprobe -r amdxdna
   pkexec setpci -s 00:08.2 BRIDGE_CONTROL=0x42  # assert SBR
   sleep 0.1
   pkexec setpci -s 00:08.2 BRIDGE_CONTROL=0x02  # deassert
   pkexec sh -c 'echo 1 > /sys/bus/pci/rescan'
   pkexec modprobe amdxdna
   ```
   This actually pulses PERST# on bus c6 (resets c6:00.0 + c6:00.1).
   Useful when step 2 doesn't suffice. Confirmed working at the PCIe
   layer (BAR enable cycles), but does NOT recover SMU/MGMT_ERT-level
   wedges -- the on-NPU controllers live downstream of the PCIe reset
   domain.

4. **Suspend/resume** -- `systemctl suspend` drops the SoC to retention
   voltage, clearing on-NPU controller state. Required for SMU wedges
   from bring-up failures.

5. **Reboot** -- last resort.

FLR on the NPU function is advertised in DevCap (`FLReset+`) and listed
as the kernel's `reset_method = flr` for c6:00.1, but per upstream
xdna-driver feedback (May 2026) it's unreliable and unsupported in
practice. SBR on the upstream bridge is the working PCIe-layer path.
Always remove `amdxdna` before any PCIe reset -- the driver holds state
that corrupts the device on hot-reset. PCIe BDFs shift when hardware
changes (e.g., GPU swaps renumber the bus); `lspci | grep "IPU Device"`
finds the current BDF.

### Working-directory conventions

**Never put tools, scripts, or persistent work products in `/tmp`.**
This PC cannot suspend and reboots often, which wipes `/tmp`. Tools
live under `xdna-emu/tools/` or `xdna-emu/scripts/`. Experiment results
go under `xdna-emu/build/experiments/` or `~/npu-work/experiments/`.
Only truly ephemeral data (sandbox temp dirs for `cargo test`, log
tees for a specific run) should use `/tmp/claude-1000/`.

### Developer environment state

These describe the current machine's setup. Other contributors will
substitute their own values.

- **Kernel**: custom `7.0.6-custom+`. Out-of-tree `amdxdna` is
  managed by DKMS via `xrt-amdxdna/2.23.0`, source at
  `/usr/src/xrt-amdxdna-2.23.0/`.  Userspace plugin (the SHIM at
  `/opt/xilinx/xrt/lib/libxrt_driver_xdna.so.*`) is delivered by the
  `xrt_plugin-amdxdna` .deb.  A bare `./build.sh -release` BUILDS
  both halves but INSTALLS neither; the `.deb` sits at
  `build/Release/xrt_plugin.*-amdxdna.deb` until `dpkg -i`'d.  Module
  is signed at install time with our MOK key
  (`/var/lib/shim-signed/mok/MOK.{priv,der}`), so `modprobe amdxdna`
  works after every kernel upgrade with no manual signing.  After
  editing driver or SHIM source, refresh both halves with:
  ```bash
  cd ~/npu-work/xdna-driver/build
  ./build.sh -release -refresh_dkms
  ```
  `-refresh_dkms` is our local addition (commit 3509b2a, simplified
  2026-05-12).  It `pkexec dpkg -i`s the OS-matched
  `build/Release/xrt_plugin.*_${VERSION_ID}-*.deb`; the .deb postinst
  does the rest -- runs `dkms_driver.sh --install` to populate
  `/usr/src/xrt-amdxdna-2.23.0/` from a single canonical source
  (script + tarball + config, NOT a piecemeal rsync that would miss
  `configure_kernel.sh` and produce feature-probe mismatches like
  `num_rqs == 0`), then `rmmod amdxdna && modprobe amdxdna` to swap
  the loaded module.  Single auth prompt, both halves of the install
  on disk, kernel module reloaded.  Stale SHIM bytes silently mask
  source changes -- always pass `-refresh_dkms` after touching
  driver OR SHIM code.

  Caveat: the .deb postinst uses `rmmod` which fails (and the install
  aborts) if the device is busy or the module is wedged.  In that case,
  free the device first (kill any process holding `/dev/accel/accel0`),
  or fall back to a manual `pkexec sh -c 'modprobe -r amdxdna; dpkg -i
  build/Release/xrt_plugin.*_${VERSION_ID}-*.deb'`.
- **amdxdna module options pinned**: `/etc/modprobe.d/amdxdna.conf`
  contains `options amdxdna autosuspend_ms=-1`.
  - `autosuspend_ms=-1`: prevent runtime autosuspend. The NPU has
    been observed to wedge on auto-resume after certain mailbox
    failures (e.g. struct-size mismatches that leave firmware in
    an unrecoverable state). Pinning autosuspend off keeps the
    device alive so we never hit the broken resume path.

  Workaround for development; revert once the underlying wedges
  are diagnosed and fixed. Verify with
  `cat /sys/module/amdxdna/parameters/autosuspend_ms`
  (should report `-1`).

  **Removed 2026-05-13: `tdr_dump_ctx=1`.** Originally set so TDR
  was dump-only (no `aie2_rq_stop_all/restart_all`), avoiding the
  `synchronize_srcu` wedge in `modprobe -r` after the recover path
  poisoned the mailbox on Phoenix. But disabling TDR recovery made
  every firmware silent-drop (e.g. CHAIN_EXEC_NPU on ctrl_packet
  sweeps) into a permanent `aie2_hmm_invalidate` wedge -- the
  driver-design comment at `aie2_ctx.c:1017` explicitly relies on
  TDR to terminate the ctx if firmware doesn't respond, so without
  recovery the `dma_resv_wait_timeout(MAX_SCHEDULE_TIMEOUT)` in
  hmm_invalidate waits forever. We were rebooting from those
  wedges anyway, so the original trade is gone -- letting TDR
  recover trades a permanent wedge for a possible (not certain)
  modprobe -r wedge. See
  `docs/superpowers/findings/2026-05-13-chain-exec-npu-silent-drop-on-phoenix.md`.
- **dmesg is unrestricted**: kernel built with `kernel.dmesg_restrict=0`
  (or equivalent), so `dmesg` works without `pkexec`. Don't wrap dmesg
  in pkexec on this machine.
- **Chess license**: `HOSTID=f4289d05121f` (bound to current Wi-Fi
  card; 2 of 3 vendor-permitted swaps remaining).
- **DNS**: UConn DNS is broken. Fix per-session:
  `resolvectl dns wlp5s0 8.8.8.8 8.8.4.4` (does not persist across
  reboot).
- **mlir-aie venv**: `/home/triple/npu-work/mlir-aie/ironenv/`
- **PYTHONPATH**: `/home/triple/npu-work/mlir-aie/install/python`
- **XRT plugin**: `./scripts/rebuild-plugin.sh` builds and installs the
  debug `.so` by default (`--release` for release). Activation:
  `XDNA_EMU=1` (any value) routes XRT to the emulator at `xrt::device(0)`.
  Profile: `XDNA_EMU_RUNTIME=release|debug` (default `debug`); the plugin
  picks the matching `.so` via `XDNA_EMU_DIR` or installed symlinks.
- **Trace column offset**: emulator col=0 vs HW col=start_col (cosmetic;
  trace tools should normalize).
