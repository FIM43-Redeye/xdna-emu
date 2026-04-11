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
| [1. Core Accuracy](docs/roadmap/phase1-core-accuracy.md) | **100% ISA** | 4815/4815 ISA test points; 2660+ unit tests; bridge tests pass |
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
  `aietools/data/aie_ml/lib/python_model/model/` describe these semantics
  (mulmac.py, srs_ups.py, permute.py, constants.py). Read them to understand
  the hardware behavior, then implement independently.
- **Stream switch per-port type assignments**: mlir-aie gives port counts per
  bundle, not which index maps to which bundle type. Currently hardcoded from
  AM025 in `aie2_spec.rs`.
- **Micro-timing details**: DMA pipeline depth, NoC latency, memory bank
  conflict resolution. These affect cycle-accuracy but not functional
  correctness.

**Documentation path**: `xdna-emu/docs/xdna/` (extracted text files, not PDFs)

**aietools reference paths** (read-only, never copy):
- Vector semantics: `aietools/data/aie_ml/lib/python_model/model/`
- Register IDs: `aietools/data/aie_ml/lib/isg/me_regid.txt` (Synopsys copyright)
- AIE API headers: `aietools/include/aie_api/` (MIT licensed)
- Event types: `aietools/data/eventanalyze/event_type_table.txt`

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
| TableGen | [`.claude/components/tablegen.md`](.claude/components/tablegen.md) | Working on ISA definitions, decoder tables, llvm-aie integration (`src/tablegen/`) |
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
- **aietools**: `../aietools` - AMD proprietary tools (Chess compiler, aiesimulator, analysis tools). Read-only reference for hardware semantics not covered by open-source toolchain. See Licensing section.
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
`--serial-hw`, `--trace=sweep`, `-v <filter>`.

**Build dirs**: `mlir-aie/build/test/npu-xrt/$name/chess/` and `peano/`
**Results**: `build/bridge-test-results/YYYYMMDD/` (symlink at `build/bridge-test-results/latest`)
**Plugin**: `pkexec cp xrt-plugin/build/libxrt_driver_emu.so.2 /opt/xilinx/xrt/lib/`

### Backup: npu-test (standalone harness)

`npu-test` bypasses XRT entirely -- useful for isolated subsystem testing.
**NEVER run inside the Claude Code sandbox** (license checks, filesystem issues).

## Tracing Ecosystem

Binary trace comparison between emulator and real NPU hardware. All traces
converge to Perfetto JSON (viewable at ui.perfetto.dev).

| Tool | Purpose |
|------|---------|
| `tools/trace-inject.py` | Inject trace routing into MLIR (capacity planner, collision-aware IDs) |
| `tools/trace-sweep.py` | Multi-batch event sweep orchestrator (pre-compiled artifacts, HW serial + EMU parallel) |
| `tools/trace-trim.py` | Strip sentinel padding from raw trace buffers |
| `tools/trace-merge.py` | Merge per-batch Perfetto JSON with TRUE anchor alignment |
| `tools/trace-patch-events.py` | Patch event slots in compiled insts.bin without recompilation |
| `src/bin/trace_compare.rs` | **Rust** binary trace comparison (replaced Python -- 65GB OOM -> 11MB) |

Build: `cargo build --release --bin trace-compare`

## XRT Plugin (`xrt-plugin/`)

Driver plugin replacing the real XDNA kernel driver. XRT loads the .so,
which delegates to the Rust emulator via FFI (`src/ffi/`).

**Build**: `./scripts/rebuild-plugin.sh`
**Install**: `pkexec cp xrt-plugin/build/libxrt_driver_emu.so.2 /opt/xilinx/xrt/lib/`
**Activation**: Set `XDNA_EMU=1` before running test.exe.

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
XDNA_EMU=debug XDNA_EMU_WATCH=0xC000:40,0x428:40,0x400:40 RUST_LOG=info \
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
