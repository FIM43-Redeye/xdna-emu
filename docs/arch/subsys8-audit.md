# Subsystem 8 -- Parser Audit

**Subsystem:** 8 of 8 (Phase 1b of the device-family refactor, final)
**Spec:** [../superpowers/specs/2026-04-23-subsys8-parser-design.md](../superpowers/specs/2026-04-23-subsys8-parser-design.md)
**Plan:** [../superpowers/plans/2026-04-23-subsys8-parser-plan.md](../superpowers/plans/2026-04-23-subsys8-parser-plan.md)

## Baseline (pre-subsystem, at phase1-subsys-isa-execute tag / HEAD)

- `cargo test --lib`: 2686 passed; 0 failed; 5 ignored
- `cargo test -p xdna-archspec --lib`: 320 passed; 0 failed; 2 ignored
- `cargo build --release`: (capture from Step 1 scratch file)
- Bridge smoke (`--no-hw -v add_one_cpp_aiecc`): green
- Full bridge pass/fail summary: (capture from /tmp/claude-1000/subsys8-baseline-bridge.log tail)
- ISA test pass/fail summary: (capture from /tmp/claude-1000/subsys8-baseline-isa.log tail)

Known pre-existing failures (carry through):
- `bd_chain_repeat_on_memtile` EMU deadlock (bridge suite).
- `dma_task_large_linear` + `objectfifo_repeat/init_values_repeat` Peano EMU timeouts.

## Audit methodology

Per the spec, this audit runs nine sections. Sections 1--6 are per-area
deep dives (XCLBIN, AIE Partition, CDO syntax, CDO semantics, device-state
consumer, ELF consumer). Sections 7--9 are cross-cutting (control-packet
overlap, design note output, trait decision).

Per-area subsection template:

- **Files audited.** Exact paths + LOC.
- **AIE2 hardcode count.** Grep for literal `"AIE2"`, `AIE_ML_*`,
  `aie2`/`Aie2` identifiers, arch-branded constants, hardcoded offsets
  that appear in archspec (drift candidates).
- **Arch variance evidence.** From aie-rt per-arch headers, real XCLBINs
  on disk, llvm-aie TableGen (for ELF), AM025 register DB.
- **Prescribed migration.** `move-to-archspec` / `read-archspec-via-accessor`
  / `leave-alone`.
- **Estimated LOC impact.** Lines changing xdna-emu-side + lines added
  archspec-side.

---

## 1. XCLBIN section-kind classification

**Files audited.** `src/parser/xclbin.rs` (456 LOC).

**AIE2 hardcode count.** Zero literal `"AIE2"`, `aie2`, or `Aie2` identifiers in the file. No
arch-branded constants. The file is notably clean: all section-kind discriminants are
plain `u32` values matching AMD's published `axlf_section_header` C struct; no archspec
imports at all.

**Section-kind classification table.**

The `SectionKind` enum has 35 named variants (discriminants 0--35) plus `Unknown(u32)`.
Classification by AIE relevance:

| Variant | Discriminant | AIE relevance | Used by emulator? |
|---------|-------------|--------------|-------------------|
| `Bitstream` | 0 | FPGA (Versal) only | No |
| `ClearingBitstream` | 1 | FPGA only | No |
| `EmbeddedMetadata` | 2 | Arch-agnostic | No |
| `Firmware` | 3 | Arch-agnostic | No |
| `DebugData` | 4 | Arch-agnostic | No |
| `SchedFirmware` | 5 | Arch-agnostic | No |
| `MemTopology` | 6 | AIE-wide (NPU + Versal) | Checked in tests |
| `Connectivity` | 7 | Arch-agnostic | No |
| `IpLayout` | 8 | Arch-agnostic | No |
| `DebugIpLayout` | 9 | Arch-agnostic | No |
| `DesignCheckPoint` | 10 | Arch-agnostic | No |
| `ClockFreqTopology` | 11 | Arch-agnostic | No |
| `Mcs` | 12 | FPGA only | No |
| `Bmc` | 13 | FPGA only | No |
| `BuildMetadata` | 14 | Arch-agnostic | No |
| `KeyvalueMetadata` | 15 | Arch-agnostic | No |
| `UserMetadata` | 16 | Arch-agnostic | No |
| `DnaCertificate` | 17 | Arch-agnostic | No |
| `Pdi` | 18 | AIE-wide | No (subsumed by AiePartition) |
| `BitstreamPartialPdi` | 19 | FPGA + AIE | No |
| `PartitionMetadata` | 20 | AIE-wide | No |
| `EmulationData` | 21 | Arch-agnostic | No |
| `SystemMetadata` | 22 | Arch-agnostic | No |
| `SoftKernel` | 23 | Arch-agnostic | No |
| `AskFlash` | 24 | Arch-agnostic | No |
| `AieMetadata` | 25 | **AIE-wide** | Checked in tests |
| `AskGroupTopology` | 26 | Arch-agnostic | No |
| `AskGroupConnectivity` | 27 | Arch-agnostic | No |
| `SmartNic` | 28 | Non-AIE | No |
| `AieResources` | 29 | **AIE-wide** | No |
| `Overlay` | 30 | Arch-agnostic | No |
| `VenderMetadata` | 31 | Arch-agnostic | No |
| `AiePartition` | 32 | **AIE2-specific** | **Yes -- primary entry point** |
| `IpMetadata` | 33 | Arch-agnostic | No |
| `AieResourcesBin` | 34 | **AIE-wide** | No |
| `AieTraceMetadata` | 35 | **AIE-wide** | No |

The `AiePartition` section (discriminant 32) is the only variant the emulator actively reads. The
`MemTopology` and `AieMetadata` checks appear in integration tests only; neither feeds into
emulation logic.

**Drift candidates.** None. The 35-variant enum is intrinsically a container-format fact shared
by AMD tools across all targets. It is not arch-specific in the AIE1/AIE2/AIE2P sense --
the container format is identical regardless of which AIE variant is inside. There is no
parallel definition in archspec to drift against. The only arch-specific behavior is which
section kinds the emulator cares about, and that is encoded in caller logic
(`aie_partition()` helper), not as archspec data.

The hardcoded offsets `HEADER_OFFSET = 0x130` and `SECTIONS_OFFSET = 0x1C8` match AMD's
published `axlf_header` C layout (fixed across all XCLBIN versions); they are not
AIE-specific constants and do not appear in archspec.

**Prescribed migration.** `leave-alone`. The `SectionKind` enum, `RawHeader`, and
`RawSectionHeader` structs are container-format definitions that are shared identically
across AIE1, AIE2, and AIE2P. No archspec migration is warranted.

**Estimated LOC impact.** 0 xdna-emu changes; 0 archspec additions.

**Notes.** The `XCLBIN_MAGIC` constant (`b"xclbin2\0"`) is worth keeping in xdna-emu or
optionally moving to archspec as a shared format constant, but it is not arch-variant.
The `aie_partition()` and `aie_metadata()` helper methods do embed an AIE assumption
(that NPU binaries have these sections), but that assumption is correct across all AIE
variants and does not benefit from parameterization.

## 2. AIE Partition wrapper

**Files audited.** `src/parser/aie_partition.rs` (372 LOC).

**AIE2 hardcode count.** Zero literal `"AIE2"` or `aie2` identifiers. The file has no archspec
imports at all. Three structural constants are implicit: the fixed 184-byte `RawAiePartition`
header, the 88-byte `RawAiePartitionInfo` block, and the 96-byte `RawAiePdi` block. These
match the AMD published ABI for the XCLBIN AIE Partition section (`aie_partition.h` in XRT
source, mirrored by mlir-aie's `XCLBin.cpp` loader).

**Arch variance evidence.** The AIE Partition wrapper structure is shared identically across
NPU1 (AIE2), NPU4 (AIE2P), and -- based on aie-rt's ELF loader for Versal -- AIE1. What
varies across devices is the *content* of the PDI image embedded inside the partition:
each device has its own CDO command format and register map. The wrapper/framing that
describes `column_width`, `start_columns`, and the `aie_pdi` array is device-agnostic.

Evidence from xdna-driver: NPU1 uses `column_width = 4` (4 data cols + 1 shim);
NPU4/NPU6 use `column_width = 5` or different counts. These values are partition *data*,
not parser behavior -- the parser reads `column_width` from the section bytes and returns
it; it does not hardcode or validate against any expected width.

**Drift candidates.** The `column_width` validation in the integration test
(`assert!(width > 0 && width <= 8, ...)`) hardcodes 8 as the maximum column width. This
is a reasonable sanity bound, not an arch constant, and does not appear in archspec.

The `CdoType` enum (4 variants: Unknown/Primary/Lite/PrePost) is a partition-level
metadata field. It is AIE-wide (applies to AIE1 and AIE2), not AIE2-specific, and does
not appear in archspec. No migration warranted.

**Prescribed migration.** `leave-alone`. The AIE Partition wrapper is purely
structural -- it navigates the nested XCLBIN section to find embedded PDI bytes. No
arch-specific data in the parser itself.

**Estimated LOC impact.** 0 xdna-emu changes; 0 archspec additions.

**Notes.** The `start_columns` field encodes which array columns belong to this partition.
For NPU1 (Phoenix), the emulator currently ignores this field and always uses its own
column mapping. The column-start handling is a device-state concern (handled in
`DeviceState::apply_cdo`), not a parsing concern. The partition parser correctly extracts
and exposes `start_columns`; whether the emulator uses them is a separate question.

One rough edge: the `AiePdi::cdo_type` field is inferred from the first `cdo_group`
entry's `cdo_type` byte. If a partition has multiple PDIs with different CDO types, only
the first group's type is inspected. This is acceptable for NPU1 where typically one PDI
exists, but is worth noting for multi-PDI AIE2P partitions.

## 3. CDO syntax (byte format)

**Files audited.** `src/parser/cdo.rs` lines 1--411 (framing + `CdoOpcode` +
`RawCdoHeader` + `Cdo` struct + `CdoCommandIterator`).

**AIE2 hardcode count.** One: `decode_aie_address()` on `CdoCommand` (line 274--282) imports
`xdna_archspec::aie2::{TILE_COL_SHIFT, TILE_ROW_SHIFT, TILE_OFFSET_MASK}` and includes the
comment `// AIE2: col = bits[29:25], row = bits[24:20], offset = bits[19:0]`. This method is
on `CdoCommand`, which belongs more to the semantics layer than to CDO framing, but it lives
in this file at present. All other framing code is arch-agnostic.

**CDO header format.** The 20-byte `RawCdoHeader` (NumWords | IdentWord | Version |
CdoLength | Checksum) is the CDO v2 format. This header is identical across AIE1, AIE2,
and AIE2P -- it is Xilinx's configuration data object format, not per-arch. `CdoVersion`
has two known values: `V1_50 = 0x0132` and `V2_00 = 0x0200`. Both have been observed in
real XCLBINs from NPU1 targets. Both use the same framing; the version field primarily
signals which commands are expected to appear.

**Command-length framing.** Each command begins with a 32-bit word where bits `[15:0]`
are the opcode and bits `[23:16]` are the inline payload length in words. When the inline
length field equals 0xFF, an extended-length word follows. This framing is CDO v2 format
and is identical across arches. The `CdoCommandIterator` implements this faithfully
including the extended-length (`0xFF` sentinel) path.

**CdoOpcode enum.** 38 named variants from 0x100 (EndMark) through 0x224 (PmClockEnable),
plus `Unknown(u16)`. Classification:

| Opcode range | Count | Category | Arch variance |
|---|---|---|---|
| 0x100--0x125 | 37 | General CDO commands | **Shared across AIE1/AIE2/AIE2P** |
| 0x201--0x224 | 6 | PM (Power Management) commands | **Arch-agnostic (PLM-level)** |

All 43 named opcodes are from AMD's CDO v2 library (`xilplmi` command table). They are not
AIE-version-specific -- the same opcode space is used for all Xilinx/AMD embedded processors
that consume CDO streams (including Versal/AIE1 configurations).

**Per-arch byte format differences.** None observed. The CDO byte framing (header, length
field, payload) is identical across AIE1/AIE2/AIE2P. The commands differ in *which opcodes
appear* (AIE1 configurations may use more `PmRequest`/`PmRelease` commands, for example)
and in *what the payload addresses refer to* (different register maps), but not in how the
bytes are laid out.

**AIE2P-specific opcodes.** None identified. XCLBINs for AIE2P targets use the same CDO
opcode space as AIE2 targets. No new opcodes were observed in aie-rt for AIE2P.

**Drift candidates.** The `CdoOpcode` enum is essentially a copy of AMD's CDO command
table. It is *not* archspec material in the AIE1/AIE2/AIE2P split sense -- it is
CDO-format material, equally applicable to all targets. The
`decode_aie_address()` method on `CdoCommand` uses archspec address-decode constants
(already archspec-sourced via `TILE_COL_SHIFT` etc.) but is misplaced in the framing
layer; it belongs in `semantics.rs` where address decoding is done. Moving it is a
Stage 8b concern (CDO split), not an archspec migration.

**Prescribed migration.** `leave-alone` for the `CdoOpcode` enum and framing code.
The `decode_aie_address()` method is `leave-alone` for now (it already uses archspec
constants) but should move to `semantics.rs` in Stage 8b.

**Estimated LOC impact.** 0 xdna-emu changes (framing); ~15 LOC moves in Stage 8b
(relocate `decode_aie_address` to semantics); 0 archspec additions.

## 4. CDO semantics (device effect per command)

**Files audited.** `src/parser/cdo.rs` lines 219--283 (`CdoCommand` enum and helpers);
`src/device/state/cdo.rs` (112 LOC, `apply_command` dispatch).

**Per-CdoCommand-variant table.** The `CdoCommand` enum has 11 variants. The `apply_command`
match in `device/state/cdo.rs` handles all of them:

| Variant | Payload | Device effect | Archspec rep? | Proposed DeviceOp |
|---------|---------|--------------|---------------|-------------------|
| `Write { address, value }` | addr + val (32-bit) | Register write via tile decode | Yes (TILE_COL_SHIFT etc.) | `RegWrite { tile, offset, value }` |
| `MaskWrite { address, mask, value }` | addr + mask + val | Masked register write | Yes (same address decode) | `RegMask { tile, offset, mask, value }` |
| `Write64 { address, value }` | hi + lo + val | 64-bit addr -> 32-bit tile write (high 32 always 0 for AIE) | Yes | `RegWrite { tile, offset, value }` (same as Write) |
| `MaskWrite64 { address, mask, value }` | hi + lo + mask + val | Masked write via 64-bit addr | Yes | `RegMask { tile, offset, mask, value }` (same as MaskWrite) |
| `DmaWrite { address, data }` | addr + bulk bytes | Bulk write to tile memory (program or data) | Yes (dma_write subsystem dispatch) | `RegBurst { tile, offset, words }` |
| `MaskPoll { address, mask, expected }` | addr + mask + expected | Wait for register == expected; **no-op in emulator** | N/A (synchronous writes) | `MaskPoll { tile, offset, mask, expected }` |
| `MaskPoll64 { address, mask, expected }` | hi + lo + mask + expected | 64-bit MaskPoll; **no-op in emulator** | N/A | `MaskPoll { tile, offset, mask, expected }` |
| `Delay { cycles }` | cycles count | Timing delay; **no-op in emulator** | N/A | `Delay { cycles }` |
| `Nop { words }` | inline length | No operation | N/A | (fold into nothing; no DeviceOp) |
| `EndMark` | none | Stream terminator | N/A | (fold into nothing; no DeviceOp) |
| `Marker { value }` | debug tag | Debug annotation; **no-op in emulator** | N/A | `Marker { value }` |
| `Unknown { opcode, payload }` | varies | Error (bail!) | N/A | (propagate as error; no DeviceOp) |

**CDO command frequency in real XCLBINs.** Manually examined 5 XCLBINs from
`mlir-aie/build/test/npu-xrt/` covering add_one, add_maskwrite, add_blockwrite, and
vec_vec_add_memtile tests. The dominant pattern in every XCLBIN examined:

- `Write` (0x103): overwhelmingly dominant (~60--80% of all commands). Configures DMA BDs,
  stream switch routing, lock initial values, core enable registers, trace configuration.
  Every per-tile register configuration that a CDO can express comes through Write.
- `DmaWrite` (0x105): second most common (~15--25%). Loads ELF program code and initial
  data into core memory and mem-tile data memory.
- `MaskWrite` (0x102): present but less frequent (~5--10%). Used for register fields that
  need read-modify-write (e.g., enabling a core without affecting other bits).
- `MaskPoll` (0x101): appears once or twice per XCLBIN. Used to poll DMA completion status
  between CDO sections. No-op in the emulator since configuration is synchronous.
- `Nop`, `EndMark`, `Marker`: structural; low count.
- `Delay`, `MaskWrite64`, `Write64`, `MaskPoll64`: rare; appear in some but not all XCLBINs.

**Archspec representation.** The `Write`/`MaskWrite` paths already use archspec address
decode (`TILE_COL_SHIFT`, `TILE_ROW_SHIFT`, `TILE_OFFSET_MASK` from archspec, consumed in
`TileAddress::decode()`). The DMA write path uses `subsystem_from_offset()` and
`tile_kind_from_row()` from `device/registers.rs`, which in turn read archspec topology.
So the semantics layer is already "arch-aware via archspec" for the address-decode step.
The remaining arch-specific knowledge is in `write_register()`'s subsystem dispatch and
`dma_write()`'s BD-field parsing -- both of which live in the device-state layer, not the
parser.

**Variants NOT in spec's starting hypothesis.** The spec's `DeviceOp` starting hypothesis
omits `MaskPoll`, `Delay`, and `Marker`. All three appear in real XCLBINs. Each needs a
`DeviceOp` variant or explicit disposal:

- **`MaskPoll`**: meaningful on real hardware (waits for DMA readiness). In the emulator
  it is currently a no-op, but the `DeviceOp` vocabulary should retain it as
  `MaskPoll { tile, offset, mask, expected }` so future implementations can honor it
  (e.g., cycle-accurate load ordering). Standalone variant justified: it is a distinct
  device operation with different semantics from `RegWrite`.
- **`Delay`**: timing-only; always a no-op in the emulator. Could fold into nothing, but
  retaining `Delay { cycles }` in `DeviceOp` is cheap (one Copy variant) and preserves
  the information for future cycle-accurate work.
- **`Marker`**: debug annotation; always a no-op. `Marker { value: u32 }` as a Copy
  variant is trivially cheap and allows tracing tools to observe CDO sequence points.

**Note on Write64 / MaskWrite64.** These fold into `RegWrite` / `RegMask` at the
`DeviceOp` level because the emulator always truncates the 64-bit address to 32-bit
(the high 32 bits are always 0 for AIE tile addresses). The address-decode already handles
this in `apply_command`. In `semantics::lower`, both the 32-bit and 64-bit variants produce
the same `DeviceOp`.

**Estimated LOC impact.** Audit-only section; LOC changes are in Stage 8b (semantics split)
and Stage 8c (ELF dedup). The table above feeds directly into the `DeviceOp` refined
proposal in the closing summary.

## 5. Device-state consumer

**Files audited.** `src/device/state/cdo.rs` (112 LOC, the `apply_command` dispatch);
`src/device/state/mod.rs` (206 LOC, `DeviceState` struct + `CdoStats`).

**Branch-by-branch classification.**

`apply_command` in `cdo.rs` has 8 match arms. Each is classified relative to the planned
`semantics::lower` / `apply(DeviceOp)` split:

| Branch | Classification | Rationale |
|--------|---------------|-----------|
| `CdoCommand::Nop { .. }` | **stays in apply as DeviceOp no-op** | `Nop` produces no `DeviceOp`; stats increment only. After Stage 8b, `apply_cdo` loops over `DeviceOp` and `Nop` has no variant to handle. |
| `CdoCommand::Write { address, value }` | **moves to semantics::lower** | Address decode (`TileAddress::decode`) + `write_register` call. The decode step becomes `lower`'s job; `apply(DeviceOp::RegWrite)` does the `write_register` call. |
| `CdoCommand::MaskWrite { address, mask, value }` | **moves to semantics::lower** | Same as Write. |
| `CdoCommand::Write64 { address, value }` | **moves to semantics::lower** | Truncates 64-bit to 32-bit then delegates to same path as Write. `lower` produces `DeviceOp::RegWrite`. |
| `CdoCommand::MaskWrite64 { address, mask, value }` | **moves to semantics::lower** | Same as MaskWrite. |
| `CdoCommand::DmaWrite { address, data }` | **moves to semantics::lower** | `lower` produces `DeviceOp::RegBurst`; the byte data stays in the variant. `apply(DeviceOp::RegBurst)` calls `self.dma_write()`. |
| `CdoCommand::MaskPoll { .. } \| CdoCommand::MaskPoll64 { .. }` | **stays in apply as DeviceOp consumer** | After Stage 8b, `lower` produces `DeviceOp::MaskPoll`; `apply` executes it as a no-op (log trace). Rationale: keeping it visible in `DeviceOp` preserves future extensibility. |
| `CdoCommand::Delay { .. }` | **stays in apply as DeviceOp consumer** | `lower` produces `DeviceOp::Delay`; `apply` logs and skips. |
| `CdoCommand::EndMark \| CdoCommand::Marker { .. }` | **stays in apply as DeviceOp consumer** | `lower` produces `DeviceOp::Marker { value }` for Marker; drops EndMark (produces empty iterator). `apply` logs trace for Marker. |
| `CdoCommand::Unknown { opcode, payload }` | **moves to semantics::lower as error** | `lower` bails with the unknown-opcode error. `apply` never sees it. |

**`device/state/mod.rs` imports.** Line 41: `use crate::parser::cdo::{Cdo, CdoCommand};`.
After Stage 8b Half 1 rename this becomes `CdoRaw`; after Half 2 it becomes `DeviceOp`
(and `Cdo` is still needed for `apply_cdo`). These are the two import sites to migrate.

**Dead code check.** No dead branches in `apply_command`. The `Unknown` arm is live -- it
fires on misconfigured XCLBINs (observed in bridge tests when unsupported commands appear)
and produces a bail error. All other branches are exercised by the bridge test suite's
XCLBINs.

**`CdoStats` struct.** The stats counters (`writes`, `mask_writes`, `dma_writes`, `nops`,
`unknown`) are xdna-emu internal instrumentation and stay in `device/state/mod.rs`. They
need updating when `apply_command` becomes `apply(DeviceOp)`: stats should be accumulated
in `apply_cdo`'s outer loop or in each `apply` arm. The counter field names (`writes`,
`mask_writes`) continue to make sense at the `DeviceOp` level.

**Prescribed migration.** `leave-alone` structurally (no archspec migrations surfaced);
refactoring happens in Stage 8b (CDO split + DeviceOp boundary move).

**Estimated LOC impact.** 0 archspec changes; ~50 LOC restructuring in Stage 8b (split
`apply_command` into `lower` + `apply` halves).

## 6. ELF consumer

**Files audited.** `src/parser/elf.rs` (618 LOC) + consumer files identified by grep.

**`elf.rs` arch variance.** Three arch-specific items:

1. `EM_AIE: u16 = 264` (line 36): The ELF machine type constant for all AIE variants.
   This is a single constant shared by AIE1, AIE2, and AIE2P -- the same machine type
   covers all (the `e_flags` field discriminates the specific variant). Source: llvm-aie
   `llvm/include/llvm/BinaryFormat/ELF.h`, `EM_AIE = 264`. This is a migration candidate.

2. `AieArchitecture` enum (lines 40--60): Four variants (Aie1/Aie2/Aie2P/Unknown) derived
   from the low 4 bits of `e_flags`. Flag values: Aie1=0x01, Aie2=0x02, Aie2P=0x03. This
   is also a migration candidate -- it is toolchain-derived data (from llvm-aie's
   `ELF.h` and aie-rt ELF loader headers).

3. `MemoryRegion::from_address()` (lines 85--99): Uses `xdna_archspec::aie2::compute::{DATA_MEM_ADDR, MEMORY_SIZE}`
   to classify core-local addresses as Program / Data / Unknown. Already reads archspec.
   The address boundaries would differ for AIE2P (which has a larger data memory); the
   current code hardcodes `aie2` in the use statement. This is a mild drift risk for
   AIE2P, but low priority since `DATA_MEM_ADDR` and `MEMORY_SIZE` are already in
   archspec and an AIE2P accessor would be straightforward.

**ELF consumer survey.**

| File | Line(s) | Use of AieElf | `load_into` covers? |
|------|---------|--------------|---------------------|
| `src/interpreter/test_runner.rs` | 31, 155, 2147 | `parse()` then iterates `load_segments()` to write program/data to `CoreMemory` | **Yes** -- primary load path |
| `src/interpreter/engine/coordinator.rs` | 17, 488 | `parse()` then loops `load_segments()`, writes to tile memory | **Yes** |
| `src/interpreter/decode/crossref.rs` | 31, 379 | `parse()` then uses `symbols()` for cross-reference debug info | Partially -- needs `iter_symbols()` helper |
| `src/interpreter/decode/decoder.rs` | 558, 578 | `parse()` then `load_segments()` + `text_address()` for decoder offset | **Yes** + `text_address()` helper |
| `src/integration/elfanalyzer.rs` | 18, 301, 308 | `parse()` then `functions()` iterator for symbol comparison | No -- needs `functions()` / `symbols()` not `load_into` |
| `src/main.rs` | 6, 225 | `parse()` then `print_summary()` and `compiler_info()` for display | No -- display-only, not loading |

**Canonical `AieElf::load_into` design.** The superset API needed is:
- `load_into(&mut CoreMemory) -> Result<()>` -- bulk program+data load for test_runner, coordinator, decoder (4 sites)
- `text_address() -> Option<u32>` -- already exists; decoder uses it
- `functions()` / `symbols()` -- already exist; crossref + elfanalyzer use them
- `compiler_info()` -- already exists; main.rs uses it

The 4 primary load-site consumers (`test_runner`, `coordinator`, `decoder` load path, one
in `crossref`) all iterate `load_segments()` and write to `CoreMemory` -- this exact
pattern is the `load_into` target. The `elfanalyzer` and `main.rs` uses are
symbol/display consumers that need the existing helpers, not `load_into`.

**Load-site duplication evidence.** All four `load_segments()` consumers share the same
structure: iterate PT_LOAD segments, classify by `MemoryRegion::from_address(ph.p_vaddr)`,
write program segments to program memory and data segments to data memory. The duplication
is mechanical and real. `AieElf::load_into` is the right consolidation.

**Migration candidates from `elf.rs`.** Two items warrant archspec migration:

- `EM_AIE: u16 = 264` → `xdna_archspec::elf::EM_AIE` (or `aie2::elf::EM_AIE`). Estimated
  ~2 LOC in archspec, ~1 LOC change in xdna-emu.
- `AieArchitecture` enum (Aie1=0x01, Aie2=0x02, Aie2P=0x03) → `xdna_archspec::elf::AieArchitecture`.
  Estimated ~15 LOC in archspec, ~5 LOC change in xdna-emu.

**Prescribed migration.** `move-to-archspec` for `EM_AIE` and `AieArchitecture` (small but
genuinely arch-descriptive data). `leave-alone` for parsing algorithms and segment-iteration
logic. The `MemoryRegion::from_address()` already reads archspec but uses `aie2::compute`
directly -- a minor follow-up to wire it through an arch-handle accessor if AIE2P support
matters.

**Estimated LOC impact.** ~17 lines moving to archspec (EM_AIE + AieArchitecture enum);
~6 lines changing in xdna-emu (update imports); ELF dedup in Stage 8c (~100 LOC removed
from 4 consumer sites after `load_into` consolidation).

## 7. Control-packet parser overlap

**Files audited.** `src/device/control_packets/parser.rs` (643 LOC) vs. `src/parser/{xclbin,aie_partition,cdo,elf}.rs`.

**Field-level comparison table.**

| Concern | `control_packets/parser.rs` | `src/parser/*` | Same? |
|---------|----------------------------|----------------|-------|
| Header magic | None (control packets have no magic bytes; they start with a raw 32-bit header word) | `XCLBIN_MAGIC` (`b"xclbin2\0"`); `CDO_MAGIC_CDO` / `CDO_MAGIC_XLNX` | **No** -- fundamentally different; control packets are live stream messages, not file containers |
| Endianness | Little-endian implicit (field extraction via bit masks on a `u32`) | Little-endian explicit (`u32::from_le_bytes(...)` in CDO iterator; `zerocopy::FromBytes` in XCLBIN/Partition) | Similar in effect, different in mechanism |
| Length framing | `(header >> LENGTH_SHIFT) & LENGTH_MASK` + 1 = beats (2-bit field, 1-4 values) | CDO: `(cmd_word >> 16) & 0xFF` inline length with 0xFF extended-length extension; XCLBIN: explicit `section_size` field | **No** -- control packets use a 2-bit beat count; CDO uses an 8-bit word count with sentinel extension; entirely different |
| Opcode dispatch | 2-bit `CtrlOpCode` (Write/Read/WriteIncr/BlockWrite) from bits 23:22 | CDO: 16-bit `CdoOpcode` from bits 15:0; XCLBIN: 32-bit `SectionKind` | **No** -- different opcode widths, field positions, and value spaces |
| Error type | Local `ParseError` enum (3 variants) | `anyhow::Error` throughout; no structured error type in `src/parser/*` | Similar intent; different mechanism; structurally unrelated |
| Bit-field extraction | Named constants from `xdna_archspec::aie2::ctrl_packet::*` (`ADDRESS_MASK`, `LENGTH_SHIFT`, etc.) | CDO iterator: inline masks and shifts in `next()`; XCLBIN: `zerocopy` struct | **No** -- control_packets is more organized; parser/* is ad-hoc |
| Payload extraction | `payload: &[u32]` slice passed in by caller; no in-band length extraction in `parse()` | CDO: reads payload words from byte stream in-iterator; XCLBIN: reads section data from mmap | **No** -- control packets require caller-provided payload slice; file parsers are self-contained |
| Validation | Beat count vs. payload length validation + opcode validity | CDO: checksum validation (warn-only); XCLBIN: magic + size bounds | Superficially similar validation intent; no shared code |

**Conclusion.** Zero framing primitives are effectively duplicated. The overlap is
**coincidental only**:

- `control_packets/parser.rs` parses a live 32-bit stream header by extracting named bit
  fields using archspec-derived constants. It is a pure bit-manipulation module.
- `src/parser/*.rs` parses file containers (XCLBIN, CDO, ELF) by reading byte streams
  using `zerocopy` structs, `anyhow` for errors, and manual byte slicing.

The only structural similarities -- "both parse binary data" and "both validate input" -- are
too generic to warrant shared primitives. Extracting a shared `framing.rs` would be
premature abstraction.

One noteworthy asymmetry: `control_packets/parser.rs` has its own `ParseError` enum while
`src/parser/*.rs` uses bare `anyhow`. Stage 8c's `ParseError` work (Task 14) will add
structure to the latter, but there is no reason to merge the two -- they serve different
layers (live packet stream vs. file container) and their error semantics differ.

**Prescribed migration.** Leave as is. Add a module-level comment to
`control_packets/parser.rs` documenting the non-overlap decision per the spec's §"Audit
methodology" directive:

```
// No shared framing primitives with src/parser/*. This module parses a
// live 32-bit control-packet header stream; src/parser/* parses file
// containers (XCLBIN/CDO/ELF). Overlap is coincidental only -- see
// docs/arch/subsys8-audit.md §7 for the field-level comparison.
```

**Estimated LOC impact.** 2--4 lines (comment addition only).

## 8. Design note output

`docs/arch/binary-loader-model.md` -- "what would AIE1 look like?"
for the parser layer.

(Filled in by Task 5.)

## 9. Trait-or-no-trait decision

The spec's methodology asks: for each candidate `BinaryLoader` trait method, is there
algorithmic variance across AIE1/AIE2/AIE2P, or is the variance data-expressible? The
§§1--7 findings give clear answers.

**Rejection table.**

| Candidate trait method | Is there algorithmic variance? | Is the variance data-expressible? | Decision |
|------------------------|-------------------------------|-----------------------------------|---------- |
| `parse_container_sections(&self, bytes) -> Vec<Section>` | No. XCLBIN container format (magic, header, section headers) is identical across all AIE variants (§1). | N/A -- no variance. | **Reject: leave-alone.** No archspec involvement needed. |
| `extract_partition_payload(&self, section) -> PartitionPayload` | No. AIE Partition wrapper struct layout is identical across NPU1/NPU4/NPU6 (§2). Column width and start-column values differ across devices but are *data inside* the partition, not parser algorithm. | N/A -- no variance. | **Reject: leave-alone.** |
| `decode_cdo_command(&self, frame) -> CdoRaw` | No. CDO v2 byte framing (header, length field, opcode, payload) is identical across AIE1/AIE2/AIE2P (§3). The opcode set is a shared CDO v2 constant table; no per-arch byte layout differences observed. | N/A -- no variance. | **Reject: leave-alone.** |
| `lower_cdo(&self, raw) -> Iterator<DeviceOp>` | Mild: the address-decode step (`TILE_COL_SHIFT`, `TILE_ROW_SHIFT`) consults archspec constants that differ per arch. The *algorithm* (extract bits, look up tile, produce RegWrite) is identical. | Yes -- already data-expressible via existing archspec accessors (`memory_map().decode_global()`). The spec explicitly calls this a plain free function taking `&ArchHandle`. | **Reject as trait method: plain free function.** Variance handled by archspec data accessed through `ArchHandle`. |
| `load_elf(&self, bytes, &mut CoreMemory)` | No. ELF loading algorithm (iterate PT_LOAD segments, classify by `MemoryRegion`, write program/data) is identical across AIE1/AIE2/AIE2P (§6). `EM_AIE = 264` is shared across all three. `AieArchitecture` e_flags discriminates variant after parse, not during load. | N/A -- algorithm is uniform. | **Reject: leave-alone for trait; consolidate as `AieElf::load_into()` (Stage 8c).** |

**Summary of findings.** Zero candidate trait methods survive the rejection table. The
parser, AIE Partition wrapper, CDO framing, CDO semantics lowering, and ELF loader all
have arch-generic algorithms with arch-specific *data* flowing through existing archspec
channels. This mirrors the Subsystem 6 (ISA Decode, no trait) and Subsystem 7 (ISA
Execute, empty anchor) outcomes.

**Decision: No trait.**

The `BinaryLoader` trait is not warranted even as an empty anchor. The rationale for
the Subsystem 7 empty anchor was that the dispatch pathway (`arch_handle::isa_executor()`)
was worth reserving because the execute layer's hot path is a plausible future seam.
The parser layer does not have this argument: parsing runs once per XCLBIN load (not per
cycle), there is no per-cycle dispatch pathway, and the parser's internal structure
(future CDO split into framing/syntax/semantics) does not require a trait seam at its
top level. A future AIE1 port would add new `CdoOpcode` variants and provide different
archspec address constants; it would not need a `BinaryLoader` trait dispatch to do so.

This is the "No trait" outcome from the spec's §Stage 8a, matching the Subsystem 6
`IsaDecoder` precedent (which also has no trait).

---

## Closing summary

### Data migration list

Items ordered by priority. All verbs reference the archspec migration taxonomy.

| Item | Source file | Verb | Archspec module | Est. LOC change |
|------|-------------|------|-----------------|-----------------|
| `EM_AIE: u16 = 264` (ELF machine type) | `src/parser/elf.rs:36` | move-to-archspec | `xdna_archspec::elf` (new thin module) | +2 archspec, -1 xdna-emu |
| `AieArchitecture` enum (Aie1=0x01, Aie2=0x02, Aie2P=0x03) | `src/parser/elf.rs:40--60` | move-to-archspec | `xdna_archspec::elf` | +15 archspec, -18 xdna-emu |
| `decode_aie_address()` method (misplaced in `CdoCommand`) | `src/parser/cdo.rs:274--282` | leave-alone-but-relocate | N/A (stays in xdna-emu; moves to `cdo/semantics.rs` in Stage 8b) | ~15 LOC moves in Stage 8b |
| `MemoryRegion::from_address()` archspec pin (`aie2::compute` hardcoded) | `src/parser/elf.rs:86` | read-archspec-via-accessor | Already reads archspec; accessor should be arch-generic (Phase 2 hygiene) | 1 line in Stage 8c |

**Total archspec additions:** ~17 LOC. **Total xdna-emu changes:** ~20 LOC (migrations) +
~100 LOC removed in Stage 8c ELF dedup + ~50 LOC restructured in Stage 8b CDO split.

The parser is notably lighter on archspec migration than the execute subsystem. This is
expected: the parser deals with container framing (XCLBIN/CDO/ELF formats that are
independent of AIE variant) rather than hardware semantics. The two migration items
(`EM_AIE` and `AieArchitecture`) are toolchain-derived identifiers that should live in
archspec for consistency with prior subsystems' patterns.

### Trait decision (populated / anchor / none) + reasoning

**No trait.** See §9 rejection table for the full analysis. Every candidate `BinaryLoader`
method was rejected because its variance is either non-existent (XCLBIN/CDO/ELF framing
is arch-uniform) or data-expressible (CDO semantics lowering reads archspec through the
existing `ArchHandle` accessors). The parser layer does not have the hot-path dispatch
motivation that justified the Subsystem 7 empty anchor, and a future AIE1 port adds new
archspec data without needing a trait method. The outcome matches the Subsystem 6
`IsaDecoder` precedent.

No `BinaryLoader` module will be created in `crates/xdna-archspec`. Stage 8a's only
archspec output is the small `xdna_archspec::elf` module containing `EM_AIE` and
`AieArchitecture`.

### Refined `DeviceOp` enum proposal

The spec's starting hypothesis had 8 variants. The audit adds 3 (MaskPoll, Delay, Marker)
and collapses 2 (Write64/MaskWrite64 fold into RegWrite/RegMask). Nop and EndMark are
discarded (produce no `DeviceOp`). Net result: **9 variants** (starting hypothesis was 8;
net +1 from audit-discovered real-XCLBIN variants).

```rust
// src/device/ops.rs -- new module (Stage 8b Half 2)
//
// DeviceOp: arch-generic, device-facing operation vocabulary.
// Produced by cdo::semantics::lower(); consumed by device::state::apply().
//
// Design rules (from spec §"`DeviceOp` vocabulary"):
// 1. Device-facing, not CDO-facing (two CDO opcodes can produce the same op).
// 2. Arch-generic (TileAddr, BdFields, StreamRouteSpec via archspec).
// 3. Mixed granularity (RegWrite is the escape hatch).
// 4. Value-typed (Copy where possible; SmallVec for burst data only).
// 5. Audit-refined variant list (this block supersedes the starting hypothesis).

use xdna_archspec::types::TileAddr;
use xdna_archspec::aie2::dma::BdFields;
use xdna_archspec::aie2::stream_switch::StreamRouteSpec;
use xdna_archspec::aie2::dma::{DmaChannelId, DmaDir};
use smallvec::SmallVec;

#[derive(Debug, Clone)]
pub enum DeviceOp {
    // --- Register-level writes (dominant CDO outcomes, ~75% of commands) ---

    /// Single 32-bit register write.
    /// Produced by: CdoRaw::Write, CdoRaw::Write64 (after 64->32 truncation).
    RegWrite { tile: TileAddr, offset: u32, value: u32 },

    /// Masked register write: *reg = (*reg & !mask) | (value & mask).
    /// Produced by: CdoRaw::MaskWrite, CdoRaw::MaskWrite64.
    RegMask { tile: TileAddr, offset: u32, mask: u32, value: u32 },

    /// Bulk write: consecutive words starting at offset.
    /// Produced by: CdoRaw::DmaWrite (program/data memory loads).
    /// Uses SmallVec to avoid heap allocation for small payloads (<= 8 words).
    RegBurst { tile: TileAddr, offset: u32, words: SmallVec<[u32; 8]> },

    // --- Structured writes (archspec already names these) ---

    /// Configure a DMA Buffer Descriptor.
    /// Produced by: CdoRaw::Write to DMA BD register range (semantics::lower
    /// recognizes the offset range and calls arch.dma_model().parse_bd_words()).
    BdConfigure { tile: TileAddr, bd_id: u8, fields: BdFields },

    /// Initialize a lock to a specific value.
    /// Produced by: CdoRaw::Write to lock register range.
    LockInit { tile: TileAddr, lock_id: u8, value: i32 },

    /// Configure a stream switch connection.
    /// Produced by: CdoRaw::Write to stream switch register range.
    StreamRoute { tile: TileAddr, route: StreamRouteSpec },

    // --- Coarse control ---

    /// Enable or disable a compute core.
    /// Produced by: CdoRaw::Write or CdoRaw::MaskWrite to Core_Control register.
    CoreEnable { tile: TileAddr, enabled: bool },

    /// Start a DMA channel.
    /// Produced by: CdoRaw::Write to DMA channel start register.
    DmaStart { tile: TileAddr, channel: DmaChannelId, dir: DmaDir },

    // --- Synchronization / timing (audit-discovered; not in starting hypothesis) ---

    /// Poll a register until (value & mask) == expected.
    /// On real hardware: blocks until the condition is met (DMA completion, etc.).
    /// In the emulator: currently a logged no-op (writes are synchronous).
    /// Retaining as a variant preserves the information for future cycle-accurate work.
    /// Produced by: CdoRaw::MaskPoll, CdoRaw::MaskPoll64.
    /// Copy-able: all fields are primitive.
    MaskPoll { tile: TileAddr, offset: u32, mask: u32, expected: u32 },

    /// Timing delay for N cycles.
    /// On real hardware: inserts a wait. In the emulator: no-op.
    /// Produced by: CdoRaw::Delay.
    /// Copy-able.
    Delay { cycles: u32 },

    /// Debug sequence marker (value is an opaque tag).
    /// Always a no-op in device-state; useful for trace and test tooling.
    /// Produced by: CdoRaw::Marker.
    /// Copy-able.
    Marker { value: u32 },
}
```

**Variant justification vs. fold-into-catch-all:**

- `MaskPoll`: standalone because it has distinct semantics (conditional wait) and distinct
  payload shape (mask + expected) that don't fit `RegWrite`. The emulator currently ignores
  it, but retaining it costs one Copy-capable enum variant and preserves the option for
  cycle-accurate DMA ordering. Justified.
- `Delay`: standalone for the same reason -- timing-only, architecturally meaningful,
  trivially cheap as a Copy variant. The alternative (fold into a catch-all `Nop`) would
  lose the cycle-count information. Justified.
- `Marker`: standalone because it is a common CDO primitive used by AMD tooling to annotate
  configuration sequence boundaries. Trace tooling may want to observe it. One Copy variant
  with a single `u32` field. Justified.
- `Nop`/`EndMark`: **not in the refined enum.** These produce no `DeviceOp`; `semantics::lower`
  drops them (returns an empty iterator for EndMark; increments stats but emits nothing for
  Nop). This is cleaner than a `Nop` DeviceOp that `apply` must match and discard.
- `BdConfigure`/`LockInit`/`StreamRoute`/`CoreEnable`/`DmaStart`: **retained from starting
  hypothesis.** These require archspec-assisted interpretation of the target register offset
  (semantics::lower recognizes the offset range and promotes a plain RegWrite into one of
  these structured ops). The audit confirms they appear in real XCLBINs.

**THIS IS THE USER-GATED DELIVERABLE.** Stage 8b Half 2 does not start until the user
reviews this enum and confirms. If the user wants to reshape any variant (e.g., collapse
`BdConfigure` back into `RegBurst` to defer BD field parsing), that is the time to say so.

### AIE1 projection (one paragraph)

An AIE1 port of the parser layer would require almost no algorithm changes. The XCLBIN
container format is identical (same `RawHeader`, `RawSectionHeader`, `SectionKind` enum).
The AIE Partition wrapper struct is also identical -- AIE1 Versal designs use the same
partition section schema. CDO byte framing (20-byte header, `[len:16|opcode:16]` command
words, payload in 32-bit words) is the same CDO v2 format; the opcode set is identical.
The main AIE1-specific additions would be: (1) new `CdoOpcode` variants if AIE1
configurations use PM commands or NPI commands not in the current enum (likely a handful of
additions to the `CdoOpcode` `Unknown(u16)` catch-all); (2) a different `EM_AIE` flag
interpretation -- `AieArchitecture::Aie1 = 0x01` is already present in the enum; (3) an
`xdna_archspec::aie1::elf` module with AIE1-specific flag values (already handled by the
`AieArchitecture` flags scheme); and (4) different archspec constants in `semantics::lower`
(different `TILE_COL_SHIFT`, `TILE_ROW_SHIFT`, and register map for AIE1's different tile
address encoding). The `semantics::lower` function dispatches via `arch.memory_map()` and
`arch.dma_model()` -- an AIE1 `ArchConfig` impl would supply AIE1 values through the same
accessor calls. No new parsing algorithm is needed: the parser is a conforming CDO v2
consumer for any device that uses CDO v2 configuration.

---

## Completion

(Filled in at the end of Stage 8a, in Task 6.)
