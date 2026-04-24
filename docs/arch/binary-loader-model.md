# Binary Loader Model -- Design Note

**Subsystem:** 8 (Phase 1b)
**Tag:** `phase1-subsys-parser-arch`
**Spec:** [../superpowers/specs/2026-04-23-subsys8-parser-design.md](../superpowers/specs/2026-04-23-subsys8-parser-design.md)
**Audit:** [subsys8-audit.md](subsys8-audit.md)

This document is the mandatory per-seam design note required by the
parent device-family refactor. It explains where arch-specific data
lives after Stage 8a's migrations, what the `BinaryLoader` trait's
surface is (if any), and what an AIE1 port of the parser would
look like.

Filled in after Task 2 (audit) lands and Task 5 (trait decision)
lands.

---

## What lives where

| Data / code | Module | Notes |
|-------------|--------|-------|
| `EM_AIE` (ELF machine type = 264) | `xdna_archspec::elf` | Moved from `src/parser/elf.rs` in Task 4. |
| `AieArchitecture` enum + `from_e_flags` | `xdna_archspec::elf` | Moved from `src/parser/elf.rs` in Task 4. |
| XCLBIN container parsing (`Xclbin`, `SectionKind`) | `src/parser/xclbin.rs` | Arch-agnostic; stays in xdna-emu. |
| AIE Partition wrapper (`AiePartition`) | `src/parser/aie_partition.rs` | Arch-agnostic framing; stays in xdna-emu. |
| CDO framing (`CdoVersion`, `RawCdoHeader`, `find_cdo_offset`) | `src/parser/cdo/framing.rs` | Arch-uniform byte-level framing. Created in Stage 8b Half 1. |
| CDO syntax (`CdoOpcode`, `CdoRaw`, `Cdo`) | `src/parser/cdo/syntax.rs` | Typed commands; arch-uniform byte format. Created in Stage 8b Half 1. |
| CDO semantics (`lower`) + arch-handle-consuming lowering | `src/parser/cdo/semantics.rs` | Reads archspec (memory map, DMA model, etc.) through existing accessors. No new arch dispatch. Created in Stage 8b Half 2. |
| ELF parsing (`AieElf`, `MemoryRegion`, `load_into`) | `src/parser/elf.rs` | Arch-agnostic ELF reader; uses `xdna_archspec::elf` constants. Consolidated in Stage 8c. |
| `DeviceOp` enum (arch-generic device ops) | `src/device/ops.rs` | New vocabulary between parser and device-state. Created in Stage 8b Half 2. |

## Trait surface

**No trait.** The parser layer does not define a `BinaryLoader` trait in
`xdna-archspec`. This matches the Subsystem 6 `IsaDecoder` precedent and
reflects three findings from the audit:

1. **XCLBIN / AIE Partition / CDO framing is arch-uniform.** Every AIE2 NPU
   (and every AIE1 Versal design that the audit checked) uses the same XCLBIN
   magic, the same AIE Partition struct, and the same CDO v2 byte-level format.
   There is no variance to dispatch on at the container-parsing layer.

2. **CDO semantics variance is data-expressible.** The `semantics::lower`
   function consults `ArchHandle` accessors (memory map, DMA model, lock layout,
   stream switch topology) for address decoding and BD field parsing. All four
   of those accessors are already arch-aware through their respective model
   traits (Subsystems 1, 3, 4, 5). `lower` adds no new arch-dispatch surface;
   it's a plain free function that parameterises through pre-existing accessors.

3. **Parsing is not on a hot path.** Unlike ISA execute (where the interpreter
   inner loop might benefit from a trait anchor to monomorphise later), the
   parser runs once at XCLBIN load. There is no dispatch-pathway-reservation
   argument that would motivate the empty-anchor pattern Subsystem 7 used.

The single small archspec deliverable is `xdna_archspec::elf` (Task 4), which is
a data module (two constants + one enum), not a trait.

## The shape-vs-values rule, applied to parser

Subsystem 6 established the rule:

> A type belongs in archspec iff it is derivable from the toolchain without
> reference to emulator execution state.

Subsystem 7 sharpened it:

> Among arch-specific content, *data* (tables, enums, constants, feature flags)
> lives in archspec while *algorithms* stay in xdna-emu.

Applied to the parser:

- **Data** that the toolchain (llvm-aie + aie-rt) defines -- `EM_AIE`,
  `AieArchitecture`, CDO opcode identity -- lives in archspec. The Task 4
  migration moves `EM_AIE` and `AieArchitecture` specifically; `CdoOpcode` stays
  in xdna-emu because its byte-level opcode values are not defined by the
  toolchain (they're defined by AMD's CDO specification, which is arch-uniform
  and doesn't warrant the migration overhead for one enum).
- **Algorithms** -- XCLBIN section iteration, AIE Partition extraction, CDO
  command decoding, ELF section-to-memory copying -- stay in xdna-emu. These
  read archspec data where needed (Task 4 adds one such read site in
  `parser::elf`) but are themselves pure functions of bytes-in, typed-out.
- **The one interesting layering departure** is `cdo::semantics::lower`, which
  reads archspec freely (memory map, DMA model, etc.) during the byte-to-op
  translation. This is deliberate -- the spec's design philosophy §"Elegant
  over pristine purity" explicitly allows the parser to be arch-aware in the
  semantics sublayer. The syntax and framing sublayers stay arch-blind for
  testability; the semantics sublayer is arch-aware because the lowering task
  demands it.

## What would AIE1 look like?

An AIE1 port of the parser layer would require no algorithm changes and one
small archspec addition.

**Archspec additions:**
- `xdna_archspec::aie1::elf` -- if AIE1's ELF flag values ever diverge from
  AIE2's (e.g., AIE1 uses `e_flags = 0x01`). Today this is handled by the
  existing `AieArchitecture::Aie1 = 0x01` variant, so the module may not even
  need to be created.
- AIE1-specific `CdoOpcode` variants -- if AIE1 Versal designs include PM-domain
  or NPI commands not present in today's enum. These would be additions to
  `CdoOpcode`'s `Unknown(u16)` catch-all, which is in xdna-emu not archspec; the
  catch-all already handles unknown opcodes gracefully, so they don't need to
  be named unless the parser needs to interpret them.

**Parser-side changes:** **none.** `xclbin::parse`, `aie_partition::parse`,
`cdo::framing::parse_header`, `cdo::syntax::decode`, `elf::AieElf::parse`, and
`elf::AieElf::load_into` are all byte-level parsers that work against
arch-uniform formats. `cdo::semantics::lower` dispatches through `ArchHandle`
accessors already -- when `ArchConfig::Aie` is populated with AIE1 values, the
accessors return AIE1 data and the lowering works unchanged.

**Device-state side changes:** none specific to the parser -- `device::state::apply`
consumes `DeviceOp`, which is already arch-generic.

The smoothness of this projection is precisely why the audit concluded "no trait":
there is no variance at the parsing layer that a trait would dispatch on.

## Alternatives rejected

### Populated `BinaryLoader` trait (1--3 methods)

Rejected. §9's rejection table evaluated five candidate methods
(`parse_container_sections`, `extract_partition_payload`,
`decode_cdo_command`, `lower_cdo`, `load_elf`). None survived -- every
candidate either has zero algorithmic variance (container framing is
arch-uniform) or is data-expressible through existing archspec accessors
(CDO semantics). A populated trait would be ceremony.

### Empty anchor trait (mirroring `IsaExecutor`)

Rejected. Subsystem 7 justified the `IsaExecutor` empty anchor on two
grounds: (a) preserving a dispatch pathway for future seams without
cross-subsystem plumbing, and (b) hot-path dispatch might eventually
benefit from monomorphisation. Neither applies to the parser:

- Future parser seams would most likely be new archspec data (new
  opcodes, new ELF flags), not new dispatch. An anchor doesn't help
  data migrations.
- Parsing is a one-shot cold path. There is no hot-path motivation.

The cost of an empty anchor (one module, one ZST, one singleton, one
dispatch test, one accessor) was deemed not worth paying for a pathway
we don't expect to populate.

### Pre-audit commitment

Rejected by spec. The parent device-family refactor explicitly requires
audit-first design. Subsystem 5's `PortLayout` extension trait (231 LOC
of dead code deleted after the fact) is the cautionary precedent.
