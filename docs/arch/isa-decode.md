# ISA Decode -- Design Note

**Subsystem:** 6 (Phase 1b)
**Tag:** `phase1-subsys-isa-decode`
**Spec:** [../superpowers/specs/2026-04-17-subsys6-isa-decode-design.md](../superpowers/specs/2026-04-17-subsys6-isa-decode-design.md)

This document is the mandatory per-seam design note required by the
parent device-family refactor. Subsystem 6 adds no trait seam; this
note explains *why not* for ISA decode specifically, and how AIE1
support fits.

---

## What lives where

All entries below are relative to the `phase1-subsys-isa-decode` tag.

| Data/code | Module | Source of truth |
|-----------|--------|-----------------|
| TableGen runtime types (`SemanticOp`, `SlotBitMap`, `EncodingPart`, `CompositeFormatDef`, `InstrEncoding`, `RegisterModel`, `ProcessorModel`, `ItineraryInfo`, ...) | `xdna_archspec::aie2::isa::types` | llvm-aie TableGen extraction |
| Operand classification + semantic inference (`classify_operand_type`, `infer_element_type`, `refine_branch_semantic`, ...) | `xdna_archspec::aie2::isa::resolver` | Heuristics over TableGen data |
| Decoder bytecode walker | `xdna_archspec::aie2::isa::decoder_bytecode` | Data-driven |
| LLVM MCDisassembler FFI (raw) | `xdna_archspec::aie2::isa::decoder_ffi` | llvm-aie LLVM libraries |
| Root re-exports matching the old forwarder surface (`SemanticOp`, `ImplicitReg`, `BranchCondition`, `ElementType`, ...) | `xdna_archspec::aie2::isa` (via `pub use types::*` / `pub use resolver::...`) | TableGen + heuristics |
| Generated instruction tables (`gen_tablegen.rs`) | `xdna_archspec::aie2::isa::generated` | `crates/xdna-archspec/build.rs` via `build_helpers/` |
| aie-rt cross-validated constants (DMA, locks, stream ports) | `xdna_archspec::aie2::aiert::{dma,locks,ports}` | aie-rt header extraction |
| LLVM-decoded-operand -> emulator Operand classifier (`MappedOperand`, `RegisterMap`, `classify_reg_name`, `operand_from_reg_name`, `AccumWidth`) | `xdna_emu::interpreter::decode::register_map` | emulator convention |
| Operand enum (execution state) | `xdna_emu::interpreter::bundle::slot::Operand` | emulator convention |
| Interpreter register-file indexing constants (`LR_REG_INDEX`, `SP_PTR_INDEX`, `MOD_BASE_*`, ...) | `xdna_emu::interpreter::state` | emulator convention |

The split is driven by one rule: a type belongs in `xdna-archspec` iff it
is derivable from the toolchain (TableGen + LLVM + aie-rt) without
reference to emulator execution state. Types that name `Operand`,
register-file indices, or emulator convention stay in `xdna-emu`.

---

## The const-first principle, applied to ISA decode

Subsystem 1 established the rule: lift per-arch differences behind traits
only when they are *shape* differences, not *values* differences.

For ISA decode, the delta between AIE2 and a hypothetical AIE1 port is:

- Different `SlotBitMap` values (AIE1 has fewer VLIW slots and narrower
  bundles than AIE2).
- Different `CompositeFormatDef` entries (AIE1 bundles pack a different
  slot combination into a different byte layout).
- Different instruction encodings (AIE1 has fewer opcodes and different
  operand formats).
- Different `ProcessorModel` and itineraries.
- Different register classes (AIE1 lacks AIE2's accumulator width
  hierarchy).

Every one of these is a data change. The decoder bytecode walker
(`decoder_bytecode.rs`), the operand classifier
(`resolver::operand_classification`), and the semantic inference logic
(`resolver::semantic_inference`) all work off the data tables without
hardcoding slot counts, specific bundle shapes, or particular opcode
layouts.

A trait would be justified if an AIE1 instruction required a
fundamentally different decode discipline -- e.g., non-TableGen-sourced
data, or per-operand extraction that doesn't fit the fragment model.
Nothing in the AIE1 public documentation suggests that. The shape is
invariant; only the values differ.

## What would AIE1 look like?

- `xdna_archspec::aie1::isa::` module, mirroring `aie2::isa::` structure.
- Its own `build_helpers/` fed from AIE1's TableGen sources (a different
  llvm-aie directory or backend).
- Its own `decoder_ffi/` linked against an AIE1-configured LLVM build.
- The same `types`, `resolver`, `decoder_bytecode` algorithmic code --
  the current archspec types genuinely do work for both. If the types
  themselves need to change (e.g., to fit a `SlotIndex` variant that
  doesn't exist for AIE2), we'd extend the enum, not fork the code.

The register-map side (`xdna_emu::interpreter::decode::register_map`)
would stay per-emulator and per-arch: AIE1's scalar/vector/accumulator
conventions differ from AIE2's, so the LLVM-register-name to `Operand`
mapping is arch-specific and interpreter-specific.

## Where a trait could enter

The hot-path `IsaDecoder` trait candidate:

```rust
pub trait IsaDecoder {
    fn decode_slot(&self, slot: Slot, bits: u64) -> Option<DecodeResult>;
    fn bundle_layout(&self) -> &[CompositeFormatDef];
    fn instr_info(&self, opcode: u32) -> Option<&InstrInfo>;
}
```

Not introduced in Subsystem 6 because no second architecture is being
populated. If AIE1 decoder data lands and the `decode_slot` path
genuinely diverges (e.g., different LLVM entry points, different
disambiguation strategy), this is where the trait enters. Until then,
the concrete path is one FFI call away and the const-first choice holds.

---

## What about behavior seams?

- Instruction *execution* semantics (Subsystem 7, not yet landed) almost
  certainly warrant an `IsaExecutor` trait: vector rounding, saturation,
  configuration-word interpretation, and accumulator precedence genuinely
  vary between arch families.
- Bundle *parsing* (which slot-category gets the next N bits) is data,
  per `CompositeFormatDef`. No trait.
- Register-file *layout* (which LLVM name maps to which `Operand`
  variant) is emulator convention and lives in xdna-emu. If we ever
  support multiple emulator execution models per-arch, this gets a
  trait; for now it's a plain module.

The trait boundary rides the "behavior differs in shape" criterion.
**ISA decode is values. ISA execute is shapes.**
