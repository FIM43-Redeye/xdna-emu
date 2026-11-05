# TableGen Parser

Parses LLVM TableGen (.td) files from llvm-aie to generate instruction
decoder tables and semantic information for the AIE2 ISA.

The TableGen pipeline lives in `crates/xdna-archspec/src/aie2/isa/`.
Runtime consumers (the decoder itself) live in `src/interpreter/decode/`.

Use this component reference when working on ISA decoding or llvm-aie integration.

## Files

### Archspec-side (`crates/xdna-archspec/src/aie2/isa/`)

| File | Purpose |
|------|---------|
| `mod.rs` | Module root, re-exports public API, `load_from_generated()` |
| `types.rs` | Semantic vocabulary (`SemanticOp`, `ElementType`, `BranchCondition`, ...), scheduling, register model, composite formats, `TblgenOutput` |
| `resolver/mod.rs` | `InstrEncoding`, `OperandField`, and `DecoderIndex` (lookup by the name the FFI decoder reports) |
| `resolver/operand_types.rs` | Legacy per-field decode types (`OperandType`, `RegisterKind`, `CompositeEncoder`) and `AddressingMode` / `InstrMemWidth` |
| `schema.rs` | `EncodingSchema`: the TableGen record facts of each encoding (see below) |
| `decoder_ffi.rs` | C FFI to the llvm-aie disassembler: the only decode path (raw bits to instruction name) |
| `element_type_logic.rs` | Element-type token parser; the build script includes the same file |

The build script (`crates/xdna-archspec/build.rs`) runs the resolver at
build time and emits resolved encoding tables that the runtime loads with no
TableGen parsing on the hot path.

### Runtime-side (`src/interpreter/decode/`)

| File | Purpose |
|------|---------|
| `decoder.rs` | `InstructionDecoder` -- O(1) lookup decoder built from the resolved tables |
| `loader.rs` | Loads decoder tables from generated artifacts at startup |
| `composite.rs`, `crossref.rs`, `operand_extraction.rs`, `register_map.rs`, `slot_builder.rs` | Operand extraction, register mapping, and slot construction helpers |

## Encoding schema (`InstrEncoding::schema`)

Each encoding carries an `EncodingSchema` read from its TableGen record, never from its
name (`build_helpers/extract.rs::attach_encoding_schemas`, pure rules in
`build_helpers/schema_rules.rs`):

- `operands`: MC operands in `outs`-then-`ins` order (= `MCInst` index; no AIE2 operand has
  sub-operands) with the DAG name, the TableGen class, def/use, `Constraints` ties, whether
  encoding bits carry it, and its kind: a register class (family from the
  `AIE2*RegisterClass` it derives from, `RegSize`, the custom MC encoder of a composite
  `RegisterOperand`, the register a single-member class fixes) or an immediate with its LLVM
  `DecoderMethod` (`ImmDecode::decode`).
- `format_class` (the class directly below `AIE2_inst_<slot>_instr<N>`), `format_args` (its
  bit template arguments, e.g. the MAC `mdm` / `md1` / `md2`, traced through the class's
  `Inst` and read from the record's fixed bits), and `classes` (every class the record
  instantiates, bases first, minus the ones all encodings share).
- `addressing` (from the `*__ag_*` class), `mem_vt_bits` (Pat<> `MemoryVT`, propagated to
  same-format siblings; sub-word scalar accesses only), `implicit_regs` with hardware
  encodings, `stream` (blocking / TLAST from the derived format class).

The register model also carries `SubRegs` / `SubRegIndices` per register and the family of
each class (`RegisterModel::leaves`, `hw_encodings`, `base_hw_encoding`).

The schema sits beside the legacy name-derived fields (`operand_type`, `addressing_mode`,
`mem_width`, ...) that the interpreter still consumes; consumer packages switch to it and
delete the legacy field. `tools/datadriven_ledger.py` reports what the schema covers and
where it disagrees with the legacy value (M9, `schema_disagreements.tsv`).

## Parsing

The build script parses the selected llvm-aie `.td` sources with `tblgen-rs` (host LLVM 21;
the host LLVM ABI and the Peano source tree are separate inputs) and emits the resolved
encodings. There is no runtime TableGen parser: the regex parser and the runtime `Resolver`
were removed on 2026-11-04.

## Key Types

- `TblgenOutput` -- encodings by slot, scheduling model, itineraries, register model, composite formats
- `InstrEncoding` -- resolved encoding: fixed_mask, fixed_bits, operand fields, slot, semantic facts, `schema`
- `EncodingSchema` -- TableGen record facts per encoding (operands, format class, classes, ...)
- `SemanticOp` -- what an instruction does (Add, Sub, Load, Store, Br, ...)
- `RegisterModel` -- registers (hw_encoding, sub-registers) and register classes (members, size, family)
- `CompositeFormatDef` -- VLIW bundle layouts
- `DecoderIndex` -- lookup of an encoding by the instruction name the FFI decoder reports

## Parsing Results (from llvm-aie)

- 8 slots (lda, ldb, alu, mv, st, vec, lng, nop)
- ~144 format classes
- ~600+ instruction definitions
- 607 resolved encodings (2026-11-04; `tools/datadriven_ledger.py` prints the
  current count and where each encoding's semantics come from)
- about 100 `SemanticOp` variants

## llvm-aie Dependency

The shared resolver selects llvm-aie in this order:
`LLVM_AIE_PATH`, `LLVM_AIE_DIR`, `NPU_WORK_DIR/llvm-aie`, then upward
ancestor discovery. Explicit configuration is fail-closed. A usable root must
contain `llvm/lib/Target/AIE/AIE2.td` and an executable
`build/bin/llvm-config`.

That one resolved tree supplies both the AIE TableGen sources and the matching
decoder FFI libraries. No downstream build consumer performs another fallback
search.

The `tblgen-rs` crate separately finds its host LLVM 21 through `llvm-config`
on `PATH`. It honors `TABLEGEN_210_PREFIX` as an explicit host-LLVM override.
The workspace `.cargo/config.toml` intentionally does not set that variable,
because a checkout-relative prefix breaks linked worktrees.

Source files consumed:
```
llvm-aie/llvm/lib/Target/AIE/
  AIE2Slots.td             # Slot definitions
  AIE2GenInstrFormats.td   # Format classes
  AIE2GenInstrInfo.td      # Generated instruction definitions
  AIE2InstrInfo.td         # Main instruction definitions
  AIE2GenFixupInstrInfo.td # Fixup/vector instructions
  AIE2InstrPatterns.td     # Semantic patterns
```

## Conventions

- Doc tests in this module load real TableGen files and are expensive.
  The test script runs them with `nice -n 19` and limited parallelism.
- The runtime decoder never parses .td files itself; it reads the
  pre-resolved encoding tables emitted by the archspec build script; instruction
  identification goes through the LLVM disassembler FFI.
- When updating TableGen sources or adding a new instruction class,
  rebuild the archspec crate (`cargo build -p xdna-archspec`) so the
  regenerated tables flow into the runtime decoder.
