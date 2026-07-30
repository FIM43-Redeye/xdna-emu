# TableGen Parser

Parses LLVM TableGen (.td) files from llvm-aie to generate instruction
decoder tables and semantic information for the AIE2 ISA.

The TableGen pipeline lives in `crates/xdna-archspec/src/aie2/isa/`.
Runtime consumers (the decoder itself) live in `src/interpreter/decode/`.

Use this component reference when working on ISA decoding, decoder bytecode,
or llvm-aie integration.

## Files

### Archspec-side (`crates/xdna-archspec/src/aie2/isa/`)

| File | Purpose |
|------|---------|
| `mod.rs` | Module root, re-exports public API |
| `types.rs` | Core data types: `TableGenData`, `SlotDef`, `FormatClass`, `InstrDef`, `InstrEncoding`, `SemanticOp`, `SemanticPattern` |
| `resolver/mod.rs` | `Resolver` -- computes concrete encodings (mask/bits) from format-class inheritance |
| `resolver/operand_classification.rs` | Operand type classification used by the resolver |
| `resolver/semantic_inference.rs` | Semantic op inference from instruction encoding patterns |
| `decoder_bytecode.rs` | Compact bytecode form of resolved encodings (what we ship to the runtime decoder) |
| `decoder_ffi.rs` | C FFI bridging to the llvm-aie disassembler when needed |
| `element_type_logic.rs` | Element-type derivation for vector instructions |

The build script (`crates/xdna-archspec/build.rs`) runs the resolver at
build time and emits decoder tables that the runtime loads with no
TableGen parsing on the hot path.

### Runtime-side (`src/interpreter/decode/`)

| File | Purpose |
|------|---------|
| `decoder.rs` | `InstructionDecoder` -- O(1) lookup decoder built from the resolved tables |
| `loader.rs` | Loads decoder tables from generated artifacts at startup |
| `composite.rs`, `crossref.rs`, `operand_extraction.rs`, `register_map.rs`, `slot_builder.rs` | Operand extraction, register mapping, and slot construction helpers |

## Two Parsing Paths

The archspec module supports two complementary approaches:

1. **Regex parsing**: Directly parses .td source files. Fast, no external
   tool dependency, but cannot resolve all template inheritance.
2. **LLVM-backed parsing**: `tblgen-rs` uses host LLVM 21 to resolve records
   from the selected llvm-aie `.td` sources. The host LLVM ABI and the Peano
   source tree are separate inputs.

Both produce `InstrEncoding` values consumable by the decoder.

## Key Types

- `TableGenData` -- all parsed data (slots, formats, instructions, patterns)
- `SlotDef` -- VLIW slot definition (name, bit width, field name)
- `FormatClass` -- instruction format with encoding pattern and template params
- `InstrDef` -- concrete instruction (name, format, mnemonic, operands)
- `SemanticOp` -- what an instruction does (Add, Sub, Load, Store, Br, ...)
- `SemanticPattern` -- maps a `SemanticOp` to one or more instruction names
- `InstrEncoding` -- resolved encoding: fixed_mask, fixed_bits, operand fields, slot assignment
- `Resolver` -- resolves `InstrDef` + `FormatClass` into `InstrEncoding`

## Parsing Results (from llvm-aie)

- 8 slots (lda, ldb, alu, mv, st, vec, lng, nop)
- ~144 format classes
- ~600+ instruction definitions
- ~210+ fully resolved encodings
- ~48 semantic operations

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
  pre-resolved bytecode tables emitted by the archspec build script.
- When updating TableGen sources or adding a new instruction class,
  rebuild the archspec crate (`cargo build -p xdna-archspec`) so the
  regenerated tables flow into the runtime decoder.
