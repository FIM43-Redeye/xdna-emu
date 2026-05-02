# TableGen Parser

Parses LLVM TableGen (.td) files from llvm-aie to generate instruction
decoder tables and semantic information for the AIE2 ISA.

The TableGen pipeline lives in `crates/xdna-archspec/src/aie2/isa/`.
Runtime consumers (the decoder itself) live in `src/interpreter/decode/`.

Read this file when working on ISA decoding, decoder bytecode, or
llvm-aie integration.

## Files

### Archspec-side (`crates/xdna-archspec/src/aie2/isa/`)

| File | Purpose |
|------|---------|
| `mod.rs` | Module root, re-exports public API |
| `types.rs` | Core data types: `TableGenData`, `SlotDef`, `FormatClass`, `InstrDef`, `InstrEncoding`, `SemanticOp`, `SemanticPattern` |
| `resolver/mod.rs` | `Resolver` -- computes concrete encodings (mask/bits) from format-class inheritance |
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
2. **llvm-tblgen**: Runs the actual llvm-tblgen binary to get fully
   resolved records. More accurate but requires an AIE-enabled
   llvm-tblgen (we use the local llvm-aie build at `../llvm-aie/build`,
   selected via the `TABLEGEN_210_PREFIX` env var set by the workspace
   `.cargo/config.toml`).

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

The TableGen files are read from a local llvm-aie clone (default path:
`../llvm-aie`). Path is configurable via `xdna-emu.toml` (`llvm_aie_path`)
or the `LLVM_AIE_PATH` environment variable.

The tblgen-rs crate that runs llvm-tblgen reads
`TABLEGEN_<MAJOR>0_PREFIX` to find the LLVM install. The workspace
`.cargo/config.toml` sets `TABLEGEN_210_PREFIX` to `../llvm-aie/build`.

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
