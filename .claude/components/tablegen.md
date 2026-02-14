# TableGen Parser

Parses LLVM TableGen (.td) files from llvm-aie to generate instruction decoder tables and semantic information for the AIE2 ISA.

Read this file when working on anything in `src/tablegen/`.

## Files

| File | Purpose |
|------|---------|
| `mod.rs` | Public API: `load_from_llvm_aie()`, `load_via_tblgen()`, `find_aie_tblgen()` |
| `parser.rs` | Regex-based .td file parsing: slots, format classes, instructions, patterns |
| `types.rs` | Core data types: `TableGenData`, `SlotDef`, `FormatClass`, `InstrDef`, `SemanticOp`, `SemanticPattern` |
| `resolver.rs` | `Resolver` -- computes concrete encodings (mask/bits) from format class inheritance; `DecoderIndex` for O(1) lookup |
| `tblgen_records.rs` | `parse_tblgen_records()` -- parses llvm-tblgen --print-records output for fully-resolved encodings |

## Two Parsing Paths

The module supports two complementary approaches:

1. **Regex parsing** (`parser.rs`): Directly parses .td source files. Fast, no external tool dependency, but cannot resolve all template inheritance.
2. **llvm-tblgen** (`tblgen_records.rs`): Runs the actual llvm-tblgen binary to get fully-resolved records. More accurate but requires an AIE-enabled llvm-tblgen binary.

Both produce `InstrEncoding` values consumable by the decoder.

## Key Types

- `TableGenData` -- all parsed data (slots, formats, instructions, patterns)
- `SlotDef` -- VLIW slot definition (name, bit width, field name)
- `FormatClass` -- instruction format with encoding pattern and template params
- `InstrDef` -- concrete instruction (name, format, mnemonic, operands, attributes)
- `SemanticOp` -- what an instruction does (Add, Sub, Load, Store, etc.)
- `SemanticPattern` -- maps a `SemanticOp` to an instruction name
- `InstrEncoding` -- resolved encoding: fixed_mask, fixed_bits, operand fields, slot assignment
- `Resolver` -- resolves `InstrDef` + `FormatClass` into `InstrEncoding`
- `DecoderIndex` -- per-slot lookup table for O(1) instruction identification

## Parsing Results (from llvm-aie)

- 8 slots (lda, ldb, alu, mv, st, vec, lng, nop)
- ~144 format classes
- ~600+ instruction definitions
- ~210+ fully resolved encodings
- ~40+ semantic operations

## llvm-aie Dependency

The TableGen files are read from a local llvm-aie clone (default path: `../llvm-aie`). The path is configurable via:
- `xdna-emu.toml` (`llvm_aie_path` key)
- `LLVM_AIE_PATH` environment variable

Source files consumed:
```
llvm-aie/llvm/lib/Target/AIE/
  AIE2Slots.td            # Slot definitions
  AIE2GenInstrFormats.td   # Format classes
  AIE2GenInstrInfo.td      # Generated instruction definitions
  AIE2InstrInfo.td         # Main instruction definitions
  AIE2GenFixupInstrInfo.td # Fixup/vector instructions
  AIE2InstrPatterns.td     # Semantic patterns
```

## Conventions

- Doc tests in this module load real TableGen files and are expensive. The test script runs them with `nice -n 19` and limited parallelism.
- `load_from_llvm_aie()` is the primary entry point for most code.
- `load_via_tblgen()` is used when higher accuracy is needed (e.g., disambiguation of ACQ_mLockId_imm vs ACQ_mLockId_reg).
