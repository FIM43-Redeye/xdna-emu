# Build-Time TableGen Extraction Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move all TableGen instruction extraction from runtime to build time, eliminating the runtime llvm-aie dependency and producing a fully self-contained binary.

**Architecture:** Build helpers (`build_helpers/`) use the `tblgen` crate (LLVM native binding) at build time to extract 607+ instruction encodings, decoder bytecode tables, scheduling model, and register data from llvm-aie. The extraction output is formatted as Rust source (`gen_tablegen.rs`) that constructs the existing runtime types (`InstrEncoding`, `DecoderTable`, etc.). The runtime loads from generated constants instead of calling tblgen. After the switch, `native.rs`, `cpp_switch.rs`, `tblgen_records.rs`, and the runtime cache logic are deleted.

**Tech Stack:** Rust, tblgen crate (LLVM TableGen binding), Cargo build scripts, `#[path]` module includes.

---

## File Structure

### New files (build-time extraction)

| File | Responsibility |
|------|---------------|
| `build_helpers/mod.rs` | Module root, public `generate_tablegen()` entry point |
| `build_helpers/extract.rs` | tblgen crate extraction (adapted from `src/tablegen/native.rs`) |
| `build_helpers/records.rs` | Intermediate types: `BuildInstrRecord`, `BuildSlotEncoding`, `BuildEncodingBit` (adapted from `src/tablegen/tblgen_records.rs`) |
| `build_helpers/cpp_switch.rs` | C++ `AIE2InstrInfo.cpp` parser (adapted from `src/tablegen/cpp_switch.rs`) |
| `build_helpers/bytecode.rs` | Decoder bytecode text parser: `extract_all_tables()`, `parse_decoder_table_bytes()`, `parse_opcode_names_from_disasm()` (moved from `src/tablegen/decoder_bytecode.rs` -- the parser, not the interpreter) |
| `build_helpers/semantics.rs` | Semantic inference: `infer_semantic_from_structure()`, `classify_operand_type()`, `detect_addressing_mode()`, `detect_mem_width()`, etc. (adapted from `src/tablegen/resolver.rs`) |
| `build_helpers/codegen.rs` | Generates `gen_tablegen.rs` from extraction output |

### Modified files

| File | Changes |
|------|---------|
| `Cargo.toml` | Add `tblgen` to `[build-dependencies]` |
| `build.rs` | Add `#[path]` module include, call `generate_tablegen()` |
| `src/tablegen/mod.rs` | Add `include!()` for generated code, add `load_from_generated()`, remove `load_full_via_tblgen()` and related functions |
| `src/interpreter/decode/decoder.rs` | Switch `try_load_via_tblgen()` to use `load_from_generated()` |

### Deleted files (cleanup phase)

| File | Reason |
|------|--------|
| `src/tablegen/native.rs` (1281 lines) | Replaced by `build_helpers/extract.rs` |
| `src/tablegen/cpp_switch.rs` (403 lines) | Replaced by `build_helpers/cpp_switch.rs` |
| `src/tablegen/tblgen_records.rs` (348 lines) | Replaced by `build_helpers/records.rs` |

`src/tablegen/decoder_bytecode.rs` is split: parser functions move to `build_helpers/bytecode.rs`, the runtime `DecoderTable` struct + `decode()` method stays.

---

## Key Design Decisions

### Why `build_helpers/` with `#[path]` instead of xdna-archspec?

The extraction output must construct types defined in the main crate (`InstrEncoding`, `OperandType`, `SemanticOp`, etc.). Putting extraction in xdna-archspec would require either (a) moving all those types to xdna-archspec (massive import churn) or (b) duplicating types (DRY violation). Build helper modules avoid both problems: they generate Rust *text* that references main crate types, without importing them.

### Why string-based enum formatting?

Build helpers can't import the main crate's enums (build.rs compiles before the crate). Instead, helper functions format enum values as strings (e.g., `"SemanticOp::Add"`) that the generated code uses literally. The compiler validates these strings when compiling the generated code -- a typo becomes a compile error immediately.

### What the generated code looks like

`gen_tablegen.rs` defines a function `load_from_generated() -> TblgenOutput` that constructs the same data structure as the current `load_full_via_tblgen()`. All 607+ instruction encodings, decoder bytecode tables, and metadata are inlined as Rust literals. This is ~20,000-30,000 lines of generated code in `$OUT_DIR/`, never committed.

---

## Chunk 1: Build-Time Extraction Infrastructure

### Task 1: Add tblgen to build-dependencies and create module skeleton

**Files:**
- Modify: `Cargo.toml` (add `[build-dependencies]` entry)
- Create: `build_helpers/mod.rs`
- Create: `build_helpers/records.rs`
- Modify: `build.rs` (add `#[path]` include, add `cargo:rerun-if-changed`)

- [ ] **Step 1: Add tblgen to build-dependencies in Cargo.toml**

In `Cargo.toml`, add to `[build-dependencies]`:
```toml
tblgen = { git = "https://github.com/FIM43-Redeye/tblgen-rs.git", branch = "feat/varbit-init", default-features = false, features = ["llvm20-0"] }
```
This is the same line already in `[dependencies]`. Keep both for now.

- [ ] **Step 2: Create build_helpers/records.rs with intermediate types**

These mirror `src/tablegen/tblgen_records.rs` but are independent (build context types). Copy the `InstrRecord`, `SlotEncoding`, `EncodingBit` types and the `compute_fixed_bits()` / `extract_operand_fields()` / `build_fragments()` methods. Rename with `Build` prefix to avoid confusion. Keep `BuildOperandField` and `BuildFieldFragment` as local types since the build context can't import `OperandField`.

Key types:
```rust
pub struct BuildFieldFragment { pub inst_bit: u8, pub width: u8, pub target_bit: u8 }
pub struct BuildOperandField { pub name: String, pub bit_position: u8, pub width: u8, pub signed: bool, pub operand_type: String, pub is_output: bool, pub fragments: Vec<BuildFieldFragment> }
pub struct BuildEncodingBit { Zero, One, DontCare, FieldBit { field: String, bit: u8 } }
pub struct BuildSlotEncoding { pub slot: String, pub width: u8, pub parts: Vec<BuildEncodingBit> }
pub struct BuildInstrRecord { /* same fields as InstrRecord */ }
pub struct BuildInstrEncoding { /* all fields as primitives/Strings */ }
```

Where the main crate uses enum types like `SemanticOp`, `AddressingMode`, etc., the build types use `String` (e.g., `pub semantic: Option<String>` holds `"SemanticOp::Add"`).

- [ ] **Step 3: Create build_helpers/mod.rs skeleton**

```rust
pub mod records;
// Future: pub mod extract; pub mod cpp_switch; pub mod bytecode; pub mod semantics; pub mod codegen;
```

- [ ] **Step 4: Wire into build.rs**

Add at the top of `build.rs`:
```rust
#[path = "build_helpers/mod.rs"]
mod build_helpers;
```

Add `cargo:rerun-if-changed` lines for all `build_helpers/*.rs` files in `main()`.

- [ ] **Step 5: Verify build succeeds**

Run: `cargo build 2>&1 | head -20`
Expected: builds with no errors. The new modules compile but aren't called yet.

- [ ] **Step 6: Commit**

```
feat(build): add tblgen build-dependency and extraction module skeleton
```

---

### Task 2: Port semantic inference to build helpers

**Files:**
- Create: `build_helpers/semantics.rs`

This is the logic that classifies operand types, addressing modes, memory widths, element types, branch conditions, select variants, and structural semantics. Ported from `src/tablegen/resolver.rs` functions. Instead of returning enum values, these functions return `&str` that matches the enum variant path (e.g., `"OperandType::Register(RegisterKind::Scalar)"`).

- [ ] **Step 1: Create semantics.rs with classify_operand_type()**

Port `classify_operand_type()`, `parse_immediate_type()`, `extract_scale_suffix()`, and `classify_from_field_name()` from `src/tablegen/resolver.rs:198-328`. Each returns a `String` representing the `OperandType` constructor expression.

Example: where the original returns `OperandType::Register(RegisterKind::Scalar)`, the build version returns `"OperandType::Register(RegisterKind::Scalar)".to_string()`.

- [ ] **Step 2: Add detect_addressing_mode() and detect_mem_width()**

Port from `resolver.rs:44-120`. Return `&str` matching `AddressingMode::*` and `InstrMemWidth::*`.

- [ ] **Step 3: Add infer_semantic_from_structure()**

Port from `resolver.rs` (the function that maps structural flags like `mayLoad`, `hasDelaySlot`, `Defs=[lr]` to semantic operation strings). Return `Option<String>` where the string is `"SemanticOp::Load"`, etc.

- [ ] **Step 4: Add infer_element_type(), infer_branch_condition(), infer_select_variant()**

Port from `resolver.rs`. Return `Option<String>` with variant paths.

- [ ] **Step 5: Add refine_branch_semantic()**

Port from `resolver.rs`. Takes mnemonic + current semantic string, returns refined semantic string.

- [ ] **Step 6: Add to_build_encoding() on BuildInstrRecord**

Port from `tblgen_records.rs:208-323`. This converts a `BuildInstrRecord` to a `BuildInstrEncoding` using all the semantic functions above. This is the central conversion that populates all fields.

- [ ] **Step 7: Verify build succeeds**

Run: `cargo build 2>&1 | head -20`
Expected: compiles, not called yet.

- [ ] **Step 8: Commit**

```
feat(build): port semantic inference to build helpers
```

---

### Task 3: Port tblgen extraction to build helpers

**Files:**
- Create: `build_helpers/extract.rs`

This is the core extraction, adapted from `src/tablegen/native.rs`. Uses the `tblgen` crate API to parse `.td` files and produce `BuildInstrRecord` entries, plus processor model, itineraries, register model, and composite formats.

- [ ] **Step 1: Create extract.rs with parse_td_file() helper**

Port the `parse_td_file()` function from `native.rs` that creates a `tblgen::RecordKeeper`. This is the foundation all other extraction functions use.

- [ ] **Step 2: Port load_instruction_records()**

Port from `native.rs:79-350` (approximately). Extracts `BuildInstrRecord` entries from all slot instruction subclasses. Uses `tblgen::Record` API to read fields, encoding bits, flags, inputs/outputs, implicit defs/uses.

- [ ] **Step 3: Port load_pattern_records()**

Port from `native.rs`. Extracts `Pat<>` semantic mappings. Returns `HashMap<String, String>` (instr_name -> semantic_op_string).

- [ ] **Step 4: Port load_pseudo_expansion_map()**

Port from `native.rs`. Returns `HashMap<String, Vec<String>>` (pseudo -> concrete instruction names).

- [ ] **Step 5: Port load_processor_model()**

Port from `native.rs`. Returns a struct with load_latency, mispredict_penalty, etc. All scalar fields -- straightforward.

- [ ] **Step 6: Port load_itinerary_data()**

Port from `native.rs`. Returns `HashMap<String, BuildItineraryInfo>`.

- [ ] **Step 7: Port load_register_model()**

Port from `native.rs`. Returns `BuildRegisterModel` with registers and classes.

- [ ] **Step 8: Port load_composite_formats()**

Port from `native.rs`. Returns `Vec<BuildCompositeFormat>`.

- [ ] **Step 9: Create extract_all() orchestrator**

Combines all extraction functions + semantic propagation (pattern, pseudo, C++ switch, itinerary) into a single `BuildTblgenOutput`. This is the build-time equivalent of `build_tblgen_output()` in `src/tablegen/mod.rs:212-461`.

- [ ] **Step 10: Verify build succeeds**

Run: `cargo build 2>&1 | head -20`

- [ ] **Step 11: Commit**

```
feat(build): port tblgen extraction to build helpers
```

---

### Task 4: Port C++ switch parser and decoder bytecode parser

**Files:**
- Create: `build_helpers/cpp_switch.rs`
- Create: `build_helpers/bytecode.rs`

- [ ] **Step 1: Create cpp_switch.rs**

Port `parse_cpp_opcode_switch()` and `parse_switch_function()` from `src/tablegen/cpp_switch.rs:43-403`. These are pure string parsing (no crate dependencies). Can be nearly identical to the original.

- [ ] **Step 2: Create bytecode.rs**

Port the **parser** functions from `src/tablegen/decoder_bytecode.rs`:
- `extract_all_tables()` (lines 383-427)
- `parse_decoder_table_bytes()` (lines 315-358)
- `parse_opcode_names_from_disasm()` (lines 242-300)
- `match_mcd_opcode()` (lines 361-373)
- `find_table_end()` (lines 433-446)
- `read_uleb128()` (lines 203-216) -- needed by opcode name parser

These all produce a `BuildDecoderTable { bytes: Vec<u8>, opcode_names: HashMap<u32, String> }`. The runtime `DecoderTable` struct and `decode()` method stay in `src/tablegen/decoder_bytecode.rs`.

- [ ] **Step 3: Wire into extract_all()**

Update `build_helpers/extract.rs::extract_all()` to call `cpp_switch::parse_cpp_opcode_switch()` for semantic layer 3, and to run `llvm-tblgen -gen-disassembler` subprocess + `bytecode::extract_all_tables()` for decoder tables.

- [ ] **Step 4: Update build_helpers/mod.rs**

Add all module declarations.

- [ ] **Step 5: Verify build succeeds**

Run: `cargo build 2>&1 | head -20`

- [ ] **Step 6: Commit**

```
feat(build): port C++ switch and bytecode parsers to build helpers
```

---

## Chunk 2: Code Generation and Runtime Switch

### Task 5: Implement codegen (gen_tablegen.rs generation)

**Files:**
- Create: `build_helpers/codegen.rs`

This is the most complex new file. It takes a `BuildTblgenOutput` and writes a Rust source file containing a function `load_from_generated() -> TblgenOutput`.

- [ ] **Step 1: Create codegen.rs with formatting helpers**

Helper functions that format build types as Rust source strings:
- `format_encoding(enc: &BuildInstrEncoding) -> String` -- formats one `InstrEncoding { ... }` literal
- `format_operand_field(field: &BuildOperandField) -> String` -- formats one `OperandField { ... }`
- `format_field_fragment(frag: &BuildFieldFragment) -> String`
- `format_implicit_reg(reg: &BuildImplicitReg) -> String`
- `format_decoder_table(table: &BuildDecoderTable) -> String` -- formats bytecode as `vec![0x01, 0x02, ...]` and opcode names as `HashMap::from([...])`
- `format_processor_model(model: &BuildProcessorModel) -> String`
- `format_itinerary(name: &str, info: &BuildItineraryInfo) -> String`
- `format_register_model(model: &BuildRegisterModel) -> String`
- `format_composite_format(fmt: &BuildCompositeFormat) -> String`

- [ ] **Step 2: Implement generate_tablegen_file()**

Main entry point. Writes `gen_tablegen.rs` to `$OUT_DIR`. Structure:

```rust
pub fn generate_tablegen_file(output: &BuildTblgenOutput, out_dir: &Path) {
    let mut code = String::new();
    writeln!(code, "// Auto-generated by build.rs -- do not edit").unwrap();
    writeln!(code, "// Source: llvm-aie TableGen extraction at build time").unwrap();
    writeln!(code, "").unwrap();
    writeln!(code, "/// Load instruction decoder data from build-time extracted constants.").unwrap();
    writeln!(code, "pub(crate) fn load_from_generated() -> super::types::TblgenOutput {{").unwrap();
    writeln!(code, "    use super::types::*;").unwrap();
    writeln!(code, "    use super::resolver::*;").unwrap();
    writeln!(code, "    use super::decoder_bytecode::DecoderTable;").unwrap();
    writeln!(code, "    use std::collections::HashMap;").unwrap();
    // ... format all data ...
    writeln!(code, "}}").unwrap();
    fs::write(out_dir.join("gen_tablegen.rs"), code).unwrap();
}
```

- [ ] **Step 3: Verify the generated code compiles**

Temporarily add `include!()` in `src/tablegen/mod.rs` and call `load_from_generated()` from a test. Don't switch the runtime path yet.

- [ ] **Step 4: Commit**

```
feat(build): implement gen_tablegen.rs code generation
```

---

### Task 6: Wire build.rs to call extraction + codegen

**Files:**
- Modify: `build.rs` (add extraction call to `main()`)
- Modify: `src/tablegen/mod.rs` (add `include!()`)

- [ ] **Step 1: Add extraction call to build.rs main()**

After the existing slot confirmation block in `main()`, add:

```rust
// Full TableGen extraction for decoder tables (build-time)
let aie2_td = llvm_aie_path.join("llvm/lib/Target/AIE/AIE2.td");
if aie2_td.exists() {
    println!("cargo:rerun-if-changed={}", aie2_td.display());
    // Also track key .td files that affect instruction encodings
    for td in &["AIE2InstrFormats.td", "AIE2InstrInfo.td", "AIE2InstrPatterns.td",
                "AIE2Slots.td", "AIE2Schedule.td", "AIE2RegisterInfo.td"] {
        let p = llvm_aie_path.join(format!("llvm/lib/Target/AIE/{}", td));
        if p.exists() { println!("cargo:rerun-if-changed={}", p.display()); }
    }
    // Track C++ file for intrinsic switch parsing
    let cpp = llvm_aie_path.join("llvm/lib/Target/AIE/AIE2InstrInfo.cpp");
    if cpp.exists() { println!("cargo:rerun-if-changed={}", cpp.display()); }

    match build_helpers::extract_all(llvm_aie_path) {
        Ok(output) => {
            println!("cargo:warning=TableGen: extracted {} instructions across {} slots",
                output.total_instructions(), output.slot_count());
            build_helpers::codegen::generate_tablegen_file(&output, &out_dir);
        }
        Err(e) => {
            panic!("TableGen extraction failed:\n  {}\n\
                    Set LLVM_AIE_PATH to override.", e);
        }
    }
} else {
    panic!("llvm-aie not found at {} -- required for build-time TableGen extraction.\n\
            Set LLVM_AIE_PATH to override.", llvm_aie_path.display());
}
```

- [ ] **Step 2: Add include!() in src/tablegen/mod.rs**

```rust
// Build-time generated instruction tables
mod generated {
    include!(concat!(env!("OUT_DIR"), "/gen_tablegen.rs"));
}
```

- [ ] **Step 3: Build and verify generated code compiles**

Run: `cargo build 2>&1 | tail -5`
Expected: builds successfully. The generated module exists but isn't used by the runtime path yet.

- [ ] **Step 4: Commit**

```
feat(build): wire extraction pipeline into build.rs
```

---

### Task 7: Switch runtime to use generated data

**Files:**
- Modify: `src/tablegen/mod.rs`
- Modify: `src/interpreter/decode/decoder.rs`

- [ ] **Step 1: Add load_from_generated() wrapper in mod.rs**

```rust
/// Load the complete TableGen model from build-time generated constants.
///
/// This replaces `load_full_via_tblgen()`. All instruction encodings, decoder
/// bytecode, and metadata were extracted from llvm-aie at compile time.
/// No filesystem access, no subprocess, no llvm-aie required at runtime.
pub fn load_from_generated() -> types::TblgenOutput {
    generated::load_from_generated()
}
```

- [ ] **Step 2: Update decoder.rs to use generated data**

In `try_load_via_tblgen()` (or rename to `try_load()`), replace:
```rust
let output = crate::tablegen::load_full_via_tblgen(path)?;
```
with:
```rust
let output = crate::tablegen::load_from_generated();
```

Also update `load_fresh()` to remove the llvm-aie path requirement. The decoder no longer needs a filesystem path since all data is compiled in.

- [ ] **Step 3: Run tests to verify correctness**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1 | tail -20`
Expected: all ~1,573+ tests pass. The decoder now loads from generated constants.

- [ ] **Step 4: Run bridge test to verify end-to-end**

Run: `./scripts/emu-bridge-test.sh --no-hw add_one_using_dma 2>&1 | tail -10`
Expected: EMU PASS.

- [ ] **Step 5: Commit**

```
feat(tablegen): switch runtime to build-time generated decoder tables
```

---

## Chunk 3: Cleanup

### Task 8: Remove runtime tblgen dependency and dead code

**Files:**
- Delete: `src/tablegen/native.rs` (1281 lines)
- Delete: `src/tablegen/cpp_switch.rs` (403 lines)
- Delete: `src/tablegen/tblgen_records.rs` (348 lines)
- Modify: `src/tablegen/mod.rs` (remove dead imports, functions, module declarations)
- Modify: `src/tablegen/decoder_bytecode.rs` (remove parser functions that moved to build_helpers)
- Modify: `Cargo.toml` (remove `tblgen` from `[dependencies]`)
- Modify: `src/interpreter/decode/decoder.rs` (remove `try_load_via_tblgen()`, simplify loading)

- [ ] **Step 1: Remove native.rs, cpp_switch.rs, tblgen_records.rs**

Delete the three files. Remove their `mod` declarations and `pub use` re-exports from `src/tablegen/mod.rs`.

- [ ] **Step 2: Clean up decoder_bytecode.rs**

Remove the parser functions that moved to `build_helpers/bytecode.rs`:
- `extract_all_tables()`
- `parse_decoder_table_bytes()`
- `parse_opcode_names_from_disasm()`
- `match_mcd_opcode()`
- `find_table_end()`
- `read_uleb128()` (if only used by parser)

Keep: `DecoderTable` struct, `decode()` method, `field_from_instruction()`, `read_uleb128()` (if used by `decode()`), bytecode opcode constants, and all tests that test the interpreter (not the parser).

Note: `read_uleb128()` and `read_u24_le()` are used by BOTH the parser and the runtime interpreter (`decode()` method). Keep them. Only remove functions that are purely parser-side.

- [ ] **Step 3: Clean up mod.rs**

Remove:
- `load_full_via_tblgen()` function
- `load_via_tblgen()` function
- `build_tblgen_output()` function
- `run_gen_disassembler()` function
- `find_aie_tblgen()` function
- All cache functions (`compute_tblgen_cache_key`, `tblgen_cache_dir`, `try_load_cached_disasm`, `write_disasm_cache`)
- `sha2` dependency removal from Cargo.toml if it was only used by caching

Update module doc comment to reflect the new architecture.

- [ ] **Step 4: Simplify decoder.rs loading**

Replace `try_load_via_tblgen()` with a simpler `load()` that doesn't take a path:
```rust
pub fn load() -> Self {
    let output = crate::tablegen::load_from_generated();
    let format_table = /* same as before */;
    let mut decoder = Self::from_tables_with_decoders(
        output.encodings_by_slot,
        output.decoder_tables,
    );
    decoder.format_table = format_table;
    decoder
}
```

Update `load_fresh()` to call `Self::load()` without config/path logic.
Remove `is_llvm_aie_available()` if no longer needed.

- [ ] **Step 5: Remove tblgen from [dependencies]**

In `Cargo.toml`, remove the `tblgen = { ... }` line from `[dependencies]`. Keep it in `[build-dependencies]`.

Also check if `sha2` can be removed (was used only for tblgen cache key).

- [ ] **Step 6: Remove dirs dependency if only used by cache**

Check if `dirs` crate was only used by `tblgen_cache_dir()`. If so, remove from `[dependencies]`.

- [ ] **Step 7: Build and run full test suite**

Run: `cargo build 2>&1 | tail -5`
Run: `TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1 | tail -20`
Expected: builds clean (no LLVM linked in final binary), all tests pass.

- [ ] **Step 8: Verify binary size decreased**

Run: `ls -la target/debug/xdna-emu` (before and after -- LLVM's libTableGen.a is ~50MB static)
Expected: noticeable size decrease since LLVM is no longer linked.

- [ ] **Step 9: Commit**

```
refactor(tablegen): remove runtime tblgen dependency, delete 2000+ lines of dead code
```

---

### Task 9: Update tests that used runtime tblgen path

**Files:**
- Modify: `src/tablegen/mod.rs` (update remaining tests)
- Modify: `src/tablegen/decoder_bytecode.rs` (update integration test)

- [ ] **Step 1: Update mod.rs tests**

Tests like `test_acq_instruction_disambiguation`, `test_structural_semantic_inference`, etc. currently call `load_via_tblgen()`. Update them to use `load_from_generated()` instead. They no longer need the `if !llvm_aie_path.exists() { return; }` guard since the data is compiled in.

- [ ] **Step 2: Update decoder_bytecode integration test**

`test_real_mv_slot_decode` currently runs the llvm-tblgen subprocess. This test should either:
- Use the generated decoder tables (from `load_from_generated().decoder_tables`), or
- Stay as-is but be gated behind a feature flag for "live llvm-aie validation"

Prefer the first option: test the generated tables, which validates the full pipeline.

- [ ] **Step 3: Run full test suite**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1 | tail -20`
Expected: all tests pass, no tests skip with "llvm-aie not found".

- [ ] **Step 4: Commit**

```
test(tablegen): update tests to use build-time generated data
```

---

## Verification Checklist

After all tasks complete:

- [ ] `cargo build` succeeds with no warnings (except expected cargo:warning messages)
- [ ] `cargo test --lib` passes all tests (count should be >= current count minus deleted test functions)
- [ ] `./scripts/emu-bridge-test.sh --no-hw add_one_using_dma` passes
- [ ] `ldd target/debug/xdna-emu | grep -i llvm` returns nothing (no LLVM linked)
- [ ] Binary starts without requiring llvm-aie on disk
- [ ] `git diff --stat HEAD` shows net line reduction (~2000+ lines deleted from src/, ~3000+ added to build_helpers/)
