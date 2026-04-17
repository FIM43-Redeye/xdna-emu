# Design: Native TableGen Integration via `tblgen` Crate

## Problem

The emulator's instruction decoder has two critical gaps caused by incomplete
TableGen data extraction:

1. **VLIW format extraction fails for most bundle sizes.** Composite format
   definitions (I80_LDA_ALU_VEC, etc.) use multi-level field hierarchies in
   their `Inst` fields (`Inst -> instr80 -> inst_lda_alu_vec -> {lda, alu,
   vec}`). Our text parser (`tblgen_records.rs`) only resolves one level,
   producing format table entries with empty slot maps. The hand-coded
   fallback (`extract_80bit`, etc.) covers ~70% of patterns but misses
   others (e.g., `bits[1:0] = 0b11` formats). Result: bundles like
   `mova + movx + vbcst.8` fail to extract slots, crashing Peano-compiled
   tests.

2. **Instruction coverage is 35%.** The decoder index uses LLVM's
   `-gen-disassembler` bytecode tables, which include only ~210 of 607
   valid instruction encodings. LLVM's disassembler is selective (picks one
   encoding when multiples overlap, skips some variants). Instructions like
   `MOVX_alu_cg` (a basic `movx` immediate) aren't in the disassembler
   tables despite being valid hardware instructions.

## Solution: Link to LLVM's TableGen Library

Use the [`tblgen`](https://github.com/mlir-rs/tblgen-rs) Rust crate (safe
bindings to LLVM's TableGen C API) to parse `.td` files directly in-process.
LLVM's own resolver handles all inheritance, template substitution, and field
hierarchy resolution. We read fully resolved records as Rust objects -- no
subprocess, no text parsing, no JSON intermediary.

### Why This Approach

- **Tightest possible coupling** to the source of truth. Same parser LLVM
  uses internally.
- **Architecture-agnostic.** Point at `AIE2.td`, `AIE2P.td`, or any other
  LLVM target's `.td` files. No hardcoded architecture assumptions.
- **Full coverage.** Access every record LLVM knows about -- all 607
  instructions, all 78 composite formats, all 278 scheduling itineraries,
  all 317 intrinsics.
- **In-process.** No 29MB JSON blob, no disk cache, no subprocess at init
  time.
- **Proven crate.** 386K downloads, actively maintained, supports LLVM
  16-21, custom LLVM paths via `TABLEGEN_<version>_PREFIX`.

### Build Dependency

Requires linking against LLVM's TableGen library. We already build llvm-aie
locally. The `tblgen` crate accepts custom LLVM installations via
`TABLEGEN_<version>_PREFIX` environment variable, which we point at our
llvm-aie build directory.

## Required Patch: `VarBitInit` Support

The `tblgen` crate has a bug in its C++ wrapper: `BitsInit::getBit(i)`
returns an `Init*` which may be a `VarBitInit` (variable reference like
`lda{17}`), not a `BitInit` (literal 0/1). The wrapper does an unsafe
`reinterpret_cast<const BitInit*>` that produces garbage for variable
references.

**Fork:** `FIM43-Redeye/tblgen-rs`, branch `feat/varbit-init`

**Patch scope:** Add three C++ functions and corresponding Rust API:

```cpp
// cc/lib/Utility.cpp additions:
TableGenBool tableGenBitInitIsVarBit(TableGenTypedInitRef ti);
TableGenStringRef tableGenVarBitInitGetVarName(TableGenTypedInitRef ti);
size_t tableGenVarBitInitGetBitNum(TableGenTypedInitRef ti);
```

```rust
// Rust API additions to BitInit:
impl BitInit<'_> {
    pub fn is_var_bit(&self) -> bool;
    pub fn as_var_bit(&self) -> Option<(&str, usize)>;  // (field_name, bit_index)
    pub fn as_literal(&self) -> Option<bool>;
}
```

This exposes existing LLVM API (`VarBitInit::getBitVar()`,
`VarBitInit::getBitNum()`) that the crate simply missed. Clean,
upstreamable patch.

## Architecture

### New Module: `src/tablegen/native.rs`

Primary entry point replacing `parser.rs` and `tblgen_records.rs` as the
data source. Populates the existing internal types (`TableGenModel`,
`InstrEncoding`, `CompositeFormatDef`, etc.).

```rust
pub fn load_native(td_file: &Path, include_paths: &[&Path])
    -> Result<TableGenModel, Error>
{
    let keeper = TableGenParser::new()
        .add_include_paths(include_paths)
        .add_source_file(td_file)
        .parse()?;
    TableGenModel::from_record_keeper(&keeper)
}
```

### Composite Format Extraction

For each record that is a subclass of `AIE2CompositeInst`:

```rust
let inst = record.value("Inst");  // BitsInit with 80/96/112/128 bits
for i in 0..inst.num_bits() {
    let bit = inst.bit(i);
    if let Some((var_name, var_bit)) = bit.as_var_bit() {
        // Variable reference: record slot position
        // e.g., var_name="lda", var_bit=17, word_bit_pos=i
        slot_maps[var_name].push((var_bit, i));
    } else if let Some(val) = bit.as_literal() {
        // Fixed bit: add to discriminator mask
        fixed_mask |= 1 << i;
        if val { fixed_value |= 1 << i; }
    }
    // else: don't-care bit (not in mask)
}
```

No hierarchy chasing. LLVM already resolved `instr80 -> inst_lda_alu_vec ->
{lda, alu, vec}` down to leaf `VarBitInit` references. We just read them.

### Instruction Encoding Extraction

For each record that is a subclass of `AIE2Inst` or `AIE2SlotInst`, with
`isCodeGenOnly = 0` and `isComposite = 0`:

- `bits_value("Inst")` -- fully resolved encoding (concrete `BitInit` values)
- `int_value("Size")` -- instruction size in bytes
- Slot field name (`alu`, `mv`, `lda`, etc.) -- determines decoder slot
- Operand fields from `values()` iterator -- register classes, immediates

This produces all 607 valid encodings, replacing both `--print-records` and
`-gen-disassembler` as the encoding source. We build our `DecoderIndex`
from the full set.

### Additional Data Available

The same `RecordKeeper` gives us access to everything else in TableGen:

| Data | Record Class | Count | Current Source |
|------|-------------|-------|---------------|
| Composite formats | `AIE2CompositeInst` | 78 | Hand-coded + broken data-driven |
| Instruction encodings | `AIE2Inst` / `AIE2SlotInst` | 607 | `-gen-disassembler` (210) |
| Scheduling itineraries | `InstrItinClass` | 278 | Partially hardcoded |
| Register definitions | Physical registers | ~50 | Hardcoded in `state.rs` |
| Register classes | `RegisterClass` | 30+ | Hardcoded |
| Intrinsic signatures | `Intrinsic` | 317 | Not extracted |
| Calling conventions | `CallingConv` | 3 | Not used |
| Slot definitions | `AIE2SlotKind` | 8 | Hardcoded in `aie2_spec.rs` |

### Architecture Parameterization

`load_native()` takes a path to the top-level `.td` file. The internal
model doesn't know which architecture it loaded:

- `AIE2.td` -- Phoenix/Hawk Point (current target)
- `AIE2P.td` -- Strix Point (XDNA2, future)
- `aie1/AIE1.td` -- Versal (potential future)

Slot names, format patterns, instruction encodings all come from the data.
No hardcoded architecture assumptions in the resolution logic.

## Migration Path

### Phase 1: Foundation (This Work)

- Patch `tblgen` crate fork with `VarBitInit` support
- Add `src/tablegen/native.rs` loader
- Wire into `FormatTable` (composite format extraction)
- Wire into `DecoderIndex` (full instruction coverage)
- Old paths remain as fallback behind feature flag
- Verify parity: all existing tests pass, Peano `movx` decodes

### Phase 2: Scheduling Model

Extract 278 itineraries and 24+ functional units to drive cycle-accurate
timing. Replace hardcoded latency values.

### Phase 3: Register Model

Extract register definitions and classes to replace hardcoded register file
definitions in `state.rs` and composite decoder LUTs.

### Phase 4: Intrinsic Signatures

Extract 317 intrinsic signatures for vector compute validation. Use
configuration word bit layouts for VMAC/VMUL semantics.

### Phase 5: Sunset Legacy Paths

Remove `parser.rs`, `tblgen_records.rs`, `--print-records`/`--dump-json`
paths once native is proven at 100% parity.

## Files to Create/Modify

| File | Action |
|------|--------|
| `Cargo.toml` | Add `tblgen` dependency (git, our fork) |
| `src/tablegen/native.rs` | **NEW** -- native loader |
| `src/tablegen/mod.rs` | Add `native` module, new `load_native()` entry point |
| `src/interpreter/bundle/slot_layout.rs` | `FormatTable` populated from native data |
| `src/interpreter/decode/decoder.rs` | `DecoderIndex` populated from native data |
| `/tmp/claude-1000/tblgen-rs/` | Fork patches for `VarBitInit` |

## Verification

1. `cargo test --lib` -- all existing 1636+ tests pass
2. Peano `packet_flow` test decodes `movx r18, #0x0` successfully
3. Format table covers all 78 composite formats (vs. current ~0 effective)
4. Decoder index covers all 607 instructions (vs. current ~210)
5. Bridge test score improves (Peano tests that failed on decoder gaps)

## Key Design Decisions

1. **`tblgen` crate over `--dump-json`**: In-process beats subprocess. JSON
   was the backup plan; the crate is the real solution.

2. **Patch the crate, don't work around it**: The `VarBitInit` gap is a bug,
   not a design limitation. Fix it properly, contribute upstream.

3. **Old paths as fallback, not replacement**: Keep `--print-records` and
   `-gen-disassembler` paths alive during migration. Remove only after
   verified parity.

4. **Architecture-agnostic from day one**: The loader takes a `.td` path,
   not an architecture enum. No AIE2-specific logic in the resolution.
