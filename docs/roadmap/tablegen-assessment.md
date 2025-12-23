W# TableGen Parsing Assessment

**Date**: 2025-12-23
**Status**: Assessment Complete
**Recommendation**: Feasible, but multi-phase approach recommended

---

## Executive Summary

Automatically generating instruction decoders from TableGen would be ideal - it ensures accuracy and completeness. However, TableGen is a complex DSL with deep LLVM integration. This assessment explores the options and recommends a phased approach.

**Bottom line**: Much simpler than initially thought! The llvm-aie `.td` files follow
regular patterns that can be parsed with regex. No tree-sitter needed.

**Estimated: 1 week.**

---

## What We're Dealing With

### AIE2 TableGen Files (from [Xilinx/llvm-aie](https://github.com/Xilinx/llvm-aie))

| File | Purpose | Complexity |
|------|---------|------------|
| `AIE2Slots.td` | VLIW slot definitions (7 slots, bit widths) | Low |
| `AIE2InstrFormats.td` | Instruction format classes | Medium |
| `AIE2GenInstrFormats.td` | Generated format definitions | High |
| `AIE2GenInstrInfo.td` | Concrete instruction definitions | High |
| `AIE2InstrInfo.td` | Additional instructions, pseudos | Medium |
| `AIE2RegisterInfo.td` | Register definitions | Low |

**Key insight**: Most actual encodings are in `AIE2GenInstr*.td` - these follow relatively regular patterns.

### TableGen Language Features We Need

| Feature | Needed? | Complexity | Notes |
|---------|---------|------------|-------|
| `class` definitions | Yes | Medium | Template definitions |
| `def` records | Yes | Low | Concrete instances |
| `defm` (multiclass) | Maybe | High | Bulk instantiation |
| `let` bindings | Yes | Medium | Field overrides |
| `bits<N>` fields | Yes | Low | Encoding specification |
| Field concatenation `{a,b,c}` | Yes | Medium | How encodings compose |
| Inheritance chains | Yes | High | Class hierarchy resolution |
| `foreach` loops | Maybe | Medium | Used in some expansions |
| Expressions | Partially | High | Arithmetic, string ops |

### Encoding Pattern Example

```tablegen
// From AIE2GenInstrFormats.td
class AIE2_add_r_ri_inst_alu : AIE2_inst_alu_instr32 {
  let alu = {mRx0, mRx, c7s, 0b11, 0b0};  // 5+5+7+2+1 = 20 bits
}

// From AIE2GenInstrInfo.td
def ADD_add_r_ri : AIE2_add_r_ri_inst_alu<
  (outs eR:$mRx), (ins eR:$mRx0, simm7:$c7s),
  "add", "$mRx, $mRx0, $c7s"
>;
```

What we need to extract:
- Instruction name: `ADD_add_r_ri`
- Slot: `alu` (20 bits)
- Encoding: `{mRx0[4:0], mRx[4:0], c7s[6:0], 0b11, 0b0}`
- Operands: `eR` dest, `eR` src, `simm7` immediate

---

## Approach Options

### Option 1: tblgen-rs (LLVM FFI Bindings)

[tblgen-rs](https://github.com/mlir-rs/tblgen-rs) wraps LLVM's TableGen library.

**Pros:**
- Full semantic processing (inheritance, multiclass, expressions)
- Uses battle-tested LLVM code
- Guaranteed correct interpretation

**Cons:**
- Requires LLVM 16-21 installed with AIE support
- Heavy dependency (~100MB+)
- Complex build setup
- Ties us to LLVM versions

**Effort**: 1-2 weeks (if LLVM already available), 3-4 weeks otherwise

### Option 2: tree-sitter-tablegen (AST Parser)

[tree-sitter-tablegen](https://docs.rs/tree-sitter-tablegen) provides syntax parsing.

**Pros:**
- Pure Rust, no LLVM dependency
- Gives us a full AST to work with
- Fast, incremental parsing
- ~100KB dependency

**Cons:**
- Syntax only, no semantic resolution
- We implement inheritance/let bindings ourselves
- Limited to patterns we explicitly handle

**Effort**: 2-3 weeks for useful coverage

### Option 3: Custom Parser

Roll our own TableGen parser.

**Pros:**
- Full control
- Optimized for our needs
- No dependencies

**Cons:**
- TableGen is complex (classes, multiclass, expressions, etc.)
- Reinventing the wheel
- Edge cases everywhere

**Effort**: 4-8 weeks

### Option 4: Build llvm-aie, Extract Generated Tables

Build the compiler, use the generated `.inc` files.

**Pros:**
- Decoder tables are exactly what LLVM uses
- Guaranteed complete and correct

**Cons:**
- Complex build (llvm-aie requires specific toolchain)
- Parse C++ code or write custom TableGen backend
- Maintenance burden

**Effort**: 2-3 weeks (assuming successful build)

### Option 5: Continue Manual (Current Approach)

Keep pattern-based decoder, use TableGen as reference.

**Pros:**
- Works today
- Incrementally improvable
- No new dependencies

**Cons:**
- Manual effort per instruction
- May miss edge cases
- No guarantee of completeness

**Effort**: Ongoing

---

## Revised Assessment: It's Actually Simple!

After examining the actual llvm-aie files, the patterns are **highly regular**:

```tablegen
// Format class (AIE2GenInstrFormats.td):
class AIE2_alu_r_rr_inst_alu<bits<4> op, ...> : AIE2_inst_alu_instr32 {
  bits<5> mRx, mRx0, mRy;
  let alu = {mRx0, mRx, mRy, op, 0b1};  // 5+5+5+4+1 = 20 bits
}

// Instruction defs (AIE2GenInstrInfo.td):
def ADD : AIE2_alu_r_rr_inst_alu<0b0000, ...>;  // op=0000
def SUB : AIE2_alu_r_rr_inst_alu<0b0001, ...>;  // op=0001
def AND : AIE2_alu_r_rr_inst_alu<0b0100, ...>;  // op=0100
```

This can be parsed with **regex + a simple state machine**. No tree-sitter needed!

---

## Recommended Approach: Regex-Based Extraction

### Phase 1: Parse Format Classes (2-3 days)

Parse `AIE2GenInstrFormats.td` with regex to extract:

```rust
// Key regex patterns:
// 1. Slot defs: def (\w+)\s*:\s*InstSlot<"(\w+)",\s*(\d+)>
// 2. Class defs: class (AIE2_\w+).*{
// 3. Field defs: bits<(\d+)>\s+(\w+);
// 4. Encoding: let\s+(\w+)\s*=\s*\{([^}]+)\};
// 5. Instr defs: def\s+(\w+)\s*:\s*(AIE2_\w+)<([^>]*)

struct SlotDef {
    name: String,      // "alu_slot"
    bits: u8,          // 20
    field: String,     // "alu"
}

struct FormatClass {
    name: String,           // "AIE2_alu_r_rr_inst_alu"
    parent_slot: String,    // "alu"
    template_params: Vec<(String, u8)>,  // [("op", 4)]
    fields: Vec<(String, u8)>,           // [("mRx", 5), ("mRx0", 5), ...]
    encoding: Vec<EncodingPart>,         // Parsed from let slot = {...}
}

struct InstrDef {
    name: String,           // "ADD"
    format: String,         // "AIE2_alu_r_rr_inst_alu"
    template_args: Vec<u64>, // [0b0000]
    mnemonic: String,       // "add"
}
```

### Phase 2: Parse Instruction Defs (1-2 days)

Parse `AIE2GenInstrInfo.td` to get instruction→format mappings.

### Phase 3: Generate Decoder Tables (2-3 days)

Combine format classes + instruction defs to generate Rust decode tables:

```rust
// Auto-generated: src/interpreter/decode/generated.rs
pub const ALU_SLOT_INSTRS: &[InstrEncoding] = &[
    InstrEncoding {
        name: "ADD",
        mnemonic: "add",
        // From: let alu = {mRx0, mRx, mRy, op, 0b1} with op=0b0000
        // Bits: mRx0[19:15], mRx[14:10], mRy[9:5], op[4:1]=0000, [0]=1
        fixed_mask: 0b0000_0000_0000_0001_1111,  // bits that must match
        fixed_bits: 0b0000_0000_0000_0000_0001,  // expected values
        operand_fields: &[
            ("mRx0", 15, 5),  // dest at bits 19:15
            ("mRx", 10, 5),   // src1 at bits 14:10
            ("mRy", 5, 5),    // src2 at bits 9:5
        ],
    },
    // ... hundreds more
];
```

---

## Implementation Plan

### Files to Create

```
src/tablegen/
├── mod.rs              # Module exports, public API
├── parser.rs           # Regex-based .td file parsing
├── types.rs            # SlotDef, FormatClass, InstrDef
├── resolver.rs         # Resolve format→encoding mappings
└── codegen.rs          # Generate Rust decoder tables
```

### External Dependency

llvm-aie clone at `../llvm-aie` (already done):

```
/home/triple/npu-work/
├── llvm-aie/           # Cloned: github.com/Xilinx/llvm-aie
│   └── llvm/lib/Target/AIE/
│       ├── AIE2Slots.td
│       ├── AIE2GenInstrFormats.td
│       └── AIE2GenInstrInfo.td
└── xdna-emu/           # This project
```

### No New Rust Dependencies

Just use `regex` (already in tree via other deps) or simple string parsing.

---

## Risk Assessment

| Risk                            | Likelihood | Impact | Mitigation                              |
|---------------------------------|------------|--------|-----------------------------------------|
| Regex misses edge cases         | Medium     | Low    | Fall back to manual patterns            |
| Complex inheritance chains      | Low        | Medium | Most formats are simple single-parent   |
| TableGen files change upstream  | Low        | Low    | Pin to specific llvm-aie commit         |
| Encoding extraction errors      | Medium     | High   | Test against real binaries + objdump    |

---

## Timeline

| Phase                     | Days  | Deliverable                           |
|---------------------------|-------|---------------------------------------|
| Phase 1: Parse formats    | 2-3   | FormatClass structs from .td files    |
| Phase 2: Parse instrs     | 1-2   | InstrDef structs with format refs     |
| Phase 3: Generate tables  | 2-3   | Rust decode tables in generated.rs    |
| **Total**                 | **~7**| Complete decoder table generation     |

---

## Decision

**Verdict: Go for it!**

The patterns are regular enough that this is straightforward engineering, not research.
Having llvm-aie as a dev dependency makes sense - this tool exists in the NPU ecosystem.

**Next step**: Start implementing `src/tablegen/parser.rs`.

---

## References

- [Xilinx/llvm-aie](https://github.com/Xilinx/llvm-aie) - AIE LLVM backend (cloned locally)
- [LLVM TableGen Overview](https://llvm.org/docs/TableGen/) - Language reference
- [LLVM TableGen Backends](https://llvm.org/docs/TableGen/BackEnds.html) - How LLVM uses TableGen

### Key Files in llvm-aie

| File | Size | Purpose |
|------|------|---------|
| `AIE2Slots.td` | 5KB | Slot definitions (lda, ldb, alu, mv, st, vec, lng) |
| `AIE2GenInstrFormats.td` | 39KB | Format classes with encoding patterns |
| `AIE2GenInstrInfo.td` | 32KB | Instruction definitions |
| `AIE2RegisterInfo.td` | 3KB | Register class definitions |
