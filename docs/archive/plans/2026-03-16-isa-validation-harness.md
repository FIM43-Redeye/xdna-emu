# ISA-Level Validation Harness Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test every AIE2 compute instruction by generating assembly directly from TableGen, assembling with llvm-mc, and comparing HW vs EMU outputs.

**Architecture:** A Rust exporter serializes the emulator's existing TableGen ISA data to JSON. A Python generator reads this JSON and produces straight-line AIE2 assembly mega-programs that exercise each instruction with valid operands. llvm-mc assembles them in microseconds; aiecc.py packages into xclbins. A runner script sends identical inputs to HW and EMU, diffs outputs per-instruction.

**Tech Stack:** Rust (TableGen exporter), Python 3.13 (generator), llvm-mc (assembler), aiecc.py (packager), bash (runner)

**Spec:** `docs/superpowers/specs/2026-03-16-isa-validation-harness-design.md`

---

## File Structure

| File | Change | Responsibility |
|------|--------|---------------|
| `src/tablegen/resolver.rs` | Modify | Add `asm_string` field to `InstrEncoding` |
| `src/bin/export_isa.rs` | Create | CLI binary that loads TableGen, exports JSON |
| `tools/aie2-isa.json` | Create (generated) | Exported ISA metadata |
| `tools/isa-test-gen.py` | Create | Assembly mega-program generator |
| `tools/test_isa_test_gen.py` | Create | Unit tests for the generator |
| `scripts/isa-test.sh` | Create | Runner script (generate/assemble/link/run/compare) |

---

## Chunk 1: TableGen Export

### Task 1: Add asm_string to InstrEncoding and Export to JSON

**Files:**
- Modify: `src/tablegen/resolver.rs` (add `asm_string` field)
- Create: `src/bin/export_isa.rs`
- Create: `tools/aie2-isa.json` (generated output)

- [ ] **Step 1: Add asm_string field to the build-time and runtime pipelines**

The TableGen data flows through a **build-time code generation pipeline**:
`build_helpers/extract.rs` -> `build_helpers/records.rs` (BuildInstrEncoding)
-> `build_helpers/codegen.rs` -> generated Rust -> `src/tablegen/resolver.rs`
(InstrEncoding at runtime).

The `asm_string` is already extracted from TableGen records in
`build_helpers/extract.rs:175` and stored in `BuildInstrRecord`, but it
does NOT flow into `BuildInstrEncoding` or the generated code. Add it
to all three layers:

**a) `build_helpers/records.rs`**: Add `pub asm_string: String` to
`BuildInstrEncoding` struct (after `mnemonic`).

**b) `build_helpers/extract.rs`**: Where `BuildInstrEncoding` is
constructed (around line 285 and 1157+), propagate `asm_string` from
the `BuildInstrRecord`. The field is already extracted at line 175.

**c) `build_helpers/codegen.rs`**: In `format_encoding()` (line 137+),
add a `writeln!` for `asm_string` so it gets emitted into the generated
Rust code. Follow the existing pattern for string fields like `mnemonic`.

**d) `src/tablegen/resolver.rs`**: Add `pub asm_string: String` to
`InstrEncoding` (after `mnemonic`). The generated code will now populate
it at build time. Also add it to `resolve_instruction` for the runtime
resolver path: `asm_string: instr.asm_string.clone()`.

**e) Test construction sites**: Search for `InstrEncoding { ... }`
literals in test code and add `asm_string: String::new()` or an
appropriate value.

- [ ] **Step 2: Verify the emulator still builds**

Run: `cargo build --lib`
Expected: Build succeeds (may need to fix test sites that construct InstrEncoding)

- [ ] **Step 3: Create the export binary**

Create `src/bin/export_isa.rs`:

```rust
//! Export AIE2 ISA metadata to JSON for the ISA validation harness.
//!
//! Usage: cargo run --bin export_isa -- --output tools/aie2-isa.json

use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

// Import from the emulator's tablegen module.
use xdna_emu::tablegen;

#[derive(Serialize)]
struct ExportedInstruction {
    name: String,
    mnemonic: String,
    asm_string: String,
    slot: String,
    width: u8,
    operands: Vec<ExportedOperand>,
    may_load: bool,
    may_store: bool,
    is_vector: bool,
    has_complete_decoder: bool,
    sched_class: Option<String>,
}

#[derive(Serialize)]
struct ExportedOperand {
    name: String,
    bit_width: u8,
    is_output: bool,
    operand_type: String,     // "register", "immediate", "lock_id", "unknown"
    register_kind: Option<String>,  // "scalar", "pointer", "vector256", "vector512", "accumulator", etc.
    signed: Option<bool>,     // for immediates
    scale: Option<i32>,       // for immediates
    base_offset: Option<u8>,  // for RegisterWithOffset
}

#[derive(Serialize)]
struct ExportedISA {
    instructions: Vec<ExportedInstruction>,
}

fn export_operand(field: &tablegen::resolver::OperandField) -> ExportedOperand {
    use tablegen::resolver::{OperandType, RegisterKind};

    let (op_type, reg_kind, signed, scale, base_offset) = match &field.operand_type {
        OperandType::Register(kind) => {
            ("register".to_string(), Some(format_register_kind(kind)), None, None, None)
        }
        OperandType::RegisterWithOffset(kind, offset) => {
            ("register".to_string(), Some(format_register_kind(kind)), None, None, Some(*offset))
        }
        OperandType::CompositeRegister(_) => {
            ("composite_register".to_string(), None, None, None, None)
        }
        OperandType::Immediate { signed: s, scale: sc } => {
            ("immediate".to_string(), None, Some(*s), Some(*sc), None)
        }
        OperandType::LockId => {
            ("lock_id".to_string(), None, None, None, None)
        }
        OperandType::Unknown => {
            ("unknown".to_string(), None, None, None, None)
        }
    };

    ExportedOperand {
        name: field.name.clone(),
        bit_width: field.width,
        is_output: field.is_output,
        operand_type: op_type,
        register_kind: reg_kind,
        signed,
        scale,
        base_offset,
    }
}

fn format_register_kind(kind: &tablegen::resolver::RegisterKind) -> String {
    use tablegen::resolver::RegisterKind;
    match kind {
        RegisterKind::Scalar => "scalar",
        RegisterKind::Pointer => "pointer",
        RegisterKind::ModifierM => "modifier_m",
        RegisterKind::ModifierDN => "modifier_dn",
        RegisterKind::ModifierDJ => "modifier_dj",
        RegisterKind::ModifierDC => "modifier_dc",
        RegisterKind::Vector256 => "vector256",
        RegisterKind::Vector512 => "vector512",
        RegisterKind::Accumulator => "accumulator",
        RegisterKind::Control => "control",
    }.to_string()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let output_path = if let Some(idx) = args.iter().position(|a| a == "--output") {
        PathBuf::from(&args[idx + 1])
    } else {
        PathBuf::from("tools/aie2-isa.json")
    };

    let tblgen = tablegen::load_from_generated();

    let mut instructions = Vec::new();
    for (slot, encodings) in &tblgen.encodings_by_slot {
        for enc in encodings {
            instructions.push(ExportedInstruction {
                name: enc.name.clone(),
                mnemonic: enc.mnemonic.clone(),
                asm_string: enc.asm_string.clone(),
                slot: slot.clone(),
                width: enc.width,
                operands: enc.operand_fields.iter().map(export_operand).collect(),
                may_load: enc.may_load,
                may_store: enc.may_store,
                is_vector: enc.is_vector,
                has_complete_decoder: enc.has_complete_decoder,
                sched_class: enc.sched_class.clone(),
            });
        }
    }

    // Sort by name for deterministic output.
    instructions.sort_by(|a, b| a.name.cmp(&b.name));

    let isa = ExportedISA { instructions };
    let json = serde_json::to_string_pretty(&isa).expect("JSON serialization failed");
    fs::write(&output_path, json).expect("Failed to write output file");

    println!("Exported {} instructions to {}", instructions.len(), output_path.display());
}
```

- [ ] **Step 4: Add serde dependency if not present**

Check `Cargo.toml` for `serde` and `serde_json`. If missing:

```bash
cargo add serde --features derive
cargo add serde_json
```

If already present, skip this step.

- [ ] **Step 5: Build and run the exporter**

```bash
cargo build --bin export_isa 2>&1 | tail -5
cargo run --bin export_isa -- --output tools/aie2-isa.json
```

Expected: prints "Exported N instructions to tools/aie2-isa.json" where N is 200+.

- [ ] **Step 6: Verify the JSON is well-formed and has expected data**

```bash
python3 -c "
import json
isa = json.load(open('tools/aie2-isa.json'))
instrs = isa['instructions']
print(f'Total instructions: {len(instrs)}')
# Count by slot
from collections import Counter
slots = Counter(i['slot'] for i in instrs)
for slot, count in slots.most_common():
    print(f'  {slot}: {count}')
# Spot-check a known instruction
adds = [i for i in instrs if i['mnemonic'] == 'add' and i['slot'] == 'alu']
if adds:
    import json as j
    print(f'\\nSample (add): {j.dumps(adds[0], indent=2)[:500]}')
"
```

Expected: 200+ instructions across alu, vec, lda, ldb, sts slots. The
`add` instruction should have mnemonic="add", operands with scalar
register types.

- [ ] **Step 7: Run emulator tests to confirm no regression**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/tablegen/resolver.rs src/bin/export_isa.rs tools/aie2-isa.json Cargo.toml Cargo.lock
git commit -m "feat(isa-test): export AIE2 ISA metadata to JSON for validation harness"
```

---

## Chunk 2: Assembly Generator

### Task 2: Python Generator -- Instruction Classification and Operand Mapping

**Files:**
- Create: `tools/isa-test-gen.py`
- Create: `tools/test_isa_test_gen.py`

The generator reads `aie2-isa.json` and classifies each instruction as
testable or skipped. For testable instructions, it maps operand types to
concrete register names for assembly generation.

- [ ] **Step 1: Write failing tests for instruction classification**

Create `tools/test_isa_test_gen.py`:

```python
"""Unit tests for isa-test-gen.py (ISA validation harness generator)."""

import pytest
import importlib

isa_gen = importlib.import_module("isa-test-gen")


class TestInstructionClassification:
    def test_classify_alu_add(self):
        """Pure ALU instruction is testable."""
        instr = {
            "name": "ADD", "mnemonic": "add", "slot": "alu",
            "asm_string": "$mRx, $mRx0, $mRy",
            "may_load": False, "may_store": False,
            "is_vector": False, "has_complete_decoder": True,
            "operands": [
                {"name": "mRx", "is_output": True, "operand_type": "register",
                 "register_kind": "scalar", "bit_width": 5},
                {"name": "mRx0", "is_output": False, "operand_type": "register",
                 "register_kind": "scalar", "bit_width": 5},
                {"name": "mRy", "is_output": False, "operand_type": "register",
                 "register_kind": "scalar", "bit_width": 5},
            ],
        }
        status, reason = isa_gen.classify_instruction(instr)
        assert status == "testable"

    def test_skip_load_instruction(self):
        """Load instructions need valid addresses -- skip."""
        instr = {
            "name": "LDA_ri", "mnemonic": "lda", "slot": "lda",
            "may_load": True, "may_store": False,
            "operands": [],
        }
        status, reason = isa_gen.classify_instruction(instr)
        assert status == "skipped"
        assert "load" in reason

    def test_skip_store_instruction(self):
        instr = {
            "name": "ST_ri", "mnemonic": "st", "slot": "sts",
            "may_load": False, "may_store": True,
            "operands": [],
        }
        status, reason = isa_gen.classify_instruction(instr)
        assert status == "skipped"

    def test_skip_composite_register(self):
        """Composite registers need special encoding -- skip for now."""
        instr = {
            "name": "SOME_INSTR", "mnemonic": "foo", "slot": "alu",
            "may_load": False, "may_store": False,
            "is_vector": False, "has_complete_decoder": True,
            "operands": [
                {"name": "op", "is_output": True, "operand_type": "composite_register",
                 "register_kind": None, "bit_width": 5},
            ],
        }
        status, reason = isa_gen.classify_instruction(instr)
        assert status == "skipped"
        assert "composite" in reason

    def test_classify_vector_vadd(self):
        """Vector ALU instruction is testable."""
        instr = {
            "name": "VADD_32", "mnemonic": "vadd.32", "slot": "vec",
            "asm_string": "$x0, $x1, $x2",
            "may_load": False, "may_store": False,
            "is_vector": True, "has_complete_decoder": True,
            "operands": [
                {"name": "x0", "is_output": True, "operand_type": "register",
                 "register_kind": "vector512", "bit_width": 4},
                {"name": "x1", "is_output": False, "operand_type": "register",
                 "register_kind": "vector512", "bit_width": 4},
                {"name": "x2", "is_output": False, "operand_type": "register",
                 "register_kind": "vector512", "bit_width": 4},
            ],
        }
        status, reason = isa_gen.classify_instruction(instr)
        assert status == "testable"


class TestOperandMapping:
    def test_scalar_register_names(self):
        """Scalar register kind maps to r0, r15, r31."""
        names = isa_gen.register_names("scalar")
        assert "r0" in names
        assert len(names) >= 2

    def test_vector512_register_names(self):
        """Vector 512-bit maps to x0, x2, etc."""
        names = isa_gen.register_names("vector512")
        assert "x0" in names

    def test_vector256_register_names(self):
        """Vector 256-bit maps to wl0, wh0, etc."""
        names = isa_gen.register_names("vector256")
        assert any("wl" in n or "wh" in n for n in names)

    def test_accumulator_register_names(self):
        """Accumulator maps to cm0, cm2, etc."""
        names = isa_gen.register_names("accumulator")
        assert "cm0" in names

    def test_pointer_register_names(self):
        """Pointer maps to p0, p2, etc."""
        names = isa_gen.register_names("pointer")
        assert "p0" in names

    def test_immediate_values(self):
        """Immediate operand generates boundary values."""
        vals = isa_gen.immediate_values(bit_width=5, signed=True)
        assert 0 in vals
        assert 1 in vals
        assert -1 in vals

    def test_unsigned_immediate_values(self):
        vals = isa_gen.immediate_values(bit_width=3, signed=False)
        assert 0 in vals
        assert 7 in vals  # 2^3 - 1
        assert -1 not in vals
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_isa_test_gen.py -v
```

Expected: ImportError (module not created yet)

- [ ] **Step 3: Implement classification and operand mapping**

Create `tools/isa-test-gen.py`:

```python
#!/usr/bin/env python3
"""Generate ISA-level validation test programs from TableGen metadata.

Reads aie2-isa.json (exported by the emulator's TableGen exporter) and
produces straight-line AIE2 assembly mega-programs.  Each mega-program
tests ~80 instructions by loading input registers from a buffer,
executing one instruction, and storing output registers.

The assembly is fed to llvm-mc for instant assembly, then linked via
aiecc.py into xclbin binaries for HW vs EMU comparison.
"""

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Register name tables
# ---------------------------------------------------------------------------

_REGISTER_NAMES: dict[str, list[str]] = {
    "scalar": ["r0", "r1", "r15"],
    "pointer": ["p2", "p3"],
    "vector256": ["wl0", "wl2", "wh4"],
    "vector512": ["x0", "x2", "x4"],
    "accumulator": ["cm0", "cm2"],
    "modifier_m": ["m0", "m2"],
    "modifier_dn": ["dn0", "dn2"],
    "modifier_dj": ["dj0", "dj2"],
    "modifier_dc": ["dc0", "dc2"],
    "control": ["crRnd", "crSat"],
}

# Registers reserved for load/store boilerplate (never used as operands).
# p0 = input buffer base, p1 = output buffer base.
_RESERVED_REGISTERS = {"p0", "p1"}


def register_names(kind: str) -> list[str]:
    """Return representative register names for a given register kind."""
    return _REGISTER_NAMES.get(kind, [])


def immediate_values(bit_width: int, signed: bool) -> list[int]:
    """Return boundary test values for an immediate field."""
    if signed:
        max_val = (1 << (bit_width - 1)) - 1
        min_val = -(1 << (bit_width - 1))
        return [0, 1, -1, max_val, min_val]
    else:
        max_val = (1 << bit_width) - 1
        return [0, 1, max_val]


# ---------------------------------------------------------------------------
# Instruction classification
# ---------------------------------------------------------------------------

# Mnemonics that are branches/jumps (control flow, not compute).
_BRANCH_MNEMONICS = frozenset({
    "jl", "jal", "j", "jnz", "jz", "ret", "call",
    "bnez", "beqz", "bge", "blt", "bgeu", "bltu",
})


def classify_instruction(instr: dict) -> tuple[str, str]:
    """Decide whether an instruction is testable with this harness.

    Returns ("testable", "") or ("skipped", reason).
    """
    # Pseudo-instructions cannot be assembled by llvm-mc.
    if instr.get("is_pseudo"):
        return ("skipped", "pseudo-instruction")

    # Load/store instructions need valid memory addresses.
    if instr.get("may_load"):
        return ("skipped", "load instruction")
    if instr.get("may_store"):
        return ("skipped", "store instruction")

    # Branch/jump instructions are control flow.
    if instr.get("mnemonic", "") in _BRANCH_MNEMONICS:
        return ("skipped", "branch/jump instruction")

    # Instructions with composite register operands need special handling.
    for op in instr.get("operands", []):
        if op.get("operand_type") == "composite_register":
            return ("skipped", "composite register operand")

    # Instructions with unknown operand types are risky.
    for op in instr.get("operands", []):
        if op.get("operand_type") == "unknown":
            return ("skipped", f"unknown operand type: {op.get('name')}")

    # Lock instructions are synchronization primitives.
    for op in instr.get("operands", []):
        if op.get("operand_type") == "lock_id":
            return ("skipped", "lock instruction")

    # No output operands = no observable result.
    outputs = [op for op in instr.get("operands", []) if op.get("is_output")]
    if not outputs:
        return ("skipped", "no output operands")

    # Check all register operands have known register kinds.
    for op in instr.get("operands", []):
        if op.get("operand_type") == "register":
            kind = op.get("register_kind")
            if not kind or kind not in _REGISTER_NAMES:
                return ("skipped", f"unknown register kind: {kind}")

    return ("testable", "")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_isa_test_gen.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/isa-test-gen.py tools/test_isa_test_gen.py
git commit -m "feat(isa-test): instruction classifier and operand mapping"
```

---

### Task 3: Assembly Generation -- Test Points and Mega-Programs

**Files:**
- Modify: `tools/isa-test-gen.py`
- Modify: `tools/test_isa_test_gen.py`

This task adds the core assembly generation: producing test points
(load + execute + store sequences) and batching them into mega-programs.

- [ ] **Step 1: Write failing tests for test point generation**

Append to `tools/test_isa_test_gen.py`:

```python
class TestAssemblyGeneration:
    def test_scalar_add_assembly(self):
        """Generate assembly for a scalar add instruction."""
        instr = {
            "name": "ADD", "mnemonic": "add",
            "asm_string": "$mRx, $mRx0, $mRy",
            "operands": [
                {"name": "mRx", "is_output": True, "operand_type": "register",
                 "register_kind": "scalar", "bit_width": 5},
                {"name": "mRx0", "is_output": False, "operand_type": "register",
                 "register_kind": "scalar", "bit_width": 5},
                {"name": "mRy", "is_output": False, "operand_type": "register",
                 "register_kind": "scalar", "bit_width": 5},
            ],
        }
        regs = {"mRx": "r0", "mRx0": "r1", "mRy": "r2"}
        point = isa_gen.generate_test_point(instr, regs, in_offset=0, out_offset=0)
        # Should contain the instruction
        assert "add r0, r1, r2" in point
        # Should contain loads for input registers
        assert "lda" in point.lower() or "mov" in point.lower()
        # Should contain store for output register
        assert "st " in point.lower()

    def test_vector_vadd_assembly(self):
        """Generate assembly for a vector add instruction."""
        instr = {
            "name": "VADD_32", "mnemonic": "vadd.32",
            "asm_string": "$x0, $x1, $x2",
            "operands": [
                {"name": "x0", "is_output": True, "operand_type": "register",
                 "register_kind": "vector512", "bit_width": 4},
                {"name": "x1", "is_output": False, "operand_type": "register",
                 "register_kind": "vector512", "bit_width": 4},
                {"name": "x2", "is_output": False, "operand_type": "register",
                 "register_kind": "vector512", "bit_width": 4},
            ],
        }
        regs = {"x0": "x0", "x1": "x2", "x2": "x4"}
        point = isa_gen.generate_test_point(instr, regs, in_offset=0, out_offset=0)
        assert "vadd.32 x0, x2, x4" in point
        # Should have vector loads
        assert "vlda" in point.lower()
        # Should have vector stores
        assert "vst" in point.lower()

    def test_immediate_operand_in_assembly(self):
        """Immediate operands appear as literal values, not registers."""
        instr = {
            "name": "ADD_imm", "mnemonic": "add",
            "asm_string": "$mRx, $mRx0, $imm",
            "operands": [
                {"name": "mRx", "is_output": True, "operand_type": "register",
                 "register_kind": "scalar", "bit_width": 5},
                {"name": "mRx0", "is_output": False, "operand_type": "register",
                 "register_kind": "scalar", "bit_width": 5},
                {"name": "imm", "is_output": False, "operand_type": "immediate",
                 "signed": True, "bit_width": 8},
            ],
        }
        regs = {"mRx": "r0", "mRx0": "r1", "imm": "42"}
        point = isa_gen.generate_test_point(instr, regs, in_offset=0, out_offset=0)
        assert "add r0, r1, #42" in point or "add r0, r1, 42" in point

    def test_mega_program_structure(self):
        """Mega-program has header, test points, and footer."""
        points = [
            "// test 0\nnop\nadd r0, r1, r2\nnop",
            "// test 1\nnop\nsub r0, r1, r2\nnop",
        ]
        prog = isa_gen.build_mega_program(points)
        assert ".text" in prog
        assert ".globl test_kernel" in prog
        assert "test_kernel:" in prog
        assert "ret lr" in prog
        assert "add r0, r1, r2" in prog
        assert "sub r0, r1, r2" in prog
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_isa_test_gen.py::TestAssemblyGeneration -v
```

- [ ] **Step 3: Implement test point generation**

Add to `tools/isa-test-gen.py`:

```python
# ---------------------------------------------------------------------------
# Register data sizes (bytes)
# ---------------------------------------------------------------------------

_REGISTER_SIZES: dict[str, int] = {
    "scalar": 4,
    "pointer": 4,
    "vector256": 32,
    "vector512": 64,
    "accumulator": 128,  # 1024-bit accumulators
    "modifier_m": 4,
    "modifier_dn": 4,
    "modifier_dj": 4,
    "modifier_dc": 4,
    "control": 4,
}

# How to load a register from memory at [p0, #offset].
# Returns list of assembly lines.  Uses 5 NOPs after loads for pipeline safety.
_LOAD_TEMPLATES: dict[str, callable] = {}
_STORE_TEMPLATES: dict[str, callable] = {}


def _load_scalar(reg: str, offset: int) -> list[str]:
    return [f"    lda {reg}, [p0, #{offset}]"]


def _load_vector256(reg: str, offset: int) -> list[str]:
    return [f"    vlda {reg}, [p0, #{offset}]"]


def _load_vector512(reg: str, offset: int) -> list[str]:
    # x0 = {wl0, wh0}, x2 = {wl2, wh2}, x4 = {wl4, wh4}.
    # The 512-bit xN register is composed of wlN and whN (same index).
    idx = int(reg[1:])
    wl = f"wl{idx}"
    wh = f"wh{idx}"
    return [
        f"    vlda {wl}, [p0, #{offset}]",
        f"    vlda {wh}, [p0, #{offset + 32}]",
    ]


def _load_accumulator(reg: str, offset: int) -> list[str]:
    # Accumulators cannot be loaded directly from memory.
    # Load as two 512-bit vectors via x registers, then ups to accumulator.
    # For simplicity, load into x0 and use ups to promote.
    # This is a simplification -- full coverage of accumulator inputs
    # is a future enhancement.
    return [
        f"    // accumulator load: {reg} from offset {offset}",
        f"    vlda wl0, [p0, #{offset}]",
        f"    vlda wh0, [p0, #{offset + 32}]",
        f"    nop",
        f"    nop",
        f"    nop",
        f"    nop",
        f"    nop",
        f"    ups.32 {reg}, x0, r0",
    ]


def _store_scalar(reg: str, offset: int) -> list[str]:
    return [f"    st {reg}, [p1, #{offset}]"]


def _store_vector256(reg: str, offset: int) -> list[str]:
    return [f"    vst {reg}, [p1, #{offset}]"]


def _store_vector512(reg: str, offset: int) -> list[str]:
    idx = int(reg[1:])
    wl = f"wl{idx * 2}"
    wh = f"wh{idx * 2}"
    return [
        f"    vst {wl}, [p1, #{offset}]",
        f"    vst {wh}, [p1, #{offset + 32}]",
    ]


def _store_accumulator(reg: str, offset: int) -> list[str]:
    # Store all four 256-bit lanes of the accumulator.
    idx = int(reg[2:])  # "cm0" -> 0
    return [
        f"    vst amll{idx}, [p1, #{offset}]",
        f"    vst amlh{idx}, [p1, #{offset + 32}]",
        f"    vst amhl{idx}, [p1, #{offset + 64}]",
        f"    vst amhh{idx}, [p1, #{offset + 96}]",
    ]


def _load_register(kind: str, reg: str, offset: int) -> list[str]:
    """Generate load instructions for a register from input buffer."""
    loaders = {
        "scalar": _load_scalar,
        "pointer": _load_scalar,  # same as scalar for loading
        "vector256": _load_vector256,
        "vector512": _load_vector512,
        "accumulator": _load_accumulator,
    }
    loader = loaders.get(kind)
    if loader:
        return loader(reg, offset)
    # Fallback: treat as scalar
    return _load_scalar(reg, offset)


def _store_register(kind: str, reg: str, offset: int) -> list[str]:
    """Generate store instructions for a register to output buffer."""
    storers = {
        "scalar": _store_scalar,
        "pointer": _store_scalar,
        "vector256": _store_vector256,
        "vector512": _store_vector512,
        "accumulator": _store_accumulator,
    }
    storer = storers.get(kind)
    if storer:
        return storer(reg, offset)
    return _store_scalar(reg, offset)


def generate_test_point(
    instr: dict,
    regs: dict[str, str],
    in_offset: int,
    out_offset: int,
) -> str:
    """Generate assembly for one test point.

    Parameters
    ----------
    instr: Instruction metadata from aie2-isa.json.
    regs: Mapping of operand name -> concrete register/value.
    in_offset: Byte offset into input buffer for this test point.
    out_offset: Byte offset into output buffer for this test point.
    """
    lines = [f"    // {instr['name']}: {instr['mnemonic']}"]

    # Load input registers from input buffer.
    load_offset = in_offset
    for op in instr["operands"]:
        if op["is_output"]:
            continue
        if op["operand_type"] != "register":
            continue  # immediates don't need loading
        kind = op["register_kind"]
        reg = regs[op["name"]]
        lines.extend(_load_register(kind, reg, load_offset))
        load_offset += _REGISTER_SIZES.get(kind, 4)

    # Pipeline safety NOPs (conservative).
    for _ in range(5):
        lines.append("    nop")

    # Assemble the instruction from mnemonic + asm_string template.
    asm = instr["asm_string"]
    for op_name, value in regs.items():
        # Immediates get # prefix for llvm-mc syntax.
        op_info = next((o for o in instr["operands"] if o["name"] == op_name), None)
        if op_info and op_info["operand_type"] == "immediate":
            asm = asm.replace(f"${op_name}", f"#{value}")
        else:
            asm = asm.replace(f"${op_name}", value)
    lines.append(f"    {instr['mnemonic']} {asm}")

    # Pipeline safety NOPs before store.
    for _ in range(5):
        lines.append("    nop")

    # Store output registers to output buffer.
    store_offset = out_offset
    for op in instr["operands"]:
        if not op["is_output"]:
            continue
        if op["operand_type"] != "register":
            continue
        kind = op["register_kind"]
        reg = regs[op["name"]]
        lines.extend(_store_register(kind, reg, store_offset))
        store_offset += _REGISTER_SIZES.get(kind, 4)

    return "\n".join(lines)


def build_mega_program(test_points: list[str]) -> str:
    """Wrap test points into a complete assembly program."""
    header = """\
    .text
    .globl test_kernel
test_kernel:
"""
    footer = """
    ret lr
    nop
    nop
    nop
    nop
"""
    body = "\n\n".join(test_points)
    return header + body + "\n" + footer
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_isa_test_gen.py -v
```

- [ ] **Step 5: Preflight -- verify load/store syntax with llvm-mc**

Before generating hundreds of test programs, confirm the assembly syntax
llvm-mc actually accepts for loads, stores, and vector operations:

```bash
cat > /tmp/claude-1000/preflight.s << 'EOF'
    .text
    .globl test_kernel
test_kernel:
    lda r0, [p0, #0]
    lda r1, [p0, #4]
    nop
    nop
    nop
    nop
    nop
    add r2, r0, r1
    nop
    nop
    nop
    nop
    nop
    st r2, [p1, #0]
    vlda wl0, [p0, #0]
    vlda wh0, [p0, #32]
    nop
    nop
    nop
    nop
    nop
    vadd.32 x0, x0, x0
    nop
    nop
    nop
    nop
    nop
    vst wl0, [p1, #0]
    vst wh0, [p1, #32]
    ret lr
    nop
    nop
    nop
    nop
EOF
~/npu-work/llvm-aie/build/bin/llvm-mc --triple=aie2 --filetype=obj \
    /tmp/claude-1000/preflight.s -o /tmp/claude-1000/preflight.o
echo "Preflight exit: $?"
```

If any syntax is rejected, fix the load/store templates in the generator
before proceeding.

- [ ] **Step 6: Commit**

```bash
git add tools/isa-test-gen.py tools/test_isa_test_gen.py
git commit -m "feat(isa-test): assembly test point generation and mega-program builder"
```

---

### Task 4: Full Pipeline -- Generate, Classify, Batch, Write Manifest

**Files:**
- Modify: `tools/isa-test-gen.py`
- Modify: `tools/test_isa_test_gen.py`

Wire up the full pipeline: load JSON, classify all instructions, generate
operand combos, produce batched mega-programs and manifest files.

- [ ] **Step 1: Write failing test for the full pipeline**

Append to `tools/test_isa_test_gen.py`:

```python
class TestFullPipeline:
    def test_generate_all_from_json(self):
        """Full pipeline: JSON -> classification -> assembly -> manifest."""
        isa_path = Path(__file__).parent / "aie2-isa.json"
        if not isa_path.exists():
            pytest.skip("aie2-isa.json not generated yet")

        result = isa_gen.generate_all(
            isa_json_path=str(isa_path),
            out_dir="/tmp/claude-1000/isa-test-pipeline",
        )
        assert result["total_instructions"] > 0
        assert result["testable"] > 0
        assert result["skipped"] > 0
        assert result["mega_programs"] > 0

        # Check manifest exists
        manifest_path = Path("/tmp/claude-1000/isa-test-pipeline/manifest.json")
        assert manifest_path.exists()
        manifest = json.load(open(manifest_path))
        assert len(manifest["batches"]) > 0
        assert len(manifest["skipped"]) > 0

        # Check at least one .s file was written
        s_files = list(Path("/tmp/claude-1000/isa-test-pipeline").glob("batch_*.s"))
        assert len(s_files) > 0

        # Check the .s file assembles with llvm-mc
        import subprocess
        llvm_mc = Path.home() / "npu-work/llvm-aie/build/bin/llvm-mc"
        if not llvm_mc.exists():
            pytest.skip("llvm-mc not found")
        result = subprocess.run(
            [str(llvm_mc), "--triple=aie2", "--filetype=obj",
             str(s_files[0]), "-o", "/tmp/claude-1000/test_asm.o"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"llvm-mc failed: {result.stderr[:500]}"
```

- [ ] **Step 2: Implement generate_all**

Add to `tools/isa-test-gen.py`:

```python
import os


def generate_operand_combos(instr: dict) -> list[dict[str, str]]:
    """Generate operand combinations for an instruction.

    Returns a list of register/value assignments, one per test case.
    Each is a dict mapping operand name -> concrete value.
    """
    combos = []

    # Base combo: first register in each class, 0 for immediates.
    base: dict[str, str] = {}
    for op in instr["operands"]:
        if op["operand_type"] == "register":
            kind = op["register_kind"]
            names = register_names(kind)
            base[op["name"]] = names[0] if names else "r0"
        elif op["operand_type"] == "immediate":
            base[op["name"]] = "0"
        else:
            base[op["name"]] = "0"  # fallback
    combos.append(base)

    # Additional combos: vary one operand at a time.
    for op in instr["operands"]:
        if op["operand_type"] == "register":
            kind = op["register_kind"]
            for alt_name in register_names(kind)[1:]:
                variant = dict(base)
                variant[op["name"]] = alt_name
                combos.append(variant)
        elif op["operand_type"] == "immediate":
            for val in immediate_values(op["bit_width"], op.get("signed", False))[1:]:
                variant = dict(base)
                variant[op["name"]] = str(val)
                combos.append(variant)

    return combos


# Maximum test points per mega-program (conservative, based on 16KB PM
# with 16-byte VLIW bundles and ~11-13 bundles per test point).
_MAX_POINTS_PER_BATCH = 70


def generate_all(
    isa_json_path: str,
    out_dir: str,
) -> dict:
    """Full pipeline: load ISA JSON, classify, generate assembly, write files.

    Returns summary dict with counts.
    """
    isa = json.load(open(isa_json_path))
    instructions = isa["instructions"]

    os.makedirs(out_dir, exist_ok=True)

    testable = []
    skipped_list = []

    for instr in instructions:
        status, reason = classify_instruction(instr)
        if status == "testable":
            testable.append(instr)
        else:
            skipped_list.append({"name": instr["name"], "reason": reason})

    # Generate test points for all testable instructions.
    all_points = []  # (instr_name, operand_desc, asm_text, in_size, out_size)
    for instr in testable:
        combos = generate_operand_combos(instr)
        for regs in combos:
            # Calculate data sizes.
            in_size = 0
            for op in instr["operands"]:
                if not op["is_output"] and op["operand_type"] == "register":
                    in_size += _REGISTER_SIZES.get(op.get("register_kind", ""), 4)
            out_size = 0
            for op in instr["operands"]:
                if op["is_output"] and op["operand_type"] == "register":
                    out_size += _REGISTER_SIZES.get(op.get("register_kind", ""), 4)

            # Use cumulative offsets (will be set during batching).
            operand_desc = ", ".join(f"{k}={v}" for k, v in regs.items())
            all_points.append({
                "instr": instr,
                "regs": regs,
                "operand_desc": operand_desc,
                "in_size": in_size,
                "out_size": out_size,
            })

    # Batch test points into mega-programs.
    batches = []
    current_batch = []
    current_in_offset = 0
    current_out_offset = 0

    for point in all_points:
        if len(current_batch) >= _MAX_POINTS_PER_BATCH:
            batches.append((current_batch, current_in_offset, current_out_offset))
            current_batch = []
            current_in_offset = 0
            current_out_offset = 0

        asm = generate_test_point(
            point["instr"], point["regs"],
            current_in_offset, current_out_offset,
        )
        current_batch.append({
            "asm": asm,
            "name": point["instr"]["name"],
            "operands": point["operand_desc"],
            "in_offset": current_in_offset,
            "in_size": point["in_size"],
            "out_offset": current_out_offset,
            "out_size": point["out_size"],
        })
        current_in_offset += point["in_size"]
        current_out_offset += point["out_size"]

    if current_batch:
        batches.append((current_batch, current_in_offset, current_out_offset))

    # Write assembly files and manifest.
    manifest_batches = []
    for batch_idx, (batch, total_in, total_out) in enumerate(batches):
        # Write .s file.
        points_asm = [item["asm"] for item in batch]
        program = build_mega_program(points_asm)
        s_path = os.path.join(out_dir, f"batch_{batch_idx:03d}.s")
        Path(s_path).write_text(program)

        manifest_batches.append({
            "batch": batch_idx,
            "s_file": f"batch_{batch_idx:03d}.s",
            "in_size": max(4, total_in),
            "out_size": max(4, total_out),
            "test_points": [
                {
                    "instruction": item["name"],
                    "operands": item["operands"],
                    "in_offset": item["in_offset"],
                    "in_size": item["in_size"],
                    "out_offset": item["out_offset"],
                    "out_size": item["out_size"],
                }
                for item in batch
            ],
        })

    manifest = {
        "batches": manifest_batches,
        "skipped": skipped_list,
    }
    Path(os.path.join(out_dir, "manifest.json")).write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    return {
        "total_instructions": len(instructions),
        "testable": len(testable),
        "skipped": len(skipped_list),
        "test_points": len(all_points),
        "mega_programs": len(batches),
    }
```

Also add a CLI:

```python
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ISA-level validation test programs from TableGen metadata."
    )
    parser.add_argument(
        "--isa-json",
        default="tools/aie2-isa.json",
        help="Path to exported ISA JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--out-dir",
        default="build/isa-tests",
        help="Output directory (default: %(default)s)",
    )
    args = parser.parse_args()

    result = generate_all(args.isa_json, args.out_dir)
    print(f"Instructions: {result['total_instructions']} total, "
          f"{result['testable']} testable, {result['skipped']} skipped")
    print(f"Test points:  {result['test_points']}")
    print(f"Mega-programs: {result['mega_programs']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run tests**

```bash
cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_isa_test_gen.py -v
```

- [ ] **Step 4: Run the generator end-to-end and verify assembly**

```bash
python3 tools/isa-test-gen.py --isa-json tools/aie2-isa.json --out-dir build/isa-tests
# Assemble the first batch
~/npu-work/llvm-aie/build/bin/llvm-mc --triple=aie2 --filetype=obj \
    build/isa-tests/batch_000.s -o /tmp/claude-1000/batch_000.o
echo "Exit: $?"
# Disassemble to verify
~/npu-work/llvm-aie/install/bin/llvm-objdump -d /tmp/claude-1000/batch_000.o | head -40
```

- [ ] **Step 5: Commit**

```bash
git add tools/isa-test-gen.py tools/test_isa_test_gen.py
git commit -m "feat(isa-test): full pipeline with batching, manifest, and CLI"
```

---

## Chunk 3: Runner Script and End-to-End

### Task 5: Runner Script

**Files:**
- Create: `scripts/isa-test.sh`

- [ ] **Step 1: Create the runner script**

Create `scripts/isa-test.sh`:

```bash
#!/usr/bin/env bash
# scripts/isa-test.sh -- ISA-level validation harness runner.
#
# Generates assembly test programs from TableGen ISA metadata, assembles
# with llvm-mc, links via aiecc.py, runs on HW and emulator, diffs outputs.
#
# Usage:
#   scripts/isa-test.sh [options]
#
# Options:
#   --no-hw          Skip hardware runs (EMU-only)
#   --no-emu         Skip emulator runs (HW-only)
#   --seed N         PRNG seed for input data (default: 42)
#   --generate-only  Only generate and assemble, skip run
#   --compile        Force recompilation
#   -j N             Parallelism for link + EMU (default: nproc)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LLVM_MC="${PROJECT_DIR}/../llvm-aie/build/bin/llvm-mc"
PEANO_INSTALL_DIR="${PROJECT_DIR}/../llvm-aie/install"

# mlir-aie paths for host compilation
MLIR_AIE="${PROJECT_DIR}/../mlir-aie"
TEST_LIB_DIR="${MLIR_AIE}/build/runtime_lib/x86_64/test_lib"
XRT_DIR="/opt/xilinx/xrt"

ISA_JSON="${PROJECT_DIR}/tools/aie2-isa.json"
OUT_DIR="${PROJECT_DIR}/build/isa-tests"
RESULTS_DIR="/tmp/isa-test-results-$(date +%Y%m%d)"

# Defaults
RUN_HW=true
RUN_EMU=true
SEED=42
GENERATE_ONLY=false
FORCE_COMPILE=false
JOBS=$(nproc)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-hw)       RUN_HW=false; shift ;;
        --no-emu)      RUN_EMU=false; shift ;;
        --seed)        SEED="$2"; shift 2 ;;
        --generate-only) GENERATE_ONLY=true; shift ;;
        --compile)     FORCE_COMPILE=true; shift ;;
        -j)            JOBS="$2"; shift 2 ;;
        *)             echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== ISA-Level Validation Harness ==="
echo "Out dir:  $OUT_DIR"
echo ""

# ---- Phase 1: Generate ----
echo "--- Phase 1: Generate assembly ---"
python3 "${PROJECT_DIR}/tools/isa-test-gen.py" \
    --isa-json "$ISA_JSON" \
    --out-dir "$OUT_DIR"
echo ""

# ---- Phase 2: Assemble ----
echo "--- Phase 2: Assemble with llvm-mc ---"
for s_file in "$OUT_DIR"/batch_*.s; do
    o_file="${s_file%.s}.o"
    if $FORCE_COMPILE || [[ ! -f "$o_file" ]] || [[ "$s_file" -nt "$o_file" ]]; then
        "$LLVM_MC" --triple=aie2 --filetype=obj "$s_file" -o "$o_file"
    fi
done
echo "Assembled $(ls "$OUT_DIR"/batch_*.o 2>/dev/null | wc -l) object files"
echo ""

# ---- Phase 3: Link + Package ----
echo "--- Phase 3: Link and package (j=$JOBS) ---"

# Import generate_aie_mlir from the Peano test generator for MLIR templates.
# For each batch, generate MLIR, run aiecc.py with Peano linker.
python3 -c "
import json, sys, os
sys.path.insert(0, '${PROJECT_DIR}/tools')
import importlib
peano = importlib.import_module('instr-test-gen')

manifest = json.load(open('${OUT_DIR}/manifest.json'))
for batch in manifest['batches']:
    batch_dir = os.path.join('${OUT_DIR}', f'batch_{batch[\"batch\"]:03d}')
    os.makedirs(batch_dir, exist_ok=True)

    # Copy the .o file into the batch dir as kernel.o
    import shutil
    src_o = os.path.join('${OUT_DIR}', f'batch_{batch[\"batch\"]:03d}.o')
    dst_o = os.path.join(batch_dir, 'kernel.o')
    if os.path.exists(src_o):
        shutil.copy2(src_o, dst_o)

    # Generate MLIR template
    in_bytes = batch['in_size']
    out_bytes = batch['out_size']
    mlir = peano.generate_aie_mlir(in_bytes, out_bytes)
    with open(os.path.join(batch_dir, 'aie.mlir'), 'w') as f:
        f.write(mlir)

print(f'Prepared {len(manifest[\"batches\"])} batch directories')
"

# Run aiecc.py for each batch (parallel)
link_one() {
    local batch_dir="$1"
    local name="$(basename "$batch_dir")"

    if ! $FORCE_COMPILE && [[ -f "${batch_dir}/aie.xclbin" ]]; then
        return 0
    fi

    (cd "$batch_dir" && \
        nice -n 19 aiecc.py --no-aiesim --no-xchesscc --no-xbridge \
            --aie-generate-xclbin --xclbin-name=aie.xclbin \
            --aie-generate-npu-insts --npu-insts-name=insts.bin \
            aie.mlir 2>"${batch_dir}/aiecc.log") || {
        echo "  FAIL link: ${name}"
        return 1
    }
    echo "  Linked: ${name}"
}
export -f link_one
export FORCE_COMPILE

find "$OUT_DIR" -maxdepth 1 -name "batch_*" -type d | \
    xargs -P "$JOBS" -I{} bash -c 'link_one "$1"' _ {}
echo ""

if $GENERATE_ONLY; then
    echo "Generate-only mode. Done."
    exit 0
fi

# Compile shared host harness
HOST_BIN="${OUT_DIR}/test_host"
if $FORCE_COMPILE || [[ ! -f "$HOST_BIN" ]]; then
    echo "Compiling test_host..."
    python3 -c "
import sys; sys.path.insert(0, '${PROJECT_DIR}/tools')
import importlib
peano = importlib.import_module('instr-test-gen')
with open('${OUT_DIR}/test_host.cpp', 'w') as f:
    f.write(peano.generate_test_host_cpp())
"
    clang++ "${OUT_DIR}/test_host.cpp" -o "$HOST_BIN" \
        -std=c++17 -Wall \
        -I "${TEST_LIB_DIR}/include" \
        -I "${XRT_DIR}/include" \
        -L "${TEST_LIB_DIR}/lib" \
        -L "${XRT_DIR}/lib" \
        -ltest_utils -lxrt_coreutil \
        -lrt -lstdc++
fi

# ---- Phase 4+5: Run HW, EMU, Compare ----
mkdir -p "$RESULTS_DIR"
MANIFEST="${OUT_DIR}/manifest.json"

python3 -c "
import json
manifest = json.load(open('${MANIFEST}'))
for batch in manifest['batches']:
    idx = batch['batch']
    print(f'batch_{idx:03d} {batch[\"in_size\"]} {batch[\"out_size\"]}')
" | while IFS=' ' read -r name in_size out_size; do
    batch_dir="${OUT_DIR}/${name}"

    if [[ ! -f "${batch_dir}/aie.xclbin" ]]; then
        echo "  SKIP ${name}: not compiled"
        continue
    fi

    if $RUN_HW; then
        "$HOST_BIN" \
            -x "${batch_dir}/aie.xclbin" -k MLIR_AIE \
            -i "${batch_dir}/insts.bin" \
            --in-size "$in_size" --out-size "$out_size" \
            --seed "$SEED" --out-file "${RESULTS_DIR}/${name}_hw.bin" 2>/dev/null && \
            echo "  HW OK: ${name}" || echo "  HW FAIL: ${name}"
    fi

    if $RUN_EMU; then
        XDNA_EMU=1 "$HOST_BIN" \
            -x "${batch_dir}/aie.xclbin" -k MLIR_AIE \
            -i "${batch_dir}/insts.bin" \
            --in-size "$in_size" --out-size "$out_size" \
            --seed "$SEED" --out-file "${RESULTS_DIR}/${name}_emu.bin" 2>/dev/null && \
            echo "  EMU OK: ${name}" || echo "  EMU FAIL: ${name}"
    fi

    if $RUN_HW && $RUN_EMU; then
        hw="${RESULTS_DIR}/${name}_hw.bin"
        emu="${RESULTS_DIR}/${name}_emu.bin"
        if [[ -f "$hw" ]] && [[ -f "$emu" ]]; then
            if cmp -s "$hw" "$emu"; then
                echo "  MATCH: ${name}"
            else
                echo "  DIVERGE: ${name}"
                # TODO: per-instruction divergence via manifest offsets
            fi
        fi
    fi
done

echo ""
echo "=== Done ==="
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/isa-test.sh
```

- [ ] **Step 3: Test generate-only mode**

```bash
./scripts/isa-test.sh --generate-only
```

Expected: generates assembly, assembles with llvm-mc, links via aiecc.py.
All batches should assemble and link successfully.

- [ ] **Step 4: Commit**

```bash
git add scripts/isa-test.sh
git commit -m "feat(isa-test): runner script for ISA validation harness"
```

---

### Task 6: End-to-End Validation

**Files:** None (manual testing)

- [ ] **Step 1: Run EMU-only to verify basic functionality**

```bash
./scripts/isa-test.sh --no-hw --seed 42
```

Expected: all batches run in EMU without crashes. Some may produce
wrong results (emulator bugs) -- that's what we're looking for.

- [ ] **Step 2: Fix any assembly or packaging errors**

If any batch fails to assemble or link, inspect the error and fix
the generator. Common issues:
- Register names llvm-mc doesn't accept
- Immediate values out of range
- Missing NOP padding

After fixes, re-run and iterate until all batches assemble and run.

- [ ] **Step 3: Commit fixes**

```bash
git add tools/isa-test-gen.py tools/test_isa_test_gen.py
git commit -m "fix(isa-test): assembly generation fixes from end-to-end validation"
```

- [ ] **Step 4: Run full HW + EMU comparison**

```bash
./scripts/isa-test.sh --seed 42
```

Expected: HW and EMU results for each batch. Any DIVERGE results
indicate emulator bugs -- document them.

- [ ] **Step 5: Run with different seed to check reproducibility**

```bash
./scripts/isa-test.sh --seed 100 --no-hw
```

Expected: EMU results may differ from seed 42 (different inputs) but
should still be consistent across re-runs with the same seed.
