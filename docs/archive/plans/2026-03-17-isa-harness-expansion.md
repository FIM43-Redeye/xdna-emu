# ISA Harness Expansion Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the ISA validation harness from 162 testable instructions (27%) to ~406 (67%) by adding load, store, and branch test strategies.

**Architecture:** Refactor `isa-test-gen.py`'s monolithic classifier and generator into pluggable test strategies (Compute, Load, Store, Branch). Each strategy provides `can_test()`, `setup_code()`, `execute_code()`, and `capture_code()`. The batch packer, manifest format, and runner script are unchanged.

**Tech Stack:** Python 3, llvm-mc assembler, AIE2 assembly, existing aie2-isa.json

---

### Task 1: Dontcare Reclassification

Treat `unknown` operand types with names starting `dontcare` as safe (zero-filled).
This is the smallest change and unlocks ~25 instructions immediately.

**Files:**
- Modify: `tools/isa-test-gen.py:171-177` (classify_instruction)
- Modify: `tools/isa-test-gen.py:608-629` (_substitute_asm)
- Modify: `tools/test_isa_test_gen.py` (add tests)

- [ ] **Step 1: Write failing test for dontcare acceptance**

Add to `TestClassifyInstruction` in `tools/test_isa_test_gen.py`:

```python
def test_dontcare_operand_accepted(self):
    """Instructions with dontcare padding operands should be testable."""
    instr = _make_instr("PADDA", "padda", "padda\t[$mPa], #$c12s", [
        _make_reg_op("mPa", "pointer", bit_width=3),
        _make_imm_op("c12s", bit_width=12, signed=True),
        _make_unknown_op("dontcare2", bit_width=2),
    ])
    status, reason = classify_instruction(instr)
    assert status == "testable", f"Expected testable, got {status}: {reason}"

def test_non_dontcare_unknown_still_skipped(self):
    """Unknown operands that are NOT dontcare should still be skipped."""
    instr = _make_instr("FOO", "foo", "foo\t$dst, $src, $ys1", [
        _make_reg_op("dst", "scalar"),
        _make_reg_op("src", "scalar"),
        _make_unknown_op("ys1", bit_width=2),
    ])
    status, reason = classify_instruction(instr)
    assert status == "skipped"
    assert "unknown" in reason.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tools/test_isa_test_gen.py::TestClassifyInstruction::test_dontcare_operand_accepted -v`
Expected: FAIL -- dontcare2 is `unknown` type, currently rejected.

- [ ] **Step 3: Implement dontcare acceptance in classify_instruction**

In `tools/isa-test-gen.py`, modify the operand type check (line ~173-177):

```python
# Before:
if op_type == "unknown":
    return ("skipped", "unknown operand type")

# After:
if op_type == "unknown":
    if not op.get("name", "").startswith("dontcare"):
        return ("skipped", "unknown operand type")
```

Also update `_substitute_asm()` to handle unsubstituted dontcare tokens.
Dontcare operands have no `$name` in the asm_string, so they need no
substitution -- but if they DO appear, zero-fill them:

Add to the end of `_substitute_asm()`, before the return:

```python
# Zero-fill any remaining dontcare operands.
result = re.sub(r'\$dontcare\w*', '#0', result)
```

- [ ] **Step 3b: Audit real ISA data for dontcare in asm_string**

Verify the `$dontcare` regex is safe by checking whether any real instructions
have `$dontcare*` tokens in their `asm_string`:

```bash
python3 -c "
import json
with open('tools/aie2-isa.json') as f:
    data = json.load(f)
for slot, instrs in data.items():
    for i in instrs:
        asm = i.get('asm_string', '')
        if 'dontcare' in asm:
            print(f'{i[\"name\"]}: {asm}')
"
```

Expected: Either no matches (regex is harmless dead code) or matches where
`#0` substitution produces valid assembly. If any match puts dontcare in a
register position, the regex needs a guard -- but in practice, dontcare
fields are padding bits not referenced in asm_string.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tools/test_isa_test_gen.py::TestClassifyInstruction::test_dontcare_operand_accepted tools/test_isa_test_gen.py::TestClassifyInstruction::test_non_dontcare_unknown_still_skipped -v`
Expected: Both PASS.

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `python3 -m pytest tools/test_isa_test_gen.py -v`
Expected: All existing tests still pass.

- [ ] **Step 6: Verify coverage gain with --summary**

Run: `python3 tools/isa-test-gen.py --isa-json tools/aie2-isa.json --summary 2>&1 | head -20`
Expected: Testable count increases from 162 to ~187 (dontcare instructions reclassified).

- [ ] **Step 7: Commit**

```bash
git add tools/isa-test-gen.py tools/test_isa_test_gen.py
git commit -m "feat(isa-harness): reclassify dontcare padding operands as testable"
```

---

### Task 2: Strategy Base Class + ComputeStrategy Refactor

Extract existing classification and generation logic into a `ComputeStrategy`
class. This is a pure refactor -- zero behavioral change. All existing tests
must continue to pass identically.

**Files:**
- Modify: `tools/isa-test-gen.py:128-710,900-1037`
- Modify: `tools/test_isa_test_gen.py`

- [ ] **Step 1: Write test that ComputeStrategy reproduces existing behavior**

Add to `tools/test_isa_test_gen.py`:

```python
class TestComputeStrategy:
    """Tests for ComputeStrategy (refactored from existing logic)."""

    def test_scalar_add_can_test(self):
        instr = _make_instr("ADD", "add", "add\t$mRx, $mRx0, $mRy", [
            _make_reg_op("mRx", "scalar"),
            _make_reg_op("mRx0", "scalar"),
            _make_reg_op("mRy", "scalar"),
        ])
        strategy = isa_test_gen.ComputeStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"Expected can_test=True: {reason}"

    def test_load_not_compute(self):
        instr = _make_instr("LDA", "lda", "lda\t$dst, [$p, #$off]", [
            _make_reg_op("dst", "scalar"),
            _make_reg_op("p", "pointer"),
            _make_imm_op("off"),
        ], may_load=True)
        strategy = isa_test_gen.ComputeStrategy()
        can, reason = strategy.can_test(instr)
        assert not can

    def test_generate_test_point_unchanged(self):
        """ComputeStrategy must produce identical assembly to old code path."""
        instr = _make_instr("ADD", "add", "add\t$mRx, $mRx0, $mRy", [
            _make_reg_op("mRx", "scalar"),
            _make_reg_op("mRx0", "scalar"),
            _make_reg_op("mRy", "scalar"),
        ])
        regs = {"mRx": "r2", "mRx0": "r3", "mRy": "r4"}
        # Old path
        old_asm = generate_test_point(instr, regs, in_offset=0, out_offset=0)
        # New path via strategy
        strategy = isa_test_gen.ComputeStrategy()
        new_asm = strategy.generate_test_point(instr, regs, in_offset=0, out_offset=0)
        assert old_asm == new_asm
```

- [ ] **Step 2: Run test to verify it fails (ComputeStrategy doesn't exist yet)**

Run: `python3 -m pytest tools/test_isa_test_gen.py::TestComputeStrategy -v`
Expected: FAIL -- `AttributeError: module has no attribute 'ComputeStrategy'`

- [ ] **Step 3: Implement TestStrategy base and ComputeStrategy**

Add the strategy classes to `tools/isa-test-gen.py`. The `TestStrategy` base
class defines the interface. `ComputeStrategy` wraps the existing functions
with zero logic changes.

```python
# After the constants section, before classify_instruction:

class TestStrategy:
    """Base class for instruction test strategies."""

    def can_test(self, instr: dict) -> tuple[bool, str]:
        """Return (True, "") if this strategy can test the instruction,
        or (False, skip_reason) if not."""
        raise NotImplementedError

    def generate_test_point(self, instr: dict, regs: dict[str, str],
                            in_offset: int, out_offset: int) -> str:
        """Generate assembly for one test point."""
        raise NotImplementedError


class ComputeStrategy(TestStrategy):
    """Tests compute instructions (ALU, vector, moves).

    This wraps the original classify_instruction() and generate_test_point()
    with no behavioral change.
    """

    def can_test(self, instr: dict) -> tuple[bool, str]:
        status, reason = classify_instruction(instr)
        return (status == "testable", reason)

    def generate_test_point(self, instr, regs, in_offset, out_offset):
        return generate_test_point(instr, regs, in_offset, out_offset)
```

- [ ] **Step 4: Run tests to verify ComputeStrategy works**

Run: `python3 -m pytest tools/test_isa_test_gen.py::TestComputeStrategy -v`
Expected: All PASS.

- [ ] **Step 5: Add strategy dispatch to generate_all**

Add a `STRATEGIES` list and a `classify_with_strategies()` function that
tries each strategy in order. Modify `generate_all()` to use it.

```python
STRATEGIES = [
    ComputeStrategy(),
]

def classify_with_strategies(instr: dict) -> tuple[Optional[TestStrategy], str]:
    """Try each strategy in order. Return (strategy, "") or (None, reason)."""
    last_reason = "no strategy matched"
    for strategy in STRATEGIES:
        can, reason = strategy.can_test(instr)
        if can:
            return (strategy, "")
        if reason:
            last_reason = reason
    return (None, last_reason)
```

In `generate_all()`, replace the `classify_instruction()` call with
`classify_with_strategies()` and use the returned strategy's
`generate_test_point()`:

```python
# In the classification loop:
strategy, reason = classify_with_strategies(instr)
if strategy is None:
    skipped_count += 1
    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
    continue

testable_count += 1
combos = generate_operand_combos(instr)
for combo_idx, regs in enumerate(combos):
    test_point_specs.append((instr, combo_idx, regs, strategy))

# In the assembly generation loop, use strategy.generate_test_point():
asm = strategy.generate_test_point(instr, regs, ...)
```

- [ ] **Step 6: Run full test suite -- must be identical to before**

Run: `python3 -m pytest tools/test_isa_test_gen.py -v`
Expected: ALL tests pass. No behavioral change.

- [ ] **Step 7: Update --summary to use classify_with_strategies**

In `main()`, replace the `--summary` code path's `classify_instruction()` call
with `classify_with_strategies()`. This keeps `--summary` consistent with
`generate_all()` throughout development:

```python
strategy, reason = classify_with_strategies(instr)
if strategy is not None:
    testable_count += 1
    testable_instrs.append(instr)
else:
    skipped_count += 1
    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
```

- [ ] **Step 8: Verify generate_all output is identical**

Run: `python3 tools/isa-test-gen.py --isa-json tools/aie2-isa.json --summary 2>&1 | head -10`
Expected: Same testable/skipped counts as before Task 2 (with dontcare fix from Task 1).

- [ ] **Step 8: Commit**

```bash
git add tools/isa-test-gen.py tools/test_isa_test_gen.py
git commit -m "refactor(isa-harness): extract ComputeStrategy from monolithic generator"
```

---

### Task 3: LoadStrategy -- Tier 1 (Simple Loads)

Add `LoadStrategy` that handles load instructions with simple addressing
(register + immediate offset, register + index register).

**Files:**
- Modify: `tools/isa-test-gen.py`
- Modify: `tools/test_isa_test_gen.py`

- [ ] **Step 1: Write failing tests for LoadStrategy**

```python
class TestLoadStrategy:
    """Tests for LoadStrategy."""

    def test_scalar_load_can_test(self):
        """Simple scalar load with pointer + immediate offset."""
        instr = _make_instr("LDA_S16", "lda.s16",
                            "lda.s16\t$mRa, [$ptr, #$off]", [
            _make_reg_op("mRa", "scalar"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("off", bit_width=6, signed=True),
        ], may_load=True, slot="lda")
        strategy = isa_test_gen.LoadStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"Expected can_test=True: {reason}"

    def test_rejects_store_with_may_load(self):
        """Instructions with both may_load and may_store are stores, not loads."""
        instr = _make_instr("ST_2D_S16", "st.2d.s16",
                            "st.2d.s16\t$mRv, [$ptr], $mod", [
            _make_reg_op("mRv", "scalar"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_reg_op("mod", "modifier_m", bit_width=3),
        ], may_load=True, may_store=True, slot="lda")
        strategy = isa_test_gen.LoadStrategy()
        can, reason = strategy.can_test(instr)
        assert not can

    def test_rejects_no_pointer(self):
        """VLDB_4x* register-to-register shuffles have no pointer -- reject."""
        instr = _make_instr("VLDB_4x16_HI", "vldb.4x16.hi",
                            "vldb.4x16.hi\t$dst, $src", [
            _make_reg_op("dst", "vector256"),
            _make_reg_op("src", "vector256"),
        ], may_load=True, slot="ldb")
        strategy = isa_test_gen.LoadStrategy()
        can, reason = strategy.can_test(instr)
        assert not can

    def test_rejects_composite_dest(self):
        """Loads with composite destination are deferred."""
        instr = _make_instr("LDA_2D_LDASCL", "lda.2d",
                            "lda.2d\t$mLdaScl, [$ptr], $mod", [
            _make_composite_op("mLdaScl", "LdaScl", bit_width=7),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_reg_op("mod", "modifier_m", bit_width=3),
        ], may_load=True, slot="lda")
        strategy = isa_test_gen.LoadStrategy()
        can, reason = strategy.can_test(instr)
        assert not can

    def test_generates_valid_assembly(self):
        """LoadStrategy should produce setup + load + capture assembly."""
        instr = _make_instr("LDA_S16", "lda.s16",
                            "lda.s16\t$mRa, [$ptr, #$off]", [
            _make_reg_op("mRa", "scalar"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("off", bit_width=6, signed=True),
        ], may_load=True, slot="lda", sched_class="II_LDA")
        strategy = isa_test_gen.LoadStrategy()
        regs = {"mRa": "r0", "ptr": "p6", "off": "0"}
        asm = strategy.generate_test_point(instr, regs,
                                           in_offset=0, out_offset=0)
        # Should set up p6 pointing at input data
        assert "p6" in asm
        # Should contain the load instruction
        assert "lda.s16" in asm
        # Should store the loaded value
        assert "st" in asm
        # Should have NOP sled for load latency
        assert "nop" in asm

    def test_vector_load_uses_vst(self):
        """Vector loads should use vst to capture the result."""
        instr = _make_instr("VLDA_128", "vlda.128",
                            "vlda.128\t$dst, [$ptr, #$off]", [
            _make_reg_op("dst", "vector256"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("off", bit_width=6, signed=True),
        ], may_load=True, slot="ldb", sched_class="II_VLDA")
        strategy = isa_test_gen.LoadStrategy()
        regs = {"dst": "wl0", "ptr": "p6", "off": "0"}
        asm = strategy.generate_test_point(instr, regs,
                                           in_offset=0, out_offset=0)
        assert "vlda.128" in asm
        assert "vst" in asm
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tools/test_isa_test_gen.py::TestLoadStrategy -v`
Expected: FAIL -- `AttributeError: module has no attribute 'LoadStrategy'`

- [ ] **Step 3: Implement LoadStrategy**

Add `LoadStrategy` class to `tools/isa-test-gen.py`:

```python
class LoadStrategy(TestStrategy):
    """Tests load instructions by pointing at input buffer and capturing result.

    Handles Tier 1 (ptr+offset, ptr+index) and Tier 2 (ptr+modifier 2D/3D).
    Rejects: may_store=True, no pointer operand, composite destinations,
    compressed/sparse loads, stack-pointer loads.
    """

    # Compressed/sparse load prefixes that need hardware FIFO init.
    _SKIP_PREFIXES = ("vldb.compr", "vldb.sparse")

    def can_test(self, instr: dict) -> tuple[bool, str]:
        if not instr.get("may_load", False):
            return (False, "not a load instruction")
        if instr.get("may_store", False):
            return (False, "load+store combo (actually a store)")

        mnemonic = instr.get("mnemonic", "")
        if any(mnemonic.startswith(p) for p in self._SKIP_PREFIXES):
            return (False, "compressed/sparse load (needs FIFO init)")

        operands = instr.get("operands", [])

        # Must have a pointer operand (rejects VLDB_4x register shuffles).
        has_pointer = any(
            op.get("register_kind") == "pointer"
            for op in operands
            if op.get("operand_type") in REGISTER_LIKE_TYPES
        )
        if not has_pointer:
            return (False, "load with no pointer operand (register shuffle)")

        # Reject composite destinations.
        for op in operands:
            if op.get("operand_type") == "composite_register":
                return (False, "composite register destination (deferred)")

        # Detect the output (loaded register). Must have one.
        dest = self._detect_load_dest(instr)
        if dest is None:
            return (False, "no detectable load destination")

        # Reject unsupported destination kinds.
        kind = dest.get("register_kind", "")
        if kind not in KNOWN_REGISTER_KINDS:
            return (False, f"unsupported load destination kind: {kind}")

        # Reject accumulator edge cases (same as compute).
        if kind == "accumulator":
            bw = dest.get("bit_width", 0)
            if bw == 2:
                return (False, "composite sparse register (bw=2)")
            if bw == 6:
                return (False, "accumulator quarter register (bw=6)")

        # Reject stack-pointer relative (asm contains "[sp,").
        asm = instr.get("asm_string", "")
        if "[sp," in asm.lower() or "[sp]" in asm.lower():
            return (False, "stack-pointer relative load (deferred)")

        return (True, "")

    def _detect_load_dest(self, instr: dict) -> Optional[dict]:
        """For loads, the destination is the first register in asm_string."""
        outputs = detect_output_operands(instr)
        return outputs[0] if outputs else None

    def generate_test_point(self, instr, regs, in_offset, out_offset):
        lines = []
        name = instr["name"]
        lines.append(f"  // ---- test (load): {name} ----")

        dest = self._detect_load_dest(instr)
        dest_name = dest["name"]
        dest_kind = dest.get("register_kind", "scalar")
        dest_reg = regs.get(dest_name, "r0")

        # Align input offset for this destination type.
        align = 32 if dest_kind in ("vector256", "vector512", "accumulator") else 4
        in_offset = _align(in_offset, align)

        # Setup: copy p0 to p6 and advance to the data region.
        # The load instruction will read through p6 (or whichever pointer
        # the instruction uses -- we substitute p6 for it).
        lines.extend(_padda_sequence("p6", "p0", in_offset))

        # For modifier operands (2D/3D loads), zero the modifier.
        op_by_name = {op["name"]: op for op in instr.get("operands", [])}
        for op_name, op in op_by_name.items():
            kind = op.get("register_kind", "")
            if kind == "modifier_m":
                mod_reg = regs.get(op_name, "m0")
                lines.append(f"  mov {mod_reg}, #0")
            elif kind == "modifier_dj":
                dj_reg = regs.get(op_name, "dj0")
                lines.append(f"  mov {dj_reg}, #0")

        # NOP sled to cover any setup latency (padda pipeline).
        lines.extend(_nop_sled(2))

        # Execute: the load instruction itself.
        # Override pointer operand to use p6.
        load_regs = dict(regs)
        for op_name, op in op_by_name.items():
            if op.get("register_kind") == "pointer":
                load_regs[op_name] = "p6"
        # Zero any immediate offset (data is already at p6).
        # NOTE: This means we only test loads at offset zero. A future
        # enhancement could vary the offset and adjust p6 accordingly,
        # but for initial coverage this is sufficient.
        for op_name, op in op_by_name.items():
            if op.get("operand_type") == "immediate":
                load_regs[op_name] = "0"

        asm_line = "  " + _substitute_asm(instr["asm_string"], load_regs)
        lines.append(asm_line)

        # NOP sled for load result latency.
        lines.extend(_nop_sled(result_latency(instr)))

        # Capture: store loaded value to output buffer.
        out_offset = _align(out_offset, align)
        store_lines = _store_instruction(dest_reg, dest_kind, "p1", out_offset)
        lines.extend(store_lines)

        lines.append("")
        return "\n".join(lines)
```

Also add `LoadStrategy()` to the `STRATEGIES` list:

```python
STRATEGIES = [
    LoadStrategy(),      # must be before ComputeStrategy
    ComputeStrategy(),
]
```

- [ ] **Step 4: Run LoadStrategy tests**

Run: `python3 -m pytest tools/test_isa_test_gen.py::TestLoadStrategy -v`
Expected: All PASS.

- [ ] **Step 5: Run full test suite**

Run: `python3 -m pytest tools/test_isa_test_gen.py -v`
Expected: All pass. The `test_loads_skipped` test in `TestRealISA` will now
FAIL because loads are no longer skipped. **Delete `test_loads_skipped` and
replace it with:**

```python
def test_some_loads_testable(self, isa_data):
    """Some lda-slot instructions should now be testable via LoadStrategy."""
    testable = 0
    for instr in isa_data.get("lda", []):
        if instr.get("may_load"):
            strategy, _ = isa_test_gen.classify_with_strategies(instr)
            if strategy is not None:
                testable += 1
    assert testable > 0, "Expected some loads to be testable"
```

- [ ] **Step 6: Verify coverage gain**

Run: `python3 tools/isa-test-gen.py --isa-json tools/aie2-isa.json --summary 2>&1 | head -10`
Expected: Testable count increases significantly (loads added).

- [ ] **Step 7: Commit**

```bash
git add tools/isa-test-gen.py tools/test_isa_test_gen.py
git commit -m "feat(isa-harness): add LoadStrategy for load instruction testing"
```

---

### Task 4: StoreStrategy

Mirror of LoadStrategy: load known data into register, execute store-under-test
to output buffer.

**Files:**
- Modify: `tools/isa-test-gen.py`
- Modify: `tools/test_isa_test_gen.py`

- [ ] **Step 1: Write failing tests for StoreStrategy**

```python
class TestStoreStrategy:
    """Tests for StoreStrategy."""

    def test_scalar_store_can_test(self):
        instr = _make_instr("ST_S16", "st.s16",
                            "st.s16\t$mRv, [$ptr, #$off]", [
            _make_reg_op("mRv", "scalar"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("off", bit_width=6, signed=True),
        ], may_store=True, slot="st")
        strategy = isa_test_gen.StoreStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"Expected can_test=True: {reason}"

    def test_rejects_no_pointer(self):
        """Stores without pointer operands should be rejected."""
        instr = _make_instr("VST_FAKE", "vst.fake",
                            "vst.fake\t$src, $dst", [
            _make_reg_op("src", "vector256"),
            _make_reg_op("dst", "vector256"),
        ], may_store=True, slot="st")
        strategy = isa_test_gen.StoreStrategy()
        can, reason = strategy.can_test(instr)
        assert not can

    def test_rejects_composite_source(self):
        """Stores with composite register source should be rejected."""
        instr = _make_instr("ST_COMP", "st.2d",
                            "st.2d\t$mLdaScl, [$ptr], $mod", [
            _make_composite_op("mLdaScl", "LdaScl", bit_width=7),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_reg_op("mod", "modifier_m", bit_width=3),
        ], may_store=True, slot="st")
        strategy = isa_test_gen.StoreStrategy()
        can, reason = strategy.can_test(instr)
        assert not can

    def test_generates_valid_assembly(self):
        """StoreStrategy should load data then execute store."""
        instr = _make_instr("ST_S16", "st.s16",
                            "st.s16\t$mRv, [$ptr, #$off]", [
            _make_reg_op("mRv", "scalar"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("off", bit_width=6, signed=True),
        ], may_store=True, slot="st", sched_class="II_ST")
        strategy = isa_test_gen.StoreStrategy()
        regs = {"mRv": "r0", "ptr": "p7", "off": "0"}
        asm = strategy.generate_test_point(instr, regs,
                                           in_offset=0, out_offset=0)
        # Should load known data from input buffer
        assert "lda" in asm or "vlda" in asm
        # Should contain the store instruction
        assert "st.s16" in asm
        # Should use p7 for output
        assert "p7" in asm

    def test_vector_store(self):
        """Vector stores should load vector data first."""
        instr = _make_instr("VST_128", "vst.128",
                            "vst.128\t$src, [$ptr, #$off]", [
            _make_reg_op("src", "vector256"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("off", bit_width=6, signed=True),
        ], may_store=True, slot="st", sched_class="II_VST")
        strategy = isa_test_gen.StoreStrategy()
        regs = {"src": "wl0", "ptr": "p7", "off": "0"}
        asm = strategy.generate_test_point(instr, regs,
                                           in_offset=0, out_offset=0)
        assert "vlda" in asm
        assert "vst.128" in asm

    def test_store_width_s8(self):
        """st.s8 stores 1 byte."""
        instr = _make_instr("ST_S8", "st.s8",
                            "st.s8\t$mRv, [$ptr, #$off]", [
            _make_reg_op("mRv", "scalar"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("off", bit_width=6, signed=True),
        ], may_store=True, slot="st")
        strategy = isa_test_gen.StoreStrategy()
        assert strategy._compute_store_width(instr) == 1

    def test_store_width_s16(self):
        """st.s16 stores 2 bytes."""
        instr = _make_instr("ST_S16", "st.s16",
                            "st.s16\t$mRv, [$ptr, #$off]", [
            _make_reg_op("mRv", "scalar"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("off", bit_width=6, signed=True),
        ], may_store=True, slot="st")
        strategy = isa_test_gen.StoreStrategy()
        assert strategy._compute_store_width(instr) == 2

    def test_store_width_scalar_default(self):
        """Plain 'st' stores 4 bytes."""
        instr = _make_instr("ST", "st",
                            "st\t$mRv, [$ptr, #$off]", [
            _make_reg_op("mRv", "scalar"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("off", bit_width=6, signed=True),
        ], may_store=True, slot="st")
        strategy = isa_test_gen.StoreStrategy()
        assert strategy._compute_store_width(instr) == 4

    def test_store_width_vector(self):
        """vst stores 32 bytes."""
        instr = _make_instr("VST", "vst",
                            "vst\t$src, [$ptr, #$off]", [
            _make_reg_op("src", "vector256"),
            _make_reg_op("ptr", "pointer", bit_width=3),
            _make_imm_op("off", bit_width=6, signed=True),
        ], may_store=True, slot="st")
        strategy = isa_test_gen.StoreStrategy()
        assert strategy._compute_store_width(instr) == 32
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tools/test_isa_test_gen.py::TestStoreStrategy -v`
Expected: FAIL -- `AttributeError: module has no attribute 'StoreStrategy'`

- [ ] **Step 3: Implement StoreStrategy**

```python
class StoreStrategy(TestStrategy):
    """Tests store instructions by loading known data then executing the store.

    The store writes to the output buffer via p7. The verification is that
    HW and EMU produce identical output bytes.
    """

    def can_test(self, instr: dict) -> tuple[bool, str]:
        if not instr.get("may_store", False):
            return (False, "not a store instruction")

        operands = instr.get("operands", [])

        # Must have a pointer operand.
        has_pointer = any(
            op.get("register_kind") == "pointer"
            for op in operands
            if op.get("operand_type") in REGISTER_LIKE_TYPES
        )
        if not has_pointer:
            return (False, "store with no pointer operand")

        # Reject composite operands.
        for op in operands:
            if op.get("operand_type") == "composite_register":
                return (False, "composite register operand (deferred)")

        # Detect the source (data to be stored).
        source = self._detect_store_source(instr)
        if source is None:
            return (False, "no detectable store source")

        kind = source.get("register_kind", "")
        if kind not in KNOWN_REGISTER_KINDS:
            return (False, f"unsupported store source kind: {kind}")

        # Reject unknown operand types (except dontcare).
        for op in operands:
            op_type = op.get("operand_type", "unknown")
            if op_type == "unknown":
                if not op.get("name", "").startswith("dontcare"):
                    return (False, "unknown operand type")

        # Reject stack-pointer relative.
        asm = instr.get("asm_string", "")
        if "[sp," in asm.lower() or "[sp]" in asm.lower():
            return (False, "stack-pointer relative store (deferred)")

        return (True, "")

    def _detect_store_source(self, instr: dict) -> Optional[dict]:
        """For stores, the source is the first register in asm_string
        (which is NOT the pointer or modifier)."""
        asm_string = instr.get("asm_string", "")
        asm_op_names = re.findall(r'\$(\w+)', asm_string)
        op_by_name = {op["name"]: op for op in instr.get("operands", [])}

        for name in asm_op_names:
            if name not in op_by_name:
                continue
            op = op_by_name[name]
            if op.get("operand_type") not in REGISTER_LIKE_TYPES:
                continue
            kind = op.get("register_kind", "")
            # The source is the first non-pointer, non-modifier register.
            if kind not in ("pointer", "modifier_m", "modifier_dj"):
                return op
        return None

    def _compute_store_width(self, instr: dict) -> int:
        """Determine how many bytes the store instruction actually writes.

        Uses the mnemonic to detect data width: .s8=1, .s16=2, .s32/.u32=4,
        vector stores=32 or 64 bytes.
        """
        mnemonic = instr.get("mnemonic", "")
        source = self._detect_store_source(instr)
        kind = source.get("register_kind", "scalar") if source else "scalar"

        if kind in ("vector256",):
            return 32
        if kind in ("vector512",):
            return 64
        if kind in ("accumulator",):
            return 64

        # Scalar stores: check mnemonic for data width.
        if ".s8" in mnemonic or ".u8" in mnemonic:
            return 1
        if ".s16" in mnemonic or ".u16" in mnemonic:
            return 2
        return 4  # default scalar = 32 bits

    def generate_test_point(self, instr, regs, in_offset, out_offset):
        lines = []
        name = instr["name"]
        lines.append(f"  // ---- test (store): {name} ----")

        source = self._detect_store_source(instr)
        src_name = source["name"]
        src_kind = source.get("register_kind", "scalar")
        src_reg = regs.get(src_name, "r0")

        op_by_name = {op["name"]: op for op in instr.get("operands", [])}

        # Align input offset and load known data into the source register.
        align = 32 if src_kind in ("vector256", "vector512", "accumulator") else 4
        in_offset = _align(in_offset, align)
        load_lines = _load_instruction(src_reg, src_kind, "p0", in_offset)
        lines.extend(load_lines)

        # NOP sled for load latency.
        lines.extend(_nop_sled(LOAD_LATENCY))

        # Setup: copy p1 to p7 and advance to output region.
        out_offset = _align(out_offset, align)
        lines.extend(_padda_sequence("p7", "p1", out_offset))

        # For modifier operands, zero the modifier.
        for op_name, op in op_by_name.items():
            kind = op.get("register_kind", "")
            if kind == "modifier_m":
                mod_reg = regs.get(op_name, "m0")
                lines.append(f"  mov {mod_reg}, #0")
            elif kind == "modifier_dj":
                dj_reg = regs.get(op_name, "dj0")
                lines.append(f"  mov {dj_reg}, #0")

        # Execute: the store instruction itself.
        store_regs = dict(regs)
        for op_name, op in op_by_name.items():
            if op.get("register_kind") == "pointer":
                store_regs[op_name] = "p7"
        # Zero any immediate offset (p7 already points at the right place).
        for op_name, op in op_by_name.items():
            if op.get("operand_type") == "immediate":
                store_regs[op_name] = "0"

        asm_line = "  " + _substitute_asm(instr["asm_string"], store_regs)
        lines.append(asm_line)

        lines.append("")
        return "\n".join(lines)
```

Update `STRATEGIES`:

```python
STRATEGIES = [
    LoadStrategy(),
    StoreStrategy(),
    ComputeStrategy(),
]
```

**Important:** Update `_compute_input_size()` and `_compute_output_size()`
in `generate_all()` to use strategy-specific size computation. For
`StoreStrategy`, the input size is the source register size (data to load),
and the output size is the store width (from `_compute_store_width()`).

Add methods to the strategy classes:

```python
# In TestStrategy base:
def compute_input_size(self, instr, regs):
    return _compute_input_size(instr, regs)

def compute_output_size(self, instr):
    return _compute_output_size(instr)
```

Override in `LoadStrategy`:
```python
def compute_input_size(self, instr, regs):
    dest = self._detect_load_dest(instr)
    if dest is None:
        return 4
    return _operand_size(dest, instr.get("name", ""))

def compute_output_size(self, instr):
    dest = self._detect_load_dest(instr)
    if dest is None:
        return 4
    return _operand_size(dest, instr.get("name", ""))
```

Override in `StoreStrategy`:
```python
def compute_input_size(self, instr, regs):
    source = self._detect_store_source(instr)
    if source is None:
        return 4
    return _operand_size(source, instr.get("name", ""))

def compute_output_size(self, instr):
    return self._compute_store_width(instr)
```

- [ ] **Step 4: Run StoreStrategy tests**

Run: `python3 -m pytest tools/test_isa_test_gen.py::TestStoreStrategy -v`
Expected: All PASS.

- [ ] **Step 5: Update TestRealISA for stores and run full suite**

**Delete `test_stores_skipped` and replace it with:**

```python
def test_some_stores_testable(self, isa_data):
    """Some st-slot instructions should now be testable via StoreStrategy."""
    testable = 0
    for instr in isa_data.get("st", []):
        if instr.get("may_store"):
            strategy, _ = isa_test_gen.classify_with_strategies(instr)
            if strategy is not None:
                testable += 1
    assert testable > 0, "Expected some stores to be testable"
```

Run: `python3 -m pytest tools/test_isa_test_gen.py -v`
Expected: All pass.

- [ ] **Step 6: Verify coverage gain**

Run: `python3 tools/isa-test-gen.py --isa-json tools/aie2-isa.json --summary 2>&1 | head -10`
Expected: Testable count increases further (stores added). Should be ~350+.

- [ ] **Step 7: Commit**

```bash
git add tools/isa-test-gen.py tools/test_isa_test_gen.py
git commit -m "feat(isa-harness): add StoreStrategy for store instruction testing"
```

---

### Task 5: BranchStrategy

Marker-based verification for branch/jump instructions.

**Files:**
- Modify: `tools/isa-test-gen.py`
- Modify: `tools/test_isa_test_gen.py`

- [ ] **Step 1: Write failing tests for BranchStrategy**

```python
class TestBranchStrategy:
    """Tests for BranchStrategy."""

    def setup_method(self):
        """Reset label counter before each test."""
        isa_test_gen.BranchStrategy.reset_labels()

    def test_jnz_can_test(self):
        """Conditional branch jnz should be testable."""
        instr = _make_instr("JNZ", "jnz", "jnz\t$mRx, $cpmaddr", [
            _make_reg_op("mRx", "scalar"),
            _make_imm_op("cpmaddr", bit_width=20, signed=False),
        ], slot="lng")
        strategy = isa_test_gen.BranchStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"Expected can_test=True: {reason}"

    def test_jz_can_test(self):
        instr = _make_instr("JZ", "jz", "jz\t$mRx, $cpmaddr", [
            _make_reg_op("mRx", "scalar"),
            _make_imm_op("cpmaddr", bit_width=20, signed=False),
        ], slot="lng")
        strategy = isa_test_gen.BranchStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"Expected can_test=True: {reason}"

    def test_j_can_test(self):
        instr = _make_instr("J", "j", "j\t$cpmaddr", [
            _make_imm_op("cpmaddr", bit_width=20, signed=False),
        ], slot="lng")
        strategy = isa_test_gen.BranchStrategy()
        can, reason = strategy.can_test(instr)
        assert can, f"Expected can_test=True: {reason}"

    def test_ret_deferred(self):
        instr = _make_instr("RET", "ret", "ret lr", [], slot="alu")
        strategy = isa_test_gen.BranchStrategy()
        can, reason = strategy.can_test(instr)
        assert not can

    def test_jnzd_deferred(self):
        instr = _make_instr("JNZD", "jnzd", "jnzd\t$mRx, $mRx0, $mPm", [
            _make_reg_op("mRx", "scalar"),
            _make_reg_op("mRx0", "scalar"),
            _make_reg_op("mPm", "pointer"),
        ], slot="alu")
        strategy = isa_test_gen.BranchStrategy()
        can, reason = strategy.can_test(instr)
        assert not can

    def test_jnz_generates_markers(self):
        """JNZ should produce marker-based assembly with labels."""
        instr = _make_instr("JNZ", "jnz", "jnz\t$mRx, $cpmaddr", [
            _make_reg_op("mRx", "scalar"),
            _make_imm_op("cpmaddr", bit_width=20, signed=False),
        ], slot="lng")
        strategy = isa_test_gen.BranchStrategy()
        regs = {"mRx": "r0", "cpmaddr": "0"}
        asm = strategy.generate_test_point(instr, regs,
                                           in_offset=0, out_offset=0)
        # Should have labels
        assert ".Ltaken_" in asm
        assert ".Ldone_" in asm
        # Should store markers
        assert "0xAA" in asm or "#170" in asm  # 0xAA = 170
        # Should have the branch instruction
        assert "jnz" in asm

    def test_labels_are_unique(self):
        """Two branch test points should have different label suffixes."""
        instr = _make_instr("JNZ", "jnz", "jnz\t$mRx, $cpmaddr", [
            _make_reg_op("mRx", "scalar"),
            _make_imm_op("cpmaddr", bit_width=20, signed=False),
        ], slot="lng")
        strategy = isa_test_gen.BranchStrategy()
        regs = {"mRx": "r0", "cpmaddr": "0"}
        asm1 = strategy.generate_test_point(instr, regs,
                                            in_offset=0, out_offset=0)
        asm2 = strategy.generate_test_point(instr, regs,
                                            in_offset=32, out_offset=32)
        # Extract label numbers
        import re
        labels1 = set(re.findall(r'\.L\w+_(\d+)', asm1))
        labels2 = set(re.findall(r'\.L\w+_(\d+)', asm2))
        assert labels1.isdisjoint(labels2), \
            f"Labels must be unique: {labels1} vs {labels2}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tools/test_isa_test_gen.py::TestBranchStrategy -v`
Expected: FAIL -- `AttributeError: module has no attribute 'BranchStrategy'`

- [ ] **Step 3: Implement BranchStrategy**

```python
class BranchStrategy(TestStrategy):
    """Tests branch instructions using marker-based verification.

    Stores marker values to the output buffer to verify which path executed.
    Conditional branches get two test points: taken and not-taken.
    """

    # Branches we can test (immediate-target only for now).
    _TESTABLE = frozenset({"j", "jl", "jnz", "jz"})
    # Deferred branches.
    _DEFERRED = frozenset({"ret", "jnzd"})

    # Global counter for unique labels across all test points in a batch.
    _label_counter = 0

    @classmethod
    def _next_label_id(cls) -> int:
        cls._label_counter += 1
        return cls._label_counter

    @classmethod
    def reset_labels(cls):
        """Reset label counter (call at start of each batch)."""
        cls._label_counter = 0

    def can_test(self, instr: dict) -> tuple[bool, str]:
        mnemonic = instr.get("mnemonic", "")
        if mnemonic in self._DEFERRED:
            return (False, f"deferred branch: {mnemonic}")
        if mnemonic not in self._TESTABLE:
            return (False, "not a branch instruction")

        # Reject register-indirect branches (j $mPm, jl $mPm).
        # These need pointer-to-label setup; deferred for now.
        operands = instr.get("operands", [])
        has_pointer = any(
            op.get("register_kind") == "pointer"
            for op in operands
            if op.get("operand_type") in REGISTER_LIKE_TYPES
        )
        if has_pointer:
            return (False, "register-indirect branch (deferred)")

        return (True, "")

    def _is_conditional(self, instr: dict) -> bool:
        return instr.get("mnemonic", "") in ("jnz", "jz")

    def compute_input_size(self, instr, regs):
        # Conditional branches need a 4-byte condition value.
        return 4 if self._is_conditional(instr) else 0

    def compute_output_size(self, instr):
        # 8 bytes: 4 for "before" marker + 4 for "path" marker.
        return 8

    def generate_test_point(self, instr, regs, in_offset, out_offset):
        lines = []
        name = instr["name"]
        mnemonic = instr.get("mnemonic", "")
        lid = self._next_label_id()
        lines.append(f"  // ---- test (branch): {name} ----")

        BRANCH_DELAY = 5

        # For conditional branches, load the condition value from input.
        if self._is_conditional(instr):
            in_offset = _align(in_offset, 4)
            lines.extend(_scalar_load("r0", "p0", in_offset))
            lines.extend(_nop_sled(LOAD_LATENCY))

        # Store "before" marker (0xAA).
        out_offset = _align(out_offset, 4)
        lines.append(f"  mov r14, #170")  # 0xAA
        lines.extend(_scalar_store("r14", "p1", out_offset))

        # The branch instruction. Target is always a label.
        if mnemonic == "j":
            lines.append(f"  j .Ltaken_{lid}")
        elif mnemonic == "jl":
            lines.append(f"  jl .Ltaken_{lid}")
        elif mnemonic == "jnz":
            lines.append(f"  jnz r0, .Ltaken_{lid}")
        elif mnemonic == "jz":
            lines.append(f"  jz r0, .Ltaken_{lid}")

        # Branch delay slots.
        lines.extend(_nop_sled(BRANCH_DELAY))

        # Fall-through path (branch NOT taken).
        lines.append(f"  mov r14, #187")  # 0xBB
        lines.extend(_scalar_store("r14", "p1", out_offset + 4))
        lines.append(f"  j .Ldone_{lid}")
        lines.extend(_nop_sled(BRANCH_DELAY))

        # Taken path.
        lines.append(f".Ltaken_{lid}:")
        lines.append(f"  mov r14, #204")  # 0xCC
        lines.extend(_scalar_store("r14", "p1", out_offset + 4))

        # Convergence point.
        lines.append(f".Ldone_{lid}:")
        lines.append("")
        return "\n".join(lines)
```

Update `STRATEGIES`:

```python
STRATEGIES = [
    BranchStrategy(),
    LoadStrategy(),
    StoreStrategy(),
    ComputeStrategy(),
]
```

**Important:** Update `generate_all()` to call `BranchStrategy.reset_labels()`
at the start of each batch to avoid label collision across separate .s files.

Also update `generate_operand_combos()` to skip immediate variation for branch
target operands (operand name `cpmaddr`):

```python
# In generate_operand_combos, when building alternatives:
# Skip branch target immediates -- they are always labels.
if op_name == "cpmaddr" or op_name.startswith("cpmaddr"):
    defaults[op_name] = "0"  # placeholder, overridden by strategy
    alternatives[op_name] = []
    continue
```

Add a `generate_combos()` method to the strategy interface so that
`BranchStrategy` can produce paired taken/not-taken combos:

```python
# In TestStrategy base:
def generate_combos(self, instr: dict) -> list[dict[str, str]]:
    """Generate operand combos for this instruction.
    Default delegates to the shared generate_operand_combos()."""
    return generate_operand_combos(instr)
```

Override in `BranchStrategy`:

```python
def generate_combos(self, instr: dict) -> list[dict[str, str]]:
    """Conditional branches get exactly 2 combos: taken + not-taken.
    Unconditional branches get 1 combo."""
    base_combo = {}
    # Branch target operands are always labels -- set placeholder.
    for op in instr.get("operands", []):
        op_name = op["name"]
        if op_name == "cpmaddr" or op_name.startswith("cpmaddr"):
            base_combo[op_name] = "0"
        elif op.get("operand_type") in REGISTER_LIKE_TYPES:
            kind = op.get("register_kind", "")
            bw = op.get("bit_width", 0)
            names = register_names(kind, bw, operand_type=op.get("operand_type", ""),
                                   instr_name=instr.get("name", ""))
            base_combo[op_name] = names[0] if names else "r0"
        else:
            base_combo[op_name] = "0"

    if not self._is_conditional(instr):
        return [base_combo]

    # Two combos: nonzero (taken for jnz, not-taken for jz) and zero.
    combo_nonzero = dict(base_combo)
    combo_zero = dict(base_combo)
    # The condition register operand is the first scalar in asm_string.
    asm_op_names = re.findall(r'\$(\w+)', instr.get("asm_string", ""))
    op_by_name = {op["name"]: op for op in instr.get("operands", [])}
    for name in asm_op_names:
        if name in op_by_name:
            op = op_by_name[name]
            if op.get("register_kind") == "scalar":
                # Both combos use the same register; the INPUT DATA
                # determines zero vs nonzero. We use different in_offsets
                # (the harness host fills input with PRNG data where
                # offset 0 is nonzero and we prepend a zero word).
                break
    return [combo_nonzero, combo_zero]
```

Then in `generate_all()`, replace the `generate_operand_combos(instr)` call
with `strategy.generate_combos(instr)`.

- [ ] **Step 4: Run BranchStrategy tests**

Run: `python3 -m pytest tools/test_isa_test_gen.py::TestBranchStrategy -v`
Expected: All PASS.

- [ ] **Step 5: Update branch-related tests in TestRealISA**

Remove or update tests that assert branches are always skipped:

```python
def test_branches_handled(self, isa_data):
    """Some branches should be testable, others deferred."""
    testable = 0
    deferred = 0
    for slot, instrs in isa_data.items():
        for instr in instrs:
            m = instr.get("mnemonic", "")
            if m in ("j", "jl", "jnz", "jz"):
                strategy, _ = isa_test_gen.classify_with_strategies(instr)
                if strategy is not None:
                    testable += 1
                else:
                    deferred += 1
            elif m in ("ret", "jnzd"):
                strategy, _ = isa_test_gen.classify_with_strategies(instr)
                assert strategy is None, f"{m} should be deferred"
                deferred += 1
    assert testable > 0
```

- [ ] **Step 6: Run full test suite**

Run: `python3 -m pytest tools/test_isa_test_gen.py -v`
Expected: All pass.

- [ ] **Step 7: Verify final coverage**

Run: `python3 tools/isa-test-gen.py --isa-json tools/aie2-isa.json --summary 2>&1 | head -20`
Expected: Testable count is ~350-406. Print the full skip reason breakdown.

- [ ] **Step 8: Commit**

```bash
git add tools/isa-test-gen.py tools/test_isa_test_gen.py
git commit -m "feat(isa-harness): add BranchStrategy for branch instruction testing"
```

---

### Task 6: Integration Testing and Harness Run

Generate the expanded test batches, assemble with llvm-mc, and run the
full HW+EMU comparison to validate the new strategies produce correct assembly.

**Files:**
- No new code changes -- this is a validation task.

- [ ] **Step 1: Generate expanded batches**

Run: `python3 tools/isa-test-gen.py --isa-json tools/aie2-isa.json --out-dir build/isa-tests`
Verify: Testable count is in the 350-406 range. Batch count is ~100-120.

- [ ] **Step 2: Assemble all batches with llvm-mc**

Run from the test in `test_isa_test_gen.py`:
```bash
python3 -m pytest tools/test_isa_test_gen.py::TestGenerateAll::test_batch_assembles_with_llvm_mc -v
```

If any batches fail to assemble, investigate the specific instruction that
produces invalid assembly and fix the strategy's `generate_test_point()`.

- [ ] **Step 3: Run full ISA harness (EMU-only first)**

Run: `XDNA_EMU=release nice -n 19 ./scripts/isa-test.sh --no-hw -j $(nproc)`
Verify: No crashes. EMU OK for all batches.

- [ ] **Step 4: Run full ISA harness with HW comparison**

Run: `XDNA_EMU=release nice -n 19 ./scripts/isa-test.sh -j $(nproc)`
Verify: Compare HW vs EMU results. Note any divergences.

- [ ] **Step 5: Investigate and fix any divergences**

For each divergent batch, use the manifest to identify which specific
instruction caused the mismatch. Check the emulator's implementation
of that instruction against the open-source toolchain.

- [ ] **Step 6: Commit any fixes from divergence investigation**

```bash
git add -u
git commit -m "fix(isa-harness): fix divergences found during expanded ISA testing"
```

- [ ] **Step 7: Update test status memory**

Update `~/.claude/projects/-home-triple-npu-work-xdna-emu/memory/test-status-20260308.md`
with the new ISA harness results (batch count, pass/fail, coverage percentage).

---

### Task 7: Add Per-Strategy Breakdown to --summary Output

Add strategy class name annotation and per-strategy counts to `--summary`.
The `--summary` code path already uses `classify_with_strategies()` (from
Task 2 Step 7), so this just adds the strategy name to the printed output.

**Files:**
- Modify: `tools/isa-test-gen.py` (main/summary section)

- [ ] **Step 1: Add strategy name tracking to --summary**

In the `--summary` branch of `main()`, track per-strategy counts:

```python
strategy_counts: dict[str, int] = {}
# ... in the classification loop:
strategy, reason = classify_with_strategies(instr)
if strategy is not None:
    testable_count += 1
    strategy_name = type(strategy).__name__
    strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
    testable_instrs.append((instr, strategy_name))
```

Print strategy breakdown after skip reasons:

```python
print("\nStrategy breakdown:")
for sname, count in sorted(strategy_counts.items()):
    print(f"  {sname}: {count}")
```

- [ ] **Step 2: Run and verify output**

Run: `python3 tools/isa-test-gen.py --isa-json tools/aie2-isa.json --summary`
Expected: Output shows per-strategy testable counts (ComputeStrategy: ~187,
LoadStrategy: ~119, StoreStrategy: ~94, BranchStrategy: ~6).

- [ ] **Step 4: Commit**

```bash
git add tools/isa-test-gen.py tools/test_isa_test_gen.py
git commit -m "feat(isa-harness): show per-strategy breakdown in --summary output"
```
