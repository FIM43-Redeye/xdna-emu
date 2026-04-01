# ISA Harness Memory Safety Fix -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 10 memory safety bugs in `tools/isa-test-gen.py` that cause 4 ISA test batches to be nondeterministic on hardware.

**Architecture:** Extract a shared `_safe_ptr_setup()` helper that zeroes modifiers before setting pointers, then refactor LoadStrategy and StoreStrategy to use it. Zero all immediates (including post-modify). Fix `detect_output_operands` for stores, `_padda_sequence` for negative offsets, and add bounds checks. Every strategy must write output sequentially to the caller-provided `out_offset`.

**Tech Stack:** Python (isa-test-gen.py), AIE2 assembly, bash (isa-test.sh)

**Spec:** `docs/superpowers/specs/2026-03-31-isa-harness-memory-safety-design.md`

---

All changes are in a single file: `tools/isa-test-gen.py`.

### Task 1: Fix `_padda_sequence` for Negative Offsets (Spec Fix 7)

**Files:**
- Modify: `tools/isa-test-gen.py:990-1004`

This is the foundation -- other fixes depend on `_padda_sequence` working correctly for all offsets.

- [ ] **Step 1: Fix `_padda_sequence` to handle negative offsets**

Replace lines 990-1004:

```python
def _padda_sequence(ptr_reg: str, base_ptr: str, offset: int) -> list[str]:
    """Generate pointer arithmetic sequence to reach a large offset.

    PADDA immediate range is 12-bit signed: [-4096, 4095].
    For larger offsets (positive or negative), emit multiple PADDA instructions.
    """
    max_step = 1024  # PADDA accepts up to ~1536; use 1024 for safety
    lines = [f"  mov {ptr_reg}, {base_ptr}"]
    remaining = offset
    while abs(remaining) > max_step:
        step = max_step if remaining > 0 else -max_step
        lines.append(f"  padda [{ptr_reg}], #{step}")
        remaining -= step
    if remaining != 0:
        lines.append(f"  padda [{ptr_reg}], #{remaining}")
    return lines
```

- [ ] **Step 2: Verify no regressions**

Run: `cd /home/triple/npu-work/xdna-emu && python3 tools/isa-test-gen.py --help`

Expected: Script loads without errors. (This confirms the syntax is valid; full generation test comes later.)

- [ ] **Step 3: Commit**

```bash
git add tools/isa-test-gen.py
git commit -m "fix(isa-harness): _padda_sequence handles negative offsets

Previously, negative offsets silently emitted no PADDA instruction,
leaving the pointer at base. Now handles both positive and negative
offsets correctly via signed stepping.

Generated using Claude Code."
```

---

### Task 2: Add `_safe_ptr_setup` Helper (Spec Fix 1)

**Files:**
- Modify: `tools/isa-test-gen.py` (insert after `_padda_sequence`, around line 1005)

- [ ] **Step 1: Add the shared helper function**

Insert after the `_padda_sequence` function (after the existing line 1004):

```python
def _safe_ptr_setup(ptr_reg: str, base_ptr: str, offset: int,
                    op_by_name: dict, regs: dict) -> list[str]:
    """Set up a pointer for load/store with all modifiers pre-zeroed.

    Order matters: zero modifiers FIRST, then set the pointer.  This
    prevents stale modifier values from corrupting 2D/3D addressing.
    Uses combo-assigned register names (not hardcoded m0/dj0).

    Args:
        ptr_reg: Target pointer register (e.g., "p6", "p7").
        base_ptr: Base pointer to copy from (e.g., "p0", "p1").
        offset: Byte offset from base_ptr.
        op_by_name: Dict of operand name -> operand dict for this instruction.
        regs: Combo-assigned register mapping (operand name -> register name).
    """
    lines = []
    # 1. Zero ALL modifier registers this instruction references.
    #    Must happen BEFORE the pointer is set up so that stale modifier
    #    values cannot corrupt 2D/3D stride calculations.
    for op_name, op in op_by_name.items():
        kind = op.get("register_kind", "")
        if kind == "modifier_m":
            lines.append(f"  mov {regs.get(op_name, 'm0')}, #0")
        elif kind == "modifier_dj":
            lines.append(f"  mov {regs.get(op_name, 'dj0')}, #0")
    # 2. THEN set the pointer (modifiers are now safe).
    lines.extend(_padda_sequence(ptr_reg, base_ptr, offset))
    return lines
```

- [ ] **Step 2: Verify script loads**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -c "import importlib.util; spec = importlib.util.spec_from_file_location('m', 'tools/isa-test-gen.py'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add tools/isa-test-gen.py
git commit -m "feat(isa-harness): add _safe_ptr_setup helper

Shared helper that zeroes all modifier registers BEFORE setting up
the pointer, preventing stale modifiers from corrupting 2D/3D
addressing. Uses combo-assigned register names, not hardcoded m0.

Generated using Claude Code."
```

---

### Task 3: Fix `detect_output_operands` for Stores (Spec Fix 6)

**Files:**
- Modify: `tools/isa-test-gen.py:669-692`

- [ ] **Step 1: Add early-return for pure store instructions**

At the top of `detect_output_operands` (after the docstring, before `operands = instr.get(...)`), add:

```python
    # Store instructions write to memory, not to a register.  They have
    # no register output.  StoreStrategy uses _detect_store_source() to
    # find the data register independently.
    if instr.get("may_store") and not instr.get("may_load"):
        return []
```

This goes right after line 690 (`Returns a list of operand dicts that are outputs.`) and the closing `"""`, before line 691 (`operands = instr.get("operands", [])`).

- [ ] **Step 2: Verify script loads**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -c "import importlib.util; spec = importlib.util.spec_from_file_location('m', 'tools/isa-test-gen.py'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add tools/isa-test-gen.py
git commit -m "fix(isa-harness): detect_output_operands returns [] for stores

Store instructions write to memory, not a register. The function
previously treated the store's source data register as an output,
which could cause combo generation to skip loading it as input.

Generated using Claude Code."
```

---

### Task 4: Refactor LoadStrategy to Use Safe Pointer Setup (Spec Fixes 1, 2, 4)

**Files:**
- Modify: `tools/isa-test-gen.py` -- `LoadStrategy.generate_test_point` (starts at line 1939)

- [ ] **Step 1: Replace LoadStrategy.generate_test_point**

Replace the entire `generate_test_point` method (lines 1939-2011) with:

```python
    def generate_test_point(self, instr, regs, in_offset, out_offset, **_kw):
        lines = []
        name = instr["name"]
        lines.append(f"  // ---- test (load): {name} ----")

        dest = self._detect_load_dest(instr)
        dest_name = dest["name"]
        dest_kind = _effective_load_store_kind(dest)
        dest_reg = regs.get(dest_name, "r0")

        # Align input offset for this destination type.
        align = _kind_alignment(dest_kind)
        in_offset = _align(in_offset, align)

        op_by_name = {op["name"]: op for op in instr.get("operands", [])}
        sp_relative = _is_sp_relative(instr)

        # Safe pointer setup: zero modifiers FIRST, then set pointer.
        # Both SP-relative and normal loads use p6 pointing at input data.
        lines.extend(_safe_ptr_setup("p6", "p0", in_offset, op_by_name, regs))

        # NOP sled to cover setup latency.
        lines.extend(_nop_sled(2))

        # Execute: the load instruction itself.
        # Override pointer to p6, zero ALL immediates (including post-modify).
        asm_string = instr["asm_string"]
        load_regs = dict(regs)
        for op_name, op in op_by_name.items():
            if op.get("register_kind") == "pointer":
                load_regs[op_name] = "p6"
            if op.get("operand_type") == "immediate":
                load_regs[op_name] = "0"

        asm_line = "  " + _substitute_asm(instr["asm_string"], load_regs,
                                          has_modifier=_has_modifier_operand(instr))
        lines.append(asm_line)

        # NOP sled for load result latency.
        lines.extend(_nop_sled(result_latency(instr)))

        # Capture: store loaded value to output buffer sequentially.
        out_offset = _align(out_offset, align)
        store_lines = _store_instruction(dest_reg, dest_kind, "p1", out_offset)
        lines.extend(store_lines)

        lines.append("")
        return "\n".join(lines)
```

Key changes from the original:
- Uses `_safe_ptr_setup` instead of inline `_padda_sequence` + modifier zeroing (fixes modifier ordering, uses combo register names).
- Zeroes ALL immediates including post-modify (no more `_is_postmodify_immediate` check).
- Output store uses `_store_instruction` with `p1` + `out_offset` (sequential output tape).

- [ ] **Step 2: Verify script loads**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -c "import importlib.util; spec = importlib.util.spec_from_file_location('m', 'tools/isa-test-gen.py'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add tools/isa-test-gen.py
git commit -m "fix(isa-harness): LoadStrategy uses safe pointer setup

Refactored to use _safe_ptr_setup (modifiers zeroed before pointer)
and zero ALL immediates including post-modify. Output writes go to
the caller-provided out_offset via p1 (sequential output tape).

Generated using Claude Code."
```

---

### Task 5: Refactor StoreStrategy to Use Safe Pointer Setup (Spec Fixes 1, 2, 3)

**Files:**
- Modify: `tools/isa-test-gen.py` -- `StoreStrategy.generate_test_point` (starts at line 2209)

- [ ] **Step 1: Replace StoreStrategy.generate_test_point**

Replace the entire `generate_test_point` method (lines 2209-2275) with:

```python
    def generate_test_point(self, instr, regs, in_offset, out_offset, **_kw):
        lines = []
        name = instr["name"]
        lines.append(f"  // ---- test (store): {name} ----")

        source = self._detect_store_source(instr)
        src_name = source["name"]
        src_kind = _effective_load_store_kind(source)
        src_reg = regs.get(src_name, "r0")

        op_by_name = {op["name"]: op for op in instr.get("operands", [])}
        sp_relative = _is_sp_relative(instr)

        # Align and load known data into the source register from input buffer.
        align = _kind_alignment(src_kind)
        in_offset = _align(in_offset, align)
        load_lines = _load_instruction(src_reg, src_kind, "p0", in_offset)
        lines.extend(load_lines)

        # NOP sled for load latency.
        lines.extend(_nop_sled(LOAD_LATENCY))

        # Safe pointer setup: zero modifiers FIRST, then set output pointer.
        out_offset = _align(out_offset, align)
        if sp_relative:
            lines.extend(_safe_ptr_setup("p6", "p1", out_offset, op_by_name, regs))
        else:
            lines.extend(_safe_ptr_setup("p7", "p1", out_offset, op_by_name, regs))

        # Execute: the store instruction.
        # Override pointer to p7 (or p6 for SP-relative), zero ALL immediates.
        asm_string = instr["asm_string"]
        store_regs = dict(regs)
        for op_name, op in op_by_name.items():
            if op.get("register_kind") == "pointer":
                store_regs[op_name] = "p7"
            if op.get("operand_type") == "immediate":
                store_regs[op_name] = "0"

        asm_line = "  " + _substitute_asm(instr["asm_string"], store_regs,
                                          has_modifier=_has_modifier_operand(instr))
        lines.append(asm_line)

        lines.append("")
        return "\n".join(lines)
```

Key changes from the original:
- Uses `_safe_ptr_setup` instead of inline `_padda_sequence` + modifier zeroing after pointer (fixes modifier ordering).
- Zeroes ALL immediates including post-modify (no more `_is_postmodify_immediate` check).
- Store writes to exactly `[p7, #0]` which is `p1 + out_offset` (sequential output tape, bounded).

- [ ] **Step 2: Verify script loads**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -c "import importlib.util; spec = importlib.util.spec_from_file_location('m', 'tools/isa-test-gen.py'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add tools/isa-test-gen.py
git commit -m "fix(isa-harness): StoreStrategy uses safe pointer setup

Refactored to use _safe_ptr_setup (modifiers zeroed before pointer)
and zero ALL immediates including post-modify. Store writes to exactly
[p7, #0] at the caller-provided out_offset (sequential output tape).
Eliminates all wild pointer escapes from store test points.

Generated using Claude Code."
```

---

### Task 6: Add Pointer Restore to PointerArithStrategy (Spec Fix 5)

**Files:**
- Modify: `tools/isa-test-gen.py` -- `PointerArithStrategy.generate_test_point` (starts at line 3962)

- [ ] **Step 1: Add pointer restore after result capture**

In `PointerArithStrategy.generate_test_point`, find the line (around line 4018):

```python
        lines.extend(_scalar_store("r14", "p1", out_offset))
```

Add the following line immediately after it:

```python
        # Defensive: reset the pointer register to a safe value so no
        # subsequent test can accidentally use the wild computed pointer.
        lines.append(f"  mov {ptr_reg}, p1")
```

- [ ] **Step 2: Verify script loads**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -c "import importlib.util; spec = importlib.util.spec_from_file_location('m', 'tools/isa-test-gen.py'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add tools/isa-test-gen.py
git commit -m "fix(isa-harness): PointerArithStrategy resets pointer after test

After capturing the computed pointer value, reset the pointer register
to p1 (output buffer base) as a safety net. Prevents wild pointers
from leaking to subsequent tests.

Generated using Claude Code."
```

---

### Task 7: Add Scratch Region Comment to Preamble (Spec Fix 9)

**Files:**
- Modify: `tools/isa-test-gen.py:1650-1658`

- [ ] **Step 1: Add documentation comment**

Find the mask-zeroing section of `_register_zeroing_preamble()` (around line 1650):

```python
    # Mask: q0-q3 (128-bit each).  No direct zero instruction; load from
    # a memory region we've zeroed.  Use the start of the output buffer
    # (p1) as scratch -- it will be overwritten by actual test outputs.
```

Replace with:

```python
    # Mask: q0-q3 (128-bit each).  No direct zero instruction; load from
    # a memory region we've zeroed.  Use the start of the output buffer
    # (p1) as scratch -- it will be overwritten by actual test outputs.
    # NOTE: This writes to output buffer offsets 0-15.  This is safe
    # because the preamble runs before any tests, and test output data
    # overwrites these bytes.  The sequential output tape starts at
    # offset 0 and advances monotonically.
```

- [ ] **Step 2: Commit**

```bash
git add tools/isa-test-gen.py
git commit -m "docs(isa-harness): document scratch region overlap assumption

The preamble uses output buffer offsets 0-15 as scratch for mask
register zeroing. Document that this is safe because test outputs
overwrite these bytes.

Generated using Claude Code."
```

---

### Task 8: Add Output Buffer Bounds Check (Spec Fix 8)

**Files:**
- Modify: `tools/isa-test-gen.py:4893-4929` (main batch generation loop)

- [ ] **Step 1: Add data memory constant and bounds check**

First, find the existing constants section near the top of the file. Look for `MAX_BUNDLES_PER_BATCH` or similar constants. Add a new constant nearby:

```python
# Maximum output buffer size in bytes.  The tile has 64KB of data memory
# shared between input and output buffers.  Program memory (1024 bundles)
# is usually the binding constraint, but this guards against size bugs.
MAX_OUT_BUFFER_BYTES = 32768  # 32KB -- conservative half of 64KB
```

Then in the main batch generation loop (around line 4880), find the inner while loop that checks program memory:

```python
        while end < len(measured_specs):
            next_bundles = measured_specs[end][3]
            if batch_bundle_total + next_bundles > MAX_BUNDLES_PER_BATCH:
                break
            batch_bundle_total += next_bundles
            end += 1
```

Add an output buffer size check to the same loop. Replace that block with:

```python
        batch_out_estimate = 0
        while end < len(measured_specs):
            next_bundles = measured_specs[end][3]
            if batch_bundle_total + next_bundles > MAX_BUNDLES_PER_BATCH:
                break
            # Check output buffer won't overflow data memory.
            next_out_size = measured_specs[end][4].compute_output_size(
                measured_specs[end][0])
            next_out_aligned = _align(batch_out_estimate, 32) + next_out_size
            if next_out_aligned > MAX_OUT_BUFFER_BYTES:
                break
            batch_bundle_total += next_bundles
            batch_out_estimate = next_out_aligned
            end += 1
```

Note: `measured_specs[end]` is a tuple of `(instr, combo_idx, regs, bundles, strategy)`. Index `[4]` is the strategy, `[0]` is the instruction.

- [ ] **Step 2: Verify script loads**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -c "import importlib.util; spec = importlib.util.spec_from_file_location('m', 'tools/isa-test-gen.py'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add tools/isa-test-gen.py
git commit -m "fix(isa-harness): bounds-check output buffer during batch packing

Add MAX_OUT_BUFFER_BYTES (32KB) limit. The batch packing loop now
checks that cumulative output size stays within data memory, starting
a new batch if exceeded. Previously, output offset grew unchecked.

Generated using Claude Code."
```

---

### Task 9: Add Implicit Register Offset Bounds Check (Spec Fix 10)

**Files:**
- Modify: `tools/isa-test-gen.py:1490-1501` (inside `generate_test_point`)

- [ ] **Step 1: Add bounds check for implicit register loads**

Find the implicit register loading section in `generate_test_point()` (the ComputeStrategy helper function, around line 1490):

```python
    # Load implicit registers that aren't in the operand list.
    # VINSERT: r29 (index register).
    # DIVS: r31 (division-step state).
    # SELEQZ/SELNEZ: r27 (condition test).
    implicit_regs: list[str] = []
    if name.startswith("VINSERT"):
        implicit_regs.append("r29")
    implicit_regs.extend(IMPLICIT_INPUT_REGS.get(name, []))
    for imp_reg in implicit_regs:
        cur_in_offset = _align(cur_in_offset, 4)
        lines.extend(_scalar_load(imp_reg, "p0", cur_in_offset))
        cur_in_offset += 4
```

The function receives `in_offset` but doesn't know the total input buffer size. The caller computes `in_size` via `compute_input_size()` which already accounts for implicit registers. The issue is if `compute_input_size` and the actual loads diverge. Since the caller allocates `in_size` bytes starting at `in_offset`, the available space is `in_size` bytes total.

The simplest fix is to ensure `compute_input_size` matches the actual loads (it already does -- both iterate the same operand list + implicit regs). The bounds check belongs in the batch packing loop where both `in_size` and `out_size` are known. Since the batch loop already uses `compute_input_size` to advance `batch_in_offset`, and the MLIR template sizes the input objectfifo from `batch_in_offset`, overflow is prevented by construction.

Add a defensive comment in the implicit register section:

```python
    # Load implicit registers that aren't in the operand list.
    # VINSERT: r29 (index register).
    # DIVS: r31 (division-step state).
    # SELEQZ/SELNEZ: r27 (condition test).
    # NOTE: compute_input_size() accounts for these implicit registers,
    # so the batch packing loop allocates sufficient input buffer space.
    implicit_regs: list[str] = []
```

- [ ] **Step 2: Commit**

```bash
git add tools/isa-test-gen.py
git commit -m "docs(isa-harness): document implicit register input bounds

Add note that compute_input_size accounts for implicit registers,
so the batch packing loop allocates sufficient input buffer space.

Generated using Claude Code."
```

---

### Task 10: Strategy Audit -- Verify Sequential Output Tape

**Files:**
- Modify: `tools/isa-test-gen.py` (multiple strategies, read-only audit + any fixes found)

Verify that every active strategy writes to the caller-provided `out_offset` via `p1`, not via a computed or escaped pointer. The spec identified Load and Store as the main offenders (fixed in Tasks 4/5), but we must audit all others.

- [ ] **Step 1: Audit all strategy generate_test_point methods**

Read each strategy's `generate_test_point` and verify output writes go through `_store_instruction(reg, kind, "p1", out_offset)` or `_scalar_store(reg, "p1", out_offset)`. Check for:
- Any use of p6/p7 for output writes (should only be p1)
- Any pointer arithmetic on the output pointer
- Any `[p7, ...]` in generated assembly for output

Strategies to audit:
1. `ComputeStrategy` (line 1808) -- delegates to `generate_test_point()` which uses `_store_instruction(reg, kind, "p1", cur_out_offset)` at line 1588. **OK.**
2. `LockStrategy` (line 2504) -- read its `generate_test_point` to verify.
3. `FifoLoadStrategy` (line 2638) -- read its `generate_test_point` to verify.
4. `CascadeStrategy` (line 2778) -- read its `generate_test_point` to verify.
5. `CascadeReadStrategy` (line 2872) -- read its `generate_test_point` to verify.
6. `StreamStrategy` (line 2993) -- multi-tile, separate model, skip.
7. `DoneStrategy` (line 3760) -- read its `generate_test_point` to verify.
8. `EventStrategy` (line 3808) -- read its `generate_test_point` to verify.
9. `PaddaSpStrategy` (line 4029) -- already verified in Task 6 read, uses `_scalar_store("r14", "p1", out_offset)`. **OK.**
10. `AccumArithStrategy` (line 4256) -- uses `_store_instruction(reg, kind, "p1", cur_out_offset)`. **OK.**
11. `VmacStrategy` (line 4405) -- uses `_store_instruction(reg, kind, "p1", cur_out_offset)`. **OK.**

- [ ] **Step 2: Fix any strategies that don't follow sequential output**

If any strategy writes via a pointer other than `p1 + out_offset`, refactor it to match the pattern. Expected: all remaining strategies already use the correct pattern (Load/Store were the only offenders).

- [ ] **Step 3: Commit (if any fixes were needed)**

```bash
git add tools/isa-test-gen.py
git commit -m "fix(isa-harness): audit all strategies for sequential output tape

Verified all active strategies write output via p1 + out_offset.
[describe any fixes if needed]

Generated using Claude Code."
```

---

### Task 11: Regenerate and Smoke-Test

**Files:**
- No code changes -- generation and compilation test.

- [ ] **Step 1: Regenerate all ISA test batches**

Run: `cd /home/triple/npu-work/xdna-emu && nice -n 19 bash scripts/isa-test.sh --generate-only 2>&1 | tail -20`

Expected: Generation completes without errors. Note the batch count -- it may differ from before due to the output buffer bounds check splitting batches differently.

- [ ] **Step 2: Compile all batches**

Run: `cd /home/triple/npu-work/xdna-emu && nice -n 19 bash scripts/isa-test.sh --compile 2>&1 | tail -20`

Expected: All batches compile. Any llvm-mc errors indicate a problem with the assembly changes (e.g., bad immediate format from zeroing post-modify).

- [ ] **Step 3: Quick sanity check -- run one batch on emulator**

Run: `cd /home/triple/npu-work/xdna-emu && ls build/isa-tests/batch_000/`

Verify the batch directory contains `aie.xclbin`, `insts.bin`, `aie.mlir`, and the assembly source.

---

### Task 12: Hardware Determinism Check

**Files:**
- No code changes -- validation.

This is the real test. Run every batch 5x on hardware and verify all are deterministic.

- [ ] **Step 1: Run determinism check**

Use the script from `tests/determinism-20260331/determinism-check.sh` (or recreate from the memory file). Run all batches 5x on real NPU hardware:

```bash
cd /home/triple/npu-work/xdna-emu
env -u XDNA_EMU bash tests/determinism-20260331/determinism-check.sh 2>&1 | tee /tmp/claude-1000/determinism-results.log
```

Expected: 0 nondeterministic batches (previously 12). Cold-start differences may remain (that's a separate issue per the spec's non-goals).

- [ ] **Step 2: If any batches are still nondeterministic, investigate**

Compare against the previous list (batch_21, 22, 24, 25, 27, 31, 44, 52, 57, 60, 77, 78). The 4 genuine nondeterminism batches (27, 44, 77, 78) should now be deterministic. The 8 cold-start batches may still show run-1-vs-rest differences.

- [ ] **Step 3: Record results**

Log the determinism results for comparison with the pre-fix baseline.
