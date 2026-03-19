# Cascade Read Multi-Tile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable testing of 3 cascade read instructions (vmov/vmov.hi/vmov.lo from SCD) by generating producer-consumer tile pairs.

**Architecture:** CascadeReadStrategy in isa-test-gen.py generates two standalone assembly programs per test point (producer writes MCD, consumer reads SCD). isa-multi-tile-gen.py generates MLIR with cascade_flow declarations, two-hop objectfifos through memtile, and dual link_with references.

**Tech Stack:** Python 3, AIE2 assembly (llvm-mc), MLIR (aiecc.py), pytest

**Spec:** `docs/superpowers/specs/2026-03-19-cascade-read-multi-tile-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `tools/isa-test-gen.py` | Add CascadeReadStrategy, integrate cascade pairs into generate_all() |
| `tools/test_isa_test_gen.py` | Unit tests for CascadeReadStrategy |
| `tools/isa-multi-tile-gen.py` | Handle cascade_pair source_type in MLIR generation |

---

### Task 1: CascadeReadStrategy -- can_test and combos

**Files:**
- Modify: `tools/test_isa_test_gen.py` (after TestCascadeStrategy, ~line 2737)
- Modify: `tools/isa-test-gen.py` (~line 2650, after CascadeStrategy; ~line 3134 STRATEGIES list)

- [ ] **Step 1: Write failing tests**

Add to `tools/test_isa_test_gen.py` before the `TestGenerateAll` class:

```python
# ===================================================================
# CascadeReadStrategy tests
# ===================================================================

class TestCascadeReadStrategy:
    """Tests for CascadeReadStrategy -- multi-tile cascade read testing."""

    def _make_vmov_scd(self):
        return _make_instr("VMOV_mv_scd", "vmov", "vmov\t$dst, SCD", [
            {"name": "dst", "bit_width": 6, "operand_type": "composite_register",
             "register_kind": "MvBMXDst", "is_output": False,
             "signed": False, "scale": None},
        ], slot="lda")

    def _make_vmov_hi_scd(self):
        return _make_instr("VMOV_HI", "vmov.hi", "vmov.hi\t$dst, SCD", [
            {"name": "dst", "bit_width": 4, "operand_type": "unknown",
             "is_output": False, "signed": False, "scale": None},
        ], slot="lda")

    def _make_vmov_lo_scd(self):
        return _make_instr("VMOV_LO", "vmov.lo", "vmov.lo\t$dst, SCD", [
            {"name": "dst", "bit_width": 4, "operand_type": "unknown",
             "is_output": False, "signed": False, "scale": None},
        ], slot="lda")

    def _make_mcd_write(self):
        return _make_instr("VMOV_mv_mcd", "vmov", "vmov\tMCD, $src", [
            {"name": "src", "bit_width": 6, "operand_type": "composite_register",
             "register_kind": "MvBMXDst", "is_output": False,
             "signed": False, "scale": None},
        ], slot="st")

    def test_vmov_scd_can_test(self):
        strategy = isa_test_gen.CascadeReadStrategy()
        can, reason = strategy.can_test(self._make_vmov_scd())
        assert can, f"Expected can_test=True: {reason}"

    def test_vmov_hi_scd_can_test(self):
        strategy = isa_test_gen.CascadeReadStrategy()
        can, reason = strategy.can_test(self._make_vmov_hi_scd())
        assert can, f"Expected can_test=True: {reason}"

    def test_vmov_lo_scd_can_test(self):
        strategy = isa_test_gen.CascadeReadStrategy()
        can, reason = strategy.can_test(self._make_vmov_lo_scd())
        assert can, f"Expected can_test=True: {reason}"

    def test_mcd_write_rejected(self):
        strategy = isa_test_gen.CascadeReadStrategy()
        can, _ = strategy.can_test(self._make_mcd_write())
        assert not can

    def test_non_cascade_rejected(self):
        strategy = isa_test_gen.CascadeReadStrategy()
        instr = _make_instr("ADD", "add", "add\t$mRx, $mRy, $mRz", [
            _make_reg_op("mRx", "scalar"),
        ])
        can, _ = strategy.can_test(instr)
        assert not can

    def test_combo_forces_x0(self):
        """Destination register should always be x0."""
        strategy = isa_test_gen.CascadeReadStrategy()
        combos = strategy.generate_combos(self._make_vmov_scd())
        assert len(combos) == 1
        assert combos[0]["dst"] == "x0"

    def test_producer_input_size(self):
        strategy = isa_test_gen.CascadeReadStrategy()
        assert strategy.compute_producer_input_size() == 64

    def test_producer_output_size(self):
        strategy = isa_test_gen.CascadeReadStrategy()
        assert strategy.compute_producer_output_size() == 8

    def test_consumer_input_size(self):
        strategy = isa_test_gen.CascadeReadStrategy()
        assert strategy.compute_consumer_input_size() == 0

    def test_consumer_output_size_full(self):
        strategy = isa_test_gen.CascadeReadStrategy()
        assert strategy.compute_consumer_output_size(self._make_vmov_scd()) == 64

    def test_consumer_output_size_half(self):
        strategy = isa_test_gen.CascadeReadStrategy()
        assert strategy.compute_consumer_output_size(self._make_vmov_hi_scd()) == 32
        assert strategy.compute_consumer_output_size(self._make_vmov_lo_scd()) == 32
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tools/test_isa_test_gen.py -k "TestCascadeReadStrategy" -q`
Expected: 11 FAILED (AttributeError: no attribute 'CascadeReadStrategy')

- [ ] **Step 3: Implement CascadeReadStrategy (can_test, combos, sizes)**

Add to `tools/isa-test-gen.py` after `CascadeStrategy` class (~line 2650), before `ConversionStrategy`:

```python
class CascadeReadStrategy(TestStrategy):
    """Tests cascade read instructions via multi-tile producer-consumer pairs.

    Cascade reads (vmov $dst, SCD / vmov.hi / vmov.lo) stall without data
    from a neighboring tile.  This strategy generates TWO assembly programs:
    a producer (loads data, writes MCD) and a consumer (reads SCD, stores
    result).  The MLIR generator places them in the same column with
    aie.cascade_flow linking row 3 (producer) to row 2 (consumer).

    These are NOT bin-packed into mega-program batches.  Each cascade read
    instruction becomes a standalone cascade_pair batch.
    """

    def can_test(self, instr: dict) -> tuple[bool, str]:
        asm = instr.get("asm_string", "")
        if "SCD" in asm:
            return (True, "")
        return (False, "not a cascade read instruction")

    def _is_half(self, instr: dict) -> bool:
        """True for vmov.hi/vmov.lo (256-bit half reads)."""
        mnemonic = instr.get("mnemonic", "")
        return mnemonic in ("vmov.hi", "vmov.lo")

    def generate_combos(self, instr: dict) -> list[dict[str, str]]:
        return [{"dst": "x0"}]

    # -- Size methods for producer and consumer separately --

    def compute_producer_input_size(self) -> int:
        return 64  # 512-bit vector loaded from host

    def compute_producer_output_size(self) -> int:
        return 8  # before + after markers

    def compute_consumer_input_size(self) -> int:
        return 0  # data arrives via cascade, not host

    def compute_consumer_output_size(self, instr: dict) -> int:
        return 32 if self._is_half(instr) else 64

    # Required by TestStrategy base but not used for cascade pairs.
    def compute_input_size(self, instr, regs):
        return self.compute_producer_input_size()

    def compute_output_size(self, instr):
        return self.compute_producer_output_size() + self.compute_consumer_output_size(instr)

    def generate_test_point(self, instr, regs, in_offset, out_offset, **_kw):
        raise NotImplementedError(
            "CascadeReadStrategy uses generate_cascade_pair(), not generate_test_point()"
        )
```

Add `CascadeReadStrategy()` to the `STRATEGIES` list at line ~3134, BEFORE `CascadeStrategy`:

```python
STRATEGIES: list[TestStrategy] = [
    BranchStrategy(),
    LockStrategy(),
    FifoLoadStrategy(),
    CascadeReadStrategy(),   # must be before CascadeStrategy
    CascadeStrategy(),
    DoneStrategy(),
    EventStrategy(),
    PaddaSpStrategy(),
    LoadStrategy(),
    StoreStrategy(),
    ComputeStrategy(),
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tools/test_isa_test_gen.py -k "TestCascadeReadStrategy" -q`
Expected: 12 passed

- [ ] **Step 5: Run all tests to verify no regressions**

Run: `python -m pytest tools/test_isa_test_gen.py -q`
Expected: 352 passed (341 existing + 11 new)

- [ ] **Step 6: Commit**

```bash
git add tools/isa-test-gen.py tools/test_isa_test_gen.py
git commit -m "feat(isa-harness): add CascadeReadStrategy can_test, combos, sizes"
```

---

### Task 2: CascadeReadStrategy -- assembly generation

**Files:**
- Modify: `tools/test_isa_test_gen.py`
- Modify: `tools/isa-test-gen.py`

- [ ] **Step 1: Write failing tests for generate_cascade_pair**

Add to `TestCascadeReadStrategy` class in `tools/test_isa_test_gen.py`:

```python
    def test_producer_asm_loads_data(self):
        strategy = isa_test_gen.CascadeReadStrategy()
        regs = {"dst": "x0"}
        result = strategy.generate_cascade_pair(self._make_vmov_scd(), regs)
        prod = result["producer_asm"]
        assert "vlda" in prod
        assert "wl0" in prod and "wh0" in prod

    def test_producer_asm_has_nop_sled(self):
        strategy = isa_test_gen.CascadeReadStrategy()
        regs = {"dst": "x0"}
        result = strategy.generate_cascade_pair(self._make_vmov_scd(), regs)
        prod = result["producer_asm"]
        assert prod.count("nop") >= 5

    def test_producer_asm_enables_mcd(self):
        strategy = isa_test_gen.CascadeReadStrategy()
        regs = {"dst": "x0"}
        result = strategy.generate_cascade_pair(self._make_vmov_scd(), regs)
        prod = result["producer_asm"]
        assert "crMCDEn" in prod

    def test_producer_asm_writes_mcd(self):
        strategy = isa_test_gen.CascadeReadStrategy()
        regs = {"dst": "x0"}
        result = strategy.generate_cascade_pair(self._make_vmov_scd(), regs)
        prod = result["producer_asm"]
        assert "MCD" in prod and "vmov" in prod

    def test_producer_asm_has_markers(self):
        strategy = isa_test_gen.CascadeReadStrategy()
        regs = {"dst": "x0"}
        result = strategy.generate_cascade_pair(self._make_vmov_scd(), regs)
        prod = result["producer_asm"]
        assert "#170" in prod  # before
        assert "#204" in prod  # after

    def test_consumer_asm_enables_scd(self):
        strategy = isa_test_gen.CascadeReadStrategy()
        regs = {"dst": "x0"}
        result = strategy.generate_cascade_pair(self._make_vmov_scd(), regs)
        cons = result["consumer_asm"]
        assert "crSCDEn" in cons

    def test_consumer_asm_reads_scd(self):
        strategy = isa_test_gen.CascadeReadStrategy()
        regs = {"dst": "x0"}
        result = strategy.generate_cascade_pair(self._make_vmov_scd(), regs)
        cons = result["consumer_asm"]
        assert "SCD" in cons and "vmov" in cons

    def test_consumer_asm_stores_to_p0(self):
        """Consumer has one arg (output), so p0 = output buffer."""
        strategy = isa_test_gen.CascadeReadStrategy()
        regs = {"dst": "x0"}
        result = strategy.generate_cascade_pair(self._make_vmov_scd(), regs)
        cons = result["consumer_asm"]
        assert "vst" in cons and "p0" in cons

    def test_consumer_full_stores_both_halves(self):
        strategy = isa_test_gen.CascadeReadStrategy()
        regs = {"dst": "x0"}
        result = strategy.generate_cascade_pair(self._make_vmov_scd(), regs)
        cons = result["consumer_asm"]
        assert "wl0" in cons and "wh0" in cons

    def test_consumer_hi_stores_high_half_only(self):
        strategy = isa_test_gen.CascadeReadStrategy()
        regs = {"dst": "x0"}
        result = strategy.generate_cascade_pair(self._make_vmov_hi_scd(), regs)
        cons = result["consumer_asm"]
        assert "wh0" in cons
        # Should NOT store wl0
        lines = [l for l in cons.split("\n") if "vst" in l]
        assert len(lines) == 1

    def test_consumer_lo_stores_low_half_only(self):
        strategy = isa_test_gen.CascadeReadStrategy()
        regs = {"dst": "x0"}
        result = strategy.generate_cascade_pair(self._make_vmov_lo_scd(), regs)
        cons = result["consumer_asm"]
        assert "wl0" in cons
        lines = [l for l in cons.split("\n") if "vst" in l]
        assert len(lines) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tools/test_isa_test_gen.py -k "TestCascadeReadStrategy and producer" -q`
Expected: FAILED (AttributeError: no attribute 'generate_cascade_pair')

- [ ] **Step 3: Implement generate_cascade_pair**

Add to `CascadeReadStrategy` class in `tools/isa-test-gen.py`:

```python
    def generate_cascade_pair(self, instr: dict, regs: dict) -> dict:
        """Generate producer and consumer assembly programs.

        Returns:
            Dict with 'producer_asm' and 'consumer_asm' strings, each a
            complete test_kernel function body (without the mega-program
            wrapper -- that is added by build_mega_program()).
        """
        return {
            "producer_asm": self._generate_producer(),
            "consumer_asm": self._generate_consumer(instr, regs),
        }

    def _generate_producer(self) -> str:
        """Generic producer: load data from p0, write to MCD, markers to p1."""
        LOAD_LATENCY = 5
        lines = []
        lines.append("  // ---- cascade producer: load + write MCD ----")

        # Load 64 bytes from input buffer (p0) into x0.
        lines.append("  vlda wl0, [p0, #0]")
        lines.append("  vlda wh0, [p0, #32]")
        lines.extend(_nop_sled(LOAD_LATENCY))

        # Store before marker to output buffer (p1).
        lines.append("  mov r14, #170")
        lines.extend(_scalar_store("r14", "p1", 0))

        # Enable cascade output.
        lines.append("  mov r14, #1")
        lines.append("  mov crMCDEn, r14")

        # Write to MCD.
        lines.append("  vmov MCD, x0")

        # Store after marker.
        lines.append("  mov r14, #204")
        lines.extend(_scalar_store("r14", "p1", 4))

        lines.append("")
        return "\n".join(lines)

    def _generate_consumer(self, instr: dict, regs: dict) -> str:
        """Consumer: enable SCD, read cascade, store result to p0."""
        mnemonic = instr.get("mnemonic", "")
        asm_str = instr.get("asm_string", "")
        name = instr["name"]

        lines = []
        lines.append(f"  // ---- cascade consumer: {name} ----")

        # Enable cascade input.
        lines.append("  mov r14, #1")
        lines.append("  mov crSCDEn, r14")

        # Execute the cascade read instruction under test.
        asm_line = _substitute_asm(asm_str, regs)
        lines.append(f"  {asm_line}")

        # Store result to output buffer.
        # Consumer has 1 arg (output memref) -> pointer is p0.
        if mnemonic == "vmov.hi":
            lines.append("  vst wh0, [p0, #0]")
        elif mnemonic == "vmov.lo":
            lines.append("  vst wl0, [p0, #0]")
        else:
            # Full 512-bit read.
            lines.append("  vst wl0, [p0, #0]")
            lines.append("  vst wh0, [p0, #32]")

        lines.append("")
        return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tools/test_isa_test_gen.py -k "TestCascadeReadStrategy" -q`
Expected: 23 passed

- [ ] **Step 5: Commit**

```bash
git add tools/isa-test-gen.py tools/test_isa_test_gen.py
git commit -m "feat(isa-harness): CascadeReadStrategy assembly generation"
```

---

### Task 3: Integrate cascade pairs into generate_all()

**Files:**
- Modify: `tools/isa-test-gen.py` (generate_all function, ~line 3255)
- Modify: `tools/test_isa_test_gen.py` (TestGenerateAll class)

- [ ] **Step 1: Write failing test**

Add to `TestGenerateAll` class in `tools/test_isa_test_gen.py`:

```python
    def test_cascade_pairs_in_manifest(self, isa_json_path, out_dir):
        """Cascade read instructions should produce cascade_pair batches."""
        manifest = generate_all(isa_json_path, out_dir)
        cascade_batches = [
            b for b in manifest["batches"]
            if b.get("source_type") == "cascade_pair"
        ]
        # 3 cascade read instructions: vmov SCD, vmov.hi SCD, vmov.lo SCD
        assert len(cascade_batches) == 3

    def test_cascade_pair_has_both_files(self, isa_json_path, out_dir):
        """Each cascade_pair batch should have producer and consumer filenames."""
        manifest = generate_all(isa_json_path, out_dir)
        cascade_batches = [
            b for b in manifest["batches"]
            if b.get("source_type") == "cascade_pair"
        ]
        for batch in cascade_batches:
            assert "producer_filename" in batch
            assert "consumer_filename" in batch
            prod_path = os.path.join(out_dir, batch["producer_filename"])
            cons_path = os.path.join(out_dir, batch["consumer_filename"])
            assert os.path.exists(prod_path), f"Missing {prod_path}"
            assert os.path.exists(cons_path), f"Missing {cons_path}"

    def test_cascade_reads_now_testable(self, isa_json_path, out_dir):
        """Cascade reads should no longer appear in skip reasons."""
        manifest = generate_all(isa_json_path, out_dir)
        skip_reasons = manifest.get("skip_reasons", {})
        assert "cascade read (stalls without neighboring tile)" not in skip_reasons
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tools/test_isa_test_gen.py -k "test_cascade_pairs_in_manifest or test_cascade_pair_has_both_files or test_cascade_reads_now_testable" -q`
Expected: 3 FAILED

- [ ] **Step 3: Integrate cascade pairs into generate_all()**

In `tools/isa-test-gen.py`, modify `generate_all()`:

**3a.** Add a `cascade_specs` list alongside `test_point_specs` (~line 3292):

```python
    # (instr, combo_idx, regs, strategy) -- cascade pair test points.
    cascade_specs: list[tuple[dict, int, dict, CascadeReadStrategy]] = []
```

**3b.** In the main classification loop (~line 3312), divert cascade read
matches to `cascade_specs` instead of `test_point_specs`:

After `strategy, reason = classify_with_strategies(instr)` and before
`testable_count += 1`, add:

```python
        if isinstance(strategy, CascadeReadStrategy):
            testable_count += 1
            combos = strategy.generate_combos(instr)
            for combo_idx, regs in enumerate(combos):
                cascade_specs.append((instr, combo_idx, regs, strategy))
            continue
```

**3c.** After the conversion batch block (~line 3468), add cascade pair
generation:

```python
    # Generate cascade pair batches (standalone, not bin-packed).
    for instr, combo_idx, regs, strategy in cascade_specs:
        pair = strategy.generate_cascade_pair(instr, regs)

        # Wrap each in a standalone program.
        prod_program = build_mega_program([pair["producer_asm"]])
        cons_program = build_mega_program([pair["consumer_asm"]])

        prod_filename = f"batch_{batch_idx:03d}_producer.s"
        cons_filename = f"batch_{batch_idx:03d}_consumer.s"

        with open(os.path.join(out_dir, prod_filename), "w") as f:
            f.write(prod_program)
        with open(os.path.join(out_dir, cons_filename), "w") as f:
            f.write(cons_program)

        cons_out = strategy.compute_consumer_output_size(instr)
        prod_in = strategy.compute_producer_input_size()
        prod_out = strategy.compute_producer_output_size()

        batches.append({
            "batch_index": batch_idx,
            # "filename" points to consumer .s for compatibility with
            # existing TestGenerateAll tests that iterate batch["filename"].
            "filename": cons_filename,
            "source_type": "cascade_pair",
            "producer_filename": prod_filename,
            "consumer_filename": cons_filename,
            "producer_in_size": prod_in,
            "producer_out_size": prod_out,
            "consumer_in_size": 0,
            "consumer_out_size": cons_out,
            "instruction": instr["name"],
            "slot": instr.get("slot", ""),
            "test_count": 1,
            "in_size": prod_in,
            "out_size": prod_out + cons_out,
            # "tests" list for compatibility with existing test assertions.
            "tests": [{
                "instruction": instr["name"],
                "slot": instr.get("slot", ""),
                "combo_index": combo_idx,
                "operands": {k: v for k, v in regs.items()
                             if not k.startswith("_")},
                "in_offset": 0,
                "in_size": prod_in,
                "out_offset": 0,
                "out_size": prod_out + cons_out,
            }],
        })
        batch_idx += 1
```

**3d.** Update `total_test_points` (~line 3471) to include cascade specs:

```python
    total_test_points = (len(test_point_specs)
                         + total_conv_test_points
                         + len(cascade_specs))
```

**3e.** Remove the SCD skip from `classify_instruction()` (~line 587-588).
Delete these two lines:

```python
    if "SCD" in asm:
        return ("skipped", "cascade read (stalls without neighboring tile)")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tools/test_isa_test_gen.py -k "test_cascade_pairs or test_cascade_reads_now" -q`
Expected: 3 passed

- [ ] **Step 5: Run all tests and check coverage**

Run: `python -m pytest tools/test_isa_test_gen.py -q`
Expected: all passed (no regressions)

Run: `python tools/isa-test-gen.py --summary 2>&1 | head -10`
Expected: Testable: 573, Skipped: 33

- [ ] **Step 6: Verify assembly compiles**

```bash
LLVM_MC=$HOME/npu-work/llvm-aie/build/bin/llvm-mc
TMPDIR=/tmp/claude-1000 python tools/isa-test-gen.py --out-dir /tmp/claude-1000/cascade-test
for f in /tmp/claude-1000/cascade-test/batch_*_producer.s /tmp/claude-1000/cascade-test/batch_*_consumer.s; do
    $LLVM_MC --triple=aie2 --filetype=obj -o /dev/null "$f" 2>&1 && echo "OK: $(basename $f)" || echo "FAIL: $(basename $f)"
done
```

Expected: All 6 files (3 producer + 3 consumer) assemble successfully.

- [ ] **Step 7: Commit**

```bash
git add tools/isa-test-gen.py tools/test_isa_test_gen.py
git commit -m "feat(isa-harness): integrate cascade pairs into generate_all (573/606)"
```

---

### Task 4: Extend isa-multi-tile-gen.py for cascade pairs

**Files:**
- Modify: `tools/isa-multi-tile-gen.py`

This task extends the MLIR generator to handle `cascade_pair` batches.
When a batch has `source_type: "cascade_pair"`, the generator produces
two tiles in the same column with `aie.cascade_flow` and two-hop
objectfifos through the memtile.

- [ ] **Step 1: Write tests for cascade MLIR generation**

Create a test section at the bottom of `tools/isa-multi-tile-gen.py`
or in a new file.  Since the existing file has no tests, add a small
self-test block:

Actually, add tests inline to `tools/test_isa_test_gen.py` since they
share infrastructure:

```python
class TestMultiTileCascade:
    """Tests for cascade_pair MLIR generation in isa-multi-tile-gen."""

    @pytest.fixture
    def multi(self):
        """Import isa-multi-tile-gen.py as a module."""
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("multi", "tools/isa-multi-tile-gen.py")
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _make_cascade_batch(self, idx=0):
        return {
            "batch_index": idx,
            "source_type": "cascade_pair",
            "producer_filename": f"batch_{idx:03d}_producer.s",
            "consumer_filename": f"batch_{idx:03d}_consumer.s",
            "producer_in_size": 64,
            "producer_out_size": 8,
            "consumer_in_size": 0,
            "consumer_out_size": 64,
            "instruction": "VMOV_mv_scd",
            "slot": "lda",
            "in_size": 64,
            "out_size": 72,
            "test_count": 1,
        }

    def _make_normal_batch(self, idx=1):
        return {
            "batch_index": idx,
            "filename": f"batch_{idx:03d}.s",
            "source_type": "assembly",
            "in_size": 128,
            "out_size": 256,
            "test_count": 4,
        }

    def test_cascade_mlir_has_cascade_flow(self, multi):
        mlir = multi.generate_phase_mlir([self._make_cascade_batch()], 0)
        assert "aie.cascade_flow" in mlir

    def test_cascade_mlir_has_two_compute_tiles(self, multi):
        mlir = multi.generate_phase_mlir([self._make_cascade_batch()], 0)
        assert "aie.tile(0, 2)" in mlir  # consumer
        assert "aie.tile(0, 3)" in mlir  # producer

    def test_cascade_mlir_has_memtile(self, multi):
        mlir = multi.generate_phase_mlir([self._make_cascade_batch()], 0)
        assert "aie.tile(0, 1)" in mlir  # memtile

    def test_cascade_mlir_has_objectfifo_link(self, multi):
        mlir = multi.generate_phase_mlir([self._make_cascade_batch()], 0)
        assert "objectfifo.link" in mlir

    def test_cascade_mixed_with_normal(self, multi):
        batches = [self._make_cascade_batch(0), self._make_normal_batch(1)]
        mlir = multi.generate_phase_mlir(batches, 0)
        # Should have cascade flow for col 0 and normal tile for col 1
        assert "aie.cascade_flow" in mlir
        assert "aie.tile(1, 2)" in mlir  # normal batch col 1
        assert "aie.tile(0, 3)" in mlir  # cascade producer col 0

    def test_cascade_mlir_has_dual_link_with(self, multi):
        mlir = multi.generate_phase_mlir([self._make_cascade_batch()], 0)
        assert "producer.o" in mlir
        assert "consumer.o" in mlir

    def test_cascade_consumer_one_arg(self, multi):
        """Consumer function should have exactly one memref arg (output)."""
        mlir = multi.generate_phase_mlir([self._make_cascade_batch()], 0)
        # Find the consumer function declaration
        for line in mlir.split("\n"):
            if "test_kernel_0_cons" in line and "func.func" in line:
                # Should have one memref arg, not two
                assert line.count("memref") == 1, f"Consumer should have 1 arg: {line}"
                break
        else:
            assert False, "Consumer function declaration not found"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tools/test_isa_test_gen.py -k "TestMultiTileCascade" -q`
Expected: FAILED (cascade_flow not in mlir)

- [ ] **Step 3: Implement cascade support in generate_phase_mlir**

Modify `generate_phase_mlir()` in `tools/isa-multi-tile-gen.py`.  The
function currently iterates batches assuming one tile per column.  Add
cascade handling by checking `source_type`:

The key changes:

**3a.** After shim tile declarations, add memtile declarations for
cascade columns and row-3 tile declarations:

```python
    # Memtile + row-3 tiles for cascade columns
    for col, batch in enumerate(batches):
        if batch.get("source_type") == "cascade_pair":
            lines.append(f"    %tile_{col}_1 = aie.tile({col}, 1)")
    for col, batch in enumerate(batches):
        if batch.get("source_type") == "cascade_pair":
            lines.append(f"    %tile_{col}_3 = aie.tile({col}, 3)")
```

**3b.** Add cascade_flow declarations:

```python
    # Cascade flow declarations
    for col, batch in enumerate(batches):
        if batch.get("source_type") == "cascade_pair":
            lines.append(f"    aie.cascade_flow(%tile_{col}_3, %tile_{col}_2)")
    lines.append("")
```

**3c.** In the per-tile objectfifo/function/core loop, branch on
source_type:

For `cascade_pair`:
- Producer input objectfifo: two-hop via memtile with objectfifo.link
- Producer output objectfifo: two-hop via memtile with objectfifo.link
- Consumer output objectfifo: direct to shim (row 2 -> row 0)
- Two function declarations: `test_kernel_N_prod` (2 memref args) and
  `test_kernel_N_cons` (1 memref arg)
- Two core blocks: producer acquires prod_in + prod_out, consumer
  acquires cons_out only

For `assembly` (existing behavior): no changes needed.

**3d.** In the runtime sequence, handle cascade pair objectfifos:
- Output DMAs for producer out (shim-side objectfifo) and consumer out
- Input DMA for producer in (shim-side objectfifo)
- dma_wait for all outputs

The changes are additions interspersed with existing code.  Here is the
structural skeleton for each section of `generate_phase_mlir()`:

**Tile declarations** -- after existing shim + row-2 loops, add:
```python
    # Memtile and row-3 tiles for cascade columns
    cascade_cols = [col for col, b in enumerate(batches)
                    if b.get("source_type") == "cascade_pair"]
    for col in cascade_cols:
        lines.append(f"    %tile_{col}_1 = aie.tile({col}, 1)")
    for col in cascade_cols:
        lines.append(f"    %tile_{col}_3 = aie.tile({col}, 3)")
    # Cascade flow declarations
    for col in cascade_cols:
        lines.append(f"    aie.cascade_flow(%tile_{col}_3, %tile_{col}_2)")
    lines.append("")
```

**Objectfifos and functions** -- in the existing per-column loop,
branch on `source_type`:
```python
    for col, batch in enumerate(batches):
        if batch.get("source_type") == "cascade_pair":
            prod_in_elems = max(1, batch["producer_in_size"] // 4)
            prod_out_elems = max(1, batch["producer_out_size"] // 4)
            cons_out_elems = max(1, batch["consumer_out_size"] // 4)
            prod_o = batch["producer_filename"].replace(".s", ".o")
            cons_o = batch["consumer_filename"].replace(".s", ".o")

            # Producer input: two-hop via memtile
            lines.append(
                f"    aie.objectfifo @of_prod_in_{col}_0(%tile_{col}_0, "
                f"{{%tile_{col}_1}}, 2 : i32) "
                f": !aie.objectfifo<memref<{prod_in_elems}xi32>>")
            lines.append(
                f"    aie.objectfifo @of_prod_in_{col}_1(%tile_{col}_1, "
                f"{{%tile_{col}_3}}, 2 : i32) "
                f": !aie.objectfifo<memref<{prod_in_elems}xi32>>")
            lines.append(
                f"    aie.objectfifo.link [@of_prod_in_{col}_0] -> "
                f"[@of_prod_in_{col}_1] ([] [])")

            # Producer output: two-hop via memtile
            lines.append(
                f"    aie.objectfifo @of_prod_out_{col}_0(%tile_{col}_3, "
                f"{{%tile_{col}_1}}, 2 : i32) "
                f": !aie.objectfifo<memref<{prod_out_elems}xi32>>")
            lines.append(
                f"    aie.objectfifo @of_prod_out_{col}_1(%tile_{col}_1, "
                f"{{%tile_{col}_0}}, 2 : i32) "
                f": !aie.objectfifo<memref<{prod_out_elems}xi32>>")
            lines.append(
                f"    aie.objectfifo.link [@of_prod_out_{col}_0] -> "
                f"[@of_prod_out_{col}_1] ([] [])")

            # Consumer output: direct to shim
            lines.append(
                f"    aie.objectfifo @of_cons_out_{col}(%tile_{col}_2, "
                f"{{%tile_{col}_0}}, 2 : i32) "
                f": !aie.objectfifo<memref<{cons_out_elems}xi32>>")
            lines.append("")

            # Function declarations
            lines.append(
                f'    func.func private @test_kernel_{col}_prod'
                f"(memref<{prod_in_elems}xi32>, memref<{prod_out_elems}xi32>) "
                f'attributes {{link_with = "{prod_o}"}}')
            lines.append(
                f'    func.func private @test_kernel_{col}_cons'
                f"(memref<{cons_out_elems}xi32>) "
                f'attributes {{link_with = "{cons_o}"}}')
            lines.append("")
        else:
            # ... existing normal batch objectfifo + function code ...
```

**Core blocks** -- in the existing per-column core loop:
```python
    for col, batch in enumerate(batches):
        if batch.get("source_type") == "cascade_pair":
            prod_in_elems = max(1, batch["producer_in_size"] // 4)
            prod_out_elems = max(1, batch["producer_out_size"] // 4)
            cons_out_elems = max(1, batch["consumer_out_size"] // 4)

            # Producer core (row 3)
            lines.append(f"    aie.core(%tile_{col}_3) {{")
            lines.append(
                f"      %pin = aie.objectfifo.acquire @of_prod_in_{col}_1"
                f"(Consume, 1) : !aie.objectfifosubview<memref<{prod_in_elems}xi32>>")
            lines.append(
                f"      %pin_e = aie.objectfifo.subview.access %pin[0] "
                f": !aie.objectfifosubview<memref<{prod_in_elems}xi32>> "
                f"-> memref<{prod_in_elems}xi32>")
            lines.append(
                f"      %pout = aie.objectfifo.acquire @of_prod_out_{col}_0"
                f"(Produce, 1) : !aie.objectfifosubview<memref<{prod_out_elems}xi32>>")
            lines.append(
                f"      %pout_e = aie.objectfifo.subview.access %pout[0] "
                f": !aie.objectfifosubview<memref<{prod_out_elems}xi32>> "
                f"-> memref<{prod_out_elems}xi32>")
            lines.append(
                f"      func.call @test_kernel_{col}_prod(%pin_e, %pout_e) "
                f": (memref<{prod_in_elems}xi32>, memref<{prod_out_elems}xi32>) -> ()")
            lines.append(f"      aie.objectfifo.release @of_prod_in_{col}_1(Consume, 1)")
            lines.append(f"      aie.objectfifo.release @of_prod_out_{col}_0(Produce, 1)")
            lines.append("      aie.end")
            lines.append("    }")

            # Consumer core (row 2)
            lines.append(f"    aie.core(%tile_{col}_2) {{")
            lines.append(
                f"      %cout = aie.objectfifo.acquire @of_cons_out_{col}"
                f"(Produce, 1) : !aie.objectfifosubview<memref<{cons_out_elems}xi32>>")
            lines.append(
                f"      %cout_e = aie.objectfifo.subview.access %cout[0] "
                f": !aie.objectfifosubview<memref<{cons_out_elems}xi32>> "
                f"-> memref<{cons_out_elems}xi32>")
            lines.append(
                f"      func.call @test_kernel_{col}_cons(%cout_e) "
                f": (memref<{cons_out_elems}xi32>) -> ()")
            lines.append(f"      aie.objectfifo.release @of_cons_out_{col}(Produce, 1)")
            lines.append("      aie.end")
            lines.append("    }")
            lines.append("")
        else:
            # ... existing normal batch core block ...
```

**Runtime sequence** -- DMA operations for cascade columns.  Use the
batch dict's `producer_in_size`, `producer_out_size`, `consumer_out_size`
to compute sub-offsets within the phase buffer:

```python
    # For cascade pairs, decompose the combined in_size/out_size into
    # sub-regions for producer-in, producer-out, and consumer-out.
    # The phase buffer layout packs them as:
    #   input:  [prod_in]
    #   output: [prod_out | cons_out]

    for col, batch in enumerate(batches):
        if batch.get("source_type") == "cascade_pair":
            prod_in_elems = max(1, batch["producer_in_size"] // 4)
            prod_out_elems = max(1, batch["producer_out_size"] // 4)
            cons_out_elems = max(1, batch["consumer_out_size"] // 4)

            in_off = layout["in_offsets_elems"][col]
            out_off = layout["out_offsets_elems"][col]
            # Producer output starts at out_off, consumer output follows
            prod_out_off = out_off
            cons_out_off = out_off + prod_out_elems

            # Output DMAs (issue_token)
            # Producer output via of_prod_out_N_1 (memtile-to-shim leg)
            lines.append(
                f"      aiex.npu.dma_memcpy_nd("
                f"%out[%c0, %c0, %c0, <prod_out_off_const>]"
                f"[%c1, %c1, %c1, <prod_out_len_const>]"
                f"[%c0, %c0, %c0, %c1]) "
                f"{{metadata = @of_prod_out_{col}_1, id = {dma_id} : i64, "
                f"issue_token = true}} : memref<{total_out}xi32>")
            # Consumer output via of_cons_out_N
            lines.append(
                f"      aiex.npu.dma_memcpy_nd("
                f"%out[%c0, %c0, %c0, <cons_out_off_const>]"
                f"[%c1, %c1, %c1, <cons_out_len_const>]"
                f"[%c0, %c0, %c0, %c1]) "
                f"{{metadata = @of_cons_out_{col}, id = {dma_id+1} : i64, "
                f"issue_token = true}} : memref<{total_out}xi32>")

            # Input DMA
            # Producer input via of_prod_in_N_0 (shim-to-memtile leg)
            lines.append(
                f"      aiex.npu.dma_memcpy_nd("
                f"%in[%c0, %c0, %c0, <prod_in_off_const>]"
                f"[%c1, %c1, %c1, <prod_in_len_const>]"
                f"[%c0, %c0, %c0, %c1]) "
                f"{{metadata = @of_prod_in_{col}_0, id = {dma_id+2} : i64}} "
                f": memref<{total_in}xi32>")

            # dma_wait for outputs
            lines.append(f"      aiex.npu.dma_wait {{symbol = @of_prod_out_{col}_1}}")
            lines.append(f"      aiex.npu.dma_wait {{symbol = @of_cons_out_{col}}}")
```

DMA ID assignment: cascade pairs consume 3 IDs (prod_out, cons_out,
prod_in).  Track `dma_id` as a counter across all columns.  Normal
batches consume 2 IDs (in + out) as they do currently.

The `<..._const>` placeholders above represent `arith.constant` SSA
values that must be declared in the constants block of the runtime
sequence, following the existing pattern for `%c_in_off_N` / `%c_in_len_N`.
For cascade pairs, declare: `%c_prod_in_off_N`, `%c_prod_in_len_N`,
`%c_prod_out_off_N`, `%c_prod_out_len_N`, `%c_cons_out_off_N`,
`%c_cons_out_len_N`.

Reference: `mlir-aie/test/npu-xrt/cascade_flows/aie.mlir` for the
exact `dma_memcpy_nd` and `dma_wait` syntax with objectfifo.link'd
objectfifos.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tools/test_isa_test_gen.py -k "TestMultiTileCascade" -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add tools/isa-multi-tile-gen.py tools/test_isa_test_gen.py
git commit -m "feat(isa-multi-tile): cascade_pair MLIR generation with cascade_flow"
```

---

### Task 5: Symbol renaming for cascade pairs

**Files:**
- Modify: `tools/isa-multi-tile-gen.py` (prepare_phase_objects, ~line 109)

- [ ] **Step 1: Write failing test**

Add to `TestMultiTileCascade` in `tools/test_isa_test_gen.py`:

```python
    def test_prepare_phase_objects_cascade(self, tmp_path):
        """Symbol renaming should produce _prod and _cons suffixes."""
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("multi", "tools/isa-multi-tile-gen.py")
        multi = module_from_spec(spec)
        spec.loader.exec_module(multi)

        # Create fake .o files with a test_kernel symbol
        obj_dir = str(tmp_path / "obj")
        os.makedirs(obj_dir)
        phase_dir = str(tmp_path / "phase")

        batch = self._make_cascade_batch(0)
        # Create minimal ELF-like files (just need to exist for the test)
        # This test requires actual llvm-objcopy, so skip if not available
        objcopy = multi.LLVM_OBJCOPY
        if not os.path.isfile(objcopy):
            pytest.skip("llvm-objcopy not found")

        # We need real .o files -- assemble minimal programs
        llvm_mc = os.path.expanduser(
            "~/npu-work/llvm-aie/build/bin/llvm-mc"
        )
        if not os.path.isfile(llvm_mc):
            pytest.skip("llvm-mc not found")

        for suffix in ("producer", "consumer"):
            src = os.path.join(obj_dir, f"batch_000_{suffix}.s")
            with open(src, "w") as f:
                f.write(".text\n.globl test_kernel\ntest_kernel:\nnop\n")
            obj = os.path.join(obj_dir, f"batch_000_{suffix}.o")
            subprocess.run(
                [llvm_mc, "--triple=aie2", "--filetype=obj", "-o", obj, src],
                check=True,
            )

        renamed = multi.prepare_phase_objects([batch], phase_dir, obj_dir)
        assert len(renamed) == 2
        assert "producer" in renamed[0]
        assert "consumer" in renamed[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tools/test_isa_test_gen.py -k "test_prepare_phase_objects_cascade" -q`
Expected: FAILED

- [ ] **Step 3: Implement cascade symbol renaming**

Modify `prepare_phase_objects()` in `tools/isa-multi-tile-gen.py` to
handle cascade pairs.  When `batch.get("source_type") == "cascade_pair"`,
process two .o files instead of one:

```python
def prepare_phase_objects(batches, phase_dir, obj_dir):
    os.makedirs(phase_dir, exist_ok=True)
    result = []
    for col, batch in enumerate(batches):
        if batch.get("source_type") == "cascade_pair":
            # Two .o files per cascade pair
            for suffix, sym_suffix in [("producer", "prod"), ("consumer", "cons")]:
                src_name = batch.get(f"{suffix}_filename", "").replace(".s", ".o")
                src = os.path.join(obj_dir, src_name)
                dst = os.path.join(phase_dir, src_name)
                shutil.copy2(src, dst)
                subprocess.run(
                    [LLVM_OBJCOPY, "--redefine-sym",
                     f"test_kernel=test_kernel_{col}_{sym_suffix}", dst],
                    check=True,
                )
                result.append(src_name)
        else:
            # Existing single-tile logic
            src = os.path.join(obj_dir, f"batch_{batch['batch_index']}.o")
            o_name = _obj_filename(batch)
            dst = os.path.join(phase_dir, o_name)
            shutil.copy2(src, dst)
            subprocess.run(
                [LLVM_OBJCOPY, "--redefine-sym",
                 f"test_kernel=test_kernel_{col}", dst],
                check=True,
            )
            result.append(o_name)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tools/test_isa_test_gen.py -k "test_prepare_phase_objects_cascade" -q`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add tools/isa-multi-tile-gen.py tools/test_isa_test_gen.py
git commit -m "feat(isa-multi-tile): cascade pair symbol renaming (_prod/_cons)"
```

---

### Task 6: End-to-end verification

**Files:** None modified (verification only)

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest tools/test_isa_test_gen.py -q
```

Expected: all passed, no regressions

- [ ] **Step 2: Check coverage numbers**

```bash
python tools/isa-test-gen.py --summary 2>&1 | grep -E "^(Testable|Skipped|Total)"
```

Expected: Testable: 573, Skipped: 33, Total: 606

- [ ] **Step 3: Verify all assembly compiles**

```bash
LLVM_MC=$HOME/npu-work/llvm-aie/build/bin/llvm-mc
TMPDIR=/tmp/claude-1000 python tools/isa-test-gen.py --out-dir /tmp/claude-1000/cascade-e2e
fail=0
for f in /tmp/claude-1000/cascade-e2e/batch_*.s; do
    $LLVM_MC --triple=aie2 --filetype=obj -o /dev/null "$f" 2>/dev/null || fail=$((fail+1))
done
echo "Assembly failures: $fail (expect 9 pre-existing)"
```

- [ ] **Step 4: Verify skip reasons are clean**

```bash
python tools/isa-test-gen.py --summary 2>&1 | sed -n '/Skip reasons:/,/Testable/p'
```

Expected: No "cascade read" entry in skip reasons.

- [ ] **Step 5: Final commit (if any fixups needed)**

```bash
git add -u
git commit -m "fix(isa-harness): cascade read e2e verification fixups"
```
