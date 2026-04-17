# Multi-Tile ISA Validation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-tile ISA validation harness with a multi-tile architecture that tests all instructions (including conversion instructions) across all NPU tiles in parallel, with phased program reload.

**Architecture:** Batch .o files (from llvm-mc for assembly, llc for LLVM IR) are assigned to NPU tiles (up to 4 per phase, one per column). A Python MLIR generator produces one aie.mlir per phase, with per-tile objectfifos, link_with references, and DMA configuration. aiecc.py compiles each phase to a single xclbin. The runner iterates phases sequentially, sharing one pseudorandom input buffer across all phases.

**Tech Stack:** Python 3.13, MLIR-AIE (aiecc.py), Peano llc/llvm-mc, XRT, Bash

---

## File Structure

| File | Responsibility |
|------|---------------|
| `tools/isa-test-gen.py` | **Modify**: Add `ConversionStrategy` that generates LLVM IR (.ll) for fused load+convert and convert+store instructions |
| `tools/isa-multi-tile-gen.py` | **Create**: Takes manifest.json, assigns batches to tiles/phases, generates one aie.mlir per phase |
| `tools/test_isa_multi_tile_gen.py` | **Create**: Unit tests for multi-tile MLIR generation |
| `tools/test_isa_test_gen.py` | **Modify**: Add tests for ConversionStrategy |
| `tools/instr-test-gen.py` | **Modify**: Add `generate_multi_tile_mlir()` for N-tile MLIR template |
| `scripts/isa-test.sh` | **Modify**: Add llc compilation path for .ll batches; add multi-tile phase execution mode |

## Constraints and Reference

**NPU1 topology:**
- 4 columns (0-3), compute rows 2-5, shim row 0, mem tile row 1
- 16 compute tiles total, but **start with 4** (one per column, row 2) -- known-working pattern from `ctrl_packet_reconfig_4x1_cores`
- Each shim: 2 DMA channels (MM2S + S2MM), 16 BDs
- Compute tile: 16KB program memory (1024 bundles), 64KB local memory

**Key reference files:**
- Current MLIR template: `tools/instr-test-gen.py:314` (`generate_aie_mlir()`)
- Current runner: `scripts/isa-test.sh`
- Multi-tile MLIR example: `mlir-aie/test/npu-xrt/ctrl_packet_reconfig_4x1_cores/aie.mlir`
- External kernel linking: `mlir-aie/test/npu-xrt/cascade_flows/aie.mlir` (uses `link_with`)
- Peano llc: `/home/triple/npu-work/llvm-aie/install/bin/llc` (has aie2 target)
- Conversion intrinsic patterns: `llvm-aie/llvm/lib/Target/AIE/AIE2InstrPatterns.td:496-548`

**MLIR pattern for per-tile external kernels:**
```mlir
func.func private @test_kernel_0(memref<Nxi32>, memref<Mxi32>)
  attributes {link_with = "batch_000.o"}
func.func private @test_kernel_1(memref<Nxi32>, memref<Mxi32>)
  attributes {link_with = "batch_001.o"}

aie.core(%tile_0_2) {
  // acquire objectfifo, call @test_kernel_0, release
}
aie.core(%tile_1_2) {
  // acquire objectfifo, call @test_kernel_1, release
}
```

**Conversion intrinsics (LLVM IR):**
```
UPS:    @llvm.aie2.acc32.v16.I256.ups(<16 x i16>, i32 shift, i32 sign)  -> <8 x i64>
SRS:    @llvm.aie2.I256.v16.acc32.srs(<8 x i64>, i32 shift, i32 sign)  -> <16 x i16>
PACK:   @llvm.aie2.pack_I4_I8 / pack_I8_I16
UNPACK: @llvm.aie2.unpack_I8_I4 / unpack_I16_I8
```
The compiler fuses load+UPS into `vlda.ups.*` and SRS+store into `vst.srs.*` automatically (proven in PoC).

---

## Task 1: ConversionStrategy -- LLVM IR Generation

Add a new strategy to `isa-test-gen.py` that generates LLVM IR (.ll files) instead of assembly (.s files) for the 26 conversion instruction families (148 total instruction definitions).

**Files:**
- Modify: `tools/isa-test-gen.py`
- Test: `tools/test_isa_test_gen.py`

### Intrinsic mapping table

Each conversion operation maps to a specific LLVM intrinsic + type signature:

| Mnemonic pattern | Intrinsic | Input type | Output type | Sign arg |
|-----------------|-----------|------------|-------------|----------|
| `vlda.ups.s32.s16` | `acc32.v16.I256.ups` | `<16 x i16>` | `<8 x i64>` | 1 (signed) |
| `vlda.ups.s32.d16` | `acc32.v16.I256.ups` | `<16 x i16>` | `<8 x i64>` | 0 (unsigned) |
| `vlda.ups.s32.s8` | `acc32.v32.I256.ups` | `<32 x i8>` | `<8 x i64>` | 1 |
| `vlda.ups.s32.d8` | `acc32.v32.I256.ups` | `<32 x i8>` | `<8 x i64>` | 0 |
| `vlda.ups.s64.s32` | `acc64.v8.I256.ups` | `<8 x i32>` | `<8 x i64>` | 1 |
| `vlda.ups.s64.d32` | `acc64.v8.I256.ups` | `<8 x i32>` | `<8 x i64>` | 0 |
| `vlda.ups.s64.s16` | `acc64.v16.I256.ups` | `<16 x i16>` | `<8 x i64>` | 1 |
| `vlda.ups.s64.d16` | `acc64.v16.I256.ups` | `<16 x i16>` | `<8 x i64>` | 0 |
| `vst.srs.s16.s32` | `I256.v16.acc32.srs` | `<8 x i64>` | `<16 x i16>` | 1 |
| `vst.srs.d16.s32` | `I256.v16.acc32.srs` | `<8 x i64>` | `<16 x i16>` | 0 |
| `vst.srs.s8.s32` | `I256.v32.acc32.srs` | `<8 x i64>` | `<32 x i8>` | 1 |
| `vst.srs.d8.s32` | `I256.v32.acc32.srs` | `<8 x i64>` | `<32 x i8>` | 0 |
| `vst.srs.s32.s64` | `I256.v8.acc64.srs` | `<8 x i64>` | `<8 x i32>` | 1 |
| `vst.srs.d32.s64` | `I256.v8.acc64.srs` | `<8 x i64>` | `<8 x i32>` | 0 |
| `vst.srs.s16.s64` | `I256.v16.acc64.srs` | `<8 x i64>` | `<16 x i16>` | 1 |
| `vst.srs.d16.s64` | `I256.v16.acc64.srs` | `<8 x i64>` | `<16 x i16>` | 0 |
| `vst.pack.s4.s8` | `pack_I4_I8` | `<32 x i8>` | `<32 x i8>` (half) | -- |
| `vst.pack.d4.d8` | `pack_I4_I8` | `<32 x i8>` | `<32 x i8>` (half) | -- |
| `vst.pack.s8.s16` | `pack_I8_I16` | `<16 x i16>` | `<16 x i16>` (half) | -- |
| `vst.pack.d8.d16` | `pack_I8_I16` | `<16 x i16>` | `<16 x i16>` (half) | -- |
| `vldb.unpack.s8.s4` | `unpack_I8_I4` | `<32 x i8>` (half) | `<32 x i8>` | -- |
| `vldb.unpack.d8.d4` | `unpack_I8_I4` | `<32 x i8>` (half) | `<32 x i8>` | -- |
| `vldb.unpack.s16.s8` | `unpack_I16_I8` | `<32 x i8>` | `<16 x i16>` | -- |
| `vldb.unpack.d16.d8` | `unpack_I16_I8` | `<32 x i8>` | `<16 x i16>` | -- |

**Note:** `vlda.conv.fp32.bf16` and `vst.conv.bf16.fp32` are excluded -- the AIE2 intrinsics for bf16/fp32 conversion (`v16accfloat_to_v16bf16`, `v16bf16_to_v16accfloat`) have different signatures than the fused load/store pattern and need separate handling. Total: **24 conversion operations** (not 26).

### Step-by-step

- [ ] **Step 1: Write test for conversion intrinsic mapping**

In `tools/test_isa_test_gen.py`, add:

```python
class TestConversionIntrinsicMap(unittest.TestCase):
    """Test the CONVERSION_INTRINSICS mapping table."""

    def test_all_24_base_operations_covered(self):
        """Every unique mnemonic base maps to an intrinsic."""
        from isa_test_gen import CONVERSION_INTRINSICS
        # 8 UPS + 8 SRS + 4 PACK + 4 UNPACK = 24
        # (bf16/fp32 conv excluded -- different intrinsic signatures)
        self.assertEqual(len(CONVERSION_INTRINSICS), 24)

    def test_ups_signed_unsigned_differ_only_in_sign_arg(self):
        from isa_test_gen import CONVERSION_INTRINSICS
        s = CONVERSION_INTRINSICS["vlda.ups.s32.s16"]
        d = CONVERSION_INTRINSICS["vlda.ups.s32.d16"]
        self.assertEqual(s["intrinsic"], d["intrinsic"])
        self.assertEqual(s["sign"], 1)
        self.assertEqual(d["sign"], 0)

    def test_ups_entry_has_required_fields(self):
        from isa_test_gen import CONVERSION_INTRINSICS
        entry = CONVERSION_INTRINSICS["vlda.ups.s32.s16"]
        for field in ("intrinsic", "in_type", "out_type", "sign",
                      "in_bytes", "out_bytes"):
            self.assertIn(field, entry, f"Missing field: {field}")
```

- [ ] **Step 2: Run test, verify it fails**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_isa_test_gen.py::TestConversionIntrinsicMap -v`
Expected: ImportError (CONVERSION_INTRINSICS not defined)

- [ ] **Step 3: Implement CONVERSION_INTRINSICS table**

Add to `tools/isa-test-gen.py` near the top constants section:

```python
CONVERSION_INTRINSICS: dict[str, dict] = {
    # UPS: load + upshift (vector -> accumulator)
    "vlda.ups.s32.s16": {"intrinsic": "acc32.v16.I256.ups", "in_type": "<16 x i16>", "out_type": "<8 x i64>", "sign": 1, "in_bytes": 32, "out_bytes": 64},
    "vlda.ups.s32.d16": {"intrinsic": "acc32.v16.I256.ups", "in_type": "<16 x i16>", "out_type": "<8 x i64>", "sign": 0, "in_bytes": 32, "out_bytes": 64},
    # ... (all 26 entries from the table above)
}
```

- [ ] **Step 4: Run test, verify it passes**

- [ ] **Step 5: Write test for .ll generation**

```python
class TestConversionLLGeneration(unittest.TestCase):
    def test_ups_generates_valid_ll(self):
        from isa_test_gen import generate_conversion_ll
        ll = generate_conversion_ll([
            {"mnemonic": "vlda.ups.s32.s16", "in_offset": 0, "out_offset": 0},
        ])
        self.assertIn("@llvm.aie2.acc32.v16.I256.ups", ll)
        self.assertIn("target triple", ll)
        self.assertIn("define void @test_kernel", ll)
        self.assertIn("load volatile", ll)
        self.assertIn("store volatile", ll)

    def test_srs_generates_valid_ll(self):
        from isa_test_gen import generate_conversion_ll
        ll = generate_conversion_ll([
            {"mnemonic": "vst.srs.s16.s32", "in_offset": 0, "out_offset": 0},
        ])
        # SRS needs a UPS to load data into accumulator first
        self.assertIn("@llvm.aie2.acc32.v16.I256.ups", ll)
        self.assertIn("@llvm.aie2.I256.v16.acc32.srs", ll)

    def test_multiple_conversions_sequential(self):
        from isa_test_gen import generate_conversion_ll
        ll = generate_conversion_ll([
            {"mnemonic": "vlda.ups.s32.s16", "in_offset": 0, "out_offset": 0},
            {"mnemonic": "vlda.ups.s32.d16", "in_offset": 32, "out_offset": 32},
        ])
        # Both intrinsics should appear
        self.assertEqual(ll.count("@llvm.aie2.acc32.v16.I256.ups"), 2)
```

- [ ] **Step 6: Run test, verify it fails**

- [ ] **Step 7: Implement generate_conversion_ll()**

This function generates LLVM IR for a batch of conversion test points:

```python
def generate_conversion_ll(test_points: list[dict]) -> str:
    """Generate LLVM IR that exercises conversion instructions.

    Each test point loads from in_ptr + offset, calls the appropriate
    intrinsic, and stores to out_ptr + offset.  The compiler fuses
    load+UPS into vlda.ups and SRS+store into vst.srs automatically.

    Args:
        test_points: List of dicts with 'mnemonic', 'in_offset', 'out_offset'.

    Returns:
        Complete .ll file as string.
    """
    lines = ['target triple = "aie2"', '']

    # Collect all needed intrinsic declarations
    declared = set()
    body_lines = []

    for i, tp in enumerate(test_points):
        entry = CONVERSION_INTRINSICS[tp["mnemonic"]]
        in_off = tp["in_offset"]
        out_off = tp["out_offset"]

        # Generate GEP + load + intrinsic + store for this test point
        # ... (see implementation details in step)
        pass

    # Build function
    lines.append('define void @test_kernel(ptr %in, ptr %out) {')
    lines.append('entry:')
    lines.extend(body_lines)
    lines.append('  ret void')
    lines.append('}')
    lines.append('')

    # Declare intrinsics
    for decl in sorted(declared):
        lines.append(decl)

    return '\n'.join(lines)
```

The key IR pattern for each UPS test point:
```llvm
  ; UPS test: vlda.ups.s32.s16
  %in_N = getelementptr i8, ptr %in, i64 <in_offset>
  %out_N = getelementptr i8, ptr %out, i64 <out_offset>
  %v_N = load volatile <16 x i16>, ptr %in_N, align 32
  %acc_N = call <8 x i64> @llvm.aie2.acc32.v16.I256.ups(<16 x i16> %v_N, i32 0, i32 1)
  %r_N = call <16 x i16> @llvm.aie2.I256.v16.acc32.srs(<8 x i64> %acc_N, i32 0, i32 1)
  store volatile <16 x i16> %r_N, ptr %out_N, align 32
```

For SRS test points, the pattern is similar but the instruction under test is the SRS (the UPS is just setup).

- [ ] **Step 8: Run test, verify it passes**

- [ ] **Step 9: Wire ConversionStrategy into classify_with_strategies()**

Modify `classify_with_strategies()` to recognize conversion instructions and assign `ConversionStrategy` instead of skipping them.

```python
class ConversionStrategy(TestStrategy):
    """Strategy for conversion load/store instructions (vlda.ups, vst.srs, etc.)."""

    @staticmethod
    def can_handle(instr: dict) -> tuple[bool, str]:
        mnemonic = instr["mnemonic"]
        # Strip 2D/3D addressing -- same conversion semantics
        base = re.sub(r'\.(2d|3d)\.', '.', mnemonic)
        if base in CONVERSION_INTRINSICS:
            return (True, "")
        return (False, "not a conversion instruction")
```

- [ ] **Step 10: Integrate conversion batches into generate_all()**

Conversion test points get their own batches with `source_type: "llvm_ir"` in the manifest. The batch file is `.ll` instead of `.s`.

All 26 unique conversions fit in a single batch (small code size).

- [ ] **Step 11: Run full test suite, verify no regressions**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_isa_test_gen.py -v`

- [ ] **Step 12: Commit**

```bash
git add tools/isa-test-gen.py tools/test_isa_test_gen.py
git commit -m "feat(isa-harness): add ConversionStrategy for fused load/store+convert instructions

Generates LLVM IR (.ll files) that call Peano intrinsics (ups, srs, pack,
unpack, conv).  The compiler automatically fuses load+intrinsic into
vlda.ups/vldb.unpack and intrinsic+store into vst.srs/vst.pack.

Covers 26 unique conversion operations (148 instruction definitions)
that llvm-mc cannot assemble directly."
```

---

## Task 2: llc Compilation Pipeline

Add llc-based compilation for .ll batches alongside the existing llvm-mc path.

**Files:**
- Modify: `scripts/isa-test.sh`

- [ ] **Step 1: Add llc path detection**

In `scripts/isa-test.sh`, after `LLVM_MC` definition:

```bash
PEANO_LLC="${HOME}/npu-work/llvm-aie/install/bin/llc"
```

- [ ] **Step 2: Modify assemble_one() to handle .ll files**

```bash
assemble_one() {
    local batch_idx="$1"
    local filename="$2"
    local s_path="${OUT_DIR}/${filename}"
    local o_path="${OUT_DIR}/batch_${batch_idx}.o"

    # Skip if already assembled (unless --compile).
    if ! $FORCE_COMPILE && [[ -f "$o_path" ]] && [[ "$o_path" -nt "$s_path" ]]; then
        return 0
    fi

    if [[ "$filename" == *.ll ]]; then
        # LLVM IR: compile with llc
        # --issue-limit=1 constrains the scheduler to one instruction per
        # cycle, preventing VLIW packing that would interfere with the
        # test harness's single-instruction-at-a-time validation model.
        nice -n 19 "$PEANO_LLC" -mtriple=aie2 --issue-limit=1 \
            -filetype=obj -o "$o_path" "$s_path" \
            2>"${OUT_DIR}/batch_${batch_idx}.llc.log" && \
            echo "  LLC OK: ${filename}" || \
            echo "  LLC FAIL: ${filename} (see batch_${batch_idx}.llc.log)"
    else
        # Assembly: assemble with llvm-mc
        nice -n 19 "$LLVM_MC" --triple=aie2 --filetype=obj \
            -o "$o_path" "$s_path" \
            2>"${OUT_DIR}/batch_${batch_idx}.mc.log" && \
            echo "  ASM OK: ${filename}" || \
            echo "  ASM FAIL: ${filename} (see batch_${batch_idx}.mc.log)"
    fi
}
```

- [ ] **Step 3: Verify .ll compilation works end-to-end**

Run: `scripts/isa-test.sh --generate-only` then manually compile a .ll batch with llc.

- [ ] **Step 4: Commit**

```bash
git add scripts/isa-test.sh
git commit -m "feat(isa-harness): add llc compilation path for LLVM IR batches"
```

---

## Task 3: Multi-Tile MLIR Generator

Create a Python module that takes the manifest.json (with batch assignments) and generates multi-tile MLIR files -- one per phase, each phase using up to 4 tiles (one per column).

**Files:**
- Create: `tools/isa-multi-tile-gen.py`
- Create: `tools/test_isa_multi_tile_gen.py`

### Design

Each phase gets an MLIR file with this structure:
```
aie.device(npu1) {
  // 4 shim tiles + 4 compute tiles
  // Per-tile: buffer, locks, objectfifo, core with link_with
  // Per-column: flow from shim to compute and back
  // runtime_sequence: DMA all inputs, DMA all outputs, wait
}
```

**Input buffer layout**: One contiguous DDR buffer. Each tile reads from a different offset. The offset is computed from cumulative batch input sizes.

**Output buffer layout**: Same pattern. Each tile writes to a different offset in one contiguous DDR output buffer.

### Step-by-step

- [ ] **Step 1: Write test for phase assignment**

```python
class TestPhaseAssignment(unittest.TestCase):
    def test_4_batches_fit_in_one_phase(self):
        from isa_multi_tile_gen import assign_phases
        batches = [{"batch_index": i, "in_size": 1024, "out_size": 512}
                   for i in range(4)]
        phases = assign_phases(batches, tiles_per_phase=4)
        self.assertEqual(len(phases), 1)
        self.assertEqual(len(phases[0]), 4)

    def test_8_batches_need_two_phases(self):
        from isa_multi_tile_gen import assign_phases
        batches = [{"batch_index": i, "in_size": 1024, "out_size": 512}
                   for i in range(8)]
        phases = assign_phases(batches, tiles_per_phase=4)
        self.assertEqual(len(phases), 2)

    def test_remainder_gets_partial_phase(self):
        from isa_multi_tile_gen import assign_phases
        batches = [{"batch_index": i, "in_size": 1024, "out_size": 512}
                   for i in range(5)]
        phases = assign_phases(batches, tiles_per_phase=4)
        self.assertEqual(len(phases), 2)
        self.assertEqual(len(phases[1]), 1)  # partial
```

- [ ] **Step 2: Run test, verify it fails**

- [ ] **Step 3: Implement assign_phases()**

```python
def assign_phases(batches: list[dict], tiles_per_phase: int = 4) -> list[list[dict]]:
    """Assign batches to phases, each phase using up to tiles_per_phase tiles."""
    phases = []
    for i in range(0, len(batches), tiles_per_phase):
        phases.append(batches[i:i + tiles_per_phase])
    return phases
```

- [ ] **Step 4: Run test, verify it passes**

- [ ] **Step 5: Write test for multi-tile MLIR generation**

```python
class TestMultiTileMlir(unittest.TestCase):
    def test_single_tile_generates_valid_mlir(self):
        from isa_multi_tile_gen import generate_phase_mlir
        batches = [{"batch_index": 0, "in_size": 128, "out_size": 64,
                     "filename": "batch_000.s"}]
        mlir = generate_phase_mlir(batches, phase_idx=0)
        self.assertIn("aie.device(npu1)", mlir)
        self.assertIn("aie.tile(0, 2)", mlir)
        self.assertIn('link_with = "batch_000.o"', mlir)
        self.assertIn("@test_kernel_0", mlir)

    def test_four_tiles_generates_four_cores(self):
        from isa_multi_tile_gen import generate_phase_mlir
        batches = [{"batch_index": i, "in_size": 128, "out_size": 64,
                     "filename": f"batch_{i:03d}.s"}
                    for i in range(4)]
        mlir = generate_phase_mlir(batches, phase_idx=0)
        for col in range(4):
            self.assertIn(f"aie.tile({col}, 2)", mlir)
            self.assertIn(f"@test_kernel_{col}", mlir)
        # Should NOT have tiles in columns beyond batch count
        self.assertIn("aie.tile(3, 2)", mlir)

    def test_each_tile_has_own_objectfifo(self):
        from isa_multi_tile_gen import generate_phase_mlir
        batches = [{"batch_index": i, "in_size": 128, "out_size": 64,
                     "filename": f"batch_{i:03d}.s"}
                    for i in range(2)]
        mlir = generate_phase_mlir(batches, phase_idx=0)
        self.assertIn("@of_in_0", mlir)
        self.assertIn("@of_in_1", mlir)
        self.assertIn("@of_out_0", mlir)
        self.assertIn("@of_out_1", mlir)
```

- [ ] **Step 6: Run test, verify it fails**

- [ ] **Step 7: Implement generate_phase_mlir()**

This is the core of the multi-tile generator. Each tile gets:
- Its own objectfifo pair (input + output) with correct element count
- Its own `func.func private @test_kernel_N` with `link_with = "batch_NNN.o"`
- Its own `aie.core` block that acquires, calls, releases

The runtime_sequence DMAs all inputs and outputs in parallel, using different offsets into the DDR buffer for each tile.

```python
def generate_phase_mlir(batches: list[dict], phase_idx: int) -> str:
    """Generate MLIR for one phase with up to 4 tiles.

    Args:
        batches: List of batch dicts from manifest.json (up to 4).
        phase_idx: Phase index (for naming).

    Returns:
        Complete MLIR module as string.
    """
    n_tiles = len(batches)
    assert 1 <= n_tiles <= 4

    # Use npu1_{n}col for partial phases (avoids unused column config)
    device = f"npu1_{n_tiles}col" if n_tiles < 4 else "npu1"
    lines = ["module {"]
    lines.append(f"  aie.device({device}) {{")

    # Declare tiles
    for col in range(n_tiles):
        lines.append(f"    %tile_{col}_0 = aie.tile({col}, 0)")
    for col in range(n_tiles):
        lines.append(f"    %tile_{col}_2 = aie.tile({col}, 2)")
    lines.append("")

    # Per-tile: objectfifo + func decl + core
    for col, batch in enumerate(batches):
        in_elems = max(1, batch["in_size"] // 4)
        out_elems = max(1, batch["out_size"] // 4)
        o_file = batch["filename"].replace(".s", ".o").replace(".ll", ".o")

        # Objectfifos
        lines.append(f"    aie.objectfifo @of_in_{col}(%tile_{col}_0, {{%tile_{col}_2}}, 2 : i32) : !aie.objectfifo<memref<{in_elems}xi32>>")
        lines.append(f"    aie.objectfifo @of_out_{col}(%tile_{col}_2, {{%tile_{col}_0}}, 2 : i32) : !aie.objectfifo<memref<{out_elems}xi32>>")
        lines.append("")

        # Function declaration
        lines.append(f'    func.func private @test_kernel_{col}(memref<{in_elems}xi32>, memref<{out_elems}xi32>) attributes {{link_with = "{o_file}"}}')
        lines.append("")

        # Core
        lines.append(f"    aie.core(%tile_{col}_2) {{")
        lines.append(f"      %sub_in = aie.objectfifo.acquire @of_in_{col}(Consume, 1) : !aie.objectfifosubview<memref<{in_elems}xi32>>")
        lines.append(f"      %elem_in = aie.objectfifo.subview.access %sub_in[0] : !aie.objectfifosubview<memref<{in_elems}xi32>> -> memref<{in_elems}xi32>")
        lines.append(f"      %sub_out = aie.objectfifo.acquire @of_out_{col}(Produce, 1) : !aie.objectfifosubview<memref<{out_elems}xi32>>")
        lines.append(f"      %elem_out = aie.objectfifo.subview.access %sub_out[0] : !aie.objectfifosubview<memref<{out_elems}xi32>> -> memref<{out_elems}xi32>")
        lines.append(f"      func.call @test_kernel_{col}(%elem_in, %elem_out) : (memref<{in_elems}xi32>, memref<{out_elems}xi32>) -> ()")
        lines.append(f"      aie.objectfifo.release @of_in_{col}(Consume, 1)")
        lines.append(f"      aie.objectfifo.release @of_out_{col}(Produce, 1)")
        lines.append(f"      aie.end")
        lines.append(f"    }}")
        lines.append("")

    # Runtime sequence -- DMA all tiles
    # Compute total buffer sizes for the combined DDR buffers
    total_in = sum(max(1, b["in_size"] // 4) for b in batches)
    total_out = sum(max(1, b["out_size"] // 4) for b in batches)

    lines.append(f"    aie.runtime_sequence(%in : memref<{total_in}xi32>, %buf : memref<{total_in}xi32>, %out : memref<{total_out}xi32>) {{")
    lines.append(f"      %c0 = arith.constant 0 : i64")
    lines.append(f"      %c1 = arith.constant 1 : i64")

    # DMA outputs first (set up receive before sending).
    # issue_token on each output DMA -- matches ctrl_packet_reconfig_4x1_cores pattern.
    out_offset = 0
    for col, batch in enumerate(batches):
        out_elems = max(1, batch["out_size"] // 4)
        lines.append(f"      %c_out_off_{col} = arith.constant {out_offset} : i64")
        lines.append(f"      %c_out_len_{col} = arith.constant {out_elems} : i64")
        lines.append(f"      aiex.npu.dma_memcpy_nd(%out[%c0, %c0, %c0, %c_out_off_{col}][%c1, %c1, %c1, %c_out_len_{col}][%c0, %c0, %c0, %c1]) {{metadata = @of_out_{col}, id = {col + n_tiles} : i64, issue_token = true}} : memref<{total_out}xi32>")
        out_offset += out_elems

    # DMA inputs (no issue_token -- synchronization is on output completion)
    in_offset = 0
    for col, batch in enumerate(batches):
        in_elems = max(1, batch["in_size"] // 4)
        lines.append(f"      %c_in_off_{col} = arith.constant {in_offset} : i64")
        lines.append(f"      %c_in_len_{col} = arith.constant {in_elems} : i64")
        lines.append(f"      aiex.npu.dma_memcpy_nd(%in[%c0, %c0, %c0, %c_in_off_{col}][%c1, %c1, %c1, %c_in_len_{col}][%c0, %c0, %c0, %c1]) {{metadata = @of_in_{col}, id = {col} : i64}} : memref<{total_in}xi32>")
        in_offset += in_elems

    # Wait for all outputs
    for col in range(n_tiles):
        lines.append(f"      aiex.npu.dma_wait {{symbol = @of_out_{col}}}")

    lines.append(f"    }}")
    lines.append(f"  }}")
    lines.append(f"}}")

    return "\n".join(lines)
```

**Important**: Every `func.func private @test_kernel_N` has a DIFFERENT name, even if they all have the same signature. This is because the link_with attribute is per-function, and each tile calls a different .o file. The function inside each .o is always named `test_kernel` (the C symbol), but the MLIR declaration name must be unique.

Wait -- there's a catch. The .o file exports `test_kernel`, not `test_kernel_0`. The MLIR `link_with` attribute tells aiecc which .o to link for that function. But the linker needs the symbol names to match. We need the MLIR function name to match the symbol in the .o.

Two options:
a) All MLIR functions named `@test_kernel`, each with different `link_with` -- but MLIR requires unique function names.
b) Each .o renames its symbol -- but that changes the assembly generator.

Looking at the cascade_flows example: it uses `@extern_kernel1`, `@extern_kernel2`, `@extern_kernel3` in MLIR, matching the actual C function names `extern_kernel1`, `extern_kernel2`, `extern_kernel3` in the .o files. The symbol names MUST match.

So we need each batch .o to export a unique symbol. Options:
1. Use `objcopy --redefine-sym` to rename `test_kernel` to `test_kernel_N` after assembly
2. Change the assembly generator to use a per-batch function name
3. Use a wrapper: each core's MLIR calls `@test_kernel` but links a DIFFERENT .o -- if this works with link_with

Let me re-read the AIEAssignCoreLinkFiles pass. It assigns .o files per core based on which functions that core calls. If two cores both call `@test_kernel` but with different `link_with`, does the pass handle it?

Actually, looking more carefully: `link_with` is on the `func.func` declaration, not on the call site. So there's ONE declaration of `@test_kernel` with ONE `link_with`. Two cores calling it would both link the same .o.

So option (1) is the way: rename the symbol in each .o to be unique. `llvm-objcopy --redefine-sym test_kernel=test_kernel_0 batch_000.o`.

This is a small addition to the packaging step.

- [ ] **Step 8: Run test, verify it passes**

- [ ] **Step 9: Write test for buffer offset calculation**

```python
class TestBufferLayout(unittest.TestCase):
    def test_contiguous_output_offsets_in_i32_elements(self):
        """Offsets are in i32 elements (matching MLIR memref<Nxi32>)."""
        from isa_multi_tile_gen import compute_phase_buffer_layout
        batches = [
            {"batch_index": 0, "in_size": 128, "out_size": 64},
            {"batch_index": 1, "in_size": 256, "out_size": 128},
        ]
        layout = compute_phase_buffer_layout(batches)
        # 128 bytes = 32 i32 elements, 64 bytes = 16 i32 elements
        self.assertEqual(layout["in_offsets_elems"], [0, 32])
        self.assertEqual(layout["out_offsets_elems"], [0, 16])
        self.assertEqual(layout["total_in_elems"], 96)
        self.assertEqual(layout["total_out_elems"], 48)
        # Also track byte offsets for result splitting
        self.assertEqual(layout["in_offsets_bytes"], [0, 128])
        self.assertEqual(layout["out_offsets_bytes"], [0, 64])
```

- [ ] **Step 10: Run test, verify it fails**

- [ ] **Step 11: Implement compute_phase_buffer_layout()**

- [ ] **Step 12: Run test, verify it passes**

- [ ] **Step 13: Commit**

```bash
git add tools/isa-multi-tile-gen.py tools/test_isa_multi_tile_gen.py
git commit -m "feat(isa-harness): multi-tile MLIR generator

Assigns batches to NPU tiles (up to 4 per phase, one per column).
Generates one aie.mlir per phase with per-tile objectfifos, link_with
references, and DMA configuration for parallel execution."
```

---

## Task 4: Symbol Renaming for Per-Tile Linking

Each batch .o exports `test_kernel`. For multi-tile MLIR, each tile needs a unique symbol. Add a renaming step.

**Files:**
- Modify: `scripts/isa-test.sh`
- Modify: `tools/isa-multi-tile-gen.py`

- [ ] **Step 1: Add llvm-objcopy symbol rename to packaging**

After assembly/compilation, rename the symbol:

```bash
# In the packaging phase, before generating MLIR:
LLVM_OBJCOPY="${HOME}/npu-work/llvm-aie/install/bin/llvm-objcopy"

rename_kernel_symbol() {
    local batch_idx="$1"
    local phase_dir="$2"
    local col_idx="$3"  # column index within this phase (0-3)
    local o_src="${OUT_DIR}/batch_${batch_idx}.o"
    local o_dst="${phase_dir}/batch_${batch_idx}.o"
    cp "$o_src" "$o_dst"
    # Rename to match MLIR @test_kernel_{col} -- col is the tile column
    # index within this phase, NOT the global batch index.
    "$LLVM_OBJCOPY" --redefine-sym "test_kernel=test_kernel_${col_idx}" "$o_dst"
}
```

- [ ] **Step 2: Verify llvm-objcopy is available and works**

```bash
/home/triple/npu-work/llvm-aie/install/bin/llvm-objcopy --version
```

- [ ] **Step 3: Update generate_phase_mlir to use batch-indexed symbol names**

The `link_with` references `batch_NNN.o` and the function is named `@test_kernel_NNN`.

- [ ] **Step 4: Test with a simple 2-batch phase**

Manually create a 2-tile MLIR, compile with aiecc.py, verify it produces a valid xclbin.

- [ ] **Step 5: Commit**

```bash
git add scripts/isa-test.sh tools/isa-multi-tile-gen.py
git commit -m "feat(isa-harness): per-tile symbol renaming for multi-tile linking"
```

---

## Task 5: Multi-Tile Runner Integration

Update the runner script to use phase-based execution instead of per-batch execution.

**Files:**
- Modify: `scripts/isa-test.sh`
- Modify: `tools/instr-test-gen.py` (update `generate_test_host_cpp`)

### Design

The runner flow becomes:
1. **Generate** batches (existing, + conversion .ll batches)
2. **Assemble/Compile** all batches to .o (llvm-mc for .s, llc for .ll)
3. **Phase assignment**: Group batches into phases of 4
4. **Package phases**: For each phase, generate multi-tile MLIR, copy+rename .o files, run aiecc.py
5. **Run HW**: For each phase, run test_host with combined buffers (serial across phases)
6. **Run EMU**: Same as HW but with XDNA_EMU (parallel across phases)
7. **Compare**: Split combined output by tile, compare per-batch

- [ ] **Step 1: Add --multi-tile flag to isa-test.sh**

```bash
MULTI_TILE=false
# ... in arg parsing:
--multi-tile) MULTI_TILE=true; shift ;;
```

When `--multi-tile` is set, use the new phase-based pipeline. Otherwise, fall back to the existing single-tile pipeline for compatibility.

- [ ] **Step 2: Add phase packaging function**

```bash
package_phase() {
    local phase_idx="$1"
    local batch_list="$2"  # comma-separated batch indices
    local phase_dir="${OUT_DIR}/phase_${phase_idx}"

    mkdir -p "$phase_dir"

    # Copy and rename .o files
    IFS=',' read -ra INDICES <<< "$batch_list"
    for i in "${!INDICES[@]}"; do
        local bidx="${INDICES[$i]}"
        rename_kernel_symbol "$bidx" "$phase_dir" "$i"
    done

    # Generate multi-tile MLIR
    python3 "${PROJECT_DIR}/tools/isa-multi-tile-gen.py" \
        --manifest "$MANIFEST" \
        --batches "$batch_list" \
        --phase-idx "$phase_idx" \
        --out-dir "$phase_dir"

    # Run aiecc.py
    (cd "$phase_dir" && \
        nice -n 19 aiecc.py --no-aiesim --no-xchesscc --no-xbridge \
            --aie-generate-xclbin --xclbin-name=aie.xclbin \
            --aie-generate-npu-insts --npu-insts-name=insts.bin \
            aie.mlir 2>"${phase_dir}/aiecc.log") && \
        echo "  PKG OK: phase_${phase_idx}" || \
        echo "  PKG FAIL: phase_${phase_idx}"
}
```

- [ ] **Step 3: Update test_host.cpp for combined buffers**

The host program needs to handle larger combined input/output buffers. The current `--in-size` and `--out-size` args already support arbitrary sizes, so this may work as-is. Verify.

**PRNG note**: In multi-tile mode, the host fills one large combined input buffer with PRNG data. Each tile reads from a different offset within that buffer, so per-tile inputs differ from when the batch ran solo (where each batch had its own PRNG-filled buffer starting from offset 0). This means multi-tile validation is **HW-vs-EMU only** -- we cannot compare multi-tile output to single-tile output. This is fine since the goal is HW/EMU agreement, not single-vs-multi equivalence. Both HW and EMU see the same combined input buffer, so their outputs should match.

- [ ] **Step 4: Add phase execution and result splitting**

After running a phase, the output buffer contains results from all tiles concatenated. The comparison step needs to split by tile and compare per-batch:

```bash
compare_phase() {
    local phase_idx="$1"
    local batch_list="$2"

    IFS=',' read -ra INDICES <<< "$batch_list"
    local out_offset=0

    for i in "${!INDICES[@]}"; do
        local bidx="${INDICES[$i]}"
        local out_size=$(get_batch_out_size "$bidx")

        # Extract per-tile output from combined buffer
        dd if="${RESULTS_DIR}/phase_${phase_idx}_hw.bin" \
           bs=1 skip=$out_offset count=$out_size \
           of="${RESULTS_DIR}/batch_${bidx}_hw.bin" 2>/dev/null
        dd if="${RESULTS_DIR}/phase_${phase_idx}_emu.bin" \
           bs=1 skip=$out_offset count=$out_size \
           of="${RESULTS_DIR}/batch_${bidx}_emu.bin" 2>/dev/null

        out_offset=$((out_offset + out_size))
    done
}
```

- [ ] **Step 5: End-to-end test with 4 batches**

Run: `scripts/isa-test.sh --multi-tile --filter 'batch_00[0-3]' --no-hw`

Verify: 4 batches assigned to 1 phase, 1 xclbin generated, EMU produces output.

- [ ] **Step 6: Full run**

Run: `scripts/isa-test.sh --multi-tile --no-hw`

Verify: All batches processed in ~N/4 phases.

- [ ] **Step 7: Commit**

```bash
git add scripts/isa-test.sh tools/isa-multi-tile-gen.py tools/instr-test-gen.py
git commit -m "feat(isa-harness): multi-tile phase execution

Groups batches into phases of 4 tiles (one per NPU column). Each phase
compiles to a single xclbin, runs all tiles in parallel. Reduces total
runs from ~80 to ~20 phases."
```

---

## Task 6: Validation and Performance Measurement

- [ ] **Step 1: Run single-tile baseline (existing)**

```bash
scripts/isa-test.sh --no-hw 2>&1 | tee /tmp/isa-baseline-single.log
```

Record: total time, per-phase times.

- [ ] **Step 2: Run multi-tile**

```bash
scripts/isa-test.sh --multi-tile --no-hw 2>&1 | tee /tmp/isa-baseline-multi.log
```

Record: total time, per-phase times.

- [ ] **Step 3: Compare results**

Every batch should produce identical output whether run single-tile or multi-tile. Verify by diffing the per-batch result files.

- [ ] **Step 4: Run with HW comparison**

```bash
scripts/isa-test.sh --multi-tile 2>&1 | tee /tmp/isa-multi-hw.log
```

- [ ] **Step 5: Document results and commit**

---

## Task 7 (Deferred): 16-Tile Expansion

Expand from 4 tiles per phase (one per column) to 16 tiles (4 per column, rows 2-5). This requires:

1. Routing through mem tiles (row 1) for rows 3-5
2. More complex objectfifo chains (shim -> mem tile -> compute tile)
3. BD sharing across 4 tiles per column (16 BDs per shim, 4 tiles)

**Estimated impact**: Reduces phases from ~20 to ~5. Worth doing once 4-tile is proven.

**Key reference**: Check if MLIR-AIE objectfifo handles multi-row routing automatically (it may -- the abstraction is designed for this).

---

## Task 8 (Deferred): Control Packet Program Reload

Use control packets to reload program memory without regenerating the full xclbin. This would allow:

1. One base xclbin with routing/DMA configuration
2. Control packets that write new program memory to each tile
3. Single-xclbin execution with multiple program phases

**Key reference**: `mlir-aie/test/npu-xrt/ctrl_packet_reconfig_4x1_cores/` shows the `@base` + `@main` pattern with control packet reconfiguration.

**Estimated impact**: Eliminates aiecc.py per-phase compilation entirely. All phases share one xclbin, with control packets handling program reload between phases.

---

## Summary

| Task | What | Adds |
|------|------|------|
| 1 | ConversionStrategy | +24 unique conversion ops (144 instruction defs) |
| 2 | llc pipeline | .ll compilation alongside .s |
| 3 | Multi-tile MLIR gen | 4-tile phase MLIR generation |
| 4 | Symbol renaming | Per-tile kernel linking |
| 5 | Multi-tile runner | Phase-based execution + result splitting |
| 6 | Validation | Correctness + performance measurement |
| 7 | 16-tile (deferred) | 4x more parallelism |
| 8 | Control packets (deferred) | Eliminate per-phase compilation |

**Expected coverage after Task 1-6:**
- Current: 345/606 testable (57%)
- +144 conversion instructions (24 ops x ~6 addr variants): 489/606 (81%)
- Execution speed: ~4x faster (4 tiles per phase vs 1)
- Packaging speed: ~3.6x faster (fewer aiecc.py invocations)
