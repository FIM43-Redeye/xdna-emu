# Phase B: Trace-Based HW Cycle Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automate per-test cycle-count capture on real NPU hardware using mlir-aie's declarative trace API and a generic metadata-driven XRT runner, producing `cycles.HW.<test>.txt` files that feed the Phase C budget calculator.

**Architecture:** Three cooperating components. (1) An MLIR trace injector (`tools/mlir-trace-inject.py`) that reads a compiled-ready `.mlir`, adds declarative `aie.trace` ops for all compute tiles plus `aie.trace.host_config` + `aie.trace.start_config` inside the runtime sequence, and writes a `-traced.mlir`. (2) A generic C++ runner (`bridge-runner/bridge-trace-runner.cpp`) that reads kernel argument metadata from the xclbin via `xrt::xclbin::kernel::get_args()`, allocates and binds buffers by argument name (not by hardcoded `group_id(N)`), launches the kernel, and syncs back the `trace` buffer. (3) A cycle extractor (`tools/trace-to-cycles.py`) that wraps mlir-aie's `parse_trace()` + `get_cycles_summary()` and emits a single-line `cycles.HW.<test>.txt`. `scripts/emu-bridge-test.sh` gains a `--with-hw-cycles` flag that runs the injector + compile + runner + extractor as a side-pipeline for any test. The five legacy Python trace tools move to `tools/deprecated/` (kept for possible unique capabilities).

**Tech Stack:** mlir-aie Python bindings (AIE dialect ops), XRT C++ API (`xrt::xclbin`, `xrt::bo`, `xrt::kernel`, `xrt::hw_context`), CMake (runner build), Bash (bridge script), Python 3.13 (injector + extractor), pytest.

---

> **Sweep-as-of 2026-05-01:** All 15 tasks landed per the in-document update note in the parent plan (`2026-04-22-cycle-budget-testing.md`) and the validation results captured in `2026-04-22-phase-b-trace-cycle-capture-validation.md`. Pipeline produces valid `cycles.HW.<test>.<compiler>.txt` files for vector-bearing tests; `vector_scalar_using_dma`=41181 cycles is the reference success. Steps below were executed organically rather than ticked one-by-one; this sweep flips the checkboxes to match the verified completion state.


## Scope Check

This plan covers a single subsystem: automated HW cycle capture via mlir-aie trace infrastructure. It depends only on Phase A (already shipped) and a compiled xclbin — it does not touch emulator internals. Phase C (budget calculation + override file loader) is a separate plan that consumes the `cycles.HW.*.txt` files this plan produces.

---

## File Structure

**New files:**

- `tools/mlir-trace-inject.py` — The MLIR trace injector. Takes `input.mlir --out output.mlir [--buffer-size N] [--events EVENT1,EVENT2,...]`. Exit code 0 on success, 2 if input already has trace ops (idempotency guard), 1 on error.
- `tools/trace-to-cycles.py` — Post-processor. Takes `--trace-bin path --xclbin-mlir path --out cycles.txt`. Wraps mlir-aie's `parse_trace()` + `get_cycles_summary()`; emits one integer (total HW cycles for the primary compute tile).
- `tools/tests/test_mlir_trace_inject.py` — pytest for the injector.
- `tools/tests/test_trace_to_cycles.py` — pytest for the extractor.
- `tools/tests/fixtures/sample_untraced.mlir` — minimal AIE design for injector round-trip tests.
- `tools/tests/fixtures/sample_traced_trace.json` — canned Perfetto JSON for extractor unit test.
- `bridge-runner/CMakeLists.txt` — CMake for the runner.
- `bridge-runner/bridge-trace-runner.cpp` — Generic XRT runner.
- `bridge-runner/README.md` — Brief usage doc.
- `tools/deprecated/README.md` — Explains why the old trace tools live here.

**Modified files:**

- `scripts/emu-bridge-test.sh` — Add `--with-hw-cycles` flag; new `_run_hw_cycles_pipeline()` helper; invoke after each HW test pass when flag is set.

**Moved files:**

- `tools/trace-inject.py` → `tools/deprecated/trace-inject.py`
- `tools/trace-sweep.py` → `tools/deprecated/trace-sweep.py`
- `tools/trace-trim.py` → `tools/deprecated/trace-trim.py`
- `tools/trace-merge.py` → `tools/deprecated/trace-merge.py`
- `tools/trace-patch-events.py` → `tools/deprecated/trace-patch-events.py`

---

## Non-Obvious Facts the Implementer Needs

1. **Bridge test xclbin kernarg convention (reference, not assumed)**: Current bridge tests use `xrt::kernel::group_id(N)` with N ∈ {1..7} and hardcoded per-arg semantics (1=instr, 2=instr_size, 3=in1, 4=in2, 5=out, 6=ctrlpkts, 7=trace). **Our runner must not assume this** — it uses `xrt::xclbin::kernel::get_args()` to read arg names and dispatches by name. This is the whole point of the "generic" runner. See mlir-aie `test/npu-xrt/add_blockwrite/test.cpp` for the old hardcoded style, and `xrt_test_wrapper.h` for the template-bound middle ground we're superseding.

2. **Trace register setup is inside `bo_instr`**: Once the MLIR injector adds declarative `aie.trace` ops, `AIETraceToConfig` and `AIEInsertTraceFlows` passes (run automatically inside `aiecc.py`) lower them to `aiex.npu_write32` ops in the runtime sequence. These become part of the instruction buffer that `bo_instr` carries to the NPU. **The runner does not issue any special trace-setup calls** — it just launches the kernel normally and reads the trace buffer afterward.

3. **Trace buffer kernarg is identified by POSITION, not by name** (revised during Task 6 execution — original plan said by name "trace"): After injection, the xclbin's kernarg metadata includes an additional output arg. The newer mlir-aie convention names this arg `trace`, but the toolchain version installed on this workstation (built from `xlnx_rel_v2025.2`) emits it as a generic `bo4` (the Nth buffer in declaration order). To stay portable across both conventions, the runner classifies kernargs POSITIONALLY — the trace buffer is "the last buffer kernarg after inputs/outputs are accounted for." Other kernarg roles (opcode, instr, ninstr/instr_size, middle I/O buffers) are also identified by position rather than by exact string matches. The spirit of the plan ("read xclbin metadata, don't hardcode `group_id(N)`") still holds: we do read `xrt::xclbin::kernel::get_args()`, we just dispatch on the position/type classification rather than on the name string. Decision user-approved 2026-04-22 after the Task 6 aiecc integration test revealed the `bo4` naming.

4. **Idempotency**: If a `.mlir` already contains `aie.trace` ops (someone added them by hand), the injector must not double-inject. It detects existing trace ops and bails with exit code 2; the bridge pipeline treats that as "already instrumented, skip injection."

5. **Events we capture**: Default event set is `CoreEvent.INSTR_VECTOR` (vector-instruction retirement) plus `CoreEvent.INSTR_EVENT_0` / `CoreEvent.INSTR_EVENT_1` (standard kernel-boundary markers, if the kernel happens to emit them). The cycle extractor uses the first and last event timestamps across all core-tile records as the interval, not a specific pair — this works whether or not the kernel uses INSTR_EVENT_0/1 markers.

6. **Trace buffer size**: Default 8 KiB. Plenty for the short kernels bridge tests run. Grows via `--buffer-size` flag if we hit overflow later.

7. **Why we verify the mlir-aie Python API exists before writing code**: The declarative API functions (`configure_trace`, `start_trace`) are designed to be called from within an IRON Python design while a context is active. We're instead parsing a pre-existing MLIR string, walking it, and constructing `aie.trace` ops programmatically via the dialect Python bindings. Task 2 explicitly includes a verification step that walks the mlir-aie Python API surface; if the expected bindings are missing we escalate, because guessing here will waste time.

---

## Task 1: Move deprecated trace tools

**Files:**
- Move: `tools/trace-inject.py`, `tools/trace-sweep.py`, `tools/trace-trim.py`, `tools/trace-merge.py`, `tools/trace-patch-events.py` → `tools/deprecated/`
- Create: `tools/deprecated/README.md`

- [x] **Step 1: Create deprecated directory**

```bash
mkdir -p /home/triple/npu-work/xdna-emu/tools/deprecated
```

- [x] **Step 2: Move the five trace tools**

```bash
cd /home/triple/npu-work/xdna-emu
git mv tools/trace-inject.py tools/deprecated/
git mv tools/trace-sweep.py tools/deprecated/
git mv tools/trace-trim.py tools/deprecated/
git mv tools/trace-merge.py tools/deprecated/
git mv tools/trace-patch-events.py tools/deprecated/
```

- [x] **Step 3: Create README explaining the move**

Create `tools/deprecated/README.md` with this content:

```markdown
# Deprecated Trace Tools

These tools predate mlir-aie's declarative trace IRON API (PR #2988 / commit
e4f35d643c, merged 2026-03-30). The mainline cycle-capture pipeline now uses
`tools/mlir-trace-inject.py` + `bridge-runner/bridge-trace-runner` +
`tools/trace-to-cycles.py`, which delegate the heavy lifting to mlir-aie's
`aie.utils.trace` module (`configure_trace`, `start_trace`, `parse_trace`,
`get_cycles_summary`).

## Why we kept these files

The declarative API covers our needs but may not cover every one-off use case
the old tools served. We keep the source here, outside the default tooling
path, so anyone needing a dropped capability can:

1. Open the old tool to see how it did the thing
2. Either port that capability into the mainline pipeline or run the old tool
   directly (it still works — it just isn't invoked by default)

## Contents

| File | Original purpose |
|------|------------------|
| `trace-inject.py` | Inject trace routing into MLIR (capacity planner, collision-aware IDs) |
| `trace-sweep.py` | Multi-batch event sweep orchestrator |
| `trace-trim.py` | Strip sentinel padding from raw trace buffers |
| `trace-merge.py` | Merge per-batch Perfetto JSON with TRUE anchor alignment |
| `trace-patch-events.py` | Patch event slots in compiled insts.bin without recompilation |

## Do not add new callers

Any new tracing feature belongs in the mainline pipeline, not here. If the
mainline pipeline lacks a capability you need, extend it — don't reach back
for these.
```

- [x] **Step 4: Grep for any remaining references to the moved tools**

Run: `grep -rn "tools/trace-inject\|tools/trace-sweep\|tools/trace-trim\|tools/trace-merge\|tools/trace-patch-events" /home/triple/npu-work/xdna-emu --include='*.sh' --include='*.py' --include='*.md' --include='*.rs'`

Expected: only matches inside `tools/deprecated/` (the tools referencing themselves) and maybe CLAUDE.md. Fix CLAUDE.md references if any — point them at `tools/deprecated/`.

- [x] **Step 5: Verify nothing in the repo is broken by the move**

```bash
cd /home/triple/npu-work/xdna-emu
./scripts/emu-bridge-test.sh --no-hw --peano-only -v add_one 2>&1 | tail -20
```

Expected: test passes with no "trace-inject.py: not found" errors. (This filter picks a single quick test; the goal is "nothing explodes," not a full run.)

- [x] **Step 6: Commit**

```bash
git add tools/deprecated/ tools/ docs/ CLAUDE.md
git commit -m "refactor: move legacy trace tools to tools/deprecated/

The five Python trace tools (trace-inject, trace-sweep, trace-trim,
trace-merge, trace-patch-events) predate mlir-aie's declarative trace
IRON API. The new pipeline (Phase B of cycle-budget testing) supersedes
them. Kept in tools/deprecated/ with a README explaining the move, in
case any of them have capabilities the new pipeline misses.
"
```

---

## Task 2: MLIR injector — scaffolding + API verification

**Files:**
- Create: `tools/mlir-trace-inject.py`
- Create: `tools/tests/test_mlir_trace_inject.py`
- Create: `tools/tests/fixtures/sample_untraced.mlir`

- [x] **Step 1: Verify the mlir-aie Python API is available**

Run this probe to confirm the bindings exist:

```bash
cd /home/triple/npu-work/xdna-emu
PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python python3 -c "
import aie.ir
import aie.dialects.aie as aied
print('aie dialect module:', aied.__file__)
# Look for trace ops — exact names may vary; report what's available
trace_names = [n for n in dir(aied) if 'trace' in n.lower() or 'Trace' in n]
print('trace-related symbols:', trace_names)
# Parser entry point
print('Module.parse available:', hasattr(aie.ir.Module, 'parse'))
"
```

Expected: prints a list of trace-related symbols. We expect to see at least one of: `TraceOp`, `trace`, `TraceEventOp`, `TraceHostConfigOp`, `TraceStartConfigOp`.

If the list is empty or only shows unrelated symbols (e.g., just `trace_utils`), escalate. The plan assumes `aie.dialects.aie` exposes trace ops; if the bindings aren't yet Python-accessible we need a different approach (inline editing via `aie-opt` CLI, or a C++ pass).

Record what you find in the injector source as a comment at the top, so future readers know which specific symbols we depend on.

- [x] **Step 2: Create the sample fixture**

Create `tools/tests/fixtures/sample_untraced.mlir`:

```mlir
module {
  aie.device(npu1_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %buf = aie.buffer(%t02) {sym_name = "buf0"} : memref<16xi32>
    aie.core(%t02) {
      aie.end
    }
    aiex.runtime_sequence(%arg0: memref<16xi32>) {
      aiex.npu_dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @inbound} : memref<16xi32>
      aiex.npu_dma_wait {symbol = @inbound}
    }
  }
}
```

If this MLIR doesn't parse cleanly under the mlir-aie version in `install/`, adjust the syntax minimally — the goal is a valid untraced design, not a specific shape. Document any syntax adjustments needed as a comment in the fixture file.

- [x] **Step 3: Write the failing injector test (round-trip identity with --no-op mode)**

Create `tools/tests/test_mlir_trace_inject.py`:

```python
"""Tests for tools/mlir-trace-inject.py."""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INJECTOR = REPO / "tools" / "mlir-trace-inject.py"
FIXTURES = REPO / "tools" / "tests" / "fixtures"
UNTRACED = FIXTURES / "sample_untraced.mlir"


def _run(args, check=True):
    return subprocess.run(
        ["python3", str(INJECTOR), *args],
        capture_output=True,
        text=True,
        check=check,
    )


def test_injector_exists_and_prints_help():
    r = _run(["--help"])
    assert "usage" in r.stdout.lower() or "usage" in r.stderr.lower()


def test_injector_no_op_mode_round_trips(tmp_path):
    """With --no-op, injector should read and write the MLIR unchanged."""
    out = tmp_path / "out.mlir"
    r = _run(["--no-op", "--input", str(UNTRACED), "--out", str(out)])
    assert r.returncode == 0, f"injector failed: stderr={r.stderr}"
    original = UNTRACED.read_text()
    result = out.read_text()
    # The mlir-aie parser may normalize whitespace; compare parsed structure,
    # not raw text. At minimum, tile count and op kinds should match.
    assert result.count("aie.tile") == original.count("aie.tile")
    assert result.count("aie.device") == original.count("aie.device")
```

- [x] **Step 4: Run the test to verify it fails**

Run: `cd /home/triple/npu-work/xdna-emu && PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python pytest tools/tests/test_mlir_trace_inject.py -v`

Expected: FAIL with "No such file or directory: .../mlir-trace-inject.py" or similar.

- [x] **Step 5: Write the minimal injector skeleton**

Create `tools/mlir-trace-inject.py`:

```python
#!/usr/bin/env python3
"""
mlir-trace-inject.py -- Programmatically add declarative trace ops to an
AIE MLIR design.

Uses mlir-aie's Python bindings to parse an existing .mlir, walk its
aie.device body, insert aie.trace declarations for each compute tile, and
insert aie.trace.host_config + aie.trace.start_config ops at the top of
the aiex.runtime_sequence body.

Rationale: mlir-aie's declarative trace IRON API (aie.utils.trace) is
designed to be called from within an IRON Python design while a context
is active. We instead need to inject into pre-existing MLIR produced by
an arbitrary upstream tool. Rather than text-munging, we parse into the
MLIR IR and construct ops through the aie dialect Python bindings.

Dependencies (verified in Task 2 Step 1 of the plan):
  - aie.ir (MLIR Python bindings from mlir-aie install/python)
  - aie.dialects.aie (AIE dialect with trace ops)

Usage:
  mlir-trace-inject.py --input design.mlir --out design-traced.mlir
  mlir-trace-inject.py --no-op --input design.mlir --out copy.mlir
"""
import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--input", required=True, help="path to input .mlir")
    p.add_argument("--out", required=True, help="path to write traced .mlir")
    p.add_argument("--buffer-size", type=int, default=8192,
                   help="trace buffer size in bytes (default: 8192)")
    p.add_argument("--no-op", action="store_true",
                   help="parse and reserialize without injecting (for testing)")
    return p.parse_args()


def main():
    args = parse_args()
    text = Path(args.input).read_text()
    # Import here so --help works without the mlir-aie env activated.
    import aie.ir as ir
    import aie.dialects.aie as aied
    with ir.Context() as ctx:
        aied.register_dialect(ctx)
        module = ir.Module.parse(text)
        if not args.no_op:
            # Injection logic lands in Tasks 3-5.
            raise NotImplementedError("injection not yet implemented")
        Path(args.out).write_text(str(module))


if __name__ == "__main__":
    sys.exit(main() or 0)
```

- [x] **Step 6: Make the script executable and rerun the test**

```bash
chmod +x /home/triple/npu-work/xdna-emu/tools/mlir-trace-inject.py
cd /home/triple/npu-work/xdna-emu
PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python pytest tools/tests/test_mlir_trace_inject.py -v
```

Expected: both tests PASS.

- [x] **Step 7: Commit**

```bash
git add tools/mlir-trace-inject.py tools/tests/test_mlir_trace_inject.py tools/tests/fixtures/sample_untraced.mlir
git commit -m "feat(trace): scaffolding for MLIR trace injector

Skeleton of tools/mlir-trace-inject.py with --no-op parse/reserialize
mode. Verifies the mlir-aie Python bindings expose the trace dialect
ops we'll need in later tasks. pytest round-trip test confirms the
MLIR parser + serializer behave as expected on a minimal fixture.

No trace injection yet -- that lands in Tasks 3-5."
```

---

## Task 3: MLIR injector — idempotency guard

**Files:**
- Modify: `tools/mlir-trace-inject.py`
- Modify: `tools/tests/test_mlir_trace_inject.py`
- Create: `tools/tests/fixtures/sample_already_traced.mlir`

- [x] **Step 1: Create the already-traced fixture**

Create `tools/tests/fixtures/sample_already_traced.mlir` — copy of `sample_untraced.mlir` with a single `aie.trace` op added inside the device body. Use whatever exact trace op spelling the mlir-aie bindings exposed in Task 2 Step 1. A reasonable template:

```mlir
module {
  aie.device(npu1_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.trace @trace_existing(%t02) {
      aie.trace.mode "Event-Time"
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_VECTOR">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    %buf = aie.buffer(%t02) {sym_name = "buf0"} : memref<16xi32>
    aie.core(%t02) { aie.end }
    aiex.runtime_sequence(%arg0: memref<16xi32>) {
      aiex.npu_dma_wait {symbol = @inbound}
    }
  }
}
```

If this syntax doesn't parse under the installed mlir-aie, simplify until it does. What matters is that the fixture contains at least one `aie.trace` op.

- [x] **Step 2: Add the failing test**

Append to `tools/tests/test_mlir_trace_inject.py`:

```python
ALREADY_TRACED = FIXTURES / "sample_already_traced.mlir"


def test_injector_bails_on_already_traced(tmp_path):
    """If input already has aie.trace ops, injector should refuse (exit 2)."""
    out = tmp_path / "out.mlir"
    r = _run(
        ["--input", str(ALREADY_TRACED), "--out", str(out)],
        check=False,
    )
    assert r.returncode == 2, f"expected exit 2, got {r.returncode}; stderr={r.stderr}"
    assert "already" in r.stderr.lower() or "trace" in r.stderr.lower()
    assert not out.exists(), "output file should not be written when injector refuses"
```

- [x] **Step 3: Run the test to verify it fails**

Run: `cd /home/triple/npu-work/xdna-emu && PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python pytest tools/tests/test_mlir_trace_inject.py::test_injector_bails_on_already_traced -v`

Expected: FAIL (currently injector raises `NotImplementedError`, not exit 2).

- [x] **Step 4: Implement the idempotency check**

In `tools/mlir-trace-inject.py`, replace the `if not args.no_op:` body with:

```python
        # Detect existing trace ops -- walking the module is safer than
        # text-searching for "aie.trace" because substring matches can fire
        # on unrelated strings (e.g., attribute names).
        found_trace = False
        def _walk(op):
            nonlocal found_trace
            if op.operation.name == "aie.trace":
                found_trace = True
                return
            for region in op.operation.regions:
                for block in region.blocks:
                    for inner in block.operations:
                        _walk(inner)
        for op in module.body.operations:
            _walk(op)
        if found_trace and not args.no_op:
            print(
                f"error: {args.input} already contains aie.trace ops; "
                "refusing to double-inject (exit 2)",
                file=sys.stderr,
            )
            return 2
        if not args.no_op:
            # Injection logic lands in Tasks 4-5.
            raise NotImplementedError("injection not yet implemented")
        Path(args.out).write_text(str(module))
```

Note: MLIR Python binding op-traversal syntax varies. The exact
`op.operation.name` / `op.operation.regions` spelling may need adjustment
based on what Task 2 Step 1 revealed. If your traversal API differs,
adapt the `_walk` function to match.

- [x] **Step 5: Run the test to verify it passes**

Run: `cd /home/triple/npu-work/xdna-emu && PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python pytest tools/tests/test_mlir_trace_inject.py -v`

Expected: all tests PASS (including the previous round-trip tests).

- [x] **Step 6: Commit**

```bash
git add tools/mlir-trace-inject.py tools/tests/test_mlir_trace_inject.py tools/tests/fixtures/sample_already_traced.mlir
git commit -m "feat(trace): idempotency guard in MLIR trace injector

If the input MLIR already contains aie.trace ops, the injector refuses
with exit code 2 and does not write the output file. This prevents
double-injection when a test has been manually instrumented.
"
```

---

## Task 4: MLIR injector — insert `aie.trace` ops for compute tiles

**Files:**
- Modify: `tools/mlir-trace-inject.py`
- Modify: `tools/tests/test_mlir_trace_inject.py`

- [x] **Step 1: Add the failing test**

Append to `tools/tests/test_mlir_trace_inject.py`:

```python
def test_injector_adds_trace_decl_per_compute_tile(tmp_path):
    """Each non-shim tile in the input should get one aie.trace decl."""
    out = tmp_path / "out.mlir"
    r = _run(["--input", str(UNTRACED), "--out", str(out)])
    assert r.returncode == 0, f"stderr={r.stderr}"
    result = out.read_text()
    # Fixture has one compute tile (0, 2). Shim tile (0, 0) is not compute.
    trace_count = result.count("aie.trace ")  # trailing space avoids matching "aie.trace.event"
    # We expect exactly 1 trace decl (for tile 0,2).
    assert trace_count == 1, f"expected 1 aie.trace decl, got {trace_count}\n---\n{result}"
    # Confirm the tile it attaches to:
    assert "%t02" in result or "tile(0, 2)" in result.replace(" ", "")
```

- [x] **Step 2: Run the test to verify it fails**

Run: `cd /home/triple/npu-work/xdna-emu && PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python pytest tools/tests/test_mlir_trace_inject.py::test_injector_adds_trace_decl_per_compute_tile -v`

Expected: FAIL with `NotImplementedError` or similar.

- [x] **Step 3: Implement tile discovery and trace op construction**

This is the heart of the injector. You have two construction paths, and which one works depends on what Task 2 Step 1 revealed about the mlir-aie Python bindings:

**Path A (strongly preferred, if the bindings expose trace op constructors):**
Construct each `aie.trace` op by calling the generated Python class, e.g. `aied.TraceOp(tile=t02, ...)`. Use the same pattern you see in `mlir-aie/python/utils/trace/setup.py` around `configure_trace()`.

**Path B (fallback, if Python constructors aren't exposed):**
Build a snippet of MLIR text with the trace ops, parse it into a temporary module, then splice the trace ops into the real module's device body via `op.move_before(...)` or similar IR-surgery APIs.

Replace the `if not args.no_op:` body (currently `raise NotImplementedError`) with logic that:

1. Walks `module.body.operations` to find the `aie.device` op (should be exactly one at the top).
2. Inside the device body, walks operations to find `aie.tile` ops and groups them by (col, row).
3. Skips shim tiles (row == 0). Skips mem tiles — mem tiles are architecture-specific; for NPU1 memtile row is 1. Use `row > 1` as the compute-tile predicate for npu1 and npu1_1col. Record this as a known limitation; a generalized check would look at the device model, but for our bridge-test targets (npu1, npu1_1col) this is correct.
4. For each compute tile, constructs an `aie.trace` op with:
   - `sym_name = "trace_t{col}_{row}"`
   - mode `"Event-Time"`
   - packet `id=1 type=core` (IDs auto-allocated per trace by mlir-aie's pass; starting at 1 is safe)
   - events: `INSTR_VECTOR`, `INSTR_EVENT_0`, `INSTR_EVENT_1` (first is required for cycle extraction; last two are common kernel markers)
   - `start broadcast=15`, `stop broadcast=14`
5. Inserts the trace op into the device body after the tile declarations but before `aie.core` bodies (order matters for MLIR verification).

Pseudocode skeleton (adapt to the actual bindings):

```python
# --- inside main() after idempotency check, before `Path(args.out).write_text(...)` ---
device_op = None
for op in module.body.operations:
    if op.operation.name == "aie.device":
        device_op = op
        break
if device_op is None:
    print(f"error: no aie.device op found in {args.input}", file=sys.stderr)
    return 1

# Collect compute tiles inside the device body.
tiles = []  # list of (col, row, ssa_value)
device_body = device_op.operation.regions[0].blocks[0]
for inner in device_body.operations:
    if inner.operation.name == "aie.tile":
        # Extract col/row attrs. Exact attribute names: check mlir-aie's AIETile op.
        col = int(inner.operation.attributes["col"].value)
        row = int(inner.operation.attributes["row"].value)
        if row > 1:  # compute tile on npu1 / npu1_1col
            tiles.append((col, row, inner.result))

if not tiles:
    print(f"warning: no compute tiles found in {args.input}; "
          "writing through unchanged (nothing to trace)", file=sys.stderr)
    Path(args.out).write_text(str(module))
    return 0

# Construct aie.trace ops, one per compute tile, and append to device body.
# IMPORTANT: insert BEFORE any aie.core op to satisfy op ordering constraints.
# If Path A works:
for col, row, tile_val in tiles:
    trace_op = aied.TraceOp(
        tile=tile_val,
        sym_name=f"trace_t{col}_{row}",
        # ...event list, mode, packet, broadcast channels...
    )
    # Insert before first aie.core in the device body
    # (exact API: check what the mlir-aie bindings expose)

# If Path B is needed, build this MLIR text per tile:
trace_text_template = """
aie.trace @trace_t{col}_{row}(%arg0) {{
  aie.trace.mode "Event-Time"
  aie.trace.packet id={pid} type=core
  aie.trace.event<"INSTR_VECTOR">
  aie.trace.event<"INSTR_EVENT_0">
  aie.trace.event<"INSTR_EVENT_1">
  aie.trace.start broadcast=15
  aie.trace.stop broadcast=14
}}
"""
# Then parse and splice -- the exact splice API depends on the bindings.
```

Document in the commit message which path (A or B) you took and why.

- [x] **Step 4: Run the test to verify it passes**

Run: `cd /home/triple/npu-work/xdna-emu && PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python pytest tools/tests/test_mlir_trace_inject.py -v`

Expected: all tests PASS.

- [x] **Step 5: Visually inspect a traced output**

```bash
cd /home/triple/npu-work/xdna-emu
mkdir -p /tmp/claude-1000/trace-test
PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python \
  python3 tools/mlir-trace-inject.py \
    --input tools/tests/fixtures/sample_untraced.mlir \
    --out /tmp/claude-1000/trace-test/sample_traced.mlir
diff tools/tests/fixtures/sample_untraced.mlir /tmp/claude-1000/trace-test/sample_traced.mlir
```

Expected: diff shows added `aie.trace` op(s). Skim the output for obvious wrongness (e.g., trace ops outside the device body).

- [x] **Step 6: Commit**

```bash
git add tools/mlir-trace-inject.py tools/tests/test_mlir_trace_inject.py
git commit -m "feat(trace): inject aie.trace ops for compute tiles

Walks the aie.device body, identifies compute tiles (row > 1 for
npu1_1col / npu1), and inserts one aie.trace decl per tile with a
default event set (INSTR_VECTOR, INSTR_EVENT_0, INSTR_EVENT_1).

Runtime-sequence host_config + start_config lands in Task 5.
"
```

---

## Task 5: MLIR injector — insert runtime-sequence trace host_config + start_config

**Files:**
- Modify: `tools/mlir-trace-inject.py`
- Modify: `tools/tests/test_mlir_trace_inject.py`

- [x] **Step 1: Add the failing test**

Append to `tools/tests/test_mlir_trace_inject.py`:

```python
def test_injector_adds_runtime_sequence_trace_config(tmp_path):
    """The aiex.runtime_sequence body should start with trace.host_config +
    one trace.start_config per trace decl, before existing runtime ops."""
    out = tmp_path / "out.mlir"
    r = _run([
        "--input", str(UNTRACED),
        "--out", str(out),
        "--buffer-size", "16384",
    ])
    assert r.returncode == 0, f"stderr={r.stderr}"
    result = out.read_text()
    assert "aie.trace.host_config" in result
    assert "aie.trace.start_config" in result
    # Buffer size should flow through
    assert "16384" in result, "custom buffer size did not reach the output"
```

- [x] **Step 2: Run the test to verify it fails**

Run: `cd /home/triple/npu-work/xdna-emu && PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python pytest tools/tests/test_mlir_trace_inject.py::test_injector_adds_runtime_sequence_trace_config -v`

Expected: FAIL — no `aie.trace.host_config` in the output yet.

- [x] **Step 3: Implement runtime-sequence injection**

Extend the injector logic inside `main()` (after compute-tile trace-op insertion, still inside `if not args.no_op:`):

1. Find the `aiex.runtime_sequence` op inside the device body.
2. At the very start of its body, insert:
   ```
   aie.trace.host_config buffer_size=<args.buffer_size>
   aie.trace.start_config @trace_t{col}_{row}   // one per compute tile
   ```

Pseudocode skeleton (adapt to bindings):

```python
# Find the runtime sequence op -- exact op name may be "aiex.runtime_sequence".
rt_op = None
for inner in device_body.operations:
    if inner.operation.name == "aiex.runtime_sequence":
        rt_op = inner
        break
if rt_op is None:
    print("warning: no aiex.runtime_sequence found; "
          "trace configuration not activated", file=sys.stderr)
    Path(args.out).write_text(str(module))
    return 0

rt_body = rt_op.operation.regions[0].blocks[0]

# Inserting at the top of a block: use ir.InsertionPoint.at_block_begin(rt_body)
# Check the mlir-aie bindings for the exact spelling.
with ir.InsertionPoint.at_block_begin(rt_body):
    # aie.trace.host_config buffer_size=<N>
    aied.TraceHostConfigOp(buffer_size=args.buffer_size)
    for col, row, _ in tiles:
        # aie.trace.start_config @trace_t{col}_{row}
        aied.TraceStartConfigOp(f"trace_t{col}_{row}")
```

If the Python constructors for `TraceHostConfigOp` / `TraceStartConfigOp` don't exist, fall back to Path B (construct MLIR text and splice). Document which path you took.

- [x] **Step 4: Run the test to verify it passes**

Run: `cd /home/triple/npu-work/xdna-emu && PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python pytest tools/tests/test_mlir_trace_inject.py -v`

Expected: all tests PASS.

- [x] **Step 5: Commit**

```bash
git add tools/mlir-trace-inject.py tools/tests/test_mlir_trace_inject.py
git commit -m "feat(trace): inject trace host_config + start_config into runtime sequence

The runtime sequence body now starts with:
  aie.trace.host_config buffer_size=<N>
  aie.trace.start_config @trace_t<col>_<row>   (one per compute tile)

This activates the trace declarations inserted in Task 4. mlir-aie's
AIEInsertTraceFlows pass will lower the declarative ops to register
writes during aiecc.py compilation.
"
```

---

## Task 6: MLIR injector — end-to-end with `aiecc.py`

**Files:**
- Modify: `tools/tests/test_mlir_trace_inject.py` (integration test)

This task has no new product code — just proves the injector output survives a real compile.

- [x] **Step 1: Pick a real bridge test MLIR to use as a compile target**

Find a small, known-good test. Good candidates include `add_one`, `add_256`, or similar. The test's MLIR typically lives at `mlir-aie/test/npu-xrt/<name>/aie.mlir` (or is generated from a .py).

```bash
find /home/triple/npu-work/mlir-aie/test/npu-xrt -maxdepth 3 -name "aie*.mlir" | head -10
```

Pick one. Record the path in the integration test.

- [x] **Step 2: Add the integration test**

Append to `tools/tests/test_mlir_trace_inject.py`:

```python
import os
import shutil

# Path to a small known-good bridge test design.
# If this path no longer exists, pick another from
#   find /home/triple/npu-work/mlir-aie/test/npu-xrt -name "aie*.mlir"
BRIDGE_TEST_MLIR = Path(
    "/home/triple/npu-work/mlir-aie/test/npu-xrt/add_one/aie.mlir"
)


def test_injector_output_compiles_with_aiecc(tmp_path):
    """Traced MLIR should compile cleanly via aiecc.py."""
    if not BRIDGE_TEST_MLIR.exists():
        import pytest
        pytest.skip(f"bridge test MLIR not found: {BRIDGE_TEST_MLIR}")
    if shutil.which("aiecc.py") is None:
        import pytest
        pytest.skip("aiecc.py not on PATH; activate mlir-aie environment")

    traced = tmp_path / "aie-traced.mlir"
    # Inject
    r = _run(["--input", str(BRIDGE_TEST_MLIR), "--out", str(traced)])
    assert r.returncode == 0, f"injector failed: stderr={r.stderr}"

    # Compile
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    r2 = subprocess.run(
        [
            "aiecc.py",
            "--aie-generate-xclbin",
            "--aie-generate-npu-insts",
            "--no-compile-host",
            f"--xclbin-name={build_dir}/aie-traced.xclbin",
            f"--npu-insts-name={build_dir}/insts.bin",
            str(traced),
        ],
        capture_output=True,
        text=True,
        cwd=str(build_dir),
    )
    assert r2.returncode == 0, (
        f"aiecc failed:\nSTDOUT:\n{r2.stdout}\nSTDERR:\n{r2.stderr}"
    )
    assert (build_dir / "aie-traced.xclbin").exists()
```

- [x] **Step 3: Run the test**

```bash
cd /home/triple/npu-work/xdna-emu
source /home/triple/npu-work/toolchain-build/activate-npu-env.sh
PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python \
  pytest tools/tests/test_mlir_trace_inject.py::test_injector_output_compiles_with_aiecc -v
```

Expected: PASS. If it fails, read the aiecc output carefully — often the failure points to a specific malformed op in the injected MLIR, and you go fix the injector.

Common failure modes and how to read them:
- `error: expected 'aie.end'` — trace ops inserted inside a region that required a terminator; fix insertion point
- `operation must be in aie.device region` — trace ops went to the module body instead of device body
- `unknown op 'aie.trace'` — bindings version mismatch; check that mlir-aie install is the one with PR #2988

- [x] **Step 4: Verify the compiled xclbin has a `trace` kernarg**

```bash
cd /home/triple/npu-work/xdna-emu
xclbinutil --info --input /tmp/pytest-of-*/pytest-current/test_injector_output_compiles0/build/aie-traced.xclbin 2>&1 | grep -i -E "(arg|trace|kernel)" | head -30
```

Expected: output mentions `trace` in a kernarg entry. If it doesn't, the trace injection didn't survive lowering — investigate `AIEInsertTraceFlows` pass in mlir-aie.

- [x] **Step 5: Commit**

```bash
git add tools/tests/test_mlir_trace_inject.py
git commit -m "test(trace): end-to-end compile of injected MLIR via aiecc.py

Integration test that injects trace ops into a real bridge-test MLIR
(add_one/aie.mlir) and confirms aiecc.py produces a valid xclbin with
a 'trace' kernarg. Skips if aiecc.py is not on PATH.
"
```

---

## Task 7: Generic runner — skeleton + CLI

**Files:**
- Create: `bridge-runner/CMakeLists.txt`
- Create: `bridge-runner/bridge-trace-runner.cpp`
- Create: `bridge-runner/README.md`

- [x] **Step 1: Create the CMakeLists**

Create `bridge-runner/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(bridge-trace-runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# XRT comes from /opt/xilinx/xrt by default.
set(XRT_ROOT "/opt/xilinx/xrt" CACHE PATH "XRT install root")
include_directories(SYSTEM "${XRT_ROOT}/include")
link_directories("${XRT_ROOT}/lib")

add_executable(bridge-trace-runner bridge-trace-runner.cpp)
target_link_libraries(bridge-trace-runner PRIVATE xrt_coreutil uuid)
target_compile_options(bridge-trace-runner PRIVATE -Wall -Wextra)

install(TARGETS bridge-trace-runner RUNTIME DESTINATION bin)
```

- [x] **Step 2: Create the runner skeleton**

Create `bridge-runner/bridge-trace-runner.cpp`:

```cpp
// SPDX-License-Identifier: MIT
//
// bridge-trace-runner.cpp -- Generic XRT runner for trace-instrumented
// AIE xclbins.
//
// Reads kernel argument metadata from the xclbin via
// xrt::xclbin::kernel::get_args() and allocates/binds buffers by argument
// name, not by hardcoded group_id(N). Any xclbin that follows the
// mlir-aie kernarg convention (instr, instr_size, inputs, outputs,
// optional ctrlpkts, optional trace) works without source changes.
//
// Output: raw trace buffer contents written to --trace-out, ready to be
// parsed by tools/trace-to-cycles.py.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct CliArgs {
    std::string xclbin;
    std::string kernel_name;       // empty = auto-detect single kernel
    std::string instr_bin;         // path to insts.bin
    std::string trace_out;         // where to dump raw trace bytes
    std::vector<std::string> inputs;   // paths to input buffer binaries
    std::vector<std::string> outputs;  // paths where outputs get written
    uint64_t trace_size_bytes = 8192;
    bool verbose = false;
};

void print_usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s --xclbin <path> --instr <insts.bin> "
        "--trace-out <path> [--kernel <name>] [--input <bin>]... "
        "[--output <path>]... [--trace-size N] [-v]\n",
        argv0);
}

int parse_cli(int argc, char** argv, CliArgs& out) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need_val = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s needs a value\n", flag);
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--xclbin")            out.xclbin = need_val("--xclbin");
        else if (a == "--kernel")       out.kernel_name = need_val("--kernel");
        else if (a == "--instr")        out.instr_bin = need_val("--instr");
        else if (a == "--trace-out")    out.trace_out = need_val("--trace-out");
        else if (a == "--input")        out.inputs.push_back(need_val("--input"));
        else if (a == "--output")       out.outputs.push_back(need_val("--output"));
        else if (a == "--trace-size")   out.trace_size_bytes = std::strtoull(need_val("--trace-size"), nullptr, 0);
        else if (a == "-v" || a == "--verbose") out.verbose = true;
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); std::exit(0); }
        else {
            std::fprintf(stderr, "error: unknown arg: %s\n", a.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }
    if (out.xclbin.empty() || out.instr_bin.empty() || out.trace_out.empty()) {
        std::fprintf(stderr, "error: --xclbin, --instr, and --trace-out are required\n");
        print_usage(argv[0]);
        return 1;
    }
    return 0;
}

} // anonymous namespace

int main(int argc, char** argv) {
    CliArgs args;
    if (int rc = parse_cli(argc, argv, args); rc != 0) return rc;
    std::fprintf(stderr, "bridge-trace-runner: xclbin=%s instr=%s trace_out=%s\n",
                 args.xclbin.c_str(), args.instr_bin.c_str(), args.trace_out.c_str());
    // XRT logic lands in Tasks 8-10.
    return 0;
}
```

- [x] **Step 3: Create the README**

Create `bridge-runner/README.md`:

```markdown
# bridge-trace-runner

Generic XRT runner for trace-instrumented AIE xclbins.

Replaces the per-test `test.exe` for bridge tests when the test has been
trace-instrumented via `tools/mlir-trace-inject.py`. Instead of hardcoding
`group_id(1..7)` like traditional bridge tests do, this runner reads the
xclbin's kernarg metadata and dispatches each argument by name.

## Build

    cmake -S bridge-runner -B bridge-runner/build
    cmake --build bridge-runner/build

## Usage

    bridge-trace-runner \
      --xclbin aie-traced.xclbin \
      --instr insts.bin \
      --input in0.bin \
      --output out0.bin \
      --trace-out trace.bin \
      --trace-size 8192

The runner looks up each of these kernargs by name in the xclbin's
kernel metadata:

  - `instr` (instruction buffer, fed from `--instr`)
  - `instr_size` (scalar, derived from instr file size)
  - `in`, `in0`, `in1`, `inA`, `inB` (input buffers, consumed from `--input`
    in order)
  - `out`, `out0`, `out1` (output buffers, written to `--output` paths in
    order)
  - `ctrlpkts` (optional control packets; if present and no `--ctrlpkts`
    flag given, an 8-byte zeroed buffer is bound)
  - `trace` (optional trace buffer; size from `--trace-size`; contents
    written to `--trace-out`)

Exit 0 on success.
```

- [x] **Step 4: Build the skeleton**

```bash
cd /home/triple/npu-work/xdna-emu
cmake -S bridge-runner -B bridge-runner/build -DCMAKE_BUILD_TYPE=Debug
cmake --build bridge-runner/build
```

Expected: builds cleanly, produces `bridge-runner/build/bridge-trace-runner`.

- [x] **Step 5: Smoke-test the CLI**

```bash
/home/triple/npu-work/xdna-emu/bridge-runner/build/bridge-trace-runner --help
/home/triple/npu-work/xdna-emu/bridge-runner/build/bridge-trace-runner --xclbin /dev/null --instr /dev/null --trace-out /tmp/claude-1000/foo
```

Expected: `--help` prints usage (returns 0); the second invocation prints the
`xclbin=...` line and exits 0. No XRT calls yet.

- [x] **Step 6: Commit**

```bash
git add bridge-runner/
git commit -m "feat(bridge-runner): skeleton + CLI for generic trace-aware runner

Parses CLI args only; no XRT logic yet. CMake build produces a single
binary at bridge-runner/build/bridge-trace-runner. README documents the
kernarg-name dispatch convention we'll implement in Tasks 8-10.
"
```

---

## Task 8: Generic runner — load xclbin and enumerate kernargs

**Files:**
- Modify: `bridge-runner/bridge-trace-runner.cpp`

- [x] **Step 1: Add metadata enumeration to the runner**

In `bridge-runner/bridge-trace-runner.cpp`, add after the `CliArgs`
struct / before `main()`:

```cpp
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"

namespace {

struct KernArgInfo {
    std::string name;
    std::string host_type;   // e.g., "uint32_t*", "uint32_t"
    size_t index = 0;
    uint64_t size = 0;       // in bytes; for pointers, the buffer size
    uint64_t offset = 0;
    bool is_scalar() const { return host_type.find('*') == std::string::npos; }
};

std::vector<KernArgInfo> read_kernel_args(
    const xrt::xclbin& xclbin,
    const std::string& kernel_name_hint,
    std::string& chosen_kernel_name,
    bool verbose)
{
    auto kernels = xclbin.get_kernels();
    if (kernels.empty()) throw std::runtime_error("xclbin has no kernels");
    const xrt::xclbin::kernel* picked = nullptr;
    for (const auto& k : kernels) {
        if (kernel_name_hint.empty() ||
            k.get_name().find(kernel_name_hint) != std::string::npos) {
            picked = &k;
            break;
        }
    }
    if (!picked) {
        throw std::runtime_error("no kernel matches --kernel hint: " + kernel_name_hint);
    }
    chosen_kernel_name = picked->get_name();
    std::vector<KernArgInfo> out;
    for (const auto& a : picked->get_args()) {
        KernArgInfo k;
        k.name = a.get_name();
        k.host_type = a.get_host_type();
        k.index = a.get_index();
        k.size = a.get_size();
        k.offset = a.get_offset();
        out.push_back(k);
        if (verbose) {
            std::fprintf(stderr,
                "  arg[%zu] name=%-12s host_type=%-16s size=%lu offset=%lu\n",
                k.index, k.name.c_str(), k.host_type.c_str(),
                (unsigned long)k.size, (unsigned long)k.offset);
        }
    }
    return out;
}

} // anonymous namespace
```

And in `main()`, replace the single-line debug print with:

```cpp
    xrt::device device(0);
    xrt::xclbin xclbin(args.xclbin);
    std::string kernel_name;
    auto kargs = read_kernel_args(xclbin, args.kernel_name, kernel_name, /*verbose=*/true);
    std::fprintf(stderr, "bridge-trace-runner: kernel=%s, %zu args\n",
                 kernel_name.c_str(), kargs.size());
    // Allocation + execution lands in Tasks 9-10.
    return 0;
```

- [x] **Step 2: Build**

```bash
cd /home/triple/npu-work/xdna-emu
cmake --build bridge-runner/build
```

Expected: builds cleanly.

- [x] **Step 3: Smoke-test against a real xclbin**

Find a recently-built traced xclbin (from Task 6's integration test output or from a fresh run of the bridge test). If none exists, run:

```bash
cd /home/triple/npu-work/xdna-emu
source /home/triple/npu-work/toolchain-build/activate-npu-env.sh
mkdir -p /tmp/claude-1000/trace-runner-test
PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python \
  python3 tools/mlir-trace-inject.py \
  --input /home/triple/npu-work/mlir-aie/test/npu-xrt/add_one/aie.mlir \
  --out /tmp/claude-1000/trace-runner-test/aie-traced.mlir
cd /tmp/claude-1000/trace-runner-test
aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
  --xclbin-name=aie-traced.xclbin --npu-insts-name=insts.bin aie-traced.mlir
```

Then run:

```bash
/home/triple/npu-work/xdna-emu/bridge-runner/build/bridge-trace-runner \
  --xclbin /tmp/claude-1000/trace-runner-test/aie-traced.xclbin \
  --instr /tmp/claude-1000/trace-runner-test/insts.bin \
  --trace-out /tmp/claude-1000/trace-runner-test/trace.bin \
  -v
```

Expected output shows a list of kernargs with names matching the mlir-aie
convention (opcode, instr, instr_size, in*, out*, trace, and possibly
ctrlpkts). Exits 0.

- [x] **Step 4: Commit**

```bash
git add bridge-runner/bridge-trace-runner.cpp
git commit -m "feat(bridge-runner): enumerate xclbin kernel args

Reads xrt::xclbin::kernel::get_args() and prints name/type/size for each.
Allocation + execution lands in next tasks.
"
```

---

## Task 9: Generic runner — allocate and bind buffers by name

**Files:**
- Modify: `bridge-runner/bridge-trace-runner.cpp`

- [x] **Step 1: Add buffer allocation + binding logic**

Expand `main()` to allocate BOs based on arg name. Insert between the
`kargs` enumeration and the `return 0;` line:

```cpp
    device.register_xclbin(xclbin);
    xrt::hw_context context(device, xclbin.get_uuid());
    xrt::kernel kernel(context, kernel_name);

    // Load the instruction binary.
    std::ifstream instr_file(args.instr_bin, std::ios::binary | std::ios::ate);
    if (!instr_file) throw std::runtime_error("cannot open instr file: " + args.instr_bin);
    size_t instr_bytes = instr_file.tellg();
    instr_file.seekg(0);
    std::vector<uint32_t> instr_words(instr_bytes / sizeof(uint32_t));
    instr_file.read(reinterpret_cast<char*>(instr_words.data()), instr_bytes);
    if (args.verbose) std::fprintf(stderr, "  loaded %zu instruction words\n", instr_words.size());

    // Per-arg binding state.
    std::vector<xrt::bo> bos;                // stable storage for BOs
    std::vector<std::variant<uint32_t, xrt::bo*>> run_args_by_index;
    run_args_by_index.resize(kargs.size(), std::monostate{});  // will fill in order

    // We'll fill run_args_by_index[k.index] for each k, then pass in index order.
    // However xrt::run accepts positional args; we build a positional vector matching
    // the xclbin's kernarg order.

    size_t input_idx = 0;
    size_t output_idx = 0;
    xrt::bo* trace_bo_ptr = nullptr;

    for (const auto& k : kargs) {
        if (k.name == "opcode") {
            run_args_by_index[k.index] = uint32_t{3};  // AIE kernel opcode
        } else if (k.name == "instr") {
            auto bo = xrt::bo(device, instr_words.size() * sizeof(uint32_t),
                              XCL_BO_FLAGS_CACHEABLE, kernel.group_id(k.index));
            std::memcpy(bo.map<void*>(), instr_words.data(),
                        instr_words.size() * sizeof(uint32_t));
            bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bos.push_back(std::move(bo));
            run_args_by_index[k.index] = &bos.back();
        } else if (k.name == "instr_size") {
            run_args_by_index[k.index] = static_cast<uint32_t>(instr_words.size());
        } else if (k.name == "trace") {
            auto bo = xrt::bo(device, args.trace_size_bytes,
                              XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(k.index));
            std::memset(bo.map<void*>(), 0, args.trace_size_bytes);
            bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bos.push_back(std::move(bo));
            trace_bo_ptr = &bos.back();
            run_args_by_index[k.index] = trace_bo_ptr;
        } else if (k.name == "ctrlpkts") {
            // Minimal 8-byte zeroed buffer.
            auto bo = xrt::bo(device, 8, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(k.index));
            std::memset(bo.map<void*>(), 0, 8);
            bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bos.push_back(std::move(bo));
            run_args_by_index[k.index] = &bos.back();
        } else if (k.name.rfind("in", 0) == 0) {
            // Input: read next --input file, or zeroes if none provided.
            auto bo = xrt::bo(device, k.size, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(k.index));
            if (input_idx < args.inputs.size()) {
                std::ifstream f(args.inputs[input_idx], std::ios::binary);
                f.read(static_cast<char*>(bo.map<void*>()), k.size);
            } else {
                std::memset(bo.map<void*>(), 0, k.size);
            }
            input_idx++;
            bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bos.push_back(std::move(bo));
            run_args_by_index[k.index] = &bos.back();
        } else if (k.name.rfind("out", 0) == 0) {
            auto bo = xrt::bo(device, k.size, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(k.index));
            std::memset(bo.map<void*>(), 0, k.size);
            bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bos.push_back(std::move(bo));
            run_args_by_index[k.index] = &bos.back();
            output_idx++;
        } else {
            throw std::runtime_error("unknown kernarg name: " + k.name);
        }
    }
```

Note: `std::variant<uint32_t, xrt::bo*>` with `std::monostate` requires `#include <variant>`. Add it with the other includes.

- [x] **Step 2: Build**

```bash
cd /home/triple/npu-work/xdna-emu
cmake --build bridge-runner/build
```

Expected: builds cleanly.

- [x] **Step 3: Run smoke test (allocation only, no kernel run yet)**

Using the traced xclbin from Task 8 Step 3:

```bash
/home/triple/npu-work/xdna-emu/bridge-runner/build/bridge-trace-runner \
  --xclbin /tmp/claude-1000/trace-runner-test/aie-traced.xclbin \
  --instr /tmp/claude-1000/trace-runner-test/insts.bin \
  --trace-out /tmp/claude-1000/trace-runner-test/trace.bin \
  -v
```

Expected: prints arg list, "loaded N instruction words", exits 0. No crashes.

- [x] **Step 4: Commit**

```bash
git add bridge-runner/bridge-trace-runner.cpp
git commit -m "feat(bridge-runner): allocate and bind BOs by kernarg name

For each kernarg discovered in the xclbin metadata, allocate the
appropriate buffer: instr from file, instr_size as scalar, inputs
from --input files (or zero-filled), outputs zeroed, trace zeroed
at the requested --trace-size, ctrlpkts minimal stub. Unknown arg
names throw.
"
```

---

## Task 10: Generic runner — launch kernel and dump trace

**Files:**
- Modify: `bridge-runner/bridge-trace-runner.cpp`

- [x] **Step 1: Add kernel launch + trace sync-back**

After the kernarg-binding loop in `main()`, before `return 0;`:

```cpp
    // Build the positional arg vector and launch.
    xrt::run run(kernel);
    for (size_t i = 0; i < run_args_by_index.size(); ++i) {
        std::visit([&](auto&& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, uint32_t>) {
                run.set_arg(static_cast<int>(i), v);
            } else if constexpr (std::is_same_v<T, xrt::bo*>) {
                run.set_arg(static_cast<int>(i), *v);
            } else {
                throw std::runtime_error(
                    "kernarg " + std::to_string(i) + " was not bound");
            }
        }, run_args_by_index[i]);
    }
    if (args.verbose) std::fprintf(stderr, "  launching kernel\n");
    run.start();
    auto state = run.wait(std::chrono::seconds(30));
    if (state != ERT_CMD_STATE_COMPLETED) {
        std::fprintf(stderr, "error: kernel did not complete (state=%d)\n", (int)state);
        return 1;
    }

    // Sync the trace buffer back and write to --trace-out.
    if (trace_bo_ptr) {
        trace_bo_ptr->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::ofstream out(args.trace_out, std::ios::binary);
        if (!out) throw std::runtime_error("cannot open trace-out: " + args.trace_out);
        out.write(static_cast<const char*>(trace_bo_ptr->map<const void*>()),
                  args.trace_size_bytes);
        out.close();
        if (args.verbose)
            std::fprintf(stderr, "  wrote %lu bytes of trace to %s\n",
                         (unsigned long)args.trace_size_bytes, args.trace_out.c_str());
    } else {
        std::fprintf(stderr, "warning: xclbin has no 'trace' kernarg; "
                             "nothing written to --trace-out\n");
    }

    // Optionally write outputs if --output paths were provided.
    size_t out_file_idx = 0;
    for (const auto& k : kargs) {
        if (k.name.rfind("out", 0) != 0) continue;
        if (out_file_idx >= args.outputs.size()) { out_file_idx++; continue; }
        // Find the matching bo -- we added them in the same order kargs were iterated.
        // For simplicity, re-scan bos by index offset. Better: track the mapping.
        // Here we just rely on kargs/bos ordering: skip opcode/scalar args.
        // To keep things robust, fetch via the variant again:
        auto& v = run_args_by_index[k.index];
        if (auto* pp = std::get_if<xrt::bo*>(&v); pp) {
            (*pp)->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            std::ofstream of(args.outputs[out_file_idx], std::ios::binary);
            of.write(static_cast<const char*>((*pp)->map<const void*>()), k.size);
        }
        out_file_idx++;
    }
```

Add `#include <chrono>` and `#include "xrt/xrt_bo.h"` if not already present.

- [x] **Step 2: Build**

```bash
cd /home/triple/npu-work/xdna-emu
cmake --build bridge-runner/build
```

Expected: builds cleanly.

- [x] **Step 3: Run on real hardware**

Using the traced xclbin from earlier, run against the actual NPU:

```bash
/home/triple/npu-work/xdna-emu/bridge-runner/build/bridge-trace-runner \
  --xclbin /tmp/claude-1000/trace-runner-test/aie-traced.xclbin \
  --instr /tmp/claude-1000/trace-runner-test/insts.bin \
  --trace-out /tmp/claude-1000/trace-runner-test/trace.bin \
  --trace-size 8192 \
  -v
```

Expected: runs without error, writes 8192 bytes to `trace.bin`. Verify
the trace buffer is non-zero:

```bash
xxd /tmp/claude-1000/trace-runner-test/trace.bin | head -5
```

Expected: at least some non-zero bytes (trace packet headers + event
timestamps). If it's all zeros, trace wasn't activated — debug via
`RUST_LOG=info` XRT dmesg output and check that `AIEInsertTraceFlows`
actually emitted the register writes.

- [x] **Step 4: Commit**

```bash
git add bridge-runner/bridge-trace-runner.cpp
git commit -m "feat(bridge-runner): launch kernel + dump trace buffer

Issues xrt::run with the positional arg vector built from kernarg
metadata, waits up to 30s, syncs the trace buffer back, and writes
it to --trace-out. Outputs optionally written to --output paths.
"
```

---

## Task 11: Cycle extractor — `tools/trace-to-cycles.py`

**Files:**
- Create: `tools/trace-to-cycles.py`
- Create: `tools/tests/test_trace_to_cycles.py`
- Create: `tools/tests/fixtures/sample_traced_trace.json` (canned Perfetto JSON)

- [x] **Step 1: Create the canned Perfetto fixture**

Create `tools/tests/fixtures/sample_traced_trace.json`:

```json
[
  {"name": "process_name", "ph": "M", "pid": 0, "args": {"name": "tile_0_2_core"}},
  {"name": "INSTR_EVENT_0", "ph": "B", "ts": 100, "pid": 0, "tid": 0},
  {"name": "INSTR_VECTOR", "ph": "B", "ts": 105, "pid": 0, "tid": 0},
  {"name": "INSTR_VECTOR", "ph": "E", "ts": 150, "pid": 0, "tid": 0},
  {"name": "INSTR_VECTOR", "ph": "B", "ts": 160, "pid": 0, "tid": 0},
  {"name": "INSTR_VECTOR", "ph": "E", "ts": 500, "pid": 0, "tid": 0},
  {"name": "INSTR_EVENT_1", "ph": "B", "ts": 510, "pid": 0, "tid": 0}
]
```

Expected cycle count from this fixture: `510 - 100 = 410`.

- [x] **Step 2: Write the failing test**

Create `tools/tests/test_trace_to_cycles.py`:

```python
"""Tests for tools/trace-to-cycles.py."""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXTRACTOR = REPO / "tools" / "trace-to-cycles.py"
FIXTURES = REPO / "tools" / "tests" / "fixtures"
TRACE_JSON = FIXTURES / "sample_traced_trace.json"


def _run(args, check=True):
    return subprocess.run(
        ["python3", str(EXTRACTOR), *args],
        capture_output=True, text=True, check=check,
    )


def test_extractor_help():
    r = _run(["--help"])
    assert "usage" in r.stdout.lower() or "usage" in r.stderr.lower()


def test_extractor_reads_json_and_emits_cycles(tmp_path):
    """Given a canned Perfetto JSON, extractor should emit one integer line."""
    out = tmp_path / "cycles.txt"
    r = _run(["--trace-json", str(TRACE_JSON), "--out", str(out)])
    assert r.returncode == 0, f"stderr={r.stderr}"
    content = out.read_text().strip()
    assert content == "410", f"expected '410', got {content!r}"
```

- [x] **Step 3: Run the test to verify it fails**

Run: `cd /home/triple/npu-work/xdna-emu && pytest tools/tests/test_trace_to_cycles.py -v`

Expected: FAIL — extractor script doesn't exist.

- [x] **Step 4: Write the extractor**

Create `tools/trace-to-cycles.py`:

```python
#!/usr/bin/env python3
"""
trace-to-cycles.py -- Extract per-test HW cycle counts from a trace buffer.

Two input modes:

  --trace-bin   : raw bytes from bridge-trace-runner's --trace-out, paired
                  with --xclbin-mlir to locate the MLIR spec that describes
                  the trace layout. Invokes mlir-aie's parse_trace() to
                  produce Perfetto JSON, then computes cycle delta from
                  first-to-last event timestamp on the primary compute tile.

  --trace-json  : already-parsed Perfetto JSON (from parse_trace or
                  compatible). Same cycle-delta computation. Used in unit
                  tests with canned fixtures, and whenever the caller has
                  already parsed.

Output: a single line with an integer cycle count, written to --out.
"""
import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace-bin", help="raw trace bytes from bridge-trace-runner")
    src.add_argument("--trace-json", help="pre-parsed Perfetto JSON file")
    p.add_argument("--xclbin-mlir", help="MLIR used to build the xclbin (required with --trace-bin)")
    p.add_argument("--out", required=True, help="output path for single-integer cycle count")
    return p.parse_args()


def cycles_from_trace_json(events):
    """Return max(ts) - min(ts) across all non-metadata events."""
    ts_values = [e["ts"] for e in events
                 if isinstance(e, dict) and "ts" in e and e.get("ph") in ("B", "E", "X", "i")]
    if not ts_values:
        raise ValueError("trace JSON has no event timestamps")
    return max(ts_values) - min(ts_values)


def main():
    args = parse_args()
    if args.trace_json:
        events = json.loads(Path(args.trace_json).read_text())
    else:
        if not args.xclbin_mlir:
            print("error: --trace-bin requires --xclbin-mlir", file=sys.stderr)
            return 1
        # Lazy import -- only needed for the --trace-bin path.
        try:
            import numpy as np
            from aie.utils.trace.parse import parse_trace
        except ImportError as e:
            print(f"error: mlir-aie trace module not importable: {e}\n"
                  f"  ensure PYTHONPATH includes mlir-aie/install/python",
                  file=sys.stderr)
            return 1
        raw = np.fromfile(args.trace_bin, dtype=np.uint32)
        mlir_text = Path(args.xclbin_mlir).read_text()
        events = parse_trace(raw, mlir_text)

    cycles = cycles_from_trace_json(events)
    Path(args.out).write_text(f"{cycles}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
```

- [x] **Step 5: Make executable and run the test**

```bash
chmod +x /home/triple/npu-work/xdna-emu/tools/trace-to-cycles.py
cd /home/triple/npu-work/xdna-emu
pytest tools/tests/test_trace_to_cycles.py -v
```

Expected: both tests PASS.

- [x] **Step 6: Commit**

```bash
git add tools/trace-to-cycles.py tools/tests/test_trace_to_cycles.py tools/tests/fixtures/sample_traced_trace.json
git commit -m "feat(trace): add cycle extractor for trace buffers

tools/trace-to-cycles.py reads either raw trace bytes (via mlir-aie
parse_trace) or pre-parsed Perfetto JSON and emits a single integer
cycle count to the --out path. Cycle count is derived from
max_ts - min_ts across all timestamped events.
"
```

---

## Task 12: Bridge script — `--with-hw-cycles` flag scaffolding

**Files:**
- Modify: `scripts/emu-bridge-test.sh`

- [x] **Step 1: Add the flag handler**

Open `scripts/emu-bridge-test.sh`. Find the argument parsing section
(look for the existing `--no-hw` or `--no-timeout` flag handling). Add
a new flag:

```bash
# In the flag parser:
    --with-hw-cycles)
        WITH_HW_CYCLES=true
        shift
        ;;
```

In the defaults section at the top of the script (where `NO_TIMEOUT=false`,
`NO_HW=false`, etc. live), add:

```bash
WITH_HW_CYCLES=${WITH_HW_CYCLES:-false}
export WITH_HW_CYCLES
```

In the usage/help block, add:

```
  --with-hw-cycles     Run the trace-based HW cycle capture pipeline for
                       each HW test; emits cycles.HW.<test>.txt beside
                       the HW result.
```

- [x] **Step 2: Add the pipeline helper (stub)**

Near the other `_` helper functions, add:

```bash
# _run_hw_cycles_pipeline <test_dir> <xclbin> <kernel> <instr>
# Runs trace inject -> aiecc (traced) -> bridge-trace-runner -> trace-to-cycles.
# Output file: $test_dir/cycles.HW.<variant>.txt
_run_hw_cycles_pipeline() {
    local test_dir="$1"
    local xclbin="$2"
    local kernel="$3"
    local instr="$4"
    # Implementation lands in Task 13.
    echo "_run_hw_cycles_pipeline: stub (test_dir=$test_dir)" >&2
    return 0
}
```

- [x] **Step 3: Verify the flag parses**

```bash
cd /home/triple/npu-work/xdna-emu
./scripts/emu-bridge-test.sh --help 2>&1 | grep -i with-hw-cycles
./scripts/emu-bridge-test.sh --with-hw-cycles --no-hw --peano-only -v add_one 2>&1 | tail -20
```

Expected: `--help` lists the new flag; running with `--with-hw-cycles` and
`--no-hw` (note: the `--no-hw` path doesn't run cycles pipeline yet, so this
is just a flag-parse smoke test) exits cleanly.

- [x] **Step 4: Commit**

```bash
git add scripts/emu-bridge-test.sh
git commit -m "feat(bridge): add --with-hw-cycles flag scaffolding

Parses the flag and exports WITH_HW_CYCLES. Stub helper
_run_hw_cycles_pipeline is wired into the script ready for the
full pipeline in Task 13.
"
```

---

## Task 13: Bridge script — wire in full pipeline

**Files:**
- Modify: `scripts/emu-bridge-test.sh`

- [x] **Step 1: Replace the pipeline stub with the real thing**

In `scripts/emu-bridge-test.sh`, replace the body of `_run_hw_cycles_pipeline`:

```bash
_run_hw_cycles_pipeline() {
    local test_dir="$1"
    local xclbin="$2"
    local kernel="$3"
    local instr="$4"
    local variant="$5"   # chess | peano
    local test_name
    test_name="$(basename "$test_dir")"

    # Only run if the xclbin has a 'trace' kernarg. Otherwise skip silently.
    if ! xclbinutil --info --input "$xclbin" 2>/dev/null | grep -q "trace"; then
        echo "[hw-cycles] $test_name ($variant): no trace kernarg; skipping" >&2
        return 0
    fi

    local work_dir="$test_dir/hw-cycles"
    mkdir -p "$work_dir"

    # Path to the MLIR used to build this xclbin. By convention it sits
    # next to the xclbin as aie.mlir. If not present we can't run parse_trace.
    local mlir_path="$test_dir/aie.mlir"
    if [[ ! -f "$mlir_path" ]]; then
        # Look in typical mlir-aie build locations:
        mlir_path="$(dirname "$xclbin")/aie.mlir"
    fi

    local trace_bin="$work_dir/trace.$variant.bin"
    local cycles_txt="$test_dir/cycles.HW.$variant.txt"

    # Step 1: invoke the runner.
    if ! /home/triple/npu-work/xdna-emu/bridge-runner/build/bridge-trace-runner \
        --xclbin "$xclbin" \
        --kernel "$kernel" \
        --instr "$instr" \
        --trace-out "$trace_bin" \
        --trace-size 8192 \
        2>"$work_dir/runner.$variant.log"; then
        echo "[hw-cycles] $test_name ($variant): runner failed; see $work_dir/runner.$variant.log" >&2
        return 1
    fi

    # Step 2: extract cycles.
    if [[ -f "$mlir_path" ]]; then
        PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python \
            python3 /home/triple/npu-work/xdna-emu/tools/trace-to-cycles.py \
            --trace-bin "$trace_bin" \
            --xclbin-mlir "$mlir_path" \
            --out "$cycles_txt" 2>"$work_dir/extract.$variant.log" || {
            echo "[hw-cycles] $test_name ($variant): extractor failed; see $work_dir/extract.$variant.log" >&2
            return 1
        }
        echo "[hw-cycles] $test_name ($variant): cycles=$(cat "$cycles_txt")" >&2
    else
        echo "[hw-cycles] $test_name ($variant): no MLIR found at $mlir_path; cycles not extracted" >&2
        return 1
    fi
    return 0
}
```

- [x] **Step 2: Add the pre-compile injector step**

Find the compile phase (where `aiecc.py` is invoked per test). Before
each `aiecc.py` invocation, when `WITH_HW_CYCLES=true` and we're
compiling for HW, inject trace ops into the MLIR.

This is the riskiest part of the plan — the exact location depends on
how `emu-bridge-test.sh` currently structures compile. Read the script
around the `aiecc.py` invocation to understand the flow. The pattern to
apply:

```bash
# Before the aiecc.py call:
if [[ "$WITH_HW_CYCLES" == "true" ]]; then
    local traced_mlir="$work_dir/aie-traced.mlir"
    PYTHONPATH=/home/triple/npu-work/mlir-aie/install/python \
        python3 /home/triple/npu-work/xdna-emu/tools/mlir-trace-inject.py \
        --input "$mlir_path" \
        --out "$traced_mlir" \
        --buffer-size 8192 \
        2>"$work_dir/inject.log"
    local inject_rc=$?
    if [[ $inject_rc -eq 0 ]]; then
        mlir_path="$traced_mlir"
    elif [[ $inject_rc -eq 2 ]]; then
        echo "[hw-cycles] already traced; using original MLIR" >&2
    else
        echo "[hw-cycles] injection failed; falling back to untraced" >&2
    fi
fi
```

Place this where `aiecc.py`'s input MLIR is chosen. Usually the script
has a local like `local src_mlir="..."`; you want to reassign `src_mlir`
to `traced_mlir` on successful injection.

- [x] **Step 3: Call `_run_hw_cycles_pipeline` after each HW test**

Find where the HW test is launched (likely near a `timeout` or `run.wait`
equivalent in the run phase). Immediately after the HW result is determined
and the `result` variable is set, if `WITH_HW_CYCLES=true` and the result
is PASS, call:

```bash
if [[ "$WITH_HW_CYCLES" == "true" && "$result" == "PASS" ]]; then
    _run_hw_cycles_pipeline "$test_dir" "$xclbin" "$kernel" "$instr" "$variant" || true
fi
```

The `|| true` ensures a cycles-pipeline failure doesn't mark the test
as failed — we treat cycle capture as best-effort.

- [x] **Step 4: Dry-run against a single test**

```bash
cd /home/triple/npu-work/xdna-emu
./scripts/emu-bridge-test.sh --with-hw-cycles --no-timeout --peano-only -v add_one 2>&1 | tail -30
```

Expected: test runs HW (on the real NPU), then prints a
`[hw-cycles] add_one (peano): cycles=NNN` line. Verify the
`cycles.HW.peano.txt` file exists in the test's output directory with
a single integer inside.

- [x] **Step 5: Commit**

```bash
git add scripts/emu-bridge-test.sh
git commit -m "feat(bridge): wire hw-cycles pipeline into emu-bridge-test.sh

--with-hw-cycles now runs the full pipeline per test:
  1. mlir-trace-inject rewrites the MLIR with trace ops
  2. aiecc compiles the traced MLIR (standard path)
  3. bridge-trace-runner executes on real HW, dumps trace buffer
  4. trace-to-cycles parses the trace and writes cycles.HW.<variant>.txt

Cycles capture failures are non-fatal (|| true); they log warnings but
don't mark the test as failed.
"
```

---

## Task 14: End-to-end validation on 10 bridge tests

**Files:**
- Create: `docs/superpowers/plans/2026-04-22-phase-b-trace-cycle-capture-validation.md`

- [x] **Step 1: Pick 10 representative bridge tests**

Skim `mlir-aie/test/npu-xrt/` and pick 10 tests spanning:
- Simple tests (add_one, add_256)
- Multi-core tests (tests with more than one compute tile)
- Tests with ctrlpkts
- Tests with memtiles

Record the list in the validation doc.

- [x] **Step 2: Run the pipeline across the selected tests**

```bash
cd /home/triple/npu-work/xdna-emu
TESTS="add_one add_256 cascade_flows column_specific add_blockwrite memtile_read_write lock_test ctrl_packet_reconfig matmul_small one_core_with_ctrl"
FILTER_RE="$(echo "$TESTS" | tr ' ' '|')"
./scripts/emu-bridge-test.sh --with-hw-cycles --no-timeout --peano-only -v "($FILTER_RE)" 2>&1 \
  | tee /tmp/claude-1000/phase-b-validation.log
```

(Adjust `FILTER_RE` if your bridge script's filter syntax differs.)

Expected: each test completes HW with a `cycles.HW.peano.txt` file
beside the normal result file.

- [x] **Step 3: Collect cycle counts and sanity-check**

```bash
cd /home/triple/npu-work/xdna-emu
find build/bridge-test-results/latest -name "cycles.HW.*.txt" -print -exec cat {} \;
```

Sanity checks to apply (these become the validation doc's verdict):
- Every selected test produced a `cycles.HW.*.txt` file.
- Cycle counts are non-zero (> 100) and not absurdly large (< 10^8).
- Tests with more compute map to more cycles (e.g., matmul_small > add_one).

If any test is missing a cycles file, investigate the `inject.log` and
`runner.*.log` for that test before moving on.

- [x] **Step 4: Write the validation doc**

Create `docs/superpowers/plans/2026-04-22-phase-b-trace-cycle-capture-validation.md`
with a results table:

```markdown
# Phase B Validation Results (2026-04-22)

| Test | HW Cycles | Injection | Runner | Extraction | Notes |
|------|-----------|-----------|--------|------------|-------|
| add_one | ... | OK | OK | OK | |
| ... | | | | | |

## Issues found
- ...

## Verdict
OK / needs fixes before Phase C can proceed.
```

Fill it in from the actual results.

- [x] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-04-22-phase-b-trace-cycle-capture-validation.md
git commit -m "docs(phase-b): validation results across 10 bridge tests

Confirms the trace-based cycle capture pipeline works end-to-end on
representative tests and produces plausible cycle counts. See doc
for the full results table and any follow-ups.
"
```

---

## Task 15: Phase B completion gate

**Files:**
- Modify: `NEXT-STEPS.md`
- Modify: `docs/superpowers/plans/2026-04-22-cycle-budget-testing.md`

- [x] **Step 1: Update NEXT-STEPS.md**

Update the Phase B section to reflect completion:

- Mark the Phase B task block complete.
- Add a pointer to the validation doc.
- Note which tests have cycles files available, so Phase C knows where to read from.

- [x] **Step 2: Update the original cycle-budget plan**

In `docs/superpowers/plans/2026-04-22-cycle-budget-testing.md`, update the
Phase B section to reference the new plan and its outcome. The original
Phase B pivot note should now also link to the validation doc.

- [x] **Step 3: Surface Phase C readiness**

Phase C tasks (#82-#85 in the task tracker) become unblocked. They can
now assume that `cycles.HW.<test>.<variant>.txt` files exist under
`build/bridge-test-results/<date>/<test>/` for any test run with
`--with-hw-cycles`.

Decide: do Phase D.3 (HW spot-check, task #81) and Phase C tasks get
bundled into one more plan, or executed individually from the existing
cycle-budget plan? Note the decision in `NEXT-STEPS.md`.

- [x] **Step 4: Commit**

```bash
git add NEXT-STEPS.md docs/superpowers/plans/2026-04-22-cycle-budget-testing.md
git commit -m "docs(phase-b): mark Phase B complete, unblock Phase C

Phase B pipeline (mlir-trace-inject + bridge-trace-runner +
trace-to-cycles) is validated across 10 bridge tests. Phase C
tasks can now consume cycles.HW.<test>.<variant>.txt files
produced by --with-hw-cycles runs.
"
```

---

## Self-Review

**Spec coverage against the architecture summary:**

- ✅ MLIR trace injector: Tasks 2-6 cover scaffolding, idempotency, compute-tile injection, runtime-sequence injection, end-to-end compile.
- ✅ Generic metadata-driven runner: Tasks 7-10 cover scaffolding, metadata read, allocation by name, launch + trace dump.
- ✅ Cycle extractor: Task 11 covers fixture, test, implementation.
- ✅ Bridge script integration: Tasks 12-13 cover flag + pipeline.
- ✅ Deprecation of old tools: Task 1.
- ✅ End-to-end validation: Task 14.
- ✅ Phase gate / handoff: Task 15.

**Placeholder scan:** `TODO`/`TBD`/`implement later`: None. Lookup/adapt notes are present where the exact Python binding API isn't fully nailed down — those are flagged as research steps within the task, not handoffs to the reader's imagination.

**Type consistency:** `bridge-trace-runner` CLI flags are stable across tasks (`--xclbin`, `--instr`, `--trace-out`, `--trace-size`, `--kernel`, `--input`, `--output`, `-v`). `mlir-trace-inject.py` CLI is stable (`--input`, `--out`, `--buffer-size`, `--no-op`). `trace-to-cycles.py` CLI is stable (`--trace-bin` XOR `--trace-json`, `--xclbin-mlir`, `--out`). Output file naming is consistent: `cycles.HW.<variant>.txt`.

**Notable risks called out in the plan:**

1. **mlir-aie Python bindings for trace ops may not expose direct op constructors** — Task 2 Step 1 probes first; Tasks 4-5 give both Path A (direct constructors) and Path B (text splice) to handle either world.
2. **MLIR injection insertion points are subtle** — Tasks 4-5 call out op-ordering constraints (e.g., trace decls must go before `aie.core` bodies); Task 6 catches mis-orderings via real aiecc compile.
3. **Bridge-script compile-phase location for injector hook is script-specific** — Task 13 Step 2 explicitly tells the implementer to read the script first rather than blindly applying a patch.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-22-phase-b-trace-cycle-capture.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, two-stage review (spec compliance then code quality) between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session with checkpoints for user review.

Which approach?
