# debug_halt §8 Close-out Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the three desk-actionable §8 forward-commitments (Core_Status RESET-bit fidelity, OUTBUF_ADDR + TRAP_PC unified derivation, G1 bounded-escalation bookkeeping) while leaving the two HW-budget-gated commitments explicitly OPEN.

**Architecture:** Item 3 unifies the runtime core-enable path with the CDO `Core_Control=0x1` write path through `write_control()`, eliminating the RESET-bit divergence by construction. Item 2 adds a golden-record-preserving build-time derivation tool that re-derives `OUTBUF_ADDR` (ELF symbol) and `TRAP_PC` (structured disasm scrape) every build and fails loudly on drift, without rewriting the committed Phase A artifact bytes. Item 1 is documentation bookkeeping.

**Tech Stack:** Rust (emulator + `cargo test --lib`), Python 3 (derivation tool + `unittest`), mlir-aie lit test harness (`run.lit`), `llvm-nm-aie` / `llvm-objdump-aie`.

**Spec:** `docs/superpowers/specs/2026-05-19-debug-halt-section8-closeout-design.md`

**Working branch:** `dev` (Phase B chain ends at `e3ebb39`; spec committed at `3ae53fe`). All work is additive on `dev`.

**Sandbox note:** Run Rust tests as `TMPDIR=/tmp/claude-1000 cargo test --lib` to avoid `/tmp` sandbox failures. Run bare (no `| tail`/`| grep`).

---

## Task 1: Item 3 — Core_Status RESET-bit reconciliation

Unify the runtime enable path with the CDO write path. After this task, enabling a core via `Coordinator::enable_core()` clears `reset`, so a debug-halted core reports `Core_Status = 0x10001` (matching silicon) instead of `0x10003`.

**Files:**
- Modify: `src/device/core_debug/mod.rs` (add `enable()` near `set_enabled` at line 602)
- Modify: `src/interpreter/engine/coordinator.rs:389` (`enable_core` body)
- Test: `src/device/core_debug/tests.rs` (control-register test block, after line 192)

**Context the engineer needs:**
- `CoreDebugState::write_control(value)` (`mod.rs:447`) sets `enabled = (value & CTRL_ENABLE_MASK)!=0` and `reset = (value & CTRL_RESET_MASK)!=0`; the reset-clears-runtime-state block only runs `if self.reset`.
- `CTRL_ENABLE_MASK` (`mod.rs:122`) is module-private (`const CTRL_ENABLE_MASK: u32 = 1 << 0`).
- `read_status()` (`mod.rs:380`) sets bit 1 (`STATUS_RESET_LSB`) iff `self.reset`, bit 16 (`STATUS_DEBUG_HALT_LSB`) iff `self.halted`.
- `Coordinator::enable_core()` (`coordinator.rs:383-391`) currently does `core.enabled = true;` then `tile.core_debug.set_enabled(true);`.
- Existing test style: see `control_enable_sets_enabled` (`tests.rs:147`) — `CoreDebugState::new()`, `write_control`, `is_enabled()`, `is_reset()`.

- [ ] **Step 1: Write the failing unit test for `enable()`**

Append to `src/device/core_debug/tests.rs` after line 192 (the end of `control_roundtrip_read`'s body — place after its closing `}`):

```rust
#[test]
fn enable_method_matches_cdo_enable_write() {
    // The runtime enable path must be byte-identical to a CDO
    // Core_Control=0x1 write: enabled set, reset cleared (Core_Status
    // RESET-bit fidelity, §8 close-out).
    let mut state = CoreDebugState::new();
    assert!(state.is_reset(), "fresh state starts in reset");
    state.enable();
    assert!(state.is_enabled(), "enable() sets enabled");
    assert!(!state.is_reset(), "enable() clears reset");
    assert_eq!(state.read_control(), CTRL_ENABLE_MASK);
}

#[test]
fn enable_then_halt_reports_no_reset_bit() {
    // Debug-halted after the runtime enable path: Core_Status must be
    // DEBUG_HALT|ENABLE (0x10001), NOT DEBUG_HALT|RESET|ENABLE (0x10003).
    let mut state = CoreDebugState::new();
    state.enable();
    state.halted = true;
    let status = state.read_status();
    assert_eq!(status & (1 << STATUS_RESET_LSB), 0, "RESET bit must be clear");
    assert_ne!(status & (1 << STATUS_ENABLE_LSB), 0, "ENABLE bit set");
    assert_ne!(status & (1 << STATUS_DEBUG_HALT_LSB), 0, "DEBUG_HALT bit set");
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib enable_method_matches_cdo_enable_write enable_then_halt_reports_no_reset_bit`
Expected: FAIL — compilation error `no method named `enable` found for struct `CoreDebugState``.

- [ ] **Step 3: Add the `enable()` method**

In `src/device/core_debug/mod.rs`, immediately after `set_enabled` (the method whose body is `self.enabled = enabled;`, ending at line 604), insert:

```rust
    /// Enable the core via the same register semantics as a CDO
    /// `Core_Control = 0x1` write: sets `enabled`, clears `reset`.
    ///
    /// The runtime enable path (`Coordinator::enable_core`) routes
    /// through this so it cannot diverge from the CDO write path. Before
    /// this existed, the runtime path used `set_enabled(true)` (which
    /// never touched `reset`), leaving `reset` at its `true` default and
    /// making a debug-halted core report `Core_Status = 0x10003`
    /// instead of the silicon-correct `0x10001` (§8 close-out,
    /// 2026-05-19).
    pub fn enable(&mut self) {
        self.write_control(CTRL_ENABLE_MASK);
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib enable_method_matches_cdo_enable_write enable_then_halt_reports_no_reset_bit`
Expected: PASS (2 passed).

- [ ] **Step 5: Wire `enable_core` through `enable()`**

In `src/interpreter/engine/coordinator.rs`, in `enable_core` (line 383), replace the mirror line. Current body:

```rust
    pub fn enable_core(&mut self, col: usize, row: usize) {
        if let Some(core) = self.get_core_mut(col, row) {
            core.enabled = true;
        }
        // Mirror to CoreDebugState so Core_Status register shows enabled.
        if let Some(tile) = self.device.tile_mut(col, row) {
            tile.core_debug.set_enabled(true);
        }
    }
```

Change the mirror block to:

```rust
        // Mirror to CoreDebugState via the same register semantics as a
        // CDO Core_Control=0x1 write (sets enabled, clears reset) so the
        // runtime enable path cannot diverge from the CDO write path
        // (Core_Status RESET-bit fidelity, §8 close-out 2026-05-19).
        if let Some(tile) = self.device.tile_mut(col, row) {
            tile.core_debug.enable();
        }
```

`disable_core` is left unchanged (it stays on `set_enabled(false)`; disabling does not re-assert reset, and it is not the indicted path).

- [ ] **Step 6: Add the coordinator-level regression test**

Find the existing coordinator test module in `src/interpreter/engine/coordinator.rs` (search for `mod tests` / `#[cfg(test)]`). Add this test alongside the existing coordinator tests (use the same harness/imports the neighbouring tests use to construct a `Coordinator`; mirror an existing test that calls `enable_core`):

```rust
    #[test]
    fn enable_core_clears_core_debug_reset() {
        // §8 close-out: the runtime enable path must clear reset so a
        // halted core reports Core_Status 0x10001, not 0x10003.
        let mut coord = test_coordinator_single_core();
        coord.enable_core(0, 2);
        let tile = coord.device.tile_mut(0, 2).expect("tile (0,2)");
        assert!(!tile.core_debug.is_reset(), "enable_core must clear reset");
        assert!(tile.core_debug.is_enabled(), "enable_core sets enabled");
    }
```

If no `test_coordinator_single_core()` helper exists, use whatever constructor the nearest existing `enable_core`-touching test uses (e.g. the count-step guard test `count_step_budget_not_consumed_by_stall_cycles` added in `e3ebb39` — copy its coordinator setup). Do NOT invent a new harness.

- [ ] **Step 7: Run the coordinator test**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib enable_core_clears_core_debug_reset`
Expected: PASS.

- [ ] **Step 8: Audit and fix stale-expectation fallout**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib` (bare — never pipe test output through `tail`/`grep`; Claude Code clips long output automatically). Read the failure list from the clipped output.
For any newly-failing test: it enabled a core and then asserted the *old* RESET-set behaviour (`is_reset()==true` after enable, `read_control()`/`read_status()` with bit 1 set after the enable path). Correct the expectation to the new, intended behaviour (reset clear after enable). These corrections ARE the §8 fix landing — do not work around them. No test hardcodes the literal `0x10003`/`65539` (pre-verified), so this set is small or empty.

- [ ] **Step 9: Full library test sweep**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: all pass, 0 failed.

- [ ] **Step 10: Commit**

```bash
git add src/device/core_debug/mod.rs src/interpreter/engine/coordinator.rs src/device/core_debug/tests.rs
git commit -m "debug-halt §8 close-out Item 3: unify runtime enable path via write_control

enable_core() now routes core-debug enable through write_control()
(CoreDebugState::enable), making it byte-identical to a CDO
Core_Control=0x1 write: enabled set, reset cleared. A debug-halted
core now reports Core_Status 0x10001 (silicon-correct) instead of
0x10003. Closes the §8 Core_Status RESET-bit divergence.

Generated using Claude Code."
```

---

## Task 2: Item 2a — derivation tool + parser tests (no build wiring yet)

Build the `debug-halt-probe-derive.py` tool and its parser unit tests against a real captured disasm fixture. No `run.lit` / template changes yet — this task delivers a tested, standalone tool.

**Files:**
- Create: `tools/debug-halt-probe-derive.py`
- Create: `tools/test_debug_halt_probe_derive.py`
- Create: `tools/fixtures/debug_halt_probe_nm.txt`
- Create: `tools/fixtures/debug_halt_probe_objdump.txt`

**Context the engineer needs:**
- A compiled core ELF already exists at
  `../mlir-aie/build/test/npu-xrt/debug_halt_probe/chess/aie_arch.mlir.prj/main_core_0_2.elf`.
- `nm` on it yields the line `00070400 A output_buffer` (verified). The
  emulator’s `OUTBUF_ADDR = symbol_value & 0xFFFF` ⇒ `0x0400`.
- The trap bundle: `aie.mlir` documents
  `0x17a: ... movxm p0, #0x70400` (materialize the `output_buffer`
  base into pointer `p0`) and `0x184: st ..., [p0, #4] ...` (store the
  `0xBB` marker to `output_buffer[1]`). `TRAP_PC = 0x184`,
  `TRAP_PC14 = 0x184 & 0x3FFF = 0x184`,
  `PC_EVENT0_VALUE = 0x80000000 | 0x184 = 0x80000184`.
- The env activation puts `llvm-nm-aie` / `llvm-objdump-aie` on PATH.
- Existing Python tool-test pattern: `tools/test_inject_maskpoll.py`.

- [ ] **Step 1: Capture the real fixtures**

Run (bare; redirect, then trim with the Read tool — do not pipe through `head`/`tail`):

```bash
ELF=../mlir-aie/build/test/npu-xrt/debug_halt_probe/chess/aie_arch.mlir.prj/main_core_0_2.elf
mkdir -p tools/fixtures
(llvm-nm-aie "$ELF" 2>/dev/null || nm "$ELF") > tools/fixtures/debug_halt_probe_nm.txt
(llvm-objdump-aie -d "$ELF" 2>/dev/null || llvm-objdump -d "$ELF") > tools/fixtures/debug_halt_probe_objdump.txt
```

Then open both fixture files with the Read tool and confirm:
- `debug_halt_probe_nm.txt` contains a line ending `output_buffer` with hex value `00070400`.
- `debug_halt_probe_objdump.txt` contains the `movxm p0, #0x70400`-style materialization and a `st ... [p0, #4]` store. Note the EXACT objdump line format (address column, bundle separator) — the parser regexes in Step 3 must match what you actually see, not the paraphrase above.

Trim `debug_halt_probe_objdump.txt` to a focused window: keep from a few lines before the `#0x70400` materialization through a few lines after the `[p0, #4]` store (enough context for the parser; the full dump is large and not needed as a fixture).

- [ ] **Step 2: Write the failing parser tests**

Create `tools/test_debug_halt_probe_derive.py`:

```python
#!/usr/bin/env python3
"""Unit tests for debug-halt-probe-derive.py parsers (real-fixture oracle)."""
import importlib.util
import pathlib
import unittest

_HERE = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "dhpd", _HERE / "debug-halt-probe-derive.py")
dhpd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dhpd)


class ParseOutbufAddr(unittest.TestCase):
    def test_known_symbol(self):
        nm = (_HERE / "fixtures" / "debug_halt_probe_nm.txt").read_text()
        self.assertEqual(dhpd.parse_outbuf_addr(nm), 0x0400)

    def test_missing_symbol_is_hard_error(self):
        with self.assertRaises(dhpd.DeriveError):
            dhpd.parse_outbuf_addr("0000abcd A something_else\n")


class ParseTrapPc(unittest.TestCase):
    def test_known_bundle(self):
        objd = (_HERE / "fixtures" / "debug_halt_probe_objdump.txt").read_text()
        self.assertEqual(dhpd.parse_trap_pc(objd, outbuf_full=0x70400), 0x184)

    def test_unlocatable_bundle_is_hard_error(self):
        with self.assertRaises(dhpd.DeriveError):
            dhpd.parse_trap_pc("nothing relevant here\n", outbuf_full=0x70400)


class PcEvent0(unittest.TestCase):
    def test_formula(self):
        self.assertEqual(dhpd.pc_event0_value(0x184), 0x80000184)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python tools/test_debug_halt_probe_derive.py`
Expected: FAIL — `FileNotFoundError`/`ModuleNotFound` for `debug-halt-probe-derive.py` (not yet created).

- [ ] **Step 4: Implement the derivation tool**

Create `tools/debug-halt-probe-derive.py`. Implement `parse_outbuf_addr`, `parse_trap_pc`, `pc_event0_value`, plus the golden-record-preserving CLI. Finalize the two regexes against the EXACT format observed in the Step-1 fixtures (the patterns below are the starting point; adjust to the real dump):

```python
#!/usr/bin/env python3
"""Derive debug_halt_probe magic constants from the compiled core ELF.

OUTBUF_ADDR     = value of the `output_buffer` ELF symbol & 0xFFFF
TRAP_PC         = PC of the bundle storing the 0xBB marker to
                  output_buffer[1] (the `st ..., [p0, #4]` where p0 was
                  materialized with the output_buffer base address)
PC_EVENT0_VALUE = 0x80000000 | (TRAP_PC & 0x3FFF)

Golden-record-preserving: regenerate aie.mlir/test.cpp from .in
templates with the derived values, diff against the committed golden
files. Match -> emit the golden bytes verbatim, exit 0. Drift -> print
committed-vs-derived and exit 1 (fails the build loudly). A missing
symbol or unlocatable trap bundle is a hard error, exit 2 -- never emit
a default.
"""
import argparse
import re
import shutil
import subprocess
import sys


class DeriveError(Exception):
    """Symbol/bundle could not be located -- never emit a default."""


def _tool(preferred, fallback):
    return shutil.which(preferred) or shutil.which(fallback) or fallback


def run_nm(elf):
    return subprocess.run([_tool("llvm-nm-aie", "nm"), elf],
                          capture_output=True, text=True, check=True).stdout


def run_objdump(elf):
    return subprocess.run([_tool("llvm-objdump-aie", "llvm-objdump"),
                           "-d", elf],
                          capture_output=True, text=True, check=True).stdout


_NM_RE = re.compile(r"^\s*([0-9a-fA-F]+)\s+\S+\s+output_buffer\s*$",
                    re.MULTILINE)


def parse_outbuf_addr(nm_text):
    m = _NM_RE.search(nm_text)
    if not m:
        raise DeriveError("output_buffer symbol not found in nm output")
    return int(m.group(1), 16) & 0xFFFF


# Address column at start of a disasm line, e.g. "  184: ..." (ADJUST
# to the real llvm-objdump-aie format seen in the Step-1 fixture).
_ADDR_RE = re.compile(r"^\s*([0-9a-fA-F]+):")


def parse_trap_pc(objdump_text, outbuf_full):
    """PC of the bundle that stores to [p0,#4] where p0 == output_buffer base.

    Strategy (deterministic given the kernel):
      1. Find the line materializing `outbuf_full` into a pointer reg
         (e.g. `movxm pN, #0x70400`); capture the register name pN.
      2. Find the next line that is a store to `[pN, #4]`.
      3. Return that line's leading address column as TRAP_PC.
    Adjust regexes to the exact fixture format. Never guess: if either
    anchor is absent, raise DeriveError.
    """
    hexaddr = f"#0x{outbuf_full:x}"
    lines = objdump_text.splitlines()
    ptr_reg = None
    for i, ln in enumerate(lines):
        if ptr_reg is None:
            m = re.search(r"movxm\s+(p\d+)\s*,\s*" + re.escape(hexaddr), ln)
            if m:
                ptr_reg = m.group(1)
            continue
        if re.search(r"\bst\b[^;]*\[\s*" + re.escape(ptr_reg) + r"\s*,\s*#4\s*\]", ln):
            am = _ADDR_RE.search(ln)
            if not am:
                # store bundle's address may be on the bundle header line;
                # walk back to the nearest preceding address column.
                for back in range(i, -1, -1):
                    am = _ADDR_RE.search(lines[back])
                    if am:
                        break
            if not am:
                raise DeriveError("trap store found but no address column")
            return int(am.group(1), 16)
    raise DeriveError(
        "could not locate the [p0,#4] store of the output_buffer base")


def pc_event0_value(trap_pc):
    return 0x80000000 | (trap_pc & 0x3FFF)


def _render(template_path, subs):
    text = open(template_path).read()
    for token, value in subs.items():
        text = text.replace(token, value)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--elf", required=True,
                    help="compiled core ELF (scratch first-pass build)")
    ap.add_argument("--in-mlir", required=True)
    ap.add_argument("--in-cpp", required=True)
    ap.add_argument("--golden-mlir", required=True)
    ap.add_argument("--golden-cpp", required=True)
    ap.add_argument("--out-mlir", required=True)
    ap.add_argument("--out-cpp", required=True)
    args = ap.parse_args(argv)

    try:
        nm_text = run_nm(args.elf)
        outbuf = parse_outbuf_addr(nm_text)
        # Recover the full symbol value for the pointer-materialization match.
        full = int(_NM_RE.search(nm_text).group(1), 16)
        trap_pc = parse_trap_pc(run_objdump(args.elf), outbuf_full=full)
    except DeriveError as e:
        print(f"FATAL: debug_halt_probe derivation failed: {e}",
              file=sys.stderr)
        return 2

    pcev0 = pc_event0_value(trap_pc)
    subs_mlir = {"@PC_EVENT0_VALUE@": f"0x{pcev0:08x}",
                 "@TRAP_PC@": f"0x{trap_pc:x}"}
    subs_cpp = {"@OUTBUF_ADDR@": f"0x{outbuf:04x}"}

    gen_mlir = _render(args.in_mlir, subs_mlir)
    gen_cpp = _render(args.in_cpp, subs_cpp)
    golden_mlir = open(args.golden_mlir).read()
    golden_cpp = open(args.golden_cpp).read()

    drift = []
    if gen_mlir != golden_mlir:
        drift.append("aie.mlir")
    if gen_cpp != golden_cpp:
        drift.append("test.cpp")

    if drift:
        print("FATAL: debug_halt_probe constants have drifted from the "
              f"committed golden record ({', '.join(drift)}).",
              file=sys.stderr)
        print(f"  derived OUTBUF_ADDR=0x{outbuf:04x} "
              f"TRAP_PC=0x{trap_pc:x} PC_EVENT0=0x{pcev0:08x}",
              file=sys.stderr)
        print("  The probe is a permanent Phase A artifact. If the "
              "kernel/allocation change was intentional, commit the "
              "regenerated aie.mlir/test.cpp as the new golden record "
              "and re-run; otherwise revert the kernel change.",
              file=sys.stderr)
        return 1

    # Match: emit the committed golden bytes verbatim (Phase A bytes
    # preserved; G1/G2 integrity intact).
    open(args.out_mlir, "w").write(golden_mlir)
    open(args.out_cpp, "w").write(golden_cpp)
    print(f"debug_halt_probe constants verified: OUTBUF_ADDR=0x{outbuf:04x} "
          f"TRAP_PC=0x{trap_pc:x} PC_EVENT0=0x{pcev0:08x} (golden match)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the parser tests to verify they pass**

Run: `python tools/test_debug_halt_probe_derive.py`
Expected: PASS (`OK`, 5 tests). If `parse_trap_pc`/`parse_outbuf_addr` fail, the regexes do not match the real fixture format from Step 1 — adjust the regexes (not the expected values: `0x0400` / `0x184` / `0x80000184` are the silicon-derived oracle) until green.

- [ ] **Step 6: Commit**

```bash
git add tools/debug-halt-probe-derive.py tools/test_debug_halt_probe_derive.py tools/fixtures/debug_halt_probe_nm.txt tools/fixtures/debug_halt_probe_objdump.txt
git commit -m "debug-halt §8 close-out Item 2a: probe constant derivation tool

debug-halt-probe-derive.py derives OUTBUF_ADDR (output_buffer ELF
symbol & 0xFFFF) and TRAP_PC (structured llvm-objdump-aie scrape of the
[p0,#4] store of the output_buffer base) from the compiled core ELF,
with a missing symbol/bundle a hard error (never a default). Parser
unit tests pin the silicon-derived oracle (0x0400 / 0x184 / 0x80000184)
against real captured nm/objdump fixtures.

Generated using Claude Code."
```

---

## Task 3: Item 2b — templatize probe + single-pass guarded build in run.lit

> **Revised 2026-05-19 during execution (Maya-approved).** The original
> two-pass design (scratch `cp` + first `aiecc` + second `aiecc`) is
> incompatible with `emu-bridge-test.sh`, the probe's primary validation
> path: the bridge harness skips all `cp` RUN lines
> (`emu-bridge-test.sh:719`) and synthesizes its own single
> `aie_arch.mlir` (≈ lines 1522-1535), so `scratch_arch.mlir` is never
> created and the two-pass Pass-1 fails under the bridge. The
> single-pass design below compiles once and runs derive+guard against
> the produced ELF: functionally equivalent on a match, equally loud on
> drift, and compatible with both upstream `llvm-lit` and the bridge
> harness. See spec §2.5. **Steps 1-3 (templates + round-trip) were
> already completed correctly and the templates committed (mlir-aie
> `424ce198be`); this revision only changes Step 4 (run.lit) and Steps
> 5-6 — amend the existing mlir-aie commit.**

Wire the tool into the probe build: templatize the two golden files, add a single-compile → derive → guard flow to `run.lit`. The committed `aie.mlir`/`test.cpp` stay byte-unchanged as the golden record.

**Files:**
- Create: `../mlir-aie/test/npu-xrt/debug_halt_probe/aie.mlir.in`
- Create: `../mlir-aie/test/npu-xrt/debug_halt_probe/test.cpp.in`
- Modify: `../mlir-aie/test/npu-xrt/debug_halt_probe/run.lit`

**Context the engineer needs:**
- Current `run.lit` RUN lines (verbatim):
  ```
  // RUN: cp %S/aie.mlir aie_arch.mlir
  // RUN: %run_on_npu1% sed 's/NPUDEVICE/npu1_1col/g' -i aie_arch.mlir
  // RUN: %python aiecc.py --no-aiesim --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=basic-sequential --xclbin-name=aie.xclbin --npu-insts-name=insts.bin aie_arch.mlir
  // RUN: clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags
  // RUN: %run_on_npu1% ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin
  // RUN: %run_on_npu2% ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin
  ```
- `aie.mlir:160`: `aiex.npu.write32 {address = 0x32020 : ui32, column = 0 : i32, row = 2 : i32, value = 0x80000184 : ui32}`
- `test.cpp:54`: `static constexpr uint32_t OUTBUF_ADDR = 0x0400;`
- The probe dir is `mlir-aie/test/npu-xrt/debug_halt_probe/`; `xdna-emu` is a sibling of `mlir-aie` under `npu-work`. From the test source dir `%S`, the tool is at `%S/../../../../xdna-emu/tools/debug-halt-probe-derive.py` (four `..`: debug_halt_probe → npu-xrt → test → mlir-aie → npu-work). The implementer MUST verify the depth with `ls %S/../../../../xdna-emu/tools/debug-halt-probe-derive.py`-equivalent before finalizing.

- [ ] **Step 1: Create `aie.mlir.in` from the golden, tokenizing the PC_Event0 value**

```bash
cp ../mlir-aie/test/npu-xrt/debug_halt_probe/aie.mlir ../mlir-aie/test/npu-xrt/debug_halt_probe/aie.mlir.in
```

Then edit `aie.mlir.in` line 160 only: replace `value = 0x80000184 : ui32` with `value = @PC_EVENT0_VALUE@ : ui32`. Leave every comment (including the `TRAP_PC`/`OUTBUF_ADDR` derivation notes) untouched — comments are documentation, not substituted. Do NOT touch `aie.mlir` (golden).

- [ ] **Step 2: Create `test.cpp.in` from the golden, tokenizing OUTBUF_ADDR**

```bash
cp ../mlir-aie/test/npu-xrt/debug_halt_probe/test.cpp ../mlir-aie/test/npu-xrt/debug_halt_probe/test.cpp.in
```

Then edit `test.cpp.in` line 54 only: replace `static constexpr uint32_t OUTBUF_ADDR = 0x0400;` with `static constexpr uint32_t OUTBUF_ADDR = @OUTBUF_ADDR@;`. Leave all comments untouched. Do NOT touch `test.cpp` (golden).

- [ ] **Step 3: Verify the templates render back to the golden bytes**

Run (proves the round-trip the build will rely on):

```bash
python - <<'PY'
import subprocess, sys, pathlib
d = pathlib.Path("../mlir-aie/test/npu-xrt/debug_halt_probe")
rc = subprocess.run([sys.executable, "tools/debug-halt-probe-derive.py",
  "--elf", "../mlir-aie/build/test/npu-xrt/debug_halt_probe/chess/aie_arch.mlir.prj/main_core_0_2.elf",
  "--in-mlir", str(d/"aie.mlir.in"), "--in-cpp", str(d/"test.cpp.in"),
  "--golden-mlir", str(d/"aie.mlir"), "--golden-cpp", str(d/"test.cpp"),
  "--out-mlir", "/tmp/claude-1000/dhp_aie.mlir",
  "--out-cpp", "/tmp/claude-1000/dhp_test.cpp"]).returncode
print("rc=", rc)
sys.exit(rc)
PY
```

Expected: prints `... (golden match)` and `rc= 0`. If it reports drift, the template tokenization in Step 1/2 does not round-trip — fix the `.in` files (a stray byte difference) until it matches. (Output path uses `/tmp/claude-1000` — ephemeral, allowed.)

- [ ] **Step 4: Rewrite `run.lit` as a single-pass guarded build**

Replace the RUN block in `../mlir-aie/test/npu-xrt/debug_halt_probe/run.lit` with (keep the header comment / `REQUIRES: ryzen_ai` lines above it intact):

```
// Compile the committed golden once (aie_arch.mlir = golden aie.mlir
// with NPUDEVICE substituted; under the bridge harness aie_arch.mlir is
// synthesized from canonical aie.mlir, the cp line being skipped there).
// RUN: cp %S/aie.mlir aie_arch.mlir
// RUN: %run_on_npu1% sed 's/NPUDEVICE/npu1_1col/g' -i aie_arch.mlir
// RUN: %python aiecc.py --no-aiesim --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=basic-sequential --xclbin-name=aie.xclbin --npu-insts-name=insts.bin aie_arch.mlir
//
// Derive + guard (after aiecc, before host compile/run): re-derive
// OUTBUF_ADDR/TRAP_PC from the compiled core ELF, regenerate from the
// .in templates, diff against the committed golden aie.mlir/test.cpp.
// Match -> exit 0 (the bytes just compiled ARE the golden; proceed).
// Drift -> non-zero, the test fails HERE (host lines below do not run),
// printing committed-vs-derived. One compile suffices: golden is the
// only kernel source and the tool emits golden-on-match, so a separate
// scratch pass is redundant; the --out-* files are a throwaway guard
// side-effect (nothing consumes them in single-pass).
// RUN: %python %S/../../../../xdna-emu/tools/debug-halt-probe-derive.py --elf aie_arch.mlir.prj/*core*.elf --in-mlir %S/aie.mlir.in --in-cpp %S/test.cpp.in --golden-mlir %S/aie.mlir --golden-cpp %S/test.cpp --out-mlir derived_check_aie.mlir --out-cpp derived_check_test.cpp
//
// RUN: clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags
// RUN: %run_on_npu1% ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin
// RUN: %run_on_npu2% ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin
```

Notes for the engineer:
- The ELF glob `aie_arch.mlir.prj/*core*.elf` mirrors the observed project layout (`aie_arch.mlir.prj/main_core_0_2.elf`). Verify it resolves to **exactly one** file after the `aiecc` step (both under lit and the bridge build dir); tighten the glob if more than one `*core*.elf` is emitted.
- Confirm the `%S/../../../../xdna-emu` depth resolves to the real `xdna-emu/tools/` path (pre-verified to resolve; re-confirm).
- The host compile uses `%S/test.cpp` (the golden, unchanged from the original probe). `derived_check_aie.mlir`/`derived_check_test.cpp` are written by the tool but consumed by nothing — they exist only so the tool's CLI contract is satisfied; the guard is purely its exit code.
- Keep the `cp %S/aie.mlir aie_arch.mlir` first line (matches the original probe): under upstream lit it creates `aie_arch.mlir`; under the bridge harness it is skipped but the harness synthesizes `aie_arch.mlir` from canonical `aie.mlir` itself — either way `aie_arch.mlir` exists for `aiecc`.

- [ ] **Step 5: Verify the probe builds and the guard passes — under BOTH paths**

This is the crux of the redesign: the single-pass `run.lit` must work under the bridge harness (the primary validation path), not only upstream lit.

(a) Round-trip (already proven in Step 3; re-confirm if `.in` touched): the manual derive against the existing chess ELF prints `... (golden match)` and `rc=0`, and the regenerated outputs are byte-identical to the committed golden.

(b) Bridge path: `./scripts/emu-bridge-test.sh --no-hw -v debug_halt_probe 2>&1 | tee /tmp/claude-1000/dhp-build.log`
Expected: the harness synthesizes `aie_arch.mlir`, runs `aiecc`, then runs the `%python …/debug-halt-probe-derive.py …` RUN line (it is not a `cp` line, so it is NOT skipped), which prints `… (golden match)`; the EMU run reaches the established `MASKPOLL_UNSATISFIED_EMU` / `TRAP_VERDICT:BEFORE_COMMIT` baseline (unchanged — golden bytes are byte-identical to before). Confirm in the log that the derive RUN line actually executed under the bridge and the `aie_arch.mlir.prj/*core*.elf` glob resolved. If the bridge does NOT execute the derive line (e.g. it filters non-aiecc/non-run commands), STOP and report — that is a real integration gap to resolve, not to paper over.
If running the bridge would contend for NPU hardware, use `--no-hw` (no HW contention) — that still exercises compile + derive+guard, which is what this task must prove. Only if `emu-bridge-test.sh` is entirely unavailable, fall back to (a) plus a manual replay of the exact RUN sequence (cp/sed/aiecc/derive) in a scratch dir, confirming the derive step exits 0 with `golden match` against the freshly-compiled ELF.

- [ ] **Step 6: Amend the existing mlir-aie commit**

Steps 1-2 already committed the templates at mlir-aie `424ce198be`. This revision changes only `run.lit`; amend it into that same commit (do NOT create a second commit; the mlir-aie commit is unpushed).

```bash
git -C ../mlir-aie add test/npu-xrt/debug_halt_probe/aie.mlir.in test/npu-xrt/debug_halt_probe/test.cpp.in test/npu-xrt/debug_halt_probe/run.lit
git -C ../mlir-aie commit --amend -m "debug-halt §8 close-out Item 2b: single-pass guarded probe build

run.lit compiles the committed golden once, then re-derives
OUTBUF_ADDR/TRAP_PC from the core ELF, regenerates from .in templates,
and guards against the committed golden aie.mlir/test.cpp (byte-unchanged
Phase A record). Drift fails the test loudly before host compile/run; a
match proceeds. Single-pass is bridge-harness-compatible (no scratch cp)
and equivalent to two-pass on the golden path. Silent rot is impossible.

Generated using Claude Code."
```

CRITICAL: `../mlir-aie` is a SEPARATE git repo with unrelated pre-existing dirty files. Use `git -C ../mlir-aie` with the three explicit paths ONLY. NEVER `git add -A`/`.` there. NEVER push. Confirm `git -C ../mlir-aie status` shows `aie.mlir`/`test.cpp` (no `.in`) NOT modified (golden intact) before and after. The xdna-emu-side tool/tests were already committed in Task 2 — do not recommit those.

---

## Task 4: Item 1 — §8 bookkeeping, findings addendum, coverage narrative

Pure documentation. Flip the three resolved §8 bullets, add the findings RESOLVED addendum, update the coverage narrative, regenerate artifacts. Folded last so it closes §8 after the implementation landed.

**Files:**
- Modify: `docs/superpowers/specs/2026-05-18-debug-halt-design.md` (§8, lines ~645-722)
- Modify: `docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md` (~line 101)
- Modify: `crates/xdna-archspec/src/coverage/units.rs` (`debug_halt` narrative, line ~158)
- Regenerate: `docs/coverage/aie2/implementation-gaps.md`, `subsystem-index.md`

- [ ] **Step 1: Flip the three resolved §8 bullets**

In `docs/superpowers/specs/2026-05-18-debug-halt-design.md` §8:

Find the bullet beginning `**G1 halt-timing (bounded escalation).**` and insert, immediately after that bold lead-in (before the existing sentence):
> `**RESOLVED (2026-05-19):** the contingency never fired -- G1 was DERIVED and SHIPPED (findings 2026-05-18 conclusion (a)). Retained below for history.`

Find the bullet beginning `**Probe-artifact robustness: \`OUTBUF_ADDR\`.**` and insert after the bold lead-in:
> `**RESOLVED (2026-05-19):** unified golden-record-preserving build-time derivation -- see docs/superpowers/specs/2026-05-19-debug-halt-section8-closeout-design.md (OUTBUF_ADDR and the sibling TRAP_PC both now derived + guarded). Retained below for history.`

Find the bullet beginning `**\`Core_Status\` RESET-bit EMU/HW divergence.**` and insert after the bold lead-in:
> `**RESOLVED (2026-05-19):** runtime enable path routed through write_control -- see 2026-05-19-debug-halt-section8-closeout-design.md. EMU now reports Core_Status 0x10001. Retained below for history.`

Leave the `**Count-step silicon-fidelity.**` and `**Resume hardware-verification.**` bullets **unchanged** (they stay OPEN).

- [ ] **Step 2: Add the findings RESOLVED addendum**

In `docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md`, immediately after line 101 (`zero; \`TRAP_VERDICT:BEFORE_COMMIT\` -- matching HW.`), add a new paragraph:

```
**RESOLVED (2026-05-19, §8 close-out):** the EMU `Core_Status = 0x10003`
RESET-bit divergence noted above is fixed. `Coordinator::enable_core`
now routes the core-debug enable through `write_control` (the same
register semantics as a CDO `Core_Control=0x1` write), clearing `reset`.
EMU now reports `Core_Status = 0x10001`, matching HW. See
`docs/superpowers/specs/2026-05-19-debug-halt-section8-closeout-design.md`.
```

- [ ] **Step 3: Update the coverage narrative**

In `crates/xdna-archspec/src/coverage/units.rs`, the `debug_halt` `d(...)` narrative (line ~158) currently ends:
`Open (tracked, spec section 8): count-step finer silicon characterization (decrement cadence / larger-N / 0x11-on-silicon -- only N=4 observed).`

Replace that closing sentence with:
`Section 8 close-out (2026-05-19): Core_Status RESET-bit divergence and OUTBUF_ADDR/TRAP_PC probe fragility RESOLVED; G1 bounded-escalation tracker retired (contingency never fired). Open (tracked, spec section 8, HW-budget-gated): count-step finer silicon characterization (decrement cadence / larger-N / 0x11-on-silicon -- only N=4 observed) and resume hardware-verification.`

- [ ] **Step 4: Regenerate coverage artifacts**

Run: `cargo run -p xdna-archspec --example gen_coverage_artifacts`
Expected: regenerates `docs/coverage/aie2/implementation-gaps.md` and `subsystem-index.md`. `git diff --stat` should show only the narrative line changing for `debug_halt` (still `Modeled { Full }`) and no other subsystem moving.

- [ ] **Step 5: Verify spine test still green**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib -p xdna-archspec`
Expected: all pass (including `implementation_gaps_source_is_the_spine_not_semantic`, hardened in `2bf87d4` — `debug_halt` stays Full so it must not appear in the PARTIAL/STUB gaps list).

- [ ] **Step 6: Full library test sweep**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: all pass, 0 failed.

- [ ] **Step 7: Commit (closes §8)**

```bash
git add docs/superpowers/specs/2026-05-18-debug-halt-design.md docs/superpowers/findings/2026-05-18-debug-halt-timing-and-single-step-count.md crates/xdna-archspec/src/coverage/units.rs docs/coverage/aie2/implementation-gaps.md docs/coverage/aie2/subsystem-index.md
git commit -m "debug-halt §8 close-out Item 1: bookkeeping -- 3 commitments resolved

Flips the G1 bounded-escalation (contingency never fired), OUTBUF_ADDR,
and Core_Status RESET-bit §8 bullets to RESOLVED; adds the findings
Core_Status RESOLVED addendum; updates the debug_halt coverage
narrative and regenerates artifacts (zero-drift except the narrative;
debug_halt stays Modeled Full). Count-step silicon-fidelity and resume
HW-verification remain OPEN by design (HW-budget-gated).

Generated using Claude Code."
```

---

## Final Review

After all four tasks, dispatch a whole-implementation code review (per subagent-driven-development): verify Item 3 unifies the enable paths with no clobber regression, Item 2 preserves the golden bytes and fails loudly on drift, Item 1 bookkeeping is accurate and the two HW-gated commitments remain OPEN and discoverable. Then use superpowers:finishing-a-development-branch.

## Out of scope (stays OPEN, by design)

Count-step finer silicon-fidelity and resume hardware-verification are HW-budget-gated, untouched, and remain surfaced in §8 and the `debug_halt` coverage narrative. Do not attempt them from the desk.
