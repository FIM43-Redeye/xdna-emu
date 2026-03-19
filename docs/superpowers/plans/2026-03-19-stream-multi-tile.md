# Stream Multi-Tile ISA Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable testing of ~26 stream-related instructions via a StreamStrategy class and stream_pair multi-tile MLIR support, closing coverage from 573/606 toward ~98%.

**Architecture:** StreamStrategy operates in three modes (stream_write, stream_read, ss_status) using the same two-tile column-slot layout as cascade pairs but with `aie.flow` routing instead of `aie.cascade_flow`. mov.d* instructions get dual testing: ComputeStrategy for non-stream combos, StreamStrategy for stream-source combos.

**Tech Stack:** Python 3, pytest, llvm-mc (Peano assembler), mlir-aie MLIR dialect

**Spec:** `docs/superpowers/specs/2026-03-19-stream-multi-tile-design.md`

---

### Task 1: StreamStrategy can_test, combos, and sizes

Add the StreamStrategy class skeleton with `can_test()`, `generate_combos()`,
and size computation methods.  Remove mov.d* from STREAM_MNEMONICS.  Remove
SS status skip from `classify_instruction()`.

**Files:**
- Modify: `tools/isa-test-gen.py` (lines 213-218 STREAM_MNEMONICS, lines 570-584 classify_instruction, insert new class after line 2760)
- Test: `tools/test_isa_test_gen.py`

**Context for implementer:**
- Read the spec at `docs/superpowers/specs/2026-03-19-stream-multi-tile-design.md`, sections "Instruction Inventory" and "Components > 1. StreamStrategy".
- Read lines 2649-2760 of `tools/isa-test-gen.py` for the CascadeReadStrategy pattern to follow.
- Read lines 2969-3138 of `tools/test_isa_test_gen.py` for the TestCascadeReadStrategy pattern to follow.
- The `_make_instr` helper at the top of the test file creates mock instruction dicts.
- The `_make_reg_op` helper creates mock operand dicts.
- `STREAM_MNEMONICS` is at line 213.  Remove `mov.d1` through `mov.d6`.
- `classify_instruction()` stream checks are at lines 570-584.  Remove the SS status skip (lines 578-584) since StreamStrategy will handle it.  Keep the STREAM_MNEMONICS check (line 571) -- it will still catch stream writes that StreamStrategy hasn't claimed yet (belt-and-suspenders; both paths produce "skipped" for unclaimed stream instructions).  Actually, since StreamStrategy will be in the STRATEGIES list and claim them first, the STREAM_MNEMONICS check in classify_instruction becomes unreachable for those mnemonics.  Remove it too for clarity, same as we did for the SCD skip.
- STRATEGIES list is at line 3249.  Insert `StreamStrategy()` after `CascadeStrategy()` (line 3254) and before `DoneStrategy()` (line 3255).

- [ ] **Step 1: Write failing tests for can_test**

Add `TestStreamStrategy` class to `tools/test_isa_test_gen.py`.  Follow the TestCascadeReadStrategy pattern.  Add these helper methods and test cases:

```python
class TestStreamStrategy:
    """Tests for StreamStrategy -- multi-tile stream instruction testing."""

    def _make_stream_write_scalar(self):
        """mov.nb ms, $src -- non-blocking scalar stream write."""
        return _make_instr("MOV_NB_mv_scl2ms", "mov.nb", "mov.nb\tms, $src", [
            {"name": "src", "bit_width": 7, "operand_type": "composite_register",
             "register_kind": "LdaScl", "is_output": False,
             "signed": False, "scale": None},
        ], slot="st")

    def _make_stream_write_blocking(self):
        """mov ms, $src -- blocking scalar stream write (mnemonic=mov)."""
        return _make_instr("MOV_mv_scl2ms", "mov", "mov\tms, $src", [
            {"name": "src", "bit_width": 7, "operand_type": "composite_register",
             "register_kind": "LdaScl", "is_output": False,
             "signed": False, "scale": None},
        ], slot="st")

    def _make_stream_write_ph(self):
        """mov.ph ms, $id, $pcktType -- packet header stream write."""
        return _make_instr("MOV_mv_ph2ms", "mov.ph", "mov.ph\tms, $id, $pcktType", [
            {"name": "id", "bit_width": 5, "operand_type": "register",
             "register_kind": "scalar", "is_output": False,
             "signed": False, "scale": None},
            {"name": "pcktType", "bit_width": 3, "operand_type": "immediate",
             "is_output": False, "signed": False, "scale": None},
        ], slot="st")

    def _make_stream_write_cph(self):
        """mov.cph ms, $addr, $nw, $op, $id -- config packet header."""
        return _make_instr("MOV_mv_cph2ms", "mov.cph",
                           "mov.cph\tms, $addr, $nw, $op, $id", [
            {"name": "addr", "bit_width": 3, "operand_type": "register",
             "register_kind": "modifier_m", "is_output": False,
             "signed": False, "scale": None},
            {"name": "nw", "bit_width": 2, "operand_type": "immediate",
             "is_output": False, "signed": False, "scale": None},
            {"name": "op", "bit_width": 2, "operand_type": "immediate",
             "is_output": False, "signed": True, "scale": None},
            {"name": "id", "bit_width": 5, "operand_type": "register",
             "register_kind": "scalar", "is_output": False,
             "signed": False, "scale": None},
        ], slot="st")

    def _make_stream_write_tlast_reg(self):
        """mov.nb ms, $src, $tlast -- doTlast_reg variant."""
        return _make_instr("MOV_NB_mv_scl2ms_doTlast_reg", "mov.nb",
                           "mov.nb\tms, $src, $tlast", [
            {"name": "src", "bit_width": 7, "operand_type": "composite_register",
             "register_kind": "LdaScl", "is_output": False,
             "signed": False, "scale": None},
        ], slot="st")

    def _make_mov_d1(self):
        """mov.d1 $dst, $src -- delayed move with MvSclSrc (includes ss)."""
        return _make_instr("MOV_D1", "mov.d1", "mov.d1\t$dst, $src", [
            {"name": "dst", "bit_width": 5, "operand_type": "register",
             "register_kind": "scalar", "is_output": False,
             "signed": False, "scale": None},
            {"name": "src", "bit_width": 7, "operand_type": "composite_register",
             "register_kind": "MvSclSrc", "is_output": False,
             "signed": False, "scale": None},
        ], slot="mv")

    def _make_ss_status_read(self):
        """mov $mRa, SS -- blocking stream switch status read."""
        return _make_instr("MOV_mv_ss2scl", "mov", "mov\t$mRa, SS", [
            {"name": "mRa", "bit_width": 5, "operand_type": "register",
             "register_kind": "scalar", "is_output": False,
             "signed": False, "scale": None},
        ], slot="lda")

    def _make_ss_status_read_nb(self):
        """mov.nb $mRa, SS -- non-blocking SS status read (mnemonic collision)."""
        return _make_instr("MOV_NB_mv_ss2scl", "mov.nb", "mov.nb\t$mRa, SS", [
            {"name": "mRa", "bit_width": 5, "operand_type": "register",
             "register_kind": "scalar", "is_output": False,
             "signed": False, "scale": None},
        ], slot="lda")

    # -- can_test --

    def test_stream_write_scalar_can_test(self):
        strategy = isa_test_gen.StreamStrategy()
        can, reason = strategy.can_test(self._make_stream_write_scalar())
        assert can, f"Expected can_test=True: {reason}"

    def test_stream_write_blocking_can_test(self):
        """Blocking mov ms, $src (mnemonic=mov) should be claimed."""
        strategy = isa_test_gen.StreamStrategy()
        can, reason = strategy.can_test(self._make_stream_write_blocking())
        assert can, f"Expected can_test=True: {reason}"

    def test_stream_write_ph_can_test(self):
        strategy = isa_test_gen.StreamStrategy()
        can, reason = strategy.can_test(self._make_stream_write_ph())
        assert can, f"Expected can_test=True: {reason}"

    def test_stream_write_cph_can_test(self):
        strategy = isa_test_gen.StreamStrategy()
        can, reason = strategy.can_test(self._make_stream_write_cph())
        assert can, f"Expected can_test=True: {reason}"

    def test_stream_write_tlast_reg_can_test(self):
        strategy = isa_test_gen.StreamStrategy()
        can, reason = strategy.can_test(self._make_stream_write_tlast_reg())
        assert can, f"Expected can_test=True: {reason}"

    def test_mov_d1_can_test(self):
        strategy = isa_test_gen.StreamStrategy()
        can, reason = strategy.can_test(self._make_mov_d1())
        assert can, f"Expected can_test=True: {reason}"

    def test_ss_status_can_test(self):
        strategy = isa_test_gen.StreamStrategy()
        can, reason = strategy.can_test(self._make_ss_status_read())
        assert can, f"Expected can_test=True: {reason}"

    def test_ss_status_nb_can_test(self):
        """mov.nb $mRa, SS shares mnemonic with mov.nb ms, $src.
        SS check must come first to route correctly."""
        strategy = isa_test_gen.StreamStrategy()
        can, reason = strategy.can_test(self._make_ss_status_read_nb())
        assert can, f"Expected can_test=True: {reason}"

    def test_non_stream_rejected(self):
        strategy = isa_test_gen.StreamStrategy()
        instr = _make_instr("ADD", "add", "add\t$mRx, $mRy, $mRz", [
            _make_reg_op("mRx", "scalar"),
        ])
        can, _ = strategy.can_test(instr)
        assert not can

    # -- mode detection --

    def test_mode_stream_write(self):
        strategy = isa_test_gen.StreamStrategy()
        assert strategy._detect_mode(self._make_stream_write_scalar()) == "stream_write"

    def test_mode_stream_read(self):
        strategy = isa_test_gen.StreamStrategy()
        assert strategy._detect_mode(self._make_mov_d1()) == "stream_read"

    def test_mode_ss_status(self):
        strategy = isa_test_gen.StreamStrategy()
        assert strategy._detect_mode(self._make_ss_status_read()) == "ss_status"

    def test_mode_ss_nb_routes_to_status_not_write(self):
        """mov.nb with SS must route to ss_status, not stream_write."""
        strategy = isa_test_gen.StreamStrategy()
        assert strategy._detect_mode(self._make_ss_status_read_nb()) == "ss_status"

    # -- generate_combos --

    def test_combos_scalar_write(self):
        strategy = isa_test_gen.StreamStrategy()
        combos = strategy.generate_combos(self._make_stream_write_scalar())
        assert len(combos) == 1
        assert combos[0]["src"] == "r0"

    def test_combos_ph_write(self):
        strategy = isa_test_gen.StreamStrategy()
        combos = strategy.generate_combos(self._make_stream_write_ph())
        assert len(combos) == 1
        assert "id" in combos[0]
        assert "pcktType" in combos[0]

    def test_combos_cph_write(self):
        strategy = isa_test_gen.StreamStrategy()
        combos = strategy.generate_combos(self._make_stream_write_cph())
        assert len(combos) == 1
        assert combos[0]["addr"] == "m0"
        assert "nw" in combos[0]
        assert "op" in combos[0]
        assert "id" in combos[0]

    def test_combos_tlast_reg(self):
        strategy = isa_test_gen.StreamStrategy()
        combos = strategy.generate_combos(self._make_stream_write_tlast_reg())
        assert len(combos) == 1
        assert "tlast" in combos[0]

    def test_combos_mov_d1(self):
        strategy = isa_test_gen.StreamStrategy()
        combos = strategy.generate_combos(self._make_mov_d1())
        assert len(combos) == 1
        assert combos[0]["src"] == "ss0"
        assert combos[0]["dst"] == "r0"

    def test_combos_ss_status(self):
        strategy = isa_test_gen.StreamStrategy()
        combos = strategy.generate_combos(self._make_ss_status_read())
        assert len(combos) == 1
        assert combos[0]["mRa"] == "r0"

    # -- buffer sizes --

    def test_producer_out_size_stream_write(self):
        strategy = isa_test_gen.StreamStrategy()
        instr = self._make_stream_write_scalar()
        assert strategy.compute_producer_output_size(instr) == 8

    def test_consumer_out_size_stream_write(self):
        strategy = isa_test_gen.StreamStrategy()
        instr = self._make_stream_write_scalar()
        assert strategy.compute_consumer_output_size(instr) == 4

    def test_producer_out_size_stream_read(self):
        strategy = isa_test_gen.StreamStrategy()
        instr = self._make_mov_d1()
        assert strategy.compute_producer_output_size(instr) == 8

    def test_consumer_out_size_stream_read(self):
        strategy = isa_test_gen.StreamStrategy()
        instr = self._make_mov_d1()
        assert strategy.compute_consumer_output_size(instr) == 4

    def test_producer_out_size_ss_status(self):
        strategy = isa_test_gen.StreamStrategy()
        instr = self._make_ss_status_read()
        assert strategy.compute_producer_output_size(instr) == 8

    def test_consumer_out_size_ss_status(self):
        strategy = isa_test_gen.StreamStrategy()
        instr = self._make_ss_status_read()
        assert strategy.compute_consumer_output_size(instr) == 12

    def test_producer_in_size_always_zero(self):
        strategy = isa_test_gen.StreamStrategy()
        for instr in [self._make_stream_write_scalar(),
                      self._make_mov_d1(),
                      self._make_ss_status_read()]:
            assert strategy.compute_producer_input_size(instr) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tools/test_isa_test_gen.py::TestStreamStrategy -v 2>&1 | head -50`
Expected: FAIL with `AttributeError: module 'isa-test-gen' has no attribute 'StreamStrategy'`

- [ ] **Step 3: Implement StreamStrategy class**

Add to `tools/isa-test-gen.py` after line 2760 (after CascadeReadStrategy, before ConversionStrategy).  Follow the CascadeReadStrategy pattern.

```python
import re  # already imported at top of file

class StreamStrategy(TestStrategy):
    """Tests stream instructions via multi-tile producer-consumer pairs.

    Three modes:
    - stream_write: test tile (producer, row 3) writes to ms, helper
      (consumer, row 2) reads from ss0 and stores result.
    - stream_read: helper (producer, row 3) writes to ms, test tile
      (consumer, row 2) reads from ss0 via mov.d* and stores result.
    - ss_status: helper (producer, row 3) writes to ms to establish
      flow, test tile (consumer, row 2) reads SS register.

    These are NOT bin-packed.  Each becomes a standalone stream_pair batch.
    """

    def _detect_mode(self, instr: dict) -> str:
        """Determine which mode this instruction uses."""
        asm = instr.get("asm_string", "")
        mnemonic = instr.get("mnemonic", "")
        # SS status reads -- check BEFORE mnemonic (mov.nb collision)
        if "SS" in asm and "mov" in mnemonic:
            return "ss_status"
        # Stream writes -- word-boundary check for "ms" in asm_string
        # (avoids false matches on register names containing "ms")
        if re.search(r'\bms\b', asm):
            return "stream_write"
        # Stream reads -- mov.d* with composite source
        if re.match(r"mov\.d[1-6]", mnemonic):
            return "stream_read"
        return ""

    def can_test(self, instr: dict) -> tuple[bool, str]:
        mode = self._detect_mode(instr)
        if mode:
            return (True, "")
        return (False, "not a stream instruction")

    def _has_tlast_reg(self, instr: dict) -> bool:
        """True for doTlast_reg variants with $tlast in asm_string."""
        return "$tlast" in instr.get("asm_string", "")

    def generate_combos(self, instr: dict) -> list[dict[str, str]]:
        mode = self._detect_mode(instr)
        asm = instr.get("asm_string", "")

        if mode == "ss_status":
            return [{"mRa": "r0"}]

        if mode == "stream_read":
            return [{"dst": "r0", "src": "ss0"}]

        # stream_write -- pattern depends on asm content
        combo: dict[str, str] = {}
        if "cph" in asm:
            combo = {"addr": "m0", "nw": "0", "op": "0", "id": "r0"}
        elif "ph" in asm:
            combo = {"id": "r0", "pcktType": "0"}
        else:
            combo = {"src": "r0"}

        if self._has_tlast_reg(instr):
            combo["tlast"] = "r1"

        return [combo]

    def compute_producer_input_size(self, instr: dict) -> int:
        return 0  # test values are immediates

    def compute_producer_output_size(self, instr: dict) -> int:
        return 8  # markers (before + after)

    def compute_consumer_output_size(self, instr: dict) -> int:
        mode = self._detect_mode(instr)
        if mode == "ss_status":
            return 12  # markers (8) + status value (4)
        return 4  # received 32-bit word

    def compute_input_size(self, instr, regs):
        return 0

    def compute_output_size(self, instr):
        return self.compute_producer_output_size(instr) + self.compute_consumer_output_size(instr)

    def generate_test_point(self, instr, regs, in_offset, out_offset, **_kw):
        raise NotImplementedError(
            "StreamStrategy uses generate_stream_pair(), not generate_test_point()"
        )
```

Also update STREAM_MNEMONICS (line 213) -- remove mov.d1-d6:
```python
STREAM_MNEMONICS = frozenset({
    "mov.nb", "mov.nb.tlast", "mov.tlast",
    "mov.ph", "mov.ph.nb", "mov.ph.nb.tlast", "mov.ph.tlast",
    "mov.cph", "mov.cph.nb", "mov.cph.nb.tlast", "mov.cph.tlast",
})
```

Remove stream/SS skips from `classify_instruction()` (lines 570-584).
Remove the STREAM_MNEMONICS check (lines 570-572) and the SS status
check (lines 578-584).  StreamStrategy claims these via the strategy
chain now.  Keep the MCD skip (lines 587-588) since
`classify_instruction()` is also called by a legacy code path (line
1603) that bypasses the strategy chain.

Add `StreamStrategy()` to the STRATEGIES list after `CascadeStrategy()`:
```python
STRATEGIES: list[TestStrategy] = [
    BranchStrategy(),
    LockStrategy(),
    FifoLoadStrategy(),
    CascadeReadStrategy(),
    CascadeStrategy(),
    StreamStrategy(),       # <-- new
    DoneStrategy(),
    EventStrategy(),
    PaddaSpStrategy(),
    LoadStrategy(),
    StoreStrategy(),
    ComputeStrategy(),
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tools/test_isa_test_gen.py::TestStreamStrategy -v`
Expected: All tests PASS

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `python3 -m pytest tools/test_isa_test_gen.py -v 2>&1 | tail -20`
Expected: All existing tests still pass (373+)

- [ ] **Step 6: Commit**

```bash
git add tools/isa-test-gen.py tools/test_isa_test_gen.py
git commit -m "feat(isa-harness): StreamStrategy can_test, combos, and sizes"
```

---

### Task 2: StreamStrategy assembly generation

Implement `generate_stream_pair()` with producer/consumer assembly for
all three modes.

**Files:**
- Modify: `tools/isa-test-gen.py` (StreamStrategy class)
- Test: `tools/test_isa_test_gen.py`

**Context for implementer:**
- Read the spec section "Components > 2. Assembly Templates" for exact assembly patterns.
- Read CascadeReadStrategy's `generate_cascade_pair()` (lines 2697-2754) and `_generate_producer()` / `_generate_consumer()` for the pattern to follow.
- Helper functions: `_substitute_asm()` substitutes operand values into asm_string, `_scalar_store()` generates st instruction, `_nop_sled()` generates N nops.
- Key design points:
  - All assembly uses `p0` for output (single-arg functions).
  - Stream write producer: loads test value via immediate, stores markers, executes stream write.
  - Stream write consumer (helper): reads ss0 via `mov.d1`, 1 nop, stores to output.
  - Stream read producer (helper): loads test value, stores markers, writes to ms.
  - Stream read consumer: executes test instruction (mov.d*), N nops matching delay, stores result.
  - SS status: consumer stores markers around `mov r0, SS`, plus the status value.
  - For cph variants: zero modifier_m register before use (`mov m0, #0`).
  - For doTlast_reg variants: set tlast register (`mov r1, #1`).
  - The delay value for mov.d* NOP sleds: extract from mnemonic (`mov.d3` -> 3 nops).

- [ ] **Step 1: Write failing tests for assembly generation**

Add to `TestStreamStrategy` in `tools/test_isa_test_gen.py`:

```python
    # -- generate_stream_pair: stream_write mode --

    def test_stream_write_producer_has_markers(self):
        strategy = isa_test_gen.StreamStrategy()
        regs = {"src": "r0"}
        result = strategy.generate_stream_pair(self._make_stream_write_scalar(), regs)
        prod = result["producer_asm"]
        assert "#170" in prod  # before marker
        assert "#204" in prod  # after marker

    def test_stream_write_producer_uses_p0(self):
        """Single-arg function: output mapped to p0."""
        strategy = isa_test_gen.StreamStrategy()
        regs = {"src": "r0"}
        result = strategy.generate_stream_pair(self._make_stream_write_scalar(), regs)
        prod = result["producer_asm"]
        assert "p0" in prod
        assert "p1" not in prod

    def test_stream_write_producer_has_test_instruction(self):
        strategy = isa_test_gen.StreamStrategy()
        regs = {"src": "r0"}
        result = strategy.generate_stream_pair(self._make_stream_write_scalar(), regs)
        prod = result["producer_asm"]
        assert "mov.nb" in prod and "ms" in prod

    def test_stream_write_consumer_reads_ss0(self):
        strategy = isa_test_gen.StreamStrategy()
        regs = {"src": "r0"}
        result = strategy.generate_stream_pair(self._make_stream_write_scalar(), regs)
        cons = result["consumer_asm"]
        assert "ss0" in cons

    def test_stream_write_consumer_stores_to_p0(self):
        strategy = isa_test_gen.StreamStrategy()
        regs = {"src": "r0"}
        result = strategy.generate_stream_pair(self._make_stream_write_scalar(), regs)
        cons = result["consumer_asm"]
        assert "p0" in cons

    def test_stream_write_consumer_has_nop_after_read(self):
        """Helper consumer uses mov.d1 which needs 1 nop before store."""
        strategy = isa_test_gen.StreamStrategy()
        regs = {"src": "r0"}
        result = strategy.generate_stream_pair(self._make_stream_write_scalar(), regs)
        cons = result["consumer_asm"]
        lines = cons.strip().split("\n")
        ss0_idx = next(i for i, l in enumerate(lines) if "ss0" in l)
        st_idx = next(i for i, l in enumerate(lines) if "st " in l.lower() and "p0" in l)
        assert st_idx > ss0_idx + 1, "Need at least 1 nop between mov.d1 and store"

    # -- stream_write: cph variant --

    def test_stream_write_cph_zeros_modifier(self):
        strategy = isa_test_gen.StreamStrategy()
        regs = {"addr": "m0", "nw": "0", "op": "0", "id": "r0"}
        result = strategy.generate_stream_pair(self._make_stream_write_cph(), regs)
        prod = result["producer_asm"]
        assert "m0" in prod and "#0" in prod

    # -- stream_write: doTlast_reg variant --

    def test_stream_write_tlast_reg_sets_register(self):
        strategy = isa_test_gen.StreamStrategy()
        regs = {"src": "r0", "tlast": "r1"}
        result = strategy.generate_stream_pair(self._make_stream_write_tlast_reg(), regs)
        prod = result["producer_asm"]
        assert "r1" in prod

    # -- generate_stream_pair: stream_read mode --

    def test_stream_read_consumer_has_test_instruction(self):
        strategy = isa_test_gen.StreamStrategy()
        regs = {"dst": "r0", "src": "ss0"}
        result = strategy.generate_stream_pair(self._make_mov_d1(), regs)
        cons = result["consumer_asm"]
        assert "mov.d1" in cons and "ss0" in cons

    def test_stream_read_consumer_has_nop_sled(self):
        """mov.d1 needs 1 nop after the read."""
        strategy = isa_test_gen.StreamStrategy()
        regs = {"dst": "r0", "src": "ss0"}
        result = strategy.generate_stream_pair(self._make_mov_d1(), regs)
        cons = result["consumer_asm"]
        lines = cons.strip().split("\n")
        d1_idx = next(i for i, l in enumerate(lines) if "mov.d1" in l)
        st_idx = next(i for i, l in enumerate(lines) if "st " in l.lower() and "p0" in l)
        assert st_idx > d1_idx + 1, "Need nop sled between mov.d1 and store"

    def test_stream_read_producer_writes_ms(self):
        strategy = isa_test_gen.StreamStrategy()
        regs = {"dst": "r0", "src": "ss0"}
        result = strategy.generate_stream_pair(self._make_mov_d1(), regs)
        prod = result["producer_asm"]
        assert "ms" in prod

    def test_stream_read_producer_has_markers(self):
        strategy = isa_test_gen.StreamStrategy()
        regs = {"dst": "r0", "src": "ss0"}
        result = strategy.generate_stream_pair(self._make_mov_d1(), regs)
        prod = result["producer_asm"]
        assert "#170" in prod
        assert "#204" in prod

    # -- generate_stream_pair: ss_status mode --

    def test_ss_status_consumer_has_ss_read(self):
        strategy = isa_test_gen.StreamStrategy()
        regs = {"mRa": "r0"}
        result = strategy.generate_stream_pair(self._make_ss_status_read(), regs)
        cons = result["consumer_asm"]
        assert "SS" in cons

    def test_ss_status_consumer_has_markers(self):
        strategy = isa_test_gen.StreamStrategy()
        regs = {"mRa": "r0"}
        result = strategy.generate_stream_pair(self._make_ss_status_read(), regs)
        cons = result["consumer_asm"]
        assert "#170" in cons
        assert "#204" in cons

    def test_ss_status_consumer_stores_value(self):
        strategy = isa_test_gen.StreamStrategy()
        regs = {"mRa": "r0"}
        result = strategy.generate_stream_pair(self._make_ss_status_read(), regs)
        cons = result["consumer_asm"]
        # Should store the SS value in addition to markers
        st_lines = [l for l in cons.split("\n") if "st " in l.lower() and "p0" in l]
        assert len(st_lines) >= 3, "Need 3 stores: before marker, after marker, status value"

    def test_ss_status_producer_writes_ms(self):
        strategy = isa_test_gen.StreamStrategy()
        regs = {"mRa": "r0"}
        result = strategy.generate_stream_pair(self._make_ss_status_read(), regs)
        prod = result["producer_asm"]
        assert "ms" in prod
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tools/test_isa_test_gen.py::TestStreamStrategy -k "stream_pair or stream_write_producer or stream_write_consumer or stream_read or ss_status" -v 2>&1 | head -40`
Expected: FAIL with `AttributeError: 'StreamStrategy' object has no attribute 'generate_stream_pair'`

- [ ] **Step 3: Implement generate_stream_pair and helpers**

Add these methods to the StreamStrategy class in `tools/isa-test-gen.py`:

```python
    def _delay_count(self, instr: dict) -> int:
        """Extract delay count from mov.d* mnemonic (e.g., mov.d3 -> 3)."""
        mnemonic = instr.get("mnemonic", "")
        m = re.match(r"mov\.d(\d)", mnemonic)
        return int(m.group(1)) if m else 1

    def generate_stream_pair(self, instr: dict, regs: dict) -> dict:
        """Generate producer and consumer assembly programs.

        Returns dict with 'producer_asm' and 'consumer_asm' strings.
        """
        mode = self._detect_mode(instr)
        if mode == "stream_write":
            return {
                "producer_asm": self._gen_write_producer(instr, regs),
                "consumer_asm": self._gen_write_consumer(),
            }
        elif mode == "stream_read":
            return {
                "producer_asm": self._gen_read_producer(),
                "consumer_asm": self._gen_read_consumer(instr, regs),
            }
        else:  # ss_status
            return {
                "producer_asm": self._gen_status_producer(),
                "consumer_asm": self._gen_status_consumer(instr, regs),
            }

    def _gen_write_producer(self, instr: dict, regs: dict) -> str:
        """Test tile: loads immediate, markers, executes stream write."""
        asm_str = instr.get("asm_string", "")
        name = instr["name"]
        lines = []
        lines.append(f"  // ---- stream write producer (test): {name} ----")
        # Set up source register with known test value.
        lines.append("  mov r0, #0xBEEF")
        # For doTlast_reg variants, set tlast register.
        if self._has_tlast_reg(instr):
            lines.append("  mov r1, #1")
        # For cph variants, zero modifier_m register.
        if "cph" in asm_str:
            lines.append("  mov m0, #0")
        # Before marker.
        lines.append("  mov r14, #170")
        lines.extend(_scalar_store("r14", "p0", 0))
        # Execute the stream write under test.
        asm_line = _substitute_asm(asm_str, regs)
        lines.append(f"  {asm_line}")
        # After marker.
        lines.append("  mov r14, #204")
        lines.extend(_scalar_store("r14", "p0", 4))
        lines.append("")
        return "\n".join(lines)

    def _gen_write_consumer(self) -> str:
        """Helper: drain stream via mov.d1, store received value."""
        lines = []
        lines.append("  // ---- stream write consumer (helper): drain ss0 ----")
        lines.append("  mov.d1 r0, ss0")
        lines.append("  nop")  # 1-cycle delay for mov.d1
        lines.extend(_scalar_store("r0", "p0", 0))
        lines.append("")
        return "\n".join(lines)

    def _gen_read_producer(self) -> str:
        """Helper: write known value to ms."""
        lines = []
        lines.append("  // ---- stream read producer (helper): write ms ----")
        lines.append("  mov r0, #0xBEEF")
        lines.append("  mov r14, #170")
        lines.extend(_scalar_store("r14", "p0", 0))
        lines.append("  mov ms, r0")
        lines.append("  mov r14, #204")
        lines.extend(_scalar_store("r14", "p0", 4))
        lines.append("")
        return "\n".join(lines)

    def _gen_read_consumer(self, instr: dict, regs: dict) -> str:
        """Test tile: execute mov.d* stream read, store result."""
        asm_str = instr.get("asm_string", "")
        name = instr["name"]
        delay = self._delay_count(instr)
        lines = []
        lines.append(f"  // ---- stream read consumer (test): {name} ----")
        asm_line = _substitute_asm(asm_str, regs)
        lines.append(f"  {asm_line}")
        lines.extend(_nop_sled(delay))
        lines.extend(_scalar_store("r0", "p0", 0))
        lines.append("")
        return "\n".join(lines)

    def _gen_status_producer(self) -> str:
        """Helper: write dummy value to ms to establish active flow."""
        lines = []
        lines.append("  // ---- SS status producer (helper): establish flow ----")
        lines.append("  mov r0, #0xDEAD")
        lines.append("  mov r14, #170")
        lines.extend(_scalar_store("r14", "p0", 0))
        lines.append("  mov ms, r0")
        lines.append("  mov r14, #204")
        lines.extend(_scalar_store("r14", "p0", 4))
        lines.append("")
        return "\n".join(lines)

    def _gen_status_consumer(self, instr: dict, regs: dict) -> str:
        """Test tile: read SS register with markers."""
        asm_str = instr.get("asm_string", "")
        name = instr["name"]
        lines = []
        lines.append(f"  // ---- SS status consumer (test): {name} ----")
        lines.append("  mov r14, #170")
        lines.extend(_scalar_store("r14", "p0", 0))
        asm_line = _substitute_asm(asm_str, regs)
        lines.append(f"  {asm_line}")
        lines.append("  mov r14, #204")
        lines.extend(_scalar_store("r14", "p0", 4))
        lines.extend(_scalar_store("r0", "p0", 8))
        lines.append("")
        return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tools/test_isa_test_gen.py::TestStreamStrategy -v`
Expected: All TestStreamStrategy tests PASS

- [ ] **Step 5: Run full test suite**

Run: `python3 -m pytest tools/test_isa_test_gen.py -v 2>&1 | tail -20`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add tools/isa-test-gen.py tools/test_isa_test_gen.py
git commit -m "feat(isa-harness): StreamStrategy assembly generation (3 modes)"
```

---

### Task 3: generate_all integration with stream_specs and mov.d* second pass

Wire StreamStrategy into the main `generate_all()` function: collect
stream specs separately, process them as standalone stream_pair batches,
and add a second pass for mov.d* non-stream combos via ComputeStrategy.

**Files:**
- Modify: `tools/isa-test-gen.py` (generate_all function, lines 3399-3660)
- Test: `tools/test_isa_test_gen.py`

**Context for implementer:**
- Read the spec section "Components > 5. Integration with generate_all()".
- Read lines 3414-3652 of `tools/isa-test-gen.py` for the cascade_specs pattern to replicate.
- The cascade handling collects specs in `cascade_specs` list during classification, then processes them after normal batching into standalone batch pairs.
- Stream specs follow the identical pattern: collect during classification when `isinstance(strategy, StreamStrategy)`, then process after cascade batches.
- **mov.d* dual testing**: After the main classification loop, do a second pass over all instructions.  For each mov.d* instruction, check if ComputeStrategy can handle it.  If so, add to `test_point_specs` for bin-packing into normal batches.
- The `build_mega_program()` function wraps assembly snippets into a complete program with `test_kernel` symbol and `ret lr` epilogue.
- Manifest `total_test_points` must include stream specs count.

- [ ] **Step 1: Write failing tests for generate_all integration**

Add to `TestGenerateAll` in `tools/test_isa_test_gen.py`:

```python
    def test_stream_pairs_in_manifest(self, isa_json_path, out_dir):
        """Stream instructions should produce stream_pair batches."""
        manifest = generate_all(isa_json_path, out_dir)
        stream_batches = [
            b for b in manifest["batches"]
            if b.get("source_type") == "stream_pair"
        ]
        assert len(stream_batches) > 0, "Expected stream_pair batches"

    def test_stream_pair_has_both_files(self, isa_json_path, out_dir):
        """Each stream_pair batch should have producer and consumer filenames."""
        manifest = generate_all(isa_json_path, out_dir)
        stream_batches = [
            b for b in manifest["batches"]
            if b.get("source_type") == "stream_pair"
        ]
        for batch in stream_batches:
            assert "producer_filename" in batch
            assert "consumer_filename" in batch
            prod_path = os.path.join(out_dir, batch["producer_filename"])
            cons_path = os.path.join(out_dir, batch["consumer_filename"])
            assert os.path.exists(prod_path), f"Missing {prod_path}"
            assert os.path.exists(cons_path), f"Missing {cons_path}"

    def test_stream_instructions_now_testable(self, isa_json_path, out_dir):
        """Stream instructions should no longer appear in skip reasons."""
        manifest = generate_all(isa_json_path, out_dir)
        skip_reasons = manifest.get("skip_reasons", {})
        assert "stream instruction" not in skip_reasons
        assert "stream switch status read (hangs without streams)" not in skip_reasons

    def test_mov_d_dual_testing(self, isa_json_path, out_dir):
        """mov.d* should appear in both normal batches and stream_pair batches."""
        manifest = generate_all(isa_json_path, out_dir)
        stream_batches = [
            b for b in manifest["batches"]
            if b.get("source_type") == "stream_pair"
        ]
        normal_batches = [
            b for b in manifest["batches"]
            if b.get("source_type", "assembly") == "assembly"
        ]
        # Check stream pairs contain mov.d* instructions
        stream_instrs = {b["instruction"] for b in stream_batches
                         if "instruction" in b}
        has_d_stream = any("MOV_D" in i for i in stream_instrs)
        assert has_d_stream, "Expected mov.d* in stream_pair batches"

        # Check normal batches contain mov.d* test points
        normal_instrs = set()
        for batch in normal_batches:
            for test in batch.get("tests", []):
                normal_instrs.add(test.get("instruction", ""))
        has_d_normal = any("MOV_D" in i for i in normal_instrs)
        assert has_d_normal, "Expected mov.d* in normal assembly batches"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tools/test_isa_test_gen.py::TestGenerateAll::test_stream_pairs_in_manifest -v`
Expected: FAIL (no stream_pair batches yet)

- [ ] **Step 3: Implement generate_all changes**

In `tools/isa-test-gen.py`, modify `generate_all()`:

1. Add `stream_specs` list alongside `cascade_specs` (near line 3416):
```python
    stream_specs: list[tuple[dict, int, dict, StreamStrategy]] = []
```

2. In the main classification loop (near line 3438), add stream handling:
```python
        if isinstance(strategy, StreamStrategy):
            testable_count += 1
            combos = strategy.generate_combos(instr)
            for combo_idx, regs in enumerate(combos):
                regs["_combo_idx"] = combo_idx
                stream_specs.append((instr, combo_idx, regs, strategy))
            continue
```

3. After the main classification loop, add mov.d* second pass:
```python
    # Second pass: mov.d* instructions also get non-stream combos via
    # ComputeStrategy for dual testing (stream + non-stream).
    compute = ComputeStrategy()
    for instr in all_instrs:
        mnemonic = instr.get("mnemonic", "")
        if re.match(r"mov\.d[1-6]", mnemonic):
            can, _ = compute.can_test(instr)
            if can:
                testable_count += 1
                combos = compute.generate_combos(instr)
                for combo_idx, regs in enumerate(combos):
                    regs["_combo_idx"] = combo_idx
                    test_point_specs.append((instr, combo_idx, regs, compute))
```

4. After the cascade batch generation (near line 3649), add stream batch generation -- follow the exact same pattern as cascade:
```python
    # Generate stream_pair batches.
    for instr, combo_idx, regs, strategy in stream_specs:
        pair = strategy.generate_stream_pair(instr, regs)

        prod_asm = pair["producer_asm"]
        cons_asm = pair["consumer_asm"]

        prod_program = build_mega_program([prod_asm])
        cons_program = build_mega_program([cons_asm])

        prod_filename = f"batch_{batch_idx:03d}_producer.s"
        cons_filename = f"batch_{batch_idx:03d}_consumer.s"

        with open(os.path.join(out_dir, prod_filename), "w") as f:
            f.write(prod_program)
        with open(os.path.join(out_dir, cons_filename), "w") as f:
            f.write(cons_program)

        batches.append({
            "batch_index": batch_idx,
            "filename": cons_filename,
            "source_type": "stream_pair",
            "producer_filename": prod_filename,
            "consumer_filename": cons_filename,
            "producer_in_size": strategy.compute_producer_input_size(instr),
            "producer_out_size": strategy.compute_producer_output_size(instr),
            "consumer_in_size": 0,
            "consumer_out_size": strategy.compute_consumer_output_size(instr),
            "instruction": instr["name"],
            "slot": instr.get("slot", ""),
            "test_count": 1,
            "in_size": 0,
            "out_size": (strategy.compute_producer_output_size(instr)
                         + strategy.compute_consumer_output_size(instr)),
            "tests": [{
                "instruction": instr["name"],
                "slot": instr.get("slot", ""),
                "combo_index": combo_idx,
                "operands": {k: v for k, v in regs.items()
                             if not k.startswith("_")},
                "in_offset": 0,
                "in_size": 0,
                "out_offset": 0,
                "out_size": (strategy.compute_producer_output_size(instr)
                             + strategy.compute_consumer_output_size(instr)),
            }],
        })
        batch_idx += 1
```

5. Update `total_test_points` to include stream specs:
```python
    total_test_points = (len(test_point_specs) + total_conv_test_points
        + len(cascade_specs) + len(stream_specs))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tools/test_isa_test_gen.py::TestGenerateAll -v`
Expected: All TestGenerateAll tests PASS (including new stream tests)

- [ ] **Step 5: Run full test suite**

Run: `python3 -m pytest tools/test_isa_test_gen.py -v 2>&1 | tail -20`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add tools/isa-test-gen.py tools/test_isa_test_gen.py
git commit -m "feat(isa-harness): integrate stream_specs into generate_all"
```

---

### Task 4: MLIR generation for stream_pair source_type

Extend `isa-multi-tile-gen.py` to handle `stream_pair` batches: emit
`aie.flow` routing, objectfifos without producer input, and adjusted
core blocks.

**Files:**
- Modify: `tools/isa-multi-tile-gen.py` (add _is_stream, _emit_stream_objectfifos, _emit_stream_cores, extend runtime sequence)
- Test: `tools/test_isa_test_gen.py`

**Context for implementer:**
- Read the spec section "Components > 6. MLIR Generation".
- Read `_is_cascade()` (line 195), `_emit_cascade_objectfifos()` (lines 455-518), `_emit_cascade_cores()` (lines 549-591), and the runtime sequence (lines 275-420) of `tools/isa-multi-tile-gen.py`.
- Key differences from cascade:
  - `aie.flow(%tile_N_3, Core : 0, %tile_N_2, Core : 0)` instead of `aie.cascade_flow`
  - No producer input objectfifo (producer_in_size == 0)
  - Producer function takes ONE memref arg (output only), not two
  - Both producer and consumer take single memref args
  - Runtime sequence has NO input DMA for stream pairs
- The `generate_phase_mlir()` function (starts around line 200) builds MLIR by iterating batches and calling emit helpers for each.  The tile declaration section and routing declaration section need to dispatch on source_type.
- `compute_phase_buffer_layout()` needs to handle stream pairs (in_size=0).

- [ ] **Step 1: Write failing tests for stream MLIR generation**

Add `TestMultiTileStream` class to `tools/test_isa_test_gen.py`:

```python
class TestMultiTileStream:
    """Tests for stream_pair MLIR generation in isa-multi-tile-gen."""

    @pytest.fixture
    def multi(self):
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("multi", "tools/isa-multi-tile-gen.py")
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _make_stream_batch(self, idx=0, cons_out=4):
        return {
            "batch_index": idx,
            "source_type": "stream_pair",
            "producer_filename": f"batch_{idx:03d}_producer.s",
            "consumer_filename": f"batch_{idx:03d}_consumer.s",
            "producer_in_size": 0,
            "producer_out_size": 8,
            "consumer_in_size": 0,
            "consumer_out_size": cons_out,
            "instruction": "MOV_NB_mv_scl2ms",
            "slot": "st",
            "in_size": 0,
            "out_size": 8 + cons_out,
            "test_count": 1,
            "filename": f"batch_{idx:03d}_consumer.s",
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

    def _make_cascade_batch(self, idx=2):
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
            "filename": f"batch_{idx:03d}_consumer.s",
        }

    def test_stream_mlir_has_aie_flow(self, multi):
        mlir = multi.generate_phase_mlir([self._make_stream_batch()], 0)
        assert "aie.flow" in mlir
        assert "Core : 0" in mlir

    def test_stream_mlir_no_cascade_flow(self, multi):
        mlir = multi.generate_phase_mlir([self._make_stream_batch()], 0)
        assert "cascade_flow" not in mlir

    def test_stream_mlir_has_two_compute_tiles(self, multi):
        mlir = multi.generate_phase_mlir([self._make_stream_batch()], 0)
        assert "aie.tile(0, 2)" in mlir
        assert "aie.tile(0, 3)" in mlir

    def test_stream_mlir_has_memtile(self, multi):
        mlir = multi.generate_phase_mlir([self._make_stream_batch()], 0)
        assert "aie.tile(0, 1)" in mlir

    def test_stream_mlir_no_producer_input_objectfifo(self, multi):
        """Stream pairs have no producer input -- test values are immediates."""
        mlir = multi.generate_phase_mlir([self._make_stream_batch()], 0)
        assert "of_prod_in" not in mlir

    def test_stream_mlir_has_producer_output(self, multi):
        mlir = multi.generate_phase_mlir([self._make_stream_batch()], 0)
        assert "of_prod_out_0" in mlir

    def test_stream_mlir_has_consumer_output(self, multi):
        mlir = multi.generate_phase_mlir([self._make_stream_batch()], 0)
        assert "of_cons_out_0" in mlir

    def test_stream_producer_one_arg(self, multi):
        """Producer function takes one memref arg (output only)."""
        mlir = multi.generate_phase_mlir([self._make_stream_batch()], 0)
        for line in mlir.split("\n"):
            if "test_kernel_0_prod" in line and "func.func" in line:
                assert line.count("memref") == 1, f"Producer should have 1 arg: {line}"
                break
        else:
            assert False, "Producer function declaration not found"

    def test_stream_consumer_one_arg(self, multi):
        mlir = multi.generate_phase_mlir([self._make_stream_batch()], 0)
        for line in mlir.split("\n"):
            if "test_kernel_0_cons" in line and "func.func" in line:
                assert line.count("memref") == 1, f"Consumer should have 1 arg: {line}"
                break
        else:
            assert False, "Consumer function declaration not found"

    def test_stream_mixed_with_normal_and_cascade(self, multi):
        batches = [
            self._make_stream_batch(0),
            self._make_normal_batch(1),
            self._make_cascade_batch(2),
        ]
        mlir = multi.generate_phase_mlir(batches, 0)
        assert "aie.flow" in mlir       # stream
        assert "cascade_flow" in mlir    # cascade
        assert "of_in_1" in mlir         # normal

    def test_stream_no_input_dma(self, multi):
        """Runtime sequence should have no input DMA for stream pairs."""
        mlir = multi.generate_phase_mlir([self._make_stream_batch()], 0)
        assert "of_prod_in" not in mlir

    def test_stream_ss_status_larger_consumer_output(self, multi):
        """SS status mode has consumer_out_size=12 (3xi32)."""
        batch = self._make_stream_batch(cons_out=12)
        mlir = multi.generate_phase_mlir([batch], 0)
        assert "memref<3xi32>" in mlir

    def test_stream_has_objectfifo_link(self, multi):
        """Producer output uses two-hop via memtile."""
        mlir = multi.generate_phase_mlir([self._make_stream_batch()], 0)
        assert "objectfifo.link" in mlir
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tools/test_isa_test_gen.py::TestMultiTileStream -v 2>&1 | head -30`
Expected: FAIL (stream_pair not handled yet)

- [ ] **Step 3: Implement stream MLIR generation**

In `tools/isa-multi-tile-gen.py`:

1. Add `_is_stream()` helper after `_is_cascade()` (line 197):
```python
def _is_stream(batch: dict) -> bool:
    """Return True if batch is a stream_pair."""
    return batch.get("source_type") == "stream_pair"

def _is_multi_tile(batch: dict) -> bool:
    """Return True if batch uses two tiles (cascade or stream)."""
    return _is_cascade(batch) or _is_stream(batch)
```

2. Add `_emit_stream_objectfifos()` after `_emit_cascade_objectfifos()`:
```python
def _emit_stream_objectfifos(lines: list[str], col: int, batch: dict) -> None:
    """Emit objectfifos and function declarations for a stream_pair.

    Like cascade but simpler: no producer input objectfifo.
    Data flow:
      Producer output: row3 -> memtile -> shim (two-hop with link)
      Consumer output: row2 -> shim (direct)
    """
    prod_out_elems = max(1, batch["producer_out_size"] // 4)
    cons_out_elems = max(1, batch["consumer_out_size"] // 4)
    prod_obj = _obj_filename(batch, "producer_filename")
    cons_obj = _obj_filename(batch, "consumer_filename")

    # NO producer input objectfifo -- test values are immediates.

    # Producer output: producer(row3) -> memtile(row1) -> shim(row0)
    lines.append(
        f"    aie.objectfifo @of_prod_out_{col}_0(%tile_{col}_3, "
        f"{{%tile_{col}_1}}, 1 : i32) "
        f": !aie.objectfifo<memref<{prod_out_elems}xi32>>"
    )
    lines.append(
        f"    aie.objectfifo @of_prod_out_{col}_1(%tile_{col}_1, "
        f"{{%tile_{col}_0}}, 1 : i32) "
        f": !aie.objectfifo<memref<{prod_out_elems}xi32>>"
    )
    lines.append(
        f"    aie.objectfifo.link [@of_prod_out_{col}_0] -> "
        f"[@of_prod_out_{col}_1] ([] [])"
    )
    lines.append("")

    # Consumer output: consumer(row2) -> shim(row0) (direct)
    lines.append(
        f"    aie.objectfifo @of_cons_out_{col}(%tile_{col}_2, "
        f"{{%tile_{col}_0}}, 2 : i32) "
        f": !aie.objectfifo<memref<{cons_out_elems}xi32>>"
    )
    lines.append("")

    # Function declarations -- both take single memref (output only)
    lines.append(
        f'    func.func private @test_kernel_{col}_prod'
        f"(memref<{prod_out_elems}xi32>) "
        f'attributes {{link_with = "{prod_obj}"}}'
    )
    lines.append(
        f'    func.func private @test_kernel_{col}_cons'
        f"(memref<{cons_out_elems}xi32>) "
        f'attributes {{link_with = "{cons_obj}"}}'
    )
    lines.append("")
```

3. Add `_emit_stream_cores()` after `_emit_cascade_cores()`:
```python
def _emit_stream_cores(lines: list[str], col: int, batch: dict) -> None:
    """Emit core blocks for a stream_pair (both single-arg functions)."""
    prod_out_elems = max(1, batch["producer_out_size"] // 4)
    cons_out_elems = max(1, batch["consumer_out_size"] // 4)

    # Producer core (row 3): output only, no input objectfifo
    lines.append(f"    aie.core(%tile_{col}_3) {{")
    lines.append(
        f"      %sub_out = aie.objectfifo.acquire @of_prod_out_{col}_0(Produce, 1) "
        f": !aie.objectfifosubview<memref<{prod_out_elems}xi32>>"
    )
    lines.append(
        f"      %elem_out = aie.objectfifo.subview.access %sub_out[0] "
        f": !aie.objectfifosubview<memref<{prod_out_elems}xi32>> -> memref<{prod_out_elems}xi32>"
    )
    lines.append(
        f"      func.call @test_kernel_{col}_prod(%elem_out) "
        f": (memref<{prod_out_elems}xi32>) -> ()"
    )
    lines.append(f"      aie.objectfifo.release @of_prod_out_{col}_0(Produce, 1)")
    lines.append("      aie.end")
    lines.append("    }")
    lines.append("")

    # Consumer core (row 2): output only
    lines.append(f"    aie.core(%tile_{col}_2) {{")
    lines.append(
        f"      %sub_out = aie.objectfifo.acquire @of_cons_out_{col}(Produce, 1) "
        f": !aie.objectfifosubview<memref<{cons_out_elems}xi32>>"
    )
    lines.append(
        f"      %elem_out = aie.objectfifo.subview.access %sub_out[0] "
        f": !aie.objectfifosubview<memref<{cons_out_elems}xi32>> -> memref<{cons_out_elems}xi32>"
    )
    lines.append(
        f"      func.call @test_kernel_{col}_cons(%elem_out) "
        f": (memref<{cons_out_elems}xi32>) -> ()"
    )
    lines.append(f"      aie.objectfifo.release @of_cons_out_{col}(Produce, 1)")
    lines.append("      aie.end")
    lines.append("    }")
    lines.append("")
```

4. Update `generate_phase_mlir()` to dispatch on source_type.

The existing code uses `cascade_cols = [col for col, b in enumerate(batches) if _is_cascade(b)]` for tile declarations and routing.  Refactor this:

```python
# Replace cascade_cols with multi_tile_cols, then separate for routing
multi_tile_cols = [col for col, b in enumerate(batches) if _is_multi_tile(b)]
cascade_cols = [col for col, b in enumerate(batches) if _is_cascade(b)]
stream_cols = [col for col, b in enumerate(batches) if _is_stream(b)]
```

Use `multi_tile_cols` for tile declarations (memtile row 1, compute row 3).  Use `cascade_cols` for `aie.cascade_flow` and `stream_cols` for `aie.flow`:

```python
# Routing declarations (after tile declarations)
for col in cascade_cols:
    lines.append(f"    aie.cascade_flow(%tile_{col}_3, %tile_{col}_2)")
for col in stream_cols:
    lines.append(
        f"    aie.flow(%tile_{col}_3, Core : 0, %tile_{col}_2, Core : 0)"
    )
```

In the objectfifo and core emission sections, add stream dispatch:
```python
for col, batch in enumerate(batches):
    if _is_cascade(batch):
        _emit_cascade_objectfifos(lines, col, batch)
    elif _is_stream(batch):
        _emit_stream_objectfifos(lines, col, batch)
    else:
        _emit_normal_objectfifos(lines, col, batch)

for col, batch in enumerate(batches):
    if _is_cascade(batch):
        _emit_cascade_cores(lines, col, batch)
    elif _is_stream(batch):
        _emit_stream_cores(lines, col, batch)
    else:
        _emit_normal_core(lines, col, batch)
```

In the runtime sequence, handle stream pairs explicitly:

**Constants section** -- stream pairs emit output offsets only, NO input offsets:
```python
for col, batch in enumerate(batches):
    if _is_stream(batch):
        prod_out_elems = max(1, batch["producer_out_size"] // 4)
        cons_out_elems = max(1, batch["consumer_out_size"] // 4)
        out_off = layout["out_offsets_elems"][col]
        lines.append(
            f"      %c_prod_out_off_{col} = arith.constant {out_off} : i64"
        )
        lines.append(
            f"      %c_prod_out_len_{col} = arith.constant {prod_out_elems} : i64"
        )
        cons_out_off = out_off + prod_out_elems
        lines.append(
            f"      %c_cons_out_off_{col} = arith.constant {cons_out_off} : i64"
        )
        lines.append(
            f"      %c_cons_out_len_{col} = arith.constant {cons_out_elems} : i64"
        )
    elif _is_cascade(batch):
        # ... existing cascade constants (includes input offsets) ...
    else:
        # ... existing normal constants ...
```

**Output DMAs** -- stream pairs emit two output DMAs, identical to cascade:
```python
if _is_stream(batch):
    prod_out_elems = max(1, batch["producer_out_size"] // 4)
    cons_out_elems = max(1, batch["consumer_out_size"] // 4)
    lines.append(
        f"      aiex.npu.dma_memcpy_nd("
        f"%out[%c0, %c0, %c0, %c_prod_out_off_{col}]"
        f"[%c1, %c1, %c1, %c_prod_out_len_{col}]"
        f"[%c0, %c0, %c0, %c1]) "
        f"{{metadata = @of_prod_out_{col}_1, id = {dma_id} : i64, "
        f"issue_token = true}} : memref<{total_out}xi32>"
    )
    dma_id += 1
    lines.append(
        f"      aiex.npu.dma_memcpy_nd("
        f"%out[%c0, %c0, %c0, %c_cons_out_off_{col}]"
        f"[%c1, %c1, %c1, %c_cons_out_len_{col}]"
        f"[%c0, %c0, %c0, %c1]) "
        f"{{metadata = @of_cons_out_{col}, id = {dma_id} : i64, "
        f"issue_token = true}} : memref<{total_out}xi32>"
    )
    dma_id += 1
```

**Input DMAs** -- SKIP stream pairs entirely:
```python
for col, batch in enumerate(batches):
    if _is_stream(batch):
        continue  # no input objectfifo for stream pairs
    elif _is_cascade(batch):
        # ... existing cascade input DMA ...
    else:
        # ... existing normal input DMA ...
    input_dma_id += 1
```

**dma_wait** -- stream pairs wait for both outputs, same as cascade:
```python
if _is_stream(batch):
    lines.append(
        f"      aiex.npu.dma_wait {{symbol = @of_prod_out_{col}_1}}"
    )
    lines.append(
        f"      aiex.npu.dma_wait {{symbol = @of_cons_out_{col}}}"
    )
```

5. Update `compute_phase_buffer_layout()` to handle stream pairs where
`in_size == 0`.  The existing code does `in_elems = max(1, batch["in_size"] // 4)`.
For stream pairs this produces a phantom 1-element allocation.  Fix by
checking `_is_stream()` and setting `in_elems = 0`:

```python
for batch in batches:
    if _is_stream(batch):
        in_elems = 0  # no input data for stream pairs
    else:
        in_elems = max(1, batch["in_size"] // 4)
    # ... rest of layout computation ...
```

6. Update `prepare_phase_objects()` to handle stream pairs -- same as
cascade (two .o files with _prod/_cons suffixes).  Replace the
`if _is_cascade(batch):` check with `if _is_multi_tile(batch):`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tools/test_isa_test_gen.py::TestMultiTileStream -v`
Expected: All TestMultiTileStream tests PASS

- [ ] **Step 5: Run full test suite**

Run: `python3 -m pytest tools/test_isa_test_gen.py -v 2>&1 | tail -20`
Expected: All tests pass (including TestMultiTileCascade unchanged)

- [ ] **Step 6: Commit**

```bash
git add tools/isa-multi-tile-gen.py tools/test_isa_test_gen.py
git commit -m "feat(isa-multi-tile): stream_pair MLIR generation with aie.flow"
```

---

### Task 5: E2E verification

Run the full generator, verify all stream pair `.s` files assemble,
check skip reasons are clean, and verify coverage numbers.

**Files:**
- No files modified (verification only)

**Context for implementer:**
- The generator outputs to a temp directory.  Run `python3 tools/isa-test-gen.py --out /tmp/stream-test` to generate all batches.
- Check `manifest.json` for stream_pair batches and skip reasons.
- Assemble each `.s` file with `llvm-mc` to verify they're valid AIE2 assembly.
- The environment needs `source ~/npu-work/toolchain-build/activate-npu-env.sh` for llvm-mc.
- Check that `mov.d1 r0, ss0` and `mov ms, r0` (the helper instructions) actually assemble.  If they don't, we need to fix the helper assembly in StreamStrategy.

- [ ] **Step 1: Generate all batches**

```bash
python3 tools/isa-test-gen.py --out /tmp/stream-verify
```

- [ ] **Step 2: Check manifest for stream_pair batches**

```bash
python3 -c "
import json
m = json.load(open('/tmp/stream-verify/manifest.json'))
stream = [b for b in m['batches'] if b.get('source_type') == 'stream_pair']
print(f'Stream pair batches: {len(stream)}')
for b in stream:
    print(f'  {b[\"instruction\"]} -> {b[\"producer_filename\"]}, {b[\"consumer_filename\"]}')
print(f'Testable: {m[\"testable_instructions\"]}')
print(f'Skipped: {m[\"skipped_instructions\"]}')
print('Skip reasons:')
for reason, count in sorted(m['skip_reasons'].items()):
    print(f'  {reason}: {count}')
"
```

- [ ] **Step 3: Verify helper instructions assemble**

```bash
echo "mov.d1 r0, ss0" | llvm-mc --triple=aie2 --filetype=obj -o /dev/null 2>&1
echo "mov ms, r0" | llvm-mc --triple=aie2 --filetype=obj -o /dev/null 2>&1
```

If either fails, the helper instruction needs to be changed.  Check error messages and find a working alternative.

- [ ] **Step 4: Assemble all stream pair files**

```bash
cd /tmp/stream-verify
FAIL=0
for f in batch_*_producer.s batch_*_consumer.s; do
    if ! llvm-mc --triple=aie2 --filetype=obj -o /dev/null "$f" 2>/dev/null; then
        echo "FAIL: $f"
        llvm-mc --triple=aie2 --filetype=obj -o /dev/null "$f" 2>&1
        FAIL=$((FAIL + 1))
    fi
done
echo "$FAIL failures"
```

- [ ] **Step 5: Verify skip reasons are clean**

Expected skip reasons after this work:
- `no output (nop/side-effect)`: 7
- `store instruction (needs validation)`: 1 (or similar)
- `no output operands detected`: 1 (or similar)
- `llvm-mc assembly error`: some (pre-existing)

Stream-related skip reasons should be GONE:
- `stream instruction` -- GONE
- `stream switch status read (hangs without streams)` -- GONE

- [ ] **Step 6: Report final coverage**

Print the final testable/skipped/total numbers.  Update the memory file if coverage has changed significantly.

- [ ] **Step 7: Commit any fixes**

If any assembly issues were found and fixed in steps 3-4, commit the fixes:
```bash
git add tools/isa-test-gen.py
git commit -m "fix(isa-harness): fix stream helper assembly for llvm-mc"
```
