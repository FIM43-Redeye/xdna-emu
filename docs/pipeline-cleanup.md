# Interpretation Pipeline Cleanup Tracker

Post-Operation-removal review (2026-03-01). Issues found by tracing the
full decode -> execute -> timing pipeline.

## Status Key

- `[ ]` Not started
- `[~]` In progress
- `[x]` Done
- `[-]` Deferred / won't fix

---

## P0 -- Correctness Bugs

### 1. StreamOps never called in execution chain
- **File**: `src/interpreter/execute/cycle_accurate.rs`
- **Problem**: `StreamOps::execute()` is implemented and tested but never
  called from `execute_slot_with_mem_locks()`. StreamRead/StreamWrite
  instructions silently do nothing at runtime.
- **Fix**: Insert StreamOps call between CascadeOps and ControlUnit,
  matching the existing CascadeOps pattern (match on StreamResult,
  convert Stall to WaitStream).
- **Status**: `[x]` -- Wired in, matching CascadeOps pattern.

### 2. `is_control_flow()` misses BrCond
- **File**: `src/interpreter/bundle/slot.rs`
- **Problem**: Conditional branches (jnz, jz) not classified as control
  flow. `has_control_flow()` and `control_op()` return false for them.
- **Fix**: Add `SemanticOp::BrCond` to the match. Consider Halt/Done.
- **Status**: `[x]` -- BrCond added.

---

## P1 -- Dead Code / Duplication

### 3. Delete ScalarAlu stub
- **File**: `src/interpreter/execute/scalar.rs`
- **Problem**: `execute()` returns false unconditionally. Three orphaned
  helper methods. Duplicate `ControlReg` handling (also in semantic.rs).
  Tests actually call `execute_semantic()`, not ScalarAlu.
- **Fix**: Delete struct and helpers. Move tests to semantic.rs test
  block. Remove ScalarAlu call from cycle_accurate.rs dispatch chain.
- **Status**: `[x]` -- Module deleted. 5 unique tests moved to
  semantic.rs. ScalarAlu removed from dispatch chain and all docs.

### 4. Delete tombstone stubs in semantic.rs
- **File**: `src/interpreter/execute/semantic.rs`
- **Problem**: `execute_load`, `execute_store`, `execute_branch`,
  `execute_branch_cond` always return false. Six wasted function calls
  per branch/memory instruction.
- **Fix**: Remove Load/Store/Br/BrCond from execute_semantic() match.
  Let them fall to the `_ =>` arm. Delete the four stub functions.
- **Status**: `[x]` -- Match arms and functions deleted.

### 5. Extract shared read_operand / write_dest
- **Files**: `semantic.rs`, `stream.rs`
- **Problem**: Nearly identical `read_operand()` / `write_dest()` in
  multiple modules. Adding a new operand type means editing all of them.
- **Fix**: Make semantic.rs versions `pub(super)`. Other modules import
  them. Eliminates duplicate implementations.
- **Status**: `[x]` -- semantic.rs helpers are now `pub(super)`.
  stream.rs uses shared versions. scalar.rs deleted (issue 3).
  Also fixed latent bug: stream.rs wrote PointerReg directly instead
  of using deferred pipeline writes.

---

## P2 -- Architectural Issues

### 6. PointerAdd/PointerMov misplaced in MemoryUnit
- **Files**: `memory.rs`, `semantic.rs`
- **Problem**: Pure register ops in MemoryUnit. Split placement with
  partial pointer handling in `execute_add()`.
- **Fix**: Consolidate pointer arithmetic in execute_semantic(). Move
  PointerAdd/PointerMov handlers from memory.rs. Merge the special
  padda/paddb tied-register pattern in execute_add().
- **Status**: `[ ]`

### 7. Dead hazard/stall infrastructure
- **File**: `cycle_accurate.rs`
- **Problem**: `check_hazards()`, `check_memory_conflict()`,
  `execute_slot()` are `#[allow(dead_code)]`. Stall block always
  computes zero. `record_writes()` / `record_memory_access()` update
  state never read back.
- **Fix**: Delete or cfg-gate the dead methods. Remove the dead stall
  block. Keep record_* if hazard model is planned soon, otherwise gate.
- **Status**: `[ ]`

---

## P3 -- Cleanup Opportunities

### 8. Dead `opcodes` module
- **File**: `src/interpreter/decode/mod.rs`
- **Problem**: Pre-TableGen manual decoder constants (NOP_ZERO, NOP_AIE,
  high_nibble::*). Never referenced. Also `extract_reg()`/`extract_imm()`
  may be unused.
- **Fix**: Delete module. Confirm helper functions have no callers.
- **Status**: `[ ]`

### 9. OperationKey indirection layer
- **File**: `src/interpreter/timing/latency.rs`
- **Problem**: 30-variant enum that just maps from SemanticOp. Adding a
  new SemanticOp requires editing key_from_semantic() (127 lines).
- **Fix**: Replace with HashMap<(SemanticOp, bool), OperationTiming> or
  a packed key. Eliminates the intermediate enum entirely.
- **Status**: `[ ]`

### 10. Duplicate doc comment on get_branch_target()
- **File**: `src/interpreter/execute/control.rs`
- **Problem**: Same doc comment appears twice (copy-paste artifact).
- **Fix**: Delete the duplicate block.
- **Status**: `[x]` -- Duplicate removed.
