# Codebase Cleanup Sweep

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up the codebase after the 100% ISA accuracy push -- remove dead code, decompose monolithic files, deduplicate shared logic, and refresh stale documentation.

**Architecture:** Bottom-up cleanup: low-risk removals first (dead code, stale docs), then structural decomposition (vector.rs split, extract dedup), then polish (FFI docs, component docs refresh). Every task must leave `cargo test --lib` at 2659+ passing and ISA suite at 100%.

**Tech Stack:** Rust, cargo test, ISA test harness (`scripts/isa-test.sh`)

---

## Guiding Principles

1. **Protect 100%** -- run `cargo test --lib` after every structural change. Run ISA suite after completing each major task.
2. **One concern at a time** -- don't mix "move code" with "rewrite code". Extract first, simplify second.
3. **No behavior changes** -- this is a refactor. If a test starts failing, the refactor is wrong, not the test.
4. **Follow existing patterns** -- the codebase has conventions. Cleanup means making things consistent, not inventing new patterns.

## File Map (current state)

### Largest files (refactoring candidates)
| File | Lines | Issue |
|------|-------|-------|
| `src/interpreter/execute/vector.rs` | 6,003 | Monolith: 134 functions, duplicated wide/half dispatch |
| `src/interpreter/decode/decoder.rs` | 3,055 | Stable, well-structured -- leave alone |
| `src/interpreter/execute/memory.rs` | 3,044 | Large but cohesive -- leave alone for now |
| `src/interpreter/execute/vector_matmul.rs` | 3,005 | Duplicated extract functions, sparse overlap |
| `src/interpreter/execute/vmac_routing.rs` | 2,862 | Generated data tables -- read-only, leave alone |
| `src/interpreter/state/context.rs` | 2,793 | State management -- leave alone |
| `src/ffi/mod.rs` | 1,868 | Needs safety docs, not a split |

### Files to create (extraction targets)
| File | Purpose |
|------|---------|
| `src/interpreter/execute/vector_arith.rs` | Arithmetic ops extracted from vector.rs |
| `src/interpreter/execute/vector_compare.rs` | Comparison ops extracted from vector.rs |
| `src/interpreter/execute/vector_misc.rs` | Remaining ops (shuffle, select, etc.) from vector.rs |
| `src/interpreter/execute/element_extract.rs` | Shared element extraction utilities |

---

## Task 1: Dead Code Removal

**Files:**
- Modify: `src/interpreter/execute/vector.rs`
- Modify: `src/interpreter/execute/vector_matmul.rs`
- Modify: `src/interpreter/execute/vector_config.rs`
- Modify: `src/interpreter/execute/vector_permute.rs`
- Modify: any files with `#[allow(dead_code)]`

- [ ] **Step 1: Find all dead code markers**

```bash
grep -rn "#\[allow(dead_code)\]" src/ | grep -v test
grep -rn "// TODO\|// FIXME\|// HACK\|// XXX" src/
```

Document each instance: file, line, what it is, whether it should be removed or resolved.

- [ ] **Step 2: Remove genuinely dead functions**

For each `#[allow(dead_code)]` function: check if it's called anywhere with `grep -rn "function_name" src/`. If unused, delete it. If used only in tests, move the allow to the test module.

- [ ] **Step 3: Resolve or document TODOs**

For each TODO/FIXME: either fix it (if trivial) or convert to a doc comment explaining why it's deferred. Remove stale TODOs that reference completed work.

- [ ] **Step 4: Remove unused imports**

```bash
cargo fix --lib -p xdna-emu --allow-dirty
```

Review the changes, stage, and commit.

- [ ] **Step 5: Run tests**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib -- -q
```

Expected: 2659+ tests pass, 0 failures.

- [ ] **Step 6: Commit**

```bash
git add -A src/
git commit -m "chore: remove dead code, unused imports, and stale TODOs"
```

---

## Task 2: Extract Shared Element Utilities

**Files:**
- Create: `src/interpreter/execute/element_extract.rs`
- Modify: `src/interpreter/execute/vector_matmul.rs` (remove duplicated functions)
- Modify: `src/interpreter/execute/vector_pack.rs` (use shared utilities)
- Modify: `src/interpreter/execute/mod.rs` (add module)

- [ ] **Step 1: Identify the duplicated functions**

Read these functions and document their signatures:
- `vector_matmul.rs::extract_element_bytes()` (~line 543)
- `vector_matmul.rs::extract_element_512()` (~line 1347)
- `vector_pack.rs::extract_lane()` (~line 271)

Note the differences: input type (`[u8; 128]` vs `Vec512` vs `[u32; 8]`), signed/unsigned, bit widths.

- [ ] **Step 2: Write the shared module with tests**

Create `element_extract.rs` with a unified extraction interface. The key insight: all variants read N bits from a byte-addressable buffer with optional sign extension. Use a trait or generic function over a `AsRef<[u8]>` input.

Write tests that cover: 4-bit signed/unsigned, 8-bit, 16-bit, 32-bit extraction. Test edge cases: extraction at byte boundary, nibble boundary, sign extension of 0x8 in 4-bit.

- [ ] **Step 3: Run tests for the new module**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib element_extract -- -q
```

- [ ] **Step 4: Migrate callers one at a time**

Replace `extract_element_bytes()` calls in vector_matmul.rs with the shared function. Run tests. Then replace `extract_element_512()`. Run tests. Then replace `extract_lane()` in vector_pack.rs. Run tests after each.

- [ ] **Step 5: Delete the now-unused original functions**

Remove the old implementations from vector_matmul.rs and vector_pack.rs.

- [ ] **Step 6: Run full test suite**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib -- -q
```

- [ ] **Step 7: Commit**

```bash
git add src/interpreter/execute/element_extract.rs src/interpreter/execute/mod.rs \
       src/interpreter/execute/vector_matmul.rs src/interpreter/execute/vector_pack.rs
git commit -m "refactor: extract shared element extraction utilities"
```

---

## Task 3: Decompose vector.rs (6003 lines -> ~4 files)

This is the biggest structural change. The goal is NOT to rewrite logic -- just move functions to appropriate submodules. The dispatch in vector.rs stays; the implementations move.

**Files:**
- Modify: `src/interpreter/execute/vector.rs` (shrink to dispatch + imports)
- Create: `src/interpreter/execute/vector_arith.rs` (arithmetic: add, sub, mul, min, max, abs, neg, floor, ceil)
- Create: `src/interpreter/execute/vector_compare.rs` (comparisons: eq, ne, lt, ge, sel)
- Create: `src/interpreter/execute/vector_misc.rs` (shuffle, broadcast, insert, extract, shift, bitwise)
- Modify: `src/interpreter/execute/mod.rs` (add modules)

- [ ] **Step 1: Catalog vector.rs functions by category**

Read vector.rs and list every `fn` with its line range and semantic category. Group into:
- Arithmetic (VADD, VSUB, VMUL, VMIN, VMAX, VABS, VNEG, VFLOOR, VCEIL, etc.)
- Comparison (VEQ, VNE, VLT, VGE, VSEL, etc.)
- Matmul-adjacent (already in vector_matmul.rs -- skip)
- Everything else (VSHUFFLE, VBCAST, VINSERT, VEXTRACT, VSHIFT, bitwise)

Document which functions call which helpers (dependency graph).

- [ ] **Step 2: Extract arithmetic ops to vector_arith.rs**

Move arithmetic functions to the new file. Keep the `use` imports minimal -- only what each function needs. Update vector.rs to `pub(crate) use vector_arith::*` or call through the module.

Key: do NOT change any function signatures or logic. Pure move.

- [ ] **Step 3: Run tests**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib -- -q
```

All 2659+ must pass.

- [ ] **Step 4: Extract comparison ops to vector_compare.rs**

Same process. Move, don't modify.

- [ ] **Step 5: Run tests**

- [ ] **Step 6: Extract misc ops to vector_misc.rs**

Same process.

- [ ] **Step 7: Run tests**

- [ ] **Step 8: Verify vector.rs is now just dispatch + shared helpers**

After extraction, vector.rs should contain:
- The main `execute_vector()` dispatch function
- Shared helper functions used by multiple categories
- Module-level imports

Target: vector.rs under 1500 lines (dispatch + helpers).

- [ ] **Step 9: Run full test suite + ISA tests**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib -- -q
nice -n 19 ./scripts/isa-test.sh
```

ISA must remain 4815/4815 (100%).

- [ ] **Step 10: Commit**

```bash
git add src/interpreter/execute/
git commit -m "refactor: decompose vector.rs into arith/compare/misc submodules"
```

---

## Task 4: Consolidate Sparse Routing Logic

**Files:**
- Modify: `src/interpreter/execute/vmac_hw.rs`
- Modify: `src/interpreter/execute/vector_matmul.rs`

- [ ] **Step 1: Map the overlap**

Read both files and document where sparse routing logic is duplicated:
- `vmac_hw.rs::mask2sel()` vs any mask processing in vector_matmul.rs
- Group-to-inner_k mapping
- Column routing calculation
- Element selection dispatch

Identify which version is authoritative (vmac_hw.rs, since it has oracle-verified routing).

- [ ] **Step 2: Make vmac_hw routing functions pub(crate)**

Any routing functions that vector_matmul.rs reimplements should be made `pub(crate)` in vmac_hw.rs so vector_matmul.rs can call them.

- [ ] **Step 3: Replace duplicated logic in vector_matmul.rs**

One function at a time: replace the vector_matmul.rs version with a call to the vmac_hw.rs version. Run tests after each replacement.

- [ ] **Step 4: Run full test suite + ISA**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib -- -q
nice -n 19 ./scripts/isa-test.sh
```

- [ ] **Step 5: Commit**

```bash
git commit -am "refactor: consolidate sparse routing into vmac_hw module"
```

---

## Task 5: FFI Safety Documentation

**Files:**
- Modify: `src/ffi/mod.rs`

- [ ] **Step 1: Add module-level safety contract**

At the top of ffi/mod.rs, document:
- What invariants the C caller must uphold
- Lifetime requirements for handles
- Thread safety guarantees (or lack thereof)
- Error propagation model (thread_local storage)

- [ ] **Step 2: Add per-function safety comments**

For each `unsafe extern "C" fn`: add a `// SAFETY:` comment explaining why the unsafe is required and what invariants are assumed.

- [ ] **Step 3: Commit**

```bash
git commit -am "docs: add safety contracts to FFI boundary"
```

---

## Task 6: Documentation Refresh

**Files:**
- Modify: `ROADMAP.md`
- Modify: `docs/roadmap/phase1-core-accuracy.md`
- Modify: `.claude/components/interpreter.md`
- Modify: `.claude/components/parser.md`
- Modify: `.claude/components/tablegen.md`
- Modify: `.claude/components/visual.md`
- Delete or update: `docs/README.md`

- [ ] **Step 1: Update ROADMAP.md**

Update Phase 1 status to reflect 100% ISA accuracy. Update test counts. Mark VERIFIED items. Remove CLAIMED markers for things now proven.

- [ ] **Step 2: Update phase1-core-accuracy.md**

Refresh exit criteria. Document which are met, which remain. Update the confidence markers.

- [ ] **Step 3: Refresh component docs**

For each `.claude/components/*.md` file dated Feb 12:
- Read the corresponding source module
- Update line counts, function lists, architectural notes
- Remove references to approaches that have been abandoned
- Add notes about the sparse vmac pipeline, bf16 path, and other recent additions

- [ ] **Step 4: Clean up docs/README.md**

Either delete (if CLAUDE.md covers everything) or update to reflect current state. It's 4 months old.

- [ ] **Step 5: Commit**

```bash
git add ROADMAP.md docs/ .claude/components/
git commit -m "docs: refresh documentation to reflect 100% ISA accuracy milestone"
```

---

## Execution Order & Risk Assessment

| Task | Risk | Time | Dependencies |
|------|------|------|-------------|
| 1. Dead code removal | Low | 15 min | None |
| 2. Element extract dedup | Low | 30 min | None |
| 3. vector.rs decomposition | Medium | 60 min | Task 1 (fewer lines to move) |
| 4. Sparse routing consolidation | Medium | 30 min | Task 2 |
| 5. FFI safety docs | Zero | 15 min | None |
| 6. Documentation refresh | Zero | 20 min | All code tasks done |

Tasks 1, 2, and 5 are independent and can be parallelized. Task 3 should follow Task 1. Task 4 should follow Task 2. Task 6 should be last (so line counts and descriptions are final).

## Verification Gate

After all tasks complete:

```bash
# Unit tests
TMPDIR=/tmp/claude-1000 cargo test --lib -- -q
# Expected: 2659+ pass, 0 fail

# ISA accuracy
nice -n 19 ./scripts/isa-test.sh
# Expected: 4815/4815 (100.0%)

# No new warnings
cargo build --lib 2>&1 | grep -c "warning:"
# Expected: 0 (or same as before)
```
