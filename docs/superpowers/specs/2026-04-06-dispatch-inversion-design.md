# Vector Dispatch Inversion

**Goal:** Invert ownership so operations own their semantics and dispatch
is a trivial routing table. Make any contributor able to find where an
operation lives without grepping.

**Constraint:** 2,660 unit tests and 4,815/4,815 ISA test points must
remain green throughout. Each operation moves one at a time.

---

## Problem

`execute_half()` (500 lines, 43 arms) and `execute_wide()` (1,025 lines,
35 arms) are parallel dispatch functions where each match arm *is* the
operation's implementation. The operation's behavior is defined by its
position in the dispatch table, not by the operation itself.

This means:
- Dispatch owns the ISA -- you can't refactor dispatch without
  refactoring every operation's semantics.
- Two parallel tables duplicate routing for narrow/wide variants.
- Match arms inline 10-50 lines of operand handling, special-case
  detection, and result writing. Review diffs are enormous.
- New contributors cannot find where an operation lives without reading
  a 1,500-line match statement.

Two files already own their dispatch correctly: `vector_matmul.rs`
(`execute_matmul`) and `vector_semantic.rs` (`execute_vector_semantic`).
The other 37 compute functions have no dispatch ownership.

## Design

### Principle

**Operations own their semantics. Dispatch is a table.**

Each operation family gets an `execute_<family>()` method on `VectorAlu`
that owns the full lifecycle: width detection, operand reads, compute,
special-case handling, result writes. Dispatch calls it in one line.

### File Structure

```
execute/
  vector_dispatch.rs     -- single match table, ~150 lines
  vector_helpers.rs      -- shared VectorAlu methods (operand access, masks, bridges)
  vector_arith.rs        -- execute_add, execute_sub, execute_mul, execute_min,
                            execute_max, execute_neg, execute_neg_add,
                            execute_accumulate, execute_shl, execute_srl,
                            execute_sra, execute_abs_gtz, execute_neg_gtz,
                            execute_neg_ltz, execute_sub_lt, execute_sub_ge,
                            execute_maxdiff_lt, execute_floor
                            + existing compute functions (vector_add, etc.)
  vector_compare.rs      -- execute_cmp, execute_setge, execute_setlt,
                            execute_seteq, execute_maxlt, execute_minge,
                            execute_select
                            + existing compute functions
  vector_misc.rs         -- execute_shuffle, execute_broadcast, execute_extract,
                            execute_insert, execute_push, execute_align,
                            execute_bitwise_and/or/xor/not, execute_copy,
                            execute_clear
                            + existing compute functions
  vector_convert.rs      -- execute_convert, execute_floor_bf16
                            (extracted from vector.rs, new file)
  vector_srs.rs          -- execute_srs (promote existing code to own dispatch)
  vector_ups.rs          -- execute_ups (promote existing code to own dispatch)
  vector_pack.rs         -- execute_pack, execute_unpack
  vector_matmul.rs       -- execute_matmul (already correct, no change)
  ...other existing files unchanged...
```

`vector.rs` is deleted. Its contents are distributed:
- Dispatch logic -> `vector_dispatch.rs`
- Helper methods -> `vector_helpers.rs`
- SRS/Convert pipelines -> `vector_srs.rs` / `vector_convert.rs`
- Accumulator ops -> `vector_arith.rs` (execute_acc_add_sub, execute_acc_negate)
- Wide bridge helpers -> `vector_helpers.rs`
- Wide element manipulation -> `vector_misc.rs`
- Tests -> distributed to the file that owns each operation

### Dispatch Table

`vector_dispatch.rs` contains one function:

```rust
impl VectorAlu {
    pub fn execute(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
        let Some(semantic) = op.semantic else { return false };
        if !op.is_vector { return false }

        // Skip fused ops with memory operands (MemoryUnit handles those)
        if /* existing fused-op guard */ { return false }

        let et = op.element_type.unwrap_or(ElementType::Int32);

        match semantic {
            // Arithmetic -- simple binary ops use generic dispatcher
            Add                         => Self::execute_add(op, ctx, et),
            Sub                         => Self::execute_binary_elementwise(op, ctx, et, Self::vector_sub),
            Mul                         => Self::execute_binary_elementwise(op, ctx, et, Self::vector_mul),
            Min                         => Self::execute_binary_elementwise(op, ctx, et, Self::vector_min),
            Max                         => Self::execute_binary_elementwise(op, ctx, et, Self::vector_max),
            Neg                         => Self::execute_neg(op, ctx, et),
            NegAdd                      => Self::execute_neg_add(op, ctx, et),
            Shl                         => Self::execute_shl(op, ctx, et),
            Srl                         => Self::execute_srl(op, ctx, et),
            Sra                         => Self::execute_sra(op, ctx, et),
            AbsGtz                      => Self::execute_abs_gtz(op, ctx, et),
            NegGtz                      => Self::execute_neg_gtz(op, ctx, et),
            NegLtz                      => Self::execute_neg_ltz(op, ctx, et),
            SubLt                       => Self::execute_sub_lt(op, ctx, et),
            SubGe                       => Self::execute_sub_ge(op, ctx, et),
            MaxDiffLt                   => Self::execute_maxdiff_lt(op, ctx, et),
            Accumulate | AccumSub
            | AccumNegAdd | AccumNegSub => Self::execute_accumulate(op, ctx, et),

            // Comparison
            Cmp                         => Self::execute_cmp(op, ctx, et),
            SetGe                       => Self::execute_setge(op, ctx, et),
            SetLt                       => Self::execute_setlt(op, ctx, et),
            SetEq                       => Self::execute_seteq(op, ctx, et),
            MaxLt                       => Self::execute_maxlt(op, ctx, et),
            MinGe                       => Self::execute_minge(op, ctx, et),
            VectorSelect                => Self::execute_select(op, ctx, et),

            // Data movement
            Shuffle                     => Self::execute_shuffle(op, ctx, et),
            VectorBroadcast             => Self::execute_broadcast(op, ctx, et),
            VectorExtract               => Self::execute_extract(op, ctx, et),
            VectorInsert                => Self::execute_insert(op, ctx, et),
            VectorPushHi                => Self::execute_push(op, ctx, et),
            Align                       => Self::execute_align(op, ctx, et),
            Copy                        => Self::execute_copy(op, ctx, et),
            VectorClear                 => Self::execute_clear(op, ctx, et),
            And                         => Self::execute_binary_typeless(op, ctx, Self::vector_bitwise_and),
            Or                          => Self::execute_binary_typeless(op, ctx, Self::vector_bitwise_or),
            Xor                         => Self::execute_binary_typeless(op, ctx, Self::vector_bitwise_xor),
            Not                         => Self::execute_unary_typeless(op, ctx, Self::vector_bitwise_not),
            Pack                        => Self::execute_pack(op, ctx, et),
            Unpack                      => Self::execute_unpack(op, ctx, et),

            // Pipelines
            Srs                         => Self::execute_srs(op, ctx, et),
            Ups                         => Self::execute_ups(op, ctx, et),
            Convert                     => Self::execute_convert(op, ctx, et),

            // Matrix engine
            Mac | MatMul | MatMulSub | NegMul
            | NegMatMul | AddMac | SubMac
                                        => vector_matmul::execute_matmul(op, ctx),

            _                           => false,
        }
    }
}
```

No `execute_half` / `execute_wide` split. Each `execute_<op>` handles
both widths internally.

### Operation Function Signature

Every operation follows the same pattern:

```rust
impl VectorAlu {
    /// VADD / VADDSUB: element-wise vector addition.
    pub(super) fn execute_add(op: &SlotOp, ctx: &mut ExecutionContext, et: ElementType) -> bool {
        if op.is_wide_vector {
            // wide path: read 512-bit sources, compute, write 512-bit dest
        } else {
            // narrow path: read 256-bit sources, compute, write 256-bit dest
        }
        true
    }
}
```

The function:
1. Checks `op.is_wide_vector` (or accum_width, or other width indicators)
2. Reads operands using shared helpers from `vector_helpers.rs`
3. Calls the pure compute function (e.g., `Self::vector_add()`)
4. Writes results using shared helpers
5. Returns `true`

For operations where the wide path is just "do the narrow thing on both
halves," the function calls the existing bridge helpers from
`vector_helpers.rs`.

### Helper Functions (vector_helpers.rs)

Shared `VectorAlu` methods used by all operation files:

**Operand access:**
- `get_vector_source()`, `get_two_vector_sources()`
- `get_wide_vec_source()`, `get_two_wide_vec_sources()`
- `read_vector_operand()`
- `get_scalar_source()`, `get_scalar_source_64()`, `get_nth_scalar_source()`
- `get_lane_index()`, `get_shift_amount()`
- `get_config_register()`
- `get_acc_source()`, `get_acc_dest()`

**Result writing:**
- `write_vector_dest()`, `write_wide_vec_dest()`, `write_wide_acc_dest()`
- `write_scalar_dest()`, `write_scalar_dest_wide()`
- `write_cmp_dest()`, `write_cmp_dest_wide()`

**Comparison masks:**
- `condense_comparison_mask()`, `pack_comparison_flags()`

**Wide bridges:**
- `wide_element_wise_unary()`, `wide_element_wise_binary()`
- `execute_wide_fallback()`, `increment_vector_regs()`

**Wide element manipulation:**
- `wide_vector_push()`, `extract_wide_element()`
- `insert_wide_element()`, `insert_wide_element_64()`
- `wide_vector_shift()`

**Conversion utilities:**
- `bf16_to_f32()`, `f32_to_bf16()`

### Test Distribution

Tests move to the file that owns the operation they test:
- `test_vector_add_*` -> `vector_arith.rs`
- `test_vector_cmp` -> `vector_compare.rs`
- `test_vector_shuffle_*` -> `vector_misc.rs`
- `test_vector_srs_*` -> `vector_srs.rs`
- `test_vconv_*`, `test_vfloor_*` -> `vector_convert.rs`
- `test_wide_ups_*`, `test_wide_srs_*` -> `vector_srs.rs` / `vector_ups.rs`
- `test_wide_setge_*`, `test_wide_setlt_*` -> `vector_compare.rs`
- `test_wide_vector_push_*`, `test_*_wide_element_*` -> `vector_misc.rs`
- `test_wide_vector_align_*` -> `vector_misc.rs`
- `test_vadd_f_*` -> `vector_arith.rs` (float arithmetic)
- `test_vector_mov` -> `vector_misc.rs` (Copy)

### Execution Order

One operation at a time. After each move: `cargo test --lib`.

1. Create `vector_helpers.rs` -- extract all helper methods from vector.rs.
   This is zero behavioral change (same methods, same struct, different file).
2. Create `vector_dispatch.rs` -- start with a copy of `execute()` that
   still calls `execute_half`/`execute_wide`. Verify tests pass.
3. Pick one simple operation (e.g., Add). Write `execute_add` in
   `vector_arith.rs`. Update dispatch to call it. Delete the Add arms
   from execute_half and execute_wide. Verify.
4. Repeat for each operation family, simplest first.
5. Once all operations are extracted, execute_half and execute_wide are
   empty. Delete them. Delete vector.rs.
6. Run ISA suite as final gate.

### What Does NOT Change

- `VectorAlu` struct (remains a unit struct, methods on it)
- Pure compute functions (`vector_add`, `vector_sub`, etc.) -- already extracted
- `vector_matmul.rs` -- already owns its dispatch
- `vector_permute.rs` -- shuffle routing tables, called by execute_shuffle
- `vmac_hw.rs`, `vmac_routing.rs` -- hardware pipeline, called by matmul
- `vector_float.rs` -- float helpers, called by operations
- `vector_config.rs` -- MAC config parsing

### Success Criteria

- `vector.rs` is deleted
- `vector_dispatch.rs` is under 200 lines
- Every match arm is a single function call
- No operation's semantics live in the dispatch file
- 2,660+ unit tests pass
- 4,815/4,815 ISA test points pass
- No new files beyond: `vector_dispatch.rs`, `vector_helpers.rs`,
  `vector_convert.rs` (everything else already exists)
