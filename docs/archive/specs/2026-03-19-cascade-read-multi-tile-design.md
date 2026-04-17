# Cascade Read Multi-Tile ISA Harness Design

## Goal

Enable testing of the 3 cascade read instructions (`vmov $dst, SCD`,
`vmov.hi $dst, SCD`, `vmov.lo $dst, SCD`) by extending the ISA harness
to support multi-tile test points where a producer tile writes data into
the cascade link and a consumer tile reads it back.

## Architecture

A cascade pair occupies one column slot in the existing phase layout.
Within that column, two compute tiles cooperate:

- **Producer** at `tile(col, 3)`: loads known data from the host input
  buffer, enables crMCDEn, writes to MCD (cascade output south).
- **Consumer** at `tile(col, 2)`: enables crSCDEn, executes the cascade
  read instruction under test, stores the result to the host output buffer.
- **Link**: `aie.cascade_flow(%tile_col_3, %tile_col_2)` configures the
  cascade direction (south from row 3 to row 2).

The cascade pair coexists with normal single-tile batches in the same
phase.  Up to 4 column slots per phase; a cascade pair uses one slot.
Column counting for `npu1_Ncol` device target treats each cascade pair
as one column, same as a normal batch.

## Components

### 1. CascadeReadStrategy (`isa-test-gen.py`)

New strategy class that handles the 3 SCD read instructions.  Replaces
the current skip reason "cascade read (stalls without neighboring tile)".

**Skip removal**: SCD instructions are currently rejected at TWO points:
1. `CascadeStrategy.can_test()` -- returns `(False, "cascade read ...")`
   when asm contains `SCD`.
2. `classify_instruction()` lines 587-588 -- returns `("skipped",
   "cascade read ...")` when asm contains `SCD`.

The new `CascadeReadStrategy` must be placed BEFORE both in the strategy
list.  Since `classify_instruction()` is only called by
`ComputeStrategy.can_test()` (the last strategy), and
`CascadeReadStrategy` precedes it, the SCD check in
`classify_instruction()` becomes unreachable for SCD instructions.  No
code removal is strictly needed, but the SCD skip in
`classify_instruction()` should be removed for clarity.

**`can_test()`**: Accepts instructions with `SCD` in the asm_string.
Rejects everything else.

**`generate_cascade_pair()`**: New method (not `generate_test_point`)
that returns a dict with `producer_asm` and `consumer_asm` strings.

**Destination register handling**: All 3 instructions always target x0
(`generate_combos()` returns `{"dst": "x0"}`).  This avoids the
complexity of mapping arbitrary destination registers to the correct
wl/wh store pair.

- `vmov $dst, SCD`: full 512-bit read into x0, store via wl0+wh0 (64 bytes)
- `vmov.hi $dst, SCD`: high 256 bits into wh0, store via wh0 (32 bytes)
- `vmov.lo $dst, SCD`: low 256 bits into wl0, store via wl0 (32 bytes)

Consumer output size varies by instruction: 64 bytes for full vmov,
32 bytes for vmov.hi/vmov.lo.

**Pointer register convention**: The consumer function takes one memref
argument (output buffer), which the calling convention places in `p0`.
Consumer assembly uses `p0` for output stores, not `p1`.  The producer
function takes two memref arguments (input, output), so `p0` = input
and `p1` = output markers -- consistent with existing conventions.

Producer (generic, same for all 3 instructions):
```asm
test_kernel:
  // Load 64 bytes from input buffer into x0
  vlda wl0, [p0, #0]
  vlda wh0, [p0, #32]
  nop
  nop
  nop
  nop
  nop
  // Store before marker
  mov r14, #170
  st r14, [p1, #0]
  // Enable cascade output
  mov r14, #1
  mov crMCDEn, r14
  // Write to MCD
  vmov MCD, x0
  // Store after marker
  mov r14, #204
  st r14, [p1, #4]
  ret lr
  nop; nop; nop; nop
```

Consumer for `vmov $dst, SCD` (full read):
```asm
test_kernel:
  // Enable cascade input
  mov r14, #1
  mov crSCDEn, r14
  // Execute cascade read under test
  vmov x0, SCD
  // Store result to output buffer (p0 = output, single-arg function)
  vst wl0, [p0, #0]
  vst wh0, [p0, #32]
  ret lr
  nop; nop; nop; nop
```

Consumer for `vmov.lo $dst, SCD` (low half):
```asm
test_kernel:
  mov r14, #1
  mov crSCDEn, r14
  vmov.lo x0, SCD
  vst wl0, [p0, #0]
  ret lr
  nop; nop; nop; nop
```

**Buffer sizes:**
- Producer: input 64 bytes (source data), output 8 bytes (markers)
- Consumer (full vmov): input 0 bytes, output 64 bytes
- Consumer (vmov.hi/vmov.lo): input 0 bytes, output 32 bytes

**Manifest entry:**
```json
{
  "source_type": "cascade_pair",
  "producer_filename": "batch_NNN_producer.s",
  "consumer_filename": "batch_NNN_consumer.s",
  "producer_in_size": 64,
  "producer_out_size": 8,
  "consumer_in_size": 0,
  "consumer_out_size": 64,
  "instruction": "VMOV_mv_scd",
  "slot": "lda"
}
```

**Integration with `generate_all()`**: The main classification loop in
`generate_all()` currently collects `test_point_specs` for bin-packing
into mega-program batches.  Cascade pairs are NOT bin-packed.  Instead:

1. During classification, when `CascadeReadStrategy` matches, the test
   point is added to a separate `cascade_specs` list (not
   `test_point_specs`).
2. After the main batching loop, each cascade spec is processed
   individually:
   - Call `strategy.generate_cascade_pair(instr, regs)` to get producer
     and consumer assembly strings.
   - Wrap each in `build_mega_program()` (standalone programs with
     `test_kernel` symbol and `ret lr` epilogue).
   - Assemble both via llvm-mc into `.o` files.
   - Write a manifest entry with `source_type: "cascade_pair"`.
3. Cascade pairs get sequential batch indices after the normal batches.

### 2. MLIR Generation (`isa-multi-tile-gen.py`)

**`generate_phase_mlir()`** extended to inspect each batch's
`source_type`:

- `"assembly"` (existing): one tile at `(col, 2)`, one objectfifo pair
  (in + out), one `link_with` function.
- `"cascade_pair"` (new): two tiles at `(col, 2)` and `(col, 3)`,
  three objectfifos (producer in, producer out, consumer out), two
  `link_with` functions, plus `aie.cascade_flow`.

**Memtile routing for row 3**: Data from shim (row 0) to compute (row 3)
must pass through the memtile (row 1).  Following the pattern in
`mlir-aie/test/npu-xrt/cascade_flows/aie.mlir`, use explicit two-hop
objectfifos with `objectfifo.link`:

```mlir
%tile_N_0 = aie.tile(N, 0)   // shim
%tile_N_1 = aie.tile(N, 1)   // memtile (relay)
%tile_N_2 = aie.tile(N, 2)   // consumer (reads SCD)
%tile_N_3 = aie.tile(N, 3)   // producer (writes MCD)

aie.cascade_flow(%tile_N_3, %tile_N_2)

// Producer input: shim -> memtile -> row 3 (two-hop)
aie.objectfifo @of_prod_in_N_0(%tile_N_0, {%tile_N_1}, 2 : i32)
    : !aie.objectfifo<memref<16xi32>>
aie.objectfifo @of_prod_in_N_1(%tile_N_1, {%tile_N_3}, 2 : i32)
    : !aie.objectfifo<memref<16xi32>>
aie.objectfifo.link [@of_prod_in_N_0] -> [@of_prod_in_N_1] ([] [])

// Producer output: row 3 -> memtile -> shim (two-hop)
aie.objectfifo @of_prod_out_N_0(%tile_N_3, {%tile_N_1}, 2 : i32)
    : !aie.objectfifo<memref<2xi32>>
aie.objectfifo @of_prod_out_N_1(%tile_N_1, {%tile_N_0}, 2 : i32)
    : !aie.objectfifo<memref<2xi32>>
aie.objectfifo.link [@of_prod_out_N_0] -> [@of_prod_out_N_1] ([] [])

// Consumer output: row 2 -> shim (direct, same as existing)
aie.objectfifo @of_cons_out_N(%tile_N_2, {%tile_N_0}, 2 : i32)
    : !aie.objectfifo<memref<16xi32>>

func.func private @test_kernel_N_prod(memref<16xi32>, memref<2xi32>)
    attributes {link_with = "batch_NNN_producer.o"}
func.func private @test_kernel_N_cons(memref<16xi32>)
    attributes {link_with = "batch_NNN_consumer.o"}

aie.core(%tile_N_3) {
  %in = aie.objectfifo.acquire @of_prod_in_N_1(Consume, 1) ...
  %out = aie.objectfifo.acquire @of_prod_out_N_0(Produce, 1) ...
  func.call @test_kernel_N_prod(%in_elem, %out_elem) ...
  aie.objectfifo.release @of_prod_in_N_1(Consume, 1)
  aie.objectfifo.release @of_prod_out_N_0(Produce, 1)
  aie.end
}

aie.core(%tile_N_2) {
  %out = aie.objectfifo.acquire @of_cons_out_N(Produce, 1) ...
  func.call @test_kernel_N_cons(%out_elem) ...
  aie.objectfifo.release @of_cons_out_N(Produce, 1)
  aie.end
}
```

The consumer core takes only one memref argument (output buffer).  It
does not acquire an input objectfifo because its data arrives via the
cascade link.

The consumer output objectfifo goes directly to the shim (row 2 to
row 0) since row 2 is adjacent to the memtile -- this matches the
existing pattern used for normal single-tile batches.

### 3. Symbol Renaming

`prepare_phase_objects()` extended for cascade pairs:

- Normal batch: `test_kernel` -> `test_kernel_N`
- Cascade producer: `test_kernel` -> `test_kernel_N_prod`
- Cascade consumer: `test_kernel` -> `test_kernel_N_cons`

### 4. Buffer Layout

`compute_phase_buffer_layout()` extended to handle cascade pairs by
computing separate offsets for producer and consumer buffers.  A cascade
pair contributes to the shared host buffers:

- Input buffer: +64 bytes (producer source data)
- Output buffer: +8 bytes (producer markers) + 32 or 64 bytes
  (consumer result, depending on instruction)

### 5. Runtime Sequence

The runtime sequence adds DMA operations for all objectfifos belonging
to cascade pairs.  For the two-hop objectfifos, the DMA metadata
references the shim-side objectfifo name (the `_0` suffix variant):

1. Output DMAs first (producer markers via `@of_prod_out_N_1`, consumer
   result via `@of_cons_out_N`), with `issue_token = true`
2. Input DMAs (producer source data via `@of_prod_in_N_0`)
3. `dma_wait` for all output objectfifos

DMA IDs are assigned sequentially across all columns, with cascade
pairs consuming multiple IDs for their objectfifos.

## Testing

### Unit Tests (`test_isa_test_gen.py`)

New `TestCascadeReadStrategy` class covering:
- `can_test` accepts vmov/vmov.hi/vmov.lo with SCD
- `can_test` rejects MCD writes and non-cascade instructions
- Producer assembly contains vlda, nop sled, crMCDEn, vmov MCD, markers
- Consumer assembly contains crSCDEn, the instruction under test, vst
  to p0 (not p1)
- Correct buffer sizes (producer in=64, out=8; consumer varies by
  instruction)
- One combo per instruction, destination always x0

### MLIR Generation Tests

Verify `generate_phase_mlir()` produces valid MLIR for:
- A phase with one cascade pair
- A phase mixing one cascade pair and one normal batch
- Correct cascade_flow declaration
- Correct two-hop objectfifo routing through memtile for producer
- Correct link_with attributes and function signatures
- Consumer function has one argument (output only)

### Integration

After implementation, generate all batches and verify:
- All cascade pair `.s` files assemble with llvm-mc
- The generated MLIR compiles with aiecc.py
- Cascade read instructions move from skipped to testable

## Future: Stream Instructions

The same column-slot pattern applies to streams.  A stream pair would
use stream switch routing instead of `aie.cascade_flow`.  The multi-tile
MLIR generation and buffer layout infrastructure built here is directly
reusable.  Stream support is out of scope for this design.

## Files Modified

| File | Change |
|------|--------|
| `tools/isa-test-gen.py` | Add CascadeReadStrategy, remove SCD skip from classify_instruction, update generate_all with cascade_specs separation and cascade_pair manifest entries |
| `tools/isa-multi-tile-gen.py` | Handle cascade_pair source_type: two tiles per column, memtile declaration, two-hop objectfifos with objectfifo.link, cascade_flow, dual link_with functions |
| `tools/test_isa_test_gen.py` | Add TestCascadeReadStrategy unit tests |
