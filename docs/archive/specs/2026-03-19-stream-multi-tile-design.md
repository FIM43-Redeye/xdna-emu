# Stream Multi-Tile ISA Harness Design

## Goal

Enable testing of stream-related instructions by extending the ISA
harness with a StreamStrategy class and stream_pair multi-tile support.
This closes the coverage gap from 573/606 (94.6%) toward ~98%.

## Instruction Inventory

### Stream Write Instructions (slot=st)

All write scalar data to `ms` (master stream output port).  The ISA
JSON contains 18 definitions across three patterns (6 per pattern):
base, non-blocking, tlast-mnemonic, nb+tlast, and two doTlast_reg
variants.

**Pattern A -- Scalar transfer (6 definitions):**
- `MOV_mv_scl2ms`: `mov ms, $src` (blocking)
- `MOV_NB_mv_scl2ms`: `mov.nb ms, $src` (non-blocking)
- `MOV_TLAST_mv_scl2ms`: `mov.tlast ms, $src` (blocking + TLAST)
- `MOV_NB_TLAST_mv_scl2ms`: `mov.nb.tlast ms, $src` (nb + TLAST)
- `MOV_mv_scl2ms_doTlast_reg`: `mov ms, $src, $tlast` (blocking, register TLAST)
- `MOV_NB_mv_scl2ms_doTlast_reg`: `mov.nb ms, $src, $tlast` (nb, register TLAST)

**Pattern B -- Packet header (6 definitions):**
- `MOV_mv_ph2ms`: `mov.ph ms, $id, $pcktType`
- `MOV_NB_mv_ph2ms`: `mov.ph.nb ms, $id, $pcktType`
- `MOV_TLAST_mv_ph2ms`: `mov.ph.tlast ms, $id, $pcktType`
- `MOV_NB_TLAST_mv_ph2ms`: `mov.ph.nb.tlast ms, $id, $pcktType`
- `MOV_mv_ph2ms_doTlast_reg`: `mov.ph ms, $id, $pcktType, $tlast`
- `MOV_NB_mv_ph2ms_doTlast_reg`: `mov.ph.nb ms, $id, $pcktType, $tlast`

**Pattern C -- Configurable packet header (6 definitions):**
- `MOV_mv_cph2ms`: `mov.cph ms, $addr, $nw, $op, $id`
- `MOV_NB_mv_cph2ms`: `mov.cph.nb ms, $addr, $nw, $op, $id`
- `MOV_TLAST_mv_cph2ms`: `mov.cph.tlast ms, $addr, $nw, $op, $id`
- `MOV_NB_TLAST_mv_cph2ms`: `mov.cph.nb.tlast ms, $addr, $nw, $op, $id`
- `MOV_mv_cph2ms_doTlast_reg`: `mov.cph ms, $addr, $nw, $op, $id, $tlast`
- `MOV_NB_mv_cph2ms_doTlast_reg`: `mov.cph.nb ms, $addr, $nw, $op, $id, $tlast`

Note: Some mnemonics in STREAM_MNEMONICS (mov.nb.tlast, mov.ph.tlast,
mov.cph.tlast) may not have corresponding ISA JSON entries.  If they
appear nowhere in the instruction database, they produce zero test
points (harmless).  The actual count of testable stream writes is
determined by what the ISA JSON contains.

**doTlast_reg handling**: The 6 `_doTlast_reg` variants have `$tlast`
in their asm_string.  The `$tlast` operand may or may not appear in
the operands list.  If missing from operands, `_substitute_asm` will
leave it unsubstituted.  StreamStrategy handles this by adding
`"tlast": "r1"` to the combo dict for doTlast_reg variants.  The
generator sets `r1` to 0 or 1 before the stream instruction.

### Stream Read Instructions (6 instructions, slot=mv)

Delayed register moves where source can be `ss0` (stream slave port 0).

- `MOV_D1` through `MOV_D6`: `mov.d1 $dst, $src` through `mov.d6 $dst, $src`

Source operand is `MvSclSrc` composite (7-bit), which includes stream
slave ports.  These are dual-natured:
- With non-stream source (r0, r1, etc.): plain delayed scalar moves,
  handled by ComputeStrategy.
- With stream source (ss0): stream reads, handled by StreamStrategy.

### SS Status Read (2 instructions, slot=lda)

- `MOV_mv_ss2scl`: `mov $mRa, SS` -- blocking read of stream switch status.
- `MOV_NB_mv_ss2scl`: `mov.nb $mRa, SS` -- non-blocking read.

Both hang or return meaningless data without active streams.

**Mnemonic collision**: `MOV_NB_mv_ss2scl` shares mnemonic `mov.nb`
with `MOV_NB_mv_scl2ms` (stream write).  StreamStrategy's `can_test()`
must check for `SS` in asm_string BEFORE checking the mnemonic against
STREAM_MNEMONICS to route SS reads to ss_status mode correctly.

## Architecture

### Two-Tile Setup

Same physical layout as cascade pairs: two compute tiles in one column
slot, rows 2 and 3.  The routing declaration is the only difference.

```
Row 3: tile(col, 3) -- producer (writes to ms)
Row 2: tile(col, 2) -- consumer (reads from ss0)
Row 1: tile(col, 1) -- memtile (relay for row 3 data)
Row 0: tile(col, 0) -- shim (host DMA)

Stream route: aie.flow(%tile_col_3, Core : 0, %tile_col_2, Core : 0)
```

The stream pair coexists with normal single-tile batches and cascade
pairs in the same phase.  Column counting treats each stream pair as
one column slot, same as cascade pairs and normal batches.

### Three Internal Modes

StreamStrategy operates in three modes based on the instruction:

| Mode | Test tile | Helper tile | Instructions |
|------|-----------|-------------|--------------|
| stream_write | Producer (row 3) | Consumer (row 2) | all ms-write variants |
| stream_read | Consumer (row 2) | Producer (row 3) | mov.d1-d6 with ss source |
| ss_status | Consumer (row 2) | Producer (row 3) | mov/mov.nb $r, SS |

For ss_status mode, the test tile is placed at row 2 (consumer side of
the flow).  The helper at row 3 writes a dummy value to ms to
establish an active flow, ensuring the SS register reflects real stream
state.  The test tile at row 2 has an inbound stream port from the
flow, so its SS register has meaningful status to read.

### Known-Good Helper Instructions

The helper tile uses known-good instructions to feed/drain the stream:

- **Helper consumer** (drains stream writes): Uses `mov.d1 r0, ss0`
  (delayed read from stream port 0, 1-cycle delay).  This is itself a
  stream read instruction, but when testing stream writes, the
  consumer is the HELPER side -- if the write fails, no data arrives
  and we detect the failure via missing consumer output.

- **Helper producer** (feeds stream reads and SS status): Uses
  `mov ms, r0` (blocking write to master stream port 0).  This is the
  `MOV_mv_scl2ms` instruction (mnemonic `mov`, not in STREAM_MNEMONICS
  frozenset).  Note: this instruction is ALSO tested as a stream write,
  so it gets validated independently.

The circular dependency is practical, not theoretical: each test pair
has exactly one known-good side and one test side.  If both failed
simultaneously, we'd see zero output (detectable).

**Assembly verification**: Both `mov.d1 r0, ss0` and `mov ms, r0` must
be verified with llvm-mc during implementation.  If either fails to
assemble, fall back to an alternative encoding (e.g., different
register, different delay variant).

## Components

### 1. StreamStrategy (`isa-test-gen.py`)

New strategy class handling all stream-related instructions.

**STREAM_MNEMONICS update**: Remove `mov.d1` through `mov.d6` from the
frozenset.  Add `mov` to handle the blocking `MOV_mv_scl2ms` -- BUT
only when asm_string contains `ms`.  Since `mov` is a common mnemonic,
the can_test() logic must check asm_string content, not just the
mnemonic.

**`can_test()` logic** (order matters):

```python
def can_test(self, instr):
    asm = instr.get("asm_string", "")
    mnemonic = instr.get("mnemonic", "")

    # 1. SS status reads -- check BEFORE mnemonic matching
    #    (mov.nb collision: both SS read and ms write share mov.nb)
    if "SS" in asm and "mov" in mnemonic:
        return (True, "")

    # 2. Stream writes -- ms in asm_string
    if "ms" in asm and mnemonic in STREAM_MNEMONICS_EXTENDED:
        return (True, "")

    # 3. Blocking stream write (mnemonic=mov, not in frozenset)
    if "ms," in asm and mnemonic == "mov":
        return (True, "")

    # 4. Stream reads -- mov.d* with composite source
    if re.match(r"mov\.d[1-6]", mnemonic):
        # Check if source operand supports stream registers
        for op in instr.get("operands", []):
            if op["name"] == "src" and op.get("register_kind") == "MvSclSrc":
                return (True, "")

    return (False, "not a stream instruction")
```

**`generate_combos()`**: Returns operand combinations appropriate to
each instruction pattern:

- Pattern A (scalar write): `{"src": "r0"}`.
- Pattern A doTlast_reg: `{"src": "r0", "tlast": "r1"}`.
- Pattern B (packet header): `{"id": "r0", "pcktType": "0"}`.
- Pattern B doTlast_reg: `{"id": "r0", "pcktType": "0", "tlast": "r1"}`.
- Pattern C (config packet): `{"addr": "m0", "nw": "0", "op": "0",
  "id": "r0"}`.  Note: `m0` is a modifier_m register; the assembly
  template zeroes it before use to avoid undefined post-modify behavior.
- Pattern C doTlast_reg: same + `"tlast": "r1"`.
- mov.d* stream: `{"dst": "r0", "src": "ss0"}`.
- SS status: `{"mRa": "r0"}`.

One combo per instruction -- minimal verification that the encoding and
stream path work.

**`generate_stream_pair()`**: New method (parallel to cascade's
`generate_cascade_pair()`) that returns a dict with `producer_asm` and
`consumer_asm` strings.  The method inspects the instruction's mode to
determine which tile gets the test instruction and which gets the
known-good helper.

**`compute_output_size()`**: Returns combined output for both tiles.

### 2. Assembly Templates

**Stream write producer** (test tile at row 3, single-arg function,
p0 = output markers):
```asm
test_kernel:
  // Load test value into source register
  mov r0, #0xBEEF
  // (For doTlast_reg variants: mov r1, #1)
  // (For cph variants: mov m0, #0)
  // Store before marker
  mov r14, #170
  st r14, [p0, #0]
  // Execute stream write under test
  mov.nb ms, r0           // <-- substituted per instruction
  // Store after marker
  mov r14, #204
  st r14, [p0, #4]
  ret lr
  nop; nop; nop; nop
```

Note: producer function takes ONE memref arg (output buffer for
markers).  The calling convention maps it to `p0`.  There is no input
objectfifo (test values are immediates).

**Stream write consumer** (helper at row 2, single-arg, p0 = output):
```asm
test_kernel:
  // Blocking read from stream port 0 (known-good helper)
  mov.d1 r0, ss0
  nop                      // 1-cycle delay for mov.d1 result
  // Store received value to output buffer
  st r0, [p0, #0]
  ret lr
  nop; nop; nop; nop
```

**Stream read producer** (helper at row 3, single-arg, p0 = output markers):
```asm
test_kernel:
  // Load test value
  mov r0, #0xBEEF
  // Store before marker
  mov r14, #170
  st r14, [p0, #0]
  // Write to stream (known-good helper)
  mov ms, r0
  // Store after marker
  mov r14, #204
  st r14, [p0, #4]
  ret lr
  nop; nop; nop; nop
```

**Stream read consumer** (test tile at row 2, single-arg, p0 = output):
```asm
test_kernel:
  // Execute stream read under test
  mov.d1 r0, ss0          // <-- substituted per instruction
  // NOP sled for pipeline delay (mov.dN has N-cycle delay)
  nop                      // <-- N nops for mov.dN
  // Store received value to output buffer
  st r0, [p0, #0]
  ret lr
  nop; nop; nop; nop
```

The NOP count after the instruction matches the delay value: mov.d1
gets 1 nop, mov.d6 gets 6 nops.  This ensures the result is available
in r0 before the store.

**SS status read** (test tile at row 2, single-arg, p0 = output):
```asm
test_kernel:
  // Store before marker
  mov r14, #170
  st r14, [p0, #0]
  // Read stream switch status (needs active flow)
  mov r0, SS               // <-- substituted (mov or mov.nb)
  // Store after marker
  mov r14, #204
  st r14, [p0, #4]
  // Store status value
  st r0, [p0, #8]
  ret lr
  nop; nop; nop; nop
```

**SS status helper** (producer at row 3, single-arg, p0 = output markers):
```asm
test_kernel:
  // Write dummy value to establish active stream flow
  mov r0, #0xDEAD
  mov r14, #170
  st r14, [p0, #0]
  mov ms, r0
  mov r14, #204
  st r14, [p0, #4]
  ret lr
  nop; nop; nop; nop
```

### 3. Buffer Sizes

| Mode | Producer in | Producer out | Consumer out |
|------|------------|-------------|-------------|
| stream_write | 0 | 8 (markers) | 4 (received word) |
| stream_read | 0 | 8 (markers) | 4 (received word) |
| ss_status | 0 | 8 (markers) | 12 (markers + status value) |

Producer input is 0 bytes for all modes -- test values are loaded via
immediate, not from a host buffer.  No producer input objectfifo.

For ss_status mode, the consumer (test tile) outputs 12 bytes: 4 bytes
before marker + 4 bytes after marker + 4 bytes status value.  The
producer (helper) outputs 8 bytes (markers only).

### 4. Manifest Entry

```json
{
  "source_type": "stream_pair",
  "batch_index": N,
  "producer_filename": "batch_NNN_producer.s",
  "consumer_filename": "batch_NNN_consumer.s",
  "producer_in_size": 0,
  "producer_out_size": 8,
  "consumer_out_size": 4,
  "instruction": "MOV_NB_mv_scl2ms",
  "test_count": 1,
  "filename": "batch_NNN_producer.s",
  "tests": [{"instruction": "MOV_NB_mv_scl2ms", "slot": "st"}],
  "in_size": 0,
  "out_size": 12
}
```

The `filename` and `tests` keys provide backward compatibility with
existing TestGenerateAll tests that iterate over these fields.

### 5. Integration with generate_all()

**Dual-testing for mov.d***: The classification loop in `generate_all()`
must process mov.d* instructions TWICE -- once via StreamStrategy (for
stream-source combos) and once via ComputeStrategy (for non-stream
combos).  The current `classify_with_strategies()` breaks after the
first match, so it cannot handle this directly.

Implementation approach: after the main classification loop, perform a
second pass over all instructions to find mov.d* instructions that
ComputeStrategy can also handle.  For each match, add to
`test_point_specs` (for bin-packing into normal batches).

```python
# In generate_all():
cascade_specs = []
stream_specs = []

for instr in instructions:
    for strategy in STRATEGIES:
        ok, reason = strategy.can_test(instr)
        if ok:
            if isinstance(strategy, CascadeReadStrategy):
                cascade_specs.append((instr, strategy))
            elif isinstance(strategy, StreamStrategy):
                stream_specs.append((instr, strategy))
            else:
                test_point_specs.append(...)
            break

# Second pass: mov.d* also get non-stream combos via ComputeStrategy
for instr in instructions:
    mnemonic = instr.get("mnemonic", "")
    if re.match(r"mov\.d[1-6]", mnemonic):
        compute = ComputeStrategy()
        ok, reason = compute.can_test(instr)
        if ok:
            test_point_specs.append(...)

# After normal batching loop:
# 1. Process cascade_specs (existing)
# 2. Process stream_specs (new, identical pattern)
```

Stream specs follow the same processing as cascade specs: each stream
spec is processed individually, `generate_stream_pair()` returns
producer and consumer assembly, both are wrapped in
`build_mega_program()`, assembled via llvm-mc, and written with a
`stream_pair` manifest entry.

### 6. MLIR Generation (`isa-multi-tile-gen.py`)

**Source type dispatch** in `generate_phase_mlir()`:

- `"assembly"` (existing): one tile, one objectfifo pair, one link_with.
- `"cascade_pair"` (existing): two tiles + `aie.cascade_flow`.
- `"stream_pair"` (new): two tiles + `aie.flow`.

**Helper predicate**:
```python
def _is_stream(batch: dict) -> bool:
    return batch.get("source_type") == "stream_pair"
```

**Stream pair MLIR** (no producer input objectfifo):

```mlir
%tile_N_0 = aie.tile(N, 0)   // shim
%tile_N_1 = aie.tile(N, 1)   // memtile (relay)
%tile_N_2 = aie.tile(N, 2)   // consumer
%tile_N_3 = aie.tile(N, 3)   // producer

// Stream route (replaces aie.cascade_flow)
aie.flow(%tile_N_3, Core : 0, %tile_N_2, Core : 0)

// Producer output: row 3 -> memtile -> shim (two-hop, markers)
aie.objectfifo @of_prod_out_N_0(%tile_N_3, {%tile_N_1}, 2 : i32)
    : !aie.objectfifo<memref<2xi32>>
aie.objectfifo @of_prod_out_N_1(%tile_N_1, {%tile_N_0}, 2 : i32)
    : !aie.objectfifo<memref<2xi32>>
aie.objectfifo.link [@of_prod_out_N_0] -> [@of_prod_out_N_1] ([] [])

// Consumer output: row 2 -> shim (direct)
aie.objectfifo @of_cons_out_N(%tile_N_2, {%tile_N_0}, 2 : i32)
    : !aie.objectfifo<memref<1xi32>>

func.func private @test_kernel_N_prod(memref<2xi32>)
    attributes {link_with = "batch_NNN_producer.o"}
func.func private @test_kernel_N_cons(memref<1xi32>)
    attributes {link_with = "batch_NNN_consumer.o"}

aie.core(%tile_N_3) {
  %out = aie.objectfifo.acquire @of_prod_out_N_0(Produce, 1) ...
  func.call @test_kernel_N_prod(%out_elem) ...
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

Key differences from cascade pair:
- No producer input objectfifo (test values are immediates).
- `aie.flow` instead of `aie.cascade_flow`.
- Producer function takes one memref arg (output only), not two.
- Consumer output memref size varies by mode.

For ss_status mode, the consumer output is `memref<3xi32>` (markers +
status value) and the MLIR is otherwise identical.

**Shared infrastructure**: The two-hop objectfifo pattern (producer
output via memtile) and the direct objectfifo pattern (consumer output
to shim) are shared between cascade and stream pairs.  The code should
be factored so that `_emit_stream_objectfifos()` and
`_emit_stream_cores()` reuse the underlying helpers.

### 7. Symbol Renaming

Same as cascade in `prepare_phase_objects()`:
- Producer: `test_kernel` -> `test_kernel_N_prod`
- Consumer: `test_kernel` -> `test_kernel_N_cons`

### 8. Buffer Layout

`compute_phase_buffer_layout()` extended for stream pairs.  A stream
pair contributes to the shared host buffers:

- Input buffer: +0 bytes (no producer input data)
- Output buffer: producer_out_size + consumer_out_size

### 9. Runtime Sequence

For stream pairs, the runtime sequence adds DMA operations:

1. Output DMAs first:
   - Producer markers via `@of_prod_out_N_1` (issue_token = true)
   - Consumer result via `@of_cons_out_N` (issue_token = true)
2. No input DMAs (no producer input objectfifo)
3. `dma_wait` for all output objectfifos

DMA metadata references shim-side objectfifo names (the `_1` suffix
for two-hop producer output, the direct name for consumer output).

## Coverage Impact

The exact count depends on which ISA JSON definitions StreamStrategy
successfully claims.  Based on the ISA JSON audit:

| Category | ISA JSON definitions | New test points |
|----------|---------------------|-----------------|
| Stream writes (all patterns) | 15-18 | +15 to +18 |
| mov.d* stream combos | 6 | +6 |
| mov.d* non-stream (ComputeStrategy) | 6 | +6 |
| SS status reads | 2 | +2 |
| **Estimated total** | | **+29 to +32** |

Some STREAM_MNEMONICS entries may not have corresponding ISA JSON
definitions (producing zero test points).  Some ISA entries share
mnemonics (e.g., base + doTlast_reg variants of the same mnemonic).
The precise coverage number will be determined by running the generator
after implementation.

Remaining untestable after this work: 7 NOPs, 1 store blocked by
llvm-mc issue 858, 1 no-output-operands edge case.

## Testing

### Unit Tests (`test_isa_test_gen.py`)

New `TestStreamStrategy` class covering:
- `can_test` accepts stream write instructions (all patterns)
- `can_test` accepts mov.d* with stream source operand
- `can_test` accepts SS status read (both blocking and non-blocking)
- `can_test` rejects non-stream instructions
- `can_test` routes SS reads to ss_status mode BEFORE mnemonic check
  (mov.nb collision test)
- Producer assembly uses p0 for output (single-arg function)
- Consumer assembly uses p0 for output (single-arg function)
- Stream read consumer has correct NOP sled for pipeline delay
- doTlast_reg combos include $tlast operand
- cph combos include modifier_m register zeroing
- Correct buffer sizes per mode
- One combo per instruction

### mov.d* Dual Coverage Tests

Verify that mov.d1-d6:
- Are picked up by StreamStrategy for stream-source combos
- Are picked up by ComputeStrategy for non-stream combos
  (second pass in generate_all)
- Produce test points in both normal batches AND stream pair batches

### MLIR Generation Tests

Verify `generate_phase_mlir()` produces valid MLIR for:
- A phase with one stream pair
- A phase mixing stream pair, cascade pair, and normal batch
- Correct `aie.flow` declaration (not cascade_flow)
- No producer input objectfifo (unlike cascade)
- Producer function has one argument (output only)
- Consumer function has one argument (output only)
- Correct runtime sequence (no input DMA for stream pairs)
- ss_status mode: consumer output is 3xi32, no consumer input

### Integration

After implementation, generate all batches and verify:
- All stream pair `.s` files assemble with llvm-mc
- Known-good helper instructions (`mov.d1 r0, ss0` and `mov ms, r0`)
  assemble correctly
- Stream instructions move from skipped to testable
- mov.d* appears in both normal and stream test lists

## Future: Core-to-Shim Path

The design supports adding a core-to-shim routing mode later, where
stream output routes directly to the shim DMA without a consumer
compute tile.  This would use `aie.flow(%tile_N_2, Core : 0,
%tile_N_0, DMA : 0)` and a shim-side DMA receive.  The StreamStrategy
and MLIR generator already separate routing from data handling, making
this extension straightforward.

## Files Modified

| File | Change |
|------|--------|
| `tools/isa-test-gen.py` | Add StreamStrategy, remove mov.d* from STREAM_MNEMONICS, update can_test with SS-first ordering, handle doTlast_reg and blocking mov, update generate_all with stream_specs and mov.d* second pass |
| `tools/isa-multi-tile-gen.py` | Handle stream_pair source_type: aie.flow routing, no producer input objectfifo, simpler buffer layout |
| `tools/test_isa_test_gen.py` | Add TestStreamStrategy, mov.d* dual coverage tests, stream MLIR generation tests, mov.nb collision test |
