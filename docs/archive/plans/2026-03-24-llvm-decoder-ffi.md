# Plan: LLVM Decoder FFI Integration

## Problem

Our bytecode decoder interpreter doesn't implement `TRY_DECODE` correctly.
LLVM's `TRY_DECODE` calls per-instruction decoder functions (via `decoder_idx`)
that validate register class membership and reject mismatches. Our interpreter
accepts the first match, causing collisions like VSEL_16 winning over
VPUSH_HI_32 for certain register encodings.

Root cause: a single missing validation path in the decoder, not patchable
with mask/bits heuristics (both instructions' fixed bits genuinely match).

## Proven Approach

Link LLVM's own AIE2 disassembler into the emulator via C FFI. **Already
proven working** -- the test binary at `decoder_ffi/test_decoder` compiles,
links, and initializes successfully.

### What exists now

- `decoder_ffi/aie2_decoder.h` -- C interface (header)
- `decoder_ffi/aie2_decoder.cpp` -- Thin wrapper around LLVM MCDisassembler
- `decoder_ffi/aie2_decoder.o` -- Compiled object (works)
- `decoder_ffi/test_decoder` -- Linked test binary (runs, returns SUCCESS)
- Various `.inc` files from the standalone shim attempt (can be cleaned up)

### Compile command (proven)

```bash
LLVM_BUILD=/home/triple/npu-work/llvm-aie/build
LLVM_SRC=/home/triple/npu-work/llvm-aie/llvm

g++ -std=c++17 -c -O2 \
  -I${LLVM_BUILD}/include \
  -I${LLVM_SRC}/include \
  -I${LLVM_BUILD}/lib/Target/AIE \
  -I${LLVM_SRC}/lib/Target/AIE \
  -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS \
  -o aie2_decoder.o aie2_decoder.cpp
```

### Link command (proven)

```bash
g++ -o test_decoder aie2_decoder.o test_main.o \
  -L${LLVM_BUILD}/lib \
  $(${LLVM_BUILD}/bin/llvm-config --libs aie) \
  $(${LLVM_BUILD}/bin/llvm-config --system-libs)
```

## Steps to Integrate

### 1. Wire into build.rs

Use the `cc` crate to compile `aie2_decoder.cpp` and link LLVM libraries.
The build script already knows `LLVM_AIE_PATH`. Add:

- Compile `decoder_ffi/aie2_decoder.cpp` with the include paths above
- Link flags from `llvm-config --libs aie` and `--system-libs`
- Expose `LLVM_AIE_BUILD` path (currently `LLVM_AIE_PATH/../../build` or
  configurable)

### 2. Implement per-slot decode in the C wrapper

The current `aie2_decode_slot()` is a stub. LLVM's public API
(`getInstruction()`) works on full VLIW bundles, not individual slots.

Two options:
- **Option A**: Feed synthetic VLIW bundles (pack one slot into a minimal
  bundle format). Requires understanding the format codes but reuses the
  public API.
- **Option B**: Call the internal `decodeInstruction()` directly on per-slot
  decoder tables (DecoderTableMv32, etc.). These are `static` in the .inc
  file but the disassembler .cpp file has slot decoder wrappers that call
  them. We'd need to expose those.

Option B is cleaner. The slot decoders (`decodeMvSlot`, etc.) already exist
in AIE2Disassembler.cpp. We'd add thin `extern "C"` wrappers that call them.

### 3. Rust FFI bindings

In `src/tablegen/decoder_bytecode.rs` (or a new `src/decoder_ffi/` module):

```rust
extern "C" {
    fn aie2_decode_slot(slot: u32, insn_bits: u64) -> Aie2DecodeResult;
    fn aie2_opcode_name(opcode: u32) -> *const c_char;
    fn aie2_decoder_init() -> i32;
}
```

### 4. Replace the bytecode interpreter

The `SlotIndex::decode()` method currently calls our bytecode interpreter.
Replace with:

```rust
pub fn decode(&self, word: u64) -> Option<(&InstrEncoding, HashMap<String, u64>)> {
    let result = unsafe { aie2_decode_slot(self.slot_id, word) };
    if result.success == 0 { return None; }
    let name = unsafe { aie2_opcode_name(result.opcode) };
    // ... map to InstrEncoding and extract operands
}
```

### 5. Verify

- Run the VPUSH_HI test: `cargo test test_vpush_hi_32_decoder`
- Run the full ISA sweep: `./scripts/isa-test.sh`
- Confirm zero decoder collisions

## What to keep vs clean up

**Keep** (proven working code):
- `decoder_ffi/aie2_decoder.h` -- the C interface
- `decoder_ffi/aie2_decoder.cpp` -- the LLVM wrapper
- `src/tablegen/decoder_bytecode.rs` -- keep the existing bytecode interpreter
  as a fallback/reference until the FFI path is fully validated

**Clean up later** (standalone shim attempt, superseded by LLVM linking):
- `decoder_ffi/AIE2GenRegisterEnum.inc`
- `decoder_ffi/AIE2RegisterDecoderTables.inc`
- `decoder_ffi/AIE2CustomDecoders.inc`
- `decoder_ffi/AIE2GenDisassemblerTables.inc` (filtered copy)

**Revert** (debug tracing added during investigation):
- Remove trace/eprintln code from `decoder_bytecode.rs`
- Remove the `try_decode_validators` mechanism (superseded by LLVM FFI)

## Notes

- Static link adds ~60MB to binary (acceptable for dev tool)
- FFI call cost is negligible (one function call, no allocations)
- LLVM dependency is already in the build graph (TableGen extraction)
- `llvm-config --libs aie` gives the complete library set
- System libs needed: `-lrt -ldl -lm -lz -lzstd -lxml2`
