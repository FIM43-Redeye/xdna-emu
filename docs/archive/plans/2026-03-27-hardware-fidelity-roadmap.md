# Hardware Fidelity Roadmap

**Goal**: Make every logical component of the emulator mirror the corresponding
logical component of the real AIE2 silicon. No hacks, no kludges -- understand
what the hardware does and reflect it faithfully.

**Baseline**: 67.1% ISA accuracy (3307/4925), 2611 unit tests.

---

## Phase 1: Structural Cleanup (Days 1-2)

Fix code organization issues that make the emulator harder to reason about
and maintain. These don't change behavior but make the architecture honest.

### 1.1 Data-drive DMA convenience constructors

**What**: `DmaEngine::new_compute_tile(col, row)` hardcodes `(2, 2, 16, 16)`.
Same for mem_tile and shim_tile. These should accept ArchConfig or pull from
the device model.

**Files**: `src/device/dma/engine/mod.rs:187-198`

**Approach**: Add `new_from_arch(col, row, tile_type, arch: &ArchConfig)` that
reads channel counts, BD counts, and lock counts from the config. Keep the
old constructors as `#[cfg(test)]` convenience wrappers so the 80+ test call
sites don't all need updating.

**Risk**: Low. Values are correct for AIE2. This is future-proofing.

### 1.2 Data-drive stream switch port type assignments

**What**: TileCtrl port indices (compute=3, memtile=6, shim=0) are hardcoded
from AM025. Packet slot count (4) and arbiter count (8) are hardcoded.

**Files**: `src/device/stream_switch/mod.rs:131-223`

**Approach**: Add per-tile-type port layout to ArchConfig (or a new
StreamSwitchSpec struct). Extract from AM025 JSON or hardcode in arch
constants with clear provenance comments. The key win is having ONE place
where port assignments live, not scattered across constructors.

**Risk**: Low. Values are correct. Centralizes knowledge.

### 1.3 Name constant for bank row size

**What**: `>> 4` (16-byte bank rows) appears as a magic shift in banking.rs.

**Files**: `src/device/banking.rs:20,35,39`

**Approach**: Define `BANK_ROW_SHIFT: u32 = 4` (or derive from
`PHYSICAL_BANK_WIDTH_BITS / 8`) and use it. One-line change per site.

**Risk**: Zero.

### 1.4 Clean up remaining cold-path name-matching in decoder

**What**: ~9 cold-path name-matching instances in decoder.rs semantic
overrides. Some are redundant (LLVM returns correct names), some genuinely
need name detection until better signals exist.

**Files**: `src/interpreter/decode/decoder.rs` (semantic_override block),
`src/tablegen/resolver.rs` (branch condition inference)

**Approach**: Audit each override. Remove redundant ones. For the rest,
document WHY name-matching is needed (e.g., "no TableGen signal for cascade
direction") so we know what to fix when upstream data becomes available.

**Risk**: Low. Each removal tested against ISA baseline.

### 1.5 CORE_ID write protection and initialization

**What**: CORE_ID register (r37) is read-only in hardware but writable in
emulator. Not initialized from tile position.

**Files**: `src/interpreter/state/registers.rs:129` (write method),
`src/interpreter/state/context.rs:658` (constructor)

**Approach**:
- Add `tile_col: u8, tile_row: u8` to ExecutionContext::new()
- Initialize CORE_ID to `(col << 16) | row` (per AM020 encoding)
- Log warning on write to CORE_ID (don't crash -- real programs may do
  harmless writes that get ignored by hardware)

**Risk**: Low. May need to update test constructors.

---

## Phase 2: Real Mismatches (Days 2-4)

Fix behavioral differences that cause incorrect results or wrong timing.
These directly impact ISA accuracy and bridge test fidelity.

### 2.1 Y-register (1024-bit vector) aliasing

**What**: AIE2 has y0-y3 (1024-bit) as aliases over groups of 4 consecutive
w-registers. VMUL sparse instructions use eYs operands. Currently missing --
any instruction with a y-register operand will silently get wrong data.

**Files**: `src/interpreter/state/registers.rs` (VectorRegisterFile),
`src/interpreter/decode/decoder.rs` (operand mapping)

**Approach**:
- Add `read_quad(base: u8) -> [u32; 32]` and `write_quad(base: u8, data)`
  to VectorRegisterFile (reads 4 consecutive 256-bit registers)
- Map y-register operands in decoder: y0 -> VectorReg(0) with is_quad flag,
  or use a new WideVectorReg(base) operand variant
- Wire through to execution (VMUL sparse already reads wide vectors)

**Verification**: Check if any ISA tests use y-register operands. If so,
expect accuracy improvement.

**Risk**: Medium. Operand plumbing touches decoder and execution.

### 2.2 Bank conflict stalls

**What**: When core and DMA access the same memory bank in the same cycle,
hardware stalls one of them. We detect the conflict (bitmask tracking) and
emit trace events, but don't actually stall.

**Files**: `src/interpreter/engine/coordinator.rs:826-867` (conflict detection),
`src/device/tile/mod.rs:225-228` (cycle_dma_banks)

**Approach**:
- When conflict detected, delay the DMA transfer by 1 cycle (DMA yields to
  core, per AM020 arbitration priority)
- Track stall cycles in DMA channel FSM
- This affects timing only, not functional results

**Verification**: Bridge test cycle counts should move closer to hardware.

**Risk**: Medium. Timing changes may shift other test behavior. Must verify
no regression in functional results.

### 2.3 Stream latency: per-hop instead of flat

**What**: Inter-tile stream routing uses a fixed latency regardless of
distance. Hardware has ~3-4 cycles per hop.

**Files**: `src/device/array/mod.rs:146-173` (InFlightWord),
`src/device/array/routing.rs` (route_streams)

**Approach**:
- Calculate Manhattan distance between source and destination tiles
- Multiply by ROUTE_LATENCY_PER_HOP (already defined per architecture)
- Set InFlightWord.cycles_remaining accordingly

**Verification**: Trace comparison should show more realistic stream timing.

**Risk**: Low. Only affects timing, not functional results.

### 2.4 Lock register read masking (6-bit)

**What**: Hardware lock value register is 6-bit [5:0]. Values outside 0-63
alias when read back. Emulator stores full i8 range without masking.

**Files**: `src/device/tile/locks.rs:348-351`

**Approach**: Mask lock value on read to 7-bit signed (per aie-rt
XAIEML_LOCK_VALUE_MASK 0x7F). This only matters for programs that read
lock values via register, which is uncommon but possible.

**Risk**: Very low.

---

## Phase 3: Accuracy Deep Dives (Days 4-7)

With structural issues fixed and real mismatches addressed, dig into the
remaining 32.9% ISA failures by category.

### 3.1 Pointer operations (1.4% passing -- 69 failures)

**What**: Nearly zero passing. Likely a fundamental issue with pointer
register addressing, modifier registers, or post-modify execution.

**Approach**: Pick 2-3 failing pointer tests, trace execution step by step,
compare to hardware output. The 2D/3D AGU (dimension registers dn/dj/dc)
may be the root cause -- post-modify works for simple cases but
multi-dimensional patterns are not implemented.

### 3.2 Vector broadcast (39.3% -- 51 failures)

**What**: VBCST instructions failing at ~40%. Likely an operand routing
issue or element type mismatch.

**Approach**: Sample failing tests, compare emulator output to hardware
output byte-by-byte. Look for systematic patterns (e.g., all 8-bit
broadcasts fail, all 32-bit pass).

### 3.3 Vector arithmetic (52.9% -- 714 failures)

**What**: Largest failure bucket. Heterogeneous -- includes VADD, VSUB,
VMUL, comparisons, shifts, etc. Need to break down by sub-category.

**Approach**: Parse the analysis log to identify which specific operations
fail. Group by operation type. Attack the largest/simplest group first
(e.g., if all VBAND fails, that's one fix for many test points).

### 3.4 Vector MAC (48.2% -- 114 failures)

**What**: MAC/matmul failures. May be related to accumulator width handling,
config register parsing, or bf16 computation errors.

**Approach**: Cross-reference with the accum_width investigation from the
previous session. The VADD/VADD_F failures (0/18 and 3/15) may share root
cause with MAC failures.

---

## Phase 4: Future Hardening (Week 2+)

### 4.1 Structural hazard detection

Add execution unit contention checking. Hardware has 1 vector ALU, 1 store
port, 2 load ports. Bundle validation should verify no structural conflicts.
This is a debug aid more than a correctness fix (compiler won't generate
invalid bundles, but it catches decoder bugs).

### 4.2 2D/3D AGU (address generation unit)

Full multi-dimensional addressing using dn (size), dj (stride), dc (count)
registers. Currently only simple post-modify works. This is needed for
programs with complex memory access patterns.

### 4.3 FP status registers

Connect float/conversion instructions to srFPFlags, srF2IFlags status
registers. Low priority since most programs don't read these.

### 4.4 CDO opcode graceful handling

Currently 43 of 108 aie-rt CDO opcodes are handled. Unknown opcodes crash
via bail!(). This is CORRECT behavior -- fail fast. But we should:
- Add the opcode NAME to the error message (not just hex)
- Categorize unhandled opcodes as "safe to skip" (NPI, SEM, debug/logging)
  vs "must handle" (DMA, config, sync)
- Warn-and-skip for safe-to-skip categories
- Keep bail!() for must-handle categories

### 4.5 AIE2P preparation

- D3 dimension support for compute tile BDs
- Channel count differences (if any)
- New register classes from llvm-aie

---

## Verification Protocol

After each change:

```bash
# Unit tests (must stay >= 2611, no regression)
TMPDIR=/tmp/claude-1000 cargo test --lib

# Build release
TMPDIR=/tmp/claude-1000 cargo build --release

# ISA accuracy (must stay >= 67.1%, hopefully improve)
XDNA_EMU=release ./scripts/isa-test.sh --no-hw

# Bridge tests (periodic, not after every change)
./scripts/emu-bridge-test.sh --no-hw
```

## Success Criteria

- Every logical component of the emulator has a clear 1:1 mapping to the
  hardware component it models
- No magic numbers without provenance comments
- No name-matching in hot execution paths (all structural dispatch)
- ISA accuracy trending upward from 67.1% baseline
- Bridge test timing closer to hardware (stream latency, bank stalls)
