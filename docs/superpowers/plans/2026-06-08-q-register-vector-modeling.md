# Mission: Model AIE2 q-registers as 128-bit vector data

- Date: 2026-06-08. Status: **staging / orientation** (pre-implementation).
- Parent: #103 Half B (`2026-06-08-vector-compute-halfB-silicon.md`). Decision:
  Maya chose to **lean all the way in** -- make the emulator genuinely execute
  compiled vector kernels through its interpreter, not route around the gap.
- This doc is the centering artifact. Read it first when resuming.

## Why this mission exists

Half-B authored `vec_srs_i32` -- the **first compiled vector-compute kernel ever
run through the emulator's decode -> execute interpreter**. (The ~75-kernel
bridge corpus uses zero vector intrinsics; Half-A verified the *execute*
arithmetic via hand-built `SlotOp`s, never via decoded compiler output.) It
immediately surfaced gaps, because the compiled vector register machinery has
never been exercised end-to-end.

Two gaps found:

1. **Wide-cm SRS odd-base write panic** -- FIXED (commit 4472bfb). `cm` (1024-bit
   source) reduced to int16 yields 256 bits (one `mWa`), not a 512-bit pair;
   `execute_srs_wide` must pick write width by `op.is_wide_vector` (dest class),
   not by the accumulator source width.

2. **q-registers modeled as mask/control regs, not vector data** -- THIS MISSION.
   Chess compiles int32->int16 SRS as: `vlda.ups.s64.s32 bmlN` (8-lane low-half
   loads) -> `vsrs.s16.s64 wX, cmN` (full-cm SRS) -> `vmov qN, wX; st qN`
   (128-bit q-register stores). The emulator maps `q` to `ControlReg(16+idx)`
   (`register_map.rs:209`), so the `vmov`/`st` of result data into `q` writes
   nothing valid and the output buffer stays `0xDEADBEEF` poison (observed:
   `got` alternates -16657/-8531 = 0xBEEF/0xDEAD for all 48 lanes).

Critically: this gap blocks the emulator from **executing** the compiled kernel,
but NOT Half-B's silicon evidence. The capture kernel bakes the model golden as
the host reference, so the HW leg is HW-vs-golden; combined with Half-A
(EMU==golden) that transitively gives EMU==silicon. We are fixing this for
**emulator completeness** (running real vector kernels -- valuable for the GUI
debugger, future kernels, and EMU==HW direct cross-checks), which Maya chose to
invest in now.

## The blocker to resolve first (Task 1): authoritative register hierarchy

We do NOT yet have the definitive AIE2 vector sub-register structure. Derive it
from llvm-aie TableGen (`/home/triple/npu-work/llvm-aie/llvm/lib/Target/AIE/`,
the same `AIE2*RegisterInfo.td` / `SubRegIndex` pattern that gave the
accumulator widths). Open questions to nail with quoted TableGen:

- Width of `q` (hypothesis: 128-bit) and its relationship to `wl`/`wh`/`w`/`x`.
  Known so far (from the accumulator-derivation Explore, partial): `x` = 512-bit
  = `{wl, wh}` each **256-bit** (`sub_256_lo`/`sub_256_hi`), `y` = 1024-bit =
  two `x`. Where does 128-bit `q` sit -- is it a half of `wl`/`wh`
  (so q = quarter of x), an independent 128-bit class, or both? Find the
  `q` register class + its SubRegIndices.
- The register classes used by the failing ops: `vmov qN, wX` and `st qN` --
  what operand classes, and what does `st q` with `#0x10` (16-byte) imply.
- Confirm `vst.128 wl0` semantics: does `.128` store the low 128 bits of a
  256-bit `wl`, or is `wl` itself addressable at 128?

Deliverable: a definitive table {q/w/wl/wh/x/y -> bit width -> sub-register
relationships} with TableGen citations. This grounds every design choice below.

## Current emulator model (what we're changing)

- `Operand::VectorReg(u8)` = 256-bit (`[u32; 8]`), in `bundle/slot.rs`.
- `register_map.rs`: `wl_n -> VectorReg(2n)`, `wh_n -> VectorReg(2n+1)` (each
  256-bit); `x` = "wide" = a VectorReg pair (512-bit) via
  `registers.rs::read_wide`/`write_wide` (even base required).
- `q -> ControlReg(16+idx)` (line 209) -- WRONG for vector-data use. Also
  `qx -> SparseQxReg` (sparse composite, line 214) -- a separate thing, leave it.
- `vector.rs` register file is `[u32; 8]`-per-register; there is no 128-bit
  addressable vector unit today.

## Design space (decide after Task 1)

How to represent a 128-bit vector-data `q` operand:

- **Option A -- new `Operand::QuarterReg(u8)` (Vec128)**: explicit 128-bit
  operand; `vmov q, w` reads low/high 128 of a VectorReg; `st q` stores 128
  bits. Cleanest typing; touches decode map, vmov execute, store execute.
- **Option B -- map q into VectorReg space with a width tag**: reuse VectorReg
  but carry a 128-bit width on the SlotOp. Fewer new variants, but width
  plumbing leaks into many ops.
- **Option C -- model q as a half-of-w addressing**: `q_n -> (VectorReg(n/2),
  half)`. Mirrors hardware sub-register reality if Task 1 confirms q = half of
  wl/wh.

Lean toward A unless Task 1 shows q is cleanly a w-half (then C). Schema-first:
define the operand/width representation before wiring execute.

## Implementation plan (TDD, post Task 1)

1. **Derive hierarchy** (Task 1 above). Write it into this doc.
2. **Decode**: `register_map.rs` map `q` to the chosen vector-data operand. RED:
   a decode test on `vmov q0, wh0` / `st q0` bytes asserting q -> vector-data
   operand (not ControlReg). The `vec_srs_i32` ELF at
   `mlir-aie/build/test/npu-xrt/vec_srs_i32/chess/aie_arch.mlir.prj/main_core_0_2.elf`
   is the byte source (disasm via `llvm-aie/build/bin/llvm-objdump -d
   --triple=aie2`).
3. **Execute `vmov q, w`**: copy the correct 128-bit half of the 256-bit source
   into the q destination. RED execute test first.
4. **Execute `st q`**: store 128 bits to memory. RED test (likely in the store
   path, `execute/memory.rs`). Mind `#0x10` post-increment.
5. **Integration acceptance**: `./scripts/emu-bridge-test.sh --no-hw vec_srs_i32`
   goes green (after `cargo build -p xdna-emu-ffi`). This is the real RED->GREEN
   for the whole mission. Watch for the NEXT gap behind q (see below).
6. **Full `cargo test --lib`** clean; commit per logical unit.

## Known related gaps to watch (may surface behind q)

- **Partial-cm load**: the UPS loads only `bmlN` (low 8 lanes); the `bmhN` high
  half is never written, yet `vsrs.s16.s64 wX, cmN` reads the full 16-lane cm.
  On HW the high-8 outputs are simply not stored (only the low 128 bits go out
  via `q`). Confirm the emulator's cm read + 128-bit store drops the garbage
  half correctly rather than corrupting the valid half.
- **Half-register destinations** (`wh0` = `VectorReg(1)`, odd): now written
  narrow by the SRS fix; verify the q-`vmov` reads the right half.
- **`vst.128 wl0`**: the one direct 128-bit vector store (not via q) -- same
  128-bit store path.

## Acceptance

Mission done when `vec_srs_i32` passes EMU-smoke (`--no-hw`, output == baked
golden) through the real interpreter, `cargo test --lib` is clean, and the
register-hierarchy derivation is documented here. Then Half-B resumes: author
the remaining capture kernels (each may surface its own constellation -- UPS,
Pack, MAC), and the HW-gated capture + Verified flip.

## Pointers

- Fix landed: commit 4472bfb (wide-cm SRS), kernel: 01ca4b9 (vec_srs_i32).
- Branch: `trace-level-event-emission`.
- Key files: `src/interpreter/decode/register_map.rs` (q mapping),
  `src/interpreter/bundle/slot.rs` (Operand enum),
  `src/interpreter/state/registers.rs` (vector file, read/write_wide),
  `src/interpreter/execute/{vector_helpers,vector_srs,memory}.rs`.
- Repro disasm: `2026-06-08` srs_i32 sequence quoted above.
