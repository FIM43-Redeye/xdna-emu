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

## Task 1 FINDINGS (2026-06-08): register hierarchy + the real root cause

### Authoritative AIE2 vector register hierarchy (TableGen-cited)

From `llvm-aie/llvm/lib/Target/AIE/AIE2GenRegisterInfo.td` and
`AIE2RegisterInfo.td`:

| Reg | Width | Class | Composition / SubRegIndex |
|-----|-------|-------|---------------------------|
| `q0-q3` | **128** | `AIE2Vector128RegisterClass` (`[v16i8,v8i16,v8bf16,v4i32,v4f32,i128]`) | `ql`+`qh` (64+64) via `sub_lo_mask`/`sub_hi_mask`. TableGen comment: **"128-bit mask registers"** |
| `wl/wh` | 256 | `AIE2Vector256RegisterClass` | leaf |
| `x` | 512 | `AIE2Vector512RegisterClass` | `wl`(lo)+`wh`(hi) via `sub_256_lo`/`sub_256_hi` |
| `y` | 1024 | `AIE2Vector1024RegisterClass` | two `x` via `sub_512_lo`/`sub_512_hi` |
| `qwl/qwh` | **320** | composite | `q`(64-via-`sub_q`)+`w`(256-via-`sub_w`) |
| `qx` | 640 | composite | `x`(512)+`q`(128) via `sub_sparse_x`/`sub_sparse_q` -- our `SparseQxReg` |

**Decisive structural fact: `q` is INDEPENDENT 128-bit storage, NOT a sub-register
half of `wl`/`wh`.** Proof: `qwl = q + wl` is a **320-bit** composite (128 would
be absorbed if q overlapped wl; 320 = 64-aligned `sub_q` + 256 `sub_w` means q
and w are concatenated, i.e. disjoint). So design **Option C (q as a w-half) is
dead.** The hardware-faithful model is an independent 128-bit q file (Option A's
storage). (Side note: AMD's aiesim punts on this -- `//QX_t ext_qx` is commented
out in `aietools .../me.h`; it models 128-bit values as overlay views of w. That
is an aiesim impl shortcut, not the architecture. TableGen wins.)

### Ground-truth instruction sequence (the compiled SRS, `srs.o.lst`)

`VLDA.UPS.s64.s32 bmlN` x6 (only the low-8-lane `bml` halves loaded; `bmh` never
written -> **partial-cm confirmed**) -> `VSRS.s16.s64 wlN/whN, cmN, s0` x6 (full-cm
SRS into 256-bit w-regs; only low-8 int16 lanes valid) -> `VMOV qN, whN/wlN` (copy
**low 128 bits** of the w-reg = the 8 valid lanes into a q) -> `ST qN,[p1],#16`
(128-bit store) x6, plus one `VST.128 wl0,[p1,#0]` (direct low-128-of-w store).

### THE REFRAME: 128-bit q storage already exists; only one edge is missing

The staging hypothesis ("build a 128-bit q model from scratch -- new Operand
variant, new register file, decode rework") is **wrong/oversized**. The emulator
already has:

- **A 128-bit q register file**: `ctx.mask` (`[u32;4]` x 4), today labelled
  "mask." This IS the q file.
- **q decode**: `register_map.rs:209` maps `q -> ControlReg(16+idx)` (and
  `ql/qh/qwl/qwh` to 28..35 / 20..27). This encoding is FINE -- it routes to
  `ctx.mask`.
- **The store-read path**: `read_store_data_wide` (`memory/mod.rs:1231`) already
  reads `ControlReg(16..19)` from `ctx.mask` and returns 128 bits. `ST q` works.
- **A pipelined 128-bit vector-write path**: `context.rs:800` already writes
  `ControlReg(16..19)` + `vec_value` into `ctx.mask`. 
- **`VST.128 wl0`**: already works (VectorReg source -> `read_store_data_wide` ->
  QuadWord low-128 store at `memory/mod.rs:334`).

**The single missing edge: `VMOV q, w` execute.** `execute_copy`
(`vector_misc.rs:630`) for `VMOV q0, wh0` (dest `ControlReg(16)`, src
`VectorReg(1)`, not accum, not wide) falls into the "narrow vector move" branch
-> `write_vector_dest`, which only handles VectorReg/AccumReg/ScalarReg dests.
**The q (`ControlReg 16..19`) destination is silently dropped** -> `ctx.mask`
never receives w's low 128 -> `ST q` later reads stale storage -> output stays
`0xDEADBEEF` poison. That single dropped move is the whole bug.

### Revised design decision

No new `Operand` variant, no new register file, no decode rework. **Extend the
copy path to route a q-mask destination (`ControlReg 16..19`, plus `ql/qh` and
the `qwl/qwh` wide-mask forms if a kernel needs them) into `ctx.mask`, copying
the LOW 128 bits of the 256-bit w source.** Faithful to TableGen (q is the
already-existing independent 128-bit file) and minimal.

Naming nit to consider (non-blocking): `ctx.mask` / "mask register" is now doing
double duty as the q vector-data file. A rename to `ctx.q` (or a doc note that
mask == q) would reduce future confusion, but is cosmetic -- defer unless cheap.

## Implementation plan (TDD, post Task 1 -- REVISED per findings)

Task 1 is done (above). Decode, store-read, pipelined-write, and `VST.128` already
work; `ST q` already works. The only missing edge is `VMOV q, w` execute.

1. **Execute `vmov q, w`** (the fix): in `execute_copy` (`vector_misc.rs:630`),
   detect a q-mask destination (`ControlReg(16..19)`, and `ql/qh` 28..35,
   `qwl/qwh` 20..27 if needed) and copy the **low 128 bits** (`src[0..4]`) of the
   256-bit w source into `ctx.mask` via the existing pipelined-write mechanism
   (so the read-before-write VLIW bundle ordering is honoured -- critical: the
   SRS schedule does `VSRS wh0,cmK ; VMOV q,wh0` and `ST q ; VMOV q,whN` in the
   same bundles, relying on read-old/write-new). RED: an execute unit test on a
   `VMOV q0, wh0` SlotOp asserting `ctx.mask[0] == low 128 of VectorReg(1)`;
   watch it fail (q-dest dropped), then GREEN.
2. **Integration acceptance**: `./scripts/emu-bridge-test.sh --no-hw vec_srs_i32`
   goes green (after `cargo build -p xdna-emu-ffi`). The real end-to-end
   RED->GREEN. Watch for the NEXT gap behind q (partial-cm read tolerance below).
3. **Full `cargo test --lib`** clean; commit per logical unit.

Byte source if more decode evidence is needed: `srs.o.lst` (Chess listing, has
the cleanest disasm) and the ELF at `.../chess/aie_arch.mlir.prj/main_core_0_2.elf`
(disasm via `llvm-aie/build/bin/llvm-objdump -d --triple=aie2`).

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
