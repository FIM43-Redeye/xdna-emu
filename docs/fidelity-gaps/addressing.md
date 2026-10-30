---
class: addressing
subsystem: AIE2 address generation -- the 2D / 3D descriptor post-modify (`[p], d`) of every load and store
posture: resolved -- one rule, pinned on silicon for every TableGen family; the residuals are unobserved corners, not unvalidated model
status: resolved (rule, descriptor layout, classification); open: count-write latency, descriptors d4..d7 as 2D operands, unaligned and cross-tile accesses
---

# Addressing (2D / 3D descriptor post-modify)

An instruction `op ..., [p], d` accesses memory at `p` and then updates the pointer and the
counts from a descriptor. This page records what silicon does, how the emulator is organised
around it, and what was tried and rejected. The rule lives in
`src/interpreter/execute/memory/agu.rs`; the register layout comes from TableGen
(`xdna_archspec::aie2::descriptors`); the single place a pointer is updated after an access is
`MemoryUnit::apply_post_modify`.

## The rule (NPU1 silicon)

A 2D descriptor `d<k>` is the registers `m<k>` (modifier), `dn<k>` (size), `dj<k>` (stride) and
`dc<k>` (count). A 3D descriptor `d<k>_3d` is the pair `d<k>` (inner) and `d<k+4>` (outer); its
modifier is `m<k>`. Comparisons are unsigned, strides are two's complement.

```text
2D:  if dc < dn { p += dj; dc += 1 } else { p += m; dc = 0 }
3D:  if dc1 < dn1      { p += dj1; dc1 += 1 }
     else if dc2 < dn2 { p += dj2; dc1 = 0; dc2 += 1 }
     else              { p += m;   dc1 = dc2 = 0 }
```

It was first pinned through the fused `vldb.unpack` loads (`vec_unpack_addr`, 1,888 accesses,
[vector-compute.md](vector-compute.md)). The `vec_agu_*` kernels then ran every other family
and checked the same rule: 9,456 records (pointer after the access, both counts), all equal.

| Kernel | Families (2D and 3D unless noted) | Records |
|---|---|---|
| `vec_agu_ldscalar` | `lda`, `lda.s8`, `lda.u8`, `lda.s16`, `lda.u16` | 1,200 |
| `vec_agu_stscalar` | `st`, `st.s8`, `st.s16` | 720 |
| `vec_agu_vload` | `lda q` (mask register), `vldb` (256 bit), `vldb.128`, `vlda` into the accumulator halves (`am`, 32- and 64-bit lanes) | 1,200 |
| `vec_agu_vstore` | `st q` (2D only), `vst`, `vst.128`, `vst` from the accumulator halves | 1,056 |
| `vec_agu_ups_s` | `vlda.ups.s32.s16`, `.s32.s8`, `.s64.s32`, `.s64.s16`, `vlda.conv.fp32.bf16` | 1,200 |
| `vec_agu_ups_d` | `vlda.ups.s32.d16`, `.s32.d8`, `.s64.d32`, `.s64.d16` | 960 |
| `vec_agu_srs_s` | `vst.srs.s16.s32`, `.s8.s32`, `.s16.s64`, `.s32.s64` | 960 |
| `vec_agu_srs_d` | `vst.srs.d16.s32`, `.d8.s32`, `.d16.s64`, `.d32.s64` | 960 |
| `vec_agu_pack` | `vst.pack.s8.s16`, `.d8.d16`, `.s4.s8`, `.d4.d8`, `vst.conv.bf16.fp32` | 1,200 |

Each chunk runs four concurrent walkers (so the compiler allocates several descriptor registers
at once) over three groups of descriptor sets, with counts that start below, at and above their
sizes, size 0, negative strides and negative modifiers. The captured Chess disassembly (stored in
the goldens next to the data) shows `d0`..`d3` for the 2D forms and `d0_3d`..`d3_3d` for the 3D
forms; every family was seen with at least `d0`, most with all four.

Findings:

- **One rule for all of them.** Loads, stores, scalar, vector, accumulator, q-register, fused
  ups / conv loads and fused srs / pack / conv stores update the pointer and the counts
  identically. Nothing depends on the access width: the update does not use the number of bytes
  moved.
- **The access is at the old pointer.** In every family the data moved is the data at `p`, never
  at the updated pointer.
- **Data.** For the raw families (scalar and vector loads and stores) every loaded word and the
  whole stored memory image matched (2,880 vector words loaded, 40 store images of 2 KiB).
  The fused families' data paths are replayed end to end by the EMU bridge run of the same
  kernels instead (see below).
- **The count register is read back two bundles after the access at the earliest.** Chess never
  reads `dc<k>` sooner (`st dc0, [p], #4` appears two or more bundles after the `.2d` access
  that updates it), so silicon's count-write latency is at most that, and the emulator's
  immediate write is consistent with all 9,456 records. The exact latency is not pinned: it needs
  a read of the count in the bundle right after the access, which the compiler does not emit.

## Emulator design

- **Classification from TableGen.** `AddressingMode::TwoDim` / `ThreeDim` are recognised from the
  register class of the instruction's descriptor input operand (`eD` derives from
  `AIE2Dim2DRegisterClass`, `eDS` from `AIE2Dim3DRegisterClass`) and cross-checked against the
  `*__ag_pstm_2d` / `*__ag_pstm_3d` addressing class; the build fails if the two disagree. The
  record names of these instructions carry no `_pstm_`, so name matching never saw them (they
  decoded as unclassified and nothing updated their pointer at all).
- **How many instructions.** 46 two-dimensional and 46 three-dimensional TableGen records (92 in
  all, including the `padda` / `paddb` / `padds` pointer arithmetic and the eight `vldb.*.unpack`
  forms): `aie2::descriptors::ACCESS_RECORDS_2D` / `_3D`, generated from the same extraction; a
  unit test proves each one is classified and that nothing else is. (The task brief spoke of 142;
  the TableGen has 92 such records. The 70 distinct mnemonics the captures reached are decoded
  from the real instruction bytes by `agu_goldens`.)
- **Descriptor layout is data.** Which registers a descriptor operand names (`d<k>` = `m<k>`,
  `dn<k>`, `dj<k>`, `dc<k>` through `sub_mod` / `sub_dim_size` / `sub_dim_stride` /
  `sub_dim_count`; `d<k>_3d` = `d<k>` and `d<k+4>` through `sub_lo_dim` / `sub_hi_dim`) is
  extracted from TableGen and read by `descriptor_step`; the interpreter keeps no copy of the
  `k` / `k + 4` grouping. This is the AIE2 table. AIE2P gets its own from its own TableGen.
- **One path.** The decoder moves the descriptor from the sources to `PostModify::Descriptor`;
  `apply_post_modify` takes the step with the timing of its own post-modify (immediately after the
  access, like an immediate or register post-modify) for every load and store. The fused-unpack
  special case is gone. `padda` / `paddb` / `padds .2d/.3d` keep the descriptor as a source (they
  are pointer arithmetic, not an access) and call the same `descriptor_step`.
- **A descriptor can no longer be read as a modifier register.** Descriptor operand indices
  (`MOD_BASE_D3D` = 32 and up) used to wrap modulo the 32 modifier registers, so `d0` (36) read
  `m4`. `ModifierRegisterFile` now panics on an index outside m / dn / dj / dc, and
  `PostModify::Register` naming a descriptor panics in `apply_post_modify`.

## Rejected alternatives

| Alternative | Why not |
|---|---|
| Classify 2D / 3D by instruction name (`.2d` / `.3d`) | Works today, silently breaks on a renamed record; the class and the operand register class are what the hardware encoding actually has. A unit test keeps the mnemonic check as an independent oracle only. |
| Keep the descriptor among the sources and special-case it in each handler | 25 handlers, one forgotten; this is how the non-unpack families lost their update. |
| Let `PostModify::Register` carry descriptors and decide at run time | The wrap-around read is the bug. |
| A separate update for stores or for narrow accesses | Silicon applies the same step (9,456 records). |
| Hand-written `k` / `k + 4` in the interpreter | It is TableGen data; AIE2P need not share it. |

## Not reached

- `st.3d` with a q register: Chess compiles the 3D store of a 128-bit vector to `vst.3d.128`, so
  that one pair is not captured (`stq` has 2D only).
- Scalar loads (`lda.2d` and friends) appeared only with `d0` in the compiled code; vector, fused
  and store forms used `d0`..`d3`.
- `padda` / `paddb` / `padds .2d/.3d` are covered by their own unit test, not by a capture here
  (they appeared in the compiled kernels as address updates between accesses).

## Open

| Gap | Status |
|---|---|
| count-write latency | at most two bundles (observed); not pinned lower |
| `d4`..`d7` as 2D operands | the compiler allocated `d0`..`d3` for four walkers; the 3D forms use `d4`..`d7` as the outer dimension, which the 3D captures exercise |
| counts and sizes above 20 bits or negative | counts are masked to 20 bits on write like every modifier; values above 2^20 not captured |
| unaligned and cross-tile accesses | all captured addresses are aligned and local |
| Chess emits `vst am, [sp, #0]` as an accumulator spill in some register-pressure shapes | the LLVM decoder rejects it (`imm12x32_neg` with raw 0), so the emulator cannot run such code; the `vec_agu_srs_d` d8 form was reshaped to avoid it. Not an addressing gap |
