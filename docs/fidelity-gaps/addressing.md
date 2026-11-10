---
class: addressing
subsystem: AIE2 address generation -- the 2D / 3D descriptor post-modify (`[p], d`) of every load and store
posture: resolved -- one rule, pinned on silicon for every TableGen family; the residuals are unobserved corners, not unvalidated model
status: resolved (rule, descriptor layout, classification, unaligned accesses, address-bit decode, loads with no memory behind them); open: count-write latency, descriptors d4..d7 as 2D operands, west of column 1 (column-0 contents), vldb.compr
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

**Large and negative counts** (2026-11-10, ISA batches 211..218, `DescriptorCountStrategy` in
`tools/isa-test-gen.py`). `padda.2d` / `padda.3d` with sizes 3, 0x7FFFF, 0x80000, 0xFFFFF,
0x100003, 0xFFFFFFFF against counts 0, 2, 3, 0x7FFFF, 0x80000, 0xFFFFF, 0x100000 (84 points, the
3D ones on the outer pair): every write keeps the low 20 bits (0x100003 reads back 3,
0xFFFFFFFF reads back 0xFFFFF, zero-extended), and the comparison is unsigned over those 20
bits (size 0xFFFFF with count 0x80000 steps, size 0x80000 with count 0x7FFFF steps, count
0x80000 against size 0x80000 wraps). The count carries into bit 19 without sign effects
(0x7FFFF steps to 0x80000). HW r1 == r2, EMU == HW on every point.

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

## Unaligned accesses (NPU1 silicon)

Probe: `tools/gen_ldst_align_probe.py` (one hand-written core through the topology harness;
tests in `tools/tests/test_gen_ldst_align_probe.py`). Every instruction family is swept byte
by byte across a window; stores go to a marker-filled buffer and the bytes that changed are
read back, loads read a window whose byte is its index and are dumped through an aligned
store. 328 cases over 42 kernels, run twice on silicon (identical), 0 driver resets. Data:
`build/experiments/ldst-align/gen8` (`r1`, `r2`; `e1`/`e2` are the emulator), checked by
`verify.py` (rule) and `diffhw.py` (emulator against silicon). llvm-aie states no rule (only a
comment at `AIE2GenFixupInstrInfo.td:592`, "128-bits vectors are 128-bits aligned in memory").

**One rule for every load and store:** the access touches the effective address rounded down to
its width. The data is not rotated or split; nothing faults.

| Family | Width (bytes) |
|---|---|
| `st` / `lda` word | 4 |
| `st.s16`, `lda.s16`, `lda.u16` | 2 |
| `st.s8`, `lda.s8`, `lda.u8` | 1 (exact) |
| `st q`, `lda q`, `vst.128`, `vlda.128`, `vldb.128` | 16 |
| `vst`, `vlda`, `vldb` (w and accumulator halves), `vlda.ups.*`, `vlda.conv.*`, `vldb.unpack.*`, `vst.srs.*`, `vst.pack.*`, `vst.conv.*` | 32 |

Example: a `vst` at base+48 lands at base+32 (the step-5 finding); a word store at +3 writes the
word at +0.

- **Pointer update.** `[p], #imm`, `[p], m` and `[p], d` (2D and 3D) leave the pointer at
  old + step, low bits kept. The mask applies to the access only.
- **Indexed forms.** `[p, dj]`: the mask applies to the sum of base and index, not to the base.
  Immediate offsets are multiples of the width, so the two cannot be told apart there.
- **Stack-relative.** `[sp, #imm]` follows the same rule; `sp` is unchanged.
- **Emulator.** `MemoryUnit::get_address` / `get_store_address` round down by the decoded
  `mem_width`, which comes from TableGen; the bank and watch accounting sees the masked address.

Three further things the probe found:

- **`vldb.4x16/32/64.lo/hi` (LUT gathers) have their own rule**, not floor-to-width. For a lane
  with 32-bit value `v` and granule mask `M` (4, 8, 16 bytes for 16, 32, 64 bit):
  `eff = (v & ~31) | (((v & 31) & ~M) >> 1) | (odd lane ? 0x10 : 0)`. The read is 8 bytes and
  bytes past the 16-byte row of `eff` read as zero. It fits 64 addresses x two window positions
  x three sizes x lo and hi x distinct per-lane addresses exactly. The previous emulator model
  (`v & M`, no halving, no row cut) was right only for 32-byte-aligned lane values.
  `lut_lane_address` in `memory/mod.rs`.
- **`vldb.sparse.*` fetches a 32-byte aligned block**, rotated to start at the pointer rounded
  down to a word; bytes past the block end wrap to its start (`sparse_fetch_line`). Stream
  position follows the pointer at word granularity.
- **A 128-bit store wrote 32 bytes in the emulator**, zeroing the 16 bytes after it. Silicon
  changes 16. Fixed (`write_quad_to_memory`). `VST_128_ag_spill` also lacks TableGen
  `Uses = [SP]`, so it decoded as a bare immediate; the sp-relative test now also checks the
  assembly string.

Not probed: `vldb.compr.*` (unmodelled), `lda.tm` / `st.tm`, stale ring bytes after a sparse
line (see Open).

## Which address bits pick the memory (NPU1 silicon)

Gather lanes are full 32-bit values, so a `vldb.4x*` operand can carry any address bits; the
Chess suite's random lanes (`me_detail::load_4x*`, 12 cases) read memory the kernel never
wrote. The probe `tools/gen_lut_addr_probe.py` (suite `build/experiments/harness/load4x/wrap3`,
HW r1 == r2, no TDR) sets the bits on purpose against regions the kernel itself filled (its
input window, the top 512 bytes of its memory, a window it stores into the north neighbour):

- **Only bits 16-17 select the direction** (3 own memory, 2 north, 1 west, 0 south); bits 18
  to 31 are ignored. `0x3xxxx`, `0xBxxxx`, `0xFxxxx` and `0xFFF7xxxx` read the tile's own
  memory, `0x2xxxx`, `0xAxxxx`, `0xExxxx` the north neighbour. 168 of 168 gather lanes.
- **Pointer accesses decode the same way** (`vlda`, `vldb`, `lda`, `vst` through a 20-bit
  pointer at those aliases): 28 of 28.
- **Bits 5-15 pass through** (lanes across the whole 64 KiB, the last row at `0x7fffe` included,
  with the usual cut at the 16-byte row end).
- **Emulator.** `MemoryQuadrant::from_address` takes `4 | ((addr >> 16) & 3)`; it used to read
  `addr / 0x10000` as a byte and called everything outside 4..6 local, so north aliases read
  the own memory. A gather lane in the last 16-byte row of memory read zeros (a whole-element
  bounds check); it now reads the row like any other. EMU == HW on both probe cases.
- **The generator** (`tools/chess-test-gen.py`) now points every address operand into a
  256-byte table in the kernel's own input (see `docs/components/testing.md`).

Memory nobody in the kernel wrote (case `lut_unwritten`) is deterministic on one machine but is
not the kernel's: the west quadrant of a column-1 tile is the DPU-reserved column 0 and returns
its contents (see Open); the south quadrant of row 2 (the memory-tile row, no core memory) is
answered by the load-port rings of the next section. Replayed over that case, the rule names a
source for all 32 south lanes: 16 are column-0 rows that the case's own west lanes also read
(byte-equal, 0 mismatches), 10 are column-0 rows at offsets no west lane showed, and 6 are
port-B ring slots left by the previous kernel (`ptr_alias`'s last north line). None is data the
emulator has, so the case stays an exception for the column-0 node and for ring state across
launches (replay: `build/experiments/harness/port-latch/wrap3-replay`).

## Loads with no memory behind them (NPU1 silicon)

A core in row 2 sits above the memory-tile row, so its south quadrant has no core data memory;
a core in row 5 has nothing to its north. Probe: `tools/gen_port_latch_probe.py` (tests in
`tools/tests/test_gen_port_latch_probe.py`), one hand-written core in physical column 2 (its
west neighbour is real memory the kernel fills) or row 5, every input halfword naming its
region and offset, so the analysis names the access each returned byte came from. Suite
`build/experiments/harness/port-latch/suite`: 12 cases, HW r1 == r2 (`ring_f`: f1 == f2), no
driver resets. A row-3 control (`ctrl`, real memory to the south) reads what the kernel stored
there.

**Each load port keeps a ring of the rows it fetched, and a load with no memory behind it is
answered from that ring.**

- Port A (`vlda`, `lda` in every width, `lda q`, `vlda.ups`, `vlda.conv`) and port B (`vldb`,
  `vldb.128`, `vldb.unpack`, `vldb.sparse`) each have two rings of six 16-byte entries, one per
  half of the 256-bit path (address bit 4), each with its own pointer.
- A load with memory behind it (own or neighbour) writes the whole 16-byte row it fetched into
  the slot at its half's pointer and advances that pointer: a 256-bit load both halves, a 128-bit
  or scalar load only the half its address selects, with the full row even for one byte
  (`ring_d`).
- A load with no memory behind it writes nothing and returns the slot at the pointer (the bytes
  at its in-row offset), then advances the pointer the same way. So a run of south `vldb`s
  returns the last six `vldb` lines, oldest first, then repeats with period six whatever the
  address (`sweep`, `ring_a` b0, `ring_b` b2).
- Stores and the other port's loads never touch a ring (`ring_a`, `ring_d`, `ring_e`).
- A `vldb.4x*` gather is one access on each port: lanes 0-1 (or 4-5) on A, 2-3 (or 6-7) on B,
  even lanes through the low half, odd lanes through the high half (`ring_c`, `ring_e`). This
  is why `lut_unwritten`'s south lanes 0-1 returned an earlier west gather's rows: six port-A
  accesses back.
- North of row 5 behaves the same (`ring_top`): the rule is "no core memory in that direction",
  not something of the memory tile.
- The rings survive kernel launches (the harness scrub uses only port-A loads, so port B keeps
  what ran before): such a read depends on the previous kernel. Every probe kernel flushes both
  rings first. The emulator carries them like the data registers (a core reset keeps them,
  `interpreter::state::carried`; host-firmware-dispatch.md, "Core register state across
  launches"); a fresh emulated device starts with zero rings.

Silicon does not write anything for a store into such a direction; the emulator drops it (the
memory tile used to receive core stores and answer core loads south of row 2).

**Emulator.** `interpreter::state::load_ring` holds the rings in `ExecutionContext`; every load
in `MemoryUnit` fetches through `fetch_row` / `ring_row` (`port_line`, `port_quad`,
`port_scalar`, the gather lanes, unpack, sparse fetches, the fused `vlda.ups` / `vlda.conv`,
which now also reach neighbour memory). `NeighborMemory::data` tells core memory, no memory (a
memory or shim tile, or outside the array) and an isolated direction apart; an isolated read
still returns zeros (unprobed). Unit tests: `memory/load_ring_tests.rs` (one per silicon case
family), `state/load_ring.rs`, `memory/neighbor.rs`. EMU == HW on all 12 probe cases.

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

| Gap | Status | Obs. |
|---|---|------|
| count-write latency | at most two bundles (observed); not pinned lower | confirmable: sweep count-write to use distance |
| `d4`..`d7` as 2D operands | the compiler allocated `d0`..`d3` for four walkers; the 3D forms use `d4`..`d7` as the outer dimension, which the 3D captures exercise | confirmable: hand-written asm, d4..d7 |
| counts and sizes above 20 bits or negative | masked to 20 bits on write, compared unsigned; **confirmed 2026-11-10** (see "Large and negative counts") | confirmed |
| west of column 1 | silicon returns the DPU-reserved column 0's contents, which the emulator does not have (`load4x/wrap3` `lut_unwritten`: its west lanes, and through the rings most of its south lanes) | confirmed |
| ring state across kernel launches | **modelled (2026-11-06)**: a core reset keeps both rings, within one emulator instance and, with `XDNA_EMU_STATE_DIR`, across processes. The harness still differs: every phase is a new process on a fresh emulated device (zero rings), while silicon holds the previous case's rings (the scrub loads only through port A), so `lut_unwritten` keeps its 6 port-B lanes | confirmed |
| loads into an isolated direction | `Tile_Control` isolation: the emulator returns zeros and writes zeros to the ring; unprobed | confirmable: load into isolated direction |
| a sparse peek / pop whose fetch is dropped | assumed not to touch the port-B ring (no fetch); unprobed | confirmable: sparse peek with dropped fetch |
| `vldb.compr.*` alignment | the instructions are unmodelled in the emulator and were not probed | confirmable: vldb.compr probe |
| bytes a sparse peek reads past the first fetched line | they come from stale ring slots; silicon and the emulator differ there (sparse_4 / sparse_8, 16 of 64 records, the last 2 bytes of each) | confirmed |
| Chess emits `padds [sp], #imm` (stack adjust in the store slot) around spills in some register-pressure shapes | llvm-aie defines the `ag_pstm_sp_imm` stack adjust only for the A and B load units (`PADDA_sp_imm`, `PADDB_sp_imm`), so its disassembler, and the emulator with it, rejects the bundle (`St DECODE FAIL`). **FIXED 2026-11-10.** The one-def llvm-aie fix (`PADDS_sp_imm`, Xilinx/llvm-aie#1456) is upstream, and the emulator decodes from it. `vec_agu_srs_d` is back to its original d8 form (16-bit sources), whose Chess code carries `padds [sp], #0x20`: silicon r1 == r2, EMU == HW on all 44032 output words, and the golden was recaptured. A Peano install older than the fix still cannot print that bundle, so its llvm-objdump asserts there. A decode sweep of every Chess listing on hand (`decode-sweep`) found no other undecodable encoding. Not an addressing-rule gap | confirmed |
