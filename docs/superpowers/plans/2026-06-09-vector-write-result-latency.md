# Vector result visibility via the AIE2 bypass/forwarding network

**Task #108 follow-on.** Supersedes the earlier "bundle-seq write-deferral"
content in this file (that approach is abandoned -- see "Why the previous
approach was wrong").

## Goal

Model AIE2 vector-register result visibility *faithfully*, as a
producer-result x consumer-operand **bypass/forwarding network** derived from
the llvm-aie TableGen itinerary data -- not a per-op flat latency, and not a
targeted store-lag hack. We build the general machinery now, including paths
not currently exercised by any kernel, because the aim is global fidelity: a
future kernel that hits a NoBypass-producer -> ALU-consumer edge should already
be correct.

## Background: what the hardware does

Every AIE2 instruction's itinerary (`AIE2Schedule.td`) gives, per operand:
- an **operand cycle** (`operand_cycles[i]`): operand 0 = result/def latency,
  operands 1.. = source read cycles (almost always 1).
- a **bypass class**: one of `MOV_Bypass` (scalar/vector ALU, W/X file),
  `VEC_Bypass` (accumulator/CM domain: MAC/MUL), or `NoBypass` (loads, stores,
  cross-domain moves, flags).

LLVM's `getOperandLatency` (MCInstrItineraries.h) computes the def->use latency:

```
latency = DefCycle - UseCycle + 1
if (Def.Bypass[defIdx] == Use.Bypass[useIdx]  &&  class != NoBypass)
    latency -= 1            // AIE2 getNumBypassedCycles: flat -1 per forward
```

With the typical `DefCycle = 2`, `UseCycle = 1`: base latency 2, forwarded 1.

### The four vector-register cases (all verified against compiled two_col/bf16)

| Producer (def) | Consumer (use) | Forward? | Visible at |
|---|---|---|---|
| VINSERT/VSEL/VBCST/VEXTBCST/VADD (`MOV_Bypass`, L=2) | vector ALU op (`MOV_Bypass`) | yes | issue+1 |
| same | store data (`NoBypass`) | no | issue+2 |
| VMOV BM->X (`NoBypass` def, L=2) | anything | no | issue+2 |
| (load VLDB `NoBypass`, L=7) | anything | no | issue+7 (existing load path) |

Compiled-schedule evidence (chess `threshold.o.lst`):
- `VINSERT.8 x1` -> `VEXTBCST.8 x0, x1` reads x1 **one bundle later** (issue+1,
  ALU bypass). This is the edge the bundle-seq deferral broke.
- `VSEL.8 x4` -> `VST wh4` reads x4 **two bundles later** (issue+2, store).
- bf16 `VMOV x2,bml0` -> next-bundle `VST wl2` must read the **old** x2
  (issue+2): NoBypass def, store consumer.

## Why the previous approach was wrong

The bundle-seq write-deferral deferred the *architectural write* by the op's
full result latency (2) for **all** consumers. That over-delays the ALU bypass
consumers the compiler scheduled at issue+1 (`VEXTBCST` reads stale `x1`), which
corrupts two_col's threshold (mask becomes all-true -> entire output buffer
`0xffffffff`; ~1754 element mismatches). The bug was never cycle-vs-bundle; it
was that ALU forwarding (issue+1) and store reads (issue+2) need *different*
visibility for the *same* write. A single commit bundle cannot express that.

## The model

One resolution rule. A vector-register write records `(reg, value,
issue_bundle B, l_def, def_bypass)`. A read of `reg` by a consumer with
`use_bypass` at `cur_bundle` sees the write iff:

```
forward     = (def_bypass == use_bypass) && (def_bypass != NoBypass)
visible_at  = B + l_def - (forward ? 1 : 0)        // use_cycle assumed 1
visible     = visible_at <= cur_bundle
```

The read returns the value of the **latest-issued** write to `reg` that is
visible to it, else the committed (fully-landed) value. Within-bundle reads see
old values automatically (a write issued this bundle has `visible_at >=
cur_bundle+1`), so this **subsumes** the per-bundle vector snapshot.

Consumer classification for the **vector register file** (faithful, because the
itinerary marks vector-data use operands `MOV_Bypass` and store-data operands
`NoBypass`):
- **store-data reads of vector regs -> `NoBypass`**
- **all other vector-reg reads (compute) -> `MOV_Bypass`**

The accumulator file (`VEC_Bypass`, MAC L=5) keeps its existing deferral path
for now; unifying it under this model is a noted future extension (see Scope).

## Implementation steps

### 1. Extract per-opcode def bypass (data)
`InstrInfo` already carries `latency` (= `operand_cycles[0]`) and an opaque
`sched_class` index. Add the **operand-0 bypass class** per opcode.
- Preferred: extend the C++ FFI shim (`aie2_get_instr_info`) to also return the
  def forwarding id from `InstrItineraryData::Forwardings[FirstOperandCycle]`
  (0 = NoBypass; nonzero = MOV/VEC). Add `def_bypass: u16` to `RawInstrInfo` /
  `InstrInfo`. This is the same runtime source as `latency` -- cleanest.
- Alternative (no C++): if `ProcessorModel.itineraries` (already holds
  `bypasses: Vec<String>` by class name) can be linked to opcodes via a
  sched_class-index->name map, use that. Decide at implementation time; prefer
  the FFI route if the index->name link is absent.

Represent as an enum `Bypass { No, Mov, Vec }` in archspec; expose
`LatencyTable::def_bypass(llvm_opcode) -> Bypass` alongside the latency lookup.

### 2. Pending-write model (context.rs)
Replace the bundle-seq vector deferral with bypass-aware in-flight writes.
`PendingWrite` (or a dedicated `PendingVecWrite` list) for vector regs carries
`{reg, value, issue_bundle, l_def, def_bypass}`. Keep `bundle_seq` as the
issue-bundle clock (it is the right clock: issue-slot-relative). Drop
`ready_bundle` keyed on a single commit bundle.

- `queue_vector_reg_write(reg, value, l_def, def_bypass)` and the wide variant.
- `commit_pending_writes`: move a vec write into the committed file once
  `cur_bundle >= B + l_def` (its max/NoBypass visibility), latest-issue-wins.
- `read_vector(reg, use_bypass)`: resolve per the rule above (committed value
  overridden by the latest visible in-flight write).

### 3. Read-site classification (vector_helpers / memory / cascade / stream)
- Compute reads (`read_vector_operand`, `get_wide_vec_source`, etc.) ->
  `use_bypass = Mov`.
- Store-data reads (memory `execute_vector_store` / `read_store_register`) ->
  `use_bypass = No`.
- Audit every `ctx.vector.read(...)` site; route through the bypass-aware read
  with the correct class. (Cascade/stream vector reads: classify per their
  itinerary; default compute=Mov unless evidence says otherwise.)

### 4. Producer plumbing (cycle_accurate.rs)
At slot dispatch, look up the producing op's `(l_def, def_bypass)` from the
latency table and pass them into the vector write. `result_latency` becomes
`(l_def, def_bypass)`.

### 5. Revert the bundle-seq deferral
Remove `ready_bundle`-keyed vector commit, the `result_latency: u8` single
field's deferral use, and the now-unused bits. Keep:
- the float MAC latency = 6 fix (`VECTOR_MAC_F`) -- independent, correct.
- the matmul accumulator deferral (`queue_matmul_accum_write`) -- accumulator
  file, separate path.

### 6. Remove debug instrumentation
Strip `XDNA_EMU_VEC_DEBUG` / `[VECDBG-W/WW/S]` from vector_helpers.rs and
memory/mod.rs.

## Test plan (TDD -- write the failing test first)

Unit tests (context.rs) for the four matrix cases, each driving `bundle_seq`:
1. `Mov` def -> compute read visible at issue+1, not issue+0.
2. `Mov` def -> store read visible at issue+2, not issue+1 (the bf16 edge).
3. `No` def (VMOV BM->X) -> compute read visible at issue+2, not issue+1.
4. `No` def -> store read visible at issue+2.
5. Wide (512-bit) write splits and both halves resolve correctly.
6. Within-bundle read-old still holds (snapshot subsumed).
7. Latest-issue-wins when two writes to one reg are in flight.

Then, in order:
- `cargo test --lib` green.
- `cargo build -p xdna-emu-ffi`.
- Bridge `--no-hw`: `vec_mac_bf16` PASS, `two_col` PASS,
  `matrix_multiplication_using_cascade.cascade` PASS.
- Full `--no-hw` sweep: **0 regressions** vs the 06-08 baseline.

Only then commit, split into: (1) float MAC latency (already staged),
(2) bypass-network vector visibility model.

## Scope boundaries / future work
- This task models the **vector register file** bypass network. The
  **accumulator file** (`VEC_Bypass`, MAC->MAC forwarding at L-1) keeps its
  current deferral; folding it into the same resolution rule is a follow-up.
- Per-operand *use*-bypass is currently collapsed to compute(`Mov`) vs
  store(`No`), which is faithful for the vector file. If a future op reads a
  vector reg via a genuinely different use bypass, extend step 3 to extract the
  real per-operand use bypass from the itinerary.
