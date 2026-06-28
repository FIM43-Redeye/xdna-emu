# Vector fuzzer Tier 1 extension: matmul, cmp-flag, shifts, accumulator ops

**Goal.** Close the never-adjacency-fuzzed Tier 1 families before the Phoenix
swap. Reuse the existing typed-pipeline fuzzer (#112) wholesale; this is a
table/generator extension plus two new lowering shapes.

**Families to add (table entries; same edge-weighted input pool, ledger
keys per op/type/mode, credit only on silicon match):**

1. **Matrix engine.** `aie::mmul` shapes per Half-A audit: 4x8x4 i8, 4x4x4 i16,
   2x4x8 / 4x8x4 bf16; mac and mul accumulate paths; matmul-sub and negmul
   variants where Peano supports the intrinsic. Output is an accumulator:
   couple back into the pipeline through the existing SRS coupler (acc ->
   vector slice). One key per shape x variant.
2. **CMP-flag / side-effect family.** sublt, subge, maxdifflt; set_lt/ge/eq
   scalar masks (store mask as u32/u64 to the slice, then rebroadcast to a
   vector lane to keep the chain typed); maxlt/minge mux-with-flags forms.
3. **General shifts.** shl/srl/sra on i8x64/i16x32 (only i32 covered today);
   mode = shift amount class {1, mid, width-1}.
4. **Vector-movement leftovers.** vextract+vinsert pair (extract lane, reinsert
   permuted), valign, vpush/vpop accumulator stack — only where Peano emits
   them; UB-on-silicon variants get the shuffle_fill treatment, documented.
5. **Cascade.** Probe-first: try put_mcd/get_scd through Peano; cascade SS
   routing self-loops to the same core where supported. If Peano can't emit
   them, document as Chess-only and defer (deferred Chess multi-file item).

**Lowering work.** Two new couplers: AccToVec (SRS, mode 0 fixed) and
MaskToVec (store + broadcast). Same per-stage 64B slice banking, GlobalISel
bf16 noinline workaround stays.

**Process** (same as #112): builder subagent -> compile-clean offline (no HW)
-> 50-seed smoke on HW jobs 8 -> overnight campaign to ledger-complete
(>=10 hits/key) -> divergences fixed at root cause, credit only on match.

**Done =** all new keys covered >=10 with 0 divergent/unreachable on Phoenix
silicon, corpus banked alongside the existing one; #113 picks up after.

## Build findings (2026-06-10)

**Cascade probe: Peano compiles put_mcd/get_scd cleanly.** Both the raw vector
form (`put_mcd(v16int32, 1)` / `get_scd_v16int32(1)`) and the accumulator form
(`put_mcd(v16acc32, 1)` / `get_scd_v16acc32(1)`) compile at -O2 with no
GlobalISel issues (probe under `build/experiments/vector-fuzzer-tier1-probe/`).
Cascade table entries are still **deferred**: on AIE2 the cascade output of a
core feeds the neighboring tile, not itself, so a single-tile fuzz kernel
doing put_mcd then get_scd stalls unless stream-switch cascade self-loop
routing is configured -- that routing work (and HW confirmation that Phoenix
supports a self-loop at all) is out of scope for this extension. Compiler
reach is proven; entries can land once routing is in place.

**New GlobalISel landmine (bf16 mmul).** `to_vector<bfloat16>()` of an mmul
accumulator must be SINGLE-USE. Duplicating the converted vector
(`aie::concat(c, c)`) asserts Peano GlobalISel `selectG_AIE_STORE_CONV:
"Expected SSA"` at -O2 and -O1, helper or not. The table entry instead fills
the upper half from the b operand (`concat(c, b.extract<16>(0))`) -- fully
defined, single conv use, compiles clean. shuffle/insert barriers do not help.

**Other probe deltas from the plan:**
- `aie::sub_lt`/`sub_ge` do not exist in this aie_api; the flag-writing
  sub-compare forms are reached through `aie::max_cmp`/`min_cmp` (vmax_lt /
  vmin_ge with mask GPR), with the mask folded into lanes via broadcast-add.
- Scalar lane insert is `vector::set(value, idx)` (no scalar `insert`
  overload).
- valign is not a separate aie_api spelling; the concat-shift form is already
  covered by the existing shuffle_up_fill/shuffle_down_fill entries.
- mmul i16 4x4x4, i8 4x8x4 (mul+mac) and bf16 4x8x4 (mul+mac) all compile.
