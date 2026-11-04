---
class: vector-compute
subsystem: AIE2 vector unit -- result forwarding/bypass network + special-value semantics
posture: mostly-resolved -- the forwarding model is HW-verified for the bf16 matmul class; the residuals are scoped deferrals and one uncharacterized HW-state regime, not unvalidated model
status: 2 deferred (bypass-matrix fold, benign operand alignment); NaN payload resolved (datapath regime)
---

# Vector Compute Gaps

The AIE2 per-operand bypass/forwarding matrix is implemented and validated
against the chess bridge sweep (89/89 PASS, 0 regressions) **and against real
Phoenix (NPU1) silicon**: `vec_mac_bf16`, `two_col`, and all three
`matrix_multiplication_using_cascade` variants (plain/buffer/cascade) PASS
HW==EMU on the bridge (2026-06-09), with `vec_mac_bf16` and `two_col` also
diffing CLEAN on the trace comparison. The W/X-file forwarding model is therefore
HW-verified for the bf16 matmul class; the gaps below are scoped deferrals, not
unvalidated model.

## Result forwarding / bypass network

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| Accumulator/CM-domain (`VEC_Bypass`) result visibility | folded into the accumulator timing model below: every accumulator write is queued with its itinerary def cycle and bypass class, every read carries its use cycle and class | `src/interpreter/state/registers.rs` (`AccumulatorRegisterFile::resolve`), `src/interpreter/state/context.rs` (`queue_acc_write*`) | **RESOLVED** 2026-10-31: 468 producer/consumer pairs measured on silicon, all reproduced exactly (see "Accumulator read/write timing"). |
| `source_forward` positional alignment for merged memory operands | `SlotOp::source_forward` aligns exactly for compute ops; may drift for fused memory operands that merge address/data into a single TableGen operand list position | `src/interpreter/bundle/slot.rs` (`FIXME(source-forward-memory-alignment)`); `src/interpreter/decode/operand_extraction.rs` | **DOCUMENTED / benign.** The bypass model only reads vector-register compute sources; memory operands use `NoBypass` or are not vector-reg reads, so this positional drift does not affect result visibility decisions. |

## Accumulator read/write timing

One rule covers every accumulator write and read the emulator models, derived from a hand-scheduled
probe on silicon and equal to the TableGen itinerary operand cycles.

**Rule.** Let `def` be the accumulator result cycle of the producer's itinerary (`operand_cycles` of
the def), `use` the cycle of the consumer's accumulator source, and `fwd` 1 when both have the same
non-`NoBypass` class (`VEC_Bypass`; `MOV_Bypass` is the other class). A consumer issued at bundle `b`
sees a write issued at bundle `p` exactly when

    b + use >= p + def + 1 - fwd

The distance at which a consumer first sees the new value is `t = def - use + 1 - fwd`. A consumer
whose use cycle is 2 or more (the MAC family, `vadd`) reads at `issue + use`, so a same-bundle `vmov`
write is already visible to it; a consumer with use cycle 1 (`vst`, `vsrs`, `vmov`, `vconv`) reads at
issue. Dense and sparse forms are identical, producers and consumers alike. The write queue is
`AccumulatorRegisterFile::pending`; a read resolves the overlay of the writes visible to its
(`use`, class) pair (`resolve`, `read_with`), and the bundle clock is `ExecutionContext::bundle_seq`.

**Fitted table** (silicon, 22 producers x 28 consumers, 468 pairs, 456 exact and 12 censored at the
first RAW distance; free difference-constraint fit with 0 inconsistent edges over the 404 non-forwarded
pairs, the 64 forwarded pairs all exactly one less). Every row agrees with TableGen.

| Producer | itinerary | def | class | visible from issue + | Consumer | itinerary | use | class |
|---|---|---|---|---|---|---|---|---|
| `vmul`, `vmac`, `vaddmac` (int, dense and sparse) | II_VMUL / VMAC / VADDMAC | 5 | VEC | 6 | MAC accumulator source (int, `acc1`/`acc2`) | II_VMAC / VADDMAC | 3 | VEC |
| `vmul.f`, `vmac.f`, `vaddmac.f`, `vaddmsc.f` | II_VMULf / VMACf / VADDMACf | 6 | none | 7 | MAC accumulator source (float) | II_VMACf / VADDMACf | 3 | none |
| `vadd` (int) | II_VACC | 5 | VEC | 6 | `vadd` (int) accumulator source | II_VACC | 3 | VEC |
| `vadd.f` | II_VACCf | 6 | none | 7 | `vadd.f` accumulator source | II_VACCf | 3 | none |
| `vclr` | II_VCLR | 5 | VEC | 6 | `vsrs`, `vsrsm`, `vst.srs` | II_VSRS / VSRSM / VST_SRS | 1 | none |
| `vmov cm,cm` / `bm,bm` / `bm,x` | II_VMOV_CM / VMOV_X_BM_XM | 2 | none | 3 | `vmov x,bm` / `cm,cm` / `bm,bm` | II_VMOV_X / VMOV_CM | 1 | none |
| `vups` | II_VUPS | 3 | none | 4 | `vst` of `amll..amhh` | II_VST_AM | 1 | none |
| `vlda.ups` | II_VLDA_UPS | 9 | none | 10 | `vconv` | II_VCONV | 1 | none |
| `vlda am` (quarter), `vlda.conv` | II_VLDA_AM / VLDA_CONV | 7 | none | 8 | | | | |

The old per-kernel patches this replaces (sparse MACs reading their accumulators at issue, sparse
`vaddmac.f` landing at 6 against the dense 7, `vmov cm,cm` landing at `result_latency`) were each a
compensation for the missing "+1" and for modelling the MAC read at issue instead of `issue + use`.

**Probe and evidence.** `tools/gen_acc_timing_probe.py` generates raw-asm kernels (assembled by
llvm-mc, written as a suite and built/run by `python3 -m harness suite`) that place a
producer and a consumer at signed distance `s` (RAW `+d`, WAR `-d`, `d` = 0..9) and classify each
output word OLD or NEW against two in-kernel references (`tools/analyze_acc_timing.py`);
`tools/fit_acc_timing.py` fits the read and write cycles with no TableGen input and then compares
with the itineraries. Silicon was run twice and is bit-identical (165 kernels, 9690 points). Evidence:
`build/experiments/sparse/acc-timing/probe-v2/` (`r1`, `r2` raw, `analysis_r1.txt`,
`thresholds_r1.json`, `FIT.md`; emulator runs `e2` and `e3`; at threshold level `compare_r1_e2.txt` shows 0 of 494 pairs differ, byte level see below).
`src/interpreter/state/acc_timing_golden.rs` replays the ~370 measured thresholds against the
register file.

**Accumulator loads (2026-11-04).** Plain `vlda amll` / `vlda amlh` were not producers in that probe,
and the emulator committed them like scalar loads, straight into the register file at issue + 7: one
bundle early against the rule for a MAC reader (use 3) and on the wrong side of a same-bundle float
MAC result (both landed in one bundle and the MAC won). The Chess complex-float `mac_elem_8` /
`msc_elem_8` kernels schedule exactly that (`vlda amlh0, [p7, #32] ; vmul.f bml0, ..` then the
final `vadd.f` reads `bml0` expecting the loaded accumulator), so EMU lost lanes 8..15 of the
accumulator input. The probe grew `vlda_amll`, `vlda_amlh`, `vlda_conv` producers and write-after-write
points (`--waw A:B`, A issued `s` bundles after B, references at `s = +-14`): directory
`build/experiments/harness/cfloat/ldtiming` (37 kernels, 2228 points, silicon r1 == r2), plus
`ldtiming2` (two narrow `vmov am, q` subjects as controls, 8 kernels, 496 points, r1 == r2). Every
threshold is the rule with def 7: MAC/`vadd` readers from distance 5, use-1 readers from 7, and in
WAW the later-landing write wins (`vmul.f`: the load wins from `s = 0`; int MAC, `vclr`: `-1`;
`vmov bm`/`cm`: `-4`; `vups`: `-3`; `vlda.ups`: `+2`), the later-issued one on a tie. The loads are
now ordinary accumulator-file writes (`ExecutionContext::queue_accum_load` ->
`queue_lanes_write`, def 7, no bypass); EMU == silicon on all 2724 points;
`execute/acc_load_timing_goldens.rs` replays thresholds through the executor. `vlda.conv` already
followed the rule. The one change in behaviour beyond the rule: an accumulator load in flight across a
taken branch no longer gets the scalar loads' extra branch cycle (`delay_pending_writes`), like every
other accumulator write.

**Resource collisions (MODELLED 2026-11-01; open items listed at the end).** Compilers never
emit a schedule that holds one TableGen unit twice in a cycle, but a hand-written schedule can, and
silicon is deterministic about what happens (every probe run twice, byte-identical). The first probe
(`probe-v2`) saw 24 such points among 9690; the collision probes (`tools/gen_collision_probe.py`,
`build/experiments/collisions/probe` .. `probe12`, 12 directories, 11394 points) take every pair of
instructions whose resolved reservations overlap on an accumulator-datapath unit and run it at the
colliding issue distance plus the neighbouring distances as controls. The emulator now matches silicon
on all of them (counts below). The 24 `probe-v2` points were three cases: a `vmov` source read in
the cycle a `vadd`/`vaddmac` reads its second accumulator (16 points: rule 3), a `vclr` landing in the
same cycle as a float MAC/add result (6: rule 1 or 2 by port) and a `vlda.ups` landing with a
`vaddmac.f` on one `bm` (2: rule 2, the `vaddmac.f` port wins). Nothing here changes a schedule the resource model allows: the
resolver only acts when two tagged writes share a register file and a landing cycle, and the ISA
suite (426 PASS, 0 FAIL) and the bridge suite (`--no-hw`: Chess 90 pass, 0 fail; Peano 105 pass, 1 fail,
the known `dynamic_chain_passthrough/8`) did not move.

*Where the model gets its facts.* No unit name, stage number or instruction name is in the emulator.
`SlotOp.ports` (`PortUse`, built in `decode/ports.rs` from the decoder FFI's resolved stages) carries
unit indices:

- **Write port**: the unit the resolved itinerary holds in the cycle before the result lands
  (`operand_cycles[0] - 1`) that writes the destination's register file. Which units write which
  file is derived from the itineraries and the destination register classes of their instructions
  (`TblgenOutput::itinerary_output_classes`, from the instruction records): the vector file has
  three write ports and the accumulator file two. A class that reserves ports of both files
  (`vshuffle`, `vbcstshfl`: TableGen over-reserves, silicon collides only on the unit of the file it
  writes) is resolved by the destination register. Priority between two ports is unit-index order.
- **Accumulator read port**: the units held (not merely reserved) in the cycle of the first
  accumulator source's use (`use_cycle - 1`). Instructions the tables hold nothing for there
  (`vmac`, `vmsc.f`, `vneg*`) have none and, as silicon showed, never collide on a read.
- A write is stamped at issue with its port, an issue sequence number and its landing bundle
  (`issue + result operand cycle`); the stamp is carried through every deferred path (`PendingAccAdd`,
  `PendingMatmul`, `PendingUpsLoad`, `PendingUnpack`) and the cycle-based load commits (a stand-in
  entry in the register file takes part in the resolution; the commit applies its verdict).

*Write side* (`state/write_collision.rs`, a pure function from chunks to chunks). The accumulator
file has four 256-bit banks (quarter `q` of half `h` of any `cm` is bank `2h + q`), the vector file
two (`wl`, `wh`); a write is a list of 256-bit chunks numbered in the order the instruction produces
them.

1. *Same port, same landing cycle.* The port has one data lane per chunk ordinal; lane `k` carries
   ordinal `k` of the latest-issued writer that has one. Bank `p` takes lane `p` (else the lane of the chunk's own ordinal, else
   lane 0, when no writer drives lane `p`) and goes to the register of the latest-issued writer that targets `p`; earlier
   writers' chunks on that bank are dropped. Banks nobody targets are untouched; registers do not
   matter, the bank is the resource. A port with one writer passes through.
2. *Different ports, same landing cycle.* Only chunks on the same register and bank conflict; the
   lower-indexed port wins with its own data. Different registers both land.

*Read side.*

3. *Accumulator read-port steal* (`CM_RM` in the tables). A reader that reads its accumulator late
   (`vadd`/`vaddmac` family: cycle `issue + 2`) gets, for its **last accumulator source** and for the banks
   the mover reads, the source of a **mover** that read through the same port in that cycle. A mover is
   an instruction that reads an accumulator and writes through the last write port of a file: the
   `vmov` family out of an accumulator and `vsrsm`, and exactly those (pinned by
   `decode/ports_goldens.rs`). Later-issued movers win a bank. `vsrs`, `vconv`, `vst*`, `vfloor` read
   through the same port but do not steal.
4. *SRS shifter wired OR* (`UPPER_SRS`). Two instructions that hold the shifter's second read unit in
   the same bundle - a mover with a 512-bit source (`vsrsm`) and an instruction with a full 1024-bit
   `cm` source (`vsrs` of an `s64` accumulator, `vst.srs`, `vsrs.s8.s32`) - share the shifter's upper
   half: the mover's source is OR-ed with the partner's high half and the partner's high half with the
   mover's source, per reader (a shared source register is not OR-ed into the partner's low half).
   Instructions with a 512-bit source (`vsrs bm`) do not take part.
5. *s8 partner.* `vsrs.s8.s32` / `vst.srs.s8.s32` do not convert the OR-ed upper bus: their upper
   16 bytes are the low halfword of each of the first eight 32-bit words of the mover's result.
6. *Mixed lane modes.* With a nonzero shift the shifter takes the lane mode of the wider of the
   pair: a 32-bit-lane `vsrsm` beside a 64-bit-lane partner shifts the OR-ed bus as 64-bit lanes
   (carrying across the two words of a lane) and returns the low 32 bits of each shifted lane, whatever
   its own element type. A 64-bit-lane `vsrsm` beside a 32-bit-lane partner is its own normal case.
   Pinned at two shifts (4 and 9) over all pairs.

Producers that were written at issue are timed now: `vlda.conv` lands at its itinerary cycle (7)
like every other accumulator producer, and so do the plain `vlda am` loads (2026-11-04, see
"Accumulator loads" above).

*Narrow odd-bank writers in the accumulator file* (2026-11-04). In rule 1 a narrow (single-chunk)
accumulator writer on an odd bank also drives lane 1 of its port, with or without a cascade read:
a `vlda amlh0` or `vmov amlh0, q1` issued after a `vlda.ups` / `vups` of `bml0` and landing in the
same cycle puts its data in both banks of `bml0` (on bank 0, `amll0`, it leaves lane 1 alone). The
vector file keeps the plain rule (a narrow `wh` writer does not, probe6). Evidence:
`cfloat/ldtiming` (`vlda_amlh` vs `vlda_ups` at `s = +2`) and `cfloat/ldtiming2` (`vmov_amlh_q` vs
`vups` at `+1`, vs `vlda_ups` at `+7`); the legacy collision probes stay exact (see the sweep line in
`build/experiments/harness/PROGRESS-cfloat-mac.md`). It subsumes rule 2 of the cascade section for
accumulator destinations.

*Inferred, not read off the tables* (silicon-derived predicates, each pinned by a test to exactly the
observed set): the mover predicate; the "secondary read unit" (highest-indexed unit held in the read
cycle when two are); the steal victim being the last accumulator source and a late reader (offset > 0);
the steal covering the banks the mover reads; the OR needing a full-source partner and a half-source
mover; the s8 partner selected by its output element type; the wider-lane-mode rule; the lane
data model of rule 1 (ordinal vs bank). All of them reproduce every silicon point; the cases the
probes could not reach (a `bmh` mover, three writers in one cycle) follow the same rules by
construction and are unconfirmed.

*Evidence and exact counts* (EMU vs silicon run r1, differing points / points; `before` is the
emulator at the start of this work, `after` the model):

| Probe (dir under `build/experiments/collisions/`) | Content | before | after |
|---|---|---|---|
| `probe` | v1: every pair of accumulator-datapath units, 20 kernels (1122 points) | 85 | 0 |
| `probe2` | v2: cross-domain pairs, cross-port write pairs, reads the tables leave out (2601 points) | 294 | 0 |
| `probe3` | 34 more instructions (every SRS shape, `W_*`/`CM_WM` holders, shuffles, `vmov am,q`) (3356 points) | 275 | 0 |
| `probe4` | `UPPER_SRS`, one-hot bit patterns that tell halves apart (464 points) | 64 | 0 |
| `probe5` | same with odd accumulator lanes nonzero (131 points) | 8 | 0 |
| `probe6` | vector-file family incl. `wh` destinations and cross-port pairs (1541 points) | 208 | 0 |
| `probe7` | 28 single-instruction references (sibling encodings) (28 points) | 0 | 0 |
| `probe8` | s8 partners x the four `vsrsm` shapes, shift 0 (93 points) | 16 | 0 |
| `probe9` | same pairs, SRS shift 4 (93 points) | 16 | 0 |
| `probe10` | all `UPPER_SRS` pairs, one-hot patterns, shift 4 (655 points) | 90 | 0 |
| `probe11` | same pairs, odd lanes nonzero, shift 4 (655 points) | 90 | 0 |
| `probe12` | same pairs, odd lanes nonzero, shift 9 (confirms the lane-mode rule; made after the baseline) (655 points) | n/a | 0 |
| `neg` | `vneg*` / `vnegmac` family (46 points) | 1 (`vnegmac`, fixed here) | 0 |
| `sparse/acc-timing/probe-v2` | the accumulator timing probe the 24 points came from (9690 points) | 24 | 0 |

Replays: `src/interpreter/execute/collision_goldens.rs` runs 99 silicon pair points (one per port group,
configuration and outcome signature) through the cycle-accurate executor; the resolver has unit tests
for each rule and `tools/mutation_check_collisions.py` breaks each rule and confirms a test fails
(`build/experiments/collisions/mutation_check.txt`).

*Chess scalar-port finding.* The checker (`collision-check elf`) over compiled code finds no violation
of the accumulator or vector register-file ports in either compiler. It does find 666 over the 98
Chess bridge ELFs (all scalar-side: `P_RM_PORT` 471, `P_WM_PORT` 159, and the `M`/`DN`/`DJ`/`DC` write
ports 9 each) against 0 over the 108 Peano ELFs, and 496 `PART_WORD_STORE` hits in the ISA-suite
kernels: TableGen is more conservative than Chess on the pointer and modifier ports (and on the
part-word store unit), so Chess legitimately schedules through reservations the model says are held.
Those are real silicon schedules, and silicon is fine with them; they are not collisions in the
sense of this section.

### Cascade-read (`VMOV_CASCADE_READ`) collisions

Derivation. TableGen's `II_VMOV_CASCADE_READ` = `[EmptyCycles<1>, AvoidSemaphore<6>, EmptyCycles<5>,
PrefixCycle<W_WA_PORT>, SimpleCycle<CM_WM_PORT>]` (latency 7): `collision-check ops` gives
`W_WA_PORT:6, CM_WM_PORT:6` plus the reserved `SEMAPHORE:1..6`. The register-port side puts it in the same
design space as every other writer, so `tools/gen_collision_probe.py` (after `enable_cascade()`, three
catalogue entries `vmov {x,bml,bmh}, SCD`) derives its pairs: 317 pair points against 88 catalogue
instructions, 634 controls, 375 solo/zero references, config `sep` and `dstshare`.

Probe. `tools/gen_cascade_collision_probe.py` runs those kernels on the topology harness: the
consumer tile (0,2) runs the generated kernel plus a cascade prologue (enable SCD, a counted loop so the
producer has filled the cascade), the producer tile (0,3) enables MCD and streams exactly as many
identical words as the kernel has SCD reads. Directory `build/experiments/cascade-collisions/probe3`:
22 kernels, 1326 points, silicon r1 == r2 byte-identical on every kernel, no TDR; `analyze_collision_probe.py
check` reports every pair/control/reference as designed (alignment guard in the tool: an unaligned loop
label re-runs its setup on silicon, see core-compute-timing.md).

Result. The existing resolver explained every point whose destination is an `x` register and all but
36 accumulator-destination points; two rules, both read off probe3, close the rest:

1. A chunk on a bank no writer drives a lane for takes the lane of **its own ordinal** before lane 0
   (`bmh` write beside a `bm` mover: both of its chunks take the mover's two lanes, not lane 0 twice).
2. **Beside a cascade read with an accumulator destination, a narrow (single-chunk) writer on an odd
   bank also drives lane 1** of the port: against `vmov bm, SCD` an `am` quarter mover on bank 1 or 3
   supplies the data of the read's second chunk (bank 3: the read's second chunk lands the mover's data;
   banks 0 and 2: it keeps the SCD data); against the plain 512-bit movers the old rule stands. `x`
   destinations need no cascade-specific rule.

EMU vs silicon after both rules: 0/1326 points differ (pair 0/317, control 0/634, ref 0/375); the old
collision probes stay exact (see the sweep line in `build/experiments/cascade-collisions/PROGRESS.md`);
75 probe3 points are replayed as goldens (`collision_goldens_cascade_data.rs`,
`every_silicon_cascade_read_collision_replays_exactly`), and the mutation check has seven new mutations
(`build/experiments/cascade-collisions/mutation_cascade.txt`). The read's own timing (hold at the fifth
bundle after the vmov, destination visible 7 issued bundles after it, no interlock) is in
core-compute-timing.md.

(2026-11-04: rule 2 holds in the accumulator file without a cascade read too, see "Narrow odd-bank
writers in the accumulator file" above; it stays as written for `x` destinations.)

Open within this section: rule 2 is an observation about narrow odd-bank writers beside a cascade read;
the probe cannot say whether the cause is the narrow writer's lane or the read's second chunk, and only
the `am` quarter movers exercise it. The `SEMAPHORE` side is unprobed (above).

**Open (not modelled), each with its reason.**

- `VMOV_CASCADE_READ` collisions were **modelled** 2026-11-01 (section below). Still unprobed: the
  `SEMAPHORE` reservation (the itinerary holds the semaphore unit for the six cycles after the vmov;
  lock `acq`/`rel`, the stream moves and `vmov MCD` also hold it), because a probe needs a lock or a
  stream source under the pair and the single-bank accumulator probes never touched them; a cascade
  read colliding with a second cascade read (the two cannot issue in one bundle and are six apart on
  the port); and the AIE2P cascade.
- `UPS_UNIT` cannot be separated from `CM_WM_PORT`: every instruction that holds the UPS unit holds
  the write port exactly two cycles later (`vups` 0 and 2, `vlda.ups` 6 and 8), so two instructions
  colliding on the UPS unit collide on the write port at the same instant and no probe can tell which
  one silicon arbitrates on. The model treats the pair as the write-port collision it is.
- Unprobed classes follow the same rules without silicon: a `vunpack` landing on the vector file
  (`W_RS`, the third write port). (`vlda.am` quarters were probed 2026-11-04, see "Accumulator
  loads".)
- Unconfirmed extrapolations listed under "Inferred" above.
- Collisions the tables forbid on units that are not an accumulator or vector register-file port
  (pointer, modifier and scalar ports, the part-word store unit) are not modelled; Chess uses them
  on purpose (above).
- Tied consumers (`vaddmac*` `acc1`, destination equals source) have no WAR points of their own and
  are censored at `t = 1` (unchanged).

## Special-value NaN payload (`vadd.f`/`vsub.f`, bf16 + fp32)

> Superseded for the accumulator-to-accumulator `vadd.f`/`vsub.f`/`vnegadd.f`/
> `vnegsub.f`/`vneg.f` by the fp32 accumulator section at the end of this file:
> a fresh 2026-10-29 NPU1 capture (`vec_acc_addsub_f32`, three bit-identical
> runs) follows the MAC datapath's NaN rule on every lane, with no second
> regime observed. The text below is the older two-regime reading.

**RESOLVED 2026-06-11** -- full mechanism and evidence in
`xdna-emu-priv: research/specs/2026-06-11-nan-payload-datapath-regime.md`.

The result NaN payload when an operand is a NaN has **two real silicon regimes**:
a **datapath regime** (dominant) where the payload is the actual exp-255
mantissa-datapath sum `r`, and a **canonical regime** (rare) that forces payload
1. The regime is residual float-unit HW state -- global across lanes,
deterministic within a session, but it varies across sessions and survives a
driver reload. This is **not** bf16-specific, **not** two-NaN-operand-only, and
**not** in the bf16 narrow -- it is the shared `aie2_acc_fp32_add` datapath
(fp32 shows it identically), and `+Inf + -NaN` is exactly the trigger.

The model now produces the **datapath regime** (FIM43-Redeye's call: it is what the ALU
computes; canonical is a residual-state suppression). One change in
`aie2_acc_fp32_add`: `use_r = (a_nan || b_nan) && !overflow` lets the datapath
sum `r` through instead of zeroing it. Validated **8160/8160** against fresh
dominant-regime NPU1 captures (bf16+fp32, add+sub); `cargo test --lib`
3408/3408; seed 6159 via EMU now reproduces `0xFF8C`.

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| special-value NaN payload: datapath vs canonical regime | HW has two residual-state regimes for `vadd.f`/`vsub.f` NaN-input payloads; we now model the dominant **datapath** regime (payload = exp-255 mantissa sum `r`). HW in the rare **canonical** regime forces payload 1 and would mismatch the payload bits (a HW-state artifact, not a model bug). | `src/interpreter/execute/vector_float.rs` (`aie2_acc_fp32_add`, `use_r` gate). | **RESOLVED (datapath regime) 2026-06-11.** Validated 8160/8160 vs dominant-regime HW. Residual open: *what* flips the regime is uncharacterized (not driver reload); low priority since the payload is functionally dead. See the design note. |


## SRS rounding/saturation and f32->bf16 conversion

**HW-verified** against NPU1 (Phoenix) silicon captures of 2026-10-28
(`tools/golden/silicon/vec_srs_i32*.json`, 77 kernels; `vec_conv_bf16*.json`,
27 kernels). `srs_lane` / `shift_round` (`src/interpreter/execute/vector_srs.rs`)
and `f32_to_bf16` (`vector_float.rs`) are checked lane by lane against every one
of those files by `matches_every_silicon_srs_capture` and
`matches_every_silicon_bf16_capture`. The kernel parameters come from
`tests/vector-verify/<kernel>/`; the captures cover every `crRnd` value 0-15,
`crSat` none/saturate/symmetric, and SRS shifts 0, 1, 2, 3, 4, 5, 8, 12, 16, 24, 31.
Mode names follow llvm-aie `aiev2_defines.h`.

**Shift.** The effective shift equals the configured shift at every shift tried:
the unrounded result is exactly `clamp16(v >> s)`. There is no hidden bias, so the
former `srs_shift_bias` architecture parameter was removed.

**Rounding.** Let `q = floor(v / 2^s)` and "dropped" the low `s` bits of `v`
(two's complement, so for negative `v` the dropped bits are those of `v - 2^s*q`).

| crRnd | name | rule |
|-------|------|------|
| 0 | floor | `q` |
| 1 | ceil | `q+1` if any dropped bit set |
| 2 | sym_floor | toward zero: `q+1` if inexact and `v < 0` |
| 3 | sym_ceil | away from zero: `q+1` if inexact and `v >= 0` |
| 8 | neg_inf | nearest; tie goes to `q` |
| 9 | pos_inf | nearest; tie goes to `q+1` |
| 10 | sym_zero | nearest; tie toward zero |
| 11 | sym_inf | nearest; tie away from zero |
| 12 | conv_even | nearest; tie to the even result |
| 13 | conv_odd | nearest; tie to the odd result |
| 4, 6 | undefined; `RoundingMode::InexactEven` | if inexact, the even one of `q`, `q+1`; exact values unchanged |
| 5, 7 | undefined; `RoundingMode::InexactOdd` | if inexact, the odd one of `q`, `q+1`; exact values unchanged |
| 14 | undefined; alias of 12 | |
| 15 | undefined; alias of 13 | |

The undefined modes are real and reachable (`RoundingMode::from_raw` maps all 16
`crRnd` values; only values above 15 give `None`). 4 and 6 are indistinguishable
on silicon, as are 5 and 7, and 14/15 equal 12/13. The data separates 4/6 from
conv_even (13 of 64 lanes differ) and 5/7 from conv_odd (11 of 64).

**Saturation.** Rounding happens first, then: `crSat` none wraps to the output
width (low bits kept); saturate clamps to `[-2^(n-1), 2^(n-1)-1]`; symmetric
clamps to `[-(2^(n-1)-1), 2^(n-1)-1]`. A rounding carry is saturated, not wrapped.
Unsigned outputs clamp to `[0, 2^n-1]` (symmetric makes no difference). Int16
output is what the captures exercise; the unsigned and 8/32-bit paths follow the
same rule but have no dedicated silicon capture.

**f32 -> bf16 (`to_v16bfloat16`, governed by `crRnd`).** The modes act as an
SRS shift of 16 on the signed f32 bit pattern (sign-magnitude, rounded by value):

- Normal values, **denormals** and zeros are rounded; denormals are not flushed.
  A rounding carry bumps the exponent, so rounding the largest finite value up
  gives infinity (verified at `0x7F7FFFFF` and the exp-254 neighbours).
- Sign is always preserved: a negative value rounded to zero yields `-0`.
- Infinities pass through.
- **NaN is never rounded**: the high 16 bits are kept in every mode (sign,
  exponent, top 7 payload bits). If those leave a zero payload (NaN payload only
  in the low 16 bits, e.g. `0x7F800001`, `0x7F808000`), payload bit 0 is set so
  the result stays a NaN (`0x7F81` / `0xFF81`). The captures contain 27 distinct
  NaN inputs, all consistent with this.

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| SRS paths other than int32 -> int16, signed | unsigned and signed 8/16/32-bit outputs of 32- and 64-bit accumulators, register `vsrs` and fused `vst.srs`, every `crSat`, shifts 0/1/4/8/15: exact against silicon (dynsign probe suite, see "Run-time sign") | `srs_lane` callers in `vector_srs.rs` | **RESOLVED** 2026-11-04 for those shifts and `crRnd` 0 (the `crRnd` sweep above is int16 only) |
| `crSat` value 2 | saturates exactly like 3 (symmetric for a signed output, `[0, umax]` for an unsigned one) | `SrsConfig::saturate` / `symmetric_saturate` | **RESOLVED** 2026-11-04 (dynsign probe suite, every `d` SRS form) |

## Run-time sign (`d` forms) and UPS saturation

Several vector instructions have an `s` form, signed by encoding, and a `d` form that reads its
sign at run time from the control register its TableGen record lists in `Uses`:

| Register | Forms | Read at (AIE2Schedule.td) |
|----------|-------|---------------------------|
| crUPSSign | `vups.s32.d8/.d16`, `vups.s64.d16/.d32` (w2b, w2c, x2c), `vlda[.2d/.3d].ups.*.d*` | `vups`: cycle 1; `vlda.ups`: cycle 7, with the shift and crSat (silicon) |
| crSRSSign | `vsrs.dN.*`, `vsrsm.dN.*`, `vst[.2d/.3d].srs.dN.*` | cycle 1 |
| crVaddSign | `vabs_gtz`, `vlt`, `vge`, `vmax_lt`, `vmin_ge`, `vmaxdiff_lt`, `vsub_lt`, `vsub_ge` (`.d8/.d16/.d32`), `vextract.d8/.d16/.d32/.d64` | cycle 1 |
| crPackSign / crUnpackSign | `vpack` / `vunpack` `d` forms | see the pack and unpack sections |

**Rule** (`src/interpreter/execute/dynamic_sign.rs`). The decoder labels a `d` lane type unsigned;
the op's sign register comes from its decoded implicit uses, and when its bit 0 is set the unsigned
type fields become signed (the source for UPS, the output for SRS, the lanes for the crVaddSign
ops). Only bit 0 counts: sign words 2 and 0xFFFFFFFE behave as 0, 3 and 0xFFFFFFFF as 1.

**UPS saturation.** Any nonzero `crSat` (1, 2 or 3) clamps the shifted lane to the signed range of
the accumulator lane, never symmetrically (lane `0x8000 << 16` signed with `crSat` 3 gives
`0x80000000`); `crSat` 0 keeps the low bits. The shift is exact up to the 6-bit maximum (a 32-bit
lane shifted by 63 overflows the 64-bit lane and saturates). `vlda.ups` (every addressing mode)
reads crSat in the same bundle as the sign and shift, at issue+6: a write issued 5 bundles after
the load is seen, one issued 6 bundles after is not (the itinerary's `/*crSat*/8` is a cycle late);
Chess and Peano both write crUPSSign and crSat in the load's shadow.

**Control-register writes land at the end of their bundle**: a reader issued in the same bundle sees
the old value (Chess pipelines a record loop with the next record's `movx crSat` beside this
record's `vups`; silicon converts with the old value).

**Evidence.** `tools/gen_dynamic_sign_probe.py` writes a Peano suite (68 cases: every UPS form as
fused `vlda.ups` and as register `vups` including x2c, every SRS form as `vsrs` and fused
`vst.srs`, the eight crVaddSign ops at 8/16/32 bits, `vextract` at all four widths) and a Chess
suite (6 cases, register `vups` w2b/w2c in a software-pipelined loop), sweeping six sign words,
8 shifts per UPS width, 5 per SRS form, all four `crSat` values and edge-rich lanes:
5808 records, HW r1 == r2 and EMU == HW on all of them, plus the 26 chess-full cases that showed
the gap (`lups`, `sups`, `ups_to_*`, `abs`, `gtz`, `abs_gtz` with a run-time sign).

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| `d` forms of UPS, SRS and the crVaddSign ops | all zero-extended / unsigned regardless of the register (only `vextract.d8/.d16` followed crVaddSign) | `dynamic_sign.rs` | **RESOLVED** 2026-11-04 |
| UPS saturation | not modelled (always wrapped) | `ups_lane_signed` | **RESOLVED** 2026-11-04 |
| `vlda.2d/.3d.ups.*.d*`, `vst.2d/.3d.srs.d*`, `vsrsm.d*`, `vlda.ups` idx / pstm / pstm_imm | same code path and `Uses` entry as the probed forms: exact against silicon, data and flag (satflags probe, hand-written assembly) | `dynamic_sign.rs` | **RESOLVED** 2026-11-04 |
| `srUPS_of` / `srSRS_of` | not produced | `status_flags.rs` | **RESOLVED** 2026-11-04, see "Status flags" |
| Exact read cycle of crUPSSign, crSat and the shift for `vlda.ups` | crSat was read a bundle after the sign and shift (itinerary 8 vs 7) | `PendingUpsLoad` | **RESOLVED** 2026-11-04: all three at issue+6 (satflags `upsread` late-write probe, 7 forms x 3 registers x offsets 0..10, HW r1 == r2, EMU == HW) |

## Status flags (`srSRS_of`, `srUPS_of`)

TableGen lists `Defs = [srSRS_of]` on every SRS form (`vsrs`, `vsrsm`, `vst[.2d/.3d].srs`, `s`
and `d`) and `Defs = [srUPS_of]` on every UPS form (`vups` w2b/w2c/x2c, `vlda[.2d/.3d].ups` in
every addressing mode). Silicon (`src/interpreter/execute/status_flags.rs`):

- The flag is raised when **any lane** leaves the output range: the shifted value (SRS: shifted
  and rounded by crRnd) before any clamp. One lane at the start, middle or end is enough.
- **crSat does not matter**: crSat 0 wraps the data and still raises the flag. The range is the
  output lane's under the run-time sign of a `d` form (crSRSSign / crUPSSign bit 0); the
  symmetric SRS modes (crSat 2 and 3) exclude the signed minimum, so a lane landing exactly on it
  raises the flag there and not under crSat 0/1. UPS has no symmetric range.
- **Sticky**: an in-range conversion leaves it set; only a `mov` to the register clears it.
- **Visible** to a `mov rX, srY_of` issued 3 bundles after an SRS (`vsrs`, `vsrsm`, `vst.srs`:
  the itinerary's def cycle agrees), and at the UPS's accumulator result cycle: 3 bundles after a
  `vups`, 9 after a `vlda.ups` (the itinerary says 1 and 8). A `mov` that writes the register one
  bundle before the flag is visible lands after it and wins; one issued a bundle earlier is
  overridden by the flag (`tools/gen_flag_clear_probe.py`: `vsrs`, `vst.srs`, `vups`, `vlda.ups`,
  clears at 0..11 bundles, HW r1 == r2, EMU == HW).

Evidence: `tools/gen_sat_flags_probe.py`, 174 assembly cases over 74 forms (every SRS and UPS
shape, `s` and `d`, every addressing mode of the fused load and store, `vsrsm`, x2c), 5356
records crossing ten operand kinds with every crSat, both signs, two shifts, sticky/overwrite
pairs from either initial flag and a crRnd sweep. HW r1 == r2 174/174; the analyzer's model
(`--analyze`) matches 5356/5356; EMU == HW 174/174 (master: flags differed in 3091 records, data
in none).

TableGen note: the brief for this work flagged `VLDA_UPS_S64_D32_ag_pstm_nrm` as encoding
`mv_ups_base<1,1>` (the signed encoding) while listing crUPSSign. The llvm-aie checked against
(12f5fc51) has `mv_ups_base<1, 0>` there, llvm-mc encodes it differently from its `s32` twin, and
a scripted check finds no `d`/`s` name vs sign-bit mismatch in the file. On silicon the form
honours crUPSSign: its flag follows the run-time sign in the satflags probe, and in the `upsread`
probe flipping crUPSSign before its read changes the converted lanes (top-bit-set source lanes
zero- vs sign-extended).

## bf16 multiply-accumulate datapath (`vmul.f`, `vmac.f`, `vmsc.f`, `vaddmac.f`, ...)

Modeled in `vector_matmul/bf16.rs` (`mac_lane`), derived from NPU1 silicon only
(no model, no vendor source). A lane is the sum of its bf16 products and its
accumulator inputs; the same datapath serves the dense 4x8x4 tile (K = 8) and the
16-lane elementwise form (lane L = A[L]*B[L] + A[L+16]*B[L+16], K = 2).

**Datapath (every item pinned by a silicon capture, see Evidence):**

1. Terms are exact: product `(1.ma)(1.mb) * 2^(ea+eb-254)` (16-bit significand,
   so no product rounding exists), accumulators as stored. Exponent fields 255
   (Inf/NaN) and 0 (denormal, read as field 1) are ordinary numbers. Zero terms
   do not take part in the reference exponent.
2. Reference exponent `E` = largest IEEE exponent of any term (acc1 and acc2
   included). Products use `ea+eb-127`, i.e. a product in [2, 4) has its leading
   bit at `E+1`.
3. Every term is aligned to `E` and rounded **to nearest-even onto a grid of
   weight `2^(E-23)`**. The largest term is exact; a term below half a grid step
   vanishes (this is why `2^30 + 1 - 2^30` gives 0 and why K position is
   irrelevant).
4. The grid integers are summed exactly. Summation order, tree shape and K slot
   have no effect (random wide-spread tiles and permuted multisets all match).
5. The sum is rounded once to fp32 (RNE). A sum with magnitude below 2^-126
   flushes to a **signed** zero **before** rounding (2^-126 - 2^-150, a tie that
   RNE would round up to the smallest normal, reads 0); an exact zero is +0;
   overflow after rounding is Inf. Denormal inputs and accumulators are used as
   is; only results flush.
6. Sign controls: product negated for `negmul`, `msc`, `addmsc`, XOR the config
   word `sub_mul` bit; acc1 negated by `sub_acc1`, acc2 by `sub_acc2`; `zero_acc`
   drops acc1. acc1 and acc2 join the **same** aligned sum (one rounding), they
   are not added afterwards.
7. Specials: any NaN input or Inf*0 -> NaN; Inf terms of one sign -> that Inf;
   Inf terms of both signs -> NaN. A NaN lane is the ordinary numeric result of
   step 5 computed with Inf/NaN exponent fields taken at face value, then
   exponent forced to 255 and mantissa bit 0 set. When that numeric result
   overflows or flushes its mantissa is 0, so nearly all NaNs read
   `0x7f800001` / `0xff800001` (sign = sign of the numeric result; 0*Inf and
   0*NaN with nothing else give +). Only cancelling special terms (e.g. Inf
   against NaN of opposite sign) leave a payload, the normalized difference of
   the two significands (`0xff820001`, `0xff940001`, `0x7ffc0001`, ...). No NaN
   input payload is otherwise propagated.

**Model fit** (silicon lanes reproduced bit-exactly; candidate models scored on
65,536 two-term elementwise lanes with gaps 0..41, random signs and slots, plus
the older K = 8 captures):

| Model | Exact lanes |
|-------|-------------|
| exact sum, one RNE (previous emulator model) | 62,489 (95.35%); 299/384 `vec_mac_bf16`, 772/848 finite `_ovf` |
| fixed-point aligned to the largest leading bit, truncate, RNE, best F=25 | 63,063 (96.23%) |
| aligned to max E, per-term RNE onto `2^(E-22)` / `2^(E-24)` | 87.65% / 94.69% |
| aligned to max E, per-term truncate onto `2^(E-23)` | 88.65% |
| aligned to max E, per-term round-half-away onto `2^(E-23)` | 99.15% |
| **aligned to max E, per-term RNE onto `2^(E-23)`, one RNE (adopted)** | **65,536 (100%)** |

The decisive probe is the decision boundary of the small term: with a large
product in [2, 4) (result ulp 2 grid steps) the up/down decision on the fraction
of the small term was exactly "xi even: up iff fraction >= 0.75; xi odd: up iff
> 0.25", which is RNE onto the half-ulp grid followed by RNE onto the ulp
(double rounding). Large products in [1, 2) show plain correct rounding because
grid and ulp coincide. Cutoff: terms 25 or more exponent steps below the largest
product are dropped.

**Evidence** (captures in `tools/golden/silicon/`, kernels in
`tests/vector-verify/`):

- `vec_mac_bf16_forms` (this work): 23 forms x 336 designed tiles = 123,648
  lanes, bit-exact in the emulator both through `mac_lane` and end to end through
  `execute_matmul`. Sections: gap sweep 0..47 with power-of-two, all-ones, tie and
  sparse small mantissas in both slots and sign relations; permuted multisets;
  accumulators at -34..+33 exponent offsets with tie/all-ones mantissas; edges
  (zero products with huge partner exponent, denormal inputs/accumulators, the
  2^-126 boundary, overflow/carry, signed zeros); sampled Inf/NaN space;
  random K = 8 tiles with exponent spreads 2..60.
- `vec_mac_bf16`, `vec_mac_bf16_ovf`: now exact on all lanes including the NaN/Inf
  tiles that were previously skipped (the earlier gaps are closed).
- Exploration beyond the committed goldens (runtime-input probe, ~700k lanes, all
  reproduced with zero mismatches; data under `build/experiments/r4-d6/`): 52,488
  special-value lanes (9^4 product-pair operands x 8 accumulator operands, four
  elementwise forms), 49,152-lane wild/tiny/huge K = 8 tiles with denormals and
  specials, and the 23-form accumulator sweeps. Discriminating cases: denormal
  accumulator reference exponent -126 (E=-127 or -125 lose 6 and 24 lanes), denormal
  bf16 inputs use field 1 not a normalized exponent (10 lanes), flush before vs
  after rounding (2^-126 boundary tiles), zero products not setting `E`.
- Silicon is deterministic: the forms kernel gave bit-identical output on two
  consecutive HW runs, and every exploration batch run twice agreed on all tiles.

**Timing note.** The float add* forms land with the other float MACs (itinerary def 6, visible from
issue + 7); see "Accumulator read/write timing".

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| `vnegmac.f`, `vsubmac.f`, `vsubmsc.f` | no silicon capture (Chess/Peano emit no bf16 intrinsic for them) | `execute_matmul` bf16 gate | gated "[NOT MODELED]" |
| sparse bf16 4x16x4 | resolved with the other sparse modes (see "Sparse MAC and sparse loads"); only the `vmul` narrow form is reachable from the toolchain | `vector_matmul/sparse.rs` | closed |
| `vaddmac.f` / `vaddmsc.f` latency | dense and sparse both def 6, visible from issue + 7, pinned by the timing probe at controlled distances | `execute_matmul` latency | closed |
| fp32-input MAC (emulated through bf16 splits in aie_api / the Chess `_accuracy_*` routines) | the split chain (`vconv.bf16.fp32` of an fp32 accumulator, `vmsc.f` residuals against 1.0, `vmul.f` cross products, `vadd.f` sums) is the bf16 datapath above applied in sequence; nothing split-specific is needed | `bf16::mac_lane`, `vector_float.rs` | **closed 2026-11-04**: the Chess intrinsic suite's fp32-via-bf16 cases (`*_accuracy_{safe,fast,low}`, real and complex, random bit patterns with Inf/NaN inputs) are EMU == silicon on 55 of 57; the 4 cfloat `mac`/`msc_elem_8` cases that differed were the `vlda am` timing (above), and the remaining 2 (`mul_2x8_8x2_accuracy_{safe,fast}`) read an uninitialized accumulator (next row) |
| Chess `mul_2x8_8x2(v16float, v16cfloat)` (plain and `_accuracy_*`) | the compiled kernel reads `bmh0` before writing it (`vmov x1, bmh0` near entry) and the garbage reaches every output lane through the bf16 split, so the result depends on the previous kernel's leftover accumulator state: silicon gives a different output for the same input after a different predecessor (`cfloat/seven` vs the full suite run); EMU fills registers with `XDNA_EMU_REG_INIT` (default `0xDEADBEEF`) | n/a (undefined input) | **not an emulator gap**; the cases are not a function of their inputs |
| other NaN payload sources | payload from a cancelling special pair derived for products and accumulators; NaN produced by an fp32 accumulator whose mantissa has bits beyond bf16 precision is covered only by the sampled capture | `bf16::mac_lane` | closed for the sampled space |

## fp32 accumulator add/sub/neg (`vadd.f`, `vsub.f`, `vnegadd.f`, `vnegsub.f`, `vneg.f`)

**HW-verified** against `tools/golden/silicon/vec_acc_addsub_f32.json`
(`tests/vector-verify/vec_acc_addsub_f32/`, generated by
`tools/gen_acc_addsub_f32.py`): 20 forms x 704 designed tiles x 16 lanes =
225,280 words, bit-identical across three hardware runs. The two identity forms
return the inputs unchanged (no input-path transformation), and the other 18
forms (the five instructions plus their `zero_acc1` / `sub_acc1` / `sub_acc2`
config variants) are reproduced by the emulator on all 202,752 lanes
(`acc_add_matches_silicon_on_every_form_lane`, `vector_matmul/bf16.rs`).

**Rule.** The accumulator adders are the bf16 MAC lane (`mac_lane`) with no
product terms: the two operands are accumulator terms, each with its effective
negation, and everything else is the MAC datapath unchanged -- align to the
larger exponent with per-term round-to-nearest-even onto the `2^(E-23)` grid,
exact integer sum, one RNE to fp32, a sum below `2^-126` flushes to a signed
zero before rounding, an exact zero is `+0`, overflow gives Inf, denormal
operands enter with exponent field 1, and NaN/Inf follow the MAC rules
(NaN encoded `?f800001`, payload only from cancelling specials).

| Instruction | acc1 sign | acc2 sign |
|-------------|-----------|-----------|
| `vadd.f` | + | + |
| `vsub.f` | + | - |
| `vnegadd.f` | - | - |
| `vnegsub.f` | - | + |
| `vneg.f` | - | (unused) |

The config word negates an operand again by XOR: bit 12 (`dynAcc0Neg`) acc1,
bit 13 (`dynAcc1Neg`) acc2; bit 0 (`dynZeroAccum`) drops acc1 (a `vneg.f`
with it set, or any sum with no terms, gives `+0`). The positions come from
llvm-aie's `VecConf` and the constants Chess emits for the `_conf` intrinsics
(`0x101c`, `0x201c`, `0x301c`, `0x1d`); the fp32 mode bits `0x1c` are
irrelevant to the result.

**Hypothesis tested first, and held:** the D6 lane (`mac_lane(&[], accs, false)`)
with the accumulator terms and their negations reproduced 202,752 / 202,752
lanes with no change.

**Rejected alternatives** (on the 10,725 finite-operand `vadd.f` lanes with no
Inf/NaN):

| Model | Exact lanes |
|-------|-------------|
| plain IEEE-754 fp32 add, round-to-nearest-even | 9,658 (90.0%) |
| IEEE add with denormal operands flushed | 9,492 (88.5%) |
| IEEE add with denormal operands and result flushed | 9,670 (90.2%) |
| **MAC-datapath lane (adopted)** | **10,725 (100%)** |

The deciding lanes are the cancellation and tie cases: the larger operand is
exact and the smaller one is rounded to the grid before the sum, so the result
differs from a single IEEE rounding (double rounding), and a nonzero sum below
`2^-126` is `+-0` rather than the smallest normal or a denormal.

**`vneg.f` is not a sign flip.** It is the adder with one operand: a denormal
becomes `-0` (`0x007fffff` -> `0x80000000`), `+0` stays `+0` (not `-0`), and a
NaN is re-encoded with its sign inverted (`0x7fc00000` -> `0xff800001`).
The integer-style path `execute_acc_negate` used before did flush the denormal
but turned `+0` into `-0`, kept a NaN positive and ignored `sub_acc1`; it is
replaced by the shared lane (`float_vneg_is_not_a_bare_sign_flip`).

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| fp32 add/neg result latency | `vadd.f` reads its accumulators at use cycle 3 and its result is visible from issue + 7 (`II_VACCf` def 6); both pinned by the timing probe | `PendingAccAdd` | closed |
| integer accumulator add/sub config bits | `execute_acc_add_sub` decoded integer `sub_acc1`/`sub_acc2` from bits 11/12 (bit 11 is `dynMulNeg`); TableGen's `VecConf` and the constants Chess emits for `add_conf` (`0x1`, `0x400`, `0x1000`, `0x2000`) put them at 12/13 | `execute_acc_add_sub` | fixed from TableGen; integer variants silicon-confirmed 2026-11-03 (`accops`, [mac-conf](mac-conf.md)); integer `vneg` also honours bit 12 |
| float vector-register add/sub gate | `float_addsub_not_modeled` (x/w float `Add`/`Sub`/`SubLt`/`SubGe`) is unreachable: no TableGen encoding pairs those semantics with a float element type | `vector_arith.rs` | guard only; `float_add_sub_is_reachable_only_through_the_accumulator_family` asserts it |

## Wide VSHIFT / VSHIFT_ALIGN (512-bit x registers)

**HW-verified** against `tools/golden/silicon/vec_vshift_wide.json`
(`tests/vector-verify/vec_vshift_wide/`, `tools/gen_vshift_wide.py`): operands
A = bytes 1..64 and B = bytes 65..128 (every byte distinct and non-zero, so each
output byte names its source and a zero is a fill), 38 chunks x 64 entries = 2,432
(step, shift) combinations, bit-identical across three hardware runs, all
reproduced by `wide_shift_matches_silicon_on_every_entry` (`vector_misc.rs`).

`vshift d, s1, s2, r`: the 128-byte concatenation `s2:s1` (s1 low) is shifted
down by the shift register and the low 64 bytes kept; `out[k] = cat[shift + k]`,
zero past the end. The register is read as a whole unsigned 32-bit value: 64
returns s2, and 128 or any larger value (0x100, 0x10005, 0x80000000, 0xffffffff,
a "negative" amount) returns zeros, so there is no wrap and no masking.

`vshift.align d, s1, s, s2, r` reads the step register `s0..s3` as `s & 7`
(0x21, 0x41 behave as 1; 0x20, 0x100 as 0). Steps 1..4 select a pre-offset `p` of
4, 8, 16, 32 bytes; steps 0 and 5..7 select none and the result is the plain
shift. With `p` and `shift < 64`, bytes below `64 - shift` come from `s1[p + k]`
(zero past its end) independent of `shift`, and `s2` is laid over from byte
`64 - shift` on (it wins any overlap). For `shift >= 64` the pre-offset has no
effect.

**Rejected:** the narrow `vector_align` model (an unsigned 6-bit shift, 256-bit
operands) is wrong for these forms in three ways: it masks the amount to six bits,
it has no pre-offset, and it concatenates 256-bit halves.

**`vmov w, am*` quarter moves.** Chess loads fp32 operands with `vlda amll/amlh`
and reaches a bm register through `vmov wh, amlh` then `vmov bm, x`. The narrow
accumulator-to-w move took one 32-bit lane from each u64 word (and the reverse
zero-extended), which duplicated lanes and failed `vec_acc_addsub_f32` on the
emulator. A quarter (`QuarterLow` / `QuarterHigh`) is now moved raw as four u64
words of two 32-bit lanes, matching the quarter loads and stores
(`test_vmov_w_acc_quarter_is_a_raw_256_bit_move`). The remaining narrow
`vmov w, am` widths (Half) keep the legacy one-lane-per-word behavior,
unverified.

## Integer MAC modes without a matmul tile (elementwise, convolution, two-channel)

**Silicon-derived** (NPU1, 2026-10-29) from `tools/golden/silicon/vec_macmode_*.json`
(`tests/vector-verify/vec_macmode_*/`, generated by `tools/gen_mac_modes.py`;
decoder `tools/analyze_mac_mode_probe.py`; config words `tools/mac_mode_config_words.py`).
Eight modes `aiev2_vmult.h` builds (the intrinsic stems below) had no geometry: the
emulator rejected their config word and dropped the op with its destination
unwritten. Each now has a closed-form lane map, replayed lane for lane through
`execute_matmul` (`vector_matmul/mode_goldens.rs`) and against the table alone
(`aie2::matmul::tests`): 188,416 probe lanes and 301,056 held-out lanes, all four
signedness flavours, every lane bit-exact; the 16 kernels are bit-identical across
three hardware runs.

**Method.** Per mode, a PROBE kernel and a HELD-OUT kernel. The probe uses that the
multiply is bilinear: with A = e_i (one-hot) lane l reads the sum of B[j] over the
partners j of i, so against B = ones it counts the partners and against the ramp
B[j] = j + 1 it names the partner when the count is one (and the mirror image with
B one-hot). Every (A index, B index) pair of every lane falls out, with its
coefficient; the A-side and B-side decodings agree, no cell had more than one
partner, and all coefficients are 1. A first probe design (B bit planes, 896 tiles)
decoded to the same pair sets before it was replaced by the smaller ramp design.
The held-out kernel (seeded random data over the full element range, 25 constant
and 32 per-element extreme tiles of 0, 1, max, min and -1, then uniform) runs
`mul` -> `mac` -> `msc` chained on the previous result, in ss/su/us/uu, and was
never used to derive a map.

**Lane maps** (element indices; `MacShape::operand_indices`). Parameters come from
the intrinsic family name and are cross-checked against the vector types at build time.

| Mode (amode,bmode,variant) | Family | Kind and parameters | Lane `l` sums |
|---|---|---|---|
| (0,1,1) i8 x i8 -> acc32 | `elem_32_2` | `Elementwise { lanes: 32, k: 2 }` | `A[l + 32t] * B[l + 32t]`, t < 2 |
| (0,3,1) i16 x i16 -> acc32 | `elem_32` | `Elementwise { lanes: 32, k: 1 }` | `A[l] * B[l]` |
| (1,3,2) i16 x i16 -> acc64 | `elem_16_2` | `Elementwise { lanes: 16, k: 2 }` | `A[l + 16t] * B[l + 16t]`, t < 2 |
| (2,3,1) bf16 -> accfloat | `elem_16_2` | `Elementwise { lanes: 16, k: 2 }` | same map (folded in, see below) |
| (0,1,2) i8 x i8 -> acc32 | `conv_4x4_8ch` | `Conv { outputs: 4, taps: 4, channels: 8 }` | `A[(p+t)*8 + c] * B[t*8 + c]`, `l = p*8 + c` |
| (0,1,3) i8 x i8 -> acc32 | `conv_8x8_4ch` | `Conv { outputs: 8, taps: 8, channels: 4 }` | `A[(p+t)*4 + c] * B[t*4 + c]`, `l = p*4 + c` |
| (0,1,4) i8 x i8 -> acc32 | `conv_32x8` | `Conv { outputs: 32, taps: 8, channels: 1 }` | `A[l + t] * B[t]`, t < 8 |
| (1,3,3) i16 x i16 -> acc64 | `conv_16x4` | `Conv { outputs: 16, taps: 4, channels: 1 }` | `A[l + t] * B[t]`, t < 4 |
| (0,2,1) i16 x i8 -> acc32 | `4x4_4x4_2ch` | `MultiChannel { rows: 4, inner: 4, cols: 4, channels: 2 }` | `A[(r*4 + k)*2 + ch] * B[(k*4 + c)*2 + ch]`, `l = (r*4 + c)*2 + ch` |

A convolution is a one-dimensional sliding window over channel-interleaved data: A is
the signal (`outputs + taps - 1` positions are read), B the filter taps, and the filter
is per channel. The convolution B operand is read only up to `taps * channels` (conv_16x4
reads B[0..4] of 32; conv_32x8 B[0..8] of 64) and the two-channel mode reads B[0..32] of
its 64 int8 elements; the remaining B elements are never touched. The two-channel mode is
two independent 4x4 matmuls with the channel as the innermost index of A, B and the result.

**Bf16 elementwise folded in.** The hand-coded `lookup_bf16_geometry` variant 1 (16
lanes, `A[L]*B[L] + A[L+16]*B[L+16]`) is the same `Elementwise { lanes: 16, k: 2 }`
kind and now comes from the extracted table; its D6 goldens
(`vec_mac_bf16_forms.json`) still pass bit for bit.

**Signedness and accumulation.** Config bit 9 is X (A) signedness and bit 8 is Y (B),
as for the matmul modes; a signed operand is sign-extended per element, an unsigned
one zero-extended, and the product is exact before summation. All four flavours share
one lane map (the probe relations are identical). Accumulators wrap modulo 2^32 (acc32)
or 2^64 (acc64), no saturation; `mac` adds and `msc` subtracts the product to the
accumulator the previous instruction left, exactly as the matmul modes do. Reading an
acc64 lane back as two int32 halves needs `crSat = none` (`srs` by 0 and by 32).

**Config words Chess emits** (zero_acc = 0, no sub bits; `+ 0x100` sets sgn_y, `+ 0x200`
sgn_x, so `uu` is the base word and `ss` adds `0x300`):

| Mode | `uu` word | decodes to |
|---|---|---|
| elem_32_2 | `0x0028` | (0,1,1) |
| conv_4x4_8ch | `0x0048` | (0,1,2) |
| conv_8x8_4ch | `0x0068` | (0,1,3) |
| conv_32x8 | `0x0088` | (0,1,4) |
| 4x4_4x4_2ch | `0x0030` | (0,2,1) |
| elem_32 | `0x0038` | (0,3,1) |
| elem_16_2 (int) | `0x005a` | (1,3,2) |
| conv_16x4 | `0x007a` | (1,3,3) |

**Rejected alternatives** (exact lanes on the uniform-random held-out tiles, ss, `mul`):

| Mode | Hypothesis | Exact lanes |
|---|---|---|
| elem_32_2 | adjacent pairs `A[2l]B[2l] + A[2l+1]B[2l+1]` | 0 / 1,760 |
| elem_16_2 | adjacent pairs | 0 / 880 |
| conv_4x4_8ch | channel-major planes | 0 / 1,760 |
| conv_4x4_8ch | plain 1-D window, one filter for all lanes | 0 / 1,760 |
| 4x4_4x4_2ch | two stacked matmuls, channel outermost | 0 / 1,760 |
| all eight | adopted maps | 100% |

| Gap | Model vs hardware | Where | Status / rationale |
|-----|-------------------|-------|--------------------|
| config word with no geometry | silicon multiplies nothing for a key no mode uses (every product zero, accumulator inputs still added); the emulator now does the same | `vector_matmul/mod.rs`, `Multiplier::Idle` | closed 2026-11-03, see [mac-conf](mac-conf.md) |
| mode forms beyond mul/mac/msc | every kind (`vnegmul` ... `vsubmsc`) runs the shared post-adder on every mode; the `flags`/`modes` probes pin all kinds on the int8, int16, complex, float-conf and idle modes | `execute_matmul` | closed 2026-11-03, see [mac-conf](mac-conf.md) |
| mode result latency | the modes use the itinerary def cycle of their instruction (the integer MAC family 5, visible from issue + 6), the same rule the timing probe pins for the dense and sparse MACs | `execute_matmul` | closed |

## Unpack: fused `vldb.unpack.*` and standalone `vunpack.*`

**RESOLVED 2026-10-30.** Pinned on NPU1 silicon with four capture kernels from
`tools/gen_vldb_unpack.py` (goldens `tools/golden/silicon/vec_unpack_*.json`, each
bit-identical across three runs and passing against its own capture), replayed lane for lane
by `execute/unpack_goldens.rs`. Both families have four forms, `s8.s4`, `s16.s8`, `d8.d4` and
`d16.d8`; the fused load also has the index, post-modify, 2D and 3D addressing flavours (16
distinct mnemonics plus the shared index/post-modify mnemonic, 20 TableGen records). The
disassembly of the compiled kernels contains every one of them: `vldb.unpack.*` with
`[p, dj]` and `[p], m`, `vldb.2d.unpack.*` and `vldb.3d.unpack.*` with `[p], d0`, and
`vunpack.*` with a w-register source, for all four forms. No TableGen form was unreachable.

**Lane map (all eight instructions, all four forms).** The 256-bit source is `256/bits_in`
packed elements, element 0 in the least-significant bits (low nibble first for the 4-bit
forms). Result lane `i` is element `i`, extended to `bits_out`; the 512-bit result holds
`512/bits_out` lanes, equal in number, so the high half of the result continues the order.
The fused load always consumes 32 bytes and fills the whole X register; the old model read 8
to 16 bytes and wrote 256 bits.

| Form | Source elements | Result lanes |
|---|---|---|
| `s8.s4`, `d8.d4` | 64 nibbles | 64 x 8-bit |
| `s16.s8`, `d16.d8` | 32 bytes | 32 x 16-bit |

**Signedness.** The `s` forms (encoding `sgn = 1`) always sign-extend, with `crUnpackSign`
at 0 or at 1: the register is not read. The `d` forms (`sgn = 0`, `Uses = [crUnpackSign]`)
sign-extend when the register is 1 and zero-extend when it is 0. Only the values 0 and 1 are
observable from C++ (`unpack(v, bool)` writes a bool); bit 0 is what the emulator tests.

**`crUnpackSign` timing.** The register is read late, at pipeline stage 7 as the itineraries
say (operand cycle 7 of II_VUNPACK and II_VLDB_UNPACK*). `vec_unpack_crtiming` drives the sign
from a run-time table inside a software-pipelined loop; Chess schedules the `movx
crUnpackSign` after the `vldb.unpack.d*` and `vunpack.d*` it controls (one bundle after for
the first of an iteration's pair, in the same bundle for the standalone form), and silicon
used the written sign in all 1,536 vectors (6 sign patterns x 64 blocks x 4 vectors). The
emulator samples the sign with the source at issue+6 (`PendingUnpack`), for the fused load as
well as the standalone form. Not pinned: how late the write can be before it misses the read
(only Chess's own schedules, writes up to five bundles after the issue, were observed).

**Fused-load pointer behavior** (`vec_unpack_addr`: 1,888 accesses, 4 forms x 2 sign values):

| Flavour | Load address | Pointer afterwards |
|---|---|---|
| index `[p, dj]` | `p + dj` (bytes) | unchanged |
| post-modify `[p], m` | `p` | `p + m`, the whole modifier in bytes, for `m` in 0, 32..1024, -32..-512; independent of the 32 bytes consumed |
| 2D `[p], d` | `p` | `if dc < dn { p += dj; dc += 1 } else { p += m; dc = 0 }` |
| 3D `[p], d` | `p` | `if dc1 < dn1 { p += dj1; dc1 += 1 } else if dc2 < dn2 { p += dj2; dc1 = 0; dc2 += 1 } else { p += m; dc1 = dc2 = 0 }` |

A descriptor `d<k>` is the register group `m<k>, dn<k>, dj<k>, dc<k>`; the 3D operand `d<k>_3d`
adds `d<k+4>` as the outer dimension (TableGen: `d0 = [m0, dn0, dj0, dc0]`, `d0_3d = [d0,
d4]`). Counts that start at or above their size take the wrap branch (4 sets). The data loaded
and every count matched in all 1,888 records. The rule is in `execute/memory/agu.rs`; every other 2D / 3D family follows it
([addressing.md](addressing.md)).

**Replay counts.** Probe: 8,832 vectors / 423,936 lanes; held-out random set: 6,144 vectors /
294,912 lanes; 1,888 addressing accesses; 1,536 timing vectors; all equal to silicon.

**Rejected alternatives** (exact vectors out of 8,832 on the probe, 6,144 on the held-out set,
all forms and both sign values):

| Hypothesis | Probe | Held-out |
|---|---|---|
| adopted: lane `i` = element `i` | 8,832 | 6,144 |
| the high half of the result restarts at element 0 | 336 | 0 |
| the two halves interleave | 336 | 0 |
| adjacent elements swapped | 240 | 0 |
| reversed order | 240 | 0 |
| `d` forms always unsigned (the old `unpack_widths_from_name`) | 6,759 | 4,608 |
| `d` forms always signed | 6,759 | 4,608 |
| `s` forms follow the register | 6,759 | 4,608 |

(The surviving matches of the wrong maps are blocks whose lanes are zero, uniform or
symmetric, and for the sign rules the vectors the hypothesis happens to agree on.)

**What was wrong in the emulator** (fixed): the fused load built a 256-bit result from 8 to 16
bytes; its 4-bit forms fell through to a raw load; it read signedness from the element types,
and the standalone path from the `D` in the name, so `crUnpackSign` was never consulted; the
`BFloat16 -> Float32` arm was reached by no encoding; and two separate lane maps existed. The
control register write to `crUnpackSign` was discarded. One lane map (`unpack_lanes`) now
serves both paths, per-form widths and the sign source come from the archspec form table
(`aie2::unpack`, extracted from TableGen), and the descriptor update of 2D/3D loads exists for
the first time (before, the descriptor operand was added to the address as an offset).

**Found on the way (fixed, from the EMU side of the `vec_unpack_addr` bridge run).** Three
older emulator bugs that the kernel's compiled code (unrolled 2D/3D walks with pointer
spills) exposed:

- `padda/paddb/padds.2d` and `.3d` (pointer arithmetic with a descriptor) were treated as
  adds of the descriptor operand; they now run the same descriptor update as the 2D/3D loads.
- A scalar read of an `m`, `dn`, `dj` or `dc` register zero-extended its 20 bits; silicon
  sign-extends (`vec_unpack_addr` computes `base + dj` for negative `dj` and gets `base - 32`).
- A store whose data is a pointer register with a post-increment or immediate offset
  (`st p6, [p3], #4`, a spill) took the data pointer for the address and the address pointer
  for the data. The decoder now keeps the address pointer last for stores, and the store and
  post-modify handlers use it.

**Open questions.**

| Gap | Model vs hardware | Status |
|---|---|---|
| unaligned fused loads | silicon rounds the address down to 32 bytes (`vlda.ups`, `vlda.conv`, `vldb.unpack`, 328-case probe); the emulator does the same | resolved: [`addressing.md`](addressing.md), "Unaligned accesses" |
| `crUnpackSign` values other than 0 and 1 | bit 0 decides; higher bits ignored | open; unobservable from `unpack(v, bool)`, needs a raw register write |
| how late a `crUnpackSign` write can be | writes up to five bundles after the issue are honored | open; only Chess's own schedules observed |
| descriptors other than `d0` / `d0_3d` | register mapping `d<k>` = `m<k>, dn<k>, dj<k>, dc<k>` taken from TableGen; `d0`..`d3` and `d0_3d`..`d3_3d` exercised by the `vec_agu_*` captures | resolved for `d0`..`d3`, see [addressing.md](addressing.md) |
| negative or very large counts and sizes | compared unsigned | open; only 0..7 observed |
| cross-tile fused unpack loads | go through the shared vector read, not captured | open |
| AIE2P | `unpackSign0` / `unpackSign1`: the form table names the register and `PendingUnpack` resolves it by id, so no single register is assumed; not implemented | deferred |

## Pack: fused `vst.pack.*` and standalone `vpack.*`

**RESOLVED 2026-10-30.** Pinned on NPU1 silicon with four capture kernels from
`tools/gen_pack.py` (goldens `tools/golden/silicon/vec_pack_*.json`, each bit-identical across
two runs and passing against its own capture), replayed lane for lane by
`execute/pack_goldens.rs`. Both families have four forms, `s8.s16`, `d8.d16`, `s4.s8` and
`d4.d8`; the fused store also has the index, post-modify, 2D and 3D flavours (16 distinct
mnemonics, 28 TableGen records). Before this only `vpack.s8.s16` under `crSat` 0, 1 and 3
had been checked (`vec_pack_i16*`).

**Disassembly proof** (Chess compile of the capture kernels, `gen_pack.py proof`, recorded in
each golden): all four `vpack.*` forms with an x-register source and a w-register destination;
all four `vst.pack.*` forms with `[p, dj]` and `[p], m` (`vec_pack_addr`), `[p], #imm` and `[p,
#imm]` (the lane and timing kernels). The 2D / 3D flavours are not in these kernels; the
`vec_agu_pack` capture covers `vst.2d.pack.*` and `vst.3d.pack.*` for the same forms. No
TableGen form was unreachable. A standalone `vpack` needs a consumer other than a store (the
kernels add a zero vector), or Chess fuses it into a `vst.pack`.

**Lane map (all eight instructions, all four forms).** Result lane `i` is source lane `i`; the
512-bit source holds `512/bits_in` lanes and the 256-bit result all of them, `bits_out` wide,
lane 0 in the least-significant bits (low nibble first for the 4-bit forms). The two 256-bit
halves of the source continue the order. The fused store and the standalone instruction agree
for every form, value and register setting (nothing differs between them in the 48,640
vectors of the lane captures).

**Signedness.** The `s` forms are signed and do not read `crPackSign`. A `d` form
(`Uses = [crSat, crPackSign]`) behaves exactly like its `s` twin when `crPackSign` is 1, and
unsigned when it is 0: the input lane is read as an unsigned number (int16 -1 is 65535) and
clamps to `[0, 2^bits_out - 1]`. The archspec table derives this from `Uses`, not from the
`d` / `s` letter (`SignSource`, shared with unpack).

**`crSat`** (low two bits; `PackMode::for_pack`), where `n` is `bits_out`:

| crSat | signed lanes | unsigned lanes |
|---|---|---|
| 0 | truncate (low `n` bits) | truncate |
| 1 | saturate to `[-2^(n-1), 2^(n-1)-1]` | saturate to `[0, 2^n-1]` |
| 2 | **truncate** | **saturate** to `[0, 2^n-1]` |
| 3 | symmetric: `[-(2^(n-1)-1), 2^(n-1)-1]`, so -128 becomes -127 (int8), -8 becomes -7 (int4) | saturate to `[0, 2^n-1]` |

The crSat 2 row is asymmetric and was a surprise: the signed path truncates, the unsigned path
saturates. crSat 2 is reachable from C++ with a run-time value
(`set_saturation(static_cast<saturation_mode>(2))`; Chess emits `mov crSat, #2`). The
unsigned columns of crSat 1 and 3 are identical (no unsigned symmetric mode).

**Register timing.** The itineraries read `crSat` and `crPackSign` at stage 1 and the source
at stage 2, unlike `vunpack` (stage 7), and nothing in the captures contradicts a read at
issue. `vec_pack_crtiming` drives the sign from a run-time table inside software-pipelined
loops; Chess writes `crPackSign` in the same bundle as the `d` form (`vst.pack.d4.d8 ...;
movx crPackSign, r0` in the fused loop, `vpack.d4.d8 ...; movx crPackSign, r24` in the
standalone loop), and silicon used the value written before that bundle in all 1,536 vectors
(6 sign patterns x 64 blocks x 4 vectors). So a pack sees a control register as of the start of
its own bundle. Not pinned: how early a write must precede a pack to be seen (only Chess's own
schedules were observed).

**Fused-store addressing** (`vec_pack_addr`: 2,048 stores, 4 forms x 8 register passes x 32
blocks x 2 groups): the data lands where the index offset (`[p, dj]` with dj = -64, -32, 0,
+32 bytes) or the post-modify chain (`[p], m` with m = -32, starting at the last slot) says,
and the pointer after four post-modifies is `start + 4m`. The post-modify is applied to the
last store of a chain too (the kernel keeps the pointer live; Chess otherwise turns the last
store into an index store).

**Replay counts.** Probe: 32,256 vectors; held-out random set: 16,384
vectors; 2,048 addressing stores; 1,536 timing
vectors; all equal to silicon. (Each vector is 256 bits of result: 32 or 64 lanes.)

**Rejected alternatives** (exact vectors out of 32,256 on the probe, 16,384 on the held-out
set, all forms, all `crSat` values and both `crPackSign` values):

| Hypothesis | Probe | Held-out |
|---|---|---|
| adopted | 32,256 | 16,384 |
| `d` forms always unsigned, `crPackSign` ignored (the old `pack_widths_from_name`) | 27,070 | 13,312 |
| `d` forms always signed | 27,070 | 13,312 |
| `s` forms follow the register | 27,070 | 13,312 |
| crSat 2 truncates for both signs (the old `from_sat_flags`) | 30,542 | 15,360 |
| crSat 2 saturates for both signs | 27,108 | 13,312 |
| the two halves of the source swapped | 7,618 | 1,386 |
| even and odd lanes interleaved | 3,626 | 1,404 |
| reversed order | 3,390 | 1,386 |

(The surviving matches of the wrong rules are the vectors they happen to agree on: blocks of
zeros, in-range values, and the crSat 0 truncation passes.)

**What was wrong in the emulator** (fixed): `pack_widths_from_name` parsed the mnemonic and
read every `d` form as unsigned, so `crPackSign` was never consulted and a `d` form under
`crPackSign = 1` was wrong; `crSat` 2 was read as truncation for every form, wrong for
unsigned lanes; the standalone and fused paths each carried a copy of the same lane code; and
a second, two-source branch of the standalone handler did a 32-to-16 truncation that no AIE2
pack instruction does. One lane map (`pack_lanes`, behind `pack_register`) now serves both
paths, per-form widths and the sign source come from the archspec form table
(`aie2::pack`, extracted from TableGen, with a build-time check that it covers every pack
record), and the clamp rule is `PackMode::for_pack`. `PackMode::from_sat_flags` is unchanged
(the model-validation harness uses it; SRS and UPS do not go through `PackMode`).

**The legacy two-source branch was dead.** `execute_pack` dispatched to it when
`is_wide_vector` was false. The decoder sets that flag from the TableGen operand types, and
every `VPACK_*` record's source is `mXs`, an x register (512 bit): the recorded Chess
bundles of all four forms decode to wide ops
(`recorded_chess_bundles_decode_to_wide_pack_ops`). The branch and its `vector_pack` helper
are deleted; a non-wide pack now logs an error and does nothing.

**Open questions.**

| Gap | Model vs hardware | Status |
|---|---|---|
| `crPackSign` values other than 0 and 1 | bit 0 decides; higher bits ignored | open; unobservable from `pack(v, int)` as used, needs a raw register write |
| `crSat` values above 3 | only the low two bits are used | open; not captured |
| how early a `crPackSign` / `crSat` write must precede a pack | a write in the same bundle is not seen, an earlier one is | open; only Chess's own schedules observed |
| unsigned crSat 2 saturating | observed, no explanation in the sources we may read | the table is the observation |
| 2D / 3D `vst.pack` data | go through the same lane map; addressing pinned by `vec_agu_pack` | the lane captures use linear addressing only |
| AIE2P | `vpack` there has `packSign0` / `packSign1` and a `crPackSize` register, a different layout; the AIE2 table does not describe it | deferred |

## Sparse MAC and sparse loads (`*_sparse_{narrow,wide}`, `vldb.sparse.*`)

**RESOLVED 2026-10-31 for NPU1 (AIE2).** The sparse MAC is pinned with 26 capture kernels from
`tools/gen_mac_sparse.py` (`vec_macsparse_<mode>_{probe,ops,rand,conf,alt}`, plus
`vec_macsparse_4x16_16x4_spec` for the float mode; the held-out `rand` and `spec` tiles never
entered the derivation) and replayed lane for lane by
`vector_matmul/sparse_goldens.rs`. The sparse loads are pinned with six capture kernels from
`tools/gen_vldb_sparse.py` (`vec_sparseload_{proto,ring}_w{4,8,16}`, bit-identical across two
runs once the register halves a `*_lo` step leaves undefined are zeroed), replayed record for
record through `MemoryUnit::execute` by `execute/sparse_load_goldens.rs`. The AIE2P layout is
not assumed to match: the facts are in `xdna_archspec::aie2`.

**Disassembly proof.** The Chess (and, for the int4 mode, Peano) core ELFs contain every sparse
form that can be written: all five modes, narrow and wide, for every MAC-family op, and all
twelve `vldb.sparse.{reset,fill,peek,pop}.{4,8,16}` mnemonics (recorded in each golden's
`encodings`). Unreachable: the bf16 narrow forms other than `vmul` (the toolchain headers emit
them malformed), so bf16 sparse is captured for the wide forms and for narrow `vmul`. Chess
exposes no int4 sparse multiply; that mode was captured through Peano.

### Sparse MAC rule

The sparse operand is a `qx` register: 512 stored-data bits (`x_n`, the 2:4-compressed B) and a
128-bit mask (`q_n`). Logical B is `inner x cols`; the mask names which logical units are
present.

- The unit is one element, except for int4 where it is a byte holding two neighbouring columns
  (low nibble first). A unit owns `128 / (inner * unit_cols)` mask bits: one for the 8-bit modes,
  two for the 16-bit and bf16 modes (selected if either bit is set).
- A group is four consecutive `k` in one unit column. With three or four units selected the
  group is zero. With one selected, its stored slot is 0 if it is at position 0 or 1 and 1 if at
  2 or 3; with two selected, slots 0 and 1 in ascending order. Stored rows are `2 * group + slot`
  (stored data is `inner / 2` rows by `cols`, row-major).
- A multiply then is the dense multiply of A against the expanded B. Narrow encodings read A
  from an x register, wide from a y register; for a mode whose A is wider than an x register the
  narrow form sees zero rows beyond it, and the wide form of a narrow mode ignores the upper
  half of the y register.
- Op combine, all five modes and all ten ops (`conf` captures): `+-P +-acc1 +-acc2`, with the
  config word's `sub_mul`, `sub_acc1`, `sub_acc2` flipping signs, `shift16` shifting acc1 left
  by 16 and `zero_acc1` dropping it.

Both signedness flavours of the integer modes follow the same rule. bf16 follows the same mask
rule (two bits per element); its float combine is the bf16 datapath's own.

**Absent is not zero (float mode).** A B element the mask drops takes no part in the multiplier:
its product is not a term at all, so an Inf or NaN in A that meets only dropped elements does
not make the lane NaN, and no Inf*0 arises from them (this includes the elements of a group with
three or four selected units). A kept element that holds a stored zero is an ordinary zero and
still makes Inf*0 a NaN. Found when the ISA suite's random bf16 inputs (which include Inf and
NaN patterns) disagreed with an emulator that expanded dropped elements to zero: six test points,
every one a NaN lane. `vec_macsparse_4x16_16x4_spec` (112 tiles, zeros, Inf, NaNs of several
payloads, denormals and overflowing exponents at three densities, random and structured masks,
mul -> mac -> msc, 5 376 words) is bit-exact with the rule; two runs on silicon were
bit-identical. The integer modes cannot observe the difference (a dropped element is a zero
product either way).

### Sparse load model

State: a ring of ten 32-byte lines the loads write, a read position `DP` (an absolute
address, scalar slot 36), the pointer `p`, and the sticky flags `srSparse_of` and `srCompr_uf`
(scalar slots 38 and 39; a status-register move reads or clears them).

- `reset [p]`: the stream starts here. `DP = p`, the line at `p` is written at the ring's
  current write position (the ring is not cleared and keeps its stale content), `p += 32`.
- `fill [p]`: writes the line at `p` into the ring, `p += 32`.
- `pop [p]` / `peek [p]`: read an 8-byte mask at the read position `DP`, then the data bytes
  that follow it, expanded into half a register (below). The line at `p` is written after the
  read, unless the ring already holds nine lines beyond the read position (a pop that would
  run the ring ahead of itself drops its fetch). `pop` adds `8 + ceil4(popcount(mask)) mod 64`
  to `DP` and advances `p` by 32; `peek` leaves `DP`; both leave `p` alone when their
  `_and_get_pointer` variant is not used (the variant is a peek that replaces the pointer copy
  with `DP`, which is how the captures read `DP`).
- Expansion, widths 4 and 8 (identical: the stream is byte-granular): 16 groups of four mask
  bits. A group with one set bit takes one stream byte into slot 0 (bit 0 or 1) or slot 1
  (bit 2 or 3), with two set bits two bytes into slots 0 and 1, with none, three or four set
  bits is zero. Group `g` owns output bytes `2g` and `2g + 1`. Bytes are consumed in mask order,
  only by groups that are placed (a three- or four-bit group still counts in the `DP` advance).
  Width 16: the unit is a 16-bit element with two mask bits (bit `2e` selects its low byte,
  `2e + 1` its high byte); eight groups of four elements; slot `s` of group `j` sits at output
  bytes `4j + 2s`, `4j + 2s + 1`; one stream byte per set mask bit.
- The destination is `qwl_n` or `qwh_n`: vector register `2n` (or `2n + 1`) and the low (or
  high) 64 bits of `q_n`. Latency 7 to the destination, 6 to `DP`, 6 to `srSparse_of`, 5 to
  `srCompr_uf` (the itinerary).
- `srSparse_of` is set by a `pop`/`peek` whose mask has a group with more than two selected
  units. `srCompr_uf` is set when `-32 < rem <= 64` with `rem = 32 * lines written since reset
  - (DP - DP at reset)`. `clr_sparse_of` / `clr_compr_uf` clear them; `reset` does not.
- A read past what was written returns what the ring still holds, which the model reproduces
  exactly (the `ring` captures read up to a full turn past the written lines).

**What `DP` is, as far as outputs pin it.** An absolute address into the stream, set by `reset`,
read through a pointer copy, advanced only by `pop` by the amount above, independent of the
width. Together with the ring write position and the line count since `reset` it explains all
735 `dp` reads and 3 500 pop/peek data reads of each width (10 500 in all).

**Unobservable.** Whether the ring is really ten lines in the datapath or an equivalent
structure (the fetch backpressure at nine lines ahead and the stale reads fix its behaviour,
not its layout); the hardware's behaviour for streams that leave the kernel's buffer (never
captured, for safety); the registers a `*_lo` step leaves undefined (the high halves of the
result keep whatever the register held, so goldens zero them).

**Rejected alternatives** (each failed a capture): a 32-byte ring; a ring cleared at `reset`;
three-selected groups consuming bytes; a consume pointer that is `p` itself rather than a hidden
`DP`; fetching before the read.

**Open.** One mismatch is not understood: width 4 and 8, stream `dense40`, script `unprimed`,
the first `pop_lo` after a `reset` on a ring whose second line an earlier script wrote returns
the tail of its data from a different slot than the model's (one record in each of the two
captures; everything after it matches). `KNOWN_RESIDUAL` in `tools/analyze_vldb_sparse.py` and
`sparse_load_goldens.rs` keep it exact so a new difference cannot hide in it.
`vldb.compr.*` (the compress side, out of scope) shares `DP`; nothing here was learned about it
beyond that the flag named `srCompr_uf` is set by the sparse read window above.

### Sparse kernels end to end (bridge, EMU against the silicon goldens)

The unit replays feed each instruction the silicon operands. The staged kernels
(`vec_macsparse_*`, `vec_sparseload_*`) run the Chess/Peano-compiled programs through
`test.exe -> XRT -> plugin -> emulator` and compare every output word with the golden, so they
also test the schedule the compiler built around the instructions. Result (EMU, 2026-10-31):
every `probe`, `rand`, `ops`, `spec` and `conf` kernel passes, Chess and Peano, including the
64-bit `2x16_16x8` and `2x8_8x8` conf kernels and the Peano-built 4x32_32x8; the Chess `alt` builds
do not compile (the Peano `alt` builds are the ones that run); `vec_sparseload_ring_w{4,8,16}` and
`proto_w16` pass. Not passing, with what is known:

| Kernel | Words | Cause |
|---|---|---|
| `vec_sparseload_proto_w4`, `_w8` | 8 | exactly the `dense40`/`unprimed` residual above, nothing else |
| `vec_macsparse_4x16_16x4_alt` (Peano) | 190 | the first quarter (`vst amhl1`) of the second `vmul.f` result of tiles 1 and later: silicon has zeros in the even lanes there, the emulator non-zero values, so the mask or the A rows differ, not the accumulator timing (the unit replay of the same lanes is exact). Delaying the `vmov q, w` write by its itinerary latency was tried and made other lanes wrong, so the q-register write timing is not the cause. Not diagnosed further: the suspects are the pointer update right before the hardware-loop entry (`padda [p5]` one bundle before the loop's first `vlda wl2, [p5]`) and the loop-carried `vmov q1` / `vmul qx1` pair; a probe of pointer and q-register timing like the accumulator one would settle it |

Defects the end-to-end runs exposed and this work fixed (each outside the sparse rule):

- `vmov cm_dst, cm_src` and `vmov bm_dst, bm_src` wrote at issue. They now land by the
  accumulator timing model (II_VMOV_W_CM_CM, def 2; Chess stores the destination's quarters in the
  bundles around the move: `vst amhh7; vmov cm7, cm1; vst amhl7` stores cm7 before the move).
- `vmov amll3, wl7` (a w register into an accumulator quarter) wrote at issue, so the `vsrs` one
  bundle behind it read the new lanes; the Chess schedule of the 64-bit conf kernels reads the old
  register there. The quarter write is queued with a lane mask and merged when it lands.
- A `vmov` from an odd accumulator register (a `bmh`) is a bm half; the decoder carries that width
  for every `vmov bm/cm` form (`test_vmov_acc_to_acc_carries_the_accumulator_width`).
- The sparse MAC special cases of the first end-to-end pass (sparse MACs reading their accumulators
  at issue, sparse `vaddmac.f` landing at 6 against the dense 7) are gone: one rule covers dense and
  sparse (see "Accumulator read/write timing").

The `vec_sparseload_*` kernels now store zeros for the register halves a `*_lo` step leaves
undefined (`ZERO_UNDEFINED_HALF`), so a capture needs no normalization: the regenerated kernels
reproduce the previous (normalized) goldens on silicon word for word.

**Open.** (1) `vec_macsparse_4x16_16x4_alt` (Peano) and the `dense40`/`unprimed` read above.
(2) The write-write order of two in-flight writes to one accumulator register is issue order, not
landing order; the probe measured read-after-write and write-after-read only. (3) A kernel that
pops from the load unit and feeds the MAC (`vldb.sparse.pop` then `vmul` on the same `qx`) was not
captured; the two halves are verified separately and each against compiled schedules.
