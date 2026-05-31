# Findings: BUG-B — AIE2 zero-overhead-loop store flush (investigation state)

**Date:** 2026-05-31
**Follow-on to:** `2026-05-30-buga-fix-and-retriage.md` (which flagged BUG-B)
**Status:** Root cause understood; faithful emulation **not yet achieved**. Best
model reaches 1574/1577 but regresses 2 seeds. The exact flush condition is
**cycle-exact** and needs the aiesimulator oracle (Option 1). This document is
the handoff so that work can resume after a context compaction.

---

## Headline

BUG-B is the differential fuzzer's "NPU zeros every 4th sub-word element"
divergence. It is **a real hardware effect of a Peano codegen quirk**, not an
emulator compute bug and not a fuzzer-harness artifact:

- The fuzz kernel is well-formed. **Chess compiles it correctly** and the
  Chess binary produces fully-correct output on silicon (the Chess experiment,
  `build/experiments/buga-chess/`).
- **Peano's simple-unroll path parks a partial-word store (`st.s8`/`st.s16`) in
  the loop-end (`LE`) bundle.** On a zero-overhead-loop **back-edge**, that store
  is **flushed by hardware** (committed only on the final fall-through), so every
  4th element keeps its initial zero. The last element survives (final iteration
  falls through, no back-edge).
- The emulator must **reproduce this effect** (its purpose is to be an
  open-source aiesimulator — faithful to what silicon does with a given binary,
  including the effects of badly-scheduled code), **without** breaking
  correctly-scheduled loops.

The hard part: the flush is **cycle-exact**. Two structurally identical loops
diverge on silicon (one flushes, one commits), so no static "store is at LE"
rule suffices.

---

## Confirmed hardware facts (grounded, keep these)

1. **Partial-word stores are read-modify-write and commit late at stage E11
   (issue+11).** `AIE2Schedule.td:133-141`: "ISA says load is in E5, store is in
   E11, so we need to be 7 cycles apart"; `MemInstrItinData<II_STHB, ... [7,1,1],
   MemoryCycles<[5,11]>>` (`AIE2Schedule.td:733-737`). AM020:3921 "8-bit and
   16-bit stores are implemented as read-modify-write instructions";
   AM020:3911 "Load and store units manage the 5-cycle latency of data memory."
2. **i32 / full-word stores are immediate** (one store port, not RMW). They are
   NOT in the emulator's `pending_stores` queue and are unaffected by any flush.
3. **The loop is run by a Program Control Unit (PCU)** with its own **fetch
   counter `fc`** (distinct from `pc`) and a **shadow loop count `lci`**
   (AM020 register table ~line 4059-4097). The PCU fetches ahead and redirects
   fetch from `LE` to `LS`. The back-edge is a **fetch redirect**, which is the
   physical origin of the last-bundle behavior.
4. **llvm-aie has NO scheduler constraint keeping stores away from `LE`, and is
   silent on back-edge flush** (`AIEBaseSubtarget.cpp:304-320` only enforces a
   7-bundle setup-to-`LE` distance for the ls/le/lc register writes to settle;
   `AIEBaseInstrInfo.cpp:1471`). So the flush is undocumented emergent pipeline
   behavior — it cannot be derived from the toolchain, only observed.
5. **DMA descriptors are innocent.** `examples/decode_cdo_bds.rs` decoded the CDO
   BD registers independently of the emulator's parser: every hop is plain
   contiguous (i8 memtile BD len=32 words, i32=128 words, all stride/wrap/zero
   fields = 0). The drop is NOT in the data plane; it is the core store.
6. **Emulator modeling gap:** the EMU uses partial-word store data-read latency
   **6** (`PARTIAL_WORD_STORE_DATA_LATENCY`, tuned for add_21_i8/add_12_i8 value
   forwarding), but the actual **memory commit is E11**. The data-read (value
   sampling) and the memory-commit are two distinct events the EMU conflates.

---

## The models tried, and why each failed (do not repeat these)

The validation method throughout: every fuzz seed that ever mismatched has its
**true HW output saved as `build/fuzz/seed_N/npu_output.bin`** (written by the
fuzzer on mismatch; HW output is deterministic). `examples/validate_seeds.rs`
replays every such seed through the in-process emulator and diffs byte-for-byte
against the saved HW — a **free, no-HW regression gate over ~1577 seeds**.
`examples/check_le_squash.rs` does one seed with a readable dump.

Baseline (no fix): the 53 BUG-B seeds mismatch (EMU correct-per-C, HW drops
every 4th). Everything else matches.

- **Model 1 — squash `issue_pc == LE` for BOTH stores and register writes.**
  Result: **1525 → 1252 pass, 53 → 326 mismatch.** Catastrophic. It deleted
  loop-carried **register writes / loads** that live in the `LE` bundle of
  *pipelined* loops (e.g. seed_13's `LE` bundle is `lda.s16 r5; add r9...; mov
  r1...`), which hardware commits every iteration. Lesson: **register writes
  (loads, pointer/index updates) in the `LE` bundle are NOT flushed by HW.**

- **Model 2 — flush ALL pending stores at the back-edge (`pending_stores.clear()`).**
  Reproduced seed_18/seed_1218 but **over-flushed pipelined body stores**
  (seed_13 diffs at idx 9, 13). In a software-pipelined loop, `st.s16` stores
  issued in the body are legitimately in-flight across the back-edge and DO
  commit. Lesson: **earlier body stores survive the back-edge; only the store in
  the `LE` bundle itself is at risk.**

- **Model 3 (current best) — flush pending stores with `issue_pc == LE` only;
  leave register writes alone.** Result: **1574/1577.** Fixed BUG-B AND all 273
  pipelined regressions from Model 1. Implemented as: add `issue_pc: u32` to
  `PendingStore`, set it to `self.pc` in `queue_pending_store`, and in
  `check_hardware_loop` on the back-edge branch (`new_lc > 0`):
  `self.pending_stores.retain(|ps| ps.issue_pc != le);`. Four TDD tests
  (`zol_backedge_flushes_pending_store`, `zol_backedge_preserves_earlier_body_store`,
  `zol_final_iteration_keeps_pending_store`, `zol_backedge_preserves_pending_register_write`).
  **Remaining failure: it over-flushes 2 seeds (1826, 1781).**

---

## The wall: seed_18 vs seed_1826

Both are **simple-unroll i8 loops** (no `chess_prepare_for_pipelining`) with a
`st.s8` in the `LE` bundle, store data produced in the immediately-preceding
bundle. Yet on silicon:

- **seed_18**: HW **flushes** the `LE` `st.s8` → idx 3 = 0. (Model 3 matches.)
- **seed_1826**: HW **commits** the `LE` `st.s8` → idx 3 = 48. (Model 3 wrongly
  flushes → EMU 0 ≠ HW 48.)

No static rule distinguishes them. The only visible difference is the **latency
of the instruction producing the store's data**:
- seed_18 LE-store data = `mul r2, r1, r1` (integer multiply, **2-cycle**
  latency per AM020).
- seed_1826 LE-store data = `lshl r8, r5, r9` (shift, **1-cycle** latency).

**Unproven hypothesis worth testing first with the oracle:** the store is
flushed iff its **data operand is not yet available when the back-edge fires**
(i.e. the producing op's result hasn't landed), not merely because the store
sits at `LE`. mul (2cyc) misses the window; lshl (1cyc) makes it. A load-fed
store (7cyc) would also miss. This would also explain why register-fed loads in
the `LE` bundle survive (they feed the register file, not a memory commit).
**Do not implement this on faith — confirm against aiesim.**

---

## Residual mismatches under Model 3 (the 3 of 1577)

| seed | dtype/size | nature | disposition |
|------|-----------|--------|-------------|
| 1826 | i8 / 256  | **over-flush** — HW commits LE `st.s8` (lshl data), EMU drops it | the discriminator for the cycle-exact rule |
| 1781 | i16 / 64  | **over-flush** — same class | same |
| 1964 | i8 / 256  | **separate compute divergence** (value diff −64 vs −16, both nonzero; `<<`/`*` semantics) | deferred; NOT ZOL. Also note seed_17 (i32) is a similar pre-existing compute divergence, untouched by the store flush. |

---

## Key seed matrix (for any future model)

| seed | dtype/size | role |
|------|-----------|------|
| 18   | i8 / 128  | BUG-B, simple unroll, HW **flushes** LE store. Model 3 ✓ |
| 1218 | i16 / 256 | BUG-B. Model 3 ✓ |
| 13   | i16 / 128 | pipelined; LE bundle is a **load**; HW commits all. Model 1 broke it, Model 3 ✓ |
| 1826 | i8 / 256  | simple unroll, LE `st.s8`, HW **commits** (lshl data). **Model 3 over-flushes.** |
| 1781 | i16 / 64  | over-flush, same class |
| 1964 | i8 / 256  | separate compute divergence (deferred) |
| 3    | i32 / 128 | passing reference; nop at LE, no store there |
| 17   | i32 / 128 | pre-existing compute divergence (full-word, not ZOL) |

---

## Tools built (kept, committed)

- `examples/decode_cdo_bds.rs` — decode a raw CDO's DMA BD registers (len, d0/d1/d2
  stride+wrap, zero-pad) independently of the emulator's BD parser. Usage:
  `cargo run --example decode_cdo_bds -- build/fuzz/seed_18/aie.mlir.prj/main_aie_cdo_init.bin`
- `examples/check_le_squash.rs` — run one seed through the in-process EMU and diff
  against saved `npu_output.bin`. Usage:
  `cargo run --example check_le_squash -- build/fuzz/seed_18 128 i8`
- `examples/validate_seeds.rs` — batch: replay every seed with a saved HW output
  and report EMU==HW / EMU!=HW. Usage: `cargo run --release --example validate_seeds`

Disassembly oracle for AIE2 ELFs:
`/home/triple/npu-work/llvm-aie/install/bin/llvm-objdump -d --triple=aie2 <elf>`

---

## Next step: Option 1 — aiesimulator oracle

Goal: confirm AMD's cycle-accurate sim reproduces both seed_18 (flush) and
seed_1826 (commit), then read the cycle trace to derive the **exact** flush
condition (test the data-availability hypothesis above first).

- aiesimulator: `amd-unified-software/aietools/bin/aiesimulator` (cycle-accurate
  `aie2simmsm`; functional `aie2simmsm_func`).
- **There is prior aiesim work in this repo:** `build/experiments/2026-05-13-chess-aiesim/`
  and `2026-05-13-chess-O0/` — check these for a working single-core/multi-tile
  aiesim invocation to copy rather than build the flow from scratch.
- The Chess build flow used for the experiment is in `build/experiments/buga-chess/`
  (source the env: `source /home/triple/npu-work/toolchain-build/activate-npu-env.sh`,
  then `xchesscc_wrapper aie2 ...` + `aiecc.py --xchesscc --xbridge ... --aiesim`).
- aietools is a **read-only reference** (never copy code/data); the real NPU is
  ground truth, aiesim is a debugging aid — but here aiesim's value is the
  cycle-by-cycle pipeline visibility the silicon can't give.

Then refine the flush condition in `context.rs` `check_hardware_loop` and re-run
`validate_seeds` (free) until 1576/1577 (everything but the deferred seed_1964),
then the HW fuzzer (`./target/release/xdna-emu fuzz --seed 1 --iterations 2000
--hw`, ~28 min) as the final guarantee, then commit, then file the Peano issue.

---

## Code state at handoff

- `src/interpreter/state/context.rs`: **Model 3 committed as WIP** (issue_pc field
  + `retain(issue_pc != le)` on the back-edge + 4 TDD tests). Full lib suite
  passes (3249/0). It reproduces BUG-B and fixes the 273 regressions but
  over-flushes seed_1826/1781 — the aiesim work refines the flush condition from
  here (or replaces it if the data-availability model proves correct).
- BUG-B is a confirmed **Peano codegen bug** to be reported upstream (issue-first)
  once the emulator side is settled.
