# SP-5b -> R3b NEXT-STEPS (resume artifact)

**Purpose.** Resume pointer for a fresh session (intended: a Fable session) picking
up SP-5b after Phase 1 (R1) merged. Two threads live here: (1) a mechanical
**blocker** on the R1 silicon gate, and (2) the next **build**, R3b Phase 2. It
also flags where a super-genius thinker earns its keep before SP-5c.

This doc is a *pointer*, not a spec. The governing artifact is the design doc:
`docs/superpowers/specs/2026-06-30-sp5b-kernel-hw-bringup-design.md` (rev2) --
Sec.5 = R3b-PC, Sec.6 = R3b-TM go/no-go. Read it, not a paraphrase of it here.
Parent: `docs/superpowers/specs/2026-06-30-sp5b-measurement-apparatus-design.md`.

---

## A. Where we are (SP-5 state, 2026-07-01)

| Piece | State |
|---|---|
| SP-5a -- calibration enablement | DONE + merged (`d850a88f`) |
| SP-5b software-core -- override seam + R1/R3b extractors + emu plumbing | DONE + merged (`af8b1208`) |
| **SP-5b Phase 1 (R1 instrument)** | **Merged to master `e2b76986` with an explicit deferred-gate caveat (below)** |
| SP-5b Phase 2 (R3b-PC, primary d_h/d_v instrument) | **NOT started -- the next build (Sec.C)** |
| SP-5b Phase 3 (R3b-TM fallback) | NOT started; gated go/no-go after Phase 2 (design Sec.6) |
| SP-5c -- Phoenix measurement campaign, flip `calibrated` | NOT started; HW-gated, one-way. The genius audit (Sec.D) gates entry here. |

R1 (Phase 1) is the *lowest*-risk and most-narrowed instrument (within-column
`d_v` + intra only; ~30cy cross-column jitter defeats its `d_h`). **R3b is
primary for the money quantities `d_h/d_v` and does not exist yet** -- Phase 1
being done is a partial summit.

---

## B. BLOCKER -- the R1 silicon gate page fault (mechanical, must close before SP-5c trusts R1)

The Phase-1 branch merged with the silicon gate **deferred-red, not passed**
(the merge commit `e2b76986` records this in full). The first Phoenix gate run
(2026-07-01, `bash build/experiments/sp5-skew/r1_gate.sh`) proved the instrument
**runs** (20/20 rc-0, valid traces, col-1 placement fix validated on silicon,
decode+normalize clean every run) but FAILED loud on `IO_PAGE_FAULT`s from our
BDF (`amdxdna 0000:c6:00.1`), intermittent across 13/20 runs.

- **Leading hypothesis: trace-buffer overflow.** Gate `TRACE_SIZE=16384` (16KB,
  cloned from SP-3's ~1250-event spike); this kernel emits ~2100 events.
  `trace.bin` is exactly 16384 on every run (always fully filled). "Clean" runs
  may have *silently corrupted* adjacent memory rather than faulting -- so R1 is
  proven to RUN on silicon but NOT proven to produce trustworthy data.
- **Resume plan** (full detail: `memory/project_sp5b_r1_gate_iommu_fault_inflight.md`):
  (1) compute bytes-per-event x ~2200 vs 16384; (2) one cheap HW run with a
  larger `--trace-size` (e.g. 65536) -- if faults vanish, confirmed; (3) fix =
  size the R1 kernel's trace BO to real event volume + headroom (gate
  `TRACE_SIZE` **and** kernel trace-BO alloc must agree); (4) re-run gate, expect
  all-clean + range-0 tally.
- **This is craft, not a Fable job.** Sonnet/Opus implementer work. But it is a
  hard precondition: **SP-5c must not trust R1 silicon data until this gate
  passes green.**

---

## C. NEXT BUILD -- R3b Phase 2 (R3b-PC), design doc Sec.5

Primary instrument for `d_h/d_v`. Reads a **local single-clock perf-counter
interval** on each measured tile, immune to the cross-column trace jitter that
forced R1 within-column -- that immunity is *why* R3b, not R1, is primary for
`d_h`.

**Already built, consumed UNCHANGED (do not rebuild):**
- `tools/calibration/skew/r3b_extract.py` -- `extract_r3b({dn_h, dn_v, r}) -> {d_h, d_v, fit_residual}`.
- `tools/calibration/skew/_solve.py`, `schema.py`. The runtime-override seam
  (`src/device/state/effects.rs`). **All 21 skew tests green as of this doc.**
- R3b needs **no emulator work** (design Sec.5.3): the emu already handles two
  independent floods; a distinct-channel `s2` does not trip the ch15
  single-source guard (`effects.rs:633`); rank-2 solve synthetic-tested 5/5.

**Net-new to build (design Sec.5.1/5.2/5.4):**
1. `mlir-aie/test/npu-xrt/sp5_skew_r3b_pc/` -- **hand-authored MLIR** (Shape A;
   IRON hides the register writes). Two hand-authored floods `s1`/`s2` from
   different corners (`Event_Broadcast{N}` + `Event_Generate`, distinct channels/
   generate events, **no `Timer_Control.Reset_Event` on measured tiles** -- the
   perf counter is a separate HW unit from the timer). Perf-counter per measured
   tile (`Performance_Control0`, start=s1 event, stop=s2 event). Geometry spans
   **both axes** (>=3 collinear horizontally for `d_h`, >=3 vertically for `d_v`).
   Runtime-seq ordering is load-bearing: **configure counters on ALL measured
   tiles -> generate(s1) -> generate(s2) -> readback** (config MUST precede s1).
2. **The critical-path task: counter READBACK.** `write32` is write-only. Read
   `Performance_Counter0` via a **control-packet register-read** post-run + a
   **net-new readback host** that binds/dumps the readback BO. Fallback:
   post-run core-`LDA 0x31520` -> store -> DMA out (heavier, core-program shape).
   Budget this as the highest-effort task -- the hard part is *readout*, not config.
3. `tools/calibration/skew/r3b_observe.py` -- reads the readback buffer, maps
   tiles to `{dn_h, dn_v}` via a kernel-dir `geometry.json`, emits the dicts.
   Shared by both R3b kernels. Frozen-fixture unit test.
4. `build/experiments/sp5-skew/r3b_pc_gate.sh` -- N serial runs: rc-0, no TDR,
   rank-sufficient both axes, **`s1`-before-`s2` non-inversion check**, **counter
   `b`-vector range-0 across runs**. **No value assertions** (SP-5b produces no
   number; numbers are SP-5c).

**Verified-anchor drift corrections (fold into the kernel work -- the spec text is
stale on these three, everything else matches exactly):**
- **`npu1_4col` does NOT exist.** Named targets are `npu1_1col/2col/3col`
  (`mlir-aie/include/aie/Dialect/AIE/IR/AIEAttrs.td:118-120`); the 4-column model
  is the bare `npu1`. R3b's >=3-column partition -> target **`npu1_3col`** (design
  Sec.3/5.1 label is wrong).
- **`control_packet` read = generic `$opcode` I32 attr, not a named READ enum**
  (`AIEX.td:944`); the lowering pass (`AIECtrlPacketToDma.cpp`) lowers generically
  and does not branch on read/write in-source. Don't hunt for a READ enum.
- **TM intrinsic (Phase 3) path/line:** `__builtin_aiev2_read_tm` lives at
  `llvm-aie/llvm/include/llvm/IR/IntrinsicsAIE2.td` (builtin def `:934`; class
  `AIEV2ReadTM` `:469`), NOT under `llvm/lib/Target/AIE/`. Content correct.

**Verified-and-correct anchors** (no action, just confidence): `Performance_Control0`
@ `0x31500` with `Cnt0_Start_Event` 6:0 / `Cnt0_Stop_Event` 14:8; `Performance_Counter0`
@ `0x31520`; `Timer_Control` @ `0x34000`; `BROADCAST_15`=122 (core+mem),
`USER_EVENT_2`=126 (core/mem), shim `USER_EVENT_0`=126/`USER_EVENT_1`=127 (no shim
USER_EVENT_2); flood template `AIEInsertTraceFlows.cpp:672-723`; SP-3 seed
`spike_bringup/of_q0_lean.py` (2 cores, rows 2,3).

**Build order + model tiering (matches the arc's discipline):** the R3b kernel +
readback host + observe bridge + gate are **implementer craft** (Sonnet under a
plan). The judgment-dense parts -- the readback-mechanism design decision, and
the soundness audit (Sec.D) -- are the Opus/Fable work.

---

## D. Where FABLE earns its keep -- the pre-SP-5c soundness audit (no HW, run against merged specs)

The persistent risk across this whole arc is **laundering modeled data as
measured**, and SP-5c is the *irreversible, Phoenix-gated* step where a subtly
circular or under-identified inference bites -- we would not find out until
`calibrated` is flipped and we are "validating" against HW with a broken oracle,
after burning part of a finite Phoenix window. A super-genius should catch that
**on paper, where it is cheap.** Highest-leverage use of a Fable session:

1. **Model-structure identifiability (top priority).** The model *structure* is
   unvalidated, not just its values -- SP-5c needs >=3 collinear same-kind tiles/
   axis to *falsify* the 4-knob shape (per-hop uniformity? do `d_h/d_v`
   collapse?). Can R1+R3b **as designed** actually falsify the shape, or do they
   only fit constants to an *assumed* shape? (design Sec.4.1/5.1, parent Sec.4.5).
2. **Joint identifiability of the two instruments.** R1 gives `{d_v,
   intra_contrast}`; R3b gives `d_h/d_v`. Do they jointly pin all 4 knobs with no
   hidden unobservable combination? Is the sign reconciliation between them sound,
   or does it smuggle an assumption? (gauge freedom: design Sec.4.2.)
3. **The Delta_wall cross-domain assumption.** SP-4b left R1's cross-domain
   `Delta_wall` *assumed* (contingent on fidelity-gap row 51, not SP-4a-proven).
   Does it contaminate R1's within-column recovery, or is it genuinely orthogonal?
4. **Is the instrument-role flip actually correct?** R3b was made primary because
   ~30cy jitter defeats R1's `d_h`. Is R3b genuinely *immune* to that jitter, or
   does the same physics hit its perf-counter interval somewhere we have not traced?

**Recommended Fable session shape:** (i) run this soundness audit against the two
merged design docs + `docs/trace/cross-domain-skew-limit.md`; (ii) *then* author
the R3b Phase-2 execution plan (writing-plans) incorporating whatever the audit
forces; (iii) hand kernel/host/gate implementation to Sonnet subagents under that
plan. The audit is the deliverable that justifies a genius; the plumbing does not.

---

## E. Pointers and guardrails

**Governing docs:** design Sec.5/6 (above); parent apparatus spec; skew-limit doc
(`docs/trace/cross-domain-skew-limit.md`, routes + epistemic boundary).

**Live memory:** `memory/project_timer_sync_arc_inflight.md` (the whole #140 arc),
`memory/project_sp5b_r1_gate_iommu_fault_inflight.md` (the page-fault blocker),
`memory/project_framework_arc_inflight.md` (what unparks when #140 closes).

**Key files:** software core `tools/calibration/skew/*`; seam
`src/device/state/effects.rs`; in-process runner `src/testing/xclbin_suite.rs`;
decode `tools/trace_decoder/` + `tools/parse-trace.py`; gate template
`build/experiments/sp3-spike-trace/task3_gate.sh`; kernel authoring refs
`mlir-aie/test/npu-xrt/vec_mul_event_trace/aie.mlir` (Shape A core+mem trace) and
`AIEInsertTraceFlows.cpp:672-723`.

**Run the merged core:** `python3 -m pytest tools/test_skew_*.py -q` (21 pass);
`cargo test --lib` (3583 pass, seam unset = byte-identical).

**HW-safety (non-negotiable, per project memory + CLAUDE.md):** HW is the cheap
oracle -- err toward more HW runs. Never run two HW suites concurrently. **No
`xrt-smi` during a HW run** (segfaults this devbox). `pkexec` not `sudo`. TDR
recovery: `pkexec sh -c 'modprobe -r amdxdna && modprobe amdxdna'`. Reboot is
handed to the user, never self-run. Rebuild the FFI `.so`
(`cargo build -p xdna-emu-ffi [--release]`) before any plugin/gate use -- a stale
`.so` causes phantom bugs. Never pipe build/test through `tail`/`grep`.

**SP-5b -> SP-5c handoff (design Sec.11):** SP-5b ships apparatus that runs +
reproduces (range-0), producing **no number and no evidence any number is
correct**. Correctness is entirely SP-5c's human causal-vs-HW gate. The
`calibrated` flip + 3 regression-guard updates live in SP-5c
(`crates/xdna-archspec/src/runtime.rs:807`,
`src/interpreter/engine/coordinator.rs:4078`, `src/device/state/effects.rs:1355`).
