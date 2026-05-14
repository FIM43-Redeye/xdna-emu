---
date: 2026-05-05
status: posted
issue_url: https://github.com/Xilinx/mlir-aie/issues/3047
relates_to: A.2b mode-2 plan, Phase 7
---

# Upstream issue text for mlir-aie (posted as #3047)

> **Posted 2026-05-05** as
> [mlir-aie #3047](https://github.com/Xilinx/mlir-aie/issues/3047).
> The proposed issue title and body below are kept as the archival
> record of what we sent up; any further public discussion lands
> on the live issue + in "Public discussion" sections of the
> relevant findings docs.

---
(note from user: hi I edited this I EDITED IT AGAIN fixed the typo this time)
---

## Proposed issue title

> Mode-2 tracing (execution/inst_exec) has major documentation gaps

---

## Proposed issue body

So I'm building an open-source NPU emulator for AMD XDNA (because I really want to run AI inference on this FREE AI COPROCESSOR and the ability to visualize spatial dataflow seemed useful) and I've been implementing mode-2 (execution/`inst_exec`) trace decode/encode against real Phoenix (NPU1, AIE2) hardware. In the process I've collected a few empirical findings about mode-2 that I couldn't find documented anywhere, plus a couple of suggestions that might be useful upstream. I wanted to post them here in case the project has prior knowledge that supersedes my observations, or interest in any of this work landing in `parse.py`.

This is my first upstream post on the topic, happy to break it into multiple issues if that's preferred. Pretty new to open source in general, so please pardon any faux pas.

### Setup

- **Hardware**: Phoenix (NPU1, AIE2)
- **Method**: bridge runner that loads `xclbin` via XRT, with custom `insts.bin` patching to control trace mode and event routing. Decoder is a from-scratch implementation with bit-level layout derived by squinting extremely hard at aietools. I've cross-validated decoder output against `aiesimulator`'s VCD where comparable.
- **Reference kernels**: small mlir-aie examples (e.g. `add_one_using_dma`) plus my own controlled fixtures for ZOL trip-count sweeps.

I haven't been able to find documentation for mode-2 frame semantics in:

- mlir-aie `python/utils/trace/parse.py` and `event_ir.py` (mode 0 / mode 1 only)
- aie-rt `xaie_trace.h` (declares `XAie_TraceMode` with three values, no frame structure)
- aietools anywhere, no semantic documentation, relevant symbols, or anything else
- llvm-aie ZOL codegen (no trace markers)
- AM020 chapter 2 (states only that the trace records "the minimum set of information to allow an offline debugger to reconstruct the program execution flow", lists three categories, gives no frame-level detail)

If I missed an existing source, please point at it and I'll close this issue.

### Findings

#### 1. LC frame bit-28 ("flag") is empirically inert

The LC frame layout has a 1-bit field at position 28 of its 32-bit payload. I initially modeled it as "flag = 1 iff `lc_after == 0`" (i.e., set on the iteration where the loop counter exhausts) but couldn't reproduce flag=1 in any controlled fixture.

Captured 9 ZOL traces with trip counts N ∈ {1, 2, 4, 8, 16, 64, 256, 1024, 16384}, each running 64 outer-wrapper passes:

| N      | LC frames | flag=1 count | count field |
|-------:|----------:|-------------:|------------:|
|      1 |        64 |            0 | =N          |
|      2 |        64 |            0 | =N          |
|      4 |        64 |            0 | =N          |
|      8 |        64 |            0 | =N          |
|     16 |        64 |            0 | =N          |
|     64 |        64 |            0 | =N          |
|    256 |        64 |            0 | =N          |
|   1024 |        64 |            0 | =N          |
|  16384 |        64 |            0 | =N          |

In every capture: exactly 64 LC frames (one per outer-wrapper pass = one per ZOL invocation), `count = N` (the value loaded into LC at loop start, **not** the running register), `flag = 0` always.

Conclusions I landed on:

- LC frames are emitted **once per ZOL invocation, not per iteration**.
- The 28-bit count field stores the **initial** trip count, not any running-decrement value.
- I could not reproduce flag=1. The leading hypotheses for what would trigger it (nested-loop save/restore, branch-out-of-loop) didn't fire in my fixtures.

Question: is flag=1 even a thing on AIE2? If yes, what triggers it?

#### 2. Atoms fire at branch resolution, not per cycle

AM020 §2 (chapter 2, lines 299-305) says:

> The trace unit in the AIE-ML can operate in execution-trace mode. In real time, the unit will send, via the AXI4-Stream, a minimum set of information to allow an offline debugger to reconstruct the program execution flow. ... The information includes:
> - Conditional and unconditional direct branches
> - All indirect branches
> - Zero-overhead-loop LC

Reading this as "ARM-ETM-style branch trace": atoms (E / N) fire at conditional-branch retire, never per cycle.

This matches the captured stream. On `add_one_using_dma.chess` (loop body of 4 iters), HW emits exactly 4 E_atoms + 2 N_atoms across the whole trace, corresponding to JNZD outcomes instead of the hundreds of atoms a per-cycle model would produce.

A first-pass implementation that emits atoms per cycle (as I initially did) inflates atom counts by ~100×. Rule that matches HW:

- **E_atom**: condition fired, conditional branch taken.
- **N_atom**: condition false, fall-through.
- Unconditional branches and RETs emit no atoms.

Question: is there a public reference for this beyond the AM020 sentence I quoted? I'm inferring from "ARM-ETM-style" behavior; explicit confirmation in mlir-aie or aie-rt would be helpful for downstream tools. Documentation-fishing is especially unpleasant for all the new folks who want to take off running with their friendly neighborhood NPU.

#### 3. New_PC: only at dynamic-destination branches

Direct unconditional branches (immediate target encoded in the instruction) do **not** emit New_PC; the offline debugger reads the target from the ELF.

New_PC fires only when the destination cannot be deduced statically:

- **RET** (`ret`): New_PC = caller's post-delay-slot resume PC.
- **Indirect taken branches** (`JNZD r, r, p7`, indirect `J`/`JL` via register).

This dominates the trace structure: every JL+RET pair contributes one New_PC, on the RET side.

Question: same as above, is this layout documented anywhere upstream? It's consistent with how mlir-aie's `event_ir.py` mode-1 output works (PC stamps, but only at externally-observable boundaries) so I suspect it's already folklore but undocumented.

#### 4. Conditional-indirect: atom only, no New_PC

`JNZD r, r, p7` is conditional-indirect: destination is in `p7`, condition is in the JNZD itself. Per AM020's "all indirect branches" line I expected New_PC on the taken side, but HW skips it and only the atom fires. I've handled this in my trace_unit, it's a small but consistent observation across every JNZD-driven loop in my test corpus.

Question: is this an intentional optimization (conditional result + LC structure together let the debugger reconstruct the target without an explicit New_PC) or a HW quirk?

#### 5. Anchor_PC reflects fetch PC, not execute PC

Mode-2's Start frame includes a 14-bit `anchor_pc`. I initially modeled it as the execute-stage PC at the moment start_event fires. Comparing to HW on `add_one_using_dma`:

- HW anchor_pc = 832 (inside `llvm___aie2___acquire`'s NOP pad -- the lock-acquire stall region).
- EMU anchor_pc (after a first round of fixes) = 816 -- same region but 16 bytes earlier.

The 16-byte offset is exactly the AIE2 fetch+decode pipeline depth during a lock-acquire stall: HW's fetch has already pulled in the next bundle (often the function epilog -- RET, delay slots) before the pipeline buffer fills. The `Core_PC` register and trace events reflect that fetch PC, not the execute PC.

Question: is the fetch-vs-execute PC distinction documented for `Core_PC` reads? The implication is that any `xrt_smi`-style register inspection during a stall returns the fetch PC, which is potentially surprising for downstream tooling.

### Existing issues I cross-checked

I searched the issues tracker before posting and found these:

- **#2001** / **PR #2058** ("Trace not generated when building some designs with peano") -- I initially flagged the same symptom on `add_one_using_dma` and similar tests, but on re-investigation my pipeline already emits the post-#2058 register sequence (packet-routed Trace→shim DMA flow plus a `gen_trace_done_aie2`-equivalent event-fire at the end of the runtime sequence -- both via mlir-aie's declarative trace API + `AIEXInlineTraceConfig` lowering). The empirical "Peano vs Chess" differential I'd recorded was an artifact of cross-batch BD-reuse contamination in my sweep harness, not Peano-specific HW behavior. So there's nothing here I need from the project; flagging for completeness.
- **#3013** (open) "Trace infrastructure improvement ideas" has adjacent items (timer sync across multi-shim destinations) but doesn't cover the mode-2 frame semantics I describe above.
- **#3012** (Python binding enums for trace events) and **#2356** (trace overrun check) are orthogonal.

If any of findings 1-5 are already tracked and I missed it, please point me at the existing issue and I'll close this one.

### Specific asks

1. Does anyone have prior knowledge of what bit 28 of the LC frame means, or fixtures that reach flag=1?
2. Is there a public reference for AIE2 mode-2 frame semantics beyond AM020 §2?
3. Is mode-2 trace decoding planned for mlir-aie, and what would be helpful for it if so?

Happy to share fixtures, decoder source, and bridge-runner configs if useful. Cheers!

---

## Internal notes (NOT part of the issue body)

### Why post this

A.2b plan Phase 7 (the last open phase). The community
benefits from knowing about findings 1-5; the project
benefits from feedback that might invalidate or extend our
model before we cement it in code.

### Findings sourced from

- LC bit-28 flag: `docs/superpowers/findings/2026-04-30-mode2-lc-flag-semantics.md`
- Atom + New_PC semantics: `docs/archive/findings/2026-05-01-mode2-atom-and-new-pc-semantics.md`
- JNZD conditional-indirect: xdna-emu commit `0eecfbf`
- Fetch-PC pipeline depth: xdna-emu commit `c0abfd0`
- Cross-checked existing issues #2001 / #2058 / #3013 / #3012 / #2356 / #3000
  (all surveyed 2026-05-05 -- see `docs/coverage/peano-trace-window-gap.md`
  for the #2001 disposition)

### After posting

1. User posts to https://github.com/Xilinx/mlir-aie/issues
2. Update this file's frontmatter `status:` to "posted" and add
   the issue URL.
3. Append a "Public discussion" section to:
   - `docs/superpowers/findings/2026-04-30-mode2-lc-flag-semantics.md`
   - `docs/archive/findings/2026-05-01-mode2-atom-and-new-pc-semantics.md`
   - `docs/coverage/peano-trace-window-gap.md`
4. Mark task #324 completed.
